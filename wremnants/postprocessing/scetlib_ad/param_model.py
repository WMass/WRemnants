"""rabbit ParamModel for a fully differentiable SCETlib prediction.

The prediction is assembled in three steps::

    1  SCETlib cached rule replay  ->  sigma(p; g)       boson level, gen grid
    2  fold through the response R ->  sigma_reco(p; b)  gen -> reco
    3  ratio to the anchor         ->  rnorm(b, proc)    handed to rabbit

with ``sigma_reco(b) = sum_g R_raw(b, g) / N_gen(g) * sigma(p; g)`` and
``rnorm = sigma_reco(p) / sigma_reco(p_anchor)`` on the signal process (in
gen-level mode the ratio is taken on ``sigma_gen`` directly). rabbit multiplies
the card's templates by ``rnorm``.

``p`` is SCETlib's own differentiation vector. Which parameters it holds is a
property of the cache (``gradient_param_names()``), and can include:

* ``alphaS``; with the PDF set's alpha_s member pair in the cache it is the
  PDF-consistent coupling rather than alpha_s at fixed PDF;
* the nonperturbative lambdas of the Collins-Soper and TMD form factors;
* the theory nuisance parameters (``gamma_cusp``, ``gamma_mu_q``, ``gamma_nu``,
  ``s``, ``h_qqV`` and the five beam-function TNPs), when the runcard declares a
  ``[TNPs]`` block;
* PDF eigenvector coefficients, exact at ``c_e = 0, +-1``;
* the profile scales ``kappa_R``, ``kappa_F`` and the transition points
  ``x1..x3`` (registered by ``set_diff_scales(1)``).

By default (:data:`params.DEFAULT_FROZEN`) the NP shape constants,
``resumTNP_b_qqDS``, ``resumTransition1/3`` and ``kappa_R`` / ``kappa_F`` are
held at the anchor; the central transition point ``resumTransition2`` is
profiled. The mu_R / mu_F uncertainty is carried instead by two frozen gen-level
envelope nuisances (:mod:`scale_envelope`): a profiled ``kappa_F`` has a
response that is even in its nuisance, so a symmetric prior has two minima.

Derivatives: ``ScetlibCachedXsecTF`` is TF-differentiable (its backward pass is
a ``custom_gradient`` contracting Hessian-vector products), so nested
``GradientTape``s work and the fit Hessian is exact. Map rabbit's fit vector
into SCETlib's layout with a constant 0/1 matmul, never
``tensor_scatter_nd_update``: a scatter's gradient is a ``tf.IndexedSlices``,
on which the bridge's second-order payloads (``.numpy()``) fail.

XLA cannot compile a ``PyFunc``, so the fit must run with ``--jitCompile off``
(checked at construction).

Blinding: rabbit's blinding is a change of variables inside the likelihood, so
``compute()`` receives the true physical parameter values. Nothing in
``compute``, ``_ratio_from_param``, ``_sigma_gen``, ``_physical``,
``_physical_tf`` or ``_full_vector`` may print, log or format a parameter value,
including in exception messages; use ``tf.Assert`` with a constant string, not
``tf.debugging.assert_*`` (which prints the tensor). All prints in this file
happen at construction time, before any blinding offset is armed.
"""

import configparser
import copy
import re

import numpy as np
import tensorflow as tf

from rabbit.param_models.param_model import ParamModel
from wremnants.postprocessing.scetlib_ad import params as adp
from wremnants.postprocessing.scetlib_ad import response as response_mod
from wremnants.postprocessing.scetlib_ad.response import (
    DEFAULT_RESPONSE_GROUP,
    RATIO_FLOOR_MIN,
    RATIO_FLOOR_SCALE,
    R_info_from_auxiliary,
    corr_config_from_meta,
    crop_R_to_fit,
    marginalize_R_reco,
)
from wremnants.postprocessing.scetlib_ad.scale_envelope import (
    KAPPA_F,
    FrozenEnvelope,
    scale_spec,
)
from wremnants.postprocessing.scetlib_ad.xsec_backend import (
    ScetlibADXsec,
    config_as_dict,
)

DTYPE = tf.float64

# Regex (matched case-insensitively against indata.systs) -> the registered
# parameter that makes a card syst a double count. Only entries whose model
# parameter is fitted are enforced. A regex, so that the PDF eigenvector
# templates (pdf<N><SET>...) are told apart from pdfAlphaS.
_CONFLICTS = (
    (
        r"scetlibnp",
        "any NP lambda",
        lambda names: any(n.startswith("lambda") for n in names),
    ),
    (r"^pdfalphas", "alphaS", lambda names: "alphaS" in names),
    (
        r"^resumtnp",
        "a resummation TNP",
        lambda names: any(n.startswith(adp.TNP_PREFIX_OUT) for n in names),
    ),
    (
        r"^pdf\d+",
        "a PDF eigenvector coefficient",
        lambda names: any(n.startswith(adp.PDF_PREFIX_OUT) for n in names),
    ),
    (
        r"^resumfoscale",
        "the muR / muF scale uncertainty",
        # Both the profiled kappa_R / kappa_F and the frozen envelope
        # (scale_envelope.py) are the physics of the card's resumFOScale*.
        lambda names: any(
            n in ("resumScaleMuR", "resumScaleMuF") + adp.SCALE_ENVELOPE_NAMES
            for n in names
        ),
    ),
    (
        r"^resumtransition",
        "a matching transition point",
        lambda names: any(n.startswith("resumTransition") for n in names),
    ),
)


def _as_name_tuple(value):
    """Spec tokens arrive as strings; accept ``a,b`` as well as a real tuple."""
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(s.strip() for s in value.split(",") if s.strip())
    return tuple(value)


class SCETlibADParamModel(ParamModel):
    """Fit SCETlib's own differentiable parameters directly.

    Usage::

        --paramModel wremnants.postprocessing.scetlib_ad.SCETlibADParamModel \\
            cache=<cache>.npz conf=<runcard>.conf gen_level=1 [key=value ...]
    """

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        """``key=value`` spec tokens, typed off the ``__init__`` default.

        A later duplicate key wins, so overrides can be appended to a spec.
        """
        import inspect

        sig = inspect.signature(cls.__init__)
        valid = {
            n: p
            for n, p in sig.parameters.items()
            if n not in ("self", "indata")
            and p.kind is not inspect.Parameter.VAR_KEYWORD
        }
        positional = []
        for tok in args:
            key = tok.split("=", 1)[0] if isinstance(tok, str) and "=" in tok else None
            if key is not None and key not in valid:
                # Every argument is keyword-with-default, so an unknown key is a
                # typo; never fall through to positional.
                raise TypeError(
                    f"{cls.__name__}: unknown spec token {key!r}. Valid tokens: "
                    + ", ".join(sorted(valid))
                )
            if key is not None:
                val = tok.split("=", 1)[1]
                default = valid[key].default
                if isinstance(default, bool):
                    val = str(val).strip().lower() in ("1", "true", "yes", "on")
                elif isinstance(default, float):
                    val = float(val)
                elif isinstance(default, int):
                    val = int(val)
                kwargs[key] = val
            else:
                positional.append(tok)
        return cls(indata, *positional, **kwargs)

    def __init__(
        self,
        indata,
        cache=None,
        conf=None,
        gen_level=False,
        signal_proc="Zmumu",
        Q_lo=60.0,
        Q_hi=120.0,
        fit_params=None,
        poi_params="alphaS",
        threads=0,
        priors=True,
        prior_sigmas=None,
        xparam_default=None,
        pdf_coeff_scale=None,
        anchor_source="correction",
        anchor_override=None,
        response_group=DEFAULT_RESPONSE_GROUP,
        scale_envelope=True,
        scale_envelope_qt_min=20.0,
        scale_envelope_response="linear",
        scale_envelope_points=3,
        fo_muf_poly=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        cache, conf
            The ``.npz`` written by ``ScetlibCachedXsecTF.save`` and the SCETlib
            runcard it was built from. Both are required.
        gen_level
            Gen-level mode: the fit channel is the gen (qT, |Y|) binning and
            ``compute()`` returns ``sigma_gen(p) / sigma_gen(p_anchor)`` per gen
            bin, with no fold. Otherwise R is read from the datacard's response
            auxiliary.
        signal_proc
            Process whose column carries the ratio; the others stay at 1.
        Q_lo, Q_hi
            The mass window the cache's single Q bin must span.
        fit_params
            Comma-separated rabbit-facing names to fit. Default (and ``all``):
            every parameter the cache carries except ``params.DEFAULT_FROZEN``.
            Parameters not fitted are held at their anchor.
        poi_params
            Subset of ``fit_params`` reported as POIs. Default ``alphaS``.
        threads
            SCETlib worker threads (0 = one per hardware thread).
        priors
            Declare Gaussian priors (default on): ``params.FREE_PARAMS`` are
            free, every other parameter gets sigma = 1 in rabbit's units.
            Fitting TNPs with priors off is refused.
        prior_sigmas
            Per-name override, ``name=value,...`` or a Mapping, in rabbit's
            units. ``nan`` frees a parameter.
        xparam_default
            ``name=value,...`` shifting the fit start (and the prior mean) off
            the anchor, for injection / closure tests; reparametrised names are
            in theta units. The ratio denominator always stays the anchor.
        pdf_coeff_scale
            Confidence-level convention for the PDF eigenvector coefficients:
            the fitted ``pdfEig{i}`` is a unit nuisance theta and SCETlib is
            evaluated at ``c_e = pdf_coeff_scale * theta``. Default: resolved
            from ``theory_utils.pdfMap`` for the runcard's ``pdf_set`` and the
            card's ``noi``, the same factor ``add_pdf_uncertainty`` applies to
            the templates. 1 switches it off; a float overrides.
        anchor_source
            Where the parameter central values come from. ``"correction"``
            (default): the theory correction the card's templates were
            reweighted with, as recorded in the card metadata; a card that
            records none is refused. ``"cache"``: the cache's own build point.
            The ratio is 1 at the start either way, so a wrong anchor is
            invisible prefit; ``"cache"`` is logged loudly.
        anchor_override
            ``name=value,...`` supplying a central value the correction does not
            record (e.g. an NP coefficient compiled into SCETlib with no runtime
            key). Printed at construction.
        response_group
            The datacard auxiliary group holding R and N_gen.
        scale_envelope
            Carry the mu_R / mu_F uncertainty as a frozen, symmetrized envelope
            (``scale_envelope.py``) instead of profiling ``kappa_R`` /
            ``kappa_F``. Default on; naming the two scales in ``fit_params``
            while it is on is refused as a double count.
        scale_envelope_qt_min
            Zero the envelope in gen bins whose LOWER qT edge is below this, in
            GeV (default 20; ``nan`` applies it everywhere). Whether the
            threshold is a gen-bin edge is reported at construction.
        scale_envelope_response
            ``"linear"`` (default): ``1 + theta . k``, exactly odd in theta;
            ``"lognormal"``: ``exp(theta . k)``, rabbit's card convention.
        scale_envelope_points
            ``3`` (default): mu_R alone, the point set the datacard's
            ``resumFOScale`` templates are built from
            (``theory_corrections.renorm_scale_vars``). ``7``: the (mu_R, mu_F)
            envelope (``theory_corrections.renorm_fact_scale_vars``).
        fo_muf_poly
            Degree of the fixed-order muF polynomial; ``None`` (default) keeps
            SCETlib's default. ``0`` skips building it, which makes the first
            evaluation much faster but removes the kappa_F derivative;
            ``_check_fo_muf_poly`` refuses the configurations that need it.
        """
        self.indata = indata
        if cache is None or conf is None:
            raise ValueError(
                "SCETlibADParamModel needs both cache=<cache>.npz and "
                "conf=<runcard>.conf spec tokens."
            )
        self._require_no_xla(kwargs)

        self._response_group = str(response_group)
        self.gen_level = bool(gen_level)

        # ---- Backend: rebuild the calculation and load the cache.
        self.core = ScetlibADXsec(conf, cache, threads=threads, fo_muf_poly=fo_muf_poly)
        self.scetlib_names = list(self.core.param_names)
        self.rabbit_names = [adp.rabbit_name(n) for n in self.scetlib_names]
        # ---- The anchor: the point the prediction is a ratio to, taken from
        # the theory correction the card's templates carry (not the cache).
        self._anchor = self._resolve_anchor(anchor_source, anchor_override)

        # ---- Gen binning, and (reco path) the response matrix.
        self._setup_binning(indata, Q_lo, Q_hi)

        # ---- PDF confidence-level convention (see params.pdf_coeff_scale).
        self._conf_path = conf
        self.pdf_coeff_scale = self._resolve_pdf_coeff_scale(pdf_coeff_scale)

        # ---- Scale envelope: resolved before registration (its nuisances are
        # fitted parameters), evaluated after (it needs the anchor vector).
        self._env_spec = None
        if scale_envelope:
            qt_min = float(scale_envelope_qt_min)
            self._env_spec = scale_spec(
                int(scale_envelope_points),
                qt_min=None if np.isnan(qt_min) else qt_min,
                name=adp.SCALE_ENVELOPE_STEM,
                response=str(scale_envelope_response),
            )

        # ---- Parameter registration. Everything the fit does NOT expose stays
        # pinned at the anchor, so the SCETlib vector is always complete.
        self._register_params(fit_params, poi_params, xparam_default)

        # ---- Central: the ratio denominator, evaluated at the anchor so the
        # ratio is exactly 1 at the start.
        sigma_gen_anchor = self._sigma_gen_np(self._p_base_anchor)
        self.sigma_gen_central_flat = tf.constant(sigma_gen_anchor, dtype=DTYPE)
        self._build_scale_envelope(sigma_gen_anchor)
        if self.gen_level:
            self.sigma_reco_central = None
        else:
            self.sigma_reco_central = tf.linalg.matvec(
                self.R, self.sigma_gen_central_flat
            )
            n_bad = int(tf.reduce_sum(tf.cast(self.sigma_reco_central <= 0, tf.int32)))
            if n_bad:
                raise ValueError(
                    f"SCETlibADParamModel: {n_bad} reco bins have non-positive "
                    f"sigma_reco at the anchor. Likely a binning mismatch between "
                    f"R and the fit-tensor reco axes."
                )

        self._check_double_counting()
        self._check_no_inert_params()
        self._check_fo_muf_poly()

        # ---- Process column.
        procs = [p.decode() if isinstance(p, bytes) else str(p) for p in indata.procs]
        if signal_proc not in procs:
            raise ValueError(
                f"SCETlibADParamModel: signal_proc={signal_proc!r} not in "
                f"indata.procs={procs[:10]}..."
            )
        self.signal_proc_idx = procs.index(signal_proc)
        self.nproc = indata.nproc
        self._signal_col_mask = tf.reshape(
            tf.one_hot(self.signal_proc_idx, self.nproc, dtype=indata.dtype),
            [1, self.nproc],
        )

        self._setup_priors(priors, prior_sigmas)

        print(
            f"[SCETlibADParamModel] {self.core} | {self._fold.describe()} | "
            f"{'gen-level' if self.gen_level else 'reco'} | "
            f"fitting {self._n_scetlib} of {self.core.n_params} SCETlib "
            f"parameter(s)"
            + (
                f" + {len(self._env_names)} envelope nuisance(s) "
                f"{list(self._env_names)}"
                if self._env_names
                else ""
            )
            + f" = {self.nparams} total "
            f"({self.npoi} POI: {[n for n in self._param_order[: self.npoi]]})",
            flush=True,
        )

    # =========================================================================
    # construction helpers
    # =========================================================================

    def _require_no_xla(self, kwargs):
        """Refuse to build under XLA -- a PyFunc has no XLA lowering.

        rabbit resolves ``--jitCompile auto`` to True in dense mode, and the
        failure would otherwise be an opaque compile error in the first loss
        evaluation.
        """
        opt = str(kwargs.get("jitCompile", "auto")).lower()
        sparse = bool(getattr(self.indata, "sparse", False))
        # --eager turns every tf.function into eager execution, so jit_compile
        # never applies and a PyFunc is fine.
        if kwargs.get("eager") or opt == "off" or (opt == "auto" and sparse):
            return
        raise ValueError(
            "SCETlibADParamModel calls into SCETlib through tf.py_function, "
            "which XLA cannot compile. Re-run with --jitCompile off "
            f"(got --jitCompile {opt}" + (", dense input)." if not sparse else ").")
        )

    def _setup_binning(self, indata, Q_lo, Q_hi):
        """Resolve the gen grid (and R, in the reco path) and map it onto the cache."""
        if self.gen_level:
            gen_axes = self._fit_axes(indata)
            if len(gen_axes) != 2:
                raise NotImplementedError(
                    "gen_level SCETlibADParamModel expects a single fit channel "
                    "with 2 gen axes (qT, |Y|); got "
                    f"{[n for n, _ in gen_axes]}"
                )
            self.R = None
            self.reco_shape = None
        else:
            R_info = R_info_from_auxiliary(indata, self._response_group)
            fit_reco_axes = self._fit_axes(indata)
            R_full, R_reco_axes = marginalize_R_reco(
                R_info["R"], R_info["reco_axes"], [n for n, _ in fit_reco_axes]
            )
            R_arr = crop_R_to_fit(R_full, R_reco_axes, fit_reco_axes)
            self.reco_shape = R_arr.shape[: len(fit_reco_axes)]
            gen_axes = R_info["gen_axes"]
            if R_info.get("N_gen") is None:
                raise ValueError(
                    f"SCETlibADParamModel: the {self._response_group!r} "
                    "auxiliary has no N_gen "
                    "(gen-total). Rebuild the datacard from a histmaker output "
                    "that carries the 'prefsr' xnorm hist."
                )
            n_reco = int(np.prod(self.reco_shape))
            n_gen = int(np.prod([len(e) - 1 for _, e in gen_axes]))
            R_raw = tf.constant(R_arr.reshape(n_reco, n_gen), dtype=DTYPE)
            n_gen_flat = tf.constant(
                np.asarray(R_info["N_gen"]).reshape(-1), dtype=DTYPE
            )
            # P(b|g) = R_raw / N_gen: normalise each gen column by the gen total
            # so R carries no MC gen spectrum (empty gen bins keep a zero column).
            safe = tf.where(n_gen_flat > 0, n_gen_flat, tf.ones_like(n_gen_flat))
            self.R = R_raw / safe[tf.newaxis, :]

        self.gen_axes = [
            (name, np.asarray(edges, dtype=np.float64)) for name, edges in gen_axes
        ]
        self.gen_shape = tuple(len(e) - 1 for _, e in self.gen_axes)
        self.Q_lo, self.Q_hi = float(Q_lo), float(Q_hi)

        # compute() returns one row per fit bin, so its shape must be the card's.
        n_rows = int(np.prod(self.gen_shape if self.gen_level else self.reco_shape))
        if n_rows != int(indata.nbins):
            raise ValueError(
                f"SCETlibADParamModel: the model produces {n_rows} bins "
                f"({'gen' if self.gen_level else 'reco'} shape "
                f"{self.gen_shape if self.gen_level else self.reco_shape}) but the "
                f"card has {int(indata.nbins)}."
            )
        # Exact sum of cache bins onto the gen grid (any nesting order, signed Y
        # folded onto |Y|, finer cache bins); raises if the cache does not tile it.
        self._fold = self.core.fold_for(self.gen_axes, self.Q_lo, self.Q_hi)

    def _fit_axes(self, indata):
        """(name, edges) of each axis of the single non-masked channel."""
        non_masked = [
            (name, info)
            for name, info in indata.channel_info.items()
            if not info.get("masked", False)
        ]
        if len(non_masked) != 1:
            raise NotImplementedError(
                f"SCETlibADParamModel supports a single non-masked channel; got "
                f"{len(non_masked)}: {[n for n, _ in non_masked]}"
            )
        _, info = non_masked[0]
        return [
            (ax.name, np.asarray(ax.edges, dtype=np.float64)) for ax in info["axes"]
        ]

    def _resolve_pdf_coeff_scale(self, override):
        """The 90%->68% (and per-set inflation) factor for ``pdfEig{i}``.

        Read from ``theory_utils.pdfMap`` for the runcard's LHAPDF set and the
        card's nuisance of interest, as ``add_pdf_uncertainty`` does.
        """
        if not any(n.startswith(adp.PDF_PREFIX_OUT) for n in self.rabbit_names):
            return 1.0
        if override is not None:
            scale = float(override)
            print(
                f"[SCETlibADParamModel] pdf_coeff_scale = {scale:.6g} (explicit "
                f"override; the map's own value was not used)",
                flush=True,
            )
            return scale

        cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
        cfg.read(self._conf_path)
        pdf_set = None
        for sec in cfg.sections():
            if cfg.has_option(sec, "pdf_set"):
                pdf_set = cfg.get(sec, "pdf_set").strip()
                break
        if pdf_set is None:
            raise ValueError(
                f"SCETlibADParamModel: the runcard {self._conf_path} declares no "
                f"pdf_set, so the confidence-level convention of the "
                f"{sum(1 for n in self.rabbit_names if n.startswith(adp.PDF_PREFIX_OUT))} "
                f"eigenvector coefficients is unknown. Pass pdf_coeff_scale."
            )
        args = (self.indata.metadata or {}).get("meta_info", {}).get("args", {})
        noi = args.get("noi") or list(adp.PDF_COEFF_SCALE_NOI_DEFAULT)
        scale = adp.pdf_coeff_scale(pdf_set, noi)
        print(
            f"[SCETlibADParamModel] pdf_coeff_scale = {scale:.6g} "
            f"({pdf_set}, noi={list(noi)}): pdfEig{{i}} = +-1 is 1 sigma, "
            f"evaluated at c_e = {scale:.5f}",
            flush=True,
        )
        return scale

    def _register_params(self, fit_params, poi_params, xparam_default):
        """Decide which SCETlib parameters rabbit sees, and their start values."""
        available = list(self.rabbit_names)
        # Default: every registered parameter except params.DEFAULT_FROZEN,
        # TNPs included (each carries an N(0,1) prior). 'all' is the same set.
        if _as_name_tuple(fit_params) == ("all",):
            requested = tuple(n for n in available if n not in adp.DEFAULT_FROZEN)
        else:
            requested = _as_name_tuple(fit_params) or tuple(
                n for n in available if n not in adp.DEFAULT_FROZEN
            )
        unknown = [n for n in requested if n not in available]
        if unknown:
            raise ValueError(
                f"SCETlibADParamModel: fit_params {unknown} are not in this "
                f"cache's parameter set {available}. A parameter can only be "
                f"fitted if the runcard declared it before the rules were built."
            )
        # Naming the scales in fit_params overrides DEFAULT_FROZEN; together
        # with the envelope that would count the same uncertainty twice.
        if self._env_spec is not None:
            both = [n for n in requested if n in ("resumScaleMuR", "resumScaleMuF")]
            if both:
                raise ValueError(
                    f"SCETlibADParamModel: fit_params names {both} while the "
                    f"7-point scale envelope is on. Both describe the mu_R / "
                    f"mu_F missing-higher-order uncertainty, so running them "
                    f"together double-counts it. Pass scale_envelope=0 to "
                    f"profile the scales directly (which is the treatment the "
                    f"envelope replaced -- see scale_envelope.py), or drop them "
                    f"from fit_params."
                )
        pois = _as_name_tuple(poi_params)
        bad_pois = [n for n in pois if n not in requested]
        if bad_pois:
            raise ValueError(
                f"SCETlibADParamModel: poi_params {bad_pois} are not in "
                f"fit_params {list(requested)}."
            )
        # rabbit's layout: POIs first, then POUs. The SCETlib parameters form a
        # contiguous prefix (``param[:self._n_scetlib]``), the envelope
        # nuisances the tail; the latter are not SCETlib parameters and must
        # not enter _fit_idx / _select / _rp_*.
        nou = tuple(n for n in requested if n not in pois)
        self._scetlib_order = tuple(pois) + nou
        self._n_scetlib = len(self._scetlib_order)
        env_names = () if self._env_spec is None else self._env_spec.param_names
        self._env_names = tuple(env_names)
        self._param_order = self._scetlib_order + self._env_names
        self.npoi = len(pois)
        self.npou = len(nou) + len(self._env_names)
        self.params = np.array([p.encode() for p in self._param_order])

        # Position of each fitted parameter inside SCETlib's own vector.
        self._fit_idx = np.array(
            [available.index(n) for n in self._scetlib_order], dtype=np.int64
        )
        # rabbit -> SCETlib map as a constant 0/1 matrix, never a scatter: its
        # IndexedSlices gradient breaks the bridge at second order.
        self._select = np.zeros((len(available), self._n_scetlib))
        self._select[self._fit_idx, np.arange(self._n_scetlib)] = 1.0

        # Reparametrisation (params.REPARAM): the fitted parameter is a unit
        # nuisance theta and SCETlib gets a physical value f(theta). Stored as
        # coefficient vectors so one elementwise expression covers every one.
        n_fit = self._n_scetlib
        self._rp_log = np.zeros(n_fit, dtype=bool)
        self._rp_quad = np.zeros(n_fit, dtype=bool)
        self._rp_L = np.zeros(n_fit)
        self._rp_c = np.zeros((3, n_fit))
        for i, name in enumerate(self._scetlib_order):
            spec = adp.reparam(name)
            if spec is None:
                continue
            kind, coeffs = spec
            if kind == "log":
                self._rp_log[i] = True
                (self._rp_L[i],) = coeffs
            elif kind == "quad":
                self._rp_quad[i] = True
                self._rp_c[:, i] = coeffs
            elif kind == "unit":
                # value = <anchor> + width * theta, so theta = 0 is the anchor
                # exactly; stored as the quad map (c0, c1, 0).
                (width,) = coeffs
                self._rp_quad[i] = True
                self._rp_c[:, i] = (
                    float(self._anchor[self._fit_idx[i]]),
                    float(width),
                    0.0,
                )
            else:
                raise ValueError(f"params.REPARAM: unknown kind {kind!r}")
        self._rp_id = ~(self._rp_log | self._rp_quad)
        # Linear scale on the identity branch: 1 except for the PDF eigenvector
        # coefficients, where it makes theta = +-1 mean 1 sigma (c = scale*theta).
        self._rp_scale = np.ones(n_fit)
        if self.pdf_coeff_scale != 1.0:
            for i, name in enumerate(self._scetlib_order):
                if name.startswith(adp.PDF_PREFIX_OUT):
                    self._rp_scale[i] = self.pdf_coeff_scale
        self._reparametrised = tuple(
            n
            for n, f in zip(self._scetlib_order, ~self._rp_id | (self._rp_scale != 1.0))
            if f
        )

        # Start values: the anchor, optionally shifted for injection tests.
        # _p_base_anchor is the unshifted ratio denominator; _p_base carries the
        # shift for non-fitted slots (fitted slots come from the fit vector).
        self._p_base_anchor = self._anchor.copy()
        self._p_base = self._anchor.copy()
        defaults = self._anchor[self._fit_idx].copy()
        # A reparametrised parameter starts at theta = 0; check that maps back
        # onto the anchor, so the ratio is exactly 1 at the start.
        defaults[~self._rp_id] = 0.0
        round_trip = self._physical(defaults)
        if not np.allclose(round_trip, self._anchor[self._fit_idx], rtol=0, atol=1e-12):
            bad = [
                (n, float(a), float(b))
                for n, a, b in zip(
                    self._scetlib_order, round_trip, self._anchor[self._fit_idx]
                )
                if abs(a - b) > 1e-12
            ]
            raise ValueError(
                "scetlib_ad: the REPARAM maps do not reproduce the anchor at "
                f"theta = 0, so sigma_gen/sigma_central would not be 1: {bad}\n"
                "    NB the anchor comes from the theory correction, so a "
                "HARDCODED map offset (resumTransition2's quad c0 = 0.6) that no "
                "longer matches the correction's own value trips this. That is "
                "the intended refusal: fix REPARAM, do not move the anchor."
            )
        # Envelope nuisances: unit nuisances centred on zero (the anchor).
        defaults = np.concatenate([defaults, np.zeros(len(self._env_names))])
        for name, val in _parse_kv(xparam_default).items():
            if name not in available and name not in self._env_names:
                raise KeyError(f"xparam_default: unknown parameter {name!r}")
            if name in self._param_order:
                defaults[self._param_order.index(name)] = val
            else:
                # not fitted: pin the held value at the shifted point
                self._p_base[available.index(name)] = val
                print(
                    f"[SCETlibADParamModel] xparam_default {name}={val:g} applies "
                    f"to a NON-fitted parameter; it is pinned there, not floated.",
                    flush=True,
                )
        if xparam_default and self._reparametrised:
            print(
                "[SCETlibADParamModel] NB xparam_default for "
                f"{list(self._reparametrised)} is in THETA units (unit nuisance), "
                "not physical units.",
                flush=True,
            )
        if xparam_default:
            print(
                "[SCETlibADParamModel] start shifted: "
                f"{dict(zip(self._param_order, defaults))}",
                flush=True,
            )

        # lambdas can be legitimately zero or negative (delta_lambda2), so store
        # POIs directly rather than as sqrt(value).
        self.allowNegativeParam = True
        self.is_linear = False
        self.xparamdefault = tf.constant(defaults, dtype=self.indata.dtype)

        # alphaS is a unit nuisance (width 0.002), so rabbit's additive POI
        # blinding hides its value and leaves its uncertainty and impacts intact.

        active = set(self._param_order)
        groups = {
            label: tuple(p for p in members if p in active)
            for label, members in adp.IMPACT_GROUP_MEMBERS.items()
        }
        tnps = adp.tnp_group(self._param_order)
        if tnps:
            groups["resumTNP"] = tnps
        # Eigenvector coefficients depend on the cache, so they are grouped
        # here as ``pdfEig`` (a PDF-set-independent label).
        eigs = adp.pdf_group(self._param_order)
        if eigs:
            groups["pdfEig"] = eigs
        self.param_impact_groups = {k: v for k, v in groups.items() if v}

    def _setup_priors(self, priors, prior_sigmas):
        """Declare ``prior_sigmas`` only when asked; rabbit's Fitter keys off it."""
        tnps = adp.tnp_group(self._param_order)
        if tnps and not priors:
            # |theta| = 1 IS the recommended TNP variation; a free TNP would
            # absorb physical effects unconstrained.
            raise ValueError(
                f"SCETlibADParamModel: {len(tnps)} theory nuisance parameter(s) "
                f"are being fitted ({', '.join(tnps[:4])}"
                f"{', ...' if len(tnps) > 4 else ''}) but priors are off. TNPs "
                f"carry an N(0,1) constraint by construction. Pass priors=1 (the "
                f"registry gives every TNP sigma=1 and leaves the lambdas free), "
                f"or drop them from fit_params."
            )
        if not priors:
            return
        overrides = (
            _parse_kv(prior_sigmas)
            if not isinstance(prior_sigmas, dict)
            else dict(prior_sigmas)
        )
        unknown = [k for k in overrides if k not in self._param_order]
        if unknown:
            raise KeyError(
                f"prior_sigmas: {unknown} are not fitted parameters "
                f"({list(self._param_order)})"
            )
        sigmas = np.empty(self.nparams, dtype=np.float64)
        for i, name in enumerate(self._param_order):
            s = overrides.get(name, adp.prior_sigma(name))
            sigmas[i] = np.nan if s is None else float(s)
        self.prior_sigmas = sigmas
        constrained = {
            n: s for n, s in zip(self._param_order, sigmas) if np.isfinite(s) and s > 0
        }
        print(
            f"[SCETlibADParamModel] Gaussian priors on {len(constrained)} "
            f"parameter(s): {constrained}",
            flush=True,
        )

    def _cache_config(self):
        """The cache's runcard as ``{section: {key: str}}``.

        ``core.conf`` is the runcard layered on SCETlib's ``defaults.conf``
        (see :func:`xsec_backend.config_as_dict`).
        """
        return config_as_dict(self.core.conf)

    def _resolve_anchor(self, anchor_source, anchor_override):
        """The anchor vector over ``self.rabbit_names``, and the config check.

        The ratio multiplies the card's templates, which were reweighted by a
        theory correction, so ``p_anchor`` must be the point that correction was
        computed at. The cache's build point only affects accuracy: the NP
        lambdas and TNPs are exact anywhere, while alphaS (PDF member pair) and
        the eigenvectors (exact at c = 0, +-1) are interpolated.
        """
        cache_anchor = np.asarray(self.core.anchor, dtype=np.float64)
        source = str(anchor_source).strip().lower()
        if source not in ("correction", "cache"):
            raise ValueError(
                f"SCETlibADParamModel: anchor_source={anchor_source!r} is not "
                "understood. Use 'correction' (the theory correction the card's "
                "templates carry -- the default, and the only self-consistent "
                "choice) or 'cache'."
            )

        entry = corr_config_from_meta(getattr(self.indata, "metadata", None) or {})
        if entry is None:
            if source == "cache":
                print(
                    "[SCETlibADParamModel] WARNING: anchor_source=cache and the "
                    "card records no theory-correction config, so the central "
                    "values are the CACHE's build point and nothing cross-checks "
                    "them against the templates. The ratio is 1 at the start "
                    "either way, so a mismatch cannot be seen in any prefit "
                    "plot. Verify by hand.",
                    flush=True,
                )
                return cache_anchor
            raise ValueError(
                "SCETlibADParamModel: the card records no theory-correction "
                f"config ({response_mod.CORR_CONFIG_META_KEY}), so the parameter "
                "central values are UNKNOWN. The model predicts a ratio to the "
                "anchor which multiplies this card's templates, and those "
                "templates were reweighted by a correction whose settings are "
                "not recorded here -- so there is nothing to form the ratio "
                "about. Taking the values from the cache instead is the silent "
                "failure this refusal exists to prevent: the ratio would still "
                "be 1 at the fit start.\n"
                "    Fix: rerun the histmaker with a WRemnants that records it "
                "(histmaker_tools._add_scetlib_corr_meta).\n"
                "    Or, deliberately and on your own authority, pass "
                "anchor_source=cache."
            )

        # --theoryCorrAltOnly: the nominal templates carry no correction, so
        # there is no correction anchor for them.
        if not entry.get("applied_to_nominal", True):
            raise ValueError(
                "SCETlibADParamModel: the card's histmaker ran with "
                "--theoryCorrAltOnly, so its nominal templates carry NO theory "
                "correction. There is no correction anchor for them, and the "
                "recorded config describes a prediction the templates never "
                "saw. Rerun the histmaker with the correction applied to the "
                "nominal."
            )

        cfg = entry["config"]
        overrides = _parse_kv(anchor_override)
        unknown = [n for n in overrides if n not in self.rabbit_names]
        if unknown:
            raise KeyError(
                f"anchor_override: unknown parameter(s) {unknown}. Registered: "
                f"{self.rabbit_names}"
            )

        anchor = cache_anchor.copy()
        n_corr, structural, missing, uncovered = 0, [], [], []
        for i, name in enumerate(self.rabbit_names):
            if name in overrides:
                anchor[i] = overrides[name]
                continue
            where = adp.corr_anchor_key(name)
            if where is not None:
                value = adp.corr_anchor_value(cfg, name)
                if value is None:
                    missing.append((name, f"{where[0]}.{where[1]}"))
                else:
                    anchor[i] = value
                    n_corr += 1
                continue
            central = adp.structural_central(name)
            if central is not None:
                anchor[i] = central
                structural.append(name)
                continue
            uncovered.append(name)

        if uncovered:
            raise ValueError(
                "SCETlibADParamModel: no stated source for the central value of "
                f"{uncovered}. The correction-key -> registry map is "
                "hand-maintained, so this is what a SCETlib rename looks like. "
                "Add the parameter to params.CORR_ANCHOR_KEYS (if the runcard "
                "records it) or params.STRUCTURAL_CENTRAL (if its central value "
                "is fixed by how SCETlib registers it)."
            )
        if missing:
            raise ValueError(
                "SCETlibADParamModel: the recorded correction config does not "
                "carry a central value for:\n"
                + "\n".join(f"    {n} <- {k}" for n, k in missing)
                + "\nSo the anchor for those parameters is unknown. There is "
                "deliberately no defaults.conf fallback here: a runcard can keep "
                "a COMPILED-IN default with no runtime key -- an older tanh_6 "
                "build hardcoded the CS-side lambda_6_nu at 0.0007 while this "
                "checkout defaults it to 0 -- so a fallback would quietly hand "
                "over the wrong anchor, which is the failure this refusal "
                "exists to prevent.\n"
                "    If you know the value, state it: "
                "anchor_override=<name>=<value>. It is then recorded in the "
                "fit's spec and printed here."
            )

        # ---- Is the cache the same calculation the templates carry?
        refuse, warn, corr_only = adp.compare_corr_config(cfg, self._cache_config())
        shift = np.abs(anchor - cache_anchor)
        worst = int(np.argmax(shift)) if shift.size else 0
        print(
            f"[SCETlibADParamModel] anchor from the theory correction "
            f"{entry.get('tag')!r} ({entry.get('basename')}): {n_corr} of "
            f"{len(self.rabbit_names)} central values read from its runcard, "
            f"{len(structural)} fixed by SCETlib's registration "
            f"(resumScaleMuR/MuF at 1, pdfEig* at 0)"
            + (f", {len(overrides)} overridden {dict(overrides)}" if overrides else "")
            + f". Largest shift from the cache's own anchor: "
            f"{self.rabbit_names[worst]} {shift[worst]:.3g}. "
            f"Config cross-check: {len(refuse)} refusing, {len(warn)} warning, "
            f"{len(corr_only)} key(s) the cache runcard does not carry (SCETlib "
            f"defaults, not compared).",
            flush=True,
        )
        if corr_only:
            print(
                f"[SCETlibADParamModel] not compared (correction-only): "
                f"{corr_only}",
                flush=True,
            )
        if warn:
            print(
                "[SCETlibADParamModel] WARNING: the cache runcard disagrees with "
                "the correction on these parameter VALUES. The anchor above "
                "follows the CORRECTION, which is right -- the tape is exact "
                "away from the cache's build point, so a lambda or TNP mismatch "
                "is benign. alphaS is the exception: it is served by an "
                "interpolated PDF member pair, so a mismatch there is accepted "
                "interpolation error, not a free pass.\n"
                + "\n".join(
                    f"    {n}: correction {c!r} vs cache {h!r}" for n, c, h in warn
                ),
                flush=True,
            )
        if refuse:
            raise ValueError(
                "SCETlibADParamModel: the cache was built from a DIFFERENT "
                "calculation than the theory correction the card's templates "
                "carry, so no parameter value can reconcile them:\n"
                + "\n".join(
                    f"    {n}: correction {c!r} vs cache {h!r}" for n, c, h in refuse
                )
                + "\nRebuild the cache from the correction's runcard, or the "
                "card from a histmaker using the cache's."
            )

        if source == "cache":
            print(
                "[SCETlibADParamModel] WARNING: anchor_source=cache -- the "
                "central values above are DISCARDED and the cache's own build "
                "point is used instead, even though the card's templates were "
                "reweighted at the correction's. Nothing downstream can see the "
                "difference (the ratio is 1 at the start either way).",
                flush=True,
            )
            return cache_anchor
        return anchor

    def _sigma_gen_overrides(self, overrides):
        """sigma_gen (flat) at the anchor with PHYSICAL overrides applied.

        Used for the envelope legs, which are physical (kappa_R, kappa_F)
        values, so no reparametrisation is involved.
        """
        p = self._p_base_anchor.copy()
        for name, val in overrides.items():
            if name not in self.rabbit_names:
                raise KeyError(
                    f"scale_envelope: this cache has no parameter {name!r}; it "
                    f"carries {self.rabbit_names}. The envelope legs need the "
                    f"profile scales, which means a cache built with "
                    f"set_diff_scales(1) and the muF member pair."
                )
            p[self.rabbit_names.index(name)] = float(val)
        return self._sigma_gen_np(p)

    def _build_scale_envelope(self, sigma_gen_anchor):
        """Evaluate the frozen scale envelope, ONCE, at the anchor."""
        self._scale_env = None
        if self._env_spec is None:
            return
        spec = self._env_spec
        # The nominal leg must be the anchor (the ratio's denominator).
        for name, want in spec.legs[0].items():
            have = float(self._anchor[self.rabbit_names.index(name)])
            if abs(have - want) > 1e-12:
                raise ValueError(
                    f"SCETlibADParamModel: the scale envelope's nominal leg is "
                    f"{name} = {want:g} but the anchor has {have:g}. The "
                    f"envelope has to be taken around the ratio's own "
                    f"denominator; a runcard whose central scale choice is not "
                    f"1 needs the legs rescaled onto it, which is not "
                    f"implemented."
                )
        self._scale_env = FrozenEnvelope(
            spec,
            self._sigma_gen_overrides,
            self.gen_axes,
            sigma_nom=sigma_gen_anchor,
            dtype=DTYPE,
        )
        # Report whether the threshold is a gen-bin edge (a bin counts only if
        # its lower edge clears it).
        edge_note = ""
        if spec.qt_min is not None:
            edges = self.gen_axes[spec.qt_axis_index][1]
            on_edge = bool(np.any(np.isclose(edges, spec.qt_min, rtol=0, atol=1e-9)))
            if on_edge:
                edge_note = f" | {spec.qt_min:g} GeV IS a gen-bin edge"
            else:
                nxt = edges[edges >= spec.qt_min]
                edge_note = (
                    f" | WARNING {spec.qt_min:g} GeV is NOT a gen-bin edge; the "
                    f"envelope starts at the next edge up, "
                    f"{float(nxt[0]) if nxt.size else float('nan'):g} GeV"
                )
        print(
            f"[SCETlibADParamModel] scale envelope {self._scale_env.describe()}"
            f"{edge_note}",
            flush=True,
        )

    def _check_fo_muf_poly(self):
        """Refuse the configurations that need a kappa_F response.

        ``fo_muf_poly=0`` skips the fixed-order muF polynomial and with it the
        kappa_F derivative (all other columns are unchanged). That would
        silently give a fitted ``resumScaleMuF`` a zero Jacobian column, and an
        envelope whose legs move kappa_F (the 7-point set) legs identical to
        nominal, i.e. an envelope that is too narrow.
        """
        if getattr(self.core, "fo_muf_poly", None) != 0:
            return
        if KAPPA_F in self._scetlib_order:
            raise ValueError(
                f"SCETlibADParamModel: fo_muf_poly=0 removes the fixed-order "
                f"kappa_F derivative, but {KAPPA_F!r} is being FITTED. Its "
                f"Jacobian column would be identically zero and the covariance "
                f"singular. Either drop it from fit_params (it is in "
                f"params.DEFAULT_FROZEN for independent reasons) or leave "
                f"fo_muf_poly at its default."
            )
        if self._env_spec is not None:
            moved = [
                i
                for i, leg in enumerate(self._env_spec.legs)
                if float(leg.get(KAPPA_F, 1.0)) != 1.0
            ]
            if moved:
                raise ValueError(
                    f"SCETlibADParamModel: fo_muf_poly=0 removes the "
                    f"fixed-order kappa_F derivative, but the scale envelope "
                    f"{self._env_spec.name!r} has {len(moved)} leg(s) that move "
                    f"{KAPPA_F} (indices {moved}). Those legs would evaluate "
                    f"identical to nominal and the envelope would be silently "
                    f"too narrow. Use the 3-point mu_R-only envelope "
                    f"(scale_envelope_points=3, the default) or leave "
                    f"fo_muf_poly at its default."
                )

    def _check_no_inert_params(self):
        """Refuse a fitted parameter the prediction does not depend on.

        A zero Jacobian column (e.g. ``resumTNP_b_qqDS`` for the Z) is a zero
        row and column of the NLL Hessian, i.e. a singular covariance.
        """
        p_start = np.asarray(self.xparamdefault.numpy(), dtype=np.float64)
        _, jac = self.core.values_and_jacobian(self._full_vector(p_start))
        J = self._fold(np.asarray(jac, dtype=np.float64))[:, self._fit_idx]
        scale = np.max(np.abs(J)) or 1.0
        dead = [
            n
            for i, n in enumerate(self._scetlib_order)
            if np.max(np.abs(J[:, i])) <= 1e-12 * scale
        ]
        if dead:
            raise ValueError(
                f"SCETlibADParamModel: {len(dead)} fitted parameter(s) have an "
                f"identically zero derivative at the start point and would make "
                f"the covariance singular: {', '.join(dead)}. Drop them from "
                f"fit_params."
            )
        # The envelope nuisances have no SCETlib column; their shape is zero if
        # the qT threshold lies above every gen bin.
        if self._scale_env is not None:
            env_dead = [
                n
                for n, k in zip(
                    self._scale_env.param_names,
                    (self._scale_env.k_avg, self._scale_env.k_diff),
                )
                if not np.any(k != 0.0)
            ]
            if env_dead:
                raise ValueError(
                    f"SCETlibADParamModel: the scale-envelope nuisance(s) "
                    f"{', '.join(env_dead)} are identically zero over this "
                    f"card's gen binning, so the prediction does not depend on "
                    f"them and the covariance would be singular. Either the qT "
                    f"threshold ({self._env_spec.qt_min} GeV) is above every gen "
                    f"bin, or the envelope really is degenerate (a perfectly "
                    f"symmetric envelope makes SymDiff vanish -- in which case "
                    f"it carries no information and should not be fitted)."
                )

    def _check_double_counting(self):
        """Refuse a card that still carries the templates our parameters replace."""
        systs = getattr(self.indata, "systs", None)
        if systs is None or len(systs) == 0:
            return
        names = [s.decode() if isinstance(s, bytes) else str(s) for s in systs]
        lowered = [(s, s.lower()) for s in names]
        for pattern, what, applies in _CONFLICTS:
            if not applies(self._param_order):
                continue
            rx = re.compile(pattern)
            clash = [s for s, low in lowered if rx.search(low)]
            if clash:
                raise ValueError(
                    f"[SCETlibADParamModel] {len(clash)} card syst(s) matching "
                    f"{pattern!r} describe the same physics as the fitted "
                    f"{what}; running both double-counts. Remake the datacard "
                    f"with setupRabbit --excludeNuisances '{pattern}' "
                    f"(case as in the card). Conflicting systs:\n"
                    + "\n".join(f"    {s}" for s in clash[:20])
                )

    # =========================================================================
    # evaluation
    # =========================================================================

    def _physical(self, theta):
        """SCETlib-block fit values -> the PHYSICAL values SCETlib expects.

        Applies params.REPARAM and the PDF coefficient scale. ``theta`` is the
        SCETlib block only, ``param[:self._n_scetlib]``.
        """
        t = np.asarray(theta, dtype=np.float64)
        return (
            np.where(self._rp_id, t * self._rp_scale, 0.0)
            + np.where(self._rp_log, np.exp(t * self._rp_L), 0.0)
            + np.where(
                self._rp_quad,
                self._rp_c[0] + self._rp_c[1] * t + self._rp_c[2] * t * t,
                0.0,
            )
        )

    def _physical_tf(self, theta):
        """:meth:`_physical` in TensorFlow, so the map is differentiated too."""
        t = tf.cast(theta, DTYPE)
        c = tf.constant(self._rp_c, dtype=DTYPE)
        return (
            tf.constant((self._rp_id * self._rp_scale).astype(np.float64), dtype=DTYPE)
            * t
            + tf.constant(self._rp_log.astype(np.float64), dtype=DTYPE)
            * tf.exp(t * tf.constant(self._rp_L, dtype=DTYPE))
            + tf.constant(self._rp_quad.astype(np.float64), dtype=DTYPE)
            * (c[0] + c[1] * t + c[2] * t * t)
        )

    def _full_vector(self, fit_values):
        """Fitted values (envelope tail dropped) -> full SCETlib vector."""
        p = self._p_base.copy()
        p[self._fit_idx] = self._physical(np.asarray(fit_values)[: self._n_scetlib])
        return p

    def _sigma_gen_np(self, p_full):
        """sigma_gen on the gen grid (flattened) at a full SCETlib vector."""
        vals, _ = self.core.values_and_jacobian(p_full)
        return self._fold(np.asarray(vals, dtype=np.float64))

    def scetlib_full_vector_tf(self, param):
        """The COMPLETE SCETlib parameter vector the cross section is evaluated
        at, as a TF expression of this model's fit vector ``param``.

        This is the one theta -> physical map; anything else that evaluates
        SCETlib at the fit point should take its parameters from here.
        """
        # A slice is fine where a scatter is not: its gradient is dense.
        p = self._physical_tf(param[: self._n_scetlib])
        # held (non-fitted slots, zero where fitted) + S.p = the full vector.
        held = self._p_base.copy()
        held[self._fit_idx] = 0.0
        return tf.constant(held, dtype=DTYPE) + tf.linalg.matvec(
            tf.constant(self._select, dtype=DTYPE), p
        )

    def _sigma_gen(self, param):
        """sigma_gen on the gen grid, differentiable via the SCETlib bridge."""
        return self._fold.fold_tf(self.core.tf_fn(self.scetlib_full_vector_tf(param)))

    def _ratio_from_param(self, param):
        """Per-fit-bin ratio to the anchor prediction, softly floored positive."""
        sigma_gen = self._sigma_gen(param)
        if self._scale_env is not None:
            nsc = self._n_scetlib
            # Gen-level template: the frozen envelope multiplies sigma_gen
            # before the response fold.
            sigma_gen = sigma_gen * self._scale_env.factor_tf(param[nsc:])
        if self.gen_level:
            ratio = sigma_gen / self.sigma_gen_central_flat
        else:
            ratio = tf.linalg.matvec(self.R, sigma_gen) / self.sigma_reco_central
        # A pathological point can drive sigma negative (NaN Poisson NLL);
        # soft-floor it far below any physical response so the loss stays finite.
        scale = tf.constant(RATIO_FLOOR_SCALE, dtype=ratio.dtype)
        return tf.maximum(
            scale * tf.math.softplus(ratio / scale),
            tf.constant(RATIO_FLOOR_MIN, dtype=ratio.dtype),
        )

    def compute(self, param, full=False):
        """(N_bins, N_proc) multiplicative scaling; only the signal column moves."""
        ratio = self._ratio_from_param(param)
        ratio_col = tf.cast(tf.reshape(ratio, [-1, 1]), self.indata.dtype)
        rnorm = 1.0 + (ratio_col - 1.0) * self._signal_col_mask

        n_masked = int(getattr(self.indata, "nbinsmasked", 0) or 0)
        if full and n_masked:
            # Masked channels sit after the fit bins in the full tensor and carry
            # no model prediction, so they scale by 1.
            rnorm = tf.concat(
                [rnorm, tf.ones([n_masked, self.nproc], dtype=rnorm.dtype)], axis=0
            )
        return rnorm

    # =========================================================================
    # introspection (validation scripts)
    # =========================================================================

    def sigma_gen_at(self, **overrides):
        """sigma_gen on the gen grid at the anchor with named overrides applied.

        E.g. ``model.sigma_gen_at(lambda2_nu=0.12)``, for validation.
        """
        p = self._p_base_anchor.copy()
        for name, val in overrides.items():
            if name not in self.rabbit_names:
                raise KeyError(f"sigma_gen_at: unknown parameter {name!r}")
            p[self.rabbit_names.index(name)] = float(val)
        return self._sigma_gen_np(p).reshape(self.gen_shape)

    def __deepcopy__(self, memo):
        """Copy the model but share the (immutable, expensive) SCETlib backend."""
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k == "core":
                new.core = v
            else:
                new.__dict__[k] = copy.deepcopy(v, memo)
        return new


def _parse_kv(spec):
    """``"a=1,b=2"`` (or a Mapping, or None) -> ``{"a": 1.0, "b": 2.0}``."""
    if not spec:
        return {}
    if not isinstance(spec, str):
        return {str(k): float(v) for k, v in dict(spec).items()}
    out = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"expected name=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = float(v)
    return out
