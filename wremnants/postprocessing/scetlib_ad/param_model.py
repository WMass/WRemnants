"""rabbit ParamModel for a fully differentiable SCETlib prediction.

The prediction is assembled in three steps::

    1  SCETlib cached rule replay  ->  sigma(p; g)       boson level, gen grid
    2  fold through the response R ->  sigma_reco(p; b)  gen -> reco
    3  ratio to the reference      ->  rnorm(b, proc)    handed to rabbit

``p`` is SCETlib's own differentiation vector, so every theory parameter the
calculation exposes is a continuous fit parameter with exact derivatives:

* ``alphaS``                     -- the strong coupling;
* the nonperturbative lambdas    -- the Collins-Soper and TMD form factors;
* the theory nuisance parameters -- ``gamma_cusp``, ``gamma_mu_q``,
  ``gamma_nu``, ``s``, ``h_qqV`` and the five beam-function TNPs, present
  whenever the runcard declares a ``[TNPs]`` block (which is also what makes the
  prediction N^{3+0}LL rather than N3LL);
* PDF eigenvector coefficients   -- carried as additional differentiable columns
  by a cache built with PDF variations, exact at ``c_e = 0, +-1``. Supplying the
  PDF set's alpha_s member pair alongside them folds the
  ``dsigma/dPDF . dPDF/dalpha_s`` piece into the ``alphaS`` slot, so ``alphaS``
  becomes the PDF-consistent coupling rather than alpha_s at fixed PDF.

* the profile scales           -- ``set_diff_scales(1)`` registers the
  resummation ``kappa_R`` and the three matching transition points
  ``x1..x3`` as differentiable, so these no longer need template nuisances.
  ``kappa_F`` gets a slot too but is INERT in the kernel: it does nothing
  unless the cache was built with the muF member pair, and a fit that tries to
  float it is refused (see :meth:`_check_no_inert_params`).

Which of these are present is a property of the cache, not of this file: the
model reads ``gradient_param_names()`` and registers what it finds.

VALIDATION STATUS of the scale directions, which is NOT uniform -- see
``scripts/rabbit/scetlib_ad/validate_variations.py``:

* the TNPs reproduce their templates to 1e-4..1e-16, the NP lambdas to ~1e-3;
* ``kappa_R`` reproduces ``kappaFO2.-kappaf0.5`` to 4.5e-03 but
  ``kappaFO0.5-kappaf2.`` only to 4.0e-02 -- the down direction is 10x worse
  than the up direction, and this is the direction that dominates
  sigma(alpha_s) (rho(alphaS, resumScaleMuR) = +0.93);
* the TRANSITION POINTS DISAGREE IN SIGN with their templates, and the cause is
  upstream, not here. All three ``transition_points*`` variations move the
  prediction the opposite way from the reference (e.g. model [1.0000, 1.1593]
  against reference [0.9602, 1.0000]). Making the identical physical change
  through the RUNCARD with ``set_diff_scales`` off reproduces the template to
  2e-6; through the registered parameter with it on, the response is sign-flipped
  and roughly -7x in slope. Moving the transition points moves ``muF`` by ~20%
  (``muF`` has its own profile over the same points) while the per-node beam
  convolutions stay frozen at the config's ``muF`` -- they shift 7-16% over that
  range. ``kappa_R`` escapes this because ``set_muR_factor`` holds ``muF`` fixed
  by construction. Not fixable from Python: the ``muF`` machinery interpolates a
  GLOBAL member while the induced shift is PER NODE. All three
  ``resumTransition*`` are therefore in :data:`params.DEFAULT_FROZEN`; that
  removes the transition-point uncertainty from the fit, which is a known gap
  rather than a fix. Re-run ``validate_variations.py`` before unfreezing.

How the derivatives get into TensorFlow
--------------------------------------
``ScetlibCachedXsecTF`` is an ordinary TF-differentiable function: its backward
pass is itself a ``custom_gradient`` whose own gradient contracts Hessian-vector
products, so nested ``GradientTape``s work and TF drives every C++ call. The model
simply calls it inside the graph, exactly as
``examples/matched_ad/tf_gradients.py`` does. There is no surrogate anywhere --
autodiff differentiates the real prediction.

One requirement that imposes, and that is easy to break by accident: map rabbit's
fit vector into SCETlib's layout with a CONSTANT 0/1 MATRIX MULTIPLY, never
``tensor_scatter_nd_update``. rabbit's vector holds only the fitted parameters,
POIs first, while SCETlib's holds every registered parameter in registry order, so
some mapping is unavoidable. A scatter's backward pass contains a gather, whose
gradient TF represents as ``tf.IndexedSlices``, and the bridge's second-order
py_function payloads call ``.numpy()`` on the incoming cotangent and fail on it --
so anything past first order breaks. The matmul is bit-identical (entries are
exactly 0 and 1) and free at these sizes (at most ~25 x 25).

XLA cannot compile a ``PyFunc``, so the fit MUST run with ``--jitCompile off``.
The model checks this at construction.

Blinding: NOTHING AT OR BELOW ``compute()`` MAY PRINT A PARAMETER VALUE
---------------------------------------------------------------------
rabbit's blinding is a change of variables inside the likelihood, so
``compute()`` is handed the TRUE physical parameter values -- that is the whole
point, since SCETlib has to be evaluated at the real alpha_s. The blinded
quantity is rabbit's internal coordinate, which is what gets reported.

The consequence is that this file is the one place that can unblind the fit by
accident. So: nothing in ``compute``, ``_ratio_from_param``, ``_sigma_gen``,
``_physical``, ``_physical_tf`` or ``_full_vector`` may ``print``, log, or
format a parameter VALUE into a message -- including an exception message.
Names, bin counts, bounds and the ANCHOR are all fine (the anchor is public by
construction: it is read from the correction's own runcard).

Every print in this file is deliberately at CONSTRUCTION time, before a fitter
exists and therefore before any offset is armed. If you add a domain assert to
the compute path, use ``tf.Assert`` with a constant string: the
``tf.debugging.assert_*`` helpers print the offending tensor by default, and
here that tensor IS the physical alpha_s.
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
from wremnants.postprocessing.scetlib_ad.xsec_backend import ScetlibADXsec

DTYPE = tf.float64

# Substring -> the registered parameter that makes a card syst a double count.
# Checked case-insensitively against indata.systs; only the entries whose model
# parameter is actually fitted are enforced, so a lambda-only run still tolerates
# a card carrying pdfAlphaS.
# Regex (matched case-insensitively against indata.systs) -> the registered
# parameter that makes a card syst a double count. Only the entries whose model
# parameter is actually FITTED are enforced, so a lambda-only run still tolerates
# a card carrying pdfAlphaS.
#
# Regex, not substring: the PDF eigenvector templates are pdf<N><SET>Sym{Avg,Diff}
# and must be told apart from pdfAlphaS, which is a different physics direction
# living in a different model parameter.
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
        "the muR / muF profile scales",
        lambda names: any(n in ("resumScaleMuR", "resumScaleMuF") for n in names),
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

        A later duplicate key wins, which is what lets a driver append overrides
        to a spec it inherited from a previous step's recorded arguments.
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
                # Do NOT fall through to positional: every argument of this model
                # is keyword-with-default, so an unknown key=value is a typo (or a
                # token that has been removed), and silently treating it as
                # positional produces "got multiple values for argument 'cache'".
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
        **kwargs,
    ):
        """
        Parameters
        ----------
        cache, conf
            The ``.npz`` written by ``ScetlibCachedXsecTF.save`` and the SCETlib
            runcard it was built from. Both are required: the cache holds the
            compressed rules and the frozen fixed-order grid, and the runcard is
            what rebuilds the identical calculation they attach to.
        gen_level
            Gen-level sigmaUL mode. The fit channel IS the gen (qT, |Y|) binning,
            so there is no response matrix and no fold: ``compute()`` returns the
            per-gen-bin ratio ``sigma_gen(p) / sigma_gen(p_anchor)``. Otherwise
            the reco fold reads R from the datacard's response auxiliary
            (see ``response_group``).
        signal_proc
            Process whose column carries the ratio; the others stay at 1.
        Q_lo, Q_hi
            The mass window the cache's single Q bin must span.
        fit_params
            Comma-separated rabbit-facing names to expose to the fit. Default:
            every parameter the cache carries except ``params.DEFAULT_FROZEN``
            (the tanh saturation scales and the b* convention, which are shape
            constants). Parameters not listed are held at their anchor and
            never reach rabbit, so they cannot contribute a zero-derivative
            (singular) Hessian row.
        poi_params
            Subset of ``fit_params`` reported as POIs (they must come first in
            the fitter's layout). Default ``alphaS``.
        threads
            SCETlib worker threads for the batch replay (0 = one per hardware
            thread).
        priors
            Declare Gaussian priors. rabbit applies priors whenever a model
            declares ``prior_sigmas``, so this token IS the decision. Off by
            default -- everything floats free. Note the TNP defaults are sigma=1
            (they are genuine nuisances), unlike the lambdas.
        prior_sigmas
            Per-name override, ``name=value,...`` or a Mapping. ``nan`` frees a
            parameter.
        xparam_default
            ``name=value,...`` shifting the fit START (and the prior mean) off
            the anchor, for injection / closure tests. The ratio
            DENOMINATOR is not moved -- it always stays the anchor.
        pdf_coeff_scale
            Confidence-level convention for the PDF eigenvector coefficients:
            the fitted ``pdfEig{i}`` is a unit nuisance theta and SCETlib is
            evaluated at ``c_e = pdf_coeff_scale * theta``. Default: resolved
            from ``theory_utils.pdfMap`` for the runcard's own ``pdf_set`` and
            the card's own ``noi`` (CT18Z + alphaS -> 1/1.645 = 0.60790), which
            is the same product ``add_pdf_uncertainty`` applies to the templates
            this replaces. Pass 1 to switch it off (theta = +-1 is then the raw
            member, i.e. 90% CL for CT18Z), or a float to override.
        anchor_source
            Where the parameter central values come from. ``"correction"``
            (default) reads them off the theory correction the card's templates
            were reweighted with, which is the only self-consistent choice: the
            model's prediction is a ratio to the anchor MULTIPLYING those
            templates, so the anchor is whatever the correction used. A card
            that records no correction config is then a hard error, because the
            central values are unknown and the cache must not be allowed to
            invent them.

            ``"cache"`` takes them from the cache instead. Deliberately a word
            rather than an off switch -- it declares a physics choice someone
            owns, namely "predict around the cache's own build point even though
            the templates were built somewhere else". That is the
            silent-wrong-answer trap documented in
            ``knowledge/20_frameworks/gen_level_sigmaul_fit.md``: the ratio is
            still 1 at the start, so nothing looks broken, but the response is
            evaluated at the wrong point. Logged loudly every time.
        anchor_override
            ``name=value,...`` supplying a central value the correction does not
            record, for the one case where neither artefact has it: a runcard
            whose NP form kept a COMPILED-IN default with no runtime key (an
            older ``tanh_6`` build hardcoded the CS-side ``lambda_6_nu`` at
            0.0007 and offered no way to set it). Overriding is auditable --
            recorded in the fit's spec and printed at construction -- where
            falling back to ``defaults.conf`` would quietly hand over 0.
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
        self.core = ScetlibADXsec(conf, cache, threads=threads)
        self.scetlib_names = list(self.core.param_names)
        self.rabbit_names = [adp.rabbit_name(n) for n in self.scetlib_names]
        # ---- The anchor: the point the prediction is a RATIO TO. It comes
        # from the theory correction the card's templates carry, NOT the
        # cache -- see params.CORR_ANCHOR_KEYS.
        self._anchor = self._resolve_anchor(anchor_source, anchor_override)

        # ---- Gen binning, and (reco path) the response matrix.
        self._setup_binning(indata, Q_lo, Q_hi)

        # ---- PDF confidence-level convention (see params.pdf_coeff_scale).
        self._conf_path = conf
        self.pdf_coeff_scale = self._resolve_pdf_coeff_scale(pdf_coeff_scale)

        # ---- Parameter registration. Everything the fit does NOT expose stays
        # pinned at the anchor, so the SCETlib vector is always complete.
        self._register_params(fit_params, poi_params, xparam_default)

        # ---- Central: the ratio denominator, evaluated by the model itself at
        # the anchor so the ratio is exactly 1 at the start whatever the card's
        # own template looks like.
        sigma_gen_anchor = self._sigma_gen_np(self._p_base_anchor)
        self.sigma_gen_central_flat = tf.constant(sigma_gen_anchor, dtype=DTYPE)
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
            f"fitting {self.nparams} of {self.core.n_params} "
            f"({self.npoi} POI: {[n for n in self._param_order[: self.npoi]]})",
            flush=True,
        )

    # =========================================================================
    # construction helpers
    # =========================================================================

    def _require_no_xla(self, kwargs):
        """Refuse to build under XLA -- a PyFunc has no XLA lowering.

        rabbit resolves ``--jitCompile auto`` to True in dense mode
        (Fitter.__init__), and the failure is an opaque compile error deep in the
        first loss evaluation, so trip here with the fix in the message.
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
            # R must encode only the gen->reco mapping, not the MC's absolute gen
            # spectrum: normalise each gen column by the gen-total (empty gen bins
            # keep a zero column).
            safe = tf.where(n_gen_flat > 0, n_gen_flat, tf.ones_like(n_gen_flat))
            self.R = R_raw / safe[tf.newaxis, :]

        self.gen_axes = [
            (name, np.asarray(edges, dtype=np.float64)) for name, edges in gen_axes
        ]
        self.gen_shape = tuple(len(e) - 1 for _, e in self.gen_axes)
        self.Q_lo, self.Q_hi = float(Q_lo), float(Q_hi)

        # compute() returns one row per fit bin, so the shape it builds must be
        # the card's. A mismatch here would surface as an opaque broadcast error
        # inside the first loss evaluation.
        n_rows = int(np.prod(self.gen_shape if self.gen_level else self.reco_shape))
        if n_rows != int(indata.nbins):
            raise ValueError(
                f"SCETlibADParamModel: the model produces {n_rows} bins "
                f"({'gen' if self.gen_level else 'reco'} shape "
                f"{self.gen_shape if self.gen_level else self.reco_shape}) but the "
                f"card has {int(indata.nbins)}."
            )
        # Exact sum of cache bins onto the gen grid: handles a different nesting
        # order, a signed-Y cache folded onto |Y|, and a cache finer than the fit's
        # gen binning. Coverage is verified, so a cache that does not tile this
        # card's gen bins raises here rather than integrating over less phase space.
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

        Read from ``theory_utils.pdfMap``, never hard coded: the runcard names
        the LHAPDF set the cache was built with and the datacard records the
        nuisance of interest, which is exactly the pair
        ``add_pdf_uncertainty`` used when it scaled the templates this replaces.
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
        # Default: every registered direction except the frozen shape constants
        # -- alpha_s, the NP lambdas, the profile scales AND the TNPs. The TNPs
        # were excluded here until 2026-08-27 on the grounds that floating ten
        # theory nuisances should be explicit; that was the wrong default (Luca).
        # They are the resummation theory uncertainty of this prediction, an
        # analysis-faithful runcard registers all ten, and leaving them fixed
        # understates the uncertainty by default -- the more dangerous of the two
        # failure modes. They are not UNconstrained: each carries an N(0,1)
        # constraint by construction, which is why `priors` now defaults to True
        # (floating a TNP with priors off still raises, see _setup_priors).
        # 'all' is therefore now the same set as the default; it is kept because
        # it is explicit at call sites and in the meta_info of every fit run
        # before this change.
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
        # Floating a direction whose response we have measured to be wrong is
        # allowed -- it is how the fix gets tested -- but never silently.
        for n in requested:
            if n in adp.KNOWN_BAD_RESPONSE:
                print(
                    f"[SCETlibADParamModel] WARNING: floating {n!r}, whose "
                    f"response is {adp.KNOWN_BAD_RESPONSE[n]}. It is in "
                    f"DEFAULT_FROZEN for that reason; this fit is a study, not a "
                    f"physics result."
                )
        pois = _as_name_tuple(poi_params)
        bad_pois = [n for n in pois if n not in requested]
        if bad_pois:
            raise ValueError(
                f"SCETlibADParamModel: poi_params {bad_pois} are not in "
                f"fit_params {list(requested)}."
            )
        # rabbit's layout contract: all POIs first, then the POUs.
        nou = tuple(n for n in requested if n not in pois)
        self._param_order = tuple(pois) + nou
        self.npoi = len(pois)
        self.npou = len(nou)
        self.params = np.array([p.encode() for p in self._param_order])

        # Position of each fitted parameter inside SCETlib's own vector. rabbit's
        # vector is NOT SCETlib's: it holds only what we fit, POIs first, while
        # SCETlib's has every registered parameter in its registry order.
        self._fit_idx = np.array(
            [available.index(n) for n in self._param_order], dtype=np.int64
        )
        # The map from rabbit's vector to SCETlib's, as a constant 0/1 matrix.
        # Deliberately NOT tensor_scatter_nd_update: a scatter's backward pass
        # contains a gather, whose gradient TF represents as tf.IndexedSlices, and
        # the SCETlib bridge's second-order py_function payloads call .numpy() on
        # the incoming cotangent and die on it. A matmul against a constant gives
        # a dense gradient, so differentiation survives at second
        # order. Bit-identical to the scatter (the entries are exactly 0 and 1)
        # and negligible in cost: (n_scetlib, n_fit) is at most ~25 x 25.
        self._select = np.zeros((len(available), len(self._param_order)))
        self._select[self._fit_idx, np.arange(len(self._param_order))] = 1.0

        # Reparametrisation (see params.REPARAM): for the profile scales the
        # FITTED parameter is a unit nuisance theta and the PHYSICAL value handed
        # to SCETlib is a function of it. Stored as coefficient vectors so one
        # vectorised expression covers every parameter, identity included, and
        # the TF path stays a handful of elementwise ops with exact derivatives.
        n_fit = len(self._param_order)
        self._rp_log = np.zeros(n_fit, dtype=bool)
        self._rp_quad = np.zeros(n_fit, dtype=bool)
        self._rp_L = np.zeros(n_fit)
        self._rp_c = np.zeros((3, n_fit))
        for i, name in enumerate(self._param_order):
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
                # value = <anchor> + width * theta. The offset is taken
                # from the anchor rather than written in REPARAM so theta = 0
                # reproduces it exactly, whatever the correction set, without
                # the map having to be kept in step by hand. Reuses the quad
                # branch: (c0, c1, 0) IS this linear map, so no new TF path.
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
        # Linear coefficient map on the identity branch, 1.0 everywhere except
        # the PDF eigenvectors: theta = +-1 has to mean 1 sigma, and a CT18Z
        # member is 90% CL. Applied to the COEFFICIENT, so the quadratic part of
        # I(c) is picked up at c^2 = scale^2 as it physically should be, rather
        # than at scale as the template route is forced to do.
        self._rp_scale = np.ones(n_fit)
        if self.pdf_coeff_scale != 1.0:
            for i, name in enumerate(self._param_order):
                if name.startswith(adp.PDF_PREFIX_OUT):
                    self._rp_scale[i] = self.pdf_coeff_scale
        self._reparametrised = tuple(
            n
            for n, f in zip(self._param_order, ~self._rp_id | (self._rp_scale != 1.0))
            if f
        )

        # Start values: the anchor, optionally shifted for injection tests.
        # _p_base_anchor is the UNSHIFTED full vector and stays the ratio
        # denominator; _p_base carries the shift for the non-fitted slots only
        # (fitted slots are overwritten from the fit vector on every call).
        self._p_base_anchor = self._anchor.copy()
        self._p_base = self._anchor.copy()
        defaults = self._anchor[self._fit_idx].copy()
        # A reparametrised parameter starts at theta = 0, NOT at its physical
        # anchor. The check below is what guarantees theta = 0 maps back onto the
        # anchor, so the ratio-to-central is exactly 1 at the start; a mistyped
        # coefficient would otherwise shift the whole prediction silently.
        defaults[~self._rp_id] = 0.0
        round_trip = self._physical(defaults)
        if not np.allclose(round_trip, self._anchor[self._fit_idx], rtol=0, atol=1e-12):
            bad = [
                (n, float(a), float(b))
                for n, a, b in zip(
                    self._param_order, round_trip, self._anchor[self._fit_idx]
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
        for name, val in _parse_kv(xparam_default).items():
            if name not in available:
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

        # Blind the POI ADDITIVELY, the way rabbit already blinds a nuisance of
        # interest -- which is what alphaS WAS before this model: the pdfAlphaS
        # template nuisance, zero-centred with |theta| = 1 equal to
        # Delta(alpha_s) = 0.002. It has the same shape here, but it is a POI
        # rather than a card nuisance, and rabbit's POI blinding is
        # MULTIPLICATIVE because its default POI is a signal strength that
        # scales yields.
        #
        # That form is wrong for us twice over. alphaS is fed to a CALCULATION
        # with a restricted domain, not used to scale yields, so a wide
        # multiplicative factor can put SCETlib somewhere it cannot be
        # evaluated (that is the 2026-09-09 bug: the fit opened at
        # xparamdefault * offset). And because the reported coordinate is then
        # alphaS_true / offset, the curvature scales as offset^2, so
        # sigma(alphaS), the POI row of the covariance and every impact on
        # alphaS all come out divided by a random number -- leaving only the
        # RELATIVE uncertainty usable, where the template treatment gave us the
        # absolute one.
        #
        # Additive restores it: a translation has unit Jacobian, so the central
        # value is hidden while the uncertainty, the covariance and the impacts
        # are exactly the unblinded ones.
        self.blind_additive = True

        active = set(self._param_order)
        groups = {
            label: tuple(p for p in members if p in active)
            for label, members in adp.IMPACT_GROUP_MEMBERS.items()
        }
        tnps = adp.tnp_group(self._param_order)
        if tnps:
            groups["resumTNP"] = tnps
        # The eigenvector coefficients are as dynamic as the TNPs (0 or n_eig of
        # them, depending on the cache), so they cannot live in the static
        # IMPACT_GROUP_MEMBERS table. Grouped as ``pdfEig``: deliberately NOT the
        # card's own ``pdfCT18ZNoAlphaS``, because the label would be a lie on a
        # cache built from any other PDF set. Compare the two by number.
        eigs = adp.pdf_group(self._param_order)
        if eigs:
            groups["pdfEig"] = eigs
        self.param_impact_groups = {k: v for k, v in groups.items() if v}

    def _setup_priors(self, priors, prior_sigmas):
        """Declare ``prior_sigmas`` only when asked; rabbit's Fitter keys off it."""
        tnps = adp.tnp_group(self._param_order)
        if tnps and not priors:
            # theta is normalised upstream so |theta| = 1 IS the recommended
            # variation; floating a TNP free discards that, and the fit will
            # happily absorb a physical effect into an unconstrained nuisance.
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

        ``core.conf``, not the runcard file: that is the runcard LAYERED ON
        SCETlib's ``defaults.conf``, which is what the calculation is actually
        configured from. It matters for coverage. Against the bare file, 33 of
        the correction's 107 keys have no counterpart and so cannot be compared
        at all -- the eleven per-flavour ``lambda2_*``, ``lambda4_i``,
        ``lambda6``, ``lambda6_nu``, ``np_model_tmd``, ``kappafo``,
        ``transition_type``, ``scale_setting`` -- and those are exactly the ones
        that would matter if a build defaulted them differently. Against the
        layered config every one of the 107 has a counterpart (measured
        2026-09-10), so the blind spot closes.

        The residual asymmetry: these are TODAY's defaults, while the correction
        side is the config SCETlib resolved when it ran. A ``defaults.conf`` edit
        since then therefore reads as a real difference -- which, for the cache,
        it is.
        """
        conf = self.core.conf
        return {section: dict(conf[section]) for section in conf.sections()}

    def _resolve_anchor(self, anchor_source, anchor_override):
        """The anchor vector over ``self.rabbit_names``, and the config check.

        The prediction is ``sigma_gen(p) / sigma_gen(p_anchor)`` multiplying the
        card's templates, and those templates were reweighted by a theory
        correction. So the ratio is 1 at the fit start only if ``p_anchor`` is
        the point THAT correction was computed at: the correction defines the
        anchor, and the cache -- a different SCETlib artefact -- does not get to.

        Note precisely what does and does not depend on where the CACHE was
        built. Correctness of the ratio is about where the cache is EVALUATED,
        which is this vector. Where it was built affects ROBUSTNESS only: the NP
        lambdas and TNPs ride the AD tape and are exact anywhere, while alphaS is
        served by a PDF member pair and the eigenvectors are exact only at
        c = 0, +-1, so those two degrade with distance.
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

        # --theoryCorrAltOnly carries the correction as alternates only, so the
        # nominal templates are UNCORRECTED. There is then no correction anchor
        # for them at all, which no choice of anchor_source can supply.
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
        # _parse_kv takes a "name=value,..." string, a Mapping, or None, and
        # coerces the values to float either way.
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

    def _check_no_inert_params(self):
        """Refuse a fitted parameter the prediction does not depend on.

        Not hypothetical: with the analysis runcard ``tnp_b_qqDS`` has an
        identically zero gradient for the Z (the channel it scales does not
        contribute), and ``lambda_inf`` / ``lambda_inf_nu`` are nearly inert at
        the anchor. A zero Jacobian column is a zero row and column of the NLL
        Hessian, i.e. a singular covariance and a meaningless "uncertainty".
        """
        p_start = np.asarray(self.xparamdefault.numpy(), dtype=np.float64)
        _, jac = self.core.values_and_jacobian(self._full_vector(p_start))
        J = self._fold(np.asarray(jac, dtype=np.float64))[:, self._fit_idx]
        scale = np.max(np.abs(J)) or 1.0
        dead = [
            n
            for i, n in enumerate(self._param_order)
            if np.max(np.abs(J[:, i])) <= 1e-12 * scale
        ]
        if dead:
            raise ValueError(
                f"SCETlibADParamModel: {len(dead)} fitted parameter(s) have an "
                f"identically zero derivative at the start point and would make "
                f"the covariance singular: {', '.join(dead)}. Drop them from "
                f"fit_params."
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
        """Fit values -> the PHYSICAL values SCETlib expects (numpy).

        Identity for everything except the reparametrised profile scales; see
        params.REPARAM for why those are unit nuisances.
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
        """:meth:`_physical` in TensorFlow, so the map is differentiated too.

        The chain rule does the rest: TF differentiates the map, SCETlib supplies
        d(sigma)/d(physical), and the composite gradient and Hessian stay exact.
        """
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
        """Fitted values -> the complete SCETlib parameter vector."""
        p = self._p_base.copy()
        p[self._fit_idx] = self._physical(fit_values)
        return p

    def _sigma_gen_np(self, p_full):
        """sigma_gen on the gen grid (flattened) at a full SCETlib vector."""
        vals, _ = self.core.values_and_jacobian(p_full)
        return self._fold(np.asarray(vals, dtype=np.float64))

    def _sigma_gen(self, param):
        """sigma_gen on the gen grid, differentiable via the SCETlib bridge.

        ``ScetlibCachedXsecTF.__call__`` is an ordinary TF-differentiable function
        -- its backward pass is itself a ``custom_gradient`` whose own gradient
        contracts Hessian-vector products -- so nested tapes work and TF drives
        every C++ call. Nothing here is a surrogate; autodiff sees the real thing.
        """
        p = self._physical_tf(param)
        # held = the non-fitted slots at their anchor, zero where we fit, so
        # held + S.p reconstructs the full vector. See _select on why this is a
        # matmul and not a scatter.
        held = self._p_base.copy()
        held[self._fit_idx] = 0.0
        p_full = tf.constant(held, dtype=DTYPE) + tf.linalg.matvec(
            tf.constant(self._select, dtype=DTYPE), p
        )
        return self._fold.fold_tf(self.core.tf_fn(p_full))

    def _ratio_from_param(self, param):
        """Per-fit-bin ratio to the anchor prediction, softly floored positive."""
        sigma_gen = self._sigma_gen(param)
        if self.gen_level:
            ratio = sigma_gen / self.sigma_gen_central_flat
        else:
            ratio = tf.linalg.matvec(self.R, sigma_gen) / self.sigma_reco_central
        # The rules put no wall in front of pathological parameters: a bad point
        # can drive sigma negative, which is a NaN Poisson NLL and a dead
        # gradient. Soft-floor so such a point is a large-but-finite penalty with
        # a usable gradient. The scale is far below any physical response, so the
        # anchor closure is untouched.
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

        ``model.sigma_gen_at(lambda2_nu=0.12)`` -- used by the validation
        scripts to compare against a native SCETlib run.
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
