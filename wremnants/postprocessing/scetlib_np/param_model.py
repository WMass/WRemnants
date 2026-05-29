"""SCETlibNPParamModel — continuous-λ rabbit ParamModel for SCETlib NP.

Architecture (template-style fit):

    σ_reco(λ; b) = Σ_g  R(b, g) · σ_gen(λ; g)
    ratio(b)     = σ_reco(λ; b) / σ_reco(λ_central; b)
    rnorm        = ratio per reco bin, broadcast over signal proc; ones elsewhere.

R is the (reco × gen) response matrix loaded from the upstream unfolding
histmaker output (a separate hdf5 from the fit-tensor input).

λ_central is read from the fit-tensor's meta_info_input via the upstream
SCETlib correction pkl (see :mod:`scetlib_lambda_central`).

σ_gen(λ; g) is evaluated on the btgrid then integrated over Q (arctan_Q²
Simpson) and rebinned (Simpson) onto the unfolding hist's gen edges
(ptVGen, absYVGen). The absYVGen-side rebin folds the signed btgrid Y axis
into |Y| bins (NP is Y-symmetric: F_eff depends on Y², γ_ν^NP doesn't depend
on Y at all).

The 8 v1 parameters (all factorisable through the current btgrid):

    γ_ν^NP (CS-side):       lambda2_nu, lambda4_nu, lambda_inf_nu
    F_eff  (TMD-effective): lambda2, lambda4, lambda6, delta_lambda2, lambda_inf

The np_model and np_model_nu strings are fixed at construction (from
λ_central). All λ values are TF Variables — differentiable in the fit.
"""

from typing import Mapping, Optional

import numpy as np
import tensorflow as tf

from rabbit.param_models.param_model import ParamModel
from wremnants.postprocessing.scetlib_np import (
    btgrid_cache,
)
from wremnants.postprocessing.scetlib_np import btgrid_integrate as fz_int
from wremnants.postprocessing.scetlib_np import btgrid_tf as fz_tf
from wremnants.postprocessing.scetlib_np import lambda_central as scetlib_lambda_central
from wremnants.postprocessing.scetlib_np import response_matrix as fz_R

# Ordered list of the v1 continuous λ. CS-side first, then TMD-effective.
GNU_PARAMS = ("lambda2_nu", "lambda4_nu", "lambda_inf_nu")
EFF_PARAMS = ("lambda2", "lambda4", "lambda6", "delta_lambda2", "lambda_inf")
ALL_PARAMS = GNU_PARAMS + EFF_PARAMS

# Physical lower bounds used by the Taylor surrogate so the finite-difference
# step (λ_central − h) can't cross into a region where F_eff / γ_ν^NP blow up.
# - lambda6 enters as `lambda6 · bT⁵ / lambda_inf` inside a tanh; if lambda6 is
#   negative the tanh saturates to −1 at large bT and the outer exp inverts
#   sign → exp(+huge). So lambda6 ≥ 0 is enforced.
# - lambda_inf / lambda_inf_nu sit in denominators (and in `exp(−2·lambda_inf·bT)`);
#   require strictly positive with a safety margin.
PARAM_MIN_VALUE = {
    "lambda2_nu": None,
    "lambda4_nu": None,
    "lambda_inf_nu": 0.05,
    "lambda2": None,
    "lambda4": None,
    "lambda6": 0.0,
    "delta_lambda2": None,
    "lambda_inf": 0.05,
}


# Theorist-recommended Gaussian prior widths for the SCETlib NP λ parameters.
# Source: NP-NP discussion slide, central values 2026-05, plus a wide
# in-house default for delta_lambda2 (the theorist hasn't quoted a width
# for it; 0 ± 0.2 is comfortably wider than its expected scale).
#
#     λ₂^ν       = 0.15 ± 0.10
#     Λ₂         = 0.40 ⁺⁰·⁶₋₀.₄   (asymmetric)
#     Λ₄         = 0.40 ⁺⁰·⁶₋₀.₄   (asymmetric)
#     δ Λ₂       = 0.00 ± 0.20      (in-house wide default; not from the slide)
#
# Asymmetric uncertainties (Λ₂, Λ₄) are approximated by a symmetric Gaussian
# with σ = (σ⁺ + σ⁻) / 2. Slightly conservative on the upper side, slightly
# loose on the lower side. A future patch can implement a split-Gaussian if
# needed.
#
# Any λ not listed here gets σ = NaN by default → no prior, floats free.
# Currently those are: lambda_inf, lambda_inf_nu, lambda4_nu, lambda6.
# They are expected to be FROZEN via rabbit's --freezeParameters until
# the theorist provides priors for them.
THEORIST_PRIOR_SIGMAS = {
    "lambda2_nu": 0.10,
    "lambda2": 0.50,  # 0.4 ⁺⁰·⁶₋₀.₄ -> symmetric average
    "lambda4": 0.50,  # 0.4 ⁺⁰·⁶₋₀.₄ -> symmetric average
    "delta_lambda2": 0.20,  # 0 ± 0.20 wide default (no theorist value yet)
}


def _crop_R_to_fit(R, R_reco_axes, fit_reco_axes, tol=1e-9):
    """Crop R's trailing reco bins so its reco shape matches the fit.

    R's reco binning is typically a superset of the fit's (e.g. R has one
    extra overflow ptll bin past the fit's last edge). For each reco axis,
    require that R's leading edges match the fit's edges; crop R along that
    axis to keep only the matching bins.
    """
    if len(R_reco_axes) != len(fit_reco_axes):
        raise ValueError(
            f"Reco axis count mismatch: R has {len(R_reco_axes)}, "
            f"fit has {len(fit_reco_axes)}"
        )
    for (rname, redges), (fname, fedges) in zip(R_reco_axes, fit_reco_axes):
        if rname != fname:
            raise ValueError(f"Reco axis name mismatch: R={rname!r} vs fit={fname!r}")
        fnb = len(fedges)
        if len(redges) < fnb:
            raise ValueError(
                f"Reco axis {rname}: R has {len(redges)-1} bins, fit needs "
                f"{fnb-1}. R is missing edges."
            )
        if not np.allclose(redges[:fnb], fedges, atol=tol):
            raise ValueError(
                f"Reco axis {rname}: leading R edges don't match fit edges. "
                f"R[:{fnb}]={list(redges[:fnb])} vs fit={list(fedges)}"
            )
    slices = tuple(slice(0, len(fedges) - 1) for (_, fedges) in fit_reco_axes)
    # Keep all gen axes (the remaining axes of R).
    slices += (slice(None),) * (R.ndim - len(fit_reco_axes))
    return R[slices]


class SCETlibNPParamModel(ParamModel):

    def __init__(
        self,
        indata,
        unfolding_hdf5_path: str,
        btgrid_dir: str,
        lambda_central: Optional[Mapping] = None,
        signal_proc: str = "Zmumu",
        Q_lo: float = 60.0,
        Q_hi: float = 120.0,
        poi_params: Optional[tuple] = (),
        prior_sigmas: Optional[Mapping] = None,
        use_taylor: bool = False,
        taylor_order: int = 2,
        taylor_h_rel: float = 0.10,
        taylor_h_min: float = 0.005,
        **kwargs,
    ):
        """Construct the ParamModel.

        Parameters
        ----------
        indata
            rabbit's input-data structure (passed by ``ph.load_models``).
        unfolding_hdf5_path
            Path to the upstream histmaker output containing
            ``nominal_postfsr_yieldsUnfolding`` for R.
        btgrid_dir
            Directory of the SCETlib bT-grid shards (fineall).
        lambda_central
            Dict with two sub-dicts ``eff_params`` and ``gnu_params`` (same
            shape as returned by :func:`scetlib_lambda_central.read_lambda_central`).
            By default λ_central is auto-detected from ``indata.metadata``;
            pass this only to override or to support hand-built indata that
            lacks metadata.
        signal_proc
            Name of the signal process whose reco yields get the per-bin
            ratio. Other processes get factor 1.
        Q_lo, Q_hi
            Z mass window for the Q-integration on the btgrid.
        poi_params
            Tuple of parameter names (subset of ``ALL_PARAMS``) to treat as
            POIs (reported as POIs in the fit output). The rest are reported
            as model nuisances (npou). The POI vs POU split is independent
            of the prior assignment (see ``prior_sigmas``).
        prior_sigmas
            Per-name override dict for the Gaussian prior σ on each
            parameter. Defaults come from ``THEORIST_PRIOR_SIGMAS``:

                lambda2_nu : 0.10
                lambda2    : 0.50   (symmetric approx of +0.6/-0.4)
                lambda4    : 0.50   (symmetric approx of +0.6/-0.4)

            All other params default to ``NaN`` → no prior, float free; in
            practice they are expected to be frozen with rabbit's
            ``--freezeParameters`` until the theorist provides priors for
            them. Pass ``np.nan`` here to free a constrained param, or a
            finite value to add a prior on one that defaults to NaN.
            Only consumed when the fitter is invoked with
            ``--paramModelPriors``; otherwise everything floats free.
            Prior mean for each param is ``self.xparamdefault`` (the
            runcard's λ_central).
        """
        self.indata = indata

        # ---- Double-counting guard
        # If the histmaker baked discrete NP κ-template variations into the
        # input HDF5, those systs describe the same physics as our continuous
        # λ POUs. Running both → double-counting (the discrete syst absorbs
        # whatever shape variation our ParamModel should describe). Warn
        # loudly if any such systs are present and unfrozen.
        self._check_discrete_np_double_counting(kwargs.get("freezeParameters"))

        # ---- λ_central
        # Three sources of λ_central, in priority order:
        #   1. ``lambda_central`` constructor arg (explicit dict).
        #   2. ``SCETLIB_NP_LAMBDA_CENTRAL_JSON`` env var (JSON-encoded dict
        #      with ``eff_params`` and ``gnu_params``). Useful when the upstream
        #      SCETlib pkl isn't accessible (e.g. colleague's input).
        #   3. Auto-detect from the fit hdf5's theoryCorr → upstream pkl.
        import json
        import os

        env_lc = os.environ.get("SCETLIB_NP_LAMBDA_CENTRAL_JSON", "").strip()
        if lambda_central is None and env_lc:
            try:
                lambda_central = json.loads(env_lc)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"SCETLIB_NP_LAMBDA_CENTRAL_JSON must be valid JSON; got {exc}"
                )
            print(f"[SCETlibNPParamModel] λ_central from env var", flush=True)
        if lambda_central is None:
            # Auto-detect from indata.metadata (loaded by rabbit's
            # FitInputData from the input HDF5's "meta" group).
            indata_meta = getattr(indata, "metadata", None) or {}
            if not indata_meta:
                raise ValueError(
                    "SCETlibNPParamModel: indata has no metadata; pass "
                    "lambda_central explicitly or use an indata that "
                    "carries metadata."
                )
            lambda_central = scetlib_lambda_central.read_lambda_central_from_meta(
                indata_meta, _source="indata.metadata"
            )
        self.eff_central = dict(lambda_central["eff_params"])
        self.gnu_central = dict(lambda_central["gnu_params"])
        self.np_model = self.eff_central["np_model"]
        self.np_model_nu = self.gnu_central["np_model_nu"]

        # ---- btgrid + dense layout
        grid = btgrid_cache.load(btgrid_dir)
        self._btgrid_meta = dict(
            shards=grid["n_shards"],
            n_bins=len(grid["bins"]),
        )
        idx_map = fz_int.dense_index_map(grid["bins"])
        self.Q_unique = idx_map["Q_unique"]
        self.Y_unique = idx_map["Y_unique"]
        self.qT_unique = idx_map["qT_unique"]
        self.flat_idx = tf.constant(idx_map["flat_idx"], dtype=tf.int64)

        # Cache btgrid arrays as TF constants.
        self.bT = tf.constant(grid["bT"], dtype=fz_tf.DTYPE)
        self.b_bar = tf.constant(grid["b_bar"], dtype=fz_tf.DTYPE)
        self.I_pert = tf.constant(grid["I_pert"][0], dtype=fz_tf.DTYPE)  # (Nbins, Nbt)
        self.C_nu = tf.constant(grid["C_nu"][0], dtype=fz_tf.DTYPE)

        # Per-bin qT and Y (from the bin tuple), for reconstruct_batch_tf.
        bins = grid["bins"]
        self.qT_per_bin = tf.constant(
            np.array([b[2] for b in bins], dtype=np.float64), dtype=fz_tf.DTYPE
        )
        Y_pb_np = np.array([b[1] for b in bins], dtype=np.float64)
        self.Y_per_bin = tf.constant(Y_pb_np, dtype=fz_tf.DTYPE)

        # F_eff depends on the bin only through Y (not Q or qT), and Y takes few
        # distinct values across the grid. Precompute the unique-Y map so
        # reconstruct_batch_tf evaluates the NP transcendentals on NY rows and
        # gathers, instead of recomputing identical rows for every (Q, qT).
        Y_feff_unique_np, Y_feff_inv_np = np.unique(Y_pb_np, return_inverse=True)
        self.Y_feff_unique = tf.constant(Y_feff_unique_np, dtype=fz_tf.DTYPE)
        self.Y_feff_inverse_idx = tf.constant(
            Y_feff_inv_np.reshape(-1).astype(np.int32), dtype=tf.int32
        )

        # Precompute the bT·J0(qT·bT) kernel (λ-independent).
        self.bT_J0_kernel = fz_tf.build_bT_J0_kernel(self.qT_per_bin, self.bT)
        self.bT_simpson_w = tf.constant(
            fz_tf.simpson_weights(np.asarray(self.bT)), dtype=fz_tf.DTYPE
        )

        # ---- Q-integration weights (arctan_Q² Simpson on Z mass window).
        self.Q_weights = tf.constant(
            fz_int.q_integrate_weights(self.Q_unique, Q_lo, Q_hi),
            dtype=fz_tf.DTYPE,
        )

        # ---- R matrix
        R_info = fz_R.load_R(unfolding_hdf5_path)
        # The fit-tensor's reco binning may differ from R's by trailing
        # overflow bins (e.g. R has ptll [0, …, 44, 100] while the fit ends
        # at 44). Crop R's trailing bins so the reco shape matches.
        fit_reco_axes = self._fit_reco_axes(indata)
        R_arr = _crop_R_to_fit(R_info["R"], R_info["reco_axes"], fit_reco_axes)
        # Tighten the metadata to match the cropped R.
        self.reco_shape = R_arr.shape[: len(fit_reco_axes)]
        self.gen_shape = R_arr.shape[len(fit_reco_axes) :]
        N_reco = int(np.prod(self.reco_shape))
        N_gen = int(np.prod(self.gen_shape))
        self.R = tf.constant(R_arr.reshape(N_reco, N_gen), dtype=fz_tf.DTYPE)
        self._reco_axes_meta = [
            (name, fit_axes[1])
            for (name, fit_axes) in zip(
                [a[0] for a in R_info["reco_axes"]],
                fit_reco_axes,
            )
        ]
        self._gen_axes_meta = R_info["gen_axes"]

        # ---- Rebin weights: btgrid (NY signed) → (NabsYVGen) via |Y| folding
        # and (NqT) → (NptVGen).
        absY_edges = self._gen_axes_meta[1][1]  # absYVGen edges
        ptVGen_edges = self._gen_axes_meta[0][1]  # ptVGen edges

        # |Y| folding: σ(Y) is symmetric in Y so the absY-bin integral is
        # 2·∫_{absY_lo}^{absY_hi} σ(Y) dY. Use Y >= 0 source samples and
        # multiply by 2.
        Y_pos_mask = self.Y_unique >= 0
        Y_pos = self.Y_unique[Y_pos_mask]
        absY_rebin_pos = fz_int.rebin_weights(Y_pos, absY_edges, name="absY")
        # Pad to full NY: zero on negative-Y columns.
        W_absY = np.zeros((absY_edges.size - 1, self.Y_unique.size), dtype=np.float64)
        W_absY[:, Y_pos_mask] = 2.0 * absY_rebin_pos
        self.W_absY = tf.constant(W_absY, dtype=fz_tf.DTYPE)

        # qT rebin: btgrid qT (signed nonneg, NqT=141) → ptVGen edges (e.g. 20 bins, 0-44).
        # Anything past ptVGen_max is out of fit range; we drop it silently for now
        # (events with gen qT > ptVGen_max are routed through R's overflow which is
        # not present in our materialised R — see plan doc, dyturbo-handoff item).
        self.W_ptVGen = tf.constant(
            fz_int.rebin_weights(self.qT_unique, ptVGen_edges, name="ptVGen"),
            dtype=fz_tf.DTYPE,
        )

        # ---- Cache σ_reco(λ_central): the denominator of the ratio.
        sigma_gen_central = self._sigma_gen_at(self.eff_central, self.gnu_central)
        # σ_reco(λ_central) = R · σ_gen(λ_central). Flatten the gen axes.
        gen_flat = tf.reshape(sigma_gen_central, [-1])
        self.sigma_reco_central = tf.linalg.matvec(self.R, gen_flat)  # (N_reco,)
        # Sanity floor — if any reco bin has zero or negative central yield,
        # the ratio would blow up. Use the central yield directly; any genuine
        # zero is a binning issue to flag.
        if tf.reduce_any(self.sigma_reco_central <= 0).numpy():
            n_bad = int(tf.reduce_sum(tf.cast(self.sigma_reco_central <= 0, tf.int32)))
            raise ValueError(
                f"SCETlibNPParamModel: {n_bad} reco bins have non-positive "
                f"σ_reco(λ_central). Likely a binning mismatch between R and "
                f"the fit-tensor reco axes."
            )

        # ---- Process index: signal column gets the ratio, others get 1.
        procs = [p.decode() if isinstance(p, bytes) else str(p) for p in indata.procs]
        if signal_proc not in procs:
            raise ValueError(
                f"SCETlibNPParamModel: signal_proc={signal_proc!r} not in "
                f"indata.procs={procs[:10]}..."
            )
        self.signal_proc_idx = procs.index(signal_proc)
        self.nproc = indata.nproc

        # ---- ParamModel registration (POIs first, then NOUs).
        poi_params = tuple(poi_params or ())
        nou_params = tuple(p for p in ALL_PARAMS if p not in poi_params)
        self._param_order = poi_params + nou_params
        self.npoi = len(poi_params)
        self.npou = len(nou_params)
        self.params = np.array([p.encode() for p in self._param_order])

        # Defaults: λ_central values per parameter. Optionally overridden by
        # the ``SCETLIB_NP_XPARAMDEFAULT`` env var — comma-separated
        # ``name=value`` pairs (for closure tests where the data-generating
        # / fit-start point should differ from the card's λ_central).
        central_lookup = {**self.eff_central, **self.gnu_central}
        defaults = np.array(
            [central_lookup[p] for p in self._param_order], dtype=np.float64
        )
        import os

        env_override = os.environ.get("SCETLIB_NP_XPARAMDEFAULT", "").strip()
        if env_override:
            overrides = dict(
                tuple(s.split("=")) for s in env_override.split(",") if s.strip()
            )
            for name, val in overrides.items():
                name = name.strip()
                if name not in self._param_order:
                    raise KeyError(f"SCETLIB_NP_XPARAMDEFAULT: unknown param {name!r}")
                i = self._param_order.index(name)
                defaults[i] = float(val)
            print(
                f"[SCETlibNPParamModel] xparamdefault overridden: {dict(zip(self._param_order, defaults))}"
            )
        # rabbit's set_param_default expects an internal-storage convention
        # where POIs (npoi entries) are SQRT(value) if not allowNegativeParam.
        # For our λ which can in principle be tiny / zero (delta_lambda2),
        # default to allowNegativeParam=True so the stored value == λ directly.
        self.allowNegativeParam = True
        self.is_linear = False
        self.xparamdefault = tf.constant(defaults, dtype=indata.dtype)

        # Gaussian priors (consumed by rabbit's Fitter when --paramModelPriors
        # is set; ignored otherwise). Default σ values come from
        # ``THEORIST_PRIOR_SIGMAS`` (lambda2_nu, lambda2, lambda4 — the only
        # params the theorist provides widths for as of 2026-05).
        # All other params default to σ = NaN → no prior, float free; in
        # practice those should be frozen via rabbit's --freezeParameters
        # until the theorist gives priors for them.
        # The ``prior_sigmas`` kwarg is a per-name override dict; pass NaN to
        # force a parameter free, or a finite value to add / change a prior.
        # The mean of each prior is λ_central (i.e. self.xparamdefault).
        prior_sigmas = dict(prior_sigmas or {})
        sigmas_arr = np.empty(self.nparams, dtype=np.float64)
        for i, p in enumerate(self._param_order):
            if p in prior_sigmas:
                sigmas_arr[i] = float(prior_sigmas[p])  # explicit override (may be NaN)
            elif p in THEORIST_PRIOR_SIGMAS:
                sigmas_arr[i] = THEORIST_PRIOR_SIGMAS[p]  # theorist recommendation
            else:
                sigmas_arr[i] = np.nan  # free (expected to be frozen)
        self.prior_sigmas = sigmas_arr
        # prior_means defaults to xparamdefault if not set, so don't store
        # redundantly — Fitter will fall back to xparamdefault.

        # ---- Taylor surrogate (optional, default OFF)
        # Precompute σ_gen(λ_central) plus first/second derivatives so that
        # per-fit-step compute() is a polynomial in (λ − λ_central) instead of
        # a full Hankel/Simpson integral. Drops per-step cost from O(10s) to
        # O(ms). Accuracy: validated against the full integral; expected
        # sub-percent within typical NP variation ranges for quadratic order.
        # Off by default — opt in via ``use_taylor=True`` kwarg (or env var
        # ``SCETLIB_NP_USE_TAYLOR=1``). Env var ``SCETLIB_NP_USE_TAYLOR=0``
        # also forces it off (useful when the kwarg is set by a CLI wrapper).
        env_taylor = os.environ.get("SCETLIB_NP_USE_TAYLOR", "").strip().lower()
        if env_taylor in ("0", "false", "no", "off"):
            use_taylor = False
            print(
                "[SCETlibNPParamModel] Taylor surrogate disabled by env var", flush=True
            )
        elif env_taylor in ("1", "true", "yes", "on"):
            use_taylor = True
            print(
                "[SCETlibNPParamModel] Taylor surrogate enabled by env var", flush=True
            )
        self.use_taylor = use_taylor
        self.taylor_order = int(taylor_order)
        if self.use_taylor:
            print(
                f"[SCETlibNPParamModel] Taylor surrogate ON "
                f"(order={int(taylor_order)}, h_rel={taylor_h_rel}, "
                f"h_min={taylor_h_min}) — per-step compute() is a polynomial "
                f"in (λ − λ_central), not the full Hankel integral",
                flush=True,
            )
            self._build_taylor_surrogate(
                h_rel=taylor_h_rel, h_min=taylor_h_min, order=self.taylor_order
            )
        else:
            print(
                "[SCETlibNPParamModel] Taylor surrogate OFF — per-step "
                "compute() runs the full Hankel/Simpson integral",
                flush=True,
            )

    # =========================================================================
    # Helpers
    # =========================================================================

    # Substrings (case-insensitive) that mark indata.systs as discrete
    # NP-template variations of one of our 8 continuous λ parameters.
    # Matching is case-insensitive because the histmaker uses inconsistent
    # casing (canonical names are uppercase Lambda, but some configurations
    # serialize them lowercase).
    #
    # Canonical names (see theory_variation_labels.py):
    #   chargeVgenNP0scetlibNPZLambda2          → catches "scetlibnpzlambda"
    #   chargeVgenNP0scetlibNPZLambda4          → catches "scetlibnpzlambda"
    #   chargeVgenNP0scetlibNPZDelta_Lambda2    → catches "scetlibnpzdelta"
    #   chargeVgenNP0scetlibNPLambda2  (W-side) → catches "scetlibnplambda"
    #   chargeVgenNP0scetlibNPLambda4  (W-side) → catches "scetlibnplambda"
    #   chargeVgenNP0scetlibNPDelta_Lambda2     → catches "scetlibnpdelta"
    #   scetlibNPgamma                          → catches "scetlibnpgamma"
    #   scetlibNPgammaEigvar{1,2,3}             → catches "scetlibnpgamma"
    #   scetlibNPgammaLambda{2,4,Inf}           → catches "scetlibnpgamma"
    _DISCRETE_NP_PATTERNS = (
        "scetlibnpzlambda",  # Z-side Lambda2 / Lambda4 templates
        "scetlibnpzdelta",  # Z-side Delta_Lambda2 template
        "scetlibnplambda",  # W-side Lambda2 / Lambda4 templates
        "scetlibnpdelta",  # W-side Delta_Lambda2 template
        "scetlibnpgamma",  # all γ_ν templates (Lambda2/4/Inf, Eigvar1/2/3, "gamma")
    )

    def _check_discrete_np_double_counting(self, freeze_patterns):
        """Detect indata systs that overlap with our continuous λ POUs.

        Three outcomes:
          - The discrete NP syst is **absent** from ``indata.systs`` entirely
            (histmaker didn't include it): nothing to do, silent return.
          - The discrete NP syst is **present and matched** by
            ``freeze_patterns``: it's frozen at central → no double-counting,
            silent return.
          - The discrete NP syst is **present and unfrozen**: it overlaps
            with one of our continuous λ POUs and will absorb shape variation
            the ParamModel should describe → print a loud banner with the
            exact freeze args to add.
        """
        import re

        systs = getattr(self.indata, "systs", None)
        if systs is None or len(systs) == 0:
            return
        syst_names = [s.decode() if isinstance(s, bytes) else str(s) for s in systs]

        # Case-insensitive substring match: canonical names use uppercase
        # Lambda but some histmaker outputs lowercase the names.
        conflicting = [
            s
            for s in syst_names
            if any(pat in s.lower() for pat in self._DISCRETE_NP_PATTERNS)
        ]
        if not conflicting:
            return  # not in indata.systs at all → nothing to warn about

        # Which of those are NOT already covered by a user-supplied freeze
        # pattern (exact match or anchored regex)?
        patterns = list(freeze_patterns or [])
        unfrozen = []
        for s in conflicting:
            covered = False
            for pat in patterns:
                if pat == s:
                    covered = True
                    break
                try:
                    if re.fullmatch(pat, s):
                        covered = True
                        break
                except re.error:
                    continue
            if not covered:
                unfrozen.append(s)

        if not unfrozen:
            return  # all conflicting systs are already frozen by the user

        print(
            "\n"
            "===================================================================\n"
            "[SCETlibNPParamModel] DOUBLE-COUNTING WARNING\n"
            "===================================================================\n"
            f"Detected {len(unfrozen)} discrete NP κ-template syst(s) in the\n"
            "input HDF5 that describe the same physics as this ParamModel's\n"
            "continuous λ parameters. Running both leads to double-counting:\n"
            "the discrete syst absorbs shape variation that the ParamModel\n"
            "should describe (the indata syst will show a spurious pull, and\n"
            "the postfit λ values are not what the data actually prefers).\n\n"
            "Unfrozen conflicting systs:\n"
            + "\n".join(f"    {s}" for s in unfrozen)
            + "\n\n"
            "Fix by adding to --freezeParameters, e.g.:\n"
            "    --freezeParameters '.*scetlibNPZ.*lambda.*' "
            "'.*scetlibNPgammaLambda.*' ...\n"
            "or list them explicitly:\n"
            "    --freezeParameters " + " ".join(repr(s) for s in unfrozen) + "\n"
            "===================================================================",
            flush=True,
        )

    def _fit_reco_axes(self, indata):
        """Read the (name, edges) of each reco axis from the (single) channel.

        Multi-channel support is deferred to v2; this raises if there's >1
        non-masked channel.
        """
        non_masked = [
            (name, info)
            for name, info in indata.channel_info.items()
            if not info.get("masked", False)
        ]
        if len(non_masked) != 1:
            raise NotImplementedError(
                f"SCETlibNPParamModel v1 supports a single non-masked channel; "
                f"got {len(non_masked)}: {[n for n, _ in non_masked]}"
            )
        _, info = non_masked[0]
        return [
            (ax.name, np.asarray(ax.edges, dtype=np.float64)) for ax in info["axes"]
        ]

    # =========================================================================
    # σ_gen evaluation
    # =========================================================================

    def _sigma_gen_at(self, eff_params, gnu_params):
        """Evaluate σ_gen(λ) on R's gen binning. Returns shape (NptVGen, NabsYVGen)."""
        # 1. Reconstruct σ on the btgrid's flat (Nbins,) layout.
        sigma_flat = fz_tf.reconstruct_batch_tf(
            qT_per_bin=self.qT_per_bin,
            bT=self.bT,
            I_pert=self.I_pert,
            C_nu=self.C_nu,
            b_bar=self.b_bar,
            Y_per_bin=self.Y_per_bin,
            eff_params={k: v for k, v in eff_params.items() if k != "np_model"},
            gnu_params={k: v for k, v in gnu_params.items() if k != "np_model_nu"},
            np_model=self.np_model,
            np_model_nu=self.np_model_nu,
            bT_J0_kernel=self.bT_J0_kernel,
            bT_simpson_weights=self.bT_simpson_w,
            Y_unique=self.Y_feff_unique,
            Y_inverse_idx=self.Y_feff_inverse_idx,
        )
        # 2. Sparse → dense (NQ, NY, NqT). Missing cells get 0.
        sigma_dense = fz_int.sparse_to_dense_tf(sigma_flat, self.flat_idx)
        # 3. Integrate over Q (arctan_Q² Simpson) → (NY, NqT).
        sigma_YqT = fz_int.integrate_over_Q_tf(sigma_dense, self.Q_weights)
        # 4. Rebin Y (signed) → absYVGen (|Y|-folded): (NabsYVGen, NqT).
        sigma_absY_qT = fz_int.rebin_axis_tf(sigma_YqT, axis=0, weights=self.W_absY)
        # 5. Rebin qT → ptVGen: (NabsYVGen, NptVGen).
        sigma_absY_ptV = fz_int.rebin_axis_tf(
            sigma_absY_qT, axis=1, weights=self.W_ptVGen
        )
        # 6. Reorder to (NptVGen, NabsYVGen) to match R's gen axis order.
        return tf.transpose(sigma_absY_ptV, perm=[1, 0])

    # =========================================================================
    # Taylor surrogate
    # =========================================================================

    def _eff_gnu_from_array(self, lambdas_np):
        """Helper: numpy 8-vector → (eff_params dict, gnu_params dict)."""
        eff = {"np_model": self.np_model}
        gnu = {"np_model_nu": self.np_model_nu}
        for i, name in enumerate(self._param_order):
            v = float(lambdas_np[i])
            if name in EFF_PARAMS:
                eff[name] = v
            elif name in GNU_PARAMS:
                gnu[name] = v
            else:
                raise KeyError(name)
        return eff, gnu

    def _sigma_gen_at_lambdas(self, lambdas_np):
        """Wrapper around :meth:`_sigma_gen_at` taking a flat numpy λ vector."""
        eff, gnu = self._eff_gnu_from_array(lambdas_np)
        return self._sigma_gen_at(eff, gnu).numpy()

    def _build_taylor_surrogate(self, h_rel, h_min, order):
        """Precompute σ_gen(λ_central), ∂σ_gen/∂λ_i, and (order≥2) ∂²σ_gen/∂λ_i∂λ_j.

        Uses central finite differences with step size h_i = max(|λ_i|·h_rel, h_min).
        Total full-Hankel evaluations: 1 + 16 + (28 if order≥2) = 45.

        Stored as ``tf.constant`` so the per-step ``_sigma_gen_taylor`` is
        a small polynomial evaluation.
        """
        import time

        t_start = time.time()
        lambda_central = self.xparamdefault.numpy().astype(np.float64)
        n = len(lambda_central)
        h_i = np.maximum(np.abs(lambda_central) * h_rel, h_min)
        # Per-parameter strategy:
        #   "central" — symmetric ±h FD: D = (σ⁺ − σ⁻)/(2h), H_ii = (σ⁺ − 2σ₀ + σ⁻)/h²
        #   "forward" — one-sided FD using σ₀, σ⁺ at h, σ⁺⁺ at 2h. Used when
        #       λ_c is at the physical lower bound (e.g. lambda6 = 0 in
        #       FranksVals). σ⁻ would be unphysical and produce overflow.
        # Off-diagonal Hessian uses the σ⁺⁺_ij stencil which doesn't need σ⁻,
        # so it works for both modes.
        fd_mode = ["central"] * n
        for i, name in enumerate(self._param_order):
            min_v = PARAM_MIN_VALUE.get(name)
            if min_v is None:
                continue
            max_h_allowed = lambda_central[i] - min_v
            if max_h_allowed <= 0:
                # Boundary case: use forward FD. Keep h_i at its preferred value.
                fd_mode[i] = "forward"
            else:
                h_i[i] = min(h_i[i], 0.95 * max_h_allowed)
        print(
            f"[SCETlibNPParamModel] building Taylor surrogate (order={order}); "
            f"h_i={dict(zip(self._param_order, h_i.round(4).tolist()))}",
            flush=True,
        )

        # σ₀ at λ_central
        sigma_0 = self._sigma_gen_at_lambdas(lambda_central)  # (NptVGen, NabsYVGen)
        elapsed = time.time() - t_start
        print(f"  central done in {elapsed:.1f}s; shape {sigma_0.shape}", flush=True)

        # σ₊ᵢ at λ_central + h_i e_i; σ₋ᵢ at λ_central − h_i e_i
        sigma_plus = np.empty((n,) + sigma_0.shape, dtype=np.float64)
        sigma_minus = np.empty_like(sigma_plus)
        # Storage: for "central" mode params we keep σ⁻; for "forward" mode
        # params we keep σ⁺⁺ (at +2h) in the same slot. The use site selects
        # the correct stencil per-param.
        sigma_other = np.empty_like(sigma_plus)
        for i in range(n):
            lp = lambda_central.copy()
            lp[i] += h_i[i]
            sigma_plus[i] = self._sigma_gen_at_lambdas(lp)
            if fd_mode[i] == "central":
                lm = lambda_central.copy()
                lm[i] -= h_i[i]
                sigma_other[i] = self._sigma_gen_at_lambdas(lm)
                tag = "±h"
            else:  # forward
                lpp = lambda_central.copy()
                lpp[i] += 2.0 * h_i[i]
                sigma_other[i] = self._sigma_gen_at_lambdas(lpp)
                tag = "+h,+2h (forward)"
            print(
                f"  {tag} for {self._param_order[i]} ({i+1}/{n}) at "
                f"t+{time.time()-t_start:.1f}s",
                flush=True,
            )
        sigma_minus = sigma_other  # retain old name for the central-FD slots

        # First derivatives D_i and diagonal Hessian H_ii — stencil depends on mode.
        D = np.empty_like(sigma_plus)
        H_full = None
        if order >= 2:
            H_full = np.zeros((n, n) + sigma_0.shape, dtype=np.float64)
        for i in range(n):
            if fd_mode[i] == "central":
                D[i] = (sigma_plus[i] - sigma_minus[i]) / (2.0 * h_i[i])
                if H_full is not None:
                    H_full[i, i] = (sigma_plus[i] - 2.0 * sigma_0 + sigma_minus[i]) / (
                        h_i[i] ** 2
                    )
            else:  # forward; sigma_minus[i] actually holds σ⁺⁺ at +2h
                D[i] = (4.0 * sigma_plus[i] - sigma_minus[i] - 3.0 * sigma_0) / (
                    2.0 * h_i[i]
                )
                if H_full is not None:
                    H_full[i, i] = (sigma_minus[i] - 2.0 * sigma_plus[i] + sigma_0) / (
                        h_i[i] ** 2
                    )
        if H_full is not None:
            # Off-diagonals: 1-sided stencil
            #   H_ij ≈ [σ(λ_c + h_i e_i + h_j e_j) − σ(λ_c + h_i e_i)
            #           − σ(λ_c + h_j e_j) + σ(λ_c)] / (h_i h_j)
            # 8·7/2 = 28 additional evaluations.
            for i in range(n):
                for j in range(i + 1, n):
                    lpp = lambda_central.copy()
                    lpp[i] += h_i[i]
                    lpp[j] += h_i[j]
                    sigma_pp = self._sigma_gen_at_lambdas(lpp)
                    H_ij = (sigma_pp - sigma_plus[i] - sigma_plus[j] + sigma_0) / (
                        h_i[i] * h_i[j]
                    )
                    H_full[i, j] = H_ij
                    H_full[j, i] = H_ij  # symmetric
                    print(
                        f"  H[{self._param_order[i]},{self._param_order[j]}] "
                        f"at t+{time.time()-t_start:.1f}s",
                        flush=True,
                    )

        # Stash as tf.constants
        dtype = self.indata.dtype
        self._taylor_lambda_central = tf.constant(lambda_central, dtype=dtype)
        self._taylor_sigma_central_2d = tf.constant(sigma_0, dtype=dtype)
        self._taylor_D = tf.constant(D, dtype=dtype)  # (n, NptVGen, NabsYVGen)
        self._taylor_H = (
            tf.constant(H_full, dtype=dtype) if H_full is not None else None
        )
        self._taylor_h_i = h_i  # kept for diagnostics
        print(
            f"[SCETlibNPParamModel] surrogate ready in "
            f"{time.time()-t_start:.1f}s; storage "
            f"~{(1 + n + (n*n if H_full is not None else 0)) * sigma_0.size * 8 / 1e6:.1f} MB",
            flush=True,
        )

    def _sigma_gen_taylor(self, lambdas_tf):
        """Evaluate σ_gen(λ) via the precomputed Taylor surrogate.

        Returns shape (NptVGen, NabsYVGen). All ops are linear in (λ−λ_central)
        for order=1, plus a quadratic correction for order=2.
        """
        delta = lambdas_tf - self._taylor_lambda_central  # (n,)
        sigma = self._taylor_sigma_central_2d + tf.tensordot(
            delta, self._taylor_D, axes=1
        )  # (NptVGen, NabsYVGen)
        if self._taylor_H is not None:
            quad = 0.5 * tf.einsum("i,j,ijab->ab", delta, delta, self._taylor_H)
            sigma = sigma + quad
        return sigma

    # =========================================================================
    # compute
    # =========================================================================

    def _unpack_params(self, param):
        """Map flat param tensor to eff/gnu dicts in canonical order."""
        eff_params = {"np_model": self.np_model}
        gnu_params = {"np_model_nu": self.np_model_nu}
        # param values are stored directly (allowNegativeParam=True).
        for i, name in enumerate(self._param_order):
            v = param[i]
            if name in EFF_PARAMS:
                eff_params[name] = v
            elif name in GNU_PARAMS:
                gnu_params[name] = v
            else:
                raise KeyError(name)
        return eff_params, gnu_params

    def compute(self, param, full=False):
        """Return per-(bin, proc) scaling tensor.

        Shape: (N_reco, N_proc). Signal-proc column carries the per-reco-bin
        ratio σ_reco(λ; b) / σ_reco(λ_central; b); other columns are 1.
        """
        import time

        t_start = time.perf_counter()
        if self.use_taylor:
            # Fast path: polynomial in (λ − λ_central) with precomputed coefficients.
            sigma_gen = self._sigma_gen_taylor(param)  # (NptVGen, NabsYVGen)
        else:
            eff_params, gnu_params = self._unpack_params(param)
            sigma_gen = self._sigma_gen_at(eff_params, gnu_params)
        gen_flat = tf.reshape(sigma_gen, [-1])
        sigma_reco = tf.linalg.matvec(self.R, gen_flat)  # (N_reco,)
        ratio = sigma_reco / self.sigma_reco_central  # (N_reco,)

        # Build (N_reco, N_proc) scaling: ones except signal column.
        N_reco = int(self.sigma_reco_central.shape[0])
        col_ones = tf.ones([N_reco, self.nproc], dtype=self.indata.dtype)
        # Place ratio into the signal column.
        ratio_col = tf.cast(tf.reshape(ratio, [N_reco, 1]), self.indata.dtype)
        # mask: one-hot vector for signal_proc_idx.
        mask = tf.one_hot(
            self.signal_proc_idx, self.nproc, dtype=self.indata.dtype
        )  # (N_proc,)
        # rnorm = ones + (ratio - 1) * one_hot_signal
        # → (ratio at signal col, 1 elsewhere)
        rnorm = col_ones + (ratio_col - 1.0) * mask[tf.newaxis, :]

        # ---- Diagnostic timing ----
        if not hasattr(self, "_n_compute_calls"):
            self._n_compute_calls = 0
            self._t_compute_total = 0.0
            import atexit

            atexit.register(self._print_compute_summary)
        self._n_compute_calls += 1
        self._t_compute_total += time.perf_counter() - t_start
        if self._n_compute_calls % 100 == 0:
            n, t = self._n_compute_calls, self._t_compute_total
            print(
                f"[SCETlibNPParamModel] compute() running: {n} calls, "
                f"total {t:.1f}s, mean {t*1000/n:.1f} ms/call",
                flush=True,
            )
        return rnorm

    def _print_compute_summary(self):
        """Print final tally of compute() calls and time. Atexit hook."""
        n = getattr(self, "_n_compute_calls", 0)
        t = getattr(self, "_t_compute_total", 0.0)
        if n == 0:
            return
        print(
            f"[SCETlibNPParamModel] compute() FINAL: {n} calls, "
            f"total {t:.2f}s wall in compute(), "
            f"mean {t*1000/n:.2f} ms/call",
            flush=True,
        )
