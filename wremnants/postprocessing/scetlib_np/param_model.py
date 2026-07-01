"""SCETlibNPParamModel — continuous-λ rabbit ParamModel for SCETlib NP.

Scales the signal reco template by a per-bin ratio of the SCETlib
nonperturbative (NP) prediction at the fitted λ vs at λ_central. Built in four
steps; :mod:`response_matrix`, :mod:`btgrid_tf` and the validation scripts refer
here for the derivation.

    Step 1   btgrid Hankel + Q integral   →  σ_resum(λ; g)   resummed, gen grid
    Step 2   + fixed-order matching       →  σ_gen(λ; g) = σ_resum + σ_ns
    Step 3   fold through response R       →  σ_reco(λ; b)    gen → reco
    Step 4   ratio vs λ_central           →  rnorm(b, proc)  handed to rabbit

Steps 1–3 build the absolute cross section; Step 4 forms the per-bin variation
the fit consumes.

Code split: Steps 1–2 (the datacard-free physics — btgrid integral → matched
σ_gen on a gen grid, from btgrid + λ_central + gen edges) live in
:class:`~wremnants.postprocessing.scetlib_np.sigma_gen.SigmaGenModel`. This class
is the loader/rabbit adapter: it resolves λ_central, the gen grid, and (in the
reco path) R / N_gen from the datacard, holds a ``SigmaGenModel`` as
``self.core``, and adds Steps 3–4. ``_sigma_gen_at`` / ``_sigma_YqT_native_at``
alias the core; its physics attributes (``eff_central``, ``np_model``,
``sigma_ns``, …) are reachable as ``model.<name>`` via ``__getattr__``.

NP physical validity: a wrong-sign point (λ2_ν < 0, or the λ2_eff < 0
divergence) makes the form factors anti-damping and the differential σ(qT)
oscillate negative, but the qT-rebin into the coarse gen grid averages that away
— so the binned σ_gen, σ_reco and likelihood stay smooth and positive, and the
fit gets no signal from the likelihood to avoid the unphysical region.
Enforcement is the rabbit ``Regularizer`` ``np_damping_wall.NPDampingWall`` (a
one-sided hinge on the λ, via ``-r``; hardness via ``--regularizationStrength``),
encoding the tanh_2 damping conditions (CS: λ2_ν ≥ 0, λ4_ν ≥ 0; TMD: λ2_Y ≥ 0
and 3·λ∞²·λ4 + λ2_Y³ ≥ 0). Postfit-only validity checks (``np_damping_ok``,
``spectrum_negativity``) live in ``param_model_diagnostics``.

Indices: Q, Y, qT are the btgrid axes (mass / rapidity / qT); g = gen bin
(ptVGen, absYVGen); b = reco bin (ptll, yll, cosThetaStarll_quantile,
phiStarll_quantile). λ splits into λ_eff (F_eff) and λ_ν (γ_ν^NP).

=============================================================================
Step 1 — σ_resum(λ; g): the resummed prediction from the bT grid
=============================================================================

(1a) Per-(Q, Y, qT) bT-space (Hankel) integral — the NP parameters enter HERE:

    σ(Q, Y, qT; λ) = qT · ∫ dbT  bT · J₀(qT·bT)
                          · I_pert(Q, Y, qT;  b*(bT))
                          · exp[ C_ν(Q, Y, qT;  bT) · γ_ν^NP(b*(bT); λ_ν) ]
                          · F_eff(Y;  b*(bT); λ_eff)

Each factor, bare bT vs the b*-frozen b̄T:

  bT          bare bT, the Hankel integration variable. Enters ONLY the
              measure factor bT and the Fourier kernel J₀(qT·bT). The leading
              qT· prefactor is SCETlib's x = qT·bT integration convention.
  b*(bT)      the b*-prescription b_star_global(bT), cached as ``b_bar``: bT
              frozen below b_max so the NP factors never reach the Landau pole.
              EVERY NP-carrying factor is evaluated at b*(bT) — I_pert, γ_ν^NP,
              and F_eff — never at bare bT.
  I_pert      perturbative bT-space integrand with NP off. Cached per
              (Q, Y, qT) along the bT axis; λ-independent.
  C_ν         coefficient of γ_ν^NP in the rapidity (CS) log evolution. Cached
              per (Q, Y, qT) AND bT — a full (Nbins, Nbt) array; λ-independent.
              Taken at the BARE bT: the lone NP-exponent factor NOT frozen to
              b* (per the SCETlib convention).
              Because it varies with bT it sits inside the bT integral (not a
              constant out front); because it varies with Q the Q integral
              cannot be collapsed ahead of the fit — the λ-dependence does not
              factor through ∫dQ (exp is nonlinear in C_ν·γ_ν^NP).
  γ_ν^NP      CS-side NP rapidity anomalous dimension; depends on b*(bT) and
              λ_ν only (no Q/Y/qT). λ-dependent.
  F_eff       TMD-effective NP factor; depends on Y² and b*(bT) and λ_eff only
              (no Q/qT). λ-dependent.

Only γ_ν^NP and F_eff carry λ; everything else (bT, J₀, I_pert, C_ν, b*) is
λ-independent and precomputed at construction (the bT·J₀ kernel, bT Simpson
weights, arctan_Q² Q-weights). The reconstruction uses a memory-factorized layout
(deduplicated (I_pert, C_ν) rows + J₀ on the unique-qT grid + Simpson-as-matmul),
~6× smaller than the dense (Nbins, Nbt) and small enough for a 32 GB GPU; it is
memoized in a ``.npz`` next to ``combined_btgrid.pkl`` (staleness auto-detected)
so repeat constructions skip the ~18 GB raw load.

(1b) Integrate over Q, then rebin onto the gen grid:

    σ_resum(λ; g) = rebin_{qT→ptVGen, |Y|→absYVGen} [ ∫_{Q_lo}^{Q_hi} dQ  σ(Q, Y, qT; λ) ]

The Q integral uses an arctan_Q² Simpson rule (the x = arctan((Q²−q0²)/(q0·Γ))
transform flattens the Breit-Wigner Z peak). The (Y, qT) result is rebinned
(Simpson) onto the unfolding hist's gen edges: qT → ptVGen, and the signed btgrid
Y axis folded into |Y| → absYVGen. The |Y| fold is valid because NP is
Y-symmetric (F_eff depends on Y², γ_ν^NP doesn't depend on Y at all).

=============================================================================
Step 2 — σ_gen(λ; g): add the NP-independent fixed-order nonsingular
=============================================================================

    σ_gen(λ; g) = σ_resum(λ; g) + σ_ns(g)

    σ_ns(g)     = rebin_{qT→ptVGen, |Y|→absYVGen} [ σ_DYTurbo^FO − σ_SCETlib-sing^FO ]

The fixed-order matching adds the nonsingular piece σ_ns = (DYTurbo fixed order)
− (SCETlib singular fixed order), from the original FO inputs (the
``…_nnlo_sing…combined.pkl`` and the DYTurbo ``results_…scetlibmatch.txt``),
Q-windowed to [Q_lo, Q_hi], |Y|-folded, zeroed below qt_cutoff, and summed onto
the (ptVGen, absYVGen) gen bins (see :func:`compute_nonsingular_gen`).

σ_ns is NP-INDEPENDENT (the same for every λ) and added at GEN level, so it folds
through the same response R as σ_resum in Step 3. Because the fit uses a ratio
(Step 3), σ_ns DILUTES the NP variation where the FO dominates (high qT). σ_ns is
ALWAYS included (it is what the histmaker nominal carries); for resum-only
diagnostics subtract the exposed ``sigma_ns`` from ``sigma_gen_central`` (and
re-fold with ``R`` for reco level).

Known limitation — qT > 100 GeV truncation. Both σ_gen ingredients stop at
qT = 100 (the bT grid and the DYTurbo FO input), so σ_gen(λ; g) ≡ 0 above 100,
while R and N_gen lump the full unbounded qT > last_edge into the (last_edge, 100]
overflow (PTVGEN_OVERFLOW_EDGE, see :mod:`response_matrix`). The under-fed overflow
makes the top in-range reco bin (ptll [37, 44]) ~3% low vs the histmaker nominal
and trips the agreement guard, but does NOT bias the fit: the missing piece is
fixed-order, NP-independent, so it cancels in the Step-4 ratio. A high-ptll
extension would need DYTurbo FO past 100, higher bT-grid/overflow ceilings, and
the histmaker correction remade (it shares the truncated inputs).

=============================================================================
Step 3 — σ_reco(λ; b): fold gen → reco through the response matrix
=============================================================================

    P(b | g)     = R_raw(b, g) / N_gen(g)               (efficiency × migration)
    σ_reco(λ; b) = Σ_g  P(b | g) · σ_gen(λ; g)

g = gen bin (ptVGen, |Y|), summed over by Σ_g; b = reco bin (ptll, yll,
cosThetaStarll_quantile, phiStarll_quantile). P(b | g) is the gen→reco map (one
reco column per gen bin), so σ_reco is σ_gen pushed through the detector; Σ_g is
``tf.linalg.matvec(self.R, σ_gen_flat)``. Pure detector folding — no λ_central or
ratio here (that is Step 4).

Factors (all loaded by :mod:`response_matrix`; see it for the hist mechanics):

  R_raw(b, g)   reco×gen yield from the unfolding histmaker output
                (``nominal_prefsr_yieldsUnfolding``, sample Zmumu): slice
                acceptance=True (gen-fiducial), then project to reco×gen —
                which SUMS the helicitySig axis. R is filled with
                ``nominal_weight_helicity``, a PARTITION of the event weight
                into the 8 helicity pieces that ADD BACK UP, so the physical
                yield is the helicitySig SUM. Units: reco-selected,
                gen-fiducial weighted event counts (already × reco efficiency).
  N_gen(g)      gen-total normalizer: the xnorm ``prefsr`` hist, gen-fiducial
                but BEFORE reco selection. Takes the UL component
                (helicitySig = −1), NOT the sum — N_gen is filled with
                ``csAngularMoments``, a moment expansion whose A_i bins (0..7)
                do NOT sum to σ (some are negative); UL is the
                angular-integrated total.
                ⚠ Same axis as R, OPPOSITE reduction: R is a weight partition
                (SUM helicitySig), N_gen is a moment expansion (take UL).
                Taking UL of R instead would discard the angular partition and
                inflate the closure ~15×.

Why N_gen and not the reco-passing marginal Σ_b R_raw(b, g): that marginal
already carries efficiency, so dividing by it cancels efficiency (migration-only)
and closes far worse — efficiency is strongly gen-dependent (ε ≈ 0.07–0.54 here).
N_gen(g), the true generated total, makes P = R_raw/N_gen the theory-independent
gen→reco map. σ_gen, R, N_gen must all sit at the same (pre-FSR) gen level; the
postfsr variants close ~1% worse.

=============================================================================
Step 4 — rnorm(b, proc): the per-reco-bin variation handed to rabbit
=============================================================================

    ratio(b)       = σ_reco(λ; b) / σ_reco(λ_central; b)
    rnorm(b, proc) = 1 + (ratio(b) − 1) · [proc is signal]   (1 in every other proc)

The only object that leaves the model: compute() returns rnorm(b, proc), and
rabbit multiplies the signal reco template by it (other processes stay at 1).
Dividing by σ_reco(λ_central) (precomputed at construction) cancels the
event-count↔σ scale and overall normalization, so rnorm carries purely the SHAPE
of the per-reco-bin NP variation.

-----------------------------------------------------------------------------
Parameters and inputs
-----------------------------------------------------------------------------

The 8 v1 parameters λ (all factorisable through the current btgrid):

    γ_ν^NP (CS-side):       lambda2_nu, lambda4_nu, lambda_inf_nu
    F_eff  (TMD-effective): lambda2, lambda4, lambda6, delta_lambda2, lambda_inf

λ_central is read from the fit-tensor's metadata, where the histmaker stored
the SCETlib correction's NP runcard (see :mod:`lambda_central`). The np_model and
np_model_nu strings are fixed at construction (from λ_central). All λ values are
TF Variables, differentiable in the fit.

=============================================================================
Postfit Hessian / covariance (uncertainties on λ)
=============================================================================

rabbit's covariance step differentiates the bT fold once per fit parameter
(~3754) and can't see it depends only on the ≤8 λ, re-materializing the
(Ng × Nbt) ≈ 8.75 GB slab per parameter → ~33 TB → OOM. Fix: a straight-through
surrogate (``_ratio_straightthrough`` + ``_ratio_compact_jac`` /
``_ratio_compact_hess``) exposing only a compact quadratic in λ to autodiff
(J = dratio/dλ, K = d²ratio/dλ² by forward-mode AD) while the exact ratio value
stays under stop_gradient, so the big slab never enters the differentiated graph.
Enable via the ``hessian_straightthrough=1`` / ``hessian_gn=1`` spec tokens and a
two-pass run (fit ``--noHessian``, then covariance at the postfit with
``--externalPostfit --noFit``). GN (``hessian_gn=1``, drops the K term) is exact
for Asimov and the default; full-K is for real/toy data. Do NOT enable during the
fit. Full derivation, GN-vs-full-K, and the exact commands: ``docs/HESSIAN_PLAN.md``.
"""

import os
from typing import Mapping, Optional

import numpy as np
import tensorflow as tf

from rabbit.param_models.param_model import ParamModel
from wremnants.postprocessing.scetlib_np import btgrid_tf as fz_tf
from wremnants.postprocessing.scetlib_np import lambda_central as scetlib_lambda_central

# Physics core (Steps 1–2) lives in :mod:`sigma_gen`; this module is the
# datacard/rabbit adapter (Steps 3–4), holding a :class:`SigmaGenModel` as
# ``self.core``. The λ-name tuples, σ_ns builder, and default-btgrid helper are
# re-exported here for backward compat (older imports referenced them on this
# module). ``fz_tf`` is still needed for the reco-side tensor dtype (``fz_tf.DTYPE``).
from wremnants.postprocessing.scetlib_np.params import DEFAULT_PRIOR_SIGMAS
from wremnants.postprocessing.scetlib_np.sigma_gen import (  # noqa: F401
    ALL_PARAMS,
    EFF_PARAMS,
    GNU_PARAMS,
    SigmaGenModel,
    _NONSING_DYTURBO_DEFAULT,
    _NONSING_FO_SING_DEFAULT,
    _default_btgrid_dir,
    compute_nonsingular_gen,
)

_DISCRETE_NP_SUBSTRING = "scetlibnp"

# Positivity floor for the per-bin reco ratio in compute() (see its docstring).
# A pathological λ (e.g. λ4 < 0 with the bounded-tanh model) can drive σ_reco —
# and the predicted signal yield — negative, giving NaN Poisson NLL and stalling
# the minimizer. Soft-floor the ratio to a small positive value (softplus, not a
# hard clamp), so a bad point is a LARGE-BUT-FINITE penalty with non-zero gradient.
#   RATIO_FLOOR_SCALE — softplus transition width. FAR below any physical response
#     so healthy ratios (~0.9–1.1, every validated λ-variation) pass to machine
#     precision: scale·softplus(r/scale) == r for r ≫ scale.
#   RATIO_FLOOR_MIN — hard positive ground: softplus underflows to exactly 0 for
#     the extreme (r ~ -1e43) case, keeping the yield strictly > 0 (no NaN).
RATIO_FLOOR_SCALE = 1.0e-4
RATIO_FLOOR_MIN = 1.0e-9

def _crop_R_to_fit(R, R_reco_axes, fit_reco_axes, tol=1e-9):
    """Crop R's trailing reco bins so its reco shape matches the fit.

    R's reco binning is typically a superset of the fit's (e.g. R has one
    extra overflow ptll bin past the fit's last edge). For each reco axis,
    require R's leading edges to match the fit's edges and crop R along that
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


def _R_info_from_auxiliary(indata):
    """Reconstruct the response-matrix dict from the datacard's ``scetlib_np``
    auxiliary bundle.

    setupRabbit extracts R (and the gen-total N_gen, reco/gen axis names + edges)
    once from the unfolding histmaker output and embeds it in the fit input via
    rabbit's ``add_auxiliary``; rabbit exposes it as ``FitInputData.auxiliary``.
    The model reads R ONLY from there (one source, one path), so R is always
    consistent with the run that produced the datacard. The returned dict matches
    :func:`response_matrix.load_R`'s shape for the keys this model consumes
    (``R``, ``N_gen``, and ``reco_axes`` / ``gen_axes`` as ordered
    ``(name, edges)`` lists).
    """
    aux = getattr(indata, "auxiliary", None) or {}
    if "scetlib_np" not in aux:
        raise ValueError(
            "SCETlibNPParamModel: the datacard has no 'scetlib_np' auxiliary (the "
            "reco×gen response matrix R). Rebuild the datacard with a setupRabbit "
            "that embeds it from a mz_dilepton --unfolding input (it must carry "
            "'nominal_prefsr_yieldsUnfolding' and the 'prefsr' gen-total)."
        )
    bundle = aux["scetlib_np"]
    n_gen = bundle.get("N_gen")
    return dict(
        R=np.asarray(bundle["R"], dtype=np.float64),
        N_gen=None if n_gen is None else np.asarray(n_gen, dtype=np.float64),
        reco_axes=[
            (name, np.asarray(bundle[f"edges__{name}"], dtype=np.float64))
            for name in bundle["reco_axes"]
        ],
        gen_axes=[
            (name, np.asarray(bundle[f"edges__{name}"], dtype=np.float64))
            for name in bundle["gen_axes"]
        ],
    )


class SCETlibNPParamModel(ParamModel):

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        import inspect

        sig = inspect.signature(cls.__init__)
        valid = {n: p for n, p in sig.parameters.items() if n not in ("self", "indata")}
        positional = []
        for tok in args:
            key = tok.split("=", 1)[0] if isinstance(tok, str) and "=" in tok else None
            if key in valid:
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
        btgrid_dir: Optional[str] = None,
        lambda_central=None,
        signal_proc: str = "Zmumu",
        Q_lo: float = 60.0,
        Q_hi: float = 120.0,
        poi_params: Optional[tuple] = (),
        priors: bool = False,
        prior_sigmas: Optional[Mapping] = None,
        nonsingular_fo_sing: str = _NONSING_FO_SING_DEFAULT,
        nonsingular_dyturbo: str = _NONSING_DYTURBO_DEFAULT,
        nonsingular_qt_cutoff: float = 1.0,
        xparam_default: Optional[str] = None,
        hessian_straightthrough: bool = False,
        hessian_gn: bool = False,
        gen_level: bool = False,
        check_agreement: bool = True,
        check_agreement_threshold: float = 0.005,
        check_agreement_strict: bool = False,
        check_agreement_min_yield: float = 0.0,
        np_model_fit: Optional[str] = None,
        np_model_nu_fit: Optional[str] = None,
        **kwargs,
    ):
        """Construct the ParamModel.

        Usage::
            --paramModel wremnants.postprocessing.scetlib_np.SCETlibNPParamModel [key=value ...]

        Parameters
        ----------
        indata
            rabbit's input-data structure (passed by ``ph.load_models``). The
            reco×gen response matrix R (and the gen-total N_gen, axis names +
            edges) is read from ``indata.auxiliary["scetlib_np"]``, embedded in
            the datacard by setupRabbit from a ``mz_dilepton.py --unfolding``
            histmaker output (see :func:`_R_info_from_auxiliary` and
            :mod:`response_matrix`). There is no file-path argument; R always
            comes from the fit input it is consistent with.
        btgrid_dir
            Directory holding the SCETlib bT-grid ``combined_btgrid.pkl``.
            Defaults (when None) to the shared data-area copy next to NanoAOD
            (``_default_btgrid_dir()``, built on the ROOT/narf-free
            ``wremnants.utilities.data_paths.getDataPath()``); pass explicitly
            at non-subMIT sites.
        lambda_central
            Dict with two sub-dicts ``eff_params`` and ``gnu_params`` (same
            shape as returned by :func:`lambda_central.read_lambda_central`).
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
        priors
            Enable Gaussian priors on the λ parameters (spec token
            ``priors=1``). Rabbit applies priors whenever a ParamModel
            *declares* ``prior_sigmas``; there is no rabbit-side CLA (the
            old ``--paramModelPriors`` flag was dropped in WMass/rabbit#133),
            the model itself decides. This token IS that decision: only when
            set does the model declare ``prior_sigmas``. Default off →
            everything floats free.
        prior_sigmas
            Per-name override for the Gaussian prior σ on each parameter:
            a Mapping, or as spec token the comma-separated string form
            ``prior_sigmas=lambda2=0.3,delta_lambda2=nan`` (same format as
            ``xparam_default``). Defaults come from ``DEFAULT_PRIOR_SIGMAS``
        nonsingular_fo_sing, nonsingular_dyturbo
            Paths to the σ_ns inputs (SCETlib singular pkl / DYTurbo
            scetlibmatch txt); default to the wremnants-data
            TheoryCorrections copies. σ_ns is always included: the matched
            σ_gen^matched(λ) = σ_gen^resum(λ) + σ_ns is what the histmaker
            nominal carries; resum-only diagnostics subtract ``sigma_ns``.
        nonsingular_qt_cutoff
            Low-qT cutoff (GeV) below which σ_ns is zeroed (the FO−singular
            difference is numerically unreliable at tiny qT).
        xparam_default
            Comma-separated ``name=value,...`` string shifting the fit START
            (and the prior mean) off the runcard's λ_central, for closure /
            injection tests. The truth (ratio denominator) is NOT moved.
        hessian_straightthrough
            Expose compact λ-derivatives (J, optionally K) to autodiff while
            keeping the exact value, for the one-shot two-pass covariance
            recipe ONLY (see the module docstring); never set during a fit.
        hessian_gn
            With ``hessian_straightthrough``: Gauss-Newton, keep J and drop
            the K term (exact for Asimov, where the residual vanishes).
        gen_level
            Gen-level σUL fit mode (spec token ``gen_level=1``). The fit channel
            IS the gen (ptVGen, |Y|) binning, so there is NO response matrix and
            NO gen→reco fold (Step 3 is skipped): compute() returns the
            per-GEN-bin ratio σ_gen(λ) / σ_gen(λ_central) from Steps 1–2. The
            gen binning is read from the single fit channel's axes, so no
            ``scetlib_np`` auxiliary / R / N_gen is needed. Used for the
            direct-theory σUL closure (this ParamModel as the λ model, fit
            against injected gen-level σUL pseudodata).
        check_agreement
            Run the in-fit reco agreement guard at construction (default ON;
            spec token ``check_agreement=0`` disables). It compares the model's
            σ_reco(λ_central) SHAPE to the card's signal nominal template
            and trips if ANY bin's |σ_reco(λ_c)/card_nominal − 1| exceeds
            ``check_agreement_threshold``, catching the misuse where the
            model's λ_central baseline doesn't match the card.
        check_agreement_threshold
            Per-bin trip threshold for ``check_agreement`` (fractional;
            default 0.005 = 0.5%).
        check_agreement_strict
            Raise instead of warn when the guard trips (spec token
            ``check_agreement_strict=1``). Default warns (lists worst bins).
        check_agreement_min_yield
            Optional reference-yield floor (fraction of the max reco bin) below
            which bins are ignored by the guard, suppressing sparse/near-empty
            corner bins where a shape residual is meaningless. Default 0 (every
            bin counts).
        np_model_fit, np_model_nu_fit
            Optional override of the F_eff / γ_ν functional form used for the
            NUMERATOR σ(λ) (spec tokens ``np_model_fit=...`` / ``np_model_nu_fit=
            tanh_6``). The DENOMINATOR (central σ(λ_c), the ratio's reference) is
            ALWAYS the card's form — fixed by λ_central, immutable — so the model
            stays consistent with the histmaker template. Default (None) → the
            numerator uses the card form too (rnorm(λ_c)=1, unchanged behaviour).
            Setting e.g. ``np_model_nu_fit=tanh_6`` makes the fit predict in
            tanh_6 while transporting from the tanh_2 baseline:
            rnorm = σ^(fit)(λ) / σ^(card)(λ_c). NOTE this is a model change —
            validate the fit form against a matching SCETlib reference before
            trusting results, and ``lambda6_nu`` (the tanh_6 b⁶ coefficient) is a
            normal fittable λ (default 0, inert under tanh_2).
        """
        self.indata = indata

        if btgrid_dir is None:
            btgrid_dir = _default_btgrid_dir()

        self._check_discrete_np_double_counting()

        # ---- λ_central
        # Anchor point of the model; must match the SCETlib NP runcard the input's
        # theory correction was built with (so rnorm(λ_central) == 1). Two sources,
        # priority order:
        #   1. ``lambda_central`` constructor arg (explicit dict) — standalone
        #      diagnostic scripts; not reachable from the rabbit CLI.
        #   2. Auto-detect from the fit hdf5's propagated histmaker metadata — the
        #      production path. No CLI override by design: a mismatched anchor
        #      silently biases the fit, so an input lacking the metadata must be
        #      remade, not overridden.
        lambda_central_source = (
            "constructor-arg" if lambda_central is not None else None
        )
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
            lambda_central_source = "auto-detect:indata.metadata theoryCorr"
            print(
                f"[SCETlibNPParamModel] λ_central auto-detected from indata.metadata",
                flush=True,
            )
        print(f"[SCETlibNPParamModel] λ_central:", flush=True)
        for key, value in lambda_central.items():
            print(f"  {key} = {value!r}", flush=True)

        self.lambda_central_source = lambda_central_source

        # ---- Hessian straight-through switches (module docstring's two-pass
        # recipe). Spec tokens hessian_straightthrough=1 / hessian_gn=1, recorded
        # in the fitresults meta via the stored --paramModel spec.
        self._hess_st = bool(hessian_straightthrough)
        self._hess_gn = bool(hessian_gn)

        # ---- Gen/reco binning. Resolve the gen grid — and, in the reco path, the
        # response matrix R (and gen-total N_gen) — from the datacard, then hand the
        # gen edges to the physics core (Steps 1–2: btgrid integral → matched σ_gen
        # on that grid). gen_level=1: the fit channel IS the gen (ptVGen, absY)
        # binning, so NO response matrix and NO gen→reco fold — compute() returns the
        # per-GEN-bin ratio σ_gen(λ)/σ_gen(λ_central) — and the scetlib_np auxiliary
        # / N_gen are not required.
        self.gen_level = bool(gen_level)
        if self.gen_level:
            gen_axes = self._fit_reco_axes(indata)
            if len(gen_axes) != 2:
                raise NotImplementedError(
                    "gen_level SCETlibNPParamModel expects a single fit channel "
                    "with 2 gen axes (ptVGen, absY); got "
                    f"{[n for n, _ in gen_axes]}"
                )
            R_arr = None
            N_gen_arr = None
            self.reco_shape = None
            self._reco_axes_meta = None
        else:
            # ---- R matrix (read from the datacard's scetlib_np auxiliary)
            R_info = _R_info_from_auxiliary(indata)
            # Fit-tensor reco binning may differ from R's by trailing overflow bins
            # (e.g. R has ptll [0, …, 44, 100] while the fit ends at 44). Crop R's
            # trailing bins so the reco shape matches.
            fit_reco_axes = self._fit_reco_axes(indata)
            R_arr = _crop_R_to_fit(R_info["R"], R_info["reco_axes"], fit_reco_axes)
            self.reco_shape = R_arr.shape[: len(fit_reco_axes)]
            gen_axes = R_info["gen_axes"]
            # Gen-total denominator N_gen(g) from the xnorm hist ("prefsr"):
            # generated fiducial yield per gen bin (pre-reco-selection). Dividing R
            # by it gives the theory-independent efficiency×migration response.
            # REQUIRED: setupRabbit only embeds the response when the gen-total is
            # present, so N_gen should always be here; raise if not (a σ_gen(λ_c)
            # proxy would make the central closure circular — see module docstring).
            if R_info.get("N_gen") is None:
                raise ValueError(
                    "SCETlibNPParamModel: the 'scetlib_np' auxiliary has no N_gen "
                    "(gen-total). Rebuild the datacard from a histmaker output that "
                    "carries the 'prefsr' xnorm hist."
                )
            N_gen_arr = R_info["N_gen"]
            self._reco_axes_meta = [
                (name, fit_axes[1])
                for (name, fit_axes) in zip(
                    [a[0] for a in R_info["reco_axes"]],
                    fit_reco_axes,
                )
            ]
        self._gen_axes_meta = gen_axes

        # ---- Physics core (Steps 1–2): btgrid Hankel + arctan-Q² Simpson +
        # |Y|/qT rebin + NP-independent fixed-order nonsingular → matched σ_gen(λ)
        # on the (ptVGen, absYVGen) gen grid. Datacard-free, rebuildable standalone
        # from (btgrid, λ_central, gen edges); the model delegates all σ_gen
        # evaluation to it (see _sigma_gen_at / _sigma_YqT_native_at and __getattr__
        # forwarding of its physics attributes — eff_central, np_model, Y_unique,
        # sigma_ns, …).
        self.core = SigmaGenModel(
            btgrid_dir=btgrid_dir,
            lambda_central=lambda_central,
            gen_axes=gen_axes,
            Q_lo=Q_lo,
            Q_hi=Q_hi,
            nonsingular_fo_sing=nonsingular_fo_sing,
            nonsingular_dyturbo=nonsingular_dyturbo,
            nonsingular_qt_cutoff=nonsingular_qt_cutoff,
        )
        self.gen_shape = self.core.gen_shape

        # ---- Numerator form (default = card form). The denominator (central
        # σ(λ_c) below) ALWAYS uses the card's form via the core, so consistency
        # with the histmaker template is fixed; only the numerator σ(λ) may use a
        # different form when np_model_(nu_)fit is given (see the constructor doc).
        self._np_model_fit = np_model_fit or self.core.np_model
        self._np_model_nu_fit = np_model_nu_fit or self.core.np_model_nu
        if self._np_model_fit not in fz_tf.EFF_MODELS:
            raise ValueError(
                f"np_model_fit={self._np_model_fit!r} not in {sorted(fz_tf.EFF_MODELS)}"
            )
        if self._np_model_nu_fit not in fz_tf.GNU_MODELS:
            raise ValueError(
                f"np_model_nu_fit={self._np_model_nu_fit!r} not in "
                f"{sorted(fz_tf.GNU_MODELS)}"
            )
        if (self._np_model_fit, self._np_model_nu_fit) != (
            self.core.np_model,
            self.core.np_model_nu,
        ):
            print(
                f"[SCETlibNPParamModel] NUMERATOR form overridden: "
                f"F_eff {self.core.np_model}->{self._np_model_fit}, "
                f"γ_ν {self.core.np_model_nu}->{self._np_model_nu_fit}; "
                f"denominator (central) stays card form "
                f"({self.core.np_model}/{self.core.np_model_nu}). "
                f"VALIDATE against a matching SCETlib reference before trusting.",
                flush=True,
            )

        # ---- Reco fold (Step 3) / gen-level baseline (Step 4 denominator).
        # σ_gen(λ_central) comes from the core. R must encode only the gen→reco
        # *mapping*, not the MC's absolute gen spectrum, so normalize each gen
        # column by the gen-total N_gen(g):
        #     P(b|g)        = R_raw(b,g) / N_gen(g)            (eff × migration)
        #     σ_reco(λ_c;b) = Σ_g P(b|g) · σ_gen(λ_c;g)
        # The reco-passing marginal Σ_b R_raw(b,g) is the WRONG normalizer (R is
        # post-reco-selection, already carrying efficiency; dividing by it cancels
        # efficiency and closes far worse). Because σ_reco_central then depends on
        # σ_gen(λ_c) — it does NOT collapse to R_raw·1 — the λ_central closure is a
        # genuine test of the integral.
        gen_flat = tf.reshape(self.core.sigma_gen_central, [-1])
        if self.gen_level:
            # Gen-level σUL fit: fit bins ARE the gen bins, so the per-bin ratio
            # denominator is σ_gen(λ_central) directly (no reco fold).
            self.R = None
            self._N_gen_flat = None
            self.sigma_gen_central_flat = gen_flat
        else:
            N_reco = int(np.prod(self.reco_shape))
            N_gen = int(np.prod(self.gen_shape))
            R_raw = tf.constant(R_arr.reshape(N_reco, N_gen), dtype=fz_tf.DTYPE)
            self._N_gen_flat = tf.constant(N_gen_arr.reshape(-1), dtype=fz_tf.DTYPE)
            # Guard empty gen bins (no generated events): leave column at 0.
            safe_N_gen = tf.where(
                self._N_gen_flat > 0,
                self._N_gen_flat,
                tf.ones_like(self._N_gen_flat),
            )
            self.R = R_raw / safe_N_gen[tf.newaxis, :]  # P(b|g) = R_raw / N_gen
            self.sigma_reco_central = tf.linalg.matvec(
                self.R, gen_flat
            )  # Σ_g P·σ_gen(λ_c)
            if tf.reduce_any(self.sigma_reco_central <= 0).numpy():
                n_bad = int(
                    tf.reduce_sum(tf.cast(self.sigma_reco_central <= 0, tf.int32))
                )
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
        # Precompute the (1, N_proc) one-hot selecting the signal column, reused in
        # every compute() to place the per-bin ratio (others stay at 1).
        self._signal_col_mask = tf.reshape(
            tf.one_hot(self.signal_proc_idx, self.nproc, dtype=indata.dtype),
            [1, self.nproc],
        )

        # ---- ParamModel registration (POIs first, then NOUs).
        poi_params = tuple(poi_params or ())
        nou_params = tuple(p for p in ALL_PARAMS if p not in poi_params)
        self._param_order = poi_params + nou_params
        self.npoi = len(poi_params)
        self.npou = len(nou_params)
        self.params = np.array([p.encode() for p in self._param_order])

        # Impact groups over our own parameters, consumed by the Fitter's
        # traditional impacts. rabbit's built-in systgroup machinery can't represent
        # these: its group indices are syst-relative, but our λ are POUs (model
        # nuisances) with no syst index. The Fitter resolves these labels -> floating
        # full-x indices and computes the conditional group impact from the
        # covariance. Split into the two NP sectors: CS-side γ_ν (lambda*_nu) vs
        # TMD-effective F_eff.
        #
        # ``resumNonpert`` is the SAME group name setupRabbit assigns to the discrete
        # scetlibNP* template variations in the old-style datacard (there == exactly
        # those 4 nuisances). Emitting it here (= all our lambda) makes the
        # grouped-impact bar directly comparable between the new param model and the
        # old NP variations. No collision with a syst group: the new-model datacard
        # excludes scetlibNP, so resumNonpert is absent from indata.systgroups.
        self.param_impact_groups = {
            "resumNonpert": tuple(ALL_PARAMS),
            "scetlibNPgammaNu": tuple(GNU_PARAMS),
            "scetlibNPFeff": tuple(EFF_PARAMS),
        }

        # Defaults: λ_central values per parameter. Optionally overridden by the
        # ``xparam_default=name=value,...`` spec token — comma-separated pairs (for
        # closure tests where the data-generating / fit-start point should differ
        # from the card's λ_central).
        central_lookup = {**self.eff_central, **self.gnu_central}
        defaults = np.array(
            [central_lookup[p] for p in self._param_order], dtype=np.float64
        )

        start_override = (xparam_default or "").strip()
        if start_override:
            overrides = dict(
                tuple(s.split("=")) for s in start_override.split(",") if s.strip()
            )
            for name, val in overrides.items():
                name = name.strip()
                if name not in self._param_order:
                    raise KeyError(f"xparam_default: unknown param {name!r}")
                i = self._param_order.index(name)
                defaults[i] = float(val)
            print(
                f"[SCETlibNPParamModel] xparamdefault overridden: {dict(zip(self._param_order, defaults))}",
                flush=True,
            )
        # rabbit's set_param_default stores POIs (npoi entries) as SQRT(value) if not
        # allowNegativeParam. Our λ can be tiny/zero (delta_lambda2), so default to
        # allowNegativeParam=True: stored value == λ directly.
        self.allowNegativeParam = True
        self.is_linear = False
        self.xparamdefault = tf.constant(defaults, dtype=indata.dtype)

        # Gaussian priors (semantics on the constructor args). Rabbit's Fitter
        # applies them whenever the model DECLARES ``prior_sigmas`` (no rabbit-side
        # CLA — WMass/rabbit#133), so the declaration is gated behind ``priors``: off
        # → no attribute → everything floats free. The Fitter takes prior means from
        # xparamdefault, so an xparam_default shift moves start AND prior mean
        # together (centring priors on truth while starting shifted would need
        # prior_means decoupled from xparamdefault).
        self._use_priors = bool(priors)
        # ``prior_sigmas`` may be a Mapping (programmatic) or the spec-token string
        # ``prior_sigmas=lambda2=0.3,delta_lambda2=nan`` — same comma-separated
        # name=value format as xparam_default; value ``nan`` frees the param.
        if isinstance(prior_sigmas, str):
            prior_sigmas = dict(
                tuple(s.split("=")) for s in prior_sigmas.split(",") if s.strip()
            )
        prior_sigmas = {k.strip(): v for k, v in dict(prior_sigmas or {}).items()}
        for name in prior_sigmas:
            if name not in self._param_order:
                raise KeyError(f"prior_sigmas: unknown param {name!r}")
        if self._use_priors:
            sigmas_arr = np.empty(self.nparams, dtype=np.float64)
            for i, p in enumerate(self._param_order):
                if p in prior_sigmas:
                    sigmas_arr[i] = float(
                        prior_sigmas[p]
                    )  # explicit override (may be NaN)
                elif p in DEFAULT_PRIOR_SIGMAS:
                    sigmas_arr[i] = DEFAULT_PRIOR_SIGMAS[p]  # theorist recommendation
                else:
                    sigmas_arr[i] = np.nan  # free (expected to be frozen)
            self.prior_sigmas = sigmas_arr
            # prior_means defaults to xparamdefault if not set, so don't store
            # redundantly — Fitter falls back to xparamdefault.
            print(
                "[SCETlibNPParamModel] Gaussian priors ENABLED (priors=1); "
                "applied by rabbit's Fitter (pre-#133 rabbit additionally "
                "needs --paramModelPriors):",
                flush=True,
            )
            for i, p in enumerate(self._param_order):
                if np.isfinite(sigmas_arr[i]) and sigmas_arr[i] > 0:
                    print(f"  {p}: σ = {sigmas_arr[i]:.4g}", flush=True)
        elif prior_sigmas:
            print(
                "[SCETlibNPParamModel] WARNING: prior_sigmas overrides given "
                "but priors are not enabled (pass priors=1); ignoring them — "
                "all λ float free.",
                flush=True,
            )

        # ---- In-fit reco agreement guard (default ON; check_agreement=0 off).
        # Compares the model's σ_reco(λ_central) SHAPE to the card's signal nominal
        # template (what rnorm multiplies) and trips on any bin over
        # check_agreement_threshold. Pure-numpy, runs once, never crashes the fit.
        # Skipped in gen_level mode (no reco fold). Full reco+gen report and plots in
        # param_model_diagnostics.run_card_diagnostics.
        self.check_agreement = bool(check_agreement)
        self.check_agreement_threshold = float(check_agreement_threshold)
        self.check_agreement_strict = bool(check_agreement_strict)
        self.check_agreement_min_yield = float(check_agreement_min_yield)
        if self.check_agreement and not self.gen_level:
            from wremnants.postprocessing.scetlib_np import (
                param_model_diagnostics as _diag,
            )

            _diag.run_reco_guard(
                self,
                indata,
                threshold=self.check_agreement_threshold,
                strict=self.check_agreement_strict,
                min_yield_frac=self.check_agreement_min_yield,
            )

        # ---- Publish this fully-built model on the shared indata so the
        # NPDampingWall regularizer can derive the FIT (numerator) forms and the
        # canonical λ order from it, instead of being told them a second time on
        # the -r line. indata is the only object both this model (built first in
        # rabbit_fit.load_models) and the regularizer's mapping (built afterward
        # with the same indata) share. The card form in indata.metadata is the
        # WRONG one for the wall — the wall must constrain the numerator σ(λ),
        # which np_model_(nu_)fit may override. See np_damping_wall.py.
        indata.scetlib_np_param_model = self

    @property
    def fit_forms(self):
        """The NUMERATOR (fit) NP forms the wall must constrain — NOT the card
        form (``self.np_model`` / ``self.np_model_nu``, forwarded from the core,
        is the immutable denominator form). Returns ``{"np_model", "np_model_nu"}``
        with the F_eff / γ_ν forms actually integrated for σ(λ). The canonical λ
        order is ``self._param_order`` (poi_params first, then the rest)."""
        return {
            "np_model": self._np_model_fit,
            "np_model_nu": self._np_model_nu_fit,
        }

    def _check_discrete_np_double_counting(self):
        """Refuse to run on a datacard containing discrete scetlibNP systs.

        They describe the same physics as this ParamModel's continuous λ, so
        running both double-counts: the discrete syst absorbs shape variation
        the ParamModel should describe (spurious pull on the indata syst,
        postfit λ not what the data prefers).
        """

        systs = getattr(self.indata, "systs", None)
        if systs is None or len(systs) == 0:
            return
        syst_names = [s.decode() if isinstance(s, bytes) else str(s) for s in systs]

        conflicting = [s for s in syst_names if _DISCRETE_NP_SUBSTRING in s.lower()]
        if not conflicting:
            return

        raise ValueError(
            f"[SCETlibNPParamModel] {len(conflicting)} discrete scetlibNP "
            "κ-template syst(s) found in the input HDF5; they describe the "
            "same physics as this ParamModel's continuous λ parameters and "
            "running both double-counts. Remake the datacard without them "
            "(setupRabbit --excludeNuisances '.*scetlibNP.*'). Conflicting "
            "systs:\n" + "\n".join(f"    {s}" for s in conflicting)
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
    # σ_gen evaluation — delegated to the physics core (Steps 1–2)
    # =========================================================================

    def __getattr__(self, name):
        """Forward physics attributes (σ_gen tensors, btgrid axes, λ_central) to
        ``self.core``.

        Python calls ``__getattr__`` only when normal attribute lookup fails, so
        this never shadows a real ParamModel attribute; it makes the physics
        surface the core owns (``eff_central``, ``gnu_central``, ``np_model``,
        ``Y_unique``, ``qT_unique``, ``sigma_ns``, ``sigma_YqT_central``,
        ``sigma_gen_central``, …) reachable as ``model.<name>`` for backward
        compatibility. Guarded against the ``core`` name itself and the pre-core
        construction window to avoid recursion.
        """
        if name != "core":
            core = self.__dict__.get("core")
            if core is not None and hasattr(core, name):
                return getattr(core, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def _sigma_YqT_native_at(self, eff_params, gnu_params, np_model=None, np_model_nu=None):
        """Backward-compat alias for :meth:`SigmaGenModel.sigma_YqT_native` — the
        native (NY, NqT) Q-integrated σ(λ) before the |Y|-fold / qT-rebin.
        ``np_model`` / ``np_model_nu`` override the form (default: card form)."""
        return self.core.sigma_YqT_native(
            eff_params, gnu_params, np_model=np_model, np_model_nu=np_model_nu
        )

    def _sigma_gen_at(
        self, eff_params, gnu_params, sigma_YqT=None, np_model=None, np_model_nu=None
    ):
        """Backward-compat alias for :meth:`SigmaGenModel.sigma_gen` — the matched
        σ_gen(λ) on the (NptVGen, NabsYVGen) gen grid (Steps 1–2).
        ``np_model`` / ``np_model_nu`` override the form (default: card form)."""
        return self.core.sigma_gen(
            eff_params, gnu_params, sigma_YqT=sigma_YqT,
            np_model=np_model, np_model_nu=np_model_nu,
        )

    # =========================================================================
    # λ-vector helpers
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

    # =========================================================================
    # ratio(λ) and the straight-through compact-derivative path (Hessian Phase B)
    # =========================================================================

    def _ratio_from_param(self, param):
        """λ (full param vector) → floored per-reco-bin ratio, shape (N_reco,).

        The differentiable map the straight-through Hessian path wraps. The soft
        positivity floor (see ``compute``) lives here so the normal and
        straight-through paths apply it identically.
        """
        eff_params, gnu_params = self._unpack_params(param)
        # Numerator uses the fit form (default = card form); the denominator
        # (sigma_reco_central / sigma_gen_central_flat) was built from the core's
        # card-form central, so it always stays the histmaker-consistent baseline.
        sigma_gen = self._sigma_gen_at(
            eff_params,
            gnu_params,
            np_model=self._np_model_fit,
            np_model_nu=self._np_model_nu_fit,
        )
        gen_flat = tf.reshape(sigma_gen, [-1])
        if self.gen_level:
            # Gen-level σUL fit: the fit bins ARE the gen bins — no reco fold.
            ratio = gen_flat / self.sigma_gen_central_flat  # (N_gen,)
        else:
            sigma_reco = tf.linalg.matvec(self.R, gen_flat)  # (N_reco,)
            ratio = sigma_reco / self.sigma_reco_central  # (N_reco,)
        scale = tf.constant(RATIO_FLOOR_SCALE, dtype=ratio.dtype)
        ratio = tf.maximum(
            scale * tf.math.softplus(ratio / scale),
            tf.constant(RATIO_FLOOR_MIN, dtype=ratio.dtype),
        )
        return ratio

    def _ratio_compact_jac(self, param):
        """J = d(ratio)/d(param), shape (N_reco, nparam), via forward-mode AD.

        One JVP per parameter (nparam ≤ 8), each a single bT-fold pass, NOT
        tiled over params. The compact object the Hessian needs from the
        fold."""
        n = int(param.shape[0])
        cols = []
        for i in range(n):
            tangent = tf.one_hot(i, n, dtype=param.dtype)
            with tf.autodiff.ForwardAccumulator(param, tangent) as acc:
                r = self._ratio_from_param(param)
            cols.append(acc.jvp(r))  # (N_reco,) = dratio/dparam_i
        return tf.stack(cols, axis=1)  # (N_reco, nparam)

    def _ratio_compact_hess(self, param):
        """K = d²(ratio)/d(param)², shape (N_reco, nparam, nparam), forward-over-forward.

        nparam² JVP-of-JVP passes (≤ 64), each one bT-fold pass, never tiled."""
        n = int(param.shape[0])
        rows = []
        for i in range(n):
            ti = tf.one_hot(i, n, dtype=param.dtype)
            cols = []
            for j in range(n):
                tj = tf.one_hot(j, n, dtype=param.dtype)
                with tf.autodiff.ForwardAccumulator(param, tj) as acc_j:
                    with tf.autodiff.ForwardAccumulator(param, ti) as acc_i:
                        r = self._ratio_from_param(param)
                    di = acc_i.jvp(r)  # (N_reco,)
                cols.append(acc_j.jvp(di))  # (N_reco,) = d²ratio/dparam_i dparam_j
            rows.append(tf.stack(cols, axis=1))  # (N_reco, nparam) over j
        return tf.stack(rows, axis=2)  # (N_reco, j, i) — symmetric in (i, j)

    def _ratio_straightthrough(self, param, use_curvature=True):
        """Exact ratio value, but autodiff sees only a compact quadratic in the
        ≤8 λ (J, and optionally K), so the (N_grid, N_bt) bT slab never enters
        the differentiated graph and rabbit's covariance jacobian does not OOM.

        At the evaluation point (d = 0) the value is exact, the 1st derivative is
        J, and the 2nd derivative is K. ``use_curvature=False`` keeps only J
        (Gauss-Newton / Fisher, exact for Asimov data, 8 vs 72 fold passes).
        """
        val = self._ratio_from_param(param)
        J = tf.stop_gradient(self._ratio_compact_jac(param))  # (N_reco, nparam)
        d = param - tf.stop_gradient(param)  # value 0, unit gradient
        d = tf.cast(d, J.dtype)
        out = tf.stop_gradient(val) + tf.linalg.matvec(J, d)
        if use_curvature:
            K = tf.stop_gradient(self._ratio_compact_hess(param))  # (N_reco, n, n)
            out = out + 0.5 * tf.einsum("rij,i,j->r", K, d, d)
        return out

    def compute(self, param, full=False):
        """Return per-(bin, proc) scaling tensor.

        Shape: (N_reco, N_proc). Signal-proc column carries the per-reco-bin
        ratio σ_reco(λ; b) / σ_reco(λ_central; b); other columns are 1.

        Positivity floor: the bT integral has no hard wall against pathological λ
        (e.g. λ4 < 0 with the bounded-tanh models), which, via the b*-saturated,
        hugely enhanced large-b region, can make σ_reco, and thus the predicted
        signal yield, NEGATIVE. That gives a NaN Poisson NLL and a flat gradient
        (tanh saturates), trapping the minimizer. We soft-floor the ratio to a
        small positive value (RATIO_FLOOR_SCALE / RATIO_FLOOR_MIN) so a bad point
        is a large-but-finite penalty with a usable gradient, NOT a crash. The
        scale is far below any physical response, so the validated central and
        λ-variation closures are unchanged (ratio == 1 at λ_central, to fp). This
        is a numerical safety net, not a physics constraint: it does not stop the
        fit from exploring negative-λ, it keeps that exploration finite.
        """
        # ratio(λ) per reco bin. Normal path = exact fold (used for the fit).
        # Straight-through path (Hessian-only Phase B) keeps the bT slab off the
        # autodiff graph so rabbit's covariance jacobian doesn't OOM. Toggled by the
        # hessian_straightthrough=1 spec token; hessian_gn=1 drops the curvature term
        # (Gauss-Newton/Fisher — exact for Asimov, 8 vs 72 passes). Resolved at
        # construction (self._hess_st / self._hess_gn). Do NOT enable during the fit:
        # it recomputes J(/K) every call.
        if self._hess_st:
            ratio = self._ratio_straightthrough(param, use_curvature=not self._hess_gn)
        else:
            ratio = self._ratio_from_param(param)

        # Build (N_reco, N_proc) scaling: 1 everywhere except the signal column,
        # which carries the per-bin ratio. Broadcasting the precomputed (1, N_proc)
        # one-hot avoids materializing a ones tensor / rebuilding one_hot.
        ratio_col = tf.cast(
            tf.reshape(ratio, [-1, 1]), self.indata.dtype
        )  # (N_reco, 1)
        rnorm = 1.0 + (ratio_col - 1.0) * self._signal_col_mask

        return rnorm
