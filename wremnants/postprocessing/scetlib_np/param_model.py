"""SCETlibNPParamModel — continuous-λ rabbit ParamModel for SCETlib NP.

This ParamModel scales the signal reco template by a per-bin ratio of the
SCETlib nonperturbative (NP) prediction at the fitted λ vs. at λ_central. The
prediction is built in THREE STEPS, written out in order below — read top to
bottom for the full maths. This module docstring is the SINGLE SOURCE OF TRUTH:
:mod:`response_matrix`, :func:`btgrid_tf.reconstruct_batch_tf`, the numpy
reference :mod:`btgrid_numpy`, ``sigma_reco_central.md``, and the validation
scripts all point here rather than restate it.

Pipeline at a glance (everything is a function of the NP parameters λ):

    Step 1   btgrid Hankel + Q integral  →  σ_resum(λ; g)   resummed, on the gen grid
    Step 2   + fixed-order matching      →  σ_gen(λ; g)  = σ_resum(λ; g) + σ_ns(g)
    Step 3   fold through response R      →  σ_reco(λ; b)    gen → reco
    Step 4   ratio vs λ_central          →  rnorm(b, proc)  the shape handed to rabbit

Steps 1–3 build the absolute physical cross section; Step 4 alone produces the
per-bin variation the fit consumes — they are kept separate on purpose.

Indices: Q, Y, qT are the SCETlib btgrid axes (boson mass / rapidity / qT);
g = flattened gen bin (ptVGen, absYVGen); b = flattened reco bin (ptll, yll,
cosThetaStarll_quantile, phiStarll_quantile). λ splits into λ_eff (for F_eff)
and λ_ν (for γ_ν^NP) — the 8 differentiable parameters listed at the end.

═════════════════════════════════════════════════════════════════════════════
Step 1 — σ_resum(λ; g): the resummed prediction from the bT grid
═════════════════════════════════════════════════════════════════════════════

(1a) Per-(Q, Y, qT) bT-space (Hankel) integral — the NP parameters enter HERE:

    σ(Q, Y, qT; λ) = qT · ∫ dbT  bT · J₀(qT·bT)
                          · I_pert(Q, Y, qT;  b*(bT))
                          · exp[ C_ν(Q, Y, qT;  bT) · γ_ν^NP(b*(bT); λ_ν) ]
                          · F_eff(Y;  b*(bT); λ_eff)

Where each factor lives — bare bT vs the b*-frozen b̄T:

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
              b* (per the SCETlib convention; see btgrid_numpy NP-model notes).
              Because it varies with bT it sits inside the bT integral (not a
              constant out front); because it varies with Q the Q integral
              cannot be collapsed ahead of the fit — the λ-dependence does not
              factor through ∫dQ (exp is nonlinear in C_ν·γ_ν^NP).
  γ_ν^NP      CS-side NP rapidity anomalous dimension; depends on b*(bT) and
              λ_ν only (no Q/Y/qT). λ-dependent.
  F_eff       TMD-effective NP factor; depends on Y² and b*(bT) and λ_eff only
              (no Q/qT). λ-dependent.

Only γ_ν^NP and F_eff carry λ; everything else (bT, J₀, I_pert, C_ν, b*) is
λ-independent and precomputed once — the bT·J₀ kernel, the bT Simpson weights,
and the arctan_Q² Q-integration weights are all built at construction.

(1b) Integrate over Q, then rebin onto the gen grid:

    σ_resum(λ; g) = rebin_{qT→ptVGen, |Y|→absYVGen} [ ∫_{Q_lo}^{Q_hi} dQ  σ(Q, Y, qT; λ) ]

The Q integral uses an arctan_Q² Simpson rule (the x = arctan((Q²−q0²)/(q0·Γ))
transform flattens the Breit-Wigner Z peak). The result (Y, qT) is then rebinned
(Simpson) onto the unfolding hist's gen edges: qT → ptVGen, and the signed btgrid
Y axis folded into |Y| → absYVGen. The |Y| fold is valid because NP is
Y-symmetric (F_eff depends on Y², γ_ν^NP doesn't depend on Y at all).

═════════════════════════════════════════════════════════════════════════════
Step 2 — σ_gen(λ; g): add the NP-independent fixed-order nonsingular
═════════════════════════════════════════════════════════════════════════════

    σ_gen(λ; g) = σ_resum(λ; g) + σ_ns(g)

    σ_ns(g)     = rebin_{qT→ptVGen, |Y|→absYVGen} [ σ_DYTurbo^FO − σ_SCETlib-sing^FO ]

The fixed-order matching adds the nonsingular piece σ_ns = (DYTurbo fixed order)
− (SCETlib singular fixed order), read straight from the original FO inputs
(the ``…_nnlo_sing…combined.pkl`` and the DYTurbo ``results_…scetlibmatch.txt``),
Q-windowed to [Q_lo, Q_hi], |Y|-folded, zeroed below qt_cutoff, and summed onto
the (ptVGen, absYVGen) gen bins (see :func:`compute_nonsingular_gen`).

σ_ns is NP-INDEPENDENT (the same for every λ) and is added at GEN level, so it
folds through the same response R as σ_resum in Step 3. Because the fit uses a
ratio (Step 3), σ_ns correctly DILUTES the NP variation where the FO dominates
(high qT). Set include_nonsingular=False for a resum-only model (σ_ns = 0).

═════════════════════════════════════════════════════════════════════════════
Step 3 — σ_reco(λ; b): fold gen → reco through the response matrix
═════════════════════════════════════════════════════════════════════════════

    P(b | g)     = R_raw(b, g) / N_gen(g)               (efficiency × migration)
    σ_reco(λ; b) = Σ_g  P(b | g) · σ_gen(λ; g)

where, in these expressions,
    g = the flattened GEN bin  (ptVGen, absYVGen)   — the grid σ_gen from Steps
        1–2 lives on (boson qT and |Y|), summed over by Σ_g;
    b = the flattened RECO bin (ptll, yll, cosThetaStarll_quantile,
        phiStarll_quantile) — the measured dilepton observables σ_reco lives on.
P(b | g) is the gen→reco mapping (one reco column per gen bin), so σ_reco(λ; b)
is just σ_gen pushed through the detector. The Σ_g is the matvec
``tf.linalg.matvec(self.R, σ_gen_flat)``. This step is pure detector folding —
no λ_central and no ratio enter here; that is Step 4.

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

Why N_gen and not the reco-passing marginal Σ_b R_raw(b, g): R is
post-reco-selection so that marginal already carries efficiency, and dividing
by it cancels efficiency (migration-only) — closes far worse, since efficiency
is strongly gen-dependent (ε ≈ 0.07–0.54 across gen bins on the current file).
N_gen(g) is the true generated total, so P = R_raw/N_gen is the
theory-independent gen→reco map. Pre-FSR: σ_gen, R, and N_gen must all sit at
the same QCD/boson gen level (the postfsr variants in the file close ~1% worse
— FSR mismatch).

═════════════════════════════════════════════════════════════════════════════
Step 4 — rnorm(b, proc): the per-reco-bin variation handed to rabbit
═════════════════════════════════════════════════════════════════════════════

    ratio(b)       = σ_reco(λ; b) / σ_reco(λ_central; b)
    rnorm(b, proc) = 1 + (ratio(b) − 1) · [proc is signal]   (1 in every other proc)

This is the only object that leaves the model: compute() returns rnorm(b, proc),
and rabbit multiplies the signal process's reco template (reco bin b) by it,
leaving every other process at 1. Dividing by σ_reco(λ_central) cancels the
event-count↔cross-section scale and any overall normalization, so rnorm carries
purely the SHAPE of the NP variation per reco bin — Steps 1–3 build the absolute
σ_reco(λ; b), and Step 4 reduces it to the bin-by-bin template scaling the fit
needs. σ_reco(λ_central) is precomputed once at construction as the denominator.
(If the gen-total hist is absent, σ_gen(λ_c) is used as a proxy for N_gen, but
then σ_gen cancels in σ_reco(λ_c) = R_raw·1 and the λ_central closure can't test
the integral.)

─────────────────────────────────────────────────────────────────────────────
Parameters and inputs
─────────────────────────────────────────────────────────────────────────────

The 8 v1 parameters λ (all factorisable through the current btgrid):

    γ_ν^NP (CS-side):       lambda2_nu, lambda4_nu, lambda_inf_nu
    F_eff  (TMD-effective): lambda2, lambda4, lambda6, delta_lambda2, lambda_inf

λ_central is read from the fit-tensor's meta_info_input via the upstream
SCETlib correction pkl (see :mod:`scetlib_lambda_central`). The np_model and
np_model_nu strings are fixed at construction (from λ_central). All λ values
are TF Variables — differentiable in the fit.

═════════════════════════════════════════════════════════════════════════════
Getting the postfit Hessian / covariance (uncertainties on λ)
═════════════════════════════════════════════════════════════════════════════

The fit floats λ fine, but rabbit's postfit covariance step
``loss_val_grad_hess`` → ``t2.jacobian(grad, x)`` differentiates through the bT
fold once per fit parameter (~3754). pfor cannot see that the fold depends on
the parameters only through the ≤8 λ, so it re-materializes the internal
(Ng × Nbt) ≈ 8.75 GB bT slab for EVERY parameter → ~33 TB → OOM.

Fix (this module): a "straight-through" surrogate that keeps the exact ratio
VALUE but exposes only a compact quadratic in the ≤8 λ to autodiff:

    ratio~(λ) = stop_gradient(ratio) + J·d + ½ dᵀ K d ,   d = λ − stop_gradient(λ)

with J = dratio/dλ ([Nreco, nλ]) and K = d²ratio/dλ² ([Nreco, nλ, nλ]) computed
by forward-mode AD (≤8 / ≤64 fold passes, NOT tiled). d is identically 0 so the
value is unchanged, but ∂d/∂λ = I, so ∂ratio~/∂λ = J and ∂²ratio~/∂λ² = K:
rabbit's jacobian gets the exact derivatives while the big slab stays inside
stop_gradient and never enters the differentiated graph (33 TB → a few MB).
Implemented in ``_ratio_straightthrough`` (+ ``_ratio_compact_jac``,
``_ratio_compact_hess``); selected in ``compute()`` by env flags. Full
derivation + validation in ``HESSIAN_PLAN.md``.

Two-pass recipe (rabbit still computes the Hessian; NO rabbit changes):

  1. Fit, no Hessian → postfit:
       rabbit_fit.py <FIT> --paramModel wremnants.postprocessing.scetlib_np.SCETlibNPParamModel <UNFOLD> <BTGRID> \
           --noHessian -o fit/

  2. Covariance at that postfit (no refit), straight-through ON:
       SCETLIB_NP_HESSIAN_STRAIGHTTHROUGH=1 SCETLIB_NP_HESSIAN_GN=1 \
       rabbit_fit.py <FIT> --paramModel wremnants.postprocessing.scetlib_np.SCETlibNPParamModel <UNFOLD> <BTGRID> \
           --externalPostfit fit/fitresults.hdf5 --externalPostfitResult nominal \
           --noFit -t 0 --pseudoData nominal -o cov/
     (do NOT pass --noHessian; do NOT pass --eager.) ~5 min, no OOM.
     Uncertainties are sqrt(diag(cov)) in cov/fitresults.hdf5 (results_nominal).

Env flags (both OFF by default → the fit path is unchanged):
  SCETLIB_NP_HESSIAN_STRAIGHTTHROUGH=1  use the straight-through path in compute()
  SCETLIB_NP_HESSIAN_GN=1               Gauss-Newton: keep J only, drop the K term

GN vs full-K. The Poisson Hessian is
    H_ij = Σ_b [ (n_b/μ_b²) J_bi J_bj  +  (1 − n_b/μ_b) K_bij ].
For ASIMOV data the residual (1 − n/μ) = 0, so the K term vanishes and GN (J
only) is EXACT. For real/toy data K is needed for the exact Hessian — but the
nested-forward-mode K (``_ratio_compact_hess``) currently CRASHES under rabbit's
@tf.function (the tf.where in ``_safe_div``) and is impractically slow under
--eager, so GN (the recipe above) is the working production path today. See
HESSIAN_PLAN.md §9 for the open K item and its fixes.

WARNING: keep SCETLIB_NP_HESSIAN_STRAIGHTTHROUGH UNSET during the FIT. The
surrogate recomputes J(/K) on every compute() call — fine for the one-shot
covariance pass, but it would cripple the minimizer (many gradient/HVP evals).
"""

import json
import os
import re
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
from wremnants.utilities import common as wrem_common

# Default fixed-order inputs for the NP-independent nonsingular term
# σ_ns = DYTurbo − SCETlib_singular, resolved relative to this package via
# wrem_common.data_dir (= <WRemnants>/wremnants-data/data). These are the CT18Z
# N3+0LL fixed-order pieces; both are NP-independent (same for any λ tune).
_NONSING_FO_SING_DEFAULT = os.path.join(
    wrem_common.data_dir,
    "TheoryCorrections",
    "inclusive_Z_COM13_CT18Z_N3+0LL_lattice_lambda4bugfix_fine_nnlo_sing_combined.pkl",
)
_NONSING_DYTURBO_DEFAULT = os.path.join(
    wrem_common.data_dir,
    "TheoryCorrections",
    "results_z-2d-nnlo-vj-CT18ZNNLO-{scale}-scetlibmatch.txt",
)

# Default fit inputs, so collaborators can run
# ``--paramModel wremnants.postprocessing.scetlib_np.SCETlibNPParamModel`` with no
# extra positional args. Passing the two positional args explicitly still wins.
#
# The unfolding response lives in wremnants-data (a skim of only the two hists
# load_R reads — ``nominal_prefsr_yieldsUnfolding`` + ``prefsr`` — 6.8 GB → 211 MB;
# made with scripts/inspect/open_narf_h5py.py). Resolved via wrem_common.data_dir.
_UNFOLDING_HDF5_DEFAULT = os.path.join(
    wrem_common.data_dir,
    "TheoryCorrections",
    "scetlib_np",
    "mz_dilepton_unfolding_R_skim.hdf5",
)
# The bT-grid combined pickle is ~17.5 GB (two [Nbins×Nbt] float64 slabs: I_pert
# and C_nu) — too large for wremnants-data/git-LFS — so it lives on the shared
# data area next to NanoAOD, as a sibling of NanoAOD under the data base. We
# resolve the NanoAOD base by REPLICATING dataset_tools.getDataPath()'s hostname
# logic rather than importing it: that module does ``import ROOT`` / ``import narf``
# at module scope (and wremnants.production.__init__ runs ROOT.gInterpreter +
# narf.clingutils.Load), which segfaults when imported mid-fit (after TF is up).
# Only the subMIT copy exists today — pass btgrid_dir explicitly at other sites.
_BTGRID_SUBDIR = ("scetlib_np", "Z_COM13_CT18Z_N3p0LL_btgrid_fineall")


def _nano_data_base():
    """NanoAOD base dir per host — mirrors dataset_tools.getDataPath() WITHOUT
    importing it (that import pulls ROOT/narf; see note above). Falls back to the
    subMIT path on unknown hosts (the only site with the grid today)."""
    import socket

    hostname = socket.gethostname()
    if hostname.endswith(".cern.ch"):
        return "/scratch/shared/NanoAOD"
    if hostname == "cmsanalysis.pi.infn.it":
        return "/scratchnvme/wmass/NANOV9/postVFP"
    if hostname == "cmsasymow.pi.infn.it":
        return "/scratch/wmass/y2016"
    # .mit.edu and any unknown host -> subMIT shared scratch
    return "/scratch/submit/cms/wmass/NanoAOD"


def _default_btgrid_dir():
    return os.path.join(os.path.dirname(_nano_data_base()), *_BTGRID_SUBDIR)


# Ordered list of the v1 continuous λ. CS-side first, then TMD-effective.
GNU_PARAMS = ("lambda2_nu", "lambda4_nu", "lambda_inf_nu")
EFF_PARAMS = ("lambda2", "lambda4", "lambda6", "delta_lambda2", "lambda_inf")
ALL_PARAMS = GNU_PARAMS + EFF_PARAMS

# Positivity floor for the per-bin reco ratio in compute() (see its docstring).
# A pathological λ (e.g. λ4 < 0 with the bounded-tanh model) can drive σ_reco —
# and hence the predicted signal yield — negative, which makes the Poisson NLL
# NaN and stalls the minimizer. We soft-floor the ratio to a small positive value
# so a bad point becomes a LARGE-BUT-FINITE penalty the fit can back off from,
# with a non-zero gradient through the transition (softplus, not a hard clamp).
#   RATIO_FLOOR_SCALE — softplus transition width. Chosen FAR below any physical
#     response so healthy ratios (~0.9–1.1, and every validated λ-variation) pass
#     through to machine precision: scale·softplus(r/scale) == r for r ≫ scale.
#   RATIO_FLOOR_MIN — hard positive ground: softplus underflows to exactly 0 for
#     the extreme (r ~ -1e43) case, so this keeps the yield strictly > 0 (no NaN).
RATIO_FLOOR_SCALE = 1.0e-4
RATIO_FLOOR_MIN = 1.0e-9


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


def _load_lambda_central_file(path):
    """Load a λ_central override from a JSON or YAML file.

    The file must decode to a dict with ``eff_params`` and ``gnu_params``
    sub-dicts (same shape as :func:`scetlib_lambda_central.read_lambda_central`).
    Format is chosen by extension (``.yaml``/``.yml`` → YAML, else JSON);
    YAML's loader also accepts JSON, so this is forgiving either way.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SCETLIB_NP_LAMBDA_CENTRAL_FILE points to a missing file: {path!r}"
        )
    with open(path) as f:
        text = f.read()
    if path.lower().endswith((".yaml", ".yml")):
        import yaml

        data = yaml.safe_load(text)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"SCETLIB_NP_LAMBDA_CENTRAL_FILE {path!r} is not valid JSON; got {exc}"
            ) from exc
    if (
        not isinstance(data, dict)
        or "eff_params" not in data
        or "gnu_params" not in data
    ):
        raise ValueError(
            f"SCETLIB_NP_LAMBDA_CENTRAL_FILE {path!r} must decode to a dict with "
            f"'eff_params' and 'gnu_params' keys; got {type(data).__name__} "
            f"with keys {list(data) if isinstance(data, dict) else '<n/a>'}."
        )
    return data


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


def _bin_sum_matrix(src_centers, target_edges, tol=1e-6):
    """(N_target, N_src) 0/1 matrix that SUMS bin-integrated source bins whose
    centre falls in each target bin. Source bins outside all target bins are
    dropped — a natural truncation to the target range (qT>ptVGen_max, |Y|>absY_max)."""
    src = np.asarray(src_centers, dtype=np.float64)
    edges = np.asarray(target_edges, dtype=np.float64)
    W = np.zeros((edges.size - 1, src.size), dtype=np.float64)
    for i in range(edges.size - 1):
        m = (src >= edges[i] - tol) & (src <= edges[i + 1] + tol)
        W[i, m] = 1.0
    return W


def compute_nonsingular_gen(
    fo_sing_path,
    dyturbo_path,
    gen_axes_meta,
    charge=0,
    q_lo=60.0,
    q_hi=120.0,
    qt_cutoff=1.0,
    dyturbo_axes=("Q", "Y", "qT"),
):
    """Nonsingular FO term on the model gen grid (NptVGen, NabsYVGen).

    The fixed-order/DYTurbo matching adds a NP-INDEPENDENT piece to σ_gen:
        σ_gen^matched(λ) = σ_gen^resum(λ) + σ_ns ,
    where the nonsingular is read straight from the original fixed-order inputs:
        σ_ns = (DYTurbo fixed order) − (SCETlib singular fixed order)
    — exactly the ``-hfo_sing + hfo`` that ``read_matched_scetlib_hist`` forms.
    ``fo_sing_path`` is the SCETlib singular ``…_nnlo_sing…combined.pkl``;
    ``dyturbo_path`` is the DYTurbo fixed-order ``results_…scetlibmatch.txt``
    (use ``{scale}`` → mur1-muf1 for the central). The nonsingular is zeroed below
    ``qt_cutoff`` (as make_theory_corr does), Q-windowed to [q_lo, q_hi], |Y|-folded,
    and projected onto the coarse (ptVGen, absYVGen) gen bins by SUMMING the
    bin-integrated native bins.
    """
    from wremnants.utilities.io_tools import input_tools
    from wums import boostHistHelpers as hh

    def _central(h):
        if "vars" in h.axes.name:
            names = list(h.axes["vars"])
            idx = 0
            for c in ("central", "pdf0", "nominal"):
                if c in names:
                    idx = names.index(c)
                    break
            h = h[{"vars": idx}]
        return h

    # SCETlib singular fixed order, and DYTurbo fixed order, from their own files.
    dyturbo_path = (
        dyturbo_path.format(scale="mur1-muf1")
        if "{scale}" in dyturbo_path
        else dyturbo_path
    )
    hfo_sing = _central(input_tools.read_scetlib_hist(fo_sing_path, charge=charge))
    hfo = input_tools.read_dyturbo_hist(
        [dyturbo_path], axes=list(dyturbo_axes), charge=charge
    )
    if "vars" in hfo.axes.name:
        hfo = _central(hfo)

    # Align shared physics axes (DYTurbo is coarser), then σ_ns = DYTurbo − singular.
    for ax in ("Y", "Q", "qT"):
        if ax in set(hfo.axes.name) & set(hfo_sing.axes.name):
            hfo, hfo_sing = hh.rebinHistsToCommon([hfo, hfo_sing], ax)
    nonsing_h = hh.addHists(-1.0 * hfo_sing, hfo, flow=False, by_ax_name=False)

    if "charge" in nonsing_h.axes.name:
        nonsing_h = nonsing_h[{"charge": sum}]
    # Q-window: slice(...,sum) sums ONLY the in-range Q bins (no underflow leak).
    Qe = np.asarray(nonsing_h.axes["Q"].edges, dtype=np.float64)
    qi = int(np.argmin(np.abs(Qe - q_lo)))
    qj = int(np.argmin(np.abs(Qe - q_hi)))
    nonsing_h = nonsing_h[{"Q": slice(qi, qj, sum)}]
    nonsing_h = hh.makeAbsHist(nonsing_h, "Y")  # signed Y -> |Y|

    qT_c = np.asarray(nonsing_h.axes["qT"].centers, dtype=np.float64)
    absY_c = np.asarray(nonsing_h.axes["absY"].centers, dtype=np.float64)
    v = nonsing_h.project("qT", "absY").values(flow=False)  # (qT, absY)
    v[qT_c < qt_cutoff, :] = 0.0  # zero the nonsingular below the cutoff

    ptV_edges = np.asarray(gen_axes_meta[0][1], dtype=np.float64)
    absY_edges = np.asarray(gen_axes_meta[1][1], dtype=np.float64)
    Wp = _bin_sum_matrix(qT_c, ptV_edges)  # (NptVGen, NqT)
    Wa = _bin_sum_matrix(absY_c, absY_edges)  # (NabsYVGen, NabsYsrc)
    return Wp @ v @ Wa.T  # (NptVGen, NabsYVGen)


class SCETlibNPParamModel(ParamModel):

    def __init__(
        self,
        indata,
        unfolding_hdf5_path: Optional[str] = None,
        btgrid_dir: Optional[str] = None,
        lambda_central: Optional[Mapping] = None,
        signal_proc: str = "Zmumu",
        Q_lo: float = 60.0,
        Q_hi: float = 120.0,
        poi_params: Optional[tuple] = (),
        prior_sigmas: Optional[Mapping] = None,
        include_nonsingular: bool = True,
        nonsingular_fo_sing: str = _NONSING_FO_SING_DEFAULT,
        nonsingular_dyturbo: str = _NONSING_DYTURBO_DEFAULT,
        nonsingular_qt_cutoff: float = 1.0,
        **kwargs,
    ):
        """Construct the ParamModel.

        Parameters
        ----------
        indata
            rabbit's input-data structure (passed by ``ph.load_models``).
        unfolding_hdf5_path
            Path to the upstream histmaker output containing
            ``nominal_prefsr_yieldsUnfolding`` for R (and the ``prefsr`` xnorm
            hist for N_gen) — see :mod:`response_matrix` for the defaults.
            Defaults (when None) to the skim shipped in wremnants-data
            (``_UNFOLDING_HDF5_DEFAULT``); pass explicitly to override.
        btgrid_dir
            Directory holding the SCETlib bT-grid ``combined_btgrid.pkl``.
            Defaults (when None) to the shared data-area copy next to NanoAOD
            (``_default_btgrid_dir()``, which mirrors
            ``dataset_tools.getDataPath()``'s per-host logic without importing it —
            that import pulls ROOT/narf and segfaults mid-fit); pass explicitly at
            non-subMIT sites.
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
        # Resolve default inputs so collaborators can run with no extra
        # --paramModel positional args (explicit args still override). The
        # unfolding skim ships in wremnants-data; the big bT-grid lives on the
        # shared data area next to NanoAOD (see _default_btgrid_dir).
        if unfolding_hdf5_path is None:
            unfolding_hdf5_path = _UNFOLDING_HDF5_DEFAULT
        if btgrid_dir is None:
            btgrid_dir = _default_btgrid_dir()

        # If the histmaker baked discrete NP κ-template variations into the
        # input HDF5, those systs describe the same physics as our continuous
        # λ POUs. Running both → double-counting (the discrete syst absorbs
        # whatever shape variation our ParamModel should describe). Warn
        # loudly if any such systs are present and unfrozen.
        self._check_discrete_np_double_counting(kwargs.get("freezeParameters"))

        # ---- λ_central
        # Three sources of λ_central, in priority order:
        #   1. ``lambda_central`` constructor arg (explicit dict).
        #   2. ``SCETLIB_NP_LAMBDA_CENTRAL_FILE`` env var — path to a JSON or
        #      YAML file with ``eff_params`` and ``gnu_params``. Overrides the
        #      metadata auto-detect; useful when the upstream SCETlib pkl isn't
        #      accessible (e.g. a colleague's input).
        #   3. Auto-detect from the fit hdf5's theoryCorr → upstream pkl.
        env_lc_file = os.environ.get("SCETLIB_NP_LAMBDA_CENTRAL_FILE", "").strip()
        if lambda_central is None and env_lc_file:
            lambda_central = _load_lambda_central_file(env_lc_file)
            print(
                f"[SCETlibNPParamModel] λ_central from file {env_lc_file!r}",
                flush=True,
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
            print(
                f"[SCETlibNPParamModel] λ_central auto-detected from indata.metadata",
                flush=True,
            )
        print(f"[SCETlibNPParamModel] λ_central:", flush=True)
        for key, value in lambda_central.items():
            print(f"  {key} = {value!r}", flush=True)

        self.eff_central = dict(lambda_central["eff_params"])
        self.gnu_central = dict(lambda_central["gnu_params"])
        self.np_model = self.eff_central["np_model"]
        self.np_model_nu = self.gnu_central["np_model_nu"]

        # ---- btgrid + dense layout
        grid = btgrid_cache.load(btgrid_dir)
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
        # Raw response counts; normalized to a response below.
        self._R_raw = tf.constant(R_arr.reshape(N_reco, N_gen), dtype=fz_tf.DTYPE)
        # Gen-total denominator N_gen(g) from the xnorm hist ("prefsr"): the
        # generated fiducial yield per gen bin (pre-reco-selection). Dividing R
        # by this gives the theory-independent efficiency×migration response.
        # Falls back to the σ_gen(λ_c) proxy if the gen-total isn't in the file.
        if R_info.get("N_gen") is not None:
            self._N_gen_flat = tf.constant(
                R_info["N_gen"].reshape(-1), dtype=fz_tf.DTYPE
            )
        else:
            self._N_gen_flat = None
            print(
                "[SCETlibNPParamModel] WARNING: no gen-total hist in unfolding "
                "output; falling back to σ_gen(λ_c) as the response normalizer "
                "(circular closure — see param_model docstring).",
                flush=True,
            )
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

        # qT rebin: btgrid qT (signed nonneg, NqT=141) → ptVGen edges. When
        # load_R was built with ptVGen_overflow=True (default), ptVGen_edges ends
        # in the overflow bin [last_gen_edge, PTVGEN_OVERFLOW_EDGE] (e.g. [44, 100]),
        # so rebin_weights' last row Simpson-integrates the btgrid tail qT∈(44,100]
        # into that overflow gen bin — matching R's gen-overflow column (true qT>44
        # migrating into the high-ptll reco bins). btgrid qT past the last edge
        # (>100, beyond the grid) is dropped; negligible. Without the overflow
        # column ptVGen_edges ends at 44 and that tail is simply truncated.
        self.W_ptVGen = tf.constant(
            fz_int.rebin_weights(self.qT_unique, ptVGen_edges, name="ptVGen"),
            dtype=fz_tf.DTYPE,
        )

        # ---- Normalize the response, then cache σ_reco(λ_central).
        # A response matrix must encode only the gen→reco *mapping*, not the
        # MC's absolute gen spectrum. Normalize each gen column by the gen-total
        # N_gen(g) (the xnorm "prefsr" hist — generated fiducial yield before
        # reco selection) → P(b|g) = eff×migration (theory-independent):
        #     P(b|g)        = R_raw(b,g) / N_gen(g)
        #     σ_reco(λ;b)   = Σ_g P(b|g) · σ_gen(λ;g)
        #     σ_reco(λ_c;b) = Σ_g P(b|g) · σ_gen(λ_c;g)
        # NB the reco-passing marginal Σ_b R_raw(b,g) is the WRONG normalizer:
        # it already includes efficiency (R is post-reco-selection), so dividing
        # by it cancels efficiency (migration-only) — closes far worse, ε is not
        # flat in gen bin. We use the true gen-total N_gen instead. Because
        # σ_reco_central then depends on σ_gen(λ_c) (it does NOT collapse to
        # R_raw·1), the λ_central closure is a genuine test of the integral.
        # Fallback: if the gen-total hist is absent, use σ_gen(λ_c) as a proxy
        # for N_gen (∝ σ_gen^MC) — keeps efficiency but makes the closure
        # circular (σ_gen cancels to R_raw·1).
        # Native-binning Q-integrated reconstruction (NY, NqT) on the signed-Y /
        # qT grid, BEFORE the |Y|-fold and qT-rebin — exposed so the native-binning
        # validation can compare it to the SCETlib reference / numpy factorize
        # without the projection layer.
        self.sigma_YqT_central = self._sigma_YqT_native_at(
            self.eff_central, self.gnu_central
        )
        # ---- Optional fixed-order/DYTurbo nonsingular term (NP-independent).
        # σ_gen^matched(λ) = σ_gen^resum(λ) + σ_ns. Added at GEN level, so it folds
        # through the same response R as the resummed piece. Because rnorm is a
        # ratio, this correctly DILUTES the NP variation where the FO dominates
        # (high qT). σ_ns is a constant (no λ dependence).
        self.include_nonsingular = bool(include_nonsingular)
        if self.include_nonsingular:
            _dy0 = (
                nonsingular_dyturbo.format(scale="mur1-muf1")
                if (nonsingular_dyturbo and "{scale}" in nonsingular_dyturbo)
                else nonsingular_dyturbo
            )
            missing = [
                p for p in (nonsingular_fo_sing, _dy0) if not (p and os.path.exists(p))
            ]
            if missing:
                raise FileNotFoundError(
                    "include_nonsingular=True needs the fixed-order inputs for "
                    "σ_ns = DYTurbo − SCETlib_singular, but these are missing:\n  "
                    + "\n  ".join(missing)
                    + "\nThey live under wremnants-data/data/TheoryCorrections (the "
                    "SCETlib singular …_nnlo_sing…combined.pkl and the DYTurbo "
                    "results_…scetlibmatch.txt). Pass nonsingular_fo_sing / "
                    "nonsingular_dyturbo, or set include_nonsingular=False for resum-only."
                )
            sigma_ns_np = compute_nonsingular_gen(
                nonsingular_fo_sing,
                nonsingular_dyturbo,
                self._gen_axes_meta,
                q_lo=Q_lo,
                q_hi=Q_hi,
                qt_cutoff=nonsingular_qt_cutoff,
            )
            if sigma_ns_np.shape != tuple(self.gen_shape):
                raise ValueError(
                    f"nonsingular gen shape {sigma_ns_np.shape} != model gen shape "
                    f"{tuple(self.gen_shape)}"
                )
            self.sigma_ns = tf.constant(sigma_ns_np, dtype=fz_tf.DTYPE)
        else:
            self.sigma_ns = tf.zeros(self.gen_shape, dtype=fz_tf.DTYPE)
        # Reuse the native (NY, NqT) integral already computed above for
        # sigma_YqT_central — no need to run the bT reconstruction at λ_central twice.
        sigma_gen_central = self._sigma_gen_at(
            self.eff_central, self.gnu_central, sigma_YqT=self.sigma_YqT_central
        )
        # The pure gen-level integral (NptVGen, NabsYVGen), BEFORE folding through
        # the response — used by the gen-level validation to test the integral
        # in isolation (no R).
        self.sigma_gen_central = sigma_gen_central
        gen_flat = tf.reshape(sigma_gen_central, [-1])
        if tf.reduce_any(gen_flat <= 0).numpy():
            n_bad = int(tf.reduce_sum(tf.cast(gen_flat <= 0, tf.int32)))
            raise ValueError(
                f"SCETlibNPParamModel: {n_bad} gen bins have non-positive "
                f"σ_gen(λ_central); cannot normalize / fold the response."
            )
        N_gen = self._N_gen_flat if self._N_gen_flat is not None else gen_flat
        # Guard empty gen bins (no generated events): leave column at 0.
        safe_N_gen = tf.where(N_gen > 0, N_gen, tf.ones_like(N_gen))
        self.R = self._R_raw / safe_N_gen[tf.newaxis, :]  # P(b|g) = R_raw / N_gen
        # Free the raw counts: only the normalized response self.R is used from
        # here on (compute() never touches _R_raw) — no need to hold both.
        del self._R_raw
        self.sigma_reco_central = tf.linalg.matvec(self.R, gen_flat)  # Σ_g P·σ_gen(λ_c)
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
        # Precompute the (1, N_proc) one-hot selecting the signal column, reused
        # in every compute() to place the per-bin ratio (others stay at 1).
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

        # Defaults: λ_central values per parameter. Optionally overridden by
        # the ``SCETLIB_NP_XPARAMDEFAULT`` env var — comma-separated
        # ``name=value`` pairs (for closure tests where the data-generating
        # / fit-start point should differ from the card's λ_central).
        central_lookup = {**self.eff_central, **self.gnu_central}
        defaults = np.array(
            [central_lookup[p] for p in self._param_order], dtype=np.float64
        )

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
                f"[SCETlibNPParamModel] xparamdefault overridden: {dict(zip(self._param_order, defaults))}",
                flush=True,
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

    def _sigma_YqT_native_at(self, eff_params, gnu_params):
        """Reconstruct σ(λ) on the btgrid and Q-integrate, returning the result
        in the btgrid's *native* binning: shape (NY, NqT) on the signed-Y /
        qT grid (Y_unique, qT_unique), BEFORE the |Y|-fold and qT-rebin. This
        is the object that the native-binning validation compares against the
        SCETlib spectrum reference (curve 1) and the numpy `factorize` (curve 2)."""
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
        return fz_int.integrate_over_Q_tf(sigma_dense, self.Q_weights)

    def _sigma_gen_at(self, eff_params, gnu_params, sigma_YqT=None):
        """Evaluate σ_gen(λ) on R's gen binning. Returns shape (NptVGen, NabsYVGen).

        ``sigma_YqT`` lets a caller pass an already-computed native (NY, NqT)
        integral to skip the (expensive) bT reconstruction — used at construction
        to reuse ``self.sigma_YqT_central`` instead of integrating λ_central twice.
        """
        if sigma_YqT is None:
            sigma_YqT = self._sigma_YqT_native_at(eff_params, gnu_params)
        # 4. Rebin Y (signed) → absYVGen (|Y|-folded): (NabsYVGen, NqT).
        sigma_absY_qT = fz_int.rebin_axis_tf(sigma_YqT, axis=0, weights=self.W_absY)
        # 5. Rebin qT → ptVGen: (NabsYVGen, NptVGen).
        sigma_absY_ptV = fz_int.rebin_axis_tf(
            sigma_absY_qT, axis=1, weights=self.W_ptVGen
        )
        # 6. Reorder to (NptVGen, NabsYVGen) to match R's gen axis order.
        sigma_resum = tf.transpose(sigma_absY_ptV, perm=[1, 0])
        # 7. Add the NP-independent fixed-order/DYTurbo nonsingular (zeros if off).
        return sigma_resum + self.sigma_ns

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

        The differentiable map that the straight-through Hessian path wraps. The
        soft positivity floor (see ``compute``) lives here so both the normal and
        straight-through paths apply it identically.
        """
        eff_params, gnu_params = self._unpack_params(param)
        sigma_gen = self._sigma_gen_at(eff_params, gnu_params)
        gen_flat = tf.reshape(sigma_gen, [-1])
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

        One JVP per parameter (nparam ≤ 8), each a single bT-fold pass — NOT
        tiled over params. This is the compact object the Hessian actually needs
        from the fold (see HESSIAN_PLAN.md §2)."""
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

        nparam² JVP-of-JVP passes (≤ 64), each one bT-fold pass — never tiled."""
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
        ≤8 λ (J, and optionally K) — so the (N_grid, N_bt) bT slab never enters
        the differentiated graph and rabbit's covariance jacobian does not OOM.

        At the evaluation point (d = 0) the value is exact, the 1st derivative is
        J, and the 2nd derivative is K. ``use_curvature=False`` keeps only J
        (Gauss-Newton / Fisher — exact for Asimov data, 8 vs 72 fold passes).
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
        (e.g. λ4 < 0 with the bounded-tanh models), which — via the b*-saturated,
        hugely enhanced large-b region — can make σ_reco, and thus the predicted
        signal yield, NEGATIVE. That would give a NaN Poisson NLL and a flat
        gradient (tanh saturates), trapping the minimizer. We soft-floor the ratio
        to a small positive value (RATIO_FLOOR_SCALE / RATIO_FLOOR_MIN) so a bad
        point is a large-but-finite penalty with a usable gradient, NOT a crash.
        The scale is far below any physical response, so the validated central and
        λ-variation closures are unchanged (ratio == 1 at λ_central, to fp). This
        is a numerical safety net, not a physics constraint — it does not stop the
        fit from exploring negative-λ; it just keeps that exploration finite.
        """
        # ratio(λ) per reco bin. Normal path = exact fold (used for the fit).
        # Straight-through path (Hessian-only Phase B) keeps the bT slab off the
        # autodiff graph so rabbit's covariance jacobian doesn't OOM. Toggled by
        # SCETLIB_NP_HESSIAN_STRAIGHTTHROUGH; SCETLIB_NP_HESSIAN_GN=1 drops the
        # curvature term (Gauss-Newton/Fisher — exact for Asimov, 8 vs 72 passes).
        # Do NOT enable during the fit: it recomputes J(/K) every call.
        if not hasattr(self, "_hess_st"):
            self._hess_st = bool(
                os.environ.get("SCETLIB_NP_HESSIAN_STRAIGHTTHROUGH", "").strip()
            )
        if self._hess_st:
            gn = bool(os.environ.get("SCETLIB_NP_HESSIAN_GN", "").strip())
            ratio = self._ratio_straightthrough(param, use_curvature=not gn)
        else:
            ratio = self._ratio_from_param(param)

        # Build (N_reco, N_proc) scaling: 1 everywhere except the signal column,
        # which carries the per-bin ratio. Broadcasting the precomputed (1, N_proc)
        # one-hot avoids materializing a separate ones tensor / rebuilding one_hot.
        ratio_col = tf.cast(
            tf.reshape(ratio, [-1, 1]), self.indata.dtype
        )  # (N_reco, 1)
        rnorm = 1.0 + (ratio_col - 1.0) * self._signal_col_mask

        return rnorm
