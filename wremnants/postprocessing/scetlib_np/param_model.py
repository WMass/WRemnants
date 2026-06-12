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

The default evaluation uses the memory-factorized layout (deduplicated
(I_pert, C_ν) rows + J₀ kernel on the unique-qT grid + Simpson-as-matmul),
which is numerically equivalent to the per-bin (Nbins, Nbt) layout (≲1e-14
rel., floating-point summation order only) but ~6× smaller — required to fit
a 32 GB GPU. See FACTORIZED_RECON.md in this directory; the legacy_recon=1
spec token restores the legacy layout.

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
(high qT). σ_ns is ALWAYS included (it is what the histmaker nominal carries);
for resum-only diagnostics subtract the exposed ``sigma_ns`` from
``sigma_gen_central`` (and re-fold with ``R`` for reco level).

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
       rabbit_fit.py <FIT> --paramModel wremnants.postprocessing.scetlib_np.SCETlibNPParamModel <UNFOLD> <BTGRID> \
           hessian_straightthrough=1 hessian_gn=1 \
           --externalPostfit fit/fitresults.hdf5 --externalPostfitResult nominal \
           --noFit -t 0 --pseudoData nominal -o cov/
     (do NOT pass --noHessian; do NOT pass --eager.) ~5 min, no OOM.
     Uncertainties are sqrt(diag(cov)) in cov/fitresults.hdf5 (results_nominal).

Switches (both OFF by default → the fit path is unchanged). They are spec
tokens inside the --paramModel spec (shown above) — the spec is stored in the
fitresults meta_info.args, so the configuration is recorded in the output
(env vars are NOT supported; all model knobs go through the spec):
  hessian_straightthrough=1  use the straight-through path in compute()
  hessian_gn=1               Gauss-Newton: keep J only, drop the K term
WARNING: hessian_straightthrough=1 WITHOUT hessian_gn=1 is full-K mode —
correct in principle (needed for real/toy data) but currently INFEASIBLE at
full grid scale (the 64 nested-FA passes unroll into one graph, ~TB peak →
OOM-kill). Until HESSIAN_PLAN.md §9 (precomputed chunked K) is implemented,
always pass hessian_gn=1 (exact for Asimov).

GN vs full-K. The Poisson Hessian is
    H_ij = Σ_b [ (n_b/μ_b²) J_bi J_bj  +  (1 − n_b/μ_b) K_bij ].
For ASIMOV data the residual (1 − n/μ) = 0, so the K term vanishes and GN (J
only) is EXACT — drop the K term with hessian_gn=1. For real/toy data
the K term is needed for the exact Hessian. The nested-forward-mode K
(``_ratio_compact_hess``) USED to crash under rabbit's @tf.function: building the
JVP of an ``Equal`` op that carries a tangent raised ``IndexError`` in TF's
nested-forward-mode autodiff. The culprit was the λ_inf==0 / den==0 masks in
:mod:`btgrid_tf` comparing a differentiated tensor; freezing the comparison input
(``btgrid_tf._frozen_eq_zero`` = ``tf.equal(stop_gradient(x), 0)``) removes the
tangent into ``Equal`` without changing any value or derivative (the comparison
is a measure-zero boundary ``tf.where`` never differentiates). Full-K now runs
under @tf.function and matches the exact reverse-mode Hessian to machine
precision (≤3e-16 rel; see HESSIAN_PLAN.md §7 + the isolation validation). So
both GN and full-K are available; GN remains the default for Asimov (exact and
cheaper — 8 vs 72 fold passes).

WARNING: do NOT pass hessian_straightthrough=1 during the FIT. The
surrogate recomputes J(/K) on every compute() call — fine for the one-shot
covariance pass, but it would cripple the minimizer (many gradient/HVP evals).
"""

import os
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
from wremnants.utilities.data_paths import getDataPath

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
_UNFOLDING_HDF5_DEFAULT = os.path.join(
    wrem_common.data_dir,
    "TheoryCorrections",
    "scetlib_np",
    "mz_dilepton_unfolding_R_skim.hdf5",
)
_BTGRID_SUBDIR = ("scetlib_np", "Z_COM13_CT18Z_N3p0LL_btgrid_fineall")
_DISCRETE_NP_SUBSTRING = "scetlibnp"

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

# Recommended Gaussian prior widths for the SCETlib NP λ parameters
DEFAULT_PRIOR_SIGMAS = {
    "lambda2_nu": 0.10,
    "lambda2": 0.50,  # 0.4 ⁺⁰·⁶₋₀.₄ -> symmetric average
    "lambda4": 0.50,  # 0.4 ⁺⁰·⁶₋₀.₄ -> symmetric average
    "delta_lambda2": 0.20,  # 0 ± 0.20 wide default (no theorist value yet)
}


def _default_btgrid_dir():
    base = getDataPath(fallback="/scratch/submit/cms/wmass/NanoAOD")
    return os.path.join(os.path.dirname(base), *_BTGRID_SUBDIR)


def _load_lambda_central_file(path):
    """
    Load a λ_central override from a JSON or YAML file.
    """
    return scetlib_lambda_central.load_lambda_central_file(path)


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
        unfolding_hdf5_path: Optional[str] = None,
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
        legacy_recon: bool = False,
        xparam_default: Optional[str] = None,
        hessian_straightthrough: bool = False,
        hessian_gn: bool = False,
        **kwargs,
    ):
        """Construct the ParamModel.

        Usage::
            --paramModel wremnants.postprocessing.scetlib_np.SCETlibNPParamModel [key=value ...]

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
            (``_default_btgrid_dir()``, built on the ROOT/narf-free
            ``wremnants.utilities.data_paths.getDataPath()``); pass explicitly
            at non-subMIT sites.
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
        priors
            Enable Gaussian priors on the λ parameters (spec token
            ``priors=1``). Rabbit applies priors whenever a ParamModel
            *declares* ``prior_sigmas`` — there is no rabbit-side CLA (the
            old ``--paramModelPriors`` flag was dropped in WMass/rabbit#133);
            the model itself decides. This token IS that decision: only when
            set does the model declare ``prior_sigmas``. Default off →
            everything floats free.
        prior_sigmas
            Per-name override for the Gaussian prior σ on each parameter —
            a Mapping, or as spec token the comma-separated string form
            ``prior_sigmas=lambda2=0.3,delta_lambda2=nan`` (same format as
            ``xparam_default``). Defaults come from ``DEFAULT_PRIOR_SIGMAS``:

                lambda2_nu : 0.10
                lambda2    : 0.50   (symmetric approx of +0.6/-0.4)
                lambda4    : 0.50   (symmetric approx of +0.6/-0.4)

            All other params default to ``NaN`` → no prior, float free; in
            practice they are expected to be frozen with rabbit's
            ``--freezeParameters`` until the theorist provides priors for
            them. Pass ``np.nan`` here to free a constrained param, or a
            finite value to add a prior on one that defaults to NaN.
            Only meaningful together with ``priors=1``; ignored (with a
            warning) otherwise.
            Prior mean for each param is ``self.xparamdefault`` (the
            runcard's λ_central).
        nonsingular_fo_sing, nonsingular_dyturbo
            Paths to the σ_ns inputs (SCETlib singular pkl / DYTurbo
            scetlibmatch txt); default to the wremnants-data
            TheoryCorrections copies. σ_ns is always included — the matched
            σ_gen^matched(λ) = σ_gen^resum(λ) + σ_ns is what the histmaker
            nominal carries; resum-only diagnostics subtract ``sigma_ns``.
        nonsingular_qt_cutoff
            Low-qT cutoff (GeV) below which σ_ns is zeroed (the FO−singular
            difference is numerically unreliable at tiny qT).
        legacy_recon
            Use the legacy per-bin (Nbins, Nbt) reconstruction layout instead
            of the default memory-factorized one (numerically equivalent to
            ≲1e-14 rel; for parity checks only). See FACTORIZED_RECON.md.
        xparam_default
            Comma-separated ``name=value,...`` string shifting the fit START
            (and the prior mean) off the runcard's λ_central — for closure /
            injection tests. The truth (ratio denominator) is NOT moved.
        hessian_straightthrough
            Expose compact λ-derivatives (J, optionally K) to autodiff while
            keeping the exact value — for the one-shot two-pass covariance
            recipe ONLY (see the module docstring); never set during a fit.
        hessian_gn
            With ``hessian_straightthrough``: Gauss-Newton, keep J and drop
            the K term (exact for Asimov, where the residual vanishes).
        """
        self.indata = indata

        if unfolding_hdf5_path is None:
            unfolding_hdf5_path = _UNFOLDING_HDF5_DEFAULT
        if btgrid_dir is None:
            btgrid_dir = _default_btgrid_dir()

        self._check_discrete_np_double_counting()

        # ---- λ_central
        # Three sources of λ_central, in priority order:
        #   1. ``lambda_central=<file>`` spec token — path to a JSON or YAML
        #      file with ``eff_params`` and ``gnu_params``. Overrides the
        #      metadata auto-detect; useful when the upstream SCETlib pkl isn't
        #      accessible (e.g. a colleague's input).
        #   2. ``lambda_central`` constructor arg (explicit dict, programmatic).
        #   3. Auto-detect from the fit hdf5's theoryCorr → upstream pkl.
        lambda_central_source = (
            "constructor-arg" if lambda_central is not None else None
        )
        if isinstance(lambda_central, str):
            # CLI token (lambda_central=<file>): RECOMMENDED override route —
            # the --paramModel spec is stored in the fitresults meta, so the
            # override is recorded in the output (env var/dict are not).
            lc_path = lambda_central
            lambda_central = _load_lambda_central_file(lc_path)
            lambda_central_source = f"cli-file:{lc_path}"
            print(
                f"[SCETlibNPParamModel] λ_central from CLI file {lc_path!r}",
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
            lambda_central_source = "auto-detect:indata.metadata theoryCorr"
            print(
                f"[SCETlibNPParamModel] λ_central auto-detected from indata.metadata",
                flush=True,
            )
        print(f"[SCETlibNPParamModel] λ_central:", flush=True)
        for key, value in lambda_central.items():
            print(f"  {key} = {value!r}", flush=True)

        self.lambda_central_source = lambda_central_source

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

        # Per-bin qT and Y (from the bin tuple).
        bins = grid["bins"]
        qT_pb_np = np.array([b[2] for b in bins], dtype=np.float64)
        self.qT_per_bin = tf.constant(qT_pb_np, dtype=fz_tf.DTYPE)
        Y_pb_np = np.array([b[1] for b in bins], dtype=np.float64)
        self.Y_per_bin = tf.constant(Y_pb_np, dtype=fz_tf.DTYPE)

        # F_eff depends on the bin only through Y (not Q or qT), and Y takes few
        # distinct values across the grid. Precompute the unique-Y map so the
        # reconstruction evaluates the NP transcendentals on NY rows and
        # gathers, instead of recomputing identical rows for every (Q, qT).
        Y_feff_unique_np, Y_feff_inv_np = np.unique(Y_pb_np, return_inverse=True)
        Y_feff_inv_np = Y_feff_inv_np.reshape(-1).astype(np.int32)
        self.Y_feff_unique = tf.constant(Y_feff_unique_np, dtype=fz_tf.DTYPE)
        self.Y_feff_inverse_idx = tf.constant(Y_feff_inv_np, dtype=tf.int32)

        bT_simpson_w_np = fz_tf.simpson_weights(np.asarray(grid["bT"]))
        self.bT_simpson_w = tf.constant(bT_simpson_w_np, dtype=fz_tf.DTYPE)

        # Reconstruction layout. Default: factorized (deduplicated rows +
        # unique-qT J0 kernel + Simpson-as-matmul) — numerically equivalent to
        # the legacy (Nbins, Nbt) layout (≲1e-14 rel., summation order only)
        # but ~6x smaller, which is what lets the fit run on a 32 GB GPU.
        # The legacy_recon=1 spec token selects the legacy path (parity
        # checks).
        self.factorized = not bool(legacy_recon)

        # Hessian straight-through switches (see the module docstring's
        # two-pass recipe). Spec tokens hessian_straightthrough=1 /
        # hessian_gn=1, recorded in the fitresults meta via the stored
        # --paramModel spec.
        self._hess_st = bool(hessian_straightthrough)
        self._hess_gn = bool(hessian_gn)
        if self.factorized:
            dd = fz_tf.dedup_grid_rows(
                grid["I_pert"][0], grid["C_nu"][0], Y_feff_inv_np
            )
            self.I_pert_u = tf.constant(dd["I_u"], dtype=fz_tf.DTYPE)  # (Nu, Nbt)
            # C_ν via the second-level dedup: exp(C·g) runs on the small
            # (Ncu, Nbt) table and is gathered — bit-identical, ~150x fewer
            # transcendentals, no (Nu, Nbt) C constant on device.
            self.C_nu_uu = tf.constant(dd["C_uu"], dtype=fz_tf.DTYPE)  # (Ncu, Nbt)
            self.c_of_u = tf.constant(dd["c_of_u"], dtype=tf.int32)
            self.feff_idx_u = tf.constant(dd["feff_idx_u"], dtype=tf.int32)
            # Per-bin index into the unique-qT axis. The bin qT values are by
            # construction members of qT_unique, so searchsorted is an exact
            # lookup (asserted).
            qT_idx_np = np.searchsorted(idx_map["qT_unique"], qT_pb_np)
            assert np.array_equal(idx_map["qT_unique"][qT_idx_np], qT_pb_np)
            self.gather_idx = tf.constant(
                np.stack([dd["row_uid"].astype(np.int64), qT_idx_np], axis=1),
                dtype=tf.int32,
            )
            # Drop the host-side dedup copies (the tf.constants own the data now).
            del dd
            # Weighted J0 kernel on the unique-qT grid, with the per-bin qT
            # prefactor and the Simpson weights folded in: (NqT, Nbt).
            K_u = fz_tf.build_bT_J0_kernel(
                tf.constant(idx_map["qT_unique"], dtype=fz_tf.DTYPE), self.bT
            )
            self.KwqT = (
                tf.constant(idx_map["qT_unique"], dtype=fz_tf.DTYPE)[:, tf.newaxis]
                * K_u
                * self.bT_simpson_w[tf.newaxis, :]
            )
        else:
            self.I_pert = tf.constant(
                grid["I_pert"][0], dtype=fz_tf.DTYPE
            )  # (Nbins, Nbt)
            self.C_nu = tf.constant(grid["C_nu"][0], dtype=fz_tf.DTYPE)
            # Precompute the bT·J0(qT·bT) kernel (λ-independent).
            self.bT_J0_kernel = fz_tf.build_bT_J0_kernel(self.qT_per_bin, self.bT)
        # Drop the ~17.5 GB host-side grid reference before TF graph building.
        del grid

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
        # ---- Fixed-order/DYTurbo nonsingular term (NP-independent).
        # σ_gen^matched(λ) = σ_gen^resum(λ) + σ_ns. Added at GEN level, so it folds
        # through the same response R as the resummed piece. Because rnorm is a
        # ratio, this correctly DILUTES the NP variation where the FO dominates
        # (high qT). σ_ns is a constant (no λ dependence), always included —
        # the matched σ_gen is what the histmaker nominal carries; resum-only
        # diagnostics subtract self.sigma_ns instead of rebuilding the model.
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
                "The matched model needs the fixed-order inputs for "
                "σ_ns = DYTurbo − SCETlib_singular, but these are missing:\n  "
                + "\n  ".join(missing)
                + "\nThey live under wremnants-data/data/TheoryCorrections (the "
                "SCETlib singular …_nnlo_sing…combined.pkl and the DYTurbo "
                "results_…scetlibmatch.txt). Pass nonsingular_fo_sing / "
                "nonsingular_dyturbo to point at them."
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

        # Impact groups over our own parameters, consumed by the Fitter's
        # traditional impacts. rabbit's built-in systgroup machinery can't
        # represent these: its group indices are syst-relative, but our λ are
        # POUs (model nuisances), which have no syst index. The Fitter resolves
        # these labels -> floating full-x indices and computes the conditional
        # group impact from the covariance. Split into the two NP sectors:
        # CS-side γ_ν (lambda*_nu) vs TMD-effective F_eff.
        #
        # ``resumNonpert`` is the SAME group name setupRabbit assigns to the
        # discrete scetlibNP* template variations in the old-style datacard
        # (there resumNonpert == exactly those 4 nuisances). Emitting it here
        # (= all our lambda) makes the grouped-impact bar directly comparable
        # between the new param model and the old NP variations. It does NOT
        # collide with a syst group: the new-model datacard excludes scetlibNP,
        # so resumNonpert is absent from indata.systgroups.
        self.param_impact_groups = {
            "resumNonpert": tuple(ALL_PARAMS),
            "scetlibNPgammaNu": tuple(GNU_PARAMS),
            "scetlibNPFeff": tuple(EFF_PARAMS),
        }

        # Defaults: λ_central values per parameter. Optionally overridden by
        # the ``xparam_default=name=value,...`` spec token — comma-separated
        # pairs (for closure tests where the data-generating / fit-start
        # point should differ from the card's λ_central).
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
        # rabbit's set_param_default expects an internal-storage convention
        # where POIs (npoi entries) are SQRT(value) if not allowNegativeParam.
        # For our λ which can in principle be tiny / zero (delta_lambda2),
        # default to allowNegativeParam=True so the stored value == λ directly.
        self.allowNegativeParam = True
        self.is_linear = False
        self.xparamdefault = tf.constant(defaults, dtype=indata.dtype)

        # Gaussian priors (semantics documented on the constructor args).
        # Rabbit's Fitter applies them whenever the model DECLARES
        # ``prior_sigmas`` (no rabbit-side CLA — WMass/rabbit#133), so the
        # declaration is gated behind ``priors``: off → no attribute →
        # everything floats free. The Fitter takes the prior means from
        # xparamdefault, so an xparam_default shift moves start AND prior
        # mean together (to centre priors on truth while starting shifted,
        # prior_means would have to be decoupled from xparamdefault).
        self._use_priors = bool(priors)
        # ``prior_sigmas`` may be a Mapping (programmatic) or the spec-token
        # string ``prior_sigmas=lambda2=0.3,delta_lambda2=nan`` — the same
        # comma-separated name=value format as xparam_default; value ``nan``
        # frees the param.
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
            # redundantly — Fitter will fall back to xparamdefault.
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

    def _check_discrete_np_double_counting(self):
        """Refuse to run on a datacard containing discrete scetlibNP systs.

        They describe the same physics as this ParamModel's continuous λ;
        running both double-counts (the discrete syst absorbs shape variation
        the ParamModel should describe: spurious pull on the indata syst,
        postfit λ not what the data prefers).
        """

        systs = getattr(self.indata, "systs", None)
        if systs is None or len(systs) == 0:
            return
        syst_names = [s.decode() if isinstance(s, bytes) else str(s) for s in systs]

        conflicting = [
            s for s in syst_names if self._DISCRETE_NP_SUBSTRING in s.lower()
        ]
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
    # σ_gen evaluation
    # =========================================================================

    def _sigma_YqT_native_at(self, eff_params, gnu_params):
        """Reconstruct σ(λ) on the btgrid and Q-integrate, returning the result
        in the btgrid's *native* binning: shape (NY, NqT) on the signed-Y /
        qT grid (Y_unique, qT_unique), BEFORE the |Y|-fold and qT-rebin. This
        is the object that the native-binning validation compares against the
        SCETlib spectrum reference (curve 1) and the numpy `factorize` (curve 2)."""
        # 1. Reconstruct σ on the btgrid's flat (Nbins,) layout. Factorized
        # (default) and legacy layouts are numerically equivalent (≲1e-14
        # rel.; summation order only — see FACTORIZED_RECON.md).
        eff = {k: v for k, v in eff_params.items() if k != "np_model"}
        gnu = {k: v for k, v in gnu_params.items() if k != "np_model_nu"}
        if self.factorized:
            sigma_flat = fz_tf.reconstruct_batch_factorized_tf(
                b_bar=self.b_bar,
                I_pert_u=self.I_pert_u,
                C_nu_uu=self.C_nu_uu,
                c_of_u=self.c_of_u,
                eff_params=eff,
                gnu_params=gnu,
                np_model=self.np_model,
                np_model_nu=self.np_model_nu,
                KwqT=self.KwqT,
                gather_idx=self.gather_idx,
                Y_unique=self.Y_feff_unique,
                feff_idx_u=self.feff_idx_u,
            )
        else:
            sigma_flat = fz_tf.reconstruct_batch_tf(
                qT_per_bin=self.qT_per_bin,
                bT=self.bT,
                I_pert=self.I_pert,
                C_nu=self.C_nu,
                b_bar=self.b_bar,
                Y_per_bin=self.Y_per_bin,
                eff_params=eff,
                gnu_params=gnu,
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
        # hessian_straightthrough=1 spec token;
        # hessian_gn=1 drops the curvature term
        # (Gauss-Newton/Fisher — exact for Asimov, 8 vs 72 passes). Resolved at
        # construction (self._hess_st / self._hess_gn).
        # Do NOT enable during the fit: it recomputes J(/K) every call.
        if self._hess_st:
            ratio = self._ratio_straightthrough(param, use_curvature=not self._hess_gn)
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
