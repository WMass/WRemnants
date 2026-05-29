"""Mixture model + flow for the unbinned J/ψ mass calibration.

Two-stage continuity design. The flow models only the NOMINAL (θ=0) mass shape
``p₀(m | muon_kin)`` and never conditions on θ; the θ-dependence is supplied
analytically in stage 2 (the continuity tilt, ``data_nll_continuity``).

* ``theta_scale`` ∈ R^{24×3}  — per-η-bin (A, e, M) muon scale nuisances.
* ``theta_smear`` ∈ R^{24×2}  — per-η-bin (a, c) signed width-smear coefficients.

Stage 1 (``log_p_nominal``): train the conditional flow ``p₀(m | muon_kin)`` on
simulation at θ=0 (leak-free kinematic conditioning only).

Stage 2 (``data_nll_continuity``, frozen flow): the signal density is the
nominal flow forward-folded analytically by a deterministic, invertible map
``x = μ + (1+s)(m' + s_adv(m') − μ)`` — a scale advection ``s_adv`` plus a signed
mass-density stretch ``s`` (μ = mean m_ll). ``s`` is the variance-equivalent of a
per-muon qop smear with VARIANCE ``σ²_qop = a + c·k²`` (signed/two-sided): the
smear is applied as a score-driven probability-flow displacement that reproduces
a Gaussian qop smear of mass-variance ``σ²_qop = a + c·k²`` (``_continuity_logp``),
and the SAME (a, c) drive the per-muon qop fold in the validation plots. Evaluated at
the source pre-image with the change-of-variables Jacobian (``_continuity_logp``)
— no flow derivatives, normalised by construction; the smear is a pure mass-space
stretch so it leaves the ρ conditioning untouched. Mixed with a degree-1
Bernstein background via the MLP ``f(c)``:

  data event:  p(m | c, θ) = f_0(c) p_0 + f_1(c) p_1 + (1 − f_0 − f_1)·p_s(m | c, θ)
  MC event:    p(m | c, θ) = p_s(m | c, θ)
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-use ``build_flow`` from the existing trainer.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from train_muon_response_flow import FlowWithLogProb, build_flow  # noqa: E402


# ---------------------------------------------------------------------------
# zuko monkey-patch: floor the jacobian before .log() in MonotonicTransform.
# ---------------------------------------------------------------------------
#
# zuko's ``MonotonicTransform.call_and_ladj`` returns ``y, jacobian.log()``.
# For the GF flow's GMM-CDF base, the mathematical jacobian is always
# strictly positive, but it underflows to *exactly* 0 in float32 when the
# input is far from every mixture component — typical at random init.
#  → ``log(0) = -inf`` forward → ``log_p = -inf`` → ``-log p = +inf``
#    poisons the per-batch NLL on the very first batch.
# The backward of ``log(0)`` is also degenerate (``grad_out / 0`` → ±inf
# or NaN), which is the path the anomaly tracer initially reported.
#
# Floor the jacobian at ``1e-30`` before the log: forward becomes finite
# (worst-case ``log(1e-30) ≈ -69``) and the backward is bounded
# (``clamp_min`` gradient is 0 in the clamped region, so the second-order
# autograd through the GF's f-derivative stays clean). Events at the
# clamp boundary contribute zero gradient through this layer — they're
# already in a "no signal" regime where the flow can't tell which way
# to move its mixture, so dropping their gradient is the right behaviour
# anyway; the conditioner network still receives gradient through the
# other (un-clamped) layers.
#
# We additionally clamp the transform INPUT ``x`` before ``self.f(x)``.
# The GF base maps via ``erf(x / √2)``, which in float32 saturates to
# exactly ±1 for ``|x| ≳ 5.7`` — sending the subsequent inverse-CDF / GMM
# map to ±∞ and NaN-ing the backward (the ErfBackward0 → Mul/Exp NaN the
# anomaly tracer reports). ``x`` here is *post-conditioner-affine*, so an
# external mass clamp is not enough: a sharp conditioner (or a pathological
# forward-fold mass) can push ``x`` past the saturation point even for an
# in-window event. Clamping at ±5 (erf(5/√2)=erf(3.54)=0.9999994, finite)
# is well beyond the ~N(0,1) regime the flow maps real events to, so it
# only ever bounds the pathological tail; the jacobian is taken w.r.t. the
# clamped value so the density stays self-consistent there.
def _install_zuko_jacobian_floor(
    min_jac: float = 1e-30, x_clamp: float = 5.0
) -> None:
    import zuko.transforms as _zt

    if getattr(_zt.MonotonicTransform, "_jpsi_jacobian_floored", False):
        return

    _orig_call_and_ladj = _zt.MonotonicTransform.call_and_ladj

    def _safe_call_and_ladj(self, x):
        # Reproduce the zuko logic but (a) clamp the erf input and (b) clamp
        # the jacobian before log.
        create_graph = torch.is_grad_enabled() and (
            x.requires_grad or bool(self.phi)
        )
        with torch.enable_grad():
            x = x.clone().requires_grad_()
            x_safe = x.clamp(-x_clamp, x_clamp)
            y = self.f(x_safe)
        jacobian = torch.autograd.grad(
            y, x_safe, torch.ones_like(y), create_graph=create_graph,
        )[0]
        return y, jacobian.clamp_min(min_jac).log()

    _zt.MonotonicTransform.call_and_ladj = _safe_call_and_ladj
    _zt.MonotonicTransform._jpsi_jacobian_floored = True


_install_zuko_jacobian_floor()


# ---------------------------------------------------------------------------
# zuko monkey-patch: guard the Gaussianization transform's internal overflow.
# ---------------------------------------------------------------------------
#
# ``GaussianizationTransform.f`` computes
#     erf((x · exp(scale_i) + shift_i) / √2)   →  mean over i  →  erfinv
# where (shift_i, scale_i) are the per-component parameters the *conditioner*
# predicts. Two float32 failure modes, both reached when the conditioner
# outputs large values (e.g. when the standardised θ-conditioning grows):
#   1. ``self.scale = exp(scale_i)`` overflows to +inf for scale_i ≳ 88 →
#      ``x · inf`` is inf (or 0·inf = NaN when x = 0) → poisons forward AND
#      backward (0·inf in the Mul/Exp backward — the reported MulBackward0).
#   2. the erf argument saturates erf→±1 for |arg| ≳ 5.7 → its backward is a
#      0·inf NaN even though the forward (capped by the ·(1−1e-6) term) is OK.
# Clamp scale_i before the exp (keeps self.scale finite, huge margin to inf)
# and clamp the erf argument (keeps erf away from saturation). Both clamps
# have zero gradient in the clamped region, so pathological events stop
# contributing gradient cleanly instead of NaN-ing the batch. Healthy flows
# map real events to |arg| ~ O(1) with scale_i ~ O(1), so neither clamp ever
# fires there.
def _install_gaussianization_guards(
    scale_param_clamp: float = 30.0, erf_arg_clamp: float = 5.0
) -> None:
    import zuko.transforms as _zt

    G = _zt.GaussianizationTransform
    if getattr(G, "_jpsi_gf_guarded", False):
        return

    _orig_init = G.__init__

    def _safe_init(self, shift, scale, **kwargs):
        scale = scale.clamp(-scale_param_clamp, scale_param_clamp)
        _orig_init(self, shift, scale, **kwargs)

    def _safe_f(self, x):
        arg = x[..., None] * self.scale + self.shift
        arg = arg.clamp(-erf_arg_clamp, erf_arg_clamp)
        y = torch.erf(arg / math.sqrt(2))
        y = torch.mean(y, dim=-1) * (1 - 1e-6)
        y = torch.erfinv(y) * math.sqrt(2)
        return y

    G.__init__ = _safe_init
    G.f = _safe_f
    G._jpsi_gf_guarded = True


_install_gaussianization_guards()


def _noise_active(sigma) -> bool:
    """True if the given σ should trigger noise sampling.

    Scalar 0 / None → no noise (use parameter as-is). Any Tensor → noise
    (caller is responsible for non-negative values). Positive scalar →
    noise. The trainer's adaptive-σ helper returns Tensors when adaptive
    sampling is active and the fixed scalar during warmup.
    """
    if sigma is None:
        return False
    if isinstance(sigma, torch.Tensor):
        return True
    return float(sigma) > 0.0


# Hard clamp on the standardised mass fed to the GF flow. In float32,
# erf(x/√2) saturates to exactly ±1 for |x| ≳ 5.7, which sends the flow's
# inverse-CDF map to ±∞ and NaNs the exp(−z²/2) in the backward pass. The
# J/ψ window cut (2.92–3.28 GeV) keeps real events at |mll_std| ≲ 3.9, so
# this only ever bounds pathological MC forward-fold tails (a near-zero qop
# from a large sampled smear/scale → huge mass); those tail events get the
# clamp-edge density and contribute no gradient, instead of crashing.
MLL_STD_FLOW_CLAMP = 5.0

# Match the convention of make_jpsi_crctn_helper: 24 η-bins, (A, e, M).
N_ETA_BINS = 24
N_THETA_SCALE = 3      # (A, e, M)
N_THETA_SMEAR = 2      # (a, c)
N_THETA_SCALE_PM = 2 * N_THETA_SCALE   # 6 — flat per-event scale vector
N_THETA_SMEAR_PM = 2 * N_THETA_SMEAR   # 4 — flat per-event smear vector

# Conditioning sizes.
#
#   muon_kin_std   7  = (η_+, η_-, cos φ_+, sin φ_+, cos φ_-, sin φ_-, ρ)
#                       with ρ = (pt_+ − pt_-)/(pt_+ + pt_-)
#   theta_scale_pm 6  = (A_+, e_+, M_+, A_-, e_-, M_-)         ← scale conditioning
#   theta_smear_pm 4  = (a_+, c_+, a_-, c_-)                   ← smear conditioning
#
# ``muon_kin`` is the *leak-free* kinematic conditioning: η_±, φ_± (as cos/sin
# to be wrap-free and keep φ_± recoverable for the φ-dependent detector
# response) and the pt asymmetry ρ. These span 5 of the 6 dimuon DOF, leaving
# the pt *scale* ↔ m_ll free — so the flow's target is not determined by its
# conditioning. The SIGNAL FLOW conditions on (muon_kin, θ_scale_pm,
# θ_smear_pm); the background-fraction MLP conditions on muon_kin alone (same
# kinematics, no nuisances). ``y_event`` (dilepton-level vars) is no longer
# used by either head (it carries pt_ll, which would re-pin m_ll); the loader
# still emits it but the model ignores it.
N_Y_EVENT = 7    # legacy; emitted by the loader but unused by the model
N_MUON_KIN = 7
N_FLOW_COND = N_MUON_KIN + N_THETA_SCALE_PM + N_THETA_SMEAR_PM  # 17

# Muon rest mass in GeV (J/ψ analyses use this everywhere).
MUON_MASS_GEV = 0.1056583755

# Fixed reference scales for the per-muon qop-resolution VARIANCE parameters
# (σ²_qop = a·SCALE_A + c·SCALE_C·k², k=1/pt). The physical qop variance is
# O(1e-7) (σ_qop ~ 3e-4), a terrible optimizer scale; these put the fitted (and
# injected) (a, c) at O(1) so the standard smear LR works and a runaway is
# bounded to a physical σ_qop. Calibrated so θ≈1 ≈ a +20% m_ll-variance smear on
# the J/ψ sample: a≈8e-8 (c=0) and c≈2.3e-5 (a=0) each broaden Var(m_ll) by ~20%,
# i.e. θ_a≈0.8 at SCALE_A and θ_c≈1.1 at SCALE_C. (SCALE_C was 1e-6, which left
# θ_c≈20 — far from O(1) — so θ_c chronically under-converged at the smear LR.)
SMEAR_VAR_SCALE_A = 1e-7
SMEAR_VAR_SCALE_C = 2e-5

# Invertibility floor on the probability-flow smear Jacobian G' = dx/dm'. A valid
# forward (broadening) transport has G' > 0; the floor catches the fold/over-
# sharpen region (V·∂²_m log p₀ large) gracefully, mirroring the old (1+s) ≥ 0.05.
# Used by jacobian_form="softlog" as the seam where the analytic log is replaced
# by a C¹ linear tangent extension (see _softlog_below_floor). Unused by "exp".
SMEAR_GP_FLOOR = 0.05


# Gauss-Hermite nodes for the analytic Gaussian-convolution smear operator
# (smear_operator="gh_convolution"). Cached per (n, device, dtype).
_GH_CACHE: dict = {}


def _gh_nodes(n: int, device, dtype):
    """Gauss–Hermite nodes/log-weights for ``E_{ε~N(0,1)}[f] ≈ Σ_i W_i f(ξ_i)``.

    ``∫ f(t) e^{-t²} dt ≈ Σ w_i^H f(t_i^H)`` ⇒ with ``ξ = √2 t^H`` and
    ``W = w^H/√π`` (so Σ W = 1) we get the standard-normal expectation.
    Cached. Returns ``(ξ [n], logW [n])``."""
    key = (int(n), str(device), str(dtype))
    if key not in _GH_CACHE:
        import numpy as _np
        t, w = _np.polynomial.hermite.hermgauss(int(n))
        xi = _np.sqrt(2.0) * t
        logW = _np.log(w) - 0.5 * _np.log(_np.pi)
        _GH_CACHE[key] = (
            torch.as_tensor(xi, device=device, dtype=dtype),
            torch.as_tensor(logW, device=device, dtype=dtype),
        )
    return _GH_CACHE[key]


def _softlog_below_floor(g: torch.Tensor, floor: float = SMEAR_GP_FLOOR) -> torch.Tensor:
    """``log(g.clamp_min(floor))`` PLUS a quadratic barrier ``(floor − g)⁺² /
    (2·floor²)`` for ``g < floor``. Drop-in replacement for the hard
    ``torch.log(g.clamp_min(floor))`` that:

    - matches it EXACTLY in the physical region (``g ≥ floor`` → barrier = 0),
    - is BOUNDED from below by ``log(floor)`` for any g (so the density
      cannot diverge to ±∞ from a degenerate or extreme-negative Jacobian
      — the linear-tangent variant ran away to ``−∞`` for very negative g,
      driving NLL ``→ −∞`` — a *reward* for unphysical G' < 0, the opposite
      of what we want),
    - is C¹ at the seam (barrier value and slope both zero at ``g = floor``)
      and grows quadratically past it, so the gradient w.r.t. g is
      ``−(floor − g)/floor²`` for ``g < floor`` — pulling g BACK toward the
      physical region, replacing the hard clamp's flat-NLL basin with a
      gentle restoring force whose strength scales with how far past the
      floor we've drifted.

    Returns ``log_Gp_effective`` to plug into ``−log p_θ = log_Gp_effective
    − log p₀(m')``."""
    log_safe = torch.log(g.clamp_min(floor))
    deficit = (floor - g).clamp_min(0.0)
    barrier = deficit * deficit / (2.0 * floor * floor)
    return log_safe + barrier

# Fixed reference scales for the per-muon SCALE parameters (A, e, M). The
# physical values are O(1e-4 / 1e-3 / 1e-5); these put the fitted (and injected)
# θ_scale at O(1) so the standard scale LR works and all three components share a
# well-conditioned step (physical A,e,M = θ_scale · THETA_SCALE_REF).
THETA_SCALE_REF = (1e-4, 1e-3, 1e-5)


# ---------------------------------------------------------------------------
# Degeneracy-whitening preconditioner
# ---------------------------------------------------------------------------
# (A, e) and (a, c) are each strongly degenerate over the narrow J/ψ pt range:
# the scale response basis (1, −k) for (A, e) and the smear-variance basis
# (1, k²) for (a, c) are nearly collinear, so the loss has a long tilted valley
# that Adam — a purely DIAGONAL preconditioner — cannot navigate. We rotate the
# gradient of each pair into a decorrelated basis with a fixed matrix L built
# from the per-η-bin curvature moments (⟨k⟩, ⟨k²⟩, ⟨k⁴⟩).
#
# The rotation is applied as a BACKWARD-ONLY transform (``_WhitenGradFn``):
# the forward pass is the identity, so the forward map, the softplus positivity
# reparam, the likelihood value, and the observed Fisher are ALL unchanged — only
# the optimizer's step direction is preconditioned. It is gated on ``self.training``
# so it is fully inert (identity backward too) during Fisher / diagnostics / eval,
# leaving the covariance in the physical θ basis exact. Adam's subsequent diagonal
# rescaling acts on the already-decorrelated axes and cannot reintroduce the
# cross-correlation, so Adam is kept for all parameters.


class _WhitenGradFn(torch.autograd.Function):
    """Identity forward; in backward left-multiplies the gradient w.r.t. the
    last (size-2) axis by a fixed lower-triangular whitening matrix ``L`` (the
    same gradient a forward reparam ``p = L·θ'`` would produce, but without
    touching the forward value). ``L`` is either ``[2, 2]`` (global, MLP mode)
    or batched ``[..., 2, 2]`` (per-η-bin, binned mode)."""

    @staticmethod
    def forward(ctx, x, L):
        ctx.save_for_backward(L)
        return x

    @staticmethod
    def backward(ctx, g):
        (L,) = ctx.saved_tensors
        if L.dim() == 2:
            gx = torch.einsum("...i,ij->...j", g, L)
        else:
            gx = torch.einsum("...i,...ij->...j", g, L)
        return gx, None


def _whitening_L(rho: float, max_rho: float = 0.99) -> torch.Tensor:
    """Lower-triangular ``L`` with ``L Lᵀ = C⁻¹`` for the 2×2 correlation
    ``C = [[1, ρ],[ρ, 1]]`` (ρ clamped to ±``max_rho`` to bound the whitening
    of the near-degenerate direction). Satisfies ``Lᵀ C L = I``, so a parameter
    pair whose loss curvature has correlation ρ becomes unit-conditioned in the
    rotated coordinates. Returns ``[2, 2]`` float32."""
    r = float(max(-max_rho, min(max_rho, rho)))
    C = torch.tensor([[1.0, r], [r, 1.0]], dtype=torch.float64)
    Cinv = torch.linalg.inv(C)
    L = torch.linalg.cholesky(Cinv)
    return L.to(torch.float32)


# ---------------------------------------------------------------------------
# Bernstein degree-1 background basis (unchanged)
# ---------------------------------------------------------------------------


def bernstein_d1(
    mll: torch.Tensor, m_lo: float, m_hi: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Degree-1 Bernstein basis on ``[m_lo, m_hi]``, normalised to ∫=1.

    Densities in raw m_ll units (1/GeV) so the mixture sum is dimensionally
    consistent with ``exp(log p_signal)``.
    """
    width = m_hi - m_lo
    u = (mll - m_lo) / width
    p0 = (1.0 - u) * 2.0 / width
    p1 = u * 2.0 / width
    return p0, p1


# ---------------------------------------------------------------------------
# Event-level kinematic helpers (autograd-friendly)
# ---------------------------------------------------------------------------


def _sintheta_from_eta(eta_pm: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.cosh(eta_pm)


def _event_mll(
    pt_pm: torch.Tensor, eta_pm: torch.Tensor, phi_pm: torch.Tensor
) -> torch.Tensor:
    """Two-body invariant mass for muons of mass ``MUON_MASS_GEV``.

    Inputs ``[B, 2]`` per (+, −). Output ``[B]``. Autograd-friendly.
    """
    px = pt_pm * torch.cos(phi_pm)
    py = pt_pm * torch.sin(phi_pm)
    pz = pt_pm * torch.sinh(eta_pm)
    p2 = px * px + py * py + pz * pz
    E = torch.sqrt(p2 + MUON_MASS_GEV * MUON_MASS_GEV)
    Etot = E.sum(-1)
    Px = px.sum(-1)
    Py = py.sum(-1)
    Pz = pz.sum(-1)
    m2 = Etot * Etot - (Px * Px + Py * Py + Pz * Pz)
    return torch.sqrt(m2.clamp_min(1e-12))


# ---------------------------------------------------------------------------
# Mixture MLP — unchanged
# ---------------------------------------------------------------------------


class MixtureMLP(nn.Module):
    """3-way softmax over (f_0, f_1, f_s) as a function of ``muon_kin_std``."""

    def __init__(self, n_input: int = N_MUON_KIN, hidden: int = 32, n_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        d_in = n_input
        for _ in range(n_layers):
            layers += [nn.Linear(d_in, hidden), nn.GELU()]
            d_in = hidden
        layers.append(nn.Linear(d_in, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, y_std: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(y_std), dim=-1)


class ThetaNet(nn.Module):
    """Per-muon calibration parameters (A, e, M, a, c) as a CONTINUOUS function
    of the muon (η, φ), replacing the η-binned θ. Input features (η, cosφ, sinφ),
    5 outputs. The (A, e, M) outputs are scaled by fixed references so the net
    outputs sit at O(1) (A,e ~ 1e-3, M ~ 1e-5); (a, c) are the qop-resolution
    variance coefficients in their O(1) units (SMEAR_VAR_SCALE_* applied
    downstream). The final layer is ZERO-INITIALISED, so the net outputs 0 at
    init — the binned θ=0 init (no scale/smear correction)."""

    def __init__(self, hidden: int = 32, n_layers: int = 2,
                 scale_ref=THETA_SCALE_REF):
        super().__init__()
        layers: list[nn.Module] = []
        d_in = 3  # (η, cosφ, sinφ)
        for _ in range(max(1, n_layers)):
            layers += [nn.Linear(d_in, hidden), nn.GELU()]
            d_in = hidden
        last = nn.Linear(d_in, N_THETA_SCALE + N_THETA_SMEAR)  # 5 = (A,e,M,a,c)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)
        self.register_buffer(
            "scale_ref", torch.tensor(list(scale_ref), dtype=torch.float32))

    def forward(self, eta_pm: torch.Tensor, phi_pm: torch.Tensor):
        """``eta_pm``, ``phi_pm``: ``[B, 2]``. Returns ``(AeM [B,2,3], ac [B,2,2])``
        — per-muon physical (A, e, M) and O(1) (a, c)."""
        feat = torch.stack(
            [eta_pm, torch.cos(phi_pm), torch.sin(phi_pm)], dim=-1)  # [B,2,3]
        out = self.net(feat)                                        # [B,2,5]
        return out[..., :N_THETA_SCALE] * self.scale_ref, out[..., N_THETA_SCALE:]


@contextlib.contextmanager
def _freeze_param_grads(params):
    """Temporarily set ``requires_grad=False`` on ``params`` for the duration
    of the ``with`` block, restoring the previous state afterwards.

    Used to exclude a sub-module's *parameters* from a backward while still
    propagating gradient to that sub-module's *inputs* — a frozen layer still
    passes gradient to earlier tensors. We only flip (and later restore)
    params that were ``requires_grad=True`` on entry, so this is a no-op for
    already-frozen params (e.g. during Fisher-info, where the flow is frozen).
    """
    changed = [p for p in params if p.requires_grad]
    for p in changed:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p in changed:
            p.requires_grad_(True)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class JpsiMassMixtureModel(nn.Module):
    """Unbinned J/ψ mass-fit model — two-stage continuity design (the flow models
    the nominal shape p₀(m|muon_kin); θ enters analytically in stage 2)."""

    def __init__(
        self,
        m_lo: float,
        m_hi: float,
        mll_log_scale: float,
        # Stats for standardising the flow's mass input + conditioning.
        # Passed in as plain floats / tensors; stored as buffers.
        mll_mean: float = 0.0,
        mll_std: float = 1.0,
        y_event_mean: torch.Tensor | None = None,
        y_event_std_tensor: torch.Tensor | None = None,
        muon_kin_mean: torch.Tensor | None = None,
        muon_kin_std_tensor: torch.Tensor | None = None,
        # Flow / MLP hyperparameters.
        flow_arch: str = "gf",
        flow_n_transforms: int = 5,
        flow_hidden_features: int = 128,
        flow_n_hidden_layers: int = 3,
        flow_gf_components: int = 8,
        flow_nsf_bins: int = 8,
        mlp_hidden: int = 32,
        mlp_n_layers: int = 2,
        n_eta_bins: int = N_ETA_BINS,
        # Debug toggle: drop the residual-smearing kernel + σ_qop_pm
        # conditioning entirely. ``theta_smear`` stays as a Parameter
        # (for state_dict shape consistency) but is unused; trainer is
        # expected to exclude it from the optimizer.
        smearing_enabled: bool = True,
        # Symmetric toggle for the scale: drop the T_scale forward-fold + the
        # θ_scale_pm conditioning entirely. ``theta_scale`` stays as a (zero,
        # inert) Parameter; trainer excludes it from the optimizer. With both
        # scale and smearing disabled only the flow + MLP (background) train.
        scale_enabled: bool = True,
        # Robustness floor for the qop→pt inversion. A scale/smear shift adds
        # to qop = q·sinθ/pt; if it drives qop through zero, pt → ∞ and the
        # reconstructed mass explodes (catastrophic at high |η| where |qop| is
        # smallest and the fitted σ_qop can approach |qop|). We floor the
        # shifted |qop| at ``qop_floor_frac · |qop_orig|`` with the original
        # sign, so a resolution smear can neither flip the charge nor inflate
        # pt by more than ``1/qop_floor_frac``. 0 disables the floor (legacy).
        qop_floor_frac: float = 0.25,
        # Which per-bin smear terms to *fit*: "both" (a and c), "a" (constant
        # term only), or "c" (∝1/pt term only). The constant a and the c·k
        # term are nearly degenerate over the narrow J/ψ pt range, so fitting
        # both per η-bin is ill-posed and yields the unphysical bin-to-bin
        # zig-zag. Fitting one removes the degeneracy; the non-fitted term is
        # zeroed (``smear_param_mask``) so it contributes exactly 0 to the width
        # factor s and receives no gradient (inert).
        smear_fit_params: str = "both",
        # Which per-η-bin SCALE terms to fit — a subset of "AeM". A (constant
        # δqop) and e (∝1/pt) are NEARLY DEGENERATE over the narrow J/ψ pt range
        # (A's uniform m_ll scaling vs e's ∝(k₊+k₋) shift are ~collinear), so
        # fitting both from J/ψ alone is ill-posed — the fit slides into large
        # opposite-sign (A, e). Default "AM" drops the degenerate e (the J/ψ-
        # identifiable subset: constant scale A + charge-odd sagitta M). The
        # dropped term is zeroed (scale_param_mask) → 0 advection, no gradient.
        scale_fit_params: str = "AM",
        # Number of Euler steps integrating the smear's probability-flow ODE in
        # the density (``_continuity_logp``). 1 = first-order (single score
        # displacement); more steps integrate the score-driven flow more finely
        # (smaller per-step Jacobian → more robust + accurate) at a higher
        # nested-autograd cost. Frozen p₀ score per step (exact diffusion to
        # first order in V).
        smear_flow_steps: int = 1,
        # Smear operator for the continuity density. 'pf_ode' (default): the
        # deterministic probability-flow ODE ``y ← y − (V/2N)·∂_m log p_0(y)``
        # (N = smear_flow_steps), with Jacobian via jacobian_form. Cheap but
        # diverges from the true Gaussian convolution at large V/σ² — for
        # Gaussian p_0 the PF-ODE gives σ_out = σ·exp(V/2σ²), while the true
        # convolution gives σ_out = √(σ²+V); they agree to leading order but
        # the PF-ODE over-broadens exponentially at large V (e.g. +50% σ at
        # V/σ² ≈ 2). 'gh_convolution': EXACT stochastic Gaussian convolution
        # ``p_θ(x|c) = E_ε[p_0(m'(ε)|c)/|G'|]`` via Gauss-Hermite quadrature
        # over ε ~ N(0,1) (``n_gh_nodes`` nodes), with x = m' + s_adv(m') +
        # √V(m')·ε and G'(m') = 1 + s_adv'(m') + (V'/(2√V))·ε computed via
        # autograd. Matches the per-muon qop fold operator (which IS Gaussian
        # convolution); the closure-target curve (flow at injected θ) overlaps
        # the pseudo-data by construction at large V. Requires V ≥ 0 (clamps
        # internally); use with smear_param_form='softplus' to guarantee it
        # without clamping.
        smear_operator: str = "pf_ode",
        n_gh_nodes: int = 8,
        # Positivity reparameterisation for θ_smear. 'linear' (default): the
        # raw θ_smear is the O(1) coefficient directly (physical (a, c) =
        # θ·SMEAR_VAR_SCALE, signed — supports both broadening V>0 and the
        # 'unsmear' V<0 region). 'softplus': constrains each of (a, c)
        # INDIVIDUALLY to ≥ 0 via physical = softplus(θ)·SMEAR_VAR_SCALE; the
        # per-η mask is applied AFTER softplus so frozen params (or 'a'-/'c'-
        # only modes) are EXACTLY zero in the per-muon σ_qop and the
        # transformations. Useful when you want to defend against the
        # negative-c drift (issue #2) by construction, at the cost of losing
        # the two-sided fit (the model can no longer represent MC that is too
        # broad vs data).
        smear_param_form: str = "linear",
        # Per-event normalisation correction for the transformed-flow density.
        # The forward map T_θ is NOT boundary-preserving on [m_lo, m_hi]:
        # broadening T pushes some probability mass outside the window, so
        # Z(θ;c) = ∫_window p_θ(x|c) dx < 1 and the bare `log p_θ(x)` carries a
        # `-log Z` bias per event that always pulls the fit toward smaller
        # broadening (the bias was measured at +0.34 per event at the truth in
        # forward |η| bins for a c=5e-5 injection). Three modes:
        #
        # "none" (default): no correction (current behaviour preserved).
        # "linear": leading-order boundary expansion `1 - Z ≈ p_0(m_lo)·(m_lo
        #   − T(m_lo))_+ + p_0(m_hi)·(T(m_hi) − m_hi)_+` — 2 boundary forward
        #   evals + 2 flow evals per event. Valid for small V.
        # "flow_cdf": exact via Z = F_0(T⁻¹(m_hi)|c) − F_0(T⁻¹(m_lo)|c), where
        #   F_0 is the flow's CDF (the GF's monotonic transform composed with
        #   the standard normal CDF Φ). Inverts T at the boundaries (~2 extra
        #   fixed-point inversions per event) and evaluates the CDF at the
        #   preimages. Exact up to inversion discretisation.
        norm_correction: str = "none",
        # Background mixture. True (default): the data branch's per-event NLL
        # is the full f_data(c)-weighted signal/Bernstein mixture (the model
        # for real data, which has genuine non-resonant background). False:
        # the data branch reduces to pure signal (NLL = −log p_signal); the
        # MLP `f_data` is bypassed entirely and its parameters are excluded
        # from the optimiser in stage 2. Use for validation closures (truth
        # f_bkg = 0 by construction) to remove the bkg ↔ smear degeneracy
        # where the MLP grows f_bkg in forward |η| bins to absorb tail
        # events the signal model can't broaden into.
        background_enabled: bool = True,
        # Smear-Jacobian formula for the continuity density. 'softlog' (default):
        # autograd-derived G' = dx/dm' of the actual forward map, with a C¹
        # tangent extension below SMEAR_GP_FLOOR so the optimiser is pulled BACK
        # from G' < floor instead of sliding into a flat-NLL basin. 'exp':
        # frozen-score continuous-flow approximation log G' = log(1+s_adv'(m'))
        # − V·∂²_m log p₀(m')/2 — always finite, no floor, but approximates a
        # *different* operator (extra "score constant along the trajectory"
        # assumption) — see _continuity_logp docstring caveats.
        jacobian_form: str = "softlog",
        # θ parameterisation. 'binned' (default): per-η-bin (A,e,M,a,c) tables
        # indexed by the muon's η-bin. 'mlp': a small ThetaNet maps each muon's
        # (η, φ) → (A,e,M,a,c) CONTINUOUSLY (trained in stage 2 like the
        # background MLP). The binned tables stay registered but inert in 'mlp'.
        theta_mode: str = "binned",
        theta_mlp_hidden: int = 32,
        theta_mlp_layers: int = 2,
        # Degeneracy-whitening preconditioner (see _WhitenGradFn). When True,
        # the gradient of the (A, e) and (a, c) pairs is rotated into a
        # decorrelated basis built from the per-η-bin curvature moments, so the
        # near-collinear pairs can be fit JOINTLY without the tilted-valley
        # stalling that forces dropping a degenerate term. Backward-only: the
        # forward map / likelihood / Fisher are unchanged. Only acts when BOTH
        # members of a pair are fit (scale_fit_params ⊇ {A,e} / smear_fit_params
        # == 'both'); otherwise there is no degeneracy to whiten and it is inert.
        theta_whiten: bool = False,
        theta_whiten_max_rho: float = 0.99,
        # Per-η-bin curvature moment SUMS [n_eta, 4] = (N, Σk, Σk², Σk⁴) from
        # JpsiMassPreprocStats.k_moments. Required to build the whitening; if
        # None while theta_whiten=True, whitening is disabled with a warning.
        k_moments=None,
    ):
        super().__init__()
        self.smearing_enabled = bool(smearing_enabled)
        self.scale_enabled = bool(scale_enabled)
        self.qop_floor_frac = float(qop_floor_frac)
        self.smear_flow_steps = max(1, int(smear_flow_steps))
        if smear_operator not in ("pf_ode", "gh_convolution"):
            raise ValueError(
                f"smear_operator must be 'pf_ode' or 'gh_convolution'; "
                f"got {smear_operator!r}")
        self.smear_operator = str(smear_operator)
        self.n_gh_nodes = max(2, int(n_gh_nodes))
        if jacobian_form not in ("softlog", "exp"):
            raise ValueError(
                f"jacobian_form must be 'softlog' or 'exp'; got {jacobian_form!r}")
        self.jacobian_form = str(jacobian_form)
        if smear_param_form not in ("linear", "softplus"):
            raise ValueError(
                f"smear_param_form must be 'linear' or 'softplus'; "
                f"got {smear_param_form!r}")
        self.smear_param_form = str(smear_param_form)
        if norm_correction not in ("none", "linear", "flow_cdf"):
            raise ValueError(
                f"norm_correction must be 'none', 'linear', or 'flow_cdf'; "
                f"got {norm_correction!r}")
        self.norm_correction = str(norm_correction)
        self.background_enabled = bool(background_enabled)
        if theta_mode not in ("binned", "mlp"):
            raise ValueError(f"theta_mode must be 'binned' or 'mlp', got {theta_mode!r}")
        self.theta_mode = str(theta_mode)
        self.flow_arch = str(flow_arch)

        # Per-bin smear fit mask: which of (a, c) float. The frozen column gets
        # no gradient (held at init) and is not perturbed in the MC sampling.
        if smear_fit_params not in ("both", "a", "c"):
            raise ValueError(
                f"smear_fit_params must be 'both', 'a', or 'c'; got {smear_fit_params!r}"
            )
        self.smear_fit_params = str(smear_fit_params)
        _mask = {"both": [1.0, 1.0], "a": [1.0, 0.0], "c": [0.0, 1.0]}[smear_fit_params]
        # Non-persistent: reconstructed from smear_fit_params at __init__, so
        # checkpoints without this buffer still load.
        self.register_buffer(
            "smear_param_mask", torch.tensor(_mask, dtype=torch.float32),
            persistent=False,
        )
        # Per-bin scale fit mask: which of (A, e, M) float (breaks the A/e
        # degeneracy). The dropped term gets no gradient (multiply by 0) → held
        # inert at 0. Non-persistent (reconstructed at __init__).
        if not scale_fit_params or any(ch not in "AeM" for ch in scale_fit_params):
            raise ValueError(
                f"scale_fit_params must be a non-empty subset of 'AeM'; got "
                f"{scale_fit_params!r}")
        self.scale_fit_params = str(scale_fit_params)
        _csmask = [1.0 if ch in scale_fit_params else 0.0 for ch in ("A", "e", "M")]
        self.register_buffer(
            "scale_param_mask", torch.tensor(_csmask, dtype=torch.float32),
            persistent=False,
        )

        # Two-stage continuity design: the flow models only the nominal shape
        # p₀(m|muon_kin) at θ=0 — it never conditions on θ (the θ-dependence is
        # supplied analytically in stage 2, see ``data_nll_continuity``).
        flow_inner = build_flow(
            n_features=1,
            n_cond=N_MUON_KIN,
            n_transforms=flow_n_transforms,
            hidden_features=flow_hidden_features,
            n_hidden_layers=flow_n_hidden_layers,
            architecture=self.flow_arch,
            gf_components=flow_gf_components,
            nsf_bins=flow_nsf_bins,
        )
        self.flow = FlowWithLogProb(flow_inner)

        # Background-fraction MLP conditions on the same kinematics as the
        # flow (muon_kin), minus the nuisances.
        self.mlp = MixtureMLP(
            n_input=N_MUON_KIN, hidden=mlp_hidden, n_layers=mlp_n_layers
        )

        # Learnable nuisances.
        self.theta_scale = nn.Parameter(
            torch.zeros(n_eta_bins, N_THETA_SCALE, dtype=torch.float32)
        )
        # θ_smear are signed per-η-bin qop-resolution VARIANCE coefficients
        # (a, c): σ²_qop = a + c·k² (two-sided). They drive BOTH the per-muon qop
        # fold (validation) and the mass-density stretch (density), consistently.
        # Init at 0 → σ²_qop = 0 (identity), free to broaden or unsmear.
        self.theta_smear = nn.Parameter(
            torch.zeros(n_eta_bins, N_THETA_SMEAR, dtype=torch.float32)
        )
        # 'mlp' θ: a small net maps each muon's (η, φ) → (A,e,M,a,c) continuously
        # (zero-init → 0 at start). Replaces the binned tables above (which stay
        # registered but inert). Trained in stage 2 like the background MLP.
        self.theta_net = (
            ThetaNet(hidden=theta_mlp_hidden, n_layers=theta_mlp_layers)
            if self.theta_mode == "mlp" else None
        )

        # Degeneracy-whitening preconditioner (backward-only; see _WhitenGradFn).
        self._build_whitening(
            bool(theta_whiten), float(theta_whiten_max_rho), k_moments, n_eta_bins)

        # Buffers — Bernstein window, density-rescale, standardisation stats.
        self.register_buffer("m_lo", torch.tensor(float(m_lo)))
        self.register_buffer("m_hi", torch.tensor(float(m_hi)))
        self.register_buffer("mll_log_scale", torch.tensor(float(mll_log_scale)))
        self.register_buffer("mll_mean_buf", torch.tensor(float(mll_mean)))
        self.register_buffer("mll_std_buf", torch.tensor(float(mll_std)))

        def _buf(t, n):
            if t is None:
                return torch.zeros(n, dtype=torch.float32), torch.ones(n, dtype=torch.float32)
            return torch.as_tensor(t, dtype=torch.float32), None

        if y_event_mean is None:
            y_event_mean = torch.zeros(N_Y_EVENT)
        if y_event_std_tensor is None:
            y_event_std_tensor = torch.ones(N_Y_EVENT)
        if muon_kin_mean is None:
            muon_kin_mean = torch.zeros(N_MUON_KIN)
        if muon_kin_std_tensor is None:
            muon_kin_std_tensor = torch.ones(N_MUON_KIN)
        self.register_buffer(
            "y_event_mean", torch.as_tensor(y_event_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "y_event_std", torch.as_tensor(y_event_std_tensor, dtype=torch.float32)
        )
        self.register_buffer(
            "muon_kin_mean", torch.as_tensor(muon_kin_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "muon_kin_std", torch.as_tensor(muon_kin_std_tensor, dtype=torch.float32)
        )

    # ------------------------------------------------------------------
    # Per-event helpers
    # ------------------------------------------------------------------

    def _build_whitening(self, enabled, max_rho, k_moments, n_eta) -> None:
        """Build the (A,e) and (a,c) gradient-whitening matrices from the per-η-bin
        curvature moments ``k_moments`` [n_eta, 4] = (N, Σk, Σk², Σk⁴). Registers
        per-bin ``[n_eta, 2, 2]`` and global ``[2, 2]`` buffers for each pair and
        sets the per-pair active flags. The buffers are identity (inert) when
        whitening is disabled, when the pair is not fully fit (no degeneracy to
        break), or when moments are unavailable (older stats.json)."""
        cm = self.scale_param_mask
        sm = self.smear_param_mask
        scale_pair_fit = bool(cm[0] > 0 and cm[1] > 0)   # both A and e fit
        smear_pair_fit = bool(sm[0] > 0 and sm[1] > 0)   # both a and c fit
        self.theta_whiten = bool(enabled)
        self.theta_whiten_max_rho = float(max_rho)

        eye = torch.eye(2, dtype=torch.float32)
        sW_b = eye.expand(n_eta, 2, 2).clone()
        sW_g = eye.clone()
        cW_b = eye.expand(n_eta, 2, 2).clone()
        cW_g = eye.clone()

        have_moments = bool(enabled and k_moments is not None)
        if enabled and k_moments is None:
            warnings.warn(
                "theta_whiten=True but k_moments is None (older stats.json); "
                "degeneracy whitening DISABLED.", RuntimeWarning)
        if have_moments:
            km = torch.as_tensor(k_moments, dtype=torch.float64).reshape(-1, 4)
            N = km[:, 0].clamp_min(1.0)
            k1, k2, k4 = km[:, 1] / N, km[:, 2] / N, km[:, 3] / N
            Ntot = km[:, 0].sum().clamp_min(1.0)
            g1 = km[:, 1].sum() / Ntot
            g2 = km[:, 2].sum() / Ntot
            g4 = km[:, 3].sum() / Ntot
            # ρ_scale = −⟨k⟩/√⟨k²⟩  (A,e response basis (1, −k));
            # ρ_smear = +⟨k²⟩/√⟨k⁴⟩ (a,c variance-response basis (1, k²)).
            def _rs(m1, m2):
                return float(-m1 / torch.sqrt(m2.clamp_min(1e-30)))

            def _rc(m2, m4):
                return float(m2 / torch.sqrt(m4.clamp_min(1e-30)))

            for b in range(n_eta):
                sW_b[b] = _whitening_L(_rs(k1[b], k2[b]), max_rho)
                cW_b[b] = _whitening_L(_rc(k2[b], k4[b]), max_rho)
            sW_g = _whitening_L(_rs(g1, g2), max_rho)
            cW_g = _whitening_L(_rc(g2, g4), max_rho)

        self.register_buffer("_scale_W_binned", sW_b, persistent=False)
        self.register_buffer("_scale_W_global", sW_g, persistent=False)
        self.register_buffer("_smear_W_binned", cW_b, persistent=False)
        self.register_buffer("_smear_W_global", cW_g, persistent=False)
        self._whiten_scale_active = bool(have_moments and scale_pair_fit)
        self._whiten_smear_active = bool(have_moments and smear_pair_fit)

    def _scale_AeM_pm(self, eta_pm, phi_pm, b_pm) -> torch.Tensor:
        """Per-muon PHYSICAL scale params ``[B, 2, 3] = (A, e, M)``, masked to the
        fitted terms (scale_param_mask, default A,M only — drops the A/e-
        degenerate e). 'binned': the η-bin table θ_scale[b] (O(1) fit param) ×
        THETA_SCALE_REF; 'mlp': the ThetaNet(η, φ), which already applies REF."""
        if self.theta_mode == "mlp":
            aem = self.theta_net(eta_pm, phi_pm)[0]
        else:
            aem = self.theta_scale[b_pm] * self.theta_scale.new_tensor(THETA_SCALE_REF)
        # Whiten the (A, e) gradient (backward-only; identity in eval/forward).
        if self.training and self._whiten_scale_active:
            L = (self._scale_W_global if self.theta_mode == "mlp"
                 else self._scale_W_binned[b_pm])
            # Whiten in the O(1) θ space — where THETA_SCALE_REF calibrates the
            # curvature diagonal to ≈unit, which is the only space the
            # unit-diagonal L = chol(C⁻¹) is valid in. In PHYSICAL (A,e) space
            # the A/e diagonal ratio is ⟨(∂m/∂A)²⟩/⟨(∂m/∂e)²⟩ = 1/⟨k²⟩ ≈ 140,
            # and a unit-diagonal whitening there grossly mis-scales the step
            # (→ non-finite loss at large lr). Divide out REF, whiten, multiply
            # back: the forward is the identity, only the O(1)-space gradient is
            # rotated. (The smear (a,c) is already whitened in its O(1) space.)
            ref_ae = aem.new_tensor(THETA_SCALE_REF[:2])
            ae = _WhitenGradFn.apply(aem[..., :2] / ref_ae, L) * ref_ae
            aem = torch.cat([ae, aem[..., 2:]], dim=-1)
        return aem * self.scale_param_mask

    def _smear_raw_to_effective(self, raw: torch.Tensor) -> torch.Tensor:
        """Apply the positivity reparameterisation (if any) and the per-bin
        fit mask to the raw O(1) ``θ_smear`` tensor (shape ``[..., 2]``).
        'linear' (default): identity (signed). 'softplus': ``softplus(raw)``
        so each of (a, c) ≥ 0 INDIVIDUALLY. The mask is applied AFTER the
        transform so frozen params (or 'a'-/'c'-only modes) are EXACTLY zero
        regardless of the raw value — keeping the per-muon σ_qop and all
        downstream transformations evaluated to zero for the inactive term."""
        if self.smear_param_form == "softplus":
            raw = F.softplus(raw)
        return raw * self.smear_param_mask

    def _smear_ac_pm(self, eta_pm, phi_pm, b_pm) -> torch.Tensor:
        """Per-muon EFFECTIVE qop-resolution variance coefficients ``[B, 2, 2]
        = (a, c)`` (σ²_qop = a + c·k²), with the positivity reparam (if any)
        applied and masked to the fitted term(s). 'binned': ``θ_smear[b]``;
        'mlp': the continuous ``ThetaNet(η, φ)``."""
        if self.theta_mode == "mlp":
            ac = self.theta_net(eta_pm, phi_pm)[1]
        else:
            ac = self.theta_smear[b_pm]
        ac = self._smear_raw_to_effective(ac)
        # Whiten the (a, c) gradient (backward-only; identity in eval/forward).
        # Applied AFTER softplus — the forward pass is the identity so positivity
        # is untouched; only the gradient is rotated.
        if self.training and self._whiten_smear_active:
            L = (self._smear_W_global if self.theta_mode == "mlp"
                 else self._smear_W_binned[b_pm])
            ac = _WhitenGradFn.apply(ac, L)
        return ac

    def effective_theta_smear(self) -> torch.Tensor:
        """Per-η-bin PHYSICAL smear coefficients (a, c), masked (BINNED mode only
        — used for the diagnostics curve / bootstrap σ). The fit parameter
        ``theta_smear`` is O(1) for the optimizer; the physical qop-variance
        coefficients (σ²_qop = a + c·k²) are ``effective(θ_smear) · SMEAR_VAR_SCALE``,
        where ``effective`` applies the positivity reparam (linear ↔ identity,
        softplus ↔ softplus). In 'mlp' mode evaluate ``theta_net`` on an η grid."""
        scale = self.theta_smear.new_tensor([SMEAR_VAR_SCALE_A, SMEAR_VAR_SCALE_C])
        return self._smear_raw_to_effective(self.theta_smear) * scale

    def _scale_per_event(self, eta_pm, phi_pm, b_pm) -> torch.Tensor:
        """Per-event ``[B, 6] = (A_+, e_+, M_+, A_-, e_-, M_-)``."""
        return self._scale_AeM_pm(eta_pm, phi_pm, b_pm).reshape(b_pm.shape[0], -1)

    def _qop_var_pm(self, eta_pm, phi_pm, b_pm, pt_pm) -> torch.Tensor:
        """Per-muon SIGNED qop-resolution variance
        ``σ²_qop,μ = a·SCALE_A + c·SCALE_C·k_μ²`` (k = 1/pt), from the per-muon
        (a, c). ``a``, ``c`` are COMBINED here, before any clipping. ``[B, 2]``."""
        ac = self._smear_ac_pm(eta_pm, phi_pm, b_pm)        # [B,2,2] masked
        a_pm = ac[..., 0] * SMEAR_VAR_SCALE_A
        c_pm = ac[..., 1] * SMEAR_VAR_SCALE_C
        k2 = (1.0 / pt_pm) ** 2
        return a_pm + c_pm * k2                             # [B, 2], signed

    def fold_sigma_qop_pm(
        self, pt_pm: torch.Tensor, eta_pm: torch.Tensor, phi_pm: torch.Tensor,
        b_pm: torch.Tensor,
    ) -> torch.Tensor:
        """Per-muon σ_qop for the validation FOLD, from the fitted (a, c): the
        combined qop variance ``σ²_qop = a + c·k²`` CLIPPED AT 0 *after*
        combining (a stochastic Gaussian qop kick can only broaden, so the
        unsmearing region σ² < 0 → no kick). Returns ``[B, 2]``."""
        return torch.sqrt(self._qop_var_pm(eta_pm, phi_pm, b_pm, pt_pm).clamp_min(0.0))

    # ------------------------------------------------------------------
    # T_scale (analytic + linearized)
    # ------------------------------------------------------------------

    def _delta_qop_analytic(
        self,
        AeM_pm: torch.Tensor,
        pt_pm: torch.Tensor,
        eta_pm: torch.Tensor,
        q_pm: torch.Tensor,
    ) -> torch.Tensor:
        """Analytic δqop per muon (matches ``calculateQopUnc`` in
        ``muon_calibration.hpp``)::

            δqop_i = q_i · sinθ_i · [(A_i − e_i k_i) k_i + q_i M_i]

        ``AeM_pm`` is the per-muon ``[B, 2, 3] = (A, e, M)`` (from
        ``_scale_AeM_pm`` — binned table or ThetaNet).
        """
        sintheta = _sintheta_from_eta(eta_pm)
        k_pm = 1.0 / pt_pm
        A_pm = AeM_pm[..., 0]
        e_pm = AeM_pm[..., 1]
        M_pm = AeM_pm[..., 2]
        k_unc = (A_pm - e_pm * k_pm) * k_pm + q_pm * M_pm
        return q_pm * sintheta * k_unc

    def _qop_new_to_pt(
        self,
        qop: torch.Tensor,
        qop_new: torch.Tensor,
        q_pm: torch.Tensor,
        sintheta: torch.Tensor,
    ) -> torch.Tensor:
        """Invert a shifted qop back to pt, sign-preserving and floored.

        ``qop_new = qop + shift`` may approach or cross zero (a large
        smear/scale shift), which sends ``pt = q·sinθ/qop_new`` to ∞ or
        flips the charge — unphysical. We project ``qop_new`` onto the sign
        of the original ``qop`` and floor its magnitude at
        ``qop_floor_frac · |qop|`` (so pt inflates by at most
        ``1/qop_floor_frac``). ``qop_floor_frac == 0`` restores the old
        near-zero-only guard.
        """
        if self.qop_floor_frac > 0.0:
            s = torch.sign(qop)
            floor = self.qop_floor_frac * qop.abs()
            # Signed magnitude along qop's sign, floored, then re-signed.
            qop_new = s * torch.maximum(qop_new * s, floor)
        else:
            qop_new = torch.where(
                qop_new.abs() < 1e-12,
                torch.full_like(qop_new, 1e-12) * torch.sign(qop_new + 1e-30),
                qop_new,
            )
        return q_pm * sintheta / qop_new

    def _apply_scale_pt(
        self,
        pt_pm: torch.Tensor,
        eta_pm: torch.Tensor,
        q_pm: torch.Tensor,
        delta_qop: torch.Tensor,
        sign: float,
    ) -> torch.Tensor:
        """Shift qop by ``sign·δqop`` and convert back to pt.

        Conventions: ``qop = q · sinθ / pt``. ``sign=+1`` → forward
        T_scale (truth → obs); ``sign=−1`` → inverse T_scale (obs → truth).
        """
        sintheta = _sintheta_from_eta(eta_pm)
        qop = q_pm * sintheta / pt_pm
        qop_new = qop + sign * delta_qop
        return self._qop_new_to_pt(qop, qop_new, q_pm, sintheta)

    def jacobian_mll_linearized(
        self,
        mll: torch.Tensor,
        pt_pm: torch.Tensor,
        q_pm: torch.Tensor,
        b_pm: torch.Tensor,
    ) -> torch.Tensor:
        """Per-event closed-form Jacobian J = ∂m_ll/∂θ_scale_pm.

        From ``m_ll ≈ m_ll · (1 − ½ Σ_i δqop_i / qop_i)`` and the analytic
        ``δqop_i = q_i sinθ_i [(A − e k) k + q_i M]`` (so ``δqop_i/qop_i =
        A − e k_i + q_i M pt_i``, since ``qop_i = q_i sinθ_i k_i``)::

            ∂m_ll/∂A_i = −½ m_ll               (charge-even scale)
            ∂m_ll/∂e_i = +½ m_ll · k_i         (charge-even)
            ∂m_ll/∂M_i = −½ m_ll · q_i · pt_i  (charge-odd / sagitta)

        Returns ``[B, 6]`` ordered ``(A_+, e_+, M_+, A_-, e_-, M_-)``; independent
        of ``b_pm`` (those route the gradient to ``theta_scale`` via scatter).
        """
        k_pm = 1.0 / pt_pm
        m_half = 0.5 * mll
        dA = -m_half.unsqueeze(-1).expand_as(q_pm)
        de = +m_half.unsqueeze(-1) * k_pm
        dM = -m_half.unsqueeze(-1) * q_pm * pt_pm
        return torch.stack(
            [dA[:, 0], de[:, 0], dM[:, 0], dA[:, 1], de[:, 1], dM[:, 1]], dim=-1
        )

    # ------------------------------------------------------------------
    # T_smear
    # ------------------------------------------------------------------

    def apply_smear_pt(
        self,
        pt_pm: torch.Tensor,
        eta_pm: torch.Tensor,
        q_pm: torch.Tensor,
        sigma_qop_pm: torch.Tensor,
        eps_pm: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian smear of qop_i → return smeared pt_i. ``eps_pm`` is
        the unit-normal noise the caller supplies (lets the trainer
        couple the same draw between an MLE branch and a flow-aux branch
        if it wants).
        """
        sintheta = _sintheta_from_eta(eta_pm)
        qop = q_pm * sintheta / pt_pm
        delta = sigma_qop_pm * eps_pm
        qop_new = qop + delta
        return self._qop_new_to_pt(qop, qop_new, q_pm, sintheta)

    # ------------------------------------------------------------------
    # Conditioning vector for the flow
    # ------------------------------------------------------------------

    def _standardise_mll(self, mll: torch.Tensor) -> torch.Tensor:
        return (mll - self.mll_mean_buf) / self.mll_std_buf

    # ------------------------------------------------------------------
    # Nominal (θ=0) flow density — the stage-1 template
    # ------------------------------------------------------------------

    def log_p_nominal(
        self, mll_obs: torch.Tensor, muon_kin_std_obs: torch.Tensor
    ) -> torch.Tensor:
        """``log p₀(m | muon_kin)`` — the θ=0 nominal flow density (1/GeV).

        The stage-1 target: a plain conditional density on the (uncorrected)
        reco mass, conditioned on the leak-free kinematics only — the flow
        never sees θ (the two-stage continuity design).
        """
        mll_std = self._standardise_mll(mll_obs).clamp(
            -MLL_STD_FLOW_CLAMP, MLL_STD_FLOW_CLAMP
        )
        return self.flow(mll_std.unsqueeze(-1), muon_kin_std_obs) - self.mll_log_scale

    # ------------------------------------------------------------------
    # MLP coefficients (data branch only)
    # ------------------------------------------------------------------

    def f_data(self, muon_kin_std: torch.Tensor) -> torch.Tensor:
        return self.mlp(muon_kin_std)

    # ------------------------------------------------------------------
    # Stage-2 continuity-equation data fit (frozen flow + analytic v, κ)
    #
    #   log p_s(m|c,θ) = log p₀(m|c) + δ(m,c;θ)        [+ O(θ²) renorm]
    #   δ = Σ_k θ_k g_k,   g_scale = −v′ − v·s,   g_smear = ½ κ (s′ + s²)
    #
    # p₀ is the frozen θ=0 flow (trained in stage 1, conditioned on muon_kin
    # only). v = ∂m/∂θ_scale (analytic linearised Jacobian), v′ = ∂_m v (through
    # the (m,c)→pt reconstruction), κ = ∂Var[m]/∂θ_smear (analytic). The data
    # θ-gradient flows by autograd through this δ; the flow gets no gradient.
    # ------------------------------------------------------------------

    def _flow_logp_score(self, mll: torch.Tensor, muon_kin_std: torch.Tensor):
        """Nominal density + score from the frozen θ=0 flow.

        Returns ``(log p₀, s, s′)`` (each ``[B]``): p₀ in 1/GeV,
        ``s = ∂_m log p₀``, ``s′ = ∂²_m log p₀``, via autograd in ``m``.
        """
        m = mll.detach().requires_grad_(True)
        mll_std = self._standardise_mll(m).clamp(-MLL_STD_FLOW_CLAMP, MLL_STD_FLOW_CLAMP)
        cond = self._build_flow_cond(muon_kin_std, None, None)
        logp = self.flow(mll_std.unsqueeze(-1), cond) - self.mll_log_scale  # [B]
        s = torch.autograd.grad(logp.sum(), m, create_graph=True)[0]
        s_prime = torch.autograd.grad(s.sum(), m, create_graph=True)[0]
        return logp, s, s_prime

    @staticmethod
    def _reconstruct_pt(mll, eta_pm, phi_pm, rho):
        """``(m, η_±, φ_±, ρ) → (pt₊, pt₋)`` for the (massless) dimuon.

        ``ρ = (pt₊−pt₋)/(pt₊+pt₋)``; with ``S = pt₊+pt₋`` and
        ``m² = ½ S² (1−ρ²)(cosh Δη − cos Δφ)`` this pins the pt scale. Used to
        propagate ``v`` along the `m`-direction at fixed conditioning.
        """
        d_eta = eta_pm[:, 0] - eta_pm[:, 1]
        d_phi = phi_pm[:, 0] - phi_pm[:, 1]
        ang = (torch.cosh(d_eta) - torch.cos(d_phi)).clamp_min(1e-6)
        S2 = 2.0 * mll * mll / ((1.0 - rho * rho).clamp_min(1e-6) * ang)
        S = torch.sqrt(S2.clamp_min(1e-12))
        return torch.stack([S * (1.0 + rho) * 0.5, S * (1.0 - rho) * 0.5], dim=-1)

    def _v_and_vprime(self, mll, eta_pm, phi_pm, q_pm, b_pm, rho):
        """Advective velocity ``v = ∂m/∂θ_scale_pm`` ``[B,6]`` and its `m`-
        derivative ``v′ = ∂_m v`` ``[B,6]`` at fixed conditioning.

        ``v`` is the analytic linearised mass-Jacobian; ``v′`` is its *total*
        derivative in ``m`` (the continuity "missing-Jacobian" term ``∇·v``),
        taken through the ``(m,c)→pt`` reconstruction so the kinematics track
        ``m`` at fixed ``c``.
        """
        m = mll.detach().requires_grad_(True)
        pt = self._reconstruct_pt(m, eta_pm, phi_pm, rho)
        v = self.jacobian_mll_linearized(m, pt, q_pm, b_pm)  # [B,6]
        vprime = torch.stack(
            [torch.autograd.grad(v[:, k].sum(), m, create_graph=True, retain_graph=True)[0]
             for k in range(v.shape[-1])],
            dim=-1,
        )
        return v, vprime

    def _kappa_smear(self, mll, pt_pm, eta_pm, phi_pm, q_pm):
        """Diffusion coefficients ``κ = ∂Var[m]/∂θ_smear`` per per-muon term,
        ``[B,4] = (a₊,c₊,a₋,c₋)``.

        With the variance basis ``σ_qop² = a²·1 + c²·k²`` (so ``∂σ²/∂a²=1``,
        ``∂σ²/∂c²=k²``): ``κ_{a,i}=(∂m/∂qop_i)²`` and ``κ_{c,i}=(∂m/∂qop_i)² k_i²``.
        ``∂m/∂qop_i`` is taken by autograd through ``pt_i = q_i sinθ_i / qop_i``.
        """
        sintheta = _sintheta_from_eta(eta_pm)
        qop = (q_pm * sintheta / pt_pm).detach().requires_grad_(True)
        pt_from_qop = q_pm * sintheta / qop
        m = _event_mll(pt_from_qop, eta_pm, phi_pm)
        dm_dqop = torch.autograd.grad(m.sum(), qop, create_graph=True)[0]  # [B,2]
        d2 = dm_dqop * dm_dqop  # [B,2]
        k2 = (1.0 / pt_pm) ** 2  # [B,2]
        # order (a₊, c₊, a₋, c₋)
        return torch.stack([d2[:, 0], d2[:, 0] * k2[:, 0],
                            d2[:, 1], d2[:, 1] * k2[:, 1]], dim=-1)

    def _smear_per_event_linear(self, b_pm: torch.Tensor) -> torch.Tensor:
        """Per-event smear *increments* ``[B,4] = (a₊,c₊,a₋,c₋)`` for the
        continuity tilt — the EFFECTIVE (a, c) after the positivity reparam
        (if any) and masked to the fitted term(s)."""
        th = self._smear_raw_to_effective(self.theta_smear)  # [n_eta, 2]
        return th[b_pm].reshape(b_pm.shape[0], -1)

    def _continuity_g(self, m, mk, eta_pm, phi_pm, q_pm, b_pm, rho, pt_pm):
        """First-order continuity sensitivities at ``(m, c)``:
        ``g_scale = −v′ − v·s`` ``[P,6]`` and ``g_smear = ½ κ (s′+s²)`` ``[P,4]``.
        ``pt_pm`` is the per-point pt (observed at the data mass; reconstructed
        from ``(m,c)`` on the normalisation grid)."""
        _, s, s_prime = self._flow_logp_score(m, mk)
        if self.scale_enabled:
            v, vprime = self._v_and_vprime(m, eta_pm, phi_pm, q_pm, b_pm, rho)
            g_scale = -vprime - v * s.unsqueeze(-1)
        else:
            g_scale = m.new_zeros((m.shape[0], N_THETA_SCALE_PM))
        if self.smearing_enabled:
            kappa = self._kappa_smear(m, pt_pm, eta_pm, phi_pm, q_pm)
            g_smear = 0.5 * kappa * (s_prime + s * s).unsqueeze(-1)
        else:
            g_smear = m.new_zeros((m.shape[0], N_THETA_SMEAR_PM))
        return g_scale, g_smear

    def _continuity_logZ(self, mk, eta_pm, phi_pm, q_pm, b_pm, rho,
                         theta_pm, n_grid: int = 32):
        """2nd-order cumulant log-normalisation per event:
        ``logZ ≈ E_{p₀}[δ] + ½ Var_{p₀}[δ|c]`` with ``δ = θ_pm · g(m,c)``.

        This is ``log E_{p₀}[e^δ]`` to ``O(δ³)``; the mean term corrects for the
        grid truncation/discretisation (analytically ``E_{p₀}[g]=0`` over full
        support, but not on a finite grid). ``g`` is detached (θ-independent,
        flow-frozen), so only ``θ_pm`` carries the gradient, giving the proper
        score centering ``∂_θ logZ = E[g] + Cov[g]·θ``. Moments are estimated by
        ``p₀``-weighted quadrature on an ``n_grid`` mass grid.
        ``theta_pm = [θ_scale_pm (6), θ_smear_pm (4)]`` ``[B,10]``.
        """
        B = mk.shape[0]
        dev, dt = mk.device, mk.dtype
        mg = torch.linspace(float(self.m_lo) + 1e-3, float(self.m_hi) - 1e-3,
                            n_grid, device=dev, dtype=dt)            # [G]
        # expand per-event conditioning across the grid, flatten to [B*G, ...]
        def rep(x):
            return x.unsqueeze(1).expand(B, n_grid, *x.shape[1:]).reshape(
                B * n_grid, *x.shape[1:])
        m_f = mg.unsqueeze(0).expand(B, n_grid).reshape(-1)          # [B*G]
        eta_f, phi_f, q_f, mk_f = rep(eta_pm), rep(phi_pm), rep(q_pm), rep(mk)
        b_f, rho_f = rep(b_pm), rep(rho)
        pt_f = self._reconstruct_pt(m_f, eta_f, phi_f, rho_f)
        with torch.no_grad():
            logp0_f = self.log_p_nominal(m_f, mk_f)                  # [B*G]
        gs, gsm = self._continuity_g(m_f, mk_f, eta_f, phi_f, q_f, b_f, rho_f, pt_f)
        g = torch.cat([gs, gsm], dim=-1).detach().reshape(B, n_grid, -1)  # [B,G,10]
        w = torch.softmax(logp0_f.reshape(B, n_grid), dim=-1)        # p₀ weights, Σ=1
        d = (g * theta_pm.unsqueeze(1)).sum(-1)                      # δ on grid [B,G]
        mean = (w * d).sum(-1)
        var = (w * d * d).sum(-1) - mean * mean
        return mean + 0.5 * var.clamp_min(0.0)                       # 2nd-order cumulant

    # ------------------------------------------------------------------
    # Stage-2 continuity data fit — #2 "forward-fold the flow's eval point"
    #
    #   p_θ(x) = E_{ε~N(0,1)}[ p₀(m'(ε)|c) / |G'(m'(ε))| ],
    #   x = m' + s_adv(m') + √(V(m'))·ε ,   G'(m') = ∂x/∂m'
    #     = 1 + s_adv'(m') + (V'/2√V)·ε                       (kernel Jacobian)
    #
    # s_adv(m') = Σ_k v_k(m')·θ_scale_k  (advective mass shift; v = analytic J),
    # V(m')     = Σ_k κ_k(m')·softplus(θ_smear)_k² ≥ 0  (smear variance; the
    #             effective softplus(θ) are qop STDs, so σ_qop² = a²+c²k²).
    # This evaluates the FROZEN flow only as point values at the source m'(ε)
    # (no flow derivatives), captures advection+smear to all orders in θ and the
    # x-variation of v, V (source-evaluation + Jacobian), and is normalised by
    # construction. v, κ are evaluated at the source via the pt∝m scaling at
    # fixed conditioning — swappable to a learned v/κ MLP without touching this.
    # softplus on θ_smear keeps V ≥ 0 (no ill-posed de-convolution / sharpening).
    # ------------------------------------------------------------------

    def _continuity_response(self, m_eval, m_obs, pt_obs, eta_pm, q_pm,
                             theta_scale_pm):
        """Advective mass shift ``s_adv`` at evaluation mass ``m_eval``
        (broadcasting against the per-event observables), for the analytic scale
        transform with ``pt(m_eval) = pt_obs · m_eval/m_obs``.

        ``v = (−½m, ½m k, −½m q pt)`` per muon (the corrected scale Jacobian).
        Returns ``s_adv`` shaped like ``m_eval``. (Replaceable by a learned MLP.)
        """
        scale = (m_eval / m_obs).unsqueeze(-1)             # [...,1]
        pt = pt_obs * scale                                # [...,2]
        k = 1.0 / pt
        mh = (0.5 * m_eval).unsqueeze(-1)                  # [...,1]
        dA = -mh                                           # charge-even (per muon)
        de = mh * k                                        # [...,2]
        dM = -mh * q_pm * pt                               # [...,2]
        v = torch.stack([dA[..., 0], de[..., 0], dM[..., 0],
                         dA[..., 0], de[..., 1], dM[..., 1]], dim=-1)
        return (v * theta_scale_pm).sum(-1)

    def _smear_mass_var(self, eta_pm, phi_pm, b_pm, pt_obs, m_eval, m_obs):
        """Per-event SIGNED m_ll variance added by the per-muon qop smear,
        evaluated at the mass ``m_eval`` (NOT a fixed reference): with
        ``pt(m_eval) = pt_obs·m_eval/m_obs`` (pt ∝ m at fixed angles, as in
        ``_continuity_response``) and ``σ²_qop,μ = a + c·k²``,
        ``V = (m_eval/2)² Σ_μ σ²_qop,μ / qop_μ²`` (``∂m/∂qop = −m/2qop``,
        ``1/qop² = pt²/sin²θ``) ``∝ a·m⁴ + c·m²``. Two-sided (V<0 = unsmear).
        Evaluated at the source m' inside ``_continuity_logp`` so the smear's
        mass-dependence enters the change-of-variables Jacobian (consistent with
        the advection). This is the diffusion 'time' of the probability flow."""
        pt = pt_obs * (m_eval / m_obs).unsqueeze(-1)       # pt(m_eval) [B,2]
        vq = self._qop_var_pm(eta_pm, phi_pm, b_pm, pt)    # [B,2] signed σ²_qop(pt)
        sinth = _sintheta_from_eta(eta_pm)
        inv_qop2 = (pt * pt) / (sinth * sinth)             # 1/qop² = pt²/sin²θ
        return (0.5 * m_eval) ** 2 * (vq * inv_qop2).sum(-1)  # [B] signed

    def _flow_score(self, m, mk):
        """Flow score ``∂_m log p₀(m | mk)``, via autograd. Differentiable w.r.t.
        ``m`` when ``m`` carries grad — so the change-of-variables Jacobian of the
        probability-flow smear picks up the local curvature ``∂²_m log p₀``.
        Otherwise returns a detached value (the source fixed-point path)."""
        if m.requires_grad:
            lp = self.log_p_nominal(m, mk)
            return torch.autograd.grad(lp.sum(), m, create_graph=True)[0]
        with torch.enable_grad():
            ml = m.detach().requires_grad_(True)
            lp = self.log_p_nominal(ml, mk)
            return torch.autograd.grad(lp.sum(), ml, create_graph=True)[0].detach()

    def _scale_source_rho_std(self, pt_obs, eta_pm, phi_pm, q_pm, b_pm):
        """Standardised SOURCE ρ from un-applying the scale only (no smear) —
        the smear is a pure mass-space transport, so it leaves ρ untouched.
        Returns ``[B]``."""
        pt1 = pt_obs
        if self.scale_enabled:
            AeM_pm = self._scale_AeM_pm(eta_pm, phi_pm, b_pm)
            dqop_s = self._delta_qop_analytic(AeM_pm, pt_obs, eta_pm, q_pm)
            pt1 = self._apply_scale_pt(pt_obs, eta_pm, q_pm, dqop_s, sign=-1.0)
        rho = (pt1[:, 0] - pt1[:, 1]) / (pt1[:, 0] + pt1[:, 1])
        idx = N_MUON_KIN - 1
        return (rho - self.muon_kin_mean[idx]) / self.muon_kin_std[idx]

    def _continuity_logp(self, m_obs, mk, pt_obs, eta_pm, phi_pm, q_pm, b_pm,
                         n_iter: int = 2):
        """``log p_θ(x|c)`` — dispatches on ``self.smear_operator``.

        ``"gh_convolution"``: EXACT stochastic Gaussian convolution via
        Gauss-Hermite quadrature (matches the per-muon qop fold operator that
        generates the pseudo-data; converges to the convolution by construction
        at any V). See ``_continuity_logp_gh``.

        ``"pf_ode"`` (default, doc below): deterministic probability-flow
        ODE — cheaper but over-broadens at large V/σ²."""
        if self.smear_operator == "gh_convolution":
            return self._continuity_logp_gh(
                m_obs, mk, pt_obs, eta_pm, phi_pm, q_pm, b_pm,
                n_gh=self.n_gh_nodes, n_iter=n_iter)
        return self._continuity_logp_pf_ode(
            m_obs, mk, pt_obs, eta_pm, phi_pm, q_pm, b_pm, n_iter=n_iter)

    def _continuity_logp_pf_ode(self, m_obs, mk, pt_obs, eta_pm, phi_pm, q_pm,
                                b_pm, n_iter: int = 2):
        """``log p_θ(x|c)``: the frozen nominal flow pushed through an exactly-
        normalized, INVERTIBLE TRANSPORT — a scale advection ``s_adv`` plus the
        smear as a score-driven PROBABILITY-FLOW displacement
        ``y ← y − (V/2n)·∂_m log p₀(y)`` (``smear_flow_steps`` Euler steps; the
        deterministic equivalent of a Gaussian qop smear of mass-variance V,
        broaden V>0 / sharpen V<0). Invert for the source m' by fixed point.

        Log-Jacobian — two forms (``self.jacobian_form``):

        * ``"softlog"`` (default): use the EXACT autograd Jacobian ``G' =
          dx/dm'`` of the N-step Euler forward map, fed through
          ``_softlog_below_floor`` — equal to ``log(G'.clamp_min(floor))`` in
          the physical region AND adds a C¹ quadratic barrier ``(floor−G')⁺²
          /(2·floor²)`` past the floor. The barrier value and slope are zero
          at the seam (so the physical region is unchanged) and grow
          quadratically past it, pulling G' back toward the physical region
          — replacing the hard clamp's flat-NLL basin (where the optimiser
          could drift into G' < 0 when V·∂²_m log p₀ is large at large |η|,
          with no gradient cost) with a gentle restoring force.
        * ``"exp"``: skip the autograd Jacobian and use the FROZEN-SCORE
          continuous-flow approximation ``log G' = log(1+s_adv'(m')) − V·∂²_m
          log p₀(m')/2`` — always finite, no floor needed. Approximates a
          DIFFERENT operator: assumes ∂²_m log p₀ is constant along the smear
          trajectory and is the analytic ``N→∞`` Euler-step limit of that
          frozen-score flow. Closer to the true Gaussian convolution at
          moderate V than 1-step Euler, but the assumption breaks near sharp
          features (the J/ψ peak crest) and the recovered θ may shift vs the
          'softlog' (different operator → different optimum). It also lacks
          the floor's natural cap on unphysical-sharpening rewards, so a
          large negative V is rewarded UNBOUNDEDLY by ``−log G' = +V·∂²/2``
          — only choose this when the V-too-large breakdown driving #1 is
          actually the dominant issue.

        The smear is a pure mass-space transport, so it leaves ρ untouched —
        only the scale's ρ shift is propagated to the conditioning."""
        B = m_obs.shape[0]
        theta_scale_pm = (self._scale_per_event(eta_pm, phi_pm, b_pm)
                          if self.scale_enabled
                          else m_obs.new_zeros((B, N_THETA_SCALE_PM)))
        n_step = max(1, int(self.smear_flow_steps))

        mk_src = mk
        if self.scale_enabled:
            mk_src = mk.clone()
            mk_src[..., N_MUON_KIN - 1] = self._scale_source_rho_std(
                pt_obs, eta_pm, phi_pm, q_pm, b_pm)

        def s_adv_of(me):
            return self._continuity_response(
                me, m_obs, pt_obs, eta_pm, q_pm, theta_scale_pm)

        def forward(mp):
            # scale advection, then the probability-flow smear (score displacement).
            # V (diffusion time) is evaluated at the SOURCE mass mp — so its
            # mass-dependence (V ∝ a·m⁴ + c·m²) enters the autograd Jacobian G'.
            y = mp + s_adv_of(mp)
            if self.smearing_enabled:
                V = self._smear_mass_var(eta_pm, phi_pm, b_pm, pt_obs, mp, m_obs)
                dt = V / (2.0 * n_step)
                for _ in range(n_step):
                    y = y - dt * self._flow_score(y, mk_src)
            return y

        # invert for the source m': fixed point  m' = m_obs − (forward(m') − m').
        # (Graph is built in training so m' carries the θ dependence, as for G'.)
        mp = m_obs.clone()
        for _ in range(n_iter):
            mp = m_obs - (forward(mp) - mp)
        # change-of-variables log-Jacobian: dispatch on jacobian_form.
        if self.jacobian_form == "exp":
            # Frozen-score continuous-flow approximation, no floor needed.
            log_Gp = self._log_jacobian_exp(mp, mk_src, s_adv_of, eta_pm, phi_pm,
                                            b_pm, pt_obs, m_obs)
        else:  # "softlog" (default): autograd Jacobian + tangent extension below floor.
            if mp.requires_grad:
                Gp = torch.autograd.grad(forward(mp).sum(), mp, create_graph=True)[0]
            else:
                with torch.enable_grad():
                    mpj = mp.detach().requires_grad_(True)
                    Gp = torch.autograd.grad(forward(mpj).sum(), mpj)[0].detach()
            log_Gp = _softlog_below_floor(Gp, SMEAR_GP_FLOOR)
        logp0 = self.log_p_nominal(mp, mk_src)
        log_p_theta = logp0 - log_Gp
        # Cap log_p_theta from ABOVE only (the overflow direction). The
        # downstream mixture in `data_nll_continuity` takes `.exp()` of this:
        # for jacobian_form='exp' the unbounded sharpening reward can drive
        # log_Gp → −∞ on a single event → log_p_theta → +∞ → `.exp()` → inf
        # → NaN loss. The physical log-density on this m-window lives in
        # ~[−30, +5], so cap = +50 leaves the operating regime untouched.
        # No LOWER cap: the softlog barrier deliberately drives log_p_theta
        # very negative to penalise unphysical G' < 0, and `.exp()` of a
        # large-negative number underflows cleanly to 0 (the mixture then
        # collapses to the Bernstein background — no NaN). nan_to_num catches
        # any residual NaN (rare, from autograd at boundary mp values).
        return torch.nan_to_num(log_p_theta.clamp(max=50.0), nan=0.0)

    def _continuity_logp_gh(self, m_obs, mk, pt_obs, eta_pm, phi_pm, q_pm,
                             b_pm, n_gh: int = 8, n_iter: int = 2):
        """``log p_θ(x|c)`` via the EXACT Gaussian convolution operator,
        implemented as Gauss-Hermite quadrature of the per-event source map:

            ``p_θ(x|c) = E_{ε~N(0,1)}[ p_0(m'(ε)|c_src) / |G'(m'(ε))| ]``
            ``       ≈ Σ_i W_i · p_0(m'_i | c_src,i) / |G'_i|``

        where, for each GH node ε = ξ_i:
            ``x = m'_i + s_adv(m'_i) + √V(m'_i)·ξ_i``  (forward map)
            ``G'(m') = 1 + s_adv'(m') + (V'(m')/(2√V(m')))·ε``  (autograd Jacobian)

        Equivalent — by construction — to the per-muon qop fold that generates
        the validation pseudo-data, so the closure-target curve (flow at
        injected θ) overlaps the pseudo-data at ALL V (no operator-level
        residual; only GH quadrature truncation, which is exponential in n_gh
        for smooth p_0). Cost: ``n_gh`` per-node `p_0` evaluations + the
        autograd through the source map.

        Requires V ≥ 0 (Gaussian variance is non-negative); we ``clamp_min(0)``
        the per-muon σ²_qop before the √V term. With smear_param_form='softplus'
        this is automatic; with 'linear' the clamp enforces it (the fit loses
        access to V < 0 sharpening, which is fine — that region is unphysical
        for a stochastic smear anyway). The ``jacobian_form`` switch is
        ignored here (the Jacobian is the source-map's exact autograd-derived
        ``G'`` — no PF-ODE-style score-flow Jacobian to soft-floor).

        The source ρ for the flow conditioning per GH node carries the smear's
        conditional-mean per-muon qop shift (``E[δqop_μ | δm] = (J^m_μ σ²_qop,μ
        /√V)·ε``, un-applied per node) — see ``_source_rho_std_gh``.
        """
        B = m_obs.shape[0]
        theta_scale_pm = (self._scale_per_event(eta_pm, phi_pm, b_pm)
                          if self.scale_enabled
                          else m_obs.new_zeros((B, N_THETA_SCALE_PM)))
        xi, logW = _gh_nodes(n_gh, m_obs.device, m_obs.dtype)            # [G], [G]
        G = xi.shape[0]

        # Broadcast all event conditioning with a singleton "GH node" dim.
        mo  = m_obs.unsqueeze(1)                                          # [B, 1]
        pto = pt_obs.unsqueeze(1)                                         # [B, 1, 2]
        etao = eta_pm.unsqueeze(1)                                        # [B, 1, 2]
        phio = phi_pm.unsqueeze(1)                                        # [B, 1, 2]
        qo  = q_pm.unsqueeze(1)                                           # [B, 1, 2]
        bpo = b_pm.unsqueeze(1).expand(B, G, b_pm.shape[-1])              # [B, G, 2]
        tsp = theta_scale_pm.unsqueeze(1)                                 # [B, 1, 6]
        xig = xi.view(1, G)                                               # [1, G]

        def resp(me):
            """Return (s_adv, V) at evaluation mass me [B, G]."""
            s_adv = self._continuity_response(me, mo, pto, etao, qo, tsp)
            if self.smearing_enabled:
                # _smear_mass_var: pt = pto*(me/mo).unsqueeze(-1) → [B,G,2];
                # broadcasts with etao, phio, bpo[B,G,2]. Returns V [B, G].
                V = self._smear_mass_var(etao, phio, bpo, pto, me, mo).clamp_min(0.0)
            else:
                V = me.new_zeros(me.shape)
            return s_adv, V

        def _smear_disp(Vt):
            """The √V·ε displacement per GH node; vanishes exactly if disabled."""
            if self.smearing_enabled:
                return Vt.sqrt() * xig
            return me_zero  # populated below before use

        # Pre-allocate the zero displacement (used only when smearing_enabled=False).
        me_zero = m_obs.new_zeros((B, G))

        # Fixed-point source solve  m'_i = m_obs − s_adv(m'_i) − √V(m'_i)·ξ_i.
        mp = mo.expand(B, G).clone()
        for _ in range(n_iter):
            s_adv, V = resp(mp)
            mp = mo - s_adv - _smear_disp(V)

        # Source-map Jacobian G' = ∂x/∂m' by autograd at the converged source.
        # In training (mp.requires_grad) we keep the graph so the gradient flows
        # through θ (which sources V, s_adv, and the dependence m'(θ)); in eval
        # we run under enable_grad on a fresh leaf.
        if mp.requires_grad:
            s_advj, Vj = resp(mp)
            Gx = (mp + s_advj + _smear_disp(Vj)).sum()
            Gp = torch.autograd.grad(Gx, mp, create_graph=True)[0]
        else:
            with torch.enable_grad():
                mp_j = mp.detach().requires_grad_(True)
                s_advj, Vj = resp(mp_j)
                Gx = (mp_j + s_advj + _smear_disp(Vj)).sum()
                Gp = torch.autograd.grad(Gx, mp_j)[0].detach()

        # Frozen-flow density at the per-node source points. ρ in the
        # conditioning is propagated per-node (scale un-applied + smear's
        # conditional-mean δqop un-applied per ε); η/φ are transform-invariant.
        mk_g = mk.unsqueeze(1).expand(B, G, mk.shape[-1]).clone()
        if self.scale_enabled or self.smearing_enabled:
            mk_g[..., N_MUON_KIN - 1] = self._source_rho_std_gh(
                m_obs, pt_obs, eta_pm, phi_pm, q_pm, b_pm, xi)
        logp0 = self.log_p_nominal(
            mp.reshape(-1), mk_g.reshape(B * G, -1)).reshape(B, G)

        # log p_θ(x) = logsumexp_i[ logW_i + log p_0(m'_i) − log|G'_i| ].
        log_terms = logW.view(1, G) + logp0 - torch.log(Gp.abs().clamp_min(1e-6))
        log_p_theta = torch.logsumexp(log_terms, dim=1)
        # Same upper-cap and NaN guard as the PF-ODE branch — see the
        # _continuity_logp_pf_ode tail for rationale.
        return torch.nan_to_num(log_p_theta.clamp(max=50.0), nan=0.0)

    def _source_rho_std_gh(self, m_obs, pt_obs, eta_pm, phi_pm, q_pm, b_pm, xig):
        """Standardised source ρ per GH node ``[B, G]`` for the
        ``gh_convolution`` operator's conditioning. Un-applies the scale (event-
        level, no GH dependence) and the smear's conditional-mean per-muon qop
        shift (per GH node ε = ξ_i, via ``E[δqop_μ | δm] = (J^m_μ σ²_qop,μ/√V)·ε``,
        with ``J^m_μ = ∂m/∂qop_μ = −m/(2 qop_μ)``). η, φ are exactly
        transform-invariant and don't need propagating."""
        B = m_obs.shape[0]
        G = xig.shape[-1]
        # 1) un-apply the scale (obs → truth), per event.
        pt1 = pt_obs
        if self.scale_enabled:
            AeM_pm = self._scale_AeM_pm(eta_pm, phi_pm, b_pm)
            dqop_s = self._delta_qop_analytic(AeM_pm, pt_obs, eta_pm, q_pm)
            pt1 = self._apply_scale_pt(pt_obs, eta_pm, q_pm, dqop_s, sign=-1.0)
        # 2) un-apply the smear's conditional-mean qop shift, per GH node.
        if self.smearing_enabled:
            sinth = _sintheta_from_eta(eta_pm)
            qop1 = q_pm * sinth / pt1                          # [B, 2]
            sig2 = self._qop_var_pm(
                eta_pm, phi_pm, b_pm, pt1).clamp_min(0.0)      # [B, 2] σ²_qop at source pt
            Jm = -0.5 * m_obs.unsqueeze(-1) / qop1             # [B, 2]  ∂m/∂qop
            V = (Jm * Jm * sig2).sum(-1, keepdim=True).clamp_min(1e-12)  # [B, 1]
            coef = Jm * sig2 / V.sqrt()                        # [B, 2]
            dqop_sm = coef.unsqueeze(1) * xig.view(1, G, 1)    # [B, G, 2]
            pt_src = self._apply_scale_pt(
                pt1.unsqueeze(1).expand(B, G, 2),
                eta_pm.unsqueeze(1).expand(B, G, 2),
                q_pm.unsqueeze(1).expand(B, G, 2),
                dqop_sm, sign=-1.0)                            # [B, G, 2]
            rho_src = (pt_src[..., 0] - pt_src[..., 1]) / (
                pt_src[..., 0] + pt_src[..., 1])
        else:
            r = (pt1[:, 0] - pt1[:, 1]) / (pt1[:, 0] + pt1[:, 1])
            rho_src = r.unsqueeze(1).expand(B, G)
        idx = N_MUON_KIN - 1
        return (rho_src - self.muon_kin_mean[idx]) / self.muon_kin_std[idx]

    def _log_jacobian_exp(self, mp, mk_src, s_adv_of, eta_pm, phi_pm, b_pm,
                          pt_obs, m_obs):
        """Frozen-score continuous-flow log-Jacobian:
        ``log G' = log(1 + s_adv'(m')) − V·∂²_m log p₀(m')/2``.

        Both ``s_adv'(m')`` and ``∂²_m log p₀(m')`` are taken via autograd at
        the SOURCE mass m' (frozen along the smear trajectory). Always finite
        — no floor — but a different operator approximation than the autograd
        Jacobian of the N-step Euler forward map (see _continuity_logp
        docstring). Live grads to θ are preserved when ``mp.requires_grad``.

        Both factors are guarded with ``nan_to_num``: ``∂²_m log p₀`` exposes
        the flow's tail singularities (where the Jacobian floor in the GF
        monotonic transform produces a step) more directly than the softlog
        path's autograd Gp (which combines everything through the chain rule
        — sharp pieces of the score and its derivative tend to cancel). A
        small minority of events at the m-window boundary can return inf/NaN
        d2, and even one such event poisons the batch mean. We mask those
        contributions to zero rather than dropping the events — the model is
        at the edge of its validity there anyway."""
        # ∂s_adv/∂m' (scale-advection Jacobian contribution, =0 if scale disabled).
        if self.scale_enabled:
            if mp.requires_grad:
                s = s_adv_of(mp)
                s_prime = torch.autograd.grad(
                    s.sum(), mp, create_graph=True, retain_graph=True)[0]
            else:
                with torch.enable_grad():
                    mpj = mp.detach().requires_grad_(True)
                    s_prime = torch.autograd.grad(
                        s_adv_of(mpj).sum(), mpj)[0].detach()
            log_J_scale = torch.nan_to_num(
                torch.log1p(s_prime), nan=0.0, posinf=0.0, neginf=0.0)
        else:
            log_J_scale = mp.new_zeros(mp.shape)
        # −V·∂²_m log p₀(m')/2 (smear contribution, =0 if smear disabled).
        if self.smearing_enabled:
            V = self._smear_mass_var(eta_pm, phi_pm, b_pm, pt_obs, mp, m_obs)
            if mp.requires_grad:
                score = self._flow_score(mp, mk_src)
                d2 = torch.autograd.grad(
                    score.sum(), mp, create_graph=True, retain_graph=True)[0]
            else:
                with torch.enable_grad():
                    mpj = mp.detach().requires_grad_(True)
                    score = self._flow_score(mpj, mk_src)
                    d2 = torch.autograd.grad(score.sum(), mpj)[0].detach()
            log_J_smear = torch.nan_to_num(
                -0.5 * V * d2, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            log_J_smear = mp.new_zeros(mp.shape)
        return log_J_scale + log_J_smear

    def _flow_log_cdf(self, m: torch.Tensor, mk: torch.Tensor) -> torch.Tensor:
        """``log F_0(m | mk)`` — the FLOW's CDF at observed mass ``m``.

        For the GF flow with standard-normal base, ``F_0(m|c) =
        Φ(T_mono(m_std|c))`` where ``T_mono`` is the conditional Gaussianisation
        flow's monotonic transform (zuko's ``dist.transform``) acting on the
        standardised mass. The standardisation is monotone increasing so the
        CDF transforms trivially: ``F_data(m) = F_data_std(m_std)``. Returns
        ``[B]`` log-CDF values clamped from below for log safety."""
        m_std = self._standardise_mll(m).clamp(
            -MLL_STD_FLOW_CLAMP, MLL_STD_FLOW_CLAMP).unsqueeze(-1)
        dist = self.flow.flow(mk)
        z = dist.transform(m_std).squeeze(-1)
        F = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        return F.clamp(min=1e-30).log()

    def _norm_correction_log_Z(self, m_obs, mk, pt_obs, eta_pm, phi_pm, q_pm,
                               b_pm, n_iter: int = 2) -> torch.Tensor:
        """Per-event ``log Z(θ;c) = log ∫_{m_lo}^{m_hi} p_θ(x|c) dx`` — the
        normalisation of the transformed-flow density on the m-window. Bare
        ``log p_θ(x)`` is normalised on ``T(window)``, NOT on the window; this
        correction restores the proper truncated-likelihood NLL by subtracting
        ``log Z`` from the per-event signal log-density. Dispatches on
        ``self.norm_correction``:

        * ``"none"``: return 0 (no correction).
        * ``"linear"``: leading-order boundary-leakage estimate
          ``1 − Z ≈ p_0(m_lo|c)·(m_lo − T(m_lo))_+ + p_0(m_hi|c)·(T(m_hi) − m_hi)_+``
          — only the positive parts of the boundary shift contribute (a
          narrowing T pushes mass INTO the window, no leakage). Cheap: 2
          boundary forward-map evals + 2 flow density evals per event.
        * ``"flow_cdf"``: EXACT via the flow's CDF
          ``Z(θ;c) = F_0(T⁻¹(m_hi)|c) − F_0(T⁻¹(m_lo)|c)``. Inverts T at the
          two boundary x-values by fixed-point (n_iter steps, mirroring
          ``_continuity_logp``) and evaluates F_0 via the GF monotonic
          transform + Φ. ~2× the per-event work of the bare density.

        When ``self.smear_operator == "gh_convolution"`` both 'linear' and
        'flow_cdf' route to the GH-specific exact formula
        ``Z = Σ_i W_i [F_0(m'_hi(ξ_i)) − F_0(m'_lo(ξ_i))]`` (per-GH-node
        boundary inversion + flow CDF) — the PF-ODE forward map used by the
        modes below is the wrong operator for GH, so the simple boundary
        formulae would give the wrong V-scaling. See
        ``_norm_correction_log_Z_gh``.
        """
        if self.norm_correction == "none":
            return m_obs.new_zeros(m_obs.shape)
        if self.smear_operator == "gh_convolution":
            return self._norm_correction_log_Z_gh(
                m_obs, mk, pt_obs, eta_pm, phi_pm, q_pm, b_pm)

        B = m_obs.shape[0]
        theta_scale_pm = (self._scale_per_event(eta_pm, phi_pm, b_pm)
                          if self.scale_enabled
                          else m_obs.new_zeros((B, N_THETA_SCALE_PM)))
        n_step = max(1, int(self.smear_flow_steps))

        mk_src = mk
        if self.scale_enabled:
            mk_src = mk.clone()
            mk_src[..., N_MUON_KIN - 1] = self._scale_source_rho_std(
                pt_obs, eta_pm, phi_pm, q_pm, b_pm)

        def s_adv_of(me):
            return self._continuity_response(
                me, m_obs, pt_obs, eta_pm, q_pm, theta_scale_pm)

        def forward(mp):
            y = mp + s_adv_of(mp)
            if self.smearing_enabled:
                V = self._smear_mass_var(eta_pm, phi_pm, b_pm, pt_obs, mp, m_obs)
                dt = V / (2.0 * n_step)
                for _ in range(n_step):
                    y = y - dt * self._flow_score(y, mk_src)
            return y

        m_lo = m_obs.new_full(m_obs.shape, float(self.m_lo))
        m_hi = m_obs.new_full(m_obs.shape, float(self.m_hi))

        if self.norm_correction == "linear":
            # 1 − Z ≈ Σ_boundary p_0(boundary)·max(0, boundary-shift). The
            # max() picks up only broadening (T pushing the source OUT); a
            # narrowing T has no leakage so contributes 0.
            t_lo = forward(m_lo)
            t_hi = forward(m_hi)
            left_leak = (m_lo - t_lo).clamp(min=0.0)
            right_leak = (t_hi - m_hi).clamp(min=0.0)
            log_p_lo = self.log_p_nominal(m_lo, mk_src)
            log_p_hi = self.log_p_nominal(m_hi, mk_src)
            leakage = (log_p_lo.exp() * left_leak
                       + log_p_hi.exp() * right_leak)
            # clamp at a sub-1 ceiling so log(1−leakage) stays finite even if
            # the linear estimate overshoots in pathological cases
            return torch.log1p(-leakage.clamp(0.0, 1.0 - 1e-6))

        # "flow_cdf": invert T at x = m_lo, m_hi by BISECTION on [m_lo, m_hi].
        # T is monotonic on the window (G' = 1 + V/(2σ²) > 0 at the peak), so
        # bisection converges unconditionally; the per-event fixed-point
        # `mp = x - (T(mp) - mp)` used in `_continuity_logp` is contractive
        # near a stable observation but becomes expansive / oscillates at the
        # boundaries when V/(2σ²) is large (large smear gradient there), so we
        # use bisection here instead. 24 iterations give ~22-bit precision on
        # the m-window — far below grid noise.
        n_bisect = 24
        @torch.no_grad()
        def bisect(target):
            lo = m_obs.new_full(m_obs.shape, float(self.m_lo))
            hi = m_obs.new_full(m_obs.shape, float(self.m_hi))
            for _ in range(n_bisect):
                mid = 0.5 * (lo + hi)
                t_mid = forward(mid)
                go_right = t_mid < target
                lo = torch.where(go_right, mid, lo)
                hi = torch.where(go_right, hi, mid)
            return 0.5 * (lo + hi)
        mp_lo = bisect(m_lo)
        mp_hi = bisect(m_hi)
        log_F_lo = self._flow_log_cdf(mp_lo, mk_src)
        log_F_hi = self._flow_log_cdf(mp_hi, mk_src)
        # Z = F_hi − F_lo; floor at eps in case bisection lands on the same
        # preimage (e.g., when T's image misses the window entirely).
        Z = (log_F_hi.exp() - log_F_lo.exp()).clamp(min=1e-30)
        return Z.log()

    def _norm_correction_log_Z_gh(self, m_obs, mk, pt_obs, eta_pm, phi_pm,
                                  q_pm, b_pm) -> torch.Tensor:
        """``log Z(θ;c)`` for the ``smear_operator='gh_convolution'`` path.

        The Gaussian-convolution operator is `x = m' + s_adv(m') + √V(m')·ε`,
        ε ~ N(0,1). Z over the m-window factors per GH node:

            ``Z = Σ_i W_i · [F_0(m'_hi(ξ_i)|c_src,i) − F_0(m'_lo(ξ_i)|c_src,i)]``

        where each m'_{lo,hi}(ξ_i) is the source mass at GH node ξ_i that maps
        to the corresponding boundary, found by BISECTION on [m_lo, m_hi] (the
        existing fixed-point at the boundary oscillates at large V for the
        same reason as in `_norm_correction_log_Z`'s flow_cdf branch; bisection
        is unconditional given T is monotonic in m'). Per-GH-node CDF
        evaluation uses the source ρ from `_source_rho_std_gh` for consistency
        with `_continuity_logp_gh`. Cost: 2·n_gh boundary bisections + 2·n_gh
        flow CDF evals per event — roughly 2× the GH density itself.
        """
        B = m_obs.shape[0]
        theta_scale_pm = (self._scale_per_event(eta_pm, phi_pm, b_pm)
                          if self.scale_enabled
                          else m_obs.new_zeros((B, N_THETA_SCALE_PM)))
        xi, logW = _gh_nodes(self.n_gh_nodes, m_obs.device, m_obs.dtype)
        G = xi.shape[0]

        mo = m_obs.unsqueeze(1)
        pto = pt_obs.unsqueeze(1)
        etao = eta_pm.unsqueeze(1)
        phio = phi_pm.unsqueeze(1)
        qo = q_pm.unsqueeze(1)
        bpo = b_pm.unsqueeze(1).expand(B, G, b_pm.shape[-1])
        tsp = theta_scale_pm.unsqueeze(1)
        xig = xi.view(1, G)

        def forward_at_eps(me):
            """T(me; ξ) for each (event, GH node). me [B, G] → [B, G]."""
            s_adv = self._continuity_response(me, mo, pto, etao, qo, tsp)
            if self.smearing_enabled:
                V = self._smear_mass_var(
                    etao, phio, bpo, pto, me, mo).clamp_min(0.0)
                return me + s_adv + V.sqrt() * xig
            return me + s_adv

        n_bisect = 24

        @torch.no_grad()
        def bisect(target_scalar: float):
            lo = m_obs.new_full((B, G), float(self.m_lo))
            hi = m_obs.new_full((B, G), float(self.m_hi))
            target = m_obs.new_full((B, G), float(target_scalar))
            for _ in range(n_bisect):
                mid = 0.5 * (lo + hi)
                t_mid = forward_at_eps(mid)
                go_right = t_mid < target
                lo = torch.where(go_right, mid, lo)
                hi = torch.where(go_right, hi, mid)
            return 0.5 * (lo + hi)

        mp_lo = bisect(float(self.m_lo))                   # [B, G]
        mp_hi = bisect(float(self.m_hi))                   # [B, G]

        # Per-GH-node source ρ (same construction as _continuity_logp_gh).
        mk_g = mk.unsqueeze(1).expand(B, G, mk.shape[-1]).clone()
        if self.scale_enabled or self.smearing_enabled:
            mk_g[..., N_MUON_KIN - 1] = self._source_rho_std_gh(
                m_obs, pt_obs, eta_pm, phi_pm, q_pm, b_pm, xi)
        log_F_lo = self._flow_log_cdf(
            mp_lo.reshape(-1), mk_g.reshape(B * G, -1)).reshape(B, G)
        log_F_hi = self._flow_log_cdf(
            mp_hi.reshape(-1), mk_g.reshape(B * G, -1)).reshape(B, G)
        # Per-node window mass × GH weights, then sum over nodes.
        node_window = (log_F_hi.exp() - log_F_lo.exp()).clamp_min(0.0)  # [B, G]
        W = logW.exp().view(1, G)                                       # [1, G]
        Z = (W * node_window).sum(dim=1).clamp(min=1e-30)               # [B]
        return Z.log()

    def data_nll_continuity(
        self,
        mll: torch.Tensor,
        pt_pm: torch.Tensor,
        eta_pm: torch.Tensor,
        phi_pm: torch.Tensor,
        q_pm: torch.Tensor,
        b_pm: torch.Tensor,
        muon_kin_std: torch.Tensor,
        is_data_mask: torch.Tensor,
        eps: float = 1e-30,
        n_iter: int = 2,
    ) -> torch.Tensor:
        """Per-event data NLL (``[B]``, unweighted) for stage 2.

        Signal density from the #2 direct evaluation (``_continuity_logp``):
        the frozen nominal flow forward-folded (advection + smear) by evaluating
        it at the source pre-images with the change-of-variables Jacobian — no
        flow derivatives, normalised by construction. Mixed with a degree-1
        Bernstein background via the MLP ``f(c)``. MC rows are ignored.
        """
        B = mll.shape[0]
        per = torch.zeros(B, dtype=mll.dtype, device=mll.device)
        data_idx = is_data_mask.nonzero(as_tuple=True)[0]
        if data_idx.numel() == 0:
            return per
        m = mll[data_idx]
        mk = muon_kin_std[data_idx]
        pt, eta, phi, q, b = (pt_pm[data_idx], eta_pm[data_idx], phi_pm[data_idx],
                              q_pm[data_idx], b_pm[data_idx])
        log_ps = self._continuity_logp(m, mk, pt, eta, phi, q, b, n_iter=n_iter)
        if self.norm_correction != "none":
            # Truncated-likelihood correction: the transformed-flow density is
            # naturally normalised on T(window), not on the window itself, so
            # log p_θ(x) carries a -log Z(θ;c) bias. Subtract per-event log Z
            # to restore the correct normalisation over the observation window.
            log_Z = self._norm_correction_log_Z(
                m, mk, pt, eta, phi, q, b, n_iter=n_iter)
            log_ps = log_ps - log_Z
        if self.background_enabled:
            f = self.f_data(mk)
            p0b, p1b = bernstein_d1(m, float(self.m_lo), float(self.m_hi))
            p_mix = f[:, 0] * p0b + f[:, 1] * p1b + f[:, 2] * log_ps.exp()
            per_data = -torch.log(p_mix.clamp_min(eps))
        else:
            # Background disabled (validation closure mode): the data branch
            # is pure signal — NLL = -log p_signal directly, no MLP / no
            # Bernstein. This removes the f_bkg ↔ smear degeneracy where the
            # MLP would otherwise absorb forward-|η| tails the signal can't
            # broaden into.
            per_data = -log_ps
        per = per.index_put((data_idx,), per_data, accumulate=False)
        return per
