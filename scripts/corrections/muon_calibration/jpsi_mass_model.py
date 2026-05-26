"""Mixture model + θ-conditioned flow for the unbinned J/ψ mass calibration.

Forward-folding design (both nuisances applied to MC, flow conditioned on
both; the θ gradients flow only through the flow's conditioning input):

* ``theta_scale`` ∈ R^{24×3}  — per-η-bin (A, e, M) muon scale nuisances.
* ``theta_smear`` ∈ R^{24×2}  — per-η-bin (a, c) muon smearing nuisances,
  ``σ²_qop_i = a²_{b_i} + c²_{b_i} · k²_i``.

The flow is a conditional density ``p(m | y, θ_scale_pm, σ_qop_pm)``.

MC branch (trains the flow's θ-dependence — NO θ gradient):
  Sample θ̃_scale ~ N(θ_scale.detach(), σ_scale²) and θ̃_smear ~
  N(θ_smear.detach(), σ_smear²) [σ from Adam's 2nd moment, per epoch].
  Apply BOTH transforms to the MC reco momenta at those *detached* sampled
  values — smear (qop Gaussian kernel) then scale (exact T_scale forward,
  or its linearised mass-shift; selected by ``linearize_scale``) — to get
  m_final, and condition the flow on (θ̃_scale_pm, σ_qop_pm). The detach
  means θ never gets a gradient from the transform; the flow simply learns
  how its conditioning maps to the transformed-MC density.

Data branch (fits θ — gradient via the conditioning only):
  The data mass is NOT transformed. The flow is evaluated at the observed
  m_obs, conditioned on the *live* model (θ_scale_pm, σ_qop_pm). The θ
  gradient flows purely through that conditioning input (the flow's learned
  θ-sensitivity), so the fit picks the θ whose transformed-MC template best
  matches the data. Mixed with a degree-1 Bernstein background via the MLP:

  data event:  p(m_obs | y, θ) = f_0(y) p_0 + f_1(y) p_1
                 + (1 − f_0 − f_1) · p_flow(m_obs | y, θ_scale_pm, σ_qop_pm)
  MC event:    p(m_final | y, θ̃) = p_flow(m_final | y, θ̃_scale_pm, σ̃_qop_pm)

``detach_flow_on_data`` (default on) additionally excludes the flow's
parameters from the data-branch gradient, so the template shape is pinned
to MC and the data branch only floats θ and the mixture — removing the
θ_scale ↔ flow-location degeneracy.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys

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


def _softplus_inv(y: float, floor: float = 1e-6) -> float:
    """Inverse of ``softplus``: raw such that ``softplus(raw) = y``.

    Used to initialise the unconstrained θ_smear parameter so its effective
    (softplus) value equals the requested ``smear_init``. ``y`` is floored at
    a tiny positive value (softplus⁻¹(0) = −∞), so an effective init of 0 maps
    to a large-negative raw → effective ≈ 0 (negligible smearing).
    """
    y = max(float(y), floor)
    return float(math.log(math.expm1(y)))


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


_GH_CACHE: dict = {}


def _gh_nodes(n: int, device, dtype):
    """Gauss–Hermite nodes/log-weights for ``E_{ε~N(0,1)}[f] ≈ Σ_i W_i f(ξ_i)``.

    ``∫ f(t) e^{-t²} dt ≈ Σ w_i^H f(t_i^H)`` ⇒ with ``ξ = √2 t^H`` and
    ``W = w^H/√π`` (Σ W = 1) we get the standard-normal expectation. Cached.
    Returns ``(ξ [n], logW [n])``.
    """
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
    """Unbinned J/ψ mass-fit model — θ-conditioned forward-folding design."""

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
        smear_init_a: float = 0.0,
        smear_init_c: float = 0.0,
        n_eta_bins: int = N_ETA_BINS,
        # Use the linearised mass-shift T_scale on MC (m += J·θ) instead of
        # the exact analytic qop transform. Only affects how the MC training
        # mass is generated; the conditioning (θ_scale_pm) is identical.
        linearize_scale: bool = False,
        # Fixed reference scales used to standardise the per-muon nuisance
        # conditioning (cond = θ_pm / ref). The raw params span ~1e-2…1e-5
        # across (A,e,M,a,c); dividing by their characteristic magnitudes
        # puts the flow's conditioning inputs at O(1), comparable to the
        # already-standardised y / muon_kin. Order: (A, e, M), (a, c).
        theta_scale_cond_scale: tuple = (1e-3, 1e-3, 1e-5),
        theta_smear_cond_scale: tuple = (1e-3, 1e-4),
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
        # When True, the flow's *parameters* receive no gradient from the
        # data branch (the flow is updated by the MC branch only). The data
        # branch still propagates gradient to the flow's conditioning inputs
        # — θ_scale_pm and σ_qop_pm — and to the MLP. This pins the signal
        # template's shape to the (trusted) MC, so the data branch can only
        # float θ and the mixture, breaking the θ_scale ↔ flow-location
        # degeneracy (an online "pretrain-flow-on-MC").
        detach_flow_on_data: bool = False,
        # When True, the MC-branch samples θ̃_scale / θ̃_smear around a FIXED
        # centre (the initial nuisances: θ_scale = 0, θ_smear = init) with a
        # fixed width, independent of the current model values — so the flow
        # learns a stable θ-conditional family over a fixed region rather than
        # one that follows the fit. When False (default), it samples around
        # ``θ.detach()`` (the moving model value).
        fixed_theta_sampling: bool = False,
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
        # zeroed *post-softplus* (``smear_param_mask``) so it contributes
        # exactly 0 to the σ_qop kernel and the conditioning, and receives no
        # gradient (inert). smear_init_a / smear_init_c then only set the init
        # of the fitted term.
        smear_fit_params: str = "both",
        # Whether the flow conditions on the per-muon nuisances (θ_scale_pm,
        # θ_smear_pm). True = the legacy forward-fold design (flow learns the
        # θ-dependence). False = the two-stage continuity design: the flow
        # models only the nominal shape p₀(m|muon_kin) at θ=0, and the
        # θ-dependence is supplied analytically (continuity equation, see
        # ``data_nll_continuity``). θ parameters still exist when scale/smear
        # are enabled — they're just not fed to the flow.
        theta_conditioning: bool = True,
    ):
        super().__init__()
        self.smearing_enabled = bool(smearing_enabled)
        self.scale_enabled = bool(scale_enabled)
        self.linearize_scale = bool(linearize_scale)
        self.detach_flow_on_data = bool(detach_flow_on_data)
        self.fixed_theta_sampling = bool(fixed_theta_sampling)
        self.qop_floor_frac = float(qop_floor_frac)
        self.theta_conditioning = bool(theta_conditioning)
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

        # Per-muon reference scales for standardising the nuisance
        # conditioning: [A_+,e_+,M_+,A_-,e_-,M_-] and [a_+,c_+,a_-,c_-].
        self.register_buffer(
            "theta_scale_cond_scale_buf",
            torch.tensor(list(theta_scale_cond_scale) * 2, dtype=torch.float32),
        )
        self.register_buffer(
            "theta_smear_cond_scale_buf",
            torch.tensor(list(theta_smear_cond_scale) * 2, dtype=torch.float32),
        )

        # Flow conditions on (muon_kin_std[, theta_scale_pm][, theta_smear_pm]).
        # Each θ block is dropped from the conditioner when disabled, so the
        # architecture matches what we actually feed it. With
        # ``theta_conditioning=False`` no θ block is fed at all — the flow sees
        # only muon_kin (the two-stage continuity design).
        n_flow_cond = N_MUON_KIN
        if self.theta_conditioning:
            n_flow_cond += (N_THETA_SCALE_PM if self.scale_enabled else 0)
            n_flow_cond += (N_THETA_SMEAR_PM if self.smearing_enabled else 0)
        flow_inner = build_flow(
            n_features=1,
            n_cond=n_flow_cond,
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
        self.theta_smear = nn.Parameter(
            torch.full(
                (n_eta_bins, N_THETA_SMEAR),
                0.0,
                dtype=torch.float32,
            )
        )
        # Apply user-specified init. theta_smear is unconstrained; we store
        # softplus⁻¹(smear_init) so the *effective* (softplus) init equals the
        # requested physical value.
        with torch.no_grad():
            self.theta_smear[:, 0].fill_(_softplus_inv(smear_init_a))
            self.theta_smear[:, 1].fill_(_softplus_inv(smear_init_c))

        # Fixed sampling centres for ``fixed_theta_sampling`` — the *initial*
        # nuisance values (θ_scale = 0, θ_smear = softplus⁻¹(smear_init)).
        # When that flag is on, the MC-branch samples θ̃ around these fixed
        # references with a fixed width, instead of around the (moving) model
        # values — so the flow's learned θ-conditional family is independent
        # of the fit trajectory. Non-persistent: reconstructed from smear_init
        # at __init__, so old checkpoints (without these buffers) still load.
        self.register_buffer(
            "theta_scale_sample_center", self.theta_scale.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "theta_smear_sample_center", self.theta_smear.detach().clone(),
            persistent=False,
        )

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

    def _scale_per_event(self, theta_scale: torch.Tensor, b_pm: torch.Tensor) -> torch.Tensor:
        """Look up ``[B, 6] = (A_+, e_+, M_+, A_-, e_-, M_-)`` from a θ_scale parameter."""
        return theta_scale[b_pm].reshape(b_pm.shape[0], -1)

    def _smear_per_event(self, theta_smear: torch.Tensor, b_pm: torch.Tensor) -> torch.Tensor:
        """Look up the per-event effective ``[B, 4] = (a_+, c_+, a_-, c_-)``.

        ``theta_smear`` is the *unconstrained* parameter; the effective
        (physical, ≥0) smearing terms are ``softplus(theta_smear)``. The same
        softplus(a), softplus(c) feed the smear kernel (``sigma_qop_pm``) and
        this conditioning, so they are consistent and there is no sign
        degeneracy (softplus is monotone — the old a², c² kernel was even in
        the sign of (a, c); softplus removes that).

        A non-fitted term (per ``smear_fit_params``) is zeroed *post-softplus*
        via ``smear_param_mask``, so it contributes exactly 0 to both the
        kernel and the conditioning, and — being a multiply by 0 — receives no
        gradient (the optimizer leaves it inert).
        """
        eff = F.softplus(theta_smear[b_pm]) * self.smear_param_mask  # [B,2,2]
        return eff.reshape(b_pm.shape[0], -1)

    def sigma_qop_pm(
        self, theta_smear: torch.Tensor, pt_pm: torch.Tensor, eta_pm: torch.Tensor, b_pm: torch.Tensor
    ) -> torch.Tensor:
        """Per-muon σ_qop = sqrt((m_a·softplus(a_b))² + (m_c·softplus(c_b))² k²).

        ``theta_smear`` is unconstrained; the effective a, c = softplus(·) ≥ 0.
        A non-fitted term is zeroed post-softplus by ``smear_param_mask``
        (``m_a, m_c`` ∈ {0,1}), so it drops out of the kernel entirely.
        Returns ``[B, 2]``.
        """
        a_pm = F.softplus(theta_smear[b_pm, 0]) * self.smear_param_mask[0]
        c_pm = F.softplus(theta_smear[b_pm, 1]) * self.smear_param_mask[1]
        k_pm = 1.0 / pt_pm
        var = a_pm * a_pm + c_pm * c_pm * (k_pm * k_pm)
        return torch.sqrt(var.clamp_min(1e-24))

    def effective_theta_smear(self) -> torch.Tensor:
        """``softplus(theta_smear)`` masked to the fitted terms — the physical
        (a, c) ≥ 0 per η-bin actually used by the kernel and conditioning. A
        non-fitted term reads back as exactly 0."""
        return F.softplus(self.theta_smear) * self.smear_param_mask

    # ------------------------------------------------------------------
    # T_scale (analytic + linearized)
    # ------------------------------------------------------------------

    def _delta_qop_analytic(
        self,
        theta_scale: torch.Tensor,
        pt_pm: torch.Tensor,
        eta_pm: torch.Tensor,
        q_pm: torch.Tensor,
        b_pm: torch.Tensor,
    ) -> torch.Tensor:
        """Analytic δqop per muon (matches ``calculateQopUnc`` in
        ``muon_calibration.hpp``)::

            δqop_i = q_i · sinθ_i · [(A_i − e_i k_i) k_i + q_i M_i]

        where (A, e, M) are looked up at the muon's η-bin.
        """
        sintheta = _sintheta_from_eta(eta_pm)
        k_pm = 1.0 / pt_pm
        A_pm = theta_scale[b_pm, 0]
        e_pm = theta_scale[b_pm, 1]
        M_pm = theta_scale[b_pm, 2]
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

        The ``q_i`` lives on ``M`` (charge-odd), not on ``A``/``e`` — verified
        by finite-differencing the exact ``T_scale`` transform. Returns
        ``[B, 6]`` ordered ``(A_+, e_+, M_+, A_-, e_-, M_-)``; independent of
        ``b_pm`` (those route the gradient to ``theta_scale`` via scatter).
        """
        k_pm = 1.0 / pt_pm
        m_half = 0.5 * mll
        # [B, 2] per (+, −) for each of A, e, M.
        dA = -m_half.unsqueeze(-1).expand_as(q_pm)
        de = +m_half.unsqueeze(-1) * k_pm
        dM = -m_half.unsqueeze(-1) * q_pm * pt_pm
        # Order: (A_+, e_+, M_+, A_-, e_-, M_-).
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

    def _build_flow_cond(
        self,
        muon_kin_std: torch.Tensor,
        theta_scale_pm: "torch.Tensor | None",
        theta_smear_pm: "torch.Tensor | None",
    ) -> torch.Tensor:
        """``[muon_kin_std[, θ_scale_pm/ref][, θ_smear_pm/ref]]``.

        ``muon_kin_std`` is the leak-free kinematic conditioning (η_±, cos/sin
        φ_±, ρ). Each nuisance enters as raw per-muon parameters standardised
        by its fixed reference scale: ``θ_scale_pm = (A_+,e_+,M_+,A_-,e_-,M_-)``
        and ``θ_smear_pm = (a_+,c_+,a_-,c_-)``. θ_scale_pm is dropped when scale
        is off, θ_smear_pm when smearing is off (matching ``n_flow_cond``).
        With ``theta_conditioning=False`` only ``muon_kin_std`` is returned (the
        two-stage continuity design — the flow never sees θ).
        """
        parts = [muon_kin_std]
        if self.theta_conditioning:
            if self.scale_enabled and theta_scale_pm is not None:
                parts.append(theta_scale_pm / self.theta_scale_cond_scale_buf)
            if self.smearing_enabled and theta_smear_pm is not None:
                parts.append(theta_smear_pm / self.theta_smear_cond_scale_buf)
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    def _standardise_mll(self, mll: torch.Tensor) -> torch.Tensor:
        return (mll - self.mll_mean_buf) / self.mll_std_buf

    # ------------------------------------------------------------------
    # Signal log-density at observed coords (data branch)
    # ------------------------------------------------------------------

    def log_p_nominal(
        self, mll_obs: torch.Tensor, muon_kin_std_obs: torch.Tensor
    ) -> torch.Tensor:
        """``log p₀(m | muon_kin)`` — the θ=0 nominal flow density (1/GeV).

        The stage-1 target: a plain conditional density on the (uncorrected)
        reco mass, conditioned on the leak-free kinematics only. Requires
        ``theta_conditioning=False``.
        """
        cond = self._build_flow_cond(muon_kin_std_obs, None, None)
        mll_std = self._standardise_mll(mll_obs).clamp(
            -MLL_STD_FLOW_CLAMP, MLL_STD_FLOW_CLAMP
        )
        return self.flow(mll_std.unsqueeze(-1), cond) - self.mll_log_scale

    def log_p_signal_data(
        self,
        mll_obs: torch.Tensor,
        pt_pm_obs: torch.Tensor,
        eta_pm: torch.Tensor,
        b_pm: torch.Tensor,
        y_event_std_obs: torch.Tensor,
        muon_kin_std_obs: torch.Tensor,
    ) -> torch.Tensor:
        """``log p_signal(m_obs | y_obs, θ_scale, θ_smear)`` per data event.

        The data mass is **not** transformed. The flow is evaluated at the
        observed m_obs, conditioned on the *live* model nuisances
        (θ_scale_pm and θ_smear_pm). The θ_scale / θ_smear gradients flow
        purely through this conditioning input — the flow's learned
        θ-sensitivity, trained on the forward-folded MC branch. There is
        no change-of-variables Jacobian because the observable is not
        remapped.
        """
        theta_scale_pm = (
            self._scale_per_event(self.theta_scale, b_pm)
            if self.scale_enabled else None
        )
        theta_smear_pm = (
            self._smear_per_event(self.theta_smear, b_pm)
            if self.smearing_enabled else None
        )
        cond = self._build_flow_cond(
            muon_kin_std_obs, theta_scale_pm, theta_smear_pm
        )
        mll_std = self._standardise_mll(mll_obs).clamp(
            -MLL_STD_FLOW_CLAMP, MLL_STD_FLOW_CLAMP
        )
        log_p_std = self.flow(mll_std.unsqueeze(-1), cond)
        return log_p_std - self.mll_log_scale

    # ------------------------------------------------------------------
    # Signal log-density for MC: forward-fold smear + scale, condition on
    # the *sampled* (detached) nuisances so the flow learns the θ-dependence
    # ------------------------------------------------------------------

    def log_p_signal_mc(
        self,
        pt_pm: torch.Tensor,
        eta_pm: torch.Tensor,
        phi_pm: torch.Tensor,
        q_pm: torch.Tensor,
        b_pm: torch.Tensor,
        y_event_std_obs: torch.Tensor,
        muon_kin_std_obs: torch.Tensor,
        scale_noise_sigma: "float | torch.Tensor" = 0.0,
        smear_noise_sigma: "float | torch.Tensor" = 0.0,
    ) -> torch.Tensor:
        """``log p_flow(m_final | y, θ̃_scale_pm, θ̃_smear_pm)`` per MC event.

        Both nuisances are forward-applied to the MC reco momenta at
        *sampled, detached* values θ̃ ~ N(θ.detach(), σ²) (σ scalar, or a
        per-parameter Tensor from Adam's 2nd moment — see ``_adaptive_sigma``
        in the trainer; σ=0 → sample exactly at θ.detach()). The flow is
        conditioned on those same sampled values and evaluated at the
        resulting mass, so the gradient reaching θ from this branch is zero
        — the branch trains the flow's conditional shape only. Order:
        **scale first, then smear** — exact scale applies the analytic qop
        forward to the momenta and the smear kernel acts on those scaled
        momenta; the linearised scale (``m += J·θ``, J at the pre-smear
        kinematics) is added to the smeared mass.
        """
        # ---- sampled, detached nuisances. Centre = the *fixed* init
        # reference when ``fixed_theta_sampling`` (sampling independent of the
        # model values), else the (moving) model value θ.detach(). Each
        # nuisance is dropped entirely when its block is disabled.
        if self.scale_enabled:
            scale_center = (
                self.theta_scale_sample_center if self.fixed_theta_sampling
                else self.theta_scale.detach()
            )
            theta_scale_eff = (
                scale_center + torch.randn_like(self.theta_scale) * scale_noise_sigma
                if _noise_active(scale_noise_sigma) else scale_center
            )
            theta_scale_pm_eff = self._scale_per_event(theta_scale_eff, b_pm)
        else:
            theta_scale_eff = None
            theta_scale_pm_eff = None

        if self.smearing_enabled:
            smear_center = (
                self.theta_smear_sample_center if self.fixed_theta_sampling
                else self.theta_smear.detach()
            )
            theta_smear_eff = (
                smear_center
                + torch.randn_like(self.theta_smear) * smear_noise_sigma
                * self.smear_param_mask
                if _noise_active(smear_noise_sigma) else smear_center
            )
            theta_smear_pm_eff = self._smear_per_event(theta_smear_eff, b_pm)
            eps_pm = torch.randn_like(pt_pm)
        else:
            theta_smear_eff = None
            theta_smear_pm_eff = None

        # ---- forward fold: scale FIRST (if enabled), then smear (if enabled).
        # Exact scale acts on the momenta (smear then acts on the scaled
        # momenta); the linearised scale is a mass-shift added at the end with
        # J evaluated at the pre-smear kinematics. When scale is disabled the
        # momenta pass through unscaled and the shift is 0.
        pt_cur = pt_pm
        scale_shift = 0.0
        if self.scale_enabled:
            if self.linearize_scale:
                mll_pre = _event_mll(pt_pm, eta_pm, phi_pm)
                j_pm = self.jacobian_mll_linearized(mll_pre, pt_pm, q_pm, b_pm)
                scale_shift = (j_pm * theta_scale_pm_eff).sum(-1)
            else:
                delta_qop = self._delta_qop_analytic(
                    theta_scale_eff, pt_pm, eta_pm, q_pm, b_pm
                )
                pt_cur = self._apply_scale_pt(
                    pt_pm, eta_pm, q_pm, delta_qop, sign=+1.0
                )

        if self.smearing_enabled:
            sigma_pm = self.sigma_qop_pm(theta_smear_eff, pt_cur, eta_pm, b_pm)
            pt_cur = self.apply_smear_pt(pt_cur, eta_pm, q_pm, sigma_pm, eps_pm)

        mll_final = _event_mll(pt_cur, eta_pm, phi_pm) + scale_shift

        mll_final_std = self._standardise_mll(mll_final).clamp(
            -MLL_STD_FLOW_CLAMP, MLL_STD_FLOW_CLAMP
        )
        cond = self._build_flow_cond(
            muon_kin_std_obs, theta_scale_pm_eff, theta_smear_pm_eff
        )
        log_p_std = self.flow(mll_final_std.unsqueeze(-1), cond)
        return log_p_std - self.mll_log_scale

    # ------------------------------------------------------------------
    # MLP coefficients (data branch only)
    # ------------------------------------------------------------------

    def f_data(self, muon_kin_std: torch.Tensor) -> torch.Tensor:
        return self.mlp(muon_kin_std)

    # ------------------------------------------------------------------
    # Per-event NLL (two branches), weighted sum
    # ------------------------------------------------------------------

    def event_nll(
        self,
        mll: torch.Tensor,
        pt_pm: torch.Tensor,
        eta_pm: torch.Tensor,
        phi_pm: torch.Tensor,
        q_pm: torch.Tensor,
        b_pm: torch.Tensor,
        y_event_std: torch.Tensor,
        muon_kin_std: torch.Tensor,
        is_data_mask: torch.Tensor,
        scale_noise_sigma: "float | torch.Tensor" = 0.0,
        smear_noise_sigma: "float | torch.Tensor" = 0.0,
        eps: float = 1e-30,
        mc_only: bool = False,
    ) -> torch.Tensor:
        """Per-event ``-ln p`` (shape ``[B]``, unweighted).

        MC branch: forward-fold smear+scale at *sampled, detached* θ̃ and
        condition the flow on the same θ̃ — trains the flow's conditional
        shape, no θ gradient. Data branch: evaluate the flow at the
        untransformed observed mass, conditioned on the *live* θ; the
        θ_scale / θ_smear gradients flow only through that conditioning,
        and the MLP sets the Bernstein-background mixture.

        Implementation: **each branch's flow is evaluated only on its
        own events**. We don't use ``torch.where(is_data, nll_data,
        nll_mc)`` over forward-evaluated-everywhere branches because the
        backward through the zuko GF base transform's ``jacobian.log()``
        is ``grad_output / jacobian``; the zeroed-grad rows on the
        masked-out branch hit ``0 / ~0 = NaN`` when the GMM-CDF jacobian
        happens to be tiny (typical at random init for inputs between
        the mixture components). Subset-and-scatter avoids that —
        gradient only flows where the value was actually consumed.

        Debug shortcut: when ``mc_only=True`` the data branch is skipped
        entirely — no MLP / Bernstein compute, no ``log_p_signal_data``.
        """
        B = mll.shape[0]
        per = torch.zeros(B, dtype=mll.dtype, device=mll.device)

        is_mc = ~is_data_mask
        # MC branch on MC-event subset only.
        if bool(is_mc.any()):
            mc_idx = is_mc.nonzero(as_tuple=True)[0]
            log_p_mc = self.log_p_signal_mc(
                pt_pm=pt_pm[mc_idx],
                eta_pm=eta_pm[mc_idx],
                phi_pm=phi_pm[mc_idx],
                q_pm=q_pm[mc_idx],
                b_pm=b_pm[mc_idx],
                y_event_std_obs=y_event_std[mc_idx],
                muon_kin_std_obs=muon_kin_std[mc_idx],
                scale_noise_sigma=scale_noise_sigma,
                smear_noise_sigma=smear_noise_sigma,
            )
            per = per.index_put((mc_idx,), -log_p_mc, accumulate=False)

        if mc_only:
            return per

        # Data branch on data-event subset only.
        is_data = is_data_mask
        if bool(is_data.any()):
            data_idx = is_data.nonzero(as_tuple=True)[0]
            # Optionally exclude the flow's parameters from the data-branch
            # gradient: the flow template is then fixed (MC-trained) w.r.t.
            # the data NLL, while θ_scale / θ_smear (via the conditioning)
            # and the MLP still get their data gradients.
            _flow_freeze = (
                self.flow.parameters() if self.detach_flow_on_data else []
            )
            with _freeze_param_grads(_flow_freeze):
                log_ps = self.log_p_signal_data(
                    mll_obs=mll[data_idx],
                    pt_pm_obs=pt_pm[data_idx],
                    eta_pm=eta_pm[data_idx],
                    b_pm=b_pm[data_idx],
                    y_event_std_obs=y_event_std[data_idx],
                    muon_kin_std_obs=muon_kin_std[data_idx],
                )

            f = self.f_data(muon_kin_std[data_idx])
            p0, p1 = bernstein_d1(mll[data_idx], float(self.m_lo), float(self.m_hi))
            p_mix = f[:, 0] * p0 + f[:, 1] * p1 + f[:, 2] * torch.exp(log_ps)
            nll_data = -torch.log(p_mix.clamp_min(eps))
            per = per.index_put((data_idx,), nll_data, accumulate=False)

        return per

    def nll(
        self,
        mll: torch.Tensor,
        pt_pm: torch.Tensor,
        eta_pm: torch.Tensor,
        phi_pm: torch.Tensor,
        q_pm: torch.Tensor,
        b_pm: torch.Tensor,
        y_event_std: torch.Tensor,
        muon_kin_std: torch.Tensor,
        is_data_mask: torch.Tensor,
        w: torch.Tensor,
        scale_noise_sigma: "float | torch.Tensor" = 0.0,
        smear_noise_sigma: "float | torch.Tensor" = 0.0,
        eps: float = 1e-30,
        mc_only: bool = False,
    ) -> torch.Tensor:
        """Weighted summed per-event NLL. Scalar.

        With ``mc_only=True`` the per-row NLL is ``-log p_mc`` for every
        row; combine with caller-side data-row weight masking to get
        an MC-only loss.
        """
        per = self.event_nll(
            mll=mll,
            pt_pm=pt_pm,
            eta_pm=eta_pm,
            phi_pm=phi_pm,
            q_pm=q_pm,
            b_pm=b_pm,
            y_event_std=y_event_std,
            muon_kin_std=muon_kin_std,
            is_data_mask=is_data_mask,
            scale_noise_sigma=scale_noise_sigma,
            smear_noise_sigma=smear_noise_sigma,
            eps=eps,
            mc_only=mc_only,
        )
        return (w * per).sum()

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
        Requires ``theta_conditioning=False`` (flow conditions on muon_kin).
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
        continuity tilt. ``theta_smear`` is treated as a plain (signed) variance
        increment here (init 0), masked to the fitted term(s)."""
        th = self.theta_smear * self.smear_param_mask  # [n_eta, 2]
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
                             theta_scale_pm, theta_smear_eff_pm):
        """Advective mass shift ``s_adv`` and smear variance ``V`` at evaluation
        mass ``m_eval`` (broadcasting against the per-event observables), for the
        analytic transform with ``pt(m_eval) = pt_obs · m_eval/m_obs``.

        ``v = (−½m, ½m k, −½m q pt)`` per muon (the corrected scale Jacobian);
        ``κ_a = (∂m/∂qop)² = (m/2qop)²``, ``κ_c = κ_a k²``. Returns
        ``(s_adv, V)`` shaped like ``m_eval``. (Replaceable by a learned MLP.)
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
        s_adv = (v * theta_scale_pm).sum(-1)
        sintheta = _sintheta_from_eta(eta_pm)
        qop = q_pm * sintheta / pt                         # [...,2]
        dm_dqop = -(0.5 * m_eval).unsqueeze(-1) / qop      # −m/(2 qop)
        ka = dm_dqop * dm_dqop
        kc = ka * (k * k)
        kappa = torch.stack([ka[..., 0], kc[..., 0], ka[..., 1], kc[..., 1]], dim=-1)
        # V = κ·σ_qop² = κ_a·a_eff² + κ_c·c_eff².  ``theta_smear_eff_pm`` is the
        # effective (a, c) = softplus(θ), which are qop STDs (matching
        # σ_qop² = a²+c²k² in sigma_qop_pm), so they enter V SQUARED. Using them
        # linearly inflated V by ~1/a_eff: with κ_a ~ (m/2qop)² ~ 10³ and the
        # default a_eff=1e-3, a linear V gave √V ~ 1 GeV (≫ the 0.36 GeV window),
        # flinging the GH source points out of the flow's support and driving
        # the fit to a non-finite density.
        V = (kappa * theta_smear_eff_pm.pow(2)).sum(-1).clamp_min(0.0)
        return s_adv, V

    def _continuity_logp(self, m_obs, mk, pt_obs, eta_pm, q_pm, b_pm,
                         n_gh: int = 5, n_iter: int = 2):
        """``log p_θ(m_obs | c)`` per event via the #2 direct evaluation."""
        theta_scale_pm = (self._scale_per_event(self.theta_scale, b_pm)
                          if self.scale_enabled
                          else self.theta_scale.new_zeros((b_pm.shape[0], N_THETA_SCALE_PM)))
        theta_smear_eff = (self._smear_per_event(self.theta_smear, b_pm)
                           if self.smearing_enabled
                           else self.theta_smear.new_zeros((b_pm.shape[0], N_THETA_SMEAR_PM)))
        xi, logW = _gh_nodes(n_gh, m_obs.device, m_obs.dtype)             # [G],[G]
        B = m_obs.shape[0]; G = xi.shape[0]
        mo = m_obs.unsqueeze(1)                                           # [B,1]
        pto = pt_obs.unsqueeze(1); etao = eta_pm.unsqueeze(1)             # [B,1,2]
        qo = q_pm.unsqueeze(1)
        tsp = theta_scale_pm.unsqueeze(1); tse = theta_smear_eff.unsqueeze(1)
        xig = xi.view(1, G)

        def resp(me):  # me: [B,G]
            return self._continuity_response(me, mo, pto, etao, qo, tsp, tse)

        # √V·ε smear term, with NO ε-floor: softplus(θ_smear) > 0 makes the
        # smear variance strictly positive whenever smearing is enabled, so √V
        # and its m'-gradient are finite. When smearing is disabled V ≡ 0 and the
        # term is dropped entirely (√0·ε = 0 exactly) — which also avoids
        # autograd's 0·∞ from differentiating the bare √0 in G' below (V≡0 is a
        # constant in m', so ∂√V/∂m' would be inf·0 = NaN without this).
        def _smear(Vt):
            return Vt.sqrt() * xig if self.smearing_enabled else 0.0
        # fixed-point source solve  m' = m_obs − s_adv(m') − √V(m')·ε
        mp = mo.expand(B, G).clone()
        for _ in range(n_iter):
            s_adv, V = resp(mp)
            mp = mo - s_adv - _smear(V)
        # change-of-variables Jacobian G'(m') = ∂x/∂m' by autograd in m'.
        # The m-derivative is over the cheap analytic response (s_adv, V), not
        # the flow, so this double-autograd is cheap. In the fit (grad on) we
        # differentiate the *non-detached* source so G' carries θ's FULL
        # dependence — explicit (s_adv', V') AND implicit (through m'(θ)); the
        # latter matters for the smear gradient. Under no_grad (diagnostics) we
        # only need the value, so a fresh leaf under enable_grad suffices.
        if mp.requires_grad:
            s_advj, Vj = resp(mp)
            Gx = (mp + s_advj + _smear(Vj)).sum()
            Gp = torch.autograd.grad(Gx, mp, create_graph=True)[0]
        else:
            with torch.enable_grad():
                mp_j = mp.detach().requires_grad_(True)
                s_advj, Vj = resp(mp_j)
                Gx = (mp_j + s_advj + _smear(Vj)).sum()
                Gp = torch.autograd.grad(Gx, mp_j)[0].detach()
        # frozen-flow density at the source points (POINT values only)
        mk_g = mk.unsqueeze(1).expand(B, G, mk.shape[-1]).reshape(B * G, -1)
        logp0 = self.log_p_nominal(mp.reshape(-1), mk_g).reshape(B, G)
        log_terms = logW.view(1, G) + logp0 - torch.log(Gp.abs().clamp_min(1e-6))
        return torch.logsumexp(log_terms, dim=1)

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
        n_gh: int = 5,
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
        pt, eta, q, b = (pt_pm[data_idx], eta_pm[data_idx],
                         q_pm[data_idx], b_pm[data_idx])
        log_ps = self._continuity_logp(m, mk, pt, eta, q, b, n_gh=n_gh, n_iter=n_iter)
        f = self.f_data(mk)
        p0b, p1b = bernstein_d1(m, float(self.m_lo), float(self.m_hi))
        p_mix = f[:, 0] * p0b + f[:, 1] * p1b + f[:, 2] * log_ps.exp()
        per = per.index_put((data_idx,), -torch.log(p_mix.clamp_min(eps)),
                            accumulate=False)
        return per
