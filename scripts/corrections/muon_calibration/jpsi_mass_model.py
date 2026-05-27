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
# bounded to a physical σ_qop. a ~ O(1) ≈ a few-×10% m_ll-variance smear.
SMEAR_VAR_SCALE_A = 1e-7
SMEAR_VAR_SCALE_C = 1e-6

# Invertibility floor on the probability-flow smear Jacobian G' = dx/dm'. A valid
# forward (broadening) transport has G' > 0; the floor catches the fold/over-
# sharpen region (V·∂²_m log p₀ large) gracefully, mirroring the old (1+s) ≥ 0.05.
SMEAR_GP_FLOOR = 0.05


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
        # Number of Euler steps integrating the smear's probability-flow ODE in
        # the density (``_continuity_logp``). 1 = first-order (single score
        # displacement); more steps integrate the score-driven flow more finely
        # (smaller per-step Jacobian → more robust + accurate) at a higher
        # nested-autograd cost. Frozen p₀ score per step (exact diffusion to
        # first order in V).
        smear_flow_steps: int = 1,
    ):
        super().__init__()
        self.smearing_enabled = bool(smearing_enabled)
        self.scale_enabled = bool(scale_enabled)
        self.qop_floor_frac = float(qop_floor_frac)
        self.smear_flow_steps = max(1, int(smear_flow_steps))
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

    def effective_theta_smear(self) -> torch.Tensor:
        """Per-η-bin effective smear (a, c), masked to the fitted terms. These
        are the signed per-muon qop-resolution VARIANCE coefficients:
        ``σ²_qop,μ = a + c·k²`` (k = 1/pt), two-sided — positive = broaden,
        negative = unsmear (sharpen). The SAME (a, c) drive both the per-muon
        qop fold and the mass-density stretch (see ``_qop_var_pm``)."""
        return self.theta_smear * self.smear_param_mask

    def _qop_var_pm(self, b_pm: torch.Tensor, pt_pm: torch.Tensor) -> torch.Tensor:
        """Per-muon SIGNED qop-resolution variance
        ``σ²_qop,μ = a_b·SCALE_A + c_b·SCALE_C·k_μ²`` (k = 1/pt), from the fitted
        (a, c) ∈ O(1). Signed = two-sided (un)smearing. ``a``, ``c`` are COMBINED
        here, before any clipping. Returns ``[B, 2]``."""
        eff = self.effective_theta_smear()                 # [n_bins, 2], masked
        a_pm = eff[b_pm, 0] * SMEAR_VAR_SCALE_A
        c_pm = eff[b_pm, 1] * SMEAR_VAR_SCALE_C
        k2 = (1.0 / pt_pm) ** 2
        return a_pm + c_pm * k2                             # [B, 2], signed

    def fold_sigma_qop_pm(
        self, pt_pm: torch.Tensor, eta_pm: torch.Tensor, b_pm: torch.Tensor
    ) -> torch.Tensor:
        """Per-muon σ_qop for the validation FOLD, from the fitted (a, c): the
        combined qop variance ``σ²_qop = a + c·k²`` CLIPPED AT 0 *after*
        combining (a stochastic Gaussian qop kick can only broaden, so the
        unsmearing region σ² < 0 → no kick). Returns ``[B, 2]``."""
        return torch.sqrt(self._qop_var_pm(b_pm, pt_pm).clamp_min(0.0))

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

    def _smear_mass_var(self, b_pm, pt_pm, eta_pm):
        """Per-event SIGNED m_ll variance added by the per-muon qop smear
        ``σ²_qop,μ = a + c·k²``: ``V = (μ/2)² Σ_μ σ²_qop,μ / qop_μ²`` (``∂m/∂qop =
        −m/2qop``, ``1/qop² = pt²/sin²θ``). Two-sided (V<0 = unsmear). This is the
        diffusion "time" of the probability-flow smear in ``_continuity_logp``."""
        vq = self._qop_var_pm(b_pm, pt_pm)                 # [B,2] signed σ²_qop
        sinth = _sintheta_from_eta(eta_pm)
        inv_qop2 = (pt_pm * pt_pm) / (sinth * sinth)       # 1/qop² = pt²/sin²θ
        mu = self.mll_mean_buf
        return (0.5 * mu) ** 2 * (vq * inv_qop2).sum(-1)    # [B] signed

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

    def _scale_source_rho_std(self, pt_obs, eta_pm, q_pm, b_pm):
        """Standardised SOURCE ρ from un-applying the scale only (no smear) —
        used by the 'width' smear, whose deterministic mass stretch leaves ρ
        untouched. Returns ``[B]``."""
        pt1 = pt_obs
        if self.scale_enabled:
            dqop_s = self._delta_qop_analytic(self.theta_scale, pt_obs, eta_pm, q_pm, b_pm)
            pt1 = self._apply_scale_pt(pt_obs, eta_pm, q_pm, dqop_s, sign=-1.0)
        rho = (pt1[:, 0] - pt1[:, 1]) / (pt1[:, 0] + pt1[:, 1])
        idx = N_MUON_KIN - 1
        return (rho - self.muon_kin_mean[idx]) / self.muon_kin_std[idx]

    def _continuity_logp(self, m_obs, mk, pt_obs, eta_pm, q_pm, b_pm,
                         n_iter: int = 2):
        """``log p_θ(x|c)``: the frozen nominal flow pushed through an exactly-
        normalized, INVERTIBLE TRANSPORT — a scale advection ``s_adv`` plus the
        smear as a score-driven PROBABILITY-FLOW displacement
        ``y ← y − (V/2n)·∂_m log p₀(y)`` (``smear_flow_steps`` Euler steps; the
        deterministic equivalent of a Gaussian qop smear of mass-variance V,
        broaden V>0 / sharpen V<0). Invert for the source m' by fixed point and
        divide by the autograd Jacobian ``G' = dx/dm'`` — which carries the local
        curvature ``∂²_m log p₀`` (vs the old Gaussian-shape global stretch).
        ``G'`` is floored (SMEAR_GP_FLOOR) for invertibility; the floor only bites
        where V·∂²_m log p₀ would fold the map. The smear is a pure mass-space
        transport, so it leaves ρ untouched — only the scale's ρ shift is
        propagated to the conditioning."""
        B = m_obs.shape[0]
        theta_scale_pm = (self._scale_per_event(self.theta_scale, b_pm)
                          if self.scale_enabled
                          else self.theta_scale.new_zeros((B, N_THETA_SCALE_PM)))
        V = (self._smear_mass_var(b_pm, pt_obs, eta_pm) if self.smearing_enabled
             else m_obs.new_zeros(B))                       # diffusion time (signed)
        n_step = max(1, int(self.smear_flow_steps))
        dt = V / (2.0 * n_step)

        mk_src = mk
        if self.scale_enabled:
            mk_src = mk.clone()
            mk_src[..., N_MUON_KIN - 1] = self._scale_source_rho_std(
                pt_obs, eta_pm, q_pm, b_pm)

        def s_adv_of(me):
            return self._continuity_response(
                me, m_obs, pt_obs, eta_pm, q_pm, theta_scale_pm)

        def forward(mp):
            # scale advection, then the probability-flow smear (score displacement)
            y = mp + s_adv_of(mp)
            if self.smearing_enabled:
                for _ in range(n_step):
                    y = y - dt * self._flow_score(y, mk_src)
            return y

        # invert for the source m': fixed point  m' = m_obs − (forward(m') − m').
        # (Graph is built in training so m' carries the θ dependence, as for G'.)
        mp = m_obs.clone()
        for _ in range(n_iter):
            mp = m_obs - (forward(mp) - mp)
        # change-of-variables Jacobian G' = dx/dm' (autograd in m').
        if mp.requires_grad:
            Gp = torch.autograd.grad(forward(mp).sum(), mp, create_graph=True)[0]
        else:
            with torch.enable_grad():
                mpj = mp.detach().requires_grad_(True)
                Gp = torch.autograd.grad(forward(mpj).sum(), mpj)[0].detach()
        logp0 = self.log_p_nominal(mp, mk_src)
        return logp0 - torch.log(Gp.clamp_min(SMEAR_GP_FLOOR))

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
        pt, eta, q, b = (pt_pm[data_idx], eta_pm[data_idx],
                         q_pm[data_idx], b_pm[data_idx])
        log_ps = self._continuity_logp(m, mk, pt, eta, q, b, n_iter=n_iter)
        f = self.f_data(mk)
        p0b, p1b = bernstein_d1(m, float(self.m_lo), float(self.m_hi))
        p_mix = f[:, 0] * p0b + f[:, 1] * p1b + f[:, 2] * log_ps.exp()
        per = per.index_put((data_idx,), -torch.log(p_mix.clamp_min(eps)),
                            accumulate=False)
        return per
