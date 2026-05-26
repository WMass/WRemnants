"""Train the unbinned per-event J/ψ mass-fit calibration.

End-to-end driver:
  1. Discover Arrow shards (MC + data) produced by ``jpsi_mass_fit_snapshot.py``.
  2. Compute per-column standardisation stats over the full dataset.
  3. Build the :class:`JpsiMassMixtureModel` — a θ-conditioned flow with
     per-η-bin scale (A, e, M) and smearing (a, c) nuisances.
  4. Iterate the joint MLE: the MC branch forward-folds smear+scale at
     sampled, detached θ̃ to train the flow's conditional shape; the data
     branch fits θ through the flow's conditioning. Until early stop / epoch
     limit.
  5. Optionally, compute the plug-in (observed) Fisher information w.r.t.
     ``theta_scale`` at the converged point and persist the covariance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List

import numpy as np
import torch
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from jpsi_mass_arrow_loader import (
    JpsiMassArrowLoader,
    JpsiMassPreprocStats,
    compute_jpsi_mass_stats,
    discover_shards,
)
from jpsi_mass_model import JpsiMassMixtureModel


# ---------------------------------------------------------------------------
# Stats persistence
# ---------------------------------------------------------------------------


def _stats_to_dict(s: JpsiMassPreprocStats) -> dict:
    return {
        "mll_mean": float(s.mll_mean),
        "mll_std": float(s.mll_std),
        "y_event_mean": s.y_event_mean.tolist(),
        "y_event_std": s.y_event_std.tolist(),
        "muon_kin_mean": s.muon_kin_mean.tolist(),
        "muon_kin_std": s.muon_kin_std.tolist(),
        "eta_edges": s.eta_edges.tolist(),
        "m_lo": float(s.m_lo),
        "m_hi": float(s.m_hi),
    }


def _stats_from_dict(d: dict) -> JpsiMassPreprocStats:
    return JpsiMassPreprocStats(
        mll_mean=float(d["mll_mean"]),
        mll_std=float(d["mll_std"]),
        y_event_mean=np.asarray(d["y_event_mean"], dtype=np.float32),
        y_event_std=np.asarray(d["y_event_std"], dtype=np.float32),
        muon_kin_mean=np.asarray(d["muon_kin_mean"], dtype=np.float32),
        muon_kin_std=np.asarray(d["muon_kin_std"], dtype=np.float32),
        eta_edges=np.asarray(d["eta_edges"], dtype=np.float64),
        m_lo=float(d["m_lo"]),
        m_hi=float(d["m_hi"]),
    )


# ---------------------------------------------------------------------------
# Per-batch loss helpers
# ---------------------------------------------------------------------------


def _make_amp(precision: str, device: str):
    """Return ``(autocast_ctx_factory, scaler)`` for ``--precision``.

    Matches the convention in ``train_muon_response_flow.py``:
      * fp32 → autocast disabled, no GradScaler.
      * bf16 → bfloat16 autocast, GradScaler disabled (bf16 has the
        same exponent range as fp32; loss scaling is unnecessary).
      * fp16 → float16 autocast + enabled GradScaler for loss scaling.
    """
    if device.startswith("cuda"):
        amp_device_type = "cuda"
    elif device.startswith("xpu"):
        amp_device_type = "xpu"
    else:
        amp_device_type = "cpu"
    if precision == "fp32":
        amp_dtype = torch.float32
        amp_enabled = False
    elif precision == "bf16":
        amp_dtype = torch.bfloat16
        amp_enabled = True
    elif precision == "fp16":
        amp_dtype = torch.float16
        amp_enabled = True
    else:
        raise ValueError(f"unknown precision {precision!r}")
    amp_ctx = lambda: torch.amp.autocast(
        device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled,
    )
    scaler = torch.amp.GradScaler(
        amp_device_type, enabled=(precision == "fp16"),
    )
    return amp_ctx, scaler


def _move_batch(batch: dict, device: str) -> dict:
    return {
        k: v.to(device, non_blocking=device.startswith("cuda"))
        for k, v in batch.items()
    }


def _lr_str(optim: torch.optim.Optimizer) -> str:
    """Compact current-lr string for the progress bar: a single value if all
    param groups share an lr, else the per-group lrs joined by '/'."""
    lrs = [g["lr"] for g in optim.param_groups]
    if len(set(lrs)) == 1:
        return f"{lrs[0]:.2g}"
    return "/".join(f"{x:.2g}" for x in lrs)


def _make_scheduler(args, optim, epochs):
    """Build the LR scheduler per ``--lr-schedule``. ``plateau`` reduces lr by
    ``--lr-reduce-factor`` when the val metric stalls for ``--lr-reduce-patience``
    epochs (down to ``--min-lr``); ``cosine`` decays lr→min_lr over ``epochs``;
    ``none`` keeps it fixed. Returns ``(scheduler | None, kind)``."""
    kind = getattr(args, "lr_schedule", "none")
    if kind == "plateau":
        # threshold_mode="abs": detect a plateau by the SAME absolute NLL
        # decrease the early-stop uses (val < best - patience_threshold).
        # The default "rel" mode tests val < best*(1-threshold), which with a
        # NEGATIVE NLL (a log-density) moves the bar toward zero — i.e. a flat
        # or slightly-worse epoch still "improves" — so the LR would never
        # reduce while the absolute-threshold early-stop fires anyway.
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", factor=args.lr_reduce_factor,
            patience=args.lr_reduce_patience, min_lr=args.min_lr,
            threshold=args.patience_threshold, threshold_mode="abs"), kind
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=max(1, epochs), eta_min=args.min_lr), kind
    return None, "none"


# ---------------------------------------------------------------------------
# Adaptive σ from Adam's second moment (diagonal-Fisher proxy)
# ---------------------------------------------------------------------------


def _adaptive_sigma(
    param: torch.nn.Parameter,
    optim: torch.optim.Optimizer,
    *,
    sigma_init: float,
    enabled: bool,
    warmup_steps: int,
    sigma_min: float,
    sigma_max: float,
    scale: float,
) -> "float | torch.Tensor":
    """Per-parameter σ ≈ scale / √(bias-corrected Adam v_i).

    Adam's ``exp_avg_sq`` is an EMA of (∂L/∂θ_i)² — an unbiased
    diagonal-Fisher proxy up to a constant. Sampling θ̃ ~ N(θ, σ²)
    with σ matched to 1/√I_θθ puts the noise at the Cramér-Rao-natural
    width, so the flow / DSM net see training points exactly where the
    posterior actually varies.

    Falls back to the fixed ``sigma_init`` when:
      * ``enabled=False`` (CLI toggle off),
      * Adam has no state yet (no step taken),
      * step < ``warmup_steps`` (Adam's v hasn't populated meaningfully),
      * bias correction would divide by zero,
      * the resulting σ has NaN/Inf (per-element fallback).

    Clamps to ``[sigma_min, sigma_max]`` to prevent (a) σ → 0 collapse
    near sharp optima and (b) σ → ∞ for parameters that have never seen
    gradient (e.g. η-bins not yet hit by a batch — v stays 0 forever
    for those).
    """
    if not enabled:
        return sigma_init
    state = optim.state.get(param, None)
    if state is None or "exp_avg_sq" not in state:
        return sigma_init
    step_val = state.get("step", 0)
    if isinstance(step_val, torch.Tensor):
        step_val = float(step_val.item())
    else:
        step_val = float(step_val)
    if step_val < float(warmup_steps):
        return sigma_init

    # Find this param's group to read β₂.
    beta2 = 0.999
    for g in optim.param_groups:
        if any(p is param for p in g["params"]):
            beta2 = float(g.get("betas", (0.9, 0.999))[1])
            break
    bias_corr = 1.0 - beta2 ** step_val
    if bias_corr <= 0.0:
        return sigma_init

    v_hat = state["exp_avg_sq"] / bias_corr
    sigma = scale / torch.sqrt(v_hat + 1e-30)
    sigma = sigma.clamp(min=sigma_min, max=sigma_max)
    # Per-element NaN/Inf safety: anything not finite reverts to the fixed σ.
    # ``sigma_init`` may be a scalar or a per-component tensor (e.g. the
    # (σ_A, σ_e, σ_M) scale vector), which broadcasts over sigma's last dim.
    if torch.is_tensor(sigma_init):
        fb = sigma_init.to(device=sigma.device, dtype=sigma.dtype).broadcast_to(sigma.shape)
    else:
        fb = torch.full_like(sigma, float(sigma_init))
    sigma = torch.where(torch.isfinite(sigma), sigma, fb)
    return sigma


def _nll_step(
    model: JpsiMassMixtureModel,
    batch: dict,
    scale_noise_sigma: "float | torch.Tensor" = 0.0,
    smear_noise_sigma: "float | torch.Tensor" = 0.0,
    mc_only: bool = False,
) -> torch.Tensor:
    """Per-batch summed weighted NLL."""
    return model.nll(
        mll=batch["mll"],
        pt_pm=batch["pt_pm"],
        eta_pm=batch["eta_pm"],
        phi_pm=batch["phi_pm"],
        q_pm=batch["q_pm"],
        b_pm=batch["b_pm"],
        y_event_std=batch["y_event_std"],
        muon_kin_std=batch["muon_kin_std"],
        is_data_mask=batch["is_data_mask"],
        w=batch["w"],
        scale_noise_sigma=scale_noise_sigma,
        smear_noise_sigma=smear_noise_sigma,
        mc_only=mc_only,
    )


def _nll_components(
    model: JpsiMassMixtureModel,
    batch: dict,
    scale_noise_sigma: "float | torch.Tensor" = 0.0,
    smear_noise_sigma: "float | torch.Tensor" = 0.0,
    mc_only: bool = False,
    eps: float = 1e-30,
):
    """Per-batch weighted NLL split into the data and MC (simulation)
    branches. Returns ``(total_sum, data_sum, data_w, mc_sum, mc_w)`` — all
    scalar tensors. ``total_sum = data_sum + mc_sum`` is the training
    objective numerator (identical to ``_nll_step``); the splits are for
    reporting and for the data-only early-stopping metric.
    """
    per = model.event_nll(
        mll=batch["mll"],
        pt_pm=batch["pt_pm"],
        eta_pm=batch["eta_pm"],
        phi_pm=batch["phi_pm"],
        q_pm=batch["q_pm"],
        b_pm=batch["b_pm"],
        y_event_std=batch["y_event_std"],
        muon_kin_std=batch["muon_kin_std"],
        is_data_mask=batch["is_data_mask"],
        scale_noise_sigma=scale_noise_sigma,
        smear_noise_sigma=smear_noise_sigma,
        eps=eps,
        mc_only=mc_only,
    )
    w = batch["w"]
    wp = w * per
    is_data = batch["is_data_mask"]
    is_mc = ~is_data
    data_sum = wp[is_data].sum()
    mc_sum = wp[is_mc].sum()
    data_w = w[is_data].sum()
    mc_w = w[is_mc].sum()
    return data_sum + mc_sum, data_sum, data_w, mc_sum, mc_w


def _maybe_mc_only_batch(batch: dict, mc_only: bool) -> "dict | None":
    """If ``mc_only``, drop data rows from the batch. Returns ``None``
    when the filtered batch is empty (caller should skip the step)."""
    if not mc_only:
        return batch
    mc_mask = ~batch["is_data_mask"]
    if not bool(mc_mask.any()):
        return None
    B = batch["mll"].shape[0]
    return {
        k: (v[mc_mask] if v.shape[:1] == (B,) else v)
        for k, v in batch.items()
    }


def _epoch_metrics(
    model: JpsiMassMixtureModel,
    loader: JpsiMassArrowLoader,
    device: str,
    *,
    progress: bool = True,
    desc: str = "val",
    mc_only: bool = False,
    amp_ctx=None,
    scale_noise_sigma: "float | torch.Tensor" = 0.0,
    smear_noise_sigma: "float | torch.Tensor" = 0.0,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Weighted-mean NLL over ``loader``, split by branch.

    Returns ``(data_nll, data_w, mc_nll, mc_w)`` — the per-unit-weight mean
    NLL of the data branch and of the MC (simulation) branch separately. The
    caller uses the *data* component for best-model / early-stopping when
    data is present.

    Validation evaluates under the *same* (σ̃_scale, σ̃_smear) noise
    distribution as training (``scale_noise_sigma`` / ``smear_noise_sigma``
    match what the per-batch train step used). RNG is seeded with ``seed``
    at the start of the pass
    and restored at the end, so the noise pattern is deterministic
    across epochs (no Monte Carlo fluctuation in the val curve) and
    doesn't perturb the training RNG sequence.

    Without these two fixes train_nll vs val_nll were not directly
    comparable: training averaged log-density over noise around
    ``model.theta_smear``, val evaluated at the noiseless centre — by
    Jensen's inequality the latter is always ≤ the former, leaving a
    persistent (and physically meaningless) gap of ~½·Var_δ[log p].
    """
    # Save + seed RNG (CPU + this CUDA device, if applicable).
    cpu_state = torch.random.get_rng_state()
    cuda_state = None
    cuda_avail = device.startswith("cuda") and torch.cuda.is_available()
    if cuda_avail:
        try:
            cuda_state = torch.cuda.get_rng_state(device)
        except Exception:
            cuda_state = None
    torch.manual_seed(seed)
    if cuda_avail:
        torch.cuda.manual_seed_all(seed)

    data_sum = 0.0
    data_w = 0.0
    mc_sum = 0.0
    mc_w = 0.0
    model.eval()
    bar = tqdm(loader, desc=desc, leave=False, disable=not progress, unit="batch")
    _ctx = amp_ctx if amp_ctx is not None else (lambda: torch.amp.autocast(
        device_type=("cuda" if device.startswith("cuda") else "cpu"),
        enabled=False,
    ))
    try:
        with torch.no_grad():
            for batch in bar:
                batch = _move_batch(batch, device)
                batch = _maybe_mc_only_batch(batch, mc_only)
                if batch is None:
                    continue
                with _ctx():
                    _, d_sum, d_w, m_sum, m_w = _nll_components(
                        model, batch,
                        scale_noise_sigma=scale_noise_sigma,
                        smear_noise_sigma=smear_noise_sigma,
                        mc_only=mc_only,
                    )
                data_sum += float(d_sum.item()); data_w += float(d_w.item())
                mc_sum += float(m_sum.item()); mc_w += float(m_w.item())
                bar.set_postfix_str(
                    f"data={data_sum / max(data_w, 1e-30):+.4f} "
                    f"mc={mc_sum / max(mc_w, 1e-30):+.4f}"
                )
        bar.close()
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)

    model.train()
    return (
        data_sum / max(data_w, 1e-30), data_w,
        mc_sum / max(mc_w, 1e-30), mc_w,
    )


# ---------------------------------------------------------------------------
# Fisher information at the optimum
# ---------------------------------------------------------------------------


def compute_fisher_info(
    model: JpsiMassMixtureModel,
    loader: JpsiMassArrowLoader,
    device: str,
) -> torch.Tensor:
    """Plug-in (observed) Fisher information w.r.t. ``theta_scale`` only.

    θ_scale enters the (data-branch) NLL through the flow's conditioning
    input, so the observed Hessian comes from a standard autograd
    double-backward. The flow / MLP / θ_smear are frozen; we evaluate at
    the noiseless conditioning (σ̃=0) — the fit point.
    """
    model.eval()
    for p in model.flow.parameters():
        p.requires_grad_(False)
    for p in model.mlp.parameters():
        p.requires_grad_(False)
    model.theta_smear.requires_grad_(False)
    model.theta_scale.requires_grad_(True)

    n_theta = model.theta_scale.numel()
    grad_flat = torch.zeros(n_theta, device=device, dtype=torch.float32)

    seen_rows = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        model.zero_grad(set_to_none=True)
        nll = _nll_step(model, batch, scale_noise_sigma=0.0, smear_noise_sigma=0.0)
        g = torch.autograd.grad(nll, model.theta_scale, create_graph=True)[0]
        grad_flat = grad_flat + g.flatten()
        seen_rows += int(batch["mll"].shape[0])
    if seen_rows == 0:
        raise RuntimeError(
            "compute_fisher_info: loader yielded zero events. The fisher "
            "split is empty — check --val-fraction and shard row counts."
        )
    if not grad_flat.requires_grad:
        raise RuntimeError(
            "compute_fisher_info: accumulated gradient has no grad_fn. "
            f"This usually means no data events were seen (saw {seen_rows} "
            "rows, but theta_scale only enters the loss via the data "
            "branch). Are MC and data both present in the fisher split?"
        )

    H = torch.zeros((n_theta, n_theta), device=device, dtype=torch.float32)
    for i in range(n_theta):
        retain = i < (n_theta - 1)
        H_row = torch.autograd.grad(
            grad_flat[i], model.theta_scale, retain_graph=retain
        )[0].flatten()
        H[i] = H_row.detach()

    H = 0.5 * (H + H.T)
    return H.view(24, 3, 24, 3).detach().cpu()


def _smear_active_cols(model) -> List[int]:
    """Column indices of ``theta_smear`` actually fit (smear_param_mask != 0).
    The non-fit column is zeroed post-softplus → identically zero gradient, so
    it must be excluded from the Fisher Hessian (else a singular row/col)."""
    return [c for c in range(model.theta_smear.shape[1])
            if float(model.smear_param_mask[c]) != 0.0]


def compute_fisher_info_continuity(
    model: JpsiMassMixtureModel,
    loader: JpsiMassArrowLoader,
    device: str,
    *,
    mc_as_data: bool = False,
    n_gh: int = 5,
    n_iter: int = 2,
    progress: bool = True,
):
    """Observed (plug-in) Fisher information for the two-stage continuity fit,
    over ``theta_scale`` + the ACTIVE ``theta_smear`` columns jointly, with the
    flow and the background MLP held FIXED (option 1: conditional / fixed-φ).

    H = Σ_events w · ∂²(−ln p_mixture)/∂θ² at the fit point, on the data branch
    (``data_nll_continuity`` — the objective stage 2 actually minimises, NOT the
    legacy ``model.nll``). ``theta_smear`` is the raw (pre-softplus) parameter.

    Accumulated per batch (build the batch gradient with ``create_graph`` then
    immediately take its ∂/∂θ rows and free the graph) so memory stays at one
    batch regardless of dataset size; cost is O(N_batches · n_active) backward
    passes. With ``mc_as_data`` the data branch is the MC (``~is_data_mask``)
    rows — the validation-mode pseudo-data.

    Returns ``(H [n_act, n_act] on cpu, layout dict)``.
    """
    model.eval()
    for p in model.flow.parameters():
        p.requires_grad_(False)
    for p in model.mlp.parameters():
        p.requires_grad_(False)
    model.theta_scale.requires_grad_(model.scale_enabled)
    model.theta_smear.requires_grad_(model.smearing_enabled)

    # Parameters to differentiate + the active flat layout. theta_scale: all
    # active; theta_smear: only the fitted column(s).
    params: list = []
    blocks: list = []          # (name, numel, active_local_indices)
    if model.scale_enabled:
        params.append(model.theta_scale)
        blocks.append(("scale", model.theta_scale.numel(),
                       list(range(model.theta_scale.numel()))))
    smear_cols = _smear_active_cols(model) if model.smearing_enabled else []
    if smear_cols:
        params.append(model.theta_smear)
        n_eta, n_comp = model.theta_smear.shape
        active = [b * n_comp + c for b in range(n_eta) for c in smear_cols]
        blocks.append(("smear", model.theta_smear.numel(), active))
    if not params:
        raise RuntimeError(
            "compute_fisher_info_continuity: no free parameters "
            "(--disable-scale and --disable-smearing / no active smear term).")

    # Active index into the concatenated flat [scale_flat | smear_flat] vector.
    active_idx, offset = [], 0
    for _name, numel, act in blocks:
        active_idx += [offset + a for a in act]
        offset += numel
    active_idx = torch.tensor(active_idx, dtype=torch.long, device=device)
    n_act = int(active_idx.numel())

    H = torch.zeros((n_act, n_act), device=device, dtype=torch.float32)
    grad = torch.zeros(n_act, device=device, dtype=torch.float32)  # Σ ∂(NLL)/∂θ
    sw = 0.0
    seen = 0
    bar = tqdm(loader, desc="fisher", disable=not progress, unit="batch")
    for batch in bar:
        batch = _move_batch(batch, device)
        data_mask = ~batch["is_data_mask"] if mc_as_data else batch["is_data_mask"]
        if not bool(data_mask.any()):
            continue
        per = model.data_nll_continuity(
            batch["mll"], batch["pt_pm"], batch["eta_pm"], batch["phi_pm"],
            batch["q_pm"], batch["b_pm"], batch["muon_kin_std"], data_mask,
            n_gh=n_gh, n_iter=n_iter)
        w = batch["w"] * data_mask.to(batch["w"].dtype)
        nll = (w * per).sum()
        if not torch.isfinite(nll):
            continue
        g = torch.autograd.grad(nll, params, create_graph=True)
        g_full = torch.cat([gi.reshape(-1) for gi in g])
        grad += g_full[active_idx].detach()
        for r in range(n_act):
            i = int(active_idx[r])
            row = torch.autograd.grad(
                g_full[i], params, retain_graph=True, allow_unused=True)
            row_full = torch.cat([
                (ri if ri is not None else torch.zeros_like(p)).reshape(-1)
                for ri, p in zip(row, params)])
            H[r] += row_full[active_idx].detach()
        sw += float(w.sum().item())
        seen += int(data_mask.sum().item())
        bar.set_postfix_str(f"events={seen:,}")
    bar.close()
    if seen == 0:
        raise RuntimeError(
            "compute_fisher_info_continuity: loader yielded zero events on the "
            "data branch (mc_as_data=%s). Check the split / --validation." % mc_as_data)
    H = 0.5 * (H + H.T)
    layout = {"blocks": blocks, "smear_cols": smear_cols,
              "n_scale": (model.theta_scale.numel() if model.scale_enabled else 0),
              "sw": sw, "seen": seen, "grad": grad.detach().cpu()}
    return H.detach().cpu(), layout


def _fisher_save_dict(H: torch.Tensor, layout: dict, model: JpsiMassMixtureModel) -> dict:
    """Invert H → covariance and package it (with labels, the θ_scale block in
    the legacy 24×3×24×3 layout for the diagnostics, and delta-method effective
    σ for the raw θ_smear)."""
    n_act = H.shape[0]
    # Positive-definiteness check: at a true optimum the observed information is
    # PD. Negative/zero eigenvalues flag a non-converged fit or unidentified
    # (e.g. event-starved) η-bins; the corresponding variances are not
    # trustworthy. eigvalsh is exact for the symmetric H.
    try:
        eig = torch.linalg.eigvalsh(H)
        min_eig = float(eig.min())
        n_neg_eig = int((eig <= 0).sum())
    except RuntimeError:
        min_eig = float("nan")
        n_neg_eig = -1
    cov = None
    ok = False
    try:
        cov = torch.linalg.inv(H)
        ok = bool(torch.isfinite(cov).all()) and n_neg_eig == 0
    except RuntimeError:
        ok = False
    if cov is None or not bool(torch.isfinite(cov).all()):
        try:
            cov = torch.linalg.pinv(H)
        except Exception:
            cov = None

    n_scale = layout["n_scale"]
    smear_cols = layout["smear_cols"]
    comp = ["A", "e", "M"]
    smear_comp = ["a", "c"]
    labels: List[str] = []
    if n_scale:
        for b in range(model.theta_scale.shape[0]):
            for c in range(model.theta_scale.shape[1]):
                labels.append(f"{comp[c]}[{b}]")
    for b in range(model.theta_smear.shape[0]):
        for c in smear_cols:
            labels.append(f"{smear_comp[c]}[{b}](raw)")

    out: dict = {
        "hessian": H,
        "covariance": cov,
        "ok": ok,
        "labels": labels,
        "n_scale": n_scale,
        "smear_cols": smear_cols,
        "smear_fit_params": model.smear_fit_params,
        "param_space": "scale: linear (A,e,M); smear: raw pre-softplus theta_smear",
        "n_events": layout["seen"],
        "sum_weight": layout["sw"],
        "min_eig": min_eig,
        "n_negative_eig": n_neg_eig,
    }
    # Estimated distance to minimum: EDM = ½ gᵀ V g (V = covariance = H⁻¹), the
    # predicted remaining decrease in the NLL to reach the optimum (MINUIT
    # convention). ≈0 at convergence; large ⇒ the fit hasn't reached a minimum.
    grad = layout.get("grad")
    if grad is not None:
        out["grad"] = grad
        out["grad_norm"] = float(grad.norm())
        if cov is not None:
            out["edm"] = 0.5 * float(grad @ (cov @ grad))
    # θ_scale block in the legacy layout (consumed by the diagnostics for ±1σ
    # bands + the correlation heatmap). This is the scale block of the JOINT
    # inverse, so it carries the θ_scale↔θ_smear correlation.
    if n_scale == 72:
        out["hessian_24_3_24_3"] = H[:72, :72].view(24, 3, 24, 3)
        if cov is not None:
            cov_scale = cov[:72, :72]
            out["covariance_24_3_24_3"] = cov_scale.view(24, 3, 24, 3)
            out["sigma_scale_24_3"] = torch.sqrt(
                torch.clamp(torch.diag(cov_scale), min=0.0)).view(24, 3)
    # Effective (physical, ≥0) σ on the smear a/c per η-bin via the delta method
    # (eff = softplus(raw), |∂eff/∂raw| = sigmoid(raw)); inactive columns → 0.
    if smear_cols and cov is not None:
        n_eta, n_comp = model.theta_smear.shape
        cov_smear = cov[n_scale:, n_scale:]
        sig_raw = torch.sqrt(torch.clamp(torch.diag(cov_smear), min=0.0))
        jac = torch.sigmoid(model.theta_smear.detach().cpu())  # [n_eta, n_comp]
        sig_eff = torch.zeros(n_eta, n_comp)
        k = 0
        for b in range(n_eta):
            for c in smear_cols:
                sig_eff[b, c] = jac[b, c] * sig_raw[k]
                k += 1
        out["sigma_smear_eff_24_2"] = sig_eff
    return out


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Two-stage continuity training (default)
#   stage 1 ("flow"): nominal flow p₀(m|muon_kin) at θ=0 on simulation, no θ
#                     conditioning, MC rows only.
#   stage 2 ("fit"):  freeze the flow, fit θ_scale / θ_smear / background-MLP on
#                     data via the analytic continuity tilt (model.data_nll_continuity).
# ---------------------------------------------------------------------------


def _setup_common(args, *, stats_override=None):
    """Shards + stats + train/val loaders (shared by both stages).

    ``stats_override`` (used by ``--stage fit``) supplies the standardisation
    stats from the flow checkpoint, taking precedence over ``--stats-in`` /
    recomputation so the data branch standardises exactly as the flow was
    trained.
    """
    shard_files = discover_shards(args.inputs)
    if not shard_files:
        print("error: no Arrow shards found under --inputs", file=sys.stderr)
        return None
    print(f"discovered {len(shard_files)} shard(s)")
    if stats_override is not None:
        stats = stats_override
        print("using preproc stats from the flow checkpoint")
    elif args.stats_in is not None and os.path.exists(args.stats_in):
        with open(args.stats_in) as f:
            stats = _stats_from_dict(json.load(f))
        print(f"loaded preproc stats from {args.stats_in}")
    else:
        t0 = time.time()
        stats = compute_jpsi_mass_stats(shard_files, m_lo=args.m_lo, m_hi=args.m_hi)
        print(f"computed preproc stats in {time.time() - t0:.1f}s")
    print(f"  mll: μ={stats.mll_mean:.4f}  σ={stats.mll_std:.4f}  "
          f"window [{stats.m_lo}, {stats.m_hi}]")
    os.makedirs(args.output, exist_ok=True)
    stats_path = os.path.join(args.output, "preproc_stats.json")
    with open(stats_path, "w") as f:
        json.dump(_stats_to_dict(stats), f, indent=2)
    print(f"wrote {stats_path}")
    train_loader, val_loader = _make_loaders(args, shard_files, stats)
    return shard_files, stats, train_loader, val_loader


def _make_loaders(args, shard_files, stats, *, half=None):
    """Build the ``(train, val)`` loaders for one stage. ``half`` selects a
    deterministic disjoint event half (0/1) — used by the MC-closure
    validation mode (stage 1 ← half 0, stage 2 ← half 1); ``None`` = all
    events."""
    train_loader = JpsiMassArrowLoader(
        shard_files, stats, batch_size=args.batch_size, split="train",
        val_fraction=args.val_fraction, holdout_fraction=args.holdout_fraction,
        drop_last=True, half=half)
    val_loader = JpsiMassArrowLoader(
        shard_files, stats, batch_size=args.batch_size, split="val",
        val_fraction=args.val_fraction, holdout_fraction=args.holdout_fraction,
        drop_last=False, half=half)
    return train_loader, val_loader


# Args that fix the flow's parameter shapes — these must match the saved flow
# when reloading it for --stage fit (the model is rebuilt from args before the
# flow weights are loaded). The MLP size and θ-smear fit choice are NOT here:
# only the flow is loaded from the checkpoint, so those stay free stage-2 knobs.
_FLOW_ARCH_KEYS = (
    "flow_arch", "flow_n_transforms", "flow_hidden", "flow_n_hidden",
    "gf_components", "nsf_bins",
)


def _apply_flow_arch_from_ckpt(args, ck_args: dict) -> None:
    """Override the flow-architecture args on ``args`` with the values stored
    in the flow checkpoint so the rebuilt flow matches the saved weights.
    Logs any field that changed; silently keeps the CLI value for keys the
    checkpoint doesn't carry (older checkpoints)."""
    changed = []
    for k in _FLOW_ARCH_KEYS:
        if k in ck_args:
            old = getattr(args, k, None)
            new = ck_args[k]
            if old != new:
                changed.append(f"{k}: {old}→{new}")
            setattr(args, k, new)
    if changed:
        print("  flow-architecture args set from checkpoint: " + ", ".join(changed))
    else:
        print("  flow-architecture args already match the checkpoint")


def _build_model(args, stats, device, *, theta_conditioning):
    return JpsiMassMixtureModel(
        m_lo=stats.m_lo, m_hi=stats.m_hi, mll_log_scale=stats.mll_log_scale,
        mll_mean=stats.mll_mean, mll_std=stats.mll_std,
        y_event_mean=torch.from_numpy(stats.y_event_mean),
        y_event_std_tensor=torch.from_numpy(stats.y_event_std),
        muon_kin_mean=torch.from_numpy(stats.muon_kin_mean),
        muon_kin_std_tensor=torch.from_numpy(stats.muon_kin_std),
        flow_arch=args.flow_arch, flow_n_transforms=args.flow_n_transforms,
        flow_hidden_features=args.flow_hidden, flow_n_hidden_layers=args.flow_n_hidden,
        flow_gf_components=args.gf_components, flow_nsf_bins=args.nsf_bins,
        mlp_hidden=args.mlp_hidden, mlp_n_layers=args.mlp_n_layers,
        linearize_scale=args.linearize_scale,
        smear_init_a=args.smear_init_a, smear_init_c=args.smear_init_c,
        smearing_enabled=not args.disable_smearing,
        scale_enabled=not args.disable_scale,
        qop_floor_frac=args.qop_floor_frac, smear_fit_params=args.smear_fit_params,
        theta_conditioning=theta_conditioning,
    ).to(device)


def _run_epochs(args, model, optim, train_loader, val_loader, stats, *,
                step_fn, ckpt_prefix, stage_name, epochs):
    """Generic weighted-NLL epoch loop with val + best/last checkpoints +
    early-stop. ``step_fn(model, batch) -> (loss, sum_w)`` returns the batch's
    weighted-mean NLL (scalar tensor) over the rows the stage uses and the
    corresponding Σw. Runs under ``enable_grad`` in val too (stage 2's score is
    autograd-based). Returns the best val metric."""
    best_val = float("inf"); no_improve = 0
    best_ckpt = os.path.join(args.output, f"{ckpt_prefix}_best.pt")
    last_ckpt = os.path.join(args.output, f"{ckpt_prefix}_last.pt")
    device = args.device
    # The streaming loader has no __len__; learn the batch count on epoch 1 so
    # epochs ≥2 show a true percentage-complete bar.
    n_batches_total = None
    sched, sched_kind = _make_scheduler(args, optim, epochs)
    if sched_kind != "none":
        print(f"  lr schedule: {sched_kind}"
              + (f" (factor={args.lr_reduce_factor:g}, patience={args.lr_reduce_patience}, "
                 f"min_lr={args.min_lr:g}); each lr reduction resets the early-stop "
                 f"counter (early-stop deferred while the lr keeps dropping; with "
                 f"lr-reduce-patience {args.lr_reduce_patience} < patience {args.patience} "
                 f"the lr typically reaches min_lr first, but early-stop can still fire "
                 f"at a higher lr if no reduction is pending)"
                 if sched_kind == "plateau"
                 else f" (T_max={epochs}, min_lr={args.min_lr:g})"))

    def _ckpt(epoch, bv, vm):
        return {
            "epoch": epoch, "stage": stage_name,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "theta_scale": model.theta_scale.detach().cpu(),
            "theta_smear": model.theta_smear.detach().cpu(),
            "stats": _stats_to_dict(stats), "best_val": bv, "val_metric": vm,
            "args": vars(args),
        }

    for epoch in range(1, epochs + 1):
        t0 = time.time(); model.train()
        tr_sum = 0.0; tr_w = 0.0; n_seen = 0
        lr_str = _lr_str(optim)
        # total=n_batches_total → tqdm renders a % bar (None on epoch 1).
        bar = tqdm(train_loader, total=n_batches_total,
                   desc=f"[{stage_name}] epoch {epoch:>3}/{epochs}",
                   leave=False, disable=not args.progress, unit="batch")
        for batch in bar:
            batch = _move_batch(batch, device)
            optim.zero_grad(set_to_none=True)
            loss, sw = step_fn(model, batch)
            n_seen += 1
            if sw <= 0:
                continue
            if not torch.isfinite(loss):
                if args.nan_on_step == "raise":
                    raise RuntimeError(f"non-finite loss [{stage_name}] epoch {epoch}")
                if args.nan_on_step == "skip":
                    bar.set_postfix_str("SKIP NaN"); continue
            loss.backward()
            optim.step()
            tr_sum += float(loss.item()) * sw; tr_w += sw
            bar.set_postfix_str(f"nll={tr_sum / max(tr_w, 1e-30):+.4f} lr={lr_str}")
        bar.close()
        if n_batches_total is None:
            n_batches_total = n_seen   # exact count for the % bar from epoch 2 on
        train_nll = tr_sum / max(tr_w, 1e-30)

        model.eval(); v_sum = 0.0; v_w = 0.0
        for batch in val_loader:
            batch = _move_batch(batch, device)
            with torch.enable_grad():
                loss, sw = step_fn(model, batch)
            if sw <= 0:
                continue
            v_sum += float(loss.item()) * sw; v_w += sw
        val_nll = v_sum / max(v_w, 1e-30)

        extra = ""
        if stage_name == "fit":
            extra = (f" θ_scale‖∞={model.theta_scale.abs().max().item():.3e}"
                     f" θ_smear‖∞={model.theta_smear.abs().max().item():.3e}")
        print(f"[{stage_name}] epoch {epoch:>3}: train_nll={train_nll:+.4f} "
              f"val_nll={val_nll:+.4f} (Σw={v_w:.2e}) lr={lr_str} "
              f"dt={time.time()-t0:.1f}s{extra}")

        improved = val_nll < best_val - args.patience_threshold
        if improved:
            best_val = val_nll; no_improve = 0
        else:
            no_improve += 1
        torch.save(_ckpt(epoch, best_val, val_nll), last_ckpt)
        if improved:
            torch.save(_ckpt(epoch, best_val, val_nll), best_ckpt)
        # Step the LR schedule; a reduction in ANY group resets the early-stop
        # counter, so early-stop is deferred while the lr is still dropping.
        # With lr_reduce_patience < patience the lr therefore usually reaches
        # min_lr before early-stop fires — but this is not guaranteed (e.g. if
        # patience <= lr_reduce_patience, early-stop can fire at a higher lr).
        if sched is not None:
            lr_before = [g["lr"] for g in optim.param_groups]
            sched.step(val_nll) if sched_kind == "plateau" else sched.step()
            if any(g["lr"] < lb - 1e-12 for g, lb in zip(optim.param_groups, lr_before)):
                no_improve = 0
        if not improved and not args.no_early_stop and no_improve >= args.patience:
            print(f"[{stage_name}] early-stop: no improvement for {no_improve} epochs")
            break

    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[{stage_name}] reloaded best ({best_ckpt}, val={ckpt['best_val']:+.4f})")
    return best_val


def train_stage1(args, model, train_loader, val_loader, stats) -> float:
    """Stage 1: fit the nominal flow p₀(m|muon_kin) on simulation (MC rows)."""
    print("\n=== stage 1: nominal flow on simulation (θ=0, no θ conditioning) ===")
    optim = torch.optim.Adam(model.flow.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    print(f"  optimizer: flow ({sum(p.numel() for p in model.flow.parameters()):,} params), lr={args.lr:g}")

    def step1(model, batch):
        idx = (~batch["is_data_mask"]).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return torch.zeros((), device=batch["mll"].device), 0.0
        logp = model.log_p_nominal(batch["mll"][idx], batch["muon_kin_std"][idx])
        w = batch["w"][idx]
        sw = float(w.sum().clamp_min(1e-30))
        return -(w * logp).sum() / sw, sw

    return _run_epochs(args, model, optim, train_loader, val_loader, stats,
                       step_fn=step1, ckpt_prefix="flow", stage_name="flow",
                       epochs=args.flow_epochs or args.epochs)


def train_stage2(args, model, train_loader, val_loader, stats,
                 *, mc_as_data: bool = False) -> float:
    """Stage 2: freeze the flow, fit θ + background MLP on data (continuity).

    With ``mc_as_data`` (MC-closure validation mode) the simulation rows are
    treated as the pseudo-data branch (``~is_data_mask``) so θ is fit against a
    disjoint half of simulation; the closure target is θ → 0.
    """
    src = "MC pseudo-data" if mc_as_data else "data"
    print(f"\n=== stage 2: θ + background fit on {src} (frozen flow, #2 direct-eval) ===")
    # Freeze the flow; it is the nominal template from stage 1.
    for p in model.flow.parameters():
        p.requires_grad_(False)
    # θ_scale is the advective shift (init 0, signed). θ_smear is softplus-
    # parameterised so the smear variance V ≥ 0 (no de-convolution) — leave it
    # at its softplus⁻¹(smear_init) init (effective ≈ smear_init).
    with torch.no_grad():
        model.theta_scale.zero_()
    groups = [{"params": model.mlp.parameters(), "lr": args.fit_mlp_lr}]
    tags = ["mlp"]
    if not args.disable_scale:
        groups.append({"params": [model.theta_scale], "lr": args.fit_scale_lr}); tags.append("θ_scale")
    if not args.disable_smearing:
        groups.append({"params": [model.theta_smear], "lr": args.fit_smear_lr}); tags.append("θ_smear")
    optim = torch.optim.Adam(groups)
    print(f"  optimizer groups: {', '.join(tags)}  "
          f"(lr mlp={args.fit_mlp_lr:g} scale={args.fit_scale_lr:g} smear={args.fit_smear_lr:g})")
    print(f"  signal density: #2 direct-eval (GH n_gh={args.continuity_n_gh}, "
          f"n_iter={args.continuity_n_iter}); normalised by construction")

    def step2(model, batch):
        # In validation mode the simulation rows play the role of data.
        data_mask = ~batch["is_data_mask"] if mc_as_data else batch["is_data_mask"]
        per = model.data_nll_continuity(
            batch["mll"], batch["pt_pm"], batch["eta_pm"], batch["phi_pm"],
            batch["q_pm"], batch["b_pm"], batch["muon_kin_std"], data_mask,
            n_gh=args.continuity_n_gh, n_iter=args.continuity_n_iter)
        w = batch["w"] * data_mask.to(batch["w"].dtype)
        sw = float(w.sum().clamp_min(1e-30))
        return (w * per).sum() / sw, sw

    return _run_epochs(args, model, optim, train_loader, val_loader, stats,
                       step_fn=step2, ckpt_prefix="fit", stage_name="fit",
                       epochs=args.fit_epochs or args.epochs)


def train_loop(args: argparse.Namespace) -> int:
    """Dispatch: legacy single-stage forward-fold, or the two-stage continuity
    pipeline (stage 1 flow → stage 2 fit)."""
    if args.stage == "legacy":
        if args.validation:
            print("warning: --validation has no effect with --stage legacy "
                  "(it applies only to the two-stage pipeline); ignoring.",
                  file=sys.stderr)
        return _train_loop_legacy(args)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("warning: CUDA requested but not available; falling back to CPU")
        device = args.device = "cpu"

    # --stage fit: load the flow checkpoint up front so its architecture and
    # stats drive the model build + standardisation (the model is rebuilt from
    # args before the flow weights are loaded, so they must match).
    flow_ck = None
    flow_ckpt = None
    stats_override = None
    if args.stage == "fit":
        flow_ckpt = args.flow_checkpoint or os.path.join(args.output, "flow_best.pt")
        if not os.path.exists(flow_ckpt):
            print(f"error: --stage fit needs a stage-1 flow; {flow_ckpt!r} not found "
                  f"(run --stage flow first or pass --flow-checkpoint)", file=sys.stderr)
            return 1
        print(f"loading stage-1 flow checkpoint: {flow_ckpt}")
        flow_ck = torch.load(flow_ckpt, map_location=device, weights_only=False)
        _apply_flow_arch_from_ckpt(args, flow_ck.get("args", {}) or {})
        # Reuse the flow's own standardisation unless the user forces --stats-in.
        if args.stats_in is None and flow_ck.get("stats") is not None:
            stats_override = _stats_from_dict(flow_ck["stats"])

    setup = _setup_common(args, stats_override=stats_override)
    if setup is None:
        return 1
    shard_files, stats, train_loader, val_loader = setup

    model = _build_model(args, stats, device, theta_conditioning=False)
    print(f"model: flow={model.flow_arch} (no θ conditioning), "
          f"scale={'on' if model.scale_enabled else 'off'} "
          f"smear={'on' if model.smearing_enabled else 'off'} "
          f"smear_fit={model.smear_fit_params}")

    if args.stage == "fit":
        # Load ONLY the flow weights; the background MLP and θ start fresh
        # (stage 1 leaves them at init anyway) so the MLP size / smear-fit
        # choice remain free stage-2 knobs.
        flow_sd = {k[len("flow."):]: v for k, v in flow_ck["state_dict"].items()
                   if k.startswith("flow.")}
        model.flow.load_state_dict(flow_sd)
        print(f"loaded flow weights from {flow_ckpt} ({len(flow_sd)} tensors; "
              f"mlp + θ start fresh)")

    # MC-closure validation: both stages run on simulation, with a deterministic
    # disjoint half each (stage 1 ← half 0, stage 2 ← half 1 treated as
    # pseudo-data). The disjoint halves keep stage 2 from fitting θ against the
    # very events stage 1's flow was trained on; the closure target is θ → 0.
    if args.validation:
        print("\n*** MC-closure validation mode: simulation for both stages "
              "(stage 1 ← half 0, stage 2 ← half 1 as pseudo-data); target θ → 0 ***")
        s1_train, s1_val = _make_loaders(args, shard_files, stats, half=0)
        s2_train, s2_val = _make_loaders(args, shard_files, stats, half=1)
    else:
        s1_train, s1_val = train_loader, val_loader
        s2_train, s2_val = train_loader, val_loader

    if args.stage in ("both", "flow"):
        train_stage1(args, model, s1_train, s1_val, stats)
    if args.stage in ("both", "fit"):
        train_stage2(args, model, s2_train, s2_val, stats, mc_as_data=args.validation)
        if args.fisher_info:
            _run_fisher_continuity(args, model, shard_files, stats, device)
    return 0


def _run_fisher_continuity(args, model, shard_files, stats, device) -> None:
    """Observed Fisher info for the two-stage fit (θ_scale + active θ_smear,
    fixed flow + MLP) → ``<output>/fisher_info.pt``. Evaluated on the data the
    fit used (``--fisher-split``, default the train split; half 1 in
    validation mode), so the covariance has the right statistical scale."""
    if not (model.scale_enabled or model.smearing_enabled):
        print("skipping Fisher info: both --disable-scale and --disable-smearing.")
        return
    half = 1 if args.validation else None
    loader = JpsiMassArrowLoader(
        shard_files, stats, batch_size=args.batch_size, split=args.fisher_split,
        val_fraction=args.val_fraction, holdout_fraction=args.holdout_fraction,
        drop_last=False, half=half)
    print(f"\ncomputing observed Fisher information (θ_scale + active θ_smear, "
          f"fixed flow + MLP) on split={args.fisher_split}"
          + ("  half=1 (MC pseudo-data)" if args.validation else "")
          + f"  [smear_fit={model.smear_fit_params}]")
    t0 = time.time()
    H, layout = compute_fisher_info_continuity(
        model, loader, device, mc_as_data=args.validation,
        n_gh=args.continuity_n_gh, n_iter=args.continuity_n_iter,
        progress=args.progress)
    out = _fisher_save_dict(H, layout, model)
    path = os.path.join(args.output, "fisher_info.pt")
    torch.save(out, path)
    print(f"  wrote {path}: {H.shape[0]}×{H.shape[0]} info matrix over "
          f"{len(out['labels'])} params ({layout['seen']:,} events, "
          f"Σw={layout['sw']:.2e}); "
          f"{'PD, inverted' if out['ok'] else 'NOT positive-definite → pinv (cov approximate)'} "
          f"in {time.time()-t0:.1f}s")
    if out.get("edm") is not None:
        print(f"  EDM (½ gᵀV g, est. NLL distance to minimum) = {out['edm']:.3e}"
              f"  (‖grad‖={out['grad_norm']:.3e})")
    if out["n_negative_eig"] != 0:
        print(f"  WARNING: observed information has {out['n_negative_eig']} "
              f"non-positive eigenvalue(s) (min={out['min_eig']:.3e}) — the fit is "
              f"not at a clean optimum or some η-bins are event-starved; the "
              f"corresponding variances are unreliable.")


def _train_loop_legacy(args: argparse.Namespace) -> int:
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("warning: CUDA requested but not available; falling back to CPU")
        device = "cpu"

    shard_files = discover_shards(args.inputs)
    if not shard_files:
        print("error: no Arrow shards found under --inputs", file=sys.stderr)
        return 1
    print(f"discovered {len(shard_files)} shard(s)")

    if args.stats_in is not None and os.path.exists(args.stats_in):
        with open(args.stats_in) as f:
            stats = _stats_from_dict(json.load(f))
        print(f"loaded preproc stats from {args.stats_in}")
    else:
        t0 = time.time()
        stats = compute_jpsi_mass_stats(
            shard_files, m_lo=args.m_lo, m_hi=args.m_hi
        )
        dt = time.time() - t0
        print(f"computed preproc stats in {dt:.1f}s")
    print(
        f"  mll: μ={stats.mll_mean:.4f}  σ={stats.mll_std:.4f}  "
        f"window [{stats.m_lo}, {stats.m_hi}]"
    )

    os.makedirs(args.output, exist_ok=True)
    stats_path = os.path.join(args.output, "preproc_stats.json")
    with open(stats_path, "w") as f:
        json.dump(_stats_to_dict(stats), f, indent=2)
    print(f"wrote {stats_path}")

    train_loader = JpsiMassArrowLoader(
        shard_files,
        stats,
        batch_size=args.batch_size,
        split="train",
        val_fraction=args.val_fraction,
        holdout_fraction=args.holdout_fraction,
        drop_last=True,
    )
    val_loader = JpsiMassArrowLoader(
        shard_files,
        stats,
        batch_size=args.batch_size,
        split="val",
        val_fraction=args.val_fraction,
        holdout_fraction=args.holdout_fraction,
        drop_last=False,
    )

    model = JpsiMassMixtureModel(
        m_lo=stats.m_lo,
        m_hi=stats.m_hi,
        mll_log_scale=stats.mll_log_scale,
        mll_mean=stats.mll_mean,
        mll_std=stats.mll_std,
        y_event_mean=torch.from_numpy(stats.y_event_mean),
        y_event_std_tensor=torch.from_numpy(stats.y_event_std),
        muon_kin_mean=torch.from_numpy(stats.muon_kin_mean),
        muon_kin_std_tensor=torch.from_numpy(stats.muon_kin_std),
        flow_arch=args.flow_arch,
        flow_n_transforms=args.flow_n_transforms,
        flow_hidden_features=args.flow_hidden,
        flow_n_hidden_layers=args.flow_n_hidden,
        flow_gf_components=args.gf_components,
        flow_nsf_bins=args.nsf_bins,
        mlp_hidden=args.mlp_hidden,
        mlp_n_layers=args.mlp_n_layers,
        linearize_scale=args.linearize_scale,
        smear_init_a=args.smear_init_a,
        smear_init_c=args.smear_init_c,
        smearing_enabled=not args.disable_smearing,
        scale_enabled=not args.disable_scale,
        detach_flow_on_data=args.detach_flow_on_data,
        fixed_theta_sampling=args.fixed_theta_sampling,
        qop_floor_frac=args.qop_floor_frac,
        smear_fit_params=args.smear_fit_params,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}  flow={model.flow_arch}  "
          f"scale={'linearized' if model.linearize_scale else 'exact'}")
    print(f"  detach_flow_on_data={model.detach_flow_on_data} "
          f"(flow {'trained on MC only' if model.detach_flow_on_data else 'trained on data+MC'})")
    if model.fixed_theta_sampling:
        print(f"  fixed_theta_sampling=True: θ̃ sampled around the fixed init "
              f"(θ_scale=0, θ_smear=init) with fixed widths "
              f"(scale σ_A/e/M={args.scale_noise_sigma_A:.3g}/{args.scale_noise_sigma_e:.3g}/"
              f"{args.scale_noise_sigma_M:.3g}, smear σ={args.smear_noise_sigma:.3g}), "
              f"independent of the model values [adaptive σ bypassed]")
    print(f"  θ_scale: {tuple(model.theta_scale.shape)}  init=0")
    _fit_desc = {"both": "a, c", "a": "a only (c set to 0)",
                 "c": "c only (a set to 0)"}[model.smear_fit_params]
    print(
        f"  θ_smear: {tuple(model.theta_smear.shape)}  "
        f"init=({args.smear_init_a:.3g}, {args.smear_init_c:.3g})  fit: {_fit_desc}"
    )
    if model.qop_floor_frac > 0:
        print(f"  qop_floor_frac={model.qop_floor_frac:.3g} "
              f"(pt inflation capped at {1.0 / model.qop_floor_frac:.1f}×)")

    # --mc-only: fix the model's nuisances to zero. There is no data branch
    # to fit them, so θ_scale and θ_smear are pinned at 0 and excluded from
    # the optimizer. The flow is still trained with θ̃_scale / θ̃_smear
    # *sampled around zero* (the MC branch samples around θ.detach() = 0), so
    # it still learns the full conditioning dependence — the model values are
    # just the (zero) sampling centre.
    if args.mc_only:
        with torch.no_grad():
            model.theta_scale.zero_()
            # θ_smear is softplus-reparameterised: raw=0 → effective 0.69, so
            # leave it at its (softplus⁻¹(smear_init)) init rather than zeroing
            # the raw. Default smear_init=0 → effective ≈ 0.
        eff = model.effective_theta_smear()
        print(f"  --mc-only: θ_scale fixed at 0; θ_smear fixed at init "
              f"(effective a≈{float(eff[:,0].mean()):.2e}, c≈{float(eff[:,1].mean()):.2e}); "
              f"both sampled for the flow")

    # Optimizer parameter groups — debug flags trim them:
    #   --mc-only          drops MLP, θ_scale, θ_smear (nuisances fixed).
    #   --disable-scale    drops θ_scale (no scale fold / conditioning).
    #   --disable-smearing drops θ_smear (no smearing kernel / conditioning).
    train_scale = (not args.mc_only) and (not args.disable_scale)
    train_smear = (not args.mc_only) and (not args.disable_smearing)
    param_groups = [
        {"params": model.flow.parameters(), "lr": args.lr},
    ]
    if not args.mc_only:
        param_groups.append({"params": model.mlp.parameters(), "lr": args.lr})
    if train_scale:
        param_groups.append({"params": [model.theta_scale], "lr": args.theta_scale_lr})
    if train_smear:
        param_groups.append({"params": [model.theta_smear], "lr": args.theta_smear_lr})
    optim = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)
    print(
        "  optimizer groups: flow"
        + ("" if args.mc_only else ", mlp")
        + (", θ_scale" if train_scale else "")
        + (", θ_smear" if train_smear else "")
    )
    sched, sched_kind = _make_scheduler(args, optim, args.epochs)
    if sched_kind != "none":
        print(f"  lr schedule: {sched_kind}")

    # torch.compile: skipped for GF flow (its Jacobian is computed via
    # an inner ``torch.autograd.grad`` call inside zuko's
    # ``MonotonicTransform.call_and_ladj`` and dynamo cannot trace that
    # double-autograd path). The MLP would compile fine but it's tiny —
    # the dominant cost is the flow, so compile gain is negligible if the
    # flow stays eager. Print a one-line note and leave model eager either way.
    if args.compile:
        print(
            "torch.compile: requested but the GF base flow is incompatible "
            "(double-autograd in zuko's MonotonicTransform.call_and_ladj is "
            "not traceable by dynamo). Continuing in eager mode."
        )

    # Mixed-precision setup. fp32 → no-op autocast + no-op scaler.
    amp_ctx, scaler = _make_amp(args.precision, device)
    if args.precision != "fp32":
        print(f"autocast precision: {args.precision}")

    # Per-component θ_scale sampling widths (σ_A, σ_e, σ_M) — broadcasts over
    # the last dim of theta_scale [n_eta, 3] in log_p_signal_mc and as the
    # _adaptive_sigma fallback.
    scale_sigma_vec = torch.tensor(
        [args.scale_noise_sigma_A, args.scale_noise_sigma_e, args.scale_noise_sigma_M],
        dtype=torch.float32, device=device,
    )

    best_val = float("inf")
    no_improve = 0
    best_ckpt = os.path.join(args.output, "checkpoint_best.pt")
    last_ckpt = os.path.join(args.output, "checkpoint_last.pt")

    if args.detect_anomaly:
        print("torch.autograd anomaly detection enabled (--detect-anomaly)")
        torch.autograd.set_detect_anomaly(True)

    n_batches_total = None  # learned on epoch 1 for the % bar on epochs ≥2
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        model.train()
        train_data_sum = 0.0
        train_data_w = 0.0
        train_mc_sum = 0.0
        train_mc_w = 0.0
        n_batches = 0
        lr_str = _lr_str(optim)
        bar = tqdm(
            train_loader,
            total=n_batches_total,
            desc=f"epoch {epoch:>3}/{args.epochs} train",
            leave=False,
            disable=not args.progress,
            unit="batch",
        )
        for batch in bar:
            batch = _move_batch(batch, device)
            # --mc-only: drop data rows before loss eval. Skip the step
            # if the batch happens to be all-data.
            batch = _maybe_mc_only_batch(batch, args.mc_only)
            if batch is None:
                continue
            # Sampling widths. With --fixed-theta-sampling the flow trains on
            # a fixed-width distribution (the scalar --*-noise-sigma), else the
            # adaptive σ from Adam's 2nd moment. Disabled smearing → 0.0.
            smear_sigma = 0.0 if args.disable_smearing else (
                args.smear_noise_sigma if args.fixed_theta_sampling
                else _adaptive_sigma(
                    model.theta_smear, optim,
                    sigma_init=args.smear_noise_sigma,
                    enabled=args.adaptive_sigma,
                    warmup_steps=args.adaptive_warmup_steps,
                    sigma_min=args.adaptive_smear_sigma_min,
                    sigma_max=args.adaptive_smear_sigma_max,
                    scale=args.adaptive_sigma_scale,
                )
            )
            # θ_scale sampling for the MC-branch flow conditioning is active
            # regardless of --mc-only (the flow must learn the θ_scale
            # dependence from MC); _adaptive_sigma falls back to sigma_init
            # when θ_scale has no Adam state yet. Disabled scale → 0.0.
            scale_sigma = 0.0 if args.disable_scale else (
                scale_sigma_vec if args.fixed_theta_sampling
                else _adaptive_sigma(
                    model.theta_scale, optim,
                    sigma_init=scale_sigma_vec,
                    enabled=args.adaptive_sigma,
                    warmup_steps=args.adaptive_warmup_steps,
                    sigma_min=args.adaptive_scale_sigma_min,
                    sigma_max=args.adaptive_scale_sigma_max,
                    scale=args.adaptive_sigma_scale,
                )
            )
            optim.zero_grad(set_to_none=True)
            with amp_ctx():
                nll, d_sum, d_w, m_sum, m_w = _nll_components(
                    model, batch,
                    scale_noise_sigma=scale_sigma,
                    smear_noise_sigma=smear_sigma, mc_only=args.mc_only,
                )
                w_sum = batch["w"].sum().clamp_min(1e-30)
                loss = nll / w_sum
            # NaN/Inf trip-wire — only the explicitly-asked-for behaviour.
            if not torch.isfinite(loss):
                msg = (
                    f"non-finite loss at epoch {epoch}, batch {n_batches}: "
                    f"loss={loss.item()} nll={nll.item()} "
                    f"smear_sigma={'tensor' if isinstance(smear_sigma, torch.Tensor) else smear_sigma} "
                    f"scale_sigma={'tensor' if isinstance(scale_sigma, torch.Tensor) else scale_sigma}"
                )
                if args.nan_on_step == "raise":
                    raise RuntimeError(msg)
                if args.nan_on_step == "skip":
                    bar.set_postfix_str(f"SKIP NaN (n_batches={n_batches})")
                    continue
                # "ignore" → fall through and let NaN poison the sum.
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            train_data_sum += float(d_sum.item()); train_data_w += float(d_w.item())
            train_mc_sum += float(m_sum.item()); train_mc_w += float(m_w.item())
            n_batches += 1
            bar.set_postfix_str(
                f"data={train_data_sum / max(train_data_w, 1e-30):+.4f} "
                f"mc={train_mc_sum / max(train_mc_w, 1e-30):+.4f} lr={lr_str}"
            )
        bar.close()
        if n_batches_total is None:
            n_batches_total = n_batches
        train_data_nll = train_data_sum / max(train_data_w, 1e-30)
        train_mc_nll = train_mc_sum / max(train_mc_w, 1e-30)

        # Match validation's noise distribution to training's so the numbers
        # are directly comparable. Use the same adaptive-σ helper the
        # training loop just used, then pass through to _epoch_metrics which
        # seeds the RNG deterministically.
        val_scale_sigma = 0.0 if args.disable_scale else (
            scale_sigma_vec if args.fixed_theta_sampling
            else _adaptive_sigma(
                model.theta_scale, optim,
                sigma_init=scale_sigma_vec,
                enabled=args.adaptive_sigma,
                warmup_steps=args.adaptive_warmup_steps,
                sigma_min=args.adaptive_scale_sigma_min,
                sigma_max=args.adaptive_scale_sigma_max,
                scale=args.adaptive_sigma_scale,
            )
        )
        val_smear_sigma = 0.0 if args.disable_smearing else (
            args.smear_noise_sigma if args.fixed_theta_sampling
            else _adaptive_sigma(
                model.theta_smear, optim,
                sigma_init=args.smear_noise_sigma,
                enabled=args.adaptive_sigma,
                warmup_steps=args.adaptive_warmup_steps,
                sigma_min=args.adaptive_smear_sigma_min,
                sigma_max=args.adaptive_smear_sigma_max,
                scale=args.adaptive_sigma_scale,
            )
        )
        val_data_nll, val_data_w, val_mc_nll, val_mc_w = _epoch_metrics(
            model, val_loader, device,
            progress=args.progress, desc=f"epoch {epoch:>3}/{args.epochs} val",
            mc_only=args.mc_only, amp_ctx=amp_ctx,
            scale_noise_sigma=val_scale_sigma,
            smear_noise_sigma=val_smear_sigma,
            seed=args.val_seed,
        )
        # Early-stopping / best-model metric: the DATA-branch val NLL when
        # data is present (it is what the calibration actually fits); fall
        # back to the MC NLL for --mc-only / data-less runs.
        has_data = val_data_w > 0
        val_metric = val_data_nll if has_data else val_mc_nll
        dt = time.time() - t_epoch
        print(
            f"epoch {epoch:>3}: "
            f"train[data={train_data_nll:+.4f}(Σw={train_data_w:.2e}) "
            f"mc={train_mc_nll:+.4f}(Σw={train_mc_w:.2e})] "
            f"val[data={val_data_nll:+.4f}(Σw={val_data_w:.2e}) "
            f"mc={val_mc_nll:+.4f}(Σw={val_mc_w:.2e})] "
            f"θ_scale‖∞={model.theta_scale.abs().max().item():.3e} "
            f"σ_smear‖∞={model.effective_theta_smear().max().item():.3e} "
            f"lr={lr_str} dt={dt:.1f}s  [stop on {'data' if has_data else 'mc'} val]"
        )
        if val_data_w == 0 and val_mc_w == 0:
            print(
                "  WARNING: validation split was empty this epoch — early "
                "stopping will fire on the unchanged val_nll. Check "
                "--val-fraction / shard sizes."
            )

        improved = val_metric < best_val - args.patience_threshold
        if improved:
            best_val = val_metric
            no_improve = 0
        else:
            no_improve += 1

        # Always persist the latest model; additionally persist the best.
        ckpt_dict = {
            "epoch": epoch,
            "state_dict": {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            },
            "theta_scale": model.theta_scale.detach().cpu(),
            "theta_smear": model.theta_smear.detach().cpu(),
            "stats": _stats_to_dict(stats),
            "best_val": best_val,
            "val_metric": val_metric,
            "args": vars(args),
        }
        torch.save(ckpt_dict, last_ckpt)
        if improved:
            torch.save(ckpt_dict, best_ckpt)

        # Step the LR schedule; a reduction in ANY group restarts the early-stop window.
        if sched is not None:
            lr_before = [g["lr"] for g in optim.param_groups]
            sched.step(val_metric) if sched_kind == "plateau" else sched.step()
            if any(g["lr"] < lb - 1e-12 for g, lb in zip(optim.param_groups, lr_before)):
                no_improve = 0
        if not improved and not args.no_early_stop and no_improve >= args.patience:
            print(f"early-stop: no improvement for {no_improve} epochs")
            break

    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"reloaded best checkpoint ({best_ckpt}, val={ckpt['best_val']:+.4f})")

    if args.fisher_info and args.disable_scale:
        print("skipping Fisher info: θ_scale is disabled (--disable-scale).")
    elif args.fisher_info:
        print("computing plug-in Fisher information on θ_scale")
        fisher_loader = JpsiMassArrowLoader(
            shard_files,
            stats,
            batch_size=args.batch_size,
            split="val",
            val_fraction=args.val_fraction,
            holdout_fraction=args.holdout_fraction,
            drop_last=False,
        )
        t0 = time.time()
        H = compute_fisher_info(model, fisher_loader, device)
        try:
            cov = torch.linalg.inv(H.view(72, 72)).view(24, 3, 24, 3)
            ok = True
        except RuntimeError as e:
            print(f"  warning: Hessian inversion failed ({e}); saving H only")
            cov = None
            ok = False
        out_path = os.path.join(args.output, "fisher_info.pt")
        torch.save(
            {
                "hessian_24_3_24_3": H,
                "covariance_24_3_24_3": cov,
                "ok": ok,
            },
            out_path,
        )
        print(f"  wrote {out_path} in {time.time() - t0:.1f}s")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="train_jpsi_mass_fit",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help="Arrow file(s) or director(y/ies) of .arrow files produced "
        "by jpsi_mass_fit_snapshot.py (MC and data combined).",
    )
    p.add_argument("--output", required=True, help="Output directory.")
    p.add_argument(
        "--stats-in", default=None,
        help="Path to a precomputed preproc-stats JSON; if unset, stats are "
        "computed from the input shards.",
    )
    p.add_argument(
        "--device", default=("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="Torch device.",
    )
    # Two-stage continuity pipeline (default) vs legacy forward-fold.
    p.add_argument(
        "--stage", choices=["both", "flow", "fit", "legacy"], default="both",
        help="Two-stage continuity training: 'flow' = stage 1 (nominal flow on "
        "simulation, no θ conditioning); 'fit' = stage 2 (freeze flow, fit θ + "
        "background on data via the analytic continuity tilt); 'both' = run 1 "
        "then 2 in-process (default); 'legacy' = old single-stage forward-fold "
        "with θ-conditioned flow.",
    )
    p.add_argument(
        "--flow-checkpoint", type=str, default=None,
        help="Stage-1 flow checkpoint to load for '--stage fit' "
        "(default: <output>/flow_best.pt). The flow architecture "
        "(--flow-arch/--flow-n-transforms/--flow-hidden/--flow-n-hidden/"
        "--gf-components/--nsf-bins) and the preproc stats are read from this "
        "checkpoint automatically, so they need not be re-specified; only the "
        "flow weights are loaded (the MLP and θ start fresh). Pass --stats-in "
        "to override the stats.",
    )
    p.add_argument(
        "--validation", action="store_true",
        help="MC-closure validation mode: use simulation for BOTH stages "
        "instead of data for stage 2. A deterministic disjoint half of the "
        "simulation events trains the stage-1 flow (half 0); the other half "
        "(half 1) is treated as pseudo-data for the stage-2 θ fit. The disjoint "
        "split prevents stage 2 from fitting θ against the events stage 1 "
        "trained on; the closure target is θ → 0. Two-stage pipeline only "
        "(ignored for --stage legacy); any real data in the inputs is unused.",
    )
    p.add_argument("--flow-epochs", type=int, default=0,
                   help="Max epochs for stage 1 (0 → use --epochs).")
    p.add_argument("--fit-epochs", type=int, default=0,
                   help="Max epochs for stage 2 (0 → use --epochs).")
    p.add_argument("--fit-scale-lr", type=float, default=1e-5,
                   help="Stage-2 Adam lr for θ_scale (continuity increments).")
    p.add_argument("--fit-smear-lr", type=float, default=1e-3,
                   help="Stage-2 Adam lr for θ_smear (softplus-raw; natural "
                   "scale O(1)).")
    p.add_argument("--fit-mlp-lr", type=float, default=1e-3,
                   help="Stage-2 Adam lr for the background-fraction MLP.")
    p.add_argument("--continuity-n-gh", type=int, default=5,
                   help="Gauss–Hermite nodes for the #2 smear convolution in the "
                   "stage-2 signal density (more = more accurate, more flow evals).")
    p.add_argument("--continuity-n-iter", type=int, default=2,
                   help="Fixed-point iterations for the #2 source solve "
                   "(advection+smear pre-image).")
    p.add_argument("--epochs", type=int, default=50, help="Maximum training epochs.")
    p.add_argument("--batch-size", type=int, default=65536, help="Events per batch.")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Adam lr for flow + MLP.")
    p.add_argument(
        "--theta-scale-lr", type=float, default=1e-6,
        help="Adam lr for θ_scale. Kept slow so the (MC-trained) flow "
        "template converges before θ moves — the standardised conditioning "
        "makes the θ gradients large, and Adam's step is ~lr per batch.",
    )
    p.add_argument(
        "--theta-smear-lr", type=float, default=1e-2,
        help="Adam lr for θ_smear. θ_smear is the *raw* (pre-softplus) "
        "parameter, which lives in a log-like O(1) space — Adam's step is "
        "~lr per batch in raw units, so this must be ~10⁴× larger than the "
        "linear θ_scale lr or the (softplus-saturated) param never moves off "
        "its init.",
    )
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="Adam weight decay (L2) on all optimized parameters.")
    p.add_argument("--patience", type=int, default=8,
                   help="Early-stop after this many epochs without val improvement.")
    p.add_argument("--patience-threshold", type=float, default=1e-4,
                   help="Minimum val-NLL decrease that counts as an improvement.")
    p.add_argument("--no-early-stop", action="store_true",
                   help="Disable early stopping (train the full --epochs).")
    p.add_argument("--lr-schedule", choices=["plateau", "cosine", "none"],
                   default="plateau",
                   help="LR schedule (both stages + legacy). 'plateau': reduce "
                   "lr on val plateau (ReduceLROnPlateau); 'cosine': decay to "
                   "--min-lr over --epochs; 'none': fixed lr.")
    p.add_argument("--lr-reduce-factor", type=float, default=0.3,
                   help="Plateau schedule: multiply lr by this on a plateau.")
    p.add_argument("--lr-reduce-patience", type=int, default=4,
                   help="Plateau schedule: epochs without val improvement before "
                   "reducing lr (keep < --patience so lr drops before early-stop).")
    p.add_argument("--min-lr", type=float, default=1e-7,
                   help="Floor on the scheduled lr; early-stop fires once the lr "
                   "has reached this (plateau) reductions are exhausted.")
    p.add_argument("--val-fraction", type=float, default=0.10,
                   help="Fraction of events held out for validation.")
    p.add_argument("--holdout-fraction", type=float, default=0.05,
                   help="Fraction held out from train+val (e.g. for Fisher info).")
    p.add_argument("--m-lo", type=float, default=2.92, dest="m_lo",
                   help="Lower edge of the m_ll fit window [GeV].")
    p.add_argument("--m-hi", type=float, default=3.28, dest="m_hi",
                   help="Upper edge of the m_ll fit window [GeV].")
    # Scale transform + smear init + noise sampling
    p.add_argument(
        "--linearize-scale", action="store_true",
        help="Use the linearised mass-shift T_scale on the MC branch "
        "(m += J·θ) instead of the exact analytic qop transform. Only "
        "affects how the MC training mass is generated; the flow "
        "conditioning (θ_scale_pm) is identical.",
    )
    p.add_argument("--smear-init-a", type=float, default=1e-3,
                   help="Initial *effective* (physical, ≥0) θ_smear 'a' "
                   "(hit-resolution) term. Stored internally as softplus⁻¹(a) "
                   "since θ_smear is softplus-reparameterised. Default is a "
                   "small positive value — init 0 puts the raw param deep in "
                   "the softplus-saturated tail where it cannot train.")
    p.add_argument("--smear-init-c", type=float, default=1e-4,
                   help="Initial *effective* (physical, ≥0) θ_smear 'c' "
                   "(multiple-scattering) term (softplus-reparameterised). "
                   "Small positive default (see --smear-init-a).")
    p.add_argument(
        "--qop-floor-frac", type=float, default=0.25,
        help="Robustness floor for the qop→pt inversion in the scale/smear "
        "fold: the shifted |qop| is clamped (sign-preserving) to at least "
        "this fraction of |qop_orig|, so a smear/scale shift can neither flip "
        "the charge nor inflate pt by more than 1/this. Prevents catastrophic "
        "mass blow-ups at high |η| where |qop| is small. 0 disables.",
    )
    p.add_argument(
        "--smear-fit-params", choices=["both", "a", "c"], default="both",
        help="Which per-η-bin smear terms to FIT: 'both', 'a' (constant term "
        "only), or 'c' (∝1/pt term only). The constant a and the c·k term are "
        "nearly degenerate over the narrow J/ψ pt range, so fitting both per "
        "bin is ill-posed and yields the unphysical bin-to-bin zig-zag. The "
        "non-fitted term is set to 0 (post-softplus) — removed from the σ_qop "
        "kernel and the conditioning entirely, not floated.",
    )
    p.add_argument(
        "--smear-noise-sigma",
        type=float,
        default=0.0,
        help="Fallback σ for sampling θ̃_smear around the model value on "
        "the MC branch (so the flow learns its θ_smear-dependence). Used "
        "during adaptive-σ warmup or with --no-adaptive-sigma. NOTE: θ_smear "
        "is softplus-reparameterised, so this σ is in the *raw* (pre-softplus) "
        "space — its natural scale is O(1), not the ~1e-3 of the effective a/c. "
        "0 → sample at model.theta_smear.detach() exactly.",
    )
    # θ_scale sampling widths, split per component (A, e, M) since they live
    # in different physical units. Each is the σ of the Gaussian added to that
    # component (additive, physical units) — the fixed width with
    # --fixed-theta-sampling / --no-adaptive-sigma, and the adaptive-σ fallback
    # during warmup.
    p.add_argument("--scale-noise-sigma-A", type=float, default=1e-3,
                   help="Sampling σ for θ̃_scale 'A' (additive, A units).")
    p.add_argument("--scale-noise-sigma-e", type=float, default=1e-2,
                   help="Sampling σ for θ̃_scale 'e' (additive, GeV).")
    p.add_argument("--scale-noise-sigma-M", type=float, default=1e-4,
                   help="Sampling σ for θ̃_scale 'M' (additive, 1/GeV).")
    # Flow / MLP hyperparams
    p.add_argument(
        "--flow-arch", choices=("gf", "nsf"), default="nsf",
        help="Signal flow architecture: 'nsf' = neural rational-quadratic "
        "spline flow (default) — bounded (linear tails outside ±5), so it "
        "avoids the erf/exp saturation the GF needs guards for. 'gf' = "
        "Gaussianization flow (for comparison).",
    )
    p.add_argument(
        "--nsf-bins", type=int, default=8,
        help="(--flow-arch nsf only) Number of rational-quadratic spline "
        "knots per transform.",
    )
    p.add_argument("--flow-n-transforms", type=int, default=5,
                   help="Number of stacked flow transforms.")
    p.add_argument("--flow-hidden", type=int, default=128,
                   help="Hidden width of each flow conditioner MLP.")
    p.add_argument("--flow-n-hidden", type=int, default=3,
                   help="Number of hidden layers in each flow conditioner MLP.")
    p.add_argument("--gf-components", type=int, default=8,
                   help="(--flow-arch gf only) Gaussian-mixture components per layer.")
    p.add_argument("--mlp-hidden", type=int, default=32,
                   help="Hidden width of the data-branch mixture MLP.")
    p.add_argument("--mlp-n-layers", type=int, default=2,
                   help="Number of hidden layers in the mixture MLP.")
    # Fisher
    p.add_argument("--fisher-info", action="store_true",
                   help="After training, compute + save the observed Fisher "
                   "information (Hessian + covariance) → <output>/fisher_info.pt. "
                   "Two-stage pipeline: over θ_scale + the ACTIVE θ_smear column(s) "
                   "jointly, with the flow and background MLP held fixed (fixed-φ / "
                   "conditional). Legacy pipeline: θ_scale only.")
    p.add_argument("--fisher-split", default="train",
                   choices=("train", "val", "holdout", "all"),
                   help="(two-stage) Loader split the Fisher info is summed over. "
                   "Default 'train' = the data the fit used, so the covariance has "
                   "the correct statistical scale (∝ 1/N_fit). In --validation mode "
                   "the half-1 MC pseudo-data of this split is used.")
    # Mixed-precision + torch.compile (same convention as
    # train_muon_response_flow.py).
    p.add_argument(
        "--compile", action="store_true",
        help="Wrap the model in torch.compile. Currently SKIPPED — the "
        "GF base flow's Jacobian uses an inner torch.autograd.grad call "
        "(zuko's MonotonicTransform.call_and_ladj) that dynamo cannot "
        "trace. Flag is accepted for parity with the response-flow "
        "trainer; a one-line note is printed at startup.",
    )
    p.add_argument(
        "--precision", choices=("fp32", "bf16", "fp16"), default="fp32",
        help="Training + validation forward-pass precision. fp32 = no "
        "autocast. bf16 = bfloat16 autocast (Ampere+, no GradScaler). "
        "fp16 = float16 autocast + enabled GradScaler for loss scaling.",
    )
    p.add_argument(
        "--progress", default=True, action=argparse.BooleanOptionalAction,
        help="Show a per-epoch tqdm progress bar with running NLL "
        "(--no-progress to disable).",
    )
    # Adaptive σ from Adam's second moment.
    p.add_argument(
        "--adaptive-sigma", default=True, action=argparse.BooleanOptionalAction,
        help="Use per-parameter σ ≈ scale/√(Adam v_i) for θ_smear / "
        "θ_scale noise sampling instead of the fixed --smear-noise-sigma "
        "/ --scale-noise-sigma-{A,e,M} widths. Falls back to those during warmup.",
    )
    p.add_argument(
        "--adaptive-warmup-steps", type=int, default=100,
        help="Optimizer steps to skip before switching from fixed σ to "
        "Adam-derived σ (gives v time to populate meaningfully).",
    )
    # Adaptive-σ clamps are split scale/smear because the two parameters live
    # in different units: θ_scale is the linear (A,e,M) ~1e-4…1e-2, while
    # θ_smear is the *raw* (pre-softplus) param whose natural sampling scale is
    # O(1). The floor prevents σ→0 collapse near sharp optima; the ceiling
    # prevents σ→∞ on parameters that haven't received gradient yet (v_i = 0).
    p.add_argument(
        "--adaptive-scale-sigma-min", type=float, default=1e-6,
        help="Floor on the adaptive σ for θ_scale (linear units).",
    )
    p.add_argument(
        "--adaptive-scale-sigma-max", type=float, default=1e-2,
        help="Ceiling on the adaptive σ for θ_scale (linear units).",
    )
    p.add_argument(
        "--adaptive-smear-sigma-min", type=float, default=1e-2,
        help="Floor on the adaptive σ for θ_smear (raw pre-softplus space; "
        "natural scale O(1)).",
    )
    p.add_argument(
        "--adaptive-smear-sigma-max", type=float, default=0.3,
        help="Ceiling on the adaptive σ for θ_smear (raw pre-softplus space). "
        "Kept moderate (0.3): a wide sampling band makes the flow's θ_smear "
        "conditioning poorly resolved and lets the fitted per-bin smear wander "
        "(was 1.0, which sampled effective a/c over a factor ~e).",
    )
    p.add_argument(
        "--adaptive-sigma-scale", type=float, default=1.0,
        help="Multiplier on the 1/√v_hat target. 1 ≈ Cramér-Rao-natural "
        "width.",
    )
    p.add_argument(
        "--val-seed", type=int, default=42,
        help="Fixed RNG seed used at the start of every validation pass. "
        "Keeps the σ̃_smear noise (and the per-event smearing-kernel ε) "
        "deterministic across epochs so val_nll has no Monte Carlo "
        "fluctuation and can be compared apples-to-apples with train_nll.",
    )
    # Debug flags — bisect the model down to the smallest piece that
    # reproduces the failure (NaN, instability, etc.).
    p.add_argument(
        "--mc-only", action="store_true",
        help="Train only on MC events (skip the data branch entirely — no "
        "MLP, no Bernstein mixture). The flow still learns the (θ_scale, "
        "θ_smear) conditioning from the sampled MC forward-fold. Useful for "
        "confirming the flow alone can fit MC.",
    )
    p.add_argument(
        "--disable-smearing", action="store_true",
        help="Drop the residual-smearing kernel and the θ_smear "
        "conditioning of the flow. θ_smear is excluded from the optimizer "
        "and never sampled (it stays an inert, fixed Parameter).",
    )
    p.add_argument(
        "--disable-scale", action="store_true",
        help="Drop the T_scale forward-fold and the θ_scale conditioning of "
        "the flow. θ_scale is excluded from the optimizer and never sampled "
        "(inert, fixed at 0); Fisher info is skipped. With BOTH "
        "--disable-scale and --disable-smearing, only the flow and the MLP "
        "(background normalisation) are trained.",
    )
    p.add_argument(
        "--detach-flow-on-data", default=True,
        action=argparse.BooleanOptionalAction,
        help="Exclude the flow's parameters from the data-branch gradient "
        "(default: on). The signal template is then trained on MC only; the "
        "data branch still floats θ_scale, θ_smear and the MLP mixture "
        "through the flow's inputs. Pins the template shape to the trusted "
        "MC and breaks the θ_scale ↔ flow-location degeneracy. Use "
        "--no-detach-flow-on-data to let data reshape the flow too.",
    )
    p.add_argument(
        "--fixed-theta-sampling", action="store_true",
        help="Sample θ̃_scale / θ̃_smear for the MC-branch flow training "
        "around a FIXED centre (the init: θ_scale=0, θ_smear=init) with FIXED "
        "widths (--scale-noise-sigma-{A,e,M} / --smear-noise-sigma), "
        "independent of the current model values, and bypassing the adaptive "
        "σ. The flow then learns a stable θ-conditional family over a fixed "
        "region rather than one that chases the fit. Set the --*-noise-sigma "
        "widths > 0 (note --smear-noise-sigma defaults to 0).",
    )
    p.add_argument(
        "--detect-anomaly", action="store_true",
        help="Wrap training in torch.autograd.set_detect_anomaly so the "
        "first NaN in the backward graph throws with a stack trace "
        "pointing to the offending op. Slow — debug only.",
    )
    p.add_argument(
        "--nan-on-step", default="raise", choices=("raise", "skip", "ignore"),
        help="What to do when the per-batch loss is NaN/Inf. 'raise' "
        "stops training immediately with the batch index. 'skip' drops "
        "the batch and continues. 'ignore' lets the NaN propagate (the "
        "original behaviour).",
    )
    return p.parse_args(argv)


def main() -> int:
    return train_loop(parse_args())


if __name__ == "__main__":
    sys.exit(main())
