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
from jpsi_mass_model import (
    JpsiMassMixtureModel, SMEAR_VAR_SCALE_A, SMEAR_VAR_SCALE_C, THETA_SCALE_REF)


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



def _smear_active_cols(model) -> List[int]:
    """Column indices of ``theta_smear`` actually fit (smear_param_mask != 0).
    The non-fit column is zeroed post-softplus → identically zero gradient, so
    it must be excluded from the Fisher Hessian (else a singular row/col)."""
    return [c for c in range(model.theta_smear.shape[1])
            if float(model.smear_param_mask[c]) != 0.0]


def _active_param_labels(model, smear_cols) -> List[str]:
    """Per-active-parameter labels in the [θ_scale | active θ_smear] order
    shared by the Fisher and bootstrap covariances."""
    comp = ["A", "e", "M"]
    smear_comp = ["a", "c"]
    labels: List[str] = []
    if model.scale_enabled:
        for b in range(model.theta_scale.shape[0]):
            for c in range(model.theta_scale.shape[1]):
                labels.append(f"{comp[c]}[{b}]")
    for b in range(model.theta_smear.shape[0]):
        for c in smear_cols:
            labels.append(f"{smear_comp[c]}[{b}](raw)")
    return labels


def _record_active_theta(model, smear_cols) -> torch.Tensor:
    """Flat active parameter vector ``[θ_scale (linear) | active θ_smear (raw)]``
    on cpu, matching the Fisher's active ordering (bin-major, col-minor)."""
    parts = []
    if model.scale_enabled:
        parts.append(model.theta_scale.detach().reshape(-1))
    if smear_cols:
        parts.append(model.theta_smear.detach()[:, smear_cols].reshape(-1))
    return torch.cat(parts).detach().cpu()


def _hessian_block_loop(g_full, params, active_idx, n_act):
    """Per-batch active Hessian block ``[n_act, n_act]`` by one second backward
    per row (robust; works through the nested autograd in the continuity
    density). ``g_full`` is the flat first-order gradient with its graph kept."""
    H = torch.zeros((n_act, n_act), device=g_full.device, dtype=g_full.dtype)
    for r in range(n_act):
        i = int(active_idx[r])
        row = torch.autograd.grad(
            g_full[i], params, retain_graph=True, allow_unused=True)
        row_full = torch.cat([
            (ri if ri is not None else torch.zeros_like(p)).reshape(-1)
            for ri, p in zip(row, params)])
        H[r] = row_full[active_idx]
    return H


def _hessian_block_batched(g_active, params, active_idx, n_act):
    """Per-batch active Hessian block ``[n_act, n_act]`` in ONE vectorised second
    backward via ``is_grads_batched`` (vmaps the per-row vjp over the identity
    basis). Faster than the loop but holds ~n_act copies of the backward graph,
    and the engine's vmap may not support every op in this double-backward path
    (the inner ``autograd.grad`` of the change-of-variables Jacobian, the
    fixed-point clamps) — the caller falls back to the loop on failure/OOM."""
    eye = torch.eye(n_act, device=g_active.device, dtype=g_active.dtype)
    rows = torch.autograd.grad(
        g_active, params, grad_outputs=eye,
        is_grads_batched=True, retain_graph=True, allow_unused=True)
    parts = []
    for ri, p in zip(rows, params):
        if ri is None:
            parts.append(torch.zeros(n_act, p.numel(),
                                     device=g_active.device, dtype=g_active.dtype))
        else:
            parts.append(ri.reshape(n_act, -1))
    row_full = torch.cat(parts, dim=1)        # [n_act, n_full]
    return row_full[:, active_idx]            # [n_act, n_act]


def compute_fisher_info_continuity(
    model: JpsiMassMixtureModel,
    loader: JpsiMassArrowLoader,
    device: str,
    *,
    mc_as_data: bool = False,
    n_iter: int = 2,
    progress: bool = True,
    vectorized: bool = True,
):
    """Observed (plug-in) Fisher information for the two-stage continuity fit,
    over ``theta_scale`` + the ACTIVE ``theta_smear`` columns jointly, with the
    flow and the background MLP held FIXED (option 1: conditional / fixed-φ).

    H = Σ_events w · ∂²(−ln p_mixture)/∂θ² at the fit point, on the data branch
    (``data_nll_continuity`` — the objective stage 2 actually minimises).
    ``theta_smear`` are the signed width coefficients.

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
    use_batched = bool(vectorized)  # may flip to False after a fallback
    bar = tqdm(loader, desc="fisher", disable=not progress, unit="batch")
    for batch in bar:
        batch = _move_batch(batch, device)
        data_mask = ~batch["is_data_mask"] if mc_as_data else batch["is_data_mask"]
        if not bool(data_mask.any()):
            continue
        per = model.data_nll_continuity(
            batch["mll"], batch["pt_pm"], batch["eta_pm"], batch["phi_pm"],
            batch["q_pm"], batch["b_pm"], batch["muon_kin_std"], data_mask,
            n_iter=n_iter)
        w = batch["w"] * data_mask.to(batch["w"].dtype)
        nll = (w * per).sum()
        if not torch.isfinite(nll):
            continue
        g = torch.autograd.grad(nll, params, create_graph=True)
        g_full = torch.cat([gi.reshape(-1) for gi in g])
        g_active = g_full[active_idx]
        grad += g_active.detach()
        # Vectorised second backward (one vmapped vjp over the identity basis);
        # fall back to the per-row loop on any engine failure / OOM, once.
        if use_batched:
            try:
                Hb = _hessian_block_batched(g_active, params, active_idx, n_act)
            except (RuntimeError, NotImplementedError) as e:
                use_batched = False
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                bar.write(
                    f"  note: vectorised Hessian unavailable "
                    f"({type(e).__name__}: {str(e).splitlines()[0][:80]}); "
                    f"using the per-row loop")
                Hb = _hessian_block_loop(g_full, params, active_idx, n_act)
        else:
            Hb = _hessian_block_loop(g_full, params, active_idx, n_act)
        H += Hb.detach()
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
    the 24×3×24×3 layout for the diagnostics, and delta-method effective
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
    labels = _active_param_labels(model, smear_cols)

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
    # θ_scale block in the 24×3×24×3 layout (consumed by the diagnostics for ±1σ
    # bands + the correlation heatmap). This is the scale block of the JOINT
    # inverse, so it carries the θ_scale↔θ_smear correlation.
    if n_scale == 72:
        out["hessian_24_3_24_3"] = H[:72, :72].view(24, 3, 24, 3)
        if cov is not None:
            # θ_scale is O(1); physical (A,e,M) = θ·THETA_SCALE_REF → rescale the
            # covariance block to PHYSICAL units (cov_phys = cov·ref⊗ref).
            ref72 = torch.tensor(list(THETA_SCALE_REF) * 24, dtype=cov.dtype)
            cov_scale = cov[:72, :72] * ref72.unsqueeze(0) * ref72.unsqueeze(1)
            out["covariance_24_3_24_3"] = cov_scale.view(24, 3, 24, 3)
            out["sigma_scale_24_3"] = torch.sqrt(
                torch.clamp(torch.diag(cov_scale), min=0.0)).view(24, 3)
    # PHYSICAL σ on the smear a/c per η-bin. The fit param θ_smear is O(1); the
    # physical qop-variance coefficient is θ·SMEAR_VAR_SCALE, so σ_phys =
    # SMEAR_VAR_SCALE·σ_raw (linear scaling). Inactive columns → 0.
    if smear_cols and cov is not None:
        n_eta, n_comp = model.theta_smear.shape
        cov_smear = cov[n_scale:, n_scale:]
        sig_raw = torch.sqrt(torch.clamp(torch.diag(cov_smear), min=0.0))
        smear_scale = (SMEAR_VAR_SCALE_A, SMEAR_VAR_SCALE_C)
        sig_eff = torch.zeros(n_eta, n_comp)
        k = 0
        for b in range(n_eta):
            for c in smear_cols:
                sig_eff[b, c] = smear_scale[c] * sig_raw[k]
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


def _make_loaders(args, shard_files, stats, *, half=None, inject_theta=None,
                  inject_smear=None):
    """Build the ``(train, val)`` loaders for one stage. ``half`` selects a
    deterministic disjoint event half (0/1) — used by the MC-closure
    validation mode (stage 1 ← half 0, stage 2 ← half 1); ``None`` = all
    events. ``inject_theta`` ([n_eta, 3]) / ``inject_smear`` ([n_eta, 2]) inject
    a known θ_scale shift / per-muon qop smear into the (pseudo-)data m_ll
    (validation closure)."""
    seed = int(getattr(args, "inject_smear_seed", 12345))
    train_loader = JpsiMassArrowLoader(
        shard_files, stats, batch_size=args.batch_size, split="train",
        val_fraction=args.val_fraction, holdout_fraction=args.holdout_fraction,
        drop_last=True, half=half, inject_theta_scale=inject_theta,
        inject_theta_smear=inject_smear, inject_seed=seed)
    val_loader = JpsiMassArrowLoader(
        shard_files, stats, batch_size=args.batch_size, split="val",
        val_fraction=args.val_fraction, holdout_fraction=args.holdout_fraction,
        drop_last=False, half=half, inject_theta_scale=inject_theta,
        inject_theta_smear=inject_smear, inject_seed=seed)
    return train_loader, val_loader


def _inject_theta_np(args, n_eta):
    """``[n_eta, 3]`` injected θ_scale (constant A, e, M across η) for the
    validation closure, or ``None`` if no shift was requested."""
    a = float(getattr(args, "inject_A", 0.0) or 0.0)
    e = float(getattr(args, "inject_e", 0.0) or 0.0)
    m = float(getattr(args, "inject_M", 0.0) or 0.0)
    if a == 0.0 and e == 0.0 and m == 0.0:
        return None
    t = np.zeros((int(n_eta), 3), dtype=np.float64)
    t[:, 0] = a; t[:, 1] = e; t[:, 2] = m
    return t


def _inject_smear_np(args, n_eta):
    """``[n_eta, 2]`` injected PHYSICAL qop-variance coefficients (a, c) for the
    validation closure (or ``None``): σ²_qop = a + c·k², the values the loader
    applies directly. ``--inject-a/-c`` are the physical coefficients (a ~ 1e-7,
    c ~ 1e-6 are typical); the O(1) optimizer rescaling (SMEAR_VAR_SCALE) is an
    internal detail applied to the fit parameter, not here."""
    a = float(getattr(args, "inject_a", 0.0) or 0.0)
    c = float(getattr(args, "inject_c", 0.0) or 0.0)
    if a == 0.0 and c == 0.0:
        return None
    t = np.zeros((int(n_eta), 2), dtype=np.float64)
    t[:, 0] = a; t[:, 1] = c
    return t


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


def _build_model(args, stats, device):
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
        smearing_enabled=not args.disable_smearing,
        scale_enabled=not args.disable_scale,
        qop_floor_frac=args.qop_floor_frac, smear_fit_params=args.smear_fit_params,
        smear_flow_steps=getattr(args, "smear_flow_steps", 1),
        theta_mode=("mlp" if getattr(args, "theta_mlp", False) else "binned"),
        theta_mlp_hidden=getattr(args, "theta_mlp_hidden", 32),
        theta_mlp_layers=getattr(args, "theta_mlp_layers", 2),
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
            if model.theta_mode == "mlp":
                wn = max((p.detach().abs().max().item()
                          for p in model.theta_net.parameters()), default=0.0)
                extra = f" θ_net‖w‖∞={wn:.3e}"
            else:
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
    groups = [{"params": model.mlp.parameters(), "lr": args.fit_mlp_lr}]
    tags = ["mlp"]
    if model.theta_mode == "mlp":
        # Continuous θ(η,φ): float the ThetaNet (zero-init → 0). One group;
        # the output reference scaling differentiates the A,e,M vs a,c magnitudes.
        groups.append({"params": model.theta_net.parameters(),
                       "lr": args.fit_theta_mlp_lr}); tags.append("θ_net")
        print(f"  optimizer groups: {', '.join(tags)}  "
              f"(lr mlp={args.fit_mlp_lr:g} θ_net={args.fit_theta_mlp_lr:g}); "
              f"θ = ThetaNet(η, φ) [continuous]")
    else:
        # θ_scale is the advective shift (init 0, signed). θ_smear are the signed
        # qop-variance coefficients (init 0 → σ²_qop=0, identity).
        with torch.no_grad():
            model.theta_scale.zero_()
        if not args.disable_scale:
            groups.append({"params": [model.theta_scale], "lr": args.fit_scale_lr}); tags.append("θ_scale")
        if not args.disable_smearing:
            groups.append({"params": [model.theta_smear], "lr": args.fit_smear_lr}); tags.append("θ_smear")
        print(f"  optimizer groups: {', '.join(tags)}  "
              f"(lr mlp={args.fit_mlp_lr:g} scale={args.fit_scale_lr:g} smear={args.fit_smear_lr:g})")
    optim = torch.optim.Adam(groups)
    print(f"  signal density: #2 direct-eval (advection + probability-flow smear, "
          f"flow_steps={getattr(args, 'smear_flow_steps', 1)}, n_iter="
          f"{args.continuity_n_iter}); normalised by construction")

    def step2(model, batch):
        # In validation mode the simulation rows play the role of data.
        data_mask = ~batch["is_data_mask"] if mc_as_data else batch["is_data_mask"]
        per = model.data_nll_continuity(
            batch["mll"], batch["pt_pm"], batch["eta_pm"], batch["phi_pm"],
            batch["q_pm"], batch["b_pm"], batch["muon_kin_std"], data_mask,
            n_iter=args.continuity_n_iter)
        w = batch["w"] * data_mask.to(batch["w"].dtype)
        sw = float(w.sum().clamp_min(1e-30))
        return (w * per).sum() / sw, sw

    return _run_epochs(args, model, optim, train_loader, val_loader, stats,
                       step_fn=step2, ckpt_prefix="fit", stage_name="fit",
                       epochs=args.fit_epochs or args.epochs)


def train_loop(args: argparse.Namespace) -> int:
    """The two-stage continuity pipeline (stage 1 flow → stage 2 fit)."""
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("warning: CUDA requested but not available; falling back to CPU")
        device = args.device = "cpu"

    # --stage uncertainties: load an existing FULL fit (flow + MLP + θ) from
    # --checkpoint and run only the uncertainty estimation (Fisher / bootstrap),
    # no training. Handled in its own path.
    if args.stage == "uncertainties":
        return _run_uncertainties_stage(args, device)

    # --stage fit: load the flow checkpoint up front so its architecture and
    # stats drive the model build + standardisation (the model is rebuilt from
    # args before the flow weights are loaded, so they must match).
    flow_ck = None
    flow_ckpt = None
    stats_override = None
    if args.stage == "fit":
        flow_ckpt = args.checkpoint or os.path.join(args.output, "flow_best.pt")
        if not os.path.exists(flow_ckpt):
            print(f"error: --stage fit needs a stage-1 flow; {flow_ckpt!r} not found "
                  f"(run --stage flow first or pass --checkpoint)", file=sys.stderr)
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

    model = _build_model(args, stats, device)
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
        inj = _inject_theta_np(args, len(stats.eta_edges) - 1)
        inj_sm = _inject_smear_np(args, len(stats.eta_edges) - 1)
        tgt = "θ → 0"
        if inj is not None or inj_sm is not None:
            parts = []
            if inj is not None:
                parts.append(f"(A,e,M)=({args.inject_A:g},{args.inject_e:g},{args.inject_M:g})")
            if inj_sm is not None:
                parts.append(f"(a,c)=({args.inject_a:g},{args.inject_c:g})")
            tgt = "θ → injected " + " ".join(parts)
        print(f"\n*** MC-closure validation mode: simulation for both stages "
              f"(stage 1 ← half 0, stage 2 ← half 1 as pseudo-data); target {tgt} ***")
        if inj is not None:
            print("    injecting the θ_scale shift into the stage-2 pseudo-data m_ll")
        if inj_sm is not None:
            print("    injecting the per-muon qop smear into the stage-2 pseudo-data m_ll")
        s1_train, s1_val = _make_loaders(args, shard_files, stats, half=0)  # flow: NOT injected
        s2_train, s2_val = _make_loaders(args, shard_files, stats, half=1,
                                         inject_theta=inj, inject_smear=inj_sm)
    else:
        if (_inject_theta_np(args, len(stats.eta_edges) - 1) is not None
                or _inject_smear_np(args, len(stats.eta_edges) - 1) is not None):
            print("warning: --inject-A/e/M/a/c only apply in --validation mode; ignoring.",
                  file=sys.stderr)
        s1_train, s1_val = train_loader, val_loader
        s2_train, s2_val = train_loader, val_loader

    if args.stage in ("both", "flow"):
        train_stage1(args, model, s1_train, s1_val, stats)
    if args.stage in ("both", "fit"):
        train_stage2(args, model, s2_train, s2_val, stats, mc_as_data=args.validation)
        if args.fisher_info:
            _run_fisher_continuity(args, model, shard_files, stats, device)
        if args.empirical_fisher:
            _run_empirical_fisher(args, model, shard_files, stats, device)
        if args.bootstrap > 0:
            run_bootstrap_continuity(args, model, shard_files, stats, device,
                                     mc_as_data=args.validation)
    return 0


def _run_fisher_continuity(args, model, shard_files, stats, device) -> None:
    """Observed Fisher info for the two-stage fit (θ_scale + active θ_smear,
    fixed flow + MLP) → ``<output>/fisher_info.pt``. Evaluated on the data the
    fit used (``--fisher-split``, default the train split; half 1 in
    validation mode), so the covariance has the right statistical scale."""
    if not (model.scale_enabled or model.smearing_enabled):
        print("skipping Fisher info: both --disable-scale and --disable-smearing.")
        return
    if model.theta_mode == "mlp":
        print("skipping observed Fisher info: not implemented for --theta-mlp "
              "(binned 24×3 θ layout); use --empirical-fisher over the net weights "
              "or re-run binned for the per-bin covariance.", file=sys.stderr)
        return
    half = 1 if args.validation else None
    inj = _inject_theta_np(args, len(stats.eta_edges) - 1) if args.validation else None
    inj_sm = _inject_smear_np(args, len(stats.eta_edges) - 1) if args.validation else None
    loader = JpsiMassArrowLoader(
        shard_files, stats, batch_size=args.batch_size, split=args.fisher_split,
        val_fraction=args.val_fraction, holdout_fraction=args.holdout_fraction,
        drop_last=False, half=half, inject_theta_scale=inj,
        inject_theta_smear=inj_sm, inject_seed=int(args.inject_smear_seed))
    print(f"\ncomputing observed Fisher information (θ_scale + active θ_smear, "
          f"fixed flow + MLP) on split={args.fisher_split}"
          + ("  half=1 (MC pseudo-data)" if args.validation else "")
          + f"  [smear_fit={model.smear_fit_params}]")
    t0 = time.time()
    H, layout = compute_fisher_info_continuity(
        model, loader, device, mc_as_data=args.validation,
        n_iter=args.continuity_n_iter,
        progress=args.progress, vectorized=args.fisher_vectorized)
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


def _bootstrap_save_dict(cov, mean, TH, eff_smears, conv_epochs, model, smear_cols):
    """Package the warm-start bootstrap covariance, diagnostics-compatible with
    fisher_info.pt (same θ_scale-block / σ keys; no Hessian/EDM keys)."""
    n_scale = model.theta_scale.numel() if model.scale_enabled else 0
    out = {
        "method": "warm-start Poisson bootstrap",
        "covariance": cov,
        "mean": mean,
        "replicas": TH,                       # [B, n_act] raw active vectors
        "labels": _active_param_labels(model, smear_cols),
        "n_scale": n_scale,
        "smear_cols": smear_cols,
        "smear_fit_params": model.smear_fit_params,
        "n_replicas": int(TH.shape[0]),
        "convergence_epochs": conv_epochs,
        "param_space": "scale: linear (A,e,M); smear: raw pre-softplus theta_smear",
    }
    if n_scale == 72:
        # θ_scale O(1) → physical (A,e,M) = θ·THETA_SCALE_REF; cov_phys = cov·ref⊗ref.
        ref72 = torch.tensor(list(THETA_SCALE_REF) * 24, dtype=cov.dtype)
        cov_scale = cov[:72, :72] * ref72.unsqueeze(0) * ref72.unsqueeze(1)
        out["covariance_24_3_24_3"] = cov_scale.view(24, 3, 24, 3)
        out["sigma_scale_24_3"] = torch.sqrt(
            torch.clamp(torch.diag(cov_scale), min=0.0)).view(24, 3)
    # Physical smear σ straight from the replicas' effective_theta_smear (already
    # physical) — the bootstrap gives the effective-space spread exactly.
    if smear_cols and eff_smears:
        EFF = torch.stack(eff_smears)                       # [B, n_smear_active]
        sig = EFF.std(0, unbiased=True) if EFF.shape[0] > 1 else torch.zeros(EFF.shape[1])
        n_eta, n_comp = model.theta_smear.shape
        sig_eff = torch.zeros(n_eta, n_comp)
        k = 0
        for b in range(n_eta):
            for c in smear_cols:
                sig_eff[b, c] = sig[k]
                k += 1
        out["sigma_smear_eff_24_2"] = sig_eff
    return out


def run_bootstrap_continuity(args, model, shard_files, stats, device, *,
                             mc_as_data: bool) -> None:
    """Warm-start Poisson bootstrap of the stage-2 fit → ``<output>/bootstrap_cov.pt``.

    Each replica restarts from the nominal (θ̂, φ̂), resets Adam to the initial
    fit LRs (re-raised, so the replica can actually relax), and refits θ AND the
    background MLP jointly on the SAME data Poisson(1)-reweighted (an independent
    per-event count, fixed across the replica's epochs), until its reweighted NLL
    plateaus. The covariance of {θ̂_b} folds in the background (and every other)
    uncertainty with no Hessian. Warm-starting only changes how each replica
    reaches its optimum, not where — provided it re-converges (hence the per-
    replica early stop on the reweighted NLL)."""
    B = int(args.bootstrap)
    if B <= 0:
        return
    if model.theta_mode == "mlp":
        print("skipping bootstrap: not implemented for --theta-mlp (binned 24×3 "
              "θ layout / per-bin covariance).", file=sys.stderr)
        return
    if not (model.scale_enabled or model.smearing_enabled):
        print("skipping bootstrap: both --disable-scale and --disable-smearing.")
        return
    smear_cols = _smear_active_cols(model) if model.smearing_enabled else []
    half = 1 if mc_as_data else None
    inj = _inject_theta_np(args, len(stats.eta_edges) - 1) if mc_as_data else None
    inj_sm = _inject_smear_np(args, len(stats.eta_edges) - 1) if mc_as_data else None
    loader = JpsiMassArrowLoader(
        shard_files, stats, batch_size=args.batch_size, split=args.fisher_split,
        val_fraction=args.val_fraction, holdout_fraction=args.holdout_fraction,
        drop_last=False, half=half, inject_theta_scale=inj,
        inject_theta_smear=inj_sm, inject_seed=int(args.inject_smear_seed))
    nominal_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Freeze the flow; float the MLP + active θ (the MLP must re-fit per replica
    # so its uncertainty enters the spread).
    for p in model.flow.parameters():
        p.requires_grad_(False)
    for p in model.mlp.parameters():
        p.requires_grad_(True)
    model.theta_scale.requires_grad_(model.scale_enabled)
    model.theta_smear.requires_grad_(model.smearing_enabled)

    # Convergence settings synchronised with the nominal stage-2 fit: the
    # per-replica early stop (patience + threshold), epoch cap, and LR schedule
    # all default to the nominal fit's, so each replica converges as thoroughly
    # as the nominal (monitored on the replica's reweighted NLL, since a
    # bootstrap replica has no separate validation set).
    patience = (args.bootstrap_patience if args.bootstrap_patience is not None
                else args.patience)
    max_epochs = (args.bootstrap_epochs if args.bootstrap_epochs is not None
                  else (args.fit_epochs or args.epochs))
    print(f"\nwarm-start Poisson bootstrap: {B} replicas × ≤{max_epochs} epochs "
          f"on split={args.fisher_split}"
          + ("  half=1 (MC pseudo-data)" if mc_as_data else "")
          + f"  [smear_fit={model.smear_fit_params}; "
          + ("no early stop" if args.no_early_stop
             else f"patience={patience}, threshold={args.patience_threshold:g}")
          + f", lr-schedule={args.lr_schedule}]")
    gen = torch.Generator(device=device)
    replicas, eff_smears, conv_epochs = [], [], []
    n_batches_total = None   # learned on the first epoch → % on the inner bar after
    t0 = time.time()
    rbar = tqdm(range(B), desc="bootstrap", disable=not args.progress, unit="replica")
    for b in rbar:
        model.load_state_dict(nominal_sd)          # warm-start from the nominal fit
        groups = [{"params": model.mlp.parameters(), "lr": args.fit_mlp_lr}]
        if model.scale_enabled:
            groups.append({"params": [model.theta_scale], "lr": args.fit_scale_lr})
        if model.smearing_enabled:
            groups.append({"params": [model.theta_smear], "lr": args.fit_smear_lr})
        optim = torch.optim.Adam(groups)           # fresh Adam state, re-raised LR
        sched, sched_kind = _make_scheduler(args, optim, max_epochs)  # same as nominal
        seed = args.bootstrap_seed + b
        best = float("inf"); no_improve = 0; used = 0
        model.train()
        for epoch in range(max_epochs):
            gen.manual_seed(seed)                  # same per-event Poisson each epoch
            tr_sum = 0.0; tr_w = 0.0; n_seen = 0
            # Inner per-epoch bar over batches so a (slow) epoch visibly advances;
            # total is learned on epoch 1 so later epochs render a % bar.
            ebar = tqdm(loader, total=n_batches_total, leave=False,
                        desc=f"  replica {b + 1}/{B} epoch {epoch + 1}/{args.bootstrap_epochs}",
                        disable=not args.progress, unit="batch")
            for batch in ebar:
                n_seen += 1
                batch = _move_batch(batch, device)
                data_mask = ~batch["is_data_mask"] if mc_as_data else batch["is_data_mask"]
                if not bool(data_mask.any()):
                    continue
                pois = torch.poisson(
                    torch.ones(batch["mll"].shape[0], device=device), generator=gen)
                w = batch["w"] * pois * data_mask.to(batch["w"].dtype)
                sw = float(w.sum().clamp_min(1e-30))
                if sw <= 0:
                    continue
                per = model.data_nll_continuity(
                    batch["mll"], batch["pt_pm"], batch["eta_pm"], batch["phi_pm"],
                    batch["q_pm"], batch["b_pm"], batch["muon_kin_std"], data_mask,
                    n_iter=args.continuity_n_iter)
                loss = (w * per).sum() / sw
                if not torch.isfinite(loss):
                    continue
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                tr_sum += float(loss.item()) * sw; tr_w += sw
                ebar.set_postfix_str(f"nll={tr_sum / max(tr_w, 1e-30):+.4f}")
            ebar.close()
            if n_batches_total is None:
                n_batches_total = n_seen
            used = epoch + 1
            nll = tr_sum / max(tr_w, 1e-30)
            improved = nll < best - args.patience_threshold
            if improved:
                best = nll; no_improve = 0
            else:
                no_improve += 1
            # Same LR schedule + early-stop coupling as the nominal fit: a LR
            # reduction resets the early-stop counter (so it only fires once the
            # reductions are exhausted), monitored on the replica's reweighted NLL.
            if sched is not None:
                lr_before = [g["lr"] for g in optim.param_groups]
                sched.step(nll) if sched_kind == "plateau" else sched.step()
                if any(g["lr"] < lb - 1e-12 for g, lb in zip(optim.param_groups, lr_before)):
                    no_improve = 0
            if not improved and not args.no_early_stop and no_improve >= patience:
                break
        replicas.append(_record_active_theta(model, smear_cols))
        if smear_cols:
            eff_smears.append(
                model.effective_theta_smear().detach()[:, smear_cols].reshape(-1).cpu())
        conv_epochs.append(used)
        rbar.set_postfix_str(f"ep={used} nll={best:+.4f}")
    rbar.close()
    model.load_state_dict(nominal_sd)              # leave the model at the nominal fit

    TH = torch.stack(replicas)                     # [B, n_act]
    mean = TH.mean(0)
    Xc = TH - mean
    cov = (Xc.t() @ Xc) / max(B - 1, 1)
    out = _bootstrap_save_dict(cov, mean, TH, eff_smears, conv_epochs, model, smear_cols)
    path = os.path.join(args.output, "bootstrap_cov.pt")
    torch.save(out, path)
    ce = torch.tensor(conv_epochs, dtype=torch.float32)
    n_hit_cap = int((ce >= max_epochs).sum()) if not args.no_early_stop else 0
    print(f"  wrote {path}: {B} replicas over {TH.shape[1]} params; "
          f"epochs/replica median={int(ce.median())} max={int(ce.max())} "
          f"in {time.time()-t0:.1f}s")
    if n_hit_cap:
        print(f"  WARNING: {n_hit_cap}/{B} replica(s) hit the {max_epochs}-epoch "
              f"cap without plateauing — they may be under-converged (variance "
              f"underestimated). Raise --bootstrap-epochs.")
    if "sigma_scale_24_3" in out:
        ss = out["sigma_scale_24_3"]
        print(f"  bootstrap σ(A,e,M) median over bins = "
              f"({float(ss[:,0].median()):.2e}, {float(ss[:,1].median()):.2e}, "
              f"{float(ss[:,2].median()):.2e})")


def _theta_cov_extras(cov_theta: torch.Tensor, model, smear_cols, n_scale: int) -> dict:
    """Diagnostics-compatible extras from a θ-block covariance (cpu): the
    θ_scale block in the 24×3×24×3 layout, its √diag σ, and the delta-method
    effective σ for the raw θ_smear. Shared by the Fisher / empirical builders."""
    out: dict = {}
    if n_scale == 72:
        # θ_scale O(1) → physical (A,e,M) = θ·THETA_SCALE_REF; cov_phys = cov·ref⊗ref.
        ref72 = torch.tensor(list(THETA_SCALE_REF) * 24, dtype=cov_theta.dtype)
        cs = cov_theta[:72, :72] * ref72.unsqueeze(0) * ref72.unsqueeze(1)
        out["covariance_24_3_24_3"] = cs.view(24, 3, 24, 3)
        out["sigma_scale_24_3"] = torch.sqrt(
            torch.clamp(torch.diag(cs), min=0.0)).view(24, 3)
    if smear_cols:
        n_eta, n_comp = model.theta_smear.shape
        cv = cov_theta[n_scale:, n_scale:]
        sig_raw = torch.sqrt(torch.clamp(torch.diag(cv), min=0.0))
        smear_scale = (SMEAR_VAR_SCALE_A, SMEAR_VAR_SCALE_C)  # θ→physical (linear)
        sig_eff = torch.zeros(n_eta, n_comp)
        k = 0
        for b in range(n_eta):
            for c in smear_cols:
                sig_eff[b, c] = smear_scale[c] * sig_raw[k]
                k += 1
        out["sigma_smear_eff_24_2"] = sig_eff
    return out


def compute_empirical_fisher_joint(
    model: JpsiMassMixtureModel,
    loader: JpsiMassArrowLoader,
    device: str,
    *,
    mc_as_data: bool = False,
    n_iter: int = 2,
    chunk_events: int = 64,
    max_events: int = 0,
    progress: bool = True,
    vectorized: bool = True,
):
    """Joint (θ, φ) empirical Fisher ``J = Σ_i w_i s_i s_iᵀ`` over θ_scale + the
    active θ_smear + ALL background-MLP params, from per-event scores
    ``s_i = ∇_(θ,φ) log p_{θ,φ}(x_i)`` (the data branch). PSD by construction
    (sum of outer products), so its pseudo-inverse never yields a negative
    variance; the θ-block of ``pinv(J)`` is the nuisance-marginalised θ
    covariance — background included — and uses only FIRST derivatives (no MLP
    Hessian). Per-event scores are taken in chunks via ``is_grads_batched`` with
    a per-event-loop fallback. Returns ``(J [n_act, n_act] cpu, layout)`` with
    the active vector ordered ``[θ-active | mlp]``."""
    model.eval()
    for p in model.flow.parameters():
        p.requires_grad_(False)
    for p in model.mlp.parameters():
        p.requires_grad_(True)
    model.theta_scale.requires_grad_(model.scale_enabled)
    model.theta_smear.requires_grad_(model.smearing_enabled)

    params, names, numels = [], [], []
    if model.scale_enabled:
        params.append(model.theta_scale); names.append("scale")
        numels.append(model.theta_scale.numel())
    smear_cols = _smear_active_cols(model) if model.smearing_enabled else []
    if model.smearing_enabled:
        params.append(model.theta_smear); names.append("smear")
        numels.append(model.theta_smear.numel())
    mlp_params = list(model.mlp.parameters())
    n_mlp = int(sum(p.numel() for p in mlp_params))
    for p in mlp_params:
        params.append(p); names.append("mlp"); numels.append(p.numel())
    if not params or (not model.scale_enabled and not smear_cols):
        raise RuntimeError("empirical Fisher: no active θ parameters.")

    # Active flat indices into the concatenated score: θ_scale (all) + active
    # θ_smear cols + all MLP — θ first, then φ.
    active_idx, off, n_theta_active = [], 0, 0
    for nm, ne in zip(names, numels):
        if nm == "scale":
            active_idx += list(range(off, off + ne)); n_theta_active += ne
        elif nm == "smear":
            n_eta, n_comp = model.theta_smear.shape
            a = [off + b * n_comp + c for b in range(n_eta) for c in smear_cols]
            active_idx += a; n_theta_active += len(a)
        else:
            active_idx += list(range(off, off + ne))
        off += ne
    active_idx = torch.tensor(active_idx, dtype=torch.long, device=device)
    n_act = int(active_idx.numel())

    J = torch.zeros((n_act, n_act), device=device, dtype=torch.float32)
    sw = 0.0; seen = 0; hit_cap = False
    use_batched = bool(vectorized)
    bar = tqdm(loader, desc="emp-fisher", disable=not progress, unit="batch")
    for batch in bar:
        if max_events > 0 and seen >= max_events:
            hit_cap = True; break
        batch = _move_batch(batch, device)
        data_mask = ~batch["is_data_mask"] if mc_as_data else batch["is_data_mask"]
        di = data_mask.nonzero(as_tuple=True)[0]
        if di.numel() == 0:
            continue
        per = model.data_nll_continuity(
            batch["mll"], batch["pt_pm"], batch["eta_pm"], batch["phi_pm"],
            batch["q_pm"], batch["b_pm"], batch["muon_kin_std"], data_mask,
            n_iter=n_iter)
        per_d = per[di]
        w_d = batch["w"][di].detach()
        nd = int(per_d.shape[0])
        for s0 in range(0, nd, max(1, chunk_events)):
            s1 = min(s0 + chunk_events, nd)
            c = s1 - s0
            S = None
            if use_batched:
                try:
                    eye = torch.zeros((c, nd), device=device, dtype=per_d.dtype)
                    eye[torch.arange(c, device=device),
                        torch.arange(s0, s1, device=device)] = 1.0
                    g = torch.autograd.grad(
                        per_d, params, grad_outputs=eye, is_grads_batched=True,
                        retain_graph=True, allow_unused=True)
                    S = torch.cat([
                        (gi if gi is not None else torch.zeros((c,) + p.shape,
                         device=device, dtype=per_d.dtype)).reshape(c, -1)
                        for gi, p in zip(g, params)], dim=1)
                except (RuntimeError, NotImplementedError) as e:
                    use_batched = False
                    bar.write(f"  note: vectorised per-event score unavailable "
                              f"({type(e).__name__}); using the per-event loop")
            if S is None:
                rows = []
                for j in range(s0, s1):
                    gj = torch.autograd.grad(per_d[j], params, retain_graph=True,
                                             allow_unused=True)
                    rows.append(torch.cat([
                        (x if x is not None else torch.zeros_like(p)).reshape(-1)
                        for x, p in zip(gj, params)]))
                S = torch.stack(rows)
            S = S[:, active_idx]                       # [c, n_act]
            wc = w_d[s0:s1]
            J += (S * wc.unsqueeze(1)).t() @ S         # Σ w_i s_i s_iᵀ
        seen += nd; sw += float(w_d.sum())
        bar.set_postfix_str(f"events={seen:,}")
    bar.close()
    if seen == 0:
        raise RuntimeError("empirical Fisher: zero data-branch events seen.")
    J = 0.5 * (J + J.T)                                # symmetrise (numerical)
    layout = {"n_theta_active": n_theta_active,
              "n_scale": (model.theta_scale.numel() if model.scale_enabled else 0),
              "smear_cols": smear_cols, "n_mlp": n_mlp,
              "sw": sw, "seen": seen, "hit_cap": hit_cap}
    return J.detach().cpu(), layout


def _empirical_cov_theta_block(J: torch.Tensor, n_theta: int, ridge: float) -> torch.Tensor:
    """θ-block covariance from the joint empirical Fisher ``J`` (PSD).

    ``ridge == 0`` → Moore–Penrose ``pinv(J)``: unconstrained / degenerate
    directions land in the null space and get **zero** variance (the misleading
    σ≈0 on a near-degenerate parameter, e.g. the A/e degeneracy over the J/ψ
    pt range).

    ``ridge > 0`` → scale-aware ridge ``cov = (J + ridge·diag(J))⁻¹``, computed
    as a plain inverse in the per-parameter standardised space
    ``D⁻¹(D⁻¹JD⁻¹ + ridge·I)⁻¹D⁻¹`` with ``D = √diag(J)``. A flat direction then
    reads as a LARGE but finite variance (~1/(ridge·J_ii)) rather than 0, and
    the regularisation is proportional to each parameter's own information
    (dimensionless ``ridge``), so it is well-behaved across the mixed-unit
    A/e/M/smear/MLP blocks."""
    J = 0.5 * (J + J.T)
    if ridge <= 0.0:
        cov = torch.linalg.pinv(J)
    else:
        d = torch.sqrt(torch.clamp(torch.diag(J), min=0.0))
        dmax = float(d.max()) if d.numel() else 1.0
        dinv = 1.0 / torch.clamp(d, min=1e-12 * (dmax if dmax > 0 else 1.0))
        Jt = J * dinv.unsqueeze(0) * dinv.unsqueeze(1)            # D⁻¹ J D⁻¹ (PSD)
        n = Jt.shape[0]
        eye = torch.eye(n, dtype=J.dtype, device=J.device)
        cov = torch.linalg.inv(Jt + ridge * eye)                 # PD → plain inv
        cov = cov * dinv.unsqueeze(0) * dinv.unsqueeze(1)        # back to raw units
    return cov[:n_theta, :n_theta].contiguous()


def _run_empirical_fisher(args, model, shard_files, stats, device) -> None:
    """Joint (θ,φ) empirical Fisher → ``<output>/empirical_fisher.pt`` (PSD,
    background-included θ covariance via pinv; diagnostics-compatible)."""
    if not (model.scale_enabled or model.smearing_enabled):
        print("skipping empirical Fisher: both --disable-scale and --disable-smearing.")
        return
    if model.theta_mode == "mlp":
        print("skipping empirical Fisher: not implemented for --theta-mlp (binned "
              "24×3 θ layout).", file=sys.stderr)
        return
    half = 1 if args.validation else None
    inj = _inject_theta_np(args, len(stats.eta_edges) - 1) if args.validation else None
    inj_sm = _inject_smear_np(args, len(stats.eta_edges) - 1) if args.validation else None
    loader = JpsiMassArrowLoader(
        shard_files, stats, batch_size=args.batch_size, split=args.fisher_split,
        val_fraction=args.val_fraction, holdout_fraction=args.holdout_fraction,
        drop_last=False, half=half, inject_theta_scale=inj,
        inject_theta_smear=inj_sm, inject_seed=int(args.inject_smear_seed))
    print(f"\ncomputing joint (θ,φ) empirical Fisher (per-event scores → pinv) on "
          f"split={args.fisher_split}"
          + ("  half=1 (MC pseudo-data)" if args.validation else "")
          + f"  [smear_fit={model.smear_fit_params}]"
          + (f"; ≤{args.empirical_fisher_max_events:,} events"
             if args.empirical_fisher_max_events > 0 else ""))
    t0 = time.time()
    J, layout = compute_empirical_fisher_joint(
        model, loader, device, mc_as_data=args.validation,
        n_iter=args.continuity_n_iter,
        chunk_events=args.empirical_fisher_chunk,
        max_events=args.empirical_fisher_max_events,
        progress=args.progress, vectorized=args.fisher_vectorized)
    # If capped, scale J to full statistics (J ∝ Σw): a cheap weight-only pass
    # for Σw_total, then J ← J · Σw_total/Σw_seen so the covariance has the
    # correct 1/N_fit scale.
    if layout["hit_cap"] and layout["sw"] > 0:
        sw_total = 0.0
        for batch in loader:
            dm = (~batch["is_data_mask"] if args.validation else batch["is_data_mask"])
            sw_total += float((batch["w"] * dm.to(batch["w"].dtype)).sum())
        if sw_total > layout["sw"]:
            scale = sw_total / layout["sw"]
            J = J * scale
            print(f"  scaled J by Σw_total/Σw_seen = {scale:.2f} "
                  f"(subsampled {layout['seen']:,} events)")
            layout["sw"] = sw_total
    jd = layout["n_theta_active"] + layout["n_mlp"]
    rank = int(torch.linalg.matrix_rank(J).item())
    n_t = layout["n_theta_active"]
    ridge = float(args.empirical_fisher_ridge)
    cov_theta = _empirical_cov_theta_block(J, n_t, ridge)
    out = {
        "method": (f"joint (theta,phi) empirical Fisher, ridge={ridge:g}"
                   if ridge > 0 else "joint (theta,phi) empirical Fisher, pinv"),
        "ridge": ridge,
        "covariance": cov_theta,
        "labels": _active_param_labels(model, layout["smear_cols"]),
        "n_scale": layout["n_scale"], "smear_cols": layout["smear_cols"],
        "smear_fit_params": model.smear_fit_params,
        "n_events": layout["seen"], "sum_weight": layout["sw"],
        "n_theta_active": n_t, "n_mlp": layout["n_mlp"],
        "joint_dim": jd, "joint_rank": rank,
        "param_space": "scale: linear (A,e,M); smear: raw pre-softplus theta_smear",
    }
    out.update(_theta_cov_extras(cov_theta, model, layout["smear_cols"], layout["n_scale"]))
    path = os.path.join(args.output, "empirical_fisher.pt")
    torch.save(out, path)
    print(f"  wrote {path}: joint {jd}×{jd} (θ:{n_t}, φ:{layout['n_mlp']}), "
          f"rank {rank}/{jd}; θ-block {n_t}×{n_t} via "
          f"{'ridge=' + format(ridge, 'g') if ridge > 0 else 'pinv'} "
          f"({layout['seen']:,} events, Σw={layout['sw']:.2e}) in {time.time()-t0:.1f}s")
    if "sigma_scale_24_3" in out:
        ss = out["sigma_scale_24_3"]
        print(f"  empirical σ(A,e,M) median over bins = "
              f"({float(ss[:,0].median()):.2e}, {float(ss[:,1].median()):.2e}, "
              f"{float(ss[:,2].median()):.2e})")


def _load_full_fit(args, device):
    """Load a FULL stage-2 fit (flow + MLP + θ) from ``--checkpoint`` for
    ``--stage uncertainties``. Adopts the model-defining settings (flow arch,
    MLP size, smear-fit choice, scale/smear enables, validation) and the
    standardisation stats from the checkpoint so the rebuilt model matches the
    saved weights exactly. Returns ``(path, stats, model)`` or ``None``."""
    ck_path = args.checkpoint or os.path.join(args.output, "fit_best.pt")
    if not os.path.exists(ck_path):
        print(f"error: --stage uncertainties needs a fitted checkpoint; {ck_path!r} "
              f"not found (point --checkpoint at a fit_best.pt / fit_last.pt).",
              file=sys.stderr)
        return None
    print(f"loading fit checkpoint: {ck_path}")
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    ck_args = ck.get("args", {}) or {}
    if ck_args.get("stage") == "flow":
        print("  warning: --checkpoint is a stage-1 FLOW checkpoint (θ not fit); "
              "the uncertainty will be evaluated at the un-fit θ.", file=sys.stderr)
    # Adopt the model-defining settings from the checkpoint.
    _apply_flow_arch_from_ckpt(args, ck_args)
    for k in ("mlp_hidden", "mlp_n_layers", "smear_fit_params", "smear_flow_steps",
              "qop_floor_frac", "theta_mlp", "theta_mlp_hidden", "theta_mlp_layers",
              "disable_scale", "disable_smearing", "validation",
              "inject_A", "inject_e", "inject_M",
              "inject_a", "inject_c", "inject_smear_seed"):
        if k in ck_args:
            setattr(args, k, ck_args[k])
    # Stats: --stats-in overrides; else the fit's own stats.
    if args.stats_in is not None and os.path.exists(args.stats_in):
        with open(args.stats_in) as f:
            stats = _stats_from_dict(json.load(f))
        print(f"  preproc stats from {args.stats_in}")
    elif "stats" in ck:
        stats = _stats_from_dict(ck["stats"])
        print("  preproc stats from the checkpoint")
    else:
        print("error: no stats in checkpoint and no --stats-in given.", file=sys.stderr)
        return None
    model = _build_model(args, stats, device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return ck_path, stats, model


def _run_uncertainties_stage(args, device) -> int:
    """--stage uncertainties: load an existing fit and run only the Fisher info
    and/or the warm-start bootstrap on it (no training)."""
    loaded = _load_full_fit(args, device)
    if loaded is None:
        return 1
    ck_path, stats, model = loaded
    print(f"=== stage uncertainties: full fit loaded (flow + MLP + θ) — "
          f"scale={'on' if model.scale_enabled else 'off'}, "
          f"smear={'on' if model.smearing_enabled else 'off'}, "
          f"smear_fit={model.smear_fit_params}"
          + ("  [validation: MC pseudo-data]" if args.validation else "") + " ===")
    shard_files = discover_shards(args.inputs)
    if not shard_files:
        print("error: no Arrow shards found under inputs", file=sys.stderr)
        return 1
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "preproc_stats.json"), "w") as f:
        json.dump(_stats_to_dict(stats), f, indent=2)
    if not args.fisher_info and not args.empirical_fisher and args.bootstrap <= 0:
        print("warning: --stage uncertainties but none of --fisher-info / "
              "--empirical-fisher / --bootstrap (>0) requested — nothing to "
              "compute.", file=sys.stderr)
        return 0
    if args.fisher_info:
        _run_fisher_continuity(args, model, shard_files, stats, device)
    if args.empirical_fisher:
        _run_empirical_fisher(args, model, shard_files, stats, device)
    if args.bootstrap > 0:
        run_bootstrap_continuity(args, model, shard_files, stats, device,
                                 mc_as_data=args.validation)
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
    # Two-stage continuity pipeline.
    p.add_argument(
        "--stage", choices=["both", "flow", "fit", "uncertainties"],
        default="both",
        help="Two-stage continuity training: 'flow' = stage 1 (nominal flow on "
        "simulation, no θ conditioning); 'fit' = stage 2 (freeze flow, fit θ + "
        "background on data via the analytic continuity tilt); 'both' = run 1 "
        "then 2 in-process (default); 'uncertainties' = load an existing FULL fit "
        "from --checkpoint and run only the Fisher info (--fisher-info) and/or "
        "warm-start bootstrap (--bootstrap), no training.",
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint to load. '--stage fit': a stage-1 FLOW checkpoint "
        "(default <output>/flow_best.pt); only its flow weights are loaded (MLP + θ "
        "start fresh). '--stage uncertainties': a FULL fit checkpoint "
        "(default <output>/fit_best.pt; point it at fit_last.pt to use the latest) "
        "— flow + MLP + θ are all loaded. Either way the flow architecture and the "
        "preproc stats are read from the checkpoint automatically (so they need not "
        "be re-specified); pass --stats-in to override the stats.",
    )
    p.add_argument(
        "--validation", action="store_true",
        help="MC-closure validation mode: use simulation for BOTH stages "
        "instead of data for stage 2. A deterministic disjoint half of the "
        "simulation events trains the stage-1 flow (half 0); the other half "
        "(half 1) is treated as pseudo-data for the stage-2 θ fit. The disjoint "
        "split prevents stage 2 from fitting θ against the events stage 1 "
        "trained on; the closure target is θ → 0. Any real data in the inputs "
        "is unused.",
    )
    # Inject a known θ_scale shift into the validation pseudo-data (closure with
    # a non-zero target): the stage-2 pseudo-data m_ll is advected by this scale,
    # so the fit should recover it. A constant shift per component over all η
    # bins. Only active with --validation.
    p.add_argument("--inject-A", type=float, default=0.0,
                   help="(--validation) Inject this constant A scale shift into "
                   "the pseudo-data; the fit should recover it (closure target).")
    p.add_argument("--inject-e", type=float, default=0.0,
                   help="(--validation) Inject this constant e [GeV] scale shift.")
    p.add_argument("--inject-M", type=float, default=0.0,
                   help="(--validation) Inject this constant M scale shift.")
    p.add_argument("--inject-a", type=float, default=0.0,
                   help="(--validation) Inject this constant PHYSICAL qop-variance "
                   "coefficient 'a' (σ²_qop = a + c·k², the constant hit-resolution "
                   "term; physical scale ~1e-7) into the pseudo-data via the same "
                   "per-muon qop fold as the validation plots — a Gaussian "
                   "√(a+c·k²) kick, m_ll recomputed. Physical units throughout "
                   "(the O(1) optimizer rescaling is internal); the fit recovers "
                   "the injected value (shown on the θ_smear plot).")
    p.add_argument("--inject-c", type=float, default=0.0,
                   help="(--validation) Inject this constant PHYSICAL qop-variance "
                   "coefficient 'c' (the ∝k²=1/pt² multiple-scattering term; "
                   "physical scale ~1e-6; see --inject-a).")
    p.add_argument("--inject-smear-seed", type=int, default=12345,
                   help="Seed for the injected-smear Gaussian qop kick, so the "
                   "pseudo-data realisation is reproducible across epochs/runs.")
    p.add_argument("--flow-epochs", type=int, default=0,
                   help="Max epochs for stage 1 (0 → use --epochs).")
    p.add_argument("--fit-epochs", type=int, default=0,
                   help="Max epochs for stage 2 (0 → use --epochs).")
    p.add_argument("--fit-scale-lr", type=float, default=1e-3,
                   help="Stage-2 Adam lr for θ_scale. θ_scale is O(1) (physical "
                   "A,e,M = θ·THETA_SCALE_REF=(1e-4,1e-3,1e-5)), so all three "
                   "components share a well-conditioned step at this O(1) lr.")
    p.add_argument("--fit-smear-lr", type=float, default=1e-3,
                   help="Stage-2 Adam lr for θ_smear (O(1); σ²_qop = θ·SMEAR_VAR_SCALE).")
    p.add_argument("--fit-mlp-lr", type=float, default=1e-3,
                   help="Stage-2 Adam lr for the background-fraction MLP.")
    p.add_argument("--fit-theta-mlp-lr", type=float, default=1e-3,
                   help="Stage-2 Adam lr for the θ ThetaNet (--theta-mlp). One lr "
                   "for all of (A,e,M,a,c); the net's output reference scaling "
                   "sets the relative A,e,M vs a,c magnitudes.")
    p.add_argument("--continuity-n-iter", type=int, default=2,
                   help="Fixed-point iterations for the #2 source solve "
                   "(advection+smear pre-image).")
    p.add_argument("--epochs", type=int, default=50, help="Maximum training epochs.")
    p.add_argument("--batch-size", type=int, default=65536, help="Events per batch.")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Adam lr for flow + MLP.")
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
                   help="LR schedule (both stages). 'plateau': reduce "
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
        "only), or 'c' (∝1/pt² term only). The constant a and the c·k² term are "
        "nearly degenerate over the narrow J/ψ pt range, so fitting both per "
        "bin is ill-posed and yields the unphysical bin-to-bin zig-zag (use 'a' "
        "or 'c' to break it). The non-fitted term is zeroed — removed from the "
        "σ_qop variance entirely, not floated.",
    )
    p.add_argument(
        "--smear-flow-steps", type=int, default=1,
        help="Euler steps integrating the smear's probability-flow ODE in the "
        "density (the score-driven deterministic-diffusion change of variables). "
        "1 = first-order (single score displacement); more steps integrate the "
        "score flow more finely (more robust/accurate) at a higher nested-"
        "autograd cost. The per-muon qop fold (closure plots) and the injection "
        "are exact convolutions regardless.",
    )
    p.add_argument(
        "--theta-mlp", action="store_true",
        help="Replace the per-η-bin (A,e,M,a,c) tables with a small MLP mapping "
        "each muon's (η, φ) → (A,e,M,a,c) CONTINUOUSLY (trained in stage 2 like "
        "the background MLP). Note: the observed/empirical Fisher and bootstrap "
        "uncertainties are binned-θ-only and are skipped in this mode.",
    )
    p.add_argument("--theta-mlp-hidden", type=int, default=32,
                   help="(--theta-mlp) Hidden width of the θ ThetaNet.")
    p.add_argument("--theta-mlp-layers", type=int, default=2,
                   help="(--theta-mlp) Number of hidden layers of the θ ThetaNet.")
    # θ_scale sampling widths, split per component (A, e, M) since they live
    # in different physical units. Each is the σ of the Gaussian added to that
    # component (additive, physical units) — the fixed width with
    # --fixed-theta-sampling / --no-adaptive-sigma, and the adaptive-σ fallback
    # during warmup.
    # Flow / MLP hyperparams
    p.add_argument(
        "--flow-arch", choices=("gf", "nsf"), default="gf",
        help="Signal flow architecture: 'gf' = Gaussianization flow (default) — "
        "C∞-smooth density, so the continuity score/Hessian have no knot kinks. "
        "'nsf' = neural rational-quadratic spline flow — bounded (linear tails "
        "outside ±5), avoids the erf/exp saturation the GF needs guards for, but "
        "only C¹ (the score kinks at the spline knots).",
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
    p.add_argument("--fisher-vectorized", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="(two-stage) Compute the Hessian with one vmapped "
                   "(is_grads_batched) second backward instead of a per-row loop. "
                   "Faster but holds ~n_param copies of the backward graph; "
                   "automatically falls back to the loop on engine failure / OOM. "
                   "Use --no-fisher-vectorized to force the loop. Also governs the "
                   "per-event score in --empirical-fisher.")
    # Joint (theta, phi) empirical Fisher: J = Σ w_i s_i s_iᵀ over θ + the MLP,
    # PSD → pinv covariance (background-included, Hessian-free, no negative σ).
    p.add_argument("--empirical-fisher", action="store_true",
                   help="(two-stage) After the fit, compute the JOINT (θ, MLP) "
                   "empirical Fisher J = Σ w_i s_i s_iᵀ from per-event scores and "
                   "write the θ-block of pinv(J) → <output>/empirical_fisher.pt. "
                   "PSD by construction (no negative/clamped σ), background "
                   "uncertainty included (the MLP is in the joint), and Hessian-"
                   "free (first derivatives only). Cost is O(N_events) per-event "
                   "gradients — see --empirical-fisher-max-events.")
    p.add_argument("--empirical-fisher-max-events", type=int, default=0,
                   help="Cap on events used for the empirical Fisher (default 0 = "
                   "all events). If set >0, J (an average per-event quantity) is "
                   "computed on that representative subset and rescaled by "
                   "Σw_total/Σw_seen to the full-statistics covariance scale.")
    p.add_argument("--empirical-fisher-chunk", type=int, default=64,
                   help="Events per is_grads_batched call when extracting "
                   "per-event scores (memory ≈ chunk × backward graph).")
    p.add_argument("--empirical-fisher-ridge", type=float, default=0.0,
                   help="Ridge for the empirical-Fisher covariance. 0 (default) → "
                   "Moore–Penrose pinv: unconstrained / degenerate directions get "
                   "σ=0 (e.g. the A/e near-degeneracy over the J/ψ pt range reads "
                   "as σ(e)≈0). >0 → scale-aware ridge cov=(J + ridge·diag(J))⁻¹, "
                   "so a degenerate direction reads as a LARGE finite σ instead. "
                   "Dimensionless (relative to each parameter's own information); "
                   "try ~1e-3.")
    # Warm-start Poisson bootstrap (Hessian-free covariance incl. the background)
    p.add_argument("--bootstrap", type=int, default=0,
                   help="(two-stage) After the stage-2 fit, run this many warm-start "
                   "Poisson-bootstrap replicas → <output>/bootstrap_cov.pt. Each "
                   "replica restarts from the nominal (θ̂, φ̂), re-fits θ + the "
                   "background MLP jointly on the data Poisson(1)-reweighted, and the "
                   "covariance of {θ̂_b} folds in the background uncertainty with no "
                   "Hessian. 0 = off.")
    p.add_argument("--bootstrap-epochs", type=int, default=None,
                   help="Max epochs per bootstrap replica. Default (None) inherits "
                   "the nominal stage-2 cap (--fit-epochs or --epochs). Replicas "
                   "hitting the cap without plateauing are flagged as possibly "
                   "under-converged.")
    p.add_argument("--bootstrap-patience", type=int, default=None,
                   help="Per-replica early-stop patience (epochs without "
                   "reweighted-NLL improvement > --patience-threshold). Default "
                   "(None) inherits the nominal fit's --patience, so each replica "
                   "converges with the SAME early-stop / threshold / LR-schedule "
                   "(--lr-schedule + --lr-reduce-*) as the nominal fit — monitored "
                   "on the replica's reweighted NLL (a replica has no separate val).")
    p.add_argument("--bootstrap-seed", type=int, default=12345,
                   help="Base seed for the per-replica Poisson(1) event weights "
                   "(replica b uses seed + b; reset each epoch so an event keeps its "
                   "count across the replica's epochs).")
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
    # Adaptive-σ clamps are split scale/smear because the two parameters live
    # in different units: θ_scale is the linear (A,e,M) ~1e-4…1e-2, while
    # θ_smear is the *raw* (pre-softplus) param whose natural sampling scale is
    # O(1). The floor prevents σ→0 collapse near sharp optima; the ceiling
    # prevents σ→∞ on parameters that haven't received gradient yet (v_i = 0).
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
