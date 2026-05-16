"""Diagnostic plots for a shift+smear reweight model.

Loads a checkpoint produced by ``train_shift_smear_reweight.py`` (either
``--arch mlp`` or ``--arch polyhead``) and produces:

  * **Shift closure** — per target component (r_κ, dλ, dφ) and shift
    magnitude, histograms of (raw MC, shifted MC, MLP-pred-W reweight)
    with ratios divided by the explicitly-shifted MC. The shifted-MC
    curve is the literal closure target.

  * **Smear closure** — per target component and smear magnitude,
    histograms of (raw MC, smeared MC, MLP-pred-W reweight). The
    smeared-MC curve has each event drawn from
    ``y_smeared = y + ε · σ · e_tcol`` with one ε ~ N(0, 1) per event,
    matching the rank-1 Gaussian smear our model is trained on.

  * **log r distribution** — across events, separately for shift and
    smear axes, one curve per magnitude. Shows the spread of per-event
    reweighting factors.

  * **log r magnitude scan** — weighted mean and ±1σ band of predicted
    log r vs. perturbation magnitude, separately for shift and smear,
    per axis. For small perturbations should grow linearly (score) for
    shifts and quadratically (Hessian) for smears.

The structural priors ``r(y, c, 0, 0) = 1`` (by construction) and
``r`` even in σ_vec (by Σ-input symmetry / polynomial parity) are not
plotted — they hold structurally and don't need empirical verification.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import List

import matplotlib
matplotlib.use("Agg")  # noqa: E402  — non-interactive backend; must precede pyplot
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from train_muon_response_flow import (  # noqa: E402
    PreprocStats,
    apply_preproc,
    compute_targets_and_conditioning,
    evaluate_joint,
    load_ntuples,
)
from train_shift_smear_reweight import (  # noqa: E402
    _ACTIVATIONS,
    _LOG_LOG2,
    _LOG_W_CLAMP,
    _sigma_pack_indices,
    GaussBaseline,
    ReweightMLP_B,
    ReweightMLPFactored,
    ReweightPolyhead,
    gauss_baseline_log_r,
)


TARGET_NAMES = ["r_kappa", "dlambda", "dphi"]


# Default mapping from snapshot ``source_id`` integers to display labels.
# The snapshot script assigns:
#   J/ψ:  base = 0  (Pt8toInf), base+1 = 1 (Pt0to8)
#   W/Z:  base = 100, then incremented by getDatasets() ordering.
# Z→μμ is conventionally the first W/Z entry, hence source_id = 100.
# Anything else falls back to ``source <id>``; the user can override or
# extend via ``--cmp-source-labels``.
DEFAULT_SOURCE_LABELS = {
    0: r"J/$\psi$ (p$_T$>8 GeV)",
    1: r"J/$\psi$ (p$_T$<8 GeV)",
    100: r"Z$\to\mu\mu$",
}
ZMUMU_SOURCE_ID = 100


def _tic():
    return time.perf_counter()


def _toc(label, t0):
    dt = time.perf_counter() - t0
    print(f"  [time] {label}: {dt:.2f}s")
    return time.perf_counter()


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Diagnostic plots for a shift+smear reweight "
        "model (mlp / polyhead).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", required=True, default=argparse.SUPPRESS,
        help="(required) Path to a shift_smear_reweight_{mlp,polyhead}.pt "
        "or checkpoint.pt produced by train_shift_smear_reweight.py.",
    )
    p.add_argument(
        "--input-files", nargs="+", required=True,
        default=argparse.SUPPRESS,
        help="(required) Input ROOT file(s) with the J/psi snapshot tree.",
    )
    p.add_argument(
        "--tree", default="tree",
        help="ROOT TTree name to read from --input-files.",
    )
    p.add_argument(
        "--output", default=None,
        help="Output dir for plots. None = directory containing "
        "--checkpoint.",
    )
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for model evaluation.",
    )
    p.add_argument(
        "--n-events", type=int, default=500_000,
        help="Number of muon rows to use for plotting. Subsampled "
        "post-quality-cut. -1 = use all surviving rows.",
    )
    p.add_argument(
        "--max-events", type=int, default=1_000_000,
        help="Cap on raw J/psi events kept via RDataFrame Filter("
        "rdfentry_<N). 1M events comfortably covers --n-events at "
        "~2 muons/event. -1 = load all.",
    )
    p.add_argument(
        "--max-muons", type=int, default=-1,
        help="Cap on muon rows after the event filter. -1 = no cap.",
    )
    p.add_argument(
        "--threads", type=int, default=0,
        help="RDataFrame ImplicitMT threads. 0 = ROOT auto.",
    )
    p.add_argument(
        "--pt-min", type=float, default=2.0,
        help="Min gen pt (GeV); matches snapshot script default.",
    )
    p.add_argument(
        "--pt-max", type=float, default=200.0,
        help="Max gen pt (GeV).",
    )
    p.add_argument(
        "--eta-max", type=float, default=2.4,
        help="Max |gen eta|.",
    )
    p.add_argument(
        "--shift-factors", nargs="+", type=float,
        default=[0.1, 0.3, 0.5, 1.0],
        help="Shift magnitudes (in standardized-target σ units).",
    )
    p.add_argument(
        "--smear-factors", nargs="+", type=float,
        default=[0.1, 0.3, 0.5, 1.0],
        help="Smear magnitudes (in standardized-target σ units).",
    )
    p.add_argument(
        "--smear-gh-K", type=int, default=0,
        help="Gauss-Hermite order K for an extra smear-closure curve "
        "computed from the *shift* component of the polyhead via "
        "r_smear(σ·ê) ≈ Σ_k w_k · r_shift(ε_k·σ·ê) with probabilists' "
        "K-point Hermite nodes/weights for ε ~ N(0, 1). Tests "
        "self-consistency between the pure-σ and pure-u parts of the "
        "polyhead. 0 disables the curve (default); 3 is a reasonable "
        "starting value.",
    )
    p.add_argument(
        "--n-bins", type=int, default=80,
        help="Histogram bin count for closure plots.",
    )
    p.add_argument(
        "--range-percentile", type=float, default=0.5,
        help="Percentile (each side) trimmed when setting histogram "
        "x-range from raw target distribution.",
    )
    p.add_argument(
        "--scan-magnitudes", type=int, default=11,
        help="Number of magnitudes per axis in the log-r scan plot.",
    )
    p.add_argument(
        "--batch-size", type=int, default=8192,
        help="Batch size for model evaluation.",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Seed for the per-event ε draw used to form the "
        "smeared-MC histogram (literal closure target).",
    )
    p.add_argument(
        "--flow-checkpoint", default=None,
        help="Optional path to a flow checkpoint produced by "
        "train_muon_response_flow.py. When provided, an additional "
        "diagnostic plot ``polyhead_pred_vs_flow.{png,pdf}`` is "
        "generated comparing this BCE-polyhead's predicted log W to "
        "the flow's analytic log W on a common (y, c, u, σ) grid — "
        "the analog of flow_training_diagnostics.py's "
        "``polyhead_pred_vs_true`` for direct-trained heads. "
        "Skipped if not provided. Assumes the flow's preproc is "
        "compatible with this checkpoint's preproc (i.e. trained on "
        "the same MC sample).",
    )
    p.add_argument(
        "--flow-validate-n", type=int, default=10000,
        help="Number of events to use for the optional "
        "polyhead_pred_vs_flow plot. Smaller = faster.",
    )
    # ------------------------------------------------------------------
    # Per-source target-distribution comparison window.
    # Datasets with different kinematics can't be compared at the
    # marginal level, so restrict the comparison to a phase-space cell
    # in (pt_gen, eta_gen, charge_gen); phi_gen is integrated.
    # ------------------------------------------------------------------
    p.add_argument(
        "--cmp-pt-min", type=float, default=25.0,
        help="Lower gen-pT (GeV) edge of the (deliberately narrow) "
        "phase-space window used for the per-source target-distribution "
        "comparison plot.",
    )
    p.add_argument(
        "--cmp-pt-max", type=float, default=30.0,
        help="Upper gen-pT (GeV) edge of the cmp window.",
    )
    p.add_argument(
        "--cmp-eta-min", type=float, default=0.0,
        help="Lower gen-η edge of the cmp window.",
    )
    p.add_argument(
        "--cmp-eta-max", type=float, default=0.4,
        help="Upper gen-η edge of the cmp window.",
    )
    p.add_argument(
        "--cmp-charge", choices=["pos", "neg", "both", "each"],
        default="each",
        help="Charge selection in the cmp window. ``each`` (default) "
        "produces a separate plot for q>0 and q<0; ``both`` integrates "
        "over charge; ``pos`` / ``neg`` produce one plot for the chosen "
        "charge only.",
    )
    p.add_argument(
        "--cmp-source-labels", nargs="+", default=None,
        help="Optional per-source labels of the form ``id:label`` "
        "(e.g. ``0:J/psi 100:Zmumu``). Sources without a mapping use "
        "their integer id.",
    )
    p.add_argument(
        "--cmp-ref-source", type=int, default=ZMUMU_SOURCE_ID,
        help="Reference source_id for the ratio panel of the per-source "
        "comparison plot. Defaults to the Z→μμ convention (100); falls "
        "back to the smallest source_id present if absent in the window.",
    )
    p.add_argument(
        "--cmp-max-events", type=int, default=-1,
        help="Cap on raw muon rows *read* for the per-source target-"
        "distribution plots only. -1 (default) = all available rows. "
        "These plots stream shards in parallel and apply the (pt, eta) "
        "window per-shard, so peak memory is bounded by the kept-row "
        "count, not the full dataset — typically far smaller than the "
        "inference-side caps (--n-events / --max-events).",
    )
    p.add_argument(
        "--cmp-workers", type=int, default=8,
        help="Parallel shard readers (thread pool) for the per-source "
        "comparison loader.",
    )
    return p.parse_args()


# ============================================================================
# Checkpoint loader — dispatches on arch
# ============================================================================

def load_model_from_checkpoint(checkpoint_path: str, device):
    """Load model + stats + train_config. Builds ``ReweightMLP_B`` or
    ``ReweightPolyhead`` based on ``model_config['arch']``."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    train_cfg = ckpt.get("train_config", {})
    activation_name = str(cfg.get("activation", "gelu")).lower()
    activation_cls = _ACTIVATIONS.get(activation_name, torch.nn.GELU)
    arch = cfg.get("arch", "mlp")

    # Optional analytic Gaussian baseline. Older checkpoints predate
    # this and don't carry the ``gauss_baseline`` key — treat None /
    # missing as "no baseline".
    gb_cfg = cfg.get("gauss_baseline", None)
    gauss_baseline = (
        GaussBaseline(
            n_features=int(cfg["n_features"]),
            n_cond=int(cfg["n_cond"]),
            hidden=int(gb_cfg.get("hidden", 64)),
            n_layers=int(gb_cfg.get("n_layers", 2)),
            activation=activation_cls,
        )
        if gb_cfg is not None else None
    )

    if arch == "mlp":
        # ``shift_only`` was introduced with the --include-smear flag
        # (default False = shift-only). Older mlp checkpoints that
        # predate it do not carry the key; default to False so the
        # full-Σ_pack head is rebuilt for them.
        shift_only = bool(cfg.get("shift_only", False))
        model = ReweightMLP_B(
            n_features=int(cfg["n_features"]),
            n_cond=int(cfg["n_cond"]),
            d_emb=int(cfg.get("d_emb", 32)),
            trunk_hidden=int(cfg.get("trunk_hidden", 64)),
            trunk_layers=int(cfg.get("trunk_layers", 2)),
            head_hidden=int(cfg.get("head_hidden", 32)),
            head_layers=int(cfg.get("head_layers", 2)),
            activation=activation_cls,
            shift_only=shift_only,
            gauss_baseline=gauss_baseline,
        )
    elif arch == "mlp-factored":
        # Structurally factored MLP head. ``detach_pure_{shift,smear}_in_joint``
        # affect the factorisation form (whether A depends on σ_pack,
        # whether B depends on u, whether a separate cross head C
        # exists), so they must be threaded through at construction so
        # the state_dict shapes match.
        model = ReweightMLPFactored(
            n_features=int(cfg["n_features"]),
            n_cond=int(cfg["n_cond"]),
            d_emb=int(cfg.get("d_emb", 32)),
            trunk_hidden=int(cfg.get("trunk_hidden", 64)),
            trunk_layers=int(cfg.get("trunk_layers", 2)),
            head_hidden=int(cfg.get("head_hidden", 32)),
            head_layers=int(cfg.get("head_layers", 2)),
            activation=activation_cls,
            shift_only=bool(cfg.get("shift_only", False)),
            detach_pure_shift_in_joint=bool(
                cfg.get("detach_pure_shift_in_joint", False)
            ),
            detach_pure_smear_in_joint=bool(
                cfg.get("detach_pure_smear_in_joint", False)
            ),
            gauss_baseline=gauss_baseline,
        )
    elif arch == "polyhead":
        # Older checkpoints stored these as ``hidden_features`` /
        # ``n_hidden_layers``; new ones use ``trunk_hidden`` /
        # ``trunk_layers`` (shared with the mlp arch).
        trunk_hidden = int(cfg.get(
            "trunk_hidden", cfg.get("hidden_features", 64),
        ))
        trunk_layers = int(cfg.get(
            "trunk_layers", cfg.get("n_hidden_layers", 2),
        ))
        model = ReweightPolyhead(
            n_features=int(cfg["n_features"]),
            n_cond=int(cfg["n_cond"]),
            trunk_hidden=trunk_hidden,
            trunk_layers=trunk_layers,
            max_deg_u=int(cfg.get("max_deg_u", 3)),
            max_deg_sigma=int(cfg.get("max_deg_sigma", 4)),
            max_cross_deg=int(cfg.get("max_cross_deg", 3)),
            activation=activation_cls,
            basis=str(cfg.get("basis", "monomial")),
            basis_scale_u=float(cfg.get("basis_scale_u", 1.0)),
            basis_scale_sigma=float(cfg.get("basis_scale_sigma", 1.0)),
            gauss_baseline=gauss_baseline,
        )
    else:
        raise ValueError(f"unknown arch in checkpoint: {arch!r}")

    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    stats_dict = ckpt.get("stats")
    stats = PreprocStats(**stats_dict) if stats_dict is not None else None
    return model, arch, stats, train_cfg


# ============================================================================
# Unified per-axis log r prediction
# ============================================================================

def _apply_positivity(d, positivity):
    if positivity == "exp":
        return d.clamp(min=-_LOG_W_CLAMP, max=_LOG_W_CLAMP)
    if positivity == "softplus":
        # Mirror the trainer: no input clamp; output clamp_min sized
        # to the dtype's smallest normal so the same wrap is safe
        # under fp32 / bf16 / fp16.
        tiny = torch.finfo(d.dtype).tiny
        return torch.log(F.softplus(d).clamp_min(tiny)) - _LOG_LOG2
    if positivity == "asinh":
        return torch.asinh(d)
    raise ValueError(f"unknown positivity {positivity!r}")


@torch.no_grad()
def predict_log_r_axis(
    model, arch, y_dev, c_dev, u_axis_value, sigma_axis_value, tcol,
    n_features, batch_size, positivity,
    sigma_pack_iu=None, sigma_pack_ju=None,
):
    """Per-event log r prediction with an axis-aligned perturbation.

    ``u = u_axis_value · e_tcol``, ``σ_vec = sigma_axis_value · e_tcol``.

    Both ``y_dev`` and ``c_dev`` should already be on the target device.
    Reuses small per-batch buffers for ``u``, ``σ_vec``, and (for the
    MLP arch) ``Σ_pack`` so there's no per-call CPU-side allocation
    proportional to the dataset size.
    """
    device = y_dev.device
    N = y_dev.shape[0]
    out = np.empty(N, dtype=np.float32)

    # Pre-allocate per-batch buffers on device.
    u_buf = torch.zeros(batch_size, n_features, device=device)
    sigma_vec_buf = torch.zeros(batch_size, n_features, device=device)
    u_buf[:, tcol] = float(u_axis_value)
    sigma_vec_buf[:, tcol] = float(sigma_axis_value)

    if arch in ("mlp", "mlp-factored"):
        # Σ_pack for u-axis-aligned σ is zero except for the (tcol,tcol)
        # entry = sigma_axis_value². Compute via the standard helper so
        # the indexing matches the model's expectations.
        n_sigma_pack = sigma_pack_iu.shape[0]
        sigma_pack_buf = torch.zeros(
            batch_size, n_sigma_pack, device=device,
        )
        for k in range(n_sigma_pack):
            i, j = int(sigma_pack_iu[k]), int(sigma_pack_ju[k])
            if i == tcol and j == tcol:
                sigma_pack_buf[:, k] = float(sigma_axis_value) ** 2

    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        bsz = e - s
        y = y_dev[s:e]
        c = c_dev[s:e]
        u = u_buf[:bsz]
        sigma_vec = sigma_vec_buf[:bsz]

        if arch in ("mlp", "mlp-factored"):
            sigma_pack = sigma_pack_buf[:bsz]
            u_zero = torch.zeros_like(u)
            sp_zero = torch.zeros_like(sigma_pack)
            # 2B head batching: f at (e_y, u, σ) and (e_y, 0, 0).
            ev = model.trunk_forward(y, c)             # [bsz, d_emb]
            ev2 = torch.cat([ev, ev], dim=0)
            u2 = torch.cat([u, u_zero], dim=0)
            sp2 = torch.cat([sigma_pack, sp_zero], dim=0)
            f = model.head_forward(ev2, u2, sp2)
            d = f[:bsz] - f[bsz:]
        elif arch == "polyhead":
            coefs = model(y, c)                         # [bsz, n_basis]
            d = evaluate_joint(
                coefs, u, sigma_vec, model.joint_indices,
                basis=getattr(model, "basis", "monomial"),
                scale_u=float(getattr(model, "basis_scale_u", 1.0)),
                scale_sigma=float(getattr(model, "basis_scale_sigma", 1.0)),
            )
        else:
            raise ValueError(f"unknown arch {arch!r}")

        # Add the analytic Gaussian baseline (additive log-r term)
        # if the model carries one. Mirror the trainer's
        # compute_d_quadrature so closure plots match training-time
        # log r̂ exactly.
        gb = getattr(model, "gauss_baseline", None)
        if gb is not None:
            mu_g, L_g = gb(c)
            d = d + gauss_baseline_log_r(y, mu_g, L_g, u, sigma_vec)

        log_r = _apply_positivity(d, positivity)
        out[s:e] = log_r.detach().cpu().numpy().astype(np.float32)
    return out


def predict_log_r_perevent(
    model, arch, y_dev, c_dev, u_dev, sigma_dev,
    batch_size, positivity,
    sigma_pack_iu=None, sigma_pack_ju=None,
):
    """Per-event log r prediction with **arbitrary** per-event
    ``(u, σ_vec)`` (not axis-aligned). All four input tensors are on
    the model's device with shape ``[N, *]``. Used by the optional
    flow-comparison diagnostic.
    """
    device = y_dev.device
    N = y_dev.shape[0]
    n_features = y_dev.shape[1]
    out = np.empty(N, dtype=np.float32)

    if arch in ("mlp", "mlp-factored"):
        # Build Σ_pack from per-event σ_vec via the standard outer-
        # product packing. Done per-batch to bound memory.
        n_sigma_pack = sigma_pack_iu.shape[0]

    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        bsz = e - s
        y = y_dev[s:e]
        c = c_dev[s:e]
        u = u_dev[s:e]
        sigma_vec = sigma_dev[s:e]

        if arch in ("mlp", "mlp-factored"):
            sigma_pack = torch.zeros(
                bsz, n_sigma_pack, device=device, dtype=y.dtype,
            )
            for k in range(n_sigma_pack):
                i, j = int(sigma_pack_iu[k]), int(sigma_pack_ju[k])
                sigma_pack[:, k] = sigma_vec[:, i] * sigma_vec[:, j]
            u_zero = torch.zeros_like(u)
            sp_zero = torch.zeros_like(sigma_pack)
            ev = model.trunk_forward(y, c)
            ev2 = torch.cat([ev, ev], dim=0)
            u2 = torch.cat([u, u_zero], dim=0)
            sp2 = torch.cat([sigma_pack, sp_zero], dim=0)
            f = model.head_forward(ev2, u2, sp2)
            d = f[:bsz] - f[bsz:]
        elif arch == "polyhead":
            coefs = model(y, c)
            d = evaluate_joint(
                coefs, u, sigma_vec, model.joint_indices,
                basis=getattr(model, "basis", "monomial"),
                scale_u=float(getattr(model, "basis_scale_u", 1.0)),
                scale_sigma=float(getattr(model, "basis_scale_sigma", 1.0)),
            )
        else:
            raise ValueError(f"unknown arch {arch!r}")

        # Add the analytic Gaussian baseline (additive log-r term)
        # if the model carries one.
        gb = getattr(model, "gauss_baseline", None)
        if gb is not None:
            mu_g, L_g = gb(c)
            d = d + gauss_baseline_log_r(y, mu_g, L_g, u, sigma_vec)

        log_r = _apply_positivity(d, positivity)
        out[s:e] = log_r.detach().cpu().numpy().astype(np.float32)
    return out


def _gh_nodes_weights(K):
    """Probabilists' Gauss-Hermite nodes/weights for ε ~ N(0, 1).

    Returns ``(eps_k, w_k_norm)`` with ``Σ_k w_k_norm · f(eps_k) ≈
    E_{ε~N(0,1)}[f(ε)]``. The 1/√(2π) Gaussian density is folded
    into the weights.
    """
    eps_k, w_k = np.polynomial.hermite_e.hermegauss(int(K))
    w_norm = w_k / np.sqrt(2.0 * np.pi)
    return eps_k.astype(np.float64), w_norm.astype(np.float64)


def predict_smear_via_gh_shift(
    model, arch, y_dev, c_dev, factor, tcol, K,
    n_features, batch_size, positivity,
    sigma_pack_iu=None, sigma_pack_ju=None,
):
    """Estimate per-event log r for an axis-aligned smear from the
    *shift* component of the polyhead via Gauss-Hermite integration:

        r_smear(y, c, factor·ê_tcol)
            ≈ E_{ε~N(0,1)}[ r_shift(y, c, ε·factor·ê_tcol) ]
            ≈ Σ_k (w_k / √(2π)) · r_shift(y, c, ε_k·factor·ê_tcol)

    with K probabilists' Hermite nodes ε_k and weights w_k. Each
    GH-node call evaluates the polyhead with σ_vec = 0 (so only the
    pure-u / shift basis is exercised), giving a self-consistency
    check between the model's shift and smear branches. Returns
    ``log r_smear_gh`` of shape ``[N]``, computed with log-sum-exp.
    """
    eps_k, w_norm = _gh_nodes_weights(K)
    log_w = np.log(np.maximum(w_norm, 1e-300))

    N = y_dev.shape[0]
    stack = np.empty((int(K), N), dtype=np.float64)
    for k in range(int(K)):
        u_val = float(eps_k[k]) * float(factor)
        log_r_k = predict_log_r_axis(
            model, arch, y_dev, c_dev,
            u_axis_value=u_val, sigma_axis_value=0.0, tcol=tcol,
            n_features=n_features, batch_size=batch_size,
            positivity=positivity,
            sigma_pack_iu=sigma_pack_iu, sigma_pack_ju=sigma_pack_ju,
        )
        stack[k] = log_r_k.astype(np.float64) + float(log_w[k])

    M = stack.max(axis=0)
    M_safe = np.where(np.isfinite(M), M, 0.0)
    log_sum = M_safe + np.log(np.exp(stack - M_safe).sum(axis=0))
    return log_sum.astype(np.float32)


# ============================================================================
# Plot helpers
# ============================================================================

def _save(fig, path):
    fig.savefig(path, dpi=120, bbox_inches="tight")
    pdf = path.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def _weighted_hist_err(values, bins, weights):
    """Weighted histogram with per-bin sqrt(sum w^2) statistical
    uncertainty. ``weights`` is the per-event weight array; the bin
    content is ``Σ w_i`` and the bin error is ``√Σ w_i²``.

    Returns ``(h, sigma)``, both shape ``(n_bins,)``.
    """
    w = np.asarray(weights, dtype=np.float64)
    h, _ = np.histogram(values, bins=bins, weights=w)
    h2, _ = np.histogram(values, bins=bins, weights=w * w)
    sigma = np.sqrt(np.maximum(h2, 0.0))
    return h, sigma


def _ratio_with_err(h_num, sigma_num, h_den, sigma_den):
    """Bin-wise ratio h_num/h_den with propagated uncertainty assuming
    the two histograms are *uncorrelated* per bin. In the closure
    plots this holds because ``h_pred`` and ``h_shifted`` (or
    ``h_smeared``) are filled in different bins for any given event:
    h_pred at ``y_orig`` with weight ``w·r_pred``, h_shifted at
    ``y_orig + dy``. So while the two histograms come from the same
    event sample, per-bin counts are statistically independent (modulo
    bin-edge effects, which we neglect).

    Returns ``(ratio, sigma_ratio)`` with NaN where ``h_den ≤ 0``.
    """
    h_num = np.asarray(h_num, dtype=np.float64)
    h_den = np.asarray(h_den, dtype=np.float64)
    den_safe = np.where(h_den > 0, h_den, np.nan)
    num_safe = np.where(h_num > 0, h_num, np.nan)
    ratio = h_num / den_safe
    rel = np.sqrt(
        (sigma_num / num_safe) ** 2 + (sigma_den / den_safe) ** 2
    )
    return ratio, ratio * rel


def _stepped_errorbar(ax, centers, h, sigma, color, **kwargs):
    """Draw vertical sqrt(Σw²) error bars at each bin center, matched
    visually to a step histogram. ``kwargs`` are forwarded to errorbar
    (e.g. label, lw)."""
    ax.errorbar(
        centers, h, yerr=sigma, fmt="none", ecolor=color,
        elinewidth=0.7, capsize=0, alpha=0.6, **kwargs,
    )


def _common_predict_call(
    model, arch, y_dev, c_dev, factor, tcol, mode, n_features,
    batch_size, positivity, sigma_pack_iu, sigma_pack_ju,
):
    """Thin wrapper that maps mode={shift,smear} → axis values."""
    if mode == "shift":
        return predict_log_r_axis(
            model, arch, y_dev, c_dev,
            u_axis_value=float(factor), sigma_axis_value=0.0, tcol=tcol,
            n_features=n_features, batch_size=batch_size,
            positivity=positivity,
            sigma_pack_iu=sigma_pack_iu, sigma_pack_ju=sigma_pack_ju,
        )
    if mode == "smear":
        return predict_log_r_axis(
            model, arch, y_dev, c_dev,
            u_axis_value=0.0, sigma_axis_value=float(factor), tcol=tcol,
            n_features=n_features, batch_size=batch_size,
            positivity=positivity,
            sigma_pack_iu=sigma_pack_iu, sigma_pack_ju=sigma_pack_ju,
        )
    raise ValueError(f"unknown mode {mode!r}")


# ============================================================================
# 1) Per-axis 1D shift closure
# ============================================================================

def plot_shift_closure(
    target, w_event, args, out_dir, log_r_grid_shift, target_std_per_dim,
):
    """Per-axis shift-closure plots from a precomputed log r grid."""
    n_features = target.shape[1]
    target_components = list(range(min(n_features, len(TARGET_NAMES), 3)))
    factors = args.shift_factors

    n_rows = len(factors)
    n_cols = len(target_components)
    fig, axes = plt.subplots(
        2 * n_rows, n_cols,
        figsize=(4.0 * n_cols, 3.5 * n_rows),
        sharex="col",
        gridspec_kw={"height_ratios": [3, 1] * n_rows, "hspace": 0.05},
        squeeze=False,
        layout="constrained",
    )

    for r, factor in enumerate(factors):
        for cidx, tcol in enumerate(target_components):
            log_r = log_r_grid_shift[(float(factor), int(tcol))]
            r_pred = np.exp(log_r)

            lo, hi = np.percentile(
                target[:, tcol],
                [args.range_percentile, 100 - args.range_percentile],
            )
            bins = np.linspace(lo, hi, args.n_bins + 1)
            centers = 0.5 * (bins[:-1] + bins[1:])

            dy = float(factor) * float(target_std_per_dim[tcol])
            y_shifted = target[:, tcol] + dy

            h_raw, e_raw = _weighted_hist_err(
                target[:, tcol], bins, w_event,
            )
            h_shifted, e_shifted = _weighted_hist_err(
                y_shifted, bins, w_event,
            )
            h_pred, e_pred = _weighted_hist_err(
                target[:, tcol], bins,
                (w_event * r_pred).astype(np.float64),
            )

            ax_main = axes[2 * r][cidx]
            ax_main.step(
                centers, h_raw, where="mid", color="0.4",
                linestyle=":", lw=1.0, label="raw MC",
            )
            ax_main.step(
                centers, h_shifted, where="mid", color="k", lw=1.0,
                label=f"shifted MC ({factor:g}·σ_y)",
            )
            _stepped_errorbar(ax_main, centers, h_shifted, e_shifted, "k")
            ax_main.step(
                centers, h_pred, where="mid", color="C0",
                linestyle="--", lw=1.0, label="MLP pred W reweight",
            )
            _stepped_errorbar(ax_main, centers, h_pred, e_pred, "C0")
            ax_main.set_ylabel("events")
            tname = TARGET_NAMES[tcol] if tcol < len(TARGET_NAMES) else f"target[{tcol}]"
            ax_main.set_title(
                f"shift |δ|={factor:g} along {tname}", fontsize=10,
            )
            if r == 0 and cidx == 0:
                ax_main.legend(loc="best", fontsize=7)

            ax_ratio = axes[2 * r + 1][cidx]
            ratio, e_ratio = _ratio_with_err(
                h_pred, e_pred, h_shifted, e_shifted,
            )
            ax_ratio.step(
                centers, ratio, where="mid",
                color="C0", linestyle="--", lw=1.0,
            )
            _stepped_errorbar(ax_ratio, centers, ratio, e_ratio, "C0")
            ax_ratio.axhline(1.0, color="k", lw=0.5, alpha=0.5)
            ax_ratio.set_ylabel("/ shifted MC", fontsize=8)
            ax_ratio.set_xlabel(tname)
            ratios = np.atleast_1d(ratio)
            ratios = ratios[np.isfinite(ratios)]
            if ratios.size:
                lo_r, hi_r = np.quantile(ratios, [0.05, 0.95])
                pad = max(0.05, 0.5 * (hi_r - lo_r))
                ax_ratio.set_ylim(
                    max(0.0, min(lo_r - pad, 1.0 - pad)),
                    max(hi_r + pad, 1.0 + pad),
                )

    fig.suptitle("MLP shift-reweight closure")
    _save(fig, os.path.join(out_dir, "shift_closure.png"))


# ============================================================================
# 2) Per-axis 1D smear closure
# ============================================================================

def plot_smear_closure(
    target, w_event, args, out_dir, log_r_grid_smear, target_std_per_dim,
    log_r_grid_smear_via_gh=None,
):
    """Per-axis smear-closure plots.

    For each (factor, tcol) the smeared-MC histogram is built from
    ``y_smeared = y + ε·σ·e_tcol`` with one ε ~ N(0, 1) per event —
    matching exactly the rank-1 smear the model was trained on. The
    pred-W reweight uses the smear-only log r at u = 0,
    σ_vec = factor·e_tcol.

    When ``log_r_grid_smear_via_gh`` is provided, an additional
    overlaid curve uses the *shift* component of the polyhead with
    a K-point Gauss-Hermite expansion over the per-event ε to build
    a second estimate of the smear weight. Discrepancy between the
    two reweight curves is a diagnostic of polyhead self-consistency
    between its pure-σ and pure-u parts.
    """
    n_features = target.shape[1]
    target_components = list(range(min(n_features, len(TARGET_NAMES), 3)))
    factors = args.smear_factors
    rng = np.random.default_rng(args.seed)

    n_rows = len(factors)
    n_cols = len(target_components)
    fig, axes = plt.subplots(
        2 * n_rows, n_cols,
        figsize=(4.0 * n_cols, 3.5 * n_rows),
        sharex="col",
        gridspec_kw={"height_ratios": [3, 1] * n_rows, "hspace": 0.05},
        squeeze=False,
        layout="constrained",
    )

    for r, factor in enumerate(factors):
        for cidx, tcol in enumerate(target_components):
            log_r = log_r_grid_smear[(float(factor), int(tcol))]
            r_pred = np.exp(log_r)

            lo, hi = np.percentile(
                target[:, tcol],
                [args.range_percentile, 100 - args.range_percentile],
            )
            bins = np.linspace(lo, hi, args.n_bins + 1)
            centers = 0.5 * (bins[:-1] + bins[1:])

            dy = float(factor) * float(target_std_per_dim[tcol])
            # Per-event ε draw — one draw per event, axis-aligned smear
            # along tcol. Same ε shared across factors here for cleaner
            # visual comparison row-to-row; switch to per-(factor, tcol)
            # seed if you want truly independent draws.
            eps = rng.standard_normal(target.shape[0])
            y_smeared = target[:, tcol] + dy * eps

            h_raw, e_raw = _weighted_hist_err(
                target[:, tcol], bins, w_event,
            )
            h_smeared, e_smeared = _weighted_hist_err(
                y_smeared, bins, w_event,
            )
            h_pred, e_pred = _weighted_hist_err(
                target[:, tcol], bins,
                (w_event * r_pred).astype(np.float64),
            )

            ax_main = axes[2 * r][cidx]
            ax_main.step(
                centers, h_raw, where="mid", color="0.4",
                linestyle=":", lw=1.0, label="raw MC",
            )
            ax_main.step(
                centers, h_smeared, where="mid", color="k", lw=1.0,
                label=f"smeared MC ({factor:g}·σ_y, K=1)",
            )
            _stepped_errorbar(ax_main, centers, h_smeared, e_smeared, "k")
            ax_main.step(
                centers, h_pred, where="mid", color="C2",
                linestyle="--", lw=1.0, label="MLP pred W reweight",
            )
            _stepped_errorbar(ax_main, centers, h_pred, e_pred, "C2")

            # Optional: GH-on-shift smear estimate.
            h_pred_gh = e_pred_gh = None
            if log_r_grid_smear_via_gh is not None:
                log_r_gh = log_r_grid_smear_via_gh[
                    (float(factor), int(tcol))
                ]
                r_pred_gh = np.exp(log_r_gh)
                h_pred_gh, e_pred_gh = _weighted_hist_err(
                    target[:, tcol], bins,
                    (w_event * r_pred_gh).astype(np.float64),
                )
                ax_main.step(
                    centers, h_pred_gh, where="mid", color="C1",
                    linestyle="-.", lw=1.0,
                    label=f"GH(K={int(args.smear_gh_K)}) on shift",
                )
                _stepped_errorbar(
                    ax_main, centers, h_pred_gh, e_pred_gh, "C1",
                )

            ax_main.set_ylabel("events")
            tname = TARGET_NAMES[tcol] if tcol < len(TARGET_NAMES) else f"target[{tcol}]"
            ax_main.set_title(
                f"smear |σ|={factor:g} along {tname}", fontsize=10,
            )
            if r == 0 and cidx == 0:
                ax_main.legend(loc="best", fontsize=7)

            ax_ratio = axes[2 * r + 1][cidx]
            ratio, e_ratio = _ratio_with_err(
                h_pred, e_pred, h_smeared, e_smeared,
            )
            ax_ratio.step(
                centers, ratio, where="mid",
                color="C2", linestyle="--", lw=1.0,
            )
            _stepped_errorbar(ax_ratio, centers, ratio, e_ratio, "C2")
            ratios_for_ylim = [ratio]
            if h_pred_gh is not None:
                ratio_gh, e_ratio_gh = _ratio_with_err(
                    h_pred_gh, e_pred_gh, h_smeared, e_smeared,
                )
                ax_ratio.step(
                    centers, ratio_gh, where="mid",
                    color="C1", linestyle="-.", lw=1.0,
                )
                _stepped_errorbar(
                    ax_ratio, centers, ratio_gh, e_ratio_gh, "C1",
                )
                ratios_for_ylim.append(ratio_gh)
            ax_ratio.axhline(1.0, color="k", lw=0.5, alpha=0.5)
            ax_ratio.set_ylabel("/ smeared MC", fontsize=8)
            ax_ratio.set_xlabel(tname)
            ratios = np.concatenate([
                np.atleast_1d(rr)[np.isfinite(np.atleast_1d(rr))]
                for rr in ratios_for_ylim
            ]) if ratios_for_ylim else np.empty(0)
            if ratios.size:
                lo_r, hi_r = np.quantile(ratios, [0.05, 0.95])
                pad = max(0.05, 0.5 * (hi_r - lo_r))
                ax_ratio.set_ylim(
                    max(0.0, min(lo_r - pad, 1.0 - pad)),
                    max(hi_r + pad, 1.0 + pad),
                )

    fig.suptitle(
        "MLP smear-reweight closure (rank-1 K=1 smeared MC; ratios "
        "carry K=1 sampling noise)"
    )
    _save(fig, os.path.join(out_dir, "smear_closure.png"))


# ============================================================================
# 2.5) Per-bin error comparison: polyhead-reweight vs literal shifted/
#      smeared MC. The per-bin σ = √Σwᵢ² is a direct measure of
#      effective sample size; the ratio σ_pred / σ_ref tells you the
#      statistical-inefficiency cost of using the reweight in place
#      of a literal sample. Ratio = 1 is optimal; ratio > 1 means the
#      reweight inflates the error vs a fresh sample.
# ============================================================================

def plot_closure_error_ratio(
    target, w_event, args, out_dir, log_r_grid, target_std_per_dim,
    mode,
):
    """Per-bin √Σw² error comparison: polyhead reweight vs literal
    shifted/smeared MC, with the variance-optimal constant-per-bin
    reweight as the achievable floor.

    Per (factor, axis) cell, two stacked panels:
      * Top: three overlaid step histograms (log y):
          - σ_ref = √Σ wᵢ² at perturbed y (literal shifted/smeared
            MC).
          - σ_pred = √Σ (wᵢ · r̂)² at unperturbed y (polyhead
            reweight).
          - σ_optimal = (h_ref / h_raw) · σ_raw — the variance-
            minimal per-bin error achievable by **any** reweight
            that matches the perturbed-MC bin totals (constant per-
            bin reweight). Lower bound on what the polyhead can
            possibly reach.
      * Bottom: σ_pred / σ_optimal and σ_ref / σ_optimal per bin
        (linear y). σ_pred / σ_optimal is bounded below by 1
        (variance-minimality of the optimal reweight subject to
        matching bin totals); σ_ref / σ_optimal is the
        reference-vs-floor benchmark.

    The polyhead's per-bin σ_pred can never be lower than σ_optimal,
    so σ_pred / σ_optimal ≥ 1 everywhere; values close to 1 mean
    the polyhead is achieving near-uniform per-bin reweight (the
    intra-bin reweight values vary little). Values ≫ 1 mean the
    polyhead is assigning highly variable weights within bin —
    visible signal of intra-bin reweight roughness.

    σ_ref / σ_optimal tells you how the literal sample compares to
    the achievable floor: > 1 in regions where the reference is
    statistically inferior to a perfectly-uniform reweight (i.e.
    where reweighting *can* beat a fresh sample if done right);
    < 1 where the reference is intrinsically more efficient than
    any reweight could be (e.g., perturbation moving events into
    sparse origin bins).

    ``mode = 'shift'`` uses the shifted-MC reference (deterministic
    shift along the tcol axis); ``mode = 'smear'`` uses the K=1
    stochastic smeared-MC reference. Layouts and binning match
    ``plot_shift_closure`` / ``plot_smear_closure``.
    """
    assert mode in ("shift", "smear")
    n_features = target.shape[1]
    target_components = list(range(min(n_features, len(TARGET_NAMES), 3)))
    factors = (
        args.shift_factors if mode == "shift" else args.smear_factors
    )
    rng = (
        np.random.default_rng(args.seed) if mode == "smear" else None
    )

    n_rows = len(factors)
    n_cols = len(target_components)
    fig, axes = plt.subplots(
        2 * n_rows, n_cols,
        figsize=(4.0 * n_cols, 3.5 * n_rows),
        sharex="col",
        gridspec_kw={"height_ratios": [3, 1] * n_rows, "hspace": 0.05},
        squeeze=False,
        layout="constrained",
    )

    for r, factor in enumerate(factors):
        for cidx, tcol in enumerate(target_components):
            log_r = log_r_grid[(float(factor), int(tcol))]
            r_pred = np.exp(log_r)

            lo, hi = np.percentile(
                target[:, tcol],
                [args.range_percentile, 100 - args.range_percentile],
            )
            bins = np.linspace(lo, hi, args.n_bins + 1)
            centers = 0.5 * (bins[:-1] + bins[1:])

            dy = float(factor) * float(target_std_per_dim[tcol])
            if mode == "shift":
                y_ref = target[:, tcol] + dy
                ref_label = f"shifted MC ({factor:g}·σ_y)"
            else:
                eps = rng.standard_normal(target.shape[0])
                y_ref = target[:, tcol] + dy * eps
                ref_label = (
                    f"smeared MC ({factor:g}·σ_y, K=1 stochastic)"
                )

            # Three histograms / sigma vectors:
            #   raw      — unperturbed MC at original y (bin-fill).
            #   ref      — perturbed MC at shifted/smeared y.
            #   pred     — polyhead-reweighted MC at original y.
            h_raw, e_raw = _weighted_hist_err(
                target[:, tcol], bins, w_event,
            )
            h_ref, e_ref = _weighted_hist_err(y_ref, bins, w_event)
            _, e_pred = _weighted_hist_err(
                target[:, tcol], bins,
                (w_event * r_pred).astype(np.float64),
            )
            # Optimal: constant-per-bin reweight = h_ref / h_raw.
            # σ_opt = (h_ref / h_raw) · σ_raw. NaN where h_raw == 0
            # (no events to reweight) so the line drops out cleanly
            # in those bins.
            scale = h_ref / np.where(h_raw > 0, h_raw, np.nan)
            e_optimal = scale * e_raw

            ax_main = axes[2 * r][cidx]
            ax_main.step(
                centers, e_ref, where="mid",
                color="k", lw=1.0, label=ref_label,
            )
            ax_main.step(
                centers, e_pred, where="mid",
                color="C0", linestyle="--", lw=1.0,
                label=r"polyhead reweight: $\sqrt{\sum (w \hat r)^2}$",
            )
            ax_main.step(
                centers, e_optimal, where="mid",
                color="C2", linestyle=":", lw=1.0,
                label=(
                    r"optimal reweight: $(h_{\rm ref}/h_{\rm raw})"
                    r" \sqrt{\sum w^2}$"
                ),
            )
            ax_main.set_yscale("log")
            ax_main.set_ylabel(r"$\sigma$ per bin")
            tname = (
                TARGET_NAMES[tcol] if tcol < len(TARGET_NAMES)
                else f"target[{tcol}]"
            )
            ax_main.set_title(
                f"{mode}: |{'δ' if mode == 'shift' else 'σ'}|="
                f"{factor:g} along {tname}",
                fontsize=10,
            )
            if r == 0 and cidx == 0:
                ax_main.legend(loc="best", fontsize=7)

            # Ratio panel — two step lines vs the σ_optimal floor.
            # Bins with h_raw == 0 (no events to reweight) make
            # σ_optimal NaN and drop both lines cleanly.
            denom = np.where(e_optimal > 0, e_optimal, np.nan)
            ratio_pred = e_pred / denom
            ratio_ref = e_ref / denom
            ax_ratio = axes[2 * r + 1][cidx]
            ax_ratio.step(
                centers, ratio_pred, where="mid",
                color="C0", lw=1.0,
                label=r"$\hat r$ / optimal",
            )
            ax_ratio.step(
                centers, ratio_ref, where="mid",
                color="k", lw=1.0,
                label="ref / optimal",
            )
            ax_ratio.axhline(1.0, color="k", lw=0.5, alpha=0.5)
            ax_ratio.set_ylabel(
                r"$\sigma / \sigma_{\rm optimal}$", fontsize=8,
            )
            ax_ratio.set_xlabel(tname)
            if r == 0 and cidx == 0:
                ax_ratio.legend(loc="best", fontsize=7)
            ratios_all = np.concatenate([
                ratio_pred[np.isfinite(ratio_pred)],
                ratio_ref[np.isfinite(ratio_ref)],
            ])
            if ratios_all.size:
                lo_r, hi_r = np.quantile(ratios_all, [0.05, 0.95])
                pad = max(0.05, 0.5 * (hi_r - lo_r))
                ax_ratio.set_ylim(
                    max(0.0, min(lo_r - pad, 0.9)),
                    max(hi_r + pad, 1.1),
                )

    fig.suptitle(
        f"Polyhead reweight per-bin error vs {mode} MC: "
        r"ratio relative to the optimal-reweight floor"
    )
    _save(
        fig,
        os.path.join(out_dir, f"{mode}_closure_error_ratio.png"),
    )


# ============================================================================
# 3) log r distribution (per axis × magnitude × mode)
# ============================================================================

def plot_log_r_distribution(
    n_features, w_event, args, out_dir,
    log_r_grid_shift, log_r_grid_smear,
):
    target_components = list(range(min(n_features, len(TARGET_NAMES), 3)))

    fig, axes = plt.subplots(
        2, len(target_components),
        figsize=(4.5 * len(target_components), 7.0),
        squeeze=False,
        layout="constrained",
    )

    for row, (mode, grid, factors) in enumerate([
        ("shift", log_r_grid_shift, args.shift_factors),
        ("smear", log_r_grid_smear, args.smear_factors),
    ]):
        for cidx, tcol in enumerate(target_components):
            ax = axes[row][cidx]
            all_log_r = []
            labels = []
            for factor in factors:
                all_log_r.append(grid[(float(factor), int(tcol))])
                tag = "δ" if mode == "shift" else "σ"
                labels.append(f"|{tag}|={factor:g}·σ_y")

            concat = np.concatenate(all_log_r)
            finite = concat[np.isfinite(concat)]
            if finite.size == 0:
                continue
            lo, hi = np.percentile(finite, [0.5, 99.5])
            m = max(abs(lo), abs(hi), 0.05)
            bins = np.linspace(-m, m, 80)

            centers = 0.5 * (bins[:-1] + bins[1:])
            for i, (log_r, lbl) in enumerate(zip(all_log_r, labels)):
                color = f"C{i}"
                h, sigma = _weighted_hist_err(log_r, bins, w_event)
                ax.step(
                    centers, h, where="mid", color=color, lw=1.2,
                    label=lbl,
                )
                _stepped_errorbar(ax, centers, h, sigma, color)
            ax.axvline(0.0, color="k", lw=0.5, alpha=0.5)
            ax.set_xlabel("log r")
            ax.set_ylabel("events (weighted)")
            tname = TARGET_NAMES[tcol] if tcol < len(TARGET_NAMES) else f"target[{tcol}]"
            ax.set_title(f"{mode}: log r along {tname}", fontsize=10)
            ax.legend(fontsize=7)
            ax.set_yscale("log")

    fig.suptitle("MLP predicted log r — distribution across events")
    _save(fig, os.path.join(out_dir, "log_r_distribution.png"))


# ============================================================================
# 4) log r vs |perturbation| scan (shift and smear)
# ============================================================================

def plot_log_r_scan(
    model, arch, y_dev, c_dev, n_features, w_event, args, positivity,
    out_dir, sigma_pack_iu, sigma_pack_ju,
):
    target_components = list(range(min(n_features, len(TARGET_NAMES), 3)))
    u_max = float(max(args.shift_factors))
    sigma_max = float(max(args.smear_factors))
    u_grid = np.linspace(-u_max, u_max, args.scan_magnitudes)
    # Smear is even in σ so the negative-σ side is redundant; use a
    # one-sided scan from 0 to σ_max to keep the panel readable.
    sigma_grid = np.linspace(0.0, sigma_max, args.scan_magnitudes)

    w64 = w_event.astype(np.float64)
    wsum = float(w64.sum())

    fig, axes = plt.subplots(
        2, len(target_components),
        figsize=(4.5 * len(target_components), 7.0),
        squeeze=False, layout="constrained",
    )

    for row, (mode, grid, label_x) in enumerate([
        ("shift", u_grid, "u (in σ_y units)"),
        ("smear", sigma_grid, "|σ_vec| (in σ_y units)"),
    ]):
        for cidx, tcol in enumerate(target_components):
            ax = axes[row][cidx]
            means = []
            stds = []
            sigma_means = []
            for v in grid:
                log_r = _common_predict_call(
                    model, arch, y_dev, c_dev, float(v), int(tcol),
                    mode, n_features, args.batch_size, positivity,
                    sigma_pack_iu, sigma_pack_ju,
                ).astype(np.float64)
                mean = (w64 * log_r).sum() / max(wsum, 1e-30)
                var = (w64 * (log_r - mean) ** 2).sum() / max(wsum, 1e-30)
                # Statistical uncertainty on the weighted mean:
                #   σ²<x> = Σ wᵢ² (xᵢ − <x>)² / (Σ wᵢ)²
                # = (effective-N) variance with Σw² in the numerator.
                sigma_mean_sq = (
                    (w64 * w64 * (log_r - mean) ** 2).sum()
                    / max(wsum * wsum, 1e-60)
                )
                means.append(mean)
                stds.append(math.sqrt(max(var, 0.0)))
                sigma_means.append(math.sqrt(max(sigma_mean_sq, 0.0)))
            means = np.asarray(means)
            stds = np.asarray(stds)
            sigma_means = np.asarray(sigma_means)
            color = "C0" if mode == "shift" else "C2"
            ax.errorbar(
                grid, means, yerr=sigma_means,
                fmt="o-", color=color, lw=1.2, capsize=2,
                label="⟨log r⟩  ± stat (√Σw²/Σw)",
            )
            ax.fill_between(
                grid, means - stds, means + stds,
                alpha=0.25, color=color, label="±1σ across events",
            )
            ax.axhline(0.0, color="k", lw=0.5, alpha=0.5)
            ax.axvline(0.0, color="k", lw=0.5, alpha=0.5)
            ax.set_xlabel(label_x)
            ax.set_ylabel("log r")
            tname = TARGET_NAMES[tcol] if tcol < len(TARGET_NAMES) else f"target[{tcol}]"
            ax.set_title(
                f"{mode}: ⟨log r⟩ along {tname}", fontsize=10,
            )
            ax.legend(fontsize=7)

    fig.suptitle(
        "MLP predicted log r — magnitude scan (top: shift / "
        "bottom: smear)"
    )
    _save(fig, os.path.join(out_dir, "log_r_scan.png"))


# ============================================================================
# Optional: BCE-polyhead vs flow-truth scatter
# ============================================================================

def plot_polyhead_pred_vs_flow(
    model, arch, flow, flow_stats, polyhead_stats,
    target_std, cond, n_features, w_event,
    args, out_dir,
    sigma_pack_iu=None, sigma_pack_ju=None,
):
    """Scatter / 2D-density of this script's polyhead-predicted W vs.
    a flow's analytic W on a common ``(y, c, u, σ)`` grid. Mirrors
    flow_training_diagnostics.py's ``plot_polyhead_pred_vs_true`` so
    the two estimators are comparable on the same metric.

    Three panels: SHIFT (σ=0), SMEAR (u=0), JOINT (both nonzero).
    Reports ``log_W rms`` and ``bias`` per panel, computed
    weighted-by-``w_event``. The diagonal indicates a perfect
    polyhead-vs-flow agreement; the BCE polyhead's training target
    *is* the same density ratio the flow estimates analytically, so
    in the population limit both should sit on the diagonal.

    Assumes ``flow_stats`` and ``polyhead_stats`` are compatible
    (i.e. both trained on the same MC sample with the same
    standardization). Only checks for shape compatibility — caller
    is expected to verify upstream.
    """
    device = args.device
    n_use = min(args.flow_validate_n, target_std.shape[0])
    rng = np.random.default_rng(0)
    idx = rng.choice(target_std.shape[0], size=n_use, replace=False)

    # Use the polyhead's stats to standardize. If flow's stats differ
    # we issue a warning — exact agreement requires aligned preproc.
    if not np.allclose(
        np.asarray(flow_stats.target_mean),
        np.asarray(polyhead_stats.target_mean),
        atol=1e-4,
    ) or not np.allclose(
        np.asarray(flow_stats.target_std),
        np.asarray(polyhead_stats.target_std),
        atol=1e-4,
    ):
        print(
            "  warning: flow preproc differs from polyhead preproc "
            "— pred-vs-flow plot will be approximate."
        )

    y_std_t = torch.from_numpy(target_std[idx]).to(device)
    c_std_t = torch.from_numpy(cond[idx]).to(device)
    w_sub = w_event[idx].astype(np.float32)

    # Sample perturbations in standardized target space, matching the
    # training-time sampler: |δ| uniform on [0, 1.3 · delta_max], v on
    # the unit sphere. Read magnitudes from the polyhead's training
    # config when available; fall back to the conventional 1.3.
    train_cfg = getattr(args, "_polyhead_train_cfg", {}) or {}
    delta_max_train = float(train_cfg.get("delta_max", 1.0))
    sigma_max_train = float(train_cfg.get("sigma_max", 1.0)) or 1.0
    oversample = 1.3
    half = oversample * delta_max_train
    half_sig = oversample * sigma_max_train

    g = torch.Generator().manual_seed(0)
    delta_shift = (
        torch.rand(n_use, generator=g) * 2.0 - 1.0
    ) * half
    v_shift = torch.randn(n_use, n_features, generator=g)
    v_shift = v_shift / v_shift.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    sigma_smear = torch.rand(n_use, generator=g) * half_sig
    v_smear = torch.randn(n_use, n_features, generator=g)
    v_smear = v_smear / v_smear.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    delta_smear = sigma_smear * torch.randn(n_use, generator=g)
    u_shift_full = (delta_shift.unsqueeze(-1) * v_shift).to(device)
    sigma_vec_full = (sigma_smear.unsqueeze(-1) * v_smear).to(device)
    delta_smear = delta_smear.to(device)
    v_smear = v_smear.to(device)

    # Local flow log-W computation — mirrors
    # flow_training_diagnostics._flow_log_w_at without taking a
    # cross-script dependency.
    def flow_log_w_at(u_eval):
        N = y_std_t.shape[0]
        out = np.empty(N, dtype=np.float32)
        flow.eval()
        with torch.no_grad():
            for s in range(0, N, args.batch_size):
                e = min(s + args.batch_size, N)
                y = y_std_t[s:e]
                c = c_std_t[s:e]
                u = u_eval[s:e]
                z, ladj = flow(c).transform.call_and_ladj(y)
                z_p, ladj_p = flow(c).transform.call_and_ladj(y - u)
                lw = -0.5 * (
                    (z_p * z_p).sum(dim=-1) - (z * z).sum(dim=-1)
                ) + (ladj_p - ladj)
                out[s:e] = lw.cpu().numpy().astype(np.float32)
        return out

    titles = ("SHIFT (σ=0)", "SMEAR (u=0)", "JOINT (both)")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharey=False)
    for col, title in enumerate(titles):
        if col == 0:
            u_eval = u_shift_full.clone()
            u_in = u_shift_full.clone()
            sigma_in = torch.zeros_like(sigma_vec_full)
        elif col == 1:
            u_eval = delta_smear.unsqueeze(-1) * v_smear
            u_in = torch.zeros_like(u_shift_full)
            sigma_in = sigma_vec_full.clone()
        else:
            u_eval = u_shift_full + delta_smear.unsqueeze(-1) * v_smear
            u_in = u_shift_full.clone()
            sigma_in = sigma_vec_full.clone()

        true_lw = flow_log_w_at(u_eval)
        pred_lw = predict_log_r_perevent(
            model, arch, y_std_t, c_std_t, u_in, sigma_in,
            args.batch_size, getattr(args, "_positivity", "exp"),
            sigma_pack_iu=sigma_pack_iu, sigma_pack_ju=sigma_pack_ju,
        )

        # Weighted RMS / bias of pred − true.
        err = pred_lw - true_lw
        finite = np.isfinite(err)
        e_f = err[finite]
        w_f = w_sub[finite]
        wsum = float(w_f.sum())
        if wsum <= 0.0:
            rms = bias = float("nan")
        else:
            bias = float((w_f * e_f).sum() / wsum)
            rms = float(np.sqrt((w_f * e_f * e_f).sum() / wsum))

        ax = axes[col]
        # 2D log-density scatter (mirrors flow+polyhead diagnostic).
        x = true_lw[finite] / np.log(10.0)
        y = pred_lw[finite] / np.log(10.0)
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        bins = np.linspace(lo, hi, 80)
        h, xe, ye = np.histogram2d(x, y, bins=[bins, bins], weights=w_f)
        from matplotlib.colors import LogNorm
        ax.pcolormesh(
            xe, ye, h.T,
            norm=LogNorm(vmin=max(1.0, h.max() * 1e-4), vmax=h.max()),
            cmap="viridis",
        )
        ax.plot([lo, hi], [lo, hi], "r-", lw=1.0, label="diagonal")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("log10  flow log-W")
        ax.set_ylabel("log10  BCE-polyhead log-W")
        ax.set_title(
            f"{title}\nlog-W rms={rms:.3f}  bias={bias:+.3f}",
        )
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("BCE polyhead pred  vs  flow log-W (analytic)")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "polyhead_pred_vs_flow.png"))


def plot_polyhead_axis_logw_error_vs_flow(
    model, arch, flow, flow_stats, polyhead_stats,
    target_std, cond, n_features, w_event,
    args, out_dir,
    sigma_pack_iu=None, sigma_pack_ju=None,
):
    """Per-axis polyhead-vs-flow log-W error histograms at the
    closure shift magnitudes. Mirror of
    flow_training_diagnostics.py's
    ``plot_polyhead_axis_logw_error`` for the direct-BCE polyhead.

    Lays out a grid of ``log(pred_W) − log(W_flow)`` histograms with
    rows = ``args.shift_factors`` and cols = target axes
    (r_kappa, dlambda, dphi). Each panel evaluates axis-aligned
    shifts ``u = δ · ê_axis`` for every event, with the flow's
    analytic log W as the reference, and reports weighted RMS and
    bias. Useful for direct comparison against the flow+polyhead's
    per-axis numbers — same metric, same shifts, same axes.

    Saves four files:
      * ``polyhead_axis_logw_error_vs_flow.{png,pdf}`` (linear y)
      * ``polyhead_axis_logw_error_vs_flow_log.{png,pdf}`` (log y)
    """
    device = args.device
    n_use = min(args.flow_validate_n, target_std.shape[0])
    rng = np.random.default_rng(2)
    idx = rng.choice(target_std.shape[0], size=n_use, replace=False)
    y_std_t = torch.from_numpy(target_std[idx]).to(device)
    c_std_t = torch.from_numpy(cond[idx]).to(device)
    w_sub = w_event[idx].astype(np.float32)

    deltas = list(args.shift_factors)
    target_components = list(range(min(n_features, len(TARGET_NAMES), 3)))
    target_names = TARGET_NAMES[: len(target_components)]

    n_rows = len(deltas)
    n_cols = len(target_components)

    # Local flow log-W at axis-aligned u (broadcast scalar mag * ê).
    def flow_log_w_axis(factor, axis):
        N = y_std_t.shape[0]
        out = np.empty(N, dtype=np.float32)
        u_buf = torch.zeros(args.batch_size, n_features, device=device)
        u_buf[:, axis] = float(factor)
        flow.eval()
        with torch.no_grad():
            for s in range(0, N, args.batch_size):
                e = min(s + args.batch_size, N)
                bsz = e - s
                u = u_buf[:bsz]
                y = y_std_t[s:e]
                c = c_std_t[s:e]
                z, ladj = flow(c).transform.call_and_ladj(y)
                z_p, ladj_p = flow(c).transform.call_and_ladj(y - u)
                lw = -0.5 * (
                    (z_p * z_p).sum(dim=-1) - (z * z).sum(dim=-1)
                ) + (ladj_p - ladj)
                out[s:e] = lw.cpu().numpy().astype(np.float32)
        return out

    panel_data = [[None] * n_cols for _ in range(n_rows)]
    for r, factor in enumerate(deltas):
        for c, axis in enumerate(target_components):
            true_lw = flow_log_w_axis(float(factor), int(axis))
            pred_lw = predict_log_r_axis(
                model, arch, y_std_t, c_std_t,
                u_axis_value=float(factor), sigma_axis_value=0.0,
                tcol=int(axis), n_features=n_features,
                batch_size=args.batch_size,
                positivity=getattr(args, "_positivity", "exp"),
                sigma_pack_iu=sigma_pack_iu, sigma_pack_ju=sigma_pack_ju,
            )
            log_err = pred_lw - true_lw
            finite = np.isfinite(log_err)
            le = log_err[finite]
            wle = w_sub[finite]
            wsum = float(wle.sum())
            if wsum <= 0.0:
                rms = bias = float("nan")
            else:
                rms = float(np.sqrt(np.average(le ** 2, weights=wle)))
                bias = float(np.average(le, weights=wle))
            panel_data[r][c] = (le, wle, rms, bias)

    for yscale, suffix in (("linear", ""), ("log", "_log")):
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4.0 * n_cols, 3.0 * n_rows),
            sharex="row", sharey="row", squeeze=False,
        )
        for r, factor in enumerate(deltas):
            spreads = []
            for c in range(n_cols):
                le, *_ = panel_data[r][c]
                lo, hi = np.quantile(le, [0.005, 0.995])
                spreads.append(max(abs(lo), abs(hi)))
            spread = max(spreads) if spreads else 1.0
            bins = np.linspace(-spread, spread, 81)
            for c, name in enumerate(target_names):
                ax = axes[r][c]
                le, wle, rms, bias = panel_data[r][c]
                ax.hist(
                    le, bins=bins, weights=wle, histtype="step",
                    color="C0", lw=1.5,
                )
                ax.axvline(0.0, color="k", lw=0.8, alpha=0.5)
                ax.set_title(
                    f"|δ|={factor:g} along {name}: "
                    f"rms={rms:.4f} bias={bias:+.4f}"
                )
                ax.set_yscale(yscale)
                if r == n_rows - 1:
                    ax.set_xlabel(
                        r"$\log(\hat W) - \log W_{\rm flow}$"
                    )
                if c == 0:
                    ax.set_ylabel("weighted events")
        fig.suptitle(
            "BCE polyhead reconstruction error vs flow, "
            "axis-aligned shifts"
        )
        fig.tight_layout()
        _save(
            fig,
            os.path.join(
                out_dir,
                f"polyhead_axis_logw_error_vs_flow{suffix}.png",
            ),
        )


# ============================================================================
# Per-source target-distribution comparison
# ============================================================================

def _parse_source_labels(spec):
    """Parse ``["0:Jpsi", "100:Zmumu"]`` into ``{0: "Jpsi", 100: "Zmumu"}``."""
    if spec is None:
        return {}
    out = {}
    for entry in spec:
        if ":" not in entry:
            raise ValueError(
                f"--cmp-source-labels entry {entry!r} missing ':' "
                "separator; expected ``id:label``"
            )
        sid_str, lbl = entry.split(":", 1)
        out[int(sid_str)] = lbl
    return out


def _read_manifest_source_labels(paths):
    """Locate the shard ``manifest.json`` reachable from ``paths`` and
    return its ``source_labels`` map as ``{int: str}``.

    Recognises three forms of input path:
      * a directory containing ``manifest.json`` (the standard shard dir);
      * an explicit ``manifest.json`` file;
      * a shard file -- in which case the sibling ``manifest.json`` in
        the containing directory is used.

    Last-write-wins across multiple inputs; missing files are skipped
    silently so older shards without the labels block still work.
    """
    labels: dict[int, str] = {}
    seen: set[str] = set()
    for p in paths:
        mfp = None
        if os.path.isdir(p):
            cand = os.path.join(p, "manifest.json")
            if os.path.exists(cand):
                mfp = cand
        elif os.path.isfile(p):
            if os.path.basename(p) == "manifest.json":
                mfp = p
            else:
                cand = os.path.join(
                    os.path.dirname(os.path.abspath(p)), "manifest.json",
                )
                if os.path.exists(cand):
                    mfp = cand
        if mfp is None or mfp in seen:
            continue
        seen.add(mfp)
        try:
            with open(mfp) as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  warning: failed to read {mfp}: {exc!r}")
            continue
        for k, v in (manifest.get("source_labels") or {}).items():
            try:
                labels[int(k)] = str(v)
            except (ValueError, TypeError):
                continue
    return labels


def _load_per_source_window(
    paths, *, pt_min, pt_max, eta_min, eta_max,
    max_events=-1, n_workers=8,
):
    """Parallel streaming loader specialised for the per-source target-
    distribution plots.

    Reads Arrow IPC shards via a thread pool (pyarrow IO and numpy
    vector ops release the GIL during the heavy work), applies the
    ``(pt_gen, eta_gen)`` window per shard, and computes the three
    target columns inline. Peak memory is bounded by the *kept* rows
    across all shards, not the full dataset — so a narrow window over
    the full statistics is cheap. Charge is not filtered here: the
    caller may split by charge in the rendering pass without reloading.

    Returns ``(data, rows_read, rows_kept)`` where ``data`` is either
    ``None`` (no kept rows) or a 5-tuple of arrays
    ``(target [N,3], weight [N], source_id [N], kappa_gen [N],
    eta_gen [N])``. ``rows_read`` is the total raw rows pulled off
    disk (after any ``max_events`` truncation); ``rows_kept`` counts
    those that passed the (pt, eta, finite, w>0) mask.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import pyarrow as pa
    import pyarrow.ipc as ipc

    # Reuse the shard-directory expansion already used by load_ntuples.
    from train_muon_response_flow import _expand_input_paths

    paths = _expand_input_paths(paths)
    if not paths:
        raise ValueError(
            "_load_per_source_window: no input paths after expansion"
        )
    if not all(p.endswith(".arrow") for p in paths):
        raise ValueError(
            "_load_per_source_window: only Arrow IPC shards (.arrow) "
            "or a shard directory with manifest.json are supported"
        )

    _MAGIC = b"ARROW1"

    def _read_one(p):
        with pa.memory_map(p, "r") as f:
            head = f.read(len(_MAGIC))
            f.seek(0)
            if head == _MAGIC:
                t = ipc.open_file(f).read_all()
            else:
                t = ipc.open_stream(f).read_all()
        eta_r = t["eta_reco"].to_numpy().astype(np.float64)
        phi_r = t["phi_reco"].to_numpy().astype(np.float64)
        eta_g = t["eta_gen"].to_numpy().astype(np.float64)
        phi_g = t["phi_gen"].to_numpy().astype(np.float64)
        kappa_r = t["kappa_reco"].to_numpy().astype(np.float64)
        kappa_g = t["kappa_gen"].to_numpy().astype(np.float64)
        w_raw = t["nominal_weight"].to_numpy().astype(np.float64)
        sid = t["source_id"].to_numpy().astype(np.int32, copy=False)
        n_block = eta_r.shape[0]

        abs_k = np.fabs(kappa_g)
        safe_k = np.where(abs_k > 0, abs_k, np.nan)
        pt_g = 1.0 / (safe_k * np.cosh(eta_g))
        mask = (
            np.isfinite(pt_g)
            & (pt_g >= pt_min) & (pt_g <= pt_max)
            & (eta_g >= eta_min) & (eta_g <= eta_max)
            & np.isfinite(kappa_r) & np.isfinite(kappa_g)
            & (w_raw > 0.0)
        )
        if not mask.any():
            return None, n_block
        eta_r = eta_r[mask]; phi_r = phi_r[mask]
        eta_g = eta_g[mask]; phi_g = phi_g[mask]
        kappa_r = kappa_r[mask]; kappa_g = kappa_g[mask]
        w = w_raw[mask]; sid = sid[mask]

        # Inline target columns from compute_targets_and_conditioning,
        # restricted to the masked rows so we never form full-shard
        # target arrays.
        lam_r = np.arctan(np.sinh(eta_r))
        lam_g = np.arctan(np.sinh(eta_g))
        r_kappa = kappa_r / kappa_g - 1.0
        dphi = np.arctan2(
            np.sin(phi_r - phi_g), np.cos(phi_r - phi_g),
        )
        dlambda = lam_r - lam_g
        target = np.stack([r_kappa, dlambda, dphi], axis=1)

        return (
            (target, w.astype(np.float32), sid, kappa_g, eta_g),
            n_block,
        )

    blocks = []
    rows_read = 0
    rows_kept = 0
    with ThreadPoolExecutor(max_workers=max(1, int(n_workers))) as ex:
        futures = {ex.submit(_read_one, p): p for p in paths}
        for fut in as_completed(futures):
            try:
                res, n_block = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[per-source loader] shard {futures[fut]} failed: "
                    f"{exc!r}"
                )
                continue
            rows_read += n_block
            if res is not None:
                blocks.append(res)
                rows_kept += res[0].shape[0]
            if max_events > 0 and rows_read >= max_events:
                for f in futures:
                    f.cancel()
                break

    if not blocks:
        return None, rows_read, 0

    target = np.concatenate([b[0] for b in blocks], axis=0)
    w = np.concatenate([b[1] for b in blocks])
    sid = np.concatenate([b[2] for b in blocks])
    kappa_g = np.concatenate([b[3] for b in blocks])
    eta_g = np.concatenate([b[4] for b in blocks])
    return (target, w, sid, kappa_g, eta_g), rows_read, rows_kept


def plot_per_source_target_distributions(
    target, w_event, source_id, kappa_g, eta_g, args, out_dir,
):
    """Compare the marginal target distributions across datasets
    (distinct ``source_id`` values) inside a phase-space window in
    (pt_gen, eta_gen, charge_gen). phi_gen is integrated.

    Per target component the top row shows weighted, area-normalised
    histograms (one curve per source); the bottom row is the bin-wise
    ratio to a reference source. Bin edges are taken from the
    pooled-sample quantiles inside the window so all sources share the
    same binning.
    """
    target = np.asarray(target)
    n_features = target.shape[1]
    target_components = list(range(min(n_features, len(TARGET_NAMES), 3)))

    # Reconstruct pt_gen, charge_gen from kappa_gen + eta_gen.
    kappa_g = np.asarray(kappa_g, dtype=np.float64)
    eta_g = np.asarray(eta_g, dtype=np.float64)
    abs_k = np.fabs(kappa_g)
    safe_k = np.where(abs_k > 0, abs_k, np.nan)
    pt_g = 1.0 / (safe_k * np.cosh(eta_g))
    charge_g = np.sign(kappa_g).astype(np.int8)

    # Phase-space window (pt, eta) — shared across charge modes.
    pt_min = float(args.cmp_pt_min)
    pt_max = float(args.cmp_pt_max)
    eta_min = float(args.cmp_eta_min)
    eta_max = float(args.cmp_eta_max)
    base_mask = (
        np.isfinite(pt_g)
        & (pt_g >= pt_min) & (pt_g <= pt_max)
        & (eta_g >= eta_min) & (eta_g <= eta_max)
    )

    # Charge modes: ``each`` runs pos and neg in separate plots.
    if args.cmp_charge == "each":
        charge_modes = ["pos", "neg"]
    else:
        charge_modes = [args.cmp_charge]

    # Label precedence (lowest -> highest):
    #   1. hard-coded DEFAULT_SOURCE_LABELS    -- last-resort fallback
    #   2. manifest.json's ``source_labels``   -- the authoritative
    #      id-to-sample mapping written by the snapshot/sharder pipeline
    #   3. ``--cmp-source-labels`` CLI args    -- user override
    label_map = dict(DEFAULT_SOURCE_LABELS)
    label_map.update(_read_manifest_source_labels(args.input_files))
    label_map.update(_parse_source_labels(args.cmp_source_labels))
    def _source_label(sid):
        return label_map.get(int(sid), f"source {int(sid)}")

    w_event = np.asarray(w_event, dtype=np.float64)
    source_id = np.asarray(source_id)

    n_cols = len(target_components)

    for charge_mode in charge_modes:
        if charge_mode == "pos":
            mask = base_mask & (charge_g > 0)
        elif charge_mode == "neg":
            mask = base_mask & (charge_g < 0)
        else:
            mask = base_mask

        n_in = int(mask.sum())
        if n_in == 0:
            print(
                "[per-source target distributions] window is empty "
                f"(pt∈[{pt_min},{pt_max}], η∈[{eta_min},{eta_max}], "
                f"charge={charge_mode}); skipping"
            )
            continue

        target_w = target[mask]
        w_w = w_event[mask]
        src_w = source_id[mask]

        unique_sources = np.unique(src_w).tolist()
        if len(unique_sources) < 2:
            print(
                "[per-source target distributions] only "
                f"{len(unique_sources)} source(s) in {charge_mode} window "
                f"{unique_sources}; comparison needs ≥2 sources — skipping"
            )
            continue

        if args.cmp_ref_source is not None:
            ref_source = int(args.cmp_ref_source)
            if ref_source not in unique_sources:
                print(
                    f"[per-source target distributions] --cmp-ref-source="
                    f"{ref_source} not present in {charge_mode} window; "
                    f"falling back to {unique_sources[0]}"
                )
                ref_source = int(unique_sources[0])
        else:
            ref_source = int(unique_sources[0])

        print(
            f"[per-source target distributions] window: "
            f"pt∈[{pt_min:g},{pt_max:g}] GeV, "
            f"η∈[{eta_min:g},{eta_max:g}], charge={charge_mode}; "
            f"{n_in:,} muons; sources={unique_sources}; ref={ref_source}"
        )

        # Precompute per-column binned data so we can render the same
        # plot twice (linear + log y-axis) without redoing the
        # histogramming.
        per_col = {}
        for tcol in target_components:
            y = target_w[:, tcol]
            finite = np.isfinite(y)
            if not finite.all():
                y = y[finite]
                w_col = w_w[finite]
                s_col = src_w[finite]
            else:
                w_col = w_w
                s_col = src_w
            if y.size == 0:
                continue

            pct = float(args.range_percentile)
            lo, hi = np.percentile(y, [pct, 100.0 - pct])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(np.min(y)), float(np.max(y))
                if hi <= lo:
                    hi = lo + 1.0
            bins = np.linspace(lo, hi, args.n_bins + 1)
            centers = 0.5 * (bins[:-1] + bins[1:])
            bw = bins[1] - bins[0]

            ref_mask = (s_col == ref_source)
            h_ref, e_ref = _weighted_hist_err(
                y[ref_mask], bins, w_col[ref_mask],
            )
            norm_ref = h_ref.sum()
            if norm_ref <= 0:
                print(
                    "[per-source target distributions] ref source "
                    f"{ref_source} has zero weight in column "
                    f"{TARGET_NAMES[tcol]} (charge={charge_mode}); skipping"
                )
                continue
            h_ref_n = h_ref / (norm_ref * bw)
            e_ref_n = e_ref / (norm_ref * bw)

            per_source = []
            for i, sid in enumerate(unique_sources):
                sm = (s_col == sid)
                if sm.sum() == 0:
                    continue
                h, e = _weighted_hist_err(y[sm], bins, w_col[sm])
                tot = h.sum()
                if tot <= 0:
                    continue
                h_n = h / (tot * bw)
                e_n = e / (tot * bw)
                per_source.append({
                    "sid": int(sid),
                    "n": int(sm.sum()),
                    "color": f"C{i}",
                    "h": h_n,
                    "e": e_n,
                })

            per_col[tcol] = {
                "centers": centers,
                "per_source": per_source,
                "h_ref": h_ref_n,
                "e_ref": e_ref_n,
            }

        if not per_col:
            continue

        ch_tag = {"pos": "q>0", "neg": "q<0", "both": "both q"}[charge_mode]
        ch_suffix = {"pos": "_qpos", "neg": "_qneg", "both": "_qboth"}[
            charge_mode
        ]

        for yscale, ysuffix in (("linear", ""), ("log", "_log")):
            fig, axes = plt.subplots(
                2, n_cols,
                figsize=(4.5 * n_cols, 6.5),
                sharex="col", squeeze=False,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
                layout="constrained",
            )

            for cidx, tcol in enumerate(target_components):
                ax_main = axes[0][cidx]
                ax_ratio = axes[1][cidx]

                data = per_col.get(tcol)
                if data is None:
                    continue
                centers = data["centers"]
                h_ref_n = data["h_ref"]
                e_ref_n = data["e_ref"]

                for entry in data["per_source"]:
                    sid = entry["sid"]
                    color = entry["color"]
                    lbl = f"{_source_label(sid)}  (N={entry['n']})"
                    ax_main.step(
                        centers, entry["h"], where="mid",
                        color=color, lw=1.2, label=lbl,
                    )
                    _stepped_errorbar(
                        ax_main, centers, entry["h"], entry["e"], color,
                    )

                    if sid == ref_source:
                        ax_ratio.axhline(1.0, color=color, lw=0.8)
                        continue
                    ratio, e_ratio = _ratio_with_err(
                        entry["h"], entry["e"], h_ref_n, e_ref_n,
                    )
                    ax_ratio.step(
                        centers, ratio, where="mid", color=color, lw=1.0,
                    )
                    _stepped_errorbar(
                        ax_ratio, centers, ratio, e_ratio, color,
                    )

                ax_main.set_yscale(yscale)
                ax_main.set_ylabel("p.d.f. (weighted, unit area)")
                ax_main.set_title(
                    f"{TARGET_NAMES[tcol] if tcol < len(TARGET_NAMES) else f'target[{tcol}]'}",
                    fontsize=10,
                )
                ax_main.legend(fontsize=7)
                ax_main.axvline(0.0, color="k", lw=0.4, alpha=0.4)

                ax_ratio.axhline(1.0, color="k", lw=0.4, alpha=0.4)
                ax_ratio.set_ylabel(
                    f"ratio /\n{_source_label(ref_source)}", fontsize=8,
                )
                ax_ratio.set_xlabel(
                    TARGET_NAMES[tcol]
                    if tcol < len(TARGET_NAMES) else f"target[{tcol}]"
                )
                ax_ratio.set_ylim(0.5, 1.5)

            fig.suptitle(
                f"Per-source target distributions  ·  "
                f"pt∈[{pt_min:g},{pt_max:g}] GeV, "
                f"η∈[{eta_min:g},{eta_max:g}], {ch_tag} (φ integrated)"
            )
            _save(
                fig,
                os.path.join(
                    out_dir,
                    f"per_source_target_distributions{ch_suffix}{ysuffix}.png",
                ),
            )


# ============================================================================
# main
# ============================================================================

def main():
    args = parse_args()
    if args.output is None:
        args.output = os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(args.output, exist_ok=True)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = "cuda:0"
    else:
        device = "cpu"
    args.device = device

    print(f"loading checkpoint {args.checkpoint}")
    model, arch, stats, train_cfg = load_model_from_checkpoint(
        args.checkpoint, device,
    )
    if stats is None:
        # Older / per-epoch checkpoints didn't bundle PreprocStats.
        ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        preproc_path = os.path.join(ckpt_dir, "preproc.json")
        if not os.path.exists(preproc_path):
            raise SystemExit(
                f"Checkpoint has no PreprocStats and no preproc.json "
                f"at {preproc_path}."
            )
        with open(preproc_path) as f:
            stats = PreprocStats(**json.load(f))
        print(f"  loaded preproc stats from {preproc_path}")
    positivity = str(train_cfg.get("positivity", "softplus"))
    loss_fn = str(train_cfg.get("loss_fn", "?"))
    delta_max = train_cfg.get("delta_max", "?")
    sigma_max_train = train_cfg.get("sigma_max", "?")
    print(
        f"  arch={arch} loss={loss_fn} positivity={positivity} "
        f"delta_max={delta_max} sigma_max={sigma_max_train}"
    )

    print(f"loading ntuples from {len(args.input_files)} file(s)")
    t0 = _tic()
    (
        eta_r, phi_r, eta_g, phi_g, kappa_r, kappa_g, w, source_id,
    ) = load_ntuples(
        args.input_files, args.tree, args.max_muons,
        args.pt_min, args.pt_max, args.eta_max,
        threads=args.threads, max_events=args.max_events,
    )
    t0 = _toc("load_ntuples", t0)

    # Subsample at raw-array stage (cheap rng.integers with sorted
    # gather, no O(N) np.random.choice permutation).
    N_raw = eta_r.shape[0]
    if args.n_events > 0 and N_raw > args.n_events:
        rng = np.random.default_rng(0)
        sel = np.sort(rng.integers(0, N_raw, args.n_events))
        eta_r, phi_r = eta_r[sel], phi_r[sel]
        eta_g, phi_g = eta_g[sel], phi_g[sel]
        kappa_r, kappa_g, w = kappa_r[sel], kappa_g[sel], w[sel]
        source_id = source_id[sel]
        print(
            f"  loaded {N_raw} muon rows; subsampled to "
            f"{eta_r.shape[0]} for plotting"
        )
    else:
        print(f"  loaded {N_raw} muon rows")
    t0 = _toc("subsample raw arrays", t0)

    target, cond_raw = compute_targets_and_conditioning(
        eta_r, phi_r, eta_g, phi_g, kappa_r, kappa_g,
    )
    t0 = _toc("compute_targets_and_conditioning", t0)

    w = (w / w.mean()).astype(np.float32)
    target_std, cond = apply_preproc(target, cond_raw, stats)
    t0 = _toc("apply_preproc + weight norm", t0)

    y_dev = torch.from_numpy(target_std).to(device, non_blocking=False)
    c_dev = torch.from_numpy(cond).to(device, non_blocking=False)
    n_features = target_std.shape[1]
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = _toc("data H2D to device", t0)

    target_std_per_dim = np.asarray(stats.target_std, dtype=np.float64)

    # Σ_pack indices for the MLP arch.
    sigma_pack_iu = sigma_pack_ju = None
    if arch in ("mlp", "mlp-factored"):
        iu, ju = _sigma_pack_indices(n_features)
        sigma_pack_iu, sigma_pack_ju = iu.to(device), ju.to(device)

    # Precompute the (factor × tcol × mode) log r grid. Both the
    # closure and the distribution plots iterate the same grid, so
    # caching halves the inference work for those.
    target_components = list(range(min(n_features, len(TARGET_NAMES), 3)))
    print(
        f"  predicting log r grid: shift={len(args.shift_factors)} × "
        f"smear={len(args.smear_factors)} × "
        f"axes={len(target_components)} ..."
    )
    log_r_grid_shift = {}
    log_r_grid_smear = {}
    for factor in args.shift_factors:
        for tcol in target_components:
            log_r_grid_shift[(float(factor), int(tcol))] = (
                _common_predict_call(
                    model, arch, y_dev, c_dev, float(factor), int(tcol),
                    "shift", n_features, args.batch_size, positivity,
                    sigma_pack_iu, sigma_pack_ju,
                )
            )
    for factor in args.smear_factors:
        for tcol in target_components:
            log_r_grid_smear[(float(factor), int(tcol))] = (
                _common_predict_call(
                    model, arch, y_dev, c_dev, float(factor), int(tcol),
                    "smear", n_features, args.batch_size, positivity,
                    sigma_pack_iu, sigma_pack_ju,
                )
            )
    t0 = _toc("grid prediction (shift + smear)", t0)

    # GH-on-shift smear-closure curve (self-consistency between the
    # polyhead's pure-σ and pure-u parts). Skip when --smear-gh-K=0.
    log_r_grid_smear_via_gh = None
    if int(args.smear_gh_K) > 0:
        K = int(args.smear_gh_K)
        print(
            f"  predicting GH(K={K}) smear-via-shift grid: "
            f"{len(args.smear_factors)} × {len(target_components)} "
            f"× {K} shift evals ..."
        )
        log_r_grid_smear_via_gh = {}
        for factor in args.smear_factors:
            for tcol in target_components:
                log_r_grid_smear_via_gh[(float(factor), int(tcol))] = (
                    predict_smear_via_gh_shift(
                        model, arch, y_dev, c_dev,
                        float(factor), int(tcol), K,
                        n_features, args.batch_size, positivity,
                        sigma_pack_iu=sigma_pack_iu,
                        sigma_pack_ju=sigma_pack_ju,
                    )
                )
        t0 = _toc(f"GH(K={K}) smear-via-shift grid prediction", t0)

    plot_shift_closure(
        target, w, args, args.output,
        log_r_grid_shift, target_std_per_dim,
    )
    t0 = _toc("plot_shift_closure", t0)

    plot_closure_error_ratio(
        target, w, args, args.output,
        log_r_grid_shift, target_std_per_dim, mode="shift",
    )
    t0 = _toc("plot_shift_closure_error_ratio", t0)

    plot_smear_closure(
        target, w, args, args.output,
        log_r_grid_smear, target_std_per_dim,
        log_r_grid_smear_via_gh=log_r_grid_smear_via_gh,
    )
    t0 = _toc("plot_smear_closure", t0)

    plot_closure_error_ratio(
        target, w, args, args.output,
        log_r_grid_smear, target_std_per_dim, mode="smear",
    )
    t0 = _toc("plot_smear_closure_error_ratio", t0)

    plot_log_r_distribution(
        n_features, w, args, args.output,
        log_r_grid_shift, log_r_grid_smear,
    )
    t0 = _toc("plot_log_r_distribution", t0)

    plot_log_r_scan(
        model, arch, y_dev, c_dev, n_features, w, args,
        positivity, args.output, sigma_pack_iu, sigma_pack_ju,
    )
    t0 = _toc("plot_log_r_scan", t0)

    # Per-source comparison uses its own streaming, parallel loader so
    # it can run over the full statistics (much larger than the
    # subsampled in-memory arrays the inference plots use). The (pt,
    # eta) window is applied per shard so peak memory stays bounded.
    print(
        "loading per-source comparison window via streaming loader: "
        f"pt∈[{args.cmp_pt_min:g},{args.cmp_pt_max:g}] GeV, "
        f"η∈[{args.cmp_eta_min:g},{args.cmp_eta_max:g}], "
        f"max_events={args.cmp_max_events}, workers={args.cmp_workers}"
    )
    cmp_data, cmp_read, cmp_kept = _load_per_source_window(
        args.input_files,
        pt_min=float(args.cmp_pt_min), pt_max=float(args.cmp_pt_max),
        eta_min=float(args.cmp_eta_min), eta_max=float(args.cmp_eta_max),
        max_events=int(args.cmp_max_events),
        n_workers=int(args.cmp_workers),
    )
    t0 = _toc("load per-source window (streaming)", t0)
    if cmp_data is None:
        print(
            f"[per-source target distributions] no rows passed the "
            f"window ({cmp_read} read); skipping plot"
        )
    else:
        target_cmp, w_cmp, sid_cmp, kg_cmp, eg_cmp = cmp_data
        # Normalise weights the same way the standard plots do (mean=1)
        # so legend yields and ratio axes are comparable.
        w_cmp = (w_cmp / w_cmp.mean()).astype(np.float32)
        print(
            f"  read {cmp_read:,} raw rows; kept {cmp_kept:,} after the "
            f"window cut (={100 * cmp_kept / max(cmp_read, 1):.2f}%)"
        )
        plot_per_source_target_distributions(
            target_cmp, w_cmp, sid_cmp, kg_cmp, eg_cmp,
            args, args.output,
        )
        t0 = _toc("plot_per_source_target_distributions", t0)

    if args.flow_checkpoint:
        print(f"loading flow checkpoint {args.flow_checkpoint}")
        # Lazy import: only needed when flow comparison is requested,
        # and avoids the cost of importing the heavier flow-diag module
        # for the standard diagnostic path.
        from flow_training_diagnostics import (
            load_flow_from_checkpoint,
        )
        flow, flow_stats, _flow_ckpt = load_flow_from_checkpoint(
            args.flow_checkpoint, device,
        )
        flow.eval()
        # Stash these on args so plot_polyhead_pred_vs_flow can reach
        # the polyhead's training config + positivity without a wider
        # signature change.
        args._polyhead_train_cfg = train_cfg
        args._positivity = positivity
        plot_polyhead_pred_vs_flow(
            model, arch, flow, flow_stats, stats,
            target_std, cond, n_features, w,
            args, args.output,
            sigma_pack_iu=sigma_pack_iu,
            sigma_pack_ju=sigma_pack_ju,
        )
        t0 = _toc("plot_polyhead_pred_vs_flow", t0)

        plot_polyhead_axis_logw_error_vs_flow(
            model, arch, flow, flow_stats, stats,
            target_std, cond, n_features, w,
            args, args.output,
            sigma_pack_iu=sigma_pack_iu,
            sigma_pack_ju=sigma_pack_ju,
        )
        t0 = _toc("plot_polyhead_axis_logw_error_vs_flow", t0)

    print(f"wrote diagnostic plots to {args.output}")


if __name__ == "__main__":
    main()
