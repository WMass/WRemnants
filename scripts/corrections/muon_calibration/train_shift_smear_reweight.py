"""Direct shift (and optionally shift+smear) density-ratio training
without a flow.

By default the architecture and training loop are **shift-only**:
``σ_vec ≡ 0``, perturbation sampling collapses to a single SHIFT mode,
and the σ-dependent parts of each architecture are dropped. Pass
``--include-smear`` to enable the full shift+smear+joint setup.

Two architecture modes (``--arch``):

  * ``mlp`` (default): "B" construction — dual scalar forward.
        log r = positivity( f(y, c, u, Σ_pack) − f(y, c, 0, 0) )
    with split ``trunk(y, c) → e`` and ``head(e, u, Σ_pack) → f``.
    ``Σ_pack`` is the upper-triangular packing of ``Σ = σ_vec σ_vecᵀ``
    (n(n+1)/2 entries for n = n_features). The subtraction enforces
    ``r = 1`` at zero perturbation; conditioning the head on ``Σ``
    rather than ``σ_vec`` raw makes the prediction structurally even
    in ``σ_vec``. In shift-only mode the head's Σ_pack channel is
    removed entirely (no dead weights).

  * ``polyhead``: ``trunk(y, c) → polynomial coefficients``; the
    polynomial itself encodes the (u, σ_vec) dependence. Same
    ``PolyHead`` as in ``train_muon_response_flow.py``. Supports
    configurable max degrees in u (``--max-deg-u``), σ
    (``--max-deg-sigma``, must be even), and combined cross terms
    (``--max-cross-deg``). Structural priors (no constant term, even
    in σ) are enforced by the basis-index construction. In shift-only
    mode the basis is restricted to pure-α (β = ∅) terms only.

Both architectures train with one of two flow-free density-ratio
losses on shifted (+ optionally smeared) MC pairs:

  * ``lsif`` (default) — least-squares importance fitting:
        L = E_{y~p_nom}[r̂²]  −  2·E_{y~p_pert}[r̂]
    Unique optimum r̂* = r_marginal. Unbiased even with K=1
    stochastic ε per event (the ε integral is just sampling, no
    Jensen gap).
  * ``dv`` — Donsker-Varadhan:
        L = log E_{y~p_nom}[exp(T)]  −  E_{y~p_pert}[T]
    where T = log r̂. Slightly biased upward at finite batch.

Per-event sampling:
    y_pert = y₀ + u + ε · σ_vec,    ε ~ N(0, 1) scalar
            ↑ shift     ↑ rank-1 smear noise (only with --include-smear;
                          σ_vec ≡ 0 otherwise)

Multi-GPU: ``--num-gpus N`` (or auto-detect by default) spawns N
worker processes via ``torch.multiprocessing.spawn``. Each worker takes
one GPU, with NCCL (GPU) or gloo (CPU) for the rendezvous. Data
tensors are loaded once on rank 0 and shared with workers via
``tensor.share_memory_()`` so children attach rather than re-load.
Per-rank index shards are deterministic from the global ``--seed``.
Train/val metrics are all-reduced at each epoch boundary so all ranks
see the same numbers (and hence make the same early-stopping
decisions); only rank 0 prints, writes checkpoints, and saves the
final artifact.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from train_muon_response_flow import _build_basis_aux  # noqa: E402
from train_muon_response_flow import _select_basis  # noqa: E402
from train_muon_response_flow import (  # noqa: E402
    PreprocStats,
    _joint_indices,
    _state_dict_to_cpu,
    evaluate_joint,
)


_LOG_W_CLAMP = 30.0
_LOG_LOG2 = math.log(math.log(2.0))


# ============================================================================
# Smolyak sparse grid for shift-axis training
# ============================================================================
#
# Standard Clenshaw-Curtis (Chebyshev-Lobatto) Smolyak construction in
# d dimensions at level L:
#   level L grid = ⋃_{α: α_j ≥ 0, Σ α_j ≤ L}  X_{α_1} × ... × X_{α_d}
# where X_α is the 1D Chebyshev-Lobatto node set:
#   X_0 = {0}; X_α = {-cos(π k / (n_α-1)) : k = 0,...,n_α-1}, n_α = 2^α + 1.
# This convention has the "doubling growth rule" so points at level α
# are nested in level α+1, and is exact for polynomials of total degree
# ≤ 2L.
#
# Point counts in d=3:
#   L=0: 1, L=1: 7, L=2: 25, L=3: 69, L=4: 177, L=5: 441
#
# The "auto" level for the polyhead-arch shift training picks the
# smallest L whose Smolyak grid has *more* points than the number of
# pure-shift basis functions (multi-indices α with 1 ≤ |α| ≤
# max_deg_u, in d dimensions). This makes the grid over-determine the
# polynomial coefficients with the smallest possible point budget,
# which is the natural choice given the user's "fully described by
# the polynomial" assumption.

def _smolyak_cheb_lobatto_1d(level: int):
    """Chebyshev-Lobatto nodes at Smolyak axis-level α (0-indexed):
    α=0 → {0}; α≥1 → 2^α + 1 nodes including ±1, equally spaced in
    cos angle. Nested across α."""
    if level <= 0:
        return [0.0]
    n = (1 << level) + 1
    return [
        -math.cos(math.pi * k / (n - 1)) for k in range(n)
    ]


def _smolyak_grid(d: int, L: int):
    """Smolyak sparse grid in [-1, 1]^d at level ``L``. Returns a
    deduplicated, sorted ``np.ndarray`` of shape ``[K, d]``."""
    import itertools as _it
    points = set()
    # Iterate (α_1, ..., α_d) with α_j ≥ 0 and Σ α_j ≤ L.
    for alpha in _it.product(range(L + 1), repeat=d):
        if sum(alpha) > L:
            continue
        node_sets = [_smolyak_cheb_lobatto_1d(a) for a in alpha]
        for combo in _it.product(*node_sets):
            # Round to suppress floating-point dups across nested levels.
            points.add(tuple(round(x, 12) for x in combo))
    arr = np.array(sorted(points), dtype=np.float64)
    return arr


def _n_pure_shift_basis(d: int, max_deg_u: int) -> int:
    """Number of multi-indices α ∈ ℕ^d with 1 ≤ |α| ≤ max_deg_u —
    i.e., the count of pure-shift polynomial basis functions for a
    polyhead with the given degree bound. Closed form:
    C(d + max_deg_u, d) − 1."""
    return math.comb(d + max_deg_u, d) - 1


def _auto_smolyak_level(d: int, n_target: int, max_level: int = 8) -> int:
    """Lowest Smolyak level ``L`` such that the d-dim level-L sparse
    grid has *strictly more* than ``n_target`` points. Cap at
    ``max_level`` (16k+ points in 3D) to avoid runaway grids if the
    target is unreachable."""
    for L in range(max_level + 1):
        if _smolyak_grid(d, L).shape[0] > n_target:
            return L
    return max_level


_ACTIVATIONS = {
    "gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh,
}


# ============================================================================
# Σ-pack helpers (upper-triangular flatten of σσᵀ)
# ============================================================================

def _sigma_pack_indices(n_features: int):
    """Indices into a flat n(n+1)/2 vector that mirror the upper
    triangle of σσᵀ with ``i ≤ j``. Returns ``(iu, ju)`` as 1-D long
    tensors of length ``n(n+1)/2``."""
    iu, ju = torch.triu_indices(n_features, n_features).unbind(0)
    return iu.contiguous(), ju.contiguous()


def _pack_sigma_outer(sigma_vec, iu, ju):
    """Pack ``Σ = σσᵀ`` upper-triangular: returns shape ``[..., n(n+1)/2]``
    with entries ``Σ_pack[k] = σ_{iu[k]} · σ_{ju[k]}``."""
    return sigma_vec[..., iu] * sigma_vec[..., ju]


# ============================================================================
# Gauss–Hermite quadrature helpers (probabilist's; weights sum to 1)
# ============================================================================

def _gh_nodes_weights(K: int, dtype=torch.float32):
    """Probabilist's Gauss-Hermite nodes/weights for the standard
    Gaussian ``N(0, 1)``. Returns ``(nodes, weights)`` with ``nodes``
    shape ``[K]`` and ``weights`` summing to 1. Returns ``(None, None)``
    if ``K <= 1`` (signaling the K=1 stochastic path).
    """
    if K <= 1:
        return None, None
    nodes_np, w_np = np.polynomial.hermite_e.hermegauss(K)
    w_norm = w_np / np.sqrt(2.0 * np.pi)
    return (
        torch.tensor(nodes_np, dtype=dtype),
        torch.tensor(w_norm, dtype=dtype),
    )


def _lagrange_basis_at(eps, gh_nodes):
    """Lagrange basis ``L_k(ε)`` for each event's ``ε`` at the K
    Gauss-Hermite nodes. Returns shape ``[B, K]`` with ``L[b, k] = Π_{j≠k}
    (ε_b - node_j) / (node_k - node_j)``.

    Used by the GH+residual control-variate estimator: the polynomial
    interpolant ``g(ε) = Σ_k f(ε_k) · L_k(ε)`` passes through
    ``(ε_k, f_k)`` exactly, and ``E_ε[g] = S_K`` is the GH sum
    (deterministic, zero variance), so the residual ``f(ε~) − g(ε~)``
    captures only what's beyond the polynomial truncation.
    """
    K = gh_nodes.shape[0]
    B = eps.shape[0]
    L = torch.ones(B, K, device=eps.device, dtype=eps.dtype)
    for k in range(K):
        # Π_{j≠k} (ε - node_j) / (node_k - node_j).
        for j in range(K):
            if j == k:
                continue
            L[:, k] = L[:, k] * (eps - gh_nodes[j]) / (gh_nodes[k] - gh_nodes[j])
    return L


# ============================================================================
# Architecture: B mode (dual scalar forward)
# ============================================================================

class ReweightMLP_B(nn.Module):
    """Trunk + head MLP for the dual-forward construction.

    ``trunk(y, c) → e ∈ ℝ^{d_emb}``, then ``head(e, u, Σ_pack) → f``.
    The shift+smear log-ratio is built outside as

        log r = positivity( head(e, u, Σ_pack) − head(e, 0, 0) )

    so ``r = 1`` exactly at ``(u, σ) = (0, 0)`` and is structurally
    even in ``σ_vec`` because ``Σ_pack`` is invariant under
    ``σ_vec → −σ_vec``. The score ``∂_u log r |₀`` and the smear
    Hessian are unconstrained — both fall out naturally from
    derivatives of ``f`` evaluated at zero perturbation.

    **Head structure**: the first head layer ``Linear(d_emb + F +
    n_pack, H)`` is split per-input-block into three parallel
    ``nn.Linear`` modules

        ``head_layer1_e(e) + head_layer1_u(u) + head_layer1_sigma(σ_pack)``

    (the σ-side dropped in shift-only mode), with the bias attached
    to ``head_layer1_e``. ``head_rest`` is the activation +
    remaining hidden layers + final ``Linear(H, 1)``. This is
    mathematically identical to the packed
    ``Linear(d_emb+F+n_pack, H)`` followed by the rest, with the
    same parameter count, but exposes the per-input-block structure
    so that:

      * inference can amortise ``head_layer1_e(e)`` once per event
        across many ``(u, σ)`` queries (~2× per-query head
        speedup; see :class:`CombinedMLPInference` in the export
        script);
      * the polyhead-style "exploit shared inputs across multiple
        evaluations" pattern is structural rather than something
        the export wrapper has to discover by slicing trained
        weights.

    Legacy checkpoints with the old packed-head layout (state-dict
    keys ``head.{i}.{weight,bias}``) load transparently via
    :meth:`_remap_legacy_state_dict`.
    """

    def __init__(
        self,
        n_features: int,
        n_cond: int,
        d_emb: int = 32,
        trunk_hidden: int = 64,
        trunk_layers: int = 2,
        head_hidden: int = 32,
        head_layers: int = 2,
        activation=nn.GELU,
        shift_only: bool = False,
        gauss_baseline: nn.Module = None,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.n_cond = int(n_cond)
        self.d_emb = int(d_emb)
        self.head_hidden = int(head_hidden)
        self.head_layers = int(head_layers)
        self.shift_only = bool(shift_only)
        self.n_sigma_pack = self.n_features * (self.n_features + 1) // 2
        # Optional analytic Gaussian baseline (additive log-r term).
        # When set, :func:`compute_d_quadrature` adds its closed-form
        # log-density-ratio to the head's pre-positivity ``d`` so the
        # MLP residual only fits the deviation from a Gaussian.
        self.gauss_baseline = gauss_baseline

        # Trunk: (y, c) → e
        layers = []
        prev = self.n_features + self.n_cond
        for _ in range(trunk_layers):
            layers.append(nn.Linear(prev, trunk_hidden))
            layers.append(activation())
            prev = trunk_hidden
        layers.append(nn.Linear(prev, self.d_emb))
        self.trunk = nn.Sequential(*layers)

        # Head layer 1 is split into per-block sub-Linears. Bias lives
        # on the e-side; the u- and σ-sides are bias-free, so the
        # total parameter count matches a single packed Linear.
        self.head_layer1_e = nn.Linear(self.d_emb, head_hidden, bias=True)
        self.head_layer1_u = nn.Linear(self.n_features, head_hidden, bias=False)
        # σ-side dropped in shift-only mode (would be dead weights).
        if not self.shift_only:
            self.head_layer1_sigma = nn.Linear(
                self.n_sigma_pack, head_hidden, bias=False,
            )
        else:
            self.head_layer1_sigma = None

        # Rest of head: activation + remaining hidden layers + final
        # Linear(H, 1). With the head_layers=N convention, layer 1 is
        # the input projection (now split, above), and the remaining
        # ``head_layers − 1`` hidden Linears + final scalar Linear
        # live in head_rest.
        rest = [activation()]
        prev = head_hidden
        for _ in range(self.head_layers - 1):
            rest.append(nn.Linear(prev, head_hidden))
            rest.append(activation())
            prev = head_hidden
        rest.append(nn.Linear(prev, 1))
        self.head_rest = nn.Sequential(*rest)
        # Init the final Linear near zero so f(...) ≈ const →
        # (f − f) ≈ 0 → r ≈ 1 at start.
        with torch.no_grad():
            self.head_rest[-1].weight.mul_(0.01)
            self.head_rest[-1].bias.zero_()

    def trunk_forward(self, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.trunk(torch.cat([y, c], dim=-1))

    def head_forward(
        self, e: torch.Tensor, u: torch.Tensor, sigma_pack: torch.Tensor,
    ) -> torch.Tensor:
        """``f(e, u, σ_pack) ∈ R``. Mathematically identical to the
        legacy packed-head form ``head_rest(act(W·[e, u, σ_pack] + b))``
        with ``W = [W_e | W_u | W_σ]`` and ``b = b_e``; the per-block
        decomposition is structural, not numerical."""
        x = self.head_layer1_e(e) + self.head_layer1_u(u)
        if self.head_layer1_sigma is not None:
            x = x + self.head_layer1_sigma(sigma_pack)
        return self.head_rest(x).squeeze(-1)

    # ------------------------------------------------------------------
    # Legacy state-dict shim
    # ------------------------------------------------------------------

    def _remap_legacy_state_dict(self, state_dict):
        """Migrate a pre-split-head checkpoint into the new layout.

        Old: a single ``self.head`` ``nn.Sequential`` ending in
        ``Linear(H, 1)``. State-dict keys ``head.{i}.{weight,bias}``
        with ``i = 0`` the input projection ``Linear(d_emb + F +
        n_pack, H)`` and ``i = 2k`` the remaining hidden / final
        Linears.

        New: ``head_layer1_e/_u/_sigma`` plus ``head_rest`` (an
        ``nn.Sequential`` whose ``[0]`` is the activation immediately
        following the split layer 1, ``[1]`` is the next Linear, and
        so on).

        Migration:
          * ``head.0.weight`` row-blocks split into the three
            sub-layers' weights; ``head.0.bias`` → ``head_layer1_e.bias``.
          * ``head.{i}.{weight,bias}`` for ``i ≥ 1`` →
            ``head_rest.{i-1}.{weight,bias}`` (one-index shift since
            head_rest starts with the activation that was at index 1
            of the old packed Sequential).

        If no ``head.*`` keys are present, returns the input unchanged.
        """
        legacy_prefix = "head."
        # Match either "head.0.weight" or any number; we look at all
        # keys starting with "head." that aren't already targeting the
        # new layout.
        new_keys = (
            "head_layer1_e", "head_layer1_u", "head_layer1_sigma",
            "head_rest",
        )
        legacy_keys = [
            k for k in state_dict.keys()
            if k.startswith(legacy_prefix)
            and not any(k.startswith(p) for p in new_keys)
        ]
        if not legacy_keys:
            return state_dict

        # Find which Sequential indices appear (sorted).
        layer_idxs = sorted({
            int(k.split(".")[1]) for k in legacy_keys
            if k.endswith(".weight")
        })
        if not layer_idxs:
            return state_dict

        new_state = dict(state_dict)
        first_idx = layer_idxs[0]
        # 1) Split ``head.{first_idx}.weight``: row-block-split by
        #    input dimension. Old packed input order was
        #    [e (d_emb), u (n_features), σ_pack (n_sigma_pack if not
        #    shift_only)]. Bias goes to the e-side.
        w_old = new_state.pop(f"head.{first_idx}.weight", None)
        b_old = new_state.pop(f"head.{first_idx}.bias", None)
        if w_old is not None and b_old is not None:
            n_e = self.d_emb
            n_u = self.n_features
            new_state["head_layer1_e.weight"] = w_old[:, :n_e].contiguous()
            new_state["head_layer1_e.bias"] = b_old.contiguous()
            new_state["head_layer1_u.weight"] = (
                w_old[:, n_e:n_e + n_u].contiguous()
            )
            if not self.shift_only:
                new_state["head_layer1_sigma.weight"] = (
                    w_old[:, n_e + n_u:].contiguous()
                )

        # 2) ``head.{i}.{weight,bias}`` for i > first_idx →
        #    ``head_rest.{i-1-first_idx}.{weight,bias}``. The packed
        #    sequential had layer 1 at index ``first_idx`` (typically
        #    0); head_rest starts immediately after layer 1, so its
        #    index 0 is the activation at packed index ``first_idx + 1``.
        for i in layer_idxs[1:]:
            new_idx = i - first_idx - 1
            for suf in ("weight", "bias"):
                k_old = f"head.{i}.{suf}"
                if k_old in new_state:
                    new_state[f"head_rest.{new_idx}.{suf}"] = new_state.pop(k_old)
        return new_state

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Override to transparently migrate legacy packed-head
        checkpoints. Forwards to :meth:`nn.Module.load_state_dict`
        after running :meth:`_remap_legacy_state_dict`."""
        return super().load_state_dict(
            self._remap_legacy_state_dict(state_dict),
            strict=strict, assign=assign,
        )


# ============================================================================
# Architecture: structurally factored MLP head
# ============================================================================

class ReweightMLPFactored(nn.Module):
    """Trunk + structurally factored head returning the log-ratio directly.

    The pre-positivity scalar is
    ::
        log r = ⟨u, A(e, ·)⟩ + ⟨σ_pack, B(e, ·)⟩ [ + ⟨u⊗σ_pack, C(e, u, σ_pack)⟩ ]
    where ``e = trunk(y, c)``. Both ``log r = 0`` at ``(u, σ) = (0, 0)``
    and ``σ → −σ`` invariance are structural (no dual-forward
    subtraction needed); ``σ_pack`` is the upper-triangular flat of
    ``σσᵀ``, even under ``σ → −σ``.

    The factorisation form is controlled by two flags set at
    construction time:

    | shift_detach | smear_detach | A inputs        | B inputs        | C |
    |--------------|--------------|-----------------|-----------------|---|
    | False        | False        | e, u, σ_pack    | e, u, σ_pack    | – |
    | True         | False        | e, u            | e, u, σ_pack    | – |
    | False        | True         | e, u, σ_pack    | e, σ_pack       | – |
    | True         | True         | e, u            | e, σ_pack       | ✓ |

    In the (False, False) "default" form, A's σ-dependence and B's
    u-dependence absorb the cross-term coupling; no pure-block detach
    handle is exposed. Each detach flag adds a structural factorisation
    that exposes the corresponding pure-block. When both flags are set
    the cross is moved into a dedicated head ``C(e, u, σ_pack)``
    contracting with the outer product ``u ⊗ σ_pack`` (so the cross
    vanishes whenever u=0 OR σ=0).

    The same trunk amortisation as :class:`ReweightMLP_B` applies: the
    trunk MLP runs once on ``(y, c)``, then the per-block sub-heads
    operate on the shared embedding ``e``.

    The optional ``gauss_baseline`` adds a closed-form Gaussian
    additive contribution to ``log r``, identical in form to
    :class:`ReweightMLP_B`'s.
    """

    # Flag for loss-code dispatch (so callers know head_forward already
    # returns the pre-positivity ``d`` directly, without dual-forward
    # subtraction).
    is_factored = True

    def __init__(
        self,
        n_features: int,
        n_cond: int,
        d_emb: int = 32,
        trunk_hidden: int = 64,
        trunk_layers: int = 2,
        head_hidden: int = 32,
        head_layers: int = 2,
        activation=nn.GELU,
        shift_only: bool = False,
        detach_pure_shift_in_joint: bool = False,
        detach_pure_smear_in_joint: bool = False,
        gauss_baseline: nn.Module = None,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.n_cond = int(n_cond)
        self.d_emb = int(d_emb)
        self.head_hidden = int(head_hidden)
        self.head_layers = int(head_layers)
        self.shift_only = bool(shift_only)
        self.n_sigma_pack = (
            self.n_features * (self.n_features + 1) // 2
            if not self.shift_only else 0
        )
        self.gauss_baseline = gauss_baseline
        self.detach_pure_shift_in_joint = bool(
            detach_pure_shift_in_joint
        ) and not self.shift_only
        self.detach_pure_smear_in_joint = bool(
            detach_pure_smear_in_joint
        ) and not self.shift_only
        # A independent of σ_pack iff shift-detach is on.
        self.A_uses_sigma = (
            (not self.shift_only) and not self.detach_pure_shift_in_joint
        )
        # B independent of u iff smear-detach is on.
        self.B_uses_u = (
            (not self.shift_only) and not self.detach_pure_smear_in_joint
        )
        # Separate cross head only when both flags are on (fully
        # factorised three-term form). The cross is the only place that
        # JOINT-mode gradients survive when both pure blocks are
        # detached.
        self.has_C = (
            self.detach_pure_shift_in_joint
            and self.detach_pure_smear_in_joint
        )
        # Shift-only mode: the σ side collapses out entirely; B and C
        # are unused. log r = ⟨u, A(e, u)⟩ trivially.
        if self.shift_only:
            self.A_uses_sigma = False
            self.B_uses_u = False
            self.has_C = False

        # Trunk: (y, c) → e
        layers = []
        prev = self.n_features + self.n_cond
        for _ in range(trunk_layers):
            layers.append(nn.Linear(prev, trunk_hidden))
            layers.append(activation())
            prev = trunk_hidden
        layers.append(nn.Linear(prev, self.d_emb))
        self.trunk = nn.Sequential(*layers)

        # A head: e + u (+ σ_pack) → F.
        in_dim_A = self.d_emb + self.n_features + (
            self.n_sigma_pack if self.A_uses_sigma else 0
        )
        self.A_head = self._make_mlp(
            in_dim_A, head_hidden, head_layers, activation,
            self.n_features,
        )
        # B head: e + (u +) σ_pack → n_pack. Built only if not shift-only.
        if not self.shift_only:
            in_dim_B = (
                self.d_emb
                + (self.n_features if self.B_uses_u else 0)
                + self.n_sigma_pack
            )
            self.B_head = self._make_mlp(
                in_dim_B, head_hidden, head_layers, activation,
                self.n_sigma_pack,
            )
        else:
            self.B_head = None
        # C head: e + u + σ_pack → F · n_pack. Built only when both
        # detach flags are on.
        if self.has_C:
            in_dim_C = self.d_emb + self.n_features + self.n_sigma_pack
            self.C_head = self._make_mlp(
                in_dim_C, head_hidden, head_layers, activation,
                self.n_features * self.n_sigma_pack,
            )
        else:
            self.C_head = None

        # Init the final layers near zero so log r ≈ 0 at start.
        with torch.no_grad():
            for h in (self.A_head, self.B_head, self.C_head):
                if h is None:
                    continue
                h[-1].weight.mul_(0.01)
                h[-1].bias.zero_()

    @staticmethod
    def _make_mlp(in_dim, hidden, n_layers, activation, out_dim):
        layers = []
        prev = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(prev, hidden))
            layers.append(activation())
            prev = hidden
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def trunk_forward(self, y, c):
        return self.trunk(torch.cat([y, c], dim=-1))

    def head_forward_components(
        self, e, u, sigma_pack,
    ):
        """Return ``(pure_u_d, pure_s_d, cross_d)`` for the factored
        head, each of shape ``[B]``. ``cross_d`` is ``None`` when there
        is no separate cross head. The caller can apply per-mode
        ``.detach()`` to the components before summing — used by the
        loss code for ``--detach-pure-{shift,smear}-in-joint`` gradient
        routing.
        """
        # A head: e + u (+ σ_pack)
        A_inputs = [e, u]
        if self.A_uses_sigma:
            A_inputs.append(sigma_pack)
        A = self.A_head(torch.cat(A_inputs, dim=-1))     # [..., F]
        pure_u_d = (u * A).sum(dim=-1)                    # [...]

        # B head: e + (u +) σ_pack
        if self.B_head is not None:
            B_inputs = [e]
            if self.B_uses_u:
                B_inputs.append(u)
            B_inputs.append(sigma_pack)
            B = self.B_head(torch.cat(B_inputs, dim=-1))  # [..., n_pack]
            pure_s_d = (sigma_pack * B).sum(dim=-1)
        else:
            pure_s_d = torch.zeros_like(pure_u_d)

        # C head: e + u + σ_pack
        cross_d = None
        if self.C_head is not None:
            C_flat = self.C_head(
                torch.cat([e, u, sigma_pack], dim=-1),
            )                                              # [..., F·n_pack]
            C = C_flat.view(
                *u.shape[:-1], self.n_features, self.n_sigma_pack,
            )
            cross_d = torch.einsum(
                "...i,...ij,...j->...", u, C, sigma_pack,
            )
        return pure_u_d, pure_s_d, cross_d

    def head_forward(self, e, u, sigma_pack):
        """Pre-positivity ``d = log W`` (structurally zero at the
        origin). No dual-forward subtraction is needed because the
        factored construction already enforces ``d(e, 0, 0) = 0`` and
        ``d(e, u, σ) = d(e, u, −σ)``.
        """
        pu, ps, cr = self.head_forward_components(e, u, sigma_pack)
        d = pu + ps
        if cr is not None:
            d = d + cr
        return d


# ============================================================================
# Architecture: polyhead mode (wraps the existing PolyHead)
# ============================================================================

class ReweightPolyhead(nn.Module):
    """Self-contained polyhead trunk: ``(y, c) → joint_coefs ∈ ℝ^{n_basis}``.

    The MLP's final ``Linear`` layer is sized to the number of basis
    monomials and produces the polynomial coefficients directly. The
    polynomial in ``(u, σ_vec)`` is evaluated externally via
    :func:`evaluate_joint` at any query point.

    Structural priors (no constant term, even in ``σ_vec``, capped
    degrees in u/σ/cross) are baked into the basis-index construction
    via :func:`_joint_indices` and reflected in ``self.n_basis``.

    No ``PolyHead`` wrapping — the trunk MLP IS the polyhead. We don't
    carry the flow-trainer's pure-u/pure-σ/cross masks or
    Gauss-Hermite buffers because the LSIF/DV training loop here
    doesn't use them (no stochastic K=1 smear target to vary across
    nodes — LSIF is unbiased with one ε per event by sample-source
    construction).
    """

    def __init__(
        self,
        n_features: int,
        n_cond: int,
        trunk_hidden: int = 64,
        trunk_layers: int = 2,
        max_deg_u: int = 3,
        max_deg_sigma: int = 4,
        max_cross_deg: int = 3,
        activation=nn.GELU,
        basis: str = "monomial",
        basis_scale_u: float = 1.0,
        basis_scale_sigma: float = 1.0,
        gauss_baseline: nn.Module = None,
    ):
        super().__init__()
        if max_deg_sigma % 2 != 0:
            raise ValueError(
                f"--max-deg-sigma must be even (smear-symmetry); got "
                f"{max_deg_sigma}"
            )
        if basis not in ("monomial", "chebyshev"):
            raise ValueError(f"unknown basis {basis!r}")
        self.n_features = int(n_features)
        self.n_cond = int(n_cond)
        self.max_deg_u = int(max_deg_u)
        self.max_deg_sigma = int(max_deg_sigma)
        self.max_cross_deg = int(max_cross_deg)
        self.basis = str(basis)
        self.basis_scale_u = float(basis_scale_u)
        self.basis_scale_sigma = float(basis_scale_sigma)
        # Optional analytic Gaussian baseline (additive log-r term).
        # When set, :func:`compute_d_quadrature` adds its closed-form
        # log-density-ratio to the polyhead's pre-positivity ``d`` so
        # the polynomial only fits the deviation from a Gaussian.
        self.gauss_baseline = gauss_baseline

        # Polynomial basis indices: (α, β) tuples enforcing structural
        # priors (no constant, even in σ, degree caps). Reordered so
        # that pure-u indices come first, pure-σ second, cross last —
        # this lets the three-head split below produce contiguous
        # coefficient blocks that align with the basis ordering.
        raw_indices = _joint_indices(
            n_features, max_deg_u, max_deg_sigma, max_cross_deg,
        )
        idx_pure_u = [
            (a, b) for (a, b) in raw_indices
            if len(b) == 0 and len(a) > 0
        ]
        idx_pure_s = [
            (a, b) for (a, b) in raw_indices
            if len(a) == 0 and len(b) > 0
        ]
        idx_cross = [
            (a, b) for (a, b) in raw_indices
            if len(a) > 0 and len(b) > 0
        ]
        assert (
            len(idx_pure_u) + len(idx_pure_s) + len(idx_cross)
            == len(raw_indices)
        ), "non-exhaustive split of joint indices"
        self._joint_indices = idx_pure_u + idx_pure_s + idx_cross
        self.n_basis = len(self._joint_indices)
        self._n_pure_u = len(idx_pure_u)
        self._n_pure_s = len(idx_pure_s)
        self._n_cross = len(idx_cross)

        # Boolean masks over (reordered) basis indices: pure-u
        # (β = ()), pure-σ (α = ()), and cross (both nonempty). Now
        # contiguous blocks: [0, n_pure_u), [n_pure_u, n_pure_u +
        # n_pure_s), [n_pure_u + n_pure_s, n_basis).
        is_pure_u = torch.tensor(
            [len(b) == 0 for (_, b) in self._joint_indices],
            dtype=torch.bool,
        )
        is_pure_sigma = torch.tensor(
            [len(a) == 0 for (a, _) in self._joint_indices],
            dtype=torch.bool,
        )
        self.register_buffer("is_pure_u", is_pure_u, persistent=False)
        self.register_buffer("is_pure_sigma", is_pure_sigma, persistent=False)
        self.register_buffer(
            "is_pure", is_pure_u | is_pure_sigma, persistent=False,
        )

        # Basis-evaluation auxiliary tensors as buffers — auto-moved
        # with ``.to(device)``, seen as stable inputs by torch.compile
        # and CUDA graphs. Without these (i.e., relying on the
        # runtime cache), the cache-miss path inside the compiled
        # forward creates ephemeral tensors that CUDA graphs
        # captures, and subsequent calls fail with "tensor output
        # overwritten by subsequent run".
        _aux = _build_basis_aux(self._joint_indices, n_features)
        self.register_buffer(
            "_basis_alpha_degs", _aux["alpha_degs"], persistent=False,
        )
        self.register_buffer(
            "_basis_beta_degs", _aux["beta_degs"], persistent=False,
        )
        self.register_buffer(
            "_basis_cheb_u_const",
            _aux["cheb_u_const"], persistent=False,
        )
        self.register_buffer(
            "_basis_cheb_v_const",
            _aux["cheb_v_const"], persistent=False,
        )
        self._basis_max_deg_u = _aux["max_deg_u"]
        self._basis_max_deg_sigma = _aux["max_deg_sigma"]

        # Trunk MLP: (y, c) → embedding ∈ R^trunk_hidden. The final
        # coefficient layer is split into three sub-heads (pure_u,
        # pure_σ, cross) — same total parameter count as a single
        # ``Linear(trunk_hidden, n_basis)``, but enables mode-
        # conditional skipping of irrelevant heads at forward time
        # (e.g. SHIFT events skip pure_σ and cross since σ=0 makes
        # those terms vanish in the polynomial). Combined with the
        # stratified mode sampler (sample_perturbations producing
        # exactly B/3 of each mode), the per-step cost becomes
        # B · d_emb · (n_pure_u + n_pure_σ + n_basis) / 3 instead of
        # B · d_emb · n_basis — typically ~2.5× speedup on the final
        # layer with no expressivity loss.
        layers = []
        prev = self.n_features + self.n_cond
        for _ in range(trunk_layers):
            layers.append(nn.Linear(prev, trunk_hidden))
            layers.append(activation())
            prev = trunk_hidden
        self.trunk = nn.Sequential(*layers)
        # Three sub-heads with sizes summing to n_basis. ``Linear`` is
        # only created if the corresponding block is non-empty (e.g.
        # in shift-only mode, pure_s and cross are empty).
        self.head_pure_u = (
            nn.Linear(prev, self._n_pure_u)
            if self._n_pure_u > 0 else None
        )
        self.head_pure_sigma = (
            nn.Linear(prev, self._n_pure_s)
            if self._n_pure_s > 0 else None
        )
        self.head_cross = (
            nn.Linear(prev, self._n_cross)
            if self._n_cross > 0 else None
        )
        # Init heads near zero so coefs ≈ 0 → log r ≈ 0 at start.
        with torch.no_grad():
            for head in (self.head_pure_u, self.head_pure_sigma, self.head_cross):
                if head is not None:
                    head.weight.mul_(0.01)
                    head.bias.zero_()

    @property
    def joint_indices(self):
        return self._joint_indices

    @property
    def basis_aux(self):
        """Dict of pre-registered basis-evaluation buffers, suitable
        for passing to :func:`evaluate_joint` as ``basis_aux=`` from
        compiled call sites — avoids in-graph tensor creation that
        breaks CUDA graphs (mode='reduce-overhead')."""
        return {
            "alpha_degs": self._basis_alpha_degs,
            "beta_degs": self._basis_beta_degs,
            "cheb_u_const": self._basis_cheb_u_const,
            "cheb_v_const": self._basis_cheb_v_const,
            "max_deg_u": self._basis_max_deg_u,
            "max_deg_sigma": self._basis_max_deg_sigma,
        }

    def trunk_forward(
        self,
        y: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the trunk embedding ``e ∈ R^{B, trunk_hidden}``.

        Used by the compiled training path in
        :func:`compute_d_quadrature`, which calls the per-head
        sub-modules (``head_pure_u``, etc.) directly on per-mode
        slices of ``e`` to avoid materializing the full
        ``[B, n_basis]`` coefficient tensor.
        """
        return self.trunk(torch.cat([y, c], dim=-1))

    def forward(
        self,
        y: torch.Tensor,
        c: torch.Tensor,
        mode_id: torch.Tensor = None,
    ) -> torch.Tensor:
        """Return the full ``[B, n_basis]`` coefficient tensor.

        Used by external callers (inference, diagnostics, ONNX
        export) that consume coefs as the polyhead's output. The
        compiled training path bypasses this in favor of a direct
        per-mode contraction (see :func:`compute_d_quadrature`),
        which avoids the [B, n_basis] intermediate altogether.

        ``mode_id`` is accepted but ignored — all heads run on all
        events here for the simple all-mode return.
        """
        del mode_id  # legacy kwarg, ignored
        e = self.trunk(torch.cat([y, c], dim=-1))
        parts = []
        if self.head_pure_u is not None:
            parts.append(self.head_pure_u(e))
        if self.head_pure_sigma is not None:
            parts.append(self.head_pure_sigma(e))
        if self.head_cross is not None:
            parts.append(self.head_cross(e))
        if not parts:
            return torch.zeros(
                e.shape[0], 0, device=e.device, dtype=e.dtype,
            )
        return torch.cat(parts, dim=-1)


# ============================================================================
# Analytic Gaussian baseline
# ============================================================================

class GaussBaseline(nn.Module):
    """Per-event analytic Gaussian baseline for the density-ratio.

    Models the nominal residual distribution conditional on gen-level
    kinematics as

        y | c  ~  N(μ(c), Σ(c)),    Σ(c) = L(c) L(c)ᵀ

    with μ ∈ ℝ^F and L ∈ ℝ^{F×F} lower-triangular with strictly positive
    diagonal (Cholesky factor of Σ). Both μ and L are produced by a
    small MLP from ``c`` only — they describe the *shape* of the
    nominal distribution, which is a property of the conditioning, not
    of any specific sample drawn from it (using y_nom here would give
    a circular, non-density-ratio object).

    The closed-form log-density-ratio for a shifted+convolved Gaussian
    (computed in :func:`gauss_baseline_log_r`) is then added as an
    additive term to the polyhead/mlp pre-positivity ``d``. The
    polyhead residual only has to fit the *deviation from a Gaussian*
    — which is much smoother and lower-degree than the full
    convolution-induced reweight, especially at moderate σ where the
    smearing kernel width is comparable to the core width of the
    target.

    Init produces μ = 0 and L = I (so Σ = I) so that ``log r̂_Gauss = 0``
    at start and the model recovers the legacy "no-baseline"
    behaviour at epoch 0; the baseline parameters then learn jointly
    with the polyhead residual.
    """

    def __init__(
        self,
        n_features: int,
        n_cond: int,
        hidden: int = 64,
        n_layers: int = 2,
        activation=nn.GELU,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.n_cond = int(n_cond)
        nf = self.n_features
        n_chol = nf * (nf + 1) // 2
        self.n_chol = int(n_chol)
        self.hidden = int(hidden)
        self.n_layers = int(n_layers)

        # MLP: c → (μ, raw_chol). Output dim = nf + n_chol.
        layers = []
        prev = self.n_cond
        for _ in range(self.n_layers):
            layers.append(nn.Linear(prev, self.hidden))
            layers.append(activation())
            prev = self.hidden
        layers.append(nn.Linear(prev, nf + n_chol))
        self.net = nn.Sequential(*layers)

        # Init: μ = 0, post-softplus diag(L) = 1 ⇒ L = I, Σ = I.
        # softplus(b) = 1 ⇒ b = log(e − 1).
        with torch.no_grad():
            last = self.net[-1]
            last.weight.zero_()
            last.bias.zero_()
            offset = 0
            for r in range(nf):
                for col in range(r + 1):
                    if r == col:
                        last.bias[nf + offset] = math.log(math.e - 1.0)
                    offset += 1

        # Static scatter projection: row k of ``raw_chol`` lands at
        # ``(rows[k], cols[k])`` in the [nf, nf] matrix (row-major
        # lower triangular ordering). Implemented as a constant
        # [n_chol, nf*nf] matrix so the unpack is a fixed matmul —
        # compile-friendly and CUDA-graph-safe.
        chol_proj = torch.zeros(n_chol, nf * nf)
        offset = 0
        for r in range(nf):
            for col in range(r + 1):
                chol_proj[offset, r * nf + col] = 1.0
                offset += 1
        self.register_buffer("_chol_proj", chol_proj, persistent=False)

    def forward(self, c: torch.Tensor):
        """``c: [..., n_cond] → (μ: [..., nf], L: [..., nf, nf])``.

        ``L`` is lower-triangular with strictly-positive diagonal
        (softplus-wrapped). Caller should treat the returned tensors
        as fp32-equivalent precision regardless of input dtype — the
        downstream :func:`gauss_baseline_log_r` upcasts internally.
        """
        out = self.net(c)
        nf = self.n_features
        mu = out[..., :nf]
        raw_chol = out[..., nf:]
        # Scatter ``raw_chol`` into a [..., nf, nf] lower-triangular
        # via a fixed projection, then softplus the diagonal.
        L_flat = raw_chol @ self._chol_proj           # [..., nf*nf]
        L_pre = L_flat.view(*raw_chol.shape[:-1], nf, nf)
        diag_pre = torch.diagonal(L_pre, dim1=-2, dim2=-1)
        diag_post = F.softplus(diag_pre)
        L = L_pre + torch.diag_embed(diag_post - diag_pre)
        return mu, L


def gauss_baseline_log_r(
    y: torch.Tensor,
    mu: torch.Tensor,
    L: torch.Tensor,
    u: torch.Tensor,
    sigma_vec: torch.Tensor,
) -> torch.Tensor:
    """Closed-form log density-ratio for a shifted+convolved Gaussian.

    With ``p_nom(y | c) = N(μ, Σ)``, ``Σ = L Lᵀ`` and
    ``p_pert(y | c, u, σ) = N(μ + u, Σ + diag(σ²))``, the analytic
    log-ratio at evaluation point ``y`` is

        log r̂_Gauss = -½ log[det(Σ + D_σ) / det Σ]
                       - ½ (y - μ - u)ᵀ (Σ + D_σ)⁻¹ (y - μ - u)
                       + ½ (y - μ)ᵀ Σ⁻¹ (y - μ).

    All inputs broadcast to a common leading shape ``[...]``; ``y``,
    ``mu``, ``u``, ``sigma_vec`` are ``[..., F]``; ``L`` is
    ``[..., F, F]`` lower-triangular. Returns ``[...]``.

    Computed in fp32 internally — the F=3 Cholesky / triangular solves
    are fp16-fragile near the unit-Σ region we initialise into, and
    the cost is negligible. Returns log r̂ in the input dtype.
    """
    in_dtype = y.dtype
    y32 = y.float()
    mu32 = mu.float()
    L32 = L.float()
    u32 = u.float()
    sigma32 = sigma_vec.float()

    # log det Σ from diag(L).
    diag_L = torch.diagonal(L32, dim1=-2, dim2=-1)
    log_det_Sigma = 2.0 * torch.log(diag_L).sum(-1)

    # Σ + D_σ via L Lᵀ + diag(σ²); take Cholesky for downstream solve
    # and log-det.
    Sigma = L32 @ L32.transpose(-1, -2)
    Sigma_pert = Sigma + torch.diag_embed(sigma32 * sigma32)
    L_pert = torch.linalg.cholesky(Sigma_pert)
    diag_Lp = torch.diagonal(L_pert, dim1=-2, dim2=-1)
    log_det_Sigma_pert = 2.0 * torch.log(diag_Lp).sum(-1)

    log_det_term = -0.5 * (log_det_Sigma_pert - log_det_Sigma)

    # Quadratic forms via triangular solves (single-RHS).
    r = y32 - mu32                                     # [..., F]
    r_pert = r - u32
    z_nom = torch.linalg.solve_triangular(
        L32, r.unsqueeze(-1), upper=False,
    ).squeeze(-1)
    quad_nom = (z_nom * z_nom).sum(-1)
    z_pert = torch.linalg.solve_triangular(
        L_pert, r_pert.unsqueeze(-1), upper=False,
    ).squeeze(-1)
    quad_pert = (z_pert * z_pert).sum(-1)

    log_r = log_det_term - 0.5 * quad_pert + 0.5 * quad_nom
    return log_r.to(in_dtype)


# ============================================================================
# Positivity wrap (shared)
# ============================================================================

def _apply_positivity(d: torch.Tensor, positivity: str,
                      clamp: float = _LOG_W_CLAMP) -> torch.Tensor:
    """Apply the positivity wrap to a pre-positivity scalar ``d`` to
    produce ``log r``. Same shape as input, applied elementwise.

      * ``"exp"``      — log r = clamp(d, ±clamp). Identity in the
        bulk; `r̂ = exp(log r̂)` is positive via `exp`. Symmetric
        in log r-space (odd in d). Hard cutoff at ±clamp prevents
        fp32 overflow.
      * ``"softplus"`` — log r = log(softplus(d)) − log(log 2).
        Asymmetric: compresses positive log r logarithmically,
        leaves negative log r ≈ d. `r̂ = softplus(d)/log 2 > 0`.
        No input clamp on d: PyTorch's ``F.softplus`` returns ``d``
        directly for d > threshold (default 20), so large positive
        d is handled without overflow; very negative d underflows
        ``softplus`` to 0, which is caught by ``clamp_min`` on the
        softplus output (sized to the dtype's smallest normal so
        the same wrap stays safe under fp32 / bf16 / fp16 autocast).
      * ``"asinh"``    — log r = asinh(d) = log(d + √(d² + 1)).
        Symmetric (odd in d) like `"exp"`, identity for |d| ≪ 1
        (so the Gaussian-likelihood-shaped bulk is preserved at
        leading order), and *smoothly* compresses to ±log(2|d|)
        for |d| → ∞. `r̂ = exp(asinh(d)) = d + √(d² + 1)`, which
        is strictly positive for every finite d (`√(d²+1) > |d|`)
        and grows only linearly in d — so no clamp is needed.
    """
    if positivity == "exp":
        return d.clamp(min=-clamp, max=clamp)
    if positivity == "softplus":
        tiny = torch.finfo(d.dtype).tiny
        return torch.log(F.softplus(d).clamp_min(tiny)) - _LOG_LOG2
    if positivity == "asinh":
        return torch.asinh(d)
    raise ValueError(f"unknown positivity {positivity!r}")


def _apply_positivity_r(d: torch.Tensor, positivity: str,
                        clamp: float = _LOG_W_CLAMP) -> torch.Tensor:
    """Return ``r̂(d)`` *directly*, without going through
    ``log r̂`` and exponentiating.

    Used by the LSIF path: the LSIF loss needs ``r̂`` (and ``r̂²``)
    not ``log r̂``, so going through the wrap and back via
    ``exp(log r̂)`` would be wasted work. The closed forms also avoid
    the softplus-output ``clamp_min`` (no outer ``log`` is taken
    here).

      * ``"exp"``      — r̂ = exp(clamp(d, ±clamp)). ``exp`` is
        intrinsic to this wrap; the ``clamp(±30)`` is what keeps fp32
        safe (``exp(30) ≈ 10¹³`` so ``r̂² ≈ 10²⁶`` ≪ fp32 max).
      * ``"softplus"`` — r̂ = softplus(d) / log 2. Strictly positive
        for every finite d (softplus > 0 mathematically; underflows
        gracefully to 0 in fp32 for d ≪ −87, which is a valid
        contribution of zero to LSIF). No clamp needed.
      * ``"asinh"``    — r̂ = d + √(d² + 1). Closed form, strictly
        positive for every finite d (``√(d²+1) > |d|``), grows only
        linearly in d. No clamp, no exp, no log.
    """
    if positivity == "exp":
        return torch.exp(d.clamp(min=-clamp, max=clamp))
    if positivity == "softplus":
        return F.softplus(d) / math.log(2.0)
    if positivity == "asinh":
        return d + torch.sqrt(d * d + 1.0)
    raise ValueError(f"unknown positivity {positivity!r}")


# ============================================================================
# compute_log_r_pair: branches on architecture
# ============================================================================

def compute_d_quadrature(
    model: nn.Module,
    arch: str,
    y_nom: torch.Tensor,
    y_pert_stack: torch.Tensor,
    c: torch.Tensor,
    u: torch.Tensor,
    sigma_vec: torch.Tensor,
    sigma_pack: torch.Tensor,
    mode_id: torch.Tensor = None,
    detach_pure_in_joint: bool = False,
):
    """Compute the **pre-positivity scalar** ``d`` at the nominal
    sample and at multiple ``y_pert`` quadrature positions.

    No positivity wrap is applied here — the caller picks ``log r̂`` or
    ``r̂`` (via :func:`_apply_positivity` or :func:`_apply_positivity_r`)
    depending on what the loss family needs. The LSIF path uses ``r̂``
    directly, avoiding the log/exp round-trip and the softplus output
    clamp; DV/BCE/expKL use ``log r̂`` as before.

    Args:
      ``y_pert_stack``: shape ``[n_eps, B, n_features]`` — the
        perturbed positions (one per ε quadrature node, plus optionally
        a residual sample). For K=1 stochastic, ``n_eps = 1``; for
        K-node Gauss-Hermite, ``n_eps = K``; for GH+residual,
        ``n_eps = K + 1`` with the last slot being the random sample.
      ``mode_id``: shape ``[B]`` long tensor with values 0 (SHIFT),
        1 (SMEAR), 2 (JOINT). Required when ``detach_pure_in_joint``
        is True.
      ``detach_pure_in_joint``: if True, the pure-u (σ → 0) and
        pure-σ (u → 0) contributions to the pre-positivity scalar are
        ``.detach()``-ed for events with ``mode_id == 2`` (JOINT). The
        nominal slot ``d_nom`` (with u, σ = 0) is unaffected.

          * ``polyhead``: implemented via per-event coef masking
            (zero-cost) — pure-u/pure-σ basis-index slots are detached
            for JOINT events.
          * ``mlp``: implemented via two extra head forwards
            (``f(e, u, 0)`` and ``f(e, 0, σ_pack)``) so that the
            decomposition ``d_full = d_pure_u + d_pure_σ + d_mixed`` is
            available; pure parts are detached only on JOINT events
            (≈ 2× head cost; trunk cost unchanged).

    Returns ``(d_nom, d_pert_all)`` with shapes ``[B]`` and
    ``[n_eps, B]``.

    Single batched pass: ``(n_eps + 1)·B`` trunk forwards, with the
    head dual call on top of that for the ``mlp`` arch.
    """
    n_eps, B, n_features = y_pert_stack.shape
    if detach_pure_in_joint and mode_id is None:
        raise ValueError(
            "detach_pure_in_joint=True requires mode_id to be passed",
        )

    if arch == "mlp":
        # Trunk [(n_eps+1)·B]: y_nom plus all perturbed positions.
        y_all = torch.cat(
            [y_nom, y_pert_stack.reshape(n_eps * B, n_features)], dim=0,
        )
        c_all = c.repeat(n_eps + 1, 1)
        e = model.trunk_forward(y_all, c_all)         # [(n_eps+1)·B, d_emb]

        u_all = u.repeat(n_eps + 1, 1)
        sigma_pack_all = sigma_pack.repeat(n_eps + 1, 1)
        u_zero = torch.zeros_like(u_all)
        sp_zero = torch.zeros_like(sigma_pack_all)

        if not detach_pure_in_joint:
            # Head dual: [2·(n_eps+1)·B] at (e, u, σ_pack) and (e, 0, 0).
            e_dual = torch.cat([e, e], dim=0)
            u_dual = torch.cat([u_all, u_zero], dim=0)
            sp_dual = torch.cat([sigma_pack_all, sp_zero], dim=0)
            f = model.head_forward(e_dual, u_dual, sp_dual)
            n_total = (n_eps + 1) * B
            d = f[:n_total] - f[n_total:]             # [(n_eps+1)·B]
        else:
            # 4 head queries per (event, ε-slot):
            #   f_full=f(e,u,σ_pack), f_zero=f(e,0,0),
            #   f_us =f(e,u,0),       f_sm =f(e,0,σ_pack)
            # Decompose into pure-u, pure-σ, and mixed contributions:
            #   d_full   = f_full - f_zero
            #   d_pure_u = f_us   - f_zero
            #   d_pure_s = f_sm   - f_zero
            #   d_mixed  = d_full - d_pure_u - d_pure_s
            # On JOINT events (mode==2) detach d_pure_u and d_pure_s so
            # only d_mixed back-propagates; otherwise d_used = d_full.
            e_q = torch.cat([e, e, e, e], dim=0)
            u_q = torch.cat([u_all, u_zero, u_all, u_zero], dim=0)
            sp_q = torch.cat(
                [sigma_pack_all, sp_zero, sp_zero, sigma_pack_all], dim=0,
            )
            f = model.head_forward(e_q, u_q, sp_q)
            n_total = (n_eps + 1) * B
            f_full = f[0 * n_total : 1 * n_total]
            f_zero = f[1 * n_total : 2 * n_total]
            f_us   = f[2 * n_total : 3 * n_total]
            f_sm   = f[3 * n_total : 4 * n_total]
            d_full = f_full - f_zero
            d_pure_u = f_us - f_zero
            d_pure_s = f_sm - f_zero
            # is_joint per event, broadcast across all (n_eps+1) blocks.
            is_joint_b = (mode_id == 2)
            is_joint_all = is_joint_b.unsqueeze(0).expand(
                n_eps + 1, B,
            ).reshape(-1)
            # Replace d_pure_u/d_pure_s with detach() on JOINT events.
            d_pure_u_used = torch.where(
                is_joint_all, d_pure_u.detach(), d_pure_u,
            )
            d_pure_s_used = torch.where(
                is_joint_all, d_pure_s.detach(), d_pure_s,
            )
            d_mixed = d_full - d_pure_u - d_pure_s
            d = d_pure_u_used + d_pure_s_used + d_mixed

    elif arch == "mlp-factored":
        # Structurally factored MLP head: log r = ⟨u, A⟩ + ⟨σ_pack, B⟩
        # [+ ⟨u⊗σ_pack, C⟩]. ``d`` comes from head_forward_components
        # directly (no dual-forward subtraction); the per-event
        # detach for --detach-pure-{shift,smear}-in-joint is applied
        # on JOINT-mode events.
        y_all = torch.cat(
            [y_nom, y_pert_stack.reshape(n_eps * B, n_features)], dim=0,
        )
        c_all = c.repeat(n_eps + 1, 1)
        u_all = u.repeat(n_eps + 1, 1)
        sigma_pack_all = sigma_pack.repeat(n_eps + 1, 1)
        e = model.trunk_forward(y_all, c_all)        # [(n_eps+1)·B, d_emb]
        pu, ps, cr = model.head_forward_components(
            e, u_all, sigma_pack_all,
        )
        if (
            mode_id is not None
            and (
                model.detach_pure_shift_in_joint
                or model.detach_pure_smear_in_joint
            )
        ):
            is_joint_b = (mode_id == 2)
            is_joint_all = (
                is_joint_b.unsqueeze(0)
                .expand(n_eps + 1, B)
                .reshape(-1)
            )
            if model.detach_pure_shift_in_joint:
                pu = torch.where(is_joint_all, pu.detach(), pu)
            if model.detach_pure_smear_in_joint:
                ps = torch.where(is_joint_all, ps.detach(), ps)
        d = pu + ps
        if cr is not None:
            d = d + cr

    elif arch == "polyhead":
        # Per-mode Order-2 contraction.
        #
        # The events come pre-sorted by mode_id (sample_perturbations
        # produces ``[0…0, 1…1, 2…2]`` contiguous; the (y, c) order
        # is randomized by the data loader, so the (event, mode)
        # pairing is still random). With sorted mode the per-mode
        # batches are contiguous slices of the B axis — no boolean
        # indexing, all shapes static, CUDA-graph-friendly.
        #
        # The trunk runs once on the full ``(n_eps+1)·B`` batch (one
        # large GEMM, well-utilized). The per-head matmuls then
        # operate on per-mode slices, exploiting the basis-vanishing
        # property: SHIFT events have σ=0 → only pure_u contributes,
        # SMEAR events have u=0 → only pure_σ contributes, JOINT
        # events have all three. Crucially, we **never materialize
        # the [(n_eps+1)·B, n_basis] coefs tensor** — d is computed
        # via Order-2 contraction
        #
        #     d_block = (e_block · (φ_block @ W^T)) + (φ_block @ b)
        #
        # whose intermediate is ``[N_block, d_emb]``, far cheaper in
        # bandwidth than ``[N_block, n_out]``.
        n_blocks = n_eps + 1
        # Mode-block boundaries — derived from B (deterministic given
        # the stratified sampler). Same convention as sample_perturbations.
        if model.head_pure_sigma is None and model.head_cross is None:
            # Pure shift-only polyhead: all events are SHIFT mode.
            n_shift_b, n_smear_b, n_joint_b = B, 0, 0
        else:
            base = B // 3
            extra = B - 3 * base
            n_shift_b, n_smear_b, n_joint_b = base + extra, base, base

        # Trunk: one matmul on the full (n_eps+1)·B batch.
        y_all = torch.cat(
            [y_nom, y_pert_stack.reshape(n_eps * B, n_features)], dim=0,
        )
        c_all = c.repeat(n_eps + 1, 1)
        u_all = u.repeat(n_eps + 1, 1)
        sigma_vec_all = sigma_vec.repeat(n_eps + 1, 1)
        e_flat = model.trunk_forward(y_all, c_all)     # [(n_eps+1)·B, d_emb]
        d_emb = e_flat.shape[-1]
        e_r = e_flat.view(n_blocks, B, d_emb)

        # Basis: phi on the same batch, then reshape and slice by
        # mode along the B axis. ``basis_aux=model.basis_aux`` keeps
        # the cache-miss path out of the compiled trace (avoids
        # CUDA-graph "tensor overwritten" errors).
        phi_flat = _select_basis(
            model.basis, u_all, sigma_vec_all, model.joint_indices,
            scale_u=model.basis_scale_u,
            scale_sigma=model.basis_scale_sigma,
            aux=model.basis_aux,
        )                                              # [(n_eps+1)·B, n_basis]
        phi_r = phi_flat.view(n_blocks, B, -1)

        n_pu = model._n_pure_u
        n_ps = model._n_pure_s
        n_cr = model._n_cross

        def _order2(e_blk, phi_blk, head):
            """Order-2: ``d = (e · (phi @ W)) + (phi @ b)`` per event.
            ``e_blk``: [N, d_emb]; ``phi_blk``: [N, n_out];
            ``head``: nn.Linear(d_emb, n_out). Returns [N]."""
            if head is None:
                return torch.zeros(
                    e_blk.shape[0], device=e_blk.device,
                    dtype=e_blk.dtype,
                )
            temp = phi_blk @ head.weight             # [N, d_emb]
            bias_term = phi_blk @ head.bias          # [N]
            return (e_blk * temp).sum(-1) + bias_term

        d_chunks = []

        # SHIFT events: only pure_u contributes (σ = 0).
        if n_shift_b > 0:
            e_s = e_r[:, :n_shift_b, :].reshape(-1, d_emb)
            phi_s_pu = phi_r[:, :n_shift_b, :n_pu].reshape(-1, n_pu)
            d_s = _order2(e_s, phi_s_pu, model.head_pure_u)
            d_chunks.append(d_s.view(n_blocks, n_shift_b))

        # SMEAR events: only pure_σ contributes (u = 0).
        if n_smear_b > 0:
            e_m = e_r[
                :, n_shift_b : n_shift_b + n_smear_b, :,
            ].reshape(-1, d_emb)
            phi_m_ps = phi_r[
                :, n_shift_b : n_shift_b + n_smear_b,
                n_pu : n_pu + n_ps,
            ].reshape(-1, n_ps)
            d_m = _order2(e_m, phi_m_ps, model.head_pure_sigma)
            d_chunks.append(d_m.view(n_blocks, n_smear_b))

        # JOINT events: all three head contributions; optionally
        # detach the pure-u and pure-σ contributions so JOINT-mode
        # gradient flows only through head_cross.
        if n_joint_b > 0:
            j0 = n_shift_b + n_smear_b
            e_j = e_r[:, j0:, :].reshape(-1, d_emb)
            phi_j = phi_r[:, j0:, :].reshape(-1, n_pu + n_ps + n_cr)
            d_j_pu = _order2(
                e_j, phi_j[:, :n_pu], model.head_pure_u,
            )
            d_j_ps = _order2(
                e_j, phi_j[:, n_pu : n_pu + n_ps],
                model.head_pure_sigma,
            )
            d_j_cr = _order2(
                e_j, phi_j[:, n_pu + n_ps :], model.head_cross,
            )
            if detach_pure_in_joint:
                d_j_pu = d_j_pu.detach()
                d_j_ps = d_j_ps.detach()
            d_j = (d_j_pu + d_j_ps + d_j_cr).view(n_blocks, n_joint_b)
            d_chunks.append(d_j)

        # Concatenate per-mode contributions along the B axis and
        # flatten back to [(n_eps+1)·B] for the d_nom / d_pert split.
        d = torch.cat(d_chunks, dim=1).reshape(-1)

    else:
        raise ValueError(f"unknown arch {arch!r}")

    # Optional analytic Gaussian baseline: closed-form additive log-r
    # term modelling p_nom(y|c) ~ N(μ(c), Σ(c)) and its shift+convolution
    # under (u, σ_vec). The polyhead/mlp residual then only has to fit
    # the *deviation from a Gaussian* — much smoother and lower-degree
    # than the full convolution-induced reweight, especially at
    # moderate σ where the smearing kernel width is comparable to the
    # core width of the target. ``μ, Σ`` depend only on ``c`` (they
    # describe the *shape* of p_nom, which is a property of the
    # conditional density, not of any single sample drawn from it).
    gauss = getattr(model, "gauss_baseline", None)
    if gauss is not None:
        n_blocks = n_eps + 1
        y_all_g = torch.cat(
            [y_nom, y_pert_stack.reshape(n_eps * B, n_features)], dim=0,
        )
        mu_g, L_g = gauss(c)                            # [B, F], [B, F, F]
        mu_all_g = mu_g.repeat(n_blocks, 1)
        L_all_g = L_g.repeat(n_blocks, 1, 1)
        u_all_g = u.repeat(n_blocks, 1)
        sigma_all_g = sigma_vec.repeat(n_blocks, 1)
        d = d + gauss_baseline_log_r(
            y_all_g, mu_all_g, L_all_g, u_all_g, sigma_all_g,
        )

    d_nom = d[:B]
    d_pert_all = d[B:].view(n_eps, B)
    return d_nom, d_pert_all


class ReweightWrapper(nn.Module):
    """DDP-friendly wrapper that exposes
    :func:`compute_d_quadrature` as a single ``forward()``.

    Holds the inner model (``ReweightMLP_B`` or ``ReweightPolyhead``)
    plus the static knobs (``arch``, ``detach_pure_in_joint``). All
    trunk/head accesses live inside one nn.Module forward, which is
    what DDP needs to instrument the parameter-gradient hooks.

    Returns the **pre-positivity scalar** ``(d_nom, d_pert_all)``;
    the loss step applies the appropriate wrap (``log r̂`` or ``r̂``,
    selected by loss family). The wrap therefore lives outside DDP /
    ``torch.compile`` boundaries — this lets the LSIF path use the
    closed-form ``r̂(d)`` directly without an exp/log round-trip.

    The perturbation sampling, ε-stack assembly, σ-pack packing, and
    final loss combination remain in :func:`loss_step` so the
    quadrature scheme stays orthogonal to the wrapper.
    """

    def __init__(
        self,
        model: nn.Module,
        arch: str,
        positivity: str,
        detach_pure_in_joint: bool = True,
    ):
        super().__init__()
        self.model = model
        self.arch = str(arch)
        # ``positivity`` is recorded for the checkpoint payload but
        # not used by the wrapper's forward — the wrap is applied
        # downstream in loss_step.
        self.positivity = str(positivity)
        self.detach_pure_in_joint = bool(detach_pure_in_joint)

    def forward(
        self,
        y_nom: torch.Tensor,
        y_pert_stack: torch.Tensor,
        c: torch.Tensor,
        u: torch.Tensor,
        sigma_vec: torch.Tensor,
        sigma_pack: torch.Tensor,
        mode_id: torch.Tensor,
    ):
        return compute_d_quadrature(
            self.model, self.arch,
            y_nom, y_pert_stack, c, u, sigma_vec, sigma_pack,
            mode_id=mode_id,
            detach_pure_in_joint=self.detach_pure_in_joint,
        )


def _build_eps_stack(
    batch_size: int,
    smear_K: int,
    smear_residual: bool,
    gh_nodes: torch.Tensor,
    device,
):
    """Build the per-event ε stack used to form ``y_pert_stack``.

    Returns shape ``[n_eps, B]`` with:

      * K=1 stochastic (default): ``n_eps = 1`` with one random
        ``ε ~ N(0, 1)`` per event.
      * Pure GH (K≥2, no residual): ``n_eps = K`` with the K
        Gauss-Hermite nodes broadcast across the batch.
      * GH + residual (K≥2, residual): ``n_eps = K + 1`` where the
        last slot is one extra random ε per event for the
        control-variate residual.
    """
    if smear_K <= 1:
        return torch.randn(1, batch_size, device=device)
    eps_gh = gh_nodes.view(-1, 1).expand(smear_K, batch_size)  # [K, B]
    if smear_residual:
        eps_extra = torch.randn(1, batch_size, device=device)
        return torch.cat([eps_gh, eps_extra], dim=0)
    return eps_gh


def _make_y_pert_stack(y_nom, u, sigma_vec, eps_stack):
    """``y_pert[k, b] = y_nom[b] + u[b] + ε_stack[k, b] · σ_vec[b]``."""
    return (
        y_nom.unsqueeze(0)
        + u.unsqueeze(0)
        + eps_stack.unsqueeze(-1) * sigma_vec.unsqueeze(0)
    )


def _smear_pert_estimator(
    x_pert_all,
    eps_stack,
    gh_nodes,
    gh_weights,
    smear_K: int,
    smear_residual: bool,
):
    """Loss-agnostic ε-quadrature aggregator.

    Takes a precomputed ``[n_eps, B]`` per-event scalar ``x_pert_all``
    — the loss-family-specific integrand evaluated at each
    quadrature ε-node — and returns the per-event ``E_ε[x]`` estimator.
    The caller (``loss_step``) prepares ``x_pert_all`` from
    ``d_pert_all`` via the appropriate combination of positivity wrap
    and per-loss transform:

      * LSIF   → ``x = r̂(d) = _apply_positivity_r(d, positivity)``
      * DV     → ``x = log r̂(d) = _apply_positivity(d, positivity)``
      * expKL  → ``x = log r̂(d)`` (same as DV)
      * BCE    → ``x = softplus(−log r̂(d))``

    Three quadrature regimes (matching :func:`_build_eps_stack`):

      * K=1 stochastic: just the single sample, unbiased per event.
      * Pure GH: ``S_K = Σ_k w_k · x_k``. Polynomial-exact through
        degree ``2K − 1`` in ε; truncation error ``O(σ_max^{2K})``.
      * GH + residual: ``S_K + (x(ε~) − g(ε~))``, where ``g`` is the
        Lagrange interpolant through the GH nodes. ``E_ε[g] = S_K``
        exactly, so the residual term has zero mean and the estimator
        stays unbiased; variance is ``Var(x − g)`` which captures only
        beyond-truncation features.

    Because ``x`` is already in the loss family's natural form, the
    aggregation here is purely linear in ``x`` — the same code path
    serves all four losses.
    """
    if smear_K <= 1:
        return x_pert_all[0]

    K = smear_K
    x_K = x_pert_all[:K]                             # [K, B]
    S_K = torch.einsum("k,kb->b", gh_weights, x_K)
    if not smear_residual:
        return S_K

    # Hybrid: control-variate correction.
    eps_extra = eps_stack[K]                         # [B]
    L = _lagrange_basis_at(eps_extra, gh_nodes)      # [B, K]
    g_at_eps = torch.einsum("kb,bk->b", x_K, L)
    x_extra = x_pert_all[K]
    return S_K + (x_extra - g_at_eps)


# ============================================================================
# Sampling: u, σ_vec, ε_smear
# ============================================================================

def sample_perturbations(
    batch_size: int,
    n_dim: int,
    delta_max: float,
    sigma_max: float,
    device,
    oversample: float = 1.3,
    include_smear: bool = True,
):
    """Per-event sampler returning ``(u, σ_vec, mode_id)``.

    With ``include_smear=True`` (shift+smear training) it produces a
    stratified SHIFT / SMEAR / JOINT mode mix (1/3 each), matching
    the polyhead training convention from
    ``train_muon_response_flow.py``:

      * SHIFT (mode=0): u random, σ_vec = 0.
      * SMEAR (mode=1): u = 0, σ_vec random.
      * JOINT (mode=2): both u and σ_vec random.

    Stratification ensures pure-axis queries at deployment
    (``u = 0`` or ``σ_vec = 0``) are in-distribution.

    With ``include_smear=False`` (shift-only training; the default) we
    skip smear/joint modes entirely: every event is mode=0 with a
    random ``u`` and ``σ_vec = 0``. This is what we want when the
    architecture itself is shift-only.

    Magnitude sampling within each active axis:
      * ``|u|`` uniform on ``[−oversample·delta_max, +oversample·delta_max]``
      * ``|σ_vec|`` uniform on ``[0, oversample·sigma_max]``

    Direction is uniform on the unit sphere ``S^{n_dim − 1}``. The
    ε samples for the smear noise are produced separately by
    :func:`_build_eps_stack` so the K=1 stochastic and the K≥2
    Gauss-Hermite paths share this sampler.

    The ``oversample`` factor (default 1.3) trains the model slightly
    beyond the operational range so closure at the edge is
    well-supported.
    """
    half = oversample * delta_max

    # Raw u-direction × magnitude (always populated).
    delta_u = (torch.rand(batch_size, device=device) * 2.0 - 1.0) * half
    v_u = torch.randn(batch_size, n_dim, device=device)
    v_u = v_u / v_u.norm(dim=-1, keepdim=True).clamp_min(1e-30)

    if not include_smear:
        # All-SHIFT path: σ_vec ≡ 0, mode ≡ 0.
        u = delta_u.unsqueeze(-1) * v_u
        sigma_vec = torch.zeros(batch_size, n_dim, device=device)
        mode_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        return u, sigma_vec, mode_id

    half_sig = oversample * sigma_max

    # Raw σ-direction × magnitude (always populated; gated below).
    sigma_mag = torch.rand(batch_size, device=device) * half_sig
    v_s = torch.randn(batch_size, n_dim, device=device)
    v_s = v_s / v_s.norm(dim=-1, keepdim=True).clamp_min(1e-30)

    # Stratified mode: 0=SHIFT, 1=SMEAR, 2=JOINT — exactly 1/3 each
    # (deterministic split). Mode IDs are produced in **sorted
    # contiguous** order ``[0, 0, ..., 1, 1, ..., 2, 2, ...]``; the
    # data-loader's own per-event shuffling already randomizes the
    # ``(y, c)`` ↔ mode pairing, so the explicit permutation that
    # used to live here is unnecessary and would break the contiguous
    # per-mode slicing the compiled polyhead path relies on (see
    # :func:`compute_d_quadrature`'s polyhead branch). The remainder
    # when ``batch_size`` isn't a multiple of 3 is given to SHIFT,
    # the mode that dominates the gradient signal.
    base = batch_size // 3
    extra = batch_size - 3 * base
    mode_id = torch.cat([
        torch.full((base + extra,), 0, device=device, dtype=torch.long),
        torch.full((base,), 1, device=device, dtype=torch.long),
        torch.full((base,), 2, device=device, dtype=torch.long),
    ])
    shift_active = (mode_id != 1).to(delta_u.dtype)   # SHIFT or JOINT
    smear_active = (mode_id != 0).to(sigma_mag.dtype)  # SMEAR or JOINT

    u = (delta_u * shift_active).unsqueeze(-1) * v_u
    sigma_vec = (sigma_mag * smear_active).unsqueeze(-1) * v_s
    # ε is sampled separately by ``_build_eps_stack`` (per-event for
    # K=1 stochastic, GH nodes for K≥2). When ``smear_active = 0`` the
    # σ_vec is zero so the resulting ε·σ_vec is zero regardless.
    return u, sigma_vec, mode_id


# ============================================================================
# Loss functions
# ============================================================================

def dv_loss(log_r_nom, pert_T_estimator, weights):
    """Donsker-Varadhan / MINE objective (minimization form):

        L = log E_{y~p_nom}[exp(T)] − pert_T_estimator,   T = log r̂.

    The pert side accepts an externally-supplied estimator of
    ``E_{y~p_pert}[T]`` so that K=1 stochastic, pure GH, or GH+residual
    schemes plug in uniformly. Numerically stable via ``logsumexp``."""
    wsum = weights.sum().clamp_min(1e-30)
    e_pert_T = (weights * pert_T_estimator).sum() / wsum
    log_w = torch.log(weights.clamp_min(1e-30))
    log_e_nom_expT = (
        torch.logsumexp(log_w + log_r_nom, dim=0) - torch.log(wsum)
    )
    return -e_pert_T + log_e_nom_expT


def lsif_loss(r_nom, pert_estimator, weights):
    """Least-squares importance fitting:

        L = E_{y~p_nom}[r̂²]  −  2·pert_estimator
                                ⇑
                           per-event ``E_ε[r̂]`` from K=1 stochastic,
                           pure GH, or GH+residual.

    Unique minimizer ``r̂* = r``. Unbiased for any of the three
    pert-estimator schemes.

    ``r_nom`` is fed in directly (not via ``exp(log r̂_nom)``) — the
    caller computes ``r̂`` from the pre-positivity scalar ``d`` via the
    closed-form :func:`_apply_positivity_r`. This avoids the log/exp
    round-trip and lets the softplus wrap skip its output ``clamp_min``
    on the LSIF path (``r̂ = softplus(d)/log 2`` is intrinsically
    positive)."""
    wsum = weights.sum().clamp_min(1e-30)
    e_nom_r_sq = (weights * r_nom * r_nom).sum() / wsum
    e_pert_r = (weights * pert_estimator).sum() / wsum
    return e_nom_r_sq - 2.0 * e_pert_r


def bce_loss(log_r_nom, pert_softplus_neg_estimator, weights):
    """Logistic / binary cross-entropy density-ratio loss (CARL):

        L = E_{y~p_nom}[ softplus( s) ]
          + E_{y~p_pert}[ softplus(−s) ]            with s = log r̂.

    Unique minimizer ``s* = log r`` exactly — no constant ambiguity
    (unlike DV) and per-sample unbiased (no Jensen gap). Bounded
    gradients (sigmoid-saturated) make it stable when ``r̂`` strays
    far from ``r`` early in training, at the cost of a slower
    convergence rate when ``|log r|`` is large.

    The pert side accepts an externally-supplied estimator of
    ``E_{y~p_pert}[softplus(−s)]`` so that K=1 stochastic, pure GH,
    or GH+residual all plug in uniformly (same machinery as LSIF/DV)."""
    wsum = weights.sum().clamp_min(1e-30)
    e_nom_term = (weights * F.softplus(log_r_nom)).sum() / wsum
    e_pert_term = (
        weights * pert_softplus_neg_estimator
    ).sum() / wsum
    return e_nom_term + e_pert_term


def expkl_loss(log_r_nom, pert_T_estimator, weights):
    """Exponential-KL loss (DV with the outer log dropped, a.k.a.
    f-GAN-KL):

        L = E_{y~p_nom}[ exp(s) ]  −  E_{y~p_pert}[ s ]   with s = log r̂.

    Unique minimizer ``s* = log r`` exactly (no constant ambiguity)
    and per-sample unbiased (no Jensen gap). Locally near the optimum
    the loss is

        L(s) − L(s*)  ≈  ½ · E_{y~p_pert}[ (s − log r)² ]

    so this is the L²-in-log-r analog of LSIF (which is L²-in-r-space
    weighted by ``p_nom``). Trade-off: like LSIF the gradient on the
    nom side grows as ``exp(s)``, so it can blow up early in training
    if the model strays far from ``log r ≈ 0``. Construction-B's
    structural ``s = 0`` at zero perturbation keeps init in the safe
    regime; for high-contrast deployment edges, gradient clipping or
    a BCE→expKL warm-start helps.

    The pert side accepts an externally-supplied estimator of
    ``E_{y~p_pert}[s]`` (= ``E_pert[T]`` in DV notation), so K=1
    stochastic, pure GH, or GH+residual all plug in uniformly."""
    wsum = weights.sum().clamp_min(1e-30)
    e_nom = (weights * torch.exp(log_r_nom)).sum() / wsum
    e_pert = (weights * pert_T_estimator).sum() / wsum
    return e_nom - e_pert


# ============================================================================
# Training step
# ============================================================================

def _per_mode_split(per_event_loss, weights, mode_id, n_modes=3):
    """Aggregate per-event loss into per-mode (SHIFT/SMEAR/JOINT)
    weighted means.

    Returns ``(weighted_sum, weight_sum)`` both shape ``[n_modes]``,
    on the same device/dtype as ``per_event_loss``. The caller divides
    after the (cross-rank, cross-batch) reduction. Modes with zero
    population have ``weight_sum == 0`` and the caller should treat
    that mode as 'no data this epoch'.
    """
    device = per_event_loss.device
    dtype = per_event_loss.dtype
    out_sum = torch.zeros(n_modes, device=device, dtype=dtype)
    out_w = torch.zeros(n_modes, device=device, dtype=dtype)
    pe_w = (weights * per_event_loss).to(dtype)
    for m in range(n_modes):
        mask = (mode_id == m).to(dtype)
        out_sum[m] = (pe_w * mask).sum()
        out_w[m] = (weights * mask).sum()
    return out_sum, out_w


def _shift_K_per_event_loss(
    inner_model, arch, y, c, w,
    u_smolyak, K_extra,
    delta_max, oversample,
    loss_fn, positivity,
    sigma_pack_iu=None, sigma_pack_ju=None,
):
    """Memory-optimized K-per-event shift loss (shift-only mode).

    The naive batch-replication strategy (replicate every event K
    times before the wrapper forward) duplicates the *trunk*
    computation and its activations K times per event for the y_nom
    side, which is wasteful: every event's nominal (y, c) is the
    same across all K shift evaluations. This helper bypasses the
    wrapper and computes:

      * 1 trunk forward at (y, c)             — shared across K
      * K trunk forwards at (y + u_k, c)      — one per perturbation
      * polynomial / head evaluations at K (u, σ=0) per event

    Total trunk work per event: ``1 + K_total`` forwards (vs.
    ``2·K_total`` in the naive path). Trunk-activation memory
    likewise drops by ~2×, which is the dominant memory savings at
    K_total ≳ 10.

    Returns ``(loss, split_sum, split_w)`` matching the regular
    :func:`loss_step` contract.
    """
    B = y.shape[0]
    n_features = y.shape[-1]
    device = y.device
    dtype = y.dtype

    K_smolyak = u_smolyak.shape[0] if u_smolyak is not None else 0
    K_total = K_smolyak + int(K_extra)
    if K_total == 0:
        raise ValueError("K_total must be > 0 in shift-K path")
    half = float(oversample) * float(delta_max)

    # Build u_all [B, K_total, n_features]: Smolyak shared across
    # events (broadcast view) + K_extra stochastic per-event draws.
    u_parts = []
    if K_smolyak > 0:
        u_parts.append(
            u_smolyak.to(device=device, dtype=dtype)
            .unsqueeze(0).expand(B, K_smolyak, n_features)
        )
    if K_extra > 0:
        delta = (
            torch.rand(B, K_extra, device=device, dtype=dtype) * 2.0 - 1.0
        ) * half
        v = torch.randn(B, K_extra, n_features, device=device, dtype=dtype)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        u_parts.append(delta.unsqueeze(-1) * v)
    u_all = torch.cat(u_parts, dim=1)        # [B, K_total, n_features]

    sigma_zero = torch.zeros_like(u_all)     # [B, K_total, n_features]
    n_cond = c.shape[-1]

    if arch == "polyhead":
        # 1 trunk forward at y_nom — coefs shared across all K_total.
        coefs_nom = inner_model(y, c)        # [B, n_basis]

        # K trunk forwards at y_pert_k = y + u_k.
        y_pert = y.unsqueeze(1) + u_all      # [B, K_total, n_features]
        y_pert_flat = y_pert.reshape(B * K_total, n_features)
        c_flat = c.unsqueeze(1).expand(
            B, K_total, n_cond,
        ).reshape(B * K_total, n_cond)
        coefs_pert_flat = inner_model(
            y_pert_flat, c_flat,
        )                                    # [B·K_total, n_basis]
        coefs_pert = coefs_pert_flat.view(B, K_total, -1)

        # Polynomial evaluations (no trunk gradient flow):
        # coefs_nom expand is a broadcast *view*, no copy. Gradient
        # accumulates over K_total contributions automatically.
        coefs_nom_view = coefs_nom.unsqueeze(1).expand(B, K_total, -1)
        _basis_aux = inner_model.basis_aux
        d_nom_3d = evaluate_joint(
            coefs_nom_view, u_all, sigma_zero, inner_model.joint_indices,
            basis=inner_model.basis,
            scale_u=inner_model.basis_scale_u,
            scale_sigma=inner_model.basis_scale_sigma,
            basis_aux=_basis_aux,
        )                                    # [B, K_total]
        d_pert_3d = evaluate_joint(
            coefs_pert, u_all, sigma_zero, inner_model.joint_indices,
            basis=inner_model.basis,
            scale_u=inner_model.basis_scale_u,
            scale_sigma=inner_model.basis_scale_sigma,
            basis_aux=_basis_aux,
        )                                    # [B, K_total]
    elif arch in ("mlp", "mlp-factored"):
        # Both flavours use head_forward(e, u, σ_pack). The factored
        # head's head_forward already returns ``d`` (structurally zero
        # at the origin), so ``f(e, u, 0) − f(e, 0, 0) = d − 0 = d`` —
        # the dual-forward subtraction is a structural no-op for the
        # factored variant in shift-only mode. Either way the
        # per-event scalar comes out identical to the construction
        # the loss expects.
        e_nom = inner_model.trunk_forward(y, c)      # [B, d_emb]
        d_emb = e_nom.shape[-1]

        y_pert_flat = (y.unsqueeze(1) + u_all).reshape(B * K_total, n_features)
        c_flat = c.unsqueeze(1).expand(
            B, K_total, n_cond,
        ).reshape(B * K_total, n_cond)
        e_pert_flat = inner_model.trunk_forward(
            y_pert_flat, c_flat,
        )                                    # [B·K_total, d_emb]

        # Head: f(e, u, σ_pack=0) and f(e, 0, 0). Shift-only ⇒ σ_pack
        # placeholder is [..., 0]. Aggregate the four head forwards
        # per (event, k) needed by the structural construction.
        u_flat = u_all.reshape(B * K_total, n_features)
        sp_zero = torch.empty(
            B * K_total, 0, device=device, dtype=dtype,
        )
        u_zero = torch.zeros_like(u_flat)
        # d_nom: f(e_nom, u_k, 0) - f(e_nom, 0, 0).
        # e_nom needs to be expanded to [B·K_total, d_emb] (view).
        e_nom_exp = e_nom.unsqueeze(1).expand(B, K_total, d_emb).reshape(
            B * K_total, d_emb,
        )
        f_nom_full = inner_model.head_forward(e_nom_exp, u_flat, sp_zero)
        f_nom_zero = inner_model.head_forward(e_nom_exp, u_zero, sp_zero)
        d_nom_3d = (f_nom_full - f_nom_zero).view(B, K_total)
        # d_pert: f(e_pert, u_k, 0) - f(e_pert, 0, 0).
        f_pert_full = inner_model.head_forward(e_pert_flat, u_flat, sp_zero)
        f_pert_zero = inner_model.head_forward(e_pert_flat, u_zero, sp_zero)
        d_pert_3d = (f_pert_full - f_pert_zero).view(B, K_total)
    else:
        raise ValueError(f"unknown arch {arch!r} in shift-K path")

    # Optional analytic Gaussian baseline (mirrors the
    # compute_d_quadrature path). Same μ(c), L(c) for both nominal
    # and perturbed evaluations — what differs is the y at which the
    # baseline is evaluated. Broadcast across K_total via expand.
    gauss = getattr(inner_model, "gauss_baseline", None)
    if gauss is not None:
        mu_g, L_g = gauss(c)                                # [B, F], [B, F, F]
        mu_bk = mu_g.unsqueeze(1).expand(B, K_total, n_features)
        L_bk = L_g.unsqueeze(1).expand(
            B, K_total, n_features, n_features,
        )
        y_bk = y.unsqueeze(1).expand(B, K_total, n_features)
        y_pert_bk = y.unsqueeze(1) + u_all                  # [B, K_total, F]
        d_nom_3d = d_nom_3d + gauss_baseline_log_r(
            y_bk, mu_bk, L_bk, u_all, sigma_zero,
        )
        d_pert_3d = d_pert_3d + gauss_baseline_log_r(
            y_pert_bk, mu_bk, L_bk, u_all, sigma_zero,
        )

    # Flatten K into the batch axis for the loss aggregation:
    # B' = B·K_total, weight per replica = w / K_total so per-event
    # aggregate weight equals the original w.
    d_nom_flat = d_nom_3d.reshape(B * K_total)
    d_pert_all = d_pert_3d.reshape(B * K_total).unsqueeze(0)  # [1, B·K_total]
    w_flat = w.unsqueeze(1).expand(B, K_total).reshape(
        B * K_total,
    ) / float(K_total)
    eps_stack = torch.zeros(1, B * K_total, device=device, dtype=dtype)

    if loss_fn == "lsif":
        r_nom = _apply_positivity_r(d_nom_flat, positivity)
        r_pert_all = _apply_positivity_r(d_pert_all, positivity)
        pert_estimator = _smear_pert_estimator(
            r_pert_all, eps_stack, None, None, 1, False,
        )
        loss = lsif_loss(r_nom, pert_estimator, w_flat)
        per_event = (r_nom * r_nom - 2.0 * pert_estimator).detach()
    else:
        # BCE: disable the exp-wrap ±clamp safety net (functionally
        # unused for BCE, see _apply_positivity docstring); DV /
        # expKL keep the default clamp for nom-side ``exp(log r̂)``
        # / ``logsumexp(log r̂)`` numerical safety.
        pos_clamp = (
            float("inf") if loss_fn == "bce" else _LOG_W_CLAMP
        )
        log_r_nom = _apply_positivity(
            d_nom_flat, positivity, clamp=pos_clamp,
        )
        log_r_pert_all = _apply_positivity(
            d_pert_all, positivity, clamp=pos_clamp,
        )
        if loss_fn == "bce":
            x_pert_all = F.softplus(-log_r_pert_all)
        else:
            x_pert_all = log_r_pert_all
        pert_estimator = _smear_pert_estimator(
            x_pert_all, eps_stack, None, None, 1, False,
        )
        if loss_fn == "dv":
            loss = dv_loss(log_r_nom, pert_estimator, w_flat)
            per_event = (-pert_estimator).detach()
        elif loss_fn == "bce":
            loss = bce_loss(log_r_nom, pert_estimator, w_flat)
            per_event = (
                F.softplus(log_r_nom) + pert_estimator
            ).detach()
        else:  # expkl
            loss = expkl_loss(log_r_nom, pert_estimator, w_flat)
            per_event = (
                torch.exp(log_r_nom) - pert_estimator
            ).detach()

    # All replicas sit in SHIFT mode (=0); SMEAR/JOINT slots are
    # unpopulated and will return weight=0 → +inf in the caller's
    # split bookkeeping.
    mode_flat = torch.zeros(
        B * K_total, dtype=torch.long, device=device,
    )
    split_sum, split_w = _per_mode_split(per_event, w_flat, mode_flat)
    return loss, split_sum, split_w


def loss_step(
    model, arch, y, c, w,
    delta_max, sigma_max, oversample,
    loss_fn, positivity,
    sigma_pack_iu=None, sigma_pack_ju=None,
    smear_K: int = 1, smear_residual: bool = False,
    gh_nodes=None, gh_weights=None,
    detach_pure_in_joint: bool = True,
    include_smear: bool = True,
    shift_smolyak_pts=None, shift_stochastic_extra: int = 0,
    inner_model=None,
):
    """One forward-loss step.

    Samples ``(u, σ_vec)``, builds the ε-stack (``[n_eps, B]`` of
    quadrature points), forms ``y_pert_stack``, computes the
    pre-positivity scalar ``d`` at all positions in one batched pass,
    and combines into the chosen loss family.

    ``model`` is expected to be either a :class:`ReweightWrapper` or
    its DDP / ``torch.compile`` wrapping — its ``forward()`` returns
    the **pre-positivity** ``(d_nom, d_pert_all)``. The positivity wrap
    is applied here, *outside* the wrapper, in whichever form the loss
    family needs:

      * LSIF  — uses ``r̂`` directly via :func:`_apply_positivity_r`.
        No log/exp round-trip; softplus output ``clamp_min`` not
        needed on this path.
      * DV / BCE / expKL — use ``log r̂`` via :func:`_apply_positivity`.

    ``detach_pure_in_joint`` is baked into the wrapper (it acts on
    ``d`` inside ``compute_d_quadrature``); ``positivity`` is needed
    here because the wrap is applied at this level.

    ``include_smear=False`` (shift-only path) gates the perturbation
    sampler so every event is SHIFT mode and ``σ_vec = 0``; the
    ε-stack collapses to a single zero, so ``y_pert = y + u``.

    Shift-axis multi-sample-per-event (Smolyak + stochastic) is
    activated when ``shift_smolyak_pts is not None`` or
    ``shift_stochastic_extra > 0``. Each event is then evaluated at
    ``K_total = K_smolyak + K_extra`` u values, with weights
    ``w / K_total`` per replica so the per-event aggregate weight is
    preserved. Only valid in shift-only mode (caller is expected to
    have errored out on ``include_smear=True``); under that
    constraint ``mode_id ≡ SHIFT`` for every replica.
    """
    del detach_pure_in_joint  # baked into the wrapper
    n_features = y.shape[-1]
    B = y.shape[0]
    device = y.device

    use_shift_K = (
        not include_smear
        and (shift_smolyak_pts is not None or shift_stochastic_extra > 0)
    )

    if use_shift_K:
        # Memory-optimized path: trunk(y_nom, c) is computed *once*
        # per event and shared across all K_total perturbation
        # evaluations. Bypasses the wrapper / DDP / compile and goes
        # straight to ``inner_model``; gradients still flow correctly
        # because DDP's all-reduce is registered on parameters, not
        # on the wrapped forward call.
        if inner_model is None:
            raise ValueError(
                "shift-K path requires ``inner_model`` to be passed "
                "to loss_step (used to share the y_nom trunk forward)."
            )
        return _shift_K_per_event_loss(
            inner_model, arch, y, c, w,
            shift_smolyak_pts, int(shift_stochastic_extra),
            delta_max, oversample, loss_fn, positivity,
            sigma_pack_iu=sigma_pack_iu, sigma_pack_ju=sigma_pack_ju,
        )

    u, sigma_vec, mode_id = sample_perturbations(
        B, n_features, delta_max, sigma_max, device,
        oversample=oversample, include_smear=include_smear,
    )

    if include_smear:
        eps_stack = _build_eps_stack(
            B, smear_K, smear_residual, gh_nodes, device,
        )
    else:
        # σ_vec ≡ 0 in shift-only mode, so ε·σ_vec = 0 regardless of
        # ε. Skip the random draw and force a single zero-ε slot —
        # n_eps = 1 — so the downstream estimator just returns the
        # single perturbed value unchanged. We also override the
        # estimator regime to (smear_K=1, smear_residual=False) so
        # the GH/residual code paths don't fire on a zero σ.
        eps_stack = torch.zeros(1, B, device=device, dtype=y.dtype)
        smear_K = 1
        smear_residual = False
    y_pert_stack = _make_y_pert_stack(y, u, sigma_vec, eps_stack)

    if arch in ("mlp", "mlp-factored") and sigma_pack_iu is not None:
        sigma_pack = _pack_sigma_outer(
            sigma_vec, sigma_pack_iu, sigma_pack_ju,
        )
    else:
        # Either polyhead (head ignores Σ_pack) or shift-only mlp /
        # mlp-factored (head's σ-channel is gone). Pass a [B, 0]
        # placeholder so the forward signature stays stable for
        # DDP/compile.
        sigma_pack = torch.empty(B, 0, device=device, dtype=y.dtype)

    # Wrapper returns the pre-positivity scalar d (no log/exp wrap).
    d_nom, d_pert_all = model(
        y, y_pert_stack, c, u, sigma_vec, sigma_pack, mode_id,
    )

    # Apply the wrap in the form the loss family wants. The K=1 / GH /
    # GH+residual aggregation in `_smear_pert_estimator` is uniform;
    # the loss family only differs in *which* per-event scalar gets
    # aggregated.
    if loss_fn == "lsif":
        # LSIF wants r̂ directly: r̂² on the nom side, E_ε[r̂] on the
        # pert side. No log/exp round-trip; for the softplus wrap
        # this also skips the outer log(...) and its clamp_min.
        r_nom = _apply_positivity_r(d_nom, positivity)
        r_pert_all = _apply_positivity_r(d_pert_all, positivity)
        pert_estimator = _smear_pert_estimator(
            r_pert_all, eps_stack, gh_nodes, gh_weights,
            smear_K, smear_residual,
        )
        loss = lsif_loss(r_nom, pert_estimator, w)
        # Per-event integrand for per-mode split monitoring.
        # ``r_nom² − 2·pert_estimator`` matches the expectation form
        # of ``lsif_loss``; per-mode weighted means of this quantity
        # serve as the "is this mode improving" signal.
        per_event = (
            r_nom * r_nom - 2.0 * pert_estimator
        ).detach()
        split_sum, split_w = _per_mode_split(per_event, w, mode_id)
        return loss, split_sum, split_w

    # BCE doesn't exponentiate ``log r̂`` anywhere (only
    # ``softplus(±log r̂)``, which is intrinsically bounded), so the
    # ±_LOG_W_CLAMP safety clamp on the ``exp`` positivity wrap has
    # no functional role for BCE — disabling it via ``clamp=inf``
    # lets gradient flow even at extreme |d|. For DV / expKL the
    # nom-side is ``exp(log r̂)`` (or ``logsumexp``) which still
    # needs the clamp for fp32 / fp16 safety; keep the default
    # there.
    pos_clamp = float("inf") if loss_fn == "bce" else _LOG_W_CLAMP
    log_r_nom = _apply_positivity(d_nom, positivity, clamp=pos_clamp)
    log_r_pert_all = _apply_positivity(
        d_pert_all, positivity, clamp=pos_clamp,
    )
    if loss_fn in ("dv", "expkl"):
        x_pert_all = log_r_pert_all
    elif loss_fn == "bce":
        x_pert_all = F.softplus(-log_r_pert_all)
    else:
        raise ValueError(f"unknown loss_fn {loss_fn!r}")

    pert_estimator = _smear_pert_estimator(
        x_pert_all, eps_stack, gh_nodes, gh_weights,
        smear_K, smear_residual,
    )

    if loss_fn == "dv":
        loss = dv_loss(log_r_nom, pert_estimator, w)
        # DV's nom term is a global ``log E[exp(s)]`` (logsumexp) that
        # does not decompose per-mode cleanly. We track ``−pert_T`` as
        # the per-mode signal: ``pert_T_estimator`` is a per-event
        # estimate of ``E_{ε,pert}[T]``, and DV is decreasing iff this
        # is increasing. Sign-flip for the "lower is better" patience
        # convention.
        per_event = (-pert_estimator).detach()
    elif loss_fn == "bce":
        loss = bce_loss(log_r_nom, pert_estimator, w)
        # BCE's per-event integrand is the sum of nom-side and pert-
        # side softplus terms.
        per_event = (
            F.softplus(log_r_nom) + pert_estimator
        ).detach()
    else:  # expkl
        loss = expkl_loss(log_r_nom, pert_estimator, w)
        per_event = (
            torch.exp(log_r_nom) - pert_estimator
        ).detach()
    split_sum, split_w = _per_mode_split(per_event, w, mode_id)
    return loss, split_sum, split_w


# ============================================================================
# Training loop
# ============================================================================

def _all_reduce_sum_(t: torch.Tensor, is_dist: bool) -> torch.Tensor:
    """In-place all-reduce(sum) when distributed, otherwise a no-op."""
    if is_dist:
        import torch.distributed as dist
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def _amp_setup(precision: str, device):
    """Resolve precision string into (autocast dtype, autocast enabled,
    GradScaler). bf16 doesn't need loss scaling (fp32-equivalent
    exponent range); fp16 does. fp32 disables autocast entirely."""
    amp_device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    if precision == "fp32":
        amp_dtype, amp_enabled = torch.float32, False
    elif precision == "bf16":
        amp_dtype, amp_enabled = torch.bfloat16, True
    elif precision == "fp16":
        amp_dtype, amp_enabled = torch.float16, True
    else:
        raise ValueError(f"unknown precision {precision!r}")
    scaler = torch.amp.GradScaler(
        amp_device_type, enabled=(precision == "fp16"),
    )
    return amp_device_type, amp_dtype, amp_enabled, scaler


def train_one_epoch(
    model, optimizer, train_loader, device, epoch, args, arch,
    is_rank0, amp_ctx, scaler, sigma_pack_iu, sigma_pack_ju,
    gh_nodes, gh_weights, is_dist=False,
    shift_smolyak_pts=None, shift_stochastic_extra: int = 0,
    inner_model=None,
    step_profiler=None,
):
    model.train()
    postfix_every = 50
    bar = tqdm(
        train_loader,
        desc=f"epoch {epoch:03d} train",
        leave=False,
        dynamic_ncols=True,
        disable=not is_rank0,
        mininterval=1.0,
        miniters=postfix_every,
    )
    total_loss = torch.zeros((), device=device)
    wsum = torch.zeros((), device=device)
    # Per-mode split (SHIFT, SMEAR, JOINT). Tracked via separate
    # weighted-sum and weight-sum accumulators so we can divide
    # *after* the cross-rank reduction at the end of the epoch.
    split_sum_total = torch.zeros(3, device=device)
    split_w_total = torch.zeros(3, device=device)
    profile_steps = (
        0 if step_profiler is None else int(step_profiler.enabled) and int(
            getattr(args, "profile_data_pipeline", 0) or 0
        )
    )
    for i, (x, c, w) in enumerate(bar):
        do_profile = (step_profiler is not None
                      and step_profiler.enabled
                      and i < profile_steps)
        if do_profile:
            step_profiler.start()
        x = x.to(device, non_blocking=True)
        c = c.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)
        wb = w.sum()
        if do_profile:
            step_profiler.mark("h2d")
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx():
            loss, split_sum, split_w = loss_step(
                model, arch, x, c, w,
                delta_max=args.delta_max,
                sigma_max=args.sigma_max,
                oversample=args.oversample,
                loss_fn=args.loss_fn,
                positivity=args.positivity,
                sigma_pack_iu=sigma_pack_iu,
                sigma_pack_ju=sigma_pack_ju,
                smear_K=args.smear_K,
                smear_residual=args.smear_residual,
                gh_nodes=gh_nodes, gh_weights=gh_weights,
                detach_pure_in_joint=args.detach_pure_in_joint,
                include_smear=args.include_smear,
                shift_smolyak_pts=shift_smolyak_pts,
                shift_stochastic_extra=shift_stochastic_extra,
                inner_model=inner_model,
            )
        loss = loss.float()
        if do_profile:
            step_profiler.mark("forward")
        scaler.scale(loss).backward()
        if do_profile:
            step_profiler.mark("backward")
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        if do_profile:
            step_profiler.mark("optimizer")
        total_loss = total_loss + loss.detach() * wb
        wsum = wsum + wb
        split_sum_total = split_sum_total + split_sum.detach().float()
        split_w_total = split_w_total + split_w.detach().float()
        if do_profile:
            step_profiler.mark("accum")
            step_profiler.report(i)
        if is_rank0 and (i + 1) % postfix_every == 0:
            # Rank-0-local running metric (not all-reduced); refreshed
            # cheaply between batches. The epoch-boundary report below
            # is the globally-reduced number.
            wsum_now = max(wsum.item(), 1e-30)
            bar.set_postfix(loss=f"{(total_loss.item() / wsum_now):+.4f}")
    _all_reduce_sum_(total_loss, is_dist)
    _all_reduce_sum_(wsum, is_dist)
    _all_reduce_sum_(split_sum_total, is_dist)
    _all_reduce_sum_(split_w_total, is_dist)
    epoch_loss = total_loss.item() / max(wsum.item(), 1e-30)
    # Divide post-reduction; modes with zero population (e.g. SMEAR
    # / JOINT in shift-only training) get +inf so the patience
    # logic naturally treats them as "no data this epoch" and
    # neither resets nor blocks the patience clock on them.
    split = []
    for m in range(3):
        wm = split_w_total[m].item()
        if wm > 0.0:
            split.append(split_sum_total[m].item() / wm)
        else:
            split.append(float("inf"))
    return epoch_loss, split


@torch.no_grad()
def run_val(model, val_loader, device, args, arch, amp_ctx,
            sigma_pack_iu, sigma_pack_ju, gh_nodes, gh_weights,
            is_dist=False,
            shift_smolyak_pts=None, shift_stochastic_extra: int = 0,
            inner_model=None):
    model.eval()
    total_loss = torch.zeros((), device=device)
    wsum = torch.zeros((), device=device)
    split_sum_total = torch.zeros(3, device=device)
    split_w_total = torch.zeros(3, device=device)
    for x, c, w in val_loader:
        x = x.to(device, non_blocking=True)
        c = c.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)
        wb = w.sum()
        with amp_ctx():
            loss, split_sum, split_w = loss_step(
                model, arch, x, c, w,
                delta_max=args.delta_max,
                sigma_max=args.sigma_max,
                oversample=args.oversample,
                loss_fn=args.loss_fn,
                positivity=args.positivity,
                sigma_pack_iu=sigma_pack_iu,
                sigma_pack_ju=sigma_pack_ju,
                smear_K=args.smear_K,
                smear_residual=args.smear_residual,
                gh_nodes=gh_nodes, gh_weights=gh_weights,
                detach_pure_in_joint=args.detach_pure_in_joint,
                include_smear=args.include_smear,
                shift_smolyak_pts=shift_smolyak_pts,
                shift_stochastic_extra=shift_stochastic_extra,
                inner_model=inner_model,
            )
        loss = loss.float()
        total_loss = total_loss + loss * wb
        wsum = wsum + wb
        split_sum_total = split_sum_total + split_sum.detach().float()
        split_w_total = split_w_total + split_w.detach().float()
    _all_reduce_sum_(total_loss, is_dist)
    _all_reduce_sum_(wsum, is_dist)
    _all_reduce_sum_(split_sum_total, is_dist)
    _all_reduce_sum_(split_w_total, is_dist)
    val_loss = total_loss.item() / max(wsum.item(), 1e-30)
    split = []
    for m in range(3):
        wm = split_w_total[m].item()
        if wm > 0.0:
            split.append(split_sum_total[m].item() / wm)
        else:
            split.append(float("inf"))
    return val_loss, split


def train(
    model, inner_model, arch, train_loader, val_loader, device, args,
    log_lines, stats=None, sigma_pack_iu=None, sigma_pack_ju=None,
    gh_nodes=None, gh_weights=None,
    is_dist: bool = False, is_rank0: bool = True,
    shift_smolyak_pts=None, shift_stochastic_extra: int = 0,
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=max(1, args.patience // 2),
    )

    amp_device_type, amp_dtype, amp_enabled, scaler = _amp_setup(
        args.precision, device,
    )
    amp_ctx = lambda: torch.amp.autocast(
        device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled,
    )

    # Per-step profiler — pairs with the TimedLoader's per-batch
    # split (loader_wait + trainer_step) by further breaking the
    # trainer_step into H2D / forward / backward / optimizer / accum
    # so we can tell whether GPU compute, optimizer, or CPU
    # bookkeeping is the per-step ceiling.
    from arrow_shard_loader import StepProfiler
    step_profiler = StepProfiler(
        enabled=int(getattr(args, "profile_data_pipeline", 0) or 0) > 0
        and is_rank0,
        label="train_step",
        device=device,
    )

    best_val = float("inf")
    # Per-mode (SHIFT, SMEAR, JOINT) running bests. The patience
    # clock resets on improvement of the combined val *or* any of the
    # split components against its own running best — avoids
    # premature stopping when one component (e.g. smear) is noise-
    # limited while another (e.g. shift) is still descending. The
    # combined-metric ``best_val`` and the saved ``best_state``
    # snapshot remain gated on the combined ``improved`` so the
    # checkpoint is the best by the global metric.
    best_val_split = [float("inf"), float("inf"), float("inf")]
    best_state = None
    no_improve = 0
    checkpoint_path = (
        os.path.join(args.output, "checkpoint.pt") if is_rank0 else None
    )

    # Optional ``torch.profiler`` diagnostic pass: run the first
    # ``warmup + active`` training steps under the profiler, print
    # the top hot ops, and exit before the normal epoch loop. All
    # ranks execute the steps (so DDP collectives stay synchronized);
    # only rank 0 runs the profiler / prints / exports.
    if getattr(args, "profile", False):
        from train_muon_response_flow import _run_profile_pass
        n_warmup = int(args.profile_warmup)
        n_active = int(args.profile_active)
        n_total = n_warmup + n_active
        train_iter = iter(train_loader)

        def _profile_step(_i):
            nonlocal train_iter
            try:
                x, c, w = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, c, w = next(train_iter)
            x = x.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx():
                loss, _split_sum, _split_w = loss_step(
                    model, arch, x, c, w,
                    delta_max=args.delta_max,
                    sigma_max=args.sigma_max,
                    oversample=args.oversample,
                    loss_fn=args.loss_fn,
                    positivity=args.positivity,
                    sigma_pack_iu=sigma_pack_iu,
                    sigma_pack_ju=sigma_pack_ju,
                    smear_K=args.smear_K,
                    smear_residual=args.smear_residual,
                    gh_nodes=gh_nodes, gh_weights=gh_weights,
                    detach_pure_in_joint=args.detach_pure_in_joint,
                    include_smear=args.include_smear,
                    shift_smolyak_pts=shift_smolyak_pts,
                    shift_stochastic_extra=shift_stochastic_extra,
                    inner_model=inner_model,
                )
            loss = loss.float()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=10.0,
            )
            scaler.step(optimizer)
            scaler.update()

        if is_rank0:
            output = args.profile_output
            if is_dist and output:
                import torch.distributed as _dist
                rank_i = _dist.get_rank()
                base, ext = os.path.splitext(output)
                output = f"{base}.rank{rank_i}{ext}"
            _run_profile_pass(
                _profile_step,
                n_warmup=n_warmup,
                n_active=n_active,
                output_path=output,
                label=f" {arch}/{args.loss_fn}",
            )
        else:
            # Match step count on non-rank-0 for DDP sync; no profiler
            # overhead.
            for _i in range(n_total):
                _profile_step(_i)
        return float("nan")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_split = train_one_epoch(
            model, optimizer, train_loader, device, epoch, args, arch,
            is_rank0=is_rank0, amp_ctx=amp_ctx, scaler=scaler,
            sigma_pack_iu=sigma_pack_iu, sigma_pack_ju=sigma_pack_ju,
            gh_nodes=gh_nodes, gh_weights=gh_weights,
            is_dist=is_dist,
            shift_smolyak_pts=shift_smolyak_pts,
            shift_stochastic_extra=shift_stochastic_extra,
            inner_model=inner_model,
            step_profiler=step_profiler,
        )
        val_loss, val_split = run_val(
            model, val_loader, device, args, arch, amp_ctx,
            sigma_pack_iu, sigma_pack_ju, gh_nodes, gh_weights,
            is_dist=is_dist,
            shift_smolyak_pts=shift_smolyak_pts,
            shift_stochastic_extra=shift_stochastic_extra,
            inner_model=inner_model,
        )
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        # Format split components: show as "-" when no events of
        # that mode in the batch (e.g. SMEAR / JOINT in shift-only
        # training where val_split[1,2] = +inf).
        def _fmt_comp(x):
            return f"{x:+.4f}" if math.isfinite(x) else "  -   "
        tr_sh, tr_sm, tr_jt = train_split
        va_sh, va_sm, va_jt = val_split
        line = (
            f"epoch {epoch:03d}  train_{args.loss_fn} {train_loss:+.4f}  "
            f"(sh {_fmt_comp(tr_sh)} sm {_fmt_comp(tr_sm)} "
            f"jt {_fmt_comp(tr_jt)})  "
            f"val_{args.loss_fn} {val_loss:+.4f}  "
            f"(sh {_fmt_comp(va_sh)} sm {_fmt_comp(va_sm)} "
            f"jt {_fmt_comp(va_jt)})  "
            f"lr {lr_now:.2e}  dt {time.time()-t0:.1f}s"
        )
        if is_rank0:
            print(line, flush=True)
            log_lines.append(line)

        # All ranks see the same all-reduced val_loss, so best/early-
        # stopping decisions stay synchronous without an extra reduce.
        # Patience-improvement test:
        #   * BCE / DV / expKL — absolute threshold 1e-4 (val ~ O(1)).
        #   * LSIF (L²-in-r-space, MSE-style) — relative threshold
        #     ``patience_rel_threshold · |best_val|`` since the LSIF
        #     val sits near 0 / negative, making any absolute test
        #     scale-inappropriate.
        # On the first epoch ``best_val`` is +inf — any finite
        # val_loss counts as improvement.
        rel = float(args.patience_rel_threshold)

        def _is_improvement(curr, prev_best):
            if math.isinf(prev_best):
                return curr < prev_best
            if args.loss_fn == "lsif":
                return curr < prev_best - rel * abs(prev_best)
            return curr < prev_best - 1e-4

        improved = _is_improvement(val_loss, best_val)

        # Per-component improvement: any of (sh, sm, jt) improving
        # over its own running best resets the patience clock.
        component_improved = False
        for i, val_comp in enumerate(val_split):
            if not math.isfinite(val_comp):
                continue  # mode unpopulated this epoch
            if _is_improvement(val_comp, best_val_split[i]):
                best_val_split[i] = val_comp
                component_improved = True

        if improved:
            best_val = val_loss
            if is_rank0:
                best_state = _state_dict_to_cpu(inner_model)
        if improved or component_improved:
            no_improve = 0
        else:
            no_improve += 1

        if (
            is_rank0
            and args.checkpoint_every > 0
            and (epoch % args.checkpoint_every == 0)
        ):
            ckpt = _build_ckpt(
                model=inner_model, arch=arch, args=args, epoch=epoch,
                train_loss=train_loss, val_loss=val_loss,
                best_val=best_val, stats=stats,
            )
            torch.save(ckpt, checkpoint_path)

        if no_improve >= args.patience:
            if bool(getattr(args, "no_early_stop", False)):
                # Patience threshold reached but early stopping is
                # disabled — log a notice on the first crossing only
                # and continue. The LR scheduler clock is independent
                # and keeps decaying the LR as configured.
                if no_improve == args.patience:
                    line = (
                        f"would early-stop at epoch {epoch} "
                        f"(best val {best_val:+.4f}); "
                        f"--no-early-stop set, continuing"
                    )
                    if is_rank0:
                        print(line)
                        log_lines.append(line)
            else:
                line = (
                    f"early stopping at epoch {epoch} "
                    f"(best val {best_val:+.4f})"
                )
                if is_rank0:
                    print(line)
                    log_lines.append(line)
                break

    if is_rank0 and best_state is not None:
        inner_model.load_state_dict(best_state)
    return best_val


def _build_ckpt(model, arch, args, epoch, train_loss, val_loss, best_val,
                stats):
    """Bundle a checkpoint payload. ``model_config['arch']`` discriminates
    the construction so loaders can rebuild the right architecture."""
    gb = getattr(model, "gauss_baseline", None)
    gauss_baseline_cfg = (
        {
            "hidden": int(gb.hidden),
            "n_layers": int(gb.n_layers),
        }
        if gb is not None else None
    )
    if arch == "mlp":
        model_config = {
            "arch": "mlp",
            "n_features": int(model.n_features),
            "n_cond": int(model.n_cond),
            "d_emb": int(args.d_emb),
            "trunk_hidden": int(args.trunk_hidden),
            "trunk_layers": int(args.trunk_layers),
            "head_hidden": int(args.head_hidden),
            "head_layers": int(args.head_layers),
            "activation": args.activation,
            "shift_only": bool(getattr(model, "shift_only", False)),
            "gauss_baseline": gauss_baseline_cfg,
        }
    elif arch == "mlp-factored":
        model_config = {
            "arch": "mlp-factored",
            "n_features": int(model.n_features),
            "n_cond": int(model.n_cond),
            "d_emb": int(args.d_emb),
            "trunk_hidden": int(args.trunk_hidden),
            "trunk_layers": int(args.trunk_layers),
            "head_hidden": int(args.head_hidden),
            "head_layers": int(args.head_layers),
            "activation": args.activation,
            "shift_only": bool(getattr(model, "shift_only", False)),
            "detach_pure_shift_in_joint": bool(
                getattr(model, "detach_pure_shift_in_joint", False)
            ),
            "detach_pure_smear_in_joint": bool(
                getattr(model, "detach_pure_smear_in_joint", False)
            ),
            "gauss_baseline": gauss_baseline_cfg,
        }
    elif arch == "polyhead":
        model_config = {
            "arch": "polyhead",
            "n_features": int(model.n_features),
            "n_cond": int(model.n_cond),
            "trunk_hidden": int(args.trunk_hidden),
            "trunk_layers": int(args.trunk_layers),
            "max_deg_u": int(model.max_deg_u),
            "max_deg_sigma": int(model.max_deg_sigma),
            "max_cross_deg": int(model.max_cross_deg),
            "activation": args.activation,
            "basis": str(model.basis),
            "basis_scale_u": float(model.basis_scale_u),
            "basis_scale_sigma": float(model.basis_scale_sigma),
            "gauss_baseline": gauss_baseline_cfg,
        }
    else:
        raise ValueError(f"unknown arch {arch!r}")

    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val": best_val,
        "state_dict": _state_dict_to_cpu(model),
        "model_config": model_config,
        "train_config": {
            "loss_fn": args.loss_fn,
            "positivity": args.positivity,
            "delta_max": args.delta_max,
            "sigma_max": (
                args.sigma_max if args.include_smear else 0.0
            ),
            "oversample": args.oversample,
            "precision": args.precision,
            "compile": bool(args.compile),
            "include_smear": bool(args.include_smear),
            "smear_K": (
                int(args.smear_K) if args.include_smear else 1
            ),
            "smear_residual": (
                bool(args.smear_residual) if args.include_smear else False
            ),
            "detach_pure_in_joint": (
                bool(args.detach_pure_in_joint) if args.include_smear
                else False
            ),
            "detach_pure_shift_in_joint": (
                bool(args.detach_pure_shift_in_joint)
                if args.include_smear else False
            ),
            "detach_pure_smear_in_joint": (
                bool(args.detach_pure_smear_in_joint)
                if args.include_smear else False
            ),
        },
        "stats": asdict(stats) if stats is not None else None,
    }


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Direct shift+smear reweight training (MLP-B / "
        "polyhead, DV/LSIF, no flow).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data / IO.
    p.add_argument(
        "--input-files", nargs="+", required=True,
        default=argparse.SUPPRESS,
        help="(required) One of: (a) a shard directory containing "
        "manifest.json (auto-expanded to its Arrow IPC shards); "
        "(b) explicit .arrow shard files (shell globs OK); "
        "(c) explicit .root RVec RNTuple snapshots from "
        "flow_training_snapshot.py. All entries must be the same "
        "format.",
    )
    p.add_argument(
        "--tree", default="tree",
        help="TTree name inside the input files.",
    )
    p.add_argument(
        "--output", default="./shift_smear_reweight/",
        help="Output directory (created if missing).",
    )
    p.add_argument(
        "--stats-max-rows", type=int, default=-1,
        help="Cap the rows used by the preproc-stats warmup pass. -1 "
        "= scan all rows in all shards (most accurate). 1e7 is "
        "plenty in practice and starts training in seconds rather "
        "than minutes.",
    )
    p.add_argument(
        "--robust-stats",
        dest="robust_stats",
        action="store_true",
        default=True,
        help="Compute preproc location/scale as (median, 1.4826·MAD) "
        "from a row sample instead of (mean, std) from a full scan. "
        "Heavy-tailed targets (notably r_kappa with its "
        "charge-mismeasurement peak at ~-2, and the heavy tails on "
        "dlambda / dphi) make the unweighted std much larger than "
        "the bulk's actual width, which then makes every δ=1 "
        "perturbation a multi-σ over-shift. The robust pair keeps "
        "δ=1 ~ one bulk-width. Default: on. Use ``--no-robust-stats`` "
        "to fall back to the old (mean, std) full-scan pass.",
    )
    p.add_argument(
        "--no-robust-stats",
        dest="robust_stats",
        action="store_false",
    )
    p.add_argument(
        "--robust-sample-rows", type=int, default=20_000_000,
        help="(``--robust-stats`` only) Sample size for the robust "
        "median + MAD pass. 2e7 rows fits in ~640 MB and gives "
        "stable estimates even for the rare charge-mismeasurement "
        "tail.",
    )
    p.add_argument(
        "--weight-handling",
        choices=["abs", "keep", "drop"],
        default="abs",
        help="How to handle MC@NLO-style signed event weights. "
        "``abs`` (default): take |w| and drop w==0 / non-finite — "
        "loses the destructive-interference signal but keeps every "
        "row's magnitude contribution. ``keep``: pass w through "
        "unchanged; only drops non-finite. Use for an unbiased "
        "signed weighted-NLL. ``drop``: drop w<=0 / non-finite "
        "(legacy, ~5%% loss on W/Z).",
    )
    p.add_argument(
        "--data-workers", type=int, default=4,
        help="Number of background processes that feed each rank's "
        "data pipeline. 0 = run the loader inline in the trainer "
        "process (single producer thread, prefetch=2). >0 wraps the "
        "loader in a ``torch.utils.data.DataLoader`` with this many "
        "worker processes — each opens its own shard subset, "
        "bypasses the GIL, and ships pinned tensors via shared "
        "memory. Useful when training is CPU-bound on data prep.",
    )
    p.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        default=None,
        help="Pin worker output tensors so ``Tensor.to(device, "
        "non_blocking=True)`` can DMA asynchronously. Default: "
        "auto-enabled when device is CUDA. ``--no-pin-memory`` "
        "disables it — useful when the single PyTorch pin-memory "
        "thread in the main process is a bottleneck (high "
        "``--data-workers`` but each worker sits at low CPU "
        "utilisation); H2D copies become synchronous but the "
        "main-process pin memcpy is skipped.",
    )
    p.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
    )
    p.add_argument(
        "--profile-data-pipeline",
        type=int,
        default=0,
        metavar="N",
        help="Print per-iteration ``loader_wait`` / ``trainer_step`` "
        "timings for the first N training batches of each epoch. "
        "Use this to localise the data-pipeline bottleneck: large "
        "loader_wait => workers are the ceiling (add workers or "
        "make them faster); large trainer_step => GPU / main-process "
        "consumer is the ceiling (adding workers won't help). 0 "
        "disables.",
    )
    p.add_argument(
        "--val-fraction", type=float, default=0.1,
        help="Fraction of events held out for validation.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the train/val split and PyTorch RNG.",
    )

    # Architecture selection.
    p.add_argument(
        "--arch",
        choices=["mlp", "polyhead", "mlp-factored"],
        default="mlp",
        help="Model architecture. 'mlp' (default) is the dual-scalar-"
        "forward construction; 'polyhead' uses a trunk that produces "
        "polynomial coefficients with structural priors; "
        "'mlp-factored' uses a trunk + factored head returning "
        "log W = <u, A(e, ·)> + <σ_pack, B(e, ·)>"
        " [ + <u⊗σ_pack, C(e, u, σ_pack)> ] with structural zeros at "
        "(u=0, σ=0) and σ-evenness built in. The detach flags "
        "--detach-pure-{shift,smear}-in-joint select between the "
        "full-cross default form, partially-factored two-term form, "
        "and fully-factored three-term form.",
    )

    # Trunk sizing (shared between mlp and polyhead arches).
    p.add_argument(
        "--trunk-hidden", type=int, default=64,
        help="Trunk MLP hidden width. mlp: trunk(y, c) -> e ∈ ℝ^d_emb. "
        "polyhead: trunk(y, c) -> coefs ∈ ℝ^n_basis.",
    )
    p.add_argument(
        "--trunk-layers", type=int, default=2,
        help="Number of trunk MLP hidden layers (shared).",
    )

    # MLP-B-only sizing.
    p.add_argument(
        "--d-emb", type=int, default=32,
        help="(--arch mlp) Trunk embedding dimension.",
    )
    p.add_argument(
        "--head-hidden", type=int, default=32,
        help="(--arch mlp) Head hidden width.",
    )
    p.add_argument(
        "--head-layers", type=int, default=2,
        help="(--arch mlp) Number of head hidden layers.",
    )

    # Polyhead-only sizing.
    p.add_argument(
        "--max-deg-u", type=int, default=3,
        help="(--arch polyhead) Max polynomial degree in u.",
    )
    p.add_argument(
        "--max-deg-sigma", type=int, default=4,
        help="(--arch polyhead) Max polynomial degree in σ_vec. Must "
        "be even (smear-symmetry constraint).",
    )
    p.add_argument(
        "--max-cross-deg", type=int, default=3,
        help="(--arch polyhead) Max combined degree |α|+|β| of cross "
        "(u, σ) terms.",
    )
    p.add_argument(
        "--basis", choices=["monomial", "chebyshev"], default="monomial",
        help="(--arch polyhead) Basis for the polynomial in (u, σ_vec). "
        "'monomial' uses raw u^α · σ^β (default; backwards-compatible). "
        "'chebyshev' uses tensor-product Chebyshev polynomials with "
        "axis-normalization by --basis-scale-u / --basis-scale-sigma "
        "(zero-anchored so r=1 at u=σ=0 is preserved). Same n_basis and "
        "structural priors as the monomial path; better-conditioned "
        "fitting when the perturbation range is comparable to the "
        "scale.",
    )
    p.add_argument(
        "--basis-scale-u", type=float, default=-1.0,
        help="(--arch polyhead, --basis chebyshev) Scale for u in the "
        "Chebyshev basis: T_n(u / basis_scale_u). Set to the nominal "
        "max |u| in standardized-target σ_y units so the recurrence "
        "stays in [-1, 1] over the training range. Default -1 = auto: "
        "use oversample · delta_max (the actual u-magnitude bound).",
    )
    p.add_argument(
        "--basis-scale-sigma", type=float, default=-1.0,
        help="(--arch polyhead, --basis chebyshev) Scale for σ_vec in "
        "the Chebyshev basis: T_n(σ_vec / basis_scale_sigma). Default "
        "-1 = auto: use oversample · sigma_max (the actual σ_vec bound).",
    )

    # Analytic Gaussian baseline (shared between mlp and polyhead).
    p.add_argument(
        "--gauss-baseline",
        action=argparse.BooleanOptionalAction, default=False,
        help="Add an analytic Gaussian density-ratio baseline that "
        "models p_nom(y|c) ~ N(μ(c), Σ(c)) and its shift+convolution "
        "under (u, σ_vec). The closed-form log-ratio is added "
        "additively to the polyhead/mlp pre-positivity ``d`` so the "
        "model residual only fits the deviation from a Gaussian. μ, Σ "
        "are produced by a small MLP from c only (--gauss-baseline-"
        "hidden / --gauss-baseline-layers). Init produces μ=0, Σ=I "
        "so the baseline starts at log r̂_Gauss=0 and the model "
        "recovers legacy behaviour at epoch 0.",
    )
    p.add_argument(
        "--gauss-baseline-hidden", type=int, default=64,
        help="(--gauss-baseline) Hidden width of the c → (μ, Σ) MLP.",
    )
    p.add_argument(
        "--gauss-baseline-layers", type=int, default=2,
        help="(--gauss-baseline) Number of hidden layers of the "
        "c → (μ, Σ) MLP.",
    )

    # Activation (shared).
    p.add_argument(
        "--activation", choices=["gelu", "relu", "silu", "tanh"],
        default="gelu",
        help="Activation used in trunk and head MLPs.",
    )

    # Loss / positivity / perturbation ranges.
    p.add_argument(
        "--loss-fn", choices=["dv", "lsif", "bce", "expkl"],
        default="lsif",
        help="Density-ratio loss family. "
        "'lsif' (default): least-squares importance fitting, optimum "
        "r-hat=r, unbiased per sample, locally L^2-in-r weighted by "
        "p_nom. "
        "'dv': Donsker-Varadhan, optimum log r̂=log r + const "
        "(constant ambiguity not fully resolved by the structural "
        "constraint at zero perturbation), slight finite-batch bias. "
        "'bce': logistic / binary cross-entropy (CARL), optimum "
        "log r̂=log r exactly (no constant ambiguity), per-sample "
        "unbiased, bounded gradients via sigmoid -- most stable choice "
        "but slower to converge in regions of high contrast. "
        "'expkl': exponential KL (DV with the outer log dropped), "
        "optimum log r̂=log r exactly (no constant ambiguity), "
        "per-sample unbiased, locally L^2-in-log-r weighted by p_pert "
        "(the analog of LSIF for log r-space). Unbounded gradient "
        "like LSIF; pair with low LR or BCE warm-start.",
    )
    p.add_argument(
        "--positivity", choices=["exp", "softplus", "asinh"],
        default="softplus",
        help="Positivity wrap on the pre-positivity scalar d. "
        "'softplus' (default) is fp16-safer but asymmetric (compresses "
        "positive log r). 'exp' is symmetric in log r-space (identity "
        "in d) but unbounded — clamps at +/-30 to keep fp32 safe. "
        "'asinh' is symmetric like 'exp' AND smoothly bounded "
        "(log r-hat = asinh(d) is identity for |d| << 1, "
        "log-compressed for |d| >> 1; r-hat = d + sqrt(d^2+1) grows "
        "only linearly so no clamp is needed). Best stability for "
        "LSIF / expKL with Gaussian-likelihood-shaped problems.",
    )
    p.add_argument(
        "--include-smear",
        action=argparse.BooleanOptionalAction, default=False,
        help="Include smear (and joint shift+smear) perturbations in "
        "training. Default OFF: shift-only architecture (no Sigma_pack "
        "in mlp head; no sigma-dependent basis terms in polyhead) and "
        "shift-only sampling (every event has sigma_vec=0). When ON "
        "the architecture grows to handle Sigma_pack / cross terms, "
        "sampling becomes the stratified 1/3 SHIFT / 1/3 SMEAR / 1/3 "
        "JOINT mix, and --smear-K / --smear-residual / "
        "--detach-pure-in-joint take effect. Use --include-smear to "
        "enable.",
    )
    p.add_argument(
        "--delta-max", type=float, default=1.0,
        help="Max |u| in standardized target units.",
    )
    p.add_argument(
        "--sigma-max", type=float, default=1.0,
        help="Max smear magnitude |σ_vec| in standardized target units.",
    )
    p.add_argument(
        "--oversample", type=float, default=1.3,
        help="Oversample factor on training perturbation magnitudes; "
        "1.3× covers the operational range with margin.",
    )

    # Shift-axis Smolyak sparse-grid evaluation.
    p.add_argument(
        "--shift-smolyak",
        action=argparse.BooleanOptionalAction, default=False,
        help="Use a deterministic Chebyshev-Lobatto Smolyak sparse "
        "grid for the shift (u) evaluation points, instead of one "
        "stochastic u per event per step. Each event is evaluated at "
        "all K_smolyak grid points, with weights divided by K_smolyak "
        "so the per-event aggregate weight is preserved. Only valid "
        "with --no-include-smear (shift-only training); errors out "
        "otherwise. Off by default; turn on with --shift-smolyak.",
    )
    p.add_argument(
        "--shift-smolyak-level", type=int, default=0,
        help="Smolyak level (0-indexed; level L grid has 1, 7, 25, "
        "69, 177, 441 points in d=3 for L=0..5). 0 = auto: with "
        "--arch polyhead, picks the lowest L whose grid has more "
        "points than the number of pure-shift basis functions "
        "(C(d + max_deg_u, d) − 1). With --arch mlp the auto path is "
        "not meaningful; pass an explicit positive level instead.",
    )
    p.add_argument(
        "--shift-stochastic-extra", type=int, default=0,
        help="Number of additional stochastic u samples per event on "
        "top of the Smolyak grid (or by themselves if --shift-smolyak "
        "is off). Drawn with the same direction × magnitude scheme as "
        "the standard sampler. Useful for adding broader random "
        "coverage on top of the deterministic grid; combined with "
        "--shift-smolyak this gives K_total = K_smolyak + extra "
        "evaluations per event. 0 disables.",
    )

    # Smear ε-integration.
    p.add_argument(
        "--smear-K", type=int, default=1,
        help="Quadrature nodes for the smear ε integration in the "
        "pert-side LSIF/DV expectation. K=1 (default) is K=1 "
        "stochastic — one random ε per event. K≥2 uses deterministic "
        "Gauss-Hermite (probabilist's) at K nodes — exact for "
        "polynomials in ε up to degree 2K−1, with O(σ_max^{2K}) "
        "truncation for non-polynomial features. Cost: K perturbed "
        "forwards per event vs 1 for K=1. K=3 is exact through deg 5; "
        "K=5 through deg 9. For smooth log r, K=3-5 is plenty.",
    )
    p.add_argument(
        "--smear-residual", action="store_true",
        help="(only with --smear-K >= 2) Add a stochastic residual "
        "control variate to the GH estimator: sample one extra "
        "epsilon ~ N(0, 1) per event, evaluate r-hat at the "
        "corresponding y_pert, and form S_K + (r-hat(eps~) - g(eps~)) "
        "where g is the Lagrange interpolant through the GH nodes. "
        "Cost: K+1 perturbed forwards per event (~20-30%% over pure "
        "GH). Removes the polynomial-truncation bias of pure GH "
        "(estimator becomes unbiased for arbitrary smooth f), at the "
        "cost of a small variance bump. Useful as (a) a sanity check "
        "that GH bias is negligible at your K, and (b) a fallback for "
        "large |sigma_vec| where the polynomial truncation matters.",
    )
    p.add_argument(
        "--detach-pure-in-joint",
        action=argparse.BooleanOptionalAction, default=False,
        help="(polyhead / mlp arches) Detach the pure-shift "
        "(sigma -> 0) and pure-smear (u -> 0) contributions to log r "
        "on JOINT-mode events (mode==2; both u and sigma_vec "
        "nonzero). Routes pure-axis gradients exclusively through the "
        "lower-noise SHIFT/SMEAR events and lets JOINT events train "
        "only the cross/mixed interaction. polyhead: zero cost (mask "
        "on pure-u/pure-sigma basis-index slots). mlp: ~2x head cost "
        "from two extra head forwards f(e, u, 0) and f(e, 0, "
        "sigma_pack) used to decompose d_full = d_pure_u + "
        "d_pure_sigma + d_mixed. Default: disabled. For the "
        "mlp-factored arch use --detach-pure-shift-in-joint and/or "
        "--detach-pure-smear-in-joint instead, which select the "
        "structural factorisation form.",
    )
    p.add_argument(
        "--detach-pure-shift-in-joint",
        action="store_true", default=False,
        help="(mlp-factored only) Detach the pure-shift block A on "
        "JOINT events. Equivalently: structurally factor A so it "
        "depends only on (e, u), not on σ_pack; the cross-term "
        "absorption lives in B's u-dependence. A's gradient flows "
        "only from SHIFT events.",
    )
    p.add_argument(
        "--detach-pure-smear-in-joint",
        action="store_true", default=False,
        help="(mlp-factored only) Detach the pure-smear block B on "
        "JOINT events. Equivalently: structurally factor B so it "
        "depends only on (e, σ_pack), not on u; the cross-term "
        "absorption lives in A's σ_pack-dependence. B's gradient "
        "flows only from SMEAR events. When combined with "
        "--detach-pure-shift-in-joint, the head becomes fully "
        "three-block factorised with a separate cross head C(e, u, "
        "σ_pack).",
    )

    # Optimization.
    p.add_argument(
        "--epochs", type=int, default=200,
        help="Maximum training epochs (early-stopping may end sooner).",
    )
    p.add_argument(
        "--lr", type=float, default=1e-3,
        help="Initial Adam learning rate.",
    )
    p.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="Adam weight decay (L2 regularization).",
    )
    p.add_argument(
        "--patience", type=int, default=20,
        help="Early-stopping patience: stop after this many consecutive "
        "epochs with no validation-loss improvement (see "
        "--patience-rel-threshold for the improvement criterion).",
    )
    p.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stopping without changing the LR-decay "
        "schedule. The ReduceLROnPlateau scheduler still halves the "
        "LR after ``--patience // 2`` non-improving epochs; the run "
        "continues until ``--epochs`` is reached. Useful for forcing "
        "training to use the full epoch budget when you want to see "
        "the long-tail behaviour at very small LR.",
    )
    p.add_argument(
        "--patience-rel-threshold",
        type=float,
        default=1e-4,
        help="*--loss-fn lsif only*: minimum relative improvement in "
        "the validation loss to count as 'improvement' against the "
        "patience clock; an epoch counts as improvement only if "
        "``val_loss < best_val − rel · |best_val|``. Used because "
        "LSIF (L²-in-r-space) has val near 0 / negative, making any "
        "fixed-absolute test scale-inappropriate. BCE / DV / expKL "
        "use a fixed absolute 1e-4 threshold instead since their val "
        "sits at O(1). Default: %(default)s.",
    )
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Training batch size. None auto-selects: "
        "32768 on CUDA, 16384 on CPU.",
    )

    # Run config.
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu. Multi-GPU requires cuda (see --num-gpus).",
    )
    p.add_argument(
        "--num-gpus", type=int, default=-1,
        help="Number of GPUs / worker processes. -1 (default) auto-"
        "detects via torch.cuda.device_count(). Set 1 to force "
        "single-process even on a multi-GPU host. Ignored when "
        "--device cpu.",
    )
    p.add_argument(
        "--prefetch-shuffle",
        action=argparse.BooleanOptionalAction, default=True,
        help="[default: on] Hide the per-epoch ``torch.randperm(n)`` "
        "stall by computing the next epoch's permutation in a "
        "background thread while the current epoch is still "
        "training. At ~10⁸-row training sets the shuffle alone is "
        "~5–10 s per epoch; with prefetch on, only the first epoch "
        "pays this. Disable with --no-prefetch-shuffle.",
    )
    p.add_argument(
        "--time-loader",
        action="store_true",
        help="Print one-line timing per epoch start: how long the "
        "shuffle / first-batch gather + pin took. Useful for "
        "diagnosing per-epoch startup latency. Same as "
        "``LOADER_TIME=1`` env var.",
    )
    p.add_argument(
        "--checkpoint-every", type=int, default=1,
        help="Write a checkpoint every N epochs. 0 disables periodic "
        "checkpoints (final model is still saved at end of training).",
    )
    p.add_argument(
        "--precision", choices=["fp32", "bf16", "fp16"], default="fp32",
        help="Training/validation forward-pass precision.",
    )
    p.add_argument(
        "--compile", action="store_true",
        help="Wrap the model in torch.compile(mode='reduce-overhead').",
    )

    # Profiling diagnostics.
    p.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction, default=False,
        help="If set, run a short ``torch.profiler`` pass on the "
        "first few training steps and exit (does not proceed to the "
        "epoch loop). Useful for identifying CUDA-time hotspots.",
    )
    p.add_argument(
        "--profile-warmup", type=int, default=3,
        help="Profile schedule warmup steps (not recorded).",
    )
    p.add_argument(
        "--profile-active", type=int, default=5,
        help="Profile schedule active steps (recorded).",
    )
    p.add_argument(
        "--profile-output", default=None,
        help="Optional path for Chrome trace export "
        "(viewable in chrome://tracing or Perfetto). Under DDP, the "
        "rank index is appended before the extension.",
    )
    return p.parse_args()


# ============================================================================
# main
# ============================================================================

# -----------------------------------------------------------------------------
# Worker (runs once per GPU in distributed mode, once total otherwise)
# -----------------------------------------------------------------------------

def main_worker(
    rank,
    args,
    world_size,
    master_port,
    stats,
    weight_mean,
    shard_files,
    shard_row_counts,
    n_features,
    n_cond,
):
    """One worker process. Initializes the process group (if
    distributed), opens its own per-rank streaming loader over the
    Arrow shards, builds the model + DDP wrap, and runs training.
    Only rank 0 prints, writes checkpoints, and saves the final
    artifact.
    """
    from arrow_shard_loader import ArrowShardLoader

    is_dist = world_size > 1
    is_rank0 = rank == 0

    # Device selection + optional process-group init.
    if is_dist:
        import torch.distributed as dist
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        if args.device.startswith("cuda"):
            torch.cuda.set_device(rank)
            device = f"cuda:{rank}"
            backend = "nccl"
        else:
            device = "cpu"
            backend = "gloo"
        dist.init_process_group(
            backend=backend, rank=rank, world_size=world_size,
        )
    else:
        if args.device.startswith("cuda"):
            torch.cuda.set_device(0)
            device = "cuda:0"
        else:
            device = args.device

    # Per-rank streaming loaders over the Arrow shards. Each rank
    # gets a round-robin slice of the shard list. train / val split
    # is done contiguously per record batch (first val_fraction
    # rows go to val); the shards are already globally shuffled by
    # the bucket-shuffle pass so this is unbiased.
    # ``--pin-memory`` is tristate via argparse: True/False if the
    # user passed the flag explicitly, None for the auto-default
    # (on if device is CUDA, off otherwise).
    if args.pin_memory is None:
        pin_memory = device.startswith("cuda")
    else:
        pin_memory = bool(args.pin_memory)
    n_data_workers = max(0, int(args.data_workers))
    n_train_workers_hint = max(1, n_data_workers)
    n_val_workers_hint = max(1, n_data_workers // 2)
    train_ds = ArrowShardLoader(
        shard_files, stats, weight_mean,
        world_size=world_size, rank=rank,
        batch_size=args.batch_size,
        split="train",
        val_fraction=args.val_fraction,
        shuffle=True, drop_last=True,
        pin_memory=pin_memory,
        seed=int(args.seed),
        weight_mode=args.weight_handling,
        shard_row_counts=shard_row_counts,
        num_workers_hint=n_train_workers_hint,
    )
    val_ds = ArrowShardLoader(
        shard_files, stats, weight_mean,
        world_size=world_size, rank=rank,
        batch_size=args.batch_size,
        split="val",
        val_fraction=args.val_fraction,
        shuffle=False, drop_last=False,
        pin_memory=pin_memory,
        seed=int(args.seed),
        weight_mode=args.weight_handling,
        shard_row_counts=shard_row_counts,
        num_workers_hint=n_val_workers_hint,
    )

    # When --data-workers > 0, wrap as ``DataLoader`` so each rank's
    # data pipeline is fed by N worker processes. ``batch_size=None``
    # disables PyTorch's auto-batching (our dataset already yields
    # complete training batches). With ``num_workers == 0`` we bypass
    # DataLoader entirely and let the dataset's own in-process
    # prefetch thread handle pipelining.
    if n_data_workers > 0:
        import torch.utils.data as _td
        from arrow_shard_loader import dataloader_worker_init
        train_loader = _td.DataLoader(
            train_ds,
            batch_size=None,
            num_workers=n_data_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=2,
            worker_init_fn=dataloader_worker_init,
        )
        val_loader = _td.DataLoader(
            val_ds,
            batch_size=None,
            num_workers=max(1, n_data_workers // 2),
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=2,
            worker_init_fn=dataloader_worker_init,
        )
    else:
        train_loader = train_ds
        val_loader = val_ds

    # Optional timing wrapper. Prints loader_wait / trainer_step for
    # the first ``--profile-data-pipeline`` batches of each iter()
    # call so the bottleneck (workers vs main-process+GPU) is easy
    # to read off.
    n_profile = int(getattr(args, "profile_data_pipeline", 0) or 0)
    if n_profile > 0 and is_rank0:
        from arrow_shard_loader import TimedLoader
        train_loader = TimedLoader(train_loader, n_print=n_profile, label="train")
        val_loader = TimedLoader(val_loader, n_print=n_profile, label="val")
    if is_rank0:
        print(
            f"rank {rank}/{world_size}  "
            f"~train batches {len(train_loader)}  "
            f"~val batches {len(val_loader)}  "
            f"batch {args.batch_size}  device {device}"
        )

    # Build model per --arch. In shift-only mode (--no-include-smear,
    # the default) the σ-dependent parts of each architecture are
    # dropped entirely:
    #   * mlp: head input excludes Σ_pack (shift_only=True).
    #   * polyhead: max_deg_sigma → 0 / max_cross_deg → max_deg_u, so
    #     only pure-u basis terms remain (and the σ-symmetry +
    #     no-constant priors still hold trivially).
    activation_cls = _ACTIVATIONS[args.activation]
    gauss_baseline = (
        GaussBaseline(
            n_features=n_features,
            n_cond=n_cond,
            hidden=int(args.gauss_baseline_hidden),
            n_layers=int(args.gauss_baseline_layers),
            activation=activation_cls,
        )
        if bool(args.gauss_baseline) else None
    )
    if args.arch == "mlp":
        inner_model = ReweightMLP_B(
            n_features=n_features,
            n_cond=n_cond,
            d_emb=args.d_emb,
            trunk_hidden=args.trunk_hidden,
            trunk_layers=args.trunk_layers,
            head_hidden=args.head_hidden,
            head_layers=args.head_layers,
            activation=activation_cls,
            shift_only=not args.include_smear,
            gauss_baseline=gauss_baseline,
        ).to(device)
        n_params = sum(p.numel() for p in inner_model.parameters())
        if is_rank0:
            print(
                f"ReweightMLP_B"
                f"{' (shift-only)' if not args.include_smear else ''}: "
                f"d_emb={args.d_emb} "
                f"trunk={args.trunk_hidden}×{args.trunk_layers} "
                f"head={args.head_hidden}×{args.head_layers}  "
                f"params={n_params:,}"
            )
    elif args.arch == "mlp-factored":
        inner_model = ReweightMLPFactored(
            n_features=n_features,
            n_cond=n_cond,
            d_emb=args.d_emb,
            trunk_hidden=args.trunk_hidden,
            trunk_layers=args.trunk_layers,
            head_hidden=args.head_hidden,
            head_layers=args.head_layers,
            activation=activation_cls,
            shift_only=not args.include_smear,
            detach_pure_shift_in_joint=bool(
                args.detach_pure_shift_in_joint
            ),
            detach_pure_smear_in_joint=bool(
                args.detach_pure_smear_in_joint
            ),
            gauss_baseline=gauss_baseline,
        ).to(device)
        n_params = sum(p.numel() for p in inner_model.parameters())
        if is_rank0:
            form = "default (full cross)"
            if (
                inner_model.detach_pure_shift_in_joint
                and inner_model.detach_pure_smear_in_joint
            ):
                form = "three-term factored (separate C)"
            elif inner_model.detach_pure_shift_in_joint:
                form = "shift-factored (A: e,u; cross in B)"
            elif inner_model.detach_pure_smear_in_joint:
                form = "smear-factored (B: e,σ_pack; cross in A)"
            print(
                f"ReweightMLPFactored"
                f"{' (shift-only)' if not args.include_smear else ''}: "
                f"d_emb={args.d_emb} "
                f"trunk={args.trunk_hidden}×{args.trunk_layers} "
                f"head={args.head_hidden}×{args.head_layers}  "
                f"form={form}  "
                f"params={n_params:,}"
            )
    elif args.arch == "polyhead":
        eff_max_deg_sigma = (
            args.max_deg_sigma if args.include_smear else 0
        )
        eff_max_cross_deg = (
            args.max_cross_deg if args.include_smear else args.max_deg_u
        )
        # Auto-default Chebyshev scales to the actual perturbation
        # bounds (oversample · delta_max for u, oversample · sigma_max
        # for σ_vec) so T_n(x / scale) stays in [-1, 1] over training.
        # Sentinel <= 0 from argparse means "not specified".
        eff_basis_scale_u = (
            float(args.basis_scale_u) if args.basis_scale_u > 0
            else float(args.oversample * args.delta_max)
        )
        eff_basis_scale_sigma = (
            float(args.basis_scale_sigma) if args.basis_scale_sigma > 0
            else (
                float(args.oversample * args.sigma_max)
                if args.include_smear else 1.0
            )
        )
        # Persist the resolved values back so the rest of main_worker
        # (logging, _build_ckpt) sees the effective scale.
        args.basis_scale_u = eff_basis_scale_u
        args.basis_scale_sigma = eff_basis_scale_sigma
        inner_model = ReweightPolyhead(
            n_features=n_features,
            n_cond=n_cond,
            trunk_hidden=args.trunk_hidden,
            trunk_layers=args.trunk_layers,
            max_deg_u=args.max_deg_u,
            max_deg_sigma=eff_max_deg_sigma,
            max_cross_deg=eff_max_cross_deg,
            activation=activation_cls,
            basis=args.basis,
            basis_scale_u=eff_basis_scale_u,
            basis_scale_sigma=eff_basis_scale_sigma,
            gauss_baseline=gauss_baseline,
        ).to(device)
        n_params = sum(p.numel() for p in inner_model.parameters())
        if is_rank0:
            print(
                f"ReweightPolyhead"
                f"{' (shift-only)' if not args.include_smear else ''}: "
                f"trunk={args.trunk_hidden}×"
                f"{args.trunk_layers}  "
                f"max_deg_u={args.max_deg_u} max_deg_sigma="
                f"{eff_max_deg_sigma} max_cross_deg={eff_max_cross_deg}  "
                f"basis={args.basis}"
                + (
                    f"(scale_u={args.basis_scale_u:g},"
                    f"scale_sigma={args.basis_scale_sigma:g})"
                    if args.basis == "chebyshev" else ""
                )
                + f"  n_basis={inner_model.n_basis}  "
                f"params={n_params:,}"
            )
    else:
        raise ValueError(f"unknown arch {args.arch!r}")

    if is_rank0:
        print(
            f"loss={args.loss_fn} positivity={args.positivity} "
            f"delta_max={args.delta_max} sigma_max={args.sigma_max} "
            f"precision={args.precision}"
        )

    # Effective smear settings collapse to the shift-only defaults
    # when --no-include-smear; the user-set --smear-K / --smear-residual
    # / --detach-pure-in-joint / --detach-pure-{shift,smear}-in-joint
    # are tracked for logging but ignored.
    eff_smear_K = args.smear_K if args.include_smear else 1
    eff_smear_residual = (
        bool(args.smear_residual) if args.include_smear else False
    )
    eff_detach_pure_in_joint = (
        bool(args.detach_pure_in_joint) if args.include_smear else False
    )
    eff_detach_pure_shift_in_joint = (
        bool(args.detach_pure_shift_in_joint)
        if args.include_smear else False
    )
    eff_detach_pure_smear_in_joint = (
        bool(args.detach_pure_smear_in_joint)
        if args.include_smear else False
    )

    log_lines: List[str] = []
    if args.include_smear:
        smear_mode_str = (
            f"K={eff_smear_K}"
            + (" + residual" if eff_smear_residual else "")
        )
    else:
        smear_mode_str = "off (shift-only)"
    if is_rank0:
        gauss_str = (
            f"gauss_baseline=({args.gauss_baseline_hidden}×"
            f"{args.gauss_baseline_layers})"
            if bool(args.gauss_baseline) else "gauss_baseline=off"
        )
        log_lines.append(
            f"arch={args.arch} include_smear={bool(args.include_smear)} "
            f"loss={args.loss_fn} positivity={args.positivity} "
            f"delta_max={args.delta_max} "
            f"sigma_max={args.sigma_max if args.include_smear else 0.0} "
            f"smear=({smear_mode_str}) "
            f"detach_pure_in_joint={eff_detach_pure_in_joint} "
            f"detach_pure_shift_in_joint={eff_detach_pure_shift_in_joint} "
            f"detach_pure_smear_in_joint={eff_detach_pure_smear_in_joint} "
            f"{gauss_str} "
            f"precision={args.precision} compile={bool(args.compile)} "
            f"world_size={world_size} params={n_params}"
        )
        if not args.include_smear and (
            args.smear_K > 1 or args.smear_residual
            or not args.detach_pure_in_joint
        ):
            print(
                "[note] --no-include-smear (shift-only): ignoring "
                "--smear-K / --smear-residual / "
                "--detach-pure-in-joint."
            )

    # Σ_pack indices live on device for the MLP arch (shift+smear
    # only — the shift-only head doesn't take Σ_pack).
    sigma_pack_iu = sigma_pack_ju = None
    if args.arch in ("mlp", "mlp-factored") and args.include_smear:
        iu, ju = _sigma_pack_indices(n_features)
        sigma_pack_iu = iu.to(device)
        sigma_pack_ju = ju.to(device)

    # Gauss–Hermite nodes/weights for the smear ε integration. Both
    # ``None`` for shift-only (smear off) or K=1 stochastic;
    # otherwise (K-vector, K-vector) on device.
    if args.include_smear:
        gh_nodes, gh_weights = _gh_nodes_weights(eff_smear_K)
    else:
        gh_nodes, gh_weights = None, None
    if gh_nodes is not None:
        gh_nodes = gh_nodes.to(device)
        gh_weights = gh_weights.to(device)
        if is_rank0:
            print(
                f"Gauss-Hermite ε integration: K={eff_smear_K} "
                f"{'+ residual control variate' if eff_smear_residual else ''}"
                .strip()
            )
    elif args.include_smear and eff_smear_residual and is_rank0:
        print(
            "[warning] --smear-residual has no effect when --smear-K=1; "
            "ignoring."
        )

    # Shift-axis Smolyak grid (deterministic per-event evaluation
    # points). Built once at startup, broadcast to every event during
    # training. Auto-level requires polyhead arch; for MLP, an
    # explicit ``--shift-smolyak-level`` is required because the auto
    # heuristic ("more points than pure-shift basis functions") is
    # only well-defined for the polyhead's polynomial output.
    shift_smolyak_pts = None
    shift_smolyak_level = 0
    if args.shift_smolyak:
        if args.include_smear:
            raise ValueError(
                "--shift-smolyak is only supported with "
                "--no-include-smear (shift-only training). "
                "include_smear=True interacts non-trivially with "
                "Smolyak shift-grid sampling and is not implemented."
            )
        if args.shift_smolyak_level > 0:
            shift_smolyak_level = int(args.shift_smolyak_level)
        elif args.arch == "polyhead":
            n_pure = _n_pure_shift_basis(n_features, args.max_deg_u)
            shift_smolyak_level = _auto_smolyak_level(
                n_features, n_pure,
            )
            if is_rank0:
                print(
                    f"shift-smolyak: auto-level → "
                    f"L={shift_smolyak_level} "
                    f"(d={n_features}, max_deg_u={args.max_deg_u}, "
                    f"n_pure_shift_basis={n_pure})"
                )
        else:
            raise ValueError(
                "--shift-smolyak with --arch mlp requires an "
                "explicit --shift-smolyak-level (auto-level is "
                "polyhead-specific)."
            )
        smolyak_np = _smolyak_grid(n_features, shift_smolyak_level)
        # Scale [-1, 1] → [-half, +half] to match the training
        # perturbation prior (oversample × delta_max).
        half = float(args.oversample) * float(args.delta_max)
        smolyak_np = smolyak_np * half
        shift_smolyak_pts = torch.from_numpy(
            smolyak_np
        ).to(device=device, dtype=torch.float32)
        if is_rank0:
            print(
                f"shift-smolyak: K_smolyak={shift_smolyak_pts.shape[0]} "
                f"deterministic points × half={half:.3f} "
                f"(level {shift_smolyak_level}); "
                f"K_extra_stochastic={args.shift_stochastic_extra}; "
                f"K_total={shift_smolyak_pts.shape[0] + args.shift_stochastic_extra}"
            )
            log_lines.append(
                f"shift-smolyak: level={shift_smolyak_level} "
                f"K_smolyak={shift_smolyak_pts.shape[0]} "
                f"K_extra={args.shift_stochastic_extra}"
            )

    # Wrap inner model in the single-forward DDP-friendly wrapper
    # (compute_log_r_quadrature) before any DDP / compile wraps.
    wrapper = ReweightWrapper(
        inner_model, args.arch, args.positivity,
        detach_pure_in_joint=eff_detach_pure_in_joint,
    ).to(device)

    model = wrapper
    if is_dist:
        if args.device.startswith("cuda"):
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[rank],
            )
        else:
            model = nn.parallel.DistributedDataParallel(model)
    # Compile after DDP wrap (matches train_muon_response_flow.py
    # convention). State-dict capture goes through ``inner_model``,
    # bypassing both wrappers.
    if args.compile:
        if is_rank0:
            print(
                "torch.compile: enabled (mode=reduce-overhead) — "
                "first batch will JIT-compile",
                flush=True,
            )
        model = torch.compile(model, mode="reduce-overhead")

    best_val = train(
        model, inner_model, args.arch, train_loader, val_loader,
        device, args, log_lines, stats=stats,
        sigma_pack_iu=sigma_pack_iu, sigma_pack_ju=sigma_pack_ju,
        gh_nodes=gh_nodes, gh_weights=gh_weights,
        is_dist=is_dist, is_rank0=is_rank0,
        shift_smolyak_pts=shift_smolyak_pts,
        shift_stochastic_extra=int(args.shift_stochastic_extra),
    )

    if is_rank0:
        print(f"best val_{args.loss_fn}: {best_val:+.4f}")
        log_lines.append(f"best val_{args.loss_fn}: {best_val:+.4f}")

        # Final artifact (the ckpt payload sans epoch/loss bookkeeping).
        final = _build_ckpt(
            model=inner_model, arch=args.arch, args=args, epoch=-1,
            train_loss=float("nan"), val_loss=best_val, best_val=best_val,
            stats=stats,
        )
        final_name = {
            "mlp": "shift_smear_reweight_mlp.pt",
            "mlp-factored": "shift_smear_reweight_mlp_factored.pt",
            "polyhead": "shift_smear_reweight_polyhead.pt",
        }[args.arch]
        final_path = os.path.join(args.output, final_name)
        torch.save(final, final_path)
        print(f"wrote final model to {final_path}")

        with open(os.path.join(args.output, "log.txt"), "w") as f:
            f.write("\n".join(log_lines) + "\n")

    if is_dist:
        import torch.distributed as dist
        dist.destroy_process_group()


# -----------------------------------------------------------------------------
# Main (dispatcher: loads data once, then spawns worker(s))
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from arrow_shard_loader import (
        resolve_shard_files,
        compute_stats_streaming,
    )

    shard_files, shard_row_counts = resolve_shard_files(
        args.input_files, return_counts=True,
    )
    print(
        f"resolved {len(shard_files)} Arrow shard(s) from "
        f"{len(args.input_files)} input arg(s)"
        + ("" if shard_row_counts is None else
           f"; manifest reports {sum(shard_row_counts):,} total rows")
    )

    # One streaming pass for preproc mean/std + weight mean.
    # ``--stats-max-rows`` (CLI knob added below) caps the warmup
    # to a subsample if you want training to start faster.
    stats_max = int(getattr(args, "stats_max_rows", -1) or -1)
    stats, weight_mean = compute_stats_streaming(
        shard_files, max_rows=stats_max, progress=True,
        weight_mode=args.weight_handling,
        robust=bool(getattr(args, "robust_stats", False)),
        robust_sample_rows=int(
            getattr(args, "robust_sample_rows", 1_000_000)
        ),
    )
    with open(os.path.join(args.output, "preproc.json"), "w") as f:
        json.dump(asdict(stats), f, indent=2)

    n_features = len(stats.target_names)
    n_cond = len(stats.cond_names)

    if args.batch_size is None:
        args.batch_size = 32768 if args.device != "cpu" else 16384

    # World size: auto-detect GPU count by default. CPU mode forces 1.
    if args.device.startswith("cuda"):
        if args.num_gpus == -1:
            world_size = max(1, torch.cuda.device_count())
        else:
            world_size = max(1, args.num_gpus)
    else:
        world_size = 1

    print(
        f"streaming training over {len(shard_files)} shard(s)  "
        f"world_size {world_size}  device {args.device}"
    )

    if world_size == 1:
        main_worker(
            0, args, 1, 0, stats, weight_mean, shard_files,
            shard_row_counts, n_features, n_cond,
        )
    else:
        # Pick a free port on localhost for the rendezvous.
        import socket
        sock = socket.socket()
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
        sock.close()
        import torch.multiprocessing as mp
        mp.spawn(
            main_worker,
            args=(
                args, world_size, master_port, stats, weight_mean,
                shard_files, shard_row_counts, n_features, n_cond,
            ),
            nprocs=world_size,
            join=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
