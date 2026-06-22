"""Q integration and Y/qT rebin helpers for the SCETlib bT-grid ParamModel.

All weight-construction is numpy (runs once at construction time); runtime
contractions are simple ``tf.tensordot`` / ``tf.einsum`` calls.

Three pieces:

1. :func:`dense_index_map` — build a ``(NQ, NY, NqT)`` int array mapping each
   rectangular grid cell to a flat bin index in the sparse btgrid; ``-1``
   marks missing combos. Use ``tf.gather`` with a sentinel to pad a sparse
   ``(Nbins,)`` σ tensor into a dense ``(NQ, NY, NqT)``.

2. :func:`q_integrate_weights` — produces a ``(NQ,)`` weight vector for
   ``arctan_Q²``-method Simpson integration over the Z mass window. Apply
   via ``tf.einsum('q, qyz -> yz', w, sigma)``.

3. :func:`rebin_weights` — given a fine source grid and a coarser target-bin
   edge list, builds a ``(N_target, N_source)`` Simpson weight matrix.
   Apply via ``tf.tensordot``.

Parity-tested against the numpy reference implementation (development tree).
"""

import numpy as np
import tensorflow as tf

from wremnants.postprocessing.scetlib_np.btgrid_tf import (
    _as_dtype,
    simpson_weights,
)
from wremnants.utilities import common as wrem_common

# Z resonance parameters for the Q-integration change of variable, in the
# s-dependent-width scheme (see wremnants.utilities.common). Only set the centre
# and scale of the arctan-Q^2 transform below; they do not change the physics.
MZ_S_DEP_WIDTH = wrem_common.MZ_S_DEP_WIDTH
GAMMAZ_S_DEP_WIDTH = wrem_common.GAMMAZ_S_DEP_WIDTH


# =============================================================================
# Sparse → dense index map
# =============================================================================


def dense_index_map(bins, Q_unique=None, Y_unique=None, qT_unique=None):
    """Build a dense rectangular index map for a sparse btgrid.

    Parameters
    ----------
    bins : list of tuples
        From ``load_btgrid_shards``: each element is ``(Q, Y, qT, lep)``,
        sorted lexicographically.

    Returns
    -------
    dict with keys:
        Q_unique, Y_unique, qT_unique : sorted unique axis values
        flat_idx : ndarray of shape (NQ, NY, NqT), int64. -1 marks missing.
        missing_count : int
    """
    if Q_unique is None:
        Q_unique = sorted({b[0] for b in bins})
    if Y_unique is None:
        Y_unique = sorted({b[1] for b in bins})
    if qT_unique is None:
        qT_unique = sorted({b[2] for b in bins})

    Q_unique = np.asarray(Q_unique, dtype=np.float64)
    Y_unique = np.asarray(Y_unique, dtype=np.float64)
    qT_unique = np.asarray(qT_unique, dtype=np.float64)

    Q_pos = {Q: i for i, Q in enumerate(Q_unique)}
    Y_pos = {Y: i for i, Y in enumerate(Y_unique)}
    qT_pos = {qT: i for i, qT in enumerate(qT_unique)}

    flat_idx = np.full(
        (Q_unique.size, Y_unique.size, qT_unique.size), -1, dtype=np.int64
    )
    for k, (Q, Y, qT, _lep) in enumerate(bins):
        flat_idx[Q_pos[Q], Y_pos[Y], qT_pos[qT]] = k

    missing = int(np.sum(flat_idx == -1))
    return dict(
        Q_unique=Q_unique,
        Y_unique=Y_unique,
        qT_unique=qT_unique,
        flat_idx=flat_idx,
        missing_count=missing,
    )


def sparse_to_dense_tf(sigma_flat, flat_idx):
    """Reshape a sparse ``(Nbins,)`` σ tensor to dense ``(NQ, NY, NqT)``.

    Missing cells (``flat_idx == -1``) are padded with 0. Implemented via
    ``tf.gather`` with a 0-padded sentinel row.
    """
    sigma_flat = _as_dtype(sigma_flat)
    # Append one extra "zero" entry that the -1 indices will gather.
    extended = tf.concat([sigma_flat, tf.zeros([1], dtype=sigma_flat.dtype)], axis=0)
    sentinel = tf.cast(tf.shape(sigma_flat)[0], tf.int64)  # index of the appended zero
    idx_safe = tf.where(tf.equal(flat_idx, -1), sentinel, flat_idx)
    return tf.gather(extended, idx_safe)


# =============================================================================
# Q-integration weights (arctan_Q² method, matches numpy integrate_over_Q)
# =============================================================================


def q_integrate_weights(
    Q_grid, Q_lo, Q_hi, q0=MZ_S_DEP_WIDTH, Gamma=GAMMAZ_S_DEP_WIDTH
):
    """Simpson weights for integrating over Q ∈ [Q_lo, Q_hi] in arctan-Q² space.

    Implements the same change of variable as
    the numpy-reference ``integrate_over_Q`` with ``method="arctan_Q2"``:
    x = arctan((Q² - q0²) / (q0 Γ)) flattens the Breit-Wigner peak, then
    Simpson on x with the Jacobian dQ/dx.

    Returns a ``(NQ,)`` weight vector with zeros outside ``[Q_lo, Q_hi]``.
    """
    Q_grid = np.asarray(Q_grid, dtype=np.float64)
    mask = (Q_grid >= Q_lo) & (Q_grid <= Q_hi)
    if mask.sum() < 2:
        raise ValueError(
            f"q_integrate_weights: need ≥ 2 Q samples in [{Q_lo}, {Q_hi}]; "
            f"got {mask.sum()} from Q_grid={Q_grid}"
        )
    Q_sub = Q_grid[mask]
    x = np.arctan((Q_sub**2 - q0**2) / (q0 * Gamma))
    jac = (q0 * Gamma + (Q_sub**2 - q0**2) ** 2 / (q0 * Gamma)) / (2.0 * Q_sub)
    w_simpson = simpson_weights(x)  # weights in x-space
    w_full = w_simpson * jac  # at each Q sample
    w_padded = np.zeros_like(Q_grid)
    w_padded[mask] = w_full
    return w_padded


def integrate_over_Q_tf(sigma_QYqT, Q_weights):
    """Apply precomputed Q weights. ``sigma_QYqT`` shape ``(NQ, NY, NqT)``."""
    Q_weights = _as_dtype(Q_weights)
    return tf.einsum("q, qyz -> yz", Q_weights, sigma_QYqT)


# =============================================================================
# Y / qT rebin weights
# =============================================================================


def rebin_weights(source_grid, target_edges, name="axis", tol=1e-9):
    """Build a ``(N_target, N_source)`` Simpson rebin matrix.

    For each target bin ``[target_edges[i], target_edges[i+1]]``, find the
    source samples that fall in (with ``tol`` slack) the bin's interior +
    edges, then compute Simpson weights for those samples. Returns a dense
    matrix; entries are 0 for source samples not contributing to a given
    target bin.

    Mirrors the per-bin call pattern of
    the numpy-reference ``integrate_over_axis_bin``.
    """
    source_grid = np.asarray(source_grid, dtype=np.float64)
    target_edges = np.asarray(target_edges, dtype=np.float64)
    if target_edges.ndim != 1 or target_edges.size < 2:
        raise ValueError(f"rebin_weights[{name}]: need ≥ 2 target edges")

    N_target = target_edges.size - 1
    N_source = source_grid.size
    W = np.zeros((N_target, N_source), dtype=np.float64)
    for i in range(N_target):
        lo, hi = target_edges[i], target_edges[i + 1]
        mask = (source_grid >= lo - tol) & (source_grid <= hi + tol)
        if mask.sum() < 2:
            raise ValueError(
                f"rebin_weights[{name}]: bin [{lo}, {hi}] has only "
                f"{mask.sum()} source samples; need ≥ 2"
            )
        sub_grid = source_grid[mask]
        w_sub = simpson_weights(sub_grid)
        W[i, mask] = w_sub
    return W


def rebin_axis_tf(values, axis, weights):
    """Apply rebin weights along ``axis`` of ``values``.

    ``values`` shape ``(..., N_source, ...)`` (source axis at position ``axis``).
    ``weights`` shape ``(N_target, N_source)``. Returns shape
    ``(..., N_target, ...)`` with the source axis replaced by the target axis.
    """
    values = _as_dtype(values)
    weights = _as_dtype(weights)
    rank = len(values.shape)
    if axis < 0:
        axis += rank
    # tensordot contracts values[axis] with weights[1]; target axis ends up
    # at the END of the result. Permute it back to position ``axis``.
    out = tf.tensordot(values, weights, axes=[[axis], [1]])
    perm = list(range(rank - 1))
    perm.insert(axis, rank - 1)
    return tf.transpose(out, perm)
