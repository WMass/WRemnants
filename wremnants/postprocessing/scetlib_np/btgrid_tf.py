"""TensorFlow bT-grid factorization library.

Differentiable TF transcription of the SCETlib bT-space form factors and Hankel
reconstruction (from the SCETlib C++).

  * ``np_model`` / ``np_model_nu`` fixed at trace time (set once per fit by the
    runcard); functions dispatch on the string at Python level, not ``tf.cond``.
  * λ parameters are TF tensors (typically scalars; usual broadcasting).
  * All ops differentiable in λ; branches on λ use ``tf.where`` with a safe
    denominator to avoid NaN gradients.
  * ``b_star_global`` not ported: the precomputed ``b_bar`` shard array travels
    as a ``tf.constant``.
  * Simpson weights precomputed at trace time from the static bT, Y, qT grids;
    runtime cost is ``tf.reduce_sum(w * y)``.
"""

from typing import Mapping

import numpy as np
import tensorflow as tf

# Valid-name sets + alias maps from the numpy-only params module (single source).
# The form branches below must read exactly the λ each model lists in the params
# registry (EFF_MODEL_PARAMS / GNU_MODEL_PARAMS).
from wremnants.postprocessing.scetlib_np.params import (
    EFF_MODELS,
    GNU_MODELS,
    _EFF_MODEL_ALIASES,
    _GNU_MODEL_ALIASES,
)

# float64 throughout this module.
DTYPE = tf.float64


def _as_dtype(x, dtype=DTYPE):
    """Coerce ``x`` to ``dtype`` without losing precision on Python scalars.

    ``tf.cast(0.4, tf.float64)`` round-trips through float32
    (``0.4000000059604645``); ``tf.constant(0.4, dtype=tf.float64)`` does not.
    Use wherever a possibly-Python-float input enters the graph.
    """
    if isinstance(x, (int, float)):
        return tf.constant(x, dtype=dtype)
    return tf.cast(x, dtype)


# =============================================================================
# Simpson on a static 1-D non-uniform grid.
# =============================================================================


def simpson_weights(x):
    """Weights ``w`` such that Simpson(y, x) == sum(w * y, axis=-1).

    ``x`` is a size-``N`` numpy array. Composite Simpson, trapezoid fallback on
    the last segment when N-1 is odd.
    """
    x = np.asarray(x, dtype=np.float64)
    n_intervals = x.size - 1
    if n_intervals < 1:
        return np.zeros_like(x)

    if n_intervals % 2 == 1:
        # leading n-1 intervals Simpson, last segment trapezoid
        w_lead = simpson_weights(x[:-1])
        w = np.concatenate([w_lead, [0.0]])
        h_last = x[-1] - x[-2]
        w[-2] += 0.5 * h_last
        w[-1] += 0.5 * h_last
        return w

    h = np.diff(x)
    h0 = h[0::2]
    h1 = h[1::2]
    coef = (h0 + h1) / 6.0
    w_left = coef * (2.0 - h1 / h0)
    w_mid = coef * (h0 + h1) ** 2 / (h0 * h1)
    w_right = coef * (2.0 - h0 / h1)

    w = np.zeros_like(x)
    w[0:-1:2] += w_left
    w[1::2] += w_mid
    w[2::2] += w_right
    return w


def simpson_tf(y, weights):
    """Simpson reduction along the last axis using precomputed ``weights``."""
    weights = tf.cast(weights, y.dtype)
    return tf.reduce_sum(y * weights, axis=-1)


# =============================================================================
# F_eff and gamma_nu^NP — TF transcriptions
# =============================================================================

# EFF_MODELS / GNU_MODELS (valid np_model names) come from params (imported above).


def _frozen_eq_zero(x):
    """``x == 0`` as a gradient-frozen condition for the NP-factor masks.

    The comparison is a non-differentiable, measure-zero boundary the
    surrounding ``tf.where`` never differentiates through, so freezing its input
    changes no value or derivative. REQUIRED for the full-K Hessian: the
    straight-through ``K`` path nests two ``ForwardAccumulator``s
    (forward-over-forward AD), and the JVP of an ``Equal`` op fed a
    tangent-carrying input raises ``IndexError: list index out of range`` under
    ``@tf.function`` (a TF nested-forward-mode bug). Frozen input → no tangent
    reaches the comparison; the ``tf.where`` JVP with a constant condition is
    fine. (The GN/J-only path uses one ``ForwardAccumulator``, where ``Equal``'s
    JVP is fine.)"""
    return tf.equal(tf.stop_gradient(x), 0)


def _safe_div(num, den):
    """``num / den`` with the denominator clamped to 1 where exactly zero.

    Equivalent to ``num / tf.where(den == 0, 1, den)`` with a frozen comparison
    input (see :func:`_frozen_eq_zero`) so the full-K nested forward-mode Hessian
    doesn't crash under ``@tf.function``. Gradients stay finite; the den==0
    result is masked away by the caller's final ``tf.where``."""
    den_safe = tf.where(_frozen_eq_zero(den), tf.ones_like(den), den)
    return num / den_safe


def F_eff_tf(Y, bT, values, *, np_model):
    """TMD-effective NP form factor F_eff(Y, bT) for a fixed ``np_model``.

    ``values`` maps λ name -> value (TF scalar / Variable / constant, or python
    float). Each ``np_model`` branch reads ONLY the λ its formula uses — a missing
    one raises ``KeyError`` (fail out; no fabricated default). Extra keys (e.g.
    ``np_model``) are ignored. The λ each model reads is declared in
    :data:`params.EFF_MODEL_PARAMS`; keep the two in sync."""
    if np_model not in EFF_MODELS:
        raise ValueError(f"F_eff_tf: unsupported np_model {np_model!r}")

    bT = _as_dtype(bT)
    Y = _as_dtype(Y)
    lambda2 = _as_dtype(values["lambda2"])
    lambda4 = _as_dtype(values["lambda4"])
    delta_lambda2 = _as_dtype(values["delta_lambda2"])
    lambda2_Y = lambda2 + delta_lambda2 * Y * Y

    if np_model == "signed_lambda":
        return (1.0 + lambda2_Y * bT**2) ** 2 * tf.exp(-2.0 * lambda4 * bT**4)

    arg = (lambda2_Y + lambda4 * bT**2) * bT

    if np_model == "identity":
        return tf.exp(-2.0 * bT * arg)

    # lambda_inf == 0 returns ones: compute the full formula with a safe
    # denominator, mask at the end.
    lambda_inf = _as_dtype(values["lambda_inf"])
    arg_inf = _safe_div(arg, lambda_inf)
    model = _EFF_MODEL_ALIASES.get(np_model, np_model)

    if model == "tanh_2":
        a = arg_inf + (1.0 / 3.0) * _safe_div(lambda2_Y * bT, lambda_inf) ** 3
        func = tf.tanh(a)
    elif model == "tanh_6":
        lambda6 = _as_dtype(values["lambda6"])
        a = arg_inf + _safe_div(lambda6 * bT**5, lambda_inf)
        a = a + (1.0 / 3.0) * _safe_div(lambda2_Y * bT, lambda_inf) ** 3
        func = tf.tanh(a)
    elif model == "tanh_4":
        func = tf.sqrt(tf.tanh(arg_inf**2))
    elif model == "frac_2":
        a = arg_inf + 0.5 * _safe_div(lambda2_Y * bT, lambda_inf) ** 3
        func = a / tf.sqrt(1.0 + a**2)
    elif model == "frac_4":
        func = arg_inf / tf.sqrt(tf.sqrt(1.0 + arg_inf**4))
    elif model == "exp_2":
        a = arg_inf + 0.25 * _safe_div(lambda2_Y * bT, lambda_inf) ** 3
        func = tf.sqrt(-tf.math.expm1(-(a**2)))
    elif model == "exp_4":
        func = tf.sqrt(tf.sqrt(-tf.math.expm1(-(arg_inf**4))))
    else:  # pragma: no cover — guarded above
        raise ValueError(f"F_eff_tf: unsupported np_model {np_model!r}")

    full = tf.exp(-2.0 * lambda_inf * bT * func)
    # lambda_inf == 0 -> 1 (NP off); frozen comparison input (see
    # _frozen_eq_zero) for the full-K @tf.function Hessian.
    return tf.where(_frozen_eq_zero(lambda_inf), tf.ones_like(full), full)


def gamma_nu_NP_tf(bT, values, *, np_model_nu):
    """CS-side NP rapidity anomalous dimension γ_ν^NP(bT) for fixed ``np_model_nu``.

    ``values`` maps λ name -> value; each branch reads ONLY the λ its formula uses
    (a missing one raises ``KeyError`` — fail out). Extra keys (e.g.
    ``np_model_nu``) are ignored. The λ each model reads is declared in
    :data:`params.GNU_MODEL_PARAMS`; keep the two in sync.
    """
    if np_model_nu not in GNU_MODELS:
        raise ValueError(f"gamma_nu_NP_tf: unsupported np_model_nu {np_model_nu!r}")

    bT = _as_dtype(bT)
    lambda_inf_nu = _as_dtype(values["lambda_inf_nu"])
    lambda2_nu = _as_dtype(values["lambda2_nu"])
    lambda4_nu = _as_dtype(values["lambda4_nu"])

    bT2 = bT * bT
    arg = _safe_div((lambda2_nu + lambda4_nu * bT2) * bT2, lambda_inf_nu)

    model = _GNU_MODEL_ALIASES.get(np_model_nu, np_model_nu)

    if model == "tanh_1":
        a = arg + (2.0 / 3.0) * _safe_div(lambda2_nu * bT2, lambda_inf_nu) ** 2
        func = tf.tanh(tf.sqrt(a)) ** 2
    elif model == "tanh_2":
        func = tf.tanh(arg)
    elif model == "tanh_6":
        # tanh_2 plus a b⁶ term with fittable coefficient lambda6_nu.
        lambda6_nu = _as_dtype(values["lambda6_nu"])
        a = arg + _safe_div(lambda6_nu * bT2**3, lambda_inf_nu)
        func = tf.tanh(a)
    elif model == "frac_1":
        a = arg + _safe_div(lambda2_nu * bT2, lambda_inf_nu) ** 2
        func = a / (1.0 + a)
    elif model == "frac_2":
        func = arg / tf.sqrt(1.0 + arg**2)
    elif model == "exp_1":
        a = arg + 0.5 * _safe_div(lambda2_nu * bT2, lambda_inf_nu) ** 2
        func = -tf.math.expm1(-a)
    elif model == "exp_2":
        func = tf.sqrt(-tf.math.expm1(-(arg**2)))
    else:  # pragma: no cover
        raise ValueError(f"gamma_nu_NP_tf: unsupported np_model_nu {np_model_nu!r}")

    full = -lambda_inf_nu * func
    # lambda_inf_nu == 0 -> 0 (NP off); frozen comparison input (see
    # _frozen_eq_zero) for the full-K @tf.function Hessian.
    return tf.where(_frozen_eq_zero(lambda_inf_nu), tf.zeros_like(full), full)


# =============================================================================
# Hankel reconstruction (batched, λ-differentiable)
# =============================================================================


def reconstruct_batch_tf(
    qT_per_bin,
    bT,
    I_pert,
    C_nu,
    b_bar,
    Y_per_bin,
    eff_params: Mapping,
    gnu_params: Mapping,
    *,
    np_model: str,
    np_model_nu: str,
    bT_simpson_weights=None,
    bT_J0_kernel=None,
    Y_unique=None,
    Y_inverse_idx=None,
):
    """Reconstruct σ on a batch of (Q, Y, qT) grid points from the bT integrand.

    Implements the per-(Q, Y, qT) bT-space integrand. The full formula with
    every factor and its bare-bT / b*(bT) / (Q,Y,qT) / λ dependence is in the
    :mod:`param_model` module docstring.

    Array-shape arguments are TF tensors or numpy arrays (cast on entry). The λ
    values in ``eff_params`` / ``gnu_params`` are the differentiable parameters;
    pass them as TF scalars (Variables or constants).

    ``bT_simpson_weights`` and ``bT_J0_kernel`` are optional precomputed
    constants; pass them from the ParamModel to avoid recomputing per-step.

    ``Y_unique`` / ``Y_inverse_idx`` are an optional unique-Y map (``Y_unique`` =
    sorted distinct Y, shape ``(NY,)``; ``Y_inverse_idx`` = per-bin index into
    it, shape ``(Nbins,)``). ``F_eff`` depends on the bin only through Y, so the
    NP transcendentals run on the ``NY`` unique rows and gather back to
    ``(Nbins, Nbt)`` — bit-for-bit identical to per-bin, but with the expensive
    ops and their λ-gradients on ``NY`` rows not ``Nbins`` (Q, qT don't enter
    ``F_eff``). Without the map, falls back to the full per-bin path.
    """
    qT_per_bin = _as_dtype(qT_per_bin)  # (Nbins,)
    Y_per_bin = _as_dtype(Y_per_bin)  # (Nbins,)
    bT = _as_dtype(bT)  # (Nbt,)
    b_bar = _as_dtype(b_bar)  # (Nbt,)
    I_pert = _as_dtype(I_pert)  # (Nbins, Nbt)
    C_nu = _as_dtype(C_nu)  # (Nbins, Nbt)

    if bT_J0_kernel is None:
        bT_J0_kernel = build_bT_J0_kernel(qT_per_bin, bT)
    bT_J0_kernel = _as_dtype(bT_J0_kernel)  # (Nbins, Nbt)

    if bT_simpson_weights is None:
        # bT is a tf.Tensor; to numpy for the Python-side weights
        bT_simpson_weights = simpson_weights(np.asarray(bT))
    bT_simpson_weights = _as_dtype(bT_simpson_weights)  # (Nbt,)

    g_NP = gamma_nu_NP_tf(b_bar, gnu_params, np_model_nu=np_model_nu)  # (Nbt,)
    exp_g_factor = tf.exp(C_nu * g_NP[tf.newaxis, :])  # (Nbins, Nbt)

    delta_l2_in = eff_params.get("delta_lambda2", 0.0)
    if isinstance(delta_l2_in, (int, float)) and float(delta_l2_in) == 0.0:
        # static fast path: F_eff has no Y dependence
        Feff = F_eff_tf(0.0, b_bar, eff_params, np_model=np_model)  # (Nbt,)
        Feff_b = Feff[tf.newaxis, :]
    elif Y_unique is not None and Y_inverse_idx is not None:
        # per-bin F_eff on the unique-Y rows, then gathered. Exact: identical Y
        # -> identical F_eff row for any λ; the gather only replicates rows
        # (backward scatter-adds cotangents, λ-grads unchanged). Transcendentals
        # run on (NY, Nbt), not (Nbins, Nbt).
        Y_u = _as_dtype(Y_unique)[:, tf.newaxis]  # (NY, 1)
        b_b = b_bar[tf.newaxis, :]
        Feff_u = F_eff_tf(Y_u, b_b, eff_params, np_model=np_model)  # (NY, Nbt)
        Feff_b = tf.gather(Feff_u, Y_inverse_idx)  # (Nbins, Nbt)
    else:
        # per-bin F_eff (Y dependence via delta_lambda2 * Y^2): build
        # (Nbins, Nbt) by broadcasting Y_per_bin over bT
        Y_b = Y_per_bin[:, tf.newaxis]
        b_b = b_bar[tf.newaxis, :]
        Feff_b = F_eff_tf(Y_b, b_b, eff_params, np_model=np_model)  # (Nbins, Nbt)

    integrand = bT_J0_kernel * I_pert * exp_g_factor * Feff_b  # (Nbins, Nbt)
    sigma = simpson_tf(integrand, bT_simpson_weights)  # (Nbins,)

    # qT factor from SCETlib's x = qT*bT integration convention.
    return qT_per_bin * sigma


def build_bT_J0_kernel(qT_per_bin, bT):
    """Precompute ``bT * J_0(qT*bT)`` on the (Nbins, Nbt) grid.

    λ-independent: call once at ParamModel construction, pass into
    :func:`reconstruct_batch_tf` as ``bT_J0_kernel``.

    The factorized path (:func:`reconstruct_batch_factorized_tf`) instead calls
    it with ``qT_unique`` (NqT distinct values, not the per-bin expansion),
    giving a (NqT, Nbt) kernel: same numbers, ~4000× smaller.
    """
    qT_per_bin = _as_dtype(qT_per_bin)
    bT = _as_dtype(bT)
    arg = qT_per_bin[:, tf.newaxis] * bT[tf.newaxis, :]
    return bT[tf.newaxis, :] * tf.math.special.bessel_j0(arg)


# =============================================================================
# Factorized reconstruction (GPU-memory-safe; exact)
# =============================================================================
#
# reconstruct_batch_tf's (Nbins, Nbt) layout needs several ~9 GB fp64 tensors
# (Nbins=546840, Nbt=2000) and OOMs a 32 GB GPU at construction. Two exact
# observations shrink it:
#
#   1. qT enters the λ-dependent integrand ONLY via the bT·J0(qT·bT) kernel,
#      which needs the NqT *unique* qT values, not Nbins rows.
#   2. SCETlib's profile scales are piecewise in x = qT/Q and exactly canonical
#      (qT-independent) below the first transition x1·Q, so the cached
#      I_pert / C_nu rows are BIT-IDENTICAL across qT there. The dedup below
#      discovers identical rows dynamically (byte-wise hashing + full
#      verification) — no assumption about profiles or qT ranges.
#
# The bT-Simpson reduction then becomes a (Nu, Nbt) @ (Nbt, NqT) matmul plus a
# per-bin gather. Same integrand, weights and sampling; only the floating-point
# grouping/summation order changes (≲1e-14 relative).


def dedup_grid_rows(I_pert, C_nu, feff_idx_per_bin, verbose=True):
    """Find bit-identical (I_pert, C_nu, F_eff-index) row triples.

    Construction-time numpy helper (runs once, CPU). Rows are keyed by the raw
    bytes of the I_pert row, the C_nu row and the per-bin F_eff Y-index, so two
    bins share a unique id iff their λ-dependent integrand columns are
    bit-for-bit identical for EVERY λ. Grouping verified by direct array
    comparison afterward, not hash-collision odds.

    Parameters
    ----------
    I_pert, C_nu : (Nbins, Nbt) float64 ndarrays
    feff_idx_per_bin : (Nbins,) int ndarray
        Index into the unique-Y table for the F_eff gather (Y enters F_eff via
        delta_lambda2·Y²; keying on it keeps per-unique-row F_eff well-defined
        even when delta_lambda2 floats).

    Returns
    -------
    dict with:
        I_u, C_u : (Nu, Nbt) deduplicated rows (copies, C-contiguous)
        row_uid  : (Nbins,) int32, bin -> unique-row index
        feff_idx_u : (Nu,) int32, unique row -> unique-Y index
        n_unique : int
        C_uu     : (Ncu, Nbt) second-level dedup of C_u. C_nu depends on
                   (Q, profile-qT) only, not Y, so its standalone unique-row
                   count is ~150x smaller than Nu (1888 vs 284605 on fineall).
                   The exp(C·g) transcendentals run on these rows, gathered
                   back: bit-identical, ~150x fewer exp() calls, and the
                   (Nu, Nbt) C constant never exists on device.
        c_of_u   : (Nu,) int32, unique row -> C_uu row index
        n_unique_C : int
    """
    import hashlib

    I_pert = np.ascontiguousarray(I_pert)
    C_nu = np.ascontiguousarray(C_nu)
    n_bins = I_pert.shape[0]
    feff_idx_per_bin = np.asarray(feff_idx_per_bin).reshape(-1)
    if feff_idx_per_bin.shape[0] != n_bins:
        raise ValueError("feff_idx_per_bin length != Nbins")

    seen = {}
    row_uid = np.empty(n_bins, dtype=np.int32)
    rep_rows = []  # representative bin index per unique id
    for k in range(n_bins):
        h = hashlib.blake2b(I_pert[k].tobytes(), digest_size=16)
        h.update(C_nu[k].tobytes())
        h.update(int(feff_idx_per_bin[k]).to_bytes(4, "little", signed=True))
        key = h.digest()
        uid = seen.get(key)
        if uid is None:
            uid = len(rep_rows)
            seen[key] = uid
            rep_rows.append(k)
        row_uid[k] = uid

    rep_rows = np.asarray(rep_rows, dtype=np.int64)
    I_u = np.ascontiguousarray(I_pert[rep_rows])
    C_u = np.ascontiguousarray(C_nu[rep_rows])
    feff_idx_u = feff_idx_per_bin[rep_rows].astype(np.int32)

    # Verify bit-exactly: every bin's rows must equal its representative's.
    # Chunked to bound the temporary gather copies.
    chunk = 20000
    for k0 in range(0, n_bins, chunk):
        k1 = min(k0 + chunk, n_bins)
        sel = row_uid[k0:k1]
        if not (
            np.array_equal(I_pert[k0:k1], I_u[sel])
            and np.array_equal(C_nu[k0:k1], C_u[sel])
            and np.array_equal(feff_idx_per_bin[k0:k1], feff_idx_u[sel])
        ):
            raise AssertionError(
                f"dedup_grid_rows: hash grouping failed verification in bins "
                f"[{k0}, {k1}) — this should be impossible; grid corrupt?"
            )

    # Second-level dedup of the C rows (qT-independent below the profile
    # transition AND Y-independent everywhere → ~150x smaller).
    n_u = len(rep_rows)
    seen_c = {}
    c_of_u = np.empty(n_u, dtype=np.int32)
    rep_c = []
    for k in range(n_u):
        key = hashlib.blake2b(C_u[k].tobytes(), digest_size=16).digest()
        cid = seen_c.get(key)
        if cid is None:
            cid = len(rep_c)
            seen_c[key] = cid
            rep_c.append(k)
        c_of_u[k] = cid
    C_uu = np.ascontiguousarray(C_u[np.asarray(rep_c, dtype=np.int64)])
    for k0 in range(0, n_u, chunk):
        k1 = min(k0 + chunk, n_u)
        if not np.array_equal(C_u[k0:k1], C_uu[c_of_u[k0:k1]]):
            raise AssertionError(
                f"dedup_grid_rows: C-row sub-dedup failed verification in rows "
                f"[{k0}, {k1}) — this should be impossible; grid corrupt?"
            )

    if verbose:
        print(
            f"[dedup_grid_rows] {n_bins} bins -> {n_u} unique rows "
            f"({n_bins / n_u:.2f}x dedup, verified bit-exact); "
            f"per-tensor {n_bins * I_pert.shape[1] * 8 / 1e9:.2f} GB -> "
            f"{n_u * I_pert.shape[1] * 8 / 1e9:.2f} GB; "
            f"C_nu sub-dedup {n_u} -> {len(rep_c)} rows "
            f"({n_u / len(rep_c):.0f}x, verified bit-exact)",
            flush=True,
        )
    return dict(
        I_u=I_u,
        C_u=C_u,
        row_uid=row_uid,
        feff_idx_u=feff_idx_u,
        n_unique=n_u,
        C_uu=C_uu,
        c_of_u=c_of_u,
        n_unique_C=len(rep_c),
    )


def reconstruct_batch_factorized_tf(
    b_bar,
    I_pert_u,
    C_nu_u=None,
    eff_params: Mapping = None,
    gnu_params: Mapping = None,
    *,
    np_model: str,
    np_model_nu: str,
    KwqT,
    gather_idx,
    Y_unique,
    feff_idx_u,
    C_nu_uu=None,
    c_of_u=None,
):
    """Memory-factorized, numerically-equivalent form of
    :func:`reconstruct_batch_tf`.

    Evaluates σ_i = qT_i Σ_b w_b·bT_b·J0(qT_i bT_b)·I_{u(i),b}·exp(C_{u(i),b}
    g_b)·F_{y(u(i)),b} as a (Nu, Nbt) elementwise block, a matmul against the
    weighted J0 kernel on the unique-qT grid, and a per-bin gather; no
    (Nbins, Nbt) tensor is materialized. Same integrand, weights and sampling as
    :func:`reconstruct_batch_tf`; only the floating-point multiplication grouping
    and summation order differ (≲1e-14 relative).

    Parameters
    ----------
    b_bar : (Nbt,) — b*(bT), the NP-factor argument
    I_pert_u : (Nu, Nbt) — deduplicated grid rows
        (from :func:`dedup_grid_rows`)
    C_nu_u : (Nu, Nbt), optional
        Per-unique-row C_ν. Pass this OR (``C_nu_uu``, ``c_of_u``); the latter
        is preferred (~150x fewer exp() calls, no (Nu, Nbt) C constant on
        device). Bit-identical: exp of identical rows is identical, gather only
        replicates rows.
    KwqT : (NqT, Nbt)
        ``qT_u · bT · J0(qT_u·bT) · w_simpson`` on the unique-qT grid; folds in
        reconstruct_batch_tf's per-bin qT prefactor.
    gather_idx : (Nbins, 2) int32 — per bin ``[u(i), qT_index(i)]``
    Y_unique : (NY,) — unique Y values for the F_eff evaluation
    feff_idx_u : (Nu,) int32 — unique row -> Y_unique index
    C_nu_uu : (Ncu, Nbt), optional — second-level deduplicated C_ν rows
    c_of_u : (Nu,) int32, optional — unique row -> C_uu row index
    """
    b_bar = _as_dtype(b_bar)
    I_pert_u = _as_dtype(I_pert_u)
    KwqT = _as_dtype(KwqT)

    g_NP = gamma_nu_NP_tf(b_bar, gnu_params, np_model_nu=np_model_nu)  # (Nbt,)
    if C_nu_uu is not None and c_of_u is not None:
        # exp on the ~150x smaller C-row table, replicated by the gather.
        exp_g_uu = tf.exp(_as_dtype(C_nu_uu) * g_NP[tf.newaxis, :])  # (Ncu, Nbt)
        exp_g_u = tf.gather(exp_g_uu, c_of_u)  # (Nu, Nbt)
    elif C_nu_u is not None:
        exp_g_u = tf.exp(_as_dtype(C_nu_u) * g_NP[tf.newaxis, :])  # (Nu, Nbt)
    else:
        raise ValueError(
            "reconstruct_batch_factorized_tf: pass either C_nu_u or "
            "(C_nu_uu, c_of_u)"
        )

    delta_l2_in = eff_params.get("delta_lambda2", 0.0)
    if isinstance(delta_l2_in, (int, float)) and float(delta_l2_in) == 0.0:
        # static fast path: F_eff has no Y dependence (matches the
        # reconstruct_batch_tf fast path bit-for-bit on the unique rows)
        Feff = F_eff_tf(0.0, b_bar, eff_params, np_model=np_model)  # (Nbt,)
        Feff_u = Feff[tf.newaxis, :]  # broadcast over Nu
    else:
        # F_eff on the unique-Y rows, gathered to the unique grid rows (same
        # unique-Y trick as reconstruct_batch_tf: gather replicates rows
        # bit-exactly, scatter-adds cotangents on the backward pass).
        Y_u = _as_dtype(Y_unique)[:, tf.newaxis]  # (NY, 1)
        Feff_rows = F_eff_tf(
            Y_u, b_bar[tf.newaxis, :], eff_params, np_model=np_model
        )  # (NY, Nbt)
        Feff_u = tf.gather(Feff_rows, feff_idx_u)  # (Nu, Nbt)

    M = I_pert_u * exp_g_u * Feff_u  # (Nu, Nbt)
    S = tf.matmul(M, KwqT, transpose_b=True)  # (Nu, NqT)
    return tf.gather_nd(S, gather_idx)  # (Nbins,)
