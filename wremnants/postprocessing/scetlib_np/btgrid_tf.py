"""TensorFlow port of the bT-grid factorization library.

Mirrors :mod:`scetlib_btgrid_numpy` function-by-function. The numpy module is
the byte-for-byte transcription of SCETlib C++ and the parity test in
:mod:`scetlib_btgrid_tf_parity` keeps the two in sync.

Design choices:
  * ``np_model`` / ``np_model_nu`` strings are fixed at trace time (the SCETlib
    runcard sets them once per fit). The TF functions dispatch on the string at
    Python level — no ``tf.cond``.
  * λ parameters are TF tensors (typically scalars, but broadcasting follows
    the same rules as numpy).
  * All ops are differentiable in λ. Branches on λ values use ``tf.where``
    with a safe denominator to avoid NaN gradients.
  * ``b_star_global`` is not ported — the cached ``b_bar`` array in the bT-grid
    shards is precomputed and travels as a ``tf.constant``.
  * Simpson weights are precomputed at trace time from the (static) bT, Y, qT
    grids; the runtime cost is just ``tf.reduce_sum(w * y)``.
"""

from typing import Mapping

import numpy as np
import tensorflow as tf

# Set the dtype used for all ops in this module. Match the numpy reference
# (which uses ``float`` ≡ float64) to keep parity tight.
DTYPE = tf.float64


def _as_dtype(x, dtype=DTYPE):
    """Coerce ``x`` to ``dtype`` without losing precision on Python scalars.

    ``tf.cast(0.4, tf.float64)`` round-trips through float32 (returning
    ``0.4000000059604645``); ``tf.constant(0.4, dtype=tf.float64)`` does not.
    Use this helper everywhere a possibly-Python-float input enters the graph.
    """
    if isinstance(x, (int, float)):
        return tf.constant(x, dtype=dtype)
    return tf.cast(x, dtype)


# =============================================================================
# Simpson on a static 1-D non-uniform grid.
# =============================================================================


def simpson_weights(x):
    """Return weights ``w`` such that Simpson(y, x) == sum(w * y, axis=-1).

    ``x`` is a numpy array with size ``N``. Implementation mirrors the numpy
    ``simpson`` in :mod:`scetlib_btgrid_numpy` (composite Simpson with
    trapezoid fallback on the last segment when N-1 is odd).
    """
    x = np.asarray(x, dtype=np.float64)
    n_intervals = x.size - 1
    if n_intervals < 1:
        return np.zeros_like(x)

    if n_intervals % 2 == 1:
        # leading n-1 intervals get Simpson, last segment gets trapezoid
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

EFF_MODELS = {
    "identity",
    "tanh_2",
    "tanh_6",
    "tanh_4",
    "frac_2",
    "frac_4",
    "exp_2",
    "exp_4",
    "signed_lambda",
    "hyp_tangent",
    "square_root",
}
GNU_MODELS = {
    "tanh_1",
    "tanh_2",
    "tanh_6",
    "frac_1",
    "frac_2",
    "exp_1",
    "exp_2",
    "hyp_tangent",
    "linear",
}


def _safe_div(num, den):
    """``num / den`` with the denominator clamped away from zero, masked by
    ``tf.where`` at the call site. Keeps gradients finite."""
    den_safe = tf.where(tf.equal(den, 0), tf.ones_like(den), den)
    return num / den_safe


def F_eff_tf(Y, bT, *, lambda_inf, lambda2, lambda4, lambda6, delta_lambda2, np_model):
    """TF port of :func:`scetlib_btgrid_numpy.F_eff` for a fixed ``np_model``."""
    if np_model not in EFF_MODELS:
        raise ValueError(f"F_eff_tf: unsupported np_model {np_model!r}")

    bT = _as_dtype(bT)
    Y = _as_dtype(Y)
    lambda_inf = _as_dtype(lambda_inf)
    lambda2 = _as_dtype(lambda2)
    lambda4 = _as_dtype(lambda4)
    lambda6 = _as_dtype(lambda6)
    delta_lambda2 = _as_dtype(delta_lambda2)

    if np_model == "signed_lambda":
        lambda2_Y = lambda2 + delta_lambda2 * Y * Y
        return (1.0 + lambda2_Y * bT**2) ** 2 * tf.exp(-2.0 * lambda4 * bT**4)

    lambda2_Y = lambda2 + delta_lambda2 * Y * Y
    arg = (lambda2_Y + lambda4 * bT**2) * bT

    if np_model == "identity":
        return tf.exp(-2.0 * bT * arg)

    # lambda_inf == 0 returns ones (matches numpy short-circuit). We compute
    # the full formula with a safe denominator and mask at the end.
    arg_inf = _safe_div(arg, lambda_inf)
    model = {"hyp_tangent": "tanh_2", "square_root": "frac_2"}.get(np_model, np_model)

    if model == "tanh_2":
        a = arg_inf + (1.0 / 3.0) * _safe_div(lambda2_Y * bT, lambda_inf) ** 3
        func = tf.tanh(a)
    elif model == "tanh_6":
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
    return tf.where(tf.equal(lambda_inf, 0), tf.ones_like(full), full)


def gamma_nu_NP_tf(bT, *, lambda_inf_nu, lambda2_nu, lambda4_nu, np_model_nu):
    """TF port of :func:`scetlib_btgrid_numpy.gamma_nu_NP` for fixed ``np_model_nu``."""
    if np_model_nu not in GNU_MODELS:
        raise ValueError(f"gamma_nu_NP_tf: unsupported np_model_nu {np_model_nu!r}")

    bT = _as_dtype(bT)
    lambda_inf_nu = _as_dtype(lambda_inf_nu)
    lambda2_nu = _as_dtype(lambda2_nu)
    lambda4_nu = _as_dtype(lambda4_nu)

    bT2 = bT * bT
    arg = _safe_div((lambda2_nu + lambda4_nu * bT2) * bT2, lambda_inf_nu)

    model = {"hyp_tangent": "tanh_2", "linear": "frac_1"}.get(np_model_nu, np_model_nu)

    if model == "tanh_1":
        a = arg + (2.0 / 3.0) * _safe_div(lambda2_nu * bT2, lambda_inf_nu) ** 2
        func = tf.tanh(tf.sqrt(a)) ** 2
    elif model == "tanh_2":
        func = tf.tanh(arg)
    elif model == "tanh_6":
        # NP_model_gammanu hardcodes lambda6_nu = 0.0007 (Gamma_nu.hpp:102)
        a = arg + _safe_div(0.0007 * bT2**3, lambda_inf_nu)
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
    return tf.where(tf.equal(lambda_inf_nu, 0), tf.zeros_like(full), full)


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
    """TF port of :func:`scetlib_btgrid_numpy.reconstruct_batch`.

    Implements the per-(Q, Y, qT) bT-space integrand. The full formula with
    every factor and its bare-bT / b*(bT) / (Q,Y,qT) / λ dependence is written
    out once in the :mod:`param_model` module docstring (single source of
    truth) — consult it rather than re-deriving the factors from this code.

    All array-shape arguments are TF tensors or numpy arrays (will be cast).
    The λ values inside ``eff_params`` / ``gnu_params`` are the differentiable
    parameters — pass them as TF scalars (Variables or constants).

    ``bT_simpson_weights`` and ``bT_J0_kernel`` are optional precomputed
    constants. Pass them in the ParamModel to avoid recomputing per-step.

    ``Y_unique`` / ``Y_inverse_idx`` are an optional precomputed unique-Y map
    (``Y_unique`` = sorted distinct Y values, shape ``(NY,)``; ``Y_inverse_idx``
    = per-bin index into ``Y_unique``, shape ``(Nbins,)``). ``F_eff`` depends on
    the bin only through Y, so when this map is supplied the NP transcendentals
    are evaluated on the ``NY`` unique rows and gathered back to ``(Nbins, Nbt)``
    — bit-for-bit identical to the per-bin evaluation, but the expensive ops and
    their λ-gradients run on ``NY`` rows instead of ``Nbins`` (Q and qT don't
    enter ``F_eff``). Without the map it falls back to the full per-bin path.
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
        # bT is a tf.Tensor here; convert to numpy for the Python-side weights
        bT_simpson_weights = simpson_weights(np.asarray(bT))
    bT_simpson_weights = _as_dtype(bT_simpson_weights)  # (Nbt,)

    g_NP = gamma_nu_NP_tf(b_bar, **gnu_params, np_model_nu=np_model_nu)  # (Nbt,)
    exp_g_factor = tf.exp(C_nu * g_NP[tf.newaxis, :])  # (Nbins, Nbt)

    delta_l2_in = eff_params.get("delta_lambda2", 0.0)
    if isinstance(delta_l2_in, (int, float)) and float(delta_l2_in) == 0.0:
        # static fast path: F_eff has no Y dependence
        Feff = F_eff_tf(0.0, b_bar, **eff_params, np_model=np_model)  # (Nbt,)
        Feff_b = Feff[tf.newaxis, :]
    elif Y_unique is not None and Y_inverse_idx is not None:
        # per-bin F_eff, but evaluated on the unique Y rows then gathered.
        # Exact: identical Y -> identical F_eff row for any λ; the gather only
        # replicates rows (its backward scatter-adds the cotangents, so λ-grads
        # are unchanged). Transcendentals run on (NY, Nbt), not (Nbins, Nbt).
        Y_u = _as_dtype(Y_unique)[:, tf.newaxis]  # (NY, 1)
        b_b = b_bar[tf.newaxis, :]
        Feff_u = F_eff_tf(Y_u, b_b, **eff_params, np_model=np_model)  # (NY, Nbt)
        Feff_b = tf.gather(Feff_u, Y_inverse_idx)  # (Nbins, Nbt)
    else:
        # per-bin F_eff due to Y dependence (delta_lambda2 * Y^2)
        # build (Nbins, Nbt) by broadcasting Y_per_bin over bT
        Y_b = Y_per_bin[:, tf.newaxis]
        b_b = b_bar[tf.newaxis, :]
        Feff_b = F_eff_tf(Y_b, b_b, **eff_params, np_model=np_model)  # (Nbins, Nbt)

    integrand = bT_J0_kernel * I_pert * exp_g_factor * Feff_b  # (Nbins, Nbt)
    sigma = simpson_tf(integrand, bT_simpson_weights)  # (Nbins,)

    # qT factor from SCETlib's x = qT*bT integration convention.
    return qT_per_bin * sigma


def build_bT_J0_kernel(qT_per_bin, bT):
    """Precompute ``bT * J_0(qT*bT)`` on the (Nbins, Nbt) grid.

    λ-independent — call once at ParamModel construction and pass into
    :func:`reconstruct_batch_tf` as ``bT_J0_kernel``.
    """
    qT_per_bin = _as_dtype(qT_per_bin)
    bT = _as_dtype(bT)
    arg = qT_per_bin[:, tf.newaxis] * bT[tf.newaxis, :]
    return bT[tf.newaxis, :] * tf.math.special.bessel_j0(arg)
