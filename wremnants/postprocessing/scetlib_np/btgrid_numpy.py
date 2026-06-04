# -------------------------------------------------------------------------------
# NP factorization library for the bT-grid workflow.
#
# Vendored copy of:
#   /work/submit/lavezzo/alphaS/scetlib-cms-newnp-lambda4fix/prod/scetlib_run/
#       scetlib_run/factorize.py
# Numpy-only; serves as the reference implementation against which the TF port
# (scetlib_btgrid_tf.py, Phase 2) is parity-tested. Kept in sync manually with
# the upstream scetlib repo — when the upstream changes, recopy this file and
# rerun the parity tests.
#
# Provides:
#   - Pure-numpy transcriptions of NP_model_effective (F_eff) and
#     NP_model_gammanu (gamma_nu^NP) that match the C++ code byte-for-byte.
#   - A vectorised Hankel reconstruction of sigma(qT) from a cached bT-grid.
#     The full integrand — every factor and its bare-bT / b*(bT) / (Q,Y,qT) /
#     lambda dependence — is written out ONCE in the module docstring of
#     wremnants/postprocessing/scetlib_np/param_model.py (single source of
#     truth). The reconstruct_* functions below implement it.
#   - Loaders for the bT-grid pickle shards produced by --bt-grid and for the
#     prior-art spectrum-mode "combined" pickles.
#
# Self-contained: depends on numpy only, no scipy / SCETlib runtime at import.
# -------------------------------------------------------------------------------

import glob
import os
import pickle
import sys

import numpy as np


# Compat shim: pickles produced with numpy >= 2.0 reference `numpy._core`,
# which does not exist in numpy < 2.0. Alias the old `numpy.core` under the
# new name (and a few of its submodules) so unpickling succeeds.
def _ensure_numpy_core_alias():
    if "numpy._core" in sys.modules:
        return
    try:
        import numpy._core  # noqa: F401
    except ImportError:
        try:
            from numpy import core as _np_core
        except ImportError:
            return
        sys.modules["numpy._core"] = _np_core
        for name in (
            "multiarray",
            "numeric",
            "fromnumeric",
            "umath",
            "shape_base",
            "_methods",
        ):
            sub = getattr(_np_core, name, None)
            if sub is not None:
                sys.modules[f"numpy._core.{name}"] = sub


_ensure_numpy_core_alias()


# =============================================================================
# Numerics: bare-bones bessel J0 and Simpson, both numpy-only (the container
# in which the fit runs has no scipy).
# =============================================================================


def bessel_j0(x):
    """J_0(x) via Abramowitz & Stegun 9.4.1 / 9.4.3. Accurate to ~1.6e-8."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 3.0
    xs = x[small] / 3.0
    y = xs * xs
    out[small] = (
        1.0
        - 2.2499997 * y
        + 1.2656208 * y**2
        - 0.3163866 * y**3
        + 0.0444479 * y**4
        - 0.0039444 * y**5
        + 0.0002100 * y**6
    )
    xl = np.abs(x[~small])
    z = 3.0 / xl
    f0 = (
        0.79788456
        - 0.00000077 * z
        - 0.00552740 * z**2
        - 0.00009512 * z**3
        + 0.00137237 * z**4
        - 0.00072805 * z**5
        + 0.00014476 * z**6
    )
    theta0 = (
        xl
        - 0.78539816
        - 0.04166397 * z
        - 0.00003954 * z**2
        + 0.00262573 * z**3
        - 0.00054125 * z**4
        - 0.00029333 * z**5
        + 0.00013558 * z**6
    )
    out[~small] = f0 * np.cos(theta0) / np.sqrt(xl)
    return out


def simpson(y, x, axis=-1):
    """Composite Simpson on a (possibly non-uniform) 1-D grid, vectorised
    along the given axis. Falls back to trapezoid for the last segment when
    the number of intervals is odd."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("simpson expects 1-D x")
    n = x.size - 1
    if n < 1:
        return np.zeros(
            y.shape[:-1] if axis == -1 else y.shape[:axis] + y.shape[axis + 1 :]
        )
    # move integration axis to the end for simpler slicing
    y_moved = np.moveaxis(y, axis, -1)
    if n % 2 == 1:
        lead = simpson(np.moveaxis(y_moved[..., :-1], -1, axis), x[:-1], axis=axis)
        tail = 0.5 * (y_moved[..., -1] + y_moved[..., -2]) * (x[-1] - x[-2])
        return lead + tail
    h = np.diff(x)
    h0 = h[0::2]
    h1 = h[1::2]
    s = (
        (h0 + h1)
        / 6.0
        * (
            y_moved[..., 0:-1:2] * (2.0 - h1 / h0)
            + y_moved[..., 1::2] * (h0 + h1) ** 2 / (h0 * h1)
            + y_moved[..., 2::2] * (2.0 - h0 / h1)
        )
    )
    return np.sum(s, axis=-1)


# =============================================================================
# NP model transcriptions. Mirror the C++ implementations one-to-one; expect
# bT to be the b̄T (= b_star) at which the NP factor enters.
# =============================================================================

# NP_model_effective.np_model enum values supported here:
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

# NP_model_gammanu.np_model_nu enum values supported here:
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


def F_eff(Y, bT, *, lambda_inf, lambda2, lambda4, lambda6, delta_lambda2, np_model):
    """F_eff(Y, b̄T) — NP_model_effective::operator() from NP_models.hpp."""
    bT = np.asarray(bT, dtype=float)

    if np_model == "signed_lambda":
        lambda2_Y = lambda2 + delta_lambda2 * Y * Y
        if lambda4 <= 0.0 and (lambda2 != 0.0 or delta_lambda2 != 0.0):
            raise ValueError(
                "signed_lambda requires lambda4 > 0 when lambda2 or delta_lambda2 != 0"
            )
        return (1.0 + lambda2_Y * bT**2) ** 2 * np.exp(-2.0 * lambda4 * bT**4)

    lambda2_Y = lambda2 + delta_lambda2 * Y * Y
    arg = (lambda2_Y + lambda4 * bT**2) * bT

    if np_model == "identity":
        return np.exp(-2.0 * bT * arg)

    if lambda_inf == 0.0:
        return np.ones_like(bT)

    arg = arg / lambda_inf
    # alias support
    model = {"hyp_tangent": "tanh_2", "square_root": "frac_2"}.get(np_model, np_model)

    if model == "tanh_2":
        arg = arg + (1.0 / 3.0) * (lambda2_Y * bT / lambda_inf) ** 3
        func = np.tanh(arg)
    elif model == "tanh_6":
        arg = arg + lambda6 * bT**5 / lambda_inf
        arg = arg + (1.0 / 3.0) * (lambda2_Y * bT / lambda_inf) ** 3
        func = np.tanh(arg)
    elif model == "tanh_4":
        func = np.sqrt(np.tanh(arg**2))
    elif model == "frac_2":
        arg = arg + 0.5 * (lambda2_Y * bT / lambda_inf) ** 3
        func = arg / np.sqrt(1.0 + arg**2)
    elif model == "frac_4":
        func = arg / np.sqrt(np.sqrt(1.0 + arg**4))
    elif model == "exp_2":
        arg = arg + 0.25 * (lambda2_Y * bT / lambda_inf) ** 3
        func = np.sqrt(-np.expm1(-(arg**2)))
    elif model == "exp_4":
        func = np.sqrt(np.sqrt(-np.expm1(-(arg**4))))
    else:
        raise ValueError(f"F_eff: unsupported np_model {np_model!r}")

    return np.exp(-2.0 * lambda_inf * bT * func)


def gamma_nu_NP(bT, *, lambda_inf_nu, lambda2_nu, lambda4_nu, np_model_nu):
    """gamma_nu^NP(b̄T) — NP_model_gammanu::model_gammanu() from Gamma_nu.hpp.

    Note: NP_model_gammanu has its own b_star() (b0_bmax_nu), but the C++
    code calls model_gammanu(bT) with the raw bT passed into Gamma_nu — which
    is b_star_global by the time it reaches us. So pass b̄T = b_star_global(bT)
    here; b0_bmax_nu does NOT enter model_gammanu directly.
    """
    bT = np.asarray(bT, dtype=float)
    if lambda_inf_nu == 0.0:
        return np.zeros_like(bT)

    bT2 = bT * bT
    arg = (lambda2_nu + lambda4_nu * bT2) * bT2 / lambda_inf_nu

    model = {"hyp_tangent": "tanh_2", "linear": "frac_1"}.get(np_model_nu, np_model_nu)

    if model == "tanh_1":
        arg = arg + (2.0 / 3.0) * (lambda2_nu * bT2 / lambda_inf_nu) ** 2
        func = np.tanh(np.sqrt(arg)) ** 2
    elif model == "tanh_2":
        func = np.tanh(arg)
    elif model == "tanh_6":
        # NP_model_gammanu hardcodes lambda6_nu = 0.0007 (Gamma_nu.hpp:102)
        arg = arg + 0.0007 * bT2**3 / lambda_inf_nu
        func = np.tanh(arg)
    elif model == "frac_1":
        arg = arg + (lambda2_nu * bT2 / lambda_inf_nu) ** 2
        func = arg / (1.0 + arg)
    elif model == "frac_2":
        func = arg / np.sqrt(1.0 + arg**2)
    elif model == "exp_1":
        arg = arg + 0.5 * (lambda2_nu * bT2 / lambda_inf_nu) ** 2
        func = -np.expm1(-arg)
    elif model == "exp_2":
        func = np.sqrt(-np.expm1(-(arg**2)))
    else:
        raise ValueError(f"gamma_nu_NP: unsupported np_model_nu {np_model_nu!r}")

    return -lambda_inf_nu * func


# Convenience defaults: "all NP knobs off". These reproduce the SCETlib
# configuration that --bt-grid mode caches.
NP_ZERO_EFF = dict(
    lambda_inf=0.0,
    lambda2=0.0,
    lambda4=0.0,
    lambda6=0.0,
    delta_lambda2=0.0,
    np_model="identity",
)
NP_ZERO_GNU = dict(
    lambda_inf_nu=0.0, lambda2_nu=0.0, lambda4_nu=0.0, np_model_nu="tanh_2"
)


def eff_params_from_conf(conf):
    """Read NP_model_effective parameters from a configparser [Nonperturbative]
    section (defaults to NP_ZERO_EFF if a field is absent)."""
    if "Nonperturbative" not in conf:
        return dict(NP_ZERO_EFF)
    sec = conf["Nonperturbative"]
    return dict(
        lambda_inf=sec.getfloat("lambda_inf", fallback=NP_ZERO_EFF["lambda_inf"]),
        lambda2=sec.getfloat("lambda2", fallback=NP_ZERO_EFF["lambda2"]),
        lambda4=sec.getfloat("lambda4", fallback=NP_ZERO_EFF["lambda4"]),
        lambda6=sec.getfloat("lambda6", fallback=NP_ZERO_EFF["lambda6"]),
        delta_lambda2=sec.getfloat(
            "delta_lambda2", fallback=NP_ZERO_EFF["delta_lambda2"]
        ),
        np_model=sec.get("np_model", fallback=NP_ZERO_EFF["np_model"]),
    )


def gnu_params_from_conf(conf):
    """Read NP_model_gammanu parameters from a configparser [Nonperturbative]
    section (defaults to NP_ZERO_GNU if a field is absent)."""
    if "Nonperturbative" not in conf:
        return dict(NP_ZERO_GNU)
    sec = conf["Nonperturbative"]
    return dict(
        lambda_inf_nu=sec.getfloat(
            "lambda_inf_nu", fallback=NP_ZERO_GNU["lambda_inf_nu"]
        ),
        lambda2_nu=sec.getfloat("lambda2_nu", fallback=NP_ZERO_GNU["lambda2_nu"]),
        lambda4_nu=sec.getfloat("lambda4_nu", fallback=NP_ZERO_GNU["lambda4_nu"]),
        np_model_nu=sec.get("np_model_nu", fallback=NP_ZERO_GNU["np_model_nu"]),
    )


# =============================================================================
# Hankel reconstruction
# =============================================================================


def reconstruct_one(qT, bT, I_pert, C_nu, b_bar, Y, eff_params, gnu_params):
    """Hankel-reconstruct one sigma(qT) from cached arrays at a single (Q,Y,qT).

    Returns the differential structure-function value at the point (Q,Y,qT),
    matching SCETlib's spectrum-mode ang.c convention (i.e. including the
    qT factor that arises from the integration-variable choice in SCETlib's
    integrator_de_oscillatory).

    qT       : scalar
    bT       : (Nb,) integration variable (raw bT)
    I_pert   : (Nb,) cached SCETlib integrand at NP off
    C_nu     : (Nb,) coefficient of gamma_nu^NP in log evolution
    b_bar    : (Nb,) b_star_global(bT) where NP factors evaluate
    Y        : scalar rapidity
    eff_params : dict for NP_model_effective parameters
    gnu_params : dict for NP_model_gammanu parameters
    """
    g_NP = gamma_nu_NP(b_bar, **gnu_params)
    Feff = F_eff(Y, b_bar, **eff_params)
    integrand = bT * bessel_j0(qT * bT) * I_pert * np.exp(C_nu * g_NP) * Feff
    return qT * simpson(integrand, bT)


def reconstruct_grid_QYqT(
    Q_grid,
    Y_grid,
    qT_grid,
    bT,
    I_pert,
    C_nu,
    b_bar,
    eff_params,
    gnu_params,
    verbose=True,
):
    """Reconstruct sigma at every (Q, Y, qT) sample point of a regularly-indexed
    grid. Computes J0(qT*bT) once on (NqT, Nbt) (rather than the redundant
    (NQ*NY*NqT, Nbt) the naive batch would build) and processes one Q-slice at
    a time to bound peak memory and emit progress.

    Q_grid   : (NQ,)   point values in Q
    Y_grid   : (NY,)   point values in Y
    qT_grid  : (NqT,)  point values in qT
    bT       : (Nbt,)  bT integration variable shared by all points
    I_pert   : (Npts, Nbt) cached perturbative integrand, Npts = NQ*NY*NqT,
               indexed as flat list in the same order load_btgrid_shards
               returns its `bins` list (sorted lexicographically by Q, Y, qT).
    C_nu     : (Npts, Nbt) rapidity-evolution-log coefficient
    b_bar    : (Nbt,)  b_star_global(bT)

    Returns: (NQ, NY, NqT) ndarray of sigma_factorised values.
    """
    Q_grid = np.asarray(Q_grid, dtype=float)
    Y_grid = np.asarray(Y_grid, dtype=float)
    qT_grid = np.asarray(qT_grid, dtype=float)
    bT = np.asarray(bT, dtype=float)
    NQ, NY, NqT = Q_grid.size, Y_grid.size, qT_grid.size
    Nbt = bT.size
    Npts = NQ * NY * NqT
    if I_pert.shape != (Npts, Nbt):
        raise ValueError(
            f"I_pert shape {I_pert.shape} doesn't match expected "
            f"({Npts}, {Nbt}) from {NQ} Q x {NY} Y x {NqT} qT"
        )

    # --- shared (Y-independent) factors over bT ---
    g_NP = gamma_nu_NP(b_bar, **gnu_params)  # (Nbt,)
    delta_l2 = eff_params.get("delta_lambda2", 0.0)
    if delta_l2 == 0.0:
        Feff_bT = F_eff(0.0, b_bar, **eff_params)  # (Nbt,)
    else:
        Feff_bT = None  # per-Y below

    # --- bT*J0(qT*bT) cached once over (NqT, Nbt) ---
    bT_J0 = bT[np.newaxis, :] * bessel_j0(qT_grid[:, np.newaxis] * bT[np.newaxis, :])
    # shape (NqT, Nbt); used in every Q-slice below.

    # I_pert and C_nu are stored as (NQ*NY*NqT, Nbt); reshape view to
    # (NQ, NY, NqT, Nbt) and process Q-by-Q.
    I_pert_r = I_pert.reshape(NQ, NY, NqT, Nbt)
    C_nu_r = C_nu.reshape(NQ, NY, NqT, Nbt)

    out = np.empty((NQ, NY, NqT), dtype=float)

    import time

    t0 = time.time()
    for iQ in range(NQ):
        # exp(C_nu * g_NP) on the (NY, NqT, Nbt) Q-slice -- the only piece
        # that doesn't factor across the slice
        exp_g_factor = np.exp(
            C_nu_r[iQ] * g_NP[np.newaxis, np.newaxis, :]
        )  # (NY, NqT, Nbt)
        if delta_l2 == 0.0:
            # Feff_bT shape (Nbt,); broadcast over (NY, NqT)
            integrand = (
                bT_J0[np.newaxis, :, :]
                * I_pert_r[iQ]
                * exp_g_factor
                * Feff_bT[np.newaxis, np.newaxis, :]
            )
        else:
            # Feff depends on Y (per-Y row), shape (NY, Nbt) broadcast over qT
            Feff = np.stack(
                [F_eff(Y_i, b_bar, **eff_params) for Y_i in Y_grid]
            )  # (NY, Nbt)
            integrand = (
                bT_J0[np.newaxis, :, :]
                * I_pert_r[iQ]
                * exp_g_factor
                * Feff[:, np.newaxis, :]
            )
        # Simpson over bT (last axis)
        sigma_Q = simpson(integrand, bT, axis=-1)  # (NY, NqT)
        # qT factor (SCETlib's x = qT*bT integration convention)
        out[iQ] = qT_grid[np.newaxis, :] * sigma_Q

        if verbose:
            elapsed = time.time() - t0
            print(
                f"  [hankel] Q-slice {iQ+1}/{NQ} done  "
                f"({elapsed:.1f}s elapsed, ETA {elapsed*(NQ-iQ-1)/(iQ+1):.1f}s)",
                flush=True,
            )
    return out


def integrate_over_Q(
    sigma_QYqT, Q_grid, Q_lo, Q_hi, method="arctan_Q2", q0=91.1876, Gamma=2.4952
):
    """Integrate sigma(Q, Y, qT) over Q in [Q_lo, Q_hi]. Q-samples outside
    [Q_lo, Q_hi] are dropped.

    sigma_QYqT : (NQ, NY, NqT) point values
    Q_grid     : (NQ,) Q sample positions (need not be uniform)
    Q_lo, Q_hi : integration limits
    method     : "arctan_Q2" (default; uses x = arctan((Q²-q0²)/(q0*Gamma)) so
                 the Breit-Wigner Z resonance becomes smooth) or "simpson"
                 (Simpson directly in Q) or "trapz" (trapezoid in Q).
    q0, Gamma  : resonance mass and width for the arctan_Q2 transform.
                 Defaults match Z-boson values (mZ = 91.1876, ΓZ = 2.4952).

    Returns: (NY, NqT) integrated values.
    """
    Q_grid = np.asarray(Q_grid, dtype=float)
    mask = (Q_grid >= Q_lo) & (Q_grid <= Q_hi)
    if mask.sum() < 2:
        raise ValueError(f"Need >= 2 Q samples in [{Q_lo}, {Q_hi}]; got {mask.sum()}")
    Q_sub = Q_grid[mask]
    s_sub = sigma_QYqT[mask]  # (NQ_sub, NY, NqT)
    s_moved = np.moveaxis(s_sub, 0, -1)  # (..., NQ_sub)

    if method == "simpson":
        return simpson(s_moved, Q_sub, axis=-1)
    if method == "trapz":
        return np.trapz(s_moved, Q_sub, axis=-1)
    if method == "arctan_Q2":
        # x = arctan((Q² - q0²) / (q0 * Gamma)) → flattens the Breit-Wigner peak
        x = np.arctan((Q_sub**2 - q0**2) / (q0 * Gamma))
        # dQ/dx = ( q0*Gamma + (Q² - q0²)² / (q0*Gamma) ) / (2 Q)
        jac = (q0 * Gamma + (Q_sub**2 - q0**2) ** 2 / (q0 * Gamma)) / (2.0 * Q_sub)
        return simpson(s_moved * jac, x, axis=-1)
    raise ValueError(f"integrate_over_Q: unknown method {method!r}")


def integrate_over_axis_bin(values, axis_grid, axis_lo, axis_hi, name="axis"):
    """Integrate a 1-D array of sample values over a single bin [axis_lo, axis_hi]
    using Simpson's rule on whatever sample points fall in (and on) the bin edges.

    Designed for use with a btgrid sampled at bin-edges + bin-centres (3 samples
    per bin minimum). When the grid carries the bin's 2 edges + central point,
    this is a 3-point Simpson (4th-order accurate per bin); when more samples
    fall inside the bin, simpson naturally extends to higher order via composite
    rule.

    values    : (..., N_axis) array values at axis_grid sample points
    axis_grid : (N_axis,) sample positions (sorted, not necessarily uniform)
    axis_lo, axis_hi : integration limits (bin edges)

    Returns: integrated value with the axis collapsed.
    """
    axis_grid = np.asarray(axis_grid, dtype=float)
    values = np.asarray(values, dtype=float)
    # tolerance for "on the edge" lookups (FP-imprecise bin centres show up as
    # e.g. -2.3499999999999996; bin widths are at least ~0.025 so 1e-9 is safe)
    tol = 1e-9
    mask = (axis_grid >= axis_lo - tol) & (axis_grid <= axis_hi + tol)
    if mask.sum() < 2:
        raise ValueError(
            f"integrate_over_axis_bin({name}): need >= 2 samples in "
            f"[{axis_lo}, {axis_hi}]; got {mask.sum()}"
        )
    sub = values[..., mask]
    g = axis_grid[mask]
    return simpson(sub, g, axis=-1)


def integrate_over_Y_bin(sigma_YqT, Y_grid, Y_lo, Y_hi):
    """Integrate sigma(Y, qT) over Y in [Y_lo, Y_hi] using sample points of
    Y_grid that fall in or on the bin edges. Returns (NqT,) array.

    Expects Y_grid to include Y_lo and Y_hi as samples (the edges) and ideally
    the bin centre too — Simpson uses all available samples in [Y_lo, Y_hi]."""
    # move Y axis (axis 0) to the end for integrate_over_axis_bin which uses last axis
    return integrate_over_axis_bin(
        np.moveaxis(sigma_YqT, 0, -1), Y_grid, Y_lo, Y_hi, name="Y"
    )


def integrate_over_qT_bin(sigma_YqT, qT_grid, qT_lo, qT_hi):
    """Integrate sigma(Y, qT) over qT in [qT_lo, qT_hi]. Returns (NY,) array.

    Expects qT_grid to include qT_lo and qT_hi as samples (the bin edges)."""
    return integrate_over_axis_bin(sigma_YqT, qT_grid, qT_lo, qT_hi, name="qT")


def reconstruct_batch(
    qT_per_bin, bT, I_pert, C_nu, b_bar, Y_per_bin, eff_params, gnu_params
):
    """Vectorised reconstruction for many bins at once.

    qT_per_bin : (Nbins,)   qT for each bin
    Y_per_bin  : (Nbins,)   Y  for each bin (used by F_eff's δλ_2·Y² term)
    bT         : (Nbt,)     integration variable (raw bT), shared by all bins
    I_pert     : (Nbins, Nbt) cached perturbative bT integrand
    C_nu       : (Nbins, Nbt) rapidity log coefficient
    b_bar      : (Nbt,)     b_star_global(bT), shared by all bins

    Returns: (Nbins,) sigma_factorized for each bin
    """
    qT_per_bin = np.asarray(qT_per_bin, dtype=float)
    Y_per_bin = np.asarray(Y_per_bin, dtype=float)
    bT = np.asarray(bT, dtype=float)
    I_pert = np.asarray(I_pert, dtype=float)
    C_nu = np.asarray(C_nu, dtype=float)
    b_bar = np.asarray(b_bar, dtype=float)

    # gamma_nu^NP and the gamma_nu exponential factor are bin-shared
    # (bT-dependent only); F_eff depends on Y through δλ_2·Y², so we evaluate
    # it per bin if δλ_2 != 0.
    g_NP = gamma_nu_NP(b_bar, **gnu_params)  # (Nbt,)
    exp_g_factor = np.exp(C_nu * g_NP[np.newaxis, :])  # (Nbins, Nbt)

    delta_l2 = eff_params.get("delta_lambda2", 0.0)
    if delta_l2 == 0.0:
        # F_eff has no Y dependence beyond a constant Y² factor, so identical
        # across bins -> compute once
        Feff = F_eff(0.0, b_bar, **eff_params)  # (Nbt,)
        bT_J0 = bT * bessel_j0(
            qT_per_bin[:, np.newaxis] * bT[np.newaxis, :]
        )  # (Nbins, Nbt)
        integrand = bT_J0 * I_pert * exp_g_factor * Feff[np.newaxis, :]
    else:
        # need per-bin F_eff because of Y dependence in lambda2_Y
        Feff = np.empty_like(I_pert)
        for i, Y_i in enumerate(Y_per_bin):
            Feff[i] = F_eff(Y_i, b_bar, **eff_params)
        bT_J0 = bT * bessel_j0(qT_per_bin[:, np.newaxis] * bT[np.newaxis, :])
        integrand = bT_J0 * I_pert * exp_g_factor * Feff

    # Multiply by qT to match SCETlib's spectrum-mode integration convention
    # (SCETlib's _int_bT integrates in x = qT*bT, picking up an explicit qT
    # factor via the Jacobian; we integrate in bT directly so we must add it
    # back).
    return qT_per_bin * simpson(integrand, bT, axis=-1)


# =============================================================================
# Loaders for the artefacts produced by the bT-grid condor run and by the
# spectrum-mode "combined" pickles.
# =============================================================================


def load_btgrid_shards(submitdir_or_glob, runcard_basename=None):
    """Combine bT-grid shards produced by --bt-grid mode.

    `submitdir_or_glob` may be:
      - a directory: we look for ``*_btgrid.pkl`` inside (recursively only
        one level via scetlib_outputs/).
      - a glob pattern: used directly.

    Returns dict with:
        bT      : (Nbt,)
        b_bar   : (Nbt,)
        bins    : list of (Q, Y, qT, lep) bin centres, length Nbins
        vars    : dict of variation index -> setting dict (copied from the
                  first shard; all shards are expected to carry the same set)
        I_pert  : (Nvars, Nbins, Nbt)
        C_nu   : (Nvars, Nbins, Nbt)
        config  : dict from the first shard (perturbative configuration the
                  grid was generated against)
        n_shards: int
    """
    if os.path.isdir(submitdir_or_glob):
        candidates = [
            os.path.join(submitdir_or_glob, "scetlib_outputs", "*_btgrid.pkl"),
            os.path.join(submitdir_or_glob, "*_btgrid.pkl"),
        ]
    else:
        candidates = [submitdir_or_glob]

    files = []
    for pat in candidates:
        files = sorted(glob.glob(pat))
        if files:
            break
    if not files:
        raise FileNotFoundError(f"No btgrid shards found under {submitdir_or_glob!r}")

    # First shard sets the schema; later shards must match.
    with open(files[0], "rb") as f:
        first = pickle.load(f)
    if first.get("schema_version") != "bt_grid_v1":
        raise ValueError(
            f"Unexpected schema {first.get('schema_version')!r} in {files[0]}"
        )
    bT = np.asarray(first["bT"], dtype=float)
    b_bar = np.asarray(first["b_bar"], dtype=float)
    varis = first["vars"]
    config = first["config"]
    n_vars = len(varis)
    n_bt = bT.size

    # We don't know Nbins ahead of time without scanning all shards. Walk them
    # once: build a dict of bin -> (var_idx -> (I_pert_row, C_nu_row)).
    bin_to_data = {}
    for path in files:
        with open(path, "rb") as f:
            d = pickle.load(f)
        if d.get("schema_version") != "bt_grid_v1":
            raise ValueError(
                f"Mixed schema versions: {path} has {d.get('schema_version')}"
            )
        if d["bT"].shape != bT.shape or not np.allclose(d["bT"], bT):
            raise ValueError(f"bT grid mismatch in {path}")
        bins_local = d["bins"]
        I_local = np.asarray(
            d["I_pert"], dtype=float
        )  # (Nvars_local, Nbins_local, Nbt)
        C_local = np.asarray(d["C_nu"], dtype=float)
        # Map local variation indices to the union variation order. We assume
        # all shards share the same vars dict (true when produced by the same
        # condor submission).
        var_order_local = list(d["vars"].keys())
        for b_idx, b_tup in enumerate(bins_local):
            slot = bin_to_data.setdefault(tuple(b_tup), {})
            for v_pos, v_idx in enumerate(var_order_local):
                slot[v_idx] = (I_local[v_pos, b_idx], C_local[v_pos, b_idx])

    var_order = list(varis.keys())
    bins_sorted = sorted(bin_to_data.keys(), key=lambda t: (t[0], t[1], t[2]))
    n_bins = len(bins_sorted)
    I_pert = np.full((n_vars, n_bins, n_bt), np.nan, dtype=float)
    C_nu = np.full((n_vars, n_bins, n_bt), np.nan, dtype=float)
    for b_pos, b_tup in enumerate(bins_sorted):
        per_var = bin_to_data[b_tup]
        for v_pos, v_idx in enumerate(var_order):
            if v_idx in per_var:
                I_pert[v_pos, b_pos] = per_var[v_idx][0]
                C_nu[v_pos, b_pos] = per_var[v_idx][1]

    return {
        "bT": bT,
        "b_bar": b_bar,
        "bins": bins_sorted,
        "vars": varis,
        "var_order": var_order,
        "I_pert": I_pert,
        "C_nu": C_nu,
        "config": config,
        "n_shards": len(files),
    }


def load_spectrum_reference(combined_pkl):
    """Load a 'combined' spectrum-mode pickle written by scetlib-run-qT.py.

    Returns a dict
        hist     : the hist.Hist object stored in the pickle
        config   : the [section]->dict mapping the run used
        meta_data: the meta-data dict
    Use the returned hist directly (axes are typically Q, Y, qT, lep, vars).
    """
    with open(combined_pkl, "rb") as f:
        d = pickle.load(f)
    return {
        "hist": d.get("hist"),
        "config": d.get("config", {}),
        "meta_data": d.get("meta_data", {}),
    }
