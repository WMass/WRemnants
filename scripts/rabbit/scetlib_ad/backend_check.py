#!/usr/bin/env python3
"""Standalone sanity check of the SCETlib autodiff backend.

Runs without rabbit and without a datacard: load a cache, verify the anchor
round trip, finite-difference a couple of Jacobian columns, check the Hessian,
and exercise the bin permutation. This is the first thing to run after building
a new cache.

    source <scetlib-cms>/setup.sh
    python scripts/rabbit/scetlib_ad/backend_check.py \
        --conf  <scetlib-cms>/examples/matched_ad/matched.conf \
        --cache <scetlib-cms>/examples/matched_ad/cache_debug.npz
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

from wremnants.postprocessing.scetlib_ad import params as adp  # noqa: E402
from wremnants.postprocessing.scetlib_ad.xsec_backend import (  # noqa: E402
    ScetlibADXsec,
)


def gen_axes_from_bins(bins):
    """Recover ((qT, edges), (|Y|, edges)) and the Q window from a cache bin list.

    Only valid for a cache that is a full product grid, which is what
    prepare_cache builds; it lets the check run without a datacard.
    """
    Q = np.unique(bins[:, 0:2], axis=0)
    if Q.shape[0] != 1:
        raise SystemExit(f"expected a single Q bin, got {Q}")
    y = np.unique(bins[:, 2:4], axis=0)
    t = np.unique(bins[:, 4:6], axis=0)
    y = y[np.argsort(y[:, 0])]
    t = t[np.argsort(t[:, 0])]
    y_edges = np.concatenate([y[:, 0], y[-1:, 1]])
    t_edges = np.concatenate([t[:, 0], t[-1:, 1]])
    return [("qT", t_edges), ("absY", y_edges)], float(Q[0, 0]), float(Q[0, 1])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--conf", required=True, help="SCETlib runcard the cache was built from"
    )
    ap.add_argument("--cache", required=True, help="cache .npz")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument(
        "--fd-params",
        default=None,
        help="comma-separated SCETlib parameter names to "
        "finite-difference (default: alphas + the first NP one)",
    )
    args = ap.parse_args()

    t0 = time.time()
    core = ScetlibADXsec(args.conf, args.cache, threads=args.threads)
    print(f"loaded in {time.time() - t0:.1f} s: {core}")
    print(f"  parameters ({core.n_params}):")
    for n, v in zip(core.param_names, core.anchor):
        print(f"     {n:<24} {v:12.6g}   -> rabbit {adp.rabbit_name(n)}")

    # ---- values and Jacobian at the anchor. The first call pays the worker-pool
    # build and the LHAPDF/grid touch, so time a displaced repeat too -- that
    # second number is what a minimizer iteration actually costs.
    t0 = time.time()
    val, jac = core.values_and_jacobian(core.anchor)
    dt = time.time() - t0
    p_off = core.anchor * (1.0 + 1e-3) + 1e-6
    t0 = time.time()
    core.values_and_jacobian(p_off)
    dt_warm = time.time() - t0
    print(
        f"\nvalue+jacobian at the anchor: {dt * 1e3:.0f} ms "
        f"({dt / core.n_bins * 1e3:.2f} ms/bin) first call, "
        f"{dt_warm * 1e3:.0f} ms ({dt_warm / core.n_bins * 1e3:.2f} ms/bin) warm; "
        f"sum(sigma) = {val.sum():.8g}"
    )
    if not np.all(np.isfinite(val)):
        raise SystemExit("non-finite values at the anchor")
    if np.any(val <= 0):
        n = int((val <= 0).sum())
        print(
            f"  NOTE: {n} bin(s) are non-positive at the anchor "
            f"(lowest-qT bins can be, with NP off / no large-bT damping)"
        )

    # ---- ratio to itself must be exactly 1
    val2, _ = core.values_and_jacobian(core.anchor)
    print(f"  anchor re-evaluation is bit-identical: {np.array_equal(val, val2)}")

    # ---- finite-difference a couple of Jacobian columns
    names = core.param_names
    if args.fd_params:
        which = [names.index(n) for n in args.fd_params.split(",")]
    else:
        which = [0]
        for i, n in enumerate(names):
            if n.startswith("np_gnu_lambda2") or n.startswith("np_eff_lambda2"):
                which.append(i)
                break
    print("\nfinite-difference check of d(sum sigma)/dp:")
    worst = 0.0
    for i in which:
        h = 1e-4 * max(abs(core.anchor[i]), 1e-3)
        pp, pm = core.anchor.copy(), core.anchor.copy()
        pp[i] += h
        pm[i] -= h
        fd = (
            core.values_and_jacobian(pp)[0].sum()
            - core.values_and_jacobian(pm)[0].sum()
        ) / (2 * h)
        an = jac[:, i].sum()
        rel = abs(an - fd) / max(abs(fd), 1e-300)
        worst = max(worst, rel)
        print(f"   {names[i]:<24} analytic {an:14.8g}  FD {fd:14.8g}  rel {rel:.2e}")
    # The rule reproduces the direct calculation to ~1e-15, so the only error
    # here is the FD truncation; anything above ~1e-5 means the gradient is wrong.
    status = "OK" if worst < 1e-5 else "FAIL"
    print(f"   worst relative disagreement {worst:.2e} -> {status}")

    # ---- Hessian
    t0 = time.time()
    H = core.hessian(core.anchor)
    dt = time.time() - t0
    asym = np.max(np.abs(H - np.transpose(H, (0, 2, 1))))
    scale = max(np.max(np.abs(H)), 1e-300)
    print(
        f"\nhessian: {dt * 1e3:.0f} ms ({dt / core.n_bins * 1e3:.2f} ms/bin), "
        f"shape {H.shape}, max|H - H^T|/max|H| = {asym / scale:.2e}"
    )
    print(
        f"   -> hessian costs {dt / max(dt_warm, 1e-12):.0f}x a value+jacobian "
        f"call; that ratio is why curvature is off by default during the fit "
        f"and only turned on for the covariance pass"
    )

    # ---- fold onto the gen grid, and the sum rule that proves it is exact
    gen_axes, q_lo, q_hi = gen_axes_from_bins(core.bins)
    fold = core.fold_for(gen_axes, q_lo, q_hi)
    folded = fold(val)
    print(
        f"\nfold onto the gen grid "
        f"({gen_axes[0][0]}: {len(gen_axes[0][1]) - 1}, "
        f"{gen_axes[1][0]}: {len(gen_axes[1][1]) - 1}, Q [{q_lo:g}, {q_hi:g}]): "
        f"{fold.describe()}"
    )
    # Summing bin-integrated cross sections is exact, so the folded total must
    # equal the total over the cache bins the fold used (fp round-off only).
    used = val.sum() if fold.n_dropped == 0 else None
    if used is not None:
        rel = abs(folded.sum() / used - 1.0)
        print(
            f"   sum rule: folded total vs cache total, rel {rel:.2e} "
            f"-> {'OK' if rel < 1e-12 else 'FAIL'}"
        )
        if rel >= 1e-12:
            raise SystemExit("the fold does not conserve the total cross section")

    if status != "OK":
        raise SystemExit("gradient check failed")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
