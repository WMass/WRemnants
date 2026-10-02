#!/usr/bin/env python3
"""Cross-check the SCETlib autodiff model's lambda response against scetlib_np.

The two models compute sigma_gen by completely different routes -- the AD one
replays SCETlib's own compressed bin rules, the NP one reconstructs the Hankel
integral from a cached bT grid in TensorFlow -- so agreement of the RESPONSE

    R(lambda) = sigma_gen(lambda) / sigma_gen(lambda_central)

is a sharp test of the new model. The NP model's response is itself validated
against the histmaker templates at the 0.02-0.05% level, so a disagreement here
well above that points at the new model (a wrong name map, a transposed fold, a
sign).

The two do NOT share a nonsingular: the AD path uses SCETlib's own analytic
NLO V+jet at O(alphas^2), the NP path uses DYTurbo minus the SCETlib singular
expansion. That difference largely cancels in the ratio, but not exactly, and it
grows where the fixed order dominates -- hence the low-qT / high-qT split in the
report. Read the low-qT column as the real test.

    source <scetlib-cms>/setup.sh
    python scripts/rabbit/scetlib_ad/compare_to_np_model.py \
        --conf <...>/matched.conf --cache <...>/cache_debug.npz
"""

import argparse
import os
import sys

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

# rabbit-facing name -> which scetlib_np dict it belongs in
EFF = ("lambda2", "lambda4", "lambda6", "delta_lambda2", "lambda_inf")
GNU = ("lambda2_nu", "lambda4_nu", "lambda6_nu", "lambda_inf_nu")


def gen_axes_from_bins(bins):
    Q = np.unique(bins[:, 0:2], axis=0)
    if Q.shape[0] != 1:
        raise SystemExit(f"expected a single Q bin, got {Q}")
    y = np.unique(np.abs(bins[:, 2:4]), axis=0)
    t = np.unique(bins[:, 4:6], axis=0)
    y = y[np.argsort(y[:, 0])]
    t = t[np.argsort(t[:, 0])]
    return (
        [
            ("ptVGen", np.concatenate([t[:, 0], t[-1:, 1]])),
            ("absYVGen", np.concatenate([y[:, 0], y[-1:, 1]])),
        ],
        float(Q[0, 0]),
        float(Q[0, 1]),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--conf", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument(
        "--btgrid",
        default=None,
        help="scetlib_np bT-grid dir (default: the CT18Z one the " "package resolves)",
    )
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument(
        "--rel-shifts",
        default="0.05,0.2,1.0",
        help="comma-separated RELATIVE displacements applied to each "
        "lambda in turn. Relative, because the compressed bin "
        "rules are trained in a neighbourhood of the anchor "
        "(scale=0.15 by default) and a fixed absolute shift "
        "would move a small lambda far outside it while barely "
        "moving a large one",
    )
    ap.add_argument(
        "--abs-floor",
        type=float,
        default=0.02,
        help="displacement used for a lambda whose anchor is ~0 "
        "(delta_lambda2), where a relative shift is meaningless",
    )
    ap.add_argument(
        "--qt-split",
        type=float,
        default=10.0,
        help="qT (GeV) separating the resummation-dominated region, "
        "where the two nonsingulars agree best, from the rest",
    )
    args = ap.parse_args()

    import tensorflow as tf  # noqa: F401  (imported by the NP core anyway)

    from wremnants.postprocessing.scetlib_np.sigma_gen import SigmaGenModel

    core = ScetlibADXsec(args.conf, args.cache, threads=args.threads)
    gen_axes, q_lo, q_hi = gen_axes_from_bins(core.bins)
    fold = core.fold_for(gen_axes, q_lo, q_hi)
    shape = fold.gen_shape
    print(
        f"gen grid {shape[0]} x {shape[1]}, Q [{q_lo:g}, {q_hi:g}]; "
        f"{fold.describe()}"
    )

    # The NP model's central must be the AD cache's anchor, or the two are not
    # anchored at the same point and every ratio below is meaningless.
    npsec = core.conf["Nonperturbative"]
    eff_c = {"np_model": npsec.get("np_model", "tanh_2")}
    gnu_c = {"np_model_nu": npsec.get("np_model_nu", "tanh_2")}
    for sname, value in zip(core.param_names, core.anchor):
        try:
            rname = adp.rabbit_name(sname)
        except KeyError:
            continue
        if rname in EFF:
            eff_c[rname] = float(value)
        elif rname in GNU:
            gnu_c[rname] = float(value)
    print(f"anchored at  eff {eff_c}\n             gnu {gnu_c}")

    np_core = SigmaGenModel(
        btgrid_dir=args.btgrid,
        lambda_central={"eff_params": eff_c, "gnu_params": gnu_c},
        gen_axes=gen_axes,
        Q_lo=q_lo,
        Q_hi=q_hi,
    )

    def sigma_ad(**over):
        p = core.anchor.copy()
        for rname, val in over.items():
            p[core.param_names.index(adp.scetlib_name(rname))] = val
        vals, _ = core.values_and_jacobian(p)
        return fold(np.asarray(vals, dtype=np.float64)).reshape(shape)

    def sigma_np(**over):
        eff, gnu = dict(eff_c), dict(gnu_c)
        for rname, val in over.items():
            (eff if rname in EFF else gnu)[rname] = val
        return np.asarray(np_core.sigma_gen(eff, gnu))

    ad0, np0 = sigma_ad(), sigma_np()
    qt_lo = gen_axes[0][1][:-1]
    low = qt_lo < args.qt_split

    # ABSOLUTE normalisation. The fit never sees this, because compute() divides
    # by the model's own central -- which is exactly why it is worth printing:
    # it is what would have to be right if the anchor ratio were ever dropped in
    # favour of an absolute prediction.
    print(
        f"\nABSOLUTE totals (hidden by the anchor ratio): "
        f"AD {ad0.sum():.6g}, NP {np0.sum():.6g}, AD/NP = {ad0.sum() / np0.sum():.6f}"
    )
    print(
        "   NB the AD cache integrates the POSITIVE Y side only, so a factor 2 per "
        "|Y| bin\n   is expected here and cancels in the ratio; anything beyond "
        "that is a real\n   normalisation convention to nail before going absolute."
    )

    # Informational: the two predictions do not have to agree in absolute shape
    # (different nonsingular), and compute() never uses the absolute value.
    shape_ratio = (ad0 / ad0.sum()) / (np0 / np0.sum())
    print(
        "\ncentral shape, AD vs NP (normalised; NOT required to agree -- the "
        "models use different nonsingulars, and the fit only ever uses the "
        "ratio to each model's own central):"
    )
    print(
        f"   qT < {args.qt_split:g}: max |ratio - 1| = "
        f"{np.max(np.abs(shape_ratio[low] - 1)):.2%}   "
        f"qT > {args.qt_split:g}: {np.max(np.abs(shape_ratio[~low] - 1)):.2%}"
    )

    common = [
        n for n in EFF + GNU if n in [adp.rabbit_name(s) for s in core.param_names]
    ]
    rel_shifts = [float(x) for x in args.rel_shifts.split(",") if x.strip()]
    print(
        f"\nlambda response agreement, |R_AD - R_NP| as a fraction of the "
        f"response |R_NP - 1| (max over qT < {args.qt_split:g})."
    )
    print(
        "   Scanning the displacement matters: the rules are compressed around "
        "the anchor,\n   so the agreement should IMPROVE towards small shifts. "
        "A flat or growing\n   trend towards zero displacement is a bug, not "
        "rule locality."
    )
    header = (
        "   "
        + f"{'parameter':<18}"
        + "".join(f"{'x' + format(1 + r, 'g'):>12}" for r in rel_shifts)
        + f"{'response(x' + format(1 + rel_shifts[-1], 'g') + ')':>16}"
    )
    print("\n" + header)
    worst_small = 0.0
    for name in common:
        base = float(core.anchor[core.param_names.index(adp.scetlib_name(name))])
        cells, size = [], 0.0
        for r in rel_shifts:
            step = base * r if abs(base) > 1e-9 else args.abs_floor * r / rel_shifts[-1]
            if abs(step) < 1e-12:
                cells.append("        n/a")
                continue
            r_ad = sigma_ad(**{name: base + step}) / ad0
            r_np = sigma_np(**{name: base + step}) / np0
            size = float(np.max(np.abs(r_np[low] - 1.0)))
            if size < 1e-4:
                cells.append("       inert")
                continue
            frac = float(np.max(np.abs(r_ad[low] - r_np[low])) / size)
            cells.append(f"{frac:11.1%}")
            if r == min(rel_shifts):
                worst_small = max(worst_small, frac)
        print(f"   {name:<18}" + "".join(f"{c:>12}" for c in cells) + f"{size:15.2%}")
    print(f"\n   worst disagreement at the smallest displacement: {worst_small:.1%}")
    print(
        "   A wrong name map, a transposed fold or a sign error would show up "
        "as O(1) at every\n   displacement. What remains at small displacement "
        "is the genuine difference between\n   the two predictions: different "
        "nonsingulars, and the NP model rebinning a POINT\n   grid by "
        "quadrature where SCETlib integrates each bin exactly."
    )


if __name__ == "__main__":
    main()
