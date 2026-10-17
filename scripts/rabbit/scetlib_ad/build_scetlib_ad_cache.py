#!/usr/bin/env python3
"""Build a SCETlib autodiff cache on an explicit gen grid.

Writes the matching SCETlib runcard (the base runcard with its Grid_* sections
replaced) next to the output and builds

    compressed bin rules (resummed)  +  frozen fixed-order grid (nonsingular)

The cache is valid only for the bins it was built on. The grid defaults to the
one the theory corrections are defined on: a cache feeding a correction must
cover every gen event before acceptance, whereas a datacard's gen grid is
truncated at the reco |y_ll| limit, so the phase space above it would land in
the correction's flow bin (fixed at 1, i.e. uncorrected).

The build steps themselves are SCETlib's
(``examples/matched_ad/prepare_cache.py``); this script holds the grid, writes
the runcard, cuts the bins to a ``--subset`` and drives the steps in order.

Parallelism: the expensive PDF-member stage is parallel over the nodes of all
bins at once, so ``--threads`` is the lever. ``--subset`` is for test caches and
crash granularity (there is no checkpointing); subsets can be merged with
SCETlib's ``scetlib_cache.merge_bin_caches``. Cost scales with the number of
bins, but the low-qT rows dominate, so never extrapolate from a high-qT subset.

    source <scetlib-cms>/setup.sh
    python scripts/rabbit/scetlib_ad/build_scetlib_ad_cache.py \
        --base-conf scripts/rabbit/scetlib_ad/conf/Z_CT18Z_N3p0LL_FranksVals.conf \
        -o /path/to/cachedir
"""

import argparse
import configparser
import os
import time

import numpy as np

# F401 on `configure`: it is used through _resolve_steps, so a caller can stub it.
from wremnants.postprocessing.scetlib_ad.xsec_backend import (  # noqa: F401
    _scetlib_src,
    bins_from_gen_axes,
    configure,
)

# The SCETlib build steps, exposed as attributes of this module (``__getattr__``)
# so a caller can rebind one (``mod.plan_variations = ...``) and keep the rest.
_UPSTREAM_STEPS = (
    "plan_variations",
    "check_member_range",
    "build_variations",
    "build_prologue",
    "write_cache",
    "fork_member_build",
    "alphas_of",
    "find_alphas_pair",
    "pdf_set_size",
    "ensure_beamfunc_grids",
    "load_bins",
)

_UPSTREAM = None

# What ``main`` resolves through _resolve_steps: the upstream steps + configure.
_STEPS = _UPSTREAM_STEPS + ("configure",)


def _upstream_prepare_cache():
    """The upstream builder module, ``examples/matched_ad/prepare_cache.py``.

    Loaded by path (the examples are not a package) and lazily, so
    ``--dry-run`` works without SCETlib on the path.
    """
    global _UPSTREAM
    if _UPSTREAM is not None:
        return _UPSTREAM
    import importlib.util

    path = os.path.join(_scetlib_src(), "examples", "matched_ad", "prepare_cache.py")
    if not os.path.exists(path):
        raise SystemExit(
            f"scetlib_ad: cannot find {path}, which is the builder. Point "
            "SCETLIB_SRC at a checkout of the autodiff-sigmaul branch (source "
            "its setup.sh)."
        )
    spec = importlib.util.spec_from_file_location("_scetlib_prepare_cache", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    missing = [n for n in _UPSTREAM_STEPS if not hasattr(mod, n)]
    if missing:
        raise SystemExit(
            f"{path} is missing {missing}. That checkout predates the build "
            "steps moving into the example; update SCETlib."
        )
    _UPSTREAM = mod
    return mod


def __getattr__(name):
    """Expose the upstream build steps as attributes of this module (PEP 562).

    Only reached for names not in the module dict, so a caller's assignment wins.
    """
    if name in _UPSTREAM_STEPS:
        return getattr(_upstream_prepare_cache(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _resolve_steps():
    """The build steps this run will use, an override taking precedence.

    Resolved once, from this module's globals first, then upstream. Uses
    ``globals()`` because a file loaded by path has no ``sys.modules`` entry.
    """
    import types

    g = globals()
    return types.SimpleNamespace(
        **{n: g.get(n) or getattr(_upstream_prepare_cache(), n) for n in _STEPS}
    )


# Default grid = the theory corrections' grid: one Q bin (the Z window), 17 |Y|
# bins, 70 qT bins.
DEFAULT_Q_EDGES = [60.0, 120.0]
DEFAULT_Y_EDGES = [
    0,
    0.15,
    0.3,
    0.5,
    0.7,
    0.9,
    1.1,
    1.3,
    1.5,
    1.8,
    2,
    2.5,
    2.75,
    3,
    3.25,
    3.5,
    4,
    5,
]
DEFAULT_QT_EDGES = [
    0,
    0.5,
    1,
    1.5,
    2,
    2.5,
    3,
    3.5,
    4,
    4.5,
    5,
    5.5,
    6,
    6.5,
    7,
    7.5,
    8,
    8.5,
    9,
    9.5,
    10,
    10.5,
    11,
    11.5,
    12,
    12.5,
    13,
    13.5,
    14,
    14.5,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    42,
    44,
    46,
    48,
    50,
    52,
    54,
    56,
    58,
    60,
    65,
    70,
    80,
    90,
    100,
]


def write_runcard(base_conf, out_path, gen_axes, Q_lo, Q_hi):
    """Base runcard + this build's grids, written where the cache can find it."""
    conf = configparser.ConfigParser(inline_comment_prefixes="#")
    conf.optionxform = str  # SCETlib option names are case-sensitive
    if not conf.read(base_conf):
        raise SystemExit(f"cannot read base runcard {base_conf!r}")
    for key, values in (
        ("Q", [Q_lo, Q_hi]),
        ("Y", list(gen_axes[1][1])),
        ("qT", list(gen_axes[0][1])),
    ):
        sec = f"Grid_{key}"
        if not conf.has_section(sec):
            conf.add_section(sec)
        conf[sec]["custom_grid"] = "true"
        conf[sec]["bins"] = "true"
        conf[sec]["values"] = "[" + ", ".join(f"{v:g}" for v in values) + "]"
    header = (
        "# Generated by scripts/rabbit/scetlib_ad/build_scetlib_ad_cache.py.\n"
        f"# Base runcard: {os.path.abspath(base_conf)}\n"
        "# The Grid_* sections are the gen binning this cache was built on;\n"
        "# everything else is inherited. Keep this file next to the cache --\n"
        "# the fit needs it to rebuild the identical calculation the rules\n"
        "# attach to.\n"
    )
    with open(out_path, "w") as f:
        f.write(header)
        conf.write(f)


def select_subset(bins, gen_axes, spec):
    """The rows of ``bins`` named by a '<absY indices>/<ptV indices>' subset.

    ``gen_axes`` is [(qT_name, edges), (Y_name, edges)]; ``bins_from_gen_axes``
    flattens qT-major, so a subset's bins are a literal sub-list of the full one.
    """
    n_qt = np.asarray(gen_axes[0][1]).size - 1
    n_y = np.asarray(gen_axes[1][1]).size - 1
    ysel, qsel = (x.strip() for x in spec.split("/"))
    iy = list(range(n_y)) if ysel == "*" else [int(v) for v in ysel.split(",")]
    iq = list(range(n_qt)) if qsel == "*" else [int(v) for v in qsel.split(",")]
    for v, n, what in ((iy, n_y, "absY"), (iq, n_qt, "ptV")):
        bad = [i for i in v if not 0 <= i < n]
        if bad:
            raise SystemExit(f"--subset: {what} index {bad} out of range 0..{n - 1}")
    keep = [i * n_y + j for i in iq for j in iy]
    print(
        f"   --subset: {len(keep)} of {n_y * n_qt} bins "
        f"(absY {list(iy)}, ptV {list(iq)}) -- a TEST cache, not for a fit",
        flush=True,
    )
    return bins[keep]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--y-edges",
        type=float,
        nargs="+",
        default=DEFAULT_Y_EDGES,
        help="rapidity bin edges. The DEFAULT is the shipped theory "
        "corrections' |Y| grid, which is what a cache feeding a correction "
        "needs; a narrower grid leaves the phase space above it UNCORRECTED "
        "(the correction's flow bin is exactly 1). May be signed, in which "
        "case no |Y| folding is assumed -- useful for validating against a "
        "SCETlib reference run on its own signed grid, and required for W.",
    )
    ap.add_argument(
        "--qt-edges",
        type=float,
        nargs="+",
        default=DEFAULT_QT_EDGES,
        help="qT bin edges (default: the shipped corrections' 70-bin grid, "
        "which the SCETlib production runcard also used).",
    )
    ap.add_argument(
        "--q-edges",
        type=float,
        nargs=2,
        default=DEFAULT_Q_EDGES,
        help="the single Q bin, as two edges (default: the Z window the "
        "corrections use). Gen events outside it land in the correction's Q "
        "flow bin and are uncorrected.",
    )
    ap.add_argument(
        "--base-conf",
        required=True,
        help="SCETlib runcard supplying the physics settings "
        "(orders, PDF, nonperturbative model); its Grid_* sections are replaced",
    )
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--outname", default="cache")
    ap.add_argument(
        "--subset",
        default=None,
        help="build only a SUBSET of the card's bins, as "
        "'iy0,iy1,.../iqt0,iqt1,...' of indices into the card's absY and ptV "
        "axes (either side may be '*' for all). The point is to make the REAL "
        "test -- validate_variations against the production corrections, every "
        "variation -- cheap enough to iterate on: 12 bins instead of 210 is "
        "minutes instead of hours, and it is the same code path, so a subset "
        "cache validates exactly what a full one does over the bins it covers. "
        "\n"
        "The indices must be CONTIGUOUS in both axes: the gen fold requires the "
        "cache to tile a rectangle exactly, so scattered picks are refused "
        "downstream with 'gen bin(s) are not exactly tiled by the cache'. Choose "
        "by COST, not by count: the lowest ptV rows are far more expensive than "
        "all the others together, so never extrapolate a cost from a high-qT "
        "subset.\n"
        "This is for a TEST cache, or for CRASH GRANULARITY on a long build "
        "(there is no checkpointing) -- not for speed, since the member stage "
        "is parallel over NODES and scales past the bin count. Disjoint subsets "
        "can be assembled afterwards with SCETlib's "
        "scetlib_cache.merge_bin_caches. A subset cache on its own is not for "
        "a fit.",
    )
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument(
        "--n-train",
        type=int,
        default=9,
        help="training points for the rule compression. Accuracy "
        "tracks n_train/n_params, but the non-negative "
        "least-squares solve grows roughly like n_train^2, so "
        "raising it with the parameter count is the expensive "
        "knob (see doc/autodiff-design.md)",
    )
    ap.add_argument(
        "--pdf-eig",
        type=int,
        default=-1,
        help="number of PDF eigenvector pairs (default: all of the set). 0 "
        "still keeps the alphaS pair, which is a separate direction.",
    )
    ap.add_argument(
        "--as-pair",
        default="auto",
        help="'auto' finds <set>_as_0116/_as_0120, 'off' disables (alphas then "
        "moves the calculation but NOT the PDF), or 'down,up' explicitly.",
    )
    ap.add_argument(
        "--no-muf",
        action="store_true",
        help="skip the muF member pair (then resumScaleMuF does nothing and the "
        "card's resumFOScale* must be kept).",
    )
    ap.add_argument(
        "--no-pdf",
        action="store_true",
        help="physics-only cache: no PDF eigenvectors, no alphaS pair, no muF. "
        "alphaS is then a fixed-PDF derivative -- do not quote it.",
    )
    ap.add_argument(
        "--fork-members",
        type=int,
        default=1,
        help="split the PDF-member loop over this many forked children (1 = "
        "the serial loop, the default). EXACT but SLOW, and almost never what "
        "you want: forking after the rules are built is the only way to give "
        "two processes the SAME rules (they share the parent's node cache by "
        "copy-on-write, where independent processes each adapt their own node "
        "set and cannot be merged at all), but a forked child loses the TBB "
        "worker pool -- measured 99% CPU per child against the parent's 1900% "
        "-- so each child is single-threaded and a real bin count is ~100x "
        "slower per member. To parallelise a real build, raise --threads: the "
        "member stage is parallel over NODES, not bins, and scales past the "
        "bin count. This exists for "
        "--fork-selftest, which is what proves the member merge exact.",
    )
    ap.add_argument(
        "--fork-selftest",
        action="store_true",
        help="after the forked build, build EVERY member serially in the "
        "parent as well and write <outname>.serial.npz. Same process, same "
        "node cache, same rules, so the only difference is forked-and-merged "
        "against serial: comparing the two is the decisive test of the merge "
        "(compare_caches.py --bytes). Costs one extra member pass.",
    )
    ap.add_argument(
        "--grid-jobs",
        type=int,
        default=0,
        help="parallel beamfunc-grid generation jobs (0 = one per core).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="write the runcard and report the bin count and the "
        "projected cost, without building",
    )
    args = ap.parse_args()

    for name, edges in (("--y-edges", args.y_edges), ("--qt-edges", args.qt_edges)):
        e = np.asarray(edges, dtype=np.float64)
        if e.size < 2 or np.any(np.diff(e) <= 0):
            raise SystemExit(f"{name} must be >= 2 strictly increasing edges")
    args.Q_lo, args.Q_hi = (float(v) for v in args.q_edges)
    if args.Q_hi <= args.Q_lo:
        raise SystemExit("--q-edges must be increasing")
    gen_axes = [
        ("qT", np.asarray(args.qt_edges, dtype=np.float64)),
        ("Y", np.asarray(args.y_edges, dtype=np.float64)),
    ]
    n_bins = int(np.prod([len(e) - 1 for _, e in gen_axes]))
    os.makedirs(args.outdir, exist_ok=True)
    runcard = os.path.join(args.outdir, args.outname + ".conf")
    write_runcard(args.base_conf, runcard, gen_axes, args.Q_lo, args.Q_hi)
    default_grid = list(args.y_edges) == list(DEFAULT_Y_EDGES) and list(
        args.qt_edges
    ) == list(DEFAULT_QT_EDGES)
    print(
        "gen binning: "
        + (
            "the shipped theory corrections' grid (defaults)"
            if default_grid
            else "EXPLICIT, not the shipped corrections' default grid"
        )
        + ":"
    )
    for name, edges in gen_axes:
        print(f"   {name:<12} {len(edges) - 1:4d} bins  [{edges[0]:g}, {edges[-1]:g}]")
    print(f"   Q            1 bin    [{args.Q_lo:g}, {args.Q_hi:g}]")
    print(f"   -> {n_bins} SCETlib bins; runcard written to {runcard}")
    # Per-bin averages of a large build at full parallelism; a small cache is
    # slower per bin (not enough work to fill the pool).
    print(
        f"   projected for the FULL {n_bins}-bin grid (at full parallelism, and "
        f"BEFORE any --subset): ~{n_bins * 0.34 / 60:.0f} min of rules, "
        f"~{n_bins * 2.0 / 60:.0f} min of fixed-order warming, "
        f"~{n_bins * 0.84:.0f} MB of cache. These are averages over a whole "
        f"grid: the low-qT rows cost many times the rest, so this is not a "
        f"per-shard estimate and a subset's share is NOT proportional."
    )
    if args.dry_run:
        return

    # Build steps go through this namespace so a rebound step is the one run.
    me = _resolve_steps()

    conf, sigma = me.configure(runcard, args.threads)
    bins = bins_from_gen_axes(gen_axes, args.Q_lo, args.Q_hi)
    if args.subset:
        bins = select_subset(bins, gen_axes, args.subset)
    p0 = np.asarray(sigma.gradient_central(), dtype=np.float64)
    names = list(sigma.gradient_param_names())

    sing, nons = sigma.sub_pieces()

    # Plan the variation members BEFORE the rules: the PDF eigenvector
    # coefficients are AD parameters, so registering them changes the parameter
    # vector, the anchor and the rule fingerprint.
    plan = None if args.no_pdf else me.plan_variations(p0, names, conf, args)
    if plan and plan["n_eig"]:
        # On both pieces, so the gradient columns line up by name.
        sing.set_pdf_eig_params(plan["n_eig"])
        nons.set_pdf_eig_params(plan["n_eig"])
        p0 = np.asarray(sing.gradient_central(), dtype=np.float64)
        names = list(sing.gradient_param_names())
    print(f"\n{len(p0)} differentiable parameters:")
    for n, v in zip(names, p0):
        print(f"   {n:<24} {v:.6g}")

    me.build_prologue(sigma, sing, bins, p0, args.n_train, args.threads or 0)

    out = os.path.join(args.outdir, args.outname)
    if plan is None:
        print(
            "\n--no-pdf: physics-only cache. alphaS will be a derivative at "
            "FIXED PDF, and the card's pdf*/pdfAlphaS/resumFOScale* templates "
            "must be kept.",
            flush=True,
        )
        path = me.write_cache(sing, nons, bins, plan, out)
    elif args.fork_members > 1:
        # Fork before the parent touches the calculation again (so a
        # --fork-selftest pass cannot affect the children); build_fn/write_fn
        # are passed so a rebound step reaches the children too.
        path = me.fork_member_build(
            sing,
            nons,
            bins,
            p0,
            plan,
            args,
            out,
            build_fn=me.build_variations,
            write_fn=me.write_cache,
        )
        if args.fork_selftest:
            t0 = time.time()
            me.build_variations(sing, nons, bins, p0, plan, args)
            print(
                f"selftest: all {len(plan['members'])} members serially in "
                f"{(time.time() - t0) / 60:.1f} min",
                flush=True,
            )
            me.write_cache(sing, nons, bins, plan, out + ".serial")
    else:
        me.build_variations(sing, nons, bins, p0, plan, args, 0, None)
        path = me.write_cache(sing, nons, bins, plan, out, None, None, args=args)
    print(
        "\nNow check it:\n"
        f"   python scripts/rabbit/scetlib_ad/backend_check.py "
        f"--conf {runcard} --cache {path}"
    )


if __name__ == "__main__":
    main()
