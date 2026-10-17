#!/usr/bin/env python3
"""Build a SCETlib autodiff cache on an EXPLICIT gen grid.

The cache is only valid for the bins it was built for, so the grid is given
here, as options, and defaults to the grid the shipped theory corrections are
defined on. This writes the matching SCETlib runcard next to the output and runs
the expensive build:

    compressed bin rules (resummed)  +  frozen fixed-order grid (nonsingular)

WHICH GRID, AND WHY IT IS NOT A DATACARD'S. A cache feeding the fit-time model
must compute sigma_gen on the bins the RESPONSE folds, so taking the grid from a
datacard was right while that was the only consumer. A cache feeding a THEORY
CORRECTION must cover the phase space the correction is applied to, which is
every gen event BEFORE acceptance -- and the card's gen grid is truncated at the
reco |y_ll| limit (``mz_dilepton.py``: ``[e for e in corr_edges["absY"] if e <=
y_max]``). Building a correction from a card-derived cache therefore left
|Y| > 2.5 -- 29.6 % of the Z gen cross section -- on the correction's flow bin,
which ``set_corr_ratio_flow`` fixes at exactly 1, i.e. uncorrected. Reading the
grid off a card also made the dependency circular once the correction came from
the cache: corr -> response binning -> card -> cache -> corr.

So there is no ``--card``. Pass ``--y-edges`` / ``--qt-edges`` / ``--q-edges``,
whose defaults ARE the shipped corrections' grid; a narrower grid is then a
visible choice rather than an inherited one.

WHAT IS HERE AND WHAT IS NOT. The build itself is nothing but SCETlib calls, so
it lives in SCETlib, in ``examples/matched_ad/prepare_cache.py``: the variation
plan, the node set, the rules, the member loop, the cache file. Node sets, bin
rules, member variations and the on-disk format are all theirs, and keeping our
copy of them meant learning about a layout change by getting wrong answers.
What is ours, and what this file is, is the wrapper: holding the grid, writing
the runcard from it, cutting the bins to a ``--subset``, and driving the upstream
steps in the order they have to happen.

PARALLELISM IS THREADS, NOT PROCESSES. The expensive stage is the PDF-member
loop, whose cost is ``set_pdf_keep_nodes``, and that is parallel over NODES of
all bins at once -- so ``--threads`` is the lever and it scales past the bin
count. The 770-bin production cache was built in ONE process at ``--threads
384`` (25.3 h). ``--subset`` remains, but for CRASH GRANULARITY and for building
a cheap test cache, not for speed: there is no checkpointing, so a build that
dies is lost, and subsets can be merged afterwards with SCETlib's
``scetlib_cache.merge_bin_caches``.

Cost scales with the number of gen bins. Measured on this SCETlib build:
~0.34 s/bin of rule building and ~2 s/bin of fixed-order warming, and ~0.84 MB
of cache per bin -- but those are averages over a whole grid, and the LOW-qT
rows cost many times the rest, so never extrapolate a cost from a high-qT
subset.

    source <scetlib-cms>/setup.sh
    python scripts/rabbit/scetlib_ad/build_scetlib_ad_cache.py \
        --base-conf <scetlib-cms>/examples/matched_ad/matched.conf \
        -o /path/to/cachedir
"""

import argparse
import configparser
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

# F401 on `configure`: it is used, but through _resolve_steps below rather than
# by name, so that the equivalence harness can stub it.
from wremnants.postprocessing.scetlib_ad.xsec_backend import (  # noqa: E402,F401
    _scetlib_src,
    bins_from_gen_axes,
    configure,
)

# The build steps, all of them SCETlib's, reached through the example module.
# They are exposed as attributes of THIS module (see ``__getattr__``) and
# ``main`` calls them through it, so a study that needs one step changed can
# rebind it -- ``mod.plan_variations = ...`` -- and get the upstream behaviour
# for everything else. That is how the muF knot-spacing scans work.
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

# What ``main`` resolves through _resolve_steps: the upstream steps plus
# ``configure``, which is ours but is stubbed by the equivalence harness.
_STEPS = _UPSTREAM_STEPS + ("configure",)


def _upstream_prepare_cache():
    """The upstream builder module, ``examples/matched_ad/prepare_cache.py``.

    Loaded by path rather than imported, because the SCETlib examples are not a
    package. Deliberately LAZY: ``--dry-run`` and the card readers below work
    without SCETlib on the path at all, and importing this pulls in the
    compiled extension.
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

    Only reached for names NOT in the module dict, so an assignment from a
    caller wins -- which is what makes the steps overridable.
    """
    if name in _UPSTREAM_STEPS:
        return getattr(_upstream_prepare_cache(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _resolve_steps():
    """The build steps this run will use, an override taking precedence.

    Resolved ONCE, at the top of ``main``, from this module's own globals and
    only then from the upstream module -- so a caller that did
    ``mod.plan_variations = ...`` after importing this file gets its own
    version, and everything it did not touch stays upstream's. Going through
    ``globals()`` rather than ``sys.modules[__name__]`` is deliberate: this
    file is usually loaded by path (``spec_from_file_location``), which does
    not register a module name.
    """
    import types

    g = globals()
    return types.SimpleNamespace(
        **{n: g.get(n) or getattr(_upstream_prepare_cache(), n) for n in _STEPS}
    )


# The grid the shipped theory corrections are defined on, and therefore the
# default here: a cache feeding a correction has to cover the phase space the
# correction is applied to. Q is one bin, the Z window. The |Y| edges are the
# shipped corrections' 17 bins -- the first 12 of which are what a card-derived
# cache used to stop at. qT is the 70-bin grid, which the shipped corrections
# and the SCETlib production runcard already share byte for byte.
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

    ``gen_axes`` is [(qT_name, edges), (Y_name, edges)] -- a LIST of tuples, not
    a dict -- and ``bins_from_gen_axes`` flattens qT-MAJOR (i over qT outer, j
    over absY inner). Index accordingly, so a subset cache's bins are a literal
    sub-list of the full one and nothing downstream cares.
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
    # Per-bin costs measured on a 5740-bin build with all cores busy. A small
    # cache does not reach that: with 30 bins the same build ran ~10x slower per
    # bin, because there is not enough work to fill the pool.
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

    # Every build step below is reached through this namespace, not called
    # directly, so that a rebound step is the one that runs. See _resolve_steps.
    me = _resolve_steps()

    conf, sigma = me.configure(runcard, args.threads)
    bins = bins_from_gen_axes(gen_axes, args.Q_lo, args.Q_hi)
    if args.subset:
        bins = select_subset(bins, gen_axes, args.subset)
    p0 = np.asarray(sigma.gradient_central(), dtype=np.float64)
    names = list(sigma.gradient_param_names())

    sing, nons = sigma.sub_pieces()

    # The variation members are DECIDED here, before the rules, because the PDF
    # eigenvector coefficients are ordinary AD parameters: registering them
    # changes the parameter vector, and so the anchor, the training points and
    # the rule fingerprint. Doing it afterwards would leave the rules built for
    # a shorter vector than the members are interpolated in, which is the
    # "call set_pdf_eig_params before building" error the extension raises.
    plan = None if args.no_pdf else me.plan_variations(p0, names, conf, args)
    if plan and plan["n_eig"]:
        # BOTH pieces: each kernel interpolates its own members from the
        # coefficients, so they are ordinary parameters on both sides and the
        # gradient columns line up by name.
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
        # The children inherit the state as it is RIGHT NOW -- fork before the
        # parent touches the calculation again, so a --fork-selftest pass
        # cannot influence what they build. build_fn/write_fn are passed so a
        # rebound step reaches the children too.
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
