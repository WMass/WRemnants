#!/usr/bin/env python3
"""Validate an autodiff cache against a native SCETlib prediction.

Two references are meaningful, and they test different things:

* a production run with ``calculation_piece = sing`` -- the RESUMMED cross
  section. Replaying only our cache's compressed bin rules gives the same object,
  with no matching, no fixed-order generator and no MC in between, so any
  disagreement is ours: runcard, quadrature, Q integration, rule compression.
* a ``*Corr<boson>.pkl.lz4`` theory correction -- the MATCHED prediction the
  analysis actually uses, resummed plus (fixed-order generator minus the singular
  expansion). Our cache computes its own matched total with SCETlib's in-house
  analytic V+jet instead, so this comparison measures the deliberate change of
  nonsingular as well as everything the resummed test covers.

Both sides are bin-integrated, so the two are summed onto their COMMON bin edges
-- exact, no interpolation -- and the script refuses to run unless each side
tiles that common grid exactly. A signed-Y cache is folded onto |Y| when the
reference is binned in |Y|.

    source <scetlib-cms>/setup.sh
    python scripts/rabbit/scetlib_ad/compare_to_scetlib_run.py \
        --conf <cache>.conf --cache <cache>.npz \
        --reference <...>_combined.pkl --piece resummed
    python scripts/rabbit/scetlib_ad/compare_to_scetlib_run.py \
        --conf <cache>.conf --cache <cache>.npz \
        --reference <...>_CorrZ.pkl.lz4 --piece matched
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

from wremnants.postprocessing.scetlib_ad import params as adp  # noqa: E402
from wremnants.postprocessing.scetlib_ad.validation_plots import (  # noqa: E402
    ratio_range,
    warn_if_clipped,
)
from wremnants.postprocessing.scetlib_ad.xsec_backend import (  # noqa: E402
    ScetlibADXsec,
)

TOL = 1e-9


def load_reference(path, var):
    """(values (Y, qT), Q edges, Y edges, qT edges, is_absY, config) from either
    a SCETlib production pkl or a theory-correction pkl.lz4."""
    if path.endswith(".lz4"):
        import lz4.frame

        with lz4.frame.open(path, "rb") as fh:
            d = pickle.load(fh)
        proc = next(iter(d))  # "Z" / "W"
        block = d[proc]
        key = next(
            k for k in block if k.endswith("_hist") and not k.startswith("minnlo")
        )
        h, cfg = block[key], {}
        for v in (d.get("file_meta_data") or {}).values():
            if isinstance(v, dict) and "config" in v:
                cfg = v["config"]
                break
        kind = f"theory correction ({key})"
    else:
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        h, cfg = d["hist"], d.get("config", {})
        piece = cfg.get("Calculation_settings", {}).get("calculation_piece")
        kind = f"production run (calculation_piece={piece})"

    names = [a.name for a in h.axes]
    yname = "absY" if "absY" in names else "Y"
    sel = {}
    if "vars" in names:
        sel["vars"] = var
    if "charge" in names:
        sel["charge"] = sum  # a Z corr has one charge bin; sum is a no-op there
    hs = h[sel] if sel else h
    order = [a.name for a in hs.axes]
    vals = np.asarray(hs.values())
    # -> (Q, Y, qT)
    vals = np.transpose(vals, [order.index(n) for n in ("Q", yname, "qT")])
    return (
        vals,
        np.asarray(h.axes["Q"].edges),
        np.asarray(h.axes[yname].edges),
        np.asarray(h.axes["qT"].edges),
        yname == "absY",
        cfg,
        kind,
    )


def common_edges(a, b):
    """Edges present in both, over their overlapping range."""
    lo, hi = max(a[0], b[0]), min(a[-1], b[-1])
    out = [e for e in a if np.any(np.abs(b - e) < TOL) and lo - TOL <= e <= hi + TOL]
    if len(out) < 2:
        raise SystemExit(
            f"the two binnings share fewer than two edges in their overlap "
            f"[{lo:g}, {hi:g}]; they cannot be compared without interpolating."
        )
    return np.asarray(out)


def rebin(vals, axis, fine, coarse, what):
    """Sum `vals` along `axis` from `fine` edges onto `coarse` edges (exact)."""
    idx = np.full(fine.size - 1, -1, dtype=np.int64)
    for i in range(fine.size - 1):
        lo, hi = fine[i], fine[i + 1]
        for j in range(coarse.size - 1):
            if lo >= coarse[j] - TOL and hi <= coarse[j + 1] + TOL:
                idx[i] = j
                break
        else:
            if hi > coarse[0] + TOL and lo < coarse[-1] - TOL:
                raise SystemExit(
                    f"{what} bin [{lo:g}, {hi:g}] straddles a common edge; the sum "
                    f"onto the common grid would not be exact."
                )
    covered = np.zeros(coarse.size - 1)
    for i in np.nonzero(idx >= 0)[0]:
        covered[idx[i]] += fine[i + 1] - fine[i]
    want = np.diff(coarse)
    bad = np.abs(covered - want) > TOL * np.maximum(want, 1.0)
    if bad.any():
        j = int(np.argmax(bad))
        raise SystemExit(
            f"{what} common bin [{coarse[j]:g}, {coarse[j + 1]:g}] is not exactly "
            f"tiled ({covered[j]:g} of {want[j]:g})."
        )
    out = np.zeros(
        vals.shape[:axis] + (coarse.size - 1,) + vals.shape[axis + 1 :], dtype=float
    )
    v = np.moveaxis(vals, axis, 0)
    o = np.moveaxis(out, axis, 0)
    for i in np.nonzero(idx >= 0)[0]:
        o[idx[i]] += v[i]
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--conf", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument(
        "--piece",
        choices=["matched", "resummed"],
        default="matched",
        help="which part of OUR cache to compare: the matched total (against a "
        "theory correction) or the resummed piece alone (against a "
        "calculation_piece=sing production run)",
    )
    ap.add_argument("--var", default="central", help="reference vars entry")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--plot-dir", default=None)
    ap.add_argument(
        "--y-range",
        default=None,
        help="restrict the comparison to this |Y| (or Y) range, 'LO,HI'. Both "
        "endpoints must be edges of the common grid.",
    )
    ap.add_argument(
        "--qt-range",
        default=None,
        help="restrict the comparison to this qT range, 'LO,HI'. Both endpoints "
        "must be edges of the common grid.",
    )
    ap.add_argument(
        "--tag",
        default=None,
        help="suffix for the output filenames, so a restricted slice does not "
        "overwrite the full-range plots",
    )
    args = ap.parse_args()

    rvals, rQ, rY, rT, ref_absY, cfg, kind = load_reference(args.reference, args.var)
    print(f"reference: {kind}")

    core = ScetlibADXsec(args.conf, args.cache, threads=args.threads)
    ours = (
        core.resummed_only(core.anchor)
        if args.piece == "resummed"
        else core.values_and_jacobian(core.anchor)[0]
    )
    print(f"ours:      {args.piece} total from the cache")

    _report_config(cfg, core)

    # --- our grid, from the cache's own bin list
    b = core.bins
    ourQ = np.unique(b[:, 0:2], axis=0)
    if ourQ.shape[0] != 1:
        raise SystemExit("this script expects a single Q bin in the cache")
    oY = np.unique(b[:, 2:4], axis=0)
    oT = np.unique(b[:, 4:6], axis=0)
    oY, oT = oY[np.argsort(oY[:, 0])], oT[np.argsort(oT[:, 0])]
    Ye = np.concatenate([oY[:, 0], oY[-1:, 1]])
    Te = np.concatenate([oT[:, 0], oT[-1:, 1]])
    ours = np.asarray(ours).reshape(Te.size - 1, Ye.size - 1).T  # (Y, qT)

    # --- put ours on the reference's |Y| convention.  Two cache conventions
    # exist and they need opposite treatments, so decide from the cache's own
    # edges rather than assuming one (this is the same distinction
    # xsec_backend.GenFold draws as "signed" vs "positive-side-only", but here
    # there is no card to read it off, so it comes from the Y edges).
    if ref_absY:
        if np.allclose(Ye, -Ye[::-1]):
            n = (Ye.size - 1) // 2
            ours = ours[n:] + ours[:n][::-1]
            Ye = Ye[n:]
            print(f"   folded our signed Y onto |Y|: {n} bins")
        elif Ye[0] >= -TOL:
            # positive-side-only: the edges ARE |Y| edges already, but the cache
            # holds only the Y > 0 half of each, so it is half the |Y|-binned
            # cross section.  The comparison below is absolute, so this factor
            # is not cosmetic -- without it every bin reads -50%.
            ours = 2.0 * ours
            print(
                "   cache is positive-side-only: its Y edges are already |Y|, "
                "x2 for the mirrored half"
            )
        else:
            raise SystemExit(
                "the reference is binned in |Y| but the cache's Y grid is "
                f"neither symmetric about 0 nor positive-side-only: edges "
                f"{[float(e) for e in Ye]}"
            )

    # --- reference: pick the Q bin matching ours, never sum over Q
    q = next(
        (
            j
            for j in range(rQ.size - 1)
            if abs(rQ[j] - ourQ[0, 0]) < TOL and abs(rQ[j + 1] - ourQ[0, 1]) < TOL
        ),
        None,
    )
    if q is None:
        raise SystemExit(
            f"no reference Q bin matches ours [{ourQ[0, 0]:g}, {ourQ[0, 1]:g}]; "
            f"reference Q edges {list(rQ)}"
        )
    rvals = rvals[q]
    print(f"   reference Q bin {q} = [{rQ[q]:g}, {rQ[q + 1]:g}]")

    # --- sum both onto the common edges
    cY, cT = common_edges(Ye, rY), common_edges(Te, rT)
    cY = _restrict(cY, args.y_range, "Y")
    cT = _restrict(cT, args.qt_range, "qT")
    print(
        f"   common grid: {cY.size - 1} x {cT.size - 1} bins "
        f"(|Y| {cY[0]:g}..{cY[-1]:g}, qT {cT[0]:g}..{cT[-1]:g})"
    )
    ours_c = rebin(rebin(ours, 0, Ye, cY, "our Y"), 1, Te, cT, "our qT")
    ref_c = rebin(rebin(rvals, 0, rY, cY, "reference Y"), 1, rT, cT, "reference qT")

    rel = ours_c / np.where(ref_c != 0, ref_c, np.nan) - 1.0
    print(
        f"\ntotals: ours {ours_c.sum():.8g}   reference {ref_c.sum():.8g}   "
        f"ours/ref = {ours_c.sum() / ref_c.sum():.6f}"
    )
    print(
        f"per-bin ours/ref - 1: max |.| = {np.nanmax(np.abs(rel)):.3e}   "
        f"median |.| = {np.nanmedian(np.abs(rel)):.3e}"
    )
    print("\nby qT (max over |Y|):")
    for j in range(cT.size - 1):
        print(
            f"   qT [{cT[j]:6.1f},{cT[j + 1]:6.1f}]  "
            f"max |rel| = {np.nanmax(np.abs(rel[:, j])):+.3e}"
        )

    if args.plot_dir:
        _plot(ours_c, ref_c, cY, cT, args, kind)


def _restrict(edges, spec, what):
    """Clip a common-edge array to 'LO,HI', requiring both to be real edges."""
    if not spec:
        return edges
    lo, hi = (float(x) for x in spec.split(","))
    for v in (lo, hi):
        if not np.any(np.abs(edges - v) < TOL):
            raise SystemExit(
                f"{v:g} is not an edge of the common {what} grid "
                f"{[float(e) for e in edges]}; pick one of those."
            )
    out = edges[(edges >= lo - TOL) & (edges <= hi + TOL)]
    if out.size < 2:
        raise SystemExit(f"the requested {what} range leaves no bins")
    return out


def _report_config(cfg, core):
    """Cross-check the cache against the reference run's own settings.

    Uses the SAME tables the fit uses (``params.compare_corr_config``) rather
    than a list local to this script. The local list had drifted: it compared
    Nonperturbative in full but Calculation_settings against only twelve
    hand-picked keys, and never touched QCD at all -- where ``pdf_set`` and
    ``alphas_mu0`` live.
    """
    if not cfg:
        print("   (the reference records no config; nothing to cross-check)")
        return
    ours = {section: dict(core.conf[section]) for section in core.conf.sections()}
    theirs = {
        str(section): {str(k).lower(): v for k, v in body.items()}
        for section, body in cfg.items()
        if isinstance(body, dict)
    }
    refuse, warn, ref_only = adp.compare_corr_config(theirs, ours)
    print(
        f"   settings cross-check: "
        f"{'OK' if not refuse else f'{len(refuse)} DIFFER (would REFUSE a fit): {refuse}'}"
    )
    print(
        f"   parameter values: "
        f"{'all agree' if not warn else f'{len(warn)} DIFFER: {warn}'}"
    )
    if ref_only:
        print(f"   not compared ({len(ref_only)} reference-only): {sorted(ref_only)}")


def _plot(ours_c, ref_c, Ye, Te, args, kind):
    """The qT and |Y| spectra, each integrated over the other axis, with a
    ratio-to-reference panel."""
    import hist

    from wums import output_tools, plot_tools

    # save_pdf_and_png does not create the directory; write_index_and_log adds the
    # index.php gallery, which globs *.png and links each to its .log and .pdf.
    os.makedirs(args.plot_dir, exist_ok=True)
    tag = args.piece + (f"_{args.tag}" if args.tag else "")

    def h1(v, edges, name):
        h = hist.Hist(
            hist.axis.Variable(edges, name=name, overflow=False, underflow=False)
        )
        h.values()[...] = v
        return h

    meta = {
        "reference": args.reference,
        "reference kind": kind,
        "cache": args.cache,
        "runcard": args.conf,
        "piece": args.piece,
        "vars entry": args.var,
        "ours/reference (total)": f"{ours_c.sum() / ref_c.sum():.6f}",
    }

    # One panel per axis, each summed over the other.  binwnorm because both
    # binnings are strongly non-uniform, so raw contents would show the binning
    # rather than the spectrum.  NOT logx on qT: the first bin starts at qT = 0
    # and a log x-axis collapses the whole plot onto that edge.  logy on qT
    # only -- the |Y| spectrum spans well under a decade, where a log y hides
    # the shape instead of revealing it.
    for axname, axis, edges, xlabel, ylabel, logy in (
        (
            "qT",
            0,
            Te,
            r"boson $q_\mathrm{T}$ (GeV)",
            r"$d\sigma/dq_\mathrm{T}$ (a.u.)",
            True,
        ),
        (
            "absY",
            1,
            Ye,
            r"boson $|Y|$",
            r"$d\sigma/d|Y|$ (a.u.)",
            False,
        ),
    ):
        rr, _clipped = ratio_range(
            ours_c.sum(axis=axis)
            / np.where(ref_c.sum(axis=axis) != 0, ref_c.sum(axis=axis), np.nan)
        )
        warn_if_clipped(_clipped)
        fig = plot_tools.makePlotWithRatioToRef(
            [
                h1(ref_c.sum(axis=axis), edges, axname),
                h1(ours_c.sum(axis=axis), edges, axname),
            ],
            labels=["SCETlib reference", "autodiff cache"],
            colors=["#5790fc", "#e42536"],
            linestyles=["solid", "dashed"],
            xlabel=xlabel,
            ylabel=ylabel,
            rlabel=["cache / reference"],
            rrange=[rr],
            binwnorm=1,
            logy=logy,
            yerr=False,
            nlegcols=1,
            # ratio_legend=False: with fill_between=0 wums builds the ratio-panel
            # legend from colors[-fill_between::2], and -0 is 0, so it labels the
            # FIRST histogram as if it were an uncertainty band that was never
            # drawn. Both curves are already named in the upper legend.
            ratio_legend=False,
            logoPos=0,
            cms_label="Preliminary",
            grid=True,
        )
        plot_tools.save_pdf_and_png(args.plot_dir, f"{tag}_{axname}", fig=fig)
        output_tools.write_index_and_log(
            args.plot_dir, f"{tag}_{axname}", analysis_meta_info=meta, args=args
        )

    print(f"\n   plots -> {args.plot_dir}")


if __name__ == "__main__":
    main()
