#!/usr/bin/env python3
"""Validate the model's sigma_reco against the histmaker's corrected reco hist.

The reco counterpart of ``compare_to_scetlib_run.py`` (which validates sigma_gen
against a native SCETlib run):

  model : sigma_reco = R @ sigma_gen(p_anchor), with R = R_raw / N_gen read from
          the datacard's response auxiliary.
  ref   : the histmaker 'nominal' for the signal sample, whose central weight
          carries the scetlib_dyturbo theory correction (not --theoryCorrAltOnly),
          or (--reference card) the card's indata.norm signal column.

By default only the SHAPE is tested: summarize() applies one global scale and the
plots are density-normalized (--no-match-norm compares absolutely).

Pitfalls:
  * R sums helicitySig while N_gen takes UL (handled in response.py); taking UL
    of R discards the angular partition of the cosThetaStar*/phiStar* reco bins.
  * The gen ptVGen axis must carry the trailing overflow bin, and the cache must
    span it, or sigma_reco is low in the last ptll bins.
  * Never hist.project() an aligned/cropped hist: it folds the flow bins (the
    cropped content) back in. Use project_inrange().

Run inside the wmass singularity with the SCETlib setup sourced.
"""

import argparse
import os

import numpy as np

from wremnants.postprocessing.scetlib_ad.validation_plots import (
    ratio_range,
    warn_if_clipped,
)

NOMINAL_HIST = "nominal"
SIGNAL_SAMPLE = "Zmumu_2016PostVFP"
AXIS_LABELS = {
    "ptll": r"$p_{T}^{\ell\ell}$ [GeV]",
    "yll": r"$y^{\ell\ell}$",
    "cosThetaStarll_quantile": r"$\cos\theta^{*}$ quantile",
    "phiStarll_quantile": r"$\phi^{*}$ quantile",
}


def load_nominal(histmaker_path, sample_key, hist_name=NOMINAL_HIST):
    """Load the histmaker 'nominal' Hist for the signal sample."""
    import h5py

    from wums import ioutils as wums_io

    with h5py.File(histmaker_path, "r") as f:
        if sample_key not in f:
            raise KeyError(
                f"{histmaker_path}: no {sample_key!r} group. "
                f"top-level: {[k for k in f.keys()][:12]}"
            )
        sample = wums_io.pickle_load_h5py(f[sample_key])
        out = sample["output"]
        if hist_name not in out:
            raise KeyError(
                f"{sample_key}: no {hist_name!r} hist. available: "
                f"{list(out.keys())[:15]}"
            )
        proxy = out[hist_name]
        return proxy.get() if hasattr(proxy, "get") else proxy


def align_nominal(h, reco_axes_meta, tol=1e-6):
    """Reorder + crop the nominal Hist onto the model's reco binning.

    project(*names) reorders (summing out unlisted axes); an integer-bin slice
    per axis then crops a histmaker axis that is a superset of the fit axis. The
    cropped content lands in the flow bins, so use project_inrange downstream.
    """
    names = [n for n, _ in reco_axes_meta]
    have = [a.name for a in h.axes]
    missing = [n for n in names if n not in have]
    if missing:
        raise ValueError(f"nominal hist missing axes {missing}; has {have}")
    h = h.project(*names)

    crop = {}
    for name, medges in reco_axes_meta:
        medges = np.asarray(medges, dtype=np.float64)
        hedges = np.asarray(h.axes[name].edges, dtype=np.float64)
        nb = medges.size - 1
        hits = np.where(np.isclose(hedges, medges[0], atol=tol))[0]
        if hits.size == 0:
            raise ValueError(
                f"axis {name}: model low edge {medges[0]} not found in hist "
                f"edges [{hedges[0]} .. {hedges[-1]}]"
            )
        i0 = int(hits[0])
        if i0 + nb + 1 > hedges.size or not np.allclose(
            hedges[i0 : i0 + nb + 1], medges, atol=tol
        ):
            raise ValueError(
                f"axis {name}: hist edges from index {i0} don't match model "
                f"edges. hist={hedges[i0 : i0 + nb + 1]} model={medges}"
            )
        if i0 != 0 or nb != h.axes[name].size:
            crop[name] = slice(i0, i0 + nb)
    return h[crop] if crop else h


def project_inrange(h, axis):
    """1D Hist on ``axis``, summing the OTHER axes over in-range bins only."""
    import hist as _hist

    names = [a.name for a in h.axes]
    ai = names.index(axis)
    vals = h.values(flow=False)
    other = tuple(i for i in range(vals.ndim) if i != ai)
    out = _hist.Hist(
        _hist.axis.Variable(
            h.axes[axis].edges, name=axis, underflow=False, overflow=False
        ),
        storage=_hist.storage.Double(),
    )
    out.view(flow=False)[...] = vals.sum(axis=other)
    return out


def card_signal_column(indata, signal_proc_idx, reco_shape):
    """indata.norm's signal column for the single unmasked fit channel.

    Returns ``(column reshaped to reco_shape, lumi)``. Sliced by start:stop, not
    [:nbins]: with masked channels this channel sits at an offset.
    """
    unmasked = [i for i in indata.channel_info.values() if not i.get("masked", False)]
    if len(unmasked) != 1:
        raise SystemExit(
            f"--reference card needs exactly one unmasked channel, found {len(unmasked)}"
        )
    info = unmasked[0]
    norm = indata.norm
    if getattr(indata, "sparse", False):  # --sparse cards: a tf.SparseTensor
        import tensorflow as tf

        norm = tf.sparse.to_dense(tf.sparse.reorder(norm))
    if hasattr(norm, "numpy"):
        norm = norm.numpy()
    norm = np.asarray(norm, dtype=np.float64)
    col = norm[int(info["start"]) : int(info["stop"]), signal_proc_idx]
    return col.reshape(reco_shape), float(info["lumi"])


def summarize(model_sigma, nominal, names, match_norm=True):
    """Diagnostics between the two reco tensors.

    match_norm=True applies one global scale first (a SHAPE comparison); use
    match_norm=False when the absolute normalisation matters (k*sigma_SC).
    Returns the yield-weighted mean |ratio - 1|.
    """
    assert model_sigma.shape == nominal.shape, (model_sigma.shape, nominal.shape)
    m = model_sigma.astype(np.float64)
    n = nominal.astype(np.float64)
    msum, nsum = m.sum(), n.sum()
    print(f"\n  sum model sigma_reco : {msum:.6g}")
    print(f"  sum nominal          : {nsum:.6g}")
    print(f"  total model/nominal  : {msum / nsum:.6g}")
    if match_norm:
        m_scaled = m * (nsum / msum)
        print("  -> SHAPE mode: one global scale applied, total divided out")
    else:
        m_scaled = m
        print("  -> ABSOLUTE mode: no scale matching, the total counts")

    good = n > 0
    if int((~good).sum()):
        print(f"  ({int((~good).sum())} of {n.size} bins have nominal<=0; excluded)")
    r = (m_scaled[good] / n[good]).astype(np.float64)
    w = n[good]
    wmad = float(np.average(np.abs(r - 1.0), weights=w))
    lab = "scale*model / nominal" if match_norm else "model / nominal (ABSOLUTE)"
    print(f"\n  per-bin ratio ({lab}), should be ~1:")
    print(f"    bins           : {r.size}")
    print(f"    mean / median  : {r.mean():.5f} / {np.median(r):.5f}")
    print(f"    min / max      : {r.min():.5f} / {r.max():.5f}")
    for q in (1, 5, 50, 95, 99):
        print(f"    p{q:<2d}            : {np.percentile(r, q):.5f}")
    print(f"    YIELD-WEIGHTED mean|ratio-1| : {wmad:.5f}   <-- the headline")
    for ax, name in enumerate(names):
        other = tuple(i for i in range(m.ndim) if i != ax)
        mp, npj = m_scaled.sum(axis=other), n.sum(axis=other)
        with np.errstate(divide="ignore", invalid="ignore"):
            rp = np.where(npj > 0, mp / npj, np.nan)
        fin = rp[np.isfinite(rp)]
        dev = (
            f"{np.max(np.abs(fin - 1)):.5f}" if fin.size else "n/a (no populated bins)"
        )
        print(f"    projection {name:<26} max|ratio-1| = {dev}")
    return wmad


def plot_axis(
    model_h,
    ref_h,
    axis,
    outdir,
    tag,
    meta,
    density=True,
    ref_label="histmaker nominal (theory-corrected)",
):
    """Overlay + ratio panel on one reco axis.

    density=False keeps the absolute normalisation in the picture.
    """
    import hist

    from wums import output_tools, plot_tools

    os.makedirs(outdir, exist_ok=True)
    m1, r1 = project_inrange(model_h, axis), project_inrange(ref_h, axis)

    def dens(h):
        v = h.values(flow=False).astype(np.float64)
        out = hist.Hist(
            hist.axis.Variable(
                h.axes[axis].edges, name=axis, underflow=False, overflow=False
            ),
            storage=hist.storage.Double(),
        )
        out.view(flow=False)[...] = v / v.sum() if density else v
        return out

    # Range the panel on the ratio that is actually drawn (after dens()), not on
    # the raw m1/r1, which would carry the absolute normalisation offset.
    _mv = dens(m1).values(flow=False).astype(np.float64)
    _rv = dens(r1).values(flow=False).astype(np.float64)
    rr, _clipped = ratio_range(_mv / np.where(_rv != 0, _rv, np.nan))
    warn_if_clipped(_clipped)
    fig = plot_tools.makePlotWithRatioToRef(
        [dens(r1), dens(m1)],
        labels=[ref_label, "model $\\sigma_{reco}$"],
        colors=["#5790fc", "#e42536"],
        linestyles=["solid", "dashed"],
        xlabel=AXIS_LABELS.get(axis, axis),
        ylabel="normalized" if density else "yield",
        rlabel=["model / nominal"],
        rrange=[rr],
        binwnorm=1,
        logy=False,
        yerr=False,
        nlegcols=1,
        # with fill_between=0 wums would label the first histogram as a band
        ratio_legend=False,
        logoPos=0,
        cms_label="Preliminary",
        grid=True,
    )
    name = f"reco_{tag}_{axis}" if density else f"reco_{tag}_{axis}_abs"
    plot_tools.save_pdf_and_png(outdir, name, fig=fig)
    output_tools.write_index_and_log(outdir, name, analysis_meta_info=meta, args=None)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--datacard", required=True, help="reco card WITH the response auxiliary"
    )
    ap.add_argument(
        "--histmaker",
        default=None,
        help="histmaker hdf5 carrying reco 'nominal' (required for "
        "--reference histmaker)",
    )
    ap.add_argument("--cache", required=True)
    ap.add_argument("--conf", required=True)
    ap.add_argument("--sample", default=SIGNAL_SAMPLE)
    ap.add_argument("--hist", default=NOMINAL_HIST)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--plot-dir", default=None)
    ap.add_argument("--tag", default="nominal")
    ap.add_argument(
        "--fit-params",
        default="lambda2",
        help="which parameters to register. Affects ONLY the "
        "double-counting guards -- sigma_reco at the anchor is "
        "independent of it -- so the default is one lambda, letting the "
        "check run against a card that still carries pdfAlphaS / pdf* / "
        "resum* templates.",
    )
    ap.add_argument(
        "--reference",
        choices=("histmaker", "card"),
        default="histmaker",
        help="what to compare against. 'histmaker' = the theory-corrected reco "
        "'nominal' (needs --histmaker). 'card' = indata.norm's signal column, "
        "which is what the k*sigma_SC construction actually divides by; the "
        "model is then scaled by the PHYSICAL k = lumi*1000.",
    )
    ap.add_argument(
        "--y-fold",
        default="auto",
        help="factor putting sigma on the card's |Y| convention. 'auto' reads "
        "the cache's own declaration: a positive-side-only cache holds HALF the "
        "|Y|-binned cross section, so it needs 2.0. Only matters when the "
        "normalisation is not divided out, which is why the ratio construction "
        "never had to care.",
    )
    ap.add_argument(
        "--no-match-norm",
        dest="match_norm",
        action="store_false",
        help="do NOT apply a global scale before comparing, and do not "
        "density-normalize the plots: compare absolutely.",
    )
    ap.add_argument(
        "--plot-axes",
        nargs="*",
        default=["ptll", "yll"],
        help="reco axes to plot 1D ratios for",
    )
    args = ap.parse_args()

    from rabbit.inputdata import FitInputData
    from wremnants.postprocessing.scetlib_ad.param_model import SCETlibADParamModel

    if args.reference == "histmaker" and not args.histmaker:
        raise SystemExit("--reference histmaker needs --histmaker")
    print(f"datacard  : {args.datacard}")
    print(f"reference : {args.reference}")
    print(f"histmaker : {args.histmaker}")
    print(f"cache     : {args.cache}")
    indata = FitInputData(args.datacard)
    # jitCompile="off" is required: XLA cannot lower the model's tf.py_function.
    model = SCETlibADParamModel(
        indata,
        cache=args.cache,
        conf=args.conf,
        gen_level=0,
        threads=args.threads,
        fit_params=args.fit_params,
        # the POI must be in fit_params; which one is irrelevant at the anchor
        poi_params=args.fit_params.split(",")[0],
        jitCompile="off",
    )
    reco_axes = model._fit_axes(indata)
    names = [n for n, _ in reco_axes]
    print("reco axes : " + ", ".join(f"{n}({len(e)-1})" for n, e in reco_axes))
    print(f"reco shape: {model.reco_shape}   gen shape: {model.gen_shape}")

    sig = model.sigma_reco_central
    if sig is None:
        raise SystemExit("model.sigma_reco_central is None -- did gen_level stay set?")
    m = np.asarray(sig.numpy() if hasattr(sig, "numpy") else sig, dtype=np.float64)
    m = m.reshape(model.reco_shape)

    # pb -> event yield (k = lumi * 1000, times the |Y| fold) is a property of
    # the model, so it is applied for both references.
    _, lumi = card_signal_column(indata, model.signal_proc_idx, model.reco_shape)
    k = lumi * 1000.0
    conv = getattr(getattr(model, "_fold", None), "y_convention", "unknown")
    if args.y_fold == "auto":
        y_fold = 2.0 if conv == "positive-side-only" else 1.0
    else:
        y_fold = float(args.y_fold)
    print(f"k         : lumi*1000 = {lumi} * 1000 = {k:.6g}  (physical, pb->yield)")
    print(f"y fold    : x{y_fold:g}  (cache Y convention: {conv})")
    m = m * k * y_fold

    if args.reference == "card":
        import hist as _hist

        n, _ = card_signal_column(indata, model.signal_proc_idx, model.reco_shape)
        print(f"reference : card indata.norm, proc idx {model.signal_proc_idx}")
        ref = _hist.Hist(
            *[
                _hist.axis.Variable(e, name=nm, underflow=False, overflow=False)
                for nm, e in reco_axes
            ],
            storage=_hist.storage.Double(),
        )
        ref.view(flow=False)[...] = n
    else:
        print(f"reference : histmaker {args.hist!r} for {args.sample}")
        ref = align_nominal(
            load_nominal(args.histmaker, args.sample, args.hist), reco_axes
        )
        n = np.asarray(ref.values(flow=False), dtype=np.float64)
        # Fraction of the histmaker yield outside the card's fit range (dropped
        # from both sides, so no bias; a jump flags mismatched axes).
        raw = load_nominal(args.histmaker, args.sample, args.hist)
        tot_all = float(raw.values(flow=True).sum())
        tot_used = float(n.sum())
        print(
            f"reference coverage: {tot_used:.6e} of {tot_all:.6e} "
            f"({tot_used / tot_all:.4%}); {1 - tot_used / tot_all:.2%} of the "
            "histmaker yield is outside the card's fit range (dropped from BOTH "
            "sides)"
        )
    wmad = summarize(m, n, names, match_norm=args.match_norm)

    if args.plot_dir:
        import hist

        axes = [
            hist.axis.Variable(e, name=nm, underflow=False, overflow=False)
            for nm, e in reco_axes
        ]
        mh = hist.Hist(*axes, storage=hist.storage.Double())
        mh.view(flow=False)[...] = m
        meta = {
            "datacard": args.datacard,
            "histmaker": args.histmaker,
            "cache": args.cache,
            "runcard": args.conf,
            "hist": args.hist,
            "reference": args.reference,
            "norm matching": "on (SHAPE only)" if args.match_norm else "off (ABSOLUTE)",
            "y fold": str(args.y_fold),
            "yield-weighted mean|ratio-1|": f"{wmad:.5f}",
        }
        for ax in args.plot_axes:
            if ax in names:
                plot_axis(
                    mh,
                    ref,
                    ax,
                    args.plot_dir,
                    args.tag,
                    meta,
                    density=args.match_norm,
                    ref_label=(
                        "card nominal ($\\mathrm{indata.norm}$)"
                        if args.reference == "card"
                        else "histmaker nominal (theory-corrected)"
                    ),
                )
        print(f"\n   plots -> {args.plot_dir}")


if __name__ == "__main__":
    main()
