#!/usr/bin/env python3
r"""Validate EVERY theory variation of the scetlib_ad model AT RECO LEVEL.

The reco counterpart of ``scripts/rabbit/scetlib_ad/validate_variations.py``.
That script tests the gen-level response; this one folds through the datacard's
response matrix R and compares against the histmaker's OWN reco variations, so
it is the number a 2D (ptll, yll) fit actually uses.

Four objects per direction ``v``, all per reco bin ``b = (ptll, yll)``:

  model  r_mod(b) = [R @ sigma_gen(p_v)] / [R @ sigma_gen(p_anchor)]
                    what the fit actually evaluates
  r_A(b)          = [R @ (rho_ref . sigma_gen^anchor)] / [R @ sigma_gen^anchor]
                    the CORRECTION FILE's gen response, folded the way the model
                    folds -- i.e. weighted by the model's own gen spectrum
  r_B(b)          = [R @ (rho_ref . N_gen)] / [R @ N_gen]
                    the same response, but weighted by the MC's gen spectrum,
                    which is the weight the histmaker's own fold carries
  ref    r_ref(b) = H_v(b) / H_central(b)
                    from ``nominal_ptll_yll_<corr>``, the histmaker's PER-EVENT
                    reweighting by Corr[var]/Corr[central]

and the residual factorises exactly into three terms with three different fixes:

    r_mod / r_ref = (r_mod / r_A) x (r_A / r_B) x (r_B / r_ref)
                     \__ CALC __/   \__ WGT __/   \__ GRAIN __/

* CALC  -- the model's gen-level response against the correction file's. The
  reco image of the gen-level table. Fixed by SCETlib / the cache.
* WGT   -- the same response folded against two different gen weights. Nonzero
  only where the model's gen spectrum and the MC's disagree in shape INSIDE a
  reco bin. Fixed by making the model's central match the MC's, or by finer
  gen bins.
* GRAIN -- bin-averaged response against per-event. Zero iff the correction
  ratio is constant inside every gen bin.

  It is NOT simply "gen-binning granularity, fixed by finer bins", which is
  what this said until 260909-grain-zero measured it. Two regimes, and they
  have different cures:

  - Directions carried purely as a (Q, |Y|, qT) LOOKUP -- everything in the
    main correction file -- have GRAIN ~ 0 IDENTICALLY on this card, because
    with ``--responseGenBinning theoryCorr`` the card's gen grid IS the
    correction's own lookup grid, one cell per gen bin (measured: min = max = 1
    cell over all 770). Bin-averaged folding and per-event reweighting are then
    the same operation, so there is nothing for finer bins to fix. Measured
    GRAIN > CALC in 0 of 37 such directions.
  - The PDF directions (``_pdfvars`` / ``_pdfas``) are the only generators in
    ``theory_corr_weight_map``, so their templates carry a per-event
    ``LHEPdfWeight[i]``, a function of (x1, x2, Q). No (qT, |Y|) binning can
    resolve that, however fine -- so here GRAIN is real and finer gen bins are
    NOT the cure. Measured GRAIN > CALC in 47 of 60 such directions.

Note the CENTRAL prediction has essentially no GRAIN term: R is stored as
R_raw/N_gen, so ``R @ N_gen`` reconstructs the histmaker's reco nominal up to the
reco-selected events that have no gen column at all -- gen |Y| > 2.5, which the
card's gen grid drops. Measured, that is a nearly flat -7.6e-4 (max 2.3e-3), and
a flat offset is divided out by any shape comparison. Granularity therefore
enters only through the variations.

The qT [0,1] convention difference (production zeroes its nonsingular below
1.0 GeV, ours below 0.1) is handled explicitly rather than left to dominate a
headline: ``--fix-genbin0`` recomputes the model response with the gen qT [0,1]
row replaced by the reference's, so the difference between the two runs IS that
bin's contribution.

Run inside the wmass singularity with the SCETlib setup sourced.
"""

import argparse
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# repo root (scetlib_ad -> rabbit -> scripts -> repo), resolved the same way the
# sibling scripts do, so this runs from a checkout without WREM_BASE set
_WREM = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
for _p in (_WREM, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import validate_variations as VV  # noqa: E402  (the gen-level script, reused)

from wremnants.postprocessing.scetlib_ad.validation_plots import (  # noqa: E402
    ratio_range,
    warn_if_clipped,
)

NOMINAL_HIST = "nominal"
SIGNAL_SAMPLE = "Zmumu_2016PostVFP"
CORR_MAIN = (
    "nominal_ptll_yll_scetlib_dyturbo_LatticeNPLambda4Bugfix_FranksValsVars_"
    "CT18Z_N3p0LL_N2LO_Corr"
)
CORR_AS = CORR_MAIN.replace("_N2LO_Corr", "_N2LO_pdfas_Corr")


# ---------------------------------------------------------------------------
# histmaker side
# ---------------------------------------------------------------------------
def load_hist(histmaker_path, sample_key, hist_name):
    import h5py

    from wums import ioutils as wums_io

    with h5py.File(histmaker_path, "r") as f:
        sample = wums_io.pickle_load_h5py(f[sample_key])
        out = sample["output"]
        if hist_name not in out:
            raise KeyError(f"{sample_key}: no {hist_name!r}")
        proxy = out[hist_name]
        return proxy.get() if hasattr(proxy, "get") else proxy


def reco_var_tensor(h, reco_axes, tol=1e-6):
    """(n_ptll, n_yll, n_vars) values on the CARD's reco binning, plus labels.

    ``values(flow=False)`` already drops the flow bins, so the trailing ptll
    [44, inf) overflow and the yll under/overflow never enter -- this is the
    ``slice(a, b, sum)``-style handling the scetlib_np round needed, done by
    construction rather than by projection. An integer crop then handles the
    histmaker axis being a superset of the fit axis.
    """
    names = [a.name for a in h.axes]
    if "vars" not in names:
        raise ValueError(f"hist has no 'vars' axis; has {names}")
    labels = [str(x) for x in h.axes["vars"]]
    vals = np.asarray(h.values(flow=False), dtype=np.float64)
    # move to (ptll, yll, vars)
    order = [names.index(n) for n, _ in reco_axes] + [names.index("vars")]
    if len(order) != vals.ndim:
        raise ValueError(
            f"unexpected axes {names} for reco {[n for n, _ in reco_axes]}"
        )
    vals = np.transpose(vals, order)
    crop = []
    for name, medges in reco_axes:
        hedges = np.asarray(h.axes[name].edges, dtype=np.float64)
        medges = np.asarray(medges, dtype=np.float64)
        nb = medges.size - 1
        hits = np.where(np.isclose(hedges, medges[0], atol=tol))[0]
        if hits.size == 0:
            raise ValueError(f"axis {name}: model low edge {medges[0]} not in hist")
        i0 = int(hits[0])
        if not np.allclose(hedges[i0 : i0 + nb + 1], medges, atol=tol):
            raise ValueError(f"axis {name}: edges do not match the card's")
        crop.append(slice(i0, i0 + nb))
    return vals[tuple(crop) + (slice(None),)], labels


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def summarize(dev, w, mask=None):
    """(max, unweighted mean, yield-weighted mean) of |ratio - 1|."""
    d = np.asarray(dev, float)
    good = np.isfinite(d) if mask is None else (np.isfinite(d) & mask)
    if not good.any():
        return np.nan, np.nan, np.nan
    dd, ww = d[good], np.asarray(w, float)[good]
    return float(dd.max()), float(dd.mean()), float(np.average(dd, weights=ww))


def plot_direction(label, ptll_edges, r_mod, r_ref, r_fld, outdir, meta):
    """ptll-projected response: model vs histmaker reference, with ratio panel.

    Projected by summing the yield-weighted NUMERATOR and DENOMINATOR over yll
    separately and then dividing -- never by averaging per-bin ratios, which
    would give a forward-rapidity bin with 1% of the yield the same weight as
    the peak.
    """
    import hist

    from wums import output_tools, plot_tools

    os.makedirs(outdir, exist_ok=True)

    def h1(v):
        h = hist.Hist(
            hist.axis.Variable(
                ptll_edges, name="ptll", overflow=False, underflow=False
            ),
            storage=hist.storage.Double(),
        )
        h.view(flow=False)[...] = np.asarray(v, float)
        return h

    dev = max(float(np.max(np.abs(r_ref - 1.0))), float(np.max(np.abs(r_mod - 1.0))))
    pad = max(1.2 * dev, 2.0e-3)
    rrange_, _clipped = ratio_range(r_mod / r_ref, r_fld / r_ref)
    warn_if_clipped(_clipped)
    fig = plot_tools.makePlotWithRatioToRef(
        [h1(r_ref), h1(r_mod), h1(r_fld)],
        labels=[
            f"histmaker  {label}",
            f"model  {label}",
            "ref. gen response folded with our R  (fold closure)",
        ],
        ylim=[1.0 - pad, 1.0 + pad],
        # ratio_legend=False: with fill_between=0 wums builds the ratio-panel
        # legend from colors[-fill_between::2], and -0 is 0, so it labels the
        # FIRST histogram as if it were an uncertainty band that was never
        # drawn. Both curves are already named in the upper legend.
        ratio_legend=False,
        logoPos=0,
        colors=["#5790fc", "#e42536", "#964a8b"],
        linestyles=["solid", "dashed", "dotted"],
        xlabel=r"$p_{T}^{\ell\ell}$ [GeV]",
        ylabel=r"$N_\mathrm{var}/N_\mathrm{central}$ (reco)",
        rlabel=["/ histmaker"],
        rrange=[rrange_],
        binwnorm=None,
        logy=False,
        yerr=False,
        nlegcols=1,
        cms_label="Preliminary",
        grid=True,
    )
    safe = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
    plot_tools.save_pdf_and_png(outdir, f"recovar_{safe}", fig=fig)
    output_tools.write_index_and_log(
        outdir, f"recovar_{safe}", analysis_meta_info=meta, args=None
    )


def plot_map(res, ptll_edges, yll_edges, title, outfile, cbar="ratio $-$ 1  [%]"):
    """2D map of the residual. wums.makePlot2D builds the frame but never draws
    the values, so this one is deliberately bare matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    v = float(np.nanmax(np.abs(res)))
    pc = ax.pcolormesh(
        ptll_edges,
        yll_edges,
        (res * 100.0).T,
        cmap="RdBu_r",
        vmin=-100 * v,
        vmax=100 * v,
        shading="flat",
    )
    fig.colorbar(pc, ax=ax, label=cbar)
    ax.set_xlabel(r"$p_{T}^{\ell\ell}$ [GeV]")
    ax.set_ylabel(r"$y^{\ell\ell}$")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    fig.savefig(outfile.replace(".png", ".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--datacard", required=True)
    ap.add_argument("--histmaker", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--conf", required=True)
    ap.add_argument(
        "--corr",
        nargs="+",
        default=[CORR_MAIN, CORR_AS],
        help="reco variation hists in the histmaker output",
    )
    ap.add_argument("--sample", default=SIGNAL_SAMPLE)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--plot-dir", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument(
        "--fix-genbin0",
        action="store_true",
        help="replace the model's gen qT [0,1] response by the reference's "
        "before folding, isolating the known nonsingular-cutoff convention "
        "difference in that bin",
    )
    ap.add_argument("--csv", default=None, help="write the table here")
    args = ap.parse_args()

    from rabbit.inputdata import FitInputData
    from wremnants.postprocessing.scetlib_ad.param_model import SCETlibADParamModel

    indata = FitInputData(args.datacard)
    model = SCETlibADParamModel(
        indata,
        cache=args.cache,
        conf=args.conf,
        gen_level=0,
        threads=args.threads,
        fit_params="lambda2",
        poi_params="lambda2",
        jitCompile="off",
    )
    reco_axes = model._fit_axes(indata)
    R = np.asarray(model.R.numpy(), dtype=np.float64)  # (n_reco, n_gen)
    sg0 = np.asarray(model.sigma_gen_central_flat.numpy(), float)  # (n_gen,)
    sr0 = np.asarray(model.sigma_reco_central.numpy(), float)  # (n_reco,)
    core = model.core
    names = list(core.param_names)
    anchor = np.asarray(core.anchor, float)

    (qt_name, Te), (y_name, Ye) = model.gen_axes
    if not (qt_name.startswith("ptV") and y_name.startswith("absY")):
        raise SystemExit(
            f"expected gen axes (ptVGen, absYVGen), got ({qt_name}, {y_name}); "
            "the flattening order below assumes qT-major"
        )
    nT, nY = Te.size - 1, Ye.size - 1
    reco_shape = model.reco_shape
    ptll_edges, yll_edges = reco_axes[0][1], reco_axes[1][1]
    print(f"gen grid : {qt_name} {nT} bins, {y_name} {nY} bins -> {sg0.size}")
    print(f"reco grid: {reco_shape} -> {sr0.size} bins")

    def model_gen(overrides):
        p = anchor.copy()
        for k, val in overrides.items():
            if k not in names:
                return None
            p[names.index(k)] = val
        vals, _ = core.values_and_jacobian(p)
        return model._fold(np.asarray(vals, float))  # (n_gen,) as (qT, |Y|)

    def fold(sg):
        return (R @ sg) / sr0

    # The gen-side weights the HISTMAKER's own fold uses. R here is already
    # R_raw / N_gen, so R @ N_gen reconstructs sum_g R_raw(b, g), which IS the
    # histmaker's reco nominal -- verified numerically below. That identity is
    # what makes the three-term split meaningful rather than a relabelling.
    from wremnants.postprocessing.scetlib_ad.response import R_info_from_auxiliary

    N_gen = np.asarray(R_info_from_auxiliary(indata)["N_gen"], float).reshape(-1)
    reco_from_Ngen = R @ N_gen

    def fold_w(rho, weights):
        num = R @ (rho * weights)
        den = R @ weights
        return num / den

    # ---- reference: histmaker reco variations -----------------------------
    per_file = {}
    for hname in args.corr:
        h = load_hist(args.histmaker, args.sample, hname)
        vals, labels = reco_var_tensor(h, reco_axes)
        cen = VV.central_label(labels)
        per_file[hname] = (vals, labels, cen)
        print(f"reference: {hname}  ({len(labels)} vars, central={cen!r})")

    # sanity: the corr hist's central must BE the plain nominal, bin by bin.
    # If it is not, every reference response below is a ratio to the wrong
    # denominator and the whole table is meaningless -- so this is checked, not
    # assumed.
    hn = load_hist(args.histmaker, args.sample, NOMINAL_HIST)
    nom_names = [a.name for a in hn.axes]
    nv = np.asarray(hn.values(flow=False), float)
    keep_idx = [nom_names.index(n) for n, _ in reco_axes]
    drop = tuple(i for i in range(nv.ndim) if i not in keep_idx)
    if drop:
        nv = nv.sum(axis=drop)
    rem = [n for n in nom_names if nom_names.index(n) in keep_idx]
    nv = np.transpose(nv, [rem.index(n) for n, _ in reco_axes])
    nv = nv[: len(ptll_edges) - 1, : len(yll_edges) - 1]
    vals0, labels0, cen0 = per_file[args.corr[0]]
    c0 = vals0[..., labels0.index(cen0)]
    rel = np.abs(c0.sum() / nv.sum() - 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        perbin = np.abs(np.where(nv > 0, c0 / nv - 1.0, 0.0))
    print(
        f"sanity: corr-hist central vs plain nominal -- totals {rel:.2e}, "
        f"worst bin {perbin.max():.2e}"
    )
    # IDENTITY: R is stored as R_raw / N_gen, so R @ N_gen must reproduce the
    # histmaker's reco nominal exactly. If it does, the CENTRAL prediction has
    # no fold approximation at all and every granularity effect below belongs to
    # the variations -- which is the claim the three-term split rests on.
    with np.errstate(divide="ignore", invalid="ignore"):
        idr = np.where(nv > 0, reco_from_Ngen.reshape(nv.shape) / nv - 1.0, 0.0)
    print(
        f"identity: (R @ N_gen) vs histmaker nominal -- max|dev| "
        f"{np.abs(idr).max():.2e} over {nv.size} reco bins"
    )

    # ---- the gen-level reference on the model's gen grid -------------------
    corr_files = {}
    for pkl in _corr_pickles():
        try:
            labels, cen, on_grid = _gen_reference(pkl, Ye, Te)
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"   [gen ref] skipping {os.path.basename(pkl)}: {exc}")
            continue
        corr_files[pkl] = (labels, cen, on_grid)
        print(f"gen reference: {os.path.basename(pkl)} ({len(labels)} vars)")

    def gen_ref_response(label):
        for _pkl, (labels, cen, on_grid) in corr_files.items():
            if label in labels:
                num = on_grid(label)  # (nY, nT)
                den = on_grid(cen)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rho = np.where(den != 0, num / den, np.nan)
                return rho.T.reshape(-1)  # (qT, |Y|) flattened
        return None

    rows = []
    print(
        f"\n{'direction':<32} {'TOTAL max':>10} {'TOTAL wmean':>12} "
        f"{'CALC max':>9} {'CALC wmean':>10} {'WGT max':>9} "
        f"{'GRAIN max':>9} {'GRAIN wmean':>11} "
        f"{'response':>9} {'rel':>8} {'rel_calc':>8}"
    )
    if args.fix_genbin0:
        print(
            "    *** --fix-genbin0 IS ON: the model's gen qT [0,1] response was\n"
            "    REPLACED by the reference's, so the model agrees by construction in\n"
            "    the bin where it disagrees most. Numbers below are NOT a blind test. ***"
        )
    print(
        "   CALC  = model gen response vs the correction file's, folded\n"
        "   WGT   = same response, folded with our anchor spectrum vs with N_gen\n"
        "   GRAIN = bin-averaged response vs the histmaker's per-event one\n"
        "   rel   = TOTAL wmean divided by this direction's own response size"
    )
    print("-" * 138)
    for hname in args.corr:
        vals, labels, cen = per_file[hname]
        icen = labels.index(cen)
        ref_cen = vals[..., icen]
        w = ref_cen.reshape(-1).copy()
        for L in labels:
            if L == cen:
                continue
            if args.only is not None and L not in args.only:
                continue
            ov = VV.variation_for(L)
            if ov is None:
                continue
            sgv = model_gen(ov)
            if sgv is None:
                print(f"{L:<32} SKIPPED (cache lacks {list(ov)})")
                continue
            rho_ref = gen_ref_response(L)
            if rho_ref is not None:
                nbad = int(np.sum(~np.isfinite(rho_ref)))
                if nbad:
                    print(f"   [{L}] {nbad} gen bins with no reference; set to 1")
                    rho_ref = np.where(np.isfinite(rho_ref), rho_ref, 1.0)
            if args.fix_genbin0 and rho_ref is not None:
                sgv = sgv.copy()
                sgv.reshape(nT, nY)[0, :] = (
                    rho_ref.reshape(nT, nY)[0, :] * sg0.reshape(nT, nY)[0, :]
                )
            r_mod = fold(sgv)
            with np.errstate(divide="ignore", invalid="ignore"):
                r_ref = np.where(
                    ref_cen > 0, vals[..., labels.index(L)] / ref_cen, np.nan
                )
            r_ref = r_ref.reshape(-1)
            # r_A : the REFERENCE gen response, folded with the MODEL's own
            #       anchor spectrum as the gen weight (what the model does).
            # r_B : the same response folded with the MC's gen spectrum N_gen,
            #       which is the weight the histmaker's own fold carries.
            r_A = fold_w(rho_ref, sg0) if rho_ref is not None else None
            r_B = fold_w(rho_ref, N_gen) if rho_ref is not None else None
            r_fld = r_A  # kept for the plot: the reference response, our fold

            good = np.isfinite(r_ref) & (r_ref != 0) & (w > 0)
            tot = np.abs(r_mod / r_ref - 1.0)
            tmax, tmean, twmean = summarize(tot, w, good)
            if r_A is not None:
                gmax, _, gwmean = summarize(np.abs(r_mod / r_A - 1.0), w, good)
                wmax, _, wwmean = summarize(np.abs(r_A / r_B - 1.0), w, good)
                fmax, _, fwmean = summarize(np.abs(r_B / r_ref - 1.0), w, good)
            else:
                gmax = gwmean = wmax = wwmean = fmax = fwmean = np.nan
            # How big is this direction's response in the first place? An
            # absolute residual of 1e-3 is negligible on a 10% response and
            # fatal on a 0.3% one, so the table carries both. resp is the
            # yield-weighted mean |response - 1|; rel is the residual as a
            # FRACTION of it -- the quantity "the transitions are wrong by 30%
            # of their own response" refers to.
            _, _, resp = summarize(np.abs(r_ref - 1.0), w, good)
            rel = twmean / resp if resp > 0 else np.nan
            relg = gwmean / resp if resp > 0 else np.nan
            rows.append(
                dict(
                    direction=L,
                    total_max=tmax,
                    total_mean=tmean,
                    total_wmean=twmean,
                    calc_max=gmax,
                    calc_wmean=gwmean,
                    weight_max=wmax,
                    weight_wmean=wwmean,
                    grain_max=fmax,
                    grain_wmean=fwmean,
                    response_wmean=resp,
                    rel_total=rel,
                    rel_calc=relg,
                    fix_genbin0=int(bool(args.fix_genbin0)),
                )
            )
            print(
                f"{L:<32} {tmax:10.2e} {twmean:12.2e} {gmax:9.2e} {gwmean:10.2e} "
                f"{wmax:9.2e} {fmax:9.2e} {fwmean:10.2e} {resp:9.2e} "
                f"{rel:8.3f} {relg:8.3f}"
            )
            if args.plot_dir and r_fld is not None:
                num = (vals[..., labels.index(L)]).sum(axis=1)
                den = ref_cen.sum(axis=1)
                rref1 = num / den
                mm = (r_mod.reshape(reco_shape) * ref_cen).sum(axis=1) / den
                ff = (r_fld.reshape(reco_shape) * ref_cen).sum(axis=1) / den
                plot_direction(
                    L,
                    ptll_edges,
                    mm,
                    rref1,
                    ff,
                    args.plot_dir,
                    {
                        "direction": L,
                        "model setting": str(ov),
                        "reference": f"{os.path.basename(args.histmaker)} :: {hname}",
                        "cache": os.path.basename(os.path.dirname(args.cache)),
                        "curves": "reco variation/central RESPONSE, yll summed "
                        "(numerator and denominator separately)",
                        "TOTAL max|model/ref - 1| (2D)": f"{tmax:.3e}",
                        "TOTAL yield-weighted mean (2D)": f"{twmean:.3e}",
                        "CALC part (model vs correction file, max)": f"{gmax:.3e}",
                        "WEIGHT part (max)": f"{wmax:.3e}",
                        "GRAIN part (bin-averaged vs per-event, max)": f"{fmax:.3e}",
                        "gen qT[0,1] replaced by reference": str(args.fix_genbin0),
                    },
                )
            if args.plot_dir and L in ("pdfCT18ZNNLO_as_0120", "mufup", "lambda21.0"):
                plot_map(
                    (r_mod / r_ref).reshape(reco_shape) - 1.0,
                    ptll_edges,
                    yll_edges,
                    f"{L}: model / histmaker response $-$ 1",
                    os.path.join(
                        args.plot_dir, f"map_{re.sub(r'[^A-Za-z0-9]+','_',L)}.png"
                    ),
                )

    if rows:
        worst = max(rows, key=lambda r: r["total_max"])
        print(
            f"\n{len(rows)} directions compared. "
            f"worst TOTAL max|dev| = {worst['total_max']:.2e} ({worst['direction']})"
        )
        arr = np.array([r["total_max"] for r in rows])
        print(
            f"  TOTAL max|dev|: median {np.median(arr):.2e}, "
            f"90th pct {np.percentile(arr, 90):.2e}"
        )
        gm = np.array([r["calc_max"] for r in rows], float)
        wmv = np.array([r["weight_max"] for r in rows], float)
        fm = np.array([r["grain_max"] for r in rows], float)
        print(
            f"  CALC  max|dev|: median {np.nanmedian(gm):.2e}, worst {np.nanmax(gm):.2e}"
        )
        print(
            f"  WGT   max|dev|: median {np.nanmedian(wmv):.2e}, worst {np.nanmax(wmv):.2e}"
        )
        print(
            f"  GRAIN max|dev|: median {np.nanmedian(fm):.2e}, worst {np.nanmax(fm):.2e}"
        )
        print(f"  directions where GRAIN > CALC: {int(np.sum(fm > gm))} of {len(rows)}")
    if args.csv and rows:
        import csv as _csv

        with open(args.csv, "w", newline="") as fh:
            wr = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"table -> {args.csv}")


# ---------------------------------------------------------------------------
def _corr_pickles():
    """The theory-correction pickles the gen-level reference is read from.

    ``_pdfvars`` is needed for the 58 PDF eigenvector directions to get a
    CALC/WGT/GRAIN split rather than only a total residual: the split needs the
    correction file's own gen response for that direction. A pickle that is not
    on disk is skipped with a message by the caller, so listing it costs
    nothing when it is absent.
    """
    base = os.path.join(_WREM, "wremnants-data", "data", "TheoryCorrections")
    out = []
    for tag in ("", "_pdfas", "_pdfvars"):
        p = os.path.join(
            base,
            "scetlib_dyturbo_LatticeNPLambda4Bugfix_FranksValsVars_CT18Z_"
            f"N3p0LL_N2LO{tag}_CorrZ.pkl.lz4",
        )
        if os.path.exists(p):
            out.append(p)
    if not out:
        raise SystemExit(f"no correction pickles under {base}")
    return out


def _gen_reference(path, Ye, Te):
    """(labels, central_label, on_grid) for one correction pickle."""
    h = VV.load_corr(path)
    ax = {a.name: a for a in h.axes}
    labels = [str(x) for x in ax["vars"]]
    vals = np.asarray(h.values(flow=False))
    dims = [a.name for a in h.axes]
    iQ, ich = dims.index("Q"), dims.index("charge")
    vals = np.squeeze(vals, axis=(iQ, ich))
    order = [d for d in dims if d not in ("Q", "charge")]
    vals = np.moveaxis(
        vals,
        [order.index("absY"), order.index("qT"), order.index("vars")],
        [0, 1, 2],
    )
    MY = VV.merge_matrix(ax["absY"].edges, Ye, "absY")
    MT = VV.merge_matrix(ax["qT"].edges, Te, "qT")

    def on_grid(label):
        return MY @ vals[:, :, labels.index(label)] @ MT.T

    return labels, VV.central_label(labels), on_grid


if __name__ == "__main__":
    main()
