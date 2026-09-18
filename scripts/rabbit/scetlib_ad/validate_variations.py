#!/usr/bin/env python3
"""Validate EVERY theory variation of the model against the old template.

The central-value validations (``compare_to_scetlib_run.py``, ``validate_reco.py``)
test one point. This tests the RESPONSE, which is what a fit actually uses: for
each variation the theory-correction file carries, compare

    model : sigma_gen(p_var) / sigma_gen(p_anchor)     -- the AD prediction
    ref   : Corr[var]        / Corr[central]           -- the template it replaces

Both sides are variation/central ratios, so no normalisation enters and the
comparison is non-circular. A flat 1.0 means the continuous direction reproduces
the discrete template it is meant to retire.

Needs no datacard: a gen-level ratio is cache + runcard only. The reference's
fine (absY, qT) bins are summed onto the cache's gen grid, which is exact
because both sides are bin-integrated -- summing numerator and denominator
separately, then dividing, which is what the fit's per-bin rnorm does.

Note the two JOINT variations (muf and kappaFO together): those test the model's
cross-term between muR and muF, which a template outer product cannot represent
and which is a large part of why we want the continuous treatment.
"""

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)

from wremnants.postprocessing.scetlib_ad.validation_plots import (  # noqa: E402
    ratio_range,
    warn_if_clipped,
)
from wremnants.postprocessing.scetlib_ad.xsec_backend import ScetlibADXsec  # noqa: E402

# Variation label -> {SCETlib parameter: PHYSICAL value}. Absolute values, not
# offsets. Derived from prod/scetlib_run's variations_resummed.conf; the beam
# TNPs are 'relative' mode at +-0.5, the others 'level0' at +-1.
VARIATIONS = {
    "lambda2_nu0.05": {"np_gnu_lambda2": 0.05},
    "lambda2_nu0.25": {"np_gnu_lambda2": 0.25},
    "lambda20.0": {"np_eff_lambda2": 0.0},
    "lambda21.0": {"np_eff_lambda2": 1.0},
    "delta_lambda2-0.02": {"np_eff_delta_lambda2": -0.02},
    "delta_lambda20.02": {"np_eff_delta_lambda2": 0.02},
    "lambda40.0": {"np_eff_lambda4": 0.0},
    "lambda41.0": {"np_eff_lambda4": 1.0},
    # kappaFO and kappaf move together so that muF is HELD: that combination is
    # what our single kappa_R direction represents (see the upstream commit that
    # made the scales differentiable -- a bare kappaFO is the wrong reference).
    "kappaFO0.5-kappaf2.": {"scale_kappa_R": 0.5},
    "kappaFO2.-kappaf0.5": {"scale_kappa_R": 2.0},
    # muf up/down is a full factor of 2 (Scale_provider.cpp: pow(2., _vary.muf),
    # enum up=+1 / down=-1). NB it also rescales muf_min, while our direction
    # comes from members built at kappa_F = 0.5/2.0 -- whether those agree
    # exactly is part of what this test measures.
    "mufdown": {"scale_kappa_F": 0.5},
    "mufup": {"scale_kappa_F": 2.0},
    "mufdown-kappaFO0.5-kappaf2.": {"scale_kappa_F": 0.5, "scale_kappa_R": 0.5},
    "mufup-kappaFO2.-kappaf0.5": {"scale_kappa_F": 2.0, "scale_kappa_R": 2.0},
    "transition_points0.2_0.35_1.0": {"scale_x2": 0.35},
    "transition_points0.2_0.75_1.0": {"scale_x2": 0.75},
    # old central values, a cross-check variation rather than an uncertainty --
    # and the only one that needs x1/x3, which are frozen in the fit by default.
    "transition_points0.3_0.6_0.9": {"scale_x1": 0.3, "scale_x3": 0.9},
}
for _t in ("gamma_cusp", "gamma_mu_q", "gamma_nu", "h_qqV", "s"):
    for _v in (-1.0, 1.0):
        VARIATIONS[f"{_t}{_v:g}."] = {f"tnp_{_t}": _v}
for _t in ("b_qqV", "b_qqbarV", "b_qqS", "b_qqDS", "b_qg"):
    for _v in (-0.5, 0.5):
        VARIATIONS[f"{_t}{_v:g}"] = {f"tnp_{_t}": _v}

CORR_HIST_SUFFIX = "_hist"
CENTRAL = "central"

# The alphaS variations live in a SEPARATE corr file (`*_pdfas_CorrZ`), whose
# labels carry the PDF set name -- pdfCT18ZNNLO_as_0116, or ALPHAS_116 for
# HERAPDF -- so they are resolved by pattern rather than listed. as_0118 is that
# file's CENTRAL, not a variation.
_AS_RE = re.compile(r"(?:_as_0|ALPHAS_)(\d{3})$", re.I)
# PDF eigenvectors live in `*_pdfvars_CorrZ`: pdf0 is the central and
# pdf(2i+1)/pdf(2i+2) are eigenvector i up/down, i.e. c_e = +-1. Needs a cache
# built with n_eig > 0; otherwise these are reported as skipped, not silently
# passed.
_PDF_RE = re.compile(r"^pdf(\d+)$")


def variation_for(label):
    """Label -> {SCETlib parameter: physical value}, or None if unmapped."""
    if label in VARIATIONS:
        return VARIATIONS[label]
    m = _AS_RE.search(label)
    if m:
        return {"alphas": float("0." + m.group(1))}
    m = _PDF_RE.match(label)
    if m:
        n = int(m.group(1))
        if n == 0:
            return None  # the central of that file
        i, side = (n - 1) // 2, (n - 1) % 2  # up first, then down
        return {f"pdf_eig{i}": 1.0 if side == 0 else -1.0}
    return None


def central_label(labels):
    """The label a file's variations are ratios to."""
    if CENTRAL in labels:
        return CENTRAL
    for cand in labels:
        m = _AS_RE.search(cand)
        if m and m.group(1) == "118":
            return cand
    if "pdf0" in labels:
        return "pdf0"
    raise SystemExit(f"cannot identify the central among {labels[:6]}")


def load_corr(path):
    """The theory-correction sigma hist (Q, absY, qT, charge, vars)."""
    import pickle

    import lz4.frame

    with lz4.frame.open(path, "rb") as f:
        d = pickle.load(f)
    boson = next(k for k in d if k in ("Z", "W", "Wplus", "Wminus"))
    inner = d[boson]
    key = next(k for k in inner if k.endswith(CORR_HIST_SUFFIX) and "minnlo" not in k)
    print(f"reference: {boson} / {key}")
    return inner[key]


def merge_matrix(fine, coarse, name, tol=1e-9):
    """(n_coarse, n_fine) 0/1 matrix summing fine bins into coarse ones."""
    fine = np.asarray(fine, float)
    coarse = np.asarray(coarse, float)
    M = np.zeros((coarse.size - 1, fine.size - 1))
    for k in range(coarse.size - 1):
        lo, hi = coarse[k], coarse[k + 1]
        idx = [
            i
            for i in range(fine.size - 1)
            if fine[i] >= lo - tol and fine[i + 1] <= hi + tol
        ]
        if not idx:
            raise SystemExit(f"{name}: no reference bins inside [{lo}, {hi}]")
        if abs(fine[idx[0]] - lo) > tol or abs(fine[idx[-1] + 1] - hi) > tol:
            raise SystemExit(
                f"{name}: coarse edges [{lo}, {hi}] are not reference edges; "
                f"the model grid must be a sub-binning of the correction's."
            )
        M[k, idx] = 1.0
    return M


def plot_response(label, Te, r_model, r_ref, outdir, meta):
    """Model vs template RESPONSE on one variation, |Y| integrated.

    Both curves are variation/central ratios. Y is integrated by summing sigma
    over |Y| for numerator and denominator SEPARATELY and then dividing -- not by
    averaging per-bin ratios, which would weight the low-yield forward bins the
    same as the peak.
    """
    import hist

    from wums import output_tools, plot_tools

    os.makedirs(outdir, exist_ok=True)

    def h1(v):
        h = hist.Hist(
            hist.axis.Variable(Te, name="qT", overflow=False, underflow=False),
            storage=hist.storage.Double(),
        )
        h.view(flow=False)[...] = v
        return h

    # Both curves are ratios, so centre the top panel on 1 rather than letting
    # matplotlib pick an offset range -- an off-centre axis makes a symmetric
    # response look like a trend. The floor keeps a near-null variation (e.g.
    # b_qqDS, which is identically 1) from getting a degenerate range.
    dev = max(
        float(np.max(np.abs(np.asarray(r_ref, float) - 1.0))),
        float(np.max(np.abs(np.asarray(r_model, float) - 1.0))),
    )
    pad = max(1.2 * dev, 2.0e-3)
    rr, _clipped = ratio_range(
        np.asarray(r_model, float)
        / np.where(np.asarray(r_ref, float) != 0, np.asarray(r_ref, float), np.nan)
    )
    warn_if_clipped(_clipped)
    fig = plot_tools.makePlotWithRatioToRef(
        [h1(r_ref), h1(r_model)],
        labels=[f"template  {label}", f"model  {label}"],
        ylim=[1.0 - pad, 1.0 + pad],
        # mplhep loc: 0 = above the axes, 2 = top-left INSIDE the box (default).
        # ratio_legend=False: with fill_between=0 wums builds the ratio-panel
        # legend from colors[-fill_between::2], and -0 is 0, so it labels the
        # FIRST histogram as if it were an uncertainty band that was never
        # drawn. Both curves are already named in the upper legend.
        ratio_legend=False,
        logoPos=0,
        colors=["#5790fc", "#e42536"],
        linestyles=["solid", "dashed"],
        xlabel=r"boson $q_\mathrm{T}$ (GeV)",
        ylabel=r"$\sigma_\mathrm{var}/\sigma_\mathrm{central}$",
        rlabel=["model / template"],
        rrange=[rr],
        binwnorm=None,
        logy=False,
        yerr=False,
        nlegcols=1,
        cms_label="Preliminary",
        grid=True,
    )
    safe = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
    plot_tools.save_pdf_and_png(outdir, f"var_{safe}", fig=fig)
    output_tools.write_index_and_log(
        outdir, f"var_{safe}", analysis_meta_info=meta, args=None
    )


def _plot_central(s_cen, r_cen, Te, outdir, meta, y_factor, tag):
    """Model vs reference CENTRAL cross section, |Y| integrated.

    Every variation plot is a variation/central ratio, so a common
    normalisation cancels there and the central is the one thing they cannot
    show. It is worth its own plot for that reason.

    y_factor converts the cache's rapidity convention to the reference's: a
    positive-side-only cache holds half the |Y|-integrated cross section, so it
    is short by exactly 2. That is a convention, not a disagreement, and it is
    applied here rather than divided out silently -- the ratio panel then shows
    SHAPE, which is the part that matters.
    """
    import hist

    from wums import output_tools, plot_tools

    os.makedirs(outdir, exist_ok=True)

    def h1(v):
        h = hist.Hist(
            hist.axis.Variable(Te, name="qT", overflow=False, underflow=False),
            storage=hist.storage.Double(),
        )
        h.view(flow=False)[...] = np.asarray(v, float)
        return h

    m = np.nansum(np.asarray(s_cen, float), axis=0) * y_factor
    r = np.nansum(np.asarray(r_cen, float), axis=0)
    rr2, _clipped = ratio_range(m / np.where(r != 0, r, np.nan))
    warn_if_clipped(_clipped)
    fig = plot_tools.makePlotWithRatioToRef(
        [h1(r), h1(m)],
        labels=["template  central", f"model  central (x{y_factor:g} for |Y|)"],
        # ratio_legend=False: with fill_between=0 wums builds the ratio-panel
        # legend from colors[-fill_between::2], and -0 is 0, so it labels the
        # FIRST histogram as if it were an uncertainty band that was never
        # drawn. Both curves are already named in the upper legend.
        ratio_legend=False,
        logoPos=0,
        colors=["#5790fc", "#e42536"],
        linestyles=["solid", "dashed"],
        xlabel=r"boson $q_\mathrm{T}$ (GeV)",
        ylabel=r"$\sigma$ (a.u.)",
        rlabel=["model / template"],
        rrange=[rr2],
        binwnorm=None,
        logy=True,
        yerr=False,
        nlegcols=1,
        cms_label="Preliminary",
        grid=True,
    )
    name = f"central_{tag}" if tag else "central"
    plot_tools.save_pdf_and_png(outdir, name, fig=fig)
    output_tools.write_index_and_log(outdir, name, analysis_meta_info=meta, args=None)
    ratio = m / np.where(r == 0, np.nan, r)
    print(
        f"  central shape, |Y| integrated: model/template "
        f"{np.nanmin(ratio):.4f} .. {np.nanmax(ratio):.4f}"
    )


def _one_file(
    path,
    todo,
    cen_lab,
    ref_on_grid,
    r_cen,
    s_cen,
    model_on_grid,
    Te,
    args,
    rows,
    skipped,
    cover,
):
    """Compare every mapped variation in one correction file."""
    for L in todo:
        ov = variation_for(L)
        if ov is None:
            skipped.append((L, "no mapping"))
            continue
        s_var = model_on_grid(ov)
        if s_var is None:
            skipped.append((L, f"cache lacks {list(ov)}"))
            continue
        rm = s_var / s_cen
        rr = ref_on_grid(L) / r_cen
        # cover excludes gen bins a --partial subset cache does not tile;
        # they fold to partial sums, so comparing them would invent a
        # disagreement that says nothing about the model.
        good = np.isfinite(rm) & np.isfinite(rr) & (rr != 0) & cover
        dev = np.abs(rm[good] / rr[good] - 1.0)
        # WHERE in qT the disagreement sits, which is the question that decides
        # whether a residual is the known low-qT cutoff mismatch or something
        # else. Arrays are (|Y|, qT) and C-ordered, so the qT index of the worst
        # bin is the flat argmax modulo the number of qT bins.
        with np.errstate(divide="ignore", invalid="ignore"):
            dev2 = np.where(good, np.abs(rm / rr - 1.0), np.nan)
        nqt = dev2.shape[1]
        iq = int(np.nanargmax(dev2) % nqt)
        rows.append((L, float(dev.max()), float(dev.mean())))
        print(
            f"{L:<32} {dev.max():10.2e} {dev.mean():10.2e} "
            f"[{rm[good].min():.4f},{rm[good].max():.4f}] "
            f"[{rr[good].min():.4f},{rr[good].max():.4f}] "
            f"{'[' + format(Te[iq], 'g') + ',' + format(Te[iq + 1], 'g') + ']':>12}"
        )
        if args.profile:
            per_qt = np.nanmax(dev2, axis=0)
            print("      qT profile of max|dev| over |Y|:")
            for k in range(nqt):
                bar = "#" * int(min(40, round(40 * per_qt[k] / np.nanmax(per_qt))))
                print(f"        [{Te[k]:6g},{Te[k + 1]:6g}] {per_qt[k]:10.2e} {bar}")
        if args.plot_dir:
            # Y-integrated response: sum sigma over |Y| first, then divide.
            rm1 = s_var.sum(axis=0) / s_cen.sum(axis=0)
            rr1 = ref_on_grid(L).sum(axis=0) / r_cen.sum(axis=0)
            plot_response(
                L,
                Te,
                rm1,
                rr1,
                args.plot_dir,
                {
                    "variation": L,
                    "model setting": str(ov),
                    "reference": os.path.basename(path),
                    "cache": os.path.basename(args.cache),
                    "both curves": "variation / central (a RESPONSE, not a xsec)",
                    "max|model/template - 1| (all bins)": f"{dev.max():.3e}",
                    "mean|.| (all bins)": f"{dev.mean():.3e}",
                },
            )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--corr",
        required=True,
        nargs="+",
        help="one or more scetlib_dyturbo_*_Corr<B>.pkl.lz4. Pass the matching "
        "*_pdfas_CorrZ alongside the main file to validate alphaS: the alphaS "
        "variations are NOT in the main correction.",
    )
    ap.add_argument("--cache", required=True)
    ap.add_argument("--conf", required=True)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument(
        "--partial",
        action="store_true",
        help="allow a cache that covers only SOME of the card's gen bins, and "
        "compare only the covered ones. For iterating on a subset cache built "
        "with prepare_cache_for_card --subset: ten bins in a minute instead of "
        "210 in seven hours. The number of excluded bins is always printed -- a "
        "subset must never be able to look like a clean full validation.",
    )
    ap.add_argument(
        "--only", nargs="*", default=None, help="restrict to these variation labels"
    )
    ap.add_argument("--plot-dir", default=None)
    ap.add_argument(
        "--profile",
        action="store_true",
        help="print the qT profile of max|dev| over |Y| for each variation, "
        "which separates a low-qT cutoff artefact from a genuine "
        "disagreement in the response",
    )
    args = ap.parse_args()

    core = ScetlibADXsec(args.conf, args.cache, threads=args.threads)
    names = list(core.param_names)
    print(f"cache: {core.n_bins} bins, {core.n_params} params")

    # the cache's gen grid, from the bins themselves
    b = core.bins
    yl = np.unique(np.round(b[:, 2:4], 12), axis=0)
    tl = np.unique(np.round(b[:, 4:6], 12), axis=0)
    yl, tl = yl[np.argsort(yl[:, 0])], tl[np.argsort(tl[:, 0])]
    Ye = np.concatenate([yl[:, 0], yl[-1:, 1]])
    Te = np.concatenate([tl[:, 0], tl[-1:, 1]])
    print(
        f"model grid: |Y| {Ye.size-1} bins [{Ye[0]:g}, {Ye[-1]:g}], "
        f"qT {Te.size-1} bins [{Te[0]:g}, {Te[-1]:g}]"
    )

    def make_reference(path):
        """(labels, central, ref_on_grid) for one correction file."""
        h = load_corr(path)
        ax = {a.name: a for a in h.axes}
        labels = [str(x) for x in ax["vars"]]
        vals = np.asarray(h.values(flow=False))
        dims = [a.name for a in h.axes]
        iQ, ich = dims.index("Q"), dims.index("charge")
        if vals.shape[iQ] != 1 or vals.shape[ich] != 1:
            raise SystemExit(f"{path}: expected a single Q and charge bin")
        vals = np.squeeze(vals, axis=(iQ, ich))
        order = [d for d in dims if d not in ("Q", "charge")]
        vals = np.moveaxis(
            vals,
            [order.index("absY"), order.index("qT"), order.index("vars")],
            [0, 1, 2],
        )
        MY = merge_matrix(ax["absY"].edges, Ye, "absY")
        MT = merge_matrix(ax["qT"].edges, Te, "qT")

        def ref_on_grid(label):
            return MY @ vals[:, :, labels.index(label)] @ MT.T  # (nY, nT)

        return labels, central_label(labels), ref_on_grid

    # GenFold indexes in the order the gen axes are GIVEN, and the cache was
    # built from the card as (ptVGen, absYVGen) -- gen shape (21, 10). Passing
    # them Y-first makes it read the Y edges as qT and reject the cache.
    fold = core.fold_for(
        [("ptVGen", Te), ("absYVGen", Ye)], b[0, 0], b[0, 1], partial=args.partial
    )
    # (|Y|, qT), matching the convention both sides are compared in below. With
    # --partial this excludes the gen bins a subset cache does not tile, which
    # would otherwise fold to partial sums and read as disagreement.
    cover = fold.covered_mask.T
    if not cover.all():
        print(
            f"   comparing {int(cover.sum())} of {cover.size} gen bins; "
            f"{int((~cover).sum())} not covered by this cache and EXCLUDED",
            flush=True,
        )
    anchor = core.anchor.copy()

    def model_on_grid(overrides):
        p = anchor.copy()
        for k, val in overrides.items():
            if k not in names:
                return None
            p[names.index(k)] = val
        vals_, _ = core.values_and_jacobian(p)
        # fold -> (ptVGen, absYVGen); transpose to the (|Y|, qT) convention the
        # reference side uses.
        return fold(np.asarray(vals_, float)).reshape(Te.size - 1, Ye.size - 1).T

    s_cen = model_on_grid({})

    print(
        f"\n{'variation':<32} {'max|dev|':>10} {'mean|dev|':>10} "
        f"{'model rng':>18} {'ref rng':>18} {'worst qT':>12}"
    )
    rows, skipped = [], []
    for path in args.corr:
        labels, cen_lab, ref_on_grid = make_reference(path)
        r_cen = ref_on_grid(cen_lab)
        # The CENTRAL, before any ratio. Every variation below is a
        # variation/central ratio, so a common normalisation cancels and this is
        # the one number that would not show up there. Reported as the spread of
        # model/reference over bins: a flat value means the shapes agree and the
        # offset is normalisation (the Y convention alone is a factor 2, and a
        # positive-side-only cache is short by exactly that), while a VARYING
        # value is a shape disagreement and matters.
        cc = np.where(
            cover & np.isfinite(s_cen) & np.isfinite(r_cen) & (r_cen != 0),
            s_cen / np.where(r_cen == 0, 1, r_cen),
            np.nan,
        )
        med = np.nanmedian(cc)
        print(
            f"  central model/reference: median {med:.6g}, "
            f"spread about it {np.nanmin(cc / med) - 1:+.2e} .. "
            f"{np.nanmax(cc / med) - 1:+.2e}"
        )
        todo = [
            L for L in labels if L != cen_lab and (args.only is None or L in args.only)
        ]
        if len(args.corr) > 1:
            print(f"  -- {os.path.basename(path)}  (central: {cen_lab})")
        if args.plot_dir:
            yfac = (
                2.0
                if getattr(fold, "y_convention", "") == "positive-side-only"
                else 1.0
            )
            _plot_central(
                s_cen,
                r_cen,
                Te,
                args.plot_dir,
                {
                    "what": "CENTRAL cross section, not a response",
                    "reference": f"{os.path.basename(path)} :: {cen_lab}",
                    "cache": os.path.basename(args.cache),
                    "Y convention": (
                        f"{getattr(fold, 'y_convention', '?')}, model scaled by "
                        f"x{yfac:g} to match the reference"
                    ),
                    "ratio panel shows": "SHAPE; the normalisation is the "
                    "convention factor above",
                },
                yfac,
                "pdfas" if "pdfas" in os.path.basename(path) else "",
            )
        _one_file(
            path,
            todo,
            cen_lab,
            ref_on_grid,
            r_cen,
            s_cen,
            model_on_grid,
            Te,
            args,
            rows,
            skipped,
            cover,
        )
    if skipped:
        print("\nskipped:")
        for L, why in skipped:
            print(f"   {L:<32} {why}")
    if rows:
        worst = max(rows, key=lambda r: r[1])
        print(
            f"\n{len(rows)} variations compared; worst max|dev| = "
            f"{worst[1]:.2e} ({worst[0]})"
        )


if __name__ == "__main__":
    main()
