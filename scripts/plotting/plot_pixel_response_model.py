"""Response of the muons with (R1) and without (R0) a valid pixel hit against the model
fitted to it for the response-based pixel multiplicity weights
(wremnants/postprocessing/pixel_multiplicity_utils.py), from the correction file of
scripts/corrections/make_pixel_corrections.py.

One figure per eta bin with the generated-pt bins the model is fitted in, and one figure
with the parameters of the model against eta. The model drawn is the one of the correction
file: R1 a standardized shape with a width sigma1(genPt), R0 its convolution with a Gaussian
of mean mu and width delta(genPt).
"""

import argparse
import datetime
import os
import pickle

import lz4.frame
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from wremnants.postprocessing import pixel_multiplicity_utils
from wums import logging, output_tools, plot_tools

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

COLOR_R1 = "#1f4e9c"
COLOR_R0 = "#c8321e"
QOPR_LABEL = r"$r = (q/p)_{\mathrm{reco}}\,/\,(q/p)_{\mathrm{gen}}$"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--inputFile",
        type=str,
        required=True,
        help="Correction file of scripts/corrections/make_pixel_corrections.py (with the response histograms)",
    )
    parser.add_argument(
        "-o",
        "--outpath",
        type=str,
        default=None,
        help="Output directory. Default: ~/public_html/WMass/YYMMDD_<script>/",
    )
    parser.add_argument(
        "--postfix", type=str, default="", help="Suffix appended to output filenames"
    )
    parser.add_argument(
        "--etaBins",
        type=int,
        nargs="*",
        default=None,
        help="Indices of the eta bins to plot (default: all)",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="One figure per eta and generated-pt bin instead of one per eta bin",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=4.5,
        help="Half width of the plotted range of r around 1, in units of the width of R0",
    )
    parser.add_argument(
        "--nPoints",
        type=int,
        default=70,
        help="Approximate number of bins of r across the plotted range (bins are merged)",
    )
    parser.add_argument(
        "--noSummary",
        action="store_true",
        help="Skip the figure of the model parameters against eta",
    )
    parser.add_argument(
        "--title", type=str, default="CMS", help="Title for plot_tools.add_decor"
    )
    parser.add_argument(
        "--subtitle",
        type=str,
        default="Work in progress",
        help="Subtitle for plot_tools.add_decor (e.g., 'Preliminary')",
    )
    parser.add_argument(
        "--titlePos", type=int, default=0, help="Title position (mplhep convention)"
    )
    return parser.parse_args()


def default_outpath():
    stem = os.path.splitext(os.path.basename(__file__))[0]
    today = datetime.date.today().strftime("%y%m%d")
    return os.path.expanduser(f"~/public_html/WMass/{today}_{stem}/")


def response_fit_bins(genpt_edges, qopr_edges, pt_range, genpt_fit_range):
    """Generated-pt bins of the fit with the range of r that the pt cuts leave untruncated,
    as in pixel_multiplicity_utils.make_zero_pixel_probability"""
    fit_bins = []
    for g in range(len(genpt_edges) - 1):
        if (
            genpt_edges[g] >= genpt_fit_range[0]
            and genpt_edges[g + 1] <= genpt_fit_range[1]
        ):
            xlo, xhi = pixel_multiplicity_utils.untruncated_qopr_range(
                genpt_edges[g], genpt_edges[g + 1], *pt_range
            )
            fit_bins.append((g, (max(xlo, qopr_edges[0]), min(xhi, qopr_edges[-1]))))
    return fit_bins


def model_of_correction(counts, variances, genpt_edges, qopr_edges, fit_bins, info, ie):
    """The response model of eta bin `ie` with the parameters stored in the correction file.
    The shape of R1 is recomputed from the histograms; the fit is redone as a check."""
    model = pixel_multiplicity_utils.ResponseModel(
        counts, variances, genpt_edges, qopr_edges, fit_bins
    )
    stored = {
        "m1": info["m1"][ie],
        "mu": info["mu"][ie],
        "sigma1_coeff": np.array([info["sigma1_a2"][ie], info["sigma1_b2"][ie]]),
        "delta_coeff": np.array([info["delta_a2"][ie], info["delta_b2"][ie]]),
    }
    pt = model.fit_pt
    refit = (model.sigma1(pt), model.delta(pt), model.mu)
    for key, value in stored.items():
        setattr(model, key, value)
    diff = max(
        np.abs(model.sigma1(pt) / refit[0] - 1).max(),
        np.abs(model.delta(pt) / refit[1] - 1).max(),
        abs(model.mu - refit[2]),
    )
    return model, diff


def merge_bins(a, first, factor, nbins):
    return a[first : first + nbins * factor].reshape(nbins, factor).sum(axis=1)


def draw_panel(
    ax, rax, model, counts, variances, g, pt, fit_range, labels, window, npoints
):
    """R1 and R0 of one generated-pt bin with the model, and their ratios to the model.
    Returns the chi2 and the number of degrees of freedom of R0 in the fitted range, the
    range of the simulated points, and whether a range of r outside the fit is drawn."""
    edges = model.qopr_edges
    centers = 0.5 * (edges[1:] + edges[:-1])
    width = edges[1] - edges[0]
    c1, v1 = counts[g, :, 1], variances[g, :, 1]
    c0, v0 = counts[g, :, 0], variances[g, :, 0]
    xlo, xhi = fit_range
    sel = (centers > xlo) & (centers < xhi)

    # model, normalised to the simulated muons (R0: in the fitted range)
    p1 = model._prob(pt, 1)
    p0 = model._prob(pt, 0)
    e1 = c1.sum() * p1 / p1.sum()
    e0 = p0 * c0[sel].sum() / p0[sel].sum()

    # chi2 of R0 in the fitted range, in numbers of events, bins with enough of them
    w0 = v0.sum() / c0.sum()
    n_obs, n_exp = c0[sel] / w0, e0[sel] / w0
    ok = n_exp >= 5.0
    chi2, ndf = np.sum((n_obs[ok] - n_exp[ok]) ** 2 / n_exp[ok]), ok.sum() - 1

    # plotted range and merged bins
    lo, hi = max(1.0 - window, edges[0]), min(1.0 + window, edges[-1])
    first = int(np.searchsorted(edges, lo + 1e-9, side="right") - 1)
    last = int(np.searchsorted(edges, hi - 1e-9, side="left"))
    factor = max(1, int(round((last - first) / npoints)))
    nbins = (last - first) // factor
    medges = edges[first : first + nbins * factor + 1 : factor]
    mcenters = 0.5 * (medges[1:] + medges[:-1])
    merged = lambda a: merge_bins(a, first, factor, nbins)

    ymin, ymax = np.inf, 0.0
    w1 = v1.sum() / c1.sum()
    for c, v, e, wgt, color, marker, label in [
        (c1, v1, e1, w1, COLOR_R1, "o", labels[0]),
        (c0, v0, e0, w0, COLOR_R0, "s", labels[1]),
    ]:
        norm = c.sum() * width * factor
        y, yerr, ymod = merged(c) / norm, np.sqrt(merged(v)) / norm, merged(e) / norm
        shown = y > 0
        ax.errorbar(
            mcenters[shown],
            y[shown],
            yerr=yerr[shown],
            fmt=marker,
            ms=4,
            lw=1,
            color=color,
            mfc=color if marker == "o" else "none",
            label=label,
            zorder=3,
        )
        ax.stairs(ymod, medges, color=color, lw=2, zorder=2)
        ymin, ymax = min(ymin, y[shown].min()), max(ymax, y.max())
        # ratio to the model, where the model expects enough events
        enough = merged(e) / wgt >= 10.0
        expected = np.where(enough, merged(e), 1.0)
        rax.errorbar(
            mcenters[enough],
            (merged(c) / expected)[enough],
            yerr=(np.sqrt(merged(v)) / expected)[enough],
            fmt=marker,
            ms=4,
            lw=1,
            color=color,
            mfc=color if marker == "o" else "none",
            zorder=3,
        )
    # range of r that the pt cuts of the selection truncate: not used in the fit of R0
    shaded = False
    for a, b in [(medges[0], xlo), (xhi, medges[-1])]:
        if b > a:
            shaded = True
            for axis in (ax, rax):
                axis.axvspan(a, b, color="0.85", lw=0, zorder=0)
    rax.axhline(1.0, color="black", lw=1, zorder=1)
    ax.set_xlim(medges[0], medges[-1])
    return chi2, ndf, (ymin, ymax), shaded


def plot_eta_bin(outdir, args, ie, model, counts, variances, fit_bins, edges):
    genpt_edges, eta_edges = edges
    eta_lo, eta_hi = (round(float(x), 6) + 0.0 for x in eta_edges[ie : ie + 2])
    eta_label = rf"${eta_lo:.1f} < \eta < {eta_hi:.1f}$"
    eta_name = (
        f"eta{ie:02d}_{eta_lo:+.1f}_{eta_hi:+.1f}".replace("+", "p")
        .replace("-", "m")
        .replace(".", "p")
    )
    # same range of r for all generated-pt bins: the width of R0 at the highest one
    pt_max = model.fit_pt[-1]
    window = min(args.window * np.hypot(model.sigma1(pt_max), model.delta(pt_max)), 0.5)
    labels = (
        r"MC, $\geq 1$ valid pixel hit ($R_1$)",
        r"MC, no valid pixel hit ($R_0$)",
    )

    groups = (
        [[i] for i in range(len(fit_bins))]
        if args.separate
        else [list(range(len(fit_bins)))]
    )
    for group in groups:
        ncols = len(group)
        fig, axes = plt.subplots(
            2,
            ncols,
            figsize=(1.5 + 6.2 * ncols, 9.0),
            sharex="col",
            sharey="row",
            squeeze=False,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06, "wspace": 0.05},
        )
        ylims, any_shaded = [], False
        for col, i in enumerate(group):
            g, fit_range = fit_bins[i]
            pt = model.fit_pt[i]
            ax, rax = axes[0, col], axes[1, col]
            chi2, ndf, ylim, shaded = draw_panel(
                ax,
                rax,
                model,
                counts,
                variances,
                g,
                pt,
                fit_range,
                labels,
                window,
                args.nPoints,
            )
            ylims.append(ylim)
            any_shaded = any_shaded or shaded
            ax.text(
                0.04,
                0.965,
                "\n".join(
                    [
                        eta_label,
                        rf"${genpt_edges[g]:g} < p_{{\mathrm{{T}}}}^{{\mathrm{{gen}}}} < {genpt_edges[g + 1]:g}$ GeV",
                        rf"$\sigma_1 = {100 * model.sigma1(pt):.2f}\%$, $\delta = {100 * model.delta(pt):.2f}\%$",
                        rf"$\chi^2/\mathrm{{ndf}}\ (R_0) = {chi2:.0f}/{ndf}$",
                    ]
                ),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=19,
                linespacing=1.35,
            )
            rax.set_xlabel(QOPR_LABEL)
            rax.set_ylim(0.5, 1.5)
            rax.xaxis.set_major_locator(
                plt.MaxNLocator(5, prune="both" if ncols > 1 else None)
            )
        ymax = max(y[1] for y in ylims)
        ymin = max(0.5 * min(y[0] for y in ylims), 1e-7 * ymax)
        axes[0, 0].set_yscale("log")
        # room for the text above the distributions: the upper 38% of the panel
        axes[0, 0].set_ylim(ymin, ymax * (ymax / ymin) ** (0.38 / 0.62))
        axes[0, 0].set_ylabel(r"$1/N\ \mathrm{d}N/\mathrm{d}r$")
        axes[1, 0].set_ylabel("MC / model")
        # legend: simulation as markers, model as lines of the same colour
        handles, leg_labels = axes[0, 0].get_legend_handles_labels()
        handles += [plt.Line2D([], [], color="0.3", lw=2)]
        leg_labels += [r"model, $R_0 = R_1 \otimes G(\mu, \delta)$"]
        if any_shaded:
            handles += [plt.Rectangle((0, 0), 1, 1, color="0.85", lw=0)]
            leg_labels += [r"not fitted ($p_{\mathrm{T}}$ cuts)"]
        legend_style = {
            "fontsize": 18,
            "frameon": False,
            "handlelength": 1.5,
            "borderaxespad": 0.2,
            "columnspacing": 1.5,
            "handletextpad": 0.5,
        }
        if ncols > 1:
            # one line above the panels, next to the header
            axes[0, -1].legend(
                handles,
                leg_labels,
                loc="lower right",
                bbox_to_anchor=(1.0, 1.0),
                ncol=len(handles),
                **legend_style,
            )
        else:
            axes[0, -1].legend(
                handles,
                leg_labels,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                **legend_style,
            )
        plot_tools.add_decor(
            axes[0, 0],
            args.title,
            f"Simulation {args.subtitle}",
            data=False,
            lumi=None,
            loc=args.titlePos,
            no_energy=True,
        )
        name = "_".join(
            filter(
                None,
                [
                    "pixel_response",
                    eta_name,
                    (
                        f"genPt{genpt_edges[fit_bins[group[0]][0]]:g}to{genpt_edges[fit_bins[group[0]][0] + 1]:g}"
                        if args.separate
                        else None
                    ),
                    args.postfix,
                ],
            )
        )
        plot_tools.save_pdf_and_png(outdir, name, fig=fig)
        output_tools.write_logfile(
            outdir, name, args=args, wd=os.path.dirname(__file__)
        )
        plt.close(fig)


def plot_summary(outdir, args, models, frac0, eta_edges, pt=40.0):
    """Widths and shift of the model at a generated pt, and the fraction of muons without a
    valid pixel hit, against eta"""
    eta = 0.5 * (eta_edges[1:] + eta_edges[:-1])
    xerr = 0.5 * np.diff(eta_edges)
    sigma1 = 100 * np.array([m.sigma1(pt) for m in models])
    delta = 100 * np.array([m.delta(pt) for m in models])
    mu = 100 * np.array([m.mu for m in models])

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1, 1], "hspace": 0.06},
    )
    style = {"fmt": "o", "ms": 5, "lw": 1.5}
    axes[0].errorbar(
        eta,
        sigma1,
        xerr=xerr,
        color=COLOR_R1,
        label=r"$\sigma_1$: width of $R_1$",
        **style,
    )
    axes[0].errorbar(
        eta,
        delta,
        xerr=xerr,
        color=COLOR_R0,
        mfc="none",
        label=r"$\delta$: extra width of $R_0$",
        **{**style, "fmt": "s"},
    )
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.5, 60)
    axes[0].set_ylabel("Relative width [%]")
    axes[0].legend(loc="upper center", fontsize=20, frameon=False, ncol=2)
    axes[0].text(
        0.5,
        0.80,
        rf"$p_{{\mathrm{{T}}}}^{{\mathrm{{gen}}}} = {pt:g}$ GeV",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=20,
    )
    axes[1].errorbar(
        eta, mu, xerr=xerr, color=COLOR_R0, mfc="none", **{**style, "fmt": "s"}
    )
    axes[1].axhline(0.0, color="black", lw=1)
    axes[1].set_ylabel(r"Shift $\mu$ [%]", fontsize=22)
    axes[2].errorbar(eta, 100 * frac0, xerr=xerr, color="black", **style)
    axes[2].set_ylabel("No pixel hit [%]", fontsize=22)
    axes[2].set_ylim(0, None)
    axes[2].set_xlabel(r"$\eta$")
    axes[2].set_xlim(eta_edges[0], eta_edges[-1])
    plot_tools.add_decor(
        axes[0],
        args.title,
        f"Simulation {args.subtitle}",
        data=False,
        lumi=None,
        loc=args.titlePos,
        no_energy=True,
    )
    name = "_".join(filter(None, ["pixel_response_model_parameters", args.postfix]))
    plot_tools.save_pdf_and_png(outdir, name, fig=fig)
    output_tools.write_logfile(outdir, name, args=args, wd=os.path.dirname(__file__))
    plt.close(fig)


def main():
    args = parse_args()
    logging.setup_logger(os.path.basename(__file__), 3)
    with lz4.frame.open(args.inputFile, "rb") as f:
        res = pickle.load(f)
    for key in ("hPixelResponse_mc", "zeroPixelProbModel", "hNValidPixelHitsTrig_mc"):
        if key not in res:
            raise ValueError(
                f"{args.inputFile} has no '{key}': it needs the response histograms and the "
                "model of scripts/corrections/make_pixel_corrections.py"
            )
    hresponse = res["hPixelResponse_mc"]  # [eta, genPt, qopr, pixel class]
    info = res["zeroPixelProbModel"]
    eta_edges = np.asarray(hresponse.axes["eta"].edges)
    genpt_edges = np.asarray(hresponse.axes["genPt"].edges)
    qopr_edges = np.asarray(hresponse.axes["qopr"].edges)
    pt_edges = res["hNValidPixelHitsTrig_mc"].axes["pt"].edges
    # pt range of the selection of the response histograms (older files: that of the
    # pixel hit multiplicities)
    pt_range = res.get("meta", {}).get("responsePtRange", (pt_edges[0], pt_edges[-1]))
    fit_bins = response_fit_bins(
        genpt_edges,
        qopr_edges,
        pt_range,
        res.get("meta", {}).get("genPtFitRange", (34.0, 50.0)),
    )
    values, variances = hresponse.values(), hresponse.variances()

    outdir = output_tools.make_plot_dir(args.outpath or default_outpath())
    logger.info(f"Writing plots to {outdir}")

    eta_bins = range(len(eta_edges) - 1) if args.etaBins is None else args.etaBins
    models, maxdiff = {}, 0.0
    for ie in range(len(eta_edges) - 1):
        if ie not in eta_bins and args.noSummary:
            continue
        models[ie], diff = model_of_correction(
            values[ie], variances[ie], genpt_edges, qopr_edges, fit_bins, info, ie
        )
        maxdiff = max(maxdiff, diff)
        if ie in eta_bins:
            plot_eta_bin(
                outdir,
                args,
                ie,
                models[ie],
                values[ie],
                variances[ie],
                fit_bins,
                (genpt_edges, eta_edges),
            )
    logger.info(
        f"Largest difference between the parameters of the correction file and of a new fit: {maxdiff:.1e}"
    )
    if not args.noSummary:
        counts = values.sum(axis=(1, 2))  # [eta, pixel class]
        plot_summary(
            outdir,
            args,
            [models[ie] for ie in sorted(models)],
            counts[:, 0] / counts.sum(axis=1),
            eta_edges,
        )


if __name__ == "__main__":
    main()
