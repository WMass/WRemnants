import itertools
import json

import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd

from utilities import parsing
from utilities.io_tools import conversion_tools, output_tools
from utilities.styles import styles
from wremnants import plot_tools
from wremnants.datasets.datagroups import Datagroups
from wums import boostHistHelpers as hh
from wums import logging

hep.style.use(hep.style.ROOT)

poi_type_choices = [
    "nois",
    "mu",
    "pmaskedexp",
    "pmaskedexpnorm",
    "sumpois",
    "sumpoisnorm",
    "chargepois",
    "ratiometapois",
    "helpois",
    "helmetapois",
]

parser = parsing.plot_parser()
parser.add_argument("infile", type=str, help="Combine fitresult file")
parser.add_argument(
    "--name", type=str, default="Unfolded data", help="Name for main source"
)
parser.add_argument(
    "--ylabel",
    type=str,
    default=None,
    help="Specify a y-axis label (if not it will be set automatic)",
)
parser.add_argument("--logy", action="store_true", help="Make the yscale logarithmic")
parser.add_argument("--noData", action="store_true", help="Don't plot data")
parser.add_argument(
    "--pulls",
    action="store_true",
    help="Make ratio as pulls between data and reference inputs",
)
parser.add_argument(
    "--plots",
    type=str,
    nargs="+",
    default=["xsec", "uncertainties"],
    choices=["xsec", "uncertainties", "ratio"],
    help="Define which plots to make",
)
parser.add_argument(
    "--selectionAxes",
    type=str,
    default=["qGen", "helicitySig", "A"],
    help="List of axes where for each bin a separate plot is created",
)
parser.add_argument(
    "--genFlow", action="store_true", help="Show overflow/underflow pois"
)
parser.add_argument(
    "--poiTypes",
    type=str,
    nargs="+",
    default=["pmaskedexp", "sumpois"],
    help="POI types used for the plotting",
    choices=poi_type_choices,
)
# specify a reference unfolding
parser.add_argument(
    "--reference",
    type=str,
    default=None,
    help="Optional combine fitresult file from an reference fit for comparison",
)
parser.add_argument(
    "--refName", type=str, default="Reference model", help="Name for reference source"
)
parser.add_argument(
    "--poiTypeReference",
    type=str,
    default=None,
    help="POI types used for the plotting of the reference",
    choices=poi_type_choices,
)

parser.add_argument(
    "--grouping",
    type=str,
    default=None,
    help="Select nuisances by a predefined grouping",
    choices=styles.nuisance_groupings.keys(),
)
parser.add_argument(
    "--ratioToPred", action="store_true", help="Use prediction as denominator in ratio"
)
parser.add_argument(
    "-t",
    "--translate",
    type=str,
    default=None,
    help="Specify .json file to translate labels",
)

# alternaitve predictions from histmaker output
parser.add_argument(
    "--histfile", type=str, help="Histogramer output file for comparisons"
)
parser.add_argument(
    "-n",
    "--baseName",
    type=str,
    help="Histogram name in the file (e.g., 'xnorm', 'nominal', ...)",
    default="xnorm",
)
parser.add_argument(
    "--varNames", default=["uncorr"], type=str, nargs="+", help="Name of variation hist"
)
parser.add_argument(
    "--varLabels",
    default=["MiNNLO"],
    type=str,
    nargs="+",
    help="Label(s) of variation hist for plotting",
)
parser.add_argument(
    "--varMarkers",
    default=["*", "v", "^", "x"],
    type=str,
    nargs="+",
    help="Label(s) of variation hist for plotting",
)
parser.add_argument(
    "--selectAxis",
    default=[],
    type=str,
    nargs="*",
    help="If you need to select a variation axis",
)
parser.add_argument(
    "--selectEntries",
    default=[],
    type=str,
    nargs="*",
    help="entries to read from the selected axis",
)
parser.add_argument(
    "--colors",
    default=[
        "red",
    ],
    type=str,
    nargs="+",
    help="Variation colors",
)

args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

if args.infile.endswith(".root"):
    args.infile = args.infile.replace(".root", ".hdf5")
if args.reference is not None and args.reference.endswith(".root"):
    args.reference = args.reference.replace(".root", ".hdf5")

result, meta = conversion_tools.fitresult_pois_to_hist(
    args.infile,
    poi_types=args.poiTypes,
    uncertainties=None,
    translate_poi_types=False,
    merge_gen_charge_W=False,
)

if args.reference:
    if args.poiTypeReference is None:
        poi_types_ref = args.poiTypes
    else:
        poi_types_ref = [
            args.poiTypeReference,
        ]
    result_ref, meta_ref = conversion_tools.fitresult_pois_to_hist(
        args.reference,
        poi_types=poi_types_ref,
        uncertainties=None,
        translate_poi_types=False,
        merge_gen_charge_W=False,
    )

grouping = styles.nuisance_groupings.get(args.grouping, None)

translate_label = {}
if args.translate:
    with open(args.translate) as f:
        translate_label = json.load(f)
    translate = lambda x: styles.translate_html_to_latex(translate_label.get(x, x))
else:
    translate = lambda x: x

outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)

if args.histfile:
    groups = Datagroups(args.histfile)
    groups.setNominalName(args.baseName)
    groups.setGenAxes([], [])
    groups.lumi = 0.001
    if "Wmunu" in groups.groups.keys():
        groups.copyGroup(
            "Wmunu",
            "Wminus",
            member_filter=lambda x: x.name.startswith("Wminus")
            and not x.name.endswith("OOA"),
        )
        groups.copyGroup(
            "Wmunu",
            "Wplus",
            member_filter=lambda x: x.name.startswith("Wplus")
            and not x.name.endswith("OOA"),
        )

    groups.set_rebin_action(["ptVGen", "absYVGen"], [0, 54, 0, 2.5], rename=False)


proc_dict = {"W_qGen0": "W^{-}", "W_qGen1": "W^{+}"}


def get_xlabel(names, proc=""):
    if len(names) > 1:
        label = f"{'-'.join([styles.xlabels.get(n,n).replace('(GeV)','') for n in names])} bin"
    else:
        label = styles.xlabels.get(names[0], names[0])
    if proc:
        label = label.replace(
            r"\mathrm{V}", r"\mathrm{" + proc_dict.get(proc, proc[0]) + "}"
        )
    return label


def make_yields_df(
    hists, procs, signal=None, per_bin=False, yield_only=False, percentage=True
):
    logger.debug(f"Make yield df for {procs}")
    if per_bin:

        def sum_and_unc(h, scale=100 if percentage else 1):
            return (h.values() * scale, np.sqrt(h.variances()) * scale)

    else:

        def sum_and_unc(h, scale=100 if percentage else 1):
            return (sum(h.values()) * scale, np.sqrt(sum(h.variances()) * scale))

    if per_bin:
        entries = [(i, v[0], v[1]) for i, v in enumerate(zip(*sum_and_unc(hists[0])))]
        index = "Bin"
    else:
        index = "Process"
        if signal is not None:
            entries = [
                (
                    signal,
                    sum([sum(v.values()) for k, v in zip(procs, hists) if signal in k]),
                    np.sqrt(
                        sum(
                            [
                                sum(v.variances())
                                for k, v in zip(procs, hists)
                                if signal in k
                            ]
                        )
                    ),
                )
            ]
        else:
            entries = [(k, *sum_and_unc(v)) for k, v in zip(procs, hists)]
    if yield_only:
        entries = [(e[0], e[1]) for e in entries]
        columns = [index, *procs]
    else:
        columns = [index, "Yield", "Uncertainty"]
    return pd.DataFrame(entries, columns=columns)


def plot_xsec_unfolded(
    hist_xsec,
    hist_xsec_stat=None,
    hist_ref=None,
    poi_type="mu",
    channel="ch0",
    proc="W",
    hist_others=[],
    label_others=[],
    marker_others=[],
    color_others=[],
    lumi=None,
    pulls=False,
):
    # ratio to data if there is no other histogram to make a ratio from
    ratioToData = not args.ratioToPred or len(hist_others) == 0

    logger.info(f"Make unfoled {poi_type} plot in channel {channel}")
    yLabel = args.ylabel if args.ylabel else styles.poi_types.get(poi_type, poi_type)

    # unroll histograms
    binwnorm = (
        None
        if poi_type
        in ["nois", "mu", "chargepois", "ratiometapois", "helpois", "helmetapois"]
        else 1
    )
    axes_names = hist_xsec.axes.name
    if len(axes_names) > 1:
        hist_xsec = hh.unrolledHist(
            hist_xsec, binwnorm=binwnorm, add_flow_bins=args.genFlow
        )
        hist_others = [
            hh.unrolledHist(h, binwnorm=binwnorm, add_flow_bins=args.genFlow)
            for h in hist_others
        ]
        if hist_ref is not None:
            hist_ref = hh.unrolledHist(
                hist_ref, binwnorm=binwnorm, add_flow_bins=args.genFlow
            )
        if hist_xsec_stat is not None:
            hist_xsec_stat = hh.unrolledHist(
                hist_xsec_stat, binwnorm=binwnorm, add_flow_bins=args.genFlow
            )

    xlabel = get_xlabel(axes_names, proc)

    edges = hist_xsec.axes.edges[0]
    binwidths = np.diff(edges)
    centers = hist_xsec.axes.centers[0]

    if binwnorm == 1:
        yLabel = yLabel  # +"/unit"
    else:
        binwidths = np.ones_like(binwidths)

    if pulls:
        rlabel = f"Pulls"
    elif ratioToData:
        # rlabel=f"{args.refName}/{args.name}"
        name = args.name.replace(" ", r"\ ")
        rlabel = rf"$1\,/\,{name}$"
    else:
        rlabel = rf"$Data\,/\,{label_others[0]}$"

    rrange = args.rrange

    unc = np.sqrt(hist_xsec.variances()) / binwidths
    y = hist_xsec.values() / binwidths

    if args.ylim is None:
        ymax = max(y + unc)
        ymin = min(y - unc)
        yrange = ymax - ymin
        ylim = (min(0, ymin - 0.1 * yrange), max(0, ymax + 0.1 * yrange))
    else:
        ylim = args.ylim

    # make plots
    fig, ax1, ratio_axes = plot_tools.figureWithRatio(
        hist_xsec,
        xlabel,
        yLabel,
        ylim,
        rlabel,
        rrange,
        automatic_scale=False,
        width_scale=2 if len(axes_names) > 1 else 1,
    )
    ax2 = ratio_axes[-1]

    ax1.hlines(y, edges[:-1], edges[1:], colors="black", label=args.name)
    ax1.bar(
        centers,
        height=2 * unc,
        bottom=y - unc,
        width=edges[1:] - edges[:-1],
        color="silver",
        label="Total unc.",
    )
    if hist_xsec_stat is not None:
        unc_stat = np.sqrt(hist_xsec_stat.variances()) / binwidths
        ax1.bar(
            centers,
            height=2 * unc_stat,
            bottom=y - unc_stat,
            width=edges[1:] - edges[:-1],
            color="gold",
            label="Stat unc.",
        )

    if args.genFlow:  # TODO TEST
        ax1.fill(
            [0, 18.5, 18.5, 0, 0],
            [ylim[0], *ylim, ylim[1], ylim[0]],
            color="grey",
            alpha=0.3,
        )
        ax1.fill(
            [
                len(edges) - 17.5,
                len(edges) + 0.5,
                len(edges) + 0.5,
                len(edges) - 17.5,
                len(edges) - 17.5,
            ],
            [ylim[0], *ylim, ylim[1], ylim[0]],
            color="grey",
            alpha=0.3,
        )

        ax2.fill(
            [0, 18.5, 18.5, 0, 0],
            [rrange[0], *rrange, rrange[1], rrange[0]],
            color="grey",
            alpha=0.3,
        )
        ax2.fill(
            [
                len(edges) - 17.5,
                len(edges) + 0.5,
                len(edges) + 0.5,
                len(edges) - 17.5,
                len(edges) - 17.5,
            ],
            [rrange[0], *rrange, rrange[1], rrange[0]],
            color="grey",
            alpha=0.3,
        )

    if ratioToData and not pulls:
        unc_ratio = unc / y
        ax2.bar(
            centers,
            height=2 * unc_ratio,
            bottom=1 - unc_ratio,
            width=edges[1:] - edges[:-1],
            color="silver",
            label="Total unc.",
        )
        if hist_xsec_stat is not None:
            unc_ratio_stat = unc_stat / y
            ax2.bar(
                centers,
                height=2 * unc_ratio_stat,
                bottom=1 - unc_ratio_stat,
                width=edges[1:] - edges[:-1],
                color="gold",
                label="Stat unc.",
            )

    if pulls:
        ax2.plot([min(edges), max(edges)], [0, 0], color="black", linestyle="-")
    else:
        ax2.plot([min(edges), max(edges)], [1, 1], color="black", linestyle="-")

    if ratioToData:
        hden = hist_xsec

    for i, (h, l, m, c) in enumerate(
        zip(hist_others, label_others, marker_others, color_others)
    ):

        y = h.values() / binwidths
        ax1.plot(centers, y, linewidth=0, marker=m, color=c, label=l)

        if not pulls:
            if i == 0 and not ratioToData:
                hden = h
                hep.histplot(
                    hh.divideHists(h, hden, cutoff=0, rel_unc=True),
                    yerr=False,
                    histtype="step",
                    color="black",
                    ax=ax2,
                    zorder=2,
                )
                continue

            y = hh.divideHists(h, hden, cutoff=0, rel_unc=True).values()
            ax2.plot(centers, y, linewidth=0, marker=m, color=c)

    if hist_ref is not None:
        y = hist_ref.values() / binwidths
        ax1.plot(centers, y, linewidth=0, marker="o", color="blue", label=args.refName)

        if pulls:
            hdiff = hh.addHists(hist_ref, hden, scale2=-1.0, flow=False)
            pull_values = hdiff.values() / np.sqrt(hden.variances())
            hdiff.values()[...] = pull_values

            logger.info(
                f"Min/Max pull value is {pull_values.min()}/{pull_values.max()}"
            )

            hep.histplot(
                hdiff,
                yerr=False,
                histtype="errorbar",
                color="black",
                # label="Model",
                ax=ax2,
            )
        else:
            hr = hh.divideHists(hist_ref, hden, cutoff=0, rel_unc=True)
            y = hr.values()
            ax2.plot(centers, y, linewidth=0, marker="o", color="blue")

    plot_tools.add_cms_decor(
        ax1, args.cmsDecor, data=not args.noData, lumi=lumi, loc=args.logoPos
    )
    plot_tools.addLegend(
        ax1, ncols=args.legCols, loc=args.legPos, text_size=args.legSize, reverse=False
    )
    plot_tools.fix_axes(ax1, ax2, fig, yscale=args.yscale, noSci=args.noSciy)

    outfile = f"unfolded_{poi_type}"
    outfile += f"_{proc}"
    outfile += "_" + "_".join(axes_names)
    outfile += f"_{channel}" if channel else ""
    outfile += f"_{args.postfix}" if args.postfix else ""
    if pulls:
        outfile += "_pulls"
    if args.cmsDecor == "Preliminary":
        outfile += "_preliminary"
    plot_tools.save_pdf_and_png(outdir, outfile)
    if hist_ref is not None:
        reference_yields = make_yields_df([hist_ref], ["Model"], per_bin=True)
        reference_yields[
            "Uncertainty"
        ] *= 0  # artificially set uncertainty on model hard coded to 0
    data_yields = make_yields_df([hist_xsec], ["Data"], per_bin=True)
    plot_tools.write_index_and_log(
        outdir,
        outfile,
        nround=2,
        yield_tables=(
            {"Data": data_yields, "Model": reference_yields}
            if hist_ref is not None
            else {"Data": data_yields}
        ),
        analysis_meta_info={args.infile: meta["meta_info"]},
        args=args,
    )
    plt.close()


def plot_uncertainties_unfolded(
    hist_xsec,
    hist_stat,
    hist_syst,
    poi_type,
    channel="ch0",
    proc="W",
    logy=False,
    relative_uncertainty=True,
    percentage=True,
    lumi=None,
    error_threshold=0.001,
    flow=False,  # only uncertainties are shown with a max error larger than this threshold
):
    logger.info(
        f"Make unfoled {poi_type} plot" + (f" in channel {channel}" if channel else "")
    )

    # read nominal values and uncertainties from fit result and fill histograms
    logger.debug(f"Produce histograms")

    yLabel = styles.poi_types.get(poi_type, poi_type)
    if relative_uncertainty:
        yLabel = r"$\delta$ " + yLabel
        yLabel = yLabel.replace(" [pb]", "")
        if percentage:
            yLabel += " [%]"
    else:
        yLabel = r"$\Delta$ " + yLabel

    axes_names = hist_xsec.axes.name
    xlabel = get_xlabel(axes_names, proc)

    scale = 100 if percentage else 1
    if relative_uncertainty:
        scale = scale / hist_xsec.values(flow=flow)  # /binwidths

    err = np.sqrt(hist_xsec.variances(flow=flow)) * scale
    err_stat = np.sqrt(hist_stat.variances(flow=flow)) * scale

    err_syst = hh.addHists(hist_syst, hist_xsec, scale2=-1, flow=flow).values(
        flow=flow
    ) * (scale[..., np.newaxis] if relative_uncertainty else scale)
    # err_syst = hist_syst.values(flow=flow) * (scale[...,np.newaxis] if relative_uncertainty else scale)

    hist_err = hist.Hist(*hist_xsec.axes, storage=hist.storage.Double())
    hist_err_stat = hist.Hist(*hist_stat.axes, storage=hist.storage.Double())
    hist_err_syst = hist.Hist(*hist_syst.axes, storage=hist.storage.Double())

    hist_err.view(flow=flow)[...] = err
    hist_err_stat.view(flow=flow)[...] = err_stat
    hist_err_syst.view(flow=flow)[...] = err_syst

    # unroll histograms
    binwnorm = (
        1
        if not relative_uncertainty
        and poi_type
        not in [
            "noi",
            "mu",
        ]
        else None
    )
    if len(axes_names) > 1:
        hist_err = hh.unrolledHist(
            hist_err, binwnorm=binwnorm, add_flow_bins=args.genFlow
        )
        hist_err_stat = hh.unrolledHist(
            hist_err_stat, binwnorm=binwnorm, add_flow_bins=args.genFlow
        )

    if args.ylim is None:
        if logy:
            ylim = (max(hist_err.values()) / 10000.0, 1000 * max(hist_err.values()))
        else:
            ylim = (0, 2 * max(hist_err.values()))
    else:
        ylim = args.ylim

    # make plots
    fig, ax1 = plot_tools.figure(
        hist_err, xlabel, yLabel, ylim, logy=logy, width_scale=2
    )
    hep.histplot(
        hist_err,
        yerr=False,
        histtype="step",
        color="black",
        label="Total",
        ax=ax1,
        alpha=1.0,
        # zorder=2,
    )
    hep.histplot(
        hist_err_stat,
        yerr=False,
        histtype="step",
        color="grey",
        label=translate("stat"),
        ax=ax1,
        alpha=1.0,
        # zorder=2,
    )
    uncertainties = make_yields_df(
        [hist_err],
        [
            "Total",
        ],
        per_bin=True,
        yield_only=True,
        percentage=percentage,
    )
    uncertainties["stat"] = make_yields_df(
        [hist_err_stat], ["stat"], per_bin=True, yield_only=True, percentage=percentage
    )["stat"]

    syst_labels = [
        x
        for x in hist_err_syst.axes["syst"]
        if (grouping is None or x in grouping) and x not in ["nominal", "stat", "total"]
    ]
    NUM_COLORS = len(syst_labels) - 2
    cm = mpl.colormaps["gist_rainbow"]
    # add each systematic uncertainty
    i = 0
    for syst in syst_labels:  # [:200:-1]:
        if len(axes_names) > 1:
            hist_err_syst_i = hh.unrolledHist(
                hist_err_syst[{"syst": syst}],
                binwnorm=binwnorm,
                add_flow_bins=args.genFlow,
            )
        else:
            hist_err_syst_i = hist_err_syst[{"syst": syst}]

        if max(hist_err_syst_i.values()) < error_threshold:
            logger.debug(
                f"Systematic {syst} smaller than threshold of {error_threshold}"
            )
            continue
        name = translate(syst)
        logger.debug(f"Plot systematic {syst} = {name}")

        if syst == "err_stat":
            color = "grey"
        else:
            color = cm(1.0 * i / NUM_COLORS)
            i += 1

        if i % 3 == 0:
            linestype = "-"
        elif i % 3 == 1:
            linestype = "--"
        else:
            linestype = ":"

        hep.histplot(
            hist_err_syst_i,
            yerr=False,
            histtype="step",
            color=color,
            linestyle=linestype,
            label=name,
            ax=ax1,
            alpha=1.0,
            # zorder=1,
        )

        unc_df = make_yields_df(
            [hist_err_syst_i], [name], per_bin=True, yield_only=True, percentage=True
        )
        uncertainties[name] = unc_df[name]

    if args.genFlow:
        ax1.fill(
            [0, 18.5, 18.5, 0, 0],
            [ylim[0], *ylim, ylim[1], ylim[0]],
            color="grey",
            alpha=0.3,
        )
        ax1.fill(
            [
                hist_err.size - 17.5,
                hist_err.size + 0.5,
                hist_err.size + 0.5,
                hist_err.size - 17.5,
                hist_err.size - 17.5,
            ],
            [ylim[0], *ylim, ylim[1], ylim[0]],
            color="grey",
            alpha=0.3,
        )

    if args.yscale:
        ymin, ymax = ax1.get_ylim()
        ax1.set_ylim(ymin, ymax * args.yscale)

    plot_tools.add_cms_decor(
        ax1, args.cmsDecor, data=not args.noData, lumi=lumi, loc=args.logoPos
    )
    plot_tools.addLegend(
        ax1, ncols=args.legCols, loc=args.legPos, text_size=args.legSize
    )
    plot_tools.fix_axes(ax1, None, fig, yscale=args.yscale, noSci=args.noSciy or logy)

    outfile = f"unfolded_{poi_type}{'_relative_' if relative_uncertainty else '_'}uncertainties"
    outfile += f"_{proc}"
    outfile += "_" + "_".join(axes_names)
    outfile += f"_{channel}" if channel else ""
    outfile += f"_logy" if logy else ""
    outfile += f"_{args.postfix}" if args.postfix else ""
    plot_tools.save_pdf_and_png(outdir, outfile)

    outfile += f"_{args.postfix}" if args.postfix else ""
    if args.cmsDecor == "Preliminary":
        outfile += "_preliminary"

    plot_tools.save_pdf_and_png(outdir, outfile)
    plot_tools.write_index_and_log(
        outdir,
        outfile,
        nround=2,
        yield_tables={
            f"Unfolded{' relative' if relative_uncertainty else ''} uncertainty{' [%]' if percentage else ''}": uncertainties
        },
        analysis_meta_info={args.infile: meta["meta_info"]},
        args=args,
    )

    plt.close()


def plot_uncertainties_with_ratio(
    hist_xsec,
    hist_xsec_ref,
    poi_type,
    poi_type_ref,
    hist_stat=None,
    hist_syst=None,
    hist_stat_ref=None,
    hist_syst_ref=None,
    logy=False,
    relative_uncertainty=True,
    percentage=True,
    lumi=None,
    channel="ch0",
    normalize=False,
    error_threshold=0.001,
    flow=False,
):
    logger.info(
        f"Make "
        + ("normalized " if normalize else "")
        + "unfoled xsec plot"
        + (f" in channel {channel}" if channel else "")
    )

    # read nominal values and uncertainties from fit result and fill histograms
    logger.debug(f"Produce histograms")

    yLabel = f"ref.({styles.poi_types[poi_type_ref]}) / nom.({styles.poi_types.get(poi_type, poi_type)}) -1"

    yLabel = styles.poi_types.get(poi_type, poi_type)
    if relative_uncertainty:
        yLabel = r"$\delta$ " + yLabel
        yLabel = yLabel.replace(" [pb]", "")
        if percentage:
            yLabel += " [%]"
    else:
        yLabel = r"$\Delta$ " + yLabel

    axes_names = hist_xsec.axes.name
    xlabel = get_xlabel(axes_names, proc)

    scale = 100 if percentage else 1
    if relative_uncertainty:
        scale = scale / hist_xsec.values(flow=flow)

    err = np.sqrt(hist_xsec.variances(flow=flow)) * scale
    hist_err = hist.Hist(*hist_xsec.axes, storage=hist.storage.Double())
    hist_err.view(flow=flow)[...] = err

    err_ref = np.sqrt(hist_xsec_ref.variances(flow=flow)) * scale
    hist_err_ref = hist.Hist(*hist_xsec.axes, storage=hist.storage.Double())
    hist_err_ref.view(flow=flow)[...] = err_ref

    if hist_stat is not None:
        err_stat = np.sqrt(hist_stat.variances(flow=flow)) * scale
        hist_err_stat = hist.Hist(*hist_stat.axes, storage=hist.storage.Double())
        hist_err_stat.view(flow=flow)[...] = err_stat

        err_stat_ref = np.sqrt(hist_stat_ref.variances(flow=flow)) * scale
        hist_err_stat_ref = hist.Hist(*hist_stat.axes, storage=hist.storage.Double())
        hist_err_stat_ref.view(flow=flow)[...] = err_stat_ref

    if hist_syst is not None:
        err_syst = hh.addHists(hist_syst, hist_xsec, scale2=-1, flow=flow).values(
            flow=flow
        ) * (scale[..., np.newaxis] if relative_uncertainty else scale)
        hist_err_syst = hist.Hist(*hist_syst.axes, storage=hist.storage.Double())
        hist_err_syst.view(flow=flow)[...] = err_syst

        err_syst_ref = hh.addHists(
            hist_syst_ref, hist_xsec, scale2=-1, flow=flow
        ).values(flow=flow) * (
            scale[..., np.newaxis] if relative_uncertainty else scale
        )
        hist_err_syst_ref = hist.Hist(*hist_syst.axes, storage=hist.storage.Double())
        hist_err_syst_ref.view(flow=flow)[...] = err_syst_ref

    # unroll histograms
    binwnorm = (
        1
        if not relative_uncertainty
        and poi_type
        not in [
            "noi",
            "mu",
        ]
        else None
    )
    if len(axes_names) > 1:
        hist_err = hh.unrolledHist(
            hist_err, binwnorm=binwnorm, add_flow_bins=args.genFlow
        )
        hist_err_ref = hh.unrolledHist(
            hist_err_ref, binwnorm=binwnorm, add_flow_bins=args.genFlow
        )
        if hist_stat is not None:
            hist_err_stat = hh.unrolledHist(
                hist_err_stat, binwnorm=binwnorm, add_flow_bins=args.genFlow
            )
            hist_err_stat_ref = hh.unrolledHist(
                hist_err_stat_ref, binwnorm=binwnorm, add_flow_bins=args.genFlow
            )

    if args.ylim is None:
        if logy:
            ylim = (max(hist_err.values()) / 10000.0, 1000 * max(hist_err.values()))
        else:
            ylim = (0, 1.1 * max(hist_err.values()))
    else:
        ylim = args.ylim

    name = args.name.replace("", r"\ ")
    rlabel = f"{args.refName}/{name}"
    rrange = args.rrange

    # make plots
    fig, ax1, ratio_axes = plot_tools.figureWithRatio(
        hist_err, xlabel, yLabel, ylim, rlabel, rrange, logy=logy, width_scale=2
    )
    ax2 = ratio_axes[-1]

    hep.histplot(
        [hist_err, hist_err_ref],
        yerr=False,
        histtype="step",
        color=["black", "black"],
        linestyle=["solid", "dashed"],
        label=[
            f"Total ({args.name})",
            f"Total ({args.refName})",
        ],
        ax=ax1,
        alpha=1.0,
        # zorder=2,
    )
    hep.histplot(
        hh.divideHists(hist_err_ref, hist_err, cutoff=0, rel_unc=True),
        yerr=False,
        histtype="step",
        color="black",
        ax=ax2,
        # zorder=2,
    )

    uncertainties = make_yields_df(
        [hist_err],
        [
            "Total",
        ],
        per_bin=True,
        yield_only=True,
        percentage=percentage,
    )

    if hist_stat is not None:
        hep.histplot(
            [hist_err_stat, hist_err_stat_ref],
            yerr=False,
            histtype="step",
            color=["grey", "grey"],
            linestyle=["solid", "dashed"],
            label=[
                f"{translate('stat')} ({args.name})",
                f"{translate('stat')} ({args.refName})",
            ],
            ax=ax1,
            alpha=1.0,
            # zorder=2,
        )
        hep.histplot(
            hh.divideHists(hist_err_stat_ref, hist_err_stat, cutoff=0, rel_unc=True),
            yerr=False,
            histtype="step",
            color="grey",
            ax=ax2,
            # zorder=2,
        )

        uncertainties["stat"] = make_yields_df(
            [hist_err_stat],
            ["stat"],
            per_bin=True,
            yield_only=True,
            percentage=percentage,
        )["stat"]

    edges = hist_err.axes.edges[0]
    ax2.plot([min(edges), max(edges)], [1, 1], color="black", linestyle="-")
    if hist_syst is not None:
        syst_labels = [
            x
            for x in hist_err_syst.axes["syst"]
            if (grouping is None or x in grouping)
            and x not in ["nominal", "stat", "total"]
        ]
        NUM_COLORS = len(syst_labels) - 2
        cm = mpl.colormaps["gist_rainbow"]
        # add each systematic uncertainty
        i = 0
        for syst in syst_labels[::-1]:
            if len(axes_names) > 1:
                hist_err_syst_i = hh.unrolledHist(
                    hist_err_syst[{"syst": syst}],
                    binwnorm=binwnorm,
                    add_flow_bins=args.genFlow,
                )
            else:
                hist_err_syst_i = hist_err_syst[{"syst": syst}]

            if max(hist_err_syst_i.values()) < error_threshold:
                logger.debug(
                    f"Systematic {syst} smaller than threshold of {error_threshold}"
                )
                continue
            name = translate(syst)
            logger.debug(f"Plot systematic {syst} = {name}")

            if syst == "err_stat":
                color = "grey"
            else:
                color = cm(1.0 * i / NUM_COLORS)
                i += 1

            if i % 3 == 0:
                linestype = "-"
            elif i % 3 == 1:
                linestype = "--"
            else:
                linestype = ":"

            hep.histplot(
                hist_err_syst_i,
                yerr=False,
                histtype="step",
                color=color,
                linestyle=linestype,
                label=name,
                ax=ax1,
                alpha=1.0,
                # zorder=1,
            )

            unc_df = make_yields_df(
                [hist_err_syst_i],
                [name],
                per_bin=True,
                yield_only=True,
                percentage=True,
            )
            uncertainties[name] = unc_df[name]

    if args.genFlow:
        ax1.fill(
            [0, 18.5, 18.5, 0, 0],
            [ylim[0], *ylim, ylim[1], ylim[0]],
            color="grey",
            alpha=0.3,
        )
        ax1.fill(
            [
                hist_err.size - 17.5,
                hist_err.size + 0.5,
                hist_err.size + 0.5,
                hist_err.size - 17.5,
                hist_err.size - 17.5,
            ],
            [ylim[0], *ylim, ylim[1], ylim[0]],
            color="grey",
            alpha=0.3,
        )

    if args.yscale:
        ymin, ymax = ax1.get_ylim()
        ax1.set_ylim(ymin, ymax * args.yscale)

    plot_tools.add_cms_decor(
        ax1, args.cmsDecor, data=not args.noData, lumi=lumi, loc=args.logoPos
    )
    plot_tools.addLegend(
        ax1, ncols=args.legCols, loc=args.legPos, text_size=args.legSize
    )
    plot_tools.fix_axes(ax1, None, fig, yscale=args.yscale, noSci=args.noSciy or logy)

    outfile = f"diff_unfolded_{poi_type}{'_relative_' if relative_uncertainty else '_'}uncertainties"
    outfile += f"_{proc}"
    outfile += "_" + "_".join(axes_names)
    outfile += f"_{channel}" if channel else ""
    outfile += f"_logy" if logy else ""
    outfile += f"_{args.postfix}" if args.postfix else ""
    plot_tools.save_pdf_and_png(outdir, outfile)

    outfile += f"_{args.postfix}" if args.postfix else ""
    if args.cmsDecor == "Preliminary":
        outfile += "_preliminary"

    plot_tools.save_pdf_and_png(outdir, outfile)
    plot_tools.write_index_and_log(
        outdir,
        outfile,
        nround=2,
        yield_tables={
            f"Unfolded{' relative' if relative_uncertainty else ''} uncertainty{' [%]' if percentage else ''}": uncertainties
        },
        analysis_meta_info={args.infile: meta["meta_info"]},
        args=args,
    )

    plt.close()


for poi_type, poi_result in result.items():

    for channel, channel_result in poi_result.items():
        lumi = None

        for proc, proc_result in channel_result.items():

            histo_others = []
            if args.histfile:
                groups_dict = {
                    "W": "Wmunu",
                    "W_qGen0": "Wminus",
                    "W_qGen1": "Wplus",
                    "Z": "Zmumu",
                }
                group_name = groups_dict[proc]
                for syst in args.varNames:
                    groups.loadHistsForDatagroups(
                        args.baseName,
                        syst=syst,
                        procsToRead=[group_name],
                        nominalIfMissing=False,
                    )
                    histo_other = groups.groups[group_name].hists[syst]
                    if "ptVGen" in histo_other.axes.name:
                        histo_other = hh.disableFlow(histo_other, "ptVGen")
                    histo_others.append(histo_other)

            for hist_name in filter(
                lambda x: not any([x.endswith(y) for y in ["_stat", "_syst"]]),
                proc_result.keys(),
            ):
                hist_nominal = result[poi_type][channel][proc][hist_name]
                hist_stat = result[poi_type][channel][proc][f"{hist_name}_stat"]
                # reference model
                if args.reference:
                    poi_type_ref = (
                        poi_type
                        if args.poiTypeReference is None
                        else args.poiTypeReference
                    )
                    # hist_ref = result_ref[poi_type_ref][channel][proc][hist_name]
                    # hist_ref_stat = result_ref[poi_type_ref][channel][proc][f"{hist_name}_stat"]

                    # copy to remove overflow
                    hist_ref = hist_nominal.copy()
                    hist_ref.view()[...] = result_ref[poi_type_ref][channel][proc][
                        hist_name
                    ].view()
                    hist_ref_stat = hist_nominal.copy()
                    hist_ref_stat.view()[...] = result_ref[poi_type_ref][channel][proc][
                        f"{hist_name}_stat"
                    ].view()

                if "ptVGen" in hist_nominal.axes.name:
                    hist_nominal = hh.disableFlow(hist_nominal, "ptVGen")
                if "ptVGen" in hist_stat.axes.name:
                    hist_stat = hh.disableFlow(hist_stat, "ptVGen")

                if args.selectAxis and args.selectEntries:
                    histo_others = [
                        h[{k: v}]
                        for h, k, v in zip(
                            histo_others, args.selectAxis, args.selectEntries
                        )
                    ]

                axes = hist_nominal.axes
                hists_others = [hh.projectNoFlow(h, axes.name) for h in histo_others]

                selection_axes = [a for a in axes if a.name in args.selectionAxes]
                if len(selection_axes) > 0:
                    selection_bins = [
                        np.arange(a.size) for a in axes if a.name in args.selectionAxes
                    ]
                    other_axes = [a for a in axes if a not in selection_axes]
                    iterator = itertools.product(*selection_bins)
                else:
                    iterator = [None]

                for bins in iterator:
                    h_ref = None
                    h_ref_stat = None
                    if bins == None:
                        suffix = channel
                        h_nominal = hist_nominal
                        h_stat = hist_stat
                        if args.reference:
                            h_ref = hist_ref
                            h_ref_stat = hist_ref_stat
                        h_others = hists_others
                    else:
                        idxs = {a.name: i for a, i in zip(selection_axes, bins)}
                        if len(other_axes) == 0:
                            continue
                        logger.info(
                            f"Make plot for axes {[a.name for a in other_axes]}, in bins {idxs}"
                        )
                        suffix = channel
                        for a, i in idxs.items():
                            if isinstance(hist_nominal.axes[a], hist.axis.Integer):
                                label = int(hist_nominal.axes[a].edges[i])
                                if a == "helicitySig":
                                    if i == 0:
                                        label = "SigmaUL"
                                    else:
                                        label = f"Sigma{label}"
                                else:
                                    label = f"{a}{label}"
                            else:
                                label = f"{a}{i}"
                            suffix += f"_{label}"
                        h_nominal = hist_nominal[idxs]
                        h_stat = hist_stat[idxs]
                        if args.reference:
                            h_ref = hist_ref[idxs]
                            h_ref_stat = hist_ref_stat[idxs]
                        h_others = [h[idxs] for h in hists_others]

                    if "xsec" in args.plots:
                        plot_xsec_unfolded(
                            h_nominal,
                            h_stat,
                            h_ref,
                            poi_type=poi_type,
                            channel=suffix,
                            proc=proc,
                            lumi=lumi,
                            hist_others=h_others,
                            label_others=args.varLabels,
                            marker_others=args.varMarkers,
                            color_others=args.colors,
                            pulls=args.pulls,
                        )

                    if "uncertainties" in args.plots:
                        h_systs = result[poi_type][channel][proc][f"{hist_name}_syst"]
                        if bins != None:
                            h_systs = h_systs[{**idxs, "syst": slice(None)}]

                        plot_uncertainties_unfolded(
                            h_nominal,
                            h_stat,
                            h_systs,
                            poi_type=poi_type,
                            channel=suffix,
                            proc=proc,
                            lumi=lumi,
                            relative_uncertainty=True,
                            # normalize=args.normalize, relative_uncertainty=not args.absolute,
                            logy=args.logy,
                        )

                    if "ratio" in args.plots:
                        poi_type_ref = (
                            poi_type
                            if args.poiTypeReference is None
                            else args.poiTypeReference
                        )
                        h_ref_systs = result_ref[poi_type_ref][channel][proc][
                            f"{hist_name}_syst"
                        ]
                        if bins != None:
                            h_ref_systs = h_ref_systs[{**idxs, "syst": slice(None)}]

                        plot_uncertainties_with_ratio(
                            h_nominal,
                            h_ref,
                            poi_type=poi_type,
                            poi_type_ref=poi_type_ref,
                            hist_stat=h_stat,
                            hist_stat_ref=h_ref_stat,
                            # hist_syst=h_syst, hist_syst_ref=h_ref_syst,
                            channel=channel,
                            # normalize=args.normalize, relative_uncertainty=not args.absolute,
                            logy=args.logy,
                            # process_label = process_label, axes=channel_axes
                        )

if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
    output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)
