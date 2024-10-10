import itertools
import os

import hist
import mplhep as hep
import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import colormaps
from matplotlib.lines import Line2D

from narf import ioutils
from utilities import boostHistHelpers as hh
from utilities import common, differential, logging
from utilities.io_tools import combinetf_input, input_tools, output_tools
from utilities.styles import styles
from wremnants import plot_tools

hep.style.use(hep.style.ROOT)

parser = common.plot_parser()
parser.add_argument(
    "infile", type=str, help="hdf5 file from combinetf2 or root file from combinetf1"
)
parser.add_argument(
    "--uncFile",
    type=str,
    default=None,
    help="Take the uncertainties from a second file",
)
parser.add_argument("--logy", action="store_true", help="Make the yscale logarithmic")
parser.add_argument(
    "--noLowerPanel", action="store_true", help="Don't plot the lower panel in the plot"
)
parser.add_argument(
    "--logTransform", action="store_true", help="Log transform the events"
)
parser.add_argument("--noData", action="store_true", help="Don't plot the data")
parser.add_argument(
    "--noUncertainty", action="store_true", help="Don't plot total uncertainty band"
)
parser.add_argument("--normToData", action="store_true", help="Normalize MC to data")
parser.add_argument(
    "--prefit", action="store_true", help="Make prefit plot, else postfit"
)
parser.add_argument(
    "--filterProcs",
    type=str,
    nargs="*",
    default=None,
    help="Only plot the filtered processes",
)
parser.add_argument(
    "--selectionAxes",
    type=str,
    default=["charge", "passIso", "passMT", "cosThetaStarll"],
    help="List of axes where for each bin a separate plot is created",
)
parser.add_argument(
    "--axlim", type=float, nargs="*", help="min and max for axes (2 values per axis)"
)
parser.add_argument(
    "--invertAxes",
    action="store_true",
    help="Invert the order of the axes when plotting",
)
parser.add_argument(
    "--noChisq", action="store_true", help="skip printing chisq on plot"
)
parser.add_argument(
    "--dataName", type=str, default="Data", help="Data name for plot labeling"
)
parser.add_argument(
    "--xlabel", type=str, default=None, help="x-axis label for plot labeling"
)
parser.add_argument(
    "--ylabel", type=str, default=None, help="y-axis label for plot labeling"
)
parser.add_argument(
    "--processGrouping", type=str, default=None, help="key for grouping processes"
)
parser.add_argument(
    "--binSeparationLines",
    type=float,
    default=None,
    nargs="*",
    help="Plot vertical lines for makro bin edges in unrolled plots, specify bin boundaries to plot lines, if empty plot for all",
)
parser.add_argument(
    "--extraTextLoc",
    type=float,
    nargs=2,
    default=None,
    help="Location in (x,y) for additional text, aligned to upper left",
)
parser.add_argument(
    "--varNames", type=str, nargs="*", default=None, help="Name of variation hist"
)
parser.add_argument(
    "--varLabels",
    type=str,
    nargs="*",
    default=None,
    help="Label(s) of variation hist for plotting",
)
parser.add_argument(
    "--varColors",
    type=str,
    nargs="*",
    default=None,
    help="Color(s) of variation hist for plotting",
)
parser.add_argument(
    "--varOneSided",
    type=int,
    nargs="*",
    default=[],
    help="Only plot one sided variation (1) or two default two-sided (0)",
)
parser.add_argument(
    "--scaleVariation",
    nargs="*",
    type=float,
    default=[],
    help="Scale a variation by this factor",
)
parser.add_argument(
    "--subplotSizes",
    nargs=2,
    type=int,
    default=[4, 2],
    help="Relative sizes for upper and lower panels",
)

args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)

varNames = args.varNames
if varNames is not None:
    varLabels = args.varLabels
    varColors = args.varColors
    if varLabels is None:
        # try to get labels from predefined styles
        varLabels = [styles.legend_labels_combine.get(e, e) for e in varNames]
    elif len(varLabels) != len(varNames):
        raise ValueError(
            "Must specify the same number of args for --varNames, and --varLabels"
            f" found varNames={len(varNames)} and varLabels={len(varLabels)}"
        )
    if varColors is None:
        varColors = [
            colormaps["tab10" if len(varNames) < 10 else "tab20"](i)
            for i in range(len(varNames))
        ]

    varOneSided = [
        args.varOneSided[i] if i < len(args.varOneSided) else 0
        for i in range(len(varNames))
    ]
    scaleVariation = [
        args.scaleVariation[i] if i < len(args.scaleVariation) else 1
        for i in range(len(varNames))
    ]

fittype = "prefit" if args.prefit else "postfit"
ratio = not args.noLowerPanel and not args.logTransform
diff = not args.noLowerPanel and args.logTransform
data = not args.noData

# load .hdf5 file first, must exist in combinetf and combinetf2
fitresult_h5py = combinetf_input.get_fitresult(args.infile.replace(".root", ".hdf5"))

if args.uncFile is not None:
    fitresult_h5py_unc = combinetf_input.get_fitresult(
        args.uncFile.replace(".root", ".hdf5")
    )
    fitresult_unc = ioutils.pickle_load_h5py(fitresult_h5py_unc["results"])

if "results" in fitresult_h5py.keys():
    fitresult = ioutils.pickle_load_h5py(fitresult_h5py["results"])
    combinetf2 = True
elif os.path.isfile(args.infile.replace(".hdf5", ".root")):
    fitresult = combinetf_input.get_fitresult(args.infile.replace(".hdf5", ".root"))
    combinetf2 = False
else:
    raise IOError(
        "Unknown source, input file must be either from combinetf2 or combinetf1 (in case of combinetf1 both .root and .hdf5 files must exist)"
    )

meta = input_tools.get_metadata(args.infile)
is_normalized = meta["args"].get("normalize", False) if meta is not None else False

translate_selection = {
    "charge": r"$\mathit{q}^\mu$ = ",
}
translate_selection_value = {
    "charge": {
        -1.0: "-1",
        1.0: "+1",
    },
}


def make_plot(
    h_data,
    h_inclusive,
    h_stack,
    axes,
    colors=None,
    labels=None,
    hup=None,
    hdown=None,
    variation="",
    suffix="",
    chi2=None,
    meta=None,
    saturated_chi2=False,
    lumi=None,
    selection=None,
):
    axes_names = [a.name for a in axes]

    if any(x in axes_names for x in ["ptll", "mll", "ptVgen", "ptVGen", "pt"]):
        # in case of variable bin width normalize to unit
        binwnorm = 1.0
        ylabel = r"$Events\,/\,GeV$"
    else:
        binwnorm = None
        ylabel = r"$Events\,/\,bin$"

    if args.logTransform:
        ylabel = ylabel.replace("Events", "log(Events)")

    if args.ylabel is not None:
        ylabel = args.ylabel

    histtype_data = "errorbar"
    histtype_mc = "fill"

    if any(x in axes_names for x in ["ptVgen", "absYVgen", "helicity"]):
        histtype_data = "step"
        histtype_mc = "errorbar"

    if len(h_data.axes) > 1:
        if args.invertAxes:
            logger.info("invert eta order")
            axes_names = axes_names[::-1]
            axes = axes[::-1]

        # make unrolled 1D histograms
        h_data = hh.unrolledHist(h_data, binwnorm=binwnorm, obs=axes_names)
        h_inclusive = hh.unrolledHist(h_inclusive, binwnorm=binwnorm, obs=axes_names)
        h_stack = [
            hh.unrolledHist(h, binwnorm=binwnorm, obs=axes_names) for h in h_stack
        ]

    if args.normToData:
        scale = h_data.values().sum() / h_inclusive.values().sum()
        h_stack = [hh.scaleHist(h, scale) for h in h_stack]
        h_inclusive = hh.scaleHist(h_inclusive, scale)

    if args.xlabel is not None:
        xlabel = args.xlabel
    elif len(axes_names) == 1:
        xlabel = styles.xlabels.get(axes_names[0])
    else:
        xlabel = f"({', '.join([styles.xlabels.get(s,s).replace('(GeV)','') for s in axes_names])}) bin"
    if ratio or diff:
        if args.noData:
            rlabel = ("Diff." if diff else "Ratio") + " to pred."
        else:
            rlabel = f"${args.dataName}" + ("-" if diff else r"\,/\,") + "Pred.$"

        fig, ax1, ax2 = plot_tools.figureWithRatio(
            h_data,
            xlabel,
            ylabel,
            args.ylim,
            rlabel,
            args.rrange,
            width_scale=1.25 if len(axes_names) == 1 else 1,
            subplotsizes=args.subplotSizes,
        )
    else:
        fig, ax1 = plot_tools.figure(h_data, xlabel, ylabel, args.ylim)

    hep.histplot(
        h_stack,
        xerr=False,
        yerr=False,
        histtype=histtype_mc,
        color=colors,
        label=labels,
        stack=True,
        density=False,
        binwnorm=binwnorm,
        ax=ax1,
        zorder=1,
        flow="none",
    )

    if data:
        hep.histplot(
            h_data,
            yerr=True,
            histtype=histtype_data,
            color="black",
            label=args.dataName,
            binwnorm=binwnorm,
            ax=ax1,
            alpha=1.0,
            zorder=2,
            flow="none",
        )

    if len(axes_names) > 1 and args.binSeparationLines is not None:
        # plot dashed vertical lines to sepate makro bins

        s_range = lambda x, n=1: (
            int(x) if round(x, n) == float(int(round(x, n))) else round(x, n)
        )
        s_label = styles.xlabels.get(axes_names[0], axes_names[0])
        if "(GeV)" in s_label:
            s_label = s_label.replace("(GeV)", "")
            s_unit = r"GeV"
        else:
            s_unit = ""

        max_y = np.max(h_inclusive.values()[...])
        min_y = ax1.get_ylim()[0]

        range_y = max_y - min_y

        for i in range(1, axes[0].size + 1):
            if len(args.binSeparationLines) > 0 and not any(
                np.isclose(x, axes[0].edges[i]) for x in args.binSeparationLines
            ):
                continue

            x = axes[-1].size * i
            x_lo = axes[-1].size * (i - 1)

            if i < axes[0].size + 1:
                # don't plot last line since it's the axis line already
                ax1.plot([x, x], [min_y, max_y], linestyle="--", color="black")

            if len(args.binSeparationLines) == 0 or any(
                np.isclose(x, axes[0].edges[i - 1]) for x in args.binSeparationLines
            ):
                y = min_y + range_y * (
                    0.15 if np.min(h_inclusive.values()[x_lo:x]) > max_y * 0.3 else 0.8
                )
                lo = s_range(axes[0].edges[i - 1])
                hi = s_range(axes[0].edges[i])
                plot_tools.wrap_text(
                    [s_label, f"${lo}-{hi}$", s_unit],
                    ax1,
                    x_lo,
                    y,
                    x,
                    text_size="small",
                    transform=ax1.transData,
                )

    if ratio or diff:
        extra_handles = []
        extra_labels = []
        if diff:
            h1 = hh.addHists(h_inclusive, h_inclusive, scale2=-1)
            h2 = hh.addHists(h_data, h_inclusive, scale2=-1)
        else:
            h1 = hh.divideHists(
                h_inclusive,
                h_inclusive,
                cutoff=1e-8,
                rel_unc=True,
                flow=False,
                by_ax_name=False,
            )
            h2 = hh.divideHists(h_data, h_inclusive, cutoff=0.01, rel_unc=True)

        hep.histplot(
            h1,
            histtype="step",
            color="grey",
            alpha=0.5,
            yerr=False,
            ax=ax2,
            linewidth=2,
            flow="none",
        )

        if data:
            hep.histplot(
                h2,
                histtype="errorbar",
                color="black",
                # label=args.dataName,
                yerr=True if not args.logTransform else h2.variances() ** 0.5,
                linewidth=2,
                ax=ax2,
                flow="none",
            )

        # for uncertaity bands
        edges = h_inclusive.axes[0].edges

        # need to divide by bin width
        binwidth = edges[1:] - edges[:-1] if binwnorm else 1.0
        if h_inclusive.storage_type != hist.storage.Weight:
            raise ValueError(
                f"Did not find uncertainties in {fittype} hist. Make sure you run combinetf with --computeHistErrors!"
            )

        if not args.noUncertainty:
            nom = h_inclusive.values() / binwidth
            std = np.sqrt(h_inclusive.variances()) / binwidth

            hatchstyle = None
            facecolor = "silver"
            label_unc = "Pred. unc."

            if diff:
                ax2.fill_between(
                    edges,
                    np.append((nom + std) - nom, ((nom + std) - nom)[-1]),
                    np.append((nom - std) - nom, ((nom - std) - nom)[-1]),
                    step="post",
                    facecolor=facecolor,
                    zorder=0,
                    hatch=hatchstyle,
                    edgecolor="k",
                    linewidth=0.0,
                    label=label_unc,
                )
            else:
                ax2.fill_between(
                    edges,
                    np.append((nom + std) / nom, ((nom + std) / nom)[-1]),
                    np.append((nom - std) / nom, ((nom - std) / nom)[-1]),
                    step="post",
                    facecolor=facecolor,
                    zorder=0,
                    hatch=hatchstyle,
                    edgecolor="k",
                    linewidth=0.0,
                    label=label_unc,
                )

        if hup is not None:
            linewidth = 2
            for i, (hu, hd) in enumerate(zip(hup, hdown)):

                if scaleVariation[i] != 1:
                    hdiff = hh.addHists(hu, h_inclusive, scale2=-1)
                    hdiff = hh.scaleHist(hdiff, scaleVariation[i])
                    hu = hh.addHists(hdiff, h_inclusive)

                    if not varOneSided[i]:
                        hdiff = hh.addHists(hd, h_inclusive, scale2=-1)
                        hdiff = hh.scaleHist(hdiff, scaleVariation[i])
                        hd = hh.addHists(hdiff, h_inclusive)

                if varOneSided[i]:
                    hvars = hh.divideHists(hu, h_inclusive, cutoff=0.01, rel_unc=True)
                    linestyle = "-"
                else:
                    hvars = [
                        hh.divideHists(hu, h_inclusive, cutoff=0.01, rel_unc=True),
                        hh.divideHists(hd, h_inclusive, cutoff=0.01, rel_unc=True),
                    ]
                    linestyle = ["-", "--"]

                hep.histplot(
                    hvars,
                    histtype="step",
                    color=varColors[i],
                    linestyle=linestyle,
                    yerr=False,
                    linewidth=linewidth,
                    label=varLabels[i] if varOneSided[i] else None,
                    ax=ax2,
                    flow="none",
                )
                if not varOneSided[i]:
                    extra_handles.append(
                        Line2D([0], [0], color=varColors[i], linewidth=linewidth)
                    )
                    extra_labels.append(varLabels[i])

    scale = max(1, np.divide(*ax1.get_figure().get_size_inches()) * 0.3)

    text_pieces = []
    if is_normalized:
        text_pieces.append(fittype.capitalize() + " (normalized)")
    else:
        text_pieces.append(fittype.capitalize())

    if selection is not None:
        text_pieces.extend(selection)

    if chi2 is not None:
        p_val = int(round(scipy.stats.chi2.sf(chi2[0], chi2[1]) * 100))
        if saturated_chi2:
            chi2_name = r"$\mathit{\chi}_{\mathrm{sat.}}^2/\mathit{ndf}$"
        else:
            chi2_name = r"$\mathit{\chi}^2/\mathit{ndf}$"

        if len(h_data.values()) < 100:
            text_pieces.append(chi2_name)
            text_pieces.append(
                rf"$= {round(chi2[0],1)}/{chi2[1]}\ (\mathit{{p}}={p_val}\%)$"
            )
        else:
            text_pieces.append(
                chi2_name
                + rf" = ${round(chi2[0],1)}/{chi2[1]}\ (\mathit{{p}}={p_val}\%)$"
            )

    plot_tools.add_cms_decor(
        ax1,
        args.cmsDecor,
        data=data or "Nonprompt" in labels,
        lumi=lumi if args.dataName == "Data" and not args.noData else None,
        loc=args.logoPos,
    )

    if len(h_stack) < 10:
        plot_tools.addLegend(
            ax1,
            ncols=args.legCols,
            loc=args.legPos,
            text_size=args.legSize,
            extra_text=text_pieces,
            extra_text_loc=args.extraTextLoc,
        )

    if ratio or diff:
        plot_tools.addLegend(
            ax2,
            ncols=args.lowerLegCols,
            loc=args.lowerLegPos,
            text_size=args.legSize,
            extra_handles=extra_handles,
            extra_labels=extra_labels,
            custom_handlers=["stacked"],
        )

    plot_tools.fix_axes(ax1, ax2, fig, yscale=args.yscale, noSci=args.noSciy)

    to_join = [fittype, args.postfix, *axes_names, suffix]
    outfile = "_".join(filter(lambda x: x, to_join))
    if is_normalized:
        outfile += "_normalized"
    if args.cmsDecor == "Preliminary":
        outfile += "_preliminary"

    plot_tools.save_pdf_and_png(outdir, outfile)

    stack_yields = None
    unstacked_yields = None
    kwargs = dict()
    if meta is not None:
        if "meta_info_input" in meta:
            analysis_meta_info = {
                "Combinetf2Output": meta["meta_info"],
                "AnalysisOutput": meta["meta_info_input"]["meta_info"],
            }
        else:
            analysis_meta_info = {"AnalysisOutput": meta["meta_info"]}
        kwargs["analysis_meta_info"] = analysis_meta_info

    plot_tools.write_index_and_log(
        outdir,
        outfile,
        yield_tables={
            "Stacked processes": pd.DataFrame(
                [
                    (k, sum(h.values()), sum(h.variances()) ** 0.5)
                    for k, h in zip(labels, h_stack)
                ],
                columns=["Process", "Yield", "Uncertainty"],
            ),
            "Unstacked processes": pd.DataFrame(
                [
                    (k, sum(h.values()), sum(h.variances()) ** 0.5)
                    for k, h in zip([args.dataName, "Inclusive"], [h_data, h_inclusive])
                ],
                columns=["Process", "Yield", "Uncertainty"],
            ),
        },
        args=args,
        **kwargs,
    )


def make_plots(
    hist_data,
    hist_inclusive,
    hist_stack,
    axes,
    procs,
    labels,
    colors,
    hist_var=None,
    channel="",
    *opts,
    **kwopts,
):
    if args.processGrouping is not None:
        hist_stack, labels, colors, procs = styles.process_grouping(
            args.processGrouping, hist_stack, procs
        )

    # temporary fix to take uncertainties from second file
    if args.uncFile:
        hist_unc = fitresult_unc[f"hist_{fittype}_inclusive"][channel].get()
        axes = hist_unc.axes

        hist_data = hist_data.project(*axes.name)
        hist_inclusive = hist_inclusive.project(*axes.name)
        hist_stack = [h.project(*axes.name) for h in hist_stack]

        hist_inclusive.variances(flow=True)[...] = hist_unc.variances(flow=True)

    if hist_var is not None:
        hists_down = [
            hist_var[{"downUpVar": 0, "vars": n}].project(*[a.name for a in axes])
            for n in varNames
        ]
        hists_up = [
            hist_var[{"downUpVar": 1, "vars": n}].project(*[a.name for a in axes])
            for n in varNames
        ]
    else:
        hists_down = None
        hists_up = None

    # make plots in slices (e.g. for charge plus an minus separately)
    selection_axes = [a for a in axes if a.name in args.selectionAxes]
    if len(selection_axes) > 0:
        selection_bins = [
            np.arange(a.size) for a in axes if a.name in args.selectionAxes
        ]
        other_axes = [a for a in axes if a not in selection_axes]

        for bins in itertools.product(*selection_bins):
            idxs = {a.name: i for a, i in zip(selection_axes, bins)}
            idxs_centers = {
                a.name: (
                    a.centers[i]
                    if isinstance(a, (hist.axis.Regular, hist.axis.Variable))
                    else a.edges[i]
                )
                for a, i in zip(selection_axes, bins)
            }

            h_data = hist_data[idxs]
            h_inclusive = hist_inclusive[idxs]
            h_stack = [h[idxs] for h in hist_stack]

            if hist_var is not None:
                hdown = [h[idxs] for h in hists_down]
                hup = [h[idxs] for h in hists_up]
            else:
                hdown = None
                hup = None

            if "run" in [a.name for a in selection_axes]:
                idx = idxs["run"]
                lumis = common.run_edges_lumi
                lumi = np.diff(lumis)[idx]
                logger.info(
                    f"Axis 'run' found in histogram selection_axes, set lumi to {lumi}"
                )
                kwopts["run"] = lumi
            for a, i in idxs_centers.items():
                print(a, i)
            selection = [
                f"{translate_selection[a]}{translate_selection_value[a][i]}"
                for a, i in idxs_centers.items()
            ]
            suffix = f"{channel}_" + "_".join(
                [
                    f"{a}_{str(i).replace('.','p').replace('-','m')}"
                    for a, i in idxs_centers.items()
                ]
            )
            logger.info(
                f"Make plot for axes {[a.name for a in other_axes]}, in bins {idxs}"
            )
            make_plot(
                h_data,
                h_inclusive,
                h_stack,
                other_axes,
                labels=labels,
                colors=colors,
                suffix=suffix,
                hup=hup,
                hdown=hdown,
                selection=selection,
                *opts,
                **kwopts,
            )
    else:
        make_plot(
            hist_data,
            hist_inclusive,
            hist_stack,
            axes,
            labels=labels,
            colors=colors,
            suffix=channel,
            hup=hists_up,
            hdown=hists_down,
            *opts,
            **kwopts,
        )


if combinetf2:
    meta = ioutils.pickle_load_h5py(fitresult_h5py["meta"])
    command = meta["meta_info"]["command"]
    asimov = False
    if "-t-1" in command or "-t -1" in command or "-t" not in command:
        asimov = True
    meta_input = meta["meta_info_input"]
    procs = meta["procs"].astype(str)
    if args.filterProcs is not None:
        procs = [p for p in procs if p in args.filterProcs]
    labels, colors, procs = styles.get_labels_colors_procs_sorted(procs)

    chi2 = None
    if f"chi2_{fittype}" in fitresult and not args.noChisq:
        chi2 = fitresult[f"chi2_{fittype}"], fitresult[f"ndf_{fittype}"]

    for channel, info in meta_input["channel_info"].items():
        if channel.endswith("masked"):
            continue
        hist_data = fitresult["hist_data_obs"][channel].get()
        hist_inclusive = fitresult[f"hist_{fittype}_inclusive"][channel].get()
        hist_stack = fitresult[f"hist_{fittype}"][channel].get()
        hist_stack = [hist_stack[{"processes": p}] for p in procs]

        # vary poi by postfit uncertainty
        if varNames is not None:
            hist_var = fitresult[f"hist_{fittype}_inclusive_variations"][channel].get()
        else:
            hist_var = None

        if args.logTransform:
            hist_data.variances(flow=True)[...] = (
                hist_data.variances(flow=True)[...]
                / hist_data.values(flow=True)[...] ** 2
            )
            for h in hist_stack:
                h.variances(flow=True)[...] = (
                    h.variances(flow=True)[...] / h.values(flow=True)[...] ** 2
                )

            hist_data.values(flow=True)[...] = np.log(hist_data.values(flow=True)[...])
            for h in hist_stack:
                h.values(flow=True)[...] = np.log(h.values(flow=True)[...])

        if any(x in hist_data.axes.name for x in ["helicity"]):
            if asimov:
                hist_data.values()[...] = 1e5 * np.log(hist_data.values())
            or_vals = np.copy(hist_inclusive.values())
            hist_inclusive.values()[...] = 1e5 * np.log(hist_inclusive.values())
            hist_inclusive.variances()[...] = (
                1e10 * (hist_inclusive.variances()) / np.square(or_vals)
            )

            if varNames is not None:
                hist_var.values()[...] = 1e5 * np.log(hist_var.values())
                hist_var.variances()[...] = (
                    1e10 * (hist_var.variances()) / np.square(or_vals)
                )

            for h in hist_stack:
                or_vals = np.copy(h.values())
                h.values()[...] = 1e5 * np.log(h.values())
                h.variances()[...] = 1e10 * (h.variances()) / np.square(or_vals)

        make_plots(
            hist_data,
            hist_inclusive,
            hist_stack,
            info["axes"],
            hist_var=hist_var,
            channel=channel,
            procs=procs,
            labels=labels,
            colors=colors,
            chi2=chi2,
            meta=meta,
            lumi=info["lumi"],
        )
else:
    # combinetf1
    import ROOT

    procs = [
        k.replace("expproc_", "").replace(f"_{fittype};1", "")
        for k in fitresult.keys()
        if fittype in k and k.startswith("expproc_") and "hybrid" not in k
    ]
    if args.filterProcs is not None:
        procs = [p for p in procs if p in args.filterProcs]
    labels, colors, procs = styles.get_labels_colors_procs_sorted(procs)

    if "meta" in fitresult_h5py:
        # the fit was probably done on a file generated via the hdf5 writer and we can use the axes information
        meta = ioutils.pickle_load_h5py(fitresult_h5py["meta"])
        ch_start = 0
        for channel, info in meta["channel_info"].items():
            if channel.endswith("masked"):
                continue
            shape = [len(a) for a in info["axes"]]

            ch_end = ch_start + np.prod(
                shape
            )  # in combinetf1 the channels are concatenated and we need to index one after the other

            hist_data = fitresult["obs;1"].to_hist()
            values = np.reshape(hist_data.values()[ch_start:ch_end], shape)
            hist_data = hist.Hist(
                *info["axes"],
                storage=hist.storage.Weight(),
                data=np.stack((values, values), axis=-1),
            )

            # last bin can be masked channel; slice with [:nBins]
            hist_inclusive = fitresult[f"expfull_{fittype};1"].to_hist()
            hist_inclusive = hist.Hist(
                *info["axes"],
                storage=hist.storage.Weight(),
                data=np.stack(
                    (
                        np.reshape(hist_inclusive.values()[ch_start:ch_end], shape),
                        np.reshape(hist_inclusive.variances()[ch_start:ch_end], shape),
                    ),
                    axis=-1,
                ),
            )
            hist_stack = [
                fitresult[f"expproc_{p}_{fittype};1"].to_hist() for p in procs
            ]
            hist_stack = [
                hist.Hist(
                    *info["axes"],
                    storage=hist.storage.Weight(),
                    data=np.stack(
                        (
                            np.reshape(h.values()[ch_start:ch_end], shape),
                            np.reshape(h.variances()[ch_start:ch_end], shape),
                        ),
                        axis=-1,
                    ),
                )
                for h in hist_stack
            ]

            if not args.prefit and not args.noChisq:
                rfile = ROOT.TFile.Open(args.infile.replace(".hdf5", ".root"))
                ttree = rfile.Get("fitresults")
                ttree.GetEntry(0)
                chi2 = [
                    2 * (ttree.nllvalfull - ttree.satnllvalfull),
                    np.prod([len(a) for a in info["axes"]]) - ttree.ndofpartial,
                ]
            else:
                chi2 = None

            make_plots(
                hist_data,
                hist_inclusive,
                hist_stack,
                info["axes"],
                channel=channel,
                procs=procs,
                labels=labels,
                colors=colors,
                chi2=chi2,
                meta=meta,
                saturated_chi2=True,
                lumi=info["lumi"],
            )
            ch_start = ch_end
    else:
        # the fit was probably done on a file generated via the root writer and we can't use the axes information

        # get axes from the directory name
        filename_parts = [x for x in filter(lambda x: x, args.infile.split("/"))]
        analysis = filename_parts[-2].split("_")[0]
        if analysis == "ZMassDilepton":
            all_axes = {
                # "mll": hist.axis.Regular(60, 60., 120., name = "mll", overflow=False, underflow=False),
                "mll": hist.axis.Variable(
                    [
                        60,
                        70,
                        75,
                        78,
                        80,
                        82,
                        84,
                        85,
                        86,
                        87,
                        88,
                        89,
                        90,
                        91,
                        92,
                        93,
                        94,
                        95,
                        96,
                        97,
                        98,
                        100,
                        102,
                        105,
                        110,
                        120,
                    ],
                    name="mll",
                    overflow=False,
                    underflow=False,
                ),
                "etaAbsEta": hist.axis.Variable(
                    [
                        -2.4,
                        -2.0,
                        -1.6,
                        -1.4,
                        -1.2,
                        -1.0,
                        -0.6,
                        0.0,
                        0.6,
                        1.0,
                        1.2,
                        1.4,
                        1.6,
                        2.0,
                        2.4,
                    ],
                    name="etaAbsEta",
                    overflow=False,
                    underflow=False,
                ),
                "cosThetaStarll": hist.axis.Regular(
                    2, -1.0, 1.0, name="cosThetaStarll", underflow=False, overflow=False
                ),
                "yll": hist.axis.Regular(
                    20, -2.5, 2.5, name="yll", overflow=False, underflow=False
                ),
                "ptll": hist.axis.Variable(
                    common.get_dilepton_ptV_binning(False),
                    name="ptll",
                    underflow=False,
                    overflow=False,
                ),
            }
        elif analysis == "ZMassWLike":
            all_axes = {
                "pt": hist.axis.Regular(
                    34, 26, 60, name="pt", overflow=False, underflow=False
                ),
                "eta": hist.axis.Regular(
                    48, -2.4, 2.4, name="eta", overflow=False, underflow=False
                ),
                "charge": common.axis_charge,
                "ptGen": hist.axis.Regular(
                    33, 27, 60, name="ptGen", overflow=False, underflow=False
                ),
                "absEtaGen": hist.axis.Variable(
                    differential.eta_binning,
                    name="absEtaGen",
                    overflow=False,
                    underflow=False,
                ),
                "qGen": common.axis_charge,
            }
        elif analysis == "WMass":
            all_axes = {
                "pt": hist.axis.Regular(
                    30, 26, 56, name="pt", overflow=False, underflow=False
                ),
                # "pt": hist.axis.Regular(31, 26, 57, name = "pt", overflow=False, underflow=False),
                # "pt": hist.axis.Regular(29, 27, 56, name = "ptGen", overflow=False, underflow=False),
                "eta": hist.axis.Regular(
                    48, -2.4, 2.4, name="eta", overflow=False, underflow=False
                ),
                "charge": common.axis_charge,
                "passIso": common.axis_passIso,
                "passMT": common.axis_passMT,
                "ptGen": hist.axis.Regular(
                    29, 27, 56, name="ptGen", overflow=False, underflow=False
                ),
                "absEtaGen": hist.axis.Variable(
                    differential.eta_binning,
                    name="absEtaGen",
                    overflow=False,
                    underflow=False,
                ),
                "qGen": common.axis_charge,
            }
        else:
            raise ValueError(f"Unknown analysis {analysis}, can't set the axes")

        axes = [
            all_axes[part]
            for part in filename_parts[-2].split("_")
            if part in all_axes.keys()
        ]
        if args.axlim:
            nv = len(args.axlim)
            if nv % 2:
                raise ValueError(
                    "if --axlim is specified it must have two values per axis!"
                )
            axlim = np.array(args.axlim).reshape((int(nv / 2), 2))
            axes = [
                (
                    ax
                    if lim is not None
                    else hist.axis.Variable(
                        ax.edges[(ax.edges >= lim[0]) & (ax.edges <= lim[1])]
                    )
                )
                for ax, lim in itertools.zip_longest(axes, axlim)
            ]
        shape = [len(a) for a in axes]

        hist_data = fitresult["obs;1"].to_hist()
        nBins = hist_data.shape[0]
        values = np.reshape(hist_data.values(), shape)
        hist_data = hist.Hist(
            *axes,
            storage=hist.storage.Weight(),
            data=np.stack((values, values), axis=-1),
        )

        # last bin can be masked channel; slice with [:nBins]
        hist_inclusive = fitresult[f"expfull_{fittype};1"].to_hist()[:nBins]
        hist_inclusive = hist.Hist(
            *axes,
            storage=hist.storage.Weight(),
            data=np.stack(
                (
                    np.reshape(hist_inclusive.values(), shape),
                    np.reshape(hist_inclusive.variances(), shape),
                ),
                axis=-1,
            ),
        )
        hist_stack = [
            fitresult[f"expproc_{p}_{fittype};1"].to_hist()[:nBins] for p in procs
        ]
        hist_stack = [
            hist.Hist(
                *axes,
                storage=hist.storage.Weight(),
                data=np.stack(
                    (np.reshape(h.values(), shape), np.reshape(h.variances(), shape)),
                    axis=-1,
                ),
            )
            for h in hist_stack
        ]

        if not args.prefit:
            rfile = ROOT.TFile.Open(args.infile.replace(".hdf5", ".root"))
            ttree = rfile.Get("fitresults")
            ttree.GetEntry(0)
            chi2 = [
                2 * (ttree.nllvalfull - ttree.satnllvalfull),
                np.prod([len(a) for a in axes]) - ttree.ndofpartial,
            ]
        else:
            chi2 = None

        make_plots(
            hist_data,
            hist_inclusive,
            hist_stack,
            axes,
            procs=procs,
            labels=labels,
            colors=colors,
            chi2=chi2,
            saturated_chi2=True,
        )

if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
    output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)
