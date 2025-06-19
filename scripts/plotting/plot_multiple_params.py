# this script is created from plot_decorr_params.py, and is supposed to plot the results of multiple fits on the same canvas to compare them. In principle this could be merged with plot_decorr_params.py by generalizing the latter.

import re

import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

import rabbit.io_tools
from scripts.plotting import plot_decorr_params as pdp
from utilities import parsing
from wums import logging, output_tools, plot_tools

if __name__ == "__main__":
    parser = parsing.plot_parser()
    parser.add_argument(
        "infile",
        nargs="+",
        type=str,
        help="Fitresult files from combinetf, to be compared",
    )
    parser.add_argument(
        "--infileReference",
        type=str,
        default=None,
        help="Fitresult file from combinetf with reference fit (might be the nominal as a reference)",
    )
    parser.add_argument(
        "--infileNominal",
        type=str,
        default=None,
        help="Fitresult file from combinetf with nominal fit (not necessary, it is used to normalize others to it, might be the same as reference)",
    )
    parser.add_argument(
        "--legendEntries",
        nargs="+",
        type=str,
        help="Legend entries, must be as many items as infile",
    )
    parser.add_argument(
        "--legendEntryReference",
        type=str,
        default="Reference",
        help="Legend entry for the reference file",
    )
    parser.add_argument(
        "--data",
        action="store_true",
        help="Specify if the fit is performed on data, needed for correct p-value calculation",
    )
    parser.add_argument(
        "--absoluteParam",
        action="store_true",
        help="Show plot as a function of absolute value of parameter (default is difference to SM prediction)",
    )
    parser.add_argument(
        "--showMCInput", action="store_true", help="Show MC input value in the plot"
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Add a title to the plot on the upper right",
    )
    parser.add_argument(
        "--widthScale",
        type=float,
        default=1.5,
        help="Scale the width of the figure with this factor",
    )
    parser.add_argument(
        "--partialImpact",
        nargs=2,
        type=str,
        default=["muonCalibration", "Calib. unc."],
        help="Uncertainty group to plot as partial error bar (in addition to data stat, which is always there)",
    )
    parser.add_argument(
        "--globalImpacts",
        action="store_true",
        help="Use the global impacts to plot uncertainties (they must be present in the input file)",
    )
    parser.add_argument(
        "--showWeightedAverage",
        action="store_true",
        help="Print weighted average with its uncertainty from the fit results",
    )

    parser = parsing.set_parser_default(parser, "legCols", 1)

    args = parser.parse_args()
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    if len(args.legendEntries) != len(args.infile):
        raise IOError(
            "Option --legendEntries must be given as many items as the number of input files (excluding the reference or nominal if given)"
        )

    outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)

    partialImpact, partialImpactLegend = args.partialImpact
    partial_impacts_to_read = ["stat", partialImpact]

    if args.infileReference:
        dfReference = pdp.get_values_and_impacts_as_panda(
            args.infileReference,
            partial_impacts_to_read=partial_impacts_to_read,
            global_impacts=args.globalImpacts,
        )
        fReference, mReference = rabbit.io_tools.get_fitresult(
            args.infileReference, meta=True
        )
        lumi = sum(
            [c["lumi"] for c in mReference["meta_info_input"]["channel_info"].values()]
        )
        # meta_info = meta["meta_info"]
    else:
        lumi = None
        meta_info = None

    if args.infileNominal:
        fNominal = rabbit.io_tools.get_fitresult(args.infileNominal)
        dfNominal = pdp.get_values_and_impacts_as_panda(
            args.infileNominal,
            partial_impacts_to_read=partial_impacts_to_read,
            global_impacts=args.globalImpacts,
        )

    dfs = []
    legends = []
    for i, fin in enumerate(args.infile):
        dfs.append(
            pdp.get_values_and_impacts_as_panda(
                fin,
                partial_impacts_to_read=partial_impacts_to_read,
                global_impacts=args.globalImpacts,
            )
        )
        legends.append(args.legendEntries[i])

    df = pd.concat(dfs, ignore_index=True, sort=False)

    df["Params"] = df["Name"].apply(lambda x: x.split("_")[0])
    df["Parts"] = df["Name"].apply(lambda x: x.split("_")[1:-1])

    for param, df_p in df.groupby("Params"):
        logger.info(f"Make plot for {param}")

        if param is not None and "MeV" in param:
            xlabel = param.split("MeV")[0]
            if xlabel.startswith("massShift"):
                proc = xlabel.replace("massShift", "")[0]
                xlabel = r"$\mathit{m}_\mathrm{" + str(proc) + "}$ (MeV)"
                offset = 80354 if proc == "W" else 91187.6

            if xlabel.startswith("Width"):
                proc = xlabel.replace("Width", "")[0]
                xlabel = r"$\mathit{\Gamma}_\mathrm{" + str(proc) + "}$ (MeV)"
                offset = 2091.13 if proc == "W" else 2494.13

            scale = float(
                re.search(
                    r"\d+(\.\d+)?", param.split("MeV")[0].replace("p", ".")
                ).group()
            )
            if "Diff" in param:
                scale *= 2  # take diffs by 2 as up and down pull in opposite directions
        else:
            scale = 1
            offset = 0
            xlabel = param

        if not args.absoluteParam or "Diff" in param:
            xlabel = r"$\Delta " + xlabel[1:]
            offset = 0

        logger.info(f"offset = {offset}")

        # FIXME: these next lines might not be needed anymore
        # df_p["Names"] = df_p["Name"].apply(
        #     lambda x: "".join(
        #         [x.split("MeV")[-1].split("_")[0] for x in x.split("_decorr")]
        #     )
        # )

        axis_ranges = {i: v for i, v in enumerate(args.legendEntries)}
        df_p["yticks"] = pd.Series(args.legendEntries, index=df_p.index)
        ylabel = None

        # df_p.sort_values(by=args.legendEntries, ascending=True, inplace=True)

        xCenter = 0

        val = df_p["value"].values * scale + offset
        err = df_p["err_Total"].values * scale
        err_stat = df_p["err_stat"].values * scale
        err_cal = df_p[f"err_{partialImpact}"].values * scale

        if args.infileNominal:
            if len(dfNominal) > 1:
                logger.warning(
                    f"Found {len(dfNominal)} values from the reference fit but was expecting 1, take first value"
                )
            elif len(dfNominal) == 0:
                raise RuntimeError(
                    f"Found 0 values from the reference fit but was expecting 1"
                )

            central_no_offset = dfNominal["value"].values[0] * scale
            central = central_no_offset + offset
            logger.info(f"Nominal (no offset) = {central_no_offset}")
            logger.info(f"Nominal (w/ offset) = {central}")
        else:
            central = 0

        if args.infileReference:
            if len(dfReference) > 1:
                logger.warning(
                    f"Found {len(dfReference)} values from the reference fit but was expecting 1, take first value"
                )
            elif len(dfReference) == 0:
                raise RuntimeError(
                    f"Found 0 values from the reference fit but was expecting 1"
                )

            c_err_stat = dfReference["err_stat"].values[0] * scale
            c_err_cal = dfReference[f"err_{partialImpact}"].values[0] * scale
            c_err = dfReference["err_Total"].values[0] * scale
            c = dfReference["value"].values[0] * scale + offset

            logger.info(f"Reference (before subtracting central) = {c}")
            if args.infileNominal:
                c -= central
            else:
                if not args.showMCInput:
                    central = c
                    c = 0
            logger.info(f"Reference (after subtracting central) = {c}")

        val -= central

        if args.showWeightedAverage:
            all_values = df_p["value"].values * scale - central
            all_total_uncs = df_p["err_Total"].values * scale
            # logger.warning(f"All values   : {all_values}")
            # logger.warning(f"Uncertainties: {all_total_uncs}")
            weights = 1.0 / (np.power(all_total_uncs, 2.0))
            average = np.average(all_values, weights=weights)
            # average_unc = np.sqrt(np.cov(all_values, aweights=weights))
            ## alternative way to get the variance of the mean
            # variance = np.average((all_values-average)**2, weights=weights)
            # variance = variance*len(weights)/(len(weights)-1)
            ## another test
            average_std = np.sqrt(1.0 / np.sum(weights))
            #
            # logger.warning(f"Average ({len(args.infile)} values) = {round(average,1)} +/- {round(average_unc,1)} MeV")
            # logger.warning(f"Average ({len(args.infile)} values) = {round(average,1)} +/- {round(np.sqrt(variance),1)} MeV")
            logger.warning(
                f"Average ({len(args.infile)} values) = {round(average,1)} +/- {round(average_std,1)} MeV"
            )

        yticks = df_p["yticks"].values

        if args.xlim is None:
            xlim = min(val - err), max(val + err)
            xwidth = xlim[1] - xlim[0]
            xlim = -0.05 * xwidth + xlim[0], 0.05 * xwidth + xlim[1]
        else:
            xlim = args.xlim

        ylim = (0.0, len(df_p))
        y = np.arange(0, len(df)) + 0.5

        fig, ax1 = plot_tools.figure(
            None,
            xlabel=xlabel,
            ylabel=ylabel,  # ", ".join(ylabels),
            grid=True,
            automatic_scale=False,
            width_scale=args.widthScale,
            height=4 + 0.24 * len(df_p),
            xlim=xlim,
            ylim=ylim,
        )

        if args.infileReference:

            if args.legPos in [None, "center left", "upper left"]:
                x_chi2 = 0.06
                y_chi2 = 0.12
                ha = "left"
                va = "bottom"
            else:
                raise NotImplementedError(
                    "Can only plot chi2 if legend is center or upper"
                )

            if args.showWeightedAverage:
                plot_tools.wrap_text(
                    [
                        # rf"Average: ${round(average,1)} \pm {round(average_unc,1)}$",
                        rf"Average: ${round(average,1)}$ MeV",
                    ],
                    ax1,
                    x_chi2,
                    y_chi2,
                    text_size=args.legSize,
                )

            ax1.fill_between(
                [c - c_err, c + c_err], ylim[0], ylim[1], color="gray", alpha=0.3
            )
            ax1.fill_between(
                [c - c_err_stat, c + c_err_stat],
                ylim[0],
                ylim[1],
                color="gray",
                alpha=0.3,
            )
            ax1.fill_between(
                [c - c_err_cal, c + c_err_cal],
                ylim[0],
                ylim[1],
                color="gray",
                alpha=0.3,
            )
            ax1.plot([c, c], ylim, color="black", linewidth=2, linestyle="-")

        ytickpositions = y

        ax1.set_yticks(ytickpositions, labels=yticks)
        ax1.minorticks_off()

        ax1.errorbar(
            val,
            y,
            xerr=err_stat,
            color="red",
            marker="",
            linestyle="",
            label="Stat. unc.",
            zorder=3,
        )
        ax1.errorbar(
            val,
            y,
            xerr=err_cal,
            color="orange",
            marker="",
            linestyle="",
            linewidth=5,
            label=partialImpactLegend,
            zorder=2,
        )
        ax1.errorbar(
            val,
            y,
            xerr=err,
            color="black",
            marker="o",
            linestyle="",
            label="Measurement",
            zorder=1,
            capsize=10,
            linewidth=3,
        )
        ax1.plot(
            val, y, color="black", marker="o", linestyle="", zorder=4
        )  # point on top
        # ax1.plot(val, y, color='black', marker="o") # plot black points on top

        extra_handles = [
            (
                Polygon(
                    [[0, 0], [0, 0], [0, 0], [0, 0]],
                    facecolor="gray",
                    linestyle="solid",
                    edgecolor="black",
                    linewidth=2,
                    alpha=0.3,
                ),
            )
        ]

        if args.showMCInput:
            ax1.plot(
                [offset, offset],
                ylim,
                linestyle="-",
                marker="none",
                color="black",
                label="MC input",
            )
            central = 0

        plot_tools.add_cms_decor(
            ax1, args.cmsDecor, data=True, lumi=lumi, loc=args.logoPos
        )
        plot_tools.addLegend(
            ax1,
            ncols=args.legCols,
            loc=args.legPos,
            text_size=args.legSize,
            extra_handles=extra_handles,
            extra_labels=[args.legendEntryReference],
            custom_handlers=["tripleband"],
        )

        if args.title:
            ax1.text(
                1.0,
                1.005,
                args.title,
                fontsize=28,
                horizontalalignment="right",
                verticalalignment="bottom",
                transform=ax1.transAxes,
            )

        outfile = f"decorr_{param}_"
        if args.postfix:
            outfile += f"_{args.postfix}"
        if args.cmsDecor == "Preliminary":
            outfile += "_preliminary"

        plot_tools.save_pdf_and_png(outdir, outfile)
        output_tools.write_index_and_log(
            outdir,
            outfile,
            # analysis_meta_info={"AnalysisOutput": meta_info},
            args=args,
        )

    if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
        output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)
