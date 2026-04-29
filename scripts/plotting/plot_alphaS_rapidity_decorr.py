import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from scipy.stats import chi2

import rabbit.io_tools
from wremnants.utilities import parsing
from wremnants.utilities.io_tools import rabbit_input
from wums import logging, output_tools, plot_tools

ALPHA_S_SCALE_IMPACTS = 2


def get_values_and_impacts_as_panda(
    input_file,
    partial_impacts_to_read=None,
    global_impacts=False,
    scale=1.0,
    result=None,
):
    fitres, meta = rabbit.io_tools.get_fitresult(input_file, meta=True, result=result)
    poi_names = rabbit.io_tools.get_poi_names(meta)
    poi_values = []
    totals = []
    uncertainties = {}
    for poi in poi_names:
        impacts, labels = rabbit.io_tools.read_impacts_poi(
            fitres,
            poi,
            grouped=True,
            impact_type="global" if global_impacts else "traditional",
        )
        impacts = scale * impacts
        totals.append([impacts[i] for i, k in enumerate(labels) if k == "Total"][0])
        if uncertainties == {}:
            uncertainties = {
                f"err_{k}": [impacts[i]]
                for i, k in enumerate(labels)
                if (partial_impacts_to_read is None or k in partial_impacts_to_read)
            }
        else:
            for i, k in enumerate(labels):
                if partial_impacts_to_read and k not in partial_impacts_to_read:
                    continue
                uncertainties[f"err_{k}"].append(impacts[i])
        poi_values.append(scale * fitres["parms"].get()[poi].value)

    df = pd.DataFrame(
        {"Name": poi_names, "value": poi_values, "err_Total": totals, **uncertainties}
    )
    return df


def get_yll_bin_edges(meta):
    ch_info = meta["meta_info_input"]["channel_info"]
    # take the first channel
    ch = list(ch_info.values())[0]
    axes = ch["axes"]
    for ax in axes:
        if ax.name == "yll":
            return list(ax.edges)
    return None


if __name__ == "__main__":
    parser = parsing.plot_parser()
    parser.add_argument("infile", help="Fitresult file from decorrelated alphaS fit")
    parser.add_argument(
        "--infileInclusive",
        type=str,
        default=None,
        help="Fitresult file from inclusive (non-decorrelated) alphaS fit",
    )
    parser.add_argument(
        "--result",
        default=None,
        type=str,
        help="fitresults key in file (e.g. 'asimov'). Leave empty for data fit result.",
    )
    parser.add_argument(
        "--data",
        action="store_true",
        help="Specify if the fit is performed on data",
    )
    parser.add_argument(
        "--decorrAxis",
        type=str,
        default="yll_decorr",
        help="Name of the decorrelation axis in POI names",
    )
    parser.add_argument(
        "--widthScale",
        type=float,
        default=1.5,
        help="Scale the width of the figure",
    )
    parser.add_argument(
        "--partialImpact",
        nargs=2,
        type=str,
        default=["pdfCT18ZNoAlphaS", "PDF unc."],
        help="Uncertainty group to plot as partial error bar",
    )
    parser.add_argument(
        "--globalImpacts",
        action="store_true",
        help="Use the global impacts to plot uncertainties",
    )
    parser.add_argument(
        "--poiScale",
        type=float,
        default=ALPHA_S_SCALE_IMPACTS,
        help="Scale POI values and uncertainties by this factor",
    )

    parser = parsing.set_parser_default(parser, "legCols", 1)

    args = parser.parse_args()
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)

    partialImpact, partialImpactLegend = args.partialImpact
    partial_impacts_to_read = ["stat", partialImpact]

    fitresult, meta = rabbit.io_tools.get_fitresult(args.infile, meta=True)
    poi_names = rabbit.io_tools.get_poi_names(meta)
    meta_info = meta["meta_info"]
    lumi = sum([c["lumi"] for c in meta["meta_info_input"]["channel_info"].values()])

    nll = fitresult["nllvalreduced"]

    yll_edges = get_yll_bin_edges(meta)
    if yll_edges is not None:
        logger.info(f"Found yll bin edges: {yll_edges}")

    if args.infileInclusive:
        dfInclusive = get_values_and_impacts_as_panda(
            args.infileInclusive,
            partial_impacts_to_read=partial_impacts_to_read,
            global_impacts=args.globalImpacts,
            scale=args.poiScale,
            result=args.result,
        )
        fInclusive = rabbit.io_tools.get_fitresult(args.infileInclusive)
        nll_inclusive = fInclusive["nllvalreduced"]

    df = get_values_and_impacts_as_panda(
        args.infile,
        partial_impacts_to_read=partial_impacts_to_read,
        global_impacts=args.globalImpacts,
        scale=args.poiScale,
        result=args.result,
    )

    # extract bin index from POI name using the decorrelation axis
    df["bin_idx"] = df["Name"].apply(
        lambda x: rabbit_input.decode_poi_bin(x, args.decorrAxis)
    )
    df.dropna(subset=["bin_idx"], inplace=True)
    df["bin_idx"] = df["bin_idx"].astype(int)
    df.sort_values(by="bin_idx", inplace=True)

    # build y-tick labels from bin edges
    if yll_edges is not None and len(yll_edges) == len(df) + 1:
        df["yticks"] = [
            f"${yll_edges[i]:.2g} < y_{{\\ell\\ell}} < {yll_edges[i+1]:.2g}$"
            for i in df["bin_idx"].values
        ]
    else:
        df["yticks"] = [f"bin {i}" for i in df["bin_idx"].values]

    scale = args.poiScale
    xlabel = r"$\Delta\alpha_S$"

    val = df["value"].values
    err = df["err_Total"].values
    err_stat = df["err_stat"].values
    err_part = df[f"err_{partialImpact}"].values

    if args.xlim is None:
        xlim = min(val - err), max(val + err)
        xwidth = xlim[1] - xlim[0]
        xlim = -0.05 * xwidth + xlim[0], 0.05 * xwidth + xlim[1]
    else:
        xlim = args.xlim

    ylim = (0.0, len(df))
    y = np.arange(0, len(df)) + 0.5

    fig, ax1 = plot_tools.figure(
        None,
        xlabel=xlabel,
        ylabel=None,
        grid=True,
        automatic_scale=False,
        width_scale=args.widthScale,
        height=4 + 0.24 * len(df),
        xlim=xlim,
        ylim=ylim,
    )

    if args.infileInclusive:
        ndf = len(df) - 1

        logger.info(f"nll_inclusive = {nll_inclusive}; nll = {nll}")

        chi2_stat = 2 * (nll_inclusive - nll)
        chi2_label = r"\mathit{\chi}^2/\mathit{ndf}"
        if args.result == "asimov":
            chi2_stat = 0
        elif not args.data:
            chi2_stat += ndf
            chi2_label = f"<{chi2_label}>"

        p_value = 1 - chi2.cdf(chi2_stat, ndf)
        logger.info(f"ndf = {ndf}; Chi2 = {chi2_stat}; p-value={p_value}")

        plot_tools.wrap_text(
            [
                f"${chi2_label} = {str(round(chi2_stat, 1))}/{ndf}$",
                rf"$\mathit{{p}} = {str(round(p_value * 100))}\,\%$",
            ],
            ax1,
            0.06,
            0.15,
            text_size=args.legSize,
        )

        c = dfInclusive["value"].values[0]
        c_err = dfInclusive["err_Total"].values[0]
        c_err_stat = dfInclusive["err_stat"].values[0]
        c_err_part = dfInclusive[f"err_{partialImpact}"].values[0]

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
            [c - c_err_part, c + c_err_part],
            ylim[0],
            ylim[1],
            color="gray",
            alpha=0.3,
        )
        ax1.plot([c, c], ylim, color="black", linewidth=2, linestyle="-")

        val -= c

    yticks = df["yticks"].values
    ax1.set_yticks(y, labels=yticks)
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
        xerr=err_part,
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
    ax1.plot(val, y, color="black", marker="o", linestyle="", zorder=4)

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

    plot_tools.add_cms_decor(
        ax1,
        args.cmsDecor,
        data=True,
        lumi=lumi,
        loc=args.logoPos,
        text_size=args.cmsDecorSize,
    )
    plot_tools.addLegend(
        ax1,
        ncols=args.legCols,
        loc=args.legPos,
        text_size=args.legSize,
        extra_handles=extra_handles,
        extra_labels=["Inclusive"],
        custom_handlers=["tripleband"],
    )

    outfile = "alphaS_rapidity_decorr"
    if args.postfix:
        outfile += f"_{args.postfix}"

    plot_tools.save_pdf_and_png(outdir, outfile)
    output_tools.write_index_and_log(
        outdir,
        outfile,
        analysis_meta_info={"AnalysisOutput": meta_info},
        args=args,
    )

    if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
        output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)
