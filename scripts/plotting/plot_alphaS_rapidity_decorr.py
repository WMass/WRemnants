import numpy as np
from matplotlib.patches import Polygon
from scipy.stats import chi2

import rabbit.io_tools
from scripts.plotting.plot_decorr_params import get_values_and_impacts_as_panda
from wremnants.utilities import parsing
from wremnants.utilities.io_tools import rabbit_input
from wums import logging, output_tools, plot_tools

ALPHA_S_SCALE_IMPACTS = 2


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
    parser = parsing.set_parser_default(parser, "logoPos", 0)
    parser = parsing.set_parser_default(parser, "legPos", (1.01, 0))

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
    try:
        nll_full = fitresult["nllvalfull"]
    except (KeyError, IndexError):
        nll_full = None

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
        try:
            nll_inclusive_full = fInclusive["nllvalfull"]
        except (KeyError, IndexError):
            nll_inclusive_full = None

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
    impact_label = "Global impacts" if args.globalImpacts else "Traditional impacts"
    xlabel = r"$\Delta\alpha_S$" + f" ({impact_label})"

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
        chi2_label = r"\chi^2/\mathrm{ndf}"
        if args.result == "asimov":
            chi2_stat = 0
        elif not args.data:
            chi2_stat += ndf
            chi2_label = rf"\langle {chi2_label} \rangle"

        p_value = 1 - chi2.cdf(chi2_stat, ndf)
        logger.info(f"ndf = {ndf}; Chi2 = {chi2_stat}; p-value={p_value}")

        ax1.text(
            1.01,
            0.8,
            "\n".join(
                [
                    rf"${chi2_label} = {chi2_stat:.1f}/{ndf}$",
                    rf"$\mathit{{p}} = {p_value * 100:.0f}\,\%$",
                ]
            ),
            ha="left",
            va="center",
            transform=ax1.transAxes,
            fontsize=args.legSize,
        )

        if nll_full is not None and nll_inclusive_full is not None:
            logger.info(
                f"nll_inclusive_full = {nll_inclusive_full}; nll_full = {nll_full}"
            )

            chi2_stat_full = 2 * (nll_inclusive_full - nll_full)
            chi2_label_full = r"\chi^2_\mathrm{full}/\mathrm{ndf}"
            if args.result == "asimov":
                chi2_stat_full = 0
            elif not args.data:
                chi2_stat_full += ndf
                chi2_label_full = rf"\langle {chi2_label_full} \rangle"

            p_value_full = 1 - chi2.cdf(chi2_stat_full, ndf)
            logger.info(
                f"ndf = {ndf}; Chi2 (full) = {chi2_stat_full}; p-value={p_value_full}"
            )

            ax1.text(
                1.01,
                0.65,
                "\n".join(
                    [
                        rf"${chi2_label_full} = {chi2_stat_full:.1f}/{ndf}$",
                        rf"$\mathit{{p}}_\mathrm{{full}} = {p_value_full * 100:.0f}\,\%$",
                    ]
                ),
                ha="left",
                va="center",
                transform=ax1.transAxes,
                fontsize=args.legSize,
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

        # val -= c # TODO I don't think this should be here

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
    print(val, y)
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
    leg_args = {
        "ax": ax1,
        "ncols": args.legCols,
        "loc": args.legPos,
        "text_size": args.legSize,
    }
    if args.infileInclusive:
        leg_args.update(
            {
                "extra_handles": extra_handles,
                "extra_labels": ["Inclusive"],
                "custom_handlers": ["tripleband"],
            }
        )
    plot_tools.addLegend(**leg_args)

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
