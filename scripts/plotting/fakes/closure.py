import hist
import matplotlib as mpl
import mplhep as hep
import numpy as np

from utilities import parsing
from utilities.styles import styles
from wremnants import histselections as sel
from wremnants import plot_tools
from wremnants.datasets.datagroups import Datagroups
from wums import boostHistHelpers as hh
from wums import logging, output_tools


def plot_closure(
    h,
    outdir,
    suffix="",
    outfile=f"closureABCD",
    ratio=True,
    proc="",
    ylabel="a.u.",
    smoothing_mode="binned",
    normalized=False,
    bootstrap=False,
):
    h = hh.rebinHist(h, "pt", [26, 28, 30, 33, 40, 56])
    h = hh.disableFlow(h, "pt")

    smoothing_axis_name = "pt"
    fakerate_axes = ["eta", "pt", "charge"]

    fakerate_integration_axes = [a for a in fakerate_axes if a not in args.vars]

    hss = []
    labels = []

    if bootstrap:
        # store original values
        nsamples = 10000
        seed = 42
        values = h.values(flow=True)
        hBootstrap = h.copy()

    info = dict(
        fakerate_axes=fakerate_axes,
        smoothing_axis_name=smoothing_axis_name,
        rebin_smoothing_axis=None,
        integrate_x=True,
    )

    # signal selection
    hSel_sig = sel.SignalSelectorABCD(h, **info)
    hD_sig = hSel_sig.get_hist(h)
    hss.append(hD_sig)
    labels.append("D (truth)")

    info["smoothing_mode"] = smoothing_mode
    info["smoothing_order_fakerate"] = 2 if smoothing_mode == "fakerate" else 3

    # simple ABCD
    logger.info("Make simple ABCD prediction")
    if smoothing_mode in ["full", "fakerate"]:
        hSel_simple = sel.FakeSelectorSimpleABCD(h, **info)
        hD_simple = hSel_simple.get_hist(h)
        hss.append(hD_simple)
        labels.append(f"Simple {smoothing_mode} smoothed")
    else:
        hSel_simple = sel.FakeSelectorSimpleABCD(h, **info, upper_bound_y=None)

        if bootstrap:
            # throw posson toys
            toy_shape = [nsamples, *values.shape]
            rng = np.random.default_rng(seed)
            toys = rng.poisson(values, size=toy_shape)

            vals = []
            for i in range(nsamples):
                hBootstrap.values(flow=True)[...] = toys[i, ...]
                hSel_simple.h_nominal = None
                hD_simple = hSel_simple.get_hist(hBootstrap)
                vals.append(hD_simple.values(flow=True))

            vals = np.array(vals)
            toy_mean = np.mean(vals, axis=0)
            toy_var = np.var(vals, ddof=1, axis=0)
            hD_simple.values(flow=True)[...] = toy_mean
            hD_simple.variances(flow=True)[...] = toy_var
        else:
            hD_simple = hSel_simple.get_hist(h)

        hss.append(hD_simple)
        labels.append("Simple")

    # # extrapolate ABCD x-axis
    # if not smoothing_mode in ["full", "fakerate"]:
    #     # logger.info("Make extrapolated ABCD prediction")
    #     # hSel_Xpol0 = sel.FakeSelectorExtrapolateABCD(h, fakerate_axes=fakerate_axes, extrapolation_order=0)
    #     # hD_Xpol0 = hSel_Xpol0.get_hist(h)
    #     # hss.append(hD_Xpol0)
    #     # labels.append("pol0(x)")

    #     hSel_Xpol1 = sel.FakeSelectorExtrapolateABCD(h, fakerate_axes=fakerate_axes, extrapolation_order=1, rebin_x=[0,20,40,44,49,55,62])

    #     if bootstrap:
    #         # throw posson toys
    #         toy_shape = [nsamples, *values.shape]
    #         rng = np.random.default_rng(seed)
    #         toys = rng.poisson(values, size=toy_shape)

    #         vals = []
    #         for i in range(nsamples):
    #             hBootstrap.values(flow=True)[...] = toys[i,...]
    #             hSel_Xpol1.h_nominal = None
    #             hD_Xpol1 = hSel_Xpol1.get_hist(hBootstrap)
    #             vals.append(hD_Xpol1.values(flow=True))

    #         vals = np.array(vals)
    #         toy_mean = np.mean(vals, axis=0)
    #         toy_var = np.var(vals, ddof=1, axis=0)
    #         hD_Xpol1.values(flow=True)[...] = toy_mean
    #         hD_Xpol1.variances(flow=True)[...] = toy_var
    #     else:
    #         hD_Xpol1 = hSel_Xpol1.get_hist(h, is_nominal=True)
    #     hss.append(hD_Xpol1)
    #     labels.append("pol1(x)")

    #     # hSel_Xpol1p = sel.FakeSelectorExtrapolateABCD(h, fakerate_axes=fakerate_axes, extrapolation_order=1, rebin_x=[0,11,21,40,44,49,55,62,80])
    #     # hD_Xpol1p = hSel_Xpol1p.get_hist(h)
    #     # hss.append(hD_Xpol1p)
    #     # labels.append("pol1(x)'")

    #     # hSel_Xpol2 = sel.FakeSelectorExtrapolateABCD(h, fakerate_axes=fakerate_axes, extrapolation_order=2, rebin_x=[0,11,21,40,44,49,55,62,80])
    #     # hD_Xpol2 = hSel_Xpol2.get_hist(h)
    #     # hss.append(hD_Xpol2)
    #     # labels.append("pol2(x)")

    # extended ABCD in 5 control regions
    logger.info("Make 1D extended ABCD prediction in 5 control regions")
    if smoothing_mode in ["full", "fakerate"]:
        hSel_ext5 = sel.FakeSelector1DExtendedABCD(h, **info)
        hD_ext5 = hSel_ext5.get_hist(h)
        hss.append(hD_ext5)
        labels.append(f"Ext. 1D {smoothing_mode} smoothed")
    else:
        hSel_ext5 = sel.FakeSelector1DExtendedABCD(h, **info, upper_bound_y=None)

        if bootstrap:
            # throw posson toys
            toy_shape = [nsamples, *values.shape]
            rng = np.random.default_rng(seed)
            toys = rng.poisson(values, size=toy_shape)

            vals = []
            for i in range(nsamples):
                hBootstrap.values(flow=True)[...] = toys[i, ...]
                hSel_ext5.h_nominal = None
                hD_ext5 = hSel_ext5.get_hist(hBootstrap)
                vals.append(hD_ext5.values(flow=True))

            vals = np.array(vals)
            toy_mean = np.mean(vals, axis=0)
            toy_var = np.var(vals, ddof=1, axis=0)
            hD_ext5.values(flow=True)[...] = toy_mean
            hD_ext5.variances(flow=True)[...] = toy_var
        else:
            hD_ext5 = hSel_ext5.get_hist(h)

        hss.append(hD_ext5)
        labels.append(f"Ext. 1D")

        # hSel_ext5 = sel.FakeSelector1DExtendedABCD(h, **info, upper_bound_y=hist.overflow)
        # hD_ext5 = hSel_ext5.get_hist(h)
        # hss.append(hD_ext5)
        # labels.append("ext(5) binned (iso<0.45)")

    # extended ABCD in 8 control regions
    logger.info("Make 2D extended ABCD prediction in 8 control regions")
    if smoothing_mode in ["full", "fakerate"]:
        hSel_ext8 = sel.FakeSelector2DExtendedABCD(h, **info)
        hD_ext8 = hSel_ext8.get_hist(h)
        hss.append(hD_ext8)
        labels.append(f"Ext. 2D {smoothing_mode} smoothed")

        # hSel_ext8 = sel.FakeSelector2DExtendedABCD(h, **info, full_corrfactor=True, interpolation_order=1, smoothing_order_shapecorrection=[1,1])
        # hD_ext8 = hSel_ext8.get_hist(h)
        # hss.append(hD_ext8)
        # labels.append("ext(8) smoothed (full)")
    else:
        hSel_ext8 = sel.FakeSelector2DExtendedABCD(
            h,
            **info,
            upper_bound_y=None,
            integrate_shapecorrection_x=True,
            interpolate_x=False,
            smooth_shapecorrection=False,
        )

        if bootstrap:
            # throw posson toys
            toy_shape = [nsamples, *values.shape]
            rng = np.random.default_rng(seed)
            toys = rng.poisson(values, size=toy_shape)

            vals = []
            for i in range(nsamples):
                hBootstrap.values(flow=True)[...] = toys[i, ...]
                hSel_ext8.h_nominal = None
                hD_ext8 = hSel_ext8.get_hist(hBootstrap)
                vals.append(hD_ext8.values(flow=True))

            vals = np.array(vals)
            toy_mean = np.mean(vals, axis=0)
            toy_var = np.var(vals, ddof=1, axis=0)
            hD_ext8.values(flow=True)[...] = toy_mean
            hD_ext8.variances(flow=True)[...] = toy_var
        else:
            hD_ext8 = hSel_ext8.get_hist(h)

        hss.append(hD_ext8)
        labels.append("Ext. 2D")

        # labels.append("extended 2D")

        # using fullcorrection
        # hSel_ext8 = sel.FakeSelector2DExtendedABCD(h, **info, full_corrfactor=True, upper_bound_y=None,
        #     integrate_shapecorrection_x=False, interpolate_x=False, smooth_shapecorrection=False)
        # hD_ext8 = hSel_ext8.get_hist(h)
        # hss.append(hD_ext8)
        # labels.append("ext(8) binned (full)")

        # hSel_ext8 = sel.FakeSelector2DExtendedABCD(h, **info, full_corrfactor=True, upper_bound_y=None,
        #     integrate_shapecorrection_x=False, interpolate_x=False, smooth_shapecorrection=False)
        # hD_ext8 = hSel_ext8.get_hist(h)
        # hss.append(hD_ext8)
        # labels.append("ext(8) binned (pT vs. mT)")

        # hSel_ext8 = sel.FakeSelector2DExtendedABCD(h, **info, upper_bound_y=hist.overflow,
        #     integrate_shapecorrection_x=False, interpolate_x=False, smooth_shapecorrection=False)
        # hD_ext8 = hSel_ext8.get_hist(h)
        # hss.append(hD_ext8)
        # labels.append("ext(8) binned (iso<0.45)")

        # hSel_ext8 = sel.FakeSelector2DExtendedABCD(h, **info, integrate_shapecorrection_x=True, interpolate_x=False, smooth_shapecorrection=False)
        # hD_ext8 = hSel_ext8.get_hist(h)
        # hss.append(hD_ext8)
        # labels.append("ext(8) binned (mT integrated)")

    logger.info("Make plots")

    linestyles = ["-", "-", "--", "--", ":"]
    colors = mpl.colormaps["tab10"]

    if "charge" in hss[0].axes.name and len(hss[0].axes["charge"]) == 1:
        hss = [h[{"charge": slice(None, None, hist.sum)}] for h in hss]

    axes = [
        f"abs{a.capitalize()}" if args.absval[i] else a
        for i, a in enumerate(hss[0].axes.name)
    ]

    if len(axes) > 1:
        hss = [
            hh.unrolledHist(h, obs=hss[0].axes.name[::-1], add_flow_bins=True)
            for h in hss
        ]
        xlabel = f"{'-'.join([styles.xlabels.get(a,a).replace('(GeV)','') for a in axes])} bin"
    else:
        xlabel = (
            styles.xlabels[axes[0]]
            if len(axes) == 1 and axes in styles.xlabels
            else f"{'-'.join(axes)} Bin"
        )

    scales = [sum(h.values(flow=True)) for h in hss]
    scale0 = sum(hss[0].values(flow=True))
    scales_var = [sum(h.variances(flow=True)) for h in hss]
    scale0_var = sum(hss[0].variances(flow=True))
    if normalized:
        print(scales)
        hss = [hh.scaleHist(h, 1.0 / sum(h.values(flow=True))) for h in hss]
        ylabel = "a.u."
    else:
        ylabel = "Events / bin"

    for l, s, v in zip(labels, scales, scales_var):
        print(f"{l} = {scale0/s} +/- {(v/s**2+scale0_var/scale0**2)**0.5 * scale0/s}")

    hs = hss

    ymin = 0
    ymax = max([h.values(flow=True).max() for h in hs])
    yrange = ymax - ymin
    ymin = ymin if ymin == 0 else ymin - yrange * 0.3
    ymax = ymax + yrange * 0.3

    if normalized:
        rrange = 0.8, 1.2
    else:
        rrange = args.rrange

    if ratio:
        fig, ax1, ratio_axes = plot_tools.figureWithRatio(
            hss[0],
            xlabel=xlabel,
            ylabel=ylabel,
            cms_label=args.cmsDecor,
            rlabel=f"1/{labels[0]}",
            rrange=rrange,
            automatic_scale=False,
            width_scale=1.2,
            ylim=(ymin, ymax),
        )
        ax2 = ratio_axes[-1]
    else:
        fig, ax1 = plot_tools.figure(
            hss[0],
            xlabel=xlabel,
            ylabel=ylabel,
            cms_label=args.cmsDecor,
            automatic_scale=False,
            width_scale=1.2,
            ylim=(ymin, ymax),
        )

    # plot horizontal pT lines
    for l in [24, 48, 72, 96]:
        ax1.plot([l, l], [ymin, ymax], linestyle="--", color="k")

    labels = labels[: len(hs)]
    if ratio:
        hr = [hh.divideHists(h1d, hss[0]) for h1d in hss]

        if smoothing_mode in ["binned"]:
            chi2s = [
                sum(
                    (h1d.values(flow=True) - hss[0].values(flow=True)) ** 2
                    / (h1d.variances(flow=True) + hss[0].variances(flow=True))
                )
                for h1d in hss
            ]
            ndf = len(hss[0].values(flow=True)) - normalized
            labels = [
                rf"{l} $\chi^2/ndf={round(c)}/{ndf}$" if i != 0 else l
                for i, (l, c) in enumerate(zip(labels, chi2s))
            ]

    hep.histplot(
        hs,
        histtype="step",
        color=[colors(i) for i in range(len(hss))],
        label=labels,
        linestyle=linestyles[: len(hs)],
        ax=ax1,
    )

    if ratio:
        hep.histplot(
            hr,
            yerr=True,
            histtype="step",
            color=[colors(i) for i in range(len(hr))],
            label=labels,
            linestyle=linestyles[: len(hr)],
            ax=ax2,
        )

    fontsize = ax1.xaxis.label.get_size()

    ax1.text(
        1.0,
        1.003,
        styles.process_labels[proc],
        transform=ax1.transAxes,
        fontsize=30,
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    plot_tools.addLegend(ax1, ncols=2, text_size=fontsize * 0.68)
    plot_tools.fix_axes(ax1, ax2)

    if suffix:
        outfile = f"{suffix}_{outfile}"
    if args.postfix:
        outfile += f"_{args.postfix}"

    plot_tools.save_pdf_and_png(outdir, outfile)
    plot_tools.write_index_and_log(
        outdir,
        outfile,
        # yield_tables={"Stacked processes" : stack_yields, "Unstacked processes" : unstacked_yields},
        # analysis_meta_info={"AnalysisOutput" : groups.getMetaInfo()},
        args=args,
    )


if __name__ == "__main__":
    parser = parsing.plot_parser()
    parser.add_argument(
        "infile",
        help="Output file of the analysis stage, containing ND boost histograms",
    )
    parser.add_argument(
        "-n",
        "--baseName",
        type=str,
        help="Histogram name in the file (e.g., 'nominal')",
        default="nominal",
    )
    parser.add_argument(
        "--procFilters",
        type=str,
        nargs="*",
        default=["QCD"],
        help="Filter to plot (default no filter, only specify if you want a subset",
    )
    parser.add_argument(
        "--vars",
        type=str,
        nargs="*",
        default=["eta", "pt", "charge"],
        help="Variables to be considered in rebinning",
    )
    parser.add_argument(
        "--rebin",
        type=int,
        nargs="*",
        default=[],
        help="Rebin axis by this value (default, 1, does nothing)",
    )
    parser.add_argument(
        "--absval",
        type=int,
        nargs="*",
        default=[],
        help="Take absolute value of axis if 1 (default, 0, does nothing)",
    )
    parser.add_argument(
        "--axlim",
        type=float,
        default=[],
        nargs="*",
        help="Restrict axis to this range (assumes pairs of values by axis, with trailing axes optional)",
    )
    parser.add_argument(
        "--rebinBeforeSelection",
        action="store_true",
        help="Rebin before the selection operation (e.g. before fake rate computation), default if after",
    )
    parser.add_argument(
        "--rrange",
        type=float,
        nargs=2,
        default=[0.25, 1.75],
        help="y range for ratio plot",
    )
    # x-axis for ABCD method
    parser.add_argument(
        "--xAxisName", type=str, help="Name of x-axis for ABCD method", default="mt"
    )
    parser.add_argument(
        "--xBinsSideband",
        type=float,
        nargs="*",
        help="Binning of x-axis for ABCD method in sideband region",
        default=[0, 2, 5, 9, 14, 20, 27, 40],
    )
    parser.add_argument(
        "--xBinsSignal",
        type=float,
        nargs="*",
        help="Binning of x-axis of ABCD method in signal region",
        default=[50, 70, 120],
    )
    parser.add_argument(
        "--xOrder",
        type=int,
        default=2,
        help="Order in x-axis for fakerate parameterization",
    )
    # y-axis for ABCD method
    parser.add_argument(
        "--yAxisName", type=str, help="Name of y-axis for ABCD method", default="iso"
    )
    parser.add_argument(
        "--yBinsSideband",
        type=float,
        nargs="*",
        help="Binning of y-axis of ABCD method in sideband region",
        default=[0.15, 0.2, 0.25, 0.3],
    )
    parser.add_argument(
        "--yBinsSignal",
        type=float,
        nargs="*",
        help="Binning of y-axis of ABCD method in signal region",
        default=[0, 0.15],
    )
    parser.add_argument(
        "--yOrder",
        type=int,
        default=2,
        help="Order in y-axis for fakerate parameterization",
    )
    # axis for smoothing
    parser.add_argument(
        "--smoothingAxisName",
        type=str,
        help="Name of second axis for ABCD method, 'None' means 1D fakerates",
        default=None,
    )
    parser.add_argument(
        "--smoothingOrder",
        type=int,
        nargs="*",
        default=[2, 1, 0],
        help="Order in second axis for fakerate parameterization",
    )

    args = parser.parse_args()
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)

    groups = Datagroups(args.infile, excludeGroups=None)

    if args.axlim or args.rebin or args.absval:
        groups.set_rebin_action(
            args.vars, args.axlim, args.rebin, args.absval, rename=False
        )

    logger.info(f"Load fakes")
    groups.loadHistsForDatagroups(
        args.baseName, syst="", procsToRead=args.procFilters, applySelection=False
    )

    histInfo = groups.getDatagroups()

    for proc in args.procFilters:
        h = histInfo[proc].hists[args.baseName]

        if proc != groups.fakeName:
            plot_closure(
                h, outdir, suffix=f"{proc}", proc=proc, smoothing_mode="binned"
            )
            plot_closure(
                h,
                outdir,
                suffix=f"{proc}_normalized",
                proc=proc,
                smoothing_mode="binned",
                normalized=True,
            )
            plot_closure(
                h,
                outdir,
                suffix=f"{proc}_smooth_fakerate",
                proc=proc,
                smoothing_mode="fakerate",
            )
            plot_closure(
                h,
                outdir,
                suffix=f"{proc}_smooth_full",
                proc=proc,
                smoothing_mode="full",
            )

    if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
        output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)
