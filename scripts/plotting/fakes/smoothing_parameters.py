import matplotlib as mpl
import numpy as np

from utilities import parsing
from utilities.styles import styles
from wremnants import histselections as sel
from wremnants import plot_tools
from wremnants.datasets.datagroups import Datagroups
from wums import boostHistHelpers as hh
from wums import logging, output_tools


def plot_params(h, params, params_err, label=None, suffix="", proc=""):

    logger.info(f"Min(chi2)={params.min()}; Max(chi2)={params.max()}")
    logger.info(f"Mean(chi2)={params.mean()}; std(chi2)={params.std()}")

    p_mean_label = rf"$\mu = {round(params.mean(), 1)}$"
    p_std_label = rf"$\sigma = {round(params.std(), 1)}$"

    as_histogram = False

    if as_histogram:
        xlim = [np.min([p.min() for p in params]), np.max([p.max() for p in params])]
        xlabel = "Parameter difference"
        ylabel = "Entries"
    else:
        xlim = [-2.4, 2.4]
        xlabel = styles.xlabels.get("eta", "eta")
        ylabel = "Parameter difference"

    colors = mpl.colormaps["tab10"]

    fig, ax1 = plot_tools.figure(
        h,
        ylabel=ylabel,
        xlabel=xlabel,
        cms_label=args.cmsDecor,
        xlim=xlim,
        ylim=None,
        logy=False,
        automatic_scale=False,
    )

    fontsize = ax1.xaxis.label.get_size()

    if as_histogram:
        # plot as histogram
        n, bins, _ = ax1.hist(
            params,
            bins=50,
            range=xlim,
            color=colors(0),
            # label=label,
            histtype="step",
        )
    else:
        ax1.errorbar(
            np.arange(-2.35, 2.4, 0.1),
            params,
            yerr=params_err,
            color="black",
            linestyle="",
            marker="o",
        )
        ax1.plot([-2.4, 2.4], [0, 0], linestyle="-", color="red")

    ax1.text(
        0.96,
        0.85,
        p_mean_label,
        transform=ax1.transAxes,
        fontsize=fontsize,
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    ax1.text(
        0.96,
        0.75,
        p_std_label,
        transform=ax1.transAxes,
        fontsize=fontsize,
        verticalalignment="bottom",
        horizontalalignment="right",
    )

    ax1.text(
        1.0,
        1.003,
        "QCD MC" if proc == "QCD" else styles.process_labels[proc],
        transform=ax1.transAxes,
        fontsize=fontsize,
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    # plot_tools.addLegend(ax1, ncols=1, text_size=fontsize, loc="upper left")

    outfile = "parameters" + (f"_{suffix}" if suffix != "" else "")
    if args.postfix:
        outfile += f"_{args.postfix}"

    plot_tools.save_pdf_and_png(outdir, outfile)
    plot_tools.write_index_and_log(outdir, outfile, args=args)


if __name__ == "__main__":
    parser = parsing.plot_parser()
    parser.add_argument(
        "infile",
        help="Output file of the analysis stage, containing ND boost histograms",
    )
    parser.add_argument(
        "--rrange",
        type=float,
        nargs=2,
        default=[-2.5, 2.5],
        help="y range for ratio plot",
    )

    args = parser.parse_args()
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)

    groups = Datagroups(args.infile, excludeGroups=None)

    fakerate_axes = ["eta", "pt", "charge"]
    smoothing_order_fakerate = 2
    smoothing_order_spectrum = 3
    smoothing_polynomial_spectrum = "bernstein"

    hist_fake = groups.results["QCDmuEnrichPt15PostVFP"]["output"]["unweighted"].get()

    logger.info("Make full fake prediction w/o rebinning")
    fakeselector = sel.FakeSelector1DExtendedABCD(
        hist_fake,
        fakerate_axes=fakerate_axes,
        smoothing_order_fakerate=smoothing_order_fakerate,
        smoothing_order_spectrum=smoothing_order_spectrum,
        smoothing_polynomial_spectrum=smoothing_polynomial_spectrum,
        smoothing_mode="full",
        rebin_smoothing_axis=None,
        throw_toys=None,
    )

    _0, _1 = fakeselector.calculate_fullABCD_smoothed(hist_fake, signal_region=True)
    params_d = fakeselector.spectrum_regressor.params
    cov_d = fakeselector.spectrum_regressor.cov

    hist_fake = hh.scaleHist(hist_fake, fakeselector.global_scalefactor)
    _0, _1 = fakeselector.calculate_fullABCD_smoothed(hist_fake)
    params = fakeselector.spectrum_regressor.params
    cov = fakeselector.spectrum_regressor.cov

    for ip in range(params.shape[-1]):

        for charge in (0, 1):

            p = params[..., charge, ip]
            p_d = params_d[..., charge, ip]

            p_diff = p - p_d
            p_err = (cov[..., charge, ip, ip] + cov_d[..., charge, ip, ip]) ** 0.5

            plot_params(
                hist_fake,
                p_diff,
                p_err,
                label="observed - predicted",
                suffix=f"diff_charge{charge}_{smoothing_polynomial_spectrum}_c{ip}",
                proc="QCD",
            )

    if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
        output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)
