import argparse

from makemytensor_helpers import (
    _reorder_hist_axes,
    _resolve_plot_labels,
    assert_matching_axes,
    collapse_axes,
    load_histogram,
    plot_curvature_response,
    plot_variation_projection,
    rebin_histogram,
    rebin_variation_unc_axis,
)

from rabbit import tensorwriter
from wums import boostHistHelpers as hh


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the BuToJpsiK histograms into a Rabbit tensor."
    )
    parser.add_argument(
        "-i", "--input-file", required=True, help="File that stores the histograms."
    )
    parser.add_argument(
        "-O",
        "--output",
        dest="tensor_output",
        default="./",
        help="Directory for the tensor file.",
    )
    parser.add_argument(
        "--outname", default="btojpsik_tensor.hdf5", help="Tensor filename."
    )
    parser.add_argument(
        "--channel", default="btojpsik_stuff", help="Channel name stored in the tensor."
    )
    parser.add_argument(
        "--dataset-signal",
        required=True,
        help="Dataset key for the nominal signal histogram and its variations.",
    )
    parser.add_argument(
        "--background",
        dest="backgrounds",
        action="append",
        default=[],
        metavar="PROCESS=DATASET",
        help="Background process (repeatable, format: name=dataset_key).",
    )
    parser.add_argument(
        "--signal-process",
        default="signal",
        help="Process label used for the signal template.",
    )
    parser.add_argument(
        "--signal-norm-uncertainty",
        dest="signal_norm_uncertainty",
        type=float,
        default=None,
        help="Optional lnN uncertainty applied to the signal normalization.",
    )

    parser.add_argument(
        "--systematic-labels",
        nargs="+",
        default=("A", "e", "M"),
        help="Order of the labels embedded in the 'unc' axis.",
    )
    parser.add_argument(
        "--variation-down-index",
        type=int,
        default=0,
        help="Index selecting the down variation.",
    )
    parser.add_argument(
        "--variation-up-index",
        type=int,
        default=1,
        help="Index selecting the up variation.",
    )

    parser.add_argument(
        "--mass-axis", default="bkmm_jpsimc_mass", help="Name of the mass axis."
    )
    parser.add_argument(
        "--pt-axis",
        default="bkmm_kaon_pt",
        help="Name of the kaon-pt axis to be summed over.",
    )
    parser.add_argument(
        "--eta-axis",
        default="bkmm_kaon_eta",
        help="Name of the eta axis kept in the tensor.",
    )
    parser.add_argument(
        "--charge-axis",
        default="bkmm_kaon_charge",
        help="Name of the charge axis kept in the tensor.",
    )
    parser.add_argument(
        "--systematic-axis",
        default="unc",
        help="Axis that enumerates (A,e,M)*neta bins.",
    )
    parser.add_argument(
        "--variation-axis",
        default="downUpVar",
        help="Axis that encodes up/down variations.",
    )
    parser.add_argument(
        "--curvature-axis",
        default="bkmm_kaon_curvature",
        help="Name of the curvature axis (k) used in the diagnostic plot.",
    )
    parser.add_argument(
        "--systematic-type",
        default="log_normal",
        choices=["log_normal", "normal"],
        help="TensorWriter systematic type.",
    )
    parser.add_argument(
        "--plot-curvature-response",
        nargs="*",
        metavar="SYST",
        default=None,
        help="Produce k'/k response plots per eta bin. Optionally provide a subset of systematics (e.g. A M) to plot.",
    )
    parser.add_argument(
        "-o",
        "--plot-output",
        default=None,
        help="Directory where curvature-response plots are stored.",
    )
    parser.add_argument(
        "--plot-curvature-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to (k'/k - 1); useful for magnifying or shrinking variations.",
    )
    parser.add_argument(
        "--plot-variation",
        default=None,
        help="Plot up/down variations projected onto the given axis (e.g. bkmm_jpsimc_mass).",
    )
    parser.add_argument(
        "--plot-variation-ratio",
        default=None,
        help="Plot up/down variation ratios projected onto the given axis (e.g. bkmm_jpsimc_curvature).",
    )
    parser.add_argument(
        "--bin-plot-variation",
        default=None,
        help="If set, plot the variation axis distribution in each bin of this axis.",
    )
    parser.add_argument(
        "--overlay-bin-variations",
        action="store_true",
        help="Overlay each bin of --bin-plot-variation on the same plot.",
    )
    parser.add_argument(
        "--plot-variation-eta",
        type=int,
        default=None,
        help="Optional eta-bin index for variation plots. If unset, plot all eta bins.",
    )
    parser.add_argument(
        "--plot-variation-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to variation curves for yield/ratio plots.",
    )
    parser.add_argument(
        "--debug-variation",
        action="store_true",
        help="Print debug summaries for variation templates.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    signal_hist, data_hist, variation_hist = load_histogram(
        args.input_file, args.dataset_signal
    )

    n_pt_bins = signal_hist.axes[args.pt_axis].size
    n_mass_bins = 10
    rebinning = {
        args.eta_axis: 7,
        args.mass_axis: n_mass_bins,
    }
    for axis, bins in rebinning.items():
        print(f"Rebinning {axis} into {bins} bins")

    eta_rebin_factor = None
    if args.eta_axis in rebinning:
        eta_bins = rebinning[args.eta_axis]
        eta_rebin_factor = signal_hist.axes[args.eta_axis].size // eta_bins

    signal_hist = rebin_histogram(signal_hist, rebinning)
    data_hist = rebin_histogram(data_hist, rebinning)

    variation_rebinning = dict(rebinning)
    if (
        eta_rebin_factor is not None
        and args.systematic_axis in variation_hist.axes.name
    ):
        variation_hist = rebin_variation_unc_axis(
            variation_hist,
            args.eta_axis,
            args.systematic_axis,
            args.systematic_labels,
            eta_rebin_factor,
        )
        variation_rebinning.pop(args.eta_axis, None)
    variation_hist = rebin_histogram(variation_hist, variation_rebinning)

    # lose stats quick at "high" pT...

    # import pdb
    # pdb.set_trace()

    background_hists = {}
    for spec in args.backgrounds:
        if "=" not in spec:
            raise argparse.ArgumentTypeError(
                f"Background specification '{spec}' does not contain '='."
            )
        process, dataset = spec.split("=", maxsplit=1)
        background_nominal, _, _ = load_histogram(args.input_file, dataset)
        background_hists[process] = background_nominal

    # drop_nominal_axes = [args.pt_axis, args.systematic_axis, args.variation_axis]
    # drop_nominal_axes = [args.systematic_axis, args.variation_axis]
    # signal_hist = collapse_axes(signal_hist, drop_nominal_axes)
    # for key, h in background_hists.items():
    #    background_hists[key] = collapse_axes(h, drop_nominal_axes)

    assert_matching_axes(background_hists, signal_hist, label="Background")

    # add an artificial flat background
    total_yield = signal_hist.values().sum()
    bkg_yield = total_yield * 0.40
    bkg_hist = signal_hist.copy()
    bkg_hist.values()[...] = bkg_yield / bkg_hist.size
    if bkg_hist.variances() is not None:
        # poisson
        # bkg_hist.variances()[...] = bkg_hist.values()
        # no statistical uncertainty for flat background
        bkg_hist.variances()[...] = 0.0

    # combine artificial background with signal MC for "data"
    # data_hist = hist.Hist(*signal_hist.axes, storage=hist.storage.Weight())
    # data_hist.values()[...] = signal_hist.values() + bkg_hist.values()
    # sig_vars = signal_hist.variances()
    # bkg_vars = bkg_hist.variances()
    # if sig_vars is not None and bkg_vars is not None:
    #    data_hist.variances()[...] = sig_vars + bkg_vars
    # else:
    #    # poisson
    #    data_hist.variances()[...] = data_hist.values()

    # tensor writer now
    writer = tensorwriter.TensorWriter(systematic_type=args.systematic_type)
    writer.add_channel(signal_hist.axes, name=args.channel)
    writer.add_data(data_hist, channel=args.channel)
    writer.add_process(signal_hist, args.signal_process, args.channel, signal=True)
    for proc, h in background_hists.items():
        writer.add_process(h, proc, args.channel)
    # artificial background
    writer.add_process(bkg_hist, "flatBkg", args.channel)

    # if args.signal_norm_uncertainty is not None:
    #    writer.add_norm_systematic(
    #        name="signal_norm",
    #        process=args.signal_process,
    #        channel=args.channel,
    #        uncertainty=args.signal_norm_uncertainty,
    #    )

    n_eta_bins = signal_hist.axes[args.eta_axis].size
    n_charge_bins = signal_hist.axes[args.charge_axis].size

    # free float yields in bins of pt eta charge
    new_sig_basis = hh.expand_hist_by_duplicate_axes(
        signal_hist,
        [args.charge_axis, args.pt_axis, args.eta_axis],
        ["chargeVar", "ptVar", "etaVar"],
        put_trailing=True,
        flow=False,
    )
    new_bkg_basis = hh.expand_hist_by_duplicate_axes(
        bkg_hist,
        [args.charge_axis, args.pt_axis, args.eta_axis],
        ["chargeVar", "ptVar", "etaVar"],
        put_trailing=True,
        flow=False,
    )

    procs = {args.signal_process: signal_hist, "flatBkg": bkg_hist}
    basis_by_proc = {
        args.signal_process: new_sig_basis,
        "flatBkg": new_bkg_basis,
    }
    # unc = 1.5
    unc = 0.1
    for proc, nominal_hist in procs.items():
        basis_hist = basis_by_proc[proc]
        # if proc == args.signal_process:
        #    continue
        for icharge in range(n_charge_bins):
            for ipt in range(n_pt_bins):
                for ieta in range(n_eta_bins):

                    mask = {"etaVar": ieta, "ptVar": ipt, "chargeVar": icharge}
                    bin = basis_hist[mask]

                    syst_name = f"norm_{proc}_eta{ieta}_pt{ipt}_charge{icharge}"

                    up_hist = nominal_hist + bin * unc
                    # down_hist--symmetric

                    writer.add_systematic(
                        up_hist,
                        name=syst_name,
                        process=proc,
                        channel=args.channel,
                        constrained=False,
                        noi=True,
                    )

    # signal yield systematics matching binning for A,e,M
    # for ieta in range(n_eta_bins):
    #    mask = {"etaVar": ieta}
    #    bin = new_sig_basis[mask]
    #    syst_name = f"norm_{args.signal_process}_eta{ieta}"
    #    bin = new_sig_basis[{"etaVar": ieta}]
    #    up_hist = signal_hist + bin * unc
    #
    #    writer.add_systematic(
    #        up_hist,
    #        name=syst_name,
    #        process=args.signal_process,
    #        channel=args.channel,
    #        constrained=False,
    #        noi=True
    #    )

    # A e M bitchhhh
    labels = tuple(args.systematic_labels)

    plot_labels = _resolve_plot_labels(args)
    if plot_labels:
        plot_curvature_response(signal_hist, variation_hist, args, plot_labels)
    if args.plot_variation and args.plot_variation_ratio:
        raise ValueError(
            "--plot-variation and --plot-variation-ratio are mutually exclusive."
        )
    if args.plot_variation:
        plot_variation_projection(
            signal_hist,
            variation_hist,
            args,
            args.plot_variation,
            eta_only=args.plot_variation_eta,
            mode="yield",
            bin_axis=args.bin_plot_variation,
        )
    if args.plot_variation_ratio:
        plot_variation_projection(
            signal_hist,
            variation_hist,
            args,
            args.plot_variation_ratio,
            eta_only=args.plot_variation_eta,
            mode="ratio",
            bin_axis=args.bin_plot_variation,
        )

    for eta_idx in range(n_eta_bins):
        for label in labels:
            offset = labels.index(label)
            systematic_bin = len(labels) * eta_idx + offset

            up_selection = {
                args.systematic_axis: systematic_bin,
                args.variation_axis: args.variation_up_index,
            }
            down_selection = {
                args.systematic_axis: systematic_bin,
                args.variation_axis: args.variation_down_index,
            }

            up_variation = collapse_axes(
                variation_hist[up_selection],
                [args.systematic_axis, args.variation_axis],
            )
            down_variation = collapse_axes(
                variation_hist[down_selection],
                [args.systematic_axis, args.variation_axis],
            )

            target_axes = signal_hist.axes.name
            up_variation = _reorder_hist_axes(up_variation, target_axes)
            down_variation = _reorder_hist_axes(down_variation, target_axes)

            if up_variation.axes.name != signal_hist.axes.name:
                raise RuntimeError(
                    f"Up variation axes {up_variation.axes.name} do not match nominal axes {signal_hist.axes.name}."
                )
            if down_variation.axes.name != signal_hist.axes.name:
                raise RuntimeError(
                    f"Down variation axes {down_variation.axes.name} do not match nominal axes {signal_hist.axes.name}."
                )

            up_hist = signal_hist.copy()
            down_hist = signal_hist.copy()
            for charge_idx in range(n_charge_bins):
                up_hist.values()[..., charge_idx] = up_variation.values()[
                    ..., charge_idx
                ]
                down_hist.values()[..., charge_idx] = down_variation.values()[
                    ..., charge_idx
                ]

                up_vars = up_variation.variances()
                down_vars = down_variation.variances()
                if up_vars is not None:
                    up_hist.variances()[..., charge_idx] = up_vars[..., charge_idx]
                if down_vars is not None:
                    down_hist.variances()[..., charge_idx] = down_vars[..., charge_idx]

            writer.add_systematic(
                [up_hist, down_hist],
                name=f"{label}_eta{eta_idx}",
                process=args.signal_process,
                channel=args.channel,
                constrained=False,
                noi=True,
            )
        # up_hist.values()[..., charge_idx] = up_variation.values()[..., charge_idx]
        # down_hist.values()[..., charge_idx] = down_variation.values()[..., charge_idx]
    #
    # up_vars = up_variation.variances()
    # down_vars = down_variation.variances()
    # up_hist_vars = up_hist.variances()
    # down_hist_vars = down_hist.variances()
    #
    # if up_vars is not None and up_hist_vars is not None:
    #    up_hist_vars[..., charge_idx] = up_vars[..., charge_idx]
    # if down_vars is not None and down_hist_vars is not None:
    #    down_hist_vars[..., charge_idx] = down_vars[..., charge_idx]
    #
    # writer.add_systematic(
    #    [up_hist, down_hist],
    #    name=f"{label}_eta{eta_idx}",
    #    process=args.signal_process,
    #    channel=args.channel,
    #    constrained=False,
    # )

    writer.write(args.tensor_output, args.outname, args=args)


if __name__ == "__main__":
    main()
