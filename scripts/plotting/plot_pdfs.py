import argparse

import lhapdf
import matplotlib.pyplot as plt
import numpy as np

from wremnants.utilities import theory_utils
from wums import output_tools, plot_tools

PARTON_FLAVOR_NAMES = {
    "uv": "u_{V}",
    "1": "d",
    "-1": r"\bar{d}",
    "dbar": r"\bar{d}",
    "2": "u",
    "-2": r"\bar{u}",
    "ubar": r"\bar{u}",
    "3": "s",
    "-3": r"\bar{s}",
    "sbar": r"\bar{s}",
    "dv": "d_{v}",
    "rs": "r_{s}",
}


def make_pdf_plot(
    flavor, Q_scale, x_range, all_values, all_errors, labels, colors, outdir, args
):
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    reference_central = all_values[0]

    for i, (central, err_pair) in enumerate(zip(all_values, all_errors)):
        color = colors[i] if colors and i < len(colors) else None

        # Unpack the list [err_dn, err_up]
        err_dn = err_pair[0]
        err_up = err_pair[1]

        # 1. Main Plot
        ax1.plot(x_range, central, color=color, label=labels[i])
        # Use err_dn and err_up separately
        ax1.fill_between(
            x_range, central - err_dn, central + err_up, color=color, alpha=0.2
        )

        # 2. Ratio Plot
        ratio_central = central / reference_central
        ax2.plot(x_range, ratio_central, color=color)
        ax2.fill_between(
            x_range,
            (central - err_dn) / reference_central,
            (central + err_up) / reference_central,
            color=color,
            alpha=0.2,
        )

    # Formatting
    flav_label = PARTON_FLAVOR_NAMES.get(str(flavor), flavor)
    ax1.set_ylabel(f"$x {flav_label}(x, Q^2)$", fontsize=32)
    ax1.legend(loc="upper left")
    ax1.tick_params(labelsize=24)
    ax1.text(
        0.98,
        0.95,
        f"$Q = {Q_scale}$ GeV",
        transform=ax1.transAxes,
        fontsize=20,
        ha="right",
        va="top",
    )
    if args and args.yRange:
        ax1.set_ylim(*args.yRange)
    else:
        ymin, ymax = ax1.get_ylim()
        ax1.set_ylim(ymin, ymax * 1.2)

    ax2.axhline(1.0, color="black", lw=1, ls="--")
    ax2.set_ylabel("Ratio", fontsize=28)
    ax2.set_xlabel(r"$x$", fontsize=24)
    ax2.tick_params(labelsize=24)
    ax2.set_xscale("log")
    ax2.set_xlim(1e-4, 1.0)
    ratio_range = args.ratioRange if args and args.ratioRange else [0.8, 1.2]
    ax2.set_ylim(*ratio_range)

    outfile = f"pdf_{flavor}_Q{int(Q_scale)}"
    if args and args.postfix:
        outfile += f"_{args.postfix}"

    plot_tools.save_pdf_and_png(outdir, outfile)
    output_tools.write_index_and_log(outdir, outfile, args=args)
    # plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare PDF sets and Fit Results")
    parser.add_argument("-p", "--postfix", help="Label to append to plot name")
    parser.add_argument(
        "-s", "--pdf-sets", nargs="+", default=[], help="LHAPDF set names"
    )
    parser.add_argument(
        "-r", "--fit-results", nargs="+", default=[], help="HDF5 fitresult files"
    )
    parser.add_argument(
        "--fit-types",
        nargs="+",
        default=["postfit"],
        help="Types inside fit file (e.g. prefit postfit)",
    )
    parser.add_argument("-l", "--labels", nargs="+", help="Labels for the legend")
    parser.add_argument(
        "-f",
        "--flavors",
        nargs="+",
        required=True,
        help="Flavors (uv, dv, rs, or PDG ID)",
    )
    parser.add_argument(
        "-q", "--q-scale", type=float, default=80.360, help="Q scale in GeV"
    )
    parser.add_argument("-o", "--outpath", required=True, help="Output directory")
    parser.add_argument("--colors", nargs="+", help="List of colors")
    parser.add_argument(
        "--ratioRange",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Y-axis range for the ratio panel (default: 0.8 1.2)",
    )
    parser.add_argument(
        "--yRange",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Y-axis range for the main plot panel (default: auto)",
    )
    parser.add_argument(
        "--lhapdf-path",
        default="/scratch/submit/cms/wmass/PostfitPDF/",
        help="Path to LHAPDF data",
    )

    args = parser.parse_args()
    lhapdf.pathsAppend(args.lhapdf_path)

    x_range = np.logspace(-4, -0.01, 201)[
        :-1
    ]  # Remove the last entry for consistency with the rabbit plots
    outdir = output_tools.make_plot_dir(args.outpath, "", eoscp=True)

    for flavor in args.flavors:
        all_vals = []
        all_errs = []
        plot_labels = []

        # Load LHAPDF data
        if args.pdf_sets:
            lha_vals, lha_errs = theory_utils.read_pdf_vals_and_errors(
                flavor, args.q_scale, x_range, args.pdf_sets
            )
            all_vals.extend(lha_vals)
            all_errs.extend(lha_errs)
            plot_labels.extend(args.pdf_sets)

        # Load Fit Result data
        for fit_file in args.fit_results:
            fit_vals, fit_errs = theory_utils.read_vals_and_errors_from_fit(
                fit_file, args.fit_types, flavor
            )
            all_vals.extend(fit_vals)
            all_errs.extend(fit_errs)
            # Create a label for each fit type in each file
            for ftype in args.fit_types:
                plot_labels.append(f"{fit_file.split('/')[-1]}_{ftype}")

        # Override labels if provided by user
        if args.labels and len(args.labels) == len(all_vals):
            plot_labels = args.labels

        make_pdf_plot(
            flavor,
            args.q_scale,
            x_range,
            all_vals,
            all_errs,
            plot_labels,
            args.colors,
            outdir,
            args,
        )

    if output_tools.is_eosuser_path(args.outpath):
        output_tools.copy_to_eos(outdir, args.outpath, "")


if __name__ == "__main__":
    main()
