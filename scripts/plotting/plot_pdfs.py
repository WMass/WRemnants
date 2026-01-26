import argparse

import lhapdf
import matplotlib.pyplot as plt
import numpy as np

from wums import output_tools, plot_tools

FLAVOR_NAMES = {
    "uv": "u_{V}",
    "1": "d",
    "-1": r"\bar{d}",
    "2": "u",
    "-2": r"\bar{u}",
    "3": "s",
    "-3": r"\bar{s}",
    "dv": "d_{v}",
    "rs": "r_{s}",
}


# A subset of the options (can be extended) taken from
# https://gist.github.com/bendavid/601286f2fc8d89b30d7c20d108782a76#file-plotpdf-py-L782-L823
def eval_pdf(pdf, flav, x, q):
    # Try to convert string digits to int for PDG IDs
    try:
        if (
            isinstance(flav, int)
            or flav.isdigit()
            or (flav.startswith("-") and flav[1:].isdigit())
        ):
            return pdf.xfxQ(int(flav), x, q)
    except AttributeError:
        pass

    if flav == "uv":
        return pdf.xfxQ(2, x, q) - pdf.xfxQ(-2, x, q)
    elif flav == "dv":
        return pdf.xfxQ(1, x, q) - pdf.xfxQ(-1, x, q)
    elif flav == "rs":
        denom = pdf.xfxQ(-1, x, q) + pdf.xfxQ(-2, x, q)
        return (pdf.xfxQ(3, x, q) + pdf.xfxQ(-3, x, q)) / denom if denom != 0 else 0
    else:
        raise NotImplementedError(f"Flavor type {flav} is unsupported")


def get_pdf_data(pdf_name, flavor, Q, x_range):
    pdf_set = lhapdf.getPDFSet(pdf_name)
    members = pdf_set.mkPDFs()
    # Calculate values for all members (exclude alpha_s members if present)
    all_vals = np.array(
        [
            [eval_pdf(m, flavor, x, Q) for x in x_range]
            for m in members[: pdf_set.errorInfo.nmemCore + 1]
        ]
    )
    central = all_vals[0]
    # Hessian uncertainty
    delta_f = np.sqrt(np.sum((all_vals[1:] - central) ** 2, axis=0))
    return central, delta_f


def make_pdf_plot(flavor, Q_scale, pdf_sets, labels, colors, outdir, args):
    x_range = np.logspace(-4, -0.01, 200)
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    reference_central = None

    for i, name in enumerate(pdf_sets):
        central, err = get_pdf_data(name, flavor, Q_scale, x_range)
        if i == 0:
            reference_central = central

        # 1. Main Plot
        ax1.plot(x_range, central, color=colors[i], label=labels[i])
        ax1.fill_between(
            x_range, central - err, central + err, color=colors[i], alpha=0.2
        )

        # 2. Ratio Plot
        ratio_central = central / reference_central
        ratio_err = err / reference_central
        ax2.plot(x_range, ratio_central, color=colors[i])
        ax2.fill_between(
            x_range,
            ratio_central - ratio_err,
            ratio_central + ratio_err,
            color=colors[i],
            alpha=0.2,
        )

    # Formatting
    flav_label = FLAVOR_NAMES.get(str(flavor), flavor)
    ax1.set_ylabel(f"$x {flav_label}(x, Q^2)$", fontsize=16)
    ax1.set_title(f"PDF at $Q = {Q_scale}$ GeV", fontsize=14)
    ax1.legend(loc="upper left")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.axhline(1.0, color="black", lw=1, ls="--")
    ax2.set_ylabel("Ratio to central", fontsize=14)
    ax2.set_xlabel(r"$x$", fontsize=12)
    ax2.set_xscale("log")
    ax2.set_ylim(0.8, 1.2)
    ax2.grid(True, which="both", alpha=0.3)

    outfile = f"pdf_{flavor}_Q{int(Q_scale)}"
    if args.postfix:
        outfile += f"_{args.postfix}"
    plot_tools.save_pdf_and_png(outdir, outfile)
    output_tools.write_index_and_log(
        outdir,
        outfile,
        args=args,
    )

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate PDF plots from LHAPDF sets")
    parser.add_argument("-p", "--postfix", help="Label to append to the plot name")
    parser.add_argument(
        "-s", "--pdf-sets", nargs="+", required=True, help="LHAPDF set names"
    )
    parser.add_argument("-l", "--labels", nargs="+", help="Labels for the legend")
    parser.add_argument(
        "-f", "--flavors", nargs="+", help="Flavors (uv, dv, rs, or PDG ID)"
    )
    parser.add_argument(
        "-q", "--q-scale", type=float, default=80.360, help="Q scale in GeV"
    )
    parser.add_argument("-o", "--outpath", required=True, help="Output filename")
    parser.add_argument(
        "--lhapdf-path",
        default="/scratch/submit/cms/wmass/PostfitPDF/",
        help="Additional path to LHAPDF data files (for custom sets)",
    )
    parser.add_argument("--colors", nargs="+", help="List of colors for the sets")

    args = parser.parse_args()

    # Set LHAPDF path - can also be set via environment variable LHAPDF_DATA_PATH
    lhapdf.pathsAppend(args.lhapdf_path)

    # If labels aren't provided, use PDF set names
    labels = args.labels if args.labels else args.pdf_sets

    outdir = output_tools.make_plot_dir(args.outpath, "", eoscp=True)

    for flavor in args.flavors:
        make_pdf_plot(
            flavor=flavor,
            Q_scale=args.q_scale,
            pdf_sets=args.pdf_sets,
            labels=labels,
            colors=args.colors,
            outdir=outdir,
            args=args,
        )

    if output_tools.is_eosuser_path(args.outpath):
        output_tools.copy_to_eos(outdir, args.outpath, "")


if __name__ == "__main__":
    main()
