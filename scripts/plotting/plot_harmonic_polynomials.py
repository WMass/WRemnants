import math

import hist
import matplotlib as mpl
import numpy as np

from utilities import parsing
from utilities.styles import styles
from wremnants import plot_tools
from wums import logging, output_tools


# harmonic polynomials
def p0(theta, phi):
    return 1.0 + np.cos(theta) ** 2


def p1(theta, phi):
    return 0.5 * (1.0 - 3.0 * np.cos(theta) ** 2)


def p2(theta, phi):
    return np.sin(2 * theta) * np.cos(phi)


def p3(theta, phi):
    return 0.5 * np.sin(theta) ** 2 * np.cos(2 * phi)


def p4(theta, phi):
    return np.sin(theta) * np.cos(phi)


def p5(theta, phi):
    return np.cos(theta)


def p6(theta, phi):
    return np.sin(theta) ** 2 * np.sin(2 * phi)


def p7(theta, phi):
    return np.sin(2 * theta) * np.sin(phi)


def p8(theta, phi):
    return np.sin(theta) * np.sin(phi)


def plot_harmonic_polynomials(outdir, args):
    colors = mpl.colormaps["tab10"]
    linestyles = ["solid", "dotted", "dashed", "dashdot"]

    npoints = 100

    theta_edges = np.linspace(-np.pi, 0, npoints + 1)
    cosTheta_edges = np.cos(theta_edges)
    theta = theta_edges[:-1] + np.diff(theta_edges) / 2.0
    cosTheta = cosTheta_edges[:-1] + np.diff(cosTheta_edges) / 2.0

    phi = np.linspace(-np.pi, np.pi, npoints)
    x, y = np.meshgrid(theta, phi)

    histos = []
    for pi in [p0, p1, p2, p3, p4, p5, p6, p7, p8]:
        histo = hist.Hist(
            hist.axis.Variable(
                cosTheta_edges, name="cosTheta", underflow=False, overflow=False
            ),
            hist.axis.Regular(npoints, -math.pi, math.pi, circular=True, name="phi"),
            storage=hist.storage.Double(),
        )
        histo.values()[...] = pi(x, y).T
        histos.append(histo)

    # 2D plots
    xlabel = styles.xlabels.get("costhetastarll", "cosTheta")
    ylabel = styles.xlabels.get("phistarll", "phi")
    for i, histo in enumerate(histos):
        fig = plot_tools.makeHistPlot2D(
            histo,
            cms_label=args.cmsDecor,
            xlabel=xlabel,
            ylabel=ylabel,
            scaleleg=args.scaleleg,
            has_data=False,
        )

        if i == 0:
            idx = "UL"
        else:
            idx = str(i - 1)

        outfile = "polynomial"
        outfile += f"_{idx}_cosTheta_phi"
        if args.postfix:
            outfile += f"_{args.postfix}"
        plot_tools.save_pdf_and_png(outdir, outfile)
        plot_tools.write_index_and_log(outdir, outfile, args=args)

    # 1D plots
    for axis_name, ais in [("cosTheta", [0, 1, 5]), ("phi", [3, 4, 6, 8])]:
        h1ds = [
            histo.project(axis_name)
            / np.prod([histo.axes[n].size for n in histo.axes.name if n != axis_name])
            for histo in histos
        ]

        fig, ax1 = plot_tools.figure(
            h1ds[0],
            xlabel=styles.xlabels.get(f"{axis_name.lower()}starll", axis_name),
            ylabel="Frequency",
            grid=True,
            automatic_scale=False,
            width_scale=1.2,
            logy=False,
        )

        j = 0
        for i, h1d in enumerate(h1ds):
            if i not in ais:
                continue

            val_x = h1d.axes[0].centers
            val_y = h1d.values()
            if i == 0:
                idx = r"\mathrm{UL}"
            else:
                idx = str(i - 1)
            label = rf"$\mathrm{{P}}_{idx}$"

            ax1.plot(
                val_x, val_y, color=colors(i), linestyle=linestyles[j], label=label
            )
            j += 1

        plot_tools.add_cms_decor(
            ax1, args.cmsDecor, data=False, lumi=None, loc=args.logoPos
        )
        plot_tools.addLegend(
            ax1, ncols=args.legCols, loc=args.legPos, text_size=args.legSize
        )

        outfile = "harmonic_polynomial"
        outfile += f"_{axis_name}"
        if args.postfix:
            outfile += f"_{args.postfix}"
        plot_tools.save_pdf_and_png(outdir, outfile)
        plot_tools.write_index_and_log(outdir, outfile, args=args)


if __name__ == "__main__":
    parser = parsing.plot_parser()
    args = parser.parse_args()
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)

    plot_harmonic_polynomials(outdir, args)

    if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
        output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)
