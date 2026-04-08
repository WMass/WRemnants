#!/usr/bin/env python3

import argparse

import mplhep as hep
import numpy as np

from wums import logging, output_tools, plot_tools

hep.style.use(hep.style.ROOT)


def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("xsecfile", type=str, help="File with mass, xsec")
    parser.add_argument("limitfile", type=str, help="File with mass, limits")
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        choices=["zprime", "hnl"],
        help="Model for comprison",
    )

    parser.add_argument("-o", "--outpath", type=str, default="./plots")
    parser.add_argument("--title", default="CMS", type=str)
    parser.add_argument("--subtitle", default="", type=str)
    parser.add_argument("--xlabel", default=r"$m_{Z'}$ [GeV]")
    parser.add_argument("--ylabel", default=r"$g_{Z'}$")
    parser.add_argument("--verbose", type=int, default=3)
    parser.add_argument("--noColorLogger", action="store_true")
    parser.add_argument(
        "--legPos", type=str, default="upper right", help="Set legend position"
    )
    parser.add_argument(
        "--legSize",
        type=str,
        default="small",
        help="Legend text size (small: axis ticks size, large: axis label size, number)",
    )
    parser.add_argument(
        "--legCols", type=int, default=2, help="Number of columns in legend"
    )
    parser.add_argument(
        "-p", "--postfix", type=str, help="Postfix for output file name"
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        help="Min and max values for y axis (if not specified, range set automatically)",
    )
    parser.add_argument(
        "--upperLimit",
        type=float,
        nargs="+",
        help="Upper limit of measurements",
    )
    return parser.parse_args()


def main():
    args = parseArgs()
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    outdir = output_tools.make_plot_dir(args.outpath)

    # --- load data ------------------------------------------------------------
    mass, xsec = np.loadtxt(args.xsecfile, unpack=True)
    mass2, limits = np.loadtxt(args.limitfile, unpack=True)

    # expected limits from xsec
    expected = np.array(args.upperLimit) / (xsec * 100)
    if args.model == "zprime":
        expected = np.sqrt(expected)

    # --- build figure via plot_tools -----------------------------------------
    fig, ax = plot_tools.figure(
        None,
        args.xlabel,
        args.ylabel,
        automatic_scale=False,
        xlim=(0, max(mass.max(), mass2.max()) * 1.05),
        ylim=args.ylim,
        logy=True,
    )

    if args.model == "hnl":
        first = 17
        ax.plot(
            mass,
            expected,
            color="black",
            marker="",
            linestyle="-",
            label="This analysis (expected)",
        )
        ax.plot(
            mass2[:first],
            limits[:first],
            color="brown",
            marker="",
            linestyle="-",
            label="Prompt $3\ell = (e, \mu, \tau)$ \n arXiv:2403.00100",
        )
        ax.plot(
            mass2[first - 1 :],
            limits[first - 1 :],
            color="purple",
            marker="",
            linestyle="-",
            label="Prompt 1 + 1 displaced & jet \n CMS-PAS-EXO-21-011",
        )
    elif args.model == "zprime":
        ax.plot(
            mass,
            expected,
            color="black",
            marker="",
            linestyle="-",
            label="Limit from $\sigma=100pb$",
        )
        ax.plot(
            mass2,
            limits,
            color="brown",
            marker="",
            linestyle="-",
            label="Phys.Rev.D 110 (2024) 072008, 2024.",
        )

    plot_tools.add_decor(
        ax,
        args.title,
        args.subtitle,
        data=True,
        lumi=None,
        loc=2,
    )

    plot_tools.addLegend(
        ax,
        ncols=args.legCols,
        loc=args.legPos,
        text_size=args.legSize,
    )

    to_join = ["limits_comparison", args.postfix]
    outfile = "_".join(filter(lambda x: x, to_join))
    if args.subtitle == "Preliminary":
        outfile += "_preliminary"

    plot_tools.save_pdf_and_png(outdir, outfile)

    # write index + log
    output_tools.write_index_and_log(
        outdir,
        outfile,
        analysis_meta_info={
            "xsecfile": args.xsecfile,
            "limitfile": args.limitfile,
        },
        args=args,
    )


if __name__ == "__main__":
    main()
