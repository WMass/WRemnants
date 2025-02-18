import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from utilities.io_tools import input_tools
from wums import logging

mpl.rcParams.update(
    {
        "legend.fontsize": "medium",
        "legend.frameon": False,
        "legend.handletextpad": 0.1,
        "legend.columnspacing": 0.8,
        "axes.labelsize": "medium",
        "axes.titlesize": "medium",
        "xtick.labelsize": "medium",
        "ytick.labelsize": "medium",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    }
)

logger = logging.setup_logger(__file__, 3, True)

# Plot the shapes from the histmaker and theoy corrections input file

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    required=True,
    help="Input. Output files from the histmaker in .hdf5 format (E.g. from scripts/histmakers/w_z_gen_dists.py",
)
# parser.add_argument("--theory", required=True, help="Input from theory corrections in .pkl.lz4 format (E.g. from wremnants/data/TheoryCorrections/)")
parser.add_argument(
    "-d",
    "--datasets",
    default=[
        "Z",
    ],
    nargs="+",
    help="the datasets for which the plot should be made",
)
parser.add_argument(
    "-c", "--channels", default=[""], nargs="+", help="the channels to plot"
)
parser.add_argument(
    "--axes",
    default=["absVY", "VPT"],
    nargs="+",
    help="project the histogram on one axes",
)
parser.add_argument(
    "-s",
    "--systematics",
    default=[
        "MSHT20",
    ],
    nargs="+",
    help="Systematic variations to plot;",
)
parser.add_argument(
    "-p",
    "--outpath",
    default=os.path.expanduser("~/www/WMassAnalysis"),
    type=str,
    help="Where to store the plots",
)
parser.add_argument(
    "-f", "--outfolder", type=str, default="test", help="Subfolder for output"
)

args = parser.parse_args()

output_folder = f"{args.outpath}/{args.outfolder}"

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

translate_process = {"Z": "ZmumuPostVFP", "W": "WmunuPostVFP"}

translate_systematic_theory = {
    "MSHT20as": "MSHT20as",
}

translate_systematic_minnlo = {"MSHT20as": "MSHT20alphaS002"}

translate_theory = {"absVY": "absY", "VPT": "qT"}

translate_minnlo = {"absVY": "y", "VPT": "ptVgen"}

for dataset in args.datasets:
    logger.info(f"Now at {dataset}")

    for channel in args.channels:
        logger.info(f"Now at {channel}")

        for systematic in args.systematics:
            logger.info(f"Now at {systematic}")

            nominal = input_tools.read_and_scale(
                args.input,
                translate_process[dataset],
                f"nominal_gen_pdfMSHT20",
                calculate_lumi=False,
                scale=1,
            )
            minnlo = input_tools.read_and_scale(
                args.input,
                translate_process[dataset],
                f"nominal_gen_pdf{translate_systematic_minnlo.get(systematic,systematic)}",
                calculate_lumi=False,
                scale=1,
            )
            theory = input_tools.read_and_scale(
                args.input,
                translate_process[dataset],
                f"nominal_gen_scetlib_dyturbo{translate_systematic_theory.get(systematic,systematic)}VarsCorr",
                calculate_lumi=False,
                scale=1,
            )

            var_name_minnlo = "alphasVar" if systematic == "MSHT20as" else "pdfVar"
            var_name_theory = "vars"

            for axis in args.axes:

                if axis != None:
                    a_minnlo = translate_minnlo[axis]
                    # a_theory = translate_theory[axis]

                    x = nominal[{"pdfVar": 0}].project(a_minnlo).axes[0].centers

                    var_names = [
                        str(x)
                        for x in range(
                            1, int((len(minnlo.axes[var_name_minnlo]) - 1) / 2) + 1
                        )
                    ]

                    # minnlo:
                    nom = nominal[{"pdfVar": 0}].project(a_minnlo).values()
                    std = (
                        nominal[{"pdfVar": 0}].project(a_minnlo).variances() ** 0.5
                        / nom
                    )

                    if systematic == "MSHT20as":

                        var_names = [
                            "alphaS",
                        ]
                        var_minnlo = [
                            (
                                minnlo[{var_name_minnlo: "as0116"}]
                                .project(a_minnlo)
                                .values()
                                / nom,
                                minnlo[{var_name_minnlo: "as0120"}]
                                .project(a_minnlo)
                                .values()
                                / nom,
                            ),
                        ]
                    else:

                        var_minnlo = [
                            (
                                minnlo[{var_name_minnlo: 2 * i - 1}]
                                .project(a_minnlo)
                                .values()
                                / nom,
                                minnlo[{var_name_minnlo: 2 * i}]
                                .project(a_minnlo)
                                .values()
                                / nom,
                            )
                            for i in range(1, len(var_names) + 1)
                        ]

                    # theory:
                    nom = theory[{var_name_theory: 0}].project(a_minnlo).values()

                    if systematic == "MSHT20as":
                        var_theory = [
                            (
                                theory[{var_name_theory: "pdf2"}]
                                .project(a_minnlo)
                                .values()
                                / nom,
                                theory[{var_name_theory: "pdf5"}]
                                .project(a_minnlo)
                                .values()
                                / nom,
                            ),
                        ]
                    else:
                        var_theory = [
                            (
                                theory[{var_name_theory: 2 * i - 1}]
                                .project(a_minnlo)
                                .values()
                                / nom,
                                theory[{var_name_theory: 2 * i}]
                                .project(a_minnlo)
                                .values()
                                / nom,
                            )
                            for i in range(1, len(var_names) + 1)
                        ]

                for t, m, n in zip(var_theory, var_minnlo, var_names):
                    # make the plot for each variation
                    plt.clf()
                    fig = plt.figure()
                    fig.subplots_adjust(left=0.15, right=0.99, top=0.99, bottom=0.125)
                    ax = fig.add_subplot(111)

                    ax.plot([min(x), max(x)], [1, 1], linestyle="-", color="grey")
                    ax.fill_between(
                        x,
                        1 + std,
                        1 - std,
                        color="grey",
                        alpha=0.2,
                        zorder=1,
                        label="MC stat.",
                    )

                    ax.plot(
                        x,
                        m[0],
                        linestyle="-",
                        drawstyle="steps-mid",
                        color="red",
                        label="MiNNLO Up",
                    )
                    ax.plot(
                        x,
                        m[1],
                        linestyle="-",
                        drawstyle="steps-mid",
                        color="blue",
                        label="MiNNLO Down",
                    )

                    ax.plot(
                        x,
                        t[0],
                        linestyle=":",
                        drawstyle="steps-mid",
                        color="red",
                        label="Scetlib+DYTurbo Up",
                    )
                    ax.plot(
                        x,
                        t[1],
                        linestyle=":",
                        drawstyle="steps-mid",
                        color="blue",
                        label="Scetlib+DYTurbo Down",
                    )

                    ax.set_xlabel(axis)
                    ax.set_ylabel(f"{n}/nominal")

                    yMin = min(min(min(m[0]), min(m[1])), min(min(t[0]), min(t[1])))
                    yMax = max(max(max(m[0]), max(m[1])), max(max(t[0]), max(t[1])))

                    yRange = yMax - yMin
                    yMin = yMin - yRange * 0.02
                    yMax = yMax + yRange * 0.02
                    ax.set_ylim(yMin, yMax)
                    if axis == "absVY":
                        ax.set_xlim(0, 4.6)

                    ax.legend()

                    axisname = axis.replace("-", "_")

                    outname = f"{output_folder}/variation_{axisname}"
                    if channel:
                        outname += f"_{channel}"
                    outname += f"_{dataset}_{systematic}_{n}"

                    plt.savefig(f"{outname}.png")
                    plt.savefig(f"{outname}.pdf")
                    plt.close()
