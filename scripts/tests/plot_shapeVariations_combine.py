import argparse
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import uproot

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

# Plot the shapes from the combine input file as a ratio to the nominal histogram

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--inputFile", required=True, help="combine input root file with histograms"
)
parser.add_argument(
    "-d",
    "--datasets",
    default=[
        "Wmunu",
    ],
    nargs="+",
    help="the datasets for which the plot should be made",
)
parser.add_argument(
    "-c",
    "--channels",
    default=["plus", "minus"],
    nargs="+",
    help="the channels to plot",
)
parser.add_argument(
    "--projections",
    default=["pt", "eta", "eta-pt"],
    nargs="+",
    help="project the histogram on one axes",
)
parser.add_argument(
    "-s",
    "--systematics",
    default=[
        None,
    ],
    nargs="+",
    help="Systematic variations to plot",
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

translate_project = {
    "absYll": "xaxis",
    "ptll": "xaxis",
    "pt": "yaxis",
    "eta": "xaxis",
    "eta-pt": None,
}

fin = uproot.open(args.inputFile)

for dataset in args.datasets:
    logger.info(f"Now at {dataset}")
    for channel in args.channels:
        logger.info(f"Now at {channel}")

        keys = [k.replace(";1", "") for k in fin.keys()]

        keys_skimmed = [
            x
            for x in filter(
                lambda x: x.startswith(f"x_{dataset}_") and x.endswith(f"_{channel}"),
                keys,
            )
        ]

        for systematic in args.systematics:
            logger.info(f"Now at {systematic}")

            if systematic is not None:
                systs = [
                    x
                    for x in filter(
                        lambda x: re.search(systematic, x) is not None, keys_skimmed
                    )
                ]
            else:
                systs = keys_skimmed

            if len(systs) == 0:
                logger.warning(f"No match for `{systematic}` was found")

            name_nominal = f"x_{dataset}_{channel};1"

            nominal = fin[name_nominal].to_hist()

            variations = []
            var_names = []
            for systUp, systDown in zip(
                filter(lambda x: f"Up_{channel}" in x, systs),
                filter(lambda x: f"Down_{channel}" in x, systs),
            ):

                variations.append((fin[systUp].to_hist(), fin[systDown].to_hist()))
                var_names.append(systUp.split(f"{dataset}_")[-1].split("Up")[0])

            for axis in args.projections:
                proj = translate_project.get(axis, axis)

                if proj != None:
                    x = nominal.project(proj).axes[0].centers
                    nom = nominal.project(proj).values()
                    std = nominal.project(proj).variances() ** 0.5 / nom

                    var = [
                        (
                            v[0].project(proj).values() / nom,
                            v[1].project(proj).values() / nom,
                        )
                        for v in variations
                    ]

                else:
                    nom = np.ravel(nominal.values())
                    std = np.ravel(nominal.variances()) ** 0.5 / nom

                    var = [
                        (np.ravel(v[0].values()) / nom, np.ravel(v[1].values()) / nom)
                        for v in variations
                    ]

                    x = np.arange(len(nom))

                for v, n in zip(var, var_names):
                    # make the plot for each variation
                    plt.clf()
                    fig = plt.figure()
                    fig.subplots_adjust(left=0.15, right=0.99, top=0.97, bottom=0.125)
                    ax = fig.add_subplot(111)

                    ax.plot([min(x), max(x)], [0, 0], linestyle="-", color="grey")
                    ax.fill_between(
                        x,
                        std,
                        -std,
                        color="grey",
                        alpha=0.2,
                        zorder=1,
                        label="MC stat.",
                    )

                    ax.plot(
                        x,
                        1 - v[0],
                        linestyle="-",
                        drawstyle="steps-mid",
                        color="red",
                        label="Up",
                    )
                    ax.plot(
                        x,
                        1 - v[1],
                        linestyle="--",
                        drawstyle="steps-mid",
                        color="blue",
                        label="Down",
                    )

                    ax.set_xlabel(axis)
                    ax.set_ylabel(f"1 - {n}/nominal")

                    yMin = min(min(1 - v[0]), min(1 - v[1]))
                    yMax = max(max(1 - v[0]), max(1 - v[1]))

                    yRange = yMax - yMin
                    yMin = yMin - yRange * 0.02
                    yMax = yMax + yRange * 0.02
                    ax.set_ylim(yMin, yMax)
                    if axis == "absVY":
                        ax.set_xlim(0, 4.6)

                    # ax.ticklabel_format(useOffset=False)
                    # ax.ticklabel_format(style='plain')

                    ax.legend()

                    axisname = axis.replace("-", "_")
                    plt.savefig(
                        f"{output_folder}/variation_{axisname}_{channel}_{dataset}_{n}.png"
                    )
                    plt.savefig(
                        f"{output_folder}/variation_{axisname}_{channel}_{dataset}_{n}.pdf"
                    )
                    plt.close()
