import matplotlib as mpl
import mplhep as hep
import numpy as np
import pandas as pd

import rabbit.io_tools
from utilities import parsing
from wremnants import plot_tools
from wums import logging, output_tools


def get_mass_obs(filename):
    fitresult = rabbit.io_tools.get_fitresult(filename)

    val = 100 * fitresult["nois_outvals"][...][0]
    err = 100 * (fitresult["nois_outcov"][...][0, 0] ** 0.5)

    return val, err


hep.style.use(hep.style.ROOT)

parser = parsing.plot_parser()
parser.add_argument("inputs", nargs="+", type=str, help="Paths to fitresult files")
args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)
outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)

df = pd.DataFrame(args.inputs, columns=["path"])

df["base_name"] = df["path"].apply(lambda x: x.split("/")[-2])
df["project"] = df["path"].apply(lambda x: x.split("/")[-3])
df["analysis"] = df["base_name"].apply(lambda x: x.split("_")[0])
# df["postfix"] = df["base_name"].apply(lambda x: x.split("_")[-2] )
df["postfix"] = df["project"]
df["mass"] = df["base_name"].apply(
    lambda x: int(x.split("_")[-1].split("MeV")[0].replace("massShift", "")[1:])
    * (-1 if x.endswith("Down") else 1)
)
df[["mass_obs", "mass_err"]] = df["path"].apply(get_mass_obs).apply(pd.Series)

### make plot

pulls = False
diffs = True

cm = mpl.colormaps["gist_rainbow"]

legend_labels = {
    "statOnly": "Stat. only",
    "mZUnc0MeV": r"$\Delta m^\mathrm{Z}=0\mathrm{MeV}$",
    "mZUnc100MeV": r"$\Delta m^\mathrm{Z}=100\mathrm{MeV}$",
    "mZUncFloat": r"$\Delta m^\mathrm{Z}=\inf$",
    # "mtCut0": r"$ m^\mathrm{Z}_\mathrm{T}>0$",
    # "mtCut0": r"$ m^\mathrm{Z}_\mathrm{T}>0$",
}

# for project, df_ana in df_ana.groupby("project"):
#     logger.info(f"Make plot for {project}")

for analysis, df_ana in df.groupby("analysis"):
    logger.info(f"Make plot for {analysis}")

    fig, ax1, ax2 = None, None, None

    # make axis ranges from all mass values
    xarr = df_ana["mass"].values
    # extend x-axis range by 2%
    xlim = (min(xarr) - 10, max(xarr) + 10)
    xrange = xlim[1] - xlim[0]
    xlim = xlim[0] - xrange * 0.005, xlim[1] + xrange * 0.005

    # extend y-axis range by 2%
    yarr = df_ana["mass_obs"].values
    yerr = df_ana["mass_err"].values
    ylim = min(yarr - yerr), max(yarr + yerr)
    yrange = ylim[1] - ylim[0]
    ylim = ylim[0] - yrange * 0.02, ylim[1] + yrange * 0.02

    # extend ratio range by 20%
    if pulls:
        rlabel = "Pulls"
        rarr = (xarr - yarr) / yerr
        rerr = np.zeros(len(yarr))
    elif diffs:
        rlabel = "Diff."
        rarr = yarr - xarr
        rerr = yerr
    rrange = min(rarr - rerr), max(rarr + rerr)
    rw = rrange[1] - rrange[0]
    rrange = rrange[0] - rw * 0.1, rrange[1] + rw * 0.1

    ntests = len(set(df_ana["postfix"].values))

    for i, (test, df_s) in enumerate(df_ana.groupby("postfix")):
        logger.info(f"Add points for {test} into plot")

        df_s = df_s.sort_values("mass")

        xarr = df_s["mass"].values
        yarr = df_s["mass_obs"].values
        yerr = df_s["mass_err"].values

        if pulls:
            rarr = (xarr - yarr) / yerr
        elif diffs:
            rarr = yarr - xarr
            rerr = yerr

        if fig is None:
            if analysis == "WMass":
                xlabel = "True W mass"
                ylabel = "Measured W mass"
            else:
                xlabel = "True Z mass"
                ylabel = "Measured Z mass"

            fig, ax1, ratio_axes = plot_tools.figureWithRatio(
                xarr, xlabel, ylabel, ylim, rlabel, rrange, xlim=xlim, width_scale=1
            )
            ax2 = ratio_axes[-1]

            ax1.plot(
                [xlim[0], xlim[-1]],
                [xlim[0], xlim[-1]],
                linestyle="--",
                color="grey",
                label="Expectation",
            )
            ax2.plot(
                [xlim[0], xlim[-1]],
                [0, 0],
                marker=".",
                linestyle="--",
                color="grey",
                label="Expectation",
            )

        color = cm((ntests - i) / (ntests))

        # slightly move different test sets so that all can be shown in the same figure
        xarr = xarr + 2 * (i - (ntests - 1) / 2.0)

        ax1.errorbar(
            xarr,
            yarr,
            yerr=yerr,
            marker=".",
            color=color,
            linestyle="",
            label=" ".join(
                sorted(set([legend_labels.get(t, "") for t in test.split("_")]))
            ),
        )

        if pulls:
            ax2.plot(xarr, rarr, linestyle="", marker=".", color=color)
        elif diffs:
            ax2.errorbar(xarr, rarr, yerr=rerr, linestyle="", marker=".", color=color)

    plot_tools.add_cms_decor(ax1, args.cmsDecor, data=True, lumi=None, loc=args.logoPos)
    plot_tools.addLegend(
        ax1, ncols=args.legCols, loc=args.legPos, text_size=args.legSize
    )

    outfile = f"massbias_{analysis}"
    if args.postfix:
        outfile += f"_{args.postfix}"
    plot_tools.save_pdf_and_png(outdir, outfile)
    plot_tools.write_index_and_log(outdir, outfile, args=args)
