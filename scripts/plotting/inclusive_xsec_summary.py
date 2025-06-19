# On results of fiducial inclusive cross sections and their ratios
# make a summary plot with different theory predictions
# make a latex summary table with the breakdown of uncertainties

import json
import math

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd

import rabbit.io_tools
from utilities import parsing
from utilities.io_tools import conversion_tools, output_tools, tex_tools
from utilities.styles.styles import nuisance_groupings
from wremnants import plot_tools
from wums import ioutils, logging

parser = parsing.plot_parser()
parser.add_argument("infile", type=str, help="Combine fitresult file")
parser.add_argument(
    "-t",
    "--translate",
    type=str,
    default=None,
    help="Specify .json file to translate labels",
)

args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

if args.infile.endswith(".root"):
    args.infile = args.infile.replace(".root", ".hdf5")

grouping = nuisance_groupings["max"]

fitresult = rabbit.io_tools.get_fitresult(args.infile)
meta = ioutils.pickle_load_h5py(fitresult["meta"])

translate_label = {}
if args.translate:
    with open(args.translate) as f:
        translate_label = json.load(f)

pois = [
    "W_qGen0_sumxsec",
    "W_qGen1_sumxsec",
    "W_sumxsec",
    "Z_sumxsec",
    "r_qGen_W__ratiometaratio",
    "r_WZ_ratiometaratio",
]
poi_names = [
    r"$\mathrm{W}^{-}$",
    r"$\mathrm{W}^{+}$",
    r"$\mathrm{W}$",
    r"$\mathrm{Z}$",
    r"$\mathrm{W}^{+}/\mathrm{W}^{-}$",
    r"$\mathrm{W/Z}$",
]

pois = [
    "W_qGen0_sumxsec",
    "W_qGen1_sumxsec",
    "W_sumxsec",
    "r_qGen_W_ratiometaratio",
]
poi_names = [
    r"$\mathrm{W}^{-}$",
    r"$\mathrm{W}^{+}$",
    r"$\mathrm{W}$",
    r"$\mathrm{W}^{+}/\mathrm{W}^{-}$",
]


combine = {
    "binByBinStat": "binByBinStat",
    "statMC": "binByBinStat",
    "ZmassAndWidth": "massAndWidth",
    "massShift": "massAndWidth",
    "QCDscale": "QCDscale",
    "bcQuarkMass": "QCDscale",
}
combine = {}

dfs = []
for poi, poi_name in zip(pois, poi_names):

    if poi.endswith("sumxsec"):
        channel_info = conversion_tools.combine_channels(meta, True)
        if len(channel_info.keys()) == 1:
            lumi = channel_info["chan_13TeV"]["lumi"]
        else:
            raise NotImplementedError(
                f"Found channels {[k for k in channel_info.keys()]} but only one channel is supported."
            )
        scale = 1.0 / (lumi * 1000)
    elif poi.endswith("ratio"):
        scale = 1.0

    impacts, labels, norm = rabbit.io_tools.read_impacts_poi(
        fitresult, True, add_total=True, stat=0.0, poi=poi, normalize=False
    )

    filtimpacts = []
    filtlabels = []
    for impact, label in zip(impacts, labels):
        if label not in grouping:
            continue
        if label in combine:
            label = combine[label]
            if label in filtlabels:
                idx = filtlabels.index(label)
                filtimpacts[idx] = (filtimpacts[idx] ** 2 + impact**2) ** 0.5
                continue

        filtimpacts.append(impact)
        filtlabels.append(label)

    impacts = filtimpacts
    labels = filtlabels

    df = pd.DataFrame(np.array(impacts, dtype=np.float64).T * scale, columns=["impact"])

    df["label"] = labels
    df["systematic"] = df["label"].apply(lambda l: translate_label.get(l, l))
    df["poi_name"] = poi_name
    df["norm"] = norm * scale

    dfs.append(df)

df = pd.concat(dfs)

outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)

# make latex table
relative = True  # compute relative uncertainty
percentage = True  # numbers in percentage

df_t = df.copy()
if relative:
    df_t["impact"] /= df_t["norm"]
if percentage:
    df_t["impact"] *= 100

# sorting
cat_dtype = pd.CategoricalDtype(categories=poi_names, ordered=True)
df_t["poi_name"] = df_t["poi_name"].astype(cat_dtype)


outname = "summary_table"
if args.postfix:
    outname += f"_{args.postfix}"
tex_tools.make_latex_table(
    df_t,
    output_dir=outdir,
    output_name=outname,
    column_title=None,
    caption="Uncertainties in percentage.",
    label="",
    sublabel="",
    column_name="poi_name",
    row_name="systematic",
    cell_columns=["impact"],
    cell_format=lambda x: f"${round(x,2)}$",
    sort="impact",
)

# make plot
hep.style.use(hep.style.ROOT)


plt.clf()
fig = plt.figure()
fig.subplots_adjust(left=0.15, right=0.99, top=0.99, bottom=0.125)
ax = fig.add_subplot(111)

# x axis range
lo, hi = 0.94, 1.09

# totals = []
# stats = []
norms = []
for i, poi_name in enumerate(poi_names[::-1]):
    df_g = df.loc[df["poi_name"] == poi_name]

    norm = df_g["norm"].values[0]
    total = df_g.loc[df_g["label"] == "Total"]["impact"].values[0]
    stat = df_g.loc[df_g["label"] == "stat"]["impact"].values[0]
    total_rel = total / norm
    stat_rel = stat / norm

    norms.append(norm)
    # totals.append(total)
    # stats.append(stat)

    x1 = ax.bar(
        1.0, height=1, bottom=i, width=2 * total_rel, color="silver", label="Total"
    )
    x2 = ax.bar(1.0, height=1, bottom=i, width=2 * stat_rel, color="gold", label="Stat")

    # round to two significant digits in total uncertainty
    sig_digi = 2 - int(math.floor(math.log10(abs(total)))) - 1

    if sig_digi <= 0:
        norm = int(norm)
        total = int(total)
    else:
        norm = round(norm, sig_digi)
        total = round(total, sig_digi)

    ax.text(
        lo + 0.01,
        i + 0.5,
        poi_name,
        fontsize=20,
        verticalalignment="bottom",
        horizontalalignment="left",
    )
    title = rf"${norm} \pm {total}"
    if "/" in poi_name:
        title += "$"
    else:
        title += r"\,\mathrm{pb}$"
    ax.text(
        hi - 0.05,
        i + 0.5,
        title,
        fontsize=20,
        verticalalignment="bottom",
        horizontalalignment="left",
    )

ax.text(
    hi - 0.05,
    len(poi_names) + 0.5,
    r"$\mathrm{Measured} \pm {unc}$",
    fontsize=20,
    verticalalignment="bottom",
    horizontalalignment="left",
)

x0 = ax.plot([1.0, 1.0], [0, len(norms)], color="black")
ax.plot([lo, hi], [len(norms), len(norms)], color="black")


# make legend

# Custom legend handles
p2 = (
    mpatches.Rectangle((0, 0), 1, 1, facecolor="gold", edgecolor="gold", linewidth=3),
)
p1 = (
    mpatches.Rectangle(
        (0, 0), 2, 1, facecolor="silver", edgecolor="silver", linewidth=3
    ),
)

leg_styles = [(x2[0], x1[0], x0[0])]
leg_labels = ["Measurement"]
leg = ax.legend(
    leg_styles, leg_labels, loc="upper left", ncol=len(leg_labels), fontsize=20
)

ax.set_xlim([lo, hi])
ax.set_ylim([0, len(norms) + 1])

ax.set_xlabel("1./Measurement", fontsize=20)

# Disable ticks on the top and right axes
ax.tick_params(top=False)

# Disable y-axis labels and ticks
plt.gca().set_yticklabels([])
plt.gca().set_yticks([])

plot_tools.add_cms_decor(ax, args.cmsDecor, data=True, lumi=lumi, loc=args.logoPos)

outname = "summary"
if args.postfix:
    outname += f"_{args.postfix}"
plot_tools.save_pdf_and_png(outdir, outname)

plot_tools.write_index_and_log(
    outdir,
    outname,
    analysis_meta_info={"CombinetfOutput": meta["meta_info"]},
    args=args,
)

if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
    output_tools.copy_to_eos(args.outpath, args.outfolder)
