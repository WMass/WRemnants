import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker

from utilities import parsing
from utilities.io_tools import combinetf_input, output_tools
from wremnants import plot_tools

parser = parsing.plot_parser()
parser.add_argument(
    "-i", "--fitresult", type=str, required=True, help="fitresults file from combinetf"
)
parser.add_argument("--pdg", action="store_true", help="Don't show PDG value")
args = parser.parse_args()

dfw = pd.DataFrame.from_dict(
    {
        "Name": [
            "Electroweak fit",
            "LEP combination",
            "D0",
            "CDF",
            "LHCb",
            "ATLAS",
            "PDG Average",
        ],
        "value": [80353, 80376, 80375, 80433.5, 80354, 80366.5, 80369.2],
        "err_total": [6, 33, 23, 9.4, 32, 15.9, 13.3],
        "err_stat": [6, 25, 11, 6.4, 23, 9.8, 13.3],
        "Reference": [
            # "Phys. Rev. D 110, 030001",
            "PRD 110 (2024) 030001",
            "Phys. Rep. 532 (2013) 119",
            # "Phys. Rev. Lett. 108 (2012) 151804",
            "PRL 108 (2012) 151804",
            "Science 376 (2022) 6589",
            "JHEP 01 (2022) 036",
            "arXiv:2403.15085",
            # "EPJC 84 (2024) 5, 451",
            "Phys. Rev. D 110, 030001",
        ],
        "color": ["#666666"] + ["black"] * 5 + ["navy"],
    }
)

if not args.pdg:
    dfw = dfw[dfw["Name"] != "PDG Average"]

cms_res = combinetf_input.read_groupunc_df(
    args.fitresult,
    [
        "stat",
    ],
    name="CMS",
)
cms_res["color"] = "#E42536"
cms_res["Reference"] = "This work"
dfw_cms = pd.concat((dfw, cms_res), ignore_index=True)

eoscp = output_tools.is_eosuser_path(args.outpath)
outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=eoscp)

nentries = len(dfw_cms)
xpos = 80148 + args.pdg * 10
top = nentries  # +0.5
# step = (top+0.25)/nentries
step = top / nentries
ymax = top + 1.2
legtop = top - 0.73


text_size = 15  #
fig = plot_tools.make_summary_plot(
    80353,
    6,
    None,  # Don't plot stat error separately
    r"80353 $\pm$ 6",
    dfw_cms[["Name", "value", "err_total"]].iloc[1:, :],
    center_color="#666666",
    colors=list(dfw_cms["color"][1:]),
    xlim=[80255, 80465],
    ylim=[0, ymax],
    xlabel=r"$\mathit{m}_{W}$ (MeV)",
    capsize=6,
    width_scale=1.25,
    cms_label=args.cmsDecor,
    padding=4,
    point_size=0.24,
    top_offset=0,
    # bottom_offset=offset*2,
    label_points=False,
    legend_loc="lower left",
    bbox_to_anchor=(xpos + 105, legtop),
    legtext_size=text_size,
    logoPos=args.logoPos,
    lumi=None,
)

ax = plt.gca()

text_size_large = plot_tools.get_textsize(ax, "small")
ax.annotate(
    r"$\mathit{m}_{{W}}$ in MeV",
    (80265, top + 0.5),
    fontsize=text_size,
    ha="left",
    color="black",
    annotation_clip=False,
)
for i, row in dfw_cms.iterrows():
    isCMS = row.loc["Name"] == "CMS"
    isEW = row.loc["Name"] == "Electroweak fit"
    pos = top - step * i
    ax.annotate(
        row["Name"],
        (xpos, pos),
        fontsize=text_size_large,
        ha="left",
        annotation_clip=False,
        color=row.loc["color"],
    )  # , weight=600)
    if row.loc["Name"] in ["CMS", "CDF", "ATLAS", "PDG Average"]:
        label = rf"{row.loc['value']:.1f} $\pm$ {round(row.loc['err_total'], 1):.1f}"
    else:
        label = rf"{row.loc['value']:.0f} $\pm$ {round(row.loc['err_total'], 0):.0f}"

    if not isEW:
        ax.annotate(
            label,
            (80260, pos),
            fontsize=text_size,
            ha="left",
            va="center",
            color=row.loc["color"] if isCMS or isEW else "black",
            annotation_clip=False,
        )
    ax.annotate(
        row["Reference"],
        (xpos, pos - 0.42),
        fontsize=text_size,
        ha="left",
        color="#666666",
        annotation_clip=False,
        style="italic" if isCMS else None,
    )

# ewlabel = "80355 $\pm$ 6"
# ax.annotate(ewlabel, (80265, 0), fontsize=text_size, ha="left", va="center", color="navy", annotation_clip=False)
# ax.annotate("Phys. Rev. D 110, 030001", (xpos, legtop-0.3), fontsize=text_size, ha="left", color="gray", annotation_clip=False)

ax.set_yticks(range(nentries))
ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(25))
ax.xaxis.grid(False, which="both")
ax.yaxis.grid(False, which="both")
ax.set_yticks([])

name = "resultsSummary"
if args.postfix:
    name += args.postfix
if args.cmsDecor == "Preliminary":
    name += "_preliminary"

plot_tools.save_pdf_and_png(outdir, name, fig)
plot_tools.write_index_and_log(outdir, name)
if eoscp:
    output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)
