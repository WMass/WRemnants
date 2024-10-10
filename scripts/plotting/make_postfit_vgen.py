import pickle

import hist
import numpy as np

from utilities import boostHistHelpers as hh
from utilities import common
from utilities.io_tools import combinetf2_input, input_tools, output_tools
from wremnants import plot_tools, syst_tools

parser = common.plot_parser()
parser.add_argument("--unfolded", type=str, required=False)
parser.add_argument("--gen", type=str, default=None)
parser.add_argument("-w", action="store_true")
parser.add_argument("--etapt-fit", type=str, required=True)
parser.add_argument("--ptll-fit", type=str, default=None)
parser.add_argument("--ptll-yll-fit", type=str, default=None)
parser.add_argument("--obs", type=str, default="ptVgen")
parser.add_argument("--prefit", action="store_true")
parser.add_argument(
    "--noetapt-postfit", action="store_true", help="Only take prefit but not postfit"
)

args = parser.parse_args()

fittype = "prefit" if args.prefit else "postfit"


def hist_to_up_down_unc(h):
    hunc = h.copy()
    hunc.values(flow=True)[...] = np.sqrt(h.variances(flow=True))
    hup = hh.addHists(h, hunc)
    hdown = hh.addHists(h, hunc, scale2=-1)
    return hup, hdown


def quadrature_sum_hist(hists, is_down):
    sumh = hist.Hist(
        *hists[0].axes,
        hist.axis.Integer(0, len(hists), name="vars"),
        data=np.stack(hists, axis=-1),
    )
    return hh.rssHists(sumh, syst_axis="vars")[is_down]


hnom = "nominal_gen"

unfolded_data = pickle.load(open(args.unfolded, "rb")) if args.unfolded else None

procs = ["WplusmunuPostVFP", "WminusmunuPostVFP"] if args.w else ["ZmumuPostVFP"]

etapt_fit = combinetf2_input.get_fitresult(args.etapt_fit)

if args.gen:
    # only use subset of nuisances for prefit band
    gen = input_tools.read_all_and_scale(args.gen, procs, [hnom])[0]

    scetlib_dyturbo = input_tools.read_all_and_scale(
        args.gen, procs, [f"{hnom}_scetlib_dyturboCorr"]
    )[0].project(args.obs, "vars")
    ct18z = input_tools.read_all_and_scale(args.gen, procs, [f"{hnom}_pdfCT18Z"])[
        0
    ].project(args.obs, "pdfVar")
    ct18z_as = input_tools.read_all_and_scale(
        args.gen, procs, [f"{hnom}_pdfCT18ZalphaS002"]
    )[0].project(args.obs, "alphasVar")

    # Leaving out mb and mc range since they're small...

    transforms = syst_tools.syst_transform_map(hnom, f"{hnom}_scetlib_dyturboCorr")

    theory_up = quadrature_sum_hist(
        [
            transforms["resumTNPXp0Up"]["action"](scetlib_dyturbo),
            transforms["resumNPUp"]["action"](scetlib_dyturbo),
            scetlib_dyturbo[{"vars": "renorm_scale_pt20_envelope_Up"}],
            scetlib_dyturbo[{"vars": "transition_points0.2_0.35_1.0"}],
            transforms["pdfCT18ZUp"]["action"](ct18z),
            ct18z_as[{"alphasVar": "as0120"}],
        ],
        is_down=False,
    )

    theory_down = quadrature_sum_hist(
        [
            transforms["resumTNPXp0Down"]["action"](scetlib_dyturbo),
            transforms["resumNPDown"]["action"](scetlib_dyturbo),
            scetlib_dyturbo[{"vars": "renorm_scale_pt20_envelope_Down"}],
            scetlib_dyturbo[{"vars": "transition_points0.2_0.75_1.0"}],
            transforms["pdfCT18ZDown"]["action"](ct18z),
            ct18z_as[{"alphasVar": "as0116"}],
        ],
        is_down=True,
    )
else:
    # use all nuisances for prefit band
    gen = etapt_fit[f"hist_prefit_inclusive"]["ch0"].get() / 1000.0
    theory_up, theory_down = hist_to_up_down_unc(gen)

hists_nom = [
    gen,
]
hists_err = [
    theory_up,
    theory_down,
]

labels = [
    r"prefit",
]
colors = [
    "gray",
]

if not args.noetapt_postfit:
    etapth = etapt_fit[f"hist_{fittype}_inclusive"]["ch0"].get() / 1000.0
    hists_nom.append(etapth)
    hists_err.extend(hist_to_up_down_unc(etapth))
    if args.w:
        label = r"$\mathit{m}_{W}$ "
    else:
        label = r"$\mathit{m}_{Z}$ "
    label += r"$(\mathit{p}_{T}^{\mu}, \mathit{\eta}^{\mu}, \mathit{q}^{\mu})$ "
    labels.append(label)
    colors.append("#E42536" if args.w else "#964A8B")

if unfolded_data:
    idx_unfolded = len(hists_nom)

    unfoldedh = unfolded_data["results"]["xsec"]["chan_13TeV"]["Z"][
        f"hist_{args.obs.replace('gen','Gen')}"
    ]

    # Thanks Obama (David)
    for ax in unfoldedh.axes:
        ax._ax.metadata["name"] = ax.name.replace("Gen", "gen")
    hists_nom.append(unfoldedh)
    labels.append("Unfolded asimov data" if args.prefit else "Unfolded data")
    colors.append("black")
else:
    idx_unfolded = None

if args.ptll_fit:
    ptll_fit = combinetf2_input.get_fitresult(args.ptll_fit)
    ptllh = ptll_fit[f"hist_{fittype}_inclusive"]["ch0"].get() / 1000.0
    hists_nom.append(ptllh)
    hists_err.extend(hist_to_up_down_unc(ptllh))
    if args.w:
        labels.append(
            r"$\mathit{m}_{W}$ $(\mathit{p}_{T}^{\mu}, \mathit{\eta}^{\mu}, \mathit{q}^{\mu})+\mathit{p}_{T}^{\mu\mu}$ "
        )
    else:
        labels.append(r"$\mathit{p}_{T}^{\mu\mu}$ ")
    colors.append("#f89c20")

if args.ptll_yll_fit:
    ptllyll_fit = combinetf2_input.get_fitresult(args.ptll_yll_fit)
    ptllyllh = ptllyll_fit[f"hist_{fittype}_inclusive"]["ch0"].get() / 1000.0
    hists_nom.append(ptllyllh)
    hists_err.extend(hist_to_up_down_unc(ptllyllh))
    if args.w:
        labels.append(
            r"$\mathit{m}_{W}$ $(\mathit{p}_{T}^{\mu}, \mathit{\eta}^{\mu}, \mathit{q}^{\mu})+(\mathit{p}_{T}^{\mu\mu},\mathit{y}^{\mu\mu})$ "
        )
    else:
        labels.append(r"$(\mathit{p}_{T}^{\mu\mu},\mathit{y}^{\mu\mu})$ ")
    colors.append("#5790FC")

linestyles = [
    "solid",
] * len(hists_nom)

colors = colors + [
    c for i, c in enumerate(colors) if i != idx_unfolded for _ in range(2)
]
labels = labels + [
    "" for i, l in enumerate(labels) if i != idx_unfolded for _ in range(2)
]
linestyles = linestyles + [
    l for l in ["dotted", "dashdot", "dashed", "dotted"] for _ in range(2)
]
linestyles = linestyles[: len(labels)]

if args.xlim:
    hists_nom = [
        x[complex(0, args.xlim[0]) : complex(0, args.xlim[1])] for x in hists_nom
    ]
    hists_err = [
        x[complex(0, args.xlim[0]) : complex(0, args.xlim[1])] for x in hists_err
    ]

hists = hists_nom + hists_err

xlabels = {"absYVgen": r"\mathit{y}", "ptVgen": r"\mathit{p}_{T}"}
xlabel = xlabels[args.obs]

if args.w:
    xlabel = r"$" + xlabel + "^{W}$"
    ylabel = r"$W"
else:
    xlabel = r"$" + xlabel + "^{Z}$"
    ylabel = r"$Z"
ylabel += r"\ cross\ section\ "

if args.obs in ["ptVgen"]:
    xlabel += " (GeV)"
    ylabel += r"(pb\,/\,GeV)$"
else:
    ylabel += r"(pb)$"

fig = plot_tools.makePlotWithRatioToRef(
    hists=hists_nom,
    hists_ratio=hists,
    labels=labels,
    colors=colors,
    linestyles=linestyles,
    xlabel=xlabel,
    ylabel=ylabel,
    rlabel="Ratio to prefit",
    rrange=args.rrange,
    nlegcols=args.legCols,
    lowerLegCols=args.lowerLegCols,
    lowerLegPos=args.lowerLegPos,
    yscale=args.yscale,
    ylim=args.ylim,
    xlim=None,
    binwnorm=1.0,
    baseline=True,
    yerr=False,
    fill_between=len(hists_err),
    cms_label=args.cmsDecor,
    legtext_size=args.legSize,
    dataIdx=idx_unfolded,
    width_scale=1.25,
)
eoscp = output_tools.is_eosuser_path(args.outpath)

outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=eoscp)

name = f"{args.obs}_{fittype}_{'W' if args.w else 'Wlike'}"
if not args.noetapt_postfit:
    name += "_RecoPtll"
if args.ptll_fit:
    name += "_ptll"
if args.ptll_yll_fit:
    name += "_ptllyll"
name += "_PrefitRatio"
if args.postfix:
    name += f"_{args.postfix}"
if args.cmsDecor == "Preliminary":
    name += "_preliminary"

plot_tools.save_pdf_and_png(outdir, name)
plot_tools.write_index_and_log(outdir, name)

if output_tools.is_eosuser_path(args.outpath) and eoscp:
    output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)
