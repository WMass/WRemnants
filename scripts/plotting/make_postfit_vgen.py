import pickle

import hist
import numpy as np
import pandas as pd

import narf
from utilities import boostHistHelpers as hh
from utilities import logging, parsing
from utilities.io_tools import combinetf2_input, input_tools, output_tools
from wremnants import plot_tools, syst_tools

parser = parsing.plot_parser()
parser.add_argument("--unfolded", type=str, required=False)
parser.add_argument("--gen", type=str, default=None)
parser.add_argument("-w", action="store_true")
parser.add_argument("--etapt-fit", type=str, default=None)
parser.add_argument("--ptll-fit", type=str, default=None)
parser.add_argument("--ptll-yll-fit", type=str, default=None)
parser.add_argument("--helicity-fit", type=str, default=None)
parser.add_argument("--obs", type=str, default="ptVgen")
parser.add_argument("--prefit", action="store_true")
parser.add_argument("--noPrefit", action="store_true")
parser.add_argument(
    "--noetapt-postfit", action="store_true", help="Only take prefit but not postfit"
)
parser.add_argument(
    "--ratioToData", action="store_true", help="Use data as denominator in ratio"
)
parser.add_argument("--twoRatios", action="store_true", help="Make two ratio panels")
parser.add_argument(
    "--saveForHepdata",
    action="store_true",
    help="Save histograms as ROOT to prepare HEPData",
)
parser = parsing.set_parser_attribute(parser, "rrange", "nargs", "*")

args = parser.parse_args()

logger = logging.setup_logger("make_postfit_vgen", args.verbose)

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


def load_hist(filename, fittype="postfit", helicity=False):
    fitresult = combinetf2_input.get_fitresult(filename)
    obs = {args.obs, "helicity"} if helicity else {args.obs}
    if "projections" in fitresult.keys() and len(fitresult["projections"]):
        fitresult = fitresult["projections"]
        idx = [i for (i, a) in enumerate(fitresult) if obs == set(a["axes"])][0]
        fitresult = fitresult[idx]
        h = fitresult[f"hist_{fittype}_inclusive"]
    else:
        h = fitresult[f"hist_{fittype}_inclusive"]["ch0"]

    return h.get() / 1000.0


hnom = "nominal_gen"

unfolded_data = pickle.load(open(args.unfolded, "rb")) if args.unfolded else None

procs = ["WplusmunuPostVFP", "WminusmunuPostVFP"] if args.w else ["ZmumuPostVFP"]

hists_nom = []
hists_err = []

labels = []
names = []
colors = []

if not args.prefit and not args.noPrefit:
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
        hists_nom.append(gen)
        hists_err.extend([theory_up, theory_down])
        labels.append("Prefit")
        names.append("Prefit")
        colors.append("gray")

    elif args.etapt_fit is not None:
        # use all nuisances for prefit band
        gen = load_hist(args.etapt_fit, "prefit")
        theory_up, theory_down = hist_to_up_down_unc(gen)
        hists_nom.append(gen)
        hists_err.extend([theory_up, theory_down])
        labels.append("Prefit")
        names.append("Prefit")
        colors.append("gray")

    elif args.helicity_fit is not None:
        gen = load_hist(args.helicity_fit, "prefit", helicity=True)[{"helicity": -1.0j}]

        theory_up, theory_down = hist_to_up_down_unc(gen)

        hists_nom.append(gen)
        hists_err.extend([theory_up, theory_down])
        labels.append("Helicity fit prefit")
        names.append("Helicity fit prefit")
        colors.append("gray")

if args.helicity_fit:
    helh = load_hist(args.helicity_fit, helicity=True)[{"helicity": -1.0j}]

    hists_nom.append(helh)
    hists_err.extend(hist_to_up_down_unc(helh))
    labels.append(f"Helicity fit {fittype}")
    names.append(f"{fittype} helicity")
    # colors.append("#5790FC")
    colors.append("#E42536")

if not args.noetapt_postfit:
    etapth = load_hist(args.etapt_fit)

    hists_nom.append(etapth)
    hists_err.extend(hist_to_up_down_unc(etapth))
    if args.w:
        label = r"$\mathit{m}_{W}$ "
    else:
        label = r"$\mathit{m}_{Z}$ "
    label += r"$(\mathit{p}_{T}^{\mu}, \mathit{\eta}^{\mu}, \mathit{q}^{\mu})$ "
    labels.append(label)
    names.append(f"{fittype} eta-pt")
    colors.append("#E42536" if args.w else "#964A8B")

if unfolded_data:
    idx_unfolded = len(hists_nom)

    unfoldedh_err = unfolded_data["results"]["xsec"]["chan_13TeV"]["Z"][
        f"hist_{args.obs.replace('gen','Gen')}"
    ]

    # fudging to disable pTV overflow bin |YV| distribution
    if args.obs == "absYVgen":
        unfoldedh = hh.disableFlow(
            unfolded_data["results"]["xsec"]["chan_13TeV"]["Z"]["hist_ptVGen_absYVGen"],
            "ptVGen",
        ).project("absYVGen")
        unfoldedh.variances()[...] = unfoldedh_err.variances()
    else:
        unfoldedh = unfoldedh_err

    # Thanks Obama (David)
    for ax in unfoldedh.axes:
        ax._ax.metadata["name"] = ax.name.replace("Gen", "gen")
    hists_nom.append(unfoldedh)
    labels.append("Unfolded asimov data" if args.prefit else "Unfolded data")
    names.append("Unfolded data")
    colors.append("black")
else:
    idx_unfolded = None

if args.ptll_fit:
    ptllh = load_hist(args.ptll_fit)

    hists_nom.append(ptllh)
    hists_err.extend(hist_to_up_down_unc(ptllh))
    if args.w:
        labels.append(
            r"$\mathit{m}_{W}$ $(\mathit{p}_{T}^{\mu}, \mathit{\eta}^{\mu}, \mathit{q}^{\mu})+\mathit{p}_{T}^{\mu\mu}$ "
        )
    else:
        labels.append(r"$\mathit{p}_{T}^{\mu\mu}$ ")
    names.append(f"{fittype} ptll")
    colors.append("#f89c20")

if args.ptll_yll_fit:
    ptllyllh = load_hist(args.ptll_yll_fit)

    hists_nom.append(ptllyllh)
    hists_err.extend(hist_to_up_down_unc(ptllyllh))
    if args.w:
        labels.append(
            r"$\mathit{m}_{W}$ $(\mathit{p}_{T}^{\mu}, \mathit{\eta}^{\mu}, \mathit{q}^{\mu})+(\mathit{p}_{T}^{\mu\mu},\mathit{y}^{\mu\mu})$ "
        )
    else:
        labels.append(r"$(\mathit{p}_{T}^{\mu\mu},\mathit{y}^{\mu\mu})$ ")
    names.append(f"{fittype} ptll-yll")
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

xlabels = {"absYVgen": r"|\mathit{y}^{V}|", "ptVgen": r"\mathit{p}_{T}^{V}"}
xlabel = xlabels[args.obs]

ylabel = r"$d\sigma"
if args.w:
    ylabel += r"^{W}/d" + xlabel.replace("^{V}", "")
    xlabel = r"$" + xlabel.replace("^{V}", "^{W}") + "$"
else:
    ylabel += r"^{Z}/d" + xlabel.replace("^{V}", "")
    xlabel = r"$" + xlabel.replace("^{V}", "^{Z}") + "$"
# ylabel += r"\ cross\ section\ "

if args.obs in ["ptVgen"]:
    xlabel += " (GeV)"
    ylabel += r"\ (pb\,/\,GeV)$"
else:
    ylabel += r"\ (pb)$"

rlabel = "Ratio to " + (
    "data"
    if unfolded_data and args.ratioToData
    else f"\n{labels[0]}" if args.noPrefit else "\nprefit"
)

if args.twoRatios:
    # make two ratios
    subplotsizes = [3, 2, 2]
    rlabel = [
        "Ratio to\n" + labels[3] + "fit",
        rlabel,
    ]
    midratio_idxs = [
        3,
        1,
        2,
        6,
        7,
        8,
        9,
    ]
    if len(args.rrange) == 2:
        rrange = [args.rrange, args.rrange]
    elif len(args.rrange) == 4:
        rrange = [args.rrange[:2], args.rrange[2:]]
    else:
        raise IOError(
            f"Number of arguments for rrange must be 2 or 4 but is {len(args.rrange)}"
        )
else:
    # just one ratio plot
    subplotsizes = [4, 2]
    midratio_idxs = None
    if len(args.rrange) == 2:
        rrange = args.rrange
    else:
        raise IOError(
            f"Number of arguments for rrange must be 2 but is {len(args.rrange)}"
        )

fig = plot_tools.makePlotWithRatioToRef(
    hists=hists_nom,
    hists_ratio=hists,
    midratio_idxs=midratio_idxs,
    labels=labels,
    colors=colors,
    linestyles=linestyles,
    xlabel=xlabel,
    ylabel=ylabel,
    rlabel=rlabel,
    rrange=rrange,
    nlegcols=args.legCols,
    leg_padding=args.legPadding,
    lowerLegCols=args.lowerLegCols,
    lowerLegPos=args.lowerLegPos,
    lower_leg_padding=args.lowerLegPadding,
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
    ratio_to_data=args.ratioToData,
    width_scale=1.25,
    subplotsizes=subplotsizes,
    no_sci=args.noSciy,
    lumi=16.8,
    center_rlabels=False,
    swap_ratio_panels=True,
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

if args.saveForHepdata:
    # open root file
    outfile_root = f"{outdir}/{name}.root"
    rf = input_tools.safeOpenRootFile(outfile_root, mode="recreate")
    logger.warning(f"Saving histograms for HEPData in {outfile_root}")
    for ih, h in enumerate(hists_nom):
        hroot = narf.hist_to_root(h)
        hname = names[ih]
        hroot.SetName(hname)
        hroot.SetTitle(hname)
        hroot.GetXaxis().SetTitle(xlabel)
        hroot.GetYaxis().SetTitle(ylabel)
        hroot.Write()
        logger.info(f"Saving histogram {hname}")
    rf.Close()

plot_tools.save_pdf_and_png(outdir, name)

hintegrals = [np.sum(h.values()) for h in hists_nom]

df = pd.DataFrame()
df["Name"] = names
df["Integral in pb"] = hintegrals

df[r"Ratio to prefit (%)"] = [100 * h / hintegrals[0] for h in hintegrals]
if unfolded_data:
    df[r"Ratio to unfolded data (%)"] = [
        100 * h / hintegrals[idx_unfolded] for h in hintegrals
    ]

plot_tools.write_index_and_log(
    outdir,
    name,
    yield_tables={"Differential cross sections": df},
)

if output_tools.is_eosuser_path(args.outpath) and eoscp:
    output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)
