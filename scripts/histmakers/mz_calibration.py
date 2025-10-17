import os

from utilities import common, parsing
from wremnants.datasets.datagroups import Datagroups

analysis_label = Datagroups.analysisLabel(os.path.basename(__file__))
parser, initargs = parsing.common_parser(analysis_label)

import math

import hist
import matplotlib.pyplot as plt

import narf
from wremnants import muon_calibration_pt2
from wremnants.datasets.dataset_tools import getDatasets
from wremnants.histmaker_tools import (
    aggregate_groups,
    scale_to_data,
    write_analysis_output,
)
from wums import logging

parser.add_argument(
    "--csVarsHist", action="store_true", help="Add CS variables to dilepton hist"
)
parser.add_argument("--axes", type=str, nargs="*", default=["mll", "ptll"], help="")
parser.add_argument(
    "--finePtBinning", action="store_true", help="Use fine binning for ptll"
)

parser = parsing.set_parser_default(
    parser, "aggregateGroups", ["Diboson", "Top", "Wtaunu", "Wmunu"]
)
parser = parsing.set_parser_default(parser, "excludeProcs", ["QCD"])
parser = parsing.set_parser_default(
    parser, "pt", common.get_default_ptbins(analysis_label)
)

args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)
era = args.era

logger.debug(f"\n\n  Looking for datasets in era: {era} for path: {args.dataPath}")

logger.debug(f"\n\n args.excludeProcs: {args.excludeProcs}")
logger.debug(f"\n\n args.filterProcs: {args.filterProcs}")

datasets = getDatasets(
    maxFiles=args.maxFiles,
    filt=args.filterProcs,
    excl=args.excludeProcs,
    nanoVersion="v9",
    base_path=args.dataPath,
    # extended="msht20an3lo" not in args.pdfs,
    era=era,
)

# dilepton invariant mass cuts
# mass_min, mass_max = common.get_default_mz_window()

# dilepton_ptV_binning = common.get_dilepton_ptV_binning(args.finePtBinning)

axis_yll = hist.axis.Regular(20, -2.5, 2.5, name="yll")
axis_absYll = hist.axis.Regular(10, 0.0, 2.5, name="absYll", underflow=False)

# available axes for dilepton validation plots
all_axes = {
    "mll": hist.axis.Variable(
        [
            60,
            70,
            75,
            78,
            80,
            82,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
            100,
            102,
            105,
            110,
            120,
        ],
        name="mll",
    ),
    "yll": axis_yll,
    "absYll": axis_absYll,
    # "ptll": hist.axis.Variable(dilepton_ptV_binning, name="ptll", underflow=False),
    "etaPlus": hist.axis.Variable([-2.4, -1.2, -0.3, 0.3, 1.2, 2.4], name="etaPlus"),
    "etaMinus": hist.axis.Variable([-2.4, -1.2, -0.3, 0.3, 1.2, 2.4], name="etaMinus"),
    "etaRegionSign": hist.axis.Regular(
        3, 0, 3, name="etaRegionSign", underflow=False, overflow=False
    ),
    "etaRegionRange": hist.axis.Regular(
        3, 0, 3, name="etaRegionRange", underflow=False, overflow=False
    ),
    "absEtaPlus": hist.axis.Regular(8, 0, 2.4, name="absEtaPlus"),
    "absEtaMinus": hist.axis.Regular(8, 0, 2.4, name="absEtaMinus"),
    "etaAbsEta": hist.axis.Variable(
        [
            -2.4,
            -2.0,
            -1.6,
            -1.4,
            -1.2,
            -1.0,
            -0.6,
            0.0,
            0.6,
            1.0,
            1.2,
            1.4,
            1.6,
            2.0,
            2.4,
        ],
        name="etaAbsEta",
    ),
    "etaSum": hist.axis.Regular(12, -4.8, 4.8, name="etaSum"),
    "etaDiff": hist.axis.Variable(
        [-4.8, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 4.8], name="etaDiff"
    ),
    "ptPlus": hist.axis.Regular(int(args.pt[0]), args.pt[1], args.pt[2], name="ptPlus"),
    "ptMinus": hist.axis.Regular(
        int(args.pt[0]), args.pt[1], args.pt[2], name="ptMinus"
    ),
    # "cosThetaStarll": hist.axis.Regular(
    #    200 if args.makeCSQuantileHists else 20,
    #    -1.0,
    #    1.0,
    #    name="cosThetaStarll",
    #    underflow=False,
    #    overflow=False,
    # ),
    # "phiStarll": hist.axis.Regular(
    #    200 if args.makeCSQuantileHists else 20,
    #    -math.pi,
    #    math.pi,
    #    circular=True,
    #    name="phiStarll",
    # ),
    "trigMuons_abseta0": hist.axis.Regular(
        3, 0.0, 2.4, name="trigMuons_abseta0", underflow=False
    ),
    "nonTrigMuons_eta0": hist.axis.Regular(
        int(args.eta[0]), args.eta[1], args.eta[2], name="nonTrigMuons_eta0"
    ),
    "nonTrigMuons_pt0": hist.axis.Regular(
        int(args.pt[0]), args.pt[1], args.pt[2], name="nonTrigMuons_pt0"
    ),
    "nonTrigMuons_charge0": hist.axis.Regular(
        2, -2.0, 2.0, underflow=False, overflow=False, name="nonTrigMuons_charge0"
    ),
}

all_charmonium_axes = {
    "Muon_eta": hist.axis.Regular(
        50, -2.7, 2.7, name="Muon_eta", underflow=False, overflow=False
    ),
    "Muon_pt": hist.axis.Regular(
        20, 0.0, 100.0, name="Muon_pt", underflow=False, overflow=False
    ),
    "Muon_phi": hist.axis.Regular(
        50, -math.pi, math.pi, name="Muon_phi", underflow=False, overflow=False
    ),
    "bkmm_kaon_eta": hist.axis.Regular(
        50, -2.7, 2.7, name="bkmm_kaon_eta", underflow=False, overflow=False
    ),
    "bkmm_kaon_pt": hist.axis.Regular(
        100, 0.0, 20.0, name="bkmm_kaon_pt", underflow=False, overflow=False
    ),
    "bkmm_kaon_phi": hist.axis.Regular(
        50, -math.pi, math.pi, name="bkmm_kaon_phi", underflow=False, overflow=False
    ),
    "bkmm_kaon_charge": hist.axis.Regular(
        2, -2, 2, name="bkmm_kaon_charge", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_eta": hist.axis.Regular(
        50, -2.7, 2.7, name="bkmm_jpsimc_eta", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_pt": hist.axis.Regular(
        100, 0.0, 70.0, name="bkmm_jpsimc_pt", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_phi": hist.axis.Regular(
        50, -math.pi, math.pi, name="bkmm_jpsimc_phi", underflow=False, overflow=False
    ),
    "bkmm_nomc_mass": hist.axis.Regular(
        50, 4.0, 6.5, name="bkmm_nomc_mass", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_mass": hist.axis.Regular(
        50, 4.0, 6.5, name="bkmm_jpsimc_mass", underflow=False, overflow=False
    ),
    "bkmm_nomc_massErr": hist.axis.Regular(
        30, 0.0, 0.15, name="bkmm_nomc_massErr", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_massErr": hist.axis.Regular(
        30, 0.0, 0.15, name="bkmm_jpsimc_massErr", underflow=False, overflow=False
    ),
}

# for a in args.axes:
#   if a not in all_axes.keys():
#        logger.error(
#            f" {a} is not a known axes! Supported axes choices are {list(all_axes.keys())}"
#        )

for a in args.axes:
    if a not in all_charmonium_axes.keys():
        logger.error(
            f" {a} is not a known axes! Supported axes choices are {list(all_charmonium_axes.keys())}"
        )

nominal_cols = args.axes

nominal_axes = [all_charmonium_axes[a] for a in nominal_cols]
hist_names = set()

# global so when event loop run the sumandcount pointers remain and get updated
cutflows = {}


def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")
    results = []
    cutflow = {}

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    df = df.DefinePerSample("unity", "1.0")
    # df = df.Define(
    #    "isEvenEvent", f"event % 2 {'!=' if args.flipEventNumberSplitting else '=='} 0"
    # )

    weightsum = df.SumAndCount("weight")

    cutflow["Total"] = weightsum[0]
    for var in nominal_cols:
        hist_name = f"nominal_{var}_total"
        results.append(df.HistoBoost(hist_name, [all_charmonium_axes[var]], [var]))
        hist_names.add(hist_name)

    df, cutflow_trigger = muon_calibration_pt2.define_jpsi_triggers(
        df, trigger_name="DoubleMu4_3_Jpsi"
    )
    if cutflow_trigger:
        cutflow["HLT"] = cutflow_trigger[0]
        for var in nominal_cols:
            hist_name = f"nominal_{var}_hlt"
            results.append(df.HistoBoost(hist_name, [all_charmonium_axes[var]], [var]))
            hist_names.add(hist_name)

    df, cutflow_bkmm, dfs_per_cut = muon_calibration_pt2.bkmm_selections2(df)
    for i, (selection, action) in enumerate(cutflow_bkmm.items()):
        cutflow[f"{selection}"] = action[0]
        part_hist_name = (
            selection.replace(" ", "_")
            .replace(">", "gt")
            .replace("<", "lt")
            .replace("|", "")
        )
        for var in nominal_cols:
            hist_name = f"nominal_{var}_{part_hist_name}"
            results.append(
                dfs_per_cut[i].HistoBoost(hist_name, [all_charmonium_axes[var]], [var])
            )
            hist_names.add(hist_name)

    df = muon_calibration_pt2.select_first_bkmm_candidate2(df)

    # df = muon_calibration_pt2.analyze_candidate_multiplicity(df)

    for var in nominal_cols:
        hist_name = f"nominal_{var}_onecand"
        results.append(df.HistoBoost(hist_name, [all_charmonium_axes[var]], [var]))
        hist_names.add(hist_name)

    # quick and dirty for now
    for var in nominal_cols:
        results.append(df.HistoBoost("nominal", [all_charmonium_axes[var]], [var]))
        break

    cutflows[dataset.name] = cutflow

    return results, weightsum


logger.debug(f"Datasets are {[d.name for d in datasets]}")

narf_obj = datasets[0]

# logger.info("\n\n Narf object:\n")
# logger.debug(f"narf_obj.name: {narf_obj.name}")
# logger.debug(f"narf_obj.is_data: {narf_obj.is_data}")
# logger.debug(f"narf_obj.xsec: {narf_obj.xsec}")
# logger.debug(f"narf_obj.lumi_csv: {narf_obj.lumi_csv}")
# logger.debug(f"narf_obj.lumi_json: {narf_obj.lumi_json}")
# logger.debug(f"narf_obj.group: {narf_obj.group}")

resultdict = narf.build_and_run(datasets[::-1], build_graph)

for dataset, actions in cutflows.items():
    resultdict[dataset]["cutflow"] = {
        name: action.GetValue() for name, action in actions.items()
    }

if not args.noScaleToData:
    scale_to_data(resultdict)
    aggregate_groups(datasets, resultdict, args.aggregateGroups)

write_analysis_output(
    resultdict, f"{os.path.basename(__file__).replace('py', 'hdf5')}", args
)

"""eras = ['2018A', '2018B', '2018C', '2018D']

aggregated_cutflows = {}
for dataset_name, result in resultdict.items():
    if "cutflow" not in result:
        continue
    
    # Determine aggregate name
    agg_name = dataset_name
    for era in eras:
        if f"data{era}charmonium" in dataset_name:
            agg_name = "data2018charmonium"
            break
    
    # Add to aggregated cutflows
    if agg_name not in aggregated_cutflows:
        aggregated_cutflows[agg_name] = result["cutflow"].copy()
    else:
        # Sum the cutflows
        for cut_name, value in result["cutflow"].items():
            aggregated_cutflows[agg_name][cut_name] += value

print("\n" + "="*80)
print("CUTFLOW TABLE")
print("="*80)
for dataset_name, cutflow in aggregated_cutflows.items():
    if "cutflow" not in result:
        continue
    print(f"\n{dataset_name}:")
    for cut_name, value in cutflow.items():
        print(f"  {cut_name:30s}: {value:.2E}")
print("="*80,"\n\n")"""

# Aggregate cutflows by era
eras = ["2018A", "2018B", "2018C", "2018D"]

aggregated_cutflows = {}
for dataset_name, result in resultdict.items():
    if "cutflow" not in result:
        continue

    # Determine aggregate name
    agg_name = dataset_name
    for era in eras:
        if f"data{era}charmonium" in dataset_name:
            agg_name = "data2018charmonium"
            break

    # Add to aggregated cutflows
    if agg_name not in aggregated_cutflows:
        aggregated_cutflows[agg_name] = result["cutflow"].copy()
    else:
        # Sum the cutflows
        for cut_name, value in result["cutflow"].items():
            aggregated_cutflows[agg_name][cut_name] += value

# Get cutflow data
data_cutflow = aggregated_cutflows.get("data2018charmonium", {})
mc_cutflow = aggregated_cutflows.get("BuToJpsiK", {})

# Get all cut names (selections) in order
cut_names = list(data_cutflow.keys())

# Prepare table data
table_data = []
for cut_name in cut_names:
    data_val = data_cutflow.get(cut_name, 0)
    mc_val = mc_cutflow.get(cut_name, 0)
    ratio = data_val / mc_val if mc_val != 0 else 0
    table_data.append([cut_name, f"{data_val:.2e}", f"{mc_val:.2e}", f"{ratio:.3f}"])

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, len(cut_names) * 0.4))
ax.axis("off")

# Create table
table = ax.table(
    cellText=table_data,
    colLabels=["Selection", "Data", "B -> J/psi + K", "Ratio"],
    loc="center",
)

cutflow_output = "/home/submit/pmlugato/public_html/mz/btojpsik_selection_hists/cutflow_table_fullstats.png"
plt.savefig(cutflow_output, bbox_inches="tight", dpi=300)
logger.info(f"Table saved as {cutflow_output}")
plt.close()


logger.info(
    f"hist variable names to copy paste for plotting:\n\n {' '.join([name.replace('nominal_', '') for name in hist_names])} \n\n"
)
