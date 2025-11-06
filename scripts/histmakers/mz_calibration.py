import os

from utilities import parsing
from wremnants.datasets.datagroups import Datagroups

analysis_label = Datagroups.analysisLabel(os.path.basename(__file__))
parser, initargs = parsing.common_parser(analysis_label)


import matplotlib.pyplot as plt

import narf
from wremnants import muon_calibration_pt2
from wremnants.butojpsik_histograms import all_butojpsik_axes
from wremnants.datasets.dataset_tools import getDatasets
from wremnants.histmaker_tools import (
    aggregate_groups,
    scale_to_data,
    write_analysis_output,
)
from wums import logging

parser.add_argument("--allaxes", action="store_true", help="all histograms")
parser.add_argument(
    "--selectionHists", action="store_true", help="store hist after each selection"
)
parser.add_argument(
    "--saveCutflow", type=str, default=None, help="output path for cutflow"
)
parser.add_argument(
    "--cutflowName",
    type=str,
    default=None,
    help="output filename for cutflow, default is postfix (-p)",
)
parser.add_argument(
    "--csVarsHist", action="store_true", help="Add CS variables to dilepton hist"
)
parser.add_argument("--axes", type=str, nargs="*", default=[], help="")

parser = parsing.set_parser_default(
    parser, "aggregateGroups", ["Diboson", "Top", "Wtaunu", "Wmunu"]
)
parser = parsing.set_parser_default(parser, "excludeProcs", ["QCD"])

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

# for a in args.axes:
#   if a not in all_axes.keys():
#        logger.error(
#            f" {a} is not a known axes! Supported axes choices are {list(all_axes.keys())}"
#        )

for a in args.axes:
    if a not in all_butojpsik_axes.keys():
        logger.error(
            f" {a} is not a known axes! Supported axes choices are {list(all_butojpsik_axes.keys())}"
        )

nominal_cols = args.axes
nominal_axes = [all_butojpsik_axes[a] for a in nominal_cols]
if args.allaxes:
    nominal_cols = list(all_butojpsik_axes.keys())
    nominal_axes = [all_butojpsik_axes[a] for a in all_butojpsik_axes]
hist_names = set()

# global so when event loop run the sumandcount pointers remain and get updated
cutflows = {}
signal_gen_filter_stats = {}


def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")
    results = []
    cutflow = {}

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    df = df.DefinePerSample("unity", "1.0")

    if dataset.name == "signalBuToJpsiK":
        total_evt_count = (
            df.Count()
        )  # matches evtcount in graph_builder otherwise complains
        gen_weight_before = df.Sum("genWeight")
        df = df.Filter("Any(bkmm_gen_pdgId != 0)", "require gen-matched candidate")
        gen_weight_after = df.Sum("genWeight")
        filtered_evt_count = df.Count()
        weightsum_sum = df.Sum("weight")
        weightsum = (weightsum_sum, total_evt_count)
        signal_gen_filter_stats[dataset.name] = (
            gen_weight_after,
            gen_weight_before,
            filtered_evt_count,
        )
    else:
        weightsum = df.SumAndCount("weight")

    cutflow["Total"] = weightsum[0]
    if args.selectionHists:
        for var in nominal_cols:
            # if "gen" in str(var) and dataset.is_data:
            #    results.append(df.HistoBoost(hist_name, ))
            hist_name = f"nominal_{var}_total"
            results.append(df.HistoBoost(hist_name, [all_butojpsik_axes[var]], [var]))
            hist_names.add(hist_name)

    df, cutflow_trigger = muon_calibration_pt2.define_jpsi_triggers(
        df, trigger_name="DoubleMu4_3_Jpsi"
    )
    if cutflow_trigger:
        cutflow["HLT"] = cutflow_trigger[0]
        if args.selectionHists:
            for var in nominal_cols:
                # if "gen" in str(var) and dataset.is_data:
                #    results.append(df.HistoBoost(hist_name, ))
                hist_name = f"nominal_{var}_hlt"
                results.append(
                    df.HistoBoost(hist_name, [all_butojpsik_axes[var]], [var])
                )
                hist_names.add(hist_name)

    # selectionssss (og was BPH-21-006)
    bkmm_selections = [
        (
            "dimuon cand neutral",
            "Require at least one opposite-sign dimuon candidate",
            lambda d: muon_calibration_pt2.select_opposite_sign_dimuon(d),
        ),
        (
            "muon |eta| < 1.4",
            "Require |eta| < 1.4 for both muons",
            lambda d: muon_calibration_pt2.select_muon_eta(d, 1.4),
        ),
        (
            "muon pT > 4",
            "Require pT > 4 GeV for both muons",
            lambda d: muon_calibration_pt2.select_muon_pt(d, 4),
        ),
        (
            "muon softMVA > 0.45",
            "Require soft MVA > 0.45 for both muons",
            lambda d: muon_calibration_pt2.select_muon_softmva(d, 0.45),
        ),
        (
            "dimuon pT > 7",
            "Require dimuon pT > 7 GeV",
            lambda d: muon_calibration_pt2.select_dimuon_pt(d, 7.0),
        ),
        (
            "dimuon alphaBS < 0.4",
            "Require dimuon alphaBS < 0.4",
            lambda d: muon_calibration_pt2.select_dimuon_alphabs(d, 0.4),
        ),  # og 0.4
        (
            "dimuon vtx prob > 0.1",
            "Require dimuon vertex prob > 0.1",
            lambda d: muon_calibration_pt2.select_dimuon_vtx_prob(d, 0.1),
        ),  # og 0.1
        (
            "dimuon sl3d > 4",
            "Require dimuon 3D significance > 4",
            lambda d: muon_calibration_pt2.select_dimuon_sl3d(d, 4),
        ),  # og 4
        (
            "bkmm vtx prob > 0.3",
            "Require bkmm J/psi+MC vertex prob > 0.3",
            lambda d: muon_calibration_pt2.select_bkmm_vtx_prob(d, 0.3),
        ),  # og 0.025
        (
            "bkmm mass window",
            "Require |bkmm mass - 5.3| < 0.5 GeV",
            lambda d: muon_calibration_pt2.select_bkmm_mass_window(d, 5.3, 0.5),
        ),  # og 5.4, 0.5
        # ("bkmm bmm bdt output > 0.10", "Require bkmm bmm bdt output variable > 0.10",
        # lambda d: muon_calibration_pt2.select_bkmm_bmm_bdt(d, 0.10)) # NOTE: confirm this doesn't touch kaon
    ]

    df, cutflow_bkmm, dfs_per_cut = muon_calibration_pt2.bkmm_selections(
        df, dataset.name, bkmm_selections
    )

    for i, (selection, action) in enumerate(cutflow_bkmm.items()):
        cutflow[f"{selection}"] = action[0]
        if args.selectionHists:
            part_hist_name = (
                selection.replace(" ", "_")
                .replace(">", "gt")
                .replace("<", "lt")
                .replace("|", "")
            )
            for var in nominal_cols:
                # if "gen" in str(var) and dataset.is_data:
                #    results.append(df.HistoBoost(hist_name, ))
                hist_name = f"nominal_{var}_{part_hist_name}"
                results.append(
                    dfs_per_cut[i].HistoBoost(
                        hist_name, [all_butojpsik_axes[var]], [var]
                    )
                )
                hist_names.add(hist_name)

    df = muon_calibration_pt2.select_only_passing_bkmm_candidates(
        df, signal=dataset.name == "signalBuToJpsiK", select_first=True
    )
    if args.selectionHists:
        for var in nominal_cols:
            # if "gen" in str(var) and dataset.is_data:
            #    results.append(df.HistoBoost(hist_name, ))
            hist_name = f"nominal_{var}_onecand"
            results.append(df.HistoBoost(hist_name, [all_butojpsik_axes[var]], [var]))
            hist_names.add(hist_name)

    for var in nominal_cols:
        # if "gen" in str(var) and dataset.is_data:
        #    results.append(df.HistoBoost(hist_name, ))
        hist_name = f"nominal_{var}"
        results.append(df.HistoBoost(hist_name, [all_butojpsik_axes[var]], [var]))
        hist_names.add(hist_name)
        final_var = var

    # hack to avoid some shit in makeDataMCstackratioplot or whatever
    results.append(
        df.HistoBoost(f"nominal", [all_butojpsik_axes[final_var]], [final_var])
    )

    cutflows[dataset.name] = cutflow

    return results, weightsum


logger.debug(f"Datasets are {[d.name for d in datasets]}")

narf_obj = datasets[0]

resultdict = narf.build_and_run(datasets[::-1], build_graph)

for name, (after, before, filtered_count) in signal_gen_filter_stats.items():
    resultdict[name]["gen_filter_eff"] = float(after.GetValue()) / float(
        before.GetValue()
    )
    resultdict[name]["event_count"] = float(filtered_count.GetValue())

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

##############   move shit below to utils somewhere


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
signal_cutflow = aggregated_cutflows.get("signalBuToJpsiK", {})
bjk_cutflow = aggregated_cutflows.get("BuToJpsiK", {})

# Get all cut names (selections) in order
cut_names = list(data_cutflow.keys())

# Prepare table data
table_data = []
for cut_name in cut_names:
    data_val = data_cutflow.get(cut_name, 0)
    signal_val = signal_cutflow.get(cut_name, 0)
    bjk_val = bjk_cutflow.get(cut_name, 0)
    ratio = data_val / signal_val if signal_val != 0 else 0
    ratio2 = bjk_val / signal_val if signal_val != 0 else 0
    table_data.append(
        [
            cut_name,
            f"{data_val:.2e}",
            f"{signal_val:.2e}",
            f"{bjk_val:.2e}",
            f"{ratio:.3f}",
            f"{ratio2:.3f}",
        ]
    )

fig, ax = plt.subplots(figsize=(8, len(cut_names) * 0.4))
ax.axis("off")

table = ax.table(
    cellText=table_data,
    colLabels=[
        "Selection",
        "Data",
        "Signal",
        "B->Jpsi+K",
        "Data/Signal",
        "B->Jpsi+K/Signal",
    ],
    loc="center",
)


if args.saveCutflow:
    os.makedirs(args.saveCutflow, exist_ok=True)
    cutflow_postfix = args.cutflowName if args.cutflowName else args.postfix
    cutflow_path = f"{args.saveCutflow}/cutflow_{cutflow_postfix}.png"
    plt.savefig(cutflow_path, bbox_inches="tight", dpi=300)
    logger.info(f"Table saved as {cutflow_path}")
else:
    print("\nCutflow Table:")
    print(
        f"{'Selection':<30} {'Data':>15} {'Signal':>15} {'B->Jpsi+K':>15} {'Data/Signal':>10} {'B->Jpsi+K/Signal':>10}"
    )
    print("-" * 75)
    for row in table_data:
        print(
            f"{row[0]:<30} {row[1]:>15} {row[2]:>15} {row[3]:>15} {row[4]:>10} {row[5]:>10}"
        )
    print()
plt.close()


print(
    f"hist variable names to copy paste for plotting:\n\n {' '.join([name.replace('nominal_', '') for name in hist_names])} \n\n"
)
