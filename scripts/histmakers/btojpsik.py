import os

import hist
import matplotlib.pyplot as plt
import ROOT

import narf
import wremnants.production.muon_calibration as muon_calibration
from wremnants.production import btojpsik_selections, muon_calibration
from wremnants.production.btojpsik_axes import all_butojpsik_axes
from wremnants.production.datasets.dataset_tools import getDatasets
from wremnants.production.histmaker_tools import (
    aggregate_groups,
    scale_to_data,
    write_analysis_output,
)
from wremnants.utilities import common, parsing
from wums import logging

analysis_label = common.analysis_label(os.path.basename(__file__))
parser, initargs = parsing.common_parser(analysis_label)

parser.add_argument("--allaxes", action="store_true", help="all histograms")
parser.add_argument(
    "--selectionHists", action="store_true", help="store hist after each selection"
)
parser.add_argument(
    "--include-kaon-scale-variations",
    action="store_true",
    help="uncertainty hists for parameterized model",
)
parser.add_argument("--cutflow", action="store_true", default=None, help="make cutflow")
parser.add_argument(
    "--saveCutflow", type=str, default=None, help="output path for cutflow"
)
parser.add_argument(
    "--checking-signal-stats",
    action="store_true",
    help="Disable non-signal gen matching to compare full BuToJpsiK against signalBuToJpsiK",
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

calib_filepaths = common.calib_filepaths

(
    mc_jpsi_crctn_helper,
    data_jpsi_crctn_helper,
    jpsi_crctn_MC_unc_helper,
    jpsi_crctn_data_unc_helper,
) = muon_calibration.make_jpsi_crctn_helpers(
    calib_filepaths,
    muon_corr_mc=args.muonCorrMC,
    muon_corr_data=args.muonCorrData,
    scale_var_method=args.muonScaleVariation,
    scale_A=args.scale_A,
    scale_e=args.scale_e,
    scale_M=args.scale_M,
    make_uncertainty_helper=True,
    include_covariance=False,
    central=True,
)

logger.debug(
    f"making diff weights helper with calib file {calib_filepaths['kaon_tflite_file']}"
)
diff_weights_helper = (
    ROOT.wrem.SplinesDifferentialWeightsHelper(calib_filepaths["kaon_tflite_file"])
    if (args.muonScaleVariation == "smearingWeightsSplines" or args.validationHists)
    else None  # smearingWeightsSplines is default
)

print(f"\n\n diff_weights_helper is None: {diff_weights_helper is None}")
print(f"\n\n data_jpsi_crctn_helper is None: {data_jpsi_crctn_helper is None}")
print(f"\n\n mc_jpsi_crctn_helper is None: {mc_jpsi_crctn_helper is None}")


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
nonsignal_gen_filter_stats = {}
smearing_weights_procs = []
nominal_cols_gen_smeared = None  # for unc helper, not used for smearingWeightsSplines
cols_gen_smeared = None  # for unc helper, not used for smearingWeightsSplines
isW = False


def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")
    results = []
    cutflow = {}

    storage_type = hist.storage.Double()

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        # df = df.Define("weight", "std::copysign(1.0, genWeight)")
        df = df.Define("weight", "genWeight")
        df = df.Define(
            "nominal_weight", "static_cast<double>(weight)"
        )  # stupid for now, for unc helpers later

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

    df, cutflow_trigger = btojpsik_selections.define_jpsi_triggers(
        df, trigger_name="DoubleMu4_3_Jpsi"
    )
    # cutflow_trigger = None
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

    # selections (og was BPH-21-006)
    # TODO: shouldn't have to write the numbers twice smh but don't feel like changing right now
    bkmm_selections = [
        (
            "dimuon cand neutral",
            lambda d: btojpsik_selections.select_opposite_sign_dimuon(d),
        ),
        (
            "muon |eta| < 1.4",
            lambda d: btojpsik_selections.select_muon_eta(d, 1.4),
        ),
        (
            "muon pT > 4",
            lambda d: btojpsik_selections.select_muon_pt(d, 4),
        ),
        (
            "muon softMVA > 0.45",
            lambda d: btojpsik_selections.select_muon_softmva(d, 0.45),
        ),
        (
            "dimuon pT > 7",
            lambda d: btojpsik_selections.select_dimuon_pt(d, 7.0),
        ),
        (
            "dimuon alphaBS < 0.4",
            lambda d: btojpsik_selections.select_dimuon_alphabs(d, 0.4),
        ),  # og 0.4
        (
            "dimuon vtx prob > 0.1",
            lambda d: btojpsik_selections.select_dimuon_vtx_prob(d, 0.1),
        ),  # og 0.1
        (
            "dimuon sl3d > 4",
            lambda d: btojpsik_selections.select_dimuon_sl3d(d, 4),
        ),  # og 4
        (
            "bkmm vtx prob > 0.3",
            lambda d: btojpsik_selections.select_bkmm_vtx_prob(d, 0.3),
        ),  # og 0.025
        (
            "bkmm mass window",
            lambda d: btojpsik_selections.select_bkmm_mass_window(d, 5.3, 0.1),
        ),  # og 5.4, 0.5
        # adding kaon sels to match what is used to produce maps (for now)
        (
            "kaon |eta| < 1.4",
            lambda d: btojpsik_selections.select_kaon_eta(d, 1.4),
        ),
        (
            "kaon pT < 8",
            lambda d: btojpsik_selections.select_kaon_pt(d, 8),
        ),
        (
            "bkmm bmm bdt output > 0.10",
            lambda d: btojpsik_selections.select_bkmm_bmm_bdt(d, 0.10),
        ),  # NOTE: this doesn't touch kaon so fine to use...
    ]

    df, cutflow_bkmm, dfs_per_cut = btojpsik_selections.bkmm_selections(
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

    needs_gen_match = (
        not dataset.is_data
        and dataset.name != "signalBuToJpsiK"
        and not args.checking_signal_stats
    )
    df = btojpsik_selections.select_only_passing_bkmm_candidates(
        df,
        signal=dataset.name == "signalBuToJpsiK",
        select_best=True,
        gen_match_nonsignal=needs_gen_match,
        gen_filter_stats=nonsignal_gen_filter_stats if needs_gen_match else None,
        dataset_name=dataset.name if needs_gen_match else None,
    )
    if args.selectionHists:
        for var in nominal_cols:
            # if "gen" in str(var) and dataset.is_data:
            #    results.append(df.HistoBoost(hist_name, ))
            hist_name = f"nominal_{var}_onecand"
            results.append(df.HistoBoost(hist_name, [all_butojpsik_axes[var]], [var]))
            hist_names.add(hist_name)

    ###

    # move below for checks

    # for var in nominal_cols:
    #   # if "gen" in str(var) and dataset.is_data:
    #    #    results.append(df.HistoBoost(hist_name, ))
    #    hist_name = f"nominal_{var}"
    #    results.append(df.HistoBoost(hist_name, [all_butojpsik_axes[var]], [var]))
    #    hist_names.add(hist_name)
    #    final_var = var

    # hack to avoid expectation in makeDataMCstackratioplot when doing selection hists
    # results.append(
    #    df.HistoBoost(f"nominal", [all_butojpsik_axes[final_var]], [final_var])
    # )

    ###

    # correct kaon pt w bkmm_kaon_pt, vectors of length 1
    jpsi_helper = data_jpsi_crctn_helper if dataset.is_data else mc_jpsi_crctn_helper
    df = df.Define(
        "kaon_jpsiCorrectedPt",
        jpsi_helper,
        ["bkmm_jpsimc_kaon1pt", "bkmm_jpsimc_kaon1eta", "bkmm_kaon_charge"],
    )
    # import pdb
    # pdb.set_trace()

    reco_sel_GF = "bkmm_kaon_stuff"
    df = df.Alias(f"{reco_sel_GF}_recoPt", "kaon_jpsiCorrectedPt")
    df = df.Alias(f"{reco_sel_GF}_recoEta", "bkmm_jpsimc_kaon1eta")
    df = df.Alias(f"{reco_sel_GF}_recoCharge", "bkmm_kaon_charge")
    has_gen_kinematics = not dataset.is_data and (
        dataset.name == "signalBuToJpsiK" or needs_gen_match
    )
    if has_gen_kinematics:
        # temp vars defined during selection of best cand
        df = df.Alias(f"{reco_sel_GF}_genPt", "temp_kaon_genPt")
        df = df.Alias(f"{reco_sel_GF}_genEta", "temp_kaon_genEta")
        df = df.Alias(f"{reco_sel_GF}_genCharge", "temp_kaon_genCharge")
        # df = df.Alias(f"{reco_sel_GF}_recoPt", "kaon_jpsiCorrectedPt") # using corrected pT now, before was nominalbkmm_kaon_pt

    for var in nominal_cols:
        # if "gen" in str(var) and dataset.is_data:
        #    results.append(df.HistoBoost(hist_name, ))
        hist_name = f"nominal_{var}"
        results.append(df.HistoBoost(hist_name, [all_butojpsik_axes[var]], [var]))
        hist_names.add(hist_name)
        final_var = var
    results.append(
        df.HistoBoost(f"nominal", [all_butojpsik_axes[final_var]], [final_var])
    )

    fitcols = ["bkmm_jpsimc_mass"]
    fitcols.append(f"{reco_sel_GF}_recoPt")
    fitcols.append(f"{reco_sel_GF}_recoEta")
    fitcols.append(f"{reco_sel_GF}_recoCharge")
    fitaxes = [all_butojpsik_axes[a] for a in fitcols]

    if has_gen_kinematics:
        input_kinematics = [
            f"{reco_sel_GF}_recoPt",
            f"{reco_sel_GF}_recoEta",
            f"{reco_sel_GF}_recoCharge",
            f"{reco_sel_GF}_genPt",
            f"{reco_sel_GF}_genEta",
            f"{reco_sel_GF}_genCharge",
        ]
        if diff_weights_helper:
            df = df.Define(
                f"{reco_sel_GF}_response_weight",
                diff_weights_helper,
                [*input_kinematics],
            )
            input_kinematics.append(f"{reco_sel_GF}_response_weight")

        # kaon scale variation
        if args.include_kaon_scale_variations:
            df = muon_calibration.add_jpsi_crctn_stats_unc_hists(
                args,
                df,
                fitaxes,  # need to add back
                results,
                fitcols,
                cols_gen_smeared,
                calib_filepaths,
                jpsi_crctn_data_unc_helper,
                smearing_weights_procs,
                reco_sel_GF,
                dataset.name,
                isW,
                storage_type=storage_type,
            )

        # import pdb
        # pdb.set_trace()

    if dataset.is_data:
        # df = df.Define("bkmm_kaon_curvature", "1. / bkmm_kaon_pt")
        df = df.Define("bkmm_kaon_curvature", "1. / bkmm_jpsimc_kaon1pt")

    if args.include_kaon_scale_variations:
        hist_name = "nominal_HistToFit"
        results.append(df.HistoBoost(hist_name, fitaxes, fitcols))

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

for name, (after, before, filtered_count) in nonsignal_gen_filter_stats.items():
    before_val = float(before.GetValue())
    eff = float(after.GetValue()) / before_val if before_val != 0 else 0.0
    resultdict[name]["gen_filter_eff"] = eff
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

##############   move everything below to utils somewhere

if args.cutflow:

    eras = ["2018A", "2018B", "2018C", "2018D"]

    aggregated_cutflows = {}
    for dataset_name, result in resultdict.items():
        if "cutflow" not in result:
            continue

        agg_name = dataset_name
        for era in eras:
            if f"data{era}charmonium" in dataset_name:
                agg_name = "data2018charmonium"
                break

        if agg_name not in aggregated_cutflows:
            aggregated_cutflows[agg_name] = result["cutflow"].copy()
        else:
            for cut_name, value in result["cutflow"].items():
                aggregated_cutflows[agg_name][cut_name] += value

    # construct cutflow table
    data_cutflow = aggregated_cutflows.get("data2018charmonium", {})
    signal_cutflow = aggregated_cutflows.get("signalBuToJpsiK", {})
    bjk_cutflow = aggregated_cutflows.get("BuToJpsiK", {})

    cut_names = list(data_cutflow.keys())

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
