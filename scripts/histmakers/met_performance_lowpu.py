import os

from wremnants.utilities import common, parsing
from wums import logging

analysis_label = common.analysis_label(os.path.basename(__file__))
parser, initargs = parsing.common_parser(analysis_label)

parser.add_argument(
    "--flavor",
    type=str,
    choices=["ee", "mumu"],
    default="mumu",
    help="Dilepton flavor channel",
)
parser = parsing.set_parser_default(parser, "met", "RawPFMET")
parser = parsing.set_parser_default(parser, "era", "2026_LowPU")
parser = parsing.set_parser_default(
    parser, "aggregateGroups", ["Diboson", "Top", "Wtaunu", "Wmunu", "Wenu"]
)

args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

import hist

import narf
import narf.clingutils
from wremnants.production import muon_selections
from wremnants.production.datasets.dataset_tools import getDatasets
from wremnants.production.histmaker_tools import (
    aggregate_groups,
    scale_to_data,
    write_analysis_output,
)

narf.clingutils.Declare('#include "recoil_tools.hpp"')

flavor = args.flavor
met_type = args.met

mass_min = 60
mass_max = 120
lep_pt_min = 25

datasets = getDatasets(
    maxFiles=args.maxFiles,
    filt=args.filterProcs,
    excl=list(
        set(
            args.excludeProcs
            + ([f"EGamma_{args.era}"] if flavor == "mumu" else [f"Muon_{args.era}"])
        )
    ),
    aux=args.auxiliaryProcs,
    base_path=args.dataPath,
    era=args.era,
    nanoVersion="v15",
)

for d in datasets:
    logger.info(f"Dataset {d.name}")

axis_ptl = hist.axis.Regular(100, 0.0, 200.0, name="ptl")
axis_etal = hist.axis.Regular(50, -2.5, 2.5, name="etal")
axis_mll = hist.axis.Regular(60, 60, 120, name="mll")
axis_ptll = hist.axis.Regular(150, 0, 150, name="ptll")
axis_mt = hist.axis.Regular(200, 0.0, 200.0, name="mt", underflow=False)
axis_met = hist.axis.Regular(100, 0, 100, name="MET")
axis_met_wlike = hist.axis.Regular(200, 0, 200, name="WlikeMET")
axis_recoil_para = hist.axis.Regular(150, -100, 50, name="recoil_para")
axis_recoil_perp = hist.axis.Regular(100, -50, 50, name="recoil_perp")
axis_recoil_para_qt = hist.axis.Regular(100, -50, 50, name="recoil_para_qt")
axis_npv = hist.axis.Regular(15, 0, 15, name="npv")


def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")

    results = []

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    weightsum = df.SumAndCount("weight")

    df = df.Define("isEvenEvent", "event % 2 == 0")

    if flavor == "mumu":
        df = df.Define(
            "goodLeptons",
            f"Muon_mediumId && Muon_pt > {lep_pt_min} && Muon_pt < 150"
            " && Muon_pfRelIso04_all < 0.15 && abs(Muon_eta) < 2.4 && abs(Muon_dxybs) < 0.05",
        )
        df = df.Filter("Sum(goodLeptons) == 2")
        df = df.Define("Lep_pt", "Muon_pt[goodLeptons]")
        df = df.Define("Lep_phi", "Muon_phi[goodLeptons]")
        df = df.Define("Lep_eta", "Muon_eta[goodLeptons]")
        df = df.Define("Lep_mass", "Muon_mass[goodLeptons]")
        df = df.Define("Lep_charge", "Muon_charge[goodLeptons]")
    else:
        df = df.Define(
            "goodLeptons",
            f"Electron_cutBased >= 3 && Electron_pt > {lep_pt_min} && Electron_pt < 150"
            " && Electron_pfRelIso04_all < 0.15 && abs(Electron_eta) < 2.4"
            " && !(abs(Electron_eta) > 1.4442 && abs(Electron_eta) < 1.566)"
            " && abs(Electron_dxy) < 0.05",
        )
        df = df.Filter("Sum(goodLeptons) == 2")
        df = df.Define("Lep_pt", "Electron_pt[goodLeptons]")
        df = df.Define("Lep_phi", "Electron_phi[goodLeptons]")
        df = df.Define("Lep_eta", "Electron_eta[goodLeptons]")
        df = df.Define("Lep_mass", "Electron_mass[goodLeptons]")
        df = df.Define("Lep_charge", "Electron_charge[goodLeptons]")

    df = df.Filter("(Lep_charge[0] + Lep_charge[1]) == 0")

    df = df.Define(
        "Lep1_mom4",
        "ROOT::Math::PtEtaPhiMVector(Lep_pt[0], Lep_eta[0], Lep_phi[0], Lep_mass[0])",
    )
    df = df.Define(
        "Lep2_mom4",
        "ROOT::Math::PtEtaPhiMVector(Lep_pt[1], Lep_eta[1], Lep_phi[1], Lep_mass[1])",
    )
    df = df.Define(
        "ll_mom4",
        "ROOT::Math::PxPyPzEVector(Lep1_mom4) + ROOT::Math::PxPyPzEVector(Lep2_mom4)",
    )
    df = df.Define("mll", "ll_mom4.mass()")
    df = df.Filter(f"mll > {mass_min} && mll < {mass_max}")
    df = df.Define("ptll", "ll_mom4.pt()")
    df = df.Define("yll", "ll_mom4.Rapidity()")
    df = df.Define("phill", "ll_mom4.phi()")

    df = muon_selections.apply_met_filters(df)

    # NanoAODv12 for Run 3 uses RawPFMET; older productions use RawMET
    if not df.HasColumn(f"{met_type}_pt"):
        fallback = met_type.replace("RawPFMET", "RawMET")
        if not df.HasColumn(f"{fallback}_pt"):
            raise RuntimeError(
                f"Neither {met_type}_pt nor {fallback}_pt found in dataset {dataset.name}"
            )
        logger.warning(
            f"Branch {met_type}_pt not found in {dataset.name}, falling back to {fallback}_pt"
        )
        df = df.Define(f"{met_type}_pt", f"{fallback}_pt")
        df = df.Define(f"{met_type}_phi", f"{fallback}_phi")

    # W-like selection: randomly assign one lepton as the "trigger lepton" by even/odd event
    df = df.Define("TrigLep_charge", "isEvenEvent ? -1 : 1")
    df = df.Define("NonTrigLep_charge", "-TrigLep_charge")
    df = df.Define("trigLeps", "Lep_charge == TrigLep_charge")
    df = df.Define("nonTrigLeps", "Lep_charge == NonTrigLep_charge")
    df = df.Define("TrigLep_pt", "Lep_pt[trigLeps][0]")
    df = df.Define("TrigLep_phi", "Lep_phi[trigLeps][0]")
    df = df.Define("NonTrigLep_pt", "Lep_pt[nonTrigLeps][0]")
    df = df.Define("NonTrigLep_phi", "Lep_phi[nonTrigLeps][0]")

    df = df.Define(
        "transverseMass",
        f"wrem::get_mt_wlike(TrigLep_pt, TrigLep_phi, NonTrigLep_pt, NonTrigLep_phi, {met_type}_pt, {met_type}_phi)",
    )
    df = df.Define(
        "met_wlike_TV2",
        f"wrem::get_met_wlike(NonTrigLep_pt, NonTrigLep_phi, {met_type}_pt, {met_type}_phi)",
    )
    df = df.Define("met_wlike_pt", "met_wlike_TV2.Mod()")

    df = df.Define(
        "recoil",
        f"wrem::compute_recoil_from_met({met_type}_pt, {met_type}_phi, Lep_pt, Lep_phi, ptll, phill)",
    )
    df = df.Define("recoil_para", "recoil[0]")
    df = df.Define("recoil_perp", "recoil[1]")
    df = df.Define("recoil_para_qt", "recoil_para + ptll")

    if dataset.is_data:
        df = df.DefinePerSample("nominal_weight", "1.0")
    else:
        df = df.Define("nominal_weight", "weight")

    results.append(df.HistoBoost("lep_pt", [axis_ptl], ["Lep_pt", "nominal_weight"]))
    results.append(df.HistoBoost("lep_eta", [axis_etal], ["Lep_eta", "nominal_weight"]))
    results.append(df.HistoBoost("mll", [axis_mll], ["mll", "nominal_weight"]))
    results.append(df.HistoBoost("ptll", [axis_ptll], ["ptll", "nominal_weight"]))
    results.append(
        df.HistoBoost("met", [axis_met], [f"{met_type}_pt", "nominal_weight"])
    )
    results.append(
        df.HistoBoost("transverseMass", [axis_mt], ["transverseMass", "nominal_weight"])
    )
    results.append(
        df.HistoBoost("met_wlike", [axis_met_wlike], ["met_wlike_pt", "nominal_weight"])
    )
    results.append(
        df.HistoBoost(
            "recoil_para", [axis_recoil_para], ["recoil_para", "nominal_weight"]
        )
    )
    results.append(
        df.HistoBoost(
            "recoil_perp", [axis_recoil_perp], ["recoil_perp", "nominal_weight"]
        )
    )
    results.append(
        df.HistoBoost(
            "recoil_para_qt",
            [axis_recoil_para_qt],
            ["recoil_para_qt", "nominal_weight"],
        )
    )
    results.append(df.HistoBoost("npv", [axis_npv], ["PV_npvsGood", "nominal_weight"]))

    return results, weightsum


resultdict = narf.build_and_run(datasets, build_graph)

if not args.noScaleToData:
    scale_to_data(resultdict)
    aggregate_groups(datasets, resultdict, args.aggregateGroups)

write_analysis_output(
    resultdict, f"met_performance_lowpu_{flavor}_{met_type}.hdf5", args
)
