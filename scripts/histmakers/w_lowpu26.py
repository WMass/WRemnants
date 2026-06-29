import os

from wremnants.utilities import common, parsing
from wums import logging

analysis_label = common.analysis_label(os.path.basename(__file__))
parser, initargs = parsing.common_parser(analysis_label)

parser.add_argument(
    "--flavor",
    type=str,
    choices=["mu", "e"],
    default="mu",
    help="Lepton flavor channel",
)
parser = parsing.set_parser_default(parser, "met", "RawPFMET")
parser = parsing.set_parser_default(parser, "era", "2026_LowPU")
parser = parsing.set_parser_default(
    parser, "aggregateGroups", ["Diboson", "Top", "Wtaunu", "Wenu"]
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
from wremnants.utilities import binning

narf.clingutils.Declare('#include "lowpu_utils.hpp"')

flavor = args.flavor
met_type = args.met

lep_pt_min = 26
lep_pt_max = 56

datasets = getDatasets(
    maxFiles=args.maxFiles,
    filt=args.filterProcs,
    excl=list(
        set(
            args.excludeProcs
            + ([f"EGamma_{args.era}"] if flavor == "mu" else [f"Muon_{args.era}"])
        )
    ),
    aux=args.auxiliaryProcs,
    base_path=args.dataPath,
    era=args.era,
    nanoVersion="v15",
)

for d in datasets:
    logger.info(f"Dataset {d.name}")

axis_pt = hist.axis.Regular(30, 26, 56, name="pt", underflow=False, overflow=False)
axis_eta = hist.axis.Regular(50, -2.5, 2.5, name="eta")
axis_charge = hist.axis.Regular(
    2, -2.0, 2.0, underflow=False, overflow=False, name="charge"
)
# Fine MT axis for transverseMass histogram; integer edges so that the default
# ABCD thresholds [0, 20, 40] fall on exact bin boundaries.
axis_mt = hist.axis.Regular(200, 0, 200, name="mt", underflow=False)
axis_met = hist.axis.Regular(100, 0, 100, name="MET")
axis_npv = hist.axis.Regular(15, 0, 15, name="npv")


def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")

    results = []

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    weightsum = df.SumAndCount("weight")

    # Loose muon veto: exactly one loose muon to reject Z→μμ
    df = df.Define(
        "vetoMuons",
        "Muon_looseId && Muon_pt > 15 && abs(Muon_eta) < 2.4 && abs(Muon_dxybs) < 0.05",
    )
    df = df.Filter("Sum(vetoMuons) == 1")

    df = muon_selections.veto_electrons(df)
    df = muon_selections.apply_met_filters(df)

    # Tight signal muon selection (no calibrations — raw branches)
    df = df.Define(
        "goodMuons",
        f"vetoMuons && Muon_mediumId && Muon_isGlobal && Muon_highPurity"
        f" && Muon_pt > {lep_pt_min} && Muon_pt < {lep_pt_max}"
        f" && abs(Muon_dxybs) < 0.05",
    )
    df = df.Filter("Sum(goodMuons) == 1")

    df = df.Define("Lep_pt", "Muon_pt[goodMuons][0]")
    df = df.Define("Lep_eta", "Muon_eta[goodMuons][0]")
    df = df.Define("Lep_phi", "Muon_phi[goodMuons][0]")
    df = df.Define("Lep_charge", "(float)Muon_charge[goodMuons][0]")
    df = df.Define("relIso", "Muon_pfRelIso04_all[goodMuons][0]")

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

    df = df.Define(
        "transverseMass",
        f"wrem::mt_2(Lep_pt, Lep_phi, {met_type}_pt, {met_type}_phi)",
    )
    df = df.Define("passIso", "relIso < 0.15")
    df = df.Define("passMT", "transverseMass > 40.0")

    if dataset.is_data:
        df = df.DefinePerSample("nominal_weight", "1.0")
    else:
        df = df.Define("nominal_weight", "weight")

    # Main ABCD histogram: boolean passIso/passMT axes like mw_lowPU.py so
    # that FakeSelectorSimpleABCD can compute B/D safely (divide_arrays with cutoff).
    results.append(
        df.HistoBoost(
            "nominal",
            [axis_pt, axis_eta, axis_charge, binning.axis_passIso, binning.axis_passMT],
            ["Lep_pt", "Lep_eta", "Lep_charge", "passIso", "passMT", "nominal_weight"],
        )
    )
    # Uncorrected prediction (base event weight, before any theory corrections).
    # Identical to nominal until theory corrections are applied to nominal_weight.
    results.append(
        df.HistoBoost(
            "nominal_uncorr",
            [axis_pt, axis_eta, axis_charge, binning.axis_passIso, binning.axis_passMT],
            ["Lep_pt", "Lep_eta", "Lep_charge", "passIso", "passMT", "weight"],
        )
    )

    # Fine MT distribution; includes pt+eta+charge for per-bin fakerate in ABCD,
    # and passIso (not passMT) to show the full MT range with iso sideband.
    results.append(
        df.HistoBoost(
            "transverseMass",
            [axis_mt, axis_pt, axis_eta, axis_charge, binning.axis_passIso],
            [
                "transverseMass",
                "Lep_pt",
                "Lep_eta",
                "Lep_charge",
                "passIso",
                "nominal_weight",
            ],
        )
    )

    # Signal-region-filtered observables; --selection none in the plotting script
    # is correct here since the signal region is already applied.
    df_sig = df.Filter("passMT && passIso")
    results.append(
        df_sig.HistoBoost("met", [axis_met], [f"{met_type}_pt", "nominal_weight"])
    )
    results.append(
        df_sig.HistoBoost("npv", [axis_npv], ["PV_npvsGood", "nominal_weight"])
    )

    return results, weightsum


resultdict = narf.build_and_run(datasets, build_graph)

if not args.noScaleToData:
    scale_to_data(resultdict)
    aggregate_groups(datasets, resultdict, args.aggregateGroups)

write_analysis_output(resultdict, f"w_lowpu26_{flavor}_{met_type}.hdf5", args)
