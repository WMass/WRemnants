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
parser = parsing.set_parser_default(parser, "pdfs", ["nnpdf31"])
# Only the central correction; pdfvars/pdfas variants read LHEPdfWeightAltSet11
# which is not stored in the 2026 NanoAOD.
parser = parsing.set_parser_default(
    parser, "theoryCorr", ["scetlib_dyturbo_CT18Z_N3p0LL_N2LO"]
)
parser = parsing.set_parser_default(
    parser, "aggregateGroups", ["Diboson", "Top", "Wtaunu", "Wenu"]
)

args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

import hist

import narf
import narf.clingutils
from wremnants.production import muon_selections, systematics, theory_corrections
from wremnants.production.datasets.dataset_tools import getDatasets
from wremnants.production.histmaker_tools import (
    aggregate_groups,
    scale_to_data,
    write_analysis_output,
)
from wremnants.utilities import binning, samples

narf.clingutils.Declare('#include "lowpu_utils.hpp"')

flavor = args.flavor
met_type = args.met

lep_pt_min = 25
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

axis_pt = hist.axis.Regular(
    56 - lep_pt_min, lep_pt_min, 56, name="pt", underflow=False, overflow=False
)
axis_eta = hist.axis.Regular(24, -2.4, 2.4, name="eta", underflow=False, overflow=False)
axis_charge = hist.axis.Regular(
    2, -2.0, 2.0, underflow=False, overflow=False, name="charge"
)
# Fine MT axis for transverseMass histogram
axis_mt = hist.axis.Regular(200, 0, 200, name="mt", underflow=False)
axis_met = hist.axis.Regular(100, 0, 100, name="MET")
axis_npv = hist.axis.Regular(15, 0, 15, name="npv")

nominal_axes = [
    axis_pt,
    axis_eta,
    axis_charge,
    binning.axis_passIso,
    binning.axis_passMT,
]
nominal_cols = ["Lep_pt", "Lep_eta", "Lep_charge", "passIso", "passMT"]

# Theory corrections (13 TeV files used as placeholder for 13.6 TeV 2026 samples;
# see samples.vprocs_minnlo_2026LowPU comment for details on the energy limitation).
theory_corrs = [*args.theoryCorr, *args.ewTheoryCorr]
corr_helpers = theory_corrections.load_corr_helpers(
    [d.name for d in datasets if d.name in samples.vprocs], theory_corrs
)


def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")

    results = []

    helicity_smoothing_helpers = {}

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")
    df = df.Define("isEvenEvent", "event % 2 == 0")

    weightsum = df.SumAndCount("weight")

    # Loose muon veto: exactly one loose muon to reject Z→μμ
    df = df.Define(
        "vetoMuons",
        "Muon_looseId && Muon_pt > 10 && abs(Muon_eta) < 2.4 && abs(Muon_dxybs) < 0.05",
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
    elif dataset.name in samples.vprocs:
        df = theory_corrections.define_theory_weights_and_corrs(
            df,
            dataset.name,
            corr_helpers,
            args,
            helicity_smoothing_helpers=helicity_smoothing_helpers,
        )
    else:
        df = df.Define("nominal_weight", "weight")

    # Main ABCD histogram: boolean passIso/passMT axes like mw_lowPU.py so
    # that FakeSelectorSimpleABCD can compute B/D safely (divide_arrays with cutoff).
    results.append(
        df.HistoBoost("nominal", nominal_axes, [*nominal_cols, "nominal_weight"])
    )
    # Uncorrected prediction: weight before theory corrections are folded in.
    # define_theory_weights_and_corrs defines nominal_weight_uncorr for vprocs;
    # for all other processes fall back to the raw event weight.
    uncorr_weight = (
        "nominal_weight_uncorr"
        if dataset.name in samples.vprocs and not dataset.is_data
        else "weight"
    )
    results.append(
        df.HistoBoost("nominal_uncorr", nominal_axes, [*nominal_cols, uncorr_weight])
    )

    if dataset.name in samples.vprocs:
        df = systematics.add_theory_hists(
            results,
            df,
            args,
            dataset.name,
            corr_helpers,
            helicity_smoothing_helpers,
            nominal_axes,
            nominal_cols,
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

    # Keep passIso/passMT as axes (like `nominal`) rather than pre-filtering to the
    # signal region, so the same plots can be made in other regions via --selection.
    results.append(
        df.HistoBoost(
            "met",
            [axis_met, binning.axis_passIso, binning.axis_passMT],
            [f"{met_type}_pt", "passIso", "passMT", "nominal_weight"],
        )
    )
    results.append(
        df.HistoBoost(
            "npv",
            [axis_npv, binning.axis_passIso, binning.axis_passMT],
            ["PV_npvsGood", "passIso", "passMT", "nominal_weight"],
        )
    )

    return results, weightsum


resultdict = narf.build_and_run(datasets, build_graph)

if not args.noScaleToData:
    scale_to_data(resultdict)
    aggregate_groups(datasets, resultdict, args.aggregateGroups)

write_analysis_output(resultdict, f"w_lowpu26_{flavor}_{met_type}.hdf5", args)
