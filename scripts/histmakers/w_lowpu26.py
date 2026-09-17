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

if not args.noRecoil:
    from wremnants.production import recoil_tools
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
mtw_min = 40

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
    4, lep_pt_min, lep_pt_max, name="pt", underflow=False, overflow=False
)
axis_eta = hist.axis.Regular(3, -2.4, 2.4, name="eta", underflow=False, overflow=False)
# Finer pt/eta binning for plotting (lep_pt_eta histogram below). Kept separate
# from axis_pt/axis_eta above, which stay coarse to limit the dimensionality of
# `nominal` (already carries ptW + the ABCD axes). No mt axis here either, so
# this stays small even at fine pt/eta binning: 31*24*2*2*2 = 5952 bins.
axis_pt_fine = hist.axis.Regular(
    int(lep_pt_max - lep_pt_min),
    lep_pt_min,
    lep_pt_max,
    name="pt",
    underflow=False,
    overflow=False,
)
axis_eta_fine = hist.axis.Regular(
    24, -2.4, 2.4, name="eta", underflow=False, overflow=False
)
axis_charge = hist.axis.Regular(
    2, -2.0, 2.0, underflow=False, overflow=False, name="charge"
)
# MT axis for transverseMass histogram, matching mw_lowPU.py (2017) exactly so
# both eras can be compared/overlaid directly.
axis_mt = hist.axis.Variable(
    [0, 20] + list(range(mtw_min, 150, 1)) + [150],
    name="mt",
    underflow=False,
    overflow=True,
)
axis_met = hist.axis.Regular(100, 0, 100, name="MET")
axis_npv = hist.axis.Regular(15, 0, 15, name="npv")
axis_ptW = hist.axis.Variable(
    [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 75, 90, 150],
    name="ptW",
    underflow=False,
    overflow=True,
)

nominal_axes = [
    axis_pt,
    axis_eta,
    axis_charge,
    axis_ptW,
    binning.axis_passIso,
    binning.axis_passMT,
]
nominal_cols = ["Lep_pt", "Lep_eta", "Lep_charge", "ptW", "passIso", "passMT"]

# Theory corrections (13 TeV files used as placeholder for 13.6 TeV 2026 samples;
# see samples.vprocs_minnlo_2026LowPU comment for details on the energy limitation).
theory_corrs = [*args.theoryCorr, *args.ewTheoryCorr]
corr_helpers = theory_corrections.load_corr_helpers(
    [d.name for d in datasets if d.name in samples.vprocs], theory_corrs
)

# Reuse the 2017 low-PU recoil calibration (data/MC recoil response+resolution
# tflite model, MET-XY correction, ptV reweighting). It was trained on 2017
# low-PU Z data and isn't strictly validated for 13.6 TeV/2026 conditions, but
# the RawPFMET-based MET reconstruction is the same regardless of era, so it
# should still improve the mT shape rather than hurt it. Controlled by the
# existing --noRecoil flag, same as mw_lowPU.py.
if not args.noRecoil:
    recoilHelper = recoil_tools.Recoil("lowPU", args, flavor)


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

    # recoil_tools.setup_MET() hardcodes the branch name "RawMET_pt"/"_phi" for
    # met=="RawPFMET" (the 2017 low-PU NanoAOD convention); 2026's NanoAODv15
    # instead stores it directly as RawPFMET_pt/_phi. Alias so recoil_tools
    # finds a valid branch regardless of era.
    if (
        not args.noRecoil
        and met_type == "RawPFMET"
        and not df.HasColumn("RawMET_pt")
    ):
        df = df.Alias("RawMET_pt", f"{met_type}_pt")
        df = df.Alias("RawMET_phi", f"{met_type}_phi")

    df = df.Define("passIso", "relIso < 0.15")

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

    # Recoil correction needs ptVgen/phiVgen (defined above by
    # define_theory_weights_and_corrs for vprocs), so it must run after the
    # theory-correction block.
    if not args.noRecoil:
        leps = ["Lep_pt", "Lep_eta", "Lep_phi", "Lep_charge"]
        df = recoilHelper.recoil_W(
            df, results, dataset, samples.vprocs, leps, leps, mtw_min=mtw_min
        )
    else:
        df = df.Alias("MET_corr_rec_pt", f"{met_type}_pt")
        df = df.Alias("MET_corr_rec_phi", f"{met_type}_phi")

    df = df.Define(
        "transverseMass",
        "wrem::mt_2(Lep_pt, Lep_phi, MET_corr_rec_pt, MET_corr_rec_phi)",
    )
    # Boson pt proxy: vector sum of the lepton and MET, as in mw_lowPU.py
    df = df.Define(
        "ptW",
        "wrem::pt_2(Lep_pt, Lep_phi, MET_corr_rec_pt, MET_corr_rec_phi)",
    )
    df = df.Define("passMT", f"transverseMass > {mtw_min}")

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

    # Finer pt/eta binning for plotting purposes (the pt/eta axes in `nominal`
    # are coarse to keep its dimensionality manageable). No mt axis, so this
    # stays small even at fine binning.
    results.append(
        df.HistoBoost(
            "lep_pt_eta",
            [
                axis_pt_fine,
                axis_eta_fine,
                axis_charge,
                binning.axis_passIso,
                binning.axis_passMT,
            ],
            ["Lep_pt", "Lep_eta", "Lep_charge", "passIso", "passMT", "nominal_weight"],
        )
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
            ["MET_corr_rec_pt", "passIso", "passMT", "nominal_weight"],
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
