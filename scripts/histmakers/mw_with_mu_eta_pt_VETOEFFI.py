import os

from utilities import common, parsing
from wremnants.datasets.datagroups import Datagroups
from wums import logging

analysis_label = Datagroups.analysisLabel(
    os.path.basename(__file__).replace("_VETOEFFI", "")
)
parser, initargs = parsing.common_parser(analysis_label)

import os

import hist
import ROOT

import narf
from wremnants import (
    muon_calibration,
    muon_selections,
    pileup,
    theory_corrections,
    theory_tools,
    vertex,
)
from wremnants.datasets.dataset_tools import getDatasets
from wremnants.histmaker_tools import (
    aggregate_groups,
    scale_to_data,
    write_analysis_output,
)

parser.add_argument(
    "--vetoGenPartPt",
    type=float,
    default=0.0,
    help="Minimum pT for the postFSR gen muon",
)
args = parser.parse_args()
logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

args = parser.parse_args()

thisAnalysis = ROOT.wrem.AnalysisType.Wmass

era = args.era
datasets = getDatasets(
    maxFiles=args.maxFiles,
    filt=args.filterProcs,
    excl=args.excludeProcs,
    aux=args.auxiliaryProcs,
    nanoVersion="v9",
    base_path=args.dataPath,
    oneMCfileEveryN=args.oneMCfileEveryN,
    extended="msht20an3lo" not in args.pdfs,
    era=era,
)

# custom template binning
template_neta = int(args.eta[0])
template_mineta = args.eta[1]
template_maxeta = args.eta[2]
logger.info(
    f"Eta binning: {template_neta} bins from {template_mineta} to {template_maxeta}"
)
template_npt = int(args.pt[0])
template_minpt = args.pt[1]
template_maxpt = args.pt[2]
logger.info(
    f"Pt binning: {template_npt} bins from {template_minpt} to {template_maxpt}"
)

# standard regular axes
axis_eta = hist.axis.Regular(
    template_neta,
    template_mineta,
    template_maxeta,
    name="eta",
    overflow=False,
    underflow=False,
)
axis_pt = hist.axis.Regular(
    template_npt,
    template_minpt,
    template_maxpt,
    name="pt",
    overflow=False,
    underflow=False,
)
axis_charge = common.axis_charge
axis_passVeto = hist.axis.Boolean(name="passVeto")

nominal_axes = [axis_eta, axis_pt, axis_charge, axis_passVeto]
nominal_cols = [
    "postFSRmuon_eta0",
    "postFSRmuon_pt0",
    "postFSRmuon_charge0",
    "passVeto",
]

# sum those groups up in post processing
groups_to_aggregate = args.aggregateGroups

# qcdScaleByHelicity_helper = theory_corrections.make_qcd_uncertainty_helper_by_helicity()

logger.info("Running with no scale factors")

pileup_helper = pileup.make_pileup_helper(era=era)
vertex_helper = vertex.make_vertex_helper(era=era)

calib_filepaths = common.calib_filepaths
closure_filepaths = common.closure_filepaths

(
    mc_jpsi_crctn_helper,
    data_jpsi_crctn_helper,
    jpsi_crctn_MC_unc_helper,
    jpsi_crctn_data_unc_helper,
) = muon_calibration.make_jpsi_crctn_helpers(
    args, calib_filepaths, make_uncertainty_helper=True
)

mc_calibration_helper, data_calibration_helper, calibration_uncertainty_helper = (
    muon_calibration.make_muon_calibration_helpers(args)
)

smearing_helper, smearing_uncertainty_helper = (
    (None, None) if args.noSmearing else muon_calibration.make_muon_smearing_helpers()
)

bias_helper = (
    muon_calibration.make_muon_bias_helpers(args) if args.biasCalibration else None
)

theory_corrs = [*args.theoryCorr, *args.ewTheoryCorr]
procsWithTheoryCorr = [d.name for d in datasets if d.name in common.vprocs]
if len(procsWithTheoryCorr):
    corr_helpers = theory_corrections.load_corr_helpers(
        procsWithTheoryCorr, theory_corrs
    )
else:
    corr_helpers = {}

smearing_weights_procs = []


def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")
    results = []
    isW = dataset.name in common.wprocs
    isWmunu = dataset.name in ["Wplusmunu_2016PostVFP", "Wminusmunu_2016PostVFP"]
    isZ = dataset.name in common.zprocs
    isWorZ = isW or isZ
    isTop = dataset.group == "Top"

    cvh_helper = data_calibration_helper if dataset.is_data else mc_calibration_helper
    jpsi_helper = data_jpsi_crctn_helper if dataset.is_data else mc_jpsi_crctn_helper

    df = df.Define("weight", "std::copysign(1.0, genWeight)")
    weightsum = df.SumAndCount("weight")
    df = df.Define("isEvenEvent", "event % 2 == 0")

    axes = nominal_axes
    cols = nominal_cols

    df = muon_calibration.define_corrected_muons(
        df, cvh_helper, jpsi_helper, args, dataset, smearing_helper, bias_helper
    )

    # gen match to bare muons to select only prompt muons from MC processes, but also including tau decays
    # status flags in NanoAOD: https://cms-nanoaod-integration.web.cern.ch/autoDoc/NanoAODv9/2016ULpostVFP/doc_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1.html
    postFSRmuonDef = f"GenPart_status == 1 && (GenPart_statusFlags & 1 || GenPart_statusFlags & (1 << 5)) && abs(GenPart_pdgId) == 13 && GenPart_pt > {args.vetoGenPartPt}"
    df = df.Define("postFSRmuons", postFSRmuonDef)

    # restrict to one gen muon for simplicity
    df = df.Filter("Sum(postFSRmuons) == 1")
    df = muon_selections.veto_electrons(df)
    df = muon_selections.apply_met_filters(df)

    ########################################################################
    # define event weights here since they are needed below for some helpers
    df = df.Define("weight_pu", pileup_helper, ["Pileup_nTrueInt"])
    df = df.Define("weight_vtx", vertex_helper, ["GenVtx_z", "Pileup_nTrueInt"])

    weight_expr = "weight_pu*L1PreFiringWeight_ECAL_Nom"
    if not args.noVertexWeight:
        weight_expr += "*weight_vtx"

    logger.debug(f"Exp weight defined: {weight_expr}")
    df = df.Define("exp_weight", weight_expr)
    df = theory_tools.define_theory_weights_and_corrs(
        df, dataset.name, corr_helpers, args
    )

    ########################################################################

    df = df.Define("postFSRmuon_pt0", "GenPart_pt[postFSRmuons][0]")
    df = df.Define("postFSRmuon_eta0", "GenPart_eta[postFSRmuons][0]")
    df = df.Define("postFSRmuon_phi0", "GenPart_phi[postFSRmuons][0]")
    df = df.Define(
        "postFSRmuon_charge0", "-1 * std::copysign(1.0, GenPart_pdgId[postFSRmuons][0])"
    )

    df = muon_selections.select_veto_muons(
        df,
        nMuons=0,
        condition=">=",
        ptCut=args.vetoRecoPt,
        etaCut=3.0,
        useGlobalOrTrackerVeto=False,
        tightGlobalOrTracker=True,
    )
    # might have more veto muons, but will look for at least one gen matched to the only gen muon
    df = df.Define("oneOrMoreVetoMuons", "Sum(vetoMuons) > 0")
    df = df.Define(
        "passVeto",
        "oneOrMoreVetoMuons && wrem::hasMatchDR2(postFSRmuon_eta0,postFSRmuon_phi0,Muon_eta[vetoMuons],Muon_phi[vetoMuons],0.09)",
    )

    nominal = df.HistoBoost("nominal", axes, [*cols, "nominal_weight"])
    results.append(nominal)

    return results, weightsum


resultdict = narf.build_and_run(datasets, build_graph)

if not args.noScaleToData:
    scale_to_data(resultdict)
    aggregate_groups(datasets, resultdict, groups_to_aggregate)

write_analysis_output(
    resultdict, f"{os.path.basename(__file__).replace('py', 'hdf5')}", args
)
