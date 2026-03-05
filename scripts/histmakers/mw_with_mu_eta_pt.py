import os

from utilities import common, differential, parsing
from utilities.common import data_dir
from wremnants.datasets.datagroups import Datagroups
from wums import logging

analysis_label = Datagroups.analysisLabel(os.path.basename(__file__))
parser, initargs = parsing.common_parser(analysis_label)

import math

import hist
import numpy as np
import ROOT

import narf
from utilities import common
from wremnants import (
    helicity_utils,
    muon_calibration,
    muon_efficiencies_binned,
    muon_efficiencies_newVeto,
    muon_efficiencies_smooth,
    muon_efficiencies_veto,
    muon_prefiring,
    muon_selections,
    muon_validation,
    pileup,
    syst_tools,
    theory_corrections,
    theory_tools,
    theoryAgnostic_tools,
    unfolding_tools,
    vertex,
)
from wremnants.datasets.dataset_tools import getDatasets
from wremnants.helicity_utils_polvar import makehelicityWeightHelper_polvar
from wremnants.histmaker_tools import (
    aggregate_groups,
    define_norm_weight_nRecoVtx,
    get_run_lumi_edges,
    make_muon_phi_axis,
    scale_to_data,
    write_analysis_output,
)

parser.add_argument(
    "--noGenMatchMC",
    action="store_true",
    help="Don't use gen match filter for prompt muons with MC samples (note: QCD MC never has it anyway)",
)
parser.add_argument(
    "--makeMCefficiency",
    action="store_true",
    help="Save yields vs eta-pt-ut-passMT-passIso-passTrigger to derive 3D efficiencies for MC isolation and trigger (can run also with --onlyMainHistograms)",
)
parser.add_argument(
    "--onlyTheorySyst",
    action="store_true",
    help="Keep only theory systematic variations, mainly for tests",
)
parser.add_argument(
    "--noAuxiliaryHistograms",
    action="store_true",
    help="Remove auxiliary histograms to save memory (removed by default with --unfolding or --theoryAgnostic)",
)
parser.add_argument(
    "--fineMtBinning",
    action="store_true",
    help="Use finer binning for the transverse mass axis of the nominal histogram",
)
parser.add_argument(
    "--mtCut",
    type=int,
    default=common.get_default_mtcut(analysis_label),
    help="Value for the transverse mass cut in the event selection",
)
parser.add_argument(
    "--noMtGenCut",
    action="store_true",
    help="Don't apply an mt cut at generator level (by default the same cut is applied at generator level as at detector level)",
)
parser.add_argument(
    "--vetoGenPartPt",
    type=float,
    default=15.0,
    help="Minimum pT for the postFSR gen muon when defining the variation of the veto efficiency",
)
parser.add_argument(
    "--vetoGenPartEta",
    type=float,
    default=2.4,
    help="Maximum absolute eta for the postFSR gen muon when defining the variation of the veto efficiency",
)
parser.add_argument(
    "--selectVetoEventsMC",
    action="store_true",
    help="Select events which fail the veto, by enforcing at least two prompt postFSR muons in acceptance",
)
parser.add_argument(
    "--noTrigger",
    action="store_true",
    help="Just for test: remove trigger HLT bit selection and trigger matching (should also remove scale factors with --noScaleFactors for it to make sense)",
)
parser.add_argument(
    "--selectNonPromptFromSV",
    action="store_true",
    help="Test: define a non-prompt muon enriched control region",
)
parser.add_argument(
    "--selectNonPromptFromLightMesonDecay",
    action="store_true",
    help="Test: define a non-prompt muon enriched control region with muons from light meson decays",
)
parser.add_argument(
    "--useGlobalOrTrackerVeto",
    action="store_true",
    help="Use global-or-tracker veto definition and scale factors instead of global only",
)
parser.add_argument(
    "--useRefinedVeto",
    action="store_true",
    help="Temporary option, it uses a different computation of the veto SF (only implemented for global muons)",
)
parser.add_argument(
    "--noVetoSF", action="store_true", help="Don't use SF for the veto, for tests"
)
parser.add_argument(
    "--scaleDYvetoFraction",
    type=float,
    default=-1.0,
    help="Scale fraction of DY background that should receive veto SF by this amount. Negative values do nothing",
)
parser.add_argument(
    "--addAxisSignUt",
    action="store_true",
    help="Add another fit axis with the sign of the uT recoil projection",
)
parser.add_argument(
    "--utAngleCosineCut",
    nargs=2,
    type=float,
    default=None,
    help="Cut on the cosine of the angle between the lepton and the boson",
)
parser.add_argument(
    "--useTnpMuonVarForSF",
    action="store_true",
    help="To read efficiency scale factors, use the same muon variables as used to measure them with tag-and-probe (by default the final corrected ones are used)",
)
parser.add_argument(
    "--forceValidCVH",
    action="store_true",
    help="When not applying muon scale corrections (--muonCorrData none / --muonCorrMC none), require at list that the CVH corrected variables are valid",
)
args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

useGlobalOrTrackerVeto = args.useGlobalOrTrackerVeto

if args.randomizeDataByRun and not args.addRunAxis:
    raise ValueError("Options --randomizeDataByRun only works with --addRunAxis.")

if args.selectNonPromptFromLightMesonDecay and args.selectNonPromptFromSV:
    raise ValueError(
        "Options --selectNonPromptFromSV and --selectNonPromptFromLightMesonDecay cannot be used together."
    )

if args.useRefinedVeto and args.useGlobalOrTrackerVeto:
    raise NotImplementedError(
        "Options --useGlobalOrTrackerVeto and --useRefinedVeto cannot be used together at the moment."
    )

if args.useRefinedVeto:
    pass
else:
    pass

if args.unfolding or args.theoryAgnostic:
    if args.theoryAgnostic:
        if args.theoryAgnosticGenAbsYVbinEdges and any(
            x < 0.0 for x in args.theoryAgnosticGenAbsYVbinEdges
        ):
            raise ValueError(
                "Option --genAbsYVbinEdges requires all positive values. Please check"
            )

# axes for W MC efficiencies with uT dependence for iso and trigger
axis_pt_eff_list = [
    24.0,
    26.0,
    28.0,
    30.0,
    32.0,
    34.0,
    36.0,
    38.0,
    40.0,
    42.0,
    44.0,
    47.0,
    50.0,
    55.0,
    60.0,
    65.0,
]
axis_pt_eff = hist.axis.Variable(
    axis_pt_eff_list, name="pt", overflow=False, underflow=False
)
if args.makeMCefficiency:
    # override the pt cuts (the binning is irrelevant since a different pt axis is used)
    nbinsPtEff = axis_pt_eff_list[-1] - axis_pt_eff_list[0]
    parser = parsing.set_parser_default(
        parser, "pt", [nbinsPtEff, axis_pt_eff_list[0], axis_pt_eff_list[-1]]
    )

args = parser.parse_args()

thisAnalysis = ROOT.wrem.AnalysisType.Wmass
isoBranch = muon_selections.getIsoBranch(args.isolationDefinition)
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

# transverse boson mass cut
mtw_min = args.mtCut

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
axis_phi = hist.axis.Regular(50, -math.pi, math.pi, name="phi", circular=True)
axis_muonJetPt = hist.axis.Regular(
    50, 26, 76, name="muonJetPt", underflow=False, overflow=True
)

axis_charge = common.axis_charge
axis_passIso = common.axis_passIso
axis_passMT = common.axis_passMT
axis_mt = hist.axis.Variable(
    (*np.arange(0, mtw_min + 2, 2), *np.arange(mtw_min + 5, 95, 5), 100, 120),
    name="mt",
    underflow=False,
    overflow=True,
)
axis_met = hist.axis.Regular(25, 0.0, 100.0, name="met", underflow=False, overflow=True)

# for mt, met, ptW plots, to compute the fakes properly (but FR pretty stable vs pt and also vs eta)
# may not exactly reproduce the same pt range as analysis, though
axis_fakes_eta = hist.axis.Regular(
    round((template_maxeta - template_mineta) * 10 / 2),
    args.eta[1],
    args.eta[2],
    name="eta",
    underflow=False,
    overflow=False,
)

axis_fakes_pt = hist.axis.Variable(
    common.get_binning_fakes_pt(template_minpt, template_maxpt),
    name="pt",
    overflow=False,
    underflow=False,
)

axis_mtCat = hist.axis.Variable(
    common.get_binning_fakes_mt(
        mtw_min, high_mt_bins=False, fine_mt_binning=args.fineMtBinning
    ),
    name="mt",
    underflow=False,
    overflow=True,
)
axis_isoCat = hist.axis.Variable(
    common.get_binning_fakes_relIso(high_iso_bins=False),
    name="relIso",
    underflow=False,
    overflow=True,
)
axes_abcd = [axis_mtCat, axis_isoCat]

axis_ut_analysis = hist.axis.Regular(
    2, -2, 2, underflow=False, overflow=False, name="utAngleSign"
)  # used only to separate positive/negative uT for now

if args.addAxisSignUt:
    axes_fakerate = [
        axis_fakes_eta,
        axis_fakes_pt,
        axis_charge,
        axis_ut_analysis,
        *axes_abcd,
    ]
    columns_fakerate = [
        "goodMuons_eta0",
        "goodMuons_pt0",
        "goodMuons_charge0",
        "goodMuons_angleSignUt0",
        "transverseMass",
        "goodMuons_relIso0",
    ]

    nominal_axes = [axis_eta, axis_pt, axis_charge, axis_ut_analysis, *axes_abcd]
    nominal_cols = columns_fakerate
else:
    axes_fakerate = [axis_fakes_eta, axis_fakes_pt, axis_charge, *axes_abcd]
    columns_fakerate = [
        "goodMuons_eta0",
        "goodMuons_pt0",
        "goodMuons_charge0",
        "transverseMass",
        "goodMuons_relIso0",
    ]

    nominal_axes = [axis_eta, axis_pt, axis_charge, *axes_abcd]
    nominal_cols = columns_fakerate

if args.nToysMC > 0:
    axis_toys = hist.axis.Integer(
        0, args.nToysMC, underflow=False, overflow=False, name="toys"
    )
    nominal_axes = [*nominal_axes, axis_toys]
    nominal_cols = [*nominal_cols, "toyIdxs"]

# auxiliary axes
axis_iso = hist.axis.Regular(100, 0, 25, name="iso", underflow=False, overflow=True)
axis_relIso = hist.axis.Regular(
    100, 0, 1, name="relIso", underflow=False, overflow=True
)

axis_passTrigger = hist.axis.Boolean(name="passTrigger")

# following axis is for MC truth efficiencies
axis_ut = hist.axis.Regular(40, -100, 100, overflow=True, underflow=True, name="ut")
# next axis is for dedicated studies
axis_ut_fine = hist.axis.Regular(
    70, -40.0, 100.0, name="ut", underflow=True, overflow=True
)
axis_uTAngleCosine = hist.axis.Regular(
    20, -1, 1, name="uTAngleCosine", overflow=False, underflow=False
)

# sum those groups up in post processing
groups_to_aggregate = args.aggregateGroups

if args.unfolding:
    # first and last pT bins are merged into under and overflow
    template_wpt = (template_maxpt - template_minpt) / args.unfoldingBins[1]
    min_pt_unfolding = template_minpt + template_wpt
    max_pt_unfolding = template_maxpt - template_wpt
    npt_unfolding = args.unfoldingBins[1] - 2

    unfolding_axes = {}
    unfolding_cols = {}
    for level in args.unfoldingLevels:
        # for poi as noi, need gen eta overflow bin and out of acceptance axes to keep all events and be able to reconstruct corresponding reco histogram
        a, c = differential.get_pt_eta_charge_axes(
            level,
            npt_unfolding,
            min_pt_unfolding,
            max_pt_unfolding,
            args.unfoldingBins[0] if "absEtaGen" in args.unfoldingAxes else None,
            flow_eta=args.poiAsNoi,
            add_out_of_acceptance_axis=args.poiAsNoi,
        )
        unfolding_axes[level] = a
        unfolding_cols[level] = c

        if not args.poiAsNoi:
            datasets = unfolding_tools.add_out_of_acceptance(datasets, group="Wmunu")
            # datasets = unfolding_tools.add_out_of_acceptance(datasets, group = "Wtaunu")
            if len(args.unfoldingLevels) > 1:
                logger.warning(
                    f"Exact unfolding with multiple gen level definitions is not possible, take first one: {args.unfoldingLevels[0]} and continue."
                )
                break

    # for simultaneous W and Z unfolding, only implemented for ptll-yll
    if args.unfoldingInclusive:
        cutsmap = {"fiducial": "masswindow"}
    else:
        mass_min, mass_max = common.get_default_mz_window()
        cutsmap = {
            "pt_min": args.pt[1],
            "pt_max": args.pt[2],
            "abseta_max": args.eta[2],
            "mass_min": mass_min,
            "mass_max": mass_max,
        }

    unfolder_z = unfolding_tools.UnfolderZ(
        reco_axes_edges={
            "ptll": common.ptZ_binning,
            "yll": common.yll_20quantiles_binning,
        },
        unfolding_axes_names=["ptVGen", "absYVGen", "helicitySig"],
        unfolding_levels=args.unfoldingLevels,
        poi_as_noi=args.poiAsNoi,
        fitresult=args.fitresult,
        fitresult_channel="ch0_masked",
        cutsmap=cutsmap,
    )

    if args.fitresult:
        unfolding_corr_helper = unfolding_tools.reweight_to_fitresult(
            args.fitresult, channel="ch1_masked"
        )

theory_helpers_procs = theory_corrections.make_theory_helpers(
    args.pdfs, args.theoryCorr, procs=["Z", "W"]
)

if args.theoryAgnostic:
    theoryAgnostic_axes, theoryAgnostic_cols = differential.get_theoryAgnostic_axes(
        ptV_bins=args.theoryAgnosticGenPtVbinEdges,
        absYV_bins=args.theoryAgnosticGenAbsYVbinEdges,
        ptV_flow=args.poiAsNoi,
        absYV_flow=args.poiAsNoi,
    )
    axis_helicity = helicity_utils.axis_helicity_multidim
    # the following just prepares the existence of the group for out-of-acceptance signal, but doesn't create or define the histogram yet
    if not args.poiAsNoi or (
        args.theoryAgnosticPolVar and args.theoryAgnosticSplitOOA
    ):  # this splitting is not needed for the normVar version of the theory agnostic
        datasets = unfolding_tools.add_out_of_acceptance(datasets, group="Wmunu")
        datasets = unfolding_tools.add_out_of_acceptance(datasets, group="Zmumu")
        groups_to_aggregate.append("WmunuOOA")

# axes for study of fakes
axis_mt_fakes = hist.axis.Regular(
    120, 0.0, 120.0, name="mt", underflow=False, overflow=True
)
axis_dphi_fakes = hist.axis.Regular(
    8, 0.0, np.pi, name="DphiMuonMet", underflow=False, overflow=False
)
axis_hasjet_fakes = hist.axis.Boolean(
    name="hasJets"
)  # only need case with 0 jets or > 0 for now
mTStudyForFakes_axes = [
    axis_eta,
    axis_pt,
    axis_charge,
    axis_mt_fakes,
    axis_passIso,
    axis_hasjet_fakes,
    axis_dphi_fakes,
]

axis_met = hist.axis.Regular(
    100, 0.0, 200.0, name="met", underflow=False, overflow=True
)
axis_recoWpt = hist.axis.Regular(
    40, 0.0, 80.0, name="recoWpt", underflow=False, overflow=True
)

# define helpers
muon_prefiring_helper, muon_prefiring_helper_stat, muon_prefiring_helper_syst = (
    muon_prefiring.make_muon_prefiring_helpers(era=era)
)

if args.noScaleFactors:
    logger.info("Running with no scale factors")
elif args.binnedScaleFactors:
    logger.info("Using binned scale factors and uncertainties")
    # add usePseudoSmoothing=True for tests with Asimov
    muon_efficiency_helper, muon_efficiency_helper_syst, muon_efficiency_helper_stat = (
        muon_efficiencies_binned.make_muon_efficiency_helpers_binned(
            filename=data_dir + "/muonSF/allSmooth_GtoH3D.root",
            era=era,
            max_pt=axis_pt.edges[-1],
            usePseudoSmoothing=True,
        )
    )
else:
    logger.info("Using smoothed scale factors and uncertainties")
    muon_efficiency_helper, muon_efficiency_helper_syst, muon_efficiency_helper_stat = (
        muon_efficiencies_smooth.make_muon_efficiency_helpers_smooth(
            filename=args.sfFile,
            era=era,
            what_analysis=thisAnalysis,
            max_pt=axis_pt.edges[-1],
            isoEfficiencySmoothing=args.isoEfficiencySmoothing,
            smooth3D=args.smooth3dsf,
            isoDefinition=args.isolationDefinition,
        )
    )
    if not args.noVetoSF:
        if args.useRefinedVeto:
            (
                muon_efficiency_veto_helper,
                muon_efficiency_veto_helper_syst,
                muon_efficiency_veto_helper_stat,
            ) = muon_efficiencies_newVeto.make_muon_efficiency_helpers_newVeto(
                antiveto=True, era=era
            )
        else:
            (
                muon_efficiency_veto_helper,
                muon_efficiency_veto_helper_syst,
                muon_efficiency_veto_helper_stat,
            ) = muon_efficiencies_veto.make_muon_efficiency_helpers_veto(
                useGlobalOrTrackerVeto=useGlobalOrTrackerVeto, era=era
            )

logger.info(f"SF file: {args.sfFile}")

muon_efficiency_helper_syst_altBkg = {}
if not args.noScaleFactors:
    for es in common.muonEfficiency_altBkgSyst_effSteps:
        altSFfile = args.sfFile.replace(".root", "_altBkg.root")
        logger.info(f"Additional SF file for alternate syst with {es}: {altSFfile}")
        muon_efficiency_helper_syst_altBkg[es] = (
            muon_efficiencies_smooth.make_muon_efficiency_helpers_smooth_altSyst(
                filename=altSFfile,
                era=era,
                what_analysis=thisAnalysis,
                max_pt=axis_pt.edges[-1],
                effStep=es,
            )
        )

pileup_helper = pileup.make_pileup_helper(era=era)
vertex_helper = vertex.make_vertex_helper(era=era)

calib_filepaths = common.calib_filepaths
closure_filepaths = common.closure_filepaths

diff_weights_helper = (
    ROOT.wrem.SplinesDifferentialWeightsHelper(calib_filepaths["tflite_file"])
    if (args.muonScaleVariation == "smearingWeightsSplines" or args.validationHists)
    else None
)

(
    mc_jpsi_crctn_helper,
    data_jpsi_crctn_helper,
    jpsi_crctn_MC_unc_helper,
    jpsi_crctn_data_unc_helper,
) = muon_calibration.make_jpsi_crctn_helpers(
    args, calib_filepaths, make_uncertainty_helper=True
)

z_non_closure_parametrized_helper, z_non_closure_binned_helper = (
    muon_calibration.make_Z_non_closure_helpers(
        args, calib_filepaths, closure_filepaths
    )
)

mc_calibration_helper, data_calibration_helper, calibration_uncertainty_helper = (
    muon_calibration.make_muon_calibration_helpers(args, era=era)
)

closure_unc_helper = muon_calibration.make_closure_uncertainty_helper(
    common.closure_filepaths["parametrized"]
)
closure_unc_helper_A = muon_calibration.make_uniform_closure_uncertainty_helper(
    0, common.correlated_variation_base_size["A"]
)
closure_unc_helper_M = muon_calibration.make_uniform_closure_uncertainty_helper(
    2, common.correlated_variation_base_size["M"]
)

smearing_helper, smearing_uncertainty_helper = (
    (None, None) if args.noSmearing else muon_calibration.make_muon_smearing_helpers()
)

bias_helper = (
    muon_calibration.make_muon_bias_helpers(args) if args.biasCalibration else None
)

(
    pixel_multiplicity_helper,
    pixel_multiplicity_uncertainty_helper,
    pixel_multiplicity_uncertainty_helper_stat,
) = muon_calibration.make_pixel_multiplicity_helpers(
    reverse_variations=args.reweightPixelMultiplicity
)


theory_corrs = [*args.theoryCorr, *args.ewTheoryCorr]
procsWithTheoryCorr = [d.name for d in datasets if d.name in common.vprocs]
if len(procsWithTheoryCorr):
    corr_helpers = theory_corrections.load_corr_helpers(
        procsWithTheoryCorr, theory_corrs
    )
else:
    corr_helpers = {}

# For polynominal variations
if args.theoryAgnosticPolVar:
    theoryAgnostic_helpers_minus = makehelicityWeightHelper_polvar(
        genVcharge=-1,
        fileTag=args.theoryAgnosticFileTag,
        filePath=args.theoryAgnosticFilePath,
    )
    theoryAgnostic_helpers_plus = makehelicityWeightHelper_polvar(
        genVcharge=1,
        fileTag=args.theoryAgnosticFileTag,
        filePath=args.theoryAgnosticFilePath,
    )

# Helper for muR and muF as polynomial variations
muRmuFPolVar_helpers_minus = makehelicityWeightHelper_polvar(
    genVcharge=-1,
    fileTag=args.muRmuFPolVarFileTag,
    filePath=args.muRmuFPolVarFilePath,
    noUL=True,
)
muRmuFPolVar_helpers_plus = makehelicityWeightHelper_polvar(
    genVcharge=1,
    fileTag=args.muRmuFPolVarFileTag,
    filePath=args.muRmuFPolVarFilePath,
    noUL=True,
)
muRmuFPolVar_helpers_Z = makehelicityWeightHelper_polvar(
    genVcharge=0,
    fileTag=args.muRmuFPolVarFileTag,
    filePath=args.muRmuFPolVarFilePath,
    noUL=True,
)

# recoil initialization
if not args.noRecoil:
    from wremnants import recoil_tools

    recoilHelper = recoil_tools.Recoil("highPU", args, flavor="mu")

seed_data = 2 * args.randomSeedForToys
seed_mc = 2 * args.randomSeedForToys + 1

if args.nToysMC > 0:
    toy_helper_data = ROOT.wrem.ToyHelper(
        args.nToysMC, seed_data, 1, ROOT.ROOT.GetThreadPoolSize()
    )
    toy_helper_mc = ROOT.wrem.ToyHelper(
        args.nToysMC,
        seed_mc,
        args.varianceScalingForToys,
        ROOT.ROOT.GetThreadPoolSize(),
    )

smearing_weights_procs = []


def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")
    results = []
    isW = dataset.group in ["Wmunu", "Wtaunu"]
    isBSM = dataset.name.startswith("WtoNMu")
    isWmunu = isBSM or dataset.group in ["Wmunu"]
    isZ = dataset.group in [
        "Zmumu",
        "Ztautau",
    ]
    isZveto = isZ or dataset.group in ["DYlowMass"]
    isWorZ = isW or isZ
    isTop = dataset.group == "Top"
    isQCDMC = dataset.group == "QCD"
    require_prompt = "tau" not in dataset.name  # for muon GEN-matching
    storage_type = (
        hist.storage.Double()
    )  # turn off sum weight square for systematic histograms

    theory_helpers = {}
    if isWorZ:
        theory_helpers = theory_helpers_procs[dataset.name[0]]

    # disable auxiliary histograms when unfolding to reduce memory consumptions, or when doing the original theory agnostic without --poiAsNoi
    auxiliary_histograms = True
    if args.noAuxiliaryHistograms:
        auxiliary_histograms = False

    apply_theory_corr = theory_corrs and dataset.name in corr_helpers

    cvh_helper = data_calibration_helper if dataset.is_data else mc_calibration_helper
    jpsi_helper = data_jpsi_crctn_helper if dataset.is_data else mc_jpsi_crctn_helper

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    df = df.DefinePerSample("unity", "1.0")

    if args.nToysMC > 0:
        if dataset.is_data:
            df = df.Define("toyIdxs", toy_helper_data, ["rdfslot_"])
        else:
            df = df.Define("toyIdxs", toy_helper_mc, ["rdfslot_"])

    df = df.Define("isEvenEvent", "event % 2 == 0")

    weightsum = df.SumAndCount("weight")

    axes = nominal_axes
    cols = nominal_cols

    if args.addMuonPhiAxis is not None:
        axes = [*axes, make_muon_phi_axis(args.addMuonPhiAxis)]
        cols = [*cols, "goodMuons_phi0"]

    if args.addNvtxAxis is not None:
        axes = [
            *axes,
            hist.axis.Variable(
                np.array(args.addNvtxAxis),
                name="nRecoVtx",
                underflow=False,
                overflow=False,
            ),
        ]
        cols = [*cols, "PV_npvsGood"]

    if args.addRunAxis:
        run_edges, lumi_edges = get_run_lumi_edges(args.nRunBins, era)
        run_bin_centers = [
            int(0.5 * (run_edges[i + 1] + run_edges[i]))
            for i in range(len(run_edges) - 1)
        ]
        # lumi_fractions = [(lumi_edges[i+1] + lumi_edges[i]) for i in range(len(lumi_edges) - 1)] # [0.25749, 0.22264, 0.24941, 0.27046]
        axes = [
            *axes,
            hist.axis.Variable(
                np.array(run_edges) + 0.5, name="run", underflow=False, overflow=False
            ),
        ]
        df = df.DefinePerSample(
            "lumiEdges",
            "ROOT::VecOps::RVec<double> res = {"
            + ",".join([str(x) for x in lumi_edges])
            + "}; return res;",
        )
        df = df.DefinePerSample(
            "runVals",
            "ROOT::VecOps::RVec<unsigned int> res = {"
            + ",".join([str(x) for x in run_bin_centers])
            + "}; return res;",
        )
        if dataset.is_data:
            if args.randomizeDataByRun:
                df = df.Define(
                    "run4axis",
                    "wrem::get_dummy_run_by_lumi_quantile(run, luminosityBlock, event, lumiEdges, runVals)",
                )
            else:
                df = df.Alias("run4axis", "run")
        else:
            df = df.Define(
                "run4axis",
                "wrem::get_dummy_run_by_lumi_quantile(run, luminosityBlock, event, lumiEdges, runVals)",
            )
        cols = [*cols, "run4axis"]

    if isBSM or isWmunu:
        # to compute inclusive cross section
        unfolding_tools.add_xnorm_histograms(
            results,
            df,
            args,
            dataset.name,
            corr_helpers,
            theory_helpers,
            [],
            [],
            base_name="gen",
        )

    if args.unfolding:
        if isWmunu:
            df = unfolding_tools.define_gen_level(
                df, dataset.name, args.unfoldingLevels, mode=analysis_label
            )

            cutsmap = {
                "pt_min": template_minpt,
                "pt_max": template_maxpt,
                "abseta_max": template_maxeta,
                "mtw_min": None if args.noMtGenCut else args.mtCut,
            }
            if hasattr(dataset, "out_of_acceptance"):
                df = unfolding_tools.select_fiducial_space(
                    df,
                    args.unfoldingLevels[0],
                    mode=analysis_label,
                    accept=False,
                    **cutsmap,
                )
            else:
                for level in args.unfoldingLevels:
                    df = unfolding_tools.select_fiducial_space(
                        df,
                        level,
                        mode=analysis_label,
                        accept=True,
                        select=not args.poiAsNoi,
                        **cutsmap,
                    )

                if args.fitresult:
                    logger.debug("Apply reweighting based on unfolded result")
                    df = df.Define(
                        "unfoldingWeight_tensor",
                        unfolding_corr_helper,
                        [*unfolding_corr_helper.hist.axes.name[:-1], "unity"],
                    )
                    df = df.Define(
                        "central_weight",
                        f"{unfolding_corr_helper.level}_acceptance ? unfoldingWeight_tensor(0) : unity",
                    )

                for level in args.unfoldingLevels:
                    # add full phase space histograms for inclusive cross section
                    unfolding_tools.add_xnorm_histograms(
                        results,
                        df,
                        args,
                        dataset.name,
                        corr_helpers,
                        theory_helpers,
                        [a for a in unfolding_axes[level] if a.name != "acceptance"],
                        [
                            c
                            for c in unfolding_cols[level]
                            if c != f"{level}_acceptance"
                        ],
                        base_name=f"{level}_full",
                    )

                    if args.poiAsNoi:
                        df_xnorm = df.Filter(f"{level}_acceptance")
                    else:
                        df_xnorm = df

                    unfolding_tools.add_xnorm_histograms(
                        results,
                        df_xnorm,
                        args,
                        dataset.name,
                        corr_helpers,
                        theory_helpers,
                        [a for a in unfolding_axes[level] if a.name != "acceptance"],
                        [
                            c
                            for c in unfolding_cols[level]
                            if c != f"{level}_acceptance"
                        ],
                        base_name=level,
                    )
                    if not args.poiAsNoi:
                        axes = [*nominal_axes, *unfolding_axes[level]]
                        cols = [*nominal_cols, *unfolding_cols[level]]
                        break

        elif dataset.name == "Zmumu_2016PostVFP":
            if args.unfolding and dataset.name == "Zmumu_2016PostVFP":
                df = unfolder_z.add_gen_histograms(
                    args, df, results, dataset, corr_helpers, theory_helpers
                )

                if not unfolder_z.poi_as_noi:
                    axes = [
                        *nominal_axes,
                        *unfolder_z.unfolding_axes[unfolder_z.unfolding_levels[-1]],
                    ]
                    cols = [
                        *nominal_cols,
                        *unfolder_z.unfolding_cols[unfolder_z.unfolding_levels[-1]],
                    ]

    if isWorZ:
        df = theory_tools.define_prefsr_vars(df)
        df = df.Define(
            "qtOverQ", "ptVgen/massVgen"
        )  # FIXME: should there be a protection against mass=0 and what value to use?

    if args.theoryAgnostic and isWmunu:  # should be isW to do also Wtaunu
        usePtOverM = False
        if args.theoryAgnosticPolVar:
            OOAthresholds = args.theoryAgnosticFileTag.split("_")
            ptVthresholdOOA = float(OOAthresholds[0].replace("x", "").replace("p", "."))
            absyVthresholdOOA = float(
                OOAthresholds[1].replace("y", "").replace("p", ".")
            )
            usePtOverM = True
        if hasattr(dataset, "out_of_acceptance"):
            logger.debug("Reject events in fiducial phase space")
            df = theoryAgnostic_tools.select_fiducial_space(
                df,
                ptVthresholdOOA,
                absyVthresholdOOA,
                accept=False,
                select=True,
                usePtOverM=usePtOverM,
            )
        else:
            # the in-acceptance selection must usually not be used to filter signal events when doing POIs as NOIs
            if args.theoryAgnosticPolVar and args.theoryAgnosticSplitOOA:
                logger.debug(
                    "Select events in fiducial phase space for theory agnostic analysis"
                )
                df = theoryAgnostic_tools.select_fiducial_space(
                    df,
                    ptVthresholdOOA,
                    absyVthresholdOOA,
                    accept=True,
                    select=True,
                    usePtOverM=usePtOverM,
                )

    if args.xnormOnly:
        return results, weightsum

    if not args.makeMCefficiency and not args.noTrigger:
        # remove trigger, it will be part of the efficiency selection for passing trigger
        df = df.Filter(muon_selections.hlt_string(era))

    df = muon_calibration.define_corrected_muons(
        df, cvh_helper, jpsi_helper, args, dataset, smearing_helper, bias_helper
    )

    df = muon_selections.select_veto_muons(
        df,
        nMuons=1,
        ptCut=args.vetoRecoPt,
        etaCut=args.vetoRecoEta,
        useGlobalOrTrackerVeto=useGlobalOrTrackerVeto,
    )
    df = muon_selections.select_good_muons(
        df,
        template_minpt,
        template_maxpt,
        dataset.group,
        nMuons=1,
        use_trackerMuons=args.trackerMuons,
        use_isolation=False,
        nonPromptFromSV=args.selectNonPromptFromSV,
        nonPromptFromLighMesonDecay=args.selectNonPromptFromLightMesonDecay,
        requirePixelHits=args.requirePixelHits,
    )

    # the corrected RECO muon kinematics, which is intended to be used as the nominal
    df = muon_calibration.define_corrected_reco_muon_kinematics(df)

    df = muon_selections.select_standalone_muons(
        df, dataset, args.trackerMuons, "goodMuons"
    )

    df = muon_selections.veto_electrons(df)
    df = muon_selections.apply_met_filters(df)

    if args.makeMCefficiency:
        df = df.Define(
            "GoodTrigObjs",
            f"wrem::goodMuonTriggerCandidate<wrem::Era::Era_{era}>(TrigObj_id,TrigObj_filterBits)",
        )
        hltString = muon_selections.hlt_string(era)
        df = df.Define(
            "passTrigger",
            f"{hltString} && wrem::hasTriggerMatch(goodMuons_eta0,goodMuons_phi0,TrigObj_eta[GoodTrigObjs],TrigObj_phi[GoodTrigObjs])",
        )
    elif not args.noTrigger:
        df = muon_selections.apply_triggermatching_muon(
            df, dataset, "goodMuons", era=era
        )

    if isWorZ:
        df = muon_validation.define_cvh_reco_muon_kinematics(df)
        reco_sel = "vetoMuonsPre"
        df = muon_calibration.define_genFiltered_recoMuonSel(
            df, reco_sel, require_prompt
        )
        reco_sel_GF = muon_calibration.getColName_genFiltered_recoMuonSel(
            reco_sel, require_prompt
        )
        df = muon_calibration.define_covMatFiltered_recoMuonSel(df, reco_sel_GF)
        df = muon_calibration.define_matched_gen_muons_covMat(df, reco_sel_GF)
        df = muon_calibration.define_matched_gen_muons_kinematics(df, reco_sel_GF)
        df = muon_calibration.calculate_matched_gen_muon_kinematics(df, reco_sel_GF)
        df = muon_calibration.define_matched_genSmeared_muon_kinematics(df, reco_sel_GF)
        df = muon_calibration.define_matched_reco_muon_kinematics(df, reco_sel_GF)

        reco_sel = "goodMuons"
        df = muon_calibration.define_matched_gen_muons_kinematics(df, reco_sel)
        df = muon_calibration.calculate_matched_gen_muon_kinematics(df, reco_sel)
        df = muon_calibration.define_matched_gen_muons_covMat(df, reco_sel)
        df = muon_calibration.define_matched_genSmeared_muon_kinematics(df, reco_sel)

        for var in ["Pt", "Eta", "Charge", "Qop"]:
            df = df.Define(
                f"{reco_sel}_{var.lower()}0_gen", f"{reco_sel}_gen{var.capitalize()}[0]"
            )
            df = df.Define(
                f"{reco_sel_GF}_{var.lower()}0_gen",
                f"{reco_sel_GF}_gen{var.capitalize()}[0]",
            )

            df = df.Define(
                f"{reco_sel}_{var.lower()}0_gen_smeared",
                f"{reco_sel}_genSmeared{var.capitalize()}[0]",
            )
            df = df.Define(
                f"{reco_sel_GF}_{var.lower()}0_gen_smeared",
                f"{reco_sel_GF}_genSmeared{var.capitalize()}[0]",
            )
            if var != "Qop":
                df = df.Define(
                    f"{reco_sel_GF}_{var.lower()}0_reco",
                    f"{reco_sel_GF}_reco{var.capitalize()}[0]",
                )
        df = df.Define(f"{reco_sel_GF}_covMat0", f"{reco_sel_GF}_covMat[0]")

        if args.validationHists:
            for reco_type in ["crctd", "cvh", "uncrct", "gen_smeared"]:
                df = muon_validation.define_reco_over_gen_cols(df, reco_type)

    df = df.Define("goodMuons_ptJet0", "Jet_pt[Muon_jetIdx[goodMuons][0]]")

    df = df.Define("goodMuons_relIso0", f"{isoBranch}[goodMuons][0]")
    df = df.Define("goodMuons_iso0", "goodMuons_relIso0*Muon_pt[goodMuons][0]")

    # Jet collection actually has a pt threshold of 15 GeV in MiniAOD
    df = df.Define(
        "goodCleanJetsNoPt",
        "Jet_jetId >= 6 && (Jet_pt > 50 || Jet_puId >= 4) && abs(Jet_eta) < 2.4 && wrem::cleanJetsFromLeptons(Jet_eta,Jet_phi,Muon_correctedEta[vetoMuons],Muon_correctedPhi[vetoMuons],Electron_eta[vetoElectrons],Electron_phi[vetoElectrons])",
    )
    df = df.Define("passIso", f"goodMuons_relIso0 < {args.isolationThreshold}")
    df = df.Alias(
        "goodMuons_passIso0", "passIso"
    )  # for more flexible handling of efficiency helpers, coherently with Wlike and dilepton histmakers

    ########################################################################
    # gen match to bare muons to select only prompt muons from MC processes, but also including tau decays (defined here because needed for veto SF)
    if not dataset.is_data and not isQCDMC and not args.noGenMatchMC:
        df = theory_tools.define_postfsr_vars(df)
        df = df.Filter(
            "wrem::hasMatchDR2(goodMuons_eta0,goodMuons_phi0,GenPart_eta[postfsrMuons],GenPart_phi[postfsrMuons],0.09)"
        )
        df = df.Define(
            "postfsrMuons_inAcc",
            f"postfsrMuons && abs(GenPart_eta) < {args.vetoGenPartEta} && GenPart_pt > {args.vetoGenPartPt}",
        )
        if args.selectVetoEventsMC:
            # in principle a gen muon with eta = 2.401 might still be matched to a reco muon with eta < 2.4, same for pt, so this condition is potentially fragile, but it is just for test plots
            df = df.Filter("Sum(postfsrMuons_inAcc) >= 2")
        if not args.noVetoSF:
            df = df.Define(
                "hasMatchDR2idx",
                "wrem::hasMatchDR2idx_closest(goodMuons_eta0,goodMuons_phi0,GenPart_eta[postfsrMuons_inAcc],GenPart_phi[postfsrMuons_inAcc],0.09)",
            )
            df = df.Define(
                "GenPart_charge",
                "wrem::charge_from_pdgid(GenPart_pdgId[postfsrMuons_inAcc])",
            )
            df = df.Define(
                f"vetoMuons_tnpPt0",
                "wrem::unmatched_postfsrMuon_var(GenPart_pt[postfsrMuons_inAcc],  GenPart_pt[postfsrMuons_inAcc], hasMatchDR2idx)",
            )
            df = df.Define(
                f"vetoMuons_tnpEta0",
                "wrem::unmatched_postfsrMuon_var(GenPart_eta[postfsrMuons_inAcc], GenPart_pt[postfsrMuons_inAcc], hasMatchDR2idx)",
            )
            df = df.Define(
                f"vetoMuons_tnpCharge0",
                "wrem::unmatched_postfsrMuon_var(GenPart_charge, GenPart_pt[postfsrMuons_inAcc], hasMatchDR2idx)",
            )
    if isQCDMC:
        df = theory_tools.define_postfsr_vars(df)
        df = df.Filter(
            "wrem::hasMatchDR2(goodMuons_eta0,goodMuons_phi0,GenPart_eta[postfsrMuons],GenPart_phi[postfsrMuons],0.09) == 0"
        )

    if args.forceValidCVH:
        if dataset.is_data:
            if args.muonCorrData == "none":
                logger.warning(
                    "Requiring valid CVH for data even if CVH is not applied"
                )
                df = df.Filter("Muon_cvhPt[goodMuons][0] > 0")
        else:
            if args.muonCorrMC == "none":
                # use CVH with ideal MC geometry for this check
                logger.warning(
                    "Requiring valid CVH (ideal geometry) for MC even if CVH is not applied"
                )
                df = df.Filter("Muon_cvhidealPt[goodMuons][0] > 0")

    ########################################################################
    # define event weights here since they are needed below for some helpers
    if dataset.is_data:
        df = df.DefinePerSample("nominal_weight", "1.0")
    else:
        df = df.Define("weight_pu", pileup_helper, ["Pileup_nTrueInt"])
        df = df.Define("weight_vtx", vertex_helper, ["GenVtx_z", "Pileup_nTrueInt"])
        if era == "2016PostVFP":
            if args.addRunAxis and not args.randomizeDataByRun:
                # define helpers for prefiring in each sub era
                ## TODO: modify main helper to accept era directly as an argument
                (
                    muon_prefiring_helper_BG,
                    muon_prefiring_helper_stat_BG,
                    muon_prefiring_helper_syst_BG,
                ) = muon_prefiring.make_muon_prefiring_helpers(era="2016BG")
                (
                    muon_prefiring_helper_H,
                    muon_prefiring_helper_stat_H,
                    muon_prefiring_helper_syst_H,
                ) = muon_prefiring.make_muon_prefiring_helpers(era="2016H")

                df = df.Define(
                    "weight_newMuonPrefiringSF_BG",
                    muon_prefiring_helper_BG,
                    [
                        "Muon_correctedEta",
                        "Muon_correctedPt",
                        "Muon_correctedPhi",
                        "Muon_correctedCharge",
                        "Muon_looseId",
                    ],
                )
                df = df.Define(
                    "weight_newMuonPrefiringSF_H",
                    muon_prefiring_helper_H,
                    [
                        "Muon_correctedEta",
                        "Muon_correctedPt",
                        "Muon_correctedPhi",
                        "Muon_correctedCharge",
                        "Muon_looseId",
                    ],
                )
                df = df.Define(
                    "weight_newMuonPrefiringSF",
                    "(run4axis > 280385) ? weight_newMuonPrefiringSF_H : weight_newMuonPrefiringSF_BG",
                )
            else:
                df = df.Define(
                    "weight_newMuonPrefiringSF",
                    muon_prefiring_helper,
                    [
                        "Muon_correctedEta",
                        "Muon_correctedPt",
                        "Muon_correctedPhi",
                        "Muon_correctedCharge",
                        "Muon_looseId",
                    ],
                )
            weight_expr = (
                "weight_pu*weight_newMuonPrefiringSF*L1PreFiringWeight_ECAL_Nom"
            )
        else:
            weight_expr = (
                "weight_pu*L1PreFiringWeight_Muon_Nom*L1PreFiringWeight_ECAL_Nom"
            )

        if not args.noVertexWeight:
            weight_expr += "*weight_vtx"

        # for tests to split into number of reconstructed vertices
        if args.addNvtxAxis is not None and args.normWeightNvtx is not None:
            df = define_norm_weight_nRecoVtx(df, args.addNvtxAxis, args.normWeightNvtx)
            weight_expr += "*weight_nRecoVtx"

        # Muon variables used to measure tag-and-probe efficiency SF might not be the final corrected ones
        # TODO: implement an option to choose which vatriables to use (e.g. pt-eta-charge after CVH only ?)
        useTnpMuonVarForSF = args.useTnpMuonVarForSF
        if useTnpMuonVarForSF:
            df = df.Define("goodMuons_tnpPt0", "Muon_pt[goodMuons][0]")
            df = df.Define("goodMuons_tnpEta0", "Muon_eta[goodMuons][0]")
            df = df.Define("goodMuons_tnpCharge0", "Muon_charge[goodMuons][0]")
        else:
            df = df.Alias("goodMuons_tnpPt0", "goodMuons_pt0")
            df = df.Alias("goodMuons_tnpEta0", "goodMuons_eta0")
            df = df.Alias("goodMuons_tnpCharge0", "goodMuons_charge0")

        columnsForSF = [
            "goodMuons_tnpPt0",
            "goodMuons_tnpEta0",
            "goodMuons_SApt0",
            "goodMuons_SAeta0",
            "goodMuons_tnpUT0",
            "goodMuons_tnpCharge0",
            "passIso",
        ]

        # define recoil uT, muon projected on boson pt, the latter is made using preFSR variables
        # TODO: fix it for not W/Z processes
        df = muon_selections.define_muon_uT_variable(
            df,
            isWorZ,
            smooth3dsf=args.smooth3dsf,
            colNamePrefix="goodMuons",
            addWithTnpMuonVar=useTnpMuonVarForSF,
        )
        if not useTnpMuonVarForSF:
            df = df.Alias("goodMuons_tnpUT0", "goodMuons_uT0")

        # define_muon_uT_variable defined a uT variable using gen information, to get a more precise value for the purpose of applying scale factors
        # for using it as a fit observable we need another definition based only on reco observables, since it is also needed for data
        if not args.smooth3dsf:
            columnsForSF.remove("goodMuons_tnpUT0")

        if not isQCDMC and not args.noScaleFactors:
            df = df.Define(
                "weight_fullMuonSF_withTrackingReco",
                muon_efficiency_helper,
                columnsForSF,
            )
            weight_expr += "*weight_fullMuonSF_withTrackingReco"

            if isZveto and not args.noGenMatchMC:
                if args.scaleDYvetoFraction > 0.0:
                    # weight different from 1 only for events with >=2 gen muons in acceptance but only 1 reco muon
                    df = df.Define(
                        "weight_DY",
                        f"(vetoMuons_tnpCharge0 > -99) ? {args.scaleDYvetoFraction} : 1.0",
                    )
                    weight_expr += "*weight_DY"

                if not args.noVetoSF:
                    df = df.Define(
                        "weight_vetoSF_nominal",
                        muon_efficiency_veto_helper,
                        [
                            "vetoMuons_tnpPt0",
                            "vetoMuons_tnpEta0",
                            "vetoMuons_tnpCharge0",
                        ],
                    )
                    weight_expr += "*weight_vetoSF_nominal"

        # prepare inputs for pixel multiplicity helpers
        cvhName = "cvhideal"

        df = df.Define("goodMuons_eta", "Muon_correctedEta[goodMuons]")
        df = df.Define("goodMuons_pt", "Muon_correctedPt[goodMuons]")
        df = df.Define("goodMuons_charge", "Muon_correctedCharge[goodMuons]")
        df = df.Define(
            f"goodMuons_{cvhName}NValidPixelHits",
            f"Muon_{cvhName}NValidPixelHits[goodMuons]",
        )
        df = df.Define(
            "goodMuons_triggerCat",
            "ROOT::VecOps::RVec<wrem::TriggerCat>(goodMuons_eta.size(), wrem::TriggerCat::triggering)",
        )

        pixel_multiplicity_cols = [
            "goodMuons_triggerCat",
            "goodMuons_eta",
            "goodMuons_pt",
            "goodMuons_charge",
            f"goodMuons_{cvhName}NValidPixelHits",
        ]

        if args.reweightPixelMultiplicity:
            df = df.Define(
                "weight_pixel_multiplicity",
                pixel_multiplicity_helper,
                pixel_multiplicity_cols,
            )
            weight_expr += "*weight_pixel_multiplicity"

        logger.debug(f"Exp weight defined: {weight_expr}")
        df = df.Define("exp_weight", weight_expr)
        df = theory_tools.define_theory_weights_and_corrs(
            df, dataset.name, corr_helpers, args, theory_helpers=theory_helpers
        )

        if (
            isWmunu
            and args.theoryAgnostic
            and not hasattr(dataset, "out_of_acceptance")
        ):
            df = theoryAgnostic_tools.define_helicity_weights(df)

    ########################################################################

    if not args.noRecoil:
        leps_uncorr = [
            "Muon_pt[goodMuons][0]",
            "Muon_eta[goodMuons][0]",
            "Muon_phi[goodMuons][0]",
            "Muon_charge[goodMuons][0]",
        ]
        leps_corr = [
            "goodMuons_pt0",
            "goodMuons_eta0",
            "goodMuons_phi0",
            "goodMuons_charge0",
        ]
        df = recoilHelper.recoil_W(
            df,
            results,
            dataset,
            common.wprocs_recoil,
            leps_uncorr,
            leps_corr,
            cols_fakerate=columns_fakerate,
            axes_fakerate=axes_fakerate,
            mtw_min=mtw_min,
        )  # produces corrected MET as MET_corr_rec_pt/phi

    else:
        met = "DeepMETResolutionTune" if args.met == "DeepMETReso" else args.met
        df = df.Alias("MET_corr_rec_pt", f"{met}_pt")
        df = df.Alias("MET_corr_rec_phi", f"{met}_phi")

    df = df.Define(
        "goodMuons_utReco",
        "wrem::zqtproj0(goodMuons_pt0, goodMuons_phi0, MET_corr_rec_pt, MET_corr_rec_phi)",
    )
    df = df.Define(
        "goodMuons_angleSignUt0",
        "wrem::zqtproj0_angleSign(goodMuons_pt0, goodMuons_phi0, MET_corr_rec_pt, MET_corr_rec_phi)",
    )
    df = df.Define(
        "goodMuons_angleCosineUt0",
        "wrem::zqtproj0_angleCosine(goodMuons_pt0, goodMuons_phi0, MET_corr_rec_pt, MET_corr_rec_phi)",
    )
    if args.utAngleCosineCut is not None:
        df = df.Filter(
            f"goodMuons_angleCosineUt0 >= {args.utAngleCosineCut[0]} && goodMuons_angleCosineUt0 < {args.utAngleCosineCut[1]}"
        )

    df = df.Define(
        "ptW",
        "wrem::pt_2(goodMuons_pt0, goodMuons_phi0, MET_corr_rec_pt, MET_corr_rec_phi)",
    )
    df = df.Define(
        "transverseMass",
        "wrem::mt_2(goodMuons_pt0, goodMuons_phi0, MET_corr_rec_pt, MET_corr_rec_phi)",
    )

    df = df.Define("hasCleanJet", "Sum(goodCleanJetsNoPt && Jet_pt > 30) >= 1")
    df = df.Define(
        "deltaPhiMuonMet", "std::abs(wrem::deltaPhi(goodMuons_phi0,MET_corr_rec_phi))"
    )

    if auxiliary_histograms:
        # would move the following in a function but there are too many dependencies on axes defined in the main loop
        if isQCDMC:
            ## for some tests with QCD MC only
            axis_eta_coarse_fakes = hist.axis.Regular(
                12, -2.4, 2.4, name="eta", overflow=False, underflow=False
            )
            axis_pt_coarse_fakes = hist.axis.Regular(
                8, 26, 58, name="pt", overflow=False, underflow=False
            )
            axis_mt_coarse_fakes = hist.axis.Regular(
                12, 0.0, 120.0, name="mt", underflow=False, overflow=True
            )
            axis_Njets_fakes = hist.axis.Regular(
                5, -0.5, 4.5, name="NjetsClean", underflow=False, overflow=True
            )
            axis_leadjetPt_fakes = hist.axis.Regular(
                20, 0.0, 100.0, name="leadjetPt", underflow=False, overflow=True
            )
            otherStudyForFakes_axes = [
                axis_eta_coarse_fakes,
                axis_pt_coarse_fakes,
                axis_charge,
                axis_mt_coarse_fakes,
                axis_passIso,
                axis_Njets_fakes,
                axis_leadjetPt_fakes,
                axis_dphi_fakes,
            ]

            df = df.Define("goodMuons_hasJet0", "Muon_jetIdx[goodMuons][0] != -1 ")
            df = df.Define(
                "goodMuons_jetpt0",
                "goodMuons_hasJet0 ? Jet_pt[Muon_jetIdx[goodMuons][0]] : goodMuons_pt0",
            )
            df = df.Define("goodMuons_genPartFlav0", "Muon_genPartFlav[goodMuons][0]")
            df = df.Define("nJetsClean", "Sum(goodCleanJetsNoPt)")
            df = df.Define(
                "leadjetPt", "(nJetsClean > 0) ? Jet_pt[goodCleanJetsNoPt][0] : 0.0"
            )
            # df = df.Filter("goodMuons_genPartFlav0 == 3") # FOR TESTS
            #
            axis_genPartFlav = hist.axis.Regular(
                6, -0.5, 5.5, name="Muon genPart flavor"
            )
            results.append(
                df.HistoBoost(
                    "Muon_genPartFlav_multi",
                    [
                        axis_genPartFlav,
                        axis_eta_coarse_fakes,
                        axis_pt_coarse_fakes,
                        axis_charge,
                        axis_mt_coarse_fakes,
                        axis_passIso,
                    ],
                    [
                        "goodMuons_genPartFlav0",
                        "goodMuons_eta0",
                        "goodMuons_pt0",
                        "goodMuons_charge0",
                        "transverseMass",
                        "passIso",
                        "nominal_weight",
                    ],
                )
            )
            #
            otherStudyForFakes = df.HistoBoost(
                "otherStudyForFakes",
                otherStudyForFakes_axes,
                [
                    "goodMuons_eta0",
                    "goodMuons_pt0",
                    "goodMuons_charge0",
                    "transverseMass",
                    "passIso",
                    "nJetsClean",
                    "leadjetPt",
                    "deltaPhiMuonMet",
                    "nominal_weight",
                ],
            )
            results.append(otherStudyForFakes)
            # gen match studies
            axis_match = hist.axis.Boolean(name="hasMatch")
            axis_prompt = hist.axis.Boolean(name="isPrompt")
            axis_genPt = hist.axis.Regular(
                60, 0.0, 60.0, name="genPtMuonsStatus1", underflow=False, overflow=True
            )
            df = df.Define(
                "postfsrMuonsStatus1", "GenPart_status == 1 && abs(GenPart_pdgId) == 13"
            )
            df = df.Define(
                "GenPart_isPrompt",
                "GenPart_statusFlags & 1 || GenPart_statusFlags & (1 << 5)",
            )
            df = df.Define(
                "genIsMatchedToRecoMuon",
                "wrem::hasMatchDR2collWithSingle(GenPart_eta,GenPart_phi,goodMuons_eta0,goodMuons_phi0)",
            )
            df = df.Define(
                "postfsrMuonsStatus1genMatched",
                "postfsrMuonsStatus1 && genIsMatchedToRecoMuon",
            )
            df = df.Define(
                "postfsrMuonsStatus1_pt", "GenPart_pt[postfsrMuonsStatus1genMatched]"
            )
            df = df.Define(
                "postfsrMuonsStatus1_isPrompt",
                "GenPart_isPrompt[postfsrMuonsStatus1genMatched]",
            )
            postfsrMuonsGenMatchStatus1 = df.HistoBoost(
                "postfsrMuonsGenMatchStatus1",
                [axis_genPt, axis_prompt, axis_pt, axis_genPartFlav, axis_passIso],
                [
                    "postfsrMuonsStatus1_pt",
                    "postfsrMuonsStatus1_isPrompt",
                    "goodMuons_pt0",
                    "goodMuons_genPartFlav0",
                    "passIso",
                    "nominal_weight",
                ],
            )
            results.append(postfsrMuonsGenMatchStatus1)
            df = df.Define(
                "goodMuons_genPartPt0",
                "Muon_genPartIdx[goodMuons][0] >= 0 ? GenPart_pt[Muon_genPartIdx[goodMuons][0]] : -1",
            )
            genVsRecoPt = df.HistoBoost(
                "genVsRecoPt",
                [axis_genPt, axis_pt, axis_genPartFlav, axis_passIso],
                [
                    "goodMuons_genPartPt0",
                    "goodMuons_pt0",
                    "goodMuons_genPartFlav0",
                    "passIso",
                    "nominal_weight",
                ],
            )
            results.append(genVsRecoPt)
            #
            df = df.Define(
                "postfsrMuonsStatus1prompt", "postfsrMuonsStatus1 && GenPart_isPrompt"
            )
            df = df.Define(
                "postfsrMuonsStatus1notPrompt",
                "postfsrMuonsStatus1 && !GenPart_isPrompt",
            )
            df = df.Define(
                "muonGenMatchStatus1",
                "wrem::hasMatchDR2(goodMuons_eta0,goodMuons_phi0,GenPart_eta[postfsrMuonsStatus1],GenPart_phi[postfsrMuonsStatus1])",
            )
            df = df.Define(
                "muonGenMatchStatus1prompt",
                "wrem::hasMatchDR2(goodMuons_eta0,goodMuons_phi0,GenPart_eta[postfsrMuonsStatus1prompt],GenPart_phi[postfsrMuonsStatus1prompt])",
            )
            df = df.Define(
                "muonGenMatchStatus1notPrompt",
                "wrem::hasMatchDR2(goodMuons_eta0,goodMuons_phi0,GenPart_eta[postfsrMuonsStatus1notPrompt],GenPart_phi[postfsrMuonsStatus1notPrompt])",
            )
            #
            etaPtGenMatchStatus1 = df.HistoBoost(
                "etaPtGenMatchStatus1",
                [axis_eta, axis_pt, axis_match],
                [
                    "goodMuons_eta0",
                    "goodMuons_pt0",
                    "muonGenMatchStatus1",
                    "nominal_weight",
                ],
            )
            results.append(etaPtGenMatchStatus1)
            etaPtGenMatchStatus1prompt = df.HistoBoost(
                "etaPtGenMatchStatus1prompt",
                [axis_eta, axis_pt, axis_match],
                [
                    "goodMuons_eta0",
                    "goodMuons_pt0",
                    "muonGenMatchStatus1prompt",
                    "nominal_weight",
                ],
            )
            results.append(etaPtGenMatchStatus1prompt)
            etaPtGenMatchStatus1notPrompt = df.HistoBoost(
                "etaPtGenMatchStatus1notPrompt",
                [axis_eta, axis_pt, axis_match],
                [
                    "goodMuons_eta0",
                    "goodMuons_pt0",
                    "muonGenMatchStatus1notPrompt",
                    "nominal_weight",
                ],
            )
            results.append(etaPtGenMatchStatus1notPrompt)
            ### gen mT and reco mT
            df = df.Define(
                "transverseMass_genMetRecoMuon",
                "wrem::mt_2(goodMuons_pt0, goodMuons_phi0, GenMET_pt, GenMET_phi)",
            )
            axis_genmt = hist.axis.Regular(
                120, 0.0, 120.0, name="genMt", underflow=False, overflow=True
            )
            axis_recomet = hist.axis.Regular(
                120, 0.0, 120.0, name="recoMet", underflow=False, overflow=True
            )
            axis_genmet = hist.axis.Regular(
                120, 0.0, 120.0, name="genMet", underflow=False, overflow=True
            )
            etaPtMtGenMt = df.HistoBoost(
                "etaPtMtGenMt",
                [axis_eta, axis_pt, axis_mt_fakes, axis_genmt, axis_passIso],
                [
                    "goodMuons_eta0",
                    "goodMuons_pt0",
                    "transverseMass",
                    "transverseMass_genMetRecoMuon",
                    "passIso",
                    "nominal_weight",
                ],
            )
            results.append(etaPtMtGenMt)
            etaPtMetGenMet = df.HistoBoost(
                "etaPtMetGenMet",
                [axis_eta, axis_pt, axis_recomet, axis_genmet, axis_passIso],
                [
                    "goodMuons_eta0",
                    "goodMuons_pt0",
                    "MET_corr_rec_pt",
                    "GenMET_pt",
                    "passIso",
                    "nominal_weight",
                ],
            )
            results.append(etaPtMetGenMet)
            #
            muonHasJet = df.HistoBoost(
                "muonHasJet",
                [
                    hist.axis.Regular(2, -0.5, 1.5, name="hasJet"),
                    hist.axis.Regular(2, -0.5, 1.5, name="passIsolation"),
                ],
                ["goodMuons_hasJet0", "passIso", "nominal_weight"],
            )
            results.append(muonHasJet)
            # add a cut to a new branch of the dataframe
            dfalt = df.Filter("goodMuons_hasJet0")
            axis_jetpt = hist.axis.Regular(
                40, 20, 60, name="jetpt", overflow=False, underflow=False
            )
            axis_muonpt = hist.axis.Regular(
                40, 20, 60, name="pt", overflow=False, underflow=False
            )
            muonPtVsJetPt = dfalt.HistoBoost(
                "muonPtVsJetPt",
                [axis_muonpt, axis_jetpt, axis_passIso],
                ["goodMuons_pt0", "goodMuons_jetpt0", "passIso", "nominal_weight"],
            )
            results.append(muonPtVsJetPt)
            ##
        # instead of passIso in mTStudyForFakes
        # df = df.Define("passIsoAlt", "(goodMuons_pfRelIso04_all0 * Muon_pt[goodMuons][0] / goodMuons_jetpt0) < 0.12")
        # df = df.Define("passIsoAlt", "(Muon_vtxAgnPfRelIso04_chg[goodMuons][0] * Muon_pt[goodMuons][0]) < 5.0")
        mTStudyForFakes = df.HistoBoost(
            "mTStudyForFakes",
            mTStudyForFakes_axes,
            [
                "goodMuons_eta0",
                "goodMuons_pt0",
                "goodMuons_charge0",
                "transverseMass",
                "passIso",
                "hasCleanJet",
                "deltaPhiMuonMet",
                "nominal_weight",
            ],
        )
        results.append(mTStudyForFakes)

    # add filter of deltaPhi(muon,met) before other histograms (but after histogram mTStudyForFakes)
    if args.dphiMuonMetCut > 0.0 and not args.makeMCefficiency:
        dphiMuonMetCut = args.dphiMuonMetCut * np.pi
        df = df.Filter(
            f"deltaPhiMuonMet > {dphiMuonMetCut}"
        )  # pi/4 was found to be a good threshold for signal with mT > 40 GeV

    df = df.Define("passMT", f"transverseMass >= {mtw_min}")

    if auxiliary_histograms:

        # control plots, lepton, met, to plot them later (need eta-pt to make fakes)
        results.append(
            df.HistoBoost(
                "leptonPhi",
                [axis_phi, *axes_fakerate],
                ["goodMuons_phi0", *columns_fakerate, "nominal_weight"],
            )
        )
        results.append(
            df.HistoBoost(
                "MET",
                [axis_met, *axes_fakerate],
                ["MET_corr_rec_pt", *columns_fakerate, "nominal_weight"],
            )
        )
        results.append(
            df.HistoBoost(
                "METPhi",
                [axis_phi, *axes_fakerate],
                ["MET_corr_rec_phi", *columns_fakerate, "nominal_weight"],
            )
        )
        results.append(
            df.HistoBoost(
                "deltaPhiMuonMet",
                [axis_phi, *axes_fakerate],
                ["deltaPhiMuonMet", *columns_fakerate, "nominal_weight"],
            )
        )
        results.append(
            df.HistoBoost(
                "ptW",
                [axis_recoWpt, *axes_fakerate],
                ["ptW", *columns_fakerate, "nominal_weight"],
            )
        )
        # for mt use axis with different binning
        results.append(
            df.HistoBoost(
                "transverseMass",
                [
                    axis_fakes_eta,
                    axis_fakes_pt,
                    axis_charge,
                    axis_mt,
                    common.axis_relIsoCat,
                ],
                [
                    "goodMuons_eta0",
                    "goodMuons_pt0",
                    "goodMuons_charge0",
                    "transverseMass",
                    "goodMuons_relIso0",
                    "nominal_weight",
                ],
            )
        )

        # other plots
        results.append(
            df.HistoBoost("iso", [axis_iso], ["goodMuons_iso0", "nominal_weight"])
        )
        results.append(
            df.HistoBoost(
                "relIso", [axis_relIso], ["goodMuons_relIso0", "nominal_weight"]
            )
        )

    if args.poiAsNoi:
        if (
            args.theoryAgnostic
            and isWmunu
            and not hasattr(dataset, "out_of_acceptance")
        ):  # TODO: might add Wtaunu at some point, not yet
            noiAsPoiHistName = Datagroups.histName(
                "nominal", syst="yieldsTheoryAgnostic"
            )
            logger.debug(
                f"Creating special histogram '{noiAsPoiHistName}' for theory agnostic to treat POIs as NOIs"
            )
            results.append(
                df.HistoBoost(
                    noiAsPoiHistName,
                    [*nominal_axes, *theoryAgnostic_axes],
                    [*nominal_cols, *theoryAgnostic_cols, "nominal_weight_helicity"],
                    tensor_axes=[axis_helicity],
                )
            )
            if args.theoryAgnosticPolVar:
                theoryAgnostic_helpers_cols = [
                    "qtOverQ",
                    "absYVgen",
                    "chargeVgen",
                    "csSineCosThetaPhigen",
                    "nominal_weight",
                ]
                # assume to have same coeffs for plus and minus (no reason for it not to be the case)
                for genVcharge in ["minus", "plus"]:
                    for coeffKey in theoryAgnostic_helpers_minus.keys():
                        logger.debug(
                            f"Creating theory agnostic histograms with polynomial variations for {coeffKey} and {genVcharge} gen W charge"
                        )
                        helperQ = (
                            theoryAgnostic_helpers_minus[coeffKey]
                            if genVcharge == "minus"
                            else theoryAgnostic_helpers_plus[coeffKey]
                        )
                        df = df.Define(
                            f"theoryAgnostic_{coeffKey}_{genVcharge}_tensor",
                            helperQ,
                            theoryAgnostic_helpers_cols,
                        )
                        noiAsPoiWithPolHistName = Datagroups.histName(
                            "nominal",
                            syst=f"theoryAgnosticWithPol_{coeffKey}_{genVcharge}",
                        )
                        results.append(
                            df.HistoBoost(
                                noiAsPoiWithPolHistName,
                                nominal_axes,
                                [
                                    *nominal_cols,
                                    f"theoryAgnostic_{coeffKey}_{genVcharge}_tensor",
                                ],
                                tensor_axes=helperQ.tensor_axes,
                                storage=hist.storage.Double(),
                            )
                        )

        if args.unfolding:
            if isWmunu:
                for level in args.unfoldingLevels:
                    noiAsPoiHistName = Datagroups.histName(
                        "nominal", syst=f"{level}_yieldsUnfolding"
                    )
                    logger.debug(
                        f"Creating special histogram '{noiAsPoiHistName}' for unfolding to treat POIs as NOIs"
                    )
                    results.append(
                        df.HistoBoost(
                            noiAsPoiHistName,
                            [*nominal_axes, *unfolding_axes[level]],
                            [*nominal_cols, *unfolding_cols[level], "nominal_weight"],
                        )
                    )
                    # create corresponding histogram without experimental weights, to correlate stat between gen and reco
                    weight_expr = theory_tools.build_weight_expr(
                        df,
                        exclude_weights=[
                            "exp_weight",
                        ],
                    )  # May want to exclude "ew_theory_corr_weight" in case of QCD only gen definition
                    logger.info(f"Theory weight is {weight_expr}")
                    df = df.Define(f"theory_weight_{level}", weight_expr)

                    results.append(
                        df.HistoBoost(
                            f"{noiAsPoiHistName}_theory_weight",
                            [*nominal_axes, *unfolding_axes[level]],
                            [
                                *nominal_cols,
                                *unfolding_cols[level],
                                f"theory_weight_{level}",
                            ],
                        )
                    )

            elif dataset.name == "Zmumu_2016PostVFP":
                unfolder_z.add_poi_as_noi_histograms(
                    df,
                    results,
                    nominal_axes,
                    nominal_cols,
                )

    if isWorZ and not hasattr(dataset, "out_of_acceptance"):
        theoryAgnostic_helpers_cols = [
            "qtOverQ",
            "absYVgen",
            "chargeVgen",
            "csSineCosThetaPhigen",
            "nominal_weight",
        ]
        # assume to have same coeffs for plus and minus (no reason for it not to be the case)
        if dataset.name in ["Wplusmunu_2016PostVFP", "Wplustaunu_2016PostVFP"]:
            helpers_class = muRmuFPolVar_helpers_plus
            process_name = "W"
        elif dataset.name in ["Wminusmunu_2016PostVFP", "Wminustaunu_2016PostVFP"]:
            helpers_class = muRmuFPolVar_helpers_minus
            process_name = "W"
        elif dataset.name in ["Zmumu_2016PostVFP", "Ztautau_2016PostVFP"]:
            helpers_class = muRmuFPolVar_helpers_Z
            process_name = "Z"
        else:
            helpers_class = {}

        for coeffKey in helpers_class.keys():
            logger.debug(
                f"Creating muR/muF histograms with polynomial variations for {coeffKey}"
            )
            helperQ = helpers_class[coeffKey]
            df = df.Define(
                f"muRmuFPolVar_{coeffKey}_tensor", helperQ, theoryAgnostic_helpers_cols
            )
            muRmuFWithPolHistName = Datagroups.histName(
                "nominal", syst=f"muRmuFPolVar{process_name}_{coeffKey}"
            )
            results.append(
                df.HistoBoost(
                    muRmuFWithPolHistName,
                    nominal_axes,
                    [*nominal_cols, f"muRmuFPolVar_{coeffKey}_tensor"],
                    tensor_axes=helperQ.tensor_axes,
                    storage=hist.storage.Double(),
                )
            )

    if not args.onlyMainHistograms:
        syst_tools.add_QCDbkg_jetPt_hist(
            results, df, axes, cols, jet_pt=30, storage_type=storage_type
        )

    if isQCDMC:
        unweighted = df.HistoBoost("unweighted", axes, cols)
        results.append(unweighted)

    nominal_withUt = df.HistoBoost(
        "nominal_withUt",
        [*axes, axis_ut_fine],
        [*cols, "goodMuons_utReco", "nominal_weight"],
    )
    results.append(nominal_withUt)
    nominal_withUtAngleCosine = df.HistoBoost(
        "nominal_withUtAngleCosine",
        [*axes, axis_uTAngleCosine],
        [*cols, "goodMuons_angleCosineUt0", "nominal_weight"],
    )
    results.append(nominal_withUtAngleCosine)

    if dataset.is_data:
        nominal = df.HistoBoost("nominal", axes, cols)
        results.append(nominal)
    else:
        nominal = df.HistoBoost("nominal", axes, [*cols, "nominal_weight"])
        results.append(nominal)
        results.append(
            df.HistoBoost(
                "nominal_weight",
                [hist.axis.Regular(200, -4, 4)],
                ["nominal_weight"],
                storage=hist.storage.Double(),
            )
        )

        if args.makeMCefficiency:
            axes_WeffMC = [
                axis_eta,
                axis_pt_eff,
                axis_ut,
                axis_charge,
                axis_passIso,
                axis_passMT,
                axis_passTrigger,
            ]
            cols_WeffMC = [
                "goodMuons_eta0",
                "goodMuons_pt0",
                "goodMuons_uT0",
                "goodMuons_charge0",
                "passIso",
                "passMT",
                "passTrigger",
            ]
            yieldsForWeffMC = df.HistoBoost(
                "yieldsForWeffMC", axes_WeffMC, [*cols_WeffMC, "nominal_weight"]
            )
            results.append(yieldsForWeffMC)

        if isWorZ:
            # for vertex efficiency plot in MC
            df = df.Define("absDiffGenRecoVtx_z", "std::abs(GenVtx_z - PV_z)")
            df = df.Define("goodMuons_abseta0", "abs(goodMuons_eta0)")
            axis_absDiffGenRecoVtx_z = hist.axis.Regular(
                100, 0, 2.0, name="absDiffGenRecoVtx_z", underflow=False, overflow=True
            )
            axis_prefsrWpt = hist.axis.Regular(
                50, 0.0, 100.0, name="prefsrWpt", underflow=False, overflow=True
            )
            axis_abseta = hist.axis.Regular(
                6, 0, 2.4, name="abseta", overflow=False, underflow=False
            )
            cols_vertexZstudy = [
                "goodMuons_abseta0",
                "passIso",
                "passMT",
                "absDiffGenRecoVtx_z",
                "ptVgen",
            ]
            yieldsVertexZstudy = df.HistoBoost(
                "nominal_vertexZstudy",
                [
                    axis_abseta,
                    axis_passIso,
                    axis_passMT,
                    axis_absDiffGenRecoVtx_z,
                    axis_prefsrWpt,
                ],
                [*cols_vertexZstudy, "nominal_weight"],
            )
            results.append(yieldsVertexZstudy)

        if not args.noRecoil and args.recoilUnc:
            df = recoilHelper.add_recoil_unc_W(
                df, results, dataset, cols, axes, "nominal", storage_type=storage_type
            )
        if apply_theory_corr:
            syst_tools.add_theory_corr_hists(
                results,
                df,
                axes,
                cols,
                corr_helpers[dataset.name],
                theory_corrs,
                base_name="nominal",
                modify_central_weight=not args.theoryCorrAltOnly,
                isW=isW,
                storage_type=storage_type,
            )
        if isWorZ:
            cols_gen, cols_gen_smeared = muon_calibration.make_alt_reco_and_gen_hists(
                df, results, axes, cols, reco_sel_GF
            )
            if args.validationHists:
                muon_validation.make_reco_over_gen_hists(df, results)

    if not dataset.is_data and not args.onlyMainHistograms:

        if not args.onlyTheorySyst:
            if not isQCDMC and not args.noScaleFactors:
                df = syst_tools.add_muon_efficiency_unc_hists(
                    results,
                    df,
                    muon_efficiency_helper_stat,
                    muon_efficiency_helper_syst,
                    axes,
                    cols,
                    what_analysis=thisAnalysis,
                    smooth3D=args.smooth3dsf,
                    storage_type=storage_type,
                )
                for es in common.muonEfficiency_altBkgSyst_effSteps:
                    df = syst_tools.add_muon_efficiency_unc_hists_altBkg(
                        results,
                        df,
                        muon_efficiency_helper_syst_altBkg[es],
                        axes,
                        cols,
                        what_analysis=thisAnalysis,
                        step=es,
                        storage_type=storage_type,
                    )
                if isZveto and not args.noGenMatchMC and not args.noVetoSF:
                    df = syst_tools.add_muon_efficiency_veto_unc_hists(
                        results,
                        df,
                        muon_efficiency_veto_helper_stat,
                        muon_efficiency_veto_helper_syst,
                        axes,
                        cols,
                        storage_type=storage_type,
                    )
            if era == "2016PostVFP" and args.addRunAxis and not args.randomizeDataByRun:
                # to simplify the code, use helper with largest uncertainty for all eras when splitting data
                df = syst_tools.add_L1Prefire_unc_hists(
                    results,
                    df,
                    axes,
                    cols,
                    helper_stat=muon_prefiring_helper_stat_BG,
                    helper_syst=muon_prefiring_helper_syst_BG,
                    storage_type=storage_type,
                )
            else:
                df = syst_tools.add_L1Prefire_unc_hists(
                    results,
                    df,
                    axes,
                    cols,
                    helper_stat=muon_prefiring_helper_stat,
                    helper_syst=muon_prefiring_helper_syst,
                    storage_type=storage_type,
                )

        # n.b. this is the W analysis so mass weights shouldn't be propagated
        # on the Z samples (but can still use it for dummy muon scale)

        if isWorZ:

            df = syst_tools.add_theory_hists(
                results,
                df,
                args,
                dataset.name,
                corr_helpers,
                theory_helpers,
                axes,
                cols,
                for_wmass=True,
                storage_type=storage_type,
            )

            # Don't think it makes sense to apply the mass weights to scale leptons from tau decays
            if (
                "massWeight_tensor" in df.GetColumnNames()
                and not args.onlyTheorySyst
                and not "tau" in dataset.name
            ):
                df = syst_tools.add_muonscale_hist(
                    results,
                    df,
                    args.muonCorrEtaBins,
                    args.muonCorrMag,
                    isW,
                    axes,
                    cols,
                    storage_type=storage_type,
                )
                if args.muonScaleVariation == "smearingWeightsGaus":
                    df = syst_tools.add_muonscale_smeared_hist(
                        results,
                        df,
                        args.muonCorrEtaBins,
                        args.muonCorrMag,
                        isW,
                        axes,
                        cols_gen_smeared,
                        storage_type=storage_type,
                    )

            ####################################################
            # nuisances from the muon momemtum scale calibration
            if args.muonCorrData in ["massfit", "lbl_massfit"]:
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

                # muon scale variation from stats. uncertainty on the jpsi massfit
                df = muon_calibration.add_jpsi_crctn_stats_unc_hists(
                    args,
                    df,
                    axes,
                    results,
                    cols,
                    cols_gen_smeared,
                    calib_filepaths,
                    jpsi_crctn_data_unc_helper,
                    smearing_weights_procs,
                    reco_sel_GF,
                    dataset.name,
                    isW,
                    storage_type=storage_type,
                )
                # add the ad-hoc Z non-closure nuisances from the jpsi massfit to muon scale unc
                df = muon_calibration.add_jpsi_crctn_Z_non_closure_hists(
                    args,
                    df,
                    axes,
                    results,
                    cols,
                    cols_gen_smeared,
                    z_non_closure_parametrized_helper,
                    z_non_closure_binned_helper,
                    reco_sel_GF,
                    storage_type=storage_type,
                )
                # add nuisances from the data/MC resolution mismatch
                df = muon_calibration.add_resolution_uncertainty(
                    df,
                    axes,
                    results,
                    cols,
                    smearing_uncertainty_helper,
                    reco_sel_GF,
                    storage_type=storage_type,
                )
                if args.validationHists:
                    df = muon_validation.make_hists_for_muon_scale_var_weights(
                        df, axes, results, cols, cols_gen_smeared
                    )

                # add pixel multiplicity uncertainties
                df = df.Define(
                    "nominal_pixelMultiplicitySyst_tensor",
                    pixel_multiplicity_uncertainty_helper,
                    [*pixel_multiplicity_cols, "nominal_weight"],
                )
                hist_pixelMultiplicitySyst = df.HistoBoost(
                    "nominal_pixelMultiplicitySyst",
                    axes,
                    [*cols, "nominal_pixelMultiplicitySyst_tensor"],
                    tensor_axes=pixel_multiplicity_uncertainty_helper.tensor_axes,
                    storage=hist.storage.Double(),
                )
                results.append(hist_pixelMultiplicitySyst)

                if args.pixelMultiplicityStat:
                    df = df.Define(
                        "nominal_pixelMultiplicityStat_tensor",
                        pixel_multiplicity_uncertainty_helper_stat,
                        [*pixel_multiplicity_cols, "nominal_weight"],
                    )
                    hist_pixelMultiplicityStat = df.HistoBoost(
                        "nominal_pixelMultiplicityStat",
                        axes,
                        [*cols, "nominal_pixelMultiplicityStat_tensor"],
                        tensor_axes=pixel_multiplicity_uncertainty_helper_stat.tensor_axes,
                        storage=hist.storage.Double(),
                    )
                    results.append(hist_pixelMultiplicityStat)

                # extra uncertainties from non-closure stats
                df = df.Define(
                    "muonScaleClosSyst_responseWeights_tensor_splines",
                    closure_unc_helper,
                    [*input_kinematics, "nominal_weight"],
                )
                nominal_muonScaleClosSyst_responseWeights = df.HistoBoost(
                    "nominal_muonScaleClosSyst_responseWeights",
                    axes,
                    [*cols, "muonScaleClosSyst_responseWeights_tensor_splines"],
                    tensor_axes=closure_unc_helper.tensor_axes,
                    storage=hist.storage.Double(),
                )
                results.append(nominal_muonScaleClosSyst_responseWeights)

                # extra uncertainties for A (fully correlated)
                df = df.Define(
                    "muonScaleClosASyst_responseWeights_tensor_splines",
                    closure_unc_helper_A,
                    [*input_kinematics, "nominal_weight"],
                )
                nominal_muonScaleClosASyst_responseWeights = df.HistoBoost(
                    "nominal_muonScaleClosASyst_responseWeights",
                    axes,
                    [*cols, "muonScaleClosASyst_responseWeights_tensor_splines"],
                    tensor_axes=closure_unc_helper_A.tensor_axes,
                    storage=hist.storage.Double(),
                )
                results.append(nominal_muonScaleClosASyst_responseWeights)

                # extra uncertainties for M (fully correlated)
                df = df.Define(
                    "muonScaleClosMSyst_responseWeights_tensor_splines",
                    closure_unc_helper_M,
                    [*input_kinematics, "nominal_weight"],
                )
                nominal_muonScaleClosMSyst_responseWeights = df.HistoBoost(
                    "nominal_muonScaleClosMSyst_responseWeights",
                    axes,
                    [*cols, "muonScaleClosMSyst_responseWeights_tensor_splines"],
                    tensor_axes=closure_unc_helper_M.tensor_axes,
                    storage=hist.storage.Double(),
                )
                results.append(nominal_muonScaleClosMSyst_responseWeights)

            ####################################################

            df = df.Define(
                "Muon_cvhMomCov",
                "wrem::splitNestedRVec(Muon_cvhMomCov_Vals, Muon_cvhMomCov_Counts)",
            )

    if hasattr(dataset, "out_of_acceptance"):
        # Rename dataset to not overwrite the original one
        if (
            len(smearing_weights_procs) > 0
            and smearing_weights_procs[-1] == dataset.name
        ):
            smearing_weights_procs[-1] = dataset.name + "OOA"
        dataset.name = dataset.name + "OOA"

    return results, weightsum


if args.sequentialEventLoops:
    dataset_sets = [[dataset] for dataset in datasets]
else:
    dataset_sets = [datasets]

for loop_datasets in dataset_sets:
    resultdict = narf.build_and_run(loop_datasets, build_graph)
    if not args.onlyMainHistograms and args.muonScaleVariation == "smearingWeightsGaus":
        logger.debug("Apply smearingWeights")
        muon_calibration.transport_smearing_weights_to_reco(
            resultdict, smearing_weights_procs, nonClosureScheme=args.nonClosureScheme
        )
    if args.validationHists:
        muon_validation.muon_scale_variation_from_manual_shift(resultdict)

    if not args.noScaleToData and not args.sequentialEventLoops:
        scale_to_data(resultdict)
        aggregate_groups(loop_datasets, resultdict, groups_to_aggregate)

    fout = f"{os.path.basename(__file__).replace('py', 'hdf5')}"
    fout = write_analysis_output(resultdict, fout, args)
    if not args.appendOutputFile:
        args.appendOutputFile = fout
    resultdict = None
