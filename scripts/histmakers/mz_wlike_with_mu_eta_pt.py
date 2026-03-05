import os

from utilities import common, differential, parsing
from wremnants.datasets.datagroups import Datagroups

analysis_label = Datagroups.analysisLabel(os.path.basename(__file__))
parser, initargs = parsing.common_parser(analysis_label)


import hist
import numpy as np
import ROOT

import narf
import wremnants
from wremnants import (
    helicity_utils,
    muon_calibration,
    muon_efficiencies_binned,
    muon_efficiencies_smooth,
    muon_prefiring,
    muon_selections,
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
from wums import logging

parser.add_argument(
    "--mtCut",
    type=int,
    default=common.get_default_mtcut(analysis_label),
    help="Value for the transverse mass cut in the event selection",
)  # 40 for Wmass, thus be 45 here (roughly half the boson mass)
parser.add_argument(
    "--muonIsolation",
    type=int,
    nargs=2,
    default=[1, 1],
    choices=[-1, 0, 1],
    help="Apply isolation cut to triggering and not-triggering muon (in this order): -1/1 for failing/passing isolation, 0 for skipping it",
)
parser.add_argument(
    "--addIsoMtAxes",
    action="store_true",
    help="Add iso/mT axes to the nominal ones. It is for tests to get uncertainties (mainly from SF) versus iso-mT to validate the goodness of muon SF in the fake regions. Isolation (on triggering muon) and mT cut are automatically overridden.",
)
parser.add_argument(
    "--validateVetoSF",
    action="store_true",
    help="Add histogram for validation of veto SF, loading all necessary helpers. This requires using the veto selection on the non-triggering muon, with reduced pt cut",
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
    "--fillHistNonTrig",
    action="store_true",
    help="Fill histograms with non triggering muon (for tests)",
)
parser.add_argument(
    "--flipEventNumberSplitting",
    action="store_true",
    help="Flip even with odd event numbers to consider the positive or negative muon as the W-like muon",
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
isFloatingPOIsTheoryAgnostic = args.theoryAgnostic and not args.poiAsNoi

if args.randomizeDataByRun and not args.addRunAxis:
    raise ValueError("Options --randomizeDataByRun only works with --addRunAxis.")

if isFloatingPOIsTheoryAgnostic:
    raise ValueError(
        "Theory agnostic fit with floating POIs is not currently implemented"
    )

parser = parsing.set_parser_default(
    parser, "aggregateGroups", ["Diboson", "Top", "Wtaunu", "Wmunu"]
)
parser = parsing.set_parser_default(parser, "excludeProcs", ["QCD"])
if args.addIsoMtAxes:
    parser = parsing.set_parser_default(parser, "muonIsolation", [0, 1])

if args.theoryAgnostic:
    if args.genAbsYVbinEdges and any(x < 0.0 for x in args.genAbsYVbinEdges):
        raise ValueError(
            "Option --genAbsYVbinEdges requires all positive values. Please check"
        )

args = parser.parse_args()  # parse again or new defaults won't be propagated

if args.useRefinedVeto and args.useGlobalOrTrackerVeto:
    raise NotImplementedError(
        "Options --useGlobalOrTrackerVeto and --useRefinedVeto cannot be used together at the moment."
    )
if args.validateVetoSF:
    if args.useGlobalOrTrackerVeto or not args.useRefinedVeto:
        raise NotImplementedError(
            "Option --validateVetoSF cannot be used with --useGlobalOrTrackerVeto, and requires --useRefinedVeto at the moment."
        )

# thisAnalysis flag identifies the analysis for the purpose of applying single or dilepton scale factors
# when validating the veto SF only the triggering muon has to be considered to apply the standard SF, so the helpers for a single muon selection are used
thisAnalysis = (
    ROOT.wrem.AnalysisType.Wmass
    if args.validateVetoSF
    else ROOT.wrem.AnalysisType.Wlike
)
isoBranch = muon_selections.getIsoBranch(args.isolationDefinition)
era = args.era

datasets = getDatasets(
    maxFiles=args.maxFiles,
    filt=args.filterProcs,
    excl=args.excludeProcs,
    aux=args.auxiliaryProcs,
    nanoVersion="v9",
    base_path=args.dataPath,
    extended="msht20an3lo" not in args.pdfs,
    era=era,
)

# dilepton invariant mass cuts
mass_min, mass_max = common.get_default_mz_window()

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
if args.fillHistNonTrig:
    if args.validateVetoSF:
        axis_pt = hist.axis.Regular(
            round(template_maxpt - args.vetoRecoPt),
            args.vetoRecoPt,
            template_maxpt,
            name="pt",
            overflow=False,
            underflow=False,
        )
    else:
        axis_pt = hist.axis.Regular(
            template_npt,
            template_minpt,
            template_maxpt,
            name="pt",
            overflow=False,
            underflow=False,
        )
    nominal_axes = [axis_eta, axis_pt, common.axis_charge]
    nominal_cols = ["nonTrigMuons_eta0", "nonTrigMuons_pt0", "nonTrigMuons_charge0"]
else:
    axis_pt = hist.axis.Regular(
        template_npt,
        template_minpt,
        template_maxpt,
        name="pt",
        overflow=False,
        underflow=False,
    )
    nominal_axes = [axis_eta, axis_pt, common.axis_charge]
    nominal_cols = ["trigMuons_eta0", "trigMuons_pt0", "trigMuons_charge0"]

# for isoMt region validation and related tests
# use very high upper edge as a proxy for infinity (cannot exploit overflow bins in the fit)
# can probably use numpy infinity, but this is compatible with the root conversion
axis_mtCat = hist.axis.Variable(
    [0, int(args.mtCut / 2.0), args.mtCut, 1000],
    name="mt",
    underflow=False,
    overflow=False,
)
axis_isoCat = hist.axis.Variable(
    [0, 0.15, 0.3, 100], name="relIso", underflow=False, overflow=False
)

nominal_axes = [axis_eta, axis_pt, common.axis_charge]
nominal_cols = ["trigMuons_eta0", "trigMuons_pt0", "trigMuons_charge0"]
if args.addIsoMtAxes:
    nominal_axes.extend([axis_mtCat, axis_isoCat])
    nominal_cols.extend(["transverseMass", "trigMuons_relIso0"])

if args.unfolding:
    template_wpt = (template_maxpt - template_minpt) / args.unfoldingBins[1]
    min_pt_unfolding = template_minpt + template_wpt
    max_pt_unfolding = template_maxpt - template_wpt
    npt_unfolding = args.unfoldingBins[1] - 2

    unfolding_axes = {}
    unfolding_cols = {}
    for level in args.unfoldingLevels:

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
            datasets = unfolding_tools.add_out_of_acceptance(datasets, group="Zmumu")
            datasets = unfolding_tools.add_out_of_acceptance(datasets, group="Ztautau")
            if len(args.unfoldingLevels) > 1:
                logger.warning(
                    f"Exact unfolding with multiple gen level definitions is not possible, take first one: {args.unfoldingLevels[0]} and continue."
                )
                break

    if args.fitresult:
        unfolding_corr_helper = unfolding_tools.reweight_to_fitresult(args.fitresult)

if args.theoryAgnostic:
    theoryAgnostic_axes, theoryAgnostic_cols = differential.get_theoryAgnostic_axes(
        ptV_bins=args.theoryAgnosticGenPtVbinEdges,
        absYV_bins=args.theoryAgnosticGenAbsYVbinEdges,
        ptV_flow=args.poiAsNoi,
        absYV_flow=args.poiAsNoi,
        wlike=True,
    )
    axis_helicity = helicity_utils.axis_helicity_multidim
    # the following just prepares the existence of the group for out-of-acceptance signal, but doesn't create or define the histogram yet
    if not args.poiAsNoi or (
        args.theoryAgnosticPolVar and args.theoryAgnosticSplitOOA
    ):  # this splitting is not needed for the normVar version of the theory agnostic
        raise ValueError("This option is not currently implemented")

# axes for mT measurement
axis_mt = hist.axis.Regular(200, 0.0, 200.0, name="mt", underflow=False, overflow=True)
axis_eta_mT = hist.axis.Variable([-2.4, 2.4], name="eta")

# define helpers
muon_prefiring_helper, muon_prefiring_helper_stat, muon_prefiring_helper_syst = (
    muon_prefiring.make_muon_prefiring_helpers(era=era)
)

theory_helpers_procs = theory_corrections.make_theory_helpers(
    args.pdfs, args.theoryCorr, corrs=["qcdScale", "alphaS", "pdf"]
)

# extra axes which can be used to label tensor_axes
if args.binnedScaleFactors:
    logger.info("Using binned scale factors and uncertainties")
    # add usePseudoSmoothing=True for tests with Asimov
    muon_efficiency_helper, muon_efficiency_helper_syst, muon_efficiency_helper_stat = (
        muon_efficiencies_binned.make_muon_efficiency_helpers_binned(
            filename=args.sfFile, era=era, max_pt=axis_pt.edges[-1], is_w_like=True
        )
    )
else:
    logger.info("Using smoothed scale factors and uncertainties")
    # if validating veto SF will use the main SF only on triggering muon, so it needs the helper of the single lepton analysis, otherwise it will normally use the one for the Wlike Z
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
logger.info(f"SF file: {args.sfFile}")

muon_efficiency_helper_syst_altBkg = {}
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

if args.validateVetoSF:
    logger.warning(
        "Validating veto SF using Wlike workflow: it will apply single muon scale factors on the triggering muon, and veto SF on the non triggering one"
    )
    logger.warning(
        "Note: single muon SF uncertainties are propagated using the triggering muon, and veto SF uncertainties are propagated using the non triggering one"
    )
    if args.useRefinedVeto:
        (
            muon_efficiency_veto_helper,
            muon_efficiency_veto_helper_syst,
            muon_efficiency_veto_helper_stat,
        ) = (
            wremnants.muon_efficiencies_veto_newVeto.make_muon_efficiency_helpers_newVeto
        )
    else:
        pass
    # we don't store the veto SF for this version at the moment, I think, so I can't run this validation yet
    #    muon_efficiency_veto_helper, muon_efficiency_veto_helper_syst, muon_efficiency_veto_helper_stat = wremnants.muon_efficiencies_veto.make_muon_efficiency_helpers_veto(useGlobalOrTrackerVeto = useGlobalOrTrackerVeto, era = era)

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
    mc_jpsi_crctn_unc_helper,
    data_jpsi_crctn_unc_helper,
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
corr_helpers = theory_corrections.load_corr_helpers(
    [d.name for d in datasets if d.name in common.vprocs], theory_corrs
)

# helpers for muRmuF MiNNLO polynomial variations
if args.theoryAgnosticPolVar:
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

    recoilHelper = recoil_tools.Recoil("highPU", args, flavor="mumu")


def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")
    results = []
    isW = dataset.name in common.wprocs
    isZ = dataset.name in common.zprocs
    isWorZ = isW or isZ

    theory_helpers = None
    if isWorZ:
        theory_helpers = theory_helpers_procs[dataset.name[0]]

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    df = df.DefinePerSample("unity", "1.0")
    df = df.Define(
        "isEvenEvent", f"event % 2 {'!=' if args.flipEventNumberSplitting else '=='} 0"
    )
    weightsum = df.SumAndCount("weight")

    axes = nominal_axes
    cols = nominal_cols

    if args.addMuonPhiAxis is not None:
        axes = [*axes, make_muon_phi_axis(args.addMuonPhiAxis)]
        cols = [*cols, "trigMuons_phi0"]

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

    if args.unfolding and isZ:
        df = unfolding_tools.define_gen_level(
            df, dataset.name, args.unfoldingLevels, mode=analysis_label
        )
        cutsmap = {
            "pt_min": template_minpt,
            "pt_max": template_maxpt,
            "mtw_min": args.mtCut,
            "abseta_max": template_maxeta,
            "mass_min": mass_min,
            "mass_max": mass_max,
        }

        if hasattr(dataset, "out_of_acceptance"):
            df = unfolding_tools.select_fiducial_space(
                df, mode=analysis_label, accept=False, **cutsmap
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
                    [c for c in unfolding_cols[level] if c != f"{level}_acceptance"],
                    base_name=level,
                )
                if not args.poiAsNoi:
                    axes = [*nominal_axes, *unfolding_axes[level]]
                    cols = [*nominal_cols, *unfolding_cols[level]]
                    break

    if isZ:
        df = theory_tools.define_prefsr_vars(df)
        df = df.Define(
            "qtOverQ", "ptVgen/massVgen"
        )  # FIXME: should there be a protection against mass=0 and what value to use?

    df = df.Filter(muon_selections.hlt_string(era))

    df = muon_selections.veto_electrons(df)
    df = muon_selections.apply_met_filters(df)

    cvh_helper = data_calibration_helper if dataset.is_data else mc_calibration_helper
    jpsi_helper = data_jpsi_crctn_helper if dataset.is_data else mc_jpsi_crctn_helper

    df = muon_calibration.define_corrected_muons(
        df, cvh_helper, jpsi_helper, args, dataset, smearing_helper, bias_helper
    )

    df = muon_selections.select_veto_muons(df, nMuons=2, ptCut=args.vetoRecoPt)

    isoThreshold = args.isolationThreshold

    # when validating isolation only one muon (the triggering one) has to pass the tight selection
    if args.validateVetoSF:

        # use lower pt cut from veto, and apply tighter pt cut, medium ID, and isolation later on only on triggering muon
        df = muon_selections.select_good_muons(
            df,
            args.vetoRecoPt,
            template_maxpt,
            dataset.group,
            nMuons=2,
            condition="==",
            use_trackerMuons=args.trackerMuons,
            use_isolation=False,
            isoBranch=isoBranch,
            isoThreshold=isoThreshold,
            requirePixelHits=args.requirePixelHits,
            requireID=False,
        )
        df = muon_selections.define_trigger_muons(df)
        # apply lower pt cut and medium ID on triggering muon
        df = df.Filter(
            f"trigMuons_pt0 > {template_minpt} && Muon_mediumId[trigMuons][0]"
        )
        df = muon_selections.apply_iso_muons(df, 1, 0, isoBranch, isoThreshold)

    else:

        passIsoBoth = args.muonIsolation[0] + args.muonIsolation[1] == 2
        df = muon_selections.select_good_muons(
            df,
            template_minpt,
            template_maxpt,
            dataset.group,
            nMuons=2,
            use_trackerMuons=args.trackerMuons,
            use_isolation=passIsoBoth,
            isoBranch=isoBranch,
            isoThreshold=isoThreshold,
            requirePixelHits=args.requirePixelHits,
        )

        df = muon_selections.define_trigger_muons(df)

        # iso cut applied here, if requested, because it needs the definition of trigMuons and nonTrigMuons from muon_selections.define_trigger_muons
        if not passIsoBoth:
            df = muon_selections.apply_iso_muons(
                df,
                args.muonIsolation[0],
                args.muonIsolation[1],
                isoBranch,
                isoThreshold,
            )

    df = df.Define("trigMuons_relIso0", f"{isoBranch}[trigMuons][0]")
    df = df.Define("nonTrigMuons_relIso0", f"{isoBranch}[nonTrigMuons][0]")
    df = df.Define("trigMuons_passIso0", f"trigMuons_relIso0 < {isoThreshold}")
    df = df.Define("nonTrigMuons_passIso0", f"nonTrigMuons_relIso0 < {isoThreshold}")

    df = muon_selections.select_z_candidate(df, mass_min, mass_max)

    df = muon_selections.select_standalone_muons(
        df, dataset, args.trackerMuons, "trigMuons"
    )
    df = muon_selections.select_standalone_muons(
        df, dataset, args.trackerMuons, "nonTrigMuons"
    )

    df = muon_selections.apply_triggermatching_muon(df, dataset, "trigMuons", era=era)

    useTnpMuonVarForSF = args.useTnpMuonVarForSF
    # in principle these are only needed for MC,
    # but one may want to compare tnp and corrected variables also for data
    if useTnpMuonVarForSF:
        df = df.Define("trigMuons_tnpPt0", "Muon_pt[trigMuons][0]")
        df = df.Define("trigMuons_tnpEta0", "Muon_eta[trigMuons][0]")
        df = df.Define("trigMuons_tnpCharge0", "Muon_charge[trigMuons][0]")
        df = df.Define("nonTrigMuons_tnpPt0", "Muon_pt[nonTrigMuons][0]")
        df = df.Define("nonTrigMuons_tnpEta0", "Muon_eta[nonTrigMuons][0]")
        df = df.Define("nonTrigMuons_tnpCharge0", "Muon_charge[nonTrigMuons][0]")
    else:
        df = df.Alias("trigMuons_tnpPt0", "trigMuons_pt0")
        df = df.Alias("trigMuons_tnpEta0", "trigMuons_eta0")
        df = df.Alias("trigMuons_tnpCharge0", "trigMuons_charge0")
        df = df.Alias("nonTrigMuons_tnpPt0", "nonTrigMuons_pt0")
        df = df.Alias("nonTrigMuons_tnpEta0", "nonTrigMuons_eta0")
        df = df.Alias("nonTrigMuons_tnpCharge0", "nonTrigMuons_charge0")
        #

    if args.forceValidCVH:
        if dataset.is_data:
            if args.muonCorrData == "none":
                logger.warning(
                    "Requiring valid CVH for data even if CVH is not applied"
                )
                df = df.Filter(
                    "Muon_cvhPt[trigMuons][0] > 0 && Muon_cvhPt[nonTrigMuons][0] > 0"
                )
        else:
            if args.muonCorrMC == "none":
                # use CVH with ideal MC geometry for this check
                logger.warning(
                    "Requiring valid CVH (ideal geometry) for MC even if CVH is not applied"
                )
                df = df.Filter(
                    "Muon_cvhidealPt[trigMuons][0] > 0 && Muon_cvhidealPt[nonTrigMuons][0] > 0"
                )

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
                #
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

        muonVarsForSF = [
            "tnpPt0",
            "tnpEta0",
            "SApt0",
            "SAeta0",
            "tnpUT0",
            "tnpCharge0",
            "passIso0",
        ]

        columnsForSF = [
            f"{t}Muons_{v}" for t in ["trig", "nonTrig"] for v in muonVarsForSF
        ]

        df = muon_selections.define_muon_uT_variable(
            df,
            isWorZ,
            smooth3dsf=args.smooth3dsf,
            colNamePrefix="trigMuons",
            addWithTnpMuonVar=useTnpMuonVarForSF,
        )
        df = muon_selections.define_muon_uT_variable(
            df,
            isWorZ,
            smooth3dsf=args.smooth3dsf,
            colNamePrefix="nonTrigMuons",
            addWithTnpMuonVar=useTnpMuonVarForSF,
        )
        # ut is defined in muon_selections.define_muon_uT_variable
        if not useTnpMuonVarForSF:
            df = df.Alias("trigMuons_tnpUT0", "trigMuons_uT0")
            df = df.Alias("nonTrigMuons_tnpUT0", "nonTrigMuons_uT0")

        if not args.smooth3dsf:
            columnsForSF.remove("trigMuons_tnpUT0")
            columnsForSF.remove("nonTrigMuons_tnpUT0")

        if not args.noScaleFactors:
            if args.validateVetoSF:
                columnsForSF[:] = [x for x in columnsForSF if "nonTrigMuons" not in x]
                # apply standard SF only on triggering muon using helper for single lepton, and then apply veto SF for non triggering muon
                df = df.Define(
                    "weight_fullMuonSF_withTrackingReco",
                    muon_efficiency_helper,
                    columnsForSF,
                )
                df = df.Define(
                    "weight_vetoSF_nominal",
                    muon_efficiency_veto_helper,
                    ["nonTrigMuons_pt0", "nonTrigMuons_eta0", "nonTrigMuons_charge0"],
                )
                weight_expr += (
                    "*weight_fullMuonSF_withTrackingReco*weight_vetoSF_nominal"
                )
            else:
                df = df.Define(
                    "weight_fullMuonSF_withTrackingReco",
                    muon_efficiency_helper,
                    columnsForSF,
                )
                weight_expr += "*weight_fullMuonSF_withTrackingReco"

        # prepare inputs for pixel multiplicity helpers
        cvhName = "cvhideal"

        df = df.Define(
            f"trigMuons_{cvhName}NValidPixelHits0",
            f"Muon_{cvhName}NValidPixelHits[trigMuons][0]",
        )
        df = df.Define(
            f"nonTrigMuons_{cvhName}NValidPixelHits0",
            f"Muon_{cvhName}NValidPixelHits[nonTrigMuons][0]",
        )

        df = df.DefinePerSample(
            "MuonNonTrigTrig_triggerCat",
            "ROOT::VecOps::RVec<wrem::TriggerCat>{wrem::TriggerCat::nonTriggering, wrem::TriggerCat::triggering}",
        )
        df = df.Define(
            "MuonNonTrigTrig_eta",
            "ROOT::VecOps::RVec<float>{nonTrigMuons_eta0, trigMuons_eta0}",
        )
        df = df.Define(
            "MuonNonTrigTrig_pt",
            "ROOT::VecOps::RVec<float>{nonTrigMuons_pt0, trigMuons_pt0}",
        )
        df = df.Define(
            "MuonNonTrigTrig_charge",
            "ROOT::VecOps::RVec<int>{nonTrigMuons_charge0, trigMuons_charge0}",
        )
        df = df.Define(
            f"MuonNonTrigTrig_{cvhName}NValidPixelHits",
            f"ROOT::VecOps::RVec<int>{{nonTrigMuons_{cvhName}NValidPixelHits0, trigMuons_{cvhName}NValidPixelHits0}}",
        )

        pixel_multiplicity_cols = [
            "MuonNonTrigTrig_triggerCat",
            "MuonNonTrigTrig_eta",
            "MuonNonTrigTrig_pt",
            "MuonNonTrigTrig_charge",
            f"MuonNonTrigTrig_{cvhName}NValidPixelHits",
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

    results.append(
        df.HistoBoost(
            "weight",
            [hist.axis.Regular(100, -2, 2)],
            ["nominal_weight"],
            storage=hist.storage.Double(),
        )
    )

    if isZ and args.theoryAgnostic:
        df = theoryAgnostic_tools.define_helicity_weights(df, is_z=True)

    if not args.noRecoil:
        leps_uncorr = [
            "Muon_pt[goodMuons][0]",
            "Muon_eta[goodMuons][0]",
            "Muon_phi[goodMuons][0]",
            "Muon_charge[goodMuons][0]",
            "Muon_pt[goodMuons][1]",
            "Muon_eta[goodMuons][1]",
            "Muon_phi[goodMuons][1]",
            "Muon_charge[goodMuons][1]",
        ]
        leps_corr = [
            "trigMuons_pt0",
            "trigMuons_eta0",
            "trigMuons_phi0",
            "trigMuons_charge0",
            "nonTrigMuons_pt0",
            "nonTrigMuons_eta0",
            "nonTrigMuons_phi0",
            "nonTrigMuons_charge0",
        ]
        df = recoilHelper.recoil_Z(
            df, results, dataset, common.zprocs_recoil, leps_uncorr, leps_corr
        )  # produces corrected MET as MET_corr_rec_pt/phi
    else:
        df = df.Alias("MET_corr_rec_pt", "MET_pt")
        df = df.Alias("MET_corr_rec_phi", "MET_phi")

    # TODO improve this to include muon mass?
    ###########
    # utility plots of transverse mass, with or without recoil corrections
    ###########
    met_vars = ("MET_pt", "MET_phi")
    df = df.Define(
        "transverseMass_uncorr",
        f"wrem::get_mt_wlike(trigMuons_pt0, trigMuons_phi0, nonTrigMuons_pt0, nonTrigMuons_phi0, {', '.join(met_vars)})",
    )
    results.append(
        df.HistoBoost(
            "transverseMass_uncorr",
            [axis_mt],
            ["transverseMass_uncorr", "nominal_weight"],
        )
    )
    ###########
    met_vars = ("MET_corr_rec_pt", "MET_corr_rec_phi")
    df = df.Define(
        "met_wlike_TV2",
        f"wrem::get_met_wlike(nonTrigMuons_pt0, nonTrigMuons_phi0, {', '.join(met_vars)})",
    )
    df = df.Define(
        "transverseMass",
        "wrem::get_mt_wlike(trigMuons_pt0, trigMuons_phi0, met_wlike_TV2)",
    )
    results.append(
        df.HistoBoost("transverseMass", [axis_mt], ["transverseMass", "nominal_weight"])
    )
    results.append(
        df.HistoBoost(
            "MET",
            [hist.axis.Regular(20, 0, 100, name="MET")],
            ["MET_corr_rec_pt", "nominal_weight"],
        )
    )
    df = df.Define("met_wlike_TV2_pt", "met_wlike_TV2.Mod()")
    results.append(
        df.HistoBoost(
            "WlikeMET",
            [hist.axis.Regular(20, 0, 100, name="Wlike-MET")],
            ["met_wlike_TV2_pt", "nominal_weight"],
        )
    )
    ###########

    df = df.Define("passWlikeMT", f"transverseMass >= {mtw_min}")

    if not args.onlyMainHistograms and not args.unfolding and not args.addIsoMtAxes:
        axis_mt_coarse = hist.axis.Variable(
            [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 120.0],
            name="mt",
            underflow=False,
            overflow=True,
        )
        axis_trigPassIso = hist.axis.Boolean(name=f"trig_passIso")
        axis_nonTrigPassIso = hist.axis.Boolean(name=f"nonTrig_passIso")

        nominal_bin = df.HistoBoost(
            "nominal_isoMtBins",
            [*axes, axis_trigPassIso, axis_nonTrigPassIso, axis_mt_coarse],
            [
                *cols,
                "trigMuons_passIso0",
                "nonTrigMuons_passIso0",
                "transverseMass",
                "nominal_weight",
            ],
        )
        results.append(nominal_bin)
        nominal_testIsoMtFakeRegions = df.HistoBoost(
            "nominal_testIsoMtFakeRegions",
            [*axes, axis_isoCat, axis_mtCat],
            [*cols, "trigMuons_relIso0", "transverseMass", "nominal_weight"],
        )
        results.append(nominal_testIsoMtFakeRegions)

        axis_eta_nonTrig = hist.axis.Regular(
            template_neta,
            template_mineta,
            template_maxeta,
            name="etaNonTrig",
            overflow=False,
            underflow=False,
        )
        ptMin_nonTrig = args.vetoRecoPt if args.validateVetoSF else template_minpt
        nPtBins_nonTrig = (
            round(template_maxpt - args.vetoRecoPt)
            if args.validateVetoSF
            else template_npt
        )
        axis_pt_nonTrig = hist.axis.Regular(
            nPtBins_nonTrig,
            ptMin_nonTrig,
            template_maxpt,
            name="ptNonTrig",
            overflow=False,
            underflow=False,
        )
        # nonTriggering muon charge can be assumed to be opposite of triggering one
        if args.fillHistNonTrig:
            # cols are already the nonTriggering variables, must invert axes to fill histogram (and rename them)
            axis_eta_trig = hist.axis.Regular(
                template_neta,
                template_mineta,
                template_maxeta,
                name="eta",
                overflow=False,
                underflow=False,
            )
            axis_pt_trig = hist.axis.Regular(
                template_npt,
                template_minpt,
                template_maxpt,
                name="pt",
                overflow=False,
                underflow=False,
            )
            nominal_bothMuons = df.HistoBoost(
                "nominal_bothMuons",
                [
                    axis_eta_trig,
                    axis_pt_trig,
                    common.axis_charge,
                    axis_eta_nonTrig,
                    axis_pt_nonTrig,
                    common.axis_passMT,
                ],
                [
                    "trigMuons_eta0",
                    "trigMuons_pt0",
                    "trigMuons_charge0",
                    "nonTrigMuons_eta0",
                    "nonTrigMuons_pt0",
                    "passWlikeMT",
                    "nominal_weight",
                ],
            )
        else:
            nominal_bothMuons = df.HistoBoost(
                "nominal_bothMuons",
                [*axes, axis_eta_nonTrig, axis_pt_nonTrig, common.axis_passMT],
                [
                    *cols,
                    "nonTrigMuons_eta0",
                    "nonTrigMuons_pt0",
                    "passWlikeMT",
                    "nominal_weight",
                ],
            )
        results.append(nominal_bothMuons)

    # cutting after storing mt distributions for plotting, since the cut is only on corrected met
    if args.dphiMuonMetCut > 0.0:
        df = df.Define(
            "deltaPhiMuonMet",
            "std::abs(wrem::deltaPhi(trigMuons_phi0,met_wlike_TV2.Phi()))",
        )
        df = df.Filter(f"deltaPhiMuonMet > {args.dphiMuonMetCut*np.pi}")

    if isZ:
        # for vertex efficiency plot in MC
        df = df.Define("absDiffGenRecoVtx_z", "std::abs(GenVtx_z - PV_z)")
        df = df.Define("trigMuons_abseta0", "abs(trigMuons_eta0)")
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
            "trigMuons_eta0",
            "trigMuons_passIso0",
            "passWlikeMT",
            "absDiffGenRecoVtx_z",
            "ptVgen",
        ]
        yieldsVertexZstudy = df.HistoBoost(
            "nominal_vertexZstudy",
            [
                axis_abseta,
                common.axis_passIso,
                common.axis_passMT,
                axis_absDiffGenRecoVtx_z,
                axis_prefsrWpt,
            ],
            [*cols_vertexZstudy, "nominal_weight"],
        )
        results.append(yieldsVertexZstudy)

    if not args.addIsoMtAxes:
        df = df.Filter("passWlikeMT")

    if not args.onlyMainHistograms:
        # plot reco vertex distribution before and after PU reweigthing
        # also remove vertex weights since they depend on PU
        if dataset.is_data:
            df = df.DefinePerSample("nominal_weight_noPUandVtx", "1.0")
            df = df.DefinePerSample("nominal_weight_noVtx", "1.0")
            df = df.DefinePerSample("nominal_weight_noSF", "1.0")
        else:
            df = df.Define(
                "nominal_weight_noPUandVtx", "nominal_weight/(weight_pu*weight_vtx)"
            )
            df = df.Define("nominal_weight_noVtx", "nominal_weight/weight_vtx")
            df = df.Define(
                "nominal_weight_noSF",
                "nominal_weight/weight_fullMuonSF_withTrackingReco",
            )

        axis_nRecoVtx = hist.axis.Regular(50, 0.5, 50.5, name="PV_npvsGood")
        axis_fixedGridRhoFastjetAll = hist.axis.Regular(
            50, 0, 50, name="fixedGridRhoFastjetAll"
        )
        results.append(
            df.HistoBoost(
                "PV_npvsGood_uncorr",
                [axis_nRecoVtx],
                ["PV_npvsGood", "nominal_weight_noPUandVtx"],
            )
        )
        results.append(
            df.HistoBoost(
                "PV_npvsGood_noVtx",
                [axis_nRecoVtx],
                ["PV_npvsGood", "nominal_weight_noVtx"],
            )
        )
        results.append(
            df.HistoBoost(
                "PV_npvsGood", [axis_nRecoVtx], ["PV_npvsGood", "nominal_weight"]
            )
        )
        results.append(
            df.HistoBoost(
                "fixedGridRhoFastjetAll_uncorr",
                [axis_nRecoVtx],
                ["fixedGridRhoFastjetAll", "nominal_weight_noPUandVtx"],
            )
        )
        results.append(
            df.HistoBoost(
                "fixedGridRhoFastjetAll_noVtx",
                [axis_nRecoVtx],
                ["fixedGridRhoFastjetAll", "nominal_weight_noVtx"],
            )
        )
        results.append(
            df.HistoBoost(
                "fixedGridRhoFastjetAll",
                [axis_nRecoVtx],
                ["fixedGridRhoFastjetAll", "nominal_weight"],
            )
        )
        df = df.Define(
            "trigMuons_vertexZ0", "PV_z + Muon_dz[trigMuons][0]"
        )  # define at reco level as PV_z + Muon_dz
        axis_vertexZ0 = hist.axis.Regular(200, -20, 20, name="muonVertexZ0")
        results.append(
            df.HistoBoost(
                "trigMuons_vertexZ0_uncorr",
                [axis_vertexZ0, common.axis_charge],
                [
                    "trigMuons_vertexZ0",
                    "trigMuons_charge0",
                    "nominal_weight_noPUandVtx",
                ],
            )
        )
        results.append(
            df.HistoBoost(
                "trigMuons_vertexZ0_noVtx",
                [axis_vertexZ0, common.axis_charge],
                ["trigMuons_vertexZ0", "trigMuons_charge0", "nominal_weight_noVtx"],
            )
        )
        results.append(
            df.HistoBoost(
                "trigMuons_vertexZ0",
                [axis_vertexZ0, common.axis_charge],
                ["trigMuons_vertexZ0", "trigMuons_charge0", "nominal_weight"],
            )
        )

    nominal = df.HistoBoost("nominal", axes, [*cols, "nominal_weight"])
    results.append(nominal)
    nominal_noSF = df.HistoBoost("nominal_noSF", axes, [*cols, "nominal_weight_noSF"])
    results.append(nominal_noSF)

    if useTnpMuonVarForSF and not args.onlyMainHistograms and not args.unfolding:
        df = df.Define(
            "trigMuons_deltaPt_corrMinusTnp", "trigMuons_pt0 - trigMuons_tnpPt0"
        )
        df = df.Define(
            "nonTrigMuons_deltaPt_corrMinusTnp",
            "nonTrigMuons_pt0 - nonTrigMuons_tnpPt0",
        )
        results.append(
            df.HistoBoost(
                "muon_deltaPt_corrMinusTnp",
                [
                    hist.axis.Regular(200, -10, 10, name="trigMuons_dpt"),
                    hist.axis.Regular(200, -10, 10, name="nonTrigMuons_dpt"),
                    common.axis_charge,
                ],
                [
                    "trigMuons_deltaPt_corrMinusTnp",
                    "nonTrigMuons_deltaPt_corrMinusTnp",
                    "trigMuons_charge0",
                    "nominal_weight",
                ],
            )
        )
        df = df.Define(
            "trigMuons_deltaEta_corrMinusTnp", "trigMuons_eta0 - trigMuons_tnpEta0"
        )
        df = df.Define(
            "nonTrigMuons_deltaEta_corrMinusTnp",
            "nonTrigMuons_eta0 - nonTrigMuons_tnpEta0",
        )
        results.append(
            df.HistoBoost(
                "muon_deltaEta_corrMinusTnp",
                [
                    hist.axis.Regular(120, -3.0, 3.0, name="trigMuons_deta"),
                    hist.axis.Regular(120, -3.0, 3.0, name="nonTrigMuons_deta"),
                    common.axis_charge,
                ],
                [
                    "trigMuons_deltaEta_corrMinusTnp",
                    "nonTrigMuons_deltaEta_corrMinusTnp",
                    "trigMuons_charge0",
                    "nominal_weight",
                ],
            )
        )

    if args.poiAsNoi and isZ:
        if args.theoryAgnostic and not hasattr(dataset, "out_of_acceptance"):
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
                if (
                    dataset.name == "Zmumu_2016PostVFP"
                    or dataset.name == "Ztautau_2016PostVFP"
                ):
                    helpers_class = muRmuFPolVar_helpers_Z
                    process_name = "Z"
                for coeffKey in helpers_class.keys():
                    logger.debug(
                        f"Creating muR/muF histograms with polynomial variations for {coeffKey}"
                    )
                    helperQ = helpers_class[coeffKey]
                    df = df.Define(
                        f"muRmuFPolVar_{coeffKey}_tensor",
                        helperQ,
                        theoryAgnostic_helpers_cols,
                    )
                    noiAsPoiWithPolHistName = Datagroups.histName(
                        "nominal", syst=f"muRmuFPolVar{process_name}_{coeffKey}"
                    )
                    results.append(
                        df.HistoBoost(
                            noiAsPoiWithPolHistName,
                            nominal_axes,
                            [*nominal_cols, f"muRmuFPolVar_{coeffKey}_tensor"],
                            tensor_axes=helperQ.tensor_axes,
                            storage=hist.storage.Double(),
                        )
                    )
        if args.unfolding and dataset.name == "Zmumu_2016PostVFP":
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

    if not args.noRecoil and args.recoilUnc:
        df = recoilHelper.add_recoil_unc_Z(df, results, dataset, cols, axes, "nominal")

    if not dataset.is_data and not args.onlyMainHistograms:

        df = syst_tools.add_muon_efficiency_unc_hists(
            results,
            df,
            muon_efficiency_helper_stat,
            muon_efficiency_helper_syst,
            axes,
            cols,
            what_analysis=thisAnalysis,
            singleMuonCollection="trigMuons",
            smooth3D=args.smooth3dsf,
        )
        for es in common.muonEfficiency_altBkgSyst_effSteps:
            df = syst_tools.add_muon_efficiency_unc_hists_altBkg(
                results,
                df,
                muon_efficiency_helper_syst_altBkg[es],
                axes,
                cols,
                singleMuonCollection="trigMuons",
                what_analysis=thisAnalysis,
                step=es,
            )
        if args.validateVetoSF:
            df = syst_tools.add_muon_efficiency_veto_unc_hists(
                results,
                df,
                muon_efficiency_veto_helper_stat,
                muon_efficiency_veto_helper_syst,
                axes,
                cols,
                muons="nonTrigMuons",
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
            )
        else:
            df = syst_tools.add_L1Prefire_unc_hists(
                results,
                df,
                axes,
                cols,
                helper_stat=muon_prefiring_helper_stat,
                helper_syst=muon_prefiring_helper_syst,
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
                for_wmass=False,
            )

            reco_sel = "vetoMuonsPre"
            require_prompt = "tau" not in dataset.name
            df = muon_calibration.define_genFiltered_recoMuonSel(
                df, reco_sel, require_prompt
            )
            reco_sel_GF = muon_calibration.getColName_genFiltered_recoMuonSel(
                reco_sel, require_prompt
            )
            df = muon_calibration.define_matched_gen_muons_kinematics(df, reco_sel_GF)
            df = muon_calibration.calculate_matched_gen_muon_kinematics(df, reco_sel_GF)
            df = muon_calibration.define_matched_reco_muon_kinematics(df, reco_sel_GF)

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
                df = df.Define(
                    "nominal_muonScaleSyst_responseWeights_tensor",
                    data_jpsi_crctn_unc_helper,
                    [*input_kinematics, "nominal_weight"],
                )
                muonScaleSyst_responseWeights = df.HistoBoost(
                    "nominal_muonScaleSyst_responseWeights",
                    axes,
                    [*cols, "nominal_muonScaleSyst_responseWeights_tensor"],
                    tensor_axes=data_jpsi_crctn_unc_helper.tensor_axes,
                    storage=hist.storage.Double(),
                )
                results.append(muonScaleSyst_responseWeights)

                df = muon_calibration.add_resolution_uncertainty(
                    df, axes, results, cols, smearing_uncertainty_helper, reco_sel_GF
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

                if args.nonClosureScheme in ["A-M-separated", "A-only"]:
                    # add the ad-hoc Z non-closure nuisances from the jpsi massfit to muon scale unc
                    df = df.DefinePerSample("AFlag", "0x01")
                    df = df.Define(
                        "Z_non_closure_parametrized_A",
                        z_non_closure_parametrized_helper,
                        [*input_kinematics, "nominal_weight", "AFlag"],
                    )
                    hist_Z_non_closure_parametrized_A = df.HistoBoost(
                        "nominal_Z_non_closure_parametrized_A",
                        axes,
                        [*cols, "Z_non_closure_parametrized_A"],
                        tensor_axes=z_non_closure_parametrized_helper.tensor_axes,
                        storage=hist.storage.Double(),
                    )
                    results.append(hist_Z_non_closure_parametrized_A)

                if args.nonClosureScheme in [
                    "A-M-separated",
                    "binned-plus-M",
                    "M-only",
                ]:
                    df = df.DefinePerSample("MFlag", "0x04")
                    df = df.Define(
                        "Z_non_closure_parametrized_M",
                        z_non_closure_parametrized_helper,
                        [*input_kinematics, "nominal_weight", "MFlag"],
                    )
                    hist_Z_non_closure_parametrized_M = df.HistoBoost(
                        "nominal_Z_non_closure_parametrized_M",
                        axes,
                        [*cols, "Z_non_closure_parametrized_M"],
                        tensor_axes=z_non_closure_parametrized_helper.tensor_axes,
                        storage=hist.storage.Double(),
                    )
                    results.append(hist_Z_non_closure_parametrized_M)

                if args.nonClosureScheme == "A-M-combined":
                    df = df.DefinePerSample("AMFlag", "0x01 | 0x04")
                    df = df.Define(
                        "Z_non_closure_parametrized",
                        z_non_closure_parametrized_helper,
                        [*input_kinematics, "nominal_weight", "AMFlag"],
                    )
                    hist_Z_non_closure_parametrized = df.HistoBoost(
                        (
                            "Z_non_closure_parametrized_gaus"
                            if args.muonScaleVariation == "smearingWeightsGaus"
                            else "nominal_Z_non_closure_parametrized"
                        ),
                        axes,
                        [*cols, "Z_non_closure_parametrized"],
                        tensor_axes=z_non_closure_parametrized_helper.tensor_axes,
                        storage=hist.storage.Double(),
                    )
                    results.append(hist_Z_non_closure_parametrized)

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

    if hasattr(dataset, "out_of_acceptance"):
        # Rename dataset to not overwrite the original one
        dataset.name = dataset.name + "OOA"

    return results, weightsum


resultdict = narf.build_and_run(datasets, build_graph)

if not args.noScaleToData:
    scale_to_data(resultdict)
    aggregate_groups(datasets, resultdict, args.aggregateGroups)

write_analysis_output(
    resultdict, f"{os.path.basename(__file__).replace('py', 'hdf5')}", args
)
