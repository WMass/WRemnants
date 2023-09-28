import argparse
from utilities import output_tools, common, rdf_tools, logging, differential

parser,initargs = common.common_parser(True)

import ROOT
import narf
import wremnants
from wremnants import theory_tools,syst_tools,theory_corrections, muon_calibration, muon_selections, muon_validation, unfolding_tools, theoryAgnostic_tools, helicity_utils
from wremnants.histmaker_tools import scale_to_data, aggregate_groups
from wremnants.datasets.dataset_tools import getDatasets
import hist
import lz4.frame
import math
import time
from utilities import boostHistHelpers as hh
import pathlib
import os
import numpy as np

data_dir = common.data_dir
parser.add_argument("--noScaleFactors", action="store_true", help="Don't use scale factors for efficiency (legacy option for tests)")
parser.add_argument("--lumiUncertainty", type=float, help="Uncertainty for luminosity in excess to 1 (e.g. 1.012 means 1.2\%)", default=1.012)
parser.add_argument("--noGenMatchMC", action='store_true', help="Don't use gen match filter for prompt muons with MC samples (note: QCD MC never has it anyway)")
parser.add_argument("--theoryAgnostic", action='store_true', help="Add V qT,Y axes and helicity axes for W samples")
parser.add_argument("--halfStat", action='store_true', help="Test half data and MC stat, selecting odd events, just for tests")
parser.add_argument("--makeMCefficiency", action="store_true", help="Save yields vs eta-pt-ut-passMT-passIso-passTrigger to derive 3D efficiencies for MC isolation and trigger (can run also with --onlyMainHistograms)")
parser.add_argument("--onlyTheorySyst", action="store_true", help="Keep only theory systematic variations, mainly for tests")
parser.add_argument("--oneMCfileEveryN", type=int, default=None, help="Use 1 MC file every N, where N is given by this option. Mainly for tests")
parser.add_argument("--noAuxiliaryHistograms", action="store_true", help="Remove auxiliary histograms to save memory (removed by default with --unfolding or --theoryAgnostic)")

args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

if args.theoryAgnostic or args.unfolding:
    parser = common.set_parser_default(parser, "excludeFlow", True)
    if args.theoryAgnostic:
        # temporary, to ensure running with stat only until systematics are all implemented
        logger.warning("Running theory agnostic with only nominal and mass weight histograms for now.")
        parser = common.set_parser_default(parser, "onlyMainHistograms", True)
        parser = common.set_parser_default(parser, "genVars", ["absYVgenSig", "ptVgenSig", "helicity"])
    args = parser.parse_args()
    
thisAnalysis = ROOT.wrem.AnalysisType.Wmass
datasets = getDatasets(maxFiles=args.maxFiles,
                       filt=args.filterProcs,
                       excl=args.excludeProcs, 
                       nanoVersion="v8" if args.v8 else "v9", base_path=args.dataPath, oneMCfileEveryN=args.oneMCfileEveryN)

era = args.era

# transverse boson mass cut
mtw_min = 40

# custom template binning
template_neta = int(args.eta[0])
template_mineta = args.eta[1]
template_maxeta = args.eta[2]
logger.info(f"Eta binning: {template_neta} bins from {template_mineta} to {template_maxeta}")
template_npt = int(args.pt[0])
template_minpt = args.pt[1]
template_maxpt = args.pt[2]
logger.info(f"Pt binning: {template_npt} bins from {template_minpt} to {template_maxpt}")

# standard regular axes
axis_eta = hist.axis.Regular(template_neta, template_mineta, template_maxeta, name = "eta", overflow=False, underflow=False)
axis_pt = hist.axis.Regular(template_npt, template_minpt, template_maxpt, name = "pt", overflow=False, underflow=False)

axis_charge = common.axis_charge
axis_passIso = common.axis_passIso
axis_passMT = common.axis_passMT
axis_passTrigger = hist.axis.Boolean(name = "passTrigger")

nominal_axes = [axis_eta, axis_pt, axis_charge, axis_passIso, axis_passMT]

nominal_cols = ["goodMuons_eta0", "goodMuons_pt0", "goodMuons_charge0", "passIso", "passMT"]

# axes for W MC efficiencies with uT dependence for iso and trigger
axis_pt_eff_list = [24.,26.,28.,30.,32.,34.,36.,38.,40., 42., 44., 47., 50., 55., 60., 65.]
axis_pt_eff = hist.axis.Variable(axis_pt_eff_list, name = "pt", overflow=not args.excludeFlow, underflow=not args.excludeFlow)
axis_ut = hist.axis.Regular(40, -100, 100, overflow=not args.excludeFlow, underflow=not args.excludeFlow, name = "ut")
axes_WeffMC = [axis_eta, axis_pt_eff, axis_ut, axis_charge, axis_passIso, axis_passMT, axis_passTrigger]
# sum those groups up in post processing
groups_to_aggregate = args.aggregateGroups

if args.unfolding:

    unfolding_axes, unfolding_cols = differential.get_pt_eta_axes(args.genBins[0], template_minpt, template_maxpt, args.genBins[1])
    datasets = unfolding_tools.add_out_of_acceptance(datasets, group = "Wmunu")
    groups_to_aggregate.append("BkgWmunu")

elif args.theoryAgnostic:

    theoryAgnostic_axes, theoryAgnostic_cols = differential.get_theoryAgnostic_axes()
    axis_helicity = helicity_utils.axis_helicity_multidim
    # the following just prepares the existence of the group for out-of-acceptance signal, but doesn't create or define the histogram yet
    datasets = unfolding_tools.add_out_of_acceptance(datasets, group = "Wmunu")
    groups_to_aggregate.append("BkgWmunu")

# axes for study of fakes
axis_mt_fakes = hist.axis.Regular(120, 0., 120., name = "mt", underflow=False, overflow=True)
axis_dphi_fakes = hist.axis.Regular(8, 0., np.pi, name = "DphiMuonMet", underflow=False, overflow=False)
axis_hasjet_fakes = hist.axis.Boolean(name = "hasJets") # only need case with 0 jets or > 0 for now
mTStudyForFakes_axes = [axis_eta, axis_pt, axis_charge, axis_mt_fakes, axis_passIso, axis_hasjet_fakes, axis_dphi_fakes]

# for mt, met, ptW plots, to compute the fakes properly (but FR pretty stable vs pt and also vs eta)
# may not exactly reproduce the same pt range as analysis, though
axis_eta_utilityHist = hist.axis.Regular(24, -2.4, 2.4, name = "eta", overflow=False, underflow=False)
axis_pt_utilityHist = hist.axis.Regular(6, 26, 56, name = "pt", overflow=False, underflow=False)

axis_met = hist.axis.Regular(200, 0., 200., name = "met", underflow=False, overflow=True)

# define helpers
muon_prefiring_helper, muon_prefiring_helper_stat, muon_prefiring_helper_syst = wremnants.make_muon_prefiring_helpers(era = era)

qcdScaleByHelicity_helper = wremnants.makeQCDScaleByHelicityHelper()

if args.noScaleFactors:
    logger.info("Running with no scale factors")
elif args.binnedScaleFactors:
    logger.info("Using binned scale factors and uncertainties")
    # add usePseudoSmoothing=True for tests with Asimov
    muon_efficiency_helper, muon_efficiency_helper_syst, muon_efficiency_helper_stat = wremnants.make_muon_efficiency_helpers_binned(filename = data_dir + "/testMuonSF/allSmooth_GtoH3D.root", era = era, max_pt = axis_pt.edges[-1], usePseudoSmoothing=True)
else:
    logger.info("Using smoothed scale factors and uncertainties")
    muon_efficiency_helper, muon_efficiency_helper_syst, muon_efficiency_helper_stat = wremnants.make_muon_efficiency_helpers_smooth(filename = args.sfFile, era = era, what_analysis = thisAnalysis, max_pt = axis_pt.edges[-1], isoEfficiencySmoothing = args.isoEfficiencySmoothing, smooth3D=args.smooth3dsf)
    ## this is needed to define the syst on SF from 2D ut-integrated and original no-ut-dependent SF  
    if not args.smooth3dsf and not args.sf2DnoUt:
        muon_efficiency_helper2d, muon_efficiency_helper_syst2d, muon_efficiency_helper_stat2d = wremnants.make_muon_efficiency_helpers_smooth(filename = data_dir + "/testMuonSF/allSmooth_GtoHout.root", era = era, max_pt = axis_pt.edges[-1], isoEfficiencySmoothing = args.isoEfficiencySmoothing, smooth3D=args.smooth3dsf)

logger.info(f"SF file: {args.sfFile}")

pileup_helper = wremnants.make_pileup_helper(era = era)
vertex_helper = wremnants.make_vertex_helper(era = era)

calib_filepaths = common.calib_filepaths
closure_filepaths = common.closure_filepaths

diff_weights_helper = ROOT.wrem.SplinesDifferentialWeightsHelper(calib_filepaths['tflite_file']) if (args.muonScaleVariation == 'smearingWeightsSplines' or args.validationHists) else None

mc_jpsi_crctn_helper, data_jpsi_crctn_helper, jpsi_crctn_MC_unc_helper, jpsi_crctn_data_unc_helper = muon_calibration.make_jpsi_crctn_helpers(args, calib_filepaths, make_uncertainty_helper=True)

z_non_closure_parametrized_helper, z_non_closure_binned_helper = muon_calibration.make_Z_non_closure_helpers(args, calib_filepaths, closure_filepaths)

mc_calibration_helper, data_calibration_helper, calibration_uncertainty_helper = muon_calibration.make_muon_calibration_helpers(args)

smearing_helper, smearing_uncertainty_helper = (None, None) if args.noSmearing else muon_calibration.make_muon_smearing_helpers()

bias_helper = muon_calibration.make_muon_bias_helpers(args) if args.biasCalibration else None

corr_helpers = theory_corrections.load_corr_helpers([d.name for d in datasets if d.name in common.vprocs], args.theoryCorr)

# recoil initialization
if not args.noRecoil:
    from wremnants import recoil_tools
    recoilHelper = recoil_tools.Recoil("highPU", args, flavor="mu")

######################################################
######################################################
######################################################
## FIXME/TODO
## next function should have been imported from theoryAgnostic_tools.py, but requires too many things as input,
## such as the helpers created here. Since it is effectively a specialization of the loop flow,
## it is part of the histmaker and is probably fine to have it here.
## In fact, having this custom function overriding the main graph is probably not the best idea, should rather use the same

# graph building for W sample with helicity weights
def setTheoryAgnosticGraph(df, results, dataset, reco_sel_GF, era, nominal_axes_thAgn, nominal_cols_thAgn, args):
    logger.info(f"Setting theory agnostic graph for {dataset.name}")
    df = theoryAgnostic_tools.define_helicity_weights(df)
    nominalByHelicity = df.HistoBoost("nominal", nominal_axes_thAgn, [*nominal_cols_thAgn, "nominal_weight_helicity"], tensor_axes=[axis_helicity])
    results.append(nominalByHelicity)

    if not args.onlyMainHistograms:
        if not args.onlyTheorySyst:
            df = syst_tools.add_L1Prefire_unc_hists(results, df, muon_prefiring_helper_stat, muon_prefiring_helper_syst, nominal_axes_thAgn, nominal_cols_thAgn, addhelicity=True)
            df = syst_tools.add_muon_efficiency_unc_hists(results, df, muon_efficiency_helper_stat, muon_efficiency_helper_syst, nominal_axes_thAgn, nominal_cols_thAgn, what_analysis=thisAnalysis, addhelicity=True)
        df = syst_tools.add_theory_hists(results, df, args, dataset.name, corr_helpers, qcdScaleByHelicity_helper, nominal_axes_thAgn, nominal_cols_thAgn, for_wmass=True, addhelicity=True)
    else:
        #FIXME: hardcoded to keep mass weights, this would be done in add_theory_hists
        df = syst_tools.define_mass_weights(df, dataset.name)
        syst_tools.add_massweights_hist(results, df, nominal_axes_thAgn, nominal_cols_thAgn, proc=dataset.name, addhelicity=True)
######################################################
######################################################
######################################################
    
smearing_weights_procs = []

def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")
    results = []
    isW = dataset.name in common.wprocs
    isWmunu = dataset.name in ["WplusmunuPostVFP", "WminusmunuPostVFP"]
    isZ = dataset.name in common.zprocs
    isWorZ = isW or isZ
    isTop = dataset.group == "Top"
    isQCDMC = dataset.group == "QCD"
    require_prompt = "tau" not in dataset.name # for muon GEN-matching   
    
    # disable auxiliary histograms when unfolding to reduce memory consumptions
    auxiliary_histograms = not args.unfolding and not args.theoryAgnostic and not args.noAuxiliaryHistograms

    apply_theory_corr = args.theoryCorr and dataset.name in corr_helpers

    cvh_helper = data_calibration_helper if dataset.is_data else mc_calibration_helper
    jpsi_helper = data_jpsi_crctn_helper if dataset.is_data else mc_jpsi_crctn_helper

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    weightsum = df.SumAndCount("weight")

    axes = nominal_axes
    cols = nominal_cols

    if args.unfolding and isWmunu:
        df = unfolding_tools.define_gen_level(df, args.genLevel, dataset.name, mode="wmass")
        if hasattr(dataset, "out_of_acceptance"):
            logger.debug("Reject events in fiducial phase space")
            df = unfolding_tools.select_fiducial_space(df, mode="wmass", pt_min=args.pt[1], pt_max=args.pt[2], accept=False)
        else:
            logger.debug("Select events in fiducial phase space")
            df = unfolding_tools.select_fiducial_space(df, mode="wmass", pt_min=args.pt[1], pt_max=args.pt[2], accept=True)

            unfolding_tools.add_xnorm_histograms(results, df, args, dataset.name, corr_helpers, qcdScaleByHelicity_helper, unfolding_axes, unfolding_cols)
            axes = [*nominal_axes, *unfolding_axes] 
            cols = [*nominal_cols, *unfolding_cols]

    if args.theoryAgnostic and isWmunu: # should be isW to do also Wtaunu
        df = theory_tools.define_prefsr_vars(df)
        if hasattr(dataset, "out_of_acceptance"):
            logger.debug("Reject events in fiducial phase space")
            df = theoryAgnostic_tools.select_fiducial_space(df, theoryAgnostic_axes[0].edges[-1], theoryAgnostic_axes[1].edges[-1], accept=False)
        else:
            logger.debug("Select events in fiducial phase space for theory agnostic analysis")
            df = theoryAgnostic_tools.select_fiducial_space(df, theoryAgnostic_axes[0].edges[-1], theoryAgnostic_axes[1].edges[-1], accept=True)
            theoryAgnostic_tools.add_xnorm_histograms(results, df, args, dataset.name, corr_helpers, qcdScaleByHelicity_helper, theoryAgnostic_axes, theoryAgnostic_cols)
            # helicity axis is special, defined through a tensor later, theoryAgnostic_ only includes W rapidity and pt for now
            axes = [*nominal_axes, *theoryAgnostic_axes]
            cols = [*nominal_cols, *theoryAgnostic_cols]

    if not args.makeMCefficiency:
        # remove trigger, it will be part of the efficiency selection for passing trigger
        df = df.Filter("HLT_IsoTkMu24 || HLT_IsoMu24")

    if args.halfStat:
        df = df.Filter("event % 2 == 1") # test with odd/even events

    df = muon_calibration.define_corrected_muons(df, cvh_helper, jpsi_helper, args, dataset, smearing_helper, bias_helper)

    df = muon_selections.select_veto_muons(df, nMuons=1)
    df = muon_selections.select_good_muons(df, template_minpt, template_maxpt, nMuons=1, use_trackerMuons=args.trackerMuons, use_isolation=False)

    # the corrected RECO muon kinematics, which is intended to be used as the nominal
    df = muon_calibration.define_corrected_reco_muon_kinematics(df)

    df = muon_selections.select_standalone_muons(df, dataset, args.trackerMuons, "goodMuons")
 
    df = muon_selections.veto_electrons(df)
    df = muon_selections.apply_met_filters(df)
    if args.makeMCefficiency:
        if dataset.group in common.background_MCprocs:
            df = df.Define("GoodTrigObjs", "wrem::goodMuonTriggerCandidate(TrigObj_id,TrigObj_pt,TrigObj_l1pt,TrigObj_l2pt,TrigObj_filterBits)")
        else:
            df = df.Define("GoodTrigObjs", "wrem::goodMuonTriggerCandidate(TrigObj_id,TrigObj_filterBits)")
        df = df.Define("passTrigger","(HLT_IsoTkMu24 || HLT_IsoMu24) && wrem::hasTriggerMatch(goodMuons_eta0,goodMuons_phi0,TrigObj_eta[GoodTrigObjs],TrigObj_phi[GoodTrigObjs])")

    else:
        df = muon_selections.apply_triggermatching_muon(df, dataset, "goodMuons_eta0", "goodMuons_phi0")

    # gen match to bare muons to select only prompt muons from MC processes, but also including tau decays
    # status flags in NanoAOD: https://cms-nanoaod-integration.web.cern.ch/autoDoc/NanoAODv9/2016ULpostVFP/doc_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1.html
    if not dataset.is_data and not isQCDMC and not args.noGenMatchMC:
        df = df.Define("postFSRmuons", "GenPart_status == 1 && (GenPart_statusFlags & 1 || GenPart_statusFlags & (5<<1)) && abs(GenPart_pdgId) == 13")
        df = df.Filter("wrem::hasMatchDR2(goodMuons_eta0,goodMuons_phi0,GenPart_eta[postFSRmuons],GenPart_phi[postFSRmuons],0.09)")
        
    if isWorZ:
        df = muon_validation.define_cvh_reco_muon_kinematics(df)
        reco_sel = "vetoMuonsPre"
        df = muon_calibration.define_genFiltered_recoMuonSel(df, reco_sel, require_prompt)
        reco_sel_GF = muon_calibration.getColName_genFiltered_recoMuonSel(reco_sel, require_prompt)
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

        for var in ['Pt', 'Eta', 'Charge', 'Qop']:
            df = df.Define(f"{reco_sel}_{var.lower()}0_gen", f"{reco_sel}_gen{var.capitalize()}[0]")
            df = df.Define(f"{reco_sel_GF}_{var.lower()}0_gen", f"{reco_sel_GF}_gen{var.capitalize()}[0]")

            df = df.Define(f"{reco_sel}_{var.lower()}0_gen_smeared", f"{reco_sel}_genSmeared{var.capitalize()}[0]")
            df = df.Define(f"{reco_sel_GF}_{var.lower()}0_gen_smeared", f"{reco_sel_GF}_genSmeared{var.capitalize()}[0]")
            if var != 'Qop':
                df = df.Define(f"{reco_sel_GF}_{var.lower()}0_reco", f"{reco_sel_GF}_reco{var.capitalize()}[0]")
        df = df.Define(f"{reco_sel_GF}_covMat0", f"{reco_sel_GF}_covMat[0]")

        if args.validationHists:
            for reco_type in ['crctd', 'cvh', 'uncrct', 'gen_smeared']:
                df = muon_validation.define_reco_over_gen_cols(df, reco_type)

    df = df.Define("goodMuons_pfRelIso04_all0", "Muon_pfRelIso04_all[goodMuons][0]")

    # Jet collection actually has a pt threshold of 15 GeV in MiniAOD 
    df = df.Define("goodCleanJetsNoPt", "Jet_jetId >= 6 && (Jet_pt > 50 || Jet_puId >= 4) && abs(Jet_eta) < 2.4 && wrem::cleanJetsFromLeptons(Jet_eta,Jet_phi,Muon_correctedEta[vetoMuons],Muon_correctedPhi[vetoMuons],Electron_eta[vetoElectrons],Electron_phi[vetoElectrons])")
    df = df.Define("passIso", "goodMuons_pfRelIso04_all0 < 0.15")

    ########################################################################
    # define event weights here since they are needed below for some helpers
    if dataset.is_data:
        df = df.DefinePerSample("nominal_weight", "1.0")            
    else:
        df = df.Define("weight_pu", pileup_helper, ["Pileup_nTrueInt"])
        df = df.Define("weight_vtx", vertex_helper, ["GenVtx_z", "Pileup_nTrueInt"])
        df = df.Define("weight_newMuonPrefiringSF", muon_prefiring_helper, ["Muon_correctedEta", "Muon_correctedPt", "Muon_correctedPhi", "Muon_correctedCharge", "Muon_looseId"])

        weight_expr = "weight_pu*weight_newMuonPrefiringSF*L1PreFiringWeight_ECAL_Nom"
        # define recoil uT, muon projected on boson pt, the latter is made using preFSR variables
        # TODO: fix it for not W/Z processes
        columnsForSF = ["goodMuons_pt0", "goodMuons_eta0", "goodMuons_SApt0", "goodMuons_SAeta0", "goodMuons_uT0", "goodMuons_charge0", "passIso"]
        df = muon_selections.define_muon_uT_variable(df, isWorZ, smooth3dsf=args.smooth3dsf, colNamePrefix="goodMuons")
        if not args.smooth3dsf:
            columnsForSF.remove("goodMuons_uT0")
            
        if not args.noScaleFactors:
            df = df.Define("weight_fullMuonSF_withTrackingReco", muon_efficiency_helper, columnsForSF)
            weight_expr += "*weight_fullMuonSF_withTrackingReco"

        if not args.noVertexWeight:
            weight_expr += "*weight_vtx"
        
        df = df.Define("exp_weight", weight_expr)
        df = theory_tools.define_theory_weights_and_corrs(df, dataset.name, corr_helpers, args)

    ########################################################################
    
    if not args.noRecoil:
        lep_cols = ["goodMuons_pt0", "goodMuons_phi0", "goodMuons_charge0", "Muon_pt[goodMuons][0]"]
        df = recoilHelper.recoil_W(df, results, dataset, common.vprocs, lep_cols, cols_fakerate=nominal_cols, axes_fakerate=nominal_cols, mtw_min=mtw_min) # produces corrected MET as MET_corr_rec_pt/phi
    else:
        df = df.Alias("MET_corr_rec_pt", "MET_pt")
        df = df.Alias("MET_corr_rec_phi", "MET_phi")

    df = df.Define("transverseMass", "wrem::mt_2(goodMuons_pt0, goodMuons_phi0, MET_corr_rec_pt, MET_corr_rec_phi)")
    df = df.Define("hasCleanJet", "Sum(goodCleanJetsNoPt && Jet_pt > 30) >= 1")

    df = df.Define("deltaPhiMuonMet", "std::abs(wrem::deltaPhi(goodMuons_phi0,MET_corr_rec_phi))")

    if auxiliary_histograms: 
        # couple of histograms specific for tests with fakes
        mTStudyForFakes = df.HistoBoost("mTStudyForFakes", mTStudyForFakes_axes, ["goodMuons_eta0", "goodMuons_pt0", "goodMuons_charge0", "transverseMass", "passIso", "hasCleanJet", "deltaPhiMuonMet", "nominal_weight"])
        results.append(mTStudyForFakes)

    # add filter of deltaPhi(muon,met) before other histograms (but before the previous histogram for test with fakes)
    if not args.makeMCefficiency:
        dphiMuonMetCut = args.dphiMuonMetCut * np.pi
        df = df.Filter(f"deltaPhiMuonMet > {dphiMuonMetCut}") # pi/4 was found to be a good threshold for signal with mT > 40 GeV

    df = df.Define("passMT", f"transverseMass >= {mtw_min}")

    if auxiliary_histograms:
        # utility plot, mt and met, to plot them later
        results.append(df.HistoBoost("MET", [axis_met, axis_eta_utilityHist, axis_pt_utilityHist, axis_charge, axis_passIso, axis_passMT], ["MET_corr_rec_pt", "goodMuons_eta0", "goodMuons_pt0", "goodMuons_charge0", "passIso", "passMT", "nominal_weight"]))
        results.append(df.HistoBoost("transverseMass", [axis_mt_fakes, axis_eta_utilityHist, axis_pt_utilityHist, axis_charge, axis_passIso], ["transverseMass", "goodMuons_eta0", "goodMuons_pt0", "goodMuons_charge0", "passIso", "nominal_weight"]))

    ## TODO: next part should be improved, there is quite a lot of duplication of what could happen later in the loop
    ## FIXME: should be isW, to include Wtaunu
    if isWmunu and args.theoryAgnostic:
        setTheoryAgnosticGraph(df, results, dataset, reco_sel_GF, era, axes, cols, args)
        if hasattr(dataset, "out_of_acceptance"):
            # Rename dataset to not overwrite the original one
            dataset.name = "Bkg"+dataset.name
        return results, weightsum
        
    if not args.onlyMainHistograms:
        syst_tools.add_QCDbkg_jetPt_hist(results, df, axes, cols, jet_pt=30)

    if dataset.is_data:
        nominal = df.HistoBoost("nominal", axes, cols)
        results.append(nominal)

    else:  
        nominal = df.HistoBoost("nominal", axes, [*cols, "nominal_weight"])
        results.append(nominal)
        results.append(df.HistoBoost("nominal_weight", [hist.axis.Regular(200, -4, 4)], ["nominal_weight"], storage=hist.storage.Double()))
        
        if args.makeMCefficiency:
            cols_WeffMC = ["goodMuons_eta0", "goodMuons_pt0", "goodMuons_uT0", "goodMuons_charge0",
                           "passIso", "passMT", "passTrigger"]
            yieldsForWeffMC = df.HistoBoost("yieldsForWeffMC", axes_WeffMC, [*cols_WeffMC, "nominal_weight"])
            results.append(yieldsForWeffMC)
        # df = df.Filter(f"wrem::printVar(nominal_weight)")
            
        if not args.noRecoil and args.recoilUnc:
            df = recoilHelper.add_recoil_unc_W(df, results, dataset, cols, axes, "nominal")
        if apply_theory_corr:
            results.extend(theory_tools.make_theory_corr_hists(df, "nominal", axes, cols, 
                corr_helpers[dataset.name], args.theoryCorr, modify_central_weight=not args.theoryCorrAltOnly, isW = isW)
            )
        if isWorZ:
            cols_gen, cols_gen_smeared = muon_calibration.make_alt_reco_and_gen_hists(df, results, axes, cols, reco_sel_GF)
            if args.validationHists: 
                muon_validation.make_reco_over_gen_hists(df, results)

    if not dataset.is_data and not args.onlyMainHistograms:

        if not args.onlyTheorySyst:
            if not args.noScaleFactors:
                df = syst_tools.add_muon_efficiency_unc_hists(results, df, muon_efficiency_helper_stat, muon_efficiency_helper_syst, axes, cols, what_analysis=thisAnalysis, smooth3D=args.smooth3dsf)
            df = syst_tools.add_L1Prefire_unc_hists(results, df, muon_prefiring_helper_stat, muon_prefiring_helper_syst, axes, cols)
            # luminosity, as shape variation despite being a flat scaling to facilitate propagation to fakes
            df = syst_tools.add_luminosity_unc_hists(results, df, args, axes, cols)
                
        # n.b. this is the W analysis so mass weights shouldn't be propagated
        # on the Z samples (but can still use it for dummy muon scale)
        
        if isWorZ:

            df = syst_tools.add_theory_hists(results, df, args, dataset.name, corr_helpers, qcdScaleByHelicity_helper, axes, cols, for_wmass=True)

            # Don't think it makes sense to apply the mass weights to scale leptons from tau decays
            if not args.onlyTheorySyst and not "tau" in dataset.name:
                df = syst_tools.add_muonscale_hist(results, df, args.muonCorrEtaBins, args.muonCorrMag, isW, axes, cols)
                if args.muonScaleVariation == 'smearingWeightsGaus':
                    df = syst_tools.add_muonscale_smeared_hist(results, df, args.muonCorrEtaBins, args.muonCorrMag, isW, axes, cols_gen_smeared)

            ####################################################
            # nuisances from the muon momemtum scale calibration 
            if (args.muonCorrData in ["massfit", "lbl_massfit"]):
                if diff_weights_helper:
                    df = df.Define(f'{reco_sel_GF}_response_weight', diff_weights_helper,
                        [
                            f"{reco_sel_GF}_recoPt",
                            f"{reco_sel_GF}_recoEta",
                            f"{reco_sel_GF}_recoCharge",
                            f"{reco_sel_GF}_genPt",
                            f"{reco_sel_GF}_genEta",
                            f"{reco_sel_GF}_genCharge"
                        ]
                    )

                # muon scale variation from stats. uncertainty on the jpsi massfit
                df = muon_calibration.add_jpsi_crctn_stats_unc_hists(
                    args, df, axes, results, cols, cols_gen_smeared,
                    calib_filepaths, jpsi_crctn_data_unc_helper, smearing_weights_procs,
                    reco_sel_GF, dataset.name, isW
                )
                # add the ad-hoc Z non-closure nuisances from the jpsi massfit to muon scale unc
                df = muon_calibration.add_jpsi_crctn_Z_non_closure_hists(
                    args, df, axes, results, cols, cols_gen_smeared,
                    z_non_closure_parametrized_helper, z_non_closure_binned_helper, reco_sel_GF
                )
                # add nuisances from the data/MC resolution mismatch
                df = muon_calibration.add_resolution_uncertainty(df, axes, results, cols, smearing_uncertainty_helper, reco_sel_GF)
                if args.validationHists:
                    df = muon_validation.make_hists_for_muon_scale_var_weights(
                        df, axes, results, cols, cols_gen_smeared
                    )
            ####################################################

            df = df.Define("Muon_cvhMomCov", "wrem::splitNestedRVec(Muon_cvhMomCov_Vals, Muon_cvhMomCov_Counts)")

        if not args.binnedScaleFactors and not args.smooth3dsf and not args.sf2DnoUt:
            df = df.Define("weight2dsfup", muon_efficiency_helper2d, ["goodMuons_pt0", "goodMuons_eta0", "goodMuons_SApt0", "goodMuons_SAeta0", "goodMuons_pt0", "goodMuons_charge0", "passIso"])
            # be EXTREMELY CAREFUL about the histogram files (this assumes that you have another file with the old trigger and histo SFs which also contains the same SFs for all the other steps as the central one)
            df = df.Define("nominal_weight_2dsf", "nominal_weight/weight_fullMuonSF_withTrackingReco*weight2dsfup")
            sf2d = df.HistoBoost("nominal_sf2d", nominal_axes, [*nominal_cols, "nominal_weight_2dsf"])
            results.append(sf2d)
               
    if hasattr(dataset, "out_of_acceptance"):
        # Rename dataset to not overwrite the original one
        if len(smearing_weights_procs) > 0 and smearing_weights_procs[-1] == dataset.name:
            smearing_weights_procs[-1] = "Bkg"+dataset.name
        dataset.name = "Bkg"+dataset.name

    return results, weightsum

resultdict = narf.build_and_run(datasets, build_graph)
if not args.onlyMainHistograms and args.muonScaleVariation == 'smearingWeightsGaus' and not args.theoryAgnostic:
    logger.debug("Apply smearingWeights")
    muon_calibration.transport_smearing_weights_to_reco(
        resultdict,
        smearing_weights_procs,
        nonClosureScheme = args.nonClosureScheme
    )
if args.validationHists:
    muon_validation.muon_scale_variation_from_manual_shift(resultdict)

if not args.noScaleToData:
    scale_to_data(resultdict)
    aggregate_groups(datasets, resultdict, groups_to_aggregate)

output_tools.write_analysis_output(resultdict, f"{os.path.basename(__file__).replace('py', 'hdf5')}", args, update_name=not args.forceDefaultName)
