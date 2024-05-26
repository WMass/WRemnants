from utilities import boostHistHelpers as hh, common, logging, differential
from utilities.io_tools import output_tools
from wremnants.datasets.datagroups import Datagroups
import os

analysis_label = Datagroups.analysisLabel(os.path.basename(__file__))
parser,initargs = common.common_parser(analysis_label)

import ROOT
import narf
import wremnants
from wremnants import theory_tools,syst_tools,theory_corrections, muon_validation, muon_calibration, muon_selections, unfolding_tools
from wremnants.histmaker_tools import scale_to_data, aggregate_groups
from wremnants.datasets.dataset_tools import getDatasets
import hist
import lz4.frame
import math
import time
import numpy as np

parser.add_argument("--mtCut", type=int, default=common.get_default_mtcut(analysis_label), help="Value for the transverse mass cut in the event selection") # 40 for Wmass, thus be 45 here (roughly half the boson mass)
parser.add_argument("--muonIsolation", type=int, nargs=2, default=[1,1], choices=[-1, 0, 1], help="Apply isolation cut to triggering and not-triggering muon (in this order): -1/1 for failing/passing isolation, 0 for skipping it")

initargs,_ = parser.parse_known_args()
logger = logging.setup_logger(__file__, initargs.verbose, initargs.noColorLogger)

isUnfolding = initargs.analysisMode == "unfolding"

parser = common.set_parser_default(parser, "aggregateGroups", ["Diboson", "Top", "Wtaunu", "Wmunu"])
parser = common.set_parser_default(parser, "excludeProcs", ["QCD"])

args = parser.parse_args()

thisAnalysis = ROOT.wrem.AnalysisType.Wlike
isoBranch = muon_selections.getIsoBranch(args.isolationDefinition)
era = args.era

datasets = getDatasets(maxFiles=args.maxFiles,
                       filt=args.filterProcs,
                       excl=args.excludeProcs, 
                       nanoVersion="v9", base_path=args.dataPath,
                       extended = "msht20an3lo" not in args.pdfs,
                       era=era)

# dilepton invariant mass cuts
mass_min, mass_max = common.get_default_mz_window()

# transverse boson mass cut
mtw_min=args.mtCut 

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

nominal_axes = [axis_eta, axis_pt, common.axis_charge]
nominal_cols = ["trigMuons_eta0", "trigMuons_pt0", "trigMuons_charge0"]

if isUnfolding:
    template_wpt = (template_maxpt-template_minpt)/args.genBins[0]
    min_pt_unfolding = template_minpt+template_wpt
    max_pt_unfolding = template_maxpt-template_wpt
    npt_unfolding = args.genBins[0]-2
    unfolding_axes, unfolding_cols = differential.get_pt_eta_charge_axes(npt_unfolding, min_pt_unfolding, max_pt_unfolding, args.genBins[1])
    datasets = unfolding_tools.add_out_of_acceptance(datasets, group = "Zmumu")
    datasets = unfolding_tools.add_out_of_acceptance(datasets, group = "Ztautau")

# axes for mT measurement
axis_mt = hist.axis.Regular(200, 0., 200., name = "mt",underflow=False, overflow=True)
axis_eta_mT = hist.axis.Variable([-2.4, 2.4], name = "eta")

# define helpers
muon_prefiring_helper, muon_prefiring_helper_stat, muon_prefiring_helper_syst = wremnants.make_muon_prefiring_helpers(era = era)

qcdScaleByHelicity_helper = wremnants.theory_corrections.make_qcd_uncertainty_helper_by_helicity(is_w_like = True)

# extra axes which can be used to label tensor_axes
if args.binnedScaleFactors:
    logger.info("Using binned scale factors and uncertainties")
    # add usePseudoSmoothing=True for tests with Asimov
    muon_efficiency_helper, muon_efficiency_helper_syst, muon_efficiency_helper_stat = wremnants.make_muon_efficiency_helpers_binned(filename = args.sfFile, era = era, max_pt = axis_pt.edges[-1], is_w_like = True) 
else:
    logger.info("Using smoothed scale factors and uncertainties")
    muon_efficiency_helper, muon_efficiency_helper_syst, muon_efficiency_helper_stat = wremnants.make_muon_efficiency_helpers_smooth(filename = args.sfFile, era = era, what_analysis = thisAnalysis, max_pt = axis_pt.edges[-1], isoEfficiencySmoothing = args.isoEfficiencySmoothing, smooth3D=args.smooth3dsf, isoDefinition=args.isolationDefinition)
logger.info(f"SF file: {args.sfFile}")

muon_efficiency_helper_syst_altBkg = {}
for es in common.muonEfficiency_altBkgSyst_effSteps:
    altSFfile = args.sfFile.replace(".root", "_altBkg.root")
    logger.info(f"Additional SF file for alternate syst with {es}: {altSFfile}")
    muon_efficiency_helper_syst_altBkg[es] = wremnants.make_muon_efficiency_helpers_smooth_altSyst(filename = altSFfile, era = era,
                                                                                                   what_analysis = thisAnalysis, max_pt = axis_pt.edges[-1],
                                                                                                   effStep=es)

pileup_helper = wremnants.make_pileup_helper(era = era)
vertex_helper = wremnants.make_vertex_helper(era = era)

calib_filepaths = common.calib_filepaths
closure_filepaths = common.closure_filepaths
diff_weights_helper = ROOT.wrem.SplinesDifferentialWeightsHelper(calib_filepaths['tflite_file']) if (args.muonScaleVariation == 'smearingWeightsSplines' or args.validationHists) else None
mc_jpsi_crctn_helper, data_jpsi_crctn_helper, mc_jpsi_crctn_unc_helper, data_jpsi_crctn_unc_helper = muon_calibration.make_jpsi_crctn_helpers(args, calib_filepaths, make_uncertainty_helper=True)
z_non_closure_parametrized_helper, z_non_closure_binned_helper = muon_calibration.make_Z_non_closure_helpers(args, calib_filepaths, closure_filepaths)

mc_calibration_helper, data_calibration_helper, calibration_uncertainty_helper = muon_calibration.make_muon_calibration_helpers(args,era=era)

closure_unc_helper = wremnants.muon_calibration.make_closure_uncertainty_helper(common.closure_filepaths["parametrized"])
closure_unc_helper_A = wremnants.muon_calibration.make_uniform_closure_uncertainty_helper(0, common.correlated_variation_base_size["A"])
closure_unc_helper_M = wremnants.muon_calibration.make_uniform_closure_uncertainty_helper(2, common.correlated_variation_base_size["M"])

smearing_helper, smearing_uncertainty_helper = (None, None) if args.noSmearing else muon_calibration.make_muon_smearing_helpers()

bias_helper = muon_calibration.make_muon_bias_helpers(args) if args.biasCalibration else None

pixel_multiplicity_helper, pixel_multiplicity_uncertainty_helper, pixel_multiplicity_uncertainty_helper_stat = muon_calibration.make_pixel_multiplicity_helpers(reverse_variations = args.reweightPixelMultiplicity)

theory_corrs = [*args.theoryCorr, *args.ewTheoryCorr]
corr_helpers = theory_corrections.load_corr_helpers([d.name for d in datasets if d.name in common.vprocs], theory_corrs)

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
    apply_theory_corr = theory_corrs and dataset.name in corr_helpers

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    df = df.Define("isEvenEvent", "event % 2 == 0")

    weightsum = df.SumAndCount("weight")

    axes = nominal_axes
    cols = nominal_cols

    if isUnfolding and isZ:
        df = unfolding_tools.define_gen_level(df, args.genLevel, dataset.name, mode=analysis_label)
        cutsmap = {"pt_min" : template_minpt, "pt_max" : template_maxpt, "mtw_min" : args.mtCut, "abseta_max" : template_maxeta, 
                   "mass_min" : mass_min, "mass_max" : mass_max}

        if hasattr(dataset, "out_of_acceptance"):
            logger.debug("Reject events in fiducial phase space")
            df = unfolding_tools.select_fiducial_space(df, mode=analysis_label, accept=False, **cutsmap)
        else:
            logger.debug("Select events in fiducial phase space")
            df = unfolding_tools.select_fiducial_space(df, mode=analysis_label, accept=True, **cutsmap)

            unfolding_tools.add_xnorm_histograms(results, df, args, dataset.name, corr_helpers, qcdScaleByHelicity_helper, unfolding_axes, unfolding_cols)
            axes = [*nominal_axes, *unfolding_axes] 
            cols = [*nominal_cols, *unfolding_cols]

    df = df.Filter(muon_selections.hlt_string(era))

    df = muon_selections.veto_electrons(df)
    df = muon_selections.apply_met_filters(df)

    cvh_helper = data_calibration_helper if dataset.is_data else mc_calibration_helper
    jpsi_helper = data_jpsi_crctn_helper if dataset.is_data else mc_jpsi_crctn_helper

    df = muon_calibration.define_corrected_muons(df, cvh_helper, jpsi_helper, args, dataset, smearing_helper, bias_helper)

    df = muon_selections.select_veto_muons(df, nMuons=2)

    isoThreshold = args.isolationThreshold
    passIsoBoth = (args.muonIsolation[0] + args.muonIsolation[1] == 2)
    df = muon_selections.select_good_muons(df, template_minpt, template_maxpt, dataset.group, nMuons=2, use_trackerMuons=args.trackerMuons, use_isolation=passIsoBoth, isoBranch=isoBranch, isoThreshold=isoThreshold, requirePixelHits=args.requirePixelHits)
    
    df = muon_selections.define_trigger_muons(df)

    # iso cut applied here, if requested, because it needs the definition of trigMuons and nonTrigMuons from muon_selections.define_trigger_muons
    if not passIsoBoth:
        df = muon_selections.apply_iso_muons(df, args.muonIsolation[0], args.muonIsolation[1], isoBranch, isoThreshold)

    df = df.Define("trigMuons_passIso0", f"{isoBranch}[trigMuons][0] < {isoThreshold}")
    df = df.Define("nonTrigMuons_passIso0", f"{isoBranch}[nonTrigMuons][0] < {isoThreshold}")

    df = muon_selections.select_z_candidate(df, mass_min, mass_max)

    df = muon_selections.select_standalone_muons(df, dataset, args.trackerMuons, "trigMuons")
    df = muon_selections.select_standalone_muons(df, dataset, args.trackerMuons, "nonTrigMuons")

    df = muon_selections.apply_triggermatching_muon(df, dataset, "trigMuons", era=era)

    if dataset.is_data:
        df = df.DefinePerSample("nominal_weight", "1.0")
    else:
        df = df.Define("weight_pu", pileup_helper, ["Pileup_nTrueInt"])
        df = df.Define("weight_vtx", vertex_helper, ["GenVtx_z", "Pileup_nTrueInt"])
        df = df.Define("weight_newMuonPrefiringSF", muon_prefiring_helper, ["Muon_correctedEta", "Muon_correctedPt", "Muon_correctedPhi", "Muon_correctedCharge", "Muon_looseId"])

        if era == "2016PostVFP":
            weight_expr = "weight_pu*weight_newMuonPrefiringSF*L1PreFiringWeight_ECAL_Nom"
        else:
            weight_expr = "weight_pu*L1PreFiringWeight_Muon_Nom*L1PreFiringWeight_ECAL_Nom"

        if not args.noVertexWeight:
            weight_expr += "*weight_vtx"            

        muonVarsForSF = ["pt0", "eta0", "SApt0", "SAeta0", "uT0", "charge0", "passIso0"]
        columnsForSF = [f"{t}Muons_{v}" for t in ["trig", "nonTrig"] for v in muonVarsForSF]

        df = muon_selections.define_muon_uT_variable(df, isWorZ, smooth3dsf=args.smooth3dsf, colNamePrefix="trigMuons")
        df = muon_selections.define_muon_uT_variable(df, isWorZ, smooth3dsf=args.smooth3dsf, colNamePrefix="nonTrigMuons")
        if not args.smooth3dsf:
            columnsForSF.remove("trigMuons_uT0")
            columnsForSF.remove("nonTrigMuons_uT0")
        
        if not args.noScaleFactors:
            df = df.Define("weight_fullMuonSF_withTrackingReco", muon_efficiency_helper, columnsForSF)
            weight_expr += "*weight_fullMuonSF_withTrackingReco"

        # prepare inputs for pixel multiplicity helpers
        cvhName = "cvhideal"

        df = df.Define(f"trigMuons_{cvhName}NValidPixelHits0", f"Muon_{cvhName}NValidPixelHits[trigMuons][0]")
        df = df.Define(f"nonTrigMuons_{cvhName}NValidPixelHits0", f"Muon_{cvhName}NValidPixelHits[nonTrigMuons][0]")

        df = df.DefinePerSample("MuonNonTrigTrig_triggerCat", "ROOT::VecOps::RVec<wrem::TriggerCat>{wrem::TriggerCat::nonTriggering, wrem::TriggerCat::triggering}");
        df = df.Define("MuonNonTrigTrig_eta", "ROOT::VecOps::RVec<float>{nonTrigMuons_eta0, trigMuons_eta0}")
        df = df.Define("MuonNonTrigTrig_pt", "ROOT::VecOps::RVec<float>{nonTrigMuons_pt0, trigMuons_pt0}")
        df = df.Define("MuonNonTrigTrig_charge", "ROOT::VecOps::RVec<int>{nonTrigMuons_charge0, trigMuons_charge0}")
        df = df.Define(f"MuonNonTrigTrig_{cvhName}NValidPixelHits", f"ROOT::VecOps::RVec<int>{{nonTrigMuons_{cvhName}NValidPixelHits0, trigMuons_{cvhName}NValidPixelHits0}}")

        pixel_multiplicity_cols = ["MuonNonTrigTrig_triggerCat", "MuonNonTrigTrig_eta", "MuonNonTrigTrig_pt", "MuonNonTrigTrig_charge", f"MuonNonTrigTrig_{cvhName}NValidPixelHits"]

        if args.reweightPixelMultiplicity:
            df = df.Define("weight_pixel_multiplicity", pixel_multiplicity_helper, pixel_multiplicity_cols)
            weight_expr += "*weight_pixel_multiplicity"

        logger.debug(f"Exp weight defined: {weight_expr}")
        df = df.Define("exp_weight", weight_expr)
        df = theory_tools.define_theory_weights_and_corrs(df, dataset.name, corr_helpers, args)

    results.append(df.HistoBoost("weight", [hist.axis.Regular(100, -2, 2)], ["nominal_weight"], storage=hist.storage.Double()))

    if not args.noRecoil:
        leps_uncorr = ["Muon_pt[goodMuons][0]", "Muon_eta[goodMuons][0]", "Muon_phi[goodMuons][0]", "Muon_charge[goodMuons][0]", "Muon_pt[goodMuons][1]", "Muon_eta[goodMuons][1]", "Muon_phi[goodMuons][1]", "Muon_charge[goodMuons][1]"]
        leps_corr = ["trigMuons_pt0", "trigMuons_eta0", "trigMuons_phi0", "trigMuons_charge0", "nonTrigMuons_pt0", "nonTrigMuons_eta0", "nonTrigMuons_phi0", "nonTrigMuons_charge0"]
        df = recoilHelper.recoil_Z(df, results, dataset, common.zprocs_recoil, leps_uncorr, leps_corr)  # produces corrected MET as MET_corr_rec_pt/phi
    else:
        df = df.Alias("MET_corr_rec_pt", "MET_pt")
        df = df.Alias("MET_corr_rec_phi", "MET_phi")

    #TODO improve this to include muon mass?
    ###########
    # utility plots of transverse mass, with or without recoil corrections
    ###########
    met_vars = ("MET_pt", "MET_phi")
    df = df.Define("transverseMass_uncorr", f"wrem::get_mt_wlike(trigMuons_pt0, trigMuons_phi0, nonTrigMuons_pt0, nonTrigMuons_phi0, {', '.join(met_vars)})")
    results.append(df.HistoBoost("transverseMass_uncorr", [axis_mt], ["transverseMass_uncorr", "nominal_weight"]))
    ###########
    met_vars = ("MET_corr_rec_pt", "MET_corr_rec_phi")
    df = df.Define("met_wlike_TV2", f"wrem::get_met_wlike(nonTrigMuons_pt0, nonTrigMuons_phi0, {', '.join(met_vars)})")
    df = df.Define("transverseMass", "wrem::get_mt_wlike(trigMuons_pt0, trigMuons_phi0, met_wlike_TV2)")
    results.append(df.HistoBoost("transverseMass", [axis_mt], ["transverseMass", "nominal_weight"]))
    results.append(df.HistoBoost("MET", [hist.axis.Regular(20, 0, 100, name="MET")], ["MET_corr_rec_pt", "nominal_weight"]))
    df = df.Define("met_wlike_TV2_pt", "met_wlike_TV2.Mod()")
    results.append(df.HistoBoost("WlikeMET", [hist.axis.Regular(20, 0, 100, name="Wlike-MET")], ["met_wlike_TV2_pt", "nominal_weight"]))
    ###########

    df = df.Define("passWlikeMT", f"transverseMass >= {mtw_min}")

    if not args.onlyMainHistograms:
        axis_mt_coarse = hist.axis.Variable([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 120.0], name = "mt", underflow=False, overflow=True)
        axis_trigPassIso = hist.axis.Boolean(name = f"trig_passIso")
        axis_nonTrigPassIso = hist.axis.Boolean(name = f"nonTrig_passIso")

        nominal_bin = df.HistoBoost("nominal_isoMtBins", [*axes, axis_trigPassIso, axis_nonTrigPassIso, axis_mt_coarse], [*cols, "trigMuons_passIso0", "nonTrigMuons_passIso0", "transverseMass", "nominal_weight"])
        results.append(nominal_bin)

        axis_eta_nonTrig = hist.axis.Regular(template_neta, template_mineta, template_maxeta, name = "etaNonTrig", overflow=False, underflow=False)
        axis_pt_nonTrig = hist.axis.Regular(template_npt, template_minpt, template_maxpt, name = "ptNonTrig", overflow=False, underflow=False)
        # nonTriggering muon charge can be assumed to be opposite of triggering one
        nominal_bothMuons = df.HistoBoost("nominal_bothMuons", [*axes, axis_eta_nonTrig, axis_pt_nonTrig, common.axis_passMT], [*cols, "nonTrigMuons_eta0", "nonTrigMuons_pt0", "passWlikeMT", "nominal_weight"])
        results.append(nominal_bothMuons)

    # cutting after storing mt distributions for plotting, since the cut is only on corrected met
    if args.dphiMuonMetCut > 0.0:
        df = df.Define("deltaPhiMuonMet", "std::abs(wrem::deltaPhi(trigMuons_phi0,met_wlike_TV2.Phi()))")
        df = df.Filter(f"deltaPhiMuonMet > {args.dphiMuonMetCut*np.pi}")

    df = df.Filter("passWlikeMT")

    if not args.onlyMainHistograms:
        # plot reco vertex distribution before and after PU reweigthing
        # also remove vertex weights since they depend on PU
        if dataset.is_data:
            df = df.DefinePerSample("nominal_weight_noPUandVtx", "1.0")
        else:
            df = df.Define("nominal_weight_noPUandVtx", "nominal_weight/(weight_pu*weight_vtx)")
        axis_nRecoVtx = hist.axis.Regular(50, 0.5, 50.5, name="PV_npvsGood")
        axis_fixedGridRhoFastjetAll = hist.axis.Regular(50, 0, 50, name="fixedGridRhoFastjetAll")
        results.append(df.HistoBoost("PV_npvsGood_uncorr", [axis_nRecoVtx], ["PV_npvsGood", "nominal_weight_noPUandVtx"]))
        results.append(df.HistoBoost("PV_npvsGood", [axis_nRecoVtx], ["PV_npvsGood", "nominal_weight"]))
        results.append(df.HistoBoost("fixedGridRhoFastjetAll_uncorr", [axis_nRecoVtx], ["fixedGridRhoFastjetAll", "nominal_weight_noPUandVtx"]))
        results.append(df.HistoBoost("fixedGridRhoFastjetAll", [axis_nRecoVtx], ["fixedGridRhoFastjetAll", "nominal_weight"]))

    nominal = df.HistoBoost("nominal", axes, [*cols, "nominal_weight"])
    results.append(nominal)

    if not args.noRecoil and args.recoilUnc:
        df = recoilHelper.add_recoil_unc_Z(df, results, dataset, cols, axes, "nominal")

    if not dataset.is_data and not args.onlyMainHistograms:

        df = syst_tools.add_muon_efficiency_unc_hists(results, df, muon_efficiency_helper_stat, muon_efficiency_helper_syst, axes, cols, what_analysis=thisAnalysis, smooth3D=args.smooth3dsf)
        for es in common.muonEfficiency_altBkgSyst_effSteps:
            df = syst_tools.add_muon_efficiency_unc_hists_altBkg(results, df, muon_efficiency_helper_syst_altBkg[es], axes, cols, 
                                                                 what_analysis=thisAnalysis, step=es)

        df = syst_tools.add_L1Prefire_unc_hists(results, df, muon_prefiring_helper_stat, muon_prefiring_helper_syst, axes, cols)

        # n.b. this is the W analysis so mass weights shouldn't be propagated
        # on the Z samples (but can still use it for dummy muon scale)
        if isWorZ:

            df = syst_tools.add_theory_hists(results, df, args, dataset.name, corr_helpers, qcdScaleByHelicity_helper, axes, cols, for_wmass=False)

            reco_sel = "vetoMuonsPre"
            require_prompt = "tau" not in dataset.name
            df = muon_calibration.define_genFiltered_recoMuonSel(df, reco_sel, require_prompt)
            reco_sel_GF = muon_calibration.getColName_genFiltered_recoMuonSel(reco_sel, require_prompt)
            df = muon_calibration.define_matched_gen_muons_kinematics(df, reco_sel_GF)
            df = muon_calibration.calculate_matched_gen_muon_kinematics(df, reco_sel_GF)
            df = muon_calibration.define_matched_reco_muon_kinematics(df, reco_sel_GF)

            ####################################################
            # nuisances from the muon momemtum scale calibration
            if (args.muonCorrData in ["massfit", "lbl_massfit"]):
                input_kinematics = [
                    f"{reco_sel_GF}_recoPt",
                    f"{reco_sel_GF}_recoEta",
                    f"{reco_sel_GF}_recoCharge",
                    f"{reco_sel_GF}_genPt",
                    f"{reco_sel_GF}_genEta",
                    f"{reco_sel_GF}_genCharge"
                ]
                if diff_weights_helper:
                    df = df.Define(f'{reco_sel_GF}_response_weight', diff_weights_helper, [*input_kinematics])
                    input_kinematics.append(f'{reco_sel_GF}_response_weight')

                # muon scale variation from stats. uncertainty on the jpsi massfit
                df = df.Define(
                    "nominal_muonScaleSyst_responseWeights_tensor", data_jpsi_crctn_unc_helper,
                    [*input_kinematics, "nominal_weight"]
                )
                muonScaleSyst_responseWeights = df.HistoBoost(
                    "nominal_muonScaleSyst_responseWeights", axes,
                    [*cols, "nominal_muonScaleSyst_responseWeights_tensor"],
                    tensor_axes = data_jpsi_crctn_unc_helper.tensor_axes, storage=hist.storage.Double()
                )
                results.append(muonScaleSyst_responseWeights)

                df = muon_calibration.add_resolution_uncertainty(df, axes, results, cols, smearing_uncertainty_helper, reco_sel_GF)

                # add pixel multiplicity uncertainties
                df = df.Define("nominal_pixelMultiplicitySyst_tensor", pixel_multiplicity_uncertainty_helper, [*pixel_multiplicity_cols, "nominal_weight"])
                hist_pixelMultiplicitySyst = df.HistoBoost("nominal_pixelMultiplicitySyst", axes, [*cols, "nominal_pixelMultiplicitySyst_tensor"], tensor_axes = pixel_multiplicity_uncertainty_helper.tensor_axes, storage=hist.storage.Double())
                results.append(hist_pixelMultiplicitySyst)

                if args.pixelMultiplicityStat:
                    df = df.Define("nominal_pixelMultiplicityStat_tensor", pixel_multiplicity_uncertainty_helper_stat, [*pixel_multiplicity_cols, "nominal_weight"])
                    hist_pixelMultiplicityStat = df.HistoBoost("nominal_pixelMultiplicityStat", axes, [*cols, "nominal_pixelMultiplicityStat_tensor"], tensor_axes = pixel_multiplicity_uncertainty_helper_stat.tensor_axes, storage=hist.storage.Double())
                    results.append(hist_pixelMultiplicityStat)

                if args.nonClosureScheme in ["A-M-separated", "A-only"]:
                    # add the ad-hoc Z non-closure nuisances from the jpsi massfit to muon scale unc
                    df = df.DefinePerSample("AFlag", "0x01")
                    df = df.Define(
                        "Z_non_closure_parametrized_A", z_non_closure_parametrized_helper,
                        [*input_kinematics, "nominal_weight", "AFlag"]
                    )
                    hist_Z_non_closure_parametrized_A = df.HistoBoost(
                        "nominal_Z_non_closure_parametrized_A",
                        axes, [*cols, "Z_non_closure_parametrized_A"],
                        tensor_axes = z_non_closure_parametrized_helper.tensor_axes,
                        storage=hist.storage.Double()
                    )
                    results.append(hist_Z_non_closure_parametrized_A)

                if args.nonClosureScheme in ["A-M-separated", "binned-plus-M", "M-only"]:
                    df = df.DefinePerSample("MFlag", "0x04")
                    df = df.Define(
                        "Z_non_closure_parametrized_M", z_non_closure_parametrized_helper,
                        [*input_kinematics, "nominal_weight", "MFlag"]
                    )
                    hist_Z_non_closure_parametrized_M = df.HistoBoost(
                        "nominal_Z_non_closure_parametrized_M",
                        axes, [*cols, "Z_non_closure_parametrized_M"],
                        tensor_axes = z_non_closure_parametrized_helper.tensor_axes,
                        storage=hist.storage.Double()
                    )
                    results.append(hist_Z_non_closure_parametrized_M)

                if args.nonClosureScheme == "A-M-combined":
                    df = df.DefinePerSample("AMFlag", "0x01 | 0x04")
                    df = df.Define("Z_non_closure_parametrized", z_non_closure_parametrized_helper,
                        [
                            *input_kinematics,
                            "nominal_weight",
                            "AMFlag"
                        ]
                    )
                    hist_Z_non_closure_parametrized = df.HistoBoost(
                        "Z_non_closure_parametrized_gaus" if args.muonScaleVariation == 'smearingWeightsGaus' else "nominal_Z_non_closure_parametrized",
                        axes,
                        [*cols, "Z_non_closure_parametrized"],
                        tensor_axes = z_non_closure_parametrized_helper.tensor_axes,
                        storage=hist.storage.Double()
                    )
                    results.append(hist_Z_non_closure_parametrized)


                # extra uncertainties from non-closure stats
                df = df.Define("muonScaleClosSyst_responseWeights_tensor_splines", closure_unc_helper,
                    [*input_kinematics, "nominal_weight"]
                )
                nominal_muonScaleClosSyst_responseWeights = df.HistoBoost(
                    "nominal_muonScaleClosSyst_responseWeights", axes,
                    [*cols, "muonScaleClosSyst_responseWeights_tensor_splines"],
                    tensor_axes = closure_unc_helper.tensor_axes,
                    storage = hist.storage.Double()
                )
                results.append(nominal_muonScaleClosSyst_responseWeights)

                # extra uncertainties for A (fully correlated)
                df = df.Define("muonScaleClosASyst_responseWeights_tensor_splines", closure_unc_helper_A,
                    [*input_kinematics, "nominal_weight"]
                )
                nominal_muonScaleClosASyst_responseWeights = df.HistoBoost(
                    "nominal_muonScaleClosASyst_responseWeights", axes,
                    [*cols, "muonScaleClosASyst_responseWeights_tensor_splines"],
                    tensor_axes = closure_unc_helper_A.tensor_axes,
                    storage = hist.storage.Double()
                )
                results.append(nominal_muonScaleClosASyst_responseWeights)

                # extra uncertainties for M (fully correlated)
                df = df.Define("muonScaleClosMSyst_responseWeights_tensor_splines", closure_unc_helper_M,
                    [*input_kinematics, "nominal_weight"]
                )
                nominal_muonScaleClosMSyst_responseWeights = df.HistoBoost(
                    "nominal_muonScaleClosMSyst_responseWeights", axes,
                    [*cols, "muonScaleClosMSyst_responseWeights_tensor_splines"],
                    tensor_axes = closure_unc_helper_M.tensor_axes,
                    storage = hist.storage.Double()
                )
                results.append(nominal_muonScaleClosMSyst_responseWeights)

            ####################################################

    if hasattr(dataset, "out_of_acceptance"):
        # Rename dataset to not overwrite the original one
        dataset.name = dataset.name+"OOA"

    return results, weightsum

resultdict = narf.build_and_run(datasets, build_graph)

if not args.noScaleToData:
    scale_to_data(resultdict)
    aggregate_groups(datasets, resultdict, args.aggregateGroups)

output_tools.write_analysis_output(resultdict, f"{os.path.basename(__file__).replace('py', 'hdf5')}", args)
