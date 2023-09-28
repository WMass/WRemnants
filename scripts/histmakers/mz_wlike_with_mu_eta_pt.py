from utilities import boostHistHelpers as hh, common, output_tools, logging, differential

parser,initargs = common.common_parser(True)

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
import os
import numpy as np

parser = common.set_parser_default(parser, "genVars", ["qGen", "ptGen", "absEtaGen"])
parser = common.set_parser_default(parser, "genBins", [17, 0])
parser = common.set_parser_default(parser, "pt", [34, 26, 60])
parser = common.set_parser_default(parser, "aggregateGroups", ["Diboson", "Top", "Wtaunu", "Wmunu"])

args = parser.parse_args()
logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

thisAnalysis = ROOT.wrem.AnalysisType.Wlike
datasets = getDatasets(maxFiles=args.maxFiles,
                        filt=args.filterProcs,
                        excl=args.excludeProcs, 
                        nanoVersion="v8" if args.v8 else "v9", base_path=args.dataPath)

era = args.era

# dilepton invariant mass cuts
mass_min = 60
mass_max = 120

# transverse boson mass cut
mtw_min=45 # 40 for Wmass, thus be 45 here (roughly half the boson mass)

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

# categorical axes in python bindings always have an overflow bin, so use a regular
# axis for the charge
axis_charge = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "charge")

nominal_axes = [axis_eta, axis_pt, axis_charge]
nominal_cols = ["trigMuons_eta0", "trigMuons_pt0", "trigMuons_charge0"]

if args.unfolding:
    unfolding_axes, unfolding_cols = differential.get_pt_eta_charge_axes(args.genBins[0], template_minpt, template_maxpt, args.genBins[1])
    datasets = unfolding_tools.add_out_of_acceptance(datasets, group = "Zmumu")

# axes for mT measurement
axis_mt = hist.axis.Regular(200, 0., 200., name = "mt",underflow=False, overflow=True)
axis_eta_mT = hist.axis.Variable([-2.4, 2.4], name = "eta")

# define helpers
muon_prefiring_helper, muon_prefiring_helper_stat, muon_prefiring_helper_syst = wremnants.make_muon_prefiring_helpers(era = era)

qcdScaleByHelicity_helper = wremnants.makeQCDScaleByHelicityHelper(is_w_like = True)

# extra axes which can be used to label tensor_axes
if args.binnedScaleFactors:
    logger.info("Using binned scale factors and uncertainties")
    # add usePseudoSmoothing=True for tests with Asimov
    muon_efficiency_helper, muon_efficiency_helper_syst, muon_efficiency_helper_stat = wremnants.make_muon_efficiency_helpers_binned(filename = args.sfFile, era = era, max_pt = axis_pt.edges[-1], is_w_like = True) 
else:
    logger.info("Using smoothed scale factors and uncertainties")
    muon_efficiency_helper, muon_efficiency_helper_syst, muon_efficiency_helper_stat = wremnants.make_muon_efficiency_helpers_smooth(filename = args.sfFile, era = era, what_analysis = thisAnalysis, max_pt = axis_pt.edges[-1], isoEfficiencySmoothing = args.isoEfficiencySmoothing, smooth3D=args.smooth3dsf)
logger.info(f"SF file: {args.sfFile}")

pileup_helper = wremnants.make_pileup_helper(era = era)

calib_filepaths = common.calib_filepaths
closure_filepaths = common.closure_filepaths
diff_weights_helper = ROOT.wrem.SplinesDifferentialWeightsHelper(calib_filepaths['tflite_file']) if (args.muonScaleVariation == 'smearingWeightsSplines' or args.validationHists) else None
mc_jpsi_crctn_helper, data_jpsi_crctn_helper, mc_jpsi_crctn_unc_helper, data_jpsi_crctn_unc_helper = muon_calibration.make_jpsi_crctn_helpers(args, calib_filepaths, make_uncertainty_helper=True)
z_non_closure_parametrized_helper, z_non_closure_binned_helper = muon_calibration.make_Z_non_closure_helpers(args, calib_filepaths, closure_filepaths)

mc_calibration_helper, data_calibration_helper, calibration_uncertainty_helper = muon_calibration.make_muon_calibration_helpers(args)

smearing_helper, smearing_uncertainty_helper = (None, None) if args.noSmearing else muon_calibration.make_muon_smearing_helpers()

bias_helper = muon_calibration.make_muon_bias_helpers(args) if args.biasCalibration else None

corr_helpers = theory_corrections.load_corr_helpers([d.name for d in datasets if d.name in common.vprocs], args.theoryCorr)

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
    apply_theory_corr = args.theoryCorr and dataset.name in corr_helpers

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    weightsum = df.SumAndCount("weight")

    axes = nominal_axes
    cols = nominal_cols

    if args.unfolding and dataset.name == "ZmumuPostVFP":
        df = unfolding_tools.define_gen_level(df, args.genLevel, dataset.name, mode="wlike")

        if hasattr(dataset, "out_of_acceptance"):
            logger.debug("Reject events in fiducial phase space")
            df = unfolding_tools.select_fiducial_space(df, mode="wlike", pt_min=args.pt[1], pt_max=args.pt[2], mass_min=mass_min, mass_max=mass_max, accept=False)
        else:
            logger.debug("Select events in fiducial phase space")
            df = unfolding_tools.select_fiducial_space(df, mode="wlike", pt_min=args.pt[1], pt_max=args.pt[2], mass_min=mass_min, mass_max=mass_max, accept=True)

            unfolding_tools.add_xnorm_histograms(results, df, args, dataset.name, corr_helpers, qcdScaleByHelicity_helper, unfolding_axes, unfolding_cols)
            axes = [*nominal_axes, *unfolding_axes] 
            cols = [*nominal_cols, *unfolding_cols]

    df = df.Filter("HLT_IsoTkMu24 || HLT_IsoMu24")

    df = muon_selections.veto_electrons(df)
    df = muon_selections.apply_met_filters(df)

    cvh_helper = data_calibration_helper if dataset.is_data else mc_calibration_helper
    jpsi_helper = data_jpsi_crctn_helper if dataset.is_data else mc_jpsi_crctn_helper

    df = muon_calibration.define_corrected_muons(df, cvh_helper, jpsi_helper, args, dataset, smearing_helper, bias_helper)

    df = muon_selections.select_veto_muons(df, nMuons=2)
    df = muon_selections.select_good_muons(df, template_minpt, template_maxpt, nMuons=2, use_trackerMuons=args.trackerMuons, use_isolation=True)

    df = muon_selections.define_trigger_muons(df, what_analysis=thisAnalysis)

    df = muon_selections.select_z_candidate(df, mass_min, mass_max)

    df = muon_selections.select_standalone_muons(df, dataset, args.trackerMuons, "trigMuons")
    df = muon_selections.select_standalone_muons(df, dataset, args.trackerMuons, "nonTrigMuons")

    df = muon_selections.apply_triggermatching_muon(df, dataset, "trigMuons_eta0", "trigMuons_phi0")

    if not dataset.is_data:
        df = df.Define("weight_pu", pileup_helper, ["Pileup_nTrueInt"])

        columnsForSF = ["trigMuons_pt0", "trigMuons_eta0", "trigMuons_SApt0", "trigMuons_SAeta0", "trigMuons_uT0", "trigMuons_charge0",
                        "nonTrigMuons_pt0", "nonTrigMuons_eta0", "nonTrigMuons_SApt0", "nonTrigMuons_SAeta0", "nonTrigMuons_uT0", "nonTrigMuons_charge0"]
        df = muon_selections.define_muon_uT_variable(df, isWorZ, smooth3dsf=args.smooth3dsf, colNamePrefix="trigMuons")
        df = muon_selections.define_muon_uT_variable(df, isWorZ, smooth3dsf=args.smooth3dsf, colNamePrefix="nonTrigMuons")
        if not args.smooth3dsf:
            columnsForSF.remove("trigMuons_uT0")
            columnsForSF.remove("nonTrigMuons_uT0")
        
        df = df.Define("weight_fullMuonSF_withTrackingReco", muon_efficiency_helper, columnsForSF)
        df = df.Define("weight_newMuonPrefiringSF", muon_prefiring_helper, ["Muon_correctedEta", "Muon_correctedPt", "Muon_correctedPhi", "Muon_correctedCharge", "Muon_looseId"])

        df = df.Define("exp_weight", "weight_pu*weight_fullMuonSF_withTrackingReco*weight_newMuonPrefiringSF*L1PreFiringWeight_ECAL_Nom")
        df = theory_tools.define_theory_weights_and_corrs(df, dataset.name, corr_helpers, args)
    else:
        df = df.DefinePerSample("nominal_weight", "1.0")

    results.append(df.HistoBoost("weight", [hist.axis.Regular(100, -2, 2)], ["nominal_weight"], storage=hist.storage.Double()))

    if not args.noRecoil:
        df = df.Define("yZ", "ll_mom4.Rapidity()")
        lep_cols = ["Muon_pt[goodMuons]", "Muon_phi[goodMuons]", "Muon_pt[goodMuons]"]
        trg_cols = ["trigMuons_pt0", "trigMuons_phi0", "nonTrigMuons_pt0", "nonTrigMuons_phi0"]
        df = recoilHelper.recoil_Z(df, results, dataset, common.zprocs_recoil, lep_cols, trg_cols) # produces corrected MET as MET_corr_rec_pt/phi
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
    
    # cutting after storing mt distributions for plotting, since the cut is only on corrected met
    df = df.Define("deltaPhiMuonMet", "std::abs(wrem::deltaPhi(trigMuons_phi0,met_wlike_TV2.Phi()))")
    df = df.Filter(f"deltaPhiMuonMet > {args.dphiMuonMetCut*np.pi}")
    df = df.Filter(f"transverseMass >= {mtw_min}")
    
    nominal = df.HistoBoost("nominal", axes, [*cols, "nominal_weight"])
    results.append(nominal)

    if not args.noRecoil and args.recoilUnc:
        df = recoilHelper.add_recoil_unc_Z(df, results, dataset, cols, axes, "nominal")

    if not dataset.is_data and not args.onlyMainHistograms:

        df = syst_tools.add_muon_efficiency_unc_hists(results, df, muon_efficiency_helper_stat, muon_efficiency_helper_syst, axes, cols, what_analysis=thisAnalysis, smooth3D=args.smooth3dsf)
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
            ####################################################

    if hasattr(dataset, "out_of_acceptance"):
        # Rename dataset to not overwrite the original one
        dataset.name = "Bkg"+dataset.name

    return results, weightsum

resultdict = narf.build_and_run(datasets, build_graph)

if not args.noScaleToData:
    scale_to_data(resultdict)
    aggregate_groups(datasets, resultdict, args.aggregateGroups)

output_tools.write_analysis_output(resultdict, f"{os.path.basename(__file__).replace('py', 'hdf5')}", args, update_name=not args.forceDefaultName)
