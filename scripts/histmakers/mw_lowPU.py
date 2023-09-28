import argparse
from utilities import output_tools, differential
from utilities import common, logging

parser,initargs = common.common_parser()
parser.add_argument("--lumiUncertainty", type=float, help="Uncertainty for luminosity in excess to 1 (e.g. 1.017 means 1.7\%)", default=1.017)
parser.add_argument("--flavor", type=str, choices=["e", "mu"], help="Flavor (e or mu)", default="mu")

parser = common.set_parser_default(parser, "genVars", ["ptVGen"])
args = parser.parse_args()


import narf
import wremnants
from wremnants import theory_tools, syst_tools, theory_corrections, muon_selections, unfolding_tools
from wremnants.histmaker_tools import scale_to_data, aggregate_groups
from wremnants.datasets.dataset_tools import getDatasets
import math
import hist
import ROOT
import scripts.lowPU.config as lowPUcfg


###################################
flavor = args.flavor # mu, e
if flavor == "mu":
    sigProcs = ["WminusJetsToMuNu", "WplusJetsToMuNu"]
    base_group = "Wmunu"
else:
    sigProcs = ["WminusJetsToENu", "WplusJetsToENu"]
    base_group = "Wenu"

datasets = getDatasets(maxFiles=args.maxFiles,
                        filt=args.filterProcs,
                        excl=list(set(args.excludeProcs + ["singlemuon"] if flavor=="e" else ["singleelectron"])),
                        mode="lowPU")

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

for d in datasets: logger.info(f"Dataset {d.name}")

mtw_min=40 # for Wmass (roughly half the boson mass)

# load lowPU specific libs
#ROOT.gInterpreter.AddIncludePath(f"{pathlib.Path(__file__).parent}/include/")
ROOT.gInterpreter.Declare('#include "lowpu_utils.h"')
ROOT.gInterpreter.Declare('#include "lowpu_efficiencies.h"')
ROOT.gInterpreter.Declare('#include "lowpu_prefire.h"')
ROOT.gInterpreter.Declare('#include "lowpu_rochester.h"')
ROOT.gInterpreter.Declare('#include "lowpu_recoil.h"')
ROOT.gInterpreter.Declare('#include "electron_selections.h"')

# axes used in fakerate calculation
axis_fakerate_pt = hist.axis.Variable([26., 27., 28., 29., 30., 32., 34., 37., 40., 44., 49., 56.], name = "pt", underflow=False)
axis_fakerate_eta = hist.axis.Regular(12, -2.4, 2.4, name = "eta", underflow=False, overflow=False)

# standard regular axes
axis_eta = hist.axis.Regular(args.eta[0], args.eta[1], args.eta[2], name = "eta", underflow=False, overflow=False)
axis_pt = hist.axis.Regular(args.pt[0], args.pt[1], args.pt[2], name = "pt", underflow=False)
axis_phi = hist.axis.Regular(50, -4, 4, name = "phi")
axis_charge = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "charge")
axis_iso = hist.axis.Regular(50, 0, 1, underflow=False, overflow=True, name = "iso")

axis_passMT = hist.axis.Boolean(name = "passMT")
axis_passIso = hist.axis.Boolean(name = "passIso")
axis_lin = hist.axis.Regular(5, 0, 5, name = "lin")



qcdScaleByHelicity_helper = wremnants.makeQCDScaleByHelicityHelper()
axis_ptVgen = qcdScaleByHelicity_helper.hist.axes["ptVgen"]
axis_chargeVgen = qcdScaleByHelicity_helper.hist.axes["chargeVgen"]

gen_axes = {
    "ptVGen": hist.axis.Variable([0, 8, 14, 20, 30, 40, 50, 60, 75, 90, 150], name = "ptVGen", underflow=False, overflow=False),
}

groups_to_aggregate = args.aggregateGroups

if args.unfolding:
    unfolding_axes, unfolding_cols, unfolding_selections = differential.get_dilepton_axes(args.genVars, gen_axes)
    datasets = unfolding_tools.add_out_of_acceptance(datasets, group = base_group)
    groups_to_aggregate.append(f"Bkg{base_group}")

# axes for final cards/fitting
nominal_axes = [
    axis_fakerate_pt, axis_fakerate_eta, axis_charge, 
    hist.axis.Variable([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 75, 90, 150], name = "ptll", underflow=False, overflow=True),    
    axis_passIso, axis_passMT]

# corresponding columns
nominal_cols = ["lep_pt", "lep_eta", "lep_charge", "ptll", "passIso",  "passMT"]

# mt final cards/fitting
axis_mT = hist.axis.Variable([0] + list(range(40, 100, 1)) + [100, 102, 104, 106, 108, 112, 116, 120, 130, 150, 200], name = "mt",underflow=False, overflow=True)
# axis_mT = hist.axis.Regular(100, 0, 200, name = "mt", underflow=False)

axes_mT = [axis_fakerate_pt, axis_fakerate_eta, axis_charge, axis_mT, axis_passIso]
cols_mT = ["lep_pt", "lep_eta", "lep_charge", "transverseMass",  "passIso"]

# reco_mT_axes = [common.axis_recoil_reco_ptW_lowpu, common.axis_mt_lowpu, axis_charge, axis_passMT, axis_passIso]
# gen_reco_mT_axes = [common.axis_recoil_gen_ptW_lowpu, common.axis_recoil_reco_ptW_lowpu, common.axis_mt_lowpu, axis_charge, axis_passMT, axis_passIso]

# # corresponding columns
# gen_reco_mT_cols = ["ptVgen", "recoil_corr_rec_magn", "mt", "lep_charge", "passMT", "passIso"]
# reco_mT_cols = ["recoil_corr_rec_magn", "mt", "lep_charge", "passMT", "passIso"]
    

# extra axes which can be used to label tensor_axes
down_up_axis = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "downUpVar")
axis_cutFlow = hist.axis.Regular(1, 0, 1, name = "cutFlow")

corr_helpers = theory_corrections.load_corr_helpers([d.name for d in datasets if d.name in common.vprocs_lowpu], args.theoryCorr)

# recoil initialization
if not args.noRecoil:
    from wremnants import recoil_tools
    recoilHelper = recoil_tools.Recoil("lowPU", args, flavor)

def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")

    results = []

    if dataset.is_data: df = df.DefinePerSample("weight", "1.0")
    else: df = df.Define("weight", "std::copysign(1.0, genWeight)")
    weightsum = df.SumAndCount("weight")

    axes = nominal_axes
    cols = nominal_cols

    if args.unfolding and dataset.name in sigProcs:
        df = unfolding_tools.define_gen_level(df, args.genLevel, dataset.name, mode="wmass")

        if hasattr(dataset, "out_of_acceptance"):
            logger.debug("Reject events in fiducial phase space")
            df = unfolding_tools.select_fiducial_space(df, mode="wmass", pt_min=args.pt[1], pt_max=args.pt[2], 
                mtw_min=mtw_min, selections=unfolding_selections, accept=False)
        else:
            logger.debug("Select events in fiducial phase space")
            df = unfolding_tools.select_fiducial_space(df, mode="wmass", pt_min=args.pt[1], pt_max=args.pt[2], 
                mtw_min=mtw_min, selections=unfolding_selections, accept=True)

            unfolding_tools.add_xnorm_histograms(results, df, args, dataset.name, corr_helpers, qcdScaleByHelicity_helper, unfolding_axes, unfolding_cols)
            axes = [*axes, *unfolding_axes] 
            cols = [*cols, *unfolding_cols]
    
    
    if flavor == "mu":
    
        if not dataset.is_data: 
            df = df.Define("Muon_pt_corr", "wrem::applyRochesterMC(Muon_pt, Muon_eta, Muon_phi, Muon_charge, Muon_genPartIdx, GenPart_pt, Muon_nTrackerLayers)")
            #df = df.Alias("Muon_pt_corr", "Muon_pt")
            df = df.Filter("HLT_Mu17")
            
        else: 
        
            df = df.Define("Muon_pt_corr", "wrem::applyRochesterData(Muon_pt, Muon_eta, Muon_phi, Muon_charge)")
            #df = df.Alias("Muon_pt_corr", "Muon_pt")
            df = df.Filter("HLT_HIMu17")
        
        df = df.Define("vetoMuons", "Muon_pt_corr > 10 && Muon_looseId && abs(Muon_eta) < 2.4 && abs(Muon_dxybs) < 0.05")
        df = df.Filter("Sum(vetoMuons) == 1")
        
        df = df.Define("vetoElectrons", "Electron_pt > 10 && Electron_cutBased > 0 && abs(Electron_eta) < 2.4")
        df = df.Filter("Sum(vetoElectrons) == 0")
        
        df = df.Define("goodLeptons", f"vetoMuons && Muon_pt_corr > {args.pt[1]} && Muon_mediumId") # ISO requirement comes later   && Muon_pfRelIso04_all < 0.15
        df = df.Define("goodLeptonsPlus", "goodLeptons && Muon_charge > 0")
        df = df.Define("goodLeptonsMinus", "goodLeptons && Muon_charge < 0")
        df = df.Filter("Sum(goodLeptons) == 1")

        df = df.Define("goodTrigObjs", "wrem::goodMuonTriggerCandidateLowPU(TrigObj_id, TrigObj_pt, TrigObj_l1pt, TrigObj_l2pt, TrigObj_filterBits)")

        df = df.Define("Lep_pt_uncorr", "Muon_pt[goodLeptons]")
        df = df.Define("Lep_pt", "Muon_pt_corr[goodLeptons]")
        df = df.Define("Lep_eta", "Muon_eta[goodLeptons]")
        df = df.Define("Lep_abs_eta", "abs(Lep_eta)")
        df = df.Define("Lep_phi", "Muon_phi[goodLeptons]")
        df = df.Define("Lep_charge", "Muon_charge[goodLeptons]")
        df = df.Define("Lep_mass", "Muon_mass[goodLeptons]")
        df = df.Define("Lep_iso", "Muon_pfRelIso04_all[goodLeptons]")

        df = df.Define("passIso", "Lep_iso < 0.15")

    else:
        # undo the scale/smearing corrections, needed to correct RawMET
        df = df.Define("Electron_pt_uncorr", "wrem::Egamma_undoCorrection(Electron_pt, Electron_eta, Electron_ecalCorr)")
        if not dataset.is_data: 
            df = df.Define("Electron_pt_corr", "wrem::applyEGammaScaleSmearingUnc(0, Electron_pt, Electron_eta, Electron_dEscaleUp, Electron_dEscaleDown, Electron_dEsigmaUp, Electron_dEsigmaDown, 0)")
            df = df.Filter("HLT_Ele20_WPLoose_Gsf")
            
        else: 
            df = df.Define("Electron_pt_corr", "wrem::applyEGammaScaleSmearingUnc(1, Electron_pt, Electron_eta, Electron_dEscaleUp, Electron_dEscaleDown, Electron_dEsigmaUp, Electron_dEsigmaDown, 0)")
            df = df.Filter("HLT_HIEle20_WPLoose_Gsf")

        df = df.Define("vetoElectrons", "Electron_pt_corr > 10 && Electron_cutBased > 0 && abs(Electron_eta) < 2.4")
        df = df.Filter("Sum(vetoElectrons)==1")
        
        df = df.Define("vetoMuons", "Muon_pt > 10 && Muon_looseId && abs(Muon_eta) < 2.4 && abs(Muon_dxybs) < 0.05 && abs(Muon_dz)< 0.2")
        df = df.Filter("Sum(vetoMuons) == 0")

        df = df.Define("Electron_MediumID", "wrem::electron_id::pass_cutbased_noiso<3>(Electron_vidNestedWPBitmap)")
        df = df.Define("goodLeptons", "Electron_MediumID > 0")
        df = df.Filter("Sum(goodLeptons)==1")

        df = df.Define("goodLeptonsPlus", "goodLeptons && Electron_charge > 0")
        df = df.Define("goodLeptonsMinus", "goodLeptons && Electron_charge < 0")

        df = df.Define("goodTrigObjs", "wrem::goodElectronTriggerCandidateLowPU(TrigObj_id, TrigObj_pt, TrigObj_l1pt, TrigObj_l2pt, TrigObj_filterBits)")

        df = df.Define("Lep_pt_uncorr", "Electron_pt_uncorr[goodLeptons]")
        df = df.Define("Lep_pt", "Electron_pt_corr[goodLeptons]")
        df = df.Define("Lep_eta", "Electron_eta[goodLeptons]")
        df = df.Define("Lep_phi", "Electron_phi[goodLeptons]")
        df = df.Define("Lep_charge", "Electron_charge[goodLeptons]")
        df = df.Define("Lep_mass", "Electron_mass[goodLeptons]")
        df = df.Define("Lep_iso", "Electron_pfRelIso03_all[goodLeptons]")

        df = df.Define("passIso", "wrem::electron_id::pass_iso<3>(Electron_vidNestedWPBitmap[goodLeptons])[0] > 0")

    df = df.Define("trigMatch", "wrem::hasTriggerMatchLowPU(Lep_eta, Lep_phi, TrigObj_eta[goodTrigObjs], TrigObj_phi[goodTrigObjs])")
    df = df.Define("nonTrigMatch", "wrem::inverse(trigMatch)")
    df = df.Filter("Sum(trigMatch) > 0")

    df = df.Define("lep_pt_uncorr", "Lep_pt_uncorr[0]")
    df = df.Define("lep_pt", "Lep_pt[0]")
    df = df.Define("lep_eta", "Lep_eta[0]")
    df = df.Define("lep_phi", "Lep_phi[0]")
    df = df.Define("lep_charge", "Lep_charge[0]")
    df = df.Define("lep_mass", "Lep_mass[0]")
    df = df.Define("lep_iso", "Lep_iso[0]")
    
    df = df.Filter(f"lep_pt > {args.pt[1]} && lep_pt < {args.pt[2]}")
    df = muon_selections.apply_met_filters(df)

    if not dataset.is_data: 

        if flavor == "mu":
            df = df.Define("lepSF_ISO", "wrem::lepSF(Lep_pt, Lep_eta, Lep_charge, 1)")
            df = df.Define("lepSF_IDIP", "wrem::lepSF(Lep_pt, Lep_eta, Lep_charge, 2)") # largest effect
            df = df.Define("lepSF_HLT", "wrem::lepSF_HLT_q(Lep_pt, Lep_eta, Lep_charge, 13)")
            df = df.Define("prefireCorr", "wrem::prefireCorr(0, Jet_pt, Jet_eta, Jet_phi, Jet_muEF, Jet_neEmEF, Jet_chEmEF, Photon_pt, Photon_eta, Photon_phi, Lep_pt, Lep_eta, Lep_phi)")
            df = df.Define("SFMC", "lepSF_IDIP*lepSF_ISO*lepSF_HLT*prefireCorr")
        else:
            df = df.Define("lepSF_IDISO", "wrem::lepSF(Lep_pt, Lep_eta, Lep_charge, 3)")
            df = df.Define("lepSF_HLT", "wrem::lepSF_HLT_q(Lep_pt, Lep_eta, Lep_charge, 11)")
            df = df.Define("prefireCorr", "wrem::prefireCorr(0, Jet_pt, Jet_eta, Jet_phi, Jet_muEF, Jet_neEmEF, Jet_chEmEF, Photon_pt, Photon_eta, Photon_phi, Lep_pt, Lep_eta, Lep_phi)")
            df = df.Define("SFMC", "lepSF_IDISO*lepSF_HLT*prefireCorr")

        df = df.Define("exp_weight", "1.0")# "SFMC")
        df = theory_tools.define_theory_weights_and_corrs(df, dataset.name, corr_helpers, args)
    else:
        df = df.DefinePerSample("nominal_weight", "1.0")

    df = df.Define("noTrigMatch", "Sum(trigMatch)")
    results.append(df.HistoBoost("noTrigMatch", [axis_lin], ["noTrigMatch", "nominal_weight"]))

    # Recoil calibrations
    if not args.noRecoil:
        lep_cols = ["lep_pt", "lep_phi", "lep_charge", "lep_pt_uncorr"]
        df = recoilHelper.recoil_W(df, results, dataset, common.vprocs_lowpu, lep_cols) # produces corrected MET as MET_corr_rec_pt/phi  vprocs_lowpu wprocs_recoil_lowpu
    else:
        df = df.Alias("MET_corr_rec_pt", "MET_pt")
        df = df.Alias("MET_corr_rec_phi", "MET_phi")
        df = df.Define("mT_corr_rec", "wrem::mt_2(lep_pt, lep_phi, MET_corr_rec_pt, MET_corr_rec_phi)")

    df = df.Alias("transverseMass", "mT_corr_rec")
    df = df.Define("passMT", f"transverseMass > {mtw_min}")

    results.append(df.HistoBoost("lep_pt", [axis_pt, axis_charge, axis_passMT, axis_passIso], ["lep_pt", "lep_charge", "passMT", "passIso", "nominal_weight"]))
    results.append(df.HistoBoost("lep_eta", [axis_eta, axis_charge, axis_passMT, axis_passIso], ["lep_eta", "lep_charge", "passMT", "passIso", "nominal_weight"]))
    results.append(df.HistoBoost("lep_phi", [axis_phi, axis_charge, axis_passMT, axis_passIso], ["lep_phi", "lep_charge", "passMT", "passIso", "nominal_weight"]))
    results.append(df.HistoBoost("lep_iso", [axis_iso], ["lep_iso", "nominal_weight"]))
   
    # results.append(df.HistoBoost("qcd_space", [axis_pt, axis_eta, axis_iso, axis_charge, axis_mT], ["lep_pt", "lep_eta", "lep_iso", "lep_charge", "transverseMass", "nominal_weight"]))  

    df = df.Define("pxll", "lep_pt * std::cos(lep_phi) + MET_corr_rec_pt * std::cos(MET_corr_rec_phi)")
    df = df.Define("pyll", "lep_pt * std::sin(lep_phi) + MET_corr_rec_pt * std::sin(MET_corr_rec_phi)")
    df = df.Define("ptll", "std::sqrt(pxll*pxll + pyll*pyll)")

    results.append(df.HistoBoost("nominal", axes, [*cols, "nominal_weight"]))
    results.append(df.HistoBoost("transverseMass", axes_mT, [*cols_mT, "nominal_weight"]))

    if not dataset.is_data: 
        # prefire
        df = df.Define("prefireCorr_syst", "wrem::prefireCorr_syst(Jet_pt, Jet_eta, Jet_phi, Jet_muEF, Jet_neEmEF, Jet_chEmEF, Photon_pt, Photon_eta, Photon_phi, Lep_pt, Lep_eta, Lep_phi)")
        df = df.Define("prefireCorr_syst_tensor", "Eigen::TensorFixedSize<double, Eigen::Sizes<2>> res; auto w = nominal_weight*prefireCorr_syst; std::copy(std::begin(w), std::end(w), res.data()); return res;")

        # luminosity, done here as shape variation despite being a flat scaling so to facilitate propagating to fakes afterwards
        df = df.Define("luminosityScaling", f"wrem::constantScaling(nominal_weight, {args.lumiUncertainty})")

        for n, c, a in (("nominal", cols, axes), ("transverseMass", cols_mT, axes_mT)):
                        
            results.append(df.HistoBoost(f"{n}_prefireCorr", [*a], [*c, "prefireCorr_syst_tensor"], tensor_axes = [common.down_up_axis]))

            if dataset.name in common.vprocs_lowpu:
                df = syst_tools.add_theory_hists(results, df, args, dataset.name, corr_helpers, qcdScaleByHelicity_helper, a, c, base_name=n)

            # lepton efficiencies
            if flavor == "mu":
                df = lowPUcfg.lepSF_systs(df, results, "muSF_HLT_DATA_stat", 120, "wrem::lepSF_HLT_var_mu(1, Lep_pt, Lep_eta, Lep_charge)", n, a, c)
                df = lowPUcfg.lepSF_systs(df, results, "muSF_HLT_DATA_syst", 120, "wrem::lepSF_HLT_var_mu(2, Lep_pt, Lep_eta, Lep_charge)", n, a, c)
                df = lowPUcfg.lepSF_systs(df, results, "muSF_HLT_MC_stat",   120, "wrem::lepSF_HLT_var_mu(-1, Lep_pt, Lep_eta, Lep_charge)", n, a, c)
                df = lowPUcfg.lepSF_systs(df, results, "muSF_HLT_MC_syst",   120, "wrem::lepSF_HLT_var_mu(-2, Lep_pt, Lep_eta, Lep_charge)", n, a, c)
                df = lowPUcfg.lepSF_systs(df, results, "muSF_ISO_stat",      36,  "wrem::lepSF_ISO_var_mu(1, Lep_pt, Lep_eta, Lep_charge)", n, a, c)
                df = lowPUcfg.lepSF_systs(df, results, "muSF_ISO_DATA_syst", 36,  "wrem::lepSF_ISO_var_mu(2, Lep_pt, Lep_eta, Lep_charge)", n, a, c)
                df = lowPUcfg.lepSF_systs(df, results, "muSF_ISO_MC_syst",   36,  "wrem::lepSF_ISO_var_mu(-2, Lep_pt, Lep_eta, Lep_charge)", n, a, c)
                df = lowPUcfg.lepSF_systs(df, results, "muSF_IDIP_stat",     36,  "wrem::lepSF_IDIP_var_mu(1, Lep_pt, Lep_eta, Lep_charge)", n, a, c)
                df = lowPUcfg.lepSF_systs(df, results, "muSF_IDIP_DATA_syst",36,  "wrem::lepSF_IDIP_var_mu(2, Lep_pt, Lep_eta, Lep_charge)", n, a, c)
                df = lowPUcfg.lepSF_systs(df, results, "muSF_IDIP_MC_syst",  36,  "wrem::lepSF_IDIP_var_mu(-2, Lep_pt, Lep_eta, Lep_charge)", n, a, c)
            # electron efficiency uncertainties currently don't work
            # else:
            #     df = lowPUcfg.lepSF_systs(df, results, "elSF_HLT_syst",      120, "wrem::lepSF_el_HLT_syst(Lep_pt, Lep_eta, Lep_charge)", n, a, c)
            #     df = lowPUcfg.lepSF_systs(df, results, "elSF_IDISO_syst",    36,  "wrem::lepSF_el_IDISO_syst(Lep_pt, Lep_eta, Lep_charge)", n, a, c)

            results.append(df.HistoBoost(f"{n}_luminosity", a, [*c, "luminosityScaling"], tensor_axes = [common.down_up_axis], storage=hist.storage.Double()))

            if not args.noRecoil:
                df = recoilHelper.add_recoil_unc_W(df, results, dataset, c, a, n)
            

    # if dataset.name in sigProcs:
           
        # Muon momentum scale
        # netabins = 1
        # nweights = 21
        # mag = 1.e-4
        # df = df.Define(f"muonScaleDummy{netabins}Bins", f"wrem::dummyScaleFromMassWeights<{netabins}, {nweights}>(nominal_weight, massWeight_tensor_unscaled, Lep_abs_eta, {mag})")
        # scale_etabins_axis = hist.axis.Regular(netabins, -2.4, 2.4, name="scaleEtaSlice", underflow=False, overflow=False)
        # dummyMuonScaleSyst = df.HistoBoost("mT_corr_rec_muonScaleSyst", [*axes_mT, axis_charge, axis_passMT, axis_passIso], [cols_mT, "Lep_charge", "passMT", "passIso", f"muonScaleDummy{netabins}Bins"], tensor_axes=[down_up_axis, scale_etabins_axis])
        # results.append(dummyMuonScaleSyst)

    if hasattr(dataset, "out_of_acceptance"):
        # Rename dataset to not overwrite the original one
        dataset.name = "Bkg"+dataset.name
     
    return results, weightsum


def build_graph_cutFlow(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")

    results = []

    if dataset.is_data: df = df.DefinePerSample("weight", "1.0")
    else: df = df.Define("weight", "std::copysign(1.0, genWeight)")
    df = df.Define("cutFlow", "1.0")

    weightsum = df.SumAndCount("weight")
    results.append(df.HistoBoost("cutflow_0", [axis_cutFlow], ["cutFlow", "weight"]))
    
    if flavor == "mu":
    
        # at least 1 muon in the collection
        df = df.Filter("Sum(Muon_pt) > 0")
        results.append(df.HistoBoost("cutflow_1", [axis_cutFlow], ["cutFlow", "weight"]))
    
        # loose muon
        df = df.Define("vetoMuons_1", "Muon_pt > 10 && Muon_looseId && abs(Muon_dxybs) < 0.05")
        df = df.Filter("Sum(vetoMuons_1) >= 0")
        results.append(df.HistoBoost("cutflow_2", [axis_cutFlow], ["cutFlow", "weight"]))
    
        # eta acceptance
        df = df.Define("vetoMuons_2", "vetoMuons_1 && abs(Muon_eta) < 2.4")
        df = df.Filter("Sum(vetoMuons_2) > 0")
        results.append(df.HistoBoost("cutflow_3", [axis_cutFlow], ["cutFlow", "weight"]))
        
        # ID
        df = df.Define("vetoMuons_3", "vetoMuons_2 && Muon_mediumId")
        df = df.Filter("Sum(vetoMuons_3) > 0")
        results.append(df.HistoBoost("cutflow_4", [axis_cutFlow], ["cutFlow", "weight"]))
        
        # ISO
        df = df.Define("vetoMuons_4", "vetoMuons_3 && Muon_pfRelIso04_all < 0.15")
        df = df.Filter("Sum(vetoMuons_4) > 0")
        results.append(df.HistoBoost("cutflow_5", [axis_cutFlow], ["cutFlow", "weight"]))
        
        # pT
        df = df.Define("goodLeptons", "vetoMuons_4 && Muon_pt > 25")
        df = df.Filter("Sum(goodLeptons) > 0")
        results.append(df.HistoBoost("cutflow_6", [axis_cutFlow], ["cutFlow", "weight"]))
        
        # 1 muon
        df = df.Filter("Sum(goodLeptons) == 1")
        results.append(df.HistoBoost("cutflow_7", [axis_cutFlow], ["cutFlow", "weight"]))
        
        
        # electron veto
        df = df.Define("vetoElectrons", "Electron_pt > 10 && Electron_cutBased > 0 && abs(Electron_eta) < 2.4")
        df = df.Filter("Sum(vetoElectrons) == 0")
        results.append(df.HistoBoost("cutflow_8", [axis_cutFlow], ["cutFlow", "weight"]))
        
        # trigger
        if not dataset.is_data:  df = df.Filter("HLT_Mu17") 
        else: df = df.Filter("HLT_HIMu17")      
        results.append(df.HistoBoost("cutflow_9", [axis_cutFlow], ["cutFlow", "weight"]))
        
        # trigger match
        df = df.Define("goodTrigObjs", "wrem::goodMuonTriggerCandidateLowPU(TrigObj_id, TrigObj_pt, TrigObj_l1pt, TrigObj_l2pt, TrigObj_filterBits)")
        df = df.Define("trigMatch", "wrem::hasTriggerMatchLowPU(Muon_eta[goodLeptons], Muon_phi[goodLeptons], TrigObj_eta[goodTrigObjs], TrigObj_phi[goodTrigObjs])")
        df = df.Define("nonTrigMatch", "wrem::inverse(trigMatch)")
        df = df.Filter("Sum(trigMatch) > 0")
        results.append(df.HistoBoost("cutflow_10", [axis_cutFlow], ["cutFlow", "weight"]))
        

        df = df.Define("Lep_pt", "Muon_pt[goodLeptons][0]")
        df = df.Define("Lep_eta", "Muon_eta[goodLeptons][0]")
        df = df.Define("Lep_phi", "Muon_phi[goodLeptons][0]")
        df = df.Define("Lep_charge", "Muon_charge[goodLeptons][0]")
        df = df.Define("Lep_mass", "Muon_mass[goodLeptons][0]")
        df = df.Define("Lep_iso", "Muon_pfRelIso04_all[goodLeptons][0]")
        
          
          
        # MET filters       
        df = df.Filter("Flag_globalSuperTightHalo2016Filter && Flag_EcalDeadCellTriggerPrimitiveFilter && Flag_goodVertices && Flag_HBHENoiseIsoFilter && Flag_HBHENoiseFilter && Flag_BadPFMuonFilter")
        results.append(df.HistoBoost("cutflow_11", [axis_cutFlow], ["cutFlow", "weight"]))
    

        # mT cut
        if met == "DeepMETReso": df = df.Define("mT_uncorr", "wrem::mt_2(Lep_pt, Lep_phi, DeepMETResolutionTune_pt, DeepMETResolutionTune_phi)")
        if met == "RawPFMET": df = df.Define("mT_uncorr", "wrem::mt_2(Lep_pt, Lep_phi, RawMET_pt, RawMET_phi)")
        df = df.Filter("mT_uncorr > 40")
        results.append(df.HistoBoost("cutflow_12", [axis_cutFlow], ["cutFlow", "weight"]))


    else:

        if not dataset.is_data: 
            df = df.Define("Electron_pt_corr", "wrem::applyEGammaScaleSmearingUnc(0, Electron_pt, Electron_eta, Electron_dEscaleUp, Electron_dEscaleDown, Electron_dEsigmaUp, Electron_dEsigmaDown, 0)")
            df = df.Filter("HLT_Ele20_WPLoose_Gsf")
            
        else: 
            df = df.Define("Electron_pt_corr", "wrem::applyEGammaScaleSmearingUnc(1, Electron_pt, Electron_eta, Electron_dEscaleUp, Electron_dEscaleDown, Electron_dEsigmaUp, Electron_dEsigmaDown, 0)")
            df = df.Filter("HLT_HIEle20_WPLoose_Gsf")
            
        results.append(df.HistoBoost("cutflow_1", [axis_cutFlow], ["cutFlow", "weight"]))    
        # by default the E/gamma smearings are applied, but are they propagated to the MET? 
        #df = df.Define("Electron_pt_uncorr", "wrem::Egamma_undoCorrection(Electron_pt, Electron_eta, Electron_ecalCorr)")    
        #df = df.Alias("Electron_pt_corr", "Electron_pt")
        
        df = df.Define("vetoElectrons", "Electron_pt_corr > 10 && Electron_cutBased > 0 && abs(Electron_eta) < 2.4")
        df = df.Filter("Sum(vetoElectrons)==2")
        
        df = df.Define("vetoMuons", "Muon_pt > 10 && Muon_looseId && abs(Muon_eta) < 2.4 && abs(Muon_dxybs) < 0.05 && abs(Muon_dz)< 0.2")
        df = df.Filter("Sum(vetoMuons) == 0")
        
        df = df.Define("goodLeptons", "vetoElectrons && Electron_pt_corr > 25 && Electron_cutBased >= 3 && !(abs(Electron_eta) > 1.4442 && abs(Electron_eta) < 1.566)")
        df = df.Define("goodLeptonsPlus", "goodLeptons && Electron_charge > 0")
        df = df.Define("goodLeptonsMinus", "goodLeptons && Electron_charge < 0")
        df = df.Filter("Sum(goodLeptons) == 2")
        
        df = df.Filter("(Electron_charge[goodLeptons][0] + Electron_charge[goodLeptons][1]) == 0")
        df = df.Define("goodTrigObjs", "wrem::goodElectronTriggerCandidateLowPU(TrigObj_id, TrigObj_pt, TrigObj_l1pt, TrigObj_l2pt, TrigObj_filterBits)")
        df = df.Define("trigMatch", "wrem::hasTriggerMatchLowPU(Electron_eta[goodLeptons], Electron_phi[goodLeptons], TrigObj_eta[goodTrigObjs], TrigObj_phi[goodTrigObjs])")
        df = df.Define("nonTrigMatch", "wrem::inverse(trigMatch)")
        df = df.Filter("Sum(trigMatch) > 0")
        
        df = df.Define("Lep_pt_uncorr", "Electron_pt[goodLeptons]")
        df = df.Define("Lep_pt", "Electron_pt_corr[goodLeptons]")
        df = df.Define("Lep_eta", "Electron_eta[goodLeptons]")
        df = df.Define("Lep_phi", "Electron_phi[goodLeptons]")
        df = df.Define("Lep_charge", "Electron_charge[goodLeptons][0]")
        df = df.Define("Lep_mass", "Electron_mass[goodLeptons]")
        
        if not dataset.is_data:
            df = df.Define("lepSF_IDISO", "wrem::lepSF(Electron_pt_corr[goodLeptons], Electron_eta[goodLeptons], Electron_charge[goodLeptons], 3)")
            df = df.Define("lepSF_HLT", "wrem::lepSF_HLT_q(Electron_pt_corr[goodLeptons], Electron_eta[goodLeptons], Electron_charge[goodLeptons], 11)")
            df = df.Define("prefireCorr", "wrem::prefireCorr(0, Jet_pt, Jet_eta, Jet_phi, Jet_muEF, Jet_neEmEF, Jet_chEmEF, Photon_pt, Photon_eta, Photon_phi, Electron_pt_corr[goodLeptons], Electron_eta[goodLeptons], Electron_phi[goodLeptons])")
            df = df.Define("SFMC", "lepSF_IDISO*lepSF_HLT*prefireCorr")
        
        else: df = df.Define("SFMC", "1.0")    
    
    return results, weightsum


resultdict = narf.build_and_run(datasets, build_graph)

if not args.noScaleToData:
    scale_to_data(resultdict)
    aggregate_groups(datasets, resultdict, groups_to_aggregate)

output_tools.write_analysis_output(resultdict, f"mw_lowPU_{flavor}.hdf5", args, update_name=not args.forceDefaultName)
