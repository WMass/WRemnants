import ROOT
import pathlib
import hist
import narf
import numpy as np
import boost_histogram as bh
import logging
import pickle
import lz4.frame

from utilities import common

narf.clingutils.Declare('#include "muon_efficiencies_binned.h"')
narf.clingutils.Declare('#include "muon_efficiencies_binned_vqt_real.h"')

data_dir = f"{pathlib.Path(__file__).parent}/data/"

def make_muon_efficiency_helpers_binned_vqt_real(filename = data_dir + "/testMuonSF/allSmooth_GtoH.root",
                                                 era = None, is_w_like = False, max_pt = np.inf,
                                                 usePseudoSmoothing=False, error=False, step = 2):

    # usePseudoSmoothing will use smoothed nominal histograms with the same pt binning as the original ones.
    # (should do the same for the systematic but the smoothed histogram with original binning is not available at the moment)
    # In this case the uncertainties still come from the original binned histograms.
    # This is not a physically meaningful configuration for a real analysis,
    # and it should be used only for dedicated studies with Asimov when the original version makes the fit unstable.
    # This can happen because of the fakes and antiisolation SF in the W analysis (no issue is expected for Wlike)
    
    eradict = { "2016PreVFP" : "BtoF",
                "2016PostVFP" : "GtoH" }
    eratag = eradict[era]

    axis_eta_eff = None
    axis_pt_eff = None
    # categorical axes in python bindings always have an overflow bin, so use a regular
    # axis for the charge
    axis_charge = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "SF charge")
    axis_charge_inclusive = hist.axis.Regular(1, -2., 2., underflow=False, overflow=False, name = "SF charge") # for isolation and effStat only
    isoEff_types = ["iso", "isonotrig", "antiiso", "antiisonotrig"]
    allEff_types = ["reco", "tracking", "idip", "trigger"] + isoEff_types
    axis_allEff_type = hist.axis.StrCategory(allEff_types, name = "allEff_type")
    axis_nom_syst = hist.axis.Integer(0, 2, underflow = False, overflow =False, name = "nom-syst") # only one syst for now (and the nominal in the first bin)
    
    charges = { -1. : "minus", 1. : "plus" }
    chargeDependentSteps = common.muonEfficiency_chargeDependentSteps
    
    fin = ROOT.TFile.Open(filename)
    if fin is None or fin.IsZombie():
        print(f"Error: file {filename} was not opened correctly")
        quit()

    histNameTag = "smoothWithOriginalPtBins" if usePseudoSmoothing else "original"
    histAltNameTag = "originalDataAltSig" 

    nomiAltTypes = {0: histNameTag,
                    1: histAltNameTag}

    ## because of different pt binning cannot put all histograms in the same boost histogram
    effSyst_manager = {"sf_reco": {"axisLabels" : ["reco"],
                                   "boostHist" : None},
                       "sf_tracking": {"axisLabels" : ["tracking"],
                                       "boostHist" : None},
                       "sf_other": {"axisLabels" : ["idip", "trigger", "iso", "isonotrig", "antiiso", "antiisonotrig"],
                                    "boostHist" : None},
    }
    ## start with NOMI and EFFSYST
    ############
    for effSystKey in effSyst_manager.keys():
        axis_eff_type = None
        histPrefix = "effData" if "effData" in effSystKey else "effMC" if "effMC" in effSystKey else "SF"
        for charge, charge_tag in charges.items():
            for eff_type in effSyst_manager[effSystKey]["axisLabels"]:
                for nomiAltId,nomiAltTag in nomiAltTypes.items():
                    # for iso can use histogram for efficiency variation only in data (the only one we have for now)
                    chargeTag = charge_tag if eff_type in chargeDependentSteps else "both"
                    hist_name = f"{histPrefix}_{nomiAltTag}_{eratag}_{eff_type}_{chargeTag}"
                    hist_root = fin.Get(hist_name)
                    #print(ROOT.AddressOf(hist_root))
                    if hist_root is None:
                        print(f"Error: {hist_name} onot found in file {filename}")
                        quit()
                    #print(f"syst: {eff_type} -> {hist_name}")
                    hist_hist = narf.root_to_hist(hist_root, axis_names = ["SF eta", "SF pt"])

                    if effSyst_manager[effSystKey]["boostHist"] is None:
                        axis_eta_eff = hist_hist.axes[0]
                        axis_pt_eff = hist_hist.axes[1]
                        # store all systs (currently only 1) with the nominal, for all efficiency steps
                        axis_eff_type = hist.axis.StrCategory(effSyst_manager[effSystKey]["axisLabels"], name = f"{effSystKey}_eff_type")
                        effSyst_manager[effSystKey]["boostHist"] = hist.Hist(axis_eta_eff, axis_pt_eff, axis_charge, axis_eff_type, axis_nom_syst, name = effSystKey, storage = hist.storage.Weight())
                    # could use max_pt to remove some of the pt bins for the input histogram
                    effSyst_manager[effSystKey]["boostHist"].view(flow=True)[:, :, axis_charge.index(charge), axis_eff_type.index(eff_type), nomiAltId] = hist_hist.view(flow=True)[:,:]

        # set overflow and underflow eta-pt bins equal to adjacent bins
        effSyst_manager[effSystKey]["boostHist"].view(flow=True)[0, ...] = effSyst_manager[effSystKey]["boostHist"].view(flow=True)[1, ...]
        effSyst_manager[effSystKey]["boostHist"].view(flow=True)[axis_eta_eff.extent-1, ...] = effSyst_manager[effSystKey]["boostHist"].view(flow=True)[axis_eta_eff.extent-2, ...]
        effSyst_manager[effSystKey]["boostHist"].view(flow=True)[:, 0, ...] = effSyst_manager[effSystKey]["boostHist"].view(flow=True)[:, 1, ...]
        effSyst_manager[effSystKey]["boostHist"].view(flow=True)[:, axis_pt_eff.extent-1, ...] = effSyst_manager[effSystKey]["boostHist"].view(flow=True)[:, axis_pt_eff.extent-2, ...]
    
    sf_reco_pyroot = narf.hist_to_pyroot_boost(effSyst_manager["sf_reco"]["boostHist"])
    sf_tracking_pyroot = narf.hist_to_pyroot_boost(effSyst_manager["sf_tracking"]["boostHist"])
    sf_other_pyroot = narf.hist_to_pyroot_boost(effSyst_manager["sf_other"]["boostHist"])
    
    filenames = ROOT.std.vector('string')() #order is isolation, triggerminus, triggerplus
    histonames = ROOT.std.vector('string')()
    isolation3dfilename = f"{data_dir}/testMuonSF/isolation3DSFUT.root"
    isolation3dhistoname = "SF3D_nominal_isolation"
    filenames.push_back(isolation3dfilename)
    histonames.push_back(isolation3dhistoname)
    triggerminus3dfilename = f"{data_dir}/testMuonSF/triggerminus3DSFUT.root"
    triggerminus3dhistoname = "SF3D_nominal_trigger_minus"
    triggerplus3dfilename = f"{data_dir}/testMuonSF/triggerplus3DSFUT.root"
    triggerplus3dhistoname = "SF3D_nominal_trigger_plus"
    filenames.push_back(triggerminus3dfilename)
    histonames.push_back(triggerminus3dhistoname)
    filenames.push_back(triggerplus3dfilename)
    histonames.push_back(triggerplus3dhistoname)
    
    helper = ROOT.wrem_vqt_real.muon_efficiency_binned_helper[str(is_w_like).lower(),
                                                              type(sf_other_pyroot),
                                                              type(sf_tracking_pyroot),
                                                              type(sf_reco_pyroot)](
                                                                  ROOT.std.move(sf_other_pyroot),
                                                                  ROOT.std.move(sf_tracking_pyroot),
                                                                  ROOT.std.move(sf_reco_pyroot),
                                                                  filenames,
                                                                  histonames,
                                                                  error,
                                                                  step
                                                              )
    helper2 = ROOT.wrem.muon_efficiency_binned_helper[str(is_w_like).lower(),
                                                      type(sf_other_pyroot),
                                                      type(sf_tracking_pyroot),
                                                      type(sf_reco_pyroot)](
                                                          ROOT.std.move(sf_other_pyroot),
                                                          ROOT.std.move(sf_tracking_pyroot),
                                                          ROOT.std.move(sf_reco_pyroot),
                                                      )
    helper_syst = ROOT.wrem.muon_efficiency_binned_helper_syst[str(is_w_like).lower(),
                                                               type(sf_other_pyroot),
                                                               type(sf_tracking_pyroot),
                                                               type(sf_reco_pyroot)](
                                                                   helper2
                                                               )

    # define axis for syst variations with all steps
    axis_all = hist.axis.Integer(0, 5, underflow = False, overflow = False, name = "reco-tracking-idip-trigger-iso")
    helper_syst.tensor_axes = [axis_all]


    ##############
    ## now the EFFSTAT    
    effStat_manager = {"sf_reco": {"nPtBins" : None,
                                   "nCharges" : None,
                                   "axisLabels" : ["reco"],
                                   "boostHist" : None,
                                   "helper" : None},
                       "sf_tracking": {"nPtBins" : None,
                                       "nCharges" : None,
                                       "axisLabels" : ["tracking"],
                                       "boostHist" : None,
                                       "helper" : None},
                       "sf_idip": {"nPtBins" : None,
                                   "nCharges" : None,
                                   "axisLabels" : ["idip"],
                                   "boostHist" : None,
                                   "helper" : None},
                       "sf_trigger": {"nPtBins" : None,
                                     "nCharges" : None,
                                      "axisLabels" : ["trigger"],
                                      "boostHist" : None,
                                      "helper" : None},
                       "sf_iso": {"nPtBins" : None,
                                  "nCharges" : None,
                                  "axisLabels" : ["iso", "isonotrig", "antiiso", "antiisonotrig"],
                                  "boostHist" : None,
                                  "helper" : None},
                       # "sf_iso_effData": {"nPtBins" : None,
                       #                    "axisLabels" : ["iso", "isonotrig", "antiiso", "antiisonotrig"],
                       #                    "boostHist" : None,
                       #                    "helper" : None},
                       # "sf_iso_effMC": {"nPtBins" : None,
                       #                  "axisLabels" : ["iso", "isonotrig", "antiiso", "antiisonotrig"],
                       #                  "boostHist" : None,
                       #                  "helper" : None},
    }

    for effStatKey in effStat_manager.keys():
        axis_eff_type = None
        axis_charge_def = None
        histPrefix = "effData" if "effData" in effStatKey else "effMC" if "effMC" in effStatKey else "SF"
        for ic, (charge, charge_tag) in enumerate(charges.items()):
            for eff_type in effStat_manager[effStatKey]["axisLabels"]:
                if eff_type in chargeDependentSteps:
                    axis_charge_def = axis_charge
                    chargeTag = charge_tag
                else:
                    axis_charge_def = axis_charge_inclusive
                    chargeTag = "both"                    
                    if ic: continue # must continue after having set the charge axis
                hist_name = f"{histPrefix}_{histNameTag}_{eratag}_{eff_type}_{chargeTag}"
                hist_root = fin.Get(hist_name)
                if hist_root is None:
                    print(f"Error: {hist_name} onot found in file {filename}")
                    quit()

                nPtBins = int(hist_root.GetNbinsY())
                effStat_manager[effStatKey]["nPtBins"] = nPtBins
                hist_hist = narf.root_to_hist(hist_root, axis_names = ["SF eta", "SF pt"])

                if effStat_manager[effStatKey]["boostHist"] is None:
                    axis_eta_eff = hist_hist.axes[0]
                    axis_pt_eff = hist_hist.axes[1]
                    axis_eff_type = hist.axis.StrCategory(effStat_manager[effStatKey]["axisLabels"], name = f"{effStatKey}_eff_type")
                    effStat_manager[effStatKey]["nCharges"] = 2 if eff_type in chargeDependentSteps else 1
                    effStat_manager[effStatKey]["boostHist"] = hist.Hist(axis_eta_eff,
                                                                         axis_pt_eff,
                                                                         axis_charge_def,
                                                                         axis_eff_type,
                                                                         name = effStatKey,
                                                                         storage = hist.storage.Weight())
                    
                effStat_manager[effStatKey]["boostHist"].view(flow=True)[:, :, axis_charge_def.index(charge), axis_eff_type.index(eff_type)] = hist_hist.view(flow=True)[:,:]
                
        # set overflow and underflow equal to adjacent bins
        effStat_manager[effStatKey]["boostHist"].view(flow=True)[0, ...] = effStat_manager[effStatKey]["boostHist"].view(flow=True)[1, ...]
        effStat_manager[effStatKey]["boostHist"].view(flow=True)[axis_eta_eff.extent-1, ...] = effStat_manager[effStatKey]["boostHist"].view(flow=True)[axis_eta_eff.extent-2, ...]
        effStat_manager[effStatKey]["boostHist"].view(flow=True)[:, 0, ...] = effStat_manager[effStatKey]["boostHist"].view(flow=True)[:, 1, ...]
        effStat_manager[effStatKey]["boostHist"].view(flow=True)[:, axis_pt_eff.extent-1, ...] = effStat_manager[effStatKey]["boostHist"].view(flow=True)[:, axis_pt_eff.extent-2, ...]

        netabins = axis_eta_eff.size
        #originalTnpPtBins = [24., 26., 28., 30., 32., 34., 36., 38., 40., 42., 44., 47., 50., 55.0, 60., 65.]
        # this works because we are using the histogram after smoothing but with the original TnP pt binning 
        nptbins = np.count_nonzero(axis_pt_eff.edges < max_pt) if "tracking" not in effStatKey else axis_pt_eff.size
        logging.info(f"Using {nptbins} pt bins for {effStatKey}")
        ncharges = effStat_manager[effStatKey]["nCharges"]

        sf_stat_pyroot = narf.hist_to_pyroot_boost(effStat_manager[effStatKey]["boostHist"])
        if "sf_iso" in effStatKey:
            helper_stat = ROOT.wrem.muon_efficiency_binned_helper_stat_iso[str(is_w_like).lower(), netabins, nptbins, ncharges, type(sf_stat_pyroot)]( ROOT.std.move(sf_stat_pyroot) )
        else:
            helper_stat = ROOT.wrem.muon_efficiency_binned_helper_stat[str(is_w_like).lower(), netabins, nptbins, ncharges, type(sf_stat_pyroot)]( ROOT.std.move(sf_stat_pyroot) )

        # make new versions of these axes without overflow/underflow to index the tensor
        if isinstance(axis_eta_eff, bh.axis.Regular):
            axis_eta_eff_tensor = hist.axis.Regular(axis_eta_eff.size, axis_eta_eff.edges[0], axis_eta_eff.edges[-1], name = axis_eta_eff.name, overflow = False, underflow = False)
        elif isinstance(axis_eta_eff, bh.axis.Variable):
            axis_eta_eff_tensor = hist.axis.Variable(axis_eta_eff.edges, name = axis_eta_eff.name, overflow = False, underflow = False)
        # for pt need to additionally remove the out of range bins if any
        if isinstance(axis_pt_eff, bh.axis.Regular):
            axis_pt_eff_tensor = hist.axis.Regular(axis_pt_eff.size, axis_pt_eff.edges[0], axis_pt_eff.edges[-1], name = "nPtBins", overflow = False, underflow = False)
        elif isinstance(axis_pt_eff, bh.axis.Variable):
            axis_pt_eff_tensor = hist.axis.Variable(axis_pt_eff.edges[:nptbins+1], name = "nPtBins", overflow = False, underflow = False)

        #print(f"{effStatKey} nPtBins-tensor = {axis_pt_eff_tensor.size}")
        helper_stat.tensor_axes = [axis_eta_eff_tensor, axis_pt_eff_tensor, axis_charge_def]
        effStat_manager[effStatKey]["helper"] = helper_stat

    fin.Close()
    ####
    # return nomi, effSyst, and a dictionary with effStat to use them by name
    return helper, helper_syst, {k : effStat_manager[k]["helper"] for k in effStat_manager.keys()}
