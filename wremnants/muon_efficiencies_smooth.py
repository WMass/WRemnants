import ROOT
import pathlib
import hist
import narf
import numpy as np
import boost_histogram as bh
import pickle
import lz4.frame
import pdb
import copy
import os.path

from utilities import boostHistHelpers as hh
from utilities import common, logging, input_tools
logger = logging.child_logger(__name__)

narf.clingutils.Declare('#include "muon_efficiencies_smooth.h"')

data_dir = common.data_dir

def cloneAxis(ax, overflow=False, underflow=False, newName=None):
    axName = newName if newName else ax.name
    if isinstance(ax, bh.axis.Regular):
        newax = hist.axis.Regular(ax.size, ax.edges[0], ax.edges[-1], name=axName, overflow=overflow, underflow=underflow)
    elif isinstance(ax, bh.axis.Variable):
        newax = hist.axis.Variable(ax.edges, name=axName, overflow=overflow, underflow=underflow)
    else:
        logger.error(f"In cloneAxis(): only Regular and Variable axes are supported for now")
        quit()
    return newax

# TODO: change is_w_like for a python enum AnalysisType (see include/defines.h)
def make_muon_efficiency_helpers_smooth(filename = data_dir + "/testMuonSF/allSmooth_GtoH.root",
                                        era = None,
                                        what_analysis = ROOT.wrem.AnalysisType.Wmass,
                                        max_pt = np.inf,
                                        isoEfficiencySmoothing = False,
                                        smooth3D=False):
    
    logger.debug(f"Make efficiency helper smooth")

    # need the following hack to call the helpers with this enum class from python
    if what_analysis == ROOT.wrem.AnalysisType.Wmass:
        templateAnalysisArg = "wrem::AnalysisType::Wmass"
    elif what_analysis == ROOT.wrem.AnalysisType.Wlike:
        templateAnalysisArg = "wrem::AnalysisType::Wlike"
    elif what_analysis == ROOT.wrem.AnalysisType.Dilepton:
        templateAnalysisArg = "wrem::AnalysisType::Dilepton"
    else:
        logger.error(f"Analysis {what_analysis} not implemented. Please check")
        quit()
    logger.debug(f"Running {templateAnalysisArg.split('::')[-1]} analysis")
        
    eradict = { "2016PreVFP" : "BtoF",
                "2016PostVFP" : "GtoH" }
    eratag = eradict[era]

    axis_eta_eff = None
    axis_pt_eff = None
    # categorical axes in python bindings always have an overflow bin, so use a regular
    # axis for the charge
    axis_charge = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "SF charge")
    axis_charge_inclusive = hist.axis.Regular(1, -2., 2., underflow=False, overflow=False, name = "SF charge") # for isolation and effStat only
    isoEff_types = ["iso", "isonotrig", "antiiso", "isoantitrig"]
    trigEff_types = ["trigger", "antitrigger"] # antitrigger is P(failTrigger|IDIP), SF obtained from trigger SF as (1-SF*effMC)/(1-effMC), similarly to antiiso
    allEff_types = ["reco", "tracking", "idip"] + trigEff_types + isoEff_types
    eff_types_3D = [] if not smooth3D else [x for x in trigEff_types] + [x for x in isoEff_types]
    eff_types_2D = [x for x in allEff_types if x not in eff_types_3D]
    logger.info(f"SF steps in 2D (eta-pt): {eff_types_2D}")
    logger.info(f"SF steps in 3D (eta-pt-ut): {eff_types_3D}")
    axis_allEff_type = hist.axis.StrCategory(allEff_types, name = "allEff_type")
    axis_eff_type_2D = hist.axis.StrCategory(eff_types_2D, name = "eff_types_2D_etapt")
    axis_eff_type_3D = hist.axis.StrCategory(eff_types_3D, name = "eff_types_3D_etaptut")
    effSyst_decorrEtaEdges = [round(-2.4 + 0.1*i,1) for i in range(49)]
    Nsyst = 1 + (len(effSyst_decorrEtaEdges) - 1) # 1 inclusive variation + all decorrelated bins
    axis_nom_syst = hist.axis.Integer(0, 1+Nsyst, underflow = False, overflow =False, name = "nom-systs") # nominal in first bin

    charges = { -1. : "minus", 1. : "plus" }
    chargeDependentSteps = common.muonEfficiency_chargeDependentSteps

    fin = input_tools.safeOpenRootFile(filename)

    dict_SF3D = None
    if len(eff_types_3D):
        fileSF3D = f"{data_dir}/testMuonSF/smoothSF3D_uTm30to100.pkl.lz4"
        if not os.path.isfile(fileSF3D):
            raise IOError(f"Couldn't read 3D SF file {fileSF3D}, make sure you have it.")
        logger.info(f"3D SF read from {fileSF3D}")
        with lz4.frame.open(fileSF3D) as f3D:
            dict_SF3D = pickle.load(f3D)
    else:
        logger.warning("Using all SF in 2D")

    ## start with NOMI and SYST
    sf_syst_2D = None
    ############
    sf_syst_from2D_for3D = {}
    # all ut-independent steps, for the nominal and systematics
    for charge, charge_tag in charges.items():
        for eff_type in allEff_types: # should only loop on axis_eff_type_2D but must get syst in 2D to propagate to 3D later
            # for iso can use histogram for efficiency variation only in data (the only one we have for now)
            nameTag = "nomiAndAlt_onlyDataVar" if (isoEfficiencySmoothing and any(x in eff_type for x in ["iso", "antiiso"])) else "nomiAndAlt"
            chargeTag = charge_tag if eff_type in chargeDependentSteps else "both"
            hist_name = f"SF_{nameTag}_{eratag}_{eff_type}_{chargeTag}"
            ## temporary patch for missing histograms
            if eff_type == "antitrigger":
                hist_name = hist_name.replace("antitrigger", "trigger")
                logger.warning(f"Substituting temporarily missing 2D histogram for 'antitrigger' with 'trigger'")
            elif eff_type == "isoantitrig":
                hist_name = hist_name.replace("isoantitrig", "isonotrig")
                logger.warning(f"Substituting temporarily missing 2D histogram for 'isoantitrig' with 'isonotrig'")
            hist_root = input_tools.safeGetRootObject(fin, hist_name)
            #logger.debug(f"syst: {eff_type} -> {hist_name}")

            hist_hist = narf.root_to_hist(hist_root, axis_names = ["SF eta", "SF pt", "nomi-statUpDown-syst"])
            # the following axis might change for different histograms, because of a different number of effStat variations
            axis_nomiAlt_eff = hist_hist.axes[2]
            if eff_type not in axis_eff_type_2D:
                key = f"{eff_type}_{chargeTag}"
                if key not in sf_syst_from2D_for3D.keys():
                    # take syst/nomi histogram ratio in 2D (eta-pt)
                    sf_syst_from2D_for3D[key] = hh.divideHists(hist_hist[:,:,axis_nomiAlt_eff.size-1], hist_hist[:,:,0], createNew=True)
                    logger.debug(f"Storing 2D eta-pt syst for {key} SF to propagate to 3D version")
                    # histogram is saved, now skip the rest
                continue
            
            if sf_syst_2D is None:
                axis_eta_eff = hist_hist.axes[0]
                axis_pt_eff = hist_hist.axes[1]
                # store all systs (currently only 1) with the nominal, for all efficiency steps
                sf_syst_2D = hist.Hist(axis_eta_eff, axis_pt_eff, axis_charge, axis_eff_type_2D, axis_nom_syst, name = "sf_syst_2D", storage = hist.storage.Weight())
            # could use max_pt to remove some of the pt bins for the input histogram
            # extract nominal (first bin that is not underflow) and put in corresponding bin of destination
            sf_syst_2D.view(flow=False)[:, :, axis_charge.index(charge), axis_eff_type_2D.index(eff_type), 0] = hist_hist.view(flow=False)[:,:,0]
            # extract syst (last bin except overflow) and put in corresponding bin of destination (bin 1 is the second bin because no underflow)
            sf_syst_2D.view(flow=False)[:, :, axis_charge.index(charge), axis_eff_type_2D.index(eff_type), 1] = hist_hist.view(flow=False)[:,:,axis_nomiAlt_eff.size-1]
            for isyst in range(len(effSyst_decorrEtaEdges)-1):
                # first copy the nominal
                sf_syst_2D.view(flow=False)[:, :, axis_charge.index(charge), axis_eff_type_2D.index(eff_type), 2+isyst] = hist_hist.view(flow=False)[:,:,0]
                # now update with actual syst all eta bins inside interval [effSyst_decorrEtaEdges[isyst], effSyst_decorrEtaEdges[isyst+1]]
                # add epsilon to ensure picking the bin on the right of the edge (for the right edge given by
                # effSyst_decorrEtaEdges[isyst+1]] the range selection in boost later on will stop at the left
                #edge of the chosen bin number, e.g. h[b:b+1] will pick the range containing the single bin b, unlike in ROOT
                indexEtaLow = axis_eta_eff.index(effSyst_decorrEtaEdges[isyst] + 0.001) # add epsilon to ensure picking the bin on the right of the edge
                indexEtaHigh = axis_eta_eff.index(effSyst_decorrEtaEdges[isyst+1] + 0.001) 
                sf_syst_2D.view(flow=False)[indexEtaLow:indexEtaHigh, :, axis_charge.index(charge), axis_eff_type_2D.index(eff_type), 2+isyst] = hist_hist.view(flow=False)[indexEtaLow:indexEtaHigh, :,axis_nomiAlt_eff.size-1]
                        
    # set overflow and underflow eta-pt bins equal to adjacent bins
    sf_syst_2D.view(flow=True)[0, ...] = sf_syst_2D.view(flow=True)[1, ...]
    sf_syst_2D.view(flow=True)[axis_eta_eff.extent-1, ...] = sf_syst_2D.view(flow=True)[axis_eta_eff.extent-2, ...]
    sf_syst_2D.view(flow=True)[:, 0, ...] = sf_syst_2D.view(flow=True)[:, 1, ...]
    sf_syst_2D.view(flow=True)[:, axis_pt_eff.extent-1, ...] = sf_syst_2D.view(flow=True)[:, axis_pt_eff.extent-2, ...]

    ## now proceed with loading 3D SF if existing, then create helpers as appropriate
    if len(eff_types_3D):

        sf_syst_3D = None
        ############
        # all ut-dependent steps, for the nominal and systematics
        for charge, charge_tag in charges.items():
            for eff_type in axis_eff_type_3D:
                effTypeNameInHist = f"{eff_type}{charge_tag}" if eff_type in chargeDependentSteps else eff_type
                hist_hist = dict_SF3D[f"smoothSF3D_{effTypeNameInHist}"]
                #logger.debug(f"{charge} {eff_type}: hist_hist = {hist_hist.axes}")
                if sf_syst_3D is None:
                    ## Redefine axes to force the existence of overflow bins
                    axis_eta_eff = cloneAxis(hist_hist.axes[0], overflow=True, underflow=True)
                    axis_pt_eff = cloneAxis(hist_hist.axes[1], overflow=True, underflow=True)
                    axis_ut_eff = cloneAxis(hist_hist.axes[2], overflow=True, underflow=True)
                    # ut axis is put last to keep same axes sorting as for 2D histograms created before
                    sf_syst_3D = hist.Hist(axis_eta_eff, axis_pt_eff, axis_charge,
                                           axis_eff_type_3D, axis_nom_syst, axis_ut_eff,
                                           name = "sf_syst_3D", storage = hist.storage.Weight())
                # could use max_pt to remove some of the pt bins for the input histogram
                #
                # extract nominal (first bin that is not underflow) and put in corresponding bin of destination (bin 0 is the first bin because no underflow)
                ## Note: must call sf_syst_3D with flow=False because it has overflow bins but hist_hist does not (it was made without them in 4D)
                nominalLayer = hist_hist.view(flow=False)[:,:,:,0]
                sf_syst_3D.view(flow=False)[:, :, axis_charge.index(charge), axis_eff_type_3D.index(eff_type), 0, :] = nominalLayer #hist_hist.view(flow=False)[:,:,:,0]
                # take syst/nomi histogram ratio in 2D (eta-pt), and broadcast into eta-pt-ut)
                chargeTag = charge_tag if eff_type in chargeDependentSteps else "both"
                syst_view_etaPt = sf_syst_from2D_for3D[f"{eff_type}_{chargeTag}"].values(flow=False) #only need values
                # broadcast systOverNomi into a 3D histogram adding ut axis
                shape_etaPtUt = nominalLayer.shape
                # The transpose is because numpy works right to left in broadcasting, and we have the ut axis on the right
                tmp_broadcast = np.broadcast_to(syst_view_etaPt.T, shape_etaPtUt[::-1])
                syst_view_etaPtUt = tmp_broadcast.T
                # fill syst with product of nominal and 2D syst/nomi ratio
                # first copy the nominal and then update values (no variances) multiplying by the syst/nomi ratio
                sf_syst_3D.view(flow=False)[:, :, axis_charge.index(charge), axis_eff_type_3D.index(eff_type), 1, :] = nominalLayer
                sf_syst_3D.values(flow=False)[:, :, axis_charge.index(charge), axis_eff_type_3D.index(eff_type), 1, :] *= syst_view_etaPtUt
                for isyst in range(len(effSyst_decorrEtaEdges)-1):
                    # first copy the nominal
                    sf_syst_3D.view(flow=False)[:, :, axis_charge.index(charge), axis_eff_type_3D.index(eff_type), 2+isyst, :] = nominalLayer # hist_hist.view(flow=False)[:,:,:,0]
                    # now update with actual syst all eta bins inside interval [effSyst_decorrEtaEdges[isyst], effSyst_decorrEtaEdges[isyst+1]]
                    # add epsilon to ensure picking the bin on the right of the edge (for the right edge given by
                    # effSyst_decorrEtaEdges[isyst+1]] the range selection in boost later on will stop at the left
                    #edge of the chosen bin number, e.g. h[b:b+1] will pick the range containing the single bin b, unlike in ROOT
                    # also do not sum 1 because sf_syst_3D.view(flow=False) will consider 0 the first bin index (with flow=True instead 0 is the underflow but only if it exists, otherwise 0 is the first bin)
                    indexEtaLow = axis_eta_eff.index(effSyst_decorrEtaEdges[isyst] + 0.001) # add epsilon to ensure picking the bin on the right of the edge
                    indexEtaHigh = axis_eta_eff.index(effSyst_decorrEtaEdges[isyst+1] + 0.001) 
                    sf_syst_3D.view(flow=False)[indexEtaLow:indexEtaHigh, :, axis_charge.index(charge), axis_eff_type_3D.index(eff_type), 2+isyst, :] *= syst_view_etaPtUt[indexEtaLow:indexEtaHigh, :, :] # hist_hist.view(flow=False)[indexEtaLow:indexEtaHigh, :, :, 0]
                    
        # set overflow and underflow eta-pt bins equal to adjacent bins
        sf_syst_3D.view(flow=True)[0, ...]                       = sf_syst_3D.view(flow=True)[1, ...]
        sf_syst_3D.view(flow=True)[axis_eta_eff.extent-1, ...]   = sf_syst_3D.view(flow=True)[axis_eta_eff.extent-2, ...]
        sf_syst_3D.view(flow=True)[:, 0, ...]                    = sf_syst_3D.view(flow=True)[:, 1, ...]
        sf_syst_3D.view(flow=True)[:, axis_pt_eff.extent-1, ...] = sf_syst_3D.view(flow=True)[:, axis_pt_eff.extent-2, ...]
        sf_syst_3D.view(flow=True)[..., 0]                       = sf_syst_3D.view(flow=True)[..., 1]
        sf_syst_3D.view(flow=True)[..., axis_ut_eff.extent-1]    = sf_syst_3D.view(flow=True)[..., axis_ut_eff.extent-2]

        ## try rebinning for tests about memory usage
        rebinUt = 0
        if rebinUt:
            logger.error(f"Attention, rebinning ut by {rebinUt} as a test")
            sf_syst_3D = sf_syst_3D[{axis_ut_eff.name : hist.rebin(rebinUt)}]
        #logger.debug(f"sf_syst_2D.shape = {sf_syst_2D.shape}")
        #logger.debug(f"sf_syst_3D.shape = {sf_syst_3D.shape}") # this is currently (48, 205, 2, 5, 50, N_ut) for eta,pt,charge,effType,nomiAndSyst,ut        
        sf_syst_2D_pyroot = narf.hist_to_pyroot_boost(sf_syst_2D)
        sf_syst_3D_pyroot = narf.hist_to_pyroot_boost(sf_syst_3D)
        # nomi and syst are stored in the same histogram, just use different helpers to override the () operator for now, until RDF is improved
        helper = ROOT.wrem.muon_efficiency_smooth3D_helper[templateAnalysisArg, Nsyst,
                                                           type(sf_syst_2D_pyroot),
                                                           type(sf_syst_3D_pyroot)](
                                                               ROOT.std.move(sf_syst_2D_pyroot),
                                                               ROOT.std.move(sf_syst_3D_pyroot)
                                                           )
        helper_syst = ROOT.wrem.muon_efficiency_smooth3D_helper_syst[templateAnalysisArg, Nsyst, type(sf_syst_2D_pyroot), type(sf_syst_3D_pyroot)]( helper )
        # define axis for syst variations with all steps
        axis_all = hist.axis.Integer(0, 5, underflow = False, overflow = False, name = "reco-tracking-idip-trigger-iso")
        axis_nsyst = hist.axis.Integer(0, Nsyst, underflow = False, overflow = False, name = "n_syst_variations")
        helper_syst.tensor_axes = [axis_all, axis_nsyst]
        #

    else:
        # case with only 2D histograms
        sf_syst_2D_pyroot = narf.hist_to_pyroot_boost(sf_syst_2D)
        # nomi and syst are stored in the same histogram, just use different helpers to override the () operator for now, until RDF is improved
        helper = ROOT.wrem.muon_efficiency_smooth_helper[templateAnalysisArg, Nsyst, type(sf_syst_2D_pyroot)]( ROOT.std.move(sf_syst_2D_pyroot) )
        helper_syst = ROOT.wrem.muon_efficiency_smooth_helper_syst[templateAnalysisArg, Nsyst, type(sf_syst_2D_pyroot)]( helper )
        # define axis for syst variations with all steps
        axis_all = hist.axis.Integer(0, 5, underflow = False, overflow = False, name = "reco-tracking-idip-trigger-iso")
        axis_nsyst = hist.axis.Integer(0, Nsyst, underflow = False, overflow = False, name = "n_syst_variations")
        helper_syst.tensor_axes = [axis_all, axis_nsyst]
        #
                       
    ##############
    ## now the EFFSTAT
    
    # for the stat histograms, an axis will contain the efficiency type, in some cases it might have a single bin (e.g. tracking, reco, and idip)
    # but for trigger and isolation needs more, because of the different cases to handle inside the helper class (pass/fail trigger or iso/antiiso with/without trigger)
    # could actually stack reco-idip-trig together, but better to have the helper deal with a single histogram at a time
    # then there will be a specialization of the class for isolation
    # in this way the labels for the effType axis can be used to activate some internal behaviour as needed (for example trigger won't have variations for not triggering lepton)

    # when smooth3D = True we read histograms from root or boost depending on how they were made (some steps are still from previous root files since they have no uT dependence)
    # however, we always add the uT axis to the boost histogram in input to the tensor, to simplify the usage, but when is3D = False a dummy ut axis with 1 bin is created
    effStat_manager = {"sf_reco": {"nPtEigenBins" : None,
                                   "nCharges" : None,
                                   "axisLabels" : ["reco"],
                                   "boostHist" : None,
                                   "helper" : None,
                                   "is3D": False},
                       "sf_tracking": {"nPtEigenBins" : None,
                                       "nCharges" : None,
                                       "axisLabels" : ["tracking"],
                                       "boostHist" : None,
                                       "helper" : None,
                                       "is3D": False},
                       "sf_idip": {"nPtEigenBins" : None,
                                   "nCharges" : None,
                                   "axisLabels" : ["idip"],
                                   "boostHist" : None,
                                   "helper" : None,
                                   "is3D": False},
                       "sf_trigger": {"nPtEigenBins" : None,
                                      "nCharges" : None,
                                      "axisLabels" : ["trigger", "antitrigger"],
                                      "boostHist" : None,
                                      "helper" : None,
                                      "is3D": smooth3D},
                       }
    if not isoEfficiencySmoothing:
        effStat_manager["sf_iso"] = {"nPtEigenBins" : None,
                                     "nCharges" : None,
                                     "axisLabels" : ["iso", "isonotrig", "antiiso", "isoantitrig"],
                                     "boostHist" : None,
                                     "helper" : None,
                                     "is3D": smooth3D}

    else:
        # these were never done in 3D, so can stay with is3D = False
        effStat_manager["sf_iso_effData"] =  {"nPtEigenBins" : None,
                                              "nCharges" : None,
                                              "axisLabels" : ["iso", "isonotrig", "antiiso", "isoantitrig"],
                                              "boostHist" : None,
                                              "helper" : None,
                                              "is3D": False},
        effStat_manager["sf_iso_effMC"] = {"nPtEigenBins" : None,
                                           "nCharges" : None,
                                           "axisLabels" : ["iso", "isonotrig", "antiiso", "isoantitrig"],
                                           "boostHist" : None,
                                           "helper" : None,
                                           "is3D": False}
        
    for effStatKey in effStat_manager.keys():
        nom_up_effStat_axis = None
        axis_eff_type = None
        axis_charge_def = None
        is3D = effStat_manager[effStatKey]["is3D"]
        for ic, (charge, charge_tag) in enumerate(charges.items()):
            for eff_type in effStat_manager[effStatKey]["axisLabels"]:
                
                if eff_type in chargeDependentSteps:
                    axis_charge_def = axis_charge
                    chargeTag = charge_tag
                else:
                    axis_charge_def = axis_charge_inclusive
                    chargeTag = "both"                    
                    if ic: continue # must continue after having set the charge axis

                if is3D:
                    effTypeNameInHist = f"{eff_type}{charge_tag}" if eff_type in chargeDependentSteps else eff_type
                    hist_hist = dict_SF3D[f"smoothSF3D_{effTypeNameInHist}"] # this is a 4D histogram, eta-pt-ut-var, with NO OVERFLOWS (unlike the standard 2D case where it comes from root)
                    nPtEigenBins = int(hist_hist.axes[3].size - 1) # 1 nominal + N eigen variations, we need N here
                else:
                    nameTag = "nomiAndAlt"
                    if "effData" in effStatKey:
                        nameTag += "_onlyDataVar"
                    elif "effMC" in effStatKey:
                        nameTag += "_onlyMCVar"
                    hist_name = f"SF_{nameTag}_{eratag}_{eff_type}_{chargeTag}"
                    if eff_type == "antitrigger":
                        hist_name = hist_name.replace("antitrigger", "trigger")
                        logger.warning(f"Substituting temporarily missing 2D histogram for 'antitrigger' with 'trigger'")
                    elif eff_type == "isoantitrig":
                        hist_name = hist_name.replace("isoantitrig", "isonotrig")
                        logger.warning(f"Substituting temporarily missing 2D histogram for 'isoantitrig' with 'isonotrig'")

                    hist_root = input_tools.safeGetRootObject(fin, hist_name)
                    # logger.info(f"stat: {effStatKey}|{eff_type} -> {hist_name}")
                    # logger.info(ROOT.AddressOf(hist_root))
                    if hist_root is None:
                        logger.info(f"Error: {hist_name} not found in file {filename}")
                        quit()
                    # Z axis has nominal, 2*N effStat (first N up then N down) and one syst in the last bin
                    nPtEigenBins = int(int(hist_root.GetNbinsZ() - 2)/ 2)
                    hist_hist = narf.root_to_hist(hist_root, axis_names = ["SF eta", "SF pt", "nomi-statUpDown-syst"])

                if nom_up_effStat_axis is None:
                    effStat_manager[effStatKey]["nPtEigenBins"] = nPtEigenBins
                    effStat_manager[effStatKey]["nCharges"] = 2 if eff_type in chargeDependentSteps else 1
                    nom_up_effStat_axis = hist.axis.Regular(int(1+nPtEigenBins), -0.5, 0.5+nPtEigenBins, underflow=False, overflow=False, name = "nomUpVar")

                if effStat_manager[effStatKey]["boostHist"] is None:
                    axis_eta_eff = cloneAxis(hist_hist.axes[0], overflow=True, underflow=True, newName="SF eta")
                    axis_pt_eff = cloneAxis(hist_hist.axes[1], overflow=True, underflow=True, newName="SF pt")
                    axis_ut_eff = cloneAxis(hist_hist.axes[2], overflow=True, underflow=True, newName="SF ut") if is3D else hist.axis.Regular(1, -1e6, 1e6, name = "SF ut")
                    axis_eff_type = hist.axis.StrCategory(effStat_manager[effStatKey]["axisLabels"], name = f"{effStatKey}_eff_type")
                    if smooth3D:
                        effStat_manager[effStatKey]["boostHist"] = hist.Hist(axis_eta_eff, axis_pt_eff, axis_charge_def,
                                                                             axis_eff_type,
                                                                             nom_up_effStat_axis,
                                                                             axis_ut_eff,
                                                                             name = effStatKey,
                                                                             storage = hist.storage.Weight())
                    else:
                        effStat_manager[effStatKey]["boostHist"] = hist.Hist(axis_eta_eff, axis_pt_eff, axis_charge_def,
                                                                             axis_eff_type,
                                                                             nom_up_effStat_axis,
                                                                             name = effStatKey,
                                                                             storage = hist.storage.Weight())

                        
                # hist_hist may or may not have overflows, but the left-hand side histogram have them: read with flow=False to get only things in acceptance here
                if smooth3D:
                    if is3D:
                        # hist_hist has dimension 4, ut as third axes
                        effStat_manager[effStatKey]["boostHist"].view(flow=False)[:, :, axis_charge_def.index(charge), axis_eff_type.index(eff_type), nom_up_effStat_axis.index(0), :] = hist_hist.view(flow=False)[:,:,:, 0]
                        for iup in range(1, 1 + nPtEigenBins):
                            effStat_manager[effStatKey]["boostHist"].view(flow=False)[:, :, axis_charge_def.index(charge), axis_eff_type.index(eff_type), nom_up_effStat_axis.index(iup), :] = hist_hist.view(flow=False)[:,:,:, iup]
                    else:
                        # hist_hist has dimension 3, no ut axis
                        effStat_manager[effStatKey]["boostHist"].view(flow=False)[:, :, axis_charge_def.index(charge), axis_eff_type.index(eff_type), nom_up_effStat_axis.index(0), 0] = hist_hist.view(flow=False)[:,:, 0]
                        for iup in range(1, 1 + nPtEigenBins):
                            effStat_manager[effStatKey]["boostHist"].view(flow=False)[:, :, axis_charge_def.index(charge), axis_eff_type.index(eff_type), nom_up_effStat_axis.index(iup), 0] = hist_hist.view(flow=False)[:,:, iup]
                else:
                    # boostHist has no ut axis
                    effStat_manager[effStatKey]["boostHist"].view(flow=False)[:, :, axis_charge_def.index(charge), axis_eff_type.index(eff_type), nom_up_effStat_axis.index(0)] = hist_hist.view(flow=False)[:,:, 0]
                    for iup in range(1, 1 + nPtEigenBins):
                        effStat_manager[effStatKey]["boostHist"].view(flow=False)[:, :, axis_charge_def.index(charge), axis_eff_type.index(eff_type), nom_up_effStat_axis.index(iup)] = hist_hist.view(flow=False)[:,:, iup]
                    

        # set overflow and underflow equal to adjacent bins
        effStat_manager[effStatKey]["boostHist"].view(flow=True)[0, ...] = effStat_manager[effStatKey]["boostHist"].view(flow=True)[1, ...]
        effStat_manager[effStatKey]["boostHist"].view(flow=True)[axis_eta_eff.extent-1, ...] = effStat_manager[effStatKey]["boostHist"].view(flow=True)[axis_eta_eff.extent-2, ...]
        effStat_manager[effStatKey]["boostHist"].view(flow=True)[:, 0, ...] = effStat_manager[effStatKey]["boostHist"].view(flow=True)[:, 1, ...]
        effStat_manager[effStatKey]["boostHist"].view(flow=True)[:, axis_pt_eff.extent-1, ...] = effStat_manager[effStatKey]["boostHist"].view(flow=True)[:, axis_pt_eff.extent-2, ...]
        if smooth3D:
            effStat_manager[effStatKey]["boostHist"].view(flow=True)[..., 0] = effStat_manager[effStatKey]["boostHist"].view(flow=True)[..., 1]
            effStat_manager[effStatKey]["boostHist"].view(flow=True)[..., axis_ut_eff.extent-1] = effStat_manager[effStatKey]["boostHist"].view(flow=True)[..., axis_ut_eff.extent-2]

        netabins = axis_eta_eff.size
        ncharges = effStat_manager[effStatKey]["nCharges"]
        sf_stat_pyroot = narf.hist_to_pyroot_boost(effStat_manager[effStatKey]["boostHist"])
        if smooth3D:
            if "sf_iso" in effStatKey:
                helper_stat = ROOT.wrem.muon_efficiency_smooth_helper_stat_iso_utDep[templateAnalysisArg, netabins, nPtEigenBins, ncharges, type(sf_stat_pyroot)]( ROOT.std.move(sf_stat_pyroot) )
            else:
                helper_stat = ROOT.wrem.muon_efficiency_smooth_helper_stat_utDep[templateAnalysisArg, netabins, nPtEigenBins, ncharges, type(sf_stat_pyroot)]( ROOT.std.move(sf_stat_pyroot) )            
        else:
            if "sf_iso" in effStatKey:
                helper_stat = ROOT.wrem.muon_efficiency_smooth_helper_stat_iso[templateAnalysisArg, netabins, nPtEigenBins, ncharges, type(sf_stat_pyroot)]( ROOT.std.move(sf_stat_pyroot) )
            else:
                helper_stat = ROOT.wrem.muon_efficiency_smooth_helper_stat[templateAnalysisArg, netabins, nPtEigenBins, ncharges, type(sf_stat_pyroot)]( ROOT.std.move(sf_stat_pyroot) )
        # make new versions of these axes without overflow/underflow to index the tensor
        if isinstance(axis_eta_eff, bh.axis.Regular):
            axis_eta_eff_tensor = hist.axis.Regular(axis_eta_eff.size, axis_eta_eff.edges[0], axis_eta_eff.edges[-1], name = axis_eta_eff.name, overflow = False, underflow = False)
        elif isinstance(axis_eta_eff, bh.axis.Variable):
            axis_eta_eff_tensor = hist.axis.Variable(axis_eta_eff.edges, name = axis_eta_eff.name, overflow = False, underflow = False)
        axis_ptEigen_eff_tensor = hist.axis.Integer(0, effStat_manager[effStatKey]["nPtEigenBins"], underflow = False, overflow =False, name = "nPtEigenBins")    
        effStatTensorAxes = [axis_eta_eff_tensor, axis_ptEigen_eff_tensor, axis_charge_def]
        helper_stat.tensor_axes = effStatTensorAxes
        effStat_manager[effStatKey]["helper"] = helper_stat

    fin.Close()

    logger.debug(f"Return efficiency helper!")

    ####
    # return nomi, effsyst, and a dictionary with effStat to use them by name
    return helper, helper_syst, {k : effStat_manager[k]["helper"] for k in effStat_manager.keys()}
