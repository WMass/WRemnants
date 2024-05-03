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
from utilities import common, logging
from utilities.io_tools import input_tools
logger = logging.child_logger(__name__)

narf.clingutils.Declare('#include "muon_efficiencies_veto.h"')

data_dir = common.data_dir

def make_muon_efficiency_helpers_veto(filename_plus = data_dir + "/muonSF/smoothedSFandEffi_newveto_regular_GtoH_plus.root",
                                      filename_minus = data_dir + "/muonSF/smoothedSFandEffi_newveto_regular_GtoH_minus.root",
                                      era = None):
    
    logger.debug(f"Make efficiency helper veto")

    #FIXME:to be updated for when other eras are available
        
    eradict = { "2016PreVFP" : "BtoF",
                "2016PostVFP" : "GtoH",
                "2017" : "GtoH", #FIXME: update later when SF for 2018 is available
                "2018" : "GtoH" }
    eratag = eradict[era]

    effSyst_decorrEtaEdges = [round(-2.4 + 0.1*i,1) for i in range(49)]
    Nsyst = 1 + (len(effSyst_decorrEtaEdges) - 1) # 1 inclusive variation + all decorrelated bins

    Steps = 3 #we decided to compute the syst variations on the veto SFs independently for each of the tnp fits (using only global muons in the muon definition we fit reco, "tracking", looseID + dxybs)

    file_plus = input_tools.safeOpenRootFile(f"{filename_plus}")
    plus_histo = input_tools.safeGetRootObject(file_plus,"SF_nomiAndAlt_GtoH_newveto_plus")
    file_minus = input_tools.safeOpenRootFile(f"{filename_minus}")
    minus_histo = input_tools.safeGetRootObject(file_minus,"SF_nomiAndAlt_GtoH_newveto_minus")

    NPtEigenBins = int(int(plus_histo.GetNbinsZ() - 1 - Steps)/ 2)

    NEtaBins = plus_histo.GetNbinsX()

    yaxis = plus_histo.GetYaxis()
    minpt = yaxis.GetBinLowEdge(1)
    maxpt = yaxis.GetBinUpEdge(yaxis.GetNbins())
    xaxis = plus_histo.GetXaxis()
    mineta = xaxis.GetBinLowEdge(1)
    maxeta = xaxis.GetBinUpEdge(xaxis.GetNbins())

    hist_plus = narf.root_to_hist(plus_histo, axis_names = ["SF eta", "SF pt", "nomi-statUpDown-syst"])
    hist_minus = narf.root_to_hist(minus_histo, axis_names = ["SF eta", "SF pt", "nomi-statUpDown-syst"])
    hist_plus_pyroot = narf.hist_to_pyroot_boost(hist_plus)
    hist_minus_pyroot = narf.hist_to_pyroot_boost(hist_minus)

    helper = ROOT.wrem.muon_efficiency_veto_helper[type(hist_plus_pyroot),NEtaBins,NPtEigenBins,Nsyst,Steps](
                                                                                                                 ROOT.std.move(hist_plus_pyroot),
                                                                                                                 ROOT.std.move(hist_minus_pyroot),
                                                                                                                 minpt,
                                                                                                                 maxpt,
                                                                                                                 mineta,
                                                                                                                 maxeta
                                                                                                             )

    helper_syst = ROOT.wrem.muon_efficiency_veto_helper_syst[type(hist_plus_pyroot),NEtaBins,NPtEigenBins,Nsyst,Steps](
                                                                                                                           helper
                                                                                                                       )

    axis_all = hist.axis.Integer(0, Steps, underflow = False, overflow = False, name = "veto_reco-veto_tracking-veto_idip")
    axis_nsyst = hist.axis.Integer(0, Nsyst, underflow = False, overflow = False, name = "n_syst_variations")

    helper_syst.tensor_axes = [axis_all, axis_nsyst]

    helper_stat = ROOT.wrem.muon_efficiency_veto_helper_stat[type(hist_plus_pyroot),NEtaBins,NPtEigenBins,Nsyst,Steps](
                                                                                                                           helper
                                                                                                                       )

    axis_eta_eff = hist_plus.axes[0]

    # make new versions of these axes without overflow/underflow to index the tensor
    if isinstance(axis_eta_eff, bh.axis.Regular):
        axis_eta_eff_tensor = hist.axis.Regular(axis_eta_eff.size, axis_eta_eff.edges[0], axis_eta_eff.edges[-1], name = axis_eta_eff.name, overflow = False, underflow = False)
    elif isinstance(axis_eta_eff, bh.axis.Variable):
        axis_eta_eff_tensor = hist.axis.Variable(axis_eta_eff.edges, name = axis_eta_eff.name, overflow = False, underflow = False)

    axis_ptEigen_eff_tensor = hist.axis.Integer(0, NPtEigenBins, underflow = False, overflow = False, name = "nPtEigenBins") #the independent steps for veto syst are added at the end of the axis

    axis_charge_def = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "SF charge")

    effStatTensorAxes = [axis_eta_eff_tensor, axis_ptEigen_eff_tensor, axis_charge_def]
    helper_stat.tensor_axes = effStatTensorAxes

    file_plus.Close()
    file_minus.Close()

    return helper, helper_syst, helper_stat
