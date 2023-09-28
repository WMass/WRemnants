
import sys,array,math,os,copy,json
import numpy as np

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

import functions
import plotter

import lz4.frame
import pickle
import narf
import numpy as np

from wremnants.datasets import datagroups




if __name__ == "__main__":

    met = "DeepMETReso" # PFMET, RawPFMET DeepMETReso
    flavor = "mumu" # mu, e, mumu, ee
    lowPU = False

    ####################################################################
    if lowPU:
        recoil_qTbins = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 70, 5)) + list(range(70, 120, 10)) + [120, 150, 300]
        
        datagroups = Datagroups("lowPU_%s_%s_%s.pkl.lz4" % (flavor, met, pdf))
        sig = "Zmumu"
        data = "SingleMuon"
        bkgs = ['EWK', 'Top'] 

        h_sig = narf.hist_to_root(functions.readBoostHistProc(datagroups, "qT", [sig]))
        h_data = narf.hist_to_root(functions.readBoostHistProc(datagroups, "qT", [data]))
        h_bkgs = narf.hist_to_root(functions.readBoostHistProc(datagroups, "qT", bkgs))
        
        h_sig = functions.Rebin(h_sig, recoil_qTbins)
        h_data = functions.Rebin(h_data, recoil_qTbins)
        h_bkgs = functions.Rebin(h_bkgs, recoil_qTbins)
        h_data.Add(h_bkgs, -1)
        
        # normalize histograms, to preserve total normalization
        h_sig.Scale(1./h_sig.Integral())
        h_data.Scale(1./h_data.Integral())
        
        weights = []
        for i in range(1, h_sig.GetNbinsX()+1):
            w = h_data.GetBinContent(i)/h_sig.GetBinContent(i)
            weights.append(w)
            print(h_data.GetBinLowEdge(i), h_data.GetBinLowEdge(i+1), w)

        outDict = {}
        outDict['qT_bins'] = recoil_qTbins
        outDict['weights'] = weights
        functions.writeJSON("wremnants/data/recoil/lowPU/%s_%s/qT_reweighting.json" % (flavor, met), outDict)

    else:
        recoil_qTbins = list(range(0, 50, 1)) + list(range(50, 80, 2)) + list(range(80, 120, 5)) + list(range(120, 160, 10)) + list(range(160, 300, 20)) + [500]
        recoil_qTbins = list(range(0, 201, 1))
        
        #groups = datagroups.Datagroups(f"mz_wlike_with_mu_eta_pt_{met}.hdf5")
        groups = datagroups.Datagroups(f"mz_wlike_with_mu_eta_pt_DeepMETReso_nnpdf31_noPTcut.hdf5")
        
        sig = "Zmumu"
        data = "Data"
        bkgs = ['Ztautau', 'Other']

        h_sig = narf.hist_to_root(functions.readBoostHistProc(groups, "qT", [sig]))
        h_data = narf.hist_to_root(functions.readBoostHistProc(groups, "qT", [data]))
        h_bkgs = narf.hist_to_root(functions.readBoostHistProc(groups, "qT", bkgs))
        
        h_sig = functions.Rebin(h_sig, recoil_qTbins)
        h_data = functions.Rebin(h_data, recoil_qTbins)
        h_bkgs = functions.Rebin(h_bkgs, recoil_qTbins)
        h_data.Add(h_bkgs, -1)
        
        # normalize histograms, to preserve total normalization
        h_sig.Scale(1./h_sig.Integral())
        h_data.Scale(1./h_data.Integral())
        
        weights = []
        for i in range(1, h_sig.GetNbinsX()+1):
            w = h_data.GetBinContent(i)/h_sig.GetBinContent(i) if h_sig.GetBinContent(i) > 0 and h_data.GetBinContent(i) else 1
            weights.append(w)
            print(h_data.GetBinLowEdge(i), h_data.GetBinLowEdge(i+1), w)

        outDict = {}
        outDict['qT_bins'] = recoil_qTbins
        outDict['weights'] = weights
        functions.writeJSON("wremnants-data/data/recoil/highPU/%s_%s/qT_reweighting.json" % (flavor, met), outDict)


