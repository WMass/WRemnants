#!/usr/bin/env python3

# python w-mass-13TeV/makeMtEfficiency.py plots/testNanoAOD/WmassPlots/histograms_isoChargeMtPtEta_fakeRegion_deepMET_NanoV9/checkPlots_lowIso_plus/postVFP//plots_fakerate.root --mt mt_MET mt_deepMET --mtleg PFMET deepMET --signal Wmunu_plus;   python w-mass-13TeV/makeMtEfficiency.py plots/testNanoAOD/WmassPlots/histograms_isoChargeMtPtEta_fakeRegion_deepMET_NanoV9/checkPlots_highIso_plus/postVFP//plots_fakerate.root --mt mt_MET mt_deepMET --mtleg PFMET deepMET --signal Wmunu_plus;   python w-mass-13TeV/makeMtEfficiency.py plots/testNanoAOD/WmassPlots/histograms_isoChargeMtPtEta_fakeRegion_deepMET_NanoV9/checkPlots_lowIso_plus_1morejet/postVFP//plots_fakerate.root --mt mt_MET mt_deepMET --mtleg PFMET deepMET --signal Wmunu_plus --invertmt;   python w-mass-13TeV/makeMtEfficiency.py plots/testNanoAOD/WmassPlots/histograms_isoChargeMtPtEta_fakeRegion_deepMET_NanoV9/checkPlots_highIso_plus_1morejet/postVFP//plots_fakerate.root --mt mt_MET mt_deepMET --mtleg PFMET deepMET --signal Wmunu_plus --invertmt

# take mT distributions and check S/B for different mT cuts (ideally done for low isolation region without mT cuts)
import os, os.path, re, array, math
import time
import argparse
import shutil
import ctypes

import narf
import narf.fitutils
import wremnants
import hist
import lz4.frame, pickle
from wremnants.datasets.datagroups2016 import make_datagroups_2016
from wremnants import histselections as sel

import numpy as np

from utilities import boostHistHelpers as hh, common, logging

## safe batch mode                                 
import sys
args = sys.argv[:]
sys.argv = ['-b']
import ROOT
sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

from copy import *

from scripts.analysisTools.plotUtils.utility import *
from scripts.analysisTools.tests.cropNegativeTemplateBins import cropNegativeContent

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5fileDeepMet", type=str, nargs=1, help="Input file with histograms for deepMet")
    parser.add_argument("hdf5filePFMet", type=str, nargs=1, help="Input file with histograms for PFMet")
    parser.add_argument("outputfolder",  type=str, nargs=1)
    
    parser.add_argument("--mt", type=str, nargs="+", default=["mt_MET"], help="mt histograms to use")
    parser.add_argument("--mtleg", type=str, nargs="+", default=["PFMET"], help="mt name for legend")
    parser.add_argument("--signal", type=str, default="Wmunu_plus", help="Process to be used for signal efficiency")
    parser.add_argument("--postfix", type=str, default=None, help="Postfix for output folder")
    parser.add_argument(     '--invertmt', default=False , action='store_true',   help='Invert mt cut to derive efficiency (default is mT>threshold)')
    args = parser.parse_args()
           
    fname = args.rootfile[0]
    postfix = f"_{args.postfix}" if args.postfix else ""
    outdir = f"{args.outputfolder[0]}/mtCutEfficiency{postfix}/"
    createPlotDirAndCopyPhp(outdir)
    
    ROOT.TH1.SetDefaultSumw2()
    
    nomihists = {}
    hqcd = {}
    hother = {}

def getHistograms(inputfile):
    
    groups = make_datagroups_2016(inputfile, applySelection=False)
    datasets = groups.getNames() # this has all the original defined groups
    datasetsNoQCD = list(filter(lambda x: x != "QCD", datasets)) # exclude QCD MC if present
    logger.info(f"All original datasets available {datasets}")
    inputHistName = "mTStudyForFakes"
    groups.setNominalName(inputHistName)
    groups.loadHistsForDatagroups(inputHistName, syst="", procsToRead=datasets, applySelection=False)
    histInfo = groups.getDatagroups() # keys are same as returned by groups.getNames() 
    rootHists = {d: None for d in datasetsNoQCD}
    s = hist.tag.Slicer()
    for d in datasets:
        #print(d)
        hnarf = histInfo[d].hists[inputHistName]
        # start integrating
        hnarf = hnarf[{"eta" : s[::hist.sum],
                       "pt" : s[0:hist.len:hist.sum], # make sure to remove overflow here, we often don't cut on pt 
                       "hasJets": s[::hist.sum]
                
    
    # read histograms
    infile = safeOpenFile(fname)
    for mt in args.mt:
        nomihists[mt] = {}
        hqcd[mt] = None
        hother[mt] = None
        for k in infile.GetListOfKeys():
            name = k.GetName()
            if any(x in name for x in ["stack", "canvas"]):
                continue
            if mt not in name:
                continue
            proc = name.split(f"{mt}_")[1]
            print(f"--> Reading {name}: process {proc}")
            nomihists[mt][proc] = safeGetObject(infile, name, detach=True)
        hqcd[mt] = copy.deepcopy(nomihists[mt]["data"].Clone(f"{mt}_data_fakes"))
        hqcd[mt].Add(nomihists[mt]["background"],-1.0) # "background includes everything but data
        cropNegativeContent(hqcd[mt],cropValue=0.0) # QCD can get negative from data-MC, set negative bins to 0 not to bias efficiency
        nomihists[mt]["data_fakes"] = hqcd[mt]
        hother[mt] = copy.deepcopy(nomihists[mt]["background"].Clone(f"{mt}_other"))
        hother[mt].Add(nomihists[mt][args.signal],-1.0)
        nomihists[mt]["other"] = hother[mt]
    infile.Close()

    cutEff = {}
    binStep = 2
    nBinsTot = nomihists[mt]["data"].GetNbinsX()
    # cut only until mT ~ 70, no meaning in cutting further
    maxImt = 1
    for imt in range(1, nBinsTot+1, binStep):
        if nomihists[mt]["data"].GetBinLowEdge(imt) > 70:
            maxImt = imt
            break
    mtcutbins = range(1, imt, binStep)
    grList = []
    grLeg = []
    grLeg2 = []
    grColors = [ROOT.kBlack, ROOT.kRed+2, ROOT.kBlue, ROOT.kGreen+2]
    err = ctypes.c_double(0.) # or ROOT.Double(0.0) (deprecated in newer ROOT versions), just to pass a reference and use TH1::IntegralAndError()

    procLegMap = {"data_fakes" : "Multijet",
                  "Wmunu_plus" : "W^{+}#rightarrow#mu#nu",
                  "Wmunu_minus": "W^{-}#rightarrow#mu#nu",
                  "other"      : "Other bkg"
    }

    for imt,mt in enumerate(args.mt):
        cutEff[mt] = {"data_fakes" : ROOT.TGraphErrors(),
                      args.signal  : ROOT.TGraphErrors(),
                      "other"      : ROOT.TGraphErrors()
        }
        for ip,proc in enumerate(cutEff[mt].keys()):
            procLeg = proc
            if proc in procLegMap.keys():
                procLeg = procLegMap[proc]
            for mtcutbin in mtcutbins:
                if args.invertmt:
                    eff = nomihists[mt][proc].IntegralAndError(0, mtcutbin, err)
                    eff /= nomihists[mt][proc].IntegralAndError(0,nBinsTot+1, err)
                else:
                    eff = nomihists[mt][proc].IntegralAndError(mtcutbin, nBinsTot+1, err)
                    eff /= nomihists[mt][proc].IntegralAndError(0,nBinsTot+1, err)
                cutEff[mt][proc].SetPoint(cutEff[mt][proc].GetN(), nomihists[mt][proc].GetBinLowEdge(mtcutbin), eff)
            cutEff[mt][proc].SetLineColor(grColors[ip])    
            cutEff[mt][proc].SetMarkerColor(grColors[ip])    
            cutEff[mt][proc].SetLineWidth(2)    
            cutEff[mt][proc].SetLineStyle(imt + 1)
            cutEff[mt][proc].SetMarkerStyle((20 + imt) if len(args.mt) > 2 else (20 + imt + 4*imt)) # for two cases it should make full or open marker    
            grList.append(cutEff[mt][proc])
            grLeg.append(f"{args.mtleg[imt]} {procLeg}")

    adjustSettings_CMS_lumi()
    canvas1D = ROOT.TCanvas("canvas1D", "", 900, 800)
    adjustSettings_CMS_lumi()
    signCut = "<" if args.invertmt else ">"
    
    drawGraphCMS(grList, "Transverse mass threshold (GeV)", f"Efficiency for m_{{T}} {signCut} threshold::0.0,1.5", "mtCutEfficiency", outdir, grLeg,
                 legendCoords="0.15,0.7,0.9,0.94;2", passCanvas=canvas1D, graphDrawStyle="pl", legEntryStyle="PL", useOriginalGraphStyle=True)

    # get S/B
    for den in [args.signal, "background"]:
    
        grSoverB = {}
        maxy = 0.0
        miny = 10000.0
        for imt,mt in enumerate(args.mt):    
            grSoverB[mt] = ROOT.TGraphErrors()
            sob = 1.0
            for mtcutbin in mtcutbins:
                if args.invertmt:
                    sob = nomihists[mt]["data_fakes"].IntegralAndError(0, mtcutbin, err)
                    sob /= nomihists[mt][den].IntegralAndError(0,mtcutbin, err)
                else:
                    sob = nomihists[mt]["data_fakes"].IntegralAndError(mtcutbin, nBinsTot+1, err)
                    sob /= nomihists[mt][den].IntegralAndError(mtcutbin, nBinsTot+1, err)
                grSoverB[mt].SetPoint(grSoverB[mt].GetN(), nomihists[mt]["background"].GetBinLowEdge(mtcutbin), sob)
                maxy = max(maxy, sob)
                miny = min(miny, sob)
            grSoverB[mt].SetLineColor(grColors[imt])    
            grSoverB[mt].SetMarkerColor(grColors[imt])    
            grSoverB[mt].SetLineWidth(2)    
            grSoverB[mt].SetMarkerStyle((20 + imt) if len(args.mt) > 2 else (20 + imt + 4*imt))
            grLeg2.append(f"{args.mtleg[imt]}")


        diff = maxy - miny
        miny = miny - 0.1 * diff
        maxy = maxy + 0.1 * diff
        denLabel =  "EW" if den == "background" else den
        drawGraphCMS([grSoverB[imt] for imt in grSoverB.keys()], "Transverse mass threshold (GeV)", f"QCD/{denLabel} yields for m_{{T}} {signCut} threshold::{miny},{maxy}",
                     f"QCDover{denLabel}_yieldRatio", outdir, grLeg2, legendCoords="0.6,0.48,0.9,0.64;1",
                     passCanvas=canvas1D, graphDrawStyle="pl", legEntryStyle="PL", useOriginalGraphStyle=True)
