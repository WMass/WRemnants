#!/usr/bin/env python3

## make MC efficiency for W vs eta-pt-ut for trigger and isolation
## Requires event yields made with the W  histmaker, with proper event selection

from wremnants.datasets.datagroups2016 import make_datagroups_2016
from wremnants import histselections as sel
#from wremnants import plot_tools,theory_tools,syst_tools
from utilities import boostHistHelpers as hh
from utilities import common, logging, output_tools, input_tools

import narf
import wremnants
from wremnants import theory_tools,syst_tools,theory_corrections
import hist

import numpy as np

import pickle
import lz4.frame

import argparse
import os
import shutil
import re

## safe batch mode
import sys
args = sys.argv[:]
sys.argv = ['-b']
import ROOT
sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

from copy import *

#sys.path.append(os.getcwd() + "/plotUtils/")
#from utility import *
from scripts.analysisTools.plotUtils.utility import *
from scripts.analysisTools.w_mass_13TeV.plotPrefitTemplatesWRemnants import plotPrefitHistograms

sys.path.append(os.getcwd())

def getBoostEff(n_pass, n_tot, integrateVar=[], rebinUt=-1):
    s = hist.tag.Slicer()
    num = n_pass.copy()
    den = n_tot.copy()
    if rebinUt > 0 and "ut" not in integrateVar:
        num = num[{"ut": s[::hist.rebin(rebinUt)]}]
        den = den[ {"ut": s[::hist.rebin(rebinUt)]}]
    if len(integrateVar):
        num = num[{v: s[::hist.sum] for v in integrateVar}]
        den = den[ {v: s[::hist.sum] for v in integrateVar}]
    eff_boost = hh.divideHists(num, den, cutoff=0.1, allowBroadcast=True, createNew=True, cutoff_val=0.0)
    vals = eff_boost.values()
    vals[vals < 0.0] = 0.0
    vals[vals > 1.0] = 1.0
    eff_boost.values()[...] = vals
    return eff_boost

def getEtaPtEff(n_pass, n_tot, getRoot=False, rootName="", rootTitle=""):
    eff_boost = getBoostEff(n_pass, n_tot, integrateVar=["ut"])
    if getRoot:
        eff_root = narf.hist_to_root(eff_boost)
        eff_root.SetName(rootName if len(rootName) else "tmp_root_eff")
        eff_root.SetTitle(rootTitle)
        return eff_root
    else:
        return eff_boost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfile", type=str, nargs=1, help="Input file with histograms (pkl.lz4 or hdf5 file)")
    parser.add_argument("outdir",   type=str, nargs=1, help="Output folder")
    parser.add_argument("-v", "--verbose", type=int, default=3, choices=[0,1,2,3,4], help="Set verbosity level with logging, the larger the more verbose");
    parser.add_argument("-n", "--baseName", type=str, help="Histogram name in the file", default="yieldsForWeffMC")
    parser.add_argument(     '--nContours', default=51, type=int, help='Number of contours in palette. Default is 51')
    parser.add_argument(     '--palette'  , default=87, type=int, help='Set palette: default is a built-in one, 55 is kRainbow')
    parser.add_argument(     '--invertPalette', action='store_true',   help='Inverte color ordering in palette')
    parser.add_argument(     '--passMt', action='store_true',   help='Measure efficiencies only for events passing mT, otherwise stay inclusive')
    parser.add_argument('-p','--processes', default=["Wmunu"], nargs='+', type=str,
                        help='Choose what processes to plot, otherwise all are done')
    parser.add_argument(     '--rebinUt', default=-1, type=int, help='If positive, rebin yields versus uT by this number before deriving efficiencies in 3D')
    args = parser.parse_args()

    logger = logging.setup_logger(os.path.basename(__file__), args.verbose)
    fname = args.inputfile[0]
    outdir = args.outdir[0]
    createPlotDirAndCopyPhp(outdir)
        
    ROOT.TH1.SetDefaultSumw2()

    canvas = ROOT.TCanvas("canvas", "", 800, 700)
    adjustSettings_CMS_lumi()
    canvas1D = ROOT.TCanvas("canvas1D", "", 800, 900)
    xAxisName = "Muon #eta"
    yAxisName = "Muon p_{T} (GeV)"

    groups = make_datagroups_2016(fname, filterGroups=args.processes, excludeGroups=None if args.processes else ['QCD'])
    datasets = groups.getNames()
    # if args.processes is not None and len(args.processes):
    #     datasets = list(filter(lambda x: x in args.processes, datasets))
    logger.debug(f"Using these processes: {datasets}")
    inputHistName = args.baseName
    groups.setNominalName(inputHistName)
    groups.loadHistsForDatagroups(inputHistName, syst="", procsToRead=datasets, applySelection=False)
    histInfo = groups.getDatagroups() # keys are same as returned by groups.getNames()
    s = hist.tag.Slicer()
    resultDict = {}
    for d in datasets:
        logger.info(f"Running on process {d}")
        hin = histInfo[d].hists[inputHistName]
        logger.debug(hin.axes)
        #logger.debug(h.sum(flow=True))

        # create copy to do other checks with uT
        hTestUt = hin.copy()
        h = hin.copy()
        
        if args.passMt:
            h = h[{"passMT" : True}]
        else:
            h = h[{"passMT" : s[::hist.sum]}]
        hiso_passTrigger = h[{"charge" : s[::hist.sum],
                              "passTrigger" : True}]
        hiso_failTrigger = h[{"charge" : s[::hist.sum],
                              "passTrigger" : False}]
        hiso_noTrigger = h[{"charge" : s[::hist.sum],
                            "passTrigger" : s[::hist.sum]}]
        
        n_iso_pass = hiso_passTrigger[{"passIso" : True}]
        n_iso_fail = hiso_passTrigger[{"passIso" : False}]
        n_iso_tot  = hiso_passTrigger[{"passIso" : s[::hist.sum]}]

        n_isoantitrig_pass = hiso_failTrigger[{"passIso" : True}]
        n_isoantitrig_fail = hiso_failTrigger[{"passIso" : False}]
        n_isoantitrig_tot  = hiso_failTrigger[{"passIso" : s[::hist.sum]}]

        n_isonotrig_pass = hiso_noTrigger[{"passIso" : True}]
        n_isonotrig_fail = hiso_noTrigger[{"passIso" : False}]
        n_isonotrig_tot  = hiso_noTrigger[{"passIso" : s[::hist.sum]}]

        # for trigger, integrate in isolation, and select a specific charge
        htriggerplus = h[{"charge" : s[complex(0,1)],
                          "passIso" : s[::hist.sum]}]
        htriggerminus = h[{"charge" : s[complex(0,-1)],
                           "passIso" : s[::hist.sum]}]

        n_triggerplus_pass = htriggerplus[{"passTrigger" : True}]
        n_triggerplus_fail = htriggerplus[{"passTrigger" : False}]
        n_triggerplus_tot  = htriggerplus[{"passTrigger" : s[::hist.sum]}]

        n_triggerminus_pass = htriggerminus[{"passTrigger" : True}]
        n_triggerminus_fail = htriggerminus[{"passTrigger" : False}]
        n_triggerminus_tot  = htriggerminus[{"passTrigger" : s[::hist.sum]}]

        # NOTE: to simplify our lives, when computing efficiencies as ratio of yields n/N and root is used
        # then the uncertainty is obtained using option B for TH1::Divide, which uses binomial uncertainties.
        # However, this implies 0 uncertainty when the efficiency is close to 0 or 1, but we won't really use
        # these uncertainties, since the 2D smoothing with spline doesn't use them.
        # otherwise can use eff = npass/(npass+nfail) and use the formula for standard propagation
        # eff_err = 1./(nTot*nTot) * std::sqrt( nP*nP* e_nF*e_nF + nF*nF * e_nP*e_nP ) with nTot = nP + nF
        # For now we use boost directly, using n/N but assuming uncorrelated n and N, so the uncertainties are maximally wrong (but we might not use them)
        
        # test, integrate uT and plot efficiencies vs eta-pt
        eff_iso = getEtaPtEff(n_iso_pass, n_iso_tot, getRoot=True, rootName=f"{d}_MC_eff_iso", rootTitle="P(iso | trig)")
        drawCorrelationPlot(eff_iso, xAxisName, xAxisName, f"MC isolation efficiency",
                            f"{eff_iso.GetName()}", plotLabel="ForceTitle", outdir=outdir,
                            smoothPlot=False, drawProfileX=False, scaleToUnitArea=False,
                            draw_both0_noLog1_onlyLog2=1, passCanvas=canvas,
                            nContours=args.nContours, palette=args.palette, invertePalette=args.invertPalette)

        eff_isonotrig = getEtaPtEff(n_isonotrig_pass, n_isonotrig_tot, getRoot=True, rootName=f"{d}_MC_eff_isonotrig", rootTitle="P(iso | ID+IP)")
        drawCorrelationPlot(eff_isonotrig, xAxisName, xAxisName, f"MC isolation efficiency",
                            f"{eff_isonotrig.GetName()}", plotLabel="ForceTitle", outdir=outdir,
                            smoothPlot=False, drawProfileX=False, scaleToUnitArea=False,
                            draw_both0_noLog1_onlyLog2=1, passCanvas=canvas,
                            nContours=args.nContours, palette=args.palette, invertePalette=args.invertPalette)

        eff_isoantitrig = getEtaPtEff(n_isoantitrig_pass, n_isoantitrig_tot, getRoot=True, rootName=f"{d}_MC_eff_isoantitrig", rootTitle="P(iso | ID+IP & fail trig)")
        drawCorrelationPlot(eff_isoantitrig, xAxisName, xAxisName, f"MC isolation efficiency",
                            f"{eff_isoantitrig.GetName()}", plotLabel="ForceTitle", outdir=outdir,
                            smoothPlot=False, drawProfileX=False, scaleToUnitArea=False,
                            draw_both0_noLog1_onlyLog2=1, passCanvas=canvas,
                            nContours=args.nContours, palette=args.palette, invertePalette=args.invertPalette)

        eff_triggerplus = getEtaPtEff(n_triggerplus_pass, n_triggerplus_tot, getRoot=True, rootName=f"{d}_MC_eff_triggerplus", rootTitle="P(trig | ID+IP)")
        drawCorrelationPlot(eff_triggerplus, xAxisName, xAxisName, f"MC trigger plus efficiency",
                            f"{eff_triggerplus.GetName()}", plotLabel="ForceTitle", outdir=outdir,
                            smoothPlot=False, drawProfileX=False, scaleToUnitArea=False,
                            draw_both0_noLog1_onlyLog2=1, passCanvas=canvas,
                            nContours=args.nContours, palette=args.palette, invertePalette=args.invertPalette)

        eff_triggerminus = getEtaPtEff(n_triggerminus_pass, n_triggerminus_tot, getRoot=True, rootName=f"{d}_MC_eff_triggerminus", rootTitle="P(trig | ID+IP)")
        drawCorrelationPlot(eff_triggerminus, xAxisName, xAxisName, f"MC trigger minus efficiency",
                            f"{eff_triggerminus.GetName()}", plotLabel="ForceTitle", outdir=outdir,
                            smoothPlot=False, drawProfileX=False, scaleToUnitArea=False,
                            draw_both0_noLog1_onlyLog2=1, passCanvas=canvas,
                            nContours=args.nContours, palette=args.palette, invertePalette=args.invertPalette)

        eff_iso_3D = getBoostEff(n_iso_pass, n_iso_tot, rebinUt=args.rebinUt)
        eff_isonotrig_3D = getBoostEff(n_isonotrig_pass, n_isonotrig_tot, rebinUt=args.rebinUt)
        eff_isoantitrig_3D = getBoostEff(n_isoantitrig_pass, n_isoantitrig_tot, rebinUt=args.rebinUt)
        eff_triggerplus_3D = getBoostEff(n_triggerplus_pass, n_triggerplus_tot, rebinUt=args.rebinUt)
        eff_triggerminus_3D = getBoostEff(n_triggerminus_pass, n_triggerminus_tot, rebinUt=args.rebinUt)

        resultDict[f"{d}_MC_eff_iso_etaptut"] = eff_iso_3D
        resultDict[f"{d}_MC_eff_isonotrig_etaptut"] = eff_isonotrig_3D
        resultDict[f"{d}_MC_eff_isoantitrig_etaptut"] = eff_isoantitrig_3D
        resultDict[f"{d}_MC_eff_triggerplus_etaptut"] = eff_triggerplus_3D
        resultDict[f"{d}_MC_eff_triggerminus_etaptut"] = eff_triggerminus_3D

        # plot uT distribution for each charge, vs mT (pass or fail, or inclusive), integrate anything else
        # do it for events passing trigger ans isolation
        nPtBins = hTestUt.axes["pt"].size
        hTestUt = hTestUt[{"eta" : s[::hist.sum],
                           "pt" :  s[0:nPtBins:hist.sum], # would s[::hist.sum] sum overflow pt bins?
                           "passTrigger" : True,
                           "passIso" : True}]
        for charge in [-1, 1]:
            chargeStr = "plus" if charge > 0 else "minus"
            hut = hTestUt[{"charge" : s[complex(0,charge)]}]
            hut_allMt = hut[{"passMT": s[::hist.sum]}]
            hut_passMt = hut[{"passMT": True}]
            hut_failMt = hut[{"passMT": False}]
            allHists = [hut_allMt, hut_passMt, hut_failMt]
            allHistsRoot = []
            hNamesRoot = ["allMT", "passMT", "failMT"]
            for ih,htmp in enumerate(allHists):
                hroot = narf.hist_to_root(htmp)
                hroot.Scale(1./hroot.Integral()) # normalize to unit area to get shape
                hNamesRoot[ih] += f"_{chargeStr}_{d}"
                hroot.SetName(hNamesRoot[ih])
                allHistsRoot.append(hroot)                
            drawNTH1(allHistsRoot, hNamesRoot, "Projected recoil u_{T} (GeV)", "Normalized units", f"ut_{d}_{chargeStr}",
                     outdir, draw_both0_noLog1_onlyLog2=1, topMargin=0.05, labelRatioTmp="X / incl.::0.5,1.5",
                     legendCoords="0.2,0.8,0.77,0.92;1", passCanvas=canvas1D, skipLumi=True,
                     onlyLineColor=True, useLineFirstHistogram=True)
                
    postfix = ""
    toAppend = []
    if args.passMt:
        toAppend.append("passMt")
    if args.rebinUt > 0:
        toAppend.append(f"rebinUt{args.rebinUt}")     
    postfix = "_".join(toAppend)
    if len(postfix):
        postfix = "_" + postfix

    resultDict.update({"meta_info" : output_tools.metaInfoDict(args=args)})    
        
    outfile = outdir + f"efficiencies3D{postfix}.pkl.lz4"
    logger.info(f"Going to store 3D histograms {resultDict.keys()} in file {outfile}")
    with lz4.frame.open(outfile, 'wb') as f:
        pickle.dump(resultDict, f, protocol=pickle.HIGHEST_PROTOCOL)

    outfileRoot = outdir + f"efficiencies2D{postfix}.root"
    logger.info(f"Going to store 2D root histograms in file {outfileRoot}")
    rf = safeOpenFile(outfileRoot, mode="RECREATE")
    eff_iso.Write()
    eff_isonotrig.Write()
    eff_isoantitrig.Write()
    eff_triggerplus.Write()
    eff_triggerminus.Write()
    rf.Close()
