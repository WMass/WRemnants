#!/usr/bin/env python3

from wremnants.datasets.datagroups2016 import make_datagroups_2016
from wremnants import histselections as sel
#from wremnants import plot_tools,theory_tools,syst_tools
from utilities import boostHistHelpers as hh,common, logging

import narf
import wremnants
from wremnants import theory_tools,syst_tools,theory_corrections
import hist

import numpy as np
from utilities import input_tools

import lz4.frame

import argparse
import os
import shutil
import logging
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

from scripts.analysisTools.plotUtils.utility import *

sys.path.append(os.getcwd())

# should really avoid having numbers by hand, but this started with less then 10 in all graphs together :) 

# true data uncertainty with no BBB
def getTrueDataUncertainty(yvals_data_N, yvals_data_NoverAlpha, alpha=2):
    yvals_true  = [0.0 for i in range(len(yvals_data_N))]
    for iy,y in enumerate(yvals_data_N):
        r = yvals_data_NoverAlpha[iy]/y
        denFact = alpha * r * r - 1.0
        if denFact <= 0.0:
            print(f"Error for x = {xvals[iy]}: r = {r} is too small for alpha = {alpha} (must be >= 1./sqrt({alpha}) )")
            fact = 0
        else:
            fact = r * np.sqrt(alpha - 1)/ np.sqrt(denFact)
        yvals_true[iy] = fact * y
    return yvals_true

# true data uncertainty with BBB
def getTrueTotalUncertainty(yvals_data_N, yvals_data_NoverAlpha, alpha=2, rho=1):
    yvals_true  = [0.0 for i in range(len(yvals_data_N))]
    for iy,y in enumerate(yvals_data_N):
        r = yvals_data_NoverAlpha[iy]/y
        denFact2 = alpha * r * r * (1+rho) / (1+alpha*rho) - 1.0
        if denFact2 < 0.0:
            print(f"Error for x = {xvals[iy]}: denFact2 = {denFact2} is too small for alpha = {alpha} and rho = {rho}")
            fact = 0
        else:
            fact = np.sqrt((alpha - 1)/ (1.0 + alpha * rho)) * r / np.sqrt(denFact2)
        yvals_true[iy] = fact * y
    return yvals_true

def runWithBkg():
    
    outdir = "/eos/user/m/mciprian/www/WMassAnalysis/fromMyWremnants/fitResults/theoryAgnostic/v2/WMass_pt_eta_statOnly/impactScalingWithRecoPtBinning/"
    createPlotDirAndCopyPhp(outdir)

    adjustSettings_CMS_lumi()
    canvas1D = ROOT.TCanvas("canvas1D", "", 900, 800)

    xvals = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
    yvals_gdata = [ 6.6, 12.2, 16.4, 21.9, 26.2, 30.8]
    yvals_gmc   = [ 9.3, 12.8, 17.0, 22.7, 27.2, 31.9]
    yvals_gtot  = [11.4, 17.7, 24.0, 31.6, 37.7, 44.3]

    gdata = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_gdata))
    gmc   = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_gmc))
    gtot  = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_gtot))
    glist = [gtot, gdata, gmc]
    leg = ["Total", "Data stat", "MC stat"]
    
    drawGraphCMS(glist, "p_{T} bin width (GeV)", "Impact on m_{W} (MeV)::0,50", "impacts_mw_ptRecoBinWidth",
                 outdir, leg,
                 legendCoords="0.2,0.7,0.6,0.88;1", vecMCcolors=[ROOT.kBlack, ROOT.kRed, ROOT.kBlue],
                 passCanvas=canvas1D, graphDrawStyle="pl", legEntryStyle="PL",
                 skipLumi=True)

    xvals = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
    yvals_halfMC_gdata = [ 2.11, 9.14, 12.5, 17.4, 21.4, 25.9]
    yvals_halfMC_gmc   = [ 9.17, 13.1, 17.8, 24.7, 30.4, 36.6]
    yvals_halfMC_gtot  = [ 9.41, 15.9, 21.7, 30.3, 37.1, 44.8]
    
    gdata_halfMC = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_halfMC_gdata))
    gmc_halfMC   = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_halfMC_gmc))
    gtot_halfMC  = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_halfMC_gtot))
    glist_halfMC = [gtot_halfMC, gdata_halfMC, gmc_halfMC]
    tag = "(1/2 MC stat)"
    leg_halfMC = [f"{l} {tag}" for l in leg]
    
    drawGraphCMS(glist_halfMC, "p_{T} bin width (GeV)", "Impact on m_{W} (MeV)::0,50", "impacts_mw_ptRecoBinWidth_halfMCstat",
                 outdir, leg_halfMC,
                 legendCoords="0.2,0.7,0.6,0.88;1", vecMCcolors=[ROOT.kBlack, ROOT.kRed, ROOT.kBlue],
                 passCanvas=canvas1D, graphDrawStyle="pl", legEntryStyle="PL",
                 skipLumi=True)

    # together
    drawGraphCMS([*glist, *glist_halfMC], "p_{T} bin width (GeV)", "Impact on m_{W} (MeV)::0,50",
                 "impacts_mw_ptRecoBinWidth_overlaid",
                 outdir, [*leg, *leg_halfMC],
                 legendCoords="0.55,0.14,0.95,0.44;1",
                 vecMCcolors=[ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kBlack, ROOT.kRed, ROOT.kBlue],
                 vecLineStyle=[1,1,1,2,2,2],
                 vecMarkerStyle=[20,20,20,25,25,25],
                 passCanvas=canvas1D, graphDrawStyle="pl", legEntryStyle="PL",
                 skipLumi=True)

    xvals_traditional =       [  0.1, 0.25, 0.5,  1.0,  1.5,  2.0]
    yvals_traditional_gdata = [ 2.27, 2.37, 2.39, 2.40, 2.41, 2.42]
    yvals_traditional_gmc   = [ 2.50, 2.49, 2.51, 2.52, 2.52, 2.53]
    yvals_traditional_gtot  = [ 7.69, 9.01, 9.88, 10.9, 12.2, 9.18]
    
    gdata_traditional = ROOT.TGraph(len(xvals_traditional), array('d', xvals_traditional), array('d', yvals_traditional_gdata))
    gmc_traditional   = ROOT.TGraph(len(xvals_traditional), array('d', xvals_traditional), array('d', yvals_traditional_gmc))
    gtot_traditional  = ROOT.TGraph(len(xvals_traditional), array('d', xvals_traditional), array('d', yvals_traditional_gtot))
    glist_traditional = [gtot_traditional, gdata_traditional, gmc_traditional]
    tag = ""
    leg_traditional = [f"{l} {tag}" for l in leg]
    
    drawGraphCMS(glist_traditional, "p_{T} bin width (GeV)", "Impact on m_{W} (MeV)::0,15", "impacts_mw_ptRecoBinWidth_traditional",
                 outdir, leg_traditional,
                 legendCoords="0.2,0.7,0.6,0.88;1", vecMCcolors=[ROOT.kBlack, ROOT.kRed, ROOT.kBlue],
                 passCanvas=canvas1D, graphDrawStyle="pl", legEntryStyle="PL",
                 skipLumi=True)

def runSignalOnly():
    outdir = "/eos/user/m/mciprian/www/WMassAnalysis/fromMyWremnants/fitResults/theoryAgnostic/v2/signalOnly/WMass_pt_eta_statOnly/impactScalingWithRecoPtBinning/"
    createPlotDirAndCopyPhp(outdir)

    adjustSettings_CMS_lumi()
    canvas1D = ROOT.TCanvas("canvas1D", "", 900, 800)

    xvals =       [ 0.1,  0.2,   0.4,   0.6,   1.0,   1.2,   1.5,   2.0]
    yvals_gdata = [ 7.86, 10.70, 14.16, 16.59, 20.51, 22.21, 24.67, 29.40]
    yvals_gmc   = [ 7.56, 10.31, 13.66, 16.05, 19.88, 21.54, 23.95, 28.55]
    yvals_gtot  = [10.9,  14.9,  19.7,  23.1,  28.6 , 30.9,  34.4,  41.0]

    gdata = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_gdata))
    gmc   = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_gmc))
    gtot  = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_gtot))
    glist = [gtot, gdata, gmc]
    leg = ["Total", "Data stat", "MC stat"]
    
    drawGraphCMS(glist, "p_{T} bin width (GeV)", "Impact on m_{W} (MeV)::0,50", "impacts_mw_ptRecoBinWidth_signalOnly",
                 outdir, leg,
                 legendCoords="0.2,0.7,0.6,0.88;1", vecMCcolors=[ROOT.kBlack, ROOT.kRed, ROOT.kBlue],
                 passCanvas=canvas1D, graphDrawStyle="pl", legEntryStyle="PL",
                 skipLumi=True)

    xvals =              [ 0.1,  0.2,   0.4,   0.6,   1.0,   1.2,   1.5,   2.0]
    yvals_halfMC_gdata = [ 5.72,  7.94, 10.85, 12.94, 16.50, 18.22, 20.30, 24.78]
    yvals_halfMC_gmc   = [ 7.76, 10.79, 14.75, 17.63, 22.52, 24.89, 27.73, 33.90]
    yvals_halfMC_gtot  = [ 9.64, 13.4,  18.3,  21.9,  27.9,  30.8,  34.3,  42.0]
    
    gdata_halfMC = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_halfMC_gdata))
    gmc_halfMC   = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_halfMC_gmc))
    gtot_halfMC  = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_halfMC_gtot))
    glist_halfMC = [gtot_halfMC, gdata_halfMC, gmc_halfMC]
    tag_halfMC = "(1/2 MC stat)"
    leg_halfMC = [f"{l} {tag_halfMC}" for l in leg]

    yvals_halfMCother_gdata = [ 5.701, 7.905, 10.832, 12.942, 16.463, 17.954, 20.133, 23.537]
    yvals_halfMCother_gtot  = [ 9.62,  13.35, 18.31,  21.91,  27.92,  30.47,  34.16,  40.00]
    gdata_halfMCother = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_halfMCother_gdata))
    gtot_halfMCother  = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_halfMCother_gtot))
    tag_halfMCother = "(1/2 MC stat other)"
    leg_halfMCother = [f"{l} {tag_halfMCother}" for l in leg]
    
    drawGraphCMS(glist_halfMC, "p_{T} bin width (GeV)", "Impact on m_{W} (MeV)::0,50", "impacts_mw_ptRecoBinWidth_halfMCstat",
                 outdir, leg_halfMC,
                 legendCoords="0.2,0.7,0.6,0.88;1",
                 vecMCcolors=[ROOT.kBlack, ROOT.kRed, ROOT.kBlue],
                 vecLineStyle=[2,2,2],
                 vecMarkerStyle=[25,25,25],
                 passCanvas=canvas1D, graphDrawStyle="pl", legEntryStyle="PL",
                 skipLumi=True)

    # together full and 1/2 stat
    drawGraphCMS([*glist, *glist_halfMC], "p_{T} bin width (GeV)", "Impact on m_{W} (MeV)::0,50",
                 "impacts_mw_ptRecoBinWidth_overlaid",
                 outdir, [*leg, *leg_halfMC],
                 legendCoords="0.55,0.14,0.95,0.44;1",
                 vecMCcolors=[ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kBlack, ROOT.kRed, ROOT.kBlue],
                 vecLineStyle=[1,1,1,2,2,2],
                 vecMarkerStyle=[20,20,20,25,25,25],
                 passCanvas=canvas1D, graphDrawStyle="pl", legEntryStyle="PL",
                 skipLumi=True)

    yvals_1over4MC_gdata = [ 4.142, 5.805, 8.099, 9.784, 12.966, 14.361, 16.029, 20.351]
    yvals_1over4MC_gtot = [ 8.9, 12.5, 17.5, 21.1, 28.0, 31.0, 34.6, 44.0]
    gdata_1over4MC = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_1over4MC_gdata))
    gtot_1over4MC = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_1over4MC_gtot))
    tag_1over4MC = "(1/4 MC stat)"
    leg_1over4MC = [f"{l} {tag_1over4MC}" for l in leg]

    yvals_other1over4MC_gdata = [4.122, 5.798, 8.147, 9.884, 12.809, 14.061, 15.813, 19.158]
    yvals_other1over4MC_gtot = [ 8.91,  12.54, 17.63, 21.41, 27.78,  30.49,  34.39, 41.68]
    gdata_other1over4MC = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_other1over4MC_gdata))
    gtot_other1over4MC = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_other1over4MC_gtot))
    tag_other1over4MC = "(1/4 MC stat other)"
    leg_other1over4MC = [f"{l} {tag_other1over4MC}" for l in leg]

    # compute true sigma from Lorenzo's calculations, using gdata
    yvals_true_gdata = getTrueDataUncertainty(yvals_gdata, yvals_halfMC_gdata)
    gdata_true = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_true_gdata))
    # same but other half of MC, in case the stat was not equally divided
    yvals_true_gdata_otherHalf = getTrueDataUncertainty(yvals_gdata, yvals_halfMCother_gdata)
    gdata_true_otherHalf = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_true_gdata_otherHalf))

    yvals_true_gdata_fromHalf = getTrueDataUncertainty(yvals_halfMC_gdata, yvals_1over4MC_gdata)
    gdata_true_fromHalf = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_true_gdata_fromHalf))
    # same but from other half (might also use another fourth, to be done
    # here, because of how this half and fourth were made, they are really independent events, before they were subsets
    yvals_true_gdata_fromOtherHalf = getTrueDataUncertainty(yvals_halfMCother_gdata, yvals_1over4MC_gdata)
    gdata_true_fromOtherHalf = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_true_gdata_fromOtherHalf))
    # another case
    yvals_true_gdata_fromHalfOtherFourth = getTrueDataUncertainty(yvals_halfMC_gdata, yvals_other1over4MC_gdata)
    gdata_true_fromHalfOtherFourth = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_true_gdata_fromHalfOtherFourth))

    # now go from 1/4 to 1 directly
    yvals_true_gdata_FourthToFull = getTrueDataUncertainty(yvals_gdata, yvals_1over4MC_gdata, alpha=4)
    gdata_true_FourthToFull = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_true_gdata_FourthToFull))

    yvals_true_gdata_otherFourthToFull = getTrueDataUncertainty(yvals_gdata, yvals_other1over4MC_gdata, alpha=4)
    gdata_true_otherFourthToFull = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_true_gdata_otherFourthToFull))
    ###
    
    drawGraphCMS([gdata_true, gdata, gdata_halfMC],
                 "p_{T} bin width (GeV)", "Impact on m_{W} (MeV)::0,50",
                 "impacts_mw_ptRecoBinWidth_trueSensitivityData",
                 outdir,
                 ["Data stat true", "Data stat (full MC stat)", f"Data stat {tag_halfMC}"],
                 legendCoords="0.2,0.7,0.6,0.88;1",
                 vecMCcolors=[ROOT.kBlack, ROOT.kRed, ROOT.kBlue],
                 vecLineStyle=[1,1,1],
                 vecMarkerStyle=[20,20,20],
                 passCanvas=canvas1D, graphDrawStyle="pl", legEntryStyle="PL",
                 skipLumi=True)

    drawGraphCMS([gdata_true, gdata_true_otherHalf, gdata, gdata_halfMC, gdata_halfMCother],
                 "p_{T} bin width (GeV)", "Impact on m_{W} (MeV)::0,50",
                 "impacts_mw_ptRecoBinWidth_trueSensitivityData_testOtherHalfMC",
                 outdir,
                 ["Data stat true", "Data stat true (other)", "Data stat (full MC stat)", f"Data stat {tag_halfMC}", f"Data stat {tag_halfMCother}"],
                 legendCoords="0.2,0.68,0.6,0.93;1",
                 vecMCcolors=[ROOT.kBlack, ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kBlue],
                 vecLineStyle=[1,2,1,1,2],
                 vecMarkerStyle=[20,25,20,20,25],
                 passCanvas=canvas1D, graphDrawStyle="pl", legEntryStyle="PL",
                 skipLumi=True)

    drawGraphCMS([gdata, gdata_halfMC, gdata_halfMCother, gdata_1over4MC, gdata_other1over4MC],
                 "p_{T} bin width (GeV)", "Impact on m_{W} (MeV)::0,50",
                 "impacts_mw_ptRecoBinWidth_testAllData",
                 outdir,
                 ["Data stat (full MC stat)", f"Data stat {tag_halfMC}", f"Data stat {tag_halfMCother}", f"Data stat {tag_1over4MC}", f"Data stat {tag_other1over4MC}"],
                 legendCoords="0.2,0.68,0.6,0.93;1",
                 vecMCcolors=[ROOT.kBlack, ROOT.kRed, ROOT.kRed, ROOT.kBlue, ROOT.kBlue],
                 vecLineStyle=[1,1,2,1,2],
                 vecMarkerStyle=[20,20,25,20,25],
                 passCanvas=canvas1D, graphDrawStyle="pl", legEntryStyle="PL",
                 skipLumi=True)
    
    drawGraphCMS([gtot, gtot_halfMC, gtot_halfMCother, gtot_1over4MC, gtot_other1over4MC],
                 "p_{T} bin width (GeV)", "Impact on m_{W} (MeV)::0,50",
                 "impacts_mw_ptRecoBinWidth_totalCompareFractionMC",
                 outdir,
                 ["Total (full MC stat)", f"Total {tag_halfMC}", f"Total {tag_halfMCother}", f"Total {tag_1over4MC}", f"Total {tag_other1over4MC}"],
                 legendCoords="0.48,0.2,0.92,0.5;1",
                 vecMCcolors=[ROOT.kBlack, ROOT.kRed, ROOT.kRed, ROOT.kBlue, ROOT.kBlue],
                 vecLineStyle=[1,1,2,1,2],
                 vecMarkerStyle=[20,20,25,20,25],
                 passCanvas=canvas1D, graphDrawStyle="pl", legEntryStyle="PL",
                 skipLumi=True)
    
    drawGraphCMS([gdata_true, gdata_true_fromHalf, gdata, gdata_halfMC, gdata_1over4MC],
                 "p_{T} bin width (GeV)", "Impact on m_{W} (MeV)::0,50",
                 "impacts_mw_ptRecoBinWidth_trueSensitivityData_add1over4",
                 outdir,
                 ["Data stat true (1/2 to full)", "Data stat true (1/4 to 1/2)", "Data stat (full MC stat)", f"Data stat {tag_halfMC}", f"Data stat {tag_1over4MC}"],
                 legendCoords="0.2,0.62,0.6,0.92;1",
                 vecMCcolors=[ROOT.kBlack, ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kBlue],
                 vecLineStyle=[1,2,1,1,2],
                 vecMarkerStyle=[20,25,20,20,25],
                 passCanvas=canvas1D, graphDrawStyle="pl", legEntryStyle="PL",
                 skipLumi=True)

    drawGraphCMS([gdata_true, gdata_true_otherHalf, gdata_true_fromHalf, gdata_true_fromOtherHalf, gdata_true_fromHalfOtherFourth],
                 "p_{T} bin width (GeV)", "Impact on m_{W} (MeV)::0,50",
                 "impacts_mw_ptRecoBinWidth_trueSensitivityData_add1over4_testDifferentHalf",
                 outdir,
                 ["Data stat true (1/2 to full)", "Data stat true (other 1/2 to full)", "Data stat true (1/4 to 1/2)", "Data stat true (1/4 to other 1/2)", "Data stat true (other 1/4 to other 1/2)"],
                 legendCoords="0.4,0.17,0.92,0.52;1",
                 vecMCcolors=[ROOT.kBlack, ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kBlue],
                 vecLineStyle=[1,2,1,2,1],
                 vecMarkerStyle=[20,25,20,20,25],
                 passCanvas=canvas1D, graphDrawStyle="pl", legEntryStyle="PL",
                 skipLumi=True)

    ##
    drawGraphCMS([gdata_true, gdata_true_FourthToFull, gdata_true_otherFourthToFull, gdata, gdata_1over4MC, gdata_other1over4MC],
                 "p_{T} bin width (GeV)", "Impact on m_{W} (MeV)::0,50",
                 "impacts_mw_ptRecoBinWidth_trueSensitivityData_checkFourthToFull",
                 outdir,
                 ["Data stat true (1/2 to full)", "Data stat true (1/4 to 1)", "Data stat true (other 1/4 to 1)", "Data stat (full MC stat)", f"Data stat {tag_1over4MC}", f"Data stat {tag_other1over4MC}"],
                 legendCoords="0.18,0.65,0.62,0.99;1",
                 vecMCcolors=[ROOT.kBlack, ROOT.kGreen+3, ROOT.kGreen+3, ROOT.kRed, ROOT.kBlue, ROOT.kBlue],
                 vecLineStyle=[1,1,2,1,1,2],
                 vecMarkerStyle=[20,20,25,20,20,25],
                 passCanvas=canvas1D, graphDrawStyle="pl", legEntryStyle="PL",
                 skipLumi=True, solidLegend=True)

    # now use total uncertainty with extended formula
    yvals_true_gtot_FourthToFull = getTrueTotalUncertainty(yvals_gtot, yvals_1over4MC_gtot, alpha=4, rho=1)
    yvals_true_gtot_otherFourthToFull = getTrueTotalUncertainty(yvals_gtot, yvals_other1over4MC_gtot, alpha=4, rho=1)
    #
    yvals_true_gtot_HalfToFull = getTrueTotalUncertainty(yvals_gtot, yvals_halfMC_gtot, alpha=2, rho=1)
    yvals_true_gtot_otherHalfToFull = getTrueTotalUncertainty(yvals_gtot, yvals_halfMCother_gtot, alpha=2, rho=1)
    #
    yvals_true_gtot_fromHalfAndFourth = getTrueTotalUncertainty(yvals_halfMC_gtot, yvals_1over4MC_gtot, alpha=2, rho=2)
    yvals_true_gtot_fromOtherHalfAndFourth = getTrueTotalUncertainty(yvals_halfMCother_gtot, yvals_1over4MC_gtot, alpha=2, rho=2)
    ###
    gtot_true_FourthToFull = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_true_gtot_FourthToFull))
    gtot_true_otherFourthToFull = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_true_gtot_otherFourthToFull))
    #
    gtot_true_HalfToFull = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_true_gtot_HalfToFull))
    gtot_true_otherHalfToFull = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_true_gtot_otherHalfToFull))
    #
    gtot_true_fromHalfAndFourth = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_true_gtot_fromHalfAndFourth))
    gtot_true_fromOtherHalfAndFourth = ROOT.TGraph(len(xvals), array('d', xvals), array('d', yvals_true_gtot_fromOtherHalfAndFourth))

    drawGraphCMS([gtot, gtot_true_HalfToFull, gtot_true_otherHalfToFull, gtot_true_FourthToFull, gtot_true_otherFourthToFull],
                 "p_{T} bin width (GeV)", "Impact on m_{W} (MeV)::0,50",
                 "impacts_mw_ptRecoBinWidth_trueTotalFromFormulaWithBBB_fractToFull",
                 outdir,
                 ["Total (full MC stat)", "True total (1/2 MC to full)", "True total (other 1/2 MC to full)", "True total (1/4 MC to full)", "True total (other 1/4 MC to full)"],
                 legendCoords="0.48,0.15,0.92,0.5;1",
                 vecMCcolors=[ROOT.kBlack, ROOT.kRed, ROOT.kRed, ROOT.kBlue, ROOT.kBlue],
                 vecLineStyle=[1,1,2,1,2],
                 vecMarkerStyle=[20,20,25,20,25],
                 passCanvas=canvas1D, graphDrawStyle="pl", legEntryStyle="PL",
                 skipLumi=True)

    drawGraphCMS([gtot, gtot_true_HalfToFull, gtot_true_otherHalfToFull, gtot_true_fromHalfAndFourth, gtot_true_fromOtherHalfAndFourth],
                 "p_{T} bin width (GeV)", "Impact on m_{W} (MeV)::0,50",
                 "impacts_mw_ptRecoBinWidth_trueTotalFromFormulaWithBBB_halfToFull",
                 outdir,
                 ["Total (full MC stat)", "True total (1/2 MC to full)", "True total (other 1/2 MC to full)", "True total (1/4 to 1/2 MC)", "True total (1/4 to other 1/2 MC)"],
                 legendCoords="0.48,0.15,0.92,0.5;1",
                 vecMCcolors=[ROOT.kBlack, ROOT.kRed, ROOT.kRed, ROOT.kBlue, ROOT.kBlue],
                 vecLineStyle=[1,1,2,1,2],
                 vecMarkerStyle=[20,20,25,20,25],
                 passCanvas=canvas1D, graphDrawStyle="pl", legEntryStyle="PL",
                 skipLumi=True)
    
    rf = safeOpenFile(outdir+"impactMW_signalOnly_statOnly.root", mode="RECREATE")
    rf.cd()
    gdata.Write("gdata_fullMCstat")  
    gmc.Write("gmc_fullMCstat")    
    gtot.Write("gtot_fullMCstat")    
    gdata_halfMC.Write("gdata_halfMCstat")    
    gmc_halfMC.Write("gmc_halfMCstat")    
    gtot_halfMC.Write("gtot_halfMCstat")    
    rf.Close()
    
if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    ROOT.TH1.SetDefaultSumw2()

    runSignalOnly()
