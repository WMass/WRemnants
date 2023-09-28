#!/usr/bin/env python3

# run all pieces (and merge)
# python w_mass_13TeV/smoothLeptonScaleFactors.py /path/to/tnp/efficiencies_ERA/mu_STEP_CHARGE/allEfficiencies_2D.root /path/to/tnp/smoothLeptonScaleFactors/ --run-all [--do-merge] [--do-steps isonotrig iso isoplus isominus triggerplus triggerminus idip idipplus idipminus tracking trackingplus trackingminus reco recoplus recominus]

#
# TIPS:
# --> input path is usually the main folder produced by the TnP code, which contains a folder like efficiencies_GtoH/
# --> for bookkeeping it is better to create output folder smoothLeptonScaleFactors/ where efficiencies_GtoH/ is located
# --> add -d just to print commands without running
# --> add --do-merge to do everything in one go (remove --run-all if only merging is needed)
# --> ERA, STEP, CHARGE are keywords that are converted internally in runFiles() when doing multiple steps
# --> a summary of bad fits (if any) is printed on stdout for each step, but also in a txt file for easier check
#     when running multiple steps in series. For the smoothing to make sense, status and covstatus MUST be 0 for all fits

import os, re, array, math
import argparse
from copy import *

import numpy as np
import tensorflow as tf
import hist
import boost_histogram as bh
import narf
import narf.fitutils
import subprocess

from functools import partial

import utilitiesCMG
utilities = utilitiesCMG.util()

## safe batch mode
import sys
args = sys.argv[:]
sys.argv = ['-b']
import ROOT
sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

#sys.path.append(os.getcwd() + "/plotUtils/")
#from utility import *
from scripts.analysisTools.plotUtils.utility import *

import wremnants

# for a quick summary at the end
badFitsID_data = {}
badFitsID_mc = {}
badFitsID_sf = {}
badCovMatrixID_data = {}
badCovMatrixID_mc = {}
badCovMatrixID_sf = {}

# need to define functions as global below to avoid them being deleted out of fitTurnOnTF
pol2_tf_scaled = None
pol3_tf_scaled = None
pol4_tf_scaled = None
polN_tf_scaled = None
# analysis acceptance, to draw vertical lines on plots
min_pt_accept_ = 26.0
max_pt_accept_ = 55.0

#############################################
# some functions for tensorflow

# for root to be consistent with functions passed to tensorflow
def pol2_root(xvals, parms, xLowVal = 0.0, xFitRange = 1.0):
    xscaled = (xvals[0] - xLowVal) / xFitRange
    return parms[0] + parms[1]*xscaled + parms[2]*xscaled**2

def pol3_root(xvals, parms, xLowVal = 0.0, xFitRange = 1.0):
    xscaled = (xvals[0] - xLowVal) / xFitRange
    return parms[0] + parms[1]*xscaled + parms[2]*xscaled**2 + parms[3]*xscaled**3

def pol4_root(xvals, parms, xLowVal = 0.0, xFitRange = 1.0):
    xscaled = (xvals[0] - xLowVal) / xFitRange 
    return parms[0] + parms[1]*xscaled + parms[2]*xscaled**2 + parms[3]*xscaled**3 + parms[4]*xscaled**4

def erf_tf(xvals, parms):
    return parms[0] * (1.0 + tf.math.erf( (xvals[0] - parms[1]) / parms[2] ))

def antiErf_tf(xvals, parms):
    return 1.0 - parms[0] * (1.0 + tf.math.erf( (xvals[0] - parms[1]) / parms[2] ))

#############################################

def getReducedChi2andLabel(func):
    if func.GetNDF():
        reducedChi2 = func.GetChisquare() / func.GetNDF()
        lineChi2 = "#chi^{{2}} = {chi2:.2g} / {ndf}".format(chi2=func.GetChisquare(),ndf=int(func.GetNDF()))
    else:
        reducedChi2 = 0
        lineChi2 = "BAD! #chi^{{2}} = {chi2:.2g} / {ndf}".format(chi2=func.GetChisquare(),ndf=int(func.GetNDF()))

    return float(reducedChi2),lineChi2


def copyHisto(h1, h2, copyError=False):

    if h1.GetDimension() != h2.GetDimension():
        print(f"Error in copyHisto(): histograms have different dimensions. Dim(h1)={h1.GetDimension()}  Dim(h2)={h2.GetDimension()}. Exit")
        quit()

    if h1.GetDimension() == 1:
        for ix in range(h2.GetNbinsX()+2):
            if copyError:
                h1.SetBinContent(ix, h2.GetBinContent(ix, iy))
                h1.SetBinError(ix, h2.GetBinError(ix, iy))
            else:
                h1.SetBinContent(ix, h2.GetBinError(ix, iy))
                h1.SetBinError(ix, 0.0)
    elif h1.GetDimension() == 2:
        for ix in range(h2.GetNbinsX()+2):
            for iy in range(h2.GetNbinsY()+2):
                if copyError:
                    h1.SetBinContent(ix,iy, h2.GetBinError(ix, iy))
                    h1.SetBinError(ix,iy, 0.0)
                else:
                    h1.SetBinContent(ix,iy, h2.GetBinContent(ix, iy))
                    h1.SetBinError(ix,iy, h2.GetBinError(ix, iy))

    else:
        print("Error in copyHisto(): function not implemented for dimension > 2. Exit")
        quit()        

def make1Dhist(namePrefix, h2D, ptbins, step):
    hpt = {}
    for x in range(1, h2D.GetNbinsX()+1):
        binID = x-1
        hpt[binID] = ROOT.TH1D(f"{namePrefix}_{binID}",
                               "%s: %.4g < #eta < %.4g" % (step, h2D.GetXaxis().GetBinLowEdge(x), h2D.GetXaxis().GetBinLowEdge(x+1)),
                               len(ptbins)-1, array('d',ptbins)
        )
        for y in range(1, h2D.GetNbinsY()+1):
            hpt[binID].SetBinContent(y, h2D.GetBinContent(x,y))         
            hpt[binID].SetBinError(y, h2D.GetBinError(x,y))
    return hpt
    
############

def quickPlotTH1(c, h, outname,channel, postfix=""):
    c.cd()
    h.Scale(1./h.Integral())
    h.GetXaxis().SetTitleOffset(1.2)
    h.GetXaxis().SetTitleSize(0.05)
    h.GetXaxis().SetLabelSize(0.06)
    #h.GetXaxis().LabelsOption("v")
    h.GetYaxis().SetTitle("Fraction of events")
    h.GetYaxis().SetTitleOffset(1.15)
    h.GetYaxis().SetTitleSize(0.05)
    h.GetYaxis().SetLabelSize(0.04)
    h.Draw("HIST")
    for ext in ["png","pdf"]:
        c.SaveAs(f"{outname}bestFitFunction{postfix}_{channel}.{ext}")

#########

def getCoordinateNDC(x, canvas, vert=False):
    canvas.Update() # might be needed
    if vert:
        return (x - canvas.GetY1()) / (canvas.GetY2() - canvas.GetY1())
    else:
        return (x - canvas.GetX1()) / (canvas.GetX2() - canvas.GetX1())


def fitTurnOnTF(hist, key, outname, mc, channel="el", hist_chosenFunc=0, drawFit=True, 
                step=None,
                fitRange=None,
                hist_reducedChi2=None,
                #hist_FuncParam_vs_eta=None,
                #hist_FuncCovMatrix_vs_eta=None,  # TH3, eta on x and cov matrix in yz
                charge = "both",
                etabins = [],
                widthPtSmooth=0.2,
                hist_nomiAndAlt_etapt=None,
                histAlt = None,
                efficiencyFitPolDegree=4,
                addCurve=None,
                addCurveLegEntry=""
):

    doingSF = True if mc == "SF" else False
    doSpline = True if efficiencyFitPolDegree < 0 else False
    
    chargeText = ""
    if charge == "plus": chargeText = "positive"
    if charge == "minus": chargeText = "negative"
    
    originalMaxPt = hist.GetXaxis().GetBinLowEdge(1+hist.GetNbinsX())

    outdir = "{out}{mc}/".format(out=outname,mc=mc)
    createPlotDirAndCopyPhp(outdir)
    adjustSettings_CMS_lumi()
    
    leftMargin = 0.15
    rightMargin = 0.04
    bottomMargin = 0.12
    canvas = ROOT.TCanvas(f"canvas_{mc}_{key}","",700,900)
    canvas.SetTickx(1)
    canvas.SetTicky(1)
    canvas.cd()
    canvas.SetLeftMargin(leftMargin)
    canvas.SetBottomMargin(bottomMargin)
    canvas.SetRightMargin(rightMargin)
    canvas.cd()                           

    setTDRStyle()
    lowerPanelHeight = 0.3 # for bottom panel with pulls or residuals
    canvas.SetBottomMargin(lowerPanelHeight)
    pad2 = ROOT.TPad("pad2","pad2",0,0.,1,0.9)
    pad2.SetTopMargin(1-lowerPanelHeight)
    pad2.SetRightMargin(rightMargin)
    pad2.SetLeftMargin(leftMargin)
    pad2.SetFillColor(0)
    pad2.SetGridy(1)
    pad2.SetFillStyle(0)
    
    hist.SetLineColor(ROOT.kBlack)
    hist.SetMarkerColor(ROOT.kBlack)
    hist.SetMarkerStyle(20)
    hist.SetMarkerSize(1)
    
    hist.GetXaxis().SetLabelSize(0)
    hist.GetXaxis().SetTitle("")
    # hist.GetXaxis().SetTitle(f"{chargeText} muon p_{{T}} (GeV)")
    # hist.GetXaxis().SetTitleOffset(1.1)
    # hist.GetXaxis().SetTitleSize(0.05)
    # hist.GetXaxis().SetLabelSize(0.04)

    maxFitRange = hist.GetXaxis().GetBinLowEdge(1+hist.GetNbinsX())
    minFitRange = hist.GetXaxis().GetBinLowEdge(1)
    if fitRange != None:
        if fitRange[1] > 0:
            maxFitRange = fitRange[1]
        if fitRange[0] > 0:
            minFitRange = fitRange[0]
    hist.GetXaxis().SetRangeUser(minFitRange, maxFitRange)

    if mc == "SF":
        hist.GetYaxis().SetTitle("Data/MC scale factor")
    else:
        hist.GetYaxis().SetTitle("{mc} efficiency".format(mc=mc))
    hist.GetYaxis().SetTitleOffset(1.45)
    hist.GetYaxis().SetTitleSize(0.05)
    hist.GetYaxis().SetLabelSize(0.04)
    if histAlt:
        histAlt.SetStats(0)
        histAlt.SetLineColor(ROOT.kGray+2)
        histAlt.SetMarkerColor(ROOT.kGray+2)
        histAlt.SetMarkerStyle(ROOT.kOpenCircle)
        histAlt.SetMarkerSize(1)
        miny,maxy = getMinMaxMultiHisto([hist, histAlt], sumError=True)
    else:
        miny,maxy = getMinMaxHisto(hist, sumError=True)
    offset = 0.1 * (maxy - miny)
    upOffset = offset * (3.5 if doingSF else 3.5)
    miny -= offset
    maxy += upOffset
    hist.GetYaxis().SetRangeUser(miny, maxy)
    hist.SetStats(0)
    hist.Draw("EP")
    ##
    ## TODO
    ## Unlike root, tensorflow fits use the bin centers to run the actual fit, and there is no concept of fit range
    ## so, if one wants to fit a subset of the histogram range one needs to pass the slice of the boost histogram
    ## currently this is not implemented        
            
    ###################
    # fits
    ####################
    boost_hist = narf.root_to_hist(hist)
    if histAlt:
        boost_hist_alt = narf.root_to_hist(histAlt)

    ###############################################################
    fitFunction = None
    fitres_TF = None
    defaultFunc = ""
    badFitsID = None
    badCovMatrixID = None
    
    xFitRange = maxFitRange - minFitRange

    if doingSF:

        if step == "tracking":

            global pol2_tf_scaled
            if pol2_tf_scaled == None:
                pol2_tf_scaled = partial(pol2_root, xLowVal=minFitRange, xFitRange=xFitRange)
            params = np.array([1.0, 0.0, 0.0])
            res_tf1_pol2 = narf.fitutils.fit_hist(boost_hist, pol2_tf_scaled, params)
            tf1_pol2 = ROOT.TF1("tf1_pol2", pol2_tf_scaled, minFitRange, maxFitRange, len(params))
            tf1_pol2.SetParameters( np.array( res_tf1_pol2["x"], dtype=np.dtype('d') ) )
            tf1_pol2.SetLineWidth(3)
            tf1_pol2.SetLineColor(ROOT.kRed+2)

            fitres_TF = {"pol2_tf" : res_tf1_pol2}
            fitFunction = {
                "pol2_tf" : {
                    "func" : tf1_pol2,
                    "leg"  : "Pol2",
                    "hist": hist,
                }
            }
            defaultFunc = "pol2_tf"
            if histAlt:
                params = np.array([1.0, 0.0, 0.0])
                res_tf1_pol2_alt = narf.fitutils.fit_hist(boost_hist_alt, pol2_tf_scaled, params)
                tf1_pol2_alt = ROOT.TF1("tf1_pol2_alt", pol2_tf_scaled, minFitRange, maxFitRange, len(params))
                tf1_pol2_alt.SetParameters( np.array( res_tf1_pol2_alt["x"], dtype=np.dtype('d') ) )
                tf1_pol2_alt.SetLineWidth(2)
                tf1_pol2_alt.SetLineColor(ROOT.kAzure+2)
                fitres_TF["pol2_alt_tf"] = res_tf1_pol2_alt
                fitFunction["pol2_alt_tf"] = {"func" : tf1_pol2_alt,
                                              "leg" : "dataAltSig",
                                              "hist": histAlt,
                }
        else:
            global pol3_tf_scaled
            if pol3_tf_scaled == None:
                pol3_tf_scaled = partial(pol3_root, xLowVal=minFitRange, xFitRange=xFitRange)
            params = np.array([1.0, 0.0, 0.0, 0.0])
            res_tf1_pol3 = narf.fitutils.fit_hist(boost_hist, pol3_tf_scaled, params)
            tf1_pol3 = ROOT.TF1("tf1_pol3", pol3_tf_scaled, minFitRange, maxFitRange, len(params))
            tf1_pol3.SetParameters( np.array( res_tf1_pol3["x"], dtype=np.dtype('d') ) )
            tf1_pol3.SetLineWidth(3)
            tf1_pol3.SetLineColor(ROOT.kRed+2)

            # tf1_pol3_test = ROOT.TF1("tf1_pol3_test", pol3_tf_scaled, minFitRange, maxFitRange, len(params))
            # tf1_pol3_test.SetParameters( np.array( res_tf1_pol3["x"], dtype=np.dtype('d') ) )
            # tf1_pol3_test.SetLineStyle(ROOT.kDashed)
            # tf1_pol3_test.SetLineWidth(5)
            # tf1_pol3_test.SetLineColor(ROOT.kBlue)
            # fitopt = "FMBRQS+" # add FM if using Minuit   
            # hist.Fit(tf1_pol3_test, fitopt)

            fitres_TF = {"pol3_tf" : res_tf1_pol3}
            fitFunction = {
                "pol3_tf" : {
                    "func" : tf1_pol3,
                    "leg"  : "Pol3",
                    "hist": hist,
                }
            }
            defaultFunc = "pol3_tf"
            if histAlt:
                params = np.array([1.0, 0.0, 0.0, 0.0])
                res_tf1_pol3_alt = narf.fitutils.fit_hist(boost_hist_alt, pol3_tf_scaled, params)
                tf1_pol3_alt = ROOT.TF1("tf1_pol3_alt", pol3_tf_scaled, minFitRange, maxFitRange, len(params))
                tf1_pol3_alt.SetParameters( np.array( res_tf1_pol3_alt["x"], dtype=np.dtype('d') ) )
                tf1_pol3_alt.SetLineWidth(2)
                tf1_pol3_alt.SetLineColor(ROOT.kAzure+2)
                fitres_TF["pol3_alt_tf"] = res_tf1_pol3_alt
                fitFunction["pol3_alt_tf"] = {"func" : tf1_pol3_alt,
                                              "leg" : "dataAltSig",
                                              "hist": histAlt,
                }
                
        ###
        badFitsID = badFitsID_sf
        badCovMatrixID = badCovMatrixID_sf
    else:

        if step == "antiiso":
            tf1_erf = ROOT.TF1("tf1_erf","1.0 - [0] * (1.0 + TMath::Erf((x-[1])/[2]))", minFitRange, maxFitRange)
            res_tf1_erf = narf.fitutils.fit_hist(boost_hist, antiErf_tf, np.array([1.0, 35.0, 3.0]))
        else:
            tf1_erf = ROOT.TF1("tf1_erf","[0] * (1.0 + TMath::Erf((x-[1])/[2]))", minFitRange, maxFitRange)
            res_tf1_erf = narf.fitutils.fit_hist(boost_hist, erf_tf, np.array([1.0, 35.0, 3.0]))
        tf1_erf.SetParameters( np.array( res_tf1_erf["x"], dtype=np.dtype('d') ) )
        tf1_erf.SetLineWidth(2)
        tf1_erf.SetLineStyle(ROOT.kDashed)
        tf1_erf.SetLineColor(ROOT.kBlue)
            
        if doSpline:
            tf1_polN = ROOT.TSpline3(hist, "b1e1")
            tf1_polN.SetLineWidth(3)
            tf1_polN.SetLineColor(ROOT.kRed+2)
            # tf1_polN.Draw("pclsame")
            res_tf1_polN = None
        else:
            global polN_tf_scaled
            if polN_tf_scaled == None:
                polN_tf_scaled = partial(polN_root_, xLowVal=minFitRange, xFitRange=xFitRange, degree=efficiencyFitPolDegree)
            params = np.array([1.0] + [0.0 for i in range(efficiencyFitPolDegree)])
            res_tf1_polN = narf.fitutils.fit_hist(boost_hist, polN_tf_scaled, params)
            tf1_polN = ROOT.TF1(f"tf1_pol{efficiencyFitPolDegree}", polN_tf_scaled, minFitRange, maxFitRange, len(params))
            tf1_polN.SetParameters( np.array( res_tf1_polN["x"], dtype=np.dtype('d') ) )
            tf1_polN.SetLineWidth(3)
            tf1_polN.SetLineColor(ROOT.kRed+2)
        #
            
        fitres_TF = {"erf" : res_tf1_erf,
                     "polN_tf" : res_tf1_polN,
        }
        fitFunction = {
            "polN_tf" : {
                "func" : tf1_polN,
                "leg"  : "Spline" if doSpline else f"Pol{efficiencyFitPolDegree}",
                "hist": hist,

            },
            "erf" : {
                "func" : tf1_erf,
                "leg"  : "1-Erf" if step == "antiiso" else "Erf",
                "hist": hist,
                
            },
        }
        defaultFunc = "polN_tf"
        if histAlt:
            if doSpline:
                tf1_polN_alt = ROOT.TSpline3(hist, "b1e1")
                tf1_polN_alt.SetLineWidth(2)
                tf1_polN_alt.SetLineColor(ROOT.kAzure+2)
                # tf1_polN.Draw("pclsame")
                res_tf1_polN_alt = None
            else:
                params = np.array([1.0] + [0.0 for i in range(efficiencyFitPolDegree)])
                res_tf1_polN_alt = narf.fitutils.fit_hist(boost_hist_alt, polN_tf_scaled, params)
                tf1_polN_alt = ROOT.TF1("tf1_polN_alt", polN_tf_scaled, minFitRange, maxFitRange, len(params))
                tf1_polN_alt.SetParameters( np.array( res_tf1_polN_alt["x"], dtype=np.dtype('d') ) )
                tf1_polN_alt.SetLineWidth(2)
                tf1_polN_alt.SetLineColor(ROOT.kAzure+2)
            fitres_TF["polN_alt_tf"] = res_tf1_polN_alt
            fitFunction["polN_alt_tf"] = {"func" : tf1_polN_alt,
                                          "leg" : "dataAltSig",
                                          "hist": histAlt,
            }

        if mc == "MC":
            badFitsID = badFitsID_mc
            badCovMatrixID = badCovMatrixID_mc
        else:
            badFitsID = badFitsID_data
            badCovMatrixID = badCovMatrixID_data

    ###############################################################
            
    for fr in fitres_TF.keys():
        if fitres_TF[fr] == None: continue
        status = fitres_TF[fr]["status"]
        covstatus = fitres_TF[fr]["covstatus"]
        if status:
            print(f"-----> Function {fr} had status {status}")
            if key not in badFitsID.keys():
                badFitsID[key] = {}
            badFitsID[key][fr] = status
        if covstatus:
            print(f"-----> Function {fr} had covstatus {covstatus}")
            if key not in badCovMatrixID.keys():
                badCovMatrixID[key] = {}
            badCovMatrixID[key][fr] = covstatus
        ## security check on chi2, to make sure TF1 for plotting is consistent with actual fit    
        manualChi2 = 0.0
        h = fitFunction[fr]["hist"]
        for ib in range(1, 1+h.GetNbinsX()):
            pt = h.GetXaxis().GetBinCenter(ib)
            item = (h.GetBinContent(ib) - fitFunction[fr]["func"].Eval(pt))/h.GetBinError(ib)
            manualChi2 += item * item
        fitChi2 = fitres_TF[fr]["loss_val"]
        if abs(manualChi2 - fitChi2) > 0.01:
            print(f"   ====> {fr}: manual/fit chi2  = {manualChi2}/{fitChi2}")
            
    npar = 0 if doSpline else fitFunction[defaultFunc]["func"].GetNpar()
    
    hband = ROOT.TH1D("hband", "", int(math.ceil((maxFitRange-minFitRange))/widthPtSmooth), minFitRange, maxFitRange)
    if hist_nomiAndAlt_etapt is not None and hband.GetNbinsX() != hist_nomiAndAlt_etapt.GetNbinsY():
        print("ERROR: hband and hist_nomiAndAlt_etapt have a different number of pt bins ({hband.GetNbinsX()} and {hist_nomiAndAlt_etapt.GetNbinsY()}), please check!")
        quit()
    hband.SetStats(0)
    if not doSpline:
        hband.SetFillColor(ROOT.kGray)
    #hband.SetLineColor(fitFunction[defaultFunc]["func"].GetLineColor())
    #hband.SetFillStyle(3001)
    for ib in range(1, hband.GetNbinsX()+1):
        pt = hband.GetBinCenter(ib)
        val = max(0.001, fitFunction[defaultFunc]["func"].Eval(pt))
        # protect against efficiency becoming larger than 1.0
        # usually it happens because of the pol3 extrapolation for very high pt, which is not used for the analysis
        # for MC set to slightly less than 1.0 to avoid issues with antiiso when doing 1-eff and then data/MC ratio (do the same for data just in case)
        if val >= 1.0 and not doingSF:
            val = 0.9995
        hband.SetBinContent(ib, val)
        if hist_nomiAndAlt_etapt:
            # assuming the pt binning is the same, which it should, although the code should be made more robust
            hist_nomiAndAlt_etapt.SetBinContent(key+1, ib, 1, val)

    
    # store all variations for faster access below
    altParameters = np.array([np.zeros(npar, dtype=np.dtype('d'))] * (npar * 2), dtype=np.dtype('d'))
    if not doSpline:
        # diagonalize and get eigenvalues and eigenvectors
        e, v = np.linalg.eigh(fitres_TF[defaultFunc]["cov"])
        #print(altParameters)
        for ivar in range(npar):
            shift = np.sqrt(e[ivar]) * v[:, ivar]
            altParameters[ivar]      = fitres_TF[defaultFunc]["x"] + shift
            altParameters[ivar+npar] = fitres_TF[defaultFunc]["x"] - shift

    tf1_func_alt = ROOT.TF1()
    tf1_func_alt.SetName("tf1_func_alt")
    fitFunction[defaultFunc]["func"].Copy(tf1_func_alt)
    tf1_func_alt.SetLineWidth(2)
    if hist_nomiAndAlt_etapt is not None:
        lastSystBin = hist_nomiAndAlt_etapt.GetNbinsZ()
    for ib in range(1, hband.GetNbinsX()+1):
        pt = hband.GetBinCenter(ib)
        err = 0.0
        for ivar in range(npar):
            # set parameters for a given hessian
            tf1_func_alt.SetParameters(altParameters[ivar]) # this is for Up variations, Down ones could not be the mirror image
            funcVal = max(0.001, tf1_func_alt.Eval(pt))
            if funcVal >= 1.0 and not doingSF:
                funcVal = 0.9995
            diff = funcVal - hband.GetBinContent(ib)
            err += diff * diff
            if hist_nomiAndAlt_etapt is not None:
                # now fill TH3, also with down variations
                hist_nomiAndAlt_etapt.SetBinContent(key+1, ib, 2+ivar, funcVal)
                # repeat for Down variations
                tf1_func_alt.SetParameters(altParameters[ivar+npar])
                funcVal = max(0.001, tf1_func_alt.Eval(pt))
                if funcVal >= 1.0 and not doingSF:
                    funcVal = 0.9995
                hist_nomiAndAlt_etapt.SetBinContent(key+1, ib, 2+ivar+npar, funcVal)
        err = math.sqrt(err)
        hband.SetBinError(ib, err)
        if hist_nomiAndAlt_etapt is not None:
            hist_nomiAndAlt_etapt.SetBinError(key+1, ib, 1, err)
            if histAlt:
                for f in fitFunction.keys():
                    if "_alt_tf" in f:
                        funcVal = max(0.001, fitFunction[f]["func"].Eval(pt))
                        if funcVal >= 1.0 and not doingSF:
                            funcVal = 0.9995
                        hist_nomiAndAlt_etapt.SetBinContent(key+1, ib, lastSystBin, funcVal)
        
    hband.Draw("E4SAME")
    # redraw to have them on top
    for f in fitFunction.keys():
        fitFunction[f]["func"].Draw("LSAME")
    if addCurve:
        addCurve.SetLineColor(ROOT.kGreen+2)
        addCurve.SetLineStyle(1)
        addCurve.SetLineWidth(2)
        addCurve.SetFillColorAlpha(ROOT.kGreen+1, 0.35)
        addCurve.SetFillStyle(1001)
        addCurve.Draw("LE4 SAME")
        #addCurve.Draw("C SAME")
        #addCurve.Draw("HISTSAME")
        chi2curve = 0.0
        for ib in range(1, 1+hist.GetNbinsX()):
            item = (hist.GetBinContent(ib) - addCurve.GetBinContent(addCurve.GetXaxis().FindFixBin(hist.GetBinCenter(ib))))/hist.GetBinError(ib)
            chi2curve += item * item
    hist.Draw("EPSAME")
    if histAlt:
        histAlt.Draw("EPSAME")

    ## draw curves if needed on top of the band
    # colors_alt = [ROOT.kBlue, ROOT.kMagenta, ROOT.kCyan+1, ROOT.kOrange+2, ROOT.kGreen+2, ROOT.kViolet, ROOT.kSpring+9, ROOT.kPink]
    # for ivar in range(npar):
    #     #tf1_func_alt.SetLineStyle(ROOT.kDotted)
    #     tf1_func_alt.SetLineColor(colors_alt[ivar])
    #     # set parameters for a given hessian
    #     tf1_func_alt.SetParameters(altParameters[ivar])
    #     tf1_func_alt.DrawCopy("LSAME") # can also not draw these lines, too busy plot otherwise
    #     #tf1_func_alt.SetParameters(altParameters[ivar+npar])
    #     #tf1_func_alt.DrawCopy("LSAME") # can also not draw these lines, too busy plot otherwise

    #######

    nFits = len(fitFunction.keys())

    upLeg = 0.92
    downLeg = max(0.5, upLeg - 0.06 * nFits)
    if addCurve:
        downLeg = max(0.5, downLeg - 0.03) # add some more vertical space, but not too much
    leftLeg = 0.16
    rightLeg = 0.95
    
    leg = ROOT.TLegend(leftLeg, downLeg, rightLeg, upLeg)
    leg.SetFillColor(0)
    #leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetFillColorAlpha(0,0.6)

    hNomiForLegend = ROOT.TH1D("hNomiForLegend", "", 1, 0, 1)
    hNomiForLegend.SetStats(0)
    hAltForLegend = ROOT.TH1D("hAltForLegend", "", 1, 0, 1)
    hAltForLegend.SetStats(0)
    
    # for chosen function
    reducedChi2 = 0.0
    for f in fitFunction.keys():
        legEntry = f"Nomi {fitFunction[f]['leg']}" if f == defaultFunc else fitFunction[f]["leg"]
        if doSpline:
            chi2 = 0
            ndof = 1
        else:
            chi2 = fitres_TF[f]["loss_val"]
            ndof = int(hist.GetNbinsX() - fitFunction[f]["func"].GetNpar())
            legEntry += f"   #chi^{{2}} = {round(chi2,1)} / {ndof}"
            chi2prob = ROOT.TMath.Prob(chi2, ndof)
            if chi2prob < 0.05:
                perc_chi2prob = 100.0 * chi2prob
                sign = "="
                if perc_chi2prob < 0.1:
                    perc_chi2prob = 0.1
                    sign = "<"
                legEntry += " (prob {} {}%)".format(sign, round(perc_chi2prob,1))
        if f == defaultFunc:
            hNomiForLegend.SetMarkerColor(hist.GetMarkerColor())
            hNomiForLegend.SetMarkerStyle(hist.GetMarkerStyle())
            hNomiForLegend.SetFillColor(hband.GetFillColor())
            hNomiForLegend.SetLineColor(fitFunction[f]["func"].GetLineColor())
            leg.AddEntry(hNomiForLegend, legEntry, 'EPLF')  
            reducedChi2 = chi2/ndof
        elif "_alt_tf" in f:
            hAltForLegend.SetMarkerColor(histAlt.GetMarkerColor())
            hAltForLegend.SetMarkerStyle(histAlt.GetMarkerStyle())
            hAltForLegend.SetLineColor(fitFunction[f]["func"].GetLineColor())            
            leg.AddEntry(hAltForLegend, legEntry, 'EPL')
        else:
            leg.AddEntry(fitFunction[f]["func"], legEntry, 'L')  

    if addCurve:
        leg.AddEntry(addCurve, f"{addCurveLegEntry}, #chi^{{2}} = {round(chi2curve,1)}", 'LF')
            
    #leg.AddEntry(hband, "Model uncertainty", 'F')
    leg.Draw('same')
    canvas.RedrawAxis("sameaxis")

    setTDRStyle()
    ROOT.gStyle.SetOptTitle(1)  # use histogram title with binning as canvas title

    if hist_reducedChi2:
        hist_reducedChi2.Fill(reducedChi2)
    if hist_chosenFunc:
        hist_chosenFunc.Fill(defaultFunc, 1)

    # lat = ROOT.TLatex()
    # line = ""
    # lineChi2 = ""
    # lat.SetNDC();
    # lat.SetTextSize(0.045);
    # lat.SetTextFont(42);
    # lat.SetTextColor(1);        
    # xmin = 0.20 
    # yhi = 0.85
    # lat.DrawLatex(xmin, yhi, line);
    # lat.DrawLatex(xmin, yhi-0.05, lineChi2);

    ### Draw band to highlight acceptance
    hAcceptBand = copy.deepcopy(hist.Clone("hAcceptBand"))
    yMaxBand = downLeg * (canvas.GetY2() - canvas.GetY1()) + canvas.GetY1()
    for i in range(1, 1 + hAcceptBand.GetNbinsX()):
        if hAcceptBand.GetBinCenter(i) > 26.0 and hAcceptBand.GetBinCenter(i) < 55.0:
            hAcceptBand.SetBinContent(i, 0.0)
            hAcceptBand.SetBinError(i, 0.0)
        else:
            hAcceptBand.SetBinContent(i, yMaxBand)
            hAcceptBand.SetBinError(i, 0.0)
    hAcceptBand.SetFillColorAlpha(ROOT.kYellow+1, 0.5)
    if step != "tracking":
        hAcceptBand.Draw("HIST SAME")
    ########################################
    
    # Now the bottom panel
    pad2.Draw()
    pad2.cd()

    frame = hist.Clone("frame")
    frame.SetTitle("")
    frame.GetXaxis().SetTitleOffset(1.2)
    frame.GetXaxis().SetTitleSize(0.05)
    frame.GetXaxis().SetLabelSize(0.04)
    frame.SetStats(0)
    frame.Reset("ICES")
    frame.GetXaxis().SetRangeUser(minFitRange, maxFitRange)
    frame.GetYaxis().SetNdivisions(5)
    denLabel = fitFunction[defaultFunc]["leg"]
    frame.GetYaxis().SetTitle(f"X / {denLabel}") # or Pulls
    frame.GetYaxis().SetTitleOffset(1.45)
    frame.GetYaxis().SetTitleSize(0.05)
    frame.GetYaxis().SetLabelSize(0.04)
    frame.GetYaxis().CenterTitle()
    frame.GetXaxis().SetTitle(f"{chargeText} muon p_{{T}} (GeV)")

    ratios = []
    defaultIndex = 0
    den_noerr = copy.deepcopy(hband.Clone("den_noerr"))
    den = copy.deepcopy(hband.Clone("den"))
    den.SetTitle("")    
    for iBin in range (1, den_noerr.GetNbinsX()+1):
        den_noerr.SetBinError(iBin, 0.)
    den.Divide(den_noerr)
    for ifunc,f in enumerate(fitFunction.keys()):
        if f == defaultFunc:
            continue
        ratio = copy.deepcopy(hband.Clone(f"ratio_{f}"))
        ratio.Reset("ICESM")
        ratio.SetMarkerSize(0)
        ratio.SetMarkerStyle(0)
        ratio.SetLineColor(fitFunction[f]["func"].GetLineColor())
        ratio.SetLineWidth(2)
        ratio.SetFillColor(0)
        ratio.SetLineStyle(fitFunction[f]["func"].GetLineStyle())
        for ib in range(1, 1 + ratio.GetNbinsX()):
            xval = ratio.GetBinCenter(ib)
            ratio.SetBinContent(ib, fitFunction[f]["func"].Eval(xval)/ den_noerr.GetBinContent(ib))
        ratios.append(ratio)
    # now additional curve if any
    if addCurve:
        ratioCurve = copy.deepcopy(addCurve.Clone(f"ratioCurve_addCurve"))
        ratioCurve.Reset("ICESM")
        ratioCurve.SetMarkerSize(0)
        ratioCurve.SetMarkerStyle(0)
        ratioCurve.SetLineColor(addCurve.GetLineColor())
        ratioCurve.SetLineWidth(2)
        ratioCurve.SetFillColor(0)
        ratioCurve.SetLineStyle(addCurve.GetLineStyle())
        for ib in range(1, 1 + ratioCurve.GetNbinsX()):
            xval = ratioCurve.GetBinCenter(ib)
            ratioCurve.SetBinContent(ib, addCurve.GetBinContent(addCurve.GetXaxis().FindFixBin(xval))/ den_noerr.GetBinContent(ib))
        
    # now data which is less granular
    dataRatio = copy.deepcopy(hist.Clone("dataRatio"))
    for ib in range(1, 1 + dataRatio.GetNbinsX()):
        ibinDen = hband.GetXaxis().FindFixBin(dataRatio.GetBinCenter(ib))
        denVal = hband.GetBinContent(ibinDen)
        dataRatio.SetBinContent(ib, dataRatio.GetBinContent(ib) / denVal)
        dataRatio.SetBinError(  ib, dataRatio.GetBinError(ib)   / denVal)
        
    rminy, rmaxy = getMinMaxMultiHisto(ratios+[den, dataRatio], excludeEmpty=True, sumError=True,
                                       excludeUnderflow=True, excludeOverflow=True)
    # append here so that the min max doesn't include it, sometimes the last points skyrocket
    if addCurve:
        ratios.append(ratioCurve)

    diffy = rmaxy - rminy
    offset = 0.1
    rminy -= offset * diffy
    rmaxy += offset * diffy
    frame.GetYaxis().SetRangeUser(rminy, rmaxy)
    frame.Draw()
    den.Draw("E4SAME")
    denOnlyLine = copy.deepcopy(den.Clone("denOnlyLine"))
    denOnlyLine.SetFillColor(0)
    denOnlyLine.SetLineWidth(1)
    denOnlyLine.SetLineColor(fitFunction[defaultFunc]["func"].GetLineColor())
    denOnlyLine.Draw("HIST SAME")
    for ip,p in enumerate(ratios):
        drawOpt = "C" # like HIST but smooth curve through points
        p.Draw(f"{drawOpt}SAME")    
    dataRatio.Draw("EP SAME")

    # # now draw vertical lines to highlight acceptance (could have shadow panels too)
    # vertline = ROOT.TLine(36, canvas.GetUymin(), 36, canvas.GetUymax())
    # vertline.SetLineColor(ROOT.kBlack)
    # vertline.SetLineStyle(2)
    # vertline.SetLineWidth(2)
    # min_pt_accept_NDC = getCoordinateNDC(min_pt_accept_, canvas)
    # max_pt_accept_NDC = getCoordinateNDC(max_pt_accept_, canvas)
    # vertline.DrawLineNDC(min_pt_accept_NDC, bottomMargin, min_pt_accept_NDC, downLeg)
    # vertline.DrawLineNDC(max_pt_accept_NDC, bottomMargin, max_pt_accept_NDC, downLeg)
    # #vertline.DrawLine(min_pt_accept_, miny, min_pt_accept_, maxy)
    # #vertline.DrawLine(max_pt_accept_, miny, max_pt_accept_, maxy)
    # ###########

    ### Draw band to highlight acceptance
    hAcceptBandRatio = copy.deepcopy(hist.Clone("hAcceptBandRatio"))
    hAcceptBandRatio.Reset("ICESM")
    yMaxBand = maxy
    for i in range(1, 1 + hAcceptBandRatio.GetNbinsX()):
        if hAcceptBandRatio.GetBinCenter(i) > 26.0 and hAcceptBandRatio.GetBinCenter(i) < 55.0:
            hAcceptBandRatio.SetBinContent(i, 0.0)
            hAcceptBandRatio.SetBinError(i, 0.0)
        else:
            hAcceptBandRatio.SetBinContent(i, yMaxBand)
            hAcceptBandRatio.SetBinError(i, 0.0)
    hAcceptBandRatio.SetFillColorAlpha(ROOT.kYellow+1, 0.5)
    if step != "tracking":
        hAcceptBandRatio.Draw("HIST SAME")
    ########################################
    
    pad2.RedrawAxis("sameaxis")

    #---------------------------------
    
    tmpch = ""
    if charge != "both":
        tmpch = "_" + charge
    for ext in ["pdf","png"]:
        if mc == "SF":
            canvas.SaveAs("{out}sf_pt_{ch}_eta{b}{charge}.{ext}".format(out=outdir,ch=channel,b=key,charge=tmpch,ext=ext))            
        else:
            canvas.SaveAs("{out}eff{mc}_pt_{ch}_eta{b}{charge}.{ext}".format(out=outdir,mc=mc,ch=channel,b=key,charge=tmpch,ext=ext))                            
    return fitFunction[defaultFunc]["func"]

############################################################################

def drawReducedChi2(c, hist_reducedChi2_data, hist_reducedChi2_MC, hist_reducedChi2_sf):
    # now the chi2 histogram
    # put overflow in last bin
    # but get number of entries before (same as integral since we used Fill() without weights)
    entries_MC = hist_reducedChi2_MC.GetEntries()
    entries_data = hist_reducedChi2_data.GetEntries()
    entries_sf = hist_reducedChi2_sf.GetEntries()

    lastBin = hist_reducedChi2_data.GetNbinsX()
    hist_reducedChi2_data.SetBinContent(lastBin, hist_reducedChi2_data.GetBinContent(lastBin) + hist_reducedChi2_data.GetBinContent(1+lastBin) )
    hist_reducedChi2_data.SetBinError(lastBin, math.sqrt( hist_reducedChi2_data.GetBinError(lastBin)*hist_reducedChi2_data.GetBinError(lastBin) 
                                                          + hist_reducedChi2_data.GetBinError(1+lastBin)*hist_reducedChi2_data.GetBinError(1+lastBin) ) 
)
    lastBin = hist_reducedChi2_MC.GetNbinsX()
    hist_reducedChi2_MC.SetBinContent(lastBin, hist_reducedChi2_MC.GetBinContent(lastBin) + hist_reducedChi2_MC.GetBinContent(1+lastBin) )
    hist_reducedChi2_MC.SetBinError(lastBin, math.sqrt( hist_reducedChi2_MC.GetBinError(lastBin)*hist_reducedChi2_MC.GetBinError(lastBin) 
                                                        + hist_reducedChi2_MC.GetBinError(1+lastBin)*hist_reducedChi2_MC.GetBinError(1+lastBin) ) 
    )
    lastBin = hist_reducedChi2_sf.GetNbinsX()
    hist_reducedChi2_sf.SetBinContent(lastBin, hist_reducedChi2_sf.GetBinContent(lastBin) + hist_reducedChi2_sf.GetBinContent(1+lastBin) )
    hist_reducedChi2_sf.SetBinError(lastBin, math.sqrt( hist_reducedChi2_sf.GetBinError(lastBin)*hist_reducedChi2_sf.GetBinError(lastBin) 
                                                        + hist_reducedChi2_sf.GetBinError(1+lastBin)*hist_reducedChi2_sf.GetBinError(1+lastBin) ) 
    )

    tmpmin,maxY = getMinMaxMultiHisto([hist_reducedChi2_MC, hist_reducedChi2_data, hist_reducedChi2_sf], sumError=True)
    maxY = 1.5 * maxY

    hist_reducedChi2_data.GetXaxis().SetTitleOffset(1.2)
    hist_reducedChi2_data.SetLineWidth(2)
    hist_reducedChi2_data.SetLineColor(ROOT.kRed+2)
    hist_reducedChi2_data.SetFillColor(ROOT.kRed+2)
    hist_reducedChi2_data.SetFillStyle(3003)
    hist_reducedChi2_data.GetXaxis().SetTitleSize(0.05)
    hist_reducedChi2_data.GetXaxis().SetLabelSize(0.06)
    hist_reducedChi2_data.GetYaxis().SetTitleOffset(1.15)
    hist_reducedChi2_data.GetYaxis().SetTitleSize(0.05)
    hist_reducedChi2_data.GetYaxis().SetLabelSize(0.04)
    hist_reducedChi2_data.GetYaxis().SetTitle("Events")
    hist_reducedChi2_data.GetYaxis().SetRangeUser(0,maxY)
    hist_reducedChi2_data.GetXaxis().SetTitle("#chi^{2} / NDF")
    hist_reducedChi2_data.Draw("HIST")
    lat = ROOT.TLatex()
    xmin = 0.20 
    yhi = 0.85
    lat.SetNDC();
    lat.SetTextSize(0.045);
    lat.SetTextFont(62);
    lat.DrawLatex(xmin,yhi,"")
    lat.DrawLatex(xmin,yhi-0.05,"entries")
    lat.DrawLatex(xmin,yhi-0.1,"mean")
    lat.DrawLatex(xmin,yhi-0.15,"rms")
    line1 = "= {0}".format(int(entries_data))
    line2 = "= {:.2f}".format(hist_reducedChi2_data.GetMean())
    line3 = "= {:.2f}".format(hist_reducedChi2_data.GetStdDev())
    lat.SetTextFont(42);
    lat.SetTextColor(ROOT.kRed+2);
    xmin = xmin + 0.2 
    lat.DrawLatex(xmin,yhi,"data")
    lat.DrawLatex(xmin,yhi-0.05,line1)
    lat.DrawLatex(xmin,yhi-0.1,line2)
    lat.DrawLatex(xmin,yhi-0.15,line3)
    # now MC
    hist_reducedChi2_MC.SetLineWidth(2)
    hist_reducedChi2_MC.SetLineColor(ROOT.kBlack)
    hist_reducedChi2_MC.SetFillColor(ROOT.kGray)
    #hist_reducedChi2_MC.SetFillStyle(3004)
    hist_reducedChi2_MC.Draw("HIST SAME")
    #######################
    # redraw some stuff that might be covered by FillColor
    histCopy = hist_reducedChi2_MC.DrawCopy("HIST SAME")
    histCopy.SetFillColor(0)
    histCopy.SetFillStyle(0)
    hist_reducedChi2_data.Draw("HIST SAME")
    c.RedrawAxis("sameaxis")
    #################
    line1 = "= {0}".format(int(entries_MC))
    line2 = "= {:.2f}".format(hist_reducedChi2_MC.GetMean())
    line3 = "= {:.2f}".format(hist_reducedChi2_MC.GetStdDev())
    lat.SetTextColor(ROOT.kBlack);
    xmin = xmin + 0.2
    lat.DrawLatex(xmin,yhi,"MC")
    lat.DrawLatex(xmin,yhi-0.05,line1)
    lat.DrawLatex(xmin,yhi-0.1,line2)
    lat.DrawLatex(xmin,yhi-0.15,line3)
    # now SF
    hist_reducedChi2_sf.SetLineWidth(2)
    hist_reducedChi2_sf.SetLineColor(ROOT.kBlue)
    hist_reducedChi2_sf.Draw("HIST SAME")
    #######################
    # redraw some stuff that might be covered by FillColor
    c.RedrawAxis("sameaxis")
    #################
    line1 = "= {0}".format(int(entries_sf))
    line2 = "= {:.2f}".format(hist_reducedChi2_sf.GetMean())
    line3 = "= {:.2f}".format(hist_reducedChi2_sf.GetStdDev())
    lat.SetTextColor(ROOT.kBlue);
    xmin = xmin + 0.2
    lat.DrawLatex(xmin,yhi,"SF")
    lat.DrawLatex(xmin,yhi-0.05,line1)
    lat.DrawLatex(xmin,yhi-0.1,line2)
    lat.DrawLatex(xmin,yhi-0.15,line3)
    
    for ext in ["png","pdf"]:
        c.SaveAs("{out}reducedChi2_{ch}.{ext}".format(out=outname,ch=channel,ext=ext))

############################################################################

minmaxSF = {"trigger"      : "0.65,1.15",
            "triggerplus"  : "0.65,1.15",
            "triggerminus" : "0.65,1.15",
            "idip"         : "0.95,1.01",
            "idipplus"     : "0.95,1.01",
            "idipminus"    : "0.95,1.01",
            "iso"          : "0.975,1.025",
            "isoplus"      : "0.975,1.025",
            "isominus"     : "0.975,1.025",
            "isonotrig"    : "0.97,1.03",
            # usually one doesn't smooth antiiso efficiencies, they come directly from the iso ones after smoothing
            "antiiso"      : "0.6,1.25",
            "antiisonotrig": "0.6,1.25",
            #
            "tracking"     : "0.98,1.01",
            "trackingplus" : "0.98,1.01",
            "trackingminus": "0.98,1.01",
            "reco"         : "0.94,1.02",
            "recoplus"     : "0.94,1.02",
            "recominus"    : "0.94,1.02",
}


def mergeFiles(args):

    steps = args.doSteps
    inputDir = args.outdir[0] # outdir is the input directory for the merge command
    era = args.era
    outdir = f"{inputDir}/{era}/"

    files = []
    for step in steps:
        charge = "both"
        if "plus" in step:
            charge = "plus"
            step = step.replace(charge, "")
        elif "minus" in step:
            charge = "minus"
            step = step.replace(charge, "")
        files.append(f"{inputDir}/{era}/mu_{step}_{charge}/smoothedSFandEffi_{step}_{era}_{charge}.root")
    outfile = f"{outdir}/allSmooth_{era}.root"
    mergeCmd = f"hadd -f {outfile} " + " ".join(files)
    print()
    print("Merging file with this command:")
    safeSystem(mergeCmd, args.dryRun)
    print()

def runFiles(args):

    steps = args.doSteps
    era = args.era

    print(f"---> steps = {steps}")
    print("Running these commands")
    print()
    for step in steps:
        charge = "both"
        if "plus" in step:
            charge = "plus"
            step = step.replace(charge, "")
        elif "minus" in step:
            charge = "minus"
            step = step.replace(charge, "")
        inputFile = args.inputfile[0].replace("_ERA", f"_{args.era}").replace("_STEP", f"_{step}").replace("_CHARGE", f"_{charge}")
            
        cmd = f"python w_mass_13TeV/smoothLeptonScaleFactors.py {inputFile} {args.outdir[0]} -c {charge} -s {step}"
        cmd += f" --input-hist-names '{args.inputHistNames}' --input-hist-names-alt '{args.inputHistNamesAlt}'"
        if step not in ["iso", "isonotrig", "antiiso", "antiisonotrig"]:
            cmd += " --skip-eff"
        else:
            cmd += f" --fit-pol-degree-efficiency {args.fitPolDegreeEfficiency}"
        print()
        safeSystem(cmd, args.dryRun)
        print()

    if args.doMerge:
        mergeFiles(args)

        
if __name__ == "__main__":
            
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile',  type=str, nargs=1, help='input root file with TH2')
    parser.add_argument('outdir', type=str, nargs=1, help='output directory to save things')
    parser.add_argument('-c','--charge', default='both', choices=['both', 'plus', 'minus'], type=str, help='Plus or minus if the efficiencies were derived separately for each charge. If both, assumes no charge splitting in the inputs')
    parser.add_argument('-e','--era',  dest='era',     default='GtoH', choices=['GtoH'], type=str, help='Efficiency era')
    parser.add_argument('-s','--step', dest='step', default='iso', choices=list(minmaxSF.keys()), help='Working point to smooth')
    parser.add_argument('-r','--pt-fit-range', dest='ptFitRange', type=float, nargs=2, default=[-1, -1], help='Pt range fo the fit: pass two values for min and max. If one of them (or both) is negative, the corresponding histogram range is used')
    parser.add_argument('-w','--width-pt',     dest='widthPt',default='0.2', type=float, help='Pt bin width for the smoothed histogram')
    parser.add_argument(     '--set-max-pt-histo',     dest='setMaxPtHisto', default='-1.0', type=float, help='Set upper pt for output histograms. If negative use default max from input histograms')
    parser.add_argument(    '--input-hist-names', dest='inputHistNames', default='EffData2D,EffMC2D,SF2D_nominal', type=str, help='Pass comma separated list of 3  names, for eff(data),eff(MC),SF, to be used instead of the default names')
    parser.add_argument(    '--input-hist-names-alt', dest='inputHistNamesAlt', default='EffDataAltSig2D,SF2D_dataAltSig', type=str, help='Pass comma separated list of 2  names for alternate variations, for eff(data),SF, to be used instead of the default names')
    parser.add_argument(     '--palette'  , dest='palette',      default=87, type=int, help='Set palette: default is a built-in one, 55 is kRainbow')
    parser.add_argument(     '--fit-pol-degree-efficiency'  , dest='fitPolDegreeEfficiency', default=4, type=int, help='Degree for polynomial used in the fits to efficiencies (-1 will use a spline)')
    parser.add_argument(     '--skip-eff', dest='skipEff', action="store_true", default=False, help='Skip efficiencies and do only SF (to save time and if one only wants to smooth SF directly)')
    # utility option to print commands to do all files
    parser.add_argument('-d',  '--dryRun', action='store_true', help='Do not execute commands, just print them')
    parser.add_argument(     '--run-all', dest='runAll', action="store_true", default=False, help='Make and run commands to run all steps specified in --merge-steps')
    parser.add_argument(     '--do-steps', dest='doSteps', nargs='+', default=["isonotrig", "iso", "triggerplus", "triggerminus", "idip", "tracking", "reco"], choices=list(minmaxSF.keys()), help='Working points to smooth when running --run-all or --do-merge')
    # option to merge files once they exist
    parser.add_argument(     '--do-merge', dest='doMerge', action="store_true", default=False, help='Merge efficiency files if they all exist')
    
    args = parser.parse_args()
    logger = logging.setup_logger(os.path.basename(__file__), 3, True)
    
    if args.fitPolDegreeEfficiency < -1:
        args.fitPolDegreeEfficiency = -1
        
    ROOT.TH1.SetDefaultSumw2()

    if args.runAll:
        runFiles(args)
        quit()

    if args.doMerge:
        mergeFiles(args)
        quit()

    ### TODO:
    ### options to set fit range and some histogram definition in the fit function are not always consistent.
    ### also, one should slice the boost histogram used for the fit, but it is not implemented yet
    ### If one uses the full range everything is fine, otherwise it should be checked
    if any(x > 0 for x in args.ptFitRange):
        print("WARNING: using a restricted fit range is currently not implemented.")
        quit()

    if any(x in args.step for x in ["plus", "minus"]):
        args.charge = "plus" if "plus" in args.step else "minus"
        args.step = args.step.replace(args.charge, "")
        
    channel = "mu"
    charge = "" if args.charge == "both" else "positive" if args.charge == "plus" else "negative"
    lepton = f"{charge} muon"

    outname = args.outdir[0]
    addStringToEnd(outname,"/",notAddIfEndswithMatch=True)
    outname += f"{args.era}/{channel}_{args.step}_{args.charge}/"
    createPlotDirAndCopyPhp(outname)
    
    outfilename = f"smoothedSFandEffi_{args.step}_{args.era}_{args.charge}.root"

    #########################################    
    #########################################    

    datahistname = f"effData_{args.step}_{args.era}_{args.charge}"
    mchistname   = f"effMC_{args.step}_{args.era}_{args.charge}"
    sfhistname   = f"SF2D_nominal_{args.step}_{args.era}_{args.charge}"
            
    if len(args.inputHistNames):
        datahistname,mchistname,sfhistname = args.inputHistNames.split(",")

    datahistnameAlt = f"effData_altSig_{args.step}_{args.era}_{args.charge}"
    sfhistnameAlt   = f"SF2D_dataAltSig_{args.step}_{args.era}_{args.charge}"
    if len(args.inputHistNamesAlt):
        datahistnameAlt,sfhistnameAlt = args.inputHistNamesAlt.split(",")
        
    tfile = safeOpenFile(args.inputfile[0])
    hsf =   safeGetObject(tfile, sfhistname)
    hsfAlt = safeGetObject(tfile, sfhistnameAlt)
    if args.skipEff:
        hdata = copy.deepcopy(hsf.Clone("data_dummy"))
        hdata.Reset("ICESM")
        hmc = copy.deepcopy(hdata.Clone("mc_dummy"))
        hdataAlt = copy.deepcopy(hdata.Clone("dataAlt_dummy"))
    else:
        hdata = safeGetObject(tfile, datahistname)
        hmc =   safeGetObject(tfile, mchistname)
        hdataAlt = safeGetObject(tfile, datahistnameAlt)        
    tfile.Close()
        
    etabins = [round(hdata.GetXaxis().GetBinLowEdge(i), 1) for i in range(1, 2 + hdata.GetNbinsX())]
    ptbins =  [round(hdata.GetYaxis().GetBinLowEdge(i), 1) for i in range(1, 2 + hdata.GetNbinsY())]

    # utility histogram to show what function will be used (usually we choose pol4 or pol3 for efficiencies of scale factors)
    hist_chosenFunc = ROOT.TH1D("chosenFitFunc", "Best fit function for each #eta bin for data or MC", 3, 0, 3)
    hist_chosenFunc.GetXaxis().SetBinLabel(1, "erf")
    hist_chosenFunc.GetXaxis().SetBinLabel(2, "tf1_pol3")
    hist_chosenFunc.GetXaxis().SetBinLabel(3, f"pol{args.fitPolDegreeEfficiency}_tf")
    # 
    hist_chosenFunc_SF = ROOT.TH1D("chosenFitFunc_SF", "Best fit function for each #eta bin for SF", 3, 0, 3)
    hist_chosenFunc_SF.GetXaxis().SetBinLabel(1, "tf1_cheb3")
    hist_chosenFunc_SF.GetXaxis().SetBinLabel(2, "tf1_pol2")
    hist_chosenFunc_SF.GetXaxis().SetBinLabel(3, "pol3_tf")

    hist_reducedChi2_data = ROOT.TH1D("reducedChi2_data", "Reduced #chi^{2}", 25, 0, 5)
    hist_reducedChi2_data.StatOverflows() # use underflow and overflow to compute mean and RMS
    hist_reducedChi2_MC = ROOT.TH1D("reducedChi2_MC", "Reduced #chi^{2}", 25, 0, 5)
    hist_reducedChi2_MC.StatOverflows() # use underflow and overflow to compute mean and RMS
    hist_reducedChi2_sf = ROOT.TH1D("reducedChi2_sf", "Reduced #chi^{2}", 25, 0, 5)
    hist_reducedChi2_sf.StatOverflows() # use underflow and overflow to compute mean and RMS

    ######################
    # to make ratio
    ######################
    ratioData =  ROOT.TH2D("dataEfficiencyRatio","Original/smooth Data efficiency ratio",
                           len(etabins)-1, array('d',etabins),
                           len(ptbins)-1, array('d',ptbins)
                           )
    ratioMC =  ROOT.TH2D("mcEfficiencyRatio","Original/smooth MC efficiency ratio",
                           len(etabins)-1, array('d',etabins),
                           len(ptbins)-1, array('d',ptbins)
                           )
    ratioSF =  ROOT.TH2D("scaleFactorRatio","Original/smooth scale factor ratio",
                         len(etabins)-1, array('d',etabins),
                         len(ptbins)-1, array('d',ptbins)
                         )

    copyHisto(ratioData,hdata)
    copyHisto(ratioMC,hmc)
    copyHisto(ratioSF,hsf)

    ######################
    # to make ratio
    ######################
    pullData =  ROOT.TH2D("dataEfficiencyPull","Original/smooth Data efficiency pull",
                           len(etabins)-1, array('d',etabins),
                           len(ptbins)-1, array('d',ptbins)
                           )
    pullMC =  ROOT.TH2D("mcEfficiencyPull","Original/smooth MC efficiency pull",
                           len(etabins)-1, array('d',etabins),
                           len(ptbins)-1, array('d',ptbins)
                           )
    pullSF =  ROOT.TH2D("scaleFactorPull","Original/smooth scale factor pull",
                         len(etabins)-1, array('d',etabins),
                         len(ptbins)-1, array('d',ptbins)
                         )
    pullSFfromSmoothEffi =  ROOT.TH2D("scaleFactorPullFromSmoothEffi","Original/smooth scale factor pull (SF from smooth effi)",
                                      len(etabins)-1, array('d',etabins),
                                      len(ptbins)-1, array('d',ptbins)
    )

    copyHisto(pullData, hdata)
    copyHisto(pullMC, hmc)
    copyHisto(pullSF, hsf)
    copyHisto(pullSFfromSmoothEffi, hsf)
    errData = copy.deepcopy(hdata.Clone("errData"))
    errMC   = copy.deepcopy(hmc.Clone("errMC"))
    errSF   = copy.deepcopy(hsf.Clone("errSF"))
    copyHisto(errData, hdata, copyError=True)
    copyHisto(errMC, hmc, copyError=True)
    copyHisto(errSF, hsf, copyError=True)
    
    #############
    # these will be used to check the smoothed efficiency
    ###############
    maxPtHisto = hdata.GetYaxis().GetBinLowEdge(1+hdata.GetNbinsY())
    if args.setMaxPtHisto > 0.0:
        maxPtHisto = args.setMaxPtHisto

    nFinePtBins = int(math.ceil(maxPtHisto - hdata.GetYaxis().GetBinLowEdge(1))/args.widthPt)
    minPtHisto = hdata.GetYaxis().GetBinLowEdge(1)
                               
    hdataSmoothCheck_origBinPt = ROOT.TH2D("hdataSmoothCheck_origBinPt","Data smoothed efficiency",
                                           len(etabins)-1, array('d',etabins),
                                           len(ptbins)-1, array('d',ptbins)
                                           )
    hmcSmoothCheck_origBinPt = ROOT.TH2D("hmcSmoothCheck_origBinPt","MC smoothed efficiency",
                                         len(etabins)-1, array('d',etabins),
                                         len(ptbins)-1, array('d',ptbins)
                                         )
    hsfSmoothCheck_origBinPt = ROOT.TH2D("hsfSmoothCheck_origBinPt","Data/MC smoothed scale factor",
                                         len(etabins)-1, array('d',etabins),
                                         len(ptbins)-1, array('d',ptbins)
                                         )

    ## prepare histograms with nominal eta-pt efficiency or scale factor, and other variations on the Z axis
    ## will save both up and down variations for effStat and (currently) only 1 syst, so have 1+2*nPar+1 bins
    ## assume we have 48 eta bins to use constructor of TH3
    nBinsEff = 2 + 2 * (1 + args.fitPolDegreeEfficiency) # fits with pol4 (5 parameters), then nominal in first bin and syst in last
    hist_effData_nomiAndAlt_etapt = ROOT.TH3D("hist_effData_nomiAndAlt_etapt",
                                              "Smooth nominal and alternate data efficiency",
                                              48,-2.40,2.40,
                                              nFinePtBins, minPtHisto, maxPtHisto,
                                              nBinsEff,0.5,0.5+nBinsEff)
    hist_effMC_nomiAndAlt_etapt = ROOT.TH3D("hist_effMC_nomiAndAlt_etapt",
                                              "Smooth nominal and alternate MC efficiency",
                                              48,-2.40,2.40,
                                              nFinePtBins, minPtHisto, maxPtHisto,
                                              nBinsEff,0.5,0.5+nBinsEff)
    # TODO: avoid magic numbers, use option as for efficiencies
    nBinsSF = 8 if args.step == "tracking" else 10 # fits are done with pol2 for tracking (3 parameters) and pol3 otherwise (4 parameters), then this is 1 + 2 * nBinsSF + 1
    hist_SF_nomiAndAlt_etapt = ROOT.TH3D("hist_SF_nomiAndAlt_etapt",
                                         "Smooth nominal and alternate scale factor",
                                         48,-2.40,2.40,
                                         nFinePtBins, minPtHisto, maxPtHisto,
                                         nBinsSF,0.5,0.5+nBinsSF)

    # hmc and hdata have eta on X and pt on Y
    # we select slices at constant eta and fit along pt with some function

    label = args.step + (args.charge if args.charge != "both" else "")
    hsfpt = make1Dhist("hsfpt", hsf, ptbins, label)
    hsfptAlt = make1Dhist("hsfptAlt", hsfAlt, ptbins, label)

    outfolder_eigenVars = f"{outname}/eigenDecomposition/" 

    zaxisRange = ""
    zaxisRangeSF = "::" + minmaxSF[args.step]

    if not args.skipEff:
        hmcpt = make1Dhist("hmcpt", hmc, ptbins, label)
        ###########################
        # first MC
        ###########################
        for key in hmcpt:
            bestFitFunc = fitTurnOnTF(hmcpt[key], key, outname, "MC",channel=channel,
                                      hist_chosenFunc=hist_chosenFunc, 
                                      step=args.step,
                                      fitRange=args.ptFitRange,
                                      hist_reducedChi2=hist_reducedChi2_MC,
                                      #hist_FuncParam_vs_eta=hist_FuncParam_vs_eta_mc,
                                      #hist_FuncCovMatrix_vs_eta=hist_FuncCovMatrix_vs_eta_mc,
                                      charge=args.charge,
                                      etabins=etabins,
                                      widthPtSmooth=args.widthPt,
                                      hist_nomiAndAlt_etapt=hist_effMC_nomiAndAlt_etapt,
                                      efficiencyFitPolDegree=args.fitPolDegreeEfficiency

            )
            for ipt in range(1, hmcSmoothCheck_origBinPt.GetNbinsY()+1):
                ptval = hmcSmoothCheck_origBinPt.GetYaxis().GetBinCenter(ipt)
                hmcSmoothCheck_origBinPt.SetBinContent(key+1, ipt, bestFitFunc.Eval(ptval))
                hmcSmoothCheck_origBinPt.SetBinError(  key+1, ipt, hmc.GetBinError(key+1, ipt))
        hmcSmoothCheck = getTH2fromTH3(hist_effMC_nomiAndAlt_etapt, "hmcSmoothCheck", 1, 1)            
        hmcSmoothCheck.SetTitle("Smooth MC efficiency")
        ###########################
        # now data
        ###########################
        hdatapt = make1Dhist("hdatapt", hdata, ptbins, label)
        hdataptAlt = make1Dhist("hdataptAlt", hdataAlt, ptbins, label)
        for key in hdatapt:

            bestFitFunc = fitTurnOnTF(hdatapt[key],key,outname, "Data",channel=channel,hist_chosenFunc=hist_chosenFunc, 
                                      step=args.step,
                                      fitRange=args.ptFitRange,
                                      hist_reducedChi2=hist_reducedChi2_data,
                                      #hist_FuncParam_vs_eta=hist_FuncParam_vs_eta_data,
                                      #hist_FuncCovMatrix_vs_eta=hist_FuncCovMatrix_vs_eta_data,
                                      charge=args.charge,
                                      etabins=etabins,
                                      widthPtSmooth=args.widthPt,
                                      hist_nomiAndAlt_etapt=hist_effData_nomiAndAlt_etapt,
                                      histAlt=hdataptAlt[key],
                                      efficiencyFitPolDegree=args.fitPolDegreeEfficiency

            )
            for ipt in range(1,hdataSmoothCheck_origBinPt.GetNbinsY()+1):
                ptval = hdataSmoothCheck_origBinPt.GetYaxis().GetBinCenter(ipt)
                hdataSmoothCheck_origBinPt.SetBinContent(key+1, ipt, bestFitFunc.Eval(ptval))
                hdataSmoothCheck_origBinPt.SetBinError(  key+1, ipt, hdata.GetBinError(key+1, ipt))
        hdataSmoothCheck = getTH2fromTH3(hist_effData_nomiAndAlt_etapt, "hdataSmoothCheck", 1, 1)
        hdataSmoothCheck.SetTitle("Smooth data efficiency")

        # make scale factor: data/MC
        scaleFactor = copy.deepcopy(hdataSmoothCheck.Clone("scaleFactor"))
        scaleFactor.SetTitle("Scale factor from smoothed efficiencies")
        scaleFactor.Divide(hmcSmoothCheck)
        scaleFactor.SetMinimum(scaleFactor.GetBinContent(scaleFactor.GetMinimumBin()))
        scaleFactor.SetMaximum(scaleFactor.GetBinContent(scaleFactor.GetMaximumBin()))

    # ###########################
    # # now direct SF smoothing
    # ###########################
    for key in hsfpt:
        smoothSFfromEffiTMP = None
        if not args.skipEff:
            # this is to compare direct SF smoothing with efficiency smoothing
            smoothSFfromEffiTMP = scaleFactor.ProjectionY(f"{args.step}TMP_{key}", key+1, key+1, "e")
        bestFitFunc = fitTurnOnTF(hsfpt[key],key,outname, "SF",channel=channel,hist_chosenFunc=hist_chosenFunc_SF, 
                                  step=args.step,
                                  fitRange=args.ptFitRange,
                                  hist_reducedChi2=hist_reducedChi2_sf,
                                  #hist_FuncParam_vs_eta=hist_FuncParam_vs_eta_sf,
                                  #hist_FuncCovMatrix_vs_eta=hist_FuncCovMatrix_vs_eta_sf,
                                  charge=args.charge,
                                  etabins=etabins,
                                  widthPtSmooth=args.widthPt,
                                  hist_nomiAndAlt_etapt=hist_SF_nomiAndAlt_etapt,
                                  histAlt=hsfptAlt[key],
                                  addCurve=smoothSFfromEffiTMP,
                                  addCurveLegEntry=f"SF from pol{args.fitPolDegreeEfficiency} effi"
        )
        for ipt in range(1,hsfSmoothCheck_origBinPt.GetNbinsY()+1):
            ptval = hsfSmoothCheck_origBinPt.GetYaxis().GetBinCenter(ipt)
            hsfSmoothCheck_origBinPt.SetBinContent(key+1,ipt, bestFitFunc.Eval(ptval))
            hsfSmoothCheck_origBinPt.SetBinError(  key+1, ipt, hsf.GetBinError(key+1, ipt))
    hsfSmoothCheck = getTH2fromTH3(hist_SF_nomiAndAlt_etapt, "hsfSmoothCheck", 1, 1)
    hsfSmoothCheck.SetTitle("Smooth scale factor")
    #################################
    # start to make plots
    #################################

    canvas = ROOT.TCanvas("canvas","",700,625)

    # plot eigen variations
    if not args.skipEff:
        nvars = int((hist_effMC_nomiAndAlt_etapt.GetNbinsZ() - 2) / 2)
        for iv in range(nvars):
            binVar = 2 + iv
            # MC
            hvar = getTH2fromTH3(hist_effMC_nomiAndAlt_etapt, f"effStatVar_p{iv}_effMC",binVar, binVar)
            hvar.SetTitle(f"Eff. stat. nuisance p{iv}")
            hvar.Add(hmcSmoothCheck, -1.0)
            drawCorrelationPlot(hvar, "{lep} #eta".format(lep=lepton), "{lep} p_{{T}} [GeV]".format(lep=lepton), "Alternate - nominal (MC efficiency)",
                                hvar.GetName(), "ForceTitle", outfolder_eigenVars,
                                palette=args.palette, passCanvas=canvas)
            # data
            hvar = getTH2fromTH3(hist_effData_nomiAndAlt_etapt, f"effStatVar_p{iv}_effData",binVar, binVar)
            hvar.SetTitle(f"Eff. stat. nuisance p{iv}")
            hvar.Add(hdataSmoothCheck, -1.0)
            drawCorrelationPlot(hvar, "{lep} #eta".format(lep=lepton), "{lep} p_{{T}} [GeV]".format(lep=lepton), "Alternate - nominal (data efficiency)",
                                hvar.GetName(), "ForceTitle", outfolder_eigenVars,
                                palette=args.palette, passCanvas=canvas)
        # syst only for data
        systBin = hist_effData_nomiAndAlt_etapt.GetNbinsZ()
        hvar = getTH2fromTH3(hist_effData_nomiAndAlt_etapt, "effSystVar_effData", systBin, systBin)
        hvar.SetTitle("Eff. syst. nuisance")
        hvar.Add(hdataSmoothCheck, -1.0)
        drawCorrelationPlot(hvar, "{lep} #eta".format(lep=lepton), "{lep} p_{{T}} [GeV]".format(lep=lepton), "Alternate - nominal (data efficiency)",
                            hvar.GetName(), "ForceTitle", outfolder_eigenVars,
                            palette=args.palette, passCanvas=canvas)

            
    # SF
    nvars = int((hist_SF_nomiAndAlt_etapt.GetNbinsZ() - 2) / 2)
    for iv in range(nvars):
        binVar = 2 + iv
        hvar = getTH2fromTH3(hist_SF_nomiAndAlt_etapt, f"effStatVar_p{iv}_SF",binVar, binVar)
        hvar.SetTitle(f"Eff. stat. nuisance p{iv}")
        hvar.Add(hsfSmoothCheck, -1.0)
        drawCorrelationPlot(hvar, "{lep} #eta".format(lep=lepton), "{lep} p_{{T}} [GeV]".format(lep=lepton), "Alternate - nominal (scale factor)",
                            hvar.GetName(), "ForceTitle", outfolder_eigenVars,
                            palette=args.palette, passCanvas=canvas)
    #
    systBin = hist_SF_nomiAndAlt_etapt.GetNbinsZ()
    hvar = getTH2fromTH3(hist_SF_nomiAndAlt_etapt, "effSystVar_SF", systBin, systBin)
    hvar.SetTitle("Eff. syst. nuisance")
    hvar.Add(hsfSmoothCheck, -1.0)
    drawCorrelationPlot(hvar, "{lep} #eta".format(lep=lepton), "{lep} p_{{T}} [GeV]".format(lep=lepton), "Alternate - nominal (data efficiency)",
                        hvar.GetName(), "ForceTitle", outfolder_eigenVars,
                        palette=args.palette, passCanvas=canvas)

    # plot original histograms
    drawCorrelationPlot(hmc,"{lep} #eta".format(lep=lepton),"{lep} p_{{T}} [GeV]".format(lep=lepton),"MC efficiency%s" % zaxisRange,
                        "inputEfficiency_MC","",outname,palette=args.palette,passCanvas=canvas)
    drawCorrelationPlot(hdata,"{lep} #eta".format(lep=lepton),"{lep} p_{{T}} [GeV]".format(lep=lepton),"Data efficiency%s" % zaxisRange,
                        "inputEfficiency_Data","",outname,palette=args.palette,passCanvas=canvas)
    drawCorrelationPlot(hsf,"{lep} #eta".format(lep=lepton),"{lep} p_{{T}} [GeV]".format(lep=lepton),"Data/MC scale factor%s" % zaxisRangeSF,
                        "inputScaleFactor","",outname,palette=args.palette,passCanvas=canvas)

    # now the smoothed ones
    drawCorrelationPlot(hsfSmoothCheck,"{lep} #eta".format(lep=lepton),"{lep} p_{{T}} [GeV]".format(lep=lepton),"Data/MC smoothed scale factor%s" % zaxisRangeSF,
                        "smoothScaleFactorDirectly","ForceTitle",outname,palette=args.palette,passCanvas=canvas)
    if not args.skipEff:
        # plot smooth efficiencies and SF made from them
        drawCorrelationPlot(hmcSmoothCheck,"{lep} #eta".format(lep=lepton),"{lep} p_{{T}} [GeV]".format(lep=lepton),"MC smoothed efficiency%s" % zaxisRange,
                            "smoothEfficiency_MC","ForceTitle",outname,palette=args.palette,passCanvas=canvas)
        drawCorrelationPlot(hdataSmoothCheck,"{lep} #eta".format(lep=lepton),"{lep} p_{{T}} [GeV]".format(lep=lepton),"Data smoothed efficiency%s" % zaxisRange,
                            "smoothEfficiency_Data","ForceTitle",outname,palette=args.palette,passCanvas=canvas)
        drawCorrelationPlot(scaleFactor,
                            "{lep} #eta".format(lep=lepton),"{lep} p_{{T}} [GeV]".format(lep=lepton),
                            "Data/MC scale factor%s" % zaxisRangeSF,
                            "smoothScaleFactor","ForceTitle",outname,palette=args.palette,passCanvas=canvas)

    #################################
    # plot also with original binning
    ################################

    # divide before drawing the denominator, whose axis settings are modified by drawCorrelationPlot and seems to affect the ratio as well if divided afterwards
    ratioData.Divide(hdataSmoothCheck_origBinPt)
    ratioMC.Divide(hmcSmoothCheck_origBinPt)
    
    drawCorrelationPlot(hmcSmoothCheck_origBinPt,"{lep} #eta".format(lep=lepton),"{lep} p_{{T}} [GeV]".format(lep=lepton),"MC smoothed efficiency%s" % zaxisRange,
                        "smoothEfficiency_MC_origBinPt","ForceTitle",outname,palette=args.palette,passCanvas=canvas)
    drawCorrelationPlot(hdataSmoothCheck_origBinPt,"{lep} #eta".format(lep=lepton),"{lep} p_{{T}} [GeV]".format(lep=lepton),"Data smoothed efficiency%s" % zaxisRange,
                        "smoothEfficiency_Data_origBinPt","ForceTitle",outname,palette=args.palette,passCanvas=canvas)
    drawCorrelationPlot(hsfSmoothCheck_origBinPt,"{lep} #eta".format(lep=lepton),"{lep} p_{{T}} [GeV]".format(lep=lepton),"Data/MC smoothed scale factor%s" % zaxisRangeSF,
                        "smoothScaleFactorDirectly_origBinPt","ForceTitle",outname,palette=args.palette,passCanvas=canvas)

    # scale factor: data/MC
    scaleFactor_origBinPt = ROOT.TH2D("scaleFactor_origBinPt","Scale factor",
                                      len(etabins)-1, array('d',etabins),
                                      len(ptbins)-1, array('d',ptbins)
                                      )
    copyHisto(scaleFactor_origBinPt, hdataSmoothCheck_origBinPt)
    scaleFactor_origBinPt.Divide(hmcSmoothCheck_origBinPt)
    scaleFactor_origBinPt.SetMinimum(scaleFactor_origBinPt.GetBinContent(scaleFactor_origBinPt.GetMinimumBin()))
    scaleFactor_origBinPt.SetMaximum(scaleFactor_origBinPt.GetBinContent(scaleFactor_origBinPt.GetMaximumBin()))

    # to make ratio, divide before passing to function, to avoid changes in the histogram
    ratioSF.Divide(scaleFactor_origBinPt)

    #scaleFactor_origBinPt.GetZaxis().SetTitle("Data/MC scale factor")
    drawCorrelationPlot(scaleFactor_origBinPt,"{lep} #eta".format(lep=lepton),"{lep} p_{{T}} [GeV]".format(lep=lepton),"Data/MC scale factor%s" % zaxisRangeSF,
                        "smoothScaleFactor_origBinPt","ForceTitle",outname,palette=args.palette,passCanvas=canvas)


    ######################
    # finally SF(smooth)/SF(original)
    ######################
    drawCorrelationPlot(ratioData,"{lep} #eta".format(lep=lepton),"{lep} p_{{T}} [GeV]".format(lep=lepton),"Data efficiency ratio (original/smooth)::0.99,1.01",
                        "dataEfficiencyRatio","ForceTitle",outname,palette=args.palette,passCanvas=canvas)
    drawCorrelationPlot(ratioMC,"{lep} #eta".format(lep=lepton),"{lep} p_{{T}} [GeV]".format(lep=lepton),"MC efficiency ratio (original/smooth)::0.99,1.01",
                        "mcEfficiencyRatio","ForceTitle",outname,palette=args.palette,passCanvas=canvas)
    drawCorrelationPlot(ratioSF,"{lep} #eta".format(lep=lepton),"{lep} p_{{T}} [GeV]".format(lep=lepton),"scale factor ratio (original/smooth)::0.99,1.01",
                        "scaleFactorRatio","ForceTitle",outname,palette=args.palette,passCanvas=canvas)

    pullData.Add(hdataSmoothCheck_origBinPt, -1.0)
    pullData.Divide(errData)
    pullMC.Add(hmcSmoothCheck_origBinPt, -1.0)
    pullMC.Divide(errMC)
    pullSF.Add(hsfSmoothCheck_origBinPt, -1.0)
    pullSF.Divide(errSF)
    drawCorrelationPlot(pullData,"{lep} #eta".format(lep=lepton),"{lep} p_{{T}} [GeV]".format(lep=lepton),"Data eff. pull (original-smooth)/err::-5.0,5.0",
                        "dataEfficiencyPull","ForceTitle",outname,nContours=10,palette=args.palette,passCanvas=canvas)
    drawCorrelationPlot(pullMC,"{lep} #eta".format(lep=lepton),"{lep} p_{{T}} [GeV]".format(lep=lepton),"MC eff. pull (original-smooth)/err::-5.0,5.0",
                        "mcEfficiencyPull","ForceTitle",outname,nContours=10,palette=args.palette,passCanvas=canvas)
    drawCorrelationPlot(pullSF,"{lep} #eta".format(lep=lepton),"{lep} p_{{T}} [GeV]".format(lep=lepton),"scale factor pull (original-smoothDirectly)/err::-5.0,5.0",
                        "scaleFactorPull","ForceTitle",outname,nContours=10,palette=args.palette,passCanvas=canvas)
    if not args.skipEff:
        # add also ratio of ratioData and ratioMC, because in each there might be trend and we want to see if in the double ratio they would cancel
        doubleRatioDataMC = copy.deepcopy(ratioData.Clone("doubleRatioDataMC"))
        doubleRatioDataMC.SetTitle("(smooth/binned)_{data} / (smooth/binned)_{MC}")
        doubleRatioDataMC.Divide(ratioMC)
        drawCorrelationPlot(doubleRatioDataMC,"{lep} #eta".format(lep=lepton),"{lep} p_{{T}} [GeV]".format(lep=lepton),
                            "Double ratio",
                            "doubleRatio_smoothOverBinnedEffi_DataAndMC","ForceTitle",
                            outname,palette=args.palette,passCanvas=canvas)
        # additional pulls
        pullSFfromSmoothEffi.Add(scaleFactor_origBinPt, -1.0)
        pullSFfromSmoothEffi.Divide(errSF)
        drawCorrelationPlot(pullSFfromSmoothEffi,"{lep} #eta".format(lep=lepton),"{lep} p_{{T}} [GeV]".format(lep=lepton),"scale factor pull (original-smoothFromEffi)/err::-5.0,5.0",
                        "scaleFactorPull_SFfromSmoothEffi","ForceTitle",outname,nContours=10,palette=args.palette,passCanvas=canvas)

        SFpullRatio = copy.deepcopy(pullSFfromSmoothEffi.Clone("scaleFactorPullRatio_SFfromSmoothEffiOverDirectSmoothing"))
        SFpullRatio.Divide(pullSF)
        SFpullRatio.SetTitle("")
        drawCorrelationPlot(SFpullRatio, "{lep} #eta".format(lep=lepton),"{lep} p_{{T}} [GeV]".format(lep=lepton),
                            "SF pull ratio: smoothFromEffi / directSmoothing",
                            SFpullRatio.GetName(), "ForceTitle",outname,nContours=10,palette=args.palette,passCanvas=canvas)
    # ######################
    # # See the difference between smoothing Data and MC efficiency and taking the ratio or smoothing directly the efficiency ratio
    # ######################    
    ratioSF_smoothNumDen_smoothRatio = ROOT.TH2D("ratioSF_smoothNumDen_smoothRatio","SF ratio: smooth eff or ratio directly",
                                                 len(etabins)-1, array('d',etabins),
                                                 len(ptbins)-1, array('d',ptbins)
    )

    copyHisto(ratioSF_smoothNumDen_smoothRatio,scaleFactor_origBinPt)
    ratioSF_smoothNumDen_smoothRatio.Divide(hsfSmoothCheck_origBinPt)
    drawCorrelationPlot(ratioSF_smoothNumDen_smoothRatio,"{lep} #eta".format(lep=lepton),"{lep} p_{{T}} [GeV]".format(lep=lepton),
                        "SF ratio: smooth eff or ratio directly::0.999,1.001",
                        "ratioSF_smoothNumDen_smoothRatio","ForceTitle",outname,palette=args.palette,passCanvas=canvas)

    # same as ratioSF_smoothNumDen_smoothRatio but with fine pt binning
    if not args.skipEff:
        ratioSF_smoothEffiOverSmoothDirectly = copy.deepcopy(hdataSmoothCheck.Clone("ratioSF_smoothEffiOverSmoothDirectly"))
        ratioSF_smoothEffiOverSmoothDirectly.SetTitle("SF ratio: smooth eff or ratio directly")
        ratioSF_smoothEffiOverSmoothDirectly.Divide(hmcSmoothCheck)
        ratioSF_smoothEffiOverSmoothDirectly.Divide(hsfSmoothCheck)
        drawCorrelationPlot(ratioSF_smoothEffiOverSmoothDirectly,"{lep} #eta".format(lep=lepton),"{lep} p_{{T}} [GeV]".format(lep=lepton),
                            "SF ratio: smooth eff or ratio directly::0.999,1.001",
                            "ratioSF_smoothEffiOverSmoothDirectly","ForceTitle",outname,palette=args.palette,passCanvas=canvas)
        
        
    c = ROOT.TCanvas("c","",700,700)
    c.SetTickx(1)
    c.SetTicky(1)
    c.cd()
    c.SetLeftMargin(0.14)
    c.SetRightMargin(0.06)
    c.cd()
    if not args.skipEff:
        quickPlotTH1(c, hist_chosenFunc, outname, channel)
    quickPlotTH1(c, hist_chosenFunc_SF, outname, channel, postfix="SF")

    # draw the chi2 plots
    drawReducedChi2(c, hist_reducedChi2_data, hist_reducedChi2_MC, hist_reducedChi2_sf)

    hist_postfix = f"_{args.era}_{args.step}_{args.charge}"
    # now get SF dividing efficiencies, but for the variations use either nominal data or nominal MC, will end up with two SF histograms
    # cannot get scale factors variations from ratio of corresponding efficiencies, because data and MC variations are totally independent
    # once this is done, make antiisolation histograms
    hasSpecialHistForIsolation = False
    if args.step in ["iso", "isonotrig"] and not args.skipEff:
        hasSpecialHistForIsolation = True
        # broadcast nominal MC efficiency from TH2 into TH3
        h3_effMC_nominal = copy.deepcopy(hist_effMC_nomiAndAlt_etapt.Clone("effMC_broadcast"))
        ROOT.wrem.broadCastTH2intoTH3(h3_effMC_nominal, hmcSmoothCheck)
        # create SF with variations from data efficiency only
        hist_SF_nomiAndAlt_onlyDataVar_etapt = copy.deepcopy(hist_effData_nomiAndAlt_etapt.Clone("SF_nomiAndAlt_onlyDataVar" + hist_postfix))
        hist_SF_nomiAndAlt_onlyDataVar_etapt.SetTitle("Smooth nominal and alternate SF (only data eff variations)")
        hist_SF_nomiAndAlt_onlyDataVar_etapt.Divide(h3_effMC_nominal)
        # create SF with variations from MC efficiency only  
        hist_SF_nomiAndAlt_onlyMCVar_etapt = copy.deepcopy(hist_effData_nomiAndAlt_etapt.Clone("SF_nomiAndAlt_onlyMCVar" + hist_postfix))
        hist_SF_nomiAndAlt_onlyMCVar_etapt.SetTitle("Smooth nominal and alternate SF (only MC eff variations)")
        # broadcast nominal data efficiency from TH2 into TH3
        ROOT.wrem.broadCastTH2intoTH3(hist_SF_nomiAndAlt_onlyMCVar_etapt, hdataSmoothCheck)
        hist_SF_nomiAndAlt_onlyMCVar_etapt.Divide(hist_effMC_nomiAndAlt_etapt)
        #
        # now antiisolation
        #
        hist_postfix_anti = f"_{args.era}_anti{args.step}_{args.charge}"
        # clone histogram for effData and initialize to 1 before subtraction
        hanti_effData = copy.deepcopy(hist_effData_nomiAndAlt_etapt.Clone("effData_nomiAndAlt" + hist_postfix_anti))
        hanti_effData.Reset("ICESM")
        hanti_effData.SetTitle("Smooth nominal and alternate data efficiency")
        ROOT.wrem.initializeRootHistogram(hanti_effData, 1.0);
        # clone into MC histogram so it is also initialized to 1
        hanti_effMC = copy.deepcopy(hanti_effData.Clone("effMC_nomiAndAlt" + hist_postfix_anti))
        hanti_effMC.SetTitle("Smooth nominal and alternate MC efficiency")
        # subtract efficiencies for isolation
        hanti_effData.Add(hist_effData_nomiAndAlt_etapt, -1.0)
        hanti_effMC.Add(hist_effMC_nomiAndAlt_etapt, -1.0)
        #
        # now prepare scale factors
        # get nominal MC antiiso efficiency and broadcast into a TH3 compatible with the final histogram
        hanti_effMC_nomi_3D = copy.deepcopy(hanti_effMC.Clone("hanti_effMC_nomi_3D"))
        ROOT.wrem.broadCastTH2intoTH3(hanti_effMC_nomi_3D, ROOT.wrem.projectTH2FromTH3(hanti_effMC, "hanti_effMC_nomi_2D", 1))
        # clone histogram antiiso data efficiency and its variations, to be divided by nominal antiiso MC efficiency 
        hanti_SF_nomiAndAlt_onlyDataVar_etapt = copy.deepcopy(hanti_effData.Clone("SF_nomiAndAlt_onlyDataVar" + hist_postfix_anti))
        hanti_SF_nomiAndAlt_onlyDataVar_etapt.SetTitle("Smooth nominal and alternate SF (only data eff variations)")
        hanti_SF_nomiAndAlt_onlyDataVar_etapt.Divide(hanti_effMC_nomi_3D)
        # repeat for MC variations, cloning nominal antiisolation data efficiency, broadcasting into TH3, and finally dividing by MC efficiency
        hanti_SF_nomiAndAlt_onlyMCVar_etapt = copy.deepcopy(hanti_effData.Clone("SF_nomiAndAlt_onlyMCVar" + hist_postfix_anti))
        hanti_SF_nomiAndAlt_onlyMCVar_etapt.SetTitle("Smooth nominal and alternate SF (only MC eff variations)")
        ROOT.wrem.broadCastTH2intoTH3(hanti_SF_nomiAndAlt_onlyMCVar_etapt, ROOT.wrem.projectTH2FromTH3(hanti_effData, "hanti_effData_nomi_2D", 1))
        hanti_SF_nomiAndAlt_onlyMCVar_etapt.Divide(hanti_effMC)
        # for completeness make also original antiiso efficiencies and scale factors
        hanti_effData_original = copy.deepcopy(hdata.Clone("effData_original" + hist_postfix_anti))
        ROOT.wrem.initializeRootHistogram(hanti_effData_original, 1.0)
        hanti_effMC_original = copy.deepcopy(hanti_effData_original.Clone("effMC_original" + hist_postfix_anti))
        hanti_effData_original.Add(hdata, -1.0)
        hanti_effMC_original.Add(hmc, -1.0)
        hanti_SF_original = copy.deepcopy(hanti_effData_original.Clone("SF_original" + hist_postfix_anti))
        hanti_SF_original.Divide(hanti_effMC_original)
        # as the last thing, make smoothed antiiso efficiencies and scale factors with the original pt binning (uncertainties from original binned things)
        hanti_hdataSmoothCheck_origBinPt = copy.deepcopy(hdataSmoothCheck_origBinPt.Clone("effData_smoothWithOriginalPtBins" + hist_postfix_anti))
        ROOT.wrem.initializeRootHistogram(hanti_hdataSmoothCheck_origBinPt, 1.0)
        hanti_hmcSmoothCheck_origBinPt = copy.deepcopy(hanti_hdataSmoothCheck_origBinPt.Clone("effMC_smoothWithOriginalPtBins" + hist_postfix_anti))
        hanti_hdataSmoothCheck_origBinPt.Add(hdataSmoothCheck_origBinPt, -1.0)
        hanti_hmcSmoothCheck_origBinPt.Add(hmcSmoothCheck_origBinPt, -1.0)
        # for the SF have to divide the smooth efficiencies, cannot trivially invert the isolation sf
        hanti_hsfSmoothCheck_origBinPt = copy.deepcopy(hanti_hdataSmoothCheck_origBinPt.Clone("SF_smoothWithOriginalPtBins" + hist_postfix_anti))
        hanti_hsfSmoothCheck_origBinPt.Divide(hanti_hmcSmoothCheck_origBinPt)
        # now the original dataAltSig for SF
        hanti_SF_originalDataAltSig = copy.deepcopy(hanti_effData_original.Clone("SF_originalDataAltSig" + hist_postfix_anti))
        ROOT.wrem.initializeRootHistogram(hanti_SF_originalDataAltSig, 1.0)
        hanti_SF_originalDataAltSig.Add(hdataAlt, -1.0)
        hanti_SF_originalDataAltSig.Divide(hanti_effMC_original)

        ### now do some antiiso SF direct smoothing, and compared with outcome of iso efficiency smoothing
        label_anti = "anti" + label
        hsfpt_anti = make1Dhist("hsfpt_anti", hanti_SF_original, ptbins, label_anti)
        hsfptAlt_anti = make1Dhist("hsfptAlt_anti", hanti_SF_originalDataAltSig, ptbins, label_anti)
        # get antiiso SF from smooth iso efficiencies, with fine pt bins
        nomiAntiisoSFfromSmoothIsoEffi = getTH2fromTH3(hanti_SF_nomiAndAlt_onlyMCVar_etapt, "nomiAntiisoSFfromSmoothIsoEffi", 1, 1)
        # ###########################
        # # now SF
        # ###########################
        for key in hsfpt_anti:
            antiisoTMP = nomiAntiisoSFfromSmoothIsoEffi.ProjectionY(f"antiisoTMP_{key}", key+1, key+1, "e")
            bestFitFunc = fitTurnOnTF(hsfpt_anti[key],key, outname+f"/anti{args.step}_test", "SF", 
                                      step="anti"+args.step,
                                      fitRange=args.ptFitRange,
                                      charge=args.charge,
                                      etabins=etabins,
                                      widthPtSmooth=args.widthPt,
                                      histAlt=hsfptAlt_anti[key],
                                      addCurve=antiisoTMP,
                                      addCurveLegEntry=f"SF from pol{args.fitPolDegreeEfficiency} iso effi"
            )
            
        
    ###########################
    # Now save things
    ###########################
    tfile = ROOT.TFile.Open(outname+outfilename,'recreate')
    hsf.Write("SF_original" + hist_postfix)
    hdata.Write("effData_original" + hist_postfix)
    hmc.Write("effMC_original" + hist_postfix)
    hsfAlt.Write("SF_originalDataAltSig" + hist_postfix)
    hsfSmoothCheck_origBinPt.Write("SF_smoothWithOriginalPtBins" + hist_postfix) # this comes from direct SF smoothing (the one from efficiencies is named SF_fromSmoothEfficiencyRatio_... )
    if not args.skipEff:
        hdataSmoothCheck_origBinPt.Write("effData_smoothWithOriginalPtBins" + hist_postfix)
        hmcSmoothCheck_origBinPt.Write("effMC_smoothWithOriginalPtBins" + hist_postfix)
        scaleFactor.Write("SF_fromSmoothEfficiencyRatio" + hist_postfix)
        #ratioSF_smoothNumDen_smoothRatio.Write(ratioSF_smoothNumDen_smoothRatio.GetName() + hist_postfix)
        ratioSF_smoothEffiOverSmoothDirectly.Write(ratioSF_smoothEffiOverSmoothDirectly.GetName() + hist_postfix)
        hist_effData_nomiAndAlt_etapt.Write("effData_nomiAndAlt" + hist_postfix)
        hist_effMC_nomiAndAlt_etapt.Write("effMC_nomiAndAlt" + hist_postfix)
    hist_SF_nomiAndAlt_etapt.Write("SF_nomiAndAlt" + hist_postfix)
    if hasSpecialHistForIsolation:
        hist_SF_nomiAndAlt_onlyDataVar_etapt.Write()
        hist_SF_nomiAndAlt_onlyMCVar_etapt.Write()
        #hist_SF_nomiAndAlt_dataMCVar_etapt.Write()
        hanti_effData.Write()
        hanti_effMC.Write()
        hanti_SF_nomiAndAlt_onlyDataVar_etapt.Write()
        hanti_SF_nomiAndAlt_onlyMCVar_etapt.Write()
        #hanti_SF_nomiAndAlt_dataMCVar_etapt.Write()
        hanti_hdataSmoothCheck_origBinPt.Write()
        hanti_hmcSmoothCheck_origBinPt.Write()
        hanti_hsfSmoothCheck_origBinPt.Write() # equivalent of scaleFactor histogram from ratio of smooth efficiencies (smoothing SF directly for antiiso doesn't make sense)
        hanti_effData_original.Write()
        hanti_effMC_original.Write()
        hanti_SF_original.Write()
        hanti_SF_originalDataAltSig.Write()
    tfile.Close()
    print()
    print(f"Created file {outname+outfilename}")
    print()

    with open(outname+outfilename.replace(".root", ".txt"), "w+") as outf:
        outf.write("="*30 + "\n")
        outf.write("Summary of bad fits (Erf for data/MC and pol3 for SF)\n")
        outf.write("="*30 + "\n")
        outf.write("### Bad fit status (Data/MC/SF,  key,  fitstatus)\n")
        for key in sorted(badFitsID_data.keys()):
            outf.write(f"DATA  {key}  {badFitsID_data[key]}\n")
        for key in sorted(badFitsID_mc.keys()):
            outf.write(f"MC    {key}  {badFitsID_mc[key]}\n")
        for key in sorted(badFitsID_sf.keys()):
            outf.write(f"SF    {key}  {badFitsID_sf[key]}\n")
        outf.write("-"*30 + "\n")
        outf.write("### Bad covariance matrix status (Data/MC/SF,  key,  covquality)\n")
        for key in sorted(badCovMatrixID_data.keys()):
            outf.write(f"DATA  {key}  {badCovMatrixID_data[key]}\n")
        for key in sorted(badCovMatrixID_mc.keys()):
            outf.write(f"MC  {key}  {badCovMatrixID_mc[key]}\n")
        for key in sorted(badCovMatrixID_sf.keys()):
            outf.write(f"SF  {key}  {badCovMatrixID_sf[key]}\n")
        outf.seek(0)
        print(outf.read())
        

