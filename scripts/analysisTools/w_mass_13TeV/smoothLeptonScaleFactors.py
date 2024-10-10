#!/usr/bin/env python3

# run a single step (e.g. iso)
# python w_mass_13TeV/smoothLeptonScaleFactors.py /path/to/tnp/efficiencies_GtoH/mu_iso_both/allEfficiencies_2D.root /path/to/tnp/smoothLeptonScaleFactors/test/ -s iso

# run all pieces (and merge)
# python w_mass_13TeV/smoothLeptonScaleFactors.py /path/to/tnp/efficiencies_ERA/mu_STEP_CHARGE/allEfficiencies_2D.root /path/to/tnp/smoothLeptonScaleFactors/ --run-all [--do-merge] [--do-steps isonotrig iso triggerplus triggerminus idipplus idipminus trackingplus trackingminus recoplus recominus]

#
# TIPS:
# --> input path is usually the main folder produced by the TnP code, which contains a folder like efficiencies_GtoH/
# --> for bookkeeping it is better to create output folder smoothLeptonScaleFactors/ where efficiencies_GtoH/ is located
# --> add -d just to print commands without running
# --> add --do-merge to do everything in one go (remove --run-all if only merging is needed)
# --> ERA, STEP, CHARGE are keywords that are converted internally in runFiles() when doing multiple steps
# --> a summary of bad fits (if any) is printed on stdout for each step, but also in a txt file for easier check
#     when running multiple steps in series. For the smoothing to make sense, status and covstatus MUST be 0 for all fits

import array
import math
import os
import pickle
from copy import *
from functools import partial

import hist
import lz4.frame
import numpy as np
import tensorflow as tf
import utilitiesCMG
from scipy.interpolate import RegularGridInterpolator

import narf
import narf.fitutils
from utilities import common

utilities = utilitiesCMG.util()

## safe batch mode
import sys

args = sys.argv[:]
sys.argv = ["-b"]
import ROOT

sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

# sys.path.append(os.getcwd() + "/plotUtils/")
# from utility import *
from scripts.analysisTools.plotUtils.utility import *
## TODO: move this script to scripts/analysisTools/w_mass_13TeV/
from scripts.analysisTools.w_mass_13TeV.run2Dsmoothing import \
    makeAntiSFfromSFandEffi

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
def pol2_root(xvals, parms, xLowVal=0.0, xFitRange=1.0):
    xscaled = (xvals[0] - xLowVal) / xFitRange
    return parms[0] + parms[1] * xscaled + parms[2] * xscaled**2


def pol3_root(xvals, parms, xLowVal=0.0, xFitRange=1.0):
    xscaled = (xvals[0] - xLowVal) / xFitRange
    return parms[0] + parms[1] * xscaled + parms[2] * xscaled**2 + parms[3] * xscaled**3


def pol4_root(xvals, parms, xLowVal=0.0, xFitRange=1.0):
    xscaled = (xvals[0] - xLowVal) / xFitRange
    return (
        parms[0]
        + parms[1] * xscaled
        + parms[2] * xscaled**2
        + parms[3] * xscaled**3
        + parms[4] * xscaled**4
    )


def erf_tf(xvals, parms):
    return parms[0] * (1.0 + tf.math.erf((xvals[0] - parms[1]) / parms[2]))


def antiErf_tf(xvals, parms):
    return 1.0 - parms[0] * (1.0 + tf.math.erf((xvals[0] - parms[1]) / parms[2]))


#############################################


def getReducedChi2andLabel(func):
    if func.GetNDF():
        reducedChi2 = func.GetChisquare() / func.GetNDF()
        lineChi2 = "#chi^{{2}}/ndf = {chi2:.2g} / {ndf}".format(
            chi2=func.GetChisquare(), ndf=int(func.GetNDF())
        )
    else:
        reducedChi2 = 0
        lineChi2 = "BAD! #chi^{{2}}/ndf = {chi2:.2g} / {ndf}".format(
            chi2=func.GetChisquare(), ndf=int(func.GetNDF())
        )

    return float(reducedChi2), lineChi2


def copyHisto(h1, h2, copyError=False):

    if h1.GetDimension() != h2.GetDimension():
        print(
            f"Error in copyHisto(): histograms have different dimensions. Dim(h1)={h1.GetDimension()}  Dim(h2)={h2.GetDimension()}. Exit"
        )
        quit()

    if h1.GetDimension() == 1:
        for ix in range(h2.GetNbinsX() + 2):
            if copyError:
                h1.SetBinContent(ix, h2.GetBinContent(ix, iy))
                h1.SetBinError(ix, h2.GetBinError(ix, iy))
            else:
                h1.SetBinContent(ix, h2.GetBinError(ix, iy))
                h1.SetBinError(ix, 0.0)
    elif h1.GetDimension() == 2:
        for ix in range(h2.GetNbinsX() + 2):
            for iy in range(h2.GetNbinsY() + 2):
                if copyError:
                    h1.SetBinContent(ix, iy, h2.GetBinError(ix, iy))
                    h1.SetBinError(ix, iy, 0.0)
                else:
                    h1.SetBinContent(ix, iy, h2.GetBinContent(ix, iy))
                    h1.SetBinError(ix, iy, h2.GetBinError(ix, iy))

    else:
        print("Error in copyHisto(): function not implemented for dimension > 2. Exit")
        quit()


def make1Dhist(namePrefix, h2D, ptbins, step):
    hpt = {}
    # add labels as needed
    stepDict = {
        "reco": "Reconstruction",
        "tracking": "Tracking",
        "idip": "ID + impact parameter",  # "ID + d_{xy}",
        "trigger": "Trigger",
        "iso": "Isolation",
        "antiiso": "Failed isolation",
        "veto": "Veto",
    }
    stepStr = stepDict[
        step.replace("plus", "").replace("minus", "").replace("both", "")
    ]
    for x in range(1, h2D.GetNbinsX() + 1):
        binID = x - 1
        hpt[binID] = ROOT.TH1D(
            f"{namePrefix}_{binID}",
            "%s: %.4g < #eta^{#mu} < %.4g"
            % (
                stepStr,
                h2D.GetXaxis().GetBinLowEdge(x),
                h2D.GetXaxis().GetBinLowEdge(x + 1),
            ),
            len(ptbins) - 1,
            array("d", ptbins),
        )
        for y in range(1, h2D.GetNbinsY() + 1):
            hpt[binID].SetBinContent(y, h2D.GetBinContent(x, y))
            hpt[binID].SetBinError(y, h2D.GetBinError(x, y))
    return hpt


############


def quickPlotTH1(c, h, outdir, channel, postfix=""):
    c.cd()
    h.Scale(1.0 / h.Integral())
    h.GetXaxis().SetTitleOffset(1.2)
    h.GetXaxis().SetTitleSize(0.05)
    h.GetXaxis().SetLabelSize(0.06)
    # h.GetXaxis().LabelsOption("v")
    h.GetYaxis().SetTitle("Fraction of events")
    h.GetYaxis().SetTitleOffset(1.15)
    h.GetYaxis().SetTitleSize(0.05)
    h.GetYaxis().SetLabelSize(0.04)
    h.Draw("HIST")
    for ext in ["png", "pdf"]:
        c.SaveAs(f"{outdir}bestFitFunction{postfix}_{channel}.{ext}")


#########


def getCoordinateNDC(x, canvas, vert=False):
    canvas.Update()  # might be needed
    if vert:
        return (x - canvas.GetY1()) / (canvas.GetY2() - canvas.GetY1())
    else:
        return (x - canvas.GetX1()) / (canvas.GetX2() - canvas.GetX1())


def fitTurnOnTF(
    histo,
    key,
    outdir,
    mc,
    channel="el",
    hist_chosenFunc=0,
    drawFit=True,
    step=None,
    fitRange=None,
    hist_reducedChi2=None,
    charge="both",
    etabins=[],
    widthPtSmooth=0.2,
    hist_nomiAndAlt_etapt=None,
    histoAlt=None,
    efficiencyFitPolDegree=4,
    addCurve=None,
    addCurveLegEntry="",
):

    doingSF = True if mc == "SF" else False
    doSpline = True if efficiencyFitPolDegree < 0 else False

    histoTitle = histo.GetTitle()
    histo.SetTitle("")

    chargeText = ""
    if charge == "plus":
        chargeText = "Positive"
    if charge == "minus":
        chargeText = "Negative"

    originalMaxPt = histo.GetXaxis().GetBinLowEdge(1 + histo.GetNbinsX())

    outfolder = "{out}{mc}/".format(out=outdir, mc=mc)
    createPlotDirAndCopyPhp(outfolder)
    adjustSettings_CMS_lumi()

    leftMargin = 0.16
    rightMargin = 0.03
    bottomMargin = 0.12
    canvas = ROOT.TCanvas(f"canvas_{mc}_{key}", "", 700, 900)
    canvas.SetTickx(1)
    canvas.SetTicky(1)
    canvas.cd()
    canvas.SetLeftMargin(leftMargin)
    canvas.SetBottomMargin(bottomMargin)
    canvas.SetRightMargin(rightMargin)
    canvas.cd()

    setTDRStyle()
    lowerPanelHeight = 0.3  # for bottom panel with pulls or residuals
    canvas.SetBottomMargin(lowerPanelHeight)
    pad2 = ROOT.TPad("pad2", "pad2", 0, 0.0, 1, 0.95)
    pad2.SetTopMargin(1 - lowerPanelHeight)
    pad2.SetRightMargin(rightMargin)
    pad2.SetLeftMargin(leftMargin)
    pad2.SetFillColor(0)
    pad2.SetGridy(1)
    pad2.SetFillStyle(0)

    histo.SetLineColor(ROOT.kBlack)
    histo.SetMarkerColor(ROOT.kBlack)
    histo.SetMarkerStyle(20)
    histo.SetMarkerSize(1)

    histo.GetXaxis().SetLabelSize(0)
    histo.GetXaxis().SetTitle("")

    maxFitRange = histo.GetXaxis().GetBinLowEdge(1 + histo.GetNbinsX())
    minFitRange = histo.GetXaxis().GetBinLowEdge(1)
    originalMaxFitRange = maxFitRange
    originalMinFitRange = minFitRange
    hasNarrowerFitRange = False
    nHistPointsForChi2 = histo.GetNbinsX()
    if fitRange != None:
        hasNarrowerFitRange = True
        if fitRange[1] > 0:
            maxFitRange = fitRange[1]
        if fitRange[0] > 0:
            minFitRange = fitRange[0]
        nHistPointsForChi2 = (
            1
            + histo.GetXaxis().FindFixBin(maxFitRange - 0.001)
            - histo.GetXaxis().FindFixBin(minFitRange + 0.001)
        )
    # histo.GetXaxis().SetRangeUser(minFitRange, maxFitRange)

    if mc == "SF":
        histo.GetYaxis().SetTitle("Data_{ }/^{ }MC efficiency scale factor")
    else:
        histo.GetYaxis().SetTitle("{mc} efficiency".format(mc=mc))
    histo.GetYaxis().SetTitleOffset(1.55)
    histo.GetYaxis().SetTitleSize(0.05)
    histo.GetYaxis().SetLabelSize(0.04)
    histo.GetYaxis().SetLabelOffset(0.01)
    if histoAlt:
        histoAlt.SetStats(0)
        histoAlt.SetLineColor(ROOT.kGray + 2)
        histoAlt.SetMarkerColor(ROOT.kGray + 2)
        histoAlt.SetMarkerStyle(ROOT.kOpenCircle)
        histoAlt.SetMarkerSize(1)
        miny, maxy = getMinMaxMultiHisto([histo, histoAlt], sumError=True)
    else:
        miny, maxy = getMinMaxHisto(histo, sumError=True)
    offset = 0.1 * (maxy - miny)
    upOffset = offset * 4.2  # was 3.5 with 2 lines in the legend and no header
    miny -= offset
    maxy += upOffset
    histo.GetYaxis().SetRangeUser(miny, maxy)
    histo.SetStats(0)
    histo.Draw("EP")
    ##
    ## TODO
    ## Unlike root, tensorflow fits use the bin centers to run the actual fit, and there is no concept of fit range
    ## so, if one wants to fit a subset of the histogram range one needs to pass the slice of the boost histogram
    ## currently this is not implemented

    ###################
    # fits
    ####################
    boost_hist = narf.root_to_hist(histo)
    s = hist.tag.Slicer()
    if hasNarrowerFitRange:
        boost_hist = boost_hist[
            {0: s[complex(0, minFitRange) : complex(0, maxFitRange + 0.001)]}
        ]
    if histoAlt:
        boost_hist_alt = narf.root_to_hist(histoAlt)
        if hasNarrowerFitRange:
            boost_hist_alt = boost_hist_alt[
                {0: s[complex(0, minFitRange) : complex(0, maxFitRange + 0.001)]}
            ]

    ###############################################################
    fitFunction = None
    fitres_TF = None
    defaultFunc = ""
    badFitsID = None
    badCovMatrixID = None

    xFitRange = maxFitRange - minFitRange

    redColor = ROOT.TColor.GetColor("#e42536")
    blueColor = ROOT.TColor.GetColor("#5790fc")

    if doingSF:

        if (
            step == "tracking" and histo.GetNbinsX() == 4
        ):  # use pol2 only with 4 bins, otherwise pol3

            global pol2_tf_scaled
            if pol2_tf_scaled == None:
                pol2_tf_scaled = partial(
                    pol2_root, xLowVal=minFitRange, xFitRange=xFitRange
                )
            params = np.array([1.0, 0.0, 0.0])
            res_tf1_pol2 = narf.fitutils.fit_hist(boost_hist, pol2_tf_scaled, params)
            # for plotting purpose define the TF1 in the original range
            tf1_pol2 = ROOT.TF1(
                "tf1_pol2", pol2_tf_scaled, minFitRange, maxFitRange, len(params)
            )
            tf1_pol2.SetParameters(np.array(res_tf1_pol2["x"], dtype=np.float64))
            tf1_pol2.SetLineWidth(3)
            tf1_pol2.SetLineColor(redColor)

            fitres_TF = {"pol2_tf": res_tf1_pol2}
            fitFunction = {
                "pol2_tf": {
                    "func": tf1_pol2,
                    "leg": "Pol2",
                    "hist": histo,
                }
            }
            defaultFunc = "pol2_tf"
            if histoAlt:
                params = np.array([1.0, 0.0, 0.0])
                res_tf1_pol2_alt = narf.fitutils.fit_hist(
                    boost_hist_alt, pol2_tf_scaled, params
                )
                tf1_pol2_alt = ROOT.TF1(
                    "tf1_pol2_alt",
                    pol2_tf_scaled,
                    minFitRange,
                    maxFitRange,
                    len(params),
                )
                tf1_pol2_alt.SetParameters(
                    np.array(res_tf1_pol2_alt["x"], dtype=np.float64)
                )
                tf1_pol2_alt.SetLineWidth(2)
                tf1_pol2_alt.SetLineColor(blueColor)
                fitres_TF["pol2_alt_tf"] = res_tf1_pol2_alt
                fitFunction["pol2_alt_tf"] = {
                    "func": tf1_pol2_alt,
                    "leg": "dataAltSig",
                    "hist": histoAlt,
                }
        else:
            global pol3_tf_scaled
            if pol3_tf_scaled == None:
                pol3_tf_scaled = partial(
                    pol3_root, xLowVal=minFitRange, xFitRange=xFitRange
                )
            params = np.array([1.0, 0.0, 0.0, 0.0])
            res_tf1_pol3 = narf.fitutils.fit_hist(boost_hist, pol3_tf_scaled, params)
            tf1_pol3 = ROOT.TF1(
                "tf1_pol3", pol3_tf_scaled, minFitRange, maxFitRange, len(params)
            )
            tf1_pol3.SetParameters(np.array(res_tf1_pol3["x"], dtype=np.float64))
            tf1_pol3.SetLineWidth(3)
            tf1_pol3.SetLineColor(redColor)

            # tf1_pol3_test = ROOT.TF1("tf1_pol3_test", pol3_tf_scaled, minFitRange, maxFitRange, len(params))
            # tf1_pol3_test.SetParameters( np.array( res_tf1_pol3["x"], dtype=np.float64 ) )
            # tf1_pol3_test.SetLineStyle(ROOT.kDashed)
            # tf1_pol3_test.SetLineWidth(5)
            # tf1_pol3_test.SetLineColor(ROOT.kBlue)
            # fitopt = "FMBRQS+" # add FM if using Minuit
            # histo.Fit(tf1_pol3_test, fitopt)

            fitres_TF = {"pol3_tf": res_tf1_pol3}
            fitFunction = {
                "pol3_tf": {
                    "func": tf1_pol3,
                    "leg": "Pol3",
                    "hist": histo,
                }
            }
            defaultFunc = "pol3_tf"
            if histoAlt:
                params = np.array([1.0, 0.0, 0.0, 0.0])
                res_tf1_pol3_alt = narf.fitutils.fit_hist(
                    boost_hist_alt, pol3_tf_scaled, params
                )
                tf1_pol3_alt = ROOT.TF1(
                    "tf1_pol3_alt",
                    pol3_tf_scaled,
                    minFitRange,
                    maxFitRange,
                    len(params),
                )
                tf1_pol3_alt.SetParameters(
                    np.array(res_tf1_pol3_alt["x"], dtype=np.float64)
                )
                tf1_pol3_alt.SetLineWidth(2)
                tf1_pol3_alt.SetLineColor(blueColor)
                fitres_TF["pol3_alt_tf"] = res_tf1_pol3_alt
                fitFunction["pol3_alt_tf"] = {
                    "func": tf1_pol3_alt,
                    "leg": "dataAltSig",
                    "hist": histoAlt,
                }

        ###
        badFitsID = badFitsID_sf
        badCovMatrixID = badCovMatrixID_sf
    else:

        if step == "antiiso":
            tf1_erf = ROOT.TF1(
                "tf1_erf",
                "1.0 - [0] * (1.0 + TMath::Erf((x-[1])/[2]))",
                minFitRange,
                maxFitRange,
            )
            res_tf1_erf = narf.fitutils.fit_hist(
                boost_hist, antiErf_tf, np.array([1.0, 35.0, 3.0])
            )
        else:
            tf1_erf = ROOT.TF1(
                "tf1_erf",
                "[0] * (1.0 + TMath::Erf((x-[1])/[2]))",
                minFitRange,
                maxFitRange,
            )
            res_tf1_erf = narf.fitutils.fit_hist(
                boost_hist, erf_tf, np.array([1.0, 35.0, 3.0])
            )
        tf1_erf.SetParameters(np.array(res_tf1_erf["x"], dtype=np.float64))
        tf1_erf.SetLineWidth(2)
        tf1_erf.SetLineStyle(ROOT.kDashed)
        tf1_erf.SetLineColor(ROOT.kBlue)

        if doSpline:
            tf1_polN = ROOT.TSpline3(histo, "b1e1")
            tf1_polN.SetLineWidth(3)
            tf1_polN.SetLineColor(redColor)
            # tf1_polN.Draw("pclsame")
            res_tf1_polN = None
        else:
            global polN_tf_scaled
            if polN_tf_scaled == None:
                polN_tf_scaled = partial(
                    polN_root_,
                    xLowVal=minFitRange,
                    xFitRange=xFitRange,
                    degree=efficiencyFitPolDegree,
                )
            params = np.array([1.0] + [0.0 for i in range(efficiencyFitPolDegree)])
            res_tf1_polN = narf.fitutils.fit_hist(boost_hist, polN_tf_scaled, params)
            tf1_polN = ROOT.TF1(
                f"tf1_pol{efficiencyFitPolDegree}",
                polN_tf_scaled,
                minFitRange,
                maxFitRange,
                len(params),
            )
            tf1_polN.SetParameters(np.array(res_tf1_polN["x"], dtype=np.float64))
            tf1_polN.SetLineWidth(3)
            tf1_polN.SetLineColor(redColor)
        #

        fitres_TF = {
            "erf": res_tf1_erf,
            "polN_tf": res_tf1_polN,
        }
        fitFunction = {
            "polN_tf": {
                "func": tf1_polN,
                "leg": "Spline" if doSpline else f"Pol{efficiencyFitPolDegree}",
                "hist": histo,
            },
            "erf": {
                "func": tf1_erf,
                "leg": "1-Erf" if step == "antiiso" else "Erf",
                "hist": histo,
            },
        }
        defaultFunc = "polN_tf"
        if histoAlt:
            if doSpline:
                tf1_polN_alt = ROOT.TSpline3(histo, "b1e1")
                tf1_polN_alt.SetLineWidth(2)
                tf1_polN_alt.SetLineColor(blueColor)
                # tf1_polN.Draw("pclsame")
                res_tf1_polN_alt = None
            else:
                params = np.array([1.0] + [0.0 for i in range(efficiencyFitPolDegree)])
                res_tf1_polN_alt = narf.fitutils.fit_hist(
                    boost_hist_alt, polN_tf_scaled, params
                )
                tf1_polN_alt = ROOT.TF1(
                    "tf1_polN_alt",
                    polN_tf_scaled,
                    minFitRange,
                    maxFitRange,
                    len(params),
                )
                tf1_polN_alt.SetParameters(
                    np.array(res_tf1_polN_alt["x"], dtype=np.float64)
                )
                tf1_polN_alt.SetLineWidth(2)
                tf1_polN_alt.SetLineColor(blueColor)
            fitres_TF["polN_alt_tf"] = res_tf1_polN_alt
            fitFunction["polN_alt_tf"] = {
                "func": tf1_polN_alt,
                "leg": "dataAltSig",
                "hist": histoAlt,
            }

        if mc == "MC":
            badFitsID = badFitsID_mc
            badCovMatrixID = badCovMatrixID_mc
        else:
            badFitsID = badFitsID_data
            badCovMatrixID = badCovMatrixID_data

    ###############################################################

    for fr in fitres_TF.keys():
        if fitres_TF[fr] == None:
            continue
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
        for ib in range(1, 1 + h.GetNbinsX()):
            pt = h.GetXaxis().GetBinCenter(ib)
            # if the fit range was narrower than the histogram original range, don't use points outside the range
            if pt < minFitRange or pt > maxFitRange:
                continue
            item = (
                h.GetBinContent(ib) - fitFunction[fr]["func"].Eval(pt)
            ) / h.GetBinError(ib)
            manualChi2 += item * item
        fitChi2 = fitres_TF[fr]["loss_val"]
        if abs(manualChi2 - fitChi2) > 0.01:
            print(f"-----> {fr}: manual/fit chi2  = {manualChi2}/{fitChi2}")

    npar = 0 if doSpline else fitFunction[defaultFunc]["func"].GetNpar()

    hband = ROOT.TH1D(
        "hband",
        "",
        int(math.ceil((originalMaxFitRange - originalMinFitRange)) / widthPtSmooth),
        originalMinFitRange,
        originalMaxFitRange,
    )
    if (
        hist_nomiAndAlt_etapt is not None
        and hband.GetNbinsX() != hist_nomiAndAlt_etapt.GetNbinsY()
    ):
        print(
            "ERROR: hband and hist_nomiAndAlt_etapt have a different number of pt bins ({hband.GetNbinsX()} and {hist_nomiAndAlt_etapt.GetNbinsY()}), please check!"
        )
        quit()
    hband.SetStats(0)
    if not doSpline:
        hband.SetFillColor(ROOT.kGray)
    # hband.SetLineColor(fitFunction[defaultFunc]["func"].GetLineColor())
    # hband.SetFillStyle(3001)
    for ib in range(1, hband.GetNbinsX() + 1):
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
            hist_nomiAndAlt_etapt.SetBinContent(key + 1, ib, 1, val)

    # store all variations for faster access below
    altParameters = np.array(
        [np.zeros(npar, dtype=np.float64)] * (npar * 2), dtype=np.float64
    )
    if not doSpline:
        # diagonalize and get eigenvalues and eigenvectors
        e, v = np.linalg.eigh(fitres_TF[defaultFunc]["cov"])
        # print(altParameters)
        for ivar in range(npar):
            shift = np.sqrt(e[ivar]) * v[:, ivar]
            altParameters[ivar] = fitres_TF[defaultFunc]["x"] + shift
            altParameters[ivar + npar] = fitres_TF[defaultFunc]["x"] - shift

    tf1_func_alt = ROOT.TF1()
    tf1_func_alt.SetName("tf1_func_alt")
    fitFunction[defaultFunc]["func"].Copy(tf1_func_alt)
    tf1_func_alt.SetLineWidth(2)
    if hist_nomiAndAlt_etapt is not None:
        lastSystBin = hist_nomiAndAlt_etapt.GetNbinsZ()
    for ib in range(1, hband.GetNbinsX() + 1):
        pt = hband.GetBinCenter(ib)
        err = 0.0
        for ivar in range(npar):
            # set parameters for a given hessian
            tf1_func_alt.SetParameters(
                altParameters[ivar]
            )  # this is for Up variations, Down ones could not be the mirror image
            funcVal = max(0.001, tf1_func_alt.Eval(pt))
            if funcVal >= 1.0 and not doingSF:
                funcVal = 0.9995
            diff = funcVal - hband.GetBinContent(ib)
            err += diff * diff
            if hist_nomiAndAlt_etapt is not None:
                # now fill TH3, also with down variations
                hist_nomiAndAlt_etapt.SetBinContent(key + 1, ib, 2 + ivar, funcVal)
                # repeat for Down variations
                tf1_func_alt.SetParameters(altParameters[ivar + npar])
                funcVal = max(0.001, tf1_func_alt.Eval(pt))
                if funcVal >= 1.0 and not doingSF:
                    funcVal = 0.9995
                hist_nomiAndAlt_etapt.SetBinContent(
                    key + 1, ib, 2 + ivar + npar, funcVal
                )
        err = math.sqrt(err)
        hband.SetBinError(ib, err)
        if hist_nomiAndAlt_etapt is not None:
            hist_nomiAndAlt_etapt.SetBinError(key + 1, ib, 1, err)
            if histoAlt:
                for f in fitFunction.keys():
                    if "_alt_tf" in f:
                        funcVal = max(0.001, fitFunction[f]["func"].Eval(pt))
                        if funcVal >= 1.0 and not doingSF:
                            funcVal = 0.9995
                        hist_nomiAndAlt_etapt.SetBinContent(
                            key + 1, ib, lastSystBin, funcVal
                        )

    hband.Draw("E4SAME")
    # redraw to have them on top
    for f in fitFunction.keys():
        fitFunction[f]["func"].Draw("LSAME")
    if addCurve:
        addCurve.SetLineColor(ROOT.kGreen + 2)
        addCurve.SetLineStyle(1)
        addCurve.SetLineWidth(2)
        addCurve.SetFillColorAlpha(ROOT.kGreen + 1, 0.35)
        addCurve.SetFillStyle(1001)
        addCurve.Draw("LE4 SAME")
        # addCurve.Draw("C SAME")
        # addCurve.Draw("HISTSAME")
        chi2curve = 0.0
        for ib in range(1, 1 + histo.GetNbinsX()):
            item = (
                histo.GetBinContent(ib)
                - addCurve.GetBinContent(
                    addCurve.GetXaxis().FindFixBin(histo.GetBinCenter(ib))
                )
            ) / histo.GetBinError(ib)
            chi2curve += item * item
    histo.Draw("EPSAME")
    if histoAlt:
        histoAlt.Draw("EPSAME")

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

    legLineWidth = 0.065
    upLeg = 0.93 - legLineWidth
    downLeg = max(0.5, upLeg - legLineWidth * nFits)
    if addCurve:
        downLeg = max(
            0.5, downLeg - 0.03
        )  # add some more vertical space, but not too much
    leftLeg = 0.16
    rightLeg = 0.98

    lat = ROOT.TLatex()
    lat.SetNDC()
    lat.SetTextSize(0.04)
    lat.SetTextFont(42)
    lat.SetTextColor(ROOT.kBlack)
    lat.DrawLatex(leftLeg + 0.03, upLeg + 0.015, histoTitle)

    leg = ROOT.TLegend(leftLeg, downLeg, rightLeg, upLeg)
    # leg.SetHeader(histoTitle)
    leg.SetFillColor(0)
    # leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetFillColorAlpha(0, 0.6)
    leg.SetNColumns(2)
    leg.SetTextSize(0.032)

    hNomiForLegend = ROOT.TH1D("hNomiForLegend", "", 1, 0, 1)
    hNomiForLegend.SetStats(0)
    hAltForLegend = ROOT.TH1D("hAltForLegend", "", 1, 0, 1)
    hAltForLegend.SetStats(0)
    hNomiFitForLegend = ROOT.TH1D("hNomiFitForLegend", "", 1, 0, 1)
    hNomiFitForLegend.SetStats(0)
    hAltFitForLegend = ROOT.TH1D("hAltFitForLegend", "", 1, 0, 1)
    hAltFitForLegend.SetStats(0)

    # for chosen function
    reducedChi2 = 0.0
    for f in fitFunction.keys():
        # legEntry = f"Nomi {fitFunction[f]['leg']}" if f == defaultFunc else fitFunction[f]["leg"]
        legEntry = f"Model + unc." if f == defaultFunc else "Alt. model"
        if doSpline:
            chi2 = 0
            ndof = 1
        else:
            chi2 = fitres_TF[f]["loss_val"]
            ndof = int(nHistPointsForChi2 - fitFunction[f]["func"].GetNpar())
            legEntry += f"   #chi^{{2}} / ndf = {round(chi2,1)} / {ndof}"
            chi2prob = ROOT.TMath.Prob(chi2, ndof)
            if chi2prob < 0.05:
                perc_chi2prob = 100.0 * chi2prob
                sign = "="
                if perc_chi2prob < 0.1:
                    perc_chi2prob = 0.1
                    sign = "<"
                legEntry += " (prob {} {}%)".format(sign, round(perc_chi2prob, 1))
        if f == defaultFunc:
            hNomiForLegend.SetMarkerColor(histo.GetMarkerColor())
            hNomiForLegend.SetMarkerStyle(histo.GetMarkerStyle())
            hNomiFitForLegend.SetFillColor(hband.GetFillColor())
            hNomiFitForLegend.SetLineWidth(2)
            hNomiFitForLegend.SetLineColor(fitFunction[f]["func"].GetLineColor())
            leg.AddEntry(hNomiForLegend, "Nominal", "EPL")
            leg.AddEntry(hNomiFitForLegend, legEntry, "LF")
            reducedChi2 = chi2 / ndof
        elif "_alt_tf" in f:
            hAltForLegend.SetMarkerColor(histoAlt.GetMarkerColor())
            hAltForLegend.SetLineColor(histoAlt.GetMarkerColor())
            hAltForLegend.SetMarkerStyle(histoAlt.GetMarkerStyle())
            hAltFitForLegend.SetLineColor(fitFunction[f]["func"].GetLineColor())
            hAltFitForLegend.SetLineWidth(2)
            #
            leg.AddEntry(hAltForLegend, "Alternate", "EPL")
            leg.AddEntry(hAltFitForLegend, legEntry, "L")
        else:
            leg.AddEntry(fitFunction[f]["func"], legEntry, "L")

    if addCurve:
        leg.AddEntry(
            addCurve, f"{addCurveLegEntry}, #chi^{{2}} = {round(chi2curve,1)}", "LF"
        )

    # leg.AddEntry(hband, "Model uncertainty", 'F')
    leg.Draw("same")
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
    # hAcceptBand = copy.deepcopy(histo.Clone("hAcceptBand"))
    hAcceptBand = ROOT.TH1D("hAcceptBand", "", 41, 24, 65)
    if step != "tracking" and "veto" not in step:
        yMaxBand = downLeg * (canvas.GetY2() - canvas.GetY1()) + canvas.GetY1()
        for i in range(1, 1 + hAcceptBand.GetNbinsX()):
            if (
                hAcceptBand.GetBinCenter(i) > 26.0
                and hAcceptBand.GetBinCenter(i) < 56.0
            ):
                hAcceptBand.SetBinContent(i, 0.0)
                hAcceptBand.SetBinError(i, 0.0)
            else:
                hAcceptBand.SetBinContent(i, yMaxBand)
                hAcceptBand.SetBinError(i, 0.0)
        hAcceptBand.SetFillColorAlpha(ROOT.kYellow + 1, 0.5)
        hAcceptBand.Draw("HIST SAME")
    ########################################

    lumi = "16.8"
    latCMS = ROOT.TLatex()
    latCMS.SetNDC()
    latCMS.SetTextFont(42)
    latCMS.SetTextColor(ROOT.kBlack)
    latCMS.SetTextSize(0.042)
    latCMS.DrawLatex(leftMargin, 0.94, "#bf{CMS} #it{Preliminary}")
    if lumi != None:
        latCMS.DrawLatex(0.675, 0.94, "%s fb^{-1} (13 TeV)" % lumi)
    else:
        latCMS.DrawLatex(0.775, 0.94, "(13 TeV)")

    # Now the bottom panel
    pad2.Draw()
    pad2.cd()

    frame = histo.Clone("frame")
    frame.SetTitle("")
    frame.GetXaxis().SetTitleOffset(1.2)
    frame.GetXaxis().SetTitleSize(0.05)
    frame.GetXaxis().SetLabelSize(0.04)
    frame.SetStats(0)
    frame.Reset("ICES")
    frame.GetXaxis().SetRangeUser(minFitRange, maxFitRange)
    frame.GetYaxis().SetNdivisions(5)
    denLabel = fitFunction[defaultFunc]["leg"]
    # frame.GetYaxis().SetTitle(f"X / {denLabel}") # or Pulls
    frame.GetYaxis().SetTitle(f"Ratio to fit")  # or Pulls
    frame.GetYaxis().SetTitleOffset(1.55)
    frame.GetYaxis().SetTitleSize(0.05)
    frame.GetYaxis().SetLabelSize(0.04)
    frame.GetYaxis().SetLabelOffset(0.01)
    frame.GetYaxis().CenterTitle()
    frame.GetXaxis().SetTitle(f"{chargeText} muon #it{{p}}_{{T}} (GeV)")

    ratios = []
    defaultIndex = 0
    den_noerr = copy.deepcopy(hband.Clone("den_noerr"))
    den = copy.deepcopy(hband.Clone("den"))
    den.SetTitle("")
    for iBin in range(1, den_noerr.GetNbinsX() + 1):
        den_noerr.SetBinError(iBin, 0.0)
    den.Divide(den_noerr)
    for ifunc, f in enumerate(fitFunction.keys()):
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
            ratio.SetBinContent(
                ib, fitFunction[f]["func"].Eval(xval) / den_noerr.GetBinContent(ib)
            )
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
            ratioCurve.SetBinContent(
                ib,
                addCurve.GetBinContent(addCurve.GetXaxis().FindFixBin(xval))
                / den_noerr.GetBinContent(ib),
            )

    # now data which is less granular
    dataRatio = copy.deepcopy(histo.Clone("dataRatio"))
    for ib in range(1, 1 + dataRatio.GetNbinsX()):
        ibinDen = hband.GetXaxis().FindFixBin(dataRatio.GetBinCenter(ib))
        denVal = hband.GetBinContent(ibinDen)
        dataRatio.SetBinContent(ib, dataRatio.GetBinContent(ib) / denVal)
        dataRatio.SetBinError(ib, dataRatio.GetBinError(ib) / denVal)

    rminy, rmaxy = getMinMaxMultiHisto(
        ratios + [den, dataRatio],
        excludeEmpty=True,
        sumError=True,
        excludeUnderflow=True,
        excludeOverflow=True,
    )
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
    for ip, p in enumerate(ratios):
        drawOpt = "C"  # like HIST but smooth curve through points
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
    # hAcceptBandRatio = copy.deepcopy(histo.Clone("hAcceptBandRatio"))
    hAcceptBandRatio = ROOT.TH1D("hAcceptBand", "", 41, 24, 65)

    if step != "tracking" and "veto" not in step:
        hAcceptBandRatio.Reset("ICESM")
        yMaxBand = rmaxy
        for i in range(1, 1 + hAcceptBandRatio.GetNbinsX()):
            if (
                hAcceptBandRatio.GetBinCenter(i) > 26.0
                and hAcceptBandRatio.GetBinCenter(i) < 56.0
            ):
                hAcceptBandRatio.SetBinContent(i, 0.0)
                hAcceptBandRatio.SetBinError(i, 0.0)
            else:
                hAcceptBandRatio.SetBinContent(i, yMaxBand)
                hAcceptBandRatio.SetBinError(i, 0.0)
        hAcceptBandRatio.SetFillColorAlpha(ROOT.kYellow + 1, 0.5)
        hAcceptBandRatio.Draw("HIST SAME")
    ########################################

    pad2.RedrawAxis("sameaxis")

    # ---------------------------------

    tmpch = ""
    if charge != "both":
        tmpch = "_" + charge
    for ext in ["pdf", "png"]:
        if mc == "SF":
            canvas.SaveAs(
                "{out}sf_pt_{ch}_eta{b}{charge}.{ext}".format(
                    out=outfolder, ch=channel, b=key, charge=tmpch, ext=ext
                )
            )
        else:
            canvas.SaveAs(
                "{out}eff{mc}_pt_{ch}_eta{b}{charge}.{ext}".format(
                    out=outfolder, mc=mc, ch=channel, b=key, charge=tmpch, ext=ext
                )
            )
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
    hist_reducedChi2_data.SetBinContent(
        lastBin,
        hist_reducedChi2_data.GetBinContent(lastBin)
        + hist_reducedChi2_data.GetBinContent(1 + lastBin),
    )
    hist_reducedChi2_data.SetBinError(
        lastBin,
        math.sqrt(
            hist_reducedChi2_data.GetBinError(lastBin)
            * hist_reducedChi2_data.GetBinError(lastBin)
            + hist_reducedChi2_data.GetBinError(1 + lastBin)
            * hist_reducedChi2_data.GetBinError(1 + lastBin)
        ),
    )
    lastBin = hist_reducedChi2_MC.GetNbinsX()
    hist_reducedChi2_MC.SetBinContent(
        lastBin,
        hist_reducedChi2_MC.GetBinContent(lastBin)
        + hist_reducedChi2_MC.GetBinContent(1 + lastBin),
    )
    hist_reducedChi2_MC.SetBinError(
        lastBin,
        math.sqrt(
            hist_reducedChi2_MC.GetBinError(lastBin)
            * hist_reducedChi2_MC.GetBinError(lastBin)
            + hist_reducedChi2_MC.GetBinError(1 + lastBin)
            * hist_reducedChi2_MC.GetBinError(1 + lastBin)
        ),
    )
    lastBin = hist_reducedChi2_sf.GetNbinsX()
    hist_reducedChi2_sf.SetBinContent(
        lastBin,
        hist_reducedChi2_sf.GetBinContent(lastBin)
        + hist_reducedChi2_sf.GetBinContent(1 + lastBin),
    )
    hist_reducedChi2_sf.SetBinError(
        lastBin,
        math.sqrt(
            hist_reducedChi2_sf.GetBinError(lastBin)
            * hist_reducedChi2_sf.GetBinError(lastBin)
            + hist_reducedChi2_sf.GetBinError(1 + lastBin)
            * hist_reducedChi2_sf.GetBinError(1 + lastBin)
        ),
    )

    tmpmin, maxY = getMinMaxMultiHisto(
        [hist_reducedChi2_MC, hist_reducedChi2_data, hist_reducedChi2_sf], sumError=True
    )
    maxY = 1.5 * maxY

    hist_reducedChi2_data.GetXaxis().SetTitleOffset(1.2)
    hist_reducedChi2_data.SetLineWidth(2)
    hist_reducedChi2_data.SetLineColor(ROOT.kRed + 2)
    hist_reducedChi2_data.SetFillColor(ROOT.kRed + 2)
    hist_reducedChi2_data.SetFillStyle(3003)
    hist_reducedChi2_data.GetXaxis().SetTitleSize(0.05)
    hist_reducedChi2_data.GetXaxis().SetLabelSize(0.06)
    hist_reducedChi2_data.GetYaxis().SetTitleOffset(1.15)
    hist_reducedChi2_data.GetYaxis().SetTitleSize(0.05)
    hist_reducedChi2_data.GetYaxis().SetLabelSize(0.04)
    hist_reducedChi2_data.GetYaxis().SetTitle("Events")
    hist_reducedChi2_data.GetYaxis().SetRangeUser(0, maxY)
    hist_reducedChi2_data.GetXaxis().SetTitle("#chi^{2} / NDF")
    hist_reducedChi2_data.Draw("HIST")
    lat = ROOT.TLatex()
    xmin = 0.20
    yhi = 0.85
    lat.SetNDC()
    lat.SetTextSize(0.045)
    lat.SetTextFont(62)
    lat.DrawLatex(xmin, yhi, "")
    lat.DrawLatex(xmin, yhi - 0.05, "entries")
    lat.DrawLatex(xmin, yhi - 0.1, "mean")
    lat.DrawLatex(xmin, yhi - 0.15, "rms")
    line1 = "= {0}".format(int(entries_data))
    line2 = "= {:.2f}".format(hist_reducedChi2_data.GetMean())
    line3 = "= {:.2f}".format(hist_reducedChi2_data.GetStdDev())
    lat.SetTextFont(42)
    lat.SetTextColor(ROOT.kRed + 2)
    xmin = xmin + 0.2
    lat.DrawLatex(xmin, yhi, "data")
    lat.DrawLatex(xmin, yhi - 0.05, line1)
    lat.DrawLatex(xmin, yhi - 0.1, line2)
    lat.DrawLatex(xmin, yhi - 0.15, line3)
    # now MC
    hist_reducedChi2_MC.SetLineWidth(2)
    hist_reducedChi2_MC.SetLineColor(ROOT.kBlack)
    hist_reducedChi2_MC.SetFillColor(ROOT.kGray)
    # hist_reducedChi2_MC.SetFillStyle(3004)
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
    lat.SetTextColor(ROOT.kBlack)
    xmin = xmin + 0.2
    lat.DrawLatex(xmin, yhi, "MC")
    lat.DrawLatex(xmin, yhi - 0.05, line1)
    lat.DrawLatex(xmin, yhi - 0.1, line2)
    lat.DrawLatex(xmin, yhi - 0.15, line3)
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
    lat.SetTextColor(ROOT.kBlue)
    xmin = xmin + 0.2
    lat.DrawLatex(xmin, yhi, "SF")
    lat.DrawLatex(xmin, yhi - 0.05, line1)
    lat.DrawLatex(xmin, yhi - 0.1, line2)
    lat.DrawLatex(xmin, yhi - 0.15, line3)

    for ext in ["png", "pdf"]:
        c.SaveAs("{out}reducedChi2_{ch}.{ext}".format(out=outdir, ch=channel, ext=ext))


############################################################################

minmaxSF = {
    "trigger": "0.65,1.15",
    "triggerplus": "0.65,1.15",
    "triggerminus": "0.65,1.15",
    "idip": "0.95,1.01",
    "idipplus": "0.95,1.01",
    "idipminus": "0.95,1.01",
    "iso": "0.975,1.025",
    "isoplus": "0.975,1.025",
    "isominus": "0.975,1.025",
    "isonotrig": "0.97,1.03",
    "isoantitrig": "0.80,1.3",
    # usually one doesn't smooth antiiso efficiencies, they come directly from the iso ones after smoothing
    "antiiso": "0.6,1.25",
    "antiisonotrig": "0.6,1.25",
    #
    "tracking": "0.98,1.01",
    "trackingplus": "0.98,1.01",
    "trackingminus": "0.98,1.01",
    "reco": "0.94,1.02",
    "recoplus": "0.94,1.02",
    "recominus": "0.94,1.02",
    "vetotracking": "0.95,1.05",
    "vetotrackingplus": "0.95,1.05",
    "vetotrackingminus": "0.95,1.05",
    "vetoreco": "0.92,1.04",
    "vetorecoplus": "0.92,1.04",
    "vetorecominus": "0.92,1.04",
    "vetoidip": "0.98,1.02",
    "vetoidipplus": "0.98,1.02",
    "vetoidipminus": "0.98,1.02",
}

stepsWithAntiSF = [
    "iso",
    "isoplus",
    "isominus",
    "trigger",
    "triggerplus",
    "triggerminus",
]
# the veto is special since it should be a product of reco*tracking*idip SF before doing the antiVersion


def mergeFiles(args):

    steps = args.doSteps
    inputDir = args.outdir[0]  # outdir is the input directory for the merge command
    # if inputDir is read using the eos mount it might fail (should copy files with xrdcp first)
    # for now keep it as it is
    era = args.era
    outfolder = f"{inputDir}/{era}/"

    files = []
    for step in steps:
        charge = "both"
        if "plus" in step:
            charge = "plus"
            step = step.replace(charge, "")
        elif "minus" in step:
            charge = "minus"
            step = step.replace(charge, "")
        files.append(
            f"{inputDir}/{era}/mu_{step}_{charge}/smoothedSFandEffi_{step}_{era}_{charge}.root"
        )
    outfile = f"{outfolder}/allSmooth_{era}.root"
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
        inputFile = (
            args.inputfile[0]
            .replace("_ERA", f"_{era}")
            .replace("_STEP", f"_{step}")
            .replace("_CHARGE", f"_{charge}")
        )

        cmd = f"python scripts/analysisTools/w_mass_13TeV/smoothLeptonScaleFactors.py {inputFile} {args.outdir[0]} -c {charge} -s {step}"
        cmd += f" --input-hist-names '{args.inputHistNames}' --input-hist-names-alt '{args.inputHistNamesAlt}'"
        ## now we no longer smooth efficiencies, but keep commented in case we need it again
        # if step in ["iso", "isonotrig", "antiiso", "antiisonotrig"]:
        #     cmd += f" --no-skip-eff --fit-pol-degree-efficiency {args.fitPolDegreeEfficiency}"
        print()
        safeSystem(cmd, args.dryRun)
        print()

    if args.doMerge:
        mergeFiles(args)


if __name__ == "__main__":

    parser = common_plot_parser()
    parser.add_argument("inputfile", type=str, nargs=1, help="input root file with TH2")
    parser.add_argument(
        "outdir", type=str, nargs=1, help="output directory to save things"
    )
    parser.add_argument(
        "-c",
        "--charge",
        default="both",
        choices=["both", "plus", "minus"],
        type=str,
        help="Plus or minus if the efficiencies were derived separately for each charge. If both, assumes no charge splitting in the inputs",
    )
    parser.add_argument(
        "-e",
        "--era",
        dest="era",
        default="GtoH",
        choices=["GtoH"],
        type=str,
        help="Efficiency era",
    )
    parser.add_argument(
        "-s",
        "--step",
        dest="step",
        default="iso",
        choices=list(minmaxSF.keys()),
        help="Working point to smooth",
    )
    parser.add_argument(
        "-r",
        "--pt-fit-range",
        dest="ptFitRange",
        type=float,
        nargs=2,
        default=[-1, -1],
        help="Pt range fo the fit: pass two values for min and max. If one of them (or both) is negative, the corresponding histogram range is used",
    )
    parser.add_argument(
        "-w",
        "--width-pt",
        dest="widthPt",
        default="0.2",
        type=float,
        help="Pt bin width for the smoothed histogram",
    )
    parser.add_argument(
        "--set-max-pt-histo",
        dest="setMaxPtHisto",
        default="-1.0",
        type=float,
        help="Set upper pt for output histograms. If negative use default max from input histograms",
    )
    parser.add_argument(
        "--input-hist-names",
        dest="inputHistNames",
        default="EffData2D,EffMC2D,SF2D_nominal",
        type=str,
        help="Pass comma separated list of 3  names, for eff(data),eff(MC),SF, to be used instead of the default names",
    )
    parser.add_argument(
        "--input-hist-names-alt",
        dest="inputHistNamesAlt",
        default="EffDataAltSig2D,SF2D_dataAltSig",
        type=str,
        help="Pass comma separated list of 2  names for alternate variations, for eff(data),SF, to be used instead of the default names",
    )
    parser.add_argument(
        "--fit-pol-degree-efficiency",
        dest="fitPolDegreeEfficiency",
        default=4,
        type=int,
        help="Degree for polynomial used in the fits to efficiencies (-1 will use a spline)",
    )
    parser.add_argument(
        "--no-skip-eff",
        dest="skipEff",
        action="store_false",
        help="Do not skip efficiency smoothing (default is to do only SF to save time and if one only wants to smooth SF directly)",
    )
    # utility option to print commands to do all files
    parser.add_argument(
        "-d",
        "--dryRun",
        action="store_true",
        help="Do not execute commands, just print them",
    )
    parser.add_argument(
        "--run-all",
        dest="runAll",
        action="store_true",
        help="Make and run commands to run all steps specified in --merge-steps",
    )
    parser.add_argument(
        "--do-steps",
        dest="doSteps",
        nargs="+",
        default=[
            "isoantitrig",
            "isonotrig"
            "iso"
            "triggerplus"
            "triggerminus"
            "idipplus"
            "idipminus"
            "trackingplus"
            "trackingminus"
            "recoplus"
            "recominus",
        ],
        choices=list(minmaxSF.keys()),
        help="Working points to smooth when running --run-all or --do-merge",
    )
    # option to merge files once they exist
    parser.add_argument(
        "--do-merge",
        dest="doMerge",
        action="store_true",
        help="Merge efficiency files if they all exist",
    )
    # option to select definition of isolation, to use correct MC truth efficiencies (mainly relevant for antiiso SF, for antitrigger it shouldn't matter
    parser.add_argument(
        "--isolationDefinition",
        choices=["iso04vtxAgn", "iso04"],
        default="iso04vtxAgn",
        help="Isolation type (and corresponding scale factors)",
    )

    args = parser.parse_args()
    logger = logging.setup_logger(os.path.basename(__file__), args.verbose, True)

    if args.fitPolDegreeEfficiency < -1:
        args.fitPolDegreeEfficiency = -1

    ROOT.TH1.SetDefaultSumw2()

    if args.runAll:
        runFiles(args)
        quit()

    if args.doMerge:
        mergeFiles(args)
        quit()

    if any(x in args.step for x in ["plus", "minus"]):
        args.charge = "plus" if "plus" in args.step else "minus"
        args.step = args.step.replace(args.charge, "")

    channel = "mu"
    charge = (
        ""
        if args.charge == "both"
        else "positive" if args.charge == "plus" else "negative"
    )
    lepton = f"{charge} muon"

    outdir_original = args.outdir[0]
    addStringToEnd(outdir_original, "/", notAddIfEndswithMatch=True)
    outdir_original += f"{args.era}/{channel}_{args.step}_{args.charge}/"
    outdir = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)

    outfilename = f"smoothedSFandEffi_{args.step}_{args.era}_{args.charge}.root"

    #########################################
    #########################################

    datahistname = f"effData_{args.step}_{args.era}_{args.charge}"
    mchistname = f"effMC_{args.step}_{args.era}_{args.charge}"
    sfhistname = f"SF2D_nominal_{args.step}_{args.era}_{args.charge}"

    if len(args.inputHistNames):
        datahistname, mchistname, sfhistname = args.inputHistNames.split(",")

    datahistnameAlt = f"effData_altSig_{args.step}_{args.era}_{args.charge}"
    sfhistnameAlt = f"SF2D_dataAltSig_{args.step}_{args.era}_{args.charge}"
    if len(args.inputHistNamesAlt):
        datahistnameAlt, sfhistnameAlt = args.inputHistNamesAlt.split(",")

    tfile = safeOpenFile(args.inputfile[0])
    hsf = safeGetObject(tfile, sfhistname)
    hsfAlt = safeGetObject(tfile, sfhistnameAlt)
    hdata = safeGetObject(tfile, datahistname)
    hmc = safeGetObject(tfile, mchistname)
    hdataAlt = safeGetObject(tfile, datahistnameAlt)
    tfile.Close()

    etabins = [
        round(hdata.GetXaxis().GetBinLowEdge(i), 1)
        for i in range(1, 2 + hdata.GetNbinsX())
    ]
    ptbins = [
        round(hdata.GetYaxis().GetBinLowEdge(i), 1)
        for i in range(1, 2 + hdata.GetNbinsY())
    ]

    # utility histogram to show what function will be used (usually we choose pol4 or pol3 for efficiencies of scale factors)
    hist_chosenFunc = ROOT.TH1D(
        "chosenFitFunc", "Best fit function for each #eta bin for data or MC", 3, 0, 3
    )
    hist_chosenFunc.GetXaxis().SetBinLabel(1, "erf")
    hist_chosenFunc.GetXaxis().SetBinLabel(2, "tf1_pol3")
    hist_chosenFunc.GetXaxis().SetBinLabel(3, f"pol{args.fitPolDegreeEfficiency}_tf")
    #
    hist_chosenFunc_SF = ROOT.TH1D(
        "chosenFitFunc_SF", "Best fit function for each #eta bin for SF", 3, 0, 3
    )
    hist_chosenFunc_SF.GetXaxis().SetBinLabel(1, "tf1_cheb3")
    hist_chosenFunc_SF.GetXaxis().SetBinLabel(2, "tf1_pol2")
    hist_chosenFunc_SF.GetXaxis().SetBinLabel(3, "pol3_tf")

    hist_reducedChi2_data = ROOT.TH1D("reducedChi2_data", "Reduced #chi^{2}", 25, 0, 5)
    hist_reducedChi2_data.StatOverflows()  # use underflow and overflow to compute mean and RMS
    hist_reducedChi2_MC = ROOT.TH1D("reducedChi2_MC", "Reduced #chi^{2}", 25, 0, 5)
    hist_reducedChi2_MC.StatOverflows()  # use underflow and overflow to compute mean and RMS
    hist_reducedChi2_sf = ROOT.TH1D("reducedChi2_sf", "Reduced #chi^{2}", 25, 0, 5)
    hist_reducedChi2_sf.StatOverflows()  # use underflow and overflow to compute mean and RMS

    ######################
    # to make ratio
    ######################
    ratioData = ROOT.TH2D(
        "dataEfficiencyRatio",
        "Original/smooth Data efficiency ratio",
        len(etabins) - 1,
        array("d", etabins),
        len(ptbins) - 1,
        array("d", ptbins),
    )
    ratioMC = ROOT.TH2D(
        "mcEfficiencyRatio",
        "Original/smooth MC efficiency ratio",
        len(etabins) - 1,
        array("d", etabins),
        len(ptbins) - 1,
        array("d", ptbins),
    )
    ratioSF = ROOT.TH2D(
        "scaleFactorRatio",
        "Original/smooth scale factor ratio",
        len(etabins) - 1,
        array("d", etabins),
        len(ptbins) - 1,
        array("d", ptbins),
    )

    copyHisto(ratioData, hdata)
    copyHisto(ratioMC, hmc)
    copyHisto(ratioSF, hsf)

    ######################
    # to make ratio
    ######################
    pullData = ROOT.TH2D(
        "dataEfficiencyPull",
        "Original/smooth Data efficiency pull",
        len(etabins) - 1,
        array("d", etabins),
        len(ptbins) - 1,
        array("d", ptbins),
    )
    pullMC = ROOT.TH2D(
        "mcEfficiencyPull",
        "Original/smooth MC efficiency pull",
        len(etabins) - 1,
        array("d", etabins),
        len(ptbins) - 1,
        array("d", ptbins),
    )
    pullSF = ROOT.TH2D(
        "scaleFactorPull",
        "Original/smooth scale factor pull",
        len(etabins) - 1,
        array("d", etabins),
        len(ptbins) - 1,
        array("d", ptbins),
    )
    pullSFfromSmoothEffi = ROOT.TH2D(
        "scaleFactorPullFromSmoothEffi",
        "Original/smooth scale factor pull (SF from smooth effi)",
        len(etabins) - 1,
        array("d", etabins),
        len(ptbins) - 1,
        array("d", ptbins),
    )

    if not args.skipEff:
        copyHisto(pullData, hdata)
        copyHisto(pullMC, hmc)
    copyHisto(pullSF, hsf)
    copyHisto(pullSFfromSmoothEffi, hsf)
    errData = copy.deepcopy(hdata.Clone("errData"))
    errMC = copy.deepcopy(hmc.Clone("errMC"))
    errSF = copy.deepcopy(hsf.Clone("errSF"))
    copyHisto(errData, hdata, copyError=True)
    copyHisto(errMC, hmc, copyError=True)
    copyHisto(errSF, hsf, copyError=True)

    #############
    # these will be used to check the smoothed efficiency
    ###############
    maxPtHisto = hdata.GetYaxis().GetBinLowEdge(1 + hdata.GetNbinsY())
    if args.setMaxPtHisto > 0.0:
        maxPtHisto = args.setMaxPtHisto

    nFinePtBins = int(
        math.ceil(maxPtHisto - hdata.GetYaxis().GetBinLowEdge(1)) / args.widthPt
    )
    minPtHisto = hdata.GetYaxis().GetBinLowEdge(1)

    hdataSmoothCheck_origBinPt = ROOT.TH2D(
        "hdataSmoothCheck_origBinPt",
        "Data smoothed efficiency",
        len(etabins) - 1,
        array("d", etabins),
        len(ptbins) - 1,
        array("d", ptbins),
    )
    hmcSmoothCheck_origBinPt = ROOT.TH2D(
        "hmcSmoothCheck_origBinPt",
        "MC smoothed efficiency",
        len(etabins) - 1,
        array("d", etabins),
        len(ptbins) - 1,
        array("d", ptbins),
    )
    hsfSmoothCheck_origBinPt = ROOT.TH2D(
        "hsfSmoothCheck_origBinPt",
        "Data/MC smoothed scale factor",
        len(etabins) - 1,
        array("d", etabins),
        len(ptbins) - 1,
        array("d", ptbins),
    )

    ## prepare histograms with nominal eta-pt efficiency or scale factor, and other variations on the Z axis
    ## will save both up and down variations for effStat and (currently) only 1 syst, so have 1+2*nPar+1 bins
    ## assume we have 48 eta bins to use constructor of TH3
    nBinsEff = 2 + 2 * (
        1 + args.fitPolDegreeEfficiency
    )  # fits with pol4 (5 parameters), then nominal in first bin and syst in last
    hist_effData_nomiAndAlt_etapt = ROOT.TH3D(
        "hist_effData_nomiAndAlt_etapt",
        "Smooth nominal and alternate data efficiency",
        48,
        -2.40,
        2.40,
        nFinePtBins,
        minPtHisto,
        maxPtHisto,
        nBinsEff,
        0.5,
        0.5 + nBinsEff,
    )
    hist_effMC_nomiAndAlt_etapt = ROOT.TH3D(
        "hist_effMC_nomiAndAlt_etapt",
        "Smooth nominal and alternate MC efficiency",
        48,
        -2.40,
        2.40,
        nFinePtBins,
        minPtHisto,
        maxPtHisto,
        nBinsEff,
        0.5,
        0.5 + nBinsEff,
    )
    # TODO: avoid magic numbers, use option as for efficiencies
    nBinsSF = (
        8 if args.step == "tracking" else 10
    )  # fits are done with pol2 for tracking (3 parameters) and pol3 otherwise (4 parameters), then this is 1 + 2 * statVarsUp + 1
    hist_SF_nomiAndAlt_etapt = ROOT.TH3D(
        "hist_SF_nomiAndAlt_etapt",
        "Smooth nominal and alternate scale factor",
        48,
        -2.40,
        2.40,
        nFinePtBins,
        minPtHisto,
        maxPtHisto,
        nBinsSF,
        0.5,
        0.5 + nBinsSF,
    )

    # hmc and hdata have eta on X and pt on Y
    # we select slices at constant eta and fit along pt with some function

    label = args.step + (args.charge if args.charge != "both" else "")
    hsfpt = make1Dhist("hsfpt", hsf, ptbins, label)
    hsfptAlt = make1Dhist("hsfptAlt", hsfAlt, ptbins, label)

    outfolder_eigenVars = f"{outdir}/eigenDecomposition/"

    zaxisRange = ""
    zaxisRangeSF = "::" + minmaxSF[args.step]

    if not args.skipEff:
        hmcpt = make1Dhist("hmcpt", hmc, ptbins, label)
        ###########################
        # first MC
        ###########################
        for key in hmcpt:
            bestFitFunc = fitTurnOnTF(
                hmcpt[key],
                key,
                outdir,
                "MC",
                channel=channel,
                hist_chosenFunc=hist_chosenFunc,
                step=args.step,
                fitRange=args.ptFitRange,
                hist_reducedChi2=hist_reducedChi2_MC,
                charge=args.charge,
                etabins=etabins,
                widthPtSmooth=args.widthPt,
                hist_nomiAndAlt_etapt=hist_effMC_nomiAndAlt_etapt,
                efficiencyFitPolDegree=args.fitPolDegreeEfficiency,
            )
            for ipt in range(1, hmcSmoothCheck_origBinPt.GetNbinsY() + 1):
                ptval = hmcSmoothCheck_origBinPt.GetYaxis().GetBinCenter(ipt)
                hmcSmoothCheck_origBinPt.SetBinContent(
                    key + 1, ipt, bestFitFunc.Eval(ptval)
                )
                hmcSmoothCheck_origBinPt.SetBinError(
                    key + 1, ipt, hmc.GetBinError(key + 1, ipt)
                )
        hmcSmoothCheck = getTH2fromTH3(
            hist_effMC_nomiAndAlt_etapt, "hmcSmoothCheck", 1, 1
        )
        hmcSmoothCheck.SetTitle("Smooth MC efficiency")
        ###########################
        # now data
        ###########################
        hdatapt = make1Dhist("hdatapt", hdata, ptbins, label)
        hdataptAlt = make1Dhist("hdataptAlt", hdataAlt, ptbins, label)
        for key in hdatapt:

            bestFitFunc = fitTurnOnTF(
                hdatapt[key],
                key,
                outdir,
                "Data",
                channel=channel,
                hist_chosenFunc=hist_chosenFunc,
                step=args.step,
                fitRange=args.ptFitRange,
                hist_reducedChi2=hist_reducedChi2_data,
                charge=args.charge,
                etabins=etabins,
                widthPtSmooth=args.widthPt,
                hist_nomiAndAlt_etapt=hist_effData_nomiAndAlt_etapt,
                histoAlt=hdataptAlt[key],
                efficiencyFitPolDegree=args.fitPolDegreeEfficiency,
            )
            for ipt in range(1, hdataSmoothCheck_origBinPt.GetNbinsY() + 1):
                ptval = hdataSmoothCheck_origBinPt.GetYaxis().GetBinCenter(ipt)
                hdataSmoothCheck_origBinPt.SetBinContent(
                    key + 1, ipt, bestFitFunc.Eval(ptval)
                )
                hdataSmoothCheck_origBinPt.SetBinError(
                    key + 1, ipt, hdata.GetBinError(key + 1, ipt)
                )
        hdataSmoothCheck = getTH2fromTH3(
            hist_effData_nomiAndAlt_etapt, "hdataSmoothCheck", 1, 1
        )
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
            smoothSFfromEffiTMP = scaleFactor.ProjectionY(
                f"{args.step}TMP_{key}", key + 1, key + 1, "e"
            )
        bestFitFunc = fitTurnOnTF(
            hsfpt[key],
            key,
            outdir,
            "SF",
            channel=channel,
            hist_chosenFunc=hist_chosenFunc_SF,
            step=args.step,
            fitRange=args.ptFitRange,
            hist_reducedChi2=hist_reducedChi2_sf,
            charge=args.charge,
            etabins=etabins,
            widthPtSmooth=args.widthPt,
            hist_nomiAndAlt_etapt=hist_SF_nomiAndAlt_etapt,
            histoAlt=hsfptAlt[key],
            addCurve=smoothSFfromEffiTMP,
            addCurveLegEntry=f"SF from pol{args.fitPolDegreeEfficiency} effi",
        )
        for ipt in range(1, hsfSmoothCheck_origBinPt.GetNbinsY() + 1):
            ptval = hsfSmoothCheck_origBinPt.GetYaxis().GetBinCenter(ipt)
            hsfSmoothCheck_origBinPt.SetBinContent(
                key + 1, ipt, bestFitFunc.Eval(ptval)
            )
            hsfSmoothCheck_origBinPt.SetBinError(
                key + 1, ipt, hsf.GetBinError(key + 1, ipt)
            )
    hsfSmoothCheck = getTH2fromTH3(hist_SF_nomiAndAlt_etapt, "hsfSmoothCheck", 1, 1)
    hsfSmoothCheck.SetTitle("Smooth scale factor")
    #################################
    # start to make plots
    #################################

    canvas = ROOT.TCanvas("canvas", "", 700, 625)

    # plot eigen variations
    systVarTitle = "Alternate/nominal"  # "Alternate - nominal"
    if not args.skipEff:
        nvars = int((hist_effMC_nomiAndAlt_etapt.GetNbinsZ() - 2) / 2)
        for iv in range(nvars):
            binVar = 2 + iv
            # MC
            hvar = getTH2fromTH3(
                hist_effMC_nomiAndAlt_etapt, f"effStatVar_p{iv}_effMC", binVar, binVar
            )
            hvar.SetTitle(f"Eff. stat. nuisance p{iv}")
            # hvar.Add(hmcSmoothCheck, -1.0)
            hvar.Divide(hmcSmoothCheck)
            drawCorrelationPlot(
                hvar,
                "{lep} #eta".format(lep=lepton),
                "{lep} p_{{T}} [GeV]".format(lep=lepton),
                f"{systVarTitle} (MC efficiency)",
                hvar.GetName(),
                "ForceTitle",
                outfolder_eigenVars,
                palette=args.palette,
                passCanvas=canvas,
            )
            # data
            hvar = getTH2fromTH3(
                hist_effData_nomiAndAlt_etapt,
                f"effStatVar_p{iv}_effData",
                binVar,
                binVar,
            )
            hvar.SetTitle(f"Eff. stat. nuisance p{iv}")
            # hvar.Add(hdataSmoothCheck, -1.0)
            hvar.Divide(hdataSmoothCheck)
            drawCorrelationPlot(
                hvar,
                "{lep} #eta".format(lep=lepton),
                "{lep} p_{{T}} [GeV]".format(lep=lepton),
                f"{systVarTitle} (data efficiency)",
                hvar.GetName(),
                "ForceTitle",
                outfolder_eigenVars,
                palette=args.palette,
                passCanvas=canvas,
            )
        # syst only for data
        systBin = hist_effData_nomiAndAlt_etapt.GetNbinsZ()
        hvar = getTH2fromTH3(
            hist_effData_nomiAndAlt_etapt, "effSystVar_effData", systBin, systBin
        )
        hvar.SetTitle("Eff. syst. nuisance")
        # hvar.Add(hdataSmoothCheck, -1.0)
        hvar.Divide(hdataSmoothCheck)
        drawCorrelationPlot(
            hvar,
            "{lep} #eta".format(lep=lepton),
            "{lep} p_{{T}} [GeV]".format(lep=lepton),
            f"{systVarTitle} (data efficiency)",
            hvar.GetName(),
            "ForceTitle",
            outfolder_eigenVars,
            palette=args.palette,
            passCanvas=canvas,
        )

    # SF
    nvars = int((hist_SF_nomiAndAlt_etapt.GetNbinsZ() - 2) / 2)
    for iv in range(nvars):
        binVar = 2 + iv
        hvar = getTH2fromTH3(
            hist_SF_nomiAndAlt_etapt, f"effStatVar_p{iv}_SF", binVar, binVar
        )
        hvar.SetTitle(f"Eff. stat. nuisance p{iv}")
        # hvar.Add(hsfSmoothCheck, -1.0)
        hvar.Divide(hsfSmoothCheck)
        drawCorrelationPlot(
            hvar,
            "{lep} #eta".format(lep=lepton),
            "{lep} p_{{T}} [GeV]".format(lep=lepton),
            f"{systVarTitle} (scale factor)",
            hvar.GetName(),
            "ForceTitle",
            outfolder_eigenVars,
            palette=args.palette,
            passCanvas=canvas,
        )
    #
    systBin = hist_SF_nomiAndAlt_etapt.GetNbinsZ()
    hvar = getTH2fromTH3(hist_SF_nomiAndAlt_etapt, "effSystVar_SF", systBin, systBin)
    hvar.SetTitle("Eff. syst. nuisance")
    # hvar.Add(hsfSmoothCheck, -1.0)
    hvar.Divide(hsfSmoothCheck)
    drawCorrelationPlot(
        hvar,
        "{lep} #eta".format(lep=lepton),
        "{lep} p_{{T}} [GeV]".format(lep=lepton),
        f"{systVarTitle} (data efficiency)",
        hvar.GetName(),
        "ForceTitle",
        outfolder_eigenVars,
        palette=args.palette,
        passCanvas=canvas,
    )

    # plot original histograms
    drawCorrelationPlot(
        hmc,
        "{lep} #eta".format(lep=lepton),
        "{lep} p_{{T}} [GeV]".format(lep=lepton),
        "MC efficiency%s" % zaxisRange,
        "inputEfficiency_MC",
        "",
        outdir,
        palette=args.palette,
        passCanvas=canvas,
    )
    drawCorrelationPlot(
        hdata,
        "{lep} #eta".format(lep=lepton),
        "{lep} p_{{T}} [GeV]".format(lep=lepton),
        "Data efficiency%s" % zaxisRange,
        "inputEfficiency_Data",
        "",
        outdir,
        palette=args.palette,
        passCanvas=canvas,
    )
    drawCorrelationPlot(
        hsf,
        "{lep} #eta".format(lep=lepton),
        "{lep} p_{{T}} [GeV]".format(lep=lepton),
        "Data/MC scale factor%s" % zaxisRangeSF,
        "inputScaleFactor",
        "",
        outdir,
        palette=args.palette,
        passCanvas=canvas,
    )

    # now the smoothed ones
    drawCorrelationPlot(
        hsfSmoothCheck,
        "{lep} #eta".format(lep=lepton),
        "{lep} p_{{T}} [GeV]".format(lep=lepton),
        "Data/MC smoothed scale factor%s" % zaxisRangeSF,
        "smoothScaleFactorDirectly",
        "ForceTitle",
        outdir,
        palette=args.palette,
        passCanvas=canvas,
    )
    if not args.skipEff:
        # plot smooth efficiencies and SF made from them
        drawCorrelationPlot(
            hmcSmoothCheck,
            "{lep} #eta".format(lep=lepton),
            "{lep} p_{{T}} [GeV]".format(lep=lepton),
            "MC smoothed efficiency%s" % zaxisRange,
            "smoothEfficiency_MC",
            "ForceTitle",
            outdir,
            palette=args.palette,
            passCanvas=canvas,
        )
        drawCorrelationPlot(
            hdataSmoothCheck,
            "{lep} #eta".format(lep=lepton),
            "{lep} p_{{T}} [GeV]".format(lep=lepton),
            "Data smoothed efficiency%s" % zaxisRange,
            "smoothEfficiency_Data",
            "ForceTitle",
            outdir,
            palette=args.palette,
            passCanvas=canvas,
        )
        drawCorrelationPlot(
            scaleFactor,
            "{lep} #eta".format(lep=lepton),
            "{lep} p_{{T}} [GeV]".format(lep=lepton),
            "Data/MC scale factor%s" % zaxisRangeSF,
            "smoothScaleFactor",
            "ForceTitle",
            outdir,
            palette=args.palette,
            passCanvas=canvas,
        )

    #################################
    # plot also with original binning
    ################################

    # divide before drawing the denominator, whose axis settings are modified by drawCorrelationPlot and seems to affect the ratio as well if divided afterwards
    ratioData.Divide(hdataSmoothCheck_origBinPt)
    ratioMC.Divide(hmcSmoothCheck_origBinPt)

    drawCorrelationPlot(
        hmcSmoothCheck_origBinPt,
        "{lep} #eta".format(lep=lepton),
        "{lep} p_{{T}} [GeV]".format(lep=lepton),
        "MC smoothed efficiency%s" % zaxisRange,
        "smoothEfficiency_MC_origBinPt",
        "ForceTitle",
        outdir,
        palette=args.palette,
        passCanvas=canvas,
    )
    drawCorrelationPlot(
        hdataSmoothCheck_origBinPt,
        "{lep} #eta".format(lep=lepton),
        "{lep} p_{{T}} [GeV]".format(lep=lepton),
        "Data smoothed efficiency%s" % zaxisRange,
        "smoothEfficiency_Data_origBinPt",
        "ForceTitle",
        outdir,
        palette=args.palette,
        passCanvas=canvas,
    )
    drawCorrelationPlot(
        hsfSmoothCheck_origBinPt,
        "{lep} #eta".format(lep=lepton),
        "{lep} p_{{T}} [GeV]".format(lep=lepton),
        "Data/MC smoothed scale factor%s" % zaxisRangeSF,
        "smoothScaleFactorDirectly_origBinPt",
        "ForceTitle",
        outdir,
        palette=args.palette,
        passCanvas=canvas,
    )

    # scale factor: data/MC
    scaleFactor_origBinPt = ROOT.TH2D(
        "scaleFactor_origBinPt",
        "Scale factor",
        len(etabins) - 1,
        array("d", etabins),
        len(ptbins) - 1,
        array("d", ptbins),
    )
    copyHisto(scaleFactor_origBinPt, hdataSmoothCheck_origBinPt)
    scaleFactor_origBinPt.Divide(hmcSmoothCheck_origBinPt)
    scaleFactor_origBinPt.SetMinimum(
        scaleFactor_origBinPt.GetBinContent(scaleFactor_origBinPt.GetMinimumBin())
    )
    scaleFactor_origBinPt.SetMaximum(
        scaleFactor_origBinPt.GetBinContent(scaleFactor_origBinPt.GetMaximumBin())
    )

    # to make ratio, divide before passing to function, to avoid changes in the histogram
    ratioSF.Divide(scaleFactor_origBinPt)

    # scaleFactor_origBinPt.GetZaxis().SetTitle("Data/MC scale factor")
    drawCorrelationPlot(
        scaleFactor_origBinPt,
        "{lep} #eta".format(lep=lepton),
        "{lep} p_{{T}} [GeV]".format(lep=lepton),
        "Data/MC scale factor%s" % zaxisRangeSF,
        "smoothScaleFactor_origBinPt",
        "ForceTitle",
        outdir,
        palette=args.palette,
        passCanvas=canvas,
    )

    ######################
    # finally SF(smooth)/SF(original)
    ######################
    drawCorrelationPlot(
        ratioData,
        "{lep} #eta".format(lep=lepton),
        "{lep} p_{{T}} [GeV]".format(lep=lepton),
        "Data efficiency ratio (original/smooth)::0.99,1.01",
        "dataEfficiencyRatio",
        "ForceTitle",
        outdir,
        palette=args.palette,
        passCanvas=canvas,
    )
    drawCorrelationPlot(
        ratioMC,
        "{lep} #eta".format(lep=lepton),
        "{lep} p_{{T}} [GeV]".format(lep=lepton),
        "MC efficiency ratio (original/smooth)::0.99,1.01",
        "mcEfficiencyRatio",
        "ForceTitle",
        outdir,
        palette=args.palette,
        passCanvas=canvas,
    )
    drawCorrelationPlot(
        ratioSF,
        "{lep} #eta".format(lep=lepton),
        "{lep} p_{{T}} [GeV]".format(lep=lepton),
        "scale factor ratio (original/smooth)::0.99,1.01",
        "scaleFactorRatio",
        "ForceTitle",
        outdir,
        palette=args.palette,
        passCanvas=canvas,
    )

    if not args.skipEff:
        pullData.Add(hdataSmoothCheck_origBinPt, -1.0)
        pullData.Divide(errData)
        pullMC.Add(hmcSmoothCheck_origBinPt, -1.0)
        pullMC.Divide(errMC)
    pullSF.Add(hsfSmoothCheck_origBinPt, -1.0)
    pullSF.Divide(errSF)

    drawCorrelationPlot(
        pullData,
        "{lep} #eta".format(lep=lepton),
        "{lep} p_{{T}} [GeV]".format(lep=lepton),
        "Data eff. pull (original-smooth)/err::-5.0,5.0",
        "dataEfficiencyPull",
        "ForceTitle",
        outdir,
        nContours=10,
        palette=args.palette,
        passCanvas=canvas,
    )
    drawCorrelationPlot(
        pullMC,
        "{lep} #eta".format(lep=lepton),
        "{lep} p_{{T}} [GeV]".format(lep=lepton),
        "MC eff. pull (original-smooth)/err::-5.0,5.0",
        "mcEfficiencyPull",
        "ForceTitle",
        outdir,
        nContours=10,
        palette=args.palette,
        passCanvas=canvas,
    )
    drawCorrelationPlot(
        pullSF,
        "{lep} #eta".format(lep=lepton),
        "{lep} p_{{T}} [GeV]".format(lep=lepton),
        "scale factor pull (original-smoothDirectly)/err::-5.0,5.0",
        "scaleFactorPull",
        "ForceTitle",
        outdir,
        nContours=10,
        palette=args.palette,
        passCanvas=canvas,
    )
    if not args.skipEff:
        # add also ratio of ratioData and ratioMC, because in each there might be trend and we want to see if in the double ratio they would cancel
        doubleRatioDataMC = copy.deepcopy(ratioData.Clone("doubleRatioDataMC"))
        doubleRatioDataMC.SetTitle("(smooth/binned)_{data} / (smooth/binned)_{MC}")
        doubleRatioDataMC.Divide(ratioMC)
        drawCorrelationPlot(
            doubleRatioDataMC,
            "{lep} #eta".format(lep=lepton),
            "{lep} p_{{T}} [GeV]".format(lep=lepton),
            "Double ratio",
            "doubleRatio_smoothOverBinnedEffi_DataAndMC",
            "ForceTitle",
            outdir,
            palette=args.palette,
            passCanvas=canvas,
        )
        # additional pulls
        pullSFfromSmoothEffi.Add(scaleFactor_origBinPt, -1.0)
        pullSFfromSmoothEffi.Divide(errSF)
        drawCorrelationPlot(
            pullSFfromSmoothEffi,
            "{lep} #eta".format(lep=lepton),
            "{lep} p_{{T}} [GeV]".format(lep=lepton),
            "scale factor pull (original-smoothFromEffi)/err::-5.0,5.0",
            "scaleFactorPull_SFfromSmoothEffi",
            "ForceTitle",
            outdir,
            nContours=10,
            palette=args.palette,
            passCanvas=canvas,
        )

        SFpullRatio = copy.deepcopy(
            pullSFfromSmoothEffi.Clone(
                "scaleFactorPullRatio_SFfromSmoothEffiOverDirectSmoothing"
            )
        )
        SFpullRatio.Divide(pullSF)
        SFpullRatio.SetTitle("")
        drawCorrelationPlot(
            SFpullRatio,
            "{lep} #eta".format(lep=lepton),
            "{lep} p_{{T}} [GeV]".format(lep=lepton),
            "SF pull ratio: smoothFromEffi / directSmoothing",
            SFpullRatio.GetName(),
            "ForceTitle",
            outdir,
            nContours=10,
            palette=args.palette,
            passCanvas=canvas,
        )
    # ######################
    # # See the difference between smoothing Data and MC efficiency and taking the ratio or smoothing directly the efficiency ratio
    # ######################
    ratioSF_smoothNumDen_smoothRatio = ROOT.TH2D(
        "ratioSF_smoothNumDen_smoothRatio",
        "SF ratio: smooth eff or ratio directly",
        len(etabins) - 1,
        array("d", etabins),
        len(ptbins) - 1,
        array("d", ptbins),
    )

    copyHisto(ratioSF_smoothNumDen_smoothRatio, scaleFactor_origBinPt)
    ratioSF_smoothNumDen_smoothRatio.Divide(hsfSmoothCheck_origBinPt)
    drawCorrelationPlot(
        ratioSF_smoothNumDen_smoothRatio,
        "{lep} #eta".format(lep=lepton),
        "{lep} p_{{T}} [GeV]".format(lep=lepton),
        "SF ratio: smooth eff or ratio directly::0.999,1.001",
        "ratioSF_smoothNumDen_smoothRatio",
        "ForceTitle",
        outdir,
        palette=args.palette,
        passCanvas=canvas,
    )

    # same as ratioSF_smoothNumDen_smoothRatio but with fine pt binning
    if not args.skipEff:
        ratioSF_smoothEffiOverSmoothDirectly = copy.deepcopy(
            hdataSmoothCheck.Clone("ratioSF_smoothEffiOverSmoothDirectly")
        )
        ratioSF_smoothEffiOverSmoothDirectly.SetTitle(
            "SF ratio: smooth eff or ratio directly"
        )
        ratioSF_smoothEffiOverSmoothDirectly.Divide(hmcSmoothCheck)
        ratioSF_smoothEffiOverSmoothDirectly.Divide(hsfSmoothCheck)
        drawCorrelationPlot(
            ratioSF_smoothEffiOverSmoothDirectly,
            "{lep} #eta".format(lep=lepton),
            "{lep} p_{{T}} [GeV]".format(lep=lepton),
            "SF ratio: smooth eff or ratio directly::0.999,1.001",
            "ratioSF_smoothEffiOverSmoothDirectly",
            "ForceTitle",
            outdir,
            palette=args.palette,
            passCanvas=canvas,
        )

    c = ROOT.TCanvas("c", "", 700, 700)
    c.SetTickx(1)
    c.SetTicky(1)
    c.cd()
    c.SetLeftMargin(0.14)
    c.SetRightMargin(0.06)
    c.cd()
    if not args.skipEff:
        quickPlotTH1(c, hist_chosenFunc, outdir, channel)
    quickPlotTH1(c, hist_chosenFunc_SF, outdir, channel, postfix="SF")

    # draw the chi2 plots
    drawReducedChi2(c, hist_reducedChi2_data, hist_reducedChi2_MC, hist_reducedChi2_sf)

    hist_postfix = f"_{args.era}_{args.step}_{args.charge}"

    hasAntiSFfromSFandEffi = False
    if args.step in stepsWithAntiSF:
        hasAntiSFfromSFandEffi = True
        if args.isolationDefinition == "iso04vtxAgn":
            effSmoothFile = f"{common.data_dir}/muonSF/intermediate_vtxAgnosticIso/efficiencies3D_rebinUt2_vtxAgnPfRelIso04.pkl.lz4"
        elif args.isolationDefinition == "iso04":
            # effSmoothFile = "/eos/user/m/mciprian/www/WMassAnalysis/test2Dsmoothing/makeWMCefficiency3D/noMuonCorr_noSF_allProc_noDphiCut_rebinUt2_addEffi2D/efficiencies3D_rebinUt2.pkl.lz4"
            effSmoothFile = f"{common.data_dir}/muonSF/efficiencies3D_rebinUt2.pkl.lz4"  # this might miss the 2D efficiencies, must be updated
        else:
            raise NotImplementedError(
                f"Isolation definition {args.isolationDefinition} not implemented"
            )
        with lz4.frame.open(effSmoothFile) as fileEff:
            allMCeff = pickle.load(fileEff)
            effType = args.step
            if args.charge != "both":
                effType += args.charge
            eff_boost = allMCeff[f"Wmunu_MC_eff_{effType}_etapt"]
        stepChargeTitle = f"{args.step}" + (
            "" if args.charge == "both" else args.charge
        )
        # plot ratio of W MC truth efficiencies and TnP one, before smoothing
        histMCtruthEffi2DorigBin_etapt_root = narf.hist_to_root(eff_boost)
        histMCtruthEffi2DorigBin_etapt_root.SetName(
            f"histMCtruthEffi2DorigBin_{args.step}_etapt_root"
        )
        ratio_MCtruthEffiOverTnP_etapt_root = copy.deepcopy(
            histMCtruthEffi2DorigBin_etapt_root.Clone(
                f"ratio_MCtruthEffiOverTnP_{args.step}_etapt_root"
            )
        )
        ratio_MCtruthEffiOverTnP_etapt_root.SetTitle("W MC-truth / Z tag-and-probe")
        ratio_MCtruthEffiOverTnP_etapt_root.Divide(hmc)
        drawCorrelationPlot(
            ratio_MCtruthEffiOverTnP_etapt_root,
            "{lep} #eta".format(lep=lepton),
            "{lep} p_{{T}} [GeV]".format(lep=lepton),
            f"{stepChargeTitle} MC efficiency ratio",
            ratio_MCtruthEffiOverTnP_etapt_root.GetName(),
            "ForceTitle",
            outdir,
            palette=args.palette,
            passCanvas=canvas,
        )
        productSFandMCtruthEffi = copy.deepcopy(
            histMCtruthEffi2DorigBin_etapt_root.Clone(
                f"productSFandMCtruthEffi_{args.step}_etapt"
            )
        )
        productSFandMCtruthEffi.SetTitle(stepChargeTitle)
        productSFandMCtruthEffi.Multiply(hsf)
        drawCorrelationPlot(
            productSFandMCtruthEffi,
            "{lep} #eta".format(lep=lepton),
            "{lep} p_{{T}} [GeV]".format(lep=lepton),
            f"Product of SF and W MC truth efficiency",
            productSFandMCtruthEffi.GetName(),
            "ForceTitle",
            outdir,
            palette=args.palette,
            passCanvas=canvas,
        )

        logger.info(f"Preparing W MC smooth eta-pt efficiencies for {stepChargeTitle}")
        axis_eta = hist.axis.Variable(
            etabins, name="eta", overflow=False, underflow=False
        )  # as for previous histograms, in case eta is not uniform
        axis_pt = hist.axis.Regular(
            nFinePtBins,
            minPtHisto,
            maxPtHisto,
            name="pt",
            overflow=False,
            underflow=False,
        )
        histEffi2D_etapt_boost = hist.Hist(
            axis_eta,
            axis_pt,
            name=f"smoothEffi2D_{args.step}_etapt_boost",
            storage=hist.storage.Weight(),
        )
        # smooth efficiency vs pt in each eta bin using a spline, then fill the histogram with fine pt binning
        for ieta in range(len(etabins) - 1):
            etaLow = round(etabins[ieta], 1)
            etaHigh = round(etabins[ieta + 1], 1)
            etaRange = f"{etaLow} < #eta^{{#mu}} < {etaHigh}"
            etaCenter = 0.5 * (etaHigh + etaLow)
            eta_index = eff_boost.axes[0].index(etaCenter)
            eff_boost_pt = eff_boost[{0: eta_index}]  # from 2D (eta-pt) to 1D (pt)
            xvals = [
                tf.constant(center, dtype=tf.float64)
                for center in eff_boost_pt.axes.centers
            ]
            ptvals = np.reshape(xvals[0], [-1])
            yvals = eff_boost_pt.values()
            yvals[np.isnan(yvals)] = (
                0  # protection against bins where no events were selected (extreme ut for instance), set efficiency to 0 instead of 1
            )
            # logger.warning(etaRange)
            # logger.warning(f"ptvals = {ptvals}")
            # logger.warning(f"yvals = {yvals}")
            eff_boost_pt.values()[...] = yvals
            # the grid interpolator will be created up to the extreme bin centers, so need bounds_error=False to allow the extrapolation to extend outside until the bin edges
            # and then we can set its extrapolation value to fill_value ('None' uses the extrapolation from the curve inside accpetance)
            interp = RegularGridInterpolator(
                (ptvals,), yvals, method="cubic", bounds_error=False, fill_value=None
            )
            xvalsFine = [
                tf.constant(center, dtype=tf.float64)
                for center in histEffi2D_etapt_boost.axes.centers
            ]
            ptvalsFine = np.reshape(xvalsFine[1], [-1])
            pts = np.array(ptvalsFine)
            # print(pts)
            smoothVals = interp(pts)
            # print(smoothVals)
            histEffi2D_etapt_boost.values()[eta_index:] = smoothVals
        histEffi2D_etapt_boost.variances()[...] = np.zeros_like(
            histEffi2D_etapt_boost.variances()
        )
        histEffi2D_etapt_root = narf.hist_to_root(histEffi2D_etapt_boost)
        histEffi2D_etapt_root.SetName(f"smoothEffi2D_{args.step}_etapt_root")
        histEffi2D_etapt_root.SetTitle(stepChargeTitle)
        # plot W MC efficiencies after spline interpolation as a check
        drawCorrelationPlot(
            histEffi2D_etapt_root,
            "{lep} #eta".format(lep=lepton),
            "{lep} p_{{T}} [GeV]".format(lep=lepton),
            "W MC efficiency (spline interp.)",
            histEffi2D_etapt_root.GetName(),
            "ForceTitle",
            outdir,
            palette=args.palette,
            passCanvas=canvas,
        )

        logger.info("Done with efficiencies")

        # for completeness make also original antiiso efficiencies and scale factors
        # they also must be defined using the formula antiSF = (1 - SF * eff) / (1 - eff),
        # where eff must be the MC truth one (on W or Z, it should hopefully not matter too much, although the uT dependence is not modeled here)
        # however, here we use antiSF = (1 - effData) / (1 - effMC) using TnP efficiencies, to illustrate the difference in plots
        hist_postfix_anti = f"_{args.era}_anti{args.step}_{args.charge}"
        hanti_effData_original = copy.deepcopy(
            hdata.Clone("effData_original" + hist_postfix_anti)
        )
        ROOT.wrem.initializeRootHistogram(hanti_effData_original, 1.0)
        hanti_effData_original.SetTitle(f"Anti{args.step} data efficiency")
        hanti_effMC_original = copy.deepcopy(
            hanti_effData_original.Clone("effMC_original" + hist_postfix_anti)
        )
        hanti_effMC_original.SetTitle(f"Anti{args.step} MC efficiency")
        hanti_effData_original.Add(hdata, -1.0)
        hanti_effMC_original.Add(hmc, -1.0)
        hanti_SF_original = copy.deepcopy(
            hanti_effData_original.Clone("SF_original" + hist_postfix_anti)
        )
        hanti_SF_original.SetTitle(f"Anti{args.step} scale factors")
        hanti_SF_original.Divide(hanti_effMC_original)
        # now the original dataAltSig for SF
        hanti_SF_originalDataAltSig = copy.deepcopy(
            hanti_effData_original.Clone("SF_originalDataAltSig" + hist_postfix_anti)
        )
        ROOT.wrem.initializeRootHistogram(hanti_SF_originalDataAltSig, 1.0)
        hanti_SF_originalDataAltSig.Add(hdataAlt, -1.0)
        hanti_SF_originalDataAltSig.Divide(hanti_effMC_original)
        # prepare some 1D hists to run plots of antiSF
        label_anti = "anti" + label
        hsfpt_anti = make1Dhist("hsfpt_anti", hanti_SF_original, ptbins, label_anti)
        hsfptAlt_anti = make1Dhist(
            "hsfptAlt_anti", hanti_SF_originalDataAltSig, ptbins, label_anti
        )
        drawCorrelationPlot(
            hanti_SF_original,
            "{lep} #eta".format(lep=lepton),
            "{lep} p_{{T}} [GeV]".format(lep=lepton),
            f"Anti{args.step} scale factors",
            hanti_SF_original.GetName(),
            "ForceTitle",
            outdir,
            palette=args.palette,
            passCanvas=canvas,
        )
        drawCorrelationPlot(
            hanti_effData_original,
            "{lep} #eta".format(lep=lepton),
            "{lep} p_{{T}} [GeV]".format(lep=lepton),
            f"Anti{args.step} data efficiency",
            hanti_effData_original.GetName(),
            "ForceTitle",
            outdir,
            palette=args.palette,
            passCanvas=canvas,
        )
        drawCorrelationPlot(
            hanti_effMC_original,
            "{lep} #eta".format(lep=lepton),
            "{lep} p_{{T}} [GeV]".format(lep=lepton),
            f"Anti{args.step} MC efficiency",
            hanti_effMC_original.GetName(),
            "ForceTitle",
            outdir,
            palette=args.palette,
            passCanvas=canvas,
        )

    # now get SF dividing efficiencies, but for the variations use either nominal data or nominal MC, will end up with two SF histograms
    # cannot get scale factors variations from ratio of corresponding efficiencies, because data and MC variations are totally independent
    # once this is done, make antiisolation histograms
    ##
    ## this following part is a bit obsolete, because we no longer smooth efficiencies to get antiSF
    hasSFhistFromEfficiencySmoothing = False
    if args.step in stepsWithAntiSF and not args.skipEff:
        hasSFhistFromEfficiencySmoothing = True
        # broadcast nominal MC efficiency from TH2 into TH3
        h3_effMC_nominal = copy.deepcopy(
            hist_effMC_nomiAndAlt_etapt.Clone("effMC_broadcast")
        )
        ROOT.wrem.broadCastTH2intoTH3(h3_effMC_nominal, hmcSmoothCheck)
        # create SF with variations from data efficiency only
        hist_SF_nomiAndAlt_onlyDataVar_etapt = copy.deepcopy(
            hist_effData_nomiAndAlt_etapt.Clone(
                "SF_nomiAndAlt_onlyDataVar" + hist_postfix
            )
        )
        hist_SF_nomiAndAlt_onlyDataVar_etapt.SetTitle(
            "Smooth nominal and alternate SF (only data eff variations)"
        )
        hist_SF_nomiAndAlt_onlyDataVar_etapt.Divide(h3_effMC_nominal)
        # create SF with variations from MC efficiency only
        hist_SF_nomiAndAlt_onlyMCVar_etapt = copy.deepcopy(
            hist_effData_nomiAndAlt_etapt.Clone(
                "SF_nomiAndAlt_onlyMCVar" + hist_postfix
            )
        )
        hist_SF_nomiAndAlt_onlyMCVar_etapt.SetTitle(
            "Smooth nominal and alternate SF (only MC eff variations)"
        )
        # broadcast nominal data efficiency from TH2 into TH3
        ROOT.wrem.broadCastTH2intoTH3(
            hist_SF_nomiAndAlt_onlyMCVar_etapt, hdataSmoothCheck
        )
        hist_SF_nomiAndAlt_onlyMCVar_etapt.Divide(hist_effMC_nomiAndAlt_etapt)
        #
        # now antiisolation
        #
        # clone histogram for effData and initialize to 1 before subtraction
        hanti_effData = copy.deepcopy(
            hist_effData_nomiAndAlt_etapt.Clone(
                "effData_nomiAndAlt" + hist_postfix_anti
            )
        )
        hanti_effData.Reset("ICESM")
        hanti_effData.SetTitle("Smooth nominal and alternate data efficiency")
        ROOT.wrem.initializeRootHistogram(hanti_effData, 1.0)
        # clone into MC histogram so it is also initialized to 1
        hanti_effMC = copy.deepcopy(
            hanti_effData.Clone("effMC_nomiAndAlt" + hist_postfix_anti)
        )
        hanti_effMC.SetTitle("Smooth nominal and alternate MC efficiency")
        # subtract efficiencies for isolation
        hanti_effData.Add(hist_effData_nomiAndAlt_etapt, -1.0)
        hanti_effMC.Add(hist_effMC_nomiAndAlt_etapt, -1.0)
        #
        # now prepare scale factors
        # get nominal MC antiiso efficiency and broadcast into a TH3 compatible with the final histogram
        hanti_effMC_nomi_3D = copy.deepcopy(hanti_effMC.Clone("hanti_effMC_nomi_3D"))
        ROOT.wrem.broadCastTH2intoTH3(
            hanti_effMC_nomi_3D,
            ROOT.wrem.projectTH2FromTH3(hanti_effMC, "hanti_effMC_nomi_2D", 1),
        )
        # clone histogram antiiso data efficiency and its variations, to be divided by nominal antiiso MC efficiency
        hanti_SF_nomiAndAlt_onlyDataVar_etapt = copy.deepcopy(
            hanti_effData.Clone("SF_nomiAndAlt_onlyDataVar" + hist_postfix_anti)
        )
        hanti_SF_nomiAndAlt_onlyDataVar_etapt.SetTitle(
            "Smooth nominal and alternate SF (only data eff variations)"
        )
        hanti_SF_nomiAndAlt_onlyDataVar_etapt.Divide(hanti_effMC_nomi_3D)
        # repeat for MC variations, cloning nominal antiisolation data efficiency, broadcasting into TH3, and finally dividing by MC efficiency
        hanti_SF_nomiAndAlt_onlyMCVar_etapt = copy.deepcopy(
            hanti_effData.Clone("SF_nomiAndAlt_onlyMCVar" + hist_postfix_anti)
        )
        hanti_SF_nomiAndAlt_onlyMCVar_etapt.SetTitle(
            "Smooth nominal and alternate SF (only MC eff variations)"
        )
        ROOT.wrem.broadCastTH2intoTH3(
            hanti_SF_nomiAndAlt_onlyMCVar_etapt,
            ROOT.wrem.projectTH2FromTH3(hanti_effData, "hanti_effData_nomi_2D", 1),
        )
        hanti_SF_nomiAndAlt_onlyMCVar_etapt.Divide(hanti_effMC)
        # as the last thing, make smoothed antiiso efficiencies and scale factors with the original pt binning (uncertainties from original binned things)
        hanti_hdataSmoothCheck_origBinPt = copy.deepcopy(
            hdataSmoothCheck_origBinPt.Clone(
                "effData_smoothWithOriginalPtBins" + hist_postfix_anti
            )
        )
        ROOT.wrem.initializeRootHistogram(hanti_hdataSmoothCheck_origBinPt, 1.0)
        hanti_hmcSmoothCheck_origBinPt = copy.deepcopy(
            hanti_hdataSmoothCheck_origBinPt.Clone(
                "effMC_smoothWithOriginalPtBins" + hist_postfix_anti
            )
        )
        hanti_hdataSmoothCheck_origBinPt.Add(hdataSmoothCheck_origBinPt, -1.0)
        hanti_hmcSmoothCheck_origBinPt.Add(hmcSmoothCheck_origBinPt, -1.0)
        # for the SF have to divide the smooth efficiencies, cannot trivially invert the isolation sf
        hanti_hsfSmoothCheck_origBinPt = copy.deepcopy(
            hanti_hdataSmoothCheck_origBinPt.Clone(
                "SF_smoothWithOriginalPtBins" + hist_postfix_anti
            )
        )
        hanti_hsfSmoothCheck_origBinPt.Divide(hanti_hmcSmoothCheck_origBinPt)

        ### now do some antiiso SF direct smoothing, and compared with outcome of iso efficiency smoothing
        # get antiiso SF from smooth iso efficiencies, with fine pt bins
        nomiAntiisoSFfromSmoothIsoEffi = getTH2fromTH3(
            hanti_SF_nomiAndAlt_onlyMCVar_etapt, "nomiAntiisoSFfromSmoothIsoEffi", 1, 1
        )
        # ###########################
        # # now SF
        # ###########################
        for key in hsfpt_anti:
            antiisoTMP = nomiAntiisoSFfromSmoothIsoEffi.ProjectionY(
                f"antiisoTMP_{key}", key + 1, key + 1, "e"
            )
            bestFitFunc = fitTurnOnTF(
                hsfpt_anti[key],
                key,
                outdir + f"/anti{args.step}_fromSmoothEffi",
                "SF",
                step="anti" + args.step,
                fitRange=args.ptFitRange,
                charge=args.charge,
                etabins=etabins,
                widthPtSmooth=args.widthPt,
                histoAlt=hsfptAlt_anti[key],
                addCurve=antiisoTMP,
                addCurveLegEntry=f"SF from pol{args.fitPolDegreeEfficiency} {args.step} effi",
            )

    # prepare antiiso or antitrigger SF using direct SF smoothing and W MC truth efficiencies
    if hasAntiSFfromSFandEffi:
        hist_SF_nomiAndAlt_etapt_boost = narf.root_to_hist(
            hist_SF_nomiAndAlt_etapt,
            axis_names=["SF eta", "SF pt", "nomi-statUpDown-syst"],
        )
        # convert SF and effi to antiSF
        # broadcast effi in 2D to match 3D dimensionality of SF histogram (with 3rd axis containing nomi-stat-syst)
        # TODO: can surely do all in boost without converting to root and having to use root based methods
        hist_postfix_anti = f"_{args.era}_anti{args.step}_{args.charge}"
        histEffi2D_etapt_root_broadcast3D = copy.deepcopy(
            hist_SF_nomiAndAlt_etapt.Clone(
                f"histEffi2D_etapt_root_broadcast3D{hist_postfix_anti}"
            )
        )
        histEffi2D_etapt_root_broadcast3D.Reset("ICESM")
        ROOT.wrem.broadCastTH2intoTH3(
            histEffi2D_etapt_root_broadcast3D, histEffi2D_etapt_root
        )
        histEffi2D_etapt_boost_broadcast3D = narf.root_to_hist(
            histEffi2D_etapt_root_broadcast3D,
            axis_names=["effi eta", "effi pt", "nomi-statUpDown-syst"],
        )
        hanti_SF_nomiAndAlt_etapt_boost = makeAntiSFfromSFandEffi(
            hist_SF_nomiAndAlt_etapt_boost,
            histEffi2D_etapt_boost_broadcast3D,
            args.step,
        )
        # convert back to root and set name
        hanti_SF_nomiAndAlt_etapt = narf.hist_to_root(hanti_SF_nomiAndAlt_etapt_boost)
        hanti_SF_nomiAndAlt_etapt.SetName(f"SF_nomiAndAlt{hist_postfix_anti}")
        hanti_SF_nomiAndAlt_etapt.SetTitle(f"SF_nomiAndAlt{hist_postfix_anti}")
        ##
        ## some plots of nominal antiSF from SF smoothing with W MC efficiencies (comparing to direct antiSF smoothing)
        # get antiiso SF from smooth antiSF, with fine pt bins
        nomiAntiSFfromSFandEffi = getTH2fromTH3(
            hanti_SF_nomiAndAlt_etapt, f"nomiAnti{args.step}SFfromSFandEffi", 1, 1
        )
        nomiAntiSFfromSFandEffi.SetTitle("From smooth SF and W MC effi")
        # set uncertainty properly from variations stored in the 3rd axis
        # TODO: avoid python loop
        nVarsUp = int((nBinsSF - 2) / 2)
        for ix in range(1, 1 + nomiAntiSFfromSFandEffi.GetNbinsX()):
            for iy in range(1, 1 + nomiAntiSFfromSFandEffi.GetNbinsY()):
                unc = 0
                # read stat vars from bin 2 to 1+nVarsUp
                for iz in range(2, 2 + nVarsUp):
                    diff = hanti_SF_nomiAndAlt_etapt.GetBinContent(
                        ix, iy, iz
                    ) - hanti_SF_nomiAndAlt_etapt.GetBinContent(ix, iy, 1)
                    unc += diff * diff
                nomiAntiSFfromSFandEffi.SetBinError(ix, iy, math.sqrt(unc))
        #
        drawCorrelationPlot(
            nomiAntiSFfromSFandEffi,
            "{lep} #eta".format(lep=lepton),
            "{lep} p_{{T}} [GeV]".format(lep=lepton),
            f"Anti{args.step} smooth SF",
            nomiAntiSFfromSFandEffi.GetName(),
            "ForceTitle",
            outdir,
            palette=args.palette,
            passCanvas=canvas,
        )

        # ###########################
        # # now SF
        # ###########################
        for key in hsfpt_anti:
            antiisoTMP = nomiAntiSFfromSFandEffi.ProjectionY(
                f"antiisoTMP2_{key}", key + 1, key + 1, "e"
            )
            bestFitFunc = fitTurnOnTF(
                hsfpt_anti[key],
                key,
                outdir + f"/anti{args.step}_fromSFandEffi",
                "SF",
                step="anti" + args.step,
                fitRange=args.ptFitRange,
                charge=args.charge,
                etabins=etabins,
                widthPtSmooth=args.widthPt,
                histoAlt=hsfptAlt_anti[key],
                addCurve=antiisoTMP,
                addCurveLegEntry=f"From {args.step} SF and W MC effi",
            )

    ###########################
    # Now save things
    ###########################
    tfile = ROOT.TFile.Open(outdir + outfilename, "recreate")
    hsf.Write("SF_original" + hist_postfix)
    hdata.Write("effData_original" + hist_postfix)
    hmc.Write("effMC_original" + hist_postfix)
    hsfAlt.Write("SF_originalDataAltSig" + hist_postfix)
    hsfSmoothCheck_origBinPt.Write(
        "SF_smoothWithOriginalPtBins" + hist_postfix
    )  # this comes from direct SF smoothing (the one from efficiencies is named SF_fromSmoothEfficiencyRatio_... )
    if not args.skipEff:
        hdataSmoothCheck_origBinPt.Write(
            "effData_smoothWithOriginalPtBins" + hist_postfix
        )
        hmcSmoothCheck_origBinPt.Write("effMC_smoothWithOriginalPtBins" + hist_postfix)
        scaleFactor.Write("SF_fromSmoothEfficiencyRatio" + hist_postfix)
        # ratioSF_smoothNumDen_smoothRatio.Write(ratioSF_smoothNumDen_smoothRatio.GetName() + hist_postfix)
        ratioSF_smoothEffiOverSmoothDirectly.Write(
            ratioSF_smoothEffiOverSmoothDirectly.GetName() + hist_postfix
        )
        hist_effData_nomiAndAlt_etapt.Write("effData_nomiAndAlt" + hist_postfix)
        hist_effMC_nomiAndAlt_etapt.Write("effMC_nomiAndAlt" + hist_postfix)
    hist_SF_nomiAndAlt_etapt.Write("SF_nomiAndAlt" + hist_postfix)
    if hasSFhistFromEfficiencySmoothing:
        hist_SF_nomiAndAlt_onlyDataVar_etapt.Write()
        hist_SF_nomiAndAlt_onlyMCVar_etapt.Write()
        hanti_effData.Write()
        hanti_effMC.Write()
        hanti_SF_nomiAndAlt_onlyDataVar_etapt.Write()
        hanti_SF_nomiAndAlt_onlyMCVar_etapt.Write()
        hanti_hdataSmoothCheck_origBinPt.Write()
        hanti_hmcSmoothCheck_origBinPt.Write()
        hanti_hsfSmoothCheck_origBinPt.Write()  # equivalent of scaleFactor histogram from ratio of smooth efficiencies (smoothing SF directly for antiiso doesn't make sense)
        hanti_effData_original.Write()
        hanti_effMC_original.Write()
        hanti_SF_original.Write()
        hanti_SF_originalDataAltSig.Write()
    if hasAntiSFfromSFandEffi:
        hanti_SF_nomiAndAlt_etapt.Write()
    tfile.Close()
    print()
    print(f"Created file {outdir+outfilename}")
    print()

    with open(outdir + outfilename.replace(".root", ".txt"), "w+") as outf:
        outf.write("=" * 30 + "\n")
        outf.write("Summary of bad fits (Erf for data/MC and pol3 for SF)\n")
        outf.write("=" * 30 + "\n")
        outf.write("### Bad fit status (Data/MC/SF,  key,  fitstatus)\n")
        for key in sorted(badFitsID_data.keys()):
            outf.write(f"DATA  {key}  {badFitsID_data[key]}\n")
        for key in sorted(badFitsID_mc.keys()):
            outf.write(f"MC    {key}  {badFitsID_mc[key]}\n")
        for key in sorted(badFitsID_sf.keys()):
            outf.write(f"SF    {key}  {badFitsID_sf[key]}\n")
        outf.write("-" * 30 + "\n")
        outf.write("### Bad covariance matrix status (Data/MC/SF,  key,  covquality)\n")
        for key in sorted(badCovMatrixID_data.keys()):
            outf.write(f"DATA  {key}  {badCovMatrixID_data[key]}\n")
        for key in sorted(badCovMatrixID_mc.keys()):
            outf.write(f"MC  {key}  {badCovMatrixID_mc[key]}\n")
        for key in sorted(badCovMatrixID_sf.keys()):
            outf.write(f"SF  {key}  {badCovMatrixID_sf[key]}\n")
        outf.seek(0)
        print(outf.read())

    copyOutputToEos(outdir, outdir_original, eoscp=args.eoscp)
