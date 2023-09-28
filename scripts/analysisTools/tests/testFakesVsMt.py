#!/usr/bin/env python3

## example
# python tests/testFakesVsMt.py /scratch/mciprian/CombineStudies/12July2023/mw_with_mu_eta_pt_scetlib_dyturboCorr_testFakes_deepMet.hdf5 plots/fromMyWremnants/fitResults/12July2023/testFakes_3DSF/deepMET/noDphiCut/testFakesVsMt/ --palette 87 --rebinx 4 --rebiny 2 --mtBinEdges "0,5,10,15,20,25,30,35,40,45,50,55,60,65" --mtNominalRange "0,40" --mtFitRange "0,40" --fitPolDegree 1 --integralMtMethod sideband -v 4 --maxPt 50  --met deepMET [--dphiStudy] [--jetCut] [--dphiMuonMetCut 0.25]
#
# Note: --jet-cut is used to enforce the jet requirement to derive the FRF,
#       however the validation is made using the nominal histogram, which may or may not have had that cut included
#       so one has to use it based on the input pkl file in order for the validation to be consistent (even though
#       the derivation of the correction is still well defined, since a different histogram is used to compute it)

import os, re, array, math
import time
import argparse

import narf
import narf.fitutils
import wremnants
import hist
import lz4.frame, pickle
from wremnants.datasets.datagroups import Datagroups
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

#sys.path.append(os.getcwd())
from scripts.analysisTools.tests.cropNegativeTemplateBins import cropNegativeContent
from scripts.analysisTools.tests.testPlots1D import plotDistribution1D

# will use as input the histogram mTStudyForFakes with eta-pt-charge-mt-passIso-hasJet, where mt is the full distribution

# function to plot 1D data/MC distributions for the iso/nJet bins we have in this study
def plotProjection1D(rootHists, datasets, outfolder_dataMC, canvas1Dshapes=None, chargeBin=1,
                     projectAxisToKeep=0, isoAxisRange=[1,1], jetAxisRange=[1,2],
                     xAxisName="variable", plotName="variable_failIso_jetInclusive", mTvalueRange=[],
                     rebinVariable=None):

    firstBinMt = 1
    lastBinMt = 1 + rootHists["Data"].GetAxis(3).GetNbins()
    if len(mTvalueRange):
        if len(mTvalueRange) != 2:
            logger.error(f"In plotProjection1D(): mTvalueRange must have two values, it was {mTvalueRange}")
            quit()
        # [40, -1] selects from 40 onwards
        # [40, 100] selects from 40 to 100
        firstBinMt = rootHists["Data"].GetAxis(3).FindFixBin(mTvalueRange[0]+0.001)
        if mTvalueRange[1] > mTvalueRange[0]:
            lastBinMt  = rootHists["Data"].GetAxis(3).FindFixBin(mTvalueRange[1]-0.001)
            
    hdata = None
    hmc = {}
    for d in datasets:
        rootHists[d].GetAxis(2).SetRange(chargeBin, chargeBin)
        rootHists[d].GetAxis(3).SetRange(firstBinMt, lastBinMt)
        rootHists[d].GetAxis(4).SetRange(isoAxisRange[0], isoAxisRange[1])
        rootHists[d].GetAxis(5).SetRange(jetAxisRange[0], jetAxisRange[1])
        rootHists[d].GetAxis(6).SetRange(1, rootHists[d].GetAxis(6).GetNbins())
        if d == "Data":
            hdata = rootHists[d].Projection(projectAxisToKeep, "EO")
            hdata.SetName(f"{plotName}_{d}")
            if rebinVariable:
                hdata.Rebin(rebinVariable)
        else:
            hmc[d] = rootHists[d].Projection(projectAxisToKeep, "EO")
            hmc[d].SetName(f"{plotName}_{d}")
            cropNegativeContent(hmc[d])  # should I crop the 2D before projecting?
            hmc[d].SetFillColor(colors_plots_[d])
            hmc[d].SetLineColor(ROOT.kBlack)
            hmc[d].SetMarkerSize(0)
            hmc[d].SetMarkerStyle(0)
            if rebinVariable:
                hmc[d].Rebin(rebinVariable)

    plotDistribution1D(hdata, hmc, datasets,
                       outfolder_dataMC, canvas1Dshapes=canvas1Dshapes,
                       xAxisName=xAxisName, plotName=plotName)

def plotProjection1Dfrom3D(rootHists, datasets, outfolder_dataMC, canvas1Dshapes=None, chargeBin=1,
                           projectAxisToKeep=0, xAxisName="variable", plotName="variable_failIso_jetInclusive",
                           correctionFakes=None):

    hdata = None
    hmc = {}
    axisProj = "x" if projectAxisToKeep == 0 else "y" if projectAxisToKeep == 1 else "z"
    for d in datasets:
        rootHists[d].GetZaxis().SetRange(chargeBin, chargeBin)
        logger.warning("In plotProjection1Dfrom3D(): setting pt axis to exclude overflows from projections")
        rootHists[d].GetYaxis().SetRange(1, rootHists[d].GetNbinsY())
        if d == "Data":
            hdata = rootHists[d].Project3D(f"{axisProj}eo")
            hdata.SetName(f"{plotName}_{d}")
        else:
            if correctionFakes != None and d in ["Fake", "QCD"]:
                # correct 2D before going to 1D
                #print(f"nX,nY bins = {rootHists[d].GetNbinsX()}, {rootHists[d].GetNbinsY()}")
                hmc2D = rootHists[d].Project3D(f"yxe")
                cropNegativeContent(hmc2D)
                hmc2D.SetName(f"{plotName}_{d}_2Dtmp")                                                 
                print()
                print(f"PlotName: {plotName}")
                print(f"Fakes norm before corr: {hmc2D.Integral()}")
                scaleTH2byOtherTH2(hmc2D, correctionFakes, scaleUncertainty=True)
                print(f"Fakes norm after corr : {hmc2D.Integral()}")
                print()
                if axisProj == "x":
                    hmc[d] = hmc2D.ProjectionX(f"{plotName}_{d}", 1, hmc2D.GetNbinsY(), "eo")
                elif axisProj == "y":
                    hmc[d] = hmc2D.ProjectionY(f"{plotName}_{d}", 1, hmc2D.GetNbinsX(), "eo")
                else:
                    print("Error in plotProjection1Dfrom3D: 'z' axis not permitted for TH2. Abort")
                    quit()
            else:
                hmc[d] = rootHists[d].Project3D(f"{axisProj}eo")
                hmc[d].SetName(f"{plotName}_{d}")
            #cropNegativeContent(hmc[d]) # should I crop the 2D before projecting?
            hmc[d].SetFillColor(colors_plots_[d])
            hmc[d].SetLineColor(ROOT.kBlack)
            hmc[d].SetMarkerSize(0)
            hmc[d].SetMarkerStyle(0)
        
    plotDistribution1D(hdata, hmc, datasets,
                       outfolder_dataMC, canvas1Dshapes=canvas1Dshapes,
                       xAxisName=xAxisName, plotName=plotName)


def drawAndFitFRF(h1,
                  labelXtmp="xaxis", labelYtmp="yaxis",
                  canvasName="default", outdir="./",
                  rebinFactorX=0,
                  draw_both0_noLog1_onlyLog2=1,
                  leftMargin=0.15,
                  rightMargin=0.04,
                  drawStatBox=False,
                  legendCoords="0.15,0.35,0.8,0.9",  # x1,x2,y1,y2
                  canvasSize="600,700",  # use X,Y to pass X and Y size     
                  passCanvas=None,
                  lumi=None,
                  moreTextLatex="",
                  fitRange="0,40", # xmin and xmax
                  mTshape=None,
                  fitPolDegree=3,
                  useBinnedCorr=False,
                  # following two are test histograms which make sense for pol1 fits
                  histoChi2diffTest=None,
                  histoPullsPol1Slope=None
):

    # moreTextLatex is used to write text with TLatex, and the syntax is textToWrite::x1,y1,ypass,textsize
    # where x1 and y1 are the coordinates the first line, and ypass is how much below y1 the second line is (and so on for following lines)
    # multiple text lines can be defined in textToWrite separated by ";", e.g. line1;line2;line3
    
    if (rebinFactorX): 
        if isinstance(rebinFactorX, int): h1.Rebin(rebinFactorX)
        # case in which rebinFactorX is a list of bin edges
        else:                             h1.Rebin(len(rebinFactorX)-1,"",array('d',rebinFactorX)) 

    xAxisName,setXAxisRangeFromUser,xmin,xmax = getAxisRangeFromUser(labelXtmp)
    yAxisName,setYAxisRangeFromUser,ymin,ymax = getAxisRangeFromUser(labelYtmp)

    yAxisTitleOffset = 1.45 if leftMargin > 0.1 else 0.6

    addStringToEnd(outdir,"/",notAddIfEndswithMatch=True)
    createPlotDirAndCopyPhp(outdir)

    cw,ch = canvasSize.split(',')
    #canvas = ROOT.TCanvas("canvas",h2D.GetTitle() if plotLabel == "ForceTitle" else "",700,625)
    canvas = passCanvas if passCanvas != None else ROOT.TCanvas("canvas","",int(cw),int(ch))
    canvas.SetTickx(1)
    canvas.SetTicky(1)
    canvas.cd()
    canvas.SetLeftMargin(leftMargin)
    canvas.SetRightMargin(rightMargin)
    canvas.cd()

    frame = h1.Clone("frame")
    frame.GetXaxis().SetLabelSize(0.04)
    frame.SetStats(0)

    h1.SetLineColor(ROOT.kBlack)
    h1.SetMarkerColor(ROOT.kBlack)
    h1.SetMarkerStyle(20)
    #h1.SetMarkerSize(0)

    if ymin == ymax == 0.0:
        ymin,ymax = getMinMaxHisto(h1,excludeEmpty=True,sumError=True)            
        diff = ymax - ymin
        ymin -= diff * 0.25
        ymax += diff * 0.45
        if ymin < 0: ymin = 0

    h1.GetXaxis().SetTitle(xAxisName)
    h1.GetXaxis().SetTitleOffset(1.2)
    h1.GetXaxis().SetTitleSize(0.05)
    h1.GetXaxis().SetLabelSize(0.04)
    h1.GetYaxis().SetTitle(yAxisName)
    h1.GetYaxis().SetTitleOffset(yAxisTitleOffset) 
    h1.GetYaxis().SetTitleSize(0.05)
    h1.GetYaxis().SetLabelSize(0.04)
    h1.GetYaxis().SetRangeUser(ymin, ymax)
    h1.GetYaxis().SetTickSize(0.01)
    if setXAxisRangeFromUser: h1.GetXaxis().SetRangeUser(xmin,xmax)

    # draw with bands
    # h1.Draw("HIST")
    # h1err = h1.Clone("h1err")
    # h1err.SetFillColor(ROOT.kRed+2)
    # h1err.SetFillStyle(3001)  # 3001 is better than 3002 for pdf, while 3002 is perfect for png
    # h1err.Draw("E2same")
    h1.Draw("EP")
    
    xMinFit, xMaxFit = map(float, fitRange.split(','))
    #print(f"Fit range [{xMinFit}, {xMaxFit}] ({xMaxFit-xMinFit} wide)")

    # tensorflow fit cannot select a subrange to fit, so need to make the boost histogram directly from
    # a new ROOT histograms defined in the actual fit range
    fitBinLow  = h1.GetXaxis().FindFixBin(xMinFit + 0.001)
    fitBinHigh = h1.GetXaxis().FindFixBin(xMaxFit + 0.001)
    fitBinEdges = [round(h1.GetXaxis().GetBinLowEdge(i),2) for i in range(fitBinLow, fitBinHigh+1)]
    #print(fitBinEdges)
    h1fitRange = ROOT.TH1D(f"{h1.GetName()}_fitRange", "", len(fitBinEdges)-1, array("d", fitBinEdges))
    for ib in range(1, 1+h1fitRange.GetNbinsX()):
        binOriginalHisto = ib + fitBinLow - 1
        h1fitRange.SetBinContent(ib, h1.GetBinContent(binOriginalHisto))
        h1fitRange.SetBinError(ib, h1.GetBinError(binOriginalHisto))
    boost_hist = narf.root_to_hist(h1fitRange)

    # FIT WITH TENSORFLOW
    global polN_scaled_global
    fitModel = f"pol{fitPolDegree}" # just for legend, set here to make sure it is changed consistently with line below
    params = np.array([1.0] + [0.0 for i in range(fitPolDegree)])
    if polN_scaled_global == None:
        polN_scaled_global = partial(polN_root_, xLowVal=xMinFit, xFitRange=xMaxFit-xMinFit, degree=fitPolDegree)

    fitres_tf1 = narf.fitutils.fit_hist(boost_hist, polN_scaled_global, params)
    f1 = ROOT.TF1("f1",polN_scaled_global, xMinFit, xMaxFit, len(params))
    status = fitres_tf1["status"]
    covstatus = fitres_tf1["covstatus"]
    if status:
        print(f"\n\n-----> STATUS: fit had status {status}\n\n")
        if status not in badFitsID.keys():
            badFitsID[status] = 1
        else:
            badFitsID[status] += 1                                       
    if covstatus:
        print(f"\n\n-----> COV: fit had covstatus {covstatus}\n\n")
        if covstatus not in badCovMatrixID.keys():
            badCovMatrixID[covstatus] = 1
        else:
            badCovMatrixID[covstatus] += 1                                       

    # get chi 2:
    chi2 = fitres_tf1["loss_val"]
    ndof = int(len(fitBinEdges)-1 - f1.GetNpar())
    chi2prob = ROOT.TMath.Prob(chi2, ndof)
    chi2text = f"#chi^{{2}} = {round(chi2,1)} / {ndof}"
    if chi2prob < 0.05:
        perc_chi2prob = 100.0 * chi2prob
        sign = "="
        if perc_chi2prob < 0.1:
            perc_chi2prob = 0.1
            sign = "<"
        chi2text += " (prob {} {}%)".format(sign, round(perc_chi2prob,1))
        
    global pol0_scaled_global
    if fitPolDegree == 1:
        params0 = np.array([1.0])
        if pol0_scaled_global == None:
            pol0_scaled_global = partial(polN_root_, xLowVal=xMinFit, xFitRange=xMaxFit-xMinFit, degree=0)
        fitres_tf1_pol0 = narf.fitutils.fit_hist(boost_hist, pol0_scaled_global, params0)
        fpol0 = ROOT.TF1("fpol0",pol0_scaled_global, xMinFit, xMaxFit, len(params0))
        chi2pol0 = fitres_tf1_pol0["loss_val"]
        chi2diffTest = ROOT.TMath.Prob(chi2pol0 - chi2, 1)
        if histoChi2diffTest:
            histoChi2diffTest.Fill(chi2diffTest)
        if histoPullsPol1Slope:
            histoPullsPol1Slope.Fill(fitres_tf1["x"][1]/np.sqrt(fitres_tf1["cov"][1][1]))
    f1.SetParameters( np.array( fitres_tf1["x"], dtype=np.dtype('d') ) )
    f1.SetLineWidth(3)
    #f1.SetLineStyle(9) # ROOT.kDashed == thin dashes, almost dotted
    f1.SetLineColor(ROOT.kBlue+1)
    # draw extrapolation
    f2 = ROOT.TF1("f2",polN_scaled_global, xMaxFit, h1.GetXaxis().GetBinLowEdge(1+h1.GetNbinsX()), len(params))
    for i in range(1+fitPolDegree):
        f2.SetParameter(i, f1.GetParameter(i))
    f2.SetLineColor(ROOT.kRed+2)
    f2.SetLineWidth(3)
    f2.SetLineStyle(9)

    xvalMin = h1.GetXaxis().GetBinLowEdge(1)
    xvalMax = h1.GetXaxis().GetBinUpEdge(h1.GetNbinsX())
    funcFullRange = ROOT.TF1("funcFullRange", polN_scaled_global, xvalMin, 200, len(params)) # define this one until "infinity"
    funcFullRange.SetParameters( np.array( fitres_tf1["x"], dtype=np.dtype('d') ) )
    hband = ROOT.TH1D("hband", "", 400, xvalMin, xvalMax)
    hband.SetStats(0)
    hband.SetFillColor(ROOT.kGray)
    #hband.SetFillStyle(3001)
    for ib in range(1, hband.GetNbinsX()+1):
        xval = hband.GetBinCenter(ib)
        val = f1.Eval(xval)
        #print(f"val({xval}) = {val}")
        hband.SetBinContent(ib, val)

    npar = len(params)
    # diagonalize and get eigenvalues and eigenvectors
    e, v = np.linalg.eigh(fitres_tf1["cov"])
    # store all variations for faster access below
    altParameters = np.array([np.zeros(npar, dtype=np.dtype('d'))] * (npar * 2), dtype=np.dtype('d'))
    #print(altParameters)
    for ivar in range(npar):
        shift = np.sqrt(e[ivar]) * v[:, ivar]
        altParameters[ivar]      = fitres_tf1["x"] + shift
        altParameters[ivar+npar] = fitres_tf1["x"] - shift

    tf1_func_alt = ROOT.TF1()
    tf1_func_alt.SetName("tf1_func_alt")
    funcFullRange.Copy(tf1_func_alt)
    tf1_func_alt.SetLineWidth(2)

    for ib in range(1, hband.GetNbinsX()+1):
        xval = hband.GetBinCenter(ib)
        err = 0.0
        for ivar in range(npar):
            # set parameters for a given hessian
            # make band using only up variations, the result for down ones should be very symmetric 
            tf1_func_alt.SetParameters(altParameters[ivar]) # this is for Up variations
            funcVal = tf1_func_alt.Eval(xval)
            diff = funcVal - hband.GetBinContent(ib)
            err += diff * diff
        err = math.sqrt(err)
        hband.SetBinError(ib, err)

    hband.Draw("E2 SAME")
    f1.Draw("L SAME")
    if xMaxFit < xvalMax:
        f2.Draw("L SAME")
    h1.Draw("EP SAME")
        
    nColumnsLeg = 1
    if ";" in legendCoords: 
        nColumnsLeg = int(legendCoords.split(";")[1])
    legcoords = [float(x) for x in (legendCoords.split(";")[0]).split(',')]
    lx1,lx2,ly1,ly2 = legcoords[0],legcoords[1],legcoords[2],legcoords[3]
    # resize if one line is not needed 
    if xMaxFit >= xvalMax:
        ly1 = ly2 - 0.75 * (ly2 - ly1)
    leg = ROOT.TLegend(lx1,ly1,lx2,ly2)
    #leg.SetFillColor(0)
    #leg.SetFillStyle(0)
    #leg.SetBorderSize(0)
        
    leg.SetNColumns(nColumnsLeg)
    leg.AddEntry(h1,"Measurement","PE")
    leg.AddEntry(f1,f"Fit {fitModel} in [{int(xMinFit)}, {int(xMaxFit)}]","L")
    if xMaxFit < xvalMax:
        leg.AddEntry(f2,f"Extrapolation","L")
    leg.AddEntry(hband,"Fit uncertainty","F")
    leg.Draw("same")
    canvas.RedrawAxis("sameaxis")

    if drawStatBox:
        ROOT.gPad.Update()
        ROOT.gStyle.SetOptStat(1110)
        ROOT.gStyle.SetOptFit(1102)
    else:
        h1.SetStats(0)

    # redraw legend, or vertical lines appear on top of it
    leg.Draw("same")

    if len(moreTextLatex):
        realtext = moreTextLatex.split("::")[0]
        x1,y1,ypass,textsize = 0.75,0.8,0.08,0.035
        if "::" in moreTextLatex:
            x1,y1,ypass,textsize = (float(x) for x in (moreTextLatex.split("::")[1]).split(","))            
        lat = ROOT.TLatex()
        lat.SetNDC();
        lat.SetTextFont(42)        
        lat.SetTextSize(textsize)
        for itx,tx in enumerate(realtext.split(";")):
            lat.DrawLatex(x1,y1-itx*ypass,tx)
        lat.DrawLatex(x1,y1-(itx+1)*ypass, chi2text)
            
    setTDRStyle()
    if leftMargin > 0.1:
        if lumi != None: CMS_lumi(canvas,lumi,True,False)
        else:            CMS_lumi(canvas,"",True,False)
    else:
        latCMS = ROOT.TLatex()
        latCMS.SetNDC();
        latCMS.SetTextFont(42)
        latCMS.SetTextSize(0.045)
        latCMS.DrawLatex(0.1, 0.95, '#bf{CMS} #it{Preliminary}')
        if lumi != None: latCMS.DrawLatex(0.85, 0.95, '%s fb^{-1} (13 TeV)' % lumi)
        else:            latCMS.DrawLatex(0.90, 0.95, '(13 TeV)' % lumi)

    if draw_both0_noLog1_onlyLog2 != 2:
        canvas.SaveAs(outdir + canvasName + ".png")
        canvas.SaveAs(outdir + canvasName + ".pdf")

    if draw_both0_noLog1_onlyLog2 != 1:        
        if yAxisName == "a.u.": 
            h1.GetYaxis().SetRangeUser(max(0.0001,h1.GetMinimum()*0.8),h1.GetMaximum()*100)
        else:
            h1.GetYaxis().SetRangeUser(max(0.001,h1.GetMinimum()*0.8),h1.GetMaximum()*100)
        canvas.SetLogy()
        canvas.SaveAs(outdir + canvasName + "_logY.png")
        canvas.SaveAs(outdir + canvasName + "_logY.pdf")
        canvas.SetLogy(0)

    # when deriving the average mT at which to evaluate the FRF, use the average of the FRF over the mT distribution
    # essentially integral of FRF(mT)*G(mT)*dmT, where G is the normalized shape of the mT distribution 

    averageFRF = 0.0
    averageFRFcap = 0.0
    averageFRFunc = {i : {"Up" : 0.0, "Down" : 0.0} for i in range(npar)}
    averageFRFunc["quadsum"] = {"Up" : 0.0, "Down" : 0.0}

    if mTshape:
        ## get integral in restricted range to normalize the mT PDF
        #mTshape.GetXaxis().SetRange(1, mTshape.GetXaxis().FindFixBin(xvalMax-0.001))

        integralMtNorm = 1./mTshape.Integral()
        for ib in range(1, 1+mTshape.GetNbinsX()):
            mTbinVal = mTshape.GetBinCenter(ib)
            mTbinValCap = min(mTbinVal, h1.GetXaxis().GetBinUpEdge(h1.GetNbinsX())) # assume constant after histogram range
            valFRF = h1.GetBinContent(max(1, min(h1.GetXaxis().FindFixBin(mTbinVal), h1.GetNbinsX()))) if useBinnedCorr else funcFullRange.Eval(mTbinVal)
            valFRFcap = h1.GetBinContent(max(1, min(h1.GetXaxis().FindFixBin(mTbinValCap), h1.GetNbinsX()))) if useBinnedCorr else funcFullRange.Eval(mTbinValCap)
            mTshapeBinContent = mTshape.GetBinContent(ib)
            addVal = max(0.0, valFRF) * mTshapeBinContent
            addValCap =  max(0.0, valFRFcap) * mTshapeBinContent
            averageFRF += addVal
            averageFRFcap += addValCap
            #print(f"{ib}: mT = {mTbinVal}, FRF={valFRF}, PDF(mT) = {mTshapeBinContent*integralMtNorm}, addVal = {addVal*integralMtNorm}")
            for ivar in range(npar):
                # set parameters for a given hessian
                tf1_func_alt.SetParameters(altParameters[ivar]) # this is for Up variations
                funcVal = max(0.0, tf1_func_alt.Eval(mTbinVal))
                averageFRFunc[ivar]["Up"] += (funcVal * mTshapeBinContent)
                tf1_func_alt.SetParameters(altParameters[ivar+npar]) # this is for Down variations
                funcVal = max(0.0, tf1_func_alt.Eval(mTbinVal))
                averageFRFunc[ivar]["Down"] += (funcVal * mTshapeBinContent)
        averageFRF *= integralMtNorm
        averageFRFcap *= integralMtNorm
        for ivar in range(npar):
            averageFRFunc[ivar]["Up"] *= integralMtNorm
            averageFRFunc[ivar]["Down"] *= integralMtNorm

    # also get total uncertainty summing in quadrature
    for updown in ["Up", "Down"]:
        averageFRFunc["quadsum"][updown] = 0.0
        for ivar in range(npar):
            averageFRFunc["quadsum"][updown] += pow(averageFRFunc[ivar][updown] - averageFRF, 2)
        averageFRFunc["quadsum"][updown] = np.sqrt(averageFRFunc["quadsum"][updown])
        if updown == "Up":
            averageFRFunc["quadsum"][updown] = averageFRF + averageFRFunc["quadsum"][updown]
        else:
            averageFRFunc["quadsum"][updown] = averageFRF - averageFRFunc["quadsum"][updown]
        
    return averageFRF, averageFRFcap, averageFRFunc

################################################################

# need to define functions as global below to avoid them being deleted out of fitTurnOnTF
polN_scaled_global = None
pol0_scaled_global = None # only needed with straight line fit, to perform check fitting with horizontal line
badFitsID = {}
badCovMatrixID = {}

def runStudy(charge, outfolder, rootfilename, args):

    hFRFcorr = None
    hFRFcorrUnc = None
    hFRFcorrTestCap = None
    hFRF_proj = None
    histoChi2diffTest = None
    histoPullsPol1Slope = None

    etaLabel = "#eta" if not args.absEta else "|#eta|"
    
    xAxisName = args.xAxisName
    yAxisName = args.yAxisName
    zAxisName = args.met + " " + args.zAxisName

    createPlotDirAndCopyPhp(outfolder)
    outfolder_dataMC = f"{outfolder}/shapesDataMC/"
    createPlotDirAndCopyPhp(outfolder_dataMC)

    adjustSettings_CMS_lumi()    
    canvas = ROOT.TCanvas("canvas","",800,800)
    canvas1D = ROOT.TCanvas("canvas1D","",800,700)
    canvas1Dshapes = ROOT.TCanvas("canvas1Dshapes","",700,800)

    canvas_unroll = ROOT.TCanvas("canvas_unroll","",3000,800)
    leftMargin = 0.06
    rightMargin = 0.01
    bottomMargin = 0.12
    canvas_unroll.SetTickx(1)
    canvas_unroll.SetTicky(1)
    canvas_unroll.cd()
    canvas_unroll.SetLeftMargin(leftMargin)
    canvas_unroll.SetRightMargin(rightMargin)
    canvas_unroll.cd()
    canvas_unroll.SetBottomMargin(bottomMargin)

    groups = Datagroups(args.inputfile[0])
    datasets = groups.getNames() # this has all the original defined groups
    datasetsNoQCD = list(filter(lambda x: x != "QCD", datasets)) # exclude QCD MC if present
    datasetsNoFakes = list(filter(lambda x: x != "Fake", datasets)) 
    datasetsNoQCDFakes = list(filter(lambda x: x not in ["QCD", "Fake"], datasets))
    logger.info(f"All original datasets available {datasets}")
    inputHistName = "mTStudyForFakes"
    groups.setNominalName(inputHistName)
    groups.loadHistsForDatagroups(inputHistName, syst="", procsToRead=datasets, applySelection=False)
    histInfo = groups.getDatagroups() # keys are same as returned by groups.getNames() 
    rootHists = {d: None for d in datasets} # for the fake study, eta-pt aleady rebinned
    rootHists_forplots = {d: None for d in datasets} # for 1D plots, mt rebinned by 2 but no eta-pt rebinning
    rootHists_asNominal = {d: None for d in datasets}
    rootHists_forMtWithDataDrivenFakes = {d: None for d in datasetsNoQCD}
    hnarf_fakerateDeltaPhi = None
    for d in datasets:
        #print(d)
        hnarf = histInfo[d].hists[inputHistName]
        hnarf_forplots = hnarf.copy()
        hnarf_asNominal = hnarf.copy()
        # rebin eta-pt for actual test with fakes, while collapsing other axes for the simple plots as needed,
        # this is only to avoid that THn are too big
        s = hist.tag.Slicer()
        hnarf = hnarf[{"eta" : s[::hist.rebin(args.rebinEta)]}]
        hnarf = hnarf[{"pt" : s[::hist.rebin(args.rebinPt)]}]
        # for fake rate versus deltaPhi
        # TODO: move to the other function specific for deltaPhi studies?
        if d == ("QCD" if args.useQCDMC else "Fake"):
            # note, with python/boost/hist slicing the second edge is excluded from projections,
            # also when it is used as actual bin id, so s[1:1:hist.sum] doesn't work, while s[1:2:hist.sum] picks bin 1
            hnarf_fakerateDeltaPhi = copy.deepcopy(hnarf)
            hnarf_fakerateDeltaPhi = hnarf_fakerateDeltaPhi[{"pt": s[0:complex(0,args.maxPt)]}]
            # rebin a bit more in pt and eta
            hnarf_fakerateDeltaPhi = hnarf_fakerateDeltaPhi[{"eta" : s[::hist.rebin(2)]}]
            hnarf_fakerateDeltaPhi = hnarf_fakerateDeltaPhi[{"pt" : s[::hist.rebin(2)]}]
            if args.jetCut:
                hnarf_fakerateDeltaPhi = hnarf_fakerateDeltaPhi[{"hasJets": True}]
            else:
                hnarf_fakerateDeltaPhi = hnarf_fakerateDeltaPhi[{"hasJets": s[::hist.sum]}]
            lowMtUpperBound = int(args.mtNominalRange.split(",")[1])
            hnarf_fakerateDeltaPhi = hnarf_fakerateDeltaPhi[{"mt" : s[:complex(0,lowMtUpperBound):hist.sum]}]
            chargeIndex = 0 if charge == "minus" else 1
            hnarf_fakerateDeltaPhi = hnarf_fakerateDeltaPhi[{"charge" : s[chargeIndex:chargeIndex+1:hist.sum]}]
            #rebin deltaPhi by 4 or 2 (16 bins originally, bu only 8 in newer files)
            hnarf_fakerateDeltaPhi = hnarf_fakerateDeltaPhi[{"DphiMuonMet" : s[::hist.rebin(2)]}]
            # make all TH of FRF in bins of dphi
            histo_fakes_dphiBins = narf.hist_to_root(hnarf_fakerateDeltaPhi) # this is a THnD with eta-pt-passIso-DphiMuonMet
            histo_FRFvsDphi = {}
            nBinsDphi = histo_fakes_dphiBins.GetAxis(3).GetNbins()
            for idp in range(nBinsDphi):
                idpBin = idp +1
                histo_fakes_dphiBins.GetAxis(3).SetRange(idpBin, idpBin)
                histo_fakes_dphiBins.GetAxis(2).SetRange(1, 1)
                histo_fakes_dphiBins_failIso = histo_fakes_dphiBins.Projection(1, 0, "E")
                histo_fakes_dphiBins_failIso.SetName(f"histo_fakes_dphiBins_failIso_idp{idp}")
                histo_fakes_dphiBins.GetAxis(3).SetRange(idpBin, idpBin)
                histo_fakes_dphiBins.GetAxis(2).SetRange(2, 2)
                histo_fakes_dphiBins_passIso = histo_fakes_dphiBins.Projection(1, 0, "E")
                histo_fakes_dphiBins_passIso.SetName(f"histo_fakes_dphiBins_passIso_idp{idp}")
                histo_FRFvsDphi[idp] = copy.deepcopy(histo_fakes_dphiBins_passIso.Clone(f"histo_FRFvsDphi_idp{idp}"))
                histo_FRFvsDphi[idp].Divide(histo_fakes_dphiBins_failIso)
            ##
            histo_fakes_dphiBins.GetAxis(3).SetRange(1, nBinsDphi)
            histo_fakes_dphiBins.GetAxis(2).SetRange(1, 1)
            histo_fakes_dphiBins_failIso = histo_fakes_dphiBins.Projection(1, 0, "E")
            histo_fakes_dphiBins_failIso.SetName("histo_fakes_dphiBins_failIso_idpInclusive")
            histo_fakes_dphiBins.GetAxis(3).SetRange(1, nBinsDphi)
            histo_fakes_dphiBins.GetAxis(2).SetRange(2, 2)
            histo_fakes_dphiBins_passIso = histo_fakes_dphiBins.Projection(1, 0, "E")
            histo_fakes_dphiBins_passIso.SetName("histo_fakes_dphiBins_passIso_idpInclusive")
            histo_FRFvsDphi["inclusive"] = copy.deepcopy(histo_fakes_dphiBins_passIso.Clone("histo_FRFvsDphi_idpInclusive"))
            histo_FRFvsDphi["inclusive"].Divide(histo_fakes_dphiBins_failIso)
            ##
            # now should plot
            hists = [unroll2Dto1D(histo_FRFvsDphi["inclusive"], newname=f"unrolled_{histo_FRFvsDphi['inclusive'].GetName()}", cropNegativeBins=False)]
            for idp in range(nBinsDphi):
                hists.append(unroll2Dto1D(histo_FRFvsDphi[idp], newname=f"unrolled_{histo_FRFvsDphi[idp].GetName()}", cropNegativeBins=False))
            
            legEntries = ["inclusive"] + [f"#Delta#phi bin {idp}" for idp in range(nBinsDphi)]
            ptBinRanges = []
            for ipt in range(histo_FRFvsDphi["inclusive"].GetNbinsY()):
                ptBinRanges.append("[{ptmin},{ptmax}] GeV".format(ptmin=int(histo_FRFvsDphi["inclusive"].GetYaxis().GetBinLowEdge(ipt+1)),
                                                                  ptmax=int(histo_FRFvsDphi["inclusive"].GetYaxis().GetBinLowEdge(ipt+2))))
            drawNTH1(hists, legEntries, "Unrolled #eta-p_{T} bin", "Fakerate factor",
                     f"FRFvsDeltaPhi_{charge}", outfolder,
                     leftMargin=0.06, rightMargin=0.01, labelRatioTmp="BinX/inclusive::0.7,1.3",
                     legendCoords="0.06,0.99,0.91,0.99;6", lowerPanelHeight=0.5, skipLumi=True, passCanvas=canvas_unroll,
                     drawVertLines="{a},{b}".format(a=histo_FRFvsDphi["inclusive"].GetNbinsY(),b=histo_FRFvsDphi["inclusive"].GetNbinsX()),
                     textForLines=ptBinRanges, transparentLegend=False, drawErrorAll=True,
                     onlyLineColor=True, noErrorRatioDen=False, useLineFirstHistogram=True, setOnlyLineRatio=True, lineWidth=1)
        ##
        dphiMuonMetCut = args.dphiMuonMetCut * np.pi
        hnarf = hnarf[{"DphiMuonMet" : s[complex(0, 0.01+dphiMuonMetCut):complex(0, np.pi):hist.sum]}] # test dphi cut
        rootHists[d] = narf.hist_to_root(hnarf) # this is a THnD with eta-pt-charge-mt-passIso-hasJets-DphiMuonMet
        hnarf_forplots = hnarf_forplots[{"DphiMuonMet" : s[complex(0, 0.01+dphiMuonMetCut):complex(0, np.pi)]}] # cut but do not integrate
        hnarf_forplots = hnarf_forplots[{"mt" : s[::hist.rebin(2)]}]
        rootHists_forplots[d] = narf.hist_to_root(hnarf_forplots) # this is a THnD with eta-pt-charge-mt-passIso-hasJets-DphiMuonMet
        # prepare nominal histograms in signal region for some plots
        # integrate njet and Dphi in appropriate range
        hnarf_asNominal = hnarf_asNominal[{"hasJets" : s[::hist.sum], # jet inclusive                  
                                           "DphiMuonMet" : s[complex(0, 0.01+dphiMuonMetCut):complex(0, np.pi):hist.sum]}]
        hnarf_forMtWithDataDrivenFakes = hnarf_asNominal.copy()
        mtThreshold = float(args.mtNominalRange.split(",")[-1])
        # now select signal region
        if d == "Fake":
            # do ABCD method for Fakes, using binned mT axis
            hnarf_asNominal = sel.fakeHistABCD(hnarf_asNominal,
                                               thresholdMT=mtThreshold,
                                               axis_name_mt="mt",
                                               integrateMT=True)
            hnarf_forMtWithDataDrivenFakes = sel.fakeHistABCD(hnarf_forMtWithDataDrivenFakes,
                                                              thresholdMT=mtThreshold,
                                                              axis_name_mt="mt",
                                                              integrateMT=False)
        else:
            nMtBins = hnarf_asNominal.axes["mt"].size
            # just select signal region: pass isolation and mT > mtThreshold with overflow included
            hnarf_asNominal = hnarf_asNominal[{"passIso" : True, 
                                               "mt" : s[complex(0,mtThreshold):hist.overflow:hist.sum],
                                               }]
            hnarf_forMtWithDataDrivenFakes = hnarf_forMtWithDataDrivenFakes[{"passIso" : True, 
                                                                             "mt" : s[complex(0,mtThreshold):nMtBins],
                                                                             }]
        rootHists_asNominal[d] = narf.hist_to_root(hnarf_asNominal)
        # now integrate eta-pt to keep mt
        nPtBins = hnarf_asNominal.axes["pt"].size
        hnarf_forMtWithDataDrivenFakes = hnarf_forMtWithDataDrivenFakes[{"eta" : s[::hist.sum],
                                                                         "pt"  : s[0:nPtBins:hist.sum],
                                                                         "charge" : 0 if charge == "minus" else 1,
                                                                         }]
        if d in datasetsNoQCD:
            rootHists_forMtWithDataDrivenFakes[d] = narf.hist_to_root(hnarf_forMtWithDataDrivenFakes)
    ########
    ########
    # data-MC already done in the management of the groups above
    # should check that the subtraction by hand yields the same results and uncertainties
    # the other processes are still needed to make other plots with data and MC, like mT in the different regions
    histo_fakes = copy.deepcopy(rootHists["QCD" if args.useQCDMC else "Fake"])    
    
    ### get the standard signal region with the data-driven fakes,
    ### it will be needed to make the plot with/without the corrections
    ### try using histogram "mTStudyForFakes", so the deltaPhi cut can be added/removed at will
    ### (the nominal histogram from the loop will always have the dphi cut since it is default in histmaker)
    #
    # inputHistName = "nominal"
    # groups.setNominalName(inputHistName)
    # groups.setSelectOp(sel.histWmass_passMT_passIso)
    # groups.setSelectOp(sel.fakeHistABCD, processes=["Fake"])
    # datasets_nominal = datasetsNoFakes if args.useQCDMC else datasetsNoQCD
    # groups.loadHistsForDatagroups(inputHistName, syst="", procsToRead=datasets_nominal)
    # histInfo_nominal = groups.getDatagroups()
    # rootHists_nominal = {d: None for d in datasets_nominal}
    # for d in rootHists_nominal.keys():
    #     hnarf_nominal = histInfo_nominal[d].hists[inputHistName]
    #     rootHists_nominal[d] = narf.hist_to_root(hnarf_nominal) # this is a TH3D with eta-pt-charge
    #######
    datasets_nominal = datasetsNoFakes if args.useQCDMC else datasetsNoQCD
    rootHists_nominal = {d: None for d in datasets_nominal}
    for d in rootHists_nominal.keys():
        rootHists_nominal[d] = rootHists_asNominal[d] # this is a TH3D with eta-pt-charge
        
    # get a copy of the eta-pt-charge histogram to use as template with same binning as the analysis histograms
    # it will be filled with the correction uncertainty to be used as a systematic
    # adding charge in its name even if this is a TH3: in this way we will have the correction for plus defined for both charges,
    # but the wrong charge is filled with 0, so that it will be a dummy variation equal to nominal when it is used in setupCombineWmass.py
    etaPtChargeTemplate = copy.deepcopy(rootHists_nominal[d].Clone(f"etaPtCharge_mtCorrection_{charge}"))
    etaPtChargeTemplate.SetTitle("FRF correction uncertainty (quadrature sum, Up)")
    etaPtChargeTemplate.Reset("ICESM")
        
    axisVar = {0 : ["muon_eta", "#eta"],
               1 : ["muon_pt",  "p_{T} (GeV)"],
               3 : ["mT", args.met + " m_{T} (GeV)"],
               6 : ["deltaPhiMuonMET", "#Delta#phi(#mu,MET) (GeV)"]
    }
               
    # bin number from root histogram
    chargeBin = 1 if charge == "minus" else 2

    # plot mT with data-driven fakes, as a test (should add the mT correction in fact)
    hmcMt = {}
    for d in datasetsNoQCD:
        if d == "Data":
            continue
        hmcMt[d] = rootHists_forMtWithDataDrivenFakes[d]
    hdataMt = rootHists_forMtWithDataDrivenFakes["Data"]
    plotDistribution1D(hdataMt, hmcMt, datasetsNoQCD,
                       outfolder_dataMC, canvas1Dshapes=canvas1Dshapes,
                       xAxisName=f"{args.met} m_{{T}} (GeV)", plotName=f"mT_passIso_jetInclusive_passMT_dataDriveFakes")
    
    
    # plot mT, eta, pt in some regions iso-nJet regions, for checks
    # don't plot fakes here
    for xbin in axisVar.keys():
        # Dphi in 4 mT bins, for jet inclusive and pass/fail iso
        if xbin == 6:
            mtRanges = [0, 20, 40, 60, -1]
            for imt in range(len(mtRanges)-1):
                mtLow = str(mtRanges[imt])
                mtHigh = "inf" if mtRanges[imt+1] < 0 else str(mtRanges[imt+1])
                mtTitle = f"mT{mtLow}to{mtHigh}"
                mtRange = [mtRanges[imt], mtRanges[imt+1]]
                plotProjection1D(rootHists_forplots, datasetsNoFakes, outfolder_dataMC, canvas1Dshapes=canvas1Dshapes,
                                 chargeBin=chargeBin,
                                 projectAxisToKeep=xbin, xAxisName=axisVar[xbin][1],
                                 plotName=f"{axisVar[xbin][0]}_failIso_jetInclusive_{mtTitle}",
                                 isoAxisRange=[1,1], jetAxisRange=[1,2],
                                 mTvalueRange=mtRange)
                plotProjection1D(rootHists_forplots, datasetsNoFakes, outfolder_dataMC, canvas1Dshapes=canvas1Dshapes,
                                 chargeBin=chargeBin,
                                 projectAxisToKeep=xbin, xAxisName=axisVar[xbin][1],
                                 plotName=f"{axisVar[xbin][0]}_passIso_jetInclusive_{mtTitle}",
                                 isoAxisRange=[2,2], jetAxisRange=[1,2],
                                 mTvalueRange=mtRange)
            continue  # no need for the rest of the plots for this variable
        ######
        
        plotProjection1D(rootHists_forplots, datasetsNoFakes, outfolder_dataMC,
                         canvas1Dshapes=canvas1Dshapes, chargeBin=chargeBin,
                         projectAxisToKeep=xbin, xAxisName=axisVar[xbin][1], plotName=f"{axisVar[xbin][0]}_passIso_1orMoreJet",
                         isoAxisRange=[2,2], jetAxisRange=[2,2])
        plotProjection1D(rootHists_forplots, datasetsNoFakes, outfolder_dataMC,
                         canvas1Dshapes=canvas1Dshapes, chargeBin=chargeBin,
                         projectAxisToKeep=xbin, xAxisName=axisVar[xbin][1], plotName=f"{axisVar[xbin][0]}_passIso_jetInclusive",
                         isoAxisRange=[2,2], jetAxisRange=[1,2])
        plotProjection1D(rootHists_forplots, datasetsNoFakes, outfolder_dataMC,
                         canvas1Dshapes=canvas1Dshapes, chargeBin=chargeBin,
                         projectAxisToKeep=xbin, xAxisName=axisVar[xbin][1], plotName=f"{axisVar[xbin][0]}_failIso_1orMoreJet",
                         isoAxisRange=[1,1], jetAxisRange=[2,2])
        plotProjection1D(rootHists_forplots, datasetsNoFakes, outfolder_dataMC,
                         canvas1Dshapes=canvas1Dshapes, chargeBin=chargeBin,
                         projectAxisToKeep=xbin, xAxisName=axisVar[xbin][1], plotName=f"{axisVar[xbin][0]}_failIso_jetInclusive",
                         isoAxisRange=[1,1], jetAxisRange=[1,2])
        # signal region adding mT cut too
        plotProjection1D(rootHists_forplots, datasetsNoFakes, outfolder_dataMC,
                         canvas1Dshapes=canvas1Dshapes, chargeBin=chargeBin,
                         projectAxisToKeep=xbin, xAxisName=axisVar[xbin][1], plotName=f"{axisVar[xbin][0]}_passIso_jetInclusive_passMt",
                         isoAxisRange=[2,2], jetAxisRange=[1,2], mTvalueRange=[40,-1])

    ###################################
    ###################################
    ###   Now the actual study on fakes
    ###################################
    ###################################

    # select events with at least one jet (or jet inclusive depending on what you need to do)
    # jetInclusive: GetAxis(5).SetRange(1, 2)
    # 1orMoreJet:   GetAxis(5).SetRange(2, 2)

    # set charge from charge axis
    histo_fakes.GetAxis(2).SetRange(chargeBin, chargeBin)
    histo_fakes.GetAxis(4).SetRange(2, 2) # passIso, equivalent to lowIso
    histo_fakes.GetAxis(5).SetRange(2 if args.jetCut else 1, 2)
    # now get a TH3
    histoPassIso = histo_fakes.Projection(0, 1, 3, "E")
    histoPassIso.SetName("fakes_passIso")
    histoPassIso.SetTitle("fakes_passIso")
    histo_fakes.GetAxis(2).SetRange(chargeBin, chargeBin)
    histo_fakes.GetAxis(4).SetRange(1, 1) # FailIso
    histo_fakes.GetAxis(5).SetRange(2 if args.jetCut else 1, 2)
    histoFailIso = histo_fakes.Projection(0, 1, 3, "E")
    histoFailIso.SetName("fakes_failIso")
    histoFailIso.SetTitle("fakes_failIso")

    # to get the mean only in the desired mt range, get projection using only mt bins in the high mt region
    # this is always done in the jet inclusive case
    mtThreshold = float(args.mtNominalRange.split(",")[-1])
    histo_fakes.GetAxis(2).SetRange(chargeBin, chargeBin)
    histo_fakes.GetAxis(3).SetRange(histo_fakes.GetAxis(3).FindFixBin(mtThreshold+0.001),
                                    histo_fakes.GetAxis(3).GetNbins()) # high mT
    histo_fakes.GetAxis(4).SetRange(1, 1) # FailIso
    histo_fakes.GetAxis(5).SetRange(1, 2) # jet inclusive
    histoPassMtFailIso = histo_fakes.Projection(0, 1, 3, "E")
    histoPassMtFailIso.SetName("fakes_passMt_failIso_jetInclusive")
    histoPassMtFailIso.SetTitle("fakes_passMt_failIso_jetInclusive")

    cropNegativeContent(histoPassIso)
    cropNegativeContent(histoFailIso)
    cropNegativeContent(histoPassMtFailIso)

    # compute FRF vs eta-pt and original binning to reproduce the nominal histograms and make plots
    # but don't use the jet cut here, to see how it gets without it
    # also integrate mt < mtThreshold
    histo_fakes.GetAxis(2).SetRange(chargeBin, chargeBin)
    histo_fakes.GetAxis(3).SetRange(1, histo_fakes.GetAxis(3).FindFixBin(mtThreshold-0.001))
    histo_fakes.GetAxis(4).SetRange(2, 2) # passIso, equivalent to lowIso
    histo_fakes.GetAxis(5).SetRange(1, 2) # always integrate jet here
    # now get a TH2 for passIso
    histoPassIsoLowMt_jetInclusive_vsEtaPt = histo_fakes.Projection(1, 0, "E")
    histoPassIsoLowMt_jetInclusive_vsEtaPt.SetName("fakes_passIsoLowMt_jetInclusive_vsEtaPt")
    histoPassIsoLowMt_jetInclusive_vsEtaPt.SetTitle("fakes_passIsoLowMt_jetInclusive_vsEtaPt")
    cropNegativeContent(histoPassIsoLowMt_jetInclusive_vsEtaPt)
    # now get a TH2 for failIso
    histo_fakes.GetAxis(4).SetRange(1, 1) # failIso, equivalent to highIso
    histoFailIsoLowMt_jetInclusive_vsEtaPt = histo_fakes.Projection(1, 0, "E")
    histoFailIsoLowMt_jetInclusive_vsEtaPt.SetName("fakes_failIsoLowMt_jetInclusive_vsEtaPt")
    histoFailIsoLowMt_jetInclusive_vsEtaPt.SetTitle("fakes_failIsoLowMt_jetInclusive_vsEtaPt")
    cropNegativeContent(histoFailIsoLowMt_jetInclusive_vsEtaPt)
    # get FRF vs etapt for jetInclusive events
    FRF_jetInclusive_vsEtaPt = copy.deepcopy(histoPassIsoLowMt_jetInclusive_vsEtaPt.Clone(f"FRF_jetInclusive_vsEtaPt_{charge}"))
    FRF_jetInclusive_vsEtaPt.SetTitle(f"jet inclusive, mT < {mtThreshold} GeV")
    FRF_jetInclusive_vsEtaPt.Divide(histoFailIsoLowMt_jetInclusive_vsEtaPt)
    # now get the data-driven QCD prediction vs eta-pt, start by cloning histoPassMtFailIso integrating mT        
    #dataDrivenQCDtemplate_jetInclusive = getTH2fromTH3(histoPassMtFailIso, "dataDrivenQCDtemplate_jetInclusive", 1, histoPassMtFailIso.GetNbinsZ())
    #dataDrivenQCDtemplate_jetInclusive.Multiply(FRF_jetInclusive_vsEtaPt)

    ## now redo the same for the >=1 jet region
    ## could be done from previous TH3 histograms integrating in mT, but they don't always have the same jet cut 
    histo_fakes.GetAxis(4).SetRange(2, 2) # passIso, equivalent to lowIso
    histo_fakes.GetAxis(5).SetRange(2, 2) # always integrate jet here
    # now get a TH2 for passIso
    histoPassIsoLowMt_1orMoreJet_vsEtaPt = histo_fakes.Projection(1, 0, "E")
    histoPassIsoLowMt_1orMoreJet_vsEtaPt.SetName("fakes_passIsoLowMt_1orMoreJet_vsEtaPt")
    histoPassIsoLowMt_1orMoreJet_vsEtaPt.SetTitle("fakes_passIsoLowMt_1orMoreJet_vsEtaPt")
    cropNegativeContent(histoPassIsoLowMt_1orMoreJet_vsEtaPt)
    # now get a TH2 for failIso
    histo_fakes.GetAxis(4).SetRange(1, 1) # failIso, equivalent to highIso
    histoFailIsoLowMt_1orMoreJet_vsEtaPt = histo_fakes.Projection(1, 0, "E")
    histoFailIsoLowMt_1orMoreJet_vsEtaPt.SetName("fakes_failIsoLowMt_1orMoreJet_vsEtaPt")
    histoFailIsoLowMt_1orMoreJet_vsEtaPt.SetTitle("fakes_failIsoLowMt_1orMoreJet_vsEtaPt")
    cropNegativeContent(histoFailIsoLowMt_1orMoreJet_vsEtaPt)
    # get FRF vs etapt for 1orMoreJet events
    FRF_1orMoreJet_vsEtaPt = copy.deepcopy(histoPassIsoLowMt_1orMoreJet_vsEtaPt.Clone(f"FRF_1orMoreJet_vsEtaPt_{charge}"))
    FRF_1orMoreJet_vsEtaPt.SetTitle(f">= 1 jet, mT < {mtThreshold} GeV")
    FRF_1orMoreJet_vsEtaPt.Divide(histoFailIsoLowMt_1orMoreJet_vsEtaPt)
    # now get the data-driven QCD prediction vs eta-pt, start by cloning histoPassMtFailIso integrating mT        
    #dataDrivenQCDtemplate_1orMoreJet = getTH2fromTH3(histoPassMtFailIso, "dataDrivenQCDtemplate_1orMoreJet", 1, histoPassMtFailIso.GetNbinsZ())
    #dataDrivenQCDtemplate_1orMoreJet.Multiply(FRF_1orMoreJet_vsEtaPt)

    # make some plots
    drawCorrelationPlot(FRF_jetInclusive_vsEtaPt,
                        xAxisName,
                        yAxisName,
                        "Fakerate factor",
                        FRF_jetInclusive_vsEtaPt.GetName(), plotLabel="ForceTitle", outdir=outfolder,
                        draw_both0_noLog1_onlyLog2=1, nContours=args.nContours, palette=args.palette,
                        invertePalette=args.invertePalette, passCanvas=canvas, skipLumi=True)
    drawCorrelationPlot(FRF_1orMoreJet_vsEtaPt,
                        xAxisName,
                        yAxisName,
                        "Fakerate factor",
                        FRF_1orMoreJet_vsEtaPt.GetName(), plotLabel="ForceTitle", outdir=outfolder,
                        draw_both0_noLog1_onlyLog2=1, nContours=args.nContours, palette=args.palette,
                        invertePalette=args.invertePalette, passCanvas=canvas, skipLumi=True)
    # and also the ratio between jet inclusive and >=1 jet
    FRF_ratioJet_vsEtaPt = copy.deepcopy(FRF_jetInclusive_vsEtaPt.Clone(f"FRF_ratioJet_vsEtaPt_{charge}"))
    FRF_ratioJet_vsEtaPt.SetTitle(f"jetInclusive/1orMoreJet, mT < {mtThreshold} GeV")
    FRF_ratioJet_vsEtaPt.Divide(FRF_1orMoreJet_vsEtaPt)
    drawCorrelationPlot(FRF_ratioJet_vsEtaPt,
                        xAxisName,
                        yAxisName,
                        "Ratio of fakerate factors::0.8,1.6",
                        FRF_ratioJet_vsEtaPt.GetName(), plotLabel="ForceTitle", outdir=outfolder,
                        draw_both0_noLog1_onlyLog2=1, nContours=args.nContours, palette=args.palette,
                        invertePalette=args.invertePalette, passCanvas=canvas, skipLumi=True)

    # don't rebin mT here (default is 1 GeV),
    # otherwise the new binning might no longer be consistent with the one passed to --mt-bin-edges
    #histoPassIso.Rebin3D(args.rebinEta, args.rebinPt, 1)
    #histoFailIso.Rebin3D(args.rebinEta, args.rebinPt, 1)
    #histoPassMtFailIso.Rebin3D(args.rebinEta, args.rebinPt, 1)

    mtEdges = [round(int(x),1) for x in args.mtBinEdges.split(',')] 
    nMtBins = len(mtEdges) -1
    ratio = []
    for imt in range(nMtBins):
        lowEdge = mtEdges[imt]
        highEdge = mtEdges[imt+1]
        binStart = histoPassIso.GetZaxis().FindFixBin(lowEdge)
        binEnd = histoPassIso.GetZaxis().FindFixBin(highEdge+0.001) - 1 # bin up edges belong to "next" bin
        h2PassIso = getTH2fromTH3(histoPassIso, f"pt_eta_mt{lowEdge}to{highEdge}_passIso", binStart, binEnd)
        h2FailIso = getTH2fromTH3(histoFailIso, f"pt_eta_mt{lowEdge}to{highEdge}_failIso", binStart, binEnd)

        h2PassIso.SetTitle("Low Iso: %s m_{T} #in [%d, %d]" % (args.met, lowEdge, highEdge))
        h2FailIso.SetTitle("High Iso: %s m_{T} #in [%d, %d]" % (args.met, lowEdge, highEdge))
        if not args.skipPlot2D:
            drawCorrelationPlot(h2PassIso,
                                xAxisName,
                                yAxisName,
                                "Events (data - MC)",
                                h2PassIso.GetName(), plotLabel="ForceTitle", outdir=outfolder,
                                draw_both0_noLog1_onlyLog2=1, nContours=args.nContours, palette=args.palette,
                                invertePalette=args.invertePalette, passCanvas=canvas, skipLumi=True)
            drawCorrelationPlot(h2FailIso,
                                xAxisName,
                                yAxisName,
                                "Events (data - MC)",
                                h2FailIso.GetName(), plotLabel="ForceTitle", outdir=outfolder,
                                draw_both0_noLog1_onlyLog2=1, nContours=args.nContours, palette=args.palette,
                                invertePalette=args.invertePalette, passCanvas=canvas, skipLumi=True)

        ratio.append(h2PassIso.Clone(f"fakerateFactor_mt{lowEdge}to{highEdge}"))
        ratio[imt].SetTitle("%s m_{T} #in [%d, %d]" % (args.met, lowEdge, highEdge))
        ratio[imt].Divide(h2FailIso)
        if not args.skipPlot2D:
            drawCorrelationPlot(ratio[imt],
                                xAxisName,
                                yAxisName,
                                "fakerate factor: N(iso) / N(not-iso)::0,3",
                                ratio[imt].GetName(), plotLabel="ForceTitle", outdir=outfolder,
                                draw_both0_noLog1_onlyLog2=1, nContours=args.nContours, palette=args.palette,
                                invertePalette=args.invertePalette, passCanvas=canvas, skipLumi=True)


    if args.mtNominalRange:
        lowEdge, highEdge = map(int, args.mtNominalRange.split(','))
        binStart = histoPassIso.GetZaxis().FindFixBin(lowEdge)
        binEnd = histoPassIso.GetZaxis().FindFixBin(highEdge+0.001) - 1 # bin up edges belong to "next" bin
        h2PassIso = getTH2fromTH3(histoPassIso, f"pt_eta_nominalmt{lowEdge}to{highEdge}_passIso", binStart, binEnd)
        h2FailIso = getTH2fromTH3(histoFailIso, f"pt_eta_nominalmt{lowEdge}to{highEdge}_failIso", binStart, binEnd)
        cropNegativeContent(h2PassIso)
        cropNegativeContent(h2FailIso)
        nominalFakerateFactor = h2PassIso.Clone(f"nominalFakerateFactor_mt{lowEdge}to{highEdge}")
        nominalFakerateFactor.SetTitle("%s m_{T} #in [%d, %d]" % (args.met, lowEdge, highEdge))        
        nominalFakerateFactor.Divide(h2FailIso)

    # rounding with python is sometimes tricky, add an epsilon to get the desired number
    etaLow = round(0.01 + h2PassIso.GetXaxis().GetBinLowEdge(1), 1)
    etaHigh = round(0.01 + h2PassIso.GetXaxis().GetBinLowEdge(1+h2PassIso.GetNbinsX()), 1)
    ptLow = round(0.01 + h2PassIso.GetYaxis().GetBinLowEdge(1), 1)
    ptHigh = round(0.01 + h2PassIso.GetYaxis().GetBinLowEdge(1+h2PassIso.GetNbinsY()), 1)

    if args.maxPt > 0:
        nPtBins = min(h2PassIso.GetNbinsY(), max(1, h2PassIso.GetYaxis().FindFixBin(args.maxPt-0.001)))
    else:
        nPtBins = h2PassIso.GetNbinsY()
    rangeMaxPt = round(h2PassIso.GetYaxis().GetBinLowEdge(nPtBins+1)+0.01,1)
    hFRFcorr = ROOT.TH2D(f"fakerateFactorCorrection_{charge}", "%s m_{T} > %d GeV" % (args.met, int(args.mtNominalRange.split(',')[1])),
                                 h2PassIso.GetNbinsX(), round(etaLow,1), round(etaHigh,1),
                                 nPtBins, round(ptLow,1), rangeMaxPt)
    hFRFcorrTestCap = copy.deepcopy(hFRFcorr.Clone(f"fakerateFactorCorrTestCap_{charge}"))
    if args.fitPolDegree == 1:
        histoChi2diffTest = ROOT.TH1D(f"histoChi2diffTest_{charge}", "Probability(#chi^{2}_{0} - #chi^{2}_{1})", 10, 0, 1)
        histoPullsPol1Slope = ROOT.TH1D(f"histoPullsPol1Slope_{charge}", "Pulls for slope parameter in pol1 fits", 20, -5, 5)

    # now preparing a summary for each pt bin
    for ipt in range(1, 1+nPtBins):
        ptBinLow = int(h2PassIso.GetYaxis().GetBinLowEdge(ipt))
        ptBinHigh = int(h2PassIso.GetYaxis().GetBinLowEdge(ipt+1))
        fakerateFactor_vs_etaMt = ROOT.TH2D("fakerateFactor_vs_etaMt_pt%dto%d" % (ptBinLow, ptBinHigh),
                                            "Muon p_{T} #in [%d, %d] GeV" % (ptBinLow, ptBinHigh),
                                            h2PassIso.GetNbinsX(), round(etaLow,1), round(etaHigh,1),
                                            nMtBins, array("d", mtEdges)
                                           )

        outfolder1D = outfolder + "fakerateFactor_fits_pt%dto%d/" % (ptBinLow, ptBinHigh)
        createPlotDirAndCopyPhp(outfolder1D)


        for ieta in range(1, 1+fakerateFactor_vs_etaMt.GetNbinsX()):

            etaBinLowNoRound = fakerateFactor_vs_etaMt.GetXaxis().GetBinLowEdge(ieta)
            etaBinHighNoRound = fakerateFactor_vs_etaMt.GetXaxis().GetBinLowEdge(ieta+1)
            etaBinLow =  round(0.01 + etaBinLowNoRound, 1)
            etaBinHigh = round(0.01 + etaBinHighNoRound, 1)
            # print(f"ieta = {ieta}    {ptBinLow} < pt < {ptBinHigh}     {etaBinLow} < eta < {etaBinHigh}    {etaBinLow} < etaNoRound < {etaBinHigh}")
            hFRfactorVsMt = ROOT.TH1D(f"hFRfactorVsMt_ieta{ieta}_pt{ptBinLow}to{ptBinHigh}",
                                      "%.1f < %s < %.1f, p_{T} #in [%d, %d] GeV" % (etaBinLow, etaLabel, etaBinHigh, ptBinLow, ptBinHigh),
                                      nMtBins, array("d", mtEdges))

            # to make easier computation of correction factor below
            hTmp = []

            for imt in range(1, 1+fakerateFactor_vs_etaMt.GetNbinsY()):
                binContent = ratio[imt-1].GetBinContent(ieta, ipt)
                binError = ratio[imt-1].GetBinError(ieta, ipt)
                fakerateFactor_vs_etaMt.SetBinContent(ieta, imt, binContent)
                fakerateFactor_vs_etaMt.SetBinError(ieta, imt, binError)
                hFRfactorVsMt.SetBinContent(imt, binContent)
                hFRfactorVsMt.SetBinError(  imt, binError)
                if nMtBins == 2:
                    hTmp.append(ROOT.TH1D(f"hTmp{imt}","",1,0,1))
                    hTmp[imt-1].SetBinContent(1, max(0.0, binContent))
                    hTmp[imt-1].SetBinError(  1, binError)

            textLatex = "%.1f < %s < %.1f;%d < p_{T} < %d GeV::0.2,0.88,0.08,0.045" % (etaBinLow, etaLabel, etaBinHigh,
                                                                                       ptBinLow, ptBinHigh)
            if nMtBins > 2:
                ## get high mt value to evaluate the correction, can use the mean of the mt distribution for each etapt bin 
                if args.integralMtMethod == "sideband":
                    projMt = histoPassMtFailIso.ProjectionZ(f"projZ_{histoPassMtFailIso.GetName()}", ieta, ieta, ipt, ipt, "eo")
                elif args.integralMtMethod == "fullRange":
                    ## or maybe in the full range to get the actual average
                    histoFailIso.GetZaxis().SetRange(1, histoFailIso.GetNbinsZ())
                    projMt = histoFailIso.ProjectionZ(f"projZ_{histoFailIso.GetName()}", ieta, ieta, ipt, ipt, "eo")
                cropNegativeContent(projMt)
                valFRF, valFRFCap, uncFRF = drawAndFitFRF(hFRfactorVsMt, zAxisName, "Fakerate factor: N(iso) / N(not-iso)",
                                                          hFRfactorVsMt.GetName(),
                                                          outfolder1D, passCanvas=canvas1D,
                                                          moreTextLatex=textLatex,
                                                          legendCoords="0.64,0.96,0.69,0.93", fitRange=args.mtFitRange,
                                                          mTshape=projMt,
                                                          fitPolDegree=args.fitPolDegree,
                                                          useBinnedCorr=args.useBinnedCorrection,
                                                          histoChi2diffTest=histoChi2diffTest,
                                                          histoPullsPol1Slope=histoPullsPol1Slope)
                #print(f"{valFRF}, {valFRFCap}")
                if valFRF < 0:
                    printLine(marker=" ")
                    printLine()
                    logger.warning(f"Warning: ieta = {ieta},   ipt = {ipt},   FRF = {valFRF}")
                    logger.warning("Setting FRF to 0.01!")
                    printLine()
                    printLine(marker=" ")
                    valFRF = 0.01
                for k in uncFRF.keys():
                    if any(x < 0.0 for x in uncFRF[k].values()):
                        logger.warning(f"var {k}: Up/Down = {uncFRF[k]['Up']} / {uncFRF[k]['Down']}")
                inverseNomiFRF = 1.0/nominalFakerateFactor.GetBinContent(ieta, ipt)
                hFRFcorr.SetBinContent(ieta, ipt, valFRF * inverseNomiFRF)
                hFRFcorrTestCap.SetBinContent(ieta, ipt, valFRFCap * inverseNomiFRF)
                if hFRFcorrUnc == None:
                    hFRFcorrUnc = {}
                    for key in uncFRF.keys():
                        hFRFcorrUnc[f"{key}_Up"] = copy.deepcopy(hFRFcorr.Clone(f"hFRFcorrUnc_{key}Up_{charge}"))
                        hFRFcorrUnc[f"{key}_Up"].Reset("ICESM")
                        hFRFcorrUnc[f"{key}_Down"] = copy.deepcopy(hFRFcorr.Clone(f"hFRFcorrUnc_{key}Down_{charge}"))
                        hFRFcorrUnc[f"{key}_Down"].Reset("ICESM")
                for key in uncFRF.keys():
                    hFRFcorrUnc[f"{key}_Up"].SetBinContent(ieta, ipt, uncFRF[key]["Up"] * inverseNomiFRF )
                    hFRFcorrUnc[f"{key}_Down"].SetBinContent(ieta, ipt, uncFRF[key]["Down"] * inverseNomiFRF)
            elif nMtBins == 2:
                hTmp[1].Divide(hTmp[0])
                hFRFcorr.SetBinContent(ieta, ipt, hTmp[1].GetBinContent(1))
                hFRFcorr.SetBinError(  ieta, ipt, hTmp[1].GetBinError(1))
                drawSingleTH1(hFRfactorVsMt, zAxisName, "Fakerate factor: N(iso) / N(not-iso)", hFRfactorVsMt.GetName(),
                              outfolder1D, lowerPanelHeight=0.0, passCanvas=canvas1D, moreTextLatex=textLatex,
                              legendCoords="0.64,0.96,0.77,0.93")

        if not args.skipPlot2D:
            drawCorrelationPlot(fakerateFactor_vs_etaMt,
                                xAxisName,
                                zAxisName,
                                "fakerate factor: N(iso) / N(not-iso)",
                                fakerateFactor_vs_etaMt.GetName(), plotLabel="ForceTitle", outdir=outfolder,
                                draw_both0_noLog1_onlyLog2=1, nContours=args.nContours, palette=args.palette,
                                invertePalette=args.invertePalette, passCanvas=canvas, skipLumi=True)

    if args.integralMtMethod == "sideband":
        if args.jetCut:
            corrRange = "0,1.0"
        else:
            corrRange = "0.7,1.3"
    else:
        minCorr = hFRFcorr.GetBinContent(hFRFcorr.GetMinimumBin())
        maxCorr = hFRFcorr.GetBinContent(hFRFcorr.GetMaximumBin())
        diff = max(abs(maxCorr-1.0), abs(minCorr-1.0))
        corrRange = f"{1.0-diff},{1.0+diff}"
        corrRange = "0.5,1.5"

    drawCorrelationPlot(hFRFcorr,
                        xAxisName,
                        yAxisName,
                        f"QCD template correction::{corrRange}",
                        hFRFcorr.GetName(), plotLabel="ForceTitle", outdir=outfolder,
                        draw_both0_noLog1_onlyLog2=1, nContours=args.nContours, palette=args.palette,
                        invertePalette=args.invertePalette, passCanvas=canvas, skipLumi=True)
    if args.fitPolDegree:
        drawTH1(histoChi2diffTest,
                "#chi^{2} difference probability (pol0 versus pol1)",
                f"Events::0,{1.2*histoChi2diffTest.GetBinContent(histoChi2diffTest.GetMaximumBin())}",
                histoChi2diffTest.GetName(), outfolder, passCanvas=canvas1D,
                statBoxSpec=210, skipTdrStyle=True)
        drawTH1(histoPullsPol1Slope,
                "Pulls", "Events",
                histoPullsPol1Slope.GetName(), outfolder, passCanvas=canvas1D,
                statBoxSpec=210, fitString="gaus;LEMSQ+;;-5;5", skipTdrStyle=True)

    if hFRFcorrUnc is not None:
        # do some unrolling and plot FRF correction with uncertainty bands
        nomi_unrolled = unroll2Dto1D(hFRFcorr, newname=f"unrolled_{hFRFcorr.GetName()}", cropNegativeBins=False)
        hList = [nomi_unrolled]
        legEntries = ["nomi"]
        colorVec = [ROOT.kRed+2, ROOT.kRed+1,
                    ROOT.kAzure+2, ROOT.kAzure+1,
                    ROOT.kOrange+2, ROOT.kOrange+1,
                    ROOT.kGreen+2, ROOT.kGreen+1] # colors only passed for variations
        if hFRFcorrTestCap:
            hList.append( unroll2Dto1D(hFRFcorrTestCap,
                                       newname=f"unrolled_{hFRFcorrTestCap.GetName()}",
                                       cropNegativeBins=False) )
            legEntries.append( "syst(cap FRF)" )
            colorVec = [ROOT.kGreen+3] + colorVec
        ptBinRanges = []
        for ipt in range(hFRFcorr.GetNbinsY()):
            ptBinRanges.append("[{ptmin},{ptmax}] GeV".format(ptmin=int(hFRFcorr.GetYaxis().GetBinLowEdge(ipt+1)),
                                                              ptmax=int(hFRFcorr.GetYaxis().GetBinLowEdge(ipt+2))))

        for ivar in uncFRF.keys():
            if ivar == "quadsum":
                continue
            hList.append( unroll2Dto1D(hFRFcorrUnc[f"{ivar}_Up"],
                                       newname=f"unrolled_{hFRFcorrUnc[f'{ivar}_Up'].GetName()}",
                                       cropNegativeBins=False) )
            hList.append( unroll2Dto1D(hFRFcorrUnc[f"{ivar}_Down"],
                                       newname=f"unrolled_{hFRFcorrUnc[f'{ivar}_Down'].GetName()}",
                                       cropNegativeBins=False) )
            legEntries.extend([f"{ivar}_Up", f"{ivar}_Down"])
        drawNTH1(hList, legEntries, "Unrolled eta-p_{T} bin", "FRF correction", f"FRFcorrAndUnc_etaPt_{charge}", outfolder,
                 leftMargin=0.06, rightMargin=0.01, labelRatioTmp="Syst/nomi",
                 legendCoords="0.06,0.99,0.91,0.99;6", lowerPanelHeight=0.5, skipLumi=True, passCanvas=canvas_unroll,
                 drawVertLines="{a},{b}".format(a=hFRFcorr.GetNbinsY(),b=hFRFcorr.GetNbinsX()),
                 textForLines=ptBinRanges, transparentLegend=False,
                 onlyLineColor=True, noErrorRatioDen=True, useLineFirstHistogram=True, setOnlyLineRatio=True, lineWidth=1,
                 colorVec=colorVec, drawLineTopPanel=1.0)

    if hFRFcorrTestCap is not None:
        drawCorrelationPlot(hFRFcorrTestCap,
                            xAxisName,
                            yAxisName,
                            f"QCD template correction::{corrRange}",
                            hFRFcorrTestCap.GetName(), plotLabel="ForceTitle", outdir=outfolder,
                            draw_both0_noLog1_onlyLog2=1, nContours=args.nContours, palette=args.palette,
                            invertePalette=args.invertePalette, passCanvas=canvas, skipLumi=True)
    # also plot average vs pt and eta
    # currently doesn't plot as I expect
    # hFRF_profileX = hFRFcorr.ProfileX(hFRFcorr.GetName()+"_profileX",
    #                                                            1, hFRFcorr.GetNbinsY())
    # hFRF_profileY = hFRFcorr.ProfileY(hFRFcorr.GetName()+"_profileY",
    #                                                            1, hFRFcorr.GetNbinsX())
    # drawTH1(hFRF_profileX, xAxisName, f"QCD template correction", hFRF_profileX.GetName(), outfolder, drawStatBox=False,
    #         plotTitleLatex=f"Charge {charge}")
    # drawTH1(hFRF_profileY, yAxisName, f"QCD template correction", hFRF_profileY.GetName(), outfolder, drawStatBox=False,
    #         plotTitleLatex=f"Charge {charge}")
    if hFRFcorr.GetNbinsX() == 1 or hFRFcorr.GetNbinsY() == 1:
        if hFRFcorr.GetNbinsY() == 1:
            hFRF_proj = hFRFcorr.ProjectionX(hFRFcorr.GetName()+"_projX", 1, 1, "e")
            axisName1D = xAxisName
        else:
            hFRF_proj = hFRFcorr.ProjectionY(hFRFcorr.GetName()+"_projY", 1, 1, "e")
            axisName1D = yAxisName
        drawTH1(hFRF_proj, axisName1D, f"QCD template correction::{corrRange}", hFRF_proj.GetName(), outfolder, drawStatBox=False,
                plotTitleLatex=f"Charge {charge}")

    # plot 1D eta and pt in the signal region with the correction on the fakes, all from the nominal histogram
    for xbin in axisVar.keys():
        # only eta and pt, a bit hardcoded ...
        if xbin > 1:
            continue
        plotProjection1Dfrom3D(rootHists_nominal, datasetsNoQCDFakes, outfolder_dataMC, canvas1Dshapes=canvas1Dshapes, chargeBin=chargeBin,
                               projectAxisToKeep=xbin, xAxisName=axisVar[xbin][1],
                               plotName=f"{axisVar[xbin][0]}_passIso_jetInclusive_passMt_noFakes",
                               correctionFakes=None)
        plotProjection1Dfrom3D(rootHists_nominal, datasets_nominal, outfolder_dataMC, canvas1Dshapes=canvas1Dshapes, chargeBin=chargeBin,
                               projectAxisToKeep=xbin, xAxisName=axisVar[xbin][1],
                               plotName=f"{axisVar[xbin][0]}_passIso_jetInclusive_passMt_FakesNoCorr",
                               correctionFakes=None)
        plotProjection1Dfrom3D(rootHists_nominal, datasets_nominal, outfolder_dataMC,
                               canvas1Dshapes=canvas1Dshapes, chargeBin=chargeBin,
                               projectAxisToKeep=xbin, xAxisName=axisVar[xbin][1],
                               plotName=f"{axisVar[xbin][0]}_passIso_jetInclusive_passMt_correctFakes",
                               correctionFakes=hFRFcorr)

    outFile = outfolder + rootfilename
    fout = safeOpenFile(outFile, mode="RECREATE")
    hFRFcorr.Write()
    if hFRF_proj:
        hFRF_proj.Write()
    if hFRFcorrUnc:
        for k in hFRFcorrUnc:
            hFRFcorrUnc[k].Write()
        # fill the template
        # first create the TH2 layer with more bins
        etaPtTemplate = ROOT.wrem.projectTH2FromTH3(etaPtChargeTemplate, f"FRF_corrUnc_quadSum_{charge}", 1, 2)
        etaPtTemplate.SetTitle("")
        ROOT.wrem.initializeRootHistogram(etaPtTemplate, 1.0)
        # hFRFcorrUnc contains corr+unc, have to get unc only
        hFRFcorrUnc_tmp = copy.deepcopy(hFRFcorrUnc["quadsum_Up"].Clone(f"hFRFcorrUnc_tmp_{charge}"))
        hFRFcorrUnc_tmp.Add(hFRFcorr, -1.0)
        scaleTH2byOtherTH2(etaPtTemplate, hFRFcorrUnc_tmp, scaleUncertainty=False)
        drawCorrelationPlot(etaPtTemplate,
                            xAxisName,
                            yAxisName,
                            f"FRF correction uncertainty (quadrature sum)",
                            etaPtTemplate.GetName(), plotLabel="ForceTitle", outdir=outfolder,
                            draw_both0_noLog1_onlyLog2=1, nContours=args.nContours, palette=args.palette,
                            invertePalette=args.invertePalette, passCanvas=canvas, skipLumi=True)
        ROOT.wrem.broadCastTH2intoTH3(etaPtChargeTemplate, etaPtTemplate, chargeBin, chargeBin, True)
        etaPtChargeTemplate.Write()

    print()
    print(f"Saving FRF correction vs eta-pt in file\n{outFile}\nfor charge {charge}")
    print()
    fout.Close()

    print("-"*30)
    print("### Bad fit status")
    print("### | status | number of bad bins|")
    for key in sorted(badFitsID.keys()):
        print(f"{key}  {badFitsID[key]}")
    print("-"*30)
    print("### Bad covariance matrix status")
    print("### | status | number of bad bins|")
    for key in sorted(badCovMatrixID.keys()):
        print(f"{key}  {badCovMatrixID[key]}")
    print("-"*30)

######


def runStudyVsDphi(charge, outfolder, args):

    etaLabel = "#eta" if not args.absEta else "|#eta|"
    
    xAxisName = args.xAxisName
    yAxisName = args.yAxisName
    zAxisName = args.zAxisName

    createPlotDirAndCopyPhp(outfolder)

    adjustSettings_CMS_lumi()    
    canvas = ROOT.TCanvas("canvas","",800,800)
    canvas1D = ROOT.TCanvas("canvas1D","",800,700)
    canvas1Dshapes = ROOT.TCanvas("canvas1Dshapes","",700,800)

    canvas_unroll = ROOT.TCanvas("canvas_unroll","",3000,800)
    leftMargin = 0.06
    rightMargin = 0.01
    bottomMargin = 0.12
    canvas_unroll.SetTickx(1)
    canvas_unroll.SetTicky(1)
    canvas_unroll.cd()
    canvas_unroll.SetLeftMargin(leftMargin)
    canvas_unroll.SetRightMargin(rightMargin)
    canvas_unroll.cd()
    canvas_unroll.SetBottomMargin(bottomMargin)

    groups = make_datagroups_2016(args.inputfile[0], applySelection=False)
    datasets = groups.getNames() # this has all the original defined groups
    datasetsForStudy = ["QCD", "Fake"]
    inputHistName = "mTStudyForFakes"
    groups.setNominalName(inputHistName)
    groups.loadHistsForDatagroups(inputHistName, syst="", procsToRead=datasetsForStudy, applySelection=False)
    histInfo = groups.getDatagroups() # keys are same as returned by groups.getNames() 
    for d in datasetsForStudy:
        hnarf = histInfo[d].hists[inputHistName]
        # rebin eta-pt for actual test with fakes, while collapsing other axes for the simple plots as needed,
        # this is only to avoid that THn are too big
        s = hist.tag.Slicer()
        hnarf = hnarf[{"eta" : s[::hist.rebin(args.rebinEta)]}]
        hnarf = hnarf[{"pt" : s[::hist.rebin(args.rebinPt)]}]
        # note, with python/boost/hist slicing the second edge is excluded from projections,
        # also when it is used as actual bin id, so s[1:1:hist.sum] doesn't work, while s[1:2:hist.sum] picks bin 1
        hnarf = hnarf[{"pt": s[0:complex(0,args.maxPt)]}]
        if args.jetCut:
            hnarf = hnarf[{"hasJets": True}]
        else:
            hnarf = hnarf[{"hasJets": s[::hist.sum]}]
        # rebin a bit more in eta-pt
        hnarf = hnarf[{"eta" : s[::hist.rebin(2)]}]
        hnarf = hnarf[{"pt" : s[::hist.rebin(2)]}]
        lowMtUpperBound = int(args.mtNominalRange.split(",")[1])
        hnarf = hnarf[{"mt" : s[:complex(0,lowMtUpperBound):hist.sum]}]
        chargeIndex = 0 if charge == "minus" else 1
        hnarf = hnarf[{"charge" : s[chargeIndex:chargeIndex+1:hist.sum]}]
        # make all TH of FRF in bins of dphi
        histo_fakes_dphiBins = narf.hist_to_root(hnarf) # this is a THnD with eta-pt-passIso-DphiMuonMet
        histo_FRFvsDphi = {}
        nBinsDphi = histo_fakes_dphiBins.GetAxis(3).GetNbins()
        for idp in range(nBinsDphi):
            idpBin = idp +1
            histo_fakes_dphiBins.GetAxis(3).SetRange(idpBin, idpBin)
            histo_fakes_dphiBins.GetAxis(2).SetRange(1, 1)
            histo_fakes_dphiBins_failIso = histo_fakes_dphiBins.Projection(1, 0, "E")
            histo_fakes_dphiBins_failIso.SetName(f"histo_fakes_dphiBins_failIso_idp{idp}")
            histo_fakes_dphiBins.GetAxis(3).SetRange(idpBin, idpBin)
            histo_fakes_dphiBins.GetAxis(2).SetRange(2, 2)
            histo_fakes_dphiBins_passIso = histo_fakes_dphiBins.Projection(1, 0, "E")
            histo_fakes_dphiBins_passIso.SetName(f"histo_fakes_dphiBins_passIso_idp{idp}")
            histo_FRFvsDphi[idp] = copy.deepcopy(histo_fakes_dphiBins_passIso.Clone(f"histo_FRFvsDphi_idp{idp}"))
            histo_FRFvsDphi[idp].Divide(histo_fakes_dphiBins_failIso)
        ##
        histo_fakes_dphiBins.GetAxis(3).SetRange(1, nBinsDphi)
        histo_fakes_dphiBins.GetAxis(2).SetRange(1, 1)
        histo_fakes_dphiBins_failIso = histo_fakes_dphiBins.Projection(1, 0, "E")
        histo_fakes_dphiBins_failIso.SetName("histo_fakes_dphiBins_failIso_idpInclusive")
        histo_fakes_dphiBins.GetAxis(3).SetRange(1, nBinsDphi)
        histo_fakes_dphiBins.GetAxis(2).SetRange(2, 2)
        histo_fakes_dphiBins_passIso = histo_fakes_dphiBins.Projection(1, 0, "E")
        histo_fakes_dphiBins_passIso.SetName("histo_fakes_dphiBins_passIso_idpInclusive")
        histo_FRFvsDphi["inclusive"] = copy.deepcopy(histo_fakes_dphiBins_passIso.Clone("histo_FRFvsDphi_idpInclusive"))
        histo_FRFvsDphi["inclusive"].Divide(histo_fakes_dphiBins_failIso)
        ##
        # select a few eta-pt bins with index numbers (after rebinning)
        etapt_bins = [(1, 2), (2, 2), (3, 2)]
        hists = {b : ROOT.TH1D(f"FRFvsDphi_{d}_ieta{b[0]}_ipt{b[1]}", "", nBinsDphi, 0, np.pi) for b in etapt_bins}
        for b in hists.keys():
            for i in range(hists[b].GetNbinsX()):
                hists[b].SetBinContent(i+1, histo_FRFvsDphi[i].GetBinContent(b[0], b[1]))
                hists[b].SetBinError(i+1, histo_FRFvsDphi[i].GetBinError(b[0], b[1]))

        htmp = histo_FRFvsDphi["inclusive"]
        hs = []
        legEntries = []
        for b in etapt_bins:
            hs.append(hists[b])
            etaText = f"{htmp.GetXaxis().GetBinLowEdge(b[0]):.1f} < #eta < {htmp.GetXaxis().GetBinLowEdge(b[0]+1):.1f}"
            ptText = f"{htmp.GetYaxis().GetBinLowEdge(b[1]):.0f} < p_{{T}} < {htmp.GetYaxis().GetBinLowEdge(b[1]+1):.0f}"
            legEntries.append(f"{etaText} && {ptText} GeV")
            
        drawNTH1(hs, legEntries, "#Delta#phi(#mu,MET)", "Fakerate factor",
                 f"FRFvsDeltaPhi_{d}_someBins_{charge}", outfolder,
                 leftMargin=0.14, rightMargin=0.04, topMargin=0.16, lowerPanelHeight=0.0,
                 legendCoords="0.14,0.96,0.84,0.99;1", skipLumi=True, passCanvas=canvas,
                 transparentLegend=False, drawErrorAll=True,
                 onlyLineColor=True, noErrorRatioDen=False, useLineFirstHistogram=True, setOnlyLineRatio=True, lineWidth=2)
                
    ##


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("inputfile", type=str, nargs=1)
    parser.add_argument("outputfolder",   type=str, nargs=1)
    parser.add_argument("-x", "--xAxisName", default="Muon #eta", help="x axis name")
    parser.add_argument("-y", "--yAxisName", default="Muon p_{T} (GeV)", help="y axis name")
    parser.add_argument("-z", "--zAxisName", default="m_{T} (GeV)", help="z axis name")
    parser.add_argument("--met", default="deepMET", choices=["deepMET","PFMET"], help="Met type, which also updates the mT definition when used as axis labels")
    parser.add_argument("-c", "--charge", default="both", choices=["plus", "minus", "both"], help="charge")
    parser.add_argument("--mtBinEdges", default="0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75", type=str, help="Comma-separated list of bin edges for mT")
    parser.add_argument("--mtNominalRange", default="0,40", type=str, help="Comma-separated list of 2 bin edges for mT, representing the nominal range, used to derive the correction using also option --mt-value-correction")
    parser.add_argument("--mtFitRange", default="0,40", type=str, help="Comma-separated list of 2 bin edges for mT, representing the fit range, might be the same as --mtNominalRange but not necessarily")
    parser.add_argument("--fitPolDegree", default=1, type=int, help="Degree for polynomial used in the fits")
    parser.add_argument("--maxPt", default=50, type=float, help="Do study up to this pt value (-1 means full range)")
    parser.add_argument("--integralMtMethod", default="sideband", choices=["sideband", "fullRange"], type=str, help="How to integrate mT distribution to derive the FRF correction (default is 'sideband', orthogonal to the nominal fake region, 'fullRange' uses the full range)")
    parser.add_argument("--useBinnedCorrection", action="store_true",   help="Use binned FRF to derive the correction (mainly for tests, deprecated option)")
    parser.add_argument("--jetCut", action="store_true",   help="Use jet cut to derive the FRF (sample will be more QCD enriched but might bias the FRF)")
    parser.add_argument("--rebinx", dest="rebinEta", default=1, type=int, help="To rebin x axis (eta)")
    parser.add_argument("--rebiny", dest="rebinPt", default=1, type=int, help="To rebin y axis (pt)")
    parser.add_argument("--nContours", dest="nContours",    default=51, type=int, help="Number of contours in palette. Default is 51 (let it be odd, so the central strip is white if the range is symmetric)")
    parser.add_argument("--palette"  , dest="palette",      default=87, type=int, help="Set palette: default is a built-in one, 55 is kRainbow")
    parser.add_argument("--invertPalette", dest="invertePalette" , action="store_true",   help="Inverte color ordering in palette")
    parser.add_argument("--absEta", dest="absEta" , action="store_true",   help="Do study using histograms folded into absolute value of pseudorapidity")
    parser.add_argument("--allPlot2D", dest="skipPlot2D" , action="store_false",   help="Do some 2D plots with FRF vs eta-Mt, pt-mT, and so on")
    parser.add_argument("--postfix", default="", type=str, help="Postfix for folder name")
    parser.add_argument("-v", "--verbose", type=int, default=3, choices=[0,1,2,3,4], help="Set verbosity level with logging, the larger the more verbose");
    parser.add_argument("--useQCDMC", action="store_true",   help="Make study using QCD MC instead of data-driven fakes, as a cross check")
    parser.add_argument("--dphiMuonMetCut", type=float, help="Threshold to cut |deltaPhi| > thr*np.pi between muon and met", default=-1.0)
    parser.add_argument("--dphiStudy", action="store_true",   help="Only do some plots for dphi study skipping the rest")
    args = parser.parse_args()

    logger = logging.setup_logger(os.path.basename(__file__), args.verbose)
    if args.absEta:
        logger.error("Option --absolute-eta not implemented correctly yet. Abort")
        quit()

    if args.charge == "both":
        charges = ["plus", "minus"]
    else:
        charges = [args.charge]

    ROOT.TH1.SetDefaultSumw2()

    subFolder = f"/{args.integralMtMethod}Integral_fit{args.mtFitRange.replace(',','to')}_pol{args.fitPolDegree}"
    if args.rebinEta > 1:
        subFolder += f"_rebinEta{args.rebinEta}"
    if args.rebinPt > 1:
        subFolder += f"_rebinPt{args.rebinPt}"
    if args.jetCut:
        subFolder += "_1orMoreJetFRF"
    else:
        subFolder += "_jetInclusiveFRF"
    if args.maxPt > 0:
        subFolder += f"_maxPt{int(args.maxPt)}"
    if args.useQCDMC:
        subFolder += "_QCDMCtruth"
    if args.postfix:
        subFolder += f"_{args.postfix}"
    subFolder += "/"
    mainOutputFolder = args.outputfolder[0] + subFolder

    fname = "fakerateFactorMtBasedCorrection_vsEtaPt.root"

    filesToMerge = []
    for charge in charges:
        outputFolder = f"{mainOutputFolder}/{charge}/"
        fnameCharge = fname.replace(".root", f"_{charge}.root")
        if args.dphiStudy:
            runStudyVsDphi(charge, outputFolder, args)
        else:
            runStudy(charge, outputFolder, fnameCharge, args)
        filesToMerge.append(outputFolder+fnameCharge)
        
    if len(charges) == 2 and not args.dphiStudy:
        outFile = mainOutputFolder + fname
        mergeCmd = f"hadd -f {outFile} {' '.join(filesToMerge)}"
        safeSystem(mergeCmd)
        print()
        print(f"Saving all FRF corrections vs eta-pt in file {outFile}")
        print()
