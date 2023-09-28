#!/usr/bin/env python3

import os, re, array, math
import argparse
from copy import *

import itertools
import numpy as np
import tensorflow as tf
import hist
import boost_histogram as bh
import narf
import narf.fitutils
import pickle
import lz4.frame
import time
from functools import partial
from scipy.interpolate import RegularGridInterpolator
from utilities import boostHistHelpers as hh, common, output_tools, logging

## safe batch mode
import sys
args = sys.argv[:]
sys.argv = ['-b']
import ROOT
sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

data_dir = common.data_dir

from scripts.analysisTools.plotUtils.utility import *

import wremnants
logger = logging.setup_logger(__file__, 3, False)

def polN_2d(xvals, parms,
            xLowVal = 0.0, xFitRange = 1.0, degreeX=2,
            yLowVal = 0.0, yFitRange = 1.0, degreeY=3):

    parms2d = tf.reshape(parms, [degreeY+1, degreeX+1])
    
    xscaled = (xvals[0] - xLowVal) / xFitRange
    yscaled = (xvals[1] - yLowVal) / yFitRange
    ret = 0.0
    for dx in range(1+degreeX):
        powx = xscaled**dx if dx else 1.0
        for dy in range(1+degreeY):
            powy = yscaled**dy if dy else 1.0
            ret += parms2d[dy][dx] * powx * powy
    return ret


def runSmoothing(inputfile, histname, outdir, step, args, effHist=None):

    outdirNew = outdir
    addStringToEnd(outdirNew, "/", notAddIfEndswithMatch=True)
    outdirNew += os.path.basename(inputfile).replace(".root","")
    outdirNew += "/"
    createPlotDirAndCopyPhp(outdirNew)
    if args.debugPlots:
        outdir_debug = outdirNew + f"/debugPlots/"
        createPlotDirAndCopyPhp(outdir_debug)

    logger.info(inputfile)
    logger.info(histname)
    logger.info(step)

    adjustSettings_CMS_lumi()
    
    leftMargin = 0.15
    rightMargin = 0.04
    bottomMargin = 0.12
    canvas = ROOT.TCanvas(f"canvas","",800,700)
    canvas.SetTickx(1)
    canvas.SetTicky(1)
    canvas.cd()
    canvas.SetLeftMargin(leftMargin)
    canvas.SetBottomMargin(bottomMargin)
    canvas.SetRightMargin(rightMargin)
    canvas.cd()                                   
        
    dtype = tf.float64
    sfhistname = histname #"SF3D_nominal_isolation" # uT - eta - pt
    tfile = safeOpenFile(inputfile)
    hsf =   safeGetObject(tfile, sfhistname)
    uT_binOffset = 1 # 1 to exclude first bin, use 0 to use all
    uT_binOffset_high = uT_binOffset
    if args.utHigh is not None:
        lastUtBinToFit = hsf.GetXaxis().FindFixBin(args.utHigh+0.001) - 1 # -1 because if we choose uT = 70 then the last bin is the one with 70 as upper edge
        uT_binOffset_high = hsf.GetNbinsX() - lastUtBinToFit
    utEdges  = [round(hsf.GetXaxis().GetBinLowEdge(i), 1) for i in range(1+uT_binOffset, 2+hsf.GetNbinsX()-uT_binOffset_high)] # remove extreme bins for now
    etaEdges = [round(hsf.GetYaxis().GetBinLowEdge(i), 1) for i in range(1, 2+hsf.GetNbinsY())]
    ptEdges  = [round(hsf.GetZaxis().GetBinLowEdge(i), 1) for i in range(1, 2+hsf.GetNbinsZ())]
    if uT_binOffset == 0:
        # redefine last uT bins with a sensible range (twice the width of the second to last bin)
        stretch = 2.0
        utEdges[0] = utEdges[1] - stretch * (utEdges[2] - utEdges[1])
        utEdges[-1] = utEdges[-2] - stretch * (utEdges[-3] - utEdges[-2])
        name = hsf.GetName()
        hsf.SetName(f"{name}_AAA")
        hsfTmp = ROOT.TH3D(name, hsf.GetTitle(),
                           len(utEdges)-1, array('d', utEdges),
                           len(etaEdges)-1, array('d', etaEdges),
                           len(ptEdges)-1, array('d', ptEdges))
        ROOT.wrem.fillTH3fromTH3part(hsfTmp, hsf) # fill new histogram with original values, they have same number of bins
        hsf = hsfTmp # replace the input histogram, to use it later
        logger.warning(f"Setting first and last uT edges to {utEdges[0]} and {utEdges[-1]}")
        
    hsf.GetXaxis().SetRange(1+uT_binOffset, hsf.GetNbinsX()-uT_binOffset_high) # remove extreme bins for now, they extend up to infinity
    
    # 
    polnx = args.polDegree[0]
    polny = args.polDegree[1]

    # for consistency with how tensorflow reads the histogram to fit, it is better to use bin centers to define the range.
    # It is then fine even if the fit function extends outside this range
    ptLow = hsf.GetZaxis().GetBinCenter(1)
    ptHigh = hsf.GetZaxis().GetBinCenter(hsf.GetNbinsZ())
    ptRange = ptHigh - ptLow
    utLow = hsf.GetXaxis().GetBinCenter(1+uT_binOffset) # remove first bin, whose range is too extreme
    utHigh = hsf.GetXaxis().GetBinCenter(hsf.GetNbinsX()-uT_binOffset_high) # remove last bin, whose range is too extreme    
    utRange = utHigh - utLow  
    polN_2d_scaled = partial(polN_2d,
                             xLowVal=utLow, xFitRange=utRange, degreeX=polnx,
                             yLowVal=ptLow, yFitRange=ptRange, degreeY=polny)
    ## save actual histogram edges
    ptEdgeLow = hsf.GetZaxis().GetBinLowEdge(1)
    ptEdgeHigh = hsf.GetZaxis().GetBinLowEdge(1 + hsf.GetNbinsZ())
    utEdgeLow = hsf.GetXaxis().GetBinLowEdge(1+uT_binOffset) # remove first bin, whose range is too extreme
    utEdgeHigh = hsf.GetXaxis().GetBinLowEdge(hsf.GetNbinsX()+1-uT_binOffset_high) # remove last bin, whose range is too extreme    
    # set number of bins for ut and pt after smoothing
    utBinWidth = 2
    utNbins = int((utEdgeHigh - utEdgeLow + 0.001) / utBinWidth) # multiple of 1 GeV width
    ptNbins = 5 * int(ptEdgeHigh - ptEdgeLow + 0.001) # 0.2 GeV width

    nEtaBins = hsf.GetNbinsY()
    etaBinsToRun = args.eta if len(args.eta) else range(1, 1 + nEtaBins)
    
    # to store the pull versus eta-pt for bins of uT, for some plots
    hpull_utEtaPt = ROOT.TH3D("hpull_utEtaPt", "",
                              len(utEdges)-1, array('d', utEdges),
                              len(etaEdges)-1, array('d', etaEdges),
                              len(ptEdges)-1, array('d', ptEdges))

    # extend uT range for plotting purpose, even if eventually the final histogram will be stored in a narrower range
    extendedRange_ut = [min(-50.0, utEdgeLow), max(50.0, utEdgeHigh)]
    extendedRange_ut_nBins = int((extendedRange_ut[1] - extendedRange_ut[0] + 0.001) / utBinWidth)
    # create final boost histogram with smooth SF, eta-pt-ut-ivar
    axis_eta = hist.axis.Regular(nEtaBins, etaEdges[0], etaEdges[-1], name = "eta", overflow = False, underflow = False)
    axis_pt  = hist.axis.Regular(ptNbins,  ptEdgeLow,   ptEdgeHigh,   name = "pt",  overflow = False, underflow = False)
    axis_ut  = hist.axis.Regular(utNbins,  utEdgeLow,   utEdgeHigh,   name = "ut",  overflow = False, underflow = False)
    axis_ut_eff  = hist.axis.Regular(extendedRange_ut_nBins, extendedRange_ut[0], extendedRange_ut[1], name = "ut",  overflow = False, underflow = False)
    axis_var = hist.axis.Integer(0, 1+(1+polnx)*(1+polny), underflow = False, overflow =False, name = "nomi-eigenVars")

    histSF3D_withStatVars = hist.Hist(axis_eta, axis_pt, axis_ut, axis_var,
                                      name = f"smoothSF3D_{step}",
                                      storage = hist.storage.Weight())
    histEffi3D = hist.Hist(axis_eta, axis_pt, axis_ut_eff,
                           name = f"smoothEffi3D_{step}",
                           storage = hist.storage.Weight())
    histEffi2D_ptut = hist.Hist(axis_ut_eff, axis_pt,
                                name = f"smoothEffi2D_{step}_ptut",
                                storage = hist.storage.Weight())

    if effHist != None:
        logger.info("Preparing efficiencies")
        eff_boost = effHist
        effHistRoot = narf.hist_to_root(eff_boost)
        effHistRoot.SetName(f"Wmunu_MC_effi_{step}")
        # make some plots of efficiencies versus eta-ut to see how they behave
        outdirEff = outdirNew + "efficiencies/"
        for ieta in etaBinsToRun:
            etaLow = round(effHistRoot.GetXaxis().GetBinLowEdge(ieta), 1)
            etaHigh = round(effHistRoot.GetXaxis().GetBinLowEdge(ieta+1), 1)
            etaRange = f"{etaLow} < #eta < {etaHigh}"
            etaCenter = effHistRoot.GetXaxis().GetBinCenter(ieta)
            eta_index = eff_boost.axes[0].index(etaCenter)
            #print(f"eta_index = {eta_index}")
            eff_boost_ptut = eff_boost[{0 : eta_index}] # from 3D (eta-pt-ut) to 2D (pt-ut)
            #print(f"eff_boost_ptut.shape = {eff_boost_ptut.shape}") 
            eff_boost_ptut = eff_boost_ptut.project(1, 0) # project second axis (uT) as x and first axis (pT) as y
            #logger.warning(eff_boost_ptut.axes)
            #logger.warning("")
            xvals = [tf.constant(center, dtype=dtype) for center in eff_boost_ptut.axes.centers]
            utvals = np.reshape(xvals[0], [-1])
            ptvals = np.reshape(xvals[1], [-1])
            yvals = eff_boost_ptut.values()
            yvals[np.isnan(yvals)] = 0 # protection against bins where no events were selected (extreme ut for instance), set efficiency to 0 instead of 1
            eff_boost_ptut.values()[...] = yvals
            # plot with root
            if len(args.eta):
                heff = narf.hist_to_root(eff_boost_ptut)
                heff.SetName(f"{effHistRoot.GetName()}_eta{ieta}")
                heff.SetTitle(etaRange)
                drawCorrelationPlot(heff, "Projected recoil u_{T} (GeV)", "Muon p_{T} (GeV)", f"W MC {step} efficiency::0.5,1",
                                    heff.GetName(), "ForceTitle", outdirEff,
                                    palette=87, passCanvas=canvas)
            # the grid interpolator will be created up to the extreme bin centers, so need bounds_error=False to allow the extrapolation to extend outside until the bin edges
            # and then we can set its extrapolation value to fill_value ('None' uses the extrapolation from the curve inside accpetance)
            interp = RegularGridInterpolator((utvals, ptvals), yvals, method='cubic', bounds_error=False, fill_value=None)
            xvalsFine = [tf.constant(center, dtype=dtype) for center in histEffi2D_ptut.axes.centers]
            utvalsFine = np.reshape(xvalsFine[0], [-1])
            ptvalsFine = np.reshape(xvalsFine[1], [-1])
            points = list(itertools.product(*[utvalsFine,ptvalsFine]))
            #print(f"Have to interpolate {len(points)} points ({len(utvalsFine)}*{len(ptvalsFine)} uT-pT fine bins)")            
            pts = np.array(points)
            #print(pts)
            smoothVals = interp(pts)
            #print(smoothVals)
            histEffi2D_ptut.values()[:] = np.reshape(smoothVals, (axis_ut_eff.size, ptNbins))
            histEffi3D.values()[eta_index, ...] = histEffi2D_ptut.values().T
            # set errors to 0 explicitly, although they should already be 0, we don't use them
            histEffi3D.variances()[...] = np.zeros_like(histEffi3D.variances())
            if len(args.eta):
                heffSmooth = narf.hist_to_root(histEffi2D_ptut)
                heffSmooth.SetName(f"{effHistRoot.GetName()}_eta{ieta}_smooth")
                heffSmooth.SetTitle(etaRange)
                drawCorrelationPlot(heffSmooth, "Projected recoil u_{T} (GeV)", "Muon p_{T} (GeV)", f"Smooth W MC {step} efficiency::0.5,1",
                                    heffSmooth.GetName(), "ForceTitle", outdirEff,
                                    palette=87, passCanvas=canvas)
        logger.info("Done with efficiencies")

    # set initial parameters, starting with constant at unit
    arr = [1.0] + [0.0 for x in range((polnx+1)*(polny+1)-1)]
    ##
    postfix = f"_{args.postfix}" if len(args.postfix) else ""

    hpull1D_uTpT = ROOT.TH1D("hpull1D", "", 20, -5, 5)
    hpullSummary_eta_mean  = ROOT.TH1D(f"hpullSummary_{step}_eta_mean",  f"Pull distribution mean {step}",  nEtaBins, etaEdges[0], etaEdges[-1])
    hpullSummary_eta_sigma = ROOT.TH1D(f"hpullSummary_{step}_eta_sigma", f"Pull distribution width {step}", nEtaBins, etaEdges[0], etaEdges[-1])
    
    for ieta in etaBinsToRun:
    # for ieta in range(1, 2):
        hsf.GetYaxis().SetRange(ieta, ieta)
        h = hsf.Project3D("zxe")
        h.SetName(f"{hsf.GetName()}_eta{ieta}")
        etaLow = round(hsf.GetYaxis().GetBinLowEdge(ieta), 1)
        etaCenter = hsf.GetYaxis().GetBinCenter(ieta)
        etaHigh = round(hsf.GetYaxis().GetBinLowEdge(ieta+1), 1)
        etaRange = f"{etaLow} < #eta < {etaHigh}"
        eta_index = axis_eta.index(etaCenter)
        logger.warning(f"Going to fit eta bin {ieta}, {etaRange}")
        h.SetTitle(etaRange)
        hpull = copy.deepcopy(h.Clone(f"{h.GetName()}_pull2Dfit"))
        hpull.Reset("ICESM")
        hpull1D_uTpT.Reset("ICESM")
        drawCorrelationPlot(h, "Projected recoil u_{T} (GeV)", "Muon p_{T} (GeV)", f"{step} scale factor",
                            h.GetName(), "ForceTitle", outdirNew,
                            palette=87, passCanvas=canvas)
                
        boost_hist = narf.root_to_hist(h)
        params = np.array(arr)
        res_polN_2d = narf.fitutils.fit_hist(boost_hist, polN_2d_scaled, params)
        status = res_polN_2d["status"]
        covstatus = res_polN_2d["covstatus"]
        postfit_params = res_polN_2d['x']
        npar = len(postfit_params)
        fitChi2 = round(res_polN_2d["loss_val"], 1)
        ndof = h.GetNbinsX()*h.GetNbinsY() - npar 
        logger.info(f"postfit_params = {postfit_params}")
        logger.info(f"status/covstatus = {status}/{covstatus}")
        if status != 0 or covstatus != 0:
            logger.error("BAD FIT!!!")
            quit()

        # get chi2    
        chi2text = f"#chi^{{2}} = {fitChi2} / {ndof}"
        chi2prob = ROOT.TMath.Prob(fitChi2, ndof)
        if chi2prob < 0.05:
            perc_chi2prob = 100.0 * chi2prob
            sign = "="
            if perc_chi2prob < 0.1:
                perc_chi2prob = 0.1
                sign = "<"
            chi2text += " (prob {} {}%)".format(sign, round(perc_chi2prob,1))
            
        # do it with boost histograms
        for ix in range(1,1+h.GetNbinsX()):
            bincx = h.GetXaxis().GetBinCenter(ix)
            for iy in range(1,1+h.GetNbinsY()):
                bincy = h.GetYaxis().GetBinCenter(iy)
                fitval = polN_2d_scaled([bincx, bincy], postfit_params)
                pull = (fitval - h.GetBinContent(ix, iy)) / h.GetBinError(ix, iy)
                hpull.SetBinContent(ix, iy, pull)
                hpull_utEtaPt.SetBinContent(ix, ieta, iy, pull)
                hpull1D_uTpT.Fill(pull)

        #
        if args.debugPlots:
            drawTH1(hpull1D_uTpT,
                    "Pulls in u_{T}-p_{T} plane",
                    "Number of events",
                    f"hpull_utpt_ieta{ieta}_1D",
                    outdir_debug,
                    passCanvas=canvas,
                    fitString="gaus;LEMSQ+;;-5;5",
                    plotTitleLatex=etaRange)
                
        hpullSummary_eta_mean.SetBinContent(ieta, hpull1D_uTpT.GetMean())
        hpullSummary_eta_mean.SetBinError(ieta, hpull1D_uTpT.GetMeanError())
        hpullSummary_eta_sigma.SetBinContent(ieta, hpull1D_uTpT.GetStdDev())
        hpullSummary_eta_sigma.SetBinError(ieta, hpull1D_uTpT.GetStdDevError())
        
        drawCorrelationPlot(hpull, "Projected recoil u_{T} (GeV)", "Muon p_{T} (GeV)", "Pull: (fit - meas)/meas_unc::-5,5",
                            hpull.GetName(), "ForceTitle", outdirNew,
                            palette=87, nContours=20, passCanvas=canvas)

        hfit_alt = []
        # diagonalize and get eigenvalues and eigenvectors
        logger.info("Diagonalizing covariance matrix ...")
        e, v = np.linalg.eigh(res_polN_2d["cov"])
        postfit_params_alt = np.array([np.zeros(npar, dtype=np.dtype('d'))] * (npar * 2), dtype=np.dtype('d'))
        for ivar in range(npar):
            shift = np.sqrt(e[ivar]) * v[:, ivar]
            postfit_params_alt[ivar]      = postfit_params + shift
            postfit_params_alt[ivar+npar] = postfit_params - shift

        logger.info(f"Creating smooth histogram ...")
        boost_hist_smooth = hist.Hist(axis_ut, axis_pt,
                                      name = f"htmp_fit2D_ieta{ieta}",
                                      storage = hist.storage.Weight())

        xvals = [tf.constant(center, dtype=dtype) for center in boost_hist_smooth.axes.centers]
        boost_hist_smooth.values()[...] = polN_2d_scaled(xvals, postfit_params)
        # copy into final histograms with 3D smoothed SF
        # first swap pt and ut axes from uT-pT to pT-uT by projecting onto itself with reshuffled axes
        #axes = [axis_pt.name, axis_ut.name]
        #hswap = boost_hist_smooth.project(*axes)
        histSF3D_withStatVars.values()[eta_index, :, :, 0] = boost_hist_smooth.values().T # hswap.values()[:,:]
        # convert to root for plotting
        hfit = narf.hist_to_root(boost_hist_smooth)
        hfit.SetName(f"smoothSF2D_{step}_ieta{ieta}{postfix}")
        hfit.SetTitle(f"{etaRange}   {chi2text}")
        drawCorrelationPlot(hfit, "Projected recoil u_{T} (GeV)", "Muon p_{T} (GeV)", f"Smooth {step} scale factor",
                            hfit.GetName(), "ForceTitle", outdirNew,
                            palette=87, passCanvas=canvas)

        for ivar in range(npar):
            boost_hist_smooth.values()[...] = polN_2d_scaled(xvals, postfit_params_alt[ivar])
            histSF3D_withStatVars.values()[eta_index, :, :, ivar+1] = boost_hist_smooth.values().T
            hfit_alt.append(narf.hist_to_root(boost_hist_smooth))
            hfit_alt[ivar].SetName(f"smoothSF2D_{step}_ieta{ieta}_eigen{ivar}{postfix}")
            hfit_alt[ivar].SetTitle(f"{etaRange}: eigen {ivar}")            

        if args.plotEigenVar:
            outdir_eigen = outdirNew + f"eigenDecomposition/eta_{ieta}/"
            createPlotDirAndCopyPhp(outdir_eigen)
            for ivar in range(npar):
                hratio = copy.deepcopy(hfit_alt[ivar].Clone(f"ratioSF_{hfit_alt[ivar].GetName()}"))
                hratio.Divide(hfit)
                drawCorrelationPlot(hratio, "Projected recoil u_{T} (GeV)", "Muon p_{T} (GeV)",
                                    f"Alternate / nominal {step} SF ratio",
                                    hratio.GetName(), "ForceTitle", outdir_eigen,
                                    palette=87, passCanvas=canvas)

        logger.info("-"*30)
            
    if not len(args.eta):
    outdir_pull = outdirNew + f"/pulls_etapt_utBins/"
    createPlotDirAndCopyPhp(outdir_pull)
    for iut in range(1, 1 + hpull_utEtaPt.GetNbinsX()):
        hpull_utEtaPt.GetXaxis().SetRange(iut, iut)
        hpulletapt = hpull_utEtaPt.Project3D("zye")
        hpulletapt.SetName(f"hpull_etapt_ut{iut}")
        utBinLow = hpull_utEtaPt.GetXaxis().GetBinLowEdge(iut)
        utBinHigh = hpull_utEtaPt.GetXaxis().GetBinLowEdge(iut+1)
        hpulletapt.SetTitle(f"{utBinLow} < u_{{T}} < {utBinHigh}")
        drawCorrelationPlot(hpulletapt, "Muon #eta", "Muon p_{T} (GeV)", "Pull: (fit - meas)/meas_unc::-5,5",
                            hpulletapt.GetName(), "ForceTitle", outdir_pull,
                            palette=87, nContours=20, passCanvas=canvas)
    drawSingleTH1(hpullSummary_eta_mean, "Muon #eta", "Mean of pulls in u_{T}-p_{T} plane::-1, 1",
                  hpullSummary_eta_mean.GetName(), outdir_pull, legendCoords=None,
                  lowerPanelHeight=0.0, drawLineTopPanel=0.0,
                  passCanvas=canvas, skipLumi=True)
    drawSingleTH1(hpullSummary_eta_sigma, "Muon #eta", "Width of pulls in u_{T}-p_{T} plane::0, 2",
                  hpullSummary_eta_sigma.GetName(), outdir_pull, legendCoords=None,
                  lowerPanelHeight=0.0, drawLineTopPanel=1.0,
                  passCanvas=canvas, skipLumi=True)

    # should the following be done for each eta bin as above? At least one could plot things vs pt-ut more easily
    if effHist != None and step in ["iso", "triggerplus", "triggerminus"]:
        # compute antiiso_SF = (1-SF*effMC)/ (1-effMC)
        # first make uT axis consistent
        s = bh.tag.Slicer()
        histEffi3D_asSF = histEffi3D[{"ut" : s[complex(0,axis_ut.edges[0]):complex(0,axis_ut.edges[-1])]}]
        #logger.warning("Resizing ut axis for efficiency to match SF histogram")
        #logger.warning(f"{histEffi3D_asSF.axes}")
        # remember that histSF3D_withStatVars has 4 axes, 4th is the stat variation
        num = histSF3D_withStatVars.copy()
        den = histSF3D_withStatVars.copy()
        num.values()[...] = np.ones_like(histSF3D_withStatVars.values()) # num = 1
        den = num.copy() # now den is filled with all 1
        num = hh.addHists(num, hh.multiplyHists(histSF3D_withStatVars, histEffi3D_asSF, createNew=True), createNew=False, scale2=-1.0) # 1 - SF*effMC
        den = hh.addHists(den, histEffi3D_asSF, scale2=-1.0, createNew=False) # 1 - effMC
        antiSF = hh.divideHists(num, den, createNew=True)
        antiSF.name = f"smoothSF3D_anti{step}"
        # make sure the numerator is not negative, i.e. SF*effMC < 1 or equivalently 1 - SF*effMC > 0
        denVals = den.values()
        numVals = num.values()
        # note, efficiencies are already capped to be in [0,1], but could also be exactly 0 or exactly 1
        # set antiSF=1 if eff == 1 (i.e. den=0, or num < 0, or (num < 0.5% and eff > 99.5% (i.e. den < 0.5%))
        antiSF_values = antiSF.values()
        cellsToChange = (denVals <= 0.0) | (numVals < 0) | ((numVals < 0.005) & (denVals < 0.005))
        nChangedCells = np.count_nonzero(cellsToChange)
        if nChangedCells > 0:
            frac = nChangedCells/np.product(cellsToChange.shape)
            logger.warning(f"Setting antiSF = 1.0 in {nChangedCells}/{np.product(cellsToChange.shape)} cells for step {step} ({frac:.1%})")
            antiSF_values[cellsToChange] = 1.0
            antiSF.values()[...] = antiSF_values
        #
    else:
        antiSF = None
        
    return [histSF3D_withStatVars, histEffi3D if effHist != None else None, antiSF]


if __name__ == "__main__":

    sfFolder = data_dir + "/testMuonSF/"
    # efficiencies made with scripts/analysisTools/w_mass_13TeV/makeWMCefficiency3D.py
    #
    #effSmoothFile = "/eos/user/m/mciprian/www/WMassAnalysis/test2Dsmoothing/makeWMCefficiency3D/noMuonCorr_noSF_allProc_noDphiCut/efficiencies3D.pkl.lz4"
    effSmoothFile = f"{sfFolder}efficiencies3D.pkl.lz4"
    #
    inputRootFile = {"iso"          : f"{sfFolder}isolation3DSFUT.root",
                     "isonotrig"    : f"{sfFolder}isonotrigger3DSFVQT.root",
                     "isoantitrig"  : f"{sfFolder}isofailtrigger3DSFVQT.root",
                     "triggerplus"  : f"{sfFolder}triggerplus3DSFUT.root",
                     "triggerminus" : f"{sfFolder}triggerminus3DSFUT.root",
                     }
    
    parser = argparse.ArgumentParser()
    #parser.add_argument('inputfile',  type=str, nargs=1, help='input root file with histogram')
    #parser.add_argument('histname',  type=str, nargs=1, help='Histogram name to read from the file')
    parser.add_argument('outdir', type=str, nargs=1, help='output directory to save things')
    #parser.add_argument('step', type=str, nargs=1, help='Step, also to name output histogram (should be parsed from input file though)')
    parser.add_argument('-n', '--outfilename', type=str, default='smoothSF3D.pkl.lz4', help='Output file name, extension must be pkl.lz4, which is automatically added if no extension is given')
    parser.add_argument('--eta', type=int, nargs="*", default=[], help='Select some eta bins (ID goes from 1 to Neta')
    parser.add_argument('--polDegree', type=int, nargs=2, default=[2, 3], help='Select degree of polynomial for 2D smoothing (uT-pT)')
    parser.add_argument('--plotEigenVar', action="store_true", help='Plot eigen variations (it actually produces histogram ratios alt/nomi)')
    parser.add_argument('-p', '--postfix', type=str, default="", help='Postfix for plot names (can be the step name)')
    parser.add_argument('-s', '--step', type=str, nargs="*", default=[], choices=list(inputRootFile.keys()), help='Do only these steps (default uses all)')
    parser.add_argument('--extended', action="store_true", help='Use SF with uT range extended above +30 GeV')
    parser.add_argument('--utHigh', type=float, default=None, help='Choose maximum uT at which the fit must be run (default uses full range except very last bin which is up to infinity)')
    parser.add_argument('--debugPlots', action="store_true", help='Run additional plots for debugging (might become default eventually)')
    args = parser.parse_args()

    ROOT.TH1.SetDefaultSumw2()

    if not args.outfilename.endswith(".pkl.lz4"):
        if "." in args.outfilename:
            logger.error(f"Invalid extension for output file name {args.outfilename}. It must be 'pkl.lz4'")
            quit()
        else:
            logger.info(f"Adding pkl.lz4 extension for output file name {args.outfilename}")
            args.outfilename += ".pkl.lz4"

    if args.extended:
        #effSmoothFile = "/eos/user/m/mciprian/www/WMassAnalysis/test2Dsmoothing/makeWMCefficiency3D/noMuonCorr_noSF_allProc_noDphiCut_rebinUt2/efficiencies3D_rebinUt2.pkl.lz4"
        effSmoothFile = f"{sfFolder}efficiencies3D_rebinUt2.pkl.lz4"
        #
        args.outfilename = args.outfilename.replace(".pkl.lz4", "_extended.pkl.lz4")
        inputRootFile = {"iso"          : f"{sfFolder}iso3DSFVQTextended.root",
                         "isonotrig"    : f"{sfFolder}isonotrigger3DSFVQTextended.root",
                         "isoantitrig"  : f"{sfFolder}isofailtrigger3DSFVQTextended.root",
                         "triggerplus"  : f"{sfFolder}triggerplus3DSFVQTextended.root",
                         "triggerminus" : f"{sfFolder}triggerminus3DSFVQTextended.root",
                         }
        
    with lz4.frame.open(effSmoothFile) as fileEff:
        allMCeff = pickle.load(fileEff)

    effHist = {}
    for step in ["iso", "isonotrig", "isoantitrig", "triggerplus", "triggerminus"]:
        effHist[step] = allMCeff[f"Wmunu_MC_eff_{step}_etaptut"]    
        
    work = []
    work.append([inputRootFile["iso"],          "SF3D_nominal_iso" if args.extended else "SF3D_nominal_isolation",     "iso", effHist["iso"]])
    work.append([inputRootFile["isonotrig"],    "SF3D_nominal_isonotrigger",  "isonotrig", None]) # , effHist["isonotrig"]])
    work.append([inputRootFile["isoantitrig"],  "SF3D_nominal_isofailtrigger",  "isoantitrig", None]) # , effHist["isoantitrig"]])
    work.append([inputRootFile["triggerplus"],  "SF3D_nominal_trigger_plus",  "triggerplus", effHist["triggerplus"]])
    work.append([inputRootFile["triggerminus"], "SF3D_nominal_trigger_minus", "triggerminus", effHist["triggerminus"]])

    outdir = args.outdir[0]
    
    resultDict = {}
    for w in work:
        inputfile, histname, step, eff = w
        if len(args.step) and step not in args.step:
            continue
        rets = runSmoothing(inputfile, histname, outdir, step, args, effHist=eff)
        for ret in rets:
            if ret != None:
                resultDict[ret.name] = ret
                
    resultDict.update({"meta_info" : output_tools.metaInfoDict(args=args)})
    
    outfile = outdir + args.outfilename
    logger.info(f"Going to store histograms in file {outfile}")
    logger.info(f"All keys: {resultDict.keys()}")
    time0 = time.time()
    with lz4.frame.open(outfile, 'wb') as f:
        pickle.dump(resultDict, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Output saved: {time.time()-time0}")
