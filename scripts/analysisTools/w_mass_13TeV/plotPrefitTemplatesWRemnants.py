#!/usr/bin/env python3

# example
# input file is the output of setupCombineWmass.py, with all TH2D inside
#
# python w-mass-13TeV/plotPrefitTemplatesWRemnants.py input.root outdir [--wlike]
#
# add --pt-range-projection to make projections in a restricted pt range (just for tests to check data/MC)

import re
import os, os.path
import argparse
import shutil

## safe batch mode
import sys
args = sys.argv[:]
sys.argv = ['-b']
import ROOT
sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

from copy import *
import wremnants

from scripts.analysisTools.plotUtils.utility import *

sys.path.append(os.getcwd())

def plotPrefitHistograms(hdata2D, hmc2D, outdir_dataMC, xAxisName, yAxisName,
                         lumi=None, ptRangeProjection=(0,-1), chargeLabel="",
                         canvas=None, canvasWide=None, canvas1D=None,
                         colors=None, legEntries=None, isPseudoData=False,
                         ratioRange=None):

    #TODO: make colors and legEntries a single dictionary

    if not canvas: canvas = ROOT.TCanvas("canvas", "", 800, 700)
    if not canvasWide: canvasWide = ROOT.TCanvas("canvasWide","",2400,600)                      
    adjustSettings_CMS_lumi()
    if not canvas1D: canvas1D = ROOT.TCanvas("canvas1D", "", 800, 900)

    if not colors:
        colors = colors_plots_

    if not legEntries:
        legEntries = legEntries_plots_

    createPlotDirAndCopyPhp(outdir_dataMC)

    fShapesName = outdir_dataMC + "plots.root"
    fShapes = safeOpenFile(fShapesName, mode="RECREATE")

    hmc2D = sorted(hmc2D, key= lambda x: x.Integral()) # , reverse=True) 

    stack_eta = ROOT.THStack("stack_eta", "signal and backgrounds")
    stack_pt = ROOT.THStack("stack_pt", "signal and backgrounds")
    stack_unrolled = ROOT.THStack("stack_unrolled", "signal and backgrounds")

    ratio2D = hdata2D.Clone("dataOverMC2D")
    den2D = hdata2D.Clone("sigAndBkg2D")
    den2D.Reset("ICESM")

    hdata2D.SetMarkerColor(ROOT.kBlack)
    hdata2D.SetLineColor(ROOT.kBlack)
    #hdata2D.SetLineWidth(2)
    hdata2D.SetMarkerStyle(20)
    hdata2D.SetMarkerSize(1)
    hdata2D.SetTitle("")
    hdata_unrolled = unroll2Dto1D(hdata2D, newname=f"{hdata2D.GetName()}_unrolled")

    # for projections along eta
    ptRange = ""
    if ptRangeProjection[0] < ptRangeProjection[1]:
        lowPtbin = max(1, hdata2D.GetYaxis().FindFixBin(ptRangeProjection[0]))
        highPtbin = min(hdata2D.GetNbinsY(), hdata2D.GetYaxis().FindFixBin(ptRangeProjection[1])) # hdata2D.GetNbinsY()
        ptRange = "_%gTo%g" % (hdata2D.GetYaxis().GetBinLowEdge(lowPtbin), hdata2D.GetYaxis().GetBinLowEdge(1+highPtbin))
        ptRange = ptRange.replace(".","p")
    else:
        lowPtbin = 1
        highPtbin = hdata2D.GetNbinsY()

    ratioRangeStr = ""
    if ratioRange:
        ratioRangeStr = f"::{ratioRange[0]},{ratioRange[1]}"
    
        
    hdata_eta = hdata2D.ProjectionX("data_eta",lowPtbin,highPtbin,"e")
    hdata_pt  = hdata2D.ProjectionY("data_pt",1,hdata2D.GetNbinsX(),"e")

    legend = ROOT.TLegend(0.2,0.72,0.95,0.92)
    legend.SetFillColor(0)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.SetNColumns(3)

    leg_unrolled = prepareLegend(0.2,0.72,0.95,0.90, textSize=0.045, nColumns=5)

    hdata2D.Write("data2D")
    hdata_eta.Write()
    hdata_pt.Write()
    hdata_unrolled.Write()

    dataTitle = "Pseudodata" if isPseudoData else "Data"
    legend.AddEntry(hdata2D, dataTitle, "EP")
    leg_unrolled.AddEntry(hdata_unrolled, dataTitle, "EP")
    hmc_unrolled = []
    for h in hmc2D:
        h.SetTitle("")
        h.SetFillColor(colors[h.GetName().split("_")[-2]])
        h.SetLineColor(ROOT.kBlack)
        stack_eta.Add(h.ProjectionX(f"{h.GetName()}_eta",lowPtbin,highPtbin,"e"))
        stack_pt.Add( h.ProjectionY(f"{h.GetName()}_pt",0,-1,"e"))
        den2D.Add(h)
        h.Write()
        hmc_unrolled.append(unroll2Dto1D(h, newname=f"{h.GetName()}_unrolled"))
        hmc_unrolled[-1].SetFillColor(colors[h.GetName().split("_")[-2]])
        hmc_unrolled[-1].SetLineColor(ROOT.kBlack)
        hmc_unrolled[-1].SetMarkerSize(0)
        hmc_unrolled[-1].SetMarkerStyle(0)
        hmc_unrolled[-1].Write()
        stack_unrolled.Add(hmc_unrolled[-1])
    for i in sorted(range(len(hmc2D)), reverse=True):
        legend.AddEntry(hmc2D[i], legEntries[hmc2D[i].GetName().split("_")[-2]], "F")
        leg_unrolled.AddEntry(hmc_unrolled[i], legEntries[hmc2D[i].GetName().split("_")[-2]], "F")

    stack_eta.Write()
    stack_pt.Write()
    den2D.Write()

    # some plots to check statistical uncertainties in MC stack
    hMCstat = copy.deepcopy(den2D.Clone("hMCstat"))
    hMCstat.SetTitle("Sum of predicted processes")
    ROOT.wrem.makeHistStatUncertaintyRatio(hMCstat, den2D)
    minyMCSum,maxyMCSum = getMinMaxHisto(hMCstat, sumError=False)
    maxyMCSum = min(1.5,maxyMCSum)
    minyMCSum = max(0.0, minyMCSum)
    drawCorrelationPlot(hMCstat, xAxisName, yAxisName, "#sqrt{#sum w^{2}} / #sqrt{N}" + f"::{minyMCSum},{maxyMCSum}",
                        f"MCstatOverPoissonUncRatio_allProcs_{chargeLabel}", plotLabel="ForceTitle", outdir=outdir_dataMC,
                        palette=57, passCanvas=canvas, drawOption="COLZ0", skipLumi=True, zTitleOffSet=1.3)
    for h in hmc2D:
        if "Wmunu" in h.GetName():
            hMCstat_Wmunu = copy.deepcopy(h.Clone("hMCstat_Wmunu"))
            hMCstat_Wmunu.SetTitle("Wmunu " + chargeLabel)
            ROOT.wrem.makeHistStatUncertaintyRatio(hMCstat_Wmunu, h)
            drawCorrelationPlot(hMCstat_Wmunu, xAxisName, yAxisName, "#sqrt{#sum w^{2}} / #sqrt{N}",
                                f"MCstatOverPoissonUncRatio_Wmunu_{chargeLabel}", plotLabel="ForceTitle", outdir=outdir_dataMC,
                                palette=57, passCanvas=canvas, drawOption="COLZ0", skipLumi=True, zTitleOffSet=1.3)
        elif "Fake" in h.GetName():
            hMCstat_Fake = copy.deepcopy(h.Clone("hMCstat_Fake"))
            hMCstat_Fake.SetTitle("Fake " + chargeLabel)
            ROOT.wrem.makeHistStatUncertaintyRatio(hMCstat_Fake, h)
            minyFake,maxyFake = getMinMaxHisto(hMCstat_Fake, sumError=False)
            maxyFake = min(7.0,maxyFake)
            minyFake = max(0.0, minyFake)
            drawCorrelationPlot(hMCstat_Fake, xAxisName, yAxisName, "#sqrt{#sum w^{2}} / #sqrt{N}" + f"::{minyFake},{maxyFake}",
                                f"MCstatOverPoissonUncRatio_Fake_{chargeLabel}", plotLabel="ForceTitle", outdir=outdir_dataMC,
                                palette=57, passCanvas=canvas, drawOption="COLZ0", skipLumi=True, zTitleOffSet=1.3)
    
    ratio2D.Divide(den2D)
    ratio2D.Write()

    drawTH1dataMCstack(hdata_eta, stack_eta, "Muon #eta", "Events", "muon_eta" + ptRange,
                       outdir_dataMC, legend, ratioPadYaxisNameTmp=f"{dataTitle}/pred{ratioRangeStr}",
                       passCanvas=canvas1D, lumi=lumi,
                       drawLumiLatex=True, xcmsText=0.3, noLegendRatio=True
    )
    drawTH1dataMCstack(hdata_pt, stack_pt, "Muon p_{T} (GeV)", "Events", "muon_pt",
                       outdir_dataMC, legend, ratioPadYaxisNameTmp=f"{dataTitle}/pred{ratioRangeStr}",
                       passCanvas=canvas1D, lumi=lumi,
                       drawLumiLatex=True, xcmsText=0.3, noLegendRatio=True
    )

    ratio2D.SetTitle(f"{dataTitle} / (signal + background)")
    drawCorrelationPlot(ratio2D, xAxisName, yAxisName, f"{dataTitle}/pred",
                        f"muon_eta_pt_dataMCratio", plotLabel="ForceTitle", outdir=outdir_dataMC,
                        palette=57, passCanvas=canvas, drawOption="COLZ0", skipLumi=True)
    drawCorrelationPlot(ratio2D, xAxisName, yAxisName, f"{dataTitle}/pred. statistical uncertainty",
                        f"muon_eta_pt_dataMCratio_absUncertainty", plotLabel="ForceTitle", outdir=outdir_dataMC,
                        palette=57, passCanvas=canvas, drawOption="COLZ0", skipLumi=True, plotError=True)

    #
    etabins = [round(hdata2D.GetXaxis().GetBinLowEdge(i), 1) for i in range(1, 2 + hdata2D.GetNbinsX())]
    ptbins = [round(hdata2D.GetYaxis().GetBinLowEdge(i), 1) for i in range(1,  2 + hdata2D.GetNbinsY())]
    recoBins = templateBinning(etabins, ptbins)
    nRecoBins = recoBins.NTotBins
    #following array is used to call function dressed2DfromFit()
    binning = [recoBins.Neta, recoBins.etaBins, recoBins.Npt, recoBins.ptBins]        
    cnameUnroll = f"muon_etaPtUnrolled"
    XlabelUnroll = "unrolled template along #eta:  #eta #in [%.1f, %.1f]" % (recoBins.etaBins[0], recoBins.etaBins[-1])
    YlabelUnroll = "Events::%.2f,%.2f" % (0, 2.*hdata_unrolled.GetBinContent(hdata_unrolled.GetMaximumBin()))
    # to draw panels in the unrolled plots
    ptBinRanges = []
    for ipt in range(0,recoBins.Npt):
        #ptBinRanges.append("p_{{T}} #in [{ptmin:3g}, {ptmax:.3g}]".format(ptmin=recoBins.ptBins[ipt], ptmax=recoBins.ptBins[ipt+1]))
        ptBinRanges.append("#splitline{{[{ptmin},{ptmax}]}}{{GeV}}".format(ptmin=int(recoBins.ptBins[ipt]), ptmax=int(recoBins.ptBins[ipt+1])))

    # plot unrolled ratio to better see how it looks like
    ratio_unrolled = unroll2Dto1D(ratio2D, newname=f"{ratio2D.GetName()}_unrolled")
    ROOT.wrem.setRootHistogramError(ratio_unrolled, 0.0)
    drawSingleTH1(ratio_unrolled, XlabelUnroll, f"{dataTitle}/pred. ratio", "muon_etaPtUnrolledRatio",
                  outdir_dataMC, drawLineLowerPanel="", lowerPanelHeight=0.0, labelRatioTmp="", 
                  passCanvas=canvasWide,
                  legendCoords="0.15,0.85,0.82,0.9;2",
                  leftMargin=0.05,rightMargin=0.01,lumi=lumi, 
                  drawVertLines="{a},{b}".format(a=recoBins.Npt,b=recoBins.Neta),
                  textForLines=ptBinRanges, ytextOffsetFromTop=0.3, textSize=0.04, textAngle=30, drawLineTopPanel=1.0)

    allHists = hmc2D + [hdata2D]
    hdata2D.SetTitle(f"{dataTitle} {chargeLabel}")
    for h in hmc2D:
        h.SetTitle(legEntries[h.GetName().split("_")[-2]] + " " + chargeLabel)
    for h in allHists:
        drawCorrelationPlot(h, xAxisName, yAxisName, "Events",
                            f"muon_eta_pt_{h.GetName()}", plotLabel="ForceTitle", outdir=outdir_dataMC,
                            palette=57, passCanvas=canvas, drawOption="COLZ0", skipLumi=True)

    drawTH1dataMCstack(hdata_unrolled, stack_unrolled, XlabelUnroll, YlabelUnroll, cnameUnroll,
                       outdir_dataMC, leg_unrolled, ratioPadYaxisNameTmp=f"{dataTitle}/pred{ratioRangeStr}",
                       passCanvas=canvasWide,
                       wideCanvas=True, leftMargin=0.05,rightMargin=0.01,lumi=lumi, 
                       drawVertLines="{a},{b}".format(a=recoBins.Npt,b=recoBins.Neta),
                       textForLines=ptBinRanges, etaptbinning=binning, noLegendRatio=True, textSize=0.04, textAngle=30,
                       #noRatioPanel=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rootfile", type=str, nargs=1, help="Input file with TH2 histograms")
    parser.add_argument("outdir",   type=str, nargs=1, help="Output folder")
    parser.add_argument("-l", "--lumi",     type=str, default=None, help="Luminosity to print on canvas, by default it is not printed")
    parser.add_argument("--pt-range-projection", dest="ptRangeProjection", default=(0,-1), type=float, nargs=2, help="Pt range to select bins to use for 1D projection (for upper range remember that upper bin edge belongs to next bin in ROOT)")
    parser.add_argument("--wlike", dest="isWlike", action="store_true", help="Flag for W-like analysis")
    parser.add_argument("--pd", "--pseudodata", dest="pseudodata", type=str, default=None, help="Name for pseudodata histogram, to be used instead of x_Data (with no charge postfix, it is added in this script)")
    
    commonargs,_ = parser.parse_known_args()
    defaultProcs = ["Zmumu", "Ztautau", "Other"] if commonargs.isWlike else ["Wmunu", "Wtaunu", "Zmumu", "Ztautau", "Fake", "Top", "Diboson"]

    parser.add_argument("--pp", "--predicted-processes", dest="predictedProcesses", type=str, nargs="*", help="Use these names for predicted processes to make plots", default=defaultProcs)
    parser.add_argument("--xpp", "--exclude-predicted-processes", dest="excludePredictedProcesses", type=str, nargs="*", help="Use these names to exclude predicted processes to make plots", default=[])
    parser.add_argument('-c','--charges', dest='charges', choices=['plus', 'minus', 'both'], default='both', type=str, help='Charges to process')
    parser.add_argument("--rr", "--ratio-range", dest="ratioRange", default=(0.92,1.08), type=float, nargs=2, help="Range for ratio plot")
    args = parser.parse_args()
           
    fname = args.rootfile[0]
    outdir = args.outdir[0]
    createPlotDirAndCopyPhp(outdir)
    
    ROOT.TH1.SetDefaultSumw2()

    canvas = ROOT.TCanvas("canvas", "", 800, 700)
    cwide = ROOT.TCanvas("cwide","",2400,600)                      
    adjustSettings_CMS_lumi()    
    canvas1D = ROOT.TCanvas("canvas1D", "", 800, 900)

    processes = [x for x in args.predictedProcesses if x not in args.excludePredictedProcesses]
    if not args.pseudodata:
        processes = ["Data"] + processes
    charges = ["plus", "minus"] if args.charges == "both" else [args.charges]
    
    xAxisName = "Muon #eta"
    yAxisName = "Muon p_{T} (GeV)"

    colors = colors_plots_
    legEntries = legEntries_plots_
    
    for charge in charges:

        # read histograms
        nomihists = {}
        infile = safeOpenFile(fname)
        for proc in processes:
            nomihists[proc] = safeGetObject(infile, f"{proc}/nominal_{proc}_{charge}", detach=True) # process name as subfolder
        if args.pseudodata:
            nomihists["Data"] = safeGetObject(infile, f"{args.pseudodata}_{charge}", detach=True)
        infile.Close()

        hdata2D = nomihists["Data"]
        hmc2D = [nomihists[x] for x in nomihists.keys() if x != "Data"]

        outdir_dataMC = f"{outdir}dataMC_{charge}/"
        createPlotDirAndCopyPhp(outdir_dataMC)

        plotPrefitHistograms(hdata2D, hmc2D, outdir_dataMC, xAxisName=xAxisName, yAxisName=yAxisName,
                             lumi=args.lumi, ptRangeProjection=args.ptRangeProjection, chargeLabel=charge,
                             canvas=canvas, canvasWide=cwide, canvas1D=canvas1D,
                             colors=colors, legEntries=legEntries, isPseudoData=True if args.pseudodata else False,
                             ratioRange=args.ratioRange)

