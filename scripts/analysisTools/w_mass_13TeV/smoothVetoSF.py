#!/usr/bin/env python3

# given the smoothed pieces, construct the overall veto SF and the veto antiSF
# currently: reco*tracking*idip
# uncertainties on idip could be neglected

import os
import pickle
import sys
import time

import hist
import lz4.frame
import numpy as np
import ROOT
import tensorflow as tf
import utilitiesCMG
from scipy.interpolate import RegularGridInterpolator

import narf
import narf.fitutils
import wums.ioutils
from utilities import common
from wums import boostHistHelpers as hh

utilities = utilitiesCMG.util()

## safe batch mode
args = sys.argv[:]
sys.argv = ["-b"]

sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

data_dir = common.data_dir

from scripts.analysisTools.plotUtils.utility import (
    addStringToEnd,
    adjustSettings_CMS_lumi,
    common_plot_parser,
    copyOutputToEos,
    createPlotDirAndCopyPhp,
    drawCorrelationPlot,
    getMinMaxMultiHisto,
    logging,
    prepareLegend,
    safeGetObject,
    safeOpenFile,
    setTDRStyle,
)
from scripts.analysisTools.w_mass_13TeV.run2Dsmoothing import makeAntiSFfromSFandEffi


def getHistWithStatUncBand(hsf):
    # hsf has axis "nomi-statUpDown-syst"
    hnomi = hsf[{"nomi-statUpDown-syst": s[0]}]
    hstatVar = hsf.copy()
    # remove first bin for nominal and last bin for syst
    systbin = hsf.axes["nomi-statUpDown-syst"].size - 1
    hstatVar = hstatVar[{"nomi-statUpDown-syst": s[1:systbin]}]
    nBins = hstatVar.axes["nomi-statUpDown-syst"].size
    logger.debug(f"hstatVar now has {nBins} bins on the nomi-statUpDown-syst axis")
    # get half of the bins, for Up var (the others are the down var, but they are symmetric)
    hstatVar = hstatVar[{"nomi-statUpDown-syst": s[0 : int(nBins / 2)]}]
    nBins = hstatVar.axes["nomi-statUpDown-syst"].size
    logger.debug(f"hstatVar now has {nBins} stat vars on the nomi-statUpDown-syst axis")
    henvStatSquare = hh.rssHists(
        hstatVar, "nomi-statUpDown-syst", hnom=hnomi, returnDiffSquare=True
    )

    hnomiWithBand = hnomi.copy()
    hnomiWithBand.variances(flow=False)[...] = henvStatSquare.values(flow=False)[...]
    return hnomiWithBand


def makePlots1D(hnomi, hnomiWithStat, hsyst, outputfolder, args, tag="veto"):

    # prepare canvas for 1D plots
    leftMargin = 0.15
    rightMargin = 0.04
    bottomMargin = 0.12
    canvas1D = ROOT.TCanvas(f"canvas1D_{tag}", "", 800, 800)
    canvas1D.SetTickx(1)
    canvas1D.SetTicky(1)
    canvas1D.cd()
    canvas1D.SetLeftMargin(leftMargin)
    canvas1D.SetBottomMargin(bottomMargin)
    canvas1D.SetRightMargin(rightMargin)
    canvas1D.cd()

    setTDRStyle()

    netaBins = hnomi.axes[0].size
    hpt = {}
    hptSyst = {}
    createPlotDirAndCopyPhp(outputfolder, eoscp=args.eoscp)
    logger.debug(hnomi.axes[0].edges)

    for ieta in range(netaBins):
        histo = narf.hist_to_root(hnomi[{0: s[ieta]}])
        histo.SetName(f"hpt_ieta{ieta}_{charge}")
        etaLow = round(hnomi.axes[0].edges[ieta], 1)
        etaHigh = round(hnomi.axes[0].edges[ieta + 1], 1)
        histo.SetTitle(f"{etaLow} < #eta < {etaHigh}")
        for i, step in enumerate(steps):
            hpt[step] = narf.hist_to_root(hnomiWithStat[step][{0: s[ieta]}])
            hpt[step].SetName(f"hpt_ieta{ieta}_{step}_{charge}")
            hpt[step].SetTitle("")
            hpt[step].SetLineColor(stepColors[i])
            hpt[step].SetFillColor(stepColors[i])
            hpt[step].SetStats(0)
            if "idip" not in step:
                if "reco" in step:
                    hpt[step].SetFillColorAlpha(stepColors[i], 0.9)
                else:
                    hpt[step].SetFillColorAlpha(stepColors[i], 0.8)
            hptSyst[step] = narf.hist_to_root(hsyst[step][{0: s[ieta]}])
            hptSyst[step].SetName(f"hptSyst_ieta{ieta}_{step}_{charge}")
            hptSyst[step].SetTitle("")
            hptSyst[step].SetLineColor(stepColorsSyst[i])
            hptSyst[step].SetLineWidth(2)
            hptSyst[step].SetStats(0)

        histlist = [hpt[st] for st in steps]
        miny, maxy = getMinMaxMultiHisto([histo, *histlist], sumError=True)
        maxy = miny + 1.3 * (maxy - miny)
        # logger.warning(f"miny/maxy = {miny} / {maxy}")
        histo.SetLineColor(ROOT.kBlack)
        histo.SetMarkerColor(ROOT.kBlack)
        histo.SetMarkerStyle(0)
        histo.SetMarkerSize(0)
        histo.GetYaxis().SetTitle(f"{tag.capitalize()} scale factor")
        histo.GetXaxis().SetTitleOffset(1.2)
        histo.GetXaxis().SetTitleSize(0.05)
        histo.GetXaxis().SetLabelSize(0.04)
        histo.GetXaxis().SetTitle("Muon p_{T} (GeV)")
        histo.GetYaxis().SetRangeUser(miny, maxy)
        histo.GetYaxis().SetTitleOffset(1.45)
        histo.SetStats(0)
        histo.Draw("HIST L")
        leg = prepareLegend(0.15, 0.8, 0.95, 0.9, nColumns=2, fillColorAlpha=0.6)
        leg2 = prepareLegend(0.15, 0.7, 0.95, 0.8, nColumns=2, fillColorAlpha=0.6)
        leg.AddEntry(histo, "Nominal veto", "L")
        for i, step in enumerate(steps):
            hpt[step].Draw("E4SAME")
            leg.AddEntry(hpt[step], f"Stat. unc. {step}", "F")
        histo.Draw("HIST L SAME")
        leg.Draw("SAME")
        canvas1D.RedrawAxis("sameaxis")
        ROOT.gStyle.SetOptTitle(1)
        for ext in ["pdf", "png"]:
            canvas1D.SaveAs(f"{outputfolder}/sf_{tag}_ieta{ieta}_{charge}.{ext}")
        # repeat with syst added
        histlist = [hptSyst[st] for st in steps]
        miny, maxy = getMinMaxMultiHisto([histo, *histlist], sumError=True)
        maxy = miny + 1.3 * (maxy - miny)
        histo.GetYaxis().SetRangeUser(miny, maxy)
        for i, step in enumerate(steps):
            hptSyst[step].Draw("HIST LSAME")
            leg2.AddEntry(hptSyst[step], f"Syst. unc. {step}", "L")
        histo.Draw("HIST L SAME")
        leg2.Draw("SAME")
        canvas1D.RedrawAxis("sameaxis")
        for ext in ["pdf", "png"]:
            canvas1D.SaveAs(
                f"{outputfolder}/sf_{tag}_ieta{ieta}_{charge}_withSyst.{ext}"
            )


if __name__ == "__main__":

    parser = common_plot_parser()

    parser.add_argument(
        "-i",
        "--indir",
        default=None,
        help="Directory containing the smoothed scale factors",
    )
    parser.add_argument(
        "outdir", type=str, nargs=1, help="output directory to save things"
    )
    parser.add_argument(
        "--vetoType",
        type=str,
        default="global",
        choices=["global", "globalOrTracker"],
        help="Type of veto SF to use",
    )
    parser.add_argument(
        "--charge",
        type=str,
        default="plus",
        choices=["plus", "minus", "both"],
        help="Charge for veto SF",
    )
    parser.add_argument(
        "--era",
        type=str,
        default="2016PostVFP",
        choices=["2016PostVFP", "2017", "2018"],
    )

    args = parser.parse_args()
    logger = logging.setup_logger(os.path.basename(__file__), args.verbose, True)

    ROOT.TH1.SetDefaultSumw2()
    canvas = ROOT.TCanvas("canvas", "", 800, 700)
    adjustSettings_CMS_lumi()

    charge = args.charge
    vetoType = args.vetoType

    if args.indir is None:
        if args.era == "2016PostVFP":
            inputfolder = f"{data_dir}/muonSF/veto_{vetoType}_SF/"
        else:
            inputfolder = f"{data_dir}/muonSF/{args.era}/veto_{vetoType}_SF/"
    else:
        inputfolder = args.indir

    outdir_original = f"{args.outdir[0]}/{vetoType}_{charge}/"
    addStringToEnd(outdir_original, "/", notAddIfEndswithMatch=True)
    outdir = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)

    steps = ["vetoreco", "vetotracking", "vetoidip"]

    s = hist.tag.Slicer()
    nomiHists, hists = {}, {}
    for step in steps:
        f = f"{inputfolder}/smoothedSFandEffi_{step}_GtoH_{charge}.root"
        tfile = safeOpenFile(f)
        hists[step] = narf.root_to_hist(
            safeGetObject(tfile, f"SF_nomiAndAlt_GtoH_{step}_{charge}"),
            axis_names=["SF eta", "SF pt", "nomi-statUpDown-syst"],
        )
        nomiHists[step] = hists[step][{"nomi-statUpDown-syst": s[0]}]
        tfile.Close()

    # get product and compute all variations
    # will also make envelope of all effStat to compare the steps above, if one is much smaller it can be neglected
    vetoprodSF = {}
    for step in steps:
        logger.debug(f"Doing step {step}")
        vetoprodSF[step] = hists[step].copy()
        otherSteps = [x for x in steps if x != step]
        for x in otherSteps:
            logger.debug(f"Multiplying histogram for step {step} by step {x}")
            vetoprodSF[step] = hh.multiplyHists(
                vetoprodSF[step], nomiHists[x], createNew=True
            )

        hnomi_root = narf.hist_to_root(nomiHists[step])
        hnomi_root.SetName(f"nominalSF_{step}_{charge}")
        hnomi_root.SetTitle(f"Nominal {step} {charge}")
        drawCorrelationPlot(
            hnomi_root,
            f"Muon #eta",
            f"Muon p_{{T}} (GeV)",
            "Scale factor",
            hnomi_root.GetName(),
            "ForceTitle",
            outdir,
            palette=args.palette,
            passCanvas=canvas,
            drawOption="COLZ0",
            skipLumi=True,
        )

    nomiprod = vetoprodSF[steps[0]][{"nomi-statUpDown-syst": s[0]}]
    # logger.warning(nomiprod)
    hnomiprod_root = narf.hist_to_root(nomiprod)
    hnomiprod_root.SetName(f"nominalSF_vetoall_{charge}")
    hnomiprod_root.SetTitle(f"Nominal veto {charge}")
    drawCorrelationPlot(
        hnomiprod_root,
        f"Muon #eta",
        f"Muon p_{{T}} (GeV)",
        "Scale factor",
        hnomiprod_root.GetName(),
        "ForceTitle",
        outdir,
        palette=args.palette,
        passCanvas=canvas,
        drawOption="COLZ0",
        skipLumi=True,
    )

    hsyst = {}
    for step in steps:
        systbin = vetoprodSF[step].axes["nomi-statUpDown-syst"].size - 1
        hsyst[step] = vetoprodSF[step][{"nomi-statUpDown-syst": s[systbin]}]
        hratioSyst = hh.divideHists(hsyst[step], nomiprod)
        hratioSyst_root = narf.hist_to_root(hratioSyst)
        hratioSyst_root.SetName(f"ratioSystOverNomi_{step}_{charge}")
        hratioSyst_root.SetTitle(f"Syst/nomi {step} {charge}")
        drawCorrelationPlot(
            hratioSyst_root,
            f"Muon #eta",
            f"Muon p_{{T}} (GeV)",
            "Scale factor ratio",
            hratioSyst_root.GetName(),
            "ForceTitle",
            outdir,
            palette=args.palette,
            passCanvas=canvas,
            drawOption="COLZ0",
            skipLumi=True,
        )

    # now plot stat.unc. band from quadrature sum of all stat variations
    nomiprodWithStatUnc = {}
    for step in steps:
        nomiprodWithStatUnc[step] = getHistWithStatUncBand(vetoprodSF[step])

    # red, blue, orange
    stepColors = [
        ROOT.TColor.GetColor("#e42536"),
        ROOT.TColor.GetColor("#5790fc"),
        ROOT.TColor.GetColor("#f89c20"),
    ]
    stepColorsSyst = [ROOT.kAzure + 2, ROOT.kGreen + 2, ROOT.kGray + 3]

    outdir1D = outdir + "/pt1D/"
    makePlots1D(nomiprod, nomiprodWithStatUnc, hsyst, outdir1D, args, tag="veto")

    # read MC truth efficiency
    effSmoothFile = f"{inputfolder}/vetoEfficienciesEtaPt.pkl.lz4"
    with lz4.frame.open(effSmoothFile) as fileEff:
        allMCeff = pickle.load(fileEff)
        eff_boost = allMCeff[f"Wmunu_MC_eff_veto{charge}_etapt"]
    histEffi2D_etapt_boost = hist.Hist(
        nomiprod.axes[0],
        nomiprod.axes[1],
        name=f"smoothEffi2D_veto_etapt_boost",
        storage=hist.storage.Weight(),
    )
    # smooth efficiency vs pt in each eta bin using a spline, then fill the histogram with fine pt binning
    netaBins = nomiprod.axes[0].size
    for ieta in range(netaBins):
        etaLow = round(nomiprod.axes[0].edges[ieta], 1)
        etaHigh = round(nomiprod.axes[0].edges[ieta + 1], 1)
        etaRange = f"{etaLow} < #eta < {etaHigh}"
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
            0  # protection against bins where no events were selected (but should not have happened), set efficiency to 0 instead of 1
        )
        # logger.warning(etaRange)
        # logger.warning(f"ptvals = {ptvals}")
        # logger.warning(f"yvals = {yvals}")
        eff_boost_pt.values()[...] = yvals
        # the grid interpolator will be created up to the extreme bin centers, so need bounds_error=False to allow the extrapolation to extend outside until the bin edges
        # and then we can set its extrapolation value to fill_value ('None' uses the extrapolation from the curve inside acceptance)
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
        histEffi2D_etapt_boost.values()[eta_index:] = smoothVals
        # print(smoothVals)
        ## end of loop on eta bins
    histEffi2D_etapt_boost.variances()[...] = np.zeros_like(
        histEffi2D_etapt_boost.variances()
    )
    histEffi2D_etapt_root = narf.hist_to_root(histEffi2D_etapt_boost)
    histEffi2D_etapt_root.SetName(f"efficiency_MCtruth_vetoall_{charge}")
    histEffi2D_etapt_root.SetTitle(f"Veto {charge}")
    # plot W MC efficiencies after spline interpolation as a check
    drawCorrelationPlot(
        histEffi2D_etapt_root,
        "Muon #eta",
        "Muon p_{T} (GeV)",
        "W MC efficiency (spline interp.)",
        histEffi2D_etapt_root.GetName(),
        "ForceTitle",
        outdir,
        palette=args.palette,
        passCanvas=canvas,
    )
    logger.info("Done with efficiencies")

    antivetoprodSF = {}
    for step in steps:
        eff_broadcast_boost = hh.broadcastSystHist(
            histEffi2D_etapt_boost, vetoprodSF[step]
        )
        # anti SF = (1 - SF*eff)/(1- eff)
        antivetoprodSF[step] = makeAntiSFfromSFandEffi(
            vetoprodSF[step], eff_broadcast_boost, step
        )

    nomiAntiVeto2D = antivetoprodSF[steps[0]][
        {"nomi-statUpDown-syst": 0}
    ]  # nomi is the same for all steps
    nomiAntiVeto2D_root = narf.hist_to_root(nomiAntiVeto2D)
    nomiAntiVeto2D_root.SetName(f"nominalSF_antivetoall_{charge}")
    nomiAntiVeto2D_root.SetTitle(f"Nominal antiveto {charge}")
    drawCorrelationPlot(
        nomiAntiVeto2D_root,
        "Muon #eta",
        "Muon p_{T} (GeV)",
        "Scale factor",
        nomiAntiVeto2D_root.GetName(),
        "ForceTitle",
        outdir,
        palette=args.palette,
        passCanvas=canvas,
    )

    antiWithStatUnc = {}
    antiSyst = {}
    for step in steps:
        systbin = antivetoprodSF[step].axes["nomi-statUpDown-syst"].size - 1
        antiWithStatUnc[step] = getHistWithStatUncBand(antivetoprodSF[step])
        antiSyst[step] = antivetoprodSF[step][{"nomi-statUpDown-syst": systbin}]

    outdir1D_anti = outdir + "/pt1D_anti/"
    makePlots1D(
        nomiAntiVeto2D, antiWithStatUnc, antiSyst, outdir1D_anti, args, tag="antiveto"
    )

    # save in pkl file
    resultDict = {}
    for step in steps:
        resultDict[f"vetoSF_{vetoType}_{step}_{charge}"] = vetoprodSF[step]
        resultDict[f"antiVetoSF_{vetoType}_{step}_{charge}"] = antivetoprodSF[step]
    resultDict.update(
        {"meta_info": wums.ioutils.make_meta_info_dict(args=args, wd=common.base_dir)}
    )

    outfile = f"{outdir}/allVetoSF_{vetoType}_{charge}.pkl.lz4"
    logger.info(f"Going to store histograms in file {outfile}")
    logger.info(f"All keys: {resultDict.keys()}")
    time0 = time.time()
    with lz4.frame.open(outfile, "wb") as f:
        pickle.dump(resultDict, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Output saved: {time.time()-time0}")

    copyOutputToEos(outdir, outdir_original, eoscp=args.eoscp)
