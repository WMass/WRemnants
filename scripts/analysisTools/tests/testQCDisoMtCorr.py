#!/usr/bin/env python3

import os

## safe batch mode
import sys

import narf
from wremnants.datasets.datagroups2016 import make_datagroups_2016
from wums import logging

args = sys.argv[:]
sys.argv = ["-b"]
import ROOT

sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

import copy

from scripts.analysisTools.plotUtils.utility import (
    adjustSettings_CMS_lumi,
    common_plot_parser,
    copyOutputToEos,
    createPlotDirAndCopyPhp,
    drawCorrelationPlot,
    drawGraphCMS,
)

sys.path.append(os.getcwd())
from scripts.analysisTools.tests.testPlots1D import plotDistribution1D

if __name__ == "__main__":

    parser = common_plot_parser()
    parser.add_argument("inputfile", type=str, nargs=1)
    parser.add_argument("outputfolder", type=str, nargs=1)
    parser.add_argument(
        "-c",
        "--charge",
        default="plus",
        choices=["plus", "minus", "both"],
        help="charge",
    )
    parser.add_argument(
        "-x",
        "--x-axis-name",
        dest="xAxisName",
        default="RawPF m_{T} (GeV)",
        help="x axis name",
    )
    parser.add_argument(
        "-y",
        "--y-axis-name",
        dest="yAxisName",
        default="PFrelIso04",
        help="y axis name",
    )
    args = parser.parse_args()

    logger = logging.setup_logger(os.path.basename(__file__), args.verbose)

    ROOT.TH1.SetDefaultSumw2()

    if args.charge == "both":
        logger.warning(
            "Running both charges together with -c both is currently deprecated"
        )
        logger.warning("There is some unexpected rebinning versus mt to be fixed")
        quit()
        charges = ["plus", "minus"]
    else:
        charges = [args.charge]

    xAxisName = args.xAxisName
    yAxisName = args.yAxisName

    groups = make_datagroups_2016(args.inputfile[0], applySelection=False)
    datasetsAll = groups.getNames()
    datasetsAllNoFake = list(filter(lambda x: x != "Fake", datasetsAll))
    datasets = ["Wmunu", "QCD", "Top", "Fake"]
    datasetsNoFake = list(filter(lambda x: x != "Fake", datasets))
    logger.info(f"Will plot datasets {datasets}")
    inputHistName = "mtIsoJetCharge"
    groups.setNominalName(inputHistName)
    groups.loadHistsForDatagroups(
        inputHistName, syst="", procsToRead=datasetsAll, applySelection=False
    )
    histInfo = groups.getDatagroups()
    rootHists = {d: None for d in datasetsAll}

    adjustSettings_CMS_lumi()
    canvas = ROOT.TCanvas("canvas", "", 800, 800)
    canvas1D = ROOT.TCanvas("canvas1D", "", 800, 800)

    jetLabels = ["jetInclusive", "1orMoreJet"]

    outdir_original = args.outputfolder[0]
    outdir = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)

    for charge in charges:

        outfolder = f"{outdir}/{charge}/"
        createPlotDirAndCopyPhp(outfolder)
        # bin number from root histogram
        chargeBin = 1 if charge == "minus" else 2

        fakeRateVsMt = {
            "jetInclusive": {d: None for d in datasetsAll},
            "1orMoreJet": {d: None for d in datasetsAll},
        }

        hIso1D = {
            "jetInclusive": {d: None for d in datasetsAll},
            "1orMoreJet": {d: None for d in datasetsAll},
        }

        for d in datasetsAll:
            logger.info(f"     Process {d}")
            hnarf = histInfo[d].hists[inputHistName]
            rootHists[d] = narf.hist_to_root(
                hnarf
            )  # this is a THnD with mt-iso-Njet-charge (Njet is a boolean for 0 or >=1)

            # copy such that the original can still be used untouched to plot all processes
            histo_fakes = copy.deepcopy(rootHists[d])
            # set charge from charge axis
            histo_fakes.GetAxis(3).SetRange(chargeBin, chargeBin)

            for jetBin in range(1, 3):
                histo_fakes.GetAxis(2).SetRange(jetBin, 2)  # >= 1 jets
                # now get a TH2
                h2mtIso = histo_fakes.Projection(1, 0, "E")
                jetLabel = "jetInclusive" if jetBin == 1 else "1orMoreJet"
                jetTitle = "jet inclusive" if jetBin == 1 else ">= 1 jet"
                h2mtIso.SetName(f"mtIso_{jetLabel}_{d}")
                h2mtIso.SetTitle(f"{d}: {jetTitle}")

                hAllIso = h2mtIso.ProjectionX(
                    f"allIso_{jetLabel}_{d}", 1, 1 + h2mtIso.GetNbinsY(), "e"
                )
                hPassIso = h2mtIso.ProjectionX(
                    f"passIso_{jetLabel}_{d}",
                    1,
                    h2mtIso.GetYaxis().FindFixBin(0.15 - 0.0001),
                    "e",
                )
                hAllIso.Rebin(4)
                hPassIso.Rebin(4)
                fakeRateVsMt[jetLabel][d] = ROOT.TGraphAsymmErrors(hPassIso.GetNbinsX())
                fakeRateVsMt[jetLabel][d].SetName(f"fakeRateVsMt_{jetLabel}_{d}")
                fakeRateVsMt[jetLabel][d].Divide(
                    hPassIso, hAllIso, "cl=0.683 b(1,1) mode"
                )

                if d != "Fake":
                    hIso1D[jetLabel][d] = h2mtIso.ProjectionY(
                        f"iso_{jetLabel}_{d}", 1, 1 + h2mtIso.GetNbinsX(), "e"
                    )

                drawCorrelationPlot(
                    h2mtIso,
                    xAxisName,
                    yAxisName,
                    "Events",
                    h2mtIso.GetName(),
                    plotLabel="ForceTitle",
                    outdir=outfolder,
                    draw_both0_noLog1_onlyLog2=1,
                    drawProfileX=True,
                    nContours=args.nContours,
                    palette=args.palette,
                    invertPalette=args.invertPalette,
                    passCanvas=canvas,
                    skipLumi=True,
                )

        grList = [
            fakeRateVsMt[jetLabel][d] for d in datasetsNoFake for jetLabel in jetLabels
        ]
        vecMCcolors = [
            ROOT.kRed + 2,
            ROOT.kRed + 2,
            ROOT.kBlack,
            ROOT.kBlack,
            ROOT.kGreen + 2,
            ROOT.kGreen + 2,
        ]
        vecMarkerStyle = [
            ROOT.kFullCircle,
            ROOT.kOpenTriangleUp,
            ROOT.kFullCircle,
            ROOT.kOpenTriangleUp,
            ROOT.kFullCircle,
            ROOT.kOpenTriangleUp,
        ]
        leg = [f"{d} {jetLabel}" for d in datasetsNoFake for jetLabel in jetLabels]
        drawGraphCMS(
            grList,
            xAxisName,
            "Fake rate::0,1.3",
            "fakeRateVsMt",
            outfolder,
            leg_roc=leg,
            legendCoords="0.14,0.80,0.94,0.98;2",
            vecMCcolors=vecMCcolors,
            vecMarkerStyle=vecMarkerStyle,
            passCanvas=canvas1D,
            graphDrawStyle="EP",
            legEntryStyle="PL",
            skipLumi=True,
            solidLegend=True,
        )

        grList = [
            fakeRateVsMt[jetLabel][d] for d in ["QCD", "Fake"] for jetLabel in jetLabels
        ]
        vecMCcolors = [ROOT.kBlack, ROOT.kBlack, ROOT.kOrange + 2, ROOT.kOrange + 2]
        vecMarkerStyle = [
            ROOT.kFullCircle,
            ROOT.kOpenTriangleUp,
            ROOT.kFullCircle,
            ROOT.kOpenTriangleUp,
        ]
        leg = [
            f"{d} {jetLabel}"
            for d in ["QCD (MC)", "Fake (data)"]
            for jetLabel in jetLabels
        ]
        drawGraphCMS(
            grList,
            xAxisName,
            "Fake rate::0,1.3",
            "fakeRateVsMt_fakesFromMCandData",
            outfolder,
            leg_roc=leg,
            legendCoords="0.14,0.80,0.94,0.92;2",
            vecMCcolors=vecMCcolors,
            vecMarkerStyle=vecMarkerStyle,
            passCanvas=canvas1D,
            graphDrawStyle="EP",
            legEntryStyle="PL",
            skipLumi=True,
            solidLegend=True,
        )

        grList = [fakeRateVsMt[jetLabel][d] for d in datasets for jetLabel in jetLabels]
        vecMCcolors = [
            ROOT.kRed + 2,
            ROOT.kRed + 2,
            ROOT.kBlack,
            ROOT.kBlack,
            ROOT.kGreen + 2,
            ROOT.kGreen + 2,
            ROOT.kOrange + 2,
            ROOT.kOrange + 2,
        ]
        vecMarkerStyle = [
            ROOT.kFullCircle,
            ROOT.kOpenTriangleUp,
            ROOT.kFullCircle,
            ROOT.kOpenTriangleUp,
            ROOT.kFullCircle,
            ROOT.kOpenTriangleUp,
            ROOT.kFullCircle,
            ROOT.kOpenTriangleUp,
        ]
        leg = [f"{d} {jetLabel}" for d in datasets for jetLabel in jetLabels]
        for il, lentry in enumerate(leg):
            if "QCD" in lentry:
                leg[il] = leg[il].replace("QCD", "QCD (MC)")
            elif "Fake" in lentry:
                leg[il] = leg[il].replace("Fake", "Fake (data)")
        drawGraphCMS(
            grList,
            xAxisName,
            "Fake rate::0,1.3",
            "fakeRateVsMt_all",
            outfolder,
            leg_roc=leg,
            legendCoords="0.14,0.79,0.94,0.99;2",
            vecMCcolors=vecMCcolors,
            vecMarkerStyle=vecMarkerStyle,
            passCanvas=canvas1D,
            graphDrawStyle="EP",
            legEntryStyle="PL",
            skipLumi=True,
            solidLegend=True,
        )

        for jetLabel in jetLabels:
            plotDistribution1D(
                hIso1D[jetLabel]["Data"],
                {d: hIso1D[jetLabel][d] for d in datasetsAllNoFake if d != "Data"},
                datasetsAllNoFake,
                outfolder,
                canvas1Dshapes=canvas1D,
                xAxisName="PF rel. isolation (#DeltaR = 0.4)",
                plotName=f"isolation_{jetLabel}",
                draw_both0_noLog1_onlyLog2=0,
                ratioPadYaxisTitle="Data/pred::0.7,1.1",
            )

    copyOutputToEos(outdir, outdir_original, eoscp=args.eoscp)
