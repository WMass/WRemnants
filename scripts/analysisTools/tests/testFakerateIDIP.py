#!/usr/bin/env python3

import argparse
import os
## safe batch mode
import sys

import narf
from utilities import common
from wremnants.datasets.datagroups2016 import make_datagroups_2016

args = sys.argv[:]
sys.argv = ["-b"]
import ROOT

sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

from copy import *

from scripts.analysisTools.plotUtils.utility import *

sys.path.append(os.getcwd())
from scripts.analysisTools.tests.testPlots1D import plotDistribution1D

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
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
        default="Muon |dxybs| (cm)",
        help="y axis name (for the final TH2 plot, not the input histogram)",
    )
    parser.add_argument(
        "--nContours",
        dest="nContours",
        default=51,
        type=int,
        help="Number of contours in palette. Default is 51 (let it be odd, so the central strip is white if not using --abs-value and the range is symmetric)",
    )
    parser.add_argument(
        "--palette",
        dest="palette",
        default=55,
        type=int,
        help="Set palette: default is a built-in one, 55 is kRainbow",
    )
    parser.add_argument(
        "--invertPalette",
        dest="invertePalette",
        default=False,
        action="store_true",
        help="Inverte color ordering in palette",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4],
        help="Set verbosity level with logging, the larger the more verbose",
    )
    args = parser.parse_args()

    logger = common.setup_color_logger(os.path.basename(__file__), args.verbose)

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
    inputHistName = "mtIsoDxybsCharge"
    groups.setNominalName(inputHistName)
    groups.loadHistsForDatagroups(
        inputHistName, syst="", procsToRead=datasetsAll, applySelection=False
    )
    histInfo = groups.getDatagroups()
    rootHists = {d: None for d in datasetsAll}

    adjustSettings_CMS_lumi()
    canvas = ROOT.TCanvas("canvas", "", 800, 800)
    canvas1D = ROOT.TCanvas("canvas1D", "", 800, 800)

    for charge in charges:

        outfolder = f"{args.outputfolder[0]}/{charge}/"
        createPlotDirAndCopyPhp(outfolder)
        # bin number from root histogram
        chargeBin = 1 if charge == "minus" else 2

        fakeRateVsMt = {d: None for d in datasetsAll}
        hAbsDxybs1D = {d: None for d in datasetsAll}

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
            # integrate all iso axis
            histo_fakes.GetAxis(1).SetRange(
                1, histo_fakes.GetAxis(1).FindFixBin(0.15 - 0.0001)
            )  # pass isolation

            # now get a TH2
            h2mtAbsDxybs = histo_fakes.Projection(2, 0, "E")
            h2mtAbsDxybs.SetName(f"mtAbsDxybs_{d}")
            h2mtAbsDxybs.SetTitle(f"{d}: jet inclusive")

            hAllAbsDxybs = h2mtAbsDxybs.ProjectionX(
                f"allAbsDxybs_{d}", 1, 1 + h2mtAbsDxybs.GetNbinsY(), "e"
            )
            hPassAbsDxybs = h2mtAbsDxybs.ProjectionX(
                f"passAbsDxybs_{d}",
                1,
                h2mtAbsDxybs.GetYaxis().FindFixBin(0.05 - 0.0001),
                "e",
            )
            fakeRateVsMt[d] = ROOT.TGraphAsymmErrors(hPassAbsDxybs.GetNbinsX())
            fakeRateVsMt[d].SetName(f"fakeRateVsMt_{d}")
            fakeRateVsMt[d].Divide(hPassAbsDxybs, hAllAbsDxybs, "cl=0.683 b(1,1) mode")

            if d != "Fake":
                hAbsDxybs1D[d] = h2mtAbsDxybs.ProjectionY(
                    f"absDxybs_{d}", 1, 1 + h2mtAbsDxybs.GetNbinsX(), "e"
                )

            drawCorrelationPlot(
                h2mtAbsDxybs,
                xAxisName,
                yAxisName,
                "Events",
                h2mtAbsDxybs.GetName(),
                plotLabel="ForceTitle",
                outdir=outfolder,
                draw_both0_noLog1_onlyLog2=1,
                drawProfileX=True,
                nContours=args.nContours,
                palette=args.palette,
                invertePalette=args.invertePalette,
                passCanvas=canvas,
                skipLumi=True,
            )

        grList = [fakeRateVsMt[d] for d in datasets]
        vecMCcolors = [ROOT.kRed + 2, ROOT.kBlack, ROOT.kGreen + 2, ROOT.kOrange + 2]
        vecMarkerStyle = [
            ROOT.kFullCircle,
            ROOT.kFullCircle,
            ROOT.kFullCircle,
            ROOT.kFullCircle,
        ]
        leg = [f"{d}" for d in datasets]
        for il, lentry in enumerate(leg):
            if "QCD" in lentry:
                leg[il] = leg[il].replace("QCD", "QCD (MC)")
            elif "Fake" in lentry:
                leg[il] = leg[il].replace("Fake", "Fake (data)")
        drawGraphCMS(
            grList,
            xAxisName,
            "Efficiency muon |dxybs| < 0.05 cm::0.6,1.1",
            "fakeRateVsMt_all",
            outfolder,
            leg_roc=leg,
            legendCoords="0.14,0.87,0.94,0.99;2",
            vecMCcolors=vecMCcolors,
            vecMarkerStyle=vecMarkerStyle,
            passCanvas=canvas1D,
            graphDrawStyle="EP",
            legEntryStyle="PL",
            skipLumi=True,
            solidLegend=True,
        )

        plotDistribution1D(
            hAbsDxybs1D["Data"],
            {d: hAbsDxybs1D[d] for d in datasetsAllNoFake if d != "Data"},
            datasetsAllNoFake,
            outfolder,
            canvas1Dshapes=canvas1D,
            xAxisName=yAxisName,
            plotName=f"absDxybs",
            draw_both0_noLog1_onlyLog2=0,
            ratioPadYaxisTitle="Data/pred::0.5,1.5",
        )

        for d in datasetsAllNoFake:
            # drawSingleTH1(hAbsDxybs1D[d],
            #               yAxisName, "Events", f"absDxybs_{d}", outfolder,
            #               draw_both0_noLog1_onlyLog2=2,
            #               lowerPanelHeight=0.3, passCanvas=canvas1D, drawLineLowerPanel="",
            # )
            drawTH1(
                hAbsDxybs1D[d],
                yAxisName,
                "Events",
                f"absDxybs_{d}",
                outfolder,
                passCanvas=canvas1D,
                plotTitleLatex=d,
                setLogY=True,
            )
