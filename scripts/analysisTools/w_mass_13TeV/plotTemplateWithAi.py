#!/usr/bin/env python3

import argparse
import os
import os.path
## safe batch mode
import sys

args = sys.argv[:]
sys.argv = ["-b"]
import ROOT

sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

from copy import *

from scripts.analysisTools.plotUtils.utility import *

sys.path.append(os.getcwd())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "rootfile", type=str, nargs=1, help="Input file with TH2 histograms"
    )
    parser.add_argument("outdir", type=str, nargs=1, help="Output folder")
    parser.add_argument(
        "-c",
        "--charges",
        dest="charges",
        choices=["plus", "minus", "both"],
        default="both",
        type=str,
        help="Charges to process",
    )
    args = parser.parse_args()

    fname = args.rootfile[0]
    outdir = args.outdir[0]
    createPlotDirAndCopyPhp(outdir)

    ROOT.TH1.SetDefaultSumw2()

    adjustSettings_CMS_lumi()
    canvas1D = ROOT.TCanvas("canvas1D", "", 800, 1000)

    canvas_unroll = ROOT.TCanvas("canvas_unroll", "", 3000, 800)
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

    setTDRStyle()  # this one removes the stat box

    charges = ["plus", "minus"] if args.charges == "both" else [args.charges]

    for charge in charges:

        nomihists = {ai: None for ai in range(6)}
        # read histograms
        infile = safeOpenFile(fname)
        for ai in nomihists.keys():
            # nomihists[ai] = safeGetObject(infile, f"Wmunu/nominal_Wmunu_PtVBin0YVBin0AnfCoeff{ai}_{charge}", detach=True) # process name as subfolder
            #
            # how to read input histograms depends on input file, format has changed a lot
            # Wmunu_qGen1_absYVgenSig4_ptVgenSig9_helicity5
            icW = 0 if charge == "minus" else 1
            print(f"Reading Ai {ai-1}")
            for iyW in range(5):
                for iptW in range(10):
                    folder = (
                        f"Wmunu_qGen{icW}_absYVgenSig{iyW}_ptVgenSig{iptW}_helicity{ai}"
                    )
                    hread = f"{folder}/x_{folder}_{charge}"
                    # print(f"Reading {hread}")
                    tmp = safeGetObject(
                        infile, hread, detach=True
                    )  # process name as subfolder
                    if nomihists[ai] == None:
                        nomihists[ai] = copy.deepcopy(tmp)
                    else:
                        nomihists[ai].Add(tmp)
        infile.Close()

        outdir_dataMC = f"{outdir}dataMC_{charge}/"
        createPlotDirAndCopyPhp(outdir_dataMC)

        nomihists_unroll = {}
        for k in nomihists.keys():
            nomihists_unroll[k] = unroll2Dto1D(
                nomihists[k],
                newname=f"unrolled_{nomihists[k].GetName()}",
                cropNegativeBins=False,
            )

        nomihists_projEta = {}
        nomihists_projPt = {}
        for k in nomihists.keys():
            nomihists_projEta[k] = nomihists[k].ProjectionX(
                f"{nomihists[k].GetName()}_eta", 1, nomihists[k].GetNbinsY(), "e"
            )
            nomihists_projPt[k] = nomihists[k].ProjectionY(
                f"{nomihists[k].GetName()}_pt", 1, nomihists[k].GetNbinsX(), "e"
            )

        legEntry = {0: "Unpolarized term #sigma_{UL}"}
        for ai in range(1, 6):
            ai_id = ai - 1
            legEntry[ai] = f"Angular term #propto A{ai_id}"

        ptBinRanges = []
        for ipt in range(0, nomihists[0].GetNbinsY()):
            if ipt % 4:
                ptBinRanges.append("")
            else:
                ptBinRanges.append(
                    "#splitline{{[{ptmin},{ptmax}]}}{{GeV}}".format(
                        ptmin=int(nomihists[0].GetYaxis().GetBinLowEdge(ipt + 1)),
                        ptmax=int(nomihists[0].GetYaxis().GetBinLowEdge(ipt + 2)),
                    )
                )

        drawNTH1(
            [nomihists_unroll[ai] for ai in nomihists_unroll.keys()],
            [legEntry[l] for l in legEntry.keys()],
            "Unrolled eta-p_{T} bin",
            "Events::0,110000",
            f"WmunuCrossSection_byAngoularCoefficients_{charge}",
            outdir_dataMC,
            topMargin=0.2,
            leftMargin=0.06,
            rightMargin=0.01,
            labelRatioTmp="A_{i} / #sigma_{UL}::-0.12,0.12",
            yAxisExtendConstant=1.4,
            ytextOffsetFromTop=0.22,
            legendCoords="0.06,0.99,0.87,0.99;4",
            lowerPanelHeight=0.3,
            skipLumi=True,
            passCanvas=canvas_unroll,
            drawVertLines="{a},{b}".format(
                a=nomihists[0].GetNbinsY(), b=nomihists[0].GetNbinsX()
            ),
            textForLines=ptBinRanges,
            transparentLegend=False,
            onlyLineColor=True,
            noErrorRatioDen=True,
            useLineFirstHistogram=True,
            setOnlyLineRatio=True,
            lineWidth=1,
        )

        drawNTH1(
            [nomihists_projEta[ai] for ai in nomihists_unroll.keys()],
            [legEntry[l] for l in legEntry.keys()],
            "Muon #eta",
            "Events",
            f"WmunuCrossSection_byAngoularCoefficients_{charge}_1Deta",
            outdir_dataMC,
            topMargin=0.25,
            leftMargin=0.16,
            rightMargin=0.05,
            labelRatioTmp="A_{i} / #sigma_{UL}::-0.12,0.12",
            legendCoords="0.01,0.99,0.81,0.99;2",
            lowerPanelHeight=0.3,
            skipLumi=True,
            passCanvas=canvas1D,
            transparentLegend=False,
            onlyLineColor=True,
            noErrorRatioDen=True,
            useLineFirstHistogram=True,
            setOnlyLineRatio=True,
            lineWidth=2,
        )

        drawNTH1(
            [nomihists_projPt[ai] for ai in nomihists_unroll.keys()],
            [legEntry[l] for l in legEntry.keys()],
            "Muon p_{T} (GeV)",
            "Events",
            f"WmunuCrossSection_byAngoularCoefficients_{charge}_1Dpt",
            outdir_dataMC,
            topMargin=0.25,
            leftMargin=0.16,
            rightMargin=0.05,
            labelRatioTmp="A_{i} / #sigma_{UL}::-0.12,0.12",
            legendCoords="0.01,0.99,0.81,0.99;2",
            lowerPanelHeight=0.3,
            skipLumi=True,
            passCanvas=canvas1D,
            transparentLegend=False,
            onlyLineColor=True,
            noErrorRatioDen=True,
            useLineFirstHistogram=True,
            setOnlyLineRatio=True,
            lineWidth=2,
        )
