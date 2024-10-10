#!/usr/bin/env python3

# to get TH2 histograms for weights vs vertex z and pileup

import sys
from array import array

args = sys.argv[:]
sys.argv = ["-b"]
import ROOT

sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

# sys.path.append(os.getcwd() + "/plotUtils/")
# from utility import *
from scripts.analysisTools.plotUtils.utility import (adjustSettings_CMS_lumi,
                                                     common_plot_parser,
                                                     copyOutputToEos,
                                                     createPlotDirAndCopyPhp,
                                                     drawCorrelationPlot,
                                                     drawTH1, safeGetObject,
                                                     safeOpenFile)
from utilities import common

ROOT.gStyle.SetOptStat(0)
ROOT.TH1.SetDefaultSumw2()

data_dir = common.data_dir
pileupBins = [
    5 * i for i in range(1, 10)
]  # PU 5 to 45 (0-5 and 45-50 probably had too small statistics)
inclusiveRange = [0, 50]  # there is an additional histogram for PU from 0 to 50
inputfiles = [
    "ratios/ratio_2016postVFP_69200ub.root",
    "ratios/ratio_2016preVFP_69200ub.root",
]
outputfile = data_dir + "/vertex/vertexPileupWeights.root"
plotFolder = "plots/testNanoAOD/testvertexPileupWeights/"

if __name__ == "__main__":
    parser = common_plot_parser()
    args = parser.parse_args()

    adjustSettings_CMS_lumi()
    canvas = ROOT.TCanvas("canvas", "", 800, 800)

    outfile = safeOpenFile(outputfile, mode="RECREATE")
    outdir_original = plotFolder
    outdir = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)

    for f in inputfiles:
        era = "preVFP" if "preVFP" in f else "postVFP" if "postVFP" in f else "full2016"
        outFolder = f"{outdir}/{era}/"
        createPlotDirAndCopyPhp(outFolder)
        infile = safeOpenFile(f)
        # get vertex binning from file (although it should be 55 bins from -15 to 15 cm)
        h4bins = safeGetObject(
            infile, "ratio_%s_%s" % (inclusiveRange[0], inclusiveRange[1])
        )
        nVtx = h4bins.GetNbinsX()
        vtxLow = h4bins.GetXaxis().GetBinLowEdge(1)
        vtxHigh = h4bins.GetXaxis().GetBinLowEdge(1 + nVtx)
        #
        # clone inclusive to make a TH1D
        h1D = ROOT.TH1D(
            f"weight_vertexZ_inclusivePU{inclusiveRange[0]}to{inclusiveRange[1]}_{era}",
            f"{era}: inclusive PU in [{inclusiveRange[0]}, {inclusiveRange[1]}]",
            nVtx,
            vtxLow,
            vtxHigh,
        )
        for ivtx in range(1, 1 + h4bins.GetNbinsX()):
            h1D.SetBinContent(ivtx, h4bins.GetBinContent(ivtx))
        # now build 2D with other histograms
        h2D = ROOT.TH2D(
            f"weight_vertexZ_pileup_{era}",
            era,
            nVtx,
            vtxLow,
            vtxHigh,
            len(pileupBins) - 1,
            array("d", pileupBins),
        )
        for ipu in range(len(pileupBins) - 1):
            hname = "ratio_%s_%s" % (pileupBins[ipu], pileupBins[ipu + 1])
            h = safeGetObject(infile, hname)
            for ivtx in range(1, 1 + h.GetNbinsX()):
                h2D.SetBinContent(ivtx, ipu + 1, h.GetBinContent(ivtx))
        outfile.cd()
        eraId = (
            "GtoH" if era == "postVFP" else "BtoF" if era == "preVFP" else "BtoH"
        )  # to be consistent with keywords in other scale factors when storing histograms in the output file
        h2D.Write(h2D.GetName().replace(era, eraId))
        h1D.Write(h1D.GetName().replace(era, eraId))
        drawCorrelationPlot(
            h2D,
            "Vertex z (cm)",
            "True pileup",
            "Weight",
            f"{h2D.GetName()}_realRange",
            plotLabel="ForceTitle",
            outdir=outFolder,
            draw_both0_noLog1_onlyLog2=1,
            nContours=51,
            palette=87,
            passCanvas=canvas,
            skipLumi=True,
        )
        drawCorrelationPlot(
            h2D,
            "Vertex z (cm)",
            "True pileup",
            "Weight::0.0,5.0",
            h2D.GetName(),
            plotLabel="ForceTitle",
            outdir=outFolder,
            draw_both0_noLog1_onlyLog2=1,
            nContours=51,
            palette=87,
            passCanvas=canvas,
            skipLumi=True,
        )
        drawTH1(
            h1D,
            "Vertex z (cm)",
            "Weight",
            h1D.GetName(),
            outFolder,
            passCanvas=canvas,
            skipTdrStyle=True,
            drawStatBox=False,
        )

        infile.Close()

    outfile.Close()
    copyOutputToEos(outdir, outdir_original, eoscp=args.eoscp)
