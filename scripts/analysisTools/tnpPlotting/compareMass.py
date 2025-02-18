#!/usr/bin/env python3

import os

## safe batch mode
import sys

from wums import logging

args = sys.argv[:]
sys.argv = ["-b"]
import ROOT

sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True


# sys.path.append(os.getcwd() + "/plotUtils/")
# from utility import *
from scripts.analysisTools.plotUtils.utility import (
    adjustSettings_CMS_lumi,
    common_plot_parser,
    copyOutputToEos,
    createPlotDirAndCopyPhp,
    getMinMaxMultiHisto,
    safeGetObject,
    safeOpenFile,
)

sys.path.append(os.getcwd())

if __name__ == "__main__":

    parser = common_plot_parser()
    parser.add_argument("outputfolder", type=str, nargs=1)
    parser.add_argument(
        "--file", dest="inputfile", type=str, nargs="+", help="Input files"
    )
    parser.add_argument(
        "--hname", type=str, nargs="+", help="Histogram name to pick from each file"
    )
    parser.add_argument("--legendEntry", type=str, nargs="+", help="Legend entry")
    parser.add_argument(
        "-x", "--xAxisName", default="Invariant mass (GeV) ", help="x axis name"
    )
    parser.add_argument(
        "--rebinx", dest="rebinX", default=1, type=int, help="To rebin x axis (mass)"
    )
    parser.add_argument(
        "--rebiny", dest="rebinY", default=1, type=int, help="To rebin y axis (pt)"
    )
    parser.add_argument(
        "--rebinz", dest="rebinZ", default=1, type=int, help="To rebin z axis (eta)"
    )
    parser.add_argument(
        "--ybin",
        type=int,
        nargs=2,
        default=[0, 0],
        help="Bins for y axis to plot, default is to do all",
    )
    parser.add_argument(
        "--zbin",
        type=int,
        nargs=2,
        default=[0, 0],
        help="Bins for z axis to plot, default is to do all",
    )
    parser.add_argument(
        "--norm",
        dest="normalize",
        action="store_true",
        help="Normalize to area of first histogram",
    )
    parser.add_argument(
        "-p", "--postfix", type=str, default="", help="Postfix for output plots"
    )
    args = parser.parse_args()

    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    ROOT.TH1.SetDefaultSumw2()

    outdir_original = args.outputfolder[0]
    outdir = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)

    if len(args.inputfile) != len(args.hname) or len(args.inputfile) != len(
        args.legendEntry
    ):
        logger.error("Different number of input options for histograms")
        quit()

    hists3D = []
    for i in range(len(args.inputfile)):
        f = safeOpenFile(args.inputfile[i])
        hists3D.append(safeGetObject(f, args.hname[i]))
        f.Close()

    for h in hists3D:
        if args.rebinX > 1:
            h.RebinX(args.rebinX)
        if args.rebinY > 1:
            h.RebinY(args.rebinY)
        if args.rebinZ > 1:
            h.RebinZ(args.rebinZ)

    adjustSettings_CMS_lumi()
    canvas = ROOT.TCanvas("canvas", "", 900, 800)
    canvas.SetTickx(1)
    canvas.SetTicky(1)
    canvas.cd()
    canvas.SetLeftMargin(0.15)
    canvas.SetRightMargin(0.04)
    canvas.cd()

    logger.info(f"{hists3D[0].GetNbinsZ()} eta bins")
    logger.info(f"{hists3D[0].GetNbinsY()} pt  bins")

    iymin = 1
    iymax = hists3D[0].GetNbinsY()
    if args.ybin[0] > 0 and args.ybin[1] > 0:
        iymin, iymax = args.ybin

    izmin = 1
    izmax = hists3D[0].GetNbinsZ()
    if args.zbin[0] > 0 and args.zbin[1] > 0:
        izmin, izmax = args.zbin

    colors = [ROOT.kBlack, ROOT.kBlue, ROOT.kRed + 2, ROOT.kGreen + 2, ROOT.kOrange + 2]

    for ieta in range(1, 1 + hists3D[0].GetNbinsZ()):
        if not (izmin <= ieta <= izmax):
            continue
        for ipt in range(1, 1 + hists3D[0].GetNbinsY()):
            if not (iymin <= ipt <= iymax):
                continue

            hists = []
            for ih, h in enumerate(hists3D):
                hists.append(h.ProjectionX(f"hmass_{ih}", ipt, ipt, ieta, ieta, "e"))
                hists[-1].SetLineColor(colors[ih])
                hists[-1].SetLineWidth(2)
                if ih and args.normalize:
                    hists[-1].Scale(hists[0].Integral() / hists[-1].Integral())

            miny, maxy = getMinMaxMultiHisto(hists, excludeEmpty=False, sumError=False)

            hfirst = hists[0]
            hfirst.SetStats(0)
            hfirst.SetMarkerSize(0)
            hfirst.SetMarkerStyle(0)
            hfirst.GetXaxis().SetTitle(args.xAxisName)
            hfirst.GetXaxis().SetTitleOffset(1.3)
            hfirst.GetXaxis().SetTitleSize(0.05)
            hfirst.GetXaxis().SetLabelSize(0.04)
            hfirst.GetYaxis().SetTitle("Events / bin")
            hfirst.GetYaxis().SetTitleOffset(1.25)
            hfirst.GetYaxis().SetTitleSize(0.05)
            hfirst.GetYaxis().SetLabelSize(0.04)
            hfirst.GetYaxis().SetRangeUser(0, 1.25 * maxy)

            hfirst.Draw("HE")
            for i in range(1, len(hists)):
                hists[i].Draw("HIST SAME")

            header = "{} < #eta < {}".format(
                round(hists3D[0].GetZaxis().GetBinLowEdge(ieta), 1),
                round(hists3D[0].GetZaxis().GetBinUpEdge(ieta), 1),
            )
            header += "   ---   "
            header += "{} < p_{{T}} < {} GeV".format(
                round(hists3D[0].GetYaxis().GetBinLowEdge(ipt), 0),
                round(hists3D[0].GetYaxis().GetBinUpEdge(ipt), 0),
            )

            leg = ROOT.TLegend(0.2, 0.6, 0.9, 0.9)
            leg.SetNColumns(1)
            leg.SetFillColor(0)
            leg.SetFillStyle(0)
            leg.SetFillColorAlpha(0, 0.6)
            leg.SetBorderSize(0)
            leg.SetHeader(header)
            for il, l in enumerate(args.legendEntry):
                legEntry = l
                if il and args.normalize:
                    legEntry += " (norm)"
                leg.AddEntry(hists[il], legEntry, "L")
            leg.Draw("same")

            postfix = args.postfix
            if not postfix.startswith("_"):
                postfix = "_" + postfix
            canvasName = f"compareMass_ieta_{ieta}_ipt_{ipt}{postfix}"

            canvas.RedrawAxis("sameaxis")
            for ext in ["png", "pdf"]:
                canvas.SaveAs(f"{outdir}/{canvasName}.{ext}")

    copyOutputToEos(outdir, outdir_original, eoscp=args.eoscp)
