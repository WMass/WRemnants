#!/usr/bin/env python3

# mainly for tracking efficiencies and failing probes
# example:
# python w-mass-13TeV/compareTnpMass.py /home/m/mciprian/tnp/Steve_Marc_Raj/outputs/test_trackerMuons/tnp_tracking_data_vertexWeights1_oscharge0.root /home/m/mciprian/tnp/Steve_Marc_Raj/outputs/test_trackerMuons/tnp_tracking_mc_vertexWeights1_oscharge0.root plots/TnP/Steve_Marc_Raj/testTrackerMuons/tracking/ --zbin 1 3

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

import copy

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
    parser.add_argument("inputfileData", type=str, nargs=1, help="Input file for data")
    parser.add_argument("inputfileMC", type=str, nargs=1, help="Input file for MC")
    parser.add_argument("outputfolder", type=str, nargs=1)
    parser.add_argument(
        "-x",
        "--xAxisName",
        dest="xAxisName",
        default="Invariant mass (GeV) ",
        help="x axis name",
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
        "--showAllProbesMC",
        action="store_true",
        help="Show sum of failing and passing probes for MC (when not using --plotPassProbes, and it only works for steps where standalone muons were used)",
    )
    parser.add_argument(
        "--plotPassProbes",
        action="store_true",
        help="Plot passing probes instead of failing probes",
    )
    parser.add_argument(
        "--plotPassAltProbes",
        action="store_true",
        help="Plot passing probes instead of failing probes",
    )
    parser.add_argument(
        "--skipNorm",
        dest="normalize",
        action="store_false",
        help="Normalize to area of first histogram",
    )
    args = parser.parse_args()
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    ROOT.TH1.SetDefaultSumw2()

    if args.showAllProbesMC:
        if args.plotPassAltProbes or args.plotPassProbes:
            logger.error(
                "Can't use --plotPassProbes or --plotPassAltProbes with --showAllProbesMC"
            )
            quit()

    outdir_original = args.outputfolder[0]
    outdir = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)

    probeType = (
        "passAlt"
        if args.plotPassAltProbes
        else "pass" if args.plotPassProbes else "fail"
    )
    probeTypeHist = (
        "pass" if (args.plotPassAltProbes or args.plotPassProbes) else "fail"
    )
    passAltNamePostfix = "_alt" if probeType == "passAlt" else ""

    f = safeOpenFile(args.inputfileData[0])
    hdata3D = safeGetObject(f, f"{probeTypeHist}_mu_RunGtoH{passAltNamePostfix}")
    f.Close()

    f = safeOpenFile(args.inputfileMC[0])
    hmc3D = safeGetObject(f, f"{probeTypeHist}_mu_DY_postVFP{passAltNamePostfix}")
    hmcTot3D = copy.deepcopy(hmc3D.Clone("all_mu_DY_postVFP"))
    if args.showAllProbesMC:
        try:
            hmcalt3D = safeGetObject(f, "pass_mu_DY_postVFP_alt")
        except:
            hmcalt3D = safeGetObject(f, "pass_mu_DY_postVFP")
        hmcTot3D.Add(hmcalt3D)
    else:
        hmcalt3D = None
    f.Close()

    hists = [hmcTot3D, hmc3D, hdata3D]
    for h in hists:
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

    logger.info(f"{hmcTot3D.GetNbinsZ()} eta bins")
    logger.info(f"{hmcTot3D.GetNbinsY()} pt  bins")

    iymin = 1
    iymax = hmcTot3D.GetNbinsY()
    if args.ybin[0] > 0 and args.ybin[1] > 0:
        iymin, iymax = args.ybin

    izmin = 1
    izmax = hmcTot3D.GetNbinsZ()
    if args.zbin[0] > 0 and args.zbin[1] > 0:
        izmin, izmax = args.zbin

    for ieta in range(1, 1 + hmcTot3D.GetNbinsZ()):
        if not (izmin <= ieta <= izmax):
            continue
        for ipt in range(1, 1 + hmcTot3D.GetNbinsY()):
            if not (iymin <= ipt <= iymax):
                continue
            hmcTot = hmcTot3D.ProjectionX("hmcTot", ipt, ipt, ieta, ieta, "e")
            hmc = hmc3D.ProjectionX("hmc", ipt, ipt, ieta, ieta, "e")
            hdata = hdata3D.ProjectionX("hdata", ipt, ipt, ieta, ieta, "e")

            hmcTot.SetMarkerSize(0)
            # hmcTot.SetFillColor(ROOT.kGreen+2)
            hmcTot.SetFillColorAlpha(ROOT.kGreen + 2, 0.5)
            hmcTot.SetFillStyle(1001)

            hmc.SetLineColor(ROOT.kBlue)
            hmc.SetLineWidth(2)

            hdata.SetMarkerStyle(20)
            hdata.SetMarkerSize(1)

            if args.normalize:
                hmcTotScale = (
                    hdata.Integral() / hmcTot.Integral()
                    if hmcTot.Integral() > 0.0
                    else 1.0
                )
                hmcTot.Scale(hmcTotScale)
                hmcScale = (
                    hdata.Integral() / hmc.Integral() if hmc.Integral() > 0.0 else 1.0
                )
                hmc.Scale(hmcScale)

            miny, maxy = getMinMaxMultiHisto(
                [hdata, hmc, hmcTot] if args.showAllProbesMC else [hdata, hmc],
                excludeEmpty=False,
                sumError=False,
            )

            hframe = hmcTot if args.showAllProbesMC else hmc
            hframe.SetStats(0)
            hframe.SetMarkerSize(0)
            hframe.SetMarkerStyle(0)
            hframe.GetXaxis().SetTitle(args.xAxisName)
            hframe.GetXaxis().SetTitleOffset(1.3)
            hframe.GetXaxis().SetTitleSize(0.05)
            hframe.GetXaxis().SetLabelSize(0.04)
            hframe.GetYaxis().SetTitle("Data events / bin")
            hframe.GetYaxis().SetTitleOffset(1.25)
            hframe.GetYaxis().SetTitleSize(0.05)
            hframe.GetYaxis().SetLabelSize(0.04)
            hframe.GetYaxis().SetRangeUser(0, 1.25 * maxy)

            if args.showAllProbesMC:
                hmcTot.Draw("HE")
                hmc.Draw("HIST SAME")
            else:
                hmc.Draw("HIST")
            hdata.Draw("EP SAME")

            header = "{} < #eta < {}".format(
                round(hmcTot3D.GetZaxis().GetBinLowEdge(ieta), 1),
                round(hmcTot3D.GetZaxis().GetBinUpEdge(ieta), 1),
            )
            header += "   ---   "
            header += "{} < p_{{T}} < {} GeV".format(
                round(hmcTot3D.GetYaxis().GetBinLowEdge(ipt), 0),
                round(hmcTot3D.GetYaxis().GetBinUpEdge(ipt), 0),
            )

            leg = ROOT.TLegend(0.2, 0.78 if args.showAllProbesMC else 0.82, 0.9, 0.9)
            leg.SetNColumns(3)
            leg.SetFillColor(0)
            leg.SetFillStyle(0)
            leg.SetFillColorAlpha(0, 0.6)
            leg.SetBorderSize(0)
            leg.SetHeader(header)
            leg.AddEntry(hdata, f"Data ({probeType} probes)", "EP")
            leg.AddEntry(hmc, f"MC ({probeType} probes)", "L")
            if args.showAllProbesMC:
                leg.AddEntry(hmcTot, "MC (all probes)", "LF")
            leg.Draw("same")

            canvasName = f"{probeType}ProbeMass_ieta_{ieta}_ipt_{ipt}"

            canvas.RedrawAxis("sameaxis")
            for ext in ["png", "pdf"]:
                canvas.SaveAs(f"{outdir}/{canvasName}.{ext}")

    copyOutputToEos(outdir, outdir_original, eoscp=args.eoscp)
