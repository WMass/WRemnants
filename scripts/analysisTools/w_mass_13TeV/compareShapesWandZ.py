#!/usr/bin/env python3

## safe batch mode
import sys

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
    alt1D,
    colors_plots_,
    common_plot_parser,
    copyOutputToEos,
    createPlotDirAndCopyPhp,
    getMinMaxMultiHisto,
    h1,
    legEntries_plots_,
    safeGetObject,
    safeOpenFile,
    setTDRStyle,
    syst1DDown,
    syst1DUp,
)

if __name__ == "__main__":
    parser = common_plot_parser()
    parser.add_argument("rootfile", type=str, nargs=2, help="Input root files")
    parser.add_argument("outdir", type=str, nargs=1, help="Folder for plots")
    parser.add_argument(
        "-p",
        "--processes",
        type=str,
        nargs="+",
        default=["Wmunu", "Zmumu"],
        help="List of processes to plot (full name please)",
    )
    parser.add_argument(
        "--altLeg",
        type=str,
        default="Alt.",
        help="Legend entry for alternate histogram",
    )
    parser.add_argument(
        "--ratioRange",
        default=None,
        type=float,
        nargs=2,
        help="Range for ratio plot (if None, use default from plotted histograms)",
    )
    parser.add_argument(
        "-c",
        "--charge",
        default=None,
        choices=["plus", "minus"],
        type=str,
        help="Needs to specify the charge, since all histograms are in the same file",
    )
    parser.add_argument(
        "--syst",
        type=str,
        default="massShiftW100MeV",
        help="Name for syst to plot (without Up/Down, they are both taken automatically)",
    )
    args = parser.parse_args()

    fname0 = args.rootfile[0]
    fname1 = args.rootfile[1]
    outdir_original = args.outdir[0] + "/"
    outdir = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)

    ROOT.TGaxis.SetExponentOffset(-0.09, 0.0, "y")
    ROOT.TH1.SetDefaultSumw2()

    if not args.charge:
        print(
            "For histograms from WRemnants the charge must be specified using -c [minus|plus]."
        )
        quit()

    canvas = ROOT.TCanvas("canvas1D", "", 800, 900)
    topMargin = 0.06
    leftMargin = 0.16
    rightMargin = 0.04
    lowerPanelHeight = 0.4
    # bottomMargin = 0.1
    canvas.SetTickx(1)
    canvas.SetTicky(1)
    canvas.cd()
    canvas.SetTopMargin(topMargin)
    canvas.SetLeftMargin(leftMargin)
    canvas.SetRightMargin(rightMargin)
    # canvas.SetBottomMargin(bottomMargin)
    canvas.cd()

    setTDRStyle()  # this one removes the stat box

    pad2 = 0
    canvas.SetBottomMargin(lowerPanelHeight)
    pad2 = ROOT.TPad("pad2", "pad2", 0, 0.0, 1, 0.92)
    pad2.SetTopMargin(1 - lowerPanelHeight)
    pad2.SetRightMargin(rightMargin)
    pad2.SetLeftMargin(leftMargin)
    pad2.SetFillColor(0)
    pad2.SetGridy(1)
    pad2.SetFillStyle(0)

    processes = args.processes[:]
    nominals = {p: None for p in processes}
    systsUp = {p: None for p in processes}
    systsDown = {p: None for p in processes}
    alts = {p: None for p in processes}

    # get nominals
    rf = safeOpenFile(fname0)
    for p in processes:
        print(f"Process {p}")
        if rf.GetDirectory(p):
            # print(f"Browsing file into subfolder {p}")
            f = rf.GetDirectory(p)
            # f.cd(p)
        nominals[p] = safeGetObject(f, f"nominal_{p}_{args.charge}")
        nominals[p].SetTitle(p)
        if args.syst:
            systsUp[p] = safeGetObject(
                f, f"nominal_{p}_{args.syst}Up_{args.charge}", quitOnFail=False
            )
            if systsUp[p] == None:
                systsUp[p] = copy.deepcopy(
                    nominals[p].Clone(f"nominal_{p}_{args.syst}Up_{args.charge}")
                )
            systsUp[p].SetTitle(f"{p} {args.syst}")
            systsDown[p] = safeGetObject(
                f, f"nominal_{p}_{args.syst}Down_{args.charge}", quitOnFail=False
            )
            if systsDown[p] == None:
                systsDown[p] = copy.deepcopy(
                    nominals[p].Clone(f"nominal_{p}_{args.syst}Down_{args.charge}")
                )
            systsDown[p].SetTitle(f"{p} {args.syst}")
    rf.Close()

    rf1 = safeOpenFile(fname1)
    for p in processes:
        if rf1.GetDirectory(p):
            # print(f"Browsing file into subfolder {p}")
            f1 = rf1.GetDirectory(p)
            # f.cd(p)
        alts[p] = safeGetObject(f1, f"nominal_{p}_{args.charge}")
        alts[p].SetTitle(p)
    rf1.Close()

    chargetext = "Positive" if args.charge == "plus" else "Negative"

    pString = "And".join(processes)
    canvasName = f"compareShape_{pString}_{args.syst}_{args.charge}_projPt"
    if "Supplementary" in args.CMStext:
        canvasName += "_supplementary"
    elif "Preliminary" in args.CMStext:
        canvasName += "_preliminary"
    elif "Simulation" in args.CMStext:
        canvasName += "_simulation"

    for ip, p in enumerate(processes):
        if ip:
            h1.Add(
                nominals[p].ProjectionY(
                    f"{nominals[p].GetName()}_pt", 1, nominals[p].GetNbinsX(), "e"
                )
            )
            alt1D.Add(
                alts[p].ProjectionY(
                    f"{alts[p].GetName()}_alt_pt", 1, alts[p].GetNbinsX(), "e"
                )
            )
            syst1DUp.Add(
                systsUp[p].ProjectionY(
                    f"{systsUp[p].GetName()}_pt", 1, systsUp[p].GetNbinsX(), "e"
                )
            )
            syst1DDown.Add(
                systsDown[p].ProjectionY(
                    f"{systsDown[p].GetName()}_pt", 1, systsDown[p].GetNbinsX(), "e"
                )
            )
        else:
            h1 = nominals[p].ProjectionY(
                f"{nominals[p].GetName()}_pt", 1, nominals[p].GetNbinsX(), "e"
            )
            alt1D = alts[p].ProjectionY(
                f"{alts[p].GetName()}_alt_pt", 1, alts[p].GetNbinsX(), "e"
            )
            syst1DUp = systsUp[p].ProjectionY(
                f"{systsUp[p].GetName()}_pt", 1, systsUp[p].GetNbinsX(), "e"
            )
            syst1DDown = systsDown[p].ProjectionY(
                f"{systsDown[p].GetName()}_pt", 1, systsDown[p].GetNbinsX(), "e"
            )

    pZ = processes[1]
    h1Z = nominals[pZ].ProjectionY(
        f"{nominals[pZ].GetName()}_pt", 1, nominals[pZ].GetNbinsX(), "e"
    )

    p = processes[0]
    h1.SetFillColor(colors_plots_[p])
    h1.SetLineColor(colors_plots_[p])
    h1Z.SetFillColor(colors_plots_[pZ])
    h1Z.SetLineColor(colors_plots_[pZ])
    alt1D.SetLineColor(
        ROOT.TColor.GetColor("#f89c20")
    )  # ROOT.TColor.GetColor("#5790fc")
    alt1D.SetLineWidth(3)
    syst1DUp.SetLineStyle(1)
    syst1DUp.SetLineWidth(3)
    syst1DUp.SetLineColor(ROOT.TColor.GetColor("#964A8B"))
    syst1DDown.SetLineStyle(2)
    syst1DDown.SetLineWidth(3)
    syst1DDown.SetLineColor(ROOT.TColor.GetColor("#964A8B"))

    legLabels = []
    for p in processes:
        tmp = legEntries_plots_[p]
        if p == "Wmunu":
            tmp = (
                "W^{+ }#rightarrow^{ }#mu^{+}#nu"
                if args.charge == "plus"
                else "W^{ - }#rightarrow^{ }#mu^{-}#nu"
            )
        legLabels.append(tmp)

    # procLegend = " + ".join(legLabels)
    # procLegend = "W^{ }#rightarrow^{ }#mu#nu + Z^{ }#rightarrow^{ }#mu#mu" # "W + Z"
    procLegend = legLabels[0]
    procLegendZ = legLabels[1]
    procLegendShort = "W + Z"

    xAxisName = chargetext + " muon #it{p}_{T} (GeV)"
    yAxisName = "Events"
    yRatioAxisName = f"Ratio to {procLegendShort}"
    yAxisTitleOffset = 1.58
    transparentLegend = True
    legendCoords = f"{leftMargin+0.02},{1-rightMargin-0.01},0.74,0.92;2"
    moreTextLatex = ""
    lumi = "16.8"

    frame = h1.Clone("frame")
    frame.GetXaxis().SetLabelSize(0.04)
    frame.SetStats(0)

    hlist = [
        h1,
        syst1DUp,
        syst1DDown,
        alt1D,
    ]
    leglist = [procLegend, "#it{m}_{W} + 100 MeV", "#it{m}_{W} - 100 MeV", args.altLeg]
    ymin, ymax = getMinMaxMultiHisto(
        hlist,
        excludeEmpty=True,
        sumError=False,
        excludeUnderflow=True,
        excludeOverflow=True,
    )
    diff = ymax - ymin
    ymax = ymax + 0.7 * diff

    h1.GetXaxis().SetLabelSize(0)
    h1.GetXaxis().SetTitle("")
    h1.GetYaxis().SetTitle(yAxisName)
    h1.GetYaxis().SetTitleOffset(yAxisTitleOffset)
    h1.GetYaxis().SetTitleSize(0.05)
    h1.GetYaxis().SetLabelSize(0.04)
    h1.GetYaxis().SetRangeUser(0, ymax)
    h1.GetYaxis().SetTickSize(0.01)
    h1.Draw("HE")
    h1Z.Draw("HE SAME")
    for ih, h in enumerate(hlist[1:]):
        h.Draw("HIST SAME")

    nColumnsLeg = 1
    legHeader = ""
    if ";" in legendCoords:
        tokens = legendCoords.split(";")
        nColumnsLeg = int(tokens[1])
        if len(tokens) > 2:
            legHeader = tokens[2]
    legcoords = [float(x) for x in (legendCoords.split(";")[0]).split(",")]
    lx1, lx2, ly1, ly2 = legcoords[0], legcoords[1], legcoords[2], legcoords[3]
    leg = ROOT.TLegend(lx1, ly1, lx2, ly2)
    if transparentLegend:
        leg.SetFillColor(0)
        leg.SetFillStyle(0)
        leg.SetFillColorAlpha(0, 0.6)
        leg.SetShadowColor(0)
        leg.SetBorderSize(0)
    leg.SetNColumns(nColumnsLeg)
    if legHeader:
        leg.SetHeader(legHeader)

    for ih, h in enumerate(hlist):
        leg.AddEntry(hlist[ih], leglist[ih], "L" if ih else "F")
        if ih == 0:
            leg.AddEntry(h1Z, procLegendZ, "F")

        h.SetStats(0)
    leg.Draw("same")
    canvas.RedrawAxis("sameaxis")

    if len(moreTextLatex):
        realtext = moreTextLatex.split("::")[0]
        x1, y1, ypass, textsize = 0.75, 0.8, 0.08, 0.035
        if "::" in moreTextLatex:
            x1, y1, ypass, textsize = (
                float(x) for x in (moreTextLatex.split("::")[1]).split(",")
            )
        lat = ROOT.TLatex()
        lat.SetNDC()
        lat.SetTextFont(42)
        lat.SetTextSize(textsize)
        for itx, tx in enumerate(realtext.split(";")):
            lat.DrawLatex(x1, y1 - itx * ypass, tx)

    setTDRStyle()
    latCMS = ROOT.TLatex()
    latCMS.SetNDC()
    latCMS.SetTextFont(42)
    latCMS.SetTextSize(0.04)
    latCMS.DrawLatex(leftMargin, 0.95, f"#bf{{CMS}} #it{{{args.CMStext}}}")
    if lumi != None:
        latCMS.DrawLatex(0.678, 0.95, "%s fb^{-1} (13 TeV)" % lumi)
    else:
        latCMS.DrawLatex(0.678, 0.95, "(13 TeV)")

    if lowerPanelHeight:
        pad2.Draw()
        pad2.cd()

        frame.Reset("ICES")
        # else:
        # frame.GetYaxis().SetRangeUser(0.5,1.5)
        frame.GetYaxis().SetNdivisions(5)
        frame.GetYaxis().SetTitle(yRatioAxisName)
        frame.GetYaxis().SetTitleOffset(yAxisTitleOffset)
        frame.GetYaxis().SetTitleSize(0.05)
        frame.GetYaxis().SetLabelSize(0.04)
        frame.GetYaxis().CenterTitle()
        frame.GetXaxis().SetTitle(xAxisName)
        frame.GetXaxis().SetTitleOffset(1.2)
        frame.GetXaxis().SetTitleSize(0.05)

        if True:
            ratio = copy.deepcopy(h1.Clone("ratio"))
            den_noerr = h1.Clone("den_noerr")
            den_noerr.SetFillColor(0)
            den_noerr.SetLineColor(ROOT.kBlack)
            for iBin in range(1, den_noerr.GetNbinsX() + 1):
                den_noerr.SetBinError(iBin, 0.0)
            ratio.Divide(den_noerr)
            # den_noerr.SetFillColor(ROOT.kGray)
            frame.Draw()
            ratio.SetMarkerSize(0)
            ratio.SetMarkerStyle(0)  # important to remove dots at y = 1
            ratio.SetFillColor(ROOT.kGray + 1)
            ratio.SetLineColor(ROOT.kBlack)
            ratio.SetFillStyle(1001)
            ratio.Draw("E2same")

            ratios = []
            newymin = 0
            newymax = 0
            for i, h in enumerate(hlist[1:]):
                ratios.append(h.Clone("ratio_" + str(i + 1)))
                ratios[-1].Divide(den_noerr)
                ratios[-1].SetLineColor(h.GetLineColor())
                ratios[-1].SetMarkerSize(0)
                ratios[-1].SetMarkerStyle(0)
                ratios[-1].SetFillColor(0)
                ratios[-1].Draw("HIST SAME")

            newymin, newymax = getMinMaxMultiHisto(
                ratios,
                excludeEmpty=True,
                sumError=False,
                excludeUnderflow=True,
                excludeOverflow=True,
            )
            if newymin == newymax:
                newymin *= 0.99
                newymax *= 1.01
            newdiff = newymax - newymin
            # print(f"newdiff = {newdiff}")
            newymin = max(0, newymin - 0.1 * newdiff)
            newymax = newymax + 0.1 * newdiff
            # print(newymin, newymax)
            frame.GetYaxis().SetRangeUser(newymin, newymax)
            pad2.RedrawAxis("sameaxis")

        line = ROOT.TF1(
            "horiz_line",
            "1",
            ratio.GetXaxis().GetBinLowEdge(1),
            ratio.GetXaxis().GetBinLowEdge(ratio.GetNbinsX() + 1),
        )
        line.SetLineColor(ROOT.kBlack)
        line.SetLineWidth(1)
        line.Draw("Lsame")

        yLegRatio = 0.32
        xLegRatio = leftMargin + 0.02
        legRatio = ROOT.TLegend(
            xLegRatio, yLegRatio, xLegRatio + 0.35, yLegRatio + 0.05
        )
        legRatio.SetFillColor(0)
        legRatio.SetFillStyle(0)
        legRatio.SetFillColorAlpha(0, 0.6)
        legRatio.SetShadowColor(0)
        legRatio.SetBorderSize(0)
        legRatio.SetNColumns(1)
        legRatio.AddEntry(ratio, "Stat. unc.", "F")
        legRatio.Draw("SAME")

        pad2.RedrawAxis("sameaxis")

    draw_both0_noLog1_onlyLog2 = 1

    if draw_both0_noLog1_onlyLog2 != 2:
        canvas.SaveAs(outdir + canvasName + ".png")
        canvas.SaveAs(outdir + canvasName + ".pdf")

    if draw_both0_noLog1_onlyLog2 != 1:
        if yAxisName == "a.u.":
            h1.GetYaxis().SetRangeUser(
                max(0.0001, h1.GetMinimum() * 0.8), h1.GetMaximum() * 100
            )
        else:
            h1.GetYaxis().SetRangeUser(
                max(0.001, h1.GetMinimum() * 0.8), h1.GetMaximum() * 100
            )
        canvas.SetLogy()
        canvas.SaveAs(outdir + canvasName + "_logY.png")
        canvas.SaveAs(outdir + canvasName + "_logY.pdf")
        canvas.SetLogy(0)

    # hlist = [nomi1D, alt1D, syst1DUp, syst1DDown]
    # leglist = [legEntries_plots_[p], args.altLeg, "#it{m}_{W} + 100 MeV", "#it{m}_{W} - 100 MeV"]
    # drawNTH1(hlist, leglist, chargetext + " muon #it{p}_{T}", "Events",
    #          f"compareShape_{p}_{args.syst}_{args.charge}_projPt", outdir,
    #          topMargin=0.1, leftMargin=0.16, rightMargin=0.04, labelRatioTmp="Var / nomi",
    #          legendCoords="0.01,0.99,0.70,0.90;2", lowerPanelHeight=0.4, drawLumiLatex=True, passCanvas=canvas1D,
    #          onlyLineColor=True, useLineFirstHistogram=False, setOnlyLineRatio=True, lineWidth=2)

    copyOutputToEos(outdir, outdir_original, eoscp=args.eoscp)
