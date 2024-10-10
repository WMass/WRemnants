#!/usr/bin/env python3

# plot ratios of histogram variation over nominal
# example
# python w-mass-13TeV/makeSystRatios.py /scratch/mciprian/CombineStudies/WMass/scetlibCorr_nnpdf31_testSF_trashdebug/byHelicityPtCharge_correlateEffStatIsoByCharge/WMassCombineInput.root /eos/user/m/mciprian/www/WMassAnalysis/fromMyWremnants/Wmass_fit/scetlibCorr_nnpdf31_testSF_trashdebug/byHelicityPtCharge_correlateEffStatIsoByCharge/makeSystRatios/ -s ".*effStatTnP_trigger_eta20.*q1.*" -c plus --systPostfix effStatTnP_trigger_eta20plus

import re
## safe batch mode
import sys

args = sys.argv[:]
sys.argv = ["-b"]
import ROOT

sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

from copy import *

# sys.path.append(os.getcwd() + "/plotUtils/")
# from utility import *
from scripts.analysisTools.plotUtils.utility import *


def plotUnrolledHistogram(
    h,
    process,
    syst,
    outdir,
    canvas,
    hist2DforBins,
    yAxisTitle="syst/nomi",
    errorBars=False,
    channelCharge=None,
    canvasNamePrefix="",
    canvasFullName=None,
):

    canvas.cd()
    yTitleOffset = 0.65
    histTitle = f"process: {process},      {syst}"
    if channelCharge:
        histTitle = f"Reco charge: {channelCharge} -- " + histTitle
    h.SetTitle(histTitle)
    h.GetXaxis().SetTitleSize(0.05)
    h.GetXaxis().SetTitleOffset(1.05)
    h.GetXaxis().SetLabelSize(0.04)
    h.GetXaxis().SetTitle("Unrolled muon #eta-p_{T} bin")
    h.GetYaxis().SetTitleSize(0.05)
    h.GetYaxis().SetLabelSize(0.04)
    h.GetYaxis().SetTitleOffset(yTitleOffset)
    h.GetYaxis().SetTitle(yAxisTitle)
    h.GetYaxis().SetTickSize(0.01)

    ########
    h.SetLineColor(ROOT.kBlack)
    h.SetMarkerColor(ROOT.kBlack)
    # h.SetMarkerStyle(20)
    # h.SetMarkerSize(0.9)
    h.SetMarkerStyle(1)
    # h.SetFillColor(ROOT.kGray+3)
    # h.SetFillColorAlpha(ROOT.kGray+3, 0.4)

    miny, maxy = getMinMaxHisto(h, sumError=True if errorBars else False)
    diff = maxy - miny
    # print(f"min,max = {miny}, {maxy}:   diff = {diff}")
    # print(f"bin min,max = {h.GetMinimumBin()}, {h.GetMaximumBin()}")
    miny -= diff * 0.1
    maxy += diff * 0.3
    # print(f"new min,max = {miny}, {maxy}")
    h.GetYaxis().SetRangeUser(miny, maxy)

    h.Draw("HE" if errorBars else "HIST")

    bintext = ROOT.TLatex()
    textSize = 0.04
    textAngle = 30
    bintext.SetTextSize(textSize)
    bintext.SetTextFont(42)
    bintext.SetTextAngle(textAngle)

    # to draw panels in the unrolled plots
    nptBins = int(hist2DforBins.GetNbinsY())
    etarange = float(hist2DforBins.GetNbinsX())

    vertline = ROOT.TLine(36, 0, 36, canvas.GetUymax())
    vertline.SetLineColor(ROOT.kBlack)
    vertline.SetLineStyle(3)  # 2 larger hatches
    for i in range(1, nptBins):  # do not need line at canvas borders
        vertline.DrawLine(etarange * i, miny, etarange * i, maxy)

    ptBinRanges = []
    for ipt in range(0, nptBins):
        ptBinRanges.append(
            "#splitline{{[{ptmin},{ptmax}]}}{{GeV}}".format(
                ptmin=int(hist2DforBins.GetYaxis().GetBinLowEdge(ipt + 1)),
                ptmax=int(hist2DforBins.GetYaxis().GetBinLowEdge(ipt + 2)),
            )
        )
    offsetText = etarange / 4.0
    for i in range(0, len(ptBinRanges)):  # we need nptBins texts
        bintext.DrawLatex(etarange * i + offsetText, maxy - 0.2 * diff, ptBinRanges[i])

    line = ROOT.TF1(
        "horiz_line",
        "1",
        h.GetXaxis().GetBinLowEdge(1),
        h.GetXaxis().GetBinLowEdge(h.GetNbinsX() + 1),
    )
    line.SetLineWidth(1)
    line.SetLineColor(ROOT.kRed)
    line.Draw("Lsame")

    canvas.RedrawAxis("sameaxis")

    if canvasNamePrefix:
        canvasNamePrefix += "_"

    canvasName = (
        canvasFullName
        if canvasFullName
        else f"{canvasNamePrefix}unrolled_{process}_{syst}"
    )
    for ext in ["png", "pdf"]:
        canvas.SaveAs(f"{outdir}{canvasName}.{ext}")


if __name__ == "__main__":
    parser = common_plot_parser()
    parser.add_argument("rootfile", type=str, nargs=1, help="Input root file")
    parser.add_argument("outdir", type=str, nargs=1, help="Folder for plots")
    parser.add_argument(
        "-s",
        "--systematics",
        type=str,
        default=".*pdf.*",
        help="Comma separated list of syst names or regular expressions to select systematics to make ratios with nominal",
    )
    parser.add_argument(
        "-p",
        "--processes",
        type=str,
        default="Wmunu",
        help="Comma separated list of processes to plot (full name please)",
    )
    parser.add_argument(
        "--plot",
        dest="plot",
        default="unrolled",
        choices=["2D", "unrolled", "all"],
        help="Plots to make",
    )
    parser.add_argument(
        "--plotNominal",
        dest="plotNominal",
        default=False,
        action="store_true",
        help="Plot nominal histogram",
    )
    parser.add_argument(
        "--plotSystOriginal",
        dest="plotSystOriginal",
        default=False,
        action="store_true",
        help="Plot syst histogram also before ratio with nominal",
    )
    parser.add_argument(
        "--plotStat",
        dest="plotStat",
        default=False,
        action="store_true",
        help="Plot histogram to show statistical uncertainty on each nominal process",
    )
    parser.add_argument(
        "--addErrorBars",
        dest="addErrorBars",
        default=False,
        action="store_true",
        help="Plot error bars on ratio plots for systematics (not recommended if not for test, but better to use --plotStat to check the stat uncertainty)",
    )
    parser.add_argument(
        "--statUncRatio",
        dest="statUncRatio",
        default=False,
        action="store_true",
        help="Add ratio of stat uncertainties between syst and nomi, so see if there are correlations with fluctuating bins in the ratio of yields",
    )
    parser.add_argument(
        "-c",
        "--charge",
        dest="charge",
        default=None,
        choices=["plus", "minus"],
        type=str,
        help="Needs to specify the charge, since all histograms are in the same file (actually this should be more generally just the datacard channel)",
    )
    parser.add_argument(
        "--systPostfix",
        type=str,
        default=None,
        help="Postfix for plot showing some syst variations",
    )
    parser.add_argument(
        "--fakeSystToW",
        default=False,
        action="store_true",
        help="Make additional plots where systs on fakes are translated into signal variation",
    )
    parser.add_argument(
        "--sumFakeAndW",
        default=False,
        action="store_true",
        help="In addition to other plots, and as alternative to --fakeSystToW, sum Fake and W and show variations from varying signal or Fake only",
    )
    parser.add_argument(
        "--compareSingleSystToNomi",
        default=False,
        action="store_true",
        help="Make plots comparing each single variation with nominal, rather than putting all in the same plot when multiple are selected",
    )
    parser.add_argument(
        "--difference",
        dest="addDifference",
        action="store_true",
        help="Also plot difference of syst and nomi, not just their ratio",
    )
    parser.add_argument(
        "--ratioRange",
        default=None,
        type=float,
        nargs=2,
        help="Range for ratio plot (if None, use default from plotted histograms)",
    )
    args = parser.parse_args()

    doUnrolled = (args.plot == "unrolled") or (args.plot == "all")
    do2D = (args.plot == "2D") or (args.plot == "all")

    fname = args.rootfile[0]
    outdir_original = args.outdir[0] + "/"
    outdir = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)

    if not args.charge:
        print(
            "For histograms from WRemnants the charge must be specified using -c [minus|plus]."
        )
        quit()

    processes = args.processes.split(",")
    inputsysts = args.systematics.split(",")
    regexp_syst = re.compile(args.systematics.replace(",", "|"))

    transformFakeIntoSignalVariation = False
    if args.fakeSystToW:
        if all(x in processes for x in ["Wmunu", "Fake"]):
            transformFakeIntoSignalVariation = True
        else:
            print("--fakeSystToW requires both Fake and Wmunu histograms")
            quit()
    if args.sumFakeAndW:
        if not all(x in processes for x in ["Wmunu", "Fake"]):
            print("--sumFakeAndW requires both Fake and Wmunu histograms")
            quit()

    nominals = {p: None for p in processes}
    # print(nominals)
    ratios = {p: [] for p in processes}

    canvas = ROOT.TCanvas("canvas", "", 900, 800)
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

    rf = safeOpenFile(fname)
    # get nominals
    for p in processes:
        print(f"Process {p}")
        if rf.GetDirectory(p):
            print(f"Browsing file into subfolder {p}")
            f = rf.GetDirectory(p)
            # f.cd(p)
        nominals[p] = safeGetObject(f, f"nominal_{p}_{args.charge}")
        nominals[p].SetTitle(p)
        if args.plotNominal:
            if do2D:
                minnomi, maxnomi = getMinMaxHisto(
                    nominals[p], excludeEmpty=True, sumError=False
                )
                drawCorrelationPlot(
                    nominals[p],
                    "Muon #eta",
                    "Muon p_{T} (GeV)",
                    f"Events::{minnomi},{maxnomi}",
                    nominals[p].GetName(),
                    plotLabel="ForceTitle",
                    outdir=outdir,
                    smoothPlot=False,
                    drawProfileX=False,
                    scaleToUnitArea=False,
                    draw_both0_noLog1_onlyLog2=1,
                    palette=args.palette,
                    nContours=args.nContours,
                    invertPalette=args.invertPalette,
                    passCanvas=canvas,
                    drawOption="COLZ0",
                )
            if doUnrolled:
                nomi_unrolled = unroll2Dto1D(
                    nominals[p],
                    newname=f"unrolled_{nominals[p].GetName()}",
                    cropNegativeBins=False,
                )
                plotUnrolledHistogram(
                    nomi_unrolled,
                    p,
                    nominals[p].GetName().replace("x_", ""),
                    outdir,
                    canvas_unroll,
                    nominals[p],
                    yAxisTitle="Events",
                    channelCharge=args.charge,
                )

        if args.plotStat:
            h_relativeStatUnc = copy.deepcopy(
                nominals[p].Clone(f"relativeStatUnc_{p}_{args.charge}")
            )
            for ib in range(1 + h_relativeStatUnc.GetNcells()):
                value = (
                    0.0
                    if nominals[p].GetBinContent(ib) == 0.0
                    else nominals[p].GetBinError(ib) / nominals[p].GetBinContent(ib)
                )
                h_relativeStatUnc.SetBinContent(ib, value)
            if do2D:
                drawCorrelationPlot(
                    h_relativeStatUnc,
                    "Muon #eta",
                    "Muon p_{T} (GeV)",
                    f"Relative stat. uncertainty",
                    h_relativeStatUnc.GetName(),
                    plotLabel="ForceTitle",
                    outdir=outdir,
                    smoothPlot=False,
                    drawProfileX=False,
                    scaleToUnitArea=False,
                    draw_both0_noLog1_onlyLog2=1,
                    palette=args.palette,
                    nContours=args.nContours,
                    invertPalette=args.invertPalette,
                    passCanvas=canvas,
                    drawOption="COLZ0",
                )
            if doUnrolled:
                ratio_unrolled = unroll2Dto1D(
                    h_relativeStatUnc,
                    newname=f"unrolled_{h_relativeStatUnc.GetName()}",
                    cropNegativeBins=False,
                )
                plotUnrolledHistogram(
                    ratio_unrolled,
                    p,
                    f"relStatUnc_{args.charge}",
                    outdir,
                    canvas_unroll,
                    h_relativeStatUnc,
                    yAxisTitle="Relative stat uncertainty",
                    channelCharge=args.charge,
                )

    if args.plotStat:
        quit()

    systList = {p: [] for p in processes}
    systLeg = {p: [] for p in processes}
    systList_eta = {p: [] for p in processes}
    systList_pt = {p: [] for p in processes}
    systList_FakeOnSignal = []
    systList_eta_FakeOnSignal = []
    systList_pt_FakeOnSignal = []
    systList_WmunuAndFake = []
    systList_WmunuAndFake_eta = []
    systList_WmunuAndFake_pt = []
    for ip, p in enumerate(processes):
        systList[p].append(
            unroll2Dto1D(
                nominals[p],
                newname=f"unrolled_{nominals[p].GetName()}",
                cropNegativeBins=False,
            )
        )
        systLeg[p].append(p)
        systList_eta[p].append(
            nominals[p].ProjectionX(
                f"{nominals[p].GetName()}_eta", 1, nominals[p].GetNbinsY(), "e"
            )
        )
        systList_pt[p].append(
            nominals[p].ProjectionY(
                f"{nominals[p].GetName()}_pt", 1, nominals[p].GetNbinsX(), "e"
            )
        )
    if args.sumFakeAndW:
        systList_WmunuAndFake.append(
            copy.deepcopy(systList["Wmunu"][-1].Clone("unrolled_WmunuAndFake"))
        )
        systList_WmunuAndFake_eta.append(
            copy.deepcopy(systList_eta["Wmunu"][-1].Clone("WmunuAndFake_eta"))
        )
        systList_WmunuAndFake_pt.append(
            copy.deepcopy(systList_pt["Wmunu"][-1].Clone("WmunuAndFake_pt"))
        )
        systList_WmunuAndFake[-1].Add(systList["Fake"][-1])
        systList_WmunuAndFake_eta[-1].Add(systList_eta["Fake"][-1])
        systList_WmunuAndFake_pt[-1].Add(systList_pt["Fake"][-1])

    # now make a full loop for systematics
    for p in processes:
        if rf.GetDirectory(p):
            print(f"Browsing file into subfolder {p}")
            f = rf.GetDirectory(p)
        else:
            print(f"Warning: didn't find folder {p}, skipping it")
            continue
        for k in f.GetListOfKeys():
            name = k.GetName()
            if not name.endswith(args.charge):
                continue
            # check name also allowing for perfect matching
            if not any(x in name for x in inputsysts) and not regexp_syst.match(name):
                continue

            # TODO: find more general way to split name, if process or syst name has underscores
            if len(processes) == 1:
                pname = processes[0]
                tokens = name.split("_")
                sname = name.split(f"{pname}_")[1]
                sname = f"{'_'.join(sname.split('_')[:-1])}"
            else:
                tokens = name.split("_")
                pname = f"{tokens[1]}"
                sname = f"{'_'.join(tokens[2:-1])}"
            if pname not in processes:
                continue
            snameLeg = sname.replace("effStatSmooth", "effStat").replace("qall", "")
            systLeg[pname].append(snameLeg)
            sname += f"_{args.charge}"

            alternate = f.Get(name)
            alternate.SetDirectory(0)
            systList[pname].append(
                unroll2Dto1D(
                    alternate,
                    newname=f"unrolled_{alternate.GetName()}",
                    cropNegativeBins=False,
                )
            )
            systList_eta[pname].append(
                alternate.ProjectionX(
                    f"{alternate.GetName()}_eta", 1, alternate.GetNbinsY(), "e"
                )
            )
            systList_pt[pname].append(
                alternate.ProjectionY(
                    f"{alternate.GetName()}_pt", 1, alternate.GetNbinsX(), "e"
                )
            )
            if transformFakeIntoSignalVariation and pname == "Fake":
                systList_FakeOnSignal.append(
                    copy.deepcopy(
                        systList[pname][-1].Clone(
                            f"unrolled_{alternate.GetName()}_FakeOnSignal"
                        )
                    )
                )
                systList_eta_FakeOnSignal.append(
                    copy.deepcopy(
                        systList_eta[pname][-1].Clone(
                            f"{alternate.GetName()}_eta_FakeOnSignal"
                        )
                    )
                )
                systList_pt_FakeOnSignal.append(
                    copy.deepcopy(
                        systList_pt[pname][-1].Clone(
                            f"{alternate.GetName()}_pt_FakeOnSignal"
                        )
                    )
                )
                # subtract nominal
                systList_FakeOnSignal[-1].Add(systList["Fake"][0], -1.0)
                systList_eta_FakeOnSignal[-1].Add(systList_eta["Fake"][0], -1.0)
                systList_pt_FakeOnSignal[-1].Add(systList_pt["Fake"][0], -1.0)
                # add nominal signal
                systList_FakeOnSignal[-1].Add(systList["Wmunu"][0])
                systList_eta_FakeOnSignal[-1].Add(systList_eta["Wmunu"][0])
                systList_pt_FakeOnSignal[-1].Add(systList_pt["Wmunu"][0])

            ratio = alternate.Clone(f"systRatio_{sname}")
            difference = alternate.Clone(f"systDiff_{sname}")
            if args.plotSystOriginal:
                if do2D:
                    drawCorrelationPlot(
                        alternate,
                        "Muon #eta",
                        "Muon p_{T} (GeV)",
                        "Events",
                        f"{name}_Syst",
                        plotLabel="ForceTitle",
                        outdir=outdir,
                        smoothPlot=False,
                        drawProfileX=False,
                        scaleToUnitArea=False,
                        draw_both0_noLog1_onlyLog2=1,
                        palette=args.palette,
                        nContours=args.nContours,
                        invertPalette=args.invertPalette,
                        passCanvas=canvas,
                        drawOption="COLZ0",
                    )
                if doUnrolled:
                    plotUnrolledHistogram(
                        systList[pname][-1],
                        pname,
                        sname + "_Syst",
                        outdir,
                        canvas_unroll,
                        ratio,
                        errorBars=args.addErrorBars,
                        yAxisTitle="Events",
                        channelCharge=args.charge,
                    )

            ratio.SetDirectory(0)
            # ratios[pname].append(htmp.Divide(nominals[pname]))
            ratio.Divide(nominals[pname])
            ratio.SetTitle(f"#splitline{{syst: {sname}}}{{proc: {pname}}}")
            if args.addDifference:
                difference.SetDirectory(0)
                difference.Add(nominals[pname], -1.0)
                difference.SetTitle(f"#splitline{{syst: {sname}}}{{proc: {pname}}}")

            if do2D:
                minratio, maxratio = getMinMaxHisto(
                    ratio, excludeEmpty=True, sumError=False
                )
                drawCorrelationPlot(
                    ratio,
                    "Muon #eta",
                    "Muon p_{T} (GeV)",
                    f"{pname}: syst / nominal::{minratio},{maxratio}",
                    f"ratio_{name}",
                    plotLabel="ForceTitle",
                    outdir=outdir,
                    smoothPlot=False,
                    drawProfileX=False,
                    scaleToUnitArea=False,
                    draw_both0_noLog1_onlyLog2=1,
                    palette=args.palette,
                    nContours=args.nContours,
                    invertPalette=args.invertPalette,
                    passCanvas=canvas,
                    drawOption="COLZ0",
                )
                if args.addDifference:
                    drawCorrelationPlot(
                        difference,
                        "Muon #eta",
                        "Muon p_{T} (GeV)",
                        f"Difference: syst - nominal",
                        f"diff_{name}",
                        plotLabel="ForceTitle",
                        outdir=outdir,
                        smoothPlot=False,
                        drawProfileX=False,
                        scaleToUnitArea=False,
                        draw_both0_noLog1_onlyLog2=1,
                        palette=args.palette,
                        nContours=args.nContours,
                        invertPalette=args.invertPalette,
                        passCanvas=canvas,
                        drawOption="COLZ0",
                    )

            if doUnrolled:
                ratio_unrolled = unroll2Dto1D(
                    ratio, newname=f"unrolledRatio_{name}", cropNegativeBins=False
                )
                plotUnrolledHistogram(
                    ratio_unrolled,
                    pname,
                    sname,
                    outdir,
                    canvas_unroll,
                    ratio,
                    errorBars=args.addErrorBars,
                    channelCharge=args.charge,
                    canvasFullName=ratio_unrolled.GetName(),
                )
                if args.addDifference:
                    difference_unrolled = unroll2Dto1D(
                        difference,
                        newname=f"unrolledDifference_{name}",
                        cropNegativeBins=False,
                    )
                    plotUnrolledHistogram(
                        difference_unrolled,
                        pname,
                        sname,
                        outdir,
                        canvas_unroll,
                        ratio,
                        errorBars=args.addErrorBars,
                        channelCharge=args.charge,
                        canvasFullName=difference_unrolled.GetName(),
                        yAxisTitle="syst - nomi",
                    )

                if args.statUncRatio:
                    statUncNomi = nominals[pname].Clone(
                        f"statUnc_{nominals[pname].GetName()}"
                    )
                    fillTH2fromTH2part(statUncNomi, nominals[pname], fillWithError=True)
                    statUncAlt = alternate.Clone(f"statUnc_{alternate.GetName()}")
                    fillTH2fromTH2part(statUncAlt, alternate, fillWithError=True)
                    ratioStatUnc = statUncAlt.Clone(f"ratioStatUncSystOverNomi_{sname}")
                    ratioStatUnc.Divide(statUncNomi)
                    ratioStatUnc_unrolled = unroll2Dto1D(
                        ratioStatUnc,
                        newname=f"unrolledStatUncRatio_{name}",
                        cropNegativeBins=False,
                    )
                    plotUnrolledHistogram(
                        ratioStatUnc_unrolled,
                        pname,
                        f"{sname}_statUncRatioWithNomi",
                        outdir,
                        canvas_unroll,
                        ratioStatUnc,
                        errorBars=False,
                        yAxisTitle="stat. uncertainty ratio: syst/nomi",
                        channelCharge=args.charge,
                    )
    print()

    ratioRange = ""
    if args.ratioRange != None:
        ratioRange = f"::{args.ratioRange[0]},{args.ratioRange[1]}"

    ptBinRanges = []
    for i in range(nominals[p].GetNbinsY()):
        ptBinRanges.append("")  # keep dummy otherwise there's too much text here
    systPostfix = f"_{args.charge}"
    if args.systPostfix:
        systPostfix += f"_{args.systPostfix}"
    for p in processes:
        if args.compareSingleSystToNomi:
            systNames = [l for l in systLeg[p][1:]]
            for isys, s in enumerate(systLeg[p]):
                if isys == 0 or s not in systNames:
                    continue  # logic needed also to skip systs (Down/Up which might have been plotted already)
                systNames.remove(s)
                hlist = [systList[p][0], systList[p][isys]]
                hlist_eta = [systList_eta[p][0], systList_eta[p][isys]]
                hlist_pt = [systList_pt[p][0], systList_pt[p][isys]]
                leglist = [systLeg[p][0], systLeg[p][isys]]
                leglist_eta = [x for x in leglist]
                leglist_pt = [x for x in leglist]
                if "Up" in s or "Down" in s:
                    antis = (
                        s.replace("Down", "Up")
                        if "Down" in s
                        else s.replace("Up", "Down")
                    )
                    if antis in systNames:
                        antisID = systLeg[p].index(antis)
                        hlist.append(systList[p][antisID])
                        hlist_eta.append(systList_eta[p][antisID])
                        hlist_pt.append(systList_pt[p][antisID])
                        leglist.append(systLeg[p][antisID])
                        leglist_eta.append(systLeg[p][antisID])
                        leglist_pt.append(systLeg[p][antisID])
                        systNames.remove(antis)
                systPostfixSingle = s.replace("Down", "").replace("Up", "")
                drawNTH1(
                    hlist,
                    leglist,
                    "Unrolled eta-p_{T} bin",
                    "Events",
                    f"nominalAndSyst_{p}_{systPostfixSingle}",
                    outdir,
                    topMargin=0.2,
                    leftMargin=0.06,
                    rightMargin=0.01,
                    labelRatioTmp=f"syst/nomi{ratioRange}",
                    legendCoords="0.06,0.99,0.84,0.99;4",
                    lowerPanelHeight=0.5,
                    skipLumi=True,
                    passCanvas=canvas_unroll,
                    yAxisExtendConstant=1.4,
                    ytextOffsetFromTop=0.22,
                    drawVertLines="{a},{b}".format(
                        a=nominals[p].GetNbinsY(), b=nominals[p].GetNbinsX()
                    ),
                    textForLines=ptBinRanges,
                    transparentLegend=False,
                    onlyLineColor=True,
                    noErrorRatioDen=True,
                    useLineFirstHistogram=True,
                    setOnlyLineRatio=True,
                    lineWidth=1,
                )
                #
                drawNTH1(
                    hlist_eta,
                    leglist_eta,
                    "Muon eta",
                    "Events",
                    f"nominalAndSyst_{p}_{systPostfixSingle}_projEta",
                    outdir,
                    topMargin=0.25,
                    leftMargin=0.16,
                    rightMargin=0.05,
                    labelRatioTmp="syst/nomi",
                    legendCoords="0.01,0.99,0.80,0.99;1",
                    lowerPanelHeight=0.4,
                    skipLumi=True,
                    passCanvas=canvas1D,
                    transparentLegend=False,
                    onlyLineColor=True,
                    noErrorRatioDen=True,
                    useLineFirstHistogram=True,
                    setOnlyLineRatio=True,
                    lineWidth=2,
                )
                #
                drawNTH1(
                    hlist_pt,
                    leglist_pt,
                    "Muon p_{T} (GeV)",
                    "Events",
                    f"nominalAndSyst_{p}_{systPostfixSingle}_projPt",
                    outdir,
                    topMargin=0.25,
                    leftMargin=0.16,
                    rightMargin=0.05,
                    labelRatioTmp="syst/nomi",
                    legendCoords="0.01,0.99,0.80,0.99;1",
                    lowerPanelHeight=0.4,
                    skipLumi=True,
                    passCanvas=canvas1D,
                    transparentLegend=False,
                    onlyLineColor=True,
                    noErrorRatioDen=True,
                    useLineFirstHistogram=True,
                    setOnlyLineRatio=True,
                    lineWidth=2,
                )
        else:
            if len(systList[p]) > 11:
                print(
                    "Not running drawNTH1() function to draw curves, there are too many lines ({})".format(
                        len(systList[p])
                    )
                )
            else:
                drawNTH1(
                    systList[p],
                    systLeg[p],
                    "Unrolled eta-p_{T} bin",
                    "Events",
                    f"nominalAndSyst_{p}_{systPostfix}",
                    outdir,
                    topMargin=0.2,
                    leftMargin=0.06,
                    rightMargin=0.01,
                    labelRatioTmp=f"syst/nomi{ratioRange}",
                    legendCoords="0.06,0.99,0.84,0.99;4",
                    lowerPanelHeight=0.5,
                    skipLumi=True,
                    passCanvas=canvas_unroll,
                    yAxisExtendConstant=1.4,
                    ytextOffsetFromTop=0.22,
                    drawVertLines="{a},{b}".format(
                        a=nominals[p].GetNbinsY(), b=nominals[p].GetNbinsX()
                    ),
                    textForLines=ptBinRanges,
                    transparentLegend=False,
                    onlyLineColor=True,
                    noErrorRatioDen=True,
                    useLineFirstHistogram=True,
                    setOnlyLineRatio=True,
                    lineWidth=1,
                )
                #
                drawNTH1(
                    systList_eta[p],
                    systLeg[p],
                    "Muon eta",
                    "Events",
                    f"nominalAndSyst_{p}_{systPostfix}_projEta",
                    outdir,
                    topMargin=0.25,
                    leftMargin=0.16,
                    rightMargin=0.05,
                    labelRatioTmp="syst/nomi",
                    legendCoords="0.01,0.99,0.80,0.99;2",
                    lowerPanelHeight=0.4,
                    skipLumi=True,
                    passCanvas=canvas1D,
                    transparentLegend=False,
                    onlyLineColor=True,
                    noErrorRatioDen=True,
                    useLineFirstHistogram=True,
                    setOnlyLineRatio=True,
                    lineWidth=2,
                )
                #
                drawNTH1(
                    systList_pt[p],
                    systLeg[p],
                    "Muon p_{T} (GeV)",
                    "Events",
                    f"nominalAndSyst_{p}_{systPostfix}_projPt",
                    outdir,
                    topMargin=0.25,
                    leftMargin=0.16,
                    rightMargin=0.05,
                    labelRatioTmp="syst/nomi",
                    legendCoords="0.01,0.99,0.80,0.99;2",
                    lowerPanelHeight=0.4,
                    skipLumi=True,
                    passCanvas=canvas1D,
                    transparentLegend=False,
                    onlyLineColor=True,
                    noErrorRatioDen=True,
                    useLineFirstHistogram=True,
                    setOnlyLineRatio=True,
                    lineWidth=2,
                )
                if transformFakeIntoSignalVariation and p == "Wmunu":
                    histFakesOnSignal = [systList[p][0]] + [
                        x for x in systList_FakeOnSignal
                    ]
                    drawNTH1(
                        histFakesOnSignal,
                        systLeg[p],
                        "Unrolled eta-p_{T} bin",
                        "Events",
                        f"nominalAndSyst_{p}_{systPostfix}_FakeOnSignal",
                        outdir,
                        topMargin=0.2,
                        leftMargin=0.06,
                        rightMargin=0.01,
                        labelRatioTmp="syst/nomi",
                        legendCoords="0.06,0.99,0.84,0.99;4",
                        lowerPanelHeight=0.5,
                        skipLumi=True,
                        passCanvas=canvas_unroll,
                        yAxisExtendConstant=1.4,
                        ytextOffsetFromTop=0.22,
                        drawVertLines="{a},{b}".format(
                            a=nominals[p].GetNbinsY(), b=nominals[p].GetNbinsX()
                        ),
                        textForLines=ptBinRanges,
                        transparentLegend=False,
                        onlyLineColor=True,
                        noErrorRatioDen=True,
                        useLineFirstHistogram=True,
                        setOnlyLineRatio=True,
                        lineWidth=1,
                    )
                    #
                    histFakesOnSignal = [systList_eta[p][0]] + [
                        x for x in systList_eta_FakeOnSignal
                    ]
                    drawNTH1(
                        histFakesOnSignal,
                        systLeg[p],
                        "Muon eta",
                        "Events",
                        f"nominalAndSyst_{p}_{systPostfix}_projEta_FakeOnSignal",
                        outdir,
                        topMargin=0.25,
                        leftMargin=0.16,
                        rightMargin=0.05,
                        labelRatioTmp=f"syst/nomi{ratioRange}",
                        legendCoords="0.01,0.99,0.80,0.99;2",
                        lowerPanelHeight=0.4,
                        skipLumi=True,
                        passCanvas=canvas1D,
                        transparentLegend=False,
                        onlyLineColor=True,
                        noErrorRatioDen=True,
                        useLineFirstHistogram=True,
                        setOnlyLineRatio=True,
                        lineWidth=2,
                    )
                    #
                    histFakesOnSignal = [systList_pt[p][0]] + [
                        x for x in systList_pt_FakeOnSignal
                    ]
                    drawNTH1(
                        histFakesOnSignal,
                        systLeg[p],
                        "Muon p_{T} (GeV)",
                        "Events",
                        f"nominalAndSyst_{p}_{systPostfix}_projPt_FakeOnSignal",
                        outdir,
                        topMargin=0.25,
                        leftMargin=0.16,
                        rightMargin=0.05,
                        labelRatioTmp="syst/nomi",
                        legendCoords="0.01,0.99,0.80,0.99;2",
                        lowerPanelHeight=0.4,
                        skipLumi=True,
                        passCanvas=canvas1D,
                        transparentLegend=False,
                        onlyLineColor=True,
                        noErrorRatioDen=True,
                        useLineFirstHistogram=True,
                        setOnlyLineRatio=True,
                        lineWidth=2,
                    )

    if args.sumFakeAndW:
        # create syst vars
        systLeg_WmunuAndFake = ["Wmunu + Fake"]
        for i in range(1, len(systList["Wmunu"])):
            systList_WmunuAndFake.append(
                copy.deepcopy(
                    systList["Fake"][0].Clone(f"unrolled_WmunuAndFake_syst{i}Wmunu")
                )
            )
            systList_WmunuAndFake_eta.append(
                copy.deepcopy(
                    systList_eta["Fake"][0].Clone(f"WmunuAndFake_syst{i}Wmunu_eta")
                )
            )
            systList_WmunuAndFake_pt.append(
                copy.deepcopy(
                    systList_pt["Fake"][0].Clone(f"WmunuAndFake_syst{i}Wmunu_pt")
                )
            )
            # add Signal var
            systList_WmunuAndFake[-1].Add(systList["Wmunu"][i])
            systList_WmunuAndFake_eta[-1].Add(systList_eta["Wmunu"][i])
            systList_WmunuAndFake_pt[-1].Add(systList_pt["Wmunu"][i])
            systLeg_WmunuAndFake.append(systLeg["Wmunu"][i] + " (W)")
        for i in range(1, len(systList["Fake"])):
            systList_WmunuAndFake.append(
                copy.deepcopy(
                    systList["Wmunu"][0].Clone(f"unrolled_WmunuAndFake_syst{i}Fake")
                )
            )
            systList_WmunuAndFake_eta.append(
                copy.deepcopy(
                    systList_eta["Wmunu"][0].Clone(f"WmunuAndFake_syst{i}Fake_eta")
                )
            )
            systList_WmunuAndFake_pt.append(
                copy.deepcopy(
                    systList_pt["Wmunu"][0].Clone(f"WmunuAndFake_syst{i}Fake_pt")
                )
            )
            # add Fake var
            systList_WmunuAndFake[-1].Add(systList["Fake"][i])
            systList_WmunuAndFake_eta[-1].Add(systList_eta["Fake"][i])
            systList_WmunuAndFake_pt[-1].Add(systList_pt["Fake"][i])
            systLeg_WmunuAndFake.append(systLeg["Fake"][i] + " (Fake)")

        drawNTH1(
            systList_WmunuAndFake,
            systLeg_WmunuAndFake,
            "Unrolled eta-p_{T} bin",
            "Events",
            f"nominalAndSyst_{systPostfix}_WmunuAndFake",
            outdir,
            topMargin=0.2,
            leftMargin=0.06,
            rightMargin=0.01,
            labelRatioTmp=f"syst/nomi{ratioRange}",
            legendCoords="0.06,0.99,0.84,0.99;4",
            lowerPanelHeight=0.5,
            skipLumi=True,
            passCanvas=canvas_unroll,
            yAxisExtendConstant=1.4,
            ytextOffsetFromTop=0.22,
            drawVertLines="{a},{b}".format(
                a=nominals[p].GetNbinsY(), b=nominals[p].GetNbinsX()
            ),
            textForLines=ptBinRanges,
            transparentLegend=False,
            onlyLineColor=True,
            noErrorRatioDen=True,
            useLineFirstHistogram=True,
            setOnlyLineRatio=True,
            lineWidth=1,
        )
        #
        drawNTH1(
            systList_WmunuAndFake_eta,
            systLeg_WmunuAndFake,
            "Muon eta",
            "Events",
            f"nominalAndSyst_{systPostfix}_projEta_WmunuAndFake",
            outdir,
            topMargin=0.25,
            leftMargin=0.16,
            rightMargin=0.05,
            labelRatioTmp="syst/nomi",
            legendCoords="0.01,0.99,0.80,0.99;2",
            lowerPanelHeight=0.4,
            skipLumi=True,
            passCanvas=canvas1D,
            transparentLegend=False,
            onlyLineColor=True,
            noErrorRatioDen=True,
            useLineFirstHistogram=True,
            setOnlyLineRatio=True,
            lineWidth=2,
        )
        #
        drawNTH1(
            systList_WmunuAndFake_pt,
            systLeg_WmunuAndFake,
            "Muon p_{T} (GeV)",
            "Events",
            f"nominalAndSyst_{systPostfix}_projPt_WmunuAndFake",
            outdir,
            topMargin=0.25,
            leftMargin=0.16,
            rightMargin=0.05,
            labelRatioTmp="syst/nomi",
            legendCoords="0.01,0.99,0.80,0.99;2",
            lowerPanelHeight=0.4,
            skipLumi=True,
            passCanvas=canvas1D,
            transparentLegend=False,
            onlyLineColor=True,
            noErrorRatioDen=True,
            useLineFirstHistogram=True,
            setOnlyLineRatio=True,
            lineWidth=2,
        )
        #

    copyOutputToEos(outdir, outdir_original, eoscp=args.eoscp)
    print()
