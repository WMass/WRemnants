#!/usr/bin/env python3

# plot all SF from the input root file, and make some products for faster usage
# for isolation and antiisolation, new SF can be defined by varying only the data efficiecy or MC one, which is needed to properly account for the anticorrelation between isolated and antiisolated leptons (forcefully anticorrelating the global sf uncertainty is not correct, the 100% anticorrelation holds only for the efficiencies)
# actually, we verified that the correlation is preserved also for the SF, since isoSF and antiisoSF are found to be generally anticorrelated by more than 95% (even 99% for |eta| < 2.0) for pt < 60 GeV
# no need to do the same also for iso with no trigger, because this splitting is relevant when there are fakes, and for the 2 lepton phase space that background is negligible

# python w_mass_13TeV/plotSF.py file.root output/folder/ -e 'GtoH' -n 'trigger,idip,iso,antiiso,isonotrig,antiisonotrig' -wpc reco tracking idip trigger [--make-prod] [--skip-eff]

import logging

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
    common_plot_parser,
    copyOutputToEos,
    createPlotDirAndCopyPhp,
    drawCorrelationPlot,
    drawNTH1,
    getTH2morePtBins,
    multiplyByHistoWith1ptBin,
    multiplyByHistoWithLessPtBins,
    safeGetObject,
    safeOpenFile,
    unroll2Dto1D,
)

logging.basicConfig(level=logging.INFO)

workingPoints_ = ["reco", "tracking", "idip", "trigger", "iso", "isonotrig", "veto"]

if __name__ == "__main__":
    parser = common_plot_parser()
    parser.add_argument("rootfile", type=str, nargs=1)
    parser.add_argument(
        "outdir",
        type=str,
        nargs=1,
        help="Output folder (subfolder 'allSF/' created automatically inside)",
    )
    parser.add_argument(
        "-e",
        "--era",
        type=str,
        default="BtoF,GtoH,B,C,D,E,F,G,H",
        help="Comma separated list of eras for SF in histogram name; default: %(default)s",
    )
    parser.add_argument(
        "-n",
        "--sfnames",
        type=str,
        default="trigger,idip,iso,antiiso,isonotrig,antiisonotrig,tracking,reco",
        help="Comma separated list of SF names inside root file, which will be plotted; default: %(default)s",
    )
    parser.add_argument(
        "-wpc",
        "--workinPointsByCharge",
        default=["trigger", ",idip", "tracking", "reco"],
        nargs="*",
        type=str,
        choices=list(workingPoints_),
        help="Working points made by charge",
    )
    parser.add_argument(
        "--sf-version",
        dest="sfversions",
        type=str,
        default="nominal,dataAltSig",
        help="SF versions to plot and to use for the products, usually one would use just nominal and dataAltSig to define the systematic variation; default: %(default)s",
    )
    parser.add_argument(
        "--eff-version",
        dest="effversions",
        type=str,
        default="nominal,altSig",
        help="Efficiency versions to plot (nominal actually has no keyword); default: %(default)s",
    )
    parser.add_argument(
        "--skip-eff",
        dest="skipEfficiency",
        action="store_true",
        help="Do not plot efficiencies to save time",
    )
    parser.add_argument(
        "--make-prod",
        dest="makeProduct",
        action="store_true",
        help="Make and plot products of scale factors",
    )
    parser.add_argument(
        "--bin-pt-finely",
        dest="binPtFinely",
        default=-1,
        type=int,
        help="Bin product histograms so to increase number of pt bins",
    )
    args = parser.parse_args()

    ROOT.TH1.SetDefaultSumw2()

    # products of scale factors
    # should be updated to use working points by charge other than trigger
    # on the other hand we no longer use these products, so there is no need to make them anymore
    productsToMake = {}
    if args.makeProduct:
        productsToMake = {
            "isoTrigPlus": ["iso", "triggerplus", "idip"],
            "isoTrigMinus": ["iso", "triggerminus", "idip"],
            "isoNotrig": ["isonotrig", "idip"],
            "noisoTrigPlus": ["triggerplus", "idip"],
            "noisoTrigMinus": ["triggerminus", "idip"],
            "noisoNotrig": ["idip"],
            "antiisoTrigPlus": ["antiiso", "triggerplus", "idip"],
            "antiisoTrigMinus": ["antiiso", "triggerminus", "idip"],
            "antiisoNotrig": ["antiisonotrig", "idip"],
            "isoOnly": ["iso"],
            "isoNotrigOnly": ["isonotrig"],
            "antiisoOnly": ["antiiso"],
            "antiisoNotrigOnly": ["antiisonotrig"],
            "recoOnly": ["reco"],
            "trackingOnly": ["tracking"],
            "trigPlusOnly": ["triggerplus"],
            "trigMinusOnly": ["triggerminus"],
        }

    eras = args.era.split(",")
    fname = args.rootfile[0]
    outdir_original = args.outdir[0]  # to keep it below
    if not outdir_original.endswith("/"):
        outdir_original = outdir_original + "/"
    outdir_original += "allSF/"
    productSubfolder = "productSF/"

    outdir_local = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)

    # make folder structure
    foldersToCreate = []
    if args.makeProduct:
        foldersToCreate.append(outdir_local + productSubfolder)
    for era in eras:
        foldersToCreate.append(outdir_local + era + "/")
        if args.makeProduct:
            foldersToCreate.append(outdir_local + productSubfolder + era + "/")

    for folder in foldersToCreate:
        createPlotDirAndCopyPhp(folder)
    print()

    histsSF = {}
    histsEff = {}
    for era in eras:
        histsSF[era] = {}
        histsEff[era] = {"Data": {}, "MC": {}}

    sf_version = [str(x) for x in args.sfversions.split(",")]
    eff_version = [str(x) for x in args.effversions.split(",")]

    names = args.sfnames.split(",")
    f = safeOpenFile(fname)
    for n in names:
        print("")
        print(f"Working point: {n}")
        print("-" * 30)
        charges = ["plus", "minus"] if n in args.workinPointsByCharge else ["both"]
        for ch in charges:
            for era in eras:
                tmpEra = "F_preVFP" if era == "F" else era
                for v in sf_version:
                    hname = f"SF2D_{v}_{n}_{tmpEra}_{ch}"
                    hkey = f"{v}_{n}"
                    hkey += ch if n in args.workinPointsByCharge else ""
                    print(f"   {hkey} -> {hname}")
                    histsSF[era][hkey] = safeGetObject(f, hname)
                for v in eff_version:
                    for dataMC in ["Data", "MC"]:
                        realv = "" if v == "nominal" else f"{v}_"
                        hname = f"eff{dataMC}_{realv}{n}_{tmpEra}_{ch}"
                        hkey = f"{v}_{n}"
                        hkey += ch if n in args.workinPointsByCharge else ""
                        print(f"   {hkey} -> {hname}")
                        histsEff[era][dataMC][hkey] = safeGetObject(f, hname)
    f.Close()

    canvas = ROOT.TCanvas("canvas", "", 800, 800)

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

    # first plot efficiencies
    if not args.skipEfficiency:
        for era in eras:
            outdir = outdir_local + era + "/"
            for dataMC in ["Data", "MC"]:
                for n in list(histsEff[era][dataMC].keys()):
                    drawCorrelationPlot(
                        histsEff[era][dataMC][n],
                        "muon #eta",
                        "muon p_{T} (GeV)",
                        f"{dataMC} efficiency",
                        f"muonEff{dataMC}_{n}",
                        plotLabel="ForceTitle",
                        outdir=outdir,
                        smoothPlot=False,
                        drawProfileX=False,
                        scaleToUnitArea=False,
                        draw_both0_noLog1_onlyLog2=1,
                        passCanvas=canvas,
                        nContours=args.nContours,
                        palette=args.palette,
                        invertPalette=args.invertPalette,
                    )
                    # abs. uncertainty (only on nominal, it should be the same for all histograms, hoping the SF and efficiencies were sane in this configuration)
                    if "nominal" in n:
                        drawCorrelationPlot(
                            histsEff[era][dataMC][n],
                            "muon #eta",
                            "muon p_{T} (GeV)",
                            f"Abs. uncertainty on {dataMC} eff",
                            f"absUnc_muonEff{dataMC}_{n}",
                            plotLabel="ForceTitle",
                            outdir=outdir + "absoluteStatUncertainty/",
                            smoothPlot=False,
                            drawProfileX=False,
                            scaleToUnitArea=False,
                            draw_both0_noLog1_onlyLog2=1,
                            passCanvas=canvas,
                            plotError=True,
                            nContours=args.nContours,
                            palette=args.palette,
                            invertPalette=args.invertPalette,
                        )
                        ## rel. uncertainty
                        drawCorrelationPlot(
                            histsEff[era][dataMC][n],
                            "muon #eta",
                            "muon p_{T} (GeV)",
                            f"Rel. uncertainty on {dataMC} eff",
                            f"relUnc_muonEff{dataMC}_{n}",
                            plotLabel="ForceTitle",
                            outdir=outdir + "relativeStatUncertainty/",
                            smoothPlot=False,
                            drawProfileX=False,
                            scaleToUnitArea=False,
                            draw_both0_noLog1_onlyLog2=1,
                            passCanvas=canvas,
                            plotRelativeError=True,
                            nContours=args.nContours,
                            palette=args.palette,
                            invertPalette=args.invertPalette,
                        )

    prodHistsSF = {}
    for era in eras:
        prodHistsSF[era] = {}

    namesForCheck = [x for x in names if x not in args.workinPointsByCharge]
    for x in args.workinPointsByCharge:
        if x in names:
            namesForCheck += [f"{x}plus", f"{x}minus"]

    for key in list(productsToMake.keys()):
        if not all(x in namesForCheck for x in productsToMake[key]):
            print()
            missingFactors = ",".join(
                [x for x in productsToMake[key] if x not in namesForCheck]
            )
            print(
                f"Warning: skipping product {key} because these factors are missing: {missingFactors}"
            )
            print()
            continue
        for era in eras:
            for sfv in sf_version:
                prodname = f"fullSF2D_{sfv}_{key}_{era}"
                yedges = []
                for i, basename in enumerate(productsToMake[key]):
                    name = f"{sfv}_{basename}"
                    if i == 0:
                        stringProduct = basename
                        if (
                            basename not in ["reco", "tracking"]
                            and args.binPtFinely > 0
                            and histsSF[era][name].GetNbinsY() < args.binPtFinely
                        ):
                            print(
                                f"INFO: Going to create product histogram {prodname} with {args.binPtFinely} pt bins but same range"
                            )
                            prodHistsSF[era][prodname] = getTH2morePtBins(
                                histsSF[era][name], prodname, args.binPtFinely
                            )
                        else:
                            prodHistsSF[era][prodname] = copy.deepcopy(
                                histsSF[era][name].Clone(prodname)
                            )
                        yedges = [
                            round(
                                prodHistsSF[era][prodname].GetYaxis().GetBinLowEdge(i),
                                2,
                            )
                            for i in range(
                                1, 2 + prodHistsSF[era][prodname].GetNbinsY()
                            )
                        ]
                    else:
                        stringProduct = stringProduct + "*" + basename
                        if histsSF[era][name].GetNbinsY() == 1:
                            multiplyByHistoWith1ptBin(
                                prodHistsSF[era][prodname], histsSF[era][name]
                            )
                        elif not prodHistsSF[era][prodname].Multiply(
                            histsSF[era][name]
                        ):
                            yedgesSecond = [
                                round(histsSF[era][name].GetYaxis().GetBinLowEdge(i), 2)
                                for i in range(1, 2 + histsSF[era][name].GetNbinsY())
                            ]
                            # try to see if first histogram has pt edges which are a subset of the other one to attempt multiplication
                            if all(
                                yedgesSecond[i] in yedges
                                for i in range(len(yedgesSecond))
                            ):
                                print(
                                    f"WARNING: going to multiply prodHistsSF[{era}][{prodname}] with {name} using function multiplyByHistoWithLessPtBins()"
                                )
                                multiplyByHistoWithLessPtBins(
                                    prodHistsSF[era][prodname], histsSF[era][name]
                                )
                            else:
                                print(
                                    f"ERROR in multiplication for prodHistsSF[{era}][{prodname}] with {name}"
                                )
                                print(
                                    f"Nbins(X, Y) = {histsSF[era][name].GetNbinsX()},{histsSF[era][name].GetNbinsY()} "
                                )
                                quit()

                prodHistsSF[era][prodname].SetTitle(f"{stringProduct}")
                print(f"{era}: {sfv} -> {stringProduct}")

    fullout = outdir_local + "scaleFactorProduct.root"
    f = ROOT.TFile.Open(fullout, "RECREATE")
    # put new ones in this files, without copying original histograms (although some are actually bare copies when the product has a single factor)
    if not f or not f.IsOpen():
        raise RuntimeError(f"Error when opening file {fullout}")
    f.cd()
    for era in eras:
        # save the product of scale factors
        for n in list(prodHistsSF[era].keys()):
            prodHistsSF[era][n].Write(n)
    f.Close()

    minmax = {
        "trigger": "0.65,1.15",
        "idip": "0.95,1.01",
        "iso": "0.975,1.025",
        "antiiso": "0.6,1.25",
        "isonotrig": "0.97,1.03",
        "antiisonotrig": "0.6,1.25",
        "tracking": "0.98,1.01",
        "reco": "0.94,1.02",
    }

    for era in eras:

        outdir = outdir_local + era + "/"

        for n in list(histsSF[era].keys()):
            version = str(n.split("_")[0])
            ntmp = str(n.split("_")[1])
            ntmpNoCharge = ntmp.replace("plus", "").replace("minus", "")
            if ntmpNoCharge in minmax.keys():
                zrange = f"::{minmax[ntmpNoCharge]}"
            else:
                zrange = ""

            drawCorrelationPlot(
                histsSF[era][n],
                "muon #eta",
                "muon p_{T} (GeV)",
                f"Data/MC scale factor{zrange}",
                f"muonSF_{n}",
                plotLabel="ForceTitle",
                outdir=outdir,
                smoothPlot=False,
                drawProfileX=False,
                scaleToUnitArea=False,
                draw_both0_noLog1_onlyLog2=1,
                passCanvas=canvas,
                nContours=args.nContours,
                palette=args.palette,
                invertPalette=args.invertPalette,
            )
            # abs. uncertainty (only on nominal, it should be the same for all histograms, hoping the SF and efficiencies were sane in this configuration)
            if "nominal" in n:
                drawCorrelationPlot(
                    histsSF[era][n],
                    "muon #eta",
                    "muon p_{T} (GeV)",
                    f"Abs. uncertainty on SF",
                    f"absUnc_muonSF_{n}",
                    plotLabel="ForceTitle",
                    outdir=outdir + "absoluteStatUncertainty/",
                    smoothPlot=False,
                    drawProfileX=False,
                    scaleToUnitArea=False,
                    draw_both0_noLog1_onlyLog2=1,
                    passCanvas=canvas,
                    plotError=True,
                    nContours=args.nContours,
                    palette=args.palette,
                    invertPalette=args.invertPalette,
                )
                ## rel. uncertainty
                drawCorrelationPlot(
                    histsSF[era][n],
                    "muon #eta",
                    "muon p_{T} (GeV)",
                    f"Rel. uncertainty on SF",
                    f"relUnc_muonSF_{n}",
                    plotLabel="ForceTitle",
                    outdir=outdir + "relativeStatUncertainty/",
                    smoothPlot=False,
                    drawProfileX=False,
                    scaleToUnitArea=False,
                    draw_both0_noLog1_onlyLog2=1,
                    passCanvas=canvas,
                    plotRelativeError=True,
                    nContours=args.nContours,
                    palette=args.palette,
                    invertPalette=args.invertPalette,
                )
                # plot systs
                hnomi = copy.deepcopy(histsSF[era][n].Clone("hnomi"))
                unrolledSF = [
                    unroll2Dto1D(
                        histsSF[era][n], newname=f"unrolled_{n}", cropNegativeBins=False
                    )
                ]
                unrolledLeg = [f"Nominal {ntmp} scale factors"]
                for n2 in list(histsSF[era].keys()):
                    if "nominal" in n2:
                        continue
                    v2 = n2.split("_")[0]
                    n2tmp = n2.split("_")[1]
                    if ntmp != n2tmp:
                        continue
                    hsyst = copy.deepcopy(histsSF[era][n2].Clone(f"hsyst_{n2}"))
                    unrolledSF.append(
                        unroll2Dto1D(
                            hsyst, newname=f"unrolled_{n2}", cropNegativeBins=False
                        )
                    )
                    unrolledLeg.append(f"{v2} syst")
                    hsyst.Add(hnomi, -1)
                    drawCorrelationPlot(
                        hsyst,
                        "muon #eta",
                        "muon p_{T} (GeV)",
                        f"{v2} syst unc. on SF",
                        f"muonSF_{n2}_syst",
                        plotLabel="ForceTitle",
                        outdir=outdir + "absoluteSystUncertainty/",
                        smoothPlot=False,
                        drawProfileX=False,
                        scaleToUnitArea=False,
                        draw_both0_noLog1_onlyLog2=1,
                        passCanvas=canvas,
                        nContours=args.nContours,
                        palette=args.palette,
                        invertPalette=args.invertPalette,
                    )
                    hsyst.Divide(hnomi)
                    drawCorrelationPlot(
                        hsyst,
                        "muon #eta",
                        "muon p_{T} (GeV)",
                        f"{v2} relative syst unc. on SF",
                        f"muonSF_{n2}_systRel",
                        plotLabel="ForceTitle",
                        outdir=outdir + "relativeSystUncertainty/",
                        smoothPlot=False,
                        drawProfileX=False,
                        scaleToUnitArea=False,
                        draw_both0_noLog1_onlyLog2=1,
                        passCanvas=canvas,
                        nContours=args.nContours,
                        palette=args.palette,
                        invertPalette=args.invertPalette,
                    )
                ptBinRanges = []
                for ipt in range(hnomi.GetNbinsY()):
                    ptBinRanges.append(
                        "#splitline{{[{ptmin},{ptmax}]}}{{GeV}}".format(
                            ptmin=int(hnomi.GetYaxis().GetBinLowEdge(ipt + 1)),
                            ptmax=int(hnomi.GetYaxis().GetBinLowEdge(ipt + 2)),
                        )
                    )

                drawNTH1(
                    unrolledSF,
                    unrolledLeg,
                    "Unrolled tag-and-probe eta-p_{T} bin",
                    f"Scale factor",
                    f"scaleFactorAndUncertainty_{ntmp}",
                    outdir + "unrolled/",
                    leftMargin=0.06,
                    rightMargin=0.01,
                    labelRatioTmp="Syst/Nominal",
                    legendCoords="0.2,0.8,0.9,0.98;5",
                    lowerPanelHeight=0.5,
                    skipLumi=True,
                    passCanvas=canvas_unroll,
                    drawVertLines="{a},{b}".format(
                        a=hnomi.GetNbinsY(), b=hnomi.GetNbinsX()
                    ),
                    yAxisExtendConstant=1.4,
                    textForLines=ptBinRanges,
                    transparentLegend=False,
                    drawErrorAll=False,
                    onlyLineColor=True,
                    useLineFirstHistogram=True,
                    setRatioRangeFromHisto=True,
                    setOnlyLineRatio=False,
                    lineWidth=1,
                    ytextOffsetFromTop=0.2,
                    useMultiHistRatioOption=True,
                )

        outdir = outdir_local + productSubfolder + era + "/"

        for n in list(prodHistsSF[era].keys()):
            drawCorrelationPlot(
                prodHistsSF[era][n],
                "muon #eta",
                "muon p_{T} (GeV)",
                "Data/MC scale factor product",
                f"{n}",
                plotLabel="ForceTitle",
                outdir=outdir,
                smoothPlot=False,
                drawProfileX=False,
                scaleToUnitArea=False,
                draw_both0_noLog1_onlyLog2=1,
                passCanvas=canvas,
                nContours=args.nContours,
                palette=args.palette,
                invertPalette=args.invertPalette,
            )
            # plot absolute error
            drawCorrelationPlot(
                prodHistsSF[era][n],
                "muon #eta",
                "muon p_{T} (GeV)",
                "Abs. uncertainty on SF product",
                f"absUnc_{n}",
                plotLabel="ForceTitle",
                outdir=outdir + "absoluteStatUncertainty/",
                smoothPlot=False,
                drawProfileX=False,
                scaleToUnitArea=False,
                draw_both0_noLog1_onlyLog2=1,
                passCanvas=canvas,
                plotError=True,
                nContours=args.nContours,
                palette=args.palette,
                invertPalette=args.invertPalette,
            )
            ## plot relative error
            drawCorrelationPlot(
                prodHistsSF[era][n],
                "muon #eta",
                "muon p_{T} (GeV)",
                "Rel. uncertainty on SF product",
                f"relUnc_{n}",
                plotLabel="ForceTitle",
                outdir=outdir + "relativeStatUncertainty/",
                smoothPlot=False,
                drawProfileX=False,
                scaleToUnitArea=False,
                draw_both0_noLog1_onlyLog2=1,
                passCanvas=canvas,
                plotRelativeError=True,
                nContours=args.nContours,
                palette=args.palette,
                invertPalette=args.invertPalette,
            )

    copyOutputToEos(outdir_local, outdir_original, eoscp=args.eoscp)
