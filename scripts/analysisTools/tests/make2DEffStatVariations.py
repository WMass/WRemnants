#!/usr/bin/env python3

import argparse
import os
## safe batch mode
import sys

args = sys.argv[:]
sys.argv = ["-b"]
import ROOT

sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

sys.path.append(os.getcwd() + "/plotUtils/")
from utility import (copy, createPlotDirAndCopyPhp, drawCorrelationPlot,
                     fillTH3binFromTH2, np, safeGetObject, safeOpenFile)


def effStatVariations(
    outdir,
    covHisto,
    parHisto,
    nbins_pt,
    ptmin,
    ptmax,
    smoothFunction="pol3",
    suffix=None,
    skipPlot=False,
    getDiff=True,
    palette=87,
):

    lepton = "Muon"
    # defined in plotUtils/utility.h
    createPlotDirAndCopyPhp(outdir)

    nbins_eta = parHisto.GetNbinsX()
    etamin = round(parHisto.GetXaxis().GetBinLowEdge(1), 2)
    etamax = round(parHisto.GetXaxis().GetBinLowEdge(1 + nbins_eta), 2)
    # how many parameters the interpolation function has might depends on the eta bins, let's use the Y dimension of parHisto for now
    systHistos = []
    npars = parHisto.GetNbinsY()
    nomiHisto = ROOT.TH2D(
        f"nominal", "Nominal", nbins_eta, etamin, etamax, nbins_pt, ptmin, ptmax
    )
    for ip in range(npars):
        systHistos.append(
            ROOT.TH2D(
                f"p{ip}",
                "Variation for parameter {ip}",
                nbins_eta,
                etamin,
                etamax,
                nbins_pt,
                ptmin,
                ptmax,
            )
        )

    systCalc = ROOT.wrem.EtaPtCorrelatedEfficiency(covHisto, parHisto, ptmin, ptmax)
    systCalc.setSmoothingFunction(smoothFunction)

    # one way to do it
    tf1_func_nomi = ROOT.TF1(f"nomi_{smoothFunction}", smoothFunction, ptmin, ptmax)
    tf1_func_var = ROOT.TF1(f"var_{smoothFunction}", smoothFunction, ptmin, ptmax)
    for ieta in range(nbins_eta):
        tf1_func_nomi.SetParameters(
            np.array(
                [parHisto.GetBinContent(ieta + 1, ip + 1) for ip in range(npars)],
                dtype=np.dtype("d"),
            )
        )
        vec = ROOT.std.vector["double"]()
        vec = systCalc.DoEffSyst(ieta + 1)
        for ivar in range(npars):
            startIndex = npars * ivar
            # set parameters for a given hessian
            tf1_func_var.SetParameters(
                np.array(
                    [vec[i] for i in range(startIndex, startIndex + npars)],
                    dtype=np.dtype("d"),
                )
            )
            # now loop on pt bins to fill the histogram from the function and its variations
            for ipt in range(nbins_pt):
                pt = systHistos[0].GetYaxis().GetBinCenter(ipt + 1)
                if ivar:
                    nomi = nomiHisto.GetBinContent(ieta + 1, ipt + 1)
                else:
                    nomi = tf1_func_nomi.Eval(pt)
                    nomiHisto.SetBinContent(ieta + 1, ipt + 1, nomi)
                syst = tf1_func_var.Eval(pt)
                if getDiff:
                    syst -= nomi
                systHistos[ivar].SetBinContent(ieta + 1, ipt + 1, syst)

    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPalette(87)
    canv = ROOT.TCanvas("effStatVar_canv", "", 900, 700)

    xaxisTitle = f"{lepton} #eta"
    yaxisTitle = f"{lepton} p_{{T}} (GeV)"
    zaxisTitle = "Alternate - nominal" if getDiff else "Alternate"
    postfix = f"_{suffix}" if suffix else ""
    canvasName = f"effStatVar_{smoothFunction}_nominal{postfix}"

    allHist = ROOT.TH3D(
        f"interpolationAndVars",
        "Nominal in first Z bin",
        nbins_eta,
        etamin,
        etamax,
        nbins_pt,
        ptmin,
        ptmax,
        npars + 1,
        -0.5,
        0.5 + npars,
    )
    fillTH3binFromTH2(allHist, nomiHisto, 1)
    if not skipPlot:
        drawCorrelationPlot(
            nomiHisto,
            xaxisTitle,
            yaxisTitle,
            "Nominal",
            canvasName,
            "ForceTitle",
            outdir,
            1,
            1,
            False,
            False,
            False,
            1,
            leftMargin=0.14,
            rightMargin=0.22,
            palette=palette,
            passCanvas=canv,
        )

    for ivar in range(npars):
        if not skipPlot:
            systHistos[ivar].SetTitle(f"Nuisance p{ivar} for {smoothFunction}")
            canvasName = f"effStatVar_{smoothFunction}_p{ivar}{postfix}"
            drawCorrelationPlot(
                systHistos[ivar],
                xaxisTitle,
                yaxisTitle,
                zaxisTitle,
                canvasName,
                "ForceTitle",
                outdir,
                1,
                1,
                False,
                False,
                False,
                1,
                leftMargin=0.14,
                rightMargin=0.22,
                palette=palette,
                passCanvas=canv,
            )
        # now if the histograms have the variations, sum the nominal before storing in the final histogram
        # so that the final histogram will have nominal scale factors and alternate scale factors directly
        sh = copy.deepcopy(systHistos[ivar].Clone(f"{systHistos[ivar].GetName()}_tmp"))
        if getDiff:
            sh.Add(nomiHisto)
        fillTH3binFromTH2(allHist, sh, 2 + ivar)

    return allHist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "covariancefile", type=str, nargs=1, help="File with covariance matrix inside"
    )
    parser.add_argument(
        "outdir", type=str, nargs=1, help="Output folder to save things"
    )
    parser.add_argument(
        "-n",
        "--outfilename",
        type=str,
        help="Name for output file with efficiency variations",
    )
    parser.add_argument(
        "-c",
        "--covTH3",
        dest="covariance",
        type=str,
        default="hist_FuncCovMatrix_vs_eta_sf",
        help="TH3D histogram containing eta as x axis and NxN cov matrix as other dimensions",
    )
    parser.add_argument(
        "-p",
        "--parTH2",
        dest="parameters",
        type=str,
        default="hist_FuncParam_vs_eta_sf",
        help="TH2D histogram containing eta as x-axis and N parameters of fit function as y-axis",
    )
    parser.add_argument(
        "-s",
        "--suffix",
        type=str,
        default=None,
        help="suffix for the ROOT file and plots",
    )
    parser.add_argument(
        "--ptbins",
        default=None,
        nargs=3,
        type=str,
        required=True,
        help="Pass npt, ptmin, ptmax, for the smoothed histogram",
    )
    parser.add_argument(
        "-f",
        "--smoothFunction",
        type=str,
        default="pol3",
        choices=["pol3", "pol2", "erf"],
        help="Smoothing function",
    )
    parser.add_argument(
        "--plotDiff",
        action="store_true",
        help="Plot variations of the interpolation functions (alt-nomi), instead of the alternate function itself (but note that what is saved in the output file is always the alternate regardless)",
    )
    args = parser.parse_args()

    outdir = args.outdir[0]
    baseOutFileName = (
        "effStatVariations.root" if not args.outfilename else args.outfilename
    )
    outfilename = outdir + "/" + baseOutFileName
    if args.suffix:
        outfilename = outfilename.replace(".root", "_%s.root" % args.suffix)
    # defined in plotUtils/utility.h
    createPlotDirAndCopyPhp(outdir)

    tf = safeOpenFile(args.covariancefile[0])
    covHisto = safeGetObject(tf, args.covariance)
    parHisto = safeGetObject(tf, args.parameters)
    tf.Close()

    nbins_pt = int(args.ptbins[0])
    ptmin = float(args.ptbins[1])
    ptmax = float(args.ptbins[2])

    hist3 = effStatVariations(
        outdir,
        covHisto,
        parHisto,
        nbins_pt,
        ptmin,
        ptmax,
        smoothFunction=args.smoothFunction,
        suffix=args.suffix,
        getDiff=args.plotDiff,
        palette=87,
    )

    outf = safeOpenFile(outfilename, mode="recreate")
    outf.cd()
    # for ivar in range(npars):
    #    systHistos[ivar].Write()
    hist3.Write()
    outf.Close()
    print()
    print(f"Saving histograms in file {outfilename}")
    print()
