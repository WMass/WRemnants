#!/usr/bin/env python3

# study isolation efficiency and antiisolation one, focusing on stat uncertainty and how to correlated them

import logging
import math

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
    fillTH2fromTH2part,
    safeGetObject,
    safeOpenFile,
)

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = common_plot_parser()
    parser.add_argument("rootfile", type=str, nargs=1)
    parser.add_argument(
        "outdir",
        type=str,
        nargs=1,
        help="Output folder (subfolder 'checkEffStatUncertaintyIso/' created automatically inside)",
    )
    parser.add_argument(
        "-e",
        "--era",
        type=str,
        default="GtoH",
        help="Comma separated list of eras for SF in histogram name; default: %(default)s",
    )
    parser.add_argument(
        "-n",
        "--sfnames",
        type=str,
        default="iso,antiiso",
        help="Comma separated list of SF names inside root file, which will be plotted (trigger uses both plus and minus automatically); default: %(default)s",
    )
    args = parser.parse_args()

    ROOT.TH1.SetDefaultSumw2()

    eras = args.era.split(",")
    fname = args.rootfile[0]
    outdir_original = args.outdir[0]  # to keep it below
    if not outdir_original.endswith("/"):
        outdir_original = outdir_original + "/"
    outdir_original += "checkEffStatUncertaintyIso/"

    outdir_local = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)
    # make folder structure
    for era in eras:
        folder = outdir_local + era + "/"
        createPlotDirAndCopyPhp(folder)

    names = args.sfnames.split(",")
    hists = {
        "SF2D": {e: {n: None for n in names} for e in eras},
        "effData": {e: {n: None for n in names} for e in eras},
        "effMC": {e: {n: None for n in names} for e in eras},
    }

    f = safeOpenFile(fname)
    for htype in hists.keys():
        for n in names:
            for era in eras:
                v = "nominal_" if htype == "SF2D" else ""
                hname = f"{htype}_{v}{n}_{era}_both"
                print(f"{n} -> {hname}")
                hists[htype][era][n] = safeGetObject(f, hname)
                hists[htype][era][n].SetTitle(f"{htype} {era} {n}")
    f.Close()

    canvas = ROOT.TCanvas("canvas", "", 800, 800)

    for era in eras:
        outdir = outdir_original + era + "/"

        effRelUnc = {
            "effData": {"iso": None, "antiiso": None},
            "effMC": {"iso": None, "antiiso": None},
        }

        for n in names:

            rangeRelUnc = "0.0,0.3" if "antiiso" in n else "0.0,0.01"

            for htype in hists.keys():

                plotVar = (
                    "data/MC SF"
                    if htype == "SF2D"
                    else "data efficiency" if htype == "effData" else "MC efficiency"
                )
                drawCorrelationPlot(
                    hists[htype][era][n],
                    "muon #eta",
                    "muon p_{T} (GeV)",
                    f"{n} {plotVar}",
                    f"{htype}_{n}_value",
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
                ## abs. uncertainty
                drawCorrelationPlot(
                    hists[htype][era][n],
                    "muon #eta",
                    "muon p_{T} (GeV)",
                    f"abs. unc. on {n} {plotVar}",
                    f"{htype}_{n}_absUnc",
                    plotLabel="ForceTitle",
                    outdir=outdir,
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
                    hists[htype][era][n],
                    "muon #eta",
                    "muon p_{T} (GeV)",
                    f"rel. unc. on {n} {plotVar}::{rangeRelUnc}",
                    f"{htype}_{n}_relUnc",
                    plotLabel="ForceTitle",
                    outdir=outdir,
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

            # do also ratio of relative uncertainty between data and MC efficiency
            effRelUnc["effData"][n] = copy.deepcopy(
                hists["effData"][era][n].Clone(f"effRelUncData_{n}")
            )
            fillTH2fromTH2part(
                effRelUnc["effData"][n],
                hists["effData"][era][n],
                fillWithError=True,
                useRelativeError=True,
            )
            effRelUnc["effMC"][n] = copy.deepcopy(
                hists["effMC"][era][n].Clone(f"effRelUncMC_{n}")
            )
            fillTH2fromTH2part(
                effRelUnc["effMC"][n],
                hists["effMC"][era][n],
                fillWithError=True,
                useRelativeError=True,
            )
            effRelUncRatio_dataOverMC = copy.deepcopy(
                effRelUnc["effData"][n].Clone("effRelUncRatio_dataOverMC")
            )
            effRelUncRatio_dataOverMC.Divide(effRelUnc["effMC"][n])
            effRelUncRatio_dataOverMC.SetTitle(f"{era} {n}")
            # and plot it
            drawCorrelationPlot(
                effRelUncRatio_dataOverMC,
                "muon #eta",
                "muon p_{T} (GeV)",
                f"ratio of rel.unc. of efficiency (data/MC)::0.5,2.0",
                f"relUncRatio_dataOverMC_{n}",
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

            # further studies
            # define SF by shifting either the data efficiency or the MC one, so to get two independent SF uncertainties
            # these are expected to be uncorrelated, so the actual SF uncertanties obtained propagating both from the original ratio should be the sum in quadrature of these two pieces
            #
            # first, we consider the same a/b ratio but propagating uncertainties from either a or b
            # should be equivalent to the variation of sf' - sf where sf' = (a+da)/b, or similarly shifting b
            dataEff_noErr = copy.deepcopy(
                hists["effData"][era][n].Clone("dataEff_noErr")
            )
            mcEff_noErr = copy.deepcopy(hists["effMC"][era][n].Clone("mcEff_noErr"))
            for ib in range(dataEff_noErr.GetNcells() + 1):
                dataEff_noErr.SetBinError(ib, 0.0)
                mcEff_noErr.SetBinError(ib, 0.0)

            sf_dataEffUncOnly = copy.deepcopy(
                hists["effData"][era][n].Clone(
                    hists["SF2D"][era][n].GetName() + "_dataEffUncOnly"
                )
            )  # clone from effData but using name as SF2D, adding postfix
            sf_dataEffUncOnly.SetTitle(f"{era} {n} SF, only data unc")
            sf_dataEffUncOnly.Divide(mcEff_noErr)
            sf_mcEffUncOnly = copy.deepcopy(
                dataEff_noErr.Clone(hists["SF2D"][era][n].GetName() + "_mcEffUncOnly")
            )  # clone from effData but using name as SF2D, adding postfix
            sf_mcEffUncOnly.Divide(hists["effMC"][era][n])
            sf_mcEffUncOnly.SetTitle(f"{era} {n} SF, only MC unc")
            # now plot only relative uncertainty of sf when only numerator or denominator had a non zero uncertainty
            drawCorrelationPlot(
                sf_dataEffUncOnly,
                "muon #eta",
                "muon p_{T} (GeV)",
                f"rel. unc. on {n} data/MC SF::{rangeRelUnc}",
                f"SF2D_{n}_relUnc_dataEffUncOnly",
                plotLabel="ForceTitle",
                outdir=outdir,
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
            drawCorrelationPlot(
                sf_mcEffUncOnly,
                "muon #eta",
                "muon p_{T} (GeV)",
                f"rel. unc. on {n} data/MC SF::{rangeRelUnc}",
                f"SF2D_{n}_relUnc_mcEffUncOnly",
                plotLabel="ForceTitle",
                outdir=outdir,
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

            # now define really alternative sf shifting efficiency in either data or MC by the uncertainty. For iso or antiiso the shifts must be opposite, since eff(antiiso) = 1 - eff(iso)
            dataEffUnc = copy.deepcopy(hists["effData"][era][n].Clone("dataEffUnc"))
            mcEffUnc = copy.deepcopy(hists["effMC"][era][n].Clone("mcEffUnc"))
            fillTH2fromTH2part(dataEffUnc, hists["effData"][era][n], fillWithError=True)
            fillTH2fromTH2part(mcEffUnc, hists["effMC"][era][n], fillWithError=True)

            dataEff_shift = copy.deepcopy(
                hists["effData"][era][n].Clone(
                    hists["effData"][era][n].GetName() + "_shiftUnc"
                )
            )
            dataEff_shift.Add(dataEffUnc, 1.0 if n == "iso" else -1.0)
            mcEff_shift = copy.deepcopy(
                hists["effMC"][era][n].Clone(
                    hists["effMC"][era][n].GetName() + "_shiftUnc"
                )
            )
            mcEff_shift.Add(
                mcEffUnc, -1.0 if n == "iso" else 1.0
            )  # invert sign of MC eff variation compared to data, so that the sf shifts in the same direction and can be more easily compared (data and MC are uncorrelated anyways)

            sf_dataEffShift = copy.deepcopy(
                dataEff_shift.Clone(
                    hists["SF2D"][era][n].GetName() + "_dataEffShiftOnly"
                )
            )  # clone from effData but using name as SF2D, adding postfix
            sf_dataEffShift.SetTitle(f"{era} {n} SF, only data eff shift")
            sf_dataEffShift.Divide(mcEff_noErr)
            sf_mcEffShift = copy.deepcopy(
                dataEff_noErr.Clone(hists["SF2D"][era][n].GetName() + "_mcEffShiftOnly")
            )  # clone from effData but using name as SF2D, adding postfix
            sf_dataEffShift.SetTitle(f"{era} {n} SF, only MC eff shift")
            sf_mcEffShift.Divide(mcEff_shift)
            # now plot shifted sf
            drawCorrelationPlot(
                sf_dataEffShift,
                "muon #eta",
                "muon p_{T} (GeV)",
                f"{n} scale factor",
                f"SF2D_{n}_value_dataEffShiftOnly",
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
            drawCorrelationPlot(
                sf_mcEffShift,
                "muon #eta",
                "muon p_{T} (GeV)",
                f"{n} scale factor",
                f"SF2D_{n}_value_mcEffShiftOnly",
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
            # and now plot (sf_shift - sf) / sf, which will also show the sign of the variation
            # in order to have same sign for the z axis we multiply by -1.0 when doing antiiso
            sf_dataEffShift.Add(hists["SF2D"][era][n], -1.0)
            sf_dataEffShift.Divide(hists["SF2D"][era][n])
            sf_dataEffShift.SetTitle(f"{era} {n} (#DeltaSF from data eff shift)")
            sf_mcEffShift.Add(hists["SF2D"][era][n], -1.0)
            sf_mcEffShift.Divide(hists["SF2D"][era][n])
            sf_mcEffShift.SetTitle(f"{era} {n} (#DeltaSF from MC eff shift)")
            deltaSF = "#DeltaSF"
            if n == "antiiso":
                deltaSF = "(-1*#DeltaSF)"
                sf_dataEffShift.Scale(-1.0)
                sf_mcEffShift.Scale(-1.0)
            drawCorrelationPlot(
                sf_dataEffShift,
                "muon #eta",
                "muon p_{T} (GeV)",
                f"{n} {deltaSF} / SF::{rangeRelUnc}",
                f"SF2D_{n}_relVariation_dataEffShiftOnly",
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
            drawCorrelationPlot(
                sf_mcEffShift,
                "muon #eta",
                "muon p_{T} (GeV)",
                f"{n} {deltaSF} / SF::{rangeRelUnc}",
                f"SF2D_{n}_relVariation_mcEffShiftOnly",
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

        # check correlations between S and Sbar where S is the iso sf and Sbar is the antiiso SF
        # calling a and b the data and MC efficiencies, and da and db their statistical uncertainties, the elements of the covariance matrix are:
        #
        # c01 = -da*da/(b*(1-b)) - db*db*a*(1-a)/(b*b*(1-b)*(1-b))
        # c00 = [(da/a)**2 + (db/b)**2)]*(a/b)**2
        # c11 = [(da/1-a)**2 + (db/1-b)**2)]* [(1-a)/(1-b)]**2
        # rho = c01/sqrt(c00*c11)
        #
        # doing some math and simplifying some terms at numerator and denominator in rho one finally obtains
        #
        # rho = - [(da/a)*da/(1-a) + (db/b)*db/(1-b)] / sqrt( [(da/a)**2 + (db/b)**2] * [(da/(1-a))**2 + (db/(1-b))**2] )
        #
        # in general rho is identically 1 for each a and b only if either da or db are 0 (so the correlation between efficiencies translates into the one between sf), or trivially if a=1-a and b=1-b
        # in general the correlation might be small, so let's plot it

        # utility definitions
        xd = effRelUnc["effData"]["iso"]
        yd = effRelUnc["effData"]["antiiso"]
        xm = effRelUnc["effMC"]["iso"]
        ym = effRelUnc["effMC"]["antiiso"]
        #
        tmp1 = copy.deepcopy(xd.Clone("tmphist1"))
        tmp2 = copy.deepcopy(xm.Clone("tmphist2"))
        tmp1.Multiply(yd)
        tmp2.Multiply(ym)
        tmp1.Add(tmp2)
        rho = copy.deepcopy(tmp1.Clone("correlationSF_iso_antiiso"))
        rho.Scale(-1.0)

        den = copy.deepcopy(tmp1.Clone("tmpden"))
        den.Reset("ICESM")
        fillTH2fromTH2part(tmp1, effRelUnc["effData"]["iso"])
        tmp1.Multiply(tmp1)
        fillTH2fromTH2part(tmp2, effRelUnc["effMC"]["iso"])
        tmp2.Multiply(tmp2)
        den.Add(tmp1)
        den.Add(tmp2)
        fillTH2fromTH2part(tmp1, effRelUnc["effData"]["antiiso"])
        tmp1.Multiply(tmp1)
        fillTH2fromTH2part(tmp2, effRelUnc["effMC"]["antiiso"])
        tmp2.Multiply(tmp2)
        tmp1.Add(tmp2)
        den.Multiply(tmp1)
        # now sqrt(den)
        for ib in range(1 + den.GetNcells()):
            den.SetBinContent(ib, math.sqrt(den.GetBinContent(ib)))
        rho.Divide(den)
        rho.SetTitle(era)
        drawCorrelationPlot(
            rho,
            "muon #eta",
            "muon p_{T} (GeV)",
            f"Correlation coeff. : SF(iso) vs SF(antiiso)",
            rho.GetName(),
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

    copyOutputToEos(outdir, outdir_original, eoscp=args.eoscp)
