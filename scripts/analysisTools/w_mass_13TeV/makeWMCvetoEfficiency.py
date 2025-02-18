#!/usr/bin/env python3

## make MC efficiency for W vs eta-pt-ut for trigger and isolation
## Requires event yields made with the W  histmaker, with proper event selection
## scripts/histmakers/mw_with_mu_eta_pt_VETOEFFI.py

# example
# python scripts/analysisTools/w_mass_13TeV/makeWMCvetoEfficiency.py /scratch/mciprian/CombineStudies/testZmumuVeto/WMCtruthVetoEffi/mw_with_mu_eta_pt_VETOEFFI_scetlib_dyturboCorr_maxFiles_m1_genPt0_noRecoPtEta.hdf5 scripts/analysisTools/plots/fromMyWremnants/testZmumuVeto/WMCtruthVetoEffi_genPt0_noRecoPtEta/ -v 4 --rebinPt 2

import os
import pickle

## safe batch mode
import sys

import hist
import lz4.frame

import narf
import wums.ioutils
from utilities import common
from wremnants.datasets.datagroups import Datagroups

# from wremnants import plot_tools,theory_tools,syst_tools
from wums import boostHistHelpers as hh
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
    drawCorrelationPlot,
    drawNTH1,
    getMinMaxMultiHisto,
    safeOpenFile,
)

sys.path.append(os.getcwd())


def getBoostEff(n_pass, n_tot, integrateVar=[]):
    s = hist.tag.Slicer()
    num = n_pass.copy()
    den = n_tot.copy()
    if len(integrateVar):
        num = num[{v: s[:: hist.sum] for v in integrateVar}]
        den = den[{v: s[:: hist.sum] for v in integrateVar}]
    eff_boost = hh.divideHists(
        num, den, cutoff=0.1, allowBroadcast=True, createNew=True, cutoff_val=0.0
    )
    vals = eff_boost.values()
    vals[vals < 0.0] = 0.0
    vals[vals > 1.0] = 1.0
    eff_boost.values()[...] = vals
    return eff_boost


def getEtaPtEff(
    n_pass, n_tot, getRoot=False, rootName="", rootTitle="", integrateVar=[]
):
    eff_boost = getBoostEff(n_pass, n_tot, integrateVar=integrateVar)
    if getRoot:
        eff_root = narf.hist_to_root(eff_boost)
        eff_root.SetName(rootName if len(rootName) else "tmp_root_eff")
        eff_root.SetTitle(rootTitle)
        return eff_boost, eff_root
    else:
        return eff_boost, None


if __name__ == "__main__":
    parser = common_plot_parser()
    parser.add_argument(
        "inputfile",
        type=str,
        nargs=1,
        help="Input file with histograms (pkl.lz4 or hdf5 file)",
    )
    parser.add_argument("outdir", type=str, nargs=1, help="Output folder")
    parser.add_argument(
        "-n",
        "--baseName",
        type=str,
        help="Histogram name in the file",
        default="nominal",
    )
    parser.add_argument(
        "-p",
        "--processes",
        default=["Wmunu"],
        nargs="*",
        type=str,
        help="Choose what processes to plot, otherwise all are done",
    )
    parser.add_argument(
        "--rebinPt",
        default=-1,
        type=int,
        help="If positive, rebin yields versus pT by this number before deriving efficiencies (it happens after selecting the pt range)",
    )
    parser.add_argument(
        "--ptRange", default=None, nargs=2, type=float, help="Pt range for plot"
    )
    parser.add_argument(
        "--rr",
        dest="ratioRange",
        default=(0.98, 1.02),
        type=float,
        nargs=2,
        help="Range for ratio plot",
    )
    args = parser.parse_args()

    logger = logging.setup_logger(os.path.basename(__file__), args.verbose)
    fname = args.inputfile[0]
    outdir_original = args.outdir[0]
    outdir = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)

    ROOT.TH1.SetDefaultSumw2()

    canvas = ROOT.TCanvas("canvas", "", 800, 700)
    adjustSettings_CMS_lumi()
    canvas1D = ROOT.TCanvas("canvas1D", "", 800, 900)
    xAxisName = "PostFSR muon #eta"
    yAxisName = "PostFSR muon p_{T} (GeV)"

    ratioRangeStr = f"::{args.ratioRange[0]},{args.ratioRange[1]}"

    groups = Datagroups(fname, mode="w_mass")
    datasets = groups.getNames()
    if args.processes is not None and len(args.processes):
        datasets = list(filter(lambda x: x in args.processes, datasets))
    else:
        datasets = list(filter(lambda x: x != "QCD", datasets))

    logger.debug(f"Using these processes: {datasets}")
    inputHistName = args.baseName
    groups.setNominalName(inputHistName)
    groups.loadHistsForDatagroups(
        inputHistName, syst="", procsToRead=datasets, applySelection=False
    )
    histInfo = groups.getDatagroups()  # keys are same as returned by groups.getNames()
    s = hist.tag.Slicer()
    resultDict = {}
    for d in datasets:
        logger.info(f"Running on process {d}")
        hin = histInfo[d].hists[inputHistName]
        logger.debug(hin.axes)
        h = hin.copy()
        hplot = hin.copy()

        if args.ptRange != None:
            h = h[{"pt": s[complex(0, args.ptRange[0]) : complex(0, args.ptRange[1])]}]
        if args.rebinPt > 0:
            h = h[{"pt": s[:: hist.rebin(args.rebinPt)]}]

        n_veto_pass = h[{"passVeto": True, "charge": s[:: hist.sum]}]
        n_veto_fail = h[{"passVeto": False, "charge": s[:: hist.sum]}]
        n_veto_tot = h[{"passVeto": s[:: hist.sum], "charge": s[:: hist.sum]}]

        n_vetoplus_pass = h[{"passVeto": True, "charge": s[complex(0, 1)]}]
        n_vetoplus_fail = h[{"passVeto": False, "charge": s[complex(0, 1)]}]
        n_vetoplus_tot = h[{"passVeto": s[:: hist.sum], "charge": s[complex(0, 1)]}]

        n_vetominus_pass = h[{"passVeto": True, "charge": s[complex(0, -1)]}]
        n_vetominus_fail = h[{"passVeto": False, "charge": s[complex(0, -1)]}]
        n_vetominus_tot = h[{"passVeto": s[:: hist.sum], "charge": s[complex(0, -1)]}]

        # plot yields
        yields_pass_root = narf.hist_to_root(n_veto_pass)
        yields_pass_root.SetName("yields_veto_pass")
        yields_fail_root = narf.hist_to_root(n_veto_fail)
        yields_fail_root.SetName("yields_veto_fail")
        yields_tot_root = narf.hist_to_root(n_veto_tot)
        yields_tot_root.SetName("yields_veto_tot")

        drawCorrelationPlot(
            yields_pass_root,
            xAxisName,
            yAxisName,
            f"Events passing veto",
            f"{yields_pass_root.GetName()}",
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
            yields_fail_root,
            xAxisName,
            yAxisName,
            f"Events failing veto",
            f"{yields_fail_root.GetName()}",
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
            yields_tot_root,
            xAxisName,
            yAxisName,
            f"Events (denominator)",
            f"{yields_tot_root.GetName()}",
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

        # NOTE: to simplify our lives, when computing efficiencies as ratio of yields n/N and root is used
        # then the uncertainty is obtained using option B for TH1::Divide, which uses binomial uncertainties.
        # However, this implies 0 uncertainty when the efficiency is close to 0 or 1, but we won't really use
        # these uncertainties, since the 2D smoothing with spline doesn't use them.
        # otherwise can use eff = npass/(npass+nfail) and use the formula for standard propagation
        # eff_err = 1./(nTot*nTot) * std::sqrt( nP*nP* e_nF*e_nF + nF*nF * e_nP*e_nP ) with nTot = nP + nF
        # For now we use boost directly, using n/N but assuming uncorrelated n and N, so the uncertainties are maximally wrong (but we might not use them)

        eff_veto_boost2D, eff_veto = getEtaPtEff(
            n_veto_pass,
            n_veto_tot,
            getRoot=True,
            rootName=f"{d}_MC_eff_veto",
            rootTitle="P(veto | gen)",
        )
        drawCorrelationPlot(
            eff_veto,
            xAxisName,
            yAxisName,
            f"MC veto efficiency",
            f"{eff_veto.GetName()}",
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

        eff_vetoplus_boost2D, eff_vetoplus = getEtaPtEff(
            n_vetoplus_pass,
            n_vetoplus_tot,
            getRoot=True,
            rootName=f"{d}_MC_eff_vetoplus",
            rootTitle="P(vetoplus | gen)",
        )
        drawCorrelationPlot(
            eff_vetoplus,
            xAxisName,
            yAxisName,
            f"MC veto efficiency (charge plus)",
            f"{eff_vetoplus.GetName()}",
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

        eff_vetominus_boost2D, eff_vetominus = getEtaPtEff(
            n_vetominus_pass,
            n_vetominus_tot,
            getRoot=True,
            rootName=f"{d}_MC_eff_vetominus",
            rootTitle="P(vetominus | gen)",
        )
        drawCorrelationPlot(
            eff_vetominus,
            xAxisName,
            yAxisName,
            f"MC veto efficiency (charge minus)",
            f"{eff_vetominus.GetName()}",
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

        effRatio_plusOverMinus = copy.deepcopy(
            eff_vetoplus.Clone("effRatio_plusOverMinus")
        )
        effRatio_plusOverMinus.SetTitle("Charge plus / minus")
        effRatio_plusOverMinus.Divide(eff_vetominus)
        drawCorrelationPlot(
            effRatio_plusOverMinus,
            xAxisName,
            yAxisName,
            f"MC veto efficiency ratio",
            f"{effRatio_plusOverMinus.GetName()}",
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

        eff_veto_boost1Deta, eff_veto_eta = getEtaPtEff(
            n_veto_pass,
            n_veto_tot,
            getRoot=True,
            rootName=f"{d}_MC_eff_veto_1Deta",
            rootTitle="P(veto | gen)",
            integrateVar=["pt"],
        )
        eff_vetoplus_boost1Deta, eff_vetoplus_eta = getEtaPtEff(
            n_vetoplus_pass,
            n_vetoplus_tot,
            getRoot=True,
            rootName=f"{d}_MC_eff_vetoplus_1Deta",
            rootTitle="P(vetoplus | gen)",
            integrateVar=["pt"],
        )
        eff_vetominus_boost1Deta, eff_vetominus_eta = getEtaPtEff(
            n_vetominus_pass,
            n_vetominus_tot,
            getRoot=True,
            rootName=f"{d}_MC_eff_vetominus_1Deta",
            rootTitle="P(vetominus | gen)",
            integrateVar=["pt"],
        )
        drawNTH1(
            [eff_veto_eta, eff_vetoplus_eta, eff_vetominus_eta],
            legEntries=["Inclusive", "Charge plus", "Charge minus"],
            labelXtmp=xAxisName,
            labelYtmp="MC veto efficiency",
            canvasName="vetoEfficiency_1Deta",
            outdir=outdir,
            labelRatioTmp=f"x/incl.{ratioRangeStr}",
            topMargin=0.05,
            legendCoords="0.16,0.94,0.84,0.94;3",  # x1,x2,y1,y2
            passCanvas=canvas1D,
            skipLumi=True,
            transparentLegend=True,
            onlyLineColor=True,
            useLineFirstHistogram=True,
            drawErrorAll=True,
            yAxisExtendConstant=1.4,
        )

        eff_veto_boost1Dpt, eff_veto_pt = getEtaPtEff(
            n_veto_pass,
            n_veto_tot,
            getRoot=True,
            rootName=f"{d}_MC_eff_veto_1Dpt",
            rootTitle="P(veto | gen)",
            integrateVar=["eta"],
        )
        eff_vetoplus_boost1Dpt, eff_vetoplus_pt = getEtaPtEff(
            n_vetoplus_pass,
            n_vetoplus_tot,
            getRoot=True,
            rootName=f"{d}_MC_eff_vetoplus_1Dpt",
            rootTitle="P(vetoplus | gen)",
            integrateVar=["eta"],
        )
        eff_vetominus_boost1Dpt, eff_vetominus_pt = getEtaPtEff(
            n_vetominus_pass,
            n_vetominus_tot,
            getRoot=True,
            rootName=f"{d}_MC_eff_vetominus_1Dpt",
            rootTitle="P(vetominus | gen)",
            integrateVar=["eta"],
        )
        minEffi, _ = getMinMaxMultiHisto(
            [eff_vetoplus_pt, eff_vetominus_pt, eff_veto_pt]
        )
        minPtForRange = min(0.992, minEffi)
        yRangeEff = f"::{minPtForRange},1.002"
        drawNTH1(
            [eff_veto_pt, eff_vetoplus_pt, eff_vetominus_pt],
            legEntries=["Inclusive", "Charge plus", "Charge minus"],
            labelXtmp=yAxisName,
            labelYtmp="MC veto efficiency",
            canvasName="vetoEfficiency_1Dpt",
            outdir=outdir,
            labelRatioTmp=f"x/incl.{ratioRangeStr}",
            topMargin=0.05,
            legendCoords="0.16,0.94,0.84,0.94;3",  # x1,x2,y1,y2
            passCanvas=canvas1D,
            skipLumi=True,
            transparentLegend=True,
            onlyLineColor=True,
            useLineFirstHistogram=True,
            drawErrorAll=True,
            yAxisExtendConstant=1.4,
        )

        resultDict[f"{d}_MC_eff_veto_etapt"] = eff_veto_boost2D
        resultDict[f"{d}_MC_eff_vetoplus_etapt"] = eff_vetoplus_boost2D
        resultDict[f"{d}_MC_eff_vetominus_etapt"] = eff_vetominus_boost2D

    postfix = ""
    # toAppend = []
    # postfix = "_".join(toAppend)
    if len(postfix):
        postfix = "_" + postfix

    resultDict.update(
        {"meta_info": wums.ioutils.make_meta_info_dict(args=args, wd=common.base_dir)}
    )

    outfile = outdir + f"vetoEfficienciesEtaPt{postfix}.pkl.lz4"
    logger.info(f"Going to store 2D histograms {resultDict.keys()} in file {outfile}")
    with lz4.frame.open(outfile, "wb") as f:
        pickle.dump(resultDict, f, protocol=pickle.HIGHEST_PROTOCOL)

    outfileRoot = outdir + f"vetoEfficiencies2D{postfix}.root"
    logger.info(f"Going to store 2D root histograms in file {outfileRoot}")
    rf = safeOpenFile(outfileRoot, mode="RECREATE")
    eff_veto.Write()
    eff_vetoplus.Write()
    eff_vetominus.Write()
    rf.Close()
    copyOutputToEos(outdir, outdir_original, eoscp=args.eoscp)
