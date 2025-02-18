#!/usr/bin/env python3

## make study using a histogram created specifically for QCD MC (but in principle it can be used for other processes too if the histograms exist
## axes should be these ones ('eta', 'pt', 'charge', 'mt', 'passIso', 'NjetsClean', 'leadjetPt', 'DphiMuonMet')

# example
# python scripts/analysisTools/w_mass_13TeV/makeQCDMCstudy.py /scratch/mciprian/CombineStudies/TRASHTEST/mw_with_mu_eta_pt_scetlib_dyturboCorr_maxFiles_m1_extMC_noSF.hdf5 scripts/analysisTools/plots/fromMyWremnants/testFakes_studies/  -v 4

import os

## safe batch mode
import sys

import hist

import narf
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

from scripts.analysisTools.plotUtils.utility import (
    adjustSettings_CMS_lumi,
    common_plot_parser,
    copyOutputToEos,
    createPlotDirAndCopyPhp,
    drawCorrelationPlot,
    drawNTH1,
    drawTH1,
    getMinMaxHisto,
)

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
        help="Histogram name in the file (it depends on what study you run)",
        default="otherStudyForFakes",
    )
    parser.add_argument(
        "-p",
        "--processes",
        default=["QCD"],
        nargs="*",
        type=str,
        help="Choose what processes to plot, otherwise all are done",
    )
    parser.add_argument(
        "-r", "--runStudy", type=int, default=0, choices=range(5), help="What to run"
    )
    args = parser.parse_args()

    logger = logging.setup_logger(os.path.basename(__file__), args.verbose)

    ###############################################################################################

    def runKinematics(args):

        fname = args.inputfile[0]
        outdir_original = args.outdir[0]
        outdir = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)

        ROOT.TH1.SetDefaultSumw2()

        canvas = ROOT.TCanvas("canvas", "", 800, 700)
        adjustSettings_CMS_lumi()
        canvas1D = ROOT.TCanvas("canvas1D", "", 800, 900)

        groups = Datagroups(fname, mode="w_mass")
        datasets = groups.getNames()
        if args.processes is not None and len(args.processes):
            datasets = list(filter(lambda x: x in args.processes, datasets))
        else:
            datasets = list(filter(lambda x: x == "QCD", datasets))

        logger.debug(f"Using these processes: {datasets}")
        inputHistName = args.baseName
        groups.setNominalName(inputHistName)
        groups.loadHistsForDatagroups(
            inputHistName, syst="", procsToRead=datasets, applySelection=False
        )
        histInfo = (
            groups.getDatagroups()
        )  # keys are same as returned by groups.getNames()
        s = hist.tag.Slicer()
        for d in datasets:
            logger.info(f"Running on process {d}")
            hin = histInfo[d].hists[inputHistName]
            logger.debug(hin.axes)
            # ('eta', 'pt', 'charge', 'mt', 'passIso', 'NjetsClean', 'leadjetPt', 'DphiMuonMet')
            hin = hin[{"charge": s[:: hist.sum]}]  # integrate charges for now
            # make eta inot abseta for now
            makeAbsEta = True  # make an option for this
            etaAxisName = "eta"
            if makeAbsEta:
                hin = hh.makeAbsHist(hin, "eta")
                etaAxisName = "abseta"
            hpass = hin[{"passIso": True}]
            hfail = hin[{"passIso": False}]
            htest = hin.copy()

            h_njets_mt = hin[
                {
                    "pt": s[:: hist.sum],
                    "leadjetPt": s[:: hist.sum],
                    "DphiMuonMet": s[:: hist.sum],
                    etaAxisName: s[:: hist.sum],
                }
            ]
            h_njets_mt_pass = h_njets_mt[{"passIso": True}]
            h_njets_mt_fail = h_njets_mt[{"passIso": False}]

            hroot_njets_mt_pass = narf.hist_to_root(h_njets_mt_pass)
            hroot_njets_mt_pass.SetName("njets_mt_pass")
            hroot_njets_mt_pass.SetTitle("Pass isolation")
            drawCorrelationPlot(
                hroot_njets_mt_pass,
                "DeepMET m_{T} (GeV)",
                "Number of jets (p_{T} > 15 GeV)",
                f"Events",
                f"{hroot_njets_mt_pass.GetName()}",
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

            hroot_njets_mt_fail = narf.hist_to_root(h_njets_mt_fail)
            hroot_njets_mt_fail.SetName("njets_mt_fail")
            hroot_njets_mt_fail.SetTitle("Fail isolation")
            drawCorrelationPlot(
                hroot_njets_mt_fail,
                "DeepMET m_{T} (GeV)",
                "Number of jets (p_{T} > 15 GeV)",
                f"Events",
                f"{hroot_njets_mt_fail.GetName()}",
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

            hroot_njets_mt_FRF = copy.deepcopy(
                hroot_njets_mt_pass.Clone("njets_mt_FRF")
            )
            hroot_njets_mt_FRF.Divide(hroot_njets_mt_fail)
            hroot_njets_mt_FRF.SetTitle("")
            drawCorrelationPlot(
                hroot_njets_mt_FRF,
                "DeepMET m_{T} (GeV)",
                "Number of jets (p_{T} > 15 GeV)",
                f"Fakerate factor::0,1.5",
                f"{hroot_njets_mt_FRF.GetName()}",
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

            h_njets_mt_ptLow_pass = hpass[
                {
                    "pt": s[0 : 1 : hist.sum],
                    "leadjetPt": s[:: hist.sum],
                    "DphiMuonMet": s[:: hist.sum],
                    etaAxisName: s[:: hist.sum],
                }
            ]
            h_njets_mt_ptLow_fail = hfail[
                {
                    "pt": s[0 : 1 : hist.sum],
                    "leadjetPt": s[:: hist.sum],
                    "DphiMuonMet": s[:: hist.sum],
                    etaAxisName: s[:: hist.sum],
                }
            ]
            h_njets_mt_ptHigh_pass = hpass[
                {
                    "pt": s[2 :: hist.sum],
                    "leadjetPt": s[:: hist.sum],
                    "DphiMuonMet": s[:: hist.sum],
                    etaAxisName: s[:: hist.sum],
                }
            ]
            h_njets_mt_ptHigh_fail = hfail[
                {
                    "pt": s[2 :: hist.sum],
                    "leadjetPt": s[:: hist.sum],
                    "DphiMuonMet": s[:: hist.sum],
                    etaAxisName: s[:: hist.sum],
                }
            ]

            hroot_njets_mt_ptLow_pass = narf.hist_to_root(h_njets_mt_ptLow_pass)
            hroot_njets_mt_ptLow_pass.SetName("njets_mt_ptLow_pass")
            ptLow_leftEdge = int(hin.axes["pt"].edges[0])
            ptLow_rightEdge = int(hin.axes["pt"].edges[1])
            ptLowRange_text = f"{ptLow_leftEdge} < p_{{T}} < {ptLow_rightEdge} GeV"
            hroot_njets_mt_ptLow_pass.SetTitle(f"Pass isolation, {ptLowRange_text}")
            drawCorrelationPlot(
                hroot_njets_mt_ptLow_pass,
                "DeepMET m_{T} (GeV)",
                "Number of jets (p_{T} > 15 GeV)",
                f"Events",
                f"{hroot_njets_mt_ptLow_pass.GetName()}",
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

            hroot_njets_mt_ptLow_fail = narf.hist_to_root(h_njets_mt_ptLow_fail)
            hroot_njets_mt_ptLow_fail.SetName("njets_mt_ptLow_fail")
            hroot_njets_mt_ptLow_fail.SetTitle(f"Fail isolation, {ptLowRange_text}")
            drawCorrelationPlot(
                hroot_njets_mt_ptLow_fail,
                "DeepMET m_{T} (GeV)",
                "Number of jets (p_{T} > 15 GeV)",
                f"Events",
                f"{hroot_njets_mt_ptLow_fail.GetName()}",
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

            hroot_njets_mt_ptLow_FRF = copy.deepcopy(
                hroot_njets_mt_ptLow_pass.Clone("njets_mt_ptLow_FRF")
            )
            hroot_njets_mt_ptLow_FRF.Divide(hroot_njets_mt_ptLow_fail)
            hroot_njets_mt_ptLow_FRF.SetTitle(f"{ptLowRange_text}")
            drawCorrelationPlot(
                hroot_njets_mt_ptLow_FRF,
                "DeepMET m_{T} (GeV)",
                "Number of jets (p_{T} > 15 GeV)",
                f"Fakerate factor::0,1.5",
                f"{hroot_njets_mt_ptLow_FRF.GetName()}",
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

            ### now pt high
            hroot_njets_mt_ptHigh_pass = narf.hist_to_root(h_njets_mt_ptHigh_pass)
            hroot_njets_mt_ptHigh_pass.SetName("njets_mt_ptHigh_pass")
            ptHigh_leftEdge = int(hin.axes["pt"].edges[2])
            ptHigh_rightEdge = int(hin.axes["pt"].edges[-1])
            ptHighRange_text = f"{ptHigh_leftEdge} < p_{{T}} < {ptHigh_rightEdge} GeV"
            hroot_njets_mt_ptHigh_pass.SetTitle(f"Pass isolation, {ptHighRange_text}")
            drawCorrelationPlot(
                hroot_njets_mt_ptHigh_pass,
                "DeepMET m_{T} (GeV)",
                "Number of jets (p_{T} > 15 GeV)",
                f"Events",
                f"{hroot_njets_mt_ptHigh_pass.GetName()}",
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

            hroot_njets_mt_ptHigh_fail = narf.hist_to_root(h_njets_mt_ptHigh_fail)
            hroot_njets_mt_ptHigh_fail.SetName("njets_mt_ptHigh_fail")
            hroot_njets_mt_ptHigh_fail.SetTitle(f"Fail isolation, {ptHighRange_text}")
            drawCorrelationPlot(
                hroot_njets_mt_ptHigh_fail,
                "DeepMET m_{T} (GeV)",
                "Number of jets (p_{T} > 15 GeV)",
                f"Events",
                f"{hroot_njets_mt_ptHigh_fail.GetName()}",
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

            hroot_njets_mt_ptHigh_FRF = copy.deepcopy(
                hroot_njets_mt_ptHigh_pass.Clone("njets_mt_ptHigh_FRF")
            )
            hroot_njets_mt_ptHigh_FRF.Divide(hroot_njets_mt_ptHigh_fail)
            hroot_njets_mt_ptHigh_FRF.SetTitle(f"{ptHighRange_text}")
            drawCorrelationPlot(
                hroot_njets_mt_ptHigh_FRF,
                "DeepMET m_{T} (GeV)",
                "Number of jets (p_{T} > 15 GeV)",
                f"Fakerate factor::0,1.5",
                f"{hroot_njets_mt_ptHigh_FRF.GetName()}",
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

            outdirTest = outdir + "/testShapes/"
            createPlotDirAndCopyPhp(outdirTest)

            newMtEdges = [i * 5 for i in range(12)] + [60, 80, 120]
            htest = hh.rebinHist(htest, "mt", newMtEdges)

            ptEdges = htest.axes["pt"].edges
            newPtEdges = [ptEdges[0], ptEdges[1], *ptEdges[2::3]]
            logger.warning(f"For test with NjetsClean, rebinning pt to {newPtEdges}")
            htest = hh.rebinHist(htest, "pt", newPtEdges)

            etaEdges = htest.axes[etaAxisName].edges
            ptEdges = htest.axes["pt"].edges
            etaLatex = "|#eta|" if makeAbsEta else "#eta"
            for ieta in range(htest.axes[etaAxisName].size):
                for ipt in range(htest.axes["pt"].size):
                    etaRangeText = f"{round(etaEdges[ieta],1)} < {etaLatex} < {round(etaEdges[ieta+1],1)}"
                    ptRangeText = f"{ptEdges[ipt]} < p_{{T}} < {ptEdges[ipt+1]} GeV"
                    htestReduced = htest[
                        {
                            "pt": s[ipt : ipt + 1 : hist.sum],
                            etaAxisName: s[ieta : ieta + 1 : hist.sum],
                            "leadjetPt": s[:: hist.sum],
                            "DphiMuonMet": s[:: hist.sum],
                        }
                    ]

                    h_0jet_pass = htestReduced[
                        {
                            "NjetsClean": s[0],
                            "passIso": True,
                        }
                    ]
                    h_1orMoreJet_pass = htestReduced[
                        {
                            "NjetsClean": s[1 :: hist.sum],
                            "passIso": True,
                        }
                    ]
                    h_2orMoreJet_pass = htestReduced[
                        {
                            "NjetsClean": s[2 :: hist.sum],
                            "passIso": True,
                        }
                    ]
                    h_0jet_fail = htestReduced[
                        {
                            "NjetsClean": s[0],
                            "passIso": False,
                        }
                    ]
                    h_1orMoreJet_fail = htestReduced[
                        {
                            "NjetsClean": s[1 :: hist.sum],
                            "passIso": False,
                        }
                    ]
                    h_2orMoreJet_fail = htestReduced[
                        {
                            "NjetsClean": s[2 :: hist.sum],
                            "passIso": False,
                        }
                    ]

                    h_0jet_FRF = hh.divideHists(h_0jet_pass, h_0jet_fail)
                    hroot_0jet_FRF = narf.hist_to_root(h_0jet_FRF)
                    hroot_0jet_FRF.SetName(f"h_0jet_FRF_ipt{ipt}_ieta{ieta}")

                    h_1orMoreJet_FRF = hh.divideHists(
                        h_1orMoreJet_pass, h_1orMoreJet_fail
                    )
                    hroot_1orMoreJet_FRF = narf.hist_to_root(h_1orMoreJet_FRF)
                    hroot_1orMoreJet_FRF.SetName(
                        f"h_1orMoreJet_FRF_ipt{ipt}_ieta{ieta}"
                    )

                    h_2orMoreJet_FRF = hh.divideHists(
                        h_2orMoreJet_pass, h_2orMoreJet_fail
                    )
                    hroot_2orMoreJet_FRF = narf.hist_to_root(h_2orMoreJet_FRF)
                    hroot_2orMoreJet_FRF.SetName(
                        f"h_2orMoreJet_FRF_ipt{ipt}_ieta{ieta}"
                    )

                    drawNTH1(
                        [hroot_0jet_FRF, hroot_1orMoreJet_FRF, hroot_2orMoreJet_FRF],
                        ["0 jets", ">= 1 jets", ">= 2 jets"],
                        "DeepMet m_{T} (GeV)",
                        "Fakerate factor",
                        f"FRFvsNjetsClean_ipt{ipt}_ieta{ieta}",
                        outdirTest,
                        topMargin=0.05,
                        leftMargin=0.16,
                        rightMargin=0.05,
                        labelRatioTmp=">= N-j / 0-jet::0.5,1.5",
                        legendCoords="0.7,0.94,0.76,0.94;1",
                        transparentLegend=True,
                        lowerPanelHeight=0.4,
                        skipLumi=True,
                        passCanvas=canvas1D,
                        noErrorRatioDen=False,
                        drawErrorAll=True,
                        onlyLineColor=True,
                        useLineFirstHistogram=True,
                        setOnlyLineRatio=True,
                        lineWidth=2,
                        moreTextLatex=f"{etaRangeText}      {ptRangeText}::0.18,0.97,0.05,0.035",
                    )

        copyOutputToEos(outdir, outdir_original, eoscp=args.eoscp)

    ###############################################################################################

    def runGenMatching(args):

        fname = args.inputfile[0]
        outdir_original = args.outdir[0] + "/genMatchStudy/"
        outdir = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)

        ROOT.TH1.SetDefaultSumw2()

        canvas = ROOT.TCanvas("canvas", "", 800, 700)
        adjustSettings_CMS_lumi()
        canvas1D = ROOT.TCanvas("canvas1D", "", 800, 900)

        groups = Datagroups(fname, mode="w_mass")
        datasets = groups.getNames()
        if args.processes is not None and len(args.processes):
            datasets = list(filter(lambda x: x in args.processes, datasets))
        else:
            datasets = list(filter(lambda x: x == "QCD", datasets))

        logger.debug(f"Using these processes: {datasets}")
        inputHistName = args.baseName
        groups.setNominalName(inputHistName)
        groups.loadHistsForDatagroups(
            inputHistName, syst="", procsToRead=datasets, applySelection=False
        )
        histInfo = (
            groups.getDatagroups()
        )  # keys are same as returned by groups.getNames()
        s = hist.tag.Slicer()

        for d in datasets:
            logger.info(f"Running on process {d}")
            hin = histInfo[d].hists[inputHistName]
            logger.debug(hin.axes)

            makeAbsEta = True  # make an option for this
            etaAxisName = "eta"
            etaAxisTitle = "Reco muon #eta"
            if makeAbsEta:
                hin = hh.makeAbsHist(hin, "eta")
                etaAxisName = "abseta"
                etaAxisTitle = "Reco muon |#eta|"

            hin = hin[{etaAxisName: s[:: hist.rebin(4)], "pt": s[:: hist.rebin(4)]}]
            hpass = hin[{"hasMatch": True}]
            htotal = hin[{"hasMatch": s[:: hist.sum]}]

            hfrac = hh.divideHists(hpass, htotal)
            hfrac_root = narf.hist_to_root(hfrac)
            hfrac_root.SetName(f"hfrac_{inputHistName}")
            hfrac_root.SetTitle(inputHistName)

            drawCorrelationPlot(
                hfrac_root,
                etaAxisTitle,
                "Reco muon p_{T} (GeV)",
                f"Fraction of events with gen match",
                f"{hfrac_root.GetName()}",
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

    ###############################################################################################

    def runGenMt(args):

        useMet = False
        if "Met" in args.baseName:
            useMet = True

        fname = args.inputfile[0]
        outdir_original = args.outdir[0] + (
            "/genMetStudy/" if useMet else "/genMtStudy/"
        )
        outdir = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)

        ROOT.TH1.SetDefaultSumw2()

        canvas = ROOT.TCanvas("canvas", "", 800, 700)
        adjustSettings_CMS_lumi()
        canvas1D = ROOT.TCanvas("canvas1D", "", 800, 900)

        groups = Datagroups(fname, mode="w_mass")
        datasets = groups.getNames()
        if args.processes is not None and len(args.processes):
            datasets = list(filter(lambda x: x in args.processes, datasets))
        else:
            datasets = list(filter(lambda x: x == "QCD", datasets))

        logger.debug(f"Using these processes: {datasets}")
        inputHistName = args.baseName
        groups.setNominalName(inputHistName)
        groups.loadHistsForDatagroups(
            inputHistName, syst="", procsToRead=datasets, applySelection=False
        )
        histInfo = (
            groups.getDatagroups()
        )  # keys are same as returned by groups.getNames()
        s = hist.tag.Slicer()

        for d in datasets:
            logger.info(f"Running on process {d}")
            hin = histInfo[d].hists[inputHistName]
            logger.debug(hin.axes)

            makeAbsEta = True  # make an option for this
            etaAxisName = "eta"
            etaAxisTitle = "Reco muon #eta"
            if makeAbsEta:
                hin = hh.makeAbsHist(hin, "eta")
                etaAxisName = "abseta"
                etaAxisTitle = "Reco muon |#eta|"

            hin = hin[
                {
                    etaAxisName: s[:: hist.rebin(8)],
                    "pt": s[:: hist.rebin(8)],
                    "recoMet" if useMet else "mt": s[:: hist.rebin(5)],
                    "genMet" if useMet else "genMt": s[:: hist.rebin(5)],
                }
            ]

            hpass = hin[{"passIso": True}]
            hfail = hin[{"passIso": False}]
            htotal = hin[{"passIso": s[:: hist.sum]}]

            etaEdges = hin.axes[etaAxisName].edges
            ptEdges = hin.axes["pt"].edges
            etaLatex = "|#eta|" if makeAbsEta else "#eta"

            xAxisName = "Reco m_{T} (GeV)"
            yAxisName = "Gen m_{T} with reco muon (GeV)"
            if useMet:
                xAxisName = "Reco MET (GeV)"
                yAxisName = "Gen MET (GeV)"

            for ieta in range(hin.axes[etaAxisName].size):
                for ipt in range(hin.axes["pt"].size):
                    etaRangeText = f"{round(etaEdges[ieta],1)} < {etaLatex} < {round(etaEdges[ieta+1],1)}"
                    ptRangeText = f"{ptEdges[ipt]} < p_{{T}} < {ptEdges[ipt+1]} GeV"

                    hp = hpass[{"pt": s[ipt], etaAxisName: s[ieta]}]
                    hf = hfail[{"pt": s[ipt], etaAxisName: s[ieta]}]
                    ht = htotal[{"pt": s[ipt], etaAxisName: s[ieta]}]

                    hpr = narf.hist_to_root(hp)
                    hpr.SetName(f"mtVsGenMt_passIso_ipt{ipt}_ieta{ieta}")
                    hpr.SetTitle(f"passIso {etaRangeText}     {ptRangeText}")
                    drawCorrelationPlot(
                        hpr,
                        xAxisName,
                        yAxisName,
                        f"Events",
                        f"{hpr.GetName()}",
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

                    hfr = narf.hist_to_root(hf)
                    hfr.SetName(f"mtVsGenMt_failIso_ipt{ipt}_ieta{ieta}")
                    hfr.SetTitle(f"failIso {etaRangeText}     {ptRangeText}")
                    drawCorrelationPlot(
                        hfr,
                        xAxisName,
                        yAxisName,
                        f"Events",
                        f"{hfr.GetName()}",
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

                    htr = narf.hist_to_root(ht)
                    htr.SetName(f"mtVsGenMt_anyIso_ipt{ipt}_ieta{ieta}")
                    htr.SetTitle(f"AnyIso {etaRangeText}     {ptRangeText}")
                    drawCorrelationPlot(
                        htr,
                        xAxisName,
                        yAxisName,
                        f"Events",
                        f"{htr.GetName()}",
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

    ###############################################################################################

    def runJetMuon(args):

        useMet = False
        if "Met" in args.baseName:
            useMet = True

        fname = args.inputfile[0]
        outdir_original = args.outdir[0] + "/jetAndMuon/"
        outdir = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)

        ROOT.TH1.SetDefaultSumw2()

        canvas = ROOT.TCanvas("canvas", "", 800, 700)
        adjustSettings_CMS_lumi()
        canvas1D = ROOT.TCanvas("canvas1D", "", 800, 900)

        groups = Datagroups(fname, mode="w_mass")
        datasets = groups.getNames()
        if args.processes is not None and len(args.processes):
            datasets = list(filter(lambda x: x in args.processes, datasets))
        else:
            datasets = list(filter(lambda x: x == "QCD", datasets))

        logger.debug(f"Using these processes: {datasets}")
        inputHistName = args.baseName
        groups.setNominalName(inputHistName)
        groups.loadHistsForDatagroups(
            inputHistName, syst="", procsToRead=datasets, applySelection=False
        )
        histInfo = (
            groups.getDatagroups()
        )  # keys are same as returned by groups.getNames()
        s = hist.tag.Slicer()

        for d in datasets:
            logger.info(f"Running on process {d}")
            hin = histInfo[d].hists[inputHistName]
            logger.debug(hin.axes)

            hp = hin[{"passIso": True}]
            hf = hin[{"passIso": False}]
            ht = hin[{"passIso": s[:: hist.sum]}]

            xAxisName = "Muon p_{T} (GeV)"
            yAxisName = "Jet p_{T} with #DeltaR(jet,muon)<0.4 (GeV)"

            hpr = narf.hist_to_root(hp)
            hpr.SetName(f"jetMuonPt_passIso")
            hpr.SetTitle(f"passIso")
            drawCorrelationPlot(
                hpr,
                xAxisName,
                yAxisName,
                f"Events",
                f"{hpr.GetName()}",
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

            hfr = narf.hist_to_root(hf)
            hfr.SetName(f"jetMuonPt_failIso")
            hfr.SetTitle(f"failIso")
            drawCorrelationPlot(
                hfr,
                xAxisName,
                yAxisName,
                f"Events",
                f"{hfr.GetName()}",
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

            htr = narf.hist_to_root(ht)
            htr.SetName(f"jetMuonPt_anyIso")
            htr.SetTitle(f"AnyIso")
            drawCorrelationPlot(
                htr,
                xAxisName,
                yAxisName,
                f"Events",
                f"{htr.GetName()}",
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

    def runGenMuonFlavor(args):

        fname = args.inputfile[0]
        outdir_original = f"{args.outdir[0]}/genMuonFlavor/"
        outdir = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)

        ROOT.TH1.SetDefaultSumw2()

        canvas = ROOT.TCanvas("canvas", "", 800, 700)
        adjustSettings_CMS_lumi()
        canvas1D = ROOT.TCanvas("canvas1D", "", 800, 900)
        # canvas1D.SetGridx(1)
        canvas1D.SetGridy(1)

        groups = Datagroups(fname, mode="w_mass")
        datasets = groups.getNames()
        if args.processes is not None and len(args.processes):
            datasets = list(filter(lambda x: x in args.processes, datasets))
        else:
            datasets = list(filter(lambda x: x == "QCD", datasets))

        logger.debug(f"Using these processes: {datasets}")
        inputHistName = args.baseName  # Muon_genPartFlav_multi
        groups.setNominalName(inputHistName)
        groups.loadHistsForDatagroups(
            inputHistName, syst="", procsToRead=datasets, applySelection=False
        )
        histInfo = (
            groups.getDatagroups()
        )  # keys are same as returned by groups.getNames()
        s = hist.tag.Slicer()
        for d in datasets:
            logger.info(f"Running on process {d}")
            hin = histInfo[d].hists[inputHistName]
            logger.debug(hin.axes)
            # ('goodMuons_genPartFlav0', 'eta', 'pt', 'charge', 'mt', 'passIso')
            # integrate charge and fold eta into abseta
            hin = hin[{"charge": s[:: hist.sum]}]
            hAbsetaPt = hh.makeAbsHist(hin, "eta")
            # integrate eta-pt-charge for now
            hin = hin[{"pt": s[:: hist.sum], "eta": s[:: hist.sum]}]
            # make eta into abseta for now
            # makeAbsEta = True # make an option for this
            # etaAxisName = "eta"
            # if makeAbsEta:
            #    hin = hh.makeAbsHist(hin, "eta")
            #    etaAxisName = "abseta"

            newMtEdges = [0, 20, 40, 60, 80, 120]
            nMtBins = len(newMtEdges) - 1
            hin = hh.rebinHist(hin, "mt", newMtEdges)

            hpass = hin[{"passIso": True}]
            hfail = hin[{"passIso": False}]

            muonFlavAxisName = "0=unmatched, 1=prompt, 3=light, 4=c, 5=b"

            hFRF = hh.divideHists(hpass, hfail)
            hrootFRF = narf.hist_to_root(hFRF)
            minFRF, maxFRF = getMinMaxHisto(hrootFRF, sumError=False)
            hrootFRF.SetName("fakerateFactor_genPartFlav_mT")
            hrootFRF.SetTitle(f"min = {round(minFRF, 1)},   max = {round(maxFRF, 1)}")

            drawCorrelationPlot(
                hrootFRF,
                muonFlavAxisName,
                "m_{T} (GeV)",
                f"Fakerate factor (QCD MC)",
                hrootFRF.GetName(),
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
            # zoomed range
            drawCorrelationPlot(
                hrootFRF,
                muonFlavAxisName,
                "m_{T} (GeV)",
                f"Fakerate factor (QCD MC)::{minFRF},2.5",
                f"{hrootFRF.GetName()}_zoom",
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

            hrootPass = narf.hist_to_root(hpass)
            hrootPass.SetName("yields_passIso_genPartFlav_mT")
            hrootPass.SetTitle(f"Pass isolation")

            hrootFail = narf.hist_to_root(hfail)
            hrootFail.SetName("yields_failIso_genPartFlav_mT")
            hrootFail.SetTitle(f"Fail isolation")

            drawCorrelationPlot(
                hrootPass,
                muonFlavAxisName,
                "m_{T} (GeV)",
                f"Events (QCD MC)",
                hrootPass.GetName(),
                plotLabel="ForceTitle",
                outdir=outdir,
                smoothPlot=False,
                drawProfileX=False,
                scaleToUnitArea=False,
                draw_both0_noLog1_onlyLog2=0,
                passCanvas=canvas,
                nContours=args.nContours,
                palette=args.palette,
                invertPalette=args.invertPalette,
            )
            drawCorrelationPlot(
                hrootFail,
                muonFlavAxisName,
                "m_{T} (GeV)",
                f"Events (QCD MC)",
                hrootFail.GetName(),
                plotLabel="ForceTitle",
                outdir=outdir,
                smoothPlot=False,
                drawProfileX=False,
                scaleToUnitArea=False,
                draw_both0_noLog1_onlyLog2=0,
                passCanvas=canvas,
                nContours=args.nContours,
                palette=args.palette,
                invertPalette=args.invertPalette,
            )

            # now plot 1D distribution of events for each iso-mT bin (with binning from ABCD method)
            mtBinSlicesABCD = [
                s[0 : 1 : hist.sum],
                s[1 : 2 : hist.sum],
                s[2 :: hist.sum],
            ]
            for ips, passIso in enumerate([False, True]):
                for imt, mtSlice in enumerate(mtBinSlicesABCD):
                    hin1D = hin[{"passIso": passIso, "mt": mtSlice}]
                    isoTag = "pass iso" if passIso else "fail iso"
                    mtTag = (
                        "m_{T} < 20 GeV"
                        if imt == 0
                        else "20 < m_{T} < 40 GeV" if imt == 1 else "m_{T} > 40 GeV"
                    )
                    hroot1D = narf.hist_to_root(hin1D)
                    hroot1D.SetName(f"muonGenPartFlav_iso{ips}_mt{imt}")
                    hroot1D.SetTitle(f"QCD MC, {isoTag}, {mtTag}")
                    hroot1D.Scale(1.0 / hroot1D.Integral())
                    drawTH1(
                        hroot1D,
                        muonFlavAxisName,
                        "Fraction of events::0,0.75",
                        hroot1D.GetName(),
                        outdir,
                        passCanvas=canvas1D,
                        plotTitleLatex=hroot1D.GetTitle(),
                        drawStatBox=False,
                        histFillColor=ROOT.kGray,
                    )

        copyOutputToEos(outdir, outdir_original, eoscp=args.eoscp)

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################

    if args.runStudy == 0:
        runKinematics(args)
    elif args.runStudy == 1:
        runGenMatching(args)
    elif args.runStudy == 2:
        runGenMt(args)
    elif args.runStudy == 3:
        runJetMuon(args)
    elif args.runStudy == 4:
        runGenMuonFlavor(args)
