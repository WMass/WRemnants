#!/usr/bin/env python3

# W-
# python scripts/analysisTools/w_mass_13TeV/makeVertexStudy.py /scratch/mciprian/CombineStudies/vertexStudy/mw_TEST_scetlib_dyturboCorr.hdf5 scripts/analysisTools/plots/fromMyWremnants/vertexStudy/testNew/Wminusmunu/ -n vertexStudyHisto_noCut vertexStudyHisto_vetoMuon vertexStudyHisto_goodMuon vertexStudyHisto_fullSelNoMT --dzCut 0.1 -v 4 -p WminusmunuPostVFP
# W+
# python scripts/analysisTools/w_mass_13TeV/makeVertexStudy.py /scratch/mciprian/CombineStudies/vertexStudy/mw_TEST_scetlib_dyturboCorr.hdf5 scripts/analysisTools/plots/fromMyWremnants/vertexStudy/testNew/Wplusmunu/ -n vertexStudyHisto_noCut vertexStudyHisto_vetoMuon vertexStudyHisto_goodMuon vertexStudyHisto_fullSelNoMT --dzCut 0.1 -v 4 -p WplusmunuPostVFP
# Z Wlike
# python scripts/analysisTools/w_mass_13TeV/makeVertexStudy.py /scratch/mciprian/CombineStudies/vertexStudy/mw_TEST_scetlib_dyturboCorr.hdf5 scripts/analysisTools/plots/fromMyWremnants/vertexStudy/testNew/Zmumu_Wlike/ -n vertexStudyHisto_noCut vertexStudyHisto_vetoMuon vertexStudyHisto_goodMuon vertexStudyHisto_fullSelNoMT --dzCut 0.1 -v 4 -p ZmumuPostVFP
# Z dilepton
# python scripts/analysisTools/w_mass_13TeV/makeVertexStudy.py /scratch/mciprian/CombineStudies/vertexStudy/mw_TEST_scetlib_dyturboCorr_ZdileptonSelection.hdf5 scripts/analysisTools/plots/fromMyWremnants/vertexStudy/testNew/Zmumu_dilepton/ -n vertexStudyHisto_noCut vertexStudyHisto_vetoMuon vertexStudyHisto_goodMuon vertexStudyHisto_fullSelNoMT --dzCut 0.1 -v 4 -p ZmumuPostVFP --Zdilepton


import os

## safe batch mode
import sys

import hist

import narf
from wremnants.datasets.datagroups import Datagroups

# from wremnants import plot_tools,theory_tools,syst_tools
from wums import logging

args = sys.argv[:]
sys.argv = ["-b"]
import ROOT

sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

from scripts.analysisTools.plotUtils.utility import (
    adjustSettings_CMS_lumi,
    common_plot_parser,
    copyOutputToEos,
    createPlotDirAndCopyPhp,
    drawGraphCMS,
    legEntries_plots_,
)


def getVertexEfficiency(h, rebin=1, dzCut_cm=0.1, label=""):
    # h is pt vs vtxDiff_z
    # get 1D histograms versus pt in two versions, integrating all vtx_z, or only vtx_z < 1 mm (meaning < 0.1 since it is in cm)
    s = hist.tag.Slicer()
    if rebin != 1:
        h = h[{1: s[:: hist.rebin(rebin)]}]
    hptTot = h[{"absDiffGenRecoVtx_z": s[:: hist.sum]}]
    hptVtx1mm = h[
        {"absDiffGenRecoVtx_z": s[0 : complex(0, dzCut_cm + 0.0001) : hist.sum]}
    ]
    ptAxisName = hptTot.axes[0].name
    hrootNum = narf.hist_to_root(hptVtx1mm)
    hrootDen = narf.hist_to_root(hptTot)
    grAsErr = ROOT.TGraphAsymmErrors()
    graphName = f"vtxEff_{ptAxisName}"
    if len(label):
        graphName = f"{graphName}_{label}"
    grAsErr.SetName(graphName)
    grAsErr.Divide(hrootNum, hrootDen, "cl=0.683 b(1,1) mode")
    return grAsErr


if __name__ == "__main__":

    workingPoints = ["noCut", "vetoMuon", "goodMuon", "goodMuonAndSA", "fullSelNoMT"]

    parser = common_plot_parser()
    parser.add_argument("inputfile", type=str, help="Input file with histograms")
    parser.add_argument("outdir", type=str, help="Output folder")
    parser.add_argument(
        "-n",
        "--baseName",
        type=str,
        help="Histogram name in the file",
        default="nominal_vertexZstudy",
    )
    parser.add_argument(
        "--dzCut",
        default=0.1,
        type=float,
        help="Vertex dz(gen,reco) threshold in cm to calculate the efficiency",
    )
    # parser.add_argument('-p','--process', default="WplusmunuPostVFP", choices=["WplusmunuPostVFP", "WminusmunuPostVFP", "ZmumuPostVFP"], type=str, help='Choose what process to plot')
    parser.add_argument(
        "-p",
        "--process",
        default="Wmunu",
        choices=["Wmunu", "Zmumu"],
        type=str,
        help="Choose what process to plot",
    )
    parser.add_argument(
        "--Zdilepton",
        action="store_true",
        help="When using Z process, do stuff for a dilepton selection with both muons in acceptance",
    )
    args = parser.parse_args()

    logger = logging.setup_logger(os.path.basename(__file__), args.verbose)
    fname = args.inputfile
    outdir_original = args.outdir
    outdir = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)

    ROOT.TH1.SetDefaultSumw2()

    canvas = ROOT.TCanvas("canvas", "", 800, 700)
    adjustSettings_CMS_lumi()
    canvas1D = ROOT.TCanvas("canvas1D", "", 800, 900)

    groups = Datagroups(fname, mode="w_mass")
    datasets = groups.getNames()
    logger.info(f"Using these processes: {datasets}")
    datasets = list(filter(lambda x: x == args.process, groups.getNames()))
    s = hist.tag.Slicer()

    gr_vpts = {}

    logger.info(f"Using these processes: {datasets}")
    inputHistName = args.baseName
    groups.setNominalName(inputHistName)
    groups.loadHistsForDatagroups(
        inputHistName, syst="", procsToRead=datasets, applySelection=False
    )
    histInfo = groups.getDatagroups()  # keys are same as returned by groups.getNames()

    absEtaEdges4Plot = [0.0, 0.8, 1.6, 2.4]
    nAbsEtaBins = len(absEtaEdges4Plot) - 1
    xAxisName = "PreFSR boson #it{p}_{T} (GeV)"
    yAxisName = "Efficiency: vertex dz(gen,reco) < %g mm" % (10.0 * args.dzCut)

    for d in datasets:
        logger.info(f"Running on process {d}")
        hin = histInfo[d].hists[inputHistName]
        logger.debug(hin.axes)
        # abseta, passIso, passMT, absDiffGenRecoVtx_z, prefsrWpt

        etaBinLegEntry = []
        for ieta, eta in enumerate(absEtaEdges4Plot):
            if ieta == nAbsEtaBins:
                continue
            etaUp = absEtaEdges4Plot[ieta + 1]
            etaBinLabel = (
                "abseta"
                + str(eta).replace(".", "p")
                + "to"
                + str(etaUp).replace(".", "p")
            )
            etaBinLegEntry.append(
                "%s < |^{ }#eta^{#mu }| < %s" % (round(eta, 1), round(etaUp, 1))
            )
            if eta == 0.0:
                etaBinLegEntry[-1] = "|^{ }#eta^{#mu }| < %s" % round(etaUp, 1)
            # backward compatibility
            if "abseta" in hin.axes.name:
                h = hin[
                    {
                        "abseta": s[
                            complex(0, eta) : complex(0, nAbsEtaBins) : hist.sum
                        ],
                        "passIso": True,
                        "passMT": True,
                    }
                ]
            else:
                h = hin[
                    {
                        "abeta": s[
                            complex(0, eta) : complex(0, nAbsEtaBins) : hist.sum
                        ],
                        "passIso": True,
                        "passMT": True,
                    }
                ]
            if not args.Zdilepton:
                if args.process == "ZmumuPostVFP":
                    # require second gen lepton out of acceptance to mimic W, otherwise one would have had 2 reco leptons modulo selection efficiency
                    pass
                else:
                    # for W can integrate everything, since the other lepton is the neutrino
                    pass
            else:
                # require second lepton inside acceptance at gen level
                pass

            gr = getVertexEfficiency(h, rebin=1, dzCut_cm=args.dzCut, label=etaBinLabel)
            ymin = 1.0
            ymin = min(ymin, min(list(gr.GetY())))
            ymin = ymin - 0.1 * (1.0 - ymin)
            ymin = min(ymin, 0.95)
            gr_vpts[etaBinLabel] = gr

    wps = list(gr_vpts.keys())
    ymin = 1.0
    for n, g in gr_vpts.items():
        ymin = min(ymin, min(list(g.GetY())))
    ymin = ymin - 0.1 * (1.0 - ymin)
    ymin = min(ymin, 0.95)

    postfixForCanvasName = f"{args.process}"
    textForPlot = legEntries_plots_[args.process]
    if args.process == "Zmumu":
        if args.Zdilepton:
            textForPlot += " (2 muons in acceptance)"
            postfixForCanvasName += "_2muonInAcc"
        else:
            textForPlot += " (1 muon in acceptance)"
            postfixForCanvasName += "_1muonInAcc"

    colors = [
        ROOT.TColor.GetColor("#e42536"),
        ROOT.TColor.GetColor("#5790fc"),
        ROOT.TColor.GetColor("#964A8B"),
        ROOT.TColor.GetColor("#f89c20"),
        ROOT.TColor.GetColor("#9c9ca1"),
    ]
    markers = []
    for iwp, wp in enumerate(wps):
        gr_vpts[wp].SetMarkerColor(colors[iwp])
        gr_vpts[wp].SetMarkerStyle(20 + iwp)
        gr_vpts[wp].SetLineColor(colors[iwp])

    if args.Zdilepton:
        textForPlot += "::0.2,0.45"
    else:
        textForPlot += "::0.2,0.85"

    drawGraphCMS(
        [gr_vpts[wp] for wp in wps],
        xAxisName,
        f"{yAxisName}::{ymin},1.0",
        f"vtxEff_genBosonPt_{postfixForCanvasName}",
        outdir,
        leg_roc=etaBinLegEntry[:],
        legendCoords="0.52,0.18,0.92,0.39;1",
        passCanvas=canvas,
        etabinText=textForPlot,
        skipLumi=False,
        drawLumiLatex=True,
        useOriginalGraphStyle=True,
        graphDrawStyle="EP",
        legEntryStyle="EP",
        cmsText=args.CMStext,
        solidLegend=True,
    )
    copyOutputToEos(outdir, outdir_original, eoscp=args.eoscp)
