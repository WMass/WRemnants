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

import boost_histogram as bh
import h5py
import hist

import narf
from utilities.io_tools import input_tools

# from wremnants import plot_tools,theory_tools,syst_tools
from wums import ioutils, logging

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
    drawGraphCMS,
)

sys.path.append(os.getcwd())


def getVertexEfficiency(h, rebin=1, dzCut_cm=0.1, label=""):
    # h is pt vs vtxDiff_z
    # get 1D histograms versus pt in two versions, integrating all vtx_z, or only vtx_z < 1 mm (meaning < 0.1 since it is in cm)
    s = hist.tag.Slicer()
    if rebin != 1:
        h = h[{1: s[:: hist.rebin(rebin)]}]
    hptTot = h[{"genVtx_PV_Zdiff": s[:: hist.sum]}]
    hptVtx1mm = h[
        {
            "genVtx_PV_Zdiff": s[
                bh.loc(-dzCut_cm + 0.0001) : bh.loc(dzCut_cm + 0.0001) : hist.sum
            ]
        }
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
        nargs="+",
        help="Histogram name in the file",
        default=["vertexStudyHisto_vetoMuon"],
        choices=[f"vertexStudyHisto_{wp}" for wp in workingPoints],
    )
    parser.add_argument(
        "--dzCut",
        default=0.1,
        type=float,
        help="Vertex dz(gen,reco) threshold in cm to calculate the efficiency",
    )
    parser.add_argument(
        "-p",
        "--process",
        default="WplusmunuPostVFP",
        choices=["WplusmunuPostVFP", "WminusmunuPostVFP", "ZmumuPostVFP"],
        type=str,
        help="Choose what process to plot",
    )
    parser.add_argument(
        "--Zdilepton",
        action="store_true",
        help="When using Z process, do stuff for a dilepton selection, rather than Wlike",
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
    xAxisName = "Muon #eta"
    yAxisName = "Muon p_{T} (GeV)"

    h5file = h5py.File(fname, "r")
    results = input_tools.load_results_h5py(h5file)

    s = hist.tag.Slicer()

    gr_vpts = {}

    for n in args.baseName:

        # names of axes: genVtx_PV_Zdiff, genBosonPt, genMuonPt, genMuonAbsEta, genOtherLepAbsEta
        logger.info(f"Running on process {args.process}")
        histObj = results[args.process]["output"][n]
        if isinstance(histObj, ioutils.H5PickleProxy):
            histObj = histObj.get()
        h = histObj.copy()
        logger.debug(h.axes)

        nBins_genMuonAbsEta = h.axes["genMuonAbsEta"].size
        nBins_genMuonPt = h.axes["genMuonPt"].size
        nBins_genBosonPt = h.axes["genBosonPt"].size
        nBins_genOtherLepAbsEta = h.axes["genOtherLepAbsEta"].size

        workingPoint = "_".join(n.split("_")[1:])
        yAxisName = "Efficiency: vertex dz(gen,reco) < %g mm" % (10.0 * args.dzCut)

        # integrate gen muon eta in acceptance (for Z it is the muon lepton, so negative muon)
        # also remove muon pt bins below 26 GeV (the trigger selection may cut them away anyway)
        h = h[{"genMuonAbsEta": s[0 : nBins_genMuonAbsEta : hist.sum]}]
        if not args.Zdilepton:
            if args.process == "ZmumuPostVFP":
                # require second gen lepton out of acceptance to mimic W, otherwise one would have had 2 reco leptons modulo selection efficiency
                # pass
                h = h[{"genOtherLepAbsEta": s[hist.overflow]}]
            else:
                # for W can integrate everything, since the other lepton is the neutrino
                h = h[{"genOtherLepAbsEta": s[:: hist.sum]}]
        else:
            # require second lepton inside acceptance at gen level
            h = h[{"genOtherLepAbsEta": s[0 : nBins_genOtherLepAbsEta : hist.sum]}]

        # get histogram versus boson pt or muon pt
        hvpt = h[{"genMuonPt": s[:: hist.sum]}]
        # hmupt = h[{"genBosonPt" : s[0:nBins_genBosonPt:hist.sum]}]

        for hpt in [hvpt]:  # , hmupt]:
            ptLabel = hpt.axes[1].name
            isBoson = True if ptLabel == "genBosonPt" else False
            xAxisName = "PreFSR " + ("boson" if isBoson else "muon") + " p_{T} [GeV]"
            gr = getVertexEfficiency(
                hpt, rebin=2 if isBoson else 1, dzCut_cm=args.dzCut, label=workingPoint
            )
            ymin = 1.0
            ymin = min(ymin, min(list(gr.GetY())))
            ymin = ymin - 0.1 * (1.0 - ymin)
            ymin = min(ymin, 0.95)
            # drawGraphCMS([gr], xAxisName, f"{yAxisName}::{ymin},1.0", gr.GetName(), outdir, leg_roc=[workingPoint],
            #             passCanvas=canvas, skipLumi=True)
            if isBoson:
                gr_vpts[workingPoint] = gr

    wps = list(gr_vpts.keys())
    xAxisName = "PreFSR boson p_{T} [GeV]"
    ymin = 1.0
    for n, g in gr_vpts.items():
        ymin = min(ymin, min(list(g.GetY())))
    ymin = ymin - 0.1 * (1.0 - ymin)
    ymin = min(ymin, 0.95)

    postfixForCanvasName = f"{args.process}"
    textForPlot = f"{args.process}"
    if args.process == "ZmumuPostVFP":
        if args.Zdilepton:
            textForPlot += " (2 muons in acceptance)"
            postfixForCanvasName += "_2muonInAcc"
        else:
            textForPlot += " (1 muon in acceptance)"
            postfixForCanvasName += "_1muonInAcc"

    drawGraphCMS(
        [gr_vpts[wp] for wp in wps],
        xAxisName,
        f"{yAxisName}::{ymin},1.0",
        f"vtxEff_genBosonPt_multiWP_{postfixForCanvasName}",
        outdir,
        leg_roc=wps[:],
        legendCoords="0.55,0.30,0.95,0.58;1",
        passCanvas=canvas,
        etabinText=f"{textForPlot}::0.18,0.15",
        skipLumi=True,
    )
    copyOutputToEos(outdir, outdir_original, eoscp=args.eoscp)
