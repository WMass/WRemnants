#!/usr/bin/env python3

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
)
from scripts.analysisTools.w_mass_13TeV.plotPrefitTemplatesWRemnants import (
    plotPrefitHistograms,
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
        default="nominal_isoMtBins",
    )
    parser.add_argument(
        "-p",
        "--processes",
        default=None,
        nargs="*",
        type=str,
        help="Choose what processes to plot, otherwise all are done",
    )
    parser.add_argument(
        "--muonIsolation",
        type=int,
        nargs=2,
        default=[1, 1],
        choices=[-1, 0, 1],
        help="Apply isolation cut to triggering and not-triggering muon (in this order): -1/1 for failing/passing isolation, 0 for integrating it",
    )
    parser.add_argument(
        "--mtRange",
        type=float,
        nargs=2,
        default=[0, 40],
        choices=[-1.0, 0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 120.0],
        help="Apply mT cut, if upper edge is negative integrate the overflow",
    )
    parser.add_argument(
        "--charge", type=int, default=1, choices=[-1, 1], help="Charge selection"
    )
    parser.add_argument(
        "--rr",
        "--ratio-range",
        dest="ratioRange",
        default=(0.9, 1.1),
        type=float,
        nargs=2,
        help="Range for ratio plot",
    )
    args = parser.parse_args()

    logger = logging.setup_logger(os.path.basename(__file__), args.verbose)

    charges = {-1: "minus", 1: "plus"}
    isoVals = {-1: "FailIso", 0: "NoIso", 1: "PassIso"}
    chargeTag = charges[args.charge]
    chargeBin = 0 if args.charge == -1 else 1
    trigMuon_isoTag = isoVals[args.muonIsolation[0]]
    nonTrigMuon_isoTag = isoVals[args.muonIsolation[1]]
    highMtTag = "Inf" if args.mtRange[1] < 0 else int(args.mtRange[1])
    mtRangeTag = f"mT{int(args.mtRange[0])}To{highMtTag}"
    outdirTag = f"{chargeTag}/trigMuon{trigMuon_isoTag}_nonTrigMuon{nonTrigMuon_isoTag}/{mtRangeTag}/"
    ###############################################################################################
    fname = args.inputfile[0]
    outdir_original = f"{args.outdir[0]}/{outdirTag}/"
    outdir = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)

    ROOT.TH1.SetDefaultSumw2()

    canvas = ROOT.TCanvas("canvas", "", 800, 700)
    cwide = ROOT.TCanvas("cwide", "", 2400, 600)
    adjustSettings_CMS_lumi()
    canvas1D = ROOT.TCanvas("canvas1D", "", 800, 900)

    groups = Datagroups(fname)
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
    eps = 0.0001
    hdata2D = None
    hmc2D = []
    for d in datasets:
        logger.info(f"Running on process {d}")
        hin = histInfo[d].hists[inputHistName]
        logger.debug(hin.axes)
        ###
        upMtBin = (
            complex(0, args.mtRange[1] + eps)
            if args.mtRange[1] != -1
            else hist.overflow
        )
        h = hin[
            {
                "mt": s[complex(0, args.mtRange[0] + eps) : upMtBin : hist.sum],
                "charge": s[chargeBin],
            }
        ]
        ###
        if args.muonIsolation[0] < 0:
            h = h[{"trig_passIso": False}]
        elif args.muonIsolation[0] == 0:
            h = h[{"trig_passIso": s[:: hist.sum]}]
        else:
            h = h[{"trig_passIso": True}]
        ###
        if args.muonIsolation[1] < 0:
            h = h[{"nonTrig_passIso": False}]
        elif args.muonIsolation[1] == 0:
            h = h[{"nonTrig_passIso": s[:: hist.sum]}]
        else:
            h = h[{"nonTrig_passIso": True}]
        logger.debug(h.axes)
        if d == "Data":
            hdata2D = narf.hist_to_root(h)
            hdata2D.SetName(f"{d}_{chargeTag}")
            hdata2D.SetTitle(f"{d} {chargeTag}")
        else:
            hmc2D.append(narf.hist_to_root(h))
            hmc2D[-1].SetName(f"{d}_{chargeTag}")
            hmc2D[-1].SetTitle(f"{d}_{chargeTag}")
    # end of process loop
    plotPrefitHistograms(
        hdata2D,
        hmc2D,
        outdir,
        xAxisName="Triggering muon #eta",
        yAxisName="Triggering muon p_{T} (GeV)",
        chargeLabel=chargeTag,
        canvas=canvas,
        canvasWide=cwide,
        canvas1D=canvas1D,
        ratioRange=args.ratioRange,
        lumi=16.8,
    )
    copyOutputToEos(outdir, outdir_original, eoscp=args.eoscp)
