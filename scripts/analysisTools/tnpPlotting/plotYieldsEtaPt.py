#!/usr/bin/env python3

# open root files, integrate mass axis and plot yields vs eta-pt for passing and failing probes

import os

## safe batch mode
import sys

from wums import logging

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
    drawCorrelationPlot,
    safeGetObject,
    safeOpenFile,
)

sys.path.append(os.getcwd())

if __name__ == "__main__":

    parser = common_plot_parser()
    parser.add_argument("inputfile", type=str, nargs=1, help="Input file")
    parser.add_argument("outputfolder", type=str, nargs=1)
    parser.add_argument(
        "--postfix",
        type=str,
        default=None,
        help="Postfix for output name and plot title",
    )
    parser.add_argument(
        "--noIntegrateMassOverflows",
        action="store_true",
        help="When integrating mass, do not integrate also the overflow bins",
    )
    args = parser.parse_args()

    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    ROOT.TH1.SetDefaultSumw2()

    outdir_original = args.outputfolder[0]
    outdir = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)

    # get all TH3 histograms with mass-pt-eta
    projOpt = "yze"
    if args.noIntegrateMassOverflows:
        projOpt += " nuf nof"

    f = safeOpenFile(args.inputfile[0])
    hists = {}
    for k in f.GetListOfKeys():
        name = k.GetName()
        h = safeGetObject(f, name)
        if "TH3" not in h.ClassName():
            continue
        logger.info(f"Reading {h.ClassName()} {name}")
        hists[name] = h.Project3D(projOpt)
        hists[name].SetDirectory(0)
        hists[name].SetName(f"{name}_etapt")
        title = "Pass probes" if name.startswith("pass") else "Fail probes"
        if "pass" in name and "_alt" in name:
            title += " (SA vars)"
        if args.postfix:
            title += f" {args.postfix}"
        hists[name].SetTitle(title)
    f.Close()

    adjustSettings_CMS_lumi()
    canvas = ROOT.TCanvas("canvas", "", 900, 800)

    postfixCanvas = f"_{args.postfix}"

    for n in hists.keys():
        drawCorrelationPlot(
            hists[n],
            "Muon #eta",
            "Muon p_{T} (GeV)",
            "Events",
            f"{hists[n].GetName()}{postfixCanvas}",
            plotLabel="ForceTitle",
            outdir=outdir,
            palette=args.palette,
            nContours=args.nContours,
            invertPalette=args.invertPalette,
            passCanvas=canvas,
            drawOption="COLZ0",
        )

    copyOutputToEos(outdir, outdir_original, eoscp=args.eoscp)
