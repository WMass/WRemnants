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
        default="nominal_testIsoMtFakeRegions",
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
        "--isoBinEdges",
        type=int,
        nargs="+",
        default=[0, 0.15, 0.3],
        help="Bins to test (overflow added automatically)",
    )
    parser.add_argument(
        "--mtBinEdges",
        type=float,
        nargs="+",
        default=[0, 22, 45],
        help="mT bins to test (overflow added automatically)",
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=1,
        choices=[-1, 0, 1],
        help="Triggering muon charge selection (0 integrates both charges)",
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

    charges = {-1: "minus", 0: "both", 1: "plus"}
    chargeTag = charges[args.charge]
    chargeBin = 0 if args.charge == -1 else 1
    outdirTag = f"trigMuonCharge_{chargeTag}_nonTrigMuonPassIso/"
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
    chargeSlice = s[chargeBin] if args.charge != 0 else s[:: hist.sum]
    eps = 0.0001
    isoBinEdges = args.isoBinEdges
    mtBinEdges = args.mtBinEdges

    for jiso, iso in enumerate(isoBinEdges):
        isoLow = round(iso, 2)
        isoLowStr = str(isoLow).replace(".", "p")
        maxJiso = len(isoBinEdges) - 1
        isoHighStr = str(round(isoBinEdges[jiso + 1], 2)) if jiso < maxJiso else "Inf"
        isoHighStr = isoHighStr.replace(".", "p")
        isoTag = f"iso{isoLowStr}To{isoHighStr}"
        for jmt, mt in enumerate(mtBinEdges):
            mtLowStr = str(round(mt))
            maxJmt = len(mtBinEdges) - 1
            mtHighStr = str(round(mtBinEdges[jmt + 1])) if jmt < maxJmt else "Inf"
            mtTag = f"mt{mtLowStr}To{mtHighStr}"
            logger.info(
                f"Processing bin: {isoLowStr} < relIso < {isoHighStr} && {mtLowStr} < mT < {mtHighStr} GeV"
            )
            #
            hdata2D = None
            hmc2D = []
            binTag = f"{isoTag}_{mtTag}"
            outdirBin = f"{outdir}/{binTag}/"
            createPlotDirAndCopyPhp(outdirBin, eoscp=args.eoscp)
            for d in datasets:
                logger.info(f"Running on process {d}")
                hin = histInfo[d].hists[inputHistName]
                # logger.debug(hin.axes)
                ###
                # the edges we use might require integrating more bins, so can't just use the jxxx index
                mtSlice = s[complex(0, mt + eps) :: hist.sum]
                if jmt < maxJmt:
                    mtSlice = s[
                        complex(0, mt + eps) : complex(
                            0, mtBinEdges[jmt + 1] + eps
                        ) : hist.sum
                    ]
                isoSlice = s[complex(0, iso + eps) :: hist.sum]
                if jiso < maxJiso:
                    isoSlice = s[
                        complex(0, iso + eps) : complex(
                            0, isoBinEdges[jiso + 1] + eps
                        ) : hist.sum
                    ]
                h = hin[{"mt": mtSlice, "relIso": isoSlice, "charge": chargeSlice}]
                # logger.debug(h.axes)
                if d == "Data":
                    hdata2D = narf.hist_to_root(h)
                    hdata2D.SetName(f"{d}_{chargeTag}")
                    hdata2D.SetTitle(f"{d} {binTag} {chargeTag}")
                else:
                    hmc2D.append(narf.hist_to_root(h))
                    hmc2D[-1].SetName(f"{d}_{chargeTag}")
                    hmc2D[-1].SetTitle(f"{d} {binTag} {chargeTag}")
            # end of process loop
            plotPrefitHistograms(
                hdata2D,
                hmc2D,
                outdirBin,
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
