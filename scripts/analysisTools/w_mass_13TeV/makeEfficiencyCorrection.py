#!/usr/bin/env python3

import copy
import os
import pickle
import sys
from array import array

import hist
import lz4.frame
import numpy as np

import narf
import wums.output_tools
from utilities import common
from wremnants.datasets.datagroups import Datagroups
from wremnants.muon_efficiencies_smooth import cloneAxis
from wums import boostHistHelpers as hh
from wums import logging

## safe batch mode
args = sys.argv[:]
sys.argv = ["-b"]
import ROOT

sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

base_dir = common.base_dir

from scripts.analysisTools.plotUtils.utility import (
    adjustSettings_CMS_lumi,
    common_plot_parser,
    copyOutputToEos,
    createPlotDirAndCopyPhp,
    drawNTH1,
)

if __name__ == "__main__":
    parser = common_plot_parser()
    #
    parser.add_argument(
        "inputfile",
        type=str,
        help="Input file with histograms (pkl.lz4 or hdf5 file)",
    )
    parser.add_argument("outdir", type=str, help="Output folder")
    parser.add_argument(
        "-c",
        "--charge",
        type=int,
        choices=[-1, 0, 1],
        default=0,
        help="Select charge for plots, otherwise sum the charges if 0",
    )
    parser.add_argument(
        "--corrvar",
        type=str,
        default="run",
        choices=["run", "phi"],
        help="Variable to be used for the correction",
    )
    parser.add_argument(
        "--projvar",
        type=str,
        default="eta",
        choices=["eta", "pt"],
        help="Final variable to plot and compute the corrections for",
    )

    args = parser.parse_args()
    s = hist.tag.Slicer()

    logger = logging.setup_logger(os.path.basename(__file__), args.verbose)

    outdir_original = args.outdir
    outdir = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)

    corrvar = args.corrvar

    # for possible 2D
    canvas = ROOT.TCanvas("canvas", "", 800, 700)
    adjustSettings_CMS_lumi()
    # for 1D plots
    canvas1D = ROOT.TCanvas("canvas1D", "", 800, 900)

    procFilters = ["Data", "Zmumu"]

    groups = Datagroups(
        args.inputfile,
        filterGroups=procFilters,
        excludeGroups=None,
    )
    datasets = groups.getNames()
    logger.info(f"Will plot datasets {datasets}")

    groups.setNominalName("nominal")
    groups.loadHistsForDatagroups(
        "nominal",
        syst="",
        procsToRead=datasets,
        applySelection=True,
    )
    histInfo = groups.groups
    projvar = args.projvar
    keep_axes = [projvar, corrvar]

    x_title = {
        "eta": "#eta^{#mu}",
        "pt": "#it{p}_{T}^{#mu} (GeV)",
        "charge": "#it{q}^{#mu}",
    }

    logger.warning(histInfo[datasets[0]].hists["nominal"].axes)

    nCorrBins = histInfo[datasets[0]].hists["nominal"].axes[corrvar].size
    corrEdges = histInfo[datasets[0]].hists["nominal"].axes[corrvar].edges
    logger.info(f"Detected {nCorrBins} {corrvar} bins with edges {corrEdges}")
    if args.corrvar == "run":
        # nbins, array with lumi per bin (to check e.g. in scripts/plotting/plot_decorr_params.py)
        lumiPerRun = {
            2: [8.07, 8.74],
            3: [4.33, 7.94, 4.55],
            4: [4.33, 3.74, 4.19, 4.55],
            5: [2.33, 3.92, 3.90, 3.92, 2.74],
        }
        logger.info(f"with luminosity {lumiPerRun[nCorrBins]}")
        totLumi = np.sum(lumiPerRun[nCorrBins])

    heffiroot = {}
    for d in datasets:
        logger.info(f"Process {d}")
        onehist = histInfo[d].hists["nominal"]
        if args.charge:
            onehist = onehist[{"charge": -1.0j if args.charge == -1 else 1.0j}]
        h_d = onehist.project(*keep_axes)
        hincl_d = onehist.project(*[x for x in keep_axes if x != corrvar])
        heffiroot[d] = {}

        for i in range(nCorrBins):
            if args.corrvar == "run":
                hi = hh.scaleHist(
                    h_d[{corrvar: s[i]}], totLumi / lumiPerRun[nCorrBins][i]
                )
            else:
                hi = hh.scaleHist(h_d[{corrvar: s[i]}], nCorrBins)
            heffi = hh.divideHists(hi, hincl_d)
            heffiroot[d][i] = narf.hist_to_root(heffi)
            heffiroot[d][i].SetName(f"heffi_{d}_{corrvar}Bin{i}")
            heffiroot[d][i].SetTitle("")

    legymin = 0.42
    legymax = legymin + 0.06 * int(0.51 * (nCorrBins + 1))

    if args.corrvar == "run":
        legEntries = [
            f"Run {i}: {lumiPerRun[nCorrBins][i]} fb^{{-1}}" for i in range(nCorrBins)
        ]
    elif args.corrvar == "phi":
        legEntries = [
            f"{round(corrEdges[i], 2)} < #phi^{{#mu}} < {round(corrEdges[i+1], 2)}"
            for i in range(nCorrBins)
        ]
    else:
        legEntries = [f"{corrvar} bin {i}" for i in range(nCorrBins)]

    legProc = {
        "Data": "data",
        "Zmumu": "Z#rightarrow#mu#mu MC",
    }

    chargePostfix = ""
    if args.charge:
        sign = "minus" if args.charge < 0 else "plus"
        chargePostfix = f"_{sign}"

    # create a TH2 and convrt into boost again to save it
    nx = heffiroot["Data"][0].GetNbinsX()
    xmin = heffiroot["Data"][0].GetXaxis().GetBinLowEdge(1)
    xmax = heffiroot["Data"][0].GetXaxis().GetBinLowEdge(1 + nx)
    corrEdgesArray = array("d", corrEdges)
    logger.warning(f"{projvar} {nx} {xmin} {xmax}")
    logger.warning(corrEdgesArray)
    hsf2D = ROOT.TH2D(
        f"dataMC_ZmumuEffCorr_{projvar}_{corrvar}Bin{chargePostfix}",
        "",
        nx,
        xmin,
        xmax,
        nCorrBins,
        corrEdgesArray,
    )

    hsfroot = {}
    for i in range(nCorrBins):
        hsfroot[i] = copy.deepcopy(
            heffiroot["Data"][i].Clone(f"hsfroot_{corrvar}Bin{i}")
        )
        hsfroot[i].Divide(heffiroot["Zmumu"][i])
        idi = i + 1
        for j in range(nx):
            idj = j + 1
            corr = hsfroot[i].GetBinContent(idj)
            corr_unc = hsfroot[i].GetBinError(idj)
            hsf2D.SetBinContent(idj, idi, corr)
            hsf2D.SetBinError(idj, idi, corr_unc)
            logger.debug(
                f"{corrvar}-eta ID = {i} {j}: {round(corr,3)} +/- {round(corr_unc,3)}"
            )

    title_var = "Run" if corrvar == "run" else "#phi" if corrvar == "phi" else "CorrVar"

    effRange = "0.85,1.05" if corrvar == "run" else "0.80,1.1"

    for d in datasets:
        drawNTH1(
            [heffiroot[d][i] for i in range(nCorrBins)],
            legEntries,
            x_title[projvar],
            f"Efficiency ({legProc[d]})::{effRange}",
            f"efficiency_{d}_{projvar}_{nCorrBins}{corrvar}Bins{chargePostfix}",
            outdir,
            topMargin=0.06,
            leftMargin=0.16,
            rightMargin=0.04,
            labelRatioTmp=title_var + " #it{x}^{ }/^{ }" + title_var + " 0::0.93,1.07",
            legendCoords=f"0.18,0.96,{legymin},{legymax};2",
            transparentLegend=True,
            lowerPanelHeight=0.4,
            drawLumiLatex=True,
            passCanvas=canvas1D,
            noErrorRatioDen=False,
            drawErrorAll=True,
            onlyLineColor=True,
            useLineFirstHistogram=True,
            setOnlyLineRatio=True,
            lineWidth=2,
            moreTextLatex="",  # f"{selection_text}::0.18,0.50,0.05,0.04",
        )
    #
    drawNTH1(
        [hsfroot[i] for i in range(nCorrBins)],
        legEntries,
        x_title[projvar],
        f"Data/MC scale factor::0.8,1.1",
        f"scaleFactor_{projvar}_{nCorrBins}{corrvar}Bins{chargePostfix}",
        outdir,
        topMargin=0.06,
        leftMargin=0.16,
        rightMargin=0.04,
        labelRatioTmp=title_var + " #it{i}^{ }/^{ }" + title_var + " 0::0.93,1.07",
        legendCoords=f"0.18,0.96,{legymin},{legymax};2",
        transparentLegend=True,
        lowerPanelHeight=0.4,
        drawLumiLatex=True,
        passCanvas=canvas1D,
        noErrorRatioDen=False,
        drawErrorAll=True,
        onlyLineColor=True,
        useLineFirstHistogram=True,
        setOnlyLineRatio=True,
        lineWidth=2,
        moreTextLatex="",  # f"{selection_text}::0.18,0.50,0.05,0.04",
    )

    # logger.warning(f"{hsf2D.GetYaxis().GetBinLowEdge(1)} {hsf2D.GetYaxis().GetBinLowEdge(nCorrBins+1)}")
    resultDict = {}
    hsf2D_narf_withOF = narf.root_to_hist(hsf2D, axis_names=[projvar, corrvar])
    # logger.info(f"{hsf2D_narf_withOF}")
    # remove overflows
    newAxes = [
        cloneAxis(x, overflow=False, underflow=False, newName=None)
        for x in hsf2D_narf_withOF.axes
    ]
    hsf2D_narf = hist.Hist(
        *newAxes,
        name=hsf2D.GetName(),
        storage=hist.storage.Weight(),
    )
    hsf2D_narf.values(flow=False)[...] = hsf2D_narf_withOF.values(flow=False)[...]
    hsf2D_narf.variances(flow=False)[...] = hsf2D_narf_withOF.variances(flow=False)[...]
    resultDict[hsf2D.GetName()] = hsf2D_narf
    logger.info(f"Saving histogram with name {hsf2D.GetName()}")
    logger.info(f"{hsf2D_narf}")
    resultDict.update(
        {"meta_info": wums.output_tools.make_meta_info_dict(args=args, wd=base_dir)}
    )

    outfile = (
        outdir
        + f"/dataMC_ZmumuEffCorr_{projvar}_{nCorrBins}{corrvar}Bins{chargePostfix}.pkl.lz4"
    )
    with lz4.frame.open(outfile, "wb") as f:
        pickle.dump(resultDict, f, protocol=pickle.HIGHEST_PROTOCOL)

    copyOutputToEos(outdir, outdir_original, eoscp=args.eoscp)
