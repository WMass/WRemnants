#!/usr/bin/env python3
import copy
import os
import sys

import hist
import ROOT

import narf
from wremnants.datasets.datagroups import Datagroups

# from wremnants import plot_tools,theory_tools,syst_tools
from wums import logging

## safe batch mode
args = sys.argv[:]
sys.argv = ["-b"]
sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True


from scripts.analysisTools.plotUtils.utility import (
    adjustSettings_CMS_lumi,
    colors_plots_,
    common_plot_parser,
    copyOutputToEos,
    createPlotDirAndCopyPhp,
    drawCorrelationPlot,
    drawTH1dataMCstack,
    legEntries_plots_,
)

sys.path.append(os.getcwd())
from scripts.analysisTools.tests.cropNegativeTemplateBins import cropNegativeContent


def plotDistribution1D(
    hdata,
    hmc,
    datasets,
    outfolder_dataMC,
    canvas1Dshapes=None,
    xAxisName="variable",
    plotName="variable_failIso_jetInclusive",
    draw_both0_noLog1_onlyLog2=1,
    ratioPadYaxisTitle="Data/pred::0.9,1.1",
    scaleToUnitArea=False,
    noRatioPanel=False,
):

    createPlotDirAndCopyPhp(outfolder_dataMC)
    if not canvas1Dshapes:
        canvas1Dshapes = ROOT.TCanvas("canvas1Dshapes", "", 700, 800)

    nColumns = 3
    legendLowY = 0.82 if len(hmc) < nColumns else 0.72
    legend = ROOT.TLegend(0.2, legendLowY, 0.95, 0.92)
    legend.SetFillColor(0)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.SetNColumns(nColumns)

    stackIntegral = 0.0
    for d in datasets:
        if d == "Data":
            legend.AddEntry(hdata, "Data", "EP")
        else:
            cropNegativeContent(hmc[d])
            hmc[d].SetFillColor(colors_plots_[d])
            hmc[d].SetLineColor(ROOT.kBlack)
            hmc[d].SetMarkerSize(0)
            hmc[d].SetMarkerStyle(0)
            stackIntegral += hmc[d].Integral()

    if scaleToUnitArea:
        hdata.Scale(1.0 / hdata.Integral())

    stack_1D = ROOT.THStack("stack_1D", "signal and backgrounds")
    hmcSortedKeys = sorted(hmc.keys(), key=lambda x: hmc[x].Integral())
    for i in hmcSortedKeys:
        if scaleToUnitArea:
            hmc[i].Scale(1.0 / stackIntegral)
        stack_1D.Add(hmc[i])
    # reverse sorting for legend, first the ones with larger integral
    for i in list(reversed(hmcSortedKeys)):
        legend.AddEntry(hmc[i], legEntries_plots_[i], "LF")

    drawTH1dataMCstack(
        hdata,
        stack_1D,
        xAxisName,
        "Fraction of events" if scaleToUnitArea else "Events",
        plotName,
        outfolder_dataMC,
        legend,
        ratioPadYaxisNameTmp=ratioPadYaxisTitle,
        passCanvas=canvas1Dshapes,
        # xcmsText=-1 if scaleToUnitArea else 0.3,
        lumi="16.8",
        drawLumiLatex=True,
        noLegendRatio=True,
        draw_both0_noLog1_onlyLog2=draw_both0_noLog1_onlyLog2,
        noRatioPanel=noRatioPanel,
        topMargin=0.06,
    )


if __name__ == "__main__":

    parser = common_plot_parser()
    parser.add_argument("inputfile", type=str, nargs=1)
    parser.add_argument("outputfolder", type=str, nargs=1)
    parser.add_argument(
        "-p",
        "--processes",
        default=None,
        nargs="*",
        type=str,
        help="Choose what processes to plot, otherwise all are done",
    )
    parser.add_argument(
        "--excludeProcesses",
        default=None,
        nargs="*",
        type=str,
        help="Don't run over processes belonging to these groups (only accepts exact group names)",
    )
    parser.add_argument(
        "--plot", nargs="+", type=str, help="Choose what distribution to plot by name"
    )
    parser.add_argument("-x", "--xAxisName", nargs="+", type=str, help="x axis name")
    parser.add_argument(
        "-r",
        "--ratioRange",
        nargs=2,
        type=float,
        default=[0.9, 1.1],
        help="Min and max of ratio range",
    )
    parser.add_argument(
        "-y", "--yAxisName", nargs="+", type=str, help="y axis name (only for 2D plots)"
    )
    parser.add_argument(
        "-l",
        "--lumi",
        type=float,
        default=None,
        help="Normalization for 2D plots (if the input does not have data the luminosity is set to 1/fb)",
    )
    parser.add_argument(
        "--normUnitArea", action="store_true", help="Scale histogram to unit area"
    )
    parser.add_argument(
        "--drawLog",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Draw independent variable axis with log scale (0=both, 1=noLog, 2=onlyLog)",
    )
    parser.add_argument(
        "--project1D",
        type=str,
        default=None,
        help="Project n-dimensional distribution into this 1D variable",
    )
    parser.add_argument(
        "--selectAxis",
        nargs="*",
        default=[],
        type=str,
        help="Select axes by slicing, as axName=min,max, or axName=True (or =False) for boolean axis, or just axName to integrate all range",
    )
    parser.add_argument(
        "--postfix",
        type=str,
        default="",
        help="Add postfix to output plot names to distinguish different versions (default uses the plot name passed to --plot)",
    )
    args = parser.parse_args()

    logger = logging.setup_logger(os.path.basename(__file__), args.verbose)

    fname = args.inputfile[0]
    outdir_original = args.outputfolder[0]
    outdir = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)

    ROOT.TH1.SetDefaultSumw2()

    adjustSettings_CMS_lumi()
    canvas1D = ROOT.TCanvas("canvas1D", "", 800, 900)
    canvas2D = ROOT.TCanvas("canvas2D", "", 900, 800)

    groups = Datagroups(fname)
    if args.lumi:
        groups.lumi = args.lumi
        logger.warning(f"Renormalizing MC to {args.lumi}/fb")
    datasets = groups.getNames()
    if args.excludeProcesses is not None and len(args.excludeProcesses):
        datasets = list(filter(lambda x: x not in args.excludeProcesses, datasets))
    if args.processes is not None and len(args.processes):
        datasets = list(filter(lambda x: x in args.processes, datasets))
    logger.info(f"Will plot datasets {datasets}")

    ratioMin = args.ratioRange[0]
    ratioMax = args.ratioRange[1]
    ratioPadYaxisTitle = f"Data/pred::{ratioMin},{ratioMax}"

    for ip, p in enumerate(args.plot):

        if len(args.postfix):
            pname = f"{p}_{args.postfix}"
        else:
            pname = p

        groups.setNominalName(p)

        if len(args.selectAxis):
            s = hist.tag.Slicer()
            presel = {}
            logger.debug(args.selectAxis)
            logger.debug(f"Will apply the global preselection")
            epsilon = 0.00001
            for ps in args.selectAxis:
                if "=" in ps:
                    axName, axRange = ps.split("=")
                    if "," in ps:
                        axMin, axMax = map(float, axRange.split(","))
                        logger.info(f"{axName} in [{axMin},{axMax}]")
                        presel[axName] = s[
                            complex(0, axMin) : complex(0, axMax + epsilon) : hist.sum
                        ]
                    else:
                        trueOrFalse = axRange in ["True", "true", "TRUE", "1"]
                        logger.info(f"{axName} {trueOrFalse}")
                        presel[axName] = trueOrFalse
                else:
                    logger.info(f"Integrating {ps} axis")
                    presel[ps] = s[:: hist.sum]
            groups.setGlobalAction(lambda h: h[presel])

        groups.loadHistsForDatagroups(p, syst="", procsToRead=datasets)
        histInfo = groups.getDatagroups()

        rootHists = {}
        is2Dplot = False
        for d in datasets:
            hnarf = histInfo[d].hists[p]
            if args.project1D:
                if args.project1D not in hnarf.axes.name:
                    raise ValueError(
                        f"Histogram has axes {hnarf.axes.name} but requested axis for projection is {args.project1D}"
                    )
                else:
                    if any(ax != args.project1D for ax in hnarf.axes.name):
                        logger.info(
                            f"Projecting {hnarf.name} into 1D histogram versus {args.project1D}"
                        )
                        hnarf = hnarf.project(args.project1D)
            rootHists[d] = narf.hist_to_root(hnarf)
            logger.warning(d)
            rootHists[d].SetName(f"{p}_{d}")
            if len(hnarf.axes) == 2:
                is2Dplot = True
                xAxisName = args.xAxisName[ip]
                yAxisName = args.yAxisName[ip]
                rootHists[d].SetTitle(d)
                pname2D = rootHists[d].GetName()
                if len(args.postfix):
                    pname2D += f"_{args.postfix}"
                drawCorrelationPlot(
                    rootHists[d],
                    xAxisName,
                    yAxisName,
                    "Events",
                    pname2D,
                    plotLabel="ForceTitle",
                    outdir=outdir,
                    smoothPlot=False,
                    drawProfileX=False,
                    scaleToUnitArea=args.normUnitArea,
                    draw_both0_noLog1_onlyLog2=args.drawLog,
                    passCanvas=canvas2D,
                )

        if is2Dplot:
            continue

        hdata = (
            rootHists["Data"]
            if "Data" in rootHists.keys()
            else copy.deepcopy(rootHists[datasets[0]].Clone("dummyData"))
        )
        hmc = {d: rootHists[d] for d in datasets if d != "Data"}
        plotDistribution1D(
            hdata,
            hmc,
            datasets,
            outdir,
            canvas1Dshapes=canvas1D,
            xAxisName=args.xAxisName[ip],
            plotName=pname,
            ratioPadYaxisTitle=ratioPadYaxisTitle,
            scaleToUnitArea=args.normUnitArea,
            noRatioPanel="Data" not in rootHists.keys(),
            draw_both0_noLog1_onlyLog2=args.drawLog,
        )

    copyOutputToEos(outdir, outdir_original, eoscp=args.eoscp)
