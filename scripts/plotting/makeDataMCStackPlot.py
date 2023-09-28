from wremnants.datasets.datagroups import Datagroups
from wremnants import histselections as sel
from wremnants import plot_tools,theory_tools,syst_tools
from utilities import boostHistHelpers as hh,common,output_tools
import matplotlib.pyplot as plt
from matplotlib import colormaps
import argparse
import os
import shutil
from wremnants import logging
import pathlib
import hist
import re
import numpy as np

xlabels = {
    "pt" : r"p$_{T}^{\ell}$ (GeV)",
    "eta" : r"$\eta^{\ell}$",
    "ptll" : r"p$_{\mathrm{T}}^{\ell\ell}$ (GeV)",
    "yll" : r"y$^{\ell\ell}$",
    "mll" : r"m$_{\ell\ell}$ (GeV)",
    "ewMll" : r"m$^{\mathrm{EW}}_{\ell\ell}$ (GeV)",
    "costhetastarll" : r"$\cos{\phi^{\star}_{\ell\ell}}$",
    "phistarll" : r"$\phi^{\star}_{\ell\ell}$",
    "MET_pt" : r"p$_{\mathrm{T}}^{miss}$ (GeV)",
    "MET" : r"p$_{\mathrm{T}}^{miss}$ (GeV)",
    "met" : r"p$_{\mathrm{T}}^{miss}$ (GeV)",
    "mt" : r"m$_{T}^{\ell\nu}$ (GeV)",
    "etaPlus" : r"$\eta^{\ell(+)}$",
    "etaMinus" : r"$\eta^{\ell(-)}$",
    "ptPlus" : r"p$_{\mathrm{T}}^{\ell(+)}$ (GeV)",
    "ptMinus" : r"p$_{\mathrm{T}}^{\ell(-)}$ (GeV)",
    "etaSum":r"$\eta^{\ell(+)} + \eta^{\ell(-)}$",
    "etaDiff":r"$\eta^{\ell(+)} - \eta^{\ell(-)}$",
    "massVgen": "massVgen",
    "ewMll": "ewMll",
    "ewMlly": "ewMlly",
    "ewLogDeltaM": "ewLogDeltaM",
    # add 2d unrolled plots 
    "pt-eta" : r"(p$_{T}^{\ell}$, $\eta^{\ell}$) bin",
    "ptll-yll":r"p$_{\mathrm{T}}^{\ell\ell}$, y$^{\ell\ell}$ bin",
    "mll-yll":r"m$_{\ell\ell}$, y$^{\ell\ell}$ bin",
    "mll-ptll":r"m$_{\ell\ell}$, p$_{\mathrm{T}}^{\ell\ell}$ bin",
    "mll-etaPlus":r"m$_{\ell\ell}$, $\eta^{\ell(+)}$ bin",
    "mll-etaMinus":r"m$_{\ell\ell}$, $\eta^{\ell(-)}$ bin",
    "etaPlus-etaMinus":r"$\eta^{\ell(+)}$, $\eta^{\ell(-)}$ bin",
    "etaSum-etaDiff":r"$\eta^{\ell(+)} + \eta^{\ell(-)}$, $\eta^{\ell(+)} - \eta^{\ell(-)}$ bin",
    # add 3d unrolled plots 
    "mll-etaPlus-etaMinus":r"m$_{\ell\ell}$, $\eta^{\ell(+)}$, $\eta^{\ell(-)}$ bin",
    "mll-etaSum-etaDiff":r"m$_{\ell\ell}$, $\eta^{\ell(+)} + \eta^{\ell(-)}$, $\eta^{\ell(+)} - \eta^{\ell(-)}$ bin",
}

parser = argparse.ArgumentParser()
parser.add_argument("infile", help="Output file of the analysis stage, containing ND boost histogrdams")
parser.add_argument("--ratioToData", action='store_true', help="Use data as denominator in ratio")
parser.add_argument("-n", "--baseName", type=str, help="Histogram name in the file (e.g., 'nominal')", default="nominal")
parser.add_argument("--nominalRef", type=str, help="Specify the nominal his if baseName is a variation hist (for plotting alt hists)")
parser.add_argument("--hists", type=str, nargs='+', required=True, help="List of histograms to plot")
parser.add_argument("-c", "--channel", type=str, choices=["plus", "minus", "all"], default="all", help="Select channel to plot")
parser.add_argument("-p", "--outpath", type=str, default=os.path.expanduser("~/www/WMassAnalysis"), help="Base path for output")
parser.add_argument("-f", "--outfolder", type=str, default="test", help="Subfolder for output")
parser.add_argument("-r", "--rrange", type=float, nargs=2, default=[0.9, 1.1], help="y range for ratio plot")
parser.add_argument("--rebin", type=int, default=1, help="Rebin (for now must be an int)")
parser.add_argument("--ylim", type=float, nargs=2, help="Min and max values for y axis (if not specified, range set automatically)")
parser.add_argument("--yscale", type=float, help="Scale the upper y axis by this factor (useful when auto scaling cuts off legend)")
parser.add_argument("--xlim", type=float, nargs=2, help="min and max for x axis")
parser.add_argument("-a", "--name_append", default="", type=str, help="Name to append to file name")
parser.add_argument("--debug", action='store_true', help="Print debug output")
parser.add_argument("--procFilters", type=str, nargs="*", help="Filter to plot (default no filter, only specify if you want a subset")
parser.add_argument("--noData", action='store_true', help="Don't plot data")
parser.add_argument("--noFill", action='store_true', help="Don't fill stack")
parser.add_argument("--scaleleg", type=float, default=1.0, help="Scale legend text")
parser.add_argument("--fitresult", type=str, help="Specify a fitresult root file to draw the postfit distributions with uncertainty bands")
parser.add_argument("--prefit", action='store_true', help="Use the prefit uncertainty from the fitresult root file, instead of the postfit. (--fitresult has to be given)")
parser.add_argument("--eoscp", action='store_true', help="Use of xrdcp for eos output rather than the mount")
parser.add_argument("--selection", type=str, help="Specify custom selections as comma seperated list (e.g. '--selection passIso=0,passMT=1' )")
parser.add_argument("--presel", type=str, nargs="*", default=[], help="Specify custom selections on input histograms to integrate some axes, giving axis name and min,max (e.g. '--presel pt=ptmin,ptmax' ) or just axis name for bool axes")

subparsers = parser.add_subparsers(dest="variation")
variation = subparsers.add_parser("variation", help="Arguments for adding variation hists")
variation.add_argument("--varName", type=str, nargs='+', required=True, help="Name of variation hist")
variation.add_argument("--varLabel", type=str, nargs='+', required=True, help="Label(s) of variation hist for plotting")
variation.add_argument("--selectAxis", type=str, nargs='+', help="If you need to select a variation axis")
variation.add_argument("--selectEntries", type=str, nargs='+', help="entries to read from the selected axis")
variation.add_argument("--colors", type=str, nargs='+', help="Variation colors")
variation.add_argument("--linestyle", type=str, default=[], nargs='+', help="Linestyle for variations")
variation.add_argument("--doubleColors", action='store_true', help="Auto generate colors in pairs (useful for systematics)")
variation.add_argument("--fillBetween", action='store_true', help="Fill between uncertainty hists in ratio")
variation.add_argument("--skipFillBetween", type=int, default=0, help="Don't fill between the first N hists (only relevant if --fillBetween = True)")

args = parser.parse_args()

logger = logging.setup_logger("makeDataMCStackPlot", 4 if args.debug else 3)

def padArray(ref, matchLength):
    return ref+ref[-1:]*(len(matchLength)-len(ref))

addVariation = hasattr(args, "varName") and args.varName is not None

if args.fitresult and len(args.hists) > 1:
	raise ValueError("Multiple hists not supported for combine-based pre/post-fit plotting")

entries = []
if addVariation and (args.selectAxis or args.selectEntries):
    if not (args.selectAxis and args.selectEntries):
        raise ValueError("Must --selectAxis and --selectEntries together")
    if len(args.varLabel) != 1 and len(args.varLabel) != len(args.selectEntries):
        raise ValueError("Must specify the same number of args for --selectEntries, and --varLabel"
                         f" found selectEntries={len(args.selectEntries)} and varLabel={len(args.varLabel)}")
    if len(args.varName) < len(args.selectEntries):
        args.varName = padArray(args.varName, args.selectEntries)
    axes = padArray(args.selectAxis, args.varName)
    entries = padArray(args.selectEntries, args.varName)

outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)

groups = Datagroups(args.infile, filterGroups=args.procFilters, excludeGroups=None if args.procFilters else ['QCD'])

# There is probably a better way to do this but I don't want to deal with it
datasets = groups.getNames()
logger.info(f"Will plot datasets {datasets}")

select = {} if args.channel == "all" else {"charge" : -1.j if args.channel == "minus" else 1.j}

if len(args.presel):
    s = hist.tag.Slicer()
    presel = {}
    logger.debug(args.presel)
    logger.debug(f"Will apply the global preselection")
    for ps in args.presel:
        if "=" in ps:
            axName,axRange = ps.split("=")
            axMin,axMax = map(float, axRange.split(","))
            logger.info(f"{axName} in [{axMin},{axMax}]")
            presel[axName] = s[complex(0, axMin):complex(0, axMax):hist.sum]
        else:
            logger.info(f"Integrating boolean {ps} axis")
            presel[ps] = s[::hist.sum]
    groups.setGlobalAction(lambda h: h[presel])

if args.selection:
    for selection in args.selection.split(","):
        axis, value = selection.split("=")
        select[axis] = int(value)
    applySelection=False
else:
    applySelection=True

if not args.nominalRef:
    nominalName = args.baseName.rsplit("_", 1)[0]
    groups.setNominalName(nominalName)
    groups.loadHistsForDatagroups(args.baseName, syst="", procsToRead=datasets, applySelection=applySelection, 
        fakerateIntegrationAxes=list(set([x for h in args.hists for x in h.split("-") if x not in ["pt", "eta", "charge"]])))
else:
    nominalName = args.nominalRef
    groups.setNominalName(nominalName)
    groups.loadHistsForDatagroups(nominalName, syst=args.baseName, procsToRead=datasets, applySelection=applySelection,
        fakerateIntegrationAxes=list(set([x for h in args.hists for x in h.split("-") if x not in ["pt", "eta", "charge"]])))

exclude = ["Data"] 
unstack = exclude[:]
if args.noData:
    unstack.remove("Data")

# TODO: In should select the correct hist for the transform, not just the first
transforms = syst_tools.syst_transform_map(nominalName, args.hists[0])

if addVariation:
    logger.info(f"Adding variation {args.varName}")
    varLabels = padArray(args.varLabel, args.varName)
    # If none matplotlib will pick a random color
    ncols = len(args.varName) if not args.doubleColors else int(len(args.varName)/2)
    colors = args.colors if args.colors else [colormaps["tab10" if ncols < 10 else "tab20"](int(i/2) if args.doubleColors else i) for i in range(len(args.varName))]
    for i, (label,name,color) in enumerate(zip(varLabels,args.varName,colors)):
        entry = entries[i] if entries else None
        do_transform = entry in transforms
        name = name if name != "" else nominalName
        load_op = {}
        action=None

        if entry and entry.isdigit():
            entry = int(entry)

        if args.selectAxis or do_transform:
            transform_procs = groups.getProcNames(exclude_group=exclude)
            if do_transform:
                action = transforms[entry]["action"]
                if "procs" in transforms[entry]:
                    transform_procs = transforms[entry]["procs"]
                varname = entry
            else:
                ax = axes[i]
                action = lambda x: x[{ax : entry}] if ax in x.axes.name else x
                varname = name+str(entry)
            load_op = {p : action for p in transform_procs}
        else:
            varname = name

        reload = name != args.baseName
        # The action map will only work if reloading, otherwise need to apply some transform
        # to the already loaded hist
        if load_op and reload:
            action = None
        groups.addSummedProc(nominalName, relabel=args.baseName, name=name, label=label, exclude=exclude,
            color=color, reload=reload, rename=varname, procsToRead=datasets,
            preOpMap=load_op, action=action, forceNonzero=True)

        exclude.append(varname)
        unstack.append(varname)

groups.sortByYields(args.baseName, nominalName=nominalName)
histInfo = groups.getDatagroups()

logger.info(f"Unstacked processes are {exclude}")
prednames = list(reversed(groups.getNames([d for d in datasets if d not in exclude], exclude=False)))
logger.info(f"Stacked processes are {prednames}")

def collapseSyst(h):
    if type(h.axes[-1]) == hist.axis.StrCategory:
        return h[...,0]
    for ax in ["systIdx", "tensor_axis_0", "vars"]:
        if ax in h.axes.name:
            return h[{ax : 0}].copy()
    return h

overflow_ax = ["ptll", "chargeVgen", "massVgen", "ptVgen", "absEtaGen", "ptGen", "ptVGen", "absYVGen"]
for h in args.hists:
    if len(h.split("-")) > 1:
        action = lambda x: sel.unrolledHist(collapseSyst(x[select]), obs=h.split("-"))
    else:
        action = lambda x: hh.projectNoFlow(collapseSyst(x[select]), h, overflow_ax)
    fig = plot_tools.makeStackPlotWithRatio(histInfo, prednames, histName=args.baseName, ylim=args.ylim, yscale=args.yscale,
            fill_between=args.fillBetween if hasattr(args, "fillBetween") else None, 
            skip_fill=args.skipFillBetween if hasattr(args, "skipFillBetween") else 0,
            action=action, unstacked=unstack, 
            fitresult=args.fitresult, prefit=args.prefit,
            xlabel=xlabels.get(h,h), ylabel="Events/bin", rrange=args.rrange, binwnorm=1.0, lumi=groups.lumi,
            ratio_to_data=args.ratioToData, rlabel="Pred./Data" if args.ratioToData else "Data/Pred.",
            xlim=args.xlim, no_fill=args.noFill, cms_decor="Preliminary",
            legtext_size=20*args.scaleleg, unstacked_linestyles=args.linestyle if hasattr(args, "linestyle") else [])

    fitresultstring=""
    if args.fitresult:
        fitresultstring = "prefit" if args.prefit else "postfit"
    var_arg = None
    if "varName" in args and args.varName:
        var_arg = args.varName[0]
        if "selectEntries" in args and args.selectEntries:
            var_arg = args.selectEntries[0] if not args.selectEntries[0].isdigit() else (var_arg+args.selectEntries[0])
    to_join = [f"{h.replace('-','_')}"]+[var_arg]+[fitresultstring, args.name_append]+[args.channel.replace("all", "")]
    outfile = "_".join(filter(lambda x: x, to_join))

    plot_tools.save_pdf_and_png(outdir, outfile)

    # The action has already been applied to the underlying hist in this case
    if args.fitresult:
        action = lambda x: x

    stack_yields = groups.make_yields_df(args.baseName, prednames, norm_proc="Data", action=action)
    unstacked_yields = groups.make_yields_df(args.baseName, unstack, norm_proc="Data", action=action)
    plot_tools.write_index_and_log(outdir, outfile, 
        yield_tables={"Stacked processes" : stack_yields, "Unstacked processes" : unstacked_yields},
        analysis_meta_info={"AnalysisOutput" : groups.getMetaInfo()},
        args=args,
    )

if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
    output_tools.copy_to_eos(args.outpath, args.outfolder)
