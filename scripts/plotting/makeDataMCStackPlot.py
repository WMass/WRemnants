from wremnants.datasets.datagroups import Datagroups
from wremnants import plot_tools,theory_tools,syst_tools
from utilities import boostHistHelpers as hh, common, logging
from utilities.styles import styles
from utilities.io_tools import output_tools
import matplotlib.pyplot as plt
from matplotlib import colormaps
import argparse
import os
import shutil
import pathlib
import hist
import re
import numpy as np

parser = common.plot_parser()
parser.add_argument("infile", help="Output file of the analysis stage, containing ND boost histograms")
parser.add_argument("--ratioToData", action='store_true', help="Use data as denominator in ratio")
parser.add_argument("-n", "--baseName", type=str, help="Histogram name in the file (e.g., 'nominal')", default="nominal")
parser.add_argument("--nominalRef", type=str, help="Specify the nominal hist if baseName is a variation hist (for plotting alt hists)")
parser.add_argument("--hists", type=str, nargs='+', required=True, help="List of histograms to plot")
parser.add_argument("-c", "--channel", type=str, choices=["plus", "minus", "all"], default="all", help="Select channel to plot")
parser.add_argument("-r", "--rrange", type=float, nargs=2, default=[0.9, 1.1], help="y range for ratio plot")
parser.add_argument("--rebin", type=int, nargs='*', default=[], help="Rebin axis by this value (default, 1, does nothing)")
parser.add_argument("--absval", type=int, nargs='*', default=[], help="Take absolute value of axis if 1 (default, 0, does nothing)")
parser.add_argument("--axlim", type=float, default=[], nargs='*', help="Restrict axis to this range (assumes pairs of values by axis, with trailing axes optional)")
parser.add_argument("--rebinBeforeSelection", action='store_true', help="Rebin before the selection operation (e.g. before fake rate computation), default if after")
parser.add_argument("--logy", action='store_true', help="Enable log scale for y axis")
parser.add_argument("--ylim", type=float, nargs=2, help="Min and max values for y axis (if not specified, range set automatically)")
parser.add_argument("--yscale", type=float, help="Scale the upper y axis by this factor (useful when auto scaling cuts off legend)")
parser.add_argument("--xlim", type=float, nargs=2, help="min and max for x axis")
parser.add_argument("--procFilters", type=str, nargs="*", help="Filter to plot (default no filter, only specify if you want a subset")
parser.add_argument("--noData", action='store_true', help="Don't plot data")
parser.add_argument("--noFill", action='store_true', help="Don't fill")
parser.add_argument("--noStack", action='store_true', help="Don't stack")
parser.add_argument("--noRatio", action='store_true', help="Don't make ratio plot")
parser.add_argument("--density", action='store_true', help="Normalize each process to unity, only works with '--noStack'")
parser.add_argument("--flow", type=str, choices=["show", "sum", "hint", "none"], default="none", help="Whether plot the under/overflow bin")
parser.add_argument("--fitresult", type=str, help="Specify a fitresult root file to draw the postfit distributions with uncertainty bands")
parser.add_argument("--prefit", action='store_true', help="Use the prefit uncertainty from the fitresult root file, instead of the postfit. (--fitresult has to be given)")
parser.add_argument("--noRatioErr", action='store_false', dest="ratioError", help="Don't show stat unc in ratio")
parser.add_argument("--selection", type=str, help="Specify custom selections as comma seperated list (e.g. '--selection passIso=0,passMT=[2::hist.sum]' )")
parser.add_argument("--presel", type=str, nargs="*", default=[], help="Specify custom selections on input histograms to integrate some axes, giving axis name and min,max (e.g. '--presel pt=ptmin,ptmax' ) or just axis name for bool axes")
parser.add_argument("--normToData", action='store_true', help="Normalize MC to data")
parser.add_argument("--fakeEstimation", type=str, help="Set the mode for the fake estimation", default="extended1D", choices=["simple", "extrapolate", "extended1D", "extended2D"])
parser.add_argument("--fakeSmoothingMode", type=str, default="full", choices=["binned", "fakerate", "hybrid", "full"], help="Smoothing mode for fake estimate.")
parser.add_argument("--fakeMCCorr", type=str, default=[None], nargs="*", choices=["none", "pt", "eta", "mt"], help="axes to apply nonclosure correction from QCD MC. Leave empty for inclusive correction, use'none' for no correction")
parser.add_argument("--forceGlobalScaleFakes", default=None, type=float, help="Scale the fakes  by this factor (overriding any custom one implemented in datagroups.py in the fakeSelector).")
parser.add_argument("--fakeSmoothingOrder", type=int, default=3, help="Order of the polynomial for the smoothing of the application region or full prediction, depending on the smoothing mode")
parser.add_argument("--fakerateAxes", nargs="+", help="Axes for the fakerate binning", default=["eta","pt","charge"])
parser.add_argument("--fineGroups", action='store_true', help="Plot each group as a separate process, otherwise combine groups based on predefined dictionary")

subparsers = parser.add_subparsers(dest="variation")
variation = subparsers.add_parser("variation", help="Arguments for adding variation hists")
variation.add_argument("--varName", type=str, nargs='+', required=True, help="Name of variation hist")
variation.add_argument("--varLabel", type=str, nargs='+', required=True, help="Label(s) of variation hist for plotting")
variation.add_argument("--selectAxis", type=str, nargs='+', help="If you need to select a variation axis")
variation.add_argument("--selectEntries", type=str, nargs='+', help="entries to read from the selected axis")
variation.add_argument("--colors", type=str, nargs='+', help="Variation colors")
variation.add_argument("--linestyle", type=str, default=[], nargs='+', help="Linestyle for variations")
variation.add_argument("--doubleColors", action='store_true', help="Auto generate colors in pairs (useful for systematics)")
variation.add_argument("--fillBetween", type=int, help="Fill between first n variation hists in ratio")

args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

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

if not args.fineGroups:
    if groups.mode in styles.process_supergroups:
        for new_name, old_groups in styles.process_supergroups[groups.mode].items():
            groups.mergeGroups(old_groups, new_name)
    else:
        logger.warning(f"No supergroups found for input file with mode {groups.mode}, proceed without merging groups")

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

if args.axlim or args.rebin or args.absval:
    logger.info("Rebin")
    groups.set_rebin_action(args.hists[0].split("-"), args.axlim, args.rebin, args.absval, args.rebinBeforeSelection)

if args.selection:
    applySelection=False
    if args.selection != "none":
        translate = {"hist.overflow": hist.overflow, "hist.underflow":hist.underflow, "hist.sum":hist.sum}
        for selection in args.selection.split(","):
            axis, value = selection.split("=")
            if value.startswith("["):
                parts = [translate[p] if p in translate else int(p) if p!=str() else None for p in value[1:-1].split(":")]
                select[axis] = hist.tag.Slicer()[parts[0]:parts[1]:parts[2]]
            elif value=="hist.overflow":
                select[axis] = hist.overflow
            elif value=="hist.underflow":
                select[axis] = hist.overflow
            else:
                select[axis] = int(value)
else:
    applySelection=True

groups.fakerate_axes=args.fakerateAxes
if applySelection:
    groups.set_histselectors(
        datasets,
        args.baseName,
        smoothing_mode=args.fakeSmoothingMode,
        smoothingOrderSpectrum=args.fakeSmoothingOrder,
        integrate_x=all("mt" not in x.split("-") for x in args.hists),
        mode=args.fakeEstimation,
        forceGlobalScaleFakes=args.forceGlobalScaleFakes,
        mcCorr=args.fakeMCCorr,
        )

if not args.nominalRef:
    nominalName = args.baseName.rsplit("_", 1)[0]
    groups.setNominalName(nominalName)
    groups.loadHistsForDatagroups(args.baseName, syst="", procsToRead=datasets, applySelection=applySelection)
else:
    nominalName = args.nominalRef
    groups.setNominalName(nominalName)
    groups.loadHistsForDatagroups(nominalName, syst=args.baseName, procsToRead=datasets, applySelection=applySelection)

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
        requiresNominal = False

        if entry and entry.isdigit():
            entry = int(entry)

        if args.selectAxis or do_transform:
            transform_procs = groups.getProcNames(exclude_group=exclude)
            if do_transform:
                tmap = transforms[entry]
                action = tmap["action"]
                if "procs" in transforms[entry]:
                    transform_procs = tmap["procs"]
                varname = entry
                requiresNominal = tmap.get("requiresNominal", False)
            else:
                ax = axes[i]
                action = lambda x: x[{ax : entry}] if ax in x.axes.name else x
                varname = name+str(entry)

            if not requiresNominal:
                load_op = {p : action for p in transform_procs}
        else:
            varname = name

        reload = name != args.baseName
        # The action map will only work if reloading, otherwise need to apply some transform
        # to the already loaded hist
        if load_op and reload:
            action = None
        groups.addSummedProc(nominalName, relabel=args.baseName, name=name, label=label, exclude=exclude,
            color=color, reload=reload, rename=varname, procsToRead=datasets, actionRequiresRef=requiresNominal,
            preOpMap=load_op, action=action, forceNonzero=False, applySelection=applySelection)

        exclude.append(varname)
        unstack.append(varname)

groups.sortByYields(args.baseName, nominalName=nominalName)
histInfo = groups.getDatagroups()

logger.info(f"Unstacked processes are {exclude}")
prednames = list(reversed(groups.getNames([d for d in datasets if d not in exclude], exclude=False, match_exact=True)))
logger.info(f"Stacked processes are {prednames}")

def collapseSyst(h):
    if type(h.axes[-1]) == hist.axis.StrCategory:
        return h[...,0]
    for ax in ["systIdx", "tensor_axis_0", "vars", "pdfVar"]:
        if ax in h.axes.name:
            return h[{ax : 0}].copy()
    return h

overflow_ax = ["ptll", "chargeVgen", "massVgen", "ptVgen", "absEtaGen", "ptGen", "ptVGen", "absYVGen", "iso", "dxy", "met","mt"]
for h in args.hists:
    if any(x in h.split("-") for x in ["ptll", "mll", "ptVgen", "ptVGen"]):
        # in case of variable bin width normalize to unit
        binwnorm = 1.0
        ylabel="Events/unit"
    else:
        binwnorm = None
        ylabel="Events/bin"
    if len(h.split("-")) > 1:
        sp = h.split("-")
        action = lambda x: hh.unrolledHist(collapseSyst(x[select]), binwnorm=binwnorm, obs=sp)
        xlabel=f"{'-'.join([styles.xlabels.get(s,s).replace('(GeV)','') for s in sp])} bin"
    else:
        action = lambda x: hh.projectNoFlow(collapseSyst(x[select]), h, overflow_ax)
        xlabel=styles.xlabels.get(h,h)
    fig = plot_tools.makeStackPlotWithRatio(histInfo, prednames, histName=args.baseName, ylim=args.ylim, yscale=args.yscale, logy=args.logy,
            fill_between=args.fillBetween if hasattr(args, "fillBetween") else None, 
            action=action, unstacked=unstack, 
            fitresult=args.fitresult, prefit=args.prefit,
            xlabel=xlabel, ylabel=ylabel, rrange=args.rrange, binwnorm=binwnorm, lumi=groups.lumi,
            ratio_to_data=args.ratioToData, rlabel="Pred./Data" if args.ratioToData else "Data/Pred.",
            xlim=args.xlim, no_fill=args.noFill, no_stack=args.noStack, no_ratio=args.noRatio, density=args.density, flow=args.flow,
            cms_decor=args.cmsDecor, legtext_size=20*args.scaleleg, unstacked_linestyles=args.linestyle if hasattr(args, "linestyle") else [],
            ratio_error=args.ratioError, normalize_to_data=args.normToData)

    fitresultstring=""
    if args.fitresult:
        fitresultstring = "prefit" if args.prefit else "postfit"
    var_arg = None
    if "varName" in args and args.varName:
        var_arg = args.varName[0]
        if "selectEntries" in args and args.selectEntries:
            var_arg = args.selectEntries[0] if not args.selectEntries[0].isdigit() else (var_arg+args.selectEntries[0])
    to_join = [f"{h.replace('-','_')}"]+[var_arg]+[fitresultstring, args.postfix]+[args.channel.replace("all", "")]
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
    output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)
