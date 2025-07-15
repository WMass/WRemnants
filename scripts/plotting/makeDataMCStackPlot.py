import hist
from matplotlib import colormaps

from utilities import parsing
from utilities.styles import styles
from wremnants import syst_tools
from wremnants.datasets.datagroups import Datagroups
from wremnants.histselections import FakeSelectorSimpleABCD
from wremnants.regression import Regressor
from wums import boostHistHelpers as hh
from wums import logging, output_tools, plot_tools

parser = parsing.plot_parser()
parser.add_argument(
    "infile", help="Output file of the analysis stage, containing ND boost histograms"
)
parser.add_argument(
    "--ratioToData", action="store_true", help="Use data as denominator in ratio"
)
parser.add_argument(
    "-n",
    "--baseName",
    type=str,
    help="Histogram name in the file (e.g., 'nominal')",
    default="nominal",
)
parser.add_argument(
    "--nominalRef",
    type=str,
    help="Specify the nominal hist if baseName is a variation hist (for plotting alt hists)",
)
parser.add_argument(
    "--hists", type=str, nargs="+", required=True, help="List of histograms to plot"
)
parser.add_argument(
    "-c",
    "--channel",
    type=str,
    choices=["plus", "minus", "all"],
    default="all",
    help="Select channel to plot",
)
parser.add_argument(
    "--rebin",
    type=int,
    nargs="*",
    default=[],
    help="Rebin axis by this value (default, 1, does nothing)",
)
parser.add_argument(
    "--absval",
    type=int,
    nargs="*",
    default=[],
    help="Take absolute value of axis if 1 (default, 0, does nothing)",
)
parser.add_argument(
    "--axlim",
    type=float,
    default=[],
    nargs="*",
    help="Restrict axis to this range (assumes pairs of values by axis, with trailing axes optional)",
)
parser.add_argument(
    "--rebinBeforeSelection",
    action="store_true",
    help="Rebin before the selection operation (e.g. before fake rate computation), default if after",
)
parser.add_argument("--logy", action="store_true", help="Enable log scale for y axis")
parser.add_argument(
    "--procFilters",
    type=str,
    nargs="*",
    help="Filter to plot (default no filter, only specify if you want a subset",
)
parser.add_argument("--noData", action="store_true", help="Don't plot data")
parser.add_argument("--noFill", action="store_true", help="Don't fill")
parser.add_argument("--noStack", action="store_true", help="Don't stack")
parser.add_argument("--noRatio", action="store_true", help="Don't make ratio plot")
parser.add_argument(
    "--density",
    action="store_true",
    help="Normalize each process to unity, only works with '--noStack'",
)
parser.add_argument(
    "--flow",
    type=str,
    choices=["show", "sum", "hint", "none"],
    default="none",
    help="Whether plot the under/overflow bin",
)
parser.add_argument(
    "--noRatioErr",
    action="store_false",
    dest="ratioError",
    help="Don't show stat unc in ratio",
)
parser.add_argument(
    "--rlabel", type=str, default=None, help="Ratio y-axis label for plot labeling"
)
parser.add_argument(
    "--selection",
    type=str,
    help="Specify custom selections as comma seperated list (e.g. '--selection passIso=0,passMT=[2::hist.sum]' )",
)
parser.add_argument(
    "--presel",
    type=str,
    nargs="*",
    default=[],
    help="Specify custom selections on input histograms to integrate some axes, giving axis name and min,max (e.g. '--presel pt=ptmin,ptmax' ) or just axis name for bool axes",
)
parser.add_argument("--normToData", action="store_true", help="Normalize MC to data")
parser.add_argument(
    "--fakeEstimation",
    type=str,
    help="Set the mode for the fake estimation",
    default="extended1D",
    choices=["simple", "extrapolate", "extended1D", "extended2D"],
)
parser.add_argument(
    "--fakeMCCorr",
    type=str,
    default=[None],
    nargs="*",
    choices=["none", "pt", "eta", "mt"],
    help="axes to apply nonclosure correction from QCD MC. Leave empty for inclusive correction, use'none' for no correction",
)
parser.add_argument(
    "--forceGlobalScaleFakes",
    default=None,
    type=float,
    help="Scale the fakes  by this factor (overriding any custom one implemented in datagroups.py in the fakeSelector).",
)
parser.add_argument(
    "--fakeSmoothingMode",
    type=str,
    default="full",
    choices=FakeSelectorSimpleABCD.smoothing_modes,
    help="Smoothing mode for fake estimate.",
)
parser.add_argument(
    "--fakeSmoothingOrder",
    type=int,
    default=3,
    help="Order of the polynomial for the smoothing of the application region or full prediction, depending on the smoothing mode",
)
parser.add_argument(
    "--fakeSmoothingPolynomial",
    type=str,
    default="chebyshev",
    choices=Regressor.polynomials,
    help="Order of the polynomial for the smoothing of the application region or full prediction, depending on the smoothing mode",
)
parser.add_argument(
    "--fakerateAxes",
    nargs="+",
    help="Axes for the fakerate binning",
    default=["eta", "pt", "charge"],
)
parser.add_argument(
    "--fineGroups",
    action="store_true",
    help="Plot each group as a separate process, otherwise combine groups based on predefined dictionary",
)
parser.add_argument(
    "--subplotSizes",
    nargs=2,
    type=int,
    default=[4, 2],
    help="Relative sizes for upper and lower panels",
)
parser.add_argument(
    "--scaleRatioUnstacked",
    nargs="*",
    type=float,
    default=[],
    help="Scale a variation by this factor",
)
parser.add_argument(
    "--extraTextLoc",
    type=float,
    nargs=2,
    default=(0.05, 0.8),
    help="Location in (x,y) for additional text, aligned to upper left",
)
subparsers = parser.add_subparsers(dest="variation")
variation = subparsers.add_parser(
    "variation", help="Arguments for adding variation hists"
)
variation.add_argument(
    "--varName", type=str, nargs="+", required=True, help="Name of variation hist"
)
variation.add_argument(
    "--varLabel",
    type=str,
    nargs="*",
    default=None,
    help="Label(s) of variation hist for plotting",
)
variation.add_argument(
    "--selectAxis", type=str, nargs="+", help="If you need to select a variation axis"
)
variation.add_argument(
    "--selectEntries",
    type=str,
    nargs="+",
    help="entries to read from the selected axis",
)
variation.add_argument("--colors", type=str, nargs="+", help="Variation colors")
variation.add_argument(
    "--linestyle", type=str, default=[], nargs="+", help="Linestyle for variations"
)
variation.add_argument(
    "--doubleColors",
    action="store_true",
    help="Auto generate colors in pairs (useful for systematics)",
)
variation.add_argument(
    "--doubleLines",
    action="store_true",
    help="Auto generate colors in pairs (useful for systematics)",
)
variation.add_argument(
    "--fillBetween", type=int, help="Fill between first n variation hists in ratio"
)
variation.add_argument(
    "--lowerPanelVariations",
    type=int,
    default=0,
    help="Plot n first variations in lower panel only",
)

args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)


def padArray(ref, matchLength):
    return ref + ref[-1:] * (len(matchLength) - len(ref))


addVariation = hasattr(args, "varName") and args.varName is not None

entries = []
varLabels = args.varLabel if addVariation else []
if addVariation and (args.selectAxis or args.selectEntries):
    if not (args.selectAxis and args.selectEntries):
        raise ValueError("Must --selectAxis and --selectEntries together")
    if len(args.varName) < len(args.selectEntries):
        args.varName = padArray(args.varName, args.selectEntries)
    axes = padArray(args.selectAxis, args.varName)
    entries = padArray(args.selectEntries, args.varName)

    if args.varLabel is None:
        # try to get labels from predefined styles
        varLabels = [styles.legend_labels.get(e, e) for e in entries]
    elif len(args.varLabel) != 1 and len(args.varLabel) != len(args.selectEntries):
        raise ValueError(
            "Must specify the same number of args for --selectEntries, and --varLabel"
            f" found selectEntries={len(args.selectEntries)} and varLabel={len(args.varLabel)}"
        )

outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)

groups = Datagroups(
    args.infile,
    filterGroups=args.procFilters,
    excludeGroups=None if args.procFilters else ["QCD"],
)

if not args.fineGroups:
    if groups.mode in styles.process_supergroups:
        for new_name, old_groups in styles.process_supergroups[groups.mode].items():
            groups.mergeGroups(old_groups, new_name)
    else:
        logger.warning(
            f"No supergroups found for input file with mode {groups.mode}, proceed without merging groups"
        )

# There is probably a better way to do this but I don't want to deal with it
datasets = groups.getNames()
logger.info(f"Will plot datasets {datasets}")

select = (
    {}
    if args.channel == "all"
    else {"charge": -1.0j if args.channel == "minus" else 1.0j}
)

if len(args.presel):
    s = hist.tag.Slicer()
    presel = {}
    logger.debug(args.presel)
    logger.debug(f"Will apply the global preselection")
    for ps in args.presel:
        if "=" in ps:
            axName, axRange = ps.split("=")
            axMin, axMax = map(float, axRange.split(","))
            logger.info(f"{axName} in [{axMin},{axMax}]")
            presel[axName] = s[complex(0, axMin) : complex(0, axMax) : hist.sum]
        else:
            logger.info(f"Integrating boolean {ps} axis")
            presel[ps] = s[:: hist.sum]
    groups.setGlobalAction(lambda h: h[presel])

if args.axlim or args.rebin or args.absval:
    logger.info("Rebin")
    groups.set_rebin_action(
        args.hists[0].split("-"),
        args.axlim,
        args.rebin,
        args.absval,
        args.rebinBeforeSelection,
    )

if args.selection:
    applySelection = False
    if args.selection != "none":
        translate = {
            "hist.overflow": hist.overflow,
            "hist.underflow": hist.underflow,
            "hist.sum": hist.sum,
        }
        for selection in args.selection.split(","):
            axis, value = selection.split("=")
            if value.startswith("["):
                parts = [
                    translate[p] if p in translate else int(p) if p != str() else None
                    for p in value[1:-1].split(":")
                ]
                select[axis] = hist.tag.Slicer()[parts[0] : parts[1] : parts[2]]
            elif value == "hist.overflow":
                select[axis] = hist.overflow
            elif value == "hist.underflow":
                select[axis] = hist.overflow
            else:
                select[axis] = int(value)
else:
    applySelection = True

groups.fakerate_axes = args.fakerateAxes
if applySelection:
    groups.set_histselectors(
        datasets,
        args.baseName,
        smoothing_mode=args.fakeSmoothingMode,
        smoothingOrderSpectrum=args.fakeSmoothingOrder,
        smoothingPolynomialSpectrum=args.fakeSmoothingPolynomial,
        integrate_x=all("mt" not in x.split("-") for x in args.hists),
        mode=args.fakeEstimation,
        forceGlobalScaleFakes=args.forceGlobalScaleFakes,
        mcCorr=args.fakeMCCorr,
    )

if not args.nominalRef:
    nominalName = args.baseName.rsplit("_", 1)[0]
    groups.setNominalName(nominalName)
    groups.loadHistsForDatagroups(
        args.baseName, syst="", procsToRead=datasets, applySelection=applySelection
    )
else:
    nominalName = args.nominalRef
    groups.setNominalName(nominalName)
    groups.loadHistsForDatagroups(
        nominalName,
        syst=args.baseName,
        procsToRead=datasets,
        applySelection=applySelection,
    )

exclude = ["Data"]
unstack = exclude[:]
if args.noData:
    unstack.remove("Data")

# TODO: In should select the correct hist for the transform, not just the first
transforms = syst_tools.syst_transform_map(nominalName, args.hists[0])

if addVariation:
    logger.info(f"Adding variation {args.varName}")
    varLabels = padArray(varLabels, args.varName)
    # If none matplotlib will pick a random color
    ncols = len(args.varName) if not args.doubleColors else int(len(args.varName) / 2)
    colors = (
        args.colors
        if args.colors
        else [
            colormaps["tab10" if ncols < 10 else "tab20"](
                int(i / 2) if args.doubleColors else i
            )
            for i in range(len(args.varName))
        ]
    )
    for i, (label, name, color) in enumerate(zip(varLabels, args.varName, colors)):
        entry = entries[i] if entries else None
        do_transform = entry in transforms
        name = name if name != "" else nominalName
        load_op = {}
        action = None
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
                action = lambda x: x[{ax: entry}] if ax in x.axes.name else x
                varname = name + str(entry)

            if not requiresNominal:
                load_op = {p: action for p in transform_procs}
        else:
            varname = name

        reload = name != args.baseName
        # The action map will only work if reloading, otherwise need to apply some transform
        # to the already loaded hist
        if load_op and reload:
            action = None
        groups.addSummedProc(
            nominalName,
            relabel=args.baseName,
            name=name,
            label=label,
            exclude=exclude,
            color=color,
            reload=reload,
            rename=varname,
            procsToRead=datasets,
            actionRequiresRef=requiresNominal,
            preOpMap=load_op,
            action=action,
            forceNonzero=False,
            applySelection=applySelection,
        )

        exclude.append(varname)
        unstack.append(varname)

groups.sortByYields(args.baseName, nominalName=nominalName)
histInfo = groups.groups

logger.info(f"Unstacked processes are {exclude}")
prednames = list(
    reversed(
        groups.getNames(
            [d for d in datasets if d not in exclude], exclude=False, match_exact=True
        )
    )
)
logger.info(f"Stacked processes are {prednames}")

text_pieces = []
if args.normToData:
    text_pieces.append("Prefit" + " (normalized)")
else:
    text_pieces.append("Prefit")

if args.channel != "all":
    text_pieces.append(
        r"$\mathit{q}^\mu$ = " + ("+1" if args.channel == "plus" else "-1")
    )


def collapseSyst(h):
    if type(h.axes[-1]) == hist.axis.StrCategory:
        return h[..., 0]
    for ax in ["systIdx", "tensor_axis_0", "vars", "pdfVar"]:
        if ax in h.axes.name:
            return h[{ax: 0}].copy()
    return h


overflow_ax = [
    "ptll",
    "chargeVgen",
    "massVgen",
    "ptVgen",
    "absEtaGen",
    "ptGen",
    "ptVGen",
    "absYVGen",
    "iso",
    "dxy",
    "met",
    "mt",
]
for h in args.hists:
    if any(
        x in h.split("-")
        for x in ["pt", "ptll", "mll", "ptW", "ptVgen", "ptVGen", "ptWgen", "ptZgen"]
    ):
        # in case of variable bin width normalize to unit (which is GeV for all of these...)
        binwnorm = 1.0
        ylabel = r"$Events\,/\,GeV$"
    else:
        binwnorm = None
        ylabel = r"$Events\,/\,bin$"

    if args.rlabel is None:
        if args.noData:
            rlabel = "Ratio to nominal"
        elif args.ratioToData:
            rlabel = r"$Pred.\,/\,Data$"
        else:
            rlabel = r"$Data\,/\,Pred.$"
    else:
        rlabel = args.rlabel

    sp = h.split("-")
    xlabel = plot_tools.get_axis_label(styles, sp)
    if len(sp) > 1:
        base_action = lambda x: collapseSyst(x[select])
        action = lambda x: hh.unrolledHist(base_action(x), binwnorm=binwnorm, obs=sp)
    else:
        base_action = lambda x: hh.projectNoFlow(
            collapseSyst(x[select]), h, overflow_ax
        )
        action = base_action
        href = h if h != "ptVgen" else ("ptWgen" if "Wmunu" in prednames else "ptZgen")

    if groups.flavor in ["e", "ee"]:
        xlabel = xlabel.replace(r"\mu", "e")

    fig = plot_tools.makeStackPlotWithRatio(
        histInfo,
        prednames,
        histName=args.baseName,
        ylim=args.ylim,
        yscale=args.yscale,
        logy=args.logy,
        fill_between=args.fillBetween if hasattr(args, "fillBetween") else None,
        lower_panel_variations=(
            args.lowerPanelVariations if hasattr(args, "lowerPanelVariations") else 0
        ),
        scaleRatioUnstacked=args.scaleRatioUnstacked,
        action=action,
        unstacked=unstack,
        xlabel=xlabel,
        ylabel=ylabel,
        rrange=args.rrange,
        binwnorm=binwnorm,
        lumi=groups.lumi,
        ratio_to_data=args.ratioToData,
        rlabel=rlabel,
        xlim=args.xlim,
        no_fill=args.noFill,
        no_stack=args.noStack,
        no_ratio=args.noRatio,
        extra_text=text_pieces,
        extra_text_loc=args.extraTextLoc,
        density=args.density,
        flow=args.flow,
        cms_decor=args.cmsDecor,
        legtext_size=args.legSize,
        nlegcols=args.legCols,
        unstacked_linestyles=args.linestyle if hasattr(args, "linestyle") else [],
        double_lines=args.doubleLines if hasattr(args, "doubleLines") else False,
        ratio_error=args.ratioError,
        normalize_to_data=args.normToData,
        noSci=args.noSciy,
        logoPos=args.logoPos,
        width_scale=1.25 if len(h.split("-")) == 1 else 1,
        legPos=args.legPos,
        leg_padding=args.legPadding,
        lowerLegCols=args.lowerLegCols,
        lowerLegPos=args.lowerLegPos,
        lower_leg_padding=args.lowerLegPadding,
        subplotsizes=args.subplotSizes,
    )

    to_join = [f"{h.replace('-','_')}"]
    if "varName" in args and args.varName:
        var_arg = args.varName[0]
        if "selectEntries" in args and args.selectEntries:
            var_arg = (
                args.selectEntries[0]
                if not args.selectEntries[0].isdigit()
                else (var_arg + args.selectEntries[0])
            )
        to_join.append(var_arg)
    to_join.extend([args.postfix, args.channel.replace("all", "")])
    outfile = "_".join(filter(lambda x: x, to_join))
    if args.cmsDecor == "Preliminary":
        outfile += "_preliminary"
    print("OUT", outfile)

    plot_tools.save_pdf_and_png(outdir, outfile)

    stack_yields = groups.make_yields_df(
        args.baseName, prednames, norm_proc="Data", action=base_action
    )
    unstacked_yields = groups.make_yields_df(
        args.baseName, unstack, norm_proc="Data", action=base_action
    )
    output_tools.write_index_and_log(
        outdir,
        outfile,
        analysis_meta_info={
            "Stacked processes": stack_yields,
            "Unstacked processes": unstacked_yields,
            "AnalysisOutput": groups.getMetaInfo(),
        },
        args=args,
    )

if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
    output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)
