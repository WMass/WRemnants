import mplhep as hep
import matplotlib as mpl
import matplotlib.pyplot as plt

import itertools
import argparse
import os
import numpy as np
import pandas as pd
import hist
import json

from utilities import boostHistHelpers as hh, common, logging
from utilities.styles import styles
from utilities.io_tools import input_tools, output_tools
from utilities.io_tools.conversion_tools import fitresult_pois_to_hist
from wremnants.datasets.datagroups import Datagroups
from wremnants import plot_tools
import pdb

hep.style.use(hep.style.ROOT)

parser = common.plot_parser()
parser.add_argument("infile", type=str, help="Combine fitresult file")
parser.add_argument("--reference",  type=str, default=None, help="Optional combine fitresult file from an reference fit for comparison")
parser.add_argument("--refName",  type=str, default="Reference model", help="Name for reference source")
parser.add_argument("-r", "--rrange", type=float, nargs=2, default=[0.9,1.1], help="y range for ratio plot")
parser.add_argument("--ylim", type=float, nargs=2, help="Min and max values for y axis (if not specified, range set automatically)")
parser.add_argument("--logy", action='store_true', help="Make the yscale logarithmic")
parser.add_argument("--yscale", type=float, help="Scale the upper y axis by this factor (useful when auto scaling cuts off legend)")
parser.add_argument("--noData", action='store_true', help="Don't plot data")
parser.add_argument("--plots", type=str, nargs="+", default=["xsec", "uncertainties"], choices=["xsec", "uncertainties", "ratio"], help="Define which plots to make")
parser.add_argument("--selectionAxes", type=str, default=["qGen", ], 
    help="List of axes where for each bin a seperate plot is created")
parser.add_argument("--genFlow", action='store_true', help="Show overflow/underflow pois")
parser.add_argument("--poiTypes", type=str, nargs="+", default=["pmaskedexp", "sumpois"], help="POI types used for the plotting",
    choices=["noi", "mu", "pmaskedexp", "pmaskedexpnorm", "sumpois", "sumpoisnorm", ])
parser.add_argument("--scaleXsec", type=float, default=1.0, help="Scale xsec predictions with this number")
parser.add_argument("--grouping", type=str, default=None, help="Select nuisances by a predefined grouping", choices=styles.nuisance_groupings.keys())
parser.add_argument("--ratioToPred", action='store_true', help="Use prediction as denominator in ratio")
parser.add_argument("-t","--translate", type=str, default=None, help="Specify .json file to translate labels")

# alternaitve predictions from histmaker output
parser.add_argument("--histfile", type=str, help="Histogramer output file for comparisons")
parser.add_argument("-n", "--baseName", type=str, help="Histogram name in the file (e.g., 'xnorm', 'nominal', ...)", default="xnorm")
parser.add_argument("--varNames", default=['uncorr'], type=str, nargs='+', help="Name of variation hist")
parser.add_argument("--varLabels", default=['MiNNLO'], type=str, nargs='+', help="Label(s) of variation hist for plotting")
parser.add_argument("--selectAxis", default=[], type=str, nargs='*', help="If you need to select a variation axis")
parser.add_argument("--selectEntries", default=[], type=str, nargs='*', help="entries to read from the selected axis")
parser.add_argument("--colors", default=['red',], type=str, nargs='+', help="Variation colors")

args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

if args.infile.endswith(".root"):
    args.infile = args.infile.replace(".root", ".hdf5")
if args.reference is not None and args.reference.endswith(".root"):
    args.reference = args.reference.replace(".root", ".hdf5")

result, meta = fitresult_pois_to_hist(args.infile, poi_types=args.poiTypes, uncertainties=None, translate_poi_types=False)
if args.reference:
    result_ref, meta_ref = fitresult_pois_to_hist(args.reference, poi_types=args.poiTypes, uncertainties=None, translate_poi_types=False)

grouping = styles.nuisance_groupings.get(args.grouping, None)

translate_label = {}
if args.translate:
    with open(args.translate) as f:
        translate_label = json.load(f)    

outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)

if args.histfile:
    groups = Datagroups(args.histfile)
    groups.setNominalName(args.baseName)
    groups.setGenAxes([],[])
    groups.lumi=0.001
    if "Wmunu" in groups.groups.keys():
        groups.copyGroup("Wmunu", "Wminus", member_filter=lambda x: x.name.startswith("Wminus") and not x.name.endswith("OOA"))
        groups.copyGroup("Wmunu", "Wplus", member_filter=lambda x: x.name.startswith("Wplus") and not x.name.endswith("OOA"))

proc_dict = {"W_qGen0": "W^{-}", "W_qGen1": "W^{+}"}
def get_xlabel(names, proc=""):
    if len(names) > 1:
        label = f"{'-'.join([styles.xlabels.get(n,n).replace('(GeV)','') for n in names])} bin"
    else:
        label = styles.xlabels.get(names[0], names[0])
    if proc:
        label = label.replace("\mathrm{V}", "\mathrm{"+proc_dict.get(proc, proc[0])+"}")
    return label

def make_yields_df(hists, procs, signal=None, per_bin=False, yield_only=False, percentage=True):
    logger.debug(f"Make yield df for {procs}")
    if per_bin:
        def sum_and_unc(h,scale=100 if percentage else 1):
            return (h.values()*scale, np.sqrt(h.variances())*scale)   
    else:
        def sum_and_unc(h,scale=100 if percentage else 1):
            return (sum(h.values())*scale, np.sqrt(sum(h.variances())*scale))
    if per_bin:
        entries = [(i, v[0], v[1]) for i,v in enumerate(zip(*sum_and_unc(hists[0])))]
        index = "Bin"
    else:
        index = "Process"
        if signal is not None:
            entries = [(signal, sum([ sum(v.values()) for k,v in zip(procs, hists) if signal in k]), np.sqrt(sum([ sum(v.variances()) for k,v in zip(procs, hists) if signal in k])))]
        else:
            entries = [(k, *sum_and_unc(v)) for k,v in zip(procs, hists)]
    if yield_only:
        entries = [(e[0], e[1]) for e in entries]
        columns = [index, *procs]
    else:
        columns = [index, "Yield", "Uncertainty"]
    return pd.DataFrame(entries, columns=columns)

def plot_xsec_unfolded(hist_xsec, hist_xsec_stat=None, hist_ref=None, poi_type="mu", channel="ch0", proc="W",
    hist_others=[], label_others=[], color_others=[], lumi=None
):
    # ratio to data if there is no other histogram to make a ratio from
    ratioToData = not args.ratioToPred or len(hist_others) == 0

    logger.info(f"Make unfoled {poi_type} plot in channel {channel}")

    yLabel = styles.poi_types[poi_type]

    # unroll histograms
    binwnorm = 1 if poi_type not in ["noi", "mu",] else None
    axes_names = hist_xsec.axes.name
    if len(axes_names) > 1:
        hist_xsec = hh.unrolledHist(hist_xsec, binwnorm=binwnorm, add_flow_bins=args.genFlow)
        hist_others = [hh.unrolledHist(h, binwnorm=binwnorm, add_flow_bins=args.genFlow) for h in hist_others]
        if hist_ref is not None:
            hist_ref = hh.unrolledHist(hist_ref, binwnorm=binwnorm, add_flow_bins=args.genFlow)
        if hist_xsec_stat is not None:
            hist_xsec_stat = hh.unrolledHist(hist_xsec_stat, binwnorm=binwnorm, add_flow_bins=args.genFlow)

    xlabel = get_xlabel(axes_names, proc)

    edges = hist_xsec.axes.edges[0]
    binwidths = np.diff(edges)

    if ratioToData:
        rlabel="Pred./Data"
    else:
        rlabel=f"Data/{label_others[0]}"
    rrange = args.rrange

    if args.ylim is None:
        ylim = (0, 1.1 * max( (hist_xsec.values()+np.sqrt(hist_xsec.variances()))/(binwidths if binwnorm is not None else 1)) )
    else:
        ylim = args.ylim

    # make plots
    fig, ax1, ax2 = plot_tools.figureWithRatio(hist_xsec, xlabel, yLabel, ylim, rlabel, rrange, width_scale=2)

    hep.histplot(
        hist_xsec,
        yerr=np.sqrt(hist_xsec.variances()),
        histtype="errorbar",
        color="black",
        label="Unfolded data",
        ax=ax1,
        alpha=1.,
        zorder=2,
        binwnorm=binwnorm
    )    

    if args.genFlow: #TODO TEST
        ax1.fill([0,18.5, 18.5, 0,0], [ylim[0],*ylim,ylim[1],ylim[0]], color="grey", alpha=0.3)
        ax1.fill([len(edges)-17.5, len(edges)+0.5, len(edges)+0.5, len(edges)-17.5, len(edges)-17.5], [ylim[0],*ylim,ylim[1],ylim[0]], color="grey", alpha=0.3)

        ax2.fill([0,18.5, 18.5, 0,0], [rrange[0],*rrange,rrange[1],rrange[0]], color="grey", alpha=0.3)
        ax2.fill([len(edges)-17.5, len(edges)+0.5, len(edges)+0.5, len(edges)-17.5, len(edges)-17.5], [rrange[0],*rrange,rrange[1],rrange[0]], color="grey", alpha=0.3)

    centers = hist_xsec.axes.centers[0]

    if ratioToData:
        unc_ratio = np.sqrt(hist_xsec.variances()) /hist_xsec.values() 
        ax2.bar(centers, height=2*unc_ratio, bottom=1-unc_ratio, width=edges[1:] - edges[:-1], color="silver", label="Total")
        if hist_xsec_stat is not None:
            unc_ratio_stat = np.sqrt(hist_xsec_stat.variances()) / hist_xsec_stat.values() 
            ax2.bar(centers, height=2*unc_ratio_stat, bottom=1-unc_ratio_stat, width=edges[1:] - edges[:-1], color="gold", label="Stat")

    ax2.plot([min(edges), max(edges)], [1,1], color="black", linestyle="-")

    if ratioToData:
        hden=hist_xsec

    for i, (h, l, c) in enumerate(zip(hist_others, label_others, color_others)):
        # h_flat = hist.Hist(
        #     hist.axis.Variable(edges, underflow=False, overflow=False), storage=hist.storage.Weight())
        # h_flat.view(flow=False)[...] = np.stack([h.values(flow=args.genFlow).flatten()/bin_widths, h.variances(flow=args.genFlow).flatten()/bin_widths**2], axis=-1)

        hep.histplot(
            h,
            yerr=False,
            histtype="step",
            color=c,
            label=l,
            ax=ax1,
            alpha=1.,
            zorder=2,
            binwnorm=binwnorm
        )            

        if i==0 and not ratioToData:
            hden=h
            hep.histplot(
                hh.divideHists(h, hden, cutoff=0, rel_unc=True),
                yerr=True,
                histtype="errorbar",
                color="black",
                ax=ax2,
                zorder=2,
                binwnorm=binwnorm
            ) 
            continue

        hep.histplot(
            hh.divideHists(h, hden, cutoff=0, rel_unc=True),
            yerr=False,
            histtype="step",
            color=c,
            ax=ax2,
            zorder=2,
            binwnorm=binwnorm
        )            

    if hist_ref is not None:
        hep.histplot(
            hist_ref,
            yerr=False,
            histtype="step",
            color="blue",
            label=args.refName,
            ax=ax1,
            alpha=1.,
            zorder=2,
            binwnorm=binwnorm
        ) 

        hep.histplot(
            hh.divideHists(hist_ref, hden, cutoff=0, rel_unc=True),
            yerr=False,
            histtype="step",
            color="blue",
            # label="Model",
            ax=ax2,
            binwnorm=binwnorm
        )

    plot_tools.addLegend(ax1, ncols=2, text_size=15*args.scaleleg)
    plot_tools.addLegend(ax2, ncols=2, text_size=15*args.scaleleg)
    plot_tools.fix_axes(ax1, ax2, yscale=args.yscale)

    scale = max(1, np.divide(*ax1.get_figure().get_size_inches())*0.3)
    hep.cms.label(ax=ax1, lumi=float(f"{lumi:.3g}") if lumi is not None else None, fontsize=20*args.scaleleg*scale, 
        label=args.cmsDecor, data=not args.noData)

    outfile = f"unfolded_{poi_type}"
    outfile += f"_{proc}"
    outfile += "_"+"_".join(axes_names)
    outfile += (f"_{channel}" if channel else "")
    outfile += (f"_{args.postfix}" if args.postfix else "")
    plot_tools.save_pdf_and_png(outdir, outfile)

    if hist_ref is not None:
        reference_yields = make_yields_df([hist_ref], ["Model"], per_bin=True)
        reference_yields["Uncertainty"] *= 0 # artificially set uncertainty on model hard coded to 0
    data_yields = make_yields_df([hist_xsec], ["Data"], per_bin=True)
    plot_tools.write_index_and_log(outdir, outfile, nround=2,
        yield_tables={"Data" : data_yields, "Model": reference_yields} if hist_ref is not None else {"Data" : data_yields},
        analysis_meta_info={args.infile : meta["meta_info"]},
        args=args,
    )
    plt.close()

def plot_uncertainties_unfolded(hist_xsec, hist_stat, hist_syst, poi_type, channel="ch0", proc="W",
    logy=False, relative_uncertainty=False, percentage=True, lumi=None,
    error_threshold=0.001,   # only uncertainties are shown with a max error larger than this threshold
):
    logger.info(f"Make unfoled {poi_type} plot"+(f" in channel {channel}" if channel else ""))

    # read nominal values and uncertainties from fit result and fill histograms
    logger.debug(f"Produce histograms")

    yLabel = styles.poi_types[poi_type]
    if relative_uncertainty:
        yLabel = "$\delta$ "+ yLabel
        yLabel = yLabel.replace(" [pb]","")
        if percentage:
            yLabel += " [%]"
    else:
        yLabel = "$\Delta$ "+ yLabel

    axes_names = hist_xsec.axes.name
    xlabel=get_xlabel(axes_names, proc)

    scale = 100 if percentage else 1
    if relative_uncertainty:
        scale = scale / hist_xsec.values()#/binwidths

    err = np.sqrt(hist_xsec.variances(flow=True)) * scale
    err_stat = np.sqrt(hist_stat.variances(flow=True)) * scale
    err_syst = hh.addHists(hist_syst, hist_xsec, scale2=-1).values(flow=True) * (scale[...,np.newaxis] if relative_uncertainty else scale)

    hist_err = hist.Hist(*hist_xsec.axes, storage=hist.storage.Double())
    hist_err_stat = hist.Hist(*hist_stat.axes, storage=hist.storage.Double())
    hist_err_syst = hist.Hist(*hist_syst.axes, storage=hist.storage.Double())

    hist_err.view(flow=True)[...] = err
    hist_err_stat.view(flow=True)[...] = err_stat
    hist_err_syst.view(flow=True)[...] = err_syst

    # unroll histograms
    binwnorm = 1 if poi_type not in ["noi", "mu",] else None
    if len(axes_names) > 1:
        hist_err = hh.unrolledHist(hist_err, binwnorm=binwnorm, add_flow_bins=args.genFlow)
        hist_err_stat = hh.unrolledHist(hist_err_stat, binwnorm=binwnorm, add_flow_bins=args.genFlow)

    if args.ylim is None:
        if logy:
            ylim = (max(hist_err.values())/10000., 1000 * max(hist_err.values()))
        else:
            ylim = (0, 2 * max(hist_err.values()))
    else:
        ylim = args.ylim

    # make plots
    fig, ax1 = plot_tools.figure(hist_err, xlabel, yLabel, ylim, logy=logy, width_scale=2)
    hep.histplot(
        hist_err,
        yerr=False,
        histtype="step",
        color="black",
        label="Total",
        ax=ax1,
        alpha=1.,
        # zorder=2,
    )
    hep.histplot(
        hist_err_stat,
        yerr=False,
        histtype="step",
        color="grey",
        label=translate_label.get("stat", "stat"),
        ax=ax1,
        alpha=1.,
        # zorder=2,
    )
    uncertainties = make_yields_df([hist_err], ["Total", ], per_bin=True, yield_only=True, percentage=percentage)
    uncertainties["stat"] = make_yields_df([hist_err_stat], ["stat"], per_bin=True, yield_only=True, percentage=percentage)["stat"]

    syst_labels = [x for x in hist_err_syst.axes["syst"] if (grouping is None or x in grouping) and x not in ["nominal", "stat", "total"]]
    NUM_COLORS = len(syst_labels) - 2
    cm = mpl.colormaps["gist_rainbow"]
    # add each systematic uncertainty
    i=0
    for syst in syst_labels[::-1]:
        if len(axes_names) > 1:
            hist_err_syst_i = hh.unrolledHist(hist_err_syst[{"syst": syst}], binwnorm=binwnorm, add_flow_bins=args.genFlow)
        else:
            hist_err_syst_i = hist_err_syst[{"syst": syst}]

        if max(hist_err_syst_i.values()) < error_threshold:
            logger.debug(f"Systematic {syst} smaller than threshold of {error_threshold}")
            continue
        name = translate_label.get(syst,syst)
        logger.debug(f"Plot systematic {syst} = {name}")

        if syst =="err_stat":
            color = "grey"
        else:
            color = cm(1.*i/NUM_COLORS)
            i += 1

        if i%3 == 0:
            linestype = "-" 
        elif i%3 == 1:
            linestype = "--" 
        else:
            linestype = ":" 
        
        hep.histplot(
            hist_err_syst_i,
            yerr=False,
            histtype="step",
            color=color,
            linestyle=linestype,
            label=name,
            ax=ax1,
            alpha=1.,
            # zorder=1,
        )

        unc_df = make_yields_df([hist_err_syst_i], [name], per_bin=True, yield_only=True, percentage=True)
        uncertainties[name] = unc_df[name]

    if args.genFlow:
        ax1.fill([0,18.5, 18.5, 0,0], [ylim[0],*ylim,ylim[1],ylim[0]], color="grey", alpha=0.3)
        ax1.fill([hist_err.size-17.5, hist_err.size+0.5, hist_err.size+0.5, hist_err.size-17.5, hist_err.size-17.5], [ylim[0],*ylim,ylim[1],ylim[0]], color="grey", alpha=0.3)

    scale = max(1, np.divide(*ax1.get_figure().get_size_inches())*0.3)

    plot_tools.addLegend(ax1, ncols=4, text_size=15*args.scaleleg*scale, loc="upper left")

    if args.yscale:
        ymin, ymax = ax1.get_ylim()
        ax1.set_ylim(ymin, ymax*args.yscale)

    if not logy:
        plot_tools.redo_axis_ticks(ax1, "y")
    plot_tools.redo_axis_ticks(ax1, "x", no_labels=len(axes_names) >= 2)

    hep.cms.label(ax=ax1, lumi=float(f"{lumi:.3g}") if lumi is not None else None, fontsize=20*args.scaleleg*scale, 
        label=args.cmsDecor, data=not args.noData)

    outfile = f"unfolded_{poi_type}{'_relative_' if relative_uncertainty else '_'}uncertainties"
    outfile += f"_{proc}"
    outfile += "_"+"_".join(axes_names)
    outfile += (f"_{channel}" if channel else "")
    outfile += (f"_logy" if logy else "")
    outfile += (f"_{args.postfix}" if args.postfix else "")
    plot_tools.save_pdf_and_png(outdir, outfile)

    outfile += (f"_{args.postfix}" if args.postfix else "")
    plot_tools.save_pdf_and_png(outdir, outfile)

    plot_tools.write_index_and_log(outdir, outfile, nround=2,
        yield_tables={f"Unfolded{' relative' if relative_uncertainty else ''} uncertainty{' [%]' if percentage else ''}": uncertainties},
        analysis_meta_info={args.infile : meta["meta_info"]},
        args=args,
    )

    plt.close()


def plot_uncertainties_ratio(df, df_ref, poi_type, poi_type_ref, channel=None, edges=None, scale=1., normalize=False, 
    logy=False, process_label="", axes=None, relative_uncertainty=False, lumi=None
):
    logger.info(f"Make "+("normalized " if normalize else "")+"unfoled xsec plot"+(f" in channel {channel}" if channel else ""))

    # read nominal values and uncertainties from fit result and fill histograms
    logger.debug(f"Produce histograms")

    yLabel = f"ref.({poi_type_ref}) / nom.({poi_type}) -1"

    if poi_type in ["mu", "nois"]:
        bin_widths = 1.0
    else:
        # central values
        bin_widths = edges[1:] - edges[:-1]

    errors = df["err_total"].values/bin_widths
    errors_ref = df_ref["err_total"].values/bin_widths
    if relative_uncertainty:
        values = df["value"].values/bin_widths
        values_ref = df_ref["value"].values/bin_widths
        errors /= values
        errors_ref /= values_ref

    xx = errors_ref/errors -1

    hist_ratio = hist.Hist(hist.axis.Variable(edges, underflow=False, overflow=False))
    hist_ratio.view(flow=False)[...] = xx

    # make plots
    if args.ylim is None:
        if logy:
            ylim = (max(xx)/10000., 1000 * max(xx))
        else:
            miny = min(xx)
            maxy = max(xx)
            rangey = maxy-miny
            ylim = (miny-rangey*0.1, maxy+rangey*0.7)
    else:
        ylim = args.ylim

    xlabel = "-".join([get_xlabel(a, process_label) for a in axes])
    if len(axes) >= 2:
        xlabel = xlabel.replace("(GeV)","")
        xlabel += "Bin"

    fig, ax1 = plot_tools.figure(hist_ratio, xlabel, yLabel, ylim, logy=logy, width_scale=2)

    ax1.plot([min(edges), max(edges)], [0.,0.], color="black", linestyle="-")

    hep.histplot(
        hist_ratio,
        yerr=False,
        histtype="step",
        color="black",
        label="Total",
        ax=ax1,
        alpha=1.,
        zorder=2,
    )
    uncertainties = make_yields_df([hist_ratio], ["Total"], per_bin=True, yield_only=True)


    if args.genFlow:
        ax1.fill([0,18.5, 18.5, 0,0], [ylim[0],*ylim,ylim[1],ylim[0]], color="grey", alpha=0.3)
        ax1.fill([len(errors)-17.5, len(errors)+0.5, len(errors)+0.5, len(errors)-17.5, len(errors)-17.5], [ylim[0],*ylim,ylim[1],ylim[0]], color="grey", alpha=0.3)

    if grouping:
        sources = ["err_"+g for g in grouping]
    else:
        sources =["err_stat"]
        sources += list(sorted([s for s in filter(lambda x: x.startswith("err"), df.keys()) 
            if s.replace("err_","") not in ["stat", "total"] ])) # total and stat are added first

    NUM_COLORS = len(sources)-1
    cm = mpl.colormaps["gist_rainbow"]
    # add each source of uncertainty
    i=0
    for source in sources:

        name = source.replace("err_","")

        name = translate_label.get(name,name)

        if source =="err_stat":
            color = "grey"
        else:
            color = cm(1.*i/NUM_COLORS)
            i += 1

        if i%3 == 0:
            linestype = "-" 
        elif i%3 == 1:
            linestype = "--" 
        else:
            linestype = ":" 

        hist_unc = hist.Hist(hist.axis.Variable(edges, underflow=False, overflow=False))

        if source not in df:
            logger.warning(f"Source {source} not found in dataframe")
            continue

        errors = df[source].values/bin_widths
        errors_ref = df_ref[source].values/bin_widths
        if relative_uncertainty:
            errors /= values
            errors_ref /= values_ref

        logger.debug(f"Plot source {source}")

        hist_unc.view(flow=False)[...] = errors_ref/errors-1

        hep.histplot(
            hist_unc,
            yerr=False,
            histtype="step",
            color=color,
            linestyle=linestype,
            label=name,
            ax=ax1,
            alpha=1.,
            zorder=2,
        )

        unc_df = make_yields_df([hist_unc], [name], per_bin=True, yield_only=True)
        uncertainties[name] = unc_df[name]

    scale = max(1, np.divide(*ax1.get_figure().get_size_inches())*0.3)

    plot_tools.addLegend(ax1, ncols=4, text_size=15*args.scaleleg*scale)

    if args.yscale:
        ymin, ymax = ax1.get_ylim()
        ax1.set_ylim(ymin, ymax*args.yscale)

    if not logy:
        plot_tools.redo_axis_ticks(ax1, "y")
    plot_tools.redo_axis_ticks(ax1, "x", no_labels=len(axes) >= 2)

    hep.cms.label(ax=ax1, lumi=float(f"{lumi:.3g}") if lumi is not None else None, fontsize=20*args.scaleleg*scale, 
        label=args.cmsDecor, data=not args.noData)

    outfile = f"{input_subdir}_unfolded_uncertainties_ratio"

    if poi_type in ["mu","nois"]:
        outfile += f"_{poi_type}"
    if relative_uncertainty:
        outfile += "_relative"   
    if normalize:
        outfile += "_normalized"
    if logy:
        outfile += "_logy"
    outfile += f"_{base_process}"
    outfile += "_"+"_".join(axes)
    outfile += (f"_{channel}" if channel else "")

    outfile += (f"_{args.postfix}" if args.postfix else "")
    plot_tools.save_pdf_and_png(outdir, outfile)

    plot_tools.write_index_and_log(outdir, outfile, nround=4 if normalize else 2,
        yield_tables={"Unfolded data uncertainty [%]": uncertainties},
        analysis_meta_info={"AnalysisOutput" : groups.getMetaInfo(), 'CardmakerOutput': meta_info},
        args=args,
    )

    plt.close()


for poi_type, poi_result in result.items():

    for channel, channel_result in poi_result.items():
        lumi = None

        for proc, proc_result in channel_result.items():
            
            histo_others=[]
            if args.histfile:
                groups_dict = {"W_qGen0": "Wminus", "W_qGen1": "Wplus", "Z": "Zmumu"}
                group_name = groups_dict[proc]
                for syst in args.varNames:
                    groups.loadHistsForDatagroups(args.baseName, syst=syst, procsToRead=[group_name], nominalIfMissing=False)
                    histo_others.append(groups.groups[group_name].hists[syst])

            for hist_name in filter(lambda x: not any([x.endswith(y) for y in ["_stat","_syst"]]), proc_result.keys()):
                hist_nominal = result[poi_type][channel][proc][hist_name]
                hist_stat = result[poi_type][channel][proc][f"{hist_name}_stat"]
                hist_ref = result_ref[poi_type][channel][proc][hist_name] if args.reference else None
                if args.selectAxis and args.selectEntries:
                    histo_others = [h[{k: v}] for h, k, v in zip(histo_others, args.selectAxis, args.selectEntries)]                

                axes = hist_nominal.axes
                hists_others = [h.project(*axes.name) for h in histo_others]

                selection_axes = [a for a in axes if a.name in args.selectionAxes]
                if len(selection_axes) > 0:
                    selection_bins = [np.arange(a.size) for a in axes if a.name in args.selectionAxes]
                    other_axes = [a for a in axes if a not in selection_axes]
                    iterator = itertools.product(*selection_bins)
                else:
                    iterator = [None]

                for bins in iterator:
                    if bins == None:
                        suffix = channel
                        h_nominal = hist_nominal
                        h_stat = hist_stat
                        h_ref = hist_ref
                        h_others = hists_others
                    else: 
                        idxs = {a.name: i for a, i in zip(selection_axes, bins) } 
                        if len(other_axes) == 0:
                            continue
                        logger.info(f"Make plot for axes {[a.name for a in other_axes]}, in bins {idxs}")
                        suffix = f"{channel}_" + "_".join([f"{a}{i}" for a, i in idxs.items()])
                        h_nominal = hist_nominal[idxs]
                        h_stat = hist_stat[idxs]
                        h_ref = h_ref[idxs] if args.reference else None
                        h_others = [h[idxs] for h in hists_others]

                    if "xsec" in args.plots:
                        plot_xsec_unfolded(h_nominal, h_stat, h_ref, poi_type=poi_type, channel=suffix, proc=proc, lumi=lumi,
                            hist_others=h_others, 
                            label_others=args.varLabels, 
                            color_others=args.colors
                        )

                    if "uncertainties" in args.plots:
                        h_systs = result[poi_type][channel][proc][f"{hist_name}_syst"]
                        if bins != None:
                            h_systs = h_systs[{**idxs, "syst": slice(None)}]

                        plot_uncertainties_unfolded(h_nominal, h_stat, h_systs, poi_type=poi_type, channel=suffix, proc = proc, lumi=lumi,
                            relative_uncertainty=False,#True,
                            # normalize=args.normalize, relative_uncertainty=not args.absolute, 
                            logy=args.logy)

                        # if "ratio" in args.plots:
                        #     plot_uncertainties_ratio(data_c, data_c_ref, poi_type, poi_type_ref, edges=edges, channel=channel, scale=scale, 
                        #         normalize=args.normalize, relative_uncertainty=not args.absolute, logy=args.logy, process_label = process_label, axes=channel_axes)

if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
    output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)
