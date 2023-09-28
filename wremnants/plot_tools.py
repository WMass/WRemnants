import pathlib
import mplhep as hep
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import patches
from matplotlib.ticker import StrMethodFormatter # for setting number of decimal places on tick labels
from utilities import boostHistHelpers as hh,common,output_tools,logging
from wremnants import histselections as sel
import hist
import math
import numpy as np
import re
import os
import shutil
import sys
import datetime
import json

hep.style.use(hep.style.ROOT)

logger = logging.child_logger(__name__)

def cfgFigure(href, xlim=None, bin_density = 300,  width_scale=1, automatic_scale=True):
    hax = href.axes[0]
    if not xlim:
        xlim = [hax.edges[0], hax.edges[-1]]
    xlim_range = float(xlim[1] - xlim[0])
    original_xrange = float(hax.edges[-1] - hax.edges[0])
    if automatic_scale:
        raw_width = (hax.size/float(bin_density)) * (xlim_range / original_xrange)
        width = math.ceil(raw_width)
    else:
        width=1

    return plt.figure(figsize=(width_scale*8*width,8)), xlim

def figure(href, xlabel, ylabel, ylim=None, xlim=None,
    grid = False, plot_title = None, title_padding = 0,
    bin_density = 300, cms_label = None, logy=False, logx=False,
    width_scale=1, automatic_scale=True
):
    fig, xlim = cfgFigure(href, xlim, bin_density, width_scale, automatic_scale)

    ax1 = fig.add_subplot() 
    if cms_label: hep.cms.text(cms_label)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_xlim(xlim)

    if ylim is not None:
        ax1.set_ylim(ylim)
    else:
        ax1.autoscale(axis='y')

    if logy:
        ax1.set_yscale('log')
    if logx:
        ax1.set_xscale('log')

    if grid:  ax1.grid(which = "both")
    if plot_title: ax1.set_title(plot_title, pad = title_padding)
    return fig,ax1 

def figureWithRatio(href, xlabel, ylabel, ylim, rlabel, rrange, xlim=None,
    grid_on_main_plot = False, grid_on_ratio_plot = False, plot_title = None, title_padding = 0,
    x_ticks_ndp = None, bin_density = 300, cms_label = None, logy=False, logx=False,
    width_scale=1, automatic_scale=True
):
    fig, xlim = cfgFigure(href, xlim, bin_density, width_scale, automatic_scale)
    
    ax1 = fig.add_subplot(4, 1, (1, 3)) 
    if cms_label: hep.cms.text(cms_label)
    ax2 = fig.add_subplot(4, 1, 4) 

    ax2.set_xlabel(xlabel)
    ax1.set_xlabel(" ")
    ax1.set_ylabel(ylabel)
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    if x_ticks_ndp: ax2.xaxis.set_major_formatter(StrMethodFormatter('{x:.' + str(x_ticks_ndp) + 'f}'))
    ax2.set_ylabel(rlabel, fontsize=22)
    ax2.set_ylim(rrange)

    if ylim:
        ax1.set_ylim(ylim)
    else:
        ax1.autoscale(axis='y')

    if logy:
        ax1.set_yscale('log')
    if logx:
        ax1.set_xscale('log')
        ax2.set_xscale('log')

    if grid_on_main_plot:  ax1.grid(which = "both")
    if grid_on_ratio_plot: ax2.grid(which = "both")
    if plot_title: ax1.set_title(plot_title, pad = title_padding)
    return fig,ax1,ax2

def addLegend(ax, ncols=2, extra_text=None, extra_text_loc=(0.8, 0.7), text_size=20):
    handles, labels = ax.get_legend_handles_labels()
    
    shape = np.divide(*ax.get_figure().get_size_inches())
    #TODO: The goal is to leave the data in order, but it should be less hacky
    handles[:] = reversed(handles)
    labels[:] = reversed(labels)
    if len(handles) % 2 and ncols == 2:
        handles.insert(math.floor(len(handles)/2), patches.Patch(color='none', label = ' '))
        labels.insert(math.floor(len(labels)/2), ' ')
    text_size = text_size*(0.7 if shape == 1 else 1.3)
    leg = ax.legend(handles=handles, labels=labels, prop={'size' : text_size}, ncol=ncols, loc='upper right')

    if extra_text:
        p = leg.get_frame()
        bounds = leg.get_bbox_to_anchor().bounds
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='square', facecolor='white', alpha=0.5)

        # TODO: Figure out how to make this dynamic wrt the legend
        ax.text(*extra_text_loc, extra_text, transform=ax.transAxes, fontsize=text_size,
                verticalalignment='top', bbox=props)

def makeStackPlotWithRatio(
    histInfo, stackedProcs, histName="nominal", unstacked=None, 
    fitresult=None, prefit=False,
    xlabel="", ylabel="Events/bin", rlabel = "Data/Pred.", rrange=[0.9, 1.1], ylim=None, xlim=None, nlegcols=2,
    binwnorm=None, select={},  action = (lambda x: x), extra_text=None, extra_text_loc=(0.8, 0.7), grid = False, 
    plot_title = None, title_padding = 0, yscale=None,
    fill_between=False, skip_fill=0, ratio_to_data=False, baseline=True, legtext_size=20, cms_decor="Preliminary", lumi=16.8,
    no_fill=False, bin_density=300, unstacked_linestyles=[],
):
    colors = [histInfo[k].color for k in stackedProcs if histInfo[k].hists[histName]]
    labels = [histInfo[k].label for k in stackedProcs if histInfo[k].hists[histName]]

    to_read = stackedProcs[:]
    if "Data" in histInfo:
        to_read.append("Data")

    stack = []
    data_hist = None
    for k in to_read:
        if not histInfo[k].hists[histName]:
            logger.warning(f"Failed to find hist {histName} for proc {k}")
            continue
        h = action(histInfo[k].hists[histName])[select]
        
        # Use this if the hist has been rebinned for combine
        if xlim:
            h = h[complex(0, xlim[0]):complex(0, xlim[1])]

        # If plotting from combine, apply the action to the underlying hist.
        # Don't do this for the generic case, as it screws up the ability to make multiple plots
        if fitresult:
            histInfo[k].hists[histName] = h

        if k != "Data":
            stack.append(h)
        else:
            data_hist = h

    fig, ax1, ax2 = figureWithRatio(stack[0], xlabel, ylabel, ylim, rlabel, rrange, xlim=xlim, 
        grid_on_ratio_plot = grid, plot_title = plot_title, title_padding = title_padding, bin_density = bin_density)

    if fitresult:
        import uproot
        combine_result = uproot.open(fitresult)

        fittype = "prefit" if prefit else "postfit"

        # set histograms to prefit/postfit values
        for p in to_read:

            hname = f"expproc_{p}_{fittype}" if p != "Data" else "obs"
            vals = combine_result[hname].to_hist().values()
            if len(histInfo[p].hists[histName].values()) != len(vals):
                raise ValueError(f"The size of the combine histogram ({(vals.shape)}) is not consistent with the xlim or input hist ({histInfo[p].hists[histName].shape})")

            histInfo[p].hists[histName].values()[...] = vals
            if p == "Data":
                histInfo[p].hists[histName].variances()[...] = vals

        
        # for postfit uncertaity bands
        axis = histInfo[to_read[0]].hists[histName].axes[0].edges

        # need to divide by bin width
        binwidth = axis[1:]-axis[:-1]
        hexp = combine_result[f"expfull_{fittype}"].to_hist()
        if hexp.storage_type != hist.storage.Weight:
            raise ValueError(f"Did not find uncertainties in {fittype} hist. Make sure you run combinetf with --computeHistErrors!")
        nom = hexp.values() / binwidth
        std = np.sqrt(hexp.variances()) / binwidth

        hatchstyle = '///'
        ax1.fill_between(axis, 
                np.append(nom+std, (nom+std)[-1]), 
                np.append(nom-std, (nom-std)[-1]),
            step='post',facecolor="none", zorder=2, hatch=hatchstyle, edgecolor="k", linewidth=0.0, label="Uncertainty")

        ax2.fill_between(axis, 
                np.append((nom+std)/nom, ((nom+std)/nom)[-1]), 
                np.append((nom-std)/nom, ((nom-std)/nom)[-1]),
            step='post',facecolor="none", zorder=2, hatch=hatchstyle, edgecolor="k", linewidth=0.0)

    hep.histplot(
        stack,
        histtype="fill" if not no_fill else "step",
        color=colors,
        label=labels,
        stack=True,
        ax=ax1,
        binwnorm=binwnorm,
        zorder=1,
    )
    
    if "Data" in histInfo and ratio_to_data:
        hep.histplot(
            hh.divideHists(sum(stack), data_hist, cutoff=0.01),
            histtype="step",
            color=histInfo[stackedProcs[-1]].color,
            label=histInfo[stackedProcs[-1]].label,
            yerr=False,
            ax=ax2,
            zorder=3,
        )

    if unstacked:
        if type(unstacked) == str: 
            unstacked = unstacked.split(",")

        linestyles = ['solid']*len(unstacked)
        data_idx = -1
        if "Data" in unstacked:
            data_idx = unstacked.index("Data") 
            linestyles[data_idx] = "None"
        linestyles = np.array(linestyles, dtype=object)
        linestyles[data_idx+1:data_idx+1+len(unstacked_linestyles)] = unstacked_linestyles

        ratio_ref = data_hist if ratio_to_data else sum(stack) 
        if baseline:
            hep.histplot(
                hh.divideHists(ratio_ref, ratio_ref, cutoff=1e-8, rel_unc=True),
                histtype="step",
                color="grey",
                alpha=0.5,
                yerr=True,
                ax=ax2,
                linewidth=2,
            )

        for proc,style in zip(unstacked, linestyles):
            if ratio_to_data and proc == "Data":
                continue
            unstack = histInfo[proc].hists[histName]
            if not fitresult or proc not in to_read:
                unstack = action(unstack)[select]

            hep.histplot(
                unstack,
                yerr=True if style == "None" else False,
                histtype="errorbar" if style == "None" else "step",
                color=histInfo[proc].color,
                label=histInfo[proc].label,
                ax=ax1,
                alpha=0.7 if style != "None" else 1.,
                linestyle=style,
                binwnorm=binwnorm,
            )
            hep.histplot(
                hh.divideHists(unstack, ratio_ref, cutoff=0.01, rel_unc=True),
                histtype="errorbar" if style == "None" else "step",
                color=histInfo[proc].color,
                label=histInfo[proc].label,
                yerr=True if style == "None" else False,
                linewidth=2,
                linestyle=style,
                ax=ax2
            )

        if fill_between:
            fill_procs = [x for x in unstacked if x != "Data"]
            if not skip_fill:
                skip_fill = len(fill_procs) % 2
            logger.debug(f"Skip filling first {skip_fill}")
            for up,down in zip(fill_procs[skip_fill::2], fill_procs[skip_fill+1::2]):
                unstack_up = histInfo[up].hists[histName]
                unstack_down = histInfo[down].hists[histName]
                unstack_upr = hh.divideHists(unstack_up, ratio_ref, 1e-6)
                unstack_downr = hh.divideHists(unstack_down, ratio_ref, 1e-6)
                ax2.fill_between(unstack_upr.axes[0].edges[:-1], 
                        unstack_upr.values(), unstack_downr.values(),
                        # FIXME: Not sure if this is needed, currently not working correctly
                        #np.append(unstack_upr.values(), unstack_upr.values()[-1]), 
                        #np.append(unstack_down.values(), unstack_downr.values()[-1]),
                    step='post', color=histInfo[up].color, alpha=0.5)

    addLegend(ax1, nlegcols, extra_text=extra_text, extra_text_loc=extra_text_loc, text_size=legtext_size)
    fix_axes(ax1, ax2, yscale=yscale)

    if cms_decor:
        scale = max(1, np.divide(*ax1.get_figure().get_size_inches())*0.3)
        hep.cms.label(ax=ax1, lumi=float(f"{lumi:.3g}"), fontsize=legtext_size*scale, 
            label=cms_decor, data="Data" in histInfo)

    return fig

def makePlotWithRatioToRef(
    hists, labels, colors, linestyles=[],
    xlabel="", ylabel="Events/bin", rlabel="x/nominal",
    rrange=[0.9, 1.1], ylim=None, xlim=None, nlegcols=2, binwnorm=None, alpha=1.,
    baseline=True, data=False, autorrange=None, grid = False, extra_text=None, extra_text_loc=(0.8, 0.7),
    yerr=False, legtext_size=20, plot_title=None, x_ticks_ndp = None, bin_density = 300, yscale=None,
    logy=False, logx=False, fill_between=False, title_padding = 0, cms_label = None
):
    if len(hists) != len(labels) or len(hists) != len(colors):
        raise ValueError(f"Number of hists ({len(hists)}), colors ({len(colors)}), and labels ({len(labels)}) must agree!")
    # nominal is always at first, data is always at last, if included
    ratio_hists = [hh.divideHists(h, hists[0], cutoff=0.00001) for h in hists[not baseline:]]
    fig, ax1, ax2 = figureWithRatio(
        hists[0], xlabel, ylabel, ylim, rlabel, rrange, xlim=xlim, 
        grid_on_ratio_plot = grid, plot_title = plot_title, title_padding=title_padding,
        bin_density = bin_density, cms_label = cms_label, logy=logy, logx=logx
    )

    linestyles = linestyles+['solid']*(len(hists)-len(linestyles))
    
    count = len(hists)-data
    hep.histplot(
        hists[:count],
        histtype="step",
        color=colors[:count],
        label=labels[:count],
        linestyle=linestyles,
        stack=False,
        ax=ax1,
        yerr=yerr,
        binwnorm=binwnorm,
        alpha=alpha,
    )

    if len(hists) > 1:
        ratio_hists = [hh.divideHists(h, hists[0], cutoff=0.00001) for h in hists[not baseline:]]
        if fill_between:
            for up,down,color in zip(hists[1::2], hists[2::2], colors[1::2]):
                upr = hh.divideHists(up, hists[0], 1e-6)
                downr = hh.divideHists(down, hists[0], 1e-6)
                ax2.fill_between(upr.axes[0].edges, 
                        np.append(upr.values(), upr.values()[-1]), 
                        np.append(downr.values(), downr.values()[-1]),
                            step='post', color=color, alpha=0.5)

        count = len(ratio_hists) - data if not fill_between else 1
        hep.histplot(
            ratio_hists[(not baseline):count],
            histtype="step",
            color=colors[(not baseline):count],
            linestyle=linestyles,
            yerr=False,
            stack=False,
            ax=ax2,
            alpha=alpha,
        )
    if data:
        hep.histplot(
            hists[-1],
            histtype="errorbar",
            color=colors[-1],
            label=labels[-1],
            stack=False,
            ax=ax1,
            binwnorm=binwnorm,
            alpha=alpha,
        )
        hep.histplot(
            hh.divideHists(data, hists[0], cutoff=1.e-8),
            histtype="errorbar",
            color=colors[-1],
            label=labels[-1],
            xerr=False,
            yerr=False,
            stack=False,
            ax=ax2,
            alpha=alpha,
        )

    addLegend(ax1, nlegcols, extra_text=extra_text, extra_text_loc=extra_text_loc, text_size=legtext_size)
    
    # This seems like a bug, but it's needed
    if not xlim:
        xlim = [hists[0].axes[0].edges[0], hists[0].axes[0].edges[-1]]
    fix_axes(ax1, ax2, yscale=yscale, logy=logy)
    if x_ticks_ndp: ax2.xaxis.set_major_formatter(StrMethodFormatter('{x:.' + str(x_ticks_ndp) + 'f}'))
    return fig

def extendEdgesByFlow(href, bin_flow_width=0.02):
    # add extra bin with bin wdith of a fraction of the total width
    all_edges = []
    for axis in href.axes:
        edges = axis.edges
        axis_range = edges[-1] - edges[0]
        if axis.traits.underflow:
            edges = np.insert(edges, 0, edges[0] - axis_range*bin_flow_width)
        if axis.traits.overflow:
            edges = np.append(edges, edges[-1] + axis_range*bin_flow_width)
        all_edges.append(edges)
    if len(all_edges) == 1:
        return all_edges[0]
    else:
        return all_edges

def fix_axes(ax1, ax2, yscale=None, logy=False):
    #TODO: Would be good to get this working
    #ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    if yscale:
        ymin, ymax = ax1.get_ylim()
        ax1.set_ylim(ymin, ymax*yscale)
    if not logy:
        redo_axis_ticks(ax1, "y")
        redo_axis_ticks(ax2, "x")
    redo_axis_ticks(ax1, "x", True)
    ax1.set_xticklabels([])

def redo_axis_ticks(ax, axlabel, no_labels=False):
    autoloc = ticker.AutoLocator()
    # Need this to avoid a warning when you set the axis values manually
    fixedloc = ticker.FixedLocator(autoloc.tick_values(*getattr(ax, f"get_{axlabel}lim")()))
    getattr(ax, f"{axlabel}axis").set_major_locator(fixedloc)
    ticks = getattr(ax, f"get_{axlabel}ticks")()
    labels = [format_axis_num(x) for x in ticks] if not no_labels else []
    getattr(ax, f"set_{axlabel}ticklabels")(labels)

def format_axis_num(val):
    if type(val) == int or val.is_integer():
        # This is kinda dumb and I might change it
        return f"{val:.0f}" if val > 5 else f"{val:0.1f}"
    return f"{val:0.3g}" if val > 10 else f"{val:0.2g}"

def save_pdf_and_png(outdir, basename, fig=None):
    fname = f"{outdir}/{basename}.pdf"
    if fig:
        fig.savefig(fname, bbox_inches='tight')
        fig.savefig(fname.replace(".pdf", ".png"), bbox_inches='tight')
    else:
        plt.savefig(fname, bbox_inches='tight')
        plt.savefig(fname.replace(".pdf", ".png"), bbox_inches='tight')
    logger.info(f"Wrote file(s) {fname}(.png)")

def write_index_and_log(outpath, logname, indexname="index.php", template_dir=f"{pathlib.Path(__file__).parent}/Templates", 
        yield_tables=None, analysis_meta_info=None, args={}, nround=2):
    shutil.copyfile(f"{template_dir}/{indexname}", f"{outpath}/{indexname}")
    logname = f"{outpath}/{logname}.log"

    with open(logname, "w") as logf:
        meta_info = '-'*80 + '\n' + \
            f'Script called at {datetime.datetime.now()}\n' + \
            f'The command was: {output_tools.script_command_to_str(sys.argv, args)}\n' + \
            '-'*80 + '\n'
        logf.write(meta_info)

        if yield_tables:
            for k,v in yield_tables.items():
                logf.write(f"Yield information for {k}\n")
                logf.write("-"*80+"\n")
                logf.write(str(v.round(nround))+"\n\n")

            if "Unstacked processes" in yield_tables and "Stacked processes" in yield_tables:
                if "Data" in yield_tables["Unstacked processes"]["Process"].values:
                    unstacked = yield_tables["Unstacked processes"]
                    data_yield = unstacked[unstacked["Process"] == "Data"]["Yield"]
                    ratio = float(yield_tables["Stacked processes"]["Yield"].sum()/data_yield)*100
                    logf.write(f"===> Sum unstacked to data is {ratio:.2f}%")

        if analysis_meta_info:
            for k,analysis_info in analysis_meta_info.items():
                logf.write('\n'+'-'*80+"\n")
                logf.write(f"Meta info from input file {k}\n")
                logf.write('\n'+'-'*80+"\n")
                logf.write(json.dumps(analysis_info, indent=5).replace("\\n", "\n"))
        logger.info(f"Writing file {logname}")
