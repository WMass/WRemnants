import pathlib
import mplhep as hep
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import patches
from matplotlib.ticker import StrMethodFormatter # for setting number of decimal places on tick labels
from utilities import boostHistHelpers as hh,common,logging
from utilities.io_tools import output_tools
import hist
import math
import numpy as np
import re
import os
import shutil
import sys
import datetime
import json
import narf 

hep.style.use(hep.style.ROOT)

logger = logging.child_logger(__name__)

def cfgFigure(href, xlim=None, bin_density = 300,  width_scale=1, automatic_scale=True):
    hax = href.axes[0]
    if not xlim:
        xlim = [hax.edges[0], hax.edges[-1]]
    if not automatic_scale:
        return plt.figure(figsize=(width_scale*8,8)), xlim
    xlim_range = float(xlim[1] - xlim[0])
    original_xrange = float(hax.edges[-1] - hax.edges[0])
    raw_width = (hax.size/float(bin_density)) * (xlim_range / original_xrange)
    width = math.ceil(raw_width)

    return plt.figure(figsize=(width_scale*8*width,8)), xlim

def figure(href, xlabel, ylabel, ylim=None, xlim=None,
    grid = False, plot_title = None, title_padding = 0,
    bin_density = 300, cms_label = None, logy=False, logx=False,
    width_scale=1, height=8, automatic_scale=True
):
    if isinstance(href, hist.Hist):
        fig, xlim = cfgFigure(href, xlim, bin_density, width_scale, automatic_scale)
    else:
        if automatic_scale:
            raw_width = (len(href)/float(bin_density))
            width = math.ceil(raw_width)
        else:
            width = 1
        fig = plt.figure(figsize=(width_scale*height*width,height))

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

def addLegend(ax, ncols=2, extra_text=None, extra_text_loc=(0.8, 0.7), text_size=20, loc='upper right'):
    handles, labels = ax.get_legend_handles_labels()
    
    shape = np.divide(*ax.get_figure().get_size_inches())
    #TODO: The goal is to leave the data in order, but it should be less hacky
    handles[:] = reversed(handles)
    labels[:] = reversed(labels)
    if len(handles) % 2 and ncols == 2:
        handles.insert(math.floor(len(handles)/2), patches.Patch(color='none', label = ' '))
        labels.insert(math.floor(len(labels)/2), ' ')
    text_size = text_size*(0.7 if shape == 1 else 1.3)
    leg = ax.legend(handles=handles, labels=labels, prop={'size' : text_size}, ncol=ncols, loc=loc)

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
    xlabel="", ylabel=None, rlabel = "Data/Pred.", rrange=[0.9, 1.1], ylim=None, xlim=None, nlegcols=2,
    binwnorm=None, select={},  action = (lambda x: x), extra_text=None, extra_text_loc=(0.8, 0.7), grid = False, 
    plot_title = None, title_padding = 0, yscale=None, logy=False, logx=False, 
    fill_between=False, ratio_to_data=False, baseline=True, legtext_size=20, cms_decor="Preliminary", lumi=16.8,
    no_fill=False, no_stack=False, no_ratio=False, density=False, flow='none', bin_density=300, unstacked_linestyles=[],
    ratio_error=True, normalize_to_data=False, cutoff=1e-6,
):
    add_ratio = not (no_stack or no_ratio) 
    if ylabel is None:
        ylabel = "Events/bin" if not density else "density"

    colors = [histInfo[k].color for k in stackedProcs if histInfo[k].hists[histName]]
    labels = [histInfo[k].label for k in stackedProcs if histInfo[k].hists[histName]]

    to_read = stackedProcs[:]
    if "Data" in histInfo:
        to_read.append("Data")

    stack = []
    data_hist = None
    for k in to_read:
        if histName not in histInfo[k].hists or not histInfo[k].hists[histName]:
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

    if add_ratio:
        fig, ax1, ax2 = figureWithRatio(stack[0], xlabel, ylabel, ylim, rlabel, rrange, xlim=xlim, logy=logy, logx=logx, 
            grid_on_ratio_plot = grid, plot_title = plot_title, title_padding = title_padding, bin_density = bin_density)
    else:
        fig, ax1 = figure(stack[0], xlabel, ylabel, ylim, xlim=xlim, logy=logy, logx=logx, 
            plot_title = plot_title, title_padding = title_padding, bin_density = bin_density)

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

        if add_ratio:
            ax2.fill_between(axis, 
                    np.append((nom+std)/nom, ((nom+std)/nom)[-1]), 
                    np.append((nom-std)/nom, ((nom-std)/nom)[-1]),
                step='post',facecolor="none", zorder=2, hatch=hatchstyle, edgecolor="k", linewidth=0.0)
    
    opts=dict(stack=not no_stack, flow=flow)
    optsr=opts.copy() # no binwnorm for ratio axis
    optsr["density"]=density
    if density:
        opts["density"]=True
    else:
        opts["binwnorm"]=binwnorm

    if type(unstacked) == str: 
        unstacked = unstacked.split(",")

    scale = 1.
    if normalize_to_data:
        if "Data" not in histInfo:
            raise ValueError("Can't normalize to data without a data histogram!")

        vals = [x.value if hasattr(x, "value") else x for x in (data_hist.sum(), hh.sumHists(stack).sum())]
        scale = vals[0]/vals[1]
        logger.info(f"Rescaling all processes by {scale:0.3f} to match data norm")
        stack = [s*scale for s in stack]

    hep.histplot(
        stack,
        histtype="fill" if not no_fill else "step",
        color=colors,
        label=labels,
        ax=ax1,
        zorder=1,
        **opts
    )
    
    if "Data" in histInfo and ratio_to_data and add_ratio:
        hep.histplot(
            hh.divideHists(hh.sumHists(stack), data_hist, cutoff=cutoff, by_ax_name=False),
            histtype="step",
            color=histInfo[stackedProcs[-1]].color,
            label=histInfo[stackedProcs[-1]].label,
            yerr=False,
            ax=ax2,
            zorder=3,
            **optsr
        )

    if unstacked:

        linestyles = ['solid']*len(unstacked)
        data_idx = -1
        if "Data" in unstacked:
            data_idx = unstacked.index("Data") 
            linestyles[data_idx] = "None"
        linestyles = np.array(linestyles, dtype=object)
        linestyles[data_idx+1:data_idx+1+len(unstacked_linestyles)] = unstacked_linestyles

        ratio_ref = data_hist if ratio_to_data else hh.sumHists(stack)
        if baseline and add_ratio:
            hep.histplot(
                hh.divideHists(ratio_ref, ratio_ref, cutoff=cutoff, rel_unc=True, flow=False, by_ax_name=False),
                histtype="step",
                color="grey",
                alpha=0.5,
                yerr=ratio_error if ratio_ref.storage_type == hist.storage.Weight else False,
                ax=ax2,
                linewidth=2,
                **optsr
            )

        if fill_between and add_ratio:
            fill_procs = [x for x in unstacked if x != "Data"]
            if fill_between < 0:
                fill_between = len(fill_procs)+1
            logger.debug(f"Filling first {fill_between}")
            for up,down in zip(fill_procs[:fill_between:2], fill_procs[1:fill_between:2]):
                unstack_up = action(histInfo[up].hists[histName])*scale
                unstack_down = action(histInfo[down].hists[histName])*scale
                unstack_upr = hh.divideHists(unstack_up, ratio_ref, 1e-6, flow=False, by_ax_name=False).values()
                unstack_downr = hh.divideHists(unstack_down, ratio_ref, 1e-6, flow=False, by_ax_name=False).values()
                ax2.fill_between(unstack_up.axes[0].edges, 
                    np.insert(unstack_upr, 0, unstack_upr[0]),
                    np.insert(unstack_downr, 0, unstack_downr[0]),
                    step='pre', color=histInfo[up].color, alpha=0.5)

        for proc,style in zip(unstacked, linestyles):
            unstack = histInfo[proc].hists[histName]
            if not fitresult or proc not in to_read:
                unstack = action(unstack)[select]
            if proc != "Data":
                unstack = unstack*scale

            hep.histplot(
                unstack,
                yerr=True if style == "None" else False,
                histtype="errorbar" if style == "None" else "step",
                color=histInfo[proc].color,
                label=histInfo[proc].label,
                ax=ax1,
                alpha=0.7 if style != "None" else 1.,
                linestyle=style,
                **opts
            )
            if ratio_to_data and proc == "Data" or not add_ratio:
                continue
            stack_ratio = hh.divideHists(unstack, ratio_ref, cutoff=cutoff, rel_unc=True, flow=False, by_ax_name=False)
            hep.histplot(stack_ratio,
                histtype="errorbar" if style == "None" else "step",
                color=histInfo[proc].color,
                label=histInfo[proc].label,
                yerr=True if (style == "None" and stack_ratio.storage_type == hist.storage.Weight) else False,
                linewidth=2,
                linestyle=style,
                ax=ax2,
                **optsr
            )

    addLegend(ax1, nlegcols, extra_text=extra_text, extra_text_loc=extra_text_loc, text_size=legtext_size)
    if add_ratio:
        fix_axes(ax1, ax2, yscale=yscale, logy=logy)
    else:
        fix_axes(ax1, yscale=yscale, logy=logy)

    if cms_decor:
        lumi = float(f"{lumi:.3g}") if not density else None
        scale = max(1, np.divide(*ax1.get_figure().get_size_inches())*0.3)
        hep.cms.label(ax=ax1, lumi=lumi, fontsize=legtext_size*scale, 
            label=cms_decor, data="Data" in histInfo)

    return fig

def makePlotWithRatioToRef(
    hists, labels, colors, linestyles=[],
    xlabel="", ylabel="Events/bin", rlabel="x/nominal",
    rrange=[0.9, 1.1], ylim=None, xlim=None, nlegcols=2, binwnorm=None, alpha=1.,
    baseline=True, dataIdx=None, autorrange=None, grid = False, extra_text=None, extra_text_loc=(0.8, 0.7),
    yerr=False, legtext_size=20, plot_title=None, x_ticks_ndp = None, bin_density = 300, yscale=None,
    logy=False, logx=False, fill_between=0, title_padding = 0, cms_label = None, cutoff=1e-6,
):
    if len(hists) != len(labels) or len(hists) != len(colors):
        raise ValueError(f"Number of hists ({len(hists)}), colors ({len(colors)}), and labels ({len(labels)}) must agree!")
    ratio_hists = [hh.divideHists(h, hists[0], cutoff=cutoff, flow=False, rel_unc=True, by_ax_name=False) for h in hists[not baseline:]]
    fig, ax1, ax2 = figureWithRatio(
        hists[0], xlabel, ylabel, ylim, rlabel, rrange, xlim=xlim, 
        grid_on_ratio_plot = grid, plot_title = plot_title, title_padding=title_padding,
        bin_density = bin_density, cms_label = cms_label, logy=logy, logx=logx
    )

    linestyles = linestyles+['solid']*(len(hists)-len(linestyles))

    exclude_data = lambda x: [j for i,j in enumerate(x) if i != dataIdx]
    
    hep.histplot(
        exclude_data(hists),
        histtype="step",
        color=exclude_data(colors),
        label=exclude_data(labels),
        linestyle=exclude_data(linestyles),
        stack=False,
        ax=ax1,
        yerr=yerr,
        binwnorm=binwnorm,
        alpha=alpha,
        flow='none',
    )

    if len(hists) > 1:
        ratio_hists = [hh.divideHists(h, hists[0], cutoff=cutoff, flow=False, rel_unc=True, by_ax_name=False) for h in hists]
        if fill_between != 0:
            for upr,downr,color in zip(ratio_hists[-fill_between::2], ratio_hists[-fill_between+1::2], colors[-fill_between::2]):
                ax2.fill_between(upr.axes[0].edges, 
                        np.append(upr.values(), upr.values()[-1]), 
                        np.append(downr.values(), downr.values()[-1]),
                            step='post', color=color, alpha=0.5)

        hep.histplot(
            exclude_data(ratio_hists)[not baseline:],
            histtype="step",
            color=exclude_data(colors)[not baseline:],
            linestyle=exclude_data(linestyles)[not baseline:],
            yerr=yerr,
            stack=False,
            ax=ax2,
            alpha=alpha,
            flow='none',
        )
    if dataIdx is not None:
        hep.histplot(
            hists[dataIdx],
            histtype="errorbar",
            color=colors[dataIdx],
            label=labels[dataIdx],
            stack=False,
            ax=ax1,
            binwnorm=binwnorm,
            alpha=alpha,
            flow='none',
        )
        hep.histplot(
            hh.divideHists(hists[dataIdx], hists[0], cutoff=cutoff, flow=False, by_ax_name=False, rel_unc=True),
            histtype="errorbar",
            color=colors[dataIdx],
            xerr=False,
            yerr=True,
            stack=False,
            ax=ax2,
            alpha=alpha,
            flow='none',
        )

    addLegend(ax1, nlegcols, extra_text=extra_text, extra_text_loc=extra_text_loc, text_size=legtext_size)
    
    # This seems like a bug, but it's needed
    if not xlim:
        xlim = [hists[0].axes[0].edges[0], hists[0].axes[0].edges[-1]]
    fix_axes(ax1, ax2, yscale=yscale, logy=logy)
    if x_ticks_ndp: ax2.xaxis.set_major_formatter(StrMethodFormatter('{x:.' + str(x_ticks_ndp) + 'f}'))
    return fig

def makeHistPlot2D(h2d, flow=False, **kwargs):
    if flow:
        xedges, yedges = extendEdgesByFlow(h2d)
    else:
        edges = h2d.axes.edges
        xedges = np.reshape(edges[0], len(edges[0]))
        yedges = edges[1][0]
    values = h2d.values(flow=flow)
    variances = h2d.variances(flow=flow)
    makePlot2D(values, variances, xedges, yedges, **kwargs)

def makePlot2D(values, variances=None, xedges=None, yedges=None, 
    density=False, plot_uncertainties=False,
    xlabel="", ylabel="", zlabel="", colormap="RdBu", plot_title=None,
    ylim=None, xlim=None, zlim=None, zsymmetrize=None,
    logz=False, # logy=False, logx=False, #TODO implement
    cms_label="Work in progress", automatic_scale=False, width_scale=1.2
):
    if xedges is None or yedges is None:
        xbins, ybins = values.shape
        if xedges is None:
            xedges = np.arange(xbins)
        if yedges is None:
            yedges = np.arange(ybins)
    # if variances is None:
    #     logger.warning("No variances given, assume")
    #     variances = values

    if density:
        xbinwidths = np.diff(xedges)
        ybinwidths = np.diff(yedges)
        binwidths = np.outer(xbinwidths, ybinwidths) 
        values /= binwidths
        variances /= binwidths
    elif plot_uncertainties:
        # plot relative uncertainties instead
        values = np.sqrt(hh.relVariance(values, variances, fillOnes=True))

    if xlim is None:
        xlim = (xedges[0],xedges[-1])
    if ylim is None:
        ylim = (yedges[0],yedges[-1])

    fig, ax = figure(values, xlabel=xlabel, ylabel=ylabel, cms_label=cms_label, automatic_scale=automatic_scale, width_scale=width_scale, xlim=xlim, ylim=ylim)

    if zlim is None:
        if logz:
            zmin = min(values[values>0]) # smallest value that is not 0
        else:
            zmin = values.min()
        zmax = values.max()
        zlim = (zmin,zmax)
        
    # make symmetric range around value of zsymmetrize
    if zsymmetrize is not None:
        zrange = max((zmin-zsymmetrize), (zsymmetrize-zmax))
        zlim = [zsymmetrize-zrange, zsymmetrize+zrange]

    if logz:
        colormesh = ax.pcolormesh(xedges, yedges, values.T, cmap=getattr(mpl.cm, colormap), norm=mpl.colors.LogNorm(vmin=zlim[0], vmax=zlim[1]))
    else:
        colormesh = ax.pcolormesh(xedges, yedges, values.T, cmap=getattr(mpl.cm, colormap), vmin=zlim[0], vmax=zlim[1])
    cbar = fig.colorbar(colormesh, ax=ax)

    if plot_title:
        ax.text(1.0, 1.003, plot_title, transform=ax.transAxes, fontsize=30,
            verticalalignment='bottom', horizontalalignment="right")

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

def fix_axes(ax1, ax2=None, yscale=None, logy=False):
    #TODO: Would be good to get this working
    #ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    if yscale:
        ymin, ymax = ax1.get_ylim()
        ax1.set_ylim(ymin, ymax*yscale)
    if not logy:
        redo_axis_ticks(ax1, "y")
        if ax2 is not None:
            redo_axis_ticks(ax2, "x")
    if ax2 is not None:
        redo_axis_ticks(ax1, "x", True)
        ax1.set_xticklabels([])
    else:
        redo_axis_ticks(ax1, "x", False)


def redo_axis_ticks(ax, axlabel, no_labels=False):
    autoloc = ticker.AutoLocator()
    # Need this to avoid a warning when you set the axis values manually
    fixedloc = ticker.FixedLocator(autoloc.tick_values(*getattr(ax, f"get_{axlabel}lim")()))
    getattr(ax, f"{axlabel}axis").set_major_locator(fixedloc)
    ticks = getattr(ax, f"get_{axlabel}ticks")()
    labels = [format_axis_num(x, ticks[-1]) for x in ticks] if not no_labels else []
    getattr(ax, f"set_{axlabel}ticklabels")(labels)

def format_axis_num(val, maxval):
    if type(val) == int or val.is_integer():
        # This is kinda dumb and I might change it
        return f"{val:.0f}" if maxval > 5 else f"{val:0.1f}"
    return f"{val:0.3g}" if maxval > 10 else f"{val:0.2g}"

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
            f'The command was: {narf.ioutils.script_command_to_str(sys.argv, args)}\n' + \
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
                    data_yield = unstacked[unstacked["Process"] == "Data"]["Yield"].iloc[0]
                    ratio = float(yield_tables["Stacked processes"]["Yield"].sum()/data_yield)*100
                    logf.write(f"===> Sum unstacked to data is {ratio:.2f}%")

        if analysis_meta_info:
            for k,analysis_info in analysis_meta_info.items():
                logf.write('\n'+'-'*80+"\n")
                logf.write(f"Meta info from input file {k}\n")
                logf.write('\n'+'-'*80+"\n")
                logf.write(json.dumps(analysis_info, indent=5).replace("\\n", "\n"))
        logger.info(f"Writing file {logname}")
