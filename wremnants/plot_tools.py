import pathlib
import mplhep as hep
import matplotlib.pyplot as plt
from matplotlib import patches
from wremnants import boostHistHelpers as hh
from wremnants import histselections as sel
import math
import numpy as np
import re
import os
import logging
import shutil
import sys
import datetime

hep.style.use(hep.style.ROOT)

def figureWithRatio(href, xlabel, ylabel, ylim, rlabel, rrange, xlim=None,
    grid_on_main_plot = False, grid_on_ratio_plot = False, plot_title = None
):
    hax = href.axes[0]
    width = math.ceil(hax.size/400)
    fig = plt.figure(figsize=(8*width,8))
    ax1 = fig.add_subplot(4, 1, (1, 3)) 
    ax2 = fig.add_subplot(4, 1, 4) 

    ax2.set_xlabel(xlabel)
    ax1.set_xlabel(" ")
    ax1.set_ylabel(ylabel)
    ax1.set_xticklabels([])
    if not xlim:
        xlim = [href.axes[0].edges[0], href.axes[0].edges[href.axes[0].size-1]]
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    ax2.set_ylabel(rlabel, fontsize=22)
    ax2.set_ylim(rrange)
    if ylim:
        ax1.set_ylim(ylim)
    else:
        ax1.autoscale(axis='y')
    if grid_on_main_plot:  ax1.grid(which = "both")
    if grid_on_ratio_plot: ax2.grid(which = "both")
    if plot_title: ax1.set_title(plot_title)
    return fig,ax1,ax2

def addLegend(ax, ncols=2, extra_text=None):
    has_extra_text = extra_text is not None
    handles, labels = ax.get_legend_handles_labels()
    
    if has_extra_text:
        #handles.append(patches.Patch(color='none', label=extra_text))
        ax.plot([], [], ' ', ' ')

    shape = np.divide(*ax.get_figure().get_size_inches())
    #TODO: The goal is to leave the data in order, but it should be less hacky
    handles[:] = reversed(handles)
    labels[:] = reversed(labels)
    if len(handles) % 2 and ncols == 2:
        handles.insert(math.floor(len(handles)/2), patches.Patch(color='none', label = ' '))
        labels.insert(math.floor(len(labels)/2), ' ')
    #handles= reversed(handles)
    #labels= reversed(labels)
    ax.legend(handles=handles, labels=labels, prop={'size' : 20*(0.7 if shape == 1 else 1.3)}, ncol=ncols, loc='upper right')

def makeStackPlotWithRatio(
    histInfo, stackedProcs, histName="nominal", unstacked=None, 
    xlabel="", ylabel="Events/bin", rlabel = "Data/Pred.", rrange=[0.9, 1.1], ymax=None, xlim=None, nlegcols=2,
    binwnorm=None, select={},  action = (lambda x: x), extra_text=None, grid = False, plot_title = None,
    ratio_to_data=False,
):
    stack = [action(histInfo[k][histName][select]) for k in stackedProcs if histInfo[k][histName]]
    colors = [histInfo[k]["color"] for k in stackedProcs if histInfo[k][histName]]
    labels = [histInfo[k]["label"] for k in stackedProcs if histInfo[k][histName]]
    fig, ax1, ax2 = figureWithRatio(stack[0], xlabel, ylabel, [0, ymax] if ymax else None, rlabel, rrange, xlim=xlim, grid_on_ratio_plot = grid, plot_title = plot_title)
    
    hep.histplot(
        stack,
        histtype="fill",
        color=colors,
        label=labels,
        stack=True,
        ax=ax1,
        binwnorm=binwnorm,
    )
    
    if unstacked:
        data_hist = None
        if "Data" in histInfo and ratio_to_data:
            data_hist = action(histInfo["Data"][histName][select])
            hep.histplot(
                hh.divideHists(sum(stack), data_hist, cutoff=0.01),
                histtype="step",
                color=histInfo[stackedProcs[0]]["color"],
                label=histInfo[stackedProcs[0]]["label"],
                yerr=False,
                ax=ax2
            )

        if type(unstacked) == str: unstacked = unstacked.split(",")
        for proc in unstacked:
            unstack = action(histInfo[proc][histName][select])
            hep.histplot(
                unstack,
                yerr=True if proc == "Data" else False,
                histtype="errorbar" if (proc == "Data" or re.search("^pdf.*_sum", proc)) else "step",
                color=histInfo[proc]["color"],
                label=histInfo[proc]["label"],
                ax=ax1,
                binwnorm=binwnorm,
            )
            ratio_ref = data_hist if data_hist else sum(stack) 
            hep.histplot(
                hh.divideHists(unstack, ratio_ref, cutoff=0.01),
                histtype="errorbar" if proc == "Data" and not data_hist else "step",
                color=histInfo[proc]["color"],
                label=histInfo[proc]["label"],
                yerr=True if (proc == "Data" and not data_hist) else False,
                ax=ax2
            )

    addLegend(ax1, nlegcols, extra_text)
    return fig

def makePlotWithRatioToRef(
    hists, labels, colors, xlabel="", ylabel="Events/bin", rlabel="x/nominal",
    rrange=[0.9, 1.1], ymax=None, xlim=None, nlegcols=2, binwnorm=None, alpha=1.,
    baseline=True, data=False, autorrange=None, grid = False
):
    # nominal is always at first, data is always at last, if included
    ratio_hists = [hh.divideHists(h, hists[0], cutoff=0.00001) for h in hists[not baseline:]]
    fig, ax1, ax2 = figureWithRatio(hists[0], xlabel, ylabel, [0, ymax] if ymax else None, rlabel, rrange, xlim=xlim, grid_on_ratio_plot = grid)
    
    hep.histplot(
        hists[:len(hists) - data],
        histtype="step",
        color=colors[:(len(colors)- data)],
        label=labels[:(len(labels)- data)],
        stack=False,
        ax=ax1,
        binwnorm=binwnorm,
        alpha=alpha,
    )
    
    if len(hists) > 1:
        hep.histplot(
            ratio_hists[:len(ratio_hists) - data],
            histtype="step",
            color=colors[(not baseline):(len(colors)- data)],
            label=labels[(not baseline):(len(labels)- data)],
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
            hh.divideHists(data, hists[0], cutoff=1.e-6),
            histtype="errorbar",
            color=colors[-1],
            label=labels[-1],
            xerr=False,
            yerr=False,
            stack=False,
            ax=ax2,
            alpha=alpha,
        )

    addLegend(ax1, nlegcols)
    return fig

def make_plot_dir(outpath, outfolder):
    full_outpath = "/".join([outpath, outfolder])
    if not os.path.isdir(outpath):
        raise IOError(f"The path {outpath} doesn't not exist. You should create it (and possibly link it to your web area)")
        
    if not os.path.isdir(full_outpath):
        logging.info(f"Creating folder {full_outpath}")
        os.makedirs(full_outpath)

    return full_outpath

def save_pdf_and_png(outdir, basename):
    fname = f"{outdir}/{basename}.pdf"
    plt.savefig(fname, bbox_inches='tight')
    plt.savefig(fname.replace("pdf", "png"), bbox_inches='tight')
    logging.info(f"Wrote file(s) {fname}(.png)")

def write_index_and_log(outpath, logname, indexname="index.php", template_dir=f"{pathlib.Path(__file__).parent}/Templates"):
    if not os.path.isfile(f"{outpath}/{indexname}"):
        shutil.copyfile(f"{template_dir}/{indexname}", f"{outpath}/{indexname}")

    logdir = f"{outpath}/logs"
    if not os.path.isdir(logdir):
        os.mkdir(logdir)

    with open(f"{logdir}/{logname}.log", "w") as logf:
        meta_info = '-'*80 + '\n' + \
            'Script called at %s\n' % datetime.datetime.now() + \
            'The command was: %s\n' % ' '.join(sys.argv) + \
            '-'*80 + '\n'
        logf.write(meta_info)
