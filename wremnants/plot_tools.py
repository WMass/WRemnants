import mplhep as hep
import matplotlib.pyplot as plt
from matplotlib import patches
from wremnants import boostHistHelpers as hh
from wremnants import histselections as sel
import math
import numpy as np
hep.style.use(hep.style.ROOT)

def figureWithRatio(href, xlabel, ylabel, ylim, rlabel, rrange, xlim=None):
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
    return fig,ax1,ax2

def addLegend(ax, extra_text=None):
    has_extra_text = extra_text is not None
    handles, labels = ax.get_legend_handles_labels()
    
    if has_extra_text:
        #handles.append(patches.Patch(color='none', label=extra_text))
        ax.plot([], [], ' ', ' ')

    shape = np.divide(*ax.get_figure().get_size_inches())
    #TODO: The goal is to leave the data in order, but it should be less hacky
    handles[:] = reversed(handles)
    labels[:] = reversed(labels)
    if len(handles) % 2:
        handles.insert(math.floor(len(handles)/2), patches.Patch(color='none', label = ' '))
        labels.insert(math.floor(len(labels)/2), ' ')
    #handles= reversed(handles)
    #labels= reversed(labels)
    ax.legend(handles=handles, labels=labels, prop={'size' : 20*(0.7 if shape == 1 else 1.3)}, ncol=2, loc='upper right')

def makeStackPlotWithRatio(histInfo, stackedProcs, label="nominal", unstacked=None, xlabel="", ylabel="Events/bin", 
                rrange=[0.9, 1.1], ymax=None, xlim=None, binwnorm=None, select={}, action=None, extra_text=None):
    stack = [action(histInfo[k][label][select]) for k in stackedProcs if histInfo[k][label]]
    colors = [histInfo[k]["color"] for k in stackedProcs if histInfo[k][label]]
    labels = [histInfo[k]["label"] for k in stackedProcs if histInfo[k][label]]

    fig, ax1, ax2 = figureWithRatio(stack[0], xlabel, ylabel, [0, ymax] if ymax else None, "Data/Pred.", rrange, xlim=xlim)
            
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
        for proc in unstacked:
            unstack = action(histInfo[proc][label][select])
            hep.histplot(
                unstack,
                yerr=True if proc == "Data" else False,
                histtype="errorbar" if proc == "Data" else "step",
                color=histInfo[proc]["color"],
                label=histInfo[proc]["label"],
                ax=ax1,
                binwnorm=binwnorm,
            )
            hep.histplot(
                hh.divideHists(unstack, sum(stack), cutoff=0.01),
                histtype="errorbar" if proc == "Data" else "step",
                color=histInfo[proc]["color"],
                label=histInfo[proc]["label"],
                yerr=True if proc == "Data" else False,
                ax=ax2
            )

    addLegend(ax1, extra_text)
    return fig

def makePlotWithRatioToRef(hists, labels, colors, xlabel="", ylabel="Events/bin", rlabel="bugfix/bugged",
                rrange=[0.9, 1.1], ymax=None, xlim=None, binwnorm=None, baseline=None, autorrange=None):
    if baseline:
        ratio_plots_starting_idx = 0
    else:
        ratio_plots_starting_idx = 1
    ratio_hists = [hh.divideHists(h, hists[0], cutoff=0.00001) for h in hists[ratio_plots_starting_idx:]]
#    if autorrange:
#        rrange = [min([x.values().min() for x in ratio_hists]), max(
    fig, ax1, ax2 = figureWithRatio(hists[0], xlabel, ylabel, [0, ymax] if ymax else None, rlabel, rrange, xlim=xlim)
    
    hep.histplot(
        hists,
        histtype="step",
        color=colors,
        label=labels,
        stack=False,
        ax=ax1,
        binwnorm=binwnorm,
    )
    
    if len(hists) > 1:
        hep.histplot(
            ratio_hists,
            histtype="step",
            color=colors[1:],
            label=labels[1:],
            yerr=False,
            stack=False,
            ax=ax2,
        )
        
    addLegend(ax1)
    return fig

