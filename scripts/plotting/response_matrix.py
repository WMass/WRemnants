import argparse
import os
import mplhep as hep
import matplotlib.pyplot as plt
import numpy as np

from wremnants import logging
from wremnants import plot_tools
from wremnants.datasets.datagroups import Datagroups
from utilities import boostHistHelpers as hh, output_tools

import pdb

parser = argparse.ArgumentParser()
parser.add_argument("infile", help="Output file of the analysis stage, containing ND boost histogrdams")
parser.add_argument("--debug", action='store_true', help="Print debug output")
parser.add_argument("-o", "--outpath", type=str, default=os.path.expanduser("~/www/WMassAnalysis"), help="Base path for output")
parser.add_argument("-f", "--outfolder", type=str, default="test", help="Subfolder for output")
parser.add_argument("--procFilters", type=str, nargs="*", default="Zmumu", help="Filter to plot (default no filter, only specify if you want a subset")
parser.add_argument("-p", "--postfix", type=str, help="Postfix for output file name")
parser.add_argument("--axes", type=str, nargs="+", default=["pt-ptGen","abs(eta)-absEtaGen"], help="Define for which axes the response matrix to be plotted")
parser.add_argument("-c", "--channels", type=str, nargs="+", choices=["plus", "minus", "all"], default=["all"], help="Select channel to plot")

args = parser.parse_args()

logger = logging.setup_logger("makeDataMCStackPlot", 4 if args.debug else 3)

outdir = output_tools.make_plot_dir(args.outpath, args.outfolder)

groups = Datagroups(args.infile, filterGroups=args.procFilters, excludeGroups=None if args.procFilters else ['QCD'])

groups.setGenAxes([]) # set gen axes empty to not integrate over when loading

cms_decor = "Preliminary"

if "Wmunu" in groups.groups:
    groups.copyGroup("Wmunu", "Wmunu_qGen0", member_filter=lambda x: x.name.startswith("Wminus"))
    groups.copyGroup("Wmunu", "Wmunu_qGen1", member_filter=lambda x: x.name.startswith("Wplus"))

    groups.deleteGroup("Wmunu")

if "Zmumu" in groups.groups:
    groups.groups["Zmumu"].deleteMembers([m for m in groups.groups["Zmumu"].members if "BkgZmumu" in m.name])


datasets = groups.getNames()
logger.info(f"Will plot datasets {datasets}")

groups.loadHistsForDatagroups("nominal", syst="nominal", procsToRead=datasets)

datagroups = groups.getDatagroups()

translate_label = {
    "pt" : "$\mathrm{Reco}\ p_\mathrm{T}\ [\mathrm{GeV}]$",
    "ptGen" : "$\mathrm{Gen}\ p_\mathrm{T}\ [\mathrm{GeV}]$",
    "abs(eta)" : "$\mathrm{Reco}\ |\eta|$",
    "absEtaGen" : "$\mathrm{Gen}\ |\eta|$",
    "ptll" : "$\mathrm{Reco}\ p_\mathrm{T}(V)\ [\mathrm{GeV}]$",
    "ptVGen" : "$\mathrm{Gen}\ p_\mathrm{T}(V)\ [\mathrm{GeV}]$",
    "abs(yll)" : "$\mathrm{Reco}\ |Y(\mathrm{V})|$",
    "absYVGen" : "$\mathrm{Gen}\ |Y(\mathrm{V})|$",
}


def get_purity(matrix, xbins, ybins, flow=False):

    centers = xbins[:-1] + (xbins[1:] - xbins[:-1])/2    
    edges = ybins

    values = []
    for iBin, center in enumerate(centers):
        # find out in which gen bin(s) we are
        ilow = np.where(edges == edges[center > edges][-1])[0][0] + int(flow)
        ihigh = np.where(edges == edges[center < edges][0])[0][0] + int(flow)

        # sum corresponding diagonal element(s)
        diag = matrix[iBin, ilow:ihigh].sum()

        # sum reco bins
        reco = matrix[iBin, :].sum()

        values.append(diag/reco)

    return np.array(values)

def get_stability(matrix, xbins, ybins, flow=False):
    # stability is same computation as purity with inverted axes
    return get_purity(matrix.T, ybins, xbins, flow=flow)



for g_name, group in datagroups.items():
    hist = group.hists["nominal"]

    for channel in args.channels:
        select = {} if channel == "all" else {"charge" : -1.j if channel == "minus" else 1.j}

        for axes_string in args.axes:
            axes = axes_string.split("-")

            if axes[0].startswith("abs("):
                # mirror axis at half
                hist2d = hh.projectNoFlow(hist[select], [axes[0][4:-1], *axes[1:]])
                nbins = len(hist2d.axes.edges[0])-1
                values = hist2d.values()[:int(nbins/2)][::-1] + hist2d.values()[int(nbins/2):]
                xbins = hist2d.axes[0].edges[int(nbins/2):]
            else:
                hist2d = hh.projectNoFlow(hist[select], axes)
                values = hist2d.values()
                xbins = hist2d.axes[0].edges

            ybins = hist2d.axes[1].edges

            outname = g_name+"_"+"_".join([a.replace("(","").replace(")","") for a in axes])

            # plot purity
            fig = plt.figure(figsize=(8,4))
            ax = fig.add_subplot() 
            ax.set_xlabel(translate_label[axes[0]])
            ax.set_ylabel("Purity")

            purity = get_purity(values, xbins, ybins)

            hep.histplot(purity, xbins, color="blue")
            
            range_y = max(purity) - min(purity)
            min_y = min(purity) - range_y*0.1
            max_y = max(purity) + range_y*0.1

            ax.set_xlim([min(xbins), max(xbins)])
            ax.set_ylim([min_y, max_y])

            hep.cms.label(ax=ax, fontsize=20, label=cms_decor, data=False)

            outfile = "purity_"+outname
            plot_tools.save_pdf_and_png(outdir, outfile)



            # plot stability
            fig = plt.figure(figsize=(8,4))
            ax = fig.add_subplot() 
            ax.set_xlabel(translate_label[axes[1]])
            ax.set_ylabel("Stability")

            stability = get_stability(values, xbins, ybins)

            hep.histplot(stability, ybins, color="red")

            range_y = max(stability) - min(stability)
            min_y = min(stability) - range_y*0.1
            max_y = max(stability) + range_y*0.1

            ax.set_xlim([min(ybins), max(ybins)])
            ax.set_ylim([min_y, max_y])

            hep.cms.label(ax=ax, fontsize=20, label=cms_decor, data=False)

            outfile = "stability_"+outname
            plot_tools.save_pdf_and_png(outdir, outfile)


            # plot response matrix
            fig = plt.figure()#figsize=(8*width,8))
            ax = fig.add_subplot() 

            ax.set_xlabel(translate_label[axes[0]])
            ax.set_ylabel(translate_label[axes[1]])

            hep.hist2dplot(values, xbins=xbins, ybins=ybins, cmin=0)#, labels=(xlabels,ylabels))

            # calculate condition number
            cond = np.linalg.cond(values)
            logger.info(f"Condition number: {cond}")
            plt.text(0.2, 0.94, round(cond,1), horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, color="white")

            # ax.set_xticks(np.arange(len(xlabels))+0.5)
            # ax.set_yticks(np.arange(len(xlabels))+0.5)
            # ax.set_xticklabels(xlabels, rotation = 90)
            # ax.set_yticklabels(xlabels)

            hep.cms.label(ax=ax, fontsize=20, label=cms_decor, data=False)

            outfile = "responce_matrix_"+outname

            outfile += (f"_{channel}" if channel != "all" else "") 

            plot_tools.save_pdf_and_png(outdir, outfile)

            # plot_tools.write_index_and_log(outdir, outfile, 
            #     yield_tables={"Values" : cov_mat}, nround=2 if "correlation" in matrix else 10,
            #     analysis_meta_info=None,
            #     args=args,
            # )