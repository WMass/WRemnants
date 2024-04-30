import argparse
import os
import mplhep as hep
import matplotlib.pyplot as plt
import numpy as np

from wremnants import logging, common
from wremnants import plot_tools
from wremnants.datasets.datagroups import Datagroups
from utilities import boostHistHelpers as hh
from utilities.io_tools import output_tools

parser = common.plot_parser()
parser.add_argument("infile", help="Output file of the analysis stage, containing ND boost histogrdams")
parser.add_argument("--procFilters", type=str, nargs="*", default="Zmumu", help="Filter to plot (default no filter, only specify if you want a subset")
parser.add_argument("--axes", type=str, nargs="+", default=["pt-ptGen","abs(eta)-absEtaGen"], help="Define for which axes the response matrix to be plotted")
parser.add_argument("-c", "--channels", type=str, nargs="+", choices=["plus", "minus", "all"], default=["all"], help="Select channel to plot")

args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)

groups = Datagroups(args.infile, filterGroups=args.procFilters, excludeGroups=None if args.procFilters else ['QCD'])

groups.setGenAxes(sum_gen_axes=[]) # set gen axes empty to not integrate over when loading

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
    "ptll" : r"$\mathrm{Reco}\ p_\mathrm{T}(\ell\ell)\ [\mathrm{GeV}]$",
    "ptW" : r"$\mathrm{Reco}\ p_\mathrm{T}(\ell\nu)\ [\mathrm{GeV}]$",
    "ptVGen" : "$\mathrm{Gen}\ p_\mathrm{T}(V)\ [\mathrm{GeV}]$",
    "abs(yll)" : "$\mathrm{Reco}\ |Y(\mathrm{V})|$",
    "absYVGen" : "$\mathrm{Gen}\ |Y(\mathrm{V})|$",
}


def get_purity(matrix, xbins, ybins):

    centers = xbins[:-1] + (xbins[1:] - xbins[:-1])/2    
    edges = ybins

    values = []
    for iBin, center in enumerate(centers):
        # find out in which gen bin(s) we are
        ilow = np.where(edges == edges[center > edges][-1])[0][0]
        ihigh = np.where(edges == edges[center < edges][0])[0][0]

        # sum corresponding diagonal element(s)
        diag = matrix[iBin, ilow:ihigh].sum()

        # sum reco bins
        reco = matrix[iBin, :].sum()

        values.append(diag/reco)

    return np.array(values)

def get_stability(matrix, xbins, ybins):
    # stability is same computation as purity with inverted axes
    return get_purity(matrix.T, ybins, xbins)



for g_name, group in datagroups.items():
    hist = group.hists["nominal"]

    for channel in args.channels:
        select = {} if channel == "all" else {"charge" : -1.j if channel == "minus" else 1.j}

        for axes_string in args.axes:
            axes = axes_string.split("-")

            if (groups.mode[0] == "w" or "wlike" in groups.mode) and axes[0] == "pt":
                genFlow=True
            else:
                genFlow=False

            if axes[0].startswith("abs("):
                # mirror axis at half
                hist2d = hist[select].project(axes[0][4:-1], *axes[1:])
                nbins = len(hist2d.axes.edges[0])-1
                values = hist2d.values(flow=genFlow)[:int(nbins/2)][::-1] + hist2d.values(flow=genFlow)[int(nbins/2):]
                xbins = hist2d.axes[0].edges[int(nbins/2):]
            else:
                hist2d = hist[select].project(*axes)
                values = hist2d.values(flow=genFlow)
                xbins = hist2d.axes[0].edges

            ybins = hist2d.axes[1].edges
            if genFlow:
                ybins = np.array([ybins[0]-(ybins[1]-ybins[0]), *ybins, ybins[-1]+(ybins[-1]-ybins[-2])])

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

            hep.cms.label(ax=ax, fontsize=20, label=args.cmsDecor, data=False)

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

            hep.cms.label(ax=ax, fontsize=20, label=args.cmsDecor, data=False)

            outfile = "stability_"+outname
            plot_tools.save_pdf_and_png(outdir, outfile)


            # plot response matrix
            fig = plt.figure()#figsize=(8*width,8))
            ax = fig.add_subplot() 

            ax.set_xlabel(translate_label[axes[0]])
            ax.set_ylabel(translate_label[axes[1]])

            hep.hist2dplot(values, xbins=xbins, ybins=ybins, cmin=0)#, labels=(xlabels,ylabels))

            # calculate condition number
            nans = np.isnan(values)
            nancount = np.count_nonzero(nans)
            if nancount:
                logger.warning(f"Found {nancount} NaNs in positions {np.argwhere(nans)}. Setting to zero")
                values[nans] = 0

            cond = np.linalg.cond(values)
            logger.info(f"Condition number: {cond}")
            plt.text(0.2, 0.94, round(cond,1), horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, color="white")

            # ax.set_xticks(np.arange(len(xlabels))+0.5)
            # ax.set_yticks(np.arange(len(xlabels))+0.5)
            # ax.set_xticklabels(xlabels, rotation = 90)
            # ax.set_yticklabels(xlabels)

            hep.cms.label(ax=ax, fontsize=20, label=args.cmsDecor, data=False)

            outfile = "responce_matrix_"+outname

            outfile += (f"_{channel}" if channel != "all" else "") 

            plot_tools.save_pdf_and_png(outdir, outfile)

            plot_tools.write_index_and_log(outdir, outfile, 
            #     yield_tables={"Values" : cov_mat}, nround=2 if "correlation" in matrix else 10,
                analysis_meta_info={args.infile : groups.getMetaInfo()},
                args=args,
            )
