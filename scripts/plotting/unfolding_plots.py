import mplhep as hep
import matplotlib.pyplot as plt
import uproot
import argparse
import os
import numpy as np
import matplotlib as mpl
import pandas as pd
import boost_histogram as bh
import hist
import pdb

from utilities import boostHistHelpers as hh, logging, common
from utilities.io_tools import input_tools, combinetf_input, output_tools
from wremnants import plot_tools
from wremnants import histselections as sel
from wremnants.datasets.datagroups import Datagroups

hep.style.use(hep.style.ROOT)

parser = common.plot_parser()
parser.add_argument("infile", type=str, help="Output file of the analysis stage, containing ND boost histogrdams")
parser.add_argument("--fitresult",  type=str, help="Combine fitresult root file")
parser.add_argument("-r", "--rrange", type=float, nargs=2, default=None, help="y range for ratio plot")
parser.add_argument("--ylim", type=float, nargs=2, help="Min and max values for y axis (if not specified, range set automatically)")
parser.add_argument("--yscale", type=float, help="Scale the upper y axis by this factor (useful when auto scaling cuts off legend)")
parser.add_argument("--debug", action='store_true', help="Print debug output")
parser.add_argument("--noData", action='store_true', help="Don't plot data")
parser.add_argument("--plots", type=str, nargs="+", default=["postfit"], choices=["prefit", "postfit"], help="Define which plots to make")
parser.add_argument("-c", "--channels", type=str, nargs="+", choices=["plus", "minus"], default=["plus", "minus"], help="Select channel to plot")
parser.add_argument("-n", "--baseName", type=str, help="Histogram name in the file (e.g., 'nominal', 'xnorm')", default="nominal")
parser.add_argument("--axes", type=str, nargs="+", choices=["ptGen", "absEtaGen", "qGen"], default=["ptGen", "absEtaGen"], help="Specify axes to construct reference histogram from infile")
parser.add_argument("--addTauToSignal", action='store_true', help="Events from the same process but from tau final states are added to the signal")

args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)

fitresult = combinetf_input.get_fitresult(args.fitresult)

datagroups = Datagroups(args.infile)
isW = datagroups.mode[0] == "w"

if isW:
    base_group = "Wenu" if datagroups.flavor == "e" else "Wmunu"
else:
    base_group = "Zee" if datagroups.flavor == "ee" else "Zmumu"

# if args.addTauToSignal:
#     # add tau processes to signal
#     datagroups.groups[base_group].addMembers(datagroups.groups[base_group.replace("mu","tau")].members)
#     datagroups.deleteGroup(base_group.replace("mu","tau"))

def plot(fittype, channel=None, data=True, stack=True, density=False, ratio=True, backgrounds=True):
    logger.info(f"Make {fittype} plot"+(f" in channel {channel}" if channel else ""))

    procs = [p for p in filter(lambda x: x.startswith("expproc_") and x.endswith(f"_{fittype};1"), fitresult.keys())]

    names = [p.replace("expproc_","").replace(f"_{fittype};1","") for p in procs]

    # load reference hists
    ref_hists = []
    colors = []
    labels = []
    if args.addTauToSignal:
        names = [n.replace("mu", "tau") for n in names] + names
    for g_name in names:
        group = datagroups.groups[g_name]
        for member in group.members:
            if isW and (
                (channel =="plus" and member.name.startswith("Wminus")) 
                or (channel =="minus" and member.name.startswith("Wplus"))
            ):
                continue

            if args.baseName in datagroups.results[member.name]["output"]:
                logger.debug(f"Load datagroups member {member.name}")
                histo = datagroups.results[member.name]["output"][args.baseName].get()
                histo = histo.project(*args.axes)

                if not isW and "qGen" in args.axes:
                    index_charge = 0 if channel == "minus" else 1
                    histo = histo[{"qGen": index_charge}]

                scale = datagroups.processScaleFactor(member)
                if group.scale:
                    scale *= group.scale(member)

                histo = hh.scaleHist(histo, scale, createNew=False)

                ref_hists.append(histo)
                colors.append(datagroups.groups[g_name].color)
                labels.append(datagroups.groups[g_name].label)


    # figure out bin widths
    edges = ref_hists[0].axes.edges
    if len(edges) == 1:
        edges = np.array(edges[0])
        binwidths = edges[1:] - edges[:-1]
    elif len(edges) == 2:
        xbins, ybins = edges
        xbinwidths = np.diff(xbins.flatten())
        ybinwidths = np.diff(ybins.flatten())
        binwidths = np.outer(xbinwidths, ybinwidths).flatten()
        edges = np.arange(0.5, len(binwidths)+1.5, 1.0)
    else:
        bins = np.product([len(e.flatten())-1 for e in edges])
        edges = np.arange(0.5, bins+1.5, 1.0)
        binwidths = edges[1:] - edges[:-1]

    nbins = len(binwidths)

    # load fitresult hists
    if channel == "minus":
        bin_lo = 0
        bin_hi = int(nbins)
    elif channel =="plus":
        bin_lo = int(nbins)
        bin_hi = int(nbins*2)

    if "obs" not in [c.replace(";1","") for c in  fitresult.keys()]:
        logger.error(f"Shapes not found in fitresult file, run combine with --saveHists --computeHistErrors to get shapes.")
        return
    
    hist_data = fitresult["obs"].to_hist()[bin_lo:bin_hi]/binwidths
    hist_pred = fitresult[f"expfull_{fittype}"].to_hist()[bin_lo:bin_hi]/binwidths

    hists = [fitresult[p].to_hist()[bin_lo:bin_hi]/binwidths for p in procs]

    # flatten ref hist and divide by bin widths
    hists_ref = []
    for h in ref_hists:
        histo = hist_pred.copy()
        histo.view(flow=False)[...] = h.view(flow=False).flatten()/binwidths
        hists_ref.append(histo)

    if args.ylim is None:
        if density:
            ylim = (0, 1.1)
        elif stack:
            ylim = (0, 1.1 * max(max(hist_data.values()), max(hist_pred.values())))
        else:
            ylim = (0, 1.1 * max([max(p.values(flow=False)) for p in processes]))
    else:
        ylim = args.ylim

    if args.rrange is None:
        rrange = [0.90,1.1] if fit_type=="prefit" else [0.95, 1.05]
    else:
        rrange = args.rrange

    if density:
        ylabel = "a.u."    
    else:
        if isW:
            process_label = "W"
        else:
            process_label = "Z"

        yLabel="d$\sigma ("+process_label+")$ [pb]"
        if "ptGen" in args.axes:
            ylabel = yLabel.replace("[pb]","[pb/GeV]")

    if ratio:
        fig, ax1, ax2 = plot_tools.figureWithRatio(hist_data, "Bin number", ylabel, ylim, "Data/Pred.", rrange)
    else:
        fig, ax1 = plot_tools.figure(hist_data, "Bin number", ylabel, ylim)

    if stack:
        histtype="fill"
    else:
        histtype="step"

    hep.histplot(
        hists_ref,
        xerr=False,
        yerr=False,
        histtype=histtype,
        color=colors,
        label=labels,
        stack=stack,
        density=density,
        ax=ax1,
        zorder=1,
    )

    if data:
        hep.histplot(
            hist_data,
            yerr=True,
            histtype="errorbar",
            color="black",
            label="Data",
            ax=ax1,
            alpha=1.,
            zorder=2,
        )    

    if ratio:
        hep.histplot(
            hh.divideHists(sum(hists_ref), hist_pred, cutoff=0.01, rel_unc=False),
            histtype="errorbar",
            color="black",
            label="Ref.",
            yerr=False,
            linewidth=2,
            ax=ax2
        )

        if data:
            hep.histplot(
                hh.divideHists(hist_data, hist_pred, cutoff=0.01, rel_unc=False),
                histtype="errorbar",
                color="black",
                label="Data",
                yerr=False,
                linewidth=2,
                ax=ax2
            )

    # uncertainties
    # need to divide by bin width
    axis = hist_pred.axes[0].edges
    #binwidth =  axis[1:] - axis[:-1]
    nom = hist_pred.values() #/ binwidth
    std = np.sqrt(hist_pred.variances()) #/ binwidth

    hatchstyle = '///'
    if stack:
        ax1.fill_between(axis, 
            np.append(nom+std, (nom+std)[-1]), 
            np.append(nom-std, (nom-std)[-1]),
            step='post',facecolor="none", zorder=2, hatch=hatchstyle, edgecolor="k", linewidth=0.0, label="Uncertainty")

    if ratio:
        ax2.fill_between(axis, 
            np.append((nom+std)/nom, ((nom+std)/nom)[-1]), 
            np.append((nom-std)/nom, ((nom-std)/nom)[-1]),
            step='post',facecolor="none", zorder=2, hatch=hatchstyle, edgecolor="k", linewidth=0.0)

        plot_tools.fix_axes(ax1, ax2, yscale=args.yscale)

    scale = max(1, np.divide(*ax1.get_figure().get_size_inches())*0.3)
    hep.cms.label(ax=ax1, lumi=float(f"{args.lumi:.3g}"), fontsize=20*args.scaleleg*scale, 
        label=args.cmsDecor, data=not args.noData)

    plot_tools.addLegend(ax1, ncols=2, text_size=20*args.scaleleg)

    outfile = f"{fittype}"
    outfile += (f"_{args.postfix}" if args.postfix else "") 
    outfile += (f"_{channel}" if channel else "")
    outfile += (f"_unstacked" if not stack else "")

    plot_tools.save_pdf_and_png(outdir, outfile)

    # make yield tables
    # processes = [s*bin_widths for s in processes]
    # processes_yields = make_yields_df(processes, names)

    # summed up
    # base_process = set(proc_sig["name"])
    # summed_yields = make_yields_df(processes, names, signal=base_process)
    # if not args.noData:
    #     summed_yields = pd.concat([summed_yields, make_yields_df([hist_data*bin_widths], ["Data"])])

    plot_tools.write_index_and_log(outdir, outfile, 
        # yield_tables={"Processes" : processes_yields, "Summed processes": summed_yields},#, "Unstacked processes" : unstacked_yields},
        analysis_meta_info={args.infile : datagroups.getMetaInfo()},
        args=args,
    )

for fit_type in args.plots:
    for channel in args.channels:
        plot(fit_type, channel, data=not args.noData)
        # plot(fit_type, channel, data=False, stack=False, ratio=False, backgrounds=False, density=True)

if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
    output_tools.copy_to_eos(args.outpath, args.outfolder)
