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

from utilities import boostHistHelpers as hh, logging, input_tools, common, output_tools
from wremnants import plot_tools
from wremnants.unfolding_tools import get_bin, getProcessBins

hep.style.use(hep.style.ROOT)


parser = argparse.ArgumentParser()
parser.add_argument("infile", help="Combine fitresult root file")
# parser.add_argument("--ratioToData", action='store_true', help="Use data as denominator in ratio")
parser.add_argument("-o", "--outpath", type=str, default=os.path.expanduser("~/www/WMassAnalysis"), help="Base path for output")
parser.add_argument("-f", "--outfolder", type=str, default="", help="Subfolder for output")
parser.add_argument("-r", "--rrange", type=float, nargs=2, default=None, help="y range for ratio plot")
# parser.add_argument("--rebin", type=int, default=1, help="Rebin (for now must be an int)")
parser.add_argument("--ylim", type=float, nargs=2, help="Min and max values for y axis (if not specified, range set automatically)")
parser.add_argument("--yscale", type=float, help="Scale the upper y axis by this factor (useful when auto scaling cuts off legend)")
# parser.add_argument("--xlim", type=float, nargs=2, help="min and max for x axis")
parser.add_argument("-p", "--postfix", type=str, help="Postfix for output file name")
parser.add_argument("--debug", action='store_true', help="Print debug output")
parser.add_argument("--noData", action='store_true', help="Don't plot data")
# parser.add_argument("--noFill", action='store_true', help="Don't fill stack")
parser.add_argument("--scaleleg", type=float, default=1.0, help="Scale legend text")
parser.add_argument("--plots", type=str, nargs="+", default=["postfit"], choices=["prefit", "postfit"], help="Define which plots to make")
parser.add_argument("--lumi", type=float, default=16.8, help="Luminosity used in the fit, needed to get the absolute cross section")
parser.add_argument("-c", "--channels", type=str, nargs="+", choices=["plus", "minus", "all"], default=["plus", "minus", "all"], help="Select channel to plot")
parser.add_argument("--eoscp", action='store_true', help="Override use of xrdcp and use the mount instead")

args = parser.parse_args()

logger = logging.setup_logger("plotFitresult", 4 if args.debug else 3)

outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)

rfile = uproot.open(args.infile)

input_subdir = args.infile.split("/")[-2]


if input_subdir.startswith("W"):
    cm = mpl.colormaps["autumn"]
else:
    cm = mpl.colormaps["winter"]

cm = mpl.colormaps["gist_rainbow"]

cms_decor = "Preliminary"   

# labels
labelmap = {
    "Zmumu" : r"Z$\to\mu\mu$",
    "Zmumu_bkg" : r"Z$\to\mu\mu$ (bkg)",
    "Ztautau" : r"Z$\to\tau\tau$",
    "Wmunu_bkg" : r"W$^{\pm}\to\mu\nu$ (bkg)",
    "Wmunu" : r"W$^{\pm}\to\mu\nu$ ",
    "Wtaunu" : r"W$^{\pm}\to\tau\nu$",
    "Top" : "green",
    "Diboson" : "pink",
    "QCD" : "grey",
    "Fake" : "grey",
} 

def get_label(name):
    logger.debug(f"Get label for {name}")

    res = getProcessBins(name, axes=["ptVGen"])

    if res["proc"] in ["Zmumu", "Z"]:
        label = r"Z$\to\mu\mu$"
    elif res["proc"] in ["Wmunu", "W"]:
        charge = res.get("qGen", None)
        if charge is None:
            label = r"W$^{\pm}\to\mu\nu$" if charge else r"W$^{\pm}\to\mu\nu$"
        else:
            label = r"W$^{+}\to\mu\nu$" if charge else r"W$^{-}\to\mu\nu$"
    else: 
        raise RuntimeError(f"Unknown process {res['proc']}")

    if res.get("ptGen", -1) != -1:
        label += f"p_\mathrm{{T}}^{res['pt']}"

    if res.get("absEtaGen", -1) != -1:
        label += f"\abs{{\eta}}^{res['absEtaGen']}"

    if res.get("ptVGen", -1) != -1:
        label += f"p_\mathrm{{T}}^{res['ptVGen']}"

    if res.get("absYVGen", -1) != -1:
        label += f"\abs{{Y}}^{res['absYVGen']}"

    return name

# colors
colormap= {
    "Z" : "lightblue",
    "Z_bkg" : "lightblue",
    "Zmumu" : "lightblue",
    "Zmumu_bkg" : "lightblue",
    "Ztautau" : "darkblue",
    "W" : "darkred",
    "W_bkg" : "darkred",
    "Wmunu" : "darkred",
    "Wmunu_bkg" : "darkred",
    "Wtaunu" : "orange",
    "Top" : "green",
    "Diboson" : "pink",
    "QCD" : "grey",
    "Fake" : "grey",
    "Other" : "grey",
}


def get_color(i, nbins):
    icol = (1+i) / (nbins)
    return cm(icol) 

def make_yields_df(hists, procs, signal=None, per_bin=False):
    logger.debug(f"Make yield df for {procs}")

    if per_bin:
        def sum_and_unc(h):
            variances = h.variances() if h.variances() is not None else h.values()
            return (h.values(), np.sqrt(variances))   
    else:
        def sum_and_unc(h):
            variances = h.variances() if h.variances() is not None else h.values()
            return (sum(h.values()), np.sqrt(sum(variances)))

    if per_bin:
        entries = [(i, v[0], v[1]) for i,v in enumerate(zip(*sum_and_unc(hists[0])))]
    else:
        if signal is not None:
            entries = []
            for sig in signal:
                entries = [(sig, sum([ sum(v.values()) for k,v in zip(procs, hists) if sig in k]), np.sqrt(sum([ sum(v.variances()) for k,v in zip(procs, hists) if sig in k])))]
        else:
            entries = [(k, *sum_and_unc(v)) for k,v in zip(procs, hists)]

    return pd.DataFrame(entries, columns=["Process", "Yield", "Uncertainty"])



def plot(fittype, channel=None, data=True, stack=True, density=False, ratio=True, backgrounds=True):
    logger.info(f"Make {fittype} plot"+(f" in channel {channel}" if channel else ""))

    procs = [p for p in filter(lambda x: x.startswith("expproc_") and x.endswith(f"_{fittype};1"), rfile.keys())]

    info = pd.DataFrame([getProcessBins(p) for p in procs])

    info["sum"] = info["name"].apply(lambda x: sum(rfile[x].to_hist().values()))
    info.sort_values(by=["qGen", "ptGen", "absEtaGen", "ptVGen", "absYVGen", "sum"], inplace=True, ignore_index=True)

    proc_sig = info[~(info[[x for x in info.keys() if x.endswith("Gen")]].values ==-1).all(axis=1)]
    proc_bkg = info[(info[[x for x in info.keys() if x.endswith("Gen")]].values ==-1).all(axis=1)]

    # pre computation
    if "obs" not in [c.replace(";1","") for c in  rfile.keys()]:
        logger.error(f"Shapes not found in fitresult file, run combine with --saveHists --computeHistErrors to get shapes.")
        return
    hist_data = rfile["obs"].to_hist()
    nbins_data = len(*hist_data.axes.edges) - 1

    if channel == "minus":
        bin_lo = 0
        bin_hi = int(nbins_data/2)
    elif channel =="plus":
        bin_lo = int(nbins_data/2)
        bin_hi = nbins_data
    else:
        bin_lo = 0
        bin_hi = nbins_data

    nbins = bin_hi - bin_lo

    hist_data = rfile["obs"].to_hist()[bin_lo:bin_hi]

    # last bin of exp histograms contain xnorm counts, remove for plotting by taking range from obs histogram
    hist_pred = rfile[f"expfull_{fittype}"].to_hist()[bin_lo:bin_hi]

    #divide histogram by bin widths
    # if mode == "dilepton":
    #     bins = np.array(common.ptV_binning)
    #     bin_widths = bins[1:] - bins[:-1]
    # else: 
    bin_widths = np.ones(nbins)    

    hist_data /= bin_widths
    hist_pred /= bin_widths

    processes = [rfile[p].to_hist()[bin_lo:bin_hi]/bin_widths for p in proc_sig["name"]]
    names = [p.replace("expproc_","").replace(f"_{fittype};1","") for p in proc_sig["name"]]
    colors = [get_color(i, len(proc_sig)) for i in range(len(proc_sig))]
    labels = [get_label(p) for p in proc_sig["name"]]

    if backgrounds:
        processes += [rfile[p].to_hist()[bin_lo:bin_hi]/bin_widths for p in proc_bkg["name"]]
        names += [p.replace("expproc_","").replace(f"_{fittype};1","") for p in proc_bkg["name"]]
        colors += [colormap.get(p, "grey") for p in proc_bkg["proc"]]
        labels += [labelmap.get(p, p) for p in proc_bkg["proc"]]

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
        ylabel = "Events/bin"

    if ratio:
        fig, ax1, ax2 = plot_tools.figureWithRatio(hist_data, "Bin number", ylabel, ylim, "Data/Pred.", rrange)
    else:
        fig, ax1 = plot_tools.figure(hist_data, "Bin number", ylabel, ylim)

    if stack:
        histtype="fill"
    else:
        histtype="step"

    hep.histplot(
        processes,
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

    # plot_tools.addLegend(ax1, ncols=4, text_size=20*args.scaleleg)

    scale = max(1, np.divide(*ax1.get_figure().get_size_inches())*0.3)
    hep.cms.label(ax=ax1, lumi=float(f"{args.lumi:.3g}"), fontsize=20*args.scaleleg*scale, 
        label=cms_decor, data=not args.noData)

    outfile = f"{fittype}"
    outfile += "_"+input_subdir
    outfile += (f"_{args.postfix}" if args.postfix else "") 
    outfile += (f"_{channel}" if channel else "")
    outfile += (f"_unstacked" if not stack else "")

    plot_tools.save_pdf_and_png(outdir, outfile)

    # make yield tables
    processes = [s*bin_widths for s in processes]
    processes_yields = make_yields_df(processes, names)

    # summed up
    base_process = set(proc_sig["name"])
    summed_yields = make_yields_df(processes, names, signal=base_process)
    if not args.noData:
        summed_yields = pd.concat([summed_yields, make_yields_df([hist_data*bin_widths], ["Data"])])

    plot_tools.write_index_and_log(outdir, outfile, 
        yield_tables={"Processes" : processes_yields, "Summed processes": summed_yields},#, "Unstacked processes" : unstacked_yields},
        analysis_meta_info=None,
        args=args,
    )

for fit_type in args.plots:
    for channel in args.channels:
        plot(fit_type, channel, data=not args.noData)
        plot(fit_type, channel, data=False, stack=False, ratio=False, backgrounds=False, density=True)

if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
    output_tools.copy_to_eos(args.outpath, args.outfolder)
