import mplhep as hep
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
import os
import hist
import numpy as np
import argparse
import pandas as pd
import itertools

from narf import ioutils

import scipy.stats

from utilities import common, logging, differential, boostHistHelpers as hh
from utilities.styles import styles
from wremnants import plot_tools
from utilities.io_tools import output_tools, combinetf_input, combinetf2_input

import pdb

hep.style.use(hep.style.ROOT)

parser = common.plot_parser()
parser.add_argument("infile", type=str, help="hdf5 file from combinetf2 or root file from combinetf1")
parser.add_argument("-r", "--rrange", type=float, nargs=2, default=[0.9,1.1], help="y range for ratio plot")
parser.add_argument("--ylim", type=float, nargs=2, help="Min and max values for y axis (if not specified, range set automatically)")
parser.add_argument("--logy", action='store_true', help="Make the yscale logarithmic")
parser.add_argument("--yscale", type=float, help="Scale the upper y axis by this factor (useful when auto scaling cuts off legend)")
parser.add_argument("--noRatio", action='store_true', help="Don't make the ratio in the plot")
parser.add_argument("--noData", action='store_true', help="Don't plot the data")
parser.add_argument("--normToData", action='store_true', help="Normalize MC to data")
parser.add_argument("--prefit", action='store_true', help="Make prefit plot, else postfit")
parser.add_argument("--selectionAxes", type=str, default=["charge", "passIso", "passMT", "cosThetaStarll"], 
    help="List of axes where for each bin a seperate plot is created")
parser.add_argument("--axlim", type=float, nargs='*', help="min and max for axes (2 values per axis)")
parser.add_argument("--invertAxes", action='store_true', help="Invert the order of the axes when plotting")
parser.add_argument("--noChisq", action='store_true', help="skip printing chisq on plot")
parser.add_argument("--dataName", type=str, default="Data", help="Data name for plot labeling")
parser.add_argument("--processGrouping", type=str, default=None, help="key for grouping processes")

args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)

fittype = "prefit" if args.prefit else "postfit"
ratio = not args.noRatio
data = not args.noData


# load .hdf5 file first, must exist in combinetf and combinetf2
fitresult_h5py = combinetf_input.get_fitresult(args.infile.replace(".root",".hdf5"))
if "results" in fitresult_h5py.keys():
    fitresult = ioutils.pickle_load_h5py(fitresult_h5py["results"])
    combinetf2 = True
elif os.path.isfile(args.infile.replace(".hdf5",".root")):
    fitresult = combinetf_input.get_fitresult(args.infile.replace(".hdf5",".root"))
    combinetf2 = False
else:
    raise IOError("Unknown source, input file must be either from combinetf2 or combinetf1 (in case of combinetf1 both .root and .hdf5 files must exist)")

translate_selection = {
    "charge": {
        0 : "minus",
        1 : "plus"
    }
}

def make_plot(h_data, h_inclusive, h_stack, axes, colors=None, labels=None, suffix="", chi2=None, meta=None, saturated_chi2=False, lumi=None):
    axes_names = [a.name for a in axes]

    if any(x in axes_names for x in ["ptll", "mll", "ptVgen", "ptVGen"]):
        # in case of variable bin width normalize to unit
        binwnorm = 1.0
        ylabel="Events/unit"
    else:
        binwnorm = None
        ylabel="Events/bin"
    
    histtype_data = "errorbar"
    histtype_mc = "fill"

    if any(x in axes_names for x in ["ptVgen","absYVgen","helicity"]):
        histtype_data = "step"
        histtype_mc = "errorbar"
    
    if len(h_data.axes) > 1:
        if "eta" in axes_names[-1]:
            axes_names = axes_names[::-1]
        if args.invertAxes:
            logger.info("invert eta order")
            axes_names = axes_names[::-1]

        # make unrolled 1D histograms
        h_data = hh.unrolledHist(h_data, binwnorm=binwnorm, obs=axes_names)
        h_inclusive = hh.unrolledHist(h_inclusive, binwnorm=binwnorm, obs=axes_names)
        h_stack = [hh.unrolledHist(h, binwnorm=binwnorm, obs=axes_names) for h in h_stack]

    if args.normToData:
        scale = h_data.values().sum()/h_inclusive.values().sum()
        h_stack = [hh.scaleHist(h, scale) for h in h_stack]
        h_inclusive = hh.scaleHist(h_inclusive, scale)

    axis_name = "_".join([a for a in axes_names])
    if len(axes_names) == 1:
        xlabel=styles.xlabels.get(axes_names[0])
    else:
        xlabel=f"{'-'.join([styles.xlabels.get(s,s).replace('(GeV)','') for s in axes_names])} bin"
    if ratio:
        fig, ax1, ax2 = plot_tools.figureWithRatio(h_data, xlabel, ylabel, args.ylim, f"1/Pred.", args.rrange)
    else:
        fig, ax1 = plot_tools.figure(h_data, xlabel, ylabel, args.ylim)

    hep.histplot(
        h_stack,
        xerr=False,
        yerr=False,
        histtype=histtype_mc,
        color=colors,
        label=labels,
        stack=True,
        density=False,
        binwnorm=binwnorm,
        ax=ax1,
        zorder=1,
        flow='none',
    )

    if data:
        hep.histplot(
            h_data,
            yerr=True,
            histtype=histtype_data,
            color="black",
            label=args.dataName,
            binwnorm=binwnorm,
            ax=ax1,
            alpha=1.,
            zorder=2,
            flow='none',
        )    

    if ratio:

        hep.histplot(
            hh.divideHists(h_inclusive, h_inclusive, cutoff=1e-8, rel_unc=True, flow=False, by_ax_name=False),
            histtype="step",
            color="grey",
            alpha=0.5,
            yerr=False,
            ax=ax2,
            linewidth=2,
            flow='none',
        )

        if data:
            hep.histplot(
                hh.divideHists(h_data, h_inclusive, cutoff=0.01, rel_unc=True),
                histtype="errorbar",
                color="black",
                label=args.dataName,
                yerr=True,
                linewidth=2,
                ax=ax2
            )

            # for uncertaity bands
            edges = h_inclusive.axes[0].edges

            # need to divide by bin width
            binwidth = edges[1:]-edges[:-1] if binwnorm else 1.
            if h_inclusive.storage_type != hist.storage.Weight:
                raise ValueError(f"Did not find uncertainties in {fittype} hist. Make sure you run combinetf with --computeHistErrors!")
            nom = h_inclusive.values() / binwidth
            std = np.sqrt(h_inclusive.variances()) / binwidth

            hatchstyle = '///'
            ax1.fill_between(edges, 
                    np.append(nom+std, (nom+std)[-1]), 
                    np.append(nom-std, (nom-std)[-1]),
                step='post',facecolor="none", zorder=2, hatch=hatchstyle, edgecolor="k", linewidth=0.0, label="Uncertainty")

            ax2.fill_between(edges, 
                    np.append((nom+std)/nom, ((nom+std)/nom)[-1]), 
                    np.append((nom-std)/nom, ((nom-std)/nom)[-1]),
                step='post',facecolor="none", zorder=2, hatch=hatchstyle, edgecolor="k", linewidth=0.0)

    scale = max(1, np.divide(*ax1.get_figure().get_size_inches())*0.3)

    fontsize = ax1.xaxis.label.get_size()

    if chi2 is not None:
        p_val = round(scipy.stats.chi2.sf(chi2[0], chi2[1])*100,1)
        if saturated_chi2:
            chi2_name = "\chi_{\mathrm{sat.}}^2/ndf"
        else:
            chi2_name = "\chi^2/ndf"
        if len(h_data.values())<100:
            plt.text(0.05, 0.94, f"${chi2_name}$", horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes,
                fontsize=fontsize)  
            plt.text(0.05, 0.86, f"$= {round(chi2[0],1)}/{chi2[1]} (p={p_val}\%)$", horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes,
                fontsize=fontsize)  
        else:
            plt.text(0.05, 0.94, f"${chi2_name} = {round(chi2[0],1)}/{chi2[1]} (p={p_val}\%)$", horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes,
                fontsize=fontsize)

    plot_tools.redo_axis_ticks(ax1, "x")
    plot_tools.redo_axis_ticks(ax2, "x")

    hep.cms.label(ax=ax1, lumi=float(f"{lumi:.3g}") if lumi is not None and args.dataName=="Data" else None, 
        fontsize=fontsize, 
        label=args.cmsDecor, data=data)

    if len(h_stack) < 10:
        plot_tools.addLegend(ax1, ncols=np.ceil(len(h_stack)/3), text_size=fontsize)
    plot_tools.fix_axes(ax1, ax2, yscale=args.yscale)

    to_join = [fittype, args.postfix, axis_name, suffix]
    outfile = "_".join(filter(lambda x: x, to_join))

    plot_tools.save_pdf_and_png(outdir, outfile)

    stack_yields = None
    unstacked_yields = None
    kwargs=dict()
    if meta is not None:
        if "meta_info_input" in meta:
            analysis_meta_info={"Combinetf2Output" : meta["meta_info"], "AnalysisOutput": meta["meta_info_input"]["meta_info"]}
        else:
            analysis_meta_info={"AnalysisOutput" : meta["meta_info"]}
        kwargs["analysis_meta_info"] = analysis_meta_info

    plot_tools.write_index_and_log(outdir, outfile, 
        yield_tables={
            "Stacked processes" : pd.DataFrame([(k, sum(h.values()), sum(h.variances())**0.5) for k,h in zip(labels, h_stack)], columns=["Process", "Yield", "Uncertainty"]), 
            "Unstacked processes" : pd.DataFrame([(k, sum(h.values()), sum(h.variances())**0.5) for k,h in zip([args.dataName, "Inclusive"], [h_data, h_inclusive])], columns=["Process", "Yield", "Uncertainty"])},
        args=args, **kwargs
    )

def make_plots(hist_data, hist_inclusive, hist_stack, axes, procs, labels, colors, channel="", *opts, **kwopts):
    if args.processGrouping is not None:
        if args.processGrouping in styles.process_supergroups:
            new_stack = {}
            for new_name, old_procs in styles.process_supergroups[args.processGrouping].items():
                stacks = [hist_stack[procs.index(p)] for p in old_procs if p in procs]
                if len(stacks) == 0:
                    continue
                new_stack[new_name] = hh.sumHists(stacks)  
            
            labels, colors, procs = styles.get_labels_colors_procs_sorted([k for k in new_stack.keys()])
            hist_stack = [new_stack[p] for p in procs]
        else:
            logger.warning(f"No supergroups found for input file with mode {args.processGrouping}, proceed without merging groups")

    # make plots in slices (e.g. for charge plus an minus separately)
    selection_axes = [a for a in axes if a.name in args.selectionAxes]
    if len(selection_axes) > 0:
        selection_bins = [np.arange(a.size) for a in axes if a.name in args.selectionAxes]
        other_axes = [a for a in axes if a not in selection_axes]

        for bins in itertools.product(*selection_bins):
            idxs = {a.name: i for a, i in zip(selection_axes, bins) }
            idxs_centers = {
                a.name: a.centers[i] if isinstance(a, (hist.axis.Regular, hist.axis.Variable)) else a.edges[i]
                for a, i in zip(selection_axes, bins)
            }

            h_data = hist_data[idxs]
            h_inclusive = hist_inclusive[idxs]
            h_stack = [h[idxs] for h in hist_stack]

            if "run" in [a.name for a in selection_axes]:
                idx = idxs["run"]
                lumis = common.run_edges_lumi
                lumi = np.diff(lumis)[idx]
                logger.info(f"Axis 'run' found in histogram selection_axes, set lumi to {lumi}")
                kwopts["run"] = lumi
            for a, i in idxs_centers.items():
                print(a,i)
            suffix = f"{channel}_" + "_".join([f"{a}{i}" for a, i in idxs_centers.items()])
            logger.info(f"Make plot for axes {[a.name for a in other_axes]}, in bins {idxs}")
            make_plot(h_data, h_inclusive, h_stack, other_axes, labels=labels, colors=colors, suffix=suffix, *opts, **kwopts)
    else:
        make_plot(hist_data, hist_inclusive, hist_stack, axes, labels=labels, colors=colors, suffix=channel, *opts, **kwopts)

if combinetf2:
    meta = ioutils.pickle_load_h5py(fitresult_h5py["meta"])
    command = meta["meta_info"]["command"]
    asimov = False
    if "-t-1" in command or "-t -1" in command or "-t" not in command:
        asimov = True
    meta_input=meta["meta_info_input"]
    procs = meta["procs"].astype(str)
    labels, colors, procs = styles.get_labels_colors_procs_sorted(procs)

    chi2=None
    if f"chi2_{fittype}" in fitresult and not args.noChisq:
        chi2 = fitresult[f"chi2_{fittype}"], fitresult[f"ndf_{fittype}"]

    for channel, info in meta_input["channel_info"].items():
        if channel.endswith("masked"):
            continue
        hist_data = fitresult["hist_data_obs"][channel].get()
        hist_inclusive = fitresult[f"hist_{fittype}_inclusive"][channel].get()
        hist_stack = fitresult[f"hist_{fittype}"][channel].get()
        hist_stack = [hist_stack[{"processes" : p}] for p in procs]
        
        if any(x in hist_data.axes.name for x in ["helicity"]):
            if asimov:
                hist_data.values()[...] = 100000*np.log(hist_data.values())
            or_vals = np.copy(hist_inclusive.values())
            hist_inclusive.values()[...] = 100000*np.log(hist_inclusive.values())
            hist_inclusive.variances()[...] = 100000*100000*(hist_inclusive.variances())/np.square(or_vals)
            for h in hist_stack:
                or_vals = np.copy(h.values())
                h.values()[...] = 100000*np.log(h.values())
                h.variances()[...] = 100000*100000*(h.variances())/np.square(or_vals)
                
        make_plots(hist_data, hist_inclusive, hist_stack, info["axes"], channel=channel, procs=procs, labels=labels, colors=colors, chi2=chi2, meta=meta, lumi=info["lumi"])
else:
    # combinetf1
    import ROOT

    procs = [k.replace("expproc_","").replace(f"_{fittype};1", "") for k in fitresult.keys() if fittype in k and k.startswith("expproc_") and "hybrid" not in k]
    labels, colors, procs = styles.get_labels_colors_procs_sorted(procs)

    if "meta" in fitresult_h5py:
        # the fit was probably done on a file generated via the hdf5 writer and we can use the axes information
        meta = ioutils.pickle_load_h5py(fitresult_h5py["meta"])
        ch_start=0
        for channel, info in meta["channel_info"].items():
            if channel.endswith("masked"):
                continue
            shape = [len(a) for a in info["axes"]]

            ch_end = ch_start+np.product(shape) # in combinetf1 the channels are concatenated and we need to index one after the other

            hist_data = fitresult["obs;1"].to_hist()
            values = np.reshape(hist_data.values()[ch_start:ch_end], shape)
            hist_data = hist.Hist(*info["axes"], storage=hist.storage.Weight(), data=np.stack((values, values), axis=-1))  

            # last bin can be masked channel; slice with [:nBins]
            hist_inclusive = fitresult[f"expfull_{fittype};1"].to_hist()
            hist_inclusive = hist.Hist(*info["axes"], storage=hist.storage.Weight(), 
                data=np.stack((np.reshape(hist_inclusive.values()[ch_start:ch_end], shape), np.reshape(hist_inclusive.variances()[ch_start:ch_end], shape)), axis=-1))  
            hist_stack = [fitresult[f"expproc_{p}_{fittype};1"].to_hist() for p in procs]
            hist_stack = [hist.Hist(*info["axes"], storage=hist.storage.Weight(), 
                data=np.stack((np.reshape(h.values()[ch_start:ch_end], shape), np.reshape(h.variances()[ch_start:ch_end], shape)), axis=-1)) for h in hist_stack]

            if not args.prefit and not args.noChisq:
                rfile = ROOT.TFile.Open(args.infile.replace(".hdf5",".root"))
                ttree = rfile.Get("fitresults")
                ttree.GetEntry(0)
                chi2 = [2*(ttree.nllvalfull - ttree.satnllvalfull), np.product([len(a) for a in info["axes"]]) - ttree.ndofpartial]
            else:
                chi2 = None

            make_plots(hist_data, hist_inclusive, hist_stack, info["axes"], channel=channel, procs=procs, labels=labels, colors=colors, chi2=chi2, meta=meta, saturated_chi2=True, lumi=info["lumi"])
            ch_start = ch_end
    else:
        # the fit was probably done on a file generated via the root writer and we can't use the axes information

        # get axes from the directory name
        filename_parts = [x for x in filter(lambda x: x, args.infile.split("/"))]
        analysis = filename_parts[-2].split("_")[0]
        if analysis=="ZMassDilepton":
            all_axes = {
                # "mll": hist.axis.Regular(60, 60., 120., name = "mll", overflow=False, underflow=False),
                "mll": hist.axis.Variable([60,70,75,78,80,82,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,100,102,105,110,120], name = "mll", overflow=False, underflow=False),
                "etaAbsEta": hist.axis.Variable([-2.4, -2.0, -1.6, -1.4, -1.2, -1.0, -0.6, 0.0, 0.6, 1.0, 1.2, 1.4, 1.6, 2.0, 2.4], name = "etaAbsEta", overflow=False, underflow=False),
                "cosThetaStarll": hist.axis.Regular(2, -1., 1., name = "cosThetaStarll", underflow=False, overflow=False),
                "yll": hist.axis.Regular(20, -2.5, 2.5, name = "yll", overflow=False, underflow=False),
                "ptll": hist.axis.Variable(common.get_dilepton_ptV_binning(False), name = "ptll", underflow=False, overflow=False),
            }
        elif analysis=="ZMassWLike":
            all_axes = {
                "pt": hist.axis.Regular(34, 26, 60, name = "pt", overflow=False, underflow=False),
                "eta": hist.axis.Regular(48, -2.4, 2.4, name = "eta", overflow=False, underflow=False),
                "charge": common.axis_charge,
                "ptGen": hist.axis.Regular(33, 27, 60, name = "ptGen", overflow=False, underflow=False),
                "absEtaGen": hist.axis.Variable(differential.eta_binning, name = "absEtaGen", overflow=False, underflow=False),
                "qGen": common.axis_charge,
            }
        elif analysis=="WMass":
            all_axes = {
                "pt": hist.axis.Regular(30, 26, 56, name = "pt", overflow=False, underflow=False),
                # "pt": hist.axis.Regular(31, 26, 57, name = "pt", overflow=False, underflow=False),
                #"pt": hist.axis.Regular(29, 27, 56, name = "ptGen", overflow=False, underflow=False),
                "eta": hist.axis.Regular(48, -2.4, 2.4, name = "eta", overflow=False, underflow=False),
                "charge": common.axis_charge,
                "passIso": common.axis_passIso,
                "passMT": common.axis_passMT,
                "ptGen": hist.axis.Regular(29, 27, 56, name = "ptGen", overflow=False, underflow=False),
                "absEtaGen": hist.axis.Variable(differential.eta_binning, name = "absEtaGen", overflow=False, underflow=False),
                "qGen": common.axis_charge,
            }
        else:
            raise ValueError(f"Unknown analysis {analysis}, can't set the axes")

        axes = [all_axes[part] for part in filename_parts[-2].split("_") if part in all_axes.keys()]
        if args.axlim:
            nv = len(args.axlim)
            if nv % 2:
                raise ValueError("if --axlim is specified it must have two values per axis!")
            axlim = np.array(args.axlim).reshape((int(nv/2), 2))
            axes = [ax if lim is not None else hist.axis.Variable(ax.edges[(ax.edges >= lim[0]) & (ax.edges <= lim[1])]) 
                        for ax,lim in itertools.zip_longest(axes, axlim)]
        shape = [len(a) for a in axes]

        hist_data = fitresult["obs;1"].to_hist()
        nBins = hist_data.shape[0]
        values = np.reshape(hist_data.values(), shape)
        hist_data = hist.Hist(*axes, storage=hist.storage.Weight(), data=np.stack((values, values), axis=-1))  

        # last bin can be masked channel; slice with [:nBins]
        hist_inclusive = fitresult[f"expfull_{fittype};1"].to_hist()[:nBins]
        hist_inclusive = hist.Hist(*axes, storage=hist.storage.Weight(), 
            data=np.stack((np.reshape(hist_inclusive.values(), shape), np.reshape(hist_inclusive.variances(), shape)), axis=-1))  
        hist_stack = [fitresult[f"expproc_{p}_{fittype};1"].to_hist()[:nBins] for p in procs]
        hist_stack = [hist.Hist(*axes, storage=hist.storage.Weight(), 
            data=np.stack((np.reshape(h.values(), shape), np.reshape(h.variances(), shape)), axis=-1)) for h in hist_stack]

        if not args.prefit:
            rfile = ROOT.TFile.Open(args.infile.replace(".hdf5",".root"))
            ttree = rfile.Get("fitresults")
            ttree.GetEntry(0)
            chi2 = [2*(ttree.nllvalfull - ttree.satnllvalfull), np.product([len(a) for a in axes]) - ttree.ndofpartial]
        else:
            chi2 = None

        make_plots(hist_data, hist_inclusive, hist_stack, axes, procs=procs, labels=labels, colors=colors, chi2=chi2, saturated_chi2=True)

if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
    output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)
