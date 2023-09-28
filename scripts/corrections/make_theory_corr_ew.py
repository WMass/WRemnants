import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib import cm, ticker
import lz4.frame
import pickle
from wremnants import plot_tools, theory_corrections, theory_tools
from utilities import boostHistHelpers as hh
from utilities import common, input_tools, output_tools, logging
import hist
import argparse
import os
import h5py
import narf
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", nargs="+", type=str, default=["w_z_gen_dists_scetlib_dyturboCorr_ewinput.hdf5"], help="File containing EW hists")
parser.add_argument("--nums", nargs="+", type=str, default=["horace-nlo"], help="Numerators")
parser.add_argument("--den", type=str, default="horace-lo-photos", help="Denominatos")
parser.add_argument("--normalize", action="store_true", default=False, help="Normalize distributions before computing ratio")
parser.add_argument("--noSmoothing", action="store_true", default=False, help="Disable smoothing of corrections")
parser.add_argument("--debug", action='store_true', help="Print debug output")
parser.add_argument("--plots", nargs="*", type=str, default=["1D","2D"], choices=["1D","2D", "2Derr"], help="What plots to produce")
parser.add_argument("--baseName", default="nominal_ew", type=str, help="histogram name")
parser.add_argument("--project", default=["ewMll", "ewLogDeltaM"], nargs="*", type=str, help="axes to project to")
parser.add_argument("--showFlow", action='store_true', help="Show underlfow and overflow bins in plots")
parser.add_argument("--outpath", type=str, default=f"{common.data_dir}/TheoryCorrections", help="Output path")
parser.add_argument("-p", "--postfix", type=str, help="Postfix for plots and correction files")
parser.add_argument("-o", "--plotdir", type=str, help="Output directory for plots")

args = parser.parse_args()

logger = logging.setup_logger("make_theory_corr_ew", 4 if args.debug else 3)

procs = ['ZToMuMu', 'WplusToMuNu', 'WminusToMuNu']
charge_dict = {'ZToMuMu': 0, 'WplusToMuNu': 1, 'WminusToMuNu': 0}

procs_dict = {
    "ZToMuMu": "ZmumuPostVFP",
    "WminusToMuNu": "WminusmunuPostVFP",
    "WplusToMuNu": "WplusmunuPostVFP",
}

text_dict = {
    "ZToMuMu": r"$\mathrm{Z}\rightarrow\mu\mu$",
    "WplusToMuNu": r"$\mathrm{W}^+\rightarrow\mu\nu$",
    "WminusToMuNu": r"$\mathrm{W}^-\rightarrow\mu\nu$"
}

project = args.project

# file created with `python WRemnants/scripts/histmakers/w_z_gen_dists.py --skipAngularCoeffs --filter horace -p ewinput`

res, meta, _ = input_tools.read_infile(args.input)

corrh = {}

labels = {
    "n": r"$N^{\gamma}$",
    "pt": r"$\log_{10}(p_\mathrm{T}^{\gamma})$",
    "eta": r"$\eta^{\gamma}$",
    "ewPtll": r"$p_\mathrm{T}^{\ell\ell}$",
    "ewPTll": r"$p_\mathrm{T}^{\ell\ell}$",
    "Ptll": r"$p_\mathrm{T}^{\ell\ell}$",
    "PTll": r"$p_\mathrm{T}^{\ell\ell}$",
    "Yll": r"$Y^{\ell\ell}$", 
    "Mll": r"$m^{\ell\ell}$", 
    "ewMll": r"$m^{\ell\ell}$", 
    "PTlly": r"$p_\mathrm{T}^{\ell\ell\gamma}$", 
    "Ylly": r"$Y^{\ell\ell\gamma}$", 
    "Mlly": r"$m^{\ell\ell\gamma}$", 
    "ewMlly": r"$m^{\ell\ell\gamma}$",
    "ewLogDeltaM": r"$\log_{10}(m^{\ell\ell\gamma} - m^{\ell\ell})$",
    "ewPhosPtSum": r"$\sum (p_\mathrm{T}^{\gamma})$"
}


colors = mpl.colormaps["tab10"]

def make_plot_2d(h, name, proc, plot_error=False, cmin=None, cmax=None, flow=True, density=False, log=False):

    logger.info(f"Make plot {name} with axes {h.axes.name}")
    # average over bins
    h2d = h.project(*project)

    xlabel = labels.get(project[0],project[0])
    ylabel = labels.get(project[1],project[1])

    if flow:
        xedges, yedges = plot_tools.extendEdgesByFlow(h2d)
    else:
        edges = h2d.axes.edges
        xedges = np.reshape(edges[0], len(edges[0]))
        yedges = edges[1][0]

    if density:
        xbinwidths = np.diff(xedges)
        ybinwidths = np.diff(yedges)
        binwidths = np.outer(xbinwidths, ybinwidths) 
        h2d.values(flow=flow)[...] = h2d.values(flow=flow) / binwidths

    if plot_error:
        # plot relative errors instead
        h2d.values(flow=flow)[...] = np.sqrt(hh.relVariance(h2d.values(flow=flow), h2d.variances(flow=flow), fillOnes=True))

    xlim = (xedges[0],xedges[-1])
    ylim = (yedges[0],yedges[-1])

    fig, ax = plot_tools.figure(h2d, xlabel=xlabel, ylabel=ylabel, cms_label="Preliminary", automatic_scale=False, width_scale=1.2, xlim=xlim, ylim=ylim)

    if log:
        cmin = min(h2d.values(flow=flow)[h2d.values(flow=flow)>0]) if cmin is None else cmin # smallest value that is not 0
        cmax = h2d.values(flow=flow).max() if cmax is None else cmax
        colormesh = ax.pcolormesh(xedges, yedges, h2d.values(flow=flow).T, norm=LogNorm(vmin=cmin, vmax=cmax), cmap=cm.RdBu)
    else:
        cmin = 0 if cmin is None else cmin
        cmax = h2d.values(flow=flow).max() if cmax is None else cmax
        crange = max((cmax-1), (1-cmin))
        colormesh = ax.pcolormesh(xedges, yedges, h2d.values(flow=flow).T, vmin=1-crange, vmax=1+crange, cmap=cm.RdBu)

    cbar = fig.colorbar(colormesh, ax=ax)

    ax.text(1.0, 1.003, text_dict[proc], transform=ax.transAxes, fontsize=30,
            verticalalignment='bottom', horizontalalignment="right")

    output_tools.make_plot_dir(*args.plotdir.rsplit("/", 1))
    plot_name = f"hist2d_{name}_{proc}"
    if args.noSmoothing and "_div_" in name:
        plot_name += "_noSmoothing"
    if args.normalize:
        plot_name += "_normalize"
    if log:
        plot_name += "_log"
    if args.postfix:
        plot_name += f"_{args.postfix}"
    plot_tools.save_pdf_and_png(args.plotdir, plot_name)
    plot_tools.write_index_and_log(args.plotdir, plot_name, args=args, analysis_meta_info=meta[0])

def make_plot_1d(hists, name, proc, axis, ratio=False, normalize=False, xmin=None, xmax=None, ymin=None, ymax=None, flow=True, density=False):
    logger.info(f"Make 1D plot for {name} with axis {axis}")

    if not isinstance(hists, list):
        hists = [hists]
        name = [name]
    
    h1ds = [h.project(axis) for h in hists]

    if flow:
        xedges = plot_tools.extendEdgesByFlow(h1ds[0])
    else:
        xedges = h1ds[0].axes.edges[0]

    if normalize:
        h1ds = [h/np.sum(h.values(flow=flow)) for h in h1ds]
    if density:
        for i, h1d in enumerate(h1ds):
            binwidths = xedges[1:]-xedges[:-1]
            hh.scaleHist(h1d, 1./binwidths, createNew=False)

    ymax = ymax if ymax is not None else max([max(h.values(flow=args.showFlow)) for h in h1ds])
    ymin = ymin if ymin is not None else min([min(h.values(flow=args.showFlow)) for h in h1ds])
    yrange = ymax - ymin
    ymin = ymin if ymin == 0 else ymin - yrange*0.3
    ymax = ymax + yrange*0.3

    if xmin is not None:
        xlim = (xmin, xmax)
    else:
        xlim = (xedges[0],xedges[-1])

    if ratio:
        ylabel = "1/{0}".format(name[0].split("_div_")[-1])
    else:
        ylabel = "a.u."

    fig, ax = plot_tools.figure(h1ds[0], xlabel=labels.get(axis,project[0]), ylabel=ylabel, cms_label="Preliminary", automatic_scale=False, width_scale=1.2,
        ylim=(ymin, ymax), xlim=xlim)

    if ratio:
        ax.plot([min(xedges), max(xedges)], [1,1], color="black", linestyle="--")

    outname = ""
    for i, h1d in enumerate(h1ds):
        y = h1d.values(flow=flow)
        err = np.sqrt(h1d.variances(flow=flow))

        ax.stairs(y, xedges, color=colors(i), label=name[i].split("_div_")[0])
        ax.bar(x=xedges[:-1], height=2*err, bottom=y - err, width=np.diff(xedges), align='edge', linewidth=0, alpha=0.3, color=colors(i), zorder=-1)

    ax.text(1.0, 1.003, text_dict[proc], transform=ax.transAxes, fontsize=30,
            verticalalignment='bottom', horizontalalignment="right")
    plot_tools.addLegend(ax, ncols=1, text_size=12)

    output_tools.make_plot_dir(*args.plotdir.rsplit("/", 1))
    plot_name = f"hist{outname}_{axis}_{proc}_{name[-1]}"
    if ratio:
        plot_name += "_div"
    if args.noSmoothing and "_div_" in name[0]:
        plot_name += "_noSmoothing"
    if args.normalize:
        plot_name += "_normalize"
    if args.postfix:
        plot_name += f"_{args.postfix}"
    plot_tools.save_pdf_and_png(args.plotdir, plot_name)
    plot_tools.write_index_and_log(args.plotdir, plot_name, args=args, analysis_meta_info=meta[0])

for proc in procs:

    # Make 2D ratio
    logger.info(f'Make 2D ratio for {proc}')
    def prepare_hist(name):
        if name == 'MiNNLO':
            proc_name = procs_dict[proc]
        else:
            proc_name = f'{proc}_{name}'

        if proc_name not in res:
            return None

        histo = res[proc_name]['output'][args.baseName].get()
        logger.info(f'Integrals for {name} {np.sum(histo.values(flow=True))}')

        if args.normalize:
            histo = hh.normalize(histo)
            logger.info(f'Integral for {name} after normalizing {np.sum(histo.values(flow=True))}')
        else:
            histo = hh.scaleHist(histo, res[proc_name]["dataset"]["xsec"]*10e6/res[proc_name]['weight_sum'], createNew=False)
            logger.info(f'Integral for {name} after scaling {np.sum(histo.values(flow=True))}')

        return histo
    
    nums = []
    hnums = []
    for num in args.nums:
        hnum = prepare_hist(num)
        if hnum is not None:
            nums.append(num)
            hnums.append(hnum)

    hden = prepare_hist(args.den)

    if hden is None:
        logger.warning(f"Denominator {args.den} for process {proc} not found! Continue with next process.")
        continue

    if len(hnums) < 1:
        logger.warning(f"No nominator found for {args.nums} for process {proc}! Continue with next process.")
        continue

    h2Dratios = []
    corrhs = {}

    def make_correction(h1, h2, name):
        hratio = hh.divideHists(h1, h2)
        if not args.noSmoothing and not len(hratio.axes)>1:
            # 2D smoothing
            hratio = hh.smoothTowardsOne(hratio)
            logger.info('Integrals after smoothing {0} {1}'.format(np.sum(h2.values(flow=True)), np.sum(h2.values(flow=True)*hratio.values(flow=True))))
            if args.normalize:
                scale = np.sum(h2.values(flow=True)) / np.sum(h2.values(flow=True)*hratio.values(flow=True))
                hratio = hh.scaleHist(hratio, scale)
                logger.info('Integrals after adjustment {0} {1}'.format(np.sum(h2.values(flow=True)), np.sum(h2.values(flow=True)*hratio.values(flow=True))))

        if len(hratio.axes)>1:
            h2Dratios.append(hratio)

        # Add charge axis
        if proc[0] == 'W':
            axis_charge = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "charge")
        elif proc[0] == 'Z':
            axis_charge = hist.axis.Regular(1, -1., 1., underflow=False, overflow=False, name = "charge")
        hcharge = hist.Hist(*hratio.axes, axis_charge, storage=hratio._storage_type())
        hcharge.view(flow=True)[...,charge_dict[proc]] = hratio.view(flow=True)

        # Variations: 0=original MiNNLO, 1=Horace NLO, 2=mirrored (doubled)
        hones = hist.Hist(*hcharge.axes, storage=hist.storage.Double())
        hones.values(flow=True)[...,charge_dict[proc]] = np.ones(hratio.values(flow=True).shape)
        hmirror = hist.Hist(*hcharge.axes, storage=hist.storage.Double())
        hmirror.values(flow=True)[...] = 2*hcharge.values(flow=True) - hones.values(flow=True)
        if args.normalize:
            mirrorscale = np.sum(h2.values(flow=True)) / np.sum(h2.values(flow=True)*hmirror.values(flow=True)[...,0,charge_dict[proc]])
            logger.info(f'{proc} mirrorscale = {mirrorscale}')
            hmirror = hh.scaleHist(hmirror, mirrorscale)

        # Add syst axis
        if name not in corrh:
            corrh[name] = {}
        corrh[name][proc] = hist.Hist(*hcharge.axes, hist.axis.Regular(3, 0, 3, underflow=False, overflow=False, name="systIdx"), storage=hist.storage.Double())
        corrh[name][proc].values(flow=True)[...,0] = hones.values(flow=True)
        corrh[name][proc].values(flow=True)[...,1] = hcharge.values(flow=True)
        corrh[name][proc].values(flow=True)[...,2] = hmirror.values(flow=True)

        corrhs[name] = corrh[name]

    for num, hnum in zip(nums, hnums):
       
        make_correction(hnum, hden, f"{num}ew")

        for ax in project:
            hnum1D = hnum.project(ax)
            hden1D = hden.project(ax)

            make_correction(hnum1D, hden1D, f"{num}ew_{ax}")

    if "2D" in args.plots and len(project) > 1:
        logger.info("Make 2D control plots")
        make_plot_2d(hden, args.den, proc, density=True, cmin=10e-6, cmax=10e6, log=True)
        if "2Derr" in args.plots:
            make_plot_2d(hden, f"{args.den}_err", proc, plot_error=True, log=True)

        for i, hnum in enumerate(hnums):
            make_plot_2d(hnum, nums[i], proc, density=True, cmin=10e-6, cmax=10e6, log=True)
            if "2Derr" in args.plots:
                make_plot_2d(hnum, f"{nums[i]}_err", proc, plot_error=True, log=True)

        for i, h2Dratio in enumerate(h2Dratios):
            make_plot_2d(h2Dratio, f"{nums[i]}_div_{args.den}", proc, cmin=0.8, cmax=1.2)
            if "2Derr" in args.plots:
                make_plot_2d(h2Dratio, f"{nums[i]}_div_{args.den}_err", proc, plot_error=True, log=True)

    if "1D" in args.plots:
        logger.info("Make 1D control plots")

        for ax in project:
            # xmin, xmax = 60, 120
            xmin, xmax = None, None

            if ax in ["Ptll", "ewPtll", "ewPTll","PTll", "PTlly", "Yll", "Ylly"]:
                ymin, ymax = 0.98, 1.02
            elif ax in ["ewMll", "Mll"]:
                ymin, ymax = 0.95, 1.05
            elif ax in ["ewMlly", "Mlly", "pt"]:
                ymin, ymax = 0.8, 1.2
            elif ax in ["ewLogDeltaM"]:
                ymin, ymax = 0., 2.
            else:
                ymin, ymax = 0.9, 1.1

            hratios1D = [hh.divideHists(hnum.project(ax), hden.project(ax)) for hnum in hnums]

            make_plot_1d([*hnums, hden], [*nums, args.den], proc, ax, xmin=xmin, xmax=xmax, ymin=0, density=True)
            make_plot_1d(hratios1D, [f"{n}_div_{args.den}" for n in nums], proc, ax, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, ratio=True)

for num, corrh in corrhs.items():
    outname = num.replace('-', '')
    if args.postfix is not None:
        outname += f"_{args.postfix}"
    if 'ZToMuMu' in corrh:
        outfile = f"{args.outpath}/{outname}CorrZ.pkl.lz4"
        logger.info(f"Write correction file {outfile}")
        with lz4.frame.open(outfile, "wb") as f:
            pickle.dump({
                    'Z' : {
                        f"{outname}_minnlo_ratio" : corrh['ZToMuMu'],
                    },
                    "meta_data" : output_tools.metaInfoDict(args=args),
                }, f, protocol = pickle.HIGHEST_PROTOCOL)

    if 'WplusToMuNu' in corrh and "WminusToMuNu" in corrh:
        corrh['W'] = corrh['WplusToMuNu']+corrh['WminusToMuNu']
        outfile = f"{args.outpath}/{outname}CorrW.pkl.lz4"
        logger.info(f"Write correction file {outfile}")
        with lz4.frame.open(outfile, "wb") as f:
            pickle.dump({
                    'W' : {
                        f"{outname}_minnlo_ratio" : corrh['W'],
                    },
                    "meta_data" : output_tools.metaInfoDict(args=args),
                }, f, protocol = pickle.HIGHEST_PROTOCOL)

# logger.info(corrh['ZToMuMu'])