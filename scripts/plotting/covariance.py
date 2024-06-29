import mplhep as hep
import matplotlib.pyplot as plt
from matplotlib import cm

import uproot
import argparse
import os
import numpy as np
import pandas as pd
import hist
import pdb
from utilities import logging, common
from wremnants import plot_tools
from utilities.io_tools import input_tools, output_tools
from utilities.io_tools.combinetf_input import get_fitresult, load_covariance_pois, select_covariance_pois


hep.style.use(hep.style.ROOT)

parser = common.plot_parser()
parser.add_argument("infile", help="Combine fitresult .root or .hdf5 file")
parser.add_argument("--flow", action='store_true', help="Show overflow/underflow pois")
parser.add_argument("--plots", type=str, nargs="+", default=["covariance"], choices=["correlation", "covariance"], help="Define which plots to make")
parser.add_argument("--poiType", type=str, default="mu", choices=["nois", "mu", "pmaskedexp", "pmaskedexpnorm", "sumpois", "sumpoisnorm",], help="Parameter type to make the covariance matrix")
parser.add_argument("-c", "--channels", type=str, nargs="+", choices=["plus", "minus", "all"], default=["plus", "minus", "all"], help="Select channel to plot")

args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)

fitresult = get_fitresult(args.infile)
meta_info = input_tools.get_metadata(args.infile)

label_dict = {
    "qGen": "charge",
    "ptGen": r"$p_\mathrm{T}$",
    "absEtaGen": r"$|\eta|$",
    "ptVGen": r"$p_\mathrm{T}^{V}$",
    "absYVGen": r"$|Y^{V}|$",
}

selection_dict = {
    "Z": r"$\mathrm{Z}\rightarrow\mu\mu$",
    "W": r"$\mathrm{W}^{\pm}\rightarrow\mu\nu$",
    "W_qGen0": r"$\mathrm{W}^-\rightarrow\mu\nu$",
    "W_qGen1": r"$\mathrm{W}^+\rightarrow\mu\nu$",
}

def plot_matrix_poi(hist_cov, names, poi_type="mu", gen_axes=None, base_processes=[], selections={}, covariance=False, add_values_text=False,
    xlabel=None, ylabel=None, cms_decor="Preliminary", flow=False, clean_triangle=False, condition_number=False,
    ):
    if gen_axes is not None: 
        hist_cov = select_covariance_pois(hist_cov, names, gen_axes=gen_axes, base_processes=base_processes, selections=selections, flow=flow)

    if not covariance:
        std_devs = np.sqrt(np.diag(hist_cov.values()))

        # Calculate the correlation matrix
        hist_cov[...] = hist_cov.values() / np.outer(std_devs, std_devs)

    if xlabel is None:
        if gen_axes is not None:
            xlabel = "gen "+ "-".join([label_dict.get(a, a) for a in gen_axes if a not in selections.keys()])+" bin"
        elif poi_type == "nois":
            xlabel = "NOI"
        else:
            xlabel = "POI bin"
    if ylabel is None:
        ylabel = xlabel

    fig, ax = plot_tools.figure(hist_cov, xlabel=xlabel, ylabel=ylabel, cms_label=cms_decor, automatic_scale=False, width_scale=1.2)

    if covariance:
        cmin, cmax = None, None
    else:
        cmin, cmax = -1.0, 1.0

    # Create a mask to None out the upper triangle
    if clean_triangle:
        values = hist_cov.values().copy()
        mask = np.tril(np.ones_like(values, dtype=bool))
        values[~mask] = None
    else:
        values = hist_cov.values()

    hep.hist2dplot(values, ax=ax)#, cmap=cm.RdBu)#, clim=(cmin, cmax))#, labels=(xlabels,ylabels))

    # Loop through each cell in the matrix and add the value as text
    if add_values_text:
        for i in range(len(values)):
            for j in range(len(values[0])):
                plt.text(j + 0.5, i + 0.5, f'{values[i, j]:.5f}', ha='center', va='center', color='w')

    # calculate condition number
    if condition_number:
        cond = np.linalg.cond(hist_cov.values())
        logger.info(f"Condition number: {cond}")
        plt.text(0.2, 0.9, round(cond), horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

    key = base_process
    for k,v in selections.items():
        key += f"_{k}{v}"

    ax.text(1.0, 1.003, selection_dict.get(key, key), transform=ax.transAxes, fontsize=30,
            verticalalignment='bottom', horizontalalignment="right")

    outfile = "covariance" if covariance else "correlation"
    if base_process is not None:
        outfile += f"_{key}"
    if gen_axes is not None:
        outfile += "_" + "_".join([a for a in gen_axes if a not in selections.keys()])
    outfile += (f"_{args.postfix}_" if args.postfix else "_") + poi_type.split("_")[-1].replace("channel","")

    plot_tools.save_pdf_and_png(outdir, outfile)
    plot_tools.write_index_and_log(outdir, outfile, 
        yield_tables={"Values" : hist_cov.values()}, nround=10 if covariance else 2,
        analysis_meta_info={args.infile : meta_info},
        args=args,
    )

for plot_type in args.plots:
    
    covariance = False if plot_type == "correlation" else True

    hist_cov, names = load_covariance_pois(fitresult, args.poiType)

    for i, base_process in enumerate(map(lambda x: "W" if x.split("/")[-1].startswith("mw") else "Z", meta_info["args"]["inputFile"])):
        logger.debug(f"Now at {i}, base process {base_process}")

        # nois
        if args.poiType == "nois":
            plot_matrix_poi(hist_cov, names, args.poiType, cms_decor=args.cmsDecor, add_values_text=True,
                covariance=covariance, base_processes=[base_process,], flow=args.flow)
            continue

        # TODO, get axis information from meta data
        if args.poiType.startswith("sum"):
            axes_combinations = (("qGen","absEtaGen"), ("qGen","ptGen"), )#("ptGen"), ("absEtaGen"))
        else:
            axes_combinations = (("qGen","ptGen","absEtaGen"),)

        # elif base_process == "Z":
        #     if args.poiType.startswith("sum"):
        #         axes_combinations = ("ptVGen", "absYVGen")
        #     else:
        #         axes_combinations = (("ptVGen", "absYVGen"),)

        for channel in args.channels:
            if channel == "all":
                selection = {}
            elif channel == "plus":
                selection = {"qGen":1}
            elif channel == "minus":
                selection = {"qGen":1}

            for axes in axes_combinations:
                if isinstance(axes, str):
                    axes = [axes]
                plot_matrix_poi(hist_cov, names, args.poiType, cms_decor=args.cmsDecor, clean_triangle = True, condition_number=covariance,
                    gen_axes=axes, selections=selection, base_processes=[base_process,], covariance=covariance, flow=args.flow)

if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
    output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)