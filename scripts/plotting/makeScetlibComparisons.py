import uproot
import hist
import argparse
from wremnants import scetlib_corrections,plot_tools,boostHistHelpers as hh
import pickle
import lz4.frame
import os
from matplotlib import cm
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

# Map hist_name argument to the actual hist and it's axis stored in files
lookup = {
    "minnlo" : {
        "dirs" : {
            "z" : ("ZmumuPostVFP", "output"),
            "wp" : ("WplusmunuPostVFP", "output"),
            "wm" : ("WminusmunuPostVFP", "output"),
        },
        "ptV" : {
            "hist" : "nominal_gen",
            "axis" : "ptVgen"
        },
        "absYV" : {
            "hist" : "nominal_gen",
            "axis" : "absYVgen"
        },
    },
    "scetlib" : {
        "dirs" : {"z" : ("W"), "wp" : ("W"), "wm" : ("W")},
        "ptV" : {
            "hist" : "inclusive_pT_y_m_scetlib",
            "axis" : "pt",
        },
        "absYV" : {
            "hist" : "inclusive_pT_y_m_scetlib",
            "axis" : "y",
            "action" : lambda x: hh.makeAbsHist(x, "y"),
        },
    },
    "matrix_radish" : {
        "dirs" : {
            "z" : ["DYm50_matrix__radish"],
            #"z" : [],
            "wp" :[],
            "wm" :["wmmunu_matrix_radish"],
        },
        "ptV" : {
            "z": {
                "hist" : "ptZ_lhe_mm",
                #"hist" : "RadISH_observable_binning1_N3LL",
                "axis" : "xaxis",
                #"scale" : 1/1000.
            },
            "wm" : {"hist" : "ptW_prefsr_mn",
                    "axis" : "xaxis",
                    "scale" : 1/1000.},
        },
    }
}

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--scetlib_files", default=[], type=str, nargs='+', help="SCETlib input files (.pkl format)")
parser.add_argument("-m", "--minnlo_files", default=[], type=str, nargs='+', help="minnlo input files (.pkl format)")
parser.add_argument("-mr", "--matrix_radish_files", default=[], type=str, nargs='+', help="MATRIX+RadISH input files (.pkl format)")
parser.add_argument("-n", "--hist_name", type=str, choices=lookup["minnlo"].keys(), help="minnlo input files (.pkl format)", required=True)
parser.add_argument("-p", "--proc", type=str, choices=lookup["minnlo"]["dirs"].keys(), help="Process", required=True)
parser.add_argument("-r", "--rrange", type=float, nargs=2, default=[0.9, 1.1], help="y range for ratio plot")
parser.add_argument("--ymax", type=float, help="Max value for y axis (if not specified, range set automatically)")
parser.add_argument("-o", "--outpath", type=str, default=os.path.expanduser("~/www/WMassAnalysis"), help="Base path for output")
parser.add_argument("-f", "--outfolder", type=str, default="test", help="Subfolder for output")
parser.add_argument("-a", "--name_append", type=str, help="Name to append to file name")
parser.add_argument("--ratio_ref", type=str, default="minnlo", choices=["minnlo", "scetlib", "matrix_radish"], help="Reference hist for ratio")
parser.add_argument("--keep_full_range", action='store_true', help="Store the full range of all hists, even if it exceeds other hist ranges")
args = parser.parse_args()

cmap = cm.get_cmap("tab10")
lookup["minnlo"]["colors"] = ["red"]+[cmap(i) for i in range(len(args.minnlo_files)-1)]
lookup["scetlib"]["colors"] = ["purple"]+[cmap(9-i) for i in range(len(args.scetlib_files)-1)]
lookup["matrix_radish"]["colors"] = ["green"]+[cmap(i+len(args.minnlo_files)) for i in range(len(args.scetlib_files)-1)]

xlabels = {
    "ptV" : r"p$_{T}^{%s}$ (GeV)" % ("W" if "w" in args.proc else "Z"), 
    "absYV" : r"$|\mathrm{y}^{%s}|$"  % ("W" if "w" in args.proc else "Z"),
}

def read_pickle_hist(proc, file_name, lookup, hist_name):
    with lz4.frame.open(file_name) as f:
        output = pickle.load(f)

    #TODO: This code is so terrible, fix it!
    levels = {proc : []} if "dirs" not in lookup else lookup["dirs"]
    hist_out = output
    scale = 1.
    for level in levels[proc]:
        if "dataset" in hist_out:
            scale *= hist_out["dataset"]["xsec"]
        if "weight_sum" in hist_out:
            scale = scale/hist_out["weight_sum"]
        hist_out = hist_out[level]

    if hist_name not in lookup:
        raise ValueError(f"Couldn't find hist {hist_name} in file {file_name}")

    hist_info = lookup[hist_name]
    histND = hist_out[hist_info["hist"]]
    hist1D = histND.project(hist_info["axis"])*scale
    return hist1D

def read_scetlib_hist(proc, file_name, lookup, hist_name):
    charge = 0 if proc == "z" else (-1 if proc == "wp" else 1)
    add_nonsing = os.path.isfile(file_name.replace(".npz", "_nons.npz"))
    if not add_nonsing:
        logging.warning("Didn't find the non-singular contribution. Will make comparisons without it")
    histND = scetlib_corrections.read_scetlib_hist(file_name, add_nonsing=add_nonsing, charge=charge)
    hist1D = histND.project(lookup[hist_name]["axis"])
    return hist1D

def read_matrix_radish_hist(proc, filename, lookup, hist_name):
    f = uproot.open(filename)
    hist_info = lookup[hist_name][proc]
    h = f["/".join(lookup["dirs"][proc]+[hist_info["hist"]])].to_hist()
    if "scale" in hist_info:
        h = h*hist_info["scale"]
    return h.project(hist_info["axis"])

def read_hists(proc, files, lookup, hist_name):
    hists = []
    for fname in files:
        if ".pkl.lz4" in fname[-8:]:
            func = read_pickle_hist
        if ".npz" in fname[-4:]:
            func = read_scetlib_hist
        if ".root" in fname[-5:]:
            func = read_matrix_radish_hist
        hists.append(func(proc, fname, lookup, hist_name))
    return hists

if not args.scetlib_files and not args.minnlo_files:
    raise ValueError("Must specify at least one filename")

outdir = plot_tools.make_plot_dir(args.outpath, args.outfolder)

all_hists = []
all_labels = []
all_colors = []

generators_info = [
	("minnlo", "MiNNLO (NNLO+PS)"), 
	("scetlib", "SCETlib (N$^{3}$LL)"),
	("matrix_radish", "MATRIX+RadISH (NNLO+N$^{3}$LL)"),
]

generators_info.insert(0, generators_info.pop([x[0] for x in generators_info].index(args.ratio_ref)))

for generator, label in generators_info:
    files = getattr(args, f"{generator}_files")
    if files:
        info = lookup[generator]
        hists = read_hists(args.proc, files, info, args.hist_name)
        all_hists.extend(hists)
        all_colors.extend([info["colors"][c] for c in range(len(hists))])
        all_labels.append(label)
        if len(hists) > 1:
            all_labels.extend([f"{label} (alt {i})" for i in range(1, len(hists[1:]+1))])

all_hists = hh.rebinHistsToCommon(all_hists, axis_idx=0, keep_full_range=args.keep_full_range)

fig = plot_tools.makePlotWithRatioToRef(all_hists, colors=all_colors, labels=all_labels, alpha=0.7, ymax=args.ymax,
        rrange=args.rrange, ylabel="$\sigma$/bin", xlabel=xlabels[args.hist_name], rlabel=f"x/MiNNLO", binwnorm=1.0, nlegcols=1)

outname = f"TheoryCompHist_{args.proc}_{args.hist_name}" + ("_"+args.name_append if args.name_append else "")
plot_tools.save_pdf_and_png(outdir, outname)
plot_tools.write_index_and_log(outdir, outname)
