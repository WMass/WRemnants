import os
import pickle

import hist
import lz4.frame
from matplotlib import cm

from utilities import parsing
from utilities.io_tools import input_tools
from wremnants import plot_tools, theory_tools
from wums import boostHistHelpers as hh
from wums import logging

s = hist.tag.Slicer()
# Map hist_name argument to the actual hist and it's axis stored in files
lookup = {
    "minnlo": {
        "dirs": {
            "z": ("ZmumuPostVFP", "output"),
            "wp": ("WplusmunuPostVFP", "output"),
            "wm": ("WminusmunuPostVFP", "output"),
        },
        "ptV": {
            "hist": "nominal_gen",
            "axis": "ptVgen",
            "action": lambda x: x[
                {"massVgen": s[0 : x.axes["massVgen"].size : hist.sum]}
            ],
        },
        "absYV": {
            "hist": "nominal_gen",
            "axis": "absYVgen",
            "action": lambda x: x[
                {
                    "massVgen": s[0 : x.axes["massVgen"].size : hist.sum],
                    "ptVgen": s[0 : 40.0j : hist.sum],
                }
            ],
        },
        "mV": {
            "hist": "nominal_gen",
            "axis": "massVgen",
            "action": lambda x: x[{"ptVgen": s[0 : 40.0j : hist.sum]}],
        },
        "sigma4_ptV": {
            "hist": "helicity_moments_scale",
            "axis": "ptVgen",
            "action": lambda x: theory_tools.scale_angular_moments(
                x[
                    {
                        "muRfact": 1.0j,
                        "muFfact": 1.0j,
                        "massVgen": s[0 : x.axes["massVgen"].size : hist.sum],
                    }
                ]
            )[{"helicity": 4.0j}],
        },
        "sigma4_absYV": {
            "hist": "helicity_moments_scale",
            "axis": "absYVgen",
            "action": lambda x: theory_tools.scale_angular_moments(
                x[
                    {
                        "muRfact": 1.0j,
                        "muFfact": 1.0j,
                        "massVgen": s[0 : x.axes["massVgen"].size : hist.sum],
                        "ptVgen": s[0 : 40.0j : hist.sum],
                    }
                ]
            )[{"helicity": 4.0j}],
        },
    },
    "scetlib": {
        "dirs": {"z": ("W"), "wp": ("W"), "wm": ("W")},
        "ptV": {
            "hist": "inclusive_pT_y_m_scetlib",
            "axis": "pt",
        },
        "absYV": {
            "hist": "inclusive_pT_y_m_scetlib",
            "axis": "absy",
            "action": lambda x: hh.makeAbsHist(x[{"pt": s[0 : 40.0j : hist.sum]}], "y"),
        },
        "sigma4_ptV": {
            "axis": "pt",
        },
        "sigma4_absYV": {
            "axis": "absy",
            "action": lambda x: hh.makeAbsHist(x[{"pt": s[0 : 40.0j : hist.sum]}], "y"),
        },
        "mV": {
            # "hist" : "inclusive_pT_y_m_scetlib",
            "axis": "mass",
        },
    },
    "matrix_radish": {
        "ptV": {
            "axis": "pt",
        },
        "absYV": {
            "axis": "absy",
            "action": lambda x: hh.makeAbsHist(x, "y"),
        },
    },
    "dyturbo": {
        "ptV": {
            "axis": "pt",
            "action": None,
        },
        "absYV": {
            "axis": "absy",
            "action": lambda x: hh.makeAbsHist(x[{"pt": s[0 : 40.0j : hist.sum]}], "y"),
        },
    },
}

parser = parsing.base_parser()
parser.add_argument(
    "-s",
    "--scetlib_files",
    default=[],
    type=str,
    nargs="+",
    help="SCETlib input files (.npz format)",
)
parser.add_argument(
    "-m",
    "--minnlo_files",
    default=[],
    type=str,
    nargs="+",
    help="minnlo input files (.pkl format)",
)
parser.add_argument(
    "-mr",
    "--matrix_radish_files",
    default=[],
    type=str,
    nargs="+",
    help="MATRIX+RadISH input files (.dat format)",
)
parser.add_argument(
    "-dt",
    "--dyturbo_files",
    default=[],
    type=str,
    nargs="+",
    help="DYTurbo input files (.txt format)",
)
parser.add_argument(
    "-n",
    "--hist_name",
    type=str,
    choices=lookup["minnlo"].keys(),
    help="Observable to plot",
    required=True,
)
parser.add_argument(
    "-p",
    "--proc",
    type=str,
    choices=lookup["minnlo"]["dirs"].keys(),
    help="Process",
    required=True,
)
parser.add_argument(
    "-r",
    "--rrange",
    type=float,
    nargs=2,
    default=[0.9, 1.1],
    help="y range for ratio plot",
)
parser.add_argument(
    "--ylim",
    type=float,
    nargs=2,
    help="Range for y axis (if not specified, range set automatically)",
)
parser.add_argument(
    "-o",
    "--outpath",
    type=str,
    default=os.path.expanduser("~/www/WMassAnalysis"),
    help="Base path for output",
)
parser.add_argument(
    "-f", "--outfolder", type=str, default="test", help="Subfolder for output"
)
parser.add_argument("-a", "--name_append", type=str, help="Name to append to file name")
parser.add_argument(
    "--ratio_ref",
    type=str,
    default="minnlo",
    choices=["minnlo", "scetlib", "matrix_radish"],
    help="Reference hist for ratio",
)
parser.add_argument(
    "--keep_full_range",
    action="store_true",
    help="Store the full range of all hists, even if it exceeds other hist ranges",
)
parser.add_argument(
    "--no-radish",
    action="store_true",
    help="matrix-radish file doesn't have resummation",
)
parser.add_argument("--logy", action="store_true", help="y axis log scale")
parser.add_argument("--logx", action="store_true", help="x axis log scale")
parser.add_argument("--xlim", type=float, nargs=2, help="range of the x axis")
args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

if not args.scetlib_files:
    lookup["minnlo"]["absYV"]["action"] = lambda x: x[
        {"massVgen": s[0 : x.axes["massVgen"].size : hist.sum]}
    ]

cmap = cm.get_cmap("tab10")
lookup["minnlo"]["colors"] = ["red"] + [
    cmap(i) for i in range(len(args.minnlo_files) - 1)
]
lookup["scetlib"]["colors"] = ["purple"] + [
    cmap(9 - i) for i in range(len(args.scetlib_files) - 1)
]
lookup["matrix_radish"]["colors"] = ["green"] + [
    cmap(i + len(args.minnlo_files)) for i in range(len(args.matrix_radish_files) - 1)
]
lookup["dyturbo"]["colors"] = ["blue"] + [
    cmap(i) for i in range(len(args.dyturbo_files))
]

xlabels = {
    "ptV": r"p$_{T}^{%s}$ (GeV)" % ("W" if "w" in args.proc else "Z"),
    "absYV": r"$|\mathrm{y}^{%s}|$" % ("W" if "w" in args.proc else "Z"),
    "mV": r"$m_{%s}$ (GeV)" % ("W" if "w" in args.proc else "Z"),
}
xlabels["sigma4_ptV"] = xlabels["ptV"]
xlabels["sigma4_absYV"] = xlabels["absYV"]

ylabels = {
    "sigma4_ptV": r"$\sigma_{4}$/bin",
    "sigma4_absYV": r"$\sigma_{4}$/bin",
    "a4_ptV": r"$\A_{4}$/bin",
    "a4_absYV": r"$\A_{4}$/bin",
}


def transform_and_project(histND, scale, axis_name, action=None):
    if action is not None:
        histND = action(histND)
    hist1D = histND.project(axis_name) * scale
    return hist1D


def read_pickle_hist(proc, file_name, lookup, hist_name):
    with lz4.frame.open(file_name) as f:
        output = pickle.load(f)

    # TODO: This code is so terrible, fix it!
    levels = {proc: []} if "dirs" not in lookup else lookup["dirs"]
    hist_out = output
    scale = 1.0
    for level in levels[proc]:
        if "dataset" in hist_out:
            scale *= hist_out["dataset"]["xsec"]
        if "weight_sum" in hist_out:
            scale = scale / hist_out["weight_sum"]
        hist_out = hist_out[level]

    if hist_name not in lookup:
        raise ValueError(f"Couldn't find hist {hist_name} in file {file_name}")

    hist_info = lookup[hist_name]
    histND = hist_out[hist_info["hist"]]
    action = hist_info["action"] if "action" in hist_info else None
    return transform_and_project(histND, scale, hist_info["axis"], action)


def read_scetlib_hist(proc, file_name, lookup, hist_name):
    charge = 0 if proc == "z" else (-1 if proc == "wp" else 1)
    ext = file_name.split(".")[-1]
    nonsing = file_name.replace(f".{ext}", f"_nons.{ext}")
    if "combined_nons" in nonsing:
        nonsing = nonsing.replace("combined_nons", "nons_combined")
    if not os.path.isfile(nonsing):
        logging.warning(
            "Didn't find the non-singular contribution. Will make comparisons without it"
        )
        nonsing = ""
    histND = input_tools.read_scetlib_hist(
        file_name, nonsing=nonsing, charge=charge, flip_y_sign="A4" in file_name
    )
    if "vars" in histND.axes.name:
        histND = histND[{"vars": 0}]

    hist_info = lookup[hist_name]
    action = hist_info["action"] if "action" in hist_info else None
    return transform_and_project(histND, 1.0, hist_info["axis"], action)


def read_matrix_radish_hist(proc, filename, lookup, hist_name):
    hist_info = lookup[hist_name]
    histND = input_tools.read_matrixRadish_hist(
        filename, hist_info["axis"].replace("abs", "")
    )
    if "vars" in histND.axes.name:
        histND = histND[{"vars": 0}]
    scale = 1.0 if "scale" not in hist_info else hist_info["scale"]

    action = hist_info["action"] if "action" in hist_info else None
    return transform_and_project(histND, scale, hist_info["axis"], action)


def read_dyturbo_hist(proc, filename, lookup, hist_name):
    info = lookup[hist_name]
    axes = ("y", "pt") if "2d" in filename else [info["axis"]]
    h = input_tools.read_dyturbo_hist(filename.split(":"), axes=axes)
    return transform_and_project(h, 1.0, info["axis"], info["action"])


def read_hists(proc, files, lookup, hist_name):
    hists = []
    meta_info = {}
    for fname in files:
        if ".pkl.lz4" in fname[-8:]:
            func = read_pickle_hist
            meta_info = pickle.load(lz4.frame.open(fname))["meta_info"]
        elif ".npz" in fname[-4:] or ".pkl" in fname[-4:]:
            func = read_scetlib_hist
            if ".pkl" in fname[-4:]:
                meta_info = pickle.load(open(fname, "rb"))["meta_data"]
        elif ".dat" in fname[-4:]:
            func = read_matrix_radish_hist
        elif ".txt" in fname[-4:]:
            func = read_dyturbo_hist
        else:
            raise ValueError(f"File {fname} is not a recognized type")
        hists.append(func(proc, fname, lookup, hist_name))
    return hists, meta_info


if not args.scetlib_files and not args.minnlo_files:
    raise ValueError("Must specify at least one filename")

outdir = plot_tools.make_plot_dir(args.outpath, args.outfolder)

all_hists = []
all_labels = []
all_colors = []

generators_info = [
    ("minnlo", "MiNNLO (NNLO+PS)"),
    ("scetlib", "SCETlib (N$^{3}$LL)"),
    (
        "matrix_radish",
        "MATRIX+RadISH (NNLO+N$^{3}$LL)" if not args.no_radish else "MATRIX (NNLO)",
    ),
    ("dyturbo", "DYTurbo (NNLO+N$^{3}$LL)"),
]

short_name = {
    "minnlo": "MiNNLO",
    "scetlib": "SCETlib",
    "matrix_radish": "MAT+Rad",
    "dyturbo": "DYTurbo",
}

generators_info.insert(
    0, generators_info.pop([x[0] for x in generators_info].index(args.ratio_ref))
)

all_meta_info = {}

for generator, label in generators_info:
    files = getattr(args, f"{generator}_files")
    if files:
        info = lookup[generator]
        if args.hist_name not in info:
            logger.warning(
                f"Failed to find hist {args.hist_name} for generator {generator}. Skipping"
            )
            continue
        hists, meta_info = read_hists(args.proc, files, info, args.hist_name)

        all_hists.extend(hists)
        all_colors.extend([info["colors"][c] for c in range(len(hists))])
        all_labels.append(label)
        if len(hists) > 1:
            if len(hists) == 2:
                if generator == "dyturbo":
                    all_labels.append("DYTurbo (NNLO)")
                if generator == "matrix_radish":
                    all_labels.append("MATRIX (NNLO)")
            else:
                all_labels.extend([f"{label} (alt {i+1})" for i in range(len(hists))])

        if meta_info:
            all_meta_info[generator] = meta_info


if len(all_hists) > 1:
    all_hists = hh.rebinHistsToCommon(
        all_hists, axis_idx=0, keep_full_range=args.keep_full_range
    )

ylabel = r"$\sigma$/bin" if args.hist_name not in ylabels else ylabels[args.hist_name]
fig = plot_tools.makePlotWithRatioToRef(
    all_hists,
    colors=all_colors,
    labels=all_labels,
    alpha=0.7,
    ylim=args.ylim,
    rrange=args.rrange,
    ylabel=ylabel,
    xlabel=xlabels[args.hist_name],
    rlabel=f"x/{short_name[args.ratio_ref]}",
    binwnorm=1.0,
    nlegcols=1,
    logy=args.logy,
    logx=args.logx,
    xlim=args.xlim,
)

outname = f"TheoryCompHist_{args.proc}_{args.hist_name}" + (
    "_" + args.name_append if args.name_append else ""
)
plot_tools.save_pdf_and_png(outdir, outname)
plot_tools.write_index_and_log(outdir, outname, analysis_meta_info=all_meta_info)
