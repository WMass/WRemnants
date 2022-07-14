from wremnants import plot_tools,theory_tools
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
import pathlib
import os
import pickle
import lz4.frame
import logging
import shutil

template_dir = f"{pathlib.Path(__file__).parent}/Template"

parser = argparse.ArgumentParser()
parser.add_argument("infile", help="Output file of the analysis stage, containing ND boost histograms")
parser.add_argument("--pdfs", type=str, nargs='+', help="List of histograms to plot", choices=theory_tools.pdfMapExtended.keys(), required=True)
parser.add_argument("-c", "--channel", type=str, choices=["plus", "minus", "all"], default="all", help="Select channel to plot")
parser.add_argument("-p", "--outpath", type=str, default=os.path.expanduser("~/www/WMassAnalysis"), help="Base path for output")
parser.add_argument("-f", "--outfolder", type=str, default="test", help="Subfolder for output")
parser.add_argument("-r", "--rrange", type=float, nargs=2, default=[0.9, 1.1], help="y range for ratio plot")
parser.add_argument("-d", "--dataset", type=str, help="Dataset to plot", required=True)
parser.add_argument("--obs", type=str, choices=["ptVgen", "absYVgen"], help="Observable to plot", required=True)
parser.add_argument("--together", action='store_true', help="y range for ratio plot")
parser.add_argument("--ymax", type=float, help="Max value for y axis (if not specified, range set automatically)")
args = parser.parse_args()

with lz4.frame.open(args.infile) as f:
    out = pickle.load(f)

if args.dataset not in out:
    raise ValueError(f"Dataset {args.dataset} not found in output of file {args.infile}")

hist_dict = out[args.dataset]["output"]

axis_label = "tensor_axis_0"

for pdf in args.pdfs:
    if pdf not in theory_tools.pdfMapExtended:
        raise ValueError(f"pdf {pdf} is not a valid hist (not defined in theory_tools.pdfMapExtended)")
    pdfName = theory_tools.pdfMapExtended[pdf]["name"]
    if pdfName not in hist_dict:
        raise ValueError(f"pdf {pdfName} was not found in file {args.infile}")

pdfNames = [theory_tools.pdfMapExtended[pdf]["name"] for pdf in args.pdfs]
pdfHists = [hist_dict[pdfName] for pdfName in pdfNames]
uncType = [theory_tools.pdfMapExtended[pdf]["combine"] for pdf in args.pdfs]
uncHists = [(h[{axis_label : 0}], *theory_tools.hessianPdfUnc(h, axis_label, unc)) for h,unc in zip(pdfHists, uncType)]
names = [(pdfName+" $\pm1\sigma$", "", "") for pdfName in pdfNames]
cmap = cm.get_cmap("tab10")
colors = [[cmap(i)]*3 for i in range(len(args.pdfs))]

xlabels = {
    "ptVgen" : r"p$_{T}^{%s}$ (GeV)" % "W" if "W" in args.dataset else "Z", 
    "absYVgen" : r"$|\mathrm{y}^{%s}|$"  % "W" if "W" in args.dataset else "Z",
}

# TODO
#if args.together:

outdir = plot_tools.make_plot_dir(args.outpath, args.outfolder)

for name,color,labels,hists in zip(args.pdfs,colors, names, uncHists):
    # This is the reference
    hists1D = [x.project(args.obs) for x in uncHists[0]]
    hists1D.extend([x.project(args.obs) for x in hists])
    plot_cols = [*colors[0], *color]
    plot_labels = [*names[0], *labels]
    fig = plot_tools.makePlotWithRatioToRef(hists1D, colors=plot_cols, labels=plot_labels, alpha=0.7,
            rrange=args.rrange, ylabel="$\sigma$/bin", xlabel=xlabels[args.obs], rlabel=f"x/{args.pdfs[0].upper()}", binwnorm=1.0, nlegcols=1)
    plot_tools.save_pdf_and_png(outdir, f"{name}Hist_{args.obs}_{args.channel}")
    
plot_tools.write_html_index(outdir)
