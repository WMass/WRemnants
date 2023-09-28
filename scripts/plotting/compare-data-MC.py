from wremnants.datasets.datagroups import Datagroups
from wremnants import histselections as sel
from wremnants import plot_tools
import matplotlib.pyplot as plt
import argparse
import os
import logging

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--study", type = str, help = "the name of study for the combine files")
parser.add_argument(
    "-o", "--outdir", type = str,
    default = 'pdf_unc_RECO/wmass-pulls-no-systs/nominal_vs_pseudodata', 
    help = "The folder to put the output plots under the root wmass plot folder on eos"
)
parser.add_argument(
    "--pdf", type=str, nargs='+', 
    default = ['nnpdf31', 'ct18', 'mmht'],
    help="set of PDFs to compare"
)
#parser.add_argument("--hists", type=str, nargs='+', help="List of histograms to plot")
parser.add_argument(
    "-c", "--channels", type=str, 
    choices=["plus", "minus", "all"], default="all", 
    help="Select channel to plot"
)
parser.add_argument(
    "-x", "--observable", type=str, 
    choices=["pt", "eta", "unrolled"], default="unrolled", 
    help="Select observable to plot"
)
#parser.add_argument("-p", "--outpath", type=str, default=os.path.expanduser("~/www/WMassAnalysis"), help="Base path for output")
#parser.add_argument("-f", "--outfolder", type=str, default="test", help="Subfolder for output")
parser.add_argument("-r", "--rrange", type=float, nargs=2, default=[0.9, 1.1], help="y range for ratio plot")

args = parser.parse_args()

eos_plot_root_dir = os.environ['eos_plot_dir']
eos_wmass_plot_root_dir = eos_plot_root_dir + 'wmass/'
out_path = eos_wmass_plot_root_dir + args.outdir
combine_files_root_dir = os.environ['combine_files_dir']

if args.channels == "all": chns = ['plus', 'minus']
else: chns = args.channels.split(",")
if args.observable == "unrolled": action = lambda x: sel.unrolledHist(x)
else: action = lambda x: x.project(args.observable)

for chn in chns:
    for pdf in args.pdf:
        for pseudo in filter(lambda x: x != pdf, args.pdf):
            combine_file = Datagroups(
                f"{combine_files_root_dir}/{args.study}/{pdf}/{pseudo}CEN/WMassCombineInput.root", 
                combine = True,
                pseudodata_pdfset = pseudo
            )
            combine_file.loadHistsForDatagroups(
                "x", syst = "nominal", channel = chn, excluded_procs = "Data"
            )
            hist_info = combine_file.getDatagroups(excluded_procs = "Data")
            stacked_procs = list(filter(
                lambda x: x not in [f"pdf{pseudo.upper()}_sum", "Data"], hist_info.keys()
            ))

            fig = plot_tools.makeStackPlotWithRatio(
                hist_info, stacked_procs, unstacked = f"pdf{pseudo.upper()}_sum",
                action = action, 
                nlegcols = 2, grid = True, rrange=args.rrange,
                plot_title = f"{pdf} vs. p. data {pseudo}"
            )
            fig.savefig(f"{out_path}/{pdf}-{pseudo}CEN-{chn}-{args.observable}.png")
            fig.savefig(f"{out_path}/{pdf}-{pseudo}CEN-{chn}-{args.observable}.pdf")
