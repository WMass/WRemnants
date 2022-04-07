from wremnants.datasets.datagroups import datagroups2016
from wremnants import histselections as sel
from wremnants import plot_tools
import matplotlib.pyplot as plt
import argparse
import os
import shutil
import logging
import pathlib

template_dir = f"{pathlib.Path(__file__).parent}/Template"

parser = argparse.ArgumentParser()
parser.add_argument("infile", help="Output file of the analysis stage, containing ND boost histograms")
parser.add_argument("--wlike", action='store_true', help="Make W like plots")
parser.add_argument("-n", "--baseName", type=str, help="Histogram name in the file (e.g., 'nominal')", default="nominal")
parser.add_argument("--addUncorrected", action='store_true', help="Add uncorrected (no N3LL) distribution to plots")
parser.add_argument("--hists", type=str, nargs='+', help="List of histograms to plot")
parser.add_argument("-c", "--channel", type=str, choices=["plus", "minus", "all"], default="all", help="Select channel to plot")
parser.add_argument("-p", "--outpath", type=str, default=os.path.expanduser("~/www/WMassAnalysis"), help="Base path for output")
parser.add_argument("-f", "--outfolder", type=str, default="test", help="Subfolder for output")
parser.add_argument("-r", "--rrange", type=float, nargs=2, default=[0.9, 1.1], help="y range for ratio plot")
parser.add_argument("--ymax", type=float, help="Max value for y axis (if not specified, range set automatically)")
args = parser.parse_args()

groups = datagroups2016(args.infile, wlike=args.wlike)
groups.loadHistsForDatagroups(args.baseName, syst="")

if args.addUncorrected:
	groups.loadHistsForDatagroups(args.baseName, syst="uncorr")
	groups.addUncorrectedProc(args.baseName, "uncorr", label=r"No N$^{3}$LL Corr.")

histInfo = groups.getDatagroups()

prednames = [x for x in histInfo.keys() if x not in ["Data", "uncorr"]]
select = {} if args.channel == "all" else {"select" : -1.j if args.channel == "minus" else 1.j}

outpath = "/".join([args.outpath, args.outfolder])
if not os.path.isdir(args.outpath):
	raise IOError(f"The path {args.outpath} doesn't not exist. You should create it (and possibly link it to your web area)")
	
if not os.path.isdir(outpath):
	logging.info(f"Creating folder outpath")
	os.makedirs(outpath)

xlabels = {
	"pt" : r"p$_{T}^{\ell}$ (GeV)",
	"eta" : r"\eta^{\ell}$",
	"unrolled" : r"(p$_{T}^{\ell}$, $\eta^{\ell}$) bin",
}

#scales = {
#	"pt" : 9e6 if not args.wlike else 2e6,
#	"eta" : 5e6 if not args.wlike else 2e5,
#	"unrolled" : 1.8e5 if not args.wlike else 1.5e4,
#}

for h in args.hists:
	action = sel.unrolledHist if "unrolled" in h else lambda x: x.project(h)
	unstacked = ["Data"]
	if args.addUncorrected:
		unstacked.insert(0, "uncorr")
	fig = plot_tools.makeStackPlotWithRatio(histInfo, prednames, label=args.baseName, ymax=args.ymax, action=action, unstacked=unstacked, 
			xlabel=xlabels[h], ylabel="Events/bin", rrange=args.rrange, select=select) 
	plt.savefig("/".join([outpath, f"{h}_{args.channel}.pdf"]), bbox_inches='tight')
	plt.savefig("/".join([outpath, f"{h}_{args.channel}.png"]), bbox_inches='tight')
	
indexname = "index.php"
if not os.path.isfile(f"{outpath}/{indexname}"):
	shutil.copyfile(f"{template_dir}/{indexname}", f"{outpath}/{indexname}")
