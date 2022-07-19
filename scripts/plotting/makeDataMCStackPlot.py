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

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("infile", help="Output file of the analysis stage, containing ND boost histograms")
parser.add_argument("--wlike", action='store_true', help="Make W like plots")
parser.add_argument("--ratio_to_data", action='store_true', help="Use data as denominator in ratio")
parser.add_argument("-n", "--baseName", type=str, help="Histogram name in the file (e.g., 'nominal')", default="nominal")
parser.add_argument("--hists", type=str, nargs='+', required=True, help="List of histograms to plot")
parser.add_argument("-c", "--channel", type=str, choices=["plus", "minus", "all"], default="all", help="Select channel to plot")
parser.add_argument("-p", "--outpath", type=str, default=os.path.expanduser("~/www/WMassAnalysis"), help="Base path for output")
parser.add_argument("-f", "--outfolder", type=str, default="test", help="Subfolder for output")
parser.add_argument("-r", "--rrange", type=float, nargs=2, default=[0.9, 1.1], help="y range for ratio plot")
parser.add_argument("--ymax", type=float, help="Max value for y axis (if not specified, range set automatically)")
parser.add_argument("--xrange", type=float, nargs=2, help="min and max for x axis")
parser.add_argument("-a", "--name_append", type=str, help="Name to append to file name")

subparsers = parser.add_subparsers()
variation = subparsers.add_parser("variation", help="Arguments for adding variation hists")
variation.add_argument("--varName", type=str, required=True, help="Name of variation hist")
variation.add_argument("--varLabel", type=str, nargs='+', required=True, help="Label(s) of variation hist for plotting")
variation.add_argument("--selectAxis", type=str, help="If you need to select a variation axis")
variation.add_argument("--selectEntries", type=int, help="entries to read from the selected axis")

args = parser.parse_args()

addVariation = hasattr(args, "varName") and args.varName

if addVariation and (args.selectAxis or args.selectEntries):
    if not (args.selectAxis and args.selectEntries):
        raise ValueError("Must --selectAxis and --selectEntires together")
    if len(args.varlabel) != 1 and (len(args.varLabel) != len(args.selectAxis) or len(args.varLabel) != len(args.selectEntries)):
        raise ValueError("Must specify the same number of args for --selectAxis, --selectEntires, and --varLabel")

outdir = plot_tools.make_plot_dir(args.outpath, args.outfolder)

groups = datagroups2016(args.infile, wlike=args.wlike)
if args.baseName != "nominal":
    groups.setNominalName(args.baseName.rsplit("_", 1)[0])
groups.loadHistsForDatagroups(args.baseName, syst="")

exclude = ["Data"]

if addVariation:
    groups.loadHistsForDatagroups(args.baseName, syst=args.varName)
    groups.addSummedProc(args.baseName, name=args.varName, label=args.varLabel[0])
    if not args.selectAxis:
        exclude.append(args.varName)
    else:
        for label,ax,entry in zip(args.varLabel, args.selectAxis, args.selectEntries):
            select = lambda x: x[{ax : entry}]
            name = args.varName+str(entry)
            groups.copyWithAction(label, name=name, refproc=args.varName, 
                label=label, color='green')
            exclude.append(name)

histInfo = groups.getDatagroups()

prednames = [x for x in histInfo.keys() if x not in exclude]
select = {} if args.channel == "all" else {"select" : -1.j if args.channel == "minus" else 1.j}

outpath = "/".join([args.outpath, args.outfolder])
if not os.path.isdir(args.outpath):
    raise IOError(f"The path {args.outpath} doesn't not exist. You should create it (and possibly link it to your web area)")
    
if not os.path.isdir(outpath):
    logging.info(f"Creating folder outpath")
    os.makedirs(outpath)

xlabels = {
    "pt" : r"p$_{T}^{\ell}$ (GeV)",
    "eta" : r"$\eta^{\ell}$",
    "unrolled" : r"(p$_{T}^{\ell}$, $\eta^{\ell}$) bin",
    "ptll" : r"p$_{\mathrm{T}}^{\ell\ell}$",
    "yll" : r"y$^{\ell\ell}$",
    "mll" : r"m$_{\ell\ell}$",
    "costhetastarll" : r"$\cos{\phi^{\star}_{\ell\ell}}$",
    "phistarll" : r"$\phi^{\star}_{\ell\ell}$",
}

for h in args.hists:
    action = sel.unrolledHist if "unrolled" in h else lambda x: x.project(h)
    fig = plot_tools.makeStackPlotWithRatio(histInfo, prednames, histName=args.baseName, ymax=args.ymax, action=action, unstacked=exclude, 
            xlabel=xlabels[h], ylabel="Events/bin", rrange=args.rrange, select=select, binwnorm=1.0,
            ratio_to_data=args.ratio_to_data, rlabel="Pred./Data" if args.ratio_to_data else "Data/Pred.",
            xlim=args.xrange) 
    outfile = f"{h}_{args.channel}"+ (f"_{args.name_append}" if args.name_append else "")
    plot_tools.save_pdf_and_png(outdir, outfile)
    plot_tools.write_index_and_log(outdir, outfile)
