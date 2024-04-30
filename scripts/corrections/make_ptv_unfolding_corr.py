from utilities import boostHistHelpers as hh, common, logging
from utilities.io_tools import input_tools, output_tools
from wremnants import theory_corrections
import numpy as np
import hist
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--unfoldingFile", type=str, required=True)
parser.add_argument("-g", "--genFile", type=str, required=True)
parser.add_argument("--outpath", type=str, default=f"{common.data_dir}/TheoryCorrections", help="Output path")
parser.add_argument("--debug", action='store_true', help="Print debug output")
parser.add_argument("-p", "--postfix", type=str, default=None, help="Postfix for output file name")
parser.add_argument("--proc", type=str, required=True, choices=["z", "w", ], help="Process")
parser.add_argument("--axes", nargs="*", choices=["absYVgen", "ptVgen"], type=str, default=["ptVgen",], help="Use only specified axes in hist")
args = parser.parse_args()

logger = logging.setup_logger("make_ptv_unfolding_corr", 4 if args.debug else 3)

genh = input_tools.read_and_scale(args.genFile, "ZmumuPostVFP", "nominal_gen")

unfolded_res = pickle.load(open(args.unfoldingFile, "rb"))
unfolded_datah = unfolded_res["results"]["xsec"]["chan_13TeV"]["Z"]["_".join(["hist", *[x.replace("gen", "Gen") for x in args.axes]])]

axes = {"massVgen" : hist.axis.Regular(1, 0, 13000, name="massVgen", flow=False), 
        "absYVgen" : hist.axis.Regular(1, 0, 10, name="absYVgen", underflow=False, overflow=True), 
        "ptVgen" : None,
        "chargeVgen" : hist.axis.Regular(*(1, -1, 1) if args.proc == 'z' else (2, -2, 2), name="chargeVgen", flow=False),
        "vars" : hist.axis.Regular(1, 0, 1, name="vars")
}

# Thanks ~Obama~ (David)
for ax in unfolded_datah.axes:
    gen_name = ax.name.replace("Gen", "gen")
    ax._ax.metadata["name"] = gen_name

datah, genh = (h.project(*args.axes) for h in (unfolded_datah, genh))

for ax_name in unfolded_datah.axes.name:
    datah, genh = hh.rebinHistsToCommon([h.project(*args.axes) for h in [unfolded_datah, genh]], ax_name)
    # Use the gen axis because you need the overflow for the ratio
    axes[ax_name] = genh.axes[ax_name]

ratio = hh.divideHists(datah, genh, flow=False)
indices = tuple(slice(None) if ax in args.axes else None for ax in axes.keys())

corrh = hist.Hist(*axes.values())
# Transpose because the order is different...
corrh[...] = ratio.values(flow=True).T[indices]
corrh = theory_corrections.set_corr_ratio_flow(corrh)

output_dict = {
    "MC_data_ratio" : corrh,
    "data_hist" : unfolded_datah,
    "gen_hist" : genh,
}

meta_dict = {
    "unfolding" : input_tools.get_metadata(args.unfoldingFile),
    "gen" : input_tools.get_metadata(args.genFile),
}

fname = "data"
if "ptVgen" in args.axes:
    fname += "Ptll"
if "absYVgen" in args.axes:
    fname += "Yll"

outfile = "/".join([args.outpath, fname])
output_tools.write_theory_corr_hist(outfile, args.proc.upper(), output_dict, args, meta_dict)
logger.info(f"Average correction is {np.mean(corrh.values())}")
