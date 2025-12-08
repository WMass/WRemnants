import argparse
import pickle

import hist
import lz4.frame
import numpy as np

from utilities import common
from utilities.io_tools import input_tools
from wremnants import theory_corrections
from wums import boostHistHelpers as hh
from wums import logging, output_tools

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--unfoldingFile", type=str, required=True)
parser.add_argument("-g", "--genFile", type=str, required=True)
parser.add_argument(
    "--outpath",
    type=str,
    default=f"{common.data_dir}/TheoryCorrections",
    help="Output path",
)
parser.add_argument("--debug", action="store_true", help="Print debug output")
parser.add_argument(
    "--norm",
    type=float,
    help="Force the data/MC correction to have this norm (e.g., to 1.0 or to match the SCETlib+Dyturbo observed one)",
)
parser.add_argument(
    "-p", "--postfix", type=str, default=None, help="Postfix for output file name"
)
parser.add_argument(
    "--proc",
    type=str,
    required=True,
    choices=[
        "z",
        "w",
    ],
    help="Process",
)
parser.add_argument(
    "--uncertainties",
    type=str,
    help="Take uncertainties from this correction file (e.g., SCETlib+DYTurbo)",
)
parser.add_argument(
    "--axes",
    nargs="*",
    choices=["absYVgen", "ptVgen"],
    type=str,
    default=[
        "ptVgen",
    ],
    help="Use only specified axes in hist",
)
args = parser.parse_args()

logger = logging.setup_logger("make_ptv_unfolding_corr", 4 if args.debug else 3)

genh = input_tools.read_and_scale(args.genFile, "ZmumuPostVFP", "nominal_gen")

unfolded_res = pickle.load(open(args.unfoldingFile, "rb"))
unfolded_datah = unfolded_res["results"]["pmaskedexp"]["chan_13TeV"]["Z"][
    "_".join(["hist", *[x.replace("gen", "Gen") for x in args.axes]])
]

axes = {
    "massVgen": hist.axis.Regular(1, 0, 13000, name="massVgen", flow=False),
    "absYVgen": hist.axis.Regular(
        1, 0, 5, name="absYVgen", underflow=False, overflow=True
    ),
    "ptVgen": None,
    "chargeVgen": hist.axis.Regular(
        *(1, -1, 1) if args.proc == "z" else (2, -2, 2), name="chargeVgen", flow=False
    ),
    "vars": hist.axis.Regular(1, 0, 1, name="vars"),
}

# Thanks ~Obama~ (David)
for ax in unfolded_datah.axes:
    gen_name = ax.name.replace("Gen", "gen")
    ax._ax.metadata["name"] = gen_name

datah, genh = (h.project(*args.axes) for h in (unfolded_datah, genh))

for ax_name in unfolded_datah.axes.name:
    datah, genh = hh.rebinHistsToCommon(
        [h.project(*args.axes) for h in [unfolded_datah, genh]], ax_name
    )
    # Use the gen axis because you need the overflow for the ratio
    axes[ax_name] = genh.axes[ax_name]

if args.norm:
    print("Previous", datah.sum().value / genh.sum().value)
    datah = hh.normalize(datah, scale=args.norm, flow=False)
    genh = hh.normalize(genh, scale=1.0, flow=False)
    print("Adjusted", datah.sum().value / genh.sum().value)

ratio = hh.divideHists(datah, genh, flow=False)
indices = tuple(slice(None) if ax in args.axes else None for ax in axes.keys())

corrh = hist.Hist(*axes.values())
# Transpose because the order is different...
corrh[...] = ratio.values(flow=True).T[indices]

if args.uncertainties is not None:
    refcorr = pickle.load(lz4.frame.open(args.uncertainties))

    # TODO: Should probably be customizable, it only works with SCETlib+DYturbo
    unc_corr = refcorr[args.proc.upper()]["scetlib_dyturbo_minnlo_ratio"]

    corrh_vals = corrh.values()

    for axc, axf in (("ptVgen", "qT"), ("absYVgen", "absY")):
        ax_course = corrh.axes[axc]
        ax_fine = unc_corr.axes[axf]

        idx = np.clip(
            np.digitize(ax_fine.centers, ax_course.edges) - 1, 0, ax_course.size - 1
        )

        corrh_vals = np.take(corrh_vals, idx, axis=list(corrh.axes.name).index(axc))

    corr_fact = corrh_vals[..., 0:1] / unc_corr[..., 0:1].values()

    corrh = hist.Hist(*unc_corr.axes)
    # Broadcast syst axis
    corrh[...] = (unc_corr.values().T * corr_fact.T).T

corrh = theory_corrections.set_corr_ratio_flow(corrh)

output_dict = {
    "MC_data_ratio": corrh,
    "data_hist": unfolded_datah,
    "gen_hist": genh,
}

meta_dict = {
    "unfolding": input_tools.get_metadata(args.unfoldingFile),
    "gen": input_tools.get_metadata(args.genFile),
}

fname = "data"
if "ptVgen" in args.axes:
    fname += "Ptll"
if "absYVgen" in args.axes:
    fname += "Yll"

if args.postfix is not None:
    fname += f"_{args.postfix}"

outfile = "/".join([args.outpath, f"{fname}Corr{args.proc.upper()}"])
output_tools.write_lz4_pkl_output(
    outfile, args.proc.upper(), output_dict, common.base_dir, args, meta_dict
)
logger.info(f"Average correction is {np.mean(corrh.values())}")
