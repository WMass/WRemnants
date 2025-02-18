import argparse
import os

import hist
import numpy as np
import pandas as pd

from utilities import common
from wums import boostHistHelpers as hh
from wums import logging, output_tools

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    nargs="+",
    type=str,
    default=[f"{common.data_dir}/EWCorrections/dsig_dmll_dpTll_Zsel_ful.csv"],
    help="Input csv file with virtual corrections",
)
parser.add_argument("--debug", action="store_true", help="Print debug output")
parser.add_argument(
    "--outpath",
    type=str,
    default=f"{common.data_dir}/TheoryCorrections",
    help="Output path",
)
parser.add_argument(
    "-p", "--postfix", type=str, help="Postfix for plots and correction files"
)
parser.add_argument("--outname", type=str, default="", help="Output file name")

args = parser.parse_args()

logger = logging.setup_logger("make_theory_corr_virtual", 4 if args.debug else 3)

charge_dict = {"ZToMuMu": 0, "WplusToMuNu": 1, "WminusToMuNu": 0}

# translate to preFSR column names
preFSR_dict = {"pTll": "ptVgen", "mll": "massVgen", "pTl": "ptgen", "etal": "etagen"}

# axes where overflow/underflow should be added
overflow_axes = ["mll", "pTll", "pTl", "etal"]
underflow_axes = ["pTl", "etal"]


def ew_df_to_axis(df, name):
    axis_name = preFSR_dict[name]
    edges = np.array(sorted(set(np.append(df[f"{name}_min"], df[f"{name}_max"]))))
    opts = dict(
        name=axis_name, overflow=name in overflow_axes, underflow=name in underflow_axes
    )
    if len(edges) == max(df[f"{name}_max"]) + 1 - min(df[f"{name}_min"]) and all(
        edges == np.arange(min(edges), max(edges) + 1)
    ):
        axis = hist.axis.Regular(
            len(edges) - 1, int(min(edges)), int(max(edges)), **opts
        )
    else:
        axis = hist.axis.Variable(edges, **opts)
    return axis


def read_ew_corrections_from_csv(filename, proc):
    if not os.path.exists(filename):
        logger.warning(f"File {filename} not found")
        return False

    df = pd.read_csv(filename)

    # obtain axes names from column names that appear with min and max in the form "AXIS_min" "AXIS_max"
    ew_axes_names = [k for k in df.keys() if k.endswith("min")]
    ew_axes_names = [
        a.replace("_min", "")
        for a in ew_axes_names
        if a.replace("min", "max") in df.keys()
    ]
    # get the axes with the binning
    ew_axes = [ew_df_to_axis(df, a) for a in ew_axes_names]

    hratio = hist.Hist(*ew_axes, storage=hist.storage.Double())

    # charge axis
    if proc[0] == "W":
        axis_charge = hist.axis.Regular(
            2, -2.0, 2.0, underflow=False, overflow=False, name="charge"
        )
    elif proc[0] == "Z":
        axis_charge = hist.axis.Regular(
            1, -1.0, 1.0, underflow=False, overflow=False, name="charge"
        )

    charge_idx = charge_dict[proc]

    # syst axis
    axis_syst = hist.axis.Regular(
        3, 0, 3, underflow=False, overflow=False, name="systIdx"
    )

    # fill final histogram
    hsyst = hist.Hist(
        *hratio.axes, axis_charge, axis_syst, storage=hratio.storage_type()
    )
    hsyst.values(flow=True)[...] = np.ones(
        hsyst.axes.extent
    )  # set all bins including flow to 1
    hsyst.values(flow=False)[..., charge_idx, 0] = df["WEAK1/NOM"].values.reshape(
        hratio.axes.size
    )
    hsyst.values(flow=False)[..., charge_idx, 1] = df["WEAK2/NOM"].values.reshape(
        hratio.axes.size
    )
    hsyst.values(flow=False)[..., charge_idx, 2] = df["WEAK3/NOM"].values.reshape(
        hratio.axes.size
    )

    # set underflow and overflow bins to closest bin values
    hsyst = hh.set_flow(hsyst, val="nearest")

    return hsyst


corrh = {}

corrh["ZToMuMu"] = read_ew_corrections_from_csv(args.input[0], "ZToMuMu")

outfile = f"{args.outpath}/{args.outname}"
output_tools.write_theory_corr_hist(
    outfile, "Z", {f"{args.outname}_minnlo_ratio": corrh["ZToMuMu"]}, args
)
