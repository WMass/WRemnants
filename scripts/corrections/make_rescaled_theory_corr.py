import argparse
import os
import pickle
import re

import lz4.frame

from utilities import common
from wremnants import theory_corrections
from wums import boostHistHelpers as hh
from wums import logging, output_tools


def corr_name(corrf):
    if not corrf.endswith(".pkl.lz4"):
        raise ValueError(f"File {corrf} is not a lz4 compressed pickle file")

    match = re.match(r"(.*)Corr[Z|W]\.pkl\.lz4", os.path.basename(corrf))
    return match[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--ref-corr",
        type=str,
        help="Reference corr (usually a correction to another prediction+unc.)",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--rescale-corr",
        type=str,
        help="Correction to rescale to",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--new-corr-name",
        type=str,
        help="Name of the new correction",
        required=True,
    )
    parser.add_argument("--debug", action="store_true", help="Print debug output")
    parser.add_argument(
        "--noColorLogger",
        action="store_true",
        default=False,
        help="Do not use logging with colors",
    )
    parser.add_argument(
        "--outpath",
        type=str,
        default=f"{common.data_dir}/TheoryCorrections",
        help="Output path",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Apply spline-based smoothing to correction",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logger = logging.setup_logger(
        "make_rescaled_theory_corr", 4 if args.debug else 3, args.noColorLogger
    )

    ref = pickle.load(lz4.frame.open(args.ref_corr))
    rescale = pickle.load(lz4.frame.open(args.rescale_corr))

    proc = "Z" if "CorrZ" in args.ref_corr else "W"

    refcorr = ref[proc][corr_name(args.ref_corr) + "_minnlo_ratio"]

    ref_hist = ref[proc][corr_name(args.ref_corr) + "_hist"]
    rescale_hist = rescale[proc][corr_name(args.rescale_corr) + "_hist"]

    ratio = hh.divideHists(rescale_hist[{"vars": 0}], ref_hist[{"vars": 0}], flow=False)

    new_corr = hh.multiplyHists(refcorr, ratio, flow=False)
    if args.smooth:
        new_corr = theory_corrections.smooth_theory_corr(
            new_corr, ref[proc]["minnlo_ref_hist"], rescale_hist
        )

    output_dict = {
        args.new_corr_name + "_minnlo_ratio": new_corr,
        **ref[proc],
        **rescale[proc],
    }
    meta_dict = {
        "ref_file": ref["file_meta_data"],
        "rescale_file": rescale["file_meta_data"],
    }
    outfile = f"{args.outpath}/{args.new_corr_name}"

    output_tools.write_theory_corr_hist(
        outfile, proc, output_dict, common.base_dir, args, meta_dict
    )


if __name__ == "__main__":
    main()
