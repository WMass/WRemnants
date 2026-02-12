import os
import pickle
import re

import lz4.frame

from utilities import common, parsing
from wums import logging, output_tools


def corr_name(corrf):
    if not corrf.endswith(".pkl.lz4"):
        raise ValueError(f"File {corrf} is not a lz4 compressed pickle file")

    match = re.match(r"(.*)Corr[Z|W]\.pkl\.lz4", os.path.basename(corrf))
    return match[1]


def parse_args():
    parser = parsing.base_parser()
    parser.add_argument(
        "-c",
        "--refCorr",
        type=str,
        help="Reference corr (usually a correction to another prediction+unc.)",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--rescaleCorr",
        type=str,
        help="Correction to rescale to",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--newCorrName",
        type=str,
        help="Name of the new correction",
        required=True,
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
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    ref = pickle.load(lz4.frame.open(args.refCorr))
    rescale = pickle.load(lz4.frame.open(args.rescaleCorr))

    proc = "Z" if "CorrZ" in args.refCorr else "W"

    refcorr = ref[proc][corr_name(args.refCorr) + "_minnlo_ratio"]

    rescale_corr_name = (
        f"{corr_name(args.rescaleCorr)}_minnlo_ratio"
        if "dataPtll" not in args.rescaleCorr
        else "MC_data_ratio"
    )
    rescale_corr = rescale[proc][rescale_corr_name]

    if refcorr.shape[:-1] != rescale_corr.shape[:-1]:
        raise ValueError(
            f"Histograms {refcorr.shape} and {rescale_corr.shape} have different shapes"
        )

    new_corr = refcorr.copy()
    central_val_corr = rescale_corr[..., 0:1].view() / refcorr[..., 0:1].view()
    # Broadcast syst axis
    new_corr[...] = (refcorr.view().T * central_val_corr.T).T

    new_name = args.newCorrName + "_minnlo_ratio"
    if "dataPtll" in args.newCorrName:
        new_name = "MC_data_ratio"
        # Avoid overwriting
        rescale[proc]["MC_data_ratio_old"] = rescale[proc]["MC_data_ratio"]

    output_dict = {
        **ref[proc],
        **rescale[proc],
        new_name: new_corr,
    }
    meta_dict = {
        "ref_file": ref["file_meta_data"],
        "rescale_file": rescale["file_meta_data"],
    }
    outfile = f"{args.outpath}/{args.newCorrName}Corr{proc}.pkl.lz4"

    output_tools.write_lz4_pkl_output(
        outfile, proc, output_dict, common.base_dir, args, meta_dict
    )


if __name__ == "__main__":
    main()
