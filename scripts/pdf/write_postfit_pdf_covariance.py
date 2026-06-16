import argparse

import h5py
import numpy as np

from wremnants.postprocessing.postfit_pdf_helper import RabbitPostfitPdfHelper
from wums import logging, output_tools

parser = argparse.ArgumentParser(
    description="Write the postfit PDF covariance matrix and pulls to a simple HDF5 file."
)
parser.add_argument(
    "-f",
    "--fitresult",
    type=str,
    required=True,
    help="Path to the rabbit fit result file.",
)
parser.add_argument(
    "-o", "--output", type=str, required=True, help="Output HDF5 file path."
)
parser.add_argument(
    "--pseudoData", type=str, default=None, help="Pseudo-data label to use."
)
parser.add_argument(
    "-v", "--verbose", choices=[0, 1, 2, 3, 4], default=3, help="Set verbosity level."
)
parser.add_argument(
    "--noColorLogger", action="store_true", help="Disable colored logging output."
)
args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

pdf_helper = RabbitPostfitPdfHelper(args.fitresult, pseudoData=args.pseudoData)

cov_matrix, _ = pdf_helper.get_pdf_covariance()
labels = pdf_helper.pdf_nuisances
pulls = pdf_helper.pdf_pulls

logger.info(
    f"Writing covariance matrix ({cov_matrix.shape}), {len(labels)} labels, and pulls to {args.output}"
)

meta_info = output_tools.make_meta_info_dict(args=args)

with h5py.File(args.output, "w") as f:
    f.create_dataset("covariance", data=cov_matrix)
    f.create_dataset("pulls", data=pulls)
    f.create_dataset(
        "labels", data=np.array(labels, dtype=h5py.special_dtype(vlen=str))
    )
    f.attrs["pdf_name"] = pdf_helper.pdf_name
    f.attrs["pdf_scale"] = pdf_helper.pdf_scale
    f.attrs["pdf_symm"] = pdf_helper.pdf_symm
    f.attrs["command"] = meta_info.get("command", "")
    f.attrs["time"] = meta_info.get("time", "")
    f.attrs["git_hash"] = meta_info.get("git_hash", "")
    f.attrs["git_diff"] = meta_info.get("git_diff", "")

logger.info("Done.")
