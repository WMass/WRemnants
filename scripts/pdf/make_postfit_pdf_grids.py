import numpy as np

# Add the alias back manually to make mc2hlib work
if not hasattr(np, "int"):
    np.int = int
import argparse

import h5py

from wremnants.postprocessing.postfit_pdf_helper import (
    RabbitPostfitPdfHelper,
    SimplePostfitPdfHelper,
)
from wremnants.utilities import theory_utils
from wums import logging

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--fitresult",
    type=str,
    required=True,
    help="Path to the fit result file (rabbit HDF5 or simple covariance HDF5).",
)
parser.add_argument(
    "-o",
    "--outfolder",
    type=str,
    required=True,
    help="Output path for the postfit PDF grids (created if it doesn't already exist.",
)
parser.add_argument(
    "-p",
    "--pdfName",
    type=str,
    required=False,
    choices=["auto", *theory_utils.pdfMap.keys()],
    default="auto",
    help="Name of the PDF set to use. If 'auto', will use the PDF from the fit result metadata.",
)
parser.add_argument(
    "-v", "--verbose", choices=[0, 1, 2, 3, 4], default=3, help="Set verbosity level."
)
parser.add_argument(
    "-l", "--fitLabel", type=str, default="cmsmw", help="Label in the output PDF grids"
)
parser.add_argument(
    "-i", "--lhaid", type=str, required=True, help="LHAPDF ID to give the new set"
)
parser.add_argument(
    "--noColorLogger", action="store_true", help="Disable colored logging output."
)
parser.add_argument(
    "--pseudoData",
    type=str,
    default=None,
    help="Pseudo-data label to use (rabbit format only).",
)
parser.add_argument(
    "--symmetrizePdf",
    type=str,
    choices=["quadratic", "average"],
    default=None,
    help="Symmetrization method for asymmetric Hessian PDF uncertainties. "
    "Overrides the value stored in the fit result metadata.",
)
args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)


def is_simple_format(path):
    """Return True if the HDF5 file is in the simple covariance format."""
    with h5py.File(path, "r") as f:
        return "covariance" in f


if is_simple_format(args.fitresult):
    logger.info("Detected simple covariance HDF5 format.")
    pdf_helper = SimplePostfitPdfHelper(args.fitresult)
    if args.pdfName != "auto" and args.pdfName != pdf_helper.pdfName:
        raise ValueError(
            f"Specified PDF name {args.pdfName} does not match input PDF {pdf_helper.pdfName}."
        )
else:
    logger.info("Detected rabbit HDF5 format.")
    pdf_helper = RabbitPostfitPdfHelper(args.fitresult, pseudoData=args.pseudoData)
    if pdf_helper.pdfName is None:
        if args.pdfName == "auto":
            raise ValueError(
                "PDF name must be specified if not present in fit result metadata."
            )
        logger.warning(
            "Input metadata does not contain PDF information. Using specified PDF name."
        )
        pdf_helper._init_lhapdf_attributes(
            theory_utils.pdfMap[args.pdfName]["lha_name"]
        )
    elif args.pdfName != "auto" and args.pdfName != pdf_helper.pdfName:
        raise ValueError(
            f"Specified PDF name {args.pdfName} does not match input PDF {pdf_helper.pdfName}."
        )

if args.symmetrizePdf is not None:
    pdf_helper.pdf_symm = args.symmetrizePdf

# TODO: Need to scale back at the end to get 95% CL for consistency?

postfit_matrix, new_central, central_pdf_path = pdf_helper.compute_postfit_matrix()
pdf_helper.write_grids(
    central_pdf_path,
    args.outfolder,
    args.fitLabel,
    args.lhaid,
    postfit_matrix,
    new_central,
)
