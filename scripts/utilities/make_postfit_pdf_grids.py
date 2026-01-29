import numpy as np

# Add the alias back manually to make mc2hlib work
if not hasattr(np, "int"):
    np.int = int
import argparse
import os

import lhapdf
from mc2hlib import lh
from mc2hlib.common import load_pdf

from rabbit import io_tools
from wremnants import theory_tools
from wums import logging

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--fitresult",
    type=str,
    required=True,
    help="Path to the rabbit fit result file.",
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
    "--pdf-name",
    type=str,
    required=False,
    choices=["auto", *theory_tools.pdfMap.keys()],
    default="auto",
    help="Name of the PDF set to use. If 'auto', will use the PDF from the fit result metadata.",
)
parser.add_argument(
    "-v", "--verbose", choices=[0, 1, 2, 3, 4], default=3, help="Set verbosity level."
)
parser.add_argument(
    "--noColorLogger", action="store_true", help="Disable colored logging output."
)
args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)


def pdf_covariance(fitresult, pdf_nuisances):
    cov = fitresult["cov"].get()
    var_names = np.array(cov.axes["parms_x"])

    pdf_mask = np.isin(var_names, pdf_nuisances)
    return cov.values()[np.ix_(pdf_mask, pdf_mask)]


def postfit_eignvectors(cov_pdf):
    eigv, V = np.linalg.eigh(cov_pdf)
    return V * np.sqrt(np.maximum(eigv, 0))


# TODO: Integrate with rabbit to avoid code duplication
def quadratic_symmetrization(matrix, labels):
    symm_avg = 0.5 * (matrix.values[:, ::2] + matrix.values[:, 1::2])
    symm_diff = 0.5 * np.sqrt(3) * (matrix.values[:, ::2] - matrix.values[:, 1::2])
    avg_idx = np.char.find(labels, "Avg") != -1

    if np.count_nonzero(avg_idx) != symm_avg.shape[1]:
        raise ValueError(
            "Found inconsistent number of Avg nuisances for quadratic symmetrization."
        )

    matrix.iloc[:, avg_idx] = symm_avg
    matrix.iloc[:, ~avg_idx] = symm_diff

    return matrix


def apply_symmetrization(matrix, symm, labels):
    if symm == "quadratic":
        return quadratic_symmetrization(matrix, labels)


def write_new_grids(base_name, outfolder, postfit_matrix, central_grid, pdf_scale):
    scale_label = (
        "unscaled" if pdf_scale == 1 else f"uncx{pdf_scale:.1f}".replace(".", "p")
    )
    new_pdf = f"{os.path.basename(base_name)}_cmsmw_{scale_label}"

    outdir = os.path.join(outfolder, new_pdf)
    if not os.path.exists(outdir):
        logger.info(f"Creating output folder {outfolder}")
        os.makedirs(outdir)

    inn = open(base_name + ".info", "r")
    outbase = "/".join([outdir, new_pdf])
    out = open(outbase + ".info", "w")

    for l in inn.readlines():
        if l.find("SetDesc:") >= 0:
            out.write(
                f'SetDesc: "{pdf.pdf_name} modified by CMS mW postfit covariance, with prefit pdf unc scaled by {pdf_scale}"\n'
            )
        elif l.find("NumMembers:") >= 0:
            out.write(f"NumMembers: {nhess + 1}\n")
        elif l.find("ErrorType") >= 0:
            out.write(f"ErrorType: symmhessian\n")
        else:
            out.write(l)
    inn.close()
    out.close()

    lh.write_replica(
        0, outbase, b"PdfType: 'central'\nFormat: lhagrid1\n", central_grid
    )
    for column in postfit_matrix.columns:
        header = b"PdfType: 'error'\nFormat: lhagrid1\n"
        lh.write_replica(column + 1, outbase, header, postfit_matrix[column])
    logger.info(f"Wrote PDF grids to {outbase}")


Q = 100
# Probably possible to read from LHAPDF
max_nf = 5
photon = False

fitresult, meta = io_tools.get_fitresult(args.fitresult, meta=True)

input_meta = meta["meta_info_input"]

if "meta_info_input" not in input_meta:
    if args.pdf_name == "auto":
        raise ValueError(
            "PDF name must be specified if not present in fit result metadata."
        )

    logger.warning(
        "Input metadata does not contain PDF information. Using specified PDF name."
    )
    pdf_input = args.pdf_name
else:
    pdf_input = input_meta["meta_info_input"]["args"]["pdfs"][0]

    if args.pdf_name != "auto" and args.pdf_name != pdf_input:
        raise ValueError(
            f"Specified PDF name {args.pdf_name} does not match input PDF {pdf_input}."
        )

pdf_name = theory_tools.pdfMap[pdf_input]["lha_name"]
pdf_scale = input_meta["meta_info"]["args"]["scalePdf"]
pdf_symm = input_meta["meta_info"]["args"]["symmetrizePdfUnc"]

if pdf_scale == -1:
    pdf_scale = theory_tools.pdfMap[pdf_input][
        "inflationFactor"
    ]  # TODO: Modify name for Luca's changes
    logger.info(f"Using default inflation factor from theory_tools: {pdf_scale}")

pdf_lha = lhapdf.getPDFSet(pdf_name)
errors = pdf_lha.errorInfo

if errors.coreType not in ["hessian", "symmhessian"]:
    raise ValueError(
        f"Unsupported PDF error type: {errors.corrType}. Only Hessian PDFs are supported."
    )

symm_errors = errors.coreType == "symmhessian"

errors = pdf_lha.errorInfo
nhess = errors.nmemCore

pdf_nuis_regex = r"pdf\d+"
labels, pulls, constraints = io_tools.get_pulls_and_constraints(
    fitresult,
    keep_nuisances=pdf_nuis_regex,
)

if pulls.size - 1 == nhess:
    logger.warning(
        "Found an extra nuisance parameter. Assuming pdf1 nuisance is a duplicate of the central value"
    )
    pdf_nuis_regex = r"pdf(?![1][^\d])\d+"

    labels, pulls, constraints = io_tools.get_pulls_and_constraints(
        fitresult,
        keep_nuisances=pdf_nuis_regex,
    )

pdf, fl, xgrid = load_pdf(pdf_name, Q, max_nf, photon)
new_pdf, fl, xgrid = load_pdf(pdf_name, Q, max_nf, photon)

pdf_cov = pdf_covariance(fitresult, labels)
K = postfit_eignvectors(pdf_cov)

central_pdf_path = "/".join([lhapdf.paths()[0], pdf_name, pdf_name])
headers, grids = lh.load_all_replicas(pdf, central_pdf_path)

# Exclude alpha_s, central is excluded by default in the returned matrix
# Big matrix is hessian_i-central, scale up by pdfScale
matrix = lh.big_matrix(grids[: nhess + 1]) * pdf_scale

if not symm_errors:
    logger.info(f"Applying symmetrization {pdf_symm} to PDF uncertainties.")
    matrix = apply_symmetrization(matrix, pdf_symm, labels)

new_central = grids[0] + np.sum(pulls * matrix, axis=1)

postfit_matrix = matrix.dot(K).add(new_central, axis=0)

write_new_grids(
    central_pdf_path, args.outfolder, postfit_matrix, new_central, pdf_scale
)
