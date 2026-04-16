import argparse

import h5py
import numpy as np

parser = argparse.ArgumentParser(
    description="Read and display postfit PDF covariance, pulls, and labels from an HDF5 file."
)
parser.add_argument(
    "-f",
    "--input",
    type=str,
    required=True,
    help="Input HDF5 file written by write_postfit_pdf_covariance.py.",
)
args = parser.parse_args()

with h5py.File(args.input, "r") as f:
    cov = f["covariance"][:]
    pulls = f["pulls"][:]
    labels = f["labels"][:].astype(str)

print("labels:", labels)
print("pulls:", pulls)
print("covariance shape:", cov.shape)
print("covariance diagonal:", np.diag(cov))
