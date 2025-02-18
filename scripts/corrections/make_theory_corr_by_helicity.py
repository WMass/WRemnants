import argparse

import numpy as np

from utilities import common
from utilities.io_tools import input_tools, output_tools
from wremnants import theory_corrections
from wums import boostHistHelpers as hh
from wums import logging

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--minnlo_file",
    type=str,
    required=True,
    help="MiNNLO gen file, denominator in ratio",
)
parser.add_argument(
    "-ul",
    "--corr-ul",
    type=str,
    nargs="+",
    required=False,
    help="Reference file of the corrected sigma_UL",
)
parser.add_argument(
    "-a4",
    "--corr-a4",
    type=str,
    nargs="+",
    required=False,
    help="Reference file of the corrected sigma_4",
)
parser.add_argument(
    "-f",
    "--other-coeff-files",
    type=str,
    nargs="+",
    required=False,
    help="Files with other Ai coefficients",
)
parser.add_argument(
    "-c",
    "--other-coeffs",
    type=int,
    nargs="+",
    default=[0, 1, 2, 3],
    help="Files with other Ai coefficients",
)
parser.add_argument(
    "-g",
    "--generator",
    type=str,
    choices=[
        "dyturbo",
        "scetlib",
        "minnlo_dyturbo",
        "scetlib_dyturbo",
        "matrix_radish",
    ],
    required=True,
    help="Generator used to produce sigma_ul (and possibly A4) correction hist",
)
parser.add_argument(
    "--outpath",
    type=str,
    default=f"{common.data_dir}/TheoryCorrections",
    help="Output path",
)
parser.add_argument(
    "-p", "--postfix", type=str, help="Postfix for output file name", default=None
)
parser.add_argument(
    "--verbose", choices=[0, 1, 2, 3, 4], default=3, help="Output verbosity level"
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
    "--axes", nargs="*", type=str, default=None, help="Use only specified axes in hist"
)

args = parser.parse_args()

logger = logging.setup_base_logger("makeDataMCStackPlot", args.verbose)

if all(getattr(args, x) is None for x in ["corr_ul", "corr_a4", "other_coeffs"]):
    raise ValueError("Must specify at least one correction file")

processes = (
    ["ZmumuPostVFP"] if args.proc == "z" else ["WplusmunuPostVFP", "WminusmunuPostVFP"]
)

binning = {
    "qT": common.ptV_corr_binning,
    "absY": [0 + 0.5 * i for i in range(9)] + [5.0],
}
binning["absy"] = binning["absY"]
binning["ptVgen"] = binning["qT"]

minnloh = hh.sumHists(
    [
        input_tools.read_mu_hist_combine_tau(
            args.minnlo_file, proc, "nominal_gen_helicity_moments_scale"
        )
        for proc in processes
    ]
)
minnloh = hh.makeAbsHist(minnloh, "y")
minnloh = hh.rebinHistMultiAx(minnloh, binning)
minnloh = minnloh[{"muRfact": 1.0j, "muFfact": 1.0j}]

sigma_ulh = theory_corrections.read_combined_corrs(
    processes, args.generator, args.corr_ul, axes=args.axes, rebin=binning
)
sigma4h = theory_corrections.read_combined_corrs(
    processes, args.generator, args.corr_a4, axes=args.axes, rebin=binning
)
if sigma4h:
    sigma4h = theory_corrections.flip_hist_y_sign(
        sigma4h, "y" if "y" in sigma4h.axes.name else "Y"
    )

dyturbo_coeffs = None
if args.other_coeff_files and args.other_coeffs:
    dyturbo_coeffs = hh.sumHists(
        [
            input_tools.read_dyturbo_angular_coeffs(
                f, rebin=binning, absy=True, add_axes=[minnloh.axes["massVgen"]]
            )
            for f in args.other_coeff_files
        ]
    )

# Workaround for the different axis names
corrh = theory_corrections.make_corr_by_helicity(
    minnloh,
    sigma_ulh,
    sigma4h,
    coeff_hist=dyturbo_coeffs,
    coeffs_from_hist=args.other_coeffs,
    binning=binning,
)

outName = "Z" if args.proc == "z" else "W"
outfile = f"{args.outpath}/{args.generator}Helicity"

out_dict = {
    f"{args.generator}_minnlo_coeffs": corrh,
    f"{args.generator}_sigma4_hist": sigma4h,
    f"{args.generator}_sigmaUL_hist": sigma_ulh,
    "minnlo_helicity_hist": minnloh,
    f"other_coeffs": dyturbo_coeffs,
}

output_tools.write_theory_corr_hist(outfile, outName, out_dict, args)

logger.info("Correction binning is")
for ax in corrh.axes:
    logger.info(f"Axis {ax.name}: {ax.edges}")
for i in range(-1, 8):
    h = corrh[{"helicity": complex(0, i)}]
    helcorr = np.average(hh.divideHists(h[{"corr": True}], h[{"corr": False}]))
    logger.info(f"Average correction for sigma_{i} is {helcorr}")
logger.info(f"Wrote file {outfile}")
