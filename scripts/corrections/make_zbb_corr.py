"""
Make Zbb correction histogram by comparing massive and massless Z samples, and save the ratio and corrected histograms in a pkl.lz4 file.
Takes as input the gen level histograms from w_z_gen_dists.py histmaker ran with --addBottomAxis.
"""

import os

import hist
import numpy as np

from wremnants.utilities import common, parsing
from wremnants.utilities.io_tools import input_tools
from wums import boostHistHelpers as hh
from wums import logging, output_tools


def parse_args():
    parser = parsing.base_parser()
    parser.add_argument(
        "massiveFile",
        type=str,
        help="Input file with massive-b Z+bb sample (hdf5 or pkl.lz4).",
    )
    parser.add_argument(
        "masslessFile",
        type=str,
        help="Input file with massless Z sample (hdf5 or pkl.lz4).",
    )
    parser.add_argument(
        "--massiveProc",
        "--massive-proc",
        dest="massiveProc",
        type=str,
        default="Zbb_MiNNLO",
        help="Process name for the massive-b sample.",
    )
    parser.add_argument(
        "--masslessProc",
        "--massless-proc",
        dest="masslessProc",
        type=str,
        default="Zmumu_MiNNLO",
        help="Process name for the massless sample.",
    )
    parser.add_argument(
        "--histName",
        "--hist-name",
        dest="histName",
        type=str,
        default="nominal_gen",
        help="Histogram name to use for the correction.",
    )
    parser.add_argument(
        "--generator",
        type=str,
        default="MiNNLO_Zbb",
        help="Prefix for output keys and file name.",
    )
    parser.add_argument(
        "--outpath",
        type=str,
        default=".",
        help="Output directory for the pkl.lz4 file.",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=None,
        help="Explicit output filename (overrides generator/outpath).",
    )
    return parser.parse_args()


def get_hist(results, proc, hist_name):
    if proc not in results:
        raise KeyError(f"Process '{proc}' not found in input file")
    if hist_name not in results[proc]["output"]:
        raise KeyError(f"Histogram '{hist_name}' not found for process '{proc}'")
    hist_obj = results[proc]["output"][hist_name]
    return hist_obj.get() if hasattr(hist_obj, "get") else hist_obj


def scale_hist(results, proc, hist_obj):
    scale = results[proc]["dataset"]["xsec"] * 10e6 / results[proc]["weight_sum"]
    return hh.scaleHist(hist_obj, scale, createNew=True)


def add_dummy_axes(h2d):
    h2d = h2d.project("absYVgen", "ptVgen")
    axis_absy = h2d.axes["absYVgen"]
    axis_pt = h2d.axes["ptVgen"]
    axis_Q = hist.axis.Regular(1, 60, 120, name="Q", underflow=False, overflow=False)
    axis_charge = hist.axis.Regular(1, 0, 1, name="charge")
    axis_vars = hist.axis.StrCategory(["nominal", "mb_up"], name="vars")
    out = hist.Hist(
        axis_Q,
        axis_absy,
        axis_pt,
        axis_charge,
        axis_vars,
        storage=hist.storage.Double(),
    )
    out.values(flow=True)[...] = np.ones_like(
        out.values(flow=True)
    )  # set under/overflow to 1
    out.values()[0, :, :, 0, 1] = h2d.values()
    return out


def scale_to_integral(h, target, label):
    current = h.sum().value
    if current <= 0:
        raise ValueError(
            f"Cannot scale '{label}' to target integral: current integral is <= 0"
        )
    return hh.scaleHist(h, target / current, createNew=True)


def main():
    args = parse_args()
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    res_massive, meta_massive, _ = input_tools.read_infile(args.massiveFile)
    res_massless, meta_massless, _ = input_tools.read_infile(args.masslessFile)

    h_massive = scale_hist(
        res_massive,
        args.massiveProc,
        get_hist(res_massive, args.massiveProc, args.histName),
    )
    h_massless = scale_hist(
        res_massless,
        args.masslessProc,
        get_hist(res_massless, args.masslessProc, args.histName),
    )

    vars_2d = ["ptVgen", "absYVgen"]
    nominal = h_massless.project(*vars_2d)
    nominal_nobottom = h_massless[{"bottom_sel": 0}].project(*vars_2d)
    massive = h_massive[{"bottom_sel": 1}].project(*vars_2d)

    corrected = hh.addHists(nominal_nobottom, massive)
    h2_ratio = hh.divideHists(corrected, nominal)

    ratio_out = add_dummy_axes(h2_ratio)
    corrected_out = add_dummy_axes(corrected)
    nominal_out = add_dummy_axes(nominal)

    output_dict = {
        f"{args.generator}_minnlo_ratio": ratio_out,
        f"{args.generator}_hist": corrected_out,
        "minnlo_ref_hist": nominal_out,
    }
    meta_dict = {
        "massiveFile": args.massiveFile,
        "masslessFile": args.masslessFile,
        "massive_file": args.massiveFile,
        "massless_file": args.masslessFile,
        "massive_meta": meta_massive[0] if meta_massive else None,
        "massless_meta": meta_massless[0] if meta_massless else None,
    }
    os.makedirs(args.outpath, exist_ok=True)
    outfile = args.outfile
    if outfile is None:
        outfile = os.path.join(args.outpath, f"{args.generator}_CorrZ.pkl.lz4")

    output_tools.write_lz4_pkl_output(
        outfile, "Z", output_dict, common.base_dir, args, meta_dict
    )


if __name__ == "__main__":
    main()
