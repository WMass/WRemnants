import argparse

import hist
import numpy as np

from utilities import common
from utilities.io_tools import input_tools, output_tools
from wums import boostHistHelpers as hh
from wums import logging

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    nargs="+",
    type=str,
    default=["w_z_gen_dists_scetlib_dyturboCorr_ewinput.hdf5"],
    help="File containing EW hists",
)
parser.add_argument(
    "--nums", nargs="+", type=str, default=["horace-nlo"], help="Numerators"
)
parser.add_argument("--den", type=str, default="horace-lo-photos", help="Denominator")
parser.add_argument(
    "-p", "--postfix", type=str, help="Postfix for plots and correction files"
)
parser.add_argument(
    "--normalize",
    action="store_true",
    default=False,
    help="Normalize distributions before computing ratio",
)
parser.add_argument(
    "--noSmoothing",
    action="store_true",
    default=False,
    help="Disable smoothing of corrections",
)
parser.add_argument("--debug", action="store_true", help="Print debug output")
parser.add_argument("--baseName", default="ew_MllPTll", type=str, help="histogram name")
parser.add_argument(
    "--project",
    default=["ewMll", "ewPTll"],
    nargs="*",
    type=str,
    help="axes to project to",
)
parser.add_argument(
    "--outname", type=str, default=None, help="Output name for correction file"
)
parser.add_argument(
    "--outpath",
    type=str,
    default=f"{common.data_dir}/TheoryCorrections",
    help="Output path",
)

args = parser.parse_args()

logger = logging.setup_logger("make_theory_corr_ew", 4 if args.debug else 3)

procs = ["Zmumu", "Wplusmunu", "Wminusmunu"]
charge_dict = {"Zmumu": 0, "Wplusmunu": 1, "Wminusmunu": 0}

procs_dict = {
    "Zmumu": "ZmumuPostVFP",
    "Wminusmunu": "WminusmunuPostVFP",
    "Wplusmunu": "WplusmunuPostVFP",
}

project = args.project

# file created with `python WRemnants/scripts/histmakers/w_z_gen_dists.py --skipAngularCoeffs --filter horace -p ewinput`

res, meta, _ = input_tools.read_infile(args.input)

corrh = {}

for proc in procs:
    # Make 2D ratio
    logger.info(f"Make 2D ratio for {proc}")

    def prepare_hist(name):
        if name == "MiNNLO":
            proc_name = procs_dict[proc]
        else:
            proc_name = f"{proc}_{name}"

        if proc_name not in res:
            return None

        histo = res[proc_name]["output"][args.baseName].get()
        logger.info(f"Integrals for {name} {np.sum(histo.values(flow=True))}")

        if args.normalize:
            histo = hh.normalize(histo)
            logger.info(
                f"Integral for {name} after normalizing {np.sum(histo.values(flow=True))}"
            )
        else:
            histo = hh.scaleHist(
                histo,
                res[proc_name]["dataset"]["xsec"] / res[proc_name]["weight_sum"],
                createNew=False,
            )
            logger.info(
                f"Integral for {name} after scaling {np.sum(histo.values(flow=True))}"
            )

        if "ewMll" in histo.axes.name:
            edges = histo.axes["ewMll"].edges

            if proc[0] == "W" and ("horace" in name or "winhac" in name):
                logger.warning(
                    "horace and winhac samples with version <v5 have a wrong wmass by 30MeV too high, move axis edges by 30MeV"
                )
                edges = edges - 0.03

                axes = [
                    (
                        hist.axis.Variable(edges, underflow=False, name="ewMll")
                        if ax.name == "ewMll"
                        else ax
                    )
                    for ax in histo.axes
                ]
                new_histo = hist.Hist(*axes, storage=histo.storage_type())
                new_histo.view(flow=True)[...] = histo.view(flow=True)
                histo = new_histo

            rebinN = 4  # make one bin out of rebinN
            if rebinN != 1:
                if proc[0] == "Z":
                    # keep low and high bin edges of [0,50,60, ... 120] for technical reason
                    rebin_edges = np.append(
                        np.append(edges[:2], edges[2:-1][::rebinN]), edges[-1:]
                    )
                else:
                    rebin_edges = edges[::rebinN]

                logger.info(f"Rebin axis ewMll by {rebinN}")
                histo = hh.rebinHist(histo, "ewMll", rebin_edges)

        base_dev = args.baseName.split("_")[0]
        return histo

    nums = []
    hnums = []
    for num in args.nums:
        hnum = prepare_hist(num)
        if hnum is not None:
            nums.append(num)
            hnums.append(hnum)

    hden = prepare_hist(args.den)

    if hden is None:
        logger.warning(
            f"Denominator {args.den} for process {proc} not found! Continue with next process."
        )
        continue

    if len(hnums) < 1:
        logger.warning(
            f"No nominator found for {args.nums} for process {proc}! Continue with next process."
        )
        continue

    h2Dratios = []

    def make_correction(h1, h2, name):
        hratio = hh.divideHists(h1, h2)
        if not args.noSmoothing and not len(hratio.axes) > 1:
            # 2D smoothing
            hratio = hh.smoothTowardsOne(hratio)
            logger.info(
                "Integrals after smoothing {0} {1}".format(
                    np.sum(h2.values(flow=True)),
                    np.sum(h2.values(flow=True) * hratio.values(flow=True)),
                )
            )
            if args.normalize:
                scale = np.sum(h2.values(flow=True)) / np.sum(
                    h2.values(flow=True) * hratio.values(flow=True)
                )
                hratio = hh.scaleHist(hratio, scale)
                logger.info(
                    "Integrals after adjustment {0} {1}".format(
                        np.sum(h2.values(flow=True)),
                        np.sum(h2.values(flow=True) * hratio.values(flow=True)),
                    )
                )

        if len(hratio.axes) > 1:
            h2Dratios.append(hratio)

        # Add charge axis
        if proc[0] == "W":
            axis_charge = hist.axis.Regular(
                2, -2.0, 2.0, underflow=False, overflow=False, name="charge"
            )
        elif proc[0] == "Z":
            axis_charge = hist.axis.Regular(
                1, -1.0, 1.0, underflow=False, overflow=False, name="charge"
            )
        hcharge = hist.Hist(*hratio.axes, axis_charge, storage=hratio.storage_type())
        hcharge.view(flow=True)[..., charge_dict[proc]] = hratio.view(flow=True)

        hcharge_num = hist.Hist(*h1.axes, axis_charge, storage=h1.storage_type())
        hcharge_num.view(flow=True)[..., charge_dict[proc]] = h1.view(flow=True)

        hcharge_den = hist.Hist(*h2.axes, axis_charge, storage=h2.storage_type())
        hcharge_den.view(flow=True)[..., charge_dict[proc]] = h2.view(flow=True)

        # Variations: 0=original MiNNLO, 1=Horace NLO
        hones = hist.Hist(*hcharge.axes, storage=hcharge.storage_type())
        hones.values(flow=True)[...] = np.ones(hcharge.values(flow=True).shape)
        hones.variances(flow=True)[...] = np.zeros(hcharge.variances(flow=True).shape)

        # Add syst axis
        if name not in corrh:
            corrh[name] = {}
        corrh[name][proc] = {}
        corrh[name][proc]["ratio"] = hist.Hist(
            *hcharge.axes,
            hist.axis.Regular(2, 0, 2, underflow=False, overflow=False, name="systIdx"),
            storage=hist.storage.Double(),
        )
        corrh[name][proc]["ratio"].values(flow=True)[..., 0] = hones.values(flow=True)
        corrh[name][proc]["ratio"].values(flow=True)[..., 1] = hcharge.values(flow=True)

        corrh[name][proc]["num"] = hcharge_num
        corrh[name][proc]["den"] = hcharge_den

    for num, hnum in zip(nums, hnums):
        outname = args.outname if args.outname else num

        make_correction(hnum, hden, f"{outname}ew")

        for ax in project:
            hnum1D = hnum.project(ax)
            hden1D = hden.project(ax)

            make_correction(hnum1D, hden1D, f"{outname}ew_{ax}")

for name, corr_dict in corrh.items():
    outname = name.replace("-", "")
    if args.postfix:
        outname += f"_{args.postfix}"
    outfile = f"{args.outpath}/{outname}"

    if "Zmumu" in corr_dict:
        output_tools.write_theory_corr_hist(
            outfile,
            "Z",
            {
                f"{outname}_minnlo_ratio": corr_dict["Zmumu"]["ratio"],
                f"{outname}_num": corr_dict["Zmumu"]["num"],
                f"{outname}_den": corr_dict["Zmumu"]["den"],
            },
            args,
        )

    if "Wplusmunu" in corr_dict and "Wminusmunu" in corr_dict:
        corr_dict["W"] = {}
        for key in ["ratio", "num", "den"]:
            corr_dict["W"][key] = (
                corr_dict["Wplusmunu"][key] + corr_dict["Wminusmunu"][key]
            )
        output_tools.write_theory_corr_hist(
            outfile,
            "W",
            {
                f"{outname}_minnlo_ratio": corr_dict["W"]["ratio"],
                f"{outname}_num": corr_dict["W"]["num"],
                f"{outname}_den": corr_dict["W"]["den"],
            },
            args,
        )
