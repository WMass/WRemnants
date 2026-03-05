import os
import pathlib

import hist
import matplotlib.pyplot as plt
import numpy as np

from utilities import common, parsing
from utilities.io_tools import input_tools
from wremnants import theory_corrections
from wums import boostHistHelpers as hh
from wums import logging, output_tools, plot_tools


def parse_args():
    parser = parsing.base_parser()
    parser.add_argument(
        "-m",
        "--minnloFile",
        type=str,
        default="w_z_gen_dists.pkl.lz4",
        help="MiNNLO gen file, denominator in ratio",
    )
    parser.add_argument(
        "-c",
        "--corrFiles",
        type=str,
        nargs="+",
        required=True,
        help="Reference files for the corrections (both W+ and W- for the W)",
    )
    parser.add_argument(
        "-g",
        "--generator",
        type=str,
        choices=[
            "dyturbo",
            "scetlib",
            "scetlib_dyturbo",
            "scetlib_nnlojet",
            "matrix_radish",
        ],
        required=True,
        help="Generator used to produce correction hist",
    )
    parser.add_argument(
        "--qtCutoff",
        type=float,
        default=1.0,
        help="Upper limit for zeroing bins in the fixed order program (GeV)",
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
        "--proc",
        type=str,
        required=True,
        choices=[
            "z",
            "w",
            "bsm",
        ],
        help="Process",
    )
    parser.add_argument(
        "--minnloh",
        default="nominal_gen",
        type=str,
        help="Reference hist in MiNNLO sample",
    )
    parser.add_argument(
        "--axes",
        nargs="*",
        type=str,
        default=None,
        help="Use only specified axes in hist",
    )
    parser.add_argument(
        "--integrateAxis",
        type=str,
        default=None,
        help="Integrate over this axis after reading hist",
    )
    parser.add_argument(
        "--axlim",
        type=float,
        default=[],
        nargs="*",
        help="Restrict axis to this range. Assumes pairs of values by axis (same order), with trailing axes optional",
    )
    parser.add_argument(
        "--selectVars", type=str, nargs="*", help="Select variations from corr hist"
    )
    parser.add_argument("-o", "--plotdir", type=str, help="Output directory for plots")
    parser.add_argument(
        "--eoscp",
        action="store_true",
        help="Copy folder to eos with xrdcp rather than using the mount",
    )
    parser.add_argument(
        "--duplicateWminus",
        action="store_true",
        help="Use W- corr for W+ as well",
    )
    parser.add_argument(
        "--smooth",
        default=None,
        choices=["ratio", "numerator", "fo_sing"],
        help="Apply spline-based smoothing to correction",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the corrections",
    )
    parser.add_argument(
        "--eras",
        type=str,
        nargs="+",
        choices=common.supported_eras,
        help="Data set to process",
        default=["13TeVGen"],
    )
    args = parser.parse_args()

    return args


def read_corr(procName, generator, corrFiles, axes, qt_cutoff=1.0, smooth=None):
    logger = logging.child_logger("read_corr")
    charge = 0 if procName[0] == "Z" else (1 if "Wplus" in procName else -1)
    corr_file = corrFiles[0]
    if "scetlib" in generator:

        tools = generator.split("_")
        if len(tools) > 1:
            fo_generator = tools[1]
            scetlib_files = [x for x in corrFiles if pathlib.Path(x).suffix == ".pkl"]
            if len(scetlib_files) != 2:
                raise ValueError(
                    f"scetlib_dyturbo correction requires two SCETlib files (resummed and FO singular). Found {len(scetlib_files)}"
                )
            if not any("sing" in x for x in scetlib_files):
                raise ValueError("Must pass in a fixed order singular file")
            nnlo_sing_idx = 0 if "sing" in scetlib_files[0] else 1
            resumf = scetlib_files[~nnlo_sing_idx]
            nnlo_singf = scetlib_files[nnlo_sing_idx]

            fo_files = [x for x in corrFiles if pathlib.Path(x).suffix != ".pkl"]
            if len(fo_files) != 1:
                raise ValueError(
                    f"{generator} correction requires one fixed order file! found {len(fo_files)}"
                )

            fo_func = getattr(input_tools, f"read_matched_scetlib_{fo_generator}_hist")

            # TODO: Should probably be more general...
            smooth_args = {}
            if smooth == "fo_sing":
                smooth_args = {"smooth_nnlojet": True}
            numh = fo_func(
                resumf,
                nnlo_singf,
                fo_files[0],
                axes,
                charge=charge,
                zero_nons_bins=slice(
                    0j, complex(0, qt_cutoff)
                ),  # set bins with qT < qtCutoff GeV to 0
                **smooth_args,
            )
        else:
            nons = "auto"
            if not os.path.isfile(corr_file.replace(".", "_nons.")):
                nons = ""
                logger.warning("No nonsingular file found for SCETlib. Will skip it.")
            numh = input_tools.read_scetlib_hist(corr_file, charge=charge, nonsing=nons)
        if "Y" in numh.axes.name:
            numh = hh.makeAbsHist(numh, "Y")
        return numh
    else:
        if generator == "matrix_radish":
            h = input_tools.read_matrixRadish_hist(corr_file, "ptVgen")
        else:
            axnames = axes
            if not axnames:
                axnames = ("Y", "qT") if "2d" in corr_file else ("qT")
            h = input_tools.read_dyturbo_hist(corrFiles, axes=axnames, charge=charge)
            if "Y" in h.axes.name:
                h = hh.makeAbsHist(h, "Y")

        vars_ax = (
            h.axes["vars"]
            if "vars" in h.axes.name
            else hist.axis.StrCategory(["central"], name="vars")
        )
        hnD = hist.Hist(*h.axes, vars_ax)
        # Leave off the overflow, we won't use it anyway
        hnD[...] = np.reshape(h.values(), hnD.shape)
        numh = hnD

    return numh


def main():
    args = parse_args()

    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    ax_map = {
        "ptVgen": "qT",
        "absYVgen": "absY",
        "absy": "absY",
        "massVgen": "Q",
        "chargeVgen": "charge",
        "pdfVar": "vars",
        "alphasVar": "vars",
    }

    if args.proc == "z":
        eventgen_procs = ["Zmumu"]  # , "DYJetsToMuMuMass10to50"]
        filesByProc = {"Zmumu": args.corrFiles}
    else:
        wpfiles = list(
            filter(
                lambda x: "wp" in os.path.basename(x).lower() or "Wp" in x,
                args.corrFiles,
            )
        )
        wmfiles = list(
            filter(
                lambda x: "wm" in os.path.basename(x).lower() or "Wm" in x,
                args.corrFiles,
            )
        )
        if len(wmfiles + wpfiles) != len(args.corrFiles):
            raise ValueError("Did not consistently match all files to W+ or W-")
        if len(wpfiles) != len(wmfiles):
            if args.duplicateWminus:
                logger.warning("Using W- correction as a proxy for W+!")
                filesByProc = {"Wplusmunu": wmfiles, "Wminusmunu": wmfiles}

            else:
                raise ValueError(
                    f"Expected equal number of files for W+ and W-, found {len(wpfiles)} (Wp) and {len(wmfiles)} (Wm)"
                )
        else:
            filesByProc = {"Wplusmunu": wpfiles, "Wminusmunu": wmfiles}

        if args.proc == "w":
            eventgen_procs = ["Wplusmunu", "Wminusmunu"]
        elif args.proc == "bsm":
            eventgen_procs = [
                "WtoNMuMN5V0p001",
                "WtoNMuMN10V0p001",
                "WtoNMuMN30V0p001",
                "WtoNMuMN50V0p001",
            ]

    minnloh = hh.sumHists(
        [
            input_tools.read_mu_hist_combine_tau(
                args.minnloFile,
                proc,
                args.minnloh,
                eras=args.eras,
                combine_with_tau=args.proc != "bsm",
            )
            for proc in eventgen_procs
        ]
    )

    if "y" in minnloh.axes.name:
        minnloh = hh.makeAbsHist(minnloh, "y")

    # Rename minnlo axes to match corr, needed for the broadcast now
    for ax in minnloh.axes:
        if ax.name in ax_map:
            hh.renameAxis(minnloh, ax.name, ax_map[ax.name])

    numh = hh.sumHists(
        [
            read_corr(
                procName,
                args.generator,
                corr_file,
                args.axes,
                qt_cutoff=args.qtCutoff,
                smooth=args.smooth,
            )
            for procName, corr_file in filesByProc.items()
        ]
    )

    if args.selectVars:
        numh = numh[{"vars": args.selectVars}]

    if args.axlim:
        axes = [f"abs{x}" if x.lower() == "y" else x for x in args.axes]
        if len(args.axlim) % 2:
            raise ValueError("axlim must be in pairs of 2 (low limit, high limit)")
        numh = hh.rebinHistMultiAx(
            numh, axes, numh.axes.edges, args.axlim[::2], args.axlim[1::2]
        )

    if numh.ndim - 1 < minnloh.ndim:
        axes = []
        # NOTE: This leaves out the flow, but there shouldn't be any for the theory pred anyway
        data = numh.view()
        for i, ax in enumerate(minnloh.axes):
            if ax.name in numh.axes.name:
                axes.append(numh.axes[ax.name])
            elif not (ax.name in ax_map and ax_map[ax.name] in numh.axes.name):
                # TODO: Should be a little careful because this won't include overflow, as long as the
                # axis range is large enough, it shouldn't matter much
                axes.append(
                    hist.axis.Regular(
                        1,
                        ax.edges[0],
                        ax.edges[-1],
                        underflow=ax.traits.underflow,
                        overflow=ax.traits.overflow,
                        name=ax.name,
                    )
                )
                data = np.expand_dims(data, i)

        if axes[-1].name != "vars" and numh.axes.name[-1] == "vars":
            axes.append(numh.axes["vars"])

        numh = hist.Hist(*axes, storage=numh.storage_type(), data=data)

    if args.integrateAxis:
        if args.integrateAxis not in minnloh.axes.name:
            raise ValueError(
                f"Did not find axis {args.integrateAxis} in hist! Valid choices are {minnloh.axes.name}"
            )

        minnloh = hh.rebinHist(
            minnloh,
            args.integrateAxis,
            minnloh.axes[args.integrateAxis].edges[np.array((0, -1))],
        )
        numh = hh.rebinHist(
            numh,
            args.integrateAxis,
            numh.axes[args.integrateAxis].edges[np.array((0, -1))],
        )

    corrh_unc, minnloh, numh = theory_corrections.make_corr_from_ratio(
        minnloh, numh, smooth=args.smooth, normalize=args.normalize
    )

    if args.duplicateWminus:
        corrh_unc.view()[..., 1, 0] = corrh_unc.view()[..., 0, 0]

    nom_sum = lambda x: x.sum() if "vars" not in x.axes.name else x[{"vars": 0}].sum()
    logger.info(
        f"Minnlo norm in corr region is {nom_sum(minnloh)}, corrh norm is {nom_sum(numh)}"
    )

    corrh = hist.Hist(
        *corrh_unc.axes,
        name=corrh_unc.name,
        storage=hist.storage.Double(),
        data=corrh_unc.values(flow=True),
    )

    generator = args.generator
    if args.postfix:
        generator += f"_{args.postfix}"
    outfile = f"{args.outpath}/{generator}_Corr{args.proc.upper()}.pkl.lz4"

    meta_dict = {}
    for f in [args.minnloFile] + args.corrFiles:
        label = os.path.basename(f)
        try:
            meta = input_tools.get_metadata(f)
            meta_dict[label] = meta
            if "scetlib" in args.generator and f.endswith("pkl"):
                meta["config"] = input_tools.get_scetlib_config(f)
        except ValueError as e:
            logger.warning(f"No meta data found for file {f}")

    output_dict = {
        f"{generator}_minnlo_ratio": corrh,
        f"{generator}_hist": numh,
        "minnlo_ref_hist": minnloh,
    }

    output_tools.write_lz4_pkl_output(
        outfile, args.proc.upper(), output_dict, common.base_dir, args, meta_dict
    )

    logger.info("Correction binning is")
    for ax in corrh.axes:
        logger.info(f"Axis {ax.name}: {ax.edges}")

    index = {"charge": -1j} if args.duplicateWminus else {"charge": 0}
    if "vars" in minnloh.axes.name:
        index["vars"] = 0

    denom_yield = minnloh[index].sum()
    index["vars"] = 0
    num_yield = numh[index].sum()
    to_val = lambda x: x.value if hasattr(x, "value") else x
    norm_ratio = to_val(num_yield) / to_val(denom_yield)

    logger.info(f"Average correction is {np.average(corrh.values())}")
    logger.info(f"Normalization change (corr/minnlo) is {norm_ratio}")

    if args.plotdir:
        colors = {
            "scetlib_dyturbo": "mediumpurple",
            "scetlib_nnlojet": "pink",
            "dyturbo": "darkblue",
            "matrix_radish": "green",
        }

        xlabel = {
            "Q": "$m_{{{final_state}}}$ (GeV)",
            "qT": "$p_{{T}}^{{{final_state}}}$ (GeV)",
            "absY": "$|y^{{{final_state}}}|$",
        }

        outdir = output_tools.make_plot_dir(
            *args.plotdir.rsplit("/", 1), eoscp=args.eoscp
        )

        for charge in minnloh.axes["charge"].centers:
            if args.duplicateWminus and charge == 1:
                continue
            charge = complex(0, charge)
            if args.proc == "z":
                proc = "Z"
                final_state = "\\ell\\ell"
            else:
                proc = "Wp" if charge.imag > 0 else "Wm"
                final_state = "\\ell^{+}\\nu" if charge.imag > 0 else "\\ell^{-}\\nu"

            for imass, mass_edges in enumerate(minnloh.axes["Q"]):
                if len(minnloh.axes["Q"].centers) > 1:
                    suffix = f"_{int(mass_edges[0])}to{int(mass_edges[1])}GeV"
                    extra_text_base = [
                        f"{int(mass_edges[0])} < Q < {int(mass_edges[1])} GeV"
                    ]
                else:
                    suffix = ""
                    extra_text_base = []

                for ivar, var in enumerate(corrh.axes["vars"]):

                    if len(corrh.axes["vars"]) > 1:
                        suffix += f"_{var}"
                        extra_text = [*extra_text_base, var]
                    else:
                        extra_text = extra_text_base

                    if "vars" in minnloh.axes.name:
                        iminnloh = minnloh[{"Q": imass, "charge": charge, "vars": ivar}]
                    else:
                        iminnloh = minnloh[{"Q": imass, "charge": charge}]

                    inumh = numh[{"Q": imass, "charge": charge, "vars": ivar}]
                    icorrh = corrh[{"Q": imass, "charge": charge, "vars": ivar}]

                    fig, ax = plt.subplots(figsize=(6, 6))
                    icorrh.plot(ax=ax, cmin=0.5, cmax=1.5)

                    plot_name = f"corr2D_{generator}_MiNNLO_{proc}{suffix}"
                    plot_tools.save_pdf_and_png(outdir, plot_name)
                    output_tools.write_index_and_log(
                        outdir, plot_name, args=args, analysis_meta_info=meta_dict
                    )

                    for varm, varn in zip(iminnloh.axes.name, inumh.axes.name):
                        fig = plot_tools.makePlotWithRatioToRef(
                            [
                                iminnloh.project(varm),
                                inumh.project(varn),
                            ],
                            [
                                "MiNNLO",
                                generator.replace("_", " ").replace("FineBins ", ""),
                            ],
                            colors=["orange", "mediumpurple"],
                            linestyles=[
                                "solid",
                                "dashed",
                            ],
                            xlabel=xlabel[varm].format(final_state=final_state),
                            ylabel="Events/bin",
                            rlabel="x/MiNNLO",
                            legtext_size=24,
                            nlegcols=1,
                            rrange=[0.8, 1.2],
                            yscale=1.1,
                            xlim=None,
                            binwnorm=1.0,
                            baseline=True,
                            extra_text=extra_text,
                            extra_text_loc=(0.5, 0.7) if varm == "qT" else (0.1, 0.2),
                        )
                        plot_name = f"{varm}_{generator}_MiNNLO_{proc}{suffix}"
                        plot_tools.save_pdf_and_png(outdir, plot_name)
                        output_tools.write_index_and_log(
                            outdir, plot_name, args=args, analysis_meta_info=meta_dict
                        )

                    break  # only plot first variation
        if output_tools.is_eosuser_path(args.plotdir) and args.eoscp:
            output_tools.copy_to_eos(outdir, args.plotdir)


if __name__ == "__main__":
    main()
