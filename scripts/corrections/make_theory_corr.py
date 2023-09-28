import numpy as np
import lz4.frame
import pickle
from wremnants import plot_tools, theory_corrections, theory_tools
from utilities import boostHistHelpers as hh
from utilities import common, input_tools, output_tools, logging
import pathlib
import hist
import argparse
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--minnlo_file", type=str, default="w_z_gen_dists.pkl.lz4", help="MiNNLO gen file, denominator in ratio") 
parser.add_argument("-c", "--corr_files", type=str, nargs='+', required=True, help="Reference files for the corrections (both W+ and W- for the W)") 
parser.add_argument("-g", "--generator", type=str, choices=["dyturbo", "mcfm", "scetlib", "scetlib_dyturbo", "matrix_radish"], 
    required=True, help="Generator used to produce correction hist")
parser.add_argument("--outpath", type=str, default=f"{common.data_dir}/TheoryCorrections", help="Output path")
parser.add_argument("-p", "--postfix", type=str, help="Postfix for output file name", default=None)
parser.add_argument("--proc", type=str, required=True, choices=["z", "w", ], help="Process")
parser.add_argument("--axes", nargs="*", type=str, default=None, help="Use only specified axes in hist")
parser.add_argument("--debug", action='store_true', help="Print debug output")
parser.add_argument("--noColorLogger", action="store_true", default=False, help="Do not use logging with colors")
parser.add_argument("-o", "--plotdir", type=str, help="Output directory for plots")

args = parser.parse_args()

logger = logging.setup_logger("make_theory_corr", 4 if args.debug else 3, args.noColorLogger)

def read_corr(procName, generator, corr_files):
    charge = 0 if procName[0] == "Z" else (1 if "Wplus" in procName else -1)
    corr_file = corr_files[0]
    if "scetlib" in generator:
        if "dyturbo" in generator:
            scetlib_files = [x for x in corr_files if pathlib.Path(x).suffix == ".pkl"]
            if len(scetlib_files) != 2:
                raise ValueError(f"scetlib_dyturbo correction requires two SCETlib files (resummed and FO singular). Found {len(scetlib_files)}")
            if not any("nnlo_sing" in x for x in scetlib_files):
                raise ValueError("Must pass in a fixed order singular file")
            nnlo_sing_idx = 0 if "nnlo_sing" in scetlib_files[0] else 1
            resumf = scetlib_files[~nnlo_sing_idx]
            nnlo_singf = scetlib_files[nnlo_sing_idx]

            dyturbo_files = [x for x in corr_files if pathlib.Path(x).suffix == ".txt"]
            if len(dyturbo_files) != 1:
                raise ValueError("scetlib_dyturbo correction requires one DYTurbo file (fixed order contribution)")

            numh = input_tools.read_matched_scetlib_dyturbo_hist(resumf, nnlo_singf, dyturbo_files[0], args.axes, charge=charge)
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
        if args.generator == "matrix_radish":
            h = input_tools.read_matrixRadish_hist(corr_file, "ptVgen")
        else:
            axnames = args.axes
            if not axnames:
                axnames = ("Y", "qT") if "2d" in corr_file else ("qT")
            h = input_tools.read_dyturbo_hist(corr_files, axes=axnames, charge=charge)
            if "Y" in h.axes.name:
                h = hh.makeAbsHist(h, "Y")

        vars_ax = h.axes["vars"] if "vars" in h.axes.name else hist.axis.StrCategory(["central"], name="vars") 
        hnD = hist.Hist(*h.axes, vars_ax)
        # Leave off the overflow, we won't use it anyway
        hnD[...] = np.reshape(h.values(), hnD.shape)
        numh = hnD
    return numh

if args.proc == "z":
    filesByProc = { "ZmumuPostVFP" : args.corr_files }
elif args.proc == "w":
    wpfiles = list(filter(lambda x: "wp" in os.path.basename(x).lower(), args.corr_files))
    wmfiles = list(filter(lambda x: "wm" in os.path.basename(x).lower(), args.corr_files))
    if len(wpfiles) != len(wmfiles):
        raise ValueError(f"Expected equal number of files for W+ and W-, found {len(wpfiles)} (Wp) and {len(wmfiles)} (Wm)")
    filesByProc = { "WplusmunuPostVFP" : wpfiles,
        "WminusmunuPostVFP" : wmfiles}

minnloh = input_tools.read_all_and_scale(args.minnlo_file, list(filesByProc.keys()), ["nominal_gen"])[0]

if "y" in minnloh.axes.name:
    minnloh = hh.makeAbsHist(minnloh, "y")

# Get more stats in the correction
add_taus = True 
if add_taus:
    logger.info("Combining muon and tau decay samples for increased stats")
    from wremnants.datasets.datasetDict_v9 import Z_TAU_TO_LEP_RATIO,BR_TAUToMU
    taus = ["WplustaunuPostVFP", "WminustaunuPostVFP"] if args.proc == "w" else ["ZtautauPostVFP"]
    taush = input_tools.read_all_and_scale(args.minnlo_file, taus, ["nominal_gen"])[0]
    if "y" in taush.axes.name:
        taush = hh.makeAbsHist(taush, "y")
    # Rescale taus to mu to effectively have more stats
    minnloh = 0.5*(minnloh + taush/(Z_TAU_TO_LEP_RATIO if args.proc == "z" else BR_TAUToMU))

numh = hh.sumHists([read_corr(procName, args.generator, corr_file) for procName, corr_file in filesByProc.items()])

if numh.ndim-1 < minnloh.ndim:
    ax_map = {
        "ptVgen" : "qT",
        "absYVgen" : "absY",
        "absy" : "absY",
        "massVgen" : "Q",
        "chargeVgen" : "charge",
    }
    axes = []
    # NOTE: This leaves out the flow, but there shouldn't be any for the theory pred anyway
    data = numh.view()
    for i, ax in enumerate(minnloh.axes.name):
        axname = ax_map[ax]
        if axname in numh.axes.name:
            axes.append(numh.axes[axname])
        else:
            minnlo_ax = minnloh.axes[ax]
            # TODO: Should be a little careful because this won't include overflow, as long as the
            # axis range is large enough, it shouldn't matter much
            axes.append(hist.axis.Regular(1, minnlo_ax.edges[0], minnlo_ax.edges[-1], 
                flow=True, name=ax))
            data = np.expand_dims(data, i)
    if numh.axes.name[-1] == "vars":
        axes.append(numh.axes["vars"])

    numh = hist.Hist(*axes, storage=numh._storage_type(), data=data)


corrh_unc, minnloh, numh  = theory_corrections.make_corr_from_ratio(minnloh, numh)
corrh = hist.Hist(*corrh_unc.axes, name=corrh_unc.name, storage=hist.storage.Double(), data=corrh_unc.values(flow=True))

if args.postfix:
    args.generator += args.postfix
outfile = f"{args.outpath}/{args.generator}Corr{args.proc.upper()}.pkl.lz4"

with lz4.frame.open(outfile, "wb") as f:
    pickle.dump({
            args.proc.upper() : {
                f"{args.generator}_minnlo_ratio" : corrh,
                f"{args.generator}_hist" : numh,
                "minnlo_ref_hist" : minnloh,
            },
            "meta_data" : output_tools.metaInfoDict(args=args),
        }, f, protocol = pickle.HIGHEST_PROTOCOL)

logger.info("Correction binning is")
for ax in corrh.axes:
    logger.info(f"Axis {ax.name}: {ax.edges}")
logger.info(f"Average correction is {np.average(corrh.values())}")
logger.info(f"Normalization change (corr/minnlo) is {numh[{'vars' : 0 }].sum().value/minnloh.sum().value}")
logger.info(f"Wrote file {outfile}")

if args.plotdir:
    colors = {
        "scetlib_dyturbo" : "mediumpurple",
        "dyturbo" : "darkblue",
        "matrix_radish" : "green",
    }

    xlabel = {
        "massVgen" : "$m_{{{final_state}}}$ (GeV)",
        "ptVgen" : "$p_{{T}}^{{{final_state}}}$ (GeV)",
        "absy" : "$|y^{{{final_state}}}|$",
    }
    
    for charge in minnloh.axes["chargeVgen"].centers:
        charge = complex(0, charge)
        proc = 'Z' if args.proc == 'z' else ("Wp" if charge.imag > 0 else "Wm")

        meta_dict = {}
        for f in [args.minnlo_file]+args.corr_files:
            try:
                meta = input_tools.get_metadata(f)
                meta_dict[f] = meta
            except ValueError as e:
                logger.warning(f"No meta data found for file {f}")
                pass

        fig, ax = plt.subplots(figsize=(6, 6))
        corrh[{"vars" : 0, "charge" : charge, "massVgen" : 0}].plot(ax=ax)
        final_state = "\\ell\\ell" if args.proc == 'z' else ("\\ell^{+}\\nu" if charge.imag > 0 else "\\ell^{-}\\nu")

        plot_tools.make_plot_dir(*args.plotdir.rsplit("/", 1))
        plot_name = f"corr2D_{args.generator}_MiNNLO_{proc}"
        plot_tools.save_pdf_and_png(args.plotdir, plot_name)
        plot_tools.write_index_and_log(args.plotdir, plot_name, args=args, analysis_meta_info=meta_dict)
        
        for varm,varn in zip(minnloh.axes.name[:-1], numh.axes.name[:-2]):
            fig = plot_tools.makePlotWithRatioToRef(
                [minnloh[{"chargeVgen" : charge}].project(varm),
                    numh[{"vars" : 0, "charge" : charge}].project(varn),
                ],
                ["MiNNLO", args.generator, 
                ],
                colors=["orange", "mediumpurple"], 
                linestyles=["solid", "dashed", ],
                xlabel=xlabel[varm].format(final_state=final_state), 
                ylabel="Events/bin",
                rlabel="x/MiNNLO",
                legtext_size=24,
                rrange=[0.8, 1.2],
                yscale=1.1,
                xlim=None, binwnorm=1.0, baseline=True
            )
            plot_name = f"{varm}_{args.generator}_MiNNLO_{proc}"
            plot_tools.save_pdf_and_png(args.plotdir, plot_name)
            plot_tools.write_index_and_log(args.plotdir, plot_name, args=args,
                analysis_meta_info=meta_dict)
