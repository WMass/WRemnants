import numpy as np
from wremnants import plot_tools, theory_corrections, theory_tools
from utilities import boostHistHelpers as hh
from utilities import common, logging
from utilities.io_tools import input_tools, output_tools
import pathlib
import hist
import argparse
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--minnlo_file", type=str, default="w_z_gen_dists.pkl.lz4", help="MiNNLO gen file, denominator in ratio") 
parser.add_argument("-c", "--corr_files", type=str, nargs='+', required=True, help="Reference files for the corrections (both W+ and W- for the W)") 
parser.add_argument("-g", "--generator", type=str, choices=["dyturbo", "scetlib", "scetlib_dyturbo", "matrix_radish"], 
    required=True, help="Generator used to produce correction hist")
parser.add_argument("--outpath", type=str, default=f"{common.data_dir}/TheoryCorrections", help="Output path")
parser.add_argument("-p", "--postfix", type=str, help="Postfix for output file name", default=None)
parser.add_argument("--proc", type=str, required=True, choices=["z", "w", ], help="Process")
parser.add_argument("--minnloh", default="nominal_gen", type=str, help="Reference hist in MiNNLO sample")
parser.add_argument("--axes", nargs="*", type=str, default=None, help="Use only specified axes in hist")
parser.add_argument("--axlim", type=float, default=[], nargs='*', help="Restrict axis to this range. Assumes pairs of values by axis (same order), with trailing axes optional")
parser.add_argument("--debug", action='store_true', help="Print debug output")
parser.add_argument("--selectVars", type=str, nargs="*", help="Select variations from corr hist")
parser.add_argument("--noColorLogger", action="store_true", default=False, help="Do not use logging with colors")
parser.add_argument("-o", "--plotdir", type=str, help="Output directory for plots")
parser.add_argument("--eoscp", action="store_true", help="Copy folder to eos with xrdcp rather than using the mount")

args = parser.parse_args()

logger = logging.setup_logger("make_theory_corr", 4 if args.debug else 3, args.noColorLogger)

ax_map = {
    "ptVgen" : "qT",
    "absYVgen" : "absY",
    "absy" : "absY",
    "massVgen" : "Q",
    "chargeVgen" : "charge",
    "pdfVar" : "vars",
    "alphasVar" : "vars",
}

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

minnloh = hh.sumHists([input_tools.read_mu_hist_combine_tau(args.minnlo_file, proc, args.minnloh) for proc in filesByProc.keys()])

if "y" in minnloh.axes.name:
    minnloh = hh.makeAbsHist(minnloh, "y")

# Rename minnlo axes to match corr, needed for the broadcast now
for ax in minnloh.axes:
    if ax.name in ax_map:
        ax._ax.metadata["name"] = ax_map[ax.name]

numh = hh.sumHists([read_corr(procName, args.generator, corr_file) for procName, corr_file in filesByProc.items()])
if args.selectVars:
    numh = numh[{"vars" : args.selectVars}]

if args.axlim:
    axes = [f"abs{x}" if x.lower() == "y" else x for x in args.axes]
    if len(args.axlim) % 2:
        raise ValueError("axlim must be in pairs of 2 (low limit, high limit)")
    numh = hh.rebinHistMultiAx(numh, axes, numh.axes.edges, args.axlim[::2], args.axlim[1::2]) 

if numh.ndim-1 < minnloh.ndim:
    axes = []
    # NOTE: This leaves out the flow, but there shouldn't be any for the theory pred anyway
    data = numh.view()
    for i, ax in enumerate(minnloh.axes):
        if ax.name in numh.axes.name:
            axes.append(numh.axes[ax.name])
        elif not (ax.name in ax_map and ax_map[ax.name] in numh.axes.name):
            # TODO: Should be a little careful because this won't include overflow, as long as the
            # axis range is large enough, it shouldn't matter much
            axes.append(hist.axis.Regular(1, ax.edges[0], ax.edges[-1], 
                underflow=ax.traits.underflow, overflow=ax.traits.overflow, name=ax.name))
            data = np.expand_dims(data, i)

    if axes[-1].name != "vars" and numh.axes.name[-1] == "vars":
        axes.append(numh.axes["vars"])
    
    numh = hist.Hist(*axes, storage=numh.storage_type(), data=data)

corrh_unc, minnloh, numh  = theory_corrections.make_corr_from_ratio(minnloh, numh)

nom_sum = lambda x: x.sum() if "vars" not in x.axes.name else x[{"vars" : 0}].sum()
logger.info(f"Minnlo norm in corr region is {nom_sum(minnloh)}, corrh norm is {nom_sum(numh)}")

corrh = hist.Hist(*corrh_unc.axes, name=corrh_unc.name, storage=hist.storage.Double(), data=corrh_unc.values(flow=True))

generator = args.generator
if args.postfix:
    generator += args.postfix
outfile = f"{args.outpath}/{generator}"

meta_dict = {}
for f in [args.minnlo_file]+args.corr_files:
    label = os.path.basename(f)
    try:
        meta = input_tools.get_metadata(f)
        meta_dict[label] = meta
        if "scetlib" in args.generator and f.endswith("pkl"):
            meta["config"] = input_tools.get_scetlib_config(f)
    except ValueError as e:
        logger.warning(f"No meta data found for file {f}")
        pass

output_dict = {
    f"{generator}_minnlo_ratio" : corrh,
    f"{generator}_hist" : numh,
    "minnlo_ref_hist" : minnloh,
}

output_tools.write_theory_corr_hist(outfile, args.proc.upper(), output_dict, args, meta_dict)

logger.info("Correction binning is")
for ax in corrh.axes:
    logger.info(f"Axis {ax.name}: {ax.edges}")

num_yield = numh[{'vars' : 0 }].sum()
denom_yield = minnloh.sum() if minnloh.axes.name[-1] != "vars" else minnloh[...,0].sum()
to_val = lambda x: x.value if hasattr(x, "value") else x
norm_ratio = to_val(num_yield)/to_val(denom_yield)

logger.info(f"Average correction is {np.average(corrh.values())}")
logger.info(f"Normalization change (corr/minnlo) is {norm_ratio}")

if args.plotdir:
    colors = {
        "scetlib_dyturbo" : "mediumpurple",
        "dyturbo" : "darkblue",
        "matrix_radish" : "green",
    }

    xlabel = {
        "Q" : "$m_{{{final_state}}}$ (GeV)",
        "qT" : "$p_{{T}}^{{{final_state}}}$ (GeV)",
        "absY" : "$|y^{{{final_state}}}|$",
    }
    
    for charge in minnloh.axes["charge"].centers:
        charge = complex(0, charge)
        proc = 'Z' if args.proc == 'z' else ("Wp" if charge.imag > 0 else "Wm")

        fig, ax = plt.subplots(figsize=(6, 6))
        corrh[{"vars" : 0, "charge" : charge, "Q" : 0}].plot(ax=ax, cmin=0.5, cmax=1.5)
        final_state = "\\ell\\ell" if args.proc == 'z' else ("\\ell^{+}\\nu" if charge.imag > 0 else "\\ell^{-}\\nu")

        outdir = output_tools.make_plot_dir(*args.plotdir.rsplit("/", 1), eoscp=args.eoscp)
        plot_name = f"corr2D_{generator}_MiNNLO_{proc}"
        plot_tools.save_pdf_and_png(outdir, plot_name)
        plot_tools.write_index_and_log(outdir, plot_name, args=args, analysis_meta_info=meta_dict)
        
        for varm,varn in zip(minnloh.axes.name[:-1], numh.axes.name[:-2]):
            fig = plot_tools.makePlotWithRatioToRef(
                [minnloh[{"charge" : charge}].project(varm),
                    numh[{"vars" : 0, "charge" : charge}].project(varn),
                ],
                ["MiNNLO", generator, 
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
            plot_name = f"{varm}_{generator}_MiNNLO_{proc}"
            plot_tools.save_pdf_and_png(outdir, plot_name)
            plot_tools.write_index_and_log(outdir, plot_name, args=args,
                analysis_meta_info=meta_dict)
    if output_tools.is_eosuser_path(args.plotdir) and args.eoscp:
        output_tools.copy_to_eos(args.plotdir)
