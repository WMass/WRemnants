import numpy as np
import lz4.frame
import pickle
from wremnants import plot_tools, theory_corrections, theory_tools
from utilities import boostHistHelpers as hh
from utilities import common, input_tools, output_tools
import hist
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--minnlo_file", type=str, default="w_z_gen_dists.pkl.lz4", help="MiNNLO gen file, denominator in ratio") 
parser.add_argument("-c", "--corr_files", type=str, nargs='+', required=True, help="Reference files for the corrections (both W+ and W- for the W)") 
parser.add_argument("-g", "--generator", type=str, choices=["dyturbo", "scetlib", "matrix_radish"], 
    required=True, help="Generator used to produce correction hist")
parser.add_argument("--outpath", type=str, default=f"{common.data_dir}/TheoryCorrections", help="Output path")
parser.add_argument("-p", "--postfix", type=str, help="Postfix for output file name", default=None)
parser.add_argument("--proc", type=str, required=True, choices=["z", "w", ], help="Process")
parser.add_argument("--axes", nargs="*", type=str, default=None, help="Use only specified axes in hist")
parser.add_argument("--debug", action='store_true', help="Print debug output")

args = parser.parse_args()

logger = common.setup_base_logger("make_theory_corr", args.debug)

def read_corr(procName, generator, corr_file):
    print(procName, generator)
    charge = 0 if procName[0] == "Z" else (1 if "Wplus" in procName else -1)
    if generator == "scetlib":
        nons = "auto"
        #if True and not os.path.isfile(corr_file.replace(".", "_nons.")):
        if not os.path.isfile(corr_file.replace(".", "_nons.")):
            nons = ""
            logger.warning("No nonsingular file found for SCETlib. Will skip it. This is an approximation!")
        numh = input_tools.read_scetlib_hist(corr_file, charge=charge, nonsing=nons)
        numh = hh.makeAbsHist(numh, "Y")
    else:
        if args.generator == "matrix_radish":
            h = input_tools.read_matrixRadish_hist(corr_file, "ptVgen")
        else:
            axnames = args.axes
            if not axnames:
                axnames = ("y", "pt") if "2d" in corr_file else ("pt")
            print(charge)
            h = input_tools.read_dyturbo_hist(corr_file.split(":"), axes=axnames, charge=charge)
            if "y" in h.axes.name:
                h = hh.makeAbsHist(h, "y")

        # TODO: Should be a little careful because this won't include overflow, as long as the
        # axis range is large enough, it shouldn't matter much
        axes = []
        for ax in minnloh.axes.name:
            axname = ax.replace("Vgen", "")
            if axname in h.axes.name:
                axes.append(h.axes[axname])
            else:
                minnlo_ax = minnloh.axes[ax]
                axes.append(hist.axis.Regular(1, minnlo_ax.edges[0], minnlo_ax.edges[-1], 
                    underflow=minnlo_ax.traits.underflow, overflow=minnlo_ax.traits.overflow, name=ax))

        vars_ax = h.axes["vars"] if "vars" in h.axes.name else hist.axis.Integer(0, 1, flow=False, name="vars") 
        axes.append(vars_ax)

        h5D = hist.Hist(*axes)
        # Leave off the overflow, we won't use it anyway
        h5D[...] = np.reshape(h.values(), h5D.shape)
        numh = h5D
    return numh

if args.proc == "z":
    if len(args.corr_files) != 1:
        raise ValueError("Only one file expected for Z")
    filesByProc = { "ZmumuPostVFP" : args.corr_files[0] }
elif args.proc == "w":
    if len(args.corr_files) != 2:
        raise ValueError("Requires two files for W (W+ and W-)")
    plus_idx = 0 if "Wp" in args.corr_files[0] else 1
    filesByProc = { "WplusmunuPostVFP" : args.corr_files[plus_idx],
        "WminusmunuPostVFP" : args.corr_files[0 if plus_idx else 1]}

print(filesByProc)
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

print(numh)
print(minnloh)

corrh_unc  = theory_corrections.make_corr_from_ratio(minnloh, numh)
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
            "meta_data" : output_tools.metaInfoDict(),
        }, f, protocol = pickle.HIGHEST_PROTOCOL)

logger.info("Correction binning is")
for ax in corrh.axes:
    logger.info(f"Axis {ax.name}: {ax.edges}")
logger.info(f"Average correction is {corrh.sum()/np.ones_like(corrh).sum()}")
logger.info(f"Wrote file {outfile}")
