import numpy as np
from wremnants import theory_corrections, theory_tools
from utilities import boostHistHelpers as hh
from utilities import common, logging
from utilities.io_tools import input_tools, output_tools
import hist
import argparse
import os
import h5py
import narf
import pdb

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", nargs="?", type=str, default=["w_z_gen_dists_maxFiles_m1_powheg-weak.hdf5"], help="File containing EW hists")
parser.add_argument("--debug", action='store_true', help="Print debug output")

args = parser.parse_args()

logger = logging.setup_logger("make_theory_corr_ew_powhegFO", 4 if args.debug else 3)

procs = []
procs.append("Zmumu_powheg-weak-low")
procs.append("Zmumu_powheg-weak-peak")
procs.append("Zmumu_powheg-weak-high")


histnames = []
histnames.append("lhe_weak_angular")

hists = input_tools.read_all_and_scale(fname = args.input, procs = procs, histnames = histnames)
lhe_weak_angular = hists[0]

h = lhe_weak_angular

print(h)

s = hist.tag.Slicer()

if args.debug:
    from wremnants import plot_tools
    import mplhep as hep
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    hnum = h[{"weak" : "weak_default"}].project("massVlhe")
    hden = h[{"weak" : "weak_no_ew"}].project("massVlhe")

    plot_tools.makePlotWithRatioToRef([hden, hnum], labels = ["weak_no_ew", "weak_default"], colors = ["blue", "red"], xlim=[80.,100.], rrange=[1.008, 1.012])

    plt.savefig("plotsew/rationom.png")
    plt.close()

    axis_weak = h.axes["weak"]

    for var in axis_weak:
        hnumalt = h[{"weak" : var}].project("massVlhe")
        plot_tools.makePlotWithRatioToRef([hnum, hnumalt], labels = [var, "weak_default"], colors = ["red", "orange"], xlim=[80.,100.], rrange=[0.998, 1.002])

        plt.savefig(f"plotsew/ratio_{var}.png")
        plt.close()

# integrate over pt and phistar
h = h[{"ptVlhe" : hist.sum, "phiStarlhe" : hist.sum}]

hcorr = hist.Hist(*h.axes)
# safe default
hcorr.values(flow=True)[...] = 1.
# set variations as the ratio to NLO EW
den = h[{"weak" : "weak_default"}].values()[..., None]
hcorr[...] = np.where(den==0., np.ones_like(h.values()), h.values()/den)

# set variation for NLO EW as the ratio to LO EW
denlo = h[{"weak" : "weak_no_ew"}].values()
numnlo = h[{"weak" : "weak_default"}].values()
hcorr[{"weak" : "weak_default"}] = np.where(denlo==0., np.ones_like(numnlo), numnlo/denlo)

# extreme bins are not populated, "extend" the corrections from the neighboring mass bins
hcorr[{"massVlhe" : 0}] = hcorr[{"massVlhe" : 1}].values()
hcorr[{"massVlhe" : -1}] = hcorr[{"massVlhe" : -2}].values()
hcorr[{"massVlhe" : hist.overflow}] = hcorr[{"massVlhe" : -2}].values()

#charge axis should go at the end
hcorr = hcorr.project("massVlhe", "absYVlhe", "cosThetaStarlhe", "chargeVlhe", "weak")

print(hcorr)

# hack output name to comply with correction code
correction_name = "powhegFOEW"
hist_name = "powhegFOEW_minnlo_ratio"

res = {}
res[hist_name] = hcorr

meta_dict = {}
for f in [args.input]:
    label = os.path.basename(f)
    try:
        meta = input_tools.get_metadata(f)
        meta_dict[label] = meta
    except ValueError as e:
        logger.warning(f"No meta data found for file {f}")
        pass

output_tools.write_theory_corr_hist(correction_name, "Z", res, args, meta_dict)
