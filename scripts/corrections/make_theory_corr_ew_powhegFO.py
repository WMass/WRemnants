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
histnames.append("lhe_weak_helicity")

hists = input_tools.read_all_and_scale(fname = args.input, procs = procs, histnames = histnames)
lhe_weak_helicity = hists[0]

h = lhe_weak_helicity

print(h)

# nominal correction should be the first entry, so re-order the variations
nominal_corr = "weak_default"
weakvars = list(h.axes["weak"])
weakvars.remove(nominal_corr)
weakvars.insert(0, nominal_corr)

s = hist.tag.Slicer()

# integrate over pt and rapidity and re-order the variations
h = h[{"ptVlhe" : hist.sum, "absYVlhe" : hist.sum, "weak" : weakvars}]

# single bin axes for pt and y since the correction code assumes they're present
axis_massVlhe = h.axes["massVlhe"]
axis_absYVlhe = hist.axis.Variable([0., np.inf], underflow=False, overflow=False, name = "absYVlhe")
axis_ptVlhe = hist.axis.Variable([0., np.inf], underflow=False, overflow=False, name = "ptVlhe")
axis_chargeVlhe = h.axes["chargeVlhe"]
axis_helicity = h.axes["helicity"]
axis_uncorr_corr = hist.axis.StrCategory(["uncorr", "corr"], name="uncorr_corr")
axis_weak = h.axes["weak"]

# this is the order of the variables expected by the correction code
hcorr = hist.Hist(axis_massVlhe, axis_absYVlhe, axis_ptVlhe, axis_chargeVlhe, axis_helicity, axis_uncorr_corr, axis_weak)
# set uncorr to the denominator of the correction (LO)
hcorr[{"uncorr_corr" : "uncorr", "ptVlhe" : 0, "absYVlhe" : 0}] = h[{"weak" : "weak_no_ew"}].values(flow=True)[..., None]
hcorr[{"uncorr_corr" : "corr", "ptVlhe" : 0, "absYVlhe" : 0}] = h.values(flow=True)

# sample is NLO QCD so NNLO angular coeffs are just noise, set them to zero
for ihel in range(5, 8):
    hcorr[{"helicity" : ihel*1.j}] = np.zeros_like(hcorr[{"helicity" : ihel*1.j}].values(flow=True))

# EW corrections for NLO helicity contributions are not meaningful, so just preserve the corresponding
# angular coefficients
uncorr_UL = hcorr[{"uncorr_corr" : "uncorr", "helicity" : -1.j}].values(flow=True)
corr_UL = hcorr[{"uncorr_corr" : "corr", "helicity" : -1.j}].values(flow=True)
for ihel in range(0, 4):
    corr = hcorr[{"uncorr_corr" : "corr", "helicity" : ihel*1.j}].values(flow=True)
    uncorr = hcorr[{"uncorr_corr" : "uncorr", "helicity" : ihel*1.j}].values(flow=True)
    hcorr[{"uncorr_corr" : "corr", "helicity" : ihel*1.j}] = np.where(uncorr_UL == 0., corr, corr_UL/uncorr_UL*uncorr)

# extreme bins are not populated, "extend" the corrections from the neighboring mass bins
hcorr[{"massVlhe" : 0}] = hcorr[{"massVlhe" : 1}].values(flow=True)
hcorr[{"massVlhe" : -1}] = hcorr[{"massVlhe" : -2}].values(flow=True)
hcorr[{"massVlhe" : hist.overflow}] = hcorr[{"massVlhe" : -2}].values(flow=True)

if args.debug:
    print(hcorr)

    hnumUL = hcorr[{"uncorr_corr" : "corr", "ptVlhe" : 0, "absYVlhe" : 0, "chargeVlhe" : 0.j, "helicity" : -1.j, "weak" : 0}]
    hdenUL = hcorr[{"uncorr_corr" : "uncorr", "ptVlhe" : 0, "absYVlhe" : 0, "chargeVlhe" : 0.j, "helicity" : -1.j, "weak" : 0}]
    hrUL = hh.divideHists(hnumUL, hdenUL)

    for ihel in range(-1, 8):
        hnum = hcorr[{"uncorr_corr" : "corr", "ptVlhe" : 0, "absYVlhe" : 0, "chargeVlhe" : 0.j, "helicity" : ihel*1.j, "weak" : 0}]
        hden = hcorr[{"uncorr_corr" : "uncorr", "ptVlhe" : 0, "absYVlhe" : 0, "chargeVlhe" : 0.j, "helicity" : ihel*1.j, "weak" :0 }]

        hr = hh.divideHists(hnum, hden)
        hr2 = hh.divideHists(hr, hrUL)

        print("ihel", ihel)
        print(hr)
        print(hr2)

# hack output name to comply with correction code
correction_name = "powhegFOEWHelicity"
hist_name = "powhegFOEW_minnlo_coeffs"

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
