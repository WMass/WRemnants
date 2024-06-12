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
import ROOT

parser = argparse.ArgumentParser()

parser.add_argument("--debug", action='store_true', help="Print debug output")
parser.add_argument("--random", action='store_true', help="Produce correction with statistical fluctuations")
parser.add_argument("--project", default=["massVgen", "absYVgen", "csCosThetagen"], nargs="*", type=str, help="axes to project to")
parser.add_argument("-p", "--postfix", type=str, help="Postfix for plots and correction files")

args = parser.parse_args()

data_dir = common.data_dir

def load_renesance(dirname):

    def load_hist(fname):
        f = ROOT.TFile.Open(fname)
        f.ls()

        h = f.Get("h_dilepton_m_y_costheta")
        hxsec = f.Get("h_xsec")

        h = narf.root_to_hist(h, axis_names = ["massVgen", "yVgen", "csCosThetagen"])
        hxsec = narf.root_to_hist(hxsec)

        f.Close()

        # this is already 1.0 so no need to actually apply it, but just checking
        wnorm = hxsec.values()[0]/h.sum(flow=True).value
        print("wnorm", wnorm)

        axis_yVgen = h.axes["yVgen"]

        if 0. not in axis_yVgen.edges:
            raise ValueError("Can't consistently convert to absolute rapidity unless there is a bin edge at 0.")

        if not isinstance(axis_yVgen, hist.axis.Regular):
            raise ValueError("Expected a regular axis for the rapidity")

        axis_absYVgen = hist.axis.Regular(axis_yVgen.size//2, 0., axis_yVgen.edges[-1], underflow=False, overflow=True, name="absYVgen")

        axis_massVgen = h.axes["massVgen"]
        axis_csCosThetagen = h.axes["csCosThetagen"]

        hout = hist.Hist(axis_massVgen, axis_absYVgen, axis_csCosThetagen, storage=h.storage_type())


        for val in axis_yVgen.centers:
            hout[{"absYVgen" : abs(val)*1.j}] = hout[{"absYVgen" : abs(val)*1.j}].view(flow=True) + h[{"yVgen" : val*1.j}].view(flow=True)

        hout[{"absYVgen" : hist.overflow}] = h[{"yVgen" : hist.underflow}].view(flow=True) + h[{"yVgen" : hist.overflow}].view(flow=True)


        mass_binning = [edge for edge in axis_massVgen.edges if (edge % 10. == 0. or (edge>70. and edge<90.))]

        hout = hh.rebinHist(hout, axis_name = "massVgen", edges = mass_binning)
        hout = hout[{"absYVgen" : hist.rebin(2)}]

        print(hout.project("massVgen"))
        print(hout.project("absYVgen"))
        print(hout.project("csCosThetagen"))

        hout = hout.project(*args.project)

        print(h)
        print(hout)

        return hout

    fnameLO = f"{dirname}/plots_LO.root"
    fnameNLO = f"{dirname}/plots_NLO.root"

    hLO = load_hist(fnameLO)
    hNLO = load_hist(fnameNLO)

    if (args.random):
        rng = np.random.default_rng(12345)

        hNLO.values(flow=True)[...] = np.maximum(0., hLO.values(flow=True) + rng.standard_normal(hLO.values(flow=True).shape)*np.sqrt(hLO.variances(flow=True)))
        hNLO.variances(flow=True)[...] = hLO.variances(flow=True)


    print("hLO", hLO)
    print("hNLO", hNLO)

    hcorr = hh.divideHists(hNLO, hLO)


    print("hcorr", hcorr)
    print(np.mean(hcorr.values()))
    return hcorr

rdirwm = f"{data_dir}//EWCorrections/reneSANCe/pptowm_13tev_iqcd0_iqed0_iew1_iscale3_sum"
rdirwp = f"{data_dir}//EWCorrections/reneSANCe/pptowp_13tev_iqcd0_iqed0_iew1_iscale3_sum"


h_wm = load_renesance(rdirwm)
h_wp = load_renesance(rdirwp)

axis_chargeVgen = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name="chargeVgen")
axis_var = hist.axis.StrCategory(["nlo_ew_virtual"], name="var")

hcorr = hist.Hist(*h_wm.axes, axis_chargeVgen, axis_var)

hcorr[{"chargeVgen" : -1.j, "var" : "nlo_ew_virtual"}] = h_wm.values(flow=True)
hcorr[{"chargeVgen" : 1.j, "var" : "nlo_ew_virtual"}] = h_wp.values(flow=True)


hcorr.values(flow=True)[...] = np.maximum(hcorr.values(flow=True), 0.)

print(hcorr)
print(hcorr[{"chargeVgen" : -1.j, "var" : 0}])
print(hcorr[{"chargeVgen" : 1.j, "var" : 0}])

# hack output name to comply with correction code
correction_name = "renesanceEW"
if args.postfix:
    correction_name += f"_{args.postfix}"
hist_name = f"{correction_name}_minnlo_ratio"

res = {}
res[hist_name] = hcorr

output_tools.write_theory_corr_hist(correction_name, "W", res, args)
