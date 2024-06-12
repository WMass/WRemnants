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
parser.add_argument("--project", default=["massVgen", "absYVgen"], nargs="*", type=str, help="axes to project to")
parser.add_argument("-p", "--postfix", type=str, help="Postfix for plots and correction files")

args = parser.parse_args()

data_dir = common.data_dir


def dorebin(h):
    axis_yVgen = h.axes["yVgen"]

    if 0. not in axis_yVgen.edges:
        raise ValueError("Can't consistently convert to absolute rapidity unless there is a bin edge at 0.")

    if not isinstance(axis_yVgen, hist.axis.Regular):
        raise ValueError("Expected a regular axis for the rapidity")

    axis_absYVgen = hist.axis.Regular(axis_yVgen.size//2, 0., axis_yVgen.edges[-1], underflow=False, overflow=True, name="absYVgen")

    hout = hist.Hist(*[axis if axis.name != "yVgen" else axis_absYVgen for axis in h.axes], storage=h.storage_type())

    for val in axis_yVgen.centers:
        hout[{"absYVgen" : abs(val)*1.j}] = hout[{"absYVgen" : abs(val)*1.j}].view(flow=True) + h[{"yVgen" : val*1.j}].view(flow=True)

    hout[{"absYVgen" : hist.overflow}] = h[{"yVgen" : hist.underflow}].view(flow=True) + h[{"yVgen" : hist.overflow}].view(flow=True)

    axis_massVgen = h.axes["massVgen"]

    mass_binning = [edge for edge in axis_massVgen.edges if (edge % 10. == 0. or (edge>70. and edge<90.))]

    hout = hh.rebinHist(hout, axis_name = "massVgen", edges = mass_binning)
    hout = hout[{"absYVgen" : hist.rebin(2)}]

    hout = hout.project(*args.project)

    return hout



def load_renesance(dirname):

    def load_hist(fname):
        f = ROOT.TFile.Open(fname)
        f.ls()

        h = f.Get("h_dilepton_m_y")
        h0 = f.Get("h_dilepton_m_y_mom0")
        h4 = f.Get("h_dilepton_m_y_mom4")
        hxsec = f.Get("h_xsec")

        h = narf.root_to_hist(h, axis_names = ["massVgen", "yVgen"])
        h0 = narf.root_to_hist(h0, axis_names = ["massVgen", "yVgen"])
        h4 = narf.root_to_hist(h4, axis_names = ["massVgen", "yVgen"])
        hxsec = narf.root_to_hist(hxsec)

        f.Close()

        # this is already 1.0 so no need to actually apply it, but just checking
        wnorm = hxsec.values()[0]/h.sum(flow=True).value
        print("wnorm", wnorm)

        h = dorebin(h)
        h0 = dorebin(h0)
        h4 = dorebin(h4)

        axis_csCosThetagen = hist.axis.Regular(200, -1., 1., name="csCosThetagen")

        hout = hist.Hist(*h.axes, axis_csCosThetagen)

        costhetavals = axis_csCosThetagen.centers
        costhetavals = np.concatenate([[-1.], costhetavals, [1.]])
        costhetavals = costhetavals[None, None, ...]

        print("costhetavals", costhetavals)
        print(costhetavals.shape)

        hout[...] = h.values(flow=True)[..., None]*(1. + costhetavals**2) + h0.values(flow=True)[..., None]*0.5*(1. - 3.*costhetavals**2) + h4.values(flow=True)[..., None]*costhetavals

        return hout

    fnameLO = f"{dirname}/plots_LO.root"
    fnameNLO = f"{dirname}/plots_NLO.root"

    hLO = load_hist(fnameLO)
    hNLO = load_hist(fnameNLO)

    print("hLO", hLO)
    print("hNLO", hNLO)

    hcorr = hh.divideHists(hNLO, hLO)

    # axis_var = hist.axis.StrCategory(["nlo_ew_virtual", "no_ew_virtual"], name="var")
    # hcorr = hist.Hist(*hNLO.axes, axis_var)
    #
    # hcorr[{"var" : "nlo_ew_virtual"}] = hh.divideHists(hNLO, hLO).values(flow=True)
    # hcorr[{"var" : "no_ew_virtual"}] = np.ones_like(hLO.values(flow=True))

    print("hcorr", hcorr)
    print(np.mean(hcorr.values()))

    # print(h)
    return hcorr

rdirwm = f"{data_dir}//EWCorrections/reneSANCe/pptowm_13tev_iqcd0_iqed0_iew1_iscale3_sum"
rdirwp = f"{data_dir}//EWCorrections/reneSANCe/pptowp_13tev_iqcd0_iqed0_iew1_iscale3_sum"


h_wm = load_renesance(rdirwm)
h_wp = load_renesance(rdirwp)

axis_chargeVgen = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name="chargeVgen")
axis_var = hist.axis.StrCategory(["nlo_ew_virtual"], name="var")
# axis_var = h_wp.axes["var"]

# hcorr = hist.Hist(*[axis for axis in h_wm.axes if axis.name != "var"], axis_chargeVgen, axis_var)

hcorr = hist.Hist(*h_wm.axes, axis_chargeVgen, axis_var)

hcorr[{"chargeVgen" : -1.j, "var" : "nlo_ew_virtual"}] = h_wm.values(flow=True)
hcorr[{"chargeVgen" : 1.j, "var" : "nlo_ew_virtual"}] = h_wp.values(flow=True)

# hcorr[{"chargeVgen" : -1.j}] = h_wm.values(flow=True)
# hcorr[{"chargeVgen" : 1.j}] = h_wp.values(flow=True)

hcorr.values(flow=True)[...] = np.maximum(hcorr.values(flow=True), 0.)

print(hcorr)
print(hcorr[{"chargeVgen" : -1.j, "var" : 0}])
print(hcorr[{"chargeVgen" : 1.j, "var" : 0}])

# hack output name to comply with correction code
correction_name = "renesanceEW_smooth"
if args.postfix:
    correction_name += f"_{args.postfix}"
hist_name = f"{correction_name}_minnlo_ratio"

res = {}
res[hist_name] = hcorr

output_tools.write_theory_corr_hist(correction_name, "W", res, args)
