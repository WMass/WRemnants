import os
import ROOT
import pathlib
import hist
import narf
import numpy as np
import lz4.frame
import pickle
from .correctionsTensor_helper import makeCorrectionsTensor
from utilities import boostHistHelpers as hh, common
from wremnants import theory_tools
import logging


def load_corr_helpers(procs, generators):
    corr_helpers = {}
    for proc in procs:
        corr_helpers[proc] = {}
        for generator in generators:
            fname = f"{common.data_dir}/TheoryCorrections/{generator}Corr{proc[0]}.pkl.lz4"
            if not os.path.isfile(fname):
                logging.warning(f"Did not find correction file for process {proc}, generator {generator}. No correction will be applied for this process!")
                continue
            helper_func = make_corr_helper if "Helicity" not in generator else make_corr_by_helicity_helper
            # Hack for now
            corr_hist_name = get_corr_name(generator)
            corr_helpers[proc][generator] = helper_func(fname, proc[0], corr_hist_name)
    for generator in generators:
        if not any([generator in corr_helpers[proc] for proc in procs]):
            raise ValueError(f"Did not find correction for generator {generator} for any processes!")
    return corr_helpers


def make_corr_helper_fromnp(filename=f"{common.data_dir}/N3LLCorrections/inclusive_{{process}}_pT.npz", isW=True):
    if isW:
        corrf_Wp = np.load(filename.format(process="Wp"), allow_pickle=True)
        corrf_Wm = np.load(filename.format(process="Wm"), allow_pickle=True)
        bins = corrf_Wp["bins"]
        axis_charge = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "charge")
    else:
        corrf = np.load(filename.format(process="Z"), allow_pickle=True)
        bins = corrf["bins"]
        axis_charge = hist.axis.Regular(1, -1., 1., underflow=False, overflow=False, name = "charge")

    axis_syst = hist.axis.Regular(len(bins[0]) - 1, bins[0][0], bins[0][-1], 
                    name="systIdx", overflow=False, underflow=False)
    axis_mass = hist.axis.Variable(bins[1], name="mass")
    axis_y = hist.axis.Variable(bins[2], name="y")
    axis_pt = hist.axis.Regular(len(bins[-1]) - 1, bins[-1][0], bins[-1][-1], name="pT", underflow=False)

    corrh = hist.Hist(axis_mass, axis_y, axis_pt, axis_charge, axis_syst)
    if isW:
        corrh[...,1,:] = np.moveaxis(corrf_Wp["scetlibCorr3D_Wp"], 0, -1)
        corrh[...,0,:] = np.moveaxis(corrf_Wm["scetlibCorr3D_Wm"], 0, -1)
    else:
        corrh[...,0,:] = np.moveaxis(corrf["scetlibCorr3D_Z"], 0, -1)
    corrh[hist.underflow,...] = 1.
    corrh[hist.overflow,...] = 1.
    corrh[:,hist.underflow,...] = 1.
    corrh[:,hist.overflow,...] = 1.
    corrh[:,:,hist.overflow,...] = 1.

    return makeCorrectionsTensor(corrh, ROOT.wrem.TensorCorrectionsHelper, tensor_rank=1)

def make_corr_helper(filename, proc, histname):
    with lz4.frame.open(filename) as f:
        corr = pickle.load(f)
        corrh = corr[proc][histname]

    return makeCorrectionsTensor(corrh, ROOT.wrem.TensorCorrectionsHelper, tensor_rank=1)

def make_corr_by_helicity_helper(filename, proc, histname):
    with lz4.frame.open(filename) as f:
        corr = pickle.load(f)
        corrh = corr[proc][histname]

    return makeCorrectionsTensor(corrh, ROOT.wrem.CentralCorrByHelicityHelper, tensor_rank=3)

def get_corr_name(generator):
    # Hack for now
    label = generator.replace("1D", "")
    return f"{label}_minnlo_ratio" if "Helicity" not in generator else f"{label.replace('Helicity', '')}_minnlo_coeffs"

def load_corr_helpers(procs, generators):
    corr_helpers = {}
    for proc in procs:
        corr_helpers[proc] = {}
        for generator in generators:
            fname = f"{common.data_dir}/TheoryCorrections/{generator}Corr{proc[0]}.pkl.lz4"
            if not os.path.isfile(fname):
                logging.warning(f"Did not find correction file for process {proc}, generator {generator}. No correction will be applied for this process!")
                continue
            corr_hist_name = get_corr_name(generator)
            helper_func = make_corr_helper if "Helicity" not in generator else make_corr_by_helicity_helper
            corr_helpers[proc][generator] = helper_func(fname, proc[0], corr_hist_name)
    for generator in generators:
        if not any([generator in corr_helpers[proc] for proc in procs]):
            raise ValueError(f"Did not find correction for generator {generator} for any processes!")
    return corr_helpers

def rebin_corr_hists(hists, ndim=-1, use_predefined_bins=False):
    # Allow trailing dimensions to be different (e.g., variations)
    ndims = min([x.ndim for x in hists]) if ndim < 0 else ndim
    if use_predefined_bins:
        try:
            hists = [hh.rebinHist(h, "pt" if "pt" in h.axes.name else "ptVgen", common.ptV_binning[:-2]) for h in hists]
            hists = [hh.rebinHist(h, "absy" if "absy" in h.axes.name else "absYVgen", common.absYV_binning[:-1]) for h in hists]
        except ValueError as e:
            logging.warning("Can't rebin axes to predefined binning")
    for i in range(ndims):
        # This is a workaround for now for the fact that MiNNLO has mass binning up to
        # Inf whereas SCETlib has 13 TeV
        if all([h.axes[i].size == 1 for h in hists]):
            continue
        print(i, [h.axes[i].name for h in hists])
        hists = hh.rebinHistsToCommon(hists, i)
    return hists

# Assuming the 3 physics variable dimensions are first
def set_corr_ratio_flow(corrh):
    # Probably there's a better way to do this...
    if corrh.axes[0].traits.underflow:
        corrh[hist.underflow,...] = np.ones_like(corrh[0,...].view(flow=True))
    if corrh.axes[0].traits.overflow:
        corrh[hist.overflow,...] = np.ones_like(corrh[0,...].view(flow=True))

    if corrh.axes[1].traits.underflow:
        corrh[:,hist.underflow,...] = np.ones_like(corrh[:,0,...].view(flow=True))
    if corrh.axes[1].traits.overflow:
        corrh[:,hist.overflow,...] = np.ones_like(corrh[:,0,...].view(flow=True))

    if corrh.axes[2].traits.underflow:
        corrh[:,:,hist.underflow,...] = np.ones_like(corrh[:,:,0,...].view(flow=True))
    if corrh.axes[2].traits.overflow:
        corrh[:,:,hist.overflow,...] = np.ones_like(corrh[:,:,0,...].view(flow=True))
    return corrh

def make_corr_from_ratio(denom_hist, num_hist, rebin=False):
    denom_hist, num_hist = rebin_corr_hists([denom_hist, num_hist], use_predefined_bins=rebin)

    corrh = hh.divideHists(num_hist, denom_hist)
    return set_corr_ratio_flow(corrh)

def make_corr_by_helicity(ref_helicity_hist, target_sigmaul, target_sigma4, ndim=3):
    ref_helicity_hist, target_sigmaul, target_sigma4 = rebin_corr_hists([ref_helicity_hist, target_sigmaul, target_sigma4], ndim, True)
    
    ref_coeffs = theory_tools.moments_to_angular_coeffs(ref_helicity_hist)

    target_a4_coeff = make_angular_coeff(target_sigma4, target_sigmaul)
    sigmaUL_ratio = hh.divideHists(target_sigmaul, ref_helicity_hist[{"helicity" : -1.j}])

    corr_ax = hist.axis.Boolean(name="corr")
    vars_ax = target_sigmaul.axes["vars"]
    corr_coeffs = hist.Hist(*ref_coeffs.axes, corr_ax, vars_ax)
    # Corr = False is the uncorrected coeffs, corrected coeffs have the new A4
    # NOTE: the corrected coeffs are multiplied through by the sigmaUL correction, so that the 
    # new correction can be made as the ratio of the sum. To get the correct coeffs, this should
    # be divided back out
    corr_coeffs[...] = ref_coeffs.values(flow=True)[...,np.newaxis,np.newaxis]
    corr_coeffs[...,4.j,True,:] = target_a4_coeff.values()
    # Add back the helicity dimension and keep the variation dimension from the correction
    rescaled_coeffs = corr_coeffs[{"corr" : True}].values(flow=True)*sigmaUL_ratio.values(flow=True)[...,np.newaxis,:]
    corr_coeffs[...,True,:] = rescaled_coeffs

    corr_coeffs = set_corr_ratio_flow(corr_coeffs)
    return corr_coeffs

def make_angular_coeff(sigmai_hist, ul_hist):
    return hh.divideHists(sigmai_hist, ul_hist, cutoff=0.0001)

