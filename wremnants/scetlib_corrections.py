import ROOT
import pathlib
import hist
import narf
import numpy as np
import lz4.frame
import pickle
from .correctionsTensor_helper import makeCorrectionsTensor
from wremnants import boostHistHelpers as hh
from wremnants.common import data_dir

def make_corr_helper_fromnp(filename=f"{data_dir}/N3LLCorrections/inclusive_{{process}}_pT.npz", isW=True):
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

    print("Shape is", corrh.shape)

    return makeCorrectionsTensor(corrh, ROOT.wrem.TensorCorrectionsHelper, tensor_rank=1)

def make_corr_helper(filename=f"{data_dir}/N3LLCorrections/inclusive_pT_y_m.pkl.lz4", isW=True):
    print("Filename", filename)
    with lz4.frame.open(filename) as f:
        corr = pickle.load(f)
        corrh = corr['W' if isW else 'Z']["inclusive_pT_y_m_ratio"]

    return makeCorrectionsTensor(corrh, ROOT.wrem.TensorCorrectionsHelper, tensor_rank=1)

def make_scetlib_minnlo_corr(minnlo_hist, scetlib_hist):
    minnlo_rebin = minnlo_hist
    for minnlo_ax, scetlib_ax in zip(minnlo_hist.axes, scetlib_hist.axes[:-1]):
        minnlo_rebin = hh.rebinHist(minnlo_rebin, minnlo_ax.name, scetlib_ax.edges)

    corrh = hh.divideHists(scetlib_hist, minnlo_rebin)
    corrh[hist.underflow,...] = np.ones_like(corrh[0,...])
    corrh[hist.overflow,...] = np.ones_like(corrh[0,...])
    corrh[:,hist.underflow,...] = np.ones_like(corrh[:,0,...])
    corrh[:,hist.overflow,...] = np.ones_like(corrh[:,0,...])
    corrh[:,:,hist.overflow,...] = np.ones_like(corrh[:,:,0,...])
    return corrh

def read_scetlib_hist(path, pt_axis=None, nonsing="auto", flip_y_sign=False, charge=None):
    f = np.load(path, allow_pickle=True)
    var_axis = hist.axis.Integer(f["bins"][0][0], f["bins"][0][-1], name="vars", flow=False)
    mass_axis = hist.axis.Variable(f["bins"][1], name="mass")
    y_axis = hist.axis.Variable(f["bins"][2], name="y")
    
    if not pt_axis:
        pt_axis = hist.axis.Variable(f["bins"][3], name="pt")

    h = f["hist"]
    storage = hist.storage.Double()
    axes = [mass_axis,y_axis,pt_axis,var_axis]
    varax_idx = -1 
    vals = np.moveaxis(h, 0, varax_idx)

    if "hist_err" in f:
        err = f["hist_err"]
        storage = hist.storage.Weight()
        vals = np.stack((vals, np.moveaxis(err, 0, varax_idx)), axis=-1)

    if charge is not None:
        charge_args = (2, -2., 2.) if charge != 0 else (1, 0, 1) 
        charge_axis = hist.axis.Regular(*charge_args, flow=False, name = "charge")
        axes.insert(-1, charge_axis)
    
    scetlibh = hist.Hist(*axes, storage=storage)
    if charge is None:
        scetlibh[...] = vals
    else:
        scetlibh[...,charge_axis.index(charge),:] = vals

    if nonsing:
        if nonsing == "auto":
            nonsing = path.replace(".npz", "_nons.npz")
        nonsing = read_scetlib_hist(nonsing, pt_axis, False, 
                        flip_y_sign=flip_y_sign, charge=charge)
        scetlibh = scetlibh + nonsing
    
    if flip_y_sign:
        mid = y_axis.index(0)
        s = hist.tag.Slicer()
        scetlibh[{"y" : s[mid:]}] = scetlibh[{"y" : s[mid:]}].view()*-1

    return scetlibh 

