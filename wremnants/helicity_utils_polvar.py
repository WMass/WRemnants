import ROOT
import pathlib
import hist
import narf
import narf.clingutils
import boost_histogram as bh
import uproot
import pathlib
import hist
import pickle
import lz4.frame
import numpy as np
import h5py
import hdf5plugin

from utilities import common, logging
from utilities import boostHistHelpers as hh
from utilities.io_tools import input_tools

logger = logging.child_logger(__name__)

data_dir = common.data_dir

def makehelicityWeightHelper_polvar(genVcharge=-1, fileTag="x0p40_y3p50_V6", filePath=".", noUL = False):
    if genVcharge not in [-1, 1, 0]:
        errmsg = f"genVcharge must be -1, 1 or 0, {genVcharge} was given"
        logger.error(errmsg)
        raise ValueError(errmsg)

    charges = { -1. : "minus", 1. : "plus", 0.: "Z"}
    # these inputs should be copied somewhere
    filenamePlus = f"{filePath}/fout_syst_wp_{fileTag}.root"
    filenameMinus = f"{filePath}/fout_syst_wm_{fileTag}.root"
    filenameZ = f"{filePath}/fout_syst_z_{fileTag}.root"
    filenames = {-1 : filenamePlus,
                  1 : filenameMinus,
                  0 : filenameZ}

    chargeTag = f"genChargeV{charges[genVcharge]}"

    logger.debug(f"Preparing helicity weights for theory agnostic analysis: gen V charge {charges[genVcharge]}")

    uptorange = 8 #up to A4

    down_up_axis = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "downUpVar")

    if noUL:
        folders = [f"A{i}" for i in range(uptorange)]
        axis_helicity_part = hist.axis.Integer(-1, len(folders), name="helicityPart", overflow=False, underflow=False)
    else:
        folders = ["UL"] + [f"A{i}" for i in range(uptorange)]
        axis_helicity_part = hist.axis.Integer(-1, len(folders)-1, name="helicityPart", overflow=False, underflow=False)

    hnom_hist_full = None
    hsys_hist_full = {fld: None for fld in folders}

    filename = filenames[genVcharge]
    fin = input_tools.safeOpenRootFile(filename)
    nSysts_coeff = {}
    for ifld, fld in enumerate(folders):
        f = fin.GetDirectory(fld)
        # start with nominal
        hnom = input_tools.safeGetRootObject(f, f"h_pdf_{fld}")
        hnom_hist = narf.root_to_hist(hnom, axis_names = ["qtOverQ", "absY"])
        # create and fill boost histogram
        if hnom_hist_full is None:
            axis_qtOverQ = hnom_hist.axes[0]
            axis_absY    = hnom_hist.axes[1]
            hnom_hist_full = hist.Hist(axis_qtOverQ, axis_absY, axis_helicity_part,
                                       name = f"hnom_hist_full_{chargeTag}",
                                       storage = hist.storage.Double())
        if noUL:
            nomIndex = ifld + 1
        else:
            nomIndex = ifld
        hnom_hist_full.values(flow=False)[:, :, nomIndex] = hnom_hist.values(flow=False)[:,:]
        # now the syst
        nSysts = 0
        for k in f.GetListOfKeys():
            name = k.GetName()
            if name.startswith("h_pdf") and "syst" in name and not "muF" in name and not "muR" in name and name.endswith("Up"):
                nSysts += 1
        # create and fill boost histogram
        if hsys_hist_full[fld] is None:
            axis_nsys = hist.axis.Integer(0, nSysts, name="nSyst", overflow=False, underflow=False)
            hsys_hist_full[fld] = hist.Hist(axis_qtOverQ, axis_absY, axis_nsys, down_up_axis,
                                            name = f"hsys_hist_full_{chargeTag}_{fld}",
                                            storage = hist.storage.Double())

        nSysts_coeff[ifld] = nSysts
        for isys in range(nSysts):
            hsysDown = input_tools.safeGetRootObject(f, f"h_pdf_{fld}_syst{isys}Down")
            hsysDown_hist = narf.root_to_hist(hsysDown, axis_names = ["qtOverQ", "absY"])
            hsys_hist_full[fld].values(flow=False)[:, :, isys, 0] = hsysDown_hist.values(flow=False)[:,:]
            hsysUp = input_tools.safeGetRootObject(f, f"h_pdf_{fld}_syst{isys}Up")
            hsysUp_hist = narf.root_to_hist(hsysUp, axis_names = ["qtOverQ", "absY"])
            hsys_hist_full[fld].values(flow=False)[:, :, isys, 1] = hsysUp_hist.values(flow=False)[:,:]

    fin.Close()

    ## TODO: the weights are actually unity in a region with large rapidity and qt/Q stored in the histogram
    
    # set overflow and underflow qt/Q-y bins equal to adjacent bins
    hnom_hist_full.values(flow=True)[0, ...] = hnom_hist_full.values(flow=True)[1, ...]
    hnom_hist_full.values(flow=True)[axis_qtOverQ.extent-1, ...] = hnom_hist_full.values(flow=True)[axis_qtOverQ.extent-2, ...]
    hnom_hist_full.values(flow=True)[:, 0, ...] = hnom_hist_full.values(flow=True)[:, 1, ...]
    hnom_hist_full.values(flow=True)[:, axis_absY.extent-1, ...] = hnom_hist_full.values(flow=True)[:, axis_absY.extent-2, ...]

    helpers = {}
    for ifld, fld in enumerate(folders):

        # set overflow and underflow qt/Q-y bins equal to adjacent bins
        hsys_hist_full[fld].values(flow=True)[0, ...] = hsys_hist_full[fld].values(flow=True)[1, ...]
        hsys_hist_full[fld].values(flow=True)[axis_qtOverQ.extent-1, ...] = hsys_hist_full[fld].values(flow=True)[axis_qtOverQ.extent-2, ...]
        hsys_hist_full[fld].values(flow=True)[:, 0, ...] = hsys_hist_full[fld].values(flow=True)[:, 1, ...]
        hsys_hist_full[fld].values(flow=True)[:, axis_absY.extent-1, ...] = hsys_hist_full[fld].values(flow=True)[:, axis_absY.extent-2, ...]

        nSysts = nSysts_coeff[ifld]
        logger.debug(f"Coefficient {fld} has {nSysts} variations for each Up/Down")

        ## call the helper
        ## copy nominal in case move destroys it
        hnom_hist_full_cp = hnom_hist_full.copy()
    
        hnom_hist_full_pyroot = narf.hist_to_pyroot_boost(hnom_hist_full_cp)
        hsys_hist_full_pyroot = narf.hist_to_pyroot_boost(hsys_hist_full[fld])
        if noUL:
            varIndex = ifld + 1
        else:
            varIndex = ifld

        helper = ROOT.wrem.WeightByHelicityHelper_polvar[genVcharge,
                                                         varIndex,
                                                         nSysts,
                                                         type(hsys_hist_full_pyroot),
                                                         type(hnom_hist_full_pyroot)](ROOT.std.move(hsys_hist_full_pyroot),
                                                                                      ROOT.std.move(hnom_hist_full_pyroot))
        ## define axis for syst variations
        axis_nsyst = hist.axis.Integer(0, nSysts, name="nPolVarSyst", overflow=False, underflow=False)
        helper.tensor_axes = [axis_nsyst, down_up_axis]
        helpers[fld] = helper
        
    return {k : helpers[k] for k in helpers.keys()}
