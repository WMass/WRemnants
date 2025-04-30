import pickle

import hist
import lz4.frame
import numpy as np
import ROOT

from utilities import common
from wremnants.correctionsTensor_helper import makeCorrectionsTensor
from wums import logging

logger = logging.child_logger(__name__)

# TODO make this configurable to work in the Z case as well
# harmonize treatment of charge and mass axes between W and Z?

# the input files for this, and the corresponding gen axis definitions
# are produced from wremnants/scripts/histmakers/w_z_gen_dists.py


def makeQCDScaleByHelicityHelper(is_z=False, filename=None):
    if filename is None:
        #
        # filename = f"{common.data_dir}/angularCoefficients/w_z_coeffs.pkl.lz4" # Vpt binning based on common.ptV_binning
        filename = f"{common.data_dir}/angularCoefficients/w_z_coeffs_genVpt10quantiles.pkl.lz4"  # Vpt binning based on common.ptV_10quantiles_binning

    with lz4.frame.open(filename, "rb") as f:
        out = pickle.load(f)

    corrh = out["Z"] if is_z else out["W"]
    if np.count_nonzero(corrh[{"helicity": -1.0j}] == 0):
        logger.warning(
            "Zeros in sigma UL for the angular coefficients will give undefined behaviour!"
        )

    # histogram has to be without errors to load the tensor directly
    corrh_noerrs = hist.Hist(*corrh.axes, storage=hist.storage.Double())
    corrh_noerrs.values(flow=True)[...] = corrh.values(flow=True)

    return makeCorrectionsTensor(
        corrh_noerrs, ROOT.wrem.QCDScaleByHelicityCorrectionsHelper, tensor_rank=3
    )
