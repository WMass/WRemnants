import uproot
import ROOT
import pathlib
import hist
import narf
import pickle
import lz4.frame
from .correctionsTensor_helper import makeCorrectionsTensor
from .theory_tools import scale_tensor_axes

data_dir = f"{pathlib.Path(__file__).parent}/data/"

# TODO make this configurable to work in the Z case as well
# harmonize treatment of charge and mass axes between W and Z?

# the input files for this, and the corresponding gen axis definitions
# are produced from wremnants/scripts/histmakers/w_z_gen_dists.py

def makeQCDScaleByHelicityHelper(is_w_like = False, filename=None):
    if filename is None:
        if is_w_like:
            filename = f"{data_dir}/angularCoefficients/z_coeffs.pkl.lz4"
        else:
            filename = f"{data_dir}/angularCoefficients/w_coeffs.pkl.lz4"

    with lz4.frame.open(filename, "rb") as f:
        corrh = pickle.load(f)

    # histogram has to be without errors to load the tensor directly
    corrh_noerrs = hist.Hist(*corrh.axes, storage=hist.storage.Double())
    corrh_noerrs.values(flow=True)[...] = corrh.values(flow=True)

    return makeCorrectionsTensor(corrh_noerrs, ROOT.wrem.QCDScaleByHelicityCorrectionsHelper, tensor_rank=3)
