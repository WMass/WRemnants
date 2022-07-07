import ROOT
import pathlib

ROOT.gInterpreter.AddIncludePath(f"{pathlib.Path(__file__).parent}/include/")

ROOT.gInterpreter.Declare('#include "muonCorr.h"')
ROOT.gInterpreter.Declare('#include "histoScaling.h"')
ROOT.gInterpreter.Declare('#include "utils.h"')
ROOT.gInterpreter.Declare('#include "csVariables.h"')
#ROOT.gInterpreter.Declare('#include "lowpu_recoil.h"')

from .datasets import datasets2016
from .datasets import datasetsLowPU

from .muon_prefiring import make_muon_prefiring_helpers
from .muon_efficiencies import make_muon_efficiency_helpers
from .qcdScaleByHelicity_helper import makeQCDScaleByHelicityHelper
from .pileup import make_pileup_helper
from .syst_tools import scale_helicity_hist_to_variations
from .theory_tools import axis_helicity, scale_tensor_axes, define_prefsr_vars, moments_to_angular_coeffs
from .muon_calibration import make_muon_calibration_helpers, define_corrected_muons

data_dir = f"{pathlib.Path(__file__).parent}/data/"
