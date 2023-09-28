import ROOT
import narf
import pathlib

ROOT.gInterpreter.AddIncludePath(f"{pathlib.Path(__file__).parent}/include/")

narf.clingutils.Declare('#include "muonCorr.h"')
narf.clingutils.Declare('#include "histoScaling.h"')
narf.clingutils.Declare('#include "histHelpers.h"')
narf.clingutils.Declare('#include "utils.h"')
narf.clingutils.Declare('#include "csVariables.h"')
narf.clingutils.Declare('#include "EtaPtCorrelatedEfficiency.h"')

from .muon_prefiring import make_muon_prefiring_helpers
from .muon_efficiencies_smooth import make_muon_efficiency_helpers_smooth
from .muon_efficiencies_binned import make_muon_efficiency_helpers_binned
from .muon_efficiencies_binned_vqt import make_muon_efficiency_helpers_binned_vqt
from .muon_efficiencies_binned_vqt_integrated import make_muon_efficiency_helpers_binned_vqt_integrated
from .muon_efficiencies_binned_vqt_real import make_muon_efficiency_helpers_binned_vqt_real
from .qcdScaleByHelicity_helper import makeQCDScaleByHelicityHelper
from .pileup import make_pileup_helper
from .vertex import make_vertex_helper
from .syst_tools import scale_helicity_hist_to_variations
from .theory_tools import axis_helicity, scale_tensor_axes, define_prefsr_vars, moments_to_angular_coeffs
from .muon_calibration import *
from .helicity_utils import *

data_dir = f"{pathlib.Path(__file__).parent}/../wremnants-data/data/"
