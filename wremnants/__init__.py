import pathlib

import ROOT

import narf

ROOT.gInterpreter.AddIncludePath(f"{pathlib.Path(__file__).parent}/include/")

narf.clingutils.Load("libHist")

narf.clingutils.Declare('#include "muonCorr.hpp"')
narf.clingutils.Declare('#include "histoScaling.hpp"')
narf.clingutils.Declare('#include "histHelpers.hpp"')
narf.clingutils.Declare('#include "utils.hpp"')
narf.clingutils.Declare('#include "csVariables.hpp"')
narf.clingutils.Declare('#include "EtaPtCorrelatedEfficiency.hpp"')
narf.clingutils.Declare('#include "theoryTools.hpp"')
narf.clingutils.Declare('#include "syst_helicity_utils_polvar.hpp"')
