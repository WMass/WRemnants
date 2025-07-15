import hist
import ROOT

import narf.clingutils
from utilities import common

narf.clingutils.Declare('#include "muon_prefiring.hpp"')

data_dir = common.data_dir


def make_muon_prefiring_helpers(
    filename=data_dir + "/muonSF/L1MuonPrefiringParametriations_histograms.root",
    era=None,
):

    if not (era is None or "2016" in era):
        return None, None, None

    fin = ROOT.TFile.Open(filename)

    eradict = {
        "2016H": "2016H",
        # "2016PreVFP", "2016preVFP",
        # BG should be like preVFP, but more data was used to derive corrections
        "2016PreVFP": "2016BG",
        "2016PostVFP": "2016postVFP",
        "2017": "2016postVFP",  # this is just for creating the helper for 17/18. Need a better solution later.
        "2018": "2016postVFP",
    }

    eratag = eradict[era]

    hparameters = fin.Get(f"L1prefiring_muonparam_{eratag}")

    netabins = hparameters.GetXaxis().GetNbins()

    hparameters_hotspot = fin.Get("L1prefiring_muonparam_2016_hotspot")

    helper = ROOT.wrem.muon_prefiring_helper(hparameters, hparameters_hotspot)

    # histograms are copied into the helper class so we can close the file without detaching them
    fin.Close()

    helper_stat = ROOT.wrem.muon_prefiring_helper_stat[netabins](helper)
    helper_syst = ROOT.wrem.muon_prefiring_helper_syst(helper)

    return helper, helper_stat, helper_syst


@ROOT.pythonization("muon_prefiring_helper_stat<", ns="wrem", is_prefix=True)
def pythonize_rdataframe(klass):
    # add axes corresponding to the tensor dimensions
    klass.tensor_axes = (
        hist.axis.Integer(
            0,
            klass.NVar,
            underflow=False,
            overflow=False,
            name="etaPhiRegion",
            label="muon prefiring eta-phi regions",
        ),
        common.down_up_axis,
    )
