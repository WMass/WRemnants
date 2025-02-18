import ROOT

import narf
from utilities import common
from wums import logging

logger = logging.child_logger(__name__)

narf.clingutils.Declare('#include "pileup.hpp"')

data_dir = common.data_dir

eradict = {
    "2016B": "B",
    "2016C": "C",
    "2016D": "D",
    "2016E": "E",
    "2016F": "F",
    "2016FPostVFP": "F_postVFP",
    "2016G": "G",
    "2016H": "H",
    "2016PreVFP": "preVFP",
    "2016PostVFP": "postVFP",
    "2017": "2017",
    "2018": "2018",
}
fileDict = {
    "2016PostVFP": {
        "data": f"/pileupProfiles/pileupProfileData_2016Legacy_RunpostVFP_04June2021.root",
        "mc": "/pileupProfiles/MyMCPileupHistogram_2016Legacy_noGenWeights_preAndPostVFP.root",
    },
    "2017": {
        "data": f"/pileupProfiles/pileupHistogram-customJSON-UL2017-69200ub-99bins.root",
        "mc": "/pileupProfiles/MC2017PU.root",
    },
    "2018": {
        "data": f"/pileupProfiles/pileupHistogram-customJSON-UL2018-69200ub-99bins.root",
        "mc": "/pileupProfiles/MC2018PU.root",
    },
}


def make_pileup_helper(
    era=None, cropHighWeight=5.0, filename_data=None, filename_mc=None
):

    # following the logic from https://github.com/WMass/cmgtools-lite/blob/7488bc844ee7e7babf8376d9c7b074442b8879f0/WMass/python/plotter/pileupStuff/makePUweightPerEra.py

    if filename_data is None:
        filename_data = data_dir + fileDict[era]["data"]
    if filename_mc is None:
        filename_mc = data_dir + fileDict[era]["mc"]

    dataname = "pileup"

    fdata = ROOT.TFile.Open(filename_data)
    datahist = fdata.Get(dataname)
    datahist.SetDirectory(0)
    fdata.Close()

    # TODO get these numbers directly from the MC config instead
    fmc = ROOT.TFile.Open(filename_mc)
    logger.debug(f"Reading Data PU file-{filename_data}")
    logger.debug(f"Reading MC PU file-{filename_mc}")
    if era == "2016PostVFP":
        mchist0 = fmc.Get("Pileup_nTrueInt_Wmunu_preVFP")
        mchist1 = fmc.Get("Pileup_nTrueInt_Wmunu_postVFP")
        mchist = mchist0 + mchist1
        mchist.SetDirectory(0)
        fmc.Close()
    else:  # for 2017 & 2018
        mchist = fmc.Get("pileup")
        mchist.SetDirectory(0)
        fmc.Close()

    # normalize the histograms
    datahist.Scale(1.0 / datahist.Integral(0, datahist.GetNbinsX() + 1))
    mchist.Scale(1.0 / mchist.Integral(0, mchist.GetNbinsX() + 1))

    puweights = datahist / mchist

    for i in range(puweights.GetNbinsX() + 2):
        if mchist.GetBinContent(i) == 0.0:
            puweights.SetBinContent(i, 1.0)
        puweights.SetBinContent(i, min(puweights.GetBinContent(i), cropHighWeight))

    puweights.SetName(f"pileup_weights_{era}")
    puweights.SetTitle("")
    logger.debug("")
    logger.debug(f"PU weights for era {era}")
    logger.debug(
        [puweights.GetBinContent(i) for i in range(1, puweights.GetNbinsX() + 1)]
    )
    logger.debug("")
    logger.debug("")

    helper = ROOT.wrem.pileup_helper(puweights)

    return helper
