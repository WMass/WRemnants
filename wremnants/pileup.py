import ROOT
import pathlib
import hist
import narf
import numpy as np
import boost_histogram as bh

ROOT.gInterpreter.Declare('#include "pileup.h"')

data_dir = f"{pathlib.Path(__file__).parent}/data/"

eradict = { "2016B" : "B",
            "2016C" : "C",
            "2016D" : "D",
            "2016E" : "E",
            "2016F" : "F",
            "2016FPostVFP" : "F_postVFP",
            "2016G" : "G",
            "2016H" : "H",
            "2016PreVFP" : "preVFP",
            "2016PostVFP" : "postVFP",
            }

def make_pileup_helper(era = None, cropHighWeight = 5.,
                       filename_data = None,
                       filename_mc = data_dir + "/pileupProfiles/MyMCPileupHistogram_2016Legacy_noGenWeights_preAndPostVFP.root"):

    # following the logic from https://github.com/WMass/cmgtools-lite/blob/7488bc844ee7e7babf8376d9c7b074442b8879f0/WMass/python/plotter/pileupStuff/makePUweightPerEra.py

    if filename_data is None:
        filename_data = data_dir + f"/pileupProfiles/pileupProfileData_2016Legacy_Run{eradict[era]}_04June2021.root"

    dataname = "pileup"

    fdata = ROOT.TFile.Open(filename_data)
    datahist = fdata.Get(dataname)
    datahist.SetDirectory(0)
    fdata.Close()

    # TODO get these numbers directly from the MC config instead
    fmc = ROOT.TFile.Open(filename_mc)
    mchist0 = fmc.Get("Pileup_nTrueInt_Wmunu_preVFP")
    mchist1 = fmc.Get("Pileup_nTrueInt_Wmunu_postVFP")

    mchist = mchist0 + mchist1
    mchist.SetDirectory(0)
    fmc.Close()

    # normalize the histograms
    datahist.Scale(1./datahist.Integral(0, datahist.GetNbinsX() + 1))
    mchist.Scale(1./mchist.Integral(0, mchist.GetNbinsX() + 1))

    puweights = datahist/mchist

    for i in range(puweights.GetNbinsX() + 2):
        if mchist.GetBinContent(i) == 0.:
            puweights.SetBinContent(i, 1.)
        mchist.SetBinContent(i, min(puweights.GetBinContent(i), cropHighWeight))

    puweights.SetName(f"pileup_weights_{era}")
    puweights.SetTitle("")

    helper = ROOT.wrem.pileup_helper(puweights)

    return helper
