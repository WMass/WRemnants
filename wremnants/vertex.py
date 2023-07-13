import ROOT
import pathlib
import hist
import narf
import numpy as np
import boost_histogram as bh

narf.clingutils.Declare('#include "vertex.h"')

data_dir = f"{pathlib.Path(__file__).parent}/data/"

def make_vertex_helper(era = None,
                       filename = data_dir + "/vertex/vertexPileupWeights.root"):

    eradict = { "2016PreVFP" :  "BtoF",
                "2016PostVFP" : "GtoH",
    }
    
    fmc = ROOT.TFile.Open(filename)
    mchist = fmc.Get(f"weight_vertexZ_pileup_{eradict[era]}")
    mchist.SetDirectory(0)
    fmc.Close()

    ## for the histogram of preVFP, last PU bin in [40-45] is empty because of very small stat
    ## better to fill that bin with content of previous one, otherwise we are effectively cutting events based on PU
    if era == "2016PreVFP":
        lastPUbin = mchist.GetNbinsY()
        for ix in range(1, mchist.GetNbinsX() +1):
            mchist.SetBinContent(ix, lastPUbin, mchist.GetBinContent(ix, lastPUbin-1))
            mchist.SetBinError(  ix, lastPUbin, mchist.GetBinError(  ix, lastPUbin-1))
            
    helper = ROOT.wrem.vertex_helper(mchist)

    return helper
