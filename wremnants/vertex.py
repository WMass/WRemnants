import ROOT

import narf
from utilities import common
from wums import logging

narf.clingutils.Declare('#include "vertex.hpp"')

logger = logging.child_logger(__name__)

data_dir = common.data_dir


def make_vertex_helper(era=None, filename=None):

    eradict = {
        "2016PreVFP": "BtoF",
        "2016PostVFP": "GtoH",
        "2017": "2017",
        "2018": "2018",
    }
    filedict = {
        "2016PostVFP": "/vertex/vertexPileupWeights.root",
        "2017": "/vertex/vertexPileupWeights_2017.root",
        "2018": "/vertex/vertexPileupWeights_2018.root",
    }

    if filename is None:
        filename = data_dir + filedict[era]
        print("Vertex weight fname:", filename)
    logger.debug(
        f"vertex.py: will read weight_vertexZ_pileup_{eradict[era]} from {filename}"
    )
    fmc = ROOT.TFile.Open(filename)
    mchist = fmc.Get(f"weight_vertexZ_pileup_{eradict[era]}")
    mchist.SetDirectory(0)
    fmc.Close()

    ## for the histogram of preVFP, last PU bin in [40-45] is empty because of very small stat
    ## better to fill that bin with content of previous one, otherwise we are effectively cutting events based on PU
    if era == "2016PreVFP":
        lastPUbin = mchist.GetNbinsY()
        for ix in range(1, mchist.GetNbinsX() + 1):
            mchist.SetBinContent(ix, lastPUbin, mchist.GetBinContent(ix, lastPUbin - 1))
            mchist.SetBinError(ix, lastPUbin, mchist.GetBinError(ix, lastPUbin - 1))

    helper = ROOT.wrem.vertex_helper(mchist)

    return helper
