import hist
import numpy as np
from utilities.input_tools import safeOpenRootFile, safeGetRootObject
from utilities import boostHistHelpers as hh
from utilities import common, logging
import narf
import ROOT

logger = logging.child_logger(__name__)

hist_map = {
    "eta_pt" : "nominal",
    "eta" : "nominal",
    "pt" : "nominal",
    "mll" : "nominal",
    "ptll" : "nominal",
    "ptll_mll" : "nominal",
}

def fakeHistABCD(h, thresholdMT=40.0, fakerate_integration_axes=[], axis_name_mt="mt", integrateLowMT=True, integrateHighMT=False):
    # integrateMT=False keeps the mT axis in the returned histogram (can be used to have fakes vs mT)

    nameMT, failMT, passMT = get_mt_selection(h, thresholdMT, axis_name_mt, integrateLowMT, integrateHighMT)

    if any(a in h.axes.name for a in fakerate_integration_axes):
        fakerate_axes = [n for n in h.axes.name if n not in [*fakerate_integration_axes, common.passIsoName, nameMT]]
        hPassIsoFailMT = h[{**common.passIso, nameMT: failMT}].project(*fakerate_axes)
        hFailIsoFailMT = h[{**common.failIso, nameMT: failMT}].project(*fakerate_axes)
    else:
        hPassIsoFailMT = h[{**common.passIso, nameMT: failMT}]
        hFailIsoFailMT = h[{**common.failIso, nameMT: failMT}]

    hFRF = hh.divideHists(hPassIsoFailMT, hFailIsoFailMT, cutoff=1, createNew=True)   

    return hh.multiplyHists(hFRF, h[{**common.failIso, nameMT: passMT}])
    
def fakeHistIsoRegion(h, scale=1.):
    #return h[{"iso" : 0.3j, "mt" : hist.rebin(10)}]*scale
    return h[{"iso" : 4}]*scale

def fakeHistIsoRegionIntGen(h, scale=1.):
    print([ax.name for ax in h.axes])
    if not "qTgen" in [ax.name for ax in h.axes]:
        return h[{"iso" : 4}]*scale
    s = hist.tag.Slicer()
    print("Slicing")
    return h[{"iso" : 0, "qTgen" : s[::hist.sum]}]

def signalHistWmass(h, thresholdMT=40.0, charge=None, passIso=common.passIso, passMT=True, axis_name_mt="mt", integrateLowMT=True, integrateHighMT=False, genBin=None):
    if genBin != None:
        h = h[{"recoil_gen" : genBin}]

    nameMT, failMT, passMT = get_mt_selection(h, thresholdMT, axis_name_mt, integrateLowMT, integrateHighMT)

    sel = {**passIso, nameMT: passMT}
    if charge in [-1, 1]:
        sel.update({"charge" : -1j if charge < 0 else 1j})

    # remove ax slice if the ax does not exist
    for key in sel.copy().keys():
        if not key in [ax.name for ax in h.axes]: 
            del sel[key]

    return h[sel]

# the following are utility wrapper functions for signalHistWmass with proper region selection
def histWmass_failMT_passIso(h, thresholdMT=40.0, charge=None):
    return signalHistWmass(h, thresholdMT, charge, common.passIso, common.failMT)

def histWmass_failMT_failIso(h, thresholdMT=40.0, charge=None):
    return signalHistWmass(h, thresholdMT, charge, common.failIso, common.failMT)

def histWmass_passMT_failIso(h, thresholdMT=40.0, charge=None):
    return signalHistWmass(h, thresholdMT, charge, common.failIso, common.passMT)

def histWmass_passMT_passIso(h, thresholdMT=40.0, charge=None):
    return signalHistWmass(h, thresholdMT, charge, common.passIso, common.passMT)

# TODO: Not all hists are made with these axes
def signalHistLowPileupW(h):
    if not "qTgen" in [ax.name for ax in h.axes]:
        return h[{"iso" : 0}]
    s = hist.tag.Slicer()
    return h[{"iso" : 0, "qTgen" : s[::hist.sum]}]
    
def signalHistLowPileupZ(h):
    return h

def get_mt_selection(h, thresholdMT=40.0, axis_name_mt="mt", integrateLowMT=True, integrateHighMT=False):
    if axis_name_mt in h.axes.name:
        s = hist.tag.Slicer()
        high = h.axes[axis_name_mt].index(thresholdMT)
        failMT = s[:high:hist.sum] if integrateLowMT else s[:high:]
        passMT = s[high:hist.sum] if integrateHighMT else s[high:]
        nameMT = axis_name_mt
    else:
        failMT = 0
        passMT = 1
        nameMT = common.passMTName

    return nameMT, failMT, passMT

def unrolledHist(h, obs=["pt", "eta"]):
    hproj = h.project(*obs)
    bins = np.product(hproj.axes.size)
    newh = hist.Hist(hist.axis.Integer(0, bins), storage=hproj._storage_type())
    newh[...] = np.ravel(hproj)
    return newh

def applyCorrection(h, scale=1.0, offsetCorr=0.0, corrFile=None, corrHist=None, createNew=False):
    # originally intended to apply a correction differential in eta-pt
    # corrHist is a TH3 with eta-pt-charge
    # scale is just to apply an additional constant scaling (or only that if ever needed) before the correction from corrHist
    # offsetCorr is for utility, to add to the correction from the file, e.g. if the histogram is a scaling of x (e.g. 5%) one has to multiply the input histogram h by 1+x, so offset should be 1
    boost_corr = None
    if corrFile and corrHist:
        ## TDirectory.TContext() should restore the ROOT current directory to whatever it was before a new ROOT file was opened
        ## but it doesn't work at the moment, apparently the class miss the __enter__ member and the usage with "with" fails
        #with ROOT.TDirectory.TContext():
        f = safeOpenRootFile(corrFile, mode="READ")
        corr = safeGetRootObject(f, corrHist, detach=True)
        if offsetCorr:
            offsetHist = corr.Clone("offsetHist")
            ROOT.wrem.initializeRootHistogram(offsetHist, offsetCorr)
            corr.Add(offsetHist)
        f.Close()
        boost_corr = narf.root_to_hist(corr)
    # note: in fact hh.multiplyHists already creates a new histogram
    hnew = hh.scale(h, scale, createNew)
    if boost_corr:
        hnew = hh.multiplyHists(hnew, boost_corr)

    return hnew
