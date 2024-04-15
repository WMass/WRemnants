
import sys,array,math,os,copy,shutil,decimal
import json

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

import narf
import hist
import numpy as np


def loadJSON(jsIn):
    with open(jsIn) as f: jsDict = json.load(f)
    return jsDict

def writeJSON(jsOut, outDict):
    with open(jsOut, "w") as outfile: json.dump(outDict, outfile, indent=4)

def parseBoostHist(groups, histCfg, procName, rebin=1):

    axis = histCfg['axis']
    hName = histCfg['name']
    
    label = "%s_%s" % (hName, procName)
    groups.setHists(hName, "", label=label, procsToRead=[procName], selectSignal=False)
    bhist = groups.groups[procName][label]
    rhist = narf.hist_to_root(bhist)
    rhist = Rebin(rhist, rebin)
    rhist.SetName(label)
    rhist = doOverlow(rhist)

    print("Get histogram %s, yield=%.2f" % (label, rhist.Integral()))
    return rhist
 

def prepareDir(outDir, remove=True):
    if os.path.exists(outDir) and os.path.isdir(outDir) and remove: shutil.rmtree(outDir)
    os.system("mkdir -p %s" % outDir)

def doOverlow(h):
    n = h.GetNbinsX()
    h.SetBinContent(1, h.GetBinContent(0) + h.GetBinContent(1))
    h.SetBinContent(n, h.GetBinContent(n+1) + h.GetBinContent(n))
    h.SetBinError(1, math.hypot(h.GetBinError(0), h.GetBinError(1)))
    h.SetBinError(n, math.hypot(h.GetBinError(n+1), h.GetBinError(n)))
    h.SetBinContent(0, 0)
    h.SetBinContent(n+1, 0)
    h.SetBinContent(0, 0)
    h.SetBinContent(n+1, 0)
    return h

def parseProc(groups, histCfg, procName, syst="", rebin=1):

    axis = histCfg['axis']
    hNames = histCfg['name'].split(",")

    
    bhist = None
    for hName in hNames:
        
        bhist = groups.readProc(hName, procName, axis=axis)
        if bhist == None: continue
        label = "%s_%s" % (hName, procName)
        break
        
    print(bhist)
    rhist = narf.hist_to_root(bhist)
    rhist.Rebin(rebin)
    rhist.SetName(label)
    rhist = doOverlow(rhist)

    print("Get histogram %s, yield=%d" % (label, rhist.Integral()))
    return rhist

def rebin(h, newbins, binWidth=True):
    if isinstance(newbins, int):
        h.Rebin(newbins)
        if binWidth: h.Scale(1, "width")
        return h
    else:
        mybins = array.array('d', newbins)
        h1 = h.Rebin(len(mybins)-1, h.GetName(), mybins)
        if binWidth: h1.Scale(1, "width")
        return h1


def drange(x, y, jump):
    while x < y:
        yield float(x)
        #x += decimal.Decimal(jump)
        x += jump

def readBoostHist(groups, hName, procs, charge="combined", boost=False, integrateAxes=[], abcd=True):

    if abcd:
        groups.set_histselectors(procs, hName, mode="simple", smoothen=False, simultaneousABCD=False, integrate_x=False if "mt" in hName else True)
    groups.loadHistsForDatagroups(hName, syst="", procsToRead=procs)
    bhist = sum([groups.groups[p].hists[hName] for p in procs])

    s = hist.tag.Slicer()
    axes = [ax.name for ax in bhist.axes]
    if "eta" in axes and "pt" in axes:
        bhist = bhist[{"eta" : s[::hist.sum], "pt" : s[::hist.sum]}]
    
    for iAx in integrateAxes:
        bhist = bhist[{iAx : s[::hist.sum]}]

    if "passMT" in bhist.axes.name:
        bhist = bhist[{"passMT": True}]

    if "charge" in bhist.axes.name:
        if charge and charge == "combined": bhist = bhist[{"charge" : s[::hist.sum]}]
        elif charge and charge == "plus": bhist = bhist[{"charge" : bhist.axes["charge"].index(+1)}]
        elif charge and charge == "minus": bhist = bhist[{"charge" : bhist.axes["charge"].index(-1)}]

    if boost:
        return bhist
    rhist = narf.hist_to_root(bhist)
    rhist.SetName(f"{hName}")
    return rhist


def getLumiLabel(groups):
    if groups.lumi < 1:
        return "{:.1f} pb^{{#minus1}} (13 TeV)".format(1000.*groups.lumi)
    else:
        return "{:.1f} fb^{{#minus1}} (13 TeV)".format(groups.lumi)

def getMinMaxRange(h, xMin, xMax):
    yMin, yMax = 1e9, -1e9
    for i in range(0, h.GetNbinsX()+1):
        x_ = h.GetBinLowEdge(i)
        if x_ < xMin or x_ > xMax:
            continue
        c = h.GetBinContent(i)
        if c < yMin:
            yMin = c
        if c > yMax:
            yMax = c
    return yMin, yMax


def get_meta(groups):
    met = groups.getMetaInfo()["args"].get("met", None)
    analysis = "lowPU" if "lowpu" in groups.mode else "highPU"
    flavor = groups.flavor
    if flavor == None:
        flavor = "mu" if groups.mode=="wmass" else "mumu"
    return met, analysis, flavor