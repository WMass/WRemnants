
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
    #os.system("cp /eos/user/j/jaeyserm/www/wmass/index.php %s" % outDir)

    
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
    
    

def Rebin(h, newbins, binWidth=True):

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
        
        
def readBoostHistProc(groups, hName, procs, charge=None):

    groups.setNominalName(hName)
    groups.loadHistsForDatagroups(hName, syst="")
    bhist = sum([groups.groups[p].hists[hName] for p in procs])

    axes = [ax.name for ax in bhist.axes]
    if "charge" in axes:
        s = hist.tag.Slicer()
        if charge and charge == "combined": bhist = bhist[{"charge" : s[::hist.sum]}]
        elif charge and charge == "plus": bhist = bhist[{"charge" : bhist.axes["charge"].index(+1)}]
        elif charge and charge == "minus": bhist = bhist[{"charge" : bhist.axes["charge"].index(-1)}]

    return bhist 