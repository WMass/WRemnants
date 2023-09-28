
import sys,array,math,os,copy,fnmatch
from collections import OrderedDict

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

import plotter
import functions

import lz4.frame
import pickle
import narf
import hist
import numpy as np

from wremnants.datasets.datagroups import Datagroups



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
    hName = histCfg['name']

    label = "%s_%s" % (hName, procName)
    groups.setHists(hName, "", label=label, procsToRead=[procName], selectSignal=False)
    bhist = groups.groups[procName][label]
    
    if "charge" in histCfg:
        charge = histCfg['charge']
        s = hist.tag.Slicer()
        if charge == "combined": bhist = bhist[{"charge" : s[::hist.sum]}]
        elif charge == "plus": bhist = bhist[{"charge" : bhist.axes["charge"].index(+1)}]
        elif charge == "minus": bhist = bhist[{"charge" : bhist.axes["charge"].index(-1)}]
    #print(bhist)
    
    rhist = narf.hist_to_root(bhist)
    rhist = functions.Rebin(rhist, rebin)
    rhist.SetName(label)
    rhist = doOverlow(rhist)
   
    # print("Get histogram %s, yield=%.2f" % (label, rhist.Integral()))
    return rhist
 


	
if __name__ == "__main__":

    cuts = ["All", "#geq 0 muons", "Loose + dxdy < 0.05", "|#eta| < 2.4", "ID", "Lepton iso < 0.15", "p_{T} > 25 GeV", "== 1 muon", "Electron veto", "Trigger", "Trigger match", "MET filters", "m_{T} > 40 GeV"]

    flavor = "mu"
    met = "RawPFMET" # DeepMETReso RawPFMET
    charge = "combined" # combined plus minus
    MC_SF = 1.0
    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/W%s/plots_%s_%s/" % (flavor, charge, met)

    procs = ['EWK', 'TTbar', 'WJetsToMuNu'] 
    groups = make_datagroups_lowPU("lowPU_%s_%s_cutFlow.pkl.lz4" % (flavor, met), flavor=flavor)
    
    cutFlowHists = {}
    st = ROOT.THStack()
    st.SetName("stack")
    h_bkg = None
    
    leg = ROOT.TLegend(.20, 0.90-(len(procs))*0.05, .55, .90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)
    
    for proc in procs:
    
        hist = ROOT.TH1D("h_%s" % proc, "", len(cuts), 0, len(cuts))
        hist.SetFillColor(ROOT.TColor.GetColor(groups.groups[proc]['color']))
        hist.SetLineColor(ROOT.kBlack)
        hist.SetLineWidth(1)
        hist.SetLineStyle(1)
        
        
        for i,cut in enumerate(cuts):
            h_ = parseProc(groups, {"name": "cutflow_%d" % i, "axis": "cutFlow" }, proc)
            hist.SetBinContent(i+1, h_.GetBinContent(1))
            print(proc, i, h_.GetBinContent(1))
            
        st.Add(hist)
        leg.AddEntry(hist, groups.groups[proc]['label'], "F")
        if h_bkg == None: h_bkg = hist.Clone("h_bkg")
        else: h_bkg.Add(hist)
        
    h_bkg.SetLineColor(ROOT.kBlack)
    h_bkg.SetFillColor(0)
    h_bkg.SetLineWidth(2)
    
    
    # default cfg
    cfg = {

        'logy'              : True,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : len(cuts),
        'ymin'              : 1e5,
        'ymax'              : 1e8, # 3e6
            
        'xtitle'            : "",
        'ytitle'            : "Events",
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    }
    

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy(len(cuts))
    dummy.GetXaxis().SetLabelSize(0.7*dummy.GetXaxis().GetLabelSize())
    dummy.GetXaxis().SetLabelOffset(1.3*dummy.GetXaxis().GetLabelOffset())
    for i,cut in enumerate(cuts): dummy.GetXaxis().SetBinLabel(i+1, cut)
    dummy.GetXaxis().LabelsOption("u")
    
    
    canvas.SetGrid()
    dummy.Draw("HIST")
    st.Draw("HIST SAME")
    h_bkg.Draw("HIST SAME")
    leg.Draw("SAME")
    
    plotter.aux(canvas)  
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()  
    


    canvas.SaveAs("%s/cutFlow.png" % outDir)
    canvas.SaveAs("%s/cutFlow.pdf" % outDir)
    canvas.Close()