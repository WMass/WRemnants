
import sys,array,math,os,copy,fnmatch
from collections import OrderedDict

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

sys.path.insert(0, "scripts/lowPU/")
sys.path.insert(0, "scripts/lowPU/recoil/")
import functions
import plotter

import lz4.frame
import pickle
import narf
import numpy as np

from wremnants.datasets.datagroups import Datagroups

def doPlot(tag, xMin=-100, xMax=100, yMin=1, yMax=-1, label="", xLabel="", yRatio = 1.3, rebin=1, logPos=[.20, 0.65, .5, .82]):

    if tag == "nominal":
    
        nominal_cols = ["trigMuons_eta0", "trigMuons_pt0", "trigMuons_charge0"]
        
        bhist = functions.readBoostHistProc(groups, tag, ["Zmumu"])
        bhist_qTrw = functions.readBoostHistProc(groups, tag, ["Zmumu"])
        
        bhist = bhist[{"eta": sum, "charge": sum}]
        bhist_qTrw = bhist_qTrw[{"eta": sum, "charge": sum}]
        
        hist = narf.hist_to_root(bhist)
        hist_qTrw = narf.hist_to_root(bhist_qTrw)
        
        tag = "lep_pT"

    else:
        hist = narf.hist_to_root(functions.readBoostHistProc(groups, tag, ["Zmumu"]))
        hist_qTrw = narf.hist_to_root(functions.readBoostHistProc(groups, tag+"_qTrw", ["Zmumu"]))
    
    hist = functions.Rebin(hist, rebin)
    hist_qTrw = functions.Rebin(hist_qTrw, rebin)
    
    cfg = {

        'logy'              : True,
        'logx'              : False,
    
        'xmin'              : xMin,
        'xmax'              : xMax,
        'ymin'              : yMin,
        'ymax'              : yMax if yMax > 0 else max([hist.GetMaximum(), hist_qTrw.GetMaximum()])    ,
        
        'xtitle'            : xLabel,
        'ytitle'            : "Events",
        
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
        
        'ratiofraction'     : 0.3,
        'ytitleR'           : "Ratio",
        'yminR'             : 1-(yRatio-1),
        'ymaxR'             : yRatio,
    }   
    
    leg = ROOT.TLegend(logPos[0], logPos[1], logPos[2], logPos[3])
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.040)
    leg.SetHeader(label)

    hist.SetLineColor(ROOT.kRed)
    hist.SetFillColor(0)
    hist.SetLineWidth(2)
    
    hist_qTrw.SetLineColor(ROOT.kBlue)
    hist_qTrw.SetFillColor(0)
    hist_qTrw.SetLineWidth(2)   
    
    leg.AddEntry(hist, "Unweighted", "L")
    leg.AddEntry(hist_qTrw, "q_{T} reweighted", "L")

    h_ratio = hist.Clone("h_uncorr_ratio")
    h_ratio.Divide(hist_qTrw)
    h_ratio.SetMarkerColor(ROOT.kBlack)
    h_ratio.SetLineColor(ROOT.kBlack)
    h_ratio.SetLineWidth(2)


    
    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio(line=1)
    canvas.cd()
    padT.Draw()
    padT.cd()
    padT.SetGrid()
    padT.SetTickx()
    padT.SetTicky()
    dummyT.Draw("HIST")
    
    hist.Draw("HIST SAME")
    hist_qTrw.Draw("HIST SAME")
    leg.Draw()
    
    padT.RedrawAxis()
    padT.RedrawAxis("G")
    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.04)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.2, 0.85, procLabel)
    latex.DrawLatex(0.2, 0.80, metLabel)
    
    plotter.auxRatio()
       

    canvas.cd()
    padB.Draw()
    padB.cd()
    padB.SetGrid()
    padB.SetTickx()
    padB.SetTicky()
    
    dummyB.Draw("HIST")
    dummyL.Draw("SAME")
    
    h_ratio.Draw("HIST SAME E")

    padB.RedrawAxis()
    padB.RedrawAxis("G")
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s.png" % (outDir, tag))
    canvas.SaveAs("%s/%s.pdf" % (outDir, tag))

    dummyB.Delete()
    dummyT.Delete()
    padT.Delete()
    padB.Delete()
    


    
if __name__ == "__main__":

    # compare recoil, MET, mT, lep pT, ... for MC unreweighted and qT reweighted to data

    if False: # lowPU
        met = "RawPFMET" # RawPFMET DeepMETReso
        flavor = "mumu" # mumu, ee
        

        procLabel = "DY #rightarrow #mu^{+}#mu^{#minus}"
        metLabel = "RawPFMET"

        outDir = "/eos/user/j/jaeyserm/www/recoil/biasStudy/lowPU_zmumu"
        functions.prepareDir(outDir, False)

        groups = Datagroups("lowPU_%s_%s.pkl.lz4" % (flavor, met), flavor=flavor)
        
        recoil_qTbins = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 70, 5)) + list(range(70, 120, 10)) + [120, 150, 300]
        
        doPlot("qT", xMin=0, xMax=100, yMin=1e1, yMax= 1e5, label="", xLabel="q_{T} (GeV)", yRatio=1.15, rebin=recoil_qTbins, logPos=[.60, 0.65, .9, .82])
        
        doPlot("mT_corr_xy", xMin=0, xMax=120, yMin=1e1, yMax= 1e4, label="", xLabel="m_{T} (GeV)", yRatio=1.03, rebin=2)
        doPlot("lep_pT", xMin=26, xMax=55, yMin=1e3, yMax= 1e4, label="", xLabel="p_{T}(l) (GeV)", yRatio=1.03)
        
        doPlot("recoil_corr_xy_magn", xMin=0, xMax=100, yMin=1e1, yMax= 1e5, label="", xLabel="|U| (GeV)", yRatio=1.08, logPos=[.60, 0.65, .9, .88])
        doPlot("recoil_corr_xy_para", xMin=-50, xMax=50, yMin=1e1, yMax= 1e4, label="", xLabel="U_{#parallel} (GeV)", yRatio=1.03)
        doPlot("recoil_corr_xy_para_qT", xMin=-100, xMax=50, yMin=1e1, yMax= 1e4, label="", xLabel="U_{#parallel} #minus q_{T} (GeV)", yRatio=1.08, rebin=2)
        doPlot("recoil_corr_xy_perp", xMin=-50, xMax=50, yMin=1e1, yMax= 1e4, label="", xLabel="U_{#perp}   (GeV)", yRatio=1.03)
        doPlot("MET_corr_xy_pt", xMin=0, xMax=50, yMin=10, yMax= 1e5, label="", xLabel="MET p_{T} (GeV)", yRatio=1.03, rebin=2, logPos=[.60, 0.65, .9, .88])
 
 
 
    if True: # highPU
        met = "RawPFMET" # RawPFMET DeepMETReso
        flavor = "mumu" # mumu, ee
        

        procLabel = "DY #rightarrow #mu^{+}#mu^{#minus}"
        metLabel = "RawPFMET"

        outDir = "/eos/user/j/jaeyserm/www/recoil/biasStudy/highPU_zmumu"
        functions.prepareDir(outDir, False)

        groups = Datagroups("mz_wlike_with_mu_eta_pt_%s.pkl.lz4" % (met))
        
        recoil_qTbins = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 70, 5)) + list(range(70, 120, 10)) + [120, 150, 300]
        
        doPlot("qT", xMin=0, xMax=100, yMin=1e3, yMax= 1e7, label="", xLabel="q_{T} (GeV)", yRatio=1.15, rebin=recoil_qTbins, logPos=[.60, 0.65, .9, .82])
        
        doPlot("mT_corr_xy", xMin=0, xMax=120, yMin=1e3, yMax= 1e6, label="", xLabel="m_{T} (GeV)", yRatio=1.03, rebin=2)
        doPlot("nominal", xMin=26, xMax=55, yMin=1e5, yMax= 1e6, label="", xLabel="p_{T}(l) (GeV)", yRatio=1.03)
        
        
        

        
        doPlot("recoil_corr_xy_magn", xMin=0, xMax=100, yMin=1e3, yMax= 1e7, label="", xLabel="|U| (GeV)", yRatio=1.08, logPos=[.60, 0.65, .9, .88])
        doPlot("recoil_corr_xy_para", xMin=-50, xMax=50, yMin=1e3, yMax= 1e6, label="", xLabel="U_{#parallel} (GeV)", yRatio=1.03)
        doPlot("recoil_corr_xy_para_qT", xMin=-100, xMax=50, yMin=1e3, yMax= 1e6, label="", xLabel="U_{#parallel} #minus q_{T} (GeV)", yRatio=1.08, rebin=2)
        doPlot("recoil_corr_xy_perp", xMin=-50, xMax=50, yMin=1e3, yMax= 1e6, label="", xLabel="U_{#perp}   (GeV)", yRatio=1.03)
        doPlot("MET_corr_xy_pt", xMin=0, xMax=50, yMin=1e3, yMax= 1e7, label="", xLabel="MET p_{T} (GeV)", yRatio=1.03, rebin=2, logPos=[.60, 0.65, .9, .88])
 
 
 