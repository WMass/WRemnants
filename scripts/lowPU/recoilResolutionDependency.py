
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
import numpy as np

from wremnants.datasets.datagroups import Datagroups



def readProc(groups, hName, procName):

    label = "%s_%s" % (hName, procName)
    groups.setHists(hName, "", label=label, procsToRead=[procName], selectSignal=False)
    bhist = groups.groups[procName][label]
    return bhist
    
    
def doPlot(tag, mode, fOut, xLabel, xMin, xMax, yMin, yMax):

    b_data = readProc(datagroups, tag, dataLabel)
    b_mc = readProc(datagroups, tag, sigLabel)

    h_data = narf.hist_to_root(b_data)
    h_mc = narf.hist_to_root(b_mc)
    h_mc_x = h_mc.ProjectionX("hist_x")
    
    
    g_data = ROOT.TGraphErrors()
    g_data.SetLineColor(ROOT.kBlue)
    g_data.SetMarkerStyle(20)
    g_data.SetMarkerSize(0.9)
    g_data.SetMarkerColor(ROOT.kBlue)
    
    g_mc = ROOT.TGraphErrors()
    g_mc.SetLineColor(ROOT.kRed)
    g_mc.SetMarkerStyle(20)
    g_mc.SetMarkerSize(0.9)
    g_mc.SetMarkerColor(ROOT.kRed)
    
    for iBin in range(1, h_mc_x.GetNbinsX()+1):
    
        x = h_mc_x.GetBinCenter(iBin)
        h_data_tmp = h_data.ProjectionY("test_data", iBin, iBin)
        h_mc_tmp = h_mc.ProjectionY("test_mc", iBin, iBin)
        
        g_data.SetPoint(iBin-1, x, abs(h_data_tmp.GetRMS() if mode == "sigma" else h_data_tmp.GetMean()))
        g_data.SetPointError(iBin-1, 0, h_data_tmp.GetRMSError() if mode == "sigma" else h_data_tmp.GetMeanError())
        
        g_mc.SetPoint(iBin-1, x, abs(h_mc_tmp.GetRMS() if mode == "sigma" else h_mc_tmp.GetMean()))
        g_mc.SetPointError(iBin-1, 0, h_mc_tmp.GetRMSError() if mode == "sigma" else h_mc_tmp.GetMeanError())
   
    leg = ROOT.TLegend(.50, 0.75, .9, .90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.040)
    #leg.SetHeader(header) 

    leg.AddEntry(g_data, "Data", "PE")
    leg.AddEntry(g_mc, "MC (uncorrected)", "PE")

    if mode == "sigma": ytitle = "#sigma"
    else: ytitle = "#mu"
  
    cfg = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : xMin,
        'xmax'              : xMax,
        'ymin'              : yMin,
        'ymax'              : yMax,
            
        'xtitle'            : xLabel,
        'ytitle'            : ytitle+"_{#perp}  (GeV)" if "perp" in tag else ytitle+"_{#parallel} (GeV)",
            
        'topRight'          : lumiLabel, 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    } 
    

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy()   
    canvas.cd()
    canvas.SetTickx()
    canvas.SetTicky()
    canvas.SetGrid()
    dummy.Draw("HIST")
    g_data.Draw("PE SAME")
    g_mc.Draw("PE SAME")
    leg.Draw()
    plotter.aux()
    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.034)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.87, ytitle+"_{#perp}" if "perp" in tag else ytitle+"_{#parallel}")
    latex.DrawLatex(0.20, 0.82, "%s" % met)
        
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s_%s_%s.png" % (outDir, fOut, "perp" if "perp" in tag else "para", mode))
    canvas.SaveAs("%s/%s_%s_%s.pdf" % (outDir, fOut, "perp" if "perp" in tag else "para", mode))
    canvas.Delete()
    
    
def qT_sumEt_correlation(xMin, xMax, yMin, yMax):

    b_data = readProc(datagroups, "qT_sumEt", dataLabel)
    b_mc = readProc(datagroups, "qT_sumEt", sigLabel)

    h_data = narf.hist_to_root(b_data)
    h_mc = narf.hist_to_root(b_mc)

    fOut = "qT_sumEt_correlation"
    
    cfg = {

        'logy'              : False,
        'logx'              : False,
    
        'xmin'              : xMin,
        'xmax'              : xMax,
        'ymin'              : yMin,
        'ymax'              : yMax,
        
        'xtitle'            : "q_{T} (GeV)",
        'ytitle'            : "sumET (GeV)",
        
        'topRight'          : lumiLabel, 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
    }
    
    

    plotter.cfg = cfg
    canvas = plotter.canvas()
    canvas.SetRightMargin(0.12)
    canvas.SetLogz()
    
    dummy = plotter.dummy()   
    canvas.cd()
    dummy.Draw("HIST")
    #hist.GetZaxis().SetRangeUser(1e-2, 1e4)
    h_data.Draw("SAME COLZ")
    plotter.aux(canvas)
    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.90, "")
    #latex.DrawLatex(0.20, 0.85, met_labels[met_defs.index(met)]) 
    
    line = ROOT.TLine(cfg['xmin'], cfg['ymin'], cfg['xmax'], cfg['ymax'])
    line.SetLineColor(ROOT.kBlack)
    line.SetLineWidth(3)
    line.SetLineStyle(1)
    #line.Draw("SAME")
        
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.RedrawAxis()
    canvas.SaveAs("%s/%s.png" % (outDir, fOut))
    canvas.SaveAs("%s/%s.pdf" % (outDir, fOut))
    canvas.Delete() 
 
    
if __name__ == "__main__":

    met = "DeepMETReso" # RawPFMET DeepMETReso
    flavor = "mumu" # mu, e, mumu, ee
    lowPU = False
    
    if lowPU:

        datagroups = Datagroups("lowPU_%s_%s.pkl.lz4" % (flavor, met))
        lumiLabel = "199 pb^{#minus1} (13 TeV)"

        ##procs = ["Zmumu", "Ztautau", "Other"]
        ##data = "SingleMuon"
        ##outDir = "/eos/user/j/jaeyserm/www/wmass/highPU/recoilCorrection/resolutionDependency/recoil_%s_%s/" % (flavor, met)
        outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/recoil_%s_%s/resolutionDependency/" % (flavor, met)
        functions.prepareDir(outDir, True)
        
        dataLabel = "SingleMuon"
        sigLabel = "DYmumu"
        
        qT_sumEt_correlation(0, 100, 0, 800)
    
        doPlot("recoil_corr_xy_para_qT_npv", "sigma", "npv", "Number of primary vertices", 0, 10, 0, 20)
        doPlot("recoil_corr_xy_perp_npv", "sigma", "npv", "Number of primary vertices", 0, 10, 0, 20)
        
        doPlot("recoil_corr_xy_para_qT_sumEt", "sigma", "sumET", "SumET (GeV)", 0, 800, 0, 35)
        doPlot("recoil_corr_xy_perp_sumEt", "sigma", "sumET", "SumET (GeV)", 0, 800, 0, 35)
        
        doPlot("recoil_corr_xy_para_qT_qTbinned", "sigma", "qT", "q_{T} (GeV)", 0, 60, 0, 35)
        doPlot("recoil_corr_xy_perp_qTbinned", "sigma", "qT", "q_{T} (GeV)", 0, 60, 0, 35)
        
        doPlot("recoil_corr_xy_para_qT_rapidity", "sigma", "rapidity", "y_{ll}", -2.4, 2.4, 0, 35)
        doPlot("recoil_corr_xy_perp_rapidity", "sigma", "rapidity", "y_{ll}", -2.4, 2.4, 0, 35)
        
        doPlot("recoil_corr_xy_para_qT_njets", "sigma", "njets", "Number of jets", 0, 10, 0, 35)
        doPlot("recoil_corr_xy_perp_njets", "sigma", "njets", "Number of jets", 0, 10, 0, 35)
    
    
        doPlot("recoil_corr_xy_para_qT_npv", "mean", "npv", "Number of primary vertices", 0, 10, 0, 10)
        doPlot("recoil_corr_xy_perp_npv", "mean", "npv", "Number of primary vertices", 0, 10, -5, 5)
        
        doPlot("recoil_corr_xy_para_qT_sumEt", "mean", "sumET", "SumET (GeV)", 0, 800, 0, 10)
        doPlot("recoil_corr_xy_perp_sumEt", "mean", "sumET", "SumET (GeV)", 0, 800, -5, 5)
        
        doPlot("recoil_corr_xy_para_qT_qTbinned", "mean", "qT", "q_{T} (GeV)", 0, 60, 0, 15)
        doPlot("recoil_corr_xy_perp_qTbinned", "mean", "qT", "q_{T} (GeV)", 0, 60, -5, 5)
        
        doPlot("recoil_corr_xy_para_qT_rapidity", "mean", "rapidity", "y_{ll}", -2.4, 2.4, 0, 10)
        doPlot("recoil_corr_xy_perp_rapidity", "mean", "rapidity", "y_{ll}", -2.4, 2.4, -5, 5)
        
        doPlot("recoil_corr_xy_para_qT_njets", "mean", "njets", "Number of jets", 0, 10, 0, 15)
        doPlot("recoil_corr_xy_perp_njets", "mean", "njets", "Number of jets", 0, 10, -5, 5)
        
        
    else:

        datagroups = Datagroups("mz_wlike_with_mu_eta_pt_%s.pkl.lz4" % met, wlike=True)
        lumiLabel = "16.8 fb^{#minus1} (13 TeV)"

        procs = ["Zmumu", "Ztautau", "Other"]
        data = "Data"
        ##outDir = "/eos/user/j/jaeyserm/www/wmass/highPU/recoilCorrection/resolutionDependency/recoil_%s_%s/" % (flavor, met)
        outDir = "/eos/user/j/jaeyserm/www/wmass/highPU/recoilCorrection/recoil_%s_%s/resolutionDependency/" % (flavor, met)
        functions.prepareDir(outDir, True)
    
        dataLabel = "Data"
        sigLabel = "Zmumu"
        
        #qT_sumEt_correlation(0, 200, 0, 3000)
        
        sigmaPerpMax = 40 if met == "DeepMETReso" else 40
        sigmaParaMax = 40 if met == "DeepMETReso" else 40
    
        doPlot("recoil_corr_xy_para_qT_npv", "sigma", "npv", "Number of primary vertices", 10, 40, 0, sigmaPerpMax)
        doPlot("recoil_corr_xy_perp_npv", "sigma", "npv", "Number of primary vertices", 10, 40, 0, sigmaPerpMax)
        
        doPlot("recoil_corr_xy_para_qT_sumEt", "sigma", "sumET", "SumET (GeV)", 0, 3000, 0, sigmaPerpMax)
        doPlot("recoil_corr_xy_perp_sumEt", "sigma", "sumET", "SumET (GeV)", 0, 3000, 0, sigmaPerpMax)
        
        doPlot("recoil_corr_xy_para_qT_qTbinned", "sigma", "qT", "q_{T} (GeV)", 0, 200, 0, sigmaPerpMax)
        doPlot("recoil_corr_xy_perp_qTbinned", "sigma", "qT", "q_{T} (GeV)", 0, 200, 0, sigmaPerpMax)
        
        doPlot("recoil_corr_xy_para_qT_rapidity", "sigma", "rapidity", "y_{ll}", -2.4, 2.4, 0, sigmaPerpMax)
        doPlot("recoil_corr_xy_perp_rapidity", "sigma", "rapidity", "y_{ll}", -2.4, 2.4, 0, sigmaPerpMax)
        
        doPlot("recoil_corr_xy_para_qT_njets", "sigma", "njets", "Number of jets", 0, 12, 0, sigmaPerpMax)
        doPlot("recoil_corr_xy_perp_njets", "sigma", "njets", "Number of jets", 0, 12, 0, sigmaPerpMax)
    
    
        doPlot("recoil_corr_xy_para_qT_npv", "mean", "npv", "Number of primary vertices", 10, 40, 0, 10)
        doPlot("recoil_corr_xy_perp_npv", "mean", "npv", "Number of primary vertices", 10, 40, -5, 5)
        
        doPlot("recoil_corr_xy_para_qT_sumEt", "mean", "sumET", "SumET (GeV)", 0, 3000, 0, 10)
        doPlot("recoil_corr_xy_perp_sumEt", "mean", "sumET", "SumET (GeV)", 0, 3000, -5, 5)
        
        doPlot("recoil_corr_xy_para_qT_qTbinned", "mean", "qT", "q_{T} (GeV)", 0, 200, 0, 30)
        doPlot("recoil_corr_xy_perp_qTbinned", "mean", "qT", "q_{T} (GeV)", 0, 200, -5, 5)
        
        doPlot("recoil_corr_xy_para_qT_rapidity", "mean", "rapidity", "y_{ll}", -2.4, 2.4, 0, 10)
        doPlot("recoil_corr_xy_perp_rapidity", "mean", "rapidity", "y_{ll}", -2.4, 2.4, -5, 5)
        
        doPlot("recoil_corr_xy_para_qT_njets", "mean", "njets", "Number of jets", 0, 12, 0, 10)
        doPlot("recoil_corr_xy_perp_njets", "mean", "njets", "Number of jets", 0, 12, -5, 5)