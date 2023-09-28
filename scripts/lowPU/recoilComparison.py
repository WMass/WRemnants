
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


def doPlot(tag, qTbin, qTmin, qTmax, h_data, h_mc, h_bkg):

    bins = [-200, -100, -75, -50, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 75, 100, 200]
    bins = 1
    
    sf = (h_data.Integral()-h_bkg.Integral()) / (h_mc.Integral())

    h_data = functions.Rebin(h_data, bins, binWidth=False)
    h_mc = functions.Rebin(h_mc, bins, binWidth=False)
    h_bkg = functions.Rebin(h_bkg, bins, binWidth=False)
    
    h_mc.Scale(sf) # normalize to 1
    
    h_tot = h_mc.Clone("h_tot")
    h_tot.Add(h_bkg)

    xMin,xMax = -100, 100
    
    cfg['xmin'], cfg['xmax'] = xMin, xMax
    cfg['ymax'] = 1.75*h_data.GetMaximum()
    cfg['ymax'] = 1e4
    cfg['ymin'] = 1e-1
    

    h_data.SetLineColor(ROOT.kBlack)
    h_data.SetMarkerStyle(20)
    h_data.SetMarkerColor(ROOT.kBlack)
    h_data.SetLineColor(ROOT.kBlack)
    
    h_tot.SetLineColor(ROOT.kBlack)
    h_tot.SetFillColor(0)
    h_tot.SetLineWidth(2)
    
    h_err = h_tot.Clone("syst")
    h_err.SetFillColor(ROOT.kBlack)
    h_err.SetMarkerSize(0)
    h_err.SetLineWidth(0)
    h_err.SetFillStyle(3004)    
    
    st = ROOT.THStack()
    st.SetName("stack")
    
    h_bkg.SetFillColor(ROOT.kGray)
    h_bkg.SetLineColor(ROOT.kBlack)
    h_bkg.SetLineWidth(1)
    h_bkg.SetLineStyle(1)
    st.Add(h_bkg)

    #h_mc.SetFillColor(ROOT.TColor.GetColor(groups.groups[proc]['color']))
    h_mc.SetLineColor(ROOT.kBlack)
    h_mc.SetLineWidth(1)
    h_mc.SetLineStyle(1)
    st.Add(h_mc)
    
    
    
    # ratios (bands)
    h_bkg_ratio = h_tot.Clone("h_bkg_ratio") # nominal point, need to remove stat. error
    h_err_ratio = h_err.Clone("syst_ratio")
    h_err_ratio.Divide(h_bkg_ratio)


    # Data/MC ratio (the error bars represent the data uncertainty)
    h_ratio = h_data.Clone("h_ratio")
    for i in range(h_tot .GetNbinsX()+1): h_tot.SetBinError(i, 0) # set MC errors to zero
    h_ratio.Divide(h_tot)
    h_ratio.SetMarkerStyle(20)
    h_ratio.SetMarkerSize(0.7)
    h_ratio.SetMarkerColor(ROOT.kBlack)
    h_ratio.SetLineColor(ROOT.kBlack)
    h_ratio.SetLineWidth(1)
    
    
    #print("-----> %f" % (h_tot.Chi2Test(h_data, "CHI2/NDF")))
    
    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio(line=1)
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
    
    st.Draw("HIST SAME")
    h_err.Draw("E2 SAME")
    h_tot.Draw("HIST SAME")
    h_data.Draw("PE SAME")
 
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.85, label)
    latex.DrawLatex(0.20, 0.80, "q_{T} = [%.1f, %.1f] GeV" % (qTmin, qTmax))

    plotter.auxRatio()
       

    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.Draw("HIST")
    dummyL.Draw("SAME")
    
    h_ratio.Draw("P SAME") # E2 SAME
    h_err_ratio.Draw("E2 SAME")

    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s/%03d_recoil.png" % (outDir, tag, qTbin))
    #canvas.SaveAs("%s/%03d_recoil_%s.pdf" % (outDir, qTbin, name))

    dummyB.Delete()
    dummyT.Delete()
    padT.Delete()
    padB.Delete()
    



def readProc(hName, procName):

    label = "%s_%s" % (hName, procName)
    #print(groups_mumu.groups)
    groups_mumu.setHists(hName, "", label=label, procsToRead=[procName], selectSignal=False)
    bhist = groups_mumu.groups[procName][label]
    return bhist


def param():

    fIn = ROOT.TFile("/eos/user/j/jaeyserm/analysis/lowPU/recoil_lowPU_fitParam_RoccoR_lowPU_v0_mirror.root")
    recoil_qTbins = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + list(range(100, 150, 10)) + [150, 175, 200, 10000]
    
    
    h = fIn.Get("h_perp_mc")
    
    for i in range(1, len(recoil_qTbins)):
        
        qT = 0.5*(recoil_qTbins[i-1] + recoil_qTbins[i])
        
        print(qT, h.GetBinContent(i, 2, 1), h.GetBinContent(i, 5, 1), h.GetBinContent(i, 8, 1))
    
    fIn.Close()

def met_recoil_correlation():

    groups_mumu = Datagroups("mz_lowPU_mumu.pkl.lz4")
    
    label = "uncorr"
    groups_mumu.setHists("met_recoil_uncorr", "", label=label, procsToRead=["DYmumu"], selectSignal=False)
    bhist = groups_mumu.groups["DYmumu"][label]
    rhist = narf.hist_to_root(bhist)
    
    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/Zmumu/correlationPlots/"
    functions.prepareDir(outDir, remove=False)
    
    cfg = {

        'logy'              : False,
        'logx'              : False,
    
        'xmin'              : 0,
        'xmax'              : 50,
        'ymin'              : 0,
        'ymax'              : 50,
        
        'xtitle'            : "MET",
        'ytitle'            : "Recoil",
        
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
    }
    

    plotter.cfg = cfg
    canvas = plotter.canvas()
    canvas.SetRightMargin(0.12)
    #canvas.SetLogz()
    
    dummy = plotter.dummy()  
    canvas.cd()
    dummy.Draw("HIST")
    
    #hist.GetZaxis().SetRangeUser(1e-2, 1e4)
    rhist.Draw("SAME COLZ")
        
    plotter.aux(canvas)
    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.90, "Uncorrected MET/Recoil")
    #latex.DrawLatex(0.20, 0.85, met_labels[met_defs.index(met)]) 
    
    line = ROOT.TLine(cfg['xmin'], cfg['ymin'], cfg['xmax'], cfg['ymax'])
    line.SetLineColor(ROOT.kBlack)
    line.SetLineWidth(3)
    line.SetLineStyle(1)
    line.Draw("SAME")
        
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.RedrawAxis()
    canvas.SaveAs("%s/met_recoil_uncorr_DeepMET.png" % outDir)
    canvas.SaveAs("%s/met_recoil_uncorr_DeepMET.pdf" % outDir)
    canvas.Delete() 
 








    label = "corr"
    groups_mumu.setHists("met_recoil_corr", "", label=label, procsToRead=["DYmumu"], selectSignal=False)
    bhist = groups_mumu.groups["DYmumu"][label]
    rhist = narf.hist_to_root(bhist)
    
    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/Zmumu/correlationPlots/"
    functions.prepareDir(outDir, remove=False)
    
    cfg = {

        'logy'              : False,
        'logx'              : False,
    
        'xmin'              : 0,
        'xmax'              : 50,
        'ymin'              : 0,
        'ymax'              : 50,
        
        'xtitle'            : "MET",
        'ytitle'            : "Recoil",
        
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
    }
    

    plotter.cfg = cfg
    canvas = plotter.canvas()
    canvas.SetRightMargin(0.12)
    #canvas.SetLogz()
    
    dummy = plotter.dummy()  
    canvas.cd()
    dummy.Draw("HIST")
    
    #hist.GetZaxis().SetRangeUser(1e-2, 1e4)
    rhist.Draw("SAME COLZ")
        
    plotter.aux(canvas)
    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.90, "Corrected MET/Recoil")
    #latex.DrawLatex(0.20, 0.85, met_labels[met_defs.index(met)]) 
    
    line = ROOT.TLine(cfg['xmin'], cfg['ymin'], cfg['xmax'], cfg['ymax'])
    line.SetLineColor(ROOT.kBlack)
    line.SetLineWidth(3)
    line.SetLineStyle(1)
    line.Draw("SAME")
        
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.RedrawAxis()
    canvas.SaveAs("%s/met_recoil_corr_DeepMET.png" % outDir)
    canvas.SaveAs("%s/met_recoil_corr_DeepMET.pdf" % outDir)
    canvas.Delete() 
    
    
def recoilParameters():

    hName, tag, idx, yMin, yMax, title, yLabel = "h_para_mc", "mc_para_sigma1", 2, 0, 20, "#sigma_{1} (GeV), MC, parallel", "#sigma (GeV)"
    #hName, tag, idx, yMin, yMax, title, yLabel = "h_para_mc", "mc_para_sigma2", 5, 0, 20, "#sigma_{2} (GeV), MC, parallel", "#sigma (GeV)"
    #hName, tag, idx, yMin, yMax, title, yLabel = "h_para_mc", "mc_para_sigma3", 8, 5, 35, "#sigma_{3} (GeV), MC, parallel", "#sigma (GeV)"
    
    #hName, tag, idx, yMin, yMax, title, yLabel = "h_para_mc", "mc_para_mean1", 1, 0, 15, "#mu_{3} (GeV), MC, parallel", "#mu (GeV)"
    #hName, tag, idx, yMin, yMax, title, yLabel = "h_para_mc", "mc_para_mean2", 4, 0, 10, "#mu_{3} (GeV), MC, parallel", "#mu (GeV)"
    #hName, tag, idx, yMin, yMax, title, yLabel = "h_para_mc", "mc_para_mean3", 7, 0, 20, "#mu_{3} (GeV), MC, parallel", "#mu (GeV)"

    hName, tag, idx, yMin, yMax, title, yLabel = "mc_perp", "mc_perp_sigma1", 2, 5, 15, "#sigma_{1} (GeV), MC, perpendicular", "#sigma (GeV)"
    hName, tag, idx, yMin, yMax, title, yLabel = "mc_perp", "mc_perp_sigma2", 5, 0, 10, "#sigma_{2} (GeV), MC, perpendicular", "#sigma (GeV)"
    hName, tag, idx, yMin, yMax, title, yLabel = "mc_perp", "mc_perp_sigma3", 8, 5, 25, "#sigma_{3} (GeV), MC, perpendicular", "#sigma (GeV)"
    
    #hName, tag, idx, yMin, yMax, title, yLabel = "mc_perp", "mc_perp_mean1", 1, 0, 1, "#mu_{3} (GeV), MC, perpendicular", "#mu (GeV)"



    #hName, tag, idx, yMin, yMax, title, yLabel = "h_para_data", "data_para_sigma1", 2, 0, 20, "#sigma_{1} (GeV), Data, parallel", "#sigma (GeV)"
    #hName, tag, idx, yMin, yMax, title, yLabel = "h_para_data", "data_para_sigma2", 5, 0, 20, "#sigma_{2} (GeV), Data, parallel", "#sigma (GeV)"
    #hName, tag, idx, yMin, yMax, title, yLabel = "h_para_data", "data_para_sigma3", 8, 5, 35, "#sigma_{3} (GeV), Data, parallel", "#sigma (GeV)"
    
    #hName, tag, idx, yMin, yMax, title, yLabel = "h_para_data", "data_para_mean1", 1, 0, 15, "#mu_{3} (GeV), Data, parallel", "#mu (GeV)"
    #hName, tag, idx, yMin, yMax, title, yLabel = "h_para_data", "data_para_mean2", 4, 0, 10, "#mu_{3} (GeV), Data, parallel", "#mu (GeV)"
    #hName, tag, idx, yMin, yMax, title, yLabel = "h_para_data", "data_para_mean3", 7, 0, 20, "#mu_{3} (GeV), Data, parallel", "#mu (GeV)"

    hName, tag, idx, yMin, yMax, title, yLabel = "data_perp", "data_perp_sigma1", 2, 5, 15, "#sigma_{1} (GeV), Data, perpendicular", "#sigma (GeV)"
    hName, tag, idx, yMin, yMax, title, yLabel = "data_perp", "data_perp_sigma2", 5, 0, 10, "#sigma_{2} (GeV), Data, perpendicular", "#sigma (GeV)"
    hName, tag, idx, yMin, yMax, title, yLabel = "data_perp", "data_perp_sigma3", 8, 5, 30, "#sigma_{3} (GeV), Data, perpendicular", "#sigma (GeV)"
    
    hName, tag, idx, yMin, yMax, title, yLabel = "data_perp", "data_perp_mean1", 1, -1, 1, "#mu_{3} (GeV), Data, perpendicular", "#mu (GeV)"
  

    recoil_qTbins = list(functions.drange(0, 30, 0.5)) + list(range(30, 50, 1)) + list(range(50, 70, 2)) + list(range(70, 100, 5)) + list(range(100, 150, 10)) + [150, 200, 300, 10000]

    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/recoil_mumu_DeepMET_lowPU_RoccoR_lowPU_v0/parameterization/"
    functions.prepareDir(outDir, remove=False)
    
    fIn = ROOT.TFile("/eos/user/j/jaeyserm/analysis/lowPU/recoil_mumu_DeepMET_lowPU_RoccoR_lowPU_v0_param_data_perp.root")
    hIn = fIn.Get(hName)
 
    g = ROOT.TGraphErrors()
    g.SetLineColor(ROOT.kBlue)
    g.SetMarkerStyle(20)
    g.SetMarkerSize(0.9)
    g.SetMarkerColor(ROOT.kBlue)
    
    for i in range(1, len(recoil_qTbins)):
    
        qT = 0.5*(recoil_qTbins[i] + recoil_qTbins[i-1])
        print(i, qT, hIn.GetBinContent(i, idx, 1))
        g.SetPoint(i-1, qT, hIn.GetBinContent(i, idx, 1))
        g.SetPointError(i-1, 0, hIn.GetBinContent(i, idx, 0))
 
    
    fit = ROOT.TF1("fit", "[0]*TMath::Power(x+[1], [2])", 0, 150)
    #fit.SetParameters(7.82921e-01, 2.52028e+01, 5.78980e-01)
    #fit = ROOT.TF1("fit", "[0]*x + [1]", 0, 200)
    result = g.Fit(fit.GetName(), "NS", "", 0, 200) 
    fit.SetLineColor(ROOT.kRed)
    fit.GetXaxis().SetRangeUser(0, 200)
    fit.SetLineWidth(2)

    ## sigmas
    cfg = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : 200,
        'ymin'              : yMin,
        'ymax'              : yMax,
            
        'xtitle'            : "q_{T} (GeV)",
        'ytitle'            : yLabel,
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    } 
    

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy()   
    canvas.cd()
    dummy.Draw("HIST")
    g.Draw("PE SAME")
    fit.Draw("L SAME")
    plotter.aux()
    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.85, title)
        
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s.png" % (outDir, tag))
    canvas.SaveAs("%s/%s.pdf" % (outDir, tag))
    canvas.Delete()


def bareRecoilParameters():

    met = "RawPFMET" # RawPFMET DeepMETReso
    flavor = "mumu" # mu, e, mumu, ee

    hName = "data_perp"
    hName = "SingleMuon_perp"
    #hName = "DYmumu_para"
    qTmax = 160
    qTminFit, qTmaxFit = 0, 40

    recoil_qTbins = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 70, 5)) + list(range(70, 150, 10)) + [150, 200, 300, 10000]
    recoil_qTbins = list(range(0, 50, 1)) + list(range(50, 70, 2)) + list(range(70, 100, 5)) + list(range(100, 150, 10)) + [150, 200, 300, 10000]
    recoil_qTbins = list(range(0, 20, 1)) + list(range(20, 40, 2)) + list(range(40, 55, 3)) + list(range(55, 80, 5)) + list(range(80, 100, 10)) + [100, 125, 150, 10000]
    recoil_qTbins = list(range(0, 200, 1))
    recoil_qTbins = list(range(0, 50, 1)) + list(range(50, 70, 2)) + list(range(70, 100, 5)) + list(range(100, 150, 10)) + [150, 200, 300, 10000]
     
    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/recoil_%s_%s/parameterization/" % (flavor, met)
    functions.prepareDir(outDir, remove=False)
    
    fIn_ = "wremnants/data/lowPU/recoil_%s_%s.root" % (flavor, met)
    fIn = ROOT.TFile(fIn_)
    fIn.ls()
    g_mean = ROOT.TGraphErrors()
    g_mean.SetLineColor(ROOT.kBlue)
    g_mean.SetMarkerStyle(20)
    g_mean.SetMarkerSize(0.9)
    g_mean.SetMarkerColor(ROOT.kBlue)
    
    g_sigma = ROOT.TGraphErrors()
    g_sigma.SetLineColor(ROOT.kBlue)
    g_sigma.SetMarkerStyle(20)
    g_sigma.SetMarkerSize(0.9)
    g_sigma.SetMarkerColor(ROOT.kBlue)
    
    for i in range(1, len(recoil_qTbins)):

        h = fIn.Get("%s_bin%d" % (hName, i))
        mean, mean_err = h.GetMean(), h.GetMeanError()
        sigma, sigma_err = h.GetRMS(), h.GetRMSError()
        qT = 0.5*(recoil_qTbins[i] + recoil_qTbins[i-1])
        print(i, qT, mean, sigma)
        
        g_mean.SetPoint(i-1, qT, mean)
        g_mean.SetPointError(i-1, 0, mean_err)
        
        g_sigma.SetPoint(i-1, qT, sigma)
        g_sigma.SetPointError(i-1, 0, sigma_err)
 

    fit_sigma = ROOT.TF1("fit_sigma", "[0]*TMath::Power(x+[1], [2])", qTminFit, qTmax)
    fit_sigma.SetParameters(7.82921e-01, 2.52028e+01, 5.78980e-01)
    result = g_sigma.Fit(fit_sigma.GetName(), "NS", "", qTminFit, qTmaxFit) 
    fit_sigma.SetLineColor(ROOT.kRed)
    fit_sigma.GetXaxis().SetRangeUser(qTminFit, qTmax)
    fit_sigma.SetLineWidth(2)
    
    fit_mean = ROOT.TF1("fit_mean", "[0]*x + [1]", qTminFit, qTmaxFit)
    result = g_mean.Fit(fit_mean.GetName(), "NS", "", qTminFit, qTmaxFit) 
    fit_mean.SetLineColor(ROOT.kRed)
    fit_mean.GetXaxis().SetRangeUser(qTminFit, qTmaxFit)
    fit_mean.SetLineWidth(2)
    
    

    ## sigma
    cfg = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : qTmax,
        'ymin'              : 0,
        'ymax'              : 30,
            
        'xtitle'            : "q_{T} (GeV)",
        'ytitle'            : "#sigma (GeV)",
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    } 
    

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy()   
    canvas.cd()
    dummy.Draw("HIST")
    g_sigma.Draw("PE SAME")
    fit_sigma.Draw("L SAME")
    plotter.aux()
    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.85, "#sigma (GeV), MC, perpendicular")
        
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s_sigma.png" % (outDir, hName))
    canvas.SaveAs("%s/%s_sigma.pdf" % (outDir, hName))
    canvas.Delete()
    
    
    
    
    
    ## mean
    cfg = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : qTmax,
        'ymin'              : -15,
        'ymax'              : 15,
            
        'xtitle'            : "q_{T} (GeV)",
        'ytitle'            : "#mu (Gev)",
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    } 
    

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy()   
    canvas.cd()
    dummy.Draw("HIST")
    g_mean.Draw("PE SAME")
    fit_mean.Draw("L SAME")
    plotter.aux()
    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.90, MET_label)
    latex.DrawLatex(0.20, 0.85, "#mu(U_{#perp}  )  (GeV), MC")
        
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s_mean.png" % (outDir, hName))
    canvas.SaveAs("%s/%s_mean.pdf" % (outDir, hName))
    canvas.Delete()

def chi2():

    met = "DeepMETReso" # RawPFMET DeepMETReso
    flavor = "mumu" # mu, e, mumu, ee

    label = "DY #rightarrow #mu^{+}#mu^{#minus}, %s" % met
    ####################################################################
    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/recoil_%s_%s/" % (flavor, met)
    recoil_qTbins = list(range(0, 50, 1)) + list(range(50, 70, 2)) + list(range(70, 100, 5)) + list(range(100, 150, 10)) + [150, 200, 300, 10000]
     
    hNames = ["data_para", "data_perp", "mc_para", "mc_perp"]
    labels = ["Data para", "Data perp", "MC para", "MC perp"]
    colors = [ROOT.kRed, ROOT.kRed, ROOT.kBlue, ROOT.kBlue]
    markers = [8,21,8,21]

    fIn = ROOT.TFile("wremnants/data/lowPU/recoil_param_%s_%s.root" % (flavor, met))
    fIn.ls()
    
    graphs = []
    for i, hName in enumerate(hNames):
    
        g = ROOT.TGraphErrors()
        g.SetLineColor(colors[i])
        g.SetMarkerStyle(markers[i])
        g.SetMarkerSize(1)
        g.SetMarkerColor(colors[i])
        
        h = fIn.Get(hName)
        for j in range(1, len(recoil_qTbins)):
            qT = 0.5*(recoil_qTbins[j] + recoil_qTbins[j-1])
            g.SetPoint(j-1, qT, h.GetBinContent(j, 0, 0))
            print(qT, h.GetBinContent(j, 0, 0))
            
        graphs.append(g)
    
    cfg = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : 100,
        'ymin'              : 0,
        'ymax'              : 3,
            
        'xtitle'            : "q_{T} (GeV)",
        'ytitle'            : "#chi^{2}",
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    } 
    
    
    leg = ROOT.TLegend(.30, 0.7, .9, .90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)
    leg.SetHeader(label)

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy()   
    canvas.cd()
    dummy.Draw("HIST")
    for i,g in enumerate(graphs):
        g.Draw("SAME LP")
        leg.AddEntry(g, labels[i], "LP")
    plotter.aux()
    leg.Draw()
    
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/chi2.png" % outDir)
    canvas.SaveAs("%s/chi2.pdf" % outDir)
    canvas.Delete()
    
    
def closure_mc_perp():


    
if __name__ == "__main__":


    met = "RawPFMET" # RawPFMET DeepMETReso
    flavor = "mumu" # mu, e, mumu, ee

    if flavor == "mumu": 
        label = "DY #rightarrow #mu^{+}#mu^{#minus}, %s" % met
        sig = "DYmumu"
        data = "SingleMuon"
    if flavor == "ee": 
        label = "DY #rightarrow e^{+}e^{#minus}, %s" % met
        sig = "DYee"
        data = "SingleElectron"
    ####################################################################
    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/recoil_%s_%s/" % (flavor, met)
    fIn_hist = "wremnants/data/lowPU/recoil_%s_%s.root" % (flavor, met)
    fIn_param = "wremnants/data/lowPU/recoil_param_%s_%s.root" % (flavor, met)

    closure_mc_perp()
    sys.exit()
    #met_recoil_correlation()
    #param()
    
    MET_label = "DeepMET (resolution)"
    
    bareRecoilParameters()
    #recoilParameters()
    
    
    

    
    sys.exit()

    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/fits_Z_auto_RoccoR_lowPU_v0_mirror/"
    fIn = ROOT.TFile("/eos/user/j/jaeyserm/analysis/lowPU/recoil_lowPU_RoccoR_lowPU_v0.root")

    yRatio = 1.3
    cfg = {

        'logy'              : True,
        'logx'              : False,
    
        'xmin'              : -300,
        'xmax'              : 100,
        'ymin'              : 1e-1,
        'ymax'              : 1,
        
        'xtitle'            : "Recoil U_{#parallel} (GeV)",
        'ytitle'            : "Events / 1 GeV",
        
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
        
        'ratiofraction'     : 0.25,
        'ytitleR'           : "Pull",
        'yminR'             : 1-(yRatio-1),
        'ymaxR'             : yRatio,
    }   
    
    groups_mumu = Datagroups("mz_lowPU_mumu.pkl.lz4")
    bkg_procs = ['EWK', 'TTbar']
    
    
   
    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/Comparison/"
    dataName = "SingleMuon"
    signalName = "DYmumu"

    b_data_para_qT = readProc("recoil_corr_para_qT_qTbinned", dataName)
    b_data_perp = readProc("recoil_corr_perp_qTbinned", dataName)
    b_mc_para_qT = readProc("recoil_corr_para_qT_qTbinned", signalName) 
    b_mc_perp = readProc("recoil_corr_perp_qTbinned", signalName)
    
    

    b_bkg_para_qT, b_bkg_perp = None, None
    for bkg in bkg_procs:
        
        b_para_qT = readProc("recoil_corr_para_qT_qTbinned", bkg)
        if b_bkg_para_qT == None: b_bkg_para_qT = b_para_qT
        else: b_bkg_para_qT += b_para_qT
    
        b_perp = readProc("recoil_corr_perp_qTbinned", bkg)
        if b_bkg_perp == None: b_bkg_perp = b_perp
        else: b_bkg_perp += b_perp
      
      
    h_data_para_qT = narf.hist_to_root(b_data_para_qT)
    h_mc_para_qT = narf.hist_to_root(b_mc_para_qT)
    h_bkg_para_qT = narf.hist_to_root(b_bkg_para_qT)
    h_data_perp = narf.hist_to_root(b_data_perp)
    h_mc_perp = narf.hist_to_root(b_mc_perp)
    h_bkg_perp = narf.hist_to_root(b_bkg_perp)

    
    label = "DY #rightarrow #mu^{+}#mu^{#minus}" # #plus e^{+}e^{#minus}
    

    
    functions.prepareDir(outDir, False)
    functions.prepareDir(outDir + "/perp/", False)
    functions.prepareDir(outDir + "/para/", False)
    
    
    for iBin in range(1, h_data_para_qT.GetNbinsX()+1):
    
        qTbinMinGeV, qTbinMaxGeV = h_data_para_qT.GetXaxis().GetBinLowEdge(iBin), h_data_para_qT.GetXaxis().GetBinLowEdge(iBin+1)
        qT = h_data_para_qT.GetXaxis().GetBinCenter(iBin)
        
        print("***************** Do iBin %d [%f,%f]" % (iBin, qTbinMinGeV, qTbinMaxGeV))
        
        
        h_data = h_data_para_qT.ProjectionY("data_para_bin%d" % iBin, iBin, iBin)
        h_mc = h_mc_para_qT.ProjectionY("data_mc_bin%d" % iBin, iBin, iBin)
        h_bkg = h_bkg_para_qT.ProjectionY("data_bkg_bin%d" % iBin, iBin, iBin)
        doPlot("para", iBin, qTbinMinGeV, qTbinMaxGeV, h_data, h_mc, h_bkg)
        
        h_data = h_data_perp.ProjectionY("data_perp_bin%d" % iBin, iBin, iBin)
        h_mc = h_mc_perp.ProjectionY("data_mc_bin%d" % iBin, iBin, iBin)
        h_bkg = h_bkg_perp.ProjectionY("data_bkg_bin%d" % iBin, iBin, iBin)
        doPlot("perp", iBin, qTbinMinGeV, qTbinMaxGeV, h_data, h_mc, h_bkg)
    
   