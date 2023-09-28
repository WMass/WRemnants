
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

def doPlot(tag, h_target, h_uncorr, h_corr, xMin=-100, xMax=100, label="", xLabel=""):
    
    h_target.Scale(1./h_target.Integral())
    h_uncorr.Scale(1./h_uncorr.Integral())
    h_corr.Scale(1./h_corr.Integral())

    yRatio = 1.3
    cfg = {

        'logy'              : True,
        'logx'              : False,
    
        'xmin'              : xMin,
        'xmax'              : xMax,
        'ymin'              : 1e-6,
        'ymax'              : 100,
        
        'xtitle'            : xLabel,
        'ytitle'            : "Events / 1 GeV",
        
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
        
        'ratiofraction'     : 0.25,
        'ytitleR'           : "Source/Target",
        'yminR'             : 1-(yRatio-1),
        'ymaxR'             : yRatio,
    }   
    
    leg = ROOT.TLegend(.20, 0.65, .5, .88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.040)
    leg.SetHeader(label)

    h_target.SetLineColor(ROOT.kBlack)
    h_target.SetMarkerStyle(20)
    h_target.SetMarkerColor(ROOT.kBlack)
    h_target.SetLineColor(ROOT.kBlack)
    
    h_uncorr.SetLineColor(ROOT.kRed)
    h_uncorr.SetFillColor(0)
    h_uncorr.SetLineWidth(2)
    
    h_corr.SetLineColor(ROOT.kBlue)
    h_corr.SetFillColor(0)
    h_corr.SetLineWidth(2)   
    
    leg.AddEntry(h_target, "U_{#perp}, Z, GEN, target", "PE")
    leg.AddEntry(h_uncorr, "U_{#perp}, W, GEN, uncorrected", "LP")
    leg.AddEntry(h_corr, "U_{#perp}, W, GEN, corrected", "LP")

    h_uncorr_ratio = h_uncorr.Clone("h_uncorr_ratio")
    h_uncorr_ratio.Divide(h_target)
    h_uncorr_ratio.SetMarkerStyle(20)
    h_uncorr_ratio.SetMarkerSize(0.7)
    h_uncorr_ratio.SetMarkerColor(ROOT.kRed)
    h_uncorr_ratio.SetLineColor(ROOT.kRed)
    h_uncorr_ratio.SetLineWidth(1)

    
    h_corr_ratio = h_corr.Clone("h_corr_ratio")
    h_corr_ratio.Divide(h_target)
    h_corr_ratio.SetMarkerStyle(20)
    h_corr_ratio.SetMarkerSize(0.7)
    h_corr_ratio.SetMarkerColor(ROOT.kBlue)
    h_corr_ratio.SetLineColor(ROOT.kBlue)
    h_corr_ratio.SetLineWidth(1)
    
    
    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio(line=1)
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
    
    h_target.Draw("PE SAME")
    h_uncorr.Draw("HIST SAME")
    h_corr.Draw("PE SAME")
    leg.Draw()
    plotter.auxRatio()
       

    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.Draw("HIST")
    dummyL.Draw("SAME")
    
    h_uncorr_ratio.Draw("P SAME")
    h_corr_ratio.Draw("P SAME")

    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s.png" % (outDir, tag))
    canvas.SaveAs("%s/%s.pdf" % (outDir, tag))

    dummyB.Delete()
    dummyT.Delete()
    padT.Delete()
    padB.Delete()
    

def compareWplusMinus():

    header = "W^{#pm} #rightarrow #mu^{#pm}#nu, %s" % met
    yRatio = 1.3
    cfg = {

        'logy'              : True,
        'logx'              : False,
    
        'xmin'              : -100,
        'xmax'              : 100,
        'ymin'              : 1e-6,
        'ymax'              : 100,
        
        'xtitle'            : "U_{#parallel} + q_{T} (Gev)",
        'ytitle'            : "Events / 1 GeV",
        
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
        
        'ratiofraction'     : 0.25,
        'ytitleR'           : "W^{#plus}/W^{#minus}",
        'yminR'             : 1-(yRatio-1),
        'ymaxR'             : yRatio,
    }
    
    ########################
    ## para_qT
    ########################
    tag = "wplusminus_para_qT"
    b_wplus = readProc(datagroups_W, "recoil_corr_xy_para_qT_gen", "WplusJetsToMuNu")
    b_wminus = readProc(datagroups_W, "recoil_corr_xy_para_qT_gen", "WminusJetsToMuNu")
    h_wplus = narf.hist_to_root(b_wplus)
    h_wminus = narf.hist_to_root(b_wminus)
    
    h_wplus.Scale(1./h_wplus.Integral())
    h_wminus.Scale(1./h_wminus.Integral())
    
    leg = ROOT.TLegend(.20, 0.70, .5, .88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.040)
    leg.SetHeader(header)

    
    h_wplus.SetLineColor(ROOT.kRed)
    h_wplus.SetFillColor(0)
    h_wplus.SetLineWidth(2)
    
    h_wminus.SetLineColor(ROOT.kBlue)
    h_wminus.SetFillColor(0)
    h_wminus.SetLineWidth(2)   
    
    leg.AddEntry(h_wplus, "W^{#plus} #rightarrow #mu^{#plus}#nu", "L")
    leg.AddEntry(h_wminus, "W^{#minus} #rightarrow #mu^{#minus}#nu", "L")

    h_ratio = h_wplus.Clone("h_ratio")
    h_ratio.Divide(h_wminus)
    h_ratio.SetMarkerStyle(20)
    h_ratio.SetMarkerSize(0.7)
    h_ratio.SetMarkerColor(ROOT.kBlack)
    h_ratio.SetLineColor(ROOT.kBlack)
    h_ratio.SetLineWidth(1)

    
    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio(line=1)
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
    
    h_wplus.Draw("PE SAME")
    h_wminus.Draw("HIST SAME")
    leg.Draw()
    plotter.auxRatio()
    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.Draw("HIST")
    dummyL.Draw("SAME")
    h_ratio.Draw("P SAME")
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s.png" % (outDir, tag))
    canvas.SaveAs("%s/%s.pdf" % (outDir, tag))

    dummyB.Delete()
    dummyT.Delete()
    padT.Delete()
    padB.Delete()
    
    
    
    
    ########################
    ## perp
    ########################
    cfg['xtitle'] = "U_{#perp}  (Gev)"
    tag = "wplusminus_perp"
    b_wplus = readProc(datagroups_W, "recoil_corr_xy_perp_gen", "WplusJetsToMuNu")
    b_wminus = readProc(datagroups_W, "recoil_corr_xy_perp_gen", "WminusJetsToMuNu")
    h_wplus = narf.hist_to_root(b_wplus)
    h_wminus = narf.hist_to_root(b_wminus)
    
    h_wplus.Scale(1./h_wplus.Integral())
    h_wminus.Scale(1./h_wminus.Integral())
    
    leg = ROOT.TLegend(.20, 0.70, .5, .88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.040)
    leg.SetHeader(header)

    
    h_wplus.SetLineColor(ROOT.kRed)
    h_wplus.SetFillColor(0)
    h_wplus.SetLineWidth(2)
    
    h_wminus.SetLineColor(ROOT.kBlue)
    h_wminus.SetFillColor(0)
    h_wminus.SetLineWidth(2)   
    
    leg.AddEntry(h_wplus, "W^{#plus} #rightarrow #mu^{#plus}#nu", "L")
    leg.AddEntry(h_wminus, "W^{#minus} #rightarrow #mu^{#minus}#nu", "L")

    h_ratio = h_wplus.Clone("h_ratio")
    h_ratio.Divide(h_wminus)
    h_ratio.SetMarkerStyle(20)
    h_ratio.SetMarkerSize(0.7)
    h_ratio.SetMarkerColor(ROOT.kBlack)
    h_ratio.SetLineColor(ROOT.kBlack)
    h_ratio.SetLineWidth(1)

    
    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio(line=1)
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
    
    h_wplus.Draw("PE SAME")
    h_wminus.Draw("HIST SAME")
    leg.Draw()
    plotter.auxRatio()
    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.Draw("HIST")
    dummyL.Draw("SAME")
    h_ratio.Draw("P SAME")
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s.png" % (outDir, tag))
    canvas.SaveAs("%s/%s.pdf" % (outDir, tag))

    dummyB.Delete()
    dummyT.Delete()
    padT.Delete()
    padB.Delete()






    ########################
    ## recoil
    ########################
    cfg['xtitle'] = "|U| (Gev)"
    cfg['xmin'], cfg['xmax'] = 0, 200
    tag = "wplusminus_magn"
    b_wplus = readProc(datagroups_W, "recoil_corr_xy_magn_gen", "WplusJetsToMuNu")
    b_wminus = readProc(datagroups_W, "recoil_corr_xy_magn_gen", "WminusJetsToMuNu")
    h_wplus = narf.hist_to_root(b_wplus)
    h_wminus = narf.hist_to_root(b_wminus)
    
    h_wplus.Scale(1./h_wplus.Integral())
    h_wminus.Scale(1./h_wminus.Integral())
    
    leg = ROOT.TLegend(.20, 0.70, .5, .88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.040)
    leg.SetHeader(header)

    
    h_wplus.SetLineColor(ROOT.kRed)
    h_wplus.SetFillColor(0)
    h_wplus.SetLineWidth(2)
    
    h_wminus.SetLineColor(ROOT.kBlue)
    h_wminus.SetFillColor(0)
    h_wminus.SetLineWidth(2)   
    
    leg.AddEntry(h_wplus, "W^{#plus} #rightarrow #mu^{#plus}#nu", "L")
    leg.AddEntry(h_wminus, "W^{#minus} #rightarrow #mu^{#minus}#nu", "L")

    h_ratio = h_wplus.Clone("h_ratio")
    h_ratio.Divide(h_wminus)
    h_ratio.SetMarkerStyle(20)
    h_ratio.SetMarkerSize(0.7)
    h_ratio.SetMarkerColor(ROOT.kBlack)
    h_ratio.SetLineColor(ROOT.kBlack)
    h_ratio.SetLineWidth(1)

    
    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio(line=1)
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
    
    h_wplus.Draw("PE SAME")
    h_wminus.Draw("HIST SAME")
    leg.Draw()
    plotter.auxRatio()
    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.Draw("HIST")
    dummyL.Draw("SAME")
    h_ratio.Draw("P SAME")
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s.png" % (outDir, tag))
    canvas.SaveAs("%s/%s.pdf" % (outDir, tag))

    dummyB.Delete()
    dummyT.Delete()
    padT.Delete()
    padB.Delete()




    ########################
    ## MET
    ########################
    cfg['xtitle'] = "E_{T}^{miss} (Gev)"
    cfg['xmin'], cfg['xmax'] = 0, 100
    tag = "wplusminus_met"
    b_wplus = readProc(datagroups_W, "MET_corr_xy_pt", "WplusJetsToMuNu")
    b_wminus = readProc(datagroups_W, "MET_corr_xy_pt", "WminusJetsToMuNu")
    h_wplus = narf.hist_to_root(b_wplus)
    h_wminus = narf.hist_to_root(b_wminus)
    
    h_wplus.Rebin(2)
    h_wminus.Rebin(2)
    h_wplus.Scale(1./h_wplus.Integral())
    h_wminus.Scale(1./h_wminus.Integral())
    
    
    leg = ROOT.TLegend(.20, 0.70, .5, .88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.040)
    leg.SetHeader(header)

    
    h_wplus.SetLineColor(ROOT.kRed)
    h_wplus.SetFillColor(0)
    h_wplus.SetLineWidth(2)
    
    h_wminus.SetLineColor(ROOT.kBlue)
    h_wminus.SetFillColor(0)
    h_wminus.SetLineWidth(2)   
    
    leg.AddEntry(h_wplus, "W^{#plus} #rightarrow #mu^{#plus}#nu", "L")
    leg.AddEntry(h_wminus, "W^{#minus} #rightarrow #mu^{#minus}#nu", "L")

    h_ratio = h_wplus.Clone("h_ratio")
    h_ratio.Divide(h_wminus)
    h_ratio.SetMarkerStyle(20)
    h_ratio.SetMarkerSize(0.7)
    h_ratio.SetMarkerColor(ROOT.kBlack)
    h_ratio.SetLineColor(ROOT.kBlack)
    h_ratio.SetLineWidth(1)

    
    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio(line=1)
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
    
    h_wplus.Draw("PE SAME")
    h_wminus.Draw("HIST SAME")
    leg.Draw()
    plotter.auxRatio()
    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.Draw("HIST")
    dummyL.Draw("SAME")
    h_ratio.Draw("P SAME")
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s.png" % (outDir, tag))
    canvas.SaveAs("%s/%s.pdf" % (outDir, tag))

    dummyB.Delete()
    dummyT.Delete()
    padT.Delete()
    padB.Delete()


def compareWZ(charge="plus"):

    header = "W^{#pm} #rightarrow #mu^{#pm}#nu, %s" % met
    yRatio = 1.3
    cfg = {

        'logy'              : True,
        'logx'              : False,
    
        'xmin'              : -100,
        'xmax'              : 100,
        'ymin'              : 1e-6,
        'ymax'              : 100,
        
        'xtitle'            : "U_{#parallel} + q_{T} (Gev)",
        'ytitle'            : "Events / 1 GeV",
        
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
        
        'ratiofraction'     : 0.25,
        'ytitleR'           : "W^{#plus}/W^{#minus}",
        'yminR'             : 1-(yRatio-1),
        'ymaxR'             : yRatio,
    }
    
    ########################
    ## para_qT
    ########################
    tag = "wz_%s_para_qT" % charge
    b_Z = readProc(datagroups_Z, "recoil_corr_xy_para_qT_gen", "DYmumu") 
    b_W_uncorr = readProc(datagroups_W, "recoil_corr_xy_para_qT_gen", "W%sJetsToMuNu" % charge)
    b_W_corr = readProc(datagroups_W, "recoil_corr_wz_para_qT", "W%sJetsToMuNu" % charge)
    h_target = narf.hist_to_root(b_Z)
    h_uncorr = narf.hist_to_root(b_W_uncorr)
    h_corr = narf.hist_to_root(b_W_corr)
    
    h_target.Scale(1./h_target.Integral())
    h_uncorr.Scale(1./h_uncorr.Integral())
    h_corr.Scale(1./h_corr.Integral())

    leg = ROOT.TLegend(.20, 0.65, .5, .88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.040)
    leg.SetHeader(header)

    h_target.SetLineColor(ROOT.kBlack)
    h_target.SetMarkerStyle(20)
    h_target.SetMarkerColor(ROOT.kBlack)
    h_target.SetLineColor(ROOT.kBlack)
    
    h_uncorr.SetLineColor(ROOT.kRed)
    h_uncorr.SetFillColor(0)
    h_uncorr.SetLineWidth(2)
    
    h_corr.SetLineColor(ROOT.kBlue)
    h_corr.SetFillColor(0)
    h_corr.SetLineWidth(2)   
    
    leg.AddEntry(h_target, "Z #rightarrow #mu^{#plus}#mu^{#minus}, target", "PE")
    leg.AddEntry(h_uncorr, "W^{#%s} #rightarrow #mu^{#plus}#nu, uncorrected" % charge, "LP")
    leg.AddEntry(h_corr, "W^{#%s} #rightarrow #mu^{#plus}#nu, corrected" % charge, "LP")

    h_uncorr_ratio = h_uncorr.Clone("h_uncorr_ratio")
    h_uncorr_ratio.Divide(h_target)
    h_uncorr_ratio.SetMarkerStyle(20)
    h_uncorr_ratio.SetMarkerSize(0.7)
    h_uncorr_ratio.SetMarkerColor(ROOT.kRed)
    h_uncorr_ratio.SetLineColor(ROOT.kRed)
    h_uncorr_ratio.SetLineWidth(1)

    
    h_corr_ratio = h_corr.Clone("h_corr_ratio")
    h_corr_ratio.Divide(h_target)
    h_corr_ratio.SetMarkerStyle(20)
    h_corr_ratio.SetMarkerSize(0.7)
    h_corr_ratio.SetMarkerColor(ROOT.kBlue)
    h_corr_ratio.SetLineColor(ROOT.kBlue)
    h_corr_ratio.SetLineWidth(1)
    
    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio(line=1)
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
    
    h_target.Draw("PE SAME")
    h_uncorr.Draw("HIST SAME")
    h_corr.Draw("PE SAME")
    leg.Draw()
    plotter.auxRatio()
    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.Draw("HIST")
    dummyL.Draw("SAME")
    h_uncorr_ratio.Draw("P SAME")
    h_corr_ratio.Draw("P SAME")
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s.png" % (outDir, tag))
    canvas.SaveAs("%s/%s.pdf" % (outDir, tag))

    dummyB.Delete()
    dummyT.Delete()
    padT.Delete()
    padB.Delete()
    
    
    
    
    ########################
    ## perp
    ########################
    cfg['xtitle'] = "U_{#perp}  (Gev)"
    tag = "wz_%s_perp" % charge
    b_Z = readProc(datagroups_Z, "recoil_corr_xy_perp_gen", "DYmumu") 
    b_W_uncorr = readProc(datagroups_W, "recoil_corr_xy_perp_gen", "W%sJetsToMuNu" % charge)
    b_W_corr = readProc(datagroups_W, "recoil_corr_wz_perp", "W%sJetsToMuNu" % charge)
    h_target = narf.hist_to_root(b_Z)
    h_uncorr = narf.hist_to_root(b_W_uncorr)
    h_corr = narf.hist_to_root(b_W_corr)
    
    h_target.Scale(1./h_target.Integral())
    h_uncorr.Scale(1./h_uncorr.Integral())
    h_corr.Scale(1./h_corr.Integral())

    leg = ROOT.TLegend(.20, 0.65, .5, .88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.040)
    leg.SetHeader(header)

    h_target.SetLineColor(ROOT.kBlack)
    h_target.SetMarkerStyle(20)
    h_target.SetMarkerColor(ROOT.kBlack)
    h_target.SetLineColor(ROOT.kBlack)
    
    h_uncorr.SetLineColor(ROOT.kRed)
    h_uncorr.SetFillColor(0)
    h_uncorr.SetLineWidth(2)
    
    h_corr.SetLineColor(ROOT.kBlue)
    h_corr.SetFillColor(0)
    h_corr.SetLineWidth(2)   
    
    leg.AddEntry(h_target, "Z #rightarrow #mu^{#plus}#mu^{#minus}, target", "PE")
    leg.AddEntry(h_uncorr, "W^{#%s} #rightarrow #mu^{#plus}#nu, uncorrected" % charge, "LP")
    leg.AddEntry(h_corr, "W^{#%s} #rightarrow #mu^{#plus}#nu, corrected" % charge, "LP")

    h_uncorr_ratio = h_uncorr.Clone("h_uncorr_ratio")
    h_uncorr_ratio.Divide(h_target)
    h_uncorr_ratio.SetMarkerStyle(20)
    h_uncorr_ratio.SetMarkerSize(0.7)
    h_uncorr_ratio.SetMarkerColor(ROOT.kRed)
    h_uncorr_ratio.SetLineColor(ROOT.kRed)
    h_uncorr_ratio.SetLineWidth(1)

    
    h_corr_ratio = h_corr.Clone("h_corr_ratio")
    h_corr_ratio.Divide(h_target)
    h_corr_ratio.SetMarkerStyle(20)
    h_corr_ratio.SetMarkerSize(0.7)
    h_corr_ratio.SetMarkerColor(ROOT.kBlue)
    h_corr_ratio.SetLineColor(ROOT.kBlue)
    h_corr_ratio.SetLineWidth(1)
    
    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio(line=1)
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
    
    h_target.Draw("PE SAME")
    h_uncorr.Draw("HIST SAME")
    h_corr.Draw("PE SAME")
    leg.Draw()
    plotter.auxRatio()
    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.Draw("HIST")
    dummyL.Draw("SAME")
    h_uncorr_ratio.Draw("P SAME")
    h_corr_ratio.Draw("P SAME")
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s.png" % (outDir, tag))
    canvas.SaveAs("%s/%s.pdf" % (outDir, tag))

    dummyB.Delete()
    dummyT.Delete()
    padT.Delete()
    padB.Delete()





    ########################
    ## recoil
    ########################
    cfg['xtitle'] = "|U| (Gev)"
    cfg['xmin'], cfg['xmax'] = 0, 200
    tag = "wz_%s_magn" % charge
    b_Z = readProc(datagroups_Z, "recoil_corr_xy_magn", "DYmumu") 
    b_W_uncorr = readProc(datagroups_W, "recoil_corr_xy_magn_gen", "W%sJetsToMuNu" % charge)
    b_W_corr = readProc(datagroups_W, "recoil_corr_wz_magn", "W%sJetsToMuNu" % charge)
    h_target = narf.hist_to_root(b_Z)
    h_uncorr = narf.hist_to_root(b_W_uncorr)
    h_corr = narf.hist_to_root(b_W_corr)
    
    h_target.Scale(1./h_target.Integral())
    h_uncorr.Scale(1./h_uncorr.Integral())
    h_corr.Scale(1./h_corr.Integral())

    leg = ROOT.TLegend(.20, 0.65, .5, .88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.040)
    leg.SetHeader(header)

    h_target.SetLineColor(ROOT.kBlack)
    h_target.SetMarkerStyle(20)
    h_target.SetMarkerColor(ROOT.kBlack)
    h_target.SetLineColor(ROOT.kBlack)
    
    h_uncorr.SetLineColor(ROOT.kRed)
    h_uncorr.SetFillColor(0)
    h_uncorr.SetLineWidth(2)
    
    h_corr.SetLineColor(ROOT.kBlue)
    h_corr.SetFillColor(0)
    h_corr.SetLineWidth(2)   
    
    leg.AddEntry(h_target, "Z #rightarrow #mu^{#plus}#mu^{#minus}, target", "PE")
    leg.AddEntry(h_uncorr, "W^{#%s} #rightarrow #mu^{#plus}#nu, uncorrected" % charge, "LP")
    leg.AddEntry(h_corr, "W^{#%s} #rightarrow #mu^{#plus}#nu, corrected" % charge, "LP")

    h_uncorr_ratio = h_uncorr.Clone("h_uncorr_ratio")
    h_uncorr_ratio.Divide(h_target)
    h_uncorr_ratio.SetMarkerStyle(20)
    h_uncorr_ratio.SetMarkerSize(0.7)
    h_uncorr_ratio.SetMarkerColor(ROOT.kRed)
    h_uncorr_ratio.SetLineColor(ROOT.kRed)
    h_uncorr_ratio.SetLineWidth(1)

    
    h_corr_ratio = h_corr.Clone("h_corr_ratio")
    h_corr_ratio.Divide(h_target)
    h_corr_ratio.SetMarkerStyle(20)
    h_corr_ratio.SetMarkerSize(0.7)
    h_corr_ratio.SetMarkerColor(ROOT.kBlue)
    h_corr_ratio.SetLineColor(ROOT.kBlue)
    h_corr_ratio.SetLineWidth(1)
    
    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio(line=1)
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
    
    h_target.Draw("PE SAME")
    h_uncorr.Draw("HIST SAME")
    h_corr.Draw("PE SAME")
    leg.Draw()
    plotter.auxRatio()
    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.Draw("HIST")
    dummyL.Draw("SAME")
    h_uncorr_ratio.Draw("P SAME")
    h_corr_ratio.Draw("P SAME")
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s.png" % (outDir, tag))
    canvas.SaveAs("%s/%s.pdf" % (outDir, tag))

    dummyB.Delete()
    dummyT.Delete()
    padT.Delete()
    padB.Delete()




    ########################
    ## MET
    ########################
    cfg['xtitle'] = "E_{T}^{miss} (Gev)"
    cfg['xmin'], cfg['xmax'] = 0, 100
    tag = "wz_%s_met" % charge
    b_Z = readProc(datagroups_Z, "MET_corr_xy_pt", "DYmumu") 
    b_W_uncorr = readProc(datagroups_W, "MET_corr_xy_pt", "W%sJetsToMuNu" % charge)
    b_W_corr = readProc(datagroups_W, "MET_corr_wz_pt", "W%sJetsToMuNu" % charge)
    h_target = narf.hist_to_root(b_Z)
    h_uncorr = narf.hist_to_root(b_W_uncorr)
    h_corr = narf.hist_to_root(b_W_corr)
    
    h_target.Scale(1./h_target.Integral())
    h_uncorr.Scale(1./h_uncorr.Integral())
    h_corr.Scale(1./h_corr.Integral())

    leg = ROOT.TLegend(.20, 0.65, .5, .88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.040)
    leg.SetHeader(header)

    h_target.SetLineColor(ROOT.kBlack)
    h_target.SetMarkerStyle(20)
    h_target.SetMarkerColor(ROOT.kBlack)
    h_target.SetLineColor(ROOT.kBlack)
    
    h_uncorr.SetLineColor(ROOT.kRed)
    h_uncorr.SetFillColor(0)
    h_uncorr.SetLineWidth(2)
    
    h_corr.SetLineColor(ROOT.kBlue)
    h_corr.SetFillColor(0)
    h_corr.SetLineWidth(2)   
    
    leg.AddEntry(h_target, "Z #rightarrow #mu^{#plus}#mu^{#minus}, target", "PE")
    leg.AddEntry(h_uncorr, "W^{#%s} #rightarrow #mu^{#plus}#nu, uncorrected" % charge, "LP")
    leg.AddEntry(h_corr, "W^{#%s} #rightarrow #mu^{#plus}#nu, corrected" % charge, "LP")

    h_uncorr_ratio = h_uncorr.Clone("h_uncorr_ratio")
    h_uncorr_ratio.Divide(h_target)
    h_uncorr_ratio.SetMarkerStyle(20)
    h_uncorr_ratio.SetMarkerSize(0.7)
    h_uncorr_ratio.SetMarkerColor(ROOT.kRed)
    h_uncorr_ratio.SetLineColor(ROOT.kRed)
    h_uncorr_ratio.SetLineWidth(1)

    
    h_corr_ratio = h_corr.Clone("h_corr_ratio")
    h_corr_ratio.Divide(h_target)
    h_corr_ratio.SetMarkerStyle(20)
    h_corr_ratio.SetMarkerSize(0.7)
    h_corr_ratio.SetMarkerColor(ROOT.kBlue)
    h_corr_ratio.SetLineColor(ROOT.kBlue)
    h_corr_ratio.SetLineWidth(1)
    
    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio(line=1)
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
    
    h_target.Draw("PE SAME")
    h_uncorr.Draw("HIST SAME")
    h_corr.Draw("PE SAME")
    leg.Draw()
    plotter.auxRatio()
    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.Draw("HIST")
    dummyL.Draw("SAME")
    h_uncorr_ratio.Draw("P SAME")
    h_corr_ratio.Draw("P SAME")
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s.png" % (outDir, tag))
    canvas.SaveAs("%s/%s.pdf" % (outDir, tag))

    dummyB.Delete()
    dummyT.Delete()
    padT.Delete()
    padB.Delete()



def readProc(groups, hName, procName):

    label = "%s_%s" % (hName, procName)
    groups.setHists(hName, "", label=label, procsToRead=[procName], selectSignal=False)
    bhist = groups.groups[procName][label]
    return bhist



    

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
        
    if flavor == "mu": 
        label = "W #rightarrow #mu^{#pm}, %s" % met
        sig = ["WminusJetsToMuNu", "WplusJetsToMuNu"]
        data = "SingleMuon"
        

    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/W%s_%s/WZRecoilCorrection/" % (flavor, met)
    functions.prepareDir(outDir, False)
    
    
    ## load Z
    print("load z")
    flavor = "mumu"
    datagroups_Z = Datagroups("lowPU_%s_%s.pkl.lz4" % (flavor, met))
    
    
    ## load W
    print("load w")
    flavor = "mu"
    datagroups_W = Datagroups("lowPU_%s_%s.pkl.lz4" % (flavor, met))
    print("done")
    
    
    compareWplusMinus()
    compareWZ("plus")
    compareWZ("minus")
    sys.exit()
    
    b_Z = readProc(datagroups_Z, "recoil_corr_xy_para_qT_gen", "DYmumu") 
    
    
    
    b_W_uncorr = readProc(datagroups_W, "recoil_corr_xy_para_qT_gen", "WplusJetsToMuNu")
    b_W_corr = readProc(datagroups_W, "recoil_corr_wz_para_qT", "WplusJetsToMuNu")
    
    h_Z = narf.hist_to_root(b_Z)
    h_W_uncorr = narf.hist_to_root(b_W_uncorr)
    h_W_corr = narf.hist_to_root(b_W_corr)
    

    #doPlot("perp", h_Z, h_W_uncorr, h_W_corr, xMin=-100, xMax=100, label=label)    
    doPlot("para_qT", h_Z, h_W_uncorr, h_W_corr, xMin=-100, xMax=100, label=label, xLabel="Recoil U_{#parallel}+q_{T} (GeV)")
    
    
    sys.exit()


    

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
    

    
    
   
    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/Comparison/"
    dataName = "SingleMuon"
    signalName = "DYmumu"


    
    

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
    
   