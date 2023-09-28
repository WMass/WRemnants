
import sys,array,math,os,copy,json
import numpy as np

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

import functions
import plotter

import lz4.frame
import pickle
import narf
import numpy as np

from wremnants.datasets import datagroups


def makePlot(hist_data, hist_mc, fOut, xLabel, npv, outDir_):

    hist_data.Scale(1./hist_data.Integral())
    hist_mc.Scale(1./hist_mc.Integral())
    
    hist_data.SetLineColor(ROOT.kBlack)
    hist_data.SetLineWidth(2)
    
    hist_mc.SetLineColor(ROOT.kRed)
    hist_mc.SetLineWidth(2)
    
    ## sigmas
    cfg = {

        'logy'              : True,
        'logx'              : False,
        
        'xmin'              : -100,
        'xmax'              : 100,
        'ymin'              : 1e-5,
        'ymax'              : 1e0,
            
        'xtitle'            : xLabel,
        'ytitle'            : "Events",
            
        'topRight'          : lumi_header, 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    } 
    

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy()   
    canvas.cd()
    dummy.Draw("HIST")
    hist_data.Draw("SAME")
    hist_mc.Draw("SAME")
    plotter.aux()
    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.030)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    
    latex.DrawLatex(0.20, 0.90, "NPV = %d" % npv)
    
    latex.DrawLatex(0.20, 0.85, "Data")
    latex.DrawLatex(0.20, 0.82, "Mean %.2f #pm %.2f GeV" % (hist_data.GetMean(), hist_data.GetMeanError()))
    
    latex.DrawLatex(0.60, 0.85, "#color[2]{MC}")
    latex.DrawLatex(0.60, 0.82, "#color[2]{Mean %.2f #pm %.2f GeV}" % (hist_mc.GetMean(), hist_mc.GetMeanError()))
    
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s.png" % (outDir_, fOut))
    canvas.SaveAs("%s/%s.pdf" % (outDir_, fOut))
    canvas.Delete()


def makePlot_fit(hist_data, hist_mc, fOut, xLabel, npv, outDir_):

    hist_data.Scale(1./hist_data.Integral())
    hist_mc.Scale(1./hist_mc.Integral())
    
    hist_data.SetLineColor(ROOT.kBlack)
    hist_data.SetLineWidth(2)
    
    hist_mc.SetLineColor(ROOT.kRed)
    hist_mc.SetLineWidth(2)
    
    ## sigmas
    cfg = {

        'logy'              : True,
        'logx'              : False,
        
        'xmin'              : -100,
        'xmax'              : 100,
        'ymin'              : 1e-5,
        'ymax'              : 1e0,
            
        'xtitle'            : xLabel,
        'ytitle'            : "Events",
            
        'topRight'          : lumi_header, 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    } 
    
    fit_mc = ROOT.TF1("fit_mc", "gaus", -100, 100)
    hist_mc.Fit("fit_mc")
    
    fit_data = ROOT.TF1("fit_data", "gaus", -100, 100)
    hist_data.Fit("fit_data")

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy()   
    canvas.cd()
    dummy.Draw("HIST")
    hist_data.Draw("SAME")
    hist_mc.Draw("SAME")
    
    fit_mc.Draw("SAME")
    fit_data.Draw("SAME")
    plotter.aux()
    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.030)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    
    latex.DrawLatex(0.20, 0.90, "NPV = %d" % npv)
    
    latex.DrawLatex(0.20, 0.85, "Data")
    latex.DrawLatex(0.20, 0.82, "Mean %.2f #pm %.2f GeV" % (hist_data.GetMean(), hist_data.GetMeanError()))
    latex.DrawLatex(0.20, 0.79, "#color[2]{Mean %.2f #pm %.2f GeV}" % (fit_data.GetParameter(1), fit_data.GetParError(1)))
    
    latex.DrawLatex(0.60, 0.85, "#color[2]{MC}")
    latex.DrawLatex(0.60, 0.82, "#color[2]{Mean %.2f #pm %.2f GeV}" % (hist_mc.GetMean(), hist_mc.GetMeanError()))
    latex.DrawLatex(0.60, 0.79, "#color[2]{Mean %.2f #pm %.2f GeV}" % (fit_mc.GetParameter(1), fit_mc.GetParError(1)))
    
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s.png" % (outDir_, fOut))
    canvas.SaveAs("%s/%s.pdf" % (outDir_, fOut))
    canvas.Delete()
    
    return fit_data.GetParameter(1), fit_data.GetParError(1), fit_mc.GetParameter(1), fit_mc.GetParError(1)

def METxyCorrection(direction = "x", corrType="uncorr", polyOrderData=-1, polyOrderMC=-1, procs=[], data="", yMin=-5, yMax=5):

    # corrType uncorr or corr_xy
    outDir_ = "%s/%s" % (outDir, corrType)
    functions.prepareDir(outDir_, False)

    
    b_data = functions.readBoostHistProc(groups, "MET%s_%s_npv" % (direction, corrType), [data])
    b_mc = functions.readBoostHistProc(groups, "MET%s_%s_npv" % (direction, corrType), procs)

    h_data = narf.hist_to_root(b_data)
    h_mc = narf.hist_to_root(b_mc)
    
    g_data = ROOT.TGraphErrors()
    g_data.SetLineColor(ROOT.kBlack)
    g_data.SetMarkerStyle(20)
    g_data.SetMarkerSize(1)
    g_data.SetMarkerColor(ROOT.kBlack)
    
    g_mc = ROOT.TGraphErrors()
    g_mc.SetLineColor(ROOT.kRed)
    g_mc.SetMarkerStyle(21)
    g_mc.SetMarkerSize(1)
    g_mc.SetMarkerColor(ROOT.kRed)
    
    outDict = {"data": { 'nominal': {} }, "mc": { 'nominal': {} }}
    for iBin in range(1, npv_max):

        hist_data = h_data.ProjectionY("data_bin%d" % iBin, iBin, iBin)
        hist_mc = h_mc.ProjectionY("mc_bin%d" % iBin, iBin, iBin)
        
        mean_data, mean_data_err = hist_data.GetMean(), hist_data.GetMeanError()
        mean_mc, mean_mc_err = hist_mc.GetMean(), hist_mc.GetMeanError()
        print(iBin, mean_mc, mean_mc_err)
        
        makePlot(hist_data, hist_mc, "npv_%d_%d_MET%s" % (iBin, iBin, direction), "MET %s (GeV)" % direction, iBin, outDir_)
        #mean_data, mean_data_err, mean_mc, mean_mc_err = makePlot_fit(hist_data, hist_mc, "npv_%d_%d_MET%s" % (iBin, iBin, direction), "MET %s (GeV)" % direction, iBin, outDir_)
        
        g_data.SetPoint(iBin-1, iBin, mean_data)
        g_data.SetPointError(iBin-1, 0, mean_data_err)
        g_mc.SetPoint(iBin-1, iBin, mean_mc)
        g_mc.SetPointError(iBin-1, 0, mean_mc_err)
  
    if polyOrderData > 0:
        fit_data = ROOT.TF1("fit_data_%s" % direction, "pol%d" % polyOrderData, 0, npv_max)
        result = g_data.Fit(fit_data.GetName(), "NSE", "", npv_fit_min, npv_fit_max) 
        fit_data.SetLineColor(ROOT.kBlack)
        fit_data.GetXaxis().SetRangeUser(0, npv_max)
        fit_data.SetLineWidth(2)
        for iP in range(0, polyOrderData+1): outDict['data']['nominal']['p%d' % iP] = fit_data.GetParameter(iP)
        outDict['data']['nominal']['polyOrder'] = polyOrderData
    
    if polyOrderMC > 0:
        fit_mc = ROOT.TF1("fit_mc_%s" % direction, "pol%d" % polyOrderMC, 0, npv_max)
        result = g_mc.Fit(fit_mc.GetName(), "NSE", "", npv_fit_min, npv_fit_max) 
        fit_mc.SetLineColor(ROOT.kRed)
        fit_mc.SetLineStyle(ROOT.kDashed)
        fit_mc.GetXaxis().SetRangeUser(0, npv_max)
        fit_mc.SetLineWidth(2)
        for iP in range(0, polyOrderMC+1): outDict['mc']['nominal']['p%d' % iP] = fit_mc.GetParameter(iP)
        outDict['mc']['nominal']['polyOrder'] = polyOrderMC
  
    ## sigmas
    cfg = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : npv_max,
        'ymin'              : yMin,
        'ymax'              : yMax,
            
        'xtitle'            : "Number of primary vertices",
        'ytitle'            : "#LT MET_{%s} #GT (Gev)" % direction,
            
        'topRight'          : lumi_header, 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    } 
    

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy()   
    canvas.cd()
    dummy.Draw("HIST")
    g_data.Draw("PE SAME")
    if polyOrderData > 0: fit_data.Draw("L SAME")
    g_mc.Draw("PE SAME")
    if polyOrderMC > 0: fit_mc.Draw("L SAME")
    plotter.aux()
    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    
    if flavor == "mumu": label = "Z #rightarrow #mu^{#plus}#mu^{#minus}"
    elif flavor == "ee": label = "Z #rightarrow e^{#plus}e^{#minus}"
    elif flavor == "mu": label = "W^{#pm} #rightarrow #mu^{#pm}#nu"
    elif flavor == "e": label = "W^{#pm} #rightarrow e^{#pm}#nu"
    else: label = ""
    latex.DrawLatex(0.20, 0.90, label)
    latex.DrawLatex(0.20, 0.85, "#LT MET_{%s} #GT, %s" % (direction, met))

    leg = ROOT.TLegend(.2, 0.7, .5, .80)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)
    leg.AddEntry(g_data, "Data", "LP")
    leg.AddEntry(g_mc, "MC", "LP")
    leg.Draw("SAME")
    
    line = ROOT.TLine(0, 0, npv_max, 0)
    line.SetLineColor(ROOT.kBlue+2)
    line.SetLineWidth(2)
    line.Draw("SAME")
    
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/MET_%s_npv.png" % (outDir_, direction))
    canvas.SaveAs("%s/MET_%s_npv.pdf" % (outDir_, direction))
    canvas.Delete()
    
    return outDict
    
    
if __name__ == "__main__":

    met = "DeepMETReso" # PFMET, RawPFMET DeepMETReso
    flavor = "mumu" # mu, e, mumu, ee
    lowPU = False

    # DATA For electron channels!

    ####################################################################
    if lowPU:
        npv_max, npv_fit_min, npv_fit_max = 10, 0, 10
        lumi_header = "199 pb^{#minus1} (13 TeV)"
        
        groups = datagroupsLowPU("lowPU_%s_%s.pkl.lz4" % (flavor, met), flavor=flavor)
        procs = ['EWK', 'Top', 'Zmumu'] 
        data = "SingleMuon" if "mu" in flavor else "SingleElectron"

        outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/METxy_correction/METxy_%s_%s/" % (flavor, met)
        fOut = "wremnants/data/recoil/lowPU/%s_%s/met_xy_correction.json" % (flavor, met)
        functions.prepareDir(outDir, True)
        
        dictout = {}
        dictX = METxyCorrection(direction="x", corrType="corr_lep", polyOrderData=1, polyOrderMC=1, procs=procs, data=data)
        dictY = METxyCorrection(direction="y", corrType="corr_lep", polyOrderData=1, polyOrderMC=1, procs=procs, data=data)
        
        
        dictout['x'] = dictX
        dictout['y'] = dictY
        jsOut = json.dumps(dictout, indent = 4)
        with open(fOut, "w") as outfile: outfile.write(jsOut)
        os.system("cp %s %s" % (fOut, outDir)) # make copy to web dir

        
        METxyCorrection(direction="x", corrType="corr_xy", polyOrderData=1, polyOrderMC=1, procs=procs, data=data)
        METxyCorrection(direction="y", corrType="corr_xy", polyOrderData=1, polyOrderMC=1, procs=procs, data=data)
    
    else:
        npv_max, npv_fit_min, npv_fit_max = 60, 5, 55
        lumi_header = "16.8 fb^{#minus1} (13 TeV)"

        if flavor == "mumu":
            npv_max, npv_fit_min, npv_fit_max = 60, 5, 55
            polyOrderDataX, polyOrderMCX = 3, 3
            polyOrderDataY, polyOrderMCY = 3, 3
            groups = datagroups.Datagroups(f"mz_wlike_with_mu_eta_pt_{met}.hdf5")
            procs = ["Zmumu", "Ztautau", "Other"]
            data = "Data"
        else:
            npv_max, npv_fit_min, npv_fit_max = 60, 0, 55
            polyOrderDataX, polyOrderMCX = 3, 3
            polyOrderDataY, polyOrderMCY = 6, 3
            datagroups = datagroups2016(f"mw_with_mu_eta_pt_{met}.hdf5")
            procs = ["Zmumu", "Ztautau", "Wtaunu", "Wmunu", "Top", "Diboson"]
            data = "Data"
            
            for g in datagroups.groups:
                datagroups.groups[g]['selectOp'] = None
            

        #outDir = "/eos/user/j/jaeyserm/www/wmass/highPU/METxy_correction/METxy_%s_%s/" % (flavor, met)
        outDir = f"/home/submit/jaeyserm/public_html/wmass/highPU/METxy_correction/METxy_{flavor}_{met}/"
        fOut = "wremnants-data/data/recoil/highPU/%s_%s/met_xy_correction.json" % (flavor, met)
        functions.prepareDir(outDir, True)
        
        dictout = {}
        dictX = METxyCorrection(direction="x", corrType="corr_lep", polyOrderData=polyOrderDataX, polyOrderMC=polyOrderMCX, procs=procs, data=data, yMin=-10, yMax=6)
        dictY = METxyCorrection(direction="y", corrType="corr_lep", polyOrderData=polyOrderDataY, polyOrderMC=polyOrderMCY, procs=procs, data=data)
    
    
        dictout['x'] = dictX
        dictout['y'] = dictY
        jsOut = json.dumps(dictout, indent = 4)
        with open(fOut, "w") as outfile: outfile.write(jsOut)
        
 
        METxyCorrection(direction="x", corrType="corr_xy", polyOrderData=polyOrderDataX, polyOrderMC=polyOrderMCX, procs=procs, data=data, yMin=-10, yMax=6)
        METxyCorrection(direction="y", corrType="corr_xy", polyOrderData=polyOrderDataY, polyOrderMC=polyOrderMCY, procs=procs, data=data)
        print(fOut)
