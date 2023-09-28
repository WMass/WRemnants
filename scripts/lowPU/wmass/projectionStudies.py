
import sys,array,math,os,copy,fnmatch
from collections import OrderedDict

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

sys.path.insert(0, "scripts/lowPU/")
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
    charge = histCfg['charge']
    
    label = "%s_%s" % (hName, procName)
    groups.setHists(hName, "", label=label, procsToRead=[procName], selectSignal=False)
    bhist = groups.groups[procName][label]
    
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
 


def mTcomparison():

    mT_bins = [0, 10, 15, 20, 25, 30, 35,] + list(range(40, 100, 2)) + [100, 102, 104, 106, 108, 110, 115, 120, 125, 130, 140, 160, 200]
    
    flavor = "mu"
    charge = "combined" # combined plus minus
    MC_SF = 1.0

    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/W%s/" % flavor
    
    histCfg = {"name": "mT_uncorr", "axis": "mt", "charge": charge }
    
    met = "DeepMETReso"
    groups = Datagroups("lowPU_%s_%s.pkl.lz4" % (flavor, met))
    hist_DeepMETReso = parseProc(groups, histCfg, "WJetsToMuNu", rebin=mT_bins)
    hist_DeepMETReso.SetLineColor(ROOT.kRed)
    hist_DeepMETReso.SetLineWidth(2)
    
    met = "RawPFMET"
    groups = Datagroups("lowPU_%s_%s.pkl.lz4" % (flavor, met))
    hist_RawPFMET = parseProc(groups, histCfg, "WJetsToMuNu", rebin=mT_bins)
    hist_RawPFMET.SetLineColor(ROOT.kBlue)
    hist_RawPFMET.SetLineWidth(2)

    cfg = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 40,
        'xmax'              : 120,
        'ymin'              : 0,
        'ymax'              : 7e4,
            
        'xtitle'            : "m_{T} (GeV)",
        'ytitle'            : "Events",
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    }
    
    
    leg = ROOT.TLegend(.20, 0.80, .55, .90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.040)
    leg.AddEntry(hist_RawPFMET, "PF MET", "L")
    leg.AddEntry(hist_DeepMETReso, "DeepMET resolution", "L")

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy()
    canvas.SetGrid()
    dummy.Draw("HIST")
    hist_RawPFMET.Draw("SAME HIST")
    hist_DeepMETReso.Draw("SAME HIST")
    leg.Draw("SAME")
    
    plotter.aux(canvas)  
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()  
    canvas.SaveAs("%s/mT_MET_comparison.png" % outDir)
    canvas.SaveAs("%s/mT_MET_comparison.pdf" % outDir)
    canvas.Close()



def eventYields():

    
    flavor = "mu"
    charge = "combined" # combined plus minus
    MC_SF = 1.0

    
    procs = ['EWK', 'TTbar', 'WplusJetsToMuNu', 'WminusJetsToMuNu'] 
    histCfg = {"name": "mT_uncorr", "axis": "mt", "charge": charge }
    
    for proc in procs:
    
        met = "DeepMETReso"
        groups = Datagroups("lowPU_%s_%s.pkl.lz4" % (flavor, met))
        hist_DeepMETReso = parseProc(groups, histCfg, proc)
        hist_DeepMETReso.SetLineColor(ROOT.kRed)
        hist_DeepMETReso.SetLineWidth(2)
        
        met = "RawPFMET"
        groups = Datagroups("lowPU_%s_%s.pkl.lz4" % (flavor, met))
        hist_RawPFMET = parseProc(groups, histCfg, proc)
        hist_RawPFMET.SetLineColor(ROOT.kBlue)
        hist_RawPFMET.SetLineWidth(2)
        
        print("%s \t %.3f \t %.3f" % (proc, hist_RawPFMET.Integral(), hist_DeepMETReso.Integral()))


def expUncPlotStat():

    lumis = ["1p0", "2p5", "5p0", "10p0"]
    lumis_fb = [0.2, 0.5, 1, 2]
    met = "RawPFMET" # 
    scale=100
    group = True
    
    mets = ["RawPFMET", "DeepMETReso"]
    mets_labels = ["PF MET", "DeepMET"]
    mets_colors = [ROOT.kBlue, ROOT.kRed]
    
    graphs = {}
    for j, met in enumerate(mets):
    
        g_tot = ROOT.TGraph()
        g_tot.SetLineColor(mets_colors[j])
        g_tot.SetMarkerColor(mets_colors[j])
        g_tot.SetLineWidth(2)
        g_tot.SetMarkerSize(1.5)
        g_tot.SetMarkerStyle(8)
        
        g_stat_data = ROOT.TGraph()
        #g_stat_data.SetLineColor(mets_colors[j])
        #g_stat_data.SetMarkerColor(mets_colors[j])
        #g_stat_data.SetLineWidth(2)
        #g_stat_data.SetLineStyle(ROOT.kDashDotted)
        #g_stat_data.SetMarkerSize(1.5)
        #g_stat_data.SetMarkerStyle(21)
        g_stat_data.SetLineColor(mets_colors[j])
        g_stat_data.SetMarkerColor(mets_colors[j])
        g_stat_data.SetLineWidth(2)
        g_stat_data.SetMarkerSize(1.5)
        g_stat_data.SetMarkerStyle(8)
        
        
        g_binbybin = ROOT.TGraph()
        g_binbybin.SetLineColor(mets_colors[j])
        g_binbybin.SetMarkerColor(mets_colors[j])
        g_binbybin.SetLineWidth(2)
        g_binbybin.SetLineStyle(ROOT.kDashed)
        g_binbybin.SetMarkerSize(1.5)
        g_binbybin.SetMarkerStyle(22)
        
        i = 0
        for k,lumi in enumerate(lumis):
        

            fIn = ROOT.TFile("/home/j/jaeyserm/combine/CMSSW_10_6_20/src/lowPU_Wmass/LowPU_Wmass_mu_%s_lumi%s.root" % (met, lumi))
            tree = fIn.Get("fitresults")
            tree.GetEntry(0)
            
            name = "nuisance_group_impact_nois" if group else "nuisance_impact_nois"
            impact_hist = fIn.Get(name) # nuisance_group_impact_nois
            xLabels = np.array([impact_hist.GetXaxis().GetBinLabel(i) for i in range(1, impact_hist.GetNbinsX()+1)]) # POI
            yLabels = np.array([impact_hist.GetYaxis().GetBinLabel(i) for i in range(1, impact_hist.GetNbinsY()+1)]) # nuisances
            
            unc_tot = scale*getattr(tree, "massShift100MeV_err")
            #print(unc_tot, lumis_fb[k])
            g_tot.SetPoint(i, lumis_fb[k], unc_tot)
            
            
            
            rounded=2
            impacts = np.array([round(np.mean(np.array([impact_hist.GetBinContent(k+1, i+1) for k,xLabel in enumerate(xLabels)]))*scale, rounded) for i,yLabel in enumerate(yLabels)])

            for l in range(0, len(yLabels)):
            
                if yLabels[l] == "stat": continue
                if yLabels[l] == "binByBinStat": 
                    g_binbybin.SetPoint(i, lumis_fb[k], impacts[l])
                    g_stat_data.SetPoint(i, lumis_fb[k], math.sqrt(unc_tot**2 - impacts[l]**2))
                print(yLabels[l], unc_tot, impacts[l], math.sqrt(unc_tot**2 - impacts[l]**2))
            
            i+=1
            
            
        graphs["%s_tot" % met] = g_tot
        graphs["%s_binbybin" % met] = g_binbybin
        graphs["%s_stat_data" % met] = g_stat_data

    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/Projection/"
    
    

    cfg = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : 2,
        'ymin'              : 2,
        'ymax'              : 20,
            
        'xtitle'            : "Integrated luminosity (fb^{#minus1})",
        'ytitle'            : "Uncertainty on m_{W} (MeV)",
            
        'topRight'          : "LowPU (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    }
    
    
    #leg = ROOT.TLegend(.40, 0.6, .90, .90)
    leg = ROOT.TLegend(.40, 0.75, .90, .90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)
    leg.SetHeader("W^{#pm} #rightarrow #mu^{#pm}#nu")

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy()
    canvas.SetGrid()
    dummy.Draw("HIST")
    
    '''
    for i,met in enumerate(mets):
        leg.AddEntry(graphs["%s_tot" % met], mets_labels[i] + " (stat. tot.)", "LP")
        graphs["%s_tot" % met].Draw("SAME LP")

    for i,met in enumerate(mets):
        leg.AddEntry(graphs["%s_binbybin" % met], mets_labels[i] + " (stat. MC)", "LP")
        graphs["%s_binbybin" % met].Draw("SAME LP")
    '''    
    for i,met in enumerate(mets):
        leg.AddEntry(graphs["%s_stat_data" % met], mets_labels[i] + " (stat. data)", "LP")
        graphs["%s_stat_data" % met].Draw("SAME LP")    

    leg.Draw("SAME")
    
    plotter.aux(canvas)  
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()  
    canvas.SaveAs("%s/stat_only.png" % outDir)
    canvas.SaveAs("%s/stat_only.pdf" % outDir)
    canvas.Close()



def expUncPlotSyst():

    lumis = ["1p0", "2p5", "5p0", "10p0"]
    lumis_fb = [0.2, 0.5, 1, 2]
    met = "RawPFMET"
    scale=100
    group = True
    mode = "binned"
    header = "W^{#pm} #rightarrow #mu^{#pm}#nu, RawPFMET"
    
    inTag = "eta_inclusive" # eta_inclusive eta_inclusive_scaledRecoil
    outTag = "scaledRecoil" # cteRecoil scaledRecoil
    
    systs = ["total", "datastat", "recoil_stat", "pdfNNPDF31", "QCDscale", "muonScale", "others"]
    systs_labels = ["Total", "Stat. data", "Stat. recoil", "PDF", "QCDscale", "Muon momentum scale (10^{-4})", "Others"]
    
    systs = [["total"], ["datastat"], ["recoil_stat", "recoil_syst"], ["pdfNNPDF31"], ["QCDscale"], ["CMS_Fakes_norm"], ["others"]]
    systs_labels = ["Total", "Stat. data", "Recoil tot. %s" % ("(stat. unscaled)" if not "scaled" in outTag else "(stat. scaled, #sqrt{N})"), "PDF", "QCDscale", "Fake (norm)", "Others"]
    systs_colors = [ROOT.kBlack, ROOT.kBlue, ROOT.kRed, ROOT.kGreen+1, ROOT.kOrange+1, ROOT.kGray+2, ROOT.kMagenta]
    
    graphs = {}
    for i, syst in enumerate(systs):
    
        g = ROOT.TGraph()
        g.SetLineColor(systs_colors[i])
        g.SetMarkerColor(systs_colors[i])
        g.SetLineWidth(2)
        g.SetMarkerSize(1.5)
        g.SetMarkerStyle(8)
        graphs[i] = g
        

    for i,lumi in enumerate(lumis):
    
        # data stat
        fIn = ROOT.TFile("/home/j/jaeyserm/combine/CMSSW_10_6_20/src/Wmass_mT/fits_scaledRecoil/lowPU_mu_%s_lumi%s_statOnly.root" % (met, lumi))
        tree = fIn.Get("fitresults")
        tree.GetEntry(0)
        unc_data_stat = getattr(tree, "massShift100MeV_err")
        fIn.Close()
        
        fIn = ROOT.TFile("/home/j/jaeyserm/combine/CMSSW_10_6_20/src/Wmass_mT/fits_scaledRecoil/lowPU_mu_%s_lumi%s.root" % (met, lumi))
        tree = fIn.Get("fitresults")
        tree.GetEntry(0)
        unc_total = getattr(tree, "massShift100MeV_err")
        
            
        name = "nuisance_group_impact_nois" if group else "nuisance_impact_nois"
        impact_hist = fIn.Get(name) # nuisance_group_impact_nois
        xLabels = np.array([impact_hist.GetXaxis().GetBinLabel(l) for l in range(1, impact_hist.GetNbinsX()+1)]) # POI
        yLabels = np.array([impact_hist.GetYaxis().GetBinLabel(l) for l in range(1, impact_hist.GetNbinsY()+1)]) # nuisances
        print(yLabels)
        for k, systs_ in enumerate(systs): 
            g = graphs[k]
            unc = 0
            for syst in systs_:
                
                if syst == "datastat": unc = unc_data_stat**2
                elif syst == "total": unc = unc_total**2
                elif syst in yLabels: 
                    unc += impact_hist.GetBinContent(1, int(np.where(yLabels == syst)[0][0])+1)**2
                elif syst == "others": # look for the other systs and add in quadrature
                    for l,yLabel in enumerate(yLabels):
                        if yLabel not in sum(systs, []):
                            unc += impact_hist.GetBinContent(1, int(np.where(yLabels == yLabel)[0][0])+1)**2
                            print("_> others", yLabel)
                else: 
                    print("ERROR Category for syst %s not found" % syst)
                    sys.exit()
            unc = unc**0.5
            g.SetPoint(i, lumis_fb[i], scale*unc)
      
    outDir = "/eos/user/j/jaeyserm/www/wmass/studies_mT/lowPU/"
    

    cfg = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : 2,
        'ymin'              : 0,
        'ymax'              : 40,
            
        'xtitle'            : "Integrated luminosity (fb^{#minus1})",
        'ytitle'            : "Uncertainty on m_{W} (MeV)",
            
        'topRight'          : "LowPU (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    }
    
    
    #leg = ROOT.TLegend(.40, 0.6, .90, .90)
    leg = ROOT.TLegend(.40, 0.55, .90, .90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)
    leg.SetHeader(header)

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy()
    canvas.SetGrid()
    dummy.Draw("HIST")
  
    for k, syst in enumerate(systs):  
        leg.AddEntry(graphs[k], systs_labels[k], "LP")
        graphs[k].Draw("SAME LP")    

    leg.Draw("SAME")
    
    plotter.aux(canvas)  
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()  
    canvas.SaveAs("%s/syst_%s.png" % (outDir, outTag))
    canvas.SaveAs("%s/syst_%s.pdf" % (outDir, outTag))
    canvas.Close()




def recoilUncPlot_statSyst():

    lumis = ["1p0", "2p5", "5p0", "10p0"]
    lumis_fb = [0.2, 0.5, 1, 2]
    met = "RawPFMET"
    scale=100
    group = True
    header = "W^{#pm} #rightarrow #mu^{#pm}#nu, RawPFMET"
    
   
    g_tot = ROOT.TGraph()
    g_tot.SetLineColor(ROOT.kBlack)
    g_tot.SetMarkerColor(ROOT.kBlack)
    g_tot.SetLineWidth(2)
    g_tot.SetMarkerSize(1.5)
    g_tot.SetMarkerStyle(8)
    
    g_stat = ROOT.TGraph()
    g_stat.SetLineColor(ROOT.kBlue)
    g_stat.SetMarkerColor(ROOT.kBlue)
    g_stat.SetLineWidth(2)
    g_stat.SetMarkerSize(1.5)
    g_stat.SetMarkerStyle(8)

    g_syst = ROOT.TGraph()
    g_syst.SetLineColor(ROOT.kRed)
    g_syst.SetMarkerColor(ROOT.kRed)
    g_syst.SetLineWidth(2)
    g_syst.SetMarkerSize(1.5)
    g_syst.SetMarkerStyle(8)

    for i,lumi in enumerate(lumis):
    
        name = "nuisance_group_impact_nois" if group else "nuisance_impact_nois"
    

        
        ## scaled recoil
        fIn = ROOT.TFile("/home/j/jaeyserm/combine/CMSSW_10_6_20/src/Wmass_mT/fits_scaledRecoil_recoilOnly/lowPU_mu_%s_lumi%s.root" % (met, lumi))
        impact_hist = fIn.Get(name) # nuisance_group_impact_nois
        yLabels = np.array([impact_hist.GetYaxis().GetBinLabel(l) for l in range(1, impact_hist.GetNbinsY()+1)]) # nuisances
        unc_stat = impact_hist.GetBinContent(1, int(np.where(yLabels == "recoil_stat")[0][0])+1)
        unc_syst = impact_hist.GetBinContent(1, int(np.where(yLabels == "recoil_syst")[0][0])+1)
        unc = (unc_stat**2 + unc_syst**2)**0.5
        g_tot.SetPoint(i, lumis_fb[i], scale*unc)
        g_stat.SetPoint(i, lumis_fb[i], scale*unc_stat)
        g_syst.SetPoint(i, lumis_fb[i], scale*unc_syst)
      
    outDir = "/eos/user/j/jaeyserm/www/wmass/studies_mT/lowPU/"
    

    cfg = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : 2,
        'ymin'              : 0,
        'ymax'              : 10,
            
        'xtitle'            : "Integrated luminosity (fb^{#minus1})",
        'ytitle'            : "Recoil uncertainty on m_{W} (MeV)",
            
        'topRight'          : "LowPU (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    }
    
    
    leg = ROOT.TLegend(.40, 0.70, .90, .90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)
    leg.SetHeader(header)

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy()
    canvas.SetGrid()
    dummy.Draw("HIST")
  
    leg.AddEntry(g_tot, "Total recoil uncertainty", "LP")
    g_tot.Draw("SAME LP")
    
    leg.AddEntry(g_stat, "Statistical, scaled, #sqrt{N}", "LP")
    g_stat.Draw("SAME LP")
    
    leg.AddEntry(g_syst, "Systematic", "LP")
    g_syst.Draw("SAME LP")

    leg.Draw("SAME")
    
    plotter.aux(canvas)  
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()  
    canvas.SaveAs("%s/recoilImpact_statSyst.png" % (outDir))
    canvas.SaveAs("%s/recoilImpact_statSyst.pdf" % (outDir))
    canvas.Close()



def recoilUncPlot():

    lumis = ["1p0", "2p5", "5p0", "10p0"]
    lumis_fb = [0.2, 0.5, 1, 2]
    met = "RawPFMET"
    scale=100
    group = True
    header = "W^{#pm} #rightarrow #mu^{#pm}#nu, RawPFMET"
    
   
    g_unscaled = ROOT.TGraph()
    g_unscaled.SetLineColor(ROOT.kBlue)
    g_unscaled.SetMarkerColor(ROOT.kBlue)
    g_unscaled.SetLineWidth(2)
    g_unscaled.SetMarkerSize(1.5)
    g_unscaled.SetMarkerStyle(8)

    g_scaled = ROOT.TGraph()
    g_scaled.SetLineColor(ROOT.kRed)
    g_scaled.SetMarkerColor(ROOT.kRed)
    g_scaled.SetLineWidth(2)
    g_scaled.SetMarkerSize(1.5)
    g_scaled.SetMarkerStyle(8)

    for i,lumi in enumerate(lumis):
    
        name = "nuisance_group_impact_nois" if group else "nuisance_impact_nois"
    
        ## unscaled recoil
        fIn = ROOT.TFile("/home/j/jaeyserm/combine/CMSSW_10_6_20/src/Wmass_mT/fits_cteRecoil/lowPU_mu_%s_lumi%s.root" % (met, lumi))
        impact_hist = fIn.Get(name)
        yLabels = np.array([impact_hist.GetYaxis().GetBinLabel(l) for l in range(1, impact_hist.GetNbinsY()+1)]) # nuisances
        unc_stat = impact_hist.GetBinContent(1, int(np.where(yLabels == "recoil_stat")[0][0])+1)
        unc_syst = impact_hist.GetBinContent(1, int(np.where(yLabels == "recoil_syst")[0][0])+1)
        unc = (unc_stat**2 + unc_syst**2)**0.5
        g_unscaled.SetPoint(i, lumis_fb[i], scale*unc)
        
        
        ## scaled recoil
        fIn = ROOT.TFile("/home/j/jaeyserm/combine/CMSSW_10_6_20/src/Wmass_mT/fits_scaledRecoil/lowPU_mu_%s_lumi%s.root" % (met, lumi))
        impact_hist = fIn.Get(name) # nuisance_group_impact_nois
        yLabels = np.array([impact_hist.GetYaxis().GetBinLabel(l) for l in range(1, impact_hist.GetNbinsY()+1)]) # nuisances
        unc_stat = impact_hist.GetBinContent(1, int(np.where(yLabels == "recoil_stat")[0][0])+1)
        unc_syst = impact_hist.GetBinContent(1, int(np.where(yLabels == "recoil_syst")[0][0])+1)
        unc = (unc_stat**2 + unc_syst**2)**0.5
        g_scaled.SetPoint(i, lumis_fb[i], scale*unc)
      
    outDir = "/eos/user/j/jaeyserm/www/wmass/studies_mT/lowPU/"
    

    cfg = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : 2,
        'ymin'              : 0,
        'ymax'              : 20,
            
        'xtitle'            : "Integrated luminosity (fb^{#minus1})",
        'ytitle'            : "Recoil uncertainty on m_{W} (MeV)",
            
        'topRight'          : "LowPU (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    }
    
    
    leg = ROOT.TLegend(.40, 0.75, .90, .90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)
    leg.SetHeader(header)

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy()
    canvas.SetGrid()
    dummy.Draw("HIST")
  
    leg.AddEntry(g_unscaled, "Unscaled recoil", "LP")
    g_unscaled.Draw("SAME LP")
    
    leg.AddEntry(g_scaled, "Scaled recoil, #sqrt{N}", "LP")
    g_scaled.Draw("SAME LP")

    leg.Draw("SAME")
    
    plotter.aux(canvas)  
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()  
    canvas.SaveAs("%s/recoilImpact.png" % (outDir))
    canvas.SaveAs("%s/recoilImpact.pdf" % (outDir))
    canvas.Close()





def recoilOnlyUncPlot():

    lumis = ["1p0", "2p5", "5p0", "10p0"]
    lumis_fb = [0.2, 0.5, 1, 2]
    met = "RawPFMET"
    scale=100
    group = True
    mode = "binned" # parametric binned
    header = "W^{#pm} #rightarrow #mu^{#pm}#nu, RawPFMET, #eta inclusive"
    header = "W^{#pm} #rightarrow #mu^{#pm}#nu, RawPFMET, %s" % mode
    
    inTag = "eta_inclusive_varyRecoil" # eta_inclusive eta_inclusive_varyRecoil
    outTag = "varyRecoil" # cteRecoil varyRecoil
    
    systs = ["total", "datastat", "CMS_recoil_stat", "pdfNNPDF31", "QCDscale", "muonScale", "others"]
    systs_labels = ["Total", "Stat. data", "Stat. recoil", "PDF", "QCDscale", "Muon momentum scale (10^{-4})", "Others"]
    systs = ["total", "datastat", "CMS_recoil_stat", "pdfNNPDF31", "QCDscale"]
    systs_labels = ["Total", "Stat. data", "Stat. recoil", "PDF", "QCDscale"]
    systs_colors = [ROOT.kBlack, ROOT.kBlue, ROOT.kRed, ROOT.kGreen+1, ROOT.kOrange+1, ROOT.kYellow, ROOT.kMagenta]
    

    g_unscaled = ROOT.TGraph()
    g_unscaled.SetLineColor(ROOT.kBlue)
    g_unscaled.SetMarkerColor(ROOT.kBlue)
    g_unscaled.SetLineWidth(2)
    g_unscaled.SetMarkerSize(1.5)
    g_unscaled.SetMarkerStyle(8)

    g_scaled = ROOT.TGraph()
    g_scaled.SetLineColor(ROOT.kRed)
    g_scaled.SetMarkerColor(ROOT.kRed)
    g_scaled.SetLineWidth(2)
    g_scaled.SetMarkerSize(1.5)
    g_scaled.SetMarkerStyle(8)

    for i,lumi in enumerate(lumis):
    
        name = "nuisance_group_impact_nois" if group else "nuisance_impact_nois"
    
        ## unscaled recoil
        inTag = "eta_inclusive_recoilOnly"
        fIn = ROOT.TFile("/home/j/jaeyserm/combine/CMSSW_10_6_20/src/lowPU_Wmass/%s/%s/LowPU_Wmass_mu_%s_lumi%s.root" % (mode, inTag, met, lumi))
        tree = fIn.Get("fitresults")
        tree.GetEntry(0)
        unc_tot = getattr(tree, "massShift100MeV_noi_err")
        impact_hist = fIn.Get(name) # nuisance_group_impact_nois
        yLabels = np.array([impact_hist.GetYaxis().GetBinLabel(l) for l in range(1, impact_hist.GetNbinsY()+1)]) # nuisances
        unc = impact_hist.GetBinContent(1, int(np.where(yLabels == "CMS_recoil_stat")[0][0])+1)
        g_unscaled.SetPoint(i, lumis_fb[i], scale*unc)
        
        fIn = ROOT.TFile("/home/j/jaeyserm/combine/CMSSW_10_6_20/src/lowPU_Wmass/%s/%s/LowPU_Wmass_mu_%s_lumi%s_statOnly.root" % (mode, inTag, met, lumi))
        tree = fIn.Get("fitresults")
        tree.GetEntry(0)
        unc_stat = getattr(tree, "massShift100MeV_noi_err")
        
        print(lumi, scale*unc, unc_stat*scale, unc_tot*scale)
        
        
        ## scaled recoil
        inTag = "eta_inclusive_recoilOnly_scaledRecoil"
        fIn = ROOT.TFile("/home/j/jaeyserm/combine/CMSSW_10_6_20/src/lowPU_Wmass/%s/%s/LowPU_Wmass_mu_%s_lumi%s.root" % (mode, inTag, met, lumi))
        impact_hist = fIn.Get(name) # nuisance_group_impact_nois
        yLabels = np.array([impact_hist.GetYaxis().GetBinLabel(l) for l in range(1, impact_hist.GetNbinsY()+1)]) # nuisances
        unc = impact_hist.GetBinContent(1, int(np.where(yLabels == "CMS_recoil_stat")[0][0])+1)
        g_scaled.SetPoint(i, lumis_fb[i], scale*unc)
      
    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/Projection/"
    

    cfg = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : 2,
        'ymin'              : 0,
        'ymax'              : 10,
            
        'xtitle'            : "Integrated luminosity (fb^{#minus1})",
        'ytitle'            : "Recoil uncertainty on m_{W} (MeV)",
            
        'topRight'          : "LowPU (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    }
    
    
    leg = ROOT.TLegend(.40, 0.75, .90, .90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)
    leg.SetHeader(header)

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy()
    canvas.SetGrid()
    dummy.Draw("HIST")
  
    leg.AddEntry(g_unscaled, "Unscaled recoil", "LP")
    g_unscaled.Draw("SAME LP")
    
    leg.AddEntry(g_scaled, "Scaled recoil, #sqrt{N}", "LP")
    g_scaled.Draw("SAME LP")

    leg.Draw("SAME")
    
    plotter.aux(canvas)  
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()  
    canvas.SaveAs("%s/recoilOnlyUnc_%s.png" % (outDir, mode))
    canvas.SaveAs("%s/recoilOnlyUnc_%s.pdf" % (outDir, mode))
    canvas.Close()

	
if __name__ == "__main__":


    #mTcomparison()
    #eventYields()
    
    #expUncPlotStat()
    expUncPlotSyst()
    #recoilUncPlot()
    #recoilUncPlot_statSyst()
    #recoilOnlyUncPlot() # only recoil uncertainties
    
    