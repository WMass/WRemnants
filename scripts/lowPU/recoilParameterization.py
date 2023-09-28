
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

from wremnants.datasets.datagroups import Datagroups
def prepareDir(outDir, remove=True):

    if os.path.exists(outDir) and os.path.isdir(outDir) and remove: shutil.rmtree(outDir)
    os.system("mkdir -p %s" % outDir)
    os.system("cp /eos/user/j/jaeyserm/www/wmass/index.php %s" % outDir)


def backgroundComposition():

    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/"
    
    plotCfg = {}
    

    
    
    plotCfg['singleboson'] = {
        "color":        ROOT.TColor.GetColor(248, 206, 104),
        "label":        "EWK (DY, W #rightarrow #tau, diboson)",
        "procs":        ["WplusJetsToMuNu", "WminusJetsToMuNu", "WplusJetsToTauNu", "WminusJetsToTauNu"],
    }
    
    plotCfg['WWTo2L2Nu'] = {
        "color":        ROOT.TColor.GetColor(248, 206, 104),
        "label":        "WW",
        "procs":        ["WWTo2L2Nu"],
    }
    
    plotCfg['WZTo3LNu'] = {
        "color":        ROOT.kGray,
        "label":        "WZ",
        "procs":        ["WZTo3LNu"],
    }
    
    plotCfg['ZZ'] = {
        "color":        ROOT.TColor.GetColor(100, 192, 232),
        "label":        "ZZ",
        "procs":        ["ZZ"],
    }
    
    plotCfg['TTbar'] = {
        "color":        ROOT.TColor.GetColor(222, 90, 106),
        "label":        "TTbar",
        "procs":        ["TTTo2L2Nu", "TTToSemiLeptonic"],
    }

    bkgs = ["WZTo3LNu", "WWTo2L2Nu", "ZZ", "TTbar"]


    # the following histograms are binned in 0.5 GeV for qT, from 0 to 500 GeV
    fIn_muon = ROOT.TFile("/eos/user/j/jaeyserm/analysis/lowPU/Zmumu/output.root")
    fIn_electron = ROOT.TFile("/eos/user/j/jaeyserm/analysis/lowPU/Zee/output.root")
    
    fIn_electron.ls()
    
    st = ROOT.THStack()
    st.SetName("stack")
    h_bkg_tot = None
    
    leg = ROOT.TLegend(.60, 0.6, .85, .75)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)
    
    
    
    for bkg in bkgs:
    
        h_proc = None
        for proc in plotCfg[bkg]['procs']:
        
            thn_muon = fIn_muon.Get("lep_pt_ll_%s" % proc)
            h_muon = thn_muon.Projection(0, "E")
    
            print(proc)
            thn_electron = fIn_electron.Get("lep_pt_ll_%s" % proc)
            h_electron = thn_electron.Projection(0, "E")
            

            h_muon.Add(h_electron)
            h_muon.Scale(lumi)
            h_muon.Rebin(2)
            
            if h_bkg_tot == None: 
            
                h_bkg_tot = copy.deepcopy(h_muon)
                h_bkg_tot.SetName("h_bkg_tot")
                
            else: h_bkg_tot.Add(h_muon)
            
            if h_proc == None: 
            
                h_proc = copy.deepcopy(h_muon)
                h_proc.SetName(bkg)
                
            else: h_proc.Add(h_muon)
            
            h_muon.Delete()
            h_electron.Delete()
            
            
        h_proc.SetName(bkg)
        h_proc.SetFillColor(plotCfg[bkg]['color'])
        h_proc.SetLineColor(ROOT.kBlack)
        h_proc.SetLineWidth(1)
        h_proc.SetLineStyle(1)

        leg.AddEntry(h_proc, plotCfg[bkg]['label'], "F")
        st.Add(h_proc)
   

    h_bkg_tot.SetLineColor(ROOT.kBlack)
    h_bkg_tot.SetFillColor(0)
    h_bkg_tot.SetLineWidth(2)

    
    cfg = {

        'logy'              : False,
        'logx'              : False,
            
        'xmin'              : 0,
        'xmax'              : 200,
        'ymin'              : 0,
        'ymax'              : 20,
                
        'xtitle'            : "q_{T} (GeV)",
        'ytitle'            : "Events",
                
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
    }

        
    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy()   
    canvas.cd()
    dummy.Draw("HIST")
        
   
    st.Draw("HIST SAME")
    h_bkg_tot.Draw("HIST SAME")
    leg.Draw("SAME")
    
    plotter.aux(canvas)
           

    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.RedrawAxis()
    canvas.SaveAs("%s/bkg_composition.png" % (outDir))
    canvas.SaveAs("%s/bkg_composition.pdf" % (outDir))
    canvas.Delete()


def electronMuon():

    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/electronMuon/"
    prepareDir(outDir)
    
    isData = False
    tag = "reco"
    if isData: tag = "data"
 
    bkgs = ["TTTo2L2Nu", "TTToSemiLeptonic", "WplusJetsToMuNu", "WminusJetsToMuNu", "WZTo3LNu", "WWTo2L2Nu", "WplusJetsToTauNu", "WminusJetsToTauNu"]
    #bkgs = ["WminusJetsToTauNu"]

    # the following histograms are binned in 0.5 GeV for qT, from 0 to 500 GeV
    fIn_muon = ROOT.TFile("/eos/user/j/jaeyserm/analysis/lowPU/Zmumu/output.root")
    thn_reco_para_muon = fIn_muon.Get("recoil_uncorr_para_qt_DYmumu_MiNNLO")
    thn_reco_perp_muon = fIn_muon.Get("recoil_uncorr_perp_qt_DYmumu_MiNNLO")
    thn_reco_magn_muon = fIn_muon.Get("recoil_uncorr_magn_qt_DYmumu_MiNNLO")
    thn_data_para_muon = fIn_muon.Get("recoil_uncorr_para_qt_singlemuon")
    thn_data_perp_muon = fIn_muon.Get("recoil_uncorr_perp_qt_singlemuon")
    thn_data_magn_muon = fIn_muon.Get("recoil_uncorr_magn_qt_singlemuon")
    
    thn_data_lep_pt_muon = fIn_muon.Get("lep_pt_ll_singlemuon")
    thn_reco_lep_pt_muon = fIn_muon.Get("lep_pt_ll_DYmumu_MiNNLO")

    thn_bkgs = {}
    for bkg in bkgs:
    
        thn_bkgs['para_%s_muon' % bkg] = fIn_muon.Get("recoil_uncorr_para_qt_%s" % bkg)
        thn_bkgs['perp_%s_muon' % bkg] = fIn_muon.Get("recoil_uncorr_perp_qt_%s" % bkg)
        thn_bkgs['lep_pt_ll_%s_muon' % bkg] = fIn_muon.Get("lep_pt_ll_%s" % bkg)
    
 
    fIn_electron = ROOT.TFile("/eos/user/j/jaeyserm/analysis/lowPU/Zee/output.root")
    thn_reco_para_electron = fIn_electron.Get("recoil_uncorr_para_qt_DYee_MiNNLO")
    thn_reco_perp_electron = fIn_electron.Get("recoil_uncorr_perp_qt_DYee_MiNNLO")
    thn_reco_magn_electron = fIn_electron.Get("recoil_uncorr_magn_qt_DYee_MiNNLO")
    thn_data_para_electron = fIn_electron.Get("recoil_uncorr_para_qt_singleelectron")
    thn_data_perp_electron = fIn_electron.Get("recoil_uncorr_perp_qt_singleelectron")
    thn_data_magn_electron = fIn_electron.Get("recoil_uncorr_magn_qt_singleelectron")
    
    thn_data_lep_pt_electron = fIn_electron.Get("lep_pt_ll_singleelectron")
    thn_reco_lep_pt_electron = fIn_electron.Get("lep_pt_ll_DYee_MiNNLO")

    for bkg in bkgs:
    
        thn_bkgs['para_%s_electron' % bkg] = fIn_electron.Get("recoil_uncorr_para_qt_%s" % bkg)
        thn_bkgs['perp_%s_electron' % bkg] = fIn_electron.Get("recoil_uncorr_perp_qt_%s" % bkg)
        thn_bkgs['lep_pt_ll_%s_electron' % bkg] = fIn_muon.Get("lep_pt_ll_%s" % bkg)

    fIn_muon.Close()
    fIn_electron.Close()


    magn_muon = ROOT.TGraphErrors()
    magn_muon.SetLineColor(ROOT.kBlue)
    magn_muon.SetMarkerStyle(20)
    magn_muon.SetMarkerSize(0.9)
    magn_muon.SetMarkerColor(ROOT.kBlue)
    
    magn_electron = ROOT.TGraphErrors()
    magn_electron.SetLineColor(ROOT.kRed)
    magn_electron.SetMarkerStyle(20)
    magn_electron.SetMarkerSize(0.9)
    magn_electron.SetMarkerColor(ROOT.kRed)
    
    magn_ratio = ROOT.TGraphErrors()
    magn_ratio.SetLineColor(ROOT.kBlack)
    magn_ratio.SetMarkerStyle(20)
    magn_ratio.SetMarkerSize(0.9)
    magn_ratio.SetMarkerColor(ROOT.kBlack)

    para_mean_muon = ROOT.TGraphErrors()
    para_mean_muon.SetLineColor(ROOT.kBlue)
    para_mean_muon.SetMarkerStyle(20)
    para_mean_muon.SetMarkerSize(0.9)
    para_mean_muon.SetMarkerColor(ROOT.kBlue)
    
    para_mean_electron = ROOT.TGraphErrors()
    para_mean_electron.SetLineColor(ROOT.kRed)
    para_mean_electron.SetMarkerStyle(20)
    para_mean_electron.SetMarkerSize(0.9)
    para_mean_electron.SetMarkerColor(ROOT.kRed)
    
    para_mean_ratio = ROOT.TGraphErrors()
    para_mean_ratio.SetLineColor(ROOT.kBlack)
    para_mean_ratio.SetMarkerStyle(20)
    para_mean_ratio.SetMarkerSize(0.9)
    para_mean_ratio.SetMarkerColor(ROOT.kBlack)


    para_sigma_muon = ROOT.TGraphErrors()
    para_sigma_muon.SetLineColor(ROOT.kBlue)
    para_sigma_muon.SetMarkerStyle(20)
    para_sigma_muon.SetMarkerSize(0.9)
    para_sigma_muon.SetMarkerColor(ROOT.kBlue)
    
    para_sigma_electron = ROOT.TGraphErrors()
    para_sigma_electron.SetLineColor(ROOT.kRed)
    para_sigma_electron.SetMarkerStyle(20)
    para_sigma_electron.SetMarkerSize(0.9)
    para_sigma_electron.SetMarkerColor(ROOT.kRed)
    
    para_sigma_ratio = ROOT.TGraphErrors()
    para_sigma_ratio.SetLineColor(ROOT.kBlack)
    para_sigma_ratio.SetMarkerStyle(20)
    para_sigma_ratio.SetMarkerSize(0.9)
    para_sigma_ratio.SetMarkerColor(ROOT.kBlack)
    
    
    perp_sigma_muon = ROOT.TGraphErrors()
    perp_sigma_muon.SetLineColor(ROOT.kBlue)
    perp_sigma_muon.SetMarkerStyle(20)
    perp_sigma_muon.SetMarkerSize(0.9)
    perp_sigma_muon.SetMarkerColor(ROOT.kBlue)
    
    perp_sigma_electron = ROOT.TGraphErrors()
    perp_sigma_electron.SetLineColor(ROOT.kRed)
    perp_sigma_electron.SetMarkerStyle(20)
    perp_sigma_electron.SetMarkerSize(0.9)
    perp_sigma_electron.SetMarkerColor(ROOT.kRed)

    perp_sigma_ratio = ROOT.TGraphErrors()
    perp_sigma_ratio.SetLineColor(ROOT.kBlack)
    perp_sigma_ratio.SetMarkerStyle(20)
    perp_sigma_ratio.SetMarkerSize(0.9)
    perp_sigma_ratio.SetMarkerColor(ROOT.kBlack)
    
    
    perp_mean_muon = ROOT.TGraphErrors()
    perp_mean_muon.SetLineColor(ROOT.kBlue)
    perp_mean_muon.SetMarkerStyle(20)
    perp_mean_muon.SetMarkerSize(0.9)
    perp_mean_muon.SetMarkerColor(ROOT.kBlue)
    
    perp_mean_electron = ROOT.TGraphErrors()
    perp_mean_electron.SetLineColor(ROOT.kRed)
    perp_mean_electron.SetMarkerStyle(20)
    perp_mean_electron.SetMarkerSize(0.9)
    perp_mean_electron.SetMarkerColor(ROOT.kRed)
    
    perp_mean_ratio = ROOT.TGraphErrors()
    perp_mean_ratio.SetLineColor(ROOT.kBlack)
    perp_mean_ratio.SetMarkerStyle(20)
    perp_mean_ratio.SetMarkerSize(0.9)
    perp_mean_ratio.SetMarkerColor(ROOT.kBlack)

    for qTbin in range(0, len(qTbins)-1):
    
        qTbinMinGeV, qTbinMaxGeV = qTbins[qTbin], qTbins[qTbin+1]
        qTbinMin, qTbinMax = int(qTbins[qTbin]+1), int(qTbins[qTbin+1]) # proper indexing!
        qT = 0.5*(qTbinMinGeV + qTbinMaxGeV)
        qT_err = 0.5*(qTbinMaxGeV - qTbinMinGeV)

        print("***************** Do qTbin=[%f,%f],[%d, %d]" % (qTbinMinGeV, qTbinMaxGeV, qTbinMin, qTbinMax))
        

        #################### DATA
        if isData:
        
            # get  the correct mean qT
            thn_data_lep_pt_muon.GetAxis(0).SetRange(qTbinMin, qTbinMax)
            h_muon = thn_data_lep_pt_muon.Projection(0, "E")
            
            thn_data_lep_pt_electron.GetAxis(0).SetRange(qTbinMin, qTbinMax)
            h_electron = thn_data_lep_pt_electron.Projection(0, "E")

            for bkg in bkgs:
                
                thn_bkgs['lep_pt_ll_%s_muon' % bkg].GetAxis(0).SetRange(qTbinMin, qTbinMax)
                hist_muon_bkg = thn_bkgs['lep_pt_ll_%s_muon' % bkg].Projection(0, "E")
                hist_muon_bkg.Scale(lumi)
                h_muon.Add(hist_muon_bkg, -1)
                hist_muon_bkg.Delete()
                
                
                thn_bkgs['lep_pt_ll_%s_electron' % bkg].GetAxis(0).SetRange(qTbinMin, qTbinMax)
                hist_electron_bkg = thn_bkgs['lep_pt_ll_%s_electron' % bkg].Projection(0, "E")
                hist_electron_bkg.Scale(lumi)
                h_electron.Add(hist_electron_bkg, -1)
                hist_electron_bkg.Delete()
                
            qT_muon = h_muon.GetMean()
            qT_muon_err = h_muon.GetRMS()
            
            qT_electron = h_electron.GetMean()
            qT_electron_err = h_electron.GetRMS()
            
            h_muon.Delete()
            h_electron.Delete()
            
            # data - magnitude
            thn_data_magn_muon.GetAxis(1).SetRange(qTbinMin, qTbinMax)
            hist_muon = thn_data_magn_muon.Projection(0, "E")
            
            thn_data_magn_electron.GetAxis(1).SetRange(qTbinMin, qTbinMax)
            hist_electron = thn_data_magn_electron.Projection(0, "E")


            hist_muon.Rebin(2)
            hist_muon.Scale(1./hist_muon.Integral())
            
            hist_electron.Rebin(2)
            hist_electron.Scale(1./hist_electron.Integral())
            
            magn_muon.SetPoint(qTbin, qT, hist_muon.GetMean())
            magn_muon.SetPointError(qTbin, qT_err, 0)
            magn_electron.SetPoint(qTbin, qT, hist_electron.GetMean())
            magn_electron.SetPointError(qTbin, qT_err, 0)
            magn_ratio.SetPoint(qTbin, qT, hist_muon.GetMean()/hist_electron.GetMean())
            magn_ratio.SetPointError(qTbin, qT_err, 0)
            
            hist_muon.Delete()
            hist_electron.Delete()
            
            
            # data - parallel
            thn_data_para_muon.GetAxis(1).SetRange(qTbinMin, qTbinMax)
            hist_muon = thn_data_para_muon.Projection(0, "E")
            
            thn_data_para_electron.GetAxis(1).SetRange(qTbinMin, qTbinMax)
            hist_electron = thn_data_para_electron.Projection(0, "E")

            for bkg in bkgs:
                
                thn_bkgs['para_%s_muon' % bkg].GetAxis(1).SetRange(qTbinMin, qTbinMax)
                hist_muon_bkg = thn_bkgs['para_%s_muon' % bkg].Projection(0, "E")
                hist_muon_bkg.Scale(lumi)
                hist_muon.Add(hist_muon_bkg, -1)
                hist_muon_bkg.Delete()
                
                thn_bkgs['para_%s_electron' % bkg].GetAxis(1).SetRange(qTbinMin, qTbinMax)
                hist_electron_bkg = thn_bkgs['para_%s_electron' % bkg].Projection(0, "E")
                hist_electron_bkg.Scale(lumi)
                hist_electron.Add(hist_electron_bkg, -1)
                hist_electron_bkg.Delete()

            '''
            for iBin in range(0, hist_muon.GetNbinsX()+1):
                if hist_muon.GetBinContent(iBin) > 0:
                    newVal = hist_muon.GetBinContent(iBin) - hist_bkg.GetBinContent(iBin)
                    if newVal < 0: newVal = 0
                    hist_muon.SetBinContent(iBin, newVal)
            '''
            hist_muon.Rebin(2)
            hist_muon.Scale(1./hist_muon.Integral())
            
            hist_electron.Rebin(2)
            hist_electron.Scale(1./hist_electron.Integral())
            
            para_sigma_muon.SetPoint(qTbin, qT, hist_muon.GetRMS())
            para_sigma_muon.SetPointError(qTbin, qT_err, hist_muon.GetRMSError())
            para_mean_muon.SetPoint(qTbin, qT, hist_muon.GetMean())
            para_mean_muon.SetPointError(qTbin, qT_err, hist_muon.GetMeanError())
            
            para_sigma_electron.SetPoint(qTbin, qT, hist_electron.GetRMS())
            para_sigma_electron.SetPointError(qTbin, qT_err, hist_electron.GetRMSError())
            para_mean_electron.SetPoint(qTbin, qT, hist_electron.GetMean())
            para_mean_electron.SetPointError(qTbin, qT_err, hist_electron.GetMeanError())
            
            para_sigma_ratio.SetPoint(qTbin, qT, hist_muon.GetRMS()/hist_electron.GetRMS())
            para_sigma_ratio.SetPointError(qTbin, qT_err, 0)
            para_mean_ratio.SetPoint(qTbin, qT, hist_muon.GetMean()/hist_electron.GetMean())
            para_mean_ratio.SetPointError(qTbin, qT_err, 0)
            
            hist_muon.Delete()
            hist_electron.Delete()
            
 
            # data - perpendicular
            thn_data_perp_muon.GetAxis(1).SetRange(qTbinMin, qTbinMax)
            hist_muon = thn_data_perp_muon.Projection(0, "E")
            
            thn_data_perp_electron.GetAxis(1).SetRange(qTbinMin, qTbinMax)
            hist_electron = thn_data_perp_electron.Projection(0, "E")

            for bkg in bkgs:
                
                thn_bkgs['perp_%s_muon' % bkg].GetAxis(1).SetRange(qTbinMin, qTbinMax)
                hist_muon_bkg = thn_bkgs['perp_%s_muon' % bkg].Projection(0, "E")
                hist_muon_bkg.Scale(lumi)
                hist_muon.Add(hist_muon_bkg, -1)
                hist_muon_bkg.Delete()
                
                thn_bkgs['perp_%s_electron' % bkg].GetAxis(1).SetRange(qTbinMin, qTbinMax)
                hist_electron_bkg = thn_bkgs['perp_%s_electron' % bkg].Projection(0, "E")
                hist_electron_bkg.Scale(lumi)
                hist_electron.Add(hist_electron_bkg, -1)
                hist_electron_bkg.Delete()

            '''
            for iBin in range(0, hist_muon.GetNbinsX()+1):
                if hist_muon.GetBinContent(iBin) > 0:
                    newVal = hist_muon.GetBinContent(iBin) - hist_bkg.GetBinContent(iBin)
                    if newVal < 0: newVal = 0
                    hist_muon.SetBinContent(iBin, newVal)
            '''
            
            hist_muon.Rebin(2)
            hist_muon.Scale(1./hist_muon.Integral())
            
            hist_electron.Rebin(2)
            hist_electron.Scale(1./hist_electron.Integral())
            
            perp_sigma_muon.SetPoint(qTbin, qT, hist_muon.GetRMS())
            perp_sigma_muon.SetPointError(qTbin, qT_err, hist_muon.GetRMSError())
            perp_mean_muon.SetPoint(qTbin, qT, hist_muon.GetMean())
            perp_mean_muon.SetPointError(qTbin, qT_err, hist_muon.GetMeanError())
            
            perp_sigma_electron.SetPoint(qTbin, qT, hist_electron.GetRMS())
            perp_sigma_electron.SetPointError(qTbin, qT_err, hist_electron.GetRMSError())
            perp_mean_electron.SetPoint(qTbin, qT, hist_electron.GetMean())
            perp_mean_electron.SetPointError(qTbin, qT_err, hist_electron.GetMeanError())
            
            perp_sigma_ratio.SetPoint(qTbin, qT, hist_muon.GetRMS()/hist_electron.GetRMS())
            perp_sigma_ratio.SetPointError(qTbin, qT_err, 0)
            perp_mean_ratio.SetPoint(qTbin, qT, hist_muon.GetMean()/hist_electron.GetMean())
            perp_mean_ratio.SetPointError(qTbin, qT_err, 0)
            
            hist_muon.Delete()
            hist_electron.Delete()        

          

        else: 
        
            # reco - magnitude
            thn_reco_magn_muon.GetAxis(1).SetRange(qTbinMin, qTbinMax)
            hist_muon = thn_reco_magn_muon.Projection(0, "E")
            
            thn_reco_magn_electron.GetAxis(1).SetRange(qTbinMin, qTbinMax)
            hist_electron = thn_reco_magn_electron.Projection(0, "E")


            hist_muon.Rebin(2)
            hist_muon.Scale(1./hist_muon.Integral())
            
            hist_electron.Rebin(2)
            hist_electron.Scale(1./hist_electron.Integral())
            
            magn_muon.SetPoint(qTbin, qT, hist_muon.GetMean())
            magn_muon.SetPointError(qTbin, qT_err, 0)
            magn_electron.SetPoint(qTbin, qT, hist_electron.GetMean())
            magn_electron.SetPointError(qTbin, qT_err, 0)
            magn_ratio.SetPoint(qTbin, qT, hist_muon.GetMean()/hist_electron.GetMean())
            magn_ratio.SetPointError(qTbin, qT_err, 0)
            
            hist_muon.Delete()
            hist_electron.Delete()


            
            # reco - parallel
            thn_reco_para_muon.GetAxis(1).SetRange(qTbinMin, qTbinMax)
            hist_muon = thn_reco_para_muon.Projection(0, "E")
            
            thn_reco_para_electron.GetAxis(1).SetRange(qTbinMin, qTbinMax)
            hist_electron = thn_reco_para_electron.Projection(0, "E")

            hist_muon.Rebin(2)
            hist_muon.Scale(1./hist_muon.Integral())
            
            hist_electron.Rebin(2)
            hist_electron.Scale(1./hist_electron.Integral())
            
            para_sigma_muon.SetPoint(qTbin, qT, hist_muon.GetRMS())
            para_sigma_muon.SetPointError(qTbin, qT_err, hist_muon.GetRMSError())
            para_mean_muon.SetPoint(qTbin, qT, hist_muon.GetMean())
            para_mean_muon.SetPointError(qTbin, qT_err, hist_muon.GetMeanError())
            
            para_sigma_electron.SetPoint(qTbin, qT, hist_electron.GetRMS())
            para_sigma_electron.SetPointError(qTbin, qT_err, hist_electron.GetRMSError())
            para_mean_electron.SetPoint(qTbin, qT, hist_electron.GetMean())
            para_mean_electron.SetPointError(qTbin, qT_err, hist_electron.GetMeanError())
            
            para_sigma_ratio.SetPoint(qTbin, qT, hist_muon.GetRMS()/hist_electron.GetRMS())
            para_sigma_ratio.SetPointError(qTbin, qT_err, 0)
            para_mean_ratio.SetPoint(qTbin, qT, hist_muon.GetMean()/hist_electron.GetMean())
            para_mean_ratio.SetPointError(qTbin, qT_err, 0)
            
            hist_muon.Delete()
            hist_electron.Delete()
            
 
            # reco - perpendicular
            thn_reco_perp_muon.GetAxis(1).SetRange(qTbinMin, qTbinMax)
            hist_muon = thn_reco_perp_muon.Projection(0, "E")
            
            thn_reco_perp_electron.GetAxis(1).SetRange(qTbinMin, qTbinMax)
            hist_electron = thn_reco_perp_electron.Projection(0, "E")
            
            hist_muon.Rebin(2)
            hist_muon.Scale(1./hist_muon.Integral())
            
            hist_electron.Rebin(2)
            hist_electron.Scale(1./hist_electron.Integral())
            
            perp_sigma_muon.SetPoint(qTbin, qT, hist_muon.GetRMS())
            perp_sigma_muon.SetPointError(qTbin, qT_err, hist_muon.GetRMSError())
            perp_mean_muon.SetPoint(qTbin, qT, hist_muon.GetMean())
            perp_mean_muon.SetPointError(qTbin, qT_err, hist_muon.GetMeanError())
            
            perp_sigma_electron.SetPoint(qTbin, qT, hist_electron.GetRMS())
            perp_sigma_electron.SetPointError(qTbin, qT_err, hist_electron.GetRMSError())
            perp_mean_electron.SetPoint(qTbin, qT, hist_electron.GetMean())
            perp_mean_electron.SetPointError(qTbin, qT_err, hist_electron.GetMeanError())
            
            perp_sigma_ratio.SetPoint(qTbin, qT, hist_muon.GetRMS()/hist_electron.GetRMS())
            perp_sigma_ratio.SetPointError(qTbin, qT_err, 0)
            perp_mean_ratio.SetPoint(qTbin, qT, hist_muon.GetMean()/hist_electron.GetMean())
            perp_mean_ratio.SetPointError(qTbin, qT_err, 0)
            
            hist_muon.Delete()
            hist_electron.Delete()  


    if True:
    
        cfg = {

            'logy'              : False,
            'logx'              : False,
            
            'xmin'              : 0,
            'xmax'              : 150,
            'ymin'              : 5,
            'ymax'              : 25,
                
            'xtitle'            : "q_{T} (GeV)",
            'ytitle'            : "#sigma(U_{#parallel}) (GeV)",
                
            'topRight'          : "199 pb^{#minus1} (13 TeV)", 
            'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
            
            'ratiofraction'     : 0.25,
            'ytitleR'           : "Ratio",
            'yminR'             : 0.7,
            'ymaxR'             : 1.3,            
        }

        
        plotter.cfg = cfg
        canvas, padT, padB = plotter.canvasRatio()
        dummyT, dummyB, dummyL = plotter.dummyRatio()
            
        canvas.cd()
        padT.Draw()
        padT.cd()
        dummyT.Draw("HIST")
        
        leg = ROOT.TLegend(.20, 0.65, .6, .75)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.040)
        leg.AddEntry(para_sigma_muon, "Muon channel (%s)" % tag, "LP")
        leg.AddEntry(para_sigma_electron, "Electron channel (%s)" % tag, "LP")
            
        para_sigma_muon.Draw("PE SAME")
        para_sigma_electron.Draw("PE SAME")
        
        plotter.auxRatio()
        leg.Draw("SAME")
        
        ## BOTTOM PAD ##
        canvas.cd()
        padB.Draw()
        padB.cd()
        dummyB.Draw("HIST")
        para_sigma_ratio.Draw("PE SAME")

        dummyL.Draw()    
        canvas.Modify()
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs("%s/recoil_para_sigma_%s_%s.png" % (outDir, tag, met))
        canvas.SaveAs("%s/recoil_para_sigma_%s_%s.pdf" % (outDir, tag, met))
        canvas.Delete()



    if True:
    
    
        cfg = {

            'logy'              : False,
            'logx'              : False,
            
            'xmin'              : 0,
            'xmax'              : 150,
            'ymin'              : 5,
            'ymax'              : 25,
                
            'xtitle'            : "q_{T} (GeV)",
            'ytitle'            : "#sigma(U_{#perp}) (GeV)",
                
            'topRight'          : "199 pb^{#minus1} (13 TeV)", 
            'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
            
            'ratiofraction'     : 0.25,
            'ytitleR'           : "Ratio",
            'yminR'             : 0.7,
            'ymaxR'             : 1.3,
        }

        
        plotter.cfg = cfg
        canvas, padT, padB = plotter.canvasRatio()
        dummyT, dummyB, dummyL = plotter.dummyRatio()
            
        canvas.cd()
        padT.Draw()
        padT.cd()
        dummyT.Draw("HIST")
        
        leg = ROOT.TLegend(.20, 0.65, .6, .75)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.040)    
        leg.AddEntry(perp_sigma_muon, "Muon channel (%s)" % tag, "LP")
        leg.AddEntry(perp_sigma_electron, "Electron channel (%s)" % tag, "LP")
        
        perp_sigma_muon.Draw("PE SAME")
        perp_sigma_electron.Draw("PE SAME")

        plotter.auxRatio()
        leg.Draw("SAME")
        
        ## BOTTOM PAD ##
        canvas.cd()
        padB.Draw()
        padB.cd()
        dummyB.Draw("HIST")
        perp_sigma_ratio.Draw("PE SAME")

        dummyL.Draw()    
        canvas.Modify()
        canvas.Update()
        canvas.Draw()
        canvas.RedrawAxis()
        canvas.SaveAs("%s/recoil_perp_sigma_%s_%s.png" % (outDir, tag, met))
        canvas.SaveAs("%s/recoil_perp_sigma_%s_%s.pdf" % (outDir, tag, met))
        canvas.Delete()



    if True:
    
        cfg = {

            'logy'              : False,
            'logx'              : False,
            
            'xmin'              : 0,
            'xmax'              : 150,
            'ymin'              : -150,
            'ymax'              : 0,
                
            'xtitle'            : "q_{T} (GeV)",
            'ytitle'            : "#LT U_{#parallel} #GT (GeV)",
                
            'topRight'          : "199 pb^{#minus1} (13 TeV)", 
            'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
            
            'ratiofraction'     : 0.25,
            'ytitleR'           : "Ratio",
            'yminR'             : 0.7,
            'ymaxR'             : 1.3,
        }

        
        plotter.cfg = cfg
        canvas, padT, padB = plotter.canvasRatio()
        dummyT, dummyB, dummyL = plotter.dummyRatio()
            
        canvas.cd()
        padT.Draw()
        padT.cd()
        dummyT.Draw("HIST")
        
        leg = ROOT.TLegend(.55, 0.65, .9, .75)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.040)
        
        leg.AddEntry(para_mean_muon, "Muon channel (%s)" % tag, "LP")
        leg.AddEntry(para_mean_electron, "Electron channel (%s)" % tag, "LP")
            
        para_mean_muon.Draw("PE SAME")
        para_mean_electron.Draw("PE SAME")

        plotter.auxRatio()
        leg.Draw("SAME")
        
        ## BOTTOM PAD ##
        canvas.cd()
        padB.Draw()
        padB.cd()
        dummyB.Draw("HIST")
        para_mean_ratio.Draw("PE SAME")

        dummyL.Draw()    
        canvas.Modify()
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs("%s/recoil_para_mean_%s_%s.png" % (outDir, tag, met))
        canvas.SaveAs("%s/recoil_para_mean_%s_%s.pdf" % (outDir, tag, met))
        canvas.Delete()



    if True:
    
        cfg = {

            'logy'              : False,
            'logx'              : False,
            
            'xmin'              : 0,
            'xmax'              : 150,
            'ymin'              : -5,
            'ymax'              : 5,
                
            'xtitle'            : "q_{T} (GeV)",
            'ytitle'            : "#LT U_{#perp} #GT (GeV)",
                
            'topRight'          : "199 pb^{#minus1} (13 TeV)", 
            'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
            
            'ratiofraction'     : 0.25,
            'ytitleR'           : "Ratio",
            'yminR'             : 0.7,
            'ymaxR'             : 1.3,
        }

        
        plotter.cfg = cfg
        canvas, padT, padB = plotter.canvasRatio()
        dummyT, dummyB, dummyL = plotter.dummyRatio()
            
        canvas.cd()
        padT.Draw()
        padT.cd()
        dummyT.Draw("HIST")
        
        leg = ROOT.TLegend(.55, 0.65, .9, .75)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.040)
        
        leg.AddEntry(perp_mean_muon, "Muon channel (%s)" % tag, "LP")
        leg.AddEntry(perp_mean_electron, "Electron channel (%s)" % tag, "LP")
            
        perp_mean_muon.Draw("PE SAME")
        perp_mean_electron.Draw("PE SAME")

        plotter.auxRatio()
        leg.Draw("SAME")
        
        ## BOTTOM PAD ##
        canvas.cd()
        padB.Draw()
        padB.cd()
        dummyB.Draw("HIST")
        para_mean_ratio.Draw("PE SAME")

        dummyL.Draw()    
        canvas.Modify()
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs("%s/recoil_perp_mean_%s_%s.png" % (outDir, tag, met))
        canvas.SaveAs("%s/recoil_perp_mean_%s_%s.pdf" % (outDir, tag, met))
        canvas.Delete()


    if True:
    
        cfg = {

            'logy'              : False,
            'logx'              : False,
            
            'xmin'              : 0,
            'xmax'              : 150,
            'ymin'              : 0,
            'ymax'              : 150,
                
            'xtitle'            : "q_{T} (GeV)",
            'ytitle'            : "|U| (GeV)",
                
            'topRight'          : "199 pb^{#minus1} (13 TeV)", 
            'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
            
            'ratiofraction'     : 0.25,
            'ytitleR'           : "Ratio",
            'yminR'             : 0.7,
            'ymaxR'             : 1.3,
        }

        
        plotter.cfg = cfg
        canvas, padT, padB = plotter.canvasRatio()
        dummyT, dummyB, dummyL = plotter.dummyRatio()
            
        canvas.cd()
        padT.Draw()
        padT.cd()
        dummyT.Draw("HIST")
        
        leg = ROOT.TLegend(.25, 0.65, .6, .75)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.040)
        
        leg.AddEntry(magn_muon, "Muon channel (%s)" % tag, "LP")
        leg.AddEntry(magn_electron, "Electron channel (%s)" % tag, "LP")
            
        magn_muon.Draw("PE SAME")
        magn_electron.Draw("PE SAME")
        
        plotter.auxRatio()
        leg.Draw("SAME")
        
        ## BOTTOM PAD ##
        canvas.cd()
        padB.Draw()
        padB.cd()
        dummyB.Draw("HIST")
        magn_ratio.Draw("PE SAME")

        dummyL.Draw()    
        canvas.Modify()
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs("%s/recoil_magn_%s_%s.png" % (outDir, tag, met))
        canvas.SaveAs("%s/recoil_magn_%s_%s.pdf" % (outDir, tag, met))
        canvas.Delete()


def bareComponents():

 
    bkgs = ["TTTo2L2Nu", "TTToSemiLeptonic", "WplusJetsToMuNu", "WminusJetsToMuNu", "WZTo3LNu", "WWTo2L2Nu", "WplusJetsToTauNu", "WminusJetsToTauNu"]
    #bkgs = ["WminusJetsToTauNu"]
    #bkgs = []

    # the following histograms are binned in 0.5 GeV for qT, from 0 to 500 GeV
    fIn_muon = ROOT.TFile("/eos/user/j/jaeyserm/analysis/lowPU/Zmumu/output.root")
    thn_gen_para_muon = fIn_muon.Get("recoil_uncorr_para_qtgen_DYmumu_MiNNLO")
    thn_gen_perp_muon = fIn_muon.Get("recoil_uncorr_perp_qtgen_DYmumu_MiNNLO")
    thn_reco_para_muon = fIn_muon.Get("recoil_uncorr_para_qt_DYmumu_MiNNLO")
    thn_reco_perp_muon = fIn_muon.Get("recoil_uncorr_perp_qt_DYmumu_MiNNLO")
    thn_data_para_muon = fIn_muon.Get("recoil_uncorr_para_qt_singlemuon")
    thn_data_perp_muon = fIn_muon.Get("recoil_uncorr_perp_qt_singlemuon")
    
    thn_data_lep_pt_muon = fIn_muon.Get("lep_pt_ll_singlemuon")
    thn_reco_lep_pt_muon = fIn_muon.Get("lep_pt_ll_DYmumu_MiNNLO")

    thn_bkgs = {}
    for bkg in bkgs:
    
        thn_bkgs['para_%s_muon' % bkg] = fIn_muon.Get("recoil_uncorr_para_qt_%s" % bkg)
        thn_bkgs['perp_%s_muon' % bkg] = fIn_muon.Get("recoil_uncorr_perp_qt_%s" % bkg)
        thn_bkgs['lep_pt_ll_%s_muon' % bkg] = fIn_muon.Get("lep_pt_ll_%s" % bkg)
    
 
    fIn_electron = ROOT.TFile("/eos/user/j/jaeyserm/analysis/lowPU/Zee/output.root")
    thn_gen_para_electron = fIn_electron.Get("recoil_uncorr_para_qtgen_DYee_MiNNLO")
    thn_gen_perp_electron = fIn_electron.Get("recoil_uncorr_perp_qtgen_DYee_MiNNLO")
    thn_reco_para_electron = fIn_electron.Get("recoil_uncorr_para_qt_DYee_MiNNLO")
    thn_reco_perp_electron = fIn_electron.Get("recoil_uncorr_perp_qt_DYee_MiNNLO")
    thn_data_para_electron = fIn_electron.Get("recoil_uncorr_para_qt_singleelectron")
    thn_data_perp_electron = fIn_electron.Get("recoil_uncorr_perp_qt_singleelectron")
    thn_ttbar_para_electron = fIn_electron.Get("recoil_uncorr_para_qt_TTTo2L2Nu")
    thn_ttbar_perp_electron = fIn_electron.Get("recoil_uncorr_perp_qt_TTTo2L2Nu")
    
    thn_data_lep_pt_electron = fIn_electron.Get("lep_pt_ll_singleelectron")
    thn_reco_lep_pt_electron = fIn_electron.Get("lep_pt_ll_DYee_MiNNLO")

    for bkg in bkgs:
    
        thn_bkgs['para_%s_electron' % bkg] = fIn_electron.Get("recoil_uncorr_para_qt_%s" % bkg)
        thn_bkgs['perp_%s_electron' % bkg] = fIn_electron.Get("recoil_uncorr_perp_qt_%s" % bkg)
        thn_bkgs['lep_pt_ll_%s_electron' % bkg] = fIn_muon.Get("lep_pt_ll_%s" % bkg)

    fIn_muon.Close()
    fIn_electron.Close()


    para_mean_data = ROOT.TGraphErrors()
    para_mean_data.SetLineColor(ROOT.kBlue)
    para_mean_data.SetMarkerStyle(20)
    para_mean_data.SetMarkerSize(0.9)
    para_mean_data.SetMarkerColor(ROOT.kBlue)
    
    para_mean_reco = ROOT.TGraphErrors()
    para_mean_reco.SetLineColor(ROOT.kRed)
    para_mean_reco.SetMarkerStyle(20)
    para_mean_reco.SetMarkerSize(0.9)
    para_mean_reco.SetMarkerColor(ROOT.kRed)


    para_sigma_data = ROOT.TGraphErrors()
    para_sigma_data.SetLineColor(ROOT.kBlue)
    para_sigma_data.SetMarkerStyle(20)
    para_sigma_data.SetMarkerSize(0.9)
    para_sigma_data.SetMarkerColor(ROOT.kBlue)
    
    para_sigma_reco = ROOT.TGraphErrors()
    para_sigma_reco.SetLineColor(ROOT.kRed)
    para_sigma_reco.SetMarkerStyle(20)
    para_sigma_reco.SetMarkerSize(0.9)
    para_sigma_reco.SetMarkerColor(ROOT.kRed)
    
    
    perp_sigma_data = ROOT.TGraphErrors()
    perp_sigma_data.SetLineColor(ROOT.kBlue)
    perp_sigma_data.SetMarkerStyle(20)
    perp_sigma_data.SetMarkerSize(0.9)
    perp_sigma_data.SetMarkerColor(ROOT.kBlue)
    
    perp_sigma_reco = ROOT.TGraphErrors()
    perp_sigma_reco.SetLineColor(ROOT.kRed)
    perp_sigma_reco.SetMarkerStyle(20)
    perp_sigma_reco.SetMarkerSize(0.9)
    perp_sigma_reco.SetMarkerColor(ROOT.kRed)
    
    perp_mean_data = ROOT.TGraphErrors()
    perp_mean_data.SetLineColor(ROOT.kBlue)
    perp_mean_data.SetMarkerStyle(20)
    perp_mean_data.SetMarkerSize(0.9)
    perp_mean_data.SetMarkerColor(ROOT.kBlue)
    
    perp_mean_reco = ROOT.TGraphErrors()
    perp_mean_reco.SetLineColor(ROOT.kRed)
    perp_mean_reco.SetMarkerStyle(20)
    perp_mean_reco.SetMarkerSize(0.9)
    perp_mean_reco.SetMarkerColor(ROOT.kRed)
    
    hist_muon = thn_data_para_muon.Projection(0, "E")
    print(hist_muon.Integral())
    
    tt = 0

    for qTbin in range(0, len(qTbins)-1):
    
        qTbinMinGeV, qTbinMaxGeV = qTbins[qTbin], qTbins[qTbin+1]
        qTbinMin, qTbinMax = int(qTbins[qTbin]+1), int(qTbins[qTbin+1]) # proper indexing!
        qT = 0.5*(qTbinMinGeV + qTbinMaxGeV)

        print("***************** Do qTbin=[%f,%f],[%d, %d]" % (qTbinMinGeV, qTbinMaxGeV, qTbinMin, qTbinMax))
        

        #################### DATA
        thn_data_lep_pt_muon.GetAxis(0).SetRange(qTbinMin, qTbinMax)
        thn_data_lep_pt_electron.GetAxis(0).SetRange(qTbinMin, qTbinMax)
        h_lep_pt_muon = thn_data_lep_pt_muon.Projection(0, "E")
        h_lep_pt_electron = thn_data_lep_pt_electron.Projection(0, "E")
        h_lep_pt_muon.Add(h_lep_pt_electron)
        
        hist_bkg_qt = None
        for bkg in bkgs:
            
            thn_bkgs['lep_pt_ll_%s_muon' % bkg].GetAxis(0).SetRange(qTbinMin, qTbinMax)
            hist_muon_bkg = thn_bkgs['lep_pt_ll_%s_muon' % bkg].Projection(0, "E")
            hist_muon_bkg.SetName("lep_pt_ll_%s_%d" % (bkg, qTbinMin))
            
            thn_bkgs['lep_pt_ll_%s_electron' % bkg].GetAxis(0).SetRange(qTbinMin, qTbinMax)
            hist_electron_bkg = thn_bkgs['lep_pt_ll_%s_electron' % bkg].Projection(0, "E")
            hist_electron_bkg.SetName("lep_pt_ll_%s_%d" % (bkg, qTbinMin))
            
            if hist_bkg_qt == None: hist_bkg_qt = hist_muon_bkg
            else: hist_bkg_qt.Add(hist_muon_bkg)
            
            hist_bkg_qt.Add(hist_electron_bkg)

        hist_bkg_qt.Scale(lumi)
        h_lep_pt_muon.Add(hist_bkg_qt, -1)
        
        qT = h_lep_pt_muon.GetMean()
        qT_err = h_lep_pt_muon.GetRMS()
        
        # data - parallel
        thn_data_para_muon.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        thn_data_para_electron.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        hist_muon = thn_data_para_muon.Projection(0, "E")
        tt += hist_muon.Integral()
      
        hist_electron = thn_data_para_electron.Projection(0, "E")
        hist_muon.Add(hist_electron)

        hist_bkg = None
        for bkg in bkgs:
            
            thn_bkgs['para_%s_muon' % bkg].GetAxis(1).SetRange(qTbinMin, qTbinMax)
            hist_muon_bkg = thn_bkgs['para_%s_muon' % bkg].Projection(0, "E")
            hist_muon.SetName("muon_%s_%d" % (bkg, qTbinMin))
            
            thn_bkgs['para_%s_electron' % bkg].GetAxis(1).SetRange(qTbinMin, qTbinMax)
            hist_electron_bkg = thn_bkgs['para_%s_electron' % bkg].Projection(0, "E")
            hist_electron_bkg.SetName("muon_%s_%d" % (bkg, qTbinMin))
            
            if hist_bkg == None: hist_bkg = hist_muon_bkg
            else: hist_bkg.Add(hist_muon_bkg)
            
            hist_bkg.Add(hist_electron_bkg)

        hist_bkg.Scale(lumi)
        for iBin in range(0, hist_muon.GetNbinsX()+1):
            if hist_muon.GetBinContent(iBin) > 0:
                newVal = hist_muon.GetBinContent(iBin) - hist_bkg.GetBinContent(iBin)
                #if newVal < 0: newVal = 0
                hist_muon.SetBinContent(iBin, newVal)
        
        hist_muon.Rebin(2)      
        hist_muon.Scale(1./hist_muon.Integral())        
        para_sigma_data.SetPoint(qTbin, qT, hist_muon.GetRMS())
        para_sigma_data.SetPointError(qTbin, qT_err, hist_muon.GetRMSError())
        para_mean_data.SetPoint(qTbin, qT, hist_muon.GetMean())
        para_mean_data.SetPointError(qTbin, qT_err, hist_muon.GetMeanError())
        hist_muon.Delete()
        hist_electron.Delete()
        
 
        

        # data - perpendicular
        thn_data_perp_muon.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        thn_data_perp_electron.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        hist_muon = thn_data_perp_muon.Projection(0, "E")
        hist_electron = thn_data_perp_electron.Projection(0, "E")
        hist_muon.Add(hist_electron)

        hist_bkg = None
        for bkg in bkgs:
            
            thn_bkgs['para_%s_muon' % bkg].GetAxis(1).SetRange(qTbinMin, qTbinMax)
            hist_muon_bkg = thn_bkgs['para_%s_muon' % bkg].Projection(0, "E")
            hist_muon.SetName("muon_%s_%d" % (bkg, qTbinMin))
            
            thn_bkgs['para_%s_electron' % bkg].GetAxis(1).SetRange(qTbinMin, qTbinMax)
            hist_electron_bkg = thn_bkgs['para_%s_electron' % bkg].Projection(0, "E")
            hist_electron_bkg.SetName("muon_%s_%d" % (bkg, qTbinMin))
            
            if hist_bkg == None: hist_bkg = hist_muon_bkg
            else: hist_bkg.Add(hist_muon_bkg)
            
            hist_bkg.Add(hist_electron_bkg)

        hist_bkg.Scale(lumi)
        for iBin in range(0, hist_muon.GetNbinsX()+1):
            if hist_muon.GetBinContent(iBin) > 0:
                newVal = hist_muon.GetBinContent(iBin) - hist_bkg.GetBinContent(iBin)
                #if newVal < 0: newVal = 0
                hist_muon.SetBinContent(iBin, newVal)        
        
        
        hist_muon.Rebin(2)      
        hist_muon.Scale(1./hist_muon.Integral())    
        perp_sigma_data.SetPoint(qTbin, qT, hist_muon.GetRMS())
        perp_sigma_data.SetPointError(qTbin, qT_err, hist_muon.GetRMSError())
        perp_mean_data.SetPoint(qTbin, qT, hist_muon.GetMean())
        perp_mean_data.SetPointError(qTbin, qT_err, hist_muon.GetMeanError())
        hist_muon.Delete()
        hist_electron.Delete()

        
        #################### RECO-MC
        thn_reco_lep_pt_muon.GetAxis(0).SetRange(qTbinMin, qTbinMax)
        thn_reco_lep_pt_electron.GetAxis(0).SetRange(qTbinMin, qTbinMax)
        h_lep_pt_muon = thn_reco_lep_pt_muon.Projection(0, "E")
        h_lep_pt_electron = thn_reco_lep_pt_electron.Projection(0, "E")
        h_lep_pt_muon.Add(h_lep_pt_electron)
        qT = h_lep_pt_muon.GetMean()
        qT_err = h_lep_pt_muon.GetRMS()
        
        # mc - parallel
        thn_reco_para_muon.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        thn_reco_para_electron.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        hist_muon = thn_reco_para_muon.Projection(0, "E")
        hist_electron = thn_reco_para_electron.Projection(0, "E")
        hist_muon.Add(hist_electron)
        hist_muon.Scale(1./hist_muon.Integral())
        hist_muon.Rebin(2)
        para_sigma_reco.SetPoint(qTbin, qT, hist_muon.GetRMS())
        para_sigma_reco.SetPointError(qTbin, qT_err, hist_muon.GetRMSError())
        para_mean_reco.SetPoint(qTbin, qT, hist_muon.GetMean())
        para_mean_reco.SetPointError(qTbin, qT_err, hist_muon.GetMeanError())
        hist_muon.Delete()
        hist_electron.Delete()
        
        

        # mc - perpendicular
        thn_reco_perp_muon.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        thn_reco_perp_electron.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        hist_muon = thn_reco_perp_muon.Projection(0, "E")
        hist_electron = thn_reco_perp_electron.Projection(0, "E")
        hist_muon.Add(hist_electron)
        hist_muon.Scale(1./hist_muon.Integral())
        hist_muon.Rebin(2)
        perp_sigma_reco.SetPoint(qTbin, qT, hist_muon.GetRMS())
        perp_sigma_reco.SetPointError(qTbin, qT_err, hist_muon.GetRMSError())
        perp_mean_reco.SetPoint(qTbin, qT, hist_muon.GetMean())
        perp_mean_reco.SetPointError(qTbin, qT_err, hist_muon.GetMeanError())
        hist_muon.Delete()
        hist_electron.Delete()
    
    print(tt)

    if True:
    
        #fit_data_para = ROOT.TF1("fit_data_para", "[0]*TMath::Power(x,[1]) + [2]", 0, 150)
        #fit_data_para.SetParameters(8.80213e-02, 1, 8.56952e+00)
        fit_data_para = ROOT.TF1("fit_data_para", "[0]*x*x*x + [1]*x*x + [2]*x + [3]", 0, 260)
        #fit_data_para.FixParameter(1, 1)
        para_sigma_data.Fit("fit_data_para", "NSEW", "") # W: ignore empty bins
        fit_data_para.SetLineColor(ROOT.kBlue)
        fit_data_para.SetLineWidth(2)
    
        #fit_data = ROOT.TF1("fit_data", "[0]*x + [1]", 0, 150)
        #fit_data.SetParameters(1, 1)
        #para_sigma_data.Fit("fit_data", "0", "") # W: ignore empty bins
        #fit_data.SetLineColor(ROOT.kBlue)
        #fit_data.SetLineWidth(2)
        
        #fit_reco = ROOT.TF1("fit_reco", "[0]*x + [1]", 0, 150)
        #fit_reco.SetParameters(1, 1)
        #para_sigma_reco.Fit("fit_reco", "0", "") # W: ignore empty bins
        #fit_reco.SetLineColor(ROOT.kRed)
        #fit_reco.SetLineWidth(2)
        
        
        #fit_reco_para = ROOT.TF1("fit_reco_para", "[0]*TMath::Power(x,[1]) + [2]", 0, 150)
        #fit_reco_para.SetParameters(1, 0.5, 10)
        fit_reco_para = ROOT.TF1("fit_reco_para", "[0]*x*x*x + [1]*x*x + [2]*x + [3]", 0, 260)
        para_sigma_reco.Fit("fit_reco_para", "NSEW", "") # W: ignore empty bins
        fit_reco_para.SetLineColor(ROOT.kRed)
        fit_reco_para.SetLineWidth(2)
    
        cfg = {

            'logy'              : False,
            'logx'              : False,
            
            'xmin'              : 0,
            'xmax'              : 260,
            'ymin'              : 5,
            'ymax'              : 25,
                
            'xtitle'            : "q_{T} (GeV)",
            'ytitle'            : "#sigma(U_{#parallel}) (GeV)",
                
            'topRight'          : "199 pb^{#minus1} (13 TeV)", 
            'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
        }

        
        plotter.cfg = cfg
        canvas = plotter.canvas()
        dummy = plotter.dummy()   
        canvas.cd()
        dummy.Draw("HIST")
        
        leg = ROOT.TLegend(.20, 0.65, .6, .75)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.040)
        
        leg.AddEntry(para_sigma_data, "Data", "LP")
        leg.AddEntry(para_sigma_reco, "MC", "LP")
            
        para_sigma_data.Draw("PE SAME")
        para_sigma_reco.Draw("PE SAME")
        fit_data_para.Draw("L SAME")  
        fit_reco_para.Draw("L SAME")  
        plotter.aux(canvas)
           
        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.035)
        latex.SetTextColor(1)
        latex.SetTextFont(42)
        latex.DrawLatex(0.20, 0.90, label)
        latex.DrawLatex(0.20, 0.85, "%s" % (met_labels[met_estimators.index(met)]))
            
        leg.Draw("SAME")

        canvas.Modify()
        canvas.Update()
        canvas.Draw()
        canvas.RedrawAxis()
        canvas.SaveAs("%s/recoil_para_sigma_%s.png" % (outDir, met))
        canvas.SaveAs("%s/recoil_para_sigma_%s.pdf" % (outDir, met))
        canvas.Delete()



    if True:
    
        fit_data = ROOT.TF1("fit_data_perp", "[0]*TMath::Power(x,[1]) + [2]", 0, 150)
        fit_data.SetParameters(1, 1, 1)
        perp_sigma_data.Fit("fit_data_perp", "NSEW", "") # W: ignore empty bins
        fit_data.SetLineColor(ROOT.kBlue)
        fit_data.SetLineWidth(2)

        fit_reco = ROOT.TF1("fit_reco_perp", "[0]*TMath::Power(x,[1]) + [2]", 0, 150)
        fit_reco.SetParameters(1, 1, 1)
        perp_sigma_reco.Fit("fit_reco_perp", "NSEW", "") # W: ignore empty bins
        fit_reco.SetLineColor(ROOT.kRed)
        fit_reco.SetLineWidth(2)
    
        cfg = {

            'logy'              : False,
            'logx'              : False,
            
            'xmin'              : 0,
            'xmax'              : 200,
            'ymin'              : 5,
            'ymax'              : 25,
                
            'xtitle'            : "q_{T} (GeV)",
            'ytitle'            : "#sigma(U_{#perp}) (GeV)",
                
            'topRight'          : "199 pb^{#minus1} (13 TeV)", 
            'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
        }

        
        plotter.cfg = cfg
        canvas = plotter.canvas()
        dummy = plotter.dummy()   
        canvas.cd()
        dummy.Draw("HIST")
        
        leg = ROOT.TLegend(.20, 0.65, .6, .75)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.040)
        
        leg.AddEntry(para_sigma_data, "Data", "LP")
        leg.AddEntry(para_sigma_reco, "MC", "LP")
            
        perp_sigma_data.Draw("PE SAME")
        perp_sigma_reco.Draw("PE SAME")
        fit_data.Draw("L SAME")  
        fit_reco.Draw("L SAME")  
        plotter.aux(canvas)
           
        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.035)
        latex.SetTextColor(1)
        latex.SetTextFont(42)
        latex.DrawLatex(0.20, 0.90, label)
        latex.DrawLatex(0.20, 0.85, "%s" % (met_labels[met_estimators.index(met)]))
            
        leg.Draw("SAME")

        canvas.Modify()
        canvas.Update()
        canvas.Draw()
        canvas.RedrawAxis()
        canvas.SaveAs("%s/recoil_perp_sigma_%s.png" % (outDir, met))
        canvas.SaveAs("%s/recoil_perp_sigma_%s.pdf" % (outDir, met))
        canvas.Delete()



    if True:
    
    
        fit_data_para = ROOT.TF1("fit_data_para", "([0]*x*x + [1]*x)/([2]*x + 1)", 0, 200)
        fit_data_para.SetParameters(-6.02940e-02, -4.85091e-01, 6.36896e-02)
        para_mean_data.Fit("fit_data_para", "NSEW", "") # W: ignore empty bins
        fit_data_para.SetLineColor(ROOT.kBlue)
        fit_data_para.SetLineWidth(2)
        
        fit_reco_para = ROOT.TF1("fit_reco_para", "([0]*x*x + [1]*x)/([2]*x + 1)", 0, 200)
        fit_reco_para.SetParameters(-6.02940e-02, -4.85091e-01, 6.36896e-02)
        para_mean_reco.Fit("fit_reco_para", "NSEW", "") # W: ignore empty bins
        fit_reco_para.SetLineColor(ROOT.kRed)
        fit_reco_para.SetLineWidth(2)
    
        cfg = {

            'logy'              : False,
            'logx'              : False,
            
            'xmin'              : 0,
            'xmax'              : 200,
            'ymin'              : -200,
            'ymax'              : 20,
                
            'xtitle'            : "q_{T} (GeV)",
            'ytitle'            : "#LT U_{#parallel} #GT (GeV)",
                
            'topRight'          : "199 pb^{#minus1} (13 TeV)", 
            'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
        }

        
        plotter.cfg = cfg
        canvas = plotter.canvas()
        dummy = plotter.dummy()   
        canvas.cd()
        dummy.Draw("HIST")
        
        leg = ROOT.TLegend(.60, 0.65, .9, .75)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.040)
        
        leg.AddEntry(para_sigma_data, "Data", "LP")
        leg.AddEntry(para_sigma_reco, "MC", "LP")
            
        para_mean_data.Draw("PE SAME")
        para_mean_reco.Draw("PE SAME")
        fit_data_para.Draw("L SAME")  
        fit_reco_para.Draw("L SAME")  
        plotter.aux(canvas)
           
        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.035)
        latex.SetTextColor(1)
        latex.SetTextFont(42)
        latex.DrawLatex(0.20, 0.90, label)
        latex.DrawLatex(0.20, 0.85, "%s" % (met_labels[met_estimators.index(met)]))
            
        leg.Draw("SAME")

        canvas.Modify()
        canvas.Update()
        canvas.Draw()
        canvas.RedrawAxis()
        canvas.SaveAs("%s/recoil_para_mean_%s.png" % (outDir, met))
        canvas.SaveAs("%s/recoil_para_mean_%s.pdf" % (outDir, met))
        canvas.Delete()



    if True:
    
        cfg = {

            'logy'              : False,
            'logx'              : False,
            
            'xmin'              : 0,
            'xmax'              : 200,
            'ymin'              : -5,
            'ymax'              : 5,
                
            'xtitle'            : "q_{T} (GeV)",
            'ytitle'            : "#LT U_{#perp} #GT (GeV)",
                
            'topRight'          : "199 pb^{#minus1} (13 TeV)", 
            'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
        }

        
        plotter.cfg = cfg
        canvas = plotter.canvas()
        dummy = plotter.dummy()   
        canvas.cd()
        dummy.Draw("HIST")
        
        leg = ROOT.TLegend(.60, 0.65, .9, .75)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.040)
        
        leg.AddEntry(perp_sigma_data, "Data", "LP")
        leg.AddEntry(perp_sigma_reco, "MC", "LP")
            
        perp_mean_data.Draw("PE SAME")
        perp_mean_reco.Draw("PE SAME")
        plotter.aux(canvas)
           
        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.035)
        latex.SetTextColor(1)
        latex.SetTextFont(42)
        latex.DrawLatex(0.20, 0.90, label)
        latex.DrawLatex(0.20, 0.85, "%s" % (met_labels[met_estimators.index(met)]))
            
        leg.Draw("SAME")

        canvas.Modify()
        canvas.Update()
        canvas.Draw()
        canvas.RedrawAxis()
        canvas.SaveAs("%s/recoil_perp_mean_%s.png" % (outDir, met))
        canvas.SaveAs("%s/recoil_perp_mean_%s.pdf" % (outDir, met))
        canvas.Delete()


def doSingleFitOld(hIn, fit, cfg, fOut, label):

        
    g = ROOT.TGraphErrors()
    g.SetLineColor(ROOT.kBlack)
    g.SetMarkerStyle(20)
    g.SetMarkerSize(0.9)
    g.SetMarkerColor(ROOT.kBlack)
    g.SetLineColor(ROOT.kBlack)
           
    iPoint = 0
    for qTbin in range(1, len(qTbins)):
        
        x = 0.5*(qTbins[qTbin-1] + qTbins[qTbin])
        x_err = 0.5*(qTbins[qTbin] - qTbins[qTbin-1])
        y = hIn.GetBinContent(qTbin, idx, 1)
        y_err = hIn.GetBinContent(qTbin, idx, 0)
        g.SetPoint(iPoint, x, y)   
        g.SetPointError(iPoint, x_err, y_err)
        iPoint += 1
  
    result = g.Fit("fit", "NS", "", 0, 200) 
        
    fit.SetLineColor(ROOT.kRed)
    fit.GetXaxis().SetRangeUser(0, 200)
    fit.SetLineWidth(2)
        
        
    # error band
    values = result.GetConfidenceIntervals(0.68, False)
    interval = ROOT.TGraphErrors(g.GetN())
    interval_ratio = ROOT.TGraphErrors(g.GetN())
    for i in range(g.GetN()):
        
        interval.SetPoint(i, g.GetX()[i], fit.Eval(g.GetX()[i]))
        interval.SetPointError(i, 0, values[i])
            
        interval_ratio.SetPoint(i, g.GetX()[i], 1)
        interval_ratio.SetPointError(i, 0, values[i]/fit.Eval(g.GetX()[i]))
         
         
    # ratio 
    g_ratio = ROOT.TGraphErrors()
    g_ratio.SetLineColor(ROOT.kBlack)
    g_ratio.SetMarkerStyle(20)
    g_ratio.SetMarkerSize(0.9)
    g_ratio.SetMarkerColor(ROOT.kBlack)
    g_ratio.SetLineColor(ROOT.kBlack)
        
    iPoint = 0
    for qTbin in range(1, len(qTbins)):
        
        x = 0.5*(qTbins[qTbin-1] + qTbins[qTbin])
        x_err = 0.5*(qTbins[qTbin] - qTbins[qTbin-1])
        y = hIn.GetBinContent(qTbin, idx, 1)
        y_err = hIn.GetBinContent(qTbin, idx, 0)
        y_fit = fit.Eval(x)
        #print(x, y, y_fit, y/y_fit) 
            
        g_ratio.SetPoint(iPoint, x, y/y_fit)
        g_ratio.SetPointError(iPoint, x_err, y_err/abs(y_fit))
        iPoint += 1

        

    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio()
        
    ## TOP PAD ##
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")

        
    interval.SetFillColor(ROOT.kGreen)
    interval.Draw("3SAME")
    fit.Draw("L SAME")  
    g.Draw("PE SAME")

    plotter.auxRatio()
       
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.045)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.18, 0.87, label)
        
    ## BOTTOM PAD ##
    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.Draw("HIST")
        
    interval_ratio.SetFillColor(ROOT.kGreen)
    interval_ratio.Draw("3SAME")
    g_ratio.Draw("PE SAME")

    line = ROOT.TLine(cfg['xmin'], 1, cfg['xmax'], 1)
    line.SetLineColor(ROOT.kBlue+2)
    line.SetLineWidth(2)
    line.Draw("SAME")

    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s.png" % fOut)
    canvas.SaveAs("%s.pdf" % fOut)
    canvas.Delete()
        
def doSingleFit(hIn, fit, cfg, fOut, label, idx):

        
    g = ROOT.TGraphErrors()
    g.SetLineColor(ROOT.kBlack)
    g.SetMarkerStyle(20)
    g.SetMarkerSize(0.9)
    g.SetMarkerColor(ROOT.kBlack)
    g.SetLineColor(ROOT.kBlack)
           
    iPoint = 0
    for qTbin in range(1, len(qTbins)):
        
        x = 0.5*(qTbins[qTbin-1] + qTbins[qTbin])
        x_err = 0.5*(qTbins[qTbin] - qTbins[qTbin-1])
        y = hIn.GetBinContent(qTbin, idx, 1)
        y_err = hIn.GetBinContent(qTbin, idx, 0)
        g.SetPoint(iPoint, x, y)   
        g.SetPointError(iPoint, x_err, y_err)
        iPoint += 1
        
        #print(x, y)
  
    result = g.Fit(fit.GetName(), "NS", "", 0, 200) 
        
    fit.SetLineColor(ROOT.kRed)
    fit.GetXaxis().SetRangeUser(0, 200)
    fit.SetLineWidth(2)
        
    cov = result.GetCorrelationMatrix()
    cov.Print()
    # error band
    values = result.GetConfidenceIntervals(0.68, False)
    interval = ROOT.TGraphErrors(g.GetN())
    interval_ratio = ROOT.TGraphErrors(g.GetN())
    for i in range(g.GetN()):
        
        interval.SetPoint(i, g.GetX()[i], fit.Eval(g.GetX()[i]))
        interval.SetPointError(i, 0, values[i])
            
        interval_ratio.SetPoint(i, g.GetX()[i], 0)
        interval_ratio.SetPointError(i, 0, values[i]/fit.Eval(g.GetX()[i]))
         
         
    # ratio 
    g_ratio = ROOT.TGraphErrors()
    g_ratio.SetLineColor(ROOT.kBlack)
    g_ratio.SetMarkerStyle(20)
    g_ratio.SetMarkerSize(0.9)
    g_ratio.SetMarkerColor(ROOT.kBlack)
    g_ratio.SetLineColor(ROOT.kBlack)
        
    iPoint = 0
    for qTbin in range(1, len(qTbins)):
        
        x = 0.5*(qTbins[qTbin-1] + qTbins[qTbin])
        x_err = 0.5*(qTbins[qTbin] - qTbins[qTbin-1])
        y = hIn.GetBinContent(qTbin, idx, 1)
        y_err = hIn.GetBinContent(qTbin, idx, 0)
        y_fit = fit.Eval(x)
        #print(x, y, y_fit, y/y_fit)
        if y_err > 0: pull = (y-y_fit)/y_err
        else: pull = 0
        g_ratio.SetPoint(iPoint, x, pull)
        #g_ratio.SetPointError(iPoint, x_err, y_err/abs(y_fit))
        iPoint += 1

        

    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio()
        
    ## TOP PAD ##
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")

        
    interval.SetFillColor(ROOT.kGreen)
    interval.Draw("3SAME")
    fit.Draw("L SAME")  
    g.Draw("PE SAME")

    plotter.auxRatio()
       
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.045)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.18, 0.87, label)
        
    ## BOTTOM PAD ##
    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.Draw("HIST")
        
    interval_ratio.SetFillColor(ROOT.kGreen)
    interval_ratio.Draw("3SAME")
    g_ratio.Draw("PE SAME")

    line = ROOT.TLine(cfg['xmin'], 0, cfg['xmax'], 0)
    line.SetLineColor(ROOT.kBlue+2)
    line.SetLineWidth(2)
    line.Draw("SAME")

    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s.png" % fOut)
    canvas.SaveAs("%s.pdf" % fOut)
    canvas.Delete()
        

def compare():

    fIn_init = ROOT.TFile("wremnants/data/lowPU/recoil_fits_Z_param_init.root")
    fIn_refit = ROOT.TFile("wremnants/data/lowPU/recoil_fits_Z_param_refit.root")
    
    print("\n*** para_data ***\n")
    h_init = copy.deepcopy(fIn_init.Get("para_data"))
    h_refit = copy.deepcopy(fIn_refit.Get("para_data"))
    print("mean_a   \t %f \t %f" % (h_init.GetBinContent(1, 1), h_refit.GetBinContent(1, 1)))
    print("mean_b   \t %f \t %f" % (h_init.GetBinContent(1, 2), h_refit.GetBinContent(1, 2)))
    print("mean_c   \t %f \t %f" % (h_init.GetBinContent(1, 3), h_refit.GetBinContent(1, 3)))
    print("mean_d   \t %f \t %f" % (h_init.GetBinContent(1, 4), h_refit.GetBinContent(1, 4)))
    print("mean_e   \t %f \t %f" % (h_init.GetBinContent(1, 5), h_refit.GetBinContent(1, 5)))
    print("mean_f   \t %f \t %f" % (h_init.GetBinContent(1, 6), h_refit.GetBinContent(1, 6)))
    print("mean_g   \t %f \t %f" % (h_init.GetBinContent(1, 7), h_refit.GetBinContent(1, 7)))
    
    print("sigma1_a \t %f \t %f" % (h_init.GetBinContent(2, 1), h_refit.GetBinContent(2, 1)))
    print("sigma1_b \t %f \t %f" % (h_init.GetBinContent(2, 2), h_refit.GetBinContent(2, 2)))
    print("sigma1_c \t %f \t %f" % (h_init.GetBinContent(2, 3), h_refit.GetBinContent(2, 3)))
    
    print("sigma2_a \t %f \t %f" % (h_init.GetBinContent(3, 1), h_refit.GetBinContent(3, 1)))
    print("sigma2_b \t %f \t %f" % (h_init.GetBinContent(3, 2), h_refit.GetBinContent(3, 2)))
    print("sigma2_c \t %f \t %f" % (h_init.GetBinContent(3, 3), h_refit.GetBinContent(3, 3)))
    
    print("norm     \t %f \t %f" % (0.55, h_refit.GetBinContent(4, 1)))


    print("\n*** perp_data ***\n")
    h_init = copy.deepcopy(fIn_init.Get("perp_data"))
    h_refit = copy.deepcopy(fIn_refit.Get("perp_data"))    
    print("sigma1_a \t %f \t %f" % (h_init.GetBinContent(2, 1), h_refit.GetBinContent(2, 1)))
    print("sigma1_b \t %f \t %f" % (h_init.GetBinContent(2, 2), h_refit.GetBinContent(2, 2)))
    print("sigma1_c \t %f \t %f" % (h_init.GetBinContent(2, 3), h_refit.GetBinContent(2, 3)))
    
    print("sigma2_a \t %f \t %f" % (h_init.GetBinContent(3, 1), h_refit.GetBinContent(3, 1)))
    print("sigma2_b \t %f \t %f" % (h_init.GetBinContent(3, 2), h_refit.GetBinContent(3, 2)))
    print("sigma2_c \t %f \t %f" % (h_init.GetBinContent(3, 3), h_refit.GetBinContent(3, 3)))
    
    print("norm     \t %f \t %f" % (0.55, h_refit.GetBinContent(4, 1)))

    print("\n*** para_mc ***\n")
    h_init = copy.deepcopy(fIn_init.Get("para_mc"))
    h_refit = copy.deepcopy(fIn_refit.Get("para_mc"))
    print("mean_a   \t %f \t %f" % (h_init.GetBinContent(1, 1), h_refit.GetBinContent(1, 1)))
    print("mean_b   \t %f \t %f" % (h_init.GetBinContent(1, 2), h_refit.GetBinContent(1, 2)))
    print("mean_c   \t %f \t %f" % (h_init.GetBinContent(1, 3), h_refit.GetBinContent(1, 3)))
    print("mean_d   \t %f \t %f" % (h_init.GetBinContent(1, 4), h_refit.GetBinContent(1, 4)))
    print("mean_e   \t %f \t %f" % (h_init.GetBinContent(1, 5), h_refit.GetBinContent(1, 5)))
    
    print("sigma1_a \t %f \t %f" % (h_init.GetBinContent(2, 1), h_refit.GetBinContent(2, 1)))
    print("sigma1_b \t %f \t %f" % (h_init.GetBinContent(2, 2), h_refit.GetBinContent(2, 2)))
    print("sigma1_c \t %f \t %f" % (h_init.GetBinContent(2, 3), h_refit.GetBinContent(2, 3)))
    
    print("sigma2_a \t %f \t %f" % (h_init.GetBinContent(3, 1), h_refit.GetBinContent(3, 1)))
    print("sigma2_b \t %f \t %f" % (h_init.GetBinContent(3, 2), h_refit.GetBinContent(3, 2)))
    print("sigma2_c \t %f \t %f" % (h_init.GetBinContent(3, 3), h_refit.GetBinContent(3, 3)))
    
    print("norm     \t %f \t %f" % (0.55, h_refit.GetBinContent(4, 1)))


    print("\n*** perp_mc ***\n")
    h_init = copy.deepcopy(fIn_init.Get("perp_mc"))
    h_refit = copy.deepcopy(fIn_refit.Get("perp_mc"))    
    print("sigma1_a \t %f \t %f" % (h_init.GetBinContent(2, 1), h_refit.GetBinContent(2, 1)))
    print("sigma1_b \t %f \t %f" % (h_init.GetBinContent(2, 2), h_refit.GetBinContent(2, 2)))
    print("sigma1_c \t %f \t %f" % (h_init.GetBinContent(2, 3), h_refit.GetBinContent(2, 3)))
    
    print("sigma2_a \t %f \t %f" % (h_init.GetBinContent(3, 1), h_refit.GetBinContent(3, 1)))
    print("sigma2_b \t %f \t %f" % (h_init.GetBinContent(3, 2), h_refit.GetBinContent(3, 2)))
    print("sigma2_c \t %f \t %f" % (h_init.GetBinContent(3, 3), h_refit.GetBinContent(3, 3)))
    
    print("norm     \t %f \t %f" % (0.55, h_refit.GetBinContent(4, 1)))


def doPlot(hIn, fit, cfg, fOut, label, idx):

    '''
    g = ROOT.TGraphErrors()
    g.SetLineColor(ROOT.kBlack)
    g.SetMarkerStyle(20)
    g.SetMarkerSize(0.9)
    g.SetMarkerColor(ROOT.kBlack)
    g.SetLineColor(ROOT.kBlack)
           
    iPoint = 0
    for qTbin in range(1, len(qTbins)):
        
        x = 0.5*(qTbins[qTbin-1] + qTbins[qTbin])
        x_err = 0.5*(qTbins[qTbin] - qTbins[qTbin-1])
        y = hIn.GetBinContent(qTbin, idx, 1)
        y_err = hIn.GetBinContent(qTbin, idx, 0)
        g.SetPoint(iPoint, x, y)   
        g.SetPointError(iPoint, x_err, y_err)
        iPoint += 1
        
        #print(x, y)
  
    result = g.Fit(fit.GetName(), "NS", "", 0, 200) 
        
    fit.SetLineColor(ROOT.kRed)
    fit.GetXaxis().SetRangeUser(0, 200)
    fit.SetLineWidth(2)
        
    cov = result.GetCorrelationMatrix()
    cov.Print()
    # error band
    values = result.GetConfidenceIntervals(0.68, False)
    interval = ROOT.TGraphErrors(g.GetN())
    interval_ratio = ROOT.TGraphErrors(g.GetN())
    for i in range(g.GetN()):
        
        interval.SetPoint(i, g.GetX()[i], fit.Eval(g.GetX()[i]))
        interval.SetPointError(i, 0, values[i])
            
        interval_ratio.SetPoint(i, g.GetX()[i], 0)
        interval_ratio.SetPointError(i, 0, values[i]/fit.Eval(g.GetX()[i]))
         
         
    # ratio 
    g_ratio = ROOT.TGraphErrors()
    g_ratio.SetLineColor(ROOT.kBlack)
    g_ratio.SetMarkerStyle(20)
    g_ratio.SetMarkerSize(0.9)
    g_ratio.SetMarkerColor(ROOT.kBlack)
    g_ratio.SetLineColor(ROOT.kBlack)
        
    iPoint = 0
    for qTbin in range(1, len(qTbins)):
        
        x = 0.5*(qTbins[qTbin-1] + qTbins[qTbin])
        x_err = 0.5*(qTbins[qTbin] - qTbins[qTbin-1])
        y = hIn.GetBinContent(qTbin, idx, 1)
        y_err = hIn.GetBinContent(qTbin, idx, 0)
        y_fit = fit.Eval(x)
        #print(x, y, y_fit, y/y_fit)
        if y_err > 0: pull = (y-y_fit)/y_err
        else: pull = 0
        g_ratio.SetPoint(iPoint, x, pull)
        #g_ratio.SetPointError(iPoint, x_err, y_err/abs(y_fit))
        iPoint += 1
    '''
        

    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio()
        
    ## TOP PAD ##
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")

        
    #interval.SetFillColor(ROOT.kGreen)
    #interval.Draw("3SAME")
    fit.Draw("L SAME")  
    #g.Draw("PE SAME")

    plotter.auxRatio()
       
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.045)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.18, 0.87, label)
        
    ## BOTTOM PAD ##
    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.Draw("HIST")
        
    #interval_ratio.SetFillColor(ROOT.kGreen)
    #interval_ratio.Draw("3SAME")
    #g_ratio.Draw("PE SAME")

    line = ROOT.TLine(cfg['xmin'], 0, cfg['xmax'], 0)
    line.SetLineColor(ROOT.kBlue+2)
    line.SetLineWidth(2)
    line.Draw("SAME")

    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s.png" % fOut)
    canvas.SaveAs("%s.pdf" % fOut)
    canvas.Delete()
        

def plotParameterizations():

    fIn_refit = ROOT.TFile("wremnants/data/lowPU/recoil_fits_Z_param_refit.root")
  

    qTbins = list(range(0, 20, 1)) + list(range(20, 50, 2)) + list(range(50, 100, 5)) + list(range(100, 210, 10))
    qTbins = list(range(0, 30, 1)) + list(range(30, 60, 2)) + list(range(60, 100, 5)) + list(range(100, 210, 10))
    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/parameterization_refit/"
    
    functions.prepareDir(outDir)

    cfg = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : 200,
        'ymin'              : -200,
        'ymax'              : 20,
            
        'xtitle'            : "q_{T} (GeV)",
        'ytitle'            : "#mu(U_{#parallel}) (GeV)",
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
            
        'ratiofraction'     : 0.25,
        'ytitleR'           : "Ratio",
        'yminR'             : -6, # 0.88
        'ymaxR'             : 6, # 1.12
    } 
    
    
    

    ###################
    #### para_data_mean
    ###################
    idx = 1
    cat = "para_data_mean"
    hIn = copy.deepcopy(fIn_refit.Get("para_data"))
    
    #fit = ROOT.TF1("fit_%s" % cat, "([0]*x*x*x + [1]*x*x + [2]*x)/([3]*x*x + [4]*x + 1)", 0, 200)
    #fit.SetParameters(-2.45842e-03, -4.78214e-02, -5.35103e-01, 2.69659e-03, 6.19603e-02)
    
    sIdx = 5
    
    fit = ROOT.TF1("fit_%s" % cat, "([0]*x*x*x*x + [1]*x*x*x + [2]*x*x + [3]*x)/([4]*x*x*x + [5]*x*x + [6]*x + 1)", 0, 200)
    fit.SetParameters(hIn.GetBinContent(1, 1, sIdx), hIn.GetBinContent(1, 2, sIdx), hIn.GetBinContent(1, 3, sIdx), hIn.GetBinContent(1, 4, sIdx), hIn.GetBinContent(1, 5, sIdx), hIn.GetBinContent(1, 6, sIdx), hIn.GetBinContent(1, 7, sIdx))
    
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = -200, 20, "#mu(U_{#parallel}) (GeV)"
    doPlot(hIn, fit, cfg, "%s/%s" % (outDir, cat), "Data", idx)
    
    

    ########################
    #### para_sigma1_data
    ########################
    idx = 2
    cat = "para_sigma1_data"
    hIn = copy.deepcopy(fIn_refit.Get("para_data"))
    
    fit = ROOT.TF1("fit_%s" % cat, "[0]*TMath::Power(x+[1], [2])", 0, 200)
    fit.SetParameters(hIn.GetBinContent(2, 1, sIdx), hIn.GetBinContent(2, 2, sIdx), hIn.GetBinContent(2, 3, sIdx))
    
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = 0, 20, "#sigma_{1}(U_{#parallel}) (GeV)"
    doPlot(hIn, fit, cfg, "%s/%s" % (outDir, cat), "Data", idx)
  

if __name__ == "__main__":

    #compare()
    #plotParameterizations()
    #sys.exit()
    groups_mumu = Datagroups("mz_lowPU_mumu.pkl.lz4")
    groups_ee = Datagroups("mz_lowPU_ee.pkl.lz4")
    
    import decimal
    def drange(x, y, jump):
      while x < y:
        yield float(x)
        x += decimal.Decimal(jump)


    qTbins = list(range(0, 20, 1)) + list(range(20, 50, 2)) + list(range(50, 100, 5)) + list(range(100, 210, 10))
    qTbins = list(range(0, 30, 1)) + list(range(30, 60, 2)) + list(range(60, 100, 5)) + list(range(100, 210, 10))
    qTbins = list(drange(0, 30, 0.5)) + list(range(30, 60, 2)) + list(range(60, 100, 5)) + list(range(100, 210, 10)) 
    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/parameterization_new/"
    
    functions.prepareDir(outDir)

   
    
 
    h_para_data = ROOT.TH2D("para_data", "", 10, 0, 10, 20, 0, 20) # first idx = mean, sigma1, sigma2, norm; second idx = parameters of the fit
    h_para_mc = ROOT.TH2D("para_mc", "", 10, 0, 10, 20, 0, 20)
    h_perp_data = ROOT.TH2D("perp_data", "", 10, 0, 10, 20, 0, 20) # first idx = mean, sigma1, sigma2, norm; second idx = parameters of the fit
    h_perp_mc = ROOT.TH2D("perp_mc", "", 10, 0, 10, 20, 0, 20)


    cfg = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : 200,
        'ymin'              : -200,
        'ymax'              : 20,
            
        'xtitle'            : "q_{T} (GeV)",
        'ytitle'            : "#mu(U_{#parallel}) (GeV)",
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
            
        'ratiofraction'     : 0.25,
        'ytitleR'           : "Ratio",
        'yminR'             : -6, # 0.88
        'ymaxR'             : 6, # 1.12
    } 
    
    
    fIn = ROOT.TFile("wremnants/data/lowPU/recoil_fits_Z.root")


    ###################
    #### para_data_mean
    ###################
    idx = 1
    cat = "para_data_mean"
    hIn = fIn.Get("para_data")
    
    #fit = ROOT.TF1("fit_%s" % cat, "([0]*x*x*x + [1]*x*x + [2]*x)/([3]*x*x + [4]*x + 1)", 0, 200)
    #fit.SetParameters(-2.45842e-03, -4.78214e-02, -5.35103e-01, 2.69659e-03, 6.19603e-02)
    
    fit = ROOT.TF1("fit_%s" % cat, "([0]*x*x*x*x + [1]*x*x*x + [2]*x*x + [3]*x)/([4]*x*x*x + [5]*x*x + [6]*x + 1)", 0, 200)
    fit.SetParameters(-4.22258e-05, 2.08510e-03, -3.99666e-02, -4.37374e-01, 4.50535e-05, -1.90031e-03, 2.14798e-02)
    
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = -200, 20, "#mu(U_{#parallel}) (GeV)"
    doSingleFit(hIn, fit, cfg, "%s/%s" % (outDir, cat), "Data", idx)
    for iPar in range(0, fit.GetNpar()): h_para_data.SetBinContent(1, iPar+1, fit.GetParameter(iPar))
  
    ########################
    #### para_mc_mean
    ########################
    idx = 1
    cat = "para_mc_mean"
    hIn = fIn.Get("para_mc")
    
    fit = ROOT.TF1("fit_%s" % cat, "([0]*x*x*x + [1]*x*x + [2]*x)/([3]*x*x + [4]*x + 1)", 0, 200)
    fit.SetParameters(-2.45842e-03, -4.78214e-02, -5.35103e-01, 2.69659e-03, 6.19603e-02)
    
    fit = ROOT.TF1("fit_%s" % cat, "([0]*x*x*x*x + [1]*x*x*x + [2]*x*x + [3]*x)/([4]*x*x*x + [5]*x*x + [6]*x + 1)", 0, 200)
    fit.SetParameters(-3.19893e-03, -1.76589e-01, -6.65482e-01, -1.05425e+00, 3.49122e-03, 1.72273e-01, 2.03208e+00)
    #fit.FixParameter(5, 0)
    #fit.FixParameter(6, 0)
    
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = -200, 20, "#mu(U_{#parallel}) (GeV)"
    doSingleFit(hIn, fit, cfg, "%s/%s" % (outDir, cat), "DY", idx)
    for iPar in range(0, fit.GetNpar()): h_para_mc.SetBinContent(1, iPar+1, fit.GetParameter(iPar))


    ########################
    #### para_sigma1_data
    ########################
    idx = 2
    cat = "para_sigma1_data"
    hIn = fIn.Get("para_data")
    
    fit = ROOT.TF1("fit_%s" % cat, "[0]*TMath::Power(x+[1], [2])", 0, 200)
    fit.SetParameters(7.82921e-01, 2.52028e+01, 5.78980e-01)
    
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = 0, 20, "#sigma_{1}(U_{#parallel}) (GeV)"
    doSingleFit(hIn, fit, cfg, "%s/%s" % (outDir, cat), "Data", idx)
    for iPar in range(0, fit.GetNpar()): h_para_data.SetBinContent(2, iPar+1, fit.GetParameter(iPar))


    ########################
    #### para_sigma1_mc
    ########################
    idx = 2
    cat = "para_sigma1_mc"
    hIn = fIn.Get("para_mc")
    
    #fit = ROOT.TF1("fit_%s" % cat, "[0]*TMath::Power(x+[1], [2])", 0, 200)
    #fit.SetParameters(7.82921e-01, 2.52028e+01, 5.78980e-01)
    
    fit = ROOT.TF1("fit_%s" % cat, "[0]*TMath::Power(x+[1], [2])", 0, 200)
    fit.SetParameters(7.82921e-01, 2.52028e+01, 5.78980e-01)
    
    
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = 0, 20, "#sigma_{1}(U_{#parallel}) (GeV)"
    doSingleFit(hIn, fit, cfg, "%s/%s" % (outDir, cat), "DY", idx)
    for iPar in range(0, fit.GetNpar()): h_para_mc.SetBinContent(2, iPar+1, fit.GetParameter(iPar))


    ########################
    #### para_sigma2_data
    ########################
    idx = 3
    cat = "para_sigma2_data"
    hIn = fIn.Get("para_data")
    
    fit = ROOT.TF1("fit_%s" % cat, "[0]*TMath::Power(x+[1], [2])", 0, 200)
    fit.SetParameters(7.82921e-01, 2.52028e+01, 5.78980e-01)
    
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = 5, 35, "#sigma_{2}(U_{#parallel}) (GeV)"
    doSingleFit(hIn, fit, cfg, "%s/%s" % (outDir, cat), "Data", idx)
    for iPar in range(0, fit.GetNpar()): h_para_data.SetBinContent(3, iPar+1, fit.GetParameter(iPar))


    ########################
    #### para_sigma2_mc
    ########################
    idx = 3
    cat = "para_sigma2_mc"
    hIn = fIn.Get("para_mc")
    
    fit = ROOT.TF1("fit_%s" % cat, "[0]*TMath::Power(x+[1], [2])", 0, 200)
    fit.SetParameters(7.82921e-01, 2.52028e+01, 5.78980e-01)
    
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = 5, 35, "#sigma_{2}(U_{#parallel}) (GeV)"
    doSingleFit(hIn, fit, cfg, "%s/%s" % (outDir, cat), "DY", idx)
    for iPar in range(0, fit.GetNpar()): h_para_mc.SetBinContent(3, iPar+1, fit.GetParameter(iPar))


    
    ###################
    #### para_data_n
    ###################
    idx = 4
    cat = "para_data_norm"
    hIn = fIn.Get("para_data")
    

    fit = ROOT.TF1("fit_%s" % cat, "pol0", 0, 200)
    fit.SetParameters(0, 0.5)
    
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = 0, 1, "Norm"
    doSingleFit(hIn, fit, cfg, "%s/%s" % (outDir, cat), "Data", idx)
    for iPar in range(0, fit.GetNpar()): h_para_data.SetBinContent(4, iPar+1, fit.GetParameter(iPar))
    

    ###################
    #### para_mc_n
    ###################
    idx = 4
    cat = "para_mc_norm"
    hIn = fIn.Get("para_mc")
    

    fit = ROOT.TF1("fit_%s" % cat, "pol0", 0, 200)
    fit.SetParameters(0, 0.5)
    
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = 0, 1, "Norm"
    doSingleFit(hIn, fit, cfg, "%s/%s" % (outDir, cat), "DY", idx)
    for iPar in range(0, fit.GetNpar()): h_para_mc.SetBinContent(4, iPar+1, fit.GetParameter(iPar))
    
    
    
    




    ########################
    #### perp_sigma1_data
    ########################
    idx = 2
    cat = "perp_sigma1_data"
    hIn = fIn.Get("perp_data")
    
    fit = ROOT.TF1("fit_%s" % cat, "[0]*TMath::Power(x+[1], [2])", 0, 200)
    fit.SetParameters(7.82921e-01, 2.52028e+01, 5.78980e-01)
    
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = 0, 20, "#sigma_{1}(U_{#perp}) (GeV)"
    doSingleFit(hIn, fit, cfg, "%s/%s" % (outDir, cat), "Data", idx)
    for iPar in range(0, fit.GetNpar()): h_perp_data.SetBinContent(2, iPar+1, fit.GetParameter(iPar))


    ########################
    #### perp_sigma1_mc
    ########################
    idx = 2
    cat = "perp_sigma1_mc"
    hIn = fIn.Get("perp_mc")
    
    fit = ROOT.TF1("fit_%s" % cat, "[0]*TMath::Power(x+[1], [2])", 0, 200)
    fit.SetParameters(7.82921e-01, 2.52028e+01, 5.78980e-01)
    
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = 0, 20, "#sigma_{1}(U_{#perp}) (GeV)"
    doSingleFit(hIn, fit, cfg, "%s/%s" % (outDir, cat), "DY", idx)
    for iPar in range(0, fit.GetNpar()): h_perp_mc.SetBinContent(2, iPar+1, fit.GetParameter(iPar))



    ########################
    #### perp_sigma2_data
    ########################
    idx = 3
    cat = "perp_sigma2_data"
    hIn = fIn.Get("perp_data")
    
    fit = ROOT.TF1("fit_%s" % cat, "[0]*TMath::Power(x+[1], [2])", 0, 200)
    fit.SetParameters(7.82921e-01, 2.52028e+01, 5.78980e-01)
    
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = 0, 20, "#sigma_{2}(U_{#perp}) (GeV)"
    doSingleFit(hIn, fit, cfg, "%s/%s" % (outDir, cat), "Data", idx)
    for iPar in range(0, fit.GetNpar()): h_perp_data.SetBinContent(3, iPar+1, fit.GetParameter(iPar))


    ########################
    #### perp_sigma2_mc
    ########################
    idx = 3
    cat = "perp_sigma2_mc"
    hIn = fIn.Get("perp_mc")
    
    fit = ROOT.TF1("fit_%s" % cat, "[0]*TMath::Power(x+[1], [2])", 0, 200)
    fit.SetParameters(7.82921e-01, 2.52028e+01, 5.78980e-01)
    
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = 0, 20, "#sigma_{2}(U_{#perp}) (GeV)"
    doSingleFit(hIn, fit, cfg, "%s/%s" % (outDir, cat), "DY", idx)
    for iPar in range(0, fit.GetNpar()): h_perp_mc.SetBinContent(3, iPar+1, fit.GetParameter(iPar))
    
    
    ###################
    #### perp_data_n
    ###################
    idx = 4
    cat = "perp_data_norm"
    hIn = fIn.Get("perp_data")
    

    fit = ROOT.TF1("fit_%s" % cat, "pol0", 0, 200)
    fit.SetParameters(0, 0.5)
    
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = 0, 1, "Norm"
    doSingleFit(hIn, fit, cfg, "%s/%s" % (outDir, cat), "Data", idx)
    for iPar in range(0, fit.GetNpar()): h_para_data.SetBinContent(4, iPar+1, fit.GetParameter(iPar))
    

    ###################
    #### perp_mc_n
    ###################
    idx = 4
    cat = "perp_mc_norm"
    hIn = fIn.Get("perp_mc")
    

    fit = ROOT.TF1("fit_%s" % cat, "pol0", 0, 200)
    fit.SetParameters(0, 0.5)
    
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = 0, 1, "Norm"
    doSingleFit(hIn, fit, cfg, "%s/%s" % (outDir, cat), "DY", idx)
    for iPar in range(0, fit.GetNpar()): h_para_mc.SetBinContent(4, iPar+1, fit.GetParameter(iPar))
    

    
    fIn.Close()
    
    fOut_ = "wremnants/data/lowPU/recoil_fits_Z_param_init.root"
    fOut = ROOT.TFile(fOut_, "RECREATE")
    
    h_para_data.Write()
    h_para_mc.Write()
    h_perp_data.Write()
    h_perp_mc.Write()

    fOut.ls()
    fOut.Close()
    