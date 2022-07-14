import ROOT
import hist
import numpy as np
import copy
import logging
import sys
import decimal
from wremnants.common import data_dir

ROOT.gInterpreter.Declare('#include "lowpu_recoil.h"')

def drange(x, y, jump):
    while x < y:
        yield float(x)
        x += decimal.Decimal(jump)
        
        
class Recoil:

    def __init__(self, type_):
    
        if type_ == "highPU":
        
            self.recoil_qTbins = list(drange(0, 30, 0.5)) + list(range(30, 60, 2)) + list(range(60, 100, 5)) + list(range(100, 210, 10)) + [10000]
            rInput_binned = f"{data_dir}/recoil_fits_Z.root"
            rInput_parametric = ""
    
        elif type_ == "lowPU":
        
            self.recoil_qTbins = list(drange(0, 30, 0.5)) + list(range(30, 60, 2)) + list(range(60, 100, 5)) + list(range(100, 210, 10)) + [10000]
            self.recoil_qTbins = list(range(0, 50, 5)) + list(range(50, 100, 10)) + list(range(100, 200, 25)) + [10000]
            print(self.recoil_qTbins)
            rInput_binned = f"{data_dir}/lowPU/recoil_fits_Z.root"
            rInput_parametric = ""
            
        else: sys.exit("Recoil highPU or lowPU")

        # set the recoil correction bins in the analyzer
        qTbins_vec = ROOT.std.vector["float"]()
        for v in self.recoil_qTbins: qTbins_vec.push_back(v)
        setattr(ROOT.wrem, "qTbins", qTbins_vec)
        
        # load recoil hists
        ROOT.wrem.recoil_init(rInput_binned, f"{data_dir}/lowPU/recoil_fits_Z_param_refit.root")
        
        # define axes
        self.axis_MET_pt = hist.axis.Regular(300, 0, 300, name = "recoil_MET_pt", underflow=False)
        self.axis_MET_phi = hist.axis.Regular(50, 0, 4, name = "recoil_MET_phi")
        self.axis_recoil_magn = hist.axis.Regular(300, 0, 300, name = "recoil_magn")
        self.axis_recoil_para = hist.axis.Regular(1000, -500, 500, name = "recoil_para")
        self.axis_recoil_perp = hist.axis.Regular(1000, -500, 500, name = "recoil_perp")
        self.axis_qT = hist.axis.Variable(self.recoil_qTbins, name = "recoil_qT")
        self.axis_npv = hist.axis.Regular(50, 0.5, 50.5, name = "recoil_npv")


    def recoil_setup_Z(self, df, results, met_pt, met_phi, leptons_pt, leptons_phi, leptons_uncorr_pt, makeHists=True):

        df = df.Define("Lep1_mom2", "ROOT::Math::Polar2DVectorD(%s[0], %s[0])" % (leptons_pt, leptons_phi))
        df = df.Define("Lep2_mom2", "ROOT::Math::Polar2DVectorD(%s[1], %s[1])" % (leptons_pt, leptons_phi))
        df = df.Define("Z_mom2", "Lep1_mom2 + Lep2_mom2") # 2D vector sum of both leptons
        df = df.Define("qT", "Z_mom2.R()")
        
        df = df.Alias("MET_uncorr_pt", met_pt)
        df = df.Alias("MET_uncorr_phi", met_phi)

        df = df.Define("MET_corr_lep", "wrem::METLeptonCorrection(MET_uncorr_pt, MET_uncorr_phi, %s, %s, %s)" % (leptons_pt, leptons_uncorr_pt, leptons_phi))
        df = df.Define("MET_corr_lep_pt", "MET_corr_lep[0]")
        df = df.Define("MET_corr_lep_phi", "MET_corr_lep[1]")
        
        df = df.Define("recoil_uncorr", "wrem::recoilComponents(MET_corr_lep_pt, MET_corr_lep_phi, qT, Z_mom2.Phi())")
        df = df.Define("recoil_uncorr_magn", "recoil_uncorr[0]")
        df = df.Define("recoil_uncorr_para", "recoil_uncorr[1]")
        df = df.Define("recoil_uncorr_para_qT", "recoil_uncorr[1] + qT")
        df = df.Define("recoil_uncorr_perp", "recoil_uncorr[2]")
        
           
        
        if makeHists:
       
            results.append(df.HistoBoost("MET_uncorr_pt", [self.axis_MET_pt], ["MET_uncorr_pt", "nominal_weight"]))
            results.append(df.HistoBoost("MET_uncorr_phi", [self.axis_MET_phi], ["MET_uncorr_phi", "nominal_weight"]))
            results.append(df.HistoBoost("MET_corr_lep_pt", [self.axis_MET_pt], ["MET_corr_lep_pt", "nominal_weight"]))
            results.append(df.HistoBoost("MET_corr_lep_phi", [self.axis_MET_phi], ["MET_corr_lep_phi", "nominal_weight"]))
            
            results.append(df.HistoBoost("recoil_uncorr_magn", [self.axis_recoil_magn], ["recoil_uncorr_magn", "nominal_weight"]))
            results.append(df.HistoBoost("recoil_uncorr_para", [self.axis_recoil_para], ["recoil_uncorr_para", "nominal_weight"]))
            results.append(df.HistoBoost("recoil_uncorr_para_qT", [self.axis_recoil_perp], ["recoil_uncorr_para_qT", "nominal_weight"]))
            results.append(df.HistoBoost("recoil_uncorr_perp", [self.axis_recoil_perp], ["recoil_uncorr_perp", "nominal_weight"]))
            
            results.append(df.HistoBoost("recoil_uncorr_magn_qt", [self.axis_qT, self.axis_recoil_magn], ["qT", "recoil_uncorr_magn", "nominal_weight"]))
            results.append(df.HistoBoost("recoil_uncorr_para_qt", [self.axis_qT, self.axis_recoil_para], ["qT", "recoil_uncorr_para", "nominal_weight"]))
            results.append(df.HistoBoost("recoil_uncorr_perp_qt", [self.axis_qT, self.axis_recoil_perp], ["qT", "recoil_uncorr_perp", "nominal_weight"]))
            
            results.append(df.HistoBoost("recoil_uncorr_magn_npv", [self.axis_npv, self.axis_recoil_magn], ["PV_npvs", "recoil_uncorr_magn", "nominal_weight"]))
            results.append(df.HistoBoost("recoil_uncorr_para_npv", [self.axis_npv, self.axis_recoil_para], ["PV_npvs", "recoil_uncorr_para", "nominal_weight"]))
            results.append(df.HistoBoost("recoil_uncorr_perp_npv", [self.axis_npv, self.axis_recoil_perp], ["PV_npvs", "recoil_uncorr_perp", "nominal_weight"]))

            results.append(df.HistoBoost("qT", [self.axis_qT], ["qT", "nominal_weight"]))
            results.append(df.HistoBoost("npv", [self.axis_npv], ["PV_npvs", "nominal_weight"]))
            
        
        return df


    def recoil_apply_Z(self, df, results, dataset_name, datasets_to_apply, parametric=False): 

        if dataset_name in datasets_to_apply:
            df = df.Define("qTbin", "wrem::getqTbin(qT)")
            if parametric: df = df.Define("recoil_corr", "wrem::recoilCorrectionParametric(recoil_uncorr_para, recoil_uncorr_perp, qT)")
            else: df = df.Define("recoil_corr", "wrem::recoilCorrectionBinned(recoil_uncorr_para, recoil_uncorr_perp, qTbin)")
            df = df.Define("recoil_corr_magn", "recoil_corr[0]")
            df = df.Define("recoil_corr_para", "recoil_corr[1]")
            df = df.Define("recoil_corr_para_qT", "recoil_corr[1] + qT")
            df = df.Define("recoil_corr_perp", "recoil_corr[2]")
            
            df = df.Define("MET_corr_rec", "wrem::METCorrection(MET_corr_lep_pt, MET_corr_lep_phi, recoil_corr_para, recoil_corr_perp, qT, Z_mom2.Phi()) ")
            df = df.Define("MET_corr_rec_pt", "MET_corr_rec[0]")
            df = df.Define("MET_corr_rec_phi", "MET_corr_rec[1]")
            
            results.append(df.HistoBoost("GenMET_pt", [self.axis_MET_pt], ["GenMET_pt", "nominal_weight"]))

        else:
            df = df.Alias("recoil_corr_magn", "recoil_uncorr_magn")
            df = df.Alias("recoil_corr_para", "recoil_uncorr_para")
            df = df.Define("recoil_corr_para_qT", "recoil_uncorr_para + qT")
            df = df.Alias("recoil_corr_perp", "recoil_uncorr_perp")
            df = df.Alias("MET_corr_rec_pt", "MET_corr_lep_pt")
            df = df.Alias("MET_corr_rec_phi", "MET_corr_lep_phi")
          
        results.append(df.HistoBoost("recoil_corr_magn", [self.axis_recoil_magn], ["recoil_corr_magn", "nominal_weight"]))
        results.append(df.HistoBoost("recoil_corr_para", [self.axis_recoil_para], ["recoil_corr_para", "nominal_weight"]))
        results.append(df.HistoBoost("recoil_corr_para_qT", [self.axis_recoil_perp], ["recoil_corr_para_qT", "nominal_weight"]))
        results.append(df.HistoBoost("recoil_corr_perp", [self.axis_recoil_perp], ["recoil_corr_perp", "nominal_weight"]))
        results.append(df.HistoBoost("MET_corr_rec_pt", [self.axis_MET_pt], ["MET_corr_rec_pt", "nominal_weight"]))
        results.append(df.HistoBoost("MET_corr_rec_pt_qt", [self.axis_qT, self.axis_MET_pt], ["qT", "MET_corr_rec_pt", "nominal_weight"]))

        return df
        
    def recoil_apply_W(self, df, results, dataset_name, datasets_to_apply, parametric=False, auxPlots=False): 

        if dataset_name in datasets_to_apply:
            df = df.Define("qT_W", "ptVgen")
            df = df.Define("qTbin_W", "wrem::getqTbin(qT_W)")
            if parametric: df = df.Define("recoil_corr", "wrem::recoilCorrectionParametric(recoil_uncorr_para, recoil_uncorr_perp, qT_W)")
            else: df = df.Define("recoil_corr", "wrem::recoilCorrectionBinned(recoil_uncorr_para, recoil_uncorr_perp, qTbin_W)")
            df = df.Define("recoil_corr_magn", "recoil_corr[0]")
            df = df.Define("recoil_corr_para", "recoil_corr[1]")
            df = df.Define("recoil_corr_para_qT", "recoil_corr[1] + qT_W")
            df = df.Define("recoil_corr_perp", "recoil_corr[2]")
            
            df = df.Define("MET_corr_rec", "wrem::METCorrection(MET_corr_lep_pt, MET_corr_lep_phi, recoil_corr_para, recoil_corr_perp, qT_W, phiVgen) ")
            df = df.Define("MET_corr_rec_pt", "MET_corr_rec[0]")
            df = df.Define("MET_corr_rec_phi", "MET_corr_rec[1]")

        else:
            df = df.Alias("recoil_corr_magn", "recoil_uncorr_magn")
            df = df.Alias("recoil_corr_para", "recoil_uncorr_para")
            df = df.Define("recoil_corr_para_qT", "recoil_uncorr_para + qT_W")
            df = df.Alias("recoil_corr_perp", "recoil_uncorr_perp")
            df = df.Alias("MET_corr_rec_pt", "MET_corr_lep_pt")
            df = df.Alias("MET_corr_rec_phi", "MET_corr_lep_phi")
           
        if auxPlots:
            results.append(df.HistoBoost("recoil_corr_magn", [self.axis_recoil_magn], ["recoil_corr_magn", "nominal_weight"]))
            results.append(df.HistoBoost("recoil_corr_para", [self.axis_recoil_para], ["recoil_corr_para", "nominal_weight"]))
            results.append(df.HistoBoost("recoil_corr_para_qT", [self.axis_recoil_perp], ["recoil_corr_para_qT", "nominal_weight"]))
            results.append(df.HistoBoost("recoil_corr_perp", [self.axis_recoil_perp], ["recoil_corr_perp", "nominal_weight"]))
            results.append(df.HistoBoost("MET_corr_rec_pt", [self.axis_MET_pt], ["MET_corr_rec_pt", "nominal_weight"]))

        return df        
    

    def recoil_Z_statUnc_lowPU(self, df, results, axis_recoil_gen, axis_recoil_reco, axis_mt, axis_mll):
    
        # lowPU: propagate uncertainties to recoil, MET and mT
        
        axis_recoil_stat_unc = hist.axis.Regular(len(self.recoil_qTbins), 0, len(self.recoil_qTbins), underflow=False, overflow=False, name = "recoil_stat_unc_var")
        
        # make copies of axes for perturbation
        axis_recoil_reco_pert = hist.axis.Variable(axis_recoil_reco.edges, name = "recoil_reco_pert", underflow=False, overflow=True)
        axis_mt_pert = hist.axis.Variable(axis_mt.edges, name = "mt_pert",underflow=False, overflow=True)
        #axis_recoil_reco_pert = hist.axis.Variable([0, 5, 10, 15, 20, 30, 40, 50, 60, 75, 90, 150], name = "recoil_reco_pert", underflow=False, overflow=True)
        axis_MET_pert = hist.axis.Variable(self.axis_MET_pt.edges, name = "recoil_MET_pt_pert", underflow=False, overflow=True)
    
        # recoil stat uncertainty
        df = df.Define("recoil_corr_stat_idx", "wrem::indices_(%d, 0)" % (len(self.recoil_qTbins)))
        recoil_vars = [(1,2), (1,3), (1,4),   (2,2), (2,3),   (3,2), (3,3), (3,4),    (4,2), (4,3)]
        for k in recoil_vars:

            # perturbation for current qTbin
            df = df.Define("recoil_corr_stat_unc_%d_%d" % (k[0], k[1]), "wrem::recoilCorrectionBinned(recoil_uncorr_para, recoil_uncorr_perp, qTbin, %d, %d)" % (k[0], k[1]))
            
            df = df.Define("recoil_corr_magn_stat_unc_%d_%d" % (k[0], k[1]), "wrem::recoilCorrectionBinned_magn_StatUnc(recoil_corr_stat_unc_%d_%d, qTbin)" % (k[0], k[1]))
            results.append(df.HistoBoost("gen_reco_mll_recoilStatUnc_%d_%d" % (k[0], k[1]), [axis_recoil_gen, axis_recoil_reco, axis_recoil_reco_pert, axis_mll, axis_recoil_stat_unc], ["ptVgen", "recoil_corr_magn", "recoil_corr_magn_stat_unc_%d_%d" % (k[0], k[1]), "massZ", "recoil_corr_stat_idx", "nominal_weight"]))
            
            df = df.Define("recoil_corr_MET_stat_unc_%d_%d" % (k[0], k[1]), "wrem::recoilCorrectionBinned_MET_StatUnc(recoil_corr_stat_unc_%d_%d, qTbin, MET_corr_lep_pt, MET_corr_lep_phi, qT, Z_mom2.Phi())" % (k[0], k[1]))
            results.append(df.HistoBoost("MET_recoilStatUnc_%d_%d" % (k[0], k[1]), [self.axis_MET_pt, axis_MET_pert, axis_recoil_stat_unc], ["transverseMass", "recoil_corr_MET_stat_unc_%d_%d" % (k[0], k[1]), "recoil_corr_stat_idx", "nominal_weight"]))
            
            df = df.Define("recoil_corr_mt_stat_unc_%d_%d" % (k[0], k[1]), "wrem::recoilCorrectionBinned_mt_StatUnc(recoil_corr_MET_stat_unc_%d_%d, qTbin, TrigMuon_pt, TrigMuon_phi, NonTrigMuon_pt, NonTrigMuon_phi, MET_corr_rec_phi)" % (k[0], k[1]))
            results.append(df.HistoBoost("mt_recoilStatUnc_%d_%d" % (k[0], k[1]), [axis_mt, axis_mt_pert, axis_recoil_stat_unc], ["transverseMass", "recoil_corr_mt_stat_unc_%d_%d" % (k[0], k[1]), "recoil_corr_stat_idx", "nominal_weight"]))
        
        return df
