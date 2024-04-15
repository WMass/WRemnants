import ROOT
import hist
import numpy as np
import copy
import logging
import sys
import decimal
import json
import os
import array
from utilities import common as common
from utilities.io_tools import input_tools

import tensorflow as tf


ROOT.gInterpreter.Declare('#include "recoil_tools.h"')
ROOT.gInterpreter.Declare('#include "recoil_helper.h"')
logger = logging.getLogger("wremnants").getChild(__name__.split(".")[-1])


def RecoilCalibrationHelper(fIn, args):
    if not os.path.exists(fIn):
        logger.warning(f"Cannot find recoil tflite model {fIn}")
        logger.warning(f"Recoil corrections disabled")
        return None, None
    with open(fIn, 'rb') as f:
        model = f.read()
        interpreter = tf.lite.Interpreter(model_content=model)
        meta = interpreter.get_signature_runner('meta')()['output_00000_00000']
        nstat = int(meta[0])

    helper = ROOT.wrem.RecoilCalibrationHelper[nstat](fIn, "base_transform", args.recoilUnc)
    return helper, nstat

def RecoilCalibrationUncertaintyHelper(fIn, name, args):
    if not os.path.exists(fIn):
        return None

    try:
        helper = ROOT.wrem.RecoilCalibrationUncertaintyHelper(fIn, name)
    except:
        logger.warning(f"Cannot find recoil systematic {name}, skip")
        helper = None
    return helper

def VPTReweightHelper(fIn):
    if not os.path.exists(fIn):
        return None
    js = input_tools.read_json(fIn)
    helper = ROOT.wrem.VPTReweightHelper(js['vpt_bins'], js['weights'], min(js['vpt_bins']), max(js['vpt_bins']))
    return helper

def METXYCorrectionHelper(fIn):
    if not os.path.exists(fIn):
        logger.warning(f"Cannot find MET XY corrector file {fIn}")
        logger.warning(f"MET XY corrections disabled")
        return None, None
    js = input_tools.read_json(fIn)
    helper_data = ROOT.wrem.METXYCorrectionHelper(js['x']['data']['nom'], js['y']['data']['nom'])
    helper_mc = ROOT.wrem.METXYCorrectionHelper(js['x']['mc']['nom'], js['y']['mc']['nom'])
    return helper_data, helper_mc

class Recoil:

    def __init__(self, pu_type, args, flavor="mu"):

        self.met = args.met
        self.flavor = flavor
        self.args = args
        self.storeHists = args.recoilHists
        self.pu_type = pu_type
        self.isW = False

        self.met_xy_helper_data, self.met_xy_helper_mc = METXYCorrectionHelper(f"{common.data_dir}/recoil/{pu_type}_{self.met}/met_xy_{self.flavor}.json")
        self.vpt_reweight_helper_mc_data = VPTReweightHelper(f"{common.data_dir}/recoil/{pu_type}_{self.met}/vptrw_mc_data_mumu.json")

        recoil_tflite = f"{common.data_dir}/recoil/{pu_type}_{self.met}/model_mc_data.tflite"
        self.recoilHelper, self.nstat = RecoilCalibrationHelper(recoil_tflite, args)
        self.recoil_syst_bkg_para = RecoilCalibrationUncertaintyHelper(recoil_tflite, "syst_bkg_para", args)
        self.recoil_syst_bkg_perp = RecoilCalibrationUncertaintyHelper(recoil_tflite, "syst_bkg_perp", args)

        #self.recoil_pdf_data_para = RecoilCalibrationUncertaintyHelper(recoil_tflite, "pdf_data_para", args)
        #self.recoil_pdf_data_perp = RecoilCalibrationUncertaintyHelper(recoil_tflite, "pdf_data_perp", args)
        #self.recoil_pdf_mc_para = RecoilCalibrationUncertaintyHelper(recoil_tflite, "pdf_mc_para", args)
        #self.recoil_pdf_mc_perp = RecoilCalibrationUncertaintyHelper(recoil_tflite, "pdf_mc_perp", args)



        self.axis_MET_pt = hist.axis.Regular(200, 0, 200, name = "recoil_MET_pt", underflow=False)
        self.axis_MET_phi = hist.axis.Regular(50, -4, 4, name = "recoil_MET_phi")
        self.axis_MET_dphi = hist.axis.Regular(41, -4.1, 4.1, name = "recoil_MET_phi")
        self.axis_lep_pt = hist.axis.Regular(100, 0., 100., name = "pt", underflow=False)

        self.axis_run_no = hist.axis.Variable(list(range(277770, 284045, 1)) + [284045], name = "run_no")

        self.axis_recoil_magn = hist.axis.Regular(300, 0, 300, name = "recoil_magn", underflow=False)
        self.axis_recoil_para = hist.axis.Regular(400, -200, 200, name = "recoil_para")
        self.axis_recoil_para_qT = hist.axis.Regular(400, -300, 100, name = "recoil_para_qT")
        self.axis_recoil_perp = hist.axis.Regular(400, -200, 200, name = "recoil_perp")
        self.axis_MET_xy = hist.axis.Regular(200, -100, 100, name = "MET_xy")

        self.axis_mt = hist.axis.Regular(200, 0., 200., name = "mt", underflow=False)
        self.axis_njets = hist.axis.Regular(30, 0.5, 30.5, name = "recoil_njets")
        self.axis_npv = hist.axis.Regular(100, 0.5, 100.5, name = "recoil_npv")
        self.axis_sumEt = hist.axis.Regular(500, 0, 5000, name = "recoil_sumEt")
        self.axis_sumEt_sqrt = hist.axis.Regular(100, 0, 100, name = "recoil_sumEt_sqrt")
        self.axis_rapidity = hist.axis.Regular(24, -2.4, 2.4, name = "recoil_rapidity")

        self.axis_qt = hist.axis.Regular(300, 0, 300, name = "qt", underflow=False)
        self.axis_qt_gen = hist.axis.Regular(300, 0, 300, name = "qt_gen", underflow=False)
        self.axis_dphi = hist.axis.Regular(30, 0., np.pi, name = "dphi", underflow=False, overflow=False)
        self.axis_res_ratio = hist.axis.Regular(20000, 0, 2, name = "res", underflow=False, overflow=False)
        self.axis_res_diff = hist.axis.Regular(200000, -1, 1, name = "res", underflow=False, overflow=False)

        self.axis_charge = common.axis_charge
        self.axis_passIso = common.axis_passIso
        self.axis_passMT = common.axis_passMT

        self.axis_lep_pt_test = hist.axis.Regular(100, 0., 100., name = "pt_test", underflow=False)

        self.axis_recoil_2d_para = hist.axis.Regular(40, -10, 10, name = "axis_recoil_2d_para", underflow=False, overflow=False)
        self.axis_recoil_2d_perp = hist.axis.Regular(40, -10, 10, name = "axis_recoil_2d_perp", underflow=False, overflow=False)




    def add_histo(self, name, cols, axes, nominal_weight="nominal_weight", with_fakes=False, col_mt="mt_corr_rec"):
        if self.isW and with_fakes:
            if "mt" in [ax.name for ax in axes]: # remove "mt" from axes_fakerate in case mT is in cols/axes
                idx = [ax.name for ax in self.axes_fakerate].index("mt")
                cols_fakerate = self.cols_fakerate[:idx] + self.cols_fakerate[idx+1:]
                axes_fakerate = self.axes_fakerate[:idx] + self.axes_fakerate[idx+1:]
            else:
                cols_fakerate = [col_mt if x=="transverseMass" else x for x in self.cols_fakerate] # replace transverseMass with correct one
                axes_fakerate = self.axes_fakerate
            self.results.append(self.df.HistoBoost(name, axes+axes_fakerate, cols+cols_fakerate+[nominal_weight]))
        else:
            self.results.append(self.df.HistoBoost(name, axes, cols+[nominal_weight]))


    def recoil_Z(self, df, results, dataset, datasets_to_apply, leps_uncorr, leps_corr):

        self.isW = False
        self.df = df
        self.results = results
        self.dataset = dataset
        self.datasets_to_apply = datasets_to_apply
        
        self.df = self.df.Define("lep_uncorr_pt", f"wrem::Vec_d{{ {leps_uncorr[0]}, {leps_uncorr[4]} }}")
        self.df = self.df.Define("lep_uncorr_eta", f"wrem::Vec_d{{ {leps_uncorr[1]}, {leps_uncorr[5]} }}")
        self.df = self.df.Define("lep_uncorr_phi", f"wrem::Vec_d{{ {leps_uncorr[2]}, {leps_uncorr[6]} }}")
        self.df = self.df.Define("lep_uncorr_charge", f"wrem::Vec_i{{ {leps_uncorr[3]}, {leps_uncorr[7]} }}")
        
        # for consistency, do not propagate the muon calibration to the MET
        self.df = self.df.Alias("lep_corr_pt", "lep_uncorr_pt")
        self.df = self.df.Alias("lep_corr_eta", "lep_uncorr_eta")
        self.df = self.df.Alias("lep_corr_phi", "lep_uncorr_phi")
        self.df = self.df.Alias("lep_corr_charge", "lep_uncorr_charge")
        #self.df = self.df.Define("lep_corr_pt", f"wrem::Vec_d{{ {leps_corr[0]}, {leps_corr[4]} }}")
        #self.df = self.df.Define("lep_corr_eta", f"wrem::Vec_d{{ {leps_corr[1]}, {leps_corr[5]} }}")
        #self.df = self.df.Define("lep_corr_phi", f"wrem::Vec_d{{ {leps_corr[2]}, {leps_corr[6]} }}")
        #self.df = self.df.Define("lep_corr_charge", f"wrem::Vec_i{{ {leps_corr[3]}, {leps_corr[7]} }}")

        # for mT
        self.df = self.df.Define("lep_trg_pt", leps_corr[0])
        self.df = self.df.Define("lep_trg_eta", leps_corr[1])
        self.df = self.df.Define("lep_trg_phi", leps_corr[2])
        self.df = self.df.Define("lep_trg_charge", leps_corr[3])
        self.df = self.df.Define("lep_nontrg_pt", leps_corr[4])
        self.df = self.df.Define("lep_nontrg_eta", leps_corr[5])
        self.df = self.df.Define("lep_nontrg_phi", leps_corr[6])
        self.df = self.df.Define("lep_nontrg_charge", leps_corr[7])

        self.setup_gen_reco_vars_Z()
        self.setup_MET()
        self.setup_recoil_Z()
        self.setup_recoil_gen()
        self.apply_recoil_Z()
        if self.args.recoilUnc:
            self.setup_recoil_Z_unc()
        if self.storeHists:
            self.auxHists()

        return self.df

    def recoil_W(self, df, results, dataset, datasets_to_apply, leps_uncorr, leps_corr, cols_fakerate=[], axes_fakerate=[], mtw_min=40):

        self.isW = True
        self.df = df
        self.results = results
        self.dataset = dataset
        self.datasets_to_apply = datasets_to_apply
        self.mtw_min = mtw_min

        self.df = self.df.Define("lep_uncorr_pt", leps_uncorr[0])
        self.df = self.df.Define("lep_uncorr_eta", leps_uncorr[1])
        self.df = self.df.Define("lep_uncorr_phi", leps_uncorr[2])
        self.df = self.df.Define("lep_uncorr_charge", leps_uncorr[3])
        
        # for consistency, do not propagate the muon calibration to the MET
        self.df = self.df.Alias("lep_corr_pt", "lep_uncorr_pt")
        self.df = self.df.Alias("lep_corr_eta", "lep_uncorr_eta")
        self.df = self.df.Alias("lep_corr_phi", "lep_uncorr_phi")
        self.df = self.df.Alias("lep_corr_charge", "lep_uncorr_charge")
        #self.df = self.df.Define("lep_corr_pt", leps_corr[0])
        #self.df = self.df.Define("lep_corr_eta", leps_corr[1])
        #self.df = self.df.Define("lep_corr_phi", leps_corr[2])
        #self.df = self.df.Define("lep_corr_charge", leps_corr[3])
        #self.df = self.df.Define("lep_corr_pt", "lep_corr_pt_ + 0.5")

        # for mT
        self.df = self.df.Define("lep_trg_pt", leps_corr[0])
        self.df = self.df.Define("lep_trg_eta", leps_corr[1])
        self.df = self.df.Define("lep_trg_phi", leps_corr[2])
        self.df = self.df.Define("lep_trg_charge", leps_corr[3])

        self.cols_fakerate = cols_fakerate
        self.axes_fakerate = axes_fakerate

        self.setup_gen_reco_vars_W()
        self.setup_MET()
        self.setup_recoil_gen()
        self.apply_recoil_W()
        if self.args.recoilUnc:
            self.setup_recoil_W_unc()
        return self.df


    def setup_MET(self):

        # for the Z, leptons_pt, leptons_phi, leptons_uncorr_pt should be Vec_f with size 2
        # for the W, it should contain only a double

        if self.met == "PFMET": met_pt, met_phi = "MET_pt", "MET_phi"
        elif self.met == "RawPFMET": met_pt, met_phi = "RawMET_pt", "RawMET_phi"
        elif self.met == "DeepMETReso": met_pt, met_phi = "DeepMETResolutionTune_pt", "DeepMETResolutionTune_phi"
        elif self.met == "DeepMETPVRobust" and self.pu_type == "highPU": met_pt, met_phi = "DeepMETPVRobust_pt", "DeepMETPVRobust_phi"
        elif self.met == "DeepMETPVRobustNoPUPPI" and self.pu_type == "highPU": met_pt, met_phi = "DeepMETPVRobustNoPUPPI_pt", "DeepMETPVRobustNoPUPPI_phi"
        else:
            logger.warning(f"MET definition {self.met} not found, fall back to RawPFMET")
            met_pt, met_phi = "RawMET_pt", "RawMET_phi"

        # uncorrected MET
        self.df = self.df.Define("met_uncorr_pt", f"return (std::isfinite({met_pt})) ? {met_pt} : 10000")
        self.df = self.df.Define("met_uncorr_phi", f"return (std::isfinite({met_phi})) ? {met_phi} : 0")
        self.df = self.df.Define("met_uncorr_x", "met_uncorr_pt*cos(met_uncorr_phi)")
        self.df = self.df.Define("met_uncorr_y", "met_uncorr_pt*sin(met_uncorr_phi)")

        # lepton corrected MET
        self.df = self.df.Define("met_corr_lep", f"wrem::met_lepton_correction(met_uncorr_pt, met_uncorr_phi, lep_uncorr_pt, lep_uncorr_phi, lep_corr_pt, lep_corr_phi)")
        self.df = self.df.Define("met_corr_lep_pt", "met_corr_lep[0]")
        self.df = self.df.Define("met_corr_lep_phi", "met_corr_lep[1]")
        self.df = self.df.Define("met_corr_lep_x", "met_corr_lep_pt*cos(met_corr_lep_phi)")
        self.df = self.df.Define("met_corr_lep_y", "met_corr_lep_pt*sin(met_corr_lep_phi)")


        # phi corrected MET (XY corrections)
        if self.met_xy_helper_data != None and self.met_xy_helper_mc != None:
            self.df = self.df.Define("met_corr_xy", self.met_xy_helper_data if self.dataset.is_data else self.met_xy_helper_mc, ["met_corr_lep_pt", "met_corr_lep_phi", "PV_npvs"])
            self.df = self.df.Define("met_corr_xy_pt", "met_corr_xy[0]")
            self.df = self.df.Define("met_corr_xy_phi", "met_corr_xy[1]")
        else:
            self.df = self.df.Alias("met_corr_xy_pt", "met_corr_lep_pt")
            self.df = self.df.Alias("met_corr_xy_phi", "met_corr_lep_phi")
        self.df = self.df.Define("met_corr_xy_x", "met_corr_xy_pt*cos(met_corr_xy_phi)")
        self.df = self.df.Define("met_corr_xy_y", "met_corr_xy_pt*sin(met_corr_xy_phi)")

        if not self.storeHists: 
            return

        # histograms as function of npv, to derive/closure the XY correction
        self.add_histo("met_corr_lep_x_npv", ["PV_npvs", "met_corr_lep_x"], [self.axis_npv, self.axis_MET_xy])
        self.add_histo("met_corr_lep_y_npv", ["PV_npvs", "met_corr_lep_y"], [self.axis_npv, self.axis_MET_xy])

        self.add_histo("met_corr_xy_x_npv", ["PV_npvs", "met_corr_xy_x"], [self.axis_npv, self.axis_MET_xy])
        self.add_histo("met_corr_xy_y_npv", ["PV_npvs", "met_corr_xy_y"], [self.axis_npv, self.axis_MET_xy])
        
        
        self.df = self.df.Define("lep_pt_uncorr_over_corr", "lep_uncorr_pt/lep_corr_pt")
        self.df = self.df.Define("lep_pt_uncorr_minus_corr", "lep_uncorr_pt-lep_corr_pt")
        self.df = self.df.Define("lep_phi_uncorr_minus_corr", "lep_uncorr_phi-lep_corr_phi")
        self.df = self.df.Define("met_pt_uncorr_over_corr_lep", "met_uncorr_pt/met_corr_lep_pt")
        self.df = self.df.Define("met_pt_uncorr_minus_corr_lep", "met_uncorr_pt-met_corr_lep_pt")
        self.df = self.df.Define("met_phi_uncorr_minus_corr_lep", "met_uncorr_phi-met_corr_lep_phi")
        
        self.add_histo("lep_pt_uncorr_over_corr", ["lep_pt_uncorr_over_corr"], [self.axis_res_ratio])
        self.add_histo("lep_pt_uncorr_minus_corr", ["lep_pt_uncorr_minus_corr"], [self.axis_res_diff])
        self.add_histo("lep_phi_uncorr_minus_corr", ["lep_phi_uncorr_minus_corr"], [self.axis_res_diff])
        
        self.add_histo("met_pt_uncorr_over_corr_lep", ["met_pt_uncorr_over_corr_lep"], [self.axis_res_ratio])
        self.add_histo("met_pt_uncorr_minus_corr_lep", ["met_pt_uncorr_minus_corr_lep"], [self.axis_res_diff])
        self.add_histo("met_phi_uncorr_minus_corr_lep", ["met_phi_uncorr_minus_corr_lep"], [self.axis_res_diff])


    def recoil_vars_plots_Z(self, rec_corr, nominal_weight="nominal_weight", suffix=""):

        if not self.df.HasColumn(f"recoil_{rec_corr}_magn"):
            self.df = self.df.Define(f"recoil_{rec_corr}_magn", f"std::hypot(recoil_{rec_corr}_para_qt, recoil_{rec_corr}_perp)")
            self.df = self.df.Define(f"met_{rec_corr}_wlike", f"wrem::get_met_wlike(lep_nontrg_pt, lep_nontrg_phi, met_{rec_corr}_pt, met_{rec_corr}_phi)")
            self.df = self.df.Define(f"met_{rec_corr}_pt_wlike", f"met_{rec_corr}_wlike.Mod()")
            self.df = self.df.Define(f"met_{rec_corr}_phi_wlike", f"met_{rec_corr}_wlike.Phi()")
            self.df = self.df.Define(f"mt_{rec_corr}", f"wrem::get_mt_wlike(lep_trg_pt, lep_trg_phi, met_{rec_corr}_wlike)")
            self.df = self.df.Define(f"dphi_{rec_corr}", f"std::abs(wrem::deltaPhi(v_phi, met_{rec_corr}_phi))")
            self.df = self.df.Define(f"dphi_{rec_corr}_wlike", f"std::abs(wrem::deltaPhi(lep_trg_phi, met_{rec_corr}_phi_wlike))")

        if not self.storeHists: 
            return

        suffix = f"_{suffix}" if suffix != "" else suffix
        self.add_histo(f"recoil_{rec_corr}_magn{suffix}", [f"recoil_{rec_corr}_magn"], [self.axis_recoil_magn], nominal_weight=nominal_weight)
        self.add_histo(f"recoil_{rec_corr}_para_qt{suffix}", [f"recoil_{rec_corr}_para_qt"], [self.axis_recoil_para_qT], nominal_weight=nominal_weight)
        self.add_histo(f"recoil_{rec_corr}_para{suffix}", [f"recoil_{rec_corr}_para"], [self.axis_recoil_para], nominal_weight=nominal_weight)
        self.add_histo(f"recoil_{rec_corr}_perp{suffix}", [f"recoil_{rec_corr}_perp"], [self.axis_recoil_perp], nominal_weight=nominal_weight)

        self.add_histo(f"mt_{rec_corr}{suffix}", [f"mt_{rec_corr}"], [self.axis_mt], nominal_weight=nominal_weight)
        self.add_histo(f"met_{rec_corr}_pt{suffix}", [f"met_{rec_corr}_pt"], [self.axis_MET_pt], nominal_weight=nominal_weight)
        self.add_histo(f"met_{rec_corr}_phi{suffix}", [f"met_{rec_corr}_phi"], [self.axis_MET_phi], nominal_weight=nominal_weight)
        self.add_histo(f"met_{rec_corr}_pt_wlike{suffix}", [f"met_{rec_corr}_pt_wlike"], [self.axis_MET_pt], nominal_weight=nominal_weight)
        self.add_histo(f"met_{rec_corr}_phi_wlike{suffix}", [f"met_{rec_corr}_phi_wlike"], [self.axis_MET_phi], nominal_weight=nominal_weight)
        self.add_histo(f"dphi_{rec_corr}{suffix}", [f"dphi_{rec_corr}"], [self.axis_dphi], nominal_weight=nominal_weight)
        self.add_histo(f"dphi_{rec_corr}_wlike{suffix}", [f"dphi_{rec_corr}_wlike"], [self.axis_dphi], nominal_weight=nominal_weight)
        self.add_histo(f"met_{rec_corr}_x{suffix}", [f"met_{rec_corr}_x"], [self.axis_MET_xy], nominal_weight=nominal_weight)
        self.add_histo(f"met_{rec_corr}_y{suffix}", [f"met_{rec_corr}_y"], [self.axis_MET_xy], nominal_weight=nominal_weight)

    def recoil_vars_plots_W(self, rec_corr, nominal_weight="nominal_weight", suffix=""):

        if not self.df.HasColumn(f"recoil_{rec_corr}_magn"):
            
            self.df = self.df.Define(f"recoil_{rec_corr}_magn", f"wrem::compute_recoil_from_met_and_lepton(met_{rec_corr}_pt, met_{rec_corr}_phi, lep_trg_pt, lep_trg_phi)")
            self.df = self.df.Define(f"mt_{rec_corr}", f"wrem::mt_2(lep_trg_pt, lep_trg_phi, met_{rec_corr}_pt, met_{rec_corr}_phi)")
            self.df = self.df.Define(f"dphi_{rec_corr}", f"std::abs(wrem::deltaPhi(lep_trg_phi, met_{rec_corr}_phi))")
            #self.df = self.df.Define(f"passMT_{rec_corr}", f"mt_{rec_corr} > {self.mtw_min}")

        if not self.storeHists: 
            return

        suffix = f"_{suffix}" if suffix != "" else suffix
        self.add_histo(f"recoil_{rec_corr}_magn{suffix}", [f"recoil_{rec_corr}_magn"], [self.axis_recoil_magn], nominal_weight=nominal_weight, with_fakes=True, col_mt=f"mt_{rec_corr}")

        self.add_histo(f"mt_{rec_corr}{suffix}", [f"mt_{rec_corr}"], [self.axis_mt], nominal_weight=nominal_weight, with_fakes=True, col_mt=f"mt_{rec_corr}")
        self.add_histo(f"met_{rec_corr}_pt{suffix}", [f"met_{rec_corr}_pt"], [self.axis_MET_pt], nominal_weight=nominal_weight, with_fakes=True, col_mt=f"mt_{rec_corr}")
        self.add_histo(f"met_{rec_corr}_phi{suffix}", [f"met_{rec_corr}_phi"], [self.axis_MET_phi], nominal_weight=nominal_weight, with_fakes=True, col_mt=f"mt_{rec_corr}")
        self.add_histo(f"dphi_{rec_corr}{suffix}", [f"dphi_{rec_corr}"], [self.axis_dphi], nominal_weight=nominal_weight, with_fakes=True, col_mt=f"mt_{rec_corr}")
        self.add_histo(f"lep_{rec_corr}_pt{suffix}", ["lep_trg_pt"], [self.axis_lep_pt_test], with_fakes=True, col_mt=f"mt_{rec_corr}")

    def setup_recoil_Z(self):

        # uncorrected recoil
        self.df = self.df.Define("recoil_uncorr", "wrem::compute_recoil_from_met(met_uncorr_pt, met_uncorr_phi, lep_corr_pt, lep_corr_phi, v_pt, v_phi)")
        self.df = self.df.Define("recoil_uncorr_para", "recoil_uncorr[0]")
        self.df = self.df.Define("recoil_uncorr_perp", "recoil_uncorr[1]")
        self.df = self.df.Define("recoil_uncorr_para_qt", "recoil_uncorr_para - v_pt")

        # lep corrected recoil
        self.df = self.df.Define("recoil_corr_lep", "wrem::compute_recoil_from_met(met_corr_lep_pt, met_corr_lep_phi, lep_corr_pt, lep_corr_phi, v_pt, v_phi)")
        self.df = self.df.Define("recoil_corr_lep_para", "recoil_corr_lep[0]")
        self.df = self.df.Define("recoil_corr_lep_perp", "recoil_corr_lep[1]")
        self.df = self.df.Define("recoil_corr_lep_para_qt", "recoil_corr_lep_para - v_pt")

        # MET XY corrected recoil
        self.df = self.df.Define("recoil_corr_xy", "wrem::compute_recoil_from_met(met_corr_xy_pt, met_corr_xy_phi, lep_corr_pt, lep_corr_phi, v_pt, v_phi)")
        self.df = self.df.Define("recoil_corr_xy_para", "recoil_corr_xy[0]")
        self.df = self.df.Define("recoil_corr_xy_perp", "recoil_corr_xy[1]")
        self.df = self.df.Define("recoil_corr_xy_para_qt", "recoil_corr_xy_para - v_pt")


        self.recoil_vars_plots_Z("uncorr")
        self.recoil_vars_plots_Z("corr_lep")
        self.recoil_vars_plots_Z("corr_xy")
        self.recoil_vars_plots_Z("corr_xy", suffix="qtrw", nominal_weight="nominal_weight_vptrw_mc_data")

        if not self.storeHists: 
            return

        # recoil components binned in various parameters
        self.add_histo("recoil_corr_xy_para_v_pt", ["v_pt", "recoil_corr_xy_para"], [self.axis_qt, self.axis_recoil_para])
        self.add_histo("recoil_corr_xy_perp_v_pt", ["v_pt", "recoil_corr_xy_perp"], [self.axis_qt, self.axis_recoil_perp])

        self.add_histo("recoil_corr_xy_para_npv", ["PV_npvs", "recoil_corr_xy_para"], [self.axis_npv, self.axis_recoil_para])
        self.add_histo("recoil_corr_xy_perp_npv", ["PV_npvs", "recoil_corr_xy_perp"], [self.axis_npv, self.axis_recoil_perp])

        self.add_histo("recoil_corr_xy_para_y", ["v_y", "recoil_corr_xy_para"], [self.axis_rapidity, self.axis_recoil_para])
        self.add_histo("recoil_corr_xy_perp_y", ["v_y", "recoil_corr_xy_perp"], [self.axis_rapidity, self.axis_recoil_perp])

        self.add_histo("recoil_corr_xy_para_sumEt", ["RawMET_sumEt", "recoil_corr_xy_para"], [self.axis_sumEt, self.axis_recoil_para])
        self.add_histo("recoil_corr_xy_perp_sumEt", ["RawMET_sumEt", "recoil_corr_xy_perp"], [self.axis_sumEt, self.axis_recoil_perp])

        self.add_histo("recoil_corr_xy_para_perp", ["recoil_corr_xy_para", "recoil_corr_xy_perp"], [self.axis_recoil_para, self.axis_recoil_perp])
        self.add_histo("v_pt", ["v_pt"], [self.axis_qt])
        self.add_histo("v_pt_qtrw", ["v_pt"], [self.axis_qt], nominal_weight="nominal_weight_vptrw_mc_data")




    def auxHists(self):

        self.df = self.df.Define("RawMET_sumEt_sqrt", "sqrt(RawMET_sumEt)")
        self.df = self.df.Define("njets", "Jet_pt.size()")
        self.add_histo("njets", ["njets"], [self.axis_njets])
        self.add_histo("RawMET_sumEt", ["RawMET_sumEt"], [self.axis_sumEt])

        self.add_histo("npv", ["PV_npvs"], [self.axis_npv])
        self.add_histo("npv_RawMET_sumEt", ["PV_npvs", "RawMET_sumEt"], [self.axis_npv, self.axis_sumEt])
        self.add_histo("qT_sumEt", ["v_pt", "RawMET_sumEt"], [self.axis_qt, self.axis_sumEt])

        self.add_histo("recoil_corr_xy_para_qT_njets", ["njets", "recoil_corr_xy_para_qt"], [self.axis_njets, self.axis_recoil_para])
        self.add_histo("recoil_corr_xy_perp_njets", ["njets", "recoil_corr_xy_perp"], [self.axis_njets, self.axis_recoil_perp])

        #self.add_histo(f"lep_pt", [self.trgLep_pt], [self.axis_lep_pt])
        #self.add_histo(f"lep_pt_qtrw", [self.trgLep_pt], [self.axis_lep_pt], nominal_weight="nominal_weight_vptrw_mc_data")

        if self.dataset.name in self.datasets_to_apply:
            self.add_histo("mt_corr_xy_qt", ["mt_corr_xy", "v_gen_pt"], [self.axis_mt, self.axis_qt])

        # run-dependent stuff for data
        if self.dataset.is_data and False:
            self.add_histo("njets_runNo", ["njets", "run"], [self.axis_njets, self.axis_run_no])
            self.add_histo("RawMET_sumEt_runNo", ["RawMET_sumEt", "run"], [self.axis_sumEt, self.axis_run_no])

            self.add_histo("recoil_uncorr_para_runNo", ["recoil_uncorr_para", "run"], [self.axis_recoil_para, self.axis_run_no])
            self.add_histo("recoil_uncorr_perp_runNo", ["recoil_uncorr_perp", "run"], [self.axis_recoil_perp, self.axis_run_no])
            self.add_histo("recoil_corr_lep_para_runNo", ["recoil_corr_lep_para", "run"], [self.axis_recoil_para, self.axis_run_no])
            self.add_histo("recoil_corr_lep_perp_runNo", ["recoil_corr_lep_perp", "run"], [self.axis_recoil_perp, self.axis_run_no])
            self.add_histo("recoil_corr_xy_para_runNo", ["recoil_corr_xy_para", "run"], [self.axis_recoil_para, self.axis_run_no])
            self.add_histo("recoil_corr_xy_perp_runNo", ["recoil_corr_xy_perp", "run"], [self.axis_recoil_perp, self.axis_run_no])

            self.add_histo("METx_corr_lep_runNo", ["METx_corr_lep", "run"], [self.axis_MET_xy, self.axis_run_no])
            self.add_histo("METy_corr_lep_runNo", ["METy_corr_lep", "run"], [self.axis_MET_xy, self.axis_run_no])
            self.add_histo("npv_runNo", ["PV_npvs", "run"], [self.axis_npv, self.axis_run_no])


    def setup_gen_reco_vars_Z(self):

        self.df = self.df.Define("lep0_mom4", "ROOT::Math::PtEtaPhiMVector(lep_corr_pt[0], lep_corr_eta[0], lep_corr_phi[0], wrem::muon_mass)")
        self.df = self.df.Define("lep1_mom4", "ROOT::Math::PtEtaPhiMVector(lep_corr_pt[1], lep_corr_eta[1], lep_corr_phi[1], wrem::muon_mass)")
        self.df = self.df.Define("vmom4", "ROOT::Math::PxPyPzEVector(lep0_mom4)+ROOT::Math::PxPyPzEVector(lep1_mom4)")
        self.df = self.df.Define("v_pt", "vmom4.Pt()")
        self.df = self.df.Define("v_phi", "vmom4.Phi()")
        self.df = self.df.Define("v_y", "vmom4.Rapidity()")
        
        if self.dataset.name in self.datasets_to_apply:

            def gen_res_vars(suffix, gen_pt, gen_phi):
                self.df = self.df.Define(f"reco_over_gen_v_pt{suffix}", f"v_pt/{gen_pt}")
                self.df = self.df.Define(f"reco_minus_gen_v_pt{suffix}", f"v_pt-{gen_pt}")
                self.df = self.df.Define(f"reco_minus_gen_v_phi{suffix}", f"v_phi-{gen_phi}")
                if self.storeHists:
                    self.add_histo(f"v_gen_pt{suffix}", [gen_pt], [self.axis_qt])
                    self.add_histo(f"reco_over_gen_v_pt{suffix}", [f"reco_over_gen_v_pt{suffix}"], [self.axis_res_ratio])
                    self.add_histo(f"reco_minus_gen_v_pt{suffix}", [f"reco_minus_gen_v_pt{suffix}"], [self.axis_res_diff])
                    self.add_histo(f"reco_minus_gen_v_phi{suffix}", [f"reco_minus_gen_v_phi{suffix}"], [self.axis_res_diff])
                    self.add_histo(f"v_gen_reco_pt{suffix}", [gen_pt, "v_pt"], [self.axis_qt_gen, self.axis_qt])

            # pre-FSR
            self.df = self.df.Alias("v_gen_pt_prefsr", "ptVgen")
            self.df = self.df.Alias("v_gen_phi_prefsr", "phiVgen")
            gen_res_vars("_prefsr", "v_gen_pt_prefsr", "v_gen_phi_prefsr")

            # post-FSR
            if "postFSRleps" not in self.df.GetColumnNames():
                self.df = self.df.Define("postFSRleps", "GenPart_status == 1 && (GenPart_statusFlags&1 || GenPart_statusFlags&(1<<5)) && (GenPart_pdgId >= 11 && GenPart_pdgId <= 14)")
                self.df = self.df.Define("postFSRantileps", "GenPart_status == 1 && (GenPart_statusFlags&1 || GenPart_statusFlags&(1<<5)) && (GenPart_pdgId <= -11 && GenPart_pdgId >= -14)")
                self.df = self.df.Define("postFSRlepIdx", "ROOT::VecOps::ArgMax(GenPart_pt[postFSRleps])")
                self.df = self.df.Define("postFSRantilepIdx", "ROOT::VecOps::ArgMax(GenPart_pt[postFSRantileps])")
                self.df = self.df.Define("lepGen", "ROOT::Math::PtEtaPhiMVector(GenPart_pt[postFSRleps][postFSRlepIdx], GenPart_eta[postFSRleps][postFSRlepIdx], GenPart_phi[postFSRleps][postFSRlepIdx], GenPart_mass[postFSRleps][postFSRlepIdx])")
                self.df = self.df.Define("antilepGen", "ROOT::Math::PtEtaPhiMVector(GenPart_pt[postFSRantileps][postFSRantilepIdx], GenPart_eta[postFSRantileps][postFSRantilepIdx], GenPart_phi[postFSRantileps][postFSRantilepIdx], GenPart_mass[postFSRantileps][postFSRantilepIdx])")
                self.df = self.df.Define("VGen", "ROOT::Math::PxPyPzEVector(lepGen)+ROOT::Math::PxPyPzEVector(antilepGen)")
            self.df = self.df.Define("v_gen_pt_postfsr", "VGen.pt()")
            self.df = self.df.Define("v_gen_phi_postfsr", "VGen.Phi()")
            gen_res_vars("_postfsr", "v_gen_pt_postfsr", "v_gen_phi_postfsr")

            # proxy pre-FSR
            self.df = self.df.Define("trg_lep_mom4", f"ROOT::Math::PtEtaPhiMVector(lep_trg_pt, lep_trg_eta, lep_trg_phi, wrem::muon_mass)")
            self.df = self.df.Define("proxy_gen_prefsr", f"wrem::proxy_gen_v(genl, genlanti, trg_lep_mom4, lep_trg_charge)")
            self.df = self.df.Define("v_gen_pt_proxy_prefsr", "proxy_gen_prefsr.pt()")
            self.df = self.df.Define("v_gen_phi_proxy_prefsr", "proxy_gen_prefsr.Phi()")
            gen_res_vars("_proxy_prefsr", "v_gen_pt_proxy_prefsr", "v_gen_phi_proxy_prefsr")

            # proxy post-FSR
            self.df = self.df.Define("proxy_gen_postfsr", f"wrem::proxy_gen_v(lepGen, antilepGen, trg_lep_mom4, lep_trg_charge)")
            self.df = self.df.Define("v_gen_pt_proxy_postfsr", "proxy_gen_postfsr.pt()")
            self.df = self.df.Define("v_gen_phi_proxy_postfsr", "proxy_gen_postfsr.Phi()")
            gen_res_vars("_proxy_postfsr", "v_gen_pt_proxy_postfsr", "v_gen_phi_proxy_postfsr")

            # select the gen variable
            self.df = self.df.Alias("v_gen_pt", "v_gen_pt_proxy_postfsr")
            self.df = self.df.Alias("v_gen_phi", "v_gen_phi_proxy_postfsr")


            # reweighthings
            self.df = self.df.Define("vpt_weight_mc_data", self.vpt_reweight_helper_mc_data, ["v_pt"])
            self.df = self.df.Define("nominal_weight_vptrw_mc_data", "nominal_weight*vpt_weight_mc_data")

        else:
            self.df = self.df.Alias("nominal_weight_vptrw_mc_data", "nominal_weight")

    def setup_gen_reco_vars_W(self):
        if self.dataset.name in self.datasets_to_apply:
            def gen_res_vars(suffix, gen_pt, gen_phi):
                if self.storeHists:
                    self.add_histo(f"v_gen_pt{suffix}", [gen_pt], [self.axis_qt])

            # pre-FSR
            self.df = self.df.Alias("v_gen_pt_prefsr", "ptVgen")
            self.df = self.df.Alias("v_gen_phi_prefsr", "phiVgen")
            gen_res_vars("_prefsr", "v_gen_pt_prefsr", "v_gen_phi_prefsr")
        
            # post-FSR
            if "postFSRleps" not in self.df.GetColumnNames():
                self.df = self.df.Define("postFSRleps", "GenPart_status == 1 && (GenPart_statusFlags&1 || GenPart_statusFlags&(1<<5)) && (GenPart_pdgId >= 11 && GenPart_pdgId <= 14)")
                self.df = self.df.Define("postFSRantileps", "GenPart_status == 1 && (GenPart_statusFlags&1 || GenPart_statusFlags&(1<<5)) && (GenPart_pdgId <= -11 && GenPart_pdgId >= -14)")
                self.df = self.df.Define("postFSRlepIdx", "ROOT::VecOps::ArgMax(GenPart_pt[postFSRleps])")
                self.df = self.df.Define("postFSRantilepIdx", "ROOT::VecOps::ArgMax(GenPart_pt[postFSRantileps])")
                self.df = self.df.Define("lepGen", "ROOT::Math::PtEtaPhiMVector(GenPart_pt[postFSRleps][postFSRlepIdx], GenPart_eta[postFSRleps][postFSRlepIdx], GenPart_phi[postFSRleps][postFSRlepIdx], GenPart_mass[postFSRleps][postFSRlepIdx])")
                self.df = self.df.Define("antilepGen", "ROOT::Math::PtEtaPhiMVector(GenPart_pt[postFSRantileps][postFSRantilepIdx], GenPart_eta[postFSRantileps][postFSRantilepIdx], GenPart_phi[postFSRantileps][postFSRantilepIdx], GenPart_mass[postFSRantileps][postFSRantilepIdx])")
                self.df = self.df.Define("VGen", "ROOT::Math::PxPyPzEVector(lepGen)+ROOT::Math::PxPyPzEVector(antilepGen)")
            self.df = self.df.Define("v_gen_pt_postfsr", "VGen.pt()")
            self.df = self.df.Define("v_gen_phi_postfsr", "VGen.Phi()")
            gen_res_vars("_postfsr", "v_gen_pt_postfsr", "v_gen_phi_postfsr")

            # proxy pre-FSR
            self.df = self.df.Define("trg_lep_mom4", f"ROOT::Math::PtEtaPhiMVector(lep_corr_pt, lep_corr_eta, lep_corr_phi, wrem::muon_mass)")
            self.df = self.df.Define("proxy_gen_prefsr", f"wrem::proxy_gen_v(genl, genlanti, trg_lep_mom4, lep_corr_charge)")
            self.df = self.df.Define("v_gen_pt_proxy_prefsr", "proxy_gen_prefsr.pt()")
            self.df = self.df.Define("v_gen_phi_proxy_prefsr", "proxy_gen_prefsr.Phi()")
            gen_res_vars("_proxy_prefsr", "v_gen_pt_proxy_prefsr", "v_gen_phi_proxy_prefsr")

            # proxy post-FSR
            self.df = self.df.Define("proxy_gen_postfsr", f"wrem::proxy_gen_v(lepGen, antilepGen, trg_lep_mom4, lep_corr_charge)")
            self.df = self.df.Define("v_gen_pt_proxy_postfsr", "proxy_gen_postfsr.pt()")
            self.df = self.df.Define("v_gen_phi_proxy_postfsr", "proxy_gen_postfsr.Phi()")
            gen_res_vars("_proxy_postfsr", "v_gen_pt_proxy_postfsr", "v_gen_phi_proxy_postfsr")

            # select the gen variable
            if 'taunu' in self.dataset.name or 'tautau' in self.dataset.name:
                self.df = self.df.Alias("v_gen_pt", "v_gen_pt_prefsr")
                self.df = self.df.Alias("v_gen_phi", "v_gen_phi_prefsr")
            else:
                self.df = self.df.Alias("v_gen_pt", "v_gen_pt_proxy_postfsr")
                self.df = self.df.Alias("v_gen_phi", "v_gen_phi_proxy_postfsr")
                #self.df = self.df.Alias("v_gen_pt", "v_gen_pt_prefsr")
                #self.df = self.df.Alias("v_gen_phi", "v_gen_phi_prefsr")

            # reweighthings
            self.df = self.df.Define("vpt_weight_mc_data", self.vpt_reweight_helper_mc_data, ["v_gen_pt"])
            self.df = self.df.Define("nominal_weight_vptrw_mc_data", "nominal_weight*vpt_weight_mc_data")

        else:
            self.df = self.df.Alias("nominal_weight_vptrw_mc_data", "nominal_weight")

    def setup_recoil_gen(self):
        if not self.dataset.name in self.datasets_to_apply:
            return

        # decompose MET and dilepton (for Z) or lepton (for W) along the generator boson direction
        self.df = self.df.Define("recoil_corr_xy_gen", "wrem::compute_recoil_from_met(met_corr_xy_pt, met_corr_xy_phi, lep_corr_pt, lep_corr_phi, v_gen_pt, v_gen_phi)") # both gen pt and phi

        self.df = self.df.Define("recoil_corr_xy_para_gen", "recoil_corr_xy_gen[0]")
        self.df = self.df.Define("recoil_corr_xy_perp_gen", "recoil_corr_xy_gen[1]")
        self.df = self.df.Define("recoil_corr_xy_para_qt_gen", "recoil_corr_xy_para_gen - v_gen_pt")

        if not self.storeHists: 
            return

        self.add_histo("recoil_corr_xy_para_gen", ["recoil_corr_xy_para_gen"], [self.axis_recoil_para])
        self.add_histo("recoil_corr_xy_perp_gen", ["recoil_corr_xy_perp_gen"], [self.axis_recoil_perp])
        self.add_histo("recoil_corr_xy_para_qt_gen", ["recoil_corr_xy_para_qt_gen"], [self.axis_recoil_para_qT])

        self.add_histo("recoil_corr_xy_para_gen_v_gen_pt", ["v_gen_pt", "recoil_corr_xy_para_gen"], [self.axis_qt, self.axis_recoil_para])
        self.add_histo("recoil_corr_xy_perp_gen_v_gen_pt", ["v_gen_pt", "recoil_corr_xy_perp_gen"], [self.axis_qt, self.axis_recoil_perp])

        self.add_histo("recoil_corr_xy_para_gen_npv", ["PV_npvs", "recoil_corr_xy_para_gen"], [self.axis_npv, self.axis_recoil_para])
        self.add_histo("recoil_corr_xy_perp_gen_npv", ["PV_npvs", "recoil_corr_xy_perp_gen"], [self.axis_npv, self.axis_recoil_perp])

    def apply_recoil_Z(self):

        doGEN = False
        if self.dataset.name in self.datasets_to_apply:

            if doGEN:
                self.df = self.df.Define("recoil_corr", self.recoilHelper, ["v_gen_pt", "recoil_corr_xy_para_gen", "recoil_corr_xy_perp_gen"])
                self.df = self.df.Define("recoil_corr_rec_para", "recoil_corr.ut_para_corr(0)")
                self.df = self.df.Define("recoil_corr_rec_perp", "recoil_corr.ut_perp_corr(0)")
                self.df = self.df.Define("recoil_corr_rec_para_qt", "recoil_corr_rec_para - v_gen_pt")

                self.df = self.df.Define("met_corr_rec", "wrem::compute_met_from_recoil(recoil_corr_rec_para, recoil_corr_rec_perp, lep_corr_pt, lep_corr_phi, v_gen_pt, v_gen_phi)")
                self.df = self.df.Define("met_corr_rec_pt", "met_corr_rec[0]")
                self.df = self.df.Define("met_corr_rec_phi", "met_corr_rec[1]")
                self.df = self.df.Define("met_corr_rec_x", "met_corr_rec_pt*cos(met_corr_rec_phi)")
                self.df = self.df.Define("met_corr_rec_y", "met_corr_rec_pt*sin(met_corr_rec_phi)")
            else:
                self.df = self.df.Define("recoil_corr", self.recoilHelper, ["v_pt", "recoil_corr_xy_para", "recoil_corr_xy_perp"])
                self.df = self.df.Define("recoil_corr_rec_para", "recoil_corr.ut_para_corr(0)")
                self.df = self.df.Define("recoil_corr_rec_perp", "recoil_corr.ut_perp_corr(0)")
                self.df = self.df.Define("recoil_corr_rec_para_qt", "recoil_corr_rec_para - v_pt")

                self.df = self.df.Define("met_corr_rec", "wrem::compute_met_from_recoil(recoil_corr_rec_para, recoil_corr_rec_perp, lep_corr_pt, lep_corr_phi, v_pt, v_phi)")
                self.df = self.df.Define("met_corr_rec_pt", "met_corr_rec[0]")
                self.df = self.df.Define("met_corr_rec_phi", "met_corr_rec[1]")
                self.df = self.df.Define("met_corr_rec_x", "met_corr_rec_pt*cos(met_corr_rec_phi)")
                self.df = self.df.Define("met_corr_rec_y", "met_corr_rec_pt*sin(met_corr_rec_phi)")

        else:
            self.df = self.df.Alias("recoil_corr_rec_para", "recoil_corr_xy_para")
            self.df = self.df.Alias("recoil_corr_rec_perp", "recoil_corr_xy_perp")
            self.df = self.df.Alias("recoil_corr_rec_para_qt", "recoil_corr_xy_para_qt")
            self.df = self.df.Alias("met_corr_rec_pt", "met_corr_xy_pt")
            self.df = self.df.Alias("met_corr_rec_phi", "met_corr_xy_phi")
            self.df = self.df.Alias("met_corr_rec_x", "met_corr_xy_x")
            self.df = self.df.Alias("met_corr_rec_y", "met_corr_xy_y")

        self.df = self.df.Alias("MET_corr_rec_pt", "met_corr_rec_pt")
        self.df = self.df.Alias("MET_corr_rec_phi", "met_corr_rec_phi")

        self.recoil_vars_plots_Z("corr_rec")
        self.recoil_vars_plots_Z("corr_rec", suffix="qtrw", nominal_weight="nominal_weight_vptrw_mc_data")

        if not self.storeHists:
            return


    def apply_recoil_W(self): 

        if self.dataset.name in self.datasets_to_apply:
            
            # old stuff
            self.df = self.df.Define("recoil_corr_xy_gen_old", f"wrem::recoilComponentsGen(met_corr_xy_pt, met_corr_xy_phi, lep_corr_pt, lep_corr_phi, v_gen_phi)")
            self.df = self.df.Define("qT_gen", "ptVgen") # pre-fsr defines should be loaded
            self.df = self.df.Define("recoil_corr_xy_para_gen_old", "recoil_corr_xy_gen_old[1] + v_gen_pt")
            self.df = self.df.Define("recoil_corr_xy_para_qT_gen_old", "recoil_corr_xy_gen_old[1]")
            self.df = self.df.Define("recoil_corr_xy_perp_gen_old", "recoil_corr_xy_gen_old[2]")
            
            self.df = self.df.Define("recoil_corr_old", self.recoilHelper, ["v_gen_pt", "recoil_corr_xy_para_gen_old", "recoil_corr_xy_perp_gen_old"])
            self.df = self.df.Define("recoil_corr_rec_para_gen_old", "recoil_corr_old.ut_para_corr(0)")
            self.df = self.df.Define("recoil_corr_rec_perp_gen_old", "recoil_corr_old.ut_perp_corr(0)")
            self.df = self.df.Define("recoil_corr_rec_para_qt_gen_old", "recoil_corr_rec_para_gen_old - v_gen_pt")

            self.df = self.df.Define("MET_corr_rec_old", f"wrem::METCorrectionGen(recoil_corr_rec_para_qt_gen_old, recoil_corr_rec_perp_gen_old, lep_corr_pt, lep_corr_phi, v_gen_phi) ") 
            self.df = self.df.Define("MET_corr_rec_pt_old", "MET_corr_rec_old[0]")
            self.df = self.df.Define("MET_corr_rec_phi_old", "MET_corr_rec_old[1]")
            
            
            
            self.df = self.df.Define("recoil_corr", self.recoilHelper, ["v_gen_pt", "recoil_corr_xy_para_gen", "recoil_corr_xy_perp_gen"])
            self.df = self.df.Define("recoil_corr_rec_para_gen", "recoil_corr.ut_para_corr(0)")
            self.df = self.df.Define("recoil_corr_rec_perp_gen", "recoil_corr.ut_perp_corr(0)")
            self.df = self.df.Define("recoil_corr_rec_para_qt_gen", "recoil_corr_rec_para_gen - v_gen_pt")

            self.df = self.df.Define("met_corr_rec", "wrem::compute_met_from_recoil(recoil_corr_rec_para_gen, recoil_corr_rec_perp_gen, lep_corr_pt, lep_corr_phi, v_gen_pt, v_gen_phi)")
            self.df = self.df.Define("met_corr_rec_pt", "met_corr_rec[0]")
            self.df = self.df.Define("met_corr_rec_phi", "met_corr_rec[1]")
            
            #self.df = self.df.Define("met_corr_rec_phi", "cout << recoil_corr_rec_para_gen<< ' ' << recoil_corr_rec_perp_gen << ' ' << met_corr_rec_pt << ' ' << met_corr_rec_phi_ << ' ' <<  recoil_corr_xy_para_gen_old << ' ' << recoil_corr_xy_perp_gen_old << ' ' << MET_corr_rec_pt_old << ' ' << MET_corr_rec_phi_old << endl; return met_corr_rec_phi_;")
            
            self.df = self.df.Define("met_corr_rec_x", "met_corr_rec_pt*cos(met_corr_rec_phi)")
            self.df = self.df.Define("met_corr_rec_y", "met_corr_rec_pt*sin(met_corr_rec_phi)")

        else:
            self.df = self.df.Alias("met_corr_rec_pt", "met_corr_xy_pt")
            self.df = self.df.Alias("met_corr_rec_phi", "met_corr_xy_phi")
            self.df = self.df.Alias("met_corr_rec_x", "met_corr_xy_x")
            self.df = self.df.Alias("met_corr_rec_y", "met_corr_xy_y")

        self.df = self.df.Alias("MET_corr_rec_pt", "met_corr_rec_pt")
        self.df = self.df.Alias("MET_corr_rec_phi", "met_corr_rec_phi")

        self.recoil_vars_plots_W("uncorr")
        self.recoil_vars_plots_W("corr_lep")
        self.recoil_vars_plots_W("corr_xy")
        self.recoil_vars_plots_W("corr_xy", suffix="qtrw", nominal_weight="nominal_weight_vptrw_mc_data")
        self.recoil_vars_plots_W("corr_rec")
        self.recoil_vars_plots_W("corr_rec", suffix="qtrw", nominal_weight="nominal_weight_vptrw_mc_data")



    def add_recoil_unc_Z(self, df, results, dataset, cols, axes, hName, storage_type=hist.storage.Double()):
        if not dataset.name in self.datasets_to_apply:
            return df
        results.append(df.HistoBoost(f"{hName}_recoil_stat", axes if isinstance(axes, list) else [axes], (cols if isinstance(cols, list) else [cols]) + [self.recoil_unc_stat_weights_with_nom], tensor_axes=[self.recoil_var_ax_stat]))
        results.append(df.HistoBoost(f"{hName}_recoil_syst", axes if isinstance(axes, list) else [axes], (cols if isinstance(cols, list) else [cols]) + [self.recoil_unc_syst_weights_with_nom], tensor_axes=[self.recoil_var_ax_syst]))
        return df
        
        
    def add_recoil_unc_W(self, df, results, dataset, cols, axes, hName, storage_type=hist.storage.Double()):
        return self.add_recoil_unc_Z(df, results, dataset, cols, axes, hName) # currently use the Z implementation

    def setup_recoil_Z_unc(self):
        if not self.dataset.name in self.datasets_to_apply or not self.storeHists:
            return

        hNames, cols, axes = [], [], [] 
        if self.storeHists:
            hNames = ["recoil_corr_rec_para_qt", "recoil_corr_rec_para", "recoil_corr_rec_perp", "recoil_corr_rec_magn", "met_corr_rec_pt", "mt_corr_rec"]
            cols = hNames
            axes = [self.axis_recoil_para_qT, self.axis_recoil_para, self.axis_recoil_perp, self.axis_recoil_magn, self.axis_MET_pt, self.axis_mt]

        # statistical uncertainties
        self.df = self.df.Define("recoil_unc_stat_weights", "recoil_corr.unc_weights")
        self.recoil_unc_stat_weights_with_nom = "recoil_unc_stat_weights_with_nom"
        self.df = self.df.Define(self.recoil_unc_stat_weights_with_nom, "auto res = recoil_unc_stat_weights; res = nominal_weight*res; return res;") # 
        self.recoil_var_ax_stat = hist.axis.Integer(0, self.nstat, name="recoil_unc", underflow=False, overflow=False)

        # systematic uncertainties
        recoil_systs = []
        if self.recoil_syst_bkg_para != None:
            self.df = self.df.Define("recoil_syst_bkg_para_w", self.recoil_syst_bkg_para, ["v_pt", "recoil_corr_xy_para"])
            recoil_systs.append("recoil_syst_bkg_para_w")
        if self.recoil_syst_bkg_perp != None:
            self.df = self.df.Define("recoil_syst_bkg_perp_w", self.recoil_syst_bkg_perp, ["v_pt", "recoil_corr_xy_perp"])
            recoil_systs.append("recoil_syst_bkg_perp_w")

        # reco-gen differences
        '''
        self.df = self.df.Define("recoil_pdf_data_para_w_pert", self.recoil_pdf_data_para, ["v_gen_pt", "recoil_corr_xy_para"])
        self.df = self.df.Define("recoil_pdf_data_para_w_nom", self.recoil_pdf_data_para, ["v_pt", "recoil_corr_xy_para"])
        self.df = self.df.Define("recoil_pdf_data_para_w", "recoil_pdf_data_para_w_pert/recoil_pdf_data_para_w_nom")

        self.df = self.df.Define("recoil_pdf_data_perp_w_pert", self.recoil_pdf_data_perp, ["v_gen_pt", "recoil_corr_xy_perp"])
        self.df = self.df.Define("recoil_pdf_data_perp_w_nom", self.recoil_pdf_data_perp, ["v_pt", "recoil_corr_xy_perp"])
        self.df = self.df.Define("recoil_pdf_data_perp_w", "recoil_pdf_data_perp_w_pert/recoil_pdf_data_perp_w_nom")
        
        self.df = self.df.Define("recoil_pdf_mc_para_w_pert", self.recoil_pdf_mc_para, ["v_gen_pt", "recoil_corr_xy_para"])
        self.df = self.df.Define("recoil_pdf_mc_para_w_nom", self.recoil_pdf_mc_para, ["v_pt", "recoil_corr_xy_para"])
        self.df = self.df.Define("recoil_pdf_mc_para_w", "recoil_pdf_mc_para_w_pert/recoil_pdf_mc_para_w_nom")

        self.df = self.df.Define("recoil_pdf_mc_perp_w_pert", self.recoil_pdf_mc_perp, ["v_gen_pt", "recoil_corr_xy_perp"])
        self.df = self.df.Define("recoil_pdf_mc_perp_w_nom", self.recoil_pdf_mc_perp, ["v_pt", "recoil_corr_xy_perp"])
        self.df = self.df.Define("recoil_pdf_mc_perp_w", "recoil_pdf_mc_perp_w_pert/recoil_pdf_mc_perp_w_nom")

        self.df = self.df.Define("recoil_pdf_tot", "recoil_pdf_mc_perp_w*recoil_pdf_mc_para_w*recoil_pdf_data_perp_w*recoil_pdf_data_para_w")
        recoil_systs.append("recoil_pdf_tot")
        '''

        self.nsyst = len(recoil_systs)

        self.recoil_unc_syst_weights_with_nom = "recoil_unc_syst_weights_with_nom"
        self.df = self.df.Define(self.recoil_unc_syst_weights_with_nom, f"wrem::concatWeights<{len(recoil_systs)}>(nominal_weight, {', '.join(recoil_systs)})")
        self.recoil_var_ax_syst = hist.axis.Integer(0, self.nsyst, name="recoil_unc", underflow=False, overflow=False)
        for hName, col, ax in zip(hNames, cols, axes):
            self.results.append(self.df.HistoBoost(f"{hName}_recoil_stat", ax if isinstance(col, list) else [ax], (col if isinstance(col, list) else [col]) + [self.recoil_unc_stat_weights_with_nom], tensor_axes=[self.recoil_var_ax_stat]))
            self.results.append(self.df.HistoBoost(f"{hName}_recoil_syst", ax if isinstance(col, list) else [ax], (col if isinstance(col, list) else [col]) + [self.recoil_unc_syst_weights_with_nom], tensor_axes=[self.recoil_var_ax_syst]))


    def setup_recoil_W_unc(self):
        if not self.dataset.name in self.datasets_to_apply or not self.storeHists:
            return

        hNames, cols, axes = [], [], [] 
        if self.storeHists:
            hNames = ["met_corr_rec_pt", "mt_corr_rec"]
            cols = hNames
            axes = [self.axis_MET_pt, self.axis_mt]

        # statistical uncertainties
        self.df = self.df.Define("recoil_unc_stat_weights", "recoil_corr.unc_weights")
        self.recoil_unc_stat_weights_with_nom = "recoil_unc_stat_weights_with_nom"
        self.df = self.df.Define(self.recoil_unc_stat_weights_with_nom, "auto res = recoil_unc_stat_weights; res = nominal_weight*res; return res;") # 
        self.recoil_var_ax_stat = hist.axis.Integer(0, self.nstat, name="recoil_unc", underflow=False, overflow=False)

        # systematic uncertainties
        recoil_systs = []
        if self.recoil_syst_bkg_para != None:
            self.df = self.df.Define("recoil_syst_bkg_para_w", self.recoil_syst_bkg_para, ["v_gen_pt", "recoil_corr_xy_para_gen"])
            recoil_systs.append("recoil_syst_bkg_para_w")
        if self.recoil_syst_bkg_perp != None:
            self.df = self.df.Define("recoil_syst_bkg_perp_w", self.recoil_syst_bkg_perp, ["v_gen_pt", "recoil_corr_xy_perp_gen"])
            recoil_systs.append("recoil_syst_bkg_perp_w")

        self.nsyst = len(recoil_systs)

        self.recoil_unc_syst_weights_with_nom = "recoil_unc_syst_weights_with_nom"
        self.df = self.df.Define(self.recoil_unc_syst_weights_with_nom, f"wrem::concatWeights<{len(recoil_systs)}>(nominal_weight, {', '.join(recoil_systs)})")
        self.recoil_var_ax_syst = hist.axis.Integer(0, self.nsyst, name="recoil_unc", underflow=False, overflow=False)
        for hName, col, ax in zip(hNames, cols, axes):
            self.results.append(self.df.HistoBoost(f"{hName}_recoil_stat", ax if isinstance(col, list) else [ax], (col if isinstance(col, list) else [col]) + [self.recoil_unc_stat_weights_with_nom], tensor_axes=[self.recoil_var_ax_stat]))
            self.results.append(self.df.HistoBoost(f"{hName}_recoil_syst", ax if isinstance(col, list) else [ax], (col if isinstance(col, list) else [col]) + [self.recoil_unc_syst_weights_with_nom], tensor_axes=[self.recoil_var_ax_syst]))
