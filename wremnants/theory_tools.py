import ROOT
import hist
import numpy as np
import copy
from math import pi
from utilities import boostHistHelpers as hh,common,logging
from wremnants import theory_corrections
from scipy import ndimage
import narf.clingutils
from math import sqrt

logger = logging.child_logger(__name__)
narf.clingutils.Declare('#include "theoryTools.h"')

# this puts the bin centers at 0.5, 1.0, 2.0
axis_muRfact = hist.axis.Variable(
    [0.25, 0.75, 1.25, 2.75], name="muRfact", underflow=False, overflow=False
)
axis_muFfact = hist.axis.Variable(
    [0.25, 0.75, 1.25, 2.75], name="muFfact", underflow=False, overflow=False
)

axis_absYVgen = hist.axis.Variable(
    # [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 10],
    [0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 4., 5.], # this is the same binning as hists from theory corrections
    name = "absYVgenNP", underflow=False
)

axis_chargeWgen = hist.axis.Regular(2, -2, 2, name="chargeVgenNP", underflow=False, overflow=False)
axis_chargeZgen = hist.axis.Integer(0, 1, name="chargeVgenNP", underflow=False, overflow=False)

scale_tensor_axes = (axis_muRfact, axis_muFfact)

pdfMap = {
    "nnpdf31" : {
        "name" : "pdfNNPDF31",
        "branch" : "LHEPdfWeight",
        "combine" : "symHessian",
        "entries" : 101,
        "alphas" : ["LHEPdfWeight[0]", "LHEPdfWeight[101]", "LHEPdfWeight[102]"],
        "alphasRange" : "002",
        "inflationFactor": 2.5,
    },
    "ct18" : {
        "name" : "pdfCT18",
        "branch" : "LHEPdfWeightAltSet11",
        "combine" : "asymHessian",
        "entries" : 59,
        "alphas" : ["LHEPdfWeightAltSet11[0]", "LHEPdfWeightAltSet11[59]", "LHEPdfWeightAltSet11[62]"],
        "alphasRange" : "002",
        "scale" : 1/1.645, # Convert from 90% CL to 68%
        "inflationFactor": 1.0,
    },
    "nnpdf30" : {
        "name" : "pdfNNPDF30",
        "branch" : "LHEPdfWeightAltSet7",
        "combine" : "symHessian",
        "entries" : 101,
        "alphas" : ["LHEPdfWeightAltSet13[0]", "LHEPdfWeightAltSet15[0]", "LHEPdfWeightAltSet16[0]"],
        "alphasRange" : "001",
        "inflationFactor": 1.0, # not determined
    },
    "nnpdf40" : {
        "name" : "pdfNNPDF40",
        "branch" : "LHEPdfWeightAltSet3",
        "combine" : "symHessian",
        "entries" : 51,
        "alphas" : ["LHEPdfWeightAltSet3[0]", "LHEPdfWeightAltSet3[51]", "LHEPdfWeightAltSet3[52]"],
        "alphasRange" : "001",
        "inflationFactor": 4.0,
    },
    "pdf4lhc21" : {
        "name" : "pdfPDF4LHC21",
        "branch" : "LHEPdfWeightAltSet10",
        "combine" : "symHessian",
        "entries" : 41,
        "alphas" : ["LHEPdfWeightAltSet10[0]", "LHEPdfWeightAltSet10[41]", "LHEPdfWeightAltSet10[42]"],
        "alphasRange" : "001",
        "inflationFactor": 1.0,
    },
    "msht20" : {
        "name" : "pdfMSHT20",
        "branch" : "LHEPdfWeightAltSet12",
        "combine" : "asymHessian",
        "entries" : 65,
        "alphas" : ["LHEPdfWeightAltSet12[0]", "LHEPdfWeightAltSet12[67]", "LHEPdfWeightAltSet12[70]"],
        "alphasRange" : "002",
        "inflationFactor": 1.5,
    },
    "msht20mcrange" : {
        "name" : "pdfMSHT20mcrange",
        "branch" : "LHEPdfWeightAltSet12",
        "combine" : "asymHessian",
        "entries" : 9,
        "first_entry" : 72,
    },
    "msht20mbrange" : {
        "name" : "pdfMSHT20mbrange",
        "branch" : "LHEPdfWeightAltSet12",
        "combine" : "asymHessian",
        "entries" : 7,
        "first_entry" : 81,
    },
    "msht20mcrange_renorm" : {
        "name" : "pdfMSHT20mcrange",
        "branch" : "LHEPdfWeightAltSet12",
        "combine" : "asymHessian",
        "entries" : 9,
        "first_entry" : 72,
        "renorm" : True,
    },
    "msht20mbrange_renorm" : {
        "name" : "pdfMSHT20mbrange",
        "branch" : "LHEPdfWeightAltSet12",
        "combine" : "asymHessian",
        "entries" : 7,
        "first_entry" : 81,
        "renorm" : True,
    },
    "msht20an3lo" : {
        "name" : "pdfMSHT20an3lo",
        "branch" : "LHEPdfWeightAltSet24",
        "combine" : "asymHessian",
        "entries" : 105,
        "alphas" : ["LHEPdfWeightAltSet24[0]", "LHEPdfWeightAltSet24[108]", "LHEPdfWeightAltSet24[111]"],
        "alphasRange" : "002",
        "inflationFactor": 1.5,
    },
    "ct18z" : {
        "name" : "pdfCT18Z",
        "branch" : "LHEPdfWeightAltSet11",
        "combine" : "asymHessian",
        "entries" : 59,
        "first_entry" : 63,
        "alphas" : ["LHEPdfWeightAltSet11[63]", "LHEPdfWeightAltSet11[122]", "LHEPdfWeightAltSet11[125]"],
        "alphasRange" : "002",
        "scale" : 1/1.645, # Convert from 90% CL to 68%
        "inflationFactor": 1.0,
    },
    "atlasWZj20" : {
        "name" : "pdfATLASWZJ20",
        "branch" : "LHEPdfWeightAltSet19",
        "combine" : "asymHessian",
        "entries" : 60,
        "alphas" : ["LHEPdfWeight[0]", "LHEPdfWeight[41]", "LHEPdfWeight[42]"],
        "alphasRange" : "002",
        "inflationFactor": 1.0,  # not determined
    },
    "herapdf20" : {
        "name" : "pdfHERAPDF20",
        "branch" : "LHEPdfWeightAltSet20",
        "combine" : "asymHessian",
        "entries" : 29,
        "alphas" : ["LHEPdfWeightAltSet20[0]", "LHEPdfWeightAltSet22[0]", "LHEPdfWeightAltSet23[0]"], # alphas 116-120
        "alphasRange" : "002",
        "inflationFactor": 4.0,
    },
    "herapdf20ext" : {
        "name" : "pdfHERAPDF20ext",
        "branch" : "LHEPdfWeightAltSet21",
        "combine" : "symHessian",
        "entries" : 14,
        "alphas" : ["LHEPdfWeightAltSet20[0]", "LHEPdfWeightAltSet22[0]", "LHEPdfWeightAltSet23[0]"], # dummy AS
        "alphasRange" : "002",
        "inflationFactor": 4.0,
    },
}


only_central_pdf_datasets = [
    "Wplusmunu_bugfix",
    "Wminusmunu_bugfix",
    "Zmumu_bugfix",
    "Zmumu_bugfix_slc7",
]

extended_pdf_datasets = [x for x in common.vprocs_all if not any(y in x for y in ["NNLOPS", "MiNLO"])]

def expand_pdf_entries(pdf, alphas=False, renorm=False):
    info = pdfMap[pdf]
    if alphas:
        vals = info["alphas"]
    else:
        first_entry = info.get("first_entry", 0)
        last_entry = first_entry+info["entries"]
        vals = [info["branch"]+f"[{i}]" for i in range(first_entry, last_entry)]

    if renorm:
        vals = [f"std::clamp<float>({x}/{vals[0]}*central_pdf_weight, -theory_weight_truncate, theory_weight_truncate)" for x in vals]
    else:
        vals = [f"std::clamp<float>({x}, -theory_weight_truncate, theory_weight_truncate)" for x in vals]

    return vals

def define_scale_tensor(df):
    if "scaleWeights_tensor" in df.GetColumnNames():
        logger.debug("scaleWeight_tensor already defined, do nothing here.")
        return df
    # convert vector of scale weights to 3x3 tensor and clip weights to |weight|<10.
    df = df.Define("scaleWeights_tensor", f"wrem::makeScaleTensor(LHEScaleWeight, theory_weight_truncate);")
    df = df.Define("scaleWeights_tensor_wnom", "auto res = scaleWeights_tensor; res = nominal_weight*res; return res;")

    return df

theory_corr_weight_map = {
        "scetlib_dyturboMSHT20_pdfas" : expand_pdf_entries("msht20", alphas=True),
        "scetlib_dyturboMSHT20Vars" : expand_pdf_entries("msht20"),
        "scetlib_dyturboCT18ZVars" : expand_pdf_entries("ct18z"),
        "scetlib_dyturboCT18Z_pdfas" : expand_pdf_entries("ct18z", alphas=True),
        "scetlib_dyturboMSHT20an3lo_pdfas" : expand_pdf_entries("msht20an3lo", alphas=True),
        "scetlib_dyturboMSHT20an3loVars" : expand_pdf_entries("msht20an3lo"),
        # Tested this, better not to treat this way unless using MSHT20nnlo as central set
        #"scetlib_dyturboMSHT20mbrange" : expand_pdf_entries("msht20mbrange", renorm=True),
        #"scetlib_dyturboMSHT20mcrange" : expand_pdf_entries("msht20mcrange", renorm=True),
}

def define_dressed_vars(df, mode, flavor="mu"):
    if "dressedGenV_mom4" in df.GetColumnNames():
        logger.debug("LHE variables are already defined, do nothing here.")
        return df

    logger.info("Defining dressed variables")

    # use postfsr neutrinos
    df = define_postfsr_vars(df, mode)

    lep_pdgId = 13 if flavor == "mu" else 11

    if mode in ["wlike", "dilepton"]:
        df = df.Define("dressedLep", f"GenDressedLepton_pdgId=={lep_pdgId}")
        df = df.Define("dressedAntiLep", f"GenDressedLepton_pdgId==-{lep_pdgId}")
        
        df = df.Define("hasDressedLep", "ROOT::VecOps::Any(dressedLep)")
        df = df.Define("hasDressedAntiLep", "ROOT::VecOps::Any(dressedAntiLep)")

        df = df.Define("dressedLep_idx",     "ROOT::VecOps::ArgMax(GenDressedLepton_pt[dressedLep])")
        df = df.Define("dressedAntiLep_idx", "ROOT::VecOps::ArgMax(GenDressedLepton_pt[dressedAntiLep])")

        df = df.Define("dressedLep_pt", "hasDressedLep ? static_cast<double>(GenDressedLepton_pt[dressedLep][dressedLep_idx]) : 0")
        df = df.Define("dressedLep_eta", "hasDressedLep ? GenDressedLepton_eta[dressedLep][dressedLep_idx] : 0")
        df = df.Define("dressedLep_phi", "hasDressedLep ? GenDressedLepton_phi[dressedLep][dressedLep_idx] : 0")
        df = df.Define("dressedLep_mass", "hasDressedLep ? GenDressedLepton_mass[dressedLep][dressedLep_idx] : 0")

        df = df.Define("dressedAntiLep_pt", "hasDressedAntiLep ? static_cast<double>(GenDressedLepton_pt[dressedAntiLep][dressedAntiLep_idx]) : 0")
        df = df.Define("dressedAntiLep_eta", "hasDressedAntiLep ? GenDressedLepton_eta[dressedAntiLep][dressedAntiLep_idx] : 0")
        df = df.Define("dressedAntiLep_phi", "hasDressedAntiLep ? GenDressedLepton_phi[dressedAntiLep][dressedAntiLep_idx] : 0")
        df = df.Define("dressedAntiLep_mass", "hasDressedAntiLep ? GenDressedLepton_mass[dressedAntiLep][dressedAntiLep_idx] : 0")

        df = df.Define("dressedLep_mom4", "ROOT::Math::PtEtaPhiMVector(dressedLep_pt, dressedLep_eta, dressedLep_phi, dressedLep_mass)")
        df = df.Define("dressedAntiLep_mom4", "ROOT::Math::PtEtaPhiMVector(dressedAntiLep_pt, dressedAntiLep_eta, dressedAntiLep_phi, dressedAntiLep_mass)")

        df = df.Define('dressedGenV_mom4', 'dressedLep_mom4 + dressedAntiLep_mom4 + postfsrNeutrinos_mom4')
    else:
        df = df.Define("dressedLep", f"abs(GenDressedLepton_pdgId)=={lep_pdgId}")
        df = df.Define("hasDressedLep", "ROOT::VecOps::Any(dressedLep)")
        df = df.Define("dressedLep_idx", "ROOT::VecOps::ArgMax(GenDressedLepton_pt[dressedLep])")

        df = df.Define("dressedLep_pt", "hasDressedLep ? static_cast<double>(GenDressedLepton_pt[dressedLep][dressedLep_idx]) : 0")
        df = df.Define("dressedLep_eta", "hasDressedLep ? GenDressedLepton_eta[dressedLep][dressedLep_idx] : 0")
        df = df.Define("dressedLep_phi", "hasDressedLep ? GenDressedLepton_phi[dressedLep][dressedLep_idx] : 0")
        df = df.Define("dressedLep_mass", "hasDressedLep ? GenDressedLepton_mass[dressedLep][dressedLep_idx] : 0")

        df = df.Define("dressedLep_mom4", "ROOT::Math::PtEtaPhiMVector(dressedLep_pt, dressedLep_eta, dressedLep_phi, dressedLep_mass)")

        df = df.Define('dressedGenV_mom4', 'dressedLep_mom4 + postfsrNeutrinos_mom4')

    df = df.Define('dressed_MV', 'dressedGenV_mom4.mass()')
    df = df.Define('dressed_absYV', 'std::fabs(dressedGenV_mom4.Rapidity())')
    df = df.Define('dressed_PTV', 'dressedGenV_mom4.pt()')

    return df

def define_prefsr_vars(df, mode=None):
    if "prefsrLeps" in df.GetColumnNames():
        logger.debug("PreFSR leptons are already defined, do nothing here.")
        return df

    logger.info("Defining preFSR variables")

    df = df.Define("prefsrLeps", "wrem::prefsrLeptons(GenPart_status, GenPart_statusFlags, GenPart_pdgId, GenPart_genPartIdxMother)")
    df = df.Define("genl", "ROOT::Math::PtEtaPhiMVector(GenPart_pt[prefsrLeps[0]], GenPart_eta[prefsrLeps[0]], GenPart_phi[prefsrLeps[0]], GenPart_mass[prefsrLeps[0]])")
    df = df.Define("genlanti", "ROOT::Math::PtEtaPhiMVector(GenPart_pt[prefsrLeps[1]], GenPart_eta[prefsrLeps[1]], GenPart_phi[prefsrLeps[1]], GenPart_mass[prefsrLeps[1]])")
    df = df.Define("genV", "ROOT::Math::PxPyPzEVector(genl)+ROOT::Math::PxPyPzEVector(genlanti)")
    df = df.Define("ptVgen", "genV.pt()")
    df = df.Define("massVgen", "genV.mass()")
    df = df.Define("ptqVgen", "genV.pt()/genV.mass()")
    df = df.Define("yVgen", "genV.Rapidity()")
    df = df.Define("phiVgen", "genV.Phi()")
    df = df.Define("absYVgen", "std::fabs(yVgen)")
    df = df.Define("chargeVgen", "GenPart_pdgId[prefsrLeps[0]] + GenPart_pdgId[prefsrLeps[1]]")
    df = df.Define("csSineCosThetaPhigen", "wrem::csSineCosThetaPhi(genlanti, genl)")

    # define w and w-like variables 
    df = df.Define("qgen", "isEvenEvent ? -1 : 1")
    df = df.Define("ptgen", "isEvenEvent ? genl.pt() : genlanti.pt()")
    df = df.Define("etagen", "isEvenEvent ? genl.eta() : genlanti.eta()")
    df = df.Define("absetagen", "std::fabs(etagen)")
    df = df.Define("ptOthergen", "isEvenEvent ? genlanti.pt() : genl.pt()")
    df = df.Define("etaOthergen", "isEvenEvent ? genlanti.eta() : genl.eta()")
    df = df.Define("absetaOthergen", "std::fabs(etaOthergen)")
    df = df.Define("mTVgen", "wrem::mt_2(genl.pt(), genl.phi(), genlanti.pt(), genlanti.phi())")   

    return df

def define_postfsr_vars(df, mode=None):
    if "postfsrLeptons" in df.GetColumnNames():
        logger.debug("PostFSR leptons are already defined, do nothing here.")
        return df

    # status flags in NanoAOD: https://cms-nanoaod-integration.web.cern.ch/autoDoc/NanoAODv9/2016ULpostVFP/doc_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1.html
    # post fsr definition: is stable && (isPrompt or isDirectPromptTauDecayProduct) && is lepton
    df = df.Define("postfsrLeptons", "GenPart_status == 1 && (GenPart_statusFlags & 1 || GenPart_statusFlags & (1 << 5)) && abs(GenPart_pdgId) >= 11 && abs(GenPart_pdgId) <= 16")
    df = df.Define("postfsrElectrons", "postfsrLeptons && abs(GenPart_pdgId) == 11")
    df = df.Define("postfsrMuons", "postfsrLeptons && abs(GenPart_pdgId) == 13")
    df = df.Define("postfsrNeutrinos", "postfsrLeptons && (abs(GenPart_pdgId)==12 || abs(GenPart_pdgId)==14 || abs(GenPart_pdgId)==16)")

    df = df.Define("postfsrNeutrinos_mom4", """wrem::Sum4Vec(
            GenPart_pt[postfsrNeutrinos], GenPart_eta[postfsrNeutrinos], GenPart_phi[postfsrNeutrinos])""")

    if mode is not None:
        # defition of more complex postfsr object 
        # use fiducial gen met, see: https://twiki.cern.ch/twiki/bin/viewauth/CMS/ParticleLevelProducer
        if mode[0] == "z":
            # find the leading charged lepton and antilepton idx
            df = df.Define("postfsrLep", "postfsrLeptons && (GenPart_pdgId==11 || GenPart_pdgId==13)")
            df = df.Define("postfsrAntiLep", "postfsrLeptons && (GenPart_pdgId==-11 || GenPart_pdgId==-13)")
            
            df = df.Define("postfsrLep_idx",     "ROOT::VecOps::ArgMax(GenPart_pt[postfsrLep])")
            df = df.Define("postfsrAntiLep_idx", "ROOT::VecOps::ArgMax(GenPart_pt[postfsrAntiLep])")

            df = df.Define("postfsrLep_pt",     "isEvenEvent ? static_cast<double>(GenPart_pt[postfsrLep][postfsrLep_idx]) : static_cast<double>(GenPart_pt[postfsrAntiLep][postfsrAntiLep_idx])")
            df = df.Define("postfsrLep_eta",    "isEvenEvent ? GenPart_eta[postfsrLep][postfsrLep_idx] : GenPart_eta[postfsrAntiLep][postfsrAntiLep_idx]")
            df = df.Define("postfsrLep_phi",    "isEvenEvent ? GenPart_phi[postfsrLep][postfsrLep_idx] : GenPart_phi[postfsrAntiLep][postfsrAntiLep_idx]")
            df = df.Define("postfsrLep_mass",   "isEvenEvent ? wrem::get_pdgid_mass(GenPart_pdgId[postfsrLep][postfsrLep_idx]) : wrem::get_pdgid_mass(GenPart_pdgId[postfsrAntiLep][postfsrAntiLep_idx])")
            df = df.Define("postfsrLep_charge", "isEvenEvent ? 1 : -1")

            df = df.Define("postfsrOtherLep_pt",   "isEvenEvent ? GenPart_pt[postfsrAntiLep][postfsrAntiLep_idx] : GenPart_pt[postfsrLep][postfsrLep_idx]")
            df = df.Define("postfsrOtherLep_eta",  "isEvenEvent ? GenPart_eta[postfsrAntiLep][postfsrAntiLep_idx] : GenPart_eta[postfsrLep][postfsrLep_idx]")
            df = df.Define("postfsrOtherLep_phi",  "isEvenEvent ? GenPart_phi[postfsrAntiLep][postfsrAntiLep_idx] : GenPart_phi[postfsrLep][postfsrLep_idx]")
            df = df.Define("postfsrOtherLep_mass", "isEvenEvent ? wrem::get_pdgid_mass(GenPart_pdgId[postfsrLep][postfsrLep_idx]) : wrem::get_pdgid_mass(GenPart_pdgId[postfsrAntiLep][postfsrAntiLep_idx])")
        
            df = df.Define("postfsrOtherLep_absEta", "static_cast<double>(std::fabs(postfsrOtherLep_eta))")
        else:
            # find the leading charged lepton or antilepton idx
            df = df.Define("postfsrLep", "postfsrLeptons && (abs(GenPart_pdgId)==11 || abs(GenPart_pdgId)==13)")
            df = df.Define("postfsrLep_idx", "ROOT::VecOps::ArgMax(GenPart_pt[postfsrLep])")

            df = df.Define("postfsrLep_pt", "static_cast<double>(GenPart_pt[postfsrLep][postfsrLep_idx])")
            df = df.Define("postfsrLep_eta", "GenPart_eta[postfsrLep][postfsrLep_idx]")
            df = df.Define("postfsrLep_phi", "GenPart_phi[postfsrLep][postfsrLep_idx]")
            df = df.Define("postfsrLep_mass", "wrem::get_pdgid_mass(GenPart_pdgId[postfsrLep][postfsrLep_idx])")
            df = df.Define("postfsrLep_charge", "GenPart_pdgId[postfsrLep][postfsrLep_idx] > 0 ? -1 : 1")

        df = df.Define("postfsrLep_absEta", "static_cast<double>(std::fabs(postfsrLep_eta))")
        
        if mode[0] == "w" or "wlike" in mode:
            if "wlike" in mode:
                # for wlike selection
                df = df.Define("postfsrMET_wlike", "wrem::get_met_wlike(postfsrOtherLep_pt, postfsrOtherLep_phi, MET_fiducialGenPt, MET_fiducialGenPhi)")
                df = df.Define("postfsrMET_pt", "postfsrMET_wlike.Mod()")
                df = df.Define("postfsrMET_phi", "postfsrMET_wlike.Phi()")
            else:
                df = df.Alias("postfsrMET_pt", "MET_fiducialGenPt")
                df = df.Alias("postfsrMET_phi", "MET_fiducialGenPhi")
                df = df.Define("postfsrPTV", "wrem::pt_2(postfsrLep_pt, postfsrLep_phi, postfsrMET_pt, postfsrMET_phi)")

            df = df.Define("postfsrMT", "wrem::mt_2(postfsrLep_pt, postfsrLep_phi, postfsrMET_pt, postfsrMET_phi)")
            df = df.Define("postfsrDeltaPhiMuonMet", "std::fabs(wrem::deltaPhi(postfsrLep_phi, postfsrMET_phi))")

        # definition of boson kinematics
        if mode[0] == "z":
            # four vectors
            df = df.Define("postfsrLep_mom4", "ROOT::Math::PtEtaPhiMVector(postfsrLep_pt, postfsrLep_eta, postfsrLep_phi, postfsrLep_mass)")
            df = df.Define("postfsrAntiLep_mom4", "ROOT::Math::PtEtaPhiMVector(postfsrOtherLep_pt, postfsrOtherLep_eta, postfsrOtherLep_phi, postfsrOtherLep_mass)")

            df = df.Define('postfsrGenV_mom4', 'postfsrLep_mom4 + postfsrAntiLep_mom4')
            df = df.Define('postfsrMV', 'postfsrGenV_mom4.mass()')
            df = df.Define('postfsrYV', 'postfsrGenV_mom4.Rapidity()')
            df = df.Define('postfsrabsYV', 'std::fabs(postfsrYV)')
            df = df.Define('postfsrPTV', 'postfsrGenV_mom4.pt()')

    return df

def define_ew_vars(df):
    if "ewLeptons" in df.GetColumnNames():
        logger.debug("EW leptons are already defined, do nothing here.")
        return df

    df = df.Define("ewLeptons", "wrem::ewLeptons(GenPart_status, GenPart_statusFlags, GenPart_pdgId, GenPart_pt, GenPart_eta, GenPart_phi)")
    df = df.Define("ewPhotons", "wrem::ewPhotons(GenPart_status, GenPart_statusFlags, GenPart_pdgId, GenPart_pt, GenPart_eta, GenPart_phi)")
    df = df.Define('ewGenV', 'wrem::ewGenVPhos(ewLeptons, ewPhotons)')
    df = df.Define('ewMll', '(ewLeptons[0]+ewLeptons[1]).mass()')
    df = df.Define('ewMlly', 'ewGenV.mass()')
    df = df.Define('ewLogDeltaM', 'log10(ewMlly-ewMll)')

    df = df.Define('ewPTll', '(ewLeptons[0]+ewLeptons[1]).pt()')
    df = df.Define('ewPTlly', 'ewGenV.pt()')
    df = df.Define('ewYll', '(ewLeptons[0]+ewLeptons[1]).Rapidity()')
    df = df.Define('ewAbsYll', 'std::fabs(ewYll)')
    df = df.Define('ewYlly', 'ewGenV.Rapidity()')

    return df

def make_ew_binning(mass = 91.1535, width = 2.4932, initialStep = 0.1, bin_edges_low=[], bin_edges_high=[]):
    maxVal = ROOT.Math.breitwigner_pdf(mass, width, mass)
    bins = [mass]
    currentMass = mass
    while currentMass - mass < 100:
        binSize = maxVal / ROOT.Math.breitwigner_pdf(currentMass, width, mass) * initialStep
        currentMass += binSize
        bins.append(currentMass)
        lowMass = 2*mass - currentMass
        if lowMass - binSize > 0:
            bins.insert(0, lowMass)
    bins.insert(0, 0.)

    if bin_edges_low:
        bins = bin_edges_low + [b for b in bins if b > bin_edges_low[-1]][1:]
    if bin_edges_high:
        bins = [b for b in bins if b < bin_edges_high[-1]][:-1] + bin_edges_high

    return bins

def pdf_info_map(dataset, pdfset):
    infoMap = pdfMap 

    # Just ignore PDF variations for non W/Z samples
    if pdfset is None \
        or not (dataset[0] in ["W", "Z"] and dataset[1] not in ["W", "Z"]) \
        or "horace" in dataset or (pdfset != "nnpdf31" and dataset in only_central_pdf_datasets) \
        or pdfset not in infoMap:
        raise ValueError(f"Skipping PDF {pdfset} for dataset {dataset}")
    return infoMap[pdfset]

def define_pdf_columns(df, dataset_name, pdfs, noAltUnc):
    if len(pdfs) == 0 \
        or dataset_name not in common.vprocs_all \
        or "horace" in dataset_name \
        or "winhac" in dataset_name \
        or "LHEPdfWeight" not in df.GetColumnNames():
        logger.warning(f"Did not find PDF weights for sample {dataset_name}! Using nominal PDF in sample")
        return df

    for i, pdf in enumerate(pdfs):
        try:
            pdfInfo = pdf_info_map(dataset_name, pdf)
        except ValueError as e:
            return df

        pdfName = pdfInfo["name"]
        pdfBranch = pdfInfo["branch"]
        tensorName = f"{pdfName}Weights_tensor"
        tensorASName = f"{pdfName}ASWeights_tensor"
        entries = 1 if i != 0 and noAltUnc else pdfInfo["entries"]
        start = 0 if "first_entry" not in pdfInfo else pdfInfo["first_entry"]

        if pdfBranch not in df.GetColumnNames():
            return df

        if "renorm" in pdfInfo and pdfInfo["renorm"]:
            df = df.Define(tensorName, f"auto res = wrem::vec_to_tensor_t<double, {entries}>({pdfBranch}, {start}); res = res/res(0); "
                           "res = wrem::clip_tensor(res, theory_weight_truncate); res = res*nominal_weight; return res;")
        else:
            df = df.Define(tensorName, f"auto res = wrem::clip_tensor(wrem::vec_to_tensor_t<double, {entries}>({pdfBranch}, {start}), theory_weight_truncate); res = nominal_weight/central_pdf_weight*res; return res;")

        if i == 0:
            tensorNameNominal = tensorName

        if pdfName == "pdfMSHT20":
            df = pdfBugfixMSHT20(df, tensorName)

        if "alphas" in pdfInfo:
            df = df.Define(tensorASName, f"Eigen::TensorFixedSize<double, Eigen::Sizes<{len(pdfInfo['alphas'])}>> res; " + \
                    " ".join([f"res({i}) = nominal_weight/central_pdf_weight*{p};" for i,p in enumerate(pdfInfo['alphas'])]) + \
                    "return wrem::clip_tensor(res, theory_weight_truncate)")

    return df

def define_central_pdf_weight(df, dataset_name, pdf):
    try:
        pdfInfo = pdf_info_map(dataset_name, pdf)
    except ValueError as e:
        logger.warning(f"Did not find PDF {pdf} for sample {dataset_name}! Using nominal PDF in sample")
        return df.DefinePerSample("central_pdf_weight", "1.0")

    pdfName = pdfInfo["name"]
    pdfBranch = pdfInfo["branch"]
    if not pdfBranch in df.GetColumnNames():
        logger.warning(f"Did not find PDF branch {pdfBranch} for sample {dataset_name}! Set PDF weights to 1")
        return df.DefinePerSample("central_pdf_weight", "1.0")
    first_entry = pdfInfo.get("first_entry", 0)
    return df.Define("central_pdf_weight", f"std::clamp<float>({pdfBranch}[{first_entry}], -theory_weight_truncate, theory_weight_truncate)")

def define_theory_weights_and_corrs(df, dataset_name, helpers, args):
    if not 'powheg' in dataset_name:
        # no preFSR particles in powheg samples
        df = define_prefsr_vars(df)

    df = define_ew_vars(df)

    df = df.DefinePerSample("theory_weight_truncate", "10.")
    df = define_central_pdf_weight(df, dataset_name, args.pdfs[0] if len(args.pdfs) >= 1 else None)
    df = define_theory_corr(df, dataset_name, helpers, generators=args.theoryCorr, 
        modify_central_weight=not args.theoryCorrAltOnly)
    df = define_ew_theory_corr(df, dataset_name, helpers, generators=args.ewTheoryCorr)

    if args.highptscales:
        df = df.Define("extra_weight", "MEParamWeightAltSet3[0]")
    df = define_nominal_weight(df)
    df = define_pdf_columns(df, dataset_name, args.pdfs, args.altPdfOnlyCentral)
        
    return df 

def build_weight_expr(df, exclude_weights=[]):
    valid_cols = df.GetColumnNames()
    weights = ["weight", "central_pdf_weight", "theory_corr_weight", "ew_theory_corr_weight", "exp_weight"]
    if weights[0] not in valid_cols:
        raise ValueError(f"The weight '{weights[0]}' must be defined in the histmaker!")
    found_weights = []

    for weight in filter(lambda x: x not in exclude_weights, weights):
        if weight not in valid_cols:
            logger.warning(f"Did not find weight '{weight}'! Assuming 1.0")
        else:
            found_weights.append(weight)

    if "extra_weight" in valid_cols:
        logger.info("Adding additional weight '{extra_weight}'")
        found_weights.append("extra_weight")

    weight_expr = "*".join(found_weights)

    logger.debug(f"Weight is {weight_expr}")

    return weight_expr

def define_nominal_weight(df):
    logger.debug("Defining nominal weight")
    return df.Define(f"nominal_weight", build_weight_expr(df))

def define_ew_theory_corr(df, dataset_name, helpers, generators, modify_central_weight=False):
    df = df.Define(f"nominal_weight_ew_uncorr", build_weight_expr(df, exclude_weights=["ew_theory_corr_weight"]))

    dataset_helpers = helpers.get(dataset_name, [])

    if not modify_central_weight or not generators or generators[0] not in dataset_helpers:
        df = df.DefinePerSample("ew_theory_corr_weight", "1.0")

    for i, generator in enumerate(generators):
        if generator not in dataset_helpers:
            continue

        logger.debug(f"Now at generator {i}: {generator}")
        helper = dataset_helpers[generator]
        df = df.Define(f"ew_{generator}corr_weight", build_weight_expr(df))
        df = df.Define(f"{generator}Weight_tensor", helper, [*helper.hist.axes.name[:-2], "chargeVgen", f"ew_{generator}corr_weight"]) # multiplying with nominal QCD weight

        if i == 0 and modify_central_weight:
            df = df.Define("ew_theory_corr_weight", f"nominal_weight_ew_uncorr == 0 ? 0 : {generator}Weight_tensor(0)/nominal_weight_ew_uncorr")

    return df

def define_theory_corr(df, dataset_name, helpers, generators, modify_central_weight):
    df = df.Define(f"nominal_weight_uncorr", build_weight_expr(df, exclude_weights=["theory_corr_weight"]))

    dataset_helpers = helpers.get(dataset_name, [])
    
    if not modify_central_weight or not generators or generators[0] not in dataset_helpers:
        df = df.DefinePerSample("theory_corr_weight", "1.0")

    for i, generator in enumerate(generators):
        if generator not in dataset_helpers:
            continue

        logger.debug(f"Now at generator {i}: {generator}")

        helper = dataset_helpers[generator]

        if "Helicity" in generator:
            df = df.Define(f"{generator}Weight_tensor", helper, ["massVgen", "absYVgen", "ptVgen", "chargeVgen", "csSineCosThetaPhigen", "nominal_weight_uncorr"])
        else:
            df = define_theory_corr_weight_column(df, generator)
            df = df.Define(f"{generator}Weight_tensor", helper, ["massVgen", "absYVgen", "ptVgen", "chargeVgen", f"{generator}_corr_weight"])

        if i == 0 and modify_central_weight:
            df = df.Define("theory_corr_weight", f"nominal_weight_uncorr == 0 ? 0 : {generator}Weight_tensor(0)/nominal_weight_uncorr")

    return df

def define_theory_corr_weight_column(df, generator):
    if generator in theory_corr_weight_map:
        values = theory_corr_weight_map[generator]
        df = df.Define(f"{generator}_corr_weight", f"Eigen::TensorFixedSize<double, Eigen::Sizes<{len(values)}>> res; " + \
            "; ".join([f"res({i}) = {entry}*nominal_weight_uncorr/central_pdf_weight" for i,entry in enumerate(values)]) + \
            "; return res;")
    else:
        df = df.Alias(f"{generator}_corr_weight", "nominal_weight_uncorr")

    return df

def make_theory_corr_hists(df, name, axes, cols, helpers, generators, modify_central_weight, isW, storage_type=hist.storage.Double()):
    res = []
    
    for i, generator in enumerate(generators):
        if generator not in helpers:
            continue
        
        if i == 0 and modify_central_weight:
            res.append(df.HistoBoost(f"{name}_uncorr", axes, [*cols, "nominal_weight_uncorr"], storage=storage_type))
            if name == "nominal":
                res.append(df.HistoBoost(f"weight_uncorr", [hist.axis.Regular(100, -2, 2)], ["nominal_weight_uncorr"], storage=storage_type))

        var_axis = helpers[generator].tensor_axes[-1]

        hist_name = f"{name}_{generator}Corr"
        weight_tensor_name = f"{generator}Weight_tensor"
        unc = df.HistoBoost(hist_name, axes, [*cols, weight_tensor_name], tensor_axes=[var_axis], storage=storage_type)
        res.append(unc)

        def is_flavor_dependent_np(var_label):
            return var_label.startswith("Omega") \
                    or var_label.startswith("Delta_Omega") \
                    or var_label.startswith("Lambda2") \
                    or var_label.startswith("Delta_Lambda2") \
                    or var_label.startswith("Lambda4")

        # special treatment for Lambda2/Omega since they need to be decorrelated in charge and possibly rapidity
        if isinstance(var_axis, hist.axis.StrCategory) and any(is_flavor_dependent_np(var_label) for var_label in var_axis):
            omegaidxs = [var_axis.index(var_label) for var_label in var_axis if is_flavor_dependent_np(var_label)]

            # include nominal as well
            omegaidxs = [0] + omegaidxs

            if f"{generator}FlavDepNP" not in df.GetColumnNames():
                np_idx_helper = ROOT.wrem.index_taker[df.GetColumnType(weight_tensor_name), len(omegaidxs)](omegaidxs)

                df = df.Define(f"{generator}FlavDepNP", np_idx_helper, [weight_tensor_name])

            axis_FlavDepNP = hist.axis.StrCategory([var_axis[idx] for idx in omegaidxs], name = var_axis.name)

            hist_name_FlavDepNP = f"{name}_{generator}FlavDepNP"
            axis_chargegen = axis_chargeWgen if isW else axis_chargeZgen
            axes_FlavDepNP = axes + [axis_absYVgen, axis_chargegen]
            cols_FlavDepNP = cols + ["absYVgen", "chargeVgen", f"{generator}FlavDepNP"]
            unc_FlavDepNP = df.HistoBoost(hist_name_FlavDepNP, axes_FlavDepNP, cols_FlavDepNP, tensor_axes = [axis_FlavDepNP])
            res.append(unc_FlavDepNP)

        def is_pt_dependent_scale(var_label):
            return var_label.startswith("renorm_fact_resum_transition_scale_envelope") \
                    or var_label.startswith("renorm_fact_resum_scale_envelope")

        # special treatment for envelope of scale variations since they need to be decorrelated in pt
        if isinstance(var_axis, hist.axis.StrCategory) and any(is_pt_dependent_scale(var_label) for var_label in var_axis):

            scaleidxs = [var_axis.index(var_label) for var_label in var_axis if is_pt_dependent_scale(var_label)]

            # include nominal as well
            scaleidxs = [0] + scaleidxs

            if f"{generator}PtDepScales" not in df.GetColumnNames():
                scale_idx_helper = ROOT.wrem.index_taker[df.GetColumnType(weight_tensor_name), len(scaleidxs)](scaleidxs)

                df = df.Define(f"{generator}PtDepScales", scale_idx_helper, [weight_tensor_name])

            axis_PtDepScales = hist.axis.StrCategory([var_axis[idx] for idx in scaleidxs], name = var_axis.name)

            hist_name_PtDepScales = f"{name}_{generator}PtDepScales"
            axis_ptVgen = hist.axis.Variable(common.ptV_binning, name = "ptVgen", underflow=False)

            axes_PtDepScales = axes[:]
            cols_PtDepScales = cols[:]
            if "ptVgen" not in cols:
                axes_PtDepScales += [axis_ptVgen]
                cols_PtDepScales += ["ptVgen", f"{generator}PtDepScales"]
            unc_PtDepScales = df.HistoBoost(hist_name_PtDepScales, axes_PtDepScales, cols_PtDepScales, tensor_axes = [axis_PtDepScales])
            res.append(unc_PtDepScales)

    return res

def replace_by_neighbors(vals, replace):
    if np.count_nonzero(replace) == vals.size:
        raise ValueError("Cannot replace all values with nearest non-zero neighbour")

    indices = ndimage.distance_transform_edt(replace, return_distances=False, return_indices=True)
    return vals[tuple(indices)]

def moments_to_angular_coeffs(hist_moments_scales, cutoff=1e-5):
    if hist_moments_scales.empty():
       raise ValueError("Cannot make coefficients from empty hist")
    # broadcasting happens right to left, so move to rightmost then move back
    hel_ax = hist_moments_scales.axes["helicity"]
    hel_idx = hist_moments_scales.axes.name.index("helicity")
    vals = np.moveaxis(hist_moments_scales.view(flow=True), hel_idx, -1)
    values = vals.value if hasattr(vals,"value") else vals
    
    # select constant term, leaving dummy axis for broadcasting
    unpol_idx = hel_ax.index(-1)
    norm_vals = values[...,unpol_idx:unpol_idx+1]
    norm_vals = np.where(np.abs(norm_vals) < cutoff, np.ones_like(norm_vals), norm_vals)

    coeffs = vals / norm_vals

    coeffs = np.moveaxis(coeffs, -1, hel_idx)

    hist_coeffs_scales = hist.Hist(*hist_moments_scales.axes, storage = hist_moments_scales._storage_type(),
        name = "hist_coeffs_scales", data = coeffs
    )

    return hist_coeffs_scales

def moments_to_helicities(hist_moments_scales):
    factors = np.array([1., 1./2., 1./(2.*sqrt(2.)), 1./4, 1./(4.*sqrt(2.)),1./2.,1./2.,1./(2.*sqrt(2.)),1./(4.*sqrt(2.))])
    
    hfactors = hist.Hist(hist_moments_scales.axes["helicity"],
        data = factors
            )
    hist_moments_scales_new = hh.multiplyHists(hfactors,hist_moments_scales)

    return hist_moments_scales_new

def qcdByHelicityLabels():
    coeffs = ["const"]+[f"a{i}" for i in range(8)]
    scaleVars = ["muRmuF", "muR", "muF"]
    return  [f"{var}_{coeff}{t}" for var in scaleVars for t in ["Up", "Down"] for coeff in coeffs]

def qcdScaleNames():
    # Exclude central and extreme variations
    shifts = ["muRmuFDown", "muRDown", "", "muFDown", "", "muFUp", "muRUp", "", "muRmuFUp"]
    return ["_".join(["QCDscale", s]) if s != "" else s for s in shifts]

def pdfNames(cardTool, pdf, skipFirst=True):
    size = 101
    names = cardTool.mirrorNames(f"pdf{{i}}{pdf}", size)
    if skipFirst:
        names[0] = ""
        names[size] = ""
    # TODO: This is probably not needed anymore, check with low PU
    if False and pdf == "NNPDF31":
        names[size-2] = "pdfAlphas002Up"
        names[size-1] = "pdfAlphas002Down"
        # Drop the mirrored alphaS variations
        names[size*2-2] = ""
        names[size*2-1] = ""
    return names

def pdfNamesAsymHessian(entries, pdfset=""):
    pdfNames = ["pdf0"+pdfset.replace("pdf", "")] 
    pdfNames.extend([f"pdf{int((j+2)/2)}{pdfset.replace('pdf', '')}{'Up' if j % 2 else 'Down'}" for j in range(entries-1)])
    return pdfNames

def pdfNamesSymHessian(entries, pdfset=""):
    return [f"pdf{i+1}{pdfset.replace('pdf', '')}" for i in range(entries)]

def pdfSymmetricShifts(hdiff, axis_name):
    sq = hh.multiplyHists(hdiff, hdiff)
    ss = sq[{axis_name : hist.sum}]
    rss = hh.sqrtHist(ss)
    return rss, rss

def pdfAsymmetricShifts(hdiff, axis_name):
    # Assuming that the last axis is the syst axis
    # TODO: add some check to verify this
    def shiftHist(vals, hdiff, axis_name):
        hnew = hdiff[{axis_name : 0}]
        vals = vals*vals
        hnew[...] = np.sum(vals, axis=-1)
        return hh.sqrtHist(hnew)

    ax = hdiff.axes[axis_name] 
    underflow = hdiff.axes[axis_name].traits.underflow
    overflow = hdiff.axes[axis_name].traits.overflow
    if type(ax) == hist.axis.StrCategory and all(["Up" in x or "Down" in x for x in ax][1:]):
        # Remove the overflow from the categorical axis
        end = int((ax.size-1)/2)
        upvals = hdiff[{axis_name : [x for x in ax if "Up" in x]}].values(flow=True)[...,:end]
        downvals = hdiff[{axis_name : [x for x in ax if "Down" in x]}].values(flow=True)[...,:end]
        if upvals.shape != downvals.shape:
            raise ValueError("Malformed PDF uncertainty hist! Expect equal number of up and down vars")
    else:
        end = ax.size+underflow
        upvals = hdiff.values(flow=True)[...,1+underflow:end:2]
        downvals = hdiff.values(flow=True)[...,2+underflow:end:2]

    # The error sets are ordered up,down,up,down...
    upshift = shiftHist(upvals, hdiff, axis_name)
    downshift = shiftHist(downvals, hdiff, axis_name)
    return upshift, downshift 

def hessianPdfUnc(h, axis_name="pdfVar", uncType="symHessian", scale=1.):
    underflow = h.axes[axis_name].traits.underflow
    symmetric = uncType == "symHessian"
    diff = hh.addHists(h, -1*h[{axis_name : 0}])*scale
    if diff.axes[axis_name].traits.overflow:
        diff[...,hist.overflow] = np.zeros_like(diff[{axis_name : 0}].view(flow=True))
    shiftFunc = pdfSymmetricShifts if symmetric else pdfAsymmetricShifts
    rssUp, rssDown = shiftFunc(diff, axis_name)
    hUp = hh.addHists(h[{axis_name : 0}], 1*rssUp)
    hDown = hh.addHists(h[{axis_name : 0}], -1*rssDown)
    return hUp, hDown

def pdfBugfixMSHT20(df , tensorPDFName):
    # There is a known bug in MSHT20 where member 15 and 16 are identical
    #   to fix this, one has to be mirrored:
    #   pdf(15) = pdf(0) - (pdf(15) - pdf(0))
    return df.Redefine(tensorPDFName, 
        f"auto& res = {tensorPDFName};"
        f"res(15) = {tensorPDFName}(0) - ({tensorPDFName}(15) - {tensorPDFName}(0));"
        "return res")
