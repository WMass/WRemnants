import pathlib

from wums import logging

logger = logging.child_logger(__name__)

base_dir = pathlib.Path(__file__).parent.joinpath("..", "..").resolve()
wremnants_dir = f"{base_dir}/wremnants/"
data_dir = f"{base_dir}/wremnants-data/data/"

BR_Z_LEP = 3 * 0.0336  # PDG
BR_Z_Nu = 3 * 0.067
BR_Z_Q = 1 - (BR_Z_LEP + BR_Z_Nu)
BR_W_LEP = 3 * 0.1086  # PDG
BR_TAUToMU = 0.1739
BR_TAUToE = 0.1782
Z_TAU_TO_LEP_RATIO = 1.0 - (1.0 - BR_TAUToMU - BR_TAUToE) ** 2

# cross sections in pb at sqrt(s)=13Tev (TODO: add source information)
xsec_DYJetsToLL = 2001.9
xsec_WplusJetsToLNu = 11765.9
xsec_WminusJetsToLNu = 8703.87
xsec_DYJetsToLLMass10to50 = 6997.0

xsec_WW = 118.7
xsec_WZ = 47.13  # from https://twiki.cern.ch/twiki/bin/view/CMS/SummaryTable1G25ns
xsec_ZZ = 16.523  # from https://twiki.cern.ch/twiki/bin/view/CMS/SummaryTable1G25ns

# TODO replace by BR
xsec_WWTo2L2Nu = 12.6  # xsec_WW * BR_W_LEP * BR_W_LEP
xsec_WWTo1L1Nu = 52.146  # xsec_WW * BR_W_LEP * (1 - BR_W_LEP) * 2 # (2 is because one W or the other can go to Q)
xsec_WZTo3LNu = 4.91  # 4.42965*1.109, 1.109 is the NLO to NNLO kfactor, for this one would need to make sure about the NLO XS, depends a lot on the dilepton mass cut
xsec_WZTo2Q2L = 5.4341  # 4.9*1.109
xsec_WZTo1L1Nu2Q = 11.781  # 10.71*1.10
xsec_ZZTo2L2Nu = 0.60  # check xsec_ZZ * BR_Z_Nu * BR_Z_LEP * 2
xsec_ZZTo2Q2L = 5.1  # check xsec_ZZ * BR_Z_Q * (BR_Z_LEP+BR_Z_Nu) * 2

# ------------------------------------
# GenXsecAnalyzer:
# ------------------------------------
# Before Filter: total cross section = 3.653e+02 +- 1.572e-02 pb
# Filter efficiency (taking into account weights)= (1.98515e+06) / (1.29071e+08) = 1.538e-02 +- 1.083e-05
# Filter efficiency (event-level)= (1.98515e+06) / (1.29071e+08) = 1.538e-02 +- 1.083e-05    [TO BE USED IN MCM]
# After filter: final cross section = 5.619e+00 +- 3.965e-03 pb
# After filter: final fraction of events with negative weights = 0.000e+00 +- 0.000e+00
# After filter: final equivalent lumi for 1M events (1/fb) = 1.780e+02 +- 2.178e-01
xsec_GGtoMuMu = 5.619

# BSM heavy neutrino samples, just a dummy number
xsec_WtoNMu = 100

# input files for muon momentum scale nuisances
calib_dir = f"{data_dir}/calibration/"
closure_dir = f"{data_dir}/closure/"
calib_filepaths = {
    "mc_corrfile": {
        "idealMC_massfit": f"{calib_dir}/calibrationJMC_smeared_v718_nominal.root",
        "idealMC_lbltruth_massfit": f"{calib_dir}/calibrationJMC_smeared_v718_nominalLBL.root",
    },
    "data_corrfile": {
        "massfit": f"{calib_dir}/calibrationJDATA_ideal.root",
        "lbl_massfit": f"{calib_dir}/calibrationJDATA_MCstat_inclusive_binsel.root",
        # 'lbl_massfit': f"{calib_dir}/calibrationJZ_DATA_MCstat_binsel.root"
    },
    "mc_resofile": f"{calib_dir}/sigmaMC_LBL_JYZ.root",
    "data_resofile": f"{calib_dir}/sigmaDATA_LBL_JYZ.root",
    "tflite_file": f"{calib_dir}/muon_response.tflite",
    # 'tflite_file': f"{calib_dir}/muon_response_nosmearing.tflite"
    "kaon_tflite_file": f"{calib_dir}/kaon_response.tflite",
}
closure_filepaths = {
    "parametrized": f"{closure_dir}/parametrizedClosureZ_ORkinweight_binsel_MCstat_fullres.root",
    # 'parametrized': f"{closure_dir}/parametrizedClosureZ_ORkinweight_binsel_MCstat_simul.root",
    "binned": f"{closure_dir}/closureZ_LBL_smeared_v721.root",
}

# some constants for momentum scale uncertainties
correlated_variation_base_size = {
    "A": 1e-5,
    "M": 1e-6,
}

# following list is used in other scripts to track what steps are charge dependent
# but assumes the corresponding efficiencies were made that way
muonEfficiency_chargeDependentSteps = [
    "reco",
    "tracking",
    "idip",
    "trigger",
    "antitrigger",
]  # antitrigger = P(failTrig|IDIP), similar to antiiso = P(failIso|trigger)
muonEfficiency_altBkgSyst_effSteps = ["reco", "tracking"]
muonEfficiency_standaloneNumberOfValidHits = (
    1  # to use as "var >= this" (if this=0 the define for the cut is not used at all)
)


def hist_name(baseName, syst=""):
    if baseName != "x" and (syst == ""):
        return baseName
    if baseName in ["", "x"] and syst:
        return syst
    if syst[: len(baseName)] == baseName:
        return syst
    return "_".join([baseName, syst])


analysis_mode_map = {
    "w_z_gen_dists.py": "vgen",
    "mz_dilepton.py": "z_dilepton",
    "mz_wlike_with_mu_eta_pt.py": "z_wlike",
    "mw_with_mu_eta_pt.py": "w_mass",
    "mw_lowPU.py": "w_lowpu",
    "mz_lowPU.py": "z_lowpu",
}


def analysis_label(filename):
    if filename not in analysis_mode_map:
        logger.warning(
            f"Unrecognized analysis script {filename}! Expected one of {analysis_mode_map.keys()}"
        )
        return filename.replace(".py", "")
    else:
        return analysis_mode_map[filename]
