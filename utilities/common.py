import pathlib
import re

import hist
import numpy as np

base_dir = f"{pathlib.Path(__file__).parent}/../"
wremnants_dir = f"{pathlib.Path(__file__).parent}/../wremnants"
data_dir = f"{pathlib.Path(__file__).parent}/../wremnants-data/data/"

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

## eras
eras_run2 = ["2016PreVFP", "2016PostVFP", "2017", "2018", "13TeVGen"]

supported_eras = eras_run2 + [
    "2016PostVFP",
    "2017G",
    "2017H",
    "2023_PUAVE1",
    "2023_PUAVE2",
    "2023_PUAVE5",
    "2023_PUAVE10",
    "13TeVGen",
]

## Samples with sqrt{S} = 13 TeV
# central MiNNLO samples with muon decay
wprocs_mu_minnlo_run2 = [f"Wplusmunu_{e}" for e in eras_run2] + [
    f"Wminusmunu_{e}" for e in eras_run2
]
zprocs_mu_minnlo_run2 = [
    f"Zmumu_{e}" for e in eras_run2
]  # + [f"DYJetsToMuMuMass10to50_{e}" for e in eras_run2]

# central MiNNLO samples with muon or e decay
wprocs_emu_minnlo_2017H = [
    "Wplusmunu_2017H",
    "Wminusmunu_2017H",
    "Wplusenu_2017H",
    "Wminusenu_2017H",
]
zprocs_emu_minnlo_2017H = ["Zmumu_2017H", "Zee_2017H"]
vprocs_emu_minnlo_2017H = wprocs_emu_minnlo_2017H + zprocs_emu_minnlo_2017H

wprocs_emu_minnlo = wprocs_mu_minnlo_run2 + wprocs_emu_minnlo_2017H
zprocs_emu_minnlo = zprocs_mu_minnlo_run2 + zprocs_emu_minnlo_2017H
vprocs_emu_minnlo = wprocs_emu_minnlo + zprocs_emu_minnlo

# central MiNNLO samples with tau
wprocs_tau_minnlo_run2 = [f"Wplustaunu_{e}" for e in eras_run2] + [
    f"Wminustaunu_{e}" for e in eras_run2
]
zprocs_tau_minnlo_run2 = [f"Ztautau_{e}" for e in eras_run2] + [
    f"Ztautau10to50_{e}" for e in eras_run2
]

wprocs_tau_minnlo_2017H = [
    "Wplustaunu_2017H",
    "Wminustaunu_2017H",
]
zprocs_tau_minnlo_2017H = [
    "Ztautau_2017H",
]

wprocs_tau_minnlo = wprocs_tau_minnlo_run2 + wprocs_tau_minnlo_2017H
zprocs_tau_minnlo = zprocs_tau_minnlo_run2 + zprocs_tau_minnlo_2017H
vprocs_tau_minnlo = wprocs_tau_minnlo + zprocs_tau_minnlo

wprocs_minnlo = wprocs_emu_minnlo + wprocs_tau_minnlo
zprocs_minnlo = zprocs_emu_minnlo + zprocs_tau_minnlo
vprocs_minnlo = wprocs_minnlo + zprocs_minnlo

wprocs_2017H = wprocs_emu_minnlo_2017H + wprocs_tau_minnlo_2017H
zprocs_2017H = zprocs_emu_minnlo_2017H + zprocs_tau_minnlo_2017H
vprocs_2017H = wprocs_2017H + zprocs_2017H

# alternative gen samples at sqrt{s} = 13
wprocs_alt = [
    "Wplusmunu_MiNNLO",
    "Wminusmunu_MiNNLO",
    "Wplusmunu_MiNNLO-noqedisr",
    "Wminusmunu_MiNNLO-noqedisr",
    "Wplusmunu_horace-lo-photos",
    "Wplusmunu_horace-lo-photos-mecoff",
    "Wplusmunu_horace-nlo",
    "Wplusmunu_horace-lo",
    "Wplusmunu_horace-qed",
    "Wminusmunu_horace-lo-photos",
    "Wminusmunu_horace-lo-photos-mecoff",
    "Wminusmunu_horace-nlo",
    "Wminusmunu_horace-lo",
    "Wminusmunu_horace-qed",
    "Wplusmunu_winhac-lo-photos",
    "Wplusmunu_winhac-lo",
    "Wplusmunu_winhac-nlo",
    "Wminusmunu_winhac-lo-photos",
    "Wminusmunu_winhac-lo",
    "Wminusmunu_winhac-nlo",
    "WplusCharmToMuNu",
    "WminusCharmToMuNu",
]
zprocs_alt = [
    "Zmumu_MiNNLO",
    "Zmumu_13TeVGen",
    "ZmumuMiNLO",
    "ZmumuNNLOPS",
    "Zmumu_MiNNLO-noqedisr",
    "Zmumu_horace-lo-photos",
    "Zmumu_horace-lo-photos-isroff",
    "Zmumu_horace-lo-photos-mecoff",
    "Zmumu_horace-nlo",
    "Zmumu_horace-lo",
    "Zmumu_horace-new",
    "Zmumu_horace-qed",
    "Zmumu_horace-alpha-fsr-off-isr-off",
    "Zmumu_horace-alpha-old-fsr-off-isr-off",
    "Zmumu_horace-alpha-old-fsr-off-isr-pythia",
    "Zmumu_renesance-lo",
    "Zmumu_renesance-nlo",
    "Zmumu_powheg-lo",
    "Zmumu_powheg-nloew-qedveto",
    "Zmumu_powheg-nloew",
]

wprocs_bsm = [
    "WtoNMuMass5_2016PostVFP",
    "WtoNMuMass10_2016PostVFP",
    "WtoNMuMass30_2016PostVFP",
    "WtoNMuMass50_2016PostVFP",
    "WtoMuNuSMEFT_2016PostVFP",
]

## Samples with sqrt{S} = 5020GeV
wprocs_emu_minnlo_2017G = [
    "Wplusmunu_2017G",
    "Wminusmunu_2017G",
    "Wplusenu_2017G",
    "Wminusenu_2017G",
]
zprocs_emu_minnlo_2017G = ["Zmumu_2017G", "Zee_2017G"]
vprocs_emu_minnlo_2017G = wprocs_emu_minnlo_2017G + zprocs_emu_minnlo_2017G

wprocs_tau_minnlo_2017G = [
    "Wplustaunu_2017G",
    "Wminustaunu_2017G",
]
zprocs_tau_minnlo_2017G = [
    "Ztautau_2017G",
]
vprocs_tau_minnlo_2017G = wprocs_tau_minnlo_2017G + zprocs_tau_minnlo_2017G

wprocs_minnlo_2017G = wprocs_emu_minnlo_2017G + wprocs_tau_minnlo_2017G
zprocs_minnlo_2017G = zprocs_emu_minnlo_2017G + zprocs_tau_minnlo_2017G
vprocs_minnlo_2017G = wprocs_minnlo_2017G + zprocs_minnlo_2017G

# all W and Z samples
wprocs = wprocs_minnlo + wprocs_alt + wprocs_bsm + wprocs_minnlo_2017G
zprocs = zprocs_minnlo + zprocs_alt + zprocs_minnlo_2017G
vprocs = wprocs + zprocs + vprocs_minnlo_2017G

zprocs_recoil = ["Zmumu_2016PostVFP"]
wprocs_recoil = ["Wplusmunu_2016PostVFP", "Wminusmunu_2016PostVFP"]

zprocs_recoil_lowpu = ["Zmumu", "Zee"]
wprocs_recoil_lowpu = ["Wminusmunu", "Wminusenu", "Wplusmunu", "Wplusenu"]

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

## for W used in SMP-18-012
# 10% quantiles from aMC@NLO used in SMP-18-012 with some rounding <== This one worked fine with toys
ptV_10quantiles_binning = [
    0.0,
    2.95,
    4.73,
    6.68,
    8.98,
    11.78,
    15.33,
    20.11,
    27.17,
    40.15,
    13000.0,
]
# 5% quantiles from aMC@NLO used in SMP-18-012
ptV_20quantiles_binning = [
    0.0,
    1.971,
    2.949,
    3.838,
    4.733,
    5.674,
    6.684,
    7.781,
    8.979,
    10.303,
    11.777,
    13.435,
    15.332,
    17.525,
    20.115,
    23.245,
    27.173,
    32.414,
    40.151,
    53.858,
    13000.0,
]
# Integer rounded version of the 5% quantiles h[::hist.rebin(2)] for 10% quantiles
ptV_binning = [
    0,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    15,
    17,
    20,
    23,
    27,
    32,
    40,
    54,
    13000,
]
## for Z
# approximate 2.5% quantiles, used in SMP-25-16, SMP-25-17 for the Z detector level fits
ptZ_binning = [
    0,
    1,
    1.5,
    2,
    2.5,
    3,
    3.5,
    4,
    4.5,
    5,
    5.5,
    6,
    6.5,
    7,
    7.5,
    8,
    8.5,
    9,
    9.5,
    10,
    10.5,
    11,
    11.5,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    22,
    24,
    26,
    28,
    30,
    33,
    37,
    44,
    100,
]
# for the Z for SMP-25-016, SMP-25-17
yll_10quantiles_binning = [-2.5, -1.5, -1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0, 1.5, 2.5]
yll_20quantiles_binning = [
    -2.5,
    -1.8,
    -1.5,
    -1.3,
    -1.1,
    -0.9,
    -0.7,
    -0.5,
    -0.3,
    -0.15,
    0,
    0.15,
    0.3,
    0.5,
    0.7,
    0.9,
    1.1,
    1.3,
    1.5,
    1.8,
    2.5,
]

## for Ai based corrections and uncertainties (e.g. TheoryCorrections/ByHelicity/)
# for the W, 40 quantiles
ptWgen_binning_corr = [
    0,
    1,
    2,
    2.5,
    3,
    3.5,
    4,
    4.5,
    5,
    5.5,
    6,
    6.5,
    7,
    7.5,
    8,
    8.5,
    9,
    9.5,
    10,
    10.5,
    11,
    11.5,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    22,
    24,
    26,
    28,
    30,
    33,
    37,
    44,
    54,
    75,
    100,
    13000,
]
absYWgen_binning_corr = [
    0,
    0.25,
    0.5,
    0.75,
    1,
    1.25,
    1.5,
    1.75,
    2,
    2.25,
    2.5,
    2.75,
    3,
    3.25,
    3.5,
    3.75,
    4,
    5,
]
# for the Z, based on reco binning, but including additional bins where reco binning is too coarse
ptZgen_binning_corr = [*ptZ_binning[:-1], 54, 75, 100, 1300]
absYZgen_binning_corr = [
    *yll_20quantiles_binning[10:-1],
    2.0,
    2.25,
    2.5,
    2.75,
    3,
    3.25,
    3.5,
    3.75,
    4,
    5,
]

# categorical axes in python bindings always have an overflow bin, so use a regular axis for the charge
axis_charge = hist.axis.Regular(
    2, -2.0, 2.0, underflow=False, overflow=False, name="charge"
)

down_up_axis = hist.axis.Regular(
    2, -2.0, 2.0, underflow=False, overflow=False, name="downUpVar"
)
down_nom_up_axis = hist.axis.Regular(
    3, -1.5, 1.5, underflow=False, overflow=False, name="downNomUpVar"
)

# run edges chosen to separate eras (era F post VFP: [278769, 278808], era G [278820, 280385], era F [281613, 284044])
run_edges = np.array(
    [
        278768,
        278808,
        279588,
        279767,
        280017,
        280385,
        282037,
        283270,
        283478,
        283934,
        284044,
    ]
)
run_edges_lumi = np.array(
    [0.0, 0.419, 2.332, 4.329, 6.247, 8.072, 10.152, 12.265, 14.067, 15.994, 16.812]
)

# for fake estimation
# binary categories for simple ABCD method
passIsoName = "passIso"
passMTName = "passMT"

passIso = {passIsoName: True}
failIso = {passIsoName: False}
passMT = {passMTName: True}
failMT = {passMTName: False}

axis_passIso = hist.axis.Boolean(name=passIsoName)
axis_passMT = hist.axis.Boolean(name=passMTName)

# axes with only a few bins for beyond simple ABCD methods
axis_isoCat = hist.axis.Variable([0, 4, 8], name="iso", underflow=False, overflow=True)
axis_relIsoCat = hist.axis.Variable(
    [0, 0.15, 0.3], name="relIso", underflow=False, overflow=True
)


def get_binning_fakes_pt(min_pt, max_pt):
    edges = np.arange(min_pt, 32, 1)
    edges = np.append(
        edges, [e for e in [33, 35, 38, 41, 44, 47, 50, 53, 56] if e < max_pt][:-1]
    )
    edges = np.append(edges, [max_pt])
    ## the following lines are used to replace the previous ones when studying different pT binning and the MC stat
    # edges = np.arange(min_pt,32,1)
    # edges = np.append(edges, [e for e in [33,36,40,46,56] if e<max_pt][:-1])
    # edges = np.append(edges, [max_pt])
    # edges = np.arange(min_pt,32,1)
    # edges = np.append(edges, [e for e in [33,36,40,46,56] if e<max_pt][:-1])
    # edges = np.append(edges, [max_pt])
    # edges = np.arange(min_pt,32.1,1.2)
    # edges = np.append(edges, [e for e in [34.4, 38, 44, 56] if e<max_pt][:-1])
    # edges = np.append(edges, [max_pt])
    # edges = np.arange(min_pt,32,2)
    # edges = np.append(edges, [e for e in [32, 36, 40, 46, 56] if e<max_pt][:-1])
    # edges = np.append(edges, [max_pt])
    # edges = np.arange(min_pt, max_pt, 3)
    # edges = np.append(edges, [max_pt])

    return edges


def get_binning_fakes_mt(mt_cut=40, high_mt_bins=False, fine_mt_binning=False):
    edges = np.array([0, int(mt_cut / 2.0), mt_cut])
    if high_mt_bins:
        # needed for extended 2D method
        edges = np.append(
            edges, [e for e in [30, 32, 34, 36, 38, 40, 44, 49, 55, 62] if e > mt_cut]
        )
    if fine_mt_binning:
        end = 120
        step = 2
        edges = np.append(
            edges, np.linspace(mt_cut + step, end, int((end - mt_cut) / step))
        )
    return edges


def get_binning_fakes_relIso(high_iso_bins=False):
    edges = [0, 0.15]
    if high_iso_bins:
        # needed for extended 2D method
        edges.append(0.3)
    return edges


def get_default_ptbins(analysis_label, unfolding=False, gen=False):
    vals = [30, 26.0, 56.0] if analysis_label[0] == "w" else [34, 26.0, 60.0]
    if unfolding and gen:
        raise ValueError(
            "Inconsistent arguments for 'unfolding' and 'gen.' Must be unique"
        )

    if unfolding:
        vals[0] += 2
        vals[2] += 2
    elif gen:
        vals[0] -= 2
        vals[1] += 2
    return vals


def get_default_etabins(analysis_label=None):
    return (48, -2.4, 2.4)


def get_default_mtcut(analysis_label=None):
    return 40.0 if analysis_label[0] == "w" else 45.0


def get_default_mz_window():
    return 60, 120


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


def getIsoMtRegionID(passIso=True, passMT=True):
    return passIso * 1 + passMT * 2


def getIsoMtRegionFromID(regionID):
    return {passIsoName: regionID & 1, passMTName: regionID & 2}


def natural_sort_key(s):
    # Sort string in a number aware way by plitting the string into alphabetic and numeric parts
    parts = re.split(r"(\d+)", s)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def natural_sort(strings):
    return sorted(strings, key=natural_sort_key)


def natural_sort_dict(dictionary):
    sorted_keys = natural_sort(dictionary.keys())
    sorted_dict = {key: dictionary[key] for key in sorted_keys}
    return sorted_dict


"""
INPUT -------------------------------------------------------------------------
|* (str) string: the string to be converted to list
|
ROUTINE -----------------------------------------------------------------------
|* converts a string to a string element in a list
|  - if not comma-separated, then the whole string becomes one single element
OUTPUT ------------------------------------------------------------------------
|* (float) string: the list-lized string
+------------------------------------------------------------------------------
"""


def string_to_list(string):
    if type(string) == str:
        string = string.split(",")  # items have to be comma-separated
        return string
    elif type(string) == list:
        return string
    else:
        raise TypeError(
            "string_to_list(): cannot convert an input that is"
            "neither a single string nor a list of strings to a list"
        )


"""
INPUT -------------------------------------------------------------------------
|* list(str): a list of strings
|
ROUTINE -----------------------------------------------------------------------
|* convert the list of string to a single string by join()
|
OUTPUT ------------------------------------------------------------------------
|* (str): the resulted string
+------------------------------------------------------------------------------
"""


def list_to_string(list_str):
    if type(list_str) == str:
        return list_str
    elif type(list_str) == list:
        string = ""
        return string.join(list_str)
    else:
        raise TypeError(
            "list_to_string(): cannot convert an input that is"
            " neither a single string or a list of strings"
        )
