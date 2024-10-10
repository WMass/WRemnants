from utilities import common

lumijson = f"{common.data_dir}/lowPU/Cert_306896-307082_13TeV_PromptReco_Collisions17_JSON_LowPU_lowPU_suppressedHighPULS.txt"
lumicsv_mu = f"{common.data_dir}/lowPU/bylsoutput_HLT_HIMu17_Full.csv"
lumicsv_el = f"{common.data_dir}/lowPU/bylsoutput_HLT_HIEle20_Full.csv"

# cross sections from: https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV
BR_W_LEP = 3 * 0.1086  # PDG

dataDictLowPU = {
    "singleelectron": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/NanoAOD_v2/HighEGJet",
        ],
        "group": "Data",
        "lumicsv": lumicsv_el,
        "lumijson": lumijson,
    },
    "singlemuon": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/NanoAOD_v2/SingleMuon",
        ],
        "group": "Data",
        "lumicsv": lumicsv_mu,
        "lumijson": lumijson,
    },
    "Zee": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/NanoAOD_v3/DYJetsToEE_M-50_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": 2025.74,
        "group": "Zee",
    },
    "Wplusenu": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/NanoAOD_v3/WplusJetsToENu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": 11572.19,
        "group": "Wenu",
    },
    "Wminusenu": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/NanoAOD_v3/WminusJetsToENu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": 8677.3,
        "group": "Wenu",
    },
    "Zmumu": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/NanoAOD_v3/DYJetsToMuMu_M-50_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": 2025.74,
        "group": "Zmumu",
    },
    "Wplusmunu": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/NanoAOD_v3/WplusJetsToMuNu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": 11572.19,
        "group": "Wmunu",
    },
    "Wminusmunu": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/NanoAOD_v3/WminusJetsToMuNu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": 8677.3,
        "group": "Wmunu",
    },
    "Ztautau": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/NanoAOD_v3/DYJetsToTauTau_M-50_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": 2025.74,
        "group": "Ztautau",
    },
    "Wplustaunu": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/NanoAOD_v3/WplusJetsToTauNu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": 11572.19,
        "group": "Wtaunu",
    },
    "Wminustaunu": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/NanoAOD_v3/WminusJetsToTauNu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": 8677.3,
        "group": "Wtaunu",
    },
    "WWTo2L2Nu": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/NanoAOD_v2/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8"
        ],
        "xsec": 118.7 * BR_W_LEP * BR_W_LEP,
        "group": "Diboson",
    },
    "WZTo3LNu": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/NanoAOD_v2/WZTo3LNu_TuneCP5_13TeV-powheg-pythia8"
        ],
        "xsec": 4.912,
        "group": "Diboson",
    },
    "ZZ": {
        "filepaths": ["{BASE_PATH}/{ERA}/NanoAOD_v2/ZZ_TuneCP5_13TeV-pythia8"],
        "xsec": 16.523,
        "group": "Diboson",
    },
    "TTTo2L2Nu": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/NanoAOD_v2/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8"
        ],
        "xsec": 87.31483776,
        "group": "Top",
    },
    "TTToSemiLeptonic": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/NanoAOD_v2/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8"
        ],
        "xsec": 364.35,
        "group": "Top",
    },
}
