from utilities import common

lumijson = f"{common.data_dir}/lowPU/Cert_306896-307082_13TeV_PromptReco_Collisions17_JSON_LowPU_lowPU_suppressedHighPULS.txt"
lumicsv_mu = f"{common.data_dir}/lowPU/bylsoutput_HLT_HIMu17_Full.csv"
lumicsv_el = f"{common.data_dir}/lowPU/bylsoutput_HLT_HIEle20_Full.csv"

dataDict = {
    "HighEGJet_2017H": {
        "filepaths": [
            "{BASE_PATH}/LowPU/NanoAOD_v2/HighEGJet",
        ],
        "group": "Data",
        "lumicsv": lumicsv_el,
        "lumijson": lumijson,
    },
    "SingleMuon_2017H": {
        "filepaths": [
            "{BASE_PATH}/LowPU/NanoAOD_v2/SingleMuon",
        ],
        "group": "Data",
        "lumicsv": lumicsv_mu,
        "lumijson": lumijson,
    },
    "Zee_2017H": {
        "filepaths": [
            "{BASE_PATH}/LowPU/NanoAOD_v3/DYJetsToEE_M-50_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": common.xsec_DYJetsToLL,
        "group": "Zee",
    },
    "Wplusenu_2017H": {
        "filepaths": [
            "{BASE_PATH}/LowPU/NanoAOD_v3/WplusJetsToENu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": common.xsec_WplusJetsToLNu,
        "group": "Wenu",
    },
    "Wminusenu_2017H": {
        "filepaths": [
            "{BASE_PATH}/LowPU/NanoAOD_v3/WminusJetsToENu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": common.xsec_WminusJetsToLNu,
        "group": "Wenu",
    },
    "Zmumu_2017H": {
        "filepaths": [
            "{BASE_PATH}/LowPU/NanoAOD_v3/DYJetsToMuMu_M-50_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": common.xsec_DYJetsToLL,
        "group": "Zmumu",
    },
    "Wplusmunu_2017H": {
        "filepaths": [
            "{BASE_PATH}/LowPU/NanoAOD_v3/WplusJetsToMuNu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": common.xsec_WplusJetsToLNu,
        "group": "Wmunu",
    },
    "Wminusmunu_2017H": {
        "filepaths": [
            "{BASE_PATH}/LowPU/NanoAOD_v3/WminusJetsToMuNu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": common.xsec_WminusJetsToLNu,
        "group": "Wmunu",
    },
    "Ztautau_2017H": {
        "filepaths": [
            "{BASE_PATH}/LowPU/NanoAOD_v3/DYJetsToTauTau_M-50_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": common.xsec_DYJetsToLL,
        "group": "Ztautau",
    },
    "Wplustaunu_2017H": {
        "filepaths": [
            "{BASE_PATH}/LowPU/NanoAOD_v3/WplusJetsToTauNu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": common.xsec_WplusJetsToLNu,
        "group": "Wtaunu",
    },
    "Wminustaunu_2017H": {
        "filepaths": [
            "{BASE_PATH}/LowPU/NanoAOD_v3/WminusJetsToTauNu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": common.xsec_WminusJetsToLNu,
        "group": "Wtaunu",
    },
    "WWTo2L2Nu_2017H": {
        "filepaths": [
            "{BASE_PATH}/LowPU/NanoAOD_v2/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8"
        ],
        "xsec": 118.7 * common.BR_W_LEP * common.BR_W_LEP,
        "group": "Diboson",
    },
    "WZTo3LNu_2017H": {
        "filepaths": [
            "{BASE_PATH}/LowPU/NanoAOD_v2/WZTo3LNu_TuneCP5_13TeV-powheg-pythia8"
        ],
        "xsec": 4.912,
        "group": "Diboson",
    },
    "ZZ_2017H": {
        "filepaths": ["{BASE_PATH}/LowPU/NanoAOD_v2/ZZ_TuneCP5_13TeV-pythia8"],
        "xsec": 16.523,
        "group": "Diboson",
    },
    "TTTo2L2Nu_2017H": {
        "filepaths": [
            "{BASE_PATH}/LowPU/NanoAOD_v2/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8"
        ],
        "xsec": 87.31483776,
        "group": "Top",
    },
    "TTToSemiLeptonic_2017H": {
        "filepaths": [
            "{BASE_PATH}/LowPU/NanoAOD_v2/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8"
        ],
        "xsec": 364.35,
        "group": "Top",
    },
}
