# cross sections from: https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV
BR_W_LEP = 3 * 0.1086  # PDG

# TODO update xsecs to 13.6 TeV?

dataDict = {
    "Zmumu": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/DYJetsToMuMu_H2ErratumFix_TuneCP5_13p6TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": 2025.74,
        "group": "Zmumu",
    },
    "Wplusmunu": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/WplusJetsToMuNu_H2ErratumFix_TuneCP5_13p6TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": 11572.19,
        "group": "Wmunu",
    },
    "Wminusmunu": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/WminusJetsToMuNu_H2ErratumFix_TuneCP5_13p6TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": 8677.3,
        "group": "Wmunu",
    },
    "Ztautau": {
        "filepaths": [
            "{BASE_PATH}/2017H/NanoAOD_v3/DYJetsToTauTau_M-50_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": 2025.74,
        "group": "Ztautau",
    },
    "Wplustaunu": {
        "filepaths": [
            "{BASE_PATH}/2017H/NanoAOD_v3/WplusJetsToTauNu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": 11572.19,
        "group": "Wtaunu",
    },
    "Wminustaunu": {
        "filepaths": [
            "{BASE_PATH}/2017H/NanoAOD_v3/WminusJetsToTauNu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": 8677.3,
        "group": "Wtaunu",
    },
    "WWTo2L2Nu": {
        "filepaths": [
            "{BASE_PATH}/2017H/NanoAOD_v2/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8"
        ],
        "xsec": 118.7 * BR_W_LEP * BR_W_LEP,
        "group": "Diboson",
    },
    "WZTo3LNu": {
        "filepaths": [
            "{BASE_PATH}/2017H/NanoAOD_v2/WZTo3LNu_TuneCP5_13TeV-powheg-pythia8"
        ],
        "xsec": 4.912,
        "group": "Diboson",
    },
    "ZZ": {
        "filepaths": ["{BASE_PATH}/2017H/NanoAOD_v2/ZZ_TuneCP5_13TeV-pythia8"],
        "xsec": 16.523,
        "group": "Diboson",
    },
    "TTTo2L2Nu": {
        "filepaths": [
            "{BASE_PATH}/2017H/NanoAOD_v2/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8"
        ],
        "xsec": 87.31483776,
        "group": "Top",
    },
    "TTToSemiLeptonic": {
        "filepaths": [
            "{BASE_PATH}/2017H/NanoAOD_v2/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8"
        ],
        "xsec": 364.35,
        "group": "Top",
    },
}
