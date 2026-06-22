from wremnants.utilities import common

# TODO: update lumi JSON and CSV paths once 2026 low-PU run files are available
lumijson = f"{common.data_dir}/lowPU/TODO_Cert_2026_LowPU_JSON.txt"
lumicsv_mu = f"{common.data_dir}/lowPU/TODO_bylsoutput_HLT_Mu_2026.csv"
lumicsv_el = f"{common.data_dir}/lowPU/TODO_bylsoutput_HLT_EG_2026.csv"

dataDict = {
    # TODO: update filepaths once dataset locations are known
    "Muon_2026_LowPU": {
        "filepaths": [
            "{BASE_PATH}/TODO/Muon",
        ],
        "group": "Data",
        "lumicsv": lumicsv_mu,
        "lumijson": lumijson,
    },
    "EGamma_2026_LowPU": {
        "filepaths": [
            "{BASE_PATH}/TODO/EGamma",
        ],
        "group": "Data",
        "lumicsv": lumicsv_el,
        "lumijson": lumijson,
    },
    "Zmumu": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/TODO_DYJetsToMuMu",
        ],
        "xsec": common.xsec_DYJetsToLL,
        "group": "Zmumu",
    },
    "Zee": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/TODO_DYJetsToEE",
        ],
        "xsec": common.xsec_DYJetsToLL,
        "group": "Zee",
    },
    "Wplusmunu": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/TODO_WplusJetsToMuNu",
        ],
        "xsec": common.xsec_WplusJetsToLNu,
        "group": "Wmunu",
    },
    "Wminusmunu": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/TODO_WminusJetsToMuNu",
        ],
        "xsec": common.xsec_WminusJetsToLNu,
        "group": "Wmunu",
    },
    "Wplusenu": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/TODO_WplusJetsToENu",
        ],
        "xsec": common.xsec_WplusJetsToLNu,
        "group": "Wenu",
    },
    "Wminusenu": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/TODO_WminusJetsToENu",
        ],
        "xsec": common.xsec_WminusJetsToLNu,
        "group": "Wenu",
    },
    "Ztautau": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/TODO_DYJetsToTauTau",
        ],
        "xsec": common.xsec_DYJetsToLL,
        "group": "Ztautau",
    },
    "Wplustaunu": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/TODO_WplusJetsToTauNu",
        ],
        "xsec": common.xsec_WplusJetsToLNu,
        "group": "Wtaunu",
    },
    "Wminustaunu": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/TODO_WminusJetsToTauNu",
        ],
        "xsec": common.xsec_WminusJetsToLNu,
        "group": "Wtaunu",
    },
    "WWTo2L2Nu": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/TODO_WWTo2L2Nu",
        ],
        "xsec": common.xsec_WWTo2L2Nu,
        "group": "Diboson",
    },
    "WZTo3LNu": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/TODO_WZTo3LNu",
        ],
        "xsec": 4.912,
        "group": "Diboson",
    },
    "ZZ": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/TODO_ZZ",
        ],
        "xsec": common.xsec_ZZ,
        "group": "Diboson",
    },
    "TTTo2L2Nu": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/TODO_TTTo2L2Nu",
        ],
        "xsec": 87.31483776,
        "group": "Top",
    },
    "TTToSemiLeptonic": {
        "filepaths": [
            "{BASE_PATH}/{ERA}/TODO_TTToSemiLeptonic",
        ],
        "xsec": 364.35,
        "group": "Top",
    },
}
