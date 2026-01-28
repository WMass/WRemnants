from utilities import common

lumicsv = f"{common.data_dir}/bylsoutput_2017G.csv"
lumijson = (
    f"{common.data_dir}/Cert_306546-306826_5TeV_EOY2017ReReco_Collisions17_JSON.txt"
)

# from GenXSecAnalyzer
xsec_DYJetsToLL = 698.3  # +/- 2.133
xsec_WplusJetsToMuNu = 4477  # +/- 17.27
xsec_WminusJetsToMuNu = 2940  # +/- 9.153

dataDictV9_2017G = {
    "data2017G": {
        "filepaths": [
            "{BASE_PATH}/SingleMuon/Run2017G-UL2017_MiniAODv2_NanoAODv9_GT36-v2",
        ],
        "group": "Data",
        "lumicsv": lumicsv,
        "lumijson": lumijson,
    },
    "Zmumu5020GeV": {
        "filepaths": [
            "{BASE_PATH}/DYJetsToMuMu_H2ErratumFix_PDFExt_TuneCP5_5020GeV-powhegMiNNLO-pythia8-photos/RunIISummer20UL17pp5TeVNanoAODv9-106X_mc2017_realistic_forppRef5TeV_v3-v2",
        ],
        "xsec": xsec_DYJetsToLL,
        "group": "Zmumu",
    },
    "Ztautau5020GeV": {
        "filepaths": [
            "{BASE_PATH}/DYJetsToTauTau_TauToMuorE_H2ErratumFix_PDFExt_TuneCP5_5020GeV-powhegMiNNLO-pythia8-photos/RunIISummer20UL17pp5TeVNanoAODv9-106X_mc2017_realistic_forppRef5TeV_v3-v2",
        ],
        "xsec": xsec_DYJetsToLL * common.Z_TAU_TO_LEP_RATIO,
        "group": "Ztautau",
    },
    "Wplusmunu5020GeV": {
        "filepaths": [
            "{BASE_PATH}/WplusJetsToMuNu_H2ErratumFix_PDFExt_TuneCP5_5020GeV-powhegMiNNLO-pythia8-photos/RunIISummer20UL17pp5TeVNanoAODv9-106X_mc2017_realistic_forppRef5TeV_v3-v2",
        ],
        "xsec": xsec_WplusJetsToMuNu,
        "group": "Wmunu",
    },
    "Wminusmunu5020GeV": {
        "filepaths": [
            "{BASE_PATH}/WminusJetsToMuNu_H2ErratumFix_PDFExt_TuneCP5_5020GeV-powhegMiNNLO-pythia8-photos/RunIISummer20UL17pp5TeVNanoAODv9-106X_mc2017_realistic_forppRef5TeV_v3-v2",
        ],
        "xsec": xsec_WminusJetsToMuNu,
        "group": "Wmunu",
    },
    # "Wplustaunu5020GeV": {
    #     "filepaths": [
    #         "{BASE_PATH}/WplusJetsToTauNu_TauToMuorE_H2ErratumFix_PDFExt_TuneCP5_5020GeV-powhegMiNNLO-pythia8-photos/RunIISummer20UL17pp5TeVNanoAODv9-106X_mc2017_realistic_forppRef5TeV_v3-v2",
    #     ],
    #     "xsec": common.BR_TAUToMU * xsec_WplusJetsToMuNu,
    #     "group": "Wtaunu",
    # },
    "Wminustaunu5020GeV": {
        "filepaths": [
            "{BASE_PATH}/WminusJetsToTauNu_TauToMuorE_H2ErratumFix_PDFExt_TuneCP5_5020GeV-powhegMiNNLO-pythia8-photos/RunIISummer20UL17pp5TeVNanoAODv9-106X_mc2017_realistic_forppRef5TeV_v3-v2",
        ],
        "xsec": common.BR_TAUToMU * xsec_WminusJetsToMuNu,
        "group": "Wtaunu",
    },
}
