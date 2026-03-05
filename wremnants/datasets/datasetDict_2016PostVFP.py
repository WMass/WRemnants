import copy

from utilities import common

lumicsv = f"{common.data_dir}/bylsoutput.csv"
lumijson = (
    f"{common.data_dir}/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"
)

dataDict = {
    "SingleMuon_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/SingleMuon/NanoV9Run2016FDataPostVFP_{NANO_PROD_TAG}",
            "{BASE_PATH}/SingleMuon/NanoV9Run2016GDataPostVFP_{NANO_PROD_TAG}",
            "{BASE_PATH}/SingleMuon/NanoV9Run2016HDataPostVFP_{NANO_PROD_TAG}",
        ],
        "group": "Data",
        "lumicsv": lumicsv,
        "lumijson": lumijson,
    },
    "Zmumu_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/DYJetsToMuMu_H2ErratumFix_PDFExt_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MCPostVFP_{NANO_PROD_TAG}",
        ],
        "xsec": common.xsec_DYJetsToLL,
        "group": "Zmumu",
    },
    "DYJetsToMuMuMass10to50_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/DYJetsToMuMu_M-10to50_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MCPostVFP_{NANO_PROD_TAG}",
        ],
        "xsec": common.xsec_DYJetsToLLMass10to50,
        "group": "DYlowMass",
    },
    "Ztautau_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/DYJetsToTauTau_M-50_AtLeastOneEorMuDecay_H2ErratumFix_PDF_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MCPostVFP_{NANO_PROD_TAG}",
        ],
        # At least one tau->e or mu decay, so everything that's not all other decays
        "xsec": common.xsec_DYJetsToLL * common.Z_TAU_TO_LEP_RATIO,
        "group": "Ztautau",
    },
    "Wplusmunu_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/WplusJetsToMuNu_H2ErratumFix_PDFExt_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MCPostVFP_{NANO_PROD_TAG}",
        ],
        "xsec": common.xsec_WplusJetsToLNu,
        "group": "Wmunu",
    },
    "Wminusmunu_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/WminusJetsToMuNu_H2ErratumFix_PDFExt_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MCPostVFP_{NANO_PROD_TAG}",
        ],
        "xsec": common.xsec_WminusJetsToLNu,
        "group": "Wmunu",
    },
    "Wplustaunu_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/WplusJetsToTauNu_TauToMu_H2ErratumFix_PDFExt_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MCPostVFP_{NANO_PROD_TAG}",
        ],
        "xsec": common.BR_TAUToMU * common.xsec_WplusJetsToLNu,
        "group": "Wtaunu",
    },
    "Wminustaunu_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/WminusJetsToTauNu_TauToMu_H2ErratumFix_PDFExt_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MCPostVFP_{NANO_PROD_TAG}",
        ],
        "xsec": common.BR_TAUToMU * common.xsec_WminusJetsToLNu,
        "group": "Wtaunu",
    },
    "TTLeptonic_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NanoV9MCPostVFP_{NANO_PROD_TAG}"
        ],
        "xsec": 88.29,
        "group": "Top",
    },
    "TTSemileptonic_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NanoV9MCPostVFP_{NANO_PROD_TAG}"
        ],
        "xsec": 366.34,
        "group": "Top",
    },
    "SingleTschanLepDecays_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8/NanoV9MCPostVFP_{NANO_PROD_TAG}"
        ],
        "xsec": 3.609,
        "group": "Top",
    },
    "SingleTtWAntitop_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/NanoV9MCPostVFP_{NANO_PROD_TAG}"
        ],
        "xsec": 19.55,  # 35.85 * (1.0-((1-0.1086*3)*(1-0.1086*3))) = 19.5 pb
        "group": "Top",
    },
    "SingleTtWTop_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/NanoV9MCPostVFP_{NANO_PROD_TAG}"
        ],
        "xsec": 19.55,
        "group": "Top",
    },
    "SingleTtchanAntitop_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8/NanoV9MCPostVFP_{NANO_PROD_TAG}"
        ],
        "xsec": 80.0,
        "group": "Top",
    },
    "SingleTtchanTop_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8/NanoV9MCPostVFP_{NANO_PROD_TAG}"
        ],
        "xsec": 134.2,
        "group": "Top",
    },
    # inclusive samples, keep for reference
    # 'WWPostVFP' : {
    #                 'filepaths' :
    #                 ["{BASE_PATH}/BKGV9/WW_TuneCP5_13TeV-pythia8"],
    #                 'xsec' : 118.7,
    #                 'group' : "Diboson",
    # },
    # 'WZPostVFP' : {
    #                 'filepaths' :
    #                 ["{BASE_PATH}/BKGV9/WZ_TuneCP5_13TeV-pythia8"],
    #                 'xsec' : 47.026760,  # to check, taken from WZTo1L1Nu2Q dividing by BR: 10.71/(3*0.1086)/(1-3*0.033658-0.2)
    #                 'group' : "Diboson",
    # },
    ##
    "WWTo2L2Nu_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NanoV9MCPostVFP_{NANO_PROD_TAG}"
        ],
        "xsec": common.xsec_WWTo2L2Nu,
        "group": "Diboson",
    },
    "WWTo1L1Nu2Q_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/WWTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/NanoV9MCPostVFP_{NANO_PROD_TAG}"
        ],
        "xsec": common.xsec_WWTo1L1Nu,
        "group": "Diboson",
    },
    "WZTo3LNu_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/NanoV9MCPostVFP_{NANO_PROD_TAG}"
        ],
        "xsec": common.xsec_WZTo3LNu,
        "group": "Diboson",
    },
    "WZTo2Q2L_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/NanoV9MCPostVFP_{NANO_PROD_TAG}"
        ],
        "xsec": common.xsec_WZTo2Q2L,
        "group": "Diboson",
    },
    "WZTo1L1Nu2Q_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/NanoV9MCPostVFP_{NANO_PROD_TAG}"
        ],
        "xsec": common.xsec_WZTo1L1Nu2Q,
        "group": "Diboson",
    },
    "ZZTo2L2Nu_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8/NanoV9MCPostVFP_{NANO_PROD_TAG}"
        ],
        "xsec": common.xsec_ZZTo2L2Nu,
        "group": "Diboson",
    },
    "ZZTo2Q2L_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/NanoV9MCPostVFP_{NANO_PROD_TAG}"
        ],
        "xsec": common.xsec_ZZTo2Q2L,
        "group": "Diboson",
    },
    "QCDmuEnrichPt15_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/QCD_Pt-20_MuEnrichedPt15_TuneCP5_13TeV-pythia8/NanoV9MCPostVFP_{NANO_PROD_TAG}"
        ],
        "xsec": 238800,
        "group": "QCD",
    },
    "GGToMuMuMass5to50_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/GGToMuMu_M-5To50_TuneCP5_13TeV-pythia8/NanoV9MCPostVFP_{NANO_PROD_TAG}",
        ],
        "xsec": common.xsec_GGtoMuMu,
        "group": "PhotonInduced",
    },
    "GGToLL_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/GGToLL_TuneCP5_13TeV-pythia8/NanoV9MCPostVFP_{NANO_PROD_TAG}",
        ],
        "xsec": 14.93,
        "group": "PhotonInduced",
    },
    "QGToDYQTo2L_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/QGToDYQTo2L_TuneCP5_13TeV-pythia8-photos/NanoV9MCPostVFP_{NANO_PROD_TAG}",
        ],
        "xsec": 1.373,
        "group": "PhotonInduced",
    },
    "QGToWQToLNu_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/QGToWQToLNu_TuneCP5_13TeV-pythia8-photos/NanoV9MCPostVFP_{NANO_PROD_TAG}",
        ],
        "xsec": 4.224e01,
        "xsec_up": 4.827e01,
        "xsec_dn": 3.588e01,
        "group": "PhotonInduced",
    },
    "WtoNMuMass5_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/WtoNMu_MN-5-V-0p001_TuneCP5_13TeV_madgraph-pythia8/NanoV9MCPostVFP_{NANO_PROD_TAG}",
        ],
        "xsec": common.xsec_WtoNMu,
        "group": "WtoNMu",
        "auxiliary": True,
    },
    "WtoNMuMass10_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/WtoNMu_MN-10-V-0p001_TuneCP5_13TeV_madgraph-pythia8/NanoV9MCPostVFP_{NANO_PROD_TAG}",
        ],
        "xsec": common.xsec_WtoNMu,
        "group": "WtoNMu",
        "auxiliary": True,
    },
    "WtoNMuMass30_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/WtoNMu_MN-30-V-0p001_TuneCP5_13TeV_madgraph-pythia8/NanoV9MCPostVFP_{NANO_PROD_TAG}",
        ],
        "xsec": common.xsec_WtoNMu,
        "group": "WtoNMu",
        "auxiliary": True,
    },
    "WtoNMuMass50_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/WtoNMu_MN-50-V-0p001_TuneCP5_13TeV_madgraph-pythia8/NanoV9MCPostVFP_{NANO_PROD_TAG}",
        ],
        "xsec": common.xsec_WtoNMu,
        "group": "WtoNMu",
        "auxiliary": True,
    },
    "WtoMuNuSMEFT_2016PostVFP": {
        "filepaths": [
            "{BASE_PATH}/WtoMuNu_nuSMEFT_MNu-0p1_Lambda-246_TuneCP5_13TeV_madgraph-pythia8/NanoV9MCPostVFP_{NANO_PROD_TAG}",
        ],
        "xsec": common.xsec_WtoNMu,
        "group": "WtoNMu",
        "auxiliary": True,
    },
}

# extended version with additional samples (but missing some pdf sets)
dataDict_extended = copy.deepcopy(dataDict)

dataDict_extended["Zmumu_2016PostVFP"]["filepaths"].extend(
    [
        "{BASE_PATH}/DYJetsToMuMu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MCPostVFP_{NANO_PROD_TAG}",
    ]
)

dataDict_extended["Ztautau_2016PostVFP"]["filepaths"].extend(
    [
        "{BASE_PATH}/DYJetsToTauTau_M-50_AtLeastOneEorMuDecay_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MCPostVFP_{NANO_PROD_TAG}",
    ]
)

dataDict_extended["Wplusmunu_2016PostVFP"]["filepaths"].extend(
    [
        "{BASE_PATH}/WplusJetsToMuNu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MCPostVFP_{NANO_PROD_TAG}",
    ]
)

dataDict_extended["Wminusmunu_2016PostVFP"]["filepaths"].extend(
    [
        "{BASE_PATH}/WminusJetsToMuNu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MCPostVFP_{NANO_PROD_TAG}",
    ]
)

dataDict_extended["Wplustaunu_2016PostVFP"]["filepaths"].extend(
    [
        "{BASE_PATH}/WplusJetsToTauNu_TauToMu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MCPostVFP_{NANO_PROD_TAG}",
    ]
)

dataDict_extended["Wminustaunu_2016PostVFP"]["filepaths"].extend(
    [
        "{BASE_PATH}/WminusJetsToTauNu_TauToMu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MCPostVFP_{NANO_PROD_TAG}",
    ]
)
