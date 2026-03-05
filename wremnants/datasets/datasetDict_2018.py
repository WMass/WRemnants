import copy

from utilities import common

lumicsv = f"{common.data_dir}/bylsoutput_2018.csv"
lumijson = f"{common.data_dir}/Cert_314472-325175_13TeV_UL2018_Collisions18_HLT_IsoMu24_v_CustomJSON.txt"

dataDict = {
    "SingleMuon_2018A": {
        "filepaths": [
            "{BASE_PATH}/SingleMuon/NanoV9Run2018A_{NANO_PROD_TAG}",
        ],
        "group": "Data",
        "lumicsv": lumicsv,
        "lumijson": lumijson,
        "das_name": "private",
    },
    "SingleMuon_2018B": {
        "filepaths": [
            "{BASE_PATH}/SingleMuon/NanoV9Run2018B_{NANO_PROD_TAG}",
        ],
        "group": "Data",
        "lumicsv": lumicsv,
        "lumijson": lumijson,
        "das_name": "private",
    },
    "SingleMuon_2018C": {
        "filepaths": [
            "{BASE_PATH}/SingleMuon/NanoV9Run2018C_{NANO_PROD_TAG}",
        ],
        "group": "Data",
        "lumicsv": lumicsv,
        "lumijson": lumijson,
        "das_name": "private",
    },
    "SingleMuon_2018D": {
        "filepaths": [
            "{BASE_PATH}/SingleMuon/NanoV9Run2018D_{NANO_PROD_TAG}",
        ],
        "group": "Data",
        "lumicsv": lumicsv,
        "lumijson": lumijson,
        "das_name": "private",
    },
    "Zmumu_2018": {
        "filepaths": [
            "{BASE_PATH}/DYJetsToMuMu_H2ErratumFix_PDFExt_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MC2018_{NANO_PROD_TAG}",
        ],
        "xsec": common.xsec_DYJetsToLL,
        "group": "Zmumu",
        "das_name": "private",
    },
    "DYJetsToMuMuMass10to50_2018": {
        "filepaths": [
            "{BASE_PATH}/DYJetsToMuMu_M-10to50_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": common.xsec_DYJetsToLLMass10to50,
        "group": "DYlowMass",
        "das_name": "/DYJetsToMuMu_M-10to50_H2ErratumFix_PDFExt_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1*v1/NANOAODSIM",
    },
    "Ztautau_2018": {
        "filepaths": [
            "{BASE_PATH}/DYJetsToTauTau_M-50_AtLeastOneEorMuDecay_H2ErratumFix_PDF_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        # At least one tau->e or mu decay, so everything that's not all other decays
        "xsec": common.xsec_DYJetsToLL * common.Z_TAU_TO_LEP_RATIO,
        "group": "Ztautau",
        "das_name": "/DYJetsToTauTau_M-50_AtLeastOneEorMuDecay_H2ErratumFix_PDF_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
    },
    "Wplusmunu_2018": {
        "filepaths": [
            "{BASE_PATH}/WplusJetsToMuNu_H2ErratumFix_PDFExt_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MC2018_{NANO_PROD_TAG}",
        ],
        "xsec": common.xsec_WplusJetsToLNu,
        "group": "Wmunu",
        "das_name": "private",
    },
    "Wminusmunu_2018": {
        "filepaths": [
            "{BASE_PATH}/WminusJetsToMuNu_H2ErratumFix_PDFExt_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MC2018_{NANO_PROD_TAG}",
        ],
        "xsec": common.xsec_WminusJetsToLNu,
        "group": "Wmunu",
        "das_name": "private",
    },
    "Wplustaunu_2018": {
        "filepaths": [
            "{BASE_PATH}/WplusJetsToTauNu_TauToMu_H2ErratumFix_PDFExt_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MC2018_{NANO_PROD_TAG}",
        ],
        "xsec": common.BR_TAUToMU * common.xsec_WplusJetsToLNu,
        "group": "Wtaunu",
        "das_name": "private",
    },
    "Wminustaunu_2018": {
        "filepaths": [
            "{BASE_PATH}/WminusJetsToTauNu_TauToMu_H2ErratumFix_PDFExt_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MC2018_{NANO_PROD_TAG}",
        ],
        "xsec": common.BR_TAUToMU * common.xsec_WminusJetsToLNu,
        "group": "Wtaunu",
        "das_name": "private",
    },
    "TTLeptonic_2018": {
        "filepaths": [
            "{BASE_PATH}/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": 88.29,
        "group": "Top",
        "das_name": "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
    },
    "TTSemileptonic_2018": {
        "filepaths": [
            "{BASE_PATH}/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": 366.34,
        "group": "Top",
        "das_name": "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
    },
    "SingleTschanLepDecays_2018": {
        "filepaths": [
            "{BASE_PATH}/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": 3.609,
        "group": "Top",
        "das_name": "/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
    },
    "SingleTtWAntitop_2018": {
        "filepaths": [
            "{BASE_PATH}/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": 19.55,  # 35.85 * (1.0-((1-0.1086*3)*(1-0.1086*3))) = 19.5 pb
        "group": "Top",
        "das_name": "/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
    },
    "SingleTtWTop_2018": {
        "filepaths": [
            "{BASE_PATH}/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": 19.55,
        "group": "Top",
        "das_name": "/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
    },
    "SingleTtchanAntitop_2018": {
        "filepaths": [
            "{BASE_PATH}/ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": 80.0,
        "group": "Top",
        "das_name": "/ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
    },
    "SingleTtchanTop_2018": {
        "filepaths": [
            "{BASE_PATH}/ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": 134.2,
        "group": "Top",
        "das_name": "/ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
    },
    # inclusive samples, keep for reference
    # 'WW2018' : {
    #                 'filepaths' :
    #                 ["{BASE_PATH}/WW_TuneCP5_13TeV-pythia8/*.root/NanoV9MC2018_{NANO_PROD_TAG}"],
    #                 'xsec' : 118.7,
    #                 'group' : "Diboson",
    # },
    # 'WZ2018' : {
    #                 'filepaths' :
    #                 ["{BASE_PATH}/WZ_TuneCP5_13TeV-pythia8/*.root/NanoV9MC2018_{NANO_PROD_TAG}"],
    #                 'xsec' : 47.026760,  # to check, taken from WZTo1L1Nu2Q dividing by BR: 10.71/(3*0.1086)/(1-3*0.033658-0.2)
    #                 'group' : "Diboson",
    # },
    ##
    "WWTo2L2Nu_2018": {
        "filepaths": [
            "{BASE_PATH}/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": common.xsec_WWTo2L2Nu,
        "group": "Diboson",
        "das_name": "/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
    },
    "WWTo1L1Nu2Q_2018": {
        "filepaths": [
            "{BASE_PATH}/WWTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": common.xsec_WWTo1L1Nu,
        "group": "Diboson",
        "das_name": "/WWTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
    },
    "WZTo3LNu_2018": {
        "filepaths": [
            "{BASE_PATH}/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": common.xsec_WZTo3LNu,
        "group": "Diboson",
        "das_name": "/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
    },
    "WZTo2Q2L_2018": {
        "filepaths": [
            "{BASE_PATH}/WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": common.xsec_WZTo2Q2L,
        "group": "Diboson",
        "das_name": "/WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
    },
    "WZTo1L1Nu2Q2Q_2018": {
        "filepaths": [
            "{BASE_PATH}/WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": common.xsec_WZTo1L1Nu2Q,
        "group": "Diboson",
        "das_name": "/WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
    },
    "ZZTo2L2Nu_2018": {
        "filepaths": [
            "{BASE_PATH}/ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": common.xsec_ZZTo2L2Nu,
        "group": "Diboson",
        "das_name": "/ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
    },
    "ZZTo2Q2L_2018": {
        "filepaths": [
            "{BASE_PATH}/ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": common.xsec_ZZTo2Q2L,
        "group": "Diboson",
        "das_name": "/ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
    },
    "QCDmuEnrichPt15_2018": {
        "filepaths": [
            "{BASE_PATH}/QCD_Pt-20_MuEnrichedPt15_TuneCP5_13TeV-pythia8/NanoV9MC2018_{NANO_PROD_TAG}/"
        ],
        "xsec": 238800,
        "group": "QCD",
        "das_name": "/QCD_Pt-20_MuEnrichedPt15_TuneCP5_13TeV-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
    },
}

# extended version with additional samples (but missing some pdf sets)
dataDict_extended = copy.deepcopy(dataDict)

dataDict_extended["Zmumu_2018"]["filepaths"].extend(
    [
        "{BASE_PATH}/DYJetsToMuMu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MCPostVFP_{NANO_PROD_TAG}",
    ]
)

dataDict_extended["Ztautau_2018"]["filepaths"].extend(
    [
        "{BASE_PATH}/DYJetsToTauTau_M-50_AtLeastOneEorMuDecay_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MCPostVFP_{NANO_PROD_TAG}",
    ]
)

dataDict_extended["Wplusmunu_2018"]["filepaths"].extend(
    [
        "{BASE_PATH}/WplusJetsToMuNu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MCPostVFP_{NANO_PROD_TAG}",
    ]
)

dataDict_extended["Wminusmunu_2018"]["filepaths"].extend(
    [
        "{BASE_PATH}/WminusJetsToMuNu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MCPostVFP_{NANO_PROD_TAG}",
    ]
)

dataDict_extended["Wplustaunu_2018"]["filepaths"].extend(
    [
        "{BASE_PATH}/WplusJetsToTauNu_TauToMu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MCPostVFP_{NANO_PROD_TAG}",
    ]
)

dataDict_extended["Wminustaunu_2018"]["filepaths"].extend(
    [
        "{BASE_PATH}/WminusJetsToTauNu_TauToMu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MCPostVFP_{NANO_PROD_TAG}",
    ]
)
