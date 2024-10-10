from utilities import common

lumicsv = f"{common.data_dir}/bylsoutput_2018.csv"
lumijson = f"{common.data_dir}/Cert_314472-325175_13TeV_UL2018_Collisions18_HLT_IsoMu24_v_CustomJSON.txt"

# NOTES
# BASE_PATH is /scratchnvme/wmass/NANOV9/postVFP (so 2018 BASE path is {BASE_PATH}/../y2018/) have to update at some point
# ZtautauPostVFP sample is one available from centrl production, so
dataDictV9_2018 = {
    "dataPostVFP": {
        "filepaths": [
            "{BASE_PATH}/../y2018/SingleMuon/NanoV9Run2018A_{NANO_PROD_TAG}",
            "{BASE_PATH}/../y2018/SingleMuon/NanoV9Run2018B_{NANO_PROD_TAG}",
            "{BASE_PATH}/../y2018/SingleMuon/NanoV9Run2018C_{NANO_PROD_TAG}",
            "{BASE_PATH}/../y2018/SingleMuon/NanoV9Run2018D_{NANO_PROD_TAG}",
        ],
        "group": "Data",
        "lumicsv": lumicsv,
        "lumijson": lumijson,
        "das_name": "private",
    },
    "ZmumuPostVFP": {
        "filepaths": [
            "{BASE_PATH}/../y2018/DYJetsToMuMu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": common.xsec_ZmmPostVFP,
        "group": "Zmumu",
        "das_name": "private",
    },
    "DYJetsToMuMuMass10to50PostVFP": {
        "filepaths": [
            "{BASE_PATH}/../y2018/DYJetsToMuMu_M-10to50_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": common.xsec_ZmmMass10to50PostVFP,
        "group": "DYlowMass",
        "das_name": "/DYJetsToMuMu_M-10to50_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1*v1/NANOAODSIM",
    },
    "ZtautauPostVFP": {  # this sample needs to be produced using old Mass fix one
        "filepaths": [
            "{BASE_PATH}/../y2018//DYJetsToTauTau_M-50_AtLeastOneEorMuDecay_massWgtFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        # At least one tau->e or mu decay, so everything that's not all other decays
        "xsec": common.xsec_ZmmPostVFP * common.Z_TAU_TO_LEP_RATIO,
        "group": "Ztautau",
        "das_name": "/DYJetsToTauTau_M-50_AtLeastOneEorMuDecay_massWgtFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
    },
    "WplusmunuPostVFP": {
        "filepaths": [
            "{BASE_PATH}/../y2018/WplusJetsToMuNu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": common.xsec_WpmunuPostVFP,
        "group": "Wmunu",
        "das_name": "private",
    },
    "WminusmunuPostVFP": {
        "filepaths": [
            "{BASE_PATH}/../y2018/WminusJetsToMuNu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": common.xsec_WmmunuPostVFP,
        "group": "Wmunu",
        "das_name": "private",
    },
    "WplustaunuPostVFP": {
        "filepaths": [
            "{BASE_PATH}/../y2018/WplusJetsToTauNu_TauToMu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MC2018_{NANO_PROD_TAG}",
        ],
        "xsec": common.BR_TAUToMU * common.xsec_WpmunuPostVFP,
        "group": "Wtaunu",
        "das_name": "private",
    },
    "WminustaunuPostVFP": {
        "filepaths": [
            "{BASE_PATH}/../y2018/WminusJetsToTauNu_TauToMu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": common.BR_TAUToMU * common.xsec_WmmunuPostVFP,
        "group": "Wtaunu",
        "das_name": "private",
    },
    "TTLeptonicPostVFP": {
        "filepaths": [
            "{BASE_PATH}/../y2018/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": 88.29,
        "group": "Top",
        "das_name": "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
    },
    "TTSemileptonicPostVFP": {  ##could not copy full stat of this sample due to lack of storage
        "filepaths": [
            "{BASE_PATH}/../y2018/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": 366.34,
        "group": "Top",
        "das_name": "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
    },
    "SingleTschanLepDecaysPostVFP": {
        "filepaths": [
            "{BASE_PATH}/../y2018/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": 3.609,
        "group": "Top",
        "das_name": "/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
    },
    "SingleTtWAntitopPostVFP": {
        "filepaths": [
            "{BASE_PATH}/../y2018/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": 19.55,  # 35.85 * (1.0-((1-0.1086*3)*(1-0.1086*3))) = 19.5 pb
        "group": "Top",
        "das_name": "/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
    },
    "SingleTtWTopPostVFP": {
        "filepaths": [
            "{BASE_PATH}/../y2018/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": 19.55,
        "group": "Top",
        "das_name": "/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
    },
    "SingleTtchanAntitopPostVFP": {
        "filepaths": [
            "{BASE_PATH}/../y2018/ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": 80.0,
        "group": "Top",
        "das_name": "/ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
    },
    "SingleTtchanTopPostVFP": {
        "filepaths": [
            "{BASE_PATH}/../y2018/ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": 134.2,
        "group": "Top",
        "das_name": "/ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
    },
    # inclusive samples, keep for reference
    # 'WWPostVFP' : {
    #                 'filepaths' :
    #                 ["{BASE_PATH}/WW_TuneCP5_13TeV-pythia8/*.root/NanoV9MC2018_{NANO_PROD_TAG}"],
    #                 'xsec' : 118.7,
    #                 'group' : "Diboson",
    # },
    # 'WZPostVFP' : {
    #                 'filepaths' :
    #                 ["{BASE_PATH}/WZ_TuneCP5_13TeV-pythia8/*.root/NanoV9MC2018_{NANO_PROD_TAG}"],
    #                 'xsec' : 47.026760,  # to check, taken from WZTo1L1Nu2Q dividing by BR: 10.71/(3*0.1086)/(1-3*0.033658-0.2)
    #                 'group' : "Diboson",
    # },
    ##
    "WWTo2L2NuPostVFP": {
        "filepaths": [
            "{BASE_PATH}/../y2018/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": 12.6,  # 118.7*0.1086*0.1086*9
        "group": "Diboson",
        "das_name": "/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
    },
    "WWTo1L1Nu2QPostVFP": {
        "filepaths": [
            "{BASE_PATH}/../y2018/WWTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": 52.146,  # 118.7*[(3*0.1086)*(1-3*0.1086)]*2 (2 is because one W or the other can go to Q)
        "group": "Diboson",
        "das_name": "/WWTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
    },
    "WZTo3LNuPostVFP": {
        "filepaths": [
            "{BASE_PATH}/../y2018/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": 4.91,  # 4.42965*1.109, 1.109 is the NLO to NNLO kfactor, for this one would need to make sure about the NLO XS, depends a lot on the dilepton mass cut
        "group": "Diboson",
        "das_name": "/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
    },
    "WZTo2Q2LPostVFP": {
        "filepaths": [
            "{BASE_PATH}/../y2018/WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": 5.4341,  # 4.9*1.109
        "group": "Diboson",
        "das_name": "/WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
    },
    "WZTo1L1Nu2QPostVFP": {
        "filepaths": [
            "{BASE_PATH}/../y2018/WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": 11.781,  # 10.71*1.10
        "group": "Diboson",
        "das_name": "/WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
    },
    "ZZTo2L2NuPostVFP": {
        "filepaths": [
            "{BASE_PATH}/../y2018/ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": 0.60,
        "group": "Diboson",
        "das_name": "/ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
    },
    "ZZTo2Q2LPostVFP": {
        "filepaths": [
            "{BASE_PATH}/../y2018/ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/NanoV9MC2018_{NANO_PROD_TAG}"
        ],
        "xsec": 5.1,
        "group": "Diboson",
        "das_name": "/ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
    },
    "QCDmuEnrichPt15PostVFP": {  # Not copied
        "filepaths": [
            "{BASE_PATH}/../y2018/QCD_Pt-20_MuEnrichedPt15_TuneCP5_13TeV-pythia8/NanoV9MC2018_{NANO_PROD_TAG}/"
        ],
        "xsec": 238800,
        "group": "QCD",
        "das_name": "/QCD_Pt-20_MuEnrichedPt15_TuneCP5_13TeV-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
    },
}
