from utilities.common import xsec_ZmmPostVFP,xsec_WpmunuPostVFP,xsec_WmmunuPostVFP

# winhac cross sections from: https://gitlab.cern.ch/cms-wmass/private/issue-tracking/-/issues/34#note_7052239
xsec_winhac_WplusToMuNu_LO = 10104.50380
xsec_winhac_WplusToMuNu_NLOEW = 10117.09477
xsec_winhac_WminusToMuNu_LO = 7574.05974
xsec_winhac_WminusToMuNu_NLOEW = 7585.16621

xsec_powheg_ZToMuMu_LO = 2000.4267556051570
xsec_powheg_ZToMuMu_NLOEW = 2026.3829240506323
xsec_powheg_WplusToMuNu_LO = 11690.260441335342  
xsec_powheg_WplusToMuNu_NLOEW = 11657.708788022124 
xsec_powheg_WminusToMuNu_LO = 8687.5312426956061
xsec_powheg_WminusToMuNu_NLOEW = 8664.1686115643297

horace_v1 = False
horace_v2 = False
horace_v3 = False
horace_v4 = False
horace_v5 = False

genDataDict = {
    'ZmumuMiNLO' : { 
                   'filepaths' :
                    ["{BASE_PATH}/DYJetsToMuMu_TuneCP5_13TeV-powheg-NNLOPS-pythia8-photos/RunIISummer15wmLHEGS/221121_114507"],
                   'xsec' : 1863.,
                   'group': "Zmumu",
    },
    'ZmumuNNLOPS' : { 
                   'filepaths' :
                    ["{BASE_PATH}/DYJetsToMuMu_TuneCP5_13TeV-powheg-NNLOPS-pythia8-photos/RunIISummer15wmLHEGS/221121_114507"],
                   'xsec' : 1863.,
                   'group': "Zmumu",
    },
    'Zmumu_horace-lo-photos' : { 
                   'filepaths' :
                    ["{BASE_PATH}/DYJetsToMuMu_LO_TuneCP5_13TeV-horace-pythia8-photospp"],
                   'xsec' : xsec_powheg_ZToMuMu_LO,
                   'group': "Zmumu",
    },
    'Zmumu_horace-lo-photos-mecoff' : { 
                   'filepaths' :
                   ["{BASE_PATH}/DYJetsToMuMu_LO_TuneCP5_PhotosMecOff_13TeV-horace-pythia8-photospp"],
                   'xsec' : xsec_powheg_ZToMuMu_LO,
                   'group': "Zmumu",
    },      
    'Zmumu_horace-lo-photos-isroff' : { 
                'filepaths' :
                ["{BASE_PATH}/DYJetsToMuMu_LO_TuneCP5_ISROff_13TeV-horace-pythia8-photospp"],
                'xsec' : xsec_powheg_ZToMuMu_LO,
                'group': "Zmumu",
    },
    'Zmumu_horace-qed' : { 
                   'filepaths' :
                    ["{BASE_PATH}/DYJetsJetsToMuMu_LO_TuneCP5_13TeV-horace-pythia8"],
                   'xsec' : xsec_powheg_ZToMuMu_LO,
                   'group': "Zmumu",
    },
    'Zmumu_horace-nlo' : { 
                   'filepaths' :
                    ["{BASE_PATH}/DYJetsToMuMu_NLOEW_TuneCP5_13TeV-horace-pythia8"],
                   'xsec' : xsec_powheg_ZToMuMu_NLOEW,
                   'group': "Zmumu",
    },
    'Zmumu_powheg-lo' : { 
                   'filepaths' :
                    ["{BASE_PATH}/DYJetsToMuMu_NOEW_TuneCP5_13TeV-powheg-pythia8"],
                   'xsec' : xsec_powheg_ZToMuMu_LO,
                   'group': "Zmumu",
    },
    'Zmumu_powheg-nloew-qedveto' : { 
                   'filepaths' :
                    ["{BASE_PATH}/DYJetsToMuMu_NLOEW_QEDVeto_TuneCP5_13TeV-powheg-pythia8"],
                   'xsec' : xsec_powheg_ZToMuMu_NLOEW,
                   'group': "Zmumu",
    },
    'Zmumu_powheg-nloew' : { 
                   'filepaths' :
                    ["{BASE_PATH}//DYJetsToMuMu_NLOEW_TuneCP5_13TeV-powheg-pythia8"],
                   'xsec' : xsec_powheg_ZToMuMu_NLOEW,
                   'group': "Zmumu",
    },
    'Zmumu_MiNNLO-noqedisr' : { 
                   'filepaths' :
                    ["{BASE_PATH}/DYJetsToMuMu_H2ErratumFix_NoQEDISR_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"],
                   'xsec' : xsec_ZmmPostVFP,
                   'group': "Zmumu",
    },
    'Wplusmunu_horace-lo-photos' : { 
                   'filepaths' :
                    ["{BASE_PATH}/WplusJetsToMuNu_LO_TuneCP5_13TeV-horace-pythia8-photospp"],
                   'xsec' : xsec_powheg_WplusToMuNu_LO,
                    'group': "Wmunu",
    },
    'Wplusmunu_horace-lo-photos-mecoff' : { 
                 'filepaths' :
                ["{BASE_PATH}/WplusJetsToMuNu_LO_TuneCP5_PhotosMecOff_13TeV-horace-pythia8-photospp"],
                 'xsec' : xsec_powheg_WplusToMuNu_LO,
                 'group': "Wmunu",
    },    
    'Wplusmunu_horace-lo-photos-isroff' : { 
                'filepaths' :
                ["{BASE_PATH}/WplusJetsToMuNu_LO_NoQEDISR_TuneCP5_13TeV-horace-pythia8-photospp"],
                'xsec' : xsec_powheg_WplusToMuNu_LO,
                'group': "Wmunu",
    },  
    'Wplusmunu_horace-qed' : { 
                   'filepaths' :
                    ["{BASE_PATH}/WplusJetsToMuNu_LO_TuneCP5_13TeV-horace-pythia8"],
                   'xsec' : xsec_powheg_WplusToMuNu_LO,
                    'group': "Wmunu",
    },
    'Wplusmunu_horace-nlo' : { 
                   'filepaths' :
                    ["{BASE_PATH}/WplusJetsToMuNu_NLOEW_TuneCP5_13TeV-horace-pythia8"],
                   'xsec' : xsec_powheg_WplusToMuNu_NLOEW,
                    'group': "Wmunu",
    },
    'Wplusmunu_winhac-lo-photos' : { 
                   'filepaths' :
                    ["{BASE_PATH}/WplusJetsToMuNu_LO_TuneCP5_13TeV-winhac-pythia8-photospp"],
                   'xsec' : xsec_winhac_WplusToMuNu_LO,
                    'group': "Wmunu",
    },
    'Wplusmunu_winhac-lo' : { 
                   'filepaths' :
                    ["{BASE_PATH}/WplusJetsToMuNu_LO_TuneCP5_13TeV-winhac-pythia8"],
                   'xsec' : xsec_winhac_WplusToMuNu_LO,
                    'group': "Wmunu",
    },
    'Wplusmunu_winhac-nlo' : { 
                   'filepaths' :
                    ["{BASE_PATH}/WplusJetsToMuNu_NLOEW_TuneCP5_13TeV-winhac-pythia8"],
                   'xsec' : xsec_winhac_WplusToMuNu_NLOEW,
                    'group': "Wmunu",
    },
    'Wplusmunu_MiNNLO-noqedisr' : { 
                   'filepaths' :
                    ["{BASE_PATH}/WplusJetsToMuNu_H2ErratumFix_NoQEDISR_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"],
                   'xsec' : xsec_WpmunuPostVFP,
                    'group': "Wmunu",
    },
    'Wminusmunu_horace-lo-photos' : { 
                   'filepaths' :
                    ["{BASE_PATH}/WminusJetsToMuNu_LO_TuneCP5_13TeV-horace-pythia8-photospp"],
                   'xsec' : xsec_powheg_WminusToMuNu_LO,
                    'group': "Wmunu",
    },
    'Wminusmunu_horace-lo-photos-mecoff' : { 
                 'filepaths' :
                ["{BASE_PATH}/WminusJetsToMuNu_LO_TuneCP5_PhotosMecOff_13TeV-horace-pythia8-photospp"],
                 'xsec' : xsec_powheg_WminusToMuNu_LO,
                 'group': "Wmunu",
    },  
    'Wminusmunu_horace-lo-photos-isroff' : { 
                'filepaths' :
                ["{BASE_PATH}/WminusJetsToMuNu_LO_NoQEDISR_TuneCP5_13TeV-horace-pythia8-photospp"],
                'xsec' : xsec_powheg_WminusToMuNu_LO,
                'group': "Wmunu",
    },   
    'Wminusmunu_horace-qed' : { 
                   'filepaths' :
                    ["{BASE_PATH}/WminusJetsToMuNu_LO_TuneCP5_13TeV-horace-pythia8"],
                   'xsec' : xsec_powheg_WminusToMuNu_LO,
                    'group': "Wmunu",
    },
    'Wminusmunu_horace-nlo' : { 
                   'filepaths' :
                    ["{BASE_PATH}/WminusJetsToMuNu_NLOEW_TuneCP5_13TeV-horace-pythia8"],
                   'xsec' : xsec_powheg_WminusToMuNu_NLOEW,
                    'group': "Wmunu",
    },
    'Wminusmunu_winhac-lo-photos' : { 
                   'filepaths' :
                    ["{BASE_PATH}/WminusJetsToMuNu_LO_TuneCP5_13TeV-winhac-pythia8-photospp"],
                   'xsec' : xsec_winhac_WminusToMuNu_LO,
                    'group': "Wmunu",
    },
    'Wminusmunu_winhac-lo' : { 
                   'filepaths' :
                    ["{BASE_PATH}/WminusJetsToMuNu_LO_TuneCP5_13TeV-winhac-pythia8"],
                   'xsec' : xsec_winhac_WminusToMuNu_LO,
                    'group': "Wmunu",
    },
    'Wminusmunu_winhac-nlo' : { 
                   'filepaths' :
                    ["{BASE_PATH}/WminusJetsToMuNu_NLOEW_TuneCP5_13TeV-winhac-pythia8"],
                   'xsec' : xsec_winhac_WminusToMuNu_NLOEW,
                   'group': "Wmunu",
    },
    'Wminusmunu_MiNNLO-noqedisr' : { 
                   'filepaths' :
                    ["{BASE_PATH}/WminusJetsToMuNu_H2ErratumFix_NoQEDISR_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"],
                   'xsec' : xsec_WmmunuPostVFP,
                   'group': "Wmunu",
    },
}

# renesance
genDataDict.update({
    'Zmumu_renesance-lo' : { 
                   'filepaths' :
                    ["{BASE_PATH}/Renesance_v0/ZToMuMu_TuneCP5_13TeV-renesance_loqcd-fsr-photos-isr-pythia"],
                   'xsec' : xsec_powheg_ZToMuMu_LO,
                   'group': "Zmumu",
    },
    'Zmumu_renesance-nlo' : { 
                   'filepaths' :
                    ["{BASE_PATH}/Renesance_v0/ZToMuMu_TuneCP5_13TeV-renesance_loqcdnloweak-fsr-photos-isr-pythia"],
                   'xsec' : xsec_powheg_ZToMuMu_NLOEW,
                   'group': "Zmumu",
    },
})

# NanoLHE
# The Powheg EW LHE samples have negative weights but "genWeight" is always just 1, so we will use LHEWeight_originalXWGTUP instead. That also gives us the total cross section for each subsample, so we set that to 1 in this dict.
genDataDict.update({
    'Zmumu_powheg-weak-low' : {
                   'filepaths' :
                    ["{BASE_PATH}/NanoLHE/Zmumu_powheg-weak/46mll80"],
                   'xsec' : 1,
                   'group': "Zmumu",
    },
    'Zmumu_powheg-weak-peak' : {
                   'filepaths' :
                    ["{BASE_PATH}/NanoLHE/Zmumu_powheg-weak/80mll100"],
                   'xsec' : 1,
                   'group': "Zmumu",
    },
    'Zmumu_powheg-weak-high' : {
                   'filepaths' :
                    ["{BASE_PATH}/NanoLHE/Zmumu_powheg-weak/100mll150"],
                   'xsec' : 1,
                   'group': "Zmumu",
    },
})

if horace_v1:
    genDataDict.update({
        'Zmumu_horace-v1-alpha-old-fsr-off-isr-pythia' : { 
                'filepaths' :
                ["{BASE_PATH}/Horace_v1/ZToMuMu_TuneCP5_13TeV-horace-alpha-old-fsr-off-isr-pythia"],
                'xsec' : xsec_ZmmPostVFP,
                'group': "Zmumu",
        },
        'Zmumu_horace-v1-born-fsr-photos-isr-pythia' : { 
                'filepaths' :
                ["{BASE_PATH}/Horace_v1/ZToMuMu_TuneCP5_13TeV-horace-born-fsr-photos-isr-pythia"],
                'xsec' : xsec_ZmmPostVFP,
                'group': "Zmumu",
        },
        'Zmumu_horace-v1-born-fsr-photoslow-isr-pythia' : { 
                'filepaths' :
                ["{BASE_PATH}/Horace_v1/ZToMuMu_TuneCP5_13TeV-horace-born-fsr-photoslow-isr-pythia"],
                'xsec' : xsec_ZmmPostVFP,
                'group': "Zmumu",
        },
        'Zmumu_horace-v1-lo-photos' : { 
                'filepaths' :
                ["{BASE_PATH}/Horace_v1/ZToMuMu_TuneCP5_13TeV-horace-born-fsr-photosnopair-isr-pythia"],
                'xsec' : xsec_powheg_ZToMuMu_LO,
                'group': "Zmumu",
        },
        'Zmumu_horace-v1-born-fsr-pythia-isr-pythia' : { 
                'filepaths' :
                ["{BASE_PATH}/Horace_v1/ZToMuMu_TuneCP5_13TeV-horace-born-fsr-pythia-isr-pythia"],
                'xsec' : xsec_ZmmPostVFP,
                'group': "Zmumu",
        },
        'Zmumu_horace-v1-nlo' : { 
                'filepaths' :
                ["{BASE_PATH}/Horace_v1/ZToMuMu_TuneCP5_13TeV-horace-exp-fsr-off-isr-off"],
                'xsec' : xsec_powheg_ZToMuMu_NLOEW,
                'group': "Zmumu",
        },
        'Zmumu_horace-v1-qed' : { 
                'filepaths' :
                ["{BASE_PATH}/Horace_v1/ZToMuMu_TuneCP5_13TeV-horace-exp-old-fsr-off-isr-pythia"],
                'xsec' : xsec_ZmmPostVFP,
                'group': "Zmumu",
        },
    })

if horace_v2:
    genDataDict.update({
        'Zmumu_horace-v2-lo-photos' : { 
                'filepaths' :
                ["{BASE_PATH}/Horace_v2/ZToMuMu_TuneCP5_13TeV-horace-born-fsr-photos_nopair-isr-pythia"],
                'xsec' : xsec_powheg_ZToMuMu_LO,
                'group': "Zmumu",
        },
        'Zmumu_horace-v2-nlo' : { 
                'filepaths' :
                ["{BASE_PATH}/Horace_v2/ZToMuMu_TuneCP5_13TeV-horace-exp-new-fsr-off-isr-off"],
                'xsec' : xsec_powheg_ZToMuMu_NLOEW,
                'group': "Zmumu",
        },
        'Zmumu_horace-v2-qed' : { 
                'filepaths' :
                ["{BASE_PATH}/Horace_v2/ZToMuMu_TuneCP5_13TeV-horace-exp-old-fsr-off-isr-pythia"],
                'xsec' : xsec_ZmmPostVFP,
                'group': "Zmumu",
        },
    })

if horace_v3:
    genDataDict.update({
        'Zmumu_horace-v3-lo-photos' : { 
                'filepaths' :
                ["{BASE_PATH}/Horace_v3/ZToMuMu_TuneCP5_13TeV-horace-born-fsr-photos_nopair-isr-pythia"],
                'xsec' : xsec_powheg_ZToMuMu_LO,
                'group': "Zmumu",
        },
        'Zmumu_horace-v3-qed' : { 
                'filepaths' :
                ["{BASE_PATH}/Horace_v3/ZToMuMu_TuneCP5_13TeV-horace-exp-old-fsr-off-isr-pythia"],
                'xsec' : xsec_ZmmPostVFP,
                'group': "Zmumu",
        },
        'Zmumu_horace-v3-nlo' : { 
                'filepaths' :
                ["{BASE_PATH}/Horace_v3/ZToMuMu_TuneCP5_13TeV-horace-exp-new-fsr-off-isr-off"],
                'xsec' : xsec_powheg_ZToMuMu_NLOEW,
                'group': "Zmumu",
        },
        'Wplusmunu_horace-v3-lo-photos' : { 
                'filepaths' :
                ["{BASE_PATH}/Horace_v3/WplusToMuNu_TuneCP5_13TeV-horace-born-fsr-photos_nopair-isr-pythia"],
                'xsec' : xsec_powheg_WplusToMuNu_LO,
                'group': "Wmunu",
        },
        'Wplusmunu_horace-v3-qed' : { 
                'filepaths' :
                ["{BASE_PATH}/Horace_v3/WplusToMuNu_TuneCP5_13TeV-horace-exp-old-fsr-off-isr-pythia"],
                'xsec' : xsec_WpmunuPostVFP,
                'group': "Wmunu",
        },
        'Wplusmunu_horace-v3-nlo' : { 
                'filepaths' :
                ["{BASE_PATH}/Horace_v3/WplusToMuNu_TuneCP5_13TeV-horace-exp-new-fsr-off-isr-off"],
                'xsec' : xsec_powheg_WplusToMuNu_NLOEW,
                'group': "Wmunu",
        },
        'Wminusmunu_horace-v3-lo-photos' : { 
                'filepaths' :
                ["{BASE_PATH}/Horace_v3/WminusToMuNu_TuneCP5_13TeV-horace-born-fsr-photos_nopair-isr-pythia"],
                'xsec' : xsec_powheg_WminusToMuNu_LO,
                'group': "Wmunu",
        },
        'Wminusmunu_horace-v3-qed' : { 
                'filepaths' :
                ["{BASE_PATH}/Horace_v3/WminusToMuNu_TuneCP5_13TeV-horace-exp-old-fsr-off-isr-pythia"],
                'xsec' : xsec_WmmunuPostVFP,
                'group': "Wmunu",
        },
        'Wminusmunu_horace-v3-nlo' : { 
                'filepaths' :
                ["{BASE_PATH}/Horace_v3/WminusToMuNu_TuneCP5_13TeV-horace-exp-new-fsr-off-isr-off"],
                'xsec' : xsec_powheg_WminusToMuNu_NLOEW,
                'group': "Wmunu",
        }
    })

if horace_v5:
     genDataDict.update({
        'Zmumu_horace-v5-alpha-fsr-off-isr-off' : { 
                 'filepaths' :
                ["{BASE_PATH}/Horace_v5/ZToMuMu_TuneCP5_13TeV-horace-alpha-fsr-off-isr-off"],
                 'xsec' : xsec_ZmmPostVFP,
                 'group': "Zmumu",
         },
        'Zmumu_horace-v5-alpha-old-fsr-off-isr-off' : { 
                 'filepaths' :
                ["{BASE_PATH}/Horace_v5/ZToMuMu_TuneCP5_13TeV-horace-alpha-old-fsr-off-isr-off"],
                 'xsec' : xsec_ZmmPostVFP,
                 'group': "Zmumu",
         },
        'Zmumu_horace-v5-alpha-old-fsr-off-isr-pythia' : { 
                 'filepaths' :
                ["{BASE_PATH}/Horace_v5/ZToMuMu_TuneCP5_13TeV-horace-alpha-old-fsr-off-isr-pythia"],
                 'xsec' : xsec_ZmmPostVFP,
                 'group': "Zmumu",
         },
        'Zmumu_horace-v5-nlo' : { 
                 'filepaths' :
                ["{BASE_PATH}/Horace_v5/ZToMuMu_TuneCP5_13TeV-horace-exp-new-fsr-off-isr-off"],
                 'xsec' : xsec_powheg_ZToMuMu_NLOEW,
                 'group': "Zmumu",
         },
    })
