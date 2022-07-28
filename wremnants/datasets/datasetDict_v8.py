BR_TAUToMU = 0.1739
BR_TAUToE = 0.1782
xsec_ZmmPostVFP = 1976.1
xsec_WpmunuPostVFP = 11572.19
xsec_WmmunuPostVFP = 8562.66

dataDictV8 = {
    "dataPostVFP" : { 'name' :"dataPostVFP",
                      'filepaths' : ["/scratch/shared/NanoAOD/TrackRefitv1/SingleMuon/Run2016F_postVFP_220223_222034/*/*.root",
                                     "/scratch/shared/NanoAOD/TrackRefitv1/SingleMuon/Run2016G_220223_222128/*/*.root",
                                     "/scratch/shared/NanoAOD/TrackRefitv1/SingleMuon/Run2016H_220223_222223/*/*.root",
                      ],
    },
    "ZmmPostVFP" : { 'name' :"ZmumuPostVFP",
                     'filepaths' : 
                     ["/scratch/shared/NanoAOD/TrackRefitv1/DYJetsToMuMu_M-50_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV8MCPostVFPWeightFix/*/*/*.root"],
                     'xsec' : xsec_ZmmPostVFP,
    },
    
    "ZttPostVFP" : { 'name' :"ZtautauPostVFP",
                     'filepaths' : 
                     ["/scratch/shared/NanoAOD/TrackRefitv1/DYJetsToTauTau_M-50_AtLeastOneEorMuDecay_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV8MCPostVFPWeightFix/*/*/*.root"],
                # At least one tau->e or mu decay, so everything that's not all other decays
                     'xsec' : xsec_ZmmPostVFP*(1.-(1. - BR_TAUToMU - BR_TAUToE)**2),
    },
    "WpmunuPostVFP" : { 'name' :"WplusmunuPostVFP",
                        'filepaths' : 
                        ["/scratch/shared/NanoAOD/TrackRefitv1/WplusJetsToMuNu_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV8MCPostVFPWeightFix/*/*/*.root"],
                        'xsec' : xsec_WpmunuPostVFP,
    },
    "WmmunuPostVFP" : { 'name' :"WminusmunuPostVFP",
                        'filepaths' : 
                        ["/scratch/shared/NanoAOD/TrackRefitv1/WminusJetsToMuNu_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV8MCPostVFPWeightFix/*/*/*.root"],
                        'xsec' : xsec_WmmunuPostVFP,
    },
    "WptaunuPostVFP" : { 'name' :"WplustaunuPostVFP",
                         'filepaths' : 
                         ["/scratch/shared/NanoAOD/TrackRefitv1/WplusJetsToTauNu_TauToMu_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV8MCPostVFPWeightFix/*/*/*.root"],
                         'xsec' : BR_TAUToMU*xsec_WpmunuPostVFP,
    },
    "WmtaunuPostVFP" : { 'name' :"WminustaunuPostVFP",
                        'filepaths' : 
                         ["/scratch/shared/NanoAOD/TrackRefitv1/WminusJetsToTauNu_TauToMu_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV8MCPostVFPWeightFix/*/*/*.root"],
                         'xsec' : BR_TAUToMU*xsec_WmmunuPostVFP,
    },
    "ttbarlnuPostVFP" : { 'name' :"TTLeptonicPostVFP",
                          'filepaths' : 
                          ["/scratch/shared/originalNANO/TTbar_2l2nu_postVFP/*.root"],
                          'xsec' : 88.29,
                          'group' : "Top",
    },
    "ttbarlqPostVFP" : { 'name' :"TTSemileptonicPostVFP",
                         'filepaths' : 
                         ["/scratch/shared/originalNANO/TTbar_SemiLeptonic_postVFP/*.root"],
                         'xsec' : 365.64,
                         'group' : "Top",
    },
    # TODO: these samples and cross sections are preliminary
    "singleTop_schanLepDecaysPostVFP" : { 'name' :"SingleTschanLepDecaysPostVFP",
                                          'filepaths' : 
                                          ["/scratch/shared/originalNANO/SingleTop_schan_lepDecays_postVFP/*.root"],
                                          'xsec' : 3.74,
                                          'group' : "Top",
    },
    
    "singleTop_tWAntitopPostVFP" : { 'name' :"SingleTtWAntitopPostVFP",
                                     'filepaths' : 
                                     ["/scratch/shared/originalNANO/SingleTop_tW_antitop_noFullyHadronic_postVFP/*.root"],
                                     'xsec' : 19.55,
                                     'group' : "Top",
    },
    "singleTop_tchanAntitopPostVFP" : { 'name' :"SingleTtchanAntitopPostVFP",
                                        'filepaths' : 
                                        ["/scratch/shared/originalNANO/SingleTop_tchan_antitop_inclusive_postVFP/*.root"],
                                        'xsec' : 70.79,
                                        'group' : "Top",
    },
    "singleTop_tchanTopPostVFP" : { 'name' :"SingleTtchanTopPostVFP",
                                    'filepaths' : 
                                    ["/scratch/shared/originalNANO/SingleTop_tchan_top_inclusive_postVFP/*.root"],
                                    'xsec' : 119.71,
                                    'group' : "Top",
    },    
    # TODO: should really use the specific decay channels
    "wwPostVFP" : { 'name' :"WWPostVFP",
                    'filepaths' : 
                    ["/scratch/shared/originalNANO/WW_inclusive_postVFP/*.root"],
                    'xsec' : 75.8,
                    'group' : "Diboson",
    },
    "wzPostVFP" : { 'name' :"WZPostVFP",
                    'filepaths' : 
                    ["/scratch/shared/originalNANO/WZ_inclusive_postVFP/*.root"],
                    'xsec' : 27.6,
                    'group' : "Diboson",
    },
    "zz2l2nuPostVFP" : { 'name' :"ZZ2l2nuPostVFP",
                         'filepaths' : 
                         ["/scratch/shared/originalNANO/ZZ_2l2nu_postVFP/*.root"],
                         'xsec' : 0.564,
                         'group' : "Diboson",
    }
}
