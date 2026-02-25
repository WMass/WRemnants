import copy

from utilities.common import (
    xsec_DYJetsToLL,
    xsec_WminusJetsToLNu,
    xsec_WplusJetsToLNu,
)
from wremnants.datasets.datasetDict_2016PostVFP import dataDict as dataDict_2016PostVFP
from wremnants.datasets.datasetDict_2016PostVFP import (
    dataDict as dataDict_2016PostVFP_extended,
)
from wremnants.datasets.datasetDict_2017 import dataDict as dataDict_2017
from wremnants.datasets.datasetDict_2017H import dataDict as dataDict_2017H
from wremnants.datasets.datasetDict_2018 import dataDict as dataDict_2018
from wums import logging

logger = logging.child_logger(__name__)

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

# from https://gitlab.cern.ch/cms-wmass/private/issue-tracking/-/issues/109#note_8972785
xsec_WplusCharmToMuNu = 1607.3778900941979  # +- 4.9247228837888333
xsec_WminusCharmToMuNu = 1749.9768825908691  # +- 6.4490475901643940

horace_v1 = False
horace_v2 = False
horace_v3 = False
horace_v4 = False
horace_v5 = False


def merge_samples(
    source_dict, suffix="MiNNLO", processes=["Zmumu", "Wplusmunu", "Wminusmunu"]
):
    # merge samples from different eras
    res_dict = {}
    for k, v in source_dict.items():
        if v["group"] == "Data":
            continue
        sample, era = k.split("_")
        if len(processes) and sample not in processes:
            continue
        if suffix:
            sample = "_".join([sample, suffix])
        if sample in res_dict.keys():
            if res_dict[sample].get("xsec", None) != v["xsec"]:
                logger.warning(
                    f"Mismatch in cross section between samples {k} with xsec={v["xsec"]} and {sample} with xsec={res_dict[sample].get('xsec', None)}"
                )
            if res_dict[sample].get("group", None) != v["group"]:
                logger.warning(
                    f"Mismatch in group between samples {k} with group={v['group']} and {sample} with group={res_dict[sample].get("group", None)}"
                )

            res_dict[sample]["filepaths"] += v["filepaths"]
        else:
            res_dict[sample] = copy.deepcopy(v)
    return res_dict


# dict with NanoAOD samples
dataDict = merge_samples(
    {**dataDict_2016PostVFP, **dataDict_2017, **dataDict_2017H, **dataDict_2018}
)
dataDict_extended = merge_samples(
    {
        **dataDict_2016PostVFP_extended,
        **dataDict_2017,
        **dataDict_2017H,
        **dataDict_2018,
    }
)

# dict with NanoGen samples
dataDict_nanoGen = {
    "ZmumuMiNLO": {
        "filepaths": [
            "{BASE_PATH}/DYJetsToMuMu_TuneCP5_13TeV-powheg-NNLOPS-pythia8-photos/RunIISummer15wmLHEGS/221121_114507"
        ],
        "xsec": 1863.0,
        "group": "Zmumu",
        "auxiliary": True,
    },
    "ZmumuNNLOPS": {
        "filepaths": [
            "{BASE_PATH}/DYJetsToMuMu_TuneCP5_13TeV-powheg-NNLOPS-pythia8-photos/RunIISummer15wmLHEGS/221121_114507"
        ],
        "xsec": 1863.0,
        "group": "Zmumu",
        "auxiliary": True,
    },
    "Zmumu_horace-lo-photos": {
        "filepaths": [
            "{BASE_PATH}/DYJetsToMuMu_LO_TuneCP5_13TeV-horace-pythia8-photospp"
        ],
        "xsec": xsec_powheg_ZToMuMu_LO,
        "group": "Zmumu",
        "auxiliary": True,
    },
    "Zmumu_horace-lo-photos-mecoff": {
        "filepaths": [
            "{BASE_PATH}/DYJetsToMuMu_LO_TuneCP5_PhotosMecOff_13TeV-horace-pythia8-photospp"
        ],
        "xsec": xsec_powheg_ZToMuMu_LO,
        "group": "Zmumu",
        "auxiliary": True,
    },
    "Zmumu_horace-lo-photos-isroff": {
        "filepaths": [
            "{BASE_PATH}/DYJetsToMuMu_LO_TuneCP5_ISROff_13TeV-horace-pythia8-photospp"
        ],
        "xsec": xsec_powheg_ZToMuMu_LO,
        "group": "Zmumu",
        "auxiliary": True,
    },
    "Zmumu_horace-qed": {
        "filepaths": ["{BASE_PATH}/DYJetsJetsToMuMu_LO_TuneCP5_13TeV-horace-pythia8"],
        "xsec": xsec_powheg_ZToMuMu_LO,
        "group": "Zmumu",
        "auxiliary": True,
    },
    "Zmumu_horace-nlo": {
        "filepaths": ["{BASE_PATH}/DYJetsToMuMu_NLOEW_TuneCP5_13TeV-horace-pythia8"],
        "xsec": xsec_powheg_ZToMuMu_NLOEW,
        "group": "Zmumu",
        "auxiliary": True,
    },
    "Zmumu_powheg-lo": {
        "filepaths": ["{BASE_PATH}/DYJetsToMuMu_NOEW_TuneCP5_13TeV-powheg-pythia8"],
        "xsec": xsec_powheg_ZToMuMu_LO,
        "group": "Zmumu",
        "auxiliary": True,
    },
    "Zmumu_powheg-nloew-qedveto": {
        "filepaths": [
            "{BASE_PATH}/DYJetsToMuMu_NLOEW_QEDVeto_TuneCP5_13TeV-powheg-pythia8"
        ],
        "xsec": xsec_powheg_ZToMuMu_NLOEW,
        "group": "Zmumu",
        "auxiliary": True,
    },
    "Zmumu_powheg-nloew": {
        "filepaths": ["{BASE_PATH}//DYJetsToMuMu_NLOEW_TuneCP5_13TeV-powheg-pythia8"],
        "xsec": xsec_powheg_ZToMuMu_NLOEW,
        "group": "Zmumu",
        "auxiliary": True,
    },
    "Zmumu_MiNNLO-noqedisr": {
        "filepaths": [
            "{BASE_PATH}/DYJetsToMuMu_H2ErratumFix_NoQEDISR_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": xsec_DYJetsToLL,
        "group": "Zmumu",
        "auxiliary": True,
    },
    "Wplusmunu_horace-lo-photos": {
        "filepaths": [
            "{BASE_PATH}/WplusJetsToMuNu_LO_TuneCP5_13TeV-horace-pythia8-photospp"
        ],
        "xsec": xsec_powheg_WplusToMuNu_LO,
        "group": "Wmunu",
        "auxiliary": True,
    },
    "Wplusmunu_horace-lo-photos-mecoff": {
        "filepaths": [
            "{BASE_PATH}/WplusJetsToMuNu_LO_TuneCP5_PhotosAllMecOff_13TeV-horace-pythia8-photospp"
        ],
        "xsec": xsec_powheg_WplusToMuNu_LO,
        "group": "Wmunu",
        "auxiliary": True,
    },
    "Wplusmunu_horace-lo-photos-isroff": {
        "filepaths": [
            "{BASE_PATH}/WplusJetsToMuNu_LO_NoQEDISR_TuneCP5_13TeV-horace-pythia8-photospp"
        ],
        "xsec": xsec_powheg_WplusToMuNu_LO,
        "group": "Wmunu",
        "auxiliary": True,
    },
    "Wplusmunu_horace-qed": {
        "filepaths": ["{BASE_PATH}/WplusJetsToMuNu_LO_TuneCP5_13TeV-horace-pythia8"],
        "xsec": xsec_powheg_WplusToMuNu_LO,
        "group": "Wmunu",
        "auxiliary": True,
    },
    "Wplusmunu_horace-nlo": {
        "filepaths": ["{BASE_PATH}/WplusJetsToMuNu_NLOEW_TuneCP5_13TeV-horace-pythia8"],
        "xsec": xsec_powheg_WplusToMuNu_NLOEW,
        "group": "Wmunu",
        "auxiliary": True,
    },
    "Wplusmunu_winhac-lo-photos": {
        "filepaths": [
            "{BASE_PATH}/WplusJetsToMuNu_LO_TuneCP5_13TeV-winhac-pythia8-photospp"
        ],
        "xsec": xsec_winhac_WplusToMuNu_LO,
        "group": "Wmunu",
        "auxiliary": True,
    },
    "Wplusmunu_winhac-lo": {
        "filepaths": ["{BASE_PATH}/WplusJetsToMuNu_LO_TuneCP5_13TeV-winhac-pythia8"],
        "xsec": xsec_winhac_WplusToMuNu_LO,
        "group": "Wmunu",
        "auxiliary": True,
    },
    "Wplusmunu_winhac-nlo": {
        "filepaths": ["{BASE_PATH}/WplusJetsToMuNu_NLOEW_TuneCP5_13TeV-winhac-pythia8"],
        "xsec": xsec_winhac_WplusToMuNu_NLOEW,
        "group": "Wmunu",
        "auxiliary": True,
    },
    "Wplusmunu_MiNNLO-noqedisr": {
        "filepaths": [
            "{BASE_PATH}/WplusJetsToMuNu_H2ErratumFix_NoQEDISR_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": xsec_WplusJetsToLNu,
        "group": "Wmunu",
        "auxiliary": True,
    },
    "WplusCharmToMuNu": {
        "filepaths": [
            "{BASE_PATH}/WplusCharmToMuNu_TuneCP5_13TeV-powheg-pythia8-photos"
        ],
        "xsec": xsec_WplusCharmToMuNu,
        "group": "Wmunu",
        "auxiliary": True,
    },
    "Wminusmunu_horace-lo-photos": {
        "filepaths": [
            "{BASE_PATH}/WminusJetsToMuNu_LO_TuneCP5_13TeV-horace-pythia8-photospp"
        ],
        "xsec": xsec_powheg_WminusToMuNu_LO,
        "group": "Wmunu",
        "auxiliary": True,
    },
    "Wminusmunu_horace-lo-photos-mecoff": {
        "filepaths": [
            "{BASE_PATH}/WminusJetsToMuNu_LO_TuneCP5_PhotosAllMecOff_13TeV-horace-pythia8-photospp"
        ],
        "xsec": xsec_powheg_WminusToMuNu_LO,
        "group": "Wmunu",
        "auxiliary": True,
    },
    "Wminusmunu_horace-lo-photos-isroff": {
        "filepaths": [
            "{BASE_PATH}/WminusJetsToMuNu_LO_NoQEDISR_TuneCP5_13TeV-horace-pythia8-photospp"
        ],
        "xsec": xsec_powheg_WminusToMuNu_LO,
        "group": "Wmunu",
        "auxiliary": True,
    },
    "Wminusmunu_horace-qed": {
        "filepaths": ["{BASE_PATH}/WminusJetsToMuNu_LO_TuneCP5_13TeV-horace-pythia8"],
        "xsec": xsec_powheg_WminusToMuNu_LO,
        "group": "Wmunu",
        "auxiliary": True,
    },
    "Wminusmunu_horace-nlo": {
        "filepaths": [
            "{BASE_PATH}/WminusJetsToMuNu_NLOEW_TuneCP5_13TeV-horace-pythia8"
        ],
        "xsec": xsec_powheg_WminusToMuNu_NLOEW,
        "group": "Wmunu",
        "auxiliary": True,
    },
    "Wminusmunu_winhac-lo-photos": {
        "filepaths": [
            "{BASE_PATH}/WminusJetsToMuNu_LO_TuneCP5_13TeV-winhac-pythia8-photospp"
        ],
        "xsec": xsec_winhac_WminusToMuNu_LO,
        "group": "Wmunu",
        "auxiliary": True,
    },
    "Wminusmunu_winhac-lo": {
        "filepaths": ["{BASE_PATH}/WminusJetsToMuNu_LO_TuneCP5_13TeV-winhac-pythia8"],
        "xsec": xsec_winhac_WminusToMuNu_LO,
        "group": "Wmunu",
        "auxiliary": True,
    },
    "Wminusmunu_winhac-nlo": {
        "filepaths": [
            "{BASE_PATH}/WminusJetsToMuNu_NLOEW_TuneCP5_13TeV-winhac-pythia8"
        ],
        "xsec": xsec_winhac_WminusToMuNu_NLOEW,
        "group": "Wmunu",
        "auxiliary": True,
    },
    "Wminusmunu_MiNNLO-noqedisr": {
        "filepaths": [
            "{BASE_PATH}/WminusJetsToMuNu_H2ErratumFix_NoQEDISR_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"
        ],
        "xsec": xsec_WminusJetsToLNu,
        "group": "Wmunu",
        "auxiliary": True,
    },
    "WminusCharmToMuNu": {
        "filepaths": [
            "{BASE_PATH}/WminusCharmToMuNu_TuneCP5_13TeV-powheg-pythia8-photos"
        ],
        "xsec": xsec_WminusCharmToMuNu,
        "group": "Wmunu",
        "auxiliary": True,
    },
}

# renesance
dataDict_nanoGen.update(
    {
        "Zmumu_renesance-lo": {
            "filepaths": [
                "{BASE_PATH}/Renesance_v0/ZToMuMu_TuneCP5_13TeV-renesance_loqcd-fsr-photos-isr-pythia"
            ],
            "xsec": xsec_powheg_ZToMuMu_LO,
            "group": "Zmumu",
            "auxiliary": True,
        },
        "Zmumu_renesance-nlo": {
            "filepaths": [
                "{BASE_PATH}/Renesance_v0/ZToMuMu_TuneCP5_13TeV-renesance_loqcdnloweak-fsr-photos-isr-pythia"
            ],
            "xsec": xsec_powheg_ZToMuMu_NLOEW,
            "group": "Zmumu",
            "auxiliary": True,
        },
    }
)

# NanoLHE
# The Powheg EW LHE samples have negative weights but "genWeight" is always just 1, so we will use LHEWeight_originalXWGTUP instead. That also gives us the total cross section for each subsample, so we set that to 1 in this dict.
# TODO copy these samples to central area when they are complete
dataDict_nanoGen.update(
    {
        "Zmumu_powheg-weak-low": {
            "filepaths": ["{BASE_PATH}/svn4049/46mll80"],
            "xsec": 1,
            "group": "Zmumu",
            "auxiliary": True,
        },
        "Zmumu_powheg-weak-peak": {
            "filepaths": ["{BASE_PATH}/svn4049/80mll100"],
            "xsec": 1,
            "group": "Zmumu",
            "auxiliary": True,
        },
        "Zmumu_powheg-weak-high": {
            "filepaths": ["{BASE_PATH}/svn4049/100mll150"],
            "xsec": 1,
            "group": "Zmumu",
            "auxiliary": True,
        },
    }
)

if horace_v1:
    dataDict_nanoGen.update(
        {
            "Zmumu_horace-v1-alpha-old-fsr-off-isr-pythia": {
                "filepaths": [
                    "{BASE_PATH}/Horace_v1/ZToMuMu_TuneCP5_13TeV-horace-alpha-old-fsr-off-isr-pythia"
                ],
                "xsec": xsec_DYJetsToLL,
                "group": "Zmumu",
                "auxiliary": True,
            },
            "Zmumu_horace-v1-born-fsr-photos-isr-pythia": {
                "filepaths": [
                    "{BASE_PATH}/Horace_v1/ZToMuMu_TuneCP5_13TeV-horace-born-fsr-photos-isr-pythia"
                ],
                "xsec": xsec_DYJetsToLL,
                "group": "Zmumu",
                "auxiliary": True,
            },
            "Zmumu_horace-v1-born-fsr-photoslow-isr-pythia": {
                "filepaths": [
                    "{BASE_PATH}/Horace_v1/ZToMuMu_TuneCP5_13TeV-horace-born-fsr-photoslow-isr-pythia"
                ],
                "xsec": xsec_DYJetsToLL,
                "group": "Zmumu",
                "auxiliary": True,
            },
            "Zmumu_horace-v1-lo-photos": {
                "filepaths": [
                    "{BASE_PATH}/Horace_v1/ZToMuMu_TuneCP5_13TeV-horace-born-fsr-photosnopair-isr-pythia"
                ],
                "xsec": xsec_powheg_ZToMuMu_LO,
                "group": "Zmumu",
                "auxiliary": True,
            },
            "Zmumu_horace-v1-born-fsr-pythia-isr-pythia": {
                "filepaths": [
                    "{BASE_PATH}/Horace_v1/ZToMuMu_TuneCP5_13TeV-horace-born-fsr-pythia-isr-pythia"
                ],
                "xsec": xsec_DYJetsToLL,
                "group": "Zmumu",
                "auxiliary": True,
            },
            "Zmumu_horace-v1-nlo": {
                "filepaths": [
                    "{BASE_PATH}/Horace_v1/ZToMuMu_TuneCP5_13TeV-horace-exp-fsr-off-isr-off"
                ],
                "xsec": xsec_powheg_ZToMuMu_NLOEW,
                "group": "Zmumu",
                "auxiliary": True,
            },
            "Zmumu_horace-v1-qed": {
                "filepaths": [
                    "{BASE_PATH}/Horace_v1/ZToMuMu_TuneCP5_13TeV-horace-exp-old-fsr-off-isr-pythia"
                ],
                "xsec": xsec_DYJetsToLL,
                "group": "Zmumu",
                "auxiliary": True,
            },
        }
    )

if horace_v2:
    dataDict_nanoGen.update(
        {
            "Zmumu_horace-v2-lo-photos": {
                "filepaths": [
                    "{BASE_PATH}/Horace_v2/ZToMuMu_TuneCP5_13TeV-horace-born-fsr-photos_nopair-isr-pythia"
                ],
                "xsec": xsec_powheg_ZToMuMu_LO,
                "group": "Zmumu",
                "auxiliary": True,
            },
            "Zmumu_horace-v2-nlo": {
                "filepaths": [
                    "{BASE_PATH}/Horace_v2/ZToMuMu_TuneCP5_13TeV-horace-exp-new-fsr-off-isr-off"
                ],
                "xsec": xsec_powheg_ZToMuMu_NLOEW,
                "group": "Zmumu",
                "auxiliary": True,
            },
            "Zmumu_horace-v2-qed": {
                "filepaths": [
                    "{BASE_PATH}/Horace_v2/ZToMuMu_TuneCP5_13TeV-horace-exp-old-fsr-off-isr-pythia"
                ],
                "xsec": xsec_DYJetsToLL,
                "group": "Zmumu",
                "auxiliary": True,
            },
        }
    )

if horace_v3:
    dataDict_nanoGen.update(
        {
            "Zmumu_horace-v3-lo-photos": {
                "filepaths": [
                    "{BASE_PATH}/Horace_v3/ZToMuMu_TuneCP5_13TeV-horace-born-fsr-photos_nopair-isr-pythia"
                ],
                "xsec": xsec_powheg_ZToMuMu_LO,
                "group": "Zmumu",
                "auxiliary": True,
            },
            "Zmumu_horace-v3-qed": {
                "filepaths": [
                    "{BASE_PATH}/Horace_v3/ZToMuMu_TuneCP5_13TeV-horace-exp-old-fsr-off-isr-pythia"
                ],
                "xsec": xsec_DYJetsToLL,
                "group": "Zmumu",
                "auxiliary": True,
            },
            "Zmumu_horace-v3-nlo": {
                "filepaths": [
                    "{BASE_PATH}/Horace_v3/ZToMuMu_TuneCP5_13TeV-horace-exp-new-fsr-off-isr-off"
                ],
                "xsec": xsec_powheg_ZToMuMu_NLOEW,
                "group": "Zmumu",
                "auxiliary": True,
            },
            "Wplusmunu_horace-v3-lo-photos": {
                "filepaths": [
                    "{BASE_PATH}/Horace_v3/WplusToMuNu_TuneCP5_13TeV-horace-born-fsr-photos_nopair-isr-pythia"
                ],
                "xsec": xsec_powheg_WplusToMuNu_LO,
                "group": "Wmunu",
                "auxiliary": True,
            },
            "Wplusmunu_horace-v3-qed": {
                "filepaths": [
                    "{BASE_PATH}/Horace_v3/WplusToMuNu_TuneCP5_13TeV-horace-exp-old-fsr-off-isr-pythia"
                ],
                "xsec": xsec_WplusJetsToLNu,
                "group": "Wmunu",
                "auxiliary": True,
            },
            "Wplusmunu_horace-v3-nlo": {
                "filepaths": [
                    "{BASE_PATH}/Horace_v3/WplusToMuNu_TuneCP5_13TeV-horace-exp-new-fsr-off-isr-off"
                ],
                "xsec": xsec_powheg_WplusToMuNu_NLOEW,
                "group": "Wmunu",
                "auxiliary": True,
            },
            "Wminusmunu_horace-v3-lo-photos": {
                "filepaths": [
                    "{BASE_PATH}/Horace_v3/WminusToMuNu_TuneCP5_13TeV-horace-born-fsr-photos_nopair-isr-pythia"
                ],
                "xsec": xsec_powheg_WminusToMuNu_LO,
                "group": "Wmunu",
                "auxiliary": True,
            },
            "Wminusmunu_horace-v3-qed": {
                "filepaths": [
                    "{BASE_PATH}/Horace_v3/WminusToMuNu_TuneCP5_13TeV-horace-exp-old-fsr-off-isr-pythia"
                ],
                "xsec": xsec_WminusJetsToLNu,
                "group": "Wmunu",
                "auxiliary": True,
            },
            "Wminusmunu_horace-v3-nlo": {
                "filepaths": [
                    "{BASE_PATH}/Horace_v3/WminusToMuNu_TuneCP5_13TeV-horace-exp-new-fsr-off-isr-off"
                ],
                "xsec": xsec_powheg_WminusToMuNu_NLOEW,
                "group": "Wmunu",
                "auxiliary": True,
            },
        }
    )

if horace_v5:
    dataDict_nanoGen.update(
        {
            "Zmumu_horace-v5-alpha-fsr-off-isr-off": {
                "filepaths": [
                    "{BASE_PATH}/Horace_v5/ZToMuMu_TuneCP5_13TeV-horace-alpha-fsr-off-isr-off"
                ],
                "xsec": xsec_DYJetsToLL,
                "group": "Zmumu",
                "auxiliary": True,
            },
            "Zmumu_horace-v5-alpha-old-fsr-off-isr-off": {
                "filepaths": [
                    "{BASE_PATH}/Horace_v5/ZToMuMu_TuneCP5_13TeV-horace-alpha-old-fsr-off-isr-off"
                ],
                "xsec": xsec_DYJetsToLL,
                "group": "Zmumu",
                "auxiliary": True,
            },
            "Zmumu_horace-v5-alpha-old-fsr-off-isr-pythia": {
                "filepaths": [
                    "{BASE_PATH}/Horace_v5/ZToMuMu_TuneCP5_13TeV-horace-alpha-old-fsr-off-isr-pythia"
                ],
                "xsec": xsec_DYJetsToLL,
                "group": "Zmumu",
                "auxiliary": True,
            },
            "Zmumu_horace-v5-nlo": {
                "filepaths": [
                    "{BASE_PATH}/Horace_v5/ZToMuMu_TuneCP5_13TeV-horace-exp-new-fsr-off-isr-off"
                ],
                "xsec": xsec_powheg_ZToMuMu_NLOEW,
                "group": "Zmumu",
                "auxiliary": True,
            },
        }
    )
