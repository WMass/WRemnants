"""
Functionality to load and prepare datasets with the focus on CMS NanoAOD or NanoGEN files
"""

import importlib

import ROOT

import narf

# Path / file-list helpers live in a ROOT/narf-free module so they can be
# imported without pulling in ROOT (this package's __init__ imports ROOT and
# narf). Re-exported here for backward compatibility.
from wremnants.utilities.data_paths import (  # noqa: F401
    appendFilesXrd,
    buildFileList,
    buildFileListPosix,
    buildFileListXrd,
    getDataPath,
    makeFilelist,
)
from wums import logging

logger = logging.child_logger(__name__)

default_nfiles = {
    "Wminusmunu_2016PostVFP": 1700,
    "Wplusmunu_2016PostVFP": 2000,
    "Wminustaunu_2016PostVFP": 400,
    "Wplustaunu_2016PostVFP": 500,
    "Zmumu_2016PostVFP": 900,
    "Ztautau_2016PostVFP": 1200,
}


def is_zombie(file_path):
    # Try opening the ROOT file and check if it's a zombie file
    file = ROOT.TFile.Open(file_path)
    if not file or file.IsZombie():
        logger.warning(f"Found zombie file: {file_path}")
        return True
    file.Close()
    return False


def getDatasets(
    maxFiles=default_nfiles,
    filt=None,
    excl=None,
    aux=None,
    base_path=None,
    nanoVersion="v9",
    data_tags=[
        "TrackFitV722_NanoProdv6",
        "TrackFitV722_NanoProdv5",
        "TrackFitV722_NanoProdv3",
    ],
    mc_tags=[
        "TrackFitV722_NanoProdv6",
        "TrackFitV722_NanoProdv5",
        "TrackFitV722_NanoProdv4",
        "TrackFitV722_NanoProdv3",
    ],
    oneMCfileEveryN=None,
    checkFileForZombie=False,
    era="2016PostVFP",
    extended=False,
):

    if maxFiles is None or (isinstance(maxFiles, int) and maxFiles < -1):
        maxFiles = default_nfiles

    if not base_path:
        base_path = getDataPath()
    logger.info(f"Loading samples from {base_path}.")

    module = importlib.import_module(f"wremnants.production.datasets.datasetDict_{era}")
    if extended:
        dataDict = getattr(module, "dataDict_extended", {})
        if len(dataDict) == 0:
            raise ValueError(f"Extended datasets not defined for module '{module}'")
    else:
        dataDict = getattr(module, "dataDict")

    dataDict_NanoGen = getattr(module, "dataDict_nanoGen", {})
    if dataDict_NanoGen:
        dataDict.update(dataDict_NanoGen)

    narf_datasets = []
    for sample, info in dataDict.items():
        if excl not in [None, []] and (info["group"] in excl or sample in excl):
            continue
        if filt not in [None, []]:
            # keep the sample if it is explicitly filtered
            if info["group"] not in filt and sample not in filt:
                continue
        elif info.get("auxiliary") and (
            aux in [None, []] or (info["group"] not in aux and sample not in aux)
        ):
            # skip if it is not explicitly filtered, auxiliary, and not specified by aux
            continue

        if sample in dataDict_NanoGen.keys():
            base_path_sample = base_path.replace("NanoAOD", "NanoGen")
        else:
            base_path_sample = base_path

        is_data = info.get("group", "") == "Data"

        prod_tags = data_tags if is_data else mc_tags

        nfiles = maxFiles
        if type(maxFiles) == dict:
            nfiles = maxFiles[sample] if sample in maxFiles else -1
        paths = makeFilelist(
            info["filepaths"],
            nfiles,
            base_path=base_path_sample,
            nano_prod_tags=prod_tags,
            is_data=is_data,
            oneMCfileEveryN=oneMCfileEveryN,
            era=era,
        )

        if checkFileForZombie:
            paths = [p for p in paths if not is_zombie(p)]

        # paths = list(filter(lambda x: not ("WminusJetsToMuNu" in x and os.path.basename(x) in ["NanoV9MCPostVFP_4316.root","NanoV9MCPostVFP_4372.root","NanoV9MCPostVFP_4310.root","NanoV9MCPostVFP_4377.root","NanoV9MCPostVFP_4306.root"]), paths))

        if not paths:
            logger.warning(
                f"Failed to find any files for dataset {sample}. Looking at {info['filepaths']}. Skipping!"
            )
            continue

        narf_info = dict(
            name=sample,
            filepaths=paths,
        )

        if is_data:
            narf_info.update(
                dict(
                    is_data=True,
                    lumi_csv=info["lumicsv"],
                    lumi_json=info["lumijson"],
                    group=info["group"] if "group" in info else None,
                )
            )
        else:
            narf_info.update(
                dict(
                    xsec=info["xsec"],
                    group=info["group"] if "group" in info else None,
                )
            )
        narf_datasets.append(narf.Dataset(**narf_info))

    for sample in narf_datasets:
        if not sample.filepaths:
            logger.warning(f"Failed to find any files for sample {sample.name}!")

    return narf_datasets
