import importlib
import os
import random

import ROOT
import XRootD.client

import narf
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


def buildFileListPosix(path):
    outfiles = []
    for root, dirs, fnames in os.walk(path):
        for fname in fnames:
            if fname.lower().endswith(".root"):
                outfiles.append(f"{root}/{fname}")

    return outfiles


def appendFilesXrd(
    filelist, xrdfs, path, suffixes=[".root"], recurse=False, num_clients=16
):
    status, dirlist = xrdfs.dirlist(path, flags=XRootD.client.flags.DirListFlags.STAT)

    if not status.ok:
        if status.code == 400 and status.errno == 3011:
            logger.warning(f"XRootD directory not found: {path}")
        else:
            raise RuntimeError(
                f"Error in XRootD.client.FileSystem.dirlist: {status.message}, {status.code}, {status.errno}"
            )

        return

    for diritem in dirlist:
        is_dir = diritem.statinfo.flags & XRootD.client.flags.StatInfoFlags.IS_DIR
        is_other = diritem.statinfo.flags & XRootD.client.flags.StatInfoFlags.OTHER
        is_file = not (is_dir or is_other)

        if is_dir and recurse:
            childpath = f"{path}/{diritem.name}"
            appendFilesXrd(
                filelist,
                xrdfs,
                childpath,
                suffixes=suffixes,
                recurse=recurse,
                num_clients=num_clients,
            )
        elif is_file:
            lowername = diritem.name.lower()
            matchsuffix = False
            for suffix in suffixes:
                if lowername.endswith(suffix):
                    matchsuffix = True
                    break

            if matchsuffix:
                if num_clients > 0:
                    # construct client string if necessary to force multiple xrootd connections
                    # (needed for good performance when a single or small number of xrootd servers is used)
                    client = f"user_{random.randrange(num_clients)}"
                    outname = f"{xrdfs.url.protocol}://{client}@{xrdfs.url.hostname}:{xrdfs.url.port}/{path}/{diritem.name}"
                else:
                    outname = f"{xrdfs.url.protocol}://{xrdfs.url.hostid}/{path}/{diritem.name}"

                filelist.append(outname)


def buildFileListXrd(path, num_clients=16):
    xrdurl = XRootD.client.URL(path)

    if not xrdurl.is_valid():
        raise ValueError(f"Invalid xrootd path {path}")

    xrdfs = XRootD.client.FileSystem(xrdurl.hostid)
    xrdpath = xrdurl.path

    outfiles = []
    appendFilesXrd(outfiles, xrdfs, xrdpath, recurse=True, num_clients=num_clients)

    return outfiles


def buildFileList(path):
    xrdprefix = "root://"
    return (
        buildFileListXrd(path)
        if path.startswith(xrdprefix)
        else buildFileListPosix(path)
    )


# TODO add the rest of the samples!
def makeFilelist(
    paths,
    maxFiles=-1,
    base_path=None,
    nano_prod_tags=None,
    is_data=False,
    oneMCfileEveryN=None,
    era=None,
):
    filelist = []
    expandedPaths = []
    for orig_path in paths:
        if maxFiles > 0 and len(filelist) >= maxFiles:
            break
        # try each tag in order until files are found
        fallback = False
        for prod_tag in nano_prod_tags:
            format_args = dict(BASE_PATH=base_path, NANO_PROD_TAG=prod_tag, ERA=era)

            path = orig_path.format(**format_args)
            expandedPaths.append(path)
            logger.debug(f"Reading files from path {path}")

            files = buildFileList(path)
            if maxFiles > 0 and len(files) >= maxFiles:
                logger.info(
                    f"Booking {maxFiles} of {len(files)} files with tag {prod_tag} with path {path}"
                )
                break

            if len(files) == 0:
                fallback = True
                logger.warning(
                    f"Did not find any files for tag {prod_tag} matching path {path}!"
                )
            else:
                if fallback:
                    logger.warning(f"Falling back to tag {prod_tag} with path {path}")
                else:
                    logger.info(
                        f"Booking {maxFiles} of {len(files)} files with tag {prod_tag} with path {path}"
                    )
                break

        filelist.extend(files)

    toreturn = (
        filelist
        if maxFiles < 0 or len(filelist) < maxFiles
        else random.Random(1).sample(filelist, maxFiles)
    )

    if oneMCfileEveryN != None and not is_data:
        tmplist = []
        for i, f in enumerate(toreturn):
            if i % oneMCfileEveryN == 0:
                tmplist.append(f)
        logger.warning(f"Using {len(tmplist)} files instead of {len(toreturn)}")
        toreturn = tmplist

    logger.debug(f"Length of list is {len(toreturn)} for paths {expandedPaths}")
    return toreturn


def getDataPath():
    import socket

    hostname = socket.gethostname()

    if hostname.endswith(".cern.ch"):
        base_path = "/scratch/shared/NanoAOD"
    elif hostname.endswith(".mit.edu"):
        base_path = "/scratch/submit/cms/wmass/NanoAOD"
    elif hostname == "cmsanalysis.pi.infn.it":
        # NOTE: If anyone wants to run lowpu analysis at Pisa they'd probably want a different path
        base_path = "/scratchnvme/wmass/NANOV9/postVFP"
    elif hostname == "cmsasymow.pi.infn.it":
        base_path = "/scratch/wmass/y2016"
    return base_path


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

    module = importlib.import_module(f"wremnants.datasets.datasetDict_{era}")
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
        if filter not in [None, []] and (info["group"] in filt or sample in filt):
            # keep the sample if it is explicitly filtered
            pass
        elif info.get("auxiliary") and (
            aux in [None, []] or (info["group"] not in aux and sample not in aux)
        ):
            # skip if it is not explicitly filtered, auxiliary, and not specified by aux
            continue

        if sample in dataDict_NanoGen:
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
