"""
Functionality to load and prepare datasets with the focus on CMS NanoAOD or NanoGEN files
"""

import importlib
import os
import random

import ROOT
import XRootD.client

import narf
from wums import logging

logger = logging.child_logger(__name__)

# production tags of the custom NanoAOD per version, tried in this order
production_tags = {
    "v9": {
        "data": [
            "TrackFitV722_NanoProdv6",
            "TrackFitV722_NanoProdv5",
            "TrackFitV722_NanoProdv3",
        ],
        "mc": [
            "TrackFitV722_NanoProdv6",
            "TrackFitV722_NanoProdv5",
            "TrackFitV722_NanoProdv4",
            "TrackFitV722_NanoProdv3",
        ],
    },
    "v15": {"data": ["v1p0"], "mc": ["v1p0"]},
}

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
    nano_version="v9",
):
    filelist = []
    expandedPaths = []
    for orig_path in paths:
        if maxFiles > 0 and len(filelist) >= maxFiles:
            break
        # try each tag in order until files are found
        fallback = False
        for prod_tag in nano_prod_tags:
            format_args = dict(
                BASE_PATH=base_path,
                NANO_PROD_TAG=prod_tag,
                NANO_VERSION=nano_version.upper(),
                ERA=era,
            )

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


def getDataPath(nanoVersion="v9"):
    import socket

    # Explicit override wins (containers, custom mounts, CI runners).
    base_path = os.environ.get("WREMNANTS_DATA_PATH")
    if base_path:
        return base_path

    # In containers ``gethostname()`` returns just the short container
    # ID, so also check the FQDN via reverse-DNS — that's usually the
    # real ``<host>.cern.ch`` even from inside a Singularity image.
    short = socket.gethostname()
    fqdn = socket.getfqdn()
    names = (short, fqdn)

    if any(n.endswith(".cern.ch") for n in names):
        base_path = "/scratch/shared/NanoAOD"
    elif any(n.endswith(".mit.edu") for n in names):
        if nanoVersion == "v15":
            # output area of the NanoV15 productions (CRAB site T3_US_MIT)
            base_path = "/ceph/submit/data/group/cms/store/group/zmass/NanoAOD"
        else:
            base_path = "/scratch/submit/cms/wmass/NanoAOD"
    elif "cmsanalysis.pi.infn.it" in names:
        # NOTE: If anyone wants to run lowpu analysis at Pisa they'd probably want a different path
        base_path = "/scratchnvme/wmass/NANOV9/postVFP"
    elif "cmsasymow.pi.infn.it" in names:
        base_path = "/scratch/wmass/y2016"
    if nanoVersion == "v15" and not any(n.endswith(".mit.edu") for n in names):
        raise RuntimeError(
            f"getDataPath: no default location of the NanoAOD {nanoVersion} files for host "
            f"{short!r}. Set the WREMNANTS_DATA_PATH environment variable or pass base_path= "
            f"to getDatasets()."
        )
    elif base_path is None:
        raise RuntimeError(
            f"getDataPath: no default NanoAOD location for host "
            f"{short!r} (fqdn={fqdn!r}). Set the WREMNANTS_DATA_PATH "
            f"environment variable, pass base_path= to "
            f"getDatasets(), or add this host to "
            f"dataset_tools.getDataPath()."
        )
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
    data_tags=None,
    mc_tags=None,
    oneMCfileEveryN=None,
    checkFileForZombie=False,
    era="2016PostVFP",
    extended=False,
):

    if maxFiles is None or (isinstance(maxFiles, int) and maxFiles < -1):
        maxFiles = default_nfiles

    # versions without their own production tags (e.g. the low-PU v12) use the NanoV9 ones
    version_tags = production_tags.get(nanoVersion, production_tags["v9"])
    if data_tags is None:
        data_tags = version_tags["data"]
    if mc_tags is None:
        mc_tags = version_tags["mc"]

    if not base_path:
        base_path = getDataPath(nanoVersion)
    logger.info(f"Loading NanoAOD {nanoVersion} samples from {base_path}.")

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
            nano_version=nanoVersion,
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
