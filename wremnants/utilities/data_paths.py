"""
Site-dependent data paths and dataset file-list construction.

This module is importable WITHOUT ROOT, narf, or XRootD. It deliberately lives
under ``wremnants.utilities`` rather than ``wremnants.production.datasets``:
the ``wremnants.production`` package ``__init__`` imports ROOT and narf, so any
module below it pulls them in at import time. Code that only needs the
per-host data path (e.g. the SCETlib NP ParamModel, which runs inside a fit
where a mid-fit ROOT import can segfault) should import from here.

``wremnants.production.datasets.dataset_tools`` re-exports everything defined
here, so existing call sites keep working.

XRootD is imported lazily, only when a ``root://`` path is actually listed.
"""

import os
import random
import socket

from wums import logging

logger = logging.child_logger(__name__)


def getDataPath(fallback=None):
    """NanoAOD base directory for the current host.

    Raises ValueError on an unknown host unless ``fallback`` is given, in
    which case that path is returned instead.
    """
    hostname = socket.gethostname()

    if hostname.endswith(".cern.ch"):
        return "/scratch/shared/NanoAOD"
    elif hostname.endswith(".mit.edu"):
        return "/scratch/submit/cms/wmass/NanoAOD"
    elif hostname == "cmsanalysis.pi.infn.it":
        # NOTE: If anyone wants to run lowpu analysis at Pisa they'd probably want a different path
        return "/scratchnvme/wmass/NANOV9/postVFP"
    elif hostname == "cmsasymow.pi.infn.it":
        return "/scratch/wmass/y2016"

    if fallback is not None:
        logger.warning(
            f"No data path known for host {hostname}, falling back to {fallback}"
        )
        return fallback
    raise ValueError(
        f"No data path known for host {hostname}; pass an explicit base path"
    )


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
    import XRootD.client

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
    import XRootD.client

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
