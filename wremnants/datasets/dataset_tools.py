import narf
from utilities import logging
import subprocess
import glob
import random

logger = logging.child_logger(__name__)

def buildXrdFileList(path, xrd):
    xrdpath = path[path.find('/store'):]
    logger.debug(f"Looking for path {xrdpath}")
    # xrdfs doesn't like wildcards, just use the mount if they are included
    if "*" not in path:
        f = subprocess.check_output(['xrdfs', f'root://{xrd}', 'ls', xrdpath]).decode(sys.stdout.encoding)
        return filter(lambda x: "root" in x[-4:], f.split())
    else:
        return [f"root://{xrd}/{f}" for f in glob.glob(path)]

#TODO add the rest of the samples!
def makeFilelist(paths, maxFiles=-1, format_args={}, is_data=False, oneMCfileEveryN=None):
    filelist = []
    for path in paths:
        if format_args:
            path = path.format(**format_args)
            logger.debug(f"Reading files from path {path}")
        filelist.extend(glob.glob(path) if path[:4] != "/eos" else buildXrdFileList(path, "eoscms.cern.ch"))

    if oneMCfileEveryN != None and not is_data:
        tmplist = []
        for i,f in enumerate(filelist):
            if i % oneMCfileEveryN == 0:
                tmplist.append(f)
        logger.warning(f"Using {len(tmplist)} files instead of {len(filelist)}")
        filelist = tmplist

    return filelist if maxFiles < 0 or len(filelist) < maxFiles else random.Random(1).sample(filelist, maxFiles)

def selectProc(selection, datasets):
    if any(selection == x.group for x in datasets):
        # if the selection matches any of the group names in the given dataset, the selection is applied to groups
        return list(filter(lambda x, s=selection: x.group is not None and x.group == s, datasets))
    else:
        # otherwise, the selection is applied to sample names
        return list(filter(lambda x, s=selection: s in x.name, datasets))

def selectProcs(selections, datasets):
    new_datasets = []
    for selection in selections:
        new_datasets += selectProc(selection, datasets)

    # remove duplicates selected by multiple filters
    new_datasets = list(set(new_datasets))
    return new_datasets

def filterProcs(filters, datasets):
    if filters:
        if isinstance(filters, list):
            new_datasets = selectProcs(filters, datasets)
        elif isinstance(filters, str):
            new_datasets = selectProc(filters, datasets)
        else:
            new_datasets = list(filter(filters, datasets))
    else:
        return datasets

    if len(new_datasets) == 0:
        logger.warning("Try to filter processes/groups but didn't find any match. Continue without filtering.")
        return datasets

    return new_datasets

def excludeProcs(excludes, datasets):
    if excludes:
        if isinstance(excludes, list):
            # remove selected datasets
            return list(filter(lambda x: x not in selectProcs(excludes, datasets), datasets))
        elif isinstance(excludes, str):
            # remove selected datasets
            return list(filter(lambda x: x not in selectProc(excludes, datasets), datasets))
        else:
            return list(filter(excludes, datasets))
    else:
        return datasets
