"""
Reading and writing functions using the ROOT library
"""

import ROOT

from wums import logging

logger = logging.child_logger(__name__)


def safeGetRootObject(
    fileObject, objectName, quitOnFail=True, silent=False, detach=True
):
    obj = fileObject.Get(objectName)
    if obj is None:
        error_msg = f"Error getting {objectName} from file {fileObject.GetName()}"
        if not silent:
            logger.error(error_msg)
        if quitOnFail:
            raise IOError(error_msg)
        return None
    else:
        if detach:
            obj.SetDirectory(0)
        return obj


def safeOpenRootFile(fileName, quitOnFail=True, silent=False, mode="READ"):
    fileObject = ROOT.TFile.Open(fileName, mode)
    if not fileObject or fileObject.IsZombie():
        error_msg = f"Error when opening file {fileName}"
        if not silent:
            logger.error(error_msg)
        if quitOnFail:
            raise IOError(error_msg)
        else:
            return None
    elif not fileObject.IsOpen():
        error_msg = f"File {fileName} was not opened"
        if not silent:
            logger.error(error_msg)
        if quitOnFail:
            raise IOError(error_msg)
        else:
            return None
    else:
        return fileObject
