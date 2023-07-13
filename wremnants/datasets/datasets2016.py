import narf
from utilities import logging
import subprocess
import glob
import pathlib
import socket
#set the debug level for logging incase of full printout 
from wremnants.datasets.datasetDict_v9 import dataDictV9
from wremnants.datasets.datasetDict_v8 import dataDictV8
from wremnants.datasets.datasetDict_gen import genDataDict

from wremnants.datasets.dataset_tools import filterProcs, excludeProcs, makeFilelist

logger = logging.child_logger(__name__)

lumicsv = f"{pathlib.Path(__file__).parent.parent}/data/bylsoutput.csv"
lumijson = f"{pathlib.Path(__file__).parent.parent}/data/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"

def getDatasets(maxFiles=-1, filt=None, excl=None, mode=None, base_path=None, nanoVersion="v9", 
                data_tag="TrackFitV722_NanoProdv2", mc_tag="TrackFitV718_NanoProdv1", oneMCfileEveryN=None):
    # useMCfraction is an integer > 1 to use 1 file every useMCfraction
    if not base_path:
        hostname = socket.gethostname()
        if hostname == "lxplus8s10.cern.ch":
            base_path = "/scratch/shared/NanoAOD"
        if hostname == "cmswmass2.cern.ch":
            base_path = "/data/shared/NanoAOD"
        elif "mit.edu" in hostname:
            base_path = "/scratch/submit/cms/wmass/NanoAOD"
        elif hostname == "cmsanalysis.pi.infn.it":
            base_path = "/scratchnvme/wmass/NANOV9/postVFP"

    logger.info(f"Loading samples from {base_path}.")

    if nanoVersion == "v8":
        dataDict = dataDictV8
        logger.info('Using NanoAOD V8')
    elif nanoVersion == "v9":
        dataDict = dataDictV9
    else:
        raise ValueError("Only NanoAODv8 and NanoAODv9 are supported")

    if mode == "gen":
        dataDict.update(genDataDict)

    narf_datasets = []
    for sample,info in dataDict.items():
        if sample in genDataDict:
            base_path = base_path.replace("NanoAOD", "NanoGen")

        is_data = "data" in sample[:4]

        prod_tag = data_tag if is_data else mc_tag 
        paths = makeFilelist(info["filepaths"], maxFiles, format_args=dict(BASE_PATH=base_path, NANO_PROD_TAG=prod_tag), is_data=is_data, oneMCfileEveryN=oneMCfileEveryN)

        if not paths:
            logger.warning(f"Failed to find any files for dataset {sample}. Looking at {info['filepaths']}. Skipping!")
            continue

        narf_info = dict(
            name=sample,
            filepaths=paths,
        )

        if is_data:
            if mode == "gen":
                continue
            narf_info.update(dict(
                is_data=True,
                lumi_csv=lumicsv,
                lumi_json=lumijson,
                group=info["group"] if "group" in info else None,
            ))
        else:
            narf_info.update(dict(
                xsec=info["xsec"],
                group=info["group"] if "group" in info else None,
                )
            )
        narf_datasets.append(narf.Dataset(**narf_info))
    
    narf_datasets = filterProcs(filt, narf_datasets)
    narf_datasets = excludeProcs(excl, narf_datasets)

    for sample in narf_datasets:
        if not sample.filepaths:
            logger.warning(f"Failed to find any files for sample {sample.name}!")

    return narf_datasets

