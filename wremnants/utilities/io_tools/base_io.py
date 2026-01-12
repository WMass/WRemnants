"""
Only the basic input/output functions to avoid loading of heavy libraries.
Root dependent io functions are in root.io.
More specific input functions are in input_tools.py
More specific output functions are in input_tools.py
"""

import pickle

import h5py
import lz4.frame

from wums import ioutils, logging

logger = logging.child_logger(__name__)


def load_results_h5py(h5file):
    if "results" in h5file.keys():
        return ioutils.pickle_load_h5py(h5file["results"])
    else:
        return {k: ioutils.pickle_load_h5py(v) for k, v in h5file.items()}


def get_metadata(infile):
    results = None
    if infile.endswith(".pkl.lz4"):
        with lz4.frame.open(infile) as f:
            results = pickle.load(f)
    elif infile.endswith(".pkl"):
        with open(infile, "rb") as f:
            results = pickle.load(f)
    elif infile.endswith(".hdf5"):
        h5file = h5py.File(infile, "r")
        if "meta_info" in h5file.keys():
            return ioutils.pickle_load_h5py(h5file["meta_info"])
        meta = h5file.get("meta", h5file.get("results", None))
        results = ioutils.pickle_load_h5py(meta) if meta else None

    if results is None:
        logger.warning(
            "Failed to find results dict. Note that only pkl, hdf5, and pkl.lz4 file types are supported"
        )
        return None

    return results.get("meta_info", results.get("meta_data", results.get("meta", None)))
