import argparse
import copy

import h5py

import narf
from utilities import logging

parser = argparse.ArgumentParser()
parser.add_argument("infiles", type=str, nargs="+", help="Input hdf5 files")
parser.add_argument(
    "-p", "--postfix", type=str, help="Postfix for output file name", default="merged"
)
parser.add_argument("-o", "--outfolder", type=str, default="./", help="Output folder")
args = parser.parse_args()
logger = logging.setup_logger(__file__)


def recursive_copy(res):
    if type(res) == dict:
        return {k: recursive_copy(v) for k, v in res.items()}
    elif type(res) in [type(None), bool, int, float, str]:
        return res
    elif type(res) in [
        list,
    ]:
        return copy.deepcopy(res)
    elif type(res) == narf.ioutils.H5PickleProxy:
        # need to copy the underlying object, then make a proxy again for lazy reading of output file
        return narf.ioutils.H5PickleProxy(res.get())
    else:
        raise TypeError(f"Unknown type {type(res)} of object {res}")


results = {}
for i, infile in enumerate(args.infiles):
    logger.info(f"Now at file {infile}")
    h5file = h5py.File(infile, "r")
    result = narf.ioutils.pickle_load_h5py(h5file["results"])

    for key, value in result.items():
        logger.info(f"Copying {key} ...")
        if key in results.keys():
            if key == "meta_info":
                results[f"meta_info_{i}"] = copy.deepcopy(value)
            else:
                raise NotImplementedError(
                    f"The object with key {key} from file {infile} was already present in a previous file; No implementation to solve this conflict"
                )
        else:
            results[key] = recursive_copy(value)

outfile = args.infiles[0].split("/")[-1]
if args.postfix:
    outfile = outfile.replace(
        ".hdf5", f"_{args.postfix}.hdf5" if args.postfix else ".hdf5"
    )

logger.info(f"Writing merged results into output file {args.outfolder}/{outfile}")
with h5py.File(f"{args.outfolder}/{outfile}", "w") as f:
    narf.ioutils.pickle_dump_h5py("results", results, f)
