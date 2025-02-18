import argparse
import concurrent.futures
import copy

import boost_histogram as bh
import h5py

import wums.ioutils
from wums import logging

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="Input hdf5 file")
parser.add_argument(
    "-a", "--axis", required=True, type=str, help="axis name to split on"
)
parser.add_argument(
    "-s", "--start", type=int, default=0, help="start index for splitting"
)
parser.add_argument(
    "-e", "--end", required=True, type=int, help="end index for splitting"
)
parser.add_argument(
    "-p", "--postfix", type=str, help="Postfix for output file name", default=""
)
parser.add_argument(
    "--nominalData", action="store_true", help="use nominal data for all outputs"
)
args = parser.parse_args()
logger = logging.setup_logger(__file__)


def copy_and_slice(h5proxy, axis, idx):
    print("copy_and_slice")
    obj = h5proxy.get()
    print(type(obj))
    if isinstance(obj, bh.Histogram):
        print("is histogram")
        print(obj.axes.name)
        if axis in obj.axes.name:
            print("slicing")
            obj = obj[{axis: idx}]

    return wums.ioutils.H5PickleProxy(obj)


class DeepCopyOverride:
    def __init__(self, target_class, targetfn):
        self.target_class = target_class
        self.targetfn = targetfn
        self.oldfn = None

    def __enter__(self):
        if hasattr(self.target_class, "__deepcopy__"):
            self.oldfn = self.target_class.__deepcopy__
        self.target_class.__deepcopy__ = self.targetfn

    def __exit__(self, exc_type, exc_val, traceback):
        if self.oldfn:
            self.target_class.__deepcopy__ = self.oldfn
        else:
            del self.target_class.__deepcopy__


newfiles = True

# def copy_and_write(key, value, isplit)

with h5py.File(args.input, "r") as h5file:
    keys = list(h5file.keys())

print(keys)


for key in keys:

    with h5py.File(args.input, "r") as h5file:
        res = wums.ioutils.pickle_load_h5py(h5file[key])
        # print("res.keys", res.keys())
        # print(res["dataset"].keys())
        # quit()

        # preload all the proxied objects so we can close the file
        # TODO should this be an option in pickle_load_h5py directly?
        with DeepCopyOverride(
            wums.ioutils.H5PickleProxy,
            lambda h5proxy, memo: wums.ioutils.H5PickleProxy(h5proxy.get()),
        ):
            copy.deepcopy(res)

    def copy_and_write(isplit):

        # for isplit in range(args.start, args.end):
        print("isplit", isplit)
        # wums.ioutils.H5PickleProxy.__deepcopy__ = lambda h5proxy, memo : copy_and_slice(h5proxy, args.axis, isplit)
        # outres = copy.deepcopy(res)
        # del wums.ioutils.H5PickleProxy.__deepcopy__

        if args.nominalData and "dataset" in res and res["dataset"]["is_data"]:
            isplitsource = 0
        else:
            isplitsource = isplit

        with DeepCopyOverride(
            wums.ioutils.H5PickleProxy,
            lambda h5proxy, memo: copy_and_slice(h5proxy, args.axis, isplitsource),
        ):
            outres = copy.deepcopy(res)

        outfile = args.input.replace(
            ".hdf5", f"{args.postfix}_{args.axis}_{isplit}.hdf5"
        )
        mode = "w" if newfiles else "r+"

        print("outfile", outfile, mode)

        with h5py.File(outfile, mode) as h5out:
            wums.ioutils.pickle_dump_h5py(key, outres, h5out)

        outres = None

        return True

    with concurrent.futures.ProcessPoolExecutor(max_workers=64) as executor:
        for res in executor.map(copy_and_write, range(args.start, args.end)):
            pass

    res = None
    newfiles = False
