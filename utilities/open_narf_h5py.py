import argparse
import copy
import os
import re

import h5py
from hist import Hist

import wums
from wums import ioutils


def load_results_h5py(h5file):
    if "results" in h5file.keys():
        return ioutils.pickle_load_h5py(h5file["results"])
    else:
        return {k: ioutils.pickle_load_h5py(v) for k, v in h5file.items()}


parser = argparse.ArgumentParser(description="Read in a hdf5 file.")
parser.add_argument(
    "infile",
    type=str,
    help="hdf5 file.",
)
parser.add_argument(
    "--noListHists",
    action="store_true",
    help="Don't list all histograms, for all samples.",
)
parser.add_argument(
    "--printHists", action="store_true", help="Print histograms. (default=False)"
)
parser.add_argument(
    "--filterProcs",
    nargs="+",
    default=[],
    help="Filter processes to show info about. Supports multiple process names.",
)
parser.add_argument(
    "--filterProcsRegex",
    nargs="+",
    default=[],
    help="Filter processes to show info about using regex. Supports multiple regex patterns.",
)
parser.add_argument(
    "--excludeProcs",
    nargs="+",
    default=["meta_info"],
    help="Don't show info about these processes. Supports multiple process names.",
)
parser.add_argument(
    "--filterHists",
    nargs="+",
    type=str,
    default=None,
    help="Filter histograms to print. Supports one string that will be checked against all histogram names in the file.",
)
parser.add_argument(
    "--filterHistsRegex",
    nargs="+",
    type=str,
    default=None,
    help="Filter histograms to print using regex. Supports multiple regex patterns that will be checked against all histogram names in the file.",
)
parser.add_argument(
    "--excludeHists",
    nargs="+",
    type=str,
    default=None,
    help="Exclude histograms to print. Supports one string that will be checked against all histogram names in the file.",
)
parser.add_argument(
    "--outfile",
    type=str,
    default=None,
    help="If provided, will copy the processes and histograms selected into a new file.",
)
parser.add_argument(
    "--path",
    type=str,
    default=None,
    help="Navigate to the --path inside the HDF5 file.",
)
args = parser.parse_args()

with h5py.File(args.infile, "r") as h5file:
    results = load_results_h5py(h5file)
    print(f"Samples in file: {results.keys()}\n")

    if args.path:
        cwp = results
        for p in args.path.split("/"):
            print(f"Navigating to {p}")
            cwp = cwp[p]
        print(f"Contents at {args.path}: {cwp.keys()}")
        for k in cwp.keys():
            print(k, str(cwp[k])[:1000])

    if args.outfile:
        output = {}
        if "meta_info" in results.keys():
            output = copy.deepcopy(results["meta_info"])
        mode = "r+" if os.path.isfile(args.outfile) else "w"
        with h5py.File(args.outfile, mode) as h5out:
            wums.ioutils.pickle_dump_h5py("meta_info", output, h5out)

    if not args.noListHists:
        for sample in results.keys():
            if args.filterProcs and sample not in args.filterProcs:
                continue
            if args.filterProcsRegex:
                matched = False
                for pattern in args.filterProcsRegex:
                    if re.search(pattern, sample):
                        matched = True
                        break
                if not matched:
                    continue
            if args.excludeProcs and sample in args.excludeProcs:
                continue
            print(f"Sample: {sample}")

            if type(results[sample]) == dict:
                print(sample, results[sample].keys())
                hists = results[sample]["output"].keys()

                if args.filterHists:
                    hists = [
                        h for h in hists for filter in args.filterHists if filter in h
                    ]
                if args.excludeHists:
                    hists = [
                        h
                        for h in hists
                        for exclude in args.excludeHists
                        if exclude not in h
                    ]
                if args.filterHistsRegex:
                    filtered_hists = []
                    for h in hists:
                        for pattern in args.filterHistsRegex:
                            if re.search(pattern, h):
                                filtered_hists.append(h)
                                break
                    hists = filtered_hists

                print(f"Histograms: {hists}\n")
                if args.printHists:
                    for h in hists:
                        print(h, "\n", results[sample]["output"][h].get(), "\n")

                if args.outfile:
                    output = {}
                    for key in results[sample].keys():
                        if key != "output":
                            print(sample, key)
                            output[key] = copy.deepcopy(results[sample][key])
                    output["output"] = {}
                    for h in hists:
                        output["output"][h] = wums.ioutils.H5PickleProxy(
                            results[sample]["output"][h].get()
                        )

            elif type(results[sample]) == Hist:
                if args.printHists:
                    print(results[sample])

            if args.outfile:
                mode = "r+" if os.path.isfile(args.outfile) else "w"
                with h5py.File(args.outfile, mode) as h5out:
                    wums.ioutils.pickle_dump_h5py(sample, output, h5out)
                print("Wrote selected histograms to ", args.outfile)

            print()
