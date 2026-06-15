import argparse
import copy
import re
from contextlib import ExitStack

import h5py
from hist import Hist

import wums
from wremnants.utilities.io_tools import base_io

parser = argparse.ArgumentParser(description="Read in a hdf5 file.")
parser.add_argument(
    "infile",
    type=str,
    help="hdf5 file.",
)
parser.add_argument(
    "--fileUpdate",
    type=str,
    default=None,
    help="hdf5 file where objects should be taken from. Histograms with the same name override those in infile; histograms present only in infile are preserved.",
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


def merge_results(results, resUps):
    """Merge resUps into results at the histogram level.

    For samples present in both, histograms in resUps override those in results
    and other (non-output) keys from resUps take precedence; histograms only in
    results are preserved. Samples present only in one side are kept as-is.
    """
    for sample, sdata in resUps.items():
        if (
            sample in results
            and isinstance(results[sample], dict)
            and isinstance(sdata, dict)
        ):
            for key, val in sdata.items():
                if key == "output":
                    continue
                results[sample][key] = val
            if "output" in sdata:
                results[sample].setdefault("output", {})
                for hname, hval in sdata["output"].items():
                    results[sample]["output"][hname] = hval
        else:
            results[sample] = sdata
    return results


with ExitStack() as stack:
    h5file = stack.enter_context(h5py.File(args.infile, "r"))
    results = base_io.load_results_h5py(h5file)
    print(f"Samples in file: {results.keys()}\n")

    if args.fileUpdate:
        h5upd = stack.enter_context(h5py.File(args.fileUpdate, "r"))
        resUps = base_io.load_results_h5py(h5upd)
        print(f"Samples in update file: {resUps.keys()}\n")
        merge_results(results, resUps)

    if args.path:
        cwp = results
        for p in args.path.split("/"):
            print(f"Navigating to {p}")
            cwp = cwp[p]
        print(f"Contents at {args.path}: {cwp.keys()}")
        for k in cwp.keys():
            print(k, str(cwp[k])[:1000])

    h5out = None
    if args.outfile:
        h5out = stack.enter_context(h5py.File(args.outfile, "w"))
        if "meta_info" in results.keys():
            wums.ioutils.pickle_dump_h5py(
                "meta_info", copy.deepcopy(results["meta_info"]), h5out, override=True
            )

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

            output = None
            if type(results[sample]) == dict:
                print(sample, results[sample].keys())
                hists = list(results[sample]["output"].keys())

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

                if h5out is not None:
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
                if h5out is not None:
                    output = copy.deepcopy(results[sample])

            if h5out is not None and output is not None:
                wums.ioutils.pickle_dump_h5py(sample, output, h5out, override=True)
                print("Wrote selected histograms for sample", sample)

            print()

    if h5out is not None:
        print(f"Wrote output file: {args.outfile}")
