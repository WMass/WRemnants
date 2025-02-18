import os
import time

import h5py

from utilities import common
from wums import ioutils, logging, output_tools

logger = logging.child_logger(__name__)


def scale_to_data(result_dict):
    # scale histograms by lumi*xsec/sum(gen weights)
    time0 = time.time()

    lumi = [
        result["lumi"]
        for result in result_dict.values()
        if result["dataset"]["is_data"]
    ]
    if len(lumi) == 0:
        lumi = 1
    else:
        lumi = sum(lumi)

    logger.warning(f"Scale histograms with luminosity = {lumi} /fb")
    for d_name, result in result_dict.items():
        if result["dataset"]["is_data"]:
            continue

        xsec = result["dataset"]["xsec"]

        logger.debug(f"For dataset {d_name} with xsec={xsec}")

        scale = lumi * 1000 * xsec / result["weight_sum"]

        result["weight_sum"] = result["weight_sum"] * scale

        for h_name, histogram in result["output"].items():

            histo = histogram.get()

            histo *= scale

    logger.info(f"Scale to data: {time.time() - time0}")


def aggregate_groups(datasets, result_dict, groups_to_aggregate):
    # add members of groups together
    time0 = time.time()

    for group in groups_to_aggregate:

        dataset_names = [d.name for d in datasets if d.group == group]
        if len(dataset_names) == 0:
            continue

        logger.debug(f"Aggregate group {group}")

        resdict = None
        members = {}
        to_del = []
        for name, result in result_dict.items():
            if result["dataset"]["name"] not in dataset_names:
                continue

            logger.debug(f"Add {name}")

            for h_name, histogram in result["output"].items():
                if h_name in members.keys():
                    members[h_name].append(histogram.get())
                else:
                    members[h_name] = [histogram.get()]

            if resdict is None:
                resdict = {
                    "n_members": 1,
                    "dataset": {
                        "name": group,
                        "xsec": result["dataset"]["xsec"],
                        "filepaths": result["dataset"]["filepaths"],
                    },
                    "weight_sum": float(result["weight_sum"]),
                    "event_count": float(result["event_count"]),
                }
            else:
                resdict["dataset"]["xsec"] += result["dataset"]["xsec"]
                resdict["dataset"]["filepaths"] += result["dataset"]["filepaths"]
                resdict["n_members"] += 1
                resdict["weight_sum"] += float(result["weight_sum"])
                resdict["event_count"] += float(result["event_count"])

            to_del.append(name)

        output = {}
        for h_name, histograms in members.items():

            if len(histograms) != resdict["n_members"]:
                logger.warning(
                    f"There is a different number of histograms ({len(histograms)}) than original members {resdict['n_members']} for {h_name} from group {group}"
                )
                logger.warning("Summing them up probably leads to wrong behaviour")

            output[h_name] = ioutils.H5PickleProxy(sum(histograms))

        result_dict[group] = resdict
        result_dict[group]["output"] = output

        # delete individual datasets
        for name in to_del:
            del result_dict[name]

    logger.info(f"Aggregate groups: {time.time() - time0}")


def writeMetaInfoToRootFile(rtfile, exclude_diff="notebooks", args=None):
    import ROOT

    meta_dict = ioutils.make_meta_info_dict(exclude_diff, args=args, wd=common.base_dir)
    d = rtfile.mkdir("meta_info")
    d.cd()

    for key, value in meta_dict.items():
        out = ROOT.TNamed(str(key), str(value))
        out.Write()


def analysis_debug_output(results):
    logger.debug("")
    logger.debug("Unweighted (Weighted) events, before cut")
    logger.debug("-" * 30)
    for key, val in results.items():
        if "event_count" in val:
            logger.debug(
                f"Dataset {key.ljust(30)}:  {str(val['event_count']).ljust(15)} ({round(val['weight_sum'],1)})"
            )
            logger.debug("-" * 30)
    logger.debug("")


def write_analysis_output(results, outfile, args):
    analysis_debug_output(results)

    to_append = []
    if args.theoryCorr and not args.theoryCorrAltOnly:
        to_append.append(args.theoryCorr[0] + "Corr")
    if args.maxFiles is not None:
        to_append.append(f"maxFiles_{args.maxFiles}".replace("-", "m"))
    if len(args.pdfs) >= 1 and args.pdfs[0] != "ct18z":
        to_append.append(args.pdfs[0])
    if hasattr(args, "ptqVgen") and args.ptqVgen:
        to_append.append("vars_qtbyQ")

    if to_append and not args.forceDefaultName:
        outfile = outfile.replace(".hdf5", f"_{'_'.join(to_append)}.hdf5")

    if args.postfix:
        outfile = outfile.replace(".hdf5", f"_{args.postfix}.hdf5")

    if args.outfolder:
        if not os.path.exists(args.outfolder):
            logger.info(f"Creating output folder {args.outfolder}")
            os.makedirs(args.outfolder)
        outfile = os.path.join(args.outfolder, outfile)

    if args.appendOutputFile:
        outfile = args.appendOutputFile
        if os.path.isfile(outfile):
            logger.info(f"Analysis output will be appended to file {outfile}")
            open_as = "a"
        else:
            logger.warning(
                f"Analysis output requested to be appended to file {outfile}, but the file does not exist yet, it will be created instead"
            )
            open_as = "w"
    else:
        if os.path.isfile(outfile):
            logger.warning(
                f"Output file {outfile} exists already, it will be overwritten"
            )
        open_as = "w"

    time0 = time.time()
    with h5py.File(outfile, open_as) as f:
        for k, v in results.items():
            logger.debug(f"Pickle and dump {k}")
            ioutils.pickle_dump_h5py(k, v, f)

        if "meta_info" not in f.keys():
            ioutils.pickle_dump_h5py(
                "meta_info",
                output_tools.make_meta_info_dict(args=args, wd=common.base_dir),
                f,
            )

    logger.info(f"Writing output: {time.time()-time0}")
    logger.info(f"Output saved in {outfile}")

    return outfile
