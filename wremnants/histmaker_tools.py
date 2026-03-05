import os
import time

import h5py
import hist
import numpy as np
import ROOT

import narf
from utilities import common
from utilities.io_tools import input_tools
from wums import boostHistHelpers as hh
from wums import ioutils, logging, output_tools

narf.clingutils.Declare('#include "histHelpers.hpp"')

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
        to_append.append(args.theoryCorr[0] + "_Corr")
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
            ioutils.pickle_dump_h5py(k, v, f, override=open_as != "w")

        if "meta_info" not in f.keys():
            ioutils.pickle_dump_h5py(
                "meta_info",
                output_tools.make_meta_info_dict(args=args, wd=common.base_dir),
                f,
            )

    logger.info(f"Writing output: {time.time()-time0}")
    logger.info(f"Output saved in {outfile}")

    return outfile


def get_run_lumi_edges(nRunBins, era):
    if era == "2016PostVFP":
        if nRunBins == 2:
            run_edges = [278768, 280385, 284044]
            lumi_edges = [0.0, 0.48013, 1.0]
        elif nRunBins == 3:
            run_edges = [278768, 279767, 283270, 284044]
            lumi_edges = [0.0, 0.25749, 0.72954, 1.0]
        elif nRunBins == 4:
            run_edges = [278768, 279767, 280385, 283270, 284044]
            lumi_edges = [0.0, 0.25749, 0.48013, 0.72954, 1.0]
        elif nRunBins == 5:
            run_edges = [278768, 279588, 280017, 282037, 283478, 284044]
            lumi_edges = [0.0, 0.13871, 0.371579, 0.6038544, 0.836724, 1.0]
        else:
            raise NotImplementedError(
                f"Invalid number of bins ({nRunBins}) passed to --nRunBins."
            )
    else:
        raise NotImplementedError(
            f"Function get_run_lumi_edges() does not yet support era {era}."
        )
    return run_edges, lumi_edges


def make_quantile_helper(
    filename,
    axes,
    dependent_axes=[],
    name="nominal",
    processes=["Zmumu_2016PostVFP"],
    n_quantiles=[],
):
    """
    Helper to compute the quantile for `axes` from fine binned histogram with `name` in bins of the dependent axes
    The helper takes colums for `axes` and `dependent_axes` and returns the quantile the event falls as a fraction of 1
    If quantiles are performed in more than 1 dimension, the number of quantiles in the n lower dimensions must be given in 'n_quantiles'
    """

    h5file = h5py.File(filename, "r")
    results = input_tools.load_results_h5py(h5file)

    hIn = hh.sumHists(results[p]["output"][name].get() for p in processes)

    if isinstance(axes, str):
        axes = [axes]

    def hist_to_helper(h):
        hConv = narf.hist_to_pyroot_boost(h, tensor_rank=0)
        helper = ROOT.narf.make_hist_helper(ROOT.std.move(hConv))
        helper.hist = h
        helper.axes = h.axes
        return helper

    def cdf(arr):
        cdf_arr = np.cumsum(arr, axis=0)
        # Normalize to get values between 0 and 1
        slices_norm = [-1 if i == 0 else slice(None) for i in range(len(arr.shape))]
        slices_bc = [
            np.newaxis if i == 0 else slice(None) for i in range(len(arr.shape))
        ]
        cdf_arr /= cdf_arr[*slices_norm][*slices_bc]
        # if there are completely empty slices, set them to 0
        cdf_arr = np.nan_to_num(cdf_arr, nan=0)
        # first or last bin(s) could be negative, ensure values between 0 and 1
        cdf_arr = np.minimum(1, np.maximum(0, cdf_arr))
        return cdf_arr

    hIn = hIn.project(*axes, *dependent_axes)

    helpers = []
    if len(axes) in [1, 2]:
        # make 1D quantiles
        hFirst = hIn.project(axes[-1], *dependent_axes)
        cdf_arr = cdf(hFirst.values(flow=True))

        hFirstOut = hist.Hist(*hFirst.axes, storage=hist.storage.Double())
        hFirstOut.values(flow=True)[...] = cdf_arr

        helpers.append(hist_to_helper(hFirstOut))

        cdf_arr_second = np.empty(hIn.values(flow=True).shape)
        if len(axes) == 2:
            n = n_quantiles[-1]

            # make 2D quantiles
            if len(hIn.axes[axes[-1]]) % n != 0:
                raise RuntimeError(
                    f"Can not make {n} quantiles from axis with {len(hIn.axes[axes[-1]])} bins"
                )

            for i in range(n):
                lo = i / n
                hi = (i + 1) / n
                if hi == 1:
                    mask = cdf_arr >= lo
                else:
                    mask = (cdf_arr >= lo) & (cdf_arr < hi)

                arr = hIn.values(flow=True).copy()
                if mask is not None:
                    arr[:, ~mask] = 0
                    arr = np.sum(arr, axis=1)

                arr = cdf(arr)[:, np.newaxis, ...]
                arr = np.broadcast_to(arr, cdf_arr_second.shape)
                mask_out = np.broadcast_to(mask[np.newaxis, ...], cdf_arr_second.shape)
                cdf_arr_second = np.where(mask_out, arr, cdf_arr_second)

            hSecondOut = hist.Hist(*hIn.axes, storage=hist.storage.Double())
            hSecondOut.values(flow=True)[...] = cdf_arr_second

            helpers.append(hist_to_helper(hSecondOut))
    else:
        raise NotImplementedError(
            f"Making quantiles in {len(axes)} dimensions is not implemented."
        )

    return helpers


def make_muon_phi_axis(phi_bins, ax_name="phi", flows=False):
    # TODO: worth having different axis type (e.g. Regular with
    # circular=True) depending on the list of edges?
    if isinstance(phi_bins, int) or len(phi_bins) == 1:
        nphi = phi_bins if isinstance(phi_bins, int) else int(phi_bins[0])
        phi_width = 2.0 / nphi
        phi_edges = [(-1.0 + i * phi_width) * np.pi for i in range(nphi + 1)]
    else:
        phi_edges = [x for x in phi_bins]

    phi_axis = hist.axis.Variable(
        np.array(phi_edges), name=ax_name, underflow=flows, overflow=flows
    )

    return phi_axis


def define_norm_weight_nRecoVtx(
    df, vtx_axis_edges, vtx_norm_weight, flows_to_unit=False
):
    df = df.DefinePerSample(
        "nRecoVtxEdges",
        "ROOT::VecOps::RVec<double> res = {"
        + ",".join([str(x) for x in vtx_axis_edges])
        + "}; return res;",
    )
    df = df.DefinePerSample(
        "weightVals",
        "ROOT::VecOps::RVec<double> res = {"
        + ",".join([str(x) for x in vtx_norm_weight])
        + "}; return res;",
    )
    df = df.Define(
        "weight_nRecoVtx",
        f"wrem::get_differential_norm_weight(PV_npvsGood, nRecoVtxEdges, weightVals, {flows_to_unit})",
    )
    return df
