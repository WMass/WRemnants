import json
import os
import pickle
import re

import h5py
import hist
import lz4.frame
import numpy as np
import ROOT
import uproot

from wums import boostHistHelpers as hh
from wums import ioutils, logging

logger = logging.child_logger(__name__)

scetlib_tnp_match_expr = [
    r"^gamma_.*[+|-]\d+",
    r"^b_.*[+|-]\d+",
    r"^s[+|-]\d+",
    r"^h_.*\d+",
]


def load_results_h5py(h5file):
    if "results" in h5file.keys():
        return ioutils.pickle_load_h5py(h5file["results"])
    else:
        return {k: ioutils.pickle_load_h5py(v) for k, v in h5file.items()}


def read_and_scale_pkllz4(fname, proc, histname, calculate_lumi=False, scale=1):
    with lz4.frame.open(fname) as f:
        results = pickle.load(f)

    return load_and_scale(results, proc, histname, calculate_lumi, scale)


def read_hist_names(fname, proc):
    with h5py.File(fname, "r") as h5file:
        results = load_results_h5py(h5file)
        if proc not in results:
            raise ValueError(f"Invalid process {proc}! No output found in file {fname}")
        return results[proc]["output"].keys()


def read_keys(fname):
    with h5py.File(fname, "r") as h5file:
        results = load_results_h5py(h5file)
        return results.keys()


def read_xsec(fname, proc):
    with h5py.File(fname, "r") as h5file:
        results = load_results_h5py(h5file)
        return results[proc]["dataset"]["xsec"]


def read_sumw(fname, proc):
    with h5py.File(fname, "r") as h5file:
        results = load_results_h5py(h5file)
        return results[proc]["weight_sum"]


def read_and_scale(
    fname, proc, histname, calculate_lumi=False, scale=1, apply_xsec=True
):
    with h5py.File(fname, "r") as h5file:
        results = load_results_h5py(h5file)

        return load_and_scale(
            results, proc, histname, calculate_lumi, scale, apply_xsec
        )


def load_and_scale(
    res_dict, proc, histname, calculate_lumi=False, scale=1.0, apply_xsec=True
):
    h = res_dict[proc]["output"][histname]
    if isinstance(h, ioutils.H5PickleProxy):
        h = h.get()
    if not res_dict[proc]["dataset"]["is_data"]:
        if apply_xsec:
            scale = (
                res_dict[proc]["dataset"]["xsec"] / res_dict[proc]["weight_sum"] * scale
            )
        if calculate_lumi:
            data_keys = [
                p
                for p in res_dict.keys()
                if "dataset" in res_dict[p]
                and "is_data" in res_dict[p]["dataset"]
                and res_dict[p]["dataset"]["is_data"]
            ]
            lumi = sum([res_dict[p]["lumi"] for p in data_keys]) * 1000
            if not lumi:
                logger.warning(
                    "Did not find a data hist! Skipping calculate_lumi option"
                )
                lumi = 1
            scale *= lumi
    return h * scale


def read_all_and_scale(fname, procs, histnames, lumi=False):
    h5file = h5py.File(fname, "r")
    results = load_results_h5py(h5file)

    hists = []
    for histname in histnames:
        h = load_and_scale(results, procs[0], histname, lumi)
        for proc in procs[1:]:
            h += load_and_scale(results, proc, histname, lumi)
        hists.append(h)

    return hists


def read_scetlib_hist(path, nonsing="none", flip_y_sign=False, charge=None):
    if path[-4:] == ".npz":
        f = np.load(path, allow_pickle=True)
    elif path[-4:] == ".pkl":
        with open(path, "rb") as picklefile:
            f = pickle.load(picklefile)
    else:
        ValueError("File {path} is not a recognized file format")

    if type(f["hist"]) == hist.Hist:
        scetlibh = f["hist"]
    else:
        var_axis = hist.axis.Integer(
            f["bins"][0][0], f["bins"][0][-1], name="vars", flow=False
        )
        mass_axis = hist.axis.Variable(f["bins"][1], name="Q", flow=False)
        y_axis = hist.axis.Variable(f["bins"][2], name="Y", flow=False)

        # Use 0.1 here rather than 0, because the nonsingular behaves much better with a "cut" at > 0.1
        pt_axis = hist.axis.Variable(f["bins"][3], name="qT", flow=False)

        h = f["hist"]
        storage = hist.storage.Double()
        axes = [mass_axis, y_axis, pt_axis, var_axis]

        varax_idx = -1
        vals = np.moveaxis(h, 0, varax_idx)

        if "hist_err" in f:
            err = f["hist_err"]
            storage = hist.storage.Weight()
            vals = np.stack((vals, np.moveaxis(err, 0, varax_idx)), axis=-1)

        scetlibh = hist.Hist(*axes, storage=storage, data=vals)

    if charge is not None:
        scetlibh = add_charge_axis(scetlibh, charge)

    if nonsing and nonsing != "none":
        if nonsing == "auto":
            nonsing = path.replace(
                *((".", "_nons.") if "sing" not in path else ("sing", "nons"))
            )
        nonsingh = read_scetlib_hist(
            nonsing, nonsing="none", flip_y_sign=flip_y_sign, charge=charge
        )
        # The overflow in the categorical axis breaks the broadcast
        # FIXME: Only set central for variations that aren't present
        if "vars" in nonsingh.axes.name and nonsingh.axes["vars"].size == 1:
            nonsingh = nonsingh[{"vars": 0}]
        scetlibh = hh.addHists(scetlibh, nonsingh)
        logger.warning("Adding NLO nonsingular contribution!")
    elif nonsing != "none":
        logger.warning("Will not include nonsingular contribution!")

    if flip_y_sign:
        scetlibh = flip_hist_y_sign(scetlibh)

    return scetlibh


def flip_hist_y_sign(h, yaxis="Y"):
    centers = h.axes[yaxis].centers
    scale = np.ones_like(centers)
    scale[centers < 0] *= -1
    h.values()[...] = (
        h.values()
        * scale[(None if ax.name != yaxis else slice(0, ax.size) for ax in h.axes)]
    )
    return h


def read_dyturbo_vars_hist(base_name, var_axis=None, axes=("Y", "qT"), charge=None):

    # map from scetlib fo variations naming to dyturbo naming
    # *FIXME* this is sensitive to presence or absence of trailing zeros for kappas
    # NOTE: kappaFO varies muR and muF together, muf varies only muF
    scales_map = {
        "pdf0": "mur1-muf1",
        "kappaFO0.5-kappaf2.": "mur0p5-muf1",
        "kappaFO2.-kappaf0.5": "mur2-muf1",
        "kappaf0.5": "mur1-muf0p5",
        "kappaf2.": "mur1-muf2",
        "kappaFO0.5": "mur0p5-muf0p5",
        "kappaFO2.": "mur2-muf2",
    }

    var_hist = None
    if var_axis is None:
        var_axis = hist.axis.StrCategory(list(scales_map.keys()), name="vars")

    for i, var in enumerate(var_axis):
        if var.startswith("pdf"):
            index = var.removeprefix("pdf")
            if index.isnumeric():
                pdf_member = int(index)
            else:
                pdf_member = i
        else:
            pdf_member = 0
        if var in scales_map.keys() and var not in scales_map:
            raise ValueError(
                f"Scale variation {var} found for fo_sing piece but no corresponding variation for dyturbo"
            )
        dyturbo_scale = scales_map.get(var, "mur1-muf1")
        dyturbo_name = base_name.format(i=pdf_member, scale=dyturbo_scale)
        h = read_dyturbo_hist([dyturbo_name], axes=axes, charge=charge)
        if not var_hist:
            var_hist = hist.Hist(*h.axes, var_axis, storage=h._storage_type())
        var_hist[..., i] = h.view()

    return var_hist


def read_nnlojet_file(
    filename, axnames=["qT"], all_scales=True, other_axes=[], charge=None
):
    data = read_text_data(filename)

    edges = np.append(data[:, 0], data[-1, 2])
    step = data[0, 2] - data[0, 0]
    if np.all(edges[1:] - edges[:-1] == step):
        ax = hist.axis.Regular(
            len(edges) - 1, edges[0], edges[-1], name=axnames[0], flow=False
        )
    else:
        ax = hist.axis.Variable(edges, name=axnames[0], flow=False)

    # NOTE: The order of the scale variations in NNLOjet is set in the config file.
    # This assumes that the "desired" order has been set there
    # Very confusingly kappaFO varies muR and muF together. muf is a variation (multiplicative) of mu_F.
    # See the read_dyturbo_vars_hist for the correct mapping
    axes = [*other_axes, ax]
    if all_scales:
        var_ax = hist.axis.StrCategory(
            [
                "pdf0",
                "kappaFO0.5-kappaf2.",
                "kappaFO2.-kappaf0.5",
                "kappaf0.5",
                "kappaf2.",
                "kappaFO0.5",
                "kappaFO2.",
            ],
            name="vars",
        )
        axes.append(var_ax)

    h = hist.Hist(*axes, storage=hist.storage.Weight())
    # Switch y and pT order
    ax_idx = len(other_axes)
    res = data[:, 3:].reshape(
        h.shape[ax_idx], *h.shape[:ax_idx], *h.shape[ax_idx + 1 :], 2
    )
    h[...] = np.moveaxis(res, 0, ax_idx)

    # Text file stores errors, convert to variance
    h.variances()[...] = h.variances() * h.variances()

    if charge is not None:
        h = add_charge_axis(h, charge)

    return h * 1e-3


def read_nnlojet_ybin(refname, ybins, charge=None):
    format_decimal = lambda x: (
        "0" if x == 0 else f"{round(x, 1+(x % 1 in [0.25, 0.75]))}".replace(".", "p")
    )
    yax = hist.axis.Variable(ybins, name="Y")
    return read_nnlojet_file(
        f"{refname}__{format_decimal(ybins[0])}__{format_decimal(ybins[1])}.dat",
        other_axes=[yax],
        charge=charge,
    )


def read_nnlojet_pty_hist(
    reffile,
    ybins=np.append(
        np.append(np.array((-5.0, -4.0)), np.arange(-3.5, 3.75, 0.25)),
        np.array((4.0, 5.0)),
    ),
    charge=None,
):

    h = read_nnlojet_ybin(reffile, ybins[:2], charge=charge)

    for pair in zip(ybins[1:-1], ybins[2:]):
        h = hh.concatenateHists(
            h, read_nnlojet_ybin(reffile, pair, charge=charge), allowBroadcast=False
        )

    return h


def read_dyturbo_hist(filenames, path="", axes=("y", "pt"), charge=None, coeff=None):
    filenames = [os.path.expanduser(os.path.join(path, f)) for f in filenames]

    hists = []
    for fn in filenames:
        expandedf = fn.split("+")

        hs = []
        for f in expandedf:
            if not os.path.isfile(f):
                raise ValueError(f"{f} is not a valid file!")

        if len(expandedf) == 1:
            hs.append(read_dyturbo_file(fn, axes, charge, coeff))
        elif len(expandedf) == 2:
            hs.append(
                hh.concatenateHists(
                    *[read_dyturbo_file(f, axes, charge, coeff) for f in expandedf]
                )
            )
        else:
            raise ValueError("Concatenate only supported for 2 files at present")

        hists.extend(hs)

    if len(hists) > 1:
        hists = hh.rebinHistsToCommon(hists, 0)

    h = hh.sumHists(hists)

    if charge is not None and "charge" not in h.axes.name:
        charge_args = (2, -2.0, 2.0) if charge != 0 else (1, 0, 1)
        charge_axis = hist.axis.Regular(*charge_args, flow=False, name="charge")
        hnew = hist.Hist(*h.axes, charge_axis, storage=h._storage_type())
        hnew[..., charge_axis.index(charge)] = h.view(flow=True)
        return hnew
    else:
        return h


def expand_dyturbo_filenames(
    path, basename, varname, pieces=["n3ll_born", "n2ll_ct", "n2lo_vj"], append=None
):
    return [
        os.path.join(
            path, "_".join(filter(None, [basename, piece, varname, append])) + ".txt"
        )
        for piece in pieces
    ]


def dyturbo_varnames():
    return [
        "mur{0}_muf{1}_mures{2}".format(i, j, k).replace("0", "H")
        for i in range(3)
        for j in range(3)
        for k in range(3)
        if abs(i - j) < 2
        and abs(i - k) < 2
        and abs(j - k) < 2
        and not (i == 1 and j == 1 and k == 1)
    ]


def read_dyturbo_variations(
    path,
    basename,
    varnames,
    axes,
    pieces=["n3ll_born", "n2ll_ct", "n2lo_vj"],
    append=None,
    charge=None,
):
    central_files = expand_dyturbo_filenames(path, basename, "", pieces, append)
    centralh = read_dyturbo_hist(central_files, axes=axes, charge=charge)
    var_ax = hist.axis.Integer(0, len(varnames) + 1, name="vars")
    varh = hist.Hist(*centralh.axes, var_ax, storage=centralh._storage_type())
    varh[..., 0] = centralh.view(flow=True)
    for i, var in enumerate(varnames):
        filenames = expand_dyturbo_filenames(path, basename, var, pieces, append)
        varh[..., i + 1] = read_dyturbo_hist(filenames, axes=axes, charge=charge).view(
            flow=True
        )
    return varh


def distribution_to_hist(data):
    next_bin = data[1:, 0]
    bin_width = next_bin - data[:-1, 0]
    data[:-1, 1:] = data[:-1, 1:] * bin_width[:, np.newaxis]
    return data


# Ignoring the scale unc for now
def read_matrixRadish_hist(filename, axname="pt"):
    data = read_text_data(filename)
    # Multiply through by bin width
    data = distribution_to_hist(data)
    bins = np.unique(data[:, 0])

    ax = hist.axis.Variable(
        bins, name=axname, underflow=not (bins[0] == 0 and "pt" in axname)
    )
    var_ax = hist.axis.Integer(0, 3, name="vars", flow=False)
    h = hist.Hist(ax, var_ax, storage=hist.storage.Double())

    h[...] = data[:-1, np.array([1, 3, 5])]
    return h * 1 / 1000


def read_text_data(filename):
    data = []
    for line in open(filename).readlines():
        entry = line.split("#")[0]
        entry_data = [float(i.strip()) for i in entry.split()]
        if not entry_data:
            continue
        data.append(entry_data)
    return np.array(data, dtype=float)


def read_dyturbo_file(filename, axnames=("Y", "qT"), charge=None, coeff=None):
    if filename.endswith(".root"):
        f = uproot.open(filename)
        hname = "_".join((["wgt", coeff] if coeff else ["s"]) + [axnames[0].lower()])
        h = f[hname].to_hist()
        if coeff == "a4":
            h = -1 * h
    else:
        data = read_text_data(filename)
        # 2 numbers per axis + result + error
        if data.shape[1] != len(axnames) * 2 + 2:
            raise ValueError(
                f"Mismatch between number of axes advertised ({len(axnames)} ==> {axnames}) and found ({(data.shape[1]-2)/2})"
            )

        axes = []
        offset = True
        for i, name in enumerate(axnames):
            # Normally last line is the total cross section, also possible it isn't, so check the bin ranges
            offset = (
                offset
                and data[-1, 2 * i] == data[0, 2 * i]
                and data[-1, 2 * i + 1] == data[-2, 2 * i + 1]
            )
            bins = sorted(
                list(set(data[: len(data) - offset, 2 * i : 2 * i + 2].flatten()))
            )
            axes.append(
                hist.axis.Variable(
                    bins, name=name, underflow=not (bins[0] == 0 and "qT" in name)
                )
            )

        h = hist.Hist(*axes, storage=hist.storage.Weight())
        h[...] = np.reshape(
            data[: len(data) - offset, len(axes) * 2 :], (*h.axes.size, 2)
        )
        # Text file stores errors, convert to variance
        h.variances()[...] = h.variances() * h.variances()

    if charge is not None:
        h = add_charge_axis(h, charge)

    return h * 1 / 1000


def add_charge_axis(h, charge):
    charge_args = (2, -2.0, 2.0) if charge != 0 else (1, 0, 1)
    charge_axis = hist.axis.Regular(*charge_args, flow=False, name="charge")

    has_vars = h.axes.name[-1] == "vars"
    new_axes = (
        (*h.axes, charge_axis)
        if not has_vars
        else (*h.axes[:-1], charge_axis, h.axes[-1])
    )
    hnew = hist.Hist(*new_axes, storage=h._storage_type())
    if has_vars:
        hnew[..., charge_axis.index(charge), :] = h.view(flow=True)
    else:
        hnew[..., charge_axis.index(charge)] = h.view(flow=True)
    return hnew


def read_scetlib_resum_and_fosing(
    scetlib_resum,
    scetlib_fo_sing,
    axes=None,
    charge=None,
    coeff=None,
):
    hresum = read_scetlib_hist(scetlib_resum, charge=charge, flip_y_sign=coeff == "a4")
    hfo_sing = read_scetlib_hist(
        scetlib_fo_sing, charge=charge, flip_y_sign=coeff == "a4"
    )

    if axes:
        newaxes = [*axes, "vars"]
        if charge is not None:
            newaxes.insert(-1, "charge")
        hfo_sing = hfo_sing.project(*newaxes)
        tnp_axes = [
            x
            for x in hfo_sing.axes["vars"]
            if any(re.match(e, x) for e in scetlib_tnp_match_expr)
        ]
        # TNP variations aren't defined for nonsingular. Set to the nominal
        if tnp_axes:
            indices = tuple(hfo_sing.axes["vars"].index(tnp_axes))
            hfo_sing.view()[..., indices] = hfo_sing[..., 0].view()[..., np.newaxis]
        hresum = hresum.project(*newaxes)

    return hresum, hfo_sing


def read_matched_scetlib_dyturbo_hist(
    scetlib_resum,
    scetlib_fo_sing,
    dyturbo_fo,
    axes=None,
    charge=None,
    zero_nons_bins=0,
    coeff=None,
):

    hresum, hfo_sing = read_scetlib_resum_and_fosing(
        scetlib_resum,
        scetlib_fo_sing,
        axes,
        charge,
        coeff,
    )

    dyturbo_axes = axes if axes else hfo_sing.axes.name[:-1]
    dyturbo_vars = (
        hfo_sing.axes["vars"].size > 1
        and "{scale}" in dyturbo_fo
        or "{i}" in dyturbo_fo
    )

    if dyturbo_vars:
        hfo = read_dyturbo_vars_hist(
            dyturbo_fo, var_axis=hfo_sing.axes["vars"], axes=dyturbo_axes, charge=charge
        )
    else:
        hfo = read_dyturbo_hist(
            [dyturbo_fo], axes=dyturbo_axes, charge=charge, coeff=coeff
        )

    return read_matched_scetlib_hist(hresum, hfo_sing, hfo, zero_nons_bins)


def read_matched_scetlib_nnlojet_hist(
    scetlib_resum,
    scetlib_fo_sing,
    nnlojet_fo,
    axes=None,
    charge=None,
    zero_nons_bins=0,
    coeff=None,
    smooth_nnlojet=False,
):
    hresum, hfo_sing = read_scetlib_resum_and_fosing(
        scetlib_resum,
        scetlib_fo_sing,
        axes,
        charge,
        coeff,
    )

    if not axes:
        axes = hresum.axes[:-1].name

    if "Y" in axes and "qT" in axes:
        nnlojeth = read_nnlojet_pty_hist(
            nnlojet_fo, ybins=hresum.axes["Y"].edges, charge=charge
        )
    else:
        nnlojeth = read_nnlojet_file(nnlojet_fo, axnames=axes, charge=charge)

    if smooth_nnlojet:
        if "Y" in axes:
            ax = nnlojeth.axes["Y"]
            start_bin, end_bin = ax.index((-3.5, 3.5))
            nnlojeth = hh.smooth_hist(
                nnlojeth, "Y", exclude_axes=["qT"], start_bin=start_bin, end_bin=end_bin
            )
        if "qT" in axes:
            nnlojeth = hh.smooth_hist(
                nnlojeth, "qT", start_bin=nnlojeth.axes["qT"].index(5)
            )

    return read_matched_scetlib_hist(hresum, hfo_sing, nnlojeth, zero_nons_bins)


def read_matched_scetlib_hist(
    hresum,
    hfo_sing,
    hfo,
    zero_nons_bins=0,
):
    for ax in ["Y", "Q"]:
        if ax in set(hfo.axes.name).intersection(set(hfo_sing.axes.name)).intersection(
            set(hresum.axes.name)
        ):
            hfo, hfo_sing, hresum = hh.rebinHistsToCommon([hfo, hfo_sing, hresum], ax)
    if (
        "vars" in hfo.axes.name
        and hfo.axes["vars"].size != hfo_sing.axes["vars"].size
        and hfo.axes["vars"].size == 1
    ):
        logger.warning("No variations found for fixed order histogram!")
        hfo = hfo[{"vars": 0}]

    hnonsing = hh.addHists(-1 * hfo_sing, hfo, flow=False, by_ax_name=False)

    if "qT" in hfo.axes.name and zero_nons_bins is not None:

        def translate_slice(ax, s):
            if not isinstance(s, slice):
                return s
            start = (
                int(ax.index(s.start.imag) + s.start.real)
                if isinstance(s.start, complex)
                else s.start
            )
            stop = (
                int(ax.index(s.stop.imag) + s.stop.real + 1)
                if isinstance(s.stop, complex)
                else s.stop
            )

            return slice(start, stop, s.step)

        qt_slice = translate_slice(hnonsing.axes["qT"], zero_nons_bins)
        logger.info(f"Zeroing bins with slices {qt_slice}")
        slices = tuple(
            qt_slice if ax == "qT" else slice(None) for ax in hnonsing.axes.name
        )
        hnonsing.values(flow=True)[slices] = 0
        hnonsing.variances(flow=True)[slices] = 0

    # variations are driven by resummed result, collect common variations from nonsingular piece
    # if needed

    # remapping is needed for scale variations which have slightly different parameter
    # definitions for resummed vs fixed-order pieces
    # *FIXME* this is sensitive to presence or absence of trailing zeros for kappas
    sing_scales_map = {
        "mufdown": "kappaf0.5",
        "mufup": "kappaf2.",
        "mufdown-kappaFO0.5-kappaf2.": "kappaFO0.5",
        "mufup-kappaFO2.-kappaf0.5": "kappaFO2.",
    }

    if hnonsing.axes["vars"] != hresum.axes["vars"]:
        if not all(x in hfo.axes["vars"] for x in sing_scales_map.values()):
            hnonsing = hnonsing[..., 0]
        else:
            htmp_nonsing = hist.Hist(
                *hnonsing.axes[:-1],
                hresum.axes["vars"],
                storage=hnonsing._storage_type(),
            )

            for i, var in enumerate(hresum.axes["vars"]):
                var_nonsing = sing_scales_map.get(var, var)
                if (
                    "muf" in var or "kappaf" in var or "kappaFO" in var
                ) and var_nonsing not in hnonsing.axes["vars"]:
                    raise ValueError(
                        f"Scale variation {var} found for resummed piece which should correspond to {var_nonsing} for nonsingular piece but is not found"
                    )
                var_nonsing = (
                    var_nonsing
                    if var_nonsing in hnonsing.axes["vars"]
                    else hnonsing.axes["vars"][0]
                )

                htmp_nonsing[{"vars": i}] = hnonsing[{"vars": var_nonsing}].view(
                    flow=True
                )

            hnonsing = htmp_nonsing

    htotal = hh.addHists(hresum, hnonsing, by_ax_name=False)

    return htotal


def read_json(fIn):

    if not os.path.exists(fIn):
        logger.warning(f"File {fIn} not found")
        return False
    else:
        with open(fIn) as f:
            jsDict = json.load(f)
        return jsDict


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


def get_scetlib_config(infile):
    if infile.endswith(".pkl"):
        with open(infile, "rb") as f:
            results = pickle.load(f)
        return results["config"]
    else:
        raise ValueError("Expected scetlib output in pkl format")


def read_infile(input):
    # read histogramer input file(s)
    result = {}
    meta = []
    infiles = []
    if isinstance(input, list):
        for inpt in input:
            r, m, h = read_infile(inpt)
            result.update(r)
            meta += m
            infiles += h
        return result, meta, infiles

    logger.info(f"Load {input}")
    if input.endswith(".pkl.lz4"):
        with lz4.frame.open(input) as f:
            result = pickle.load(f)
    elif input.endswith(".hdf5"):
        h5file = h5py.File(input, "r")
        infiles = [h5file]
        result = load_results_h5py(h5file)
    else:
        raise ValueError("Unsupported file type")

    meta = result["meta_info"] if "meta_info" in result else result["meta_data"]

    return result, [meta], [infiles]


def read_dyturbo_angular_coeffs(
    dyturbof, boson=None, rebin=None, absy=True, add_axes=[]
):
    if add_axes and not all(ax.size == 1 for ax in add_axes):
        raise ValueError("Can only add axes of size 1!")

    if type(dyturbof) == str:
        dyturbof = uproot.open(dyturbof)

    if not boson:
        boson = (
            "Wp"
            if "wp" in dyturbof.file_path
            else ("Wm" if "wm" in dyturbof.file_path else "Z")
        )

    sigma_ul = dyturbof["s_qt_vs_y"].to_hist()
    for ax, name in zip(sigma_ul.axes, ["qT", "Y"]):
        ax._ax.metadata["name"] = name

    if rebin:
        sigma_ul = hh.rebinHistMultiAx(sigma_ul, rebin.keys(), rebin.values())
    if absy:
        sigma_ul = hh.makeAbsHist(sigma_ul, "Y")

    charge_range = (2, -2, 2) if "w" in boson.lower() else (1, -1, 1)
    charge = 0 if boson.lower() == "z" else (-1 if "wm" in boson.lower() else 1)
    charge_ax = hist.axis.Regular(*charge_range, name="charge", flow=False)
    h = hist.Hist(
        *add_axes,
        *reversed(sigma_ul.axes),
        charge_ax,
        hist.axis.Regular(8, 0, 8, flow=False, name="helicity"),
        storage=sigma_ul.storage_type(),
    )
    for i in range(8):
        sigma_i = dyturbof[f"wgt_a{i}_y_qt"].to_hist()
        for ax, name in zip(sigma_i.axes, ["qT", "Y"]):
            ax._ax.metadata["name"] = name
        if rebin:
            sigma_i = hh.rebinHistMultiAx(sigma_i, rebin.keys(), rebin.values())
        if absy:
            sigma_i = hh.makeAbsHist(sigma_i, "Y")
        entry = (*[0] * len(add_axes), Ellipsis, charge_ax.index(charge), i)
        # qT, y order in DY turbo is reversed wrt MiNNLO
        h[entry] = hh.divideHists(sigma_i, sigma_ul).view().T

    return h


def read_mu_hist_combine_tau(
    minnlof, mu_sample, hist_name, eras, combine_with_tau=True
):
    with h5py.File(minnlof, "r") as h5file:
        results = load_results_h5py(h5file)
        sumw = 0
        xsec = 0
        hmu = None

        for era in eras:

            mu_sample_era = f"{mu_sample}_{era}"
            if mu_sample_era not in results.keys():
                logger.warning(f"Sample {mu_sample_era} not found, continue without")
            else:
                sumw += read_sumw(minnlof, mu_sample_era)
                hmu_era = load_and_scale(
                    results, mu_sample_era, hist_name, apply_xsec=False
                )
                xsec_era = read_xsec(minnlof, mu_sample_era)
                if xsec == 0:
                    xsec = xsec_era
                elif xsec_era != xsec:
                    raise RuntimeError(
                        f"Incompatible cross sections for sample {mu_sample} across eras {eras}"
                    )

                if hmu is None:
                    hmu = hmu_era
                else:
                    hmu += hmu_era

            if combine_with_tau:
                tau_sample_era = mu_sample_era.replace("mu", "tau")
                if tau_sample_era not in results.keys():
                    logger.warning(
                        f"Sample {tau_sample_era} not found, continue without"
                    )
                    continue

                hmu_era = load_and_scale(
                    results, tau_sample_era, hist_name, apply_xsec=False
                )
                if hmu is None:
                    hmu = hmu_era
                else:
                    hmu += hmu_era

                sumw += read_sumw(minnlof, tau_sample_era)

        if xsec == 0:
            raise RuntimeError("Got cross section of 0")

        return hmu * xsec / sumw
