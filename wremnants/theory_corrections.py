import glob
import os
import pathlib
import pickle
import re

import h5py
import hist
import lz4.frame
import numpy as np
import ROOT
from scipy.interpolate import make_smoothing_spline

from utilities import common
from utilities.io_tools import input_tools
from wremnants import theory_tools
from wremnants.correctionsTensor_helper import makeCorrectionsTensor
from wums import boostHistHelpers as hh
from wums import logging

logger = logging.child_logger(__name__)


def valid_theory_corrections():
    corr_files = glob.glob(common.data_dir + "TheoryCorrections/*Corr*.pkl.lz4")
    matches = [
        re.match(r"(^.*)_Corr[W|Z|BSM]\.pkl\.lz4", os.path.basename(c))
        for c in corr_files
    ]
    return [m[1] for m in matches if m] + ["none"]


def valid_ew_theory_corrections():
    corr_files = glob.glob(
        common.data_dir + "TheoryCorrections/*[eE][wW]*Corr*.pkl.lz4"
    )
    matches = [
        re.match(r"(^.*)Corr[W|Z]\.pkl\.lz4", os.path.basename(c)) for c in corr_files
    ]
    return [m[1] for m in matches if m] + ["none"]


def load_corr_helpers(
    procs,
    generators,
    make_tensor=True,
    base_dir=f"{common.data_dir}/TheoryCorrections/",
    minnlo_ratio=True,
):
    corr_helpers = {}
    for proc in procs:
        corr_helpers[proc] = {}
        for i, generator in enumerate(generators):
            if proc.startswith("WtoNMu") and i == 0:
                label = "BSM"
            else:
                label = proc[0]

            fname = f"{base_dir}/{generator}_Corr{label}.pkl.lz4"
            if not os.path.isfile(fname):
                logger.warning(
                    f"Did not find correction file {fname} for process {proc}, generator {generator}. No correction will be applied for this process!"
                )
                continue
            logger.debug(f"Make theory correction helper for file: {fname}")
            corrh = load_corr_hist(
                fname, label, get_corr_name(generator, minnlo_ratio=minnlo_ratio)
            )
            numh = None
            if (
                (generator == generators[0])
                and ("nnlojet" in generator.lower())
                and ("pdfas" not in generator.lower())
            ):
                logger.info(
                    f"Adding statistical uncertainties for correction {generator}"
                )
                numh = load_corr_hist(fname, label, f"{generator}_hist")

            corrh = postprocess_corr_hist(corrh, numh)
            if not make_tensor:
                corr_helpers[proc][generator] = corrh
            elif "Helicity" in generator:
                corr_helpers[proc][generator] = makeCorrectionsTensor(
                    corrh, ROOT.wrem.CentralCorrByHelicityHelper, tensor_rank=3
                )
            else:
                corr_helpers[proc][generator] = makeCorrectionsTensor(
                    corrh,
                    tensor_weight=generator in theory_tools.theory_corr_weight_map,
                )
    for generator in generators:
        if not any([generator in corr_helpers[proc] for proc in procs]):
            logger.warning(
                f"Did not find correction for generator {generator} for any processes!"
            )
    return corr_helpers


def make_corr_helper_fromnp(
    filename=f"{common.data_dir}/N3LLCorrections/inclusive_{{process}}_pT.npz", isW=True
):
    if isW:
        corrf_Wp = np.load(filename.format(process="Wp"), allow_pickle=True)
        corrf_Wm = np.load(filename.format(process="Wm"), allow_pickle=True)
        bins = corrf_Wp["bins"]
        axis_charge = common.axis_chargeWgen
    else:
        corrf = np.load(filename.format(process="Z"), allow_pickle=True)
        bins = corrf["bins"]
        axis_charge = common.axis_chargeZgen

    axis_syst = hist.axis.Regular(
        len(bins[0]) - 1,
        bins[0][0],
        bins[0][-1],
        name="systIdx",
        overflow=False,
        underflow=False,
    )
    axis_mass = hist.axis.Variable(bins[1], name="mass")
    axis_y = hist.axis.Variable(bins[2], name="y")
    axis_pt = hist.axis.Regular(
        len(bins[-1]) - 1, bins[-1][0], bins[-1][-1], name="pT", underflow=False
    )

    corrh = hist.Hist(axis_mass, axis_y, axis_pt, axis_charge, axis_syst)
    if isW:
        corrh[..., 1, :] = np.moveaxis(corrf_Wp["scetlibCorr3D_Wp"], 0, -1)
        corrh[..., 0, :] = np.moveaxis(corrf_Wm["scetlibCorr3D_Wm"], 0, -1)
    else:
        corrh[..., 0, :] = np.moveaxis(corrf["scetlibCorr3D_Z"], 0, -1)
    corrh[hist.underflow, ...] = 1.0
    corrh[hist.overflow, ...] = 1.0
    corrh[:, hist.underflow, ...] = 1.0
    corrh[:, hist.overflow, ...] = 1.0
    corrh[:, :, hist.overflow, ...] = 1.0

    return makeCorrectionsTensor(corrh)


def load_corr_hist(filename, proc, histname):
    with lz4.frame.open(filename) as f:
        corr = pickle.load(f)
        corrh = corr[proc][histname]
    return corrh


def compute_envelope(
    h, name, entries, axis_name="vars", slice_axis=None, slice_val=None
):

    axis_idx = h.axes.name.index(axis_name)
    hvars = h[{axis_name: entries}]

    hvar_min = h[{axis_name: entries[0]}].copy()
    hvar_max = h[{axis_name: entries[0]}].copy()

    hvar_min.values()[...] = np.min(hvars.values(), axis=axis_idx)
    hvar_max.values()[...] = np.max(hvars.values(), axis=axis_idx)

    if slice_axis is not None:
        s = hist.tag.Slicer()

        hnom = h[{axis_name: entries[0]}]
        slice_idx = h.axes[slice_axis].index(slice_val)

        hvar_min[{slice_axis: s[:slice_idx]}] = hnom[{slice_axis: s[:slice_idx]}].view()
        hvar_max[{slice_axis: s[:slice_idx]}] = hnom[{slice_axis: s[:slice_idx]}].view()

    res = {}
    res[f"{name}_Down"] = hvar_min
    res[f"{name}_Up"] = hvar_max

    return res


def postprocess_corr_hist(corrh, numh=None):
    # extend variations with some envelopes and special kinematic slices

    if (
        "vars" not in corrh.axes.name
        or type(corrh.axes["vars"]) != hist.axis.StrCategory
    ):
        return corrh

    additional_var_hists = {}

    central_var = corrh.axes["vars"][0]
    renorm_scale_vars = [central_var, "kappaFO0.5-kappaf2.", "kappaFO2.-kappaf0.5"]

    renorm_fact_scale_vars = [
        central_var,
        "kappaFO0.5-kappaf2.",
        "kappaFO2.-kappaf0.5",
        "mufdown",
        "mufup",
        "mufdown-kappaFO0.5-kappaf2.",
        "mufup-kappaFO2.-kappaf0.5",
    ]

    resum_scales = ["muB", "nuB", "muS", "nuS"]
    resum_scale_vars_exclusive = [
        var
        for var in corrh.axes["vars"]
        if any(resum_scale in var for resum_scale in resum_scales)
    ]
    resum_scale_vars = [central_var] + resum_scale_vars_exclusive

    if len(renorm_fact_scale_vars) == 1:
        return corrh

    transition_vars_exclusive = [
        "transition_points0.2_0.35_1.0",
        "transition_points0.2_0.75_1.0",
    ]

    renorm_fact_resum_scale_vars = renorm_fact_scale_vars + resum_scale_vars_exclusive

    renorm_fact_resum_transition_scale_vars = (
        renorm_fact_resum_scale_vars + transition_vars_exclusive
    )

    if all(var in corrh.axes["vars"] for var in renorm_scale_vars):
        additional_var_hists.update(
            compute_envelope(corrh, "renorm_scale_envelope", renorm_scale_vars)
        )

        # same thing but restricted to qT>20GeV to capture only the fixed order part of the variation and
        # neglect the part at low pt which should be redundant with the TNPs
        additional_var_hists.update(
            compute_envelope(
                corrh,
                "renorm_scale_pt20_envelope",
                renorm_scale_vars,
                slice_axis="qT",
                slice_val=20.0,
            )
        )

    if all(var in corrh.axes["vars"] for var in renorm_fact_scale_vars):
        additional_var_hists.update(
            compute_envelope(
                corrh, "renorm_fact_scale_envelope", renorm_fact_scale_vars
            )
        )

        # same thing but restricted to qT>20GeV to capture only the fixed order part of the variation and
        # neglect the part at low pt which should be redundant with the TNPs
        additional_var_hists.update(
            compute_envelope(
                corrh,
                "renorm_fact_scale_pt20_envelope",
                renorm_fact_scale_vars,
                slice_axis="qT",
                slice_val=20.0,
            )
        )

    if all(var in corrh.axes["vars"] for var in renorm_fact_resum_scale_vars):
        additional_var_hists.update(
            compute_envelope(
                corrh, "renorm_fact_resum_scale_envelope", renorm_fact_resum_scale_vars
            )
        )
    if all(
        var in corrh.axes["vars"] for var in renorm_fact_resum_transition_scale_vars
    ):
        additional_var_hists.update(
            compute_envelope(
                corrh,
                "renorm_fact_resum_transition_scale_envelope",
                renorm_fact_resum_transition_scale_vars,
            )
        )
    if len(resum_scale_vars) > 1 and all(
        var in corrh.axes["vars"] for var in resum_scale_vars
    ):
        additional_var_hists.update(
            compute_envelope(corrh, "resum_scale_envelope", resum_scale_vars)
        )

    # add per-bin stat unc from correction (~= only the numerator, MiNNLO has very small stat uncs)
    if numh is not None:
        numh_nom = numh[{"vars": 0}]
        var_relative = np.sqrt(numh_nom.variances()) / numh_nom.values()
        nom_vals = corrh[{"vars": 0}].values()

        shape = var_relative.shape
        nbins = var_relative.size

        # nbins copies of the original histogram with shape `shape`
        base_up = np.broadcast_to(nom_vals, (nbins,) + shape).copy()
        base_dn = base_up.copy()

        linear_idx = np.arange(nbins)  # 1-dim array ennumerating all bins
        multi_idx = np.unravel_index(linear_idx, shape)  # n-d idx for each bin
        flat_var_rel = var_relative.ravel()  # 1-dim array of variations

        # address n-th copy of histograms (to hold the variation in the n-th bin [and no other]).
        # In that n-th copy, modify the corresponding bin in the n-dim hist
        base_up[(linear_idx,) + multi_idx] *= 1.0 + flat_var_rel
        base_dn[(linear_idx,) + multi_idx] *= 1.0 - flat_var_rel

        template = corrh[{"vars": 0}]

        for i in range(nbins):
            h_up = template.copy()
            h_dn = template.copy()
            h_up.values()[...] = base_up[i]
            h_dn.values()[...] = base_dn[i]

            additional_var_hists[f"per_bin_stat_unc_theory_corr_bin{i}Up"] = h_up
            additional_var_hists[f"per_bin_stat_unc_theory_corr_bin{i}Down"] = h_dn

    if not additional_var_hists:
        return corrh

    vars_out = list(corrh.axes["vars"]) + list(additional_var_hists.keys())

    vars_out_axis = hist.axis.StrCategory(vars_out, name="vars")
    corrh_tmp = hist.Hist(*corrh.axes[:-1], vars_out_axis, storage=corrh.storage_type())

    for i, var in enumerate(vars_out_axis):
        if var in corrh.axes["vars"]:
            corrh_tmp[{"vars": i}] = corrh[{"vars": var}].view(flow=True)
        else:
            corrh_tmp[{"vars": i}] = additional_var_hists[var].view(flow=True)

    corrh = corrh_tmp

    return corrh


def get_corr_name(generator, minnlo_ratio=True):
    # Hack for now
    label = generator.replace("1D", "")
    if (
        "dataPtll" in generator or "dataRecoPtll" in generator
    ) and "scetlib" not in generator:
        return "MC_data_ratio"
    if minnlo_ratio:
        return (
            f"{label}_minnlo_ratio"
            if "Helicity" not in generator
            else f"{label.replace('Helicity', '')}_minnlo_coeffs"
        )
    else:
        return (
            f"{label}_hist"
            if "Helicity" not in generator
            else f"{label.replace('Helicity', '')}_coeffs"
        )


def rebin_corr_hists(hists, ndim=-1, binning=None):
    # Allow trailing dimensions to be different (e.g., variations)
    ndims = min([x.ndim for x in hists]) if ndim < 0 else ndim
    if binning:
        try:
            hists = [
                h if not h else hh.rebinHistMultiAx(h, binning.keys(), binning.values())
                for h in hists
            ]
        except ValueError:
            logger.warning("Can't rebin axes to predefined binning")
        return hists

    for i in range(ndims):
        hists = hh.rebinHistsToCommon(hists, i)
    return hists


# Apply an iterative smoothing in 2D (effectively assumed to be Y, the qT)
def smooth_theory_corr(corrh, minnloh, numh, ax2_start=5):
    if corrh.ndim != 5:
        raise NotImplementedError(
            f"Currently only dimension 5 hists are supported for smoothing. Found ndim={corrh.ndim} ({corrh.axes.name})"
        )

    nc = corrh.axes["charge"].size

    # First smooth in 1D (should be rapidity)
    corrh1D = hh.divideHists(numh[{"vars": 0}].project(1), minnloh.project(1))
    ax1 = corrh.axes[1]
    spl = make_smoothing_spline(ax1.centers, corrh1D.values())
    smooth = spl(ax1.centers) / corrh1D.values()
    corrh.values()[...] = (corrh.values().T * smooth[:, np.newaxis]).T

    # This should be qT
    ax2 = corrh.axes[2]
    for i1 in range(ax1.size):
        for ic in range(corrh.axes["charge"].size):
            for iv in range(corrh.axes["vars"].size):
                spl = make_smoothing_spline(
                    ax2.centers[ax2_start:], corrh[0, i1, ax2_start:, ic, iv].values()
                )
                corrh.values()[0, i1, ax2_start:, ic, iv] = spl(ax2.centers[ax2_start:])

    return corrh


# Assuming the 3 physics variable dimensions are first
def set_corr_ratio_flow(corrh):
    # Probably there's a better way to do this...
    if corrh.axes[0].traits.underflow:
        corrh[hist.underflow, ...] = np.ones_like(corrh[0, ...].view(flow=True))
    if corrh.axes[0].traits.overflow:
        corrh[hist.overflow, ...] = np.ones_like(corrh[0, ...].view(flow=True))

    if corrh.axes[1].traits.underflow:
        corrh[:, hist.underflow, ...] = np.ones_like(corrh[:, 0, ...].view(flow=True))
    if corrh.axes[1].traits.overflow:
        corrh[:, hist.overflow, ...] = np.ones_like(corrh[:, 0, ...].view(flow=True))

    if corrh.axes[2].traits.underflow:
        corrh[:, :, hist.underflow, ...] = np.ones_like(
            corrh[:, :, 0, ...].view(flow=True)
        )
    if corrh.axes[2].traits.overflow:
        corrh[:, :, hist.overflow, ...] = np.ones_like(
            corrh[:, :, 0, ...].view(flow=True)
        )
    return corrh


def make_corr_from_ratio(
    denom_hist, num_hist, rebin=None, smooth="numerator", normalize=False
):
    denom_hist, num_hist = rebin_corr_hists([denom_hist, num_hist], binning=rebin)

    if smooth == "numerator":
        logger.info(
            "Applying spline-based smoothing to numerator before making correction hist"
        )
        num_hist = hh.smooth_hist(
            hh.smooth_hist(num_hist, "absY", exclude_axes=["qT"]), "qT", start_bin=4
        )

    if normalize:
        num_hist = hh.divideHists(num_hist, num_hist.project("vars"))
        denom_hist = hh.normalize(denom_hist, scale=1)

    corrh = hh.divideHists(num_hist, denom_hist, flow=False, by_ax_name=False)

    if smooth == "ratio":
        logger.info("Applying spline-based smoothing to correction hist ratio")
        corrh = smooth_theory_corr(corrh, denom_hist, num_hist, ax2_start=5)

    return set_corr_ratio_flow(corrh), denom_hist, num_hist


def make_corr_by_helicity(
    ref_helicity_hist,
    target_sigmaul,
    target_sigma4,
    coeff_hist=None,
    coeffs_from_hist=[],
    binning=None,
    ndim=3,
):
    ref_helicity_hist, target_sigmaul, target_sigma4 = rebin_corr_hists(
        [ref_helicity_hist, target_sigmaul, target_sigma4], ndim, binning
    )

    apply_coeff_corr = coeff_hist is not None and coeffs_from_hist

    # broadcast back mass and helicity axes (or vars axis)
    sigmaUL_ratio = (
        hh.divideHists(
            target_sigmaul,
            ref_helicity_hist[{"massVgen": 0, "helicity": -1.0j}],
            flow=False,
            by_ax_name=False,
        ).values()[np.newaxis, ..., np.newaxis, :]
        if target_sigmaul
        else np.ones_like(ref_helicity_hist)[..., np.newaxis]
    )

    ref_coeffs = theory_tools.helicity_xsec_to_angular_coeffs(ref_helicity_hist)

    corr_ax = hist.axis.Boolean(name="corr")
    vars_ax = (
        target_sigmaul.axes["vars"]
        if target_sigmaul
        else hist.axis.Regular(1, 0, 1, name="vars")
    )
    corr_coeffs = hist.Hist(*ref_coeffs.axes, corr_ax, vars_ax)
    # Corr = False is the uncorrected coeffs, corrected coeffs have the new A4
    # NOTE: the corrected coeffs are multiplied through by the sigmaUL correction, so that the
    # new correction can be made as the ratio of the sum. To get the correct coeffs, this should
    # be divided back out
    corr_coeffs[...] = ref_coeffs.values()[..., np.newaxis, np.newaxis]

    if target_sigma4:
        target_a4_coeff = make_angular_coeff(target_sigma4, target_sigmaul)
        corr_coeffs[..., 4.0j, True, :] = target_a4_coeff.values()

    if apply_coeff_corr:
        for coeff in coeffs_from_hist:
            scale = -1 if coeff in [1, 4] else 1
            idx = complex(0, coeff)
            corr_coeffs[..., idx, True, :] = (
                coeff_hist[{"helicity": idx}].values()[..., np.newaxis] * scale
            )
    # Scale by the sigmaUL correction,
    corr_coeffs.values()[..., corr_coeffs.axes["corr"].index(True), :] *= sigmaUL_ratio

    corr_coeffs = set_corr_ratio_flow(corr_coeffs)
    return corr_coeffs


def make_theory_helpers(
    pdfs,
    theory_corr=[],
    procs=["Z", "W"],
    corrs=["qcdScale", "pdf", "pdf_from_corr", "alphaS", "pdf_central"],
):

    theory_helpers_procs = {p: {} for p in procs}

    for proc in theory_helpers_procs.keys():

        if "qcdScale" in corrs:
            theory_helpers_procs[proc]["qcdScale"] = (
                make_qcd_uncertainty_helper_by_helicity(
                    is_z=proc == "Z",
                    rebin_ptVgen=False,
                    return_tensor=True,
                )
            )

        if "pdf" in corrs:
            theory_helpers_procs[proc]["pdf"] = (
                make_pdfs_uncertainties_helper_by_helicity(
                    proc=proc,
                    pdfs=pdfs,
                )
            )
        if "pdf_from_corr" in corrs:
            pdf_from_corrs = [x + "_Corr" for x in theory_corr if "pdfvar" in x]
            theory_helpers_procs[proc]["pdf_from_corr"] = (
                make_pdfs_from_corrs_uncertainties_helper_by_helicity(
                    proc=proc,
                    pdfs_from_corrs=pdf_from_corrs,
                )
            )
        if "alphaS" in corrs:
            as_vars = [x + "_Corr" for x in theory_corr if "pdfas" in x]
            theory_helpers_procs[proc]["alphaS"] = (
                make_alphaS_uncertainties_helper_by_helicity(
                    proc=proc,
                    as_vars=as_vars,
                )
            )
        if "pdf_central" in corrs:
            theory_helpers_procs[proc]["pdf_central"] = (
                make_uncertainty_helper_by_helicity(
                    proc=proc,
                    nom=theory_tools.pdfMap[pdfs[0]]["name"],
                    den="pdf_uncorr",
                    central_weights=True,
                    filename=common.data_dir
                    + f"/TheoryCorrections/ByHelicity/PDFs/w_z_gen_dists_maxFiles_m1_{pdfs[0]}_pdfByHelicity_skimmed.hdf5",
                )
            )

    return theory_helpers_procs


def make_qcd_uncertainty_helper_by_helicity(
    is_z=False,
    filename=f"{common.data_dir}/angularCoefficients/w_z_helicity_xsecs.hdf5",
    rebin_ptVgen=common.ptV_binning,
    rebin_absYVgen=False,
    rebin_massVgen=False,
    return_tensor=True,
):

    # load helicity cross sections from file
    with h5py.File(filename, "r") as h5file:
        results = input_tools.load_results_h5py(h5file)

    def get_helicity_xsecs(
        suffix="",
        rebin_ptVgen=common.ptV_binning,
        rebin_absYVgen=False,
        rebin_massVgen=2,
    ):
        h = results[f"Z{suffix}"] if is_z else results[f"W{suffix}"]

        if rebin_ptVgen:
            if type(rebin_ptVgen) is bool:
                h = hh.rebinHist(h, "ptVgen", common.ptV_binning)
            else:
                h = hh.rebinHist(h, "ptVgen", rebin_ptVgen)
        if rebin_massVgen:
            if type(rebin_massVgen) is bool:
                if is_z:
                    axis_massVgen = h.axes["massVgen"]
                    if len(axis_massVgen.edges) > 2:
                        h = hh.rebinHist(h, "massVgen", axis_massVgen.edges[::2])
            else:
                h = hh.rebinHist(h, "massVgen", rebin_massVgen)
        if rebin_absYVgen:
            h = hh.rebinHist(h, "absYVgen", rebin_absYVgen)

        return h

    helicity_xsecs = get_helicity_xsecs(
        rebin_ptVgen=rebin_ptVgen,
        rebin_absYVgen=rebin_absYVgen,
        rebin_massVgen=rebin_massVgen,
    )
    helicity_xsecs_lhe = get_helicity_xsecs(
        "_lhe",
        rebin_ptVgen=rebin_ptVgen,
        rebin_absYVgen=rebin_absYVgen,
        rebin_massVgen=rebin_massVgen,
    )

    helicity_xsecs_nom = helicity_xsecs[{"muRfact": 1.0j, "muFfact": 1.0j}].values()

    # set disallowed combinations of mur/muf equal to nominal
    helicity_xsecs.values()[..., 0, 2] = helicity_xsecs_nom
    helicity_xsecs.values()[..., 2, 0] = helicity_xsecs_nom

    # flatten scale variations and compute envelope
    helicity_xsecs_flat = np.reshape(
        helicity_xsecs.values(), (*helicity_xsecs.values().shape[:-2], -1)
    )
    helicity_xsecs_min = np.min(helicity_xsecs_flat, axis=-1)
    helicity_xsecs_max = np.max(helicity_xsecs_flat, axis=-1)

    # build variation histogram in the format expected by the corrector
    corr_ax = hist.axis.Boolean(name="corr")

    def get_names(ihel):
        base_name = f"helicity_{ihel}"
        return f"{base_name}_Down", f"{base_name}_Up"

    var_names = []
    var_names.append("nominal")
    for ihel in range(-1, 8):
        var_names.extend(get_names(ihel))

    var_names.append("pythia_shower_kt")

    vars_ax = hist.axis.StrCategory(var_names, name="vars")

    axes_no_scale = helicity_xsecs.axes[:-2]
    corr_coeffs = hist.Hist(*axes_no_scale, corr_ax, vars_ax)

    # set all helicity_xsecs, including overflow/underflow to default safe value (leads to weight of 1.0 by construction)
    corr_coeffs.values(flow=True)[...] = 1.0

    # set all helicity_xsecs equal to nominal
    corr_coeffs.values()[...] = helicity_xsecs_nom[..., None, None]

    # set envelope variations
    for ihel in range(-1, 8):
        downvar, upvar = get_names(ihel)

        corr_coeffs.values()[..., ihel + 1, 1, var_names.index(downvar)] = (
            helicity_xsecs_min[..., ihel + 1]
        )
        corr_coeffs.values()[..., ihel + 1, 1, var_names.index(upvar)] = (
            helicity_xsecs_max[..., ihel + 1]
        )

    corr_coeffs[{"corr": False, "vars": "pythia_shower_kt"}] = (
        helicity_xsecs[{"muRfact": 1.0j, "muFfact": 1.0j}].values()
        / helicity_xsecs[
            {"muRfact": 1.0j, "muFfact": 1.0j, "helicity": -1.0j}
        ].values()[..., None]
    )
    corr_coeffs[{"corr": True, "vars": "pythia_shower_kt"}] = (
        helicity_xsecs_lhe[{"muRfact": 1.0j, "muFfact": 1.0j}].values()
        / helicity_xsecs_lhe[
            {"muRfact": 1.0j, "muFfact": 1.0j, "helicity": -1.0j}
        ].values()[..., None]
    )

    if return_tensor:
        helper = makeCorrectionsTensor(
            corr_coeffs, ROOT.wrem.CentralCorrByHelicityHelper, tensor_rank=3
        )

        # override tensor_axes since the output is different here
        helper.tensor_axes = [vars_ax]

        return helper
    else:
        return corr_coeffs


def make_pdfs_uncertainties_helper_by_helicity(
    proc,
    pdfs,
    return_tensor=True,
):
    pdf_file_template = (
        common.data_dir
        + "/TheoryCorrections/ByHelicity/PDFs/w_z_gen_dists_maxFiles_m1_{pdf}_pdfByHelicity_skimmed.hdf5"
    )
    pdf_helpers = {}
    for pdf in pdfs:
        pdf_name = theory_tools.pdfMap[pdf]["name"]
        logger.debug(
            f"Making PDF uncertainty helper by helicity for PDF set {pdf_name}"
        )
        pdf_renorm_name = (
            pdf_name if theory_tools.pdfMap[pdf].get("renorm", False) else "pdf_uncorr"
        )
        pdf_helper = make_uncertainty_helper_by_helicity(
            proc=proc,
            nom=pdf_name,
            den=pdf_renorm_name,
            filename=pdf_file_template.format(pdf=pdf),
            var_ax_name="pdfVar",
            return_tensor=return_tensor,
        )
        if pdf_helper is not None:
            pdf_helpers[pdf_name] = pdf_helper
    return pdf_helpers


def make_pdfs_from_corrs_uncertainties_helper_by_helicity(
    proc,
    pdfs_from_corrs,
    return_tensor=True,
):
    pdf_file_template = (
        common.data_dir
        + "/TheoryCorrections/ByHelicity/PDFsFromCorrs/w_z_gen_dists_{pdf}_maxFiles_m1_skimmed.hdf5"
    )
    pdf_helpers = {}
    for pdf in pdfs_from_corrs:
        logger.debug(f"Making PDF uncertainty helper by helicity for theory corr {pdf}")
        pdf_helper = make_uncertainty_helper_by_helicity(
            proc=proc,
            nom=pdf,
            den="pdf_uncorr",
            central_weights=False,
            var_ax_name="vars",
            filename=pdf_file_template.format(pdf=pdf),
            return_tensor=return_tensor,
        )
        if pdf_helper is not None:
            pdf_helpers[pdf] = pdf_helper
    return pdf_helpers


def make_alphaS_uncertainties_helper_by_helicity(
    proc,
    as_vars,
    return_tensor=True,
):
    alphas_file_template = (
        common.data_dir
        + "/TheoryCorrections/ByHelicity/AlphaS/w_z_gen_dists_{as_var}_maxFiles_m1_skimmed.hdf5"
    )
    as_helpers = {}
    for as_var in as_vars:
        logger.debug(
            f"Making alphaS uncertainty helper by helicity for theory corr {as_var}"
        )
        fname = alphas_file_template.format(as_var=as_var)
        as_helper = make_uncertainty_helper_by_helicity(
            proc=proc,
            nom=as_var,
            den="theory_uncorr",
            filename=fname,
            var_ax_name="vars",
            return_tensor=return_tensor,
        )
        if as_helper is not None:
            as_helpers[as_var] = as_helper
    return as_helpers


def make_uncertainty_helper_by_helicity(
    proc,
    nom,
    den,
    filename,
    filename_den=None,
    central_weights=False,
    var_ax_name="pdfVar",
    return_tensor=True,
):
    """
    Construct a CentralCorrByHelicityHelper from helicity cross sections stored in an hdf5 file.
    """

    if filename_den is None:
        filename_den = filename

    # load helicity cross sections from file #TODO: include DYJetsToMuMuMass10to50
    proc_map = {
        "Z": ("Zmumu",),
        "W": ("Wmunu",),
    }

    def _collect_hist(hist_name, filename):
        hist_key = f"nominal_gen_{hist_name}"
        hists = []
        if not os.path.exists(filename):
            logger.warning(
                f"File {filename} does not exist. Not creating histogram of variations by helicities."
            )
            return None
        with h5py.File(filename, "r") as h5file:
            for process in proc_map.get(proc, ()):
                results = input_tools.load_results_h5py(h5file)
                if process not in results.keys():
                    logger.warning(
                        f"Did not find key for process {process} in {filename}. Not creating histogram of variations by helicities for process {process} and variation {nom}."
                    )
                    return None
                outputs = results[process].get("output", {})
                if hist_key not in outputs:
                    logger.warning(
                        f"Did not find {hist_key} in {filename}. Not creating histogram of variations by helicities for process {process} and variation {nom}."
                    )
                    return None
                hists.append(outputs[hist_key].get())
        if not hists:
            logger.warning(
                f"Process {proc} is not supported when building PDF variations."
            )
            return None
        combined = hh.sumHists(hists)
        return combined

    h_nom = _collect_hist(nom, filename)
    if h_nom is None:
        return None

    if den == nom:
        h_den = h_nom
    else:
        h_den = _collect_hist(den, filename_den)
        if h_den is None:
            return None

    # construct the correction tensor
    corr_ax = hist.axis.Boolean(name="corr")
    vars_ax = h_nom.axes[var_ax_name]
    axes_no_scale = h_nom.axes[:-1]
    if central_weights:
        # in the case we are computing a helper for the central weights, we don't need to fill all variations
        new_vars_ax = hist.axis.StrCategory(["nominal"], name="vars")
        corr_coeffs = hist.Hist(*axes_no_scale, corr_ax, new_vars_ax)
    else:
        corr_coeffs = hist.Hist(*axes_no_scale, corr_ax, vars_ax)

    # set all helicity_xsecs equal to nominal
    if var_ax_name in h_den.axes.name:
        h_den = h_den[{var_ax_name: 0}]
    corr_coeffs.values(flow=True)[...] = h_den.values(flow=True)[..., None, None]

    # set the variations
    if central_weights:
        # in the case we are computing a helper for the central weights, we don't need to fill all variations
        h_nom = h_nom[{var_ax_name: 0}]
        corr_coeffs.values(flow=True)[..., 1, :] = h_nom.values(flow=True)[..., None]
    else:
        corr_coeffs.values(flow=True)[..., 1, :] = h_nom.values(flow=True)

    if return_tensor:
        helper = makeCorrectionsTensor(
            corr_coeffs, ROOT.wrem.CentralCorrByHelicityHelper, tensor_rank=3
        )

        # override tensor_axes since the output is different here
        helper.tensor_axes = [vars_ax]

        return helper
    else:
        return corr_coeffs


def make_helicity_test_corrector(is_z=False, filename=None):

    # load hist_helicity cross sections from file
    with h5py.File(filename, "r") as h5file:
        results = input_tools.load_results_h5py(h5file)
        hist_helicity_xsec = results["Z"] if is_z else results["W"]

    coeffs = theory_tools.helicity_xsec_to_angular_coeffs(hist_helicity_xsec)

    coeffs_nom = coeffs[{"muRfact": 1.0j, "muFfact": 1.0j}].values()

    corr_ax = hist.axis.Boolean(name="corr")
    vars_ax = hist.axis.StrCategory(
        ["test_ai", "test_sigmaUL", "test_all"], name="vars"
    )

    axes_no_scale = coeffs.axes[:-2]
    corr_coeffs = hist.Hist(*axes_no_scale, corr_ax, vars_ax)

    corr_coeffs.values()[...] = coeffs_nom[..., None, None]

    # set synthetic test variation
    corr_coeffs.values()[..., 1, 1, 0] *= 1.1

    corr_coeffs.values()[..., :, 1, 1] *= 1.1

    corr_coeffs.values()[..., :, 1, 2] *= 1.1
    corr_coeffs.values()[..., 1, 1, 2] *= 1.2
    corr_coeffs.values()[..., 2, 1, 2] *= 1.3
    corr_coeffs.values()[..., 3, 1, 2] *= 1.4
    corr_coeffs.values()[..., 4, 1, 2] *= 1.5
    corr_coeffs.values()[..., 5, 1, 2] *= 1.6
    corr_coeffs.values()[..., 6, 1, 2] *= 1.7
    corr_coeffs.values()[..., 7, 1, 2] *= 1.8
    corr_coeffs.values()[..., 8, 1, 2] *= 1.9

    helper = makeCorrectionsTensor(
        corr_coeffs, ROOT.wrem.CentralCorrByHelicityHelper, tensor_rank=3
    )

    # override tensor_axes since the output is different here
    helper.tensor_axes = [vars_ax]

    return helper


def make_angular_coeff(sigmai_hist, ul_hist):
    return hh.divideHists(sigmai_hist, ul_hist, cutoff=0.0001)


def read_combined_corrs(procNames, generator, corr_files, axes=[], absy=True, rebin={}):
    h = None
    if not corr_files:
        return h

    for procName in procNames:
        if procName[0] == "W":
            proc_files = list(
                filter(lambda x: procName[:2].lower() in x.lower(), corr_files)
            )
            charge = 1 if procName[:2] == "Wp" else -1
        else:
            charge = 0
            proc_files = corr_files
        hproc = read_corr(generator, proc_files, charge, axes)
        h = hproc if not h else h + hproc

    if absy and "Y" in axes:
        h = hh.makeAbsHist(h, "Y")

    if rebin:
        h = hh.rebinHistMultiAx(h, rebin.keys(), rebin.values())

    return h


def read_corr(generator, corr_files, charge, axes=[]):
    if "scetlib" in generator:
        coeff = None
        if any("A4" in c for c in corr_files):
            coeff = "a4"
        if "dyturbo" in generator:
            scetlib_files = [x for x in corr_files if pathlib.Path(x).suffix == ".pkl"]
            if len(scetlib_files) != 2:
                raise ValueError(
                    f"scetlib_dyturbo correction requires two SCETlib files (resummed and FO singular). Found {len(scetlib_files)}"
                )
            if not any("nnlo_sing" in x for x in scetlib_files):
                raise ValueError("Must pass in a fixed order singular file")
            nnlo_sing_idx = 0 if "nnlo_sing" in scetlib_files[0] else 1
            resumf = scetlib_files[~nnlo_sing_idx]
            nnlo_singf = scetlib_files[nnlo_sing_idx]

            dyturbo_files = [x for x in corr_files if pathlib.Path(x).suffix == ".txt"]
            if len(dyturbo_files) != 1:
                raise ValueError(
                    "scetlib_dyturbo correction requires one DYTurbo file (fixed order contribution)"
                )

            corrh = input_tools.read_matched_scetlib_dyturbo_hist(
                resumf, nnlo_singf, dyturbo_files[0], axes, charge=charge, coeff=coeff
            )
        else:
            corrh = input_tools.read_scetlib_hist(
                corr_files[0], charge=charge, nonsing=None, flip_y_sign=coeff == "a4"
            )
    else:
        if generator == "matrix_radish":
            h = input_tools.read_matrixRadish_hist(corr_files[0], axes[0])
        elif generator == "dyturbo":
            h = input_tools.read_dyturbo_hist(corr_files, axes=axes, charge=charge)

        vars_ax = (
            h.axes["vars"]
            if "vars" in h.axes.name
            else hist.axis.StrCategory(["central"], name="vars")
        )
        hnD = hist.Hist(*h.axes, vars_ax)
        # Leave off the overflow, we won't use it anyway
        hnD[...] = np.reshape(h.values(), hnD.shape)
        corrh = hnD

    return corrh
