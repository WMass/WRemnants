from pathlib import Path
from typing import Optional

import hist
import numpy as np
import ROOT
import uproot
from utilities import common

import narf
from wremnants.butojpsik_histograms import all_butojpsik_axes
from wums import logging

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

logger = logging.child_logger(__name__)
data_dir = common.data_dir


def add_jpsi_crctn_stats_unc_hists(
    args,
    df,
    axes,
    results,
    nominal_cols,
    nominal_cols_gen_smeared,
    calib_filepaths,
    jpsi_crctn_data_unc_helper,
    smearing_weights_procs,
    reco_sel_GF,
    dataset_name,
    isW,
    storage_type=hist.storage.Double(),
):
    df = df.DefinePerSample("bool_true", "true")
    df = df.DefinePerSample("bool_false", "false")

    if args.muonScaleVariation == "smearingWeightsSplines" or args.validationHists:
        if args.muonScaleVariation == "smearingWeightsSplines":
            jpsi_unc_helper = jpsi_crctn_data_unc_helper
        else:
            jpsi_unc_helper = make_jpsi_crctn_unc_helper(
                calib_filepaths["data_corrfile"][args.muonCorrData],
                calib_filepaths["tflite_file"],
                scale_var_method="smearingWeightsSplines",
                dummy_mu_scale_var=args.dummyMuScaleVar,
                dummy_var_mag=args.muonCorrMag,
                plot_dir=getattr(args, "intermediate_plot_dir", None),
            )
        df = df.Define(
            "muonScaleSyst_responseWeights_tensor_splines",
            jpsi_unc_helper,
            [
                f"{reco_sel_GF}_recoPt",
                f"{reco_sel_GF}_recoEta",
                f"{reco_sel_GF}_recoCharge",
                f"{reco_sel_GF}_genPt",
                f"{reco_sel_GF}_genEta",
                f"{reco_sel_GF}_genCharge",
                f"{reco_sel_GF}_response_weight",
                "nominal_weight",
            ],
        )
        # import pdb
        # pdb.set_trace()

        # print("\n\ndebuggg\n\n")
        #
        # df = df.Define("debug_nominal_isnan", f"std::isnan({reco_sel_GF}_response_weight[0].first)")
        # df = df.Define("debug_nominal_isinf", f"std::isinf({reco_sel_GF}_response_weight[0].first)")
        # df = df.Define("debug_nominal_iszero", f"{reco_sel_GF}_response_weight[0].first == 0.0")
        #
        ## Check tensor output
        # df = df.Define(
        #    "debug_tensor_first_val",
        #    "muonScaleSyst_responseWeights_tensor_splines(0, 0)"
        # )
        # df = df.Define("debug_tensor_isnan", "std::isnan(debug_tensor_first_val)")
        #
        ## Check number of kaons
        # df = df.Define("debug_nkaons", f"{reco_sel_GF}_recoPt.size()")
        # df = df.Define("debug_has_kaons", f"{reco_sel_GF}_recoPt.size() > 0")
        #
        ## Check response weight (without lambda, just check size)
        # df = df.Define("debug_response_weight_size", f"{reco_sel_GF}_response_weight.size()")
        #
        ## Get counts
        # print("\n=== Debugging Info ===")
        # print(f"Total events: {df.Count().GetValue()}")
        # print(f"{reco_sel_GF}_response_weight is NaN: {df.Filter('debug_nominal_isnan').Count().GetValue()}")
        # print(f"{reco_sel_GF}_response_weight is Inf: {df.Filter('debug_nominal_isinf').Count().GetValue()}")
        # print(f"{reco_sel_GF}_response_weight is 0: {df.Filter('debug_nominal_iszero').Count().GetValue()}")
        # print(f"Output tensor is NaN: {df.Filter('debug_tensor_isnan').Count().GetValue()}")
        # print(f"Events with kaons: {df.Filter('debug_has_kaons').Count().GetValue()}")
        #
        ## Sample some actual values using AsNumpy (this works with MT)
        # sample_df = df.Filter("debug_has_kaons")
        # sample_data = sample_df.AsNumpy(["nominal_weight", "debug_nkaons", "debug_tensor_first_val"])
        # print(f"\nSample values (first 30):")
        # for i in range(min(30, len(sample_data["nominal_weight"]))):
        #    print(f"  Event {i}: nom_weight={sample_data['nominal_weight'][i]:.3f}, "
        #        f"nkaons={sample_data['debug_nkaons'][i]}, "
        #        f"tensor_val={sample_data['debug_tensor_first_val'][i]:.3f}")
        #
        # df_nan = df.Filter("debug_tensor_isnan")
        # df_valid = df.Filter("!debug_tensor_isnan")
        #
        ## Get more info about NaN events
        # nan_data = df_nan.AsNumpy([
        #    f"{reco_sel_GF}_recoPt",
        #    f"{reco_sel_GF}_recoEta",
        #    f"{reco_sel_GF}_recoCharge",
        #    f"{reco_sel_GF}_genPt",
        #    f"{reco_sel_GF}_genEta",
        #    f"{reco_sel_GF}_genCharge",
        #    f"{reco_sel_GF}_response_weight",
        #    "nominal_weight"
        # ])
        # valid_data = df_valid.AsNumpy([
        #    f"{reco_sel_GF}_recoPt",
        #    f"{reco_sel_GF}_recoEta",
        #    f"{reco_sel_GF}_recoCharge",
        #    f"{reco_sel_GF}_genPt",
        #    f"{reco_sel_GF}_genEta",
        #    f"{reco_sel_GF}_genCharge",
        #    f"{reco_sel_GF}_response_weight",
        #    "nominal_weight"
        # ])
        #
        # print("\n=== Analyzing NaN Events ===")
        # print(f"NaN events: {df_nan.Count().GetValue()}")
        # print(f"Valid events: {df_valid.Count().GetValue()}")
        #
        ## Check first few NaN events in detail
        # if len(nan_data[f"{reco_sel_GF}_recoPt"]) > 0:
        #    print("\nFirst few NaN event details:")
        #    idx = 0
        #    for idx in range(5):
        #        recPt = nan_data[f"{reco_sel_GF}_recoPt"][idx]
        #        recEta = nan_data[f"{reco_sel_GF}_recoEta"][idx]
        #        recCharge = nan_data[f"{reco_sel_GF}_recoCharge"][idx]
        #        genPt = nan_data[f"{reco_sel_GF}_genPt"][idx] or "not there"
        #        genEta = nan_data[f"{reco_sel_GF}_genEta"][idx] or "not there"
        #        genCharge = nan_data[f"{reco_sel_GF}_genCharge"][idx] or "not there"
        #        response_weight = nan_data[f"{reco_sel_GF}_response_weight"][idx]
        #
        #        print(f"  recPt: {recPt}")
        #        print(f"  recEta: {recEta}")
        #        print(f"  recCharge: {recCharge}")
        #        print(f"  genPt: {genPt}")
        #        print(f"  genEta: {genEta}")
        #        print(f"  genCharge: {genCharge}")
        #        print(f"  response_weight size: {len(response_weight)}")
        #        if len(response_weight) > 0:
        #            print(f"  response_weight[0]: {response_weight[0]}")
        #
        #        print("\n")
        # print("\n")
        ## Check first few valid events in detail
        # if len(valid_data[f"{reco_sel_GF}_recoPt"]) > 0:
        #    print("\nFirst few valid event details:")
        #    idx = 0
        #    count=0
        #    for idx in range(50):
        #        recPt = valid_data[f"{reco_sel_GF}_recoPt"][idx]
        #        recEta = valid_data[f"{reco_sel_GF}_recoEta"][idx]
        #        recCharge = valid_data[f"{reco_sel_GF}_recoCharge"][idx]
        #        genPt = valid_data[f"{reco_sel_GF}_genPt"][idx] or "not there"
        #        genEta = valid_data[f"{reco_sel_GF}_genEta"][idx] or "not there"
        #        genCharge = valid_data[f"{reco_sel_GF}_genCharge"][idx] or "not there"
        #        response_weight = valid_data[f"{reco_sel_GF}_response_weight"][idx]
        #        if recPt == genPt:
        #            count+=1
        #            continue
        #
        #        print(f"  recPt: {recPt}")
        #        print(f"  recEta: {recEta}")
        #        print(f"  recCharge: {recCharge}")
        #        print(f"  genPt: {genPt}")
        #        print(f"  genEta: {genEta}")
        #        print(f"  genCharge: {genCharge}")
        #        print(f"  response_weight size: {len(response_weight)}")
        #        if len(response_weight) > 0:
        #            print(f"  response_weight[0]: {response_weight[0]}")
        #
        #        print("\n")
        #    print(f"{count} events with genPt == recPt")

        # df = df.Define(
        #    "debug_response_has_nan",
        #    f"""[&]() {{
        #        for (const auto& pair : {reco_sel_GF}_response_weight) {{
        #            if (std::isnan(pair.first) || std::isnan(pair.second)) return true;
        #        }}
        #        return false;
        #    }}()"""
        # )
        #
        # df_nan_response = df.Filter("debug_response_has_nan")
        # nan_response_data = df_nan_response.AsNumpy([
        #    f"{reco_sel_GF}_recoPt",
        #    f"{reco_sel_GF}_recoEta",
        #    f"{reco_sel_GF}_genPt",
        #    f"{reco_sel_GF}_genEta",
        # ])
        #
        # print("\n=== Events with NaN response weights ===")
        # print(f"Count: {df_nan_response.Count().GetValue()}")
        # print("\nFirst 10 NaN events:")
        # for i in range(min(10, len(nan_response_data[f"{reco_sel_GF}_recoPt"]))):
        #    print(f"Event {i}:")
        #    print(f"  recoPt: {nan_response_data[f'{reco_sel_GF}_recoPt'][i]}")
        #    print(f"  recoEta: {nan_response_data[f'{reco_sel_GF}_recoEta'][i]}")
        #    print(f"  genPt: {nan_response_data[f'{reco_sel_GF}_genPt'][i]}")
        #    print(f"  genEta: {nan_response_data[f'{reco_sel_GF}_genEta'][i]}")
        #
        # print("\n\ndebuggg\n\n")

        if args.validationHists:
            muonScaleSyst_responseWeights_splines = df.HistoBoost(
                "muonScaleSyst_responseWeights_splines",
                axes,
                [*nominal_cols, "muonScaleSyst_responseWeights_tensor_splines"],
                tensor_axes=jpsi_unc_helper.tensor_axes,
                storage=hist.storage.Double(),
            )
            results.append(muonScaleSyst_responseWeights_splines)

    # Set the nominal muon scale variation.
    # If the scale var is derived from smearingWeightsGaus on the smeared-GEN,
    # the nominal will be the transported variation on RECO
    if not args.muonScaleVariation == "smearingWeightsGaus":
        if args.muonScaleVariation == "smearingWeightsSplines":
            df = df.Define(
                "nominal_muonScaleSyst_responseWeights_tensor",
                "muonScaleSyst_responseWeights_tensor_splines",
            )
        elif args.muonScaleVariation == "massWeights":
            df = df.Define(
                "nominal_muonScaleSyst_responseWeights_tensor",
                "muonScaleSyst_responseWeights_tensor_massWeights",
            )

        fitaxes = axes
        fitcols = nominal_cols
        # curvature instead of pt for study
        df = df.Define("bkmm_kaon_curvature", "1. / bkmm_jpsimc_kaon1pt")
        fitcols = nominal_cols.copy()
        # fitcols.remove("bkmm_jpsimc_kaon1pt")
        # fitcols.append("bkmm_kaon_curvature")
        fitaxes = [all_butojpsik_axes[a] for a in fitcols]
        nominal_muonScaleSyst_responseWeights = df.HistoBoost(
            "nominal_muonScaleSyst_responseWeights",
            fitaxes,
            [*fitcols, "nominal_muonScaleSyst_responseWeights_tensor"],
            tensor_axes=jpsi_crctn_data_unc_helper.tensor_axes,
            storage=storage_type,
        )
        results.append(nominal_muonScaleSyst_responseWeights)
    return df


def make_jpsi_crctn_helpers(args, calib_filepaths, make_uncertainty_helper=False):
    if args.muonCorrMC in ["idealMC_massfit", "idealMC_lbltruth_massfit"]:
        mc_corrfile = calib_filepaths["mc_corrfile"][args.muonCorrMC]
        logger.warning(
            "You apply J/Psi massfit corrections on MC, this is currenlty not recommended!"
        )
    else:
        mc_corrfile = None
    if args.muonCorrData in ["massfit", "lbl_massfit"]:
        data_corrfile = calib_filepaths["data_corrfile"][args.muonCorrData]
    else:
        data_corrfile = None
    tflite_file = calib_filepaths["tflite_file"]
    mc_helper = make_jpsi_crctn_helper(filepath=mc_corrfile) if mc_corrfile else None
    data_helper = (
        make_jpsi_crctn_helper(filepath=data_corrfile) if data_corrfile else None
    )

    if make_uncertainty_helper:
        mc_unc_helper = (
            make_jpsi_crctn_unc_helper(
                filepath_correction=mc_corrfile,
                # filepath_tflite=tflite_file,
                # n_eta_bins=24,
                # n_eta_bins=24,
                scale_var_method=args.muonScaleVariation,
                # dummy_mu_scale_var=args.dummyMuScaleVar,
                # dummy_var_mag=args.muonCorrMag,
                include_covariance=False,
                plot_dir=getattr(args, "intermediate_plot_dir", None),
            )
            if mc_corrfile
            else None
        )
        data_unc_helper = (
            make_jpsi_crctn_unc_helper(
                filepath_correction=data_corrfile,
                scale_var_method=args.muonScaleVariation,
                scale_A=args.scale_A,
                scale_e=args.scale_e,
                scale_M=args.scale_M,
                include_covariance=False,
                central=True,
                plot_dir=getattr(args, "intermediate_plot_dir", None),
            )
            if data_corrfile
            else None
        )

        return mc_helper, data_helper, mc_unc_helper, data_unc_helper
    else:
        return mc_helper, data_helper


def make_jpsi_crctn_unc_helper(
    filepath_correction,
    scale_A=1.0,
    scale_e=1.0,
    scale_M=1.0,
    isW=False,
    scale_var_method="smearingWeightsSplines",
    include_covariance=False,
    central=False,
    central_eta_min=-1.4,
    central_eta_max=1.4,
    plot_dir=None,
):

    f = ROOT.TFile.Open(filepath_correction)
    A = f.Get("A")
    e = f.Get("e")
    M = f.Get("M")

    A = narf.root_to_hist(A, axis_names=["scale_eta"])
    e = narf.root_to_hist(e, axis_names=["scale_eta"])
    M = narf.root_to_hist(M, axis_names=["scale_eta"])

    if central:
        print()

        eta_axis_orig = A.axes["scale_eta"]
        neta_orig = eta_axis_orig.size

        # Calculate how many bins to keep on each side
        n_central_bins = 28
        print(f"Extracting central {n_central_bins} eta bins")
        start_idx = (neta_orig - n_central_bins) // 2
        end_idx = start_idx + n_central_bins - 1

        print(
            f"Original: {neta_orig} bins from {eta_axis_orig.edges[0]} to {eta_axis_orig.edges[-1]}"
        )
        print(f"Selecting bins {start_idx} to {end_idx}")

        A = A[start_idx : end_idx + 1]
        e = e[start_idx : end_idx + 1]
        M = M[start_idx : end_idx + 1]

        print(
            f"New: {A.axes['scale_eta'].size} bins from {A.axes['scale_eta'].edges[0]} to {A.axes['scale_eta'].edges[-1]}"
        )
        print()

    if include_covariance:
        cov = f.Get("covariance_matrix")
        cov = narf.root_to_hist(cov)

        # would have to include some shit for central option here too but not for now

    f.Close()

    axis_eta = A.axes["scale_eta"]
    neta = axis_eta.size
    n_scale_params = 3
    if plot_dir:
        _plot_scale_params_vs_eta(
            axis_eta,
            (("A", A), ("e", e), ("M", M)),
            filepath_correction,
            plot_dir,
        )

    if include_covariance:
        logger.info(f"Eigen decomposition for muon momentum scale uncertainties")
        cov = cov.values()
        nparmscov = cov.shape[0] // neta
        nvars = neta * n_scale_params

        variances_ref = np.stack([A.variances(), e.variances(), M.variances()], axis=-1)
        variances = np.reshape(np.diag(cov), (neta, nparmscov))[:, :n_scale_params]

        if not np.all(np.isclose(variances, variances_ref, atol=0.0)):
            raise ValueError(
                "Covariance matrix is not consistent with parameter uncertainties or parameters are not in the expected order."
            )

        cov = np.reshape(cov, (neta, nparmscov, neta, nparmscov))
        cov = cov[:, :n_scale_params, :, :n_scale_params]

        scales = [scale_A, scale_e, scale_M]
        for iparm, scale in enumerate(scales):
            cov[:, iparm, :, :] *= scale
            cov[:, :, :, iparm] *= scale

        cov = np.reshape(cov, (nvars, nvars))

        e, v = np.linalg.eigh(cov)
        var_mat = np.sqrt(e[None, :]) * v
        var_mat = np.reshape(var_mat, (neta, n_scale_params, nvars))

    else:
        logger.info(
            f"Ignoring correlations and assigning a variation to each (A, e, M) for each of {neta} eta bins (no eigen decomposition)"
        )
        nvars = neta * n_scale_params

        A_unc = np.sqrt(A.variances()) * scale_A
        e_unc = np.sqrt(e.variances()) * scale_e
        M_unc = np.sqrt(M.variances()) * scale_M
        print("A uncertainties:\n  ", A_unc)
        print("e uncertainties:\n  ", e_unc)
        print("M uncertainties:\n  ", M_unc)

        if plot_dir:
            _plot_scale_param_uncs(
                axis_eta,
                (("A_unc", A_unc), ("e_unc", e_unc), ("M_unc", M_unc)),
                filepath_correction,
                plot_dir,
            )

        uncs = np.stack([A_unc, e_unc, M_unc], axis=-1)

        # each variation == one parameter in one eta bin
        var_mat = np.zeros((neta, n_scale_params, nvars))

        ivar = 0
        for ieta in range(neta):
            for iparam in range(n_scale_params):
                var_mat[ieta, iparam, ivar] = uncs[ieta, iparam]
                ivar += 1
        print("var_mat shape:", var_mat.shape)
        # print("var_mat for first few eta bins:")
        # for ieta in range(5):
        #    print(f"Eta bin {ieta}:", var_mat[ieta, :, :])

    axis_scale_params = hist.axis.Integer(
        0, n_scale_params, underflow=False, overflow=False, name="scale_params"
    )
    axis_scale_params_unc = hist.axis.Integer(
        0, nvars, underflow=False, overflow=False, name="unc"
    )

    hist_scale_params_unc = hist.Hist(
        axis_eta, axis_scale_params, axis_scale_params_unc
    )
    hist_scale_params_unc[...] = var_mat

    hist_scale_params_unc_cpp = narf.hist_to_pyroot_boost(
        hist_scale_params_unc, tensor_rank=2
    )

    if scale_var_method == "smearingWeightsGaus":
        helper = ROOT.wrem.JpsiCorrectionsUncHelper[
            type(hist_scale_params_unc_cpp).__cpp_name__
        ](ROOT.std.move(hist_scale_params_unc_cpp))
    elif scale_var_method == "smearingWeightsSplines":
        helper = ROOT.wrem.JpsiCorrectionsUncHelperSplines[
            type(hist_scale_params_unc_cpp).__cpp_name__
        ](ROOT.std.move(hist_scale_params_unc_cpp))
    elif scale_var_method == "massWeights":
        nweights = 21 if isW else 23
        helper = ROOT.wrem.JpsiCorrectionsUncHelper_massWeights[
            type(hist_scale_params_unc_cpp).__cpp_name__, nweights
        ](ROOT.std.move(hist_scale_params_unc_cpp))
    helper.tensor_axes = (hist_scale_params_unc.axes["unc"], common.down_up_axis)
    return helper


def make_jpsi_crctn_helper(filepath):
    f = uproot.open(filepath)

    # TODO: convert variable axis to regular if the bin width is uniform
    A, e, M = [x.to_hist() for x in [f["A"], f["e"], f["M"]]]

    # TODO: make this into a function in utilities/boostHistHelpers
    if (A.axes != e.axes) or (e.axes != M.axes):
        raise RuntimeError("A, e, M histograms have different axes!")
    else:
        axis_param = hist.axis.Regular(
            3, 0, 3, underflow=False, overflow=False, name="param"
        )
        hist_comb = hist.Hist(*A.axes, axis_param, storage=hist.storage.Double())
        hist_comb.view()[...] = np.stack([x.values() for x in [A, e, M]], axis=-1)

    hist_comb_cpp = narf.hist_to_pyroot_boost(hist_comb, tensor_rank=1)
    jpsi_crctn_helper = ROOT.wrem.JpsiCorrectionsRVecHelper[
        type(hist_comb_cpp).__cpp_name__
    ](ROOT.std.move(hist_comb_cpp))
    return jpsi_crctn_helper


def define_jpsi_triggers(df, trigger_name="", cutflow={}):
    if trigger_name == "":
        logger.error("no trigger name provided, cannot filter DataFrame")
        return df, None
    logger.info(f"HLT selection: {trigger_name}")
    df = df.Filter(f"HLT_{trigger_name}")
    cutflow = df.SumAndCount("weight")
    return df, cutflow


def bkmm_selections(df, dataset_name, selections):
    cutflow = {}
    dfs_per_cut = []

    logger.info("Enforce at least 1 bkmm dimuon candidate")
    df = df.Filter("nbkmm > 0")
    cutflow["bkmm dimuon cands > 0"] = df.SumAndCount("weight")
    dfs_per_cut.append(df)

    if dataset_name == "signalBuToJpsiK":
        logger.debug("Defining signal specific mask")
        df = df.Define("bkmm_passes", "bkmm_gen_pdgId != 0")
    else:
        df = df.Define(
            "bkmm_passes", "ROOT::VecOps::RVec<bool>(bkmm_mm_index.size(), true)"
        )

    for cutflow_name, description, selection_func in selections:
        logger.info(description)
        df = selection_func(df)
        df = _apply_filter(df, cutflow_name)
        cutflow[cutflow_name] = df.SumAndCount("weight")
        dfs_per_cut.append(df)

    return df, cutflow, dfs_per_cut


def select_only_passing_bkmm_candidates(
    df,
    signal: bool = False,
    select_best: bool = False,
    gen_match_nonsignal: bool = False,
    gen_filter_stats=None,
    dataset_name: Optional[str] = None,
):
    logger.info("Selecting passing bkmm candidates")

    # Signal - gen match (and only that one!)
    if signal:
        logger.info("Requiring gen-matched candidate for signal")

        # find the first (and only) gen-matched candidate, if passing
        df = df.Define(
            "bkmm_gen_match",
            """
            int gen_idx = -1;
            for (size_t i = 0; i < bkmm_passes.size(); ++i) {
                if (bkmm_passes[i] && bkmm_gen_pdgId[i] != 0) {
                    gen_idx = static_cast<int>(i);
                    break;
                }
            }
            return gen_idx;
            """,
        )

        # filter out events where no gen-matched candidate passed all selections
        df = df.Filter("bkmm_gen_match >= 0")

        # get corresponding dimuon index
        df = df.Define("mm_gen_match", "bkmm_mm_index[bkmm_gen_match]")

        # Extract only gen-matched candidate (vector to scalar)
        all_columns = df.GetColumnNames()
        for col in all_columns:
            col_name = str(col)
            col_type = df.GetColumnType(col_name)
            if "RVec" in col_type or "vector<" in col_type:
                if col_name.startswith("bkmm_"):
                    # Keep bkmm_* as length-1 vectors to match non-signal select_best branch.
                    df = df.Redefine(
                        col_name,
                        f"ROOT::VecOps::Take({col_name}, ROOT::RVec<int>{{bkmm_gen_match}})",
                    )
                if col_name.startswith("mm_"):
                    df = df.Redefine(col_name, f"{col_name}[mm_gen_match]")

        # Keep the same downstream interface used by non-signal paths.
        df = df.Alias("temp_kaon_genPt", "bkmm_gen_kaon_pt")
        df = df.Alias("temp_kaon_genEta", "bkmm_kaon_eta")
        df = df.Alias("temp_kaon_genCharge", "bkmm_kaon_charge")

        df = df.Redefine("nbkmm", "1")

        return df

    # Select only first passing candidate
    if select_best:
        logger.info("Selecting passing bkmm candidate with max vtx prob")

        df = df.Define(
            "bkmm_best_idx",
            """
            int best_idx = -1;
            float best_prob = -1.f;
            for (size_t i = 0; i < bkmm_passes.size(); ++i) {
                if (!bkmm_passes[i]) continue;
                float prob = bkmm_jpsimc_vtx_prob[i];
                if (prob > best_prob) {
                    best_prob = prob;
                    best_idx = static_cast<int>(i);
                }
            }
            return best_idx;
            """,
        )
        df = df.Filter("bkmm_best_idx >= 0")
        df = df.Define("mm_best_idx", "bkmm_mm_index[bkmm_best_idx]")

        if gen_match_nonsignal:
            # Find gen-matched index
            df = df.Define(
                "bkmm_gen_match_idx",
                """
                int gen_idx = -1;
                for (size_t i = 0; i < bkmm_gen_pdgId.size(); ++i) {
                    if (bkmm_gen_pdgId[i] != 0) {
                        gen_idx = static_cast<int>(i);
                        break;
                    }
                }
                return gen_idx;
                """,
            )

            #######################                   FILTER OUT THE -999s FOR NOW CUZ CAUSING NANs

            before_filter_weight = None
            after_filter_weight = None
            filtered_evt_count = None
            if gen_filter_stats is not None and dataset_name is not None:
                before_filter_weight = df.Sum("weight")

            df = df.Filter(
                "bkmm_gen_match_idx >= 0",
                "Require gen-matched candidate for non-signal",
            )

            if gen_filter_stats is not None and dataset_name is not None:
                after_filter_weight = df.Sum("weight")
                filtered_evt_count = df.Count()
                gen_filter_stats[dataset_name] = (
                    after_filter_weight,
                    before_filter_weight,
                    filtered_evt_count,
                )

            #######################                   FILTER OUT THE -999s FOR NOW CUZ CAUSING NANs

            # Extract gen values directly - no intermediate columns
            df = df.Define(
                "temp_kaon_genPt",  # real
                "ROOT::RVec<float>{bkmm_gen_match_idx >= 0 ? bkmm_gen_kaon_pt[bkmm_gen_match_idx] : -999.0f}",
            )
            df = df.Define(
                "temp_kaon_genEta",  # LIE !!!
                "ROOT::RVec<float>{bkmm_gen_match_idx >= 0 ? bkmm_kaon_eta[bkmm_gen_match_idx] : -999.0f}",
            )
            df = df.Define(
                "temp_kaon_genCharge",  # LIE !!!
                "ROOT::RVec<int>{bkmm_gen_match_idx >= 0 ? bkmm_kaon_charge[bkmm_gen_match_idx] : -999}",
            )

        all_columns = df.GetColumnNames()
        for col in all_columns:
            col_name = str(col)
            if "bkmm_kaon_shit" in col_name:
                continue
            col_type = df.GetColumnType(col_name)
            if "RVec" in col_type or "vector<" in col_type:
                if col_name.startswith("bkmm_"):
                    # df = df.Redefine(col_name, f"{col_name}[bkmm_best_idx]")
                    df = df.Redefine(
                        col_name,
                        f"ROOT::VecOps::Take({col_name}, ROOT::RVec<int>{{bkmm_best_idx}})",
                    )  # TRYING TO SET VECTORS OF LENGTH ONE FOR UNCERTAINTY HELPERS
                if col_name.startswith("mm_"):
                    df = df.Redefine(col_name, f"{col_name}[mm_best_idx]")

        # if gen_match_nonsignal:
        #    df = df.Alias("bkmm_kaon_genPt", "temp_kaon_genPt")
        #    df = df.Alias("bkmm_kaon_genEta", "temp_kaon_genEta")
        #    df = df.Alias("bkmm_kaon_genCharge", "temp_kaon_genCharge")

        df = df.Redefine("nbkmm", "1")
        return df

    # else: keep all passing candidates
    logger.info("Keeping all passing bkmm candidates")

    # Update scalar
    df = df.Redefine(
        "nbkmm",
        """
        int count = 0;
        for (bool pass : bkmm_passes) {
            if (pass) count++;
        }
        return count;
        """,
    )

    # mask for mm candidates corresponding to passing bkmm
    df = df.Define(
        "mm_passes",
        """
        ROOT::VecOps::RVec<bool> mm_mask(mm_kin_pt.size(), false);
        for (size_t i = 0; i < bkmm_passes.size(); ++i) {
            if (bkmm_passes[i]) {
                int mm_idx = bkmm_mm_index[i];
                if (mm_idx >= 0 && mm_idx < mm_mask.size()) {
                    mm_mask[mm_idx] = true;
                }
            }
        }
        return mm_mask;
        """,
    )

    # filter all bkmm_ and mm_ vectors to only passing candidates
    all_columns = df.GetColumnNames()
    for col in all_columns:
        col_name = str(col)
        col_type = df.GetColumnType(col_name)

        if "RVec" in col_type or "vector<" in col_type:
            if col_name.startswith("bkmm_") and col_name != "bkmm_passes":
                df = df.Redefine(col_name, f"{col_name}[bkmm_passes]")
            elif col_name.startswith("mm_") and col_name != "mm_passes":
                df = df.Redefine(col_name, f"{col_name}[mm_passes]")

    return df


def select_opposite_sign_dimuon(df):
    """Require opposite-sign dimuon candidates."""
    condition = """
    ROOT::VecOps::RVec<bool> passes = bkmm_passes;
    for (size_t i = 0; i < bkmm_mm_index.size(); ++i) {
        if (!passes[i]) continue;
        int idx = bkmm_mm_index[i];
        if (idx < 0 || idx >= mm_mu1_index.size() || idx >= mm_mu2_index.size()) {
            passes[i] = false;
            continue;
        }
        int mu1_idx = mm_mu1_index[idx];
        int mu2_idx = mm_mu2_index[idx];
        if (mu1_idx < 0 || mu1_idx >= Muon_charge.size() ||
            mu2_idx < 0 || mu2_idx >= Muon_charge.size() ||
            Muon_charge[mu1_idx] * Muon_charge[mu2_idx] >= 0) {
            passes[i] = false;
        }
    }
    return passes;
    """
    return df.Redefine("bkmm_passes", condition)


def select_kaon_eta(df, max_eta):
    """Require |eta| < max_eta for kaon."""
    # condition = _generate_abs_kaon_condition("bkmm_kaon_eta", "<", max_eta)
    condition = _generate_abs_kaon_condition("bkmm_jpsimc_kaon1eta", "<", max_eta)
    return df.Redefine("bkmm_passes", condition)


def select_kaon_pt(df, max_pt):
    """Require pT < max_pt GeV for kaon."""
    # condition = _generate_kaon_condition("bkmm_kaon_pt", "<", max_pt)
    condition = _generate_kaon_condition("bkmm_jpsimc_kaon1pt", "<", max_pt)
    return df.Redefine("bkmm_passes", condition)


def select_muon_eta(df, max_eta):
    """Require |eta| < max_eta for both muons."""
    condition = _generate_abs_muon_pair_condition("Muon_eta", "<", max_eta)
    return df.Redefine("bkmm_passes", condition)


def select_muon_pt(df, min_pt):
    """Require pT > min_pt GeV for both muons."""
    condition = _generate_muon_pair_condition("Muon_pt", ">", min_pt)
    return df.Redefine("bkmm_passes", condition)


def select_muon_softmva(df, min_mva):
    """Require soft MVA > min_mva for both muons."""
    condition = _generate_muon_pair_condition("Muon_softMva", ">", min_mva)
    return df.Redefine("bkmm_passes", condition)


def select_dimuon_pt(df, min_pt):
    """Require dimuon pT > min_pt GeV."""
    condition = _generate_dimuon_condition("mm_kin_pt", ">", min_pt)
    return df.Redefine("bkmm_passes", condition)


def select_dimuon_alphabs(df, max_alphabs):
    """Require dimuon alphaBS < max_alphabs."""
    condition = _generate_dimuon_condition("mm_kin_alphaBS", "<", max_alphabs)
    return df.Redefine("bkmm_passes", condition)


def select_dimuon_vtx_prob(df, min_prob):
    """Require dimuon vertex probability > min_prob."""
    condition = _generate_dimuon_condition("mm_kin_vtx_prob", ">", min_prob)
    return df.Redefine("bkmm_passes", condition)


def select_dimuon_sl3d(df, min_sl3d):
    """Require dimuon 3D significance > min_sl3d."""
    condition = _generate_dimuon_condition("mm_kin_sl3d", ">", min_sl3d)
    return df.Redefine("bkmm_passes", condition)


def select_bkmm_vtx_prob(df, min_prob):
    """Require bkmm J/psi+MC vertex probability > min_prob."""
    condition = _generate_bkmm_condition("bkmm_jpsimc_vtx_prob", ">", min_prob)
    return df.Redefine("bkmm_passes", condition)


def select_bkmm_mass_window(df, center, width):
    """Require |bkmm mass - center| < width GeV."""
    condition = _generate_mass_window_condition("bkmm_jpsimc_mass", center, width)
    return df.Redefine("bkmm_passes", condition)


# def select_vtx_constraints(df, cand, constraint, dim):
#    """Require cand (B or dimuon) vtx_{dim}"""
#    condition =
#    return df.Redefine("bkmm_passes", condition)


def select_bkmm_bmm_bdt(df, value):
    """Select greater than value on bkmm bmm bdt variable"""  # NOTE: NEED TO CONFIRM THIS DOESN'T USE KAON AT ALL (variable name suggests it...)
    condition = _generate_bdt_condition("bkmm_bmm", value)
    return df.Redefine("bkmm_passes", condition)


def _generate_kaon_condition(variable, operator, threshold):
    """Generate C++ code for conditions on kaon properties."""
    ops = {">": ">", "<": "<", ">=": ">=", "<=": "<="}
    op = ops[operator]

    return f"""
    ROOT::VecOps::RVec<bool> passes = bkmm_passes;
    for (size_t i = 0; i < bkmm_passes.size(); ++i) {{
        if (!passes[i]) continue;
        if (i >= {variable}.size() || !({variable}[i] {op} {threshold})) {{
            passes[i] = false;
        }}
    }}
    return passes;
    """


def _generate_abs_kaon_condition(variable, operator, threshold):
    """Generate C++ code for conditions on abs() of kaon properties."""
    ops = {">": ">", "<": "<", ">=": ">=", "<=": "<="}
    op = ops[operator]

    return f"""
    ROOT::VecOps::RVec<bool> passes = bkmm_passes;
    for (size_t i = 0; i < bkmm_passes.size(); ++i) {{
        if (!passes[i]) continue;
        if (i >= {variable}.size() || !(abs({variable}[i]) {op} {threshold})) {{
            passes[i] = false;
        }}
    }}
    return passes;
    """


def _generate_muon_pair_condition(variable, operator, threshold):
    """Generate C++ code for conditions on both muons in a dimuon pair."""
    ops = {">": ">", "<": "<", ">=": ">=", "<=": "<="}
    op = ops[operator]

    return f"""
    ROOT::VecOps::RVec<bool> passes = bkmm_passes;
    for (size_t i = 0; i < bkmm_mm_index.size(); ++i) {{
        if (!passes[i]) continue;
        int idx = bkmm_mm_index[i];
        if (idx < 0 || idx >= mm_mu1_index.size() || idx >= mm_mu2_index.size()) {{
            passes[i] = false;
            continue;
        }}
        int mu1_idx = mm_mu1_index[idx];
        int mu2_idx = mm_mu2_index[idx];
        if (mu1_idx < 0 || mu1_idx >= {variable}.size() ||
            mu2_idx < 0 || mu2_idx >= {variable}.size() ||
            !({variable}[mu1_idx] {op} {threshold} && {variable}[mu2_idx] {op} {threshold})) {{
            passes[i] = false;
        }}
    }}
    return passes;
    """


def _generate_dimuon_condition(variable, operator, threshold):
    """Generate C++ code for conditions on dimuon properties."""
    ops = {">": ">", "<": "<", ">=": ">=", "<=": "<="}
    op = ops[operator]

    return f"""
    ROOT::VecOps::RVec<bool> passes = bkmm_passes;
    for (size_t i = 0; i < bkmm_mm_index.size(); ++i) {{
        if (!passes[i]) continue;
        int idx = bkmm_mm_index[i];
        if (idx < 0 || idx >= {variable}.size() || !({variable}[idx] {op} {threshold})) {{
            passes[i] = false;
        }}
    }}
    return passes;
    """


def _generate_bkmm_condition(variable, operator, threshold):
    """Generate C++ code for conditions on bkmm candidate properties."""
    ops = {">": ">", "<": "<", ">=": ">=", "<=": "<="}
    op = ops[operator]

    return f"""
    ROOT::VecOps::RVec<bool> passes = bkmm_passes;
    for (size_t i = 0; i < {variable}.size(); ++i) {{
        if (!passes[i]) continue;
        if (!({variable}[i] {op} {threshold})) {{
            passes[i] = false;
        }}
    }}
    return passes;
    """


def _generate_abs_muon_pair_condition(variable, operator, threshold):
    """Generate C++ code for conditions on abs() of muon properties."""
    ops = {">": ">", "<": "<", ">=": ">=", "<=": "<="}
    op = ops[operator]

    return f"""
    ROOT::VecOps::RVec<bool> passes = bkmm_passes;
    for (size_t i = 0; i < bkmm_mm_index.size(); ++i) {{
        if (!passes[i]) continue;
        int idx = bkmm_mm_index[i];
        if (idx < 0 || idx >= mm_mu1_index.size() || idx >= mm_mu2_index.size()) {{
            passes[i] = false;
            continue;
        }}
        int mu1_idx = mm_mu1_index[idx];
        int mu2_idx = mm_mu2_index[idx];
        if (mu1_idx < 0 || mu1_idx >= {variable}.size() ||
            mu2_idx < 0 || mu2_idx >= {variable}.size() ||
            !(abs({variable}[mu1_idx]) {op} {threshold} && abs({variable}[mu2_idx]) {op} {threshold})) {{
            passes[i] = false;
        }}
    }}
    return passes;
    """


def _generate_mass_window_condition(variable, center, width):
    """Generate C++ code for mass window cut."""
    return f"""
    ROOT::VecOps::RVec<bool> passes = bkmm_passes;
    for (size_t i = 0; i < {variable}.size(); ++i) {{
        if (!passes[i]) continue;
        if (!(abs({variable}[i] - {center}) < {width})) {{
            passes[i] = false;
        }}
    }}
    return passes;
    """


def _generate_bdt_condition(fit, value):
    """Code for bdt cut"""
    variable = f"{fit}_bdt"
    return f"""
    ROOT::VecOps::RVec<bool> passes = bkmm_passes;
    for (size_t i = 0; i < {variable}.size(); ++i) {{
        if (!passes[i]) continue;
        if (!({variable}[i] > {value})) {{
            passes[i] = false;
        }}
    }}
    return passes
    """


def _apply_filter(df, cutflow_name):
    """Apply filter based on bkmm_passes mask."""
    filter_name = f"has_passing_{cutflow_name.replace(' ', '_').replace('|', '').replace('<', '').replace('>', '').replace('.', 'p')}"
    df = df.Define(
        filter_name,
        """
        bool result = false;
        for (bool pass : bkmm_passes) {
            if (pass) { result = true; break; }
        }
        return result;
        """,
    )
    return df.Filter(filter_name)


def analyze_candidate_multiplicity(df):
    logger.info("Analyzing candidate multiplicity")

    df_with_counts = df.Define("n_bkmm_candidates", "bkmm_mm_index.size()")

    total_events = df_with_counts.SumAndCount("weight")[0].GetValue()
    events_with_1 = (
        df_with_counts.Filter("n_bkmm_candidates == 1")
        .SumAndCount("weight")[0]
        .GetValue()
    )
    events_with_2 = (
        df_with_counts.Filter("n_bkmm_candidates == 2")
        .SumAndCount("weight")[0]
        .GetValue()
    )
    events_with_3 = (
        df_with_counts.Filter("n_bkmm_candidates == 3")
        .SumAndCount("weight")[0]
        .GetValue()
    )
    events_with_4plus = (
        df_with_counts.Filter("n_bkmm_candidates >= 4")
        .SumAndCount("weight")[0]
        .GetValue()
    )

    logger.info(f"Candidate multiplicity:")
    logger.info(f"  Total events: {total_events}")
    logger.info(
        f"  Events with 1 candidate: {events_with_1} ({100*events_with_1/total_events:.1f}%)"
    )
    logger.info(
        f"  Events with 2 candidates: {events_with_2} ({100*events_with_2/total_events:.1f}%)"
    )
    logger.info(
        f"  Events with 3 candidates: {events_with_3} ({100*events_with_3/total_events:.1f}%)"
    )
    logger.info(
        f"  Events with 4+ candidates: {events_with_4plus} ({100*events_with_4plus/total_events:.1f}%)"
    )

    return df


def inspect_dataframe(df):
    cols = df.GetColumnNames()
    for col in cols:
        if any(x in col for x in ["bkmm", "mm_", "Muon"]):
            col_type = df.GetColumnType(col)
            logger.info(f"Column {col}: type = {col_type}")


def _plot_scale_params_vs_eta(axis_eta, histograms, source_path, output_dir):
    if plt is None:
        logger.warning("matplotlib not available, skipping scale-parameter plot.")
        return
    if not output_dir:
        return
    eta_centers = axis_eta.centers
    source = Path(source_path) if source_path else Path("corrections")
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{source.stem}_scale_params_vs_eta.png"

    for idx, (label, hist_obj) in enumerate(histograms):
        values = hist_obj.values()
        print(
            f"{label}: len(eta_centers)={len(eta_centers)}, len(values)={len(values)}"
        )
        print(f"{label}: values = {values}")
        print(f"{label}: eta_centers = {eta_centers}")
        color = f"C{idx % 10}"
        fig, ax = plt.subplots()
        ax.plot(
            eta_centers,
            values,
            label=label,
            color=color,
            linestyle="None",
            marker="o",
            markersize=3,
        )
        ax.set_xlabel("eta")
        ax.set_ylabel(f"{label} value")
        ax.set_title(f"{label} vs eta")
        # if label == "A":
        #    ax.set_ylim(-0.0001, 0.0003)
        # elif label == "e":
        #    ax.set_ylim(-0.00075, 0.001)
        # elif label == "M":
        #    ax.set_ylim(-1e-5, 2e-5)
        ax.legend()
        fig.tight_layout()
        individual_outpath = outpath.with_name(f"{outpath.stem}_{label}.png")
        fig.savefig(individual_outpath)
        plt.close(fig)
        logger.info(f"Saved {label} scale-parameter plot to {individual_outpath}")


def _plot_scale_param_uncs(axis_eta, uncertainties, source_path, output_dir):
    if plt is None:
        logger.warning("matplotlib not available, skipping scale-uncertainty plot.")
        return
    if not output_dir:
        return
    eta_centers = axis_eta.centers
    source = Path(source_path) if source_path else Path("corrections")
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    base = outdir / f"{source.stem}_scale_params_unc"

    for idx, (label, values) in enumerate(uncertainties):
        color = f"C{idx % 10}"
        fig, ax = plt.subplots()
        ax.plot(
            eta_centers,
            values,
            label=label,
            color=color,
            linestyle="None",
            marker="o",
            markersize=3,
        )

        ax.set_xlabel("eta")
        ax.set_ylabel(f"{label}")
        ax.set_title(f"{label} vs eta")
        fig.tight_layout()
        individual_outpath = base.with_name(f"{base.name}_{label}.png")
        fig.savefig(individual_outpath)
        plt.close(fig)
        logger.info(f"Saved {label} uncertainty plot to {individual_outpath}")
