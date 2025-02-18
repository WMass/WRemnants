from copy import deepcopy

import hist
import numpy as np

from utilities import common
from wremnants import syst_tools, theory_corrections, theory_tools, theoryAgnostic_tools
from wums import logging

logger = logging.child_logger(__name__)


def add_out_of_acceptance(datasets, group, newGroupName=None):
    # Copy datasets from specified group to make out of acceptance contribution
    datasets_ooa = []
    for dataset in datasets:
        if dataset.group == group:
            ds = deepcopy(dataset)

            if newGroupName is None:
                ds.group = ds.group + "OOA"
            else:
                ds.group = newGroupName
            ds.out_of_acceptance = True

            datasets_ooa.append(ds)

    return datasets + datasets_ooa


def define_gen_level(df, gen_level, dataset_name, mode="w_mass"):
    # gen level definitions
    gen_levels = ["preFSR", "postFSR"]
    if gen_level not in gen_levels:
        raise ValueError(
            f"Unknown gen level '{gen_level}'! Supported gen level definitions are '{gen_levels}'."
        )

    logger.info(f"Using {gen_level} leptons")
    singlelep = mode[0] == "w" or "wlike" in mode

    if gen_level == "preFSR":
        df = theory_tools.define_prefsr_vars(df)

        # needed for fiducial phase space definition
        df = df.Alias("massVGen", "massVgen")
        df = df.Alias("ptVGen", "ptVgen")
        df = df.Alias("absYVGen", "absYVgen")
        df = df.Alias("chargeVGen", "chargeVgen")

        if singlelep:
            df = df.Alias("mTVGen", "mTVgen")

        if mode[0] == "w":
            df = df.Define("ptGen", "chargeVgen < 0 ? genl.pt() : genlanti.pt()")
            df = df.Define(
                "absEtaGen",
                "chargeVgen < 0 ? std::fabs(genl.eta()) : std::fabs(genlanti.eta())",
            )
        else:
            df = df.Define("ptGen", "event % 2 == 0 ? genl.pt() : genlanti.pt()")
            df = df.Define(
                "absEtaGen",
                "event % 2 == 0 ? std::fabs(genl.eta()) : std::fabs(genlanti.eta())",
            )
            df = df.Define("ptOtherGen", "event % 2 == 0 ? genlanti.pt() : genl.pt()")
            df = df.Define(
                "absEtaOtherGen",
                "event % 2 == 0 ? std::fabs(genlanti.eta()) : std::fabs(genl.eta())",
            )

    elif gen_level == "postFSR":
        df = theory_tools.define_postfsr_vars(df, mode=mode)

        df = df.Alias("ptGen", f"postfsrLep_pt")
        df = df.Alias("absEtaGen", f"postfsrLep_absEta")

        if singlelep:
            df = df.Alias("mTVGen", "postfsrMT")

        if mode[0] == "z":
            df = df.Alias("ptOtherGen", "postfsrOtherLep_pt")
            df = df.Alias("absEtaOtherGen", f"postfsrOtherLep_absEta")

            df = df.Alias("massVGen", "postfsrMV")
            df = df.Define("absYVGen", "postfsrabsYV")

        df = df.Alias("ptVGen", "postfsrPTV")
        df = df.Alias("chargeVGen", "postfsrChargeV")

    if "wlike" in mode:
        df = df.Define("qGen", "event % 2 == 0 ? -1 : 1")

    return df


def select_fiducial_space(df, select=True, accept=True, mode="w_mass", **kwargs):
    # Define a fiducial phase space and if select=True, either select events inside/outside
    # accept = True: select events in fiducial phase space
    # accept = False: reject events in fiducial pahse space
    fiducial = kwargs.get("fiducial")
    selmap = {
        x: None
        for x in [
            "pt_min",
            "pt_max",
            "abseta_max",
            "mass_min",
            "mass_max",
            "mtw_min",
        ]
    }

    selections = kwargs.get("selections", [])[:]
    if fiducial:
        logger.info(
            f"Using default fiducial settings for selection {fiducial} for analysis {mode}"
        )
        if fiducial not in ["inclusive", "masswindow"]:
            # Use unfolding values in gen script
            selmap["pt_min"], selmap["pt_max"] = common.get_default_ptbins(
                mode, gen="vgen" in mode
            )[1:]
            selmap["abseta_max"] = common.get_default_etabins(mode)[-1]
            if mode[0] == "w" or "wlike" in mode:
                selmap["mtw_min"] = common.get_default_mtcut(mode)
        elif fiducial == "masswindow" and mode[0] == "z":
            selmap["mass_min"], selmap["mass_max"] = common.get_default_mz_window()
    else:
        for k in selmap.keys():
            selmap[k] = kwargs.get(k)

    if selmap["abseta_max"] is not None:
        selections.append(f"absEtaGen < {selmap['abseta_max']}")
        if mode[0] == "z":
            selections.append(f"absEtaOtherGen < {selmap['abseta_max']}")

    if selmap["pt_min"] is not None:
        if "gen" in mode or "dilepton" in mode:
            selections.append(f"ptGen > {selmap['pt_min']}")
        if mode[0] == "z":
            selections.append(f"ptOtherGen > {selmap['pt_min']}")

    if selmap["pt_max"] is not None:
        if "gen" in mode or "dilepton" in mode:
            # Don't place explicit cut on lepton pT for unfolding of W/W-like, but do for gen selection
            selections.append(f"ptGen < {selmap['pt_max']}")
        if mode[0] == "z":
            selections.append(f"ptOtherGen < {selmap['pt_max']}")

    if selmap["mass_min"] is not None:
        selections.append(f"massVGen > {selmap['mass_min']}")

    if selmap["mass_max"] is not None:
        selections.append(f"massVGen < {selmap['mass_max']}")

    if selmap["mtw_min"] is not None:
        selections.append(f"mTVGen > {selmap['mtw_min']}")

    selection = " && ".join(selections)

    if selection:
        df = df.Define("acceptance", selection)
        logger.info(f"Applying fiducial selection '{selection}'")
    else:
        df = df.DefinePerSample("acceptance", "true")

    if select and accept:
        logger.debug("Select events in fiducial phase space")
        df = df.Filter("acceptance")
    elif select:
        logger.debug("Reject events in fiducial phase space")
        df = df.Filter("acceptance == 0")

    return df


def add_xnorm_histograms(
    results,
    df,
    args,
    dataset_name,
    corr_helpers,
    qcdScaleByHelicity_helper,
    unfolding_axes,
    unfolding_cols,
    add_helicity_axis=False,
):
    # add histograms before any selection
    df_xnorm = df
    df_xnorm = df_xnorm.DefinePerSample("exp_weight", "1.0")

    df_xnorm = theory_tools.define_theory_weights_and_corrs(
        df_xnorm, dataset_name, corr_helpers, args
    )

    df_xnorm = df_xnorm.Define("xnorm", "0.5")

    axis_xnorm = hist.axis.Regular(
        1, 0.0, 1.0, name="count", underflow=False, overflow=False
    )

    xnorm_axes = [axis_xnorm, *unfolding_axes]
    xnorm_cols = ["xnorm", *unfolding_cols]

    if add_helicity_axis:
        df_xnorm = theoryAgnostic_tools.define_helicity_weights(
            df_xnorm,
            filename=f"{common.data_dir}/angularCoefficients/w_z_moments_unfoldingBinning.hdf5",
        )

        from wremnants.helicity_utils import axis_helicity_multidim

        results.append(
            df_xnorm.HistoBoost(
                "xnorm",
                xnorm_axes,
                [*xnorm_cols, "nominal_weight_helicity"],
                tensor_axes=[axis_helicity_multidim],
            )
        )
    else:
        results.append(
            df_xnorm.HistoBoost("xnorm", xnorm_axes, [*xnorm_cols, "nominal_weight"])
        )

    syst_tools.add_theory_hists(
        results,
        df_xnorm,
        args,
        dataset_name,
        corr_helpers,
        qcdScaleByHelicity_helper,
        xnorm_axes,
        xnorm_cols,
        base_name="xnorm",
        addhelicity=add_helicity_axis,
        nhelicity=9,
    )


def reweight_to_fitresult(
    fitresult, axes, poi_type="nois", cme=13, process="Z", expected=False, flow=True
):
    # requires fitresult generated from 'fitresult_pois_to_hist.py'
    histname = "hist_" + "_".join([a.name for a in axes])
    if expected:
        histname += "_expected"

    import pickle

    with open(fitresult, "rb") as f:
        r = pickle.load(f)
        if process == "W":
            corrh_0 = r["results"][poi_type][f"chan_{str(cme).replace('.','p')}TeV"][
                "W_qGen0"
            ][histname]
            corrh_1 = r["results"][poi_type][f"chan_{str(cme).replace('.','p')}TeV"][
                "W_qGen1"
            ][histname]
        else:
            corrh = r["results"][poi_type][f"chan_{str(cme).replace('.','p')}TeV"][
                process
            ][histname]

    slices = [slice(None) for i in range(len(axes))]

    if "qGen" not in [a.name for a in axes]:
        # CorrectionsTensor needs charge axis
        if process == "Z":
            axes.append(hist.axis.Regular(1, -1, 1, name="chargeVGen", flow=False))
            slices.append(np.newaxis)
            values = corrh.values(flow=flow)
        elif process == "W":
            axes.append(hist.axis.Regular(2, -2, 2, name="chargeVGen", flow=False))
            slices.append(slice(None))
            values = np.stack(
                [corrh_0.values(flow=flow), corrh_1.values(flow=flow)], axis=-1
            )

    ch = hist.Hist(*axes, hist.axis.Regular(1, 0, 1, name="vars", flow=False))
    slices.append(np.newaxis)

    ch = theory_corrections.set_corr_ratio_flow(ch)
    ch.values(flow=flow)[...] = values[*slices]

    logger.debug(f"corrections from fitresult: {values}")

    from wremnants.correctionsTensor_helper import makeCorrectionsTensor

    return makeCorrectionsTensor(ch)
