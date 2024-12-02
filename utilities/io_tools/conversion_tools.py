import itertools
import re

import hist
import numpy as np

from narf import ioutils
from utilities import logging
from utilities.io_tools import combinetf_input

logger = logging.child_logger(__name__)


def get_number_from_string(
    s,
    choose_index=-1,  # >= 0 to choose single index when more are found
    return_list=False,
    return_value_if_absent=0,
):

    los = [int(i) for i in re.findall(r"\d+", s)]
    if return_list:
        return los

    if len(los) == 0:
        # returning as integer to be used in sorting expressions
        return return_value_if_absent
    elif len(los) == 1:
        return los[0]
    else:
        if choose_index < 0:
            return tuple(los)
        else:
            return los[min(choose_index, len(los) - 1)]


def get_hepdata_label(name, input_dict=None):

    if input_dict is not None and name in input_dict.keys():
        return input_dict[name]

    postfix = " [avg.]" if "SymAvg" in name else " [diff.]" if "SymDiff" in name else ""
    if "Scale_correction_unc" in name:
        parID = get_number_from_string(name, choose_index=0)
        newname = "$\\mathit{p}_{T}^{μ}$ calib. J/ψ stat. " + f"({parID})"
    elif "ScaleClos_correction_unc" in name:
        parID = get_number_from_string(name, choose_index=0)
        newname = "$\\mathit{p}_{T}^{μ}$ calib. Z closure stat. " + f"({parID})"
    elif "Resolution_correction_smearing_variation" in name:
        parID = get_number_from_string(name, choose_index=0)
        newname = "$\\mathit{p}_{T}^{μ}$ calib. resolution. " + f"({parID})"
    elif re.match("QCDscale(W|Z)", name):
        # QCDscaleZinclusive_PtV0_13000helicity_2_SymDiff
        # QCDscaleZfine_PtV3_5helicity_1_SymAvg
        tokens = re.findall(r"\d+", name)
        pt_low, pt_high, ang_coeff = map(int, tokens)
        boson = "W" if "QCDscaleW" in name else "Z"
        if "inclusive_" in name:
            newname = f"$A_{ang_coeff}$ angular coeff., {boson}, inclusive"
        elif "fine_" in name:
            boson_pt = (
                "$\\mathit{p}_{T}^{" + boson + "}$ " + f"[{pt_low}, {pt_high}] GeV"
            )
            newname = f"$A_{ang_coeff}$ angular coeff., {boson}, {boson_pt}"
    elif name.startswith("effS"):
        ## TODO: this is a bit awkward ...
        # effSyst_reco_altBkg_etaDecorr33
        # effSyst_veto_reco_etaDecorr27
        # effStat_trigger_eta9pt7q0
        # effStat_iso_eta9pt7qall
        step = ""
        tokens = name.split("_")
        if "veto" in name:
            step = f"veto {tokens[2]}" if "effSyst" in name else "veto"
        else:
            step = tokens[1]
        eta_text = "$\\eta^{\\mu}$"
        pt_text = "$\\mathit{p}_{T}^{\\mu}$"
        syst_stat = ""
        eta_pt_charge_bin_ids = get_number_from_string(name, return_list=True)
        if "effStat" in name:
            syst_stat = "stat"
            eta_bin = eta_pt_charge_bin_ids[0]
            pt_bin = eta_pt_charge_bin_ids[1]
            if name.endswith("qall"):
                charge_text = ""
            elif name.endswith("q0"):
                charge_text = "$\\mathit{q}^{μ}$ +1"
            else:
                charge_text = "$\\mathit{q}^{μ}$ -1"
            eta_pt_charge_bin = (
                f"{eta_text} {eta_bin}, {pt_text} {pt_bin}, {charge_text}"
            )
        else:
            syst_stat = "syst"
            if "altBkg" in name:
                step += " alt. bkg "
            if "etaDecorr" in name:
                eta_bin = (
                    eta_pt_charge_bin_ids[0] - 1
                )  # subtract 1 since otherwise this index starts from 1
                eta_pt_charge_bin = f"{eta_text} {eta_bin}"
            elif "fullyCorr" in name:
                eta_pt_charge_bin = f"{eta_text} correlated"
        newname = "$ε_{" + syst_stat + "}^{μ}$ " + f"{step} {eta_pt_charge_bin}"
    elif "pdf" in name:
        base = name.split("Sym")[0]
        pdfEigen = get_number_from_string(base, choose_index=0)
        pdfSet = base.split(str(pdfEigen), 1)[
            1
        ]  # split only on first occurrence to select the PDF set
        newname = f"PDF {pdfSet} ({pdfEigen})"
    elif name.startswith("FakeSmoothing"):
        # FakeSmoothing_eta17_charge0_param3
        tokens = re.findall(r"\d+", name)
        pt_text = "$\\mathit{p}_{T}^{\\mu}$"
        eta_text = "$\\eta^{\\mu}$"
        if tokens[1] == 1:
            charge_text = "$\\mathit{q}^{μ}$ +1"
        else:
            charge_text = "$\\mathit{q}^{μ}$ -1"
        newname = f"Nonprompt {pt_text} smoothing, {eta_text} {tokens[0]}, {charge_text}, param. {tokens[2]}"
    else:
        newname = name

    newname += postfix
    return newname


def transform_poi(poi_type, meta):
    # scale poi type, in case of noi, convert into poi like form
    if poi_type in ["nois"]:
        prior_norm = meta["args"]["priorNormXsec"]
        scale_norm = (
            meta["args"]["scaleNormXsecHistYields"]
            if meta["args"]["scaleNormXsecHistYields"] is not None
            else 1
        )
        val = (
            lambda theta, scale, k=prior_norm, x=scale_norm: (1 + x) ** (k * theta)
            * scale
        )
        err = (
            lambda theta, err, scale, k=prior_norm, x=scale_norm: (
                k * np.log(1 + x) * (1 + x) ** (k * theta)
            )
            * scale
            * err
        )

        # val = lambda theta, scale, k=prior_norm, x=scale_norm: (1 + x * k * theta) * scale
        # err = lambda theta, scale, k=prior_norm, x=scale_norm: (x * k * theta) * scale
        return val, err
    else:
        val = lambda poi, scale: poi * scale
        err = lambda poi, err, scale: err * scale
        return val, err


def expand_flow(val, axes, flow_axes, var=None):
    # expand val and var arrays by ones for specified flow_axes in order to broadcast them into the histogram
    for a in axes:
        if a.name not in flow_axes:
            if a.traits.underflow:
                s_ = [s for s in val.shape]
                s_[axes.index(a)] = 1
                val = np.append(np.ones(s_), val, axis=axes.index(a))
                if var is not None:
                    var = np.append(np.ones(s_), var, axis=axes.index(a))
            if a.traits.overflow:
                s_ = [s for s in val.shape]
                s_[axes.index(a)] = 1
                val = np.append(val, np.ones(s_), axis=axes.index(a))
                if var is not None:
                    var = np.append(var, np.ones(s_), axis=axes.index(a))
    if var is None:
        return val
    else:
        return val, var


def combine_channels(meta, merge_gen_charge_W):
    # merge channels with same energy
    channel_energy = {
        "2017G": "5TeV",
        "2017H": "13TeV",
        "2016preVFP": "13TeV",
        "2016postVFP": "13TeV",
        "2017": "13TeV",
        "2018": "13TeV",
    }
    # merge electron and muon into lepton channels
    channel_flavor = {
        "e": "l",
        "mu": "l",
        "ee": "ll",
        "mumu": "ll",
    }

    channel_info = {}
    for chan, info in meta["channel_info"].items():
        if chan.endswith("masked"):
            continue
        channel = f"chan_{channel_energy[info['era']]}"
        if info["flavor"] in channel_flavor:
            channel += f"_{channel_flavor[info['flavor']]}"

        logger.debug(f"Merge channel {chan} into {channel}")

        lumi = info["lumi"]
        gen_axes = info["gen_axes"]

        if merge_gen_charge_W:
            if "W_qGen0" not in gen_axes or "W_qGen1" not in gen_axes:
                logger.debug(
                    "Can't merge W, it requires W_qGen0 and W_qGen1 as separate processes"
                )
            elif gen_axes["W_qGen0"] != gen_axes["W_qGen1"]:
                raise RuntimeError(
                    "Axes for different gen charges are diffenret, gen charges can't be merged"
                )
            else:
                logger.debug("Merge W_qGen0 and W_qGen1 into W with charge axis")
                axis_qGen = hist.axis.Regular(
                    2, -2.0, 2.0, underflow=False, overflow=False, name="qGen"
                )
                gen_axes = {
                    "W": [*gen_axes["W_qGen0"], axis_qGen],
                    **{
                        k: v
                        for k, v in gen_axes.items()
                        if k not in ["W_qGen0", "W_qGen1"]
                    },
                }
                # gen_axes["W"] = [*gen_axes["W_qGen0"], axis_qGen]

        if channel not in channel_info:
            channel_info[channel] = {
                "gen_axes": gen_axes,
                "lumi": lumi,
            }
        elif gen_axes == channel_info[channel]["gen_axes"]:
            channel_info[channel]["lumi"] += lumi
        else:
            channel_info[channel]["gen_axes"].update(gen_axes)

    return channel_info


def fitresult_pois_to_hist(
    infile,
    result=None,
    poi_types=None,
    translate_poi_types=True,
    merge_channels=True,
    grouped=True,
    uncertainties=None,
    expected=False,
    merge_gen_charge_W=True,
):
    # convert POIs in fitresult into histograms
    # uncertainties, use None to get all, use [] to get none
    # grouped=True for grouped uncertainties
    # Different channels can have different year, flavor final state, particle final state, sqrt(s),
    #   if merge_channels=True the lumi is added up for final states with different flavors or eras with same sqrt(s)

    # translate the name of the keys to be written out
    target_keys = {
        "pmaskedexp": "xsec",
        "sumpois": "xsec",
        "pmaskedexpnorm": "xsec_normalized",
        "sumpoisnorm": "xsec_normalized",
    }

    fitresult = combinetf_input.get_fitresult(infile.replace(".root", ".hdf5"))
    meta = ioutils.pickle_load_h5py(fitresult["meta"])
    meta_info = meta["meta_info"]

    if poi_types is None:
        if meta_info["args"]["poiAsNoi"]:
            poi_types = [
                "nois",
                "pmaskedexp",
                "pmaskedexpnorm",
                "sumpois",
                "sumpoisnorm",
                "ratiometapois",
            ]
        else:
            poi_types = [
                "mu",
                "pmaskedexp",
                "pmaskedexpnorm",
                "sumpois",
                "sumpoisnorm",
                "ratiometapois",
            ]

    if merge_channels:
        channel_info = combine_channels(meta, merge_gen_charge_W)
    else:
        channel_info = meta["channel_info"]

    if result is None:
        result = {}
    for poi_type in poi_types:
        logger.debug(f"Now at POI type {poi_type}")

        df = combinetf_input.read_impacts_pois(
            fitresult, poi_type=poi_type, group=grouped, uncertainties=uncertainties
        )
        if df is None or len(df) == 0:
            logger.warning(
                f"POI type {poi_type} not found in histogram, continue with next one"
            )
            continue

        # find all axes where the flow bins are included in the unfolding, needed for correct reshaping
        flow_axes = list(
            set(
                [
                    l
                    for i in df["Name"]
                    .apply(
                        lambda x: [
                            i[:-1]
                            for i in x.split("_")[1:-1]
                            if len(i) and i[-1] in ["U", "O"]
                        ]
                    )
                    .values
                    if i
                    for l in i
                ]
            )
        )

        action_val, action_err = transform_poi(poi_type, meta_info)

        poi_key = (
            target_keys.get(poi_type, poi_type) if translate_poi_types else poi_type
        )
        if poi_key not in result:
            result[poi_key] = {}
        for channel, info in channel_info.items():
            logger.debug(f"Now at channel {channel}")

            if poi_type in ["pmaskedexp", "sumpois"]:
                channel_scale = 1.0 / (info["lumi"] * 1000)
            else:
                channel_scale = 1
            if channel not in result[poi_key]:
                result[poi_key][channel] = {}
            for proc, gen_axes_proc in info["gen_axes"].items():
                logger.debug(f"Now at proc {proc}")

                if any(
                    poi_type.startswith(x)
                    for x in ["sum", "ratio", "chargemeta", "helmeta"]
                ):
                    if len(gen_axes_proc) <= 1:
                        logger.info(
                            f"Skip POI type {poi_type} since there is only {len(gen_axes_proc)} gen axis"
                        )
                        continue
                    # make all possible lower dimensional gen axes combinations; wmass only combinations including qGen
                    gen_axes_permutations = [
                        list(k)
                        for n in range(1, len(gen_axes_proc))
                        for k in itertools.combinations(gen_axes_proc, n)
                    ]
                else:
                    gen_axes_permutations = [
                        gen_axes_proc[:],
                    ]

                if proc not in result[poi_key][channel]:
                    result[poi_key][channel][proc] = {}

                for axes in gen_axes_permutations:
                    logger.debug(f"Now at axes {axes}")

                    if poi_type in ["helpois", "helmetapois"]:
                        if "helicitySig" not in [a.name for a in axes]:
                            continue
                        # replace helicity cross section axis by angular coefficient axis
                        axis_coeffs = hist.axis.Integer(
                            0, 8, name="A", overflow=False, underflow=False
                        )
                        axes = [
                            axis_coeffs if a.name == "helicitySig" else a for a in axes
                        ]

                    axes_names = [a.name for a in axes]
                    shape = [a.extent if a.name in flow_axes else a.size for a in axes]

                    # TODO: clean up hard coded treatment for cross section ratios
                    if poi_type == "ratiometapois":
                        if proc not in ["W_qGen0", "r_qGen_W"]:
                            continue

                        proc = "r_qGen_W"
                        if proc not in result[poi_key][channel]:
                            result[poi_key][channel][proc] = {}

                        data = combinetf_input.select_pois(
                            df, axes_names, base_processes=proc, flow=True
                        )
                        data = data.loc[
                            data["Name"].apply(lambda x: not x.endswith("totalxsec"))
                        ]
                    else:
                        data = combinetf_input.select_pois(
                            df, axes_names, base_processes=proc, flow=True
                        )

                    for u in filter(lambda x: x.startswith("err_"), data.keys()):
                        data.loc[:, u] = action_err(
                            data["value"], data[u], channel_scale
                        )
                    data.loc[:, "value"] = action_val(data["value"], channel_scale)
                    if len(data["value"]) <= 0:
                        logger.debug(
                            f"No values found for the hist in poi {poi_key}, continue with next one"
                        )
                        continue
                    logger.debug(
                        f"The values for the hist in poi {poi_key} are {data['value'].values}"
                    )

                    values = np.reshape(data["value"].values, shape)
                    variances = np.reshape(data["err_total"].values ** 2, shape)
                    values, variances = expand_flow(values, axes, flow_axes, variances)

                    # save nominal histogram with total uncertainty
                    h_ = hist.Hist(*axes, storage=hist.storage.Weight())
                    h_.view(flow=True)[...] = np.stack([values, variances], axis=-1)

                    hist_name = "hist_" + "_".join(axes_names)
                    if expected:
                        hist_name += "_expected"
                    logger.info(
                        f"Save histogram {hist_name} for proc {proc} in channel {channel} and type {poi_key}"
                    )
                    if hist_name in result[poi_key][channel][proc]:
                        logger.warning(
                            f"Histogram {hist_name} already in result, it will be overridden"
                        )
                    result[poi_key][channel][proc][hist_name] = h_

                    # save nominal histogram with stat uncertainty
                    if "err_stat" in data.keys():
                        variances = np.reshape(data["err_stat"].values, shape) ** 2
                        variances = expand_flow(variances, axes, flow_axes)
                        h_stat = hist.Hist(*axes, storage=hist.storage.Weight())
                        h_stat.view(flow=True)[...] = np.stack(
                            [values, variances], axis=-1
                        )
                        hist_name_stat = f"{hist_name}_stat"
                        if hist_name_stat in result[poi_key][channel][proc]:
                            logger.warning(
                                f"Histogram {hist_name_stat} already in result, it will be overridden"
                            )
                        result[poi_key][channel][proc][hist_name_stat] = h_stat

                    # save other systematic uncertainties as separately varied histograms
                    labels = [
                        u.replace("err_", "")
                        for u in filter(
                            lambda x: x.startswith("err_")
                            and x not in ["err_total", "err_stat"],
                            data.keys(),
                        )
                    ]
                    if labels:
                        errs = []
                        for u in labels:
                            err = np.reshape(data[f"err_{u}"].values, shape)
                            err = expand_flow(err, axes, flow_axes)
                            errs.append(values + err)
                        systs = np.stack([values, *errs], axis=-1)
                        systs = np.append(
                            systs, np.zeros_like(values)[..., None], axis=-1
                        )  # overflow axis can't be disabled in StrCategory
                        h_syst = hist.Hist(
                            *axes,
                            hist.axis.StrCategory(["nominal", *labels], name="syst"),
                            storage=hist.storage.Double(),
                        )
                        h_syst.values(flow=True)[...] = systs
                        hist_name_syst = f"{hist_name}_syst"
                        if hist_name_syst in result[poi_key][channel][proc]:
                            logger.warning(
                                f"Histogram {hist_name_syst} already in result, it will be overwritten"
                            )
                        result[poi_key][channel][proc][hist_name_syst] = h_syst

    return result, meta
