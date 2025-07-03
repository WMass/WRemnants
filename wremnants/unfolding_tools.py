from copy import deepcopy

import hist

from utilities import common, differential
from wremnants import helicity_utils, syst_tools, theory_tools
from wremnants.datasets.datagroups import Datagroups
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


def define_gen_level(df, dataset_name, gen_levels=["prefsr", "postfsr"], mode="w_mass"):
    # gen level definitions
    known_levels = ["prefsr", "postfsr"]
    if any(g not in known_levels for g in gen_levels):
        raise ValueError(
            f"Unknown gen level in '{gen_levels}'! Supported gen level definitions are '{known_levels}'."
        )

    singlelep = mode[0] == "w" or "wlike" in mode

    if "prefsr" in gen_levels:
        df = theory_tools.define_prefsr_vars(df)

        # # needed for fiducial phase space definition
        df = df.Alias("prefsrV_mass", "massVgen")
        df = df.Alias("prefsrV_pt", "ptVgen")
        df = df.Alias("prefsrV_absY", "absYVgen")
        df = df.Alias("prefsrV_charge", "chargeVgen")

        if singlelep:
            df = df.Alias("prefsrV_mT", "mTVgen")

        if mode[0] == "w":
            df = df.Define("prefsrLep_pt", "chargeVgen < 0 ? genl.pt() : genlanti.pt()")
            df = df.Define(
                "prefsrLep_absEta",
                "chargeVgen < 0 ? std::fabs(genl.eta()) : std::fabs(genlanti.eta())",
            )
            df = df.Alias("prefsrLep_charge", "chargeVgen")
        else:
            df = df.Define("prefsrLep_pt", "event % 2 == 0 ? genl.pt() : genlanti.pt()")
            df = df.Define(
                "prefsrLep_absEta",
                "event % 2 == 0 ? std::fabs(genl.eta()) : std::fabs(genlanti.eta())",
            )
            df = df.Define(
                "prefsrOtherLep_pt", "event % 2 == 0 ? genlanti.pt() : genl.pt()"
            )
            df = df.Define(
                "prefsrOtherLep_absEta",
                "event % 2 == 0 ? std::fabs(genlanti.eta()) : std::fabs(genl.eta())",
            )
            if "wlike" in mode:
                df = df.Define("prefsrLep_charge", "event % 2 == 0 ? -1 : 1")

    if "postfsr" in gen_levels:
        df = theory_tools.define_postfsr_vars(df, mode=mode)

        if singlelep:
            df = df.Alias("postfsrV_mT", "postfsrMT")

        if mode[0] == "z":
            df = df.Alias("postfsrV_mass", "postfsrMV")
            df = df.Alias("postfsrV_absY", "postfsrabsYV")

        df = df.Alias("postfsrV_pt", "postfsrPTV")
        df = df.Alias("postfsrV_charge", "postfsrChargeV")

    return df


def select_fiducial_space(
    df, gen_level, select=True, accept=True, mode="w_mass", **kwargs
):
    # Define a fiducial phase space and if select=True, either select events inside/outside
    # accept = True: select events in fiducial phase space
    # accept = False: reject events in fiducial pahse space

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
    fiducial = kwargs.get("fiducial")
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
        selections.append(f"{gen_level}Lep_absEta < {selmap['abseta_max']}")
        if mode[0] == "z":
            selections.append(f"{gen_level}OtherLep_absEta < {selmap['abseta_max']}")

    if selmap["pt_min"] is not None:
        if "gen" in mode or "dilepton" in mode:
            selections.append(f"{gen_level}Lep_pt > {selmap['pt_min']}")
        if mode[0] == "z":
            selections.append(f"{gen_level}OtherLep_pt > {selmap['pt_min']}")

    if selmap["pt_max"] is not None:
        if "gen" in mode or "dilepton" in mode:
            # Don't place explicit cut on lepton pT for unfolding of W/W-like, but do for gen selection
            selections.append(f"{gen_level}Lep_pt < {selmap['pt_max']}")
        if mode[0] == "z":
            selections.append(f"{gen_level}OtherLep_pt < {selmap['pt_max']}")

    if selmap["mass_min"] is not None:
        selections.append(f"{gen_level}V_mass > {selmap['mass_min']}")

    if selmap["mass_max"] is not None:
        selections.append(f"{gen_level}V_mass < {selmap['mass_max']}")

    if selmap["mtw_min"] is not None:
        selections.append(f"{gen_level}V_mT > {selmap['mtw_min']}")

    selection = " && ".join(selections)

    if selection:
        df = df.Define(f"{gen_level}_acceptance", selection)
        logger.info(f"Applying fiducial selection '{selection}'")
    else:
        df = df.DefinePerSample(f"{gen_level}_acceptance", "true")

    if select and accept:
        logger.debug("Select events in fiducial phase space")
        df = df.Filter(f"{gen_level}_acceptance")
    elif select:
        logger.debug("Reject events in fiducial phase space")
        df = df.Filter(f"{gen_level}_acceptance == 0")

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
    base_name="xnorm",
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
        from wremnants.helicity_utils import axis_helicity_multidim

        df_xnorm = df_xnorm.Define(
            "helicity_moments_tensor",
            "wrem::csAngularMoments(csSineCosThetaPhigen)",
        )

        results.append(
            df_xnorm.HistoBoost(
                base_name,
                xnorm_axes,
                [*xnorm_cols, "helicity_moments_tensor", "nominal_weight"],
                tensor_axes=[axis_helicity_multidim],
                storage=hist.storage.Weight(),
            )
        )
    else:
        results.append(
            df_xnorm.HistoBoost(base_name, xnorm_axes, [*xnorm_cols, "nominal_weight"])
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
        base_name=base_name,
        addhelicity=add_helicity_axis,
        nhelicity=9,
    )

    return df_xnorm


def reweight_to_fitresult(filename, result=None, channel="ch0", flow=True):
    import wums.boostHistHelpers as hh
    from rabbit.io_tools import get_fitresult

    fitresult, meta = get_fitresult(filename, result, meta=True)

    hPrefit = fitresult["channels"][channel][f"hist_prefit_inclusive"].get()
    hPostfit = fitresult["channels"][channel][f"hist_postfit_inclusive"].get()

    hRatio = hh.divideHists(hPostfit, hPrefit)

    # get the gen level the unfolding was performed for
    level = meta["meta_info_input"]["meta_info"]["args"]["unfoldingLevel"]

    axes = []
    for ax in hRatio.axes:
        name = ax.name
        if "VGen" in ax.name:
            suffix = "V"
            var = ax.name.replace("VGen", "")
        else:
            suffix = "Lep"
            var = ax.name.replace("Gen", "")
        if var == "q":
            var = "charge"

        ax._ax.metadata["name"] = f"{level}{suffix}_{var}"
        axes.append(ax)

    hCorr = hist.Hist(*axes, hist.axis.Regular(1, 0, 1, name="vars", flow=False))
    hCorr.values(flow=flow)[...] = hRatio.values(flow=flow)[..., None]

    from wremnants.correctionsTensor_helper import makeCorrectionsTensor

    corr_helper = makeCorrectionsTensor(hCorr)
    corr_helper.level = level

    return corr_helper


class UnfolderZ:
    """
    To be used in histmakers to define columns and add histograms for unfolding of Z dilepton kinematics
    """

    def __init__(
        self,
        cutsmap,
        reco_axes_edges,
        unfolding_axes_names=None,
        unfolding_levels=None,
        poi_as_noi=True,
        fitresult=None,
        low_pu=False,
    ):
        self.analysis_label = "z_lowpu" if low_pu else "z_dilepton"
        self.cutsmap = cutsmap
        self.add_helicity_axis = "helicitySig" in unfolding_axes_names

        if not poi_as_noi and len(unfolding_levels) > 1:
            raise RuntimeError(
                "More than 1 unfolding levels at a time is only supported in poi as noi mode"
            )
        self.poi_as_noi = poi_as_noi
        self.unfolding_levels = unfolding_levels

        # self.qcdScaleByHelicity_helper =

        if self.add_helicity_axis:
            # helper to derive helicity xsec shape from event by event reweighting
            self.weightsByHelicity_helper_unfolding = helicity_utils.make_helicity_weight_helper(
                is_z=True,
                filename=f"{common.data_dir}/angularCoefficients/w_z_helicity_xsecs_scetlib_dyturboCorr_maxFiles_m1_unfoldingBinning.hdf5",
                rebi_ptVgen=True,
            )

        self.unfolding_axes = {}
        self.unfolding_cols = {}
        self.unfolding_selections = {}
        for level in self.unfolding_levels:
            a, c, s = differential.get_dilepton_axes(
                unfolding_axes_names,
                reco_axes_edges,
                level,
                add_out_of_acceptance_axis=self.poi_as_noi,
            )
            self.unfolding_axes[level] = a
            self.unfolding_cols[level] = c
            self.unfolding_selections[level] = s

            if self.add_helicity_axis:
                for ax in a:
                    if ax.name == "acceptance":
                        continue
                    # check if binning is consistent between correction helper and unfolding axes
                    wbh_axis = self.weightsByHelicity_helper_unfolding.hist.axes[
                        ax.name.replace("Gen", "gen")
                    ]
                    if any(ax.edges != wbh_axis.edges):
                        raise RuntimeError(
                            f"""
                            Unfolding axes must be consistent with axes from weightsByHelicity_helper.\n
                            Found unfolding axis {ax}\n
                            And weightsByHelicity_helper axis {wbh_axis}
                            """
                        )

        self.unfolding_corr_helper = (
            reweight_to_fitresult(fitresult) if fitresult else None
        )

    def add_gen_histograms(
        self, args, df, results, dataset, corr_helpers, qcdScaleByHelicity_helper
    ):
        df = define_gen_level(
            df, dataset.name, self.unfolding_levels, mode=self.analysis_label
        )

        if hasattr(dataset, "out_of_acceptance"):
            # only for exact unfolding
            df = select_fiducial_space(
                df,
                self.unfolding_levels[0],
                mode=self.analysis_label,
                selections=self.unfolding_selections[self.unfolding_levels[0]],
                accept=False,
                **self.cutsmap,
            )
        else:
            if self.unfolding_corr_helper:
                logger.debug("Apply reweighting based on unfolded result")
                df = df.Define(
                    "unfoldingWeight_tensor",
                    self.unfolding_corr_helper,
                    [*self.unfolding_corr_helper.hist.axes.name[:-1], "unity"],
                )
                df = df.Define(
                    "central_weight", "acceptance ? unfoldingWeight_tensor(0) : unity"
                )
            for level in self.unfolding_levels:
                df = select_fiducial_space(
                    df,
                    level,
                    mode=self.analysis_label,
                    selections=self.unfolding_selections[level],
                    select=not self.poi_as_noi,
                    accept=True,
                    **self.cutsmap,
                )

                if self.poi_as_noi:
                    df_xnorm = df.Filter(f"{level}_acceptance")
                else:
                    df_xnorm = df

                add_xnorm_histograms(
                    results,
                    df_xnorm,
                    args,
                    dataset.name,
                    corr_helpers,
                    qcdScaleByHelicity_helper,
                    [a for a in self.unfolding_axes[level] if a.name != "acceptance"],
                    [
                        c
                        for c in self.unfolding_cols[level]
                        if c != f"{level}_acceptance"
                    ],
                    add_helicity_axis=self.add_helicity_axis,
                    base_name=level,
                )
        return df

    def add_poi_as_noi_histograms(self, df, results, nominal_axes, nominal_cols):

        if self.add_helicity_axis:
            df = helicity_utils.define_helicity_weights(
                df, self.weightsByHelicity_helper_unfolding
            )

        for level in self.unfolding_levels:
            noiAsPoiHistName = Datagroups.histName(
                "nominal", syst=f"{level}_yieldsUnfolding"
            )
            logger.debug(
                f"Creating special histogram '{noiAsPoiHistName}' for unfolding to treat POIs as NOIs"
            )
            yield_axes = [*nominal_axes, *self.unfolding_axes[level]]
            yield_cols = [*nominal_cols, *self.unfolding_cols[level]]
            if self.add_helicity_axis:
                from wremnants.helicity_utils import axis_helicity_multidim

                results.append(
                    df.HistoBoost(
                        noiAsPoiHistName,
                        yield_axes,
                        [*yield_cols, "nominal_weight_helicity"],
                        tensor_axes=[axis_helicity_multidim],
                    )
                )
            else:
                results.append(
                    df.HistoBoost(
                        noiAsPoiHistName,
                        yield_axes,
                        [*yield_cols, "nominal_weight"],
                    )
                )
