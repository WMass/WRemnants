from copy import deepcopy

import hist
import numpy as np

from wremnants.production import (
    generator_level_definitions,
    helicity_utils,
    systematics,
    theory_corrections,
)
from wremnants.utilities import binning, common
from wums import logging

logger = logging.child_logger(__name__)


def add_out_of_acceptance(datasets, group, newGroupName=None):
    # Copy datasets from specified group to make out of acceptance contribution
    datasets_ooa = []
    for dataset in datasets:
        if dataset.group == group:
            ds = deepcopy(dataset)

            if newGroupName is not None:
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

    df = generator_level_definitions.define_prefsr_vars(df)

    if "prefsr" in gen_levels:
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
        df = generator_level_definitions.define_postfsr_vars(df, mode=mode)

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
            selmap["pt_min"], selmap["pt_max"] = binning.get_default_ptbins(
                mode, gen="vgen" in mode
            )[1:]
            selmap["abseta_max"] = binning.get_default_etabins(mode)[-1]
            if mode[0] == "w" or "wlike" in mode:
                selmap["mtw_min"] = binning.get_default_mtcut(mode)
        elif fiducial == "masswindow" and mode[0] == "z":
            selmap["mass_min"], selmap["mass_max"] = binning.get_default_mz_window()
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
    helicity_smoothing_helpers,
    unfolding_axes,
    unfolding_cols,
    base_name="xnorm",
    add_helicity_axis=False,
    selection=None,
):
    """Fill a gen-level (xnorm) total and its theory systematics.

    ``selection`` (optional) is applied after the weight definitions, and the
    returned node is the weighted but unselected one, so that a second gen total
    with a different selection can reuse the same weight definitions.
    """
    # add histograms before any selection
    df_xnorm = df
    df_xnorm = df_xnorm.DefinePerSample("exp_weight", "1.0")

    df_xnorm = theory_corrections.define_theory_weights_and_corrs(
        df_xnorm,
        dataset_name,
        corr_helpers,
        args,
        helicity_smoothing_helpers=helicity_smoothing_helpers,
    )

    df_xnorm = df_xnorm.Define("xnorm", "0.5")

    df_weighted = df_xnorm
    if selection is not None:
        df_xnorm = df_xnorm.Filter(selection)

    axis_xnorm = hist.axis.Regular(
        1, 0.0, 1.0, name="count", underflow=False, overflow=False
    )

    xnorm_axes = [axis_xnorm, *unfolding_axes]
    xnorm_cols = ["xnorm", *unfolding_cols]

    if add_helicity_axis:
        df_xnorm = df_xnorm.Define(
            "helicity_moments_tensor",
            "wrem::csAngularMoments(csSineCosThetaPhigen)",
        )

        results.append(
            df_xnorm.HistoBoost(
                base_name,
                xnorm_axes,
                [*xnorm_cols, "helicity_moments_tensor", "nominal_weight"],
                tensor_axes=[binning.axis_helicity_multidim],
                storage=hist.storage.Weight(),
            )
        )
    else:
        results.append(
            df_xnorm.HistoBoost(base_name, xnorm_axes, [*xnorm_cols, "nominal_weight"])
        )

    systematics.add_theory_hists(
        results,
        df_xnorm,
        args,
        dataset_name,
        corr_helpers,
        helicity_smoothing_helpers,
        xnorm_axes,
        xnorm_cols,
        base_name=base_name,
        addhelicity=add_helicity_axis,
        nhelicity=9,
    )

    return df_weighted


def reweight_to_fitresult(filename, result=None, mapping=None, channel=None):
    import wums.boostHistHelpers as hh
    from rabbit.io_tools import get_fitresult

    fitresult, meta = get_fitresult(filename[0], result, meta=True)

    def get_result(fres):
        mappings = fres["mappings"]
        if mapping is None:
            if len(mappings.keys()) == 1:
                channels = next(iter(mappings.values()))["channels"]
            else:
                raise RuntimeError(
                    f"Expected exactly 1 mapping but got {[k for k in mappings.keys()]}"
                )
        else:
            channels = mappings[mapping]["channels"]

        if channel is None:
            if len(channels.keys()) == 1:
                res = next(iter(channels.values()))
            else:
                raise RuntimeError(
                    f"Expected exactly 1 channel but got {[k for k in channels.keys()]}"
                )
        else:
            res = channels[channel]
        return res

    results = get_result(fitresult)
    hPostfit = results[f"hist_postfit_inclusive"].get()

    if len(filename) == 2:
        fitresult_den, meta_den = get_fitresult(filename[1], result, meta=True)
        results_den = get_result(fitresult_den)
        hPrefit = results_den[f"hist_prefit_inclusive"].get()
    else:
        hPrefit = results[f"hist_prefit_inclusive"].get()

    hRatio = hh.divideHists(hPostfit, hPrefit)

    # get the gen level the unfolding was performed for
    level = meta["meta_info_input"]["meta_info"]["args"]["unfoldingLevel"]

    values = hRatio.values(flow=True)

    axes = []
    for i, ax in enumerate(hRatio.axes):
        name = ax.name
        if "VGen" in name:
            suffix = "V"
            var = name.replace("VGen", "")
        else:
            suffix = "Lep"
            var = name.replace("Gen", "")
        if var == "q":
            var = "charge"

        ax._raw_metadata["name"] = f"{level}{suffix}_{var}"

        # enable flow everywhere to allow generic indexing, add slices of 1 where flow was False
        if ax.traits.underflow == False:
            new_shape = list(values.shape)
            new_shape[i] = 1
            ones_slice = np.ones(new_shape, dtype=values.dtype)
            values = np.concatenate([ones_slice, values], axis=i)
        if ax.traits.overflow == False:
            new_shape = list(values.shape)
            new_shape[i] = 1
            ones_slice = np.ones(new_shape, dtype=values.dtype)
            values = np.concatenate([values, ones_slice], axis=i)

        ax = hh.enableAxisFlow(ax)

        axes.append(ax)

    hCorr = hist.Hist(
        *axes,
        hist.axis.Regular(1, 0, 1, name="vars", flow=False),
    )
    hCorr.values(flow=True)[...] = values[..., None]

    from wremnants.production.correctionsTensor_helper import (
        makeCorrectionsTensor,
    )

    corr_helper = makeCorrectionsTensor(hCorr)
    corr_helper.level = level

    return corr_helper


def rebin_pt(edges):
    # use 2 ptll bin for each ptVGen bin, except first and last
    # first gen bin same size as reco bin, then 1 gen bin for 2 reco bins
    new_edges = np.array([*edges[:2], *edges[3::2]])
    if len(new_edges) % 2:
        # in case it's an odd number of edges, last two bins are overflow
        edges = edges[:-1]
        new_edges = np.array([*edges[:2], *edges[3::2]])
    return new_edges


def _corr_axis_gen_selection(corr_axis, edges, gen_level):
    """Gen selection restricting events to the range of a theory-correction axis
    that is not a gen axis of the response, so that the response gen total counts
    the same phase space as the prediction on the correction's grid."""
    if corr_axis == "Q":
        if len(edges) != 2:
            raise ValueError(
                f"The response gen total needs a correction with a single Q bin, "
                f"got Q edges {list(edges)}"
            )
        return (
            f"{gen_level}V_mass > {float(edges[0]):.10g} && "
            f"{gen_level}V_mass < {float(edges[-1]):.10g}"
        )
    raise ValueError(
        f"No gen selection is known for the theory correction's {corr_axis!r} "
        "axis, which is not a gen axis of the response either"
    )


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
        fitresult_mapping=f"Select",
        fitresult_channel="ch0_masked",
        low_pu=False,
        response_gen_edges=None,
        response_corr_edges=None,
    ):
        self.analysis_label = "z_lowpu" if low_pu else "z_dilepton"
        self.cutsmap = cutsmap
        self.add_helicity_axis = "helicitySig" in unfolding_axes_names

        if not poi_as_noi and len(unfolding_levels) > 1:
            raise RuntimeError(
                "More than 1 unfolding levels at a time is only supported in poi as noi mode"
            )
        elif fitresult is not None and len(unfolding_levels) > 1:
            raise RuntimeError(
                "More than 1 unfolding levels at a time is not supported when reweighting from a fitresult"
            )

        self.poi_as_noi = poi_as_noi
        self.unfolding_levels = unfolding_levels

        self.weightsByHelicity_helper_unfolding = None

        self.unfolding_axes = {}
        self.unfolding_cols = {}
        self.unfolding_selections = {}
        for level in self.unfolding_levels:
            # for poi as noi, need gen rapidity overflow bin and out of acceptance axes to keep all events and be able to reconstruct corresponding reco histogram
            a, c, s = binning.get_unfolding_dilepton_axes(
                unfolding_axes_names,
                reco_axes_edges,
                level,
                flow_y=self.poi_as_noi,
                add_out_of_acceptance_axis=self.poi_as_noi,
                rebin_pt=rebin_pt,
            )
            self.unfolding_axes[level] = a
            self.unfolding_cols[level] = c
            self.unfolding_selections[level] = s

            if self.add_helicity_axis:
                if self.weightsByHelicity_helper_unfolding is None:
                    # need to rebin to the edges used for the unfolding, and remove of out of acceptance bins (|Y|>2.5 and pT>44)
                    pt_edges = [ax for ax in a if ax.name == "ptVGen"][0].edges
                    absY_edges = [ax for ax in a if ax.name == "absYVGen"][0].edges
                    # helper to derive helicity xsec shape from event by event reweighting
                    self.weightsByHelicity_helper_unfolding = helicity_utils.make_helicity_weight_helper(
                        is_z=True,
                        rebin_ptVgen_edges=pt_edges,
                        rebin_absYVgen_edges=absY_edges,
                        filename=f"{common.data_dir}/angularCoefficients/w_z_helicity_xsecs.hdf5",
                    )

                for ax in a:
                    if ax.name == "acceptance":
                        continue
                    # check if binning is consistent between correction helper and unfolding axes
                    #   unfolding axes must a subset of corretion helper
                    wbh_axis = self.weightsByHelicity_helper_unfolding.hist.axes[
                        ax.name.replace("Gen", "gen")
                    ]

                    if any(ax.edges != wbh_axis.edges):
                        raise RuntimeError(f"""
                            Unfolding axes must be consistent with axes from weightsByHelicity_helper.\n
                            Found unfolding axis {ax}\n
                            And weightsByHelicity_helper axis {wbh_axis}
                            """)

        # Optional second set of gen axes for a response matrix on the theory
        # correction's grid, next to (and leaving untouched) the unfolding axes. No
        # helicity axis: filled with nominal_weight, i.e. the helicity sum.
        self.response_gen_edges = response_gen_edges
        self.response_axes = {}
        self.response_cols = {}
        if response_gen_edges:
            names = [n for n in unfolding_axes_names if n != "helicitySig"]
            for level in self.unfolding_levels:
                a, c, _ = binning.get_unfolding_dilepton_axes(
                    names,
                    reco_axes_edges,
                    level,
                    flow_y=self.poi_as_noi,
                    add_out_of_acceptance_axis=self.poi_as_noi,
                    rebin_pt=rebin_pt,
                    edges_override=response_gen_edges,
                )
                self.response_axes[level] = a
                self.response_cols[level] = c
                logger.info(
                    f"Response-matrix gen axes ({level}): "
                    + ", ".join(
                        f"{ax.name} {ax.size} bins"
                        + (f" up to {ax.edges[-1]:g}" if hasattr(ax, "edges") else "")
                        for ax in a
                    )
                )

        # Selection of the response gen total: the correction axes that are not
        # gen axes of the response (today Q) become explicit cuts.
        self.response_gen_selections = {}
        if self.response_axes:
            if not self.poi_as_noi:
                raise RuntimeError(
                    "A response matrix on the theory correction's gen grid needs "
                    "poi_as_noi (otherwise the gen histograms are already "
                    "acceptance-filtered)"
                )
            if not response_corr_edges:
                raise ValueError(
                    "response_gen_edges needs response_corr_edges, the full grid "
                    "of the correction (get_corr_grid_edges)"
                )
            gen_axis_names = {
                theory_corrections.GEN_TO_CORR_AXIS.get(name, name)
                for name in response_gen_edges
            }
            needs_cut = sorted(set(response_corr_edges) - gen_axis_names)
            for level in self.unfolding_levels:
                self.response_gen_selections[level] = [
                    _corr_axis_gen_selection(c, response_corr_edges[c], level)
                    for c in needs_cut
                ]
            logger.info(
                "Response gen-total selection (correction axes "
                f"{needs_cut} are not gen axes of the response, the rest are "
                "handled by the binning): "
                + (
                    " && ".join(self.response_gen_selections[self.unfolding_levels[0]])
                    or "none needed"
                )
            )

        self.unfolding_corr_helper = (
            reweight_to_fitresult(
                fitresult,
                mapping=fitresult_mapping,
                channel=fitresult_channel,
            )
            if fitresult is not None
            else None
        )

    def add_gen_histograms(
        self, args, df, results, dataset, corr_helpers, helicity_smoothing_helpers={}
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

            if self.unfolding_corr_helper:
                logger.debug("Apply reweighting based on unfolded result")
                df = df.Define(
                    "unfoldingWeight_tensor",
                    self.unfolding_corr_helper,
                    [*self.unfolding_corr_helper.hist.axes.name[:-1], "unity"],
                )
                # fitresult reweighting only supported for a single unfolding level
                # (enforced in __init__); use index 0 explicitly rather than relying
                # on loop-variable persistence from the select_fiducial_space loop above
                df = df.Define(
                    "central_weight",
                    f"{self.unfolding_levels[0]}_acceptance ? unfoldingWeight_tensor(0) : unity",
                )

            for level in self.unfolding_levels:
                # add full phase space histograms for inclusive cross section,
                #   the mass cuts has to be preFSR to compare to SMP-20-004
                df_full = df.Filter("massVgen > 60")
                df_full = df_full.Filter("massVgen < 120")
                add_xnorm_histograms(
                    results,
                    df_full,
                    args,
                    dataset.name,
                    corr_helpers,
                    helicity_smoothing_helpers,
                    [a for a in self.unfolding_axes[level] if a.name != "acceptance"],
                    [
                        c
                        for c in self.unfolding_cols[level]
                        if c != f"{level}_acceptance"
                    ],
                    add_helicity_axis=self.add_helicity_axis,
                    base_name=f"{level}_full",
                )

                # Weights defined once; the fiducial gen total takes the
                # acceptance selection, the response gen total its own below.
                df_weighted = add_xnorm_histograms(
                    results,
                    df,
                    args,
                    dataset.name,
                    corr_helpers,
                    helicity_smoothing_helpers,
                    [a for a in self.unfolding_axes[level] if a.name != "acceptance"],
                    [
                        c
                        for c in self.unfolding_cols[level]
                        if c != f"{level}_acceptance"
                    ],
                    add_helicity_axis=self.add_helicity_axis,
                    base_name=level,
                    selection=(f"{level}_acceptance" if self.poi_as_noi else None),
                )

                if self.response_axes:
                    # Response gen total: same weights, no acceptance cut, the
                    # correction's phase space instead.
                    df_response = df_weighted
                    for selection in self.response_gen_selections[level]:
                        df_response = df_response.Filter(selection)
                    results.append(
                        df_response.HistoBoost(
                            f"{level}_response",
                            [
                                a
                                for a in self.response_axes[level]
                                if a.name != "acceptance"
                            ],
                            [
                                *[
                                    c
                                    for c in self.response_cols[level]
                                    if c != f"{level}_acceptance"
                                ],
                                "nominal_weight",
                            ],
                        )
                    )

        return df

    def add_poi_as_noi_histograms(
        self, df, results, nominal_axes, nominal_cols, channel="nominal"
    ):

        if self.add_helicity_axis:
            df = helicity_utils.define_helicity_weights(
                df, self.weightsByHelicity_helper_unfolding
            )

        for level in self.unfolding_levels:
            noiAsPoiHistName = f"{channel}_{level}_yieldsUnfolding"
            logger.debug(
                f"Creating special histogram '{noiAsPoiHistName}' for unfolding to treat POIs as NOIs"
            )
            yield_axes = [*nominal_axes, *self.unfolding_axes[level]]
            yield_cols = [*nominal_cols, *self.unfolding_cols[level]]
            if self.add_helicity_axis:
                results.append(
                    df.HistoBoost(
                        noiAsPoiHistName,
                        yield_axes,
                        [*yield_cols, "nominal_weight_helicity"],
                        tensor_axes=[binning.axis_helicity_multidim],
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

                # create corresponding histogram without experimental weights, to correlate stat between gen and reco
                weight_expr = theory_corrections.build_weight_expr(
                    df,
                    exclude_weights=[
                        "exp_weight",
                    ],
                )  # May want to exclude "ew_theory_corr_weight" in case of QCD only gen definition
                logger.info(f"Theory weight is {weight_expr}")
                df = df.Define(f"theory_weight_{level}", weight_expr)

                results.append(
                    df.HistoBoost(
                        f"{noiAsPoiHistName}_theory_weight",
                        yield_axes,
                        [*yield_cols, f"theory_weight_{level}"],
                    )
                )

            if self.response_axes:
                # reco x gen on the response grid, summed over helicities
                results.append(
                    df.HistoBoost(
                        f"nominal_{level}_yieldsResponse",
                        [*nominal_axes, *self.response_axes[level]],
                        [
                            *nominal_cols,
                            *self.response_cols[level],
                            "nominal_weight",
                        ],
                    )
                )
