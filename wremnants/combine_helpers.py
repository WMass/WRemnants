import hist
import numpy as np

from utilities.io_tools import input_tools
from wremnants import histselections, syst_tools
from wums import boostHistHelpers as hh
from wums import logging

logger = logging.child_logger(__name__)


def add_mass_diff_variations(
    datagroups,
    mass_diff_var,
    name,
    processes,
    constrain=False,
    suffix="",
    label="W",
    passSystToFakes=True,
):
    mass_diff_args = dict(
        histname=name,
        name=f"massDiff{suffix}{label}",
        processes=processes,
        group=f"massDiff{label}",
        systNameReplace=[("Shift", f"Diff{suffix}")],
        skipEntries=syst_tools.massWeightNames(proc=label, exclude=50),
        noi=not constrain,
        noConstraint=not constrain,
        mirror=False,
        systAxes=["massShift"],
        passToFakes=passSystToFakes,
    )
    # mass difference by swapping the +50MeV with the -50MeV variations for half of the bins
    args = ["massShift", f"massShift{label}50MeVUp", f"massShift{label}50MeVDown"]
    if mass_diff_var == "charge":
        datagroups.addSystematic(
            **mass_diff_args,
            # # on gen level based on the sample, only possible for mW
            # preOpMap={m.name: (lambda h, swap=swap_bins: swap(h, "massShift", f"massShift{label}50MeVUp", f"massShift{label}50MeVDown"))
            #     for p in processes for g in datagroups.procGroups[p] for m in datagroups.groups[g].members if "minus" in m.name},
            # on reco level based on reco charge
            preOp=lambda h: hh.swap_histogram_bins(h, *args, "charge", 0),
        )

    elif mass_diff_var == "cosThetaStarll":
        datagroups.addSystematic(
            **mass_diff_args,
            preOp=lambda h: hh.swap_histogram_bins(
                h, *args, "cosThetaStarll", hist.tag.Slicer()[0 : complex(0, 0) :]
            ),
        )
    elif mass_diff_var == "eta-sign":
        datagroups.addSystematic(
            **mass_diff_args,
            preOp=lambda h: hh.swap_histogram_bins(
                h, *args, "eta", hist.tag.Slicer()[0 : complex(0, 0) :]
            ),
        )
    elif mass_diff_var == "eta-range":
        datagroups.addSystematic(
            **mass_diff_args,
            preOp=lambda h: hh.swap_histogram_bins(
                h, *args, "eta", hist.tag.Slicer()[complex(0, -0.9) : complex(0, 0.9) :]
            ),
        )
    elif mass_diff_var.startswith("etaRegion"):
        # 3 bins, use 3 unconstrained parameters: mass; mass0 - mass2; mass0 + mass2 - mass1
        mass_diff_args["name"] = f"massDiff1{suffix}{label}"
        mass_diff_args["systNameReplace"] = [("Shift", f"Diff1{suffix}")]
        datagroups.addSystematic(
            **mass_diff_args,
            preOp=lambda h: hh.swap_histogram_bins(
                hh.swap_histogram_bins(h, *args, mass_diff_var, 2),  # invert for mass2
                *args,
                mass_diff_var,
                1,
                axis1_replace=f"massShift{label}0MeV",
            ),  # set mass1 to nominal
        )
        mass_diff_args["name"] = f"massDiff2{suffix}{label}"
        mass_diff_args["systNameReplace"] = [("Shift", f"Diff2{suffix}")]
        datagroups.addSystematic(
            **mass_diff_args,
            preOp=lambda h: hh.swap_histogram_bins(h, *args, mass_diff_var, 1),
        )


def add_recoil_uncertainty(
    card_tool,
    samples,
    passSystToFakes=False,
    pu_type="highPU",
    flavor="",
    group_compact=True,
):
    met = input_tools.args_from_metadata(card_tool, "met")
    if flavor == "":
        flavor = input_tools.args_from_metadata(card_tool, "flavor")
    if pu_type == "highPU" and (
        met in ["RawPFMET", "DeepMETReso", "DeepMETPVRobust", "DeepMETPVRobustNoPUPPI"]
    ):
        card_tool.addSystematic(
            "recoil_stat",
            processes=samples,
            mirror=True,
            groups=[
                "recoil" if group_compact else "recoil_stat",
                "experiment",
                "expNoCalib",
            ],
            systAxes=["recoil_unc"],
            passToFakes=passSystToFakes,
        )

    if pu_type == "lowPU":
        group_compact = False
        card_tool.addSystematic(
            "recoil_syst",
            processes=samples,
            mirror=True,
            groups=[
                "recoil" if group_compact else "recoil_syst",
                "experiment",
                "expNoCalib",
            ],
            systAxes=["recoil_unc"],
            passToFakes=passSystToFakes,
        )

        card_tool.addSystematic(
            "recoil_stat",
            processes=samples,
            mirror=True,
            groups=[
                "recoil" if group_compact else "recoil_stat",
                "experiment",
                "expNoCalib",
            ],
            systAxes=["recoil_unc"],
            passToFakes=passSystToFakes,
        )


def add_explicit_BinByBinStat(
    datagroups, recovar, samples="signal_samples", wmass=False, source=None, label="Z"
):
    """
    add explicit bin by bin stat uncertainties
    Parameters:
    source (tuple of str): take variations from histogram with name given by f"{source[0]}_{source[1]}" (E.g. to correlate between detector level and gen level fits).
        If None, use variations from nominal histogram
    """

    recovar_syst = [f"_{n}" for n in recovar]
    info = dict(
        baseName="binByBinStat_" + "_".join(datagroups.procGroups[samples]) + "_",
        name=f"binByBinStat{label}",
        group=f"binByBinStat{label}",
        passToFakes=False,
        processes=[samples],
        mirror=True,
        labelsByAxis=[f"_{p}" if p != recovar[0] else p for p in recovar],
    )

    if source is not None:
        # signal region selection
        if wmass:
            action_sel = lambda h, x: histselections.SignalSelectorABCD(h[x]).get_hist(
                h[x]
            )
        else:
            action_sel = lambda h, x: h[x]

        integration_var = {
            a: hist.sum for a in datagroups.gen_axes_names
        }  # integrate out gen axes for bin by bin uncertainties
        integration_var["acceptance"] = hist.sum

        datagroups.addSystematic(
            **info,
            nominalName=source[0],
            histname=source[1],
            systAxes=recovar,
            actionRequiresNomi=True,
            action=lambda hv, hn: hh.addHists(
                hn.project(*datagroups.gen_axes_names),
                action_sel(hv, {"acceptance": True}).project(
                    *recovar, *datagroups.gen_axes_names
                ),
                scale2=(
                    np.sqrt(action_sel(hv, integration_var).variances(flow=True))
                    / action_sel(hv, integration_var).values(flow=True)
                )[..., *[np.newaxis] * len(datagroups.gen_axes_names)],
            ),
        )
    else:
        # if args.fitresult: # FIXME
        #     info["group"] = "binByBinStat"
        datagroups.addSystematic(
            **info,
            systAxes=recovar_syst,
            action=lambda h: hh.addHists(
                h.project(*recovar),
                hh.expand_hist_by_duplicate_axes(
                    h.project(*recovar), recovar, recovar_syst
                ),
                scale2=np.sqrt(h.project(*recovar).variances(flow=True))
                / h.project(*recovar).values(flow=True),
            ),
        )


def add_electroweak_uncertainty(
    card_tool,
    ewUncs,
    flavor="mu",
    samples="single_v_samples",
    passSystToFakes=True,
    wlike=False,
):
    # different uncertainty for W and Z samples
    all_samples = card_tool.procGroups[samples]
    z_samples = [p for p in all_samples if p[0] == "Z"]
    w_samples = [p for p in all_samples if p[0] == "W"]

    for ewUnc in ewUncs:
        if "renesanceEW" in ewUnc:
            if w_samples:
                # add renesance (virtual EW) uncertainty on W samples
                card_tool.addSystematic(
                    f"{ewUnc}Corr",
                    processes=w_samples,
                    preOp=lambda h: h[{"var": ["nlo_ew_virtual"]}],
                    labelsByAxis=[f"renesanceEWCorr"],
                    scale=1.0,
                    systAxes=["var"],
                    groups=[f"theory_ew_virtW_corr", "theory_ew", "theory"],
                    passToFakes=passSystToFakes,
                    mirror=True,
                )
        elif ewUnc == "powhegFOEW":
            if z_samples:
                card_tool.addSystematic(
                    f"{ewUnc}Corr",
                    preOp=lambda h: h[{"weak": ["weak_ps", "weak_aem"]}],
                    processes=z_samples,
                    labelsByAxis=[f"{ewUnc}Corr"],
                    scale=1.0,
                    systAxes=["weak"],
                    mirror=True,
                    groups=[f"theory_ew_virtZ_scheme", "theory_ew", "theory"],
                    passToFakes=passSystToFakes,
                    name="ewScheme",
                )
                card_tool.addSystematic(
                    f"{ewUnc}Corr",
                    preOp=lambda h: h[{"weak": ["weak_default"]}],
                    processes=z_samples,
                    labelsByAxis=[f"{ewUnc}Corr"],
                    scale=1.0,
                    systAxes=["weak"],
                    mirror=True,
                    groups=[f"theory_ew_virtZ_corr", "theory_ew", "theory"],
                    passToFakes=passSystToFakes,
                    name="ew",
                )
        else:
            if "FSR" in ewUnc:
                if flavor == "e":
                    logger.warning(
                        "ISR/FSR EW uncertainties are not implemented for electrons, proceed w/o"
                    )
                    continue
                scale = 1
            if "ISR" in ewUnc:
                scale = 2
            else:
                scale = 1

            if "winhac" in ewUnc:
                if not w_samples:
                    logger.warning(
                        "Winhac is not implemented for any other process than W, proceed w/o winhac EW uncertainty"
                    )
                    continue
                elif all_samples != w_samples:
                    logger.warning(
                        "Winhac is only implemented for W samples, proceed w/o winhac EW uncertainty for other samples"
                    )
                samples = w_samples
            else:
                samples = all_samples

            s = hist.tag.Slicer()
            if ewUnc.startswith("virtual_ew"):
                preOp = lambda h: h[{"systIdx": s[0:1]}]
            else:
                preOp = lambda h: h[{"systIdx": s[1:2]}]

            card_tool.addSystematic(
                f"{ewUnc}Corr",
                systAxes=["systIdx"],
                mirror=True,
                passToFakes=passSystToFakes,
                processes=samples,
                labelsByAxis=[f"{ewUnc}Corr"],
                scale=scale,
                preOp=preOp,
                groups=[f"theory_ew_{ewUnc}", "theory_ew", "theory"],
            )


def add_noi_unfolding_variations(
    datagroups,
    label,
    passSystToFakes,
    xnorm,
    poi_axes,
    prior_norm=1,
    scale_norm=0.01,
    poi_axes_flow=["ptGen", "ptVGen"],
    gen_level="postfsr",
):
    poi_axes_syst = [f"_{n}" for n in poi_axes] if xnorm else poi_axes[:]
    noi_args = dict(
        histname=gen_level if xnorm else f"nominal_{gen_level}_yieldsUnfolding",
        name=f"nominal_{gen_level}_yieldsUnfolding",
        baseName=f"{label}_",
        group=f"normXsec{label}",
        passToFakes=passSystToFakes,
        systAxes=poi_axes_syst,
        processes=["signal_samples"],
        noConstraint=True,
        noi=True,
        mirror=True,
        scale=(
            1 if prior_norm < 0 else prior_norm
        ),  # histogram represents an (args.priorNormXsec*100)% prior
        labelsByAxis=[f"_{p}" if p != poi_axes[0] else p for p in poi_axes],
    )

    def disable_flow(h, axes_names=["absYVGen", "absEtaGen"]):
        # disable flow for syst axes to not add systematic uncertainties in these bins
        for var in axes_names:
            if var in h.axes.name:
                h = hh.disableFlow(h, var)
        return h

    def get_scalemap(axes, select={}, rename_axes={}):
        # make sure each gen bin variation has a similar effect in the reco space so that
        #  we have similar sensitivity to all parameters within the given up/down variations
        # FIXME: this currently doesn't work, not sure why ...
        signal_samples = datagroups.procGroups["signal_samples"]
        hScale = datagroups.getHistsForProcAndSyst(
            signal_samples[0],
            f"{gen_level}_yieldsUnfolding",
            nominal_name="nominal",
        )
        hScale = hScale[{"acceptance": True, **select}]
        hScale.values(flow=True)[...] = abs(hScale.values(flow=True))
        hScale = hScale.project(*axes)
        hScale = disable_flow(hScale)
        for o, n in rename_axes.items():
            hScale.axes[o]._ax.metadata["name"] = n
        # scalemap with preserving normalization
        hScale.values(flow=True)[...] = (
            1.0
            / hScale.values(flow=True)
            * hScale.sum(flow=True).value
            / np.prod(hScale.values(flow=True).shape)
        )
        return hScale

    if xnorm:

        def make_poi_xnorm_variations(h, poi_axes, poi_axes_syst, norm, h_scale=None):
            hVar = hh.expand_hist_by_duplicate_axes(
                h, poi_axes[::-1], poi_axes_syst[::-1]
            )
            hVar = disable_flow(hVar, axes_names=["_absYVGen", "_absEtaGen"])
            if h_scale is not None:
                hVar = hh.multiplyHists(hVar, h_scale)
            return hh.addHists(h, hVar, scale2=norm)

        datagroups.addSystematic(
            **noi_args,
            action=make_poi_xnorm_variations,
            actionArgs=dict(
                poi_axes=poi_axes,
                poi_axes_syst=poi_axes_syst,
                norm=scale_norm,
                h_scale=get_scalemap(
                    poi_axes,
                    rename_axes={o: n for o, n in zip(poi_axes, poi_axes_syst)},
                ),
            ),
        )
    else:

        def make_poi_variations(h, poi_axes, norm, h_scale=None):
            hNom = h[
                {
                    **{ax: hist.tag.Slicer()[:: hist.sum] for ax in poi_axes},
                    "acceptance": hist.tag.Slicer()[:: hist.sum],
                }
            ]
            hVar = h[{"acceptance": True}]
            hVar = disable_flow(hVar)
            if h_scale is not None:
                hVar = hh.multiplyHists(hVar, h_scale)
            return hh.addHists(hNom, hVar, scale2=norm)

        datagroups.addSystematic(
            **noi_args,
            systAxesFlow=[n for n in poi_axes if n in poi_axes_flow],
            preOpMap={
                m.name: make_poi_variations
                for g in datagroups.procGroups["signal_samples"]
                for m in datagroups.groups[g].members
            },
            preOpArgs=dict(
                poi_axes=poi_axes,
                norm=scale_norm,
                h_scale=get_scalemap(poi_axes),
            ),
        )
