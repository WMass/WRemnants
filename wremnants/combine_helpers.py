import hist
import numpy as np

from utilities import boostHistHelpers as hh
from utilities import logging
from utilities.io_tools import input_tools
from wremnants import histselections, syst_tools

logger = logging.child_logger(__name__)


def add_mass_diff_variations(
    cardTool,
    mass_diff_var,
    name,
    processes,
    constrain=False,
    suffix="",
    label="W",
    passSystToFakes=True,
):
    mass_diff_args = dict(
        name=name,
        processes=processes,
        rename=f"massDiff{suffix}{label}",
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
        cardTool.addSystematic(
            **mass_diff_args,
            # # on gen level based on the sample, only possible for mW
            # preOpMap={m.name: (lambda h, swap=swap_bins: swap(h, "massShift", f"massShift{label}50MeVUp", f"massShift{label}50MeVDown"))
            #     for p in processes for g in cardTool.procGroups[p] for m in cardTool.datagroups.groups[g].members if "minus" in m.name},
            # on reco level based on reco charge
            preOp=lambda h: hh.swap_histogram_bins(h, *args, "charge", 0),
        )

    elif mass_diff_var == "cosThetaStarll":
        cardTool.addSystematic(
            **mass_diff_args,
            preOp=lambda h: hh.swap_histogram_bins(
                h, *args, "cosThetaStarll", hist.tag.Slicer()[0 : complex(0, 0) :]
            ),
        )
    elif mass_diff_var == "eta-sign":
        cardTool.addSystematic(
            **mass_diff_args,
            preOp=lambda h: hh.swap_histogram_bins(
                h, *args, "eta", hist.tag.Slicer()[0 : complex(0, 0) :]
            ),
        )
    elif mass_diff_var == "eta-range":
        cardTool.addSystematic(
            **mass_diff_args,
            preOp=lambda h: hh.swap_histogram_bins(
                h, *args, "eta", hist.tag.Slicer()[complex(0, -0.9) : complex(0, 0.9) :]
            ),
        )
    elif mass_diff_var.startswith("etaRegion"):
        # 3 bins, use 3 unconstrained parameters: mass; mass0 - mass2; mass0 + mass2 - mass1
        mass_diff_args["rename"] = f"massDiff1{suffix}{label}"
        mass_diff_args["systNameReplace"] = [("Shift", f"Diff1{suffix}")]
        cardTool.addSystematic(
            **mass_diff_args,
            preOp=lambda h: hh.swap_histogram_bins(
                hh.swap_histogram_bins(h, *args, mass_diff_var, 2),  # invert for mass2
                *args,
                mass_diff_var,
                1,
                axis1_replace=f"massShift{label}0MeV",
            ),  # set mass1 to nominal
        )
        mass_diff_args["rename"] = f"massDiff2{suffix}{label}"
        mass_diff_args["systNameReplace"] = [("Shift", f"Diff2{suffix}")]
        cardTool.addSystematic(
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
            group="recoil" if group_compact else "recoil_stat",
            splitGroup={"experiment": f".*", "expNoCalib": ".*"},
            systAxes=["recoil_unc"],
            passToFakes=passSystToFakes,
        )

    if pu_type == "lowPU":
        group_compact = False
        card_tool.addSystematic(
            "recoil_syst",
            processes=samples,
            mirror=True,
            group="recoil" if group_compact else "recoil_syst",
            splitGroup={"experiment": f".*", "expNoCalib": ".*"},
            systAxes=["recoil_unc"],
            passToFakes=passSystToFakes,
        )

        card_tool.addSystematic(
            "recoil_stat",
            processes=samples,
            mirror=True,
            group="recoil" if group_compact else "recoil_stat",
            splitGroup={"experiment": f".*", "expNoCalib": ".*"},
            systAxes=["recoil_unc"],
            passToFakes=passSystToFakes,
        )


def add_explicit_BinByBinStat(
    cardTool, recovar, samples="signal_samples", wmass=False, source=None, label="Z"
):
    """
    add explicit bin by bin stat uncertainties
    Parameters:
    source (tuple of str): take variations from histogram with name given by f"{source[0]}_{source[1]}" (E.g. used to correlate between masked channels).
        If None, use variations from nominal histogram
    """
    datagroups = cardTool.datagroups

    recovar_syst = [f"_{n}" for n in recovar]
    info = dict(
        baseName="binByBinStat_" + "_".join(cardTool.procGroups[samples]) + "_",
        rename=f"binByBinStat{label}",
        group=f"binByBinStat{label}",
        passToFakes=False,
        processes=[samples],
        mirror=True,
        labelsByAxis=[f"_{p}" if p != recovar[0] else p for p in recovar],
    )
    cardTool.setProcsNoStatUnc(cardTool.procGroups[samples])
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
        cardTool.addSystematic(
            **info,
            nominalName=source[0],
            name=source[1],
            systAxes=recovar,
            actionRequiresNomi=True,
            action=lambda hv, hn: hh.addHists(
                hn[{"count": hist.sum, "acceptance": hist.sum}].project(
                    *datagroups.gen_axes_names
                ),
                action_sel(hv, {"acceptance": True}).project(
                    *recovar, *datagroups.gen_axes_names
                ),
                scale2=(
                    np.sqrt(
                        action_sel(
                            hv, {"acceptance": hist.sum, **integration_var}
                        ).variances(flow=True)
                    )
                    / action_sel(
                        hv, {"acceptance": hist.sum, **integration_var}
                    ).values(flow=True)
                )[..., *[np.newaxis] * len(datagroups.gen_axes_names)],
            ),
        )
    else:
        # if args.fitresult: # FIXME
        #     info["group"] = "binByBinStat"
        cardTool.addSystematic(
            **info,
            name=cardTool.nominalName,
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
    info = dict(
        systAxes=["systIdx"],
        mirror=True,
        splitGroup={"theory_ew": f".*", "theory": f".*"},
        passToFakes=passSystToFakes,
    )
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
                    group="theory_ew_virtW_corr",
                    splitGroup={"theory_ew": f".*", "theory": f".*"},
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
                    group="theory_ew_virtZ_scheme",
                    splitGroup={"theory_ew": f".*", "theory": f".*"},
                    passToFakes=passSystToFakes,
                    rename="ewScheme",
                )
                card_tool.addSystematic(
                    f"{ewUnc}Corr",
                    preOp=lambda h: h[{"weak": ["weak_default"]}],
                    processes=z_samples,
                    labelsByAxis=[f"{ewUnc}Corr"],
                    scale=1.0,
                    systAxes=["weak"],
                    mirror=True,
                    group="theory_ew_virtZ_corr",
                    splitGroup={"theory_ew": f".*", "theory": f".*"},
                    passToFakes=passSystToFakes,
                    rename="ew",
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
                **info,
                processes=samples,
                labelsByAxis=[f"{ewUnc}Corr"],
                scale=scale,
                preOp=preOp,
                group=f"theory_ew_{ewUnc}",
            )


def add_noi_unfolding_variations(
    cardTool,
    label,
    passSystToFakes,
    xnorm,
    poi_axes,
    wmass=False,
    prior_norm=1,
    scale_norm=0.01,
    poi_axes_flow=[],  # ["ptGen", "ptVGen"],
):
    poi_axes_syst = [f"_{n}" for n in poi_axes] if xnorm else poi_axes[:]
    noi_args = dict(
        group=f"normXsec{label}",
        passToFakes=passSystToFakes,
        name=f"xnorm" if xnorm else f"yieldsUnfolding",
        rename="yieldsUnfolding",
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
        # disable flow for gen axes as these events are in out of acceptance
        for var in axes_names:
            if var in h.axes.name:
                h = hh.disableFlow(h, var)
        return h

    def get_scalemap(axes, scale=None, select={}):
        # make sure each gen bin variation has a similar effect in the reco space so that
        #  we have similar sensitivity to all parameters within the given up/down variations
        signal_samples = cardTool.procGroups["signal_samples"]
        hScale = cardTool.getHistsForProcAndSyst(
            signal_samples[0], "yieldsUnfolding", nominal_name="nominal"
        )
        hScale = hScale[{"acceptance": True, **select}]
        hScale.values(flow=True)[...] = abs(hScale.values(flow=True))
        hScale = hScale.project(*axes)
        hScale = disable_flow(hScale)
        scalemap = hScale.sum(flow=True).value / hScale.values(flow=True)
        this_scale = scale * scalemap if scale is not None else scalemap
        return this_scale

    if xnorm:

        def make_poi_xnorm_variations(h, poi_axes, poi_axes_syst, scale):
            h = disable_flow(h)
            hVar = hh.expand_hist_by_duplicate_axes(
                h, poi_axes[::-1], poi_axes_syst[::-1]
            )
            slices = [np.newaxis if a in h.axes else slice(None) for a in hVar.axes]
            hVar.values(flow=True)[...] = hVar.values(flow=True) * scale[*slices]
            return hh.addHists(h, hVar)

        scalemap = get_scalemap(poi_axes, scale_norm)

        cardTool.addSystematic(
            **noi_args,
            baseName=f"{label}_",
            action=make_poi_xnorm_variations,
            actionArgs=dict(
                poi_axes=poi_axes, poi_axes_syst=poi_axes_syst, scale=scalemap
            ),
        )
    else:

        def make_poi_variations(h, poi_axes, scale):
            hNom = h[
                {
                    **{ax: hist.tag.Slicer()[:: hist.sum] for ax in poi_axes},
                    "acceptance": hist.tag.Slicer()[:: hist.sum],
                }
            ]
            hVar = h[{"acceptance": True}]
            hVar = disable_flow(hVar)
            slices = [np.newaxis if a in hNom.axes else slice(None) for a in hVar.axes]
            hVar.values(flow=True)[...] = hVar.values(flow=True) * scale[*slices]
            return hh.addHists(hNom, hVar)

        if wmass:
            # add two sets of systematics, one for each charge
            poi_axes = [p for p in poi_axes if p != "qGen"]
            poi_axes_syst = [f"_{n}" for n in poi_axes] if xnorm else poi_axes[:]
            noi_args["labelsByAxis"] = [
                f"_{p}" if p != poi_axes[0] else p for p in poi_axes
            ]
            noi_args["systAxes"] = poi_axes_syst
            for sign, sign_idx in (("minus", 0), ("plus", 1)):
                scalemap = get_scalemap(
                    poi_axes, scale_norm, select={"charge": sign_idx}
                )
                noi_args["rename"] = f"noiW{sign}"
                cardTool.addSystematic(
                    **noi_args,
                    baseName=f"W_qGen{sign_idx}_",
                    systAxesFlow=[n for n in poi_axes if n in poi_axes_flow],
                    preOpMap={
                        m.name: (
                            make_poi_variations
                            if sign in m.name
                            else (
                                lambda h, poi_axes, scale: h[
                                    {
                                        **{
                                            ax: hist.tag.Slicer()[:: hist.sum]
                                            for ax in poi_axes
                                        },
                                        "acceptance": hist.tag.Slicer()[:: hist.sum],
                                    }
                                ]
                            )
                        )
                        for g in cardTool.procGroups["signal_samples"]
                        for m in cardTool.datagroups.groups[g].members
                    },
                    preOpArgs=dict(poi_axes=poi_axes, scale=scalemap),
                )
        else:
            scalemap = get_scalemap(poi_axes, scale_norm)
            cardTool.addSystematic(
                **noi_args,
                baseName=f"{label}_",
                systAxesFlow=[n for n in poi_axes if n in poi_axes_flow],
                preOpMap={
                    m.name: make_poi_variations
                    for g in cardTool.procGroups["signal_samples"]
                    for m in cardTool.datagroups.groups[g].members
                },
                preOpArgs=dict(poi_axes=poi_axes, scale=scalemap),
            )
