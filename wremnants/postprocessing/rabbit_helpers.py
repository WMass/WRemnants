import hist
import numpy as np

from wremnants.postprocessing import histselections
from wremnants.postprocessing.datagroups.datagroup import Datagroup_member
from wremnants.utilities import theory_utils
from wums import boostHistHelpers as hh
from wums import logging

logger = logging.child_logger(__name__)


def decorrelateByAxis(
    hvar,
    hnom,
    axisToDecorrName,
    newDecorrAxisName=None,
    axlim=[],
    rebin=[],
    absval=False,
):
    return decorrelateByAxes(
        hvar,
        hnom,
        axesToDecorrNames=[axisToDecorrName],
        newDecorrAxesNames=[newDecorrAxisName],
        axlim=[axlim],
        rebin=[rebin],
        absval=[absval],
    )


def decorrelateByAxes(
    hvar, hnom, axesToDecorrNames, newDecorrAxesNames=[], axlim=[], rebin=[], absval=[]
):

    commonMessage = f"Requested to decorrelate uncertainty in histogram {hvar.name} by {axesToDecorrNames} axes"
    if any(a not in hvar.axes.name for a in axesToDecorrNames):
        raise ValueError(
            f"{commonMessage}, but available axes for histogram are {hvar.axes.name}"
        )

    if len(newDecorrAxesNames) == 0:
        newDecorrAxesNames = [f"{n}_decorr" for n in axesToDecorrNames]
    elif len(axesToDecorrNames) != len(newDecorrAxesNames):
        raise ValueError(
            f"If newDecorrAxisName are specified, they must have the same length than axisToDecorrName, but they are {newDecorrAxesNames} and {axesToDecorrNames}."
        )

    # subtract nominal hist to get variation only
    hvar = hh.addHists(hvar, hnom, scale2=-1)
    # expand edges for variations on diagonal elements
    hvar = hh.expand_hist_by_duplicate_axes(
        hvar, axesToDecorrNames, newDecorrAxesNames, put_trailing=True
    )
    # rebin duplicated axes
    if len(axlim) or len(rebin):
        hvar = hh.rebinHistMultiAx(
            hvar, newDecorrAxesNames, rebin, axlim[::2], axlim[1::2]
        )

    for ax, absval in zip(newDecorrAxesNames, absval):
        if absval:
            logger.info(f"Taking the absolute value of axis '{ax}'")
            hvar = hh.makeAbsHist(hvar, ax, rename=False)
    # add back nominal histogram while broadcasting
    hvar = hh.addHists(hvar, hnom)

    # if there is a mirror axis, put it at the end, since CardTool.py requires it like that
    if (
        "mirror" in hvar.axes.name
        and hvar.axes.name.index("mirror") != len(hvar.shape) - 1
    ):
        sortedAxes = [n for n in hvar.axes.name if n != "mirror"]
        sortedAxes.append("mirror")
        hvar = hvar.project(*sortedAxes)

    return hvar


def correct_bw_xsec(h, h_ref):
    """
    Normalize the Breit-Wigner mass variation histograms
    to the cross section from the MiNNLO mass variation histograms.
    Assumes that the histograms have been filled with the same mass variations, in the same order.
    """

    if "massShift" in h.axes.name:
        var = "massShift"
    elif "width" in h.axes.name:
        var = "width"

    h_axis_labels = [n for n in h.axes[var]]
    h_ref_axis_labels = [n for n in h_ref.axes[var]]
    if (
        len(h_axis_labels) != len(h_ref_axis_labels)
        or h_axis_labels != h_ref_axis_labels
    ):
        logger.warning(
            f"Breit-Wigner variations do not match MiNNLO variations: {h_axis_labels} vs {h_ref_axis_labels}."
            "Cannot apply correction."
        )
        return h

    h_corr = hh.divideHists(h.project(var), h_ref.project(var))
    h = hh.multiplyHists(h, h_corr)

    return h


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
        skipEntries=theory_utils.massWeightNames(proc=label, exclude=50),
        noi=not constrain,
        noConstraint=not constrain,
        mirror=False,
        systAxes=["massShift"],
        passToFakes=passSystToFakes,
    )
    # mass difference by swapping the +50MeV with the -50MeV variations for half of the bins
    args = ["massShift", f"massShift{label}50MeVUp", f"massShift{label}50MeVDown"]
    if any(mass_diff_var == var for var in ["charge", "utAngleSign"]):
        datagroups.addSystematic(
            **mass_diff_args,
            # # on gen level based on the sample, only possible for mW
            # preOpMap={m.name: (lambda h, swap=swap_bins: swap(h, "massShift", f"massShift{label}50MeVUp", f"massShift{label}50MeVDown"))
            #     for p in processes for g in datagroups.procGroups[p] for m in datagroups.groups[g].members if "minus" in m.name},
            # on reco level based on reco charge
            preOp=lambda h: hh.swap_histogram_bins(h, *args, mass_diff_var, 0),
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


def add_width_diff_variations(
    datagroups,
    width_diff_var,
    name,
    processes,
    constrain=False,
    suffix="",
    label="W",
    passSystToFakes=True,
):
    width_diff_args = dict(
        histname=name,
        name=f"widthDiff{suffix}{label}",
        processes=processes,
        group=f"widthDiff{label}",
        systNameReplace=[(f"width{label}", f"width{label}Diff{suffix}")],
        skipEntries=theory_utils.widthWeightNames(
            proc=label, exclude=(2.09053, 2.09173)
        ),  # use 0.6 MeV variation, it is the only symmetric one we have
        noi=not constrain,
        noConstraint=not constrain,
        mirror=False,
        systAxes=["width"],
        passToFakes=passSystToFakes,
    )
    # width difference by swapping the +0.6 MeV with the -0.6 MeV variations for half of the bins
    args = ["width", f"width{label}0p6MeVUp", f"width{label}0p6MeVDown"]
    if any(width_diff_var == var for var in ["charge", "utAngleSign"]):
        datagroups.addSystematic(
            **width_diff_args,
            preOp=lambda h: hh.swap_histogram_bins(h, *args, width_diff_var, 0),
        )

    elif width_diff_var == "eta-sign":
        datagroups.addSystematic(
            **width_diff_args,
            preOp=lambda h: hh.swap_histogram_bins(
                h, *args, "eta", hist.tag.Slicer()[0 : complex(0, 0) :]
            ),
        )

    elif width_diff_var == "eta-range":
        datagroups.addSystematic(
            **width_diff_args,
            preOp=lambda h: hh.swap_histogram_bins(
                h, *args, "eta", hist.tag.Slicer()[complex(0, -0.9) : complex(0, 0.9) :]
            ),
        )


def add_V_mass_uncertainty(
    datagroups,
    processes,
    args,
    passSystToFakes=True,
    label="W",
    massVariation=100,
    constrainMass=False,
    decorwidth=False,
):

    # This function is supposed to deal with the case where the mass is the fit NOI.
    # If this is the mW fit, then the nuisance parameter controlling the uncertainty
    # in mZ has to be defined elsewhere

    massWeightName = (
        f"massWeight_widthdecor{label}" if decorwidth else f"massWeight{label}"
    )
    mass_info = dict(
        processes=processes,
        group=f"massShift",
        noi=not constrainMass,
        skipEntries=theory_utils.massWeightNames(proc=label, exclude=massVariation),
        mirror=False,
        noConstraint=not constrainMass,
        systAxes=["massShift"],
        passToFakes=passSystToFakes,
    )

    if args.breitwignerWMassWeights and label == "W":
        preOpMap = {}
        for group in ["Wmunu", "Wtaunu"]:
            if group not in datagroups.groups.keys():
                continue
            for member in datagroups.groups[group].members:
                h_ref = datagroups.readHist(
                    datagroups.nominalName, member, massWeightName
                )
                preOpMap[member.name] = lambda h, h_ref=h_ref: correct_bw_xsec(h, h_ref)

        datagroups.addSystematic(
            histname=f"breitwigner_{massWeightName}",
            name=f"massWeight{label}",
            preOpMap=preOpMap,
            **mass_info,
        )
    else:
        if len(args.fitMassDecorr) == 0:
            datagroups.addSystematic(
                massWeightName,
                **mass_info,
            )
        else:
            suffix = "".join([a.capitalize() for a in args.fitMassDecorr])
            new_names = [f"{a}_decorr" for a in args.fitMassDecorr]
            datagroups.addSystematic(
                histname=massWeightName,
                processes=processes,
                name=f"massDecorr{suffix}{label}",
                group=f"massDecorr{label}",
                # systNameReplace=[("Shift",f"Diff{suffix}")],
                skipEntries=[
                    (x, *[-1] * len(args.fitMassDecorr))
                    for x in theory_utils.massWeightNames(
                        proc=label, exclude=args.massVariation
                    )
                ],
                noi=not constrainMass,
                noConstraint=not constrainMass,
                mirror=False,
                systAxes=["massShift", *new_names],
                passToFakes=passSystToFakes,
                # isPoiHistDecorr is a special flag to deal with how the massShift variations are internally formed
                isPoiHistDecorr=len(args.fitMassDecorr),
                actionRequiresNomi=True,
                action=decorrelateByAxes,
                actionArgs=dict(
                    axesToDecorrNames=args.fitMassDecorr,
                    newDecorrAxesNames=new_names,
                    axlim=args.decorrAxlim,
                    rebin=args.decorrRebin,
                    absval=args.decorrAbsval,
                ),
            )

        if "massdiffW" in args.noi:
            suffix = "".join([a.capitalize() for a in args.massDiffWVar.split("-")])
            add_mass_diff_variations(
                datagroups,
                args.massDiffWVar,
                name=massWeightName,
                processes=processes,
                constrain=constrainMass,
                suffix=suffix,
                label=label,
                passSystToFakes=passSystToFakes,
            )

        elif "massdiffZ" in args.noi:
            suffix = "".join([a.capitalize() for a in args.massDiffZVar.split("-")])
            add_mass_diff_variations(
                datagroups,
                args.massDiffZVar,
                name=massWeightName,
                processes=processes,
                constrain=constrainMass,
                suffix=suffix,
                label=label,
                passSystToFakes=passSystToFakes,  # automatically False for Z
            )


def add_W_width_uncertainty(
    datagroups,
    processes,
    args,
    passSystToFakes=True,
    label="W",
):
    widthVarTag = ""
    if (
        args.widthVariationW[0] == args.widthVariationW[1]
        and args.widthVariationW[0] == "0.6"
    ):
        widthVarTag = "WidthW0p6MeV"
        width_info = dict(
            name=widthVarTag,
            skipEntries=theory_utils.widthWeightNames(
                proc="W", exclude=(2.09053, 2.09173)
            ),
            systNameReplace=[["2p09053GeV", "Down"], ["2p09173GeV", "Up"]],
        )
    else:
        widthLowValues = {
            "0.6": "2.09053",
            "6": "2.085",
            "48": "2.043",
        }
        widthHighValues = {
            "0.6": "2.09173",
            "36": "2.127",
        }
        widthVarDown = args.widthVariationW[0].replace(".", "p")
        widthVarUp = args.widthVariationW[1].replace(".", "p")
        widthVarTag = f"WidthWm{widthVarDown}p{widthVarUp}MeV"
        wlv = widthLowValues[args.widthVariationW[0]]
        whv = widthHighValues[args.widthVariationW[1]]
        wlvStr = wlv.replace(".", "p") + "GeV"
        whvStr = whv.replace(".", "p") + "GeV"
        width_info = dict(
            name=widthVarTag,
            skipEntries=theory_utils.widthWeightNames(
                proc="W", exclude=(float(wlv), float(whv))
            ),
            systNameReplace=[[wlvStr, "Down"], [whvStr, "Up"]],
        )

    width_info.update(
        dict(
            processes=processes,
            groups=["widthW", "theory"],
            mirror=False,
            noi="wwidth" in args.noi,
            noConstraint="wwidth" in args.noi,
            systAxes=["width"],
            passToFakes=passSystToFakes,
        )
    )
    widthWeightName = f"widthWeight{label}"
    if args.breitwignerWMassWeights:
        preOpMap = {}
        for group in ["Wmunu", "Wtaunu"]:
            if group not in datagroups.groups.keys():
                continue
            for member in datagroups.groups[group].members:
                h_ref = datagroups.readHist(
                    datagroups.nominalName, member, widthWeightName
                )
                preOpMap[member.name] = lambda h, h_ref=h_ref: correct_bw_xsec(h, h_ref)
        datagroups.addSystematic(
            histname=f"breitwigner_{widthWeightName}",
            preOpMap=preOpMap,
            **width_info,
        )
    else:
        if len(args.fitWidthDecorr) == 0:
            datagroups.addSystematic(
                widthWeightName,
                **width_info,
            )
        else:
            suffix = "".join([a.capitalize() for a in args.fitWidthDecorr])
            new_names = [f"{a}_decorr" for a in args.fitWidthDecorr]
            datagroups.addSystematic(
                histname=widthWeightName,
                processes=processes,
                name=f"widthDecorr{suffix}{label}",
                groups=[f"widthDecorr{label}", "theory"],
                skipEntries=[
                    (x, *[-1] * len(args.fitWidthDecorr))
                    for x in width_info["skipEntries"]
                ],
                noi="wwidth" in args.noi,
                noConstraint="wwidth" in args.noi,
                mirror=False,
                systAxes=["width", *new_names],
                systNameReplace=width_info["systNameReplace"],
                passToFakes=passSystToFakes,
                # isPoiHistDecorr is a special flag to deal with how the massShift variations are internally formed
                isPoiHistDecorr=len(args.fitWidthDecorr),
                actionRequiresNomi=True,
                action=decorrelateByAxes,
                actionArgs=dict(
                    axesToDecorrNames=args.fitWidthDecorr,
                    newDecorrAxesNames=new_names,
                    axlim=args.decorrAxlim,
                    rebin=args.decorrRebin,
                    absval=args.decorrAbsval,
                ),
            )

        if "widthdiffW" in args.noi:
            suffix = "".join([a.capitalize() for a in args.widthDiffWVar.split("-")])
            add_width_diff_variations(
                datagroups,
                args.widthDiffWVar,
                name=widthWeightName,
                processes=processes,
                constrain="wwidth" not in args.noi,
                suffix=suffix,
                label=label,
                passSystToFakes=passSystToFakes,
            )


def add_recoil_uncertainty(
    datagroups,
    samples,
    passSystToFakes=False,
    pu_type="highPU",
    flavor="",
    group_compact=True,
):
    met = datagroups.args_from_metadata("met")
    if flavor == "":
        flavor = datagroups.args_from_metadata("flavor")
    if pu_type == "highPU" and (
        met in ["RawPFMET", "DeepMETReso", "DeepMETPVRobust", "DeepMETPVRobustNoPUPPI"]
    ):
        datagroups.addSystematic(
            "recoil_stat",
            processes=samples,
            mirror=True,
            groups=[
                "recoil",
                "recoil_stat",
                "experiment",
                "expNoCalib",
                "expNoLumi",
            ],
            systAxes=["recoil_unc"],
            passToFakes=passSystToFakes,
        )
        datagroups.addSystematic(
            "recoil_syst",
            processes=samples,
            mirror=True,
            groups=[
                "recoil",
                "recoil_syst",
                "experiment",
                "expNoCalib",
                "expNoLumi",
            ],
            systAxes=["recoil_unc"],
            passToFakes=passSystToFakes,
        )

    if pu_type == "lowPU":
        group_compact = False
        datagroups.addSystematic(
            "recoil_syst",
            processes=samples,
            mirror=True,
            groups=[
                "recoil" if group_compact else "recoil_syst",
                "experiment",
                "expNoCalib",
                "expNoLumi",
            ],
            systAxes=["recoil_unc"],
            passToFakes=passSystToFakes,
        )

        datagroups.addSystematic(
            "recoil_stat",
            processes=samples,
            mirror=True,
            groups=[
                "recoil" if group_compact else "recoil_stat",
                "experiment",
                "expNoCalib",
                "expNoLumi",
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

    nominalName = source[0]
    histname = source[1]

    logger.info(f"Now in channel {datagroups.channel} to make beta variations")

    procs_to_add = datagroups.expandProcesses([samples])

    datagroups.loadHistsForDatagroups(
        nominalName,
        histname,
        label="syst",
        procsToRead=procs_to_add,
    )

    if len(procs_to_add) != 1:
        logger.debug(
            f"Does only work for exactly one process at a time, got {procs_to_add}!"
        )

    proc = procs_to_add[0]

    # signal region selection
    if wmass:
        action_sel = lambda h, x: histselections.SignalSelectorABCD(h[x]).get_hist(h[x])
    else:
        action_sel = lambda h, x: h[x]

    integration_var = {
        a: hist.sum for a in datagroups.gen_axes_names
    }  # integrate out gen axes for bin by bin uncertainties
    integration_var["acceptance"] = hist.sum

    hnom = datagroups.groups[proc].hists[datagroups.nominalName]
    hnom = hnom.project(*datagroups.gen_axes_names)

    hvar = datagroups.groups[proc].hists["syst"]
    hvar_acc = action_sel(hvar, {"acceptance": True}).project(
        *recovar, *datagroups.gen_axes_names
    )
    hvar_reco = action_sel(hvar, integration_var)

    rel_unc = np.sqrt(hvar_reco.variances(flow=True)) / hvar_reco.values(flow=True)

    hvar = hh.scaleHist(
        hvar_acc, rel_unc[..., *[np.newaxis] * len(datagroups.gen_axes_names)]
    )

    datagroups.writer.add_beta_variations(
        hvar,
        proc,
        source_channel=datagroups.channel.replace("_masked", ""),
        dest_channel=datagroups.channel,
    )


def add_nominal_with_correlated_BinByBinStat(
    datagroups, wmass, base_name, masked, masked_flow_axes=[]
):
    # signal MC stat is correlated between detector level and gen level with explicit parameters
    #   setting signal histogram variances to 0 in detector level
    #   subtracting signal histogram variaiances of detector level from gen level to keep only contribution that is not at detector level
    if wmass:
        action_sel = lambda h, x: histselections.SignalSelectorABCD(h[x]).get_hist(h[x])
    else:
        action_sel = lambda h, x: h[x]

    # load gen level nominal
    datagroups.loadHistsForDatagroups(
        baseName=datagroups.nominalName,
        syst=datagroups.nominalName,
        procsToRead=datagroups.groups.keys(),
        label=datagroups.nominalName,
        forceNonzero=False,
        sumFakesPartial=True,
    )

    # load generator level nominal
    gen_name = f"{base_name}_yieldsUnfolding_theory_weight"
    datagroups.loadHistsForDatagroups(
        baseName="nominal",
        syst=gen_name,
        procsToRead=datagroups.groups.keys(),
        label=gen_name,
        forceNonzero=False,
        sumFakesPartial=True,
    )

    for proc in datagroups.predictedProcesses():
        logger.info(f"Add process {proc} in channel {datagroups.channel}")

        # nominal histograms of prediction
        norm_proc_hist_reco = datagroups.groups[proc].hists[gen_name]
        norm_proc_hist = datagroups.groups[proc].hists[datagroups.nominalName]

        norm_proc_hist_reco = action_sel(norm_proc_hist_reco, {"acceptance": True})

        if norm_proc_hist_reco.axes.name != datagroups.fit_axes:
            norm_proc_hist_reco = norm_proc_hist_reco.project(*datagroups.fit_axes)

        if norm_proc_hist.axes.name != datagroups.fit_axes:
            norm_proc_hist = norm_proc_hist.project(*datagroups.fit_axes)

        norm_proc_hist.variances(flow=True)[...] = norm_proc_hist.variances(
            flow=True
        ) - norm_proc_hist_reco.variances(flow=True)

        datagroups.groups[proc].hists[datagroups.nominalName]

        if len(masked_flow_axes) > 0:
            datagroups.axes_disable_flow = [
                n
                for n in norm_proc_hist.axes.name
                if n not in masked_flow_axes and n != "helicitySig"
            ]
            norm_proc_hist = hh.disableFlow(
                norm_proc_hist, datagroups.axes_disable_flow
            )

        if datagroups.channel not in datagroups.writer.channels:
            datagroups.writer.add_channel(
                axes=norm_proc_hist.axes,
                name=datagroups.channel,
                masked=masked,
                flow=len(masked_flow_axes) > 0,
            )

        datagroups.writer.add_process(
            norm_proc_hist,
            proc,
            datagroups.channel,
            signal=proc in datagroups.unconstrainedProcesses,
        )


def add_mb_fo_uncertainty(
    datagroups,
    processes="signal_samples",
    passSystToFakes=True,
    passToFakes=None,
):
    if passToFakes is not None:
        passSystToFakes = passToFakes

    corr_hist_name = "MiNNLO_Zbb_Corr"
    processes_expanded = datagroups.expandProcesses(processes)
    processes_with_corr = []

    for proc in processes_expanded:
        members = datagroups.groups[proc].members
        has_corr = any(
            corr_hist_name in datagroups.results[member.name]["output"]
            for member in members
            if member.name in datagroups.results
            and "output" in datagroups.results[member.name]
        )
        if has_corr:
            processes_with_corr.append(proc)

    if len(processes_with_corr) == 0:
        logger.info(
            f"Skip mb_fo systematic: histogram '{corr_hist_name}' is not available"
        )
        return

    # b-quark mass uncertainty from dedicated MiNNLO_Zbb correction histogram
    datagroups.addSystematic(
        corr_hist_name,
        name="mb_fo",
        processes=processes_with_corr,
        mirror=True,
        scale=1.0,
        systAxes=["vars"],
        skipEntries=[{"vars": ["nominal"]}],
        passToFakes=passSystToFakes,
        groups=["bcQuarkMass", "theory"],
    )


def add_electroweak_uncertainty(
    datagroups,
    ewUncs,
    flavor="mu",
    samples="single_v_samples",
    passSystToFakes=True,
    wlike=False,
):
    # different uncertainty for W and Z samples
    all_samples = datagroups.procGroups[samples]
    z_samples = [p for p in all_samples if p[0] == "Z"]
    w_samples = [p for p in all_samples if p[0] == "W"]

    for ewUnc in ewUncs:
        if "renesanceEW" in ewUnc:
            if w_samples:
                # add renesance (virtual EW) uncertainty on W samples
                datagroups.addSystematic(
                    f"{ewUnc}_Corr",
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
                datagroups.addSystematic(
                    f"{ewUnc}_Corr",
                    preOp=lambda h: h[{"weak": ["weak_ps", "weak_aem"]}],
                    processes=z_samples,
                    labelsByAxis=[f"{ewUnc}_Corr"],
                    scale=1.0,
                    systAxes=["weak"],
                    mirror=True,
                    groups=[f"theory_ew_virtZ_scheme", "theory_ew", "theory"],
                    passToFakes=passSystToFakes,
                    name="ewScheme",
                )
                datagroups.addSystematic(
                    f"{ewUnc}_Corr",
                    preOp=lambda h: h[{"weak": ["weak_default"]}],
                    processes=z_samples,
                    labelsByAxis=[f"{ewUnc}_Corr"],
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

            datagroups.addSystematic(
                f"{ewUnc}_Corr",
                systAxes=["systIdx"],
                mirror=True,
                passToFakes=passSystToFakes,
                processes=samples,
                labelsByAxis=[f"{ewUnc}_Corr"],
                scale=scale,
                preOp=preOp,
                groups=[f"theory_ew_{ewUnc}", "theory_ew", "theory"],
            )


def get_scalemap(datagroups, axes, gen_level, select={}, rename_axes={}):
    # make sure each gen bin variation has a similar effect in the reco space so that
    #  we have similar sensitivity to all parameters within the given up/down variations
    #  the scale map must have identical values in the fitted and corresponding masked channel
    signal_samples = datagroups.procGroups["signal_samples"]
    hScale = datagroups.getHistsForProcAndSyst(
        signal_samples[0],
        f"{gen_level}_yieldsUnfolding",
        nominal_name="nominal",
        applySelection=False,
    )
    hScale = hScale[{"acceptance": True, **select}]
    hScale.values(flow=True)[...] = abs(hScale.values(flow=True))
    hScale = hScale.project(*axes)
    hScale = hh.disableFlow(hScale, ["absYVGen", "absEtaGen"])
    for o, n in rename_axes.items():
        hScale.axes[o]._raw_metadata["name"] = n
    # scalemap with preserving normalization
    hScale.values(flow=True)[...] = (
        1.0
        / hScale.values(flow=True)
        * hScale.sum(flow=True).value
        / np.prod(hScale.values(flow=True).shape)
    )
    return hScale


def add_noi_unfolding_variations(
    datagroups,
    label,
    passSystToFakes,
    poi_axes,
    prior_norm=1,
    scale_norm=1,
    poi_axes_flow=["ptGen", "ptVGen"],
    gen_level="postfsr",
    process="signal_samples",
    scalemap=None,
    fitresult=None,
    constrained=False,
):

    poi_axes_syst = [f"_{n}" for n in poi_axes] if datagroups.xnorm else poi_axes[:]
    noi_args = dict(
        histname=(
            gen_level if datagroups.xnorm else f"nominal_{gen_level}_yieldsUnfolding"
        ),
        name=f"nominal_{gen_level}_yieldsUnfolding",
        baseName=f"{label}_",
        group=f"normXsec{label}",
        passToFakes=passSystToFakes,
        systAxes=poi_axes_syst,
        processes=[process],
        noConstraint=not constrained,
        noi=True,
        mirror=True,
        scale=(
            1 if prior_norm < 0 else prior_norm
        ),  # histogram represents an (args.priorNormXsec*100)% prior
        labelsByAxis=[f"_{p}" if p != poi_axes[0] else p for p in poi_axes],
    )

    if fitresult is not None:
        # produce a scalemap based on uncertainties of the gen bin variations of an initial fit

        from rabbit.io_tools import get_fitresult

        mapping = "Select"

        fitresult, meta = get_fitresult(fitresult, meta=True)
        results = fitresult["mappings"][mapping]["channels"]["ch0_masked"]

        scalemap = results[f"hist_postfit_inclusive"].get()

        scalemap.values(flow=True)[...] = scalemap.variances(
            flow=True
        ) ** 0.5 / scalemap.values(flow=True)

    if datagroups.xnorm:

        def make_poi_xnorm_variations(h, poi_axes, poi_axes_syst, norm, h_scale=None):
            h = hh.disableFlow(
                h,
                [
                    "absYVGen",
                    "absEtaGen",
                ],
            )
            hVar = hh.expand_hist_by_duplicate_axes(
                h, poi_axes[::-1], poi_axes_syst[::-1]
            )

            if h_scale is not None:
                hVar = hh.multiplyHists(hVar, h_scale)
            return hh.addHists(h, hVar, scale2=norm)

        if scalemap is None:
            scalemap = get_scalemap(
                datagroups,
                poi_axes,
                gen_level,
                rename_axes={o: n for o, n in zip(poi_axes, poi_axes_syst)},
            )

        datagroups.addSystematic(
            **noi_args,
            systAxesFlow=[f"_{n}" for n in poi_axes if n in poi_axes_flow],
            action=make_poi_xnorm_variations,
            actionArgs=dict(
                poi_axes=poi_axes,
                poi_axes_syst=poi_axes_syst,
                norm=scale_norm,
                h_scale=scalemap,
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
            hVar = hh.disableFlow(hVar, ["absYVGen", "absEtaGen"])

            if h_scale is not None:
                hVar = hh.multiplyHists(hVar, h_scale)

            return hh.addHists(hNom, hVar, scale2=norm)

        if scalemap is None:  # or fitresult is not None:
            scalemap = get_scalemap(datagroups, poi_axes, gen_level)

        datagroups.addSystematic(
            **noi_args,
            systAxesFlow=[n for n in poi_axes if n in poi_axes_flow],
            preOpMap={
                m.name: make_poi_variations
                for g in datagroups.expandProcess(process)
                for m in datagroups.groups[g].members
            },
            preOpArgs=dict(
                poi_axes=poi_axes,
                norm=scale_norm,
                h_scale=scalemap,
            ),
        )


def add_bsm_mixing(
    datagroups,
    inputBaseName,
    bsm_name,
    mixing=0.01,
    passSystToFakes=True,
):

    # Scale SM down with `f(x) = 1-x'
    preOpMap = {
        m.name: lambda h, x=mixing: hh.scaleHist(h, 1 - x)
        for m in datagroups.groups["Wmunu"].members
    }

    if mixing > 0:
        # load bsm members
        bsm_member_info = datagroups.get_members_from_results(
            startswith=f"{bsm_name}_{datagroups.era}"
        )
        bsm_members = [Datagroup_member(k, v) for k, v in bsm_member_info.items()]

        if len(bsm_members) != 1:
            raise NotImplementedError(
                f"Expected exactly 1 BSM member, but got {len(bsm_members)}"
            )

        # Get SM cross section
        xsec = 0
        for m in datagroups.groups["Wmunu"].members:
            xsec += m.xsec

        # scale BSM cross section to SM cross section
        for m in bsm_members:
            m.xsec = xsec

        # Scale BSM up with `f(x) = x'
        preOpMap.update(
            {m.name: lambda h, x=mixing: hh.scaleHist(h, x) for m in bsm_members}
        )

        # add BSM sample to Wmunu group for this systematic
        datagroups.groups["Wmunu"].addMembers(bsm_members)

    datagroups.addSystematic(
        histname=inputBaseName,
        name=f"{bsm_name}_mixing",
        processes=["Wmunu"],
        mirror=True,
        noi=True,
        noConstraint=True,
        passToFakes=passSystToFakes,
        preOpMap=preOpMap,
    )

    if mixing > 0:
        # remove the sample again
        datagroups.groups["Wmunu"].deleteMembers(bsm_members)


def add_bsm_process(
    datagroups,
    bsm_name,
):
    # add BSM sample as new process
    bsm_members = datagroups.get_members_from_results(
        startswith=f"{bsm_name}_{datagroups.era}"
    )
    if len(bsm_members) != 1:
        raise NotImplementedError(
            f"Expected exactly 1 BSM member, but got {len(bsm_members)}"
        )
    # since this group is created manually, the BSM is not added to the fakes (which is likely intented thing for BSM)
    datagroups.addGroup(
        bsm_name,
        members=bsm_members,
    )
    datagroups.unconstrainedProcesses.append(bsm_name)

    # Get SM cross section
    xsec = 0
    for m in datagroups.groups["Wmunu"].members:
        xsec += m.xsec

    # scale BSM cross section to SM cross section
    for m in datagroups.groups[bsm_name].members:
        m.xsec = xsec
