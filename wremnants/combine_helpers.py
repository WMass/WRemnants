from utilities import common, logging
from utilities.io_tools import input_tools

import numpy as np

logger = logging.child_logger(__name__)

def add_recoil_uncertainty(card_tool, samples, passSystToFakes=False, pu_type="highPU", flavor="", group_compact=True):
    met = input_tools.args_from_metadata(card_tool, "met")
    if flavor == "":
        flavor = input_tools.args_from_metadata(card_tool, "flavor")
    if pu_type == "highPU" and (met in ["RawPFMET", "DeepMETReso", "DeepMETPVRobust", "DeepMETPVRobustNoPUPPI"]):

        '''
        card_tool.addSystematic("recoil_syst",
            processes=samples,
            mirror = True,
            group = "recoil" if group_compact else "recoil_syst",
            splitGroup={"experiment": f".*"},
            systAxes = ["recoil_unc"],
            passToFakes=passSystToFakes,
        )
        '''

        card_tool.addSystematic("recoil_stat",
            processes=samples,
            mirror = True,
            group = "recoil" if group_compact else "recoil_stat",
            splitGroup={"experiment": f".*"},
            systAxes = ["recoil_unc"],
            passToFakes=passSystToFakes,
        )


def add_electroweak_uncertainty(card_tool, ewUncs, flavor="mu", samples="single_v_samples", passSystToFakes=True, wlike=False):
    info = dict(
        systAxes=["systIdx"],
        mirror=True,
        splitGroup={"theory_ew" : f".*", "theory" : f".*"},
        passToFakes=passSystToFakes,
    )
    # different uncertainty for W and Z samples
    all_samples = card_tool.procGroups[samples]
    z_samples = [p for p in all_samples if p[0]=="Z"]
    w_samples = [p for p in all_samples if p[0]=="W"]
    mode = card_tool.datagroups.mode
    
    for ewUnc in ewUncs:
        if "renesanceEW" in ewUnc:
            pass
            if w_samples:
                # add renesance (virtual EW) uncertainty on W samples
                card_tool.addSystematic(f"{ewUnc}Corr",
                    processes=w_samples,
                    preOp = lambda h : h[{"var": ["nlo_ew_virtual"]}],
                    labelsByAxis=[f"renesanceEWCorr"],
                    scale=1.,
                    systAxes=["var"],
                    group = "theory_ew_virtW_corr",
                    splitGroup={"theory_ew" : f".*", "theory" : f".*"},
                    passToFakes=passSystToFakes,
                    mirror = True,
                )
        elif ewUnc == "powhegFOEW":
            if z_samples:
                card_tool.addSystematic(f"{ewUnc}Corr",
                    preOp = lambda h : h[{"weak": ["weak_ps", "weak_aem"]}],
                    processes=z_samples,
                    labelsByAxis=[f"{ewUnc}Corr"],
                    scale=1.,
                    systAxes=["weak"],
                    mirror=True,
                    group="theory_ew_virtZ_scheme",
                    splitGroup={"theory_ew" : f".*", "theory" : f".*"},
                    passToFakes=passSystToFakes,
                    rename = "ewScheme",
                )
                card_tool.addSystematic(f"{ewUnc}Corr",
                    preOp = lambda h : h[{"weak": ["weak_default"]}],
                    processes=z_samples,
                    labelsByAxis=[f"{ewUnc}Corr"],
                    scale=1.,
                    systAxes=["weak"],
                    mirror=True,
                    group="theory_ew_virtZ_corr",
                    splitGroup={"theory_ew" : f".*", "theory" : f".*"},
                    passToFakes=passSystToFakes,
                    rename = "ew",
                )
        else:
            if "FSR" in ewUnc:
                if flavor == "e":
                    logger.warning("ISR/FSR EW uncertainties are not implemented for electrons, proceed w/o")
                    continue
                scale=1
            if "ISR" in ewUnc:
                scale=2
            else:
                scale=1

            if "winhac" in ewUnc:
                if not w_samples:
                    logger.warning("Winhac is not implemented for any other process than W, proceed w/o winhac EW uncertainty")
                    continue
                elif all_samples != w_samples:
                    logger.warning("Winhac is only implemented for W samples, proceed w/o winhac EW uncertainty for other samples")
                samples = w_samples
            else:
                samples = all_samples

            card_tool.addSystematic(f"{ewUnc}Corr", **info,
                processes=samples,
                labelsByAxis=[f"{ewUnc}Corr"],
                scale=scale,
                skipEntries=[(1, -1), (2, -1)] if ewUnc.startswith("virtual_ew") else [(0, -1), (2, -1)],
                group = f"theory_ew_{ewUnc}",
            )  

def projectABCD(cardTool, h, return_variances=False, dtype="float64"):
    # in case the desired axes are different at low MT and high MT we need to project each seperately, and then concatenate

    if any(ax not in h.axes.name for ax in cardTool.getFakerateAxes()):
        logger.warning(f"Not all desired fakerate axes found in histogram. Fakerate axes are {cardTool.getFakerateAxes()}, and histogram axes are {h.axes.name}")

    fakerate_axes = [n for n in h.axes.name if n in cardTool.getFakerateAxes()]

    lowMT_axes = [n for n in h.axes.name if n in fakerate_axes]
    highMT_failIso_axes = [n for n in h.axes.name if n in [*fakerate_axes, *cardTool.fit_axes]]
    highMT_passIso_axes = [n for n in h.axes.name if n in cardTool.fit_axes]

    hist_lowMT = h[{cardTool.nameMT : cardTool.failMT}].project(*[*lowMT_axes, common.passIsoName])
    hist_highMT_failIso = h[{cardTool.nameMT : cardTool.passMT, **common.failIso}].project(*[*highMT_failIso_axes])
    hist_highMT_passIso = h[{cardTool.nameMT : cardTool.passMT, **common.passIso}].project(*[*highMT_passIso_axes])

    flat_lowMT = hist_lowMT.values(flow=False).flatten().astype(dtype)
    flat_highMT_failIso = hist_highMT_failIso.values(flow=False).flatten().astype(dtype)
    flat_highMT_passIso = hist_highMT_passIso.values(flow=False).flatten().astype(dtype)

    flat = np.append(flat_lowMT, flat_highMT_failIso)
    flat = np.append(flat, flat_highMT_passIso)

    if not return_variances:
        return flat

    flat_variances_lowMT = hist_lowMT.variances(flow=False).flatten().astype(dtype)
    flat_variances_highMT_failIso = hist_highMT_failIso.variances(flow=False).flatten().astype(dtype)
    flat_variances_highMT_passIso = hist_highMT_passIso.variances(flow=False).flatten().astype(dtype)

    flat_variances = np.append(flat_variances_lowMT, flat_variances_highMT_failIso)
    flat_variances = np.append(flat_variances, flat_variances_highMT_passIso)

    return flat, flat_variances


