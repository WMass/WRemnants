import collections.abc
import pickle
import re

import hist
import lz4.frame
import numpy as np
import pandas as pd

from wremnants.postprocessing import pdf_tools
from wremnants.utilities import binning, samples, theory_utils
from wums import boostHistHelpers as hh
from wums import logging

logger = logging.child_logger(__name__)


def syst_transform_map(base_hist, hist_name):
    pdfInfo = theory_utils.pdfMap
    pdfNames = [pdfInfo[k]["name"] for k in pdfInfo.keys()]

    def pdfUnc(h, pdfName, axis_name="pdfVar"):
        key = list(pdfInfo.keys())[list(pdfNames).index(pdfName)]
        unc = pdfInfo[key]["combine"]
        scale = pdfInfo[key]["scale"] if "scale" in pdfInfo[key] else 1.0
        return pdf_tools.hessianPdfUnc(h, uncType=unc, scale=scale, axis_name=axis_name)

    def uncHist(unc):
        return unc if base_hist == "nominal" else f"{base_hist}_{unc}"

    transforms = {}
    transforms.update(
        {
            pdf
            + "Up": {
                "action": lambda h, p=pdf: (
                    pdfUnc(h, p)[0] if "pdfVar" in h.axes.name else h
                )
            }
            for pdf in pdfNames
        }
    )
    transforms.update(
        {
            pdf
            + "Down": {
                "action": lambda h, p=pdf: (
                    pdfUnc(h, p)[1] if "pdfVar" in h.axes.name else h
                )
            }
            for pdf in pdfNames
        }
    )
    transforms["nonpromptQCDNormUp"] = {
        "action": lambda h: h.copy() * 1.11,
        "procs": ["QCDmuEnrichPt15PostVFP"],
    }
    transforms["nonpromptQCDNormDown"] = {
        "action": lambda h: h.copy() * 0.89,
        "procs": ["QCDmuEnrichPt15PostVFP"],
    }
    transforms["scetlib_dyturboMSHT20Up"] = {
        "action": lambda h: pdfUnc(h, "pdfMSHT20", "vars")[0],
        "procs": samples.vprocs,
    }
    transforms["scetlib_dyturboMSHT20Down"] = {
        "action": lambda h: pdfUnc(h, "pdfMSHT20", "vars")[1],
        "procs": samples.vprocs,
    }
    transforms["scetlib_dyturboCT18ZUp"] = {
        "action": lambda h: pdfUnc(h, "pdfCT18Z", "vars")[0],
        "procs": samples.vprocs,
    }
    transforms["scetlib_dyturboCT18ZDown"] = {
        "action": lambda h: pdfUnc(h, "pdfCT18Z", "vars")[1],
        "procs": samples.vprocs,
    }
    transforms["scetlib_dyturboMSHT20an3loUp"] = {
        "action": lambda h: pdfUnc(h, "pdfMSHT20", "vars")[0],
        "procs": samples.zprocs,
    }
    transforms["scetlib_dyturboMSHT20an3loDown"] = {
        "action": lambda h: pdfUnc(h, "pdfMSHT20", "vars")[1],
        "procs": samples.zprocs,
    }
    transforms["ewUp"] = {
        "action": lambda h, **args: (
            h if "systIdx" not in h.axes.name else h[{"systIdx": 0}]
        )
    }
    transforms["ewDown"] = {
        "requiresNominal": True,
        "action": lambda h, **args: (
            h
            if "systIdx" not in h.axes.name
            else hh.mirrorHist(h[{"systIdx": 0}], **args)
        ),
    }
    transforms["muonScaleUp"] = {
        "action": lambda h: (
            h if "unc" not in h.axes.name else hh.rssHistsMid(h, "unc")[1]
        )
    }
    transforms["muonScaleDown"] = {
        "action": lambda h: (
            h if "unc" not in h.axes.name else hh.rssHistsMid(h, "unc")[0]
        )
    }
    transforms["muonScale3Up"] = {
        "action": lambda h: (
            h if "unc" not in h.axes.name else hh.rssHistsMid(h, "unc", 3.35)[1]
        )
    }
    transforms["muonScale3Down"] = {
        "action": lambda h: (
            h if "unc" not in h.axes.name else hh.rssHistsMid(h, "unc", 3.35)[0]
        )
    }
    transforms["muonResUp"] = {
        "requiresNominal": True,
        "action": lambda h, **args: (
            h
            if "smearing_variation" not in h.axes.name
            else hh.rssHists(h, "smearing_variation", **args)[1]
        ),
    }
    transforms["muonResDown"] = {
        "requiresNominal": True,
        "action": lambda h, **args: (
            h
            if "smearing_variation" not in h.axes.name
            else hh.rssHists(h, "smearing_variation", **args)[0]
        ),
    }

    s = hist.tag.Slicer()
    transforms.update(
        {
            "QCDscale_muRmuFUp": {
                "action": lambda h: (
                    h
                    if "muRfact" not in h.axes.name
                    else h[{"muRfact": 2.0j, "muFfact": 2.0j, "ptVgen": s[:: hist.sum]}]
                )
            },
            "QCDscale_muRmuFDown": {
                "action": lambda h: (
                    h
                    if "muRfact" not in h.axes.name
                    else h[{"muRfact": 0.5j, "muFfact": 0.5j, "ptVgen": s[:: hist.sum]}]
                )
            },
            "QCDscale_muRUp": {
                "action": lambda h: (
                    h
                    if "muRfact" not in h.axes.name
                    else h[{"muRfact": 2.0j, "muFfact": 1.0j, "ptVgen": s[:: hist.sum]}]
                )
            },
            "QCDscale_muRDown": {
                "action": lambda h: (
                    h
                    if "muRfact" not in h.axes.name
                    else h[{"muRfact": 0.5j, "muFfact": 1.0j, "ptVgen": s[:: hist.sum]}]
                )
            },
            "QCDscale_muFUp": {
                "action": lambda h: (
                    h
                    if "muRfact" not in h.axes.name
                    else h[{"muRfact": 1.0j, "muFfact": 2.0j, "ptVgen": s[:: hist.sum]}]
                )
            },
            "QCDscale_muFDown": {
                "action": lambda h: (
                    h
                    if "muRfact" not in h.axes.name
                    else h[{"muRfact": 1.0j, "muFfact": 0.5j, "ptVgen": s[:: hist.sum]}]
                )
            },
            "QCDscale_cen": {
                "action": lambda h: (
                    h
                    if "muRfact" not in h.axes.name
                    else h[{"muRfact": 1.0j, "muFfact": 1.0j, "ptVgen": s[:: hist.sum]}]
                )
            },
        }
    )

    def scetlibIdx(h, i):
        return (
            h
            if not ("vars" in h.axes.name and h.axes["vars"].size > i)
            else h[{"vars": i}]
        )

    def projAx(hname):
        return hname.split("-")

    resum_tnps = [
        "pdf0",
        "gamma_cusp+1",
        "gamma_mu_q+1",
        "gamma_nu+1",
        "h_qqV-0.5",
        "s+1",
        "b_qqV+1",
        "b_qqbarV+1",
        "b_qqS+1",
        "b_qqDS+1",
        "b_qg+1",
    ]
    resum_tnpsXp1_up = [
        "pdf0",
        "gamma_cusp1.",
        "gamma_mu_q1.",
        "gamma_nu1.",
        "s1.",
        "b_qqV0.5",
        "b_qqV0.5",
        "b_qqbarV0.5",
        "b_qqS0.5",
        "b_qqDS0.5",
        "b_qg0.5",
    ]
    resum_tnpsXp1_down = [
        "pdf0",
        "gamma_cusp-1.",
        "gamma_mu_q-1.",
        "gamma_nu-1.",
        "s-1.",
        "b_qqV-2.5",
        "b_qqV-2.5",
        "b_qqbarV-2.5",
        "b_qqS-2.5",
        "b_qqDS-2.5",
        "b_qg-2.5",
    ]
    resum_tnpsXp0_up = [
        "pdf0",
        "gamma_cusp1.",
        "gamma_mu_q1.",
        "gamma_nu1.",
        "s1.",
        "b_qqV0.5",
        "b_qqV0.5",
        "b_qqbarV0.5",
        "b_qqS0.5",
        "b_qqDS0.5",
        "b_qg0.5",
    ]
    resum_tnpsXp0_down = [
        "pdf0",
        "gamma_cusp-1.",
        "gamma_mu_q-1.",
        "gamma_nu-1.",
        "s-1.",
        "b_qqV-0.5",
        "b_qqV-0.5",
        "b_qqbarV-0.5",
        "b_qqS-0.5",
        "b_qqDS-0.5",
        "b_qg-0.5",
    ]
    resum_tnpbeam_up = [
        "pdf0",
        "b_qqV0.5",
        "b_qqbarV0.5",
        "b_qqS0.5",
        "b_qqDS0.5",
        "b_qg0.5",
    ]
    resum_tnpbeam_down = [
        "pdf0",
        "b_qqV-0.5",
        "b_qqV-0.5",
        "b_qqbarV-0.5",
        "b_qqS-0.5",
        "b_qqDS-0.5",
        "b_qg-0.5",
    ]

    transforms.update(
        {
            "resumFOScaleUp": {"action": lambda h: scetlibIdx(h, 2)},
            "resumFOScaleDown": {"action": lambda h: scetlibIdx(h, 1)},
            "resumLambdaDown": {"action": lambda h: scetlibIdx(h, 3)},
            "resumLambdaUp": {"action": lambda h: scetlibIdx(h, 4)},
            "resumTransitionUp": {
                "action": lambda h: hh.syst_min_or_max_env_hist(
                    h,
                    projAx(hist_name),
                    "vars",
                    [
                        "transition_points0.2_0.65_1.1",
                        "transition_points0.4_0.55_0.7",
                        "transition_points0.2_0.45_0.7",
                        "transition_points0.4_0.75_1.1",
                    ],
                    no_flow=["ptVgen"],
                    do_min=False,
                )
            },
            "resumTransitionDown": {
                "action": lambda h: hh.syst_min_or_max_env_hist(
                    h,
                    projAx(hist_name),
                    "vars",
                    [
                        "transition_points0.2_0.65_1.1",
                        "transition_points0.4_0.55_0.7",
                        "transition_points0.2_0.45_0.7",
                        "transition_points0.4_0.75_1.1",
                    ],
                    no_flow=["ptVgen"],
                    do_min=True,
                )
            },
            "resumTNPBeamUp": {
                "action": lambda h: (
                    h
                    if "vars" not in h.axes.name
                    else hh.rssHists(h[{"vars": resum_tnpbeam_up}], "vars")[0]
                )
            },
            "resumTNPBeamDown": {
                "action": lambda h: (
                    h
                    if "vars" not in h.axes.name
                    else hh.rssHists(h[{"vars": resum_tnpbeam_down}], "vars")[1]
                )
            },
            "resumTNPXp1Up": {
                "action": lambda h: (
                    h
                    if "vars" not in h.axes.name
                    else hh.rssHists(h[{"vars": resum_tnpsXp0_up}], "vars")[0]
                )
            },
            "resumTNPXp0Down": {
                "action": lambda h: (
                    h
                    if "vars" not in h.axes.name
                    else hh.rssHists(h[{"vars": resum_tnpsXp0_down}], "vars")[1]
                )
            },
            "resumTNPXp0Up": {
                "action": lambda h: (
                    h
                    if "vars" not in h.axes.name
                    else hh.rssHists(h[{"vars": resum_tnpsXp1_up}], "vars")[0]
                )
            },
            "resumTNPXp1Down": {
                "action": lambda h: (
                    h
                    if "vars" not in h.axes.name
                    else hh.rssHists(h[{"vars": resum_tnpsXp1_down}], "vars")[1]
                )
            },
            "resumTNPx5Up": {
                "action": lambda h: (
                    h
                    if "vars" not in h.axes.name
                    else hh.rssHists(h[{"vars": resum_tnps}], "vars", scale=5)[0]
                )
            },
            "resumTNPx5Down": {
                "action": lambda h: (
                    h
                    if "vars" not in h.axes.name
                    else hh.rssHists(h[{"vars": resum_tnps}], "vars", scale=5)[1]
                )
            },
            "resumTNPx12Up": {
                "action": lambda h: (
                    h
                    if "vars" not in h.axes.name
                    else hh.rssHists(h[{"vars": resum_tnps}], "vars", scale=12)[0]
                )
            },
            "resumTNPx12Down": {
                "action": lambda h: (
                    h
                    if "vars" not in h.axes.name
                    else hh.rssHists(h[{"vars": resum_tnps}], "vars", scale=12)[1]
                )
            },
            "resumScaleAllUp": {
                "action": lambda h: (
                    h
                    if "vars" not in h.axes.name
                    else hh.syst_min_or_max_env_hist(
                        h,
                        projAx(hist_name),
                        "vars",
                        [
                            x
                            for x in h.axes["vars"]
                            if any(
                                re.match(y, x)
                                for y in [
                                    "pdf0",
                                    "^nuB.*",
                                    "nuS.*",
                                    "^muB.*",
                                    "^muS.*",
                                    "kappa.*",
                                    "muf.*",
                                ]
                            )
                        ],
                        do_min=False,
                    )
                )
            },
            "resumScaleAllDown": {
                "action": lambda h: (
                    h
                    if "vars" not in h.axes.name
                    else hh.syst_min_or_max_env_hist(
                        h,
                        projAx(hist_name),
                        "vars",
                        [
                            x
                            for x in h.axes["vars"]
                            if any(
                                re.match(y, x)
                                for y in [
                                    "pdf0",
                                    "^nuB.*",
                                    "nuS.*",
                                    "^muB.*",
                                    "^muS.*",
                                    "kappa.*",
                                    "muf.*",
                                ]
                            )
                        ],
                        do_min=True,
                    )
                )
            },
            "resumScaleUp": {
                "action": lambda h: (
                    h
                    if "vars" not in h.axes.name
                    else hh.syst_min_or_max_env_hist(
                        h,
                        projAx(hist_name),
                        "vars",
                        [
                            x
                            for x in h.axes["vars"]
                            if any(
                                re.match(y, x)
                                for y in ["pdf0", "^nuB.*", "nuS.*", "^muB.*", "^muS.*"]
                            )
                        ],
                        do_min=False,
                    )
                )
            },
            "resumScaleDown": {
                "action": lambda h: (
                    h
                    if "vars" not in h.axes.name
                    else hh.syst_min_or_max_env_hist(
                        h,
                        projAx(hist_name),
                        "vars",
                        [
                            x
                            for x in h.axes["vars"]
                            if any(
                                re.match(y, x)
                                for y in ["pdf0", "^nuB.*", "nuS.*", "^muB.*", "^muS.*"]
                            )
                        ],
                        do_min=True,
                    )
                )
            },
            "resumNPUp": {
                "action": lambda h: (
                    h
                    if "vars" not in h.axes.name
                    else hh.rssHists(
                        h[
                            {
                                "vars": [
                                    "pdf0",
                                    "Lambda2-0.25",
                                    "Lambda20.25",
                                    "Lambda4.01",
                                    "Lambda4.16",
                                    "Delta_Lambda2-0.02",
                                    "Delta_Lambda20.02",
                                ]
                            }
                        ],
                        syst_axis="vars",
                        scale=0.5,
                    )[0]
                )
            },
            "resumNPDown": {
                "action": lambda h: (
                    h
                    if "vars" not in h.axes.name
                    else hh.rssHists(
                        h[
                            {
                                "vars": [
                                    "pdf0",
                                    "Lambda2-0.25",
                                    "Lambda20.25",
                                    "Lambda4.01",
                                    "Lambda4.16",
                                    "Delta_Lambda2-0.02",
                                    "Delta_Lambda20.02",
                                ]
                            }
                        ],
                        syst_axis="vars",
                        scale=0.5,
                    )[1]
                )
            },
            "scaleTransUp": {
                "action": lambda h: (
                    h
                    if "vars" not in h.axes.name
                    else hh.rssHists(
                        h[
                            {
                                "vars": [
                                    "pdf0",
                                    "renorm_scale_pt20_envelope_Up",
                                    "transition_points0.2_0.35_1.0",
                                    "transition_points0.2_0.75_1.0",
                                ]
                            }
                        ],
                        syst_axis="vars",
                        scale=0.5,
                    )[0]
                )
            },
            "scaleTransDown": {
                "action": lambda h: (
                    h
                    if "vars" not in h.axes.name
                    else hh.rssHists(
                        h[
                            {
                                "vars": [
                                    "pdf0",
                                    "renorm_scale_pt20_envelope_Down",
                                    "transition_points0.2_0.75_1.0",
                                    "transition_points0.2_0.75_1.0",
                                ]
                            }
                        ],
                        syst_axis="vars",
                        scale=0.5,
                    )[1]
                )
            },
            "resumCSNPUp": {
                "action": lambda h: (
                    h
                    if "vars" not in h.axes.name
                    else hh.rssHists(
                        h[{"vars": ["pdf0", "c_nu-0.1-omega_nu0.5", "omega_nu0.5"]}],
                        syst_axis="vars",
                        scale=0.5,
                    )[0]
                )
            },
            "resumCSNPDown": {
                "action": lambda h: (
                    h
                    if "vars" not in h.axes.name
                    else hh.rssHists(
                        h[{"vars": ["pdf0", "c_nu-0.1-omega_nu0.5", "omega_nu0.5"]}],
                        syst_axis="vars",
                        scale=0.5,
                    )[1]
                )
            },
            "resumCSNPhalfUp": {
                "action": lambda h: (
                    h
                    if "vars" not in h.axes.name
                    else hh.rssHists(
                        h[{"vars": ["pdf0", "c_nu-0.1-omega_nu0.5", "omega_nu0.5"]}],
                        syst_axis="vars",
                        scale=0.25,
                    )[0]
                )
            },
            "resumCSNPhalfDown": {
                "action": lambda h: (
                    h
                    if "vars" not in h.axes.name
                    else hh.rssHists(
                        h[{"vars": ["pdf0", "c_nu-0.1-omega_nu0.5", "omega_nu0.5"]}],
                        syst_axis="vars",
                        scale=0.25,
                    )[1]
                )
            },
            "resumNPOmegaUp": {
                "action": lambda h: (
                    hh.syst_min_or_max_env_hist(
                        h,
                        projAx(hist_name),
                        "vars",
                        [x for x in h.axes["vars"] if re.match(r"^Omega-*\d+", x)],
                        do_min=False,
                    )
                    if "vars" in h.axes.name
                    else h
                )
            },
            "resumNPOmegaDown": {
                "action": lambda h: (
                    hh.syst_min_or_max_env_hist(
                        h,
                        projAx(hist_name),
                        "vars",
                        [x for x in h.axes["vars"] if re.match(r"^Omega-*\d+", x)],
                        do_min=True,
                    )
                    if "vars" in h.axes.name
                    else h
                )
            },
            "resumNPomega_nuUp": {
                "action": lambda h: (
                    hh.syst_min_or_max_env_hist(
                        h,
                        projAx(hist_name),
                        "vars",
                        [x for x in h.axes["vars"] if re.match(r"^omega_nu-*\d+", x)],
                        do_min=False,
                    )
                    if "vars" in h.axes.name
                    else h
                )
            },
            "resumNPomega_nuDown": {
                "action": lambda h: (
                    hh.syst_min_or_max_env_hist(
                        h,
                        projAx(hist_name),
                        "vars",
                        [x for x in h.axes["vars"] if re.match(r"^omega_nu-*\d+", x)],
                        do_min=True,
                    )
                    if "vars" in h.axes.name
                    else h
                )
            },
            "resumNPc_nuUp": {
                "action": lambda h: (
                    hh.syst_min_or_max_env_hist(
                        h,
                        projAx(hist_name),
                        "vars",
                        [x for x in h.axes["vars"] if re.match(r"^c_nu-*\d+", x)],
                        do_min=False,
                    )
                    if "vars" in h.axes.name
                    else h
                )
            },
            "resumNPc_nuDown": {
                "action": lambda h: (
                    hh.syst_min_or_max_env_hist(
                        h,
                        projAx(hist_name),
                        "vars",
                        [x for x in h.axes["vars"] if re.match(r"^c_nu-*\d+", x)],
                        do_min=True,
                    )
                    if "vars" in h.axes.name
                    else h
                )
            },
            "resumScaleMax": {
                "action": lambda h: hh.syst_min_or_max_env_hist(
                    h,
                    projAx(hist_name),
                    "vars",
                    range(9, 44),
                    no_flow=["ptVgen"],
                    do_min=False,
                )
            },
            "resumScaleMin": {
                "action": lambda h: hh.syst_min_or_max_env_hist(
                    h,
                    projAx(hist_name),
                    "vars",
                    range(9, 44),
                    no_flow=["ptVgen"],
                    do_min=True,
                )
            },
        }
    )
    for k in [
        "gamma_cusp+5",
        "gamma_mu_q+5",
        "gamma_nu+5",
        "s+5",
        "b_qqV+5",
        "b_qqbarV+5",
        "b_qqS+5",
        "b_qqDS+5",
        "b_qg+5",
    ]:
        transforms[k.replace("+5", "-5")] = {
            "action": lambda h, v=k: (
                h
                if "vars" not in h.axes.name
                else hh.mirrorHist(h[{"vars": v}], h[{"vars": "pdf0"}])
            )
        }
    transforms["h_qqV+2.0"] = {
        "action": lambda h: (
            h
            if "vars" not in h.axes.name
            else hh.mirrorHist(h[{"vars": "h_qqV-2.0"}], h[{"vars": "pdf0"}])
        )
    }
    for k in [
        "gamma_cusp+1",
        "gamma_mu_q+1",
        "gamma_nu+1",
        "s+1",
        "b_qqV+1",
        "b_qqbarV+1",
        "b_qqS+1",
        "b_qqDS+1",
        "b_qg+1",
    ]:
        transforms[k.replace("+1", "-1")] = {
            "action": lambda h, v=k: (
                h
                if "vars" not in h.axes.name
                else hh.mirrorHist(h[{"vars": v}], h[{"vars": "pdf0"}])
            )
        }
    transforms["h_qqV+0.5"] = {
        "action": lambda h: (
            h
            if "vars" not in h.axes.name
            else hh.mirrorHist(h[{"vars": "h_qqV-0.5"}], h[{"vars": "pdf0"}])
        )
    }

    return transforms


def gen_scale_helicity_hist_to_variations(
    hist_in,
    gen_obs,
    sum_axes=[],
    pt_ax="ptVgen",
    gen_axes=["ptVgen", "chargeVgen", "helicity"],
    rebinPtV=None,
):
    scale_hist = hh.expand_hist_by_duplicate_axes(
        hist_in, gen_obs, [a + "Alt" for a in gen_obs], swap_axes=True
    )

    return scale_helicity_hist_to_variations(
        scale_hist, sum_axes, pt_ax, gen_axes, rebinPtV
    )


def scale_helicity_hist_to_variations(
    scale_hist,
    sum_axes=[],
    pt_ax="ptVgen",
    gen_axes=["ptVgen", "chargeVgen", "helicity"],
    rebinPtV=None,
):
    s = hist.tag.Slicer()
    axisNames = scale_hist.axes.name

    sum_expr = {axis: s[:: hist.sum] for axis in sum_axes if axis in axisNames}
    scale_hist = scale_hist[sum_expr]
    axisNames = scale_hist.axes.name

    # select nominal QCD scales, but keep the sliced axis at size 1 for broadcasting
    nom_scale_hist = scale_hist[
        {"muRfact": s[1.0j : 1.0j + 1], "muFfact": s[1.0j : 1.0j + 1]}
    ]
    # select nominal QCD scales and project down to nominal axes
    nom_sel = {"muRfact": s[1.0j], "muFfact": s[1.0j]}
    nom_sel.update(
        {genAxis: s[:: hist.sum] for genAxis in gen_axes if genAxis in axisNames}
    )
    nom_hist = nom_scale_hist[nom_sel]

    hasHelicityAxis = "helicity" in axisNames
    hasPtAxis = pt_ax in axisNames

    if rebinPtV is not None and hasPtAxis:
        # Treat single bin array as a float
        array_rebin = (
            isinstance(rebinPtV, collections.abc.Sequence)
            or type(rebinPtV) == np.ndarray
        )
        if array_rebin and len(rebinPtV) == 1:
            rebinPtV = rebinPtV[0]
            array_rebin = False

        if array_rebin:
            scale_hist = hh.rebinHist(scale_hist, pt_ax, rebinPtV)
            nom_scale_hist = hh.rebinHist(nom_scale_hist, pt_ax, rebinPtV)
        else:
            scale_hist = scale_hist[{pt_ax: s[:: hist.rebin(rebinPtV)]}]
            nom_scale_hist = nom_scale_hist[{pt_ax: s[:: hist.rebin(rebinPtV)]}]

    # difference between a given scale and the nominal, plus the sum
    # this emulates the "weight if idx else nominal" logic and corresponds to the decorrelated
    # variations
    if scale_hist.name is None:
        out_name = (
            "scale_helicity_variations"
            if hasHelicityAxis
            else "scale_vpt_variations" if hasPtAxis else "scale_vcharge_variations"
        )
    else:
        out_name = scale_hist.name + "_variations"

    nom_axes = nom_hist.axes
    if nom_axes != scale_hist.axes[: len(nom_axes)]:
        raise ValueError(
            "Cannot convert to variations histogram becuase the assumption that the order of the gen axes "
            "and reco-like axes is respected does not hold! Gen axes must be trailing! "
            f" Found nominal (reco-like) axes {nom_axes.name}, full axes {scale_hist.axes.name}"
        )

    expd = scale_hist.ndim - nom_hist.ndim
    expandnom = np.expand_dims(
        nom_hist.values(flow=True), [-expd + i for i in range(expd)]
    )
    systhist = (
        scale_hist.values(flow=True) - nom_scale_hist.values(flow=True) + expandnom
    )

    scale_variation_hist = hist.Hist(*scale_hist.axes, name=out_name, data=systhist)

    return scale_variation_hist


def gen_hist_to_variations(
    hist_in,
    gen_obs,
    gen_axes=["ptVgen", "chargeVgen", "helicity"],
    sum_axes=[],
    rebin_axes=[],
    rebin_edges=[],
):
    for obs in gen_obs:
        hist_in = hh.expand_hist_by_duplicate_axis(
            hist_in, obs, obs + "Alt", swap_axes=True
        )

    return hist_to_variations(hist_in, gen_axes, sum_axes, rebin_axes, rebin_edges)


def hist_to_variations(
    hist_in, gen_axes=[], sum_axes=[], rebin_axes=[], rebin_edges=[]
):

    if hist_in.name is None:
        out_name = "hist_variations"
    else:
        out_name = hist_in.name + "_variations"

    s = hist.tag.Slicer()

    # do rebinning
    for rebin_axis, edges in zip(rebin_axes, rebin_edges):
        hist_in = hh.rebinHist(hist_in, rebin_axis, edges)

    axisNames = hist_in.axes.name
    sum_expr = {axis: s[:: hist.sum] for axis in sum_axes if axis in axisNames}
    hist_in = hist_in[sum_expr]
    axisNames = hist_in.axes.name

    gen_sum_expr = {n: s[:: hist.sum] for n in gen_axes if n in axisNames}
    if len(gen_sum_expr) == 0:
        # all the axes have already been projected out, nothing else to do
        return hist_in

    nom_hist = hist_in[{"vars": 0}]
    nom_hist_sum = nom_hist[gen_sum_expr]

    # slices to broadcast nom_hist and nom_hist_sum to hist_in shape
    slices_nom = [
        slice(None) if n in nom_hist.axes.name else np.newaxis
        for n in hist_in.axes.name
    ]
    slices_nom_sum = [
        slice(None) if n in nom_hist_sum.axes.name else np.newaxis
        for n in hist_in.axes.name
    ]
    variation_data = (
        hist_in.view(flow=True)
        - nom_hist.view(flow=True)[*slices_nom]
        + nom_hist_sum.view(flow=True)[*slices_nom_sum]
    )

    variation_hist = hist.Hist(
        *hist_in.axes,
        storage=hist_in.storage_type(),
        name=out_name,
        data=variation_data,
    )

    return variation_hist


def scale_hist_up_down(h, scale):
    hUp = hh.scaleHist(h, scale)
    hDown = hh.scaleHist(h, 1 / scale)

    hVar = hist.Hist(
        *[a for a in h.axes],
        binning.down_up_axis,
        storage=hist.storage.Weight(),
    )
    hVar.values(flow=True)[...] = np.stack(
        [hDown.values(flow=True), hUp.values(flow=True)], axis=-1
    )
    hVar.variances(flow=True)[...] = np.stack(
        [hDown.variances(flow=True), hUp.variances(flow=True)], axis=-1
    )
    return hVar


def scale_hist_up_down_corr_from_file(h, corr_file=None, corr_hist=None):
    # FIXME: this might not be thread safe, but it is a test for now
    with lz4.frame.open(corr_file) as f:
        corrs = pickle.load(f)
    boost_corr = corrs[corr_hist]

    hUp = hh.multiplyHists(h, boost_corr)
    hDown = hh.divideHists(h, boost_corr)

    hVar = hist.Hist(
        *[a for a in h.axes],
        binning.down_up_axis,
        storage=hist.storage.Weight(),
    )
    hVar.values(flow=True)[...] = np.stack(
        [hDown.values(flow=True), hUp.values(flow=True)], axis=-1
    )
    hVar.variances(flow=True)[...] = np.stack(
        [hDown.variances(flow=True), hUp.variances(flow=True)], axis=-1
    )
    return hVar


# TODO: Integrate with rabbit to avoid code duplication
def symmetrize_unc_matrix(matrix, labels, symm_type):
    if symm_type not in ["quadratic", "average"]:
        raise NotImplementedError(f"Symmetrization type {symm_type} not supported!")

    if type(matrix) == pd.DataFrame:
        values = matrix.values
    elif type(matrix) == hist.Hist:
        values = matrix.values()
    else:
        values = matrix

    # logkup = up - nominal
    # logkdown = nominal - down
    # Leads to a sign flip wrt rabbit
    # Assyming the unc. are organized down,up,down,up,...
    symm_avg = 0.5 * (-values[:, ::2] + values[:, 1::2])

    if symm_type == "average":
        if len(labels) * 2 != values.shape[-1]:
            raise ValueError(
                f"Number of nuisances should be half the number of eigenvectors when using average symmetrization! Found {len(labels)} nuisances and {values.shape[-1]} symmetric variations. Please check the input matrix and labels."
            )

        values[:, : len(labels)] = symm_avg

        return matrix.iloc[:, : len(labels)]

    avg_idx = np.char.find(labels, "Avg") != -1
    symm_diff = 0.5 * np.sqrt(3) * (values[:, ::2] + values[:, 1::2])

    values[:, avg_idx] = symm_avg
    values[:, ~avg_idx] = symm_diff

    if np.count_nonzero(avg_idx) != symm_avg.shape[1]:
        raise ValueError(
            f"Found inconsistent number of Avg nuisances (Avg: {np.count_nonzero(avg_idx)}, Symm: {symm_avg.shape[1]}, Diff {symm_diff.shape[1]}) for quadratic symmetrization."
        )

    return matrix
