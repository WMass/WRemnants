import ROOT
import hist
import numpy as np
from utilities import boostHistHelpers as hh, common, logging
from wremnants import theory_tools, helicity_utils
from wremnants.helicity_utils import axis_helicity
from wremnants.datasets.datagroups import Datagroups
import re
import collections.abc

logger = logging.child_logger(__name__)

def syst_transform_map(base_hist, hist_name):
    pdfInfo = theory_tools.pdfMap
    pdfNames = [pdfInfo[k]["name"] for k in pdfInfo.keys()]

    def pdfUnc(h, pdfName, axis_name="pdfVar"):
        key =  list(pdfInfo.keys())[list(pdfNames).index(pdfName)]
        unc = pdfInfo[key]["combine"]
        scale = pdfInfo[key]["scale"] if "scale" in pdfInfo[key] else 1.
        return theory_tools.hessianPdfUnc(h, uncType=unc, scale=scale, axis_name=axis_name)

    def uncHist(unc):
        return unc if base_hist == "nominal" else f"{base_hist}_{unc}"

    transforms = {}
    transforms.update({pdf+"Up" : {"action" : lambda h,p=pdf: pdfUnc(h, p)[0] if "pdfVar" in h.axes.name else h} for pdf in pdfNames})
    transforms.update({pdf+"Down" : {"action" : lambda h,p=pdf: pdfUnc(h, p)[1] if "pdfVar" in h.axes.name else h} for pdf in pdfNames})
    transforms["scetlib_dyturboMSHT20Up"] = {"action" : lambda h: pdfUnc(h, "pdfMSHT20", "vars")[0], "procs" : common.vprocs_all}
    transforms["scetlib_dyturboMSHT20Down"] = {"action" : lambda h: pdfUnc(h, "pdfMSHT20", "vars")[1], "procs" : common.vprocs_all}
    transforms["scetlib_dyturboCT18ZUp"] = {"action" : lambda h: pdfUnc(h, "pdfCT18Z", "vars")[0], "procs" : common.vprocs_all}
    transforms["scetlib_dyturboCT18ZDown"] = {"action" : lambda h: pdfUnc(h, "pdfCT18Z", "vars")[1], "procs" : common.vprocs_all}
    transforms["scetlib_dyturboMSHT20an3loUp"] = {"action" : lambda h: pdfUnc(h, "pdfMSHT20", "vars")[0], "procs" : common.zprocs_all}
    transforms["scetlib_dyturboMSHT20an3loDown"] = {"action" : lambda h: pdfUnc(h, "pdfMSHT20", "vars")[1], "procs" : common.zprocs_all}
    transforms["ewUp"] = {"action" : lambda h,**args: h if "systIdx" not in h.axes.name else h[{"systIdx" : 0}]}
    transforms["ewDown"] = {"requiresNominal" : True, "action" : lambda h,**args: h if "systIdx" not in h.axes.name else hh.mirrorHist(h[{"systIdx" : 0}], **args)}
    transforms["muonScaleUp"] = {"action" : lambda h: h if "unc" not in h.axes.name else hh.rssHistsMid(h, "unc")[1]}
    transforms["muonScaleDown"] = {"action" : lambda h: h if "unc" not in h.axes.name else hh.rssHistsMid(h, "unc")[0]}
    transforms["muonScale3Up"] = {"action" : lambda h: h if "unc" not in h.axes.name else hh.rssHistsMid(h, "unc", 3.35)[1]}
    transforms["muonScale3Down"] = {"action" : lambda h: h if "unc" not in h.axes.name else hh.rssHistsMid(h, "unc", 3.35)[0]}
    transforms["muonResUp"] = {"requiresNominal" : True, "action" : lambda h,**args: h if "smearing_variation" not in h.axes.name else hh.rssHists(h, "smearing_variation", **args)[1]}
    transforms["muonResDown"] = {"requiresNominal" : True, "action" : lambda h,**args: h if "smearing_variation" not in h.axes.name else hh.rssHists(h, "smearing_variation", **args)[0]}

    s = hist.tag.Slicer()
    transforms.update({
        "QCDscale_muRmuFUp" : {
            "action" : lambda h: h if "muRfact" not in h.axes.name else h[{"muRfact" : 2.j, "muFfact" : 2.j, "ptVgen" : s[::hist.sum]}]},
        "QCDscale_muRmuFDown" : {
            "action" : lambda h: h if "muRfact" not in h.axes.name else h[{"muRfact" : 0.5j, "muFfact" : 0.5j, "ptVgen" : s[::hist.sum]}]},
        "QCDscale_muRUp" : {
            "action" : lambda h: h if "muRfact" not in h.axes.name else h[{"muRfact" : 2.j, "muFfact" : 1.j, "ptVgen" : s[::hist.sum]}]},
        "QCDscale_muRDown" : {
            "action" : lambda h: h if "muRfact" not in h.axes.name else h[{"muRfact" : 0.5j, "muFfact" : 1.j, "ptVgen" : s[::hist.sum]}]},
        "QCDscale_muFUp" : {
            "action" : lambda h: h if "muRfact" not in h.axes.name else h[{"muRfact" : 1.j, "muFfact" : 2.j, "ptVgen" : s[::hist.sum]}]},
        "QCDscale_muFDown" : {
            "action" : lambda h: h if "muRfact" not in h.axes.name else h[{"muRfact" : 1.j, "muFfact" : 0.5j, "ptVgen" : s[::hist.sum]}]},
        "QCDscale_cen" : {
            "action" : lambda h: h if "muRfact" not in h.axes.name else h[{"muRfact" : 1.j, "muFfact" : 1.j, "ptVgen" : s[::hist.sum]}]},
    })

    def scetlibIdx(h, i):
        return h if not ("vars" in h.axes.name and h.axes["vars"].size > i) else h[{"vars" : i}]

    def projAx(hname):
        return hname.split("-")

    resum_tnps = ['pdf0', 'gamma_cusp+1', 'gamma_mu_q+1', 'gamma_nu+1', 'h_qqV-0.5', 's+1', 'b_qqV+1', 'b_qqbarV+1', 'b_qqS+1', 'b_qqDS+1', 'b_qg+1']
    resum_tnpsXp1_up = ['pdf0', 'gamma_cusp1.', 'gamma_mu_q1.', 'gamma_nu1.', 's1.', 'b_qqV0.5', 'b_qqV0.5', 'b_qqbarV0.5', 'b_qqS0.5', 'b_qqDS0.5', 'b_qg0.5']
    resum_tnpsXp1_down = ['pdf0', 'gamma_cusp-1.', 'gamma_mu_q-1.', 'gamma_nu-1.', 's-1.', 'b_qqV-2.5', 'b_qqV-2.5', 'b_qqbarV-2.5', 'b_qqS-2.5', 'b_qqDS-2.5', 'b_qg-2.5']
    resum_tnpsXp0_up = ['pdf0', 'gamma_cusp1.', 'gamma_mu_q1.', 'gamma_nu1.', 's1.', 'b_qqV0.5', 'b_qqV0.5', 'b_qqbarV0.5', 'b_qqS0.5', 'b_qqDS0.5', 'b_qg0.5']
    resum_tnpsXp0_down = ['pdf0', 'gamma_cusp-1.', 'gamma_mu_q-1.', 'gamma_nu-1.', 's-1.', 'b_qqV-0.5', 'b_qqV-0.5', 'b_qqbarV-0.5', 'b_qqS-0.5', 'b_qqDS-0.5', 'b_qg-0.5']

    transforms.update({
        "resumFOScaleUp" : {
            "action" : lambda h: scetlibIdx(h, 2)},
        "resumFOScaleDown" : {
            "action" : lambda h: scetlibIdx(h, 1)},
        "resumLambdaDown" : {
            "action" : lambda h: scetlibIdx(h, 3)},
        "resumLambdaUp" : {
            "action" : lambda h: scetlibIdx(h, 4)},
        "resumTransitionUp" : {
            "action" : lambda h: hh.syst_min_or_max_env_hist(h, projAx(hist_name), "vars", 
                ["transition_points0.2_0.65_1.1", "transition_points0.4_0.55_0.7", 
                "transition_points0.2_0.45_0.7", "transition_points0.4_0.75_1.1", ],
                 no_flow=["ptVgen"], do_min=False)},
        "resumTransitionDown" : {
            "action" : lambda h: hh.syst_min_or_max_env_hist(h, projAx(hist_name), "vars", 
                ["transition_points0.2_0.65_1.1", "transition_points0.4_0.55_0.7", 
                "transition_points0.2_0.45_0.7", "transition_points0.4_0.75_1.1", ],
                 no_flow=["ptVgen"], do_min=True)},
       "resumTNPXp1Up" : {
           "action" : lambda h: h if "vars" not in h.axes.name else hh.rssHists(h[{"vars" : resum_tnpsXp0_up}], "vars")[0]
        },
       "resumTNPXp0Down" : {
           "action" : lambda h: h if "vars" not in h.axes.name else hh.rssHists(h[{"vars" : resum_tnpsXp0_down}], "vars")[1]
        },
       "resumTNPXp0Up" : {
           "action" : lambda h: h if "vars" not in h.axes.name else hh.rssHists(h[{"vars" : resum_tnpsXp1_up}], "vars")[0]
        },
       "resumTNPXp1Down" : {
           "action" : lambda h: h if "vars" not in h.axes.name else hh.rssHists(h[{"vars" : resum_tnpsXp1_down}], "vars")[1]
        },
       "resumTNPx5Up" : {
           "action" : lambda h: h if "vars" not in h.axes.name else hh.rssHists(h[{"vars" : resum_tnps}], "vars", scale=5)[0]
        },
       "resumTNPx5Down" : {
           "action" : lambda h: h if "vars" not in h.axes.name else hh.rssHists(h[{"vars" : resum_tnps}], "vars", scale=5)[1]
        },
       "resumTNPx12Up" : {
           "action" : lambda h: h if "vars" not in h.axes.name else hh.rssHists(h[{"vars" : resum_tnps}], "vars", scale=12)[0]
        },
       "resumTNPx12Down" : {
           "action" : lambda h: h if "vars" not in h.axes.name else hh.rssHists(h[{"vars" : resum_tnps}], "vars", scale=12)[1]
        },
       "resumScaleAllUp" : {
           "action" : lambda h: h if "vars" not in h.axes.name else hh.syst_min_or_max_env_hist(h, projAx(hist_name), "vars",
                 [x for x in h.axes["vars"] if any(re.match(y, x) for y in ["pdf0", "^nuB.*", "nuS.*", "^muB.*", "^muS.*", 'kappa.*', 'muf.*', ])],
                    do_min=False)},
       "resumScaleAllDown" : {
           "action" : lambda h: h if "vars" not in h.axes.name else hh.syst_min_or_max_env_hist(h, projAx(hist_name), "vars",
                 [x for x in h.axes["vars"] if any(re.match(y, x) for y in ["pdf0", "^nuB.*", "nuS.*", "^muB.*", "^muS.*", 'kappa.*', 'muf.*', ])],
                    do_min=True)},
       "resumScaleUp" : {
           "action" : lambda h: h if "vars" not in h.axes.name else hh.syst_min_or_max_env_hist(h, projAx(hist_name), "vars",
                 [x for x in h.axes["vars"] if any(re.match(y, x) for y in ["pdf0", "^nuB.*", "nuS.*", "^muB.*", "^muS.*"])],
                    do_min=False)},
       "resumScaleDown" : {
           "action" : lambda h: h if "vars" not in h.axes.name else hh.syst_min_or_max_env_hist(h, projAx(hist_name), "vars",
                 [x for x in h.axes["vars"] if any(re.match(y, x) for y in ["pdf0", "^nuB.*", "nuS.*", "^muB.*", "^muS.*"])],
                    do_min=True)},
       "resumNPUp" : {
            "action" : lambda h: hh.syst_min_or_max_env_hist(h, projAx(hist_name), "vars", 
                 ["c_nu-0.1-omega_nu0.5", "omega_nu0.5", "Lambda2-0.25", "Lambda20.25", "Lambda4.01", 
                     "Lambda4.16","Delta_Lambda2-0.02", "Delta_Lambda20.02",],
                 no_flow=["ptVgen"], do_min=False) if "vars" in h.axes.name else h},
        "resumNPDown" : {
            "action" : lambda h: hh.syst_min_or_max_env_hist(h, projAx(hist_name), "vars", 
                 ["c_nu-0.1-omega_nu0.5", "omega_nu0.5", "Lambda2-0.25", "Lambda20.25", "Lambda4.01", 
                     "Lambda4.16","Delta_Lambda2-0.02", "Delta_Lambda20.02",],
                 no_flow=["ptVgen"], do_min=True) if "vars" in h.axes.name else h},
       "resumNPOmegaUp" : {
            "action" : lambda h: hh.syst_min_or_max_env_hist(h, projAx(hist_name), "vars", 
                [x for x in h.axes["vars"] if re.match("^Omega-*\d+", x)],
                 do_min=False) if "vars" in h.axes.name else h},
        "resumNPOmegaDown" : {
            "action" : lambda h: hh.syst_min_or_max_env_hist(h, projAx(hist_name), "vars", 
                [x for x in h.axes["vars"] if re.match("^Omega-*\d+", x)],
                 do_min=True) if "vars" in h.axes.name else h},
       "resumNPomega_nuUp" : {
            "action" : lambda h: hh.syst_min_or_max_env_hist(h, projAx(hist_name), "vars", 
                [x for x in h.axes["vars"] if re.match("^omega_nu-*\d+", x)],
                 do_min=False) if "vars" in h.axes.name else h},
        "resumNPomega_nuDown" : {
            "action" : lambda h: hh.syst_min_or_max_env_hist(h, projAx(hist_name), "vars", 
                [x for x in h.axes["vars"] if re.match("^omega_nu-*\d+", x)],
                 do_min=True) if "vars" in h.axes.name else h},
       "resumNPc_nuUp" : {
            "action" : lambda h: hh.syst_min_or_max_env_hist(h, projAx(hist_name), "vars", 
                [x for x in h.axes["vars"] if re.match("^c_nu-*\d+", x)],
                 do_min=False) if "vars" in h.axes.name else h},
        "resumNPc_nuDown" : {
            "action" : lambda h: hh.syst_min_or_max_env_hist(h, projAx(hist_name), "vars", 
                [x for x in h.axes["vars"] if re.match("^c_nu-*\d+", x)],
                 do_min=True) if "vars" in h.axes.name else h},
        "resumScaleMax" : {
            "action" : lambda h: hh.syst_min_or_max_env_hist(h, projAx(hist_name), "vars", range(9,44), no_flow=["ptVgen"], do_min=False)},
        "resumScaleMin" : {
            "action" : lambda h: hh.syst_min_or_max_env_hist(h, projAx(hist_name), "vars", range(9,44), no_flow=["ptVgen"], do_min=True)},
    })
    for k in ['gamma_cusp+5', 'gamma_mu_q+5', 'gamma_nu+5', 's+5', 'b_qqV+5', 'b_qqbarV+5', 'b_qqS+5', 'b_qqDS+5', 'b_qg+5']:
        transforms[k.replace("+5", "-5")] = {"action" : lambda h,v=k: h if "vars" not in h.axes.name else hh.mirrorHist(h[{"vars" : v}], h[{"vars" : "pdf0"}])}
    transforms['h_qqV+2.0'] = {"action" : lambda h: h if "vars" not in h.axes.name else hh.mirrorHist(h[{"vars" : 'h_qqV-2.0'}], h[{"vars" : 'pdf0'}])}
    for k in ['gamma_cusp+1', 'gamma_mu_q+1', 'gamma_nu+1', 's+1', 'b_qqV+1', 'b_qqbarV+1', 'b_qqS+1', 'b_qqDS+1', 'b_qg+1']:
        transforms[k.replace("+1", "-1")] = {"action" : lambda h,v=k: h if "vars" not in h.axes.name else hh.mirrorHist(h[{"vars" : v}], h[{"vars" : "pdf0"}])}
    transforms['h_qqV+0.5'] = {"action" : lambda h: h if "vars" not in h.axes.name else hh.mirrorHist(h[{"vars" : 'h_qqV-0.5'}], h[{"vars" : 'pdf0'}])}

    return transforms


def gen_scale_helicity_hist_to_variations(scale_hist, gen_obs, sum_axes=[], pt_ax="ptVgen", gen_axes=["ptVgen", "chargeVgen", "helicity"], rebinPtV=None):
    scale_hist = hh.expand_hist_by_duplicate_axes(hist_in, gen_obs, [a+"Alt" for a in gen_obs], swap_axes=True)

    return scale_helicity_hist_to_variations(scale_hist, sum_axes, pt_ax, gen_axes, rebinPtV)


def scale_helicity_hist_to_variations(scale_hist, sum_axes=[], pt_ax="ptVgen", gen_axes=["ptVgen", "chargeVgen", "helicity"], rebinPtV=None):
    s = hist.tag.Slicer()
    axisNames = scale_hist.axes.name

    sum_expr = {axis : s[::hist.sum] for axis in sum_axes if axis in axisNames}
    scale_hist = scale_hist[sum_expr]
    axisNames = scale_hist.axes.name
    
    # select nominal QCD scales, but keep the sliced axis at size 1 for broadcasting
    nom_scale_hist = scale_hist[{"muRfact" : s[1.j:1.j+1], "muFfact" : s[1.j:1.j+1]}]
    # select nominal QCD scales and project down to nominal axes
    nom_sel = {"muRfact" : s[1.j], "muFfact" : s[1.j] }
    nom_sel.update({genAxis : s[::hist.sum] for genAxis in gen_axes if genAxis in axisNames})
    nom_hist = nom_scale_hist[nom_sel]
    
    hasHelicityAxis = "helicity" in axisNames
    hasPtAxis = pt_ax in axisNames

    if rebinPtV is not None and hasPtAxis:
        # Treat single bin array as a float
        array_rebin = isinstance(rebinPtV, collections.abc.Sequence) or type(rebinPtV) == np.ndarray
        if array_rebin and len(rebinPtV) == 1:
            rebinPtV = rebinPtV[0]
            array_rebin = False

        if array_rebin:
            scale_hist = hh.rebinHist(scale_hist, pt_ax, rebinPtV)
            nom_scale_hist = hh.rebinHist(nom_scale_hist, pt_ax, rebinPtV)
        else:
            scale_hist = scale_hist[{pt_ax : s[::hist.rebin(rebinPtV)]}]
            nom_scale_hist = nom_scale_hist[{pt_ax : s[::hist.rebin(rebinPtV)]}]

    # difference between a given scale and the nominal, plus the sum
    # this emulates the "weight if idx else nominal" logic and corresponds to the decorrelated
    # variations
    if scale_hist.name is None:
        out_name = "scale_helicity_variations" if hasHelicityAxis else "scale_vpt_variations" if hasPtAxis else "scale_vcharge_variations"
    else:
        out_name = scale_hist.name + "_variations"

    nom_axes = nom_hist.axes
    if nom_axes != scale_hist.axes[:len(nom_axes)]:
        raise ValueError("Cannot convert to variations histogram becuase the assumption that the order of the gen axes " \
                "and reco-like axes is respected does not hold! Gen axes must be trailing! "\
                f" Found nominal (reco-like) axes {nom_axes.name}, full axes {scale_hist.axes.name}")

    expd = scale_hist.ndim - nom_hist.ndim
    expandnom = np.expand_dims(nom_hist.values(flow=True), [-expd+i for i in range(expd)])
    systhist = scale_hist.values(flow=True) - nom_scale_hist.values(flow=True) + expandnom

    scale_variation_hist = hist.Hist(*scale_hist.axes, 
                                     name = out_name, data = systhist)

    return scale_variation_hist


def decorrelateByAxis(hvar, hnom, axisToDecorrName, newDecorrAxisName=None, axlim=[], rebin=[], absval=False):
    return decorrelateByAxes(hvar, hnom, axesToDecorrNames=[axisToDecorrName], newDecorrAxesNames=[newDecorrAxisName], axlim=[axlim], rebin=[rebin], absval=[absval])


def decorrelateByAxes(hvar, hnom, axesToDecorrNames, newDecorrAxesNames=[], axlim=[], rebin=[], absval=[]):

    commonMessage = f"Requested to decorrelate uncertainty in histogram {hvar.name} by {axesToDecorrNames} axes"
    if any(a not in hvar.axes.name for a in axesToDecorrNames):
        raise ValueError(f"{commonMessage}, but available axes for histogram are {hvar.axes.name}")

    if len(newDecorrAxesNames)==0:
        newDecorrAxesNames = [f"{n}_decorr" for n in axesToDecorrNames]
    elif len(axesToDecorrNames) != len(newDecorrAxesNames):
        raise ValueError(f"If newDecorrAxisName are specified, they must have the same length than axisToDecorrName, but they are {newDecorrAxisName} and {axisToDecorrName}.")

    # subtract nominal hist to get variation only
    hvar = hh.addHists(hvar, hnom, scale2=-1)
    # expand edges for variations on diagonal elements
    hvar = hh.expand_hist_by_duplicate_axes(hvar, axesToDecorrNames, newDecorrAxesNames, put_trailing=True)
    # add back nominal histogram while broadcasting
    hvar = hh.addHists(hvar, hnom)
    if len(axlim) or len(rebin):
        hvar = hh.rebinHistMultiAx(hvar, newDecorrAxesNames, rebin, axlim[::2], axlim[1::2])


    for ax, absval in zip(newDecorrAxesNames, absval):
        if absval:
            logger.info(f"Taking the absolute value of axis '{ax}'")
            hvar = hh.makeAbsHist(hvar, ax, rename=False)

    # if there is a mirror axis, put it at the end, since CardTool.py requires it like that
    if "mirror" in hvar.axes.name and hvar.axes.name.index("mirror") != len(hvar.shape)-1:
        sortedAxes = [n for n in hvar.axes.name if n != "mirror"]
        sortedAxes.append("mirror")
        hvar = hvar.project(*sortedAxes)

    return hvar


def make_fakerate_variation(href, fakerate_axes, fakerate_axes_syst, variation_fakerate=0.5, flow=False):
    # 1) calculate fakerate in bins of fakerate axes
    # TODO: update
    nameMT, failMT, passMT = sel.get_mt_selection(href)
    hist_failMT_failIso = href[{**common.failIso, nameMT: failMT}].project(*fakerate_axes)
    hist_failMT_passIso = href[{**common.passIso, nameMT: failMT}].project(*fakerate_axes)
    fr = hh.divideHists(hist_failMT_failIso, hist_failMT_failIso+hist_failMT_passIso).values(flow=flow)

    # 2) add a variation
    diff = np.minimum(fr, 1 - fr)
    frUp = fr + variation_fakerate * diff
    hRateFail = hist.Hist(*[href.axes[n] for n in fakerate_axes], storage=hist.storage.Double(), data=frUp)
    hRatePass = hist.Hist(*[href.axes[n] for n in fakerate_axes], storage=hist.storage.Double(), data=1-frUp)

    # 3) apply the varied fakerate to the original histogram to get the veried bin contents, subtract the nominal histogram to only have the difference
    hvar = hist.Hist(*[a for a in href.axes], storage=hist.storage.Double(), data=-1*href.values(flow=flow))
    s = hist.tag.Slicer()

    # fail Iso, fail MT
    slices = [failMT if n==nameMT else 0 if n==common.passIsoName else slice(None) for n in hvar.axes.name]
    hvar.values(flow=flow)[*slices] += hh.multiplyHists(href[{common.passIsoName: s[::hist.sum], nameMT: failMT}], hRateFail).values(flow=flow)
    # pass Iso, fail MT
    slices = [failMT if n==nameMT else 1 if n==common.passIsoName else slice(None) for n in hvar.axes.name]
    hvar.values(flow=flow)[*slices] += hh.multiplyHists(href[{common.passIsoName: s[::hist.sum], nameMT: failMT}], hRatePass).values(flow=flow)
    # fail Iso, pass MT
    slices = [passMT if n==nameMT else 0 if n==common.passIsoName else slice(None) for n in hvar.axes.name]
    hvar.values(flow=flow)[*slices] += hh.multiplyHists(href[{common.passIsoName: s[::hist.sum], nameMT: passMT}], hRateFail).values(flow=flow)
    # pass Iso, pass MT
    slices = [passMT if n==nameMT else 1 if n==common.passIsoName else slice(None) for n in hvar.axes.name]
    hvar.values(flow=flow)[*slices] += hh.multiplyHists(href[{common.passIsoName: s[::hist.sum], nameMT: passMT}], hRatePass).values(flow=flow)

    # 4) expand the variations to be used as systematic axes
    hsyst = hh.expand_hist_by_duplicate_axes(hvar, fakerate_axes, fakerate_axes_syst)

    # 5) add back to the nominal histogram and broadcase the nominal histogram
    return hh.addHists(href, hsyst)


def gen_hist_to_variations(hist_in, gen_obs, gen_axes=["ptVgen", "chargeVgen", "helicity"], sum_axes=[], rebin_axes=[], rebin_edges=[]):
    for obs in gen_obs:
        hist_in = hh.expand_hist_by_duplicate_axis(hist_in, obs, obs+"Alt", swap_axes=True)

    return hist_to_variations(hist_in, gen_axes, sum_axes, rebin_axes, rebin_edges)


def hist_to_variations(hist_in, gen_axes = [], sum_axes = [], rebin_axes=[], rebin_edges=[]):

    if hist_in.name is None:
        out_name = "hist_variations"
    else:
        out_name = hist_in.name + "_variations"

    s = hist.tag.Slicer()

    #do rebinning
    for rebin_axis, edges in zip(rebin_axes, rebin_edges):
        hist_in = hh.rebinHist(hist_in, rebin_axis, edges)

    axisNames = hist_in.axes.name
    sum_expr = {axis : s[::hist.sum] for axis in sum_axes if axis in axisNames}
    hist_in = hist_in[sum_expr]
    axisNames = hist_in.axes.name

    gen_sum_expr = {n : s[::hist.sum] for n in gen_axes if n in axisNames}
    if len(gen_sum_expr) == 0:
        # all the axes have already been projected out, nothing else to do
        return hist_in

    nom_hist = hist_in[{"vars" : 0}]
    nom_hist_sum = nom_hist[gen_sum_expr]

    # slices to broadcast nom_hist and nom_hist_sum to hist_in shape
    slices_nom = [slice(None) if n in nom_hist.axes.name else np.newaxis for n in hist_in.axes.name]
    slices_nom_sum = [slice(None) if n in nom_hist_sum.axes.name else np.newaxis for n in hist_in.axes.name]
    variation_data = hist_in.view(flow=True) - nom_hist.view(flow=True)[*slices_nom] + nom_hist_sum.view(flow=True)[*slices_nom_sum]

    variation_hist = hist.Hist(*hist_in.axes, storage = hist_in._storage_type(),
                                     name = out_name, data = variation_data)

    return variation_hist

def muRmuFFullyCorrelatedVariations(hist_in):

    if hist_in.name is None:
        out_name = "hist_variations"
    else:
        out_name = hist_in.name + "_variations"

    variation_hist = hist.Hist(*hist_in.axes, name = out_name, storage = hist.storage.Double())
    variation_hist = variation_hist[...,0:1,:]

    for i in range(len(hist_in.axes[len(hist_in.axes)-2])):
        variation_hist.values(flow=True)[...,0,0] = variation_hist.values(flow=True)[...,0,0] + hist_in.values(flow=True)[...,i,0] - (hist_in.values(flow=True)[...,0,0]+hist_in.values(flow=True)[...,0,1])/2 #this average which is subtracted is the nominal histogram (by construction, it's directly related to how the weights are defined)
        variation_hist.values(flow=True)[...,0,1] = variation_hist.values(flow=True)[...,0,1] + hist_in.values(flow=True)[...,i,1] - (hist_in.values(flow=True)[...,0,0]+hist_in.values(flow=True)[...,0,1])/2
    variation_hist.values(flow=True)[...,0,0] = variation_hist.values(flow=True)[...,0,0] + (hist_in.values(flow=True)[...,0,0]+hist_in.values(flow=True)[...,0,1])/2
    variation_hist.values(flow=True)[...,0,1] = variation_hist.values(flow=True)[...,0,1] + (hist_in.values(flow=True)[...,0,0]+hist_in.values(flow=True)[...,0,1])/2

    return variation_hist


def uncertainty_hist_from_envelope(h, proj_ax, entries):
    hdown = hh.syst_min_or_max_env_hist(h, proj_ax, "vars", entries, no_flow=["ptVgen"], do_min=True)
    hup = hh.syst_min_or_max_env_hist(h, proj_ax, "vars", entries, no_flow=["ptVgen"], do_min=False)
    hnew = hist.Hist(*h.axes[:-1], common.down_up_axis, storage=h._storage_type())
    hnew[...,0] = hdown.view(flow=True)
    hnew[...,1] = hup.view(flow=True)
    return hnew


def add_syst_hist(results, df, name, axes, cols, tensor_name=None, tensor_axes=[], addhelicity=False, nhelicity=6, storage_type=hist.storage.Double()):
    """
    Add hist to results list

    Args:
        tensor_name (str): name of tensor defined as column in df (scalar weight or multi dimensional tensor; none for unweighted)
        tensor_axes (list): list of axes corresponding to the tensor
        addhelicity (bool): Add the helicity axis to the histogram
        nhelicity (int): Take first nhelicity bins of helicity tensor
    """
    if not isinstance(tensor_axes, (list, tuple)):
        tensor_axes = [tensor_axes]
    if addhelicity:
        if len(tensor_axes) == 0:
            # rank 0 tensor
            if tensor_name is None:
                # unity
                results.append(df.HistoBoost(name, axes, [*cols, "helWeight_tensor"], tensor_axes=[helicity_utils.axis_helicity_multidim], storage=storage_type))
            else:
                # scalar weight
                df = df.Define(f'{tensor_name}_helicity', f"auto res = helWeight_tensor; res = {tensor_name}*res; return res;")
                results.append(df.HistoBoost(name, axes, [*cols, f'{tensor_name}_helicity'], tensor_axes=[helicity_utils.axis_helicity_multidim], storage=storage_type))
        else:
            helper_helicity, tensor_axes_helicity = helicity_utils.make_helper_helicity(tensor_axes, nhelicity)
            df = df.Define(f'{tensor_name}_helicity', helper_helicity, [tensor_name, 'helWeight_tensor'])
            results.append(df.HistoBoost(name, axes, [*cols, f"{tensor_name}_helicity"], tensor_axes=tensor_axes_helicity, storage=storage_type))
    else:
        if len(tensor_axes) == 0:
            if tensor_name is None:
                results.append(df.HistoBoost(name, axes, cols, storage=storage_type))
            else:
                results.append(df.HistoBoost(name, axes, [*cols, tensor_name], storage=storage_type))
        else:
            results.append(df.HistoBoost(name, axes, [*cols, tensor_name], tensor_axes=tensor_axes, storage=storage_type))


def define_mass_width_sin2theta_weights(df, proc):

    # TODO can these be parsed more automatically?
    if proc in common.zprocs_all:
        m0 = 91.1876
        gamma0 = 2.4941343245745466
        massvals = [91.0876, 91.0976, 91.1076, 91.1176, 91.1276, 91.1376, 91.1476, 91.1576, 91.1676, 91.1776, 91.1876, 91.1976, 91.2076, 91.2176, 91.2276, 91.2376, 91.2476, 91.2576, 91.2676, 91.2776, 91.2876, 91.1855, 91.1897]
        widthvals = [2.49333, 2.49493, 2.4929, 2.4952, 2.4975]
        sin2thetavals = [0.23151, 0.23154, 0.23157, 0.2230, 0.2300, 0.2305, 0.2310, 0.2315, 0.2320, 0.2325, 0.2330]
    else:
        m0 = 80.379
        gamma0 = 2.0911383956149385
        massvals = [80.279, 80.289, 80.299, 80.309, 80.319, 80.329, 80.339, 80.349, 80.359, 80.369, 80.379, 80.389, 80.399, 80.409, 80.419, 80.429, 80.439, 80.449, 80.459, 80.469, 80.479]
        widthvals = [2.09053, 2.09173, 2.043, 2.085, 2.127]
        sin2thetavals = []

    nweights_mass = len(massvals)
    nweights_width = len(widthvals)
    nweights_sin2theta = len(sin2thetavals)

    if not "massWeight_tensor" in df.GetColumnNames():
        # from -100 to 100 MeV with 10 MeV increment
        if "MEParamWeight" in df.GetColumnNames():
            df = df.Alias("massWeight_col", "MEParamWeight")
        else:
            df = df.Define("massWeight_col", f"wrem::slice_vec(LHEReweightingWeight,0, {nweights_mass})")
        df = df.Define("massWeight_tensor", f"wrem::vec_to_tensor_t<double, {nweights_mass}>(massWeight_col)")
        df = df.Define("massWeight_tensor_wnom", "auto res = massWeight_tensor; res = nominal_weight*res; return res;")

        # compute modified weights which remove the width variation from the mass weights by using the width weights to compensate
        # TODO we should pass this in from outside so that we can re-use it, but it's lightweight enough that it shouldn't matter much
        helper_mass = ROOT.wrem.MassWeightHelper[nweights_mass](m0, gamma0, massvals, widthvals)

        if "MEParamWeightAltSet1" in df.GetColumnNames():
            df = df.Alias("widthWeight_col", "MEParamWeightAltSet1")
        else:
            df = df.Define("widthWeight_col", f"wrem::slice_vec(LHEReweightingWeight,{nweights_mass}, {nweights_mass+nweights_width})")

        df = df.Define("massWeight_widthdecor_tensor", helper_mass, ["massWeight_col", "widthWeight_col"])
        df = df.Define("massWeight_widthdecor_tensor_wnom", "auto res = massWeight_widthdecor_tensor; res = nominal_weight*res; return res;")

        df = df.Define("widthWeight_tensor", f"wrem::vec_to_tensor_t<double, {nweights_width}>(widthWeight_col)")
        df = df.Define("widthWeight_tensor_wnom", "auto res = widthWeight_tensor; res = nominal_weight*res; return res;")


        if proc in common.zprocs_all:
            if "MEParamWeightAltSet4" in df.GetColumnNames():
                df = df.Alias("sin2thetaWeight_col", "MEParamWeightAltSet4")
            else:
                logger.warning("sin2theta weights in new format to be defined")
                df = df.Define("sin2thetaWeight_col", f"wrem::slice_vec(LHEReweightingWeight,{nweights_mass+nweights_width}, {nweights_mass+nweights_width+nweights_sin2theta})")
            df = df.Define("sin2thetaWeight_tensor", f"wrem::vec_to_tensor_t<double, {nweights_sin2theta}>(sin2thetaWeight_col)")
            df = df.Define("sin2thetaWeight_tensor_wnom", "auto res = sin2thetaWeight_tensor; res = nominal_weight*res; return res;")

    return df

def add_massweights_hist(results, df, axes, cols, base_name="nominal", proc="", **kwargs):
    name = Datagroups.histName(base_name, syst="massWeight"+(proc[0] if len(proc) else proc))
    name_widthdecor = Datagroups.histName(base_name, syst="massWeight_widthdecor"+(proc[0] if len(proc) else proc))
    mass_axis = hist.axis.StrCategory(massWeightNames(proc=proc), name="massShift")
    add_syst_hist(results, df, name, axes, cols, "massWeight_tensor_wnom", mass_axis, **kwargs)
    add_syst_hist(results, df, name_widthdecor, axes, cols, "massWeight_widthdecor_tensor_wnom", mass_axis, **kwargs)


def massWeightNames(matches=None, proc="", exclude=[]):
    if isinstance(exclude, (int, float)):
        exclude = [exclude, ]
    central=10
    nweights=21
    names = [f"massShift{proc[0] if len(proc) else proc}{int(abs(central-i)*10)}MeV{'' if i == central else ('Down' if i < central else 'Up')}" for i in range(nweights) if int(abs(central-i)*10) not in exclude]
    if proc and (proc in common.zprocs_all or proc=="Z") and 2.1 not in exclude:
        # This is the PDG uncertainty (turned off for now since it doesn't seem to have been read into the nano)
        names.extend(["massShiftZ2p1MeVDown", "massShiftZ2p1MeVUp"])

    # If name is "" it won't be stored
    return [x if not matches or any(y in x for y in matches) else "" for x in names]



def add_widthweights_hist(results, df, axes, cols, base_name="nominal", proc="", **kwargs):
    name = Datagroups.histName(base_name, syst="widthWeight"+(proc[0] if len(proc) else proc))
    axis_width = hist.axis.StrCategory(widthWeightNames(proc=proc), name="width")
    add_syst_hist(results, df, name, axes, cols, "widthWeight_tensor_wnom", axis_width, **kwargs)


def widthWeightNames(matches=None, proc=""):
    central=3
    if proc[0] == "Z":
        widths=(2.49333, 2.49493, 2.4929, 2.4952, 2.4975)
    elif proc[0] == "W":
        widths=(2.09053, 2.09173, 2.043, 2.085, 2.127) 
    else:
        raise RuntimeError(f"No width found for process {proc}")
    # 0 and 1 are Up, Down from mass uncertainty EW fit (already accounted for in mass variations)
    # 2, 3, and 4 are PDG width Down, Central, Up
    names = [f"width{proc[0]}{str(width).replace('.','p')}GeV" for width in widths]

    return [x if not matches or any(y in x for y in matches) else "" for x in names]



def add_sin2thetaweights_hist(results, df, axes, cols, base_name="nominal", proc="", **kwargs):
    name = Datagroups.histName(base_name, syst="sin2thetaWeight"+(proc[0] if len(proc) else proc))
    axis_sin2theta = hist.axis.StrCategory(sin2thetaWeightNames(proc=proc), name="sin2theta")
    add_syst_hist(results, df, name, axes, cols, "sin2thetaWeight_tensor_wnom", axis_sin2theta, **kwargs)


def sin2thetaWeightNames(matches=None, proc=""):
    if proc[0] != "Z":
        raise RuntimeError("sin2theta weights are only defined for Z")

    sin2thetas = (0.23151, 0.23154, 0.23157, 0.2230, 0.2300, 0.2305, 0.2310, 0.2315, 0.2320, 0.2325, 0.2330)

    # 1 is the central value
    # 0 and 2 are Down, Up from uncertainty in EW fit
    names = [f"sin2theta{proc[0]}{str(sin2theta).replace('.','p')}" for sin2theta in sin2thetas]

    return [x if not matches or any(y in x for y in matches) else "" for x in names]


# weak weights from Powheg EW NanoLHE files
def define_weak_weights(df, proc):
    if not 'Zmumu_powheg-weak' in proc:
        logger.debug("weakWeight_tensor only implemented for Zmumu_powheg-weak samples.")
        return df
    if "weakWeight_tensor" in df.GetColumnNames():
        logger.debug("weakWeight_tensor already defined, do nothing here.")
        return df
    nweights = 24
    df = df.Define("weakWeight_tensor", f"wrem::vec_to_tensor_t<double, {nweights}>(LHEReweightingWeight)")
    df = df.Define("weakWeight_tensor_wnom", "auto res = weakWeight_tensor; res = LHEWeight_originalXWGTUP*res; return res;")

    # note that makeHelicityMomentPdfTensor is actually generic for any variation along one axis
    # and not specific to PDFs
    var_helper = ROOT.wrem.makeHelicityMomentPdfTensor[nweights]()
    df = df.Define("LHEWeight_originalXWGTUP_D", "static_cast<double>(LHEWeight_originalXWGTUP)")
    df = df.Define("weakWeight_tensor_helicity", var_helper, ["csSineCosThetaPhilhe", "weakWeight_tensor", "LHEWeight_originalXWGTUP_D"])

    return df


def add_weakweights_hist(results, df, axes, cols, base_name="nominal", proc="", **kwargs):
    name = Datagroups.histName(base_name, syst="weakWeight"+(proc[0] if len(proc) else proc))
    tensor_axis = hist.axis.StrCategory(weakWeightNames(proc=proc), name="weak")
    add_syst_hist(results, df, name, axes, cols, "weakWeight_tensor_wnom", tensor_axis, **kwargs)


def weakWeightNames(matches=None, proc=""):
    names = [
        # no_ew=1d0
        'weak_no_ew',
        # no_ew=0d0, weak-only=1d0
        'weak_no_ho',
        # no_ew=0d0, weak-only=1d0, ew_ho=1d0
        'weak_default',
        # no_ew=0d0, weak-only=1d0, ew_ho=1d0, PS_scheme=1d0
        'weak_ps',
        # no_ew=0d0, weak-only=1d0, ew_ho=1d0, Tmass=170.69d0
        'weak_mt_dn',
        # no_ew=0d0, weak-only=1d0, ew_ho=1d0, Tmass=174.69d0
        'weak_mt_up',
        # no_ew=0d0, weak-only=1d0, ew_ho=1d0, Zmass=91.1855d0
        'weak_mz_dn',
        # no_ew=0d0, weak-only=1d0, ew_ho=1d0, Zmass=91.1897d0
        'weak_mz_up',
        # no_ew=0d0, weak-only=1d0, ew_ho=1d0, gmu=1.1663782d-5
        'weak_gmu_dn',
        # no_ew=0d0, weak-only=1d0, ew_ho=1d0, gmu=1.1663793d-5
        'weak_gmu_up',
        # no_ew=0d0, weak-only=1d0, ew_ho=1d0, scheme=1d0, alphaem_z=0.0077561467d0
        'weak_aem',
        # no_ew=0d0, weak-only=1d0, ew_ho=1d0, FS_scheme=1d0
        'weak_fs',
        # no_ew=0d0, weak-only=1d0, ew_ho=1d0, Hmass=124.25d0
        'weak_mh_dn',
        # no_ew=0d0, weak-only=1d0, ew_ho=1d0, Hmass=126.25d0
        'weak_mh_up',
        # no_ew=0d0, weak-only=1d0, ew_ho=1d0, use-s2effin=0.23125d0
        'weak_s2eff_0p23125',
        # no_ew=0d0, weak-only=1d0, ew_ho=1d0, use-s2effin=0.23105d0
        'weak_s2eff_0p23105',
        # no_ew=0d0, weak-only=1d0, ew_ho=1d0, use-s2effin=0.22155d0
        'weak_s2eff_0p22155',
        # no_ew=0d0, weak-only=1d0, ew_ho=1d0, use-s2effin=0.23185d0
        'weak_s2eff_0p23185',
        # no_ew=0d0, weak-only=1d0, ew_ho=1d0, use-s2effin=0.23205d0
        'weak_s2eff_0p23205',
        # no_ew=0d0, weak-only=1d0, ew_ho=1d0, use-s2effin=0.23255d0
        'weak_s2eff_0p23255',
        # no_ew=0d0, weak-only=1d0, ew_ho=1d0, use-s2effin=0.23355d0
        'weak_s2eff_0p23355',
        # no_ew=0d0, weak-only=1d0, ew_ho=1d0, use-s2effin=0.23455d0
        'weak_s2eff_0p23455',
        # no_ew=0d0, weak-only=1d0, ew_ho=1d0, use-s2effin=0.22955d0
        'weak_s2eff_0p22955',
        # no_ew=0d0, weak-only=1d0, ew_ho=1d0, use-s2effin=0.22655d0
        'weak_s2eff_0p22655',
    ]

    return [x if not matches or any(y in x for y in matches) else "" for x in names]


def add_pdf_hists(results, df, dataset, axes, cols, pdfs, base_name="nominal", propagateToHelicity=False, storage_type=hist.storage.Double(), **kwargs):
    # Remove duplicates but preserve the order of the first set
    for pdf in pdfs:
        try:
            pdfInfo = theory_tools.pdf_info_map(dataset, pdf)
        except ValueError as e:
            logger.info(e)
            continue

        pdfName = pdfInfo["name"]
        tensorName = f"{pdfName}Weights_tensor"
        tensorASName = f"{pdfName}ASWeights_tensor"
        npdf=pdfInfo["entries"]
        pdfHistName = Datagroups.histName(base_name, syst=pdfName)
        names = getattr(theory_tools, f"pdfNames{'Sym' if pdfInfo['combine'] == 'symHessian' else 'Asym'}Hessian")(pdfInfo["entries"], pdfName)
        pdf_ax = hist.axis.StrCategory(names, name="pdfVar")
        if tensorName not in df.GetColumnNames():
            logger.warning(f"PDF {pdf} was not found for sample {dataset}. Skipping uncertainty hist!")
            continue
        add_syst_hist(results, df, pdfHistName, axes, cols, tensorName, pdf_ax, storage_type=storage_type, **kwargs)

        if "alphasRange" in pdfInfo:
            asr = pdfInfo["alphasRange"] 
            alphaSHistName = Datagroups.histName(base_name, syst=f"{pdfName}alphaS{asr}")
            as_ax = hist.axis.StrCategory(["as0118"]+(["as0117", "as0119"] if asr == "001" else ["as0116", "as0120"]), name="alphasVar")
            add_syst_hist(results, df, alphaSHistName, axes, cols, tensorASName, as_ax, storage_type=storage_type, **kwargs)

        if propagateToHelicity:
            pdfhelper = ROOT.wrem.makeHelicityMomentPdfTensor[npdf]()
            df = df.Define(f"helicity_helicity_xsecs_{tensorName}_tensor", pdfhelper, ["csSineCosThetaPhigen", f"{tensorName}", "unity"])
            alphahelper = ROOT.wrem.makeHelicityMomentPdfTensor[3]()
            df = df.Define(f"helicity_helicity_xsecs_{tensorASName}_tensor", alphahelper, ["csSineCosThetaPhigen", f"{tensorASName}", "unity"])
            pdfHist_hel = df.HistoBoost(f"helicity_{pdfHistName}", axes, [*cols, f"helicity_helicity_xsecs_{tensorName}_tensor"], tensor_axes=[axis_helicity,pdf_ax], storage=storage_type)
            alphaSHist_hel = df.HistoBoost(f"helicity_{alphaSHistName}", axes, [*cols, f"helicity_helicity_xsecs_{tensorASName}_tensor"], tensor_axes=[axis_helicity,as_ax], storage=storage_type)
            results.extend([pdfHist_hel, alphaSHist_hel])

    return df


def add_qcdScale_hist(results, df, axes, cols, base_name="nominal", **kwargs):
    name = Datagroups.histName(base_name, syst="qcdScale")
    add_syst_hist(results, df, name, axes, cols, "scaleWeights_tensor_wnom", theory_tools.scale_tensor_axes, **kwargs)


def add_qcdScaleByHelicityUnc_hist(results, df, helper, axes, cols, base_name="nominal", **kwargs):
    name = Datagroups.histName(base_name, syst="qcdScaleByHelicity")
    tensorName = "helicityWeight_tensor"
    df = df.Define(tensorName, helper, ["massVgen", "absYVgen", "ptVgen", "chargeVgen", "csSineCosThetaPhigen", "nominal_weight"])
    add_syst_hist(results, df, name, axes, cols, tensorName, helper.tensor_axes, **kwargs)


def add_QCDbkg_jetPt_hist(results, df, axes, cols, base_name="nominal", jet_pt=30, **kwargs):
    # branching the rdataframe to add special filter, no need to return dQCDbkGVar
    name = Datagroups.histName(base_name, syst=f"qcdJetPt{str(jet_pt)}")
    dQCDbkGVar = df.Define(f"goodCleanJetsPt{jet_pt}", f"goodCleanJetsNoPt && Jet_pt > {jet_pt}")
    dQCDbkGVar = dQCDbkGVar.Filter(f"passMT || Sum(goodCleanJetsPt{jet_pt})>=1")
    add_syst_hist(results, dQCDbkGVar, name, axes, cols, "nominal_weight", **kwargs)


def add_luminosity_unc_hists(results, df, args, axes, cols, base_name="nominal", **kwargs):
    name = Datagroups.histName(base_name, syst="luminosity")
    df = df.Define("luminosityScaling", f"wrem::constantScaling(nominal_weight, {args.lumiUncertainty})")
    add_syst_hist(results, df, name, axes, cols, "luminosityScaling", common.down_up_axis, **kwargs)

    return df


def add_theory_corr_hists(results, df, axes, cols, helpers, generators, modify_central_weight, isW, base_name="nominal", **kwargs):
   
    for i, generator in enumerate(generators):
        if generator not in helpers:
            continue

        logger.debug(f"Now at generator {i}: {generator}")
        
        if i == 0 and modify_central_weight:
            add_syst_hist(results, df, f"{base_name}_uncorr", axes, cols, "nominal_weight_uncorr", **kwargs)
            if base_name == "nominal":
                add_syst_hist(results, df, f"weight_uncorr", [hist.axis.Regular(100, -2, 2)], ["nominal_weight_uncorr"], **kwargs)

        var_axis = helpers[generator].tensor_axes[-1]

        name = Datagroups.histName(base_name, syst=f"{generator}Corr")
        weight_tensor_name = f"{generator}Weight_tensor"
        add_syst_hist(results, df, name, axes, cols, weight_tensor_name, var_axis, **kwargs)

        def is_flavor_dependent_np(var_label):
            return var_label.startswith("Omega") \
                    or var_label.startswith("Delta_Omega") \
                    or var_label.startswith("Lambda2") \
                    or var_label.startswith("Delta_Lambda2") \
                    or var_label.startswith("Lambda4")

        # special treatment for Lambda2/Omega since they need to be decorrelated in charge and possibly rapidity
        if isinstance(var_axis, hist.axis.StrCategory) and any(is_flavor_dependent_np(var_label) for var_label in var_axis):
            omegaidxs = [var_axis.index(var_label) for var_label in var_axis if is_flavor_dependent_np(var_label)]

            # include nominal as well
            omegaidxs = [0] + omegaidxs

            tensor_name = f"{generator}FlavDepNP"
            if tensor_name not in df.GetColumnNames():
                np_idx_helper = ROOT.wrem.index_taker[df.GetColumnType(weight_tensor_name), len(omegaidxs)](omegaidxs)

                df = df.Define(tensor_name, np_idx_helper, [weight_tensor_name])

            axis_FlavDepNP = hist.axis.StrCategory([var_axis[idx] for idx in omegaidxs], name = var_axis.name)

            if isW:
                axis_chargegen = hist.axis.Regular(2, -2, 2, name="chargeVgenNP", underflow=False, overflow=False)
            else:
                axis_chargegen = hist.axis.Integer(0, 1, name="chargeVgenNP", underflow=False, overflow=False)

            axes_FlavDepNP = [*axes, theory_tools.axis_absYVgen, axis_chargegen]
            cols_FlavDepNP = [*cols, "absYVgen", "chargeVgen"]
            name = Datagroups.histName(base_name, syst=tensor_name)
            add_syst_hist(results, df, name, axes_FlavDepNP, cols_FlavDepNP, tensor_name, axis_FlavDepNP, **kwargs)

        def is_pt_dependent_scale(var_label):
            return var_label.startswith("renorm_fact_resum_transition_scale_envelope") \
                    or var_label.startswith("renorm_fact_resum_scale_envelope")

        # special treatment for envelope of scale variations since they need to be decorrelated in pt
        if isinstance(var_axis, hist.axis.StrCategory) and any(is_pt_dependent_scale(var_label) for var_label in var_axis):

            scaleidxs = [var_axis.index(var_label) for var_label in var_axis if is_pt_dependent_scale(var_label)]

            # include nominal as well
            scaleidxs = [0] + scaleidxs

            tensor_name = f"{generator}PtDepScales"
            if tensor_name not in df.GetColumnNames():
                scale_idx_helper = ROOT.wrem.index_taker[df.GetColumnType(weight_tensor_name), len(scaleidxs)](scaleidxs)

                df = df.Define(tensor_name, scale_idx_helper, [weight_tensor_name])

            axis_PtDepScales = hist.axis.StrCategory([var_axis[idx] for idx in scaleidxs], name = var_axis.name)
            
            axes_PtDepScales = axes[:]
            cols_PtDepScales = cols[:]
            if "ptVgen" not in cols:
                axes_PtDepScales += [hist.axis.Variable(common.ptV_binning, name = "ptVgen", underflow=False)]
                cols_PtDepScales += ["ptVgen"]
            name = Datagroups.histName(base_name, syst=tensor_name)
            add_syst_hist(results, df, name, axes_PtDepScales, cols_PtDepScales, tensor_name, axis_PtDepScales, **kwargs)


def add_muon_efficiency_unc_hists(results, df, helper_stat, helper_syst, axes, cols, base_name="nominal", 
                                  what_analysis=ROOT.wrem.AnalysisType.Wmass, singleMuonCollection="goodMuons", customHistNameTag="", smooth3D=False, **kwargs
    ):

    if what_analysis == ROOT.wrem.AnalysisType.Wmass:
        muon_columns_stat = [f"{singleMuonCollection}_{v}" for v in ["pt0", "eta0", "uT0", "charge0"]]
        muon_columns_syst = [f"{singleMuonCollection}_{v}" for v in ["pt0", "eta0", "SApt0", "SAeta0", "uT0", "charge0", "passIso0"]]
    else:
        muvars_stat = ["pt0", "eta0", "uT0", "charge0"] # passIso0 required only for iso stat variations, added later
        muon_columns_stat_trig    = [f"trigMuons_{v}" for v in muvars_stat]
        muon_columns_stat_nonTrig = [f"nonTrigMuons_{v}" for v in muvars_stat]

        muvars_syst = ["pt0", "eta0", "SApt0", "SAeta0", "uT0", "charge0", "passIso0"]
        muon_columns_syst_trig    = [f"trigMuons_{v}" for v in muvars_syst]
        muon_columns_syst_nonTrig = [f"nonTrigMuons_{v}" for v in muvars_syst]

        # muon_columns_stat in the following does not include passIso yet, added later for iso helper
        if what_analysis == ROOT.wrem.AnalysisType.Wlike:
            muon_columns_stat = [*muon_columns_stat_trig, *muon_columns_stat_nonTrig]
            muon_columns_syst = [*muon_columns_syst_trig, *muon_columns_syst_nonTrig]
        elif what_analysis == ROOT.wrem.AnalysisType.Dilepton:
            muon_columns_stat = [*muon_columns_stat_trig, "trigMuons_passTrigger0", *muon_columns_stat_nonTrig, "nonTrigMuons_passTrigger0"]
            muon_columns_syst = [*muon_columns_syst_trig, "trigMuons_passTrigger0", *muon_columns_syst_nonTrig, "nonTrigMuons_passTrigger0"]
        else:
            raise NotImplementedError(f"add_muon_efficiency_unc_hists: analysis {what_analysis} not implemented.")            

    if not smooth3D:
        # will use different helpers and member functions
        muon_columns_stat = [x for x in muon_columns_stat if "_uT0" not in x]
        muon_columns_syst = [x for x in muon_columns_syst if "_uT0" not in x]

    # change variables for tracking, to use standalone variables
    muon_columns_stat_tracking = [x.replace("_pt0", "_SApt0").replace("_eta0", "_SAeta0") for x in muon_columns_stat]
        
    for key,helper in helper_stat.items():
        if "tracking" in key:
            muon_columns_stat_step = muon_columns_stat_tracking
        elif "iso" in key:
            if what_analysis == ROOT.wrem.AnalysisType.Wmass:
                 # iso variable called passIso rather than goodMuons_passIso0 in W histmaker
                muon_columns_stat_step = [*muon_columns_stat, f"{singleMuonCollection}_passIso0"]
            elif what_analysis == ROOT.wrem.AnalysisType.Wlike:
                muon_columns_stat_step = [*muon_columns_stat_trig, "trigMuons_passIso0",
                                          *muon_columns_stat_nonTrig, "nonTrigMuons_passIso0"]
            elif what_analysis == ROOT.wrem.AnalysisType.Dilepton:
                muon_columns_stat_step = [*muon_columns_stat_trig, "trigMuons_passIso0", "trigMuons_passTrigger0",
                                          *muon_columns_stat_nonTrig, "nonTrigMuons_passIso0", "nonTrigMuons_passTrigger0"]
        else:
            muon_columns_stat_step = muon_columns_stat

        statNameBase = "effStatTnP"
        if len(customHistNameTag):
            statNameBase += f"_{customHistNameTag}"
        df = df.Define(f"{statNameBase}_{key}_tensor", helper, [*muon_columns_stat_step, "nominal_weight"])
        name = Datagroups.histName(base_name, syst=f"{statNameBase}_{key}")
        add_syst_hist(results, df, name, axes, cols, f"{statNameBase}_{key}_tensor", helper.tensor_axes, **kwargs)

    systNameBase = "effSystTnP"
    if len(customHistNameTag):
        systNameBase += f"_{customHistNameTag}"
    df = df.Define(f"{systNameBase}_weight", helper_syst, [*muon_columns_syst, "nominal_weight"])
    name = Datagroups.histName(base_name, syst=f"{systNameBase}")
    add_syst_hist(results, df, name, axes, cols, f"{systNameBase}_weight", helper_syst.tensor_axes, **kwargs)

    return df


def add_muon_efficiency_unc_hists_altBkg(results, df, helper_syst, axes, cols, base_name="nominal", 
                                         what_analysis=ROOT.wrem.AnalysisType.Wmass, singleMuonCollection="goodMuons", step="tracking", customHistNameTag="", **kwargs
    ):

    SAvarTag = "SA" if step == "tracking" else "" 
    if what_analysis == ROOT.wrem.AnalysisType.Wmass:
        muon_columns_syst = [f"{singleMuonCollection}_{SAvarTag}pt0", f"{singleMuonCollection}_{SAvarTag}eta0", f"{singleMuonCollection}_charge0"]
    else:
        muvars_syst = [f"{SAvarTag}pt0", f"{SAvarTag}eta0", "charge0"]
        muon_columns_syst_trig    = [f"trigMuons_{v}" for v in muvars_syst]
        muon_columns_syst_nonTrig = [f"nonTrigMuons_{v}" for v in muvars_syst]
        
        if what_analysis == ROOT.wrem.AnalysisType.Wlike:
            muon_columns_syst = [*muon_columns_syst_trig, *muon_columns_syst_nonTrig]
        elif what_analysis == ROOT.wrem.AnalysisType.Dilepton:
            muon_columns_syst = [*muon_columns_syst_trig, *muon_columns_syst_nonTrig]
        else:
            raise NotImplementedError(f"add_muon_efficiency_unc_hists_altBkg: analysis {what_analysis} not implemented.")

    systNameBase = "effSystTnP_altBkg"
    if len(customHistNameTag):
        systNameBase += f"_{customHistNameTag}"

    df = df.Define(f"{systNameBase}_{step}_weight", helper_syst, [*muon_columns_syst, "nominal_weight"])
    name = Datagroups.histName(base_name, syst=f"{systNameBase}_{step}")
    add_syst_hist(results, df, name, axes, cols, f"{systNameBase}_{step}_weight", helper_syst.tensor_axes, **kwargs)

    return df


def add_muon_efficiency_veto_unc_hists(results, df, helper_stat, helper_syst, axes, cols, base_name="nominal", muons="vetoMuons", customHistNameTag="", **kwargs):
    # TODO: update for dilepton
    muon_columns_stat = [f"{muons}_{v}" for v in ["pt0", "eta0", "charge0"]]
    muon_columns_syst = [f"{muons}_{v}" for v in ["pt0", "eta0", "charge0"]]

    statNameBase = "effStatTnP"
    if len(customHistNameTag):
        statNameBase += f"_{customHistNameTag}"
    
    df = df.Define(f"{statNameBase}_veto_tensor", helper_stat, [*muon_columns_stat, "nominal_weight"])
    name = Datagroups.histName(base_name, syst=f"{statNameBase}_veto_sf")
    add_syst_hist(results, df, name, axes, cols, f"{statNameBase}_veto_tensor", helper_stat.tensor_axes, **kwargs)

    systNameBase = "effSystTnP"
    if len(customHistNameTag):
        systNameBase += f"_{customHistNameTag}"

    df = df.Define(f"{systNameBase}_veto_weight", helper_syst, [*muon_columns_syst, "nominal_weight"])
    name = Datagroups.histName(base_name, syst=f"{systNameBase}_veto")
    add_syst_hist(results, df, name, axes, cols, f"{systNameBase}_veto_weight", helper_syst.tensor_axes, **kwargs)

    return df


def add_L1Prefire_unc_hists(results, df, helper_stat, helper_syst, axes, cols, base_name="nominal", **kwargs):
    df = df.Define("muonL1PrefireStat_tensor", helper_stat, ["Muon_correctedEta", "Muon_correctedPt", "Muon_correctedPhi", "Muon_correctedCharge", "Muon_looseId", "nominal_weight"])
    name = Datagroups.histName(base_name, syst="muonL1PrefireStat")    
    add_syst_hist(results, df, name, axes, cols, "muonL1PrefireStat_tensor", helper_stat.tensor_axes, **kwargs)

    df = df.Define("muonL1PrefireSyst_tensor", helper_syst, ["Muon_correctedEta", "Muon_correctedPt", "Muon_correctedPhi", "Muon_correctedCharge", "Muon_looseId", "nominal_weight"])
    name = Datagroups.histName(base_name, syst="muonL1PrefireSyst")
    add_syst_hist(results, df, name, axes, cols, "muonL1PrefireSyst_tensor", common.down_up_axis, **kwargs)

    df = df.Define("ecalL1Prefire_tensor", "wrem::twoPointScaling(nominal_weight/L1PreFiringWeight_ECAL_Nom, L1PreFiringWeight_ECAL_Dn, L1PreFiringWeight_ECAL_Up)")
    name = Datagroups.histName(base_name, syst="ecalL1Prefire")
    add_syst_hist(results, df, name, axes, cols, "ecalL1Prefire_tensor", common.down_up_axis, **kwargs)

    return df


def add_muonscale_hist(results, df, netabins, mag, isW, axes, cols, base_name="nominal", muon_eta="goodMuons_eta0", **kwargs):
    nweights = 21 if isW else 23

    df = df.Define(f"muonScaleDummy{netabins}Bins{muon_eta}", f"wrem::dummyScaleFromMassWeights<{netabins}, {nweights}>(nominal_weight, massWeight_tensor, {muon_eta}, {mag}, {str(isW).lower()})")

    scale_etabins_axis = hist.axis.Regular(netabins, -2.4, 2.4, name="scaleEtaSlice", underflow=False, overflow=False)
    name = Datagroups.histName(base_name, syst=f"muonScaleSyst")
    add_syst_hist(results, df, name, axes, cols, f"muonScaleDummy{netabins}Bins{muon_eta}", [common.down_up_axis, scale_etabins_axis], **kwargs)

    return df


def add_muonscale_smeared_hist(results, df, netabins, mag, isW, axes, cols, base_name="nominal", muon_eta="goodMuons_eta0", *kwargs):
    # add_muonscale_hist has to be called first such that "muonScaleDummy{netabins}Bins{muon_eta}" is defined
    nweights = 21 if isW else 23

    scale_etabins_axis = hist.axis.Regular(netabins, -2.4, 2.4, name="scaleEtaSlice", underflow=False, overflow=False)
    name = Datagroups.histName(base_name, syst="muonScaleSyst_gen_smear")
    add_syst_hist(results, df, name, axes, cols, f"muonScaleDummy{netabins}Bins{muon_eta}", [common.down_up_axis, scale_etabins_axis], **kwargs)

    return df


def scetlib_scale_unc_hist(h, obs, syst_ax="vars"):
    hnew = hist.Hist(*h.axes[:-1], hist.axis.StrCategory(["central"]+scetlib_scale_vars(),
                        name=syst_ax), storage=h._storage_type())
    
    hnew[...,"central"] = h[...,"central"].view(flow=True)
    hnew[...,"resumFOScaleUp"] = h[...,"kappaFO2."].view(flow=True)
    hnew[...,"resumFOScaleDown"] = h[...,"kappaFO0.5"].view(flow=True)
    hnew[...,"resumLambdaUp"] = h[...,"lambda0.8"].view(flow=True)
    hnew[...,"resumLambdaDown"] = h[...,"lambda1.5"].view(flow=True)
    
    transition_names = [x for x in h.axes[syst_ax] if "transition" in x]    
    hnew[...,"resumTransitionUp"] = hh.syst_min_or_max_env_hist(h, obs, syst_ax, 
                                    h.axes[syst_ax].index(transition_names), do_min=False).view(flow=True)
    hnew[...,"resumTransitionDown"] = hh.syst_min_or_max_env_hist(h, obs, syst_ax, 
                                    h.axes[syst_ax].index(transition_names), do_min=True).view(flow=True)
    
    resum_names = [x for x in h.axes[syst_ax] if not any(i in x for i in ["lambda", "kappa", "transition"])]
    hnew[...,"resumScaleUp"] = hh.syst_min_or_max_env_hist(h, obs, syst_ax, 
                                    h.axes[syst_ax].index(resum_names), do_min=False).view(flow=True)
    hnew[...,"resumScaleDown"] = hh.syst_min_or_max_env_hist(h, obs, syst_ax, 
                                    h.axes[syst_ax].index(resum_names), do_min=True).view(flow=True)
    return hnew


def add_theory_hists(results, df, args, dataset_name, corr_helpers, qcdScaleByHelicity_helper, axes, cols, 
    base_name="nominal", for_wmass=True, addhelicity=False, nhelicity=6, storage_type=hist.storage.Double()
):
    logger.debug(f"Make theory histograms for {dataset_name} dataset, histogram {base_name}")
    axis_ptVgen = hist.axis.Variable(
        common.ptV_binning, 
        name = "ptVgen", underflow=False
    )
    #for hel analysis, ptVgen is part of axes/col
    ## FIXME:
    ## here should probably not force using the same ptVgen axis when addhelicity=True
    #scale_axes = [*axes, axis_chargeVgen] if addhelicity else [*axes, axis_ptVgen, axis_chargeVgen]
    #scale_cols = [*cols, "chargeVgen"] if addhelicity else [*cols, "ptVgen", "chargeVgen"]
    if "ptVgen" not in cols:
        scale_axes = [*axes, axis_ptVgen]
        scale_cols = [*cols, "ptVgen"]
    else:
        scale_axes = axes
        scale_cols = cols

    isZ = dataset_name in common.zprocs_all

    df = theory_tools.define_scale_tensor(df)

    if "MEParamWeight" not in df.GetColumnNames() and "LHEReweightingWeight" not in df.GetColumnNames():
        logger.warning("MEParamWeight not in list of columns, mass, width, and sin2theta weight tensors can not be defined")
    else:
        df = define_mass_width_sin2theta_weights(df, dataset_name)

    # common kwargs
    info = dict(base_name=base_name, addhelicity=addhelicity, nhelicity=nhelicity, storage_type=storage_type)

    add_pdf_hists(results, df, dataset_name, axes, cols, args.pdfs, **info)
    add_qcdScale_hist(results, df, scale_axes, scale_cols, **info)

    theory_corrs = [*args.theoryCorr, *args.ewTheoryCorr]
    if theory_corrs and dataset_name in corr_helpers:
        add_theory_corr_hists(results, df, axes, cols, 
            corr_helpers[dataset_name], theory_corrs, modify_central_weight=not args.theoryCorrAltOnly, isW = not isZ, **info)

    if for_wmass or isZ:
        logger.debug(f"Make QCD scale histograms for {dataset_name}")
        # there is no W backgrounds for the Wlike, make QCD scale histograms only for Z
        # should probably remove the charge here, because the Z only has a single charge and the pt distribution does not depend on which charged lepton is selected

        if qcdScaleByHelicity_helper is not None:
            add_qcdScaleByHelicityUnc_hist(results, df, qcdScaleByHelicity_helper, scale_axes, scale_cols, **info)

        if "MEParamWeight" not in df.GetColumnNames():
            return df
        # TODO: Should have consistent order here with the scetlib correction function
        add_massweights_hist(results, df, axes, cols, proc=dataset_name, **info)
        add_widthweights_hist(results, df, axes, cols, proc=dataset_name, **info)
        if isZ:
            add_sin2thetaweights_hist(results, df, axes, cols, proc=dataset_name, **info)

    return df


def add_helicity_hists(results, df, axes, cols, base_name="nominal_gen", storage=hist.storage.Double()):
    df = df.Define("helicity_xsecs_scale_tensor", "wrem::makeHelicityMomentScaleTensor(csSineCosThetaPhigen, scaleWeights_tensor, nominal_weight)")
    helicity_xsecs_scale = df.HistoBoost(
        f"{base_name}_helicity_xsecs_scale",
        axes, [*cols, "helicity_xsecs_scale_tensor"],
        tensor_axes = [axis_helicity, *theory_tools.scale_tensor_axes],
        storage=storage
        )
    results.append(helicity_xsecs_scale)

    # below logic only valid for specific columns
    if cols == ["massVgen", "absYVgen", "ptVgen", "chargeVgen"]:

        df = df.Define("helicity_xsecs_scale_lhe_tensor", "wrem::makeHelicityMomentScaleTensor(csSineCosThetaPhilhe, scaleWeights_tensor, nominal_weight)")
        lhe_cols = ["massVlhe", "absYVlhe", "ptVlhe", "chargeVlhe"]
        helicity_xsecs_scale_lhe = df.HistoBoost(
            f"{base_name}_helicity_xsecs_scale_lhe",
            axes, [*lhe_cols, "helicity_xsecs_scale_lhe_tensor"],
            tensor_axes = [axis_helicity, *theory_tools.scale_tensor_axes],
            storage=storage
        )
        results.append(helicity_xsecs_scale_lhe)

        df = df.Define("helicity_xsecs_scale_hardProcess_tensor", "wrem::makeHelicityMomentScaleTensor(csSineCosThetaPhihardProcess, scaleWeights_tensor, nominal_weight)")
        hardProcess_cols = ["massVhardProcess", "absYVhardProcess", "ptVhardProcess", "chargeVhardProcess"]
        helicity_xsecs_scale_hardProcess = df.HistoBoost(
            f"{base_name}_helicity_xsecs_scale_hardProcess",
            axes, [*hardProcess_cols, "helicity_xsecs_scale_hardProcess_tensor"],
            tensor_axes = [axis_helicity, *theory_tools.scale_tensor_axes],
            storage=storage
        )
        results.append(helicity_xsecs_scale_hardProcess)

        df = df.Define("helicity_xsecs_scale_postShower_tensor", "wrem::makeHelicityMomentScaleTensor(csSineCosThetaPhipostShower, scaleWeights_tensor, nominal_weight)")
        postShower_cols = ["massVpostShower", "absYVpostShower", "ptVpostShower", "chargeVpostShower"]
        helicity_xsecs_scale_postShower = df.HistoBoost(
            f"{base_name}_helicity_xsecs_scale_postShower",
            axes, [*postShower_cols, "helicity_xsecs_scale_postShower_tensor"],
            tensor_axes = [axis_helicity, *theory_tools.scale_tensor_axes],
            storage=storage
        )
        results.append(helicity_xsecs_scale_postShower)

        df = df.Define("helicity_xsecs_scale_postBeamRemnants_tensor", "wrem::makeHelicityMomentScaleTensor(csSineCosThetaPhipostBeamRemnants, scaleWeights_tensor, nominal_weight)")
        postBeamRemnants_cols = ["massVpostBeamRemnants", "absYVpostBeamRemnants", "ptVpostBeamRemnants", "chargeVpostBeamRemnants"]
        helicity_xsecs_scale_postBeamRemnants = df.HistoBoost(
            f"{base_name}_helicity_xsecs_scale_postBeamRemnants",
            axes, [*postBeamRemnants_cols, "helicity_xsecs_scale_postBeamRemnants_tensor"],
            tensor_axes = [axis_helicity, *theory_tools.scale_tensor_axes],
            storage=storage
        )
        results.append(helicity_xsecs_scale_postBeamRemnants)

    return df
