import ROOT
import hist
import numpy as np
from utilities import boostHistHelpers as hh, common, logging
from wremnants import theory_tools
from wremnants.datasets.datagroups import Datagroups
from wremnants.helicity_utils import *
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

    gen_sum_expr = {genAxis : s[::hist.sum] for genAxis in gen_axes if genAxis in axisNames}
    if len(gen_sum_expr) == 0:
        # all the axes have already been projected out, nothing else to do
        return hist_in

    nom_hist = hist_in[{"vars" : 0}]
    nom_hist_sum = nom_hist[gen_sum_expr]

    variation_data = hist_in.view(flow=True) - nom_hist.view(flow=True)[..., None] + nom_hist_sum.view(flow=True)[..., *len(gen_sum_expr)*[None], None]

    variation_hist = hist.Hist(*hist_in.axes, storage = hist_in._storage_type(),
                                     name = out_name, data = variation_data)

    return variation_hist

def uncertainty_hist_from_envelope(h, proj_ax, entries):
    hdown = hh.syst_min_or_max_env_hist(h, proj_ax, "vars", entries, no_flow=["ptVgen"], do_min=True)
    hup = hh.syst_min_or_max_env_hist(h, proj_ax, "vars", entries, no_flow=["ptVgen"], do_min=False)
    hnew = hist.Hist(*h.axes[:-1], common.down_up_axis, storage=h._storage_type())
    hnew[...,0] = hdown.view(flow=True)
    hnew[...,1] = hup.view(flow=True)
    return hnew

def define_mass_weights(df, proc):
    if "massWeight_tensor" in df.GetColumnNames():
        logger.debug("massWeight_tensor already defined, do nothing here.")
        return df
    nweights = 23 if proc in common.zprocs_all else 21
    # from -100 to 100 MeV with 10 MeV increment
    df = df.Define("massWeight_tensor", f"wrem::vec_to_tensor_t<double, {nweights}>(MEParamWeight)")
    df = df.Define("massWeight_tensor_wnom", "auto res = massWeight_tensor; res = nominal_weight*res; return res;")

    return df

def add_massweights_hist(results, df, axes, cols, base_name="nominal", proc="", addhelicity=False, storage_type=hist.storage.Double()):
    name = Datagroups.histName(base_name, syst="massWeight"+(proc[0] if len(proc) else proc))
    mass_axis = hist.axis.StrCategory(massWeightNames(proc=proc), name="massShift")
    if addhelicity:
        massweightHelicity, massWeight_axes = make_massweight_helper_helicity(mass_axis)
        df = df.Define("massWeight_tensor_wnom_helicity", massweightHelicity, ['massWeight_tensor_wnom', 'helWeight_tensor'])
        massWeight = df.HistoBoost(name, axes, [*cols, "massWeight_tensor_wnom_helicity"],
                                   tensor_axes=massWeight_axes,
                                   storage=storage_type)
    else:
        massWeight = df.HistoBoost(name, axes, [*cols, "massWeight_tensor_wnom"], 
                                   tensor_axes=[mass_axis], 
                                   storage=storage_type)
    results.append(massWeight)

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

def define_width_weights(df, proc):
    if "widthWeight_tensor" in df.GetColumnNames():
        logger.debug("widthWeight_tensor already defined, do nothing here.")
        return df
    nweights = 5
    df = df.Define("widthWeight_tensor", f"wrem::vec_to_tensor_t<double, {nweights}>(MEParamWeightAltSet1)")
    df = df.Define("widthWeight_tensor_wnom", "auto res = widthWeight_tensor; res = nominal_weight*res; return res;")
    return df

def add_widthweights_hist(results, df, axes, cols, base_name="nominal", proc="", storage_type=hist.storage.Double()):
    name = Datagroups.histName(base_name, syst="widthWeight"+(proc[0] if len(proc) else proc))
    widthWeight = df.HistoBoost(name, axes, [*cols, "widthWeight_tensor_wnom"], 
                    tensor_axes=[hist.axis.StrCategory(widthWeightNames(proc=proc), name="width")], 
                    storage=storage_type)
    results.append(widthWeight)

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

def add_pdf_hists(results, df, dataset, axes, cols, pdfs, base_name="nominal", addhelicity=False, propagateToHelicity=False, storage_type=hist.storage.Double()):
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

        has_as = "alphasRange" in pdfInfo
        if has_as:
            asr = pdfInfo["alphasRange"] 
            alphaSHistName = Datagroups.histName(base_name, syst=f"{pdfName}alphaS{asr}")
            as_ax = hist.axis.StrCategory(["as0118"]+(["as0117", "as0119"] if asr == "001" else ["as0116", "as0120"]), name="alphasVar")

        if addhelicity:
            pdfHeltensor, pdfHeltensor_axes =  make_pdfweight_helper_helicity(npdf, pdf_ax)
            df = df.Define(f'{tensorName}_helicity', pdfHeltensor, [tensorName, "helWeight_tensor"])
            pdfHist = df.HistoBoost(pdfHistName, axes, [*cols, f'{tensorName}_helicity'], tensor_axes=pdfHeltensor_axes, storage=storage_type)
            if has_as:
                alphaSHeltensor, alphaSHeltensor_axes =  make_pdfweight_helper_helicity(3, as_ax)
                df = df.Define(f'{tensorASName}_helicity', alphaSHeltensor, [tensorASName, "helWeight_tensor"])
                alphaSHist = df.HistoBoost(alphaSHistName, axes, [*cols, f'{tensorASName}_helicity'], tensor_axes=alphaSHeltensor_axes, storage=storage_type)
        else:
            pdfHist = df.HistoBoost(pdfHistName, axes, [*cols, tensorName], tensor_axes=[pdf_ax], storage=storage_type)
            if has_as:
                alphaSHist = df.HistoBoost(alphaSHistName, axes, [*cols, tensorASName], tensor_axes=[as_ax], storage=storage_type)

            if propagateToHelicity:

                pdfhelper = ROOT.wrem.makeHelicityMomentPdfTensor[npdf]()
                df = df.Define(f"helicity_moments_{tensorName}_tensor", pdfhelper, ["csSineCosThetaPhigen", f"{tensorName}", "unity"])
                alphahelper = ROOT.wrem.makeHelicityMomentPdfTensor[3]()
                df = df.Define(f"helicity_moments_{tensorASName}_tensor", alphahelper, ["csSineCosThetaPhigen", f"{tensorASName}", "unity"])
                pdfHist_hel = df.HistoBoost(f"helicity_{pdfHistName}", axes, [*cols, f"helicity_moments_{tensorName}_tensor"], tensor_axes=[axis_helicity,pdf_ax], storage=storage_type)
                alphaSHist_hel = df.HistoBoost(f"helicity_{alphaSHistName}", axes, [*cols, f"helicity_moments_{tensorASName}_tensor"], tensor_axes=[axis_helicity,as_ax], storage=storage_type)
                results.extend([pdfHist_hel, alphaSHist_hel])

        results.extend([pdfHist, alphaSHist])
    return df

def add_qcdScale_hist(results, df, axes, cols, base_name="nominal", addhelicity=False, storage_type=hist.storage.Double()):
    name = Datagroups.histName(base_name, syst="qcdScale")
    if addhelicity:
        qcdbyHelicity, qcdbyHelicity_axes = make_qcdscale_helper_helicity(theory_tools.scale_tensor_axes)
        df = df.Define('scaleWeights_tensor_wnom_helicity', qcdbyHelicity, ['scaleWeights_tensor_wnom', 'helWeight_tensor'])
        scaleHist = df.HistoBoost(name, axes, [*cols, "scaleWeights_tensor_wnom_helicity"], tensor_axes=qcdbyHelicity_axes, storage=storage_type)
    else:
        scaleHist = df.HistoBoost(name, axes, [*cols, "scaleWeights_tensor_wnom"], tensor_axes=theory_tools.scale_tensor_axes, storage=storage_type)
    results.append(scaleHist)

def add_qcdScaleByHelicityUnc_hist(results, df, helper, axes, cols, base_name="nominal", addhelicity=False, storage_type=hist.storage.Double()):
    name = Datagroups.histName(base_name, syst="qcdScaleByHelicity")
    if "helicityWeight_tensor" not in df.GetColumnNames():
        df = df.Define("helicityWeight_tensor", helper, ["massVgen", "absYVgen", "ptVgen", "chargeVgen", "csSineCosThetaPhigen", "nominal_weight"])
    if addhelicity:
        qcdbyHelicity, qcdbyHelicity_axes = make_qcdscale_helper_helicity(helper.tensor_axes)
        df = df.Define('scaleWeights_tensor_wnom_helicity', qcdbyHelicity, ['helicityWeight_tensor', 'helWeight_tensor'])
        qcdScaleByHelicityUnc = df.HistoBoost(name, axes, [*cols, "scaleWeights_tensor_wnom_helicity"], tensor_axes=qcdbyHelicity_axes, storage=storage_type)
    else:
        qcdScaleByHelicityUnc = df.HistoBoost(name, axes, [*cols,"helicityWeight_tensor"], tensor_axes=helper.tensor_axes, storage=storage_type)
    results.append(qcdScaleByHelicityUnc)

def add_QCDbkg_jetPt_hist(results, df, nominal_axes, nominal_cols, base_name="nominal", jet_pt=30, storage_type=hist.storage.Double()):
    # branching the rdataframe to add special filter, no need to return dQCDbkGVar
    name = Datagroups.histName(base_name, syst=f"qcdJetPt{str(jet_pt)}")
    dQCDbkGVar = df.Define(f"goodCleanJetsPt{jet_pt}", f"goodCleanJetsNoPt && Jet_pt > {jet_pt}")
    dQCDbkGVar = dQCDbkGVar.Filter(f"passMT || Sum(goodCleanJetsPt{jet_pt})>=1")
    qcdJetPt = dQCDbkGVar.HistoBoost(name, nominal_axes, [*nominal_cols, "nominal_weight"], storage=storage_type)
    results.append(qcdJetPt)

def add_luminosity_unc_hists(results, df, args, axes, cols, addhelicity=False, storage_type=hist.storage.Double()):
    # TODO: implement for theory agnostic with addhelicity=True
    if addhelicity:
        pass
    else:
        df = df.Define("luminosityScaling", f"wrem::constantScaling(nominal_weight, {args.lumiUncertainty})")
        luminosity = df.HistoBoost("nominal_luminosity", axes, [*cols, "luminosityScaling"], tensor_axes = [common.down_up_axis], storage=storage_type)
        results.append(luminosity)
    return df

# TODO: generalize to non-constant scaling if needed
def add_scaledByCondition_unc_hists(results, df, args, axes, cols, newWeightName, histName, condition, scale, storage_type=hist.storage.Double()):
    # scale represents the scaling factor of the nominal weight, 1.1 means +10%, 2.0 means + 100% and so on
    df = df.Define(newWeightName, f"({condition}) ? ({scale}*nominal_weight) : nominal_weight")
    # df = df.Filter(f"wrem::printVar({newWeightName})")
    # df = df.Filter(f"wrem::printVar(nominal_weight)")
    scaledHist = df.HistoBoost(f"nominal_{histName}", axes, [*cols, newWeightName], storage=storage_type)
    results.append(scaledHist)
    return df

def add_muon_efficiency_unc_hists(results, df, helper_stat, helper_syst, axes, cols, base_name="nominal", what_analysis=ROOT.wrem.AnalysisType.Wmass, smooth3D=False, addhelicity=False, storage_type=hist.storage.Double()):
    # TODO: update for dilepton
    if what_analysis == ROOT.wrem.AnalysisType.Wmass:
        muon_columns_stat = ["goodMuons_pt0", "goodMuons_eta0",
                             "goodMuons_uT0", "goodMuons_charge0"]
        muon_columns_syst = ["goodMuons_pt0", "goodMuons_eta0",
                             "goodMuons_SApt0", "goodMuons_SAeta0",
                             "goodMuons_uT0", "goodMuons_charge0",
                             "passIso"]
    else:
        muvars_stat = ["pt0", "eta0", "uT0", "charge0"]
        muon_columns_stat_trig    = [f"trigMuons_{v}" for v in muvars_stat]
        muon_columns_stat_nonTrig = [f"nonTrigMuons_{v}" for v in muvars_stat]

        muvars_syst = ["pt0", "eta0", "SApt0", "SAeta0", "uT0", "charge0"]
        muon_columns_syst_trig    = [f"trigMuons_{v}" for v in muvars_syst]
        muon_columns_syst_nonTrig = [f"nonTrigMuons_{v}" for v in muvars_syst]
        
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
        elif "iso" in key and what_analysis == ROOT.wrem.AnalysisType.Wmass:
            muon_columns_stat_step = muon_columns_stat + ["passIso"]
        else:
            muon_columns_stat_step = muon_columns_stat
            
        df = df.Define(f"effStatTnP_{key}_tensor", helper, [*muon_columns_stat_step, "nominal_weight"])
        name = Datagroups.histName(base_name, syst=f"effStatTnP_{key}")
        if addhelicity:
            helper_helicity, helper_helicity_axes = make_muon_eff_stat_helpers_helicity(helper)
            df = df.Define(f"effStatTnP_{key}_ByHelicity_tensor", helper_helicity, [f"effStatTnP_{key}_tensor", "helWeight_tensor"])
            effStatTnP = df.HistoBoost(name, axes, [*cols, f"effStatTnP_{key}_ByHelicity_tensor"], tensor_axes = helper_helicity_axes, storage=storage_type)
        else:
            effStatTnP = df.HistoBoost(name, axes, [*cols, f"effStatTnP_{key}_tensor"], tensor_axes = helper.tensor_axes, storage=storage_type)
        results.append(effStatTnP)
    
    df = df.Define("effSystTnP_weight", helper_syst, [*muon_columns_syst, "nominal_weight"])
    name = Datagroups.histName(base_name, syst=f"effSystTnP")
    if addhelicity:
        helper_syst_helicity, helper_syst_helicity_axes = make_muon_eff_syst_helper_helicity(helper_syst)
        df = df.Define("effSystTnP_weight_ByHelicity_tensor", helper_syst_helicity, ["effSystTnP_weight", "helWeight_tensor"])
        effSystTnP = df.HistoBoost(name, axes, [*cols, "effSystTnP_weight_ByHelicity_tensor"], tensor_axes = helper_syst_helicity_axes, storage=storage_type)
    else:
        effSystTnP = df.HistoBoost(name, axes, [*cols, "effSystTnP_weight"], tensor_axes = helper_syst.tensor_axes, storage=storage_type)
    results.append(effSystTnP)
    
    return df

def add_muon_efficiency_veto_unc_hists(results, df, helper_stat, helper_syst, axes, cols, base_name="nominal", storage_type=hist.storage.Double()):
    # TODO: update for dilepton
    muon_columns_stat = ["unmatched_postfsrMuon_pt","unmatched_postfsrMuon_eta","unmatched_postfsrMuon_charge"]
    muon_columns_syst = ["unmatched_postfsrMuon_pt","unmatched_postfsrMuon_eta","unmatched_postfsrMuon_charge"]
            
    df = df.Define("effStatTnP_veto_tensor", helper_stat, [*muon_columns_stat, "nominal_weight"])
    name = Datagroups.histName(base_name, syst="effStatTnP_veto_sf")
    effStatTnP = df.HistoBoost(name, axes, [*cols, "effStatTnP_veto_tensor"], tensor_axes = helper_stat.tensor_axes, storage=storage_type)
    results.append(effStatTnP)
    
    df = df.Define("effSystTnP_veto_weight", helper_syst, [*muon_columns_syst, "nominal_weight"])
    name = Datagroups.histName(base_name, syst="effSystTnP_veto")
    effSystTnP = df.HistoBoost(name, axes, [*cols, "effSystTnP_veto_weight"], tensor_axes = helper_syst.tensor_axes, storage=storage_type)
    results.append(effSystTnP)
    
    return df

def add_L1Prefire_unc_hists(results, df, helper_stat, helper_syst, axes, cols, base_name="nominal", addhelicity=False, storage_type=hist.storage.Double()):
    df = df.Define("muonL1PrefireStat_tensor", helper_stat, ["Muon_correctedEta", "Muon_correctedPt", "Muon_correctedPhi", "Muon_correctedCharge", "Muon_looseId", "nominal_weight"])
    name = Datagroups.histName(base_name, syst=f"muonL1PrefireStat")    

    if addhelicity:
        prefirebyhelicity_stat, prefire_axes_stat = make_muon_prefiring_helper_stat_byHelicity(helper_stat)
        df = df.Define("muonL1PrefireStatByHelicity_tensor", prefirebyhelicity_stat, ["muonL1PrefireStat_tensor", "helWeight_tensor"])
        muonL1PrefireStat = df.HistoBoost(name, axes, [*cols, "muonL1PrefireStatByHelicity_tensor"], tensor_axes = prefire_axes_stat, storage=storage_type)
    else:
        muonL1PrefireStat = df.HistoBoost(name, axes, [*cols, "muonL1PrefireStat_tensor"], tensor_axes = helper_stat.tensor_axes, storage=storage_type)
    results.append(muonL1PrefireStat)

    df = df.Define("muonL1PrefireSyst_tensor", helper_syst, ["Muon_correctedEta", "Muon_correctedPt", "Muon_correctedPhi", "Muon_correctedCharge", "Muon_looseId", "nominal_weight"])
    name = Datagroups.histName(base_name, syst=f"muonL1PrefireSyst")
    prefirebyhelicity_syst, prefire_axes_syst = make_muon_prefiring_helper_syst_byHelicity()
    if addhelicity:
        df = df.Define("muonL1PrefireSystByHelicity_tensor", prefirebyhelicity_syst, ["muonL1PrefireSyst_tensor", "helWeight_tensor"])
        muonL1PrefireSyst = df.HistoBoost(name, axes, [*cols, "muonL1PrefireSystByHelicity_tensor"], tensor_axes = prefire_axes_syst, storage=storage_type)
    else:
        muonL1PrefireSyst = df.HistoBoost(name, axes, [*cols, "muonL1PrefireSyst_tensor"], tensor_axes = [common.down_up_axis], storage=storage_type)
    results.append(muonL1PrefireSyst)

    df = df.Define("ecalL1Prefire_tensor", f"wrem::twoPointScaling(nominal_weight/L1PreFiringWeight_ECAL_Nom, L1PreFiringWeight_ECAL_Dn, L1PreFiringWeight_ECAL_Up)")
    name = Datagroups.histName(base_name, syst=f"ecalL1Prefire")
    if addhelicity:
        #can reuse the same helper since it's the tensor multiplication of same types
        df = df.Define("ecalL1PrefireByHelicity_tensor", prefirebyhelicity_syst, ["ecalL1Prefire_tensor", "helWeight_tensor"])
        ecalL1Prefire = df.HistoBoost(name, axes, [*cols, "ecalL1PrefireByHelicity_tensor"], tensor_axes = prefire_axes_syst, storage=storage_type)
    else:
        ecalL1Prefire = df.HistoBoost(name, axes, [*cols, "ecalL1Prefire_tensor"], tensor_axes = [common.down_up_axis], storage=storage_type)
    results.append(ecalL1Prefire)

    return df

def add_muonscale_hist(results, df, netabins, mag, isW, axes, cols, base_name="nominal", muon_eta="goodMuons_eta0", storage_type=hist.storage.Double()):
    nweights = 21 if isW else 23

    df = df.Define(f"muonScaleDummy{netabins}Bins{muon_eta}", f"wrem::dummyScaleFromMassWeights<{netabins}, {nweights}>(nominal_weight, massWeight_tensor, {muon_eta}, {mag}, {str(isW).lower()})")

    scale_etabins_axis = hist.axis.Regular(netabins, -2.4, 2.4, name="scaleEtaSlice", underflow=False, overflow=False)
    name = Datagroups.histName(base_name, syst=f"muonScaleSyst")

    dummyMuonScaleSyst = df.HistoBoost(name, axes, [*cols, f"muonScaleDummy{netabins}Bins{muon_eta}"], tensor_axes=[common.down_up_axis, scale_etabins_axis], storage=storage_type)
    results.append(dummyMuonScaleSyst)

    return df


def add_muonscale_smeared_hist(results, df, netabins, mag, isW, axes, cols, base_name="nominal", muon_eta="goodMuons_eta0", storage_type=hist.storage.Double()):
    # add_muonscale_hist has to be called first such that "muonScaleDummy{netabins}Bins{muon_eta}" is defined
    nweights = 21 if isW else 23

    scale_etabins_axis = hist.axis.Regular(netabins, -2.4, 2.4, name="scaleEtaSlice", underflow=False, overflow=False)
    name = Datagroups.histName(base_name, syst=f"muonScaleSyst_gen_smear")

    dummyMuonScaleSyst_gen_smear = df.HistoBoost(name, axes, [*cols, f"muonScaleDummy{netabins}Bins{muon_eta}"], tensor_axes=[common.down_up_axis, scale_etabins_axis], storage=storage_type)
    results.append(dummyMuonScaleSyst_gen_smear)

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
    base_name="nominal", for_wmass=True, addhelicity=False, storage_type=hist.storage.Double()
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

    df = theory_tools.define_scale_tensor(df)
    df = define_mass_weights(df, dataset_name)
    df = define_width_weights(df, dataset_name)

    add_pdf_hists(results, df, dataset_name, axes, cols, args.pdfs, base_name=base_name, addhelicity=addhelicity, storage_type=storage_type)
    add_qcdScale_hist(results, df, scale_axes, scale_cols, base_name=base_name, addhelicity=addhelicity, storage_type=storage_type)

    isZ = dataset_name in common.zprocs_all

    theory_corrs = [*args.theoryCorr, *args.ewTheoryCorr]
    if theory_corrs and dataset_name in corr_helpers:
        results.extend(theory_tools.make_theory_corr_hists(df, base_name, axes, cols, 
            corr_helpers[dataset_name], theory_corrs, modify_central_weight=not args.theoryCorrAltOnly, isW = not isZ, storage_type=storage_type)
        )

    if "gen" in base_name:
        df = df.Define("helicity_moments_scale_tensor", "wrem::makeHelicityMomentScaleTensor(csSineCosThetaPhigen, scaleWeights_tensor, nominal_weight)")
        helicity_moments_scale = df.HistoBoost("nominal_gen_helicity_moments_scale", axes, [*cols, "helicity_moments_scale_tensor"], tensor_axes = [axis_helicity, *theory_tools.scale_tensor_axes], storage=hist.storage.Double())
        results.append(helicity_moments_scale)

    if for_wmass or isZ:
        logger.debug(f"Make QCD scale histograms for {dataset_name}")
        # there is no W backgrounds for the Wlike, make QCD scale histograms only for Z
        # should probably remove the charge here, because the Z only has a single charge and the pt distribution does not depend on which charged lepton is selected

        if qcdScaleByHelicity_helper is not None:
            add_qcdScaleByHelicityUnc_hist(results, df, qcdScaleByHelicity_helper, scale_axes, scale_cols, 
                base_name=base_name, addhelicity=addhelicity, storage_type=storage_type)

        # TODO: Should have consistent order here with the scetlib correction function
        add_massweights_hist(results, df, axes, cols, proc=dataset_name, base_name=base_name, addhelicity=addhelicity, storage_type=storage_type)
        add_widthweights_hist(results, df, axes, cols, proc=dataset_name, base_name=base_name, storage_type=storage_type)

    return df
