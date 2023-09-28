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
    pdfInfo = theory_tools.pdfMapExtended 
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
    transforms["scetlib_dyturboMSHT20an3loUp"] = {"action" : lambda h: pdfUnc(h, "pdfMSHT20", "vars")[0], "procs" : common.zprocs_all}
    transforms["scetlib_dyturboMSHT20an3loDown"] = {"action" : lambda h: pdfUnc(h, "pdfMSHT20", "vars")[1], "procs" : common.zprocs_all}

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
       "resumTNPAllUp" : {
           "action" : lambda h: h if "vars" not in h.axes.name else theory_tools.hessianPdfUnc(h[{"vars" :
                 [x for x in h.axes["vars"] if any(re.match(y, x) for y in ["pdf0", "^gamma_", "^q_", "b_", "^s+", "^s-", "^h_"])]}], 
                "vars", uncType="asymHessian")[0]},
       "resumTNPAllDown" : {
           "action" : lambda h: h if "vars" not in h.axes.name else theory_tools.hessianPdfUnc(h[{"vars" :
                 [x for x in h.axes["vars"] if any(re.match(y, x) for y in ["pdf0", "^gamma_", "^q_", "b_", "^s+", "^s-", "^h_"])]}], 
                "vars", uncType="asymHessian")[1]},
       "resumScaleAllUp" : {
           "action" : lambda h: h if "vars" not in h.axes.name else hh.syst_min_or_max_env_hist(h, projAx(hist_name), "vars",
                 [x for x in h.axes["vars"] if any(re.match(y, x) for y in ["pdf0", "^nuB.*", "nuS.*", "^muB.*", "^muS.*",])],
                    do_min=False)},
       "resumScaleAllDown" : {
           "action" : lambda h: h if "vars" not in h.axes.name else hh.syst_min_or_max_env_hist(h, projAx(hist_name), "vars",
                 [x for x in h.axes["vars"] if any(re.match(y, x) for y in ["pdf0", "^nuB.*", "nuS.*", "^muB.*", "^muS.*",])],
                    do_min=True)},
       "resumNPUp" : {
            "action" : lambda h: hh.syst_min_or_max_env_hist(h, projAx(hist_name), "vars", 
                 [x for x in h.axes["vars"] if "Omega" in x or "cnu" in x or "omega" in x],
                 no_flow=["ptVgen"], do_min=False) if "vars" in h.axes.name else h},
        "resumNPDown" : {
            "action" : lambda h: hh.syst_min_or_max_env_hist(h, projAx(hist_name), "vars", 
                 [x for x in h.axes["vars"] if "omega" in x or "cnu" in x or "omega" in x],
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
    #for k,v in transforms.items():
    #    if any([x in k for x in ["QCDscale", "resum", "pdf"]]):
    #        #v["procs"] = common.vprocs 

    return transforms

def scale_helicity_hist_to_variations(scale_hist, sum_axes=[], rebinPtV=None):
    s = hist.tag.Slicer()
    axisNames = scale_hist.axes.name

    sum_expr = {axis : s[::hist.sum] for axis in sum_axes if axis in axisNames}
    scale_hist = scale_hist[sum_expr]
    axisNames = scale_hist.axes.name
    
    # select nominal QCD scales, but keep the sliced axis at size 1 for broadcasting
    nom_scale_hist = scale_hist[{"muRfact" : s[1.j:1.j+1], "muFfact" : s[1.j:1.j+1]}]
    # select nominal QCD scales and project down to nominal axes
    genAxes = ["ptVgen", "chargeVgen", "helicity"]
    nom_sel = {"muRfact" : s[1.j], "muFfact" : s[1.j] }
    nom_sel.update({genAxis : s[::hist.sum] for genAxis in genAxes if genAxis in axisNames})
    nom_hist = nom_scale_hist[nom_sel]
    
    hasHelicityAxis = "helicity" in axisNames
    hasPtAxis = "ptVgen" in axisNames

    if rebinPtV and hasPtAxis:
        # Treat single bin array as a float
        array_rebin = isinstance(rebinPtV, collections.abc.Sequence)
        if array_rebin and len(rebinPtV) == 1:
            rebinPtV = rebinPtV[0]
            array_rebin = False

        if array_rebin:
            scale_hist = hh.rebinHist(scale_hist, "ptVgen", rebinPtV)
            nom_scale_hist = hh.rebinHist(nom_scale_hist, "ptVgen", rebinPtV)
        else:
            scale_hist = scale_hist[{"ptVgen" : s[::hist.rebin(rebinPtV)]}]
            nom_scale_hist = nom_scale_hist[{"ptVgen" : s[::hist.rebin(rebinPtV)]}]

    # difference between a given scale and the nominal, plus the sum
    # this emulates the "weight if idx else nominal" logic and corresponds to the decorrelated
    # variations
    if scale_hist.name is None:
        out_name = "scale_helicity_variations" if hasHelicityAxis else "scale_vpt_variations" if hasPtAxis else "scale_vcharge_variations"
    else:
        out_name = scale_hist.name + "_variations"

    expd = scale_hist.ndim - nom_hist.ndim
    expandnom = np.expand_dims(nom_hist.values(flow=True), [-expd+i for i in range(expd)])
    systhist = scale_hist.values(flow=True) - nom_scale_hist.values(flow=True) + expandnom

    scale_variation_hist = hist.Hist(*scale_hist.axes, 
                                     name = out_name, data = systhist)

    return scale_variation_hist 

def hist_to_variations(hist_in):

    if hist_in.name is None:
        out_name = "hist_variations"
    else:
        out_name = hist_in.name + "_variations"

    s = hist.tag.Slicer()

    genAxes = ["absYVgenNP", "chargeVgenNP"]

    nom_hist = hist_in[{"vars" : 0}]
    nom_hist_sum = nom_hist[{genAxis : s[::hist.sum] for genAxis in genAxes}]

    variation_data = hist_in.view(flow=True) - nom_hist.view(flow=True)[...,None] + nom_hist_sum.view(flow=True)[..., None, None, None]

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

def add_massweights_hist(results, df, axes, cols, base_name="nominal", proc="", addhelicity=False):
    name = Datagroups.histName(base_name, syst="massWeight"+(proc[0] if len(proc) else proc))
    mass_axis = hist.axis.StrCategory(massWeightNames(proc=proc), name="massShift")
    if addhelicity:
        massweightHelicity, massWeight_axes = make_massweight_helper_helicity(mass_axis)
        df = df.Define("massWeight_tensor_wnom_helicity", massweightHelicity, ['massWeight_tensor_wnom', 'helWeight_tensor'])
        massWeight = df.HistoBoost(name, axes, [*cols, "massWeight_tensor_wnom_helicity"],
                                   tensor_axes=massWeight_axes,
                                   storage=hist.storage.Double())
    else:
        massWeight = df.HistoBoost(name, axes, [*cols, "massWeight_tensor_wnom"], 
                                   tensor_axes=[mass_axis], 
                                   storage=hist.storage.Double())
    results.append(massWeight)

def massWeightNames(matches=None, proc=""):
    central=10
    nweights=21
    names = [f"massShift{proc[0] if len(proc) else proc}{int(abs(central-i)*10)}MeV{'' if i == central else ('Down' if i < central else 'Up')}" for i in range(nweights)]
    if proc and proc in common.zprocs_all:
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

def add_widthweights_hist(results, df, axes, cols, base_name="nominal", proc=""):
    name = Datagroups.histName(base_name, syst="widthWeight"+(proc[0] if len(proc) else proc))
    widthWeight = df.HistoBoost(name, axes, [*cols, "widthWeight_tensor_wnom"], 
                    tensor_axes=[hist.axis.StrCategory(widthWeightNames(proc=proc), name="width")], 
                    storage=hist.storage.Double())
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
    names = [f"width{proc[0]}{str(widths[i]).replace('.','p')}GeV" for i in (0, 1)]
    # 2, 3, and 4 are PDG width Down, Central, Up
    names.append(f"widthShift{proc[0]}{str(2.3 if proc[0] == 'Z' else 42).replace('.','p')}MeVDown")
    names.append(f"width{proc[0]}{str(widths[central]).replace('.','p')}GeV")
    names.append(f"widthShift{proc[0]}{str(2.3 if proc[0] == 'Z' else 42).replace('.','p')}MeVUp")

    return [x if not matches or any(y in x for y in matches) else "" for x in names]

def add_pdf_hists(results, df, dataset, axes, cols, pdfs, base_name="nominal", addhelicity=False):
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

        if pdfInfo["alphasRange"] == "001":
            alphaSHistName = Datagroups.histName(base_name, syst=f"{pdfName}alphaS001")
            as_ax = hist.axis.StrCategory(["as0117", "as0119"], name="alphasVar")
        else:
            alphaSHistName = Datagroups.histName(base_name, syst=f"{pdfName}alphaS002")
            as_ax = hist.axis.StrCategory(["as0116", "as0120"], name="alphasVar")

        if addhelicity:
            pdfHeltensor, pdfHeltensor_axes =  make_pdfweight_helper_helicity(npdf, pdf_ax)
            df = df.Define(f'{tensorName}_helicity', pdfHeltensor, [tensorName, "helWeight_tensor"])
            pdfHist = df.HistoBoost(pdfHistName, axes, [*cols, f'{tensorName}_helicity'], tensor_axes=pdfHeltensor_axes, storage=hist.storage.Double())
            alphaSHeltensor, alphaSHeltensor_axes =  make_pdfweight_helper_helicity(2, as_ax)
            df = df.Define(f'{tensorASName}_helicity', alphaSHeltensor, [tensorASName, "helWeight_tensor"])
            alphaSHist = df.HistoBoost(alphaSHistName, axes, [*cols, f'{tensorASName}_helicity'], tensor_axes=alphaSHeltensor_axes, storage=hist.storage.Double())
        else:
            pdfHist = df.HistoBoost(pdfHistName, axes, [*cols, tensorName], tensor_axes=[pdf_ax], storage=hist.storage.Double())
            alphaSHist = df.HistoBoost(alphaSHistName, axes, [*cols, tensorASName], tensor_axes=[as_ax], storage=hist.storage.Double())
        results.extend([pdfHist, alphaSHist])
    return df

def add_qcdScale_hist(results, df, axes, cols, base_name="nominal", addhelicity=False):
    name = Datagroups.histName(base_name, syst="qcdScale")
    if addhelicity:
        qcdbyHelicity, qcdbyHelicity_axes = make_qcdscale_helper_helicity(theory_tools.scale_tensor_axes)
        df = df.Define('scaleWeights_tensor_wnom_helicity', qcdbyHelicity, ['scaleWeights_tensor_wnom', 'helWeight_tensor'])
        scaleHist = df.HistoBoost(name, axes, [*cols, "scaleWeights_tensor_wnom_helicity"], tensor_axes=qcdbyHelicity_axes, storage=hist.storage.Double())
    else:
        scaleHist = df.HistoBoost(name, axes, [*cols, "scaleWeights_tensor_wnom"], tensor_axes=theory_tools.scale_tensor_axes, storage=hist.storage.Double())
    results.append(scaleHist)

def add_qcdScaleByHelicityUnc_hist(results, df, helper, axes, cols, base_name="nominal", addhelicity=False):
    name = Datagroups.histName(base_name, syst="qcdScaleByHelicity")
    if "helicityWeight_tensor" not in df.GetColumnNames():
        df = df.Define("helicityWeight_tensor", helper, ["massVgen", "absYVgen", "ptVgen", "chargeVgen", "csSineCosThetaPhi", "scaleWeights_tensor", "nominal_weight"])
    if addhelicity:
        qcdbyHelicity, qcdbyHelicity_axes = make_qcdscale_helper_helicity(helper.tensor_axes)
        df = df.Define('scaleWeights_tensor_wnom_helicity', qcdbyHelicity, ['helicityWeight_tensor', 'helWeight_tensor'])
        qcdScaleByHelicityUnc = df.HistoBoost(name, axes, [*cols, "scaleWeights_tensor_wnom_helicity"], tensor_axes=qcdbyHelicity_axes, storage=hist.storage.Double())
    else:
        qcdScaleByHelicityUnc = df.HistoBoost(name, axes, [*cols,"helicityWeight_tensor"], tensor_axes=helper.tensor_axes, storage=hist.storage.Double())
    results.append(qcdScaleByHelicityUnc)

def add_QCDbkg_jetPt_hist(results, df, nominal_axes, nominal_cols, base_name="nominal", jet_pt=30):
    # branching the rdataframe to add special filter, no need to return dQCDbkGVar
    name = Datagroups.histName(base_name, syst=f"qcdJetPt{str(jet_pt)}")
    dQCDbkGVar = df.Define(f"goodCleanJetsPt{jet_pt}", f"goodCleanJetsNoPt && Jet_pt > {jet_pt}")
    dQCDbkGVar = dQCDbkGVar.Filter(f"passMT || Sum(goodCleanJetsPt{jet_pt})>=1")
    qcdJetPt = dQCDbkGVar.HistoBoost(name, nominal_axes, [*nominal_cols, "nominal_weight"], storage=hist.storage.Double())
    results.append(qcdJetPt)

def add_luminosity_unc_hists(results, df, args, axes, cols, addhelicity=False):
    # TODO: implement for theory agnostic with addhelicity=True
    if addhelicity:
        pass
    else:
        df = df.Define("luminosityScaling", f"wrem::constantScaling(nominal_weight, {args.lumiUncertainty})")
        luminosity = df.HistoBoost("nominal_luminosity", axes, [*cols, "luminosityScaling"], tensor_axes = [common.down_up_axis], storage=hist.storage.Double())
        results.append(luminosity)
    return df
    
def add_muon_efficiency_unc_hists(results, df, helper_stat, helper_syst, axes, cols, base_name="nominal", what_analysis=ROOT.wrem.AnalysisType.Wmass, smooth3D=False, addhelicity=False):
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
            effStatTnP = df.HistoBoost(name, axes, [*cols, f"effStatTnP_{key}_ByHelicity_tensor"], tensor_axes = helper_helicity_axes, storage=hist.storage.Double())
        else:
            effStatTnP = df.HistoBoost(name, axes, [*cols, f"effStatTnP_{key}_tensor"], tensor_axes = helper.tensor_axes, storage=hist.storage.Double())
        results.append(effStatTnP)
    
    df = df.Define("effSystTnP_weight", helper_syst, [*muon_columns_syst, "nominal_weight"])
    name = Datagroups.histName(base_name, syst=f"effSystTnP")
    if addhelicity:
        helper_syst_helicity, helper_syst_helicity_axes = make_muon_eff_syst_helper_helicity(helper_syst)
        df = df.Define("effSystTnP_weight_ByHelicity_tensor", helper_syst_helicity, ["effSystTnP_weight", "helWeight_tensor"])
        effSystTnP = df.HistoBoost(name, axes, [*cols, "effSystTnP_weight_ByHelicity_tensor"], tensor_axes = helper_syst_helicity_axes, storage=hist.storage.Double())
    else:
        effSystTnP = df.HistoBoost(name, axes, [*cols, "effSystTnP_weight"], tensor_axes = helper_syst.tensor_axes, storage=hist.storage.Double())
    results.append(effSystTnP)
    
    return df

def add_L1Prefire_unc_hists(results, df, helper_stat, helper_syst, axes, cols, base_name="nominal", addhelicity=False):
    df = df.Define("muonL1PrefireStat_tensor", helper_stat, ["Muon_correctedEta", "Muon_correctedPt", "Muon_correctedPhi", "Muon_correctedCharge", "Muon_looseId", "nominal_weight"])
    name = Datagroups.histName(base_name, syst=f"muonL1PrefireStat")    

    if addhelicity:
        prefirebyhelicity_stat, prefire_axes_stat = make_muon_prefiring_helper_stat_byHelicity(helper_stat)
        df = df.Define("muonL1PrefireStatByHelicity_tensor", prefirebyhelicity_stat, ["muonL1PrefireStat_tensor", "helWeight_tensor"])
        muonL1PrefireStat = df.HistoBoost(name, axes, [*cols, "muonL1PrefireStatByHelicity_tensor"], tensor_axes = prefire_axes_stat, storage=hist.storage.Double())
    else:
        muonL1PrefireStat = df.HistoBoost(name, axes, [*cols, "muonL1PrefireStat_tensor"], tensor_axes = helper_stat.tensor_axes, storage=hist.storage.Double())
    results.append(muonL1PrefireStat)

    df = df.Define("muonL1PrefireSyst_tensor", helper_syst, ["Muon_correctedEta", "Muon_correctedPt", "Muon_correctedPhi", "Muon_correctedCharge", "Muon_looseId", "nominal_weight"])
    name = Datagroups.histName(base_name, syst=f"muonL1PrefireSyst")
    prefirebyhelicity_syst, prefire_axes_syst = make_muon_prefiring_helper_syst_byHelicity()
    if addhelicity:
        df = df.Define("muonL1PrefireSystByHelicity_tensor", prefirebyhelicity_syst, ["muonL1PrefireSyst_tensor", "helWeight_tensor"])
        muonL1PrefireSyst = df.HistoBoost(name, axes, [*cols, "muonL1PrefireSystByHelicity_tensor"], tensor_axes = prefire_axes_syst, storage=hist.storage.Double())
    else:
        muonL1PrefireSyst = df.HistoBoost(name, axes, [*cols, "muonL1PrefireSyst_tensor"], tensor_axes = [common.down_up_axis], storage=hist.storage.Double())
    results.append(muonL1PrefireSyst)

    df = df.Define("ecalL1Prefire_tensor", f"wrem::twoPointScaling(nominal_weight/L1PreFiringWeight_ECAL_Nom, L1PreFiringWeight_ECAL_Dn, L1PreFiringWeight_ECAL_Up)")
    name = Datagroups.histName(base_name, syst=f"ecalL1Prefire")
    if addhelicity:
        #can reuse the same helper since it's the tensor multiplication of same types
        df = df.Define("ecalL1PrefireByHelicity_tensor", prefirebyhelicity_syst, ["ecalL1Prefire_tensor", "helWeight_tensor"])
        ecalL1Prefire = df.HistoBoost(name, axes, [*cols, "ecalL1PrefireByHelicity_tensor"], tensor_axes = prefire_axes_syst, storage=hist.storage.Double())
    else:
        ecalL1Prefire = df.HistoBoost(name, axes, [*cols, "ecalL1Prefire_tensor"], tensor_axes = [common.down_up_axis], storage=hist.storage.Double())
    results.append(ecalL1Prefire)

    return df

def add_muonscale_hist(results, df, netabins, mag, isW, axes, cols, base_name="nominal", muon_eta="goodMuons_eta0"):
    nweights = 21 if isW else 23

    df = df.Define(f"muonScaleDummy{netabins}Bins{muon_eta}", f"wrem::dummyScaleFromMassWeights<{netabins}, {nweights}>(nominal_weight, massWeight_tensor, {muon_eta}, {mag}, {str(isW).lower()})")

    scale_etabins_axis = hist.axis.Regular(netabins, -2.4, 2.4, name="scaleEtaSlice", underflow=False, overflow=False)
    name = Datagroups.histName(base_name, syst=f"muonScaleSyst")

    dummyMuonScaleSyst = df.HistoBoost(name, axes, [*cols, f"muonScaleDummy{netabins}Bins{muon_eta}"], tensor_axes=[common.down_up_axis, scale_etabins_axis], storage=hist.storage.Double())
    results.append(dummyMuonScaleSyst)

    return df


def add_muonscale_smeared_hist(results, df, netabins, mag, isW, axes, cols, base_name="nominal", muon_eta="goodMuons_eta0"):
    # add_muonscale_hist has to be called first such that "muonScaleDummy{netabins}Bins{muon_eta}" is defined
    nweights = 21 if isW else 23

    scale_etabins_axis = hist.axis.Regular(netabins, -2.4, 2.4, name="scaleEtaSlice", underflow=False, overflow=False)
    name = Datagroups.histName(base_name, syst=f"muonScaleSyst_gen_smear")

    dummyMuonScaleSyst_gen_smear = df.HistoBoost(name, axes, [*cols, f"muonScaleDummy{netabins}Bins{muon_eta}"], tensor_axes=[common.down_up_axis, scale_etabins_axis], storage=hist.storage.Double())
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

def add_theory_hists(results, df, args, dataset_name, corr_helpers, qcdScaleByHelicity_helper, axes, cols, base_name="nominal", for_wmass=True, addhelicity=False):
    logger.debug(f"Make theory histograms for {dataset_name} dataset, histogram {base_name}")
    axis_chargeVgen = qcdScaleByHelicity_helper.hist.axes["chargeVgen"]
    axis_ptVgen = hist.axis.Variable(
        common.ptV_10quantiles_binning, 
        name = "ptVgen", underflow=False
    )
    #for hel analysis, ptVgen is part of axes/col
    ## FIXME:
    ## here should probably not force using the same ptVgen axis when addhelicity=True
    #scale_axes = [*axes, axis_chargeVgen] if addhelicity else [*axes, axis_ptVgen, axis_chargeVgen]
    #scale_cols = [*cols, "chargeVgen"] if addhelicity else [*cols, "ptVgen", "chargeVgen"]
    scale_axes = [*axes, axis_ptVgen, axis_chargeVgen]
    scale_cols = [*cols, "ptVgen", "chargeVgen"]

    df = theory_tools.define_scale_tensor(df)
    df = define_mass_weights(df, dataset_name)
    if args.widthVariations:
        df = define_width_weights(df, dataset_name)

    add_pdf_hists(results, df, dataset_name, axes, cols, args.pdfs, base_name=base_name, addhelicity=addhelicity)
    add_qcdScale_hist(results, df, scale_axes, scale_cols, base_name=base_name, addhelicity=addhelicity)

    isZ = dataset_name in common.zprocs_all

    if args.theoryCorr and dataset_name in corr_helpers:
        results.extend(theory_tools.make_theory_corr_hists(df, base_name, axes, cols, 
            corr_helpers[dataset_name], args.theoryCorr, modify_central_weight=not args.theoryCorrAltOnly, isW = not isZ)
        )

    if for_wmass or isZ:
        logger.debug(f"Make QCD scale histograms for {dataset_name}")
        # there is no W backgrounds for the Wlike, make QCD scale histograms only for Z
        # should probably remove the charge here, because the Z only has a single charge and the pt distribution does not depend on which charged lepton is selected

        if not args.skipHelicity:
            add_qcdScaleByHelicityUnc_hist(results, df, qcdScaleByHelicity_helper, scale_axes, scale_cols, base_name=base_name, addhelicity=addhelicity)

        # TODO: Should have consistent order here with the scetlib correction function
        add_massweights_hist(results, df, axes, cols, proc=dataset_name, base_name=base_name, addhelicity=addhelicity)
        if args.widthVariations:
            add_widthweights_hist(results, df, axes, cols, proc=dataset_name, base_name=base_name)

    return df
