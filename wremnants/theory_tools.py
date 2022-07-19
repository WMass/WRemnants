import ROOT
import hist
import numpy as np
import copy
from wremnants import boostHistHelpers as hh
import logging

ROOT.gInterpreter.Declare('#include "theoryTools.h"')

# integer axis for -1 through 7
axis_helicity = hist.axis.Integer(
    -1, 8, name="helicity", overflow=False, underflow=False
)

# this puts the bin centers at 0.5, 1.0, 2.0
axis_muRfact = hist.axis.Variable(
    [0.25, 0.75, 1.25, 2.75], name="muRfact", underflow=False, overflow=False
)
axis_muFfact = hist.axis.Variable(
    [0.25, 0.75, 1.25, 2.75], name="muFfact", underflow=False, overflow=False
)

scale_tensor_axes = (axis_muRfact, axis_muFfact)

pdfMap = {
    "nnpdf31" : {
        "name" : "pdfNNPDF31",
        "branch" : "LHEPdfWeight",
        "combine" : "symHessian",
        "entries" : 101,
        "alphas" : ["LHEPdfWeight[101]", "LHEPdfWeight[102]"],
    },
    "ct18" : {
        # This has CT18 + CT18Z in it :-/
        "name" : "pdfCT18",
        "branch" : "LHEPdfWeightAltSet18",
        "combine" : "asymHessian",
        "entries" : 59,
        "alphas" : ["LHEPdfWeightAltSet18[59]", "LHEPdfWeightAltSet18[60]"],
	"alphaRange" : "002",
        "scale" : 1/1.645 # Convert from 90% CL to 68%
    },
    "mmht" : {
        "name" : "pdfMMHT",
        "branch" : "LHEPdfWeightAltSet19",
        "combine" : "asymHessian",
        "entries" : 51,
        "alphas" : ["LHEPdfWeightAltSet20[1]", "LHEPdfWeightAltSet20[2]"],
	"alphaRange" : "001",
    },
    "nnpdf30" : {
	"name" : "pdfNNPDF30",
	"branch" : "LHEPdfWeightAltSet13",
        "combine" : "symHessian",
	"entries" : 101,
	"alphas" : ["LHEPdfWeightAltSet15[0]", "LHEPdfWeightAltSet16[0]"],
	"alphaRange" : "001",
    },
}

pdfMapExtended = copy.deepcopy(pdfMap)
pdfMapExtended["ct18"]["branch"] = "LHEPdfWeightAltSet11"
pdfMapExtended["ct18"]["alphas"] = ["LHEPdfWeightAltSet11[59]", "LHEPdfWeightAltSet11[62]"]
pdfMapExtended["mmht"]["branch"] = "LHEPdfWeightAltSet13"
pdfMapExtended["mmht"]["alphas"] = ["LHEPdfWeightAltSet14[1]", "LHEPdfWeightAltSet14[2]"]
pdfMapExtended.update({
    "nnpdf40" : {
        "name" : "pdfNNPDF40",
        "branch" : "LHEPdfWeightAltSet3",
        "combine" : "symHessian",
        "entries" : 53,
        "alphas" : ["LHEPdfWeightAltSet3[51]", "LHEPdfWeightAltSet3[52]"],
        "alphasRange" : "002", # TODO: IS that true?
    },
    "pdf4lhc21" : {
        "name" : "pdfPDF4LHC21",
        "branch" : "LHEPdfWeightAltSet10",
        "combine" : "symHessian",
        "entries" : 41,
        "alphas" : ["LHEPdfWeightAltSet10[41]", "LHEPdfWeightAltSet10[42]"],
        "alphasRange" : "002", # TODO: IS that true?
    },
    "msht20" : {
        "name" : "pdfMSHT20",
        "branch" : "LHEPdfWeightAltSet12",
        "combine" : "symHessian",
        "entries" : 51,
        "alphas" : ["LHEPdfWeight[51]", "LHEPdfWeightAltSet12[52]"],
        "alphasRange" : "002", # TODO: IS that true?
    },
    "atlasWZj20" : {
        "name" : "pdfATLASWZJ20",
        "branch" : "LHEPdfWeightAltSet19",
        "combine" : "symHessian",
        "entries" : 33,
        "alphas" : ["LHEPdfWeight[41]", "LHEPdfWeight[42]"],
        "alphasRange" : "002", # TODO: IS that true?
    },


})

only_central_pdf_datasets = [
    "Wplusmunu_bugfix",
    "Wminusmunu_bugfix",
    "Zmumu_bugfix",
    "Zmumu_bugfix_slc7",
]

extended_pdf_datasets = [
    "WminusmunuPostVFP",
    "WplusmunuPostVFP",
    "ZmumuPostVFP",
]

def define_prefsr_vars(df):
    df = df.Define("prefsrLeps", "wrem::prefsrLeptons(GenPart_status, GenPart_statusFlags, GenPart_pdgId, GenPart_genPartIdxMother)")
    df = df.Define("genl", "ROOT::Math::PtEtaPhiMVector(GenPart_pt[prefsrLeps[0]], GenPart_eta[prefsrLeps[0]], GenPart_phi[prefsrLeps[0]], GenPart_mass[prefsrLeps[0]])")
    df = df.Define("genlanti", "ROOT::Math::PtEtaPhiMVector(GenPart_pt[prefsrLeps[1]], GenPart_eta[prefsrLeps[1]], GenPart_phi[prefsrLeps[1]], GenPart_mass[prefsrLeps[1]])")
    df = df.Define("genV", "ROOT::Math::PxPyPzEVector(genl)+ROOT::Math::PxPyPzEVector(genlanti)")
    df = df.Define("ptVgen", "genV.pt()")
    df = df.Define("massVgen", "genV.mass()")
    df = df.Define("yVgen", "genV.Rapidity()")
    df = df.Define("phiVgen", "genV.Phi()")
    df = df.Define("absYVgen", "std::fabs(yVgen)")
    df = df.Define("chargeVgen", "GenPart_pdgId[prefsrLeps[0]] + GenPart_pdgId[prefsrLeps[1]]")
    df = df.Define("csSineCosThetaPhi", "wrem::csSineCosThetaPhi(genl, genlanti)")
    return df

def define_scale_tensor(df):
    # convert vector of scale weights to 3x3 tensor and clip weights to |weight|<10.
    df = df.Define("scaleWeights_tensor", "wrem::makeScaleTensor(LHEScaleWeight, 10.);")
    df = df.Define("scaleWeights_tensor_wnom", "auto res = scaleWeights_tensor; res = nominal_weight*res; return res;")

    return df

def make_scale_hist(df, axes, cols, hname=""):
    scaleHist = df.HistoBoost("qcdScale" if hname=="" else f"{hname}_qcdScale", axes, [*cols, "scaleWeights_tensor_wnom"], tensor_axes=scale_tensor_axes)
    return scaleHist

def pdf_info_map(dataset, pdfset):
    infoMap = pdfMap if dataset not in extended_pdf_datasets else pdfMapExtended

    if (pdfset != "nnpdf31" and dataset in only_central_pdf_datasets) or pdfset not in infoMap:
        raise ValueError(f"Skipping PDF {pdfset} for dataset {dataset}")
    return infoMap[pdfset]

def define_and_make_pdf_hists(df, axes, cols, dataset, pdfset="nnpdf31", storeUnc=True, hname=""):
    try:
        pdfInfo = pdf_info_map(dataset, pdfset)
    except ValueError as e:
        logging.info(e)
        return []

    pdfName = pdfInfo["name"]
    pdfBranch = pdfInfo["branch"]
    tensorName = f"{pdfName}Weights_tensor"
    tensorASName = f"{pdfName}ASWeights_tensor"
    entries = pdfInfo["entries"] if storeUnc else 1

    df = df.Define(tensorName, f"auto res = wrem::clip_tensor(wrem::vec_to_tensor_t<double, {entries}>({pdfBranch}), 10.); res = nominal_weight/nominal_pdf_cen*res; return res;")
    #df = df.Define(tensorName, f"auto res = wrem::clip_tensor(wrem::vec_to_tensor_t<double, {entries}>({pdfBranch}), 10.); res = nominal_weight*res; return res;")


    df = df.Define(tensorASName, "Eigen::TensorFixedSize<double, Eigen::Sizes<2>> res; "
            f"res(0) = {pdfInfo['alphas'][0]}; "
            f"res(1) = {pdfInfo['alphas'][1]}; "
            "return wrem::clip_tensor(res, 10.)")
    if cols:
        pdfHist = df.HistoBoost(pdfName if hname=="" else f"{hname}_{pdfName}", axes, [*cols, tensorName])
        alphaSHist = df.HistoBoost(f"alphaS002{pdfName}" if hname=="" else f"{hname}_alphaS002{pdfName}", axes, [*cols, tensorASName])
        return pdfHist, alphaSHist

    return []

def pdf_central_weight(dataset, pdfset):
    pdfInfo = pdf_info_map(dataset, pdfset)
    pdfBranch = pdfInfo["branch"]
    return f"{pdfBranch}[0]"

def define_scetlib_corr(df, weight_expr, helper, corr_type):
    modify_central_weight = corr_type not in ["altHist", "altHistNoUnc"]

    if modify_central_weight:
        df = df.Define("nominal_weight_uncorr", weight_expr)
    else:
        df = df.Define("nominal_weight", weight_expr)
        df = df.Alias("nominal_weight_uncorr", "nominal_weight")

    df = df.Define("scetlibWeight_tensor", helper, ["massVgen", "absYVgen", "ptVgen", "chargeVgen", "nominal_weight_uncorr"])
    df = df.Define("scetlibCentralWeight", "scetlibWeight_tensor(0)")

    if modify_central_weight:
        df = df.Alias("nominal_weight", "scetlibCentralWeight")
    return df

def make_scetlibCorr_hists(df, name, axes, cols, helper, corr_type):
    modify_central_weight = corr_type not in ["altHist", "altHistNoUnc"]
    skipUncertainties = corr_type in ["noUnc", "altHistNoUnc"]

    res = []
    if modify_central_weight:
        nominal_uncorr = df.HistoBoost(f"{name}_uncorr", axes, [*cols, "nominal_weight_uncorr"])
        res.append(nominal_uncorr)
        res.append(df.HistoBoost("weight_uncorr", [hist.axis.Regular(100, -2, 2)], ["nominal_weight_uncorr"]))

    if skipUncertainties:
        nominal = df.HistoBoost("scetlibCorr", axes, [*cols, "scetlibCentralWeight"])
        res.append(nominal)
    else:
        unc = df.HistoBoost("scetlibUnc" if name == "nominal" else f"{name}_scetlibUnc", axes, [*cols, "scetlibWeight_tensor"], tensor_axes=helper.tensor_axes)
        res.append(unc)

    return res

def scale_angular_moments(hist_moments_scales):
    # e.g. from arxiv:1708.00008 eq. 2.13, note A_0 is NOT the const term!
    scales = np.array([1., -10., 5., 10., 4., 4., 5., 5., 4.])

    hel_idx = hist_moments_scales.axes.name.index("helicity")
    scaled_vals = np.moveaxis(hist_moments_scales.view(flow=True), hel_idx, -1)
    hist_moments_scales[...] = np.moveaxis(scaled_vals, -1, hel_idx) 
    return hist_moments_scales

def moments_to_angular_coeffs(hist_moments_scales):
    s = hist.tag.Slicer()

    # select constant term, leaving dummy axis for broadcasting
    hist_moments_scales_m1 = hist_moments_scales[{"helicity" : s[-1j:-1j+1]}]

    vals = hist_moments_scales_m1.values(flow=True)
    vals = np.moveaxis(vals, hel_idx, -1)
    # replace zero values to avoid warnings
    norm_vals = np.where( vals==0., 1., vals)

    # e.g. from arxiv:1708.00008 eq. 2.13, note A_0 is NOT the const term!
    offsets = np.array([0., 4., 0., 0., 0., 0., 0., 0., 0.])

    # broadcasting happens right to left, so move to rightmost then move back
    hel_idx = hist_moments_scales.axes.name.index("helicity")
    coeffs = np.moveaxis(scale_angular_moments(hist_moments_scale, True).view(flow=True), hel_idx, -1) / norm_vals + offsets

    # replace values in zero-xsec regions (otherwise A0 is spuriously set to 4.0 from offset)
    coeffs = np.moveaxis(np.where(vals == 0., np.zeros_like(coeffs), coeffs), -1, hel_idx)

    hist_coeffs_scales = hist.Hist(*hist_moments_scales.axes, storage = hist_moments_scales._storage_type(), name = "hist_coeffs_scales",
        data = coeffs
    )


    return hist_coeffs_scales

def qcdByHelicityLabels():
    coeffs = ["const"]+[f"a{i}" for i in range(8)]
    scaleVars = ["muRmuF", "muR", "muF"]
    return  [f"{var}_{coeff}{t}" for var in scaleVars for t in ["Up", "Down"] for coeff in coeffs]

def qcdScaleNames():
    # Exclude central and extreme variations
    shifts = ["muRmuFDown", "muRDown", "", "muFDown", "", "muFUp", "muRUp", "", "muRmuFUp"]
    return ["_".join(["QCDscale", s]) if s != "" else s for s in shifts]

def massWeightNames(matches=None, wlike=False):
    central=10
    nweights=21
    names = [f"massShift{int(abs(central-i)*10)}MeV{'Down' if i < central else 'Up'}" for i in range(nweights)]
    
    if wlike:
        # This is the PDG uncertainty
        names.extend(["massShift2p1MeVDown", "massShift2p1MeVUp"])

    # If name is "" it won't be stored
    return [x if not matches or any(y in x for y in matches) else "" for x in names]

def pdfNames(cardTool, pdf, skipFirst=True):
    size = 101
    names = cardTool.mirrorNames(f"pdf{{i}}{pdf}", size)
    if skipFirst:
        names[0] = ""
        names[size] = ""
    # TODO: This is probably not needed anymore, check with low PU
    if False and pdf == "NNPDF31":
        names[size-2] = "pdfAlphas002Up"
        names[size-1] = "pdfAlphas002Down"
        # Drop the mirrored alphaS variations
        names[size*2-2] = ""
        names[size*2-1] = ""
    return names

def pdfNamesAsymHessian(entries):
    pdfNames = [""] # Skip central weight
    pdfNames.extend(["pdf{i}{shift}".format(i=int(j/2), shift="Up" if j % 2 else "Down") for j in range(entries-1)])
    return pdfNames

def pdfSymmetricShifts(hdiff, axis_name):
    sq = hh.multiplyHists(hdiff, hdiff)
    ss = sq[{axis_name : hist.sum}]
    rss = hh.sqrtHist(ss)
    return rss, rss

def pdfAsymmetricShifts(hdiff, axis_name):
    # Assuming that the last axis is the syst axis
    # TODO: add some check to verify this
    def shiftHist(vals, varis, hdiff, axis_name):
        hnew = hdiff.copy()[{axis_name : hist.sum}]
        vals, varis = hh.multiplyWithVariance(vals, vals, varis, varis)
        ss = np.stack((np.sum(vals, axis=-1), np.sum(varis, axis=-1)), axis=-1)
        hnew[...] = ss
        return hh.sqrtHist(hnew)

    # The error sets are ordered up,down,up,down...
    upshift = shiftHist(hdiff.values(flow=True)[...,1::2], hdiff.variances(flow=True)[...,1::2], hdiff, axis_name)
    downshift = shiftHist(hdiff.values(flow=True)[...,2::2], hdiff.variances(flow=True)[...,2::2], hdiff, axis_name)
    return upshift, downshift

def hessianPdfUnc(h, axis_name="tensor_axis_0", uncType="symHessian", scale=1.):
    symmetric = uncType == "symHessian"
    diff = hh.addHists(h, -1*h[{axis_name : 0}])*scale
    shiftFunc = pdfSymmetricShifts if symmetric else pdfAsymmetricShifts
    rssUp, rssDown = shiftFunc(diff, axis_name)
    hUp = hh.addHists(h[{axis_name : 0}], rssUp)
    hDown = hh.addHists(h[{axis_name : 0}], -1*rssDown)
    return hUp, hDown
