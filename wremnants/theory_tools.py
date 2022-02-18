import ROOT
import hist
import numpy as np

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

def define_prefsr_vars(df):
    df = df.Define("prefsrLeps", "wrem::prefsrLeptons(GenPart_status, GenPart_statusFlags, GenPart_pdgId, GenPart_genPartIdxMother)")
    df = df.Define("genl", "ROOT::Math::PtEtaPhiMVector(GenPart_pt[prefsrLeps[0]], GenPart_eta[prefsrLeps[0]], GenPart_phi[prefsrLeps[0]], GenPart_mass[prefsrLeps[0]])")
    df = df.Define("genlanti", "ROOT::Math::PtEtaPhiMVector(GenPart_pt[prefsrLeps[1]], GenPart_eta[prefsrLeps[1]], GenPart_phi[prefsrLeps[1]], GenPart_mass[prefsrLeps[1]])")
    df = df.Define("genV", "ROOT::Math::PxPyPzEVector(genl)+ROOT::Math::PxPyPzEVector(genlanti)")
    df = df.Define("ptVgen", "genV.pt()")
    df = df.Define("massVgen", "genV.mass()")
    df = df.Define("yVgen", "genV.Rapidity()")
    df = df.Define("absYVgen", "std::fabs(yVgen)")
    df = df.Define("chargeVgen", "GenPart_pdgId[prefsrLeps[0]] + GenPart_pdgId[prefsrLeps[1]]")
    df = df.Define("scaleWeights_tensor", "wrem::makeScaleTensor(LHEScaleWeight);")
    df = df.Define("csSineCosThetaPhi", "wrem::csSineCosThetaPhi(genl, genlanti)")

    return df

def moments_to_angular_coeffs(hist_moments_scales):
    s = hist.tag.Slicer()

    # select constant term, leaving dummy axis for broadcasting
    hist_moments_scales_m1 = hist_moments_scales[{"helicity" : s[-1j:-1j+1]}]

    vals = hist_moments_scales_m1.values(flow=True)

    # replace zero values to avoid warnings
    norm_vals = np.where( vals==0., 1., vals)

    # e.g. from arxiv:1708.00008 eq. 2.13
    offsets = np.array([0., 4., 0., 0., 0., 0., 0., 0., 0.])
    scales = np.array([1., -10., 5., 10., 4., 4., 5., 5., 4.])

    # for broadcasting
    offsets = offsets[:, np.newaxis, np.newaxis]
    scales = scales[:, np.newaxis, np.newaxis]

    view = hist_moments_scales.view(flow=True)

    coeffs = scales*view / norm_vals + offsets

    # replace values in zero-xsec regions (otherwise A0 is spuriously set to 4.0 from offset)
    coeffs = np.where(vals == 0., 0.*view, coeffs)

    hist_coeffs_scales = hist.Hist(*hist_moments_scales.axes, storage = hist_moments_scales._storage_type(), name = "hist_coeffs_scales",
        data = coeffs
    )


    return hist_coeffs_scales
