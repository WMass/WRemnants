import argparse
import hist
import numpy as np
from rabbit import tensorwriter
import pickle
import lz4.frame
import rabbit
import rabbit.io_tools
from wremnants import theory_corrections
from wums import boostHistHelpers as hh
from utilities import common
import h5py
from utilities.io_tools import input_tools
import hist
import copy

def calculate_ais_from_helicities_hist(h_hels):
    """
    Calculate A_i histogram from helicities histogram.
    Assuming as input a histogram with the helicities as the last dimension,
    with the first element being the sigma_UL.
    Returns a the A_i's histogram.
    """

    h_hels = copy.deepcopy(h_hels) # pass by value
    vals = copy.copy(h_hels.values())
    vars = copy.copy(h_hels.variances()) # need to copy in case the storage is hist.storage.Double

    # A_i = sigma_i/sigma_UL
    num = vals[..., 1:]
    den = vals[..., 0][..., np.newaxis]
    vals[..., 1:] = np.where(
        den == 0,
        0.0,
        num / den
    )

    # treat these as poisson uncorrelated rv's
    if np.any(vars):

        print(np.max(vars[..., 0][..., np.newaxis]**0.5/den))
        print(np.max(vars[..., 1:]**0.5/num))

        vars[...,1:] = np.where(
            vals[...,0][..., np.newaxis] == 0,
            0.0,
            den**-2 * vars[..., 1:] + \
            num**2 * den**-4 * vars[..., 0][..., np.newaxis]
        )
        
    h_out = hist.Hist(
        *h_hels.axes[:-1],
        hist.axis.Integer(0,8,name="ais"),
        storage=hist.storage.Weight() if np.any(vars) else hist.storage.Double()
    )
    h_out.values()[...] = vals[...,1:]
    if np.any(vars): h_out.variances()[...] = vars[...,1:]

    return h_out


parser = argparse.ArgumentParser()
parser.add_argument(
    "infile",
    type=str,
    help="Input unfolded fit result",
)
parser.add_argument(
    "--fitresultModel",
    type=str,
    default='Select helicitySig:slice(0,1)'
)
parser.add_argument(
    "--predGenerator",
    type=str,
    default=f"scetlib_dyturbo",
    help="Generator used for the sigmaUL predictions. "
    "Expect the prediction file to be named as <generator>CorrZ.pkl.lz4. "
    "Expect the prediction histogram to be named as <generator>_hist",
)
parser.add_argument(
    "--predAiFile",
    type=str,
    default=f"/ceph/submit/data/group/cms/store/user/lavezzo/alphaS//250627_angularCoefficients/w_z_helicity_xsecs_scetlib_dyturboCorr_maxFiles_m1_alphaSunfoldingBinning.hdf5",
    help="Gen file used for the Ai predictions."
    "Will be stitched with the --predGenerator file."
)
parser.add_argument(
    "--angularCoeffs",
    action="store_true",
    default=False,
    help="Fit the angular coefficients."
    "Predictions of the Ai's from --predAiFile."
    "Predictions of the sigma_UL from infile."
)
parser.add_argument("-o", "--output", default="./", help="output directory")
parser.add_argument("--outname", default="carrot", help="output file name")
parser.add_argument(
    "--postfix",
    default=None,
    type=str,
    help="Postfix to append on output file name",
)
parser.add_argument(
    "--sparse",
    default=False,
    action="store_true",
    help="Make sparse tensor",
)
parser.add_argument(
    "--symmetrizeAll",
    default=False,
    action="store_true",
    help="Make fully symmetric tensor",
)
parser.add_argument(
    "--skipMaskedChannels",
    default=False,
    action="store_true",
    help="Skip adding masked channels",
)
parser.add_argument(
    "--systematicType",
    choices=["log_normal", "normal"],
    default="log_normal",
    help="probability density for systematic variations",
)

args = parser.parse_args()


# Build tensor
writer = tensorwriter.TensorWriter(
    sparse=args.sparse,
    systematic_type='normal' if args.angularCoeffs else args.systematicType,
    allow_negative_expectation=args.angularCoeffs
)

# load in data histogram and covariance matrix
fitresult, meta = rabbit.io_tools.get_fitresult(
    args.infile, result='asimov', meta=True
)

if args.angularCoeffs:
    try:
        h_data_ai = fitresult['physics_models'][args.fitresultModel]['channels']['AngularCoefficients ch0_masked ptVGen:rebin(0,3,6,9,12,16,20,24,28,33,44)_ch0_masked']['hist_postfit_inclusive'].get()
        h_data = fitresult['physics_models'][args.fitresultModel]['channels']['Select helicitySig:slice(0,1)_ch0_masked']['hist_postfit_inclusive'].get()[:,:,0]
        h_data_cov = fitresult['physics_models'][args.fitresultModel]['hist_postfit_inclusive_cov'].get()
    except KeyError:
        raise KeyError(f"Couldn't find {args.fitresultModel}. Available physics models are: {",".join(fitresult['physics_models'].keys())}")
    writer.add_channel(h_data_ai.axes, "chAis")
    writer.add_data(h_data_ai, "chAis")
    writer.add_channel(h_data.axes, "chSigmaUL")
    writer.add_data(h_data, "chSigmaUL")
else:
    print( fitresult['physics_models'].keys())
    h_data = fitresult['physics_models'][args.fitresultModel]['channels']['ch0_masked']['hist_postfit_inclusive'].get()[:,:,0] # grabbing the unpolarized term
    h_data_cov = fitresult['physics_models'][args.fitresultModel]['hist_postfit_inclusive_cov'].get()
    writer.add_channel(h_data.axes, "chSigmaUL")
    writer.add_data(h_data, "chSigmaUL")
writer.add_data_covariance(h_data_cov) # run with --externalCovariance

# scaling for background and variations
lumi = 16800
scale = lumi

# add background
h_sig = theory_corrections.load_corr_hist(
    f"{common.data_dir}/TheoryCorrections/{args.predGenerator}CorrZ.pkl.lz4",
    "Z",
    f"{args.predGenerator}_hist"
)[{'vars': 'pdf0'}].project("qT", "absY")
h_sig = hh.rebinHist(h_sig, 'qT', h_data.axes[0].edges)
h_sig = hh.rebinHist(h_sig, 'absY', h_data.axes[1].edges)
h_sig = h_sig * scale
writer.add_process(h_sig, "Zmumu", "chSigmaUL", signal=False)

# if set, load in the helicity cross sections predictions from MINNLO
if args.angularCoeffs:
    with h5py.File(args.predAiFile, "r") as ff:
        inputs = input_tools.load_results_h5py(ff)
        h_sig_hels = inputs['Z']
    h_sig_hels = h_sig_hels[{'muRfact': 1.0j}][{'muFfact': 1.0j}][{'chargeVgen': 0.0j}][{'massVgen': 90j}]
    h_sig_hels = h_sig_hels.project("ptVgen", "absYVgen", "helicity") # re-order the axes
    h_sig_hels *= scale
    h_sig_hels = hh.rebinHist(h_sig_hels, "ptVgen", h_data_ai.axes[0].edges)
    h_sig_ais = calculate_ais_from_helicities_hist(h_sig_hels)
    writer.add_process(h_sig_ais, "Zmumu", "chAis", signal=False)

# add systematic variations

# alphaS variation
if args.predGenerator == "scetlib_dyturbo" or True:
    theory_corrs = [
        "scetlib_dyturboCT18Z_pdfas",
    ]
    corr_helpers = theory_corrections.load_corr_helpers(
        "Z", theory_corrs, make_tensor=False, minnlo_ratio=False
    )
    h = corr_helpers['Z']['scetlib_dyturboCT18Z_pdfas']*scale
    h = h.project("qT", "absY", "vars")
    h = hh.rebinHist(h, 'qT', h_data.axes[0].edges)
    h = hh.rebinHist(h, 'absY', h_data.axes[1].edges)
    h_syst_down = h[{'vars': 2}]
    h_syst_up = h[{'vars': 1}]

    if args.angularCoeffs:
        with h5py.File(args.predAiFile.replace("w_z_helicity_xsecs", "w_z_gen_dists"), "r") as ff:
            inputs = input_tools.load_results_h5py(ff)
            h_hels = inputs['ZmumuPostVFP']['output']['nominal_gen_helicity_nominal_gen_pdfCT18ZalphaS002'].get()
        h_hels = h_hels.project("ptVgen", "absYVgen","alphasVar", "helicity")*scale
        h_hels = hh.rebinHist(h_hels, "ptVgen", h_data_ai.axes[0].edges)
        h_ais = calculate_ais_from_helicities_hist(h_hels)
        h_syst_up_ais = h_ais[{'alphasVar': 'as0120'}]
        h_syst_down_ais = h_ais[{'alphasVar': 'as0116'}] 
        writer.add_systematic(
            [h_syst_up_ais, h_syst_down_ais],
            'pdfAlphaS',
            "Zmumu",
            "chAis",
            noi=True,
            constrained=False,
            symmetrize='average',
            kfactor=1.5/2.0
        )

    writer.add_systematic(
        [h_syst_up, h_syst_down],
        'pdfAlphaS',
        "Zmumu",
        "chSigmaUL",
        noi=True,
        constrained=False,
        symmetrize='average',
        kfactor=1.5/2.0
    )
# TODO need to check this with Kenneth
elif args.predGenerator == "scetlib_nnlojetN4p0LLN3LO":
    # alphaS variation N4p0LLN3LO
    fname = f"{common.data_dir}/TheoryCorrections//{args.predGenerator}_pdfasCorrZ.pkl.lz4"
    corrh = theory_corrections.load_corr_hist(
        fname,
        "Z",
        f"{args.predGenerator}_pdfas_minnlo_ratio"
    ).project("qT", "absY", "vars")
    minnlo_ref = theory_corrections.load_corr_hist(
        fname,
        "Z",
        "minnlo_ref_hist"
    ).project("qT", "absY")
    print(corrh)
    print(minnlo_ref)
    vals = np.einsum('ijk,ij->ijk', corrh.values(), minnlo_ref.values())
    corrh[...] = vals * scale
    corrh = hh.rebinHist(corrh, 'qT', h_data.axes[0].edges)
    corrh = hh.rebinHist(corrh, 'absY', h_data.axes[1].edges)
    h_syst_up = corrh[{'vars': 2}]
    h_syst_down = corrh[{'vars': 1}]
    writer.add_systematic(
        [h_syst_up, h_syst_down],
        'pdfAlphaS',
        "Zmumu",
        "chSigmaUL",
        noi=True,
        constrained=False,
        symmetrize='average',
        #kfactor=1.5/2.0
    )
else:
    raise Exception("No valid configuration found for alphaS variation.")

# pythia showering uncertainties
if args.angularCoeffs:
    print("Now at pythia_shower_kt")
    corr_coeffs = theory_corrections.make_qcd_uncertainty_helper_by_helicity(
        is_z=True,
        filename=args.predAiFile,
        rebi_ptVgen=False,
        return_tensor=False,
    )
    corr_coeffs = corr_coeffs[{'vars': 'pythia_shower_kt'}]*scale
    corr_coeffs = corr_coeffs.project("ptVgen", "absYVgen", "corr", "helicity")
    corr_coeffs = hh.rebinHist(corr_coeffs, 'ptVgen', h_data_ai.axes[0].edges)
    corr_coeffs = hh.rebinHist(corr_coeffs, 'absYVgen', h_data_ai.axes[1].edges)
    corr_coeffs = hh.divideHists(corr_coeffs[{'corr': 1}],corr_coeffs[{'corr': 0}])
    corr_coeffs = hh.multiplyHists(corr_coeffs, h_sig_hels)
    corr_coeffs_ais = calculate_ais_from_helicities_hist(corr_coeffs)
    writer.add_systematic(
        corr_coeffs_ais,
        "pythia_shower_kt",
        "Zmumu",
        "chAis",
        mirror=True, 
        groups=["helicity_shower_kt", "angularCoeffs", "theory"],
    )

print(f"Now at {args.predGenerator}")
theory_corrs = [
    args.predGenerator,
]
corr_helpers = theory_corrections.load_corr_helpers(
    "Z", theory_corrs, make_tensor=False, minnlo_ratio=False
)
h = corr_helpers['Z'][args.predGenerator]*scale
h = h.project("qT", "absY", "vars")
h = hh.rebinHist(h, 'qT', h_data.axes[0].edges)
h = hh.rebinHist(h, 'absY', h_data.axes[1].edges)

# correlated NP uncertainties
corr_NP_uncs = [
    ["Lambda20.25","Lambda2-0.25", "chargeVgenNP0scetlibNPZLambda2"],
    ["Lambda4.01","Lambda4.16", "chargeVgenNP0scetlibNPZLambda4"],
    ["Delta_Lambda2-0.02","Delta_Lambda20.02", "chargeVgenNP0scetlibNPZDelta_Lambda2"],
]
for var in corr_NP_uncs:
    print(var)

    writer.add_systematic(
        [h[{'vars': var[1]}],  h[{'vars': var[0]}]],
        var[2],
        "Zmumu",
        "chSigmaUL",
        symmetrize='average', 
        groups=["resumTNP", "resum", "pTModeling", "theory"]
    )

# gamma NP uncertainties
gamma_NP_uncs = [
    ["omega_nu0.5","c_nu-0.1-omega_nu0.5", "scetlibNPgamma"],
]
for var in gamma_NP_uncs:
    print(var)

    writer.add_systematic(
        [h[{'vars': var[1]}],  h[{'vars': var[0]}]],
        var[2],
        "Zmumu",
        "chSigmaUL",
        symmetrize='average', 
        groups=["resumTNP", "resum", "pTModeling", "theory"]
    )

# TNP
TNP_uncs = [
    ["gamma_cusp-1.","gamma_cusp1."],
    ["gamma_mu_q-1.","gamma_mu_q1."],
    ["gamma_nu-1.", "gamma_nu1."],
    ["h_qqV-1.", "h_qqV1."],
    ["s-1.", "s1."],
    ["b_qqV-0.5", "b_qqV0.5"],
    ["b_qqbarV-0.5", "b_qqbarV0.5"],
    ["b_qqS-0.5", "b_qqS0.5"],
    ["b_qqDS-0.5", "b_qqDS0.5"],
    ["b_qg-0.5", "b_qg0.5"],
]
for var in TNP_uncs:
    print(var)

    writer.add_systematic(
        [h[{'vars': var[1]}],  h[{'vars': var[0]}]],
        "resumTNP_" + var[0].split('-')[0],
        "Zmumu",
        "chSigmaUL",
        symmetrize='average', 
        groups=["resumTNP", "resum", "pTModeling", "theory"]
    )

# transition FO scale uncertainties
transition_FO_uncs = [
    ["transition_points0.2_0.35_1.0", "transition_points0.2_0.75_1.0", "resumTransitionZ"],
    ["renorm_scale_pt20_envelope_Down", "renorm_scale_pt20_envelope_Up", "resumFOScaleZ"],
]
for var in transition_FO_uncs:
    print(var)

    writer.add_systematic(
        [h[{'vars': var[1]}],  h[{'vars': var[0]}]],
        var[2],
        "Zmumu",
        "chSigmaUL",
        symmetrize='quadratic',
        groups=["resumTransitionFOScale", "resum", "pTModeling", "theory"],
    )

# PDF uncertainties
print("Now at scetlib_dyturboCT18ZVars")
theory_corrs = [
    "scetlib_dyturboCT18ZVars",
]
corr_helpers = theory_corrections.load_corr_helpers(
    "Z", theory_corrs, make_tensor=False, minnlo_ratio=False
)
h = corr_helpers['Z']['scetlib_dyturboCT18ZVars']*scale
h = h.project("qT", "absY", "vars")
h = hh.rebinHist(h, 'qT', h_data.axes[0].edges)
h = hh.rebinHist(h, 'absY', h_data.axes[1].edges)
for ivar in range(1, len(h.axes[-1]), 2):
    print(ivar)
    h_syst_up = h[{'vars': ivar}]
    h_syst_down = h[{'vars': ivar + 1}]
    writer.add_systematic(
        [h_syst_up, h_syst_down],
        f"pdf{int((ivar+1)/2)}CT18Z",
        "Zmumu",
        "chSigmaUL",
        symmetrize='quadratic',
        kfactor=1/1.645,
        groups=["pdfCT18Z", f"pdfCT18ZNoAlphaS", "theory"],
    )

# quark mass effects
print("Now at MSHT20mcrangeCorrZ")
theory_corrs = [
    "scetlib_dyturboMSHT20mcrange",
    "scetlib_dyturboMSHT20mbrange"
]
corr_helpers = theory_corrections.load_corr_helpers(
    "Z", theory_corrs, make_tensor=False, minnlo_ratio=False
)
h = corr_helpers['Z']['scetlib_dyturboMSHT20mbrange']*scale  
h = h.project("qT", "absY", "vars")
h = hh.rebinHist(h, 'qT', h_data.axes[0].edges)
h = hh.rebinHist(h, 'absY', h_data.axes[1].edges)  
writer.add_systematic(
    [h[{'vars': 1}], h[{'vars': -1}]],
    "pdfMSHT20mbrange",
    "Zmumu",
    "chSigmaUL",
    symmetrize='quadratic',
    groups=["bcQuarkMass", "pTModeling", "theory"],
)
h = corr_helpers['Z']['scetlib_dyturboMSHT20mcrange']*scale  
h = h.project("qT", "absY", "vars")
h = hh.rebinHist(h, 'qT', h_data.axes[0].edges)
h = hh.rebinHist(h, 'absY', h_data.axes[1].edges) 
writer.add_systematic(
    [h[{'vars': 1}], h[{'vars': -1}]],
    "pdfMSHT20mcrange",
    "Zmumu",
    "chSigmaUL",
    symmetrize='quadratic',
    groups=["bcQuarkMass", "pTModeling", "theory"],
)

# EW corrections
# TODO unclear how to handle these
# print("Now at powhegFOEW")
# theory_corrs = [
#     "powhegFOEW",
# ]
# corr_helpers = theory_corrections.load_corr_helpers(
#     "Z", theory_corrs, make_tensor=False
# )
# h = corr_helpers['Z']['powhegFOEW'].project("massVlhe", "absYVlhe", "weak")
# h = h[{'massVlhe': slice(60j, 120j)}]
# h_sig_Q_absY_qT = theory_corrections.load_corr_hist(args.predFile, "Z", f"{args.predGenerator}_hist")[{'vars': 'pdf0'}].project("Q", "absY", "qT")
# var_qT_absY_Q = np.einsum('ijk,ijm->ijkm', h.values(), h_sig_Q_absY_qT.values())
# h_var = hist.Hist(
#     h_sig_Q_absY_qT.axes[0], # mass
#     h_sig_Q_absY_qT.axes[1], # absY
#     h.axes[2],               # EW variations
#     h_sig_Q_absY_qT.axes[2], # qT
# )
# h_var[...] = var_qT_absY_Q
# h_var = h_var.project("qT", "absY", "weak") # now integrate over mass
# h_var = hh.rebinHist(h_var, 'qT', h_data.axes[0].edges)
# h_var = hh.rebinHist(h_var, 'absY', h_data.axes[1].edges)
# writer.add_systematic(
#     h[{'weak': 'weak_ps'}],
#     'weak_ps',
#     "Zmumu",
#     "chSigmaUL",
#     mirror=True,
#     groups=[f"theory_ew_virtZ_scheme", "theory_ew", "theory"],
# )
# writer.add_systematic(
#     h[{'weak': 'weak_aem'}],
#     'weak_aem',
#     "Zmumu",
#     "chSigmaUL",
#     mirror=True,
#     groups=[f"theory_ew_virtZ_scheme", "theory_ew", "theory"],
# )
# writer.add_systematic(
#     h[{'weak': 'weak_default'}],
#     'weak_default',
#     "Zmumu",
#     "chSigmaUL",
#     mirror=True,
#     groups=[f"theory_ew_virtZ_corr", "theory_ew", "theory"],
# )

# ISR corrections
print("Now at pythiaew_ISR")
theory_corrs = [
    "pythiaew_ISR",
]
fname = f"{common.data_dir}/TheoryCorrections/pythiaew_ISRCorrZ.pkl.lz4"
h = theory_corrections.load_corr_hist(fname, "Z", f"{theory_corrs[0]}_minnlo_ratio") # not actually minnlo ratio
h = h.project("ptVgen", "absYVgen", "systIdx")
h = h[{'systIdx': 1}] 
h = hh.rebinHist(h, 'ptVgen', h_data.axes[0].edges)
h = hh.rebinHist(h, 'absYVgen', h_data.axes[1].edges)
h[...] = h.values() * h_sig.values() # TODO does this make sense?
writer.add_systematic(
    h,
    "pythiaew_ISRCorr0",
    "Zmumu",
    "chSigmaUL",
    mirror=True,
    kfactor=2.0,
    groups=[f"theory_ew_pythiaew_ISR", "theory_ew", "theory"],
)

# write output
directory = args.output
if directory == "":
    directory = "./"
filename = args.outname
if args.postfix:
    filename += f"_{args.postfix}"
writer.write(outfolder=directory, outfilename=filename)
print(f"Written to {directory}/{filename}.hdf5")