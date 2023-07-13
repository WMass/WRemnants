import ROOT
import pathlib
import hist
import narf
from utilities import rdf_tools
from utilities import common, logging
from utilities import boostHistHelpers as hh
import uproot
import numpy as np
import warnings
from functools import reduce
import time

logger = logging.child_logger(__name__)

narf.clingutils.Declare('#include "muon_calibration.h"')
narf.clingutils.Declare('#include "lowpu_utils.h"')

data_dir = f"{pathlib.Path(__file__).parent}/data/"

def make_muon_calibration_helpers(args,
        mc_filename=data_dir+"/calibration/correctionResults_v718_idealgeom_gensim.root", 
        data_filename=data_dir+"/calibration/correctionResults_v721_recjpsidata.root", 
        era = None):

    if args.muonCorrMC in ["trackfit_only", "lbl", "lbl_massfit"]:
        raise NotImplementedError(f"Muon calibrations for non-ideal geometry are currently not available! (needed for --muonCorrMC {args.muonCorrMC})")

    mc_helper = ROOT.wrem.CVHCorrector(mc_filename)
    data_helper = ROOT.wrem.CVHCorrector(data_filename)

    uncertainty_hist = get_dummy_uncertainties()
    uncertainty_hist_cpp = narf.hist_to_pyroot_boost(uncertainty_hist, tensor_rank = 2)
    # min gen pt = 9 GeV to avoid threshold effects
    # max weight = 10 to protect against outliers
    uncertainty_helper = ROOT.wrem.calibration_uncertainty_helper[type(uncertainty_hist_cpp)](ROOT.std.move(uncertainty_hist_cpp), 9., 10.)

    down_up_axis = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "downUpVar")
    uncertainty_helper.tensor_axes = (uncertainty_hist.axes["calvar"], down_up_axis)

    return mc_helper, data_helper, uncertainty_helper

def make_jpsi_crctn_helpers(args, make_uncertainty_helper=False):

    if args.muonCorrMC == "idealMC_massfit":
        mc_corrfile = "calibrationJMC_smeared_v718_nominal.root"
        logger.warning("You apply J/Psi massfit corrections on MC, this is currenlty not recommended!")
    elif args.muonCorrMC == "idealMC_lbltruth_massfit":
        mc_corrfile = "calibrationJMC_smeared_v718_nominalLBL.root"
        logger.warning("You apply J/Psi massfit corrections on MC, this is currenlty not recommended!")
    else:
        mc_corrfile = None

    if args.muonCorrData == "massfit":
        data_corrfile = "calibrationJDATA_ideal.root"
    elif args.muonCorrData == "lbl_massfit":
        data_corrfile = "calibrationJDATA_rewtgr_3dmap_LBL_MCstat.root" 
    else:
        data_corrfile = None

    mc_helper = make_jpsi_crctn_helper(filepath=f"{common.data_dir}/calibration/{mc_corrfile}") if mc_corrfile else None
    data_helper = make_jpsi_crctn_helper(filepath=f"{common.data_dir}/calibration/{data_corrfile}") if data_corrfile else None

    if make_uncertainty_helper:
        mc_unc_helper = make_jpsi_crctn_unc_helper(filepath=f"{common.data_dir}/calibration/{mc_corrfile}", n_eta_bins = 24) if mc_corrfile else None
        data_unc_helper = make_jpsi_crctn_unc_helper(filepath=f"{common.data_dir}/calibration/{data_corrfile}") if data_corrfile else None

        return mc_helper, data_helper, mc_unc_helper, data_unc_helper
    else:
        return mc_helper, data_helper

def make_muon_bias_helpers(args):
    # apply a bias to MC to correct for the nonclosure with data in the muon momentum scale calibration
    if args.biasCalibration is None: 
        return None

    if args.biasCalibration in ["parameterized", "A", "M"]:
        if args.muonCorrMC == "idealMC_lbltruth":
            filename = "calibrationAlignmentZ_after_LBL_v721"
        else:
            raise NotImplementedError(f"Did not find any parameterized closure file for --muonCorrMC {args.muonCorrMC}!")

        f = uproot.open(f"{data_dir}/closure/{filename}.root")

        A, M = [x.to_hist() for x in [f['AZ'], f['MZ']]]

        factor_A = 1
        factor_M = 1
        if args.biasCalibration == "A":
            factor_M = 0
        elif args.biasCalibration == "M":
            factor_A = 0

        # The bias correction follows the same parameterization as the J/Psi correction, but with e=0
        # We use the J/Psi correction helper and set e=0
        if (A.axes != M.axes):
            raise RuntimeError("A, M histograms have different axes!")
        else:
            axis_param = hist.axis.Regular(3, 0, 3, underflow = False, overflow = False, name = 'param')
            axis_cenval = hist.axis.Regular(1, 0, 1, name = 'cenval')
            hist_comb = hist.Hist(*A.axes, axis_param, axis_cenval, storage = hist.storage.Double())
            hist_comb.view()[...,0] = np.stack([x.values() for x in [A*factor_A, A*0, M/45.*factor_M]], axis = -1)

        hist_comb_cpp = narf.hist_to_pyroot_boost(hist_comb, tensor_rank = 2)
        helper = ROOT.wrem.JpsiCorrectionsRVecHelper[type(hist_comb_cpp).__cpp_name__](
            ROOT.std.move(hist_comb_cpp)
        )

    elif args.biasCalibration == "binned":
        # identify bias correction file name
        if args.smearing and args.muonCorrMC == "trackfit_only_idealMC":
            filename = "closureZ_smeared_v721"
            offset = -1
            logger.warning("You are using an outdated closure file!")
        elif args.smearing and args.muonCorrMC == "idealMC_lbltruth":
            filename = "closureZ_LBL_smeared_v721"
            offset = -1
        elif not args.smearing and args.muonCorrMC == "idealMC_massfit":
            filename = "closureZ_v718"
            offset = 0
            logger.warning("You are using an outdated closure file!")
        elif not args.smearing and args.muonCorrMC == "idealMC_lbltruth_massfit":
            filename = "closureZ_LBL_v718"
            offset = 0
            logger.warning("You are using an outdated closure file!")
        else:
            raise NotImplementedError(f"Did not find any closure file for muon momentum scale correction {args.muonCorrMC}"+str(" smeared!" if args.smearing else "!"))

        h2d = uproot.open(f"{data_dir}/closure/{filename}.root:closure").to_hist()
        # Drop the uncertainty because the Weight storage type doesn't play nice with ROOT

        h2d_nounc = hist.Hist(*h2d.axes, data=h2d.values(flow=True)+offset)
        h2d_std = hist.Hist(*h2d.axes, data=h2d.variances(flow=True)**0.5)
        # Set overflow to closest values
        h2d_nounc[hist.underflow,:][...] = h2d_nounc[0,:].view(flow=True)
        h2d_nounc[:,hist.underflow][...] = h2d_nounc[:,0].view(flow=True)
        h2d_nounc[hist.overflow,:][...] = h2d_nounc[-1,:].view(flow=True)
        h2d_nounc[:,hist.overflow][...] = h2d_nounc[:,-1].view(flow=True)

        h2d_std[hist.underflow,:][...] = h2d_std[0,:].view(flow=True)
        h2d_std[:,hist.underflow][...] = h2d_std[:,0].view(flow=True)
        h2d_std[hist.overflow,:][...] = h2d_std[-1,:].view(flow=True)
        h2d_std[:,hist.overflow][...] = h2d_std[:,-1].view(flow=True)

        h2d_cpp = narf.hist_to_pyroot_boost(h2d_nounc, tensor_rank=0)
        h2d_std_cpp = narf.hist_to_pyroot_boost(h2d_std, tensor_rank=0)

        helper = ROOT.wrem.BiasCalibrationHelper[type(h2d_cpp).__cpp_name__](ROOT.GetThreadPoolSize(), ROOT.std.move(h2d_cpp), ROOT.std.move(h2d_std_cpp))
    else:
        raise NotImplementedError(f"Correction for --bias-calibration {args.biasCalibration} not available")

    return helper

def make_muon_smearing_helpers(muonCorrMC="idealMC_lbltruth"):
    # this helper smears muon pT to match the resolution in data

    if muonCorrMC == "idealMC_lbltruth":
        filename = "smearing_LBL"
    elif muonCorrMC == "idealMC_massfit":
        filename = "smearing"
        logger.warning("You are using an outdated smearing file!")
    else:
        raise NotImplementedError(f"Did not find any smearing file for muon momentum scale correction {args.muonCorrMC}!")

    rfile = ROOT.TFile(f"{data_dir}/calibration/{filename}.root","READ")
    r2d = rfile.Get("smearing")

    helper = ROOT.wrem.SmearingHelper(ROOT.GetThreadPoolSize(), ROOT.std.move(r2d))

    return helper

def make_muon_calibration_helper_single(filename=data_dir+"/calibration/correctionResults_v718_idealgeom_gensim.root"):

    helper = ROOT.wrem.CVHCorrectorSingle[""](filename)
    return helper

def get_dummy_uncertainties():
    axis_eta = hist.axis.Regular(48, -2.4, 2.4, name = "eta")
    axis_calvar = hist.axis.Integer(0, 1, underflow=False, overflow=False, name = "calvar")
    axis_calparm = hist.axis.Integer(0, 4, underflow=False, overflow=False, name = "calparm")

    h = hist.Hist(axis_eta, axis_calvar, axis_calparm)

    # forward b-field-like
    h.values()[..., 0, 0] = 1e-4

    # set underflow and overflow to match boundaries
    h.values(flow=True)[0, ...] = h.values(flow=True)[1, ...]
    h.values(flow=True)[-1, ...] = h.values(flow=True)[-2, ...]

    return h

def make_jpsi_crctn_helper(filepath):
    f = uproot.open(filepath)

    # TODO: convert variable axis to regular if the bin width is uniform
    A, e, M = [x.to_hist() for x in [f['A'], f['e'], f['M']]]

    # TODO: make this into a function in utilities/boostHistHelpers
    if (A.axes != e.axes) or (e.axes != M.axes):
        raise RuntimeError("A, e, M histograms have different axes!")
    else:
        axis_param = hist.axis.Regular(3, 0, 3, underflow = False, overflow = False, name = 'param')
        hist_comb = hist.Hist(*A.axes, axis_param, storage = hist.storage.Double())
        hist_comb.view()[...] = np.stack([x.values() for x in [A, e, M]], axis = -1)

    hist_comb_cpp = narf.hist_to_pyroot_boost(hist_comb, tensor_rank = 1)
    jpsi_crctn_helper = ROOT.wrem.JpsiCorrectionsRVecHelper[type(hist_comb_cpp).__cpp_name__](
        ROOT.std.move(hist_comb_cpp)
    )
    return jpsi_crctn_helper

def make_jpsi_crctn_unc_helper(filepath, n_scale_params = 3, n_tot_params = 4, n_eta_bins = 48, scale = 1.0):
    f = uproot.open(filepath)
    cov = f['covariance_matrix'].to_hist()
    cov_scale_params = get_jpsi_scale_param_cov_mat(cov, n_scale_params, n_tot_params, n_eta_bins, scale)

    w,v = np.linalg.eigh(cov_scale_params)    
    var_mat = np.sqrt(w) * v
    axis_eta = hist.axis.Regular(n_eta_bins, -2.4, 2.4, name = 'eta')
    axis_scale_params = hist.axis.Regular(n_scale_params, 0, 1, name = 'scale_params')
    axis_scale_params_unc = hist.axis.Regular(
        n_eta_bins * n_scale_params, 0, 1,
        underflow = False, overflow = False,  name = 'unc'
    )
    hist_scale_params_unc = hist.Hist(axis_eta, axis_scale_params, axis_scale_params_unc)
    for i in range(n_eta_bins):
        lb, ub = i * n_scale_params, (i + 1) * n_scale_params
        hist_scale_params_unc.view()[i,...] = var_mat[lb:ub][:]
    hist_scale_params_unc_cpp = narf.hist_to_pyroot_boost(hist_scale_params_unc, tensor_rank = 2)
    jpsi_crctn_unc_helper = ROOT.wrem.JpsiCorrectionsUncHelper[type(hist_scale_params_unc_cpp).__cpp_name__](
        ROOT.std.move(hist_scale_params_unc_cpp)
    )
    jpsi_crctn_unc_helper.tensor_axes = (hist_scale_params_unc.axes['unc'], common.down_up_axis)
    return jpsi_crctn_unc_helper

def make_Z_non_closure_parametrized_helper(
    filepath = f"{data_dir}/closure/calibrationAlignmentZ_after_LBL_v721.root",
    n_eta_bins = 24, n_scale_params = 3, correlate = False
):
    f = uproot.open(filepath)
    M = f['MZ'].to_hist()
    A = f['AZ'].to_hist()

    axis_eta = hist.axis.Regular(n_eta_bins, -2.4, 2.4, name = 'eta')
    axis_scale_params = hist.axis.Regular(n_scale_params, 0, 1, name = 'scale_params')
    hist_non_closure = hist.Hist(axis_eta, axis_scale_params)
    hist_non_closure.view()[...,0] = A.values()
    hist_non_closure.view()[...,1] = np.zeros(n_eta_bins)
    hist_non_closure.view()[...,2] = M.values()

    hist_non_closure_cpp = narf.hist_to_pyroot_boost(hist_non_closure, tensor_rank = 1)
    if correlate:
        z_non_closure_helper = ROOT.wrem.ZNonClosureParametrizedHelperCorl[
            type(hist_non_closure_cpp).__cpp_name__,
            n_eta_bins
        ] (
            ROOT.std.move(hist_non_closure_cpp)
        )
        z_non_closure_helper.tensor_axes = tuple([common.down_up_axis])
        return z_non_closure_helper
    else:
        z_non_closure_helper = ROOT.wrem.ZNonClosureParametrizedHelper[
            type(hist_non_closure_cpp).__cpp_name__,
            n_eta_bins
        ] (
            ROOT.std.move(hist_non_closure_cpp)
        )
        z_non_closure_helper.tensor_axes = (
            hist.axis.Regular(n_eta_bins, 0, n_eta_bins, name = 'unc'),
            common.down_up_axis
        )
        return z_non_closure_helper

def make_Z_non_closure_binned_helper(
    filepath = f"{data_dir}/closure/closureZ_LBL_smeared_v721.root",
    n_eta_bins = 24, n_pt_bins = 5, correlate = False
):
    f = uproot.open(filepath)

    # TODO: convert variable axis to regular if the bin width is uniform
    hist_non_closure = f['closure'].to_hist()
    hist_non_closure_cpp = narf.hist_to_pyroot_boost(hist_non_closure)
    if correlate:
        z_non_closure_helper = ROOT.wrem.ZNonClosureBinnedHelperCorl[
            type(hist_non_closure_cpp).__cpp_name__,
            n_eta_bins,
            n_pt_bins
        ](
            ROOT.std.move(hist_non_closure_cpp)
        )
        z_non_closure_helper.tensor_axes = tuple([common.down_up_axis])
        return z_non_closure_helper
    else:
        z_non_closure_helper = ROOT.wrem.ZNonClosureBinnedHelper[
            type(hist_non_closure_cpp).__cpp_name__,
            n_eta_bins,
            n_pt_bins
        ](
            ROOT.std.move(hist_non_closure_cpp)
        )
        z_non_closure_helper.tensor_axes = (
            hist.axis.Regular(n_eta_bins, 0, n_eta_bins, name = 'unc_ieta'),
            hist.axis.Regular(n_pt_bins, 0, n_pt_bins, name = 'unc_ipt'),
            common.down_up_axis
        )
        print("tensor axes is: ++++++++++++++++++++++++++++++")
        print(z_non_closure_helper.tensor_axes)
        return z_non_closure_helper

# returns the cov mat of only scale parameters in eta bins, in the form of a 2D numpy array
# there are 3 scale params (A, e, M) + 3 resolution params for each eta bin in the jpsi calib file
def get_jpsi_scale_param_cov_mat(cov, n_scale_params = 3, n_tot_params = 4, n_eta_bins = 24, scale = 1.0):
    cov_dim = len(cov.axes[0].edges) - 1
    if cov_dim != n_tot_params * n_eta_bins:
        raise ValueError(
            f"dimension of the covariance matrix {cov_dim} doesn't match the input number of "
            f"total parameters {n_tot_params} times the number of eta bins {n_eta_bins}"
        )
    idx_first_param = np.arange(0, cov_dim, n_tot_params)
    idx_scale_params = np.sort(reduce(
        np.append, [(idx_first_param + i) for i in range(n_scale_params)]
    ))
    cov_scale_params = np.empty([n_scale_params * n_eta_bins, n_scale_params * n_eta_bins])
    for irow_scale_params, irow_all_params in enumerate(idx_scale_params):
        cov_scale_params[irow_scale_params, :] = scale * np.take(
            cov.values()[irow_all_params], idx_scale_params
        )
    return cov_scale_params

def crop_cov_mat(filepath, n_tot_params = 4, n_cropped_eta_bins = 2):
    f = uproot.open(filepath)
    filename = filepath.split("/")[-1]
    cov_mat = np.array(f['covariance_matrix'].to_hist().values())
    start_idx = n_tot_params * n_cropped_eta_bins
    cov_mat_cropped = cov_mat[start_idx:-start_idx, start_idx:-start_idx]
    fout = uproot.recreate(filename.split(".")[0] + "_cropped_eta" + ".root")
    fout['covariance_matrix'] = cov_mat_cropped 

def define_lblcorr_muons(df, cvh_helper, corr_branch="cvh"):

    # split the nested vectors
    df = df.Define("Muon_cvhmergedGlobalIdxs", "wrem::splitNestedRVec(Muon_cvhmergedGlobalIdxs_Vals, Muon_cvhmergedGlobalIdxs_Counts)")
    df = df.Define(f"Muon_{corr_branch}JacRef", f"wrem::splitNestedRVec(Muon_{corr_branch}JacRef_Vals, Muon_{corr_branch}JacRef_Counts)")

    df = df.Define("Muon_lblMom4Charge", cvh_helper, [f"Muon_{corr_branch}Pt", f"Muon_{corr_branch}Eta", f"Muon_{corr_branch}Phi", f"Muon_{corr_branch}Charge", "Muon_cvhmergedGlobalIdxs", f"Muon_{corr_branch}JacRef"])

    # split into individual vectors
    df = df.Define("Muon_lblPt", "ROOT::VecOps::RVec<float> res(Muon_lblMom4Charge.size()); std::transform(Muon_lblMom4Charge.begin(), Muon_lblMom4Charge.end(), res.begin(), [](const auto &x) { return x.first.Pt(); } ); return res;")
    df = df.Define("Muon_lblEta", "ROOT::VecOps::RVec<float> res(Muon_lblMom4Charge.size()); std::transform(Muon_lblMom4Charge.begin(), Muon_lblMom4Charge.end(), res.begin(), [](const auto &x) { return x.first.Eta(); } ); return res;")
    df = df.Define("Muon_lblPhi", "ROOT::VecOps::RVec<float> res(Muon_lblMom4Charge.size()); std::transform(Muon_lblMom4Charge.begin(), Muon_lblMom4Charge.end(), res.begin(), [](const auto &x) { return x.first.Phi(); } ); return res;")
    df = df.Define("Muon_lblCharge", "ROOT::VecOps::RVec<int> res(Muon_lblMom4Charge.size()); std::transform(Muon_lblMom4Charge.begin(), Muon_lblMom4Charge.end(), res.begin(), [](const auto &x) { return x.second; }); return res;")
    return df


def define_corrected_muons(df, cvh_helper, jpsi_helper, args, dataset, smearing_helper=None, bias_helper=None):
    if not (dataset.is_data or dataset.name in common.vprocs):
        corr_type = "none" 
    else:
        corr_type = args.muonCorrData if dataset.is_data else args.muonCorrMC

    # use continuous variable helix fits (cvh) with ideal or non ideal geometry 
    corr_branch = "cvhideal" if "idealMC" in corr_type else "cvh"

    # layer-by-layer corrections (lbl)
    if "lbl" in corr_type:
        df = define_lblcorr_muons(df, cvh_helper, corr_branch)
        muon = "Muon_lbl"
    elif corr_type != "none":
        muon = f"Muon_{corr_branch}"
    else:
        muon = "Muon"

    # calibrations from J/Psi massfits
    if "massfit" in corr_type:
        df = df.Define("Muon_jpsiCorrectedPt", jpsi_helper, [muon_var_name(muon, var) for var in ["pt", "eta", "charge"]])
        muon_pt = "Muon_jpsiCorrected"
    else:
        muon_pt = muon

    # Muon momentum scale resolution
    if not dataset.is_data and smearing_helper:
        df = df.Define("Muon_smearedPt", smearing_helper, ["rdfslot_", muon_var_name(muon_pt, "pt"), muon_var_name(muon, "eta")])
        muon_pt = "Muon_smeared"

    # Bias corrections from nonclosure
    if not dataset.is_data and bias_helper:
        if args.biasCalibration in ["parameterized", "A", "M"]:
            df = df.Define("Muon_biasedPtData", bias_helper, [muon_var_name(muon_pt, "pt"), muon_var_name(muon, "eta"), muon_var_name(muon, "charge")])
        else:
            df = df.Define("Muon_biasedPtData", bias_helper, ["rdfslot_", muon_var_name(muon_pt, "pt"), muon_var_name(muon, "eta")])

        # bias derived from data -> invert to apply on MC
        df = df.Define("Muon_biasedPt", "{0}/Muon_biasedPtData * {0}".format(muon_var_name(muon_pt, "pt")))

        muon_pt = "Muon_biased"

    for var in ["pt", "eta", "phi", "charge"]:
        mu_name = muon_pt if var == "pt" else muon
        df = df.Alias(muon_var_name("Muon_corrected", var), muon_var_name(mu_name, var))

    return df

def getColName_genFiltered_recoMuonSel(reco_sel = "goodMuons", require_prompt = True):
    genMatch_condition = "Prompt" if require_prompt else "FSonly"
    return f"{reco_sel}_genTruth_{genMatch_condition}"

def define_genFiltered_recoMuonSel(df, reco_sel = "goodMuons", require_prompt = True):
    col_name = getColName_genFiltered_recoMuonSel(reco_sel, require_prompt)
    require_prompt = "true" if require_prompt else "false"
    df = df.Define(
        col_name,
        (
            f"wrem::filterRecoMuonsByGenTruth("
            f"    {reco_sel},"
             "    Muon_genPartIdx," 
             "    GenPart_pdgId,"
             "    GenPart_statusFlags,"
             "    GenPart_status,"
            f"    {require_prompt}"
             ")"
        )
    )
    return df

def define_covMatFiltered_recoMuonSel(df, reco_sel = "goodMuons"):
    df = df.Redefine(
        f"{reco_sel}",
        (
            f"wrem::filterRecoMuonsByCovMat("
            f"    {reco_sel},"
             "    Muon_cvhMomCov_Vals,"
             "    Muon_cvhMomCov_Counts"
             ")"

        )
    )
    return df

def muon_var_name(mu_type, var):
        return mu_type+(f"_{var}" if mu_type == "Muon" else var.capitalize())

def define_matched_gen_muons_covMat(df, reco_sel = "goodMuons"):
    df = df.Define(
        f"{reco_sel}_covMat",
        (
            "wrem::getCovMatForSelectedRecoMuons<float>("
            f"    Muon_cvhMomCov_Vals, Muon_cvhMomCov_Counts, {reco_sel}"
            ");"
        )
    )
    return df

# this returns the kinematic variables of a collection of GEN muons matching a subset of RECO muons
def define_matched_gen_muons_kinematics(
    df,
    reco_sel = "goodMuons",
    kinematic_vars = ["pt", "eta", "phi", "charge"]
):
    for var in kinematic_vars:
        if var == "charge": var = "pdgId"
        df = df.Define(
            f"{reco_sel}_gen{var.capitalize()}",
            f"ROOT::VecOps::Take(GenPart_{var}, Muon_genPartIdx[{reco_sel}]);"
        )
        if var == "pdgId":
            df = df.Define(
                f"{reco_sel}_genCharge",
                (f"ROOT::VecOps::RVec<int> res({reco_sel}_gen{var.capitalize()}.size());"
                 "for (int i = 0; i < res.size(); i++) {"
                 f"    res[i] = ({reco_sel}_gen{var.capitalize()}[i] > 0 ? -1 : 1);"
                 "}"
                 "return res;"
                )
            )
    return df

def calculate_matched_gen_muon_kinematics(df, reco_sel = "goodMuons"):
    df = df.Define(
        f"{reco_sel}_genTheta",
        (f"ROOT::VecOps::RVec<double> res({reco_sel}_genEta.size());"
         "for (int i = 0; i < res.size(); i++) {"
         f"    res[i] = wrem::calculateTheta({reco_sel}_genEta[i]);"
         "}"
         "return res;"
        )
    )
    df = df.Define(
        f"{reco_sel}_genP",
        (f"ROOT::VecOps::RVec<double> res({reco_sel}_genPt.size());"
         "for (int i = 0; i < res.size(); i++) {"
         f"    res[i] = {reco_sel}_genPt[i] / std::sin({reco_sel}_genTheta[i]);"
         "}"
         "return res;"
        )
    )
    df = df.Define(
        f"{reco_sel}_genQop",
        (f"ROOT::VecOps::RVec<double> res({reco_sel}_genCharge.size());"
         "for (int i = 0; i < res.size(); i++) {"
         f"    res[i] = {reco_sel}_genCharge[i] / {reco_sel}_genP[i];"
         "}"
         "return res;"
        )
    )
    return df

def define_matched_genSmeared_muon_kinematics(df, reco_sel = "goodMuons"):
    df = df.Define(f"{reco_sel}_genSmearedQop", 
        (f"ROOT::VecOps::RVec<double> res({reco_sel}_genQop.size());"
         "for (int i = 0; i < res.size(); i++) {"
         "    res[i] = wrem::smearGenQop("
         f"        {reco_sel}_covMat[i],"
         f"        {reco_sel}_genQop[i]"
         "    );"
         "}"
         "return res;"
        )
    )
    df = df.Define(f"{reco_sel}_genSmearedPt", 
        (f"ROOT::VecOps::RVec<double> res({reco_sel}_genPt.size());"
         "for (int i = 0; i < res.size(); i++) {"
         f"    res[i] = {reco_sel}_genCharge[i] * std::sin({reco_sel}_genTheta[i]) / {reco_sel}_genSmearedQop[i];"
         "}"
         "return res;"
        )
    )
    df = df.Define(f"{reco_sel}_genSmearedEta", f"{reco_sel}_genEta")
    df = df.Define(f"{reco_sel}_genSmearedPhi", f"{reco_sel}_genPhi")
    df = df.Define(f"{reco_sel}_genSmearedCharge", f"{reco_sel}_genCharge")
    return df

def define_corrected_reco_muon_kinematics(df, muons="goodMuons", kinematic_vars = ["pt", "eta", "phi", "charge"], index=0):
    for var in kinematic_vars:
        df = df.Define(
            f"{muons}_{var.lower()}{index}",
            f"Muon_corrected{var.capitalize()}[{muons}][{index}]"
        )
    return df

def transport_smearing_weights_to_reco(resultdict, procs, nonClosureScheme = "A-M-separated"):
    time0 = time.time()
    
    hists_to_transport = ['muonScaleSyst_responseWeights_gensmear']
    if nonClosureScheme == "A-M-separated":
        hists_to_transport.append('Z_non_closure_parametrized_A_gensmear')
        hists_to_transport.append('Z_non_closure_parametrized_M_gensmear')
    if nonClosureScheme == "A-M-combined":
        hists_to_transport.append('Z_non_closure_parametrized_gensmear')
    if nonClosureScheme == "binned":
        hists_to_transport.append('Z_non_closure_binned_gensmear')
    if nonClosureScheme == "binned-plus-M":
        hists_to_transport.append('Z_non_closure_parametrized_M_gensmear')
        hists_to_transport.append('Z_non_closure_binned_gensmear')

    for proc in procs:

        if proc not in resultdict:
            logger.warning(f"Proc {proc} not found in output. Skipping smearing weights transportation")
            continue
        proc_hists = resultdict[proc]['output']

        nominal_reco = proc_hists['nominal'].get()
        if 'nominal_gen_smeared' in proc_hists.keys():
            nominal_gensmear = proc_hists['nominal_gen_smeared'].get()
        else:
            logger.warning(f"Histogram nominal_gen_smeared not found in {proc}")
            logger.warning("nuisances generated by smearing weights not transported to RECO kinematics!")
            return

        logger.debug(f"Process {proc}")

        for histname in hists_to_transport:
            if histname in proc_hists.keys():
                hist_gensmear = proc_hists[histname].get()
                reco_histname = 'nominal_' + histname[:-len('_gensmear')]
                hist_reco = hist.Hist(
                    *hist_gensmear.axes,
                    storage = hist_gensmear._storage_type()
                )
                
                bin_ratio = hh.divideHists(hist_gensmear, nominal_gensmear)
                hist_reco = hh.multiplyHists(nominal_reco, bin_ratio)
                proc_hists[reco_histname] = narf.ioutils.H5PickleProxy(hist_reco)
            else:
                logger.warning(f"Histogram {histname} not found in {proc}")
                logger.warning("nuisances generated by smearing weights not transported to RECO kinematics!")
                return
                
    logger.info(f"Transport smearing weights: {time.time() - time0}")

def make_alt_reco_and_gen_hists(df, results, nominal_axes, nominal_columns, matched_reco_sel = "goodMuons"):

    nominal_cols_gen = nominal_columns[:]
    nominal_cols_gen_smeared = nominal_columns[:]

    for col in ("pt", "eta", "charge"):
        idx = [i for i, x in enumerate(nominal_columns) if f"_{col}0" in x]
        if len(idx) != 1:
            logger.warning(f"None or more than one columns to match '_{col}0'! Do nothing here!")
            continue
        else:
            nominal_cols_gen[idx[0]] = f"{matched_reco_sel}_{col}0_gen"
            nominal_cols_gen_smeared[idx[0]] = f"{matched_reco_sel}_{col}0_gen_smeared"  

    results.append(df.HistoBoost("nominal_gen", nominal_axes, [*nominal_cols_gen, "nominal_weight"], storage=hist.storage.Double()))
    results.append(df.HistoBoost("nominal_gen_smeared", nominal_axes, [*nominal_cols_gen_smeared, "nominal_weight"], storage=hist.storage.Double()))

    return [nominal_cols_gen, nominal_cols_gen_smeared]
    
def define_lbl_corrections_jpsi_calibration_ntuples(df, helper):
    df = df.DefinePerSample("Muplus_charge", "1")
    df = df.DefinePerSample("Muminus_charge", "-1")

    df = df.Define("globalidxvint", "ROOT::VecOps::RVec<int>(globalidxv)")

    df = df.Define("Mupluscor_Mom4Charge", helper, ["Muplus_pt", "Muplus_eta", "Muplus_phi", "Muplus_charge", "globalidxvint", "Muplus_jacRef"])
    df = df.Define("Mupluscor_mom4", "Mupluscor_Mom4Charge.first")
    df = df.Define("Mupluscor_pt", "Mupluscor_Mom4Charge.first.Pt()")
    df = df.Define("Mupluscor_eta", "Mupluscor_Mom4Charge.first.Eta()")
    df = df.Define("Mupluscor_phi", "Mupluscor_Mom4Charge.first.Phi()")

    df = df.Define("Muminuscor_Mom4Charge", helper, ["Muminus_pt", "Muminus_eta", "Muminus_phi", "Muminus_charge", "globalidxvint", "Muminus_jacRef"])
    df = df.Define("Muminuscor_mom4", "Muminuscor_Mom4Charge.first")
    df = df.Define("Muminuscor_pt", "Muminuscor_Mom4Charge.first.Pt()")
    df = df.Define("Muminuscor_eta", "Muminuscor_Mom4Charge.first.Eta()")
    df = df.Define("Muminuscor_phi", "Muminuscor_Mom4Charge.first.Phi()")

    df = df.Define("Jpsicor_mom4", "ROOT::Math::PxPyPzEVector(Mupluscor_Mom4Charge.first) + ROOT::Math::PxPyPzEVector(Muminuscor_Mom4Charge.first)")

    df = df.Define("Jpsicor_pt", "Jpsicor_mom4.Pt()")
    df = df.Define("Jpsicor_eta", "Jpsicor_mom4.Eta()")
    df = df.Define("Jpsicor_phi", "Jpsicor_mom4.Phi()")
    df = df.Define("Jpsicor_mass", "Jpsicor_mom4.M()")

    return df

def define_passthrough_corrections_jpsi_calibration_ntuples(df):
    df = df.DefinePerSample("Muplus_charge", "1")
    df = df.DefinePerSample("Muminus_charge", "-1")

    df = df.Alias("Mupluscor_pt", "Muplus_pt")
    df = df.Alias("Mupluscor_eta", "Muplus_eta")
    df = df.Alias("Mupluscor_phi", "Muplus_phi")

    df = df.Alias("Muminuscor_pt", "Muminus_pt")
    df = df.Alias("Muminuscor_eta", "Muminus_eta")
    df = df.Alias("Muminuscor_phi", "Muminus_phi")

    df = df.Alias("Jpsicor_pt", "Jpsi_pt")
    df = df.Alias("Jpsicor_eta", "Jpsi_eta")
    df = df.Alias("Jpsicor_phi", "Jpsi_phi")
    df = df.Alias("Jpsicor_mass", "Jpsi_mass")

    df = df.Define("Mupluscor_mom4", "ROOT::Math::PtEtaPhiMVector(Mupluscor_pt, Mupluscor_eta, Mupluscor_phi, wrem::muon_mass)")
    df = df.Define("Muminuscor_mom4", "ROOT::Math::PtEtaPhiMVector(Muminuscor_pt, Muminuscor_eta, Muminuscor_phi, wrem::muon_mass)")
    df = df.Define("Jpsicor_mom4", "ROOT::Math::PtEtaPhiMVector(Jpsicor_pt, Jpsicor_eta, Jpsicor_phi, Jpsicor_mass)")

    return df
