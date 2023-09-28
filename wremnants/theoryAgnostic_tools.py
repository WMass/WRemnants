from utilities import differential
import wremnants
from wremnants import syst_tools, theory_tools, logging, helicity_utils
from copy import deepcopy
import hist

logger = logging.child_logger(__name__)

def select_fiducial_space(df, ptVgenMax, absYVgenMax, accept=True):
    # might need to account for some Ai which we neglect, we may have an out-of-acceptance per helicity state
    selection = f"ptVgen < {ptVgenMax} && absYVgen < {absYVgenMax}"
    df = df.Define("fiducial", selection)
    if accept:
        df = df.Filter("fiducial")
        logger.debug(f"Theory agnostic fiducial cut: {selection}")
    else:
        df = df.Filter("fiducial == 0")
        logger.debug(f"Theory agnostic fiducial cut (out-of-acceptance): not ({selection})")
    return df

def define_helicity_weights(df):
    # define the helicity tensor, here nominal_weight will only have theory weights, no experimental pieces, it is defined in theory_tools.define_theory_weights_and_corrs
    weightsByHelicity_helper = wremnants.makehelicityWeightHelper()
    df = df.Define("helWeight_tensor", weightsByHelicity_helper, ["massVgen", "yVgen", "ptVgen", "chargeVgen", "csSineCosThetaPhi", "nominal_weight"])
    df = df.Define("nominal_weight_helicity", "wrem::scalarmultiplyHelWeightTensor(nominal_weight,helWeight_tensor)")
    return df

def add_xnorm_histograms(results, df, args, dataset_name, corr_helpers, qcdScaleByHelicity_helper, theoryAgnostic_axes, theoryAgnostic_cols):
    # add histograms before any selection
    axis_helicity = helicity_utils.axis_helicity_multidim
    df_xnorm = df
    df_xnorm = df_xnorm.DefinePerSample("exp_weight", "1.0")
    df_xnorm = theory_tools.define_theory_weights_and_corrs(df_xnorm, dataset_name, corr_helpers, args)
    # define the helicity tensor, here nominal_weight will only have theory weights, no experimental pieces, it is defined in theory_tools.define_theory_weights_and_corrs
    df_xnorm = define_helicity_weights(df_xnorm)
    df_xnorm = df_xnorm.DefinePerSample("xnorm", "0.5")
    axis_xnorm = hist.axis.Regular(1, 0., 1., name = "count", underflow=False, overflow=False)
    xnorm_axes = [axis_xnorm, *theoryAgnostic_axes]
    xnorm_cols = ["xnorm", *theoryAgnostic_cols]
    xnormByHelicity = df_xnorm.HistoBoost("xnorm", xnorm_axes, [*xnorm_cols, "nominal_weight_helicity"], tensor_axes=[axis_helicity])
    results.append(xnormByHelicity)
    if not args.onlyMainHistograms:
        syst_tools.add_theory_hists(results, df_xnorm, args, dataset_name, corr_helpers, qcdScaleByHelicity_helper, xnorm_axes, xnorm_cols, base_name="xnorm", for_wmass=True, addhelicity=True)
    else:
        #FIXME: hardcoded to keep mass weights (this would be done in add_theory_hists) but removing all other theory systs
        df_xnorm = syst_tools.define_mass_weights(df_xnorm, dataset_name)
        syst_tools.add_massweights_hist(results, df_xnorm, xnorm_axes, xnorm_cols, base_name="xnorm", proc=dataset_name, addhelicity=True)
