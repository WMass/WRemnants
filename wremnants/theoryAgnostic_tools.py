import hist

from utilities import common
from wremnants import helicity_utils, syst_tools, theory_tools
from wums import logging

logger = logging.child_logger(__name__)


def select_fiducial_space(
    df, ptVgenMax, absYVgenMax, accept=True, select=True, usePtOverM=False
):
    # accept defines the selection for in-acceptance (IA) or out-of-acceptance (OOA)
    # select is needed to actually apply the selection. For --poiAsNoi one integrates over the gen bins to build the nominal histogram in setupRabbit/CardTool.py,
    # so all bins must be kept including overflows, and thus the explicit cut must be removed, although this is only for IA, since OOA is usually built as an independent histogram)
    # In the future the OOA might be build from the overflow bins directly (it might be possible to define multiple pieces too)
    ptvar = "qtOverQ" if usePtOverM else "ptVgen"
    selection = f"{ptvar} < {ptVgenMax} && absYVgen < {absYVgenMax}"
    df = df.Define("fiducial", selection)
    if accept:
        if select:
            df = df.Filter("fiducial")
            logger.debug(f"Theory agnostic fiducial cut: {selection}")
        else:
            logger.debug(
                f"Theory agnostic fiducial cut not explicitly applied to fill overflow bins, it was: {selection}"
            )
    else:
        df = df.Filter("fiducial == 0")
        logger.debug(
            f"Theory agnostic fiducial cut (out-of-acceptance): not ({selection})"
        )
    return df


def define_helicity_weights(df, is_z=False):
    helper = helicity_utils.make_helicity_weight_helper(
        is_z,
        filename=f"{common.data_dir}/angularCoefficients/w_z_helicity_xsecs_theoryAgnosticBinning_scetlib_dyturboCorr_maxFiles_m1.hdf5",
    )
    df = helicity_utils.define_helicity_weights(df, helper)
    return df


def add_xnorm_histograms(
    results,
    df,
    args,
    dataset_name,
    corr_helpers,
    qcdScaleByHelicity_helper,
    theoryAgnostic_axes,
    theoryAgnostic_cols,
):
    # add histograms before any selection
    axis_helicity = helicity_utils.axis_helicity_multidim
    df_xnorm = df
    df_xnorm = df_xnorm.DefinePerSample("exp_weight", "1.0")
    df_xnorm = theory_tools.define_theory_weights_and_corrs(
        df_xnorm, dataset_name, corr_helpers, args
    )
    # define the helicity tensor, here nominal_weight will only have theory weights, no experimental pieces, it is defined in theory_tools.define_theory_weights_and_corrs
    # TODO: this does not look correct since theoryAgnostic_axes only contain only polarization independent observables (pTV, YV, mV, qV)
    #   and the helicity weights are only nonzero for polarization dependent variables (cos(theta*), phi*, lepton eta, ...)

    df_xnorm = define_helicity_weights(df_xnorm, is_z=dataset_name == "ZmumuPostVFP")
    df_xnorm = df_xnorm.DefinePerSample("xnorm", "0.5")
    axis_xnorm = hist.axis.Regular(
        1, 0.0, 1.0, name="count", underflow=False, overflow=False
    )
    xnorm_axes = [axis_xnorm, *theoryAgnostic_axes]
    xnorm_cols = ["xnorm", *theoryAgnostic_cols]
    xnormByHelicity = df_xnorm.HistoBoost(
        "xnorm",
        xnorm_axes,
        [*xnorm_cols, "nominal_weight_helicity"],
        tensor_axes=[axis_helicity],
    )
    results.append(xnormByHelicity)
    if not args.onlyMainHistograms:
        syst_tools.add_theory_hists(
            results,
            df_xnorm,
            args,
            dataset_name,
            corr_helpers,
            qcdScaleByHelicity_helper,
            xnorm_axes,
            xnorm_cols,
            base_name="xnorm",
            for_wmass=True,
            addhelicity=True,
        )
    else:
        # FIXME: hardcoded to keep mass weights (this would be done in add_theory_hists) but removing all other theory systs
        df_xnorm = syst_tools.define_mass_weights(df_xnorm, dataset_name)
        syst_tools.add_massweights_hist(
            results,
            df_xnorm,
            xnorm_axes,
            xnorm_cols,
            base_name="xnorm",
            proc=dataset_name,
            addhelicity=True,
        )
