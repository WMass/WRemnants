import hist

import narf
import narf.clingutils
from wremnants.utilities import common

narf.clingutils.Declare('#include "mcstat_folds.hpp"')


def make_mc_stat_fold_axis(nfolds):
    return hist.axis.Integer(
        0,
        nfolds,
        underflow=False,
        overflow=False,
        name=common.mc_stat_fold_axis_name,
    )


def add_mc_stat_fold_hist(results, df, axes, cols, nfolds, seed=0):
    """Book the nominal histogram with an extra MC-stat fold axis.

    Each generated event is assigned to one of `nfolds` folds by a hash of
    (run, luminosityBlock, event), see wrem::mc_stat_fold. setupRabbit.py
    --mcStatFolds passes these histograms to the rabbit TensorWriter as fold
    templates for the two-half / k-fold de-biasing (rabbit --mcStatDebias).

    `cols` are the columns of `axes` followed by the weight column, as for the
    nominal histogram.
    """
    df = df.Define(
        common.mc_stat_fold_axis_name,
        f"wrem::mc_stat_fold(run, luminosityBlock, event, {nfolds}, {seed})",
    )
    results.append(
        df.HistoBoost(
            common.hist_name("nominal", syst=common.mc_stat_fold_syst_name),
            [*axes, make_mc_stat_fold_axis(nfolds)],
            [*cols[:-1], common.mc_stat_fold_axis_name, cols[-1]],
        )
    )
