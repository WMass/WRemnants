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


def add_mc_stat_fold_axis(df, axes, cols, nfolds, seed=0):
    """Add the MC-stat fold axis to the axes and columns of the MC histograms.

    Each generated event is assigned to one of `nfolds` folds by a hash of
    (run, luminosityBlock, event), see wrem::mc_stat_fold. Every histogram booked
    from the returned axes and columns, the nominal and its systematic
    variations, then splits its events into the folds. setupRabbit.py
    --mcStatFolds passes them to the rabbit TensorWriter as fold templates for
    the two-half / k-fold de-biasing (rabbit --mcStatDebias), with per-fold
    systematic variations (split logk).

    `cols` are the columns of `axes`, without the weight. Returns the dataframe
    with the fold column and new lists of axes and columns.
    """
    df = df.Define(
        common.mc_stat_fold_axis_name,
        f"wrem::mc_stat_fold(run, luminosityBlock, event, {nfolds}, {seed})",
    )
    return (
        df,
        [*axes, make_mc_stat_fold_axis(nfolds)],
        [*cols, common.mc_stat_fold_axis_name],
    )
