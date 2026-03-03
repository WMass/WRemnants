import hist
import numpy as np

from wums import boostHistHelpers as hh
from wums import logging

logger = logging.child_logger(__name__)


def pdfSymmetricShifts(hdiff, axis_name):
    sq = hh.multiplyHists(hdiff, hdiff)
    ss = sq[{axis_name: hist.sum}]
    rss = hh.sqrtHist(ss)
    return rss, rss


def pdfAsymmetricShifts(hdiff, axis_name):
    # Assuming that the last axis is the syst axis
    # TODO: add some check to verify this
    def shiftHist(vals, hdiff, axis_name):
        hnew = hdiff[{axis_name: 0}]
        vals = vals * vals
        hnew[...] = np.sum(vals, axis=-1)
        return hh.sqrtHist(hnew)

    ax = hdiff.axes[axis_name]
    underflow = hdiff.axes[axis_name].traits.underflow
    if type(ax) == hist.axis.StrCategory and all(
        ["Up" in x or "Down" in x for x in ax][1:]
    ):
        # Remove the overflow from the categorical axis
        end = int((ax.size - 1) / 2)
        upvals = hdiff[{axis_name: [x for x in ax if "Up" in x]}].values(flow=True)[
            ..., :end
        ]
        downvals = hdiff[{axis_name: [x for x in ax if "Down" in x]}].values(flow=True)[
            ..., :end
        ]
        if upvals.shape != downvals.shape:
            raise ValueError(
                "Malformed PDF uncertainty hist! Expect equal number of up and down vars"
            )
    else:
        end = ax.size + underflow
        upvals = hdiff.values(flow=True)[..., 1 + underflow : end : 2]
        downvals = hdiff.values(flow=True)[..., 2 + underflow : end : 2]

    # The error sets are ordered up,down,up,down...
    upshift = shiftHist(upvals, hdiff, axis_name)
    downshift = shiftHist(downvals, hdiff, axis_name)
    return upshift, downshift


def hessianPdfUnc(h, axis_name="pdfVar", uncType="symHessian", scale=1.0):
    symmetric = uncType == "symHessian"
    diff = hh.addHists(h, -1 * h[{axis_name: 0}]) * scale
    if diff.axes[axis_name].traits.overflow:
        diff[..., hist.overflow] = np.zeros_like(diff[{axis_name: 0}].view(flow=True))
    shiftFunc = pdfSymmetricShifts if symmetric else pdfAsymmetricShifts
    rssUp, rssDown = shiftFunc(diff, axis_name)
    hUp = hh.addHists(h[{axis_name: 0}], 1 * rssUp)
    hDown = hh.addHists(h[{axis_name: 0}], -1 * rssDown)
    return hUp, hDown
