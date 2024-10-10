import ROOT

import narf

narf.clingutils.Declare('#include "theory_corrections.hpp"')


def makeCorrectionsTensor(corrh, tensor=None, tensor_rank=1, weighted_corr=False):
    hist_dims = len(corrh.axes) - tensor_rank

    if tensor is not None:
        pass
    elif weighted_corr:
        if hist_dims != 4:
            raise NotImplementedError(
                f"Currently only 4D is supported for weighted corrections. Requested {hist_dims}."
            )
        tensor = ROOT.wrem.TensorCorrectionsHelperWeighted4D
    elif hist_dims == 2:
        tensor = ROOT.wrem.TensorCorrectionsHelper2D
    elif hist_dims == 3:
        tensor = ROOT.wrem.TensorCorrectionsHelper3D
    elif hist_dims == 4:
        tensor = ROOT.wrem.TensorCorrectionsHelper
    else:
        raise NotImplementedError(
            f"A tensor correction helper for {hist_dims} dimensions is currently not supported."
        )

    corrhConv = narf.hist_to_pyroot_boost(corrh, tensor_rank=tensor_rank)
    helper = tensor[type(corrhConv).__cpp_name__](ROOT.std.move(corrhConv))
    helper.hist = corrh
    helper.tensor_axes = corrh.axes[-1 * tensor_rank :]

    return helper
