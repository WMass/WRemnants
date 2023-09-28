import narf
import ROOT

narf.clingutils.Declare('#include "theory_corrections.h"')

def makeCorrectionsTensor(corrh, tensor=None, tensor_rank=1):

    if tensor is not None:
        pass
    elif len(corrh.axes)-tensor_rank == 2:
        tensor = ROOT.wrem.TensorCorrectionsHelper2D
    elif len(corrh.axes)-tensor_rank == 3:
        tensor = ROOT.wrem.TensorCorrectionsHelper3D
    elif len(corrh.axes)-tensor_rank == 4:
        tensor = ROOT.wrem.TensorCorrectionsHelper
    else:
        raise NotImplementedError(f"A default tensor correction helper for {len(corrh.axes)-tensor_rank} dimensions is currently not supported.")

    corrhConv = narf.hist_to_pyroot_boost(corrh, tensor_rank=tensor_rank)
    helper = tensor[type(corrhConv).__cpp_name__](ROOT.std.move(corrhConv))
    helper.hist = corrh
    helper.tensor_axes = corrh.axes[-1*tensor_rank:]
    return helper
