import ROOT

import narf

narf.clingutils.Declare('#include "theory_corrections.hpp"')


def makeCorrectionsTensor(corrh, tensor=None, tensor_rank=1, tensor_weight=False):
    """
    The correction helper is a class that takes a histogram and returns what is in the bin of the histogram
    If a "tensor" is given, an object of this template class will be created, otherwise a generic object will be created.
    For the generic object, the referenced bin has to be a tensor (i.e. the histogram can not be fully referenced to a value but must have axes that are not indexed, those will be the tensor axes)
    The tensor rank defines the number of tensor axes.
    The event weight can be a scalar (tensor_weight=False) or a tensor itself (tensor_weight=False).
    """

    corrhConv = narf.hist_to_pyroot_boost(corrh, tensor_rank=tensor_rank)

    if tensor is None:
        if tensor_rank == 0:
            raise NotImplementedError("Rank 0 tensors are not supported")

        hist_dims = len(corrh.axes) - tensor_rank

        if hist_dims > 4 or hist_dims < 1:
            raise NotImplementedError("Only 1D-4D histograms are supported")

        class_name = f"TensorCorrectionsHelper{hist_dims}D"

        tensor = getattr(ROOT.wrem, class_name)

        # the types of the columns used to index the histogram
        types = [
            "int" if "charge" in n else "double" for n in corrh.axes.name[:hist_dims]
        ]
        if not tensor_weight:
            types.append("double")

        helper = tensor[type(corrhConv).__cpp_name__, *types](ROOT.std.move(corrhConv))
    else:
        helper = tensor[type(corrhConv).__cpp_name__](ROOT.std.move(corrhConv))

    helper.hist = corrh
    helper.tensor_axes = corrh.axes[-1 * tensor_rank :]

    return helper
