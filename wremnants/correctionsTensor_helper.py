import narf
import ROOT

ROOT.gInterpreter.Declare('#include "theory_corrections.h"')

def makeCorrectionsTensor(corrh, tensor, tensor_rank):
    corrhConv = narf.hist_to_pyroot_boost(corrh, tensor_rank=tensor_rank)
    helper = tensor[type(corrhConv).__cpp_name__](ROOT.std.move(corrhConv))
    helper.hist = corrh
    helper.tensor_axes = corrh.axes[-1*tensor_rank:]
    return helper
