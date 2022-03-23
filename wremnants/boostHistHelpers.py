import hist
import boost_histogram as bh
import numpy as np
from functools import reduce

def valsAndVariances(h1, h2, allowBroadcast):
    if not allowBroadcast and len(h1.axes) != len(h2.axes):
        raise ValueError("Incompatible hists for math operation")
    if len(h1.axes) == len(h2.axes):
        return (h1.values(flow=True), h2.values(flow=True), h1.variances(flow=True), h2.variances(flow=True))
    else:
        outshape = h1.view(flow=True).shape if len(h1.shape) > len(h2.shape) else h2.view(flow=True).shape
        # The transpose is because numpy works right to left in broadcasting, and we've put the
        # syst axis on the right
        return [np.broadcast_to(x.T, outshape[::-1]).T for x in \
            [h1.values(flow=True), h2.values(flow=True), h1.variances(flow=True), h2.variances(flow=True)]]

def broadcastOutHist(h1, h2):
    if len(h1.axes) == len(h2.axes):
        return h1 if h1.axes[-1].size > h2.axes[-1].size else h2
    return h1 if len(h1.axes) > len(h2.axes) else h2

def divideHists(h1, h2, cutoff=1, allowBroadcast=True):
    h1vals,h2vals,h1vars,h2vars = valsAndVariances(h1, h2, allowBroadcast)
    # To get the broadcast shape right
    outh = h1 if not allowBroadcast else broadcastOutHist(h1, h2)
    # By the argument that 0/0 = 1
    out[np.abs(h2vals) < cutoff] = 1.
    val = np.divide(h1vals, h2vals, out=out, where=((np.abs(h2vals)>cutoff) & (np.abs(h1vals)>cutoff)))
    relvars = relVariances(h1vals, h2vals, h1vars, h2vars)
    var = val*sum(relVariances(h1vals, h2vals, h1vars, h2vars))
    var *= val
    newh = hist.Hist(*outh.axes, storage=hist.storage.Weight(),
            data=np.stack((val, var), axis=-1))
    return newh

def relVariances(h1vals, h2vals, h1vars, h2vars):
    h1nonzero = np.abs(h1vals) > 1e-3
    h2nonzero = np.abs(h2vals) > 1e-3
    rel1 = np.divide(np.divide(h1vars, h1vals, out=np.zeros_like(h1vals), where=h1nonzero),
            h1vals, out=np.zeros_like(h1vals), where=h1nonzero)
    rel2 = np.divide(np.divide(h2vars, h2vals, out=np.zeros_like(h2vals), where=h2nonzero), 
            h2vals, out=np.zeros_like(h2vals), where=h2nonzero)
    return (rel1, rel2)

def multiplyHists(h1, h2, allowBroadcast=True):
    h1vals,h2vals,h1vars,h2vars = valsAndVariances(h1, h2, allowBroadcast)
    val = h1vals*h2vals
    var = val*val*sum(relVariances(h1vals, h2vals, h1vars, h2vars))

    outh = h1 if not allowBroadcast else broadcastOutHist(h1, h2)
    newh = hist.Hist(*outh.axes, storage=hist.storage.Weight(),
            data=np.stack((val, var), axis=-1))
    return newh

def addHists(h1, h2, allowBroadcast=True):
    h1vals,h2vals,h1vars,h2vars = valsAndVariances(h1, h2, allowBroadcast)
    outh = h1 if not allowBroadcast else broadcastOutHist(h1, h2)

    newh = hist.Hist(*outh.axes, storage=hist.storage.Weight(),
            data=np.stack((h1vals+h2vals, h1vars+h2vars), axis=-1))
    return newh

def sumHists(hists):
    return reduce(addHists)

def mirrorHist(hvar, hnom, cutoff=1):
    div = divideHists(hnom, hvar, cutoff)
    hnew = multiplyHists(div, hnom)
    return hnew

def extendHistByMirror(hvar, hnom):
    hmirror = mirrorHist(hvar, hnom)
    mirrorAx = hist.axis.Integer(0,2, name="mirror", overflow=False, underflow=False)
    hnew = hist.Hist(*hvar.axes, mirrorAx, storage=hvar._storage_type())
    hnew.view(flow=True)[...] = np.stack((hvar.view(flow=True), hmirror.view(flow=True)), axis=-1)
    return hnew

def addSystAxis(h, size=1, offset=0):
    hnew = hist.Hist(*h.axes,hist.axis.Regular(size,offset,size+offset, name="systIdx"), storage=hist.storage.Weight())
    # Broadcast to new shape
    newvals = hnew.values()+h.values()[...,np.newaxis]
    newvars = hnew.variances()+h.variances()[...,np.newaxis]
    hnew[...] = np.stack((newvals, newvars), axis=-1)
    return hnew

def clipNegativeVals(h):
    hnew = hist.Hist(*h.axes, storage=hist.storage.Weight())
    vals = h.values(flow=True)
    vals[vals<0] = 0
    hnew[...] = np.stack((vals, h.variances(flow=True)), axis=-1)
    return hnew
