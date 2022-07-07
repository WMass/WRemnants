import hist
import boost_histogram as bh
import numpy as np
from functools import reduce

def valsAndVariances(h1, h2, allowBroadcast=True, transpose=True):
    if not allowBroadcast and len(h1.axes) != len(h2.axes):
        raise ValueError("Incompatible hists for math operation")
    if len(h1.axes) == len(h2.axes):
        return (h1.values(flow=True), h2.values(flow=True), h1.variances(flow=True), h2.variances(flow=True))
    else:
        outshape = h1.view(flow=True).shape if len(h1.shape) > len(h2.shape) else h2.view(flow=True).shape
        # The transpose is because numpy works right to left in broadcasting, and we've put the
        # syst axis on the right
        res = [np.broadcast_to(x.T if transpose else x, outshape[::-1] if transpose else outshape) for x in \
                [h1.values(flow=True), h2.values(flow=True), h1.variances(flow=True), h2.variances(flow=True)]]
        return [x.T for x in res] if transpose else res

def broadcastOutHist(h1, h2):
    if len(h1.axes) == len(h2.axes):
        return h1 if h1.axes[-1].size > h2.axes[-1].size else h2
    return h1 if len(h1.axes) > len(h2.axes) else h2

# returns h1/h2
def divideHists(h1, h2, cutoff=1, allowBroadcast=True):
    h1vals,h2vals,h1vars,h2vars = valsAndVariances(h1, h2, allowBroadcast)
    # To get the broadcast shape right
    outh = h1 if not allowBroadcast else broadcastOutHist(h1, h2)
    # By the argument that 0/0 = 1
    out = np.ones_like(h2vals)
    out[np.abs(h2vals) < cutoff] = 1.
    val = np.divide(h1vals, h2vals, out=out, where=((np.abs(h2vals)>cutoff) & (np.abs(h1vals)>cutoff)))
    relvars = relVariances(h1vals, h2vals, h1vars, h2vars)
    var = val*sum(relVariances(h1vals, h2vals, h1vars, h2vars))
    var *= val
    newh = hist.Hist(*outh.axes, storage=hist.storage.Weight(),
            data=np.stack((val, var), axis=-1))
    return newh

def relVariance(hvals, hvars, cutoff=1e-3):
    nonzero = np.abs(hvals) > cutoff
    out = np.copy(hvars)
    np.divide(hvars, hvals*hvals, out=out, where=nonzero),
    return out

def relVariances(h1vals, h2vals, h1vars, h2vars):
    rel1 = relVariance(h1vals, h1vars)
    rel2 = relVariance(h2vals, h2vars)
    return (rel1, rel2)

def sqrtHist(h):
    rootval = np.sqrt(h.values(flow=True))
    relvar = relVariance(h.values(flow=True), h.variances(flow=True))
    newvar = 0.5*rootval*rootval*relvar
    rooth = h.copy()
    rooth[...] = np.stack((rootval, newvar), axis=-1)
    return rooth

def multiplyWithVariance(vals1, vals2, vars1, vars2):
    val = np.multiply(vals1, vals2)
    var = val*val*sum(relVariances(vals1, vals2, vars1, vars2))
    return val, var

def multiplyHists(h1, h2, allowBroadcast=True, transpose=True):
    h1vals,h2vals,h1vars,h2vars = valsAndVariances(h1, h2, allowBroadcast, transpose)
    val,var = multiplyWithVariance(h1vals, h2vals, h1vars, h2vars)

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
    return reduce(addHists, hists)

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

def makeAbsHist(h, axis_name):
    ax = h.axes[axis_name]
    axidx = list(h.axes).index(ax)
    abs_ax = hist.axis.Variable(ax.edges[ax.index(0.):], name=f"abs{axis_name}")
    hnew = hist.Hist(*h.axes[:axidx], abs_ax, *h.axes[axidx+1:], storage=hist.storage.Weight())
    
    s = hist.tag.Slicer()
    hnew[...] = h[{axis_name : s[ax.index(0):]}].view() + np.flip(h[{axis_name : s[:ax.index(0)]}].view(), axis=axidx)
    return hnew

def rebinHist(h, axis_name, edges):
    ax = h.axes[axis_name]
    ax_idx = [a.name for a in h.axes].index(axis_name)
    if not all([x in ax.edges for x in edges]):
        raise ValueError(f"Cannot rebin histogram due to incompatible eduges for axis '{ax.name}'\n"
                            f"Edges of histogram are {ax.edges}, requested rebinning to {edges}")
        
    new_ax = hist.axis.Variable(edges, name=ax.name, overflow=ax.traits.overflow, underflow=ax.traits.underflow)
    axes = list(h.axes)
    axes[ax_idx] = new_ax
    
    hnew = hist.Hist(*axes, name=h.name, storage=h._storage_type())
    sum_edges = edges if edges[-1] != ax.edges[-1] else edges[:-1]
    # Take is used because reduceat sums i:len(array) for the last entry, in the case
    # where the final bin isn't the same between the initial and rebinned histogram, you
    # want to drop this value
    hnew.values()[...] = np.add.reduceat(h.values(), h.axes[axis_name].index(sum_edges), 
            axis=ax_idx).take(indices=range(new_ax.size), axis=ax_idx)
    hnew.variances()[...] = np.add.reduceat(h.variances(), h.axes[axis_name].index(sum_edges), 
            axis=ax_idx).take(indices=range(new_ax.size), axis=ax_idx)
    return hnew
