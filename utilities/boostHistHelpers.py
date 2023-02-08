import hist
import boost_histogram as bh
import numpy as np
from functools import reduce
import logging
import collections
from . import common

def valsAndVariances(h1, h2, allowBroadcast=True, transpose=True):
    if not allowBroadcast and len(h1.axes) != len(h2.axes):
        raise ValueError("Incompatible hists for math operation")

    flow = True
    for i,j in zip(h1.axes.traits, h2.axes.traits):
        flow = i.overflow == j.overflow and i.underflow == j.underflow
        if not flow:
            break

    if len(h1.axes) == len(h2.axes):
        return (h1.values(flow=flow), h2.values(flow=flow), h1.variances(flow=flow), h2.variances(flow=flow))
    else:
        outshape = h1.view(flow=flow).shape if len(h1.shape) > len(h2.shape) else h2.view(flow=flow).shape
        # The transpose is because numpy works right to left in broadcasting, and we've put the
        # syst axis on the right
        try:
            res = [np.broadcast_to(x.T if transpose else x, outshape[::-1] if transpose else outshape) for x in \
                    [h1.values(flow=flow), h2.values(flow=flow), h1.variances(flow=flow), h2.variances(flow=flow)]]
        except ValueError as e:
            logging.error(f"Failed to broadcast hists! h1.axes.name {h1.axes.name}, h2.axes.name {h2.axes.name}")
            raise e
        return [x.T for x in res] if transpose else res

def broadcastOutHist(h1, h2):
    if len(h1.axes) == len(h2.axes):
        return h1 if h1.axes[-1].size >= h2.axes[-1].size else h2
    return h1 if len(h1.axes) > len(h2.axes) else h2

# returns h1/h2
def divideHists(h1, h2, cutoff=1e-5, allowBroadcast=True, rel_unc=False):
    # To get the broadcast shape right
    outh = h1 if not allowBroadcast else broadcastOutHist(h1, h2)

    h1vals,h2vals,h1vars,h2vars = valsAndVariances(h1, h2, allowBroadcast)
    out = np.zeros_like(h2vals)
    # By the argument that 0/0 = 1
    out[(np.abs(h2vals) < cutoff) & (np.abs(h1vals) < cutoff)] = 1.
    val = np.divide(h1vals, h2vals, out=out, where=np.abs(h2vals)>cutoff)

    if h1._storage_type() != hist.storage.Weight() or h2._storage_type() != hist.storage.Weight():
        newh = hist.Hist(*outh.axes, data=val)
    else:
        relvars = relVariances(h1vals, h2vals, h1vars, h2vars)
        if rel_unc:
            # Treat the divisor as a constant
            var = val*val*relvars[0]
        else:
            var = val*val*sum(relvars)
        newh = hist.Hist(*outh.axes, storage=hist.storage.Weight(),
                data=np.stack((val, var), axis=-1))
    return newh

def relVariance(hvals, hvars, cutoff=1e-3, fillOnes=False):
    nonzero = np.abs(hvals) > cutoff
    if fillOnes:
        out = np.ones(hvars.shape)
    else:
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

def clipNegativeVals(h, clipValue=0):
    hnew = hist.Hist(*h.axes, storage=hist.storage.Weight())
    vals = h.values(flow=True)
    vals[vals<0] = clipValue
    hnew[...] = np.stack((vals, h.variances(flow=True)), axis=-1)
    return hnew

def scaleByLumi(h, scale, createNew=False):
    if createNew:
        hnew = hist.Hist(*h.axes, storage=hist.storage.Weight())
        hnew.values(flow=True)[...]    = scale * h.values(flow=True)
        hnew.variances(flow=True)[...] = scale * h.variances(flow=True)
        return hnew
    else:
        h.values(flow=True)[...]    *= scale
        h.variances(flow=True)[...] *= scale
        return h
    
def normalize(h, scale=1e6, createNew=True):
    scale = scale/h.sum().value
    if createNew:
        hnew = hist.Hist(*h.axes, storage=hist.storage.Weight())
        hnew.values(flow=True)[...]    = scale * h.values(flow=True)
        hnew.variances(flow=True)[...] = scale * h.variances(flow=True)
        return hnew
    else:
        h.values(flow=True)[...]    *= scale
        h.variances(flow=True)[...] *= scale
        return h

def makeAbsHist(h, axis_name):
    ax = h.axes[axis_name]
    if 0 not in ax.edges:
        raise ValueError("Can't mirror around 0 if it isn't a bin boundary")
    axidx = list(h.axes).index(ax)
    abs_ax = hist.axis.Variable(ax.edges[ax.index(0.):], underflow=False, name=f"abs{axis_name}")
    hnew = hist.Hist(*h.axes[:axidx], abs_ax, *h.axes[axidx+1:], storage=h._storage_type())
    
    s = hist.tag.Slicer()
    hnew[...] = h[{axis_name : s[ax.index(0):]}].view() + np.flip(h[{axis_name : s[:ax.index(0)]}].view(), axis=axidx)
    return hnew

def rebinHist(h, axis_name, edges):
    if type(edges) == int:
        return h[{axis_name : hist.rebin(edges)}]

    ax = h.axes[axis_name]
    ax_idx = [a.name for a in h.axes].index(axis_name)
    if not all([np.isclose(x, ax.edges).any() for x in edges]):
        raise ValueError(f"Cannot rebin histogram due to incompatible edges for axis '{ax.name}'\n"
                            f"Edges of histogram are {ax.edges}, requested rebinning to {edges}")
        
    # If you rebin to a subset of initial range, keep the overflow and underflow
    overflow = ax.traits.overflow or (edges[-1] < ax.edges[-1] and not np.isclose(edges[-1], ax.edges[-1]))
    underflow = ax.traits.underflow or (edges[0] > ax.edges[0] and not np.isclose(edges[0], ax.edges[0]))
    flow = overflow or underflow
    new_ax = hist.axis.Variable(edges, name=ax.name, overflow=overflow, underflow=underflow)
    axes = list(h.axes)
    axes[ax_idx] = new_ax
    
    hnew = hist.Hist(*axes, name=h.name, storage=h._storage_type())
    # Take is used because reduceat sums i:len(array) for the last entry, in the case
    # where the final bin isn't the same between the initial and rebinned histogram, you
    # want to drop this value. Add tolerance of 1/2 min bin width to avoid numeric issues

    offset = 0.5*np.min(ax.edges[1:]-ax.edges[:-1])

    edges_eval = edges+offset
    edge_idx = ax.index(edges_eval)

    if len(np.unique(edge_idx)) != len(edge_idx):
        raise ValueError("Did not find a unique binning. Probably this is a numeric issue with bin boundaries")

    if underflow:
        edge_idx = np.insert(edge_idx, 0, 0)
        # Only if the original axis had an underflow should you offset
        if ax.traits.underflow:
            edge_idx += 1

    # Avoid out of range error if there is no overflow
    # TODO: Understand if this is fully correct
    if edge_idx[-1] >= ax.size+ax.traits.overflow+ax.traits.underflow:
        edge_idx[-1] = ax.index(edges_eval[-1]-2*offset)
        if edge_idx[-1] == edge_idx[-2]:
            edge_idx = edge_idx[:-1]

    hnew.values(flow=flow)[...] = np.add.reduceat(h.values(flow=flow), edge_idx, 
            axis=ax_idx).take(indices=range(new_ax.size+underflow+overflow), axis=ax_idx)
    if hnew._storage_type() == hist.storage.Weight():
        hnew.variances(flow=flow)[...] = np.add.reduceat(h.variances(flow=flow), edge_idx, 
                axis=ax_idx).take(indices=range(new_ax.size+underflow+overflow), axis=ax_idx)
    return hnew

def mergeAxes(ax1, ax2):
    if ax1.edges[0] < ax2.edges[0]:
        tmp = ax1
        ax1 = ax2
        ax2 = tmp

    if np.array_equal(ax1.edges, ax2.edges):
        return ax1

    ax1_edges = ax1.edges
    ax1_merge_idx = len(ax1.edges)
    while ax1_edges[ax1_merge_idx-1] not in ax2.edges:
        ax1_merge_idx -= 1

    ax1_edges = ax1_edges[:ax1_merge_idx]
    if not ax1_edges.size:
        raise ValueError("Didn't find any common edges in two axes, can't merge")

    merge_idx = list(ax2.edges).index(ax1_edges[-1])+1
    if merge_idx < 1 or merge_idx > ax2.size:
        raise ValueError("Can't merge axes unless there is a common point of intersection!"
            f"The edges were {ax1.edges}, and {ax2.edges}")

    new_edges = np.concatenate((ax1_edges, ax2.edges[merge_idx:]))
    return hist.axis.Variable(new_edges, name=ax1.name)

def findCommonBinning(hists, axis_idx):
    if len(hists) < 2:
        raise ValueError("Can only find common binning between > 1 hists")

    orig_axes = [h.axes[axis_idx] for h in hists]
    # Set intersection with tolerance for floats
    common_edges = np.array(orig_axes[0].edges)
    for ax in orig_axes[1:]:
        common_edges = common_edges[np.isclose(np.array(ax.edges)[:,np.newaxis], common_edges).any(0)]

    edges = np.sort(common_edges)
    logging.debug(f"Common edges are {common_edges}")

    if len(edges) < 2:
        raise ValueError(f"Found < 2 common edges, cannot rebin. Axes were {orig_axes}")
    return edges

def rebinHistsToCommon(hists, axis_idx, keep_full_range=False):
    orig_axes = [h.axes[axis_idx] for h in hists]
    new_edges = findCommonBinning(hists, axis_idx)
    rebinned_hists = [rebinHist(h, ax.name, new_edges) for h,ax in zip(hists, orig_axes)]

    # TODO: This assumes that the range extension only happens in one direction,
    # specifically, that the longer range hist has higher values
    if keep_full_range:
        full_hists = []
        hists_by_max = sorted(hists, key=lambda x: x.axes[axis_idx].edges[-1])
        new_ax = rebinned_hists[0].axes[axis_idx]
        for h in hists_by_max:
            new_ax = mergeAxes(new_ax, h.axes[axis_idx])

        for h,rebinh in zip(hists, rebinned_hists):
            axes = list(rebinh.axes)
            axes[axis_idx] = new_ax
            newh = hist.Hist(*axes, name=rebinh.name, storage=h._storage_type())
            merge_idx = new_ax.index(rebinh.axes[axis_idx].edges[-1])
            # TODO: true the overflow/underflow properly
            low_vals = rebinh.view()
            max_idx = min(h.axes[axis_idx].size, h.axes[axis_idx].index(newh.axes[axis_idx].edges[-1]))
            vals = np.append(low_vals, np.take(h.view(), range(merge_idx, max_idx), axis_idx))
            zero_pad_shape = list(newh.shape)
            zero_pad_shape[axis_idx] -= vals.shape[axis_idx]
            vals = np.append(vals, np.full(zero_pad_shape, 
                hist.accumulators.WeightedSum(0, 0) if newh._storage_type() == hist.storage.Weight() else 0.))
            newh[...] = vals
            full_hists.append(newh)

        new_edges = findCommonBinning(full_hists, axis_idx)
        rebinned_hists = [rebinHist(h, h.axes[axis_idx].name, new_edges) for h in full_hists]

    return rebinned_hists
   
def projectNoFlow(h, proj_ax, exclude=[]):
    s = hist.tag.Slicer()
    if type(proj_ax) == str:
        proj_ax = [proj_ax]
    hnoflow = h[{ax : s[0:hist.overflow:hist.sum] for ax in h.axes.name if ax not in exclude+list(proj_ax)}]
    return hnoflow.project(*proj_ax) 

def syst_min_or_max_env_hist(h, proj_ax, syst_ax, indices, no_flow=[], do_min=True):
    if syst_ax not in h.axes.name:
        logging.warning(f"Did not find syst axis {syst_ax} in histogram. Returning nominal!")
        return h
    if max(indices) > h.axes[syst_ax].size:
        logging.warning(f"Range of indices exceeds length of syst axis '{syst_ax}.' Returning nominal!")
        return h

    systax_idx = h.axes.name.index(syst_ax)
    if systax_idx != h.ndim-1:
        raise ValueError("Required to have the syst axis at index -1")

    if syst_ax in proj_ax:
        proj_ax.pop(proj_ax.index(syst_ax))
        
    hvar = projectNoFlow(h, (*proj_ax, syst_ax), exclude=no_flow)

    proj_ax_idxs = [h.axes.name.index(ax) for ax in proj_ax]
    view = np.take(hvar.view(flow=True), indices, axis=-1)
    fullview = np.take(h.view(flow=True), indices, axis=-1)

    # Move project axis to second two last position so the broadcasting works
    for idx in proj_ax_idxs:
        fullview = np.moveaxis(fullview, idx, -2)
    
    op = np.argmin if do_min else np.argmax
    # Index of min/max values considering only the eventual projection
    # TODO: Check if it's a weighted sum or not
    idx = op(view.value, axis=-1)
    opview = fullview[(*np.indices(fullview.shape[:-1]), idx)]

    hnew = h[{syst_ax : 0}]
    # Now that the syst ax has been collapsed, project axes will be at last position
    for idx in reversed(proj_ax_idxs):
        opview = np.moveaxis(opview, -1, idx)
    hnew[...] = opview
    return hnew

def combineUpDownVarHists(down_hist, up_hist):
    if up_hist.axes != down_hist.axes:
        raise RuntimeError("input up and down histograms have different axes, can't combine")
    else:
        hnew = hist.Hist(*up_hist.axes, common.down_up_axis, storage=up_hist._storage_type())
        hnew.view(flow=True)[...] = np.stack((down_hist.view(flow=True), up_hist.view(flow=True)), axis = -1)
        return hnew

def smoothTowardsOne(h):
    vals = h.values(flow=True)
    vars = h.variances(flow=True)
    relErr = np.minimum(1., relVariance(vals, vars, fillOnes=True))
    newvals = (1.-relErr) * vals + relErr
    hnew = hist.Hist(*h.axes, storage=hist.storage.Weight())
    hnew.values(flow=True)[...]    = newvals
    hnew.variances(flow=True)[...] = vars
    return hnew

def set_flow(h, val="nearest"):
    raise NotImplementedError("This function doesn't actually work :(")
    for i, ax in enumerate(h.axes):
        if ax.traits.underflow:
            nearest_vals = np.take(h.values(flow=True), 1, i) 
            # FIXME Take+assign doesn't work :(
            np.take(h.values(flow=True), 0, i)[...] = nearest_vals if val == "nearest" else np.full_like(nearest_vals, val)
        if ax.traits.overflow:
            nearest_vals = np.take(h.values(flow=True), -2, i) 
            np.take(h.values(flow=True), -1, i)[...] = nearest_vals if val == "nearest" else np.full_like(nearest_vals, val)
    return h
