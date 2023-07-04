import hist
import numpy as np
from functools import reduce
import collections
from utilities import common, logging

logger = logging.child_logger(__name__)

def valsAndVariances(h1, h2):
    return h1.values(flow=True),h2.values(flow=True),h1.variances(flow=True),h2.variances(flow=True)

# Broadcast h1 to match the shape of h2
def broadcastSystHist(h1, h2):
    if h1.ndim > h2.ndim or h1.shape == h2.shape:
        return h1
    
    # Transpose because we keep syst axis last, but numpy broadcasts from the front
    new_vals = np.broadcast_to(h1.view(flow=True).T, h2.view(flow=True).T.shape).T

    return hist.Hist(*h2.axes, data=new_vals, storage=h1.storage_type())

# returns h1/h2
def divideHists(h1, h2, cutoff=1e-5, allowBroadcast=True, rel_unc=False, cutoff_val=1., createNew=True):
    if allowBroadcast:
        h1 = broadcastSystHist(h1, h2)
        h2 = broadcastSystHist(h2, h1)

    storage = h1.storage_type() if h1.storage_type == h2.storage_type else hist.storage.Double()
    outh = hist.Hist(*h1.axes, storage=storage) if createNew else h1

    h1vals,h2vals,h1vars,h2vars = valsAndVariances(h1, h2)

    # Careful not to overwrite the values of h1
    out = outh.values(flow=True) if createNew else np.full(outh.values(flow=True).shape, cutoff_val)

    # Apply cutoff to both numerator and denominator
    cutoff_criteria = np.abs(h2vals) > cutoff
    # By the argument that 0/0 = 1
    out[(np.abs(h2vals) < cutoff) & (np.abs(h1vals) < cutoff)] = 1.
    if not createNew:
        out[cutoff_criteria] = cutoff_val
    val = np.divide(h1vals, h2vals, out=out, where=cutoff_criteria)

    if outh.storage_type == hist.storage.Weight:
        relvars = relVariances(h1vals, h2vals, h1vars, h2vars, cutoff=cutoff)
        val2 = np.multiply(val, val)
        if rel_unc:
            # Treat the divisor as a constant
            var = np.multiply(val2, relvars[0], out=val2)
        else:
            relsum = np.multiply(*relvars)
            var = np.multiply(relsum, val2, out=val2)

        outh.view(flow=True)[...] = np.stack((val, var), axis=-1)
    else:
        outh.values(flow=True)[...] = val

    return outh

def relVariance(hvals, hvars, cutoff=1e-5, fillOnes=False):
    out = np.empty(hvars.shape)
    np.multiply(hvals, hvals, out=out)
    np.clip(out, a_min=cutoff, a_max=None, out=out)
    np.divide(hvars, out, out=out)
    if fillOnes:
        out[hvals<cutoff] = 1.
    return out

def relVariances(h1vals, h2vals, h1vars, h2vars, cutoff=1e-5):
    rel1 = relVariance(h1vals, h1vars, cutoff=cutoff)
    rel2 = relVariance(h2vals, h2vars, cutoff=cutoff)
    return (rel1, rel2)

# TODO: Implement this rather than relying on pdf unc function
#def rssHist(h):

def sqrtHist(h):
    rootval = np.sqrt(h.values(flow=True))
    rooth = h.copy()

    if h.storage_type == hist.storage.Double:
        rooth[...] = rootval
    else:
        relvar = relVariance(h.values(flow=True), h.variances(flow=True))
        newvar = 0.5*rootval*rootval*relvar
        rooth[...] = np.stack((rootval, newvar), axis=-1)

    return rooth

def multiplyWithVariance(vals1, vals2, vars1=None, vars2=None):
    outvals = np.empty(vals1.shape)
    outvars = np.empty(vars1.shape) if (vars1 is not None and vars2 is not None) else None
    np.multiply(vals1, vals2, out=outvals)
    if outvars is not None:
        np.multiply(outvals, outvals, out=outvars)
        relvar1, relvar2 = relVariances(vals1, vals2, vars1, vars2)
        np.add(relvar1, relvar2, out=relvar1)
        np.multiply(outvars, relvar1, out=outvars)
        
    return outvals, outvars

def multiplyHists(h1, h2, allowBroadcast=True, createNew=True):
    if allowBroadcast:
        h1 = broadcastSystHist(h1, h2)
        h2 = broadcastSystHist(h2, h1)

    if h1.storage_type == hist.storage.Double and h2.storage_type == hist.storage.Double:
        return h1*h2 

    with_variance = h1.storage_type == hist.storage.Weight and h2.storage_type == hist.storage.Weight
    outh = h1
    vals, varis = multiplyWithVariance(h1.values(flow=True), h2.values(flow=True), 
                        h1.variances(flow=True) if with_variance else None, h2.variances(flow=True) if with_variance else None)

    if createNew:
        outh = hist.Hist(*outh.axes, storage=outh.storage_type())

    outh.values(flow=True)[...] = vals
    if varis is not None:
        outh.variances(flow=True)[...] = varis

    return outh

def addHists(h1, h2, allowBroadcast=True, createNew=True):
    if allowBroadcast:
        h1 = broadcastSystHist(h1, h2)
        h2 = broadcastSystHist(h2, h1)

    h1vals,h2vals,h1vars,h2vars = valsAndVariances(h1, h2)
    outh = h1
    if createNew:
        if h1.storage_type != hist.storage.Weight or h2.storage_type != hist.storage.Weight:
            return hist.Hist(*outh.axes, data=h1vals+h2vals)
        else:
            return hist.Hist(*outh.axes, storage=hist.storage.Weight(),
                            data=np.stack((h1vals+h2vals, h1vars+h2vars), axis=-1))            
    else:
        np.add(h1vals, h2vals, out=h1vals if h1.shape == outh.shape else h2vals)
        if h1.storage_type == hist.storage.Weight and h2.storage_type == hist.storage.Weight:
            np.add(h1vars, h2vars, out=h1vars if h1.shape == outh.shape else h2vars)
        return outh                

def sumHists(hists):
    return reduce(addHists, hists)

def mirrorHist(hvar, hnom, cutoff=1):
    div = divideHists(hnom, hvar, cutoff, createNew=True)
    hnew = multiplyHists(div, hnom, createNew=False)
    return hnew

def extendHistByMirror(hvar, hnom, downAsUp=False, downAsNomi=False):
    hmirror = mirrorHist(hvar, hnom)
    if downAsNomi:
        hvar,hmirror = hmirror,hvar

    mirrorAx = hist.axis.Integer(0,2, name="mirror", overflow=False, underflow=False)

    if hvar.storage_type == hist.storage.Weight and hnom.storage_type == hist.storage.Weight:
        hnew = hist.Hist(*hvar.axes, mirrorAx, storage=hist.storage.Weight(),
                         data=np.stack((hvar.view(flow=True), hmirror.view(flow=True)), axis=-1))
    else:
        hnew = hist.Hist(*hvar.axes, mirrorAx,
                        data=np.stack((hvar.values(flow=True), hmirror.values(flow=True)), axis=-1))
    
    return hnew

def addSystAxis(h, size=1, offset=0):

    if h.storage_type == hist.storage.Double:
        hnew = hist.Hist(*h.axes,hist.axis.Regular(size,offset,size+offset, name="systIdx"))
        # Broadcast to new shape
        newvals = hnew.values()+h.values()[...,np.newaxis]
        hnew[...] = newvals
    else:
        hnew = hist.Hist(*h.axes,hist.axis.Regular(size,offset,size+offset, name="systIdx"), storage=hist.storage.Weight())
        # Broadcast to new shape
        newvals = hnew.values()+h.values()[...,np.newaxis]
        newvars = hnew.variances()+h.variances()[...,np.newaxis]
        hnew[...] = np.stack((newvals, newvars), axis=-1)

    return hnew

def clipNegativeVals(h, clipValue=0, createNew=False):
    newh = h.copy() if createNew else h
    np.clip(newh.values(flow=True), a_min=clipValue, a_max=None, out=newh.values(flow=True))
    return newh

def scaleHist(h, scale, createNew=True):
    if createNew:
        if h.storage_type == hist.storage.Double:
            hnew = hist.Hist(*h.axes)
        else:
            hnew = hist.Hist(*h.axes, storage=hist.storage.Weight())
            hnew.variances(flow=True)[...] = scale*scale * h.variances(flow=True)

        hnew.values(flow=True)[...] = scale * h.values(flow=True)

        return hnew
    else:
        h.values(flow=True)[...] *= scale
        if h.storage_type == hist.storage.Weight:
            h.variances(flow=True)[...] *= scale*scale
        return h
    
def normalize(h, scale=1e6, createNew=True):
    scale = scale/h.sum(flow=True).value
    return scaleHist(h, scale, createNew)

def makeAbsHist(h, axis_name):
    ax = h.axes[axis_name]
    axidx = list(h.axes).index(ax)
    if ax.size == 1 and -ax.edges[0] == ax.edges[-1]:
        abs_ax = hist.axis.Regular(1, 0, ax.edges[-1], underflow=False, name=f"abs{axis_name}")
        return hist.Hist(*h.axes[:axidx], abs_ax, *h.axes[axidx+1:], storage=h.storage_type(), data=h.view())

    if 0 not in ax.edges:
        raise ValueError("Can't mirror around 0 if it isn't a bin boundary")
    abs_ax = hist.axis.Variable(ax.edges[ax.index(0.):], underflow=False, name=f"abs{axis_name}")
    hnew = hist.Hist(*h.axes[:axidx], abs_ax, *h.axes[axidx+1:], storage=h.storage_type())
    
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
    
    hnew = hist.Hist(*axes, name=h.name, storage=h.storage_type())

    # Offset from bin edge to avoid numeric issues
    offset = 0.5*np.min(ax.edges[1:]-ax.edges[:-1])
    edges_eval = edges+offset
    edge_idx = ax.index(edges_eval)
    # Avoid going outside the range, reduceat will add the last index anyway
    if edge_idx[-1] == ax.size+ax.traits.overflow:
        edge_idx = edge_idx[:-1]

    if underflow:
        # Only if the original axis had an underflow should you offset
        if ax.traits.underflow:
            edge_idx += 1
        edge_idx = np.insert(edge_idx, 0, 0)

    # Take is used because reduceat sums i:len(array) for the last entry, in the case
    # where the final bin isn't the same between the initial and rebinned histogram, you
    # want to drop this value. Add tolerance of 1/2 min bin width to avoid numeric issues
    hnew.values(flow=flow)[...] = np.add.reduceat(h.values(flow=flow), edge_idx, 
            axis=ax_idx).take(indices=range(new_ax.size+underflow+overflow), axis=ax_idx)
    if hnew.storage_type == hist.storage.Weight:
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
    if merge_idx < 1 or merge_idx > ax2.size+1:
        raise ValueError("Can't merge axes unless there is a common point of intersection! "
            f"Merge index was {merge_idx} "
            f"The edges were {ax1.edges} (size={ax1.size}), and {ax2.edges} (size={ax2.size})")

    new_edges = np.concatenate((ax1_edges, ax2.edges[merge_idx:]))
    return hist.axis.Variable(new_edges, name=ax1.name)

def findAxes(hists, axis_idx):
    if type(axis_idx) in [list, tuple]:
        return [h.axes[[ax for ax in axis_idx if ax in h.axes.name][0]] for h in hists]
    else:
        return [h.axes[axis_idx] for h in hists]

def findCommonBinning(hists, axis_idx):
    if len(hists) < 2:
        raise ValueError("Can only find common binning between > 1 hists")

    orig_axes = findAxes(hists, axis_idx)
    # Set intersection with tolerance for floats
    common_edges = np.array(orig_axes[0].edges)
    for ax in orig_axes[1:]:
        common_edges = common_edges[np.isclose(np.array(ax.edges)[:,np.newaxis], common_edges).any(0)]

    edges = np.sort(common_edges)
    logger.debug(f"Common edges are {common_edges}")

    if len(edges) < 2:
        raise ValueError(f"Found < 2 common edges, cannot rebin. Axes were {orig_axes}")
    return edges

def rebinHistsToCommon(hists, axis_idx, keep_full_range=False):
    orig_axes = findAxes(hists, axis_idx)
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
            newh = hist.Hist(*axes, name=rebinh.name, storage=h.storage_type())
            merge_idx = new_ax.index(rebinh.axes[axis_idx].edges[-1])
            # TODO: true the overflow/underflow properly
            low_vals = rebinh.view()
            max_idx = min(h.axes[axis_idx].size, h.axes[axis_idx].index(newh.axes[axis_idx].edges[-1]))
            vals = np.append(low_vals, np.take(h.view(), range(merge_idx, max_idx), axis_idx))
            zero_pad_shape = list(newh.shape)
            zero_pad_shape[axis_idx] -= vals.shape[axis_idx]
            vals = np.append(vals, np.full(zero_pad_shape, 
                hist.accumulators.WeightedSum(0, 0) if newh.storage_type == hist.storage.Weight else 0.))
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

def syst_min_and_max_env_hist(h, proj_ax, syst_ax, indices, no_flow=[]):
    logger.debug(f"Taking the envelope of variation axis {syst_ax}, indices {indices}")
    hup = syst_min_or_max_env_hist(h, proj_ax, syst_ax, indices, no_flow=no_flow, do_min=False)
    hdown = syst_min_or_max_env_hist(h, proj_ax, syst_ax, indices, no_flow=no_flow, do_min=True)
    hnew = hist.Hist(*hup.axes, common.down_up_axis, storage=hup.storage_type())
    hnew[...,0] = hdown.view(flow=True)
    hnew[...,1] = hup.view(flow=True)
    return hnew

def syst_min_or_max_env_hist(h, proj_ax, syst_ax, indices, no_flow=[], do_min=True):
    if syst_ax not in h.axes.name:
        logger.warning(f"Did not find syst axis {syst_ax} in histogram. Returning nominal!")
        return h

    # Keep the order of the hist
    proj_ax = [ax for ax in h.axes.name if ax in proj_ax]
    systax_idx = h.axes.name.index(syst_ax)
    if systax_idx != h.ndim-1:
        raise ValueError("Required to have the syst axis at index -1")

    if len(indices) < 2:
        logger.warning(f"Requires at least two histograms for envelope. Returning nominal!")
        return h

    if type(indices[0]) == str:
        if all(x.isdigit() for x in indices):
            indices == [int(x) for x in indices]
        else:
            indices = h.axes[syst_ax].index(indices)

    if max(indices) > h.axes[syst_ax].size:
        logger.warning(f"Range of indices exceeds length of syst axis '{syst_ax}.' Returning nominal!")
        return h

    if syst_ax in proj_ax:
        proj_ax.pop(proj_ax.index(syst_ax))
        
    hvar = projectNoFlow(h, (*proj_ax, syst_ax), exclude=no_flow)

    view = np.take(hvar.view(flow=True), indices, axis=-1)
    fullview = np.take(h.view(flow=True), indices, axis=-1)

    # Move project axis to second to last position so the broadcasting works
    # NOTE: Be careful that you keep track of the actual order, keeping in mind that things are moving
    names = list(h.axes.name)
    initial_order = []
    for ax in proj_ax:
        idx = names.index(ax)
        initial_order.append(idx)
        # Inserts at second to last position
        names.insert(-1, names.pop(idx))
        # Moves axis to second to last position
        fullview = np.moveaxis(fullview, idx, -2)
    
    op = np.argmin if do_min else np.argmax
    # Index of min/max values considering only the eventual projection  
    idx = op(view.value if hasattr(view, "value") else view, axis=-1)
    opview = fullview[(*np.indices(fullview.shape[:-1]), idx)]

    hnew = h[{syst_ax : 0}]
    # Now that the syst ax has been collapsed, project axes will be at last position
    # Move the axes back to where they belong
    names = list(h.axes.name)
    for idx in reversed(initial_order):
        opview = np.moveaxis(opview, -1, idx)

    hnew[...] = opview
    return hnew

def combineUpDownVarHists(down_hist, up_hist):
    if up_hist.axes != down_hist.axes:
        raise RuntimeError("input up and down histograms have different axes, can't combine")
    else:
        hnew = hist.Hist(*up_hist.axes, common.down_up_axis, storage=up_hist.storage_type())
        hnew.view(flow=True)[...] = np.stack((down_hist.view(flow=True), up_hist.view(flow=True)), axis = -1)
        return hnew

def smoothTowardsOne(h):
    if h.storage_type == hist.storage.Double:
        logger.warning("Tried to smoothTowardsOne but histogram has no variances. Proceed without doing anything!")
        return h

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
