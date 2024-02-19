import hist
import numpy as np
from functools import reduce
import collections
from utilities import common, logging
import copy

logger = logging.child_logger(__name__)

def valsAndVariances(h1, h2, flow=True):
    return h1.values(flow=flow),h2.values(flow=flow),h1.variances(flow=flow),h2.variances(flow=flow)

# Broadcast h1 to match the shape of h2
def broadcastSystHist(h1, h2, flow=True, by_ax_name=True):
    if h1.ndim > h2.ndim or h1.shape == h2.shape:
        return h1

    s1 = h1.values(flow=flow).shape 
    s2 = h2.values(flow=flow).shape

    # the additional axes have to be broadcasted as leading
    # Either do this by name, or by broadcasting from the right (numpy default broadcasts from left)
    if by_ax_name:
        moves = {i: e for i, (e, n2) in enumerate(zip(s2, h2.axes.name)) if n2 not in h1.axes.name}
    else:
        moves = {h2.ndim-1-i: h2.values(flow=flow).shape[h2.ndim-1-i] for i in range(h2.ndim-h1.ndim)}

    broadcast_shape = list(moves.values()) + list(s1)

    try:
        new_vals = np.broadcast_to(h1.values(flow=flow), broadcast_shape)
    except ValueError as e:
        raise ValueError("Cannot broadcast hists with incompatible axes!\n" 
                         f"    h1.shape {h1.shape}; h2.shape: {h2.shape}\n"
                         f"    h1.axes: {h1.axes}\n"
                         f"    h2.axes: {h2.axes}")

    # move back to original order
    new_vals = np.moveaxis(new_vals, np.arange(len(moves)), list(moves.keys()))

    if new_vals.shape != h2.values(flow=flow).shape:
        raise ValueError(f"Broadcast shape {new_vals.shape} (from h1.shape={h1.values(flow=flow).shape}, axes={h1.axes.name}) " \
                            f"does not match desired shape {h2.view(flow=flow).shape} (axes={h2.axes.name})")

    if h1.storage_type == hist.storage.Weight:
        new_vars = np.broadcast_to(h1.variances(flow=flow), broadcast_shape)
        new_vars = np.moveaxis(new_vars, np.arange(len(moves)), list(moves.keys()))
        new_vals = np.stack((new_vals, new_vars), axis=-1)

    return hist.Hist(*h2.axes, data=new_vals, storage=h1.storage_type())

# returns h1/h2
def divideHists(h1, h2, cutoff=1e-5, allowBroadcast=True, rel_unc=False, cutoff_val=1., flow=True, createNew=True, by_ax_name=True):
    if allowBroadcast:
        h1 = broadcastSystHist(h1, h2, flow, by_ax_name)
        h2 = broadcastSystHist(h2, h1, flow, by_ax_name)

    storage = h1.storage_type() if h1.storage_type == h2.storage_type else hist.storage.Double()
    outh = hist.Hist(*h1.axes, storage=storage) if createNew else h1

    h1vals,h2vals,h1vars,h2vars = valsAndVariances(h1, h2, flow=flow)

    # Careful not to overwrite the values of h1
    out = outh.values(flow=flow) if createNew else np.full(outh.values(flow=flow).shape, cutoff_val)

    # Apply cutoff to both numerator and denominator
    cutoff_criteria = np.abs(h2vals) > cutoff
    # By the argument that 0/0 = 1
    out[(np.abs(h2vals) < cutoff) & (np.abs(h1vals) < cutoff)] = cutoff_val
    val = np.divide(h1vals, h2vals, out=out, where=cutoff_criteria)

    if outh.storage_type == hist.storage.Weight:
        relvars = relVariances(h1vals, h2vals, h1vars, h2vars, cutoff=cutoff)
        val2 = np.multiply(val, val)
        if rel_unc:
            # Treat the divisor as a constant
            var = np.multiply(val2, relvars[0], out=val2)
        else:
            relsum = np.add(*relvars)
            var = np.multiply(relsum, val2, out=val2)

        outh.view(flow=flow)[...] = np.stack((val, var), axis=-1)
    else:
        outh.values(flow=flow)[...] = val

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

def multiplyHists(h1, h2, allowBroadcast=True, createNew=True, flow=True):
    if allowBroadcast:
        h1 = broadcastSystHist(h1, h2, flow=flow)
        h2 = broadcastSystHist(h2, h1, flow=flow)

    if h1.storage_type == hist.storage.Double and h2.storage_type == hist.storage.Double:
        return h1*h2 

    with_variance = h1.storage_type == hist.storage.Weight and h2.storage_type == hist.storage.Weight
    outh = h1
    vals, varis = multiplyWithVariance(h1.values(flow=flow), h2.values(flow=flow), 
                        h1.variances(flow=flow) if with_variance else None, h2.variances(flow=flow) if with_variance else None)

    if createNew:
        outh = hist.Hist(*outh.axes, storage=outh.storage_type())
    outh.values(flow=flow)[...] = vals
    if varis is not None:
        outh.variances(flow=flow)[...] = varis

    return outh

# TODO: Figure out the overflow
def concatenateHists(h1, h2, allowBroadcast=True, by_ax_name=True):
    if allowBroadcast:
        h1 = broadcastSystHist(h1, h2, flow=False, by_ax_name=by_ax_name)
        h2 = broadcastSystHist(h2, h1, flow=False, by_ax_name=by_ax_name)

    axes = []
    for ax1, ax2 in zip(h1.axes, h2.axes):
        if ax1 == ax2:
            axes.append(ax1)
            continue

        if type(ax1) != type(ax2):
            raise ValueError("Cannot combine inconsistent axis types!")

        if type(ax1) == hist.axis.StrCategory:
            new_entries = list(ax1)+list(ax2)
            if len(new_entries) != len(set(new_entries)):
                raise ValueError("Cannot concatenate StrCategory axes with duplicate labels")
            axes.append(hist.axis.StrCategory(new_entries, name=ax1.name))
        else:
            if ax1.edges[0] > ax2.edges[0]:
                ax1,ax2 = ax2,ax1

            if ax1.edges[-1] == ax2.edges[0]:
                axes.append(hist.axis.Variable(np.concatenate((ax1.edges, ax2.edges[1:])), name=ax1.name, 
                    underflow=ax1.traits.underflow, overflow=ax2.traits.overflow))
            else:
                raise ValueError(f"Cannot concatenate hists with inconsistent axes: {ax1.name} and {ax2.name}")

    newh = hist.Hist(*axes, storage=h1.storage_type())
    fill_by_centers(newh, h1)
    fill_by_centers(newh, h2)

    return newh

def concatenate_syst_hists(hists, has_nominal=True):
    if len(hists) < 2:
        raise ValueError("At least two hists required to concatenate")
    hcat = hists[0]
    for h in hists[1:]:
        hcat = concatenateHists(hcat, h[...,has_nominal:], allowBroadcast=False)

    return hcat

def fill_by_centers(h, href):
    to_fill = [list(y) if type(y) == hist.axis.StrCategory else x for x,y in zip(href.axes.centers, href.axes)]
    h.view()[h.axes.index(*to_fill)] = href.view()
    return h

def addHists(h1, h2, allowBroadcast=True, createNew=True, scale1=None, scale2=None, flow=True, by_ax_name=True):
    if allowBroadcast:
        h1 = broadcastSystHist(h1, h2, flow=flow, by_ax_name=by_ax_name)
        h2 = broadcastSystHist(h2, h1, flow=flow, by_ax_name=by_ax_name)

    h1vals,h2vals,h1vars,h2vars = valsAndVariances(h1, h2, flow=flow)
    hasWeights = h1._storage_type() == hist.storage.Weight() and h2._storage_type() == hist.storage.Weight()
    # avoid scaling the variance if not needed, to save some time
    # I couldn't use hvals *= scale, otherwise I get this error: ValueError: output array is read-only
    if scale1 is not None:
        h1vals = scale1 * h1vals
        if hasWeights:
            h1vars = (scale1*scale1) * h1vars
    if scale2 is not None:
        h2vals = scale2 * h2vals
        if hasWeights:
            h2vars = (scale2*scale2) * h2vars
                    
    outh = h1
    if createNew:
        if not hasWeights:
            return hist.Hist(*outh.axes, data=h1vals+h2vals)
        else:
            return hist.Hist(*outh.axes, storage=hist.storage.Weight(),
                            data=np.stack((h1vals+h2vals, h1vars+h2vars), axis=-1))            
    else:
        outvals = h1vals if h1.shape == outh.shape else h2vals
        np.add(h1vals, h2vals, out=outvals)
        outh.values(flow=True)[...] = outvals
        if hasWeights:
            outvars = h1vars if h1.shape == outh.shape else h2vars
            np.add(h1vars, h2vars, out=outvars)
            outh.variances(flow=True)[...] = outvars
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

# add new axis and set values of old histogram to idx
def addGenChargeAxis(h, idx):
    return addGenericAxis(h, hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "qGen"), idx, add_trailing=False, flow=True)
    
def addSystAxis(h, size=1, offset=0, axname="systIdx"):
    return addGenericAxis(h, hist.axis.Regular(size,offset,size+offset, name=axName))

def addGenericAxis(h, axis, idx=None, add_trailing=True, flow=True):
    axes = [*h.axes, axis] if add_trailing else [axis, *h.axes]
    hnew = hist.Hist(*axes, storage=h.storage_type())
    if idx != None:
        # add old histogram only in single bin 
        slices = [idx if ax==axis else slice(None) for ax in hnew.axes]
        hnew.view(flow=flow)[*slices] = h.view(flow=flow)
    else:
        # Broadcast to new shape
        slices = [np.newaxis if ax==axis else slice(None) for ax in hnew.axes]
        hnew.view(flow=flow)[...] = hnew.view(flow=flow)+h.view(flow=flow)[*slices]
    return hnew

def clipNegativeVals(h, clipValue=0, createNew=False):
    newh = h.copy() if createNew else h
    np.clip(newh.values(flow=True), a_min=clipValue, a_max=None, out=newh.values(flow=True))
    return newh

def scaleHist(h, scale, createNew=True, flow=True):
    if createNew:
        if h.storage_type == hist.storage.Double:
            hnew = hist.Hist(*h.axes)
        else:
            hnew = hist.Hist(*h.axes, storage=hist.storage.Weight())
            hnew.variances(flow=flow)[...] = scale*scale * h.variances(flow=flow)

        hnew.values(flow=flow)[...] = scale * h.values(flow=flow)

        return hnew
    else:
        h.values(flow=flow)[...] *= scale
        if h.storage_type == hist.storage.Weight:
            h.variances(flow=flow)[...] *= scale*scale
        return h
    
def normalize(h, scale=1e6, createNew=True, flow=True):
    if h.storage_type == hist.storage.Weight:
        scale = scale/h.sum(flow=flow).value
    else:
        scale = scale/h.sum(flow=flow)
    return scaleHist(h, scale, createNew, flow)

def makeAbsHist(h, axis_name, rename=True):
    ax = h.axes[axis_name]
    axidx = list(h.axes).index(ax)
    axInfo = dict(underflow=False, overflow=ax.traits.overflow, name=f"abs{axis_name}" if rename else axis_name)
    if ax.size == 1 and -ax.edges[0] == ax.edges[-1]:
        abs_ax = hist.axis.Regular(1, 0, ax.edges[-1], **axInfo)
        return hist.Hist(*h.axes[:axidx], abs_ax, *h.axes[axidx+1:], storage=h.storage_type(), data=h.view(flow=True))

    if 0 not in ax.edges:
        raise ValueError("Can't mirror around 0 if it isn't a bin boundary")
    abs_ax = hist.axis.Variable(ax.edges[ax.index(0.):], **axInfo)
    hnew = hist.Hist(*h.axes[:axidx], abs_ax, *h.axes[axidx+1:], storage=h.storage_type())
    
    s = hist.tag.Slicer()
    view = h[{axis_name : s[ax.index(0):]}].view(flow=True) + np.flip(h[{axis_name : s[:ax.index(0)]}].view(flow=True), axis=axidx)
    hnew[...] = np.take(view, range(ax.traits.underflow, abs_ax.size+ax.traits.underflow), axidx)
    return hnew

# Checks if edges1 could be rebinned to edges2. Order is important!
def compatibleBins(edges1, edges2):
    comparef = np.vectorize(lambda x: np.isclose(x, edges1).any())
    return np.all(comparef(edges2))

def rebinHistMultiAx(h, axis_map):
    for ax, binning in axis_map.items():
        if ax not in h.axes.name:
            logger.debug(f"Did not find axis {ax} in hist. Skipping rebin.")
            continue
        h = rebinHist(h, ax, binning)

    return h

def rebinHist(h, axis_name, edges):
    if type(edges) == int:
        return h[{axis_name : hist.rebin(edges)}]

    ax = h.axes[axis_name]
    ax_idx = [a.name for a in h.axes].index(axis_name)

    if type(edges) == list:
        if len(edges) == len(ax.edges) and all(x == y for x,y in zip(edges, ax.edges)):
            return h
    elif type(edges) == np.array:
        if edges.shape == ax.edges.shape and np.isclose(edges, ax.edges).all():
            return h

    if not compatibleBins(ax.edges, edges):
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

    if type(ax) == hist.axis.StrCategory:
        edge_idx = np.array(edges, dtype=int)
    else:
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
    flow_add = (underflow+overflow)*flow
    hnew.values(flow=flow)[...] = np.add.reduceat(h.values(flow=flow), edge_idx, 
            axis=ax_idx).take(indices=range(new_ax.size+flow_add), axis=ax_idx)
    if hnew.storage_type == hist.storage.Weight:
        hnew.variances(flow=flow)[...] = np.add.reduceat(h.variances(flow=flow), edge_idx, 
                axis=ax_idx).take(indices=range(new_ax.size+flow_add), axis=ax_idx)
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
            # TODO: treat the overflow/underflow properly
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
    # sets the values of the underflow and overflow bins to given value; if val='nearest' the values of the nearest bins are taken
    for axis in [a.name for a in h.axes if a.traits.overflow]:
        slices = [slice(None) if a!=axis else -1 for a in h.axes.name]
        h.view(flow=True)[*slices] = h[{axis: -1}].view(flow=True) if val=="nearest" else val
    for axis in [a.name for a in h.axes if a.traits.underflow]:
        slices = [slice(None) if a!=axis else 0 for a in h.axes.name]
        h.view(flow=True)[*slices] = h[{axis: 0}].view(flow=True) if val=="nearest" else val
    return h

# For converting the helicity scale hist to variations, keeping the gen axis to be fit
# If swap_axes = True, the new axis takes the place of the old gen axis in the ordering
def expand_hist_by_duplicate_axis(href, ref_ax_name, new_ax_name, swap_axes=False, flow=True):
    if ref_ax_name not in href.axes.name:
        raise ValueError(f"Did not find axis {ref_ax_name} in hist!")

    axes = href.axes

    ref_ax_idx = axes.name.index(ref_ax_name)
    ref_ax = axes[ref_ax_name]
    new_ax = copy.deepcopy(ref_ax)
    new_ax._ax.metadata["name"] = new_ax_name

    new_axes = [new_ax, *axes]
    if swap_axes:
        new_axes[0], new_axes[ref_ax_idx+1] = new_axes[ref_ax_idx+1], new_axes[0]
    hnew = hist.Hist(*new_axes, storage=href.storage_type())

    data = np.moveaxis(href.values(flow=flow), ref_ax_idx, 0)
    # Copy data from other axis along the diagnoal, off-diagonals will be zero
    exp_data = data*np.reshape(np.identity(data.shape[0]), (data.shape[0], data.shape[0], *(1 for i in data.shape[1:])))
    hnew.values(flow=flow)[...] = np.moveaxis(exp_data, 1, ref_ax_idx+1)
    if href.storage_type == hist.storage.Weight:
        var = np.moveaxis(href.variances(flow=flow), ref_ax_idx, 0)
        exp_var = var*np.reshape(np.identity(var.shape[0]), (var.shape[0], var.shape[0], *(1 for i in var.shape[1:])))
        hnew.variances(flow=flow)[...] = np.moveaxis(exp_var, 1, ref_ax_idx+1)

    return hnew

def expand_hist_by_duplicate_axes(href, ref_ax_names, new_ax_names, **kwargs):
    for ax_name, new_ax_name in zip(ref_ax_names, new_ax_names):
        href = expand_hist_by_duplicate_axis(href, ax_name, new_ax_name, **kwargs)
    return href

def swap_histogram_bins(histo, axis1, axis1_bin1, axis1_bin2, axis2=None, axis2_slice=None, flow=False, axis1_replace=None):
    # swap content from axis1: axis1_bin1 with axis1: axis1_bin2 
    # optionally for a subset of the histogram defined by axis2: axis2_slice
    # optionally the selected bin content can be replaced by axis1_replace (example use case: setting up and down variations to nominal)
    if axis2 is not None and axis2_slice is None:
        raise ValueError(f"Requested to flip bins for axis {axis2} but the corresponding slices 'axis2_slice' are not set")
    if isinstance(axis2_slice, slice):
        # for some reason complex slicing didn't work, convert to bin number
        tmp_slice = []
        for x in ("start", "stop", "step"):
            s = getattr(axis2_slice,x)
            if isinstance(s, complex):
                tmp_slice.append(histo.axes[axis2].index(s.imag))
            else:
                tmp_slice.append(s)
        axis2_slice = slice(*tmp_slice)

    slices1 = []
    slices2 = []
    slicesR = []
    for a in histo.axes.name:
        if a == axis1:
            slices1.append(histo.axes[a].index(axis1_bin1))
            slices2.append(histo.axes[a].index(axis1_bin2))
            if axis1_replace:
                slicesR.append(histo.axes[a].index(axis1_replace))
        elif axis2 is not None and a == axis2:                  
            slices1.append(axis2_slice)
            slices2.append(axis2_slice)
            if axis1_replace:
                slicesR.append(axis2_slice)
        else:
            slices1.append(slice(None))
            slices2.append(slice(None))
            if axis1_replace:
                slicesR.append(slice(None))

    # swap bins in specified slices
    data = histo.view(flow=flow)
    new_histo = histo.copy()
    new_histo.view(flow=flow)[*slices2] = data[*slices1] if axis1_replace is None else data[*slicesR]
    new_histo.view(flow=flow)[*slices1] = data[*slices2] if axis1_replace is None else data[*slicesR]
    return new_histo

def rescaleBandVariation(histo, factor):

    if factor==1.:
        return histo
    else:
        upper_env = histo.values()[...,1]
        lower_env = histo.values()[...,0]

        var = np.abs(upper_env-lower_env)/2
        centr = (upper_env+lower_env)/2
        new_upper = factor*var+centr
        new_lower = -factor*var+centr

        # leave sigmaUL variation to 50%
        new_upper[...,0] = 1.5*np.ones((new_upper.shape[0],new_upper.shape[1]))
        new_lower[...,0] = 0.5*np.ones((new_lower.shape[0],new_lower.shape[1]))

        histo[...]= np.stack([new_lower,new_upper],axis=-1)
        return histo

def rssHists(h, syst_axis, scale=1., hnom=None):
    s = hist.tag.Slicer()

    if hnom is None:
        hnom = h[{syst_axis : 0}]

    hdiff = addHists(h, hnom, scale2=-1.)*scale
    hss = multiplyHists(hdiff, hdiff)

    hrss = sqrtHist(hss[{syst_axis : s[0:hist.overflow:hist.sum]}])
    hUp = addHists(hnom, hrss)
    hDown = addHists(hnom, hrss, scale2=-1.)

    return hUp, hDown

def rssHistsMid(h, syst_axis, scale=1.):
    s = hist.tag.Slicer()

    hnom = 0.5*(h[{"downUpVar" : -1j, syst_axis : 0 }]+h[{"downUpVar" : 1j, syst_axis : 0 }])
    hdiff = addHists(h, hnom, scale2=-1.)*scale
    hss = multiplyHists(hdiff, hdiff)

    hrss = sqrtHist(hss[{syst_axis : s[0:hist.overflow:hist.sum]}])
    hUp = addHists(hnom, hrss[{"downUpVar" : 1j}])
    hDown = addHists(hnom, hrss[{"downUpVar" : -1j}], scale2=-1.)

    return hUp, hDown
