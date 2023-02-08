import lz4.frame
import pickle
import numpy
import hist
from utilities import boostHistHelpers as hh
import numpy as np
import logging
import os

def read_and_scale(fname, proc, histname, calculate_lumi=False, scale=1):
    with lz4.frame.open(fname) as f:
        out = pickle.load(f)
        
    return load_and_scale(out, proc, histname, calculate_lumi, scale)

def load_and_scale(res_dict, proc, histname, calculate_lumi=False, scale=1.):
    h = res_dict[proc]["output"][histname]
    if not res_dict[proc]["dataset"]["is_data"]:
        scale = res_dict[proc]["dataset"]["xsec"]/res_dict[proc]["weight_sum"]*scale
        if calculate_lumi:
            data_keys = [p for p in res_dict.keys() if "dataset" in res_dict[p] and res_dict[p]["dataset"]["is_data"]]
            lumi = sum([res_dict[p]["lumi"] for p in data_keys])*1000
            if not lumi:
                logging.warning("Did not find a data hist! Skipping calculate_lumi option")
                lumi = 1
            scale *= lumi
    return h*scale

def read_all_and_scale(fname, procs, histnames, lumi=False):
    with lz4.frame.open(fname) as f:
        out = pickle.load(f)

    hists = []
    for histname in histnames:
        h = load_and_scale(out, procs[0], histname, lumi)
        for proc in procs[1:]:
            h += load_and_scale(out, proc, histname, lumi)
        hists.append(h)

    return hists

def read_scetlib_hist(path, nonsing="auto", flip_y_sign=False, charge=None):
    if path[-4:] == ".npz":
        f = np.load(path, allow_pickle=True)
    elif path[-4:] == ".pkl":
        with open(path, "rb") as picklefile:
            f = pickle.load(picklefile)
    else:
        ValueError("File {path} is not a recognized file format")

    if charge is not None:
        charge_args = (2, -2., 2.) if charge != 0 else (1, 0, 1) 
        charge_axis = hist.axis.Regular(*charge_args, flow=False, name = "charge")
    
    if type(f["hist"]) == hist.Hist:
        if charge is None:
            scetlibh = f["hist"]
        else:
            axes = f["hist"].axes
            scetlibh = hist.Hist(*axes[:-1], charge_axis, axes[-1], storage=f["hist"]._storage_type())
            scetlibh[...,charge_axis.index(charge),:] = f["hist"].view(flow=True)
    else:
        var_axis = hist.axis.Integer(f["bins"][0][0], f["bins"][0][-1], name="vars", flow=False)
        # Won't actually have overflow/underflow, but set to match MiNNLO
        mass_underflow = f["bins"][1][0] > 0.
        mass_overflow = f["bins"][1][-1] < 13000.
        mass_axis = hist.axis.Variable(f["bins"][1], name="Q", overflow=mass_overflow, underflow=mass_underflow)
        y_axis = hist.axis.Variable(f["bins"][2], name="Y")
        
        # Use 0.1 here rather than 0, because the nonsingular behaves much better with a "cut" at > 0.1
        pt_underflow = f["bins"][3][0] > 0.1
        pt_axis = hist.axis.Variable(f["bins"][3], name="qT", underflow=pt_underflow)

        h = f["hist"]
        storage = hist.storage.Double()
        axes = [mass_axis,y_axis,pt_axis,var_axis]
        if charge is not None:
            axes.insert(-1, charge_axis)

        varax_idx = -1 
        vals = np.moveaxis(h, 0, varax_idx)

        if "hist_err" in f:
            err = f["hist_err"]
            storage = hist.storage.Weight()
            vals = np.stack((vals, np.moveaxis(err, 0, varax_idx)), axis=-1)

        scetlibh = hist.Hist(*axes, storage=storage)
        if charge is None:
            scetlibh[...] = vals
        else:
            scetlibh[...,charge_axis.index(charge),:] = vals

    if nonsing and nonsing != "skip":
        if nonsing == "auto":
            nonsing = path.replace(*((".", "_nons.") if "sing" not in path else ("sing", "nons")))
        nonsingh = read_scetlib_hist(nonsing, nonsing="skip", flip_y_sign=flip_y_sign, charge=charge)
        scetlibh = hh.addHists(scetlibh, nonsingh)
    elif nonsing != "skip":
        logging.warning("Will not include nonsingular contribution!")
    
    if flip_y_sign:
        mid = y_axis.index(0)
        s = hist.tag.Slicer()
        scetlibh[{"Y" : s[mid:]}] = scetlibh[{"Y" : s[mid:]}].view()*-1

    return scetlibh 

def read_dyturbo_hist(filenames, path="", axes=("y", "pt"), charge=None):
    isfile = list(filter(lambda x: os.path.isfile(x), 
        [os.path.expanduser(os.path.join(path, f)) for f in filenames]))

    if not isfile:
        raise ValueError("Must pass in a valid file")

    hists = [read_dyturbo_file(f, axes) for f in isfile]
    if len(hists) > 1:
        hists = hh.rebinHistsToCommon(hists, 0)

    h = hh.sumHists(hists)

    if charge is not None:
        charge_args = (2, -2., 2.) if charge != 0 else (1, 0, 1) 
        charge_axis = hist.axis.Regular(*charge_args, flow=False, name = "charge")
        print(charge, charge_axis.index(charge))
        hnew = hist.Hist(*h.axes, charge_axis, storage=h._storage_type())
        hnew[...,charge_axis.index(charge)] = h.view(flow=True)
        return hnew
    else:
        return h

def expand_dyturbo_filenames(path, basename, varname, pieces=["n3ll_born", "n2ll_ct", "n2lo_vj"], append=None):
    return [os.path.join(path, "_".join(filter(None, [basename, piece, varname, append]))+".txt") for piece in pieces]

def dyturbo_varnames():
    return ["mur{0}_muf{1}_mures{2}".format(i,j,k).replace("0", "H") for i in range(3) for j in range(3) for k in range(3) 
        if abs(i-j) < 2 and abs(i-k) < 2 and abs(j-k) < 2 and not (i == 1 and j == 1 and k == 1)]

def read_dyturbo_variations(path, basename, varnames, axes, pieces=["n3ll_born", "n2ll_ct", "n2lo_vj"], append=None):
    central_files = expand_dyturbo_filenames(path, basename, "", pieces, append)
    centralh = read_dyturbo_hist(central_files, axes=axes)
    var_ax = hist.axis.Integer(0, len(varnames)+1, name="vars")
    varh = hist.Hist(*centralh.axes, var_ax, storage=centralh._storage_type())
    varh[...,0] = centralh.view(flow=True)
    for i,var in enumerate(varnames):
        filenames = expand_dyturbo_filenames(path, basename, var, pieces, append)
        varh[...,i+1] = read_dyturbo_hist(filenames, axes=axes).view(flow=True)
    return varh 

def distribution_to_hist(data):
    next_bin = data[1:,0]
    bin_width = next_bin-data[:-1,0]
    data[:-1,1:] = data[:-1,1:]*bin_width[:,np.newaxis]
    return data

# Ignoring the scale unc for now
def read_matrixRadish_hist(filename, axname="pt"):
    data = read_text_data(filename)
    # Multiply through by bin width
    data = distribution_to_hist(data)
    bins = np.unique(data[:,0])
    
    ax = hist.axis.Variable(bins, name=axname, underflow=not (bins[0] == 0 and "pt" in axname))
    var_ax = hist.axis.Integer(0, 3, name="vars", flow=False)
    h = hist.Hist(ax, var_ax, storage=hist.storage.Double())

    h[...] = data[:-1, np.array([1,3,5])]
    return h*1/1000
    
def read_text_data(filename):
    data = []
    for line in open(filename).readlines():
        entry = line.split("#")[0]
        entry_data = [float(i.strip()) for i in entry.split()]
        if not entry_data:
            continue
        data.append(entry_data)
    return np.array(data, dtype=float)

def read_dyturbo_file(filename, axnames=("y", "pt")):
    data = read_text_data(filename)
    # 2 numbers per axis + result + error
    if data.shape[1] != len(axnames)*2+2:
        raise ValueError(f"Mismatch between number of axes advertised ({len(axnames)} ==> {axnames}) and found ({(data.shape[1]-2)/2})")

    axes = []
    offset = True
    for i,name in enumerate(axnames):
        # Normally last line is the total cross section, also possible it isn't, so check the bin ranges
        offset = offset and data[-1,2*i] == data[0,2*i] and data[-1,2*i+1] == data[-2,2*i+1]
        bins = sorted(list(set(data[:len(data)-offset,2*i:2*i+2].flatten())))
        axes.append(hist.axis.Variable(bins, name=name, underflow=not (bins[0] == 0 and "pt" in name)))

    h = hist.Hist(*axes, storage=hist.storage.Weight())

    h[...] = np.reshape(data[:len(data)-offset,len(axes)*2:], (*h.axes.size, 2))
    return h*1/1000

def readImpacts(rtfile, group, sort=True, add_total=True, stat=0.0):
    histname = "nuisance_group_impact_nois" if group else "nuisance_impact_nois"
    impacts = rtfile[histname].to_hist()
    labels = np.array([impacts.axes[1].value(i) for i in range(impacts.axes[1].size)])
    total = rtfile["fitresults"][impacts.axes[0].value(0)+"_err"].array()[0]
    impacts = impacts.values()[0,:]
    if sort:
        order = np.argsort(impacts)
        impacts = impacts[order]
        labels = labels[order]
    if add_total:
        impacts = np.append(impacts, total)
        labels = np.append(labels, "Total")

    if stat > 0:
        idx = np.argwhere(labels == "stat")
        impacts[idx] = stat

    return impacts,labels
