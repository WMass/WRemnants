import h5py
import numpy as np

from narf import ioutils
from utilities import logging

logger = logging.child_logger(__name__)


def get_fitresult(fitresult_filename, meta=False):
    h5file = h5py.File(fitresult_filename, mode="r")
    h5results = ioutils.pickle_load_h5py(h5file["results"])
    if meta:
        meta = ioutils.pickle_load_h5py(h5file["meta"])
        return h5results, meta
    return h5results


def get_poi_names(fitresult):
    h = fitresult["impacts"].get()
    return np.array(h.axes["parms"])


def get_syst_labels(fitresult):
    h = fitresult["parms"].get()
    return np.array(h.axes["parms"])


def read_impacts_poi(
    fitresult,
    poi,
    grouped=False,
    global_impacts=False,
    sort=True,
    add_total=True,
    stat=0.0,
    normalize=True,
):
    # read impacts of a single POI
    impact_name = "impacts"
    if global_impacts:
        impact_name = f"global_{impact_name}"
    if grouped:
        impact_name += "_grouped"

    h_impacts = fitresult[impact_name].get()
    h_impacts = h_impacts[{"parms": poi}]

    impacts = h_impacts.values()
    labels = np.array(h_impacts.axes["impacts"])

    if sort:
        order = np.argsort(impacts)
        impacts = impacts[order]
        labels = labels[order]

    if add_total or normalize:
        h_parms = fitresult["parms"].get()
        total = np.sqrt(h_parms[{"parms": poi}].variance)

        if add_total:
            impacts = np.append(impacts, total)
            labels = np.append(labels, "Total")

        if normalize:
            impacts /= total

    if stat > 0:
        idx = np.argwhere(labels == "stat")
        impacts[idx] = stat

    return impacts, labels


def get_pulls_and_constraints(fitresult):
    h_parms = fitresult["parms"].get()
    h_parms_prefit = fitresult["parms_prefit"].get()

    pulls = h_parms.values()
    constraints = np.sqrt(h_parms.variances())
    pulls_prefit = np.zeros_like(pulls, dtype=float)

    return pulls, constraints, pulls_prefit
