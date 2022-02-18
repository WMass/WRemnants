import hist
import numpy as np

def scale_helicity_hist_to_variations(scale_hist):

    s = hist.tag.Slicer()
    # select nominal QCD scales, but keep the sliced axis at size 1 for broadcasting
    nom_scale_hist = scale_hist[{"muRfact" : s[1.j:1.j+1], "muFfact" : s[1.j:1.j+1]}]

    # select nominal QCD scales and project down to nominal axes
    nom_hist = nom_scale_hist[{"ptVgen" : s[::hist.sum], "chargeVgen" : s[::hist.sum], "helicity" : s[::hist.sum], "muRfact" : s[1.j], "muFfact" : s[1.j] }]



    # difference between a given scale and the nominal, plus the sum
    # this emulates the "weight if idx else nominal" logic and corresponds to the decorrelated
    # variations
    if scale_hist.name is None:
        out_name = "scale_helicity_variations"
    else:
        out_name = scale_hist.name + "_variations"

    scale_variation_hist = hist.Hist(*scale_hist.axes, storage = scale_hist._storage_type(), name = out_name,
                data = scale_hist.view(flow=True) - nom_scale_hist.view(flow=True) + nom_hist.view(flow=True)[..., np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis])

    return scale_variation_hist
