import hist
import numpy as np

from wums import logging

logger = logging.child_logger(__name__)


## for W used in SMP-18-012
# 10% quantiles from aMC@NLO used in SMP-18-012 with some rounding <== This one worked fine with toys
ptV_10quantiles_binning = [
    0.0,
    2.95,
    4.73,
    6.68,
    8.98,
    11.78,
    15.33,
    20.11,
    27.17,
    40.15,
    13000.0,
]
# 5% quantiles from aMC@NLO used in SMP-18-012
ptV_20quantiles_binning = [
    0.0,
    1.971,
    2.949,
    3.838,
    4.733,
    5.674,
    6.684,
    7.781,
    8.979,
    10.303,
    11.777,
    13.435,
    15.332,
    17.525,
    20.115,
    23.245,
    27.173,
    32.414,
    40.151,
    53.858,
    13000.0,
]
# Integer rounded version of the 5% quantiles h[::hist.rebin(2)] for 10% quantiles
ptV_binning = [
    0,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    15,
    17,
    20,
    23,
    27,
    32,
    40,
    54,
    13000,
]
## for Z
# approximate 2.5% quantiles, used in SMP-25-16, SMP-25-17 for the Z detector level fits
ptZ_binning = [
    0,
    1,
    1.5,
    2,
    2.5,
    3,
    3.5,
    4,
    4.5,
    5,
    5.5,
    6,
    6.5,
    7,
    7.5,
    8,
    8.5,
    9,
    9.5,
    10,
    10.5,
    11,
    11.5,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    22,
    24,
    26,
    28,
    30,
    33,
    37,
    44,
    100,
]
# for the Z for SMP-25-016, SMP-25-17
yll_10quantiles_binning = [-2.5, -1.5, -1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0, 1.5, 2.5]
yll_20quantiles_binning = [
    -2.5,
    -1.8,
    -1.5,
    -1.3,
    -1.1,
    -0.9,
    -0.7,
    -0.5,
    -0.3,
    -0.15,
    0,
    0.15,
    0.3,
    0.5,
    0.7,
    0.9,
    1.1,
    1.3,
    1.5,
    1.8,
    2.5,
]

## for Ai based corrections and uncertainties (e.g. TheoryCorrections/ByHelicity/)
# for the W, 40 quantiles
ptWgen_binning_corr = [
    0,
    1,
    2,
    2.5,
    3,
    3.5,
    4,
    4.5,
    5,
    5.5,
    6,
    6.5,
    7,
    7.5,
    8,
    8.5,
    9,
    9.5,
    10,
    10.5,
    11,
    11.5,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    22,
    24,
    26,
    28,
    30,
    33,
    37,
    44,
    54,
    75,
    100,
    13000,
]
absYWgen_binning_corr = [
    0,
    0.25,
    0.5,
    0.75,
    1,
    1.25,
    1.5,
    1.75,
    2,
    2.25,
    2.5,
    2.75,
    3,
    3.25,
    3.5,
    3.75,
    4,
    5,
]
# for the Z, based on reco binning, but including additional bins where reco binning is too coarse
ptZgen_binning_corr = [*ptZ_binning[:-1], 54, 75, 100, 1300]
absYZgen_binning_corr = [
    *yll_20quantiles_binning[10:-1],
    2.0,
    2.25,
    2.5,
    2.75,
    3,
    3.25,
    3.5,
    3.75,
    4,
    5,
]

# categorical axes in python bindings always have an overflow bin, so use a regular axis for the charge
axis_charge = hist.axis.Regular(
    2, -2.0, 2.0, underflow=False, overflow=False, name="charge"
)

down_up_axis = hist.axis.Regular(
    2, -2.0, 2.0, underflow=False, overflow=False, name="downUpVar"
)
down_nom_up_axis = hist.axis.Regular(
    3, -1.5, 1.5, underflow=False, overflow=False, name="downNomUpVar"
)

# UL, A0...A4
axis_helicity = hist.axis.Integer(
    -1, 8, name="helicity", overflow=False, underflow=False
)
axis_helicity_multidim = hist.axis.Integer(
    -1, 8, name="helicitySig", overflow=False, underflow=False
)

# run edges chosen to separate eras (era F post VFP: [278769, 278808], era G [278820, 280385], era F [281613, 284044])
run_edges = np.array(
    [
        278768,
        278808,
        279588,
        279767,
        280017,
        280385,
        282037,
        283270,
        283478,
        283934,
        284044,
    ]
)
run_edges_lumi = np.array(
    [0.0, 0.419, 2.332, 4.329, 6.247, 8.072, 10.152, 12.265, 14.067, 15.994, 16.812]
)


eta_binning_unfolding = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3,
    1.5,
    1.7,
    1.9,
    2.1,
    2.4,
]  # 18 eta bins


def get_default_ptbins(analysis_label, unfolding=False, gen=False):
    vals = [30, 26.0, 56.0] if analysis_label[0] == "w" else [34, 26.0, 60.0]
    if unfolding and gen:
        raise ValueError(
            "Inconsistent arguments for 'unfolding' and 'gen.' Must be unique"
        )

    if unfolding:
        vals[0] += 2
        vals[2] += 2
    elif gen:
        vals[0] -= 2
        vals[1] += 2
    return vals


def get_default_etabins(analysis_label=None):
    return (48, -2.4, 2.4)


def get_default_mtcut(analysis_label=None):
    return 40.0 if analysis_label[0] == "w" else 45.0


def get_default_mz_window():
    return 60, 120


def get_unfolding_pt_eta_axes(
    gen_level,
    n_bins_pt,
    min_pt,
    max_pt,
    n_bins_eta=0,
    flow_pt=True,
    flow_eta=False,
    add_out_of_acceptance_axis=False,
):

    # gen axes for differential measurement
    axis_ptGen = hist.axis.Regular(
        n_bins_pt, min_pt, max_pt, underflow=flow_pt, overflow=flow_pt, name="ptGen"
    )
    logger.debug(f"Gen bins pT: {axis_ptGen.edges}")

    axes = [axis_ptGen]
    cols = [f"{gen_level}Lep_pt"]

    if n_bins_eta is not None:
        if n_bins_eta > 0:
            axis_absEtaGen = hist.axis.Regular(
                n_bins_eta, 0, 2.4, underflow=False, overflow=flow_eta, name="absEtaGen"
            )
        else:
            axis_absEtaGen = hist.axis.Variable(
                eta_binning_unfolding,
                underflow=False,
                overflow=flow_eta,
                name="absEtaGen",
            )
        axes.append(axis_absEtaGen)
        cols.append(f"{gen_level}Lep_absEta")
        logger.debug(f"Gen bins |eta|: {axis_absEtaGen.edges}")

    if add_out_of_acceptance_axis:
        axes.append(hist.axis.Boolean(name="acceptance"))
        cols.append(f"{gen_level}_acceptance")

    return axes, cols


def get_unfolding_pt_eta_charge_axes(
    gen_level,
    n_bins_pt,
    min_pt,
    max_pt,
    n_bins_eta=0,
    flow_pt=True,
    flow_eta=False,
    add_out_of_acceptance_axis=False,
):

    axes, cols = get_unfolding_pt_eta_axes(
        gen_level,
        n_bins_pt,
        min_pt,
        max_pt,
        n_bins_eta,
        flow_pt,
        flow_eta,
        add_out_of_acceptance_axis=add_out_of_acceptance_axis,
    )

    axis_qGen = hist.axis.Regular(
        2, -2.0, 2.0, underflow=False, overflow=False, name=f"qGen"
    )
    axes.append(axis_qGen)
    cols.append(f"{gen_level}Lep_charge")

    return axes, cols


def get_unfolding_dilepton_axes(
    gen_vars,
    reco_edges,
    gen_level,
    add_out_of_acceptance_axis=False,
    flow_y=False,
    rebin_pt=None,
):
    """
    construct axes, columns, and selections for differential Z dilepton measurement from correponding reco edges. Currently supported: pT(Z), |yZ|

    gen_vars (list of str): names of gen axes to be constructed
    reco_edges (dict of lists): the key is the corresponding reco axis name and the values the edges
    gen_level (str): generator level definition (e.g. `prefsr`, `postfsr`)
    add_out_of_acceptance_axis (boolean): To add a boolean axis for the use of out of acceptance contribution
    """

    axes = []
    cols = []
    selections = []

    # selections for out of fiducial region, use overflow bin in ptVGen (i.e. not treated as out of acceptance)
    for v in gen_vars:
        if v == "helicitySig":
            # helicity is added as a tensor axis
            continue
        var = v.replace("qVGen", "charge").replace("VGen", "")
        cols.append(f"{gen_level}V_{var}")

        if v == "ptVGen":
            edges = reco_edges["ptll"]
            if rebin_pt is not None:
                edges = rebin_pt(edges)

            axes.append(
                hist.axis.Variable(
                    edges, name="ptVGen", underflow=False, overflow=True
                ),
            )
        elif v == "absYVGen":
            # 1 absYVGen for 2 yll bins (negative and positive)
            edges = reco_edges["yll"]
            if edges[len(edges) // 2] != 0:
                raise RuntimeError("Central bin edge must be 0")
            axes.append(
                hist.axis.Variable(
                    edges[len(edges) // 2 :],
                    name="absYVGen",
                    underflow=False,
                    overflow=flow_y,
                ),
            )
            selections.append(f"{gen_level}V_absY < {edges[-1]}")
        elif v in ["qVGen"]:
            axes.append(
                hist.axis.Regular(
                    2, -2.0, 2.0, underflow=False, overflow=False, name="qVGen"
                )
            )
        else:
            raise NotImplementedError(f"Unfolding dilepton axis {v} is not supported.")

    if add_out_of_acceptance_axis:
        axes.append(hist.axis.Boolean(name="acceptance"))
        cols.append(f"{gen_level}_acceptance")

    return axes, cols, selections


def get_theoryAgnostic_axes(
    ptV_bins=[], absYV_bins=[], ptV_flow=False, absYV_flow=False, wlike=False
):

    if not wlike:
        ptV_bins_init = (
            [0.0, 3.0, 6.0, 9.7, 12.4, 16.0, 21.4, 29.5, 60.0]
            if not len(ptV_bins)
            else ptV_bins
        )
        absYV_bins_init = (
            [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 3.0]
            if not len(absYV_bins)
            else absYV_bins
        )
    else:
        ptV_bins_init = (
            [0.0, 3.0, 4.8, 6.7, 9.0, 12.0, 16.01, 23.6, 60]
            if not len(ptV_bins)
            else ptV_bins
        )
        absYV_bins_init = (
            [0.0, 0.4, 0.8, 1.2, 1.6, 2.0] if not len(absYV_bins) else absYV_bins
        )

    # Note that the helicity axis is defined elsewhere, and must not be added to the list of axes returned here
    axis_ptVgen = hist.axis.Variable(
        ptV_bins_init, name="ptVgenSig", underflow=False, overflow=ptV_flow
    )

    axis_absYVgen = hist.axis.Variable(
        absYV_bins_init, name="absYVgenSig", underflow=False, overflow=absYV_flow
    )

    axes = [axis_ptVgen, axis_absYVgen]
    cols = ["ptVgen", "absYVgen"]  # name of the branch, not of the axis

    return axes, cols


def make_bw_binning(
    mass=91.1535, width=2.4932, initialStep=0.1, bin_edges_low=[], bin_edges_high=[]
):
    import ROOT

    maxVal = ROOT.Math.breitwigner_pdf(mass, width, mass)
    bins = [mass]
    currentMass = mass
    while currentMass - mass < 100:
        binSize = (
            maxVal / ROOT.Math.breitwigner_pdf(currentMass, width, mass) * initialStep
        )
        currentMass += binSize
        bins.append(currentMass)
        lowMass = 2 * mass - currentMass
        if lowMass - binSize > 0:
            bins.insert(0, lowMass)
    bins.insert(0, 0.0)

    if bin_edges_low:
        bins = bin_edges_low + [b for b in bins if b > bin_edges_low[-1]][1:]
    if bin_edges_high:
        bins = [b for b in bins if b < bin_edges_high[0]][:-1] + bin_edges_high

    return bins


# for fake estimation
# binary categories for simple ABCD method
passIsoName = "passIso"
passMTName = "passMT"

axis_passIso = hist.axis.Boolean(name=passIsoName)
axis_passMT = hist.axis.Boolean(name=passMTName)

# axes with only a few bins for beyond simple ABCD methods
axis_isoCat = hist.axis.Variable([0, 4, 8], name="iso", underflow=False, overflow=True)
axis_relIsoCat = hist.axis.Variable(
    [0, 0.15, 0.3], name="relIso", underflow=False, overflow=True
)


def get_binning_fakes_pt(min_pt, max_pt):
    edges = np.arange(min_pt, 32, 1)
    edges = np.append(
        edges, [e for e in [33, 35, 38, 41, 44, 47, 50, 53, 56] if e < max_pt][:-1]
    )
    edges = np.append(edges, [max_pt])
    ## the following lines are used to replace the previous ones when studying different pT binning and the MC stat
    # edges = np.arange(min_pt,32,1)
    # edges = np.append(edges, [e for e in [33,36,40,46,56] if e<max_pt][:-1])
    # edges = np.append(edges, [max_pt])
    # edges = np.arange(min_pt,32,1)
    # edges = np.append(edges, [e for e in [33,36,40,46,56] if e<max_pt][:-1])
    # edges = np.append(edges, [max_pt])
    # edges = np.arange(min_pt,32.1,1.2)
    # edges = np.append(edges, [e for e in [34.4, 38, 44, 56] if e<max_pt][:-1])
    # edges = np.append(edges, [max_pt])
    # edges = np.arange(min_pt,32,2)
    # edges = np.append(edges, [e for e in [32, 36, 40, 46, 56] if e<max_pt][:-1])
    # edges = np.append(edges, [max_pt])
    # edges = np.arange(min_pt, max_pt, 3)
    # edges = np.append(edges, [max_pt])

    return edges


def get_binning_fakes_mt(mt_cut=40, high_mt_bins=False, fine_mt_binning=False):
    edges = np.array([0, int(mt_cut / 2.0), mt_cut])
    if high_mt_bins:
        # needed for extended 2D method
        edges = np.append(
            edges, [e for e in [30, 32, 34, 36, 38, 40, 44, 49, 55, 62] if e > mt_cut]
        )
    if fine_mt_binning:
        end = 120
        step = 2
        edges = np.append(
            edges, np.linspace(mt_cut + step, end, int((end - mt_cut) / step))
        )
    return edges


def get_binning_fakes_relIso(high_iso_bins=False):
    edges = [0, 0.15]
    if high_iso_bins:
        # needed for extended 2D method
        edges.append(0.3)
    return edges


def add_charge_axis(h, charge):
    charge_args = (2, -2.0, 2.0) if charge != 0 else (1, 0, 1)
    charge_axis = hist.axis.Regular(*charge_args, flow=False, name="charge")

    has_vars = h.axes.name[-1] == "vars"
    new_axes = (
        (*h.axes, charge_axis)
        if not has_vars
        else (*h.axes[:-1], charge_axis, h.axes[-1])
    )
    hnew = hist.Hist(*new_axes, storage=h.storage_type())
    if has_vars:
        hnew[..., charge_axis.index(charge), :] = h.view(flow=True)
    else:
        hnew[..., charge_axis.index(charge)] = h.view(flow=True)
    return hnew
