import hist

from wums import logging

logger = logging.child_logger(__name__)

eta_binning = [
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


def get_pt_eta_axes(
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
                eta_binning, underflow=False, overflow=flow_eta, name="absEtaGen"
            )
        axes.append(axis_absEtaGen)
        cols.append(f"{gen_level}Lep_absEta")
        logger.debug(f"Gen bins |eta|: {axis_absEtaGen.edges}")

    if add_out_of_acceptance_axis:
        axes.append(hist.axis.Boolean(name="acceptance"))
        cols.append(f"{gen_level}_acceptance")

    return axes, cols


def get_pt_eta_charge_axes(
    gen_level,
    n_bins_pt,
    min_pt,
    max_pt,
    n_bins_eta=0,
    flow_pt=True,
    flow_eta=False,
    add_out_of_acceptance_axis=False,
):

    axes, cols = get_pt_eta_axes(
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


def get_dilepton_axes(
    gen_vars, reco_edges, gen_level, add_out_of_acceptance_axis=False
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
            # use 2 ptll bin for each ptVGen bin, last bin is overflow
            edges = reco_edges["ptll"]
            if len(edges) % 2:
                # in case it's an odd number of edges, last two bins are overflow
                edges = edges[:-1]
            # 1 gen bin for 2 reco bins
            edges = edges[::2]

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
                    overflow=False,
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
