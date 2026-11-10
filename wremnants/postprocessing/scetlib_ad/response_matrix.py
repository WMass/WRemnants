"""Load the (reco x gen) response matrix R and its gen total N_gen from a histmaker
output, for a parameter model that folds a gen-level prediction to reco level.

R is ``nominal_prefsr_yieldsUnfolding`` (or ``nominal_prefsr_yieldsResponse``,
written by mz_dilepton --responseGenBinning theoryCorr on the theory correction's
grid), summed over ``acceptance`` and over ``helicitySig``: R is filled with the
helicity PARTITION of the event weight, whose pieces add up to the full weight.
N_gen is the gen-level xnorm total, filled with ``csAngularMoments``, whose
angular-integrated cross section is the UL component (helicitySig = -1) only.
"""

import h5py
import numpy as np

from wums import ioutils as wums_io

# datacard auxiliary group that setupRabbit --storeResponseMatrix writes R into
AUXILIARY_GROUP = "response_matrix"
# the same group in cards written before the rename; read as a fallback
LEGACY_AUXILIARY_GROUPS = ("scetlib_np",)

RECO_AXES = ("ptll", "yll", "cosThetaStarll_quantile", "phiStarll_quantile")
GEN_AXES = ("ptVGen", "absYVGen")

# pre-FSR, the level at which the boson-level prediction is defined
DEFAULT_HIST = "nominal_prefsr_yieldsUnfolding"
# The unfolding gen total "prefsr" is acceptance-filtered, while R sums both gen
# acceptance values; the full phase-space total (60 < m < 120, no acceptance cut)
# on the same gen axes is the matching denominator.
DEFAULT_GENTOTAL = "prefsr_full"
RESPONSE_HIST = "nominal_prefsr_yieldsResponse"
RESPONSE_GENTOTAL = "prefsr_response"
DEFAULT_SAMPLE = "Zmumu_2016PostVFP"
HELICITY_AXIS = "helicitySig"

# Upper edge of the extra gen bin that holds the ptVGen overflow of the unfolding
# binning (whose last edge is the last reco ptll edge): gen qT above it migrates
# into the highest reco bins, so the model has to predict it.
PTVGEN_OVERFLOW_EDGE = 100.0


def _select_ul_helicity(h):
    """UL component (helicitySig = -1) of a csAngularMoments-filled hist, if any."""
    if HELICITY_AXIS in h.axes.name:
        h = h[{HELICITY_AXIS: h.axes[HELICITY_AXIS].index(-1)}]
    return h


def _append_axis_overflow(h, axis_name):
    """In-range values of ``h`` with the overflow of ``axis_name`` appended as one
    extra bin along that axis."""
    full = h.values(flow=True)
    inr = h.values(flow=False).astype(np.float64)
    idx, pos = [], None
    for p, ax in enumerate(h.axes):
        uf = 1 if ax.traits.underflow else 0
        if ax.name == axis_name:
            pos = p
            idx.append(slice(uf + ax.size, uf + ax.size + 1))
        else:
            idx.append(slice(uf, uf + ax.size))
    if pos is None:
        raise ValueError(f"No {axis_name!r} axis in {list(h.axes.name)}")
    over = full[tuple(idx)].astype(np.float64)
    return np.concatenate([inr, over], axis=pos)


def has_response(
    unfolding_hdf5_path,
    sample_key=DEFAULT_SAMPLE,
    hist_name=DEFAULT_HIST,
    gen_total_name=DEFAULT_GENTOTAL,
):
    """True if the histmaker output carries both the response and the gen total."""
    try:
        with h5py.File(unfolding_hdf5_path, "r") as f:
            if sample_key not in f:
                return False
            output = wums_io.pickle_load_h5py(f[sample_key])["output"]
            return hist_name in output and gen_total_name in output
    except (OSError, KeyError, TypeError):
        return False


def load_R(
    unfolding_hdf5_path,
    sample_key=DEFAULT_SAMPLE,
    hist_name=DEFAULT_HIST,
    reco_axes=RECO_AXES,
    gen_axes=GEN_AXES,
    gen_total_name=DEFAULT_GENTOTAL,
    ptVGen_overflow=True,
):
    """Load R and N_gen from the unfolding histmaker output.

    Returns a dict with ``R`` (shape (*reco_sizes, *gen_sizes)), ``N_gen`` (gen
    shape, or None), ``reco_axes`` / ``gen_axes`` (lists of (name, edges)),
    ``reco_shape``, ``gen_shape`` and ``source``.

    With ``ptVGen_overflow`` the ptVGen overflow is kept as a trailing gen bin
    [last edge, PTVGEN_OVERFLOW_EDGE], unless the gen axis already reaches that
    edge (e.g. on the theory correction's grid), where it is dropped.
    """
    with h5py.File(unfolding_hdf5_path, "r") as f:
        if sample_key not in f:
            raise KeyError(
                f"{unfolding_hdf5_path}: no '{sample_key}' group, "
                f"available: {list(f.keys())[:10]}"
            )
        output = wums_io.pickle_load_h5py(f[sample_key])["output"]
        if hist_name not in output:
            raise KeyError(f"{sample_key}: no histogram '{hist_name}'")
        proxy = output[hist_name]
        h = proxy.get() if hasattr(proxy, "get") else proxy

        missing = (set(reco_axes) | set(gen_axes) | {"acceptance"}) - set(h.axes.name)
        if missing:
            raise ValueError(f"{hist_name}: missing axes {missing}, has {h.axes.name}")

        # Sum over acceptance: a reco-selected event belongs in the response
        # whatever its gen acceptance flag. project() sums helicitySig, if present.
        h_proj = h[{"acceptance": slice(None, None, sum)}].project(
            *reco_axes, *gen_axes
        )

        if ptVGen_overflow:
            last_edge = float(h_proj.axes["ptVGen"].edges[-1])
            if last_edge >= PTVGEN_OVERFLOW_EDGE - 1e-9:
                ptVGen_overflow = False

        N_gen = None
        if gen_total_name is not None and gen_total_name in output:
            gp = output[gen_total_name]
            hg = gp.get() if hasattr(gp, "get") else gp
            hg_gen = _select_ul_helicity(hg).project(*gen_axes)
            N_gen = (
                _append_axis_overflow(hg_gen, "ptVGen")
                if ptVGen_overflow
                else hg_gen.values(flow=False).astype(np.float64)
            )

    R = (
        _append_axis_overflow(h_proj, "ptVGen")
        if ptVGen_overflow
        else h_proj.values(flow=False).astype(np.float64)
    )

    def _gen_edges(name):
        e = np.asarray(h_proj.axes[name].edges, dtype=np.float64)
        if ptVGen_overflow and name == "ptVGen":
            e = np.concatenate([e, [PTVGEN_OVERFLOW_EDGE]])
        return e

    reco_shape = R.shape[: len(reco_axes)]
    gen_shape = R.shape[len(reco_axes) :]
    if N_gen is not None and N_gen.shape != gen_shape:
        raise ValueError(
            f"gen total {gen_total_name!r} shape {N_gen.shape} != R gen shape "
            f"{gen_shape}"
        )

    return dict(
        R=R,
        N_gen=N_gen,
        reco_axes=[(name, h_proj.axes[name].edges) for name in reco_axes],
        gen_axes=[(name, _gen_edges(name)) for name in gen_axes],
        reco_shape=reco_shape,
        gen_shape=gen_shape,
        source=(unfolding_hdf5_path, sample_key, hist_name),
    )
