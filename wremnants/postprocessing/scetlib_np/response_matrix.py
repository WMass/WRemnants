"""Load the (reco × gen) response matrix R for the ParamModel.

R lives in the unfolding histmaker output (a separate hdf5 from the fit tensor)
as ``nominal_postfsr_yieldsUnfolding`` under the Z sample group. We select
``acceptance=True``, project to the reco × (ptVGen, absYVGen) axes (summing
over helicitySig per Luca's guidance), and return a numpy array plus axis
metadata.

Single entry point:

    load_R(unfolding_hdf5_path,
           sample_key="Zmumu_2016PostVFP",
           hist_name="nominal_postfsr_yieldsUnfolding") -> dict
"""

import h5py
import numpy as np

from wums import ioutils as wums_io

DEFAULT_HIST = "nominal_postfsr_yieldsUnfolding"
DEFAULT_SAMPLE = "Zmumu_2016PostVFP"

# Axes we keep, in canonical order: reco first, then gen.
RECO_AXES = ("ptll", "yll", "cosThetaStarll_quantile", "phiStarll_quantile")
GEN_AXES = ("ptVGen", "absYVGen")
# Axes we collapse: acceptance via {True} slice, helicitySig via project-out.
SUM_AXES = ("helicitySig",)


def load_R(
    unfolding_hdf5_path,
    sample_key=DEFAULT_SAMPLE,
    hist_name=DEFAULT_HIST,
    reco_axes=RECO_AXES,
    gen_axes=GEN_AXES,
):
    """Load R from the unfolding histmaker output.

    Returns a dict with:
        R           : ndarray shape (*reco_sizes, *gen_sizes), float64
        reco_axes   : list of (name, edges) tuples in the canonical order
        gen_axes    : list of (name, edges) tuples
        reco_shape  : tuple of axis sizes (reco)
        gen_shape   : tuple of axis sizes (gen)
        source      : (path, sample_key, hist_name) for traceability
    """
    with h5py.File(unfolding_hdf5_path, "r") as f:
        if sample_key not in f:
            raise KeyError(
                f"{unfolding_hdf5_path}: no '{sample_key}' group. "
                f"Available top-level: {list(f.keys())[:10]}"
            )
        sample = wums_io.pickle_load_h5py(f[sample_key])
        try:
            output = sample["output"]
        except (KeyError, TypeError) as exc:
            raise KeyError(
                f"{sample_key}: no 'output' dict — schema mismatch?"
            ) from exc
        if hist_name not in output:
            joint_candidates = [
                k for k in output.keys() if "yieldsUnfolding" in k or "Unfolding" in k
            ]
            raise KeyError(
                f"{sample_key}: '{hist_name}' missing. "
                f"Joint-hist candidates: {joint_candidates[:5]}"
            )
        proxy = output[hist_name]
        # Force materialization while the file is open.
        h = proxy.get() if hasattr(proxy, "get") else proxy

        # Sanity-check the axes.
        ax_names = [a.name for a in h.axes]
        required = set(reco_axes) | set(gen_axes) | {"acceptance"} | set(SUM_AXES)
        missing = required - set(ax_names)
        if missing:
            raise ValueError(
                f"{hist_name}: missing expected axes {missing}. " f"Got: {ax_names}"
            )

        # Select acceptance=True (gen events in fiducial), keep gen + reco axes,
        # sum over helicitySig.
        h_sel = h[{"acceptance": True}]
        h_proj = h_sel.project(*reco_axes, *gen_axes)
        # .project sums over un-listed axes (i.e. helicitySig here).

    # Out from the with-block: hist is materialized.
    R = h_proj.values(flow=False).astype(np.float64)

    reco_meta = [(name, h_proj.axes[name].edges) for name in reco_axes]
    gen_meta = [(name, h_proj.axes[name].edges) for name in gen_axes]
    reco_shape = tuple(h_proj.axes[name].size for name in reco_axes)
    gen_shape = tuple(h_proj.axes[name].size for name in gen_axes)

    return dict(
        R=R,
        reco_axes=reco_meta,
        gen_axes=gen_meta,
        reco_shape=reco_shape,
        gen_shape=gen_shape,
        source=(unfolding_hdf5_path, sample_key, hist_name),
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            f"usage: python -m {__name__.replace('.', '/')} <unfolding_hdf5> [sample_key]",
            file=sys.stderr,
        )
        sys.exit(2)
    sample = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SAMPLE
    info = load_R(sys.argv[1], sample_key=sample)
    print(f"R shape         : {info['R'].shape}")
    print(f"R sum           : {info['R'].sum():.6g}")
    print(f"reco_shape      : {info['reco_shape']}")
    print(f"gen_shape       : {info['gen_shape']}")
    for name, edges in info["reco_axes"]:
        print(
            f"  reco {name}: size={len(edges)-1} edges=[{edges[0]:.3g}, {edges[-1]:.3g}]"
        )
    for name, edges in info["gen_axes"]:
        print(
            f"  gen  {name}: size={len(edges)-1} edges=[{edges[0]:.3g}, {edges[-1]:.3g}]"
        )
