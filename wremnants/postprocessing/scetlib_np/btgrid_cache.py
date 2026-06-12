"""One-shot pickle cache for the combined SCETlib bT-grid.

Loading the fineall btgrid as 1519 individual shards takes ~110s. After the
first call, this module writes a single ``combined_btgrid.pkl`` in the btgrid
directory; subsequent calls load that in a few seconds.

Usage:
    from wremnants.postprocessing.scetlib_np import btgrid_cache
    grid = btgrid_cache.load(BTGRID_DIR)
"""

import glob
import os
import pickle
import time

import numpy as np

_COMBINED_BASENAME = "combined_btgrid.pkl"


def load_btgrid_shards(submitdir_or_glob, runcard_basename=None):
    """Combine bT-grid shards produced by --bt-grid mode.

    `submitdir_or_glob` may be:
      - a directory: we look for ``*_btgrid.pkl`` inside (recursively only
        one level via scetlib_outputs/).
      - a glob pattern: used directly.

    Returns dict with:
        bT      : (Nbt,)
        b_bar   : (Nbt,)
        bins    : list of (Q, Y, qT, lep) bin centres, length Nbins
        vars    : dict of variation index -> setting dict (copied from the
                  first shard; all shards are expected to carry the same set)
        I_pert  : (Nvars, Nbins, Nbt)
        C_nu   : (Nvars, Nbins, Nbt)
        config  : dict from the first shard (perturbative configuration the
                  grid was generated against)
        n_shards: int
    """
    if os.path.isdir(submitdir_or_glob):
        candidates = [
            os.path.join(submitdir_or_glob, "scetlib_outputs", "*_btgrid.pkl"),
            os.path.join(submitdir_or_glob, "*_btgrid.pkl"),
        ]
    else:
        candidates = [submitdir_or_glob]

    files = []
    for pat in candidates:
        files = sorted(glob.glob(pat))
        if files:
            break
    if not files:
        raise FileNotFoundError(f"No btgrid shards found under {submitdir_or_glob!r}")

    # First shard sets the schema; later shards must match.
    with open(files[0], "rb") as f:
        first = pickle.load(f)
    if first.get("schema_version") != "bt_grid_v1":
        raise ValueError(
            f"Unexpected schema {first.get('schema_version')!r} in {files[0]}"
        )
    bT = np.asarray(first["bT"], dtype=float)
    b_bar = np.asarray(first["b_bar"], dtype=float)
    varis = first["vars"]
    config = first["config"]
    n_vars = len(varis)
    n_bt = bT.size

    # We don't know Nbins ahead of time without scanning all shards. Walk them
    # once: build a dict of bin -> (var_idx -> (I_pert_row, C_nu_row)).
    bin_to_data = {}
    for path in files:
        with open(path, "rb") as f:
            d = pickle.load(f)
        if d.get("schema_version") != "bt_grid_v1":
            raise ValueError(
                f"Mixed schema versions: {path} has {d.get('schema_version')}"
            )
        if d["bT"].shape != bT.shape or not np.allclose(d["bT"], bT):
            raise ValueError(f"bT grid mismatch in {path}")
        bins_local = d["bins"]
        I_local = np.asarray(
            d["I_pert"], dtype=float
        )  # (Nvars_local, Nbins_local, Nbt)
        C_local = np.asarray(d["C_nu"], dtype=float)
        # Map local variation indices to the union variation order. We assume
        # all shards share the same vars dict (true when produced by the same
        # condor submission).
        var_order_local = list(d["vars"].keys())
        for b_idx, b_tup in enumerate(bins_local):
            slot = bin_to_data.setdefault(tuple(b_tup), {})
            for v_pos, v_idx in enumerate(var_order_local):
                slot[v_idx] = (I_local[v_pos, b_idx], C_local[v_pos, b_idx])

    var_order = list(varis.keys())
    bins_sorted = sorted(bin_to_data.keys(), key=lambda t: (t[0], t[1], t[2]))
    n_bins = len(bins_sorted)
    I_pert = np.full((n_vars, n_bins, n_bt), np.nan, dtype=float)
    C_nu = np.full((n_vars, n_bins, n_bt), np.nan, dtype=float)
    for b_pos, b_tup in enumerate(bins_sorted):
        per_var = bin_to_data[b_tup]
        for v_pos, v_idx in enumerate(var_order):
            if v_idx in per_var:
                I_pert[v_pos, b_pos] = per_var[v_idx][0]
                C_nu[v_pos, b_pos] = per_var[v_idx][1]

    return {
        "bT": bT,
        "b_bar": b_bar,
        "bins": bins_sorted,
        "vars": varis,
        "var_order": var_order,
        "I_pert": I_pert,
        "C_nu": C_nu,
        "config": config,
        "n_shards": len(files),
    }


def _shard_glob(submitdir):
    for pat in (
        os.path.join(submitdir, "scetlib_outputs", "*_btgrid.pkl"),
        os.path.join(submitdir, "*_btgrid.pkl"),
    ):
        files = glob.glob(pat)
        if files:
            return files
    return []


def _combined_path(submitdir):
    return os.path.join(submitdir, _COMBINED_BASENAME)


def _cache_is_fresh(combined, shards):
    if not os.path.exists(combined):
        return False
    if not shards:
        return True  # nothing to compare against; trust the cache
    mtime = os.path.getmtime(combined)
    return mtime >= max(os.path.getmtime(s) for s in shards)


def load(submitdir, rebuild=False, verbose=True):
    """Load the combined bT-grid for ``submitdir``.

    On first call (or when ``rebuild=True``, or when any shard is newer than
    the cached combined file), assembles the shards via
    :func:`load_btgrid_shards`, writes ``combined_btgrid.pkl``, and returns
    the dict.

    On subsequent calls, loads the pickle directly.
    """
    if not os.path.isdir(submitdir):
        raise ValueError(f"{submitdir!r} is not a directory")

    combined = _combined_path(submitdir)
    shards = _shard_glob(submitdir)

    if not rebuild and _cache_is_fresh(combined, shards):
        t0 = time.time()
        with open(combined, "rb") as f:
            grid = pickle.load(f)
        if verbose:
            print(
                f"[btgrid_cache] loaded combined pickle in {time.time()-t0:.1f}s",
                flush=True,
            )
        return grid

    if not shards:
        raise FileNotFoundError(f"No btgrid shards found under {submitdir!r}")

    t0 = time.time()
    grid = load_btgrid_shards(submitdir)
    if verbose:
        print(
            f"[btgrid_cache] assembled {grid['n_shards']} shards in "
            f"{time.time()-t0:.1f}s; writing {combined}"
        )

    tmp = combined + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(grid, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, combined)
    return grid
