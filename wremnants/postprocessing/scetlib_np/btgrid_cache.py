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

from wremnants.postprocessing.scetlib_np import btgrid_numpy as fz

_COMBINED_BASENAME = "combined_btgrid.pkl"


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
    :func:`wremnants.postprocessing.scetlib_np.btgrid_numpy.load_btgrid_shards`,
    writes ``combined_btgrid.pkl``, and returns the dict.

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
    grid = fz.load_btgrid_shards(submitdir)
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
