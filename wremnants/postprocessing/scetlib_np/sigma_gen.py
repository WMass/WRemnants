"""SigmaGenModel — the datacard-free SCETlib NP σ_gen(λ) core (Steps 1–2).

The physics half of the SCETlib NP prediction, factored out of
:class:`~wremnants.postprocessing.scetlib_np.param_model.SCETlibNPParamModel` so
it runs from a bT-grid directory, λ_central (with the np_model strings), and the
gen-bin edges — no rabbit / datacard / fit input. Owns Steps 1–2:

    Step 1   btgrid Hankel + Q integral  →  σ_resum(λ; g)   resummed, on the gen grid
    Step 2   + fixed-order matching      →  σ_gen(λ; g)  = σ_resum(λ; g) + σ_ns(g)

Every factor (b*, I_pert, C_ν, γ_ν^NP, F_eff, the arctan-Q² Q integral, the
|Y|-fold and qT→ptVGen rebin, σ_ns matching) is derived in the ``param_model.py``
module docstring; this class is that arithmetic without loader/fit-interface
concerns. Steps 3–4 (gen→reco fold, per-bin ratio) stay in
``SCETlibNPParamModel``, which holds a ``SigmaGenModel`` as ``self.core``.

Public surface (used by the validation scripts and the σ_gen-at-λ tool):

  eff_central, gnu_central, np_model, np_model_nu   λ_central + functional forms
  gen_axes, gen_shape                               the (ptVGen, absYVGen) gen grid
  Y_unique, qT_unique, Q_unique                     btgrid native axes
  sigma_ns                                          NP-independent FO nonsingular
  sigma_YqT_central                                 native (NY, NqT) resum-only σ at λ_c
  sigma_gen_central                                 matched σ_gen on the gen grid at λ_c
  sigma_YqT_native(eff, gnu)                        native (NY, NqT) σ(λ), pre-fold
  sigma_gen(eff, gnu[, sigma_YqT])                  matched σ_gen(λ) on the gen grid

``eff``/``gnu`` are dicts of the λ values plus the ``np_model`` /
``np_model_nu`` form strings, same shape as ``eff_central`` / ``gnu_central``.
Start from those and override the λ you want; un-supplied params stay at
λ_central. All tensors are TF, so ``sigma_gen`` is differentiable in λ.
"""

import os
from typing import Optional

import numpy as np
import tensorflow as tf

from wremnants.postprocessing.scetlib_np import btgrid_cache
from wremnants.postprocessing.scetlib_np import btgrid_integrate as fz_int
from wremnants.postprocessing.scetlib_np import btgrid_tf as fz_tf
from wremnants.postprocessing.scetlib_np.params import (  # noqa: F401 (re-export)
    ALL_PARAMS,
    EFF_PARAMS,
    GNU_PARAMS,
    bin_sum_matrix,
)
from wremnants.utilities import common as wrem_common
from wremnants.utilities.data_paths import getDataPath

_NONSING_FO_SING_DEFAULT = os.path.join(
    wrem_common.data_dir,
    "TheoryCorrections",
    "inclusive_Z_COM13_CT18Z_N3+0LL_lattice_lambda4bugfix_fine_nnlo_sing_combined.pkl",
)
_NONSING_DYTURBO_DEFAULT = os.path.join(
    wrem_common.data_dir,
    "TheoryCorrections",
    "results_z-2d-nnlo-vj-CT18ZNNLO-{scale}-scetlibmatch.txt",
)
_BTGRID_SUBDIR = ("scetlib_np", "Z_COM13_CT18Z_N3p0LL_btgrid_fineall")

# λ name tuples (GNU_PARAMS / EFF_PARAMS / ALL_PARAMS) and bin_sum_matrix are
# imported above from :mod:`params` and re-exported here for back-compat.


def _default_btgrid_dir():
    base = getDataPath(fallback="/scratch/submit/cms/wmass/NanoAOD")
    return os.path.join(os.path.dirname(base), *_BTGRID_SUBDIR)


def compute_nonsingular_gen(
    fo_sing_path,
    dyturbo_path,
    gen_axes_meta,
    charge=0,
    q_lo=60.0,
    q_hi=120.0,
    qt_cutoff=1.0,
    dyturbo_axes=("Q", "Y", "qT"),
):
    """Nonsingular FO term on the model gen grid (NptVGen, NabsYVGen).

    The fixed-order/DYTurbo matching adds a NP-INDEPENDENT piece to σ_gen:
        σ_gen^matched(λ) = σ_gen^resum(λ) + σ_ns ,
        σ_ns = (DYTurbo fixed order) − (SCETlib singular fixed order) ,
    the ``-hfo_sing + hfo`` that ``read_matched_scetlib_hist`` forms.
    ``fo_sing_path`` is the SCETlib singular ``…_nnlo_sing…combined.pkl``;
    ``dyturbo_path`` is the DYTurbo FO ``results_…scetlibmatch.txt`` (``{scale}``
    → mur1-muf1 for the central). σ_ns is zeroed below ``qt_cutoff`` (as
    make_theory_corr does), Q-windowed to [q_lo, q_hi], |Y|-folded, and projected
    onto the coarse (ptVGen, absYVGen) gen bins by SUMMING the bin-integrated
    native bins.
    """
    from wremnants.utilities.io_tools import input_tools
    from wums import boostHistHelpers as hh

    def _central(h):
        if "vars" in h.axes.name:
            names = list(h.axes["vars"])
            idx = 0
            for c in ("central", "pdf0", "nominal"):
                if c in names:
                    idx = names.index(c)
                    break
            h = h[{"vars": idx}]
        return h

    # SCETlib singular FO and DYTurbo FO, from their own files.
    dyturbo_path = (
        dyturbo_path.format(scale="mur1-muf1")
        if "{scale}" in dyturbo_path
        else dyturbo_path
    )
    hfo_sing = _central(input_tools.read_scetlib_hist(fo_sing_path, charge=charge))
    hfo = input_tools.read_dyturbo_hist(
        [dyturbo_path], axes=list(dyturbo_axes), charge=charge
    )
    if "vars" in hfo.axes.name:
        hfo = _central(hfo)

    # Align shared physics axes (DYTurbo is coarser), then σ_ns = DYTurbo − singular.
    for ax in ("Y", "Q", "qT"):
        if ax in set(hfo.axes.name) & set(hfo_sing.axes.name):
            hfo, hfo_sing = hh.rebinHistsToCommon([hfo, hfo_sing], ax)
    nonsing_h = hh.addHists(-1.0 * hfo_sing, hfo, flow=False, by_ax_name=False)

    if "charge" in nonsing_h.axes.name:
        nonsing_h = nonsing_h[{"charge": sum}]
    # Q-window: slice(...,sum) sums ONLY the in-range Q bins (no underflow leak).
    Qe = np.asarray(nonsing_h.axes["Q"].edges, dtype=np.float64)
    qi = int(np.argmin(np.abs(Qe - q_lo)))
    qj = int(np.argmin(np.abs(Qe - q_hi)))
    nonsing_h = nonsing_h[{"Q": slice(qi, qj, sum)}]
    nonsing_h = hh.makeAbsHist(nonsing_h, "Y")  # signed Y -> |Y|

    qT_c = np.asarray(nonsing_h.axes["qT"].centers, dtype=np.float64)
    absY_c = np.asarray(nonsing_h.axes["absY"].centers, dtype=np.float64)
    v = nonsing_h.project("qT", "absY").values(flow=False)  # (qT, absY)
    v[qT_c < qt_cutoff, :] = 0.0  # zero the nonsingular below the cutoff

    ptV_edges = np.asarray(gen_axes_meta[0][1], dtype=np.float64)
    absY_edges = np.asarray(gen_axes_meta[1][1], dtype=np.float64)
    Wp = bin_sum_matrix(qT_c, ptV_edges)  # (NptVGen, NqT)
    Wa = bin_sum_matrix(absY_c, absY_edges)  # (NabsYVGen, NabsYsrc)
    return Wp @ v @ Wa.T  # (NptVGen, NabsYVGen)


# ============================================================================
# Factorized derived cache
# ============================================================================
# ``combined_btgrid.pkl`` (:mod:`btgrid_cache`) holds the RAW grid. Deriving the
# reconstruction layout from it (sanitize non-finite cells → dense index map →
# ``dedup_grid_rows`` → weighted J0 kernel) is the slow part of construction
# (~18 GB load + dedup) and a PURE function of that grid, so we memoize the
# derived arrays in an .npz next to the pickle; repeat constructions skip the raw
# load and dedup. Invalidated on combined-pickle change (mtime+size) OR derivation
# code change (bump _FACTORIZED_SCHEMA_VERSION). The combined pickle is REQUIRED:
# it is what freshness is verified against; absent → rebuild, not trust the .npz.
# The .npz lives in the btgrid dir, shared across all users of that grid.
_FACTORIZED_CACHE_BASENAME = "combined_btgrid.factorized.npz"
_FACTORIZED_SCHEMA_VERSION = "factorized_v1"
# The arrays that fully populate the factorized layout (see _assign_factorized).
_FACTORIZED_KEYS = (
    "flat_idx", "Q_unique", "Y_unique", "qT_unique", "bT", "b_bar",
    "Y_feff_unique", "bT_simpson_w", "I_pert_u", "C_nu_uu", "c_of_u",
    "feff_idx_u", "gather_idx", "KwqT",
)


def _build_factorized_arrays(grid):
    """Derive the factorized-layout numpy arrays from a raw combined grid.

    Sanitize non-finite cells, build the dense index map, dedup the (I_pert,
    C_nu) rows (bit-exact-verified inside ``dedup_grid_rows``), fold the per-qT
    prefactor + bT Simpson weights into the unique-qT J0 kernel. A pure function
    of ``grid``; the returned dict is what :meth:`_assign_factorized` turns into
    TF constants and what the derived cache stores."""
    # Sanitize non-finite bt-grid cells. Kinematically-forbidden points
    # (x = (Q/Ecm)·e^|Y| ≥ 1, e.g. extreme forward Y near the Z peak) come back
    # as NaN from SCETlib instead of the physical 0; their true σ is 0, and
    # dedup_grid_rows' hash-group verification needs finite cells (NaN != NaN).
    for _key in ("I_pert", "C_nu"):
        _arr = grid[_key]
        if not np.isfinite(_arr).all():
            _nbad = int((~np.isfinite(_arr)).any(axis=-1).sum())
            np.nan_to_num(_arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            print(
                f"[SigmaGenModel] sanitized {_nbad} non-finite {_key} bt-grid "
                f"rows -> 0 (kinematically-forbidden cells)",
                flush=True,
            )
    idx_map = fz_int.dense_index_map(grid["bins"])
    bins = grid["bins"]
    qT_pb = np.array([b[2] for b in bins], dtype=np.float64)
    Y_pb = np.array([b[1] for b in bins], dtype=np.float64)
    # F_eff depends on the bin only through Y (few distinct values): map to
    # unique Y so the NP transcendentals run on NY rows and gather.
    Y_feff_unique, Y_feff_inv = np.unique(Y_pb, return_inverse=True)
    Y_feff_inv = Y_feff_inv.reshape(-1).astype(np.int32)
    bT = np.asarray(grid["bT"], dtype=np.float64)
    b_bar = np.asarray(grid["b_bar"], dtype=np.float64)
    bT_simpson = np.asarray(fz_tf.simpson_weights(bT), dtype=np.float64)
    # Dedup the (I_pert, C_nu) rows (~2x), plus a 2nd-level C_nu dedup — both
    # verified bit-exact inside dedup_grid_rows.
    dd = fz_tf.dedup_grid_rows(grid["I_pert"][0], grid["C_nu"][0], Y_feff_inv)
    # Per-bin index into the unique-qT axis (exact lookup, asserted).
    qT_idx = np.searchsorted(idx_map["qT_unique"], qT_pb)
    assert np.array_equal(idx_map["qT_unique"][qT_idx], qT_pb)
    gather_idx = np.stack(
        [dd["row_uid"].astype(np.int64), qT_idx.astype(np.int64)], axis=1
    ).astype(np.int32)
    # Weighted J0 kernel on the unique-qT grid (per-qT prefactor + bT Simpson
    # weights folded in): (NqT, Nbt).
    qTu = np.asarray(idx_map["qT_unique"], dtype=np.float64)
    K_u = fz_tf.build_bT_J0_kernel(
        tf.constant(qTu, dtype=fz_tf.DTYPE), tf.constant(bT, dtype=fz_tf.DTYPE)
    )
    KwqT = (
        tf.constant(qTu, dtype=fz_tf.DTYPE)[:, tf.newaxis]
        * K_u
        * tf.constant(bT_simpson, dtype=fz_tf.DTYPE)[tf.newaxis, :]
    ).numpy()
    return {
        "flat_idx": np.asarray(idx_map["flat_idx"], dtype=np.int64),
        "Q_unique": np.asarray(idx_map["Q_unique"], dtype=np.float64),
        "Y_unique": np.asarray(idx_map["Y_unique"], dtype=np.float64),
        "qT_unique": qTu,
        "bT": bT,
        "b_bar": b_bar,
        "Y_feff_unique": np.asarray(Y_feff_unique, dtype=np.float64),
        "bT_simpson_w": bT_simpson,
        "I_pert_u": np.asarray(dd["I_u"], dtype=np.float64),
        "C_nu_uu": np.asarray(dd["C_uu"], dtype=np.float64),
        "c_of_u": np.asarray(dd["c_of_u"], dtype=np.int32),
        "feff_idx_u": np.asarray(dd["feff_idx_u"], dtype=np.int32),
        "gather_idx": gather_idx,
        "KwqT": KwqT,
    }


def _read_factorized_cache(cache_path, btgrid_dir):
    """Return the cached factorized arrays (an open ``NpzFile``) if present AND
    consistent with the current combined pickle, else ``None``.

    The combined pickle is REQUIRED so freshness can be VERIFIED: it must exist
    and be fresh vs its shards, the schema tag must match, and the combined
    (mtime+size) must match what was recorded at cache-write. Absent pickle →
    cannot verify → rebuild (from the shards) rather than trust a stale .npz. Any
    read error → ``None`` (rebuild)."""
    if not os.path.exists(cache_path):
        return None
    combined = btgrid_cache.combined_path(btgrid_dir)
    if not (os.path.exists(combined) and btgrid_cache.is_combined_fresh(btgrid_dir)):
        return None
    try:
        z = np.load(cache_path, allow_pickle=False)
        st = os.stat(combined)
        if (
            str(z["schema_version"].item()) != _FACTORIZED_SCHEMA_VERSION
            or int(z["combined_mtime_ns"].item()) != int(st.st_mtime_ns)
            or int(z["combined_size"].item()) != int(st.st_size)
            or any(k not in z.files for k in _FACTORIZED_KEYS)
        ):
            return None
    except Exception as exc:  # corrupt / partial cache -> rebuild
        print(
            f"[SigmaGenModel] ignoring unreadable factorized cache "
            f"{cache_path} ({exc})",
            flush=True,
        )
        return None
    return z


def _write_factorized_cache(cache_path, arr, btgrid_dir):
    """Atomically write the derived arrays + a freshness header next to the
    combined pickle. PID-unique temp name so concurrent builders (e.g. parallel
    fit jobs) don't clobber each other; ``os.replace`` is atomic."""
    st = os.stat(btgrid_cache.combined_path(btgrid_dir))
    header = {
        "schema_version": np.array(_FACTORIZED_SCHEMA_VERSION),
        "combined_mtime_ns": np.int64(st.st_mtime_ns),
        "combined_size": np.int64(st.st_size),
    }
    tmp = f"{cache_path}.tmp.{os.getpid()}.npz"
    np.savez(tmp, **header, **arr)
    os.replace(tmp, cache_path)


class SigmaGenModel:
    """Btgrid → σ_gen(λ) on a fixed (ptVGen, absYVGen) gen grid. See module docstring."""

    def __init__(
        self,
        btgrid_dir: Optional[str] = None,
        lambda_central: Optional[dict] = None,
        gen_axes=None,
        Q_lo: float = 60.0,
        Q_hi: float = 120.0,
        nonsingular_fo_sing: str = _NONSING_FO_SING_DEFAULT,
        nonsingular_dyturbo: str = _NONSING_DYTURBO_DEFAULT,
        nonsingular_qt_cutoff: float = 1.0,
        include_nonsingular: bool = True,
    ):
        """Build the σ_gen core.

        Parameters
        ----------
        btgrid_dir
            Directory holding the SCETlib bT-grid ``combined_btgrid.pkl``.
            Defaults (when None) to the shared data-area copy next to NanoAOD.
        lambda_central
            Dict with ``eff_params`` and ``gnu_params`` sub-dicts (same shape as
            :func:`lambda_central.read_lambda_central`). Carries both the central
            λ values and the ``np_model`` / ``np_model_nu`` functional-form strings.
        gen_axes
            Ordered ``[("ptVGen", edges), ("absYVGen", edges)]`` — the gen grid
            σ_gen is rebinned onto. ``edges`` are 1-D arrays of bin edges.
        Q_lo, Q_hi
            Z mass window for the Q-integration on the btgrid.
        nonsingular_fo_sing, nonsingular_dyturbo, nonsingular_qt_cutoff
            σ_ns = DYTurbo − SCETlib_singular inputs / low-qT cutoff (see
            :func:`compute_nonsingular_gen`).
        include_nonsingular
            Add the matched FO nonsingular σ_ns (default True — the matched σ_gen
            the histmaker nominal carries). False → resum-only (σ_ns = 0), FO
            inputs not read.

        The derived factorized btgrid layout is always memoized in an .npz next to
        ``combined_btgrid.pkl`` (see "Factorized derived cache" above). No on/off
        knob: staleness is auto-detected, and a fresh cache lets construction skip
        the ~18 GB raw load and the row dedup.
        """
        if lambda_central is None:
            raise ValueError("SigmaGenModel requires lambda_central (eff/gnu params).")
        if gen_axes is None or len(gen_axes) != 2:
            raise ValueError(
                "SigmaGenModel requires gen_axes = [(ptVGen, edges), (absYVGen, edges)]."
            )
        if btgrid_dir is None:
            btgrid_dir = _default_btgrid_dir()

        # ---- λ_central + functional forms.
        self.eff_central = dict(lambda_central["eff_params"])
        self.gnu_central = dict(lambda_central["gnu_params"])
        self.np_model = self.eff_central["np_model"]
        self.np_model_nu = self.gnu_central["np_model_nu"]

        # ---- gen grid.
        self.gen_axes = [
            (name, np.asarray(edges, dtype=np.float64)) for (name, edges) in gen_axes
        ]
        self.gen_shape = tuple(len(e) - 1 for (_, e) in self.gen_axes)

        # ---- btgrid factorized reconstruction layout (Step-1 tensors). Loaded
        # from the derived .npz cache when fresh, else built from the raw grid
        # (sanitize → dense index → row dedup → weighted J0 kernel) and cached.
        # Sets: flat_idx, Q/Y/qT_unique, bT, b_bar, Y_feff_unique, bT_simpson_w,
        # I_pert_u, C_nu_uu, c_of_u, feff_idx_u, gather_idx, KwqT.
        self._setup_btgrid(btgrid_dir)

        # ---- Q-integration weights (arctan_Q² Simpson on Z mass window).
        self.Q_weights = tf.constant(
            fz_int.q_integrate_weights(self.Q_unique, Q_lo, Q_hi),
            dtype=fz_tf.DTYPE,
        )

        # ---- Rebin weights: btgrid (NY signed) → (NabsYVGen) via |Y| folding,
        # (NqT) → (NptVGen).
        ptVGen_edges = self.gen_axes[0][1]
        absY_edges = self.gen_axes[1][1]
        # |Y| folding: σ(Y) symmetric in Y, so the absY-bin integral is
        # 2·∫_{absY_lo}^{absY_hi} σ(Y) dY. Use Y >= 0 samples, multiply by 2.
        Y_pos_mask = self.Y_unique >= 0
        Y_pos = self.Y_unique[Y_pos_mask]
        absY_rebin_pos = fz_int.rebin_weights(Y_pos, absY_edges, name="absY")
        # Pad to full NY: zero on negative-Y columns.
        W_absY = np.zeros((absY_edges.size - 1, self.Y_unique.size), dtype=np.float64)
        W_absY[:, Y_pos_mask] = 2.0 * absY_rebin_pos
        self.W_absY = tf.constant(W_absY, dtype=fz_tf.DTYPE)
        # qT rebin: btgrid qT (signed nonneg, NqT=141) → ptVGen edges. With a
        # ptVGen overflow bin [last_gen_edge, OVERFLOW_EDGE] (e.g. [44, 100]),
        # rebin_weights' last row Simpson-integrates the btgrid tail qT∈(44,100]
        # into it; btgrid qT past the last edge (>100, off-grid) is dropped
        # (negligible).
        self.W_ptVGen = tf.constant(
            fz_int.rebin_weights(self.qT_unique, ptVGen_edges, name="ptVGen"),
            dtype=fz_tf.DTYPE,
        )

        # ---- Native (NY, NqT) Q-integrated reconstruction at λ_central, BEFORE
        # the |Y|-fold and qT-rebin — exposed so the native-binning validation
        # compares it to the SCETlib reference without the projection layer.
        self.sigma_YqT_central = self.sigma_YqT_native(self.eff_central, self.gnu_central)

        # ---- Fixed-order/DYTurbo nonsingular term (NP-independent).
        # σ_gen^matched(λ) = σ_gen^resum(λ) + σ_ns, added at GEN level so it folds
        # through the same response R as the resummed piece. σ_ns is constant (no
        # λ dependence); included by default (the matched σ_gen the histmaker
        # nominal carries). include_nonsingular=False → resum-only.
        if include_nonsingular:
            _dy0 = (
                nonsingular_dyturbo.format(scale="mur1-muf1")
                if (nonsingular_dyturbo and "{scale}" in nonsingular_dyturbo)
                else nonsingular_dyturbo
            )
            missing = [
                p for p in (nonsingular_fo_sing, _dy0) if not (p and os.path.exists(p))
            ]
            if missing:
                raise FileNotFoundError(
                    "The matched model needs the fixed-order inputs for "
                    "σ_ns = DYTurbo − SCETlib_singular, but these are missing:\n  "
                    + "\n  ".join(missing)
                    + "\nThey live under wremnants-data/data/TheoryCorrections (the "
                    "SCETlib singular …_nnlo_sing…combined.pkl and the DYTurbo "
                    "results_…scetlibmatch.txt). Pass nonsingular_fo_sing / "
                    "nonsingular_dyturbo to point at them (or include_nonsingular="
                    "False for resum-only)."
                )
            sigma_ns_np = compute_nonsingular_gen(
                nonsingular_fo_sing,
                nonsingular_dyturbo,
                self.gen_axes,
                q_lo=Q_lo,
                q_hi=Q_hi,
                qt_cutoff=nonsingular_qt_cutoff,
            )
            if sigma_ns_np.shape != tuple(self.gen_shape):
                raise ValueError(
                    f"nonsingular gen shape {sigma_ns_np.shape} != model gen shape "
                    f"{tuple(self.gen_shape)}"
                )
            self.sigma_ns = tf.constant(sigma_ns_np, dtype=fz_tf.DTYPE)
        else:
            self.sigma_ns = tf.zeros(self.gen_shape, dtype=fz_tf.DTYPE)

        # ---- Matched σ_gen(λ_central). Reuse the native (NY, NqT) integral
        # already computed for sigma_YqT_central (no 2nd bT reconstruction).
        sigma_gen_central = self.sigma_gen(
            self.eff_central, self.gnu_central, sigma_YqT=self.sigma_YqT_central
        )
        self.sigma_gen_central = sigma_gen_central
        gen_flat = tf.reshape(sigma_gen_central, [-1])
        if tf.reduce_any(gen_flat <= 0).numpy():
            n_bad = int(tf.reduce_sum(tf.cast(gen_flat <= 0, tf.int32)))
            raise ValueError(
                f"SigmaGenModel: {n_bad} gen bins have non-positive "
                f"σ_gen(λ_central); cannot normalize / fold the response."
            )

    # =========================================================================
    # btgrid factorized layout (derived-cache aware)
    # =========================================================================

    def _setup_btgrid(self, btgrid_dir):
        """Populate the factorized-layout tensors: from the derived cache when
        usable (see :func:`_read_factorized_cache`), else built from the combined
        grid and cached next to it for reuse. Staleness handled automatically
        (combined mtime+size + schema)."""
        cache_path = os.path.join(btgrid_dir, _FACTORIZED_CACHE_BASENAME)
        z = _read_factorized_cache(cache_path, btgrid_dir)
        if z is not None:
            self._assign_factorized(z)
            print(
                f"[SigmaGenModel] loaded factorized btgrid cache {cache_path}",
                flush=True,
            )
            return
        grid = btgrid_cache.load(btgrid_dir)
        arr = _build_factorized_arrays(grid)
        del grid  # free the ~18 GB host grid before TF graph build
        self._assign_factorized(arr)
        try:
            _write_factorized_cache(cache_path, arr, btgrid_dir)
            print(
                f"[SigmaGenModel] wrote factorized btgrid cache {cache_path}",
                flush=True,
            )
        except OSError as exc:  # read-only area etc. — still usable, just slow
            print(
                f"[SigmaGenModel] WARNING: could not write factorized cache "
                f"{cache_path} ({exc}); continuing without it",
                flush=True,
            )

    def _assign_factorized(self, a):
        """Set the factorized-layout attributes from a mapping of numpy arrays
        (:func:`_build_factorized_arrays` output or a loaded npz). Q/Y/qT_unique
        stay numpy (used in numpy rebin-weight construction); the rest become TF
        constants."""
        D = fz_tf.DTYPE
        self.flat_idx = tf.constant(a["flat_idx"], dtype=tf.int64)
        self.Q_unique = np.asarray(a["Q_unique"], dtype=np.float64)
        self.Y_unique = np.asarray(a["Y_unique"], dtype=np.float64)
        self.qT_unique = np.asarray(a["qT_unique"], dtype=np.float64)
        self.bT = tf.constant(a["bT"], dtype=D)
        self.b_bar = tf.constant(a["b_bar"], dtype=D)
        self.Y_feff_unique = tf.constant(a["Y_feff_unique"], dtype=D)
        self.bT_simpson_w = tf.constant(a["bT_simpson_w"], dtype=D)
        self.I_pert_u = tf.constant(a["I_pert_u"], dtype=D)  # (Nu, Nbt)
        self.C_nu_uu = tf.constant(a["C_nu_uu"], dtype=D)  # (Ncu, Nbt)
        self.c_of_u = tf.constant(a["c_of_u"], dtype=tf.int32)
        self.feff_idx_u = tf.constant(a["feff_idx_u"], dtype=tf.int32)
        self.gather_idx = tf.constant(a["gather_idx"], dtype=tf.int32)
        self.KwqT = tf.constant(a["KwqT"], dtype=D)

    # =========================================================================
    # σ_gen evaluation
    # =========================================================================

    def sigma_YqT_native(self, eff_params, gnu_params, np_model=None, np_model_nu=None):
        """Reconstruct σ(λ) on the btgrid and Q-integrate, in the btgrid's
        *native* binning: shape (NY, NqT) on the signed-Y / qT grid (Y_unique,
        qT_unique), BEFORE the |Y|-fold and qT-rebin. The object the native-binning
        validation compares against the SCETlib spectrum reference (curve 1) and
        the external scetlib_run.factorize (curve 2).

        ``np_model`` / ``np_model_nu`` override the functional form applied to the
        λ for THIS evaluation (default: the construction forms ``self.np_model`` /
        ``self.np_model_nu``). The bt-grid (I_pert, C_nu) is NP-model-independent,
        so a different form is just a different analytic factor on the same grid —
        used to evaluate a numerator in one form while the denominator (central)
        stays in the card's form (see ``SCETlibNPParamModel``)."""
        # 1. Reconstruct σ on the btgrid's flat (Nbins,) layout via the
        # memory-factorized path (deduplicated rows + unique-qT J0 kernel +
        # Simpson-as-matmul) — ~6x smaller than dense (Nbins, Nbt), which is what
        # lets the fit run on a 32 GB GPU.
        eff = {k: v for k, v in eff_params.items() if k != "np_model"}
        gnu = {k: v for k, v in gnu_params.items() if k != "np_model_nu"}
        sigma_flat = fz_tf.reconstruct_batch_factorized_tf(
            b_bar=self.b_bar,
            I_pert_u=self.I_pert_u,
            C_nu_uu=self.C_nu_uu,
            c_of_u=self.c_of_u,
            eff_params=eff,
            gnu_params=gnu,
            np_model=np_model or self.np_model,
            np_model_nu=np_model_nu or self.np_model_nu,
            KwqT=self.KwqT,
            gather_idx=self.gather_idx,
            Y_unique=self.Y_feff_unique,
            feff_idx_u=self.feff_idx_u,
        )
        # 2. Sparse → dense (NQ, NY, NqT). Missing cells get 0.
        sigma_dense = fz_int.sparse_to_dense_tf(sigma_flat, self.flat_idx)
        # 3. Integrate over Q (arctan_Q² Simpson) → (NY, NqT).
        return fz_int.integrate_over_Q_tf(sigma_dense, self.Q_weights)

    def sigma_gen(
        self, eff_params, gnu_params, sigma_YqT=None, np_model=None, np_model_nu=None
    ):
        """Evaluate matched σ_gen(λ) on the gen binning. Returns (NptVGen, NabsYVGen).

        ``sigma_YqT`` passes an already-computed native (NY, NqT) integral to skip
        the expensive bT reconstruction; used at construction to reuse
        ``sigma_YqT_central`` instead of integrating λ_central twice.
        ``np_model`` / ``np_model_nu`` override the functional form for this
        evaluation (default: the construction forms) — see ``sigma_YqT_native``.
        """
        if sigma_YqT is None:
            sigma_YqT = self.sigma_YqT_native(
                eff_params, gnu_params, np_model=np_model, np_model_nu=np_model_nu
            )
        # 4. Rebin Y (signed) → absYVGen (|Y|-folded): (NabsYVGen, NqT).
        sigma_absY_qT = fz_int.rebin_axis_tf(sigma_YqT, axis=0, weights=self.W_absY)
        # 5. Rebin qT → ptVGen: (NabsYVGen, NptVGen).
        sigma_absY_ptV = fz_int.rebin_axis_tf(
            sigma_absY_qT, axis=1, weights=self.W_ptVGen
        )
        # 6. Reorder to (NptVGen, NabsYVGen) to match R's gen axis order.
        sigma_resum = tf.transpose(sigma_absY_ptV, perm=[1, 0])
        # 7. Add the NP-independent fixed-order/DYTurbo nonsingular (zeros if off).
        return sigma_resum + self.sigma_ns
