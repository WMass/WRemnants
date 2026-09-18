"""SCETlib autodiff backend: cached matched sigma_UL with exact derivatives.

Thin, numpy-facing wrapper around ``ScetlibCachedXsecTF``
(``scetlib-cms/py/scetlib_tf.py``). It owns three things the param model should
not have to know about:

* rebuilding the SCETlib calculation exactly as the cache was prepared with
  (``configure_calculation`` is NOT the whole configuration -- the profile-scale
  / b* block and the nonperturbative models go through
  ``variations.set_vary``, so skipping it
  silently keeps the C++ defaults);
* loading the one-file cache (compressed bin rules for the resummed piece +
  frozen fixed-order grid for the nonsingular one) and checking it belongs to
  this configuration;
* mapping the cache's bin list onto the fit's gen grid, by an explicit
  permutation rather than by assuming both were generated in the same order.

Derivatives come from C++: ``values_and_jacobian`` returns value and Jacobian
from one call, ``hessian`` the exact per-bin Hessian. The param model injects
those into the TF graph as a local quadratic, so nothing here needs to be
differentiable.

Runtime requirements (inside the WRemnants singularity):
``source <scetlib-cms>/setup.sh`` -- it puts ``scetlib_qT`` and ``scetlib_run``
on ``PYTHONPATH``, the shared libraries on ``LD_LIBRARY_PATH``, and lifts the
stack limit (the bT integrators are deep). LHAPDF and the per-member beamfunc
grids under ``share/scetlib/beamfunc/`` must be reachable.
"""

import configparser
import os

import numpy as np

# Bin-edge matching tolerance. Gen edges come from hist axes (fp64 literals) and
# cache edges from a .npz round trip of the same literals, so exact equality is
# the expectation and this only absorbs formatting round trips.
EDGE_TOL = 1e-9


_IMPORT_HINT = (
    "scetlib_ad needs the SCETlib autodiff build on PYTHONPATH. Run\n"
    "    source <path-to>/scetlib-cms/setup.sh\n"
    "inside the container before rabbit_fit.py (it also sets LD_LIBRARY_PATH "
    "and lifts the stack limit).\n"
)


def _import_scetlib():
    """The calculation and the runcard helpers, with an actionable error.

    Deliberately does NOT import ``scetlib_tf``. That module imports
    TensorFlow, and TensorFlow is only needed to EVALUATE a cache, not to
    configure the calculation or to build one -- measured at 804 MB of RSS and
    one ``tf_Compute`` thread per core (1665 threads against 135 with the pools
    capped), charged against a 32768-threads-per-user ceiling whose failure
    mode is a ``pthread_create`` abort landing on whichever process next asks
    for a thread. A cache build calls ``configure`` and nothing else here, so
    it must not pay for that; see :func:`_import_cached_xsec`.
    """
    try:
        import scetlib_qT  # noqa: F401
        from scetlib_run import config as sl_config
        from scetlib_run import variations as sl_variations
    except ImportError as e:
        raise ImportError(_IMPORT_HINT + f"Original error: {e}") from e
    return sl_config, sl_variations


def _import_cached_xsec():
    """``ScetlibCachedXsecTF``, and with it TensorFlow. Evaluation only."""
    try:
        from scetlib_tf import ScetlibCachedXsecTF
    except ImportError as e:
        raise ImportError(_IMPORT_HINT + f"Original error: {e}") from e
    return ScetlibCachedXsecTF


def _scetlib_src():
    """Repository root of the SCETlib checkout (for prod/scetlib_run/defaults.conf)."""
    src = os.environ.get("SCETLIB_SRC")
    if src:
        return src
    # setup.sh puts <src>/prod/scetlib_run on PYTHONPATH, so the package sits at
    # <src>/prod/scetlib_run/scetlib_run/.
    import scetlib_run

    return os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(scetlib_run.__file__)))
        )
    )


def configure(config_path, threads=0, diff_scales=True, fo_resolve_muR=True):
    """Rebuild the calculation the cache was prepared with.

    Follows what ``prod/scetlib_run/scetlib-run-qT.py`` does, which is the
    path every production correction was made with: layer the runcard on
    ``prod/scetlib_run/defaults.conf``, configure the calculation, apply the
    card's electroweak parameters and fiducial volumes, then apply variation 0
    (central) through ``set_vary`` -- which is what installs the profile-scale
    / b* block and the nonperturbative models. Finally enable the frozen-node
    cache on both sub-pieces, without which the rule replay has nothing to
    replay against.

    ``configure_ew_parameters`` and ``configure_fiducial_volumes`` are NOT
    optional and are NOT part of ``configure_calculation``. Skipping them
    silently keeps SCETlib's default electroweak inputs while the card
    specifies its own -- measured as a flat **1.61%** normalization error
    against the production driver on the analysis card, which sets
    ``mZ = 91.1535``, ``GammaZ = 2.4932``, a custom ``alphaem``, ``sin2_thw``
    and CKM. With them applied, ``operator()`` here reproduces
    ``scetlib-run-qT.py`` to 1e-9. Both were once missing from the upstream
    example, which now applies them too -- do not "simplify" either away.

    ``diff_scales`` registers muR and the three matching transition points as
    differentiable parameters (``scale_kappa_R``, ``scale_x1..x3``, plus an inert
    ``scale_kappa_F`` slot that ``build_pdf_variations`` ties the muF pair to).
    It REQUIRES ``muf_follows_muB = no``: with muf tied to muB a live kappa_R
    moves muF while the beam convolutions stay frozen at their own muF. It also
    changes the parameter registry, so a cache built without it cannot be loaded
    with it -- :class:`ScetlibADXsec` therefore reads the flag off the cache
    rather than assuming.

    ``fo_resolve_muR`` resolves the fixed-order muR dependence into the frozen
    grid so the FO piece follows kappa_R in closed form. It must be set BEFORE
    the grid is built -- enabling it afterwards invalidates the cache -- and it
    samples several scales, so the warm costs roughly 3x.

    Returns ``(conf, sigma)``.
    """
    sl_config, sl_variations = _import_scetlib()
    src = _scetlib_src()
    conf = configparser.ConfigParser(inline_comment_prefixes="#")
    conf.read(os.path.join(src, "prod", "scetlib_run", "defaults.conf"))
    if not conf.read(config_path):
        raise FileNotFoundError(f"scetlib_ad: cannot read runcard {config_path!r}")

    order, alphas, decay, scales, sigma = sl_config.configure_calculation(conf)
    sl_config.configure_ew_parameters(conf, sigma)
    sl_config.configure_fiducial_volumes(conf, decay)
    if diff_scales:
        follows = conf["Calculation_settings"].get("muf_follows_muB", "no")
        if str(follows).strip().lower() in ("yes", "true", "1"):
            raise ValueError(
                "scetlib_ad: diff_scales needs muf_follows_muB = no, but the "
                f"runcard sets muf_follows_muB = {follows}. With muf tied to muB "
                "a live kappa_R moves muF while the beam convolutions stay "
                "frozen at their own muF."
            )
        sigma.set_diff_scales(1)
    if fo_resolve_muR:
        for piece in sigma.sub_pieces():
            piece.set_fo_resolve_muR(True)
    varis = sl_variations.configure_variations(
        conf,
        os.path.join(os.path.dirname(os.path.abspath(config_path)), "variations.conf"),
    )
    sl_variations.set_vary(varis[0], order, alphas, scales, sigma)

    nthreads = int(threads) if threads else (os.cpu_count() or 8)
    for piece in sigma.sub_pieces():
        piece.set_gradient_threads(nthreads)
        piece.set_gradient_node_cache(True)
    return conf, sigma


def bins_from_gen_axes(gen_axes, Q_lo, Q_hi):
    """(N, 6) SCETlib bins for a (qT, |Y|) gen grid, in the grid's flatten order.

    ``gen_axes`` is ``[(qT_name, edges), (Y_name, edges)]`` -- the order the param
    model flattens in, qT-major. The rapidity edges are used as given: pass the
    positive side only to build a folded-|Y| cache (correct for the Z, where the
    factor 2 cancels in the ratio), or signed edges to build a signed one (needed
    for W, and for a bin-by-bin comparison against a SCETlib reference run on its
    own signed grid). :class:`GenFold` accepts either.
    """
    qT_edges = np.asarray(gen_axes[0][1], dtype=np.float64)
    absY_edges = np.asarray(gen_axes[1][1], dtype=np.float64)
    return np.array(
        [
            [Q_lo, Q_hi, absY_edges[j], absY_edges[j + 1], qT_edges[i], qT_edges[i + 1]]
            for i in range(qT_edges.size - 1)
            for j in range(absY_edges.size - 1)
        ],
        dtype=np.float64,
    )


def _containing_bin(lo, hi, edges, what):
    """Index of the ``edges`` bin that [lo, hi] falls inside, or None."""
    i = int(np.searchsorted(edges, lo + EDGE_TOL) - 1)
    if i < 0 or i >= edges.size - 1:
        return None
    if hi > edges[i + 1] + EDGE_TOL:
        raise ValueError(
            f"scetlib_ad: cache {what} bin [{lo:g}, {hi:g}] straddles the gen "
            f"edge {edges[i + 1]:g}. The cache binning must nest inside the "
            f"fit's gen binning (a gen bin may be the sum of several cache bins, "
            f"but a cache bin may not span two gen bins)."
        )
    return i


class GenFold:
    """Exact sum of cache bins onto the fit's gen grid.

    A plain reindexing is not enough in general. Three things can differ between
    the cache and the fit's gen binning, and all three are handled here by
    SUMMING bin-integrated cross sections, which is exact:

    * **order** -- ``prepare_cache`` nests (Q, Y, qT) with qT innermost, while
      the gen grid (and the response matrix's gen columns) flatten qT-major;
    * **rapidity sign** -- a production cache is built on the SIGNED Y grid of
      the theory correction, so a gen |Y| bin is the sum of the +Y and -Y cache
      bins. A cache built on the positive side only is also accepted, and then
      every gen bin is short by the same factor 2 -- which cancels in the ratio
      to the central, and is reported rather than silently applied;
    * **granularity** -- a fine cache (e.g. the 70 x 82 correction grid) folds
      exactly onto any coarser gen grid whose edges are a subset of its own.

    Coverage is checked, not assumed: every gen bin must be tiled exactly by the
    cache bins assigned to it, so a cache that only partially covers a gen bin
    raises instead of quietly integrating over less phase space.
    """

    def __init__(self, bins, gen_axes, Q_lo, Q_hi, partial=False):
        """``partial`` keeps a fold over a cache that tiles only SOME gen bins.

        The fit must never use this: a partially covered gen bin folds to a
        partial sum, which is a wrong cross section rather than a missing one, so
        the default stays strict. It exists for validation on a subset cache --
        ten bins built in a minute instead of 210 in seven hours -- where the
        caller compares only the bins ``covered_mask`` marks and says so. The
        mask is the whole point: without it a subset cache would silently report
        agreement on bins it never computed.
        """
        bins = np.asarray(bins, dtype=np.float64)
        qT_edges = np.asarray(gen_axes[0][1], dtype=np.float64)
        absY_edges = np.asarray(gen_axes[1][1], dtype=np.float64)
        n_qt, n_y = qT_edges.size - 1, absY_edges.size - 1
        self.n_gen = n_qt * n_y
        self.gen_shape = (n_qt, n_y)

        rows = np.full(bins.shape[0], -1, dtype=np.int64)
        # (qT, |Y|) area each gen bin actually receives, to verify exact tiling.
        covered = np.zeros((n_qt, n_y), dtype=np.float64)
        sides = np.zeros((n_qt, n_y, 2), dtype=np.int64)  # [-Y, +Y] contributions
        for k, (q_lo, q_hi, y_lo, y_hi, t_lo, t_hi) in enumerate(bins):
            if q_lo < Q_lo - EDGE_TOL or q_hi > Q_hi + EDGE_TOL:
                raise ValueError(
                    f"scetlib_ad: cache Q bin [{q_lo:g}, {q_hi:g}] lies outside "
                    f"the fit's mass window [{Q_lo:g}, {Q_hi:g}]."
                )
            if y_lo < -EDGE_TOL and y_hi > EDGE_TOL:
                raise ValueError(
                    f"scetlib_ad: cache Y bin [{y_lo:g}, {y_hi:g}] straddles "
                    f"Y = 0 and cannot be folded onto |Y|."
                )
            negative = y_hi <= EDGE_TOL
            a, b = (abs(y_hi), abs(y_lo)) if negative else (y_lo, y_hi)
            j = _containing_bin(a, b, absY_edges, "|Y|")
            i = _containing_bin(t_lo, t_hi, qT_edges, "qT")
            if i is None or j is None:
                continue  # outside the fit's gen range: dropped, accounted below
            rows[k] = i * n_y + j
            covered[i, j] += (t_hi - t_lo) * (b - a)
            sides[i, j, 0 if negative else 1] += 1

        self.n_used = int((rows >= 0).sum())
        self.n_dropped = int(bins.shape[0] - self.n_used)

        # Dropped bins are CORRECT -- they fall outside the card's gen range and
        # the fold is still exact -- but they are not free: every one is computed
        # by SCETlib on every call and then thrown away. A couple (the ptVGen
        # overflow that load_R drops) is normal; a large fraction means the cache
        # is much finer than this card can use, which is a silent waste rather
        # than an error, and it also makes any timing attributed to "the cache"
        # misleading. Measured 2026-09-07: a 770-bin cache against a card that
        # could only use 627 of them discarded 143 (19 %) on every evaluation.
        if bins.shape[0] and self.n_dropped > 0.05 * bins.shape[0]:
            print(
                f"scetlib_ad: WARNING -- {self.n_dropped} of {bins.shape[0]} "
                f"cache bin(s) ({100.0 * self.n_dropped / bins.shape[0]:.1f} %) "
                f"fall outside this card's gen range. They are evaluated on "
                f"every call and discarded. The result is exact; the cost is "
                f"not. Either use a cache built for this card's gen binning, or "
                f"a card whose gen grid uses the cache you have."
            )

        empty = sides.sum(axis=-1) == 0
        if empty.any():
            i, j = np.argwhere(empty)[0]
            raise ValueError(
                f"scetlib_ad: {int(empty.sum())} gen bin(s) receive no cache bin "
                f"at all, e.g. qT [{qT_edges[i]:g}, {qT_edges[i + 1]:g}] x |Y| "
                f"[{absY_edges[j]:g}, {absY_edges[j + 1]:g}]. The cache does not "
                f"cover this card's gen range; rebuild it for this card."
            )
        both = (sides[..., 0] > 0) & (sides[..., 1] > 0)
        pos_only = (sides[..., 0] == 0) & (sides[..., 1] > 0)
        if both.all():
            self.y_convention = "signed"
            y_factor = 2.0
        elif pos_only.all():
            self.y_convention = "positive-side-only"
            y_factor = 1.0
        else:
            raise ValueError(
                "scetlib_ad: the cache covers |Y| on both signed sides for some "
                "gen bins and only one side for others, so the fold would apply "
                "an inconsistent factor 2 across the grid. Rebuild the cache on "
                "a uniform Y convention."
            )
        # An exactly tiled gen bin receives y_factor x its own (qT, |Y|) area.
        want = np.diff(qT_edges)[:, None] * np.diff(absY_edges)[None, :] * y_factor
        bad = np.abs(covered - want) > EDGE_TOL * np.maximum(want, 1.0)
        # (n_qT, n_absY) in the gen grid's own shape; True = exactly tiled and so
        # safe to compare. Always set, so a caller can assert on it either way.
        self.covered_mask = ~bad
        if bad.any() and partial:
            i, j = np.argwhere(bad)[0]
            print(
                f"scetlib_ad: PARTIAL fold -- {int(bad.sum())} of {bad.size} gen "
                f"bin(s) are not tiled by this cache and fold to partial sums, "
                f"e.g. qT [{qT_edges[i]:g}, {qT_edges[i + 1]:g}] x |Y| "
                f"[{absY_edges[j]:g}, {absY_edges[j + 1]:g}]. Compare only where "
                f"covered_mask is True; never fit with this."
            )
        elif bad.any():
            i, j = np.argwhere(bad)[0]
            raise ValueError(
                f"scetlib_ad: {int(bad.sum())} gen bin(s) are not exactly tiled "
                f"by the cache, e.g. qT [{qT_edges[i]:g}, {qT_edges[i + 1]:g}] x "
                f"|Y| [{absY_edges[j]:g}, {absY_edges[j + 1]:g}] receives "
                f"{covered[i, j]:g} of {want[i, j]:g} in (qT, |Y|) area. The cache "
                f"does not cover this card's gen binning; rebuild it."
            )

        # Group the surviving cache rows by destination so the fold is a single
        # reduceat rather than a scatter-add (which is slow, and is on the hot
        # path once per loss/gradient evaluation).
        keep = np.nonzero(rows >= 0)[0]
        self._order = keep[np.argsort(rows[keep], kind="stable")]
        self._starts = np.searchsorted(rows[self._order], np.arange(self.n_gen))
        # Destination gen bin per cache bin, -1 for the ones outside the gen
        # range. tf.math.unsorted_segment_sum drops negative ids, so this is the
        # same fold expressed for the in-graph path (see fold_tf).
        self.segment_ids = rows

    def __call__(self, a):
        """Fold a cache-indexed array (first axis = cache bin) onto the gen grid."""
        return np.add.reduceat(np.asarray(a)[self._order], self._starts, axis=0)

    def fold_tf(self, a):
        """The same fold, as TensorFlow ops, for the differentiate-through path."""
        import tensorflow as tf

        return tf.math.unsorted_segment_sum(
            a, tf.constant(self.segment_ids, dtype=tf.int32), self.n_gen
        )

    def describe(self):
        return (
            f"{self.n_used} cache bin(s) -> {self.n_gen} gen bin(s), "
            f"Y convention {self.y_convention}"
            + (
                f", {self.n_dropped} cache bin(s) outside the gen range"
                if self.n_dropped
                else ""
            )
        )


class ScetlibADXsec:
    """Cached matched sigma_UL(p) on a fixed bin set, with exact derivatives.

    Parameters
    ----------
    conf_path
        SCETlib runcard the cache was built from (layered on defaults.conf).
    cache_path
        The ``.npz`` written by ``ScetlibCachedXsecTF.save`` -- rules, frozen
        fixed-order grid, bins, anchor and parameter names in one file.
    threads
        Worker threads for the batch replay (0 = one per hardware thread).
    """

    @staticmethod
    def cache_param_names(cache_path):
        """The parameter names a cache was built with, without loading it.

        ``.npz`` is a zip of individual arrays, so this reads only ``names``.
        Used to configure the calculation the way the cache expects instead of
        the way we would prefer -- a mismatch is otherwise a hard load failure
        (the fingerprint hashes the names in order).
        """
        with np.load(cache_path, allow_pickle=False) as z:
            if "names" not in z.files:
                return None
            return [str(n) for n in z["names"]]

    def __init__(self, conf_path, cache_path, threads=0):
        ScetlibCachedXsecTF = _import_cached_xsec()
        self.conf_path = os.path.abspath(conf_path)
        self.cache_path = os.path.abspath(cache_path)
        # Match the cache's direction set: caches built before the scale
        # directions existed have no scale_* entries, and configuring them in
        # would change the registry and fail the fingerprint check.
        cached_names = self.cache_param_names(self.cache_path)
        want_scales = cached_names is None or any(
            n.startswith("scale_") for n in cached_names
        )
        # fo_resolve_muR likewise has to match: it changes the frozen FO grid,
        # and the docs are explicit that enabling it after the grid was built
        # invalidates the cache. Both travel together with the scale directions.
        self.conf, self._sigma = configure(
            self.conf_path,
            threads,
            diff_scales=want_scales,
            fo_resolve_muR=want_scales,
        )
        sing, nons = self._sigma.sub_pieces()
        # PDF eigenvector coefficients are ordinary AD parameters registered on
        # the CALCULATION (set_pdf_eig_params), not something the cache file can
        # install. They have to be registered BEFORE the rules are loaded: the
        # rule fingerprint hashes the parameter names in order, and the members
        # are interpolated from coefficient slots that would otherwise not exist.
        # Taken from the cache's own names, so a cache built without
        # eigenvectors is untouched.
        n_eig = sum(1 for n in (cached_names or []) if n.startswith("pdf_eig"))
        if n_eig:
            sing.set_pdf_eig_params(n_eig)
            nons.set_pdf_eig_params(n_eig)
        self._fn = ScetlibCachedXsecTF.load(self.cache_path, sing, nons)

        self.param_names = list(self._fn.param_names)
        self.n_params = len(self.param_names)
        self.bins = np.asarray(self._fn._points, dtype=np.float64)
        self.n_bins = self.bins.shape[0]
        # The anchor the rules were compressed around. Accuracy is best in its
        # neighbourhood and degrades gracefully away from it (the upstream guard
        # that used to refuse large excursions was removed), so the fit start
        # should sit here and a postfit re-check is good practice.
        self.anchor = np.asarray(self._fn.anchor, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # bin bookkeeping                                                     #
    # ------------------------------------------------------------------ #

    def fold_for(self, gen_axes, Q_lo, Q_hi, partial=False):
        """:class:`GenFold` summing this cache's bins onto the fit's gen grid.

        Matching is by VALUE, not by assuming the cache was generated in the
        grid's order: a transposed or differently-nested cache would otherwise be
        a silent wrong answer rather than an error.
        """
        return GenFold(self.bins, gen_axes, float(Q_lo), float(Q_hi), partial)

    # ------------------------------------------------------------------ #
    # evaluation                                                          #
    # ------------------------------------------------------------------ #

    @property
    def tf_fn(self):
        """The raw ``ScetlibCachedXsecTF``: a TF-differentiable function of the
        FULL parameter vector, whose gradients come from nested
        ``tf.custom_gradient`` wrappers that call back into C++. Used by the
        param model's ``differentiate=through`` mode."""
        return self._fn

    def values_and_jacobian(self, p):
        """(values (N,), jacobian (N, P)) at parameter vector ``p``."""
        return self._fn.values_and_jacobian(np.asarray(p, dtype=np.float64))

    def hessian(self, p):
        """Exact per-bin Hessians, shape (N, P, P)."""
        return np.asarray(self._fn.hessian(np.asarray(p, dtype=np.float64)))

    def resummed_only(self, p):
        """The RESUMMED (singular) piece alone, without the nonsingular.

        The cached matched total is sing + nons, and the two are stored
        separately -- compressed bin rules for the resummed piece, a frozen grid
        for the fixed-order one. Replaying only the rules gives the resummed
        piece, which is what a ``calculation_piece = sing`` SCETlib production run
        computes. That makes it the apples-to-apples reference for validating our
        configuration and quadrature, with no matching, no DYTurbo and no MiNNLO
        in between.
        """
        res = self._fn._sing.sigma_binned_rule_batch(
            self.bins, np.ascontiguousarray(p, dtype=np.float64)
        )
        return np.asarray(res["value"], dtype=np.float64)

    # ------------------------------------------------------------------ #
    # lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def __deepcopy__(self, memo):
        """Share the backend across a deepcopy instead of copying it.

        rabbit's ``Fitter.__deepcopy__`` copies the param model (saturated model,
        toys). The pybind handles here wrap immutable configuration -- the loaded
        rules and the frozen fixed-order grid -- and a real copy would mean a
        second ``configure`` (LHAPDF reload, beamfunc grid re-read) for no gain.
        """
        memo[id(self)] = self
        return self

    def __repr__(self):
        return (
            f"ScetlibADXsec(bins={self.n_bins}, params={self.n_params}, "
            f"cache={os.path.basename(self.cache_path)})"
        )
