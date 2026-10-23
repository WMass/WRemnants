"""SCETlib autodiff backend: cached matched sigma_UL with exact derivatives.

Wrapper around ``ScetlibCachedXsecTF`` (``scetlib-cms/py/scetlib_tf.py``) that:

* rebuilds the SCETlib calculation exactly as the cache was prepared with
  (``configure_calculation`` is not the whole configuration: the profile-scale
  / b* block and the NP models go through ``variations.set_vary``);
* loads the one-file cache (compressed bin rules for the resummed piece +
  frozen fixed-order grid for the nonsingular one);
* folds the cache's bins onto the fit's gen grid by value (:class:`GenFold`).

Derivatives come from C++: ``ScetlibCachedXsecTF`` is TF-differentiable through
``tf.custom_gradient`` wrappers that call back into SCETlib (:attr:`tf_fn`);
``values_and_jacobian`` and ``hessian`` expose the same in numpy.

Requires ``source <scetlib-cms>/setup.sh`` inside the container (PYTHONPATH,
LD_LIBRARY_PATH, stack limit); LHAPDF and the beamfunc grids must be reachable.
"""

import configparser
import os
import time

import numpy as np

# Bin-edge matching tolerance (edges are expected to agree exactly).
EDGE_TOL = 1e-9


_IMPORT_HINT = (
    "scetlib_ad needs the SCETlib autodiff build on PYTHONPATH. Run\n"
    "    source <path-to>/scetlib-cms/setup.sh\n"
    "inside the container before rabbit_fit.py (it also sets LD_LIBRARY_PATH "
    "and lifts the stack limit).\n"
)


def _import_scetlib():
    """The calculation and the runcard helpers, with an actionable error.

    Does not import ``scetlib_tf``: that pulls in TensorFlow (memory and a
    thread pool per core), which only evaluating a cache needs, not building
    one. See :func:`_import_cached_xsec`.
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

    Mirrors ``prod/scetlib_run/scetlib-run-qT.py``: runcard layered on
    ``defaults.conf``, calculation, the card's EW parameters and fiducial volumes
    (not part of ``configure_calculation``; skipping them keeps SCETlib's
    defaults), variation 0 via ``set_vary`` (profile scales / b*, NP models),
    then the frozen-node cache on both sub-pieces for the rule replay.

    ``diff_scales`` registers ``scale_kappa_R``, ``scale_x1..x3`` (transition
    points) and ``scale_kappa_F`` (fixed-order only, from the muF polynomial) as
    differentiable parameters. It requires ``muf_follows_muB = no`` and changes
    the parameter registry, so it must match the cache (see
    :class:`ScetlibADXsec`).

    ``fo_resolve_muR`` resolves the fixed-order muR dependence into the frozen
    grid; it must be set before the grid is built and makes the warm-up slower.

    Returns ``(conf, sigma)``.
    """
    sl_config, sl_variations = _import_scetlib()
    src = _scetlib_src()
    conf = configparser.ConfigParser(inline_comment_prefixes="#")
    defaults = os.path.join(src, "prod", "scetlib_run", "defaults.conf")
    if not conf.read(defaults):
        raise FileNotFoundError(
            f"scetlib_ad: cannot read SCETlib defaults {defaults!r}"
        )
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


def _raw_rules_path(cache_path):
    """The flat rule file extracted beside ``cache_path``, if it is still current.

    ``scripts/rabbit/scetlib_ad/extract_cache_rules.py`` writes ``<stem>.rules.bin``
    plus a ``<stem>.rules.json`` sidecar recording the zip member's CRC32 and
    sizes. Returns None (stock load) unless the sidecar matches the cache's
    current rules member, so a replaced ``cache.npz`` never pairs with stale rules.
    """
    import json
    import zipfile

    stem = cache_path[:-4] if cache_path.endswith(".npz") else cache_path
    rules, side = stem + ".rules.bin", stem + ".rules.json"
    if not os.path.exists(side):
        return None
    try:
        with open(side) as f:
            rec = json.load(f)
        with zipfile.ZipFile(cache_path) as z:
            info = z.getinfo(rec["member"])
        now = (info.CRC, info.file_size, info.compress_size)
        was = (rec["crc32"], rec["file_size"], rec["compress_size"])
        if now != was:
            print(
                f"[scetlib_ad] WARNING: {side} was extracted from a different "
                f"{rec['member']} (crc/size {was} vs {now}); ignoring it and "
                "loading the rules from the cache itself",
                flush=True,
            )
            return None
        if os.path.getsize(rules) != rec["payload_bytes"]:
            print(
                f"[scetlib_ad] WARNING: {rules} is truncated; ignoring it", flush=True
            )
            return None
    except (OSError, KeyError, ValueError) as e:
        print(f"[scetlib_ad] WARNING: cannot use {side} ({e}); ignoring it", flush=True)
        return None
    return rules


def _load_with_raw_rules(cls, cache_path, rules_path, sing, nons):
    """``ScetlibCachedXsecTF.load`` with the rule blob read from a flat file.

    The stock ``load`` holds the rule blob three times (numpy array,
    ``.tobytes()``, C++ string); ``DrellYan.load_bin_rules(path)`` reads it with
    an ifstream instead (same parser and fingerprint check). Everything else
    mirrors the stock ``load`` (``sing``/``nons``-supplied branch); keep in step.
    """
    with np.load(cache_path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files if k != "rules"}
    import scetlib_cache

    fmt = int(d["format"])
    if fmt != scetlib_cache.FORMAT:
        raise ValueError(
            f"{cache_path}: cache format {fmt}, expected {scetlib_cache.FORMAT}"
        )
    if "has_muf" in d and bool(int(d["has_muf"])):
        raise ValueError(
            f"{cache_path} declares a muF member pair, which no longer exists -- "
            "kappa_F rides the muF polynomial. Rebuild the cache."
        )
    n_eig = int(d["n_eig"])
    if n_eig and not any(
        str(n).startswith("pdf_eig") for n in sing.gradient_param_names()
    ):
        sing.set_pdf_eig_params(n_eig)
        nons.set_pdf_eig_params(n_eig)
    sing.load_bin_rules(rules_path)
    nons.load_fo_cache_bytes(d["fo"].tobytes())
    fn = cls(
        sing,
        nons,
        bins=d["bins"],
        n_eig=n_eig,
        has_as=bool(int(d["has_as"])) if "has_as" in d else False,
    )
    fn.anchor = np.array(d["anchor"], dtype=np.float64)
    stored = [str(x) for x in d["names"]]
    if stored != fn.param_names:
        raise ValueError(
            "the cached rules were built for a different parameter set:\n"
            f"  cached:  {stored}\n  current: {fn.param_names}"
        )
    return fn


def config_as_dict(conf):
    """A SCETlib runcard as ``{section: {key: str}}``.

    Pass ``core.conf`` (the runcard layered on SCETlib's ``defaults.conf``),
    not the bare runcard: keys left to the defaults (per-flavour ``lambda2_*``,
    ``np_model_tmd``, ...) would otherwise have no counterpart to compare.
    The param model compares a recorded correction config against this, so any
    writer of that record must use this same function. Values stay strings;
    keys are lowercased by ``configparser``.
    """
    return {section: dict(conf[section]) for section in conf.sections()}


def bins_from_gen_axes(gen_axes, Q_lo, Q_hi):
    """(N, 6) SCETlib bins for a (qT, |Y|) gen grid, in the grid's flatten order.

    ``gen_axes`` is ``[(qT_name, edges), (Y_name, edges)]`` -- the order the param
    model flattens in, qT-major. Rapidity edges are used as given: positive side
    only for a folded-|Y| cache, or signed (needed for W). :class:`GenFold`
    accepts either.
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

    Summing bin-integrated cross sections handles, exactly:

    * **order** -- the cache nests (Q, Y, qT) qT-innermost; the gen grid
      flattens qT-major;
    * **rapidity sign** -- a signed-Y cache folds +Y and -Y onto |Y|; a
      positive-side-only cache is accepted (factor 2 short everywhere, which
      cancels in the ratio to the central);
    * **granularity** -- a fine cache folds onto any coarser gen grid whose
      edges are a subset of its own.

    Every gen bin must be tiled exactly by its cache bins, else this raises.
    """

    def __init__(self, bins, gen_axes, Q_lo, Q_hi, partial=False):
        """``partial`` allows a cache that tiles only some gen bins.

        For validation on a subset cache only, never for a fit (a partially
        covered gen bin is a wrong cross section). Compare only where
        ``covered_mask`` is True.
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

        # Bins outside the card's gen range are dropped (the fold stays exact) but
        # are still computed on every call, so warn when that is a large fraction.
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
        # (n_qT, n_absY); True = exactly tiled, safe to compare.
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

        # Group the surviving rows by destination: the fold is one reduceat.
        keep = np.nonzero(rows >= 0)[0]
        self._order = keep[np.argsort(rows[keep], kind="stable")]
        self._starts = np.searchsorted(rows[self._order], np.arange(self.n_gen))
        # Destination gen bin per cache bin, -1 outside the gen range
        # (tf.math.unsorted_segment_sum drops negative ids; see fold_tf).
        self.segment_ids = rows

    def __call__(self, a):
        """Fold a cache-indexed array (first axis = cache bin) onto the gen grid."""
        return np.add.reduceat(np.asarray(a)[self._order], self._starts, axis=0)

    def fold_tf(self, a):
        """The same fold as TensorFlow ops, for the in-graph prediction."""
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
    fo_muf_poly
        Degree of the fixed-order muF polynomial, or ``None`` for SCETlib's
        default. ``0`` switches it off, removing the kappa_F response; the
        safety check is ``SCETlibADParamModel._check_fo_muf_poly``.
    """

    @staticmethod
    def cache_param_names(cache_path):
        """The parameter names a cache was built with, read without loading it.

        Used to configure the calculation the way the cache expects: the rule
        fingerprint hashes the names in order.
        """
        with np.load(cache_path, allow_pickle=False) as z:
            if "names" not in z.files:
                return None
            return [str(n) for n in z["names"]]

    def __init__(self, conf_path, cache_path, threads=0, fo_muf_poly=None):
        ScetlibCachedXsecTF = _import_cached_xsec()
        self.conf_path = os.path.abspath(conf_path)
        self.cache_path = os.path.abspath(cache_path)
        # Match the cache's direction set: configuring scale_* directions a cache
        # lacks would change the registry and fail the fingerprint check.
        cached_names = self.cache_param_names(self.cache_path)
        want_scales = cached_names is None or any(
            n.startswith("scale_") for n in cached_names
        )
        # fo_resolve_muR changes the frozen FO grid, so it must match as well.
        self.conf, self._sigma = configure(
            self.conf_path,
            threads,
            diff_scales=want_scales,
            fo_resolve_muR=want_scales,
        )
        sing, nons = self._sigma.sub_pieces()
        # PDF eigenvector coefficients are AD parameters on the calculation and
        # must be registered BEFORE the rules are loaded (the fingerprint hashes
        # the names in order). Count taken from the cache's own names.
        n_eig = sum(1 for n in (cached_names or []) if n.startswith("pdf_eig"))
        if n_eig:
            sing.set_pdf_eig_params(n_eig)
            nons.set_pdf_eig_params(n_eig)
        # The fixed-order muF polynomial is not stored in the cache: it is rebuilt
        # per bin on first evaluation (expensive, and paid even with kappa_F
        # frozen, since values_and_jacobian has no column subset). Off means
        # kappa_F has no response; at kappa_F = 1 the central value and the other
        # Jacobian columns are the same either way.
        self.fo_muf_poly = None if fo_muf_poly is None else int(fo_muf_poly)
        if self.fo_muf_poly is not None:
            for _piece in (sing, nons):
                _piece.set_fo_muf_poly(self.fo_muf_poly)
        # Announce the load: it is the slow, otherwise silent step of start-up.
        try:
            _sz = os.path.getsize(self.cache_path) / 2**30
            _szs = f"{_sz:.1f} GB"
        except OSError:
            _szs = "size unknown"
        print(
            f"[scetlib_ad] loading cache ({_szs}) {self.cache_path} ...",
            flush=True,
        )
        _t0 = time.time()
        _rules = _raw_rules_path(self.cache_path)
        if _rules is not None:
            print(f"[scetlib_ad]   rules from the extracted {_rules}", flush=True)
            self._fn = _load_with_raw_rules(
                ScetlibCachedXsecTF, self.cache_path, _rules, sing, nons
            )
        else:
            self._fn = ScetlibCachedXsecTF.load(self.cache_path, sing, nons)
        print(
            f"[scetlib_ad] cache loaded in {time.time() - _t0:.1f} s "
            f"({len(self._fn._points)} bins, n_eig={n_eig})",
            flush=True,
        )

        self.param_names = list(self._fn.param_names)
        self.n_params = len(self.param_names)
        self.bins = np.asarray(self._fn._points, dtype=np.float64)
        self.n_bins = self.bins.shape[0]
        # The anchor the rules were compressed around. Accuracy is best near it,
        # so the fit should start here.
        self.anchor = np.asarray(self._fn.anchor, dtype=np.float64)

    def fold_for(self, gen_axes, Q_lo, Q_hi, partial=False):
        """:class:`GenFold` summing this cache's bins onto the fit's gen grid."""
        return GenFold(self.bins, gen_axes, float(Q_lo), float(Q_hi), partial)

    @property
    def tf_fn(self):
        """The raw ``ScetlibCachedXsecTF``: a TF-differentiable function of the
        full parameter vector (nested ``tf.custom_gradient`` wrappers that call
        back into C++). The param model's prediction is built on it."""
        return self._fn

    def values_and_jacobian(self, p):
        """(values (N,), jacobian (N, P)) at parameter vector ``p``."""
        return self._fn.values_and_jacobian(np.asarray(p, dtype=np.float64))

    def hessian(self, p):
        """Exact per-bin Hessians, shape (N, P, P)."""
        return np.asarray(self._fn.hessian(np.asarray(p, dtype=np.float64)))

    def resummed_only(self, p):
        """The RESUMMED (singular) piece alone, without the nonsingular.

        Replays only the rules, i.e. what a ``calculation_piece = sing`` SCETlib
        run computes: a direct reference for validating the configuration.
        """
        res = self._fn._sing.sigma_binned_rule_batch(
            self.bins, np.ascontiguousarray(p, dtype=np.float64)
        )
        return np.asarray(res["value"], dtype=np.float64)

    def __deepcopy__(self, memo):
        """Share the backend across a deepcopy instead of copying it.

        rabbit's ``Fitter.__deepcopy__`` copies the param model; the pybind
        handles wrap immutable state (loaded rules, frozen FO grid), so a real
        copy would only repeat ``configure``.
        """
        memo[id(self)] = self
        return self

    def __repr__(self):
        return (
            f"ScetlibADXsec(bins={self.n_bins}, params={self.n_params}, "
            f"cache={os.path.basename(self.cache_path)})"
        )
