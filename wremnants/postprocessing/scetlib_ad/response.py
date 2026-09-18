"""Response matrix: folding the boson-level prediction to reco bins.

The differentiable prediction from SCETlib is a boson-level cross section on a
(Q, Y, qT) grid. Turning it into expected reco yields needs the detector, and
that comes from the MC through a response matrix. Two histograms out of the
unfolding histmaker:

* ``R_raw(b, g)`` -- MC weight of events *generated* in gen bin ``g`` and
  *reconstructed* in reco bin ``b``. Its reco axes carry everything the boson
  calculation does not: the decay angles, smearing, efficiency, acceptance.
* ``N_gen(g)``    -- MC weight of ALL events generated in gen bin ``g``, with no
  reco selection. The gen-side marginal of the same sample.

The model uses the ratio

    P(b|g) = R_raw(b, g) / N_gen(g)

which is a conditional probability -- "given generated in ``g``, land in ``b``"
-- i.e. pure efficiency times migration, carrying no information about how much
cross section there is. That last part is the point: ``R_raw`` alone has the MC's
own gen spectrum baked into it, so folding a theory prediction through ``R_raw``
would count the gen spectrum twice. Dividing by ``N_gen`` removes it.

    sigma_reco(b) = sum_g P(b|g) . sigma(p; g)

A second consequence of the same division: any gen-level reweighting of the MC
(for instance a theory correction applied at histmaker time) multiplies ``R_raw``
and ``N_gen`` by the same factor and cancels, so ``P`` is unchanged by it. That
holds wherever the reweighting is constant within a gen bin, which is why the gen
binning should resolve whatever correction the MC carries.
"""

import numpy as np

# Name of the auxiliary group the datacard carries the response in. A DATACARD
# convention fixed by whichever setupRabbit run embedded it -- not a property of
# this package -- so it is overridable, and the historical name is the default so
# existing cards keep working.
DEFAULT_RESPONSE_GROUP = "scetlib_np"

# A pathological parameter point can drive the folded cross section -- and so the
# predicted yield -- negative, which gives a NaN Poisson NLL and a dead gradient
# that strands the minimiser. Soft-floor the ratio (softplus, not a hard clamp) so
# such a point is a large-but-finite penalty with a usable gradient.
#   RATIO_FLOOR_SCALE -- softplus transition width, far below any physical
#     response, so healthy ratios pass through to machine precision:
#     scale * softplus(r / scale) == r for r >> scale.
#   RATIO_FLOOR_MIN -- hard positive ground, since softplus underflows to exactly
#     zero for extreme negatives; keeps the yield strictly > 0.
RATIO_FLOOR_SCALE = 1.0e-4
RATIO_FLOOR_MIN = 1.0e-9


def crop_R_to_fit(R, R_reco_axes, fit_reco_axes, tol=1e-9):
    """Crop R's reco bins so its reco shape matches the fit channel's.

    R is usually stored on a superset of the fit's reco binning. The fit's edges
    must appear as a *contiguous sub-range* of R's, but need not start at R's
    first edge: a low-side acceptance cut (``ptll > 5``, an ``--axlim`` dropping
    leading bins) makes the fit an interior slice of an R stored from zero. For
    each reco axis, find the offset where R's edges line up and crop to
    ``[offset, offset + nbins)``.
    """
    if len(R_reco_axes) != len(fit_reco_axes):
        raise ValueError(
            f"Reco axis count mismatch: R has {len(R_reco_axes)}, "
            f"fit has {len(fit_reco_axes)}"
        )
    axis_slices = []
    for (rname, redges), (fname, fedges) in zip(R_reco_axes, fit_reco_axes):
        if rname != fname:
            raise ValueError(f"Reco axis name mismatch: R={rname!r} vs fit={fname!r}")
        redges = np.asarray(redges)
        fedges = np.asarray(fedges)
        fnb = len(fedges)
        if len(redges) < fnb:
            raise ValueError(
                f"Reco axis {rname}: R has {len(redges) - 1} bins, fit needs "
                f"{fnb - 1}. R is missing edges."
            )
        offset = next(
            (
                k
                for k in range(len(redges) - fnb + 1)
                if np.allclose(redges[k : k + fnb], fedges, atol=tol)
            ),
            None,
        )
        if offset is None:
            raise ValueError(
                f"Reco axis {rname}: fit edges are not a contiguous sub-range of "
                f"R's edges. R={list(redges)} vs fit={list(fedges)}"
            )
        axis_slices.append(slice(offset, offset + fnb - 1))
    slices = tuple(axis_slices)
    # keep every gen axis (the trailing axes of R)
    slices += (slice(None),) * (R.ndim - len(fit_reco_axes))
    return R[slices]


def marginalize_R_reco(R, R_reco_axes, fit_axis_names, log_prefix="scetlib_ad"):
    """Sum R over the reco axes the fit channel does not have.

    The datacard embeds R at the full reco binning it was produced with,
    whatever the fit channel's dimensionality. R is a counts response, so the
    response for a lower-dimensional channel (a 1D ptll or 2D ptll-yll fit) is
    exactly the marginal over the dropped axes. Gen axes are untouched. The kept
    axes must appear in the fit's order.
    """
    R_names = [n for n, _ in R_reco_axes]
    missing = [n for n in fit_axis_names if n not in R_names]
    if missing:
        raise ValueError(f"Fit reco axes {missing} not among R's reco axes {R_names}")
    kept = [n for n in R_names if n in fit_axis_names]
    if kept != list(fit_axis_names):
        raise ValueError(
            f"Fit reco-axis order {list(fit_axis_names)} doesn't match R's "
            f"stored order {kept}"
        )
    drop = tuple(i for i, n in enumerate(R_names) if n not in fit_axis_names)
    if drop:
        R = R.sum(axis=drop)
        print(
            f"[{log_prefix}] marginalized R over reco axes "
            f"{[R_names[i] for i in drop]} (fit channel is "
            f"{len(fit_axis_names)}D: {list(fit_axis_names)})",
            flush=True,
        )
    reco_axes = [(n, e) for n, e in R_reco_axes if n in fit_axis_names]
    return R, reco_axes


def R_info_from_auxiliary(indata, group=DEFAULT_RESPONSE_GROUP):
    """Read the response bundle out of the datacard's auxiliary group.

    setupRabbit extracts R -- plus the gen total ``N_gen`` and the reco/gen axis
    names and edges -- once from the unfolding histmaker output and embeds it via
    rabbit's ``add_auxiliary``; rabbit exposes it as ``FitInputData.auxiliary``.
    Reading it only from there means R is always the one consistent with the run
    that produced this datacard, with no fit-time file path to get wrong.

    Returns ``R``, ``N_gen``, and ``reco_axes`` / ``gen_axes`` as ordered
    ``(name, edges)`` lists.
    """
    aux = getattr(indata, "auxiliary", None) or {}
    if group not in aux:
        raise ValueError(
            f"scetlib_ad: the datacard has no {group!r} auxiliary (the reco x gen "
            f"response matrix R). Rebuild it with a setupRabbit that embeds the "
            f"response from a mz_dilepton --unfolding input carrying "
            f"'nominal_prefsr_yieldsUnfolding' and the 'prefsr' gen total, or pass "
            f"response_group= if the card names it differently."
        )
    bundle = aux[group]
    n_gen = bundle.get("N_gen")
    return dict(
        R=np.asarray(bundle["R"], dtype=np.float64),
        N_gen=None if n_gen is None else np.asarray(n_gen, dtype=np.float64),
        reco_axes=[
            (name, np.asarray(bundle[f"edges__{name}"], dtype=np.float64))
            for name in bundle["reco_axes"]
        ],
        gen_axes=[
            (name, np.asarray(bundle[f"edges__{name}"], dtype=np.float64))
            for name in bundle["gen_axes"]
        ],
    )


# --- the card's recorded theory correction -----------------------------------
#
# A histmaker output records what its theory correction was generated at, and
# rabbit propagates that into the datacard. This is the read side. It is not a
# nicety: the model's prediction is a RATIO to the anchor, multiplying templates
# the correction already reweighted, so without these entries the fit does not
# know what its own denominator means -- and the ratio is 1 at the start either
# way, so a wrong anchor cannot be seen downstream.
#
# Two keys, both histmaker/datacard conventions rather than ours:
#   NP_ANCHOR_META_KEY   the older curated NP extract, still written.
#   CORR_CONFIG_META_KEY the whole resummed runcard, which is what the anchor is
#                        actually built from (see params.CORR_ANCHOR_KEYS).
NP_ANCHOR_META_KEY = "scetlib_np_lambda_central"
CORR_CONFIG_META_KEY = "scetlib_corr_config"


def meta_entry(meta, key, proc="Z", max_depth=8):
    """The per-proc value stored under *key*, or ``None``.

    rabbit nests the histmaker's ``meta_info`` under ``meta_info_input``, and
    again in a fitresult, so walk that chain. A single-proc entry is returned
    whatever it is keyed by -- a Z-only histmaker has no reason to spell it
    "Z" -- but with several procs the name must match.
    """
    cur = meta
    for _ in range(max_depth):
        if not isinstance(cur, dict):
            return None
        entry = cur.get(key)
        if isinstance(entry, dict) and entry:
            per_proc = entry.get(proc)
            if per_proc is None and len(entry) == 1:
                per_proc = next(iter(entry.values()))
            if per_proc is not None:
                return per_proc
        nxt = cur.get("meta_info_input")
        if not isinstance(nxt, dict) or nxt is cur:
            return None
        cur = nxt
    return None


def corr_config_from_meta(meta, proc="Z", max_depth=8):
    """The recorded correction runcard as ``{tag, basename, config, ...}``.

    ``None`` when the card records none, which the model treats as a refusal
    rather than a warning: the central values are then unknown, and taking them
    from the cache instead is exactly the silent-wrong-answer trap this exists
    to close.
    """
    entry = meta_entry(meta, CORR_CONFIG_META_KEY, proc=proc, max_depth=max_depth)
    if not isinstance(entry, dict) or not isinstance(entry.get("config"), dict):
        return None
    return entry
