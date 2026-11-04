"""Response matrix: folding the boson-level prediction to reco bins.

The SCETlib prediction is a boson-level cross section on a gen grid; the
detector comes from the MC through two histograms of the unfolding histmaker:

* ``R_raw(b, g)`` -- MC weight generated in gen bin ``g`` and reconstructed in
  reco bin ``b`` (decay angles, smearing, efficiency, acceptance);
* ``N_gen(g)`` -- MC weight of all events generated in ``g``, no reco selection.

The fold uses the conditional probability P(b|g) = R_raw(b, g) / N_gen(g), so
the MC's own gen spectrum is not counted twice:

    sigma_reco(b) = sum_g P(b|g) . sigma(p; g)

A gen-level reweighting of the MC that is constant within a gen bin cancels in P.
"""

import numpy as np

from wremnants.postprocessing.scetlib_ad.response_matrix import AUXILIARY_GROUP

# Datacard auxiliary group holding the response (overridable per card).
DEFAULT_RESPONSE_GROUP = AUXILIARY_GROUP

# Soft floor on the predicted ratio, so a point driving sigma negative gives a
# finite penalty with a usable gradient instead of a NaN NLL:
#   ratio -> max(SCALE * softplus(ratio / SCALE), MIN)
# SCALE is far below any physical response (healthy ratios pass unchanged); MIN
# guards softplus underflow.
RATIO_FLOOR_SCALE = 1.0e-4
RATIO_FLOOR_MIN = 1.0e-9


def crop_R_to_fit(R, R_reco_axes, fit_reco_axes, tol=1e-9):
    """Crop R's reco bins so its reco shape matches the fit channel's.

    For each reco axis the fit's edges must be a contiguous sub-range of R's
    (not necessarily starting at R's first edge, e.g. after ``--axlim``).
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

    R is a counts response, so a lower-dimensional channel's response is the
    exact marginal over the dropped axes. Kept axes must be in the fit's order.
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

    setupRabbit embeds R, the gen total ``N_gen`` and the axis names and edges
    via rabbit's ``add_auxiliary``; rabbit exposes it as
    ``FitInputData.auxiliary``.

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


# Metadata key under which the histmaker records the runcard of its theory
# correction (corr_config.py); rabbit propagates it into the datacard.
CORR_CONFIG_META_KEY = "scetlib_corr_config"


def meta_entry(meta, key, proc="Z", max_depth=8):
    """The per-proc value stored under *key*, or ``None``.

    Walks rabbit's ``meta_info_input`` nesting. A single-proc entry is returned
    whatever its key; with several procs the name must match ``proc``.
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

    ``None`` when the card records none; the model then refuses unless
    ``anchor_source='cache'`` was asked for explicitly.
    """
    entry = meta_entry(meta, CORR_CONFIG_META_KEY, proc=proc, max_depth=max_depth)
    if not isinstance(entry, dict) or not isinstance(entry.get("config"), dict):
        return None
    return entry
