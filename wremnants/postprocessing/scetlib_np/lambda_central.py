"""Central NP (lambda) parameters for the SCETlib ParamModel.

The SCETlib correction's Nonperturbative runcard lives in the upstream
``*_Corr<proc>.pkl.lz4`` under
``file_meta_data.<basename>.config.Nonperturbative``. The histmaker parses that
section when it applies the correction and writes the values to its output
metadata (key ``scetlib_np_lambda_central``); see
:func:`build_lambda_central_meta`. The fit reads them back from the metadata
rabbit propagates into the datacard / fitresults; it never re-opens the pkl.

Write side (histmaker):
    build_lambda_central_meta(theory_corr_tags, procs) -> {proc: lambda_central}

Read side (fit / postprocessing):
    read_lambda_central(hdf5_path, proc="Z") -> dict
    read_lambda_central_from_meta(meta, proc="Z") -> dict

A ``lambda_central`` dict has keys ``tag``, ``basename``, ``eff_params`` (for
NP_model_effective / F_eff) and ``gnu_params`` (for NP_model_gammanu).
"""

import os
import pickle

import h5py
import lz4.frame

from wremnants.postprocessing.scetlib_np.params import (
    EFF_MODEL_KEY,
    EFF_PARAMS,
    GNU_MODEL_KEY,
    GNU_PARAMS,
    active_params,
)
from wremnants.utilities import common as wrem_common
from wums import ioutils as wums_io

# Metadata key under which the histmaker stores the parsed central runcard.
META_KEY = "scetlib_np_lambda_central"


# =============================================================================
# Parse the Nonperturbative section out of an upstream correction pkl
# (write side -- only the histmaker runs this, pkl already in hand).
# =============================================================================


def _find_nonperturbative(corr_dict):
    """Return [(basename, Nonperturbative dict)] for every basename in the pkl.

    A correction pkl carries several basenames (resummed SCETlib file, fixed-order
    singular file, gen hist, ...). Resummed and singular files can have DIFFERENT
    Nonperturbative runcards (e.g. FranksVals), so keep all NP-bearing basenames
    and let :func:`_select_resummed` pick the central one.
    """
    out = []
    meta = corr_dict.get("file_meta_data")
    if not isinstance(meta, dict):
        raise KeyError("Correction pkl has no 'file_meta_data' entry.")
    for basename, file_meta in meta.items():
        if not isinstance(file_meta, dict):
            continue
        cfg = file_meta.get("config")
        if not isinstance(cfg, dict):
            continue
        npert = cfg.get("Nonperturbative")
        if isinstance(npert, dict):
            out.append((basename, npert))
    return out


def _parse_section(npert):
    """Split one Nonperturbative dict into the eff / gnu parameter groups.

    Numeric params absent from the runcard default to 0; a runcard only sets the
    keys its np_model uses (e.g. tanh_2 omits ``lambda6``).
    """
    eff_params = {EFF_MODEL_KEY: npert[EFF_MODEL_KEY]}
    gnu_params = {GNU_MODEL_KEY: npert[GNU_MODEL_KEY]}
    for k in EFF_PARAMS:
        eff_params[k] = float(npert.get(k, 0.0))
    for k in GNU_PARAMS:
        gnu_params[k] = float(npert.get(k, 0.0))
    return eff_params, gnu_params


def _select_resummed(sections):
    """Pick the resummed prediction's runcard from the NP-bearing basenames.

    A scetlib_dyturbo correction is built (``make_theory_corr.py``) from a
    resummed SCETlib file plus a fixed-order *singular* file subtracted in the
    matching. Only the resummed file's runcard is the central NP; the singular
    file's can differ (e.g. FranksVals). ``make_theory_corr.py`` distinguishes
    them by the ``"sing"`` substring in the filename, so keep the basename
    without ``"sing"``.
    """
    resummed = [item for item in sections if "sing" not in item[0]]
    if len(resummed) == 1:
        return resummed[0]
    if not resummed:
        raise KeyError(
            "No resummed (non-'sing') basename carries a Nonperturbative "
            f"section; basenames seen: {[bn for bn, _ in sections]}."
        )
    raise KeyError(
        "Multiple resummed basenames carry a Nonperturbative section "
        f"({[bn for bn, _ in resummed]}); cannot pick the central runcard."
    )


def extract_lambda_central(corr_dict, tag, proc):
    """Parse the central lambda parameters from a loaded correction pkl dict.

    Returns ``{tag, basename, eff_params, gnu_params}``. Raises if the pkl has
    no Nonperturbative section.
    """
    sections = _find_nonperturbative(corr_dict)
    if not sections:
        raise KeyError(
            f"No Nonperturbative section in correction pkl for tag={tag!r}, "
            f"proc={proc!r}."
        )
    basename, npert = _select_resummed(sections)
    eff_params, gnu_params = _parse_section(npert)
    return dict(
        tag=tag, basename=basename, eff_params=eff_params, gnu_params=gnu_params
    )


def _correction_pkl_path(tag, proc, data_dir=None):
    data_dir = data_dir if data_dir is not None else wrem_common.data_dir
    return os.path.join(data_dir, "TheoryCorrections", f"{tag}_Corr{proc}.pkl.lz4")


def build_lambda_central_meta(theory_corr_tags, procs=("Z", "W"), data_dir=None):
    """Build the ``scetlib_np_lambda_central`` metadata for the histmaker output.

    Opens the central correction pkl (``theory_corr_tags[0]``) per proc and
    extracts its Nonperturbative runcard. Returns ``{proc: lambda_central}`` for
    procs whose pkl exists and carries an NP section; procs without one are
    skipped (most analyses have no SCETlib NP correction). Empty dict if no tags.

    The ONLY place the upstream pkl is read; the fit reads the result back from
    metadata.
    """
    if not theory_corr_tags:
        return {}
    tag = theory_corr_tags[0]  # first entry = central; rest are pdfvars/pdfas
    out = {}
    for proc in procs:
        path = _correction_pkl_path(tag, proc, data_dir=data_dir)
        if not os.path.exists(path):
            continue
        try:
            with lz4.frame.open(path, "rb") as f:
                corr_dict = pickle.load(f)
            out[proc] = extract_lambda_central(corr_dict, tag, proc)
        except KeyError:
            # pkl present, no Nonperturbative section -- not an NP correction.
            continue
    return out


# =============================================================================
# Read the propagated metadata (read side -- fit / postprocessing).
# =============================================================================


def _iter_meta_levels(meta, max_depth=8):
    """Yield ``meta``, then ``meta['meta_info_input']``, recursively.

    The histmaker writes the key into ``meta_info``; rabbit nests that under
    ``meta_info_input`` in the datacard, and again in the fitresults. Walking the
    chain finds the key whichever file was handed in.
    """
    cur = meta
    for _ in range(max_depth):
        if not isinstance(cur, dict):
            return
        yield cur
        nxt = cur.get("meta_info_input")
        if not isinstance(nxt, dict) or nxt is cur:
            return
        cur = nxt


def _fill_missing_params(lc):
    """Validate that the card carries every λ its OWN np_model USES; return ``lc``
    unchanged otherwise (the name is historical — it no longer fills).

    A λ the card's np_model does NOT use (e.g. ``lambda6`` / ``lambda6_nu`` under
    tanh_2) is simply absent: the de-hardcoded ``btgrid_tf`` form factors read only
    the λ their branch needs, so no placeholder slot is required (previously such λ
    were filled with 0.0). A λ the np_model DOES use (per
    :func:`params.active_params`) but the metadata lacks means a stale/corrupt card
    that cannot describe its own model → raise rather than silently default it.
    Cards written before an *inert* λ was added (e.g. pre-``lambda6_nu`` tanh_2
    cards) therefore still load — the λ is neither needed nor filled."""
    eff = dict(lc.get("eff_params", {}))
    gnu = dict(lc.get("gnu_params", {}))
    needed = active_params(
        np_model=eff.get(EFF_MODEL_KEY), np_model_nu=gnu.get(GNU_MODEL_KEY)
    )
    missing_used = sorted(needed - (set(eff) | set(gnu)))
    if missing_used:
        raise KeyError(
            f"lambda_central metadata is missing λ {missing_used} that its np_model "
            f"({eff.get(EFF_MODEL_KEY)} / {gnu.get(GNU_MODEL_KEY)}) USES — the card "
            f"cannot describe its own model; remake the histmaker output."
        )
    return lc


def read_lambda_central_from_meta(meta, proc="Z", _source="<meta>"):
    """Fetch the central lambda parameters from a loaded metadata dict.

    Searches ``meta`` and any nested ``meta_info_input`` for the propagated
    ``scetlib_np_lambda_central`` entry. Raises if absent: inputs produced before
    metadata propagation must be remade (upstream-pkl resolution by filename is
    no longer supported). Params added after the card was written are filled with
    0.0 (see :func:`_fill_missing_params`).
    """
    for level in _iter_meta_levels(meta):
        lc_all = level.get(META_KEY)
        if not isinstance(lc_all, dict) or not lc_all:
            continue
        if proc in lc_all:
            return _fill_missing_params(lc_all[proc])
        if len(lc_all) == 1:
            # single proc stored -- use it whatever its label
            return _fill_missing_params(next(iter(lc_all.values())))
        raise KeyError(
            f"{_source}: {META_KEY!r} has no proc {proc!r} (have {sorted(lc_all)})."
        )
    raise KeyError(
        f"{_source}: no {META_KEY!r} in metadata. The SCETlib NP runcard is "
        f"propagated into the histmaker output since this version; remake the "
        f"histmaker output (upstream-pkl resolution by filename was removed)."
    )


def read_lambda_central(hdf5_path, proc="Z"):
    """Read the central lambda parameters referenced by an hdf5.

    Accepts a setupRabbit datacard or a rabbit ``fitresults*.hdf5`` (rabbit nests
    the datacard meta one level down; the lookup handles both). Reads the
    ``scetlib_np_lambda_central`` metadata the histmaker propagates.

    Returns ``{tag, basename, eff_params, gnu_params, source}``. Raises if the
    metadata is absent.
    """
    with h5py.File(hdf5_path, "r") as f:
        if "meta" not in f:
            raise KeyError(f"{hdf5_path}: no 'meta' group -- wrong file type?")
        meta = wums_io.pickle_load_h5py(f["meta"])

    lc = read_lambda_central_from_meta(meta, proc=proc, _source=hdf5_path)
    lc["source"] = "histmaker-metadata"
    return lc


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            f"usage: python -m {__name__.replace('.', '/')} <hdf5_path> [proc]",
            file=sys.stderr,
        )
        sys.exit(2)
    out = read_lambda_central(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "Z")
    print(f"tag      : {out['tag']}")
    print(f"basename : {out['basename']}")
    print(f"source   : {out.get('source')}")
    print(f"eff_params: {out['eff_params']}")
    print(f"gnu_params: {out['gnu_params']}")
