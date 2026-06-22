"""Central NP (lambda) parameters for the SCETlib ParamModel.

The SCETlib correction's Nonperturbative runcard lives in the upstream
``*_Corr<proc>.pkl.lz4`` file under
``file_meta_data.<basename>.config.Nonperturbative``. The histmaker reads that
section when it applies the correction and writes the parsed values into its
output metadata (key ``scetlib_np_lambda_central``); see
:func:`build_lambda_central_meta`. The fit then reads them back from the
metadata that rabbit propagates into the datacard / fitresults -- it never
re-opens the upstream pkl.

Write side (histmaker):
    build_lambda_central_meta(theory_corr_tags, procs) -> {proc: lambda_central}

Read side (fit / postprocessing):
    read_lambda_central(hdf5_path, proc="Z") -> dict
    read_lambda_central_from_meta(meta, proc="Z") -> dict

A ``lambda_central`` dict has keys ``tag``, ``basename``, ``eff_params`` (for
NP_model_effective / F_eff) and ``gnu_params`` (for NP_model_gammanu).
"""

import json
import os
import pickle

import h5py
import lz4.frame

from wremnants.utilities import common as wrem_common
from wums import ioutils as wums_io

# Metadata key under which the histmaker stores the parsed central runcard.
META_KEY = "scetlib_np_lambda_central"

# Parameter names the ParamModel needs, split by the scetlib C++ struct that
# consumes them. Nonperturbative values are strings; numeric ones get floated,
# model names stay strings.
GNU_NUMERIC = ("lambda2_nu", "lambda4_nu", "lambda_inf_nu")
GNU_STRING = ("np_model_nu",)
EFF_NUMERIC = ("lambda_inf", "lambda2", "lambda4", "lambda6", "delta_lambda2")
EFF_STRING = ("np_model",)


# =============================================================================
# Parsing the Nonperturbative section out of an upstream correction pkl
# (write side -- only the histmaker runs this, with the pkl already in hand).
# =============================================================================


def _find_nonperturbative(corr_dict):
    """Return [(basename, Nonperturbative dict)] for every basename in the pkl.

    A correction pkl usually has several basenames (resummed-singular,
    fixed-order, ...); they share one runcard so the Nonperturbative section is
    the same, but we keep them all and pick below.
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

    Numeric params absent from the runcard default to 0 -- runcards only set the
    keys their np_model uses (e.g. tanh_2 omits ``lambda6``).
    """
    eff_params = {"np_model": npert[EFF_STRING[0]]}
    gnu_params = {"np_model_nu": npert[GNU_STRING[0]]}
    for k in EFF_NUMERIC:
        eff_params[k] = float(npert.get(k, 0.0))
    for k in GNU_NUMERIC:
        gnu_params[k] = float(npert.get(k, 0.0))
    return eff_params, gnu_params


def _select_basename(sections, tag):
    """Pick the basename whose runcard to use when a pkl bundles several.

    Some pkls carry multiple NP variants (e.g. a lattice central + a FranksVals
    variant). Prefer a basename whose name shares a keyword with the tag, then
    the resummed-singular file (it carries the full NP set).
    """
    tag_lower = tag.lower()
    KEYWORDS = ("franksvals", "lattice", "newvars", "lambda6")
    matched_kw = next((k for k in KEYWORDS if k in tag_lower), None)

    def _score(name):
        name_lower = name.lower()
        score = 0
        if matched_kw and matched_kw in name_lower:
            score += 10
        if (
            "nnlo_sing" in name_lower
            or "_sing_" in name_lower
            or name_lower.endswith("sing.pkl")
        ):
            score += 1
        return score

    return sorted(sections, key=lambda item: -_score(item[0]))[0]


def extract_lambda_central(corr_dict, tag, proc):
    """Parse the central lambda parameters out of a loaded correction pkl dict.

    Returns ``{tag, basename, eff_params, gnu_params}``. Raises if the pkl has
    no Nonperturbative section.
    """
    sections = _find_nonperturbative(corr_dict)
    if not sections:
        raise KeyError(
            f"No Nonperturbative section in correction pkl for tag={tag!r}, "
            f"proc={proc!r}."
        )
    basename, npert = _select_basename(sections, tag)
    eff_params, gnu_params = _parse_section(npert)
    return dict(tag=tag, basename=basename, eff_params=eff_params, gnu_params=gnu_params)


def _correction_pkl_path(tag, proc, data_dir=None):
    data_dir = data_dir if data_dir is not None else wrem_common.data_dir
    return os.path.join(data_dir, "TheoryCorrections", f"{tag}_Corr{proc}.pkl.lz4")


def build_lambda_central_meta(theory_corr_tags, procs=("Z", "W"), data_dir=None):
    """Build the ``scetlib_np_lambda_central`` metadata for the histmaker output.

    Opens the central correction pkl (``theory_corr_tags[0]``) for each proc and
    extracts its Nonperturbative runcard. Returns ``{proc: lambda_central}`` for
    the procs whose pkl exists and carries an NP section; procs without one are
    skipped silently (most analyses have no SCETlib NP correction). Returns an
    empty dict if there are no tags.

    This is the ONLY place the upstream pkl is read; the fit reads the result
    back from metadata.
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
            # pkl present but no Nonperturbative section -- not an NP correction.
            continue
    return out


# =============================================================================
# Reading the propagated metadata (read side -- fit / postprocessing).
# =============================================================================


def _iter_meta_levels(meta, max_depth=8):
    """Yield ``meta``, then ``meta['meta_info_input']``, recursively.

    The histmaker writes the key into ``meta_info``; rabbit nests that under
    ``meta_info_input`` in the datacard, and again in the fitresults. Walking the
    chain finds the key regardless of which file we were handed.
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


def read_lambda_central_from_meta(meta, proc="Z", _source="<meta>"):
    """Fetch the central lambda parameters from an already-loaded metadata dict.

    Searches ``meta`` and any nested ``meta_info_input`` for the propagated
    ``scetlib_np_lambda_central`` entry. Raises with a clear message if it is
    absent -- old inputs produced before metadata propagation must be remade
    (resolving the upstream pkl by filename is no longer supported).
    """
    for level in _iter_meta_levels(meta):
        lc_all = level.get(META_KEY)
        if not isinstance(lc_all, dict) or not lc_all:
            continue
        if proc in lc_all:
            return dict(lc_all[proc])
        if len(lc_all) == 1:
            # single proc stored -- use it whatever its label
            return dict(next(iter(lc_all.values())))
        raise KeyError(
            f"{_source}: {META_KEY!r} has no proc {proc!r} (have {sorted(lc_all)})."
        )
    raise KeyError(
        f"{_source}: no {META_KEY!r} in metadata. The SCETlib NP runcard is "
        f"propagated into the histmaker output since this version; remake the "
        f"histmaker output (upstream-pkl resolution by filename was removed)."
    )


def load_lambda_central_file(path):
    """Load a lambda_central override from a JSON or YAML file.

    The file must decode to a dict with ``eff_params`` and ``gnu_params``
    sub-dicts. Format is chosen by extension (``.yaml``/``.yml`` -> YAML, else
    JSON; YAML also accepts JSON).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"lambda_central file missing: {path!r}")
    with open(path) as f:
        text = f.read()
    if path.lower().endswith((".yaml", ".yml")):
        import yaml

        data = yaml.safe_load(text)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"lambda_central file {path!r} is not valid JSON; got {exc}"
            ) from exc
    if (
        not isinstance(data, dict)
        or "eff_params" not in data
        or "gnu_params" not in data
    ):
        raise ValueError(
            f"lambda_central file {path!r} must decode to a dict with "
            f"'eff_params' and 'gnu_params' keys; got {type(data).__name__}."
        )
    return data


def read_lambda_central(hdf5_path, proc="Z"):
    """Read the central lambda parameters referenced by an hdf5.

    Accepts a setupRabbit datacard or a rabbit ``fitresults*.hdf5`` (rabbit
    nests the datacard meta one level down; the lookup handles both).

    Order of preference:
      1. a ``lambda_central=<file>`` token in the stored ``--paramModel`` spec
         (an explicit override the fit command recorded);
      2. the ``scetlib_np_lambda_central`` metadata propagated by the histmaker.

    Returns a dict with ``tag``, ``basename``, ``eff_params``, ``gnu_params``
    and ``source``. Raises with a clear message if neither route resolves.
    """
    with h5py.File(hdf5_path, "r") as f:
        if "meta" not in f:
            raise KeyError(f"{hdf5_path}: no 'meta' group -- wrong file type?")
        meta = wums_io.pickle_load_h5py(f["meta"])

    # Preference 1: an explicit lambda_central=<file> override.
    args_meta = (meta.get("meta_info") or {}).get("args") or {}
    for spec in args_meta.get("paramModel") or []:
        for tok in spec:
            if isinstance(tok, str) and tok.startswith("lambda_central="):
                path = tok.split("=", 1)[1]
                lc = load_lambda_central_file(path)
                lc["source"] = f"cli-file:{path}"
                return lc

    # Preference 2: the propagated histmaker metadata.
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
