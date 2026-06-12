"""λ_central auto-detect from a fit-input hdf5.

The SCETlib correction's NP runcard is preserved in the upstream
``*_Corr<proc>.pkl.lz4`` file (under ``file_meta_data.<basename>.config.Nonperturbative``),
but it is **not** propagated through the histmaker into the fit-input hdf5. We
read the correction tag from the hdf5 (``meta_info_input.args.theoryCorr``),
resolve to the upstream pkl, and extract the Nonperturbative section.

Single entry point:

    read_lambda_central(hdf5_path, proc="Z") -> dict
"""

import json
import os
import pickle
import sys

import h5py
import lz4.frame

from wremnants.utilities import common as wrem_common
from wums import ioutils as wums_io

# Names of the parameters the ParamModel cares about, split by which scetlib
# C++ struct consumes them. Values in the Nonperturbative section are strings;
# numeric ones get parsed to float, model names stay as strings.
GNU_NUMERIC = ("lambda2_nu", "lambda4_nu", "lambda_inf_nu")
GNU_STRING = ("np_model_nu",)
EFF_NUMERIC = ("lambda_inf", "lambda2", "lambda4", "lambda6", "delta_lambda2")
EFF_STRING = ("np_model",)


def _correction_pkl_path(tag, proc):
    """Resolve a theoryCorr tag to its upstream pkl.lz4 path."""
    return os.path.join(
        wrem_common.data_dir, "TheoryCorrections", f"{tag}_Corr{proc}.pkl.lz4"
    )


def _load_correction_pkl(tag, proc):
    path = _correction_pkl_path(tag, proc)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SCETlib correction pkl not found: {path!r}. "
            f"Cannot extract λ_central for theoryCorr={tag!r}."
        )
    with lz4.frame.open(path, "rb") as f:
        return pickle.load(f)


def _find_nonperturbative(corr_dict):
    """Search the upstream correction dict for ``Nonperturbative`` configs.

    Returns a list of (basename, Nonperturbative dict) tuples. There are
    typically multiple basenames (resummed-singular, fixed-order, etc.) — all
    are expected to share the same Nonperturbative section since SCETlib runs
    them with one runcard.
    """
    out = []
    meta = corr_dict.get("file_meta_data")
    if not isinstance(meta, dict):
        raise KeyError(
            "Correction pkl has no 'file_meta_data' entry — schema mismatch."
        )
    for basename, file_meta in meta.items():
        if not isinstance(file_meta, dict):
            continue
        cfg = file_meta.get("config")
        if not isinstance(cfg, dict):
            continue
        npert = cfg.get("Nonperturbative")
        if isinstance(npert, dict):
            out.append((basename, npert))
    if not out:
        raise KeyError(
            "No Nonperturbative section found in any basename of "
            "file_meta_data — λ_central undefined."
        )
    return out


def _parse_section(npert):
    """Parse one Nonperturbative dict into the two parameter groups.

    Numeric params absent from the runcard default to 0 — SCETlib runcards
    only set the keys relevant to the chosen np_model. e.g. tanh_2 setups
    typically omit ``lambda6`` (the bT⁵ coefficient only used by tanh_6).
    """
    eff_params = {"np_model": npert[EFF_STRING[0]]}
    gnu_params = {"np_model_nu": npert[GNU_STRING[0]]}
    for k in EFF_NUMERIC:
        eff_params[k] = float(npert.get(k, 0.0))
    for k in GNU_NUMERIC:
        gnu_params[k] = float(npert.get(k, 0.0))
    return eff_params, gnu_params


def read_lambda_central_from_meta(meta, proc="Z", _source="<meta>"):
    """Same as :func:`read_lambda_central`, but takes the already-loaded
    metadata dict (e.g. ``indata.metadata``) instead of an hdf5 path.

    Avoids re-opening the input HDF5 when the caller already has the meta
    in hand. ``_source`` is used only in error messages.
    """
    try:
        theory_corr = meta["meta_info_input"]["args"]["theoryCorr"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            f"{_source}: meta_info_input.args.theoryCorr missing — "
            "cannot identify the central SCETlib correction."
        ) from exc

    if not theory_corr:
        raise ValueError(f"{_source}: theoryCorr list is empty.")
    tag = theory_corr[0]  # first entry = central; rest are pdfvars/pdfas

    return _resolve_tag_to_lambda(tag, proc)


def load_lambda_central_file(path):
    """Load a λ_central override from a JSON or YAML file.

    The file must decode to a dict with ``eff_params`` and ``gnu_params``
    sub-dicts (same shape as :func:`read_lambda_central`). Format is chosen
    by extension (``.yaml``/``.yml`` → YAML, else JSON); YAML's loader also
    accepts JSON, so this is forgiving either way.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"λ_central file missing: {path!r}")
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
                f"λ_central file {path!r} is not valid JSON; got {exc}"
            ) from exc
    if (
        not isinstance(data, dict)
        or "eff_params" not in data
        or "gnu_params" not in data
    ):
        raise ValueError(
            f"λ_central file {path!r} must decode to a dict with "
            f"'eff_params' and 'gnu_params' keys; got {type(data).__name__} "
            f"with keys {list(data) if isinstance(data, dict) else '<n/a>'}."
        )
    return data


def read_lambda_central(hdf5_path, proc="Z"):
    """Extract λ_central from the SCETlib correction referenced by the hdf5.

    Accepts either a fit-input datacard (setupRabbit output) OR a rabbit
    ``fitresults*.hdf5``: rabbit propagates the whole datacard meta into the
    fitresults as ``meta_info_input``, so for fitresults the lookup is the
    same after unwrapping one level. Handy to answer "which λ_central was
    this fit's ParamModel initialized with?" from the fit output alone.

    λ_central overrides: the override route is the ``lambda_central=<file>``
    token in the ``--paramModel`` spec — the fit command (including the
    token) is stored in the fitresults ``meta_info.args``, so this function
    recovers the override automatically. A programmatic constructor-arg dict
    is NOT visible in the fit output — for such fits the theoryCorr-tag
    route below silently reports the auto-detect values (check the fit log
    for "λ_central from" lines).

    Returns a dict with:

        tag         : the central theoryCorr tag (first entry of meta args)
        pkl_path    : path to the upstream correction pkl
        basename    : the file_meta basename whose runcard was used
        eff_params  : dict for NP_model_effective (F_eff). Keys:
                      np_model, lambda_inf, lambda2, lambda4, lambda6,
                      delta_lambda2.
        gnu_params  : dict for NP_model_gammanu (γ_ν^NP). Keys:
                      np_model_nu, lambda_inf_nu, lambda2_nu, lambda4_nu.

    Raises with a clear message if any link in the chain is missing.
    """
    with h5py.File(hdf5_path, "r") as f:
        if "meta" not in f:
            raise KeyError(f"{hdf5_path}: no 'meta' group — wrong file type?")
        meta = wums_io.pickle_load_h5py(f["meta"])

    # Preference 1 (fitresults): a lambda_central=<file> token in the stored
    # --paramModel spec — the fit command records the override for free.
    args_meta = (meta.get("meta_info") or {}).get("args") or {}
    for spec in args_meta.get("paramModel") or []:
        for tok in spec:
            if isinstance(tok, str) and tok.startswith("lambda_central="):
                path = tok.split("=", 1)[1]
                lc = load_lambda_central_file(path)
                lc["source"] = f"cli-file:{path}"
                return lc

    # Fallback: resolve the datacard's theoryCorr tag. For fitresults this
    # ASSUMES no env-var/constructor override was active (those are not
    # visible in the output; check the fit log). fitresults nest the
    # datacard meta (which itself carries meta_info_input) one level down —
    # unwrap until the histmaker args are in view.
    while (
        isinstance(meta.get("meta_info_input"), dict)
        and "args" not in meta.get("meta_info_input", {})
        and "meta_info_input" in meta["meta_info_input"]
    ):
        meta = meta["meta_info_input"]
    lc = read_lambda_central_from_meta(meta, proc=proc, _source=hdf5_path)
    lc["source"] = "theoryCorr-tag (assumes no runtime override)"
    return lc


def _resolve_tag_to_lambda(tag, proc):
    """Internal: given a theoryCorr tag, load the pkl and parse λ_central."""

    corr_dict = _load_correction_pkl(tag, proc)
    sections = _find_nonperturbative(corr_dict)

    # Some correction pkls bundle multiple NP variants in one file (e.g. a
    # "lattice" central + a "FranksVals" variant). We need to pick the
    # basename that matches the analysis's NP tag. Heuristic: look for a
    # substring in the basename that also appears in the tag (case-insensitive).
    tag_lower = tag.lower()
    KEYWORDS = ("franksvals", "lattice", "newvars", "lambda6")
    matched_kw = next((k for k in KEYWORDS if k in tag_lower), None)

    def _basename_score(name):
        name_lower = name.lower()
        score = 0
        if matched_kw and matched_kw in name_lower:
            score += 10  # strong preference: matches the analysis variant
        if (
            "nnlo_sing" in name_lower
            or "_sing_" in name_lower
            or name_lower.endswith("sing.pkl")
        ):
            score += 1  # weak preference: resummed-singular carries the full NP set
        return score

    sections_sorted = sorted(sections, key=lambda item: -_basename_score(item[0]))
    basename, npert = sections_sorted[0]
    eff_params, gnu_params = _parse_section(npert)

    return dict(
        tag=tag,
        pkl_path=_correction_pkl_path(tag, proc),
        basename=basename,
        eff_params=eff_params,
        gnu_params=gnu_params,
    )


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
    print(f"pkl_path : {out['pkl_path']}")
    print(f"basename : {out['basename']}")
    print(f"eff_params: {out['eff_params']}")
    print(f"gnu_params: {out['gnu_params']}")
