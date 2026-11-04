"""Record the runcard a SCETlib theory correction was generated at.

The histmaker writes the resummed SCETlib runcard of the central --theoryCorr
file verbatim into its output metadata (``CORR_CONFIG_META_KEY``); rabbit carries
it into the datacard, and the param model reads its anchor (the point at which the
templates are already corrected) from there, see
:func:`response.corr_config_from_meta`. Values stay strings, as in the runcard.
"""

import os
import pickle

import lz4.frame

from wremnants.postprocessing.scetlib_ad.response import (  # noqa: F401
    CORR_CONFIG_META_KEY,
)
from wremnants.utilities import common as wrem_common


def _find_configs(corr_dict):
    """[(basename, config dict)] for every input file of the pkl that has one."""
    meta = corr_dict.get("file_meta_data")
    if not isinstance(meta, dict):
        raise KeyError("Correction pkl has no 'file_meta_data' entry.")
    return [
        (basename, file_meta["config"])
        for basename, file_meta in meta.items()
        if isinstance(file_meta, dict) and isinstance(file_meta.get("config"), dict)
    ]


def _select_resummed(configs):
    """The resummed (non-``sing``) runcard with a Nonperturbative section.

    A scetlib_dyturbo correction is built from a resummed SCETlib file and a
    fixed-order singular file (``sing`` in its name, as in make_theory_corr.py);
    only the resummed one is the central prediction.
    """
    resummed = [
        (bn, cfg)
        for bn, cfg in configs
        if "sing" not in bn and isinstance(cfg.get("Nonperturbative"), dict)
    ]
    if not resummed:
        raise KeyError(
            "No resummed basename with a Nonperturbative section; basenames: "
            f"{[bn for bn, _ in configs]}"
        )
    if len(resummed) > 1:
        # e.g. a W correction combining W+ and W-: no single central runcard
        raise KeyError(
            "Several resummed basenames with a Nonperturbative section "
            f"({[bn for bn, _ in resummed]}); cannot pick the central runcard"
        )
    return resummed[0]


def _lower_config(cfg):
    """``{section: {lowercased key: str value}}``, the way configparser reads it."""
    return {
        str(section): {str(k).lower(): str(v) for k, v in body.items()}
        for section, body in cfg.items()
        if isinstance(body, dict)
    }


def extract_corr_config(corr_dict, tag, applied_to_nominal=True):
    """``{tag, basename, config, applied_to_nominal}`` from a loaded correction."""
    basename, cfg = _select_resummed(_find_configs(corr_dict))
    return dict(
        tag=tag,
        basename=basename,
        config=_lower_config(cfg),
        applied_to_nominal=bool(applied_to_nominal),
    )


def build_corr_config_meta(
    theory_corr_tags, procs=("Z", "W"), data_dir=None, applied_to_nominal=True
):
    """``{proc: corr_config}`` of the central correction (``theory_corr_tags[0]``).

    Procs without a correction pkl, or whose pkl has no SCETlib NP runcard, are
    skipped. ``applied_to_nominal`` is False with --theoryCorrAltOnly, where the
    templates are not corrected and there is no anchor.
    """
    if not theory_corr_tags:
        return {}
    tag = theory_corr_tags[0]
    data_dir = data_dir if data_dir is not None else wrem_common.data_dir
    out = {}
    for proc in procs:
        path = os.path.join(data_dir, "TheoryCorrections", f"{tag}_Corr{proc}.pkl.lz4")
        if not os.path.exists(path):
            continue
        with lz4.frame.open(path, "rb") as fh:
            corr_dict = pickle.load(fh)
        try:
            out[proc] = extract_corr_config(
                corr_dict, tag, applied_to_nominal=applied_to_nominal
            )
        except KeyError:
            continue  # not a SCETlib NP correction
    return out
