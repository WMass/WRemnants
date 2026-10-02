"""Record the NP anchor -- the lambda values the theory correction was made at.

The scetlib_ad ParamModel returns ``sigma_gen(p) / sigma_gen(p_anchor)``, which
multiplies the card's templates. At the fit start ``p == p_anchor``, so the
ratio is exactly 1 and the yields ARE the card's templates -- whatever NP tune
those templates were built at. A cache built at a different tune than the card
therefore looks perfect in every prefit check and is wrong only in the
derivatives, which perturb around the wrong origin. That failure is invisible
without a recorded anchor to compare against, which is why this exists.

Write side (histmaker): :func:`build_lambda_central_meta` and
:func:`build_corr_config_meta` parse the resummed runcard out of the correction
pkl and the histmaker stores both in its output metadata; rabbit propagates them
into the datacard.
Read side (fit): ``response.corr_config_from_meta``, and the anchor the model
builds from it in ``param_model._resolve_anchor``.

Two entries, and the second is the one that matters. ``build_lambda_central_meta``
is the original curated extract: ten NP values under chosen names, enough to
CHECK an anchor. ``build_corr_config_meta`` records the runcard VERBATIM, which
is what it takes to DEFINE one -- alpha_s, the TNPs and the transition points are
anchor-bearing too, and a curated extract has to be widened by hand every time
SCETlib grows a parameter. Both are written; the extract is kept for readers that
already consume it.

Ported to scetlib_ad on 2026-09-08. It previously lived in the scetlib_np
package, which is NOT tracked on this branch, so a histmaker run from here
silently recorded no anchor and the guard was inert. The port deliberately does
NOT depend on a parameter registry: the extract records EVERY numeric key the
Nonperturbative section carries, and the config records every key of every
section, so a new lambda cannot go unrecorded.
"""

import os
import pickle

import lz4.frame

from wremnants.postprocessing.scetlib_ad.response import NP_ANCHOR_META_KEY
from wremnants.utilities import common as wrem_common

META_KEY = NP_ANCHOR_META_KEY

EFF_MODEL_KEY = "np_model"
GNU_MODEL_KEY = "np_model_nu"


def _find_configs(corr_dict):
    """[(basename, whole config dict)] for every basename in the pkl carrying one.

    A correction pkl carries several basenames (resummed SCETlib file,
    fixed-order singular file, gen hist, ...) and only some of them record a
    runcard, so keep all that do and let :func:`_select_resummed` choose. On the
    CT18Z correction there are four basenames and two configs.
    """
    meta = corr_dict.get("file_meta_data")
    if not isinstance(meta, dict):
        raise KeyError("Correction pkl has no 'file_meta_data' entry.")
    return [
        (basename, file_meta["config"])
        for basename, file_meta in meta.items()
        if isinstance(file_meta, dict) and isinstance(file_meta.get("config"), dict)
    ]


def _find_nonperturbative(corr_dict):
    """[(basename, Nonperturbative dict)] for every basename in the pkl.

    The resummed and singular files can hold DIFFERENT runcards, so keep all of
    them and let :func:`_select_resummed` choose.
    """
    return [
        (basename, cfg["Nonperturbative"])
        for basename, cfg in _find_configs(corr_dict)
        if isinstance(cfg.get("Nonperturbative"), dict)
    ]


def _select_resummed(sections):
    """The resummed prediction's runcard, which is the central NP tune.

    A scetlib_dyturbo correction is built from a resummed SCETlib file plus a
    fixed-order *singular* file subtracted in the matching, and only the
    resummed file's runcard is central. ``make_theory_corr.py`` distinguishes
    them by the ``sing`` substring in the filename.
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


def _parse_section(npert):
    """Split one Nonperturbative dict into the eff / gnu groups.

    Every numeric entry is recorded, split on the ``_nu`` suffix: the gamma_nu
    (CS) parameters carry it and the F_eff (TMD) ones do not. That reproduces
    the split the scetlib_np writer produced key-for-key, without needing its
    parameter registry to enumerate them.
    """
    eff_params, gnu_params = {}, {}
    if EFF_MODEL_KEY in npert:
        eff_params[EFF_MODEL_KEY] = npert[EFF_MODEL_KEY]
    if GNU_MODEL_KEY in npert:
        gnu_params[GNU_MODEL_KEY] = npert[GNU_MODEL_KEY]
    for key, value in npert.items():
        if key in (EFF_MODEL_KEY, GNU_MODEL_KEY):
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue  # a non-numeric setting, not a lambda
        (gnu_params if key.endswith("_nu") else eff_params)[key] = val
    return eff_params, gnu_params


def extract_lambda_central(corr_dict, tag, proc):
    """``{tag, basename, eff_params, gnu_params}`` from a loaded correction pkl."""
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


# --- the whole resummed runcard ----------------------------------------------
#
# The lambda_central entry above is a curated extract: ten NP values, chosen
# names. That was enough to CHECK an anchor and is not enough to DEFINE one --
# alpha_s, the TNPs and the transition points are anchor-bearing too, and a
# curated extract has to be widened by hand every time SCETlib grows a
# parameter. So record the resummed runcard VERBATIM as well, section by
# section, and let the reader decide what it needs. Storing the config rather
# than a path also means a later edit to the correction pkl cannot change what
# an existing histmaker output means.
#
# Values stay STRINGS, exactly as SCETlib's config carries them: every value on
# both sides of the comparison is a string, so coercing here would only invent
# a formatting difference for the reader to trip over.
CORR_CONFIG_META_KEY = "scetlib_corr_config"


def _lower_config(cfg):
    """``{section: {lowercased key: str value}}``.

    The cache runcard is mixed-case INI (``muB_min``, ``Ecm``, ``b_qqV``,
    ``muf_follows_muB``) while what a correction pkl carries is already
    lowercased, so ~16 keys would read as "present on one side only" unless both
    sides are folded. ``configparser`` lowercases by default, which is why the
    cache side needs no help -- but a reader that sets ``optionxform = str``
    would silently get the false pairs, so fold here too and make it explicit.
    """
    out = {}
    for section, body in cfg.items():
        if not isinstance(body, dict):
            continue
        out[str(section)] = {str(k).lower(): str(v) for k, v in body.items()}
    return out


def extract_corr_config(corr_dict, tag, proc, applied_to_nominal=True):
    """``{tag, basename, config, applied_to_nominal}`` from a loaded pkl.

    Selected with the SAME rule as :func:`extract_lambda_central` -- the
    non-``sing`` basename that carries a Nonperturbative section -- so the two
    metadata entries can never describe different files.
    """
    sections = _find_nonperturbative(corr_dict)
    if not sections:
        raise KeyError(
            f"No Nonperturbative section in correction pkl for tag={tag!r}, "
            f"proc={proc!r}."
        )
    basename, _ = _select_resummed(sections)
    cfg = dict(_find_configs(corr_dict))[basename]
    return dict(
        tag=tag,
        basename=basename,
        config=_lower_config(cfg),
        applied_to_nominal=bool(applied_to_nominal),
    )


def build_lambda_central_meta(theory_corr_tags, procs=("Z", "W"), data_dir=None):
    """``{proc: lambda_central}`` for the histmaker's output metadata.

    Reads the CENTRAL correction pkl (``theory_corr_tags[0]``; the rest are
    pdfvars / pdfas) for each proc. Procs whose pkl is absent or carries no
    Nonperturbative section are skipped -- most analyses have no SCETlib NP
    correction. This is the only place the upstream pkl is read for the anchor.
    """
    return _build_meta(extract_lambda_central, theory_corr_tags, procs, data_dir)


def build_corr_config_meta(
    theory_corr_tags, procs=("Z", "W"), data_dir=None, applied_to_nominal=True
):
    """``{proc: corr_config}`` for the histmaker's output metadata.

    Same source and same selection as :func:`build_lambda_central_meta`, but the
    whole resummed runcard rather than an extract. *applied_to_nominal* records
    whether the correction was actually applied to the nominal prediction: with
    ``--theoryCorrAltOnly`` it is carried as alternates only, so there is no
    correction anchor for the templates and a fit must not pretend there is one.
    """

    def extract(corr_dict, tag, proc):
        return extract_corr_config(
            corr_dict, tag, proc, applied_to_nominal=applied_to_nominal
        )

    return _build_meta(extract, theory_corr_tags, procs, data_dir)


def _build_meta(extract, theory_corr_tags, procs, data_dir):
    """``{proc: extract(...)}`` over the CENTRAL correction pkl of each proc.

    ``theory_corr_tags[0]`` is the central tag; the rest are pdfvars / pdfas.
    Procs whose pkl is absent or carries no Nonperturbative section are skipped
    -- most analyses have no SCETlib NP correction.
    """
    if not theory_corr_tags:
        return {}
    tag = theory_corr_tags[0]
    out = {}
    for proc in procs:
        path = _correction_pkl_path(tag, proc, data_dir=data_dir)
        if not os.path.exists(path):
            continue
        try:
            with lz4.frame.open(path, "rb") as fh:
                corr_dict = pickle.load(fh)
            out[proc] = extract(corr_dict, tag, proc)
        except KeyError:
            continue  # pkl present, not an NP correction
    return out
