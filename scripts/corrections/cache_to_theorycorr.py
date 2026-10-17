#!/usr/bin/env python3
"""Convert one absolute SCETlib-AD cache into three WRemnants corrections.

The cache supplies the numerator directly.  The response-grid MiNNLO file
supplies the matching central, PDF-member, and alphaS-member denominators.
No legacy correction enters the numerical construction.

Every correction ``vars`` entry is one evaluation of the FULL SCETlib parameter
vector: the base point with nothing changed for ``central`` / ``pdf0`` /
``as_0118``, and the base point with one or two entries displaced for each
variation.  The base is the cache's own anchor -- the point its bin rules were
compressed around -- optionally shifted by ``--npLambda``.

Why shifting it is legitimate, and only for some parameters: a cache is built
at one point, but that point does not mean the same thing for every parameter.
The NP lambdas and the TNP values ride the clad AD tape, so evaluating at a
different value is a real evaluation of the real prediction; alphaS and the PDF
eigenvectors are served by built members and interpolated between them, and the
profile scales and transition points are not on the tape at all.  The
distinction is defined once, in ``scetlib_ad.params.tape_served``, and this
script derives what it accepts from there rather than holding a list.

The shifted values are written into the correction's recorded SCETlib config,
because the correction's job is to state the point the histmaker's templates
were reweighted to -- which is what the fit reads back to anchor on.  A
correction whose recorded anchor is not the point it was computed at is the one
failure mode this whole chain exists to remove: the prediction ratio is 1 at
the fit start either way, so every prefit plot looks perfect and only the
derivatives are wrong.
"""

from __future__ import annotations

import argparse
import configparser
import json
import pathlib
from collections import OrderedDict

import h5py
import hist
import numpy as np

from wremnants.postprocessing.scetlib_ad import params as adp
from wremnants.postprocessing.scetlib_ad.xsec_backend import (
    ScetlibADXsec,
    config_as_dict,
)
from wremnants.production import theory_corrections
from wremnants.utilities import common, theory_utils
from wremnants.utilities.io_tools import input_tools
from wums import boostHistHelpers as hh
from wums import ioutils, output_tools


def cache_pdf(conf_path, key=None) -> tuple[str, dict, float]:
    """The PDF set the cache was built with: (pdfMap key, pdfMap entry, alphaS central).

    Read from the cache's own runcard ([QCD] pdf_set, alphas_mu0) and matched to
    ``theory_utils.pdfMap`` by LHAPDF name, so the denominator histograms and member
    labels follow the cache rather than assuming CT18Z. ``key`` (a pdfMap key)
    overrides the lookup, for a set whose runcard name is not a pdfMap lha_name.
    """
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cfg.read(conf_path)
    lha = cfg.get("QCD", "pdf_set").strip()
    as_cen = cfg.getfloat("QCD", "alphas_mu0")
    if key is None:
        hits = [k for k, v in theory_utils.pdfMap.items() if v.get("lha_name") == lha]
        if len(hits) != 1:
            raise RuntimeError(
                f"cache pdf_set {lha!r} matches {len(hits)} pdfMap entries {hits}; "
                "pass --pdf <pdfMap key>"
            )
        key = hits[0]
    info = theory_utils.pdfMap[key]
    if info.get("lha_name") != lha:
        raise RuntimeError(
            f"--pdf {key} is LHAPDF set {info.get('lha_name')!r}, but the cache was "
            f"built with {lha!r}"
        )
    return key, info, as_cen


def nonpdf_points() -> OrderedDict[str, dict[str, float]]:
    points: OrderedDict[str, dict[str, float]] = OrderedDict()
    points["central"] = {}
    points.update(
        {
            "lambda2_nu0.05": {"np_gnu_lambda2": 0.05},
            "lambda2_nu0.25": {"np_gnu_lambda2": 0.25},
            "lambda20.0": {"np_eff_lambda2": 0.0},
            "lambda21.0": {"np_eff_lambda2": 1.0},
            "delta_lambda2-0.02": {"np_eff_delta_lambda2": -0.02},
            "delta_lambda20.02": {"np_eff_delta_lambda2": 0.02},
            "lambda40.0": {"np_eff_lambda4": 0.0},
            "lambda41.0": {"np_eff_lambda4": 1.0},
            "lambda4_nu-0.5": {"np_gnu_lambda4": -0.5},
            "lambda4_nu0.5": {"np_gnu_lambda4": 0.5},
            "kappaFO0.5-kappaf2.": {"scale_kappa_R": 0.5},
            "kappaFO2.-kappaf0.5": {"scale_kappa_R": 2.0},
            "mufdown": {"scale_kappa_F": 0.5},
            "mufup": {"scale_kappa_F": 2.0},
            "mufdown-kappaFO0.5-kappaf2.": {"scale_kappa_F": 0.5, "scale_kappa_R": 0.5},
            "mufup-kappaFO2.-kappaf0.5": {"scale_kappa_F": 2.0, "scale_kappa_R": 2.0},
            "transition_points0.2_0.35_1.0": {"scale_x2": 0.35},
            "transition_points0.2_0.75_1.0": {"scale_x2": 0.75},
            "transition_points0.3_0.6_0.9": {"scale_x1": 0.3, "scale_x3": 0.9},
        }
    )
    for name in ("gamma_cusp", "gamma_mu_q", "gamma_nu", "h_qqV", "s"):
        points[f"{name}-1."] = {f"tnp_{name}": -1.0}
        points[f"{name}1."] = {f"tnp_{name}": 1.0}
    for name in ("b_qqV", "b_qqbarV", "b_qqS", "b_qqDS", "b_qg"):
        points[f"{name}-0.5"] = {f"tnp_{name}": -0.5}
        points[f"{name}0.5"] = {f"tnp_{name}": 0.5}
    return points


def load_histogram(path: pathlib.Path, name: str):
    with h5py.File(path, "r") as handle:
        sample_keys = [key for key in handle if key != "meta_info"]
        if len(sample_keys) != 1:
            raise RuntimeError(
                f"expected one denominator sample in {path}, found {sample_keys}"
            )
        sample = ioutils.pickle_load_h5py(handle[sample_keys[0]])
        if name not in sample["output"]:
            raise KeyError(
                f"{name} not in {path}; available: {sorted(sample['output'])}"
            )
        value = sample["output"][name]
        histogram = value.get() if hasattr(value, "get") else value
        scale_to_pb = float(sample["dataset"]["xsec"]) / float(sample["weight_sum"])
        return histogram * scale_to_pb


AXIS_MAP = {
    "massVgen": "Q",
    "massVGen": "Q",
    "absYVgen": "absY",
    "absYVGen": "absY",
    "ptVgen": "qT",
    "ptVGen": "qT",
    "chargeVgen": "charge",
    "chargeVGen": "charge",
    "pdfVar": "vars",
    "alphasVar": "vars",
}


def canonical_values(
    source, with_members: bool, *, singleton_q_edges: np.ndarray | None = None
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str] | None]:
    names = [AXIS_MAP.get(axis.name, axis.name) for axis in source.axes]
    expected = ["Q", "absY", "qT", "charge"] + (["vars"] if with_members else [])
    response_grid = ["absY", "qT"] + (["vars"] if with_members else [])
    values = np.asarray(source.values(flow=False), dtype=float)
    if set(names) == set(expected) and len(names) == len(expected):
        values = np.transpose(values, [names.index(name) for name in expected])
        layout = "legacy-four-dimensional"
    elif set(names) == set(response_grid) and len(names) == len(response_grid):
        if singleton_q_edges is None or len(singleton_q_edges) != 2:
            raise RuntimeError(
                "the response-grid denominator omits Q; exactly one cache Q bin "
                "must be supplied to promote it to the correction schema"
            )
        values = np.transpose(values, [names.index(name) for name in response_grid])
        # The extended denominator was deliberately produced directly on the
        # neutral-current response grid.  WRemnants correction files retain the
        # historical singleton Q and charge axes, so restore those dimensions
        # explicitly without changing or rebinnning any denominator values.
        if with_members:
            values = values[None, :, :, None, :]
        else:
            values = values[None, :, :, None]
        layout = "response-grid-two-dimensional"
    else:
        raise RuntimeError(
            f"denominator axes {names} match neither {expected} nor {response_grid}"
        )
    member_labels = None
    if with_members:
        member_labels = [str(value) for value in source.axes[names.index("vars")]]
        if len(member_labels) != values.shape[-1]:
            raise RuntimeError(
                f"denominator member-axis labels {member_labels} do not match "
                f"the values shape {values.shape}"
            )
    if layout == "response-grid-two-dimensional":
        edges = {
            "Q": np.asarray(singleton_q_edges, dtype=float),
            "absY": np.asarray(source.axes[names.index("absY")].edges, dtype=float),
            "qT": np.asarray(source.axes[names.index("qT")].edges, dtype=float),
            # This is the same singleton neutral-current charge convention used
            # by the established CorrZ files.
            "charge": np.asarray([0.0, 1.0], dtype=float),
        }
    else:
        edges = {}
        for canonical in expected:
            if canonical == "vars":
                continue
            original = source.axes[names.index(canonical)]
            edges[canonical] = np.asarray(original.edges, dtype=float)
    if values.shape[0] != 1 or values.shape[3] != 1:
        raise RuntimeError(
            f"expected one Q and charge bin, got denominator shape {values.shape}"
        )
    return values, edges, member_labels


def reorder_members(
    values: np.ndarray, labels: list[str], expected: list[str], family: str
) -> np.ndarray:
    """Select a categorical member axis by label, refusing positional guesses."""
    if len(set(labels)) != len(labels):
        raise RuntimeError(
            f"{family} denominator has duplicate member labels: {labels}"
        )
    missing = [label for label in expected if label not in labels]
    extra = [label for label in labels if label not in expected]
    if missing or extra:
        raise RuntimeError(
            f"{family} denominator member labels differ: "
            f"missing={missing}, extra={extra}"
        )
    return values[..., [labels.index(label) for label in expected]]


def physics_axes(edges: dict[str, np.ndarray]):
    return [
        hist.axis.Variable(edges["Q"], name="Q"),
        hist.axis.Variable(edges["absY"], name="absY"),
        hist.axis.Variable(edges["qT"], name="qT"),
        hist.axis.Variable(
            edges["charge"], name="charge", underflow=False, overflow=False
        ),
    ]


def make_hist(
    values: np.ndarray, edges: dict[str, np.ndarray], labels: list[str] | None
):
    axes = physics_axes(edges)
    if labels is not None:
        axes.append(hist.axis.StrCategory(labels, name="vars", growth=False))
    result = hist.Hist(*axes, storage=hist.storage.Double())
    if result.values(flow=False).shape != values.shape:
        raise RuntimeError(
            f"constructed hist shape {result.values().shape} != data {values.shape}"
        )
    result.view(flow=False)[...] = values
    return result


def load_minnlo_production(path, hist_name, eras, procs=("Zmumu", "Zmumu10to50")):
    """The MiNNLO denominator built the way ``make_theory_corr.py`` builds it.

    Not a convenience: the histmaker multiplies its OWN MiNNLO by
    ``c(g) = sigma_SCETlib(g) / sigma_MiNNLO(g)``, so the denominator has to
    estimate the same population the histmaker reweights. Take it from a
    different sample and the product carries the difference as per-cell MC
    noise -- straight into the templates, since the ratio is built with
    ``smooth=None``.

    So: sum the mu samples, combine the tau decay channel, and fold the signed
    rapidity, exactly as ``make_theory_corr.py:main`` does, then rename the
    histmaker's axis names to the correction's spelling.
    """
    with h5py.File(str(path), "r") as handle:
        present = set(handle.keys())
    usable = [p for p in procs if any(f"{p}_{e}" in present for e in eras)]
    skipped = [p for p in procs if p not in usable]
    if not usable:
        raise RuntimeError(
            f"none of {list(procs)} is in {path} (has {sorted(present)}). "
            "Without a mu sample there is no denominator."
        )
    if skipped:
        # Benign for the Z mass window: Zmumu10to50 is the 10-50 GeV sample and
        # contributes nothing between 60 and 120. Said out loud anyway, because
        # a missing sample that DID overlap would bias the denominator low and
        # the correction correspondingly high, with nothing else to notice it.
        print(
            f"denominator: using {usable}; {skipped} absent from the file "
            "and skipped -- check that none of them populates this Q range.",
            flush=True,
        )
    hists = [
        input_tools.read_mu_hist_combine_tau(
            str(path), proc, hist_name, eras=list(eras), combine_with_tau=True
        )
        for proc in usable
    ]
    minnlo = hh.sumHists(hists)
    if "y" in minnlo.axes.name:
        minnlo = hh.makeAbsHist(minnlo, "y")
    for ax in list(minnlo.axes):
        if ax.name in AXIS_MAP:
            hh.renameAxis(minnlo, ax.name, AXIS_MAP[ax.name])
    return minnlo


def rebin_minnlo_to_cache(minnlo, core: ScetlibADXsec):
    """Put the production denominator on the cache's own (Q, absY, qT) grid.

    The production file is deliberately finer than any correction (201 qT bins
    out to 13 TeV, 5 Q bins); ``make_theory_corr.py`` reaches the correction
    grid by rebinning to common edges with its numerator. Here the numerator's
    grid is the cache's, so rebin onto that directly and keep the axes exactly
    equal -- which lets the identical-axes check downstream stay strict instead
    of trusting a common-edge negotiation to have picked what we meant.
    """
    b = core.bins
    want = {
        "Q": np.unique(b[:, :2]),
        "absY": np.unique(np.abs(b[:, 2:4])),
        "qT": np.unique(b[:, 4:6]),
    }
    for axis, edges in want.items():
        have = np.asarray(minnlo.axes[axis].edges, dtype=float)
        missing = [e for e in edges if not np.any(np.abs(have - e) < 1e-9)]
        if missing:
            raise RuntimeError(
                f"the denominator's {axis} edges cannot express the cache's "
                f"grid: {missing} are not among {have.tolist()}. Rebinning "
                "would have to split a denominator bin, which it cannot do."
            )
    out = minnlo
    for axis, new_edges in want.items():
        out = hh.rebinHist(out, axis, np.asarray(new_edges, dtype=float))
    print(
        "denominator rebinned onto the cache grid: "
        + ", ".join(f"{a}[{len(want[a]) - 1}]" for a in want),
        flush=True,
    )
    return out


def crop_to_cache(values, edges, labels, core: ScetlibADXsec):
    """Cut the denominator down to the span the cache actually covers.

    ``GenFold`` refuses a gen bin the cache only partly tiles, which is right --
    a partial sum is a wrong cross section rather than a missing one. But a
    denominator produced on a wider grid than the cache is an ordinary
    situation, not an error: a correction simply does not extend past the
    prediction behind it, and outside its axes ``set_corr_ratio_flow`` leaves
    the weight at 1, which is what the shipped corrections do above qT 100.

    So crop, and say so. The cache's edges must appear in the denominator's, or
    the two grids are not compatible at all and cropping would paper over it.
    """
    qt_lo, qt_hi = float(core.bins[:, 4].min()), float(core.bins[:, 5].max())
    y_lo, y_hi = float(np.abs(core.bins[:, 2:4]).min()), float(
        np.abs(core.bins[:, 2:4]).max()
    )
    out_values, out_edges, dropped = values, dict(edges), []
    for axis, (lo, hi), pos in (("qT", (qt_lo, qt_hi), 2), ("absY", (y_lo, y_hi), 1)):
        e = np.asarray(out_edges[axis], dtype=float)
        i = int(np.argmin(np.abs(e - lo)))
        j = int(np.argmin(np.abs(e - hi)))
        if abs(e[i] - lo) > 1e-9 or abs(e[j] - hi) > 1e-9:
            raise RuntimeError(
                f"the cache's {axis} span [{lo:g}, {hi:g}] does not land on "
                f"denominator bin edges {e.tolist()}; the two grids are not "
                "compatible."
            )
        if (i, j) == (0, len(e) - 1):
            continue
        dropped.append(f"{axis}: kept [{lo:g}, {hi:g}] of [{e[0]:g}, {e[-1]:g}]")
        out_edges[axis] = e[i : j + 1]  # noqa: E203
        out_values = np.take(out_values, range(i, j), axis=pos)
    if dropped:
        print(
            "denominator cropped to the cache's coverage -- "
            + "; ".join(dropped)
            + "\n  outside it the correction is absent, i.e. the weight stays 1.",
            flush=True,
        )
    return out_values, out_edges, labels


def overridable(core: ScetlibADXsec) -> dict[str, str]:
    """``{runcard key: SCETlib parameter}`` this cache allows --npLambda to move.

    Derived from the cache's OWN registry and from ``scetlib_ad.params``; this
    script holds no list of its own.  Two conditions, and both are load-bearing:

    * ``tape_served`` -- moving the value is a real evaluation rather than an
      interpolation between built members, so the resulting correction is exact;
    * ``corr_anchor_key`` -- the value has a place in the runcard, so the point
      can be RECORDED.  An override with nowhere to be written would produce a
      correction whose recorded anchor is not the point it was computed at, and
      nothing downstream could see that.
    """
    out = {}
    for sl_name in core.param_names:
        rabbit = adp.rabbit_name(sl_name)
        if not adp.tape_served(rabbit):
            continue
        where = adp.corr_anchor_key(rabbit)
        if where is None:
            continue
        section, key, index = where
        if section != "Nonperturbative" or index is not None:
            continue  # TNPs live in a (value, mode) tuple; not yet supported
        out[key] = sl_name
    return out


def parse_np_overrides(entries, core: ScetlibADXsec) -> dict[str, float]:
    """``{runcard key: value}`` from ``--npLambda lambda2=0.5`` arguments."""
    allowed = overridable(core)
    overrides: dict[str, float] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise RuntimeError(f"--npLambda expects key=value, got {entry!r}")
        key, _, raw = entry.partition("=")
        key = key.strip().lower()
        if key not in allowed:
            raise RuntimeError(
                f"--npLambda cannot move {key!r}. This cache allows "
                f"{sorted(allowed)}. Anything else is either not exactly "
                "re-evaluable away from the build point (alphaS and the PDF "
                "eigenvectors are interpolated between built members; the "
                "profile scales and transition points are not on the AD tape) "
                "or has no place in the runcard to be recorded in. See "
                "scetlib_ad.params.tape_served."
            )
        if key in overrides:
            raise RuntimeError(f"--npLambda sets {key!r} more than once")
        try:
            overrides[key] = float(raw)
        except ValueError:
            raise RuntimeError(f"--npLambda value for {key!r} is not a number: {raw!r}")
    return overrides


def evaluation_base(core: ScetlibADXsec, overrides: dict[str, float]) -> np.ndarray:
    """The cache's anchor, with the --npLambda entries replaced."""
    allowed = overridable(core)
    base = core.anchor.copy()
    for key, value in overrides.items():
        base[core.param_names.index(allowed[key])] = value
    return base


def evaluation_config(core: ScetlibADXsec, overrides: dict[str, float]) -> dict:
    """The cache's layered runcard, stating the point we EVALUATE at.

    Everything the cache genuinely freezes is carried verbatim; only the keys
    --npLambda moved are rewritten.  This is what goes into the correction, and
    what the fit reads back to build its anchor.
    """
    cfg = config_as_dict(core.conf)
    if overrides:
        section = dict(cfg.get("Nonperturbative", {}))
        for key, value in overrides.items():
            section[key] = repr(float(value))
        cfg["Nonperturbative"] = section
    return cfg


def check_config_describes_point(core: ScetlibADXsec, cfg: dict, base: np.ndarray):
    """Refuse unless *cfg* resolves back to *base*, parameter by parameter.

    The recorded config and the evaluated vector are two descriptions of one
    point, arrived at independently: the vector is the cache's stored ``anchor``
    array (plus the overrides), the config is its runcard (plus the same
    overrides).  Nothing else checks that a cache and the ``--conf`` it is
    handed actually belong together -- so if they do not, this writer would
    record one tune and evaluate at another, and the fit would anchor on the
    wrong point with its prefit ratio still exactly 1.

    Resolved the same way ``param_model._resolve_anchor`` does, so agreement
    here is agreement with what the fit will actually compute.
    """
    rabbit_names = [adp.rabbit_name(n) for n in core.param_names]
    uncovered = adp.uncovered_params(rabbit_names)
    if uncovered:
        raise RuntimeError(
            f"no central value can be stated for {list(uncovered)}; the "
            "correction could not describe the point it was computed at."
        )
    mismatched = []
    for i, rabbit in enumerate(rabbit_names):
        value = adp.corr_anchor_value(cfg, rabbit)
        if value is None:
            value = adp.structural_central(rabbit)
        if value is None:
            raise RuntimeError(
                f"the runcard records no value for {rabbit!r}, so the "
                "correction cannot state its own anchor."
            )
        if not np.isclose(value, base[i], rtol=0.0, atol=1e-9):
            mismatched.append((rabbit, value, float(base[i])))
    if mismatched:
        detail = "\n".join(
            f"    {name}: runcard {cfg_val!r} vs evaluated {pt_val!r}"
            for name, cfg_val, pt_val in mismatched
        )
        raise RuntimeError(
            "the --conf does not describe this cache: its runcard and the "
            "cache's stored anchor disagree on\n"
            f"{detail}\n"
            "Recording one while evaluating at the other would give a "
            "correction whose anchor is invisibly wrong. Pass the runcard the "
            "cache was built with."
        )


def evaluate(
    core: ScetlibADXsec, fold, base: np.ndarray, overrides: dict[str, float]
) -> np.ndarray:
    point = base.copy()
    for name, value in overrides.items():
        if name not in core.param_names:
            raise RuntimeError(f"cache lacks required parameter {name}")
        point[core.param_names.index(name)] = value
    sigma, _ = core.values_and_jacobian(point)
    # fold returns (qT, absY); corrections use (absY, qT).  A positive-Y-only
    # Z cache contains one half of the folded |Y| cross section.
    folded = fold(np.asarray(sigma, dtype=float)).reshape(fold.gen_shape).T
    y_factor = 2.0 if getattr(fold, "y_convention", "") == "positive-side-only" else 1.0
    return y_factor * folded


def numerator(core, fold, base, edges, labels, points):
    values = np.empty((1, len(edges["absY"]) - 1, len(edges["qT"]) - 1, 1, len(labels)))
    mapping = []
    for index, (label, overrides) in enumerate(zip(labels, points)):
        values[0, :, :, 0, index] = evaluate(core, fold, base, overrides)
        mapping.append({"index": index, "label": label, "parameters": overrides})
        print(f"evaluated {index + 1:3d}/{len(labels)}  {label}", flush=True)
    return make_hist(values, edges, labels), mapping


def check_flow_is_unity(corr):
    """The correction must be the identity wherever it is not defined.

    Outside the grid there is no prediction to correct to, so the weight has to
    be 1 -- a 0 there would delete those events from the nominal instead of
    leaving them at MiNNLO. It matters more since the denominator is cropped to
    the cache's coverage: everything above the cache's top qT now lands in the
    overflow, so that bin carries real events rather than being a formality.

    ``make_corr_from_ratio`` ends in ``set_corr_ratio_flow``, which does exactly
    this for the three physics axes. Checked rather than assumed, because the
    cost of checking is nothing and the failure would be a silent hole in the
    spectrum.
    """
    view = corr.values(flow=True)
    for pos, axis in enumerate(corr.axes[:3]):
        for edge, index in (("underflow", 0), ("overflow", -1)):
            if not getattr(axis.traits, edge):
                continue
            cell = view[(slice(None),) * pos + (index,)]
            if not np.allclose(cell, 1.0):
                bad = np.unique(cell)
                raise RuntimeError(
                    f"{axis.name} {edge} of the correction is not 1 "
                    f"(values {bad[:5]}...). Events outside the grid would be "
                    "rescaled by it instead of left alone."
                )


def correction_file(
    outdir, generator, numerator_hist, denominator_hist, args, metadata
):
    if [axis.name for axis in numerator_hist.axes[:-1]] != [
        axis.name for axis in denominator_hist.axes[:4]
    ]:
        raise RuntimeError("numerator and denominator physics-axis names differ")
    for name in ("Q", "absY", "qT", "charge"):
        if not np.array_equal(
            numerator_hist.axes[name].edges, denominator_hist.axes[name].edges
        ):
            raise RuntimeError(f"refusing non-identical {name} axes")
    corr, denom, num = theory_corrections.make_corr_from_ratio(
        denominator_hist, numerator_hist, smooth=None, normalize=False
    )
    check_flow_is_unity(corr)
    outfile = outdir / f"{generator}_CorrZ.pkl.lz4"
    if outfile.exists():
        raise RuntimeError(f"refusing to overwrite existing correction {outfile}")
    payload = {
        f"{generator}_minnlo_ratio": hist.Hist(
            *corr.axes, storage=hist.storage.Double(), data=corr.values(flow=True)
        ),
        f"{generator}_hist": num,
        "minnlo_ref_hist": denom,
    }
    output_tools.write_lz4_pkl_output(
        str(outfile), "Z", payload, common.base_dir, args, metadata
    )
    return outfile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=pathlib.Path)
    parser.add_argument("--conf", required=True, type=pathlib.Path)
    parser.add_argument("--minnlo", required=True, type=pathlib.Path)
    parser.add_argument(
        "-o",
        "--outpath",
        type=pathlib.Path,
        default=pathlib.Path(common.data_dir) / "TheoryCorrections",
        help="Where the corrections are written. Defaults to the same place "
        "make_theory_corr.py writes to, which is where the histmaker looks.",
    )
    parser.add_argument("--nominal-name", required=True)
    parser.add_argument(
        "--pdfvars-name",
        default=None,
        help="Omit to build the nominal correction only. The PDF and alphaS "
        "sidecars need per-member denominators, which a production MiNNLO file "
        "carries only if it was made with --pdfs.",
    )
    parser.add_argument("--pdfas-name", default=None)
    parser.add_argument(
        "--pdf",
        default=None,
        choices=sorted(theory_utils.pdfMap),
        help="pdfMap key of the cache's PDF set, which names the per-member "
        "denominators (nominal_gen_<name>, nominal_gen_<name>alphaS<range>). "
        "Default: matched from the cache runcard's [QCD] pdf_set.",
    )
    parser.add_argument(
        "--minnloStyle",
        choices=["production", "responsegrid"],
        default="production",
        help="'production' (default) reads the denominator the way "
        "make_theory_corr.py does -- mu samples summed, tau combined, |y| "
        "folded -- then rebins onto the cache grid. Use it whenever the "
        "correction will reweight a histmaker, because the denominator must "
        "estimate the same MiNNLO population the histmaker carries. "
        "'responsegrid' takes a single-sample hist already on the gen grid.",
    )
    parser.add_argument(
        "--minnloh",
        default="nominal_gen",
        help="Denominator histogram name (make_theory_corr.py's --minnloh).",
    )
    parser.add_argument("--eras", nargs="+", default=["13TeVGen"])
    parser.add_argument(
        "--npLambda",
        dest="np_lambda",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Evaluate the cache at a different nonperturbative tune, e.g. "
        "--npLambda lambda2=0.5 lambda2_nu=0.20. Runcard key names. Only "
        "parameters that ride the AD tape may be moved (the value is then "
        "exact, not interpolated) and only ones the runcard can record (so "
        "the correction states the point it was computed at); the set is "
        "derived from scetlib_ad.params, and an unusable key is refused with "
        "the list this cache allows.",
    )
    parser.add_argument("--threads", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(vars(args), indent=2, default=str))
        return 0

    if args.minnloStyle == "production" and (args.pdfvars_name or args.pdfas_name):
        raise SystemExit(
            "--minnloStyle production builds the CENTRAL denominator only; the "
            "PDF and alphaS sidecars need per-member denominators, which are "
            "read by the 'responsegrid' route. Either drop --pdfvars-name / "
            "--pdfas-name, or use --minnloStyle responsegrid."
        )
    args.outpath.mkdir(parents=True, exist_ok=True)
    core = ScetlibADXsec(str(args.conf), str(args.cache), threads=args.threads)
    cache_q_edges = np.unique(np.asarray(core.bins[:, :2], dtype=float))
    if len(cache_q_edges) != 2:
        raise RuntimeError(
            "converter currently requires one cache Q bin, found edges "
            f"{cache_q_edges.tolist()}"
        )
    if args.minnloStyle == "production":
        central_denom_raw = rebin_minnlo_to_cache(
            load_minnlo_production(args.minnlo, args.minnloh, args.eras), core
        )
    else:
        central_denom_raw = load_histogram(args.minnlo, args.minnloh)
    central_values, edges, central_member_labels = canonical_values(
        central_denom_raw, False, singleton_q_edges=cache_q_edges
    )
    if central_member_labels is not None:
        raise RuntimeError("central denominator unexpectedly has a member axis")
    central_values, edges, _ = crop_to_cache(central_values, edges, None, core)
    fold = core.fold_for(
        [("ptVGen", edges["qT"]), ("absYVGen", edges["absY"])],
        float(edges["Q"][0]),
        float(edges["Q"][-1]),
    )
    if fold.n_dropped:
        raise RuntimeError(f"cache fold dropped {fold.n_dropped} bins")

    # The point every variation is displaced FROM, and the config that states
    # it. Built together and cross-checked, because they are two descriptions
    # of one point and a disagreement between them is invisible downstream.
    np_overrides = parse_np_overrides(args.np_lambda, core)
    base = evaluation_base(core, np_overrides)
    cfg = evaluation_config(core, np_overrides)
    check_config_describes_point(core, cfg, base)
    if np_overrides:
        moved = ", ".join(f"{k}={v!r}" for k, v in sorted(np_overrides.items()))
        print(
            f"evaluating at a shifted nonperturbative tune: {moved}\n"
            "  the correction records these values, so the fit anchors here.\n"
            "  comparing this correction against the cache's own runcard will "
            "warn on exactly these keys, which is correct.",
            flush=True,
        )
        allowed = overridable(core)
        varied = {name for point in nonpdf_points().values() for name in point}
        overlap = sorted(k for k in np_overrides if allowed[k] in varied)
        if overlap:
            print(
                f"  NOTE {overlap} also appear(s) in the variation list, whose "
                "points are ABSOLUTE; those variations are no longer centred "
                "where they were.",
                flush=True,
            )

    # The SCETlib runcard goes under the CACHE's basename, because that is the
    # artefact it describes and because lambda_central._select_resummed picks
    # the one basename WITHOUT "sing" in it -- a cache named with "sing" would
    # silently invert that selection.
    cache_basename = args.cache.name
    if "sing" in cache_basename:
        raise RuntimeError(
            f"cache basename {cache_basename!r} contains 'sing', which "
            "lambda_central._select_resummed uses to identify the "
            "fixed-order-singular file. Rename the cache."
        )
    metadata = {
        cache_basename: {
            "config": cfg,
            "cache": str(args.cache.resolve()),
            "conf": str(args.conf.resolve()),
            "np_overrides": np_overrides,
        },
        "conversion": {
            "cache": str(args.cache.resolve()),
            "conf": str(args.conf.resolve()),
            "minnlo": str(args.minnlo.resolve()),
            "rapidity_factor": (
                2.0
                if getattr(fold, "y_convention", "") == "positive-side-only"
                else 1.0
            ),
            "smoothing": None,
            "normalization": False,
        },
    }

    nominal_points = nonpdf_points()
    nominal_labels = list(nominal_points)
    nom_num, nom_map = numerator(
        core, fold, base, edges, nominal_labels, list(nominal_points.values())
    )
    nom_den = make_hist(central_values, edges, None)
    nominal_file = correction_file(
        args.outpath, args.nominal_name, nom_num, nom_den, args, metadata
    )

    pdf_file = as_file = None
    pdf_map = as_map = []
    pdf_source_labels = as_source_labels = None
    if args.pdfvars_name or args.pdfas_name:
        pdf_key, pdf_info, as_cen = cache_pdf(args.conf, args.pdf)
        pdf_name, lha = pdf_info["name"], pdf_info["lha_name"]
        if pdf_info["combine"] != "asymHessian":
            raise RuntimeError(
                f"{pdf_key}: combine = {pdf_info['combine']!r}; the cache stores an "
                "up/down member per eigenvector, so only asymHessian sets map onto it"
            )
        n_eig = sum(name.startswith("pdf_eig") for name in core.param_names)
        if 2 * n_eig + 1 != pdf_info["entries"]:
            raise RuntimeError(
                f"cache has {n_eig} eigenvector pairs, but {pdf_key} ({lha}) has "
                f"{pdf_info['entries']} members; the per-member denominator only "
                "matches a cache built with ALL of the set's eigenvectors"
            )
        print(f"PDF set {lha} (pdfMap {pdf_key!r}): {n_eig} eigenvector pairs")
        pdf_labels = ["pdf0"]
        pdf_points = [{}]
        for eig in range(n_eig):
            pdf_labels.extend([f"pdf{2 * eig + 1}", f"pdf{2 * eig + 2}"])
            pdf_points.extend([{f"pdf_eig{eig}": 1.0}, {f"pdf_eig{eig}": -1.0}])
        pdf_num, pdf_map = numerator(core, fold, base, edges, pdf_labels, pdf_points)
        pdf_den_values, pdf_edges, pdf_source_labels = canonical_values(
            load_histogram(args.minnlo, f"nominal_gen_{pdf_name}"),
            True,
            singleton_q_edges=cache_q_edges,
        )
        pdf_den_values, pdf_edges, pdf_source_labels = crop_to_cache(
            pdf_den_values, pdf_edges, pdf_source_labels, core
        )
        if any(not np.array_equal(edges[key], pdf_edges[key]) for key in edges):
            raise RuntimeError(
                "PDF denominator physics axes differ from the central denominator"
            )
        if pdf_den_values.shape[-1] != len(pdf_labels):
            raise RuntimeError(
                f"PDF denominator has {pdf_den_values.shape[-1]} members, "
                f"expected {len(pdf_labels)}"
            )
        # The MiNNLO histogram carries the WRemnants names, e.g. for CT18Z
        # pdf0CT18Z,pdf1CT18ZDown,pdf1CT18ZUp,... in raw LHAPDF member order.
        # The cache was built from members 1..2*n_eig in that same order.  Validate
        # the labels rather than pretending they are the shorter correction labels.
        expected_pdf_source = theory_utils.pdfNamesAsymHessian(
            pdf_info["entries"], pdf_name
        )
        if pdf_source_labels != expected_pdf_source:
            raise RuntimeError(
                f"PDF denominator member ordering differs from the {lha} cache "
                "member order:\n"
                f"  found:    {pdf_source_labels}\n  expected: {expected_pdf_source}"
            )
        pdf_den = make_hist(pdf_den_values, edges, pdf_labels)
        pdf_file = correction_file(
            args.outpath, args.pdfvars_name, pdf_num, pdf_den, args, metadata
        )

        # Preserve the established WRemnants correction convention: index zero is
        # the central member, followed by the low and high alphaS endpoints.  The
        # step is the set's alphasRange ("002" -> +-0.002), which is also what names
        # the denominator; labels follow the builder's <set>_as_<value*1000> sets,
        # so CT18Z keeps its historical pdfCT18ZNNLO_as_0118/0116/0120.
        as_step = int(pdf_info["alphasRange"]) / 1000.0
        # rounded so e.g. 0.118 - 0.002 is exactly 0.116, not 0.11599999999999999
        as_vals = [as_cen, round(as_cen - as_step, 6), round(as_cen + as_step, 6)]
        as_tags = [f"{round(v * 1000):04d}" for v in as_vals]
        as_labels = [f"pdf{lha}_as_{t}" for t in as_tags]
        as_points = [{}, {"alphas": as_vals[1]}, {"alphas": as_vals[2]}]
        as_num, as_map = numerator(core, fold, base, edges, as_labels, as_points)
        as_den_values, as_edges, as_source_labels = canonical_values(
            load_histogram(
                args.minnlo, f"nominal_gen_{pdf_name}alphaS{pdf_info['alphasRange']}"
            ),
            True,
            singleton_q_edges=cache_q_edges,
        )
        as_den_values, as_edges, as_source_labels = crop_to_cache(
            as_den_values, as_edges, as_source_labels, core
        )
        if any(not np.array_equal(edges[key], as_edges[key]) for key in edges):
            raise RuntimeError(
                "alphaS denominator physics axes differ from the central denominator"
            )
        if as_den_values.shape[-1] != len(as_labels):
            raise RuntimeError(
                f"alphaS denominator has {as_den_values.shape[-1]} members, expected 3"
            )
        # Select by category value even though the expected source is already in
        # this order.  This both proves central is index zero and prevents a future
        # histmaker ordering change from silently attaching the wrong denominator.
        as_den_values = reorder_members(
            as_den_values, as_source_labels, [f"as{t}" for t in as_tags], "alphaS"
        )
        as_den = make_hist(as_den_values, edges, as_labels)
        as_file = correction_file(
            args.outpath, args.pdfas_name, as_num, as_den, args, metadata
        )

    mapping = {
        "cache": str(args.cache.resolve()),
        "conf": str(args.conf.resolve()),
        "minnlo": str(args.minnlo.resolve()),
        "np_overrides": np_overrides,
        "files": {
            k: str(v)
            for k, v in (
                ("nominal", nominal_file),
                ("pdfvars", pdf_file),
                ("pdfas", as_file),
            )
            if v is not None
        },
        "nominal": nom_map,
        "pdfvars": pdf_map,
        "pdfas": as_map,
        "denominator_source_member_labels": {
            "pdfvars": pdf_source_labels,
            "pdfas": as_source_labels,
        },
        "axes": {key: value.tolist() for key, value in edges.items()},
        "fold": fold.describe(),
    }
    (args.outpath / "cache_to_theorycorr_mapping.json").write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(mapping["files"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
