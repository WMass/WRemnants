"""Read SCETlib NP λ out of a rabbit fitresults HDF5 (table / curves / toys).

OUTPUT-side companion to :mod:`lambda_central` (which reads the INPUT-side
λ_central from the upstream correction pkl). Two responsibilities, kept apart
from plotting:

  * Tabulate the λ (prefit/postfit value, prefit/postfit 1σ constraint, frozen
    flag) and print it (``read_lambdas`` + the CLI).
  * Turn a fitresults into the λ sets / toy ensembles the pure plotter
    :mod:`np_function_plots` consumes (``lambdas_from_fitresult``,
    ``sample_lambda_toys``, ``plot_series_from_fitresult``).

Two fit flavours, SEPARATE readers, both emitting the same
:class:`~np_function_plots.NPLambdas` so the plotter never learns the source:

  * NEW continuous-λ param model: λ stored PHYSICALLY in ``parms``
    (``allowNegativeParam=True``; no sqrt convention), covariance over the
    floating λ in ``cov``. ``lambdas_from_fitresult`` / ``sample_lambda_toys``.
  * OLD template-based fit: discrete NP nuisances (``scetlibNPgamma*`` …) whose
    pulls map to physical λ via a template/piecewise map.
    ``lambdas_from_template_fit`` (param-map driven).

Units: the new-model λ are already physical (see ``param_model`` /
``allowNegativeParam``); no conversion. The ``np_model`` / ``np_model_nu`` strings
the curves need come from :func:`lambda_central.read_lambda_central` (read off the
fitresults), with a CLI/argument override.

CLI (print the table)::

    python -m wremnants.postprocessing.scetlib_np.fitresult_lambdas <fitresults.hdf5>
"""

import argparse
import math
import re

import numpy as np

from rabbit import io_tools
from wremnants.postprocessing.scetlib_np import lambda_central as _lc
from wremnants.postprocessing.scetlib_np.np_function_plots import NPLambdas, Series
from wremnants.postprocessing.scetlib_np.params import ALL_PARAMS, EFF_PARAMS, GNU_PARAMS
SECTOR = {
    **{p: "gamma_nu (CS)" for p in GNU_PARAMS},
    **{p: "F_eff (TMD)" for p in EFF_PARAMS},
}
# Fallback model strings when lambda_central can't reach the upstream pkl.
DEFAULT_NP_MODEL = "tanh_6"
DEFAULT_NP_MODEL_NU = "tanh_2"


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------


def _frozen_matcher(freeze_exprs):
    """rabbit's freeze semantics: exact name OR anchored re.match. None if no list."""
    if not freeze_exprs:
        return None
    exact = set(freeze_exprs)
    compiled = [re.compile(e) for e in freeze_exprs]
    return lambda name: (name in exact) or any(r.match(name) for r in compiled)


def _parse_param_model_spec(args):
    """(model_name, {token: value}) from the recorded --paramModel arg."""
    pm = args.get("paramModel")
    if not pm:
        return None, {}
    tokens = pm[0] if isinstance(pm[0], (list, tuple)) else pm
    tokens = [t.decode() if isinstance(t, bytes) else str(t) for t in tokens]
    if not tokens:
        return None, {}
    spec = {}
    for tok in tokens[1:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            spec[k] = v
    return tokens[0], spec


def _names(parms_hist):
    return [
        n.decode() if isinstance(n, bytes) else str(n) for n in list(parms_hist.axes[0])
    ]


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------


def read_lambdas(fitresult_path, result=None, params=None):
    """Structured λ readout from a (new-model) fitresults.

    Returns a dict::

        {
          "context": {file, result, model, freeze, spec, ...},
          "params": { name: {prefit, postfit, prefit_sigma, postfit_sigma,
                             frozen, sector, present} , ... }
        }

    ``params`` defaults to all known λ; pass a list to restrict/extend.
    """
    fitresult, meta = io_tools.get_fitresult(fitresult_path, result, meta=True)
    post = fitresult["parms"].get()
    pre = fitresult["parms_prefit"].get()
    names = _names(post)
    idx = {n: i for i, n in enumerate(names)}

    pv, prv = post.values(), pre.values()
    post_var, pre_var = post.variances(), pre.variances()

    fit_args = (
        (meta.get("meta_info", {}) or {}).get("args", {})
        if isinstance(meta, dict)
        else {}
    )
    freeze = fit_args.get("freezeParameters") or []
    is_frozen = _frozen_matcher(freeze)
    model_name, spec = _parse_param_model_spec(fit_args)

    want = list(params) if params else list(ALL_PARAMS)

    out = {}
    for name in want:
        if name not in idx:
            out[name] = dict(present=False, sector=SECTOR.get(name, ""))
            continue
        i = idx[name]
        post_sigma = (
            math.sqrt(post_var[i])
            if post_var is not None and post_var[i] >= 0
            else float("nan")
        )
        pre_sigma = (
            math.sqrt(pre_var[i])
            if pre_var is not None and pre_var[i] >= 0
            else float("nan")
        )
        if is_frozen is not None:
            frozen = bool(is_frozen(name))
        else:
            frozen = (post_var is not None) and (post_var[i] == 0.0)
        out[name] = dict(
            present=True,
            sector=SECTOR.get(name, ""),
            prefit=float(prv[i]),
            postfit=float(pv[i]),
            prefit_sigma=float(pre_sigma),
            postfit_sigma=float(post_sigma),
            frozen=frozen,
        )
    return {
        "context": dict(
            file=fitresult_path,
            result=result or "(default)",
            model=model_name,
            freeze=freeze,
            spec=spec,
        ),
        "params": out,
    }


def format_lambda_table(readout):
    """Render the :func:`read_lambdas` result as the printable table string."""
    ctx, params = readout["context"], readout["params"]

    def fmt(x, nd=5):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "n/a"
        return f"{x:.{nd}g}"

    lines = []
    lines.append(f"File   : {ctx['file']}")
    lines.append(f"Result : {ctx['result']}")
    if ctx.get("model"):
        lines.append(f"Model  : {ctx['model']}")
    spec = ctx.get("spec") or {}
    if spec.get("xparam_default"):
        lines.append(f"  xparam_default (start shift) : {spec['xparam_default']}")
    if spec.get("priors") in ("1", "true", "True", "yes", "on"):
        extra = f" ({spec['prior_sigmas']})" if spec.get("prior_sigmas") else ""
        lines.append(f"  priors : ENABLED{extra}")
    if ctx.get("freeze"):
        lines.append(f"Freeze : {ctx['freeze']}")
    lines.append("Values are PHYSICAL λ (allowNegativeParam=True; no sqrt conversion).")
    lines.append("")

    cols = ("Parameter", "Sector", "Fixed", "Prefit", "Postfit", "Postfit±", "Prefit±")
    w = (16, 14, 6, 12, 12, 12, 10)
    header = "  ".join(
        f"{c:>{wi}}" if i else f"{c:<{wi}}" for i, (c, wi) in enumerate(zip(cols, w))
    )
    lines.append(header)
    lines.append("-" * len(header))
    for name, d in params.items():
        sector = d.get("sector", "")
        if not d.get("present"):
            lines.append(
                "  ".join(
                    [
                        f"{name:<{w[0]}}",
                        f"{sector:>{w[1]}}",
                        f"{'--':>{w[2]}}",
                        f"{'absent':>{w[3]}}",
                        f"{'absent':>{w[4]}}",
                        f"{'--':>{w[5]}}",
                        f"{'--':>{w[6]}}",
                    ]
                )
            )
            continue
        lines.append(
            "  ".join(
                [
                    f"{name:<{w[0]}}",
                    f"{sector:>{w[1]}}",
                    f"{('YES' if d['frozen'] else 'no'):>{w[2]}}",
                    f"{fmt(d['prefit']):>{w[3]}}",
                    f"{fmt(d['postfit']):>{w[4]}}",
                    f"{fmt(d['postfit_sigma']):>{w[5]}}",
                    f"{fmt(d['prefit_sigma']):>{w[6]}}",
                ]
            )
        )
    lines.append("")
    lines.append(
        "Postfit± / Prefit± are the 1σ constraints (sqrt of the stored variance)."
    )
    lines.append(
        "Prefit± = 0 means unconstrained (no Gaussian prior); >0 is the prior σ."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# curve / toy feeding (new continuous-λ model)
# ---------------------------------------------------------------------------


def _flat_values(fitresult_path, which="postfit", result=None):
    """{λ name: value} for all present λ, from ``parms`` (postfit) or
    ``parms_prefit`` (prefit)."""
    fitresult = io_tools.get_fitresult(fitresult_path, result)
    h = fitresult["parms" if which == "postfit" else "parms_prefit"].get()
    names = _names(h)
    vals = h.values()
    return {p: float(vals[names.index(p)]) for p in ALL_PARAMS if p in names}


def _resolve_models(fitresult_path, np_model=None, np_model_nu=None):
    """np_model strings from lambda_central (read off the fitresults), falling
    back to defaults if the upstream pkl is unreachable. Explicit arguments win."""
    if np_model and np_model_nu:
        return np_model, np_model_nu
    eff_model, gnu_model = DEFAULT_NP_MODEL, DEFAULT_NP_MODEL_NU
    try:
        lc = _lc.read_lambda_central(fitresult_path)
        eff_model = lc["eff_params"].get("np_model", eff_model)
        gnu_model = lc["gnu_params"].get("np_model_nu", gnu_model)
    except Exception as exc:  # pkl unreachable / non-NP fit: warn, use defaults
        print(
            f"[fitresult_lambdas] could not read np_model from {fitresult_path} "
            f"({exc}); using defaults {eff_model!r}/{gnu_model!r}. "
            f"Pass --np-model/--np-model-nu to override."
        )
    return np_model or eff_model, np_model_nu or gnu_model


def lambdas_from_fitresult(
    fitresult_path, which="postfit", result=None, np_model=None, np_model_nu=None
):
    """:class:`NPLambdas` for the prefit or postfit point of a new-model fit."""
    np_model, np_model_nu = _resolve_models(fitresult_path, np_model, np_model_nu)
    vals = _flat_values(fitresult_path, which=which, result=result)
    return NPLambdas.from_flat(vals, np_model, np_model_nu)


def read_lambda_covariance(fitresult_path, result=None, names=ALL_PARAMS):
    """(floating_names, mean, cov) over the FLOATING λ (variance > 0).

    Frozen λ (variance 0, absent, or pinned) are excluded from the toy band. The
    submatrix keeps the full correlations among the floating λ (strong: e.g.
    lambda2_nu↔lambda2 ≈ −1).

    Returns empties if the fitresults has no ``cov`` (e.g. a ``--noFit`` /
    no-Hessian run); callers then skip the toy band."""
    fitresult = io_tools.get_fitresult(fitresult_path, result)
    if "cov" not in fitresult.keys():
        return [], np.zeros(0), np.zeros((0, 0))
    parms = fitresult["parms"].get()
    pnames = _names(parms)
    pvals = parms.values()
    cov_h = fitresult["cov"].get()
    cov_names = _names(cov_h)
    cov = cov_h.values()
    cidx = {n: i for i, n in enumerate(cov_names)}

    floating, mean, sel = [], [], []
    for p in names:
        if p in cidx and cov[cidx[p], cidx[p]] > 0:
            floating.append(p)
            sel.append(cidx[p])
            mean.append(float(pvals[pnames.index(p)]))
    if not floating:
        return [], np.zeros(0), np.zeros((0, 0))
    sub = cov[np.ix_(sel, sel)]
    return floating, np.asarray(mean), np.asarray(sub)


def sample_lambda_toys(
    fitresult_path, n_toys=500, seed=0, result=None, np_model=None, np_model_nu=None
):
    """List of :class:`NPLambdas` toys sampled from the postfit MVN.

    Floating λ drawn jointly from their postfit covariance; frozen λ held at their
    postfit value. (For real/Asimov data the postfit point is the band centre.)"""
    np_model, np_model_nu = _resolve_models(fitresult_path, np_model, np_model_nu)
    base = _flat_values(fitresult_path, which="postfit", result=result)
    floating, mean, cov = read_lambda_covariance(fitresult_path, result=result)
    if not floating:
        return []
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mean, cov, size=n_toys)
    toys = []
    for d in draws:
        vals = dict(base)
        for k, name in enumerate(floating):
            vals[name] = float(d[k])
        toys.append(NPLambdas.from_flat(vals, np_model, np_model_nu))
    return toys


def plot_series_from_fitresult(
    fitresult_path, result=None, n_toys=500, seed=0, np_model=None, np_model_nu=None
):
    """Build the [prefit dashed, postfit solid + band] series for the plotter."""
    np_model, np_model_nu = _resolve_models(fitresult_path, np_model, np_model_nu)
    pre = lambdas_from_fitresult(
        fitresult_path, "prefit", result, np_model, np_model_nu
    )
    post = lambdas_from_fitresult(
        fitresult_path, "postfit", result, np_model, np_model_nu
    )
    toys = sample_lambda_toys(
        fitresult_path, n_toys, seed, result, np_model, np_model_nu
    )
    return [
        Series(label="prefit (λ_central)", lam=pre, color="C0", linestyle="--", lw=1.8),
        Series(label="postfit", lam=post, color="C3", linestyle="-", lw=2.0, toys=toys),
    ]


# ---------------------------------------------------------------------------
# old template-based fit reader (legacy adapter)
# ---------------------------------------------------------------------------

# AN parameter name (per side) -> param-model name.
_AN_TO_BTGRID = {
    ("CS", "lambda_2"): "lambda2_nu",
    ("CS", "lambda_4"): "lambda4_nu",
    ("CS", "lambda_inf"): "lambda_inf_nu",
    ("TMD", "Lambda_2"): "lambda2",
    ("TMD", "Lambda_4"): "lambda4",
    ("TMD", "Delta_Lambda_2"): "delta_lambda2",
    ("TMD", "Lambda_inf"): "lambda_inf",
    ("TMD", "Lambda_6"): "lambda6",
}


def _template_theta_to_physical(theta, entry, kfactor):
    """Piecewise linearization param(θ) used by the old discrete NP nuisances:
    nominal + max(θ,0)·(Up−nom)·kf − max(−θ,0)·(nom−Down)·kf."""
    nom = entry["nominal"]
    d_up = (entry["Up_template_value"] - nom) * kfactor
    d_dn = (nom - entry["Down_template_value"]) * kfactor
    return nom + max(theta, 0.0) * d_up - max(-theta, 0.0) * d_dn


def _template_base_eff_gnu(param_map):
    """eff/gnu dicts seeded with the param-map's fixed parameters + models."""
    fixed = param_map.get("fixed_parameters", {})
    eff = dict(
        lambda_inf=fixed.get("Lambda_inf_TMD", {}).get("value", 1.0),
        lambda2=0.0,
        lambda4=0.0,
        lambda6=fixed.get("Lambda_6", {}).get("value", 0.016),
        delta_lambda2=0.0,
        np_model="tanh_6",
    )
    gnu = dict(
        lambda_inf_nu=0.0, lambda2_nu=0.0, lambda4_nu=0.0,
        lambda6_nu=0.0007,  # SCETlib NP_model_gammanu b⁶ coeff (Gamma_nu.hpp:102)
        np_model_nu="tanh_6",
    )
    return eff, gnu


def _template_apply(eff, gnu, param_map, theta_by_nuis, kfactors):
    """Fill eff/gnu (copies) with physical λ from per-nuisance θ."""
    eff, gnu = dict(eff), dict(gnu)
    for nuis, entry in param_map["nuisances"].items():
        key = (entry["side"], entry["param_AN"])
        bt = _AN_TO_BTGRID.get(key)
        if bt is None:
            continue
        kf = kfactors.get(nuis, kfactors.get(entry["param_AN"], 1.0))
        val = _template_theta_to_physical(theta_by_nuis.get(nuis, 0.0), entry, kf)
        (gnu if entry["side"] == "CS" else eff)[bt] = val
    return eff, gnu


def lambdas_from_template_fit(
    fitresult_path, np_param_map, result=None, kfactors=None, n_toys=0, seed=0
):
    """Read an OLD template-based fit → (central :class:`NPLambdas`, toys list).

    ``np_param_map`` is the JSON path (or loaded dict) describing each discrete NP
    nuisance's template Up/Down and its physical AN parameter. The nuisance pulls
    map to physical λ via the same piecewise linearization the template histograms
    were built with. ``kfactors`` scales a nuisance's template delta (use when the
    workspace was built with ``--scaleParams``); keyed by rabbit nuisance name or
    AN param name.

    With ``n_toys>0`` the floating nuisances are sampled from their postfit
    covariance in NUISANCE space (linearization applied per toy), so the band
    reflects the nonlinear θ→λ map.
    """
    import json

    if isinstance(np_param_map, str):
        with open(np_param_map) as f:
            np_param_map = json.load(f)
    kfactors = kfactors or {}

    fitresult = io_tools.get_fitresult(fitresult_path, result)
    parms = fitresult["parms"].get()
    pnames = _names(parms)
    pvals = parms.values()
    nuis_list = list(np_param_map["nuisances"].keys())
    theta = {
        nm: (float(pvals[pnames.index(nm)]) if nm in pnames else 0.0)
        for nm in nuis_list
    }

    base_eff, base_gnu = _template_base_eff_gnu(np_param_map)
    c_eff, c_gnu = _template_apply(base_eff, base_gnu, np_param_map, theta, kfactors)
    central = NPLambdas(eff=c_eff, gnu=c_gnu)

    toys = []
    if n_toys > 0:
        cov_h = fitresult["cov"].get()
        cov_names = _names(cov_h)
        cov = cov_h.values()
        cidx = {n: i for i, n in enumerate(cov_names)}
        floating = [
            nm for nm in nuis_list if nm in cidx and cov[cidx[nm], cidx[nm]] > 0
        ]
        if floating:
            sel = [cidx[nm] for nm in floating]
            sub = cov[np.ix_(sel, sel)]
            mean = np.array([theta[nm] for nm in floating])
            rng = np.random.default_rng(seed)
            draws = rng.multivariate_normal(mean, sub, size=n_toys)
            for d in draws:
                th = dict(theta)
                for k, nm in enumerate(floating):
                    th[nm] = float(d[k])
                e, g = _template_apply(base_eff, base_gnu, np_param_map, th, kfactors)
                toys.append(NPLambdas(eff=e, gnu=g))
    return central, toys


# ---------------------------------------------------------------------------
# CLI (print the table)
# ---------------------------------------------------------------------------


def make_parser():
    p = argparse.ArgumentParser(
        description="Print SCETlib NP λ (prefit/postfit/constraints/fixed) from a rabbit fitresults HDF5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("infile", help="path to fitresults*.hdf5")
    p.add_argument(
        "--result",
        default=None,
        help="results group suffix (e.g. 'nominal'); default 'results'.",
    )
    p.add_argument(
        "--params",
        nargs="+",
        default=None,
        help="restrict to these parameter names (default: all known λ).",
    )
    return p


def main(argv=None):
    args = make_parser().parse_args(argv)
    readout = read_lambdas(args.infile, result=args.result, params=args.params)
    print()
    print(format_lambda_table(readout))
    print()


if __name__ == "__main__":
    main()
