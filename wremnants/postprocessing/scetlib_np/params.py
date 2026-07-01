"""Shared TF-free vocabulary and helpers for the SCETlib-NP param model.

Single source of truth for the λ parameter names, the np_model selector keys,
the reco/gen axis names, and the few numpy helpers used by both the TF core
(:mod:`sigma_gen`, :mod:`param_model`) and the TF-free tools
(:mod:`sigma_gen_at_lambda`, :mod:`np_function_plots`, :mod:`fitresult_lambdas`,
:mod:`lambda_central`). Import-light (numpy only) so the lightweight tools reach
these without pulling in the TF core.
"""

import numpy as np

# λ parameter names: CS-side γ_ν^NP first, then TMD-effective F_eff.
# lambda6_nu is the CS b⁶ coefficient — only the tanh_6 γ_ν model uses it; tanh_2
# ignores it, so it is inert there and defaults to 0 from the card runcard.
GNU_PARAMS = ("lambda2_nu", "lambda4_nu", "lambda6_nu", "lambda_inf_nu")
EFF_PARAMS = ("lambda2", "lambda4", "lambda6", "delta_lambda2", "lambda_inf")
ALL_PARAMS = GNU_PARAMS + EFF_PARAMS

# Recommended Gaussian prior widths per λ (theorist recommendations), applied by
# SCETlibNPParamModel only when priors are enabled (priors=1); a λ absent here
# floats free (NaN width). Lives in this config home keyed by the names above.
DEFAULT_PRIOR_SIGMAS = {
    "lambda2_nu": 0.10,
    "lambda4_nu": 0.50,
    "lambda6_nu": 0.10,
    "lambda2": 0.50,  # 0.4 ⁺⁰·⁶₋₀.₄ -> symmetric average
    "lambda4": 0.50,  # 0.4 ⁺⁰·⁶₋₀.₄ -> symmetric average
    "delta_lambda2": 0.20,  # 0 ± 0.20 wide default (no theorist value yet)
    "lambda6": 0.1,
}

# np_model selector keys carried alongside the numeric λ in a tune dict.
EFF_MODEL_KEY = "np_model"
GNU_MODEL_KEY = "np_model_nu"

# Fit observable axis names.
RECO_AXES = ("ptll", "yll", "cosThetaStarll_quantile", "phiStarll_quantile")
GEN_AXES = ("ptVGen", "absYVGen")

# Which λ each NP model actually uses. The form factors in btgrid_tf
# (F_eff_tf / gamma_nu_NP_tf) read a fixed subset of the λ per model string; a λ
# outside that subset is inert (e.g. lambda6 under tanh_2). Mirrored here as
# plain data so this module stays TF-free — btgrid_tf is the source of truth.
_EFF_MODEL_ALIASES = {"hyp_tangent": "tanh_2", "square_root": "frac_2"}
_GNU_MODEL_ALIASES = {"hyp_tangent": "tanh_2", "linear": "frac_1"}
# eff models with no lambda_inf damping (plain polynomial form factors).
_EFF_NO_LAMBDA_INF = {"signed_lambda", "identity"}


def active_params(np_model=None, np_model_nu=None):
    """Names of the λ the chosen NP model(s) actually use.

    Pass ``np_model`` for the F_eff (TMD) set, ``np_model_nu`` for the γ_ν (CS)
    set, or both for the union. A λ outside the returned set is inert for that
    model (``lambda6``/``lambda6_nu`` are used only by ``tanh_6``); callers can
    reject such λ instead of silently ignoring them. Source of truth:
    ``btgrid_tf.F_eff_tf`` / ``btgrid_tf.gamma_nu_NP_tf``."""
    out = set()
    if np_model is not None:
        m = _EFF_MODEL_ALIASES.get(np_model, np_model)
        out |= {"lambda2", "lambda4", "delta_lambda2"}
        if m not in _EFF_NO_LAMBDA_INF:
            out.add("lambda_inf")
        if m == "tanh_6":
            out.add("lambda6")
    if np_model_nu is not None:
        m = _GNU_MODEL_ALIASES.get(np_model_nu, np_model_nu)
        out |= {"lambda2_nu", "lambda4_nu", "lambda_inf_nu"}
        if m == "tanh_6":
            out.add("lambda6_nu")
    return out


def parse_lambda_overrides(spec):
    """Parse a ``"name=val,name=val"`` λ-override string into ``{name: float}``.

    Hard-errors (``ValueError``) on a malformed token, an unknown parameter name
    (not in :data:`ALL_PARAMS`), or a non-float value. An empty / ``None`` spec
    yields ``{}``. Model-awareness (whether a *known* λ is used by the chosen
    model) is the caller's job via :func:`active_params`."""
    out = {}
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise ValueError(f"--lambdas: expected 'name=value', got {tok!r}")
        k, v = tok.split("=", 1)
        k = k.strip()
        if k not in ALL_PARAMS:
            raise ValueError(
                f"--lambdas: unknown NP parameter {k!r} "
                f"(known: {', '.join(ALL_PARAMS)})"
            )
        try:
            out[k] = float(v)
        except ValueError:
            raise ValueError(f"--lambdas: {k}={v.strip()!r} is not a float")
    return out


def split_eff_gnu(values):
    """Split a ``{name: value}`` mapping into ``(eff_params, gnu_params)`` dicts
    by membership in EFF_PARAMS / GNU_PARAMS (values floated; names in neither,
    e.g. the model-name keys, are dropped)."""
    eff = {k: float(values[k]) for k in EFF_PARAMS if k in values}
    gnu = {k: float(values[k]) for k in GNU_PARAMS if k in values}
    return eff, gnu


def bin_sum_matrix(src_centers, target_edges, tol=1e-6):
    """(N_target, N_src) 0/1 matrix summing source bins whose centre falls in
    each target bin. Source bins outside every target bin get 0, truncating to
    the target range (e.g. qT > ptVGen_max, |Y| > absY_max)."""
    src = np.asarray(src_centers, dtype=np.float64)
    edges = np.asarray(target_edges, dtype=np.float64)
    W = np.zeros((edges.size - 1, src.size), dtype=np.float64)
    for i in range(edges.size - 1):
        m = (src >= edges[i] - tol) & (src <= edges[i + 1] + tol)
        W[i, m] = 1.0
    return W
