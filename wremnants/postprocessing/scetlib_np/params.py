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

# np_model selector keys carried alongside the numeric λ in a tune dict.
EFF_MODEL_KEY = "np_model"
GNU_MODEL_KEY = "np_model_nu"

# Fit observable axis names.
RECO_AXES = ("ptll", "yll", "cosThetaStarll_quantile", "phiStarll_quantile")
GEN_AXES = ("ptVGen", "absYVGen")

# np_model selector aliases -> canonical model name. The form-factor branches in
# btgrid_tf and the registry below key on the canonical name.
_EFF_MODEL_ALIASES = {"hyp_tangent": "tanh_2", "square_root": "frac_2"}
_GNU_MODEL_ALIASES = {"hyp_tangent": "tanh_2", "linear": "frac_1"}

# ---- Model → λ registry (single source of truth) -----------------------------
# Per model: the λ it uses and their fit defaults (value = neutral start fallback,
# sigma = default prior width, None = free). Must stay in sync with the btgrid_tf
# form branches, which read exactly these λ by name.
EFF_MODEL_PARAMS = {
    "identity": {
        "lambda2": {"value": 0.0, "sigma": 0.50},
        "lambda4": {"value": 0.0, "sigma": 0.50},
        "delta_lambda2": {"value": 0.0, "sigma": 0.20},
    },
    "signed_lambda": {
        "lambda2": {"value": 0.0, "sigma": 0.50},
        "lambda4": {"value": 0.0, "sigma": 0.50},
        "delta_lambda2": {"value": 0.0, "sigma": 0.20},
    },
    "tanh_2": {
        "lambda2": {"value": 0.0, "sigma": 0.50},
        "lambda4": {"value": 0.0, "sigma": 0.50},
        "delta_lambda2": {"value": 0.0, "sigma": 0.20},
        "lambda_inf": {"value": 0.0, "sigma": None},
    },
    "tanh_4": {
        "lambda2": {"value": 0.0, "sigma": 0.50},
        "lambda4": {"value": 0.0, "sigma": 0.50},
        "delta_lambda2": {"value": 0.0, "sigma": 0.20},
        "lambda_inf": {"value": 0.0, "sigma": None},
    },
    "tanh_6": {
        "lambda2": {"value": 0.0, "sigma": 0.50},
        "lambda4": {"value": 0.0, "sigma": 0.50},
        "delta_lambda2": {"value": 0.0, "sigma": 0.20},
        "lambda_inf": {"value": 0.0, "sigma": None},
        "lambda6": {"value": 0.0, "sigma": 0.10},
    },
    "frac_2": {
        "lambda2": {"value": 0.0, "sigma": 0.50},
        "lambda4": {"value": 0.0, "sigma": 0.50},
        "delta_lambda2": {"value": 0.0, "sigma": 0.20},
        "lambda_inf": {"value": 0.0, "sigma": None},
    },
    "frac_4": {
        "lambda2": {"value": 0.0, "sigma": 0.50},
        "lambda4": {"value": 0.0, "sigma": 0.50},
        "delta_lambda2": {"value": 0.0, "sigma": 0.20},
        "lambda_inf": {"value": 0.0, "sigma": None},
    },
    "exp_2": {
        "lambda2": {"value": 0.0, "sigma": 0.50},
        "lambda4": {"value": 0.0, "sigma": 0.50},
        "delta_lambda2": {"value": 0.0, "sigma": 0.20},
        "lambda_inf": {"value": 0.0, "sigma": None},
    },
    "exp_4": {
        "lambda2": {"value": 0.0, "sigma": 0.50},
        "lambda4": {"value": 0.0, "sigma": 0.50},
        "delta_lambda2": {"value": 0.0, "sigma": 0.20},
        "lambda_inf": {"value": 0.0, "sigma": None},
    },
}
GNU_MODEL_PARAMS = {
    "tanh_1": {
        "lambda2_nu": {"value": 0.0, "sigma": 0.10},
        "lambda4_nu": {"value": 0.0, "sigma": 0.50},
        "lambda_inf_nu": {"value": 0.0, "sigma": None},
    },
    "tanh_2": {
        "lambda2_nu": {"value": 0.0, "sigma": 0.10},
        "lambda4_nu": {"value": 0.0, "sigma": 0.50},
        "lambda_inf_nu": {"value": 0.0, "sigma": None},
    },
    "tanh_6": {
        "lambda2_nu": {"value": 0.0, "sigma": 0.10},
        "lambda4_nu": {"value": 0.0, "sigma": 0.50},
        "lambda_inf_nu": {"value": 0.0, "sigma": None},
        "lambda6_nu": {"value": 0.0, "sigma": 0.10},
    },
    "frac_1": {
        "lambda2_nu": {"value": 0.0, "sigma": 0.10},
        "lambda4_nu": {"value": 0.0, "sigma": 0.50},
        "lambda_inf_nu": {"value": 0.0, "sigma": None},
    },
    "frac_2": {
        "lambda2_nu": {"value": 0.0, "sigma": 0.10},
        "lambda4_nu": {"value": 0.0, "sigma": 0.50},
        "lambda_inf_nu": {"value": 0.0, "sigma": None},
    },
    "exp_1": {
        "lambda2_nu": {"value": 0.0, "sigma": 0.10},
        "lambda4_nu": {"value": 0.0, "sigma": 0.50},
        "lambda_inf_nu": {"value": 0.0, "sigma": None},
    },
    "exp_2": {
        "lambda2_nu": {"value": 0.0, "sigma": 0.10},
        "lambda4_nu": {"value": 0.0, "sigma": 0.50},
        "lambda_inf_nu": {"value": 0.0, "sigma": None},
    },
}

# Valid model names (canonical + aliases). The single validation set for both
# btgrid_tf (``np_model not in EFF_MODELS``) and the param model. Re-exported by
# btgrid_tf so it need not keep its own copy.
EFF_MODELS = frozenset(EFF_MODEL_PARAMS) | frozenset(_EFF_MODEL_ALIASES)
GNU_MODELS = frozenset(GNU_MODEL_PARAMS) | frozenset(_GNU_MODEL_ALIASES)


def param_defaults(np_model=None, np_model_nu=None):
    """``{name: {"value", "sigma"}}`` for the λ the given model(s) use.

    Union of the F_eff (``np_model``) and γ_ν (``np_model_nu``) registry rows,
    aliases resolved. Raises ``KeyError`` on an unknown model name."""
    out = {}
    if np_model is not None:
        out.update(EFF_MODEL_PARAMS[_EFF_MODEL_ALIASES.get(np_model, np_model)])
    if np_model_nu is not None:
        out.update(GNU_MODEL_PARAMS[_GNU_MODEL_ALIASES.get(np_model_nu, np_model_nu)])
    return out


def active_params(np_model=None, np_model_nu=None):
    """Names of the λ the chosen NP model(s) actually use (registry keys).

    Pass ``np_model`` for the F_eff (TMD) set, ``np_model_nu`` for the γ_ν (CS)
    set, or both for the union. A λ outside the returned set is inert for that
    model and is NOT a fit parameter. Source of truth: :data:`EFF_MODEL_PARAMS` /
    :data:`GNU_MODEL_PARAMS`, audited against the ``btgrid_tf`` form branches."""
    return set(param_defaults(np_model=np_model, np_model_nu=np_model_nu))


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
