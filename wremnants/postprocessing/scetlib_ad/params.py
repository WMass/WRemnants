"""Parameter registry for the SCETlib autodiff param model.

SCETlib owns the parameter vector: :meth:`DrellYan.gradient_param_names` returns
the differentiable parameters in a fixed order that is baked into the cached bin
rules (any addition or reordering invalidates a cache). This module maps that
vector to rabbit-facing names, and defines the priors, the reparametrisation to
unit nuisances (``alphaS`` included: theta = 1 is Delta(alpha_s) = 0.002, as for
the ``pdfAlphaS`` template), the default frozen set, the impact groups, and how
the anchor is read from the theory correction's recorded runcard.
"""

import ast
import math

# SCETlib gradient name -> rabbit-facing name. Exact-match renames; the TNP and
# PDF-eigenvector families are handled by prefix in :func:`rabbit_name`.
EXPLICIT_NAMES = {
    "alphas": "alphaS",
    "np_eff_lambda_inf": "lambda_inf",
    "np_eff_lambda2": "lambda2",
    "np_eff_lambda4": "lambda4",
    "np_eff_lambda6": "lambda6",
    "np_eff_delta_lambda2": "delta_lambda2",
    "np_gnu_lambda_inf": "lambda_inf_nu",
    "np_gnu_lambda2": "lambda2_nu",
    "np_gnu_lambda4": "lambda4_nu",
    "np_gnu_lambda6": "lambda6_nu",
    "np_gnu_b0_bmax": "b0_over_bmax_nu",
    # Profile scales and matching transition points (set_diff_scales(1)).
    # scale_kappa_F's response is purely fixed-order, carried on the AD tape by
    # the fixed-order muF polynomial.
    "scale_kappa_R": "resumScaleMuR",
    "scale_kappa_F": "resumScaleMuF",
    "scale_x1": "resumTransition1",
    "scale_x2": "resumTransition2",
    "scale_x3": "resumTransition3",
}

# TNPs: ``tnp_gamma_cusp`` -> ``resumTNP_gamma_cusp``, matching the setupRabbit
# group of the TNP templates so grouped impacts stay comparable.
TNP_PREFIX_IN = "tnp_"
TNP_PREFIX_OUT = "resumTNP_"

# PDF eigenvector coefficients, present when the cache carries PDF variations.
PDF_COEFF_FMT = "pdfEig{:d}"
PDF_PREFIX_IN = "pdf_eig"
PDF_PREFIX_OUT = "pdfEig"


def rabbit_name(scetlib_name):
    """SCETlib gradient-parameter name -> the name rabbit reports."""
    if scetlib_name in EXPLICIT_NAMES:
        return EXPLICIT_NAMES[scetlib_name]
    if scetlib_name.startswith(TNP_PREFIX_IN):
        return TNP_PREFIX_OUT + scetlib_name[len(TNP_PREFIX_IN) :]
    if scetlib_name.startswith(PDF_PREFIX_IN):
        return PDF_PREFIX_OUT + scetlib_name[len(PDF_PREFIX_IN) :]
    raise KeyError(
        f"scetlib_ad.params: no rabbit name for SCETlib parameter "
        f"{scetlib_name!r}. Add it to EXPLICIT_NAMES, decide whether it is FREE "
        f"or constrained (FREE_PARAMS), give it a group in "
        f"IMPACT_GROUP_MEMBERS, and say where its anchor comes from "
        f"(CORR_ANCHOR_KEYS / STRUCTURAL_CENTRAL)."
    )


def scetlib_name(rabbit):
    """Inverse of :func:`rabbit_name` (raises on an unknown name)."""
    for k, v in EXPLICIT_NAMES.items():
        if v == rabbit:
            return k
    if rabbit.startswith(TNP_PREFIX_OUT):
        return TNP_PREFIX_IN + rabbit[len(TNP_PREFIX_OUT) :]
    if rabbit.startswith(PDF_PREFIX_OUT):
        return PDF_PREFIX_IN + rabbit[len(PDF_PREFIX_OUT) :]
    raise KeyError(f"scetlib_ad.params: unknown rabbit parameter {rabbit!r}")


# --- Priors ------------------------------------------------------------------
#
# Consulted when the model has ``priors`` on (the default). Parameters in
# FREE_PARAMS float free; every other one gets sigma = 1 in RABBIT's units. For
# a REPARAM'd name that is theta, so the physical 1 sigma is the REPARAM width
# (deliberately a set, not a {name: sigma} table: a physical width written here
# would be multiplied by the map width and silently shrink the prior). TNPs and
# pdfEig* are unit-normalised upstream and need no entry.
#
#   alphaS                       the POI.
#   lambda_inf, lambda_inf_nu, b0_over_bmax_nu
#                                NP shape constants, frozen by default.
#   resumTransition1/3           frozen by default: the analysis varies only the
#                                central transition point; no reference variation
#                                exists for these, hence no REPARAM map either.
#   lambda2, lambda4, delta_lambda2, lambda2_nu, lambda4_nu
#                                NP parameters, constrained by the data (and the
#                                physical-region wall) only; pass e.g.
#                                prior_sigmas=lambda2=1,lambda4_nu=1 to constrain.
FREE_PARAMS = frozenset(
    {
        "alphaS",
        "lambda2",
        "lambda4",
        "delta_lambda2",
        "lambda2_nu",
        "lambda4_nu",
        "lambda_inf",
        "lambda_inf_nu",
        "b0_over_bmax_nu",
        "resumTransition1",
        "resumTransition3",
    }
)


def prior_sigma(rabbit):
    """Gaussian prior sigma in RABBIT's units: None if free, else 1.0."""
    return None if rabbit in FREE_PARAMS else 1.0


# --- PDF confidence-level convention -----------------------------------------
#
# Hessian members are not all 1 sigma (CT18Z's are 90% CL, and some sets carry an
# analysis inflation). As for the templates, the scale is
#
#     scale = pdf_inflation_factor(pdfMap[set], noi) * pdfMap[set]["scale"]
#
# but the model applies it to the COEFFICIENT, c_e = scale * theta, not to the
# response: SCETlib is exactly quadratic in c,
#
#     I(c) = I_0 + c (I_+ - I_-)/2 + c^2 (I_+ + I_- - 2 I_0)/2,
#
# so theta = +-1 evaluates the 68% CL point. The two agree to O(c).
PDF_COEFF_SCALE_NOI_DEFAULT = ("alphaS",)


def pdf_set_key(lha_name, pdf_map=None):
    """``theory_utils.pdfMap`` key whose ``lha_name`` is *lha_name*.

    Raises for a set not in the map rather than silently assuming scale 1.
    """
    if pdf_map is None:
        from wremnants.utilities import theory_utils

        pdf_map = theory_utils.pdfMap
    want = str(lha_name).strip().lower()
    for key, info in pdf_map.items():
        if str(info.get("lha_name", "")).lower() == want:
            return key
    raise KeyError(
        f"params.pdf_set_key: LHAPDF set {lha_name!r} is not in theory_utils."
        f"pdfMap, so its confidence-level convention is unknown. Add it there, "
        f"or pass pdf_coeff_scale=<float> explicitly."
    )


def pdf_coeff_scale(lha_name, noi=None, pdf_map=None):
    """Coefficient scale that makes ``pdfEig{i} = +-1`` the analysis' 1 sigma.

    The per-set CL ``scale`` times the nuisance-of-interest inflation, the same
    product ``postfit_pdf_helper.PostfitPdfHelper`` uses.
    """
    from wremnants.utilities import theory_utils

    pdf_map = theory_utils.pdfMap if pdf_map is None else pdf_map
    info = pdf_map[pdf_set_key(lha_name, pdf_map)]
    noi = list(PDF_COEFF_SCALE_NOI_DEFAULT if noi is None else noi)
    return float(theory_utils.pdf_inflation_factor(info, noi)) * float(
        info.get("scale", 1)
    )


# --- Reparametrisation: unit nuisances --------------------------------------
#
# SCETlib registers physical quantities; rabbit fits a unit nuisance theta and
# the model maps it to the physical value. rabbit's priors are one symmetric
# Gaussian per parameter, which cannot reproduce the asymmetric template
# variations these replace (kappa x2 and /2; x2 0.6 -> 0.35 / 0.75).
#
#   "log"  : value = exp(theta * L)          (positive by construction)
#   "quad" : value = c0 + c1*theta + c2*theta^2
#   "unit" : value = <cache anchor> + width * theta
LN2 = math.log(2.0)
REPARAM = {
    # kappa_R: theta = +-1 -> kappa_R = 2 / 0.5, matching kappaFO x2 and /2.
    "resumScaleMuR": ("log", (LN2,)),
    # kappa_F: same.
    "resumScaleMuF": ("log", (LN2,)),
    # x2: quadratic through theta = -1/0/+1 -> 0.35/0.6/0.75; monotone for
    # |theta| < 2.
    "resumTransition2": ("quad", (0.6, 0.20, -0.05)),
    # resumTransition1/3 are not reparametrised (see FREE_PARAMS).
    #
    # "unit": c0 is filled from the cache anchor, so theta = 0 reproduces it.
    # Widths: alphaS the CT18Z alphasRange (the pdfAlphaS template convention);
    # the lambdas the widths the analysis chose for their template nuisances.
    "alphaS": ("unit", (0.002,)),
    "lambda2": ("unit", (0.50,)),
    "lambda4": ("unit", (0.50,)),
    "delta_lambda2": ("unit", (0.50,)),
    "lambda2_nu": ("unit", (0.10,)),
    # lambda6 / lambda6_nu: the tanh_6 form's third coefficient (only active
    # when np_model / np_model_nu selects tanh_6).
    "lambda6": ("unit", (0.10,)),
    "lambda6_nu": ("unit", (0.10,)),
    "lambda4_nu": ("unit", (0.50,)),
}


def reparam(rabbit):
    """``(kind, coeffs)`` for a reparametrised name, else ``None``."""
    return REPARAM.get(rabbit)


# --- Defaults ----------------------------------------------------------------
#
# Parameters frozen unless the user asks otherwise. Freezing is the only way to
# drop a name: the registry is a property of the cache.
DEFAULT_FROZEN = (
    # NP shape constants: the tanh saturation and the b* convention.
    "lambda_inf",
    "lambda_inf_nu",
    "b0_over_bmax_nu",
    # Inert for the Z (zero Jacobian column, singular covariance), so it must be
    # frozen; _check_no_inert_params refuses it otherwise.
    "resumTNP_b_qqDS",
    # Only the central matching transition point is varied in the analysis.
    "resumTransition1",
    "resumTransition3",
    # The mu_R/mu_F uncertainty is the scale envelope (scale_envelope.py), and a
    # profiled kappa_F is ill-posed: its response is even in theta (ln^2 kappa_F),
    # so a symmetric prior has two mirror minima.
    "resumScaleMuR",
    "resumScaleMuF",
)

# --- Frozen gen-level envelope nuisances -------------------------------------
#
# Not SCETlib parameters: the two POUs the param model appends after the SCETlib
# block for the frozen scale envelope (scale_envelope.py). Unit nuisances, so no
# REPARAM entry and sigma = 1.
SCALE_ENVELOPE_STEM = "resumFOScaleEnv"
SCALE_ENVELOPE_NAMES = (
    SCALE_ENVELOPE_STEM + "SymAvg",
    SCALE_ENVELOPE_STEM + "SymDiff",
)

# Grouped impacts over the model's own parameters. Membership is intersected
# with the parameters actually registered.
IMPACT_GROUP_MEMBERS = {
    "resumNonpert": (
        "lambda2",
        "lambda4",
        "lambda6",
        "delta_lambda2",
        "lambda_inf",
        "lambda2_nu",
        "lambda4_nu",
        "lambda6_nu",
        "lambda_inf_nu",
        "b0_over_bmax_nu",
    ),
    "scetlibNPFeff": (
        "lambda2",
        "lambda4",
        "lambda6",
        "delta_lambda2",
        "lambda_inf",
    ),
    "scetlibNPgammaNu": (
        "lambda2_nu",
        "lambda4_nu",
        "lambda6_nu",
        "lambda_inf_nu",
        "b0_over_bmax_nu",
    ),
    # Named after the card groups they replace; covers both the envelope and
    # directly profiled scales.
    "resumScale": (
        "resumScaleMuR",
        "resumScaleMuF",
        SCALE_ENVELOPE_NAMES[0],
        SCALE_ENVELOPE_NAMES[1],
    ),
    "resumTransition": (
        "resumTransition1",
        "resumTransition2",
        "resumTransition3",
    ),
}


def tnp_group(names):
    """The ``resumTNP`` impact group for whichever TNPs are registered."""
    return tuple(n for n in names if n.startswith(TNP_PREFIX_OUT))


def pdf_group(names):
    """The ``pdf`` impact group for whichever eigenvector coefficients exist."""
    return tuple(n for n in names if n.startswith(PDF_PREFIX_OUT))


# --- The correction is the authority for the central values -------------------
#
# The model returns ``sigma_gen(p) / sigma_gen(p_anchor)``, which multiplies the
# card's templates. Those were reweighted by a theory correction, so the ratio is
# 1 at the fit start only if ``p_anchor`` is the correction's point. The
# histmaker records the correction's resummed runcard verbatim
# (``response.CORR_CONFIG_META_KEY``); these tables say where each parameter's
# anchor sits in it and which settings must agree for the cache to be the same
# calculation.

# Anchor location as ``(section, key, index)``; ``index`` picks one element of a
# list-valued setting, ``None`` takes the whole value. TNPs are handled by prefix
# in :func:`corr_anchor_key`.
CORR_ANCHOR_KEYS = {
    "alphaS": ("QCD", "alphas_mu0", None),
    "lambda_inf": ("Nonperturbative", "lambda_inf", None),
    "lambda2": ("Nonperturbative", "lambda2", None),
    "lambda4": ("Nonperturbative", "lambda4", None),
    "lambda6": ("Nonperturbative", "lambda6", None),
    "delta_lambda2": ("Nonperturbative", "delta_lambda2", None),
    "lambda_inf_nu": ("Nonperturbative", "lambda_inf_nu", None),
    "lambda2_nu": ("Nonperturbative", "lambda2_nu", None),
    "lambda4_nu": ("Nonperturbative", "lambda4_nu", None),
    "lambda6_nu": ("Nonperturbative", "lambda6_nu", None),
    "b0_over_bmax_nu": ("Nonperturbative", "b0_over_bmax_nu", None),
    # The runcard carries the transition points as one list.
    "resumTransition1": ("Calculation_settings", "transition_points", 0),
    "resumTransition2": ("Calculation_settings", "transition_points", 1),
    "resumTransition3": ("Calculation_settings", "transition_points", 2),
}

# Central values fixed by how SCETlib registers the parameter: the scale factors
# multiply the runcard's kappafo / kappaf (in CORR_REFUSE_KEYS), so 1 is the
# runcard's own choice; pdfEig* are 0 (handled in :func:`structural_central`).
STRUCTURAL_CENTRAL = {
    "resumScaleMuR": 1.0,
    "resumScaleMuF": 1.0,
}

# REFUSE: a difference means the cache computes a different function, which no
# parameter move recovers.
CORR_REFUSE_KEYS = frozenset(
    {
        # The PDF.
        "QCD.pdf_set",
        "QCD.pdf_member",
        # Perturbative content.
        "QCD.alphas_order",
        "QCD.nf",
        "Calculation_settings.fixed_order",
        "Calculation_settings.run_order",
        # What the parameters mean: the NP form,
        "Nonperturbative.np_model",
        "Nonperturbative.np_model_nu",
        "Nonperturbative.np_model_tmd",
        #   how the NP factor enters
        "Calculation_settings.form_np_prescription",
        #   how transition_points are interpreted (transition_type is an alias)
        "Calculation_settings.profile_functional_form",
        "Calculation_settings.transition_type",
        #   the b* prescription (a change silently redefines every lambda)
        "Calculation_settings.b0_over_bmax",
        "Calculation_settings.b0_over_bmax_global",
        "Calculation_settings.lambda",
        #   the scale choices the profile is built on
        "Calculation_settings.muf_follows_mub",
        "Calculation_settings.compensate_fo",
        "Calculation_settings.disable_asymmetry",
        "Calculation_settings.recoil_scheme",
        "Calculation_settings.scale_setting",
        "Calculation_settings.alphas_solution",
        "Calculation_settings.rge_solution",
        #   the base the resumScaleMuR / resumScaleMuF factors multiply
        "Calculation_settings.kappafo",
        "Calculation_settings.kappaf",
        "Calculation_settings.mufo_fixed",
        #   the profile floors
        "Calculation_settings.mu0_min",
        "Calculation_settings.mub_min",
        "Calculation_settings.mus_min",
        "Calculation_settings.nus_min",
        "Calculation_settings.muf_min",
        "Calculation_settings.muf_max",
        # The EW input and the process.
        "Electroweak.alphaem",
        "Electroweak.sin2_thw",
        "Electroweak.mz",
        "Electroweak.gammaz",
        "Electroweak.mw",
        "Electroweak.gammaw",
        "Electroweak.ckm",
        "Process.boson",
    }
)

# WARN: parameter values, which the fit can move back (``alphas_mu0`` is served
# by PDF members, so a mismatch there is accepted interpolation). Every numeric
# Nonperturbative key, and the TNP values, are also WARN by rule in
# :func:`compare_corr_config`.
CORR_WARN_KEYS = frozenset(
    {
        "QCD.alphas_mu0",
        "Calculation_settings.transition_points",
    }
)

# Every other key is not compared (e.g. calculation_piece, which differs by
# construction: the correction is resummed-only with a DYTurbo nonsingular, the
# cache computes the matched total; the Grid_* ranges; Integration accuracy).
# Keys only the correction carries are returned as "not checked", so compare
# against ``core.conf`` (the runcard layered on defaults.conf), not the bare
# runcard. The fixed-order half of a scetlib_dyturbo correction carries no
# config and is not covered.


def same_setting(a, b):
    """Are two runcard settings (strings) the same value?

    Numeric when both parse as numbers, case-insensitive otherwise.
    """
    if a is None or b is None:
        return a == b
    try:
        return abs(float(str(a).strip()) - float(str(b).strip())) < 1e-9
    except ValueError:
        return str(a).strip().lower() == str(b).strip().lower()


def split_tnp(raw):
    """``(value, mode)`` from a TNP setting, or ``(None, None)`` if unreadable.

    SCETlib stores a TNP as ``(0., 'level0')``: the value is the anchor (WARN),
    the mode is structural (REFUSE).
    """
    try:
        val, mode = ast.literal_eval(str(raw).strip())
    except (ValueError, SyntaxError, TypeError):
        return None, None
    try:
        return float(val), str(mode)
    except (TypeError, ValueError):
        return None, None


def _is_numeric(value):
    try:
        float(str(value).strip())
    except (TypeError, ValueError):
        return False
    return True


def corr_anchor_key(rabbit):
    """``(section, key, index)`` the correction records this parameter under.

    ``None`` for a parameter the correction cannot supply (see
    :func:`structural_central`).
    """
    if rabbit in CORR_ANCHOR_KEYS:
        return CORR_ANCHOR_KEYS[rabbit]
    if rabbit.startswith(TNP_PREFIX_OUT):
        # The runcard lowercases its keys; element 0 of (value, mode) is the anchor.
        return ("TNPs", rabbit[len(TNP_PREFIX_OUT) :].lower(), 0)
    return None


def structural_central(rabbit):
    """Central value fixed by SCETlib's registration, or ``None``."""
    if rabbit in STRUCTURAL_CENTRAL:
        return STRUCTURAL_CENTRAL[rabbit]
    if rabbit.startswith(PDF_PREFIX_OUT):
        return 0.0
    return None


def uncovered_params(rabbit_names):
    """Registry names for which we can state no central value at all.

    Should be empty: the map is hand-maintained, so this catches a SCETlib
    rename that would otherwise silently keep the cache value.
    """
    return tuple(
        n
        for n in rabbit_names
        if corr_anchor_key(n) is None and structural_central(n) is None
    )


# The NP model's parameters (np_eff_ / np_gnu_ families), under rabbit names.
_NP_RABBIT_NAMES = frozenset(
    rabbit
    for scetlib, rabbit in EXPLICIT_NAMES.items()
    if scetlib.startswith(("np_eff_", "np_gnu_"))
)


def tape_served(rabbit):
    """May a theory correction be produced at a value other than the build point?

    True for the NP parameters and TNP values, which ride the AD tape and are
    exact anywhere. False otherwise, conservatively: ``alphaS`` and ``pdfEig*``
    are exact only at the built PDF members. This is the only definition of the
    distinction; callers derive "may be overridden" from it plus
    :func:`corr_anchor_key`.
    """
    return rabbit in _NP_RABBIT_NAMES or rabbit.startswith(TNP_PREFIX_OUT)


def corr_anchor_value(config, rabbit):
    """The correction's central value for *rabbit*, or ``None`` if unrecorded.

    Raises on a setting that is present but unreadable.
    """
    where = corr_anchor_key(rabbit)
    if where is None:
        return None
    section, key, index = where
    body = config.get(section)
    if not isinstance(body, dict) or key not in body:
        return None
    raw = body[key]
    if section == "TNPs":
        val, _ = split_tnp(raw)
        if val is None:
            raise ValueError(
                f"{section}.{key} = {raw!r} is not a SCETlib TNP "
                f"(value, mode) tuple, so {rabbit}'s anchor cannot be read."
            )
        return val
    if index is None:
        try:
            return float(str(raw).strip())
        except (TypeError, ValueError):
            raise ValueError(
                f"{section}.{key} = {raw!r} is not numeric, so {rabbit}'s "
                f"anchor cannot be read."
            )
    try:
        seq = ast.literal_eval(str(raw).strip())
        return float(seq[index])
    except (ValueError, SyntaxError, TypeError, IndexError, KeyError):
        raise ValueError(
            f"{section}.{key} = {raw!r} is not a list with at least "
            f"{index + 1} entries, so {rabbit}'s anchor cannot be read."
        )


def compare_corr_config(corr_cfg, cache_cfg):
    """Compare a recorded correction runcard against the cache's own.

    Returns ``(refuse, warn, corr_only)``. The first two are
    ``[(name, corr_value, cache_value)]`` for the declared keys that disagree;
    ``corr_only`` names the keys only the correction carries ("not checked").
    Section names are compared as spelled, keys lowercased.
    """
    refuse, warn, corr_only = [], [], []
    for section, corr_body in sorted(corr_cfg.items()):
        if not isinstance(corr_body, dict):
            continue
        cache_body = cache_cfg.get(section)
        cache_body = (
            {str(k).lower(): v for k, v in cache_body.items()}
            if isinstance(cache_body, dict)
            else {}
        )
        for key, corr_val in sorted(corr_body.items()):
            key = str(key).lower()
            name = f"{section}.{key}"
            if key not in cache_body:
                corr_only.append(name)
                continue
            cache_val = cache_body[key]
            if section == "TNPs":
                cv, cm = split_tnp(corr_val)
                hv, hm = split_tnp(cache_val)
                if cv is None or hv is None:
                    refuse.append((f"{name} [unreadable]", corr_val, cache_val))
                    continue
                if not same_setting(cv, hv):
                    warn.append((f"{name} value", cv, hv))
                if not same_setting(cm, hm):
                    refuse.append((f"{name} mode", cm, hm))
            elif name in CORR_REFUSE_KEYS:
                if not same_setting(corr_val, cache_val):
                    refuse.append((name, corr_val, cache_val))
            elif name in CORR_WARN_KEYS or (
                section == "Nonperturbative" and _is_numeric(corr_val)
            ):
                if not same_setting(corr_val, cache_val):
                    warn.append((name, corr_val, cache_val))
    return refuse, warn, corr_only
