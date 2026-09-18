"""Parameter registry for the SCETlib autodiff param model.

SCETlib owns the parameter vector: :meth:`DrellYan.gradient_param_names` returns
the differentiable parameters in a FIXED order that is baked into the cached bin
rules (the rule fingerprint hashes the names in order, so any addition or
reordering invalidates a cache). This module is the translation layer between
that vector and the names rabbit sees.

Rabbit-facing names use the spelling the analysis tooling already reads
(``lambda2``, ``lambda2_nu``, …) so the postfit readers, the cross-run
fit-summary tools and the impact-group labels work unchanged. ``alphas`` becomes
``alphaS`` in PHYSICAL units (0.118-ish), NOT the ``pdfAlphaS`` template's
Delta(alpha_s) = 0.002-per-theta convention.
"""

import ast
import math

# --- SCETlib gradient name -> rabbit-facing name -----------------------------
#
# Every entry here is an exact-match rename. Names not listed fall through
# :func:`rabbit_name`, which handles the two open-ended families (TNPs and, in
# a later phase, PDF eigenvector coefficients).
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
    # Profile scales and matching transition points, registered by
    # set_diff_scales(1). scale_kappa_F is inert in the kernel -- the slot exists
    # only for build_pdf_variations to tie the muF member pair to -- so it does
    # nothing unless the cache was built with has_muf.
    "scale_kappa_R": "resumScaleMuR",
    "scale_kappa_F": "resumScaleMuF",
    "scale_x1": "resumTransition1",
    "scale_x2": "resumTransition2",
    "scale_x3": "resumTransition3",
}

# TNPs: ``tnp_gamma_cusp`` -> ``resumTNP_gamma_cusp``. The prefix matches the
# group setupRabbit gives the discrete TNP templates (``resumTNP``), so a
# grouped-impact bar stays comparable between the template and model treatments.
TNP_PREFIX_IN = "tnp_"
TNP_PREFIX_OUT = "resumTNP_"

# PDF eigenvector coefficients, appended after the physics parameters when the
# cache carries PDF variations (phase 4). ``c_e`` are standard N(0,1) Hessian
# coefficients.
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
# Only consulted when the model is constructed with ``priors=1``; otherwise every
# parameter floats free. ``None`` means "free even when priors are on".
#
# The lambda sigmas are the ones the analysis uses for the corresponding template
# nuisances, so the nonperturbative sector is constrained the same way whether it
# is fitted continuously or morphed. TNPs are genuine N(0,1) nuisances --
# theta is normalised upstream so |theta|=1 IS the recommended variation
# (prod/scetlib_run/examples/theory_nuisance_parameters/*.conf) -- and get
# sigma = 1 by default, unlike the free lambdas.
# NOTE the units. These sigmas are in RABBIT's parameter units, which for a
# REPARAM'd name is theta, not the physical variable. So a reparametrised
# parameter whose map already carries the physical scale wants sigma = 1.0 --
# |theta| = 1 IS 1 sigma, the same convention as the TNPs and pdfEig* -- and
# leaving the old PHYSICAL width here instead would silently tighten the prior
# by a factor of that width (lambda2 would go from 0.4 +- 0.5 to 0.4 +- 0.25).
# The physical widths the analysis chose now live in REPARAM as the map widths.
# Which parameters are FREE. Everything else is constrained at sigma = 1.
#
# That is the whole declaration, and it is deliberately a SET rather than a
# {name: sigma} table. A table invites the units bug: sigma is in RABBIT's
# units, which for a reparametrised name is theta, so a physical width left
# there gets multiplied by the map's width and the prior silently shrinks
# (lambda2 went from 0.4 +- 0.5 to 0.4 +- 0.25 exactly this way). With every
# constrained parameter at sigma = 1 there is no number here to get wrong: the
# physical 1 sigma is the REPARAM width, in one place, and
#
#     physical 1 sigma = width x sigma = width.
#
# A parameter with no REPARAM entry is already unit-normalised upstream -- the
# TNPs carry an N(0,1) constraint by construction, and pdfEig* have their CL
# convention in the coefficient map -- so sigma = 1 is right for them too, and
# they need no entry anywhere.
#
# Notes on the free ones:
#   alphaS            the POI.
#   lambda_inf, lambda_inf_nu, b0_over_bmax_nu
#                     shape constants of the NP form, frozen by default (see
#                     DEFAULT_FROZEN); free rather than constrained so a study
#                     that deliberately floats one is not fighting a prior it
#                     did not choose.
#   resumTransition1/3
#                     the analysis varies only the CENTRAL transition point
#                     ("new recommendation from Frank for variation of central
#                     transition parameter only"), so floating the outer two
#                     would ADD uncertainty the card does not carry. Frozen by
#                     default; no reference variation exists to normalise them
#                     against, which is also why they have no REPARAM map.
FREE_PARAMS = frozenset(
    {
        "alphaS",
        "lambda_inf",
        "lambda_inf_nu",
        "b0_over_bmax_nu",
        "resumTransition1",
        "resumTransition3",
    }
)


def prior_sigma(rabbit):
    """Gaussian prior sigma in RABBIT's units: None if free, else 1.0.

    There is no per-parameter width here by design -- see FREE_PARAMS. The
    physical width lives in REPARAM, so this answers only "is it constrained".
    """
    return None if rabbit in FREE_PARAMS else 1.0


# --- PDF confidence-level convention -----------------------------------------
#
# A Hessian PDF set's members are not all 1 sigma. CT18Z's are 90% CL
# displacements, and some sets additionally carry an analysis-chosen inflation.
# The analysis keeps both numbers in one place, ``theory_utils.pdfMap``, and
# composes them the same way everywhere:
#
#     scale = pdf_inflation_factor(pdfMap[set], noi) * pdfMap[set]["scale"]
#
# (``rabbit_theory_helper.add_pdf_uncertainty`` when it builds the templates;
# ``postfit_pdf_helper`` when it reads them back). For CT18Z with noi=alphaS
# that is 1.0 * 1/1.645 = 0.60790, and MEASURED off the 2D card's own logk it is
# 0.6025 +- 0.0055 over the 29 eigenvectors -- so the templates really do carry
# it and the model must too, or the swap inflates the PDF uncertainty by 1.645.
#
# WHERE it goes is the one place the model can do better than a template. The
# template route has no choice but to scale the RESPONSE, because a template is
# a fixed shape that can only morph linearly. The model evaluates SCETlib at
# whatever coefficient it is handed, and
#
#     I(c) = I_0 + c (I_+ - I_-)/2 + c^2 (I_+ + I_- - 2 I_0)/2
#
# is exactly quadratic in c (DrellYan.hpp, exact at c = 0, +-1). So the model
# scales the COEFFICIENT instead -- theta = +-1 evaluates the calculation at the
# 68% CL point in eigenvector space, which is what a 1 sigma PDF displacement
# physically is. The two agree to O(c) and differ by (scale^2 - scale) times the
# quadratic part, i.e. wherever the up and down members are not mirror images.
PDF_COEFF_SCALE_NOI_DEFAULT = ("alphaS",)


def pdf_set_key(lha_name, pdf_map=None):
    """``theory_utils.pdfMap`` key whose ``lha_name`` is *lha_name*.

    The cache runcard names the set the way LHAPDF does (``CT18ZNNLO``); the map
    is keyed by the analysis' short name (``ct18z``). Raises if the set is not
    in the map -- returning 1.0 for an unknown set would silently drop a
    convention that is a factor 1.645 for the one set we use.
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

    Mirrors ``postfit_pdf_helper.PostfitPdfHelper`` (which does the same product
    the other way round) so the model and the templates cannot drift apart:
    the per-set 90%->68% ``scale`` times the nuisance-of-interest inflation.
    """
    from wremnants.utilities import theory_utils

    pdf_map = theory_utils.pdfMap if pdf_map is None else pdf_map
    info = pdf_map[pdf_set_key(lha_name, pdf_map)]
    noi = list(PDF_COEFF_SCALE_NOI_DEFAULT if noi is None else noi)
    return float(theory_utils.pdf_inflation_factor(info, noi)) * float(
        info.get("scale", 1)
    )


# --- Reparametrisation: unit nuisances for the profile scales -----------------
#
# SCETlib registers the profile scales as the PHYSICAL quantities:
# ``scale_kappa_R`` and ``scale_kappa_F`` are kappa itself with central 1, and
# ``scale_x1..x3`` are the transition points themselves. That is the right
# interface for a calculation, but it is the wrong one for a nuisance, because
# the template variations these REPLACE are not symmetric in the physical
# variable:
#
#   kappaFO   x2 and /2          -> symmetric in ln(kappa), not in kappa
#   x2        0.6 -> 0.35, 0.75  -> genuinely asymmetric, -0.25 / +0.15
#
# and rabbit's ParamModel priors are a single symmetric Gaussian per parameter
# (fitter.py: cw = 1/sigma^2, one scalar, no up/down hook). Tuning sigma cannot
# reproduce either variation: sigma = 0.5 on a linear kappa_R gives [0.5, 1.5],
# understating the up side.
#
# So the fitted parameter is a UNIT nuisance theta and the model maps it to the
# physical value, exactly as SCETlib itself does for the PDF eigenvectors and
# the muF pair ("exact at 0, +-1, quadratic in between"). Every replaced-template
# direction is then sigma = 1, the same convention as the TNPs and pdfEig*.
#
#   "log"  : value = exp(theta * L)          theta = +-1 -> exp(+-L)
#   "quad" : value = c0 + c1*theta + c2*theta^2
#
# The log form has a second benefit: exp() is positive by construction, so it
# cannot trip SCETlib's silent `p[_muf_index] > 0. ? ... : 1.` fallback, which
# would drop the muF variation with no error at all.
LN2 = math.log(2.0)
REPARAM = {
    # kappa_R: theta = +-1 -> kappa_R = 2 / 0.5, matching kappaFO x2 and /2.
    "resumScaleMuR": ("log", (LN2,)),
    # kappa_F: same. SCETlib converts internally to t = ln(kappa_F)/ln(2), so
    # theta = +-1 lands exactly on the two members that were built (0.5, 2.0).
    "resumScaleMuF": ("log", (LN2,)),
    # x2: the quadratic through the three points the analysis actually uses --
    # theta = -1 -> 0.35, theta = 0 -> 0.6, theta = +1 -> 0.75. Monotone for
    # |theta| < 2 (the derivative 0.20 - 0.10*theta vanishes at theta = 2).
    "resumTransition2": ("quad", (0.6, 0.20, -0.05)),
    # resumTransition1/3 are deliberately NOT reparametrised: they are frozen by
    # default and no reference variation exists for them, so a study that floats
    # them should do so in the physical variable and choose its own range.
    #
    # --- "unit": value = <cache anchor> + width * theta -----------------------
    #
    # The remaining physical parameters. Before this, alphaS and the five NP
    # lambdas were the only ones rabbit saw in physical units, which is what
    # made them the odd ones out: 41 of 47 fitted parameters were already unit
    # nuisances. Normalising them keeps rabbit generic (it only ever sees
    # theta ~ O(1)) and collapses the curvature spread that made the
    # preconditioner necessary -- one block had max|diag| = 3.3e+09 against
    # singletons at exactly 1, and that nine-order range IS this units mismatch.
    #
    # c0 is NOT written here: the model fills it from the CACHE ANCHOR, so
    # theta = 0 reproduces the anchor by construction and cannot drift when a
    # cache is built at a different tune. Only the width is a choice, and it is
    # the parameter's own natural step:
    #   alphaS        the PDF set's alphasRange (0.002 for CT18Z) -- exactly the
    #                 Delta(alpha_s)-per-theta convention the pdfAlphaS template
    #                 used, so theta here means what it meant before.
    #   the lambdas   their PRIOR_SIGMAS widths, so |theta| = 1 is 1 sigma of the
    #                 prior the analysis already chose.
    "alphaS": ("unit", (0.002,)),
    "lambda2": ("unit", (0.50,)),
    "lambda4": ("unit", (0.50,)),
    "delta_lambda2": ("unit", (0.50,)),
    "lambda2_nu": ("unit", (0.10,)),
    # lambda6 / lambda6_nu: the tanh_6 form's third coefficient. Only active
    # when np_model / np_model_nu selects tanh_6, but reparametrised anyway so
    # that EVERY constrained parameter obeys the same rule -- sigma = 1 in
    # theta, physical width in the map. Leaving one physical is how a prior gets
    # silently rescaled the next time someone adds a map to it.
    "lambda6": ("unit", (0.10,)),
    "lambda6_nu": ("unit", (0.10,)),
    "lambda4_nu": ("unit", (0.50,)),
}


def reparam(rabbit):
    """``(kind, coeffs)`` for a reparametrised name, else ``None``."""
    return REPARAM.get(rabbit)


# --- Defaults ----------------------------------------------------------------
#
# Parameters frozen unless the user asks otherwise. These are shape constants of
# the SCETlib nonperturbative forms, not physics we fit: lambda_inf sets the
# saturation of the tanh forms and b0_over_bmax_nu the b* convention.
DEFAULT_FROZEN = (
    "lambda_inf",
    "lambda_inf_nu",
    "b0_over_bmax_nu",
    # Inert for the Z, not a choice: b_qqDS scales a channel that does not
    # contribute, so its whole response is O(1e-16) and its Jacobian column is
    # identically zero. A zero column is a zero row+column of the NLL Hessian,
    # i.e. a singular covariance -- `_check_no_inert_params` refuses it, so this
    # name has to be frozen rather than "fitted with a prior". Measured
    # 2026-08-27 (studies/.../260827-authoritative-validation, 06_model_defaults).
    "resumTNP_b_qqDS",
    # Only the CENTRAL matching transition point is varied in the analysis; see
    # the PRIOR_SIGMAS comment. Float these two only as a deliberate study.
    "resumTransition1",
    "resumTransition3",
)

# resumTransition2 was frozen here from 2026-08-21 to 2026-08-25 because the
# derivative SCETlib handed us for it was sign-inverted -- moving the transition
# points moves muF while the per-node beam convolutions stayed at the config's
# muF. UNFROZEN: scetlib-cms bfc6be6 feeds the induced per-node muF shift into
# the muF member interpolation, and validate_variations now gives 1.1e-03 ..
# 3.4e-03 against the production templates where it was 1.1e-01 .. 1.99e-01.
#
# Two caveats that survive the fix:
#   * the residual is the interpolation's own limit, not a bug -- the induced
#     shift is carried by a quadratic through three knots (kappa_F = 0.5/1/2),
#     exact only AT the knots, and a transition variation lands between them.
#     Upstream quotes +7.8e-04 against an independent runcard route.
#   * the fix works THROUGH the muF member pair, so it does nothing without one.
#     Upstream guards this rather than leaving it silent: moving scale_x1..x3 off
#     the anchor on a cache with no muF pair RAISES ("the result would be WRONG,
#     sign included"), and says which call builds the pair. So a --no-pdf cache
#     cannot float the transition points at all -- which is also why
#     backend_check fails on one.

# Directions whose response has been MEASURED to disagree with the template it
# replaces, keyed by rabbit-facing name -> why. These are not frozen for physics
# reasons, so anyone who floats one anyway deserves to be told once, loudly,
# rather than to find out from a pull. Keep the strings short; the detail lives in
# studies/scetlib-ad-param-model/.
# Empty since 2026-08-25: every direction the model registers now agrees with its
# production template. The worst of 39 is 1.4e-02 (mufup, qT [0,1], which is
# template precision rather than a model error -- model/direct = 1.0000 there),
# and 37 of them are <= 7.5e-03. Kept as the hook, because "this direction's
# response is known-wrong, do not profile it" is worth being able to say loudly
# rather than in a comment.
KNOWN_BAD_RESPONSE = {}


# Grouped impacts over the model's own parameters (rabbit resolves these labels
# to floating x-indices; see Fitter._resolved_param_impact_groups). Membership is
# intersected with the parameters actually registered.
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
    # Named to line up with the card groups they replace, so a grouped-impact
    # bar stays comparable between the template and model treatments.
    "resumScale": ("resumScaleMuR", "resumScaleMuF"),
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
# The model returns ``sigma_gen(p) / sigma_gen(p_anchor)``, which MULTIPLIES the
# card's templates. Those templates were reweighted by a theory correction, so
# the ratio is 1 at the fit start only if ``p_anchor`` is the point THAT
# correction was computed at. The anchor is therefore defined by the correction,
# not by the cache: we are applying a ratio on top of what the histmaker already
# produced, so we must know what the histmaker used. Without it there is nothing
# to form the ratio about.
#
# Note precisely what does and does not depend on where the CACHE was built. The
# ratio is correct insofar as the cache is EVALUATED at the correction's point,
# which is a statement about ``p_anchor`` alone. Where the cache was built
# affects ROBUSTNESS: the NP lambdas and the TNPs are AD-tape-carried and exact
# anywhere, while ``alphaS`` is member-served and the PDF eigenvectors are exact
# only at c = 0, +-1, so only those degrade away from the build point.
#
# The histmaker records the whole resummed runcard verbatim
# (``lambda_central.CORR_CONFIG_META_KEY``) and these tables say what to do with
# it: where each registry parameter's central value comes from, and which
# settings must agree for the cache to be the same calculation at all.

# Where each registry parameter's anchor lives in the correction's runcard, as
# ``(section, key, index)``. ``index`` selects one element of a list-valued
# setting (the three transition points) or of a ``(value, mode)`` TNP tuple;
# ``None`` takes the whole value. The TNP family is handled by prefix in
# :func:`corr_anchor_key`, so this table holds only the named parameters.
#
# Measured against the real cache (53 registered names, 2026-09-10): 1 alphaS
# + 8 NP lambdas + 10 TNPs + 5 profile/transition + 29 pdfEig. Every one is
# covered here or by :func:`structural_central`; the leftover set is empty, and
# :func:`uncovered_params` is what keeps that true.
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
    # SCETlib registers the transition points individually but the runcard
    # carries them as one list, so each takes its own element.
    "resumTransition1": ("Calculation_settings", "transition_points", 0),
    "resumTransition2": ("Calculation_settings", "transition_points", 1),
    "resumTransition3": ("Calculation_settings", "transition_points", 2),
}

# Parameters whose central value is fixed by how SCETlib REGISTERS them, so the
# runcard neither carries it nor could disagree with it:
#   resumScaleMuR/MuF   ad_context.cpp registers scale_kappa_R and scale_kappa_F
#                       with a hardcoded central of 1. They are multiplicative
#                       factors ON TOP of whatever the runcard's kappafo /
#                       kappaf set, so 1 is "the runcard's own scale choice".
#                       (kappafo / kappaf themselves are structural and sit in
#                       CORR_REFUSE_KEYS.)
#   pdfEig*             0 is the central member, which is what pdf_member = 0
#                       means; the coefficients are displacements from it.
STRUCTURAL_CENTRAL = {
    "resumScaleMuR": 1.0,
    "resumScaleMuF": 1.0,
}

# --- What we declared ahead of time to matter --------------------------------
#
# REFUSE: the cache computes a different FUNCTION, so no parameter move
# recovers it. Four groups, and the test for each is "would a difference here
# mean the cache is not the calculation the histmaker's templates carry?".
CORR_REFUSE_KEYS = frozenset(
    {
        # (2) The PDF. A different set is a different calculation entirely, and
        # the cache's eigenvector and alpha_s columns are built from its members.
        "QCD.pdf_set",
        "QCD.pdf_member",
        # (3) Perturbative content: what orders were computed.
        "QCD.alphas_order",
        "QCD.nf",
        "Calculation_settings.fixed_order",
        "Calculation_settings.run_order",
        # (4) What the parameters MEAN. The same number under a different
        # functional form or mode is a different prediction -- the group easiest
        # to forget, hence the length.
        #   the NP form: the tape computes a different function
        "Nonperturbative.np_model",
        "Nonperturbative.np_model_nu",
        "Nonperturbative.np_model_tmd",
        #   how the NP factor enters
        "Calculation_settings.form_np_prescription",
        #   how transition_points are INTERPRETED, without which the values in
        #   CORR_ANCHOR_KEYS are meaningless (transition_type is its alias)
        "Calculation_settings.profile_functional_form",
        "Calculation_settings.transition_type",
        #   the b* prescription. b0_over_bmax_global = 0 makes b* the identity,
        #   so a change here silently redefines every lambda.
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
        #   the base the resumScaleMuR / resumScaleMuF maps multiply. Not in the
        #   original list: added because they ARE the central value of two
        #   registry parameters (see STRUCTURAL_CENTRAL), so a difference here
        #   moves an anchor without moving any number this check would see.
        "Calculation_settings.kappafo",
        "Calculation_settings.kappaf",
        "Calculation_settings.mufo_fixed",
        #   the profile floors, which set where the resummation is cut off
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

# WARN: parameter VALUES. The AD tape is exact away from the build point, so the
# fit can move a lambda or a TNP back; a mismatch is a bookkeeping error, not a
# broken calculation. ``alphas_mu0`` is the asymmetric case -- it is served by a
# PDF member pair and INTERPOLATED, so warning there is a deliberate acceptance
# of interpolation error rather than a free pass. Every shared numeric key of
# Nonperturbative is covered by rule (see :func:`compare_corr_config`) so a new
# lambda cannot go unnoticed, and the TNP values are split out of their tuples.
CORR_WARN_KEYS = frozenset(
    {
        "QCD.alphas_mu0",
        "Calculation_settings.transition_points",
    }
)

# NOT COMPARED AT ALL. Not an allowlist -- these are simply outside the check,
# and nothing outside the two tables above is reported at all. An allowlist has
# to be maintained against a config that keeps growing, and a report nobody acts
# on trains people to ignore the output.
#   calculation_piece   differs BY CONSTRUCTION (the reference runs resummed-only
#                       and takes the nonsingular from DYTurbo; the cache
#                       computes the matched total in one go)
#   Grid_*              the fit's range versus the production grid. qT is in fact
#                       an exact 71-edge match after the corrgrid work; Q and Y
#                       are deliberately narrower.
#   Integration         quadrature accuracy (1e-3 vs 1e-4), a precision choice,
#                       consistent with the 0.0089% agreement already measured.
#   Singlet_scheme, and every key on neither table.
#
# TWO ASYMMETRIES, stated so we are not fooling ourselves, both MEASURED on the
# current pair (2026-09-10):
#
# 1. CORRECTION-ONLY keys are reported, not failed. The correction's runcard is
#    SCETlib's RESOLVED config (defaults included) while a bare cache runcard
#    holds only what was written. Against the bare file that leaves 33 keys with
#    no counterpart -- exactly the ones a build-default difference would hide
#    (lambda6, lambda6_nu, lambda4_i, np_model_tmd, the 11 per-flavour
#    lambda2_*, kappafo). The model therefore compares against ``core.conf``,
#    the runcard LAYERED ON defaults.conf, which is what the calculation is
#    configured from; against that, the count is 0.
#    :func:`compare_corr_config` returns whatever remains, so the model can say
#    what it did not check.
#
# 2. CACHE-ONLY keys are never visited AT ALL, because the loop iterates the
#    correction's keys -- a key the correction lacks has no value to disagree
#    with. That could in principle hide a real difference, so it was measured:
#    there are 12, and every one is fixed-order / matching machinery
#    (Calculation_settings.fo_order2_* and matched_nons_qt_cut). That is not an
#    accidental gap. calculation_piece is `matched` for the cache against `sing`
#    for the correction BY CONSTRUCTION -- the correction runs resummed-only and
#    takes its nonsingular from DYTurbo, the cache computes the matched total in
#    one go -- so those 12 knobs exist precisely because of that difference, and
#    they sit inside the fixed-order exclusion stated just below.
#    (fo_order2_analytic is `yes` in the cache runcard, `no` in today's
#    defaults.conf, and absent entirely from the correction's resolved config:
#    the build that made the correction predates the knob.)
#
# Also not covered: the fixed-order half of a scetlib_dyturbo correction. Only
# the resummed file's runcard is recorded, and the DYTurbo side carries no
# config, so "the correction is the authority" holds for the resummed sector.


def same_setting(a, b):
    """Are two runcard settings the same value?

    Type-tolerant on purpose: both sides carry every value as a STRING, but one
    may write ``0.118`` where the other writes ``0.1180``, and case differs on
    the enum-like settings. So compare numerically when both parse as numbers
    and case-insensitively otherwise. Coercing to float unconditionally would
    fail outright on ``tanh_2``; comparing raw strings would report ``0.``
    against ``0.0`` as a mismatch.
    """
    if a is None or b is None:
        return a == b
    try:
        return abs(float(str(a).strip()) - float(str(b).strip())) < 1e-9
    except ValueError:
        return str(a).strip().lower() == str(b).strip().lower()


def split_tnp(raw):
    """``(value, mode)`` from a TNP setting, or ``(None, None)`` if unreadable.

    SCETlib stores a TNP as ``(0., 'level0')``, and the two halves belong to
    DIFFERENT classes: the value is the anchor (WARN, the AD is exact so the fit
    can move it) while the mode is structural (REFUSE, since the same number
    means a different variation under a different mode). A whole-string compare
    would lump them together and ``float()`` on the whole string would just fail.
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

    ``None`` for a parameter the correction cannot supply -- see
    :func:`structural_central` for the ones whose central value is fixed by how
    SCETlib registers them.
    """
    if rabbit in CORR_ANCHOR_KEYS:
        return CORR_ANCHOR_KEYS[rabbit]
    if rabbit.startswith(TNP_PREFIX_OUT):
        # The runcard lowercases its keys (b_qqV -> b_qqv), and element 0 of the
        # (value, mode) tuple is the anchor.
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

    Empty for every cache we have built. Not an assertion for its own sake: the
    correction-key -> registry map is hand-maintained, so a SCETlib rename would
    otherwise drop a parameter quietly into "keep the cache value", which is the
    exact failure this machinery exists to remove.
    """
    return tuple(
        n
        for n in rabbit_names
        if corr_anchor_key(n) is None and structural_central(n) is None
    )


def corr_anchor_value(config, rabbit):
    """The correction's central value for *rabbit*, or ``None`` if unrecorded.

    Raises on a setting that is present but unreadable -- that is a bug in the
    recorded config, not a missing anchor, and the two want opposite handling.
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
    ``corr_only`` names the keys the correction carries and the cache does not,
    which are SCETlib defaults the comparison cannot see through and so are
    reported as "not checked", never as failures.

    Section names are compared as spelled (both sides use SCETlib's own INI
    spelling); keys are lowercased on both sides, since the cache runcard is
    mixed-case INI and the recorded correction config is not.
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
