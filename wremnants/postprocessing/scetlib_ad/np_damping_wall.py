"""Physical-damping wall for the SCETlib-AD param model.

Port of ``wremnants/postprocessing/scetlib_np/np_damping_wall.py`` (the
continuous-lambda btgrid model) to :class:`SCETlibADParamModel`. Same rabbit
``Regularizer`` mechanism and same hinge-loss (relu^2) walls; what changed is
where the wall gets its inputs, and that is the whole content of this file --
see "The two adaptations" below.

WHY A WALL. A wrong-sign lambda anti-damps the NP form factors: the tanh
argument goes negative, so gamma_nu^NP turns POSITIVE and f^NP GROWS with b_T
instead of decaying. The b_T integral then oscillates and the differential
sigma(qT) can go negative -- a genuinely unphysical region of parameter space,
not merely a disfavoured one. The fit does not see it as unphysical: the
qT -> ptVGen rebin averages the negative differential cross section away, and
the reco fold plus the softplus ratio floor (``response.RATIO_FLOOR_SCALE``)
keep the yields, the NLL and its gradient perfectly finite. So the minimiser
will walk into it if the data pull that way, and it did -- the 2026-09-10
blinded data fit landed at lambda2_nu = -0.058 physical, so the CS kernel
ANTI-DAMPS for every b_T below 1.04 GeV^-1
(studies/scetlib-ad-param-model/260910-blinding, and 260910-wall-port for the
condition-by-condition read). Its lambda4 = -0.0027 is NOT a violation on its
own -- see the arctanh-correction note below -- and the TMD side fails only
marginally, and only in the outermost rapidity bin. This regularizer is the
fit-time penalty that keeps the lambdas in the damping region.

Wall hardness is set at fit time by rabbit's ``--regularizationStrength``
(the penalty times ``exp(2*tau)`` in ``fitter.py``; ``tau`` is a fixed
multiplier, NOT a minimised parameter). A large strength makes this a BARRIER:
~0 inside the physical region, steeply rising outside. Free (unwalled) vs
walled Delta(loss) is then the data-model tension diagnostic -- railing against
a wall at large Delta(loss) is genuine tension, not a masked pathology.

--------------------------------------------------------------------------
The two adaptations (both silently break a naive port)
--------------------------------------------------------------------------

1. THE FITTED LAMBDAS ARE UNIT NUISANCES, NOT PHYSICAL LAMBDAS. Commit
   8f6af64f reparametrised every fitted parameter as ``physical = anchor +
   width * theta`` (``params.REPARAM``, kind ``"unit"``), and rabbit's
   ``Regularizer`` is handed ``get_x()`` (``fitter.py`` ``_compute_nll_components``),
   i.e. THETA. A wall written against physical lambdas and applied to theta is
   simply wrong -- for this cache theta = 0 is lambda2 = 0.4, not 0. So the
   wall maps theta -> physical itself, from ``params.REPARAM`` (the width) and
   the correction's own runcard (the anchor). Nothing is hardcoded here.

2. THERE IS NO ``indata.scetlib_np_param_model``. The old wall read the NP
   forms and the lambda order off the model, which published itself on the
   shared ``indata``. ``SCETlibADParamModel`` does not, so this wall derives
   everything from the same artefacts the model does:

     the NP forms   the recorded theory-correction runcard
                    (``response.corr_config_from_meta`` ->
                    ``Nonperturbative.np_model`` / ``np_model_nu``), which is
                    also what ``params.CORR_REFUSE_KEYS`` refuses a cache
                    mismatch on, so the form the wall reads is the form the
                    calculation computes.
     the anchors    ``params.corr_anchor_value`` on that same config -- the
                    model's own ``anchor_source="correction"`` default.
     the lambda positions
                    resolved BY NAME out of the fitter's ``parms``, so a
                    ``poi_params`` reorder or the saturated path's
                    ``CompositeParamModel`` (which concatenates submodel names
                    unchanged) is tracked automatically, where the old wall's
                    flat ``x[:nparams]`` indexing had to refuse it.
     the binding |Y|
                    the card's OWN gen |Y| reach (see ``_binding_absY``), not
                    a module constant -- the one number the old wall left
                    "UNDER TEST" at 5.0.

   TWO CONSEQUENCES WORTH STATING. The wall reads the CORRECTION's anchor, so
   a fit run with ``anchor_source=cache`` or ``anchor_override=`` shifts the
   model's map without shifting the wall's; and a lambda HELD out of the fit
   is read at the correction's value, so ``xparam_default=<held lambda>=v``
   pins it somewhere the wall does not know about. Both are loudly logged
   choices in the model, both are rare, and neither is detectable from
   ``indata`` -- so this wall PRINTS every anchor, width and held value it
   resolved at construction. Check that block against the model's own
   "anchor from the theory correction" line before trusting a walled fit.

BLINDING is not an issue. alphaS is blinded ADDITIVELY on the POI slot, but
the lambdas are POUs and rabbit never blinds those (``fitter.get_model_nui``),
so ``get_x()`` hands the wall their true values. The wall never reads the POI.

--------------------------------------------------------------------------
The walls
--------------------------------------------------------------------------

Read off the SCETlib source the fit actually links -- ``np_effective`` and
``gamma_nu_np_model`` in ``include/scetlib/qT/{NP_models,Gamma_nu}_formulas.hpp``
at b66f8de -- not off the AN's normalised parametrisation. Both NP models are
evaluated at the RAW b_T there (``Gamma_nu.cpp`` passes b* only to the
perturbative logarithm, and this cache has ``b0_over_bmax_global = 0``, which
makes b* the identity on the TMD side), so "for all b_T >= 0" below is the
exact necessary-and-sufficient condition, not a conservative proxy.

The damping criterion is "the tanh argument is >= 0 for all b" (gamma_nu <= 0 /
f^NP decaying), i.e. a polynomial in u = b_T^2 is non-negative on u >= 0:

  CS side   gamma_nu^NP(b) = -lambda_inf_nu * tanh( P(u)/lambda_inf_nu ) ,
            P(u) = lambda2_nu*u + lambda4_nu*u^2 + lambda6_nu*u^3
                                             (lambda6_nu = 0 for tanh_2)
      tanh_2:  lambda_inf_nu > 0 , lambda2_nu >= 0 , lambda4_nu >= 0
      tanh_6:  lambda_inf_nu > 0 , lambda2_nu >= 0 , lambda6_nu >= 0
               (leading), and lambda4_nu >= 0 OR
               lambda4_nu^2 <= 4*lambda2_nu*lambda6_nu (interior; lambda4_nu
               may dip negative while the b^6 term keeps P >= 0)

  TMD side  f^NP(Y,b) = exp(-2*lambda_inf*b*tanh(a)) ,
            a*lambda_inf = b*Q(u) ,  Q(u) = L2 + B*u + lambda6*u^2 ,
            B = lambda4 + L2^3/(3*lambda_inf^2) ,
            L2 = lambda2 + delta_lambda2*Y^2       (lambda6 = 0 for tanh_2)
      tanh_2:  lambda_inf > 0 , L2 >= 0 ,
               B >= 0  [written as 3*lambda_inf^2*lambda4 + L2^3 >= 0]
      tanh_6:  lambda_inf > 0 , L2 >= 0 , lambda6 >= 0 (leading),
               and B >= 0 OR B^2 <= 4*L2*lambda6            (interior)
            evaluated at Y = 0 AND Y = Y_max, which covers delta_lambda2 of
            either sign: L2 is monotonic in Y^2, so the binding |Y| is one of
            the two extremes.

NOTE THE ARCTANH CORRECTION. The TMD cube coefficient is
B = lambda4 + L2^3/(3*lambda_inf^2), NOT lambda4: ``np_effective`` adds
``(1/3)*pow3(lambda2_Y*bT/lambda_inf)`` for tanh_2 as well as tanh_6, which is
the leading arctanh correction that makes tanh(a) reproduce the unsaturated
``identity`` model. Three powers of lambda_inf, as in the source; AN-25-085
Eq. (eq:npf) writes one, which is a typo in the note, not a different
convention (see knowledge/30_physics_global/np_parametrization_constraints.md).
It matters physically here: lambda4 < 0 is NOT by itself unphysical, because
L2^3 can more than pay for it -- at the 2026-09-10 postfit tune
3*lambda4 + L2^3 = +0.019 at Y = 0 and only turns negative near |Y| = 2.5.
Multiplying every condition through by 3*lambda_inf^2 > 0 also keeps them
division-free.

All conditions are the lambda_inf-EXACT forms, and the tanh_2 ones are the
lambda6 -> 0 reduction of the tanh_6 ones (the leading coefficient is then
lambda4_nu / B, hence the simpler >= 0 limit). This file uses the DAMPING
criterion (P >= 0), which lets the interior coefficient dip negative inside the
discriminant bound; ``np_monotonicity.py`` uses the stricter MONOTONICITY
criterion (sqrt(3) in place of sqrt(4)).

Every condition is written uniformly as ``coeff >= bound`` with penalty
``relu2(bound - coeff)``, including the interior discriminants (whose coeff
``4*lambda2_nu*lambda6_nu - relu2(-lambda4_nu)`` is self-gating: it is
positive for lambda4_nu >= 0, so no penalty and no division by lambda6_nu).
That uniformity is what lets :func:`damping_conditions` serve the TF penalty,
the construction-time check on HELD lambdas, and the offline diagnostic from
one definition.

The small-b turn-on walls (lambda2_nu >= 0, L2 >= 0) are stronger than the
large-b limit: they forbid an anti-damping bump near b -> 0, not just the wrong
asymptote. ``smallb=0`` drops them, leaving only the limiting/interior
behaviour and the lambda_inf floors -- then use the postfit sigma(qT) >= 0
check as the real guard.

Invoke (nothing on the -r line repeats the model spec):

    rabbit_fit.py ... \\
      --regularizationStrength 5 \\
      -r wremnants.postprocessing.scetlib_ad.np_damping_wall.NPDampingWall \\
         wremnants.postprocessing.scetlib_ad.np_damping_wall.NPDampingMapping \\
         [smallb=0] [ymax=<float>]

References:
  AN-25-085 theory.tex Eqs. eq:npgamma, eq:npf
  knowledge/30_physics_global/np_parametrization_constraints.md (the algebra,
    both criteria, and the lambda_inf^3 cross-check against the source)
  param_model.py ``_resolve_anchor`` (why the correction owns the anchor)
"""

import numpy as np

# rabbit / TF imports are deferred to the lazy class factories at the bottom so
# this module stays importable without rabbit or TF -- which is what lets the
# offline verification scripts pull in damping_conditions() and the theta ->
# physical map. Everything at module level is numpy-only (params and response
# both are).
from wremnants.postprocessing.scetlib_ad import params as adp
from wremnants.postprocessing.scetlib_ad.response import (
    CORR_CONFIG_META_KEY,
    corr_config_from_meta,
)

# Forms this wall has damping conditions for. Anything else (frac_*, exp_*,
# tanh_1, tanh_4, identity, signed_lambda, off, ...) RAISES rather than
# silently applying the wrong tanh_2 / tanh_6 walls. Deliberate: a wall that
# guesses is worse than no wall, because the fit then reports a "physical"
# tune that is nothing of the kind.
SUPPORTED_FORMS = ("tanh_2", "tanh_6")

# SCETlib's backward-compatibility aliases that resolve to a SUPPORTED_FORMS
# entry (NP_models.hpp: hyp_tangent = tanh_2; Gamma_nu.hpp likewise). The other
# aliases ("square_root" -> frac_2, "linear" -> frac_1) resolve to unsupported
# forms and so fall through to the raise.
_FORM_ALIASES = {"hyp_tangent": "tanh_2"}

# The lambdas each side's conditions read, per resolved form. scetlib_ad has no
# per-form lambda registry to derive this from -- the model's parameter
# vocabulary comes from the CACHE's registered names, not from a table -- so the
# wall carries its own. These are exactly the names damping_conditions() below
# uses, which is the only list that can be right.
_CS_LAMBDAS = {
    "tanh_2": ("lambda_inf_nu", "lambda2_nu", "lambda4_nu"),
    "tanh_6": ("lambda_inf_nu", "lambda2_nu", "lambda4_nu", "lambda6_nu"),
}
_TMD_LAMBDAS = {
    "tanh_2": ("lambda_inf", "lambda2", "lambda4", "delta_lambda2"),
    "tanh_6": ("lambda_inf", "lambda2", "lambda4", "delta_lambda2", "lambda6"),
}

# Gen |Y| axis names the binding-|Y| search accepts, in the order tried. The
# card carries the gen binning either in the response auxiliary (reco fits) or
# as the fit channel's own axes (gen_level=1); both name it the same way.
ABSY_AXIS_NAMES = ("absYVGen", "absYVgenSig", "yVGen")

# Third NP knob SCETlib carries alongside np_model / np_model_nu. It is a
# SEPARATE flavour-dependent TMD-PDF factor (NP_model_TMD_PDF), so if it is on
# the TMD conditions below are incomplete. Values we accept as "not on".
_NP_MODEL_TMD_OFF = ("off", "none", "")

# Fixed knobs. Only smallb and ymax are exposed on the -r line; these two are
# constants to keep that line minimal, and they are the values the old wall
# ran with.
LAMBDA_INF_FLOOR = 1e-3  # positive floor on the lambda_inf saturation scales
NP_DAMPING_MARGIN = 5e-3  # positive cushion: enforce each damping coeff >= this
#              rather than >= 0, so the soft wall's equilibrium -- which sits a
#              hair PAST the knee, where the penalty gradient vanishes -- still
#              lands in the damping region. NB this cache's correction anchors
#              lambda4_nu at exactly 0, i.e. ON the boundary, so at theta = 0
#              the wall already contributes margin^2 = 2.5e-5 (times exp(2 tau))
#              and biases lambda4_nu up by >= 5e-3 physical = 0.01 theta, one
#              percent of its prior width. Set to 0 to switch the cushion off.


def resolve_form(form, side):
    """Canonical SUPPORTED_FORMS name for a runcard's NP form, or raise."""
    raw = str(form).strip().lower()
    resolved = _FORM_ALIASES.get(raw, raw)
    if resolved not in SUPPORTED_FORMS:
        raise NotImplementedError(
            f"np_damping_wall: no damping walls for the {side} form {form!r}"
            + (f" (an alias of {resolved!r})" if resolved != raw else "")
            + f"; supported: {sorted(SUPPORTED_FORMS)}. Add walls for it, or run "
            "that side with a supported form. Guessing tanh_2's conditions for "
            "another functional form would report a tune as physical when it is "
            "not, which is the failure this refusal exists to prevent."
        )
    return resolved


def _section(cfg, name):
    """A runcard section as a lowercased-key dict (both spellings occur)."""
    body = cfg.get(name)
    if not isinstance(body, dict):
        return {}
    return {str(k).strip().lower(): v for k, v in body.items()}


def forms_from_corr_config(cfg):
    """``(np_model, np_model_nu)`` canonical forms from a correction runcard.

    The correction is the authority for both, for the same reason it is the
    authority for the anchor (``param_model._resolve_anchor``): its runcard
    describes the calculation the card's templates were reweighted with, and
    ``params.CORR_REFUSE_KEYS`` already refuses a cache that computes a
    different NP function. So reading the form here cannot disagree with the
    form the fit evaluates.
    """
    npsec = _section(cfg, "Nonperturbative")
    missing = [k for k in ("np_model", "np_model_nu") if k not in npsec]
    if missing:
        raise ValueError(
            f"np_damping_wall: the recorded correction runcard has no "
            f"Nonperturbative.{missing} entry, so the NP functional form is "
            "unknown and the wall cannot know which conditions to impose. "
            "There is deliberately no default: SCETlib's own fallback is tanh_2 "
            "for both sides, and silently assuming it would wall the wrong "
            "function for any other build."
        )
    tmd = str(npsec.get("np_model_tmd", "off")).strip().lower()
    if tmd not in _NP_MODEL_TMD_OFF:
        raise NotImplementedError(
            f"np_damping_wall: the correction has Nonperturbative.np_model_tmd "
            f"= {tmd!r}, a SECOND (flavour-dependent TMD-PDF) nonperturbative "
            "factor on top of np_model. Its damping conditions are not "
            "implemented here, so the TMD walls below would be incomplete."
        )
    return (
        resolve_form(npsec["np_model"], "TMD (f^NP, np_model)"),
        resolve_form(npsec["np_model_nu"], "CS (gamma_nu, np_model_nu)"),
    )


def _binding_absY(indata):
    """Largest |Y| the fit's gen grid reaches, from the CARD.

    This is the ``Y_max`` the TMD walls are evaluated at, and it is a physics
    choice worth deriving rather than declaring: L2 = lambda2 +
    delta_lambda2*Y^2, so the wall on delta_lambda2 scales as 1/Y_max^2 and the
    old wall's "UNDER TEST" 5.0 constrains it FOUR TIMES harder than the 2.5
    the analysis actually uses (knowledge/.../np_parametrization_constraints.md
    takes y_max = 2.5). Demanding damping where the prediction is never
    evaluated is not conservatism, it is an extra constraint on the data.

    Two places carry the gen binning, and they agree by construction (the
    cache's ``Grid_Y`` IS the card's gen binning -- see the header
    ``prepare_cache_for_card.py`` writes into cache.conf):
      * the response auxiliary's ``gen_axes`` (reco fits), and
      * the fit channel's own axes (``gen_level=1``, where the channel IS the
        gen binning).
    Raise if neither has one, rather than fall back to a number: with no gen
    axis the wall cannot tell 2.5 from 5.
    """
    found = {}
    for group, bundle in (getattr(indata, "auxiliary", None) or {}).items():
        if not isinstance(bundle, dict):
            continue
        for name in bundle.get("gen_axes", []) or []:
            name = str(name)
            edges = bundle.get(f"edges__{name}")
            if name in ABSY_AXIS_NAMES and edges is not None:
                found[f"auxiliary[{group}].{name}"] = float(
                    np.max(np.abs(np.asarray(edges, dtype=np.float64)))
                )
    for channel, info in (getattr(indata, "channel_info", None) or {}).items():
        for axis in info.get("axes", []) or []:
            name = str(getattr(axis, "name", ""))
            if name in ABSY_AXIS_NAMES:
                found[f"channel[{channel}].{name}"] = float(
                    np.max(np.abs(np.asarray(axis.edges, dtype=np.float64)))
                )
    if not found:
        raise ValueError(
            "np_damping_wall: the card exposes no gen rapidity axis "
            f"(looked for {list(ABSY_AXIS_NAMES)} in the response auxiliary's "
            "gen_axes and in the fit channels' axes), so the binding |Y| for "
            "the f^NP walls is unknown. The reco channel's own yll is NOT a "
            "substitute -- it is the dilepton rapidity, not the boson gen "
            "rapidity the NP model's Y argument is. Pass ymax=<float> to state "
            "it, and say why in the fit's notes."
        )
    values = sorted(set(round(v, 12) for v in found.values()))
    if len(values) > 1:
        raise ValueError(
            "np_damping_wall: the card's gen rapidity axes disagree on their "
            f"reach: {found}. One of them is not the grid the model evaluates, "
            "so pass ymax=<float> explicitly."
        )
    return values[0], found


# --- theta -> physical -------------------------------------------------------
#
# params.REPARAM holds only the WIDTH for kind "unit"; the offset is the anchor,
# which the model fills from the correction so that theta = 0 reproduces it
# exactly (param_model._register_params). We therefore resolve the two halves
# from their two sources and store the result as one quadratic, exactly as the
# model does -- (anchor, width, 0) IS the linear map.


def physical_spec(cfg, name):
    """``(kind, coeffs)`` mapping this parameter's theta to its physical value.

    Kinds mirror ``params.REPARAM`` plus the identity branch:
      ``None``   value = theta            (no REPARAM entry: lambda_inf,
                                           lambda_inf_nu -- fitted, if ever,
                                           in physical units)
      ``"quad"`` value = c0 + c1*theta + c2*theta^2   ("unit" resolves to this
                                           with (anchor, width, 0))
      ``"log"``  value = exp(theta*L)

    Raises on an unknown kind so that a new ``params.REPARAM`` kind breaks here
    loudly instead of being silently treated as the identity -- which would put
    the wall on the wrong coordinate, the exact bug this port exists to avoid.
    """
    spec = adp.reparam(name)
    if spec is None:
        return (None, ())
    kind, coeffs = spec
    if kind == "log":
        return ("log", tuple(float(c) for c in coeffs))
    if kind == "quad":
        return ("quad", tuple(float(c) for c in coeffs))
    if kind == "unit":
        (width,) = coeffs
        anchor = adp.corr_anchor_value(cfg, name)
        if anchor is None:
            raise ValueError(
                f"np_damping_wall: {name} is a unit nuisance (physical = anchor "
                f"+ {float(width)} * theta) but the recorded correction runcard "
                f"carries no anchor for it "
                f"({'.'.join(str(p) for p in adp.corr_anchor_key(name)[:2])}). "
                "The wall would then be walling theta, not the physical lambda."
            )
        return ("quad", (float(anchor), float(width), 0.0))
    raise ValueError(
        f"np_damping_wall: params.REPARAM kind {kind!r} for {name} is not "
        "implemented here. Add it (and keep it identical to "
        "param_model._physical), or the wall silently constrains the wrong "
        "coordinate."
    )


def physical_from_theta(spec, theta, exp=np.exp):
    """Apply a :func:`physical_spec` map. Works on numpy scalars or TF tensors."""
    kind, coeffs = spec
    if kind is None:
        return theta
    if kind == "log":
        (length,) = coeffs
        return exp(theta * length)
    c0, c1, c2 = coeffs
    return c0 + c1 * theta + c2 * theta * theta


# --- the conditions ---------------------------------------------------------


class Condition:
    """One damping condition, ``coeff(values) >= bound``.

    ``names`` are the physical lambdas ``coeff`` reads, which is what lets the
    caller tell a condition the fit can move from one fixed by HELD lambdas.
    ``coeff`` is written with ``+ - *`` and ``relu2`` only, so the same object
    evaluates in numpy (diagnostics, held-lambda checks) and in TF (the
    penalty).
    """

    def __init__(self, label, names, coeff, bound):
        self.label = label
        self.names = tuple(names)
        self.coeff = coeff
        self.bound = bound

    def value(self, values, relu2):
        return self.coeff(values, relu2)

    def penalty(self, values, relu2):
        return relu2(self.bound - self.value(values, relu2))

    def __repr__(self):
        return f"Condition({self.label!r} >= {self.bound:g})"


def damping_conditions(
    np_model,
    np_model_nu,
    ymax,
    smallb=True,
    margin=NP_DAMPING_MARGIN,
    floor=LAMBDA_INF_FLOOR,
):
    """Every damping condition for the two resolved forms; see the module docstring.

    ``ymax`` is the binding |Y| (:func:`_binding_absY`). The TMD conditions are
    emitted twice, at Y = 0 and Y = ymax, because L2 is monotonic in Y^2 and so
    the binding rapidity is one extreme or the other depending on the sign of
    delta_lambda2 -- which the fit is free to flip.
    """
    conds = []

    # ---- CS side: P(u) = l2nu*u + l4nu*u^2 + l6nu*u^3 >= 0 for all u >= 0.
    conds.append(
        Condition(
            "lambda_inf_nu > 0 (CS saturation scale)",
            ("lambda_inf_nu",),
            lambda v, r: v["lambda_inf_nu"],
            floor,
        )
    )
    if np_model_nu == "tanh_2":
        conds.append(
            Condition(
                "lambda4_nu >= 0 (CS large-b leading)",
                ("lambda4_nu",),
                lambda v, r: v["lambda4_nu"],
                margin,
            )
        )
    else:  # tanh_6 -- lambda6_nu exists only in this vocabulary
        conds.append(
            Condition(
                "lambda6_nu >= 0 (CS large-b leading)",
                ("lambda6_nu",),
                lambda v, r: v["lambda6_nu"],
                margin,
            )
        )
        # Interior: lambda4_nu >= 0 OR lambda4_nu^2 <= 4*l2nu*l6nu. Self-gating
        # -- relu2(-l4nu) is 0 for l4nu >= 0, so a positive l4nu never triggers
        # this and there is no division by l6nu. No margin on a discriminant.
        conds.append(
            Condition(
                "4*lambda2_nu*lambda6_nu - relu(-lambda4_nu)^2 >= 0 (CS interior)",
                ("lambda2_nu", "lambda4_nu", "lambda6_nu"),
                lambda v, r: 4.0 * v["lambda2_nu"] * v["lambda6_nu"]
                - r(-v["lambda4_nu"]),
                0.0,
            )
        )
    if smallb:
        conds.append(
            Condition(
                "lambda2_nu >= 0 (CS small-b turn-on)",
                ("lambda2_nu",),
                lambda v, r: v["lambda2_nu"],
                margin,
            )
        )

    # ---- TMD side: Q(u) = L2 + B*u + lambda6*u^2 >= 0 for all u >= 0, at the
    # binding |Y|. Multiplied through by 3*lambda_inf^2 > 0 everywhere, so
    # "cubic" below is 3*lambda_inf^2*B = 3*lambda_inf^2*lambda4 + L2^3 and
    # every condition stays division-free.
    conds.append(
        Condition(
            "lambda_inf > 0 (TMD saturation scale)",
            ("lambda_inf",),
            lambda v, r: v["lambda_inf"],
            floor,
        )
    )

    def _l2Y(v, y_sq):
        return v["lambda2"] + v["delta_lambda2"] * y_sq

    def _cubic(v, y_sq):
        linf2 = v["lambda_inf"] * v["lambda_inf"]
        return 3.0 * linf2 * v["lambda4"] + _l2Y(v, y_sq) ** 3

    for y in (0.0, float(ymax)):
        y_sq = y * y
        tag = f"|Y|={y:g}"
        if smallb:
            conds.append(
                Condition(
                    f"lambda2 + delta_lambda2*Y^2 >= 0 at {tag} (TMD small-b turn-on)",
                    ("lambda2", "delta_lambda2"),
                    lambda v, r, y_sq=y_sq: _l2Y(v, y_sq),
                    margin,
                )
            )
        if np_model == "tanh_2":
            conds.append(
                Condition(
                    f"3*lambda_inf^2*lambda4 + L2^3 >= 0 at {tag} (TMD large-b)",
                    ("lambda_inf", "lambda2", "lambda4", "delta_lambda2"),
                    lambda v, r, y_sq=y_sq: _cubic(v, y_sq),
                    margin,
                )
            )
        else:  # tanh_6
            conds.append(
                Condition(
                    "lambda6 >= 0 (TMD large-b leading)",
                    ("lambda6",),
                    lambda v, r: v["lambda6"],
                    margin,
                )
            )
            # Interior: B >= 0 OR B^2 <= 4*L2*lambda6. In cubic space:
            # cubic >= 0 OR cubic^2 <= 36*lambda_inf^4*L2*lambda6. Self-gating
            # and division-free, as on the CS side.
            conds.append(
                Condition(
                    f"36*lambda_inf^4*L2*lambda6 - relu(-cubic)^2 >= 0 at {tag} "
                    "(TMD interior)",
                    (
                        "lambda_inf",
                        "lambda2",
                        "lambda4",
                        "delta_lambda2",
                        "lambda6",
                    ),
                    lambda v, r, y_sq=y_sq: 36.0
                    * v["lambda_inf"] ** 4
                    * _l2Y(v, y_sq)
                    * v["lambda6"]
                    - r(-_cubic(v, y_sq)),
                    0.0,
                )
            )
    return conds


def numpy_relu2(x):
    """``relu(x)^2`` on numpy scalars -- the diagnostic / held-lambda evaluator."""
    return np.maximum(0.0, x) ** 2


def resolve_wall_inputs(indata, ymax=None):
    """Everything the wall needs that comes from the card, with no TF involved.

    Returns a dict with the two resolved forms, the binding |Y| and where it
    came from, the required lambda names, and per-lambda
    ``(physical_spec, anchor)``. Shared with the offline verification scripts so
    that what they check is literally what the fit imposes.
    """
    entry = corr_config_from_meta(getattr(indata, "metadata", None) or {})
    if entry is None:
        raise ValueError(
            "np_damping_wall: the card records no theory-correction config "
            f"({CORR_CONFIG_META_KEY}), so neither the NP functional form nor "
            "the lambda anchors are known -- and without the anchor the fitted "
            "theta cannot be turned into a physical lambda at all. This is the "
            "same refusal SCETlibADParamModel makes for the anchor; fix it the "
            "same way (rerun the histmaker with a WRemnants that records it)."
        )
    cfg = entry["config"]
    np_model, np_model_nu = forms_from_corr_config(cfg)
    if ymax is None:
        y, y_source = _binding_absY(indata)
    else:
        y, y_source = float(ymax), {"ymax= (explicit override)": float(ymax)}
    names = tuple(dict.fromkeys(_TMD_LAMBDAS[np_model] + _CS_LAMBDAS[np_model_nu]))
    return dict(
        corr_tag=entry.get("tag"),
        np_model=np_model,
        np_model_nu=np_model_nu,
        ymax=y,
        ymax_source=y_source,
        names=names,
        specs={n: physical_spec(cfg, n) for n in names},
        anchors={n: adp.corr_anchor_value(cfg, n) for n in names},
    )


def _make_mapping_class():
    from rabbit.mappings.mapping import BaseMapping

    class NPDampingMapping(BaseMapping):
        """Vestigial BaseMapping carrying the wall's options to the regularizer.

        Options (``key=value`` tokens, both optional):
            smallb=<0|1>   enforce the small-b turn-on walls lambda2_nu >= 0 and
                           L2 >= 0 (default 1). ``smallb=0`` drops them, keeping
                           only the large-b limit / interior walls and the
                           lambda_inf floors -- i.e. constrain the limiting
                           behaviour but let the leading small-b coefficient
                           float either sign.
            ymax=<float>   override the binding |Y| for the f^NP walls. Default:
                           DERIVED from the card's own gen rapidity axis, which
                           is the range the model evaluates sigma_gen over. Only
                           override for a deliberate study, and record why -- the
                           delta_lambda2 wall scales as 1/ymax^2.

        The NP forms and the lambda anchors are derived from the card's recorded
        theory correction (see the module docstring); nothing on the -r line
        repeats the ``--paramModel`` spec.
        """

        def __init__(self, indata, key, smallb=True, ymax=None):
            super().__init__(indata, key)
            self.indata = indata
            self.smallb = bool(smallb)
            self.ymax = None if ymax is None else float(ymax)

        @classmethod
        def parse_args(cls, indata, *args):
            smallb, ymax = True, None
            for a in args:
                if "=" not in a:
                    raise ValueError(
                        f"NPDampingMapping: args are 'smallb=<0|1>' and "
                        f"'ymax=<float>', got '{a}'"
                    )
                k, v = a.split("=", 1)
                k = k.strip()
                if k == "smallb":
                    smallb = v.strip().lower() not in ("0", "false", "no", "off")
                elif k == "ymax":
                    ymax = float(v)
                else:
                    raise ValueError(
                        f"NPDampingMapping: unknown key '{k}'; only 'smallb' and "
                        "'ymax' are supported (the lambda_inf floor and the "
                        "damping margin are fixed module constants)."
                    )
            key = f"{cls.__name__} smallb={int(smallb)}" + (
                f" ymax={ymax:g}" if ymax is not None else ""
            )
            return cls(indata, key, smallb=smallb, ymax=ymax)

    return NPDampingMapping


def _make_regularizer_class():
    import tensorflow as tf

    from rabbit.regularization.regularizer import Regularizer

    class NPDampingWall(Regularizer):
        """Hinge-loss penalty enforcing NP damping, per side and per form
        (tanh_2 / tanh_6), on the PHYSICAL lambdas; see the module docstring."""

        def __init__(self, mapping, dtype):
            super().__init__(mapping, dtype)
            self.dtype = dtype
            self.mapping = mapping
            self.indata = mapping.indata
            self.enforce_small_b = bool(getattr(mapping, "smallb", True))
            self.margin = NP_DAMPING_MARGIN
            self.floor = LAMBDA_INF_FLOOR

            self.inputs = resolve_wall_inputs(
                self.indata, ymax=getattr(mapping, "ymax", None)
            )
            self.conditions = damping_conditions(
                self.inputs["np_model"],
                self.inputs["np_model_nu"],
                self.inputs["ymax"],
                smallb=self.enforce_small_b,
                margin=self.margin,
                floor=self.floor,
            )
            print(
                "[NPDampingWall] forms from the theory correction "
                f"{self.inputs['corr_tag']!r}: np_model="
                f"{self.inputs['np_model']}, np_model_nu="
                f"{self.inputs['np_model_nu']}. Binding |Y| = "
                f"{self.inputs['ymax']:g} from {self.inputs['ymax_source']}. "
                f"smallb={int(self.enforce_small_b)}, "
                f"margin={self.margin:g}, lambda_inf floor={self.floor:g}. "
                f"{len(self.conditions)} condition(s).",
                flush=True,
            )
            # theta -> physical, printed in full: this is the mapping the whole
            # port hinges on, and the only place a reader can check it against
            # the model's own anchor line.
            print(
                "[NPDampingWall] theta -> physical: "
                + ", ".join(
                    f"{n} = "
                    + (
                        "theta"
                        if self.inputs["specs"][n][0] is None
                        else (
                            f"{self.inputs['specs'][n][1][0]:g} + "
                            f"{self.inputs['specs'][n][1][1]:g}*theta"
                            if self.inputs["specs"][n][0] == "quad"
                            else str(self.inputs["specs"][n])
                        )
                    )
                    for n in self.inputs["names"]
                ),
                flush=True,
            )

            # Resolved per parameter layout, not here: the fitter can swap its
            # ParamModel mid-session (the saturated GoF path wraps it in a
            # CompositeParamModel, which reorders and resizes x), so positions
            # are re-resolved by NAME in set_expectations on every arm.
            self._idx = {}
            self._held = {}
            self._active = []

            self._cast = lambda v: tf.constant(float(v), dtype=self.dtype)
            self._relu2_tf = lambda x: tf.square(
                tf.maximum(tf.constant(0.0, dtype=self.dtype), x)
            )

        def set_expectations(self, initial_params, initial_observables, parms=None):
            names = np.asarray(parms).astype(str) if parms is not None else None
            if names is None:
                raise ValueError(
                    "NPDampingWall needs the fit's parameter NAMES to resolve "
                    "the lambdas (rabbit passes parms= to set_expectations). "
                    "Positional indexing is deliberately not used: the fitted "
                    "block is [POI | POU] in the model's own order and the "
                    "saturated path concatenates a second model's parameters "
                    "into it."
                )
            index = {n: i for i, n in enumerate(names)}
            self._idx, self._held = {}, {}
            for n in self.inputs["names"]:
                if n in index:
                    self._idx[n] = index[n]
                else:
                    # Not fitted: held at the correction's anchor for the whole
                    # fit (params.DEFAULT_FROZEN freezes lambda_inf,
                    # lambda_inf_nu, so this is the NORMAL case for those two).
                    anchor = self.inputs["anchors"][n]
                    if anchor is None:
                        raise ValueError(
                            f"NPDampingWall: {n} is required by the fit's NP "
                            f"forms (np_model={self.inputs['np_model']!r}, "
                            f"np_model_nu={self.inputs['np_model_nu']!r}) but is "
                            "neither a fit parameter nor carried by the recorded "
                            "correction runcard, so the wall has no value for "
                            "it at all."
                        )
                    self._held[n] = float(anchor)

            # A condition reading only HELD lambdas is a CONSTANT. Adding it to
            # the loss would be worse than useless: it cannot be minimised, and
            # a nonzero constant offsets the free-vs-walled Delta(loss) that is
            # the point of running this. So check it once, here, against the
            # BARE condition (no margin -- a lambda deliberately frozen exactly
            # ON the boundary, as the physical-lambda study freezes
            # lambda4_nu = 0, is a legitimate physical choice), and drop it.
            self._active = []
            dropped = []
            for cond in self.conditions:
                if any(n in self._idx for n in cond.names):
                    self._active.append(cond)
                    continue
                val = float(cond.value(self._held, numpy_relu2))
                if val < 0.0:
                    at = {n: self._held[n] for n in cond.names}
                    raise ValueError(
                        f"NPDampingWall: the condition '{cond.label}' depends "
                        f"only on lambdas HELD out of the fit ({at}) and is "
                        f"VIOLATED there ({val:.6g} < 0). No fit can repair it, "
                        "so the walled fit would minimise a constant-offset "
                        "loss around an unphysical NP function. Fix the held "
                        "values (they come from the theory correction's runcard) "
                        "or float the parameter."
                    )
                dropped.append((cond.label, val))
            print(
                f"[NPDampingWall] armed on {len(self._active)} of "
                f"{len(self.conditions)} condition(s); fitted lambdas "
                f"{sorted(self._idx)}, held at the correction's anchor "
                f"{self._held}."
                + (
                    "\n[NPDampingWall] dropped as constant (held lambdas only, "
                    "condition satisfied): "
                    + "; ".join(f"{lab} = {val:.6g}" for lab, val in dropped)
                    if dropped
                    else ""
                ),
                flush=True,
            )

        def _physical(self, params):
            """``{name: physical lambda}`` from the fitter's x, as TF tensors."""
            values = {n: self._cast(v) for n, v in self._held.items()}
            for n, i in self._idx.items():
                values[n] = physical_from_theta(
                    self.inputs["specs"][n], params[i], exp=tf.exp
                )
            return values

        def compute_nll_penalty(self, params, observables):
            values = self._physical(params)
            return tf.add_n([c.penalty(values, self._relu2_tf) for c in self._active])

    return NPDampingWall


# PEP-562 lazy class resolution: rabbit's loader does
#     module = importlib.import_module(...); cls = getattr(module, class_name)
# so the classes are synthesised on first attribute access, keeping the module
# importable without rabbit / TF (matches scetlib_np/np_damping_wall.py).
def __getattr__(name):
    if name == "NPDampingMapping":
        cls = _make_mapping_class()
        globals()["NPDampingMapping"] = cls
        return cls
    if name == "NPDampingWall":
        cls = _make_regularizer_class()
        globals()["NPDampingWall"] = cls
        return cls
    raise AttributeError(name)
