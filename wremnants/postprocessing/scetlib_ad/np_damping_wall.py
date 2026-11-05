"""Physical-damping wall on the NP lambdas of the SCETlib-AD param model.

A wrong-sign lambda anti-damps the NP form factors (gamma_nu^NP > 0, f^NP growing
with b_T), so sigma(qT) can go negative while the fitted yields stay finite. The
wall keeps the PHYSICAL lambdas in the damping region. The fitted parameters are
unit nuisances, so the wall maps theta -> anchor + width*theta itself (width from
``params.REPARAM``, anchor from the card's recorded theory-correction runcard),
reads the NP forms from the same runcard, and finds the lambdas by name in the
fit's ``parms``. Unfitted lambdas are held at the correction's anchor, so the
model's ``anchor_source=cache`` / ``anchor_override=`` are not seen by the wall.

Damping = the tanh argument is >= 0 for all b_T, i.e. a polynomial in u = b_T^2 is
non-negative on u >= 0 (both forms are evaluated at the raw b_T in SCETlib):

  CS   gamma_nu^NP = -lambda_inf_nu*tanh(P(u)/lambda_inf_nu),
       P(u) = lambda2_nu*u + lambda4_nu*u^2 + lambda6_nu*u^3     (tanh_2: no u^3)
       lambda_inf_nu > 0, lambda2_nu >= 0, leading coefficient >= 0, and for
       tanh_6 the interior 4*lambda2_nu*lambda6_nu - relu(-lambda4_nu)^2 >= 0
  TMD  f^NP = exp(-2*lambda_inf*b*tanh(a)),  a*lambda_inf = b*Q(u),
       Q(u) = L2 + B*u + lambda6*u^2,  L2 = lambda2 + delta_lambda2*Y^2,
       B = lambda4 + L2^3/(3*lambda_inf^2)  (the arctanh term of np_effective)
       lambda_inf > 0, L2 >= 0, tanh_2: cubic = 3*lambda_inf^2*lambda4 + L2^3 >= 0,
       tanh_6: lambda6 >= 0 and 36*lambda_inf^4*L2*lambda6 - relu(-cubic)^2 >= 0,
       each at Y = 0 and Y = ymax (L2 is monotonic in Y^2).

Each condition is ``coeff >= bound`` with penalty P(bound - coeff), times
exp(2*tau) from ``--regularizationStrength tau``. P is by default (``smooth=c2``)
the C^2 ramp of width d > 0, ``smooth=relu2`` gives P = relu(x)^2:

    P(x) = 0 (x <= 0),  x^3/(3d) (0 < x < d),  x^2 - d*x + d^2/3 (x >= d)

The continuous curvature avoids a trust-krylov limit cycle at an active face, at
the cost of at most ~d/2 extra overshoot. d = delta/scale, with scale the NP
exponent per unit of coeff at b_T = bmax. The tanh_6 interior discriminants have
no constant scale: they keep relu^2, or raise if ``smooth=c2`` is given explicitly.

Options (``key=value``): ``smallb=<0|1>`` the small-b walls lambda2_nu >= 0 and
L2 >= 0 (default 1); ``ymax=<float>`` the binding |Y| (default: the card's gen |Y|
reach); ``margin=<float>`` the bound of every condition except the lambda_inf
floors and the discriminants (default 0, the exact boundary: pair it with a stiff
strength); ``smooth=<c2|relu2>``; ``delta=<float>`` (1e-3) and ``bmax=<float>``
(12.6 GeV^-1), C^2 only. Usage, as a penalty or as hard trust-constr constraints:

    rabbit_fit.py ... --regularizationStrength 8 \\
      -r wremnants.postprocessing.scetlib_ad.np_damping_wall.NPDampingWall \\
         wremnants.postprocessing.scetlib_ad.np_damping_wall.NPDampingMapping \\
         [key=value ...]
    rabbit_fit.py ... --minimizerMethod trust-constr --constraints \\
      wremnants.postprocessing.scetlib_ad.np_damping_wall.NPDomainConstraints margin=0
"""

import numpy as np

# rabbit / TF imports are deferred to the lazy class factories at the bottom, so
# damping_conditions() and the theta -> physical map import without them.
from wremnants.postprocessing.scetlib_ad import params as adp
from wremnants.postprocessing.scetlib_ad.response import (
    CORR_CONFIG_META_KEY,
    corr_config_from_meta,
)

# Forms with damping conditions; anything else raises rather than guessing.
SUPPORTED_FORMS = ("tanh_2", "tanh_6")

# SCETlib backward-compatibility aliases that resolve to a supported form.
_FORM_ALIASES = {"hyp_tangent": "tanh_2"}

# The lambdas each side's conditions read, per form (the names used by
# damping_conditions()).
_CS_LAMBDAS = {
    "tanh_2": ("lambda_inf_nu", "lambda2_nu", "lambda4_nu"),
    "tanh_6": ("lambda_inf_nu", "lambda2_nu", "lambda4_nu", "lambda6_nu"),
}
_TMD_LAMBDAS = {
    "tanh_2": ("lambda_inf", "lambda2", "lambda4", "delta_lambda2"),
    "tanh_6": ("lambda_inf", "lambda2", "lambda4", "delta_lambda2", "lambda6"),
}

# Gen |Y| axis names searched for the binding |Y|, in the response auxiliary
# (reco fits) or the fit channel's axes (gen_level=1).
ABSY_AXIS_NAMES = ("absYVGen", "absYVgenSig", "yVGen")

# Values of SCETlib's separate TMD-PDF NP factor (np_model_tmd) accepted as off;
# the TMD conditions below do not cover it.
_NP_MODEL_TMD_OFF = ("off", "none", "")

LAMBDA_INF_FLOOR = 1e-3  # positive floor on the lambda_inf saturation scales
NP_DAMPING_MARGIN = 0.0  # default bound: each damping coeff >= this

SMOOTH_MODES = ("relu2", "c2")
NP_WALL_SMOOTH = "c2"
# C^2 ramp width, in units of the NP exponent at b_T = NP_WALL_BMAX
NP_WALL_C2_DELTA = 1e-3
# largest b_T [GeV^-1] of the AD cache's b_T grid
NP_WALL_BMAX = 12.6


def _parse_positive(value, what):
    """A finite float > 0, or raise."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"np_damping_wall: {what} must be a number, got {value!r}"
        ) from None
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(
            f"np_damping_wall: {what} must be finite and > 0, got {value!r}"
        )
    return out


def _parse_smooth(value):
    mode = str(value).strip().lower()
    if mode not in SMOOTH_MODES:
        raise ValueError(
            f"np_damping_wall: smooth must be one of {SMOOTH_MODES}, got {value!r}"
        )
    return mode


def c2_ramp(x, delta, maximum=np.maximum, minimum=np.minimum):
    """The C^2 penalty P(x) of the module docstring, branch-free.

    Written as xc^3/(3d) + y^2 + d*y with xc = clip(x, 0, d), y = relu(x - d):
    for x >= d that is d^2/3 + y^2 + d y = x^2 - d x + d^2/3. Pass
    ``tf.maximum`` / ``tf.minimum`` to evaluate it in TF (autodiff gives the
    exact piecewise first and second derivatives).
    """
    xc = minimum(maximum(x, 0.0), delta)
    y = maximum(x - delta, 0.0)
    return xc * xc * xc / (3.0 * delta) + y * y + delta * y


def _parse_margin(value):
    """A validated damping margin: a finite float >= 0, or raise."""
    try:
        margin = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"np_damping_wall: margin must be a number, got {value!r}"
        ) from None
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError(
            f"np_damping_wall: margin must be finite and >= 0, got {value!r}. "
            "A negative margin would wall a region PAST the damping boundary, "
            "i.e. admit anti-damping lambdas as 'physical'."
        )
    return margin


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

    ``params.CORR_REFUSE_KEYS`` refuses a cache whose NP forms differ from the
    correction's, so these are the forms the fit evaluates.
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
    """Largest |Y| the fit's gen grid reaches, from the card: the TMD walls' Y_max.

    The delta_lambda2 wall scales as 1/Y_max^2, so this is derived, not assumed.
    Read from the response auxiliary's ``gen_axes`` (reco fits) or the fit
    channel's own axes (``gen_level=1``). Returns ``(ymax, {source: value})``;
    raises if no gen |Y| axis is found or the sources disagree.
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
# params.REPARAM holds only the width for kind "unit"; the offset is the
# correction's anchor (as in param_model._register_params). Stored as the
# quadratic (anchor, width, 0).


def physical_spec(cfg, name):
    """``(kind, coeffs)`` mapping this parameter's theta to its physical value.

    Kinds mirror ``params.REPARAM`` plus the identity branch:
      ``None``   value = theta            (no REPARAM entry: lambda_inf,
                                           lambda_inf_nu -- fitted, if ever,
                                           in physical units)
      ``"quad"`` value = c0 + c1*theta + c2*theta^2   ("unit" resolves to this
                                           with (anchor, width, 0))
      ``"log"``  value = exp(theta*L)

    Raises on an unknown kind rather than treating it as the identity, which
    would put the wall on the wrong coordinate.
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

    def __init__(self, label, names, coeff, bound, scale=None):
        self.label = label
        self.names = tuple(names)
        self.coeff = coeff
        self.bound = bound
        # NP exponent per unit of coeff at b_T = bmax (None: no constant
        # scale, the tanh_6 discriminants); read by the C^2 ramp only.
        self.scale = scale

    def value(self, values, relu2):
        return self.coeff(values, relu2)

    def penalty(self, values, relu2, outer=None):
        """``outer(bound - coeff)``; ``outer`` defaults to ``relu2``."""
        return (relu2 if outer is None else outer)(
            self.bound - self.value(values, relu2)
        )

    def __repr__(self):
        return f"Condition({self.label!r} >= {self.bound:g})"


def damping_conditions(
    np_model,
    np_model_nu,
    ymax,
    smallb=True,
    margin=NP_DAMPING_MARGIN,
    floor=LAMBDA_INF_FLOOR,
    bmax=NP_WALL_BMAX,
    lambda_inf_ref=1.0,
):
    """Every damping condition for the two resolved forms; see the module docstring.

    Each condition carries its ``scale``, the NP exponent per unit of its coeff
    at b_T = ``bmax`` (CS: gamma_nu ~ -P(b^2); TMD: ln f^NP ~ -2 b^2 Q(b^2) in the
    unsaturated limit; the cubic is divided by 3 lambda_inf^2 at
    ``lambda_inf_ref``); ``scale=None`` for the tanh_6 interior discriminants.

    The TMD conditions are emitted at Y = 0 and Y = ``ymax`` (the binding |Y|,
    :func:`_binding_absY`), which covers either sign of delta_lambda2. ``margin``
    is the bound of every condition except the lambda_inf floors (``floor``) and
    the tanh_6 interior discriminants (0); callers validate it.
    """
    conds = []
    b2 = float(bmax) ** 2

    # ---- CS side: P(u) = l2nu*u + l4nu*u^2 + l6nu*u^3 >= 0 for all u >= 0.
    conds.append(
        Condition(
            "lambda_inf_nu > 0 (CS saturation scale)",
            ("lambda_inf_nu",),
            lambda v, r: v["lambda_inf_nu"],
            floor,
            scale=1.0,  # gamma_nu -> -lambda_inf_nu at large b
        )
    )
    if np_model_nu == "tanh_2":
        conds.append(
            Condition(
                "lambda4_nu >= 0 (CS large-b leading)",
                ("lambda4_nu",),
                lambda v, r: v["lambda4_nu"],
                margin,
                scale=b2 * b2,
            )
        )
    else:  # tanh_6 -- lambda6_nu exists only in this vocabulary
        conds.append(
            Condition(
                "lambda6_nu >= 0 (CS large-b leading)",
                ("lambda6_nu",),
                lambda v, r: v["lambda6_nu"],
                margin,
                scale=b2 * b2 * b2,
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
                scale=b2,
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
            scale=2.0 * float(bmax),  # ln f^NP -> -2 lambda_inf b at large b
        )
    )

    def _l2Y(v, y_sq):
        return v["lambda2"] + v["delta_lambda2"] * y_sq

    def _cubic(v, y_sq):
        linf2 = v["lambda_inf"] * v["lambda_inf"]
        return 3.0 * linf2 * v["lambda4"] + _l2Y(v, y_sq) ** 3

    if np_model == "tanh_6":  # Y-independent: once, not per |Y| endpoint
        conds.append(
            Condition(
                "lambda6 >= 0 (TMD large-b leading)",
                ("lambda6",),
                lambda v, r: v["lambda6"],
                margin,
                scale=2.0 * b2 * b2 * b2,
            )
        )

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
                    scale=2.0 * b2,
                )
            )
        if np_model == "tanh_2":
            conds.append(
                Condition(
                    f"3*lambda_inf^2*lambda4 + L2^3 >= 0 at {tag} (TMD large-b)",
                    ("lambda_inf", "lambda2", "lambda4", "delta_lambda2"),
                    lambda v, r, y_sq=y_sq: _cubic(v, y_sq),
                    margin,
                    scale=2.0 * b2 * b2 / (3.0 * float(lambda_inf_ref) ** 2),
                )
            )
        else:  # tanh_6
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
    ``(physical_spec, anchor)``.
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
        """Carries the wall's options to the regularizer (no mapping of its own).

        Options (``key=value``; see the module docstring): ``smallb=<0|1>``
        (default 1), ``ymax=<float>`` (default: the card's gen |Y| reach),
        ``margin=<float>`` (default 0, finite and >= 0), ``smooth=<c2|relu2>``
        (omitted: C^2, relu^2 on the no-scale conditions; explicit c2: those
        raise), ``delta=<float>`` (1e-3) and ``bmax=<float>`` (12.6), C^2 only.
        """

        def __init__(
            self,
            indata,
            key,
            smallb=True,
            ymax=None,
            margin=NP_DAMPING_MARGIN,
            smooth=None,
            delta=None,
            bmax=None,
        ):
            super().__init__(indata, key)
            self.indata = indata
            self.smallb = bool(smallb)
            self.ymax = None if ymax is None else float(ymax)
            self.margin = _parse_margin(margin)
            # smooth=None means "not given": NP_WALL_SMOOTH, with relu^2 kept on
            # no-scale conditions; an explicit smooth=c2 refuses them instead.
            self.smooth_explicit = smooth is not None
            self.smooth = _parse_smooth(NP_WALL_SMOOTH if smooth is None else smooth)
            if self.smooth != "c2" and (delta is not None or bmax is not None):
                raise ValueError(
                    "NPDampingMapping: delta= and bmax= set the C^2 ramp, which "
                    f"smooth={self.smooth} does not use; they would be ignored."
                )
            self.delta = (
                NP_WALL_C2_DELTA if delta is None else _parse_positive(delta, "delta")
            )
            self.bmax = NP_WALL_BMAX if bmax is None else _parse_positive(bmax, "bmax")

        @classmethod
        def parse_args(cls, indata, *args):
            smallb, ymax, margin = True, None, None
            smooth, delta, bmax = None, None, None
            for a in args:
                if "=" not in a:
                    raise ValueError(
                        f"NPDampingMapping: args are 'smallb=<0|1>', "
                        f"'ymax=<float>', 'margin=<float>', 'smooth=<relu2|c2>', "
                        f"'delta=<float>' and 'bmax=<float>', got '{a}'"
                    )
                k, v = a.split("=", 1)
                k = k.strip()
                if k == "smallb":
                    smallb = v.strip().lower() not in ("0", "false", "no", "off")
                elif k == "ymax":
                    ymax = float(v)
                elif k == "margin":
                    margin = _parse_margin(v)
                elif k == "smooth":
                    smooth = _parse_smooth(v)
                elif k == "delta":
                    delta = _parse_positive(v, "delta")
                elif k == "bmax":
                    bmax = _parse_positive(v, "bmax")
                else:
                    raise ValueError(
                        f"NPDampingMapping: unknown key '{k}'; only 'smallb', "
                        "'ymax', 'margin', 'smooth', 'delta' and 'bmax' are "
                        "supported (the lambda_inf floor is a fixed module "
                        "constant)."
                    )
            key = (
                f"{cls.__name__} smallb={int(smallb)}"
                + (f" ymax={ymax:g}" if ymax is not None else "")
                + (f" margin={margin:g}" if margin is not None else "")
                + (f" smooth={smooth}" if smooth is not None else "")
                + (f" delta={delta:g}" if delta is not None else "")
                + (f" bmax={bmax:g}" if bmax is not None else "")
            )
            return cls(
                indata,
                key,
                smallb=smallb,
                ymax=ymax,
                margin=NP_DAMPING_MARGIN if margin is None else margin,
                smooth=smooth,
                delta=delta,
                bmax=bmax,
            )

    return NPDampingMapping


def _make_regularizer_class():
    import tensorflow as tf

    from rabbit.regularization.regularizer import Regularizer

    class NPDampingWall(Regularizer):
        """Penalty enforcing NP damping on the physical lambdas (module docstring)."""

        # a penalty on the parameters alone: rabbit need not build the yields
        needs_observables = False

        def __init__(self, mapping, dtype):
            super().__init__(mapping, dtype)
            self.dtype = dtype
            self.mapping = mapping
            self.indata = mapping.indata
            self.enforce_small_b = bool(getattr(mapping, "smallb", True))
            self.margin = _parse_margin(getattr(mapping, "margin", NP_DAMPING_MARGIN))
            self.floor = LAMBDA_INF_FLOOR
            raw_smooth = getattr(mapping, "smooth", None)
            # a mapping with no smooth attribute at all is the default
            self.smooth_explicit = bool(
                getattr(mapping, "smooth_explicit", raw_smooth is not None)
            )
            self.smooth = _parse_smooth(
                NP_WALL_SMOOTH if raw_smooth is None else raw_smooth
            )
            self.delta = float(getattr(mapping, "delta", NP_WALL_C2_DELTA))
            self.bmax = float(getattr(mapping, "bmax", NP_WALL_BMAX))

            self.inputs = resolve_wall_inputs(
                self.indata, ymax=getattr(mapping, "ymax", None)
            )
            linf_ref = self.inputs["anchors"].get("lambda_inf")
            self.conditions = damping_conditions(
                self.inputs["np_model"],
                self.inputs["np_model_nu"],
                self.inputs["ymax"],
                smallb=self.enforce_small_b,
                margin=self.margin,
                floor=self.floor,
                bmax=self.bmax,
                lambda_inf_ref=1.0 if linf_ref is None else float(linf_ref),
            )
            print(
                "[NPDampingWall] forms from the theory correction "
                f"{self.inputs['corr_tag']!r}: np_model="
                f"{self.inputs['np_model']}, np_model_nu="
                f"{self.inputs['np_model_nu']}. Binding |Y| = "
                f"{self.inputs['ymax']:g} from {self.inputs['ymax_source']}. "
                f"smallb={int(self.enforce_small_b)}, "
                f"margin={self.margin:g}, lambda_inf floor={self.floor:g}, "
                f"smooth={self.smooth}"
                + ("" if self.smooth_explicit else " (default)")
                + (
                    f" (delta={self.delta:g} in the NP exponent at "
                    f"b_T={self.bmax:g} GeV^-1)"
                    if self.smooth == "c2"
                    else ""
                )
                + f". {len(self.conditions)} condition(s).",
                flush=True,
            )
            # theta -> physical, printed in full so it can be checked against the
            # model's own anchor line.
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

            # Positions are resolved by name in set_expectations on every arm:
            # the saturated GoF path wraps the model in a CompositeParamModel,
            # which reorders and resizes x.
            self._idx = {}
            self._held = {}
            self._active = []

            self._cast = lambda v: tf.constant(float(v), dtype=self.dtype)
            self._relu2_tf = lambda x: tf.square(
                tf.maximum(tf.constant(0.0, dtype=self.dtype), x)
            )
            # per-active-condition C^2 ramps, set in set_expectations (None: relu2)
            self._outer = []

        def _c2_outer(self, cond):
            """``(d, ramp)`` for one condition, d in raw units; raises if no scale.

            Only reached for a no-scale condition with an explicit smooth=c2.
            """
            if cond.scale is None:
                raise NotImplementedError(
                    f"NPDampingWall smooth=c2: the condition '{cond.label}' has "
                    "no constant physical scale (a tanh_6 interior "
                    "discriminant), so its ramp width cannot be set in physical "
                    "units. Omit smooth= to keep relu^2 on such conditions and "
                    "C^2 on the rest, or use smooth=relu2 for the whole wall."
                )
            d = float(self.delta / cond.scale)
            return d, lambda x: c2_ramp(x, d, maximum=tf.maximum, minimum=tf.minimum)

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
                    # Not fitted: held at the correction's anchor (the normal
                    # case for lambda_inf, lambda_inf_nu: params.DEFAULT_FROZEN).
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

            # A condition reading only held lambdas is a constant: check it once
            # against the bare condition (no margin, so a lambda frozen exactly
            # on the boundary, e.g. lambda4_nu = 0, passes) and drop it, so it
            # does not offset the loss.
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
            self._outer = [None] * len(self._active)
            if self.smooth == "c2":
                lines = []
                kept_relu2 = []
                for i, cond in enumerate(self._active):
                    if cond.scale is None and not self.smooth_explicit:
                        # no physical ramp width: keeps relu^2 (outer None)
                        kept_relu2.append(cond.label)
                        continue
                    d, fn = self._c2_outer(cond)
                    self._outer[i] = fn
                    lines.append(
                        f"{cond.label}: scale {cond.scale:.4g} per unit, ramp "
                        f"width {d:.3g} raw, extra overshoot <= d/2 = {d / 2:.3g} "
                        f"raw = {self.delta / 2:g} in the exponent at b_max"
                    )
                n_c2 = len(self._active) - len(kept_relu2)
                print(
                    f"[NPDampingWall] INFO smooth=c2"
                    f"{'' if self.smooth_explicit else ' (default)'}: C^2 ramp on "
                    f"{n_c2} of {len(self._active)} active condition(s); relu^2 on "
                    + (
                        f"{len(kept_relu2)} with no constant scale: "
                        + "; ".join(kept_relu2)
                        if kept_relu2
                        else "none"
                    )
                    + ".",
                    flush=True,
                )
                if lines:
                    print(
                        "[NPDampingWall] C^2 ramp per condition:\n  "
                        + "\n  ".join(lines),
                        flush=True,
                    )
            print(
                f"[NPDampingWall] armed on {len(self._active)} of "
                f"{len(self.conditions)} condition(s) at margin="
                f"{self.margin:g}; fitted lambdas "
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
            if not self._active:
                return tf.zeros([], dtype=self.dtype)
            values = self._physical(params)
            if self.smooth != "c2":
                return tf.add_n(
                    [c.penalty(values, self._relu2_tf) for c in self._active]
                )
            return tf.add_n(
                [
                    c.penalty(values, self._relu2_tf, outer=fn)
                    for c, fn in zip(self._active, self._outer)
                ]
            )

        def constraint_spec(self, params, observables):
            """The active conditions as ``(values, bounds)``, ``values >= bounds``.

            Not called by rabbit; :class:`NPDomainConstraints` builds its rows
            from it. The feasible set is exactly where the penalty is zero.
            """
            if not self._active:
                raise ValueError(
                    "NPDampingWall.constraint_spec: no active condition (not "
                    "armed yet, or every condition reads held lambdas only)"
                )
            values = self._physical(params)
            vals = tf.stack(
                [
                    tf.cast(c.value(values, self._relu2_tf), self.dtype)
                    for c in self._active
                ]
            )
            lbs = tf.constant([float(c.bound) for c in self._active], dtype=self.dtype)
            return vals, lbs

    return NPDampingWall


def _make_constraints_class():
    import tensorflow as tf

    from rabbit.constraints.constraints import Constraints

    class NPDomainConstraints(Constraints):
        """The physical-NP region as hard constraints, for rabbit's trust-constr.

        ``--minimizerMethod trust-constr --constraints <module>.NPDomainConstraints
        [smallb=<0|1>] [ymax=<float>] [margin=<float>]``. The same conditions as
        NPDampingWall (one is built and armed underneath and its
        ``constraint_spec`` supplies the rows). Use it instead of
        ``-r NPDampingWall``; the penalty-shape options (smooth, delta, bmax) are
        refused.
        """

        def __init__(self, indata, *args, **kwargs):
            super().__init__(indata)
            for a in args:
                key = a.split("=", 1)[0].strip()
                if key in ("smooth", "delta", "bmax"):
                    raise ValueError(
                        f"NPDomainConstraints: '{key}=' shapes the wall's PENALTY, "
                        "which hard constraints do not have; only smallb=, ymax= "
                        "and margin= apply."
                    )
            mapping = _lazy_class("NPDampingMapping").parse_args(indata, *args)
            self._wall = _lazy_class("NPDampingWall")(mapping, indata.dtype)

        def set_layout(self, parms):
            self._wall.set_expectations(None, None, parms=parms)
            active = self._wall._active
            self.labels = [c.label for c in active]
            self.lower = np.array([float(c.bound) for c in active], dtype=np.float64)
            self.upper = np.full(len(active), np.inf)

        def parameter_indices(self):
            # the fitted lambdas the conditions read (held ones are constants)
            return sorted(self._wall._idx.values())

        def values(self, x):
            if not self._wall._active:
                # every condition reads held lambdas only: nothing to constrain
                return tf.zeros([0], dtype=x.dtype)
            return self._wall.constraint_spec(x, None)[0]

    return NPDomainConstraints


def _lazy_class(name):
    """A lazily built class of this module, built once and cached."""
    cls = globals().get(name)
    return cls if cls is not None else __getattr__(name)


# PEP-562 lazy class resolution: rabbit's loader does
#     module = importlib.import_module(...); cls = getattr(module, class_name)
# so the classes are synthesised on first attribute access, keeping the module
# importable without rabbit / TF.
def __getattr__(name):
    if name == "NPDampingMapping":
        cls = _make_mapping_class()
        globals()["NPDampingMapping"] = cls
        return cls
    if name == "NPDampingWall":
        cls = _make_regularizer_class()
        globals()["NPDampingWall"] = cls
        return cls
    if name == "NPDomainConstraints":
        cls = _make_constraints_class()
        globals()["NPDomainConstraints"] = cls
        return cls
    raise AttributeError(name)
