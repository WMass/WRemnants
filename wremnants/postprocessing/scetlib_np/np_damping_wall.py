"""Physical-damping wall for the continuous-λ SCETlib NP param model.

Companion to ``wremnants/postprocessing/np_monotonicity.py``: same rabbit
``Regularizer`` mechanism and hinge-loss (relu²) wall structure, for the
:class:`~wremnants.postprocessing.scetlib_np.param_model.SCETlibNPParamModel`,
whose λ are direct fit parameters (the model-param block of ``x``), not discrete
template nuisances. No ``PARAM_MAP`` / θ-interpolation: the regularizer reads the
physical λ from ``x[:nparams]`` in the model's canonical parameter order.

The model FORMS and λ ORDER are DERIVED from the model, not re-declared here.
``SCETlibNPParamModel.__init__`` publishes itself on the shared ``indata``
(``indata.scetlib_np_param_model``); this regularizer — built afterward with the
same ``indata`` — reads ``model.fit_forms`` (the F_eff / γ_ν NUMERATOR forms the
fit integrates) and ``model._param_order`` (POI-reordering and all). So you pass
the form ONCE, to the model; nothing on the ``-r`` line repeats it, and a
``poi_params`` reorder is tracked automatically. The wall RAISES if no model has
registered, or if a side's form is one it has no walls for (see dispatch below).

Why a wall (see ``param_model.py``): a wrong-sign λ anti-damps the NP form
factors, so the bT-integral diverges / the differential σ(qT) oscillates negative
— a genuine unphysical region. The fit's qT→ptVGen rebin launders it by averaging
the negative differential σ away, so σ_gen / σ_reco / the NLL stay finite and the
minimizer cannot see the unphysical-ness. These λ must lie in the physical
(damping) region; this regularizer adds the fit-time penalty keeping them there.

Walls, per side and per FORM, read off the TF forms the fit integrates
(``btgrid_tf.gamma_nu_NP_tf`` / ``btgrid_tf.F_eff_tf``). The damping criterion is
"the tanh argument is ≥ 0 ∀b" (γ_ν ≤ 0 / F_eff decays), i.e. a polynomial in
u ≡ b² is non-negative on u ≥ 0:

  CS side   γ_ν^NP(b) = −λ∞_ν · tanh( P(u)/λ∞_ν ) ,
            P(u) = λ2_ν·u + λ4_ν·u² + λ6_ν·u³      (λ6_ν ≡ 0 for tanh_2)
      tanh_2:  λ∞_ν > 0 ,  λ2_ν ≥ 0 ,  λ4_ν ≥ 0
      tanh_6:  λ∞_ν > 0 ,  λ2_ν ≥ 0 ,  λ6_ν ≥ 0 (leading) ,
               and λ4_ν ≥ 0  OR  λ4_ν² ≤ 4·λ2_ν·λ6_ν   (interior; λ4_ν may dip
               negative while the b⁶ term keeps P ≥ 0)

  TMD side  F_eff(Y,b) = exp(−2·λ∞·b·tanh(a)) ,
            a·λ∞ = b·Q(u) ,  Q(u) = λ2_Y + B·u + λ6·u² ,
            B = λ4 + λ2_Y³/(3λ∞²) ,  λ2_Y = λ2 + δλ2·Y²    (λ6 ≡ 0 for tanh_2)
      tanh_2:  λ∞ > 0 ,  λ2_Y ≥ 0 ,  B ≥ 0  [≡ 3·λ∞²·λ4 + λ2_Y³ ≥ 0]
      tanh_6:  λ∞ > 0 ,  λ2_Y ≥ 0 ,  λ6 ≥ 0 (leading) ,
               and B ≥ 0  OR  B² ≤ 4·λ2_Y·λ6                (interior)
            evaluated at Y=0 and Y=Y_MAX (covers δλ2 of either sign — λ2_Y is
            monotonic in Y², so the binding |Y| is one of the two extremes).

The tanh_2 walls are EXACT in λ∞ (read off the actual TF forms, not the AN
λ∞-normalised parametrisation). They are the λ6 = 0 reduction of the tanh_6 walls
(then the leading coefficient is λ4_ν / B, hence the simpler ≥0 limit), and of the
``np_monotonicity.py`` monotonicity walls. NOTE this file uses the DAMPING
criterion (P ≥ 0), which lets the interior coefficient dip negative within the
discriminant bound; ``np_monotonicity.py`` uses the stricter MONOTONICITY
criterion (√3 in place of √4). The interior relu²(relu(−λ4_ν)² − 4·λ2_ν·λ6_ν)
penalty form is self-gating: it vanishes for λ4_ν ≥ 0 (no division by λ6).

Wall hardness is set at fit time by rabbit's ``--regularizationStrength`` (the
penalty × ``exp(2·tau)`` in ``fitter.py``; ``tau`` is a fixed multiplier, NOT a
minimised parameter). A large strength makes this a BARRIER: ≈0 inside the
physical region, steeply rising outside. Free (small strength) vs walled (large
strength) Δχ² is the data–model tension diagnostic: railing against a wall with
large Δχ² is genuine tension, not a masked pathology.

The small-b turn-on walls (λ2_ν ≥ 0, λ2_Y ≥ 0) are a stronger condition than the
large-b limit (they forbid an anti-damping bump near b→0, not just the wrong
asymptote). They can be switched off with the mapping flag ``smallb=0`` — then
ONLY the limiting/interior behaviour and the λ∞ floors are enforced, and the
leading b² coefficient floats either sign (use the postfit σ(qT)≥0 check in
``param_model_diagnostics`` as the real guard then).

Invoke (the model form is taken from ``--paramModel``; nothing repeats it here):

    rabbit_fit.py ... \\
      --regularizationStrength 3 \\
      -r wremnants.postprocessing.scetlib_np.np_damping_wall.NPDampingWall \\
         wremnants.postprocessing.scetlib_np.np_damping_wall.NPDampingMapping \\
         [smallb=0]

(``Y_MAX`` = 5 — the kinematic ceiling / btgrid Y reach — the λ∞ floor, and the
damping margin are fixed module constants, not -r options; both Y_MAX=5 and a
nonzero margin are UNDER TEST, edit the constants to change them.)

References:
  AN-25-085 theory.tex Eqs. eq:npgamma, eq:npf;
  param_model.py / sigma_gen.py docstrings (the σ_gen pipeline & the binning-
  launders-the-pathology discussion).
"""

# rabbit / TF imports deferred to the lazy class factories so the module stays
# importable without rabbit/TF (mirrors np_monotonicity.py).

# Forms this wall has damping conditions for. Anything else (frac_*, exp_*,
# tanh_1, tanh_4, signed_lambda, identity, …) raises rather than silently
# applying the wrong tanh_2/tanh_6 walls.
SUPPORTED_FORMS = ("tanh_2", "tanh_6")

# btgrid_tf form aliases that resolve to a SUPPORTED_FORMS entry (mirrors the
# alias maps in gamma_nu_NP_tf / F_eff_tf). Other aliases ("linear"->frac_1,
# "square_root"->frac_2) resolve to unsupported forms and so fall through to the
# raise. Applied to both sides; only "hyp_tangent" reaches a supported form.
_FORM_ALIASES = {"hyp_tangent": "tanh_2"}

# λ that the walls reference (always present in the model's _param_order, which
# is built from params.ALL_PARAMS); cross-checked at construction.
_REQUIRED_PARAMS = (
    "lambda2_nu",
    "lambda4_nu",
    "lambda6_nu",
    "lambda_inf_nu",
    "lambda2",
    "lambda4",
    "lambda6",
    "delta_lambda2",
    "lambda_inf",
)

# Fixed knobs (NOT CLI options — edit here to test). Only ``smallb`` is exposed
# on the -r line; these are deliberately constants to keep that line minimal.
Y_MAX = 5.0  # binding |y| for the F_eff Y-evaluation: the kinematic ceiling
#              ln(√s/Q) ≈ the btgrid Y reach (±5). UNDER TEST (was 2.5 = the |y|
#              acceptance); 5 demands physical damping over the full phase space.
LAMBDA_INF_FLOOR = 1e-3  # positive floor on the λ∞ saturation scales
NP_DAMPING_MARGIN = 5e-3  # positive cushion: enforce each damping coeff ≥ this,
#              not just ≥ 0, to keep the soft-wall equilibrium off the boundary.
#              UNDER TEST; default 0 (off) — set > 0 here to try the cushion.


def _make_mapping_class():
    from rabbit.mappings.mapping import BaseMapping

    class NPDampingMapping(BaseMapping):
        """Vestigial BaseMapping carrying the wall's option to the regularizer.

        Only option (``key=value`` token, optional):
            smallb=<0|1>   enforce the small-b turn-on walls λ2_ν≥0 and λ2_Y≥0
                           (default 1). smallb=0 drops them, keeping ONLY the
                           large-b limit/interior walls and the λ∞ floors — i.e.
                           constrain the limiting behaviour but let the leading
                           small-b coefficient float either sign.

        The model forms and λ order are derived from the registered
        ``SCETlibNPParamModel`` (see module docstring). The binding |Y| (``Y_MAX``),
        the λ∞ floor (``LAMBDA_INF_FLOOR``), and the damping cushion
        (``NP_DAMPING_MARGIN``) are FIXED module constants, not CLI options — edit
        them in this file to test, kept off the -r line on purpose.
        """

        def __init__(self, indata, key, smallb=True):
            super().__init__(indata, key)
            self.indata = indata
            self.smallb = bool(smallb)

        @classmethod
        def parse_args(cls, indata, *args):
            smallb = True
            for a in args:
                if "=" not in a:
                    raise ValueError(
                        f"NPDampingMapping: arg must be 'smallb=<0|1>', got '{a}'"
                    )
                k, v = a.split("=", 1)
                if k == "smallb":
                    smallb = v.strip().lower() not in ("0", "false", "no", "off")
                else:
                    raise ValueError(
                        f"NPDampingMapping: unknown key '{k}'; only 'smallb' is "
                        f"supported (ymax/eps/margin are fixed module constants)."
                    )
            return cls(indata, f"{cls.__name__} smallb={int(smallb)}", smallb=smallb)

    return NPDampingMapping


def _make_regularizer_class():
    import tensorflow as tf
    from rabbit.regularization.regularizer import Regularizer

    class NPDampingWall(Regularizer):
        """Hinge-loss penalty enforcing NP damping, per-side and per-form
        (tanh_2 / tanh_6); see the module docstring."""

        def __init__(self, mapping, dtype):
            super().__init__(mapping, dtype)
            self.dtype = dtype
            self.mapping = mapping
            self.indata = mapping.indata
            # ymax / eps / margin are fixed module constants (not CLI options);
            # only smallb comes from the mapping.
            self.ymax = Y_MAX
            self.eps = LAMBDA_INF_FLOOR
            self.margin = NP_DAMPING_MARGIN
            self.enforce_small_b = bool(getattr(mapping, "smallb", True))

            # Forms + λ order are DERIVED from the SCETlibNPParamModel that
            # published itself on the shared indata (built before this regularizer
            # in rabbit_fit; see param_model.py). No -r-line repetition.
            model = getattr(self.indata, "scetlib_np_param_model", None)
            if model is None:
                raise ValueError(
                    "NPDampingWall: no SCETlibNPParamModel registered on indata. "
                    "This wall derives the NP form and λ order from the param "
                    "model, so the fit must use "
                    "--paramModel ...scetlib_np.param_model.SCETlibNPParamModel "
                    "(which publishes indata.scetlib_np_param_model)."
                )

            self._order = tuple(model._param_order)
            self._pidx = {name: i for i, name in enumerate(self._order)}
            missing = [p for p in _REQUIRED_PARAMS if p not in self._pidx]
            if missing:
                raise ValueError(
                    f"NPDampingWall: model param order {self._order} is missing "
                    f"required λ {missing}."
                )

            # FIT (numerator) forms — the ones the fit integrates, which the wall
            # must constrain (NOT the card/denominator form). Resolve aliases and
            # fail on any form we have no walls for.
            forms = model.fit_forms
            self._np_model_nu = self._resolve_form(
                forms["np_model_nu"], side="CS (γ_ν, np_model_nu)"
            )
            self._np_model = self._resolve_form(
                forms["np_model"], side="TMD (F_eff, np_model)"
            )

            self._cast = lambda v: tf.constant(v, dtype=self.dtype)
            # Model-param block is x[:nparams]; nparams resolved at set_expectations.
            self._nparams = None

        @staticmethod
        def _resolve_form(form, side):
            resolved = _FORM_ALIASES.get(form, form)
            if resolved not in SUPPORTED_FORMS:
                raise NotImplementedError(
                    f"NPDampingWall: no damping walls for {side} form {form!r}"
                    + (f" (resolves to {resolved!r})" if resolved != form else "")
                    + f"; supported: {sorted(SUPPORTED_FORMS)}. Add walls for it "
                    "or run that side with a supported form."
                )
            return resolved

        def set_expectations(self, initial_params, initial_observables):
            nsyst = len(self.indata.systs)
            self._nparams = int(initial_params.shape[0]) - nsyst
            if self._nparams != len(self._order):
                raise ValueError(
                    f"NPDampingWall: the fit's model-param block is {self._nparams} "
                    f"wide but the model param order has {len(self._order)} entries "
                    f"{self._order}. A wrapping/composite param model (e.g. the "
                    "saturated goodness-of-fit path) reorders/resizes the block in "
                    "a way this wall's flat indexing cannot follow."
                )

        def _lam(self, params, name):
            # λ stored directly in the model-param block (allowNegativeParam=True),
            # so x[index] IS the physical λ — no theta interpolation.
            return params[self._pidx[name]]

        def compute_nll_penalty(self, params, observables):
            zero = self._cast(0.0)
            eps = self._cast(self.eps)
            m = self._cast(self.margin)  # positive cushion: enforce coeff ≥ margin
            three = self._cast(3.0)
            four = self._cast(4.0)
            thirtysix = self._cast(36.0)

            def relu2(x):  # hinge: 0 if x ≤ 0 else x²
                return tf.square(tf.maximum(zero, x))

            # Each damping condition "coeff ≥ 0" is enforced as "coeff ≥ margin"
            # (margin=0 → bare ≥0): relu2(margin - coeff). The soft wall's gradient
            # vanishes at the knee, so a weakly-constrained coeff settles a hair
            # past it; the margin moves the knee so the equilibrium stays damping.
            def wall(coeff):  # coeff ≥ margin
                return relu2(m - coeff)

            # ---- CS-side γ_ν^NP damping: P(u)=λ2_ν·u+λ4_ν·u²+λ6_ν·u³ ≥ 0 ∀u≥0.
            l2nu = self._lam(params, "lambda2_nu")
            l4nu = self._lam(params, "lambda4_nu")
            l6nu = self._lam(params, "lambda6_nu")
            linfnu = self._lam(params, "lambda_inf_nu")
            pens = [relu2(eps - linfnu)]  # λ∞_ν > 0 (saturation-scale regime)
            if self._np_model_nu == "tanh_2":
                pens.append(wall(l4nu))  # λ4_ν ≥ margin  (large-b leading)
            else:  # tanh_6
                pens.append(wall(l6nu))  # λ6_ν ≥ margin  (large-b leading)
                # interior: λ4_ν ≥ 0 OR λ4_ν² ≤ 4·λ2_ν·λ6_ν. Self-gating — the
                # relu2(-l4nu) vanishes for λ4_ν ≥ 0, so no penalty there; and no
                # division by λ6_ν. (No margin on the interior discriminant.)
                pens.append(relu2(relu2(-l4nu) - four * l2nu * l6nu))
            if self.enforce_small_b:
                pens.append(wall(l2nu))  # λ2_ν ≥ margin  (small-b turn-on)

            # ---- TMD-side F_eff damping: Q(u)=λ2_Y+B·u+λ6·u² ≥ 0 ∀u≥0, evaluated
            # at the binding |Y| extremes. cubic ≡ 3·λ∞²·B = 3·λ∞²·λ4 + λ2_Y³, so
            # all conditions stay division-free (multiply through by 3·λ∞² > 0).
            l2 = self._lam(params, "lambda2")
            l4 = self._lam(params, "lambda4")
            l6 = self._lam(params, "lambda6")
            dl2 = self._lam(params, "delta_lambda2")
            linf = self._lam(params, "lambda_inf")
            pens.append(relu2(eps - linf))  # λ∞ > 0
            linf2 = linf * linf
            for y_sq in (0.0, self.ymax * self.ymax):
                l2Y = l2 + dl2 * self._cast(y_sq)
                if self.enforce_small_b:
                    pens.append(wall(l2Y))  # λ2_Y ≥ margin  (small-b turn-on)
                cubic = three * linf2 * l4 + l2Y**3  # 3·λ∞²·B
                if self._np_model == "tanh_2":
                    pens.append(wall(cubic))  # B ≥ margin  (large-b leading)
                else:  # tanh_6
                    pens.append(wall(l6))  # λ6 ≥ margin  (large-b leading)
                    # interior: B ≥ 0 OR B² ≤ 4·λ2_Y·λ6. In cubic-space (cubic =
                    # 3·λ∞²·B): cubic ≥ 0 OR cubic² ≤ 36·λ∞⁴·λ2_Y·λ6. Self-gating,
                    # division-free. (No margin on the interior discriminant.)
                    bound = thirtysix * linf2 * linf2 * l2Y * l6
                    pens.append(relu2(relu2(-cubic) - bound))

            return tf.add_n(pens)

    return NPDampingWall


# PEP-562 lazy class resolution: rabbit's loader does
#     module = importlib.import_module(...); cls = getattr(module, class_name)
# so the classes are synthesised on first attribute access, keeping the module
# importable without rabbit / TF (matches np_monotonicity.py).
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
