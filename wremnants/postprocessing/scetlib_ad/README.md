# `scetlib_ad` — a fully differentiable SCETlib prediction for rabbit

A rabbit `ParamModel` in which **every theory parameter SCETlib exposes is a
continuous fit parameter with exact derivatives**, rather than a discrete template
morph whose joint response with the others is an outer product:

| parameter group | count (Z) | notes |
|---|---|---|
| `alphaS` | 1 | physical units; with a PDF α_s member pair it is the PDF-consistent coupling, not α_s at fixed PDF |
| nonperturbative λ | 8 | Collins–Soper and TMD form factors, `tanh_2`/`tanh_6` |
| theory nuisance parameters | 10 | `gamma_cusp`, `gamma_mu_q`, `gamma_nu`, `s`, `h_qqV` + 5 beam-function TNPs |
| PDF eigenvector coefficients | `n_eig` | extra differentiable columns, exact at `c_e = 0, ±1` |

Which of these exist is a property of the **cache**, not of this code: the model
reads `gradient_param_names()` and registers what it finds. Only the
profile-scale parameters — `kappaFO`, `kappaf`, `muf`, the transition points —
are outside SCETlib's autodiff (they need d/dμ of the PDF convolution grids) and
still need template nuisances.

The physics comes from the SCETlib `autodiff-sigmaul` branch: `ScetlibCachedXsecTF`
(`scetlib-cms/py/scetlib_tf.py`) replays a prepared cache — compressed bin rules
for the resummed piece plus a frozen fixed-order grid for the nonsingular one —
and returns exact first and second derivatives from clad.

| file | role |
|---|---|
| `params.py` | SCETlib ↔ rabbit name map, prior σ, POI/POU defaults, impact groups |
| `xsec_backend.py` | `ScetlibADXsec` (configure + cache load + value/J/K) and `GenFold` (cache bins → the card's gen grid) |
| `response.py` | the reco fold: `P(b|g) = R_raw/N_gen`, the datacard response auxiliary, the positivity floor |
| `param_model.py` | `SCETlibADParamModel` — the rabbit adapter |

Scripts live in `scripts/rabbit/scetlib_ad/`. The BUILD itself is SCETlib's --
`scetlib-cms/examples/matched_ad/prepare_cache.py` holds the steps
(`plan_variations`, `build_prologue`, `build_variations`, `write_cache`,
`fork_member_build`) and `scetlib-cms/py/scetlib_cache.py` the cache file
format and both merges. The two scripts here are the rabbit side: the gen axes
off a card, the runcard, `--subset`, and the shard scheduling.


| script | role |
|---|---|
| `backend_check.py` | standalone cache sanity: anchor round trip, FD-checked Jacobian, Hessian symmetry, fold sum rule |
| `prepare_cache_for_card.py` | build a cache for a card's gen binning, or an explicit `--grid-json` |
| `build_cache_parallel.py` | split the BINS across processes (`--bin-groups`) and merge the shards (`--merge-bins`, `--merge-only`) |
| `make_debug_card.py` | a self-contained gen-level card built from a cache, for closure tests |
| `compare_to_scetlib_run.py` | validate the resummed piece against a native SCETlib production run |
| `conf/Z_CT18Z_N3p0LL_FranksVals.conf` | runcard reproducing the current analysis central (see below) |

## Running

Inside the WRemnants singularity, `source setup.sh` then the SCETlib one:

```bash
source $WREM_BASE/scetlib-cms/setup.sh     # PYTHONPATH, LD_LIBRARY_PATH, ulimit -s

# ONE job: minimize, postfit Hessian, impacts.
rabbit_fit.py <gen_card>.hdf5 -v 3 \
  --paramModel wremnants.postprocessing.scetlib_ad.SCETlibADParamModel \
      cache=<cache>.npz conf=<runcard>.conf gen_level=1 threads=32 \
  --jitCompile off -t 0 --doImpacts -o <dir>
```

`--jitCompile off` is **mandatory** — the model reaches SCETlib through
`tf.py_function`, which XLA cannot compile. The model refuses to construct
otherwise, with that message.

**One pass, not two.** The fit and the postfit covariance are a single job. The
model always includes the exact second-derivative term, so the composite Hessian
is exact and there is nothing to configure — no `--noHessian` fit followed by an
`--externalPostfit --noFit` pass. Measured on a 30-bin debug card, one pass
reproduces the two-pass numbers exactly (α_s = 0.1195 ± 0.00045, identical λ
uncertainties) with a much harder-converged EDM (6.1e-22 vs 3.5e-17).

A Gauss-Newton variant dropping that term was measured ~5× faster and, on Asimov,
numerically identical — but it is an approximation on real data, so it is
deliberately **not** offered. The exact Hessian costs ~1 s/bin of serial work,
scaling as `1 + P(P+1)/2` in the parameter count: 3–80 s per minimiser iteration
on 64 threads at the few-hundred-to-1200-bin gen binnings we fit on. Reintroduce a
switch only if a fit on the full correction grid ever needs it.

Two things keep that affordable: the `(value, J, K)` triple is cached on the
parameter vector, so one C++ Hessian is computed per *distinct* point rather than
per HVP (the minimiser takes many HVPs at fixed `x`); and the exact Hessian makes
the minimiser converge harder, so it needs fewer iterations.

## How the derivatives get into TensorFlow

`ScetlibCachedXsecTF` is an ordinary TF-differentiable function — its backward pass
is itself a `custom_gradient` whose own gradient contracts Hessian-vector
products — so nested `GradientTape`s work and TF drives every C++ call. The model
just calls it inside the graph, exactly as `examples/matched_ad/tf_gradients.py`
does. **There is no surrogate anywhere**: autodiff differentiates the real
prediction, and rabbit's postfit Hessian is the real Hessian.

One requirement that imposes, and it is easy to break by accident:

> Map rabbit's fit vector into SCETlib's layout with a **constant 0/1 matrix
> multiply**, never `tensor_scatter_nd_update`.

Some mapping is unavoidable — rabbit's vector holds only the fitted parameters,
POIs first, while SCETlib's holds every registered parameter in registry order. But
a scatter's backward pass contains a gather, whose gradient TF represents as
`tf.IndexedSlices`, and the bridge's second-order py_function payloads call
`.numpy()` on the incoming cotangent and fail on it — so **everything past first
order breaks**, while first order keeps working, which makes it a nasty way to
fail. Isolated: a nested-tape HVP works on a bare `Variable`, and fails with a
scatter in front even when the scatter covers the whole vector. A constant matmul
is bit-identical (the entries are exactly 0 and 1) and free at these sizes.

Fixing it upstream — densifying the incoming cotangent before `.numpy()` in
`_uhvp_py` — would remove the trap rather than leave us relying on a comment.

## The anchor: the correction is the authority

The model returns `rnorm = σ_gen(p) / σ_gen(p_anchor)`, and rabbit **multiplies
the card's templates** by it. Those templates were reweighted by a theory
correction produced from SCETlib, so `rnorm = 1` at the fit start only if
`p_anchor` is the point *that correction* was computed at. The anchor is
therefore defined by the correction, and the cache — a different SCETlib
artefact — does not get to define it. This is not a preference; it is forced by
what the model returns.

Be precise about what depends on where the cache was **built**. Correctness of
the ratio is about where the cache is **evaluated**, which is `p_anchor`. The
build point affects *robustness* only, and only for the member-served
parameters: the 8 NP λ and the 10 TNPs ride the AD tape and are exact anywhere,
while `alphaS` comes from a PDF α_s member pair and the eigenvectors are exact
at `c_e = 0, ±1` and interpolated between.

So:

* the histmaker records the **whole resummed runcard verbatim** under
  `scetlib_corr_config` (`lambda_central.build_corr_config_meta`), alongside the
  older curated NP extract; verbatim, so a later edit to the correction pkl
  cannot change what an existing output means;
* the model builds the anchor from it (`params.CORR_ANCHOR_KEYS`) and **refuses**
  a card that records none — the central values would be unknown, and taking
  them from the cache is the silent failure the refusal exists to prevent;
* `anchor_source=cache` is the opt-out. Deliberately a *word*, not an off
  switch: it declares "predict around the cache's build point even though the
  templates were built elsewhere", which someone has to own. Logged loudly.

Why the refusal is worth its inconvenience: a wrong anchor is **invisible**. The
ratio is 1 at the fit start either way, so every prefit plot looks perfect and
only the derivatives are wrong, perturbing around the wrong origin. The α_s
blinding bug found on 2026-09-09 was exactly this shape — SCETlib was being
evaluated at α_s = 1.7e-05, the starting loss was 4.6e7 against a converged 361,
and no check fired.

### What is checked, and what is not

The cache's own runcard is compared against the recorded one on load. The
comparison covers a **curated positive list** (`params.CORR_REFUSE_KEYS`,
`CORR_WARN_KEYS`), not everything available, and nothing outside it is reported
at all — an allowlist has to be maintained against a config that keeps growing,
and a report nobody acts on trains people to ignore the output.

| class | what | on mismatch |
|---|---|---|
| REFUSE | the PDF (`pdf_set`, `pdf_member`); the orders (`alphas_order`, `nf`, `run_order`, `fixed_order`); what the parameters MEAN (`np_model*`, TNP modes, `form_np_prescription`, `profile_functional_form`, the b\* prescription, `kappafo`/`kappaf`, the profile floors, `Electroweak`, `Process.boson`) | raise — the cache computes a different function and no parameter value reconciles it |
| WARN | parameter VALUES: every shared numeric `Nonperturbative` key, the TNP values, `transition_points`, `alphas_mu0` | log; the anchor follows the **correction** |
| not compared | `Grid_*` (the fit's range vs the production grid), `Integration` (quadrature accuracy), `calculation_piece` (`sing` vs `matched` by construction), and anything on neither list | nothing |

Two asymmetries worth knowing:

* the WARN class is benign for the λ and TNPs (the tape is exact away from the
  build point) but **not** for `alphas_mu0`, which is interpolated — warning
  there is accepted interpolation error, not a free pass;
* the cache side is `core.conf`, i.e. the runcard *layered on SCETlib's*
  `defaults.conf`, because that is what the calculation is configured from.
  Against the bare runcard file, 33 of the correction's 107 keys have no
  counterpart at all — the per-flavour `lambda2_*`, `lambda4_i`, `lambda6`,
  `lambda6_nu`, `np_model_tmd`, `kappafo` — and those are exactly the ones a
  build-default difference would hide. Layered, all 107 have a counterpart;
* and the comparison iterates the **correction's** keys, so keys only the
  *cache* carries are never visited — a key the correction lacks has no value to
  disagree with. Measured, that is **12** keys and every one is fixed-order /
  matching machinery (`fo_order2_*`, `matched_nons_qt_cut`). Not an accidental
  gap: `calculation_piece` is `matched` against `sing` by construction, so those
  knobs exist *because* of that difference and sit inside the fixed-order
  exclusion below. (`fo_order2_analytic` is `yes` in the cache runcard, `no` in
  today's `defaults.conf`, and absent entirely from the correction's resolved
  config — the build that made the correction predates the knob.)

There is **no `defaults.conf` fallback for an anchor-bearing value**, and that is
deliberate. A runcard can keep a compiled-in default with no runtime key: an
older `tanh_6` build hardcoded the CS-side `lambda_6_nu` at 0.0007 while this
checkout defaults it to 0. A fallback would quietly hand over the wrong anchor.
A missing anchor value refuses instead, and the one escape is auditable —
`anchor_override=lambda6_nu=0.0007`, recorded in the fit's spec and printed at
construction.

Finally, `--theoryCorrAltOnly` refuses regardless of `anchor_source`: the nominal
templates then carry no correction at all, so there is no correction anchor for
them and the recorded config describes a prediction they never saw.

Not covered, stated so we are not fooling ourselves: the **fixed-order half** of
a `scetlib_dyturbo` correction. Only the resummed file's runcard is recorded, so
"the correction is the authority" holds for the resummed sector.

## Validating a cache

Two independent checks, both cheap relative to a fit:

- `compare_to_scetlib_run.py` — against a native SCETlib production run with
  `calculation_piece = sing`. That pkl **is** the resummed cross section,
  bin-integrated, so replaying only our cache's rules gives the same object: no
  matching, no fixed-order generator, no MC, no correction file in between. Any
  disagreement is therefore ours — runcard, quadrature, Q integration, rule
  compression. The reference's finer bins are summed onto ours, which is exact
  because both sides are bin-integrated, and the script refuses to run unless our
  edges really are a subset of the reference's.
- `backend_check.py` — the cache's own consistency, plus the fold sum rule.

**Use an analysis runcard, not `examples/matched_ad/matched.conf`.** The example
runcard is plain N3LL with SCETlib's default profile scales; the analysis is
N³⁺⁰LL — every TNP at `theta = 0` with `'level0'` — with `lambda = 0`, transition
points `[0.2, 0.6, 1.0]`, scale floors, `compensate_fo` and `collins_soper4`.
Measured on a 30-bin grid, that difference moves the λ response by 7–35% of the
response itself. `conf/Z_CT18Z_N3p0LL_FranksVals.conf` transcribes the config of
the production run the current analysis correction is built from, with the
provenance in comments.

Note the consequence: the analysis order is *defined* by the `[TNPs]` block, and a
non-`off` TNP scheme is exactly what registers a TNP as a gradient parameter. So
an analysis-faithful cache has 19 parameters, not 9. They are not fitted by
default — pass them in `fit_params`, and then `priors=1` is required (the model
refuses to float a TNP free, and refuses any parameter whose Jacobian column is
identically zero, which `resumTNP_b_qqDS` is for the Z).

Study logbook: `WRemnantsHelpers/studies/scetlib-ad-param-model/LOGBOOK.md`.
