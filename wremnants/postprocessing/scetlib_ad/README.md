# `scetlib_ad`: a differentiable SCETlib prediction for rabbit

A rabbit `ParamModel` in which the parameters SCETlib exposes (alpha_s, the
nonperturbative λ, the TNPs, PDF eigenvector coefficients, the profile scales)
are continuous fit parameters with exact first and second derivatives, via
`ScetlibCachedXsecTF` (SCETlib `autodiff-sigmaul` branch) replaying a prepared
cache. Which parameters exist is a property of the cache. The prediction:

```
sigma_reco(b) = sum_g  R_raw(b, g) / N_gen(g) * sigma(p; g)
rnorm(b)      = sigma_reco(p; b) / sigma_reco(p_anchor; b)
```

rabbit multiplies the signal templates by `rnorm` (with `gen_level=1`, the ratio
of `sigma(p; g)` on a gen-binned card). `p_anchor` is the point the card's theory
correction was computed at, read from the card metadata (`scetlib_corr_config`).

| file | role |
|---|---|
| `param_model.py` | `SCETlibADParamModel`, the rabbit adapter |
| `params.py` | name map, reparametrisation, priors, frozen defaults, impact groups |
| `xsec_backend.py` | `ScetlibADXsec` (load cache, values/Jacobian), `GenFold` |
| `response.py` | the reco fold: R from the datacard auxiliary, positivity floor |
| `response_matrix.py` | load R and N_gen from a histmaker output |
| `corr_config.py` | record the correction's runcard in the histmaker metadata |
| `scale_envelope.py` | the frozen mu_R / mu_F envelope nuisances |

## Setting up SCETlib

SCETlib is not a submodule of WRemnants. Clone `scetlib-cms` (branch `autodiff-sigmaul`) from
`gitlab.cern.ch/scetlib/contrib/scetlib-cms` (not anonymously readable: use Kerberos, ssh or a token) and
compile it inside the WRemnants singularity image, which has the clang/LLVM pair, CMake, GSL and LHAPDF it
needs:

```bash
singularity run --bind /scratch/,/work/,/home/,/ceph/ \
    /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling:latest
git clone -b autodiff-sigmaul ssh://git@gitlab.cern.ch:7999/scetlib/contrib/scetlib-cms.git
cd scetlib-cms && ./bootstrap.sh -j 32
```

`bootstrap.sh` builds the patched clad (a compile-time Clang plugin, pinned to a fixed commit) and then
SCETlib into `scetlib-cms/build`. Before using this package, in the same container:

```bash
source <path-to>/scetlib-cms/setup.sh   # or SCETLIB_BUILD=<build dir> source .../setup.sh
```

It puts `scetlib_qT`, `scetlib_tf` and `scetlib_run` on `PYTHONPATH` and sets `LD_LIBRARY_PATH` and the
stack limit. Without it the model fails at import with a message saying so.

Which SCETlib to use is a property of the cache, not of this package: WRemnants only calls SCETlib's Python
interface, and refuses a cache whose format differs from the one the loaded SCETlib reads. Use a SCETlib that
reads the cache's format; the physics of a cache is fixed when it is built.

## Building a cache

The build is SCETlib's own, and a runcard is all it takes: the runcards in
`scripts/rabbit/scetlib_ad/conf/` are complete, `[Grid_*]` included (the gen grid the
cache is built on). Pass the same runcard to the fit, which rebuilds the calculation
from it. The `[TNPs]` block is what registers the TNPs, and the `[QED]` block (QED ISR at
NLL', `qed_kappa = yes`) the QED order steps `qedStep_*`.

Inside the WRemnants container, after `source <scetlib-cms>/setup.sh`:

- on one machine:
  `python <scetlib-cms>/examples/matched_ad/prepare_cache.py --config <runcard> --out <cache> --threads N --n-train 9`.
  `--threads` is the lever (the member stage is parallel over the nodes of all bins);
  `--bins LO:HI` builds a slice of the bin list (test caches), and disjoint slices merge
  with `scetlib_cache.merge_bin_caches`.
- over HTCondor, one job per bin or bin group, then a merge:
  `<scetlib-cms>/examples/matched_ad/condor_cache_build.py` (`submit`, `status`,
  `resubmit`, `merge`; see its `--help`).

Pass `--n-train` explicitly: `prepare_cache.py` defaults to `max(9, 1.5 n_params)`, the
production caches used 9, and the rule solve costs roughly `n_train^2`.

- `extract_cache_rules.py <cache>.npz` writes `<stem>.rules.bin` beside the
  cache, which `ScetlibADXsec` then loads with far less memory.
- `backend_check.py --conf <runcard> --cache <cache>.npz` checks a cache without
  rabbit: anchor round trip, finite-difference Jacobian, Hessian symmetry, and
  the sum rule of the fold onto the gen grid.

## Making a card

```bash
python scripts/histmakers/mz_dilepton.py --unfolding --poiAsNoi \
    --theoryCorr <corr> ... --responseGenBinning theoryCorr
python scripts/rabbit/setupRabbit.py -i <histmaker>.hdf5 ... \
    --storeResponseMatrix --responseMatrixGenBinning response
```

The histmaker writes the response on the correction's gen grid and records the
correction's runcard; `setupRabbit` embeds R and N_gen as a datacard auxiliary.
Card systematics that duplicate a fitted parameter (`scetlibNP`, `pdfAlphaS`,
`resumTNP`, `pdf<N>`, `resumFOScale`, `resumTransition`) are refused; drop them
with `setupRabbit --excludeNuisances`.

## Fitting

```bash
rabbit_fit.py <card>.hdf5 \
  --paramModel wremnants.postprocessing.scetlib_ad.SCETlibADParamModel \
      cache=<cache>.npz conf=<runcard>.conf [key=value ...] \
  --jitCompile off -t 0 --doImpacts -o <dir>
```

`--jitCompile off` is required (SCETlib is reached through `tf.py_function`,
which XLA cannot compile); the model refuses to construct otherwise. Spec tokens
are documented in `SCETlibADParamModel.__init__`: `gen_level`, `fit_params`,
`poi_params`, `priors`, `prior_sigmas`, `xparam_default`, `anchor_source`, ...

Defaults, as in `params.py`:

- **Fitted**: every cache parameter except `DEFAULT_FROZEN`: `alphaS` (the POI),
  the NP λ, `resumTransition2`, the TNPs, `pdfEig*`, plus two envelope nuisances
  `resumFOScaleEnvSymAvg/SymDiff` (3-point mu_R envelope, zero below 20 GeV).
- **Frozen**: `lambda_inf`, `lambda_inf_nu`, `b0_over_bmax_nu`,
  `resumTNP_b_qqDS`, `resumTransition1/3`, `resumScaleMuR/MuF`.
- **Units**: fitted parameters are unit nuisances theta mapped to physical
  values (`REPARAM`): `alphaS` and the λ as anchor + width·theta (width 0.002 for
  `alphaS`), `resumTransition2` quadratically, `pdfEig*` scaled to 1 sigma.
- **Priors** (on by default): `FREE_PARAMS` (`alphaS`, `lambda2`, `lambda4`,
  `delta_lambda2`, `lambda2_nu`, `lambda4_nu`) are free; every other fitted
  parameter (TNPs, `pdfEig*`, `resumTransition2`, the envelope, `lambda6*`) gets
  an N(0, 1) prior in theta.

## Blinding

rabbit blinds the POI unless `--unblind` is given; `alphaS` being a unit
nuisance, the additive offset hides its value but not its uncertainty or impacts.
`compute()` receives the true values, so nothing on the evaluation path may
print, log or format a parameter value.
