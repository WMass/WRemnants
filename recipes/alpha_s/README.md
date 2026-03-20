# AlphaS Recipes

This directory contains thin, user-facing workflow entry points for the current alphaS analysis configuration.

The recipes intentionally hardcode the analysis choices that should not, or cannot, become defaults in the generic scripts:
- the alphaS PDF inputs, currently just `ct18z`
- the theory-correction list, currently the `LatticeNP CT18Z` set plus the lattice-NP MSHT20 mass-range correction inputs
- the nonperturbative uncertainty model, currently `LatticeEigvars`
- the preferred fit variables and mapping choices for the current alphaS workflows

Those shared settings live in `common.sh`.

## Available recipes

- `histmaker_z.sh`: Z reco-level alphaS histmaker.
- `fitter_z.sh`: Z reco-level alphaS fits in 2D and 4D.
- `gen_fit_z.sh`: Z gen-level fit using the covariance from a Z-only or simultaneous W+Z unfolding fit.
- `histmaker_w.sh`: W reco-level alphaS histmaker.
- `fitter_wz.sh`: simultaneous Z+W reco-level alphaS fit.
- `unfolding_wz.sh`: simultaneous Z+W unfolding fit.
- `gen_fit_wz.sh`: simultaneous Z+W gen-level fit using the covariance from a simultaneous W+Z unfolding fit.

## Stable outputs

The fit recipes create stable symlinks in the chosen output directory so that CI and users do not need to know the exact `setupRabbit.py` folder naming. The main ones are:

- `alpha_s_z_histmaker.hdf5`
- `alpha_s_w_histmaker.hdf5`
- `alpha_s_z_reco_2d_fitresult.hdf5`
- `alpha_s_z_reco_4d_fitresult.hdf5`
- `alpha_s_z_gen_fit_fitresult.hdf5`
- `alpha_s_wz_reco_fitresult.hdf5`
- `alpha_s_wz_unfolding_fitresult.hdf5`
- `alpha_s_wz_gen_fit_fitresult.hdf5`

## Typical sequences

Z reco:
```bash
recipes/alpha_s/histmaker_z.sh --outdir /path/to/output
recipes/alpha_s/fitter_z.sh /path/to/output/alpha_s_z_histmaker.hdf5 --outdir /path/to/output
```

W+Z reco:
```bash
recipes/alpha_s/histmaker_z.sh --outdir /path/to/output
recipes/alpha_s/histmaker_w.sh --outdir /path/to/output
recipes/alpha_s/fitter_wz.sh \
  /path/to/output/alpha_s_z_histmaker.hdf5 \
  /path/to/output/alpha_s_w_histmaker.hdf5 \
  --outdir /path/to/output
```

W+Z unfolding + Z and W+Z gen fits:
```bash
recipes/alpha_s/unfolding_wz.sh \
  /path/to/output/alpha_s_z_histmaker.hdf5 \
  /path/to/output/alpha_s_w_histmaker.hdf5 \
  --outdir /path/to/output
recipes/alpha_s/gen_fit_z.sh \
  /path/to/output/alpha_s_z_histmaker.hdf5 \
  /path/to/output/alpha_s_wz_unfolding_fitresult.hdf5 \
  --outdir /path/to/output
recipes/alpha_s/gen_fit_wz.sh \
  /path/to/output/alpha_s_w_histmaker.hdf5 \
  /path/to/output/alpha_s_z_histmaker.hdf5 \
  /path/to/output/alpha_s_wz_unfolding_fitresult.hdf5 \
  --outdir /path/to/output
```
