#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=recipes/alpha_s/common.sh
. "${script_dir}/common.sh"

usage() {
    cat <<'EOF'
Run the simultaneous alphaS Z+W gen-level fit using the unfolded covariance from the same MC events.

Usage:
  recipes/alpha_s/gen_fit_wz.sh W_HIST Z_HIST UNFOLDING_FITRESULT [options]

Options:
  --outdir PATH
  --postfix NAME
EOF
}

if [[ $# -lt 3 ]]; then
    usage >&2
    exit 1
fi

w_hist=$1
z_hist=$2
unfolding_fitresult=$3
shift 3

outdir="$(dirname "${unfolding_fitresult}")"
postfix="CombinedTheoryFitViaMC"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --outdir)
            outdir=$2
            shift 2
            ;;
        --postfix)
            postfix=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

mkdir -p "${outdir}"
alpha_s_setup_env

fitdir="${outdir}/Combination_ZMassDileptonWMass_${postfix}"
combine_file="${fitdir}/Combination.hdf5"
fitresult="${fitdir}/fitresults.hdf5"

setup_cmd=(
    python3
    scripts/rabbit/setupRabbit.py
    -i "${z_hist}" "${w_hist}"
    -o "${outdir}"
    --fitresult "${unfolding_fitresult}" CompositeMapping
    --fitvar "ptVGen-absYVGen" "absEtaGen-ptGen-qGen"
    --select "helicitySig -1.0j 0.0j"
    --noi alphaS wmass
    --postfix "${postfix}"
    --baseName prefsr
    --fakeSmoothingMode hybrid
    --npUnc "${alpha_s_np_unc_model}"
    --pdfUncFromCorr
)

"${setup_cmd[@]}"

fit_cmd=(
    rabbit_fit.py
    "${combine_file}"
    -o "${fitdir}"
    --doImpacts
    --globalImpacts
    --computeHistErrors
    --computeVariations
    --covarianceFit
    -t -1
    -m BaseMapping
    -m Project ch0 ptVGen
    -m Project ch0 absYVGen
)

"${fit_cmd[@]}"

ln -sfn "${fitdir}" "${outdir}/alpha_s_wz_gen_fit"
ln -sfn "${fitresult}" "${outdir}/alpha_s_wz_gen_fit_fitresult.hdf5"
