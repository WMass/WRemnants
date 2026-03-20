#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=recipes/alpha_s/common.sh
. "${script_dir}/common.sh"

usage() {
    cat <<'EOF'
Run the alphaS Z unfolding fit.

Usage:
  recipes/alpha_s/unfolding_z.sh INPUT_HIST [options]

Options:
  --outdir PATH
EOF
}

if [[ $# -lt 1 ]]; then
    usage >&2
    exit 1
fi

input_hist=$1
shift

outdir="$(dirname "${input_hist}")"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --outdir)
            outdir=$2
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

fitdir="${outdir}/ZMassDilepton_ptll_yll_cosThetaStarll_quantile_phiStarll_quantile_unfolding"
combine_file="${fitdir}/ZMassDilepton.hdf5"
fitresult="${fitdir}/fitresults_asimov.hdf5"

setup_cmd=(
    python3
    scripts/rabbit/setupRabbit.py
    -i "${input_hist}"
    -o "${outdir}"
    --analysisMode unfolding
    --poiAsNoi
    --fitvar "ptll-yll-cosThetaStarll_quantile-phiStarll_quantile"
    --genAxes "ptVGen-absYVGen-helicitySig"
    --scaleNormXsecHistYields "0.05"
    --allowNegativeExpectation
    --realData
    --systematicType normal
    --postfix unfolding
    --npUnc "${alpha_s_np_unc_model}"
    --pdfUncFromCorr
)

"${setup_cmd[@]}"

fit_cmd=(
    rabbit_fit.py
    "${combine_file}"
    -o "${fitdir}"
    --binByBinStatType normal-multiplicative
    -t -1
    --doImpacts
    --globalImpacts
    --computeHistErrors
    --computeHistImpacts
    --computeHistCov
    --compositeMapping
    -m Select "ch0_masked" "helicitySig:0"
    --postfix asimov
)

"${fit_cmd[@]}"

ln -sfn "${fitdir}" "${outdir}/alpha_s_z_unfolding"
ln -sfn "${fitresult}" "${outdir}/alpha_s_z_unfolding_fitresult.hdf5"
