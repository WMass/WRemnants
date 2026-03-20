#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=recipes/alpha_s/common.sh
. "${script_dir}/common.sh"

usage() {
    cat <<'EOF'
Run the simultaneous alphaS Z+W unfolding fit.

Usage:
  recipes/alpha_s/unfolding_wz.sh Z_HIST W_HIST [options]

Options:
  --outdir PATH
EOF
}

if [[ $# -lt 2 ]]; then
    usage >&2
    exit 1
fi

z_hist=$1
w_hist=$2
shift 2

outdir="$(dirname "${z_hist}")"

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

fitdir="${outdir}/Combination_ZMassDileptonWMass_unfolding"
combine_file="${fitdir}/Combination.hdf5"
fitresult="${fitdir}/fitresults_asimov.hdf5"

setup_cmd=(
    python3
    scripts/rabbit/setupRabbit.py
    -i "${z_hist}" "${w_hist}"
    -o "${outdir}"
    --analysisMode unfolding
    --unfoldingLevel prefsr
    --poiAsNoi
    --fitvar "ptll-yll-cosThetaStarll_quantile-phiStarll_quantile" "eta-pt-charge"
    --genAxes "ptVGen-absYVGen-helicitySig" "absEtaGen-ptGen-qGen"
    --scaleNormXsecHistYields "0.05"
    --allowNegativeExpectation
    --realData
    --systematicType normal
    --postfix unfolding
    --unfoldSimultaneousWandZ
    --npUnc "${alpha_s_np_unc_model}"
    --pdfUncFromCorr
)

"${setup_cmd[@]}"

fit_cmd=(
    rabbit_fit.py
    "${combine_file}"
    -o "${fitdir}"
    --binByBinStatType normal-additive
    -t -1
    --doImpacts
    --globalImpacts
    --saveHists
    --computeHistErrors
    --computeHistImpacts
    --computeHistCov
    --compositeMapping
    -m Select "ch0_masked" "helicitySig:0"
    -m Select "ch1_masked"
    --postfix asimov
)

"${fit_cmd[@]}"

ln -sfn "${fitdir}" "${outdir}/alpha_s_wz_unfolding"
ln -sfn "${fitresult}" "${outdir}/alpha_s_wz_unfolding_fitresult.hdf5"
