#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=recipes/alpha_s/common.sh
. "${script_dir}/common.sh"

usage() {
    cat <<'EOF'
Run the simultaneous alphaS Z+W reco fit.

Usage:
  recipes/alpha_s/fitter_wz.sh Z_HIST W_HIST [options]

Options:
  --outdir PATH
  --lumiScale X
  --2D
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
lumi_scale=1
do_2d=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --outdir)
            outdir=$2
            shift 2
            ;;
        --lumiScale)
            lumi_scale=$2
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

setup_postfix="reco"

fitdir="${outdir}/Combination_ZMassDileptonWMass_${setup_postfix}"
combine_file="${fitdir}/Combination.hdf5"
fitresult="${fitdir}/fitresults.hdf5"

setup_cmd=(
    python3
    scripts/rabbit/setupRabbit.py
    -i "${z_hist}" "${w_hist}"
    --fitvar "ptll-yll-cosThetaStarll_quantile-phiStarll_quantile" "eta-pt-charge"
    --lumiScale "${lumi_scale}" "${lumi_scale}"
    -o "${outdir}"
    --noi alphaS wmass
    --postfix "${setup_postfix}"
    --fakeSmoothingMode hybrid
    --npUnc "${alpha_s_np_unc_model}"
    --pdfUncFromCorr
)

"${setup_cmd[@]}"

fit_cmd=(
    rabbit_fit.py
    "${combine_file}"
    -t -1
    --computeVariations
    -m Project ch0 ptll yll
    -m Select ch1
    --computeHistErrors
    --doImpacts
    -o "${fitdir}"
    --globalImpacts
)

"${fit_cmd[@]}"

ln -sfn "${fitdir}" "${outdir}/alpha_s_wz_reco"
ln -sfn "${fitresult}" "${outdir}/alpha_s_wz_reco_fitresult.hdf5"
