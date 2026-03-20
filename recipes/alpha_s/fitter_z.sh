#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=recipes/alpha_s/common.sh
. "${script_dir}/common.sh"

usage() {
    cat <<'EOF'
Run the alphaS Z reco fits in 2D and 4D.

Usage:
  recipes/alpha_s/fitter_z.sh INPUT_HIST [options]

Options:
  --outdir PATH
  --lumiScale X
EOF
}

if [[ $# -lt 1 ]]; then
    usage >&2
    exit 1
fi

input_hist=$1
shift

outdir="$(dirname "${input_hist}")"
lumi_scale=1

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

for fitvar in "${alpha_s_z_reco_fitvars[@]}"; do
    if [[ "${fitvar}" == "ptll-yll" ]]; then
        fitdir="${outdir}/ZMassDilepton_ptll_yll"
        stable_dir_link="${outdir}/alpha_s_z_reco_2d"
        stable_fit_link="${outdir}/alpha_s_z_reco_2d_fitresult.hdf5"
    else
        fitdir="${outdir}/ZMassDilepton_ptll_yll_cosThetaStarll_quantile_phiStarll_quantile"
        stable_dir_link="${outdir}/alpha_s_z_reco_4d"
        stable_fit_link="${outdir}/alpha_s_z_reco_4d_fitresult.hdf5"
    fi

    combine_file="${fitdir}/ZMassDilepton.hdf5"
    fitresult="${fitdir}/fitresults.hdf5"

    setup_cmd=(
        python3
        scripts/rabbit/setupRabbit.py
        -i "${input_hist}"
        --fitvar "${fitvar}"
        --lumiScale "${lumi_scale}"
        --realData
        -o "${outdir}"
        --noi alphaS
        --npUnc "${alpha_s_np_unc_model}"
        --pdfUncFromCorr
    )

    "${setup_cmd[@]}"

    fit_cmd=(
        rabbit_fit.py
        "${combine_file}"
        -t -1
        --computeHistErrors
        --doImpacts
        -o "${fitdir}/"
    )
    "${fit_cmd[@]}"

    ln -sfn "${fitdir}" "${stable_dir_link}"
    ln -sfn "${fitresult}" "${stable_fit_link}"
done
