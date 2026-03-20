#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=recipes/alpha_s/common.sh
. "${script_dir}/common.sh"

usage() {
    cat <<'EOF'
Run the alphaS Z histmaker with the analysis defaults.

Usage:
  recipes/alpha_s/histmaker_z.sh [options]

EOF
    alpha_s_usage_common
}

outdir="${PWD}"
data_path="/scratch/shared/NanoAOD/"
max_files=-1
nthreads=-1
postfix=""
filter_procs=()
exclude_procs=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --outdir)
            outdir=$2
            shift 2
            ;;
        --dataPath)
            data_path=$2
            shift 2
            ;;
        --maxFiles)
            max_files=$2
            shift 2
            ;;
        --nThreads)
            nthreads=$2
            shift 2
            ;;
        --postfix)
            postfix=$2
            shift 2
            ;;
        --filterProcs)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                filter_procs+=("$1")
                shift
            done
            ;;
        --excludeProcs)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                exclude_procs+=("$1")
                shift
            done
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

cmd=(
    python3
    scripts/histmakers/mz_dilepton.py
    --dataPath "${data_path}"
    -o "${outdir}"
    -j "${nthreads}"
    --maxFiles "${max_files}"
    --axes ptll yll
    --csVarsHist
    --forceDefaultName
    --unfolding
    --poiAsNoi
    --unfoldingAxes ptVGen absYVGen helicitySig
    --unfoldingInclusive
    --pdfs "${alpha_s_reco_pdfs[@]}"
    --theoryCorr "${alpha_s_reco_theory_corrs[@]}"
)

if [[ -n "${postfix}" ]]; then
    cmd+=(--postfix "${postfix}")
    actual_file="${outdir}/mz_dilepton_${postfix}.hdf5"
else
    actual_file="${outdir}/mz_dilepton.hdf5"
fi

if [[ ${#filter_procs[@]} -gt 0 ]]; then
    cmd+=(--filterProcs "${filter_procs[@]}")
fi

if [[ ${#exclude_procs[@]} -gt 0 ]]; then
    cmd+=(--excludeProcs "${exclude_procs[@]}")
fi

"${cmd[@]}"

ln -sfn "${actual_file}" "${outdir}/alpha_s_z_histmaker.hdf5"
