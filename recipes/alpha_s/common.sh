#!/bin/bash

alpha_s_recipes_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
alpha_s_repo_root="$(cd "${alpha_s_recipes_dir}/../.." && pwd)"

alpha_s_reco_pdfs=(
    "ct18z"
)

alpha_s_reco_theory_corrs=(
    "scetlib_dyturbo_LatticeNP_CT18Z_N3p0LL_N2LO"
    "scetlib_dyturbo_LatticeNP_CT18Z_N3p0LL_N2LO_pdfvars"
    "scetlib_dyturbo_LatticeNP_CT18Z_N3p0LL_N2LO_pdfas"
    "scetlib_dyturbo_LatticeNP_MSHT20mcrange_N3p0LL_N2LO_pdfvars"
    "scetlib_dyturbo_LatticeNP_MSHT20mbrange_N3p0LL_N2LO_pdfvars"
)

alpha_s_plot_theory_corrs=(
    "scetlib_dyturbo_LatticeNP_CT18Z_N3p0LL_N2LO"
    "scetlib_dyturbo_LatticeNP_CT18Z_N3p0LL_N2LO_pdfvars"
    "scetlib_dyturbo_LatticeNP_CT18Z_N3p0LL_N2LO_pdfas"
)

alpha_s_z_reco_fitvars=(
    "ptll-yll"
    "ptll-yll-cosThetaStarll_quantile-phiStarll_quantile"
)

alpha_s_np_unc_model="LatticeEigvars"

alpha_s_setup_env() {
    cd "${alpha_s_repo_root}" || return 1
    if [[ -z "${WREM_BASE:-}" ]]; then
        local had_nounset=0
        if [[ $- == *u* ]]; then
            had_nounset=1
            set +u
        fi
        # shellcheck source=/dev/null
        . "${alpha_s_repo_root}/setup.sh"
        if [[ ${had_nounset} -eq 1 ]]; then
            set -u
        fi
    fi
}

alpha_s_usage_common() {
    cat <<'EOF'
Common options:
  --outdir PATH
  --dataPath PATH
  --maxFiles N
  --nThreads N
  --lumiScale X
  --postfix NAME
  --filterProcs NAME [NAME ...]
  --excludeProcs NAME [NAME ...]
EOF
}
