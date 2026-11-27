SCRIPT_FILE_REL_PATH="${BASH_SOURCE[0]}"
if [[ "$SCRIPT_FILE_REL_PATH" == "" ]]; then
  SCRIPT_FILE_REL_PATH="${(%):-%N}"
fi

export PATH="$PATH:/home/submit/david_w/.local/bin/"
WREM_BASE=$( cd "$( dirname "${SCRIPT_FILE_REL_PATH}" )" && pwd )
export WREM_BASE=$(readlink -f "$WREM_BASE")

source ${WREM_BASE}/narf/setup.sh
source ${WREM_BASE}/rabbit/setup.sh
source ${WREM_BASE}/wums/setup.sh

export PYTHONPATH="${WREM_BASE}:$PYTHONPATH"

echo "Created environment variable WREM_BASE=${WREM_BASE}"

# utility variables pointing to specific folders in the filesystem
export COMBINE_STUDIES="${WREM_BASE}/scripts/combine/"
echo "Created environment variable COMBINE_STUDIES=${COMBINE_STUDIES}"

export PLOTS="${WREM_BASE}/scripts/analysisTools/"
echo "Created environment variable PLOTS=${PLOTS}"

# GPU library path fixes (only relevant inside a Singularity/Apptainer "--nv"
# container). Two independent image bugs are worked around here:
#  1. The image bundles a stale libcuda in /usr/lib that shadows the host driver
#     injected by "--nv" in /.singularity.d/libs. Prepending that dir makes the
#     real host driver win; otherwise TensorFlow hits a CUDA driver/kernel
#     version mismatch and reports no GPU.
#  2. The TensorFlow 2.21+ build has an incomplete RPATH (e.g. it omits
#     nvidia/cusolver/lib), so it cannot dlopen all CUDA libs and silently falls
#     back to CPU with "Cannot dlopen some GPU libraries". Appending the bundled
#     NVIDIA wheel lib dirs lets it find them.
if [[ -d /.singularity.d/libs ]]; then
  export LD_LIBRARY_PATH="/.singularity.d/libs:${LD_LIBRARY_PATH}"
  for _nvlib in /opt/venv/lib/python*/site-packages/nvidia/*/lib; do
    [[ -d "${_nvlib}" ]] && LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${_nvlib}"
  done
  export LD_LIBRARY_PATH
  unset _nvlib
fi
