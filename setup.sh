SCRIPT_FILE_REL_PATH="${BASH_SOURCE[0]}"
if [[ "$SCRIPT_FILE_REL_PATH" == "" ]]; then
  SCRIPT_FILE_REL_PATH="${(%):-%N}"
fi
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
