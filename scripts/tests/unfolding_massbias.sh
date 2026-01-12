# unfolding bias test
# 0) Generate card with nominal model and pseudodata sets with different masses 
# 1) Unfold pseudodata with mW set to values of {0, 10, 20, ...}
# 2) Generate card with nominal theory model
# 3) Fit theory model to unfolded pseudo data 
# 4) Compare observed mass pulls with pseudodata value

# hyperparameters
# a) uncertainty on mass when unfolding
# a1) None
# a2) 100MeV
# b) Histmaker type
# b1) mz_wlike
# b2) mw
# c) transverse mass cut (only for mz_wlike possible on real data)
# c1) with cuts
# c2) without cut 

# input arguments: 

HISTMAKER_FILE=$1
COMBINE_OUTDIR=$2

shift
shift
OPTSTAT=""  # other options can be: --massUnc X --doStatOnly
OPTMASS=""
STATONLY=""
MASS=""
for arg in "$@"; do
  # Check if the current argument is the specific string you're looking for
  if [ "$arg" = "--doStatOnly" ]; then
    echo "Run with stat only."
    STATONLY=_statOnly
    OPTSTAT="--doStatOnly"
  elif [ "$arg" = "--massUnc" ]; then
    echo "Run with mass uncertainty."
    MASS="_massUnc"
  elif [ "$MASS" = "_massUnc" ]; then
    echo "Mass uncertainty is ${arg} MeV."
    MASS="${MASS}${arg}"
    OPTMASS="--massUnc ${arg}"
  fi

done

LABEL=Z
ANALYSIS=ZMassWLike

CMSSW_BASE=/home/${USER:0:1}/${USER}/CMSSW_10_6_30/src/

COMBINE_ANALYSIS_OUTDIR=${COMBINE_OUTDIR}/${ANALYSIS}_eta_pt_charge${STATONLY}/
COMBINE_ANALYSIS_PATH=${COMBINE_ANALYSIS_OUTDIR}/${ANALYSIS}.hdf5

# 0) Generate card with nominal model and pseudodata sets with different masses 
if [ -e $COMBINE_ANALYSIS_PATH ]; then
    echo "The file $COMBINE_ANALYSIS_PATH exists, continue using it."
else
    echo "The file $COMBINE_ANALYSIS_PATH does not exists, produce it."
    COMMAND="./scripts/ci/run_with_singularity.sh scripts/ci/setup_and_run_python.sh scripts/combine/setupCombine.py \
        -i $HISTMAKER_FILE -o $COMBINE_OUTDIR --hdf5 --sparse --unfolding \
        --pseudoData massWeight$LABEL --pseudoDataAxes massShift --pseudoDataIdxs -1 $OPTSTAT $OPTMASS"
    echo $COMMAND
    eval $COMMAND
fi


# 1) Unfold pseudodata with mW set to values of {0, 10, 20, ...}

# for ((i=-100; i<=100; i+=10)); do
for i in 0 50 100; do
    IABS=$i
    if [ $i -lt 0 ]; then
        UPDOWN="Down"
        IABS=$((-1*$IABS))
    elif [ $IABS -eq 0 ]; then
        UPDOWN=""
    else
        UPDOWN="Up"
    fi
    BIN=massShift${LABEL}${IABS}MeV$UPDOWN
    PSEUDO="massWeight${LABEL}_massShift_${BIN}"

    echo "Perform unfolding for $PSEUDO"

    # 1) Unfold pseudodata
    FITRESULT=${COMBINE_ANALYSIS_OUTDIR}/fitresults_123456789_${BIN}.hdf5

    if [ -e $FITRESULT ]; then
        echo "The file $FITRESULT exists, continue using it."
    else
        COMMAND="cmssw-cc7 --command-to-run scripts/ci/setup_and_run_combine.sh $CMSSW_BASE $COMBINE_ANALYSIS_OUTDIR \
            ${ANALYSIS}.hdf5 -p $PSEUDO --postfix $BIN --binByBinStat --correlateXsecStat"
        echo $COMMAND
        eval $COMMAND
    fi

    # 2)  Generate card with nominal theory model
    THEOMODEL_DIR=${COMBINE_OUTDIR}/${ANALYSIS}_qGen_ptGen_absEtaGen${STATONLY}_${BIN}/
    THEOMODEL=${ANALYSIS}.hdf5
    THEOMODEL_PATH=${THEOMODEL_DIR}/${THEOMODEL}

    if [ -e $THEOMODEL_PATH ]; then
        echo "The file $THEOMODEL_PATH exists, continue using it."
    else
        COMMAND="./scripts/ci/run_with_singularity.sh scripts/ci/setup_and_run_python.sh scripts/combine/setupCombine.py \
            -i $HISTMAKER_FILE -o $COMBINE_OUTDIR --hdf5 --fitvar qGen-ptGen-absEtaGen --postfix $BIN $OPTSTAT \
            --fitresult $FITRESULT"
        echo $COMMAND
        eval $COMMAND        
    fi

    # 3) Fit theory model to unfolded pseudo data 
    FITRESULT_FINAL=${THEOMODEL_DIR}/fitresults_123456789.hdf5

    if [ -e $FITRESULT_FINAL ]; then
        echo "The file $FITRESULT_FINAL exists, continue using it."
    else
        COMMAND="cmssw-cc7 --command-to-run scripts/ci/setup_and_run_combine.sh $CMSSW_BASE $THEOMODEL_DIR $THEOMODEL --chisqFit --externalCovariance" 
        echo $COMMAND
        eval $COMMAND
    fi

    # 4) Compare observed mass pulls with pseudodata value

done




