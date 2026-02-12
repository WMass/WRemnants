# Due to memory constraints run the histmaker multiple times and append the output file gradually
# run e.g. source scripts/utilities/run_histmakers.sh wmass /scratch/$USER/results_histmaker/ nominal --unfolding --genVars ptGen absEtaGen --genBins 32 24 --pt 32 25 57 --noAuxiliaryHistograms

if [[ $# -lt 3 ]]; then
	echo "Requires at least three arguments: run_histmakers.sh <MODE> <OUTPUT_DIR> <POSTFIX> (<OPTIONAL OPTS>)"
	exit 1
fi

MODE=$1
OUTPUT_DIR=$2
POSTFIX=$3
shift
shift
shift

if [ "$MODE" == "wmass" ]; then
    HISTMAKER="mw_with_mu_eta_pt"
    separateProcs=("Wminusmunu_2016PostVFP" "Wplusmunu_2016PostVFP" "Wminustaunu_2016PostVFP" "Wplustaunu_2016PostVFP")
elif [ "$MODE" == "wlike" ]; then
    HISTMAKER="mz_wlike_with_mu_eta_pt"
    separateProcs=("Zmumu_2016PostVFP" "Ztautau_2016PostVFP")
elif [ "$MODE" == "dilepton" ]; then
    HISTMAKER="mz_dilepton"
    separateProcs=("Zmumu_2016PostVFP" "Ztautau_2016PostVFP")
fi

OUTPUT_FILE=$OUTPUT_DIR/${HISTMAKER}_${POSTFIX}.hdf5

OPTS="--forceDefaultName --postfix $POSTFIX $@"

CMD="python ./scripts/histmakers/${HISTMAKER}.py \
    -o $OUTPUT_DIR $OPTS --excludeProcs ${separateProcs[@]}"
# echo $CMD
eval $CMD

# Processes that should be processed individually
for proc in "${separateProcs[@]}"; do
    CMD="python ./scripts/histmakers/${HISTMAKER}.py \
        --appendOutputFile $OUTPUT_FILE $OPTS --filterProcs $proc"
    # echo $CMD
    eval $CMD
done



