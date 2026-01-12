HISTMAKER_FILE=$1
PROJECT=$2
# PROJECT=240301_fakes_biasstudy/full_model_fakeNorm10

ANALYSIS=WMass

CMSSW_BASE=/home/d/dwalter/CMSSW_10_6_30/src
COMBINE_OUTDIR=/scratch/dwalter/CombineStudies/

PSEUDODATA=("simple" "extended1D" "extended2D")


if [ "${ANALYSIS}" == "ZMassDilepton" ]; then
	WEBDIR=${PROJECT}/dilepton_oldEfficiency
    FITVARS=( "ptll" "ptll-yll" )
    SETS=("asimov" "data" ${PSEUDODATA[@]})
elif [ "${ANALYSIS}" == "ZMassWLike" ]; then
    WEBDIR=${PROJECT}/wlike
    FITVARS=( "eta-pt-charge" )
    SETS=("asimov" ${PSEUDODATA[@]})
elif [ "${ANALYSIS}" == "WMass" ]; then
    WEBDIR=${PROJECT}/wmass
    FITVARS=( "eta-pt-charge" )
    SETS=("asimov" ${PSEUDODATA[@]})
fi

for FITVAR in "${FITVARS[@]}"; do
    echo Run fits for: ${FITVAR}

    FITVARSTING=$(echo "$FITVAR" | tr '-' '_')

    echo Run over: ${PSEUDODATA[@]}
    for PSEUDO1 in "${PSEUDODATA[@]}"; do

        SUBPROJECT=${ANALYSIS}_${FITVARSTING}

        COMBINE_ANALYSIS_OUTDIR=${COMBINE_OUTDIR}/${PROJECT}/${SUBPROJECT}_${PSEUDO1}/
        COMBINE_ANALYSIS_PATH=${COMBINE_ANALYSIS_OUTDIR}/${ANALYSIS}.hdf5

        if [ -e $COMBINE_ANALYSIS_PATH ]; then
            echo "The file $COMBINE_ANALYSIS_PATH exists, continue using it."
        else
            echo "The file $COMBINE_ANALYSIS_PATH does not exists, produce it."
            COMMAND="./scripts/ci/run_with_singularity.sh scripts/ci/setup_and_run_python.sh scripts/combine/setupCombine.py \
            -i $HISTMAKER_FILE --fakeEstimation ${PSEUDO1} --postfix ${PSEUDO1} --fitvar $FITVAR \
            -o $COMBINE_OUTDIR/${PROJECT} --hdf5 --pseudoDataFakes ${PSEUDODATA[@]}"
            echo Run command: $COMMAND
            eval $COMMAND
        fi

        echo Run over: ${PSEUDODATA[@]}
        for PSEUDO2 in "${PSEUDODATA[@]}"; do

            echo "Perform fit for $PSEUDO2"

            # 2) pseudodata fit
            FITRESULT=${COMBINE_ANALYSIS_OUTDIR}/fitresults_123456789_${PSEUDO2}.hdf5

            if [ -e $FITRESULT ]; then
                echo "The file $FITRESULT exists, continue using it."
            elif [ $PSEUDO2 == "data" ]; then
                cmssw-cc7 --command-to-run scripts/ci/setup_and_run_combine.sh $CMSSW_BASE $COMBINE_ANALYSIS_OUTDIR \
                    ${ANALYSIS}.hdf5 --postfix $PSEUDO2 --binByBinStat --doImpacts  --saveHists --computeHistErrors
            elif [ $PSEUDO2 == "asimov" ]; then
                cmssw-cc7 --command-to-run scripts/ci/setup_and_run_combine.sh $CMSSW_BASE $COMBINE_ANALYSIS_OUTDIR \
                    ${ANALYSIS}.hdf5 -t -1 --postfix $PSEUDO2 --binByBinStat --doImpacts  --saveHists --computeHistErrors
            else 
                cmssw-cc7 --command-to-run scripts/ci/setup_and_run_combine.sh $CMSSW_BASE $COMBINE_ANALYSIS_OUTDIR \
                    ${ANALYSIS}.hdf5 -p $PSEUDO2 --postfix $PSEUDO2 --binByBinStat --doImpacts  --saveHists --computeHistErrors
            fi

            # 3) plots
            # 3.1) prefit and postfit plots
            ./scripts/ci/run_with_singularity.sh scripts/ci/setup_and_run_python.sh scripts/plotting/postfitPlots.py \
                $COMBINE_ANALYSIS_OUTDIR/fitresults_123456789_${PSEUDO2}.root -o $COMBINE_ANALYSIS_OUTDIR -f $PSEUDO2 --yscale '1.2' --rrange '0.9' '1.1' --prefit
            ./scripts/ci/run_with_singularity.sh scripts/ci/setup_and_run_python.sh scripts/plotting/postfitPlots.py \
                $COMBINE_ANALYSIS_OUTDIR/fitresults_123456789_${PSEUDO2}.root -o $COMBINE_ANALYSIS_OUTDIR -f $PSEUDO2 --yscale '1.2' --rrange '0.99' '1.01'

            # 3.2) impact plots
            if [ "${PSEUDO2}" != "asimov" ]; then
                echo "Make impacts for ${PSEUDO2}"
                if [ "${ANALYSIS}" == "ZMassDilepton" ]; then
                    ./scripts/ci/run_with_singularity.sh scripts/ci/setup_and_run_python.sh scripts/combine/pullsAndImpacts.py \
                        -r $COMBINE_ANALYSIS_OUTDIR/fitresults_123456789_asimov.hdf5 -f $COMBINE_ANALYSIS_OUTDIR/fitresults_123456789_${PSEUDO2}.hdf5 \
                        -m ungrouped --sortDescending -s constraint \
                        output --outFolder $COMBINE_ANALYSIS_OUTDIR/$PSEUDO2 -o impacts${SUBPROJECT}.html --otherExtensions pdf png -n 50 
                        # --oneSidedImpacts --grouping max -t utilities/styles/nuisance_translate.json \
                else
                    ./scripts/ci/run_with_singularity.sh scripts/ci/setup_and_run_python.sh scripts/combine/pullsAndImpacts.py \
                        -f $COMBINE_ANALYSIS_OUTDIR/fitresults_123456789_${PSEUDO2}.hdf5 \
                        --oneSidedImpacts --grouping max -t utilities/styles/nuisance_translate.json \
                        output --outFolder $COMBINE_ANALYSIS_OUTDIR/$PSEUDO2 -o impacts${SUBPROJECT}.html --otherExtensions pdf png -n 50 
                        # -r $COMBINE_ANALYSIS_OUTDIR/fitresults_123456789_asimov.hdf5 \
                fi
            fi
            mkdir -p /home/d/dwalter/www/WMassAnalysis/${PROJECT}/${SUBPROJECT}_${PSEUDO1}/
            mv $COMBINE_ANALYSIS_OUTDIR/$PSEUDO2 /home/d/dwalter/www/WMassAnalysis/${PROJECT}/${SUBPROJECT}_${PSEUDO1}/
        done

        COMBINE_ANALYSIS_OUTDIR=${COMBINE_OUTDIR}/${PROJECT}/${SUBPROJECT}_closure_${PSEUDO1}/
        COMBINE_ANALYSIS_PATH=${COMBINE_ANALYSIS_OUTDIR}/${ANALYSIS}.hdf5

        if [ -e $COMBINE_ANALYSIS_PATH ]; then
            echo "The file $COMBINE_ANALYSIS_PATH exists, continue using it."
        else
            # a special pseudodata card for the closure
            COMMAND="./scripts/ci/run_with_singularity.sh scripts/ci/setup_and_run_python.sh scripts/combine/setupCombine.py \
            -i $HISTMAKER_FILE --fakeEstimation ${PSEUDO1} --postfix closure_${PSEUDO1} --fitvar $FITVAR \
            -o $COMBINE_OUTDIR/${PROJECT} --hdf5 --pseudoDataFakes closure"
            echo Run command: $COMMAND
            eval $COMMAND
        fi

        # 2) pseudodata closure
        FITRESULT=${COMBINE_ANALYSIS_OUTDIR}/fitresults_123456789_closure.hdf5

        if [ -e $FITRESULT ]; then
            echo "The file $FITRESULT exists, continue using it."
        else 
            cmssw-cc7 --command-to-run scripts/ci/setup_and_run_combine.sh $CMSSW_BASE $COMBINE_ANALYSIS_OUTDIR \
                ${ANALYSIS}.hdf5 -p closure --postfix closure --binByBinStat --doImpacts  --saveHists --computeHistErrors
        fi

        ./scripts/ci/run_with_singularity.sh scripts/ci/setup_and_run_python.sh scripts/plotting/postfitPlots.py \
            $COMBINE_ANALYSIS_OUTDIR/fitresults_123456789_closure.root -o $COMBINE_ANALYSIS_OUTDIR -f closure --yscale '1.2' --rrange '0.9' '1.1' --prefit
        ./scripts/ci/run_with_singularity.sh scripts/ci/setup_and_run_python.sh scripts/plotting/postfitPlots.py \
            $COMBINE_ANALYSIS_OUTDIR/fitresults_123456789_closure.root -o $COMBINE_ANALYSIS_OUTDIR -f closure --yscale '1.2' --rrange '0.99' '1.01'

        # 3.2) impact plots
        ./scripts/ci/run_with_singularity.sh scripts/ci/setup_and_run_python.sh scripts/combine/pullsAndImpacts.py \
            -f $COMBINE_ANALYSIS_OUTDIR/fitresults_123456789_closure.hdf5 \
            --oneSidedImpacts --grouping max -t utilities/styles/nuisance_translate.json \
            output --outFolder $COMBINE_ANALYSIS_OUTDIR/closure -o impacts${SUBPROJECT}.html --otherExtensions pdf png -n 50 

        mkdir -p /home/d/dwalter/www/WMassAnalysis/${PROJECT}/${SUBPROJECT}_closure/
    done
    # 5) make summary table
    ./scripts/ci/run_with_singularity.sh scripts/ci/setup_and_run_python.sh scripts/tests/summarytable_fakes.py \
        ${COMBINE_OUTDIR}/${PROJECT}/${SUBPROJECT}_*/fitresults_123456789_*.root -o ${COMBINE_OUTDIR}/${PROJECT}/ -f ./
        
    pdflatex -output-directory ${COMBINE_OUTDIR}/${PROJECT}/ ${COMBINE_OUTDIR}/${PROJECT}/table_${ANALYSIS}.tex
    pdflatex -output-directory ${COMBINE_OUTDIR}/${PROJECT}/ ${COMBINE_OUTDIR}/${PROJECT}/table_mass_${ANALYSIS}.tex

    mv ${COMBINE_OUTDIR}/${PROJECT}/ /home/d/dwalter/www/WMassAnalysis/${PROJECT}/
done

# # 5) make big summary table
# ./scripts/ci/run_with_singularity.sh scripts/ci/setup_and_run_python.sh scripts/tests/summarytable.py \
#     ${COMBINE_OUTDIR}/${PROJECT}/${ANALYSIS}_*/fitresults_123456789_*.root -f ${WEBDIR}

# pdflatex -output-directory /home/d/dwalter/www/WMassAnalysis/${WEBDIR}/ /home/d/dwalter/www/WMassAnalysis/${WEBDIR}/table_${ANALYSIS}.tex
