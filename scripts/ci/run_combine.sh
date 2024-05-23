#!/bin/bash
if [[ $# -lt 4 ]]; then
	echo "Requires at least four arguments: run_combine.sh <combinetf_dir> <mode> <working_dir> <combine_cards>"
	exit 1
fi

combinetf_dir=$1
mode=$2
working_dir=$3

source /cvmfs/cms.cern.ch/cmsset_default.sh
pushd $combinetf_dir
eval `scram runtime -sh`
popd

set -x

pushd $working_dir


if [[ $4 == *.hdf5 ]]; then
	# in case the card was written out as hdf5

	outfile=$4

	if [ "$mode" == "mass" ]; then
		combinetf.py --doImpacts --binByBinStat -t -1 "$outfile"
	elif [ "$mode" == "dilepton" ]; then
		combinetf.py --doImpacts --binByBinStat "$outfile" --saveHists --computeHistErrors
	elif [ "$mode" == "unfolding" ]; then
		combinetf.py --doImpacts --binByBinStat -t -1 "$outfile" --saveHists --computeHistErrors --correlateXsecStat
	elif [ "$mode" == "theoryfit" ]; then
		combinetf.py --doImpacts "$outfile" --chisqFit --externalCovariance --saveHists --computeHistErrors
	fi

else
	# in case the card(s) were written out as .txt
	maskedChannels=()
	cards=()
	for card in ${@:4}; do
		echo $card
		cards+=( $card )
		if [[ $card == xnorm* ]]; then
			IFS="=" read -ra parts <<< "$card"
			key="${parts[0]}"
			maskedChannels+=( "--maskedChan=${key}" )
		fi
	done

	card_name=$(basename ${working_dir}).txt
	combineCards.py ${cards[@]} > $card_name
	outfile=${card_name/txt/hdf5}

	if [ "$mode" == "mass" ]; then
		text2hdf5.py --X-allow-no-signal "$card_name"
		combinetf.py --doImpacts --binByBinStat -t -1 "$outfile"
	elif [ "$mode" == "dilepton" ]; then
		text2hdf5.py --X-allow-no-signal "$card_name"
		combinetf.py --doImpacts --binByBinStat "$outfile" --saveHists --computeHistErrors
	elif [ "$mode" == "unfolding" ]; then
		text2hdf5.py --X-allow-no-background "${maskedChannels[@]}" "$card_name"
		combinetf.py --doImpacts --binByBinStat -t -1 "$outfile" --saveHists --computeHistErrors --correlateXsecStat
	fi
fi

set +x
