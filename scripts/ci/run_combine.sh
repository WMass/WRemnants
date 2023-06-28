#!/bin/bash
if [[ $# -lt 4 ]]; then
	echo "Requires at least four arguments: run_combine.sh <combinetf_dir> <mode> <working_dir> <combine_cards>"
	exit 1
fi

combinetf_dir=$1
mode=$2
working_dir=$3

cards=()
for card in ${@:4}; do
	echo $card
	cards+=( $card )
done

source /cvmfs/cms.cern.ch/cmsset_default.sh
pushd $combinetf_dir
eval `scram runtime -sh`
popd

pushd $working_dir

card_name=$(basename ${working_dir}).txt
combineCards.py ${cards[@]} > $card_name

outfile=${card_name/txt/hdf5}

set -x

if [ "$mode" == "mass" ]; then
	text2hdf5.py --X-allow-no-signal "$card_name"
	combinetf.py --doImpacts --binByBinStat -t -1 "$outfile"
elif [ "$mode" == "unfold" ]; then
	text2hdf5.py --X-allow-no-background --maskedChan=xnorm "$card_name"
	combinetf.py --doImpacts --binByBinStat -t -1 --correlateXsecStat "$outfile"
fi

set +x
