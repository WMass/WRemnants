#!/bin/bash

if [[ $# -lt 2 ]]; then
	echo "Requires at least two arguments: show_impacts.sh <input_file> <output_file>"
	exit 1
fi

. ./setup.sh
python3 combinetf2/scripts/printImpacts.py $1 -s
python3 combinetf2/scripts/plot_pullsAndImpacts.py $1 --showNumbers --oneSidedImpacts --grouping max \
 --config utilities/styles/styles.py -o $2 --otherExtensions pdf png -n 50 --scaleImpacts 100 --title CMS --subtitle Preliminary
