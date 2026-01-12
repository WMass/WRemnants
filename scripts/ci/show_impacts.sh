#!/bin/bash

if [[ $# -lt 2 ]]; then
	echo "Requires at least two arguments: show_impacts.sh <input_file> <output_file>"
	exit 1
fi

. ./setup.sh
rabbit_print_impacts.py $1 -s
rabbit_plot_pulls_and_impacts.py $1 -o $2 --showNumbers --oneSidedImpacts --grouping max \
 --config wremnants/utilities/styles/styles.py --otherExtensions pdf png -n 50 --scaleImpacts 100 --title CMS --subtitle Preliminary
