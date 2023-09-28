#!/bin/bash

if [[ $# -lt 2 ]]; then
	echo "Requires at least two arguments: print_impacts.sh <input_file> <output_file>"
	exit 1
fi

. ./setup.sh
python3 scripts/combine/printImpacts.py -f $1 
if [ ! -d $2 ]; then
    mkdir -p $2
fi
python3 scripts/combine/pullsAndImpacts.py --oneSidedImpacts -f $1 output --outFolder $2 -o $3 --otherExtensions pdf png -n 50
