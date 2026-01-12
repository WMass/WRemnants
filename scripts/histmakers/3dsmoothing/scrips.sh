#!/bin/bash

#SCRIPT TO HELP WITH THE PRODUCTION OF THE HISTOGRAMS TO DEFINE THE 3D EFFICIENCIES INTEGRATED OVER THE W UT DISTRIBUTION. YOU CAN OF COURSE CHANGE theory_corr, no_recoil etc...

NAME="newupdatesdeltaphi"

python WRemnants/scripts/histmakers/mw_with_mu_eta_pt.py -p idip${NAME} --muonScaleVariation manualShift --met DeepMETReso --binnedScaleFactors --vqtTest --vqtTestReal --vqt3dsmoothing --vqtTestStep 0 --vqtTestCorrectionStep 0
python WRemnants/scripts/histmakers/mw_with_mu_eta_pt.py -p trigger${NAME} --muonScaleVariation manualShift --met DeepMETReso --binnedScaleFactors --vqtTest --vqtTestReal --vqt3dsmoothing --vqtTestStep 1 --vqtTestCorrectionStep 1
python WRemnants/scripts/histmakers/mw_with_mu_eta_pt.py -p iso${NAME} --muonScaleVariation manualShift --met DeepMETReso --binnedScaleFactors --vqtTest --vqtTestReal --vqt3dsmoothing --vqtTestStep 2 --vqtTestCorrectionStep 2
