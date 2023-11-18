import argparse
from utilities import common, rdf_tools, logging, differential
from utilities.io_tools import output_tools

parser,initargs = common.common_parser(True)

import narf
import wremnants
from wremnants import theory_tools,syst_tools,theory_corrections, muon_calibration, muon_selections, muon_validation, unfolding_tools
from wremnants.histmaker_tools import scale_to_data, aggregate_groups
from wremnants.datasets.dataset_tools import getDatasets
import hist
import lz4.frame
import math
import time
from utilities import boostHistHelpers as hh
import pathlib
import os
import numpy as np
import ROOT

parser.add_argument("--testHelpers", action="store_true", help="Test the smearing weights helper")


args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

datasets = getDatasets(maxFiles=args.maxFiles,
                        filt=args.filterProcs,
                        excl=args.excludeProcs,
                        nanoVersion="v9", base_path=args.dataPath)

era = args.era


axis_genPt = hist.axis.Regular(45, 9., 81., name = "genPt")
axis_genEta = hist.axis.Regular(50, -2.5, 2.5, name = "genEta")
axis_genCharge =  hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "genCharge")
axis_qopr = hist.axis.Regular(1001, 0., 2.0, name = "qopr")

response_axes = [axis_genPt, axis_genEta, axis_genCharge, axis_qopr]

pileup_helper = wremnants.make_pileup_helper(era = era)
vertex_helper = wremnants.make_vertex_helper(era = era)

calib_filepaths = common.calib_filepaths
mc_jpsi_crctn_helper, data_jpsi_crctn_helper, jpsi_crctn_MC_unc_helper, jpsi_crctn_data_unc_helper = muon_calibration.make_jpsi_crctn_helpers(args, calib_filepaths, make_uncertainty_helper=True)

mc_calibration_helper, data_calibration_helper, calibration_uncertainty_helper = muon_calibration.make_muon_calibration_helpers(args)

smearing_helper, smearing_uncertainty_helper = (None, None) if args.noSmearing else muon_calibration.make_muon_smearing_helpers()
bias_helper = muon_calibration.make_muon_bias_helpers(args) if args.biasCalibration else None

sigmarel = 5e-3
scalerel = 5e-4
nreps = 100

smearing_helper_simple_multi = ROOT.wrem.SmearingHelperSimpleMulti[nreps](sigmarel)

if args.testHelpers:
    response_helper = ROOT.wrem.SplinesDifferentialWeightsHelper(f"{wremnants.data_dir}/calibration/muon_response.tflite")

    smearing_helper_simple = ROOT.wrem.SmearingHelperSimple(sigmarel, ROOT.ROOT.GetThreadPoolSize())
    smearing_helper_simple_weights = ROOT.wrem.SmearingHelperSimpleWeight(sigmarel)
    smearing_helper_simple_transform = ROOT.wrem.SmearingHelperSimpleTransform(sigmarel)
    scale_helper_simple_weights = ROOT.wrem.ScaleHelperSimpleWeight(scalerel)

def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")
    results = []
    isW = dataset.name in common.wprocs
    isZ = dataset.name in common.zprocs
    isTop = dataset.group == "Top"
    isQCDMC = dataset.group == "QCD"

    cvh_helper = data_calibration_helper if dataset.is_data else mc_calibration_helper
    jpsi_helper = data_jpsi_crctn_helper if dataset.is_data else mc_jpsi_crctn_helper

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    weightsum = df.SumAndCount("weight")

    if dataset.is_data:
        df = df.DefinePerSample("nominal_weight", "1.0")
    else:
        df = df.Define("weight_pu", pileup_helper, ["Pileup_nTrueInt"])
        df = df.Define("weight_vtx", vertex_helper, ["GenVtx_z", "Pileup_nTrueInt"])

        weight_expr = "weight*weight_pu*weight_vtx"
        df = df.Define("nominal_weight", weight_expr)


    df = muon_calibration.define_corrected_muons(df, cvh_helper, jpsi_helper, args, dataset, smearing_helper, bias_helper)

    df = df.Define("vetoMuonsPre", "Muon_looseId && abs(Muon_dxybs) < 0.05 && Muon_correctedCharge != -99")


    if not dataset.is_data:
        df = df.Define("selMuons", "vetoMuonsPre && (Muon_genPartFlav == 1 || Muon_genPartFlav==15)")

        df = df.Define("selMuons_correctedPt", "Muon_correctedPt[selMuons]")
        df = df.Define("selMuons_correctedEta", "Muon_correctedEta[selMuons]")
        df = df.Define("selMuons_correctedPhi", "Muon_correctedPhi[selMuons]")
        df = df.Define("selMuons_correctedCharge", "Muon_correctedCharge[selMuons]")
        df = df.Define("selMuons_genPartIdx", "Muon_genPartIdx[selMuons]")
        df = df.Define("selMuons_genPt", "Take(GenPart_pt, selMuons_genPartIdx)")
        df = df.Define("selMuons_genEta", "Take(GenPart_eta, selMuons_genPartIdx)")
        df = df.Define("selMuons_genPhi", "Take(GenPart_phi, selMuons_genPartIdx)")
        df = df.Define("selMuons_genPdgId", "Take(GenPart_pdgId, selMuons_genPartIdx)")
        df = df.Define("selMuons_genCharge", "-1*(selMuons_genPdgId > 0) + 1*(selMuons_genPdgId < 0)")
        df = df.Define("selMuons_qop", "selMuons_correctedCharge*1.0/(selMuons_correctedPt*cosh(selMuons_correctedEta))")
        df = df.Define("selMuons_genQop", "selMuons_genCharge*1.0/(selMuons_genPt*cosh(selMuons_genEta))")
        df = df.Define("selMuons_qopr", "selMuons_qop/selMuons_genQop")


    if isW or isZ:
        response_cols = ["selMuons_genPt", "selMuons_genEta", "selMuons_genCharge", "selMuons_qopr"]
        hist_qopr = df.HistoBoost("hist_qopr", response_axes, [*response_cols, "nominal_weight"])
        results.append(hist_qopr)

        df = df.Define("selMuons_shiftedqopr", f"(1. + {scalerel})*selMuons_qopr")

        response_cols_shifted = ["selMuons_genPt", "selMuons_genEta", "selMuons_genCharge", "selMuons_shiftedqopr"]
        hist_qopr_shifted = df.HistoBoost("hist_qopr_shifted", response_axes, [*response_cols_shifted, "nominal_weight"])
        hist_qopr_shifted._hist.metadata = { "scalerel" : scalerel }
        results.append(hist_qopr_shifted)

        df = df.Define("selMuonsMulti_smearedmqop", smearing_helper_simple_multi, ["run", "luminosityBlock", "event", "selMuons_correctedPt", "selMuons_correctedEta", "selMuons_correctedCharge"])

        df = df.Define("selMuonsMulti_genPt", f"wrem::replicate_rvec(selMuons_genPt, {nreps})")
        df = df.Define("selMuonsMulti_genEta", f"wrem::replicate_rvec(selMuons_genEta, {nreps})")
        df = df.Define("selMuonsMulti_genCharge", f"wrem::replicate_rvec(selMuons_genCharge, {nreps})")
        df = df.Define("selMuonsMulti_genQop", f"wrem::replicate_rvec(selMuons_genQop, {nreps})")
        df = df.Define("selMuonsMulti_smearedqopr", "selMuonsMulti_smearedmqop/selMuonsMulti_genQop")

        response_cols_smeared_multi = ["selMuonsMulti_genPt", "selMuonsMulti_genEta", "selMuonsMulti_genCharge", "selMuonsMulti_smearedqopr"]
        hist_qopr_smearedmulti = df.HistoBoost("hist_qopr_smearedmulti", response_axes, [*response_cols_smeared_multi, "nominal_weight"])
        hist_qopr_smearedmulti._hist.metadata = { "sigmarel" : sigmarel }
        results.append(hist_qopr_smearedmulti)


        if args.testHelpers:

            df = df.Define("selMuons_response_weight", response_helper, ["selMuons_correctedPt", "selMuons_correctedEta", "selMuons_correctedCharge", "selMuons_genPt", "selMuons_genEta", "selMuons_genCharge"])

            df = df.Define("weight_smear", smearing_helper_simple_weights, ["selMuons_correctedPt", "selMuons_correctedEta", "selMuons_correctedCharge", "selMuons_response_weight", "nominal_weight"])

            df = df.DefineSlot("selMuons_smearedPt", smearing_helper_simple, ["selMuons_correctedPt", "selMuons_correctedEta", "selMuons_correctedCharge"])

            df = df.Define("selMuons_smearedqop", "selMuons_correctedCharge*1.0/(selMuons_smearedPt*cosh(selMuons_correctedEta))")
            df = df.Define("selMuons_smearedqopr", "selMuons_smearedqop/selMuons_genQop")

            df = df.Define("selMuons_transformedPt", smearing_helper_simple_transform, ["selMuons_correctedPt", "selMuons_correctedEta", "selMuons_correctedCharge", "selMuons_response_weight"])

            df = df.Define("selMuons_transformedqop", "selMuons_correctedCharge*1.0/(selMuons_transformedPt*cosh(selMuons_correctedEta))")
            df = df.Define("selMuons_transformedqopr", "selMuons_transformedqop/selMuons_genQop")

            df = df.Define("weight_scale", scale_helper_simple_weights, ["selMuons_correctedPt", "selMuons_correctedEta", "selMuons_correctedCharge", "selMuons_response_weight", "nominal_weight"])

            response_cols_smeared = ["selMuons_genPt", "selMuons_genEta", "selMuons_genCharge", "selMuons_smearedqopr"]
            hist_qopr_smeared = df.HistoBoost("hist_qopr_smeared", response_axes, [*response_cols_smeared, "nominal_weight"])
            results.append(hist_qopr_smeared)

            response_cols_transformed = ["selMuons_genPt", "selMuons_genEta", "selMuons_genCharge", "selMuons_transformedqopr"]
            hist_qopr_transformed = df.HistoBoost("hist_qopr_transformed", response_axes, [*response_cols_transformed, "nominal_weight"])
            results.append(hist_qopr_transformed)

            hist_qopr_smeared_weight = df.HistoBoost("hist_qopr_smeared_weight", response_axes, [*response_cols, "weight_smear"])
            results.append(hist_qopr_smeared_weight)

            hist_qopr_scaled_weight = df.HistoBoost("hist_qopr_scaled_weight", response_axes, [*response_cols, "weight_scale"])
            results.append(hist_qopr_scaled_weight)

    return results, weightsum

resultdict = narf.build_and_run(datasets, build_graph)

output_tools.write_analysis_output(resultdict, f"{os.path.basename(__file__).replace('py', 'hdf5')}", args, update_name=not args.forceDefaultName)

