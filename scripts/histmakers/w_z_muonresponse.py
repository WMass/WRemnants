import argparse
from utilities import output_tools, common, rdf_tools, logging, differential

parser,initargs = common.common_parser(True)

import narf
import wremnants
from wremnants import theory_tools,syst_tools,theory_corrections, muon_calibration, muon_selections, muon_validation, unfolding_tools
from wremnants.histmaker_tools import scale_to_data, aggregate_groups
import hist
import lz4.frame
import math
import time
from utilities import boostHistHelpers as hh
import pathlib
import os
import numpy as np

parser.add_argument("--testHelper", action="store_true", help="Test the smearing weights helper")


args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

datasets = wremnants.datasets2016.getDatasets(maxFiles=args.maxFiles,
                                              filt=args.filterProcs,
                                              excl=args.excludeProcs,
                                              nanoVersion="v9", base_path=args.dataPath)

era = args.era


axis_genPt = hist.axis.Regular(45, 9., 81., name = "genPt")
axis_genEta = hist.axis.Regular(48, -2.4, 2.4, name = "genEta")
axis_genCharge =  hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "genCharge")
axis_qopr = hist.axis.Regular(1001, 0., 2.0, name = "qopr")

response_axes = [axis_genPt, axis_genEta, axis_genCharge, axis_qopr]

pileup_helper = wremnants.make_pileup_helper(era = era)
vertex_helper = wremnants.make_vertex_helper(era = era)

mc_jpsi_crctn_helper, data_jpsi_crctn_helper, jpsi_crctn_MC_unc_helper, jpsi_crctn_data_unc_helper = muon_calibration.make_jpsi_crctn_helpers(args, make_uncertainty_helper=True)

mc_calibration_helper, data_calibration_helper, calibration_uncertainty_helper = muon_calibration.make_muon_calibration_helpers(args)

smearing_helper = muon_calibration.make_muon_smearing_helpers() if args.smearing else None
bias_helper = muon_calibration.make_muon_bias_helpers(args) if args.biasCalibration else None

if args.testHelper:
    smearing_test_helper = muon_calibration.make_smearing_weight_test_helper()

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

        if args.testHelper:

            df = df.Define("smearing_weight_test", smearing_test_helper, ["selMuons_correctedPt", "selMuons_correctedEta", "selMuons_correctedCharge", "selMuons_genPt", "selMuons_genEta", "selMuons_genCharge", "nominal_weight"])

            # just use the output so that it gets evaluated for testing
            df = df.Define("smearing_weight_test0", "smearing_weight_test(0)")
            mean0 = df.Mean("smearing_weight_test0")
            results.append(mean0)

    return results, weightsum

resultdict = narf.build_and_run(datasets, build_graph)

output_tools.write_analysis_output(resultdict, f"{os.path.basename(__file__).replace('py', 'hdf5')}", args, update_name=not args.forceDefaultName)

