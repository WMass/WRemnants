from utilities import boostHistHelpers as hh, common, output_tools, logging

parser,initargs = common.common_parser()

import narf
import wremnants
from wremnants import theory_tools,syst_tools,theory_corrections, theoryAgnostic_tools
from wremnants.datasets.dataset_tools import getDatasets
import hist
import math
import os
import numpy as np


parser.add_argument("--skipAngularCoeffs", action='store_true', help="Skip the conversion of helicity moments to angular coeff fractions")
parser.add_argument("--singleLeptonHists", action='store_true', help="Also store single lepton kinematics")
parser.add_argument("--skipEWHists", action='store_true', help="Also store histograms for EW reweighting. Use with --filter horace")
parser.add_argument("--absY", action='store_true', help="use absolute |Y|")
parser.add_argument("--ptqVgen", action='store_true', help="If want to store qt by q")
parser.add_argument("--addhelicity", action='store_true', help="If want to store qt by q")

parser = common.set_parser_default(parser, "filterProcs", common.vprocs)

args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

datasets = getDatasets(maxFiles=args.maxFiles,
                        filt=args.filterProcs,
                        excl=args.excludeProcs, 
                        nanoVersion="v8" if args.v8 else "v9", base_path=args.dataPath, mode='gen')

logger.debug(f"Will process samples {[d.name for d in datasets]}")

axis_massWgen = hist.axis.Variable([5., 13000.], name="massVgen", underflow=True, overflow=False)

axis_massZgen = hist.axis.Regular(12, 60., 120., name="massVgen")

axis_absYVgen = hist.axis.Variable(
    # [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 10], 
    [0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 4., 5.], # this is the same binning as hists from theory corrections
    name = "absYVgen", underflow=False
)

axis_ygen = hist.axis.Regular(200, -5., 5., name="y")
axis_rapidity = axis_absYVgen if args.absY else axis_ygen
col_rapidity =  "absYVgen" if args.absY else "yVgen"

axis_ptVgen = hist.axis.Variable(
    # list(range(0,151))+[160., 190.0, 220.0, 250.0, 300.0, 400.0, 500.0, 800.0, 1500.0], 
    list(range(0,101)), # this is the same binning as hists from theory corrections
    name = "ptVgen", underflow=False,
)
#axis_ptqVgen = hist.axis.Regular(
#    20, 0., 0.5, name = "ptqVgen", underflow=False
#)
axis_ptqVgen = hist.axis.Variable(
    [round(x, 4) for x in list(np.arange(0, 0.1 + 0.0125, 0.0125))]+[round(x, 4) for x in list(np.arange(0.1+0.025, 0.5 + 0.025, 0.025))], name = "ptqVgen", underflow=False
)

axis_chargeWgen = hist.axis.Regular(
    2, -2, 2, name="chargeVgen", underflow=False, overflow=False
)

axis_chargeZgen = hist.axis.Integer(
    0, 1, name="chargeVgen", underflow=False, overflow=False
)

axis_l_eta_gen = hist.axis.Regular(48, -2.4, 2.4, name = "eta")
axis_l_pt_gen = hist.axis.Regular(29, 26., 55., name = "pt")

corr_helpers = theory_corrections.load_corr_helpers(common.vprocs, args.theoryCorr)

def build_graph(df, dataset):
    logger.info("build graph")
    logger.info(dataset.name)
    results = []
    
    if dataset.is_data:
        raise RuntimeError("Running GEN analysis over data is not supported")

    isW = dataset.name in common.wprocs
    isZ = dataset.name in common.zprocs

    weight_expr = "std::copysign(1.0, genWeight)"
    df = df.Define("weight", weight_expr)
    # This sum should happen before any change of the weight
    weightsum = df.SumAndCount("weight")

    if "reweight_h2" in dataset.name:
        weight_expr = f"{weight_expr}*H2BugFixWeight[0]"
    elif "NNLOPS" in dataset.name:
        weight_expr = f"{weight_expr}*LHEScaleWeightAltSet1[4]"

    df = theory_tools.define_theory_weights_and_corrs(df, dataset.name, corr_helpers, args)

    if isZ:
        if(args.ptqVgen):
            nominal_axes = [axis_massZgen, axis_rapidity, axis_ptqVgen, axis_chargeZgen]
        else:
            nominal_axes = [axis_massZgen, axis_rapidity, axis_ptVgen, axis_chargeZgen]
        lep_axes = [axis_l_eta_gen, axis_l_pt_gen, axis_chargeZgen]
    else:
        if(args.ptqVgen):
            nominal_axes = [axis_massWgen, axis_rapidity, axis_ptqVgen, axis_chargeWgen]
        else:
            nominal_axes = [axis_massWgen, axis_rapidity, axis_ptVgen, axis_chargeWgen]
        lep_axes = [axis_l_eta_gen, axis_l_pt_gen, axis_chargeWgen]
   
    if(args.ptqVgen):
        nominal_cols = ["massVgen", col_rapidity, "ptqVgen", "chargeVgen"]
    else:
        nominal_cols = ["massVgen", col_rapidity, "ptVgen", "chargeVgen"]
    lep_cols = ["etaPrefsrLep", "ptPrefsrLep", "chargeVgen"]

    if args.singleLeptonHists and isW or isZ:
        if isW:
            df = df.Define('ptPrefsrLep', 'genl.pt()')
            df = df.Define('etaPrefsrLep', 'genl.eta()')
        else:
            df = df.Define('ptPrefsrLep', 'genlanti.pt()')
            df = df.Define('etaPrefsrLep', 'genlanti.eta()')
        results.append(df.HistoBoost("nominal_genlep", lep_axes, [*lep_cols, "nominal_weight"], storage=hist.storage.Double()))

    if not args.skipEWHists and (isW or isZ):
        if isZ:
            massBins = theory_tools.make_ew_binning(mass = 91.1535, width = 2.4932, initialStep=0.010)
        else:
            massBins = theory_tools.make_ew_binning(mass = 80.3815, width = 2.0904, initialStep=0.010)
        ew_cols = ['ewMll', 'ewLogDeltaM']
        axis_ewMll = hist.axis.Variable(massBins, name = "ewMll")
        axis_ewLogDeltaM = hist.axis.Regular(90, -5, 4, name = "ewLogDeltaM")
        ew_axes = [axis_ewMll, axis_ewLogDeltaM]
        results.append(df.HistoBoost("nominal_ew", ew_axes, [*ew_cols, "nominal_weight"], storage=hist.storage.Weight()))
        # auxiliary hists
        axis_ewMlly = hist.axis.Variable(massBins, name = "ewMlly")
        results.append(df.HistoBoost("nominal_ewMlly", [axis_ewMlly], ["ewMlly", "nominal_weight"], storage=hist.storage.Weight()))
        # coarse binning
        axis_Mll = hist.axis.Regular(100, 50, 150, name = "Mll")
        results.append(df.HistoBoost("nominal_Mll", [axis_Mll], ["ewMll", "nominal_weight"], storage=hist.storage.Weight()))
        axis_Mlly = hist.axis.Regular(100, 50, 150, name = "Mlly")
        results.append(df.HistoBoost("nominal_Mlly", [axis_Mlly], ["ewMlly", "nominal_weight"], storage=hist.storage.Weight()))
    nominal_gen = df.HistoBoost("nominal_gen", nominal_axes, [*nominal_cols, "nominal_weight"],  storage=hist.storage.Weight()) #storage=hist.storage.Double())

    results.append(nominal_gen)
    if not 'horace' in dataset.name:
        if "LHEScaleWeight" in df.GetColumnNames():
            df = theory_tools.define_scale_tensor(df)
            syst_tools.add_qcdScale_hist(results, df, nominal_axes, nominal_cols, "nominal_gen")
            df = df.Define("helicity_moments_scale_tensor", "wrem::makeHelicityMomentScaleTensor(csSineCosThetaPhi, scaleWeights_tensor, nominal_weight)")
            helicity_moments_scale = df.HistoBoost("helicity_moments_scale", nominal_axes, [*nominal_cols, "helicity_moments_scale_tensor"], tensor_axes = [wremnants.axis_helicity, *wremnants.scale_tensor_axes], storage=hist.storage.Weight())#, storage=hist.storage.Double())
            results.append(helicity_moments_scale)

        if "LHEPdfWeight" in df.GetColumnNames():
            #df = theoryAgnostic_tools.define_helicity_weights(df)
            syst_tools.add_pdf_hists(results, df, dataset.name, nominal_axes, nominal_cols, args.pdfs, "nominal_gen", args.addhelicity)

    if args.theoryCorr and dataset.name in corr_helpers:
        results.extend(theory_tools.make_theory_corr_hists(df, "nominal_gen", nominal_axes, nominal_cols,
            corr_helpers[dataset.name], args.theoryCorr, modify_central_weight=not args.theoryCorrAltOnly, isW=isW)
        )
        if args.singleLeptonHists:
            results.extend(theory_tools.make_theory_corr_hists(df, "nominal_genlep", lep_axes, lep_cols, 
                corr_helpers[dataset.name], args.theoryCorr, modify_central_weight=not args.theoryCorrAltOnly, isW=isW)
            )

    if "MEParamWeight" in df.GetColumnNames():
        df = syst_tools.define_mass_weights(df, proc=dataset.name)
        syst_tools.add_massweights_hist(results, df, nominal_axes, nominal_cols, "nominal_gen", proc=dataset.name)

    return results, weightsum

resultdict = narf.build_and_run(datasets, build_graph)
if(args.ptqVgen):
    output_tools.write_analysis_output(resultdict, f"{os.path.basename(__file__).replace('.py', f'_pdf_{args.pdfs[0]}_vars_qtbyQ.hdf5')}", args, update_name=not args.forceDefaultName)
else:
    output_tools.write_analysis_output(resultdict, f"{os.path.basename(__file__).replace('.py', f'_pdf_{args.pdfs[0]}_vars_qt.hdf5')}", args, update_name=not args.forceDefaultName)

logger.info("computing angular coefficients")
z_moments = None
w_moments = None

if not args.skipAngularCoeffs:
    for dataset in datasets:
        name = dataset.name
        if "helicity_moments_scale" not in resultdict[name]["output"]:
            logger.warning(f"Failed to find helicity_moments_scale hist for proc {name}. Skipping!")
            continue
        moments = resultdict[name]["output"]["helicity_moments_scale"].get()
        if name in common.zprocs:
            if z_moments is None:
                z_moments = moments
            else:
                new_moments = moments
                z_moments = hh.addHists(z_moments, new_moments, createNew=False)
        elif name in common.wprocs:
            if w_moments is None:
                w_moments = moments
            else:
                new_moments = moments
                w_moments = hh.addHists(w_moments, new_moments, createNew=False)

    coeffs={}
    # REMINDER: common.ptV_binning is not the one using 10% quantiles, and the quantiles are not a subset of this binning, but apparently it doesn't matter
    if z_moments:
        z_moments = hh.rebinHist(z_moments, axis_ptVgen.name, common.ptV_binning)
        z_moments = hh.rebinHist(z_moments, axis_massZgen.name, axis_massZgen.edges[::2])
        coeffs["Z"] = wremnants.moments_to_angular_coeffs(z_moments)
    if w_moments:
        w_moments = hh.rebinHist(w_moments, axis_ptVgen.name, common.ptV_binning)
        coeffs["W"] = wremnants.moments_to_angular_coeffs(w_moments)
    if coeffs:
        outfname = "w_z_coeffs_absY.hdf5" if args.absY else "w_z_coeffs.hdf5"
        output_tools.write_analysis_output(coeffs, outfname, args, update_name=not args.forceDefaultName)
