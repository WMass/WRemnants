from utilities import boostHistHelpers as hh, common, output_tools, logging

parser,initargs = common.common_parser()

import narf
import wremnants
from wremnants import theory_tools,syst_tools,theory_corrections
from wremnants.datasets.dataset_tools import getDatasets
import hist
import math
import os


parser.add_argument("--skipAngularCoeffs", action='store_true', help="Skip the conversion of helicity moments to angular coeff fractions")
parser.add_argument("--singleLeptonHists", action='store_true', help="Also store single lepton kinematics")
parser.add_argument("--photonHists", action='store_true', help="Also store photon kinematics")
parser.add_argument("--skipEWHists", action='store_true', help="Also store histograms for EW reweighting. Use with --filter horace")
parser.add_argument("--absY", action='store_true', help="use absolute |Y|")
parser.add_argument("--applySelection", action='store_true', help="Apply selection on leptons")
parser.add_argument("--auxiliaryHistograms", action="store_true", help="Safe auxiliary histograms (mainly for ew analysis)")

parser = common.set_parser_default(parser, "filterProcs", common.vprocs)
parser = common.set_parser_default(parser, "theoryCorr", [])

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

    isW = dataset.name.startswith("W") and dataset.name[1] not in ["W", "Z"] #in common.wprocs
    isZ = dataset.name.startswith("Z") and dataset.name[1] not in ["W", "Z"] #in common.zprocs

    weight_expr = "std::copysign(1.0, genWeight)"
    df = df.Define("weight", weight_expr)
    # This sum should happen before any change of the weight
    weightsum = df.SumAndCount("weight")

    if "reweight_h2" in dataset.name:
        weight_expr = f"{weight_expr}*H2BugFixWeight[0]"
    elif "NNLOPS" in dataset.name:
        weight_expr = f"{weight_expr}*LHEScaleWeightAltSet1[4]"

    if args.applySelection:
        if isZ:
            df = df.Filter("ewLeptons[0].pt()>25 && ewLeptons[1].pt()>25 && std::fabs(ewLeptons[0].eta())<3.5 && std::fabs(ewLeptons[1].eta())<3.5")
        elif isW:
            df = df.Filter("""(ewLeptons[0].mass() == 0 || (ewLeptons[0].pt()>25 && std::fabs(ewLeptons[0].eta())<3.5))
                           && (ewLeptons[1].mass() == 0 || (ewLeptons[1].pt()>25 && std::fabs(ewLeptons[1].eta())<3.5))""")

    df = theory_tools.define_theory_weights_and_corrs(df, dataset.name, corr_helpers, args)

    if isZ:
        nominal_axes = [axis_massZgen, axis_rapidity, axis_ptVgen, axis_chargeZgen]
        lep_axes = [axis_l_eta_gen, axis_l_pt_gen, axis_chargeZgen]
    else:
        nominal_axes = [axis_massWgen, axis_rapidity, axis_ptVgen, axis_chargeWgen]
        lep_axes = [axis_l_eta_gen, axis_l_pt_gen, axis_chargeWgen]

    nominal_cols = ["massVgen", col_rapidity, "ptVgen", "chargeVgen"]
    lep_cols = ["etaPrefsrLep", "ptPrefsrLep", "chargeVgen"]

    if args.singleLeptonHists and (isW or isZ):
        if isW:
            df = df.Define('ptPrefsrLep', 'genl.pt()')
            df = df.Define('etaPrefsrLep', 'genl.eta()')
        else:
            df = df.Define('ptPrefsrLep', 'genlanti.pt()')
            df = df.Define('etaPrefsrLep', 'genlanti.eta()')
        results.append(df.HistoBoost("nominal_genlep", lep_axes, [*lep_cols, "nominal_weight"], storage=hist.storage.Double()))

    if not args.skipEWHists and (isW or isZ):
        if isZ:
            massBins = theory_tools.make_ew_binning(mass = 91.1535, width = 2.4932, initialStep=0.010, bin_edges_low=[0,50,60], bin_edges_high=[120])
        else:
            massBins = theory_tools.make_ew_binning(mass = 80.3815, width = 2.0904, initialStep=0.010)
        axis_ewMll = hist.axis.Variable(massBins, name = "ewMll", underflow=False)
        axis_ewLogDeltaM = hist.axis.Regular(90, -5, 4, name = "ewLogDeltaM")
        axis_ewPtll = hist.axis.Variable(common.ptV_binning, underflow=False, name = "ewPTll") # hist.axis.Regular(100, 0, 100, name = "ewPtll")

        results.append(df.HistoBoost("nominal_ew", [axis_ewMll, axis_ewLogDeltaM], ['ewMll', 'ewLogDeltaM', "nominal_weight"], storage=hist.storage.Weight()))
        results.append(df.HistoBoost("nominal_ewMllPTll", [axis_ewMll, axis_ewPtll], ["ewMll", "ewPTll", "nominal_weight"], storage=hist.storage.Weight()))
        if args.auxiliaryHistograms:
            axis_ewMlly = hist.axis.Variable(massBins, name = "ewMlly")
            results.append(df.HistoBoost("nominal_ewMlly", [axis_ewMlly], ["ewMlly", "nominal_weight"], storage=hist.storage.Weight()))
            # coarse binning
            axis_Mll = hist.axis.Regular(100, 50, 150, name = "Mll")
            results.append(df.HistoBoost("nominal_Mll", [axis_Mll], ["ewMll", "nominal_weight"], storage=hist.storage.Weight()))
            axis_Mlly = hist.axis.Regular(100, 50, 150, name = "Mlly")
            results.append(df.HistoBoost("nominal_Mlly", [axis_Mlly], ["ewMlly", "nominal_weight"], storage=hist.storage.Weight()))

            axis_PTll =  hist.axis.Regular(100, 0, 100, name = "PTll")
            axis_PTlly = hist.axis.Regular(100, 0, 100, name = "PTlly")
            axis_Yll =  hist.axis.Regular(100, -5, 5, name = "Yll")
            axis_Ylly = hist.axis.Regular(100, -5, 5, name = "Ylly")
            results.append(df.HistoBoost("nominal_PTll",  [axis_PTll],  ["ewPTll", "nominal_weight"], storage=hist.storage.Weight()))
            results.append(df.HistoBoost("nominal_PTlly", [axis_PTlly], ["ewPTlly", "nominal_weight"], storage=hist.storage.Weight()))
            results.append(df.HistoBoost("nominal_Yll",  [axis_Yll],  ["ewYll", "nominal_weight"], storage=hist.storage.Weight()))
            results.append(df.HistoBoost("nominal_Ylly", [axis_Ylly], ["ewYlly", "nominal_weight"], storage=hist.storage.Weight()))

            # single lepton hists
            if args.singleLeptonHists:
                if isZ:
                    # first lepton is leading in pT
                    df = df.Define('ewLepPt1', 'ewLeptons[0].pt()')
                    df = df.Define('ewLepPt2', 'ewLeptons[1].pt()')
                    df = df.Define('ewLepEta1', 'ewLeptons[0].eta()')
                    df = df.Define('ewLepEta2', 'ewLeptons[1].eta()')
                if isW:
                    # first lepton is charged
                    df = df.Define('ewLepPt1', 'ewLeptons[0].mass() == 0 ? ewLeptons[1].pt() : ewLeptons[0].pt()')
                    df = df.Define('ewLepPt2', 'ewLeptons[0].mass() == 0 ? ewLeptons[0].pt() : ewLeptons[1].pt()')
                    df = df.Define('ewLepEta1', 'ewLeptons[0].mass() == 0 ? ewLeptons[1].eta() : ewLeptons[0].eta()')
                    df = df.Define('ewLepEta2', 'ewLeptons[0].mass() == 0 ? ewLeptons[0].eta() : ewLeptons[1].eta()')
                
                axis_ewLepPt = hist.axis.Regular(100, 0, 100, name = "pt")
                results.append(df.HistoBoost("nominal_ewLepPt1", [axis_ewLepPt], ["ewLepPt1", "nominal_weight"], storage=hist.storage.Weight()))
                results.append(df.HistoBoost("nominal_ewLepPt2", [axis_ewLepPt], ["ewLepPt2", "nominal_weight"], storage=hist.storage.Weight()))
                axis_ewLepEta = hist.axis.Regular(100, -5, 5, name = "eta")
                results.append(df.HistoBoost("nominal_ewLepEta1", [axis_ewLepEta], ["ewLepEta1", "nominal_weight"], storage=hist.storage.Weight()))
                results.append(df.HistoBoost("nominal_ewLepEta2", [axis_ewLepEta], ["ewLepEta2", "nominal_weight"], storage=hist.storage.Weight()))

            if args.photonHists:
                # photon distributions
                df = df.Define('nPhotons', 'ewPhotons.size()')
                df = df.Define('leadPhotonPt', 'ewPhotons.size() > 0 ? log10(ewPhotons[0].pt()) : -99')
                df = df.Define('leadPhotonEta', 'ewPhotons.size() > 0 ? ewPhotons[0].eta() : -99')
                df = df.Define('sublPhotonPt', 'ewPhotons.size() > 1 ? log10(ewPhotons[1].pt()) : -99')
                df = df.Define('sublPhotonEta', 'ewPhotons.size() > 1 ? ewPhotons[1].eta() : -99')
                df = df.Define('trailPhotonPt', 'ewPhotons.size() > 2 ? log10(ewPhotons[2].pt()) : -99')
                df = df.Define('trailPhotonEta', 'ewPhotons.size() > 2 ? ewPhotons[2].eta() : -99')

                axis_ewNPhotons = hist.axis.Regular(5, 0, 5, name = "n")
                results.append(df.HistoBoost("nominal_ewPhotons", [axis_ewNPhotons], ["nPhotons", "nominal_weight"], storage=hist.storage.Weight()))

                axis_photonPt = hist.axis.Regular(100, -5, 5, name = "pt")
                axis_photonEta = hist.axis.Regular(100, -5, 5, name = "eta")
                results.append(df.HistoBoost("nominal_leadPhoton", [axis_photonPt, axis_photonEta], ["leadPhotonPt", "leadPhotonEta", "nominal_weight"], storage=hist.storage.Weight()))
                results.append(df.HistoBoost("nominal_sublPhoton", [axis_photonPt, axis_photonEta], ["sublPhotonPt", "sublPhotonEta", "nominal_weight"], storage=hist.storage.Weight()))
                results.append(df.HistoBoost("nominal_trailPhoton", [axis_photonPt, axis_photonEta], ["trailPhotonPt", "trailPhotonEta", "nominal_weight"], storage=hist.storage.Weight()))

    nominal_gen = df.HistoBoost("nominal_gen", nominal_axes, [*nominal_cols, "nominal_weight"], storage=hist.storage.Double())

    results.append(nominal_gen)
    if not 'horace' in dataset.name and not 'winhac' in dataset.name:
        if "LHEScaleWeight" in df.GetColumnNames():
            df = theory_tools.define_scale_tensor(df)
            syst_tools.add_qcdScale_hist(results, df, nominal_axes, nominal_cols, "nominal_gen")
            df = df.Define("helicity_moments_scale_tensor", "wrem::makeHelicityMomentScaleTensor(csSineCosThetaPhi, scaleWeights_tensor, nominal_weight)")
            helicity_moments_scale = df.HistoBoost("helicity_moments_scale", nominal_axes, [*nominal_cols, "helicity_moments_scale_tensor"], tensor_axes = [wremnants.axis_helicity, *wremnants.scale_tensor_axes], storage=hist.storage.Double())
            results.append(helicity_moments_scale)

        if "LHEPdfWeight" in df.GetColumnNames():
            syst_tools.add_pdf_hists(results, df, dataset.name, nominal_axes, nominal_cols, args.pdfs, "nominal_gen")

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
output_tools.write_analysis_output(resultdict, f"{os.path.basename(__file__).replace('py', 'hdf5')}", args, update_name=not args.forceDefaultName)

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
