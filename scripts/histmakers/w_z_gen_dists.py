from utilities import boostHistHelpers as hh, common, logging
from utilities.io_tools import output_tools

parser,initargs = common.common_parser()

import narf
import wremnants
from wremnants import theory_tools,syst_tools,theory_corrections,unfolding_tools
from wremnants.datasets.dataset_tools import getDatasets
import hist
import math
import os
import numpy as np
from utilities.differential import get_theoryAgnostic_axes
from wremnants.datasets.datagroups import Datagroups

parser.add_argument("--skipAngularCoeffs", action='store_true', help="Skip the conversion of helicity moments to angular coeff fractions")
parser.add_argument("--propagatePDFstoHelicity", action='store_true', help="Propagate PDF uncertainties to helicity moments")
parser.add_argument("--useTheoryAgnosticBinning", action='store_true', help="Use theory agnostic binning (coarser) to produce the gen results")
parser.add_argument("--singleLeptonHists", action='store_true', help="Also store single lepton kinematics")
parser.add_argument("--photonHists", action='store_true', help="Also store photon kinematics")
parser.add_argument("--skipEWHists", action='store_true', help="Also store histograms for EW reweighting. Use with --filter horace")
parser.add_argument("--signedY", action='store_true', help="use signed Y")
parser.add_argument("--fiducial", choices=["masswindow", "dilepton", "singlelep"], help="Apply selection on leptons (No argument for inclusive)")
parser.add_argument("--auxiliaryHistograms", action="store_true", help="Safe auxiliary histograms (mainly for ew analysis)")
parser.add_argument("--ptqVgen", action='store_true', help="To store qt by Q variable instead of ptVgen, GEN only ", default=None)
parser.add_argument("--helicity", action='store_true', help="Make qcdScaleByHelicity hist")

parser = common.set_parser_default(parser, "filterProcs", common.vprocs)
parser = common.set_parser_default(parser, "theoryCorr", [])
parser = common.set_parser_default(parser, "ewTheoryCorr", [])

args = parser.parse_args()

analysis_label = Datagroups.analysisLabel(os.path.basename(__file__))
logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

datasets = getDatasets(maxFiles=args.maxFiles,
                        filt=args.filterProcs,
                        excl=args.excludeProcs,
                        extended = "msht20an3lo" not in args.pdfs,
                        nanoVersion="v9", base_path=args.dataPath, mode=analysis_label)

logger.debug(f"Will process samples {[d.name for d in datasets]}")

axis_massWgen = hist.axis.Variable([4., 13000.], name="massVgen", underflow=True, overflow=False)

axis_massZgen = hist.axis.Regular(12, 60., 120., name="massVgen")

theoryAgnostic_axes, _ = get_theoryAgnostic_axes()
axis_ptV_thag = theoryAgnostic_axes[0]
axis_yV_thag = theoryAgnostic_axes[1]

if not args.useTheoryAgnosticBinning:
    axis_absYVgen = hist.axis.Variable(
        [0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 4., 5.], # this is the same binning as hists from theory corrections
        name = "absYVgen", underflow=False
    )
else:
    axis_absYVgen = hist.axis.Variable(
        axis_yV_thag.edges, #same axis as theory agnostic norms
        name = "absYVgen", underflow=False
    )

axis_ygen = hist.axis.Regular(10, -5., 5., name="y")
axis_rapidity = axis_ygen if args.signedY else axis_absYVgen
col_rapidity =  "yVgen" if args.signedY else "absYVgen"

if not args.useTheoryAgnosticBinning:
    axis_ptVgen = hist.axis.Variable(
    (*common.get_dilepton_ptV_binning(fine=False), 13000.),
    name = "ptVgen", underflow=False,
)
else:
    axis_ptVgen = hist.axis.Variable(
    axis_ptV_thag.edges, #same axis as theory agnostic norms, 
    #common.ptV_binning,
    name = "ptVgen", underflow=False,
)

axis_ptqVgen = hist.axis.Variable(
    [round(x, 4) for x in list(np.arange(0, 0.1 + 0.0125, 0.0125))]+[round(x, 4) for x in list(np.arange(0.1+0.025, 0.5 + 0.025, 0.025))], 
    name = "ptqVgen", underflow=False
)

axis_chargeWgen = hist.axis.Regular(
    2, -2, 2, name="chargeVgen", underflow=False, overflow=False
)

axis_chargeZgen = hist.axis.Integer(
    0, 1, name="chargeVgen", underflow=False, overflow=False
)

axis_absetal_gen = hist.axis.Regular(24, 0, 2.4, name = "abseta")
axis_ptl_gen = hist.axis.Regular(34, 26., 60., name = "pt")

theory_corrs = [*args.theoryCorr, *args.ewTheoryCorr]
corr_helpers = theory_corrections.load_corr_helpers(common.vprocs, theory_corrs)

def build_graph(df, dataset):
    logger.info("build graph")
    logger.info(dataset.name)
    results = []
    
    if dataset.is_data:
        raise RuntimeError("Running GEN analysis over data is not supported")

    isW = dataset.name.startswith("W") and dataset.name[1] not in ["W", "Z"] #in common.wprocs
    isZ = dataset.name.startswith("Z") and dataset.name[1] not in ["W", "Z"] #in common.zprocs

    weight_expr = "std::copysign(1.0, genWeight)"

    if "reweight_h2" in dataset.name:
        weight_expr = f"{weight_expr}*H2BugFixWeight[0]"
    elif "NNLOPS" in dataset.name:
        weight_expr = f"{weight_expr}*LHEScaleWeightAltSet1[4]"

    df = df.Define("weight", weight_expr)
    df = df.DefinePerSample("unity","1.")
    # This sum should happen before any change of the weight
    weightsum = df.SumAndCount("weight")
    df = df.Define("isEvenEvent", "event % 2 == 0")

    df = theory_tools.define_theory_weights_and_corrs(df, dataset.name, corr_helpers, args)

    if isZ:
        nominal_axes = [axis_massZgen, axis_rapidity, axis_ptqVgen if args.ptqVgen else axis_ptVgen, axis_chargeZgen]
        lep_axes = [axis_absetal_gen, axis_ptl_gen, axis_chargeZgen]
    else:
        nominal_axes = [axis_massWgen, axis_rapidity, axis_ptqVgen if args.ptqVgen else axis_ptVgen, axis_chargeWgen]
        lep_axes = [axis_absetal_gen, axis_ptl_gen, axis_chargeWgen]
    nominal_cols = ["massVgen", col_rapidity, "ptqVgen" if args.ptqVgen else "ptVgen", "chargeVgen"]
    lep_cols = ["absEtaGen", "ptGen", "chargeVgen"]

    mode = f'{"z" if isZ else "w"}_{analysis_label}'
    if args.fiducial is not None:
        if isZ and args.fiducial == "singlelep":
            mode += "_wlike"

        df = unfolding_tools.define_gen_level(df, "preFSR", dataset.name, mode=mode)
        df = unfolding_tools.select_fiducial_space(df, mode=mode, fiducial=args.fiducial, unfolding=True)

    if args.singleLeptonHists and (isW or isZ):
        results.append(df.HistoBoost("nominal_genlep", lep_axes, [*lep_cols, "nominal_weight"], storage=hist.storage.Weight()))

    if not args.skipEWHists and (isW or isZ):
        if isZ:
            massBins = theory_tools.make_ew_binning(mass = 91.1535, width = 2.4932, initialStep=0.010, bin_edges_low=[0,50,60], bin_edges_high=[120])
        else:
            massBins = theory_tools.make_ew_binning(mass = 80.3815, width = 2.0904, initialStep=0.010)
        
        # pre FSR
        axis_genMV = hist.axis.Variable(massBins, name = "massVgen", underflow=False)
        axis_genPtV = hist.axis.Variable(common.ptV_binning, underflow=False, name = "ptVgen") 
        axis_genAbsYV = hist.axis.Regular(50, 0, 5, name = "absYVgen")
        results.append(df.HistoBoost("preFSR_massVptV", [axis_genMV, axis_genPtV], ["massVgen", "ptVgen", "nominal_weight"], storage=hist.storage.Weight()))
        results.append(df.HistoBoost("preFSR_absYVptV", [axis_genAbsYV, axis_genPtV], ["absYVgen", "ptVgen", "nominal_weight"], storage=hist.storage.Weight()))
        results.append(df.HistoBoost("preFSR_absYVmassV", [axis_genAbsYV, axis_genMV], ["absYVgen", "massVgen", "nominal_weight"], storage=hist.storage.Weight()))

        # post FSR, pre tau decay
        axis_ewMll = hist.axis.Variable(massBins, name = "ewMll", underflow=False)
        axis_ewPtll = hist.axis.Variable(common.ptV_binning, underflow=False, name = "ewPTll") 
        axis_ewAbsYll = hist.axis.Regular(50, 0, 5, name = "ewAbsYll")
        results.append(df.HistoBoost("ew_MllPTll", [axis_ewMll, axis_ewPtll], ["ewMll", "ewPTll", "nominal_weight"], storage=hist.storage.Weight()))
        results.append(df.HistoBoost("ew_YllPTll", [axis_ewAbsYll, axis_ewPtll], ["ewAbsYll", "ewPTll", "nominal_weight"], storage=hist.storage.Weight()))
        results.append(df.HistoBoost("ew_YllMll", [axis_ewAbsYll, axis_ewMll], ["ewAbsYll", "ewMll", "nominal_weight"], storage=hist.storage.Weight()))

        # dressed
        axis_ewMll = hist.axis.Variable(massBins, name = "ewMll", underflow=False)
        axis_ewPtll = hist.axis.Variable(common.ptV_binning, underflow=False, name = "ewPTll") 
        axis_ewAbsYll = hist.axis.Regular(50, 0, 5, name = "ewAbsYll")
        df = theory_tools.define_dressed_vars(df, mode=mode)
        results.append(df.HistoBoost("dressed_MllPTll", [axis_ewMll, axis_ewPtll], ["dressed_MV", "dressed_PTV", "nominal_weight"], storage=hist.storage.Weight()))
        results.append(df.HistoBoost("dressed_YllPTll", [axis_ewAbsYll, axis_ewPtll], ["dressed_absYV", "dressed_PTV", "nominal_weight"], storage=hist.storage.Weight()))
        results.append(df.HistoBoost("dressed_YllMll", [axis_ewAbsYll, axis_ewMll], ["dressed_absYV", "dressed_MV", "nominal_weight"], storage=hist.storage.Weight()))

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

                # postfsr definition
                axis_eta = hist.axis.Regular(25, 0, 2.5, name = "postfsrLep_absEta", overflow=True, underflow=False)
                axis_pt = hist.axis.Regular(50, 20, 70, name = "postfsrLep_pt", overflow=True, underflow=True)
                results.append(df_fiducial.HistoBoost("nominal_postfsr", [axis_eta, axis_pt], ["postfsrLep_absEta", "postfsrLep_pt", "nominal_weight"], storage=hist.storage.Weight()))

    if 'powheg' in dataset.name:
        return results, weightsum

    nominal_gen = df.HistoBoost("nominal_gen", nominal_axes, [*nominal_cols, "nominal_weight"], storage=hist.storage.Weight())
    results.append(nominal_gen)

    if 'horace' not in dataset.name and 'winhac' not in dataset.name and \
            "LHEScaleWeight" in df.GetColumnNames() and "LHEPdfWeight" in df.GetColumnNames():

        qcdScaleByHelicity_helper = wremnants.theory_corrections.make_qcd_uncertainty_helper_by_helicity(is_w_like = dataset.name[0] != "W") if args.helicity else None

        df = syst_tools.add_theory_hists(results, df, args, dataset.name, corr_helpers, qcdScaleByHelicity_helper, nominal_axes, nominal_cols, base_name="nominal_gen")

    return results, weightsum

resultdict = narf.build_and_run(datasets, build_graph)
output_tools.write_analysis_output(resultdict, f"{os.path.basename(__file__).replace('py', 'hdf5')}", args)

logger.info("computing angular coefficients")
z_moments = None
w_moments = None

if not args.skipAngularCoeffs:
    for dataset in datasets:
        name = dataset.name
        if "nominal_gen_helicity_moments_scale" not in resultdict[name]["output"]:
            logger.warning(f"Failed to find helicity_moments_scale hist for proc {name}. Skipping!")
            continue
        moments = resultdict[name]["output"]["nominal_gen_helicity_moments_scale"].get()
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

    moments_out={}
    # Common.ptV_binning is the approximate 5% quantiles, rounded to integers. Rebin for approx 10% quantiles
    if z_moments:
        if not args.useTheoryAgnosticBinning:
            z_moments = hh.rebinHist(z_moments, axis_ptVgen.name, common.ptV_binning[::2])
            z_moments = hh.rebinHist(z_moments, axis_massZgen.name, axis_massZgen.edges[::2])
        moments_out["Z"] = z_moments
    if w_moments:
        if not args.useTheoryAgnosticBinning:
            w_moments = hh.rebinHist(w_moments, axis_ptVgen.name, common.ptV_binning[::2])
        moments_out["W"] = w_moments
    if moments_out:
        outfname = "w_z_moments"
        if args.signedY:
            outfname += "_signedY"
        if args.useTheoryAgnosticBinning:
            outfname += "_theoryAgnosticBinning"
        outfname += ".hdf5"
        output_tools.write_analysis_output(moments_out, outfname, args)
