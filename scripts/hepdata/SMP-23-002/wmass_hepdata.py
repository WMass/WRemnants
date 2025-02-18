#!/usr/bin/env python3
import json
import os
import sys

from hepdata_lib import RootFileReader, Submission, Table, Uncertainty, Variable
from hepdata_lib.helpers import round_value_and_uncertainty

from scripts.analysisTools.plotUtils.utility import (
    common_plot_parser,  # TODO: move to main common parser
)
from utilities.common import base_dir, data_dir
from utilities.io_tools import conversion_tools, input_tools
from wums import logging


def getFitInfo():

    matrix_dict = {
        "W": {
            "file": "W/impacts_W_all_massShiftW100MeV_noi.root",
            "hist": "nuisanceInfo",
            "desc": "W boson mass fit",
            "covFile": "/scratch/shared/mw/combine_studies/mw_unblinding/WMass_eta_pt_charge/fitresults_123456789_data.root",
            "covHist": "covariance_matrix_channelmu",
        },
        "W_massDiffCharge": {
            "file": "W/impacts_W_all_massDiffCharge_massDiffChargeW50MeV_noi.root",
            "hist": "nuisanceInfo",
            "desc": "W boson mass fit (charge difference)",
            "covFile": "/scratch/shared/mw/combine_studies/mw_unblinding/WMass_eta_pt_charge_massDiffCharge/fitresults_123456789_data.root",
            "covHist": "covariance_matrix_channelmu",
        },
        "Zwlike": {
            "file": "Zwlike/impacts_Zwlike_all_massShiftZ100MeV_noi.root",
            "hist": "nuisanceInfo",
            "desc": "W-like Z boson mass fit",
            "covFile": "/scratch/shared/mw/combine_studies/mz_wlike_unblinding/ZMassWLike_eta_pt_charge/fitresults_123456789.root",
            "covHist": "covariance_matrix_channelmu",
        },
        "Zwlike_flipEventSplit": {
            "file": "Zwlike/impacts_Zwlike_all_flipEventNumberSplitting_massShiftZ100MeV_noi.root",
            "hist": "nuisanceInfo",
            "desc": "W-like Z boson mass fit (flip odd/even events to choose muon charge)",
            "covFile": "/scratch/shared/mw/combine_studies/mz_wlike_unblinding/ZMassWLike_eta_pt_charge_flipEventNumberSplitting/fitresults_123456789.root",
            "covHist": "covariance_matrix_channelmu",
        },
        "Zwlike_massDiffCharge": {
            "file": "Zwlike/impacts_Zwlike_all_fitMassDiff_charge_massDiffChargeZ50MeV_noi.root",
            "hist": "nuisanceInfo",
            "desc": "W-like Z boson mass fit (charge difference)",
            "covFile": "/scratch/shared/mw/combine_studies/mz_wlike_unblinding/ZMassWLike_eta_pt_charge_fitMassDiff_charge/fitresults_123456789.root",
            "covHist": "covariance_matrix_channelmu",
        },
        "Zdilepton": {
            "file": "Zdilepton/impacts_Zdilepton_all_massShiftZ100MeV_noi.root",
            "hist": "nuisanceInfo",
            "desc": "Z boson mass fit (using dimuon mass)",
            "covFile": "/scratch/shared/mw/combine_studies/mz_dilepton_unblinding/ZMassDilepton_mll_etaAbsEta_inclusive/fitresults_123456789.root",
            "covHist": "covariance_matrix_channelmu",
            "globImp": False,
        },
        "W_helFit": {
            "file": "W_helicityFit/impacts_W_all_helicityFit_massShiftW100MeV_noi.root",
            "hist": "nuisanceInfo",
            "desc": "W boson mass fit (helicity cross section fit)",
            "covFile": "/scratch/shared/mw/combine_studies/data_new.root",
            "covHist": "covariance_matrix_channelmu",
            "globImp": False,
        },
    }

    return matrix_dict


def make_covariance_matrix(logger=None, verbose=4):

    if logger == None:
        logger = logging.setup_logger(os.path.basename(__file__), verbose)

    matrix_dict = getFitInfo()

    convert_hepdata_json = (
        base_dir + "/utilities/styles/nuisance_translate_hepdata.json"
    )
    logger.warning(
        f"Using file {convert_hepdata_json} to convert names of nuisance parameters for HEPData"
    )
    with open(convert_hepdata_json) as hepf:
        translate_label_hepdata = json.load(hepf)

    tmp_outdir = "./tmp_material/covariance_matrix/"
    if not os.path.exists(tmp_outdir):
        os.makedirs(tmp_outdir)

    for im, m in enumerate(matrix_dict.keys()):

        matrix = matrix_dict[m]
        filename = matrix["covFile"]
        histname = matrix["covHist"]
        desc = matrix["desc"]

        logger.info("=" * 30)
        logger.info(f"{im}) {m}: {desc}")
        logger.info("=" * 30)

        # could use load_covariance_pois in utilities/io_tools/combinetf_input.py
        # which returns covariance as boost hist and a list with rows' labels
        infile = input_tools.safeOpenRootFile(filename)
        th2 = input_tools.safeGetRootObject(infile, histname, detach=True)
        infile.Close()

        logger.info("Histogram read from file, converting labels ...")
        # change labels of histogram using Latex
        nRows = th2.GetNbinsX()
        for i in range(1, 1 + nRows):
            label = th2.GetXaxis().GetBinLabel(i)
            newlabel = conversion_tools.get_hepdata_label(
                label, input_dict=translate_label_hepdata
            )
            th2.GetXaxis().SetBinLabel(i, newlabel)
            th2.GetYaxis().SetBinLabel(i, newlabel)
            if i % 100 == 0:
                sys.stdout.write(" {0:.2%}     \r".format(float(i) / nRows))
                sys.stdout.flush()

        th2.SetTitle(
            f"Covariance matrix for {desc}: mass uncertainty to be multiplied by 100 MeV"
        )
        houtputname = "covariance_matrix"
        tag = m.lower()
        outname = f"{tmp_outdir}covariance_{tag}.root"
        outfile = input_tools.safeOpenRootFile(outname, mode="recreate")
        th2.Write(houtputname)
        outfile.Close()
        logger.info(f"Histogram {houtputname} saved in file {outname}")


def make_pulls_and_constraints(sub, logger=None, verbose=4):

    if logger == None:
        logger = logging.setup_logger(os.path.basename(__file__), verbose)

    # TODO: use a common path
    basePath = data_dir + "/hepdata/SMP_23_002/"

    # tag name : {file name, matrix name inside file}
    # for mass diff there are two POIs, so two sets of impacts, take only those for the actual diff POI
    matrix_dict = getFitInfo()
    centerOfMassEnergy = 13000

    for im, m in enumerate(matrix_dict.keys()):

        boson = "W" if m.startswith("W") else "Z"
        general_hepdata_phrases = f"Proton-Proton Scattering, {boson} Production"

        digits = (
            6  # number of significant digits for numbers, ideally as large as possible
        )

        desc = matrix_dict[m]["desc"]
        filename = basePath + matrix_dict[m]["file"]
        histname = matrix_dict[m]["hist"]
        hasGlobalImp = True
        if "globImp" in matrix_dict[m].keys():
            hasGlobalImp = matrix_dict[m]["globImp"]

        impactsText = "both nominal and 'global'"
        if not hasGlobalImp:
            impactsText = "only nominal"
            logger.warning(f"Skipping global impacts for {desc}")

        logger.warning(
            f"Pulls/constraints/impacts for {desc}, stored with {digits} significant digits"
        )

        logger.info("Preparing 1 table ...")
        table = Table(f"Postfit_pulls_constraints_impacts_{m}")
        table.location = "No corresponding figure in the paper"
        table.description = f"Postfit pulls, constraints, and impacts ({impactsText}) for all nuisance parameters in the {desc}, sorted by the absolute value of the nominal impact."
        table.keywords["phrases"] = [general_hepdata_phrases]
        table.keywords["cmenergies"] = [centerOfMassEnergy]
        table.keywords["reactions"] = [f"P P --> {boson} X"]
        # Remember to update the links
        linkPublicPage = "https://cms-results.web.cern.ch/cms-results/public-results/publications/SMP-23-002/covMat/"
        tag = m.lower()
        linkCovMatrix = f"{linkPublicPage}covariance_{tag}.root"

        logger.warning("Adding a link to the covariance matrix.")
        table.add_additional_resource(
            "Covariance matrix for the measured mass parameter(s) of interest (POI) and nuisance parameters (NP). The variance of the POI is given with respect to a 100 MeV prefit uncertainty. Therefore, the square root of the corresponding element in the diagonal should be multiplied by 100 to get the actual uncertainty of the measurement in MeV (similarly, covariance cells between the POI and any given NP should be multiplied by 100)",
            linkCovMatrix,
            copy_file=False,
        )

        # open root file to get histogram and retrieve bin labels
        infile = input_tools.safeOpenRootFile(filename)
        th2 = input_tools.safeGetRootObject(infile, histname, detach=True)
        infile.Close()
        logger.info("Histogram read from file ...")

        nNuisParams = th2.GetNbinsX()
        indep_var = Variable(
            "Nuisance parameter name", is_independent=True, is_binned=False, units=""
        )

        # adding info
        impactNominal = Variable(
            "Impact (nominal)", is_independent=False, is_binned=False, units="MeV"
        )
        impactNominal.digits = digits
        impactNominal.values = []
        pull = Variable("Pull", is_independent=False, is_binned=False, units="")
        pull.digits = digits
        pull.values = []
        constraint = Variable(
            "Constraint", is_independent=False, is_binned=False, units=""
        )
        constraint.digits = digits
        constraint.values = []
        impactGlobal = Variable(
            "Impact (global)", is_independent=False, is_binned=False, units="MeV"
        )
        impactGlobal.digits = digits
        impactGlobal.values = []

        # sort in reversed order
        for i in reversed(range(1, 1 + nNuisParams)):
            latexLabel = th2.GetXaxis().GetBinLabel(i)
            indep_var.values.append(latexLabel)
            # todo: improve hardcoded sorting of these pieces
            impactNominal.values.append(th2.GetBinContent(i, 1))
            pull.values.append(th2.GetBinContent(i, 2))
            constraint.values.append(th2.GetBinContent(i, 3))
            impactGlobal.values.append(th2.GetBinContent(i, 4))

        # fill table (for the dependent variables use pulls first)
        table.add_variable(indep_var)
        table.add_variable(pull)
        table.add_variable(constraint)
        table.add_variable(impactNominal)
        if hasGlobalImp:
            table.add_variable(impactGlobal)

        # Now add table to submission
        logger.info("Adding table to submission file ...")
        sub.add_table(table)
        logger.info("Done with 1 table ...")


def make_cross_section(sub, logger=None, verbose=4):

    if logger == None:
        logger = logging.setup_logger(os.path.basename(__file__), verbose)

    # TODO: use a common path
    basePath = data_dir + "hepdata/SMP_23_002/"
    input_dict = {
        "W_pt": {
            "file": "W/ptVgen_postfit_W_RecoPtll_ptllyll_PrefitRatio.root",
            "desc": "Generator-level $\\mathit{p}_{T}^{W}$ distribution and uncertainty. The last column reports the result of a simultaneous fit to the single muon $(\\mathit{p}_{T}^{μ}, \\mathit{\\eta}^{μ}, \\mathit{q}^{μ})$ distribution in $W$ boson decays and the dimuon $(\\mathit{p}_{T}^{μμ},\\mathit{y}^{μμ})$ distribution in $Z$ boson events.",
            "observable": "DSIG/DPT",
            "isDividedByBinWidth": True,
            "addImage": True,
            "reference": "Fig A12",
        },
        ##
        "Zwlike_y": {
            "file": "Zwlike/absYVgen_postfit_Wlike_RecoPtll_ptllyll_PrefitRatio.root",
            "desc": "Unfolded measured $\\mathit{y}^{Z}$ distribution compared with the generator-level predictions before the likelihood fit or after the W-like $\\mathit{m}_{Z}$ fit or from the direct fit to the $(\\mathit{p}_{T}^{μμ},\\mathit{y}^{μμ})$ distribution.",
            "observable": "DSIG/DYRAP",
            "isDividedByBinWidth": True,
            "addImage": True,
            "reference": "Fig xx in supplementary material, and Fig A9 in paper",
        },
        ##
        "Zwlike_pt": {
            "file": "Zwlike/ptVgen_postfit_Wlike_RecoPtll_ptllyll_PrefitRatio.root",
            "desc": "Unfolded measured $\\mathit{p}_{T}^{Z}$ distribution compared with the generator-level predictions before the likelihood fit or after the W-like $\\mathit{m}_{Z}$ fit or from the direct fit to the $(\\mathit{p}_{T}^{μμ},\\mathit{y}^{μμ})$ distribution.",
            "observable": "DSIG/DPT",
            "isDividedByBinWidth": True,
            "addImage": True,
            "reference": "Fig 2 and A9",
        },
        ##
        "W_pt_helFit": {
            "file": "W/ptVgen_postfit_W_PrefitRatio_helicityFitOnly.root",
            "desc": "Generator-level $\\mathit{p}_{T}^{W}$ distribution and uncertainty, before and after running the helicity fit.",
            "observable": "DSIG/DPT",
            "isDividedByBinWidth": True,
            "addImage": True,
            "reference": "Fig A14a",
        },
        ##
        "W_y_helFit": {
            "file": "W/absYVgen_postfit_W_PrefitRatio_helicityFitOnly.root",
            "desc": "Generator-level $\\mathit{y}^{Z}$ distribution and uncertainty, before and after running the helicity fit.",
            "observable": "DSIG/DYRAP",
            "isDividedByBinWidth": True,
            "addImage": True,
            "reference": "Fig A14b",
        },
    }

    centerOfMassEnergy = 13000

    for im, m in enumerate(input_dict.keys()):

        boson = "W" if m.startswith("W") else "Z"
        general_hepdata_phrases = (
            f"Proton-Proton Scattering, {boson} Production, Cross Section"
        )

        digits = (
            6  # number of significant digits for numbers, ideally as large as possible
        )

        desc = input_dict[m]["desc"]
        if input_dict[m]["isDividedByBinWidth"]:
            desc += " The measured cross section in each bin is reported after dividing the value by the corresponding bin width."

        filename = basePath + input_dict[m]["file"]
        obs = input_dict[m]["observable"]
        ref = input_dict[m]["reference"]

        logger.info("Preparing 1 table ...")

        # loop file to get list of keys
        # but then use the hepdata_lib helpers to retrieve all the information
        infile = input_tools.safeOpenRootFile(filename)
        hnames = []
        htitles = []
        xAxisTitle = ""
        yAxisTitle = ""
        for ikey, key in enumerate(infile.GetListOfKeys()):
            name = key.GetName()
            obj = key.ReadObj()
            hnames.append(name)
            htitles.append(obj.GetTitle())
            if ikey == 0:
                xAxisTitle = obj.GetXaxis().GetTitle()
                yAxisTitle = obj.GetYaxis().GetTitle()
        infile.Close()

        table = Table(f"diffXsec_{m}")
        table.location = f"{ref}"
        table.description = f"{yAxisTitle}: {desc}"
        table.keywords["observables"] = [obs]
        table.keywords["phrases"] = [general_hepdata_phrases]
        table.keywords["cmenergies"] = [centerOfMassEnergy]
        table.keywords["reactions"] = [f"P P --> {boson} X"]

        if input_dict[m]["addImage"]:
            image = filename.replace(".root", ".png")
            logger.warning(f"Storing image {image}")
            table.add_image(f"{image}")

        reader = RootFileReader(filename)
        hists = {n: reader.read_hist_1d(n) for n in hnames}
        for hn in hists.keys():
            round_value_and_uncertainty(hists[hn], "y", "dy", 2)

        indep_var = Variable(xAxisTitle, is_independent=True, is_binned=True, units="")
        indep_var.values = [
            (round(float(x[0]), 3), round(float(x[1]), 3))
            for x in hists[hnames[0]]["x_edges"]
        ]
        table.add_variable(indep_var)

        for i, n in enumerate(hnames):
            var = Variable(
                htitles[i], is_independent=False, is_binned=False, units="pb"
            )
            var.values = hists[n]["y"]
            unc = Uncertainty("Uncertainty", is_symmetric=True)
            unc.values = hists[n]["dy"]
            var.add_uncertainty(unc)
            table.add_variable(var)

        # Now add table to submission
        logger.info("Adding table to submission file ...")
        sub.add_table(table)
        logger.info("Done with 1 table ...")


def make_mass_summary(sub, logger=None, verbose=4):

    if logger == None:
        logger = logging.setup_logger(os.path.basename(__file__), verbose)

    # TODO: use a common path
    basePath = data_dir + "/hepdata/SMP_23_002/"
    input_dict = {
        "W_ptW_modeling": {
            "file": "W/Wmass_modeling_summary.root",
            "hist": "mass_summary",
            "desc": "Comparison of the nominal $\\mathit{m}_{W}$ measurement and its uncertainty (total or only from $\\mathit{p}_{T}^{W}$ modeling), with the alternative measurements using different approaches to the $\\mathit{p}_{T}^{W}$ modeling and its uncertainty.",
            "indepVarTitle": "$\\mathit{p}_{T}^{W}$ model",
            "observable": "M",
            "addImage": True,
            "reference": "Fig A12",
        },
        #
        "W_pdfUnscaled": {
            "file": "W/Wmass_pdf_summary_unscaled.root",
            "hist": "mass_summary",
            "desc": "Comparison of the nominal $\\mathit{m}_{W}$ measurement and its uncertainty (total or only from the CT18Z PDF set), with the alternative measurements using different PDFs and their uncertainty before the scaling procedure described in the paper text. The measured mass values are those reported in Table A.7 of the paper.",
            "indepVarTitle": "PDF set",
            "observable": "M",
            "addImage": True,
            "reference": "Fig A18a",
        },
        #
        "W_pdfScaled": {
            "file": "W/Wmass_pdf_summary_scaled.root",
            "hist": "mass_summary",
            "desc": "Comparison of the nominal $\\mathit{m}_{W}$ measurement and its uncertainty (total or only from the CT18Z PDF set), with the alternative measurements using different PDFs and their uncertainty after the scaling procedure described in the paper text. The measured mass values are those reported in Table A.7 of the paper. The scale factors for the PDF uncertainties from each set are reported in Table A.3 of the paper.",
            "indepVarTitle": "PDF set",
            "observable": "M",
            "addImage": True,
            "reference": "Fig A18b",
        },
    }

    centerOfMassEnergy = 13000

    for im, m in enumerate(input_dict.keys()):

        boson = "W" if m.startswith("W") else "Z"
        general_hepdata_phrases = (
            f"Proton-Proton Scattering, {boson} Production, Cross Section"
        )
        decimalDigits = 1

        desc = input_dict[m]["desc"]
        filename = basePath + input_dict[m]["file"]
        histname = input_dict[m]["hist"]
        obs = input_dict[m]["observable"]
        ref = input_dict[m]["reference"]

        logger.info("Preparing 1 table ...")

        infile = input_tools.safeOpenRootFile(filename)
        th2 = input_tools.safeGetRootObject(infile, histname, detach=True)
        infile.Close()
        xAxisTitle = th2.GetYaxis().GetBinLabel(1)
        if th2.GetNbinsY() == 4:
            desc += " The first column reports the measured mass for each configuration, while the second one shows the difference between the measurement and the main result."

        table = Table(f"mass_summary_{m}")
        table.location = f"{ref}"
        table.description = f"{desc}"
        table.keywords["observables"] = [obs]
        table.keywords["phrases"] = [general_hepdata_phrases]
        table.keywords["cmenergies"] = [centerOfMassEnergy]
        table.keywords["reactions"] = [f"P P --> {boson} X"]

        if input_dict[m]["addImage"]:
            image = filename.replace(".root", ".png")
            logger.warning(f"Storing image {image}")
            table.add_image(f"{image}")

        indep_var_title = input_dict[m]["indepVarTitle"]
        indep_var = Variable(
            indep_var_title, is_independent=True, is_binned=False, units=""
        )
        indep_var.values = [
            th2.GetXaxis().GetBinLabel(ix + 1) for ix in range(th2.GetNbinsX())
        ]
        table.add_variable(indep_var)

        # if there are 4 columns, the 4th is the actual mass and the first is the difference with other variations
        # better to write the actual mass as the first column
        endOffsetY = 0
        if th2.GetNbinsY() == 4:
            var_title = th2.GetYaxis().GetBinLabel(th2.GetNbinsY())
            # remove units, they  are added in the hepdata entry
            var_title = var_title.replace("(MeV)", "")
            var = Variable(
                var_title, is_independent=False, is_binned=False, units="MeV"
            )
            var.digits = 6  # apparently needed to avoid internal rounding of 80360.1 to 80360.0 (O.o)
            var.values = [
                round(th2.GetBinContent(ix + 1, th2.GetNbinsY()), decimalDigits)
                for ix in range(th2.GetNbinsX())
            ]
            table.add_variable(var)
            endOffsetY = 1
        for iy in range(th2.GetNbinsY() - endOffsetY):
            var_title = th2.GetYaxis().GetBinLabel(iy + 1)
            # remove units, they  are added in the hepdata entry
            var_title = var_title.replace("(MeV)", "")
            var = Variable(
                var_title, is_independent=False, is_binned=False, units="MeV"
            )
            var.values = [
                round(th2.GetBinContent(ix + 1, iy + 1), decimalDigits)
                for ix in range(th2.GetNbinsX())
            ]
            table.add_variable(var)

        # Now add table to submission
        logger.info("Adding table to submission file ...")
        sub.add_table(table)
        logger.info("Done with 1 table ...")


if __name__ == "__main__":

    parser = common_plot_parser()
    parser.add_argument(
        "-o",
        "--outdir",
        default="./test_SMP_23_002/",
        type=str,
        help="output directory",
    )
    parser.add_argument(
        "--skipCovariance",
        action="store_true",
        help="Skip production of covariance matrices (if they already exist, it saves some time)",
    )
    parser.add_argument(
        "--onlyCovariance",
        action="store_true",
        help="Make only covariance matrices",
    )
    args = parser.parse_args()

    logger = logging.setup_logger(
        os.path.basename(__file__), args.verbose, args.noColorLogger
    )

    outdir = args.outdir
    if not outdir.endswith("/"):
        outdir += "/"
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)

    logger.warning("\n\n")
    logger.warning(
        r"To save PNG images for some plots and convert them within hepdata_lib"
    )
    logger.warning(r"you need to work within the following singularity image")
    logger.warning(
        r"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling\:v38_patch0"
    )
    logger.warning("\n\n")

    if not args.skipCovariance:
        logger.warning("Preparing covariance matrices as root files")
        make_covariance_matrix(logger=logger, verbose=args.verbose)

    if not args.onlyCovariance:

        sub = Submission()

        # sub.add_additional_resource("Something",
        #                            "Object for something",
        #                            copy_file=True)
        sub.add_link(
            "Link to public page of the analysis with additional material",
            "https://cms-results.web.cern.ch/cms-results/public-results/publications/SMP-23-002/index.html",
        )

        logger.warning("Now saving pulls/constraints/impacts")
        make_pulls_and_constraints(sub, logger=logger, verbose=args.verbose)

        logger.warning("Now making some cross section plots")
        make_cross_section(sub, logger=logger, verbose=args.verbose)

        logger.warning("Now making some mass summary plots")
        make_mass_summary(sub, logger=logger, verbose=args.verbose)

        logger.info("Creating submission file ...")
        sub.create_files(outdir, remove_old=True)
        logger.info("DONE!")
