import os
import pickle

import h5py
import hist
import lz4.frame
import numpy as np
import ROOT
from scipy.interpolate import make_smoothing_spline

from wremnants.production import generator_level_definitions, helicity_utils
from wremnants.production.correctionsTensor_helper import (
    makeCorrectionsTensor,
)
from wremnants.utilities import binning, common, samples, theory_utils
from wremnants.utilities.io_tools import base_io
from wums import boostHistHelpers as hh
from wums import logging

logger = logging.child_logger(__name__)


def expand_pdf_entries(pdf, alphas=False, renorm=False):
    info = theory_utils.pdfMap[pdf]
    first_entry = info.get("first_entry", 0)
    pdfBranch = info["branch"]
    if alphas:
        vals = info["alphas"]
    else:
        last_entry = first_entry + info["entries"]
        vals = [info["branch"] + f"[{i}]" for i in range(first_entry, last_entry)]

    if renorm:
        vals = [
            f"std::clamp<float>({x}/{vals[0]}*{pdfBranch}[{first_entry}], -theory_weight_truncate, theory_weight_truncate)"
            for x in vals
        ]
    else:
        vals = [
            f"std::clamp<float>({x}, -theory_weight_truncate, theory_weight_truncate)"
            for x in vals
        ]
    return vals


def make_theory_corr_weight_info(pdf, *, alphas=False, renorm=False):
    return {
        "weights": expand_pdf_entries(pdf, alphas=alphas, renorm=renorm),
        "renorm": renorm,
        "alphas": alphas,
    }


theory_corr_weight_map = {
    "scetlib_dyturbo_MSHT20_N3p0LL_N2LO_pdfas": make_theory_corr_weight_info(
        "msht20", alphas=True
    ),
    "scetlib_dyturbo_MSHT20_N3p0LL_N2LO_pdfvars": make_theory_corr_weight_info(
        "msht20"
    ),
    "scetlib_dyturbo_CT18Z_N3p0LL_N2LO_pdfvars": make_theory_corr_weight_info("ct18z"),
    "scetlib_dyturbo_LatticeNP_CT18Z_N3p0LL_N2LO_pdfvars": make_theory_corr_weight_info(
        "ct18z"
    ),
    "scetlib_dyturbo_CT18Z_N3p0LL_N2LO_pdfas": make_theory_corr_weight_info(
        "ct18z", alphas=True, renorm=True
    ),
    "scetlib_dyturbo_CT18Z_N3p1LL_N2LO_pdfas": make_theory_corr_weight_info(
        "ct18z", alphas=True, renorm=True
    ),
    "scetlib_dyturbo_CT18Z_N4p0LL_N2LO_pdfas": make_theory_corr_weight_info(
        "ct18z", alphas=True, renorm=True
    ),
    "scetlib_nnlojet_CT18Z_N3p1LL_N3LO_pdfas": make_theory_corr_weight_info(
        "ct18z", alphas=True, renorm=True
    ),
    "scetlib_nnlojet_CT18Z_N4p0LLN3LO_pdfas": make_theory_corr_weight_info(
        "ct18z", alphas=True, renorm=True
    ),
    "scetlib_dyturbo_LatticeNP_CT18Z_N3p0LL_N2LO_pdfas": make_theory_corr_weight_info(
        "ct18z", alphas=True, renorm=True
    ),
    "scetlib_dyturbo_MSHT20an3lo_N3p0LL_N2LO_pdfas": make_theory_corr_weight_info(
        "msht20an3lo", alphas=True
    ),
    "scetlib_dyturbo_MSHT20an3lo_N3p0LL_N2LO_pdfvars": make_theory_corr_weight_info(
        "msht20an3lo"
    ),
    "scetlib_dyturbo_LatticeNP_CT18Z_N2p1LL_N2LO_pdfas": make_theory_corr_weight_info(
        "ct18z", alphas=True, renorm=True
    ),
    "scetlib_dyturbo_LatticeNP_CT18Z_N3p1LL_N2LO_pdfas": make_theory_corr_weight_info(
        "ct18z", alphas=True, renorm=True
    ),
    "scetlib_dyturbo_LatticeNP_CT18Z_N4p0LL_N2LO_pdfas": make_theory_corr_weight_info(
        "ct18z", alphas=True, renorm=True
    ),
    "scetlib_nnlojet_LatticeNPCoarse_CT18Z_N3p1LL_N3LO_pdfas": make_theory_corr_weight_info(
        "ct18z", alphas=True, renorm=True
    ),
    "scetlib_nnlojet_LatticeNPCoarse_CT18Z_N4p0LL_N3LO_pdfas": make_theory_corr_weight_info(
        "ct18z", alphas=True, renorm=True
    ),
    "scetlib_dyturbo_LatticeNP_CT18_N3p0LL_N2LO_pdfas": make_theory_corr_weight_info(
        "ct18", alphas=True, renorm=True
    ),
    "scetlib_dyturbo_LatticeNP_HERAPDF20_N3p0LL_N2LO_pdfas": make_theory_corr_weight_info(
        "herapdf20", alphas=True, renorm=True
    ),
    "scetlib_dyturbo_LatticeNP_MSHT20_N3p0LL_N2LO_pdfas": make_theory_corr_weight_info(
        "msht20", alphas=True, renorm=True
    ),
    "scetlib_dyturbo_LatticeNP_MSHT20aN3LO_N3p0LL_N2LO_pdfas": make_theory_corr_weight_info(
        "msht20an3lo", alphas=True, renorm=True
    ),
    "scetlib_dyturbo_LatticeNP_NNPDF31_N3p0LL_N2L0_pdfas": make_theory_corr_weight_info(
        "nnpdf31", alphas=True, renorm=True
    ),
    "scetlib_dyturbo_LatticeNP_NNPDF31_N3p0LL_N2LO_pdfas": make_theory_corr_weight_info(
        "nnpdf31", alphas=True, renorm=True
    ),
    "scetlib_dyturbo_LatticeNP_NNPDF40_N3p0LL_N2LO_pdfas": make_theory_corr_weight_info(
        "nnpdf40", alphas=True, renorm=True
    ),
    "scetlib_dyturbo_LatticeNP_PDF4LHC21_N3p0LL_N2LO_pdfas": make_theory_corr_weight_info(
        "pdf4lhc21", alphas=True, renorm=True
    ),
    "scetlib_dyturbo_LatticeNP_CT18Z_N2p1LL_N2LO_pdfvars": make_theory_corr_weight_info(
        "ct18z"
    ),
    "scetlib_dyturbo_LatticeNP_CT18Z_N3p1LL_N2LO_pdfvars": make_theory_corr_weight_info(
        "ct18z"
    ),
    "scetlib_dyturbo_LatticeNP_CT18Z_N4p0LL_N2LO_pdfvars": make_theory_corr_weight_info(
        "ct18z"
    ),
    "scetlib_nnlojet_LatticeNPCoarse_CT18Z_N3p1LL_N3LO_pdfvars": make_theory_corr_weight_info(
        "ct18z"
    ),
    "scetlib_nnlojet_LatticeNPCoarse_CT18Z_N4p0LL_N3LO_pdfvars": make_theory_corr_weight_info(
        "ct18z"
    ),
    "scetlib_dyturbo_LatticeNP_CT18_N3p0LL_N2LO_pdfvars": make_theory_corr_weight_info(
        "ct18"
    ),
    "scetlib_dyturbo_LatticeNP_HERAPDF20_N3p0LL_N2LO_pdfvars": make_theory_corr_weight_info(
        "herapdf20"
    ),
    "scetlib_dyturbo_LatticeNP_HERAPDF20EXT_N3p0LL_N2LO_pdfvars": make_theory_corr_weight_info(
        "herapdf20ext"
    ),
    "scetlib_dyturbo_LatticeNP_MSHT20_N3p0LL_N2LO_pdfvars": make_theory_corr_weight_info(
        "msht20"
    ),
    "scetlib_dyturbo_LatticeNP_MSHT20aN3LO_N3p0LL_N2LO_pdfvars": make_theory_corr_weight_info(
        "msht20an3lo"
    ),
    "scetlib_dyturbo_LatticeNP_NNPDF31_N3p0LL_N2LO_pdfvars": make_theory_corr_weight_info(
        "nnpdf31"
    ),
    "scetlib_dyturbo_LatticeNP_NNPDF40_N3p0LL_N2LO_pdfvars": make_theory_corr_weight_info(
        "nnpdf40"
    ),
    "scetlib_dyturbo_LatticeNP_PDF4LHC21_N3p0LL_N2LO_pdfvars": make_theory_corr_weight_info(
        "pdf4lhc21"
    ),
    "scetlib_dyturbo_LatticeNP_MSHT20mbrange_N3p0LL_N2LO_pdfvars": make_theory_corr_weight_info(
        "msht20mbrange", renorm=True
    ),
    "scetlib_dyturbo_LatticeNP_MSHT20mcrange_N3p0LL_N2LO_pdfvars": make_theory_corr_weight_info(
        "msht20mcrange", renorm=True
    ),
    # Tested this, better not to treat this way unless using MSHT20nnlo as central set
    # "scetlib_dyturboMSHT20mbrange" : expand_pdf_entries("msht20mbrange", renorm=True),
    # "scetlib_dyturboMSHT20mcrange" : expand_pdf_entries("msht20mcrange", renorm=True),
}


def load_corr_helpers(
    procs,
    generators,
    make_tensor=True,
    base_dir=f"{common.data_dir}/TheoryCorrections/",
    minnlo_ratio=True,
):
    corr_helpers = {}
    for proc in procs:
        corr_helpers[proc] = {}
        for i, generator in enumerate(generators):
            if proc.startswith("WtoNMu") and i == 0:
                label = "BSM"
            else:
                label = proc[0]

            candidate_fnames = [
                f"{base_dir}/{generator}_Corr{label}.pkl.lz4",
                f"{base_dir}/{generator}Corr{label}.pkl.lz4",
            ]
            fname = next((f for f in candidate_fnames if os.path.isfile(f)), None)
            if fname is None:
                logger.warning(
                    f"Did not find correction file {candidate_fnames[0]} for process {proc}, generator {generator}. No correction will be applied for this process!"
                )
                continue
            logger.debug(f"Make theory correction helper for file: {fname}")
            corrh = load_corr_hist(
                fname, label, get_corr_name(generator, minnlo_ratio=minnlo_ratio)
            )
            numh = None
            if (
                (generator == generators[0])
                and ("nnlojet" in generator.lower())
                and ("pdfas" not in generator.lower())
                and ("pdfvars" not in generator.lower())
            ):
                logger.info(
                    f"Adding statistical uncertainties for correction {generator}"
                )
                numh = load_corr_hist(fname, label, f"{generator}_hist")

            corrh = postprocess_corr_hist(corrh, numh)
            if not make_tensor:
                corr_helpers[proc][generator] = corrh
            elif "Helicity" in generator:
                corr_helpers[proc][generator] = makeCorrectionsTensor(
                    corrh, ROOT.wrem.CentralCorrByHelicityHelper, tensor_rank=3
                )
            else:
                corr_helpers[proc][generator] = makeCorrectionsTensor(
                    corrh,
                    tensor_weight=generator in theory_corr_weight_map,
                )
    for generator in generators:
        if not any([generator in corr_helpers[proc] for proc in procs]):
            logger.warning(
                f"Did not find correction for generator {generator} for any processes!"
            )
    return corr_helpers


def define_theory_corr(df, dataset_name, helpers, generators, modify_central_weight):
    logger.debug("define_theory_corr")
    df = df.Define(
        f"nominal_weight_uncorr",
        build_weight_expr(df, exclude_weights=["theory_corr_weight"]),
    )

    dataset_helpers = helpers.get(dataset_name, [])

    if (
        not modify_central_weight
        or not generators
        or generators[0] not in dataset_helpers
    ):
        logger.debug("Define 'theory_corr_weight'=1.0")
        df = df.DefinePerSample("theory_corr_weight", "1.0")

    for i, generator in enumerate(generators):
        if generator not in dataset_helpers:
            continue

        logger.debug(f"Now at generator {i}: {generator}")

        helper = dataset_helpers[generator]

        if "Helicity" in generator:
            # TODO check carefully if the weight below should instead be f"{generator}_corr_weight"  (though it's irrelevant as long as there's only one theory correction)
            df = df.Define(
                f"{generator}Weight_tensor",
                helper,
                [
                    "massVgen",
                    "absYVgen",
                    "ptVgen",
                    "chargeVgen",
                    "csSineCosThetaPhigen",
                    "nominal_weight_uncorr",
                ],
            )
        else:
            df = define_theory_corr_weight_column(df, generator)
            df = df.Define(
                f"{generator}Weight_tensor",
                helper,
                [
                    "massVgen",
                    "absYVgen",
                    "ptVgen",
                    "chargeVgen",
                    f"{generator}_corr_weight",
                ],
            )

        if (i == 0) and modify_central_weight:
            logger.debug(f"applying central value correction for {generator}")
            df = df.Define(
                "theory_corr_weight",
                f"nominal_weight_uncorr == 0 ? 0 : {generator}Weight_tensor(0)/nominal_weight_uncorr",
            )

    return df


def define_theory_corr_weight_column(df, generator):
    if generator in theory_corr_weight_map:
        values = theory_corr_weight_map[generator]["weights"]
        df = df.Define(
            f"{generator}_corr_weight",
            f"Eigen::TensorFixedSize<double, Eigen::Sizes<{len(values)}>> res; "
            + "; ".join(
                [
                    f"res({i}) = {entry}*nominal_weight_uncorr/central_pdf_weight"
                    for i, entry in enumerate(values)
                ]
            )
            + "; return res;",
        )
    else:
        df = df.Alias(f"{generator}_corr_weight", "nominal_weight_uncorr")
    return df


def theory_corr_is_renorm(pdf_name):
    return theory_corr_weight_map.get(
        pdf_name.replace("_Corr", ""),
        {},
    ).get("renorm", False)


def define_pdf_columns(df, dataset_name, pdfs, noAltUnc):
    df = df.Define(
        f"nominal_weight_pdf_uncorr",
        build_weight_expr(df, exclude_weights=["central_pdf_weight"]),
    )
    if (
        len(pdfs) == 0
        or dataset_name not in samples.vprocs
        or "horace" in dataset_name
        or "winhac" in dataset_name
        or "LHEPdfWeight" not in df.GetColumnNames()
    ):
        logger.warning(
            f"Did not find PDF weights for sample {dataset_name}! Using nominal PDF in sample"
        )
        return df

    for i, pdf in enumerate(pdfs):
        try:
            pdfInfo = theory_utils.pdf_info_map(dataset_name, pdf)
        except ValueError:
            return df

        logger.info(f"Defining PDF weights for PDF set {pdf}")

        pdfName = pdfInfo["name"]
        pdfBranch = pdfInfo["branch"]
        tensorName = f"{pdfName}Weights_tensor"
        tensorASName = f"{pdfName}ASWeights_tensor"
        entries = 1 if i != 0 and noAltUnc else pdfInfo["entries"]
        start = 0 if "first_entry" not in pdfInfo else pdfInfo["first_entry"]

        if pdfBranch not in df.GetColumnNames():
            return df

        if "renorm" in pdfInfo and pdfInfo["renorm"]:
            df = df.Define(
                tensorName,
                f"auto res = wrem::vec_to_tensor_t<double, {entries}>({pdfBranch}, {start}); res = res/res(0); "
                "res = wrem::clip_tensor(res, theory_weight_truncate); res = res*nominal_weight; return res;",
            )
        else:
            df = df.Define(
                tensorName,
                f"auto res = wrem::clip_tensor(wrem::vec_to_tensor_t<double, {entries}>({pdfBranch}, {start}), theory_weight_truncate); res = nominal_weight/central_pdf_weight*res; return res;",
            )

        if pdfName == "pdfMSHT20":
            df = pdfBugfixMSHT20(df, tensorName)

        if "alphas" in pdfInfo:
            df = df.Define(
                tensorASName,
                f"Eigen::TensorFixedSize<double, Eigen::Sizes<{len(pdfInfo['alphas'])}>> res; "
                + " ".join(
                    [
                        f"res({i}) = nominal_weight/central_pdf_weight*{p};"
                        for i, p in enumerate(pdfInfo["alphas"])
                    ]
                )
                + "return wrem::clip_tensor(res, theory_weight_truncate)",
            )

    return df


def define_central_pdf_weight_from_helicities(df, dataset_name, pdf, theory_helpers):

    logger.info("Using PDF weights from helicities for the central PDF weight")
    pdf_name = theory_utils.pdfMap[pdf]["name"]

    tensorName = f"helicity{pdf_name}CentralWeight_tensor"
    if "unity" not in df.GetColumnNames():
        df = df.DefinePerSample("unity", "1.")
    df = df.Define(
        tensorName,
        theory_helpers["pdf_central"],
        [
            "massVgen",
            "absYVgen",
            "ptVgen",
            "chargeVgen",
            "csSineCosThetaPhigen",
            "unity",
        ],
    )
    return df.Define(
        "central_pdf_weight",
        f"wrem::clamp_tensor_safe({tensorName}, -theory_weight_truncate, theory_weight_truncate, 1.0)[0]",
    )


def define_central_pdf_weight(df, dataset_name, pdf):

    logger.info("Using event PDF weights for the central PDF weight")
    try:
        pdfInfo = theory_utils.pdf_info_map(dataset_name, pdf)
    except ValueError:
        logger.warning(
            f"Did not find PDF {pdf} for sample {dataset_name}! Using nominal PDF in sample"
        )
        return df.DefinePerSample("central_pdf_weight", "1.0")

    pdfBranch = pdfInfo["branch"]
    if not pdfBranch in df.GetColumnNames():
        logger.warning(
            f"Did not find PDF branch {pdfBranch} for sample {dataset_name}! Set PDF weights to 1"
        )
        return df.DefinePerSample("central_pdf_weight", "1.0")
    first_entry = pdfInfo.get("first_entry", 0)
    return df.Define(
        "central_pdf_weight",
        f"std::clamp<float>({pdfBranch}[{first_entry}], -theory_weight_truncate, theory_weight_truncate)",
    )


def define_theory_weights_and_corrs(df, dataset_name, helpers, args, theory_helpers={}):
    if "LHEPart_status" in df.GetColumnNames():
        df = generator_level_definitions.define_lhe_vars(df)

    if "powheg" not in dataset_name:
        # no preFSR particles in powheg samples
        df = generator_level_definitions.define_prefsr_vars(df)
        if not dataset_name.startswith(("WtoNMuMass", "WtoMuNuSMEFT")):
            # no intermediate bosons in some events in madgraph samples
            logger.debug(f"Define intermediate gen variables for {dataset_name}")
            df = generator_level_definitions.define_intermediate_gen_vars(
                df, "hardProcess", 21, 29
            )
            df = generator_level_definitions.define_intermediate_gen_vars(
                df, "postShower", 21, 59
            )
            df = generator_level_definitions.define_intermediate_gen_vars(
                df, "postBeamRemnants", 21, 69
            )

    if "GenPart_status" in df.GetColumnNames():
        df = generator_level_definitions.define_ew_vars(df)

    df = df.DefinePerSample("theory_weight_truncate", "10.")
    if (
        theory_helpers
        and "pdf_central" in theory_helpers.keys()
        and theory_helpers["pdf_central"] is not None
    ):
        df = define_central_pdf_weight_from_helicities(
            df,
            dataset_name,
            args.pdfs[0] if len(args.pdfs) >= 1 else None,
            theory_helpers,
        )
    else:  # if no boson-parametrized weights are available
        df = define_central_pdf_weight(
            df, dataset_name, args.pdfs[0] if len(args.pdfs) >= 1 else None
        )
    df = define_theory_corr(
        df,
        dataset_name,
        helpers,
        generators=args.theoryCorr,
        modify_central_weight=not args.theoryCorrAltOnly,
    )
    df = define_ew_theory_corr(
        df,
        dataset_name,
        helpers,
        generators=args.ewTheoryCorr,
        modify_central_weight=False,
    )

    if args.highptscales:
        df = df.Define("extra_weight", "MEParamWeightAltSet3[0]")
    df = define_nominal_weight(df)
    df = define_pdf_columns(df, dataset_name, args.pdfs, args.altPdfOnlyCentral)
    df = define_breit_wigner_weights(df, dataset_name)
    df = define_quark_mass_theory_corr(
        df,
        dataset_name,
        helpers,
        generators=getattr(args, "quarkMassCorr", []),
    )

    return df


def build_weight_expr(df, exclude_weights=[]):
    valid_cols = df.GetColumnNames()
    weights = [
        "weight",
        "central_pdf_weight",
        "theory_corr_weight",
        "ew_theory_corr_weight",
        "exp_weight",
    ]
    if weights[0] not in valid_cols:
        raise ValueError(f"The weight '{weights[0]}' must be defined in the histmaker!")
    found_weights = []

    for weight in filter(lambda x: x not in exclude_weights, weights):
        if weight not in valid_cols:
            logger.warning(f"Did not find weight '{weight}'! Assuming 1.0")
        else:
            found_weights.append(weight)

    if "extra_weight" in valid_cols:
        logger.info("Adding additional weight '{extra_weight}'")
        found_weights.append("extra_weight")

    if "central_weight" in valid_cols:
        logger.info("Adding additional central weight 'central_weight'")
        found_weights.append("central_weight")

    weight_expr = "*".join(found_weights)

    logger.debug(f"Weight is {weight_expr}")

    return weight_expr


def define_nominal_weight(df):
    logger.debug("Defining nominal weight")
    return df.Define(f"nominal_weight", build_weight_expr(df))


def define_breit_wigner_weights(df, proc):

    logger.debug("Defining Breit-Wigner weights")
    if "massVgen" not in df.GetColumnNames():
        logger.warning("Did not find massVgen, cannot define Breit-Wigner weights")
        return df.DefinePerSample("bw_weight", "1.0")

    type = 1 if "W" in proc else 0
    entries = 21 if "W" in proc else 23
    df = df.Define(
        f"breitwigner_massWeight{proc[0]}_tensor",
        f"auto res = wrem::vec_to_tensor_t<double, {entries}>(wrem::breitWignerMassWeights<{type}>(massVgen));"
        "res = res * nominal_weight;"
        "return res;",
    )

    entries = 5
    df = df.Define(
        f"breitwigner_widthWeight{proc[0]}_tensor",
        f"auto res = wrem::vec_to_tensor_t<double, {entries}>(wrem::breitWignerWidthWeights<{type}>(massVgen));"
        "res = res * nominal_weight;"
        "return res;",
    )
    return df


def define_ew_theory_corr(
    df, dataset_name, helpers, generators, modify_central_weight=False
):
    logger.debug("define_ew_theory_corr")

    if modify_central_weight:
        raise ValueError(
            "Modifying central weight not currently supported for EW corrections."
        )

    df = df.Define(
        f"nominal_weight_ew_uncorr",
        build_weight_expr(df, exclude_weights=["ew_theory_corr_weight"]),
    )

    dataset_helpers = helpers.get(dataset_name, [])

    for i, generator in enumerate(generators):
        if generator not in dataset_helpers:
            continue

        logger.debug(f"Now at generator {i}: {generator}")
        helper = dataset_helpers[generator]
        df = df.Define(f"ew_{generator}corr_weight", build_weight_expr(df))
        # hack for column names
        if generator == "powhegFOEW":
            ew_cols = [
                "massVgen",
                "absYVgen",
                "csCosThetagen",
                "chargeVgen",
                f"ew_{generator}corr_weight",
            ]
        else:
            ew_cols = [
                *helper.hist.axes.name[:-2],
                "chargeVgen",
                f"ew_{generator}corr_weight",
            ]

        df = df.Define(
            f"{generator}Weight_tensor", helper, ew_cols
        )  # multiplying with nominal QCD weight

        if generator in ["renesanceEW", "powhegFOEW"] and modify_central_weight:
            logger.debug(f"applying central value correction for {generator}")
            df = df.Define(
                "ew_theory_corr_weight",
                f"nominal_weight_ew_uncorr == 0 ? 0 : {generator}Weight_tensor(0)/nominal_weight_ew_uncorr",
            )

    if "ew_theory_corr_weight" not in df.GetColumnNames():
        logger.debug("Define 'ew_theory_corr_weight'=1.0")
        df = df.DefinePerSample("ew_theory_corr_weight", "1.0")

    return df


def define_quark_mass_theory_corr(df, dataset_name, helpers, generators):
    logger.debug("Define quark mass theory corr")
    dataset_helpers = helpers.get(dataset_name, [])

    for generator in generators:
        if generator not in dataset_helpers:
            continue

        helper = dataset_helpers[generator]
        df = df.Define(
            f"{generator}Weight_tensor",
            helper,
            [
                "massVgen",
                "absYVgen",
                "ptVgen",
                "chargeVgen",
                "nominal_weight",
            ],
        )

    return df


def pdfBugfixMSHT20(df, tensorPDFName):
    # There is a known bug in MSHT20 where member 15 and 16 are identical
    #   to fix this, one has to be mirrored:
    #   pdf(15) = pdf(0) - (pdf(15) - pdf(0))
    return df.Redefine(
        tensorPDFName,
        f"auto& res = {tensorPDFName};"
        f"res(15) = {tensorPDFName}(0) - ({tensorPDFName}(15) - {tensorPDFName}(0));"
        "return res",
    )


def load_corr_hist(filename, proc, histname):
    with lz4.frame.open(filename) as f:
        corr = pickle.load(f)
        try:
            corrh = corr[proc][histname]
        except KeyError as e:
            histname = histname.replace("N2LO", "N2L0")
            corrh = corr[proc][histname]
    return corrh


def compute_envelope(
    h, name, entries, axis_name="vars", slice_axis=None, slice_val=None
):

    axis_idx = h.axes.name.index(axis_name)
    hvars = h[{axis_name: entries}]

    hvar_min = h[{axis_name: entries[0]}].copy()
    hvar_max = h[{axis_name: entries[0]}].copy()

    hvar_min.values()[...] = np.min(hvars.values(), axis=axis_idx)
    hvar_max.values()[...] = np.max(hvars.values(), axis=axis_idx)

    if slice_axis is not None:
        s = hist.tag.Slicer()

        hnom = h[{axis_name: entries[0]}]
        slice_idx = h.axes[slice_axis].index(slice_val)

        hvar_min[{slice_axis: s[:slice_idx]}] = hnom[{slice_axis: s[:slice_idx]}].view()
        hvar_max[{slice_axis: s[:slice_idx]}] = hnom[{slice_axis: s[:slice_idx]}].view()

    res = {}
    res[f"{name}_Down"] = hvar_min
    res[f"{name}_Up"] = hvar_max

    return res


def postprocess_corr_hist(corrh, numh=None):
    # extend variations with some envelopes and special kinematic slices

    if (
        "vars" not in corrh.axes.name
        or type(corrh.axes["vars"]) != hist.axis.StrCategory
    ):
        return corrh

    additional_var_hists = {}

    central_var = corrh.axes["vars"][0]
    renorm_scale_vars = [central_var, "kappaFO0.5-kappaf2.", "kappaFO2.-kappaf0.5"]

    renorm_fact_scale_vars = [
        central_var,
        "kappaFO0.5-kappaf2.",
        "kappaFO2.-kappaf0.5",
        "mufdown",
        "mufup",
        "mufdown-kappaFO0.5-kappaf2.",
        "mufup-kappaFO2.-kappaf0.5",
    ]

    resum_scales = ["muB", "nuB", "muS", "nuS"]
    resum_scale_vars_exclusive = [
        var
        for var in corrh.axes["vars"]
        if any(resum_scale in var for resum_scale in resum_scales)
    ]
    resum_scale_vars = [central_var] + resum_scale_vars_exclusive

    if len(renorm_fact_scale_vars) == 1:
        return corrh

    transition_vars_exclusive = [
        "transition_points0.2_0.35_1.0",
        "transition_points0.2_0.75_1.0",
    ]

    renorm_fact_resum_scale_vars = renorm_fact_scale_vars + resum_scale_vars_exclusive

    renorm_fact_resum_transition_scale_vars = (
        renorm_fact_resum_scale_vars + transition_vars_exclusive
    )

    if all(var in corrh.axes["vars"] for var in renorm_scale_vars):
        additional_var_hists.update(
            compute_envelope(corrh, "renorm_scale_envelope", renorm_scale_vars)
        )

        # same thing but restricted to qT>20GeV to capture only the fixed order part of the variation and
        # neglect the part at low pt which should be redundant with the TNPs
        additional_var_hists.update(
            compute_envelope(
                corrh,
                "renorm_scale_pt20_envelope",
                renorm_scale_vars,
                slice_axis="qT",
                slice_val=20.0,
            )
        )

    if all(var in corrh.axes["vars"] for var in renorm_fact_scale_vars):
        additional_var_hists.update(
            compute_envelope(
                corrh, "renorm_fact_scale_envelope", renorm_fact_scale_vars
            )
        )

        # same thing but restricted to qT>20GeV to capture only the fixed order part of the variation and
        # neglect the part at low pt which should be redundant with the TNPs
        additional_var_hists.update(
            compute_envelope(
                corrh,
                "renorm_fact_scale_pt20_envelope",
                renorm_fact_scale_vars,
                slice_axis="qT",
                slice_val=20.0,
            )
        )

    if all(var in corrh.axes["vars"] for var in renorm_fact_resum_scale_vars):
        additional_var_hists.update(
            compute_envelope(
                corrh, "renorm_fact_resum_scale_envelope", renorm_fact_resum_scale_vars
            )
        )
    if all(
        var in corrh.axes["vars"] for var in renorm_fact_resum_transition_scale_vars
    ):
        additional_var_hists.update(
            compute_envelope(
                corrh,
                "renorm_fact_resum_transition_scale_envelope",
                renorm_fact_resum_transition_scale_vars,
            )
        )
    if len(resum_scale_vars) > 1 and all(
        var in corrh.axes["vars"] for var in resum_scale_vars
    ):
        additional_var_hists.update(
            compute_envelope(corrh, "resum_scale_envelope", resum_scale_vars)
        )

    # add per-bin stat unc from correction (~= only the numerator, MiNNLO has very small stat uncs)
    if numh is not None:
        numh_nom = numh[{"vars": 0}]
        var_relative = np.sqrt(numh_nom.variances()) / numh_nom.values()
        nom_vals = corrh[{"vars": 0}].values()

        shape = var_relative.shape
        nbins = var_relative.size

        # nbins copies of the original histogram with shape `shape`
        base_up = np.broadcast_to(nom_vals, (nbins,) + shape).copy()
        base_dn = base_up.copy()

        linear_idx = np.arange(nbins)  # 1-dim array ennumerating all bins
        multi_idx = np.unravel_index(linear_idx, shape)  # n-d idx for each bin
        flat_var_rel = var_relative.ravel()  # 1-dim array of variations

        # address n-th copy of histograms (to hold the variation in the n-th bin [and no other]).
        # In that n-th copy, modify the corresponding bin in the n-dim hist
        base_up[(linear_idx,) + multi_idx] *= 1.0 + flat_var_rel
        base_dn[(linear_idx,) + multi_idx] *= 1.0 - flat_var_rel

        template = corrh[{"vars": 0}]

        for i in range(nbins):
            h_up = template.copy()
            h_dn = template.copy()
            h_up.values()[...] = base_up[i]
            h_dn.values()[...] = base_dn[i]

            additional_var_hists[f"per_bin_stat_unc_theory_corr_bin{i}Up"] = h_up
            additional_var_hists[f"per_bin_stat_unc_theory_corr_bin{i}Down"] = h_dn

    if not additional_var_hists:
        return corrh

    vars_out = list(corrh.axes["vars"]) + list(additional_var_hists.keys())

    vars_out_axis = hist.axis.StrCategory(vars_out, name="vars")
    corrh_tmp = hist.Hist(*corrh.axes[:-1], vars_out_axis, storage=corrh.storage_type())

    for i, var in enumerate(vars_out_axis):
        if var in corrh.axes["vars"]:
            corrh_tmp[{"vars": i}] = corrh[{"vars": var}].view(flow=True)
        else:
            corrh_tmp[{"vars": i}] = additional_var_hists[var].view(flow=True)

    corrh = corrh_tmp

    return corrh


def get_corr_name(generator, minnlo_ratio=True):
    # Hack for now
    label = generator.replace("1D", "")
    if (
        "dataPtll" in generator or "dataRecoPtll" in generator
    ) and "scetlib" not in generator:
        return "MC_data_ratio"
    if minnlo_ratio:
        return (
            f"{label}_minnlo_ratio"
            if "Helicity" not in generator
            else f"{label.replace('Helicity', '')}_minnlo_coeffs"
        )
    else:
        return (
            f"{label}_hist"
            if "Helicity" not in generator
            else f"{label.replace('Helicity', '')}_coeffs"
        )


def rebin_corr_hists(hists, ndim=-1, binning=None):
    # Allow trailing dimensions to be different (e.g., variations)
    ndims = min([x.ndim for x in hists]) if ndim < 0 else ndim
    if binning:
        try:
            hists = [
                h if not h else hh.rebinHistMultiAx(h, binning.keys(), binning.values())
                for h in hists
            ]
        except ValueError:
            logger.warning("Can't rebin axes to predefined binning")
        return hists

    for i in range(ndims):
        hists = hh.rebinHistsToCommon(hists, i)
    return hists


# Apply an iterative smoothing in 2D (effectively assumed to be Y, the qT)
def smooth_theory_corr(corrh, minnloh, numh, ax2_start=5):
    if corrh.ndim != 5:
        raise NotImplementedError(
            f"Currently only dimension 5 hists are supported for smoothing. Found ndim={corrh.ndim} ({corrh.axes.name})"
        )

    nc = corrh.axes["charge"].size

    # First smooth in 1D (should be rapidity)
    corrh1D = hh.divideHists(numh[{"vars": 0}].project(1), minnloh.project(1))
    ax1 = corrh.axes[1]
    spl = make_smoothing_spline(ax1.centers, corrh1D.values())
    smooth = spl(ax1.centers) / corrh1D.values()
    corrh.values()[...] = (corrh.values().T * smooth[:, np.newaxis]).T

    # This should be qT
    ax2 = corrh.axes[2]
    for i1 in range(ax1.size):
        for ic in range(corrh.axes["charge"].size):
            for iv in range(corrh.axes["vars"].size):
                spl = make_smoothing_spline(
                    ax2.centers[ax2_start:], corrh[0, i1, ax2_start:, ic, iv].values()
                )
                corrh.values()[0, i1, ax2_start:, ic, iv] = spl(ax2.centers[ax2_start:])

    return corrh


# Assuming the 3 physics variable dimensions are first
def set_corr_ratio_flow(corrh):
    # Probably there's a better way to do this...
    if corrh.axes[0].traits.underflow:
        corrh[hist.underflow, ...] = np.ones_like(corrh[0, ...].view(flow=True))
    if corrh.axes[0].traits.overflow:
        corrh[hist.overflow, ...] = np.ones_like(corrh[0, ...].view(flow=True))

    if corrh.axes[1].traits.underflow:
        corrh[:, hist.underflow, ...] = np.ones_like(corrh[:, 0, ...].view(flow=True))
    if corrh.axes[1].traits.overflow:
        corrh[:, hist.overflow, ...] = np.ones_like(corrh[:, 0, ...].view(flow=True))

    if corrh.axes[2].traits.underflow:
        corrh[:, :, hist.underflow, ...] = np.ones_like(
            corrh[:, :, 0, ...].view(flow=True)
        )
    if corrh.axes[2].traits.overflow:
        corrh[:, :, hist.overflow, ...] = np.ones_like(
            corrh[:, :, 0, ...].view(flow=True)
        )
    return corrh


def make_corr_from_ratio(
    denom_hist, num_hist, rebin=None, smooth="numerator", normalize=False
):
    denom_hist, num_hist = rebin_corr_hists([denom_hist, num_hist], binning=rebin)

    if smooth == "numerator":
        logger.info(
            "Applying spline-based smoothing to numerator before making correction hist"
        )
        num_hist = hh.smooth_hist(
            hh.smooth_hist(num_hist, "absY", exclude_axes=["qT"]), "qT", start_bin=4
        )

    if normalize:
        num_hist = hh.divideHists(num_hist, num_hist.project("vars"))
        denom_hist = hh.normalize(denom_hist, scale=1)

    corrh = hh.divideHists(num_hist, denom_hist, flow=False, by_ax_name=False)

    if smooth == "ratio":
        logger.info("Applying spline-based smoothing to correction hist ratio")
        corrh = smooth_theory_corr(corrh, denom_hist, num_hist, ax2_start=5)

    return set_corr_ratio_flow(corrh), denom_hist, num_hist


def make_corr_by_helicity(
    ref_helicity_hist,
    target_sigmaul,
    target_sigma4,
    coeff_hist=None,
    coeffs_from_hist=[],
    binning=None,
    ndim=3,
):
    ref_helicity_hist, target_sigmaul, target_sigma4 = rebin_corr_hists(
        [ref_helicity_hist, target_sigmaul, target_sigma4], ndim, binning
    )

    apply_coeff_corr = coeff_hist is not None and coeffs_from_hist

    # broadcast back mass and helicity axes (or vars axis)
    sigmaUL_ratio = (
        hh.divideHists(
            target_sigmaul,
            ref_helicity_hist[{"massVgen": 0, "helicity": -1.0j}],
            flow=False,
            by_ax_name=False,
        ).values()[np.newaxis, ..., np.newaxis, :]
        if target_sigmaul
        else np.ones_like(ref_helicity_hist)[..., np.newaxis]
    )

    ref_coeffs = helicity_utils.helicity_xsec_to_angular_coeffs(ref_helicity_hist)

    corr_ax = hist.axis.Boolean(name="corr")
    vars_ax = (
        target_sigmaul.axes["vars"]
        if target_sigmaul
        else hist.axis.Regular(1, 0, 1, name="vars")
    )
    corr_coeffs = hist.Hist(*ref_coeffs.axes, corr_ax, vars_ax)
    # Corr = False is the uncorrected coeffs, corrected coeffs have the new A4
    # NOTE: the corrected coeffs are multiplied through by the sigmaUL correction, so that the
    # new correction can be made as the ratio of the sum. To get the correct coeffs, this should
    # be divided back out
    corr_coeffs[...] = ref_coeffs.values()[..., np.newaxis, np.newaxis]

    if target_sigma4:
        target_a4_coeff = make_angular_coeff(target_sigma4, target_sigmaul)
        corr_coeffs[..., 4.0j, True, :] = target_a4_coeff.values()

    if apply_coeff_corr:
        for coeff in coeffs_from_hist:
            scale = -1 if coeff in [1, 4] else 1
            idx = complex(0, coeff)
            corr_coeffs[..., idx, True, :] = (
                coeff_hist[{"helicity": idx}].values()[..., np.newaxis] * scale
            )
    # Scale by the sigmaUL correction,
    corr_coeffs.values()[..., corr_coeffs.axes["corr"].index(True), :] *= sigmaUL_ratio

    corr_coeffs = set_corr_ratio_flow(corr_coeffs)
    return corr_coeffs


def make_theory_helpers(
    pdfs,
    theory_corr=[],
    procs=["Z", "W"],
    corrs=["qcdScale", "pdf", "pdf_from_corr", "alphaS", "pdf_central"],
):

    theory_helpers_procs = {p: {} for p in procs}

    for proc in theory_helpers_procs.keys():

        if "qcdScale" in corrs:
            theory_helpers_procs[proc]["qcdScale"] = (
                make_qcd_uncertainty_helper_by_helicity(
                    is_z=proc == "Z",
                    rebin_ptVgen=False,
                    return_tensor=True,
                )
            )

        if "pdf" in corrs:
            theory_helpers_procs[proc]["pdf"] = (
                make_pdfs_uncertainties_helper_by_helicity(
                    proc=proc,
                    pdfs=pdfs,
                )
            )
        if "pdf_from_corr" in corrs:
            pdf_from_corrs = [x + "_Corr" for x in theory_corr if "pdfvar" in x]
            theory_helpers_procs[proc]["pdf_from_corr"] = (
                make_pdfs_from_corrs_uncertainties_helper_by_helicity(
                    proc=proc,
                    pdfs_from_corrs=pdf_from_corrs,
                )
            )
        if "alphaS" in corrs:
            as_vars = [x + "_Corr" for x in theory_corr if "pdfas" in x]
            theory_helpers_procs[proc]["alphaS"] = (
                make_alphaS_uncertainties_helper_by_helicity(
                    proc=proc,
                    as_vars=as_vars,
                )
            )
        if "pdf_central" in corrs:
            theory_helpers_procs[proc]["pdf_central"] = (
                make_uncertainty_helper_by_helicity(
                    proc=proc,
                    nom=theory_utils.pdfMap[pdfs[0]]["name"],
                    den="pdf_uncorr",
                    central_weights=True,
                    filename=common.data_dir
                    + f"/TheoryCorrections/ByHelicity/PDFs/w_z_gen_dists_maxFiles_m1_{pdfs[0]}_pdfByHelicity_skimmed.hdf5",
                )
            )

    return theory_helpers_procs


def make_qcd_uncertainty_helper_by_helicity(
    is_z=False,
    filename=f"{common.data_dir}/angularCoefficients/w_z_helicity_xsecs.hdf5",
    rebin_ptVgen=binning.ptV_binning,
    rebin_absYVgen=False,
    rebin_massVgen=False,
    return_tensor=True,
):

    # load helicity cross sections from file
    with h5py.File(filename, "r") as h5file:
        results = base_io.load_results_h5py(h5file)

    def get_helicity_xsecs(
        suffix="",
        rebin_ptVgen=binning.ptV_binning,
        rebin_absYVgen=False,
        rebin_massVgen=2,
    ):
        h = results[f"Z{suffix}"] if is_z else results[f"W{suffix}"]

        if rebin_ptVgen:
            if type(rebin_ptVgen) is bool:
                h = hh.rebinHist(h, "ptVgen", binning.ptV_binning)
            else:
                h = hh.rebinHist(h, "ptVgen", rebin_ptVgen)
        if rebin_massVgen:
            if type(rebin_massVgen) is bool:
                if is_z:
                    axis_massVgen = h.axes["massVgen"]
                    if len(axis_massVgen.edges) > 2:
                        h = hh.rebinHist(h, "massVgen", axis_massVgen.edges[::2])
            else:
                h = hh.rebinHist(h, "massVgen", rebin_massVgen)
        if rebin_absYVgen:
            h = hh.rebinHist(h, "absYVgen", rebin_absYVgen)

        return h

    helicity_xsecs = get_helicity_xsecs(
        rebin_ptVgen=rebin_ptVgen,
        rebin_absYVgen=rebin_absYVgen,
        rebin_massVgen=rebin_massVgen,
    )
    helicity_xsecs_lhe = get_helicity_xsecs(
        "_lhe",
        rebin_ptVgen=rebin_ptVgen,
        rebin_absYVgen=rebin_absYVgen,
        rebin_massVgen=rebin_massVgen,
    )

    helicity_xsecs_nom = helicity_xsecs[{"muRfact": 1.0j, "muFfact": 1.0j}].values()

    # set disallowed combinations of mur/muf equal to nominal
    helicity_xsecs.values()[..., 0, 2] = helicity_xsecs_nom
    helicity_xsecs.values()[..., 2, 0] = helicity_xsecs_nom

    # flatten scale variations and compute envelope
    helicity_xsecs_flat = np.reshape(
        helicity_xsecs.values(), (*helicity_xsecs.values().shape[:-2], -1)
    )
    helicity_xsecs_min = np.min(helicity_xsecs_flat, axis=-1)
    helicity_xsecs_max = np.max(helicity_xsecs_flat, axis=-1)

    # build variation histogram in the format expected by the corrector
    corr_ax = hist.axis.Boolean(name="corr")

    def get_names(ihel):
        base_name = f"helicity_{ihel}"
        return f"{base_name}_Down", f"{base_name}_Up"

    var_names = []
    var_names.append("nominal")
    for ihel in range(-1, 8):
        var_names.extend(get_names(ihel))

    var_names.append("pythia_shower_kt")

    vars_ax = hist.axis.StrCategory(var_names, name="vars")

    axes_no_scale = helicity_xsecs.axes[:-2]
    corr_coeffs = hist.Hist(*axes_no_scale, corr_ax, vars_ax)

    # set all helicity_xsecs, including overflow/underflow to default safe value (leads to weight of 1.0 by construction)
    corr_coeffs.values(flow=True)[...] = 1.0

    # set all helicity_xsecs equal to nominal
    corr_coeffs.values()[...] = helicity_xsecs_nom[..., None, None]

    # set envelope variations
    for ihel in range(-1, 8):
        downvar, upvar = get_names(ihel)

        corr_coeffs.values()[..., ihel + 1, 1, var_names.index(downvar)] = (
            helicity_xsecs_min[..., ihel + 1]
        )
        corr_coeffs.values()[..., ihel + 1, 1, var_names.index(upvar)] = (
            helicity_xsecs_max[..., ihel + 1]
        )

    corr_coeffs[{"corr": False, "vars": "pythia_shower_kt"}] = (
        helicity_xsecs[{"muRfact": 1.0j, "muFfact": 1.0j}].values()
        / helicity_xsecs[
            {"muRfact": 1.0j, "muFfact": 1.0j, "helicity": -1.0j}
        ].values()[..., None]
    )
    corr_coeffs[{"corr": True, "vars": "pythia_shower_kt"}] = (
        helicity_xsecs_lhe[{"muRfact": 1.0j, "muFfact": 1.0j}].values()
        / helicity_xsecs_lhe[
            {"muRfact": 1.0j, "muFfact": 1.0j, "helicity": -1.0j}
        ].values()[..., None]
    )

    if return_tensor:
        helper = makeCorrectionsTensor(
            corr_coeffs, ROOT.wrem.CentralCorrByHelicityHelper, tensor_rank=3
        )

        # override tensor_axes since the output is different here
        helper.tensor_axes = [vars_ax]

        return helper
    else:
        return corr_coeffs


def make_pdfs_uncertainties_helper_by_helicity(
    proc,
    pdfs,
    return_tensor=True,
):
    pdf_file_template = (
        common.data_dir
        + "/TheoryCorrections/ByHelicity/PDFs/w_z_gen_dists_maxFiles_m1_{pdf}_pdfByHelicity_skimmed.hdf5"
    )
    pdf_helpers = {}
    for pdf in pdfs:
        pdf_name = theory_utils.pdfMap[pdf]["name"]
        logger.debug(
            f"Making PDF uncertainty helper by helicity for PDF set {pdf_name}"
        )
        pdf_renorm_name = (
            pdf_name if theory_utils.pdfMap[pdf].get("renorm", False) else "pdf_uncorr"
        )
        pdf_helper = make_uncertainty_helper_by_helicity(
            proc=proc,
            nom=pdf_name,
            den=pdf_renorm_name,
            filename=pdf_file_template.format(pdf=pdf),
            var_ax_name="pdfVar",
            return_tensor=return_tensor,
        )
        if pdf_helper is not None:
            pdf_helpers[pdf_name] = pdf_helper
    return pdf_helpers


def make_pdfs_from_corrs_uncertainties_helper_by_helicity(
    proc,
    pdfs_from_corrs,
    return_tensor=True,
):
    pdf_file_template = (
        common.data_dir
        + "/TheoryCorrections/ByHelicity/PDFsFromCorrs/w_z_gen_dists_{pdf}_maxFiles_m1_skimmed.hdf5"
    )
    pdf_helpers = {}
    for pdf in pdfs_from_corrs:
        logger.debug(f"Making PDF uncertainty helper by helicity for theory corr {pdf}")
        den = pdf if theory_corr_is_renorm(pdf_name=pdf) else "pdf_uncorr"
        pdf_helper = make_uncertainty_helper_by_helicity(
            proc=proc,
            nom=pdf,
            den=den,
            central_weights=False,
            var_ax_name="vars",
            filename=pdf_file_template.format(pdf=pdf),
            return_tensor=return_tensor,
        )
        if pdf_helper is not None:
            pdf_helpers[pdf] = pdf_helper
    return pdf_helpers


def make_alphaS_uncertainties_helper_by_helicity(
    proc,
    as_vars,
    return_tensor=True,
):
    alphas_file_template = (
        common.data_dir
        + "/TheoryCorrections/ByHelicity/AlphaS/w_z_gen_dists_{as_var}_maxFiles_m1_skimmed.hdf5"
    )
    as_helpers = {}
    for as_var in as_vars:
        logger.debug(
            f"Making alphaS uncertainty helper by helicity for theory corr {as_var}"
        )
        fname = alphas_file_template.format(as_var=as_var)
        as_helper = make_uncertainty_helper_by_helicity(
            proc=proc,
            nom=as_var,
            den="theory_uncorr",
            filename=fname,
            var_ax_name="vars",
            return_tensor=return_tensor,
        )
        if as_helper is not None:
            as_helpers[as_var] = as_helper
    return as_helpers


def make_uncertainty_helper_by_helicity(
    proc,
    nom,
    den,
    filename,
    filename_den=None,
    central_weights=False,
    var_ax_name="pdfVar",
    return_tensor=True,
):
    """
    Construct a CentralCorrByHelicityHelper from helicity cross sections stored in an hdf5 file.
    """

    if filename_den is None:
        filename_den = filename

    # load helicity cross sections from file #TODO: include DYJetsToMuMuMass10to50
    proc_map = {
        "Z": ("Zmumu",),
        "W": ("Wmunu",),
    }

    def _collect_hist(hist_name, filename):
        hist_key = f"nominal_gen_{hist_name}"
        hists = []
        if not os.path.exists(filename):
            logger.warning(
                f"File {filename} does not exist. Not creating histogram of variations by helicities."
            )
            return None
        with h5py.File(filename, "r") as h5file:
            for process in proc_map.get(proc, ()):
                results = base_io.load_results_h5py(h5file)
                if process not in results.keys():
                    logger.warning(
                        f"Did not find key for process {process} in {filename}. Not creating histogram of variations by helicities for process {process} and variation {nom}."
                    )
                    return None
                outputs = results[process].get("output", {})
                if hist_key not in outputs:
                    logger.warning(
                        f"Did not find {hist_key} in {filename}. Not creating histogram of variations by helicities for process {process} and variation {nom}."
                    )
                    return None
                hists.append(outputs[hist_key].get())
        if not hists:
            logger.warning(
                f"Process {proc} is not supported when building PDF variations."
            )
            return None
        combined = hh.sumHists(hists)
        return combined

    h_nom = _collect_hist(nom, filename)
    if h_nom is None:
        return None

    if den == nom:
        h_den = h_nom
    else:
        h_den = _collect_hist(den, filename_den)
        if h_den is None:
            return None

    # construct the correction tensor
    corr_ax = hist.axis.Boolean(name="corr")
    vars_ax = h_nom.axes[var_ax_name]
    axes_no_scale = h_nom.axes[:-1]
    if central_weights:
        # in the case we are computing a helper for the central weights, we don't need to fill all variations
        new_vars_ax = hist.axis.StrCategory(["nominal"], name="vars")
        corr_coeffs = hist.Hist(*axes_no_scale, corr_ax, new_vars_ax)
    else:
        corr_coeffs = hist.Hist(*axes_no_scale, corr_ax, vars_ax)

    # set all helicity_xsecs equal to nominal
    if var_ax_name in h_den.axes.name:
        h_den = h_den[{var_ax_name: 0}]
    corr_coeffs.values(flow=True)[...] = h_den.values(flow=True)[..., None, None]

    # set the variations
    if central_weights:
        # in the case we are computing a helper for the central weights, we don't need to fill all variations
        h_nom = h_nom[{var_ax_name: 0}]
        corr_coeffs.values(flow=True)[..., 1, :] = h_nom.values(flow=True)[..., None]
    else:
        corr_coeffs.values(flow=True)[..., 1, :] = h_nom.values(flow=True)

    if return_tensor:
        helper = makeCorrectionsTensor(
            corr_coeffs, ROOT.wrem.CentralCorrByHelicityHelper, tensor_rank=3
        )

        # override tensor_axes since the output is different here
        helper.tensor_axes = [vars_ax]

        return helper
    else:
        return corr_coeffs


def make_helicity_test_corrector(is_z=False, filename=None):

    # load hist_helicity cross sections from file
    with h5py.File(filename, "r") as h5file:
        results = base_io.load_results_h5py(h5file)
        hist_helicity_xsec = results["Z"] if is_z else results["W"]

    coeffs = helicity_utils.helicity_xsec_to_angular_coeffs(hist_helicity_xsec)

    coeffs_nom = coeffs[{"muRfact": 1.0j, "muFfact": 1.0j}].values()

    corr_ax = hist.axis.Boolean(name="corr")
    vars_ax = hist.axis.StrCategory(
        ["test_ai", "test_sigmaUL", "test_all"], name="vars"
    )

    axes_no_scale = coeffs.axes[:-2]
    corr_coeffs = hist.Hist(*axes_no_scale, corr_ax, vars_ax)

    corr_coeffs.values()[...] = coeffs_nom[..., None, None]

    # set synthetic test variation
    corr_coeffs.values()[..., 1, 1, 0] *= 1.1

    corr_coeffs.values()[..., :, 1, 1] *= 1.1

    corr_coeffs.values()[..., :, 1, 2] *= 1.1
    corr_coeffs.values()[..., 1, 1, 2] *= 1.2
    corr_coeffs.values()[..., 2, 1, 2] *= 1.3
    corr_coeffs.values()[..., 3, 1, 2] *= 1.4
    corr_coeffs.values()[..., 4, 1, 2] *= 1.5
    corr_coeffs.values()[..., 5, 1, 2] *= 1.6
    corr_coeffs.values()[..., 6, 1, 2] *= 1.7
    corr_coeffs.values()[..., 7, 1, 2] *= 1.8
    corr_coeffs.values()[..., 8, 1, 2] *= 1.9

    helper = makeCorrectionsTensor(
        corr_coeffs, ROOT.wrem.CentralCorrByHelicityHelper, tensor_rank=3
    )

    # override tensor_axes since the output is different here
    helper.tensor_axes = [vars_ax]

    return helper


def make_angular_coeff(sigmai_hist, ul_hist):
    return hh.divideHists(sigmai_hist, ul_hist, cutoff=0.0001)
