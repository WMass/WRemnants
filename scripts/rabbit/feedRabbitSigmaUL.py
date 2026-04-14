"""Build rabbit tensor inputs for direct-theory sigmaUL fits.

Inputs:
1) An unfolded sigmaUL fit result (`--infile`) or optional sigmaUL pseudodata
     from a corrections histogram (`--pseudodataGenerator`).
2) Theory correction histograms from `wremnants-data/data/TheoryCorrections`.

Output:
- A rabbit-compatible HDF5 tensor containing the sigmaUL process and selected
    direct-theory systematics.
"""

import os

import rabbit.io_tools
from wremnants.postprocessing.theory_fit_writer import SigmaULTheoryFitWriter
from wremnants.postprocessing.theory_variation_labels import (
    BC_QUARK_MASS_VARIATIONS,
    LATTICE_CORRELATED_NP_UNCERTAINTIES,
    LATTICE_GAMMA_NP_UNCERTAINTIES,
    STANDARD_CORRELATED_NP_UNCERTAINTIES,
    STANDARD_GAMMA_NP_UNCERTAINTIES,
    TNP_UNCERTAINTIES,
    TRANSITION_FO_UNCERTAINTIES,
)
from wremnants.production import theory_corrections
from wremnants.utilities import common, parsing, theory_utils
from wums import logging, output_tools

SIGMAUL_CHANNEL = "chSigmaUL"
PROCESS_NAME = "Zmumu"


def _pdfas_generator_name(pred_generator):
    if pred_generator.endswith("_pdfas"):
        return pred_generator
    return f"{pred_generator}_pdfas"


def _pdfvars_generator_name(pred_generator):
    if pred_generator.endswith("_pdfvars"):
        return pred_generator
    return f"{pred_generator}_pdfvars"


def _join_cli_tokens(value):
    if isinstance(value, (list, tuple)):
        return " ".join(value)
    return value


def _select_baseline_variation(h, axis_name="vars", nominal_entry="pdf0"):
    if axis_name not in h.axes.name:
        raise KeyError(
            f"Expected axis '{axis_name}' in theory histogram, found axes {h.axes.name}"
        )

    try:
        return h[{axis_name: nominal_entry}]
    except Exception as exc:
        available_entries = list(h.axes[axis_name])
        raise KeyError(
            f"Expected nominal entry '{nominal_entry}' in axis '{axis_name}', "
            f"found entries {available_entries}"
        ) from exc


def _resolve_sigmaul_channel(fitresult, mapping_name, requested_channel):
    channels = fitresult["mappings"][mapping_name]["channels"]
    if requested_channel in channels:
        return requested_channel

    if " " in requested_channel:
        fallback = requested_channel.rsplit(" ", 1)[-1]
        if fallback in channels:
            return fallback

    matches = [name for name in channels if name.endswith(requested_channel)]
    if len(matches) == 1:
        return matches[0]

    raise KeyError(
        f"Unable to resolve sigmaUL channel '{requested_channel}' in mapping '{mapping_name}'. "
        f"Available channels: {list(channels.keys())}"
    )


def load_sigmaul_data(args, writer, logger):
    if args.pseudodataGenerator:
        infile = f"{common.data_dir}/TheoryCorrections/{args.pseudodataGenerator}_CorrZ.pkl.lz4"
        logger.info("Loading sigmaUL pseudodata from %s", infile)
        h_data = theory_corrections.load_corr_hist(
            infile,
            "Z",
            f"{args.pseudodataGenerator}_hist",
        )
        h_data = _select_baseline_variation(h_data)
        h_data = h_data.project("qT", "absY")
        writer.add_channel(h_data.axes, SIGMAUL_CHANNEL)
        writer.add_data(h_data, SIGMAUL_CHANNEL)
        writer.set_reference(SIGMAUL_CHANNEL, h_data)
        return None

    fitresult, meta = rabbit.io_tools.get_fitresult(
        args.infile, result="asimov", meta=True
    )
    logger.debug(
        "Available models in fit result: %s", list(fitresult["mappings"].keys())
    )

    h_data_cov = fitresult["mappings"][args.fitresultMapping][
        "hist_postfit_inclusive_cov"
    ].get()
    writer.add_data_covariance(h_data_cov)

    channel_name = _resolve_sigmaul_channel(
        fitresult, args.fitresultMapping, args.channelSigmaUL
    )
    logger.debug(
        "Using sigmaUL channel '%s' in mapping '%s'",
        channel_name,
        args.fitresultMapping,
    )
    h_data = fitresult["mappings"][args.fitresultMapping]["channels"][channel_name][
        "hist_postfit_inclusive"
    ].get()
    writer.add_channel(h_data.axes, SIGMAUL_CHANNEL)
    writer.add_data(h_data, SIGMAUL_CHANNEL)
    writer.set_reference(SIGMAUL_CHANNEL, h_data)
    return meta


def build_output_metadata(args, input_meta):
    meta = {
        "meta_info": output_tools.make_meta_info_dict(
            args=args,
            wd=common.base_dir,
        ),
    }
    if input_meta is not None:
        meta["meta_info_input"] = input_meta
    return meta


def add_sigmaul_process(args, writer):
    h_sig_sigmaul = theory_corrections.load_corr_hist(
        f"{common.data_dir}/TheoryCorrections/{args.predGenerator}_CorrZ.pkl.lz4",
        "Z",
        f"{args.predGenerator}_hist",
    )
    h_sig_sigmaul = _select_baseline_variation(h_sig_sigmaul)
    writer.add_process(h_sig_sigmaul, PROCESS_NAME, SIGMAUL_CHANNEL, signal=False)


def add_alphas_variation(args, writer, pdf_name):
    symmetrize = "average" if "alphaS" in args.nois else "quadratic"
    alphas_var_name = _pdfas_generator_name(args.predGenerator)
    alphas_vars = theory_corrections.load_corr_helpers(
        ["Z"],
        [alphas_var_name],
        make_tensor=False,
        minnlo_ratio=False,
    )
    writer.add_systematic(
        [
            alphas_vars["Z"][alphas_var_name][{"vars": 2}],
            alphas_vars["Z"][alphas_var_name][{"vars": 1}],
        ],
        "pdfAlphaS",
        PROCESS_NAME,
        SIGMAUL_CHANNEL,
        noi=("alphaS" in args.nois),
        constrained=not ("alphaS" in args.nois),
        symmetrize=symmetrize,
        kfactor=1.0,
        groups=(
            [pdf_name, f"{pdf_name}AlphaS", "theory", "theory_qcd"]
            if "alphaS" not in args.nois
            else [pdf_name]
        ),
    )


def add_pdf_bc_quark_mass_variations(args, writer, logger, pdf_name):

    if pdf_name == theory_utils.pdfMap["herapdf20"]["name"] and not args.noHERAPDF20EXT:
        logger.info(
            "Skipping PDF b/c quark-mass variations since using HERAPDF20EXT already includes them."
        )
        return

    corr_helpers = theory_corrections.load_corr_helpers(
        ["Z"],
        [helper_name for helper_name, *_ in BC_QUARK_MASS_VARIATIONS],
        make_tensor=False,
        minnlo_ratio=False,
    )

    for helper_name, nuisance_name, down_var, up_var in BC_QUARK_MASS_VARIATIONS:
        h = corr_helpers["Z"][helper_name]
        writer.add_shape_systematic(
            [
                h[{"vars": up_var}],
                h[{"vars": down_var}],
                _select_baseline_variation(h),
            ],
            nuisance_name,
            PROCESS_NAME,
            SIGMAUL_CHANNEL,
            symmetrize="quadratic",
            groups=["bcQuarkMass", "pTModeling", "theory", "theory_qcd"],
        )


def add_resummation_and_np_variations(args, writer):
    generator_vars = theory_corrections.load_corr_helpers(
        ["Z"],
        [
            args.predGenerator,
        ],
        make_tensor=False,
        minnlo_ratio=False,
    )
    nominal = generator_vars["Z"][args.predGenerator]

    if "lattice" in args.predGenerator.lower():
        corr_np_uncs = LATTICE_CORRELATED_NP_UNCERTAINTIES
        gamma_np_uncs = LATTICE_GAMMA_NP_UNCERTAINTIES
    else:
        corr_np_uncs = STANDARD_CORRELATED_NP_UNCERTAINTIES
        gamma_np_uncs = STANDARD_GAMMA_NP_UNCERTAINTIES

    for up_var, down_var, nuisance_name in corr_np_uncs:
        writer.add_systematic(
            [nominal[{"vars": up_var}], nominal[{"vars": down_var}]],
            nuisance_name,
            PROCESS_NAME,
            SIGMAUL_CHANNEL,
            symmetrize="average",
            groups=["resumNonpert", "resum", "pTModeling", "theory", "theory_qcd"],
        )

    for up_var, down_var, nuisance_name in gamma_np_uncs:
        writer.add_systematic(
            [nominal[{"vars": up_var}], nominal[{"vars": down_var}]],
            nuisance_name,
            PROCESS_NAME,
            SIGMAUL_CHANNEL,
            symmetrize="average",
            groups=["resumTNP", "resum", "pTModeling", "theory", "theory_qcd"],
        )

    for up_var, down_var in TNP_UNCERTAINTIES:
        writer.add_systematic(
            [nominal[{"vars": up_var}], nominal[{"vars": down_var}]],
            f"resumTNP_{down_var.split('-')[0]}",
            PROCESS_NAME,
            SIGMAUL_CHANNEL,
            symmetrize="average",
            groups=["resumTNP", "resum", "pTModeling", "theory", "theory_qcd"],
        )

    for up_var, down_var, nuisance_name in TRANSITION_FO_UNCERTAINTIES:
        writer.add_systematic(
            [nominal[{"vars": up_var}], nominal[{"vars": down_var}]],
            nuisance_name,
            PROCESS_NAME,
            SIGMAUL_CHANNEL,
            symmetrize="quadratic",
            groups=[
                "resumTransitionFOScale",
                "resum",
                "pTModeling",
                "theory",
                "theory_qcd",
            ],
        )


def add_pdf_variations(args, writer, logger, pdf_name, scale_pdf: float | None = None):
    pdf_var_key = _pdfvars_generator_name(args.predGenerator)
    keys_to_load = [pdf_var_key]

    pdfInfo = theory_utils.pdf_info_map("Zmumu_2016PostVFP", args.pdfs[0])
    pdfName = pdfInfo["name"]

    if scale_pdf is not None:
        scale = scale_pdf
    else:
        scale = pdfInfo.get("inflation_factor_alphaS", 1)
        scale = pdfInfo.get("scale", 1) * scale
    logger.debug(f"Using scale {scale}.")

    pdf_var_key_ext = None
    if pdfName == "pdfHERAPDF20" and not args.noHERAPDF20EXT:
        pdf_var_key_ext = pdf_var_key.replace("HERAPDF20", "HERAPDF20EXT")
        keys_to_load.append(pdf_var_key_ext)

    corr_helpers = theory_corrections.load_corr_helpers(
        ["Z"],
        keys_to_load,
        make_tensor=False,
        minnlo_ratio=False,
    )

    pdf_groups = [pdf_name, f"{pdf_name}NoAlphaS", "theory", "theory_qcd"]

    h = corr_helpers["Z"][pdf_var_key]
    for ivar in range(1, len(h.axes[-1]), 2):
        writer.add_systematic(
            [h[{"vars": ivar + 1}], h[{"vars": ivar}]],
            f"pdf{int((ivar + 1) / 2)}{args.pdfs[0].upper()}",
            PROCESS_NAME,
            SIGMAUL_CHANNEL,
            symmetrize="quadratic",
            kfactor=scale,
            groups=pdf_groups,
        )

    if pdf_var_key_ext is not None:
        h_ext = corr_helpers["Z"][pdf_var_key_ext]
        extInfo = theory_utils.pdfMap["herapdf20ext"]
        n_entries = extInfo["entries"]
        n_sym = 3
        n_asym_entries = n_entries - n_sym

        ext_suffix = "HERAPDF20EXT"

        # Asymmetric hessian variations
        for ivar in range(1, n_asym_entries, 2):
            writer.add_systematic(
                [h_ext[{"vars": ivar + 1}], h_ext[{"vars": ivar}]],
                f"pdf{int((ivar + 1) / 2)}{ext_suffix}",
                PROCESS_NAME,
                SIGMAUL_CHANNEL,
                symmetrize="quadratic",
                kfactor=scale,
                groups=pdf_groups,
            )

        # Symmetric hessian variations (mirrored)
        n_asym_pairs = (n_asym_entries - 1) // 2
        for j, ivar in enumerate(range(n_asym_entries, n_entries)):
            writer.add_systematic(
                h_ext[{"vars": ivar}],
                f"pdf{n_asym_pairs + j + 1}{ext_suffix}",
                PROCESS_NAME,
                SIGMAUL_CHANNEL,
                symmetrize="quadratic",
                kfactor=scale,
                mirror=True,
                groups=pdf_groups,
            )


def add_ew_isr_variation(args, writer):
    ew_isr_name = "pythiaew_ISR"
    corrh_num = theory_corrections.load_corr_hist(
        f"{common.data_dir}/TheoryCorrections/{ew_isr_name}_CorrZ.pkl.lz4",
        "Z",
        f"{ew_isr_name}_num",
    )
    corrh_den = theory_corrections.load_corr_hist(
        f"{common.data_dir}/TheoryCorrections/{ew_isr_name}_CorrZ.pkl.lz4",
        "Z",
        f"{ew_isr_name}_den",
    )
    writer.add_shape_systematic(
        [corrh_num, corrh_den],
        f"{ew_isr_name}_Corr",
        PROCESS_NAME,
        SIGMAUL_CHANNEL,
        kfactor=2,
        mirror=True,
        symmetrize="average",
        groups=["theory_ew", "theory"],
    )


def add_mb_fo_variations(args, writer):
    mb_fo_name = "MiNNLO_Zbb"
    numh = theory_corrections.load_corr_hist(
        f"{common.data_dir}/TheoryCorrections/{mb_fo_name}_CorrZ.pkl.lz4",
        "Z",
        f"{mb_fo_name}_hist",
    )
    denh = theory_corrections.load_corr_hist(
        f"{common.data_dir}/TheoryCorrections/{mb_fo_name}_CorrZ.pkl.lz4",
        "Z",
        f"minnlo_ref_hist",
    )
    writer.add_shape_systematic(
        [numh[{"vars": "mb_up"}], denh[{"vars": "mb_up"}]],
        "mb_fo",
        PROCESS_NAME,
        SIGMAUL_CHANNEL,
        mirror=True,
        groups=["bcQuarkMass", "theory"],
    )


def output_name(args):
    name = args.outname
    name += f"_{args.predGenerator}"
    name += f"_{'_'.join(args.nois)}"
    if args.postfix and len(args.postfix) > 0:
        name += f"_{args.postfix}"
    return name


def make_parser():
    analysis_label = common.analysis_label(os.path.basename(__file__))
    parser, _ = parsing.common_parser(analysis_label)
    parser.description = (
        "Write rabbit tensors for direct-theory sigmaUL fits from unfolded sigmaUL "
        "fit results and TheoryCorrections histograms."
    )

    parser.add_argument(
        "-i",
        "--infile",
        type=str,
        help="Input unfolded fit result for the Z sigmaUL distribution.",
    )
    parser.add_argument(
        "--fitresultMapping",
        nargs="+",
        default=["Select helicitySig:0"],
        help="Physics-model mapping to read from the unfolded fit result.",
    )
    parser.add_argument(
        "--channelSigmaUL",
        nargs="+",
        default=["ch0_masked"],
        help="Channel name for the sigmaUL distribution inside the selected fit-result mapping.",
    )
    parser.add_argument(
        "--pseudodataGenerator",
        type=str,
        default="",
        help="Optional generator name to use as sigmaUL pseudodata instead of reading an unfolded fit result.",
    )
    parser.add_argument(
        "--predGenerator",
        type=str,
        default="scetlib_dyturbo_LatticeNP_CT18Z_N3p0LL_N2LO",
        help="Generator used for the sigmaUL prediction and direct theory variations.",
    )
    parser.add_argument(
        "--nois",
        nargs="+",
        type=str,
        default=["alphaS"],
        choices=["alphaS"],
        help="Parameters of interest to expose in the written rabbit input.",
    )
    parser.add_argument("--outname", default="carrot", help="Output file name stem.")
    parser.add_argument(
        "--excludeNuisances",
        type=str,
        default="",
        help="Regex for nuisance names to exclude before writing the tensor.",
    )
    parser.add_argument(
        "--keepNuisances",
        type=str,
        default="",
        help="Regex for nuisance names to keep. If set, only matching nuisances are included.",
    )
    parser.add_argument(
        "--sparse",
        default=False,
        action="store_true",
        help="Write a sparse tensor.",
    )
    parser.add_argument(
        "--systematicType",
        choices=["log_normal", "normal"],
        default="normal",
        help="Probability density for systematic variations.",
    )
    parser.add_argument(
        "--scalePdf",
        type=float,
        default=None,
        help="Manually set the scale factor for PDF variations (overrides any scale specified in the theory corrections metadata).",
    )
    parser.add_argument(
        "--noHERAPDF20EXT",
        action="store_true",
        help="Exclude the HERAPDF20EXT variations (only applicable if using a HERAPDF20-based PDF). Useful for comparing to simultaneous PDF and alphaS fit, where this parametrization isn't available.",
    )
    return parser


def _validate_args(args):
    """
    Make sure the the first PDF (the only one used) matches the args.predGenerator.
    TODO: at some point, we should have a dataclass for each theory correction that specifies which PDF it belongs to, so we don't have to rely on string parsing of the generator name.
    """
    if args.pdfs[0].lower() not in args.predGenerator.lower():
        raise ValueError(
            f"Make sure the that the PDF you pass (--pdfs) matches the --predGenerator name."
        )


def main():
    parser = make_parser()
    args = parser.parse_args()
    args.fitresultMapping = _join_cli_tokens(args.fitresultMapping)
    args.channelSigmaUL = _join_cli_tokens(args.channelSigmaUL)
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    _validate_args(args)

    writer = SigmaULTheoryFitWriter(
        sparse=args.sparse,
        systematic_type=args.systematicType,
        allow_negative_expectation=False,
        exclude_nuisances=args.excludeNuisances,
        keep_nuisances=args.keepNuisances,
    )

    input_meta = load_sigmaul_data(args, writer, logger)
    add_sigmaul_process(args, writer)

    pdf_name = theory_utils.pdfMap[args.pdfs[0]]["name"]
    logger.info("Adding alphaS variation")
    add_alphas_variation(args, writer, pdf_name)

    logger.info("Adding direct-theory sigmaUL systematics from %s", args.predGenerator)
    add_resummation_and_np_variations(args, writer)

    logger.info("Adding PDF b/c quark-mass variations")
    add_pdf_bc_quark_mass_variations(args, writer, logger, pdf_name)

    logger.info("Adding mb FO variations")
    add_mb_fo_variations(args, writer)

    logger.info("Adding PDF variations")
    add_pdf_variations(args, writer, logger, pdf_name, args.scalePdf)

    logger.info("Adding EW ISR variation")
    add_ew_isr_variation(args, writer)

    outfolder = args.outfolder or "./"
    meta = build_output_metadata(args, input_meta)
    writer.write(
        outfolder=outfolder,
        outfilename=output_name(args),
        meta_data_dict=meta,
    )
    logger.info("Written to %s.hdf5", os.path.join(outfolder, output_name(args)))


if __name__ == "__main__":
    main()
