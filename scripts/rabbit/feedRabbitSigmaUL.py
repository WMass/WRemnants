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

from wremnants.postprocessing.theory_fit_writer import SigmaULTheoryFitWriter
from wremnants.utilities import common, parsing
from wums import logging, output_tools


def _join_cli_tokens(value):
    if isinstance(value, (list, tuple)):
        return " ".join(value)
    return value


def output_name(outname, predGenerator, nois, postfix):
    name = outname
    name += f"_{predGenerator}"
    name += f"_{'_'.join(nois)}"
    if postfix and len(postfix) > 0:
        name += f"_{postfix}"
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


def _validate_args(pdf, predGenerator):
    """
    Make sure the the first PDF (the only one used) matches the predGenerator.
    TODO: at some point, we should have a dataclass for each theory correction that specifies which PDF it belongs to, so we don't have to rely on string parsing of the generator name.
    """
    if pdf.lower() not in predGenerator.lower():
        raise ValueError(
            f"Make sure the that the PDF you pass (--pdfs) matches the --predGenerator name."
        )


def main():
    parser = make_parser()
    args = parser.parse_args()
    args.fitresultMapping = _join_cli_tokens(args.fitresultMapping)
    args.channelSigmaUL = _join_cli_tokens(args.channelSigmaUL)
    logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    _validate_args(args.pdfs[0], args.predGenerator)

    exclude_nuisances = args.excludeNuisances
    if args.noHERAPDF20EXT:
        exclude_nuisances = (
            f"{exclude_nuisances}|HERAPDF20EXT" if exclude_nuisances else "HERAPDF20EXT"
        )

    writer = SigmaULTheoryFitWriter(
        predGenerator=args.predGenerator,
        nois=args.nois,
        pdf=args.pdfs[0],
        sparse=args.sparse,
        systematic_type=args.systematicType,
        allow_negative_expectation=False,
        exclude_nuisances=exclude_nuisances,
        keep_nuisances=args.keepNuisances,
    )

    input_meta = writer.load_sigmaul_data(
        args.pseudodataGenerator,
        args.infile,
        args.fitresultMapping,
        args.channelSigmaUL,
    )
    writer.add_sigmaul_process()
    writer.add_alphas_variation()
    writer.add_resummation_and_np_variations()
    writer.add_pdf_bc_quark_mass_variations()
    writer.add_mb_fo_variations()
    writer.add_pdf_variations(args.scalePdf)
    writer.add_ew_isr_variation()

    outfolder = args.outfolder or "./"
    meta = {
        "meta_info": output_tools.make_meta_info_dict(
            args=args,
            wd=common.base_dir,
        ),
    }
    if input_meta is not None:
        meta["meta_info_input"] = input_meta
    outname = output_name(args.outname, args.predGenerator, args.nois, args.postfix)
    writer.write(
        outfolder=outfolder,
        outfilename=outname,
        meta_data_dict=meta,
    )
    writer.logger.info("Written to %s.hdf5", os.path.join(outfolder, outname))


if __name__ == "__main__":
    main()
