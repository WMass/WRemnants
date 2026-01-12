import os

import matplotlib as mpl

from wremnants.utilities import common, parsing
from wums import logging

mpl.use("Agg")
mpl.rcParams["figure.dpi"] = 300


def parse_args():
    analysis_label = common.analysis_label(os.path.basename(__file__))
    parser, _ = parsing.common_parser(analysis_label)
    parser.add_argument(
        "--inputFile",
        type=str,
        required=True,
        help="Input HDF5 file with response histograms",
    )
    parser.add_argument(
        "--outputDir",
        type=str,
        required=True,
        help="Directory where the output TFLite file will be written",
    )
    parser.add_argument(
        "--particleType",
        type=str,
        default="muon",
        choices=["kaon", "muon"],
        help="Particle type used to select the default process list and output name",
    )
    parser.add_argument(
        "--procs",
        type=str,
        nargs="*",
        default=None,
        help="Processes to merge. If omitted, use the particle-type defaults",
    )
    parser.add_argument(
        "--interpSigmaMin",
        type=float,
        default=-5.0,
        help="Minimum sigma value used to define interpolation cdf points",
    )
    parser.add_argument(
        "--interpSigmaMax",
        type=float,
        default=5.0,
        help="Maximum sigma value used to define interpolation cdf points",
    )
    parser.add_argument(
        "--interpSigmaSteps",
        type=int,
        default=21,
        help="Number of sigma steps used to define interpolation cdf points",
    )
    parser.add_argument(
        "--runDebugChecks",
        action="store_true",
        help="Evaluate interpolation and derivative outputs at a single test point",
    )
    parser.add_argument(
        "--debugGenPt",
        type=float,
        default=25.0,
        help="genPt used for debug checks and plots",
    )
    parser.add_argument(
        "--debugGenEta",
        type=float,
        default=0.1,
        help="genEta used for debug checks and plots",
    )
    parser.add_argument(
        "--debugGenCharge",
        type=float,
        default=1.0,
        help="genCharge used for debug checks and plots",
    )
    parser.add_argument(
        "--debugQopr",
        type=float,
        default=1.002,
        help="qopr used for the debug check point",
    )
    parser.add_argument(
        "--makePlots",
        action="store_true",
        help="Produce debug plots",
    )
    parser.add_argument(
        "--plotDir",
        type=str,
        default=None,
        help="Output directory for debug plots. Defaults to outputDir/response_map_debug",
    )
    parser.add_argument(
        "--plotNqoprPoints",
        type=int,
        default=400,
        help="Number of qopr points used in debug scan plots",
    )
    parser.add_argument(
        "--runMonotonicityChecks",
        action="store_true",
        help="Scan charge/pt/eta bins for quantile, CDF, and PDF monotonicity issues",
    )
    parser.add_argument(
        "--plotAllChecks",
        action="store_true",
        help="Write monotonicity-check plots for all scanned bins",
    )
    parser.add_argument(
        "--plotOnlyIssues",
        action="store_true",
        help="Write monotonicity-check plots only for bins with detected issues",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.plotAllChecks and args.plotOnlyIssues:
        raise ValueError("Use at most one of --plotAllChecks and --plotOnlyIssues")

    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    from wremnants.postprocessing import response_maps_utils

    procs = response_maps_utils.resolve_procs(args)
    output_name = response_maps_utils.output_name_with_postfix(
        args.particleType, args.postfix
    )
    logger.info("Reading response maps from %s", args.inputFile)
    logger.info("Using particle type %s with procs %s", args.particleType, procs)

    hist_response, hist_response_scaled, hist_response_smeared = (
        response_maps_utils.load_merged_response_histograms(args.inputFile, procs)
    )
    hist_response, hist_response_scaled, hist_response_smeared = (
        response_maps_utils.project_response_histograms(
            hist_response, hist_response_scaled, hist_response_smeared
        )
    )
    interp_cdfvals = response_maps_utils.make_interp_cdfvals(
        args.interpSigmaMin, args.interpSigmaMax, args.interpSigmaSteps
    )

    response_maps_utils.log_histogram_summary(
        hist_response,
        hist_response_scaled,
        hist_response_smeared,
        interp_cdfvals,
    )

    response_map = response_maps_utils.build_response_map_interpolator(
        hist_response,
        hist_response_scaled,
        hist_response_smeared,
        interp_cdfvals,
    )
    response_maps_utils.log_quantile_diagnostics(response_map)
    response_maps_utils.log_interpolator_bounds(response_map)

    if args.runDebugChecks:
        response_maps_utils.run_debug_check(
            response_map,
            args.debugGenPt,
            args.debugGenEta,
            args.debugGenCharge,
            args.debugQopr,
        )

    if args.makePlots:
        plot_dir = args.plotDir or os.path.join(args.outputDir, "response_map_debug")
        response_maps_utils.make_debug_plots(
            response_map,
            plot_dir,
            args.debugGenPt,
            args.debugGenEta,
            args.debugGenCharge,
            args.plotNqoprPoints,
        )

    if args.runMonotonicityChecks:
        monotonicity_plot_dir = None
        if args.plotAllChecks or args.plotOnlyIssues:
            monotonicity_plot_dir = args.plotDir or os.path.join(
                args.outputDir, "response_map_monotonicity_checks"
            )
        response_maps_utils.run_monotonicity_checks(
            response_map,
            hist_response,
            plot_dir=monotonicity_plot_dir,
            plot_all_checks=args.plotAllChecks,
            plot_only_issues=args.plotOnlyIssues,
        )

    tflite_model = response_maps_utils.make_tflite_model(response_map)
    response_maps_utils.write_tflite_model(args.outputDir, output_name, tflite_model)


if __name__ == "__main__":
    main()
