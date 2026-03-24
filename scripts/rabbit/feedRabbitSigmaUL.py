import copy
import os
import pprint
import re

import numpy as np

import rabbit.io_tools
from rabbit.tensorwriter import TensorWriter
from wremnants.production import theory_corrections
from wremnants.utilities import common, parsing, theory_utils
from wums import boostHistHelpers as hh
from wums import logging, output_tools

SIGMAUL_CHANNEL = "chSigmaUL"
PROCESS_NAME = "Zmumu"

STANDARD_CORRELATED_NP_UNCERTAINTIES = [
    ["Lambda20.25", "Lambda2-0.25", "chargeVgenNP0scetlibNPZLambda2"],
    ["Lambda4.16", "Lambda4.01", "chargeVgenNP0scetlibNPZLambda4"],
    [
        "Delta_Lambda20.02",
        "Delta_Lambda2-0.02",
        "chargeVgenNP0scetlibNPZDelta_Lambda2",
    ],
]
LATTICE_CORRELATED_NP_UNCERTAINTIES = [
    ["lambda20.5", "lambda20.0", "chargeVgenNP0scetlibNPLambda2"],
    ["lambda40.16", "lambda40.01", "chargeVgenNP0scetlibNPLambda4"],
    [
        "delta_lambda20.105",
        "delta_lambda20.145",
        "chargeVgenNP0scetlibNPDelta_Lambda2",
    ],
]
STANDARD_GAMMA_NP_UNCERTAINTIES = [
    ["omega_nu0.5", "c_nu-0.1-omega_nu0.5", "scetlibNPgamma"],
]
LATTICE_GAMMA_NP_UNCERTAINTIES = [
    [
        "lambda2_nu0.0696-lambda4_nu0.0122-lambda_inf_nu1.1Ext",
        "lambda2_nu0.1044-lambda4_nu0.0026-lambda_inf_nu2.1Ext",
        "scetlibNPgammaEigvar1",
    ],
    [
        "lambda2_nu0.1153-lambda4_nu0.0032-lambda_inf_nu1.6Ext",
        "lambda2_nu0.0587-lambda4_nu0.0116-lambda_inf_nu1.6Ext",
        "scetlibNPgammaEigvar2",
    ],
    [
        "lambda2_nu0.0873-lambda4_nu0.0092",
        "lambda2_nu0.0867-lambda4_nu0.0056",
        "scetlibNPgammaEigvar3",
    ],
]
TNP_UNCERTAINTIES = [
    ["gamma_cusp1.", "gamma_cusp-1."],
    ["gamma_mu_q1.", "gamma_mu_q-1."],
    ["gamma_nu1.", "gamma_nu-1."],
    ["h_qqV1.", "h_qqV-1."],
    ["s1.", "s-1."],
    ["b_qqV0.5", "b_qqV-0.5"],
    ["b_qqbarV0.5", "b_qqbarV-0.5"],
    ["b_qqS0.5", "b_qqS-0.5"],
    ["b_qqDS0.5", "b_qqDS-0.5"],
    ["b_qg0.5", "b_qg-0.5"],
]
TRANSITION_FO_UNCERTAINTIES = [
    [
        "transition_points0.2_0.75_1.0",
        "transition_points0.2_0.35_1.0",
        "resumTransitionZ",
    ],
    [
        "renorm_scale_pt20_envelope_Up",
        "renorm_scale_pt20_envelope_Down",
        "resumFOScaleZ",
    ],
]
BC_QUARK_MASS_VARIATIONS = [
    (
        "scetlib_dyturbo_LatticeNP_MSHT20mbrange_N3p0LL_N2LO_pdfvars",
        "pdfMSHT20mbrange",
        "pdf1",
        "pdf6",
    ),
    (
        "scetlib_dyturbo_LatticeNP_MSHT20mcrange_N3p0LL_N2LO_pdfvars",
        "pdfMSHT20mcrange",
        "pdf1",
        "pdf8",
    ),
]


class SigmaULTheoryFitWriter(TensorWriter):
    """
    Tensor writer for the direct-theory sigmaUL fit.
    """

    def __init__(self, exclude_nuisances="", keep_nuisances="", **kwargs):
        super().__init__(**kwargs)

        self.logger = logging.child_logger(__name__)
        self.ref = {}
        self._exclude_nuisances = (
            re.compile(exclude_nuisances) if exclude_nuisances else None
        )
        self._keep_nuisances = re.compile(keep_nuisances) if keep_nuisances else None

    def _keep_systematic(self, name):
        if self._exclude_nuisances and self._exclude_nuisances.search(name):
            return False
        if self._keep_nuisances and not self._keep_nuisances.search(name):
            return False
        return True

    def set_reference(self, channel, h, lumi=1.0, scale=1.0, postOp=None):
        self.ref[channel] = {
            "h": h,
            "lumi": lumi,
            "scale": scale,
            "postOp": postOp,
            "ptV_name": self.get_ptV_axis_name(h),
            "absYV_name": self.get_absYV_axis_name(h),
            "chargeV_name": self.get_charge_axis_name(h),
            "ptV_bins": h.axes[self.get_ptV_axis_name(h)].edges,
            "absYV_bins": h.axes[self.get_absYV_axis_name(h)].edges,
        }
        self.logger.debug("Initialized channel %s with parameters", channel)
        self.logger.debug(pprint.pformat(self.ref[channel]))

    def add_systematic(
        self,
        h,
        name,
        process,
        channel,
        rebin_pt=True,
        rebin_y=True,
        normalize=True,
        apply_postOp=True,
        format=True,
        **kwargs,
    ):
        if not self._keep_systematic(name):
            self.logger.info(
                "Skipping systematic '%s' for process '%s' in channel '%s' due to nuisance filtering.",
                name,
                process,
                channel,
            )
            return

        if format:
            if isinstance(h, (list, tuple)):
                h[0] = self.format(
                    h[0],
                    channel,
                    process,
                    rebin_pt=rebin_pt,
                    rebin_y=rebin_y,
                    normalize=normalize,
                    apply_postOp=apply_postOp,
                )
                h[1] = self.format(
                    h[1],
                    channel,
                    process,
                    rebin_pt=rebin_pt,
                    rebin_y=rebin_y,
                    normalize=normalize,
                    apply_postOp=apply_postOp,
                )
            elif kwargs.get("mirror"):
                h = self.format(
                    h,
                    channel,
                    process,
                    rebin_pt=rebin_pt,
                    rebin_y=rebin_y,
                    normalize=normalize,
                    apply_postOp=apply_postOp,
                )

        super().add_systematic(h, name, process, channel, **kwargs)

    def add_scale_systematic(
        self,
        h,
        name,
        process,
        channel,
        rebin_pt=True,
        rebin_y=True,
        normalize=True,
        apply_postOp=True,
        format=True,
        **kwargs,
    ):
        if not self._keep_systematic(name):
            self.logger.info(
                "Skipping systematic '%s' for process '%s' in channel '%s' due to nuisance filtering.",
                name,
                process,
                channel,
            )
            return

        if not kwargs.get("mirror"):
            if format:
                h[0] = self.format(
                    h[0],
                    channel,
                    process,
                    rebin_pt=rebin_pt,
                    rebin_y=rebin_y,
                    normalize=normalize,
                    apply_postOp=apply_postOp,
                )
                h[1] = self.format(
                    h[1],
                    channel,
                    process,
                    rebin_pt=rebin_pt,
                    rebin_y=rebin_y,
                    normalize=normalize,
                    apply_postOp=apply_postOp,
                )
                h[2] = self.format(
                    h[2],
                    channel,
                    process,
                    rebin_pt=rebin_pt,
                    rebin_y=rebin_y,
                    normalize=normalize,
                    apply_postOp=apply_postOp,
                )

            hup = hh.divideHists(h[0], h[2])
            hdown = hh.divideHists(h[1], h[2])
            hup = hh.multiplyHists(hup, self.ref[channel][process])
            hdown = hh.multiplyHists(hdown, self.ref[channel][process])
            super().add_systematic([hup, hdown], name, process, channel, **kwargs)
        else:
            if format:
                h[0] = self.format(
                    h[0],
                    channel,
                    process,
                    rebin_pt=rebin_pt,
                    rebin_y=rebin_y,
                    normalize=normalize,
                    apply_postOp=apply_postOp,
                )
                h[1] = self.format(
                    h[1],
                    channel,
                    process,
                    rebin_pt=rebin_pt,
                    rebin_y=rebin_y,
                    normalize=normalize,
                    apply_postOp=apply_postOp,
                )

            mirrored = hh.divideHists(h[0], h[1])
            mirrored = hh.multiplyHists(mirrored, self.ref[channel][process])
            super().add_systematic(mirrored, name, process, channel, **kwargs)

    def add_process(
        self,
        h,
        name,
        channel,
        rebin_pt=True,
        rebin_y=True,
        normalize=True,
        apply_postOp=True,
        **kwargs,
    ):
        h = self.format(
            h,
            channel,
            name,
            rebin_pt=rebin_pt,
            rebin_y=rebin_y,
            normalize=normalize,
            apply_postOp=apply_postOp,
        )
        super().add_process(h, name, channel, **kwargs)
        self.ref[channel][name] = h

    def format(
        self,
        h,
        channel,
        process,
        rebin_pt=True,
        rebin_y=True,
        normalize=True,
        apply_postOp=True,
    ):
        h = copy.deepcopy(h)
        h = self.apply_selections(h, process, channel)

        pt_axis_name = self.get_ptV_axis_name(h)
        absY_axis_name = self.get_absYV_axis_name(h)
        charge_axis_name = self.get_charge_axis_name(h)

        hh.renameAxis(h, pt_axis_name, self.ref[channel]["ptV_name"])
        hh.renameAxis(h, absY_axis_name, self.ref[channel]["absYV_name"])
        if charge_axis_name:
            hh.renameAxis(h, charge_axis_name, self.ref[channel]["chargeV_name"])

        h = hh.setFlow(h, self.ref[channel]["ptV_name"], under=False, over=True)

        if rebin_pt:
            h = hh.rebinHist(
                h, self.ref[channel]["ptV_name"], self.ref[channel]["ptV_bins"]
            )
        if rebin_y:
            h = hh.rebinHist(
                h, self.ref[channel]["absYV_name"], self.ref[channel]["absYV_bins"]
            )
        if normalize:
            h *= self.ref[channel]["lumi"] * self.ref[channel]["scale"]

        remaining_axes = list(h.axes.name)
        remaining_axes.remove(self.ref[channel]["ptV_name"])
        remaining_axes.remove(self.ref[channel]["absYV_name"])
        h = h.project(
            self.ref[channel]["ptV_name"],
            self.ref[channel]["absYV_name"],
            *remaining_axes,
        )

        if self.ref[channel]["postOp"] is not None and apply_postOp:
            h = self.ref[channel]["postOp"](h)

        return h

    def get_ptV_axis_name(self, h):
        for name in ["ptVgen", "ptVGen", "qT"]:
            if name in h.axes.name:
                return name
        self.logger.debug("Did not find pT axis. Available axes: %s", h.axes.name)

    def get_absYV_axis_name(self, h):
        for name in ["absYVgen", "absYVGen", "absY"]:
            if name in h.axes.name:
                return name
        self.logger.debug("Did not find absY axis. Available axes: %s", h.axes.name)

    def get_charge_axis_name(self, h):
        for name in ["chargeVgen", "charge", "qGen"]:
            if name in h.axes.name:
                return name
        return None

    def get_mass_axis_name(self, h):
        for name in ["massVgen", "Q"]:
            if name in h.axes.name:
                return name
        return None

    def apply_selections(self, h, process, channel):
        if process != PROCESS_NAME:
            raise ValueError(f"Unsupported process '{process}' for sigmaUL writer")

        mass_axis_name = self.get_mass_axis_name(h)
        if mass_axis_name:
            h = h[{mass_axis_name: 90.0j}]

        charge_axis_name = self.get_charge_axis_name(h)
        if charge_axis_name:
            h = h[{charge_axis_name: 0.0j}]

        if channel == SIGMAUL_CHANNEL and "helicity" in h.axes.name:
            h = h[{"helicity": -1.0j}]

        return h


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


def apply_coarse_correction(fine_nominal, coarse_ratio, check_align=True):
    """Apply a coarse multiplicative ratio onto the fine direct nominal.

    This is used for nuisance sources that are only available on a coarser grid.
    The intended operation is:

        fine_nominal * (coarse_var / coarse_nominal)
    """
    fine_axes = fine_nominal.axes
    coarse_axes = coarse_ratio.axes

    for ax in coarse_axes:
        if ax.name not in fine_axes.name:
            raise ValueError(
                f"Axis '{ax.name}' in correction histogram not found in fine histogram."
            )

        if check_align:
            fine_edges = fine_axes[ax.name].edges
            coarse_edges = ax.edges
            coarse_edges = coarse_edges[
                (coarse_edges >= min(fine_edges)) & (coarse_edges <= max(fine_edges))
            ]
            if not all(edge in fine_edges for edge in coarse_edges):
                raise ValueError(
                    f"Not all edges of axis '{ax.name}' in correction histogram are present in fine histogram."
                )

    corrected = copy.deepcopy(fine_nominal)
    other_axes = [axis for axis in fine_axes.name if axis not in coarse_axes.name]
    corrected = corrected.project(*other_axes, *coarse_axes.name)

    corrected_values = corrected.values()
    for fine_bin_idx in np.ndindex(corrected_values.shape[len(other_axes) :]):
        centers = tuple(
            corrected.axes[len(other_axes) + i].centers[idx] * 1.0j
            for i, idx in enumerate(fine_bin_idx)
        )
        corrected_values[(Ellipsis, *fine_bin_idx)] *= coarse_ratio[centers].value

    corrected.values()[...] = corrected_values
    return corrected.project(*fine_axes.name)


def rename_to_reference_axes(h, writer, channel):
    h = copy.deepcopy(h)
    pt_axis_name = writer.get_ptV_axis_name(h)
    absY_axis_name = writer.get_absYV_axis_name(h)
    charge_axis_name = writer.get_charge_axis_name(h)

    if pt_axis_name and pt_axis_name != writer.ref[channel]["ptV_name"]:
        hh.renameAxis(h, pt_axis_name, writer.ref[channel]["ptV_name"])
    if absY_axis_name and absY_axis_name != writer.ref[channel]["absYV_name"]:
        hh.renameAxis(h, absY_axis_name, writer.ref[channel]["absYV_name"])
    ref_charge_name = writer.ref[channel]["chargeV_name"]
    if charge_axis_name and ref_charge_name and charge_axis_name != ref_charge_name:
        hh.renameAxis(h, charge_axis_name, ref_charge_name)

    return h


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


def add_bc_quark_mass_variations(writer):
    corr_helpers = theory_corrections.load_corr_helpers(
        ["Z"],
        [helper_name for helper_name, *_ in BC_QUARK_MASS_VARIATIONS],
        make_tensor=False,
        minnlo_ratio=False,
    )

    for helper_name, nuisance_name, down_var, up_var in BC_QUARK_MASS_VARIATIONS:
        h = corr_helpers["Z"][helper_name]
        # Use the same multiplicative construction as add_scale_systematic:
        # nominal * (quark_mass_var / quark_mass_nominal).
        writer.add_scale_systematic(
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


def add_pdf_variations(args, writer, pdf_name):
    pdf_var_key = _pdfvars_generator_name(args.predGenerator)
    corr_helpers = theory_corrections.load_corr_helpers(
        ["Z"],
        [pdf_var_key],
        make_tensor=False,
        minnlo_ratio=False,
    )

    h = corr_helpers["Z"][pdf_var_key]
    for ivar in range(1, len(h.axes[-1]), 2):
        writer.add_systematic(
            [h[{"vars": ivar + 1}], h[{"vars": ivar}]],
            f"pdf{int((ivar + 1) / 2)}{args.pdfs[0].upper()}",
            PROCESS_NAME,
            SIGMAUL_CHANNEL,
            symmetrize="quadratic",
            kfactor=1 / 1.645,
            groups=[pdf_name, f"{pdf_name}NoAlphaS", "theory", "theory_qcd"],
        )


def output_name(args):
    name = args.outname
    name += f"_{args.predGenerator}"
    name += f"_{args.pdfs[0]}"
    name += f"_{'_'.join(args.nois)}"
    if args.postfix:
        name += f"_{args.postfix}"
    return name


def build_parser():
    analysis_label = common.analysis_label(os.path.basename(__file__))
    parser, _ = parsing.common_parser(analysis_label)

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
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.fitresultMapping = _join_cli_tokens(args.fitresultMapping)
    args.channelSigmaUL = _join_cli_tokens(args.channelSigmaUL)
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

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

    logger.info("Adding lattice-corrected b/c quark-mass variations")
    add_bc_quark_mass_variations(writer)

    logger.info("Adding PDF variations")
    add_pdf_variations(args, writer, pdf_name)

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
