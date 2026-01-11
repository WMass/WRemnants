import copy
import os
import pprint

import h5py
import hist
import numpy as np

import rabbit
import rabbit.io_tools
from rabbit.tensorwriter import TensorWriter
from utilities import common, parsing
from utilities.io_tools import input_tools
from wremnants import theory_corrections
from wremnants.datasets.datagroups import Datagroups
from wums import boostHistHelpers as hh
from wums import logging


class AlphaSTheoryFitTW(TensorWriter):
    """
    Tensor writer for the alphaS fit direct to theory predictions.
    Will format histograms correctly before adding them as data, processes, variations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.logger = logging.child_logger(__name__)
        self.ref = {}

    def set_reference(self, channel, h, lumi=16800, scale=1.0, postOp=None):
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
        self.logger.debug(f"Initialized channel {channel} with parameters")
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
        """
        Systematic variations on a process. Formatting is applied.
        """

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

        self.logger.debug(
            f"Adding systematic {name} for process {process} in channel {channel}"
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
        """
        Systematic variations that are computed as
            var_up/var_nominal * nominal
            var_down/var_nominal * nominal (or mirror)
        where nominal is the process in this channel.
        First applies the formatting, then computes the variation.
        """

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

            h = hh.divideHists(h[0], h[1])
            h = hh.multiplyHists(h, self.ref[channel][process])

            super().add_systematic(h, name, process, channel, **kwargs)

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
        ch,
        process,
        rebin_pt=True,
        rebin_y=True,
        normalize=True,
        apply_postOp=True,
    ):
        """
        Apply mass, charge selections to input histogram.
        Re-order axes to match the predictions (pt, absY, ...).
        Optionally, re-bin pt and y.
        Returns histogram with changes applied.
        """

        h = copy.deepcopy(h)  # pass by value

        h = self.apply_selections(h, process, ch)

        pt_axis_name = self.get_ptV_axis_name(h)
        absY_axis_name = self.get_absYV_axis_name(h)
        charge_axis_name = self.get_charge_axis_name(h)

        # rename axes to match the reference histogram
        hh.renameAxis(h, pt_axis_name, self.ref[ch]["ptV_name"])
        hh.renameAxis(h, absY_axis_name, self.ref[ch]["absYV_name"])
        if charge_axis_name:
            hh.renameAxis(h, charge_axis_name, self.ref[ch]["chargeV_name"])

        # disable underflow for pt, for compatibility
        h = hh.disableFlow(h, self.ref[ch]["ptV_name"], under=False, over=True)

        # optionally, rebin pt and y, and scale
        if rebin_pt:
            h = hh.rebinHist(h, self.ref[ch]["ptV_name"], self.ref[ch]["ptV_bins"])
        if rebin_y:
            h = hh.rebinHist(h, self.ref[ch]["absYV_name"], self.ref[ch]["absYV_bins"])
        if normalize:
            h *= self.ref[ch]["lumi"] * self.ref[ch]["scale"]

        # re-order remaining axes to match expected output
        remaining_axes = list(h.axes.name)
        remaining_axes.remove(self.ref[ch]["ptV_name"])
        remaining_axes.remove(self.ref[ch]["absYV_name"])
        ordered_axes = [
            self.ref[ch]["ptV_name"],
            self.ref[ch]["absYV_name"],
            *remaining_axes,
        ]
        h = h.project(*ordered_axes)

        # optionally, apply any post operation
        if self.ref[ch]["postOp"] is not None and apply_postOp:
            h = self.ref[ch]["postOp"](h)

        return h

    def get_ptV_axis_name(self, h):
        for name in ["ptVgen", "ptVGen", "qT"]:
            if name in h.axes.name:
                return name
        self.logger.debug(f"Did not find pT axis! Available axes: {h.axes.name}")

    def get_absYV_axis_name(self, h):
        for name in ["absYVgen", "absYVGen", "absY"]:
            if name in h.axes.name:
                return name
        self.logger.debug(f"Did not find absY axis! Available axes: {h.axes.name}")

    def get_ptLep_axis_name(self, h):
        for name in ["ptGen"]:
            if name in h.axes.name:
                return name
        self.logger.debug(f"Did not find pT axis! Available axes: {h.axes.name}")

    def get_absEtaLep_axis_name(self, h):
        for name in ["absEtaGen"]:
            if name in h.axes.name:
                return name
        self.logger.debug(f"Did not find absY axis! Available axes: {h.axes.name}")

    def get_charge_axis_name(self, h):
        for name in ["chargeVgen", "charge", "qGen"]:
            if name in h.axes.name:
                return name
        self.logger.debug(f"Did not find charge axis! Available axes: {h.axes.name}")

    def get_mass_axis_name(self, h):
        for name in ["massVgen", "Q"]:
            if name in h.axes.name:
                return name
        self.logger.debug(f"Did not find mass axis! Available axes: {h.axes.name}")

    def apply_selections(self, h, process, channel):

        if process == "Zmumu":

            # mass selection
            mass_axis_name = self.get_mass_axis_name(h)
            if mass_axis_name:
                h = h[{mass_axis_name: hist.sum}]

            # charge selection
            charge_axis_name = self.get_charge_axis_name(h)
            if charge_axis_name:
                h = h[{charge_axis_name: 0.0j}]

            # select sigmaUL if needed
            if channel == "chSigmaUL" and "helicity" in h.axes.name:
                h = h[{"helicity": -1.0j}]

        elif process == "Wmunu":

            # mass selection
            mass_axis_name = self.get_mass_axis_name(h)
            if mass_axis_name:
                h = h[{mass_axis_name: hist.sum}]

        return h


def calculate_ais_from_helicities_hist(h_hels):
    """
    Calculate A_i histogram from helicities histogram.
    Assuming as input a histogram with the helicities,
    with the first element being the sigma_UL.
    Returns a the A_i's histogram.
    """

    h_hels = copy.deepcopy(h_hels)  # pass by value

    # re-order histogram axes such that the last one is the helicity
    if h_hels.axes[-1].name != "helicity":
        axes = list(h_hels.axes.name)
        axes.remove("helicity")
        axes.append("helicity")
        h_hels = h_hels.project(*axes)

    vals = copy.copy(h_hels.values())
    vars = copy.copy(
        h_hels.variances()
    )  # need to copy in case the storage is hist.storage.Double

    # A_i = sigma_i/sigma_UL
    num = vals[..., 1:]
    den = vals[..., 0][..., np.newaxis]
    den = np.where(np.abs(den) < 1e-5, np.ones_like(den), den)
    vals[..., 1:] = num / den

    # treat these as poisson uncorrelated rv's
    if np.any(vars):

        vars[..., 1:] = np.where(
            vals[..., 0][..., np.newaxis] == 0,
            0.0,
            den**-2 * vars[..., 1:] + num**2 * den**-4 * vars[..., 0][..., np.newaxis],
        )

    h_out = hist.Hist(
        *h_hels.axes[:-1],
        hist.axis.Integer(0, 8, name="ais"),
        storage=hist.storage.Weight() if np.any(vars) else hist.storage.Double(),
    )
    h_out.values()[...] = vals[..., 1:]
    if np.any(vars):
        h_out.variances()[...] = vars[..., 1:]

    return h_out


def convert_WFull_to_LepFiducial(h_W_lep_fiducial, h_W_lep_inclusive):

    def _convert_WFull_to_LepFiducial(h):

        if "qT" in h.axes.name:
            hh.renameAxis(h, "qT", "ptVgen")
            hh.renameAxis(h, "absY", "absYVgen")
            hh.renameAxis(h, "charge", "chargeVgen")

        if h.axes["absYVgen"].traits.underflow:  # not needed, messes up the dimensions
            h = hh.disableFlow(h, "absYVgen", under=False, over=True)
        if h.axes["ptVgen"].traits.underflow:  # not needed, messes up the dimensions
            h = hh.disableFlow(h, "ptVgen", under=False, over=True)

        correction = hh.divideHists(h, h_W_lep_inclusive, flow=False)
        final_correction = copy.deepcopy(correction)
        final_correction.values(flow=True)[...] = np.ones_like(
            h_W_lep_inclusive.values(flow=True)
        )
        final_correction.values()[...] = correction.values()
        correction = final_correction

        out = hh.multiplyHists(h_W_lep_fiducial, correction)
        out = out.project("absEtaGen", "ptGen", "qGen")

        return out

    return _convert_WFull_to_LepFiducial


def apply_coarse_correction(fine_hist, coarse_corr, check_align=True):
    """
    Apply an N-dimensional correction histogram to an M-dimensional histogram (M>=N),
    which can be finer than the correction, using named axes to align dimensions.
    """

    fine_axes = fine_hist.axes
    coarse_axes = coarse_corr.axes

    # check axes names are compatible
    for ax in coarse_axes:
        if ax.name not in fine_axes.name:
            raise ValueError(
                f"Axis '{ax.name}' in correction histogram not found in fine histogram."
            )

        # optionally, check alignment
        if check_align:
            fine_edges = np.asarray(fine_axes[ax.name].edges)
            coarse_edges = np.asarray(ax.edges)
            coarse_edges = coarse_edges[
                (coarse_edges >= np.min(fine_edges))
                & (coarse_edges <= np.max(fine_edges))
            ]
            if not np.all(np.isin(coarse_edges, fine_edges)):
                raise ValueError(
                    f"Not all edges of axis '{ax.name}' in correction histogram are present in fine histogram."
                )

    # prepare histogram by aligning axes with the correction
    corrected = copy.deepcopy(fine_hist)
    other_axes = [a for a in fine_axes.name if a not in coarse_axes.name]
    corrected = corrected.project(*other_axes, *coarse_axes.name)

    # Iterate over all bins in fine_hist
    corrected_values = corrected.values()
    for fine_bin_idx in np.ndindex(corrected_values.shape[len(other_axes) :]):
        centers = tuple(
            [
                fine_hist.axes[len(other_axes) + i].centers[idx] * 1.0j
                for i, idx in enumerate(fine_bin_idx)
            ]
        )
        # print("Before", corrected_values[..., *fine_bin_idx])
        # print("Correction", coarse_corr[*centers].value)
        corrected_values[..., *fine_bin_idx] *= coarse_corr[*centers].value
        # print("After", corrected_values[..., *fine_bin_idx])

    # restore original order
    corrected.values()[...] = corrected_values
    corrected = corrected.project(*fine_axes.name)

    return corrected


analysis_label = Datagroups.analysisLabel(os.path.basename(__file__))
parser, initargs = parsing.common_parser(analysis_label)

parser.add_argument(
    "-i",
    "--infile",
    type=str,
    help="Input unfolded fit result for the Z distributions.",
)
parser.add_argument(
    "--fitresultMapping", type=str, default="Select helicitySig:slice(0,1)"
)
parser.add_argument(
    "--fitresultChannelSigmaUL",
    type=str,
    default="Select helicitySig:slice(0,1)_ch0_masked",
)
parser.add_argument(
    "--pseudodataGenerator",
    type=str,
    default="",
    help="If set, generate pseudo-data from the given generator prediction "
    "instead of reading from infile.",
)
parser.add_argument(
    "--predGenerator",
    type=str,
    default=f"scetlib_dyturbo",
    help="Generator used for the sigmaUL predictions. "
    "Expect the prediction file to be named as <generator>CorrZ.pkl.lz4. "
    "Expect the prediction histogram to be named as <generator>_hist",
)
parser.add_argument(
    "--constrainAlphaS", action="store_true", help="Constrain alpha_S in the fit."
)
parser.add_argument(
    "--constrainMW", action="store_true", help="Constrain mW in the fit."
)
parser.add_argument("--noFitSigmaUL", action="store_true", help="Don't fit sigmaUL.")
parser.add_argument(
    "--fitAngularCoeffs",
    action="store_true",
    default=False,
    help="Fit the angular coefficients."
    "Predictions of the Ai's from --predAiFile."
    "Predictions of the sigma_UL from infile.",
)
parser.add_argument(
    "--predAiFile",
    type=str,
    default=f"{common.data_dir}/angularCoefficients/w_z_helicity_xsecs_scetlib_dyturboCorr_maxFiles_m1_alphaSunfoldingBinning_helicity.hdf5",
    help="Gen file used for the Ai predictions."
    "Will be stitched with the --predGenerator file.",
)
parser.add_argument(
    "--fitresultChannelAis",
    type=str,
    default="AngularCoefficients ch0_masked ptVGen:rebin(0,3,6,9,12,16,20,24,28,33,44)_ch0_masked",
)
parser.add_argument(
    "-W",
    "--fitW",
    action="store_true",
    help="Include W in the fit."
    "Will use --predWFile for predictions and --infileW for the unfolded distribution.",
)
parser.add_argument("--fitresultChannelW", type=str, default="Select_ch1_masked")
parser.add_argument("--outname", default="carrot", help="output file name")
parser.add_argument(
    "--sparse",
    default=False,
    action="store_true",
    help="Make sparse tensor",
)
parser.add_argument(
    "--systematicType",
    choices=["log_normal", "normal"],
    default="normal",
    help="probability density for systematic variations",
)

args = parser.parse_args()
logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

if not args.fitW:
    args.constrainMW = True  # nothing to fit here...

# Build tensor
writer = AlphaSTheoryFitTW(
    sparse=args.sparse,
    systematic_type="normal" if args.fitAngularCoeffs else args.systematicType,
    allow_negative_expectation=args.fitAngularCoeffs,
)

# load in data histograms and covariance matrix

if args.pseudodataGenerator:
    # generate pseudo-data from a generator prediction

    logger.debug(
        f"Reading from {common.data_dir}/TheoryCorrections/{args.pseudodataGenerator}CorrZ.pkl.lz4"
    )
    h_data_sigmaUL = theory_corrections.load_corr_hist(
        f"{common.data_dir}/TheoryCorrections/{args.pseudodataGenerator}CorrZ.pkl.lz4",
        "Z",
        # f"{args.pseudodataGenerator}_hist".replace("_pdfas", ""),
        "scetlib_nnlojet_N4p0LLN3LOUnsmoothed_N3pLLFixed_hist",
    )
    if "vars" in h_data_sigmaUL.axes.name:
        h_data_sigmaUL = h_data_sigmaUL[
            {"vars": 0}
        ]  # select baseline variation # TODO bit of a hardcode
    h_data_sigmaUL = h_data_sigmaUL.project("qT", "absY")
    h_data_sigmaUL *= 16800  # lumi
    print(
        h_data_sigmaUL.variances(flow=False) ** 0.5 / h_data_sigmaUL.values(flow=False)
    )
    writer.add_channel(h_data_sigmaUL.axes, "chSigmaUL")
    writer.add_data(h_data_sigmaUL, "chSigmaUL")
    writer.set_reference("chSigmaUL", h_data_sigmaUL)

else:

    fitresult, meta = rabbit.io_tools.get_fitresult(
        args.infile, result="asimov", meta=True
    )

    # covariance across any number of physics models
    h_data_cov = fitresult["mappings"][args.fitresultMapping][
        "hist_postfit_inclusive_cov"
    ].get()
    writer.add_data_covariance(h_data_cov)  # N.B: run fit with --externalCovariance

    # the order of add_channel must be OPPOSITE of what is in postfit_cov due to rabbit

    # if set, read and initialize the W lepton channel
    if args.fitW:
        h_data_prefsrLep = fitresult["mappings"][args.fitresultMapping]["channels"][
            args.fitresultChannelW
        ]["hist_postfit_inclusive"].get()
        writer.add_channel(h_data_prefsrLep.axes, "chW")
        writer.add_data(h_data_prefsrLep, "chW")

    # if set, read and initialize Ai's channel
    if args.fitAngularCoeffs:
        h_data_ai = fitresult["mappings"][args.fitresultMapping]["channels"][
            args.fitresultChannelAis
        ]["hist_postfit_inclusive"].get()
        writer.add_channel(h_data_ai.axes, "chAis")
        writer.add_data(h_data_ai, "chAis")
        writer.set_reference(
            "chAis", h_data_ai, postOp=calculate_ais_from_helicities_hist
        )

    # read and initialize sigmaUL channel
    if not args.noFitSigmaUL:
        h_data = fitresult["mappings"][args.fitresultMapping]["channels"][
            args.fitresultChannelSigmaUL
        ]["hist_postfit_inclusive"].get()[:, :, 0]
        writer.add_channel(h_data.axes, "chSigmaUL")
        writer.add_data(h_data, "chSigmaUL")
        writer.set_reference("chSigmaUL", h_data)

# add predictions

#  Z->mumu, from your favorite generator
if not args.noFitSigmaUL:
    h_sig_sigmaUL = theory_corrections.load_corr_hist(
        f"{common.data_dir}/TheoryCorrections/{args.predGenerator}CorrZ.pkl.lz4",
        "Z",
        f"{args.predGenerator}_hist",
    )
    h_sig_sigmaUL = h_sig_sigmaUL[{"vars": "pdf0"}]  # select baseline variation
    writer.add_process(h_sig_sigmaUL, "Zmumu", "chSigmaUL", signal=False)

# if set, load in the helicity cross sections predictions from MINNLO
if args.fitAngularCoeffs:
    with h5py.File(args.predAiFile, "r") as ff:
        inputs = input_tools.load_results_h5py(ff)
        h_sig_hels = inputs["Z"]
    h_sig_hels = h_sig_hels[
        {"muRfact": 1.0j, "muFfact": 1.0j}
    ]  # select baseline variation
    writer.add_process(h_sig_hels, "Zmumu", "chAis", signal=False)

# if set, load in the W lepton distributions
if args.fitW:

    # MiNNLO(W, lep; fiducial lep)
    predWFiducialFile = "/ceph/submit/data/group/cms/store/user/lavezzo/alphaS//250710_gen_histmaker/w_z_gen_dists.cash_maxFiles_m1_MiNNLO_lepInclusive.hdf5"
    # TODO move this in wremnants-data? possibly only extract the hists we need
    with h5py.File(predWFiducialFile, "r") as h5file:

        lumi = 16800
        inputs = input_tools.load_results_h5py(h5file)

        weight_sum = inputs["WplusmunuPostVFP"]["weight_sum"]
        xsec = inputs["WplusmunuPostVFP"]["dataset"]["xsec"]
        h_Wp_lep_fiducial = inputs["WplusmunuPostVFP"]["output"][
            "nominal_gen_prefsrlep"
        ].get()
        h_Wp_lep_fiducial *= xsec * lumi / weight_sum

        weight_sum = inputs["WminusmunuPostVFP"]["weight_sum"]
        xsec = inputs["WminusmunuPostVFP"]["dataset"]["xsec"]
        h_Wm_lep_fiducial = inputs["WminusmunuPostVFP"]["output"][
            "nominal_gen_prefsrlep"
        ].get()
        h_Wm_lep_fiducial *= xsec * lumi / weight_sum

        # important to set flow=True since we are inclusive in W gen
        h_W_lep_fiducial = hh.addHists(
            h_Wp_lep_fiducial,
            h_Wm_lep_fiducial,
            flow=True,
            allowBroadcast=False,
            createNew=False,
        )
        h_W_lep_fiducial = hh.disableFlow(
            h_W_lep_fiducial, "ptVgen", under=False, over=True
        )
        h_W_lep_fiducial = hh.disableFlow(
            h_W_lep_fiducial, "absYVgen", under=False, over=True
        )
        h_W_lep_fiducial = hh.disableFlow(
            h_W_lep_fiducial, "absEtaGen", under=False, over=True
        )
        h_W_lep_fiducial = h_W_lep_fiducial.project(
            "ptVgen", "absYVgen", "chargeVgen", "absEtaGen", "ptGen", "qGen"
        )
        h_W_lep_inclusive = h_W_lep_fiducial.project("ptVgen", "absYVgen", "chargeVgen")
        h_W_lep_fiducial = hh.rebinHist(
            h_W_lep_fiducial, "ptGen", h_data_prefsrLep.axes["ptGen"].edges, flow=True
        )
        h_W_lep_fiducial = hh.rebinHist(
            h_W_lep_fiducial,
            "absEtaGen",
            h_data_prefsrLep.axes["absEtaGen"].edges,
            flow=True,
        )

    # <your favorite generator>(W; full)
    h_pred_W_full = theory_corrections.load_corr_hist(
        f"{common.data_dir}/TheoryCorrections/{args.predGenerator}CT18ZVarsCorrW.pkl.lz4",
        "W",
        f"{args.predGenerator}CT18ZVars_hist",
    )
    h_pred_W_full = h_pred_W_full[{"vars": "pdf0"}].project("qT", "absY", "charge")
    h_pred_W_full = hh.disableFlow(h_pred_W_full, "qT", under=False, over=True)
    h_pred_W_full = hh.disableFlow(h_pred_W_full, "absY", under=False, over=True)

    writer.set_reference(
        "chW",
        h_pred_W_full,
        postOp=convert_WFull_to_LepFiducial(h_W_lep_fiducial, h_W_lep_inclusive),
    )
    writer.add_process(h_pred_W_full, "Wmunu", "chW", signal=False)

# add systematic variations

bosons = []
if not args.noFitSigmaUL:
    bosons.append("Z")

# alphaS variation
logger.info("Now at variations for AlphaS")
if args.predGenerator == "scetlib_dyturbo":
    alphas_vars = theory_corrections.load_corr_helpers(
        bosons,
        ["scetlib_dyturboCT18Z_pdfas"],
        make_tensor=False,
        minnlo_ratio=False,
    )
    symmetrize = "quadratic" if args.constrainAlphaS else "average"

    if not args.noFitSigmaUL:
        writer.add_systematic(
            [
                alphas_vars["Z"]["scetlib_dyturboCT18Z_pdfas"][{"vars": 2}],
                alphas_vars["Z"]["scetlib_dyturboCT18Z_pdfas"][{"vars": 1}],
            ],
            "pdfAlphaS",
            "Zmumu",
            "chSigmaUL",
            noi=not args.constrainAlphaS,
            constrained=args.constrainAlphaS,
            symmetrize=symmetrize,
            kfactor=1.5 / 2.0,
            groups=(
                ["pdfCT18Z", "pdfCT18ZAlphaS", "theory"]
                if args.constrainAlphaS
                else ["pdfCT18Z"]
            ),
        )

    # alphaS variations for W come from same as Z
    if args.fitW:
        # TODO clean up
        with h5py.File(
            f"/ceph/submit/data/group/cms/store/user/lavezzo/alphaS/250815_debug/mw_with_mu_eta_pt_scetlib_dyturboCorr_maxFiles_m1_oldQCDscales.hdf5",
            "r",
        ) as h5file:
            results = input_tools.load_results_h5py(h5file)
            lumi = 16800
            h_Wp = results["WplusmunuPostVFP"]["output"][
                "prefsr_pdfAlphaSByHelicity"
            ].get()
            weight_sum = results["WplusmunuPostVFP"]["weight_sum"]
            xsec = results["WplusmunuPostVFP"]["dataset"]["xsec"]
            h_Wp *= xsec * lumi / weight_sum
            h_Wm = results["WminusmunuPostVFP"]["output"][
                "prefsr_pdfAlphaSByHelicity"
            ].get()
            weight_sum = results["WminusmunuPostVFP"]["weight_sum"]
            xsec = results["WminusmunuPostVFP"]["dataset"]["xsec"]
            h_Wm *= xsec * lumi / weight_sum
            h_W = hh.addHists(h_Wp, h_Wm)
            print(h_W)
            h_W = h_W.project("absEtaGen", "ptGen", "qGen", "alphasVar")

        writer.add_systematic(
            [
                h_W[{"vars": 2}],
                h_W[{"vars": 1}],
            ],
            "pdfAlphaS",
            "Wmunu",
            "chW",
            noi=not args.constrainAlphaS,
            constrained=args.constrainAlphaS,
            symmetrize=symmetrize,
            kfactor=1.5 / 2.0,
            groups=(
                ["pdfCT18Z", "pdfCT18ZAlphaS", "theory"]
                if args.constrainAlphaS
                else ["pdfCT18Z"]
            ),
            format=False,
        )

    # Ai's alphaS predictions come only from MiNNLO
    if args.fitAngularCoeffs:
        with h5py.File(
            args.predAiFile.replace("w_z_helicity_xsecs", "w_z_gen_dists"), "r"
        ) as ff:
            inputs = input_tools.load_results_h5py(ff)
            alpha_vars_hels = inputs["ZmumuPostVFP"]["output"][
                "nominal_gen_helicity_nominal_gen_pdfCT18ZalphaS002"
            ].get()
        writer.add_systematic(
            [
                alpha_vars_hels[{"alphasVar": "as0120"}],
                alpha_vars_hels[{"alphasVar": "as0116"}],
            ],
            "pdfAlphaS",
            "Zmumu",
            "chAis",
            noi=not args.constrainAlphaS,
            constrained=args.constrainAlphaS,
            symmetrize=symmetrize,
            kfactor=1.5 / 2.0,
            groups=(
                ["pdfCT18Z", "pdfCT18ZAlphaS", "theory"]
                if args.constrainAlphaS
                else ["pdfCT18Z"]
            ),
        )

else:
    raise Exception("No valid configuration found for alphaS variation.")

# mW variations
if args.fitW:

    logger.info("Now at mW variations")

    # mW variations come from MiNNLO
    with h5py.File(
        "/ceph/submit/data/group/cms/store/user/lavezzo/alphaS/250718_mW_variations/prefsr_massWeightW_MiNNLO.hdf5",
        "r",
    ) as ff:
        inputs = input_tools.load_results_h5py(ff)
        mass_vars_Wp = inputs["WplusmunuPostVFP"]["prefsr_massWeightW"].get()
        mass_vars_Wm = inputs["WminusmunuPostVFP"]["prefsr_massWeightW"].get()

    mass_vars_W = hh.addHists(mass_vars_Wp, mass_vars_Wm)
    mass_vars_W = mass_vars_W.project(
        "absEtaGen", "ptGen", "qGen", "massShift"
    )  # re-order
    mass_vars_W = hh.rebinHist(
        mass_vars_W, "ptGen", h_data_prefsrLep.axes["ptGen"].edges
    )
    mass_vars_W = hh.rebinHist(
        mass_vars_W, "absEtaGen", h_data_prefsrLep.axes["absEtaGen"].edges
    )
    mass_vars_W_nom = mass_vars_W[{"massShift": "massShiftW0MeV"}]
    mass_vars_W_up = mass_vars_W[{"massShift": "massShiftW100MeVUp"}]
    mass_vars_W_down = mass_vars_W[{"massShift": "massShiftW100MeVDown"}]

    # calculate variations on the leptons
    # (MiNNLO_var / MiNNLO_nom) * nominal
    # where nominal is MiNNLO(W, lep; fiducial) * scetlib(W; full) / MiNNLO(W; full)
    ratio_MiNNLO_up = hh.divideHists(mass_vars_W_up, mass_vars_W_nom)
    ratio_MiNNLO_down = hh.divideHists(mass_vars_W_down, mass_vars_W_nom)
    writer.ref["chW"]["Wmunu"] = hh.disableFlow(
        writer.ref["chW"]["Wmunu"], "absEtaGen", under=False, over=True
    )
    mass_var_up = hh.multiplyHists(ratio_MiNNLO_up, writer.ref["chW"]["Wmunu"])
    mass_var_down = hh.multiplyHists(ratio_MiNNLO_down, writer.ref["chW"]["Wmunu"])

    writer.add_systematic(
        [mass_var_up, mass_var_down],
        "massShiftW100MeV",
        "Wmunu",
        "chW",
        mirror=False,
        symmetrize="average",
        noi=not args.constrainMW,
        constrained=args.constrainMW,
        format=False,
        groups=["ZmassAndWidth", "theory"] if args.constrainMW else ["ZmassAndWidth"],
    )

logger.info(f"Now at variations from {args.predGenerator}")
generator_vars = theory_corrections.load_corr_helpers(
    bosons,
    [
        args.predGenerator,
        f"{args.predGenerator}MSHT20mcrange",
        f"{args.predGenerator}MSHT20mbrange",
    ],
    make_tensor=False,
    minnlo_ratio=False,
)
if args.fitW:

    # TODO clean up and change to use by helicity quark mass weights
    with h5py.File(
        "/ceph/submit/data/group/cms/store/user/lavezzo/alphaS/250815_debug/mw_with_mu_eta_pt_scetlib_dyturboCorr_maxFiles_m1_oldQCDscales.hdf5",
        "r",
    ) as h5file:
        results = input_tools.load_results_h5py(h5file)
        lumi = 16800

        h_Wp = results["WplusmunuPostVFP"]["output"]["prefsr_scetlib_dyturboCorr"].get()
        weight_sum = results["WplusmunuPostVFP"]["weight_sum"]
        xsec = results["WplusmunuPostVFP"]["dataset"]["xsec"]
        h_Wp *= xsec * lumi / weight_sum
        h_Wm = results["WminusmunuPostVFP"]["output"][
            "prefsr_scetlib_dyturboCorr"
        ].get()
        weight_sum = results["WminusmunuPostVFP"]["weight_sum"]
        xsec = results["WminusmunuPostVFP"]["dataset"]["xsec"]
        h_Wm *= xsec * lumi / weight_sum
        h_W = hh.addHists(h_Wp, h_Wm)
        generator_vars["W"] = {}
        generator_vars["W"]["scetlib_dyturbo"] = h_W.project(
            "absEtaGen", "ptGen", "qGen", "vars"
        )

        print(results["WplusmunuPostVFP"]["output"]["prefsr_pdfMSHT20mcrange"].get())
        h_Wp = results["WplusmunuPostVFP"]["output"]["prefsr_pdfMSHT20mcrange"].get()
        weight_sum = results["WplusmunuPostVFP"]["weight_sum"]
        xsec = results["WplusmunuPostVFP"]["dataset"]["xsec"]
        h_Wp *= xsec * lumi / weight_sum
        h_Wm = results["WminusmunuPostVFP"]["output"]["prefsr_pdfMSHT20mcrange"].get()
        weight_sum = results["WminusmunuPostVFP"]["weight_sum"]
        xsec = results["WminusmunuPostVFP"]["dataset"]["xsec"]
        h_Wm *= xsec * lumi / weight_sum
        h_W = hh.addHists(h_Wp, h_Wm)
        generator_vars["W"][f"{args.predGenerator}MSHT20mcrange"] = h_W.project(
            "absEtaGen", "ptGen", "qGen", "pdfVar"
        )

        h_Wp = results["WplusmunuPostVFP"]["output"]["prefsr_pdfMSHT20mbrange"].get()
        weight_sum = results["WplusmunuPostVFP"]["weight_sum"]
        xsec = results["WplusmunuPostVFP"]["dataset"]["xsec"]
        h_Wp *= xsec * lumi / weight_sum
        h_Wm = results["WminusmunuPostVFP"]["output"]["prefsr_pdfMSHT20mbrange"].get()
        weight_sum = results["WminusmunuPostVFP"]["weight_sum"]
        xsec = results["WminusmunuPostVFP"]["dataset"]["xsec"]
        h_Wm *= xsec * lumi / weight_sum
        h_W = hh.addHists(h_Wp, h_Wm)
        generator_vars["W"][f"{args.predGenerator}MSHT20mbrange"] = h_W.project(
            "absEtaGen", "ptGen", "qGen", "pdfVar"
        )

for proc in generator_vars.keys():  # loop over processes

    if proc == "Z":
        proc_name = "Zmumu"
        ch_name = "chSigmaUL"
        format = True
    elif proc == "W":
        # TODO maybe drop the 'format' for W since it only applies to the pred
        proc_name = "Wmunu"
        ch_name = "chW"
        format = False

    h = generator_vars[proc][args.predGenerator]

    # correlated NP uncertainties
    corr_NP_uncs = [
        ["Lambda20.25", "Lambda2-0.25", "chargeVgenNP0scetlibNPZLambda2"],
        ["Lambda4.16", "Lambda4.01", "chargeVgenNP0scetlibNPZLambda4"],
        [
            "Delta_Lambda20.02",
            "Delta_Lambda2-0.02",
            "chargeVgenNP0scetlibNPZDelta_Lambda2",
        ],
    ]
    for var in corr_NP_uncs:
        writer.add_systematic(
            [h[{"vars": var[0]}], h[{"vars": var[1]}]],
            var[2],
            proc_name,
            ch_name,
            symmetrize="average",
            groups=["resumNonpert", "resum", "pTModeling", "theory"],
            format=format,
        )

    # gamma NP uncertainties
    gamma_NP_uncs = [
        ["omega_nu0.5", "c_nu-0.1-omega_nu0.5", "scetlibNPgamma"],
    ]
    for var in gamma_NP_uncs:
        writer.add_systematic(
            [h[{"vars": var[0]}], h[{"vars": var[1]}]],
            var[2],
            proc_name,
            ch_name,
            symmetrize="average",
            groups=["resumTNP", "resum", "pTModeling", "theory"],
            format=format,
        )

    # TNP
    TNP_uncs = [
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
    for var in TNP_uncs:
        var_name = "resumTNP_" + var[1].split("-")[0]
        writer.add_systematic(
            [h[{"vars": var[0]}], h[{"vars": var[1]}]],
            var_name,
            proc_name,
            ch_name,
            symmetrize="average",
            groups=["resumTNP", "resum", "pTModeling", "theory"],
            format=format,
        )

    # transition FO scale uncertainties
    transition_FO_uncs = [
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
    for var in transition_FO_uncs:
        writer.add_systematic(
            [h[{"vars": var[0]}], h[{"vars": var[1]}]],
            var[2],
            proc_name,
            ch_name,
            symmetrize="quadratic",
            groups=["resumTransitionFOScale", "resum", "pTModeling", "theory"],
            format=format,
        )

    # mass quark effects
    h = generator_vars[proc][f"{args.predGenerator}MSHT20mbrange"]
    var_axis_name = "vars" if proc == "Z" else "pdfVar"
    writer.add_scale_systematic(
        [h[{var_axis_name: -1}], h[{var_axis_name: 1}], h[{var_axis_name: 0}]],
        "pdfMSHT20mbrange",
        proc_name,
        ch_name,
        symmetrize="quadratic",
        groups=["bcQuarkMass", "pTModeling", "theory"],
        format=format,
    )
    h = generator_vars[proc][f"{args.predGenerator}MSHT20mcrange"]
    writer.add_scale_systematic(
        [h[{var_axis_name: -1}], h[{var_axis_name: 1}], h[{var_axis_name: 0}]],
        "pdfMSHT20mcrange",
        proc_name,
        ch_name,
        symmetrize="quadratic",
        groups=["bcQuarkMass", "pTModeling", "theory"],
        format=format,
    )

# PDF uncertainties
logger.info("Now at PDF variations")
if args.fitAngularCoeffs:
    # for Ai's, we have MINNLO, so use it for sigmaUL + Ai's to be consistent

    # TODO fix this at some point
    with h5py.File(
        "/ceph/submit/data/group/cms/store/user/lavezzo/alphaS//250627_angularCoefficients/w_z_gen_dists.cash_scetlib_dyturboCorr_maxFiles_1000_alphaSunfoldingBinning_helicity_WZ.hdf5",
        "r",
    ) as ff:
        inputs = input_tools.load_results_h5py(ff)
        pdf_vars = inputs["ZmumuPostVFP"]["output"][
            "nominal_gen_helicity_pdfCT18Z"
        ].get()
        pdf_vars_Wp = inputs["WplusmunuPostVFP"]["output"][
            "nominal_gen_helicity_pdfCT18Z"
        ].get()
        pdf_vars_Wm = inputs["WminusmunuPostVFP"]["output"][
            "nominal_gen_helicity_pdfCT18Z"
        ].get()
        pdf_vars_W = hh.addHists(pdf_vars_Wp, pdf_vars_Wm)

    for ivar in range(1, len(pdf_vars.axes[-1]), 2):

        # sigmaUL
        if not args.noFitSigmaUL:
            writer.add_scale_systematic(
                [
                    pdf_vars[{"pdfVar": ivar + 1}],
                    pdf_vars[{"pdfVar": ivar}],
                    pdf_vars[{"pdfVar": "pdf0CT18Z"}],
                ],
                f"pdf{int((ivar+1)/2)}CT18Z",
                "Zmumu",
                "chSigmaUL",
                symmetrize="quadratic",
                kfactor=1 / 1.645,
                groups=["pdfCT18Z", f"pdfCT18ZNoAlphaS", "theory"],
            )

        # Ai's
        writer.add_scale_systematic(
            [
                pdf_vars[{"pdfVar": ivar + 1}],
                pdf_vars[{"pdfVar": ivar}],
                pdf_vars[{"pdfVar": "pdf0CT18Z"}],
            ],
            f"pdf{int((ivar+1)/2)}CT18Z",
            "Zmumu",
            "chAis",
            symmetrize="quadratic",
            kfactor=1 / 1.645,
            groups=["pdfCT18Z", f"pdfCT18ZNoAlphaS", "theory"],
        )

        if args.fitW:
            # TODO I'm 100% sure this is wrong, do it from MiNNLO
            raise Exception()
            writer.add_systematic(
                [
                    pdf_vars_W[{"helicity": -1j}][{"pdfVar": ivar + 1}],
                    pdf_vars_W[{"helicity": -1j}][{"pdfVar": ivar}],
                    pdf_vars_W[{"helicity": -1j}][{"pdfVar": "pdf0CT18Z"}],
                ],
                f"pdf{int((ivar+1)/2)}CT18Z",
                "Wmunu",
                "chW",
                symmetrize="quadratic",
                kfactor=1 / 1.645,
                groups=["pdfCT18Z", f"pdfCT18ZNoAlphaS", "theory"],
            )

else:

    # for sigmaUL only, scetlib+dyturbo has the latest & greatest PDFs
    corr_helpers = theory_corrections.load_corr_helpers(
        bosons,
        ["scetlib_dyturboCT18ZVars"],
        make_tensor=False,
        minnlo_ratio=False,
    )

    if not args.noFitSigmaUL:
        h = corr_helpers["Z"]["scetlib_dyturboCT18ZVars"]
        for ivar in range(1, len(h.axes[-1]), 2):
            writer.add_systematic(
                [h[{"vars": ivar + 1}], h[{"vars": ivar}]],
                f"pdf{int((ivar+1)/2)}CT18Z",
                "Zmumu",
                "chSigmaUL",
                symmetrize="quadratic",
                kfactor=1 / 1.645,
                groups=["pdfCT18Z", f"pdfCT18ZNoAlphaS", "theory"],
            )

    if args.fitW:

        # TODO clean this up
        with h5py.File(
            "/ceph/submit/data/group/cms/store/user/lavezzo/alphaS/250815_debug/mw_with_mu_eta_pt_scetlib_dyturboCorr_maxFiles_m1_oldQCDscales.hdf5",
            "r",
        ) as h5file:
            results = input_tools.load_results_h5py(h5file)
            lumi = 16800
            h_Wp = results["WplusmunuPostVFP"]["output"][
                "prefsr_pdfCT18ZUncertByHelicity"
            ].get()
            weight_sum = results["WplusmunuPostVFP"]["weight_sum"]
            xsec = results["WplusmunuPostVFP"]["dataset"]["xsec"]
            h_Wp *= xsec * lumi / weight_sum
            h_Wm = results["WminusmunuPostVFP"]["output"][
                "prefsr_pdfCT18ZUncertByHelicity"
            ].get()
            weight_sum = results["WminusmunuPostVFP"]["weight_sum"]
            xsec = results["WminusmunuPostVFP"]["dataset"]["xsec"]
            h_Wm *= xsec * lumi / weight_sum
            h_W = hh.addHists(h_Wp, h_Wm)
            h_W = h_W.project("absEtaGen", "ptGen", "qGen", "pdfVar")

        for ivar in range(1, int((len(h.axes[-1]) - 1) / 2)):
            writer.add_systematic(
                [
                    h_W[{"pdfVar": f"pdf{int(ivar)}CT18ZUp"}],
                    h_W[{"pdfVar": f"pdf{int(ivar)}CT18ZDown"}],
                ],
                f"pdf{int((ivar))}CT18Z",
                "Wmunu",
                "chW",
                symmetrize="quadratic",
                kfactor=1 / 1.645,
                groups=["pdfCT18Z", f"pdfCT18ZNoAlphaS", "theory"],
                format=False,
            )


# Ai's only uncertainties
if args.fitAngularCoeffs:

    qcd_helper = theory_corrections.make_qcd_uncertainty_helper_by_helicity(
        is_z=True,
        filename=args.predAiFile,
        rebin_ptVgen=h_data_ai.axes["ptVGen"].edges.tolist(),
        rebin_absYVgen=h_data_ai.axes["absYVGen"].edges.tolist(),
        rebin_massVgen=True,
        return_tensor=False,
    )

    # pythia showering uncertainties
    logger.info("Now at pythia_shower_kt")
    pythia_shower_kt = qcd_helper[{"vars": "pythia_shower_kt"}]
    writer.add_scale_systematic(
        [pythia_shower_kt[{"corr": 1}], pythia_shower_kt[{"corr": 0}]],
        "pythia_shower_kt",
        "Zmumu",
        "chAis",
        mirror=True,
        groups=["helicity_shower_kt", "angularCoeffs", "theory"],
    )

    # QCD scales
    logger.info("Now at QCD scales")

    # prepare fine binning hists
    fine_pt_binning = qcd_helper.axes["ptVgen"].edges
    nptfine = len(fine_pt_binning) - 1
    scale_inclusive = np.sqrt((nptfine - 1) / nptfine)

    for hel in range(0, 7 + 1):  # no correction on sigma_UL

        # fine binning
        qcd_scales_hel_up = qcd_helper[{"vars": f"helicity_{hel}_Up"}].project(
            "ptVgen", "absYVgen", "helicity"
        )
        qcd_scales_hel_down = qcd_helper[{"vars": f"helicity_{hel}_Down"}].project(
            "ptVgen", "absYVgen", "helicity"
        )
        qcd_scales_hel_nominal = qcd_helper[{"vars": f"nominal"}].project(
            "ptVgen", "absYVgen", "helicity"
        )

        for bin in range(len(fine_pt_binning) - 1):

            ptl = fine_pt_binning[bin]
            pth = fine_pt_binning[bin + 1]

            qcd_scales_hel_pt_up = copy.deepcopy(qcd_scales_hel_up)
            qcd_scales_hel_pt_up.values()[...] = qcd_scales_hel_nominal.values()
            qcd_scales_hel_pt_up.values()[bin, ...] = qcd_scales_hel_up[
                {"ptVgen": bin}
            ].values()

            qcd_scales_hel_pt_down = copy.deepcopy(qcd_scales_hel_down)
            qcd_scales_hel_pt_down.values()[...] = qcd_scales_hel_nominal.values()
            qcd_scales_hel_pt_down.values()[bin, ...] = qcd_scales_hel_up[
                {"ptVgen": bin}
            ].values()

            writer.add_systematic(
                [qcd_scales_hel_pt_up, qcd_scales_hel_pt_down],
                f"QCDscaleZfine_Pt{ptl}_{pth}helicity_{hel}",
                "Zmumu",
                "chAis",
                symmetrize="quadratic",
                groups=["QCDScaleZMiNNLO", "QCDscale", "angularCoeffs", "theory"],
            )

        # inclusive
        inclusive_pt_binning = [
            qcd_scales_hel_up.axes["ptVgen"].edges[0],
            qcd_scales_hel_up.axes["ptVgen"].edges[-1],
        ]
        qcd_scales_hel_int_up = hh.rebinHist(
            qcd_scales_hel_up, "ptVgen", inclusive_pt_binning
        )
        qcd_scales_hel_int_down = hh.rebinHist(
            qcd_scales_hel_down, "ptVgen", inclusive_pt_binning
        )

        for bin in range(len(fine_pt_binning) - 1):
            qcd_scales_hel_up.values()[bin, ...] = qcd_scales_hel_int_up.values()
            qcd_scales_hel_down.values()[bin, ...] = qcd_scales_hel_int_down.values()

        writer.add_systematic(
            [qcd_scales_hel_up, qcd_scales_hel_down],
            f"QCDscaleZinclusive_Pt{inclusive_pt_binning[0]}_{inclusive_pt_binning[1]}helicity_{hel}",
            "Zmumu",
            "chAis",
            kfactor=scale_inclusive,
            symmetrize="quadratic",
            groups=["QCDScaleZMiNNLO", "QCDscale", "angularCoeffs", "theory"],
        )


# QCD uncertainties by helicity
if args.fitW:

    # pythia showering uncertainties
    logger.info("Now at pythia_shower_kt (W)")

    # TODO clean this up
    with h5py.File(
        "/ceph/submit/data/group/cms/store/user/lavezzo/alphaS/250815_debug/mw_with_mu_eta_pt_scetlib_dyturboCorr_maxFiles_m1_oldQCDscales.hdf5",
        "r",
    ) as h5file:
        results = input_tools.load_results_h5py(h5file)
        lumi = 16800
        h_Wp = results["WplusmunuPostVFP"]["output"]["prefsr_qcdScaleByHelicity"].get()
        weight_sum = results["WplusmunuPostVFP"]["weight_sum"]
        xsec = results["WplusmunuPostVFP"]["dataset"]["xsec"]
        h_Wp *= xsec * lumi / weight_sum
        h_Wm = results["WminusmunuPostVFP"]["output"]["prefsr_qcdScaleByHelicity"].get()
        weight_sum = results["WminusmunuPostVFP"]["weight_sum"]
        xsec = results["WminusmunuPostVFP"]["dataset"]["xsec"]
        h_Wm *= xsec * lumi / weight_sum
        h_W = hh.addHists(h_Wp, h_Wm)

    h_W_pythia = h_W[{"vars": "pythia_shower_kt"}].project("absEtaGen", "ptGen", "qGen")
    writer.add_systematic(
        h_W_pythia,
        "helicity_shower_kt",
        "Wmunu",
        "chW",
        mirror=True,
        groups=["helicity_shower_kt", "angularCoeffs", "theory"],
        format=False,
    )

    # QCD scales
    logger.info("Now at QCD scales (W)")

    # prepare fine binning hists
    from utilities import common

    fine_pt_binning = common.ptV_binning[::2]
    inclusive_pt_binning = [
        common.ptV_binning[0],
        common.ptV_binning[-1],
    ]
    nptfine = len(fine_pt_binning) - 1
    scale_inclusive = np.sqrt((nptfine - 1) / nptfine)

    from wremnants import syst_tools

    h_W_new = syst_tools.hist_to_variations(
        copy.deepcopy(h_W),
        gen_axes=["ptVgen"],
        rebin_axes=["ptVgen"],
        rebin_edges=[fine_pt_binning],
    )
    h_W_full = syst_tools.hist_to_variations(
        copy.deepcopy(h_W),
        gen_axes=["ptVgen"],
        rebin_axes=["ptVgen"],
        rebin_edges=[inclusive_pt_binning],
    )

    for hel in range(0, 7 + 1):  # no correction on sigma_UL

        # fine binning
        qcd_scales_hel_up = h_W[{"vars": f"helicity_{hel}_Up"}].project(
            "ptVgen", "absEtaGen", "ptGen", "qGen"
        )
        qcd_scales_hel_down = h_W[{"vars": f"helicity_{hel}_Down"}].project(
            "ptVgen", "absEtaGen", "ptGen", "qGen"
        )

        var_up = h_W_new[{"vars": f"helicity_{hel}_Up"}].project(
            "ptVgen", "absEtaGen", "ptGen", "qGen"
        )
        var_down = h_W_new[{"vars": f"helicity_{hel}_Down"}].project(
            "ptVgen", "absEtaGen", "ptGen", "qGen"
        )

        for bin in range(len(fine_pt_binning) - 1):

            ptl = fine_pt_binning[bin]
            pth = fine_pt_binning[bin + 1]
            writer.add_systematic(
                [
                    var_up[{"ptVgen": bin}],
                    var_down[{"ptVgen": bin}],
                ],
                f"QCDscaleWfine_PtV{ptl}_{pth}helicity_{hel}_",
                "Wmunu",
                "chW",
                symmetrize="quadratic",
                groups=["QCDScaleWMiNNLO", "QCDscale", "angularCoeffs", "theory"],
                format=False,
            )

        var_up = h_W_full[{"vars": f"helicity_{hel}_Up", "ptVgen": 0}].project(
            "absEtaGen", "ptGen", "qGen"
        )
        var_down = h_W_full[{"vars": f"helicity_{hel}_Down", "ptVgen": 0}].project(
            "absEtaGen", "ptGen", "qGen"
        )
        writer.add_systematic(
            [var_up, var_down],
            f"QCDscaleWinclusive_PtV{inclusive_pt_binning[0]}_{inclusive_pt_binning[1]}helicity_{hel}_",
            "Wmunu",
            "chW",
            symmetrize="quadratic",
            kfactor=scale_inclusive,
            groups=["QCDScaleWMiNNLO", "QCDscale", "angularCoeffs", "theory"],
            format=False,
        )

# write output
directory = args.outfolder
if directory == "":
    directory = "./"
filename = args.outname
if not args.constrainAlphaS:
    filename += "_alphaS"
if not args.constrainMW:
    filename += "_mW"
if args.postfix:
    filename += f"_{args.postfix}"
writer.write(outfolder=directory, outfilename=filename)
logger.info(f"Written to {os.path.join(directory, filename)}.hdf5")
