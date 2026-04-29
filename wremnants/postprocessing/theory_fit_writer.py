import copy
import pprint
import re

from rabbit.tensorwriter import TensorWriter
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
from wremnants.utilities import common, theory_utils
from wums import boostHistHelpers as hh
from wums import logging


def _pdfas_generator_name(pred_generator):
    if pred_generator.endswith("_pdfas"):
        return pred_generator
    return f"{pred_generator}_pdfas"


def _pdfvars_generator_name(pred_generator):
    if pred_generator.endswith("_pdfvars"):
        return pred_generator
    return f"{pred_generator}_pdfvars"


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


class SigmaULTheoryFitWriter(TensorWriter):
    """Tensor writer for the direct-theory sigmaUL fit."""

    def __init__(
        self,
        predGenerator,
        nois,
        pdf,
        exclude_nuisances="",
        keep_nuisances="",
        process_name="Zmumu",
        sigmaul_channel="chSigmaUL",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.logger = logging.child_logger(__name__)
        self.ref = {}
        self.process_name = process_name
        self.sigmaul_channel = sigmaul_channel
        self.predGenerator = predGenerator
        self.nois = nois
        self.pdf = pdf
        self.pdf_name = theory_utils.pdfMap[pdf]["name"]
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
        ptV_name = self.get_ptV_axis_name(h)
        absYV_name = self.get_absYV_axis_name(h)
        self.ref[channel] = {
            "h": h,
            "lumi": lumi,
            "scale": scale,
            "postOp": postOp,
            "ptV_name": ptV_name,
            "absYV_name": absYV_name,
            "chargeV_name": self.get_charge_axis_name(h),
            "ptV_bins": h.axes[ptV_name].edges,
            "absYV_bins": h.axes[absYV_name].edges,
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

    def add_shape_systematic(
        self,
        h,
        name,
        process,
        channel,
        rebin_pt=True,
        rebin_y=True,
        normalize=False,
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
        raise ValueError(f"Did not find pT axis. Available axes: {list(h.axes.name)}")

    def get_absYV_axis_name(self, h):
        for name in ["absYVgen", "absYVGen", "absY"]:
            if name in h.axes.name:
                return name
        raise ValueError(f"Did not find absY axis. Available axes: {list(h.axes.name)}")

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
        if process != self.process_name:
            raise ValueError(
                f"Unsupported process '{process}' for sigmaUL writer; expected '{self.process_name}'"
            )

        mass_axis_name = self.get_mass_axis_name(h)
        if mass_axis_name:
            h = h[{mass_axis_name: 90.0j}]

        charge_axis_name = self.get_charge_axis_name(h)
        if charge_axis_name:
            h = h[{charge_axis_name: 0.0j}]

        if channel == self.sigmaul_channel and "helicity" in h.axes.name:
            h = h[{"helicity": -1.0j}]

        return h

    def load_sigmaul_data(
        self, pseudodataGenerator, infile, fitresultMapping, channelSigmaUL
    ):
        if pseudodataGenerator:
            corrfile = f"{common.data_dir}/TheoryCorrections/{pseudodataGenerator}_CorrZ.pkl.lz4"
            self.logger.info("Loading sigmaUL pseudodata from %s", corrfile)
            h_data = theory_corrections.load_corr_hist(
                corrfile,
                "Z",
                f"{pseudodataGenerator}_hist",
            )
            h_data = _select_baseline_variation(h_data)
            h_data = h_data.project("qT", "absY")
            self.add_channel(h_data.axes, self.sigmaul_channel)
            self.add_data(h_data, self.sigmaul_channel)
            self.set_reference(self.sigmaul_channel, h_data)
            return None

        import rabbit.io_tools

        fitresult, meta = rabbit.io_tools.get_fitresult(
            infile, result="asimov", meta=True
        )
        self.logger.debug(
            "Available models in fit result: %s", list(fitresult["mappings"].keys())
        )

        h_data_cov = fitresult["mappings"][fitresultMapping][
            "hist_postfit_inclusive_cov"
        ].get()
        self.add_data_covariance(h_data_cov)

        channel_name = _resolve_sigmaul_channel(
            fitresult, fitresultMapping, channelSigmaUL
        )
        self.logger.debug(
            "Using sigmaUL channel '%s' in mapping '%s'",
            channel_name,
            fitresultMapping,
        )
        h_data = fitresult["mappings"][fitresultMapping]["channels"][channel_name][
            "hist_postfit_inclusive"
        ].get()
        self.add_channel(h_data.axes, self.sigmaul_channel)
        self.add_data(h_data, self.sigmaul_channel)
        self.set_reference(self.sigmaul_channel, h_data)
        return meta

    def add_sigmaul_process(self):
        h_sig_sigmaul = theory_corrections.load_corr_hist(
            f"{common.data_dir}/TheoryCorrections/{self.predGenerator}_CorrZ.pkl.lz4",
            "Z",
            f"{self.predGenerator}_hist",
        )
        h_sig_sigmaul = _select_baseline_variation(h_sig_sigmaul)
        self.add_process(
            h_sig_sigmaul, self.process_name, self.sigmaul_channel, signal=False
        )

    def add_alphas_variation(self):
        self.logger.info("Adding alphaS variation")
        symmetrize = "average" if "alphaS" in self.nois else "quadratic"
        alphas_var_name = _pdfas_generator_name(self.predGenerator)
        alphas_vars = theory_corrections.load_corr_helpers(
            ["Z"],
            [alphas_var_name],
            make_tensor=False,
            minnlo_ratio=False,
        )
        self.add_systematic(
            [
                alphas_vars["Z"][alphas_var_name][{"vars": 2}],
                alphas_vars["Z"][alphas_var_name][{"vars": 1}],
            ],
            "pdfAlphaS",
            self.process_name,
            self.sigmaul_channel,
            noi=("alphaS" in self.nois),
            constrained=not ("alphaS" in self.nois),
            symmetrize=symmetrize,
            kfactor=1.0,
            groups=(
                [self.pdf_name, f"{self.pdf_name}AlphaS", "theory", "theory_qcd"]
                if "alphaS" not in self.nois
                else [self.pdf_name]
            ),
        )

    def add_pdf_bc_quark_mass_variations(self):
        self.logger.info("Adding PDF b/c quark-mass variations")

        if self.pdf_name == theory_utils.pdfMap["herapdf20"][
            "name"
        ] and self._keep_systematic("HERAPDF20EXT"):
            self.logger.info(
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
            self.add_shape_systematic(
                [
                    h[{"vars": up_var}],
                    h[{"vars": down_var}],
                    _select_baseline_variation(h),
                ],
                nuisance_name,
                self.process_name,
                self.sigmaul_channel,
                symmetrize="quadratic",
                groups=["bcQuarkMass", "pTModeling", "theory", "theory_qcd"],
            )

    def add_resummation_and_np_variations(self):
        self.logger.info(
            "Adding direct-theory sigmaUL systematics from %s", self.predGenerator
        )
        generator_vars = theory_corrections.load_corr_helpers(
            ["Z"],
            [
                self.predGenerator,
            ],
            make_tensor=False,
            minnlo_ratio=False,
        )
        nominal = generator_vars["Z"][self.predGenerator]

        if "lattice" in self.predGenerator.lower():
            corr_np_uncs = LATTICE_CORRELATED_NP_UNCERTAINTIES
            gamma_np_uncs = LATTICE_GAMMA_NP_UNCERTAINTIES
        else:
            corr_np_uncs = STANDARD_CORRELATED_NP_UNCERTAINTIES
            gamma_np_uncs = STANDARD_GAMMA_NP_UNCERTAINTIES

        for up_var, down_var, nuisance_name in corr_np_uncs:
            self.add_systematic(
                [nominal[{"vars": up_var}], nominal[{"vars": down_var}]],
                nuisance_name,
                self.process_name,
                self.sigmaul_channel,
                symmetrize="average",
                groups=["resumNonpert", "resum", "pTModeling", "theory", "theory_qcd"],
            )

        for up_var, down_var, nuisance_name in gamma_np_uncs:
            self.add_systematic(
                [nominal[{"vars": up_var}], nominal[{"vars": down_var}]],
                nuisance_name,
                self.process_name,
                self.sigmaul_channel,
                symmetrize="average",
                groups=["resumTNP", "resum", "pTModeling", "theory", "theory_qcd"],
            )

        for up_var, down_var in TNP_UNCERTAINTIES:
            self.add_systematic(
                [nominal[{"vars": up_var}], nominal[{"vars": down_var}]],
                f"resumTNP_{down_var.split('-')[0]}",
                self.process_name,
                self.sigmaul_channel,
                symmetrize="average",
                groups=["resumTNP", "resum", "pTModeling", "theory", "theory_qcd"],
            )

        for up_var, down_var, nuisance_name in TRANSITION_FO_UNCERTAINTIES:
            self.add_systematic(
                [nominal[{"vars": up_var}], nominal[{"vars": down_var}]],
                nuisance_name,
                self.process_name,
                self.sigmaul_channel,
                symmetrize="quadratic",
                groups=[
                    "resumTransitionFOScale",
                    "resum",
                    "pTModeling",
                    "theory",
                    "theory_qcd",
                ],
            )

    def add_pdf_variations(self, scale_pdf: float | None = None):
        self.logger.info("Adding PDF variations")
        pdf_var_key = _pdfvars_generator_name(self.predGenerator)
        keys_to_load = [pdf_var_key]

        pdfInfo = theory_utils.pdf_info_map("Zmumu_2016PostVFP", self.pdf)
        pdfName = pdfInfo["name"]

        if scale_pdf is not None:
            scale = scale_pdf
        else:
            scale = pdfInfo.get("inflation_factor_alphaS", 1)
            scale = pdfInfo.get("scale", 1) * scale
        self.logger.debug(f"Using scale {scale}.")

        pdf_var_key_ext = None
        if pdfName == "pdfHERAPDF20" and self._keep_systematic("HERAPDF20EXT"):
            pdf_var_key_ext = pdf_var_key.replace("HERAPDF20", "HERAPDF20EXT")
            keys_to_load.append(pdf_var_key_ext)

        corr_helpers = theory_corrections.load_corr_helpers(
            ["Z"],
            keys_to_load,
            make_tensor=False,
            minnlo_ratio=False,
        )

        pdf_groups = [self.pdf_name, f"{self.pdf_name}NoAlphaS", "theory", "theory_qcd"]

        h = corr_helpers["Z"][pdf_var_key]
        for ivar in range(1, len(h.axes[-1]), 2):
            self.add_systematic(
                [h[{"vars": ivar + 1}], h[{"vars": ivar}]],
                f"pdf{int((ivar + 1) / 2)}{self.pdf.upper()}",
                self.process_name,
                self.sigmaul_channel,
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
                self.add_systematic(
                    [h_ext[{"vars": ivar + 1}], h_ext[{"vars": ivar}]],
                    f"pdf{int((ivar + 1) / 2)}{ext_suffix}",
                    self.process_name,
                    self.sigmaul_channel,
                    symmetrize="quadratic",
                    kfactor=scale,
                    groups=pdf_groups,
                )

            # Symmetric hessian variations (mirrored)
            n_asym_pairs = (n_asym_entries - 1) // 2
            for j, ivar in enumerate(range(n_asym_entries, n_entries)):
                self.add_systematic(
                    h_ext[{"vars": ivar}],
                    f"pdf{n_asym_pairs + j + 1}{ext_suffix}",
                    self.process_name,
                    self.sigmaul_channel,
                    symmetrize="quadratic",
                    kfactor=scale,
                    mirror=True,
                    groups=pdf_groups,
                )

    def add_ew_isr_variation(self):
        self.logger.info("Adding EW ISR variation")
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
        self.add_shape_systematic(
            [corrh_num, corrh_den],
            f"{ew_isr_name}_Corr",
            self.process_name,
            self.sigmaul_channel,
            kfactor=2,
            mirror=True,
            symmetrize="average",
            groups=["theory_ew", "theory"],
        )

    def add_mb_fo_variations(self):
        self.logger.info("Adding mb FO variations")
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
        self.add_shape_systematic(
            [numh[{"vars": "mb_up"}], denh[{"vars": "mb_up"}]],
            "mb_fo",
            self.process_name,
            self.sigmaul_channel,
            mirror=True,
            groups=["bcQuarkMass", "theory"],
        )


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
