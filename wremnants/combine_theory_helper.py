import re

import hist
import numpy as np

from utilities import common
from wremnants import syst_tools, theory_tools
from wums import boostHistHelpers as hh
from wums import logging

logger = logging.child_logger(__name__)


def match_str_axis_entries(str_axis, match_re):
    return [x for x in str_axis if any(re.match(r, x) for r in match_re)]


class TheoryHelper(object):
    valid_np_models = [
        "Lambda",
        "Omega",
        "Delta_Lambda",
        "Lambda_Correlated",
        "Delta_Lambda_Correlated",
        "Delta_Omega",
        "binned_Omega",
        "LatticeEigvars",
        "none",
    ]

    def __init__(self, label, datagroups, args, hasNonsigSamples=False):
        toCheck = ["signal_samples", "signal_samples_inctau", "single_v_samples"]
        if hasNonsigSamples:
            if label == "W":
                toCheck.extend(["single_v_nonsig_samples", "wtau_samples"])
        for group in toCheck:
            if group not in datagroups.procGroups:
                raise ValueError(
                    f"Must define '{group}' procGroup in CardTool for theory uncertainties"
                )

        self.datagroups = datagroups
        corr_hists = self.datagroups.args_from_metadata("theoryCorr")
        self.corr_hist_name = (corr_hists[0] + "_Corr") if corr_hists else None

        self.syst_ax = "vars"
        self.corr_hist = None
        self.resumUnc = None
        self.np_model = "Delta_Lambda"
        self.pdf_from_corr = False
        self.scale_pdf_unc = -1.0
        self.mirror_tnp = True
        self.minnlo_unc = "byHelicityPt"
        self.helicity_fit_unc = False
        self.args = args
        self.label = label
        # self.correlate_fo_scale = False
        self.correlate_fo_scale = False

    def sample_label(self, sample_group):
        if sample_group not in self.datagroups.procGroups:
            raise ValueError(
                f"Failed to find sample group {sample_group} in predefined groups"
            )
        if not self.datagroups.procGroups[sample_group]:
            logger.warning(f"Sample group {sample_group} is empty")

        return self.datagroups.procGroups[sample_group][0][0]

    def configure(
        self,
        resumUnc,
        np_model,
        transitionUnc=True,
        propagate_to_fakes=True,
        tnp_scale=1.0,
        mirror_tnp=True,
        as_from_corr=True,
        pdf_from_corr=False,
        pdf_operation=None,
        samples=[],
        scale_pdf_unc=-1.0,
        minnlo_unc="byHelicityPt",
        minnlo_scale=1.0,
        from_hels=False,
        theory_symmetrize="quadratic",
        pdf_symmetrize="quadratic",
    ):

        self.set_resum_unc_type(resumUnc)
        self.set_np_model(np_model)
        self.set_propagate_to_fakes(propagate_to_fakes)
        self.set_minnlo_unc(minnlo_unc)

        self.transitionUnc = transitionUnc
        self.tnp_scale = tnp_scale
        self.mirror_tnp = mirror_tnp
        self.pdf_from_corr = pdf_from_corr
        self.as_from_corr = pdf_from_corr or as_from_corr
        self.pdf_operation = pdf_operation
        self.scale_pdf_unc = scale_pdf_unc
        self.samples = samples
        self.helicity_fit_unc = False
        self.minnlo_scale = minnlo_scale
        self.from_hels = from_hels

        convert_none = lambda x: x if x.lower() != "none" else None
        self.theory_symmetrize = convert_none(theory_symmetrize)
        self.pdf_symmetrize = convert_none(pdf_symmetrize)

    def add_all_theory_unc(self, helicity_fit_unc=False):
        self.helicity_fit_unc = helicity_fit_unc
        self.add_nonpert_unc(model=self.np_model)
        self.add_resum_unc(scale=self.tnp_scale)
        if "nnlojet" in self.corr_hist_name:
            self.add_stat_unc()
        # additional uncertainty for effect of shower and intrinsic kt on angular coeffs
        self.add_helicity_shower_kt_uncertainty()

        self.add_pdf_uncertainty(
            operation=self.pdf_operation,
            scale=self.scale_pdf_unc,
        )
        if (
            self.datagroups.args_from_metadata("pdfs")[0] != "herapdf20"
        ):  # already includes mb,mc effects
            try:
                self.add_quark_mass_vars()
            except ValueError as e:
                logger.warning(e)

    def set_minnlo_unc(self, minnloUnc):
        self.minnlo_unc = minnloUnc

    def set_resum_unc_type(self, resumUnc):
        if not resumUnc or resumUnc == "none":
            self.resumUnc = None
            return

        if not self.corr_hist_name:
            raise ValueError(
                "Cannot add resummation uncertainties. No theory correction was applied!"
            )

        # For backwards compatibility with some unblinding files that didn't have this option
        try:
            if self.datagroups.args_from_metadata("theoryCorrAltOnly"):
                raise ValueError(
                    "The theory correction was only applied as an alternate hist. Using it for systs isn't well defined!"
                )
        except IOError as e:
            logger.warning(e)

        signal_samples = self.datagroups.procGroups["signal_samples"]
        self.corr_hist = self.datagroups.getHistsForProcAndSyst(
            signal_samples[0], self.corr_hist_name
        )

        if resumUnc.startswith("tnp"):
            self.tnp_nuisance_names = [
                "b_qg",
                "b_qqDS",
                "b_qqS",
                "b_qqV",
                "b_qqbarV",
                "gamma_cusp",
                "gamma_mu_q",
                "gamma_nu",
                "h_qqV",
                "s",
            ]

            self.tnp_nuisances = match_str_axis_entries(
                self.corr_hist.axes[self.syst_ax], self.tnp_nuisance_names
            )

            if not self.tnp_nuisances:
                raise ValueError(
                    f"Did not find TNP uncertainties in hist {self.corr_hist_name}"
                )

        self.resumUnc = resumUnc

    def add_resum_unc(self, scale=1):
        if not self.resumUnc:
            logger.warning("No resummation uncertainty will be applied!")
            return

        if self.resumUnc.startswith("tnp"):
            self.add_resum_tnp_unc(scale)

            fo_scale = self.resumUnc == "tnp"
            self.add_transition_fo_scale_uncertainties(
                transition=self.transitionUnc, scale=fo_scale
            )

            if self.resumUnc == "tnp_minnlo":
                for sample_group in self.samples:
                    if self.datagroups.procGroups.get(sample_group, None):
                        # add sigma -1 uncertainty from minnlo for pt>27 GeV
                        self.add_minnlo_scale_uncertainty(
                            sample_group,
                            extra_name="highpt",
                            rebin_pt=common.ptV_binning[::2],
                            helicities_to_exclude=range(0, 8),
                            pt_min=27.0,
                            scale=self.minnlo_scale,
                            symmetrize=self.theory_symmetrize,
                        )
        elif "scale" in self.resumUnc:
            self.add_scetlib_dyturbo_scale_uncertainty(
                extra_name="inclusive",
                rebin_pt=[common.ptV_binning[0], common.ptV_binning[-1]],
                transition=self.transitionUnc,
            )
            if "binned" in self.resumUnc:
                # Add unc binned in ~10% quantiles, but keep one inclusive in pt
                # to avoid underestimating the correlated part of the uncertainty
                self.add_scetlib_dyturbo_scale_uncertainty(
                    extra_name="fine",
                    rebin_pt=common.ptV_binning[::2],
                    transition=self.transitionUnc,
                )

        if self.minnlo_unc and self.minnlo_unc not in ["none", None]:
            # sigma_-1 uncertainty is covered by scetlib-dyturbo uncertainties if they are used
            helicities_to_exclude = None if self.resumUnc == "minnlo" else [-1]
            for sample_group in ["signal_samples_inctau", "single_v_nonsig_samples"]:
                if self.datagroups.procGroups.get(sample_group, None):
                    # two sets of nuisances, one binned in ~10% quantiles, and one inclusive in pt
                    # to avoid underestimating the correlated part of the uncertainty
                    # (but scale it down to avoid double counting)
                    if self.args.muRmuFPolVar:
                        # orthogonality of chebychev polynomials means that there is no
                        # double counting with fully correlated uncertainty
                        scale_inclusive = 1.0
                    else:
                        fine_pt_binning = common.ptV_binning[::2]
                        nptfine = len(fine_pt_binning) - 1
                        scale_inclusive = np.sqrt((nptfine - 1) / nptfine)

                        self.add_minnlo_scale_uncertainty(
                            sample_group,
                            extra_name="fine",
                            rebin_pt=fine_pt_binning,
                            helicities_to_exclude=helicities_to_exclude,
                            scale=self.minnlo_scale,
                            symmetrize=self.theory_symmetrize,
                        )

                    self.add_minnlo_scale_uncertainty(
                        sample_group,
                        extra_name="inclusive",
                        rebin_pt=[common.ptV_binning[0], common.ptV_binning[-1]],
                        helicities_to_exclude=helicities_to_exclude,
                        scale=scale_inclusive * self.minnlo_scale,
                        symmetrize=self.theory_symmetrize,
                    )

    def add_minnlo_scale_uncertainty(
        self,
        sample_group,
        extra_name="",
        use_hel_hist=True,
        rebin_pt=None,
        helicities_to_exclude=None,
        pt_min=None,
        scale=1.0,
        symmetrize="quadratic",
    ):
        if not sample_group or sample_group not in self.datagroups.procGroups:
            logger.warning(
                f"Skipping QCD scale syst '{self.minnlo_unc}' for group '{sample_group}.' No process to apply it to"
            )
            return

        helicity = "Helicity" in self.minnlo_unc
        pt_binned = "Pt" in self.minnlo_unc
        scale_hist = (
            "qcdScale" if not (helicity or use_hel_hist) else "qcdScaleByHelicity"
        )
        if "helicity" in scale_hist.lower():
            use_hel_hist = True

        # All possible syst_axes
        # TODO: Move the axes to common and refer to axis_chargeVgen etc by their name attribute, not just
        # assuming the name is unchanged
        obs = self.datagroups.fit_axes
        pt_ax = "ptVgen" if "ptVgen" not in obs else "ptVgenAlt"

        syst_axes = [pt_ax, "vars"]
        syst_ax_labels = ["PtV", "var"]
        format_with_values = ["edges", "center"]

        group_name = f"QCDscale{self.sample_label(sample_group)}"
        base_name = f"{group_name}{extra_name}"

        skip_entries = []
        preop_map = {}

        # skip nominal
        skip_entries.append({"vars": "nominal"})

        # skip pythia shower and kt variations since they are handled elsewhere
        skip_entries.append({"vars": "pythia_shower_kt"})

        if helicities_to_exclude:
            for helicity in helicities_to_exclude:
                skip_entries.append({"vars": f"helicity_{helicity}_Down"})
                skip_entries.append({"vars": f"helicity_{helicity}_Up"})

        # NOTE: The map needs to be keyed on the base procs not the group names, which is
        # admittedly a bit nasty
        expanded_samples = self.datagroups.getProcGroupNames([sample_group])
        logger.debug(f"using {scale_hist} histogram for QCD scale systematics")
        logger.debug(f"expanded_samples: {expanded_samples}")

        preop_map = {}
        preop_args = {}

        if pt_binned:
            signal_samples = self.datagroups.procGroups["signal_samples"]
            binning = np.array(rebin_pt) if rebin_pt else None

            hscale = self.datagroups.getHistsForProcAndSyst(
                signal_samples[0], scale_hist
            )
            # A bit janky, but refer to the original ptVgen ax since the alt hasn't been added yet
            orig_binning = hscale.axes[pt_ax.replace("Alt", "")].edges
            if not (
                np.isclose(binning[0], orig_binning[0])
                and np.isclose(binning[-1], orig_binning[-1])
            ):
                if len(binning) == 2:
                    binning = np.array((orig_binning[0], orig_binning[-1]))
                else:
                    binning = binning[
                        (binning >= orig_binning[0] - 1e-5)
                        & (binning <= orig_binning[-1] + 1e-6)
                    ]
                logger.warning(f"Adjusting requested binning to {binning}")

            if not hh.compatibleBins(orig_binning, binning):
                logger.warning(
                    f"Requested binning {binning} is not compatible with hist binning {orig_binning}. Will not rebin!"
                )
                binning = orig_binning

            if pt_min is not None:
                pt_idx = np.argmax(binning >= pt_min)
                skip_entries.extend([{pt_ax: complex(0, x)} for x in binning[:pt_idx]])

            func = (
                syst_tools.gen_hist_to_variations
                if pt_ax == "ptVgenAlt"
                else syst_tools.hist_to_variations
            )
            preop_map = {proc: func for proc in expanded_samples}
            preop_args["gen_axes"] = [pt_ax]
            preop_args["rebin_axes"] = [pt_ax]
            preop_args["rebin_edges"] = [binning]
            if pt_ax == "ptVgenAlt":
                preop_args["gen_obs"] = ["ptVgen"]

        # Skip MiNNLO unc.
        if self.resumUnc and not (pt_binned or helicity):
            logger.warning(
                "Without pT or helicity splitting, only the SCETlib uncertainty will be applied!"
            )
        else:
            # FIXME Maybe put W and Z nuisances in the same group
            group_name += f"MiNNLO"
            self.datagroups.addSystematic(
                scale_hist,
                preOpMap=preop_map,
                preOpArgs=preop_args,
                symmetrize=symmetrize,
                processes=[sample_group],
                groups=[
                    group_name,
                    "QCDscale",
                    "angularCoeffs",
                    "theory",
                    "theory_qcd",
                ],
                splitGroup={f"angularCoeffs_A{i}": f".*helicity_{i}" for i in range(8)},
                systAxes=syst_axes,
                labelsByAxis=syst_ax_labels,
                skipEntries=skip_entries,
                baseName=base_name + "_",
                formatWithValue=format_with_values,
                passToFakes=self.propagate_to_fakes,
                scale=scale,
                name=base_name,  # Needed to allow it to be called multiple times
            )

    def add_helicity_shower_kt_uncertainty(self):
        # select the proper variation and project over gen pt unless it is one of the fit variables
        if "ptVgen" in self.datagroups.fit_axes:
            op = lambda h: h[{self.syst_ax: ["pythia_shower_kt"]}]
        else:
            op = lambda h: h[{"ptVgen": hist.sum, self.syst_ax: ["pythia_shower_kt"]}]

        processes = ["single_v_samples"]
        self.datagroups.addSystematic(
            "qcdScaleByHelicity",
            processes=processes,
            passToFakes=self.propagate_to_fakes,
            systAxes=[self.syst_ax],
            preOp=op,
            groups=["helicity_shower_kt", "angularCoeffs", "theory", "theory_qcd"],
            name="helicity_shower_kt",
            mirror=True,
        )

    def add_scetlib_dyturbo_scale_uncertainty(
        self, extra_name="", transition=True, rebin_pt=None
    ):
        obs = self.datagroups.fit_axes[:]
        pt_ax = "ptVgen" if "ptVgen" not in obs else "ptVgenAlt"

        binning = np.array(rebin_pt) if rebin_pt else None

        signal_samples = self.datagroups.procGroups["signal_samples"]
        hscale = self.datagroups.getHistsForProcAndSyst(
            signal_samples[0], self.scale_hist_name
        )
        # A bit janky, but refer to the original ptVgen ax since the alt hasn't been added yet
        orig_binning = hscale.axes[pt_ax.replace("Alt", "")].edges
        if not hh.compatibleBins(orig_binning, binning):
            logger.warning(
                f"Requested binning {binning} is not compatible with hist binning {orig_binning}. Will not rebin!"
            )
            binning = orig_binning

        for sample_group in self.samples:
            if not self.datagroups.procGroups.get(sample_group, None):
                continue

            name_append = self.sample_label(sample_group)
            name_append += extra_name

            # skip nominal
            skip_entries = []
            skip_entries.append({"vars": ["pdf0", "central"]})

            # choose the correct variations depending on whether transition variations are included
            if transition:
                sel_vars = [
                    "renorm_fact_resum_transition_scale_envelope_Down",
                    "renorm_fact_resum_transition_scale_envelope_Up",
                ]
            else:
                sel_vars = [
                    "renorm_fact_resum_scale_envelope_Down",
                    "renorm_fact_resum_scale_envelope_Up",
                ]

            syst_ax_labels = ["PtV", "var"]
            format_with_values = ["edges", "center"]

            def preop_func(h, *args, **kwargs):
                hsel = h[
                    {
                        "vars": ["pdf0" if "pdf0" in h.axes["vars"] else "central"]
                        + sel_vars
                    }
                ]
                func = (
                    syst_tools.gen_hist_to_variations
                    if pt_ax == "ptVgenAlt"
                    else syst_tools.hist_to_variations
                )
                return func(hsel, *args, **kwargs)

            preop_args = {}
            preop_args["gen_axes"] = [pt_ax]
            preop_args["rebin_axes"] = [pt_ax]
            preop_args["rebin_edges"] = [binning]

            if pt_ax == "ptVgenAlt":
                preop_args["gen_obs"] = ["ptVgen"]

            self.datagroups.addSystematic(
                self.scale_hist_name,
                processes=[sample_group],
                groups=[
                    "resumTransitionFOScale",
                    "resum",
                    "pTModeling",
                    "theory",
                    "theory_qcd",
                ],
                systAxes=[pt_ax, "vars"],
                symmetrize=self.theory_symmetrize,
                passToFakes=self.propagate_to_fakes,
                preOp=preop_func,
                preOpArgs=preop_args,
                skipEntries=skip_entries,
                labelsByAxis=syst_ax_labels,
                baseName=name_append + "_",
                formatWithValue=format_with_values,
                name=name_append,  # Needed to allow it to be called multiple times
            )

    def set_propagate_to_fakes(self, to_fakes):
        self.propagate_to_fakes = to_fakes

    def add_stat_unc(self):
        processes = ["signal_samples"]
        logger.debug(
            f"Adding theory-correction statistical uncertainties from syst entries"
        )

        self.datagroups.addSystematic(
            histname=self.corr_hist_name,
            processes=processes,
            groups=["theory"],
            systAxes=[self.syst_ax],
            passToFakes=self.propagate_to_fakes,
            preOp=lambda h: h[
                {
                    self.syst_ax: [
                        v
                        for v in self.corr_hist.axes[self.syst_ax]
                        if "per_bin_stat_unc_theory_corr" in v
                    ]
                }
            ],
            name="theoryCorrStat",
        )

    def add_resum_tnp_unc(self, magnitude, scale=1):
        syst_ax = self.corr_hist.axes[self.syst_ax]

        tnp_has_mirror = (
            "+" in self.tnp_nuisances[0]
            and self.tnp_nuisances[0].replace("+", "-") in syst_ax
        ) or (
            "-" in self.tnp_nuisances[0]
            and self.tnp_nuisances[0].replace("-", "") in syst_ax
        )
        if not tnp_has_mirror and not self.mirror_tnp:
            logger.warning(
                "Up or down variation missing in TNP histogram. Will use mirroring"
            )
            self.mirror_tnp = True

        tnp_magnitudes = ["2.5", "0.5", "1."]
        name_replace = [(f"-{x}", "Down") for x in tnp_magnitudes] + [
            (x, "Up") for x in tnp_magnitudes
        ]

        logger.debug(f"Selected TNP nuisances: {self.tnp_nuisance_names}")
        processesZ = [] if self.helicity_fit_unc else ["single_v_samples"]
        processesW = (
            ["wtau_samples", "single_v_nonsig_samples"]
            if self.helicity_fit_unc
            else ["single_v_samples"]
        )
        processes = processesW if self.label == "W" else processesZ

        self.datagroups.addSystematic(
            self.corr_hist_name,
            processes=processes,
            groups=["resumTNP", "resum", "pTModeling", "theory", "theory_qcd"],
            systAxes=["vars"],
            passToFakes=self.propagate_to_fakes,
            systNameReplace=name_replace,
            preOp=lambda h: h[
                {self.syst_ax: [h.axes[self.syst_ax][0], *self.tnp_nuisances]}
            ],
            mirror=self.mirror_tnp,
            scale=scale,
            skipEntries=[{self.syst_ax: ["central", "pdf0"]}],
            name=f"resumTNP",
            baseName=f"resumTNP_",
        )

    def add_nonpert_unc(self, model):
        if not self.resumUnc:
            return

        self.set_np_model(model)
        if self.np_model:
            self.add_gamma_np_uncertainties()
            if "Correlated" in self.np_model:
                self.add_correlated_np_uncertainties()
            else:
                self.add_uncorrelated_np_uncertainties()
        else:
            logger.warning("Will not add any nonperturbative uncertainty!")

    def set_np_model(self, model):
        if model in ["none", None]:
            self.np_model = None
            return

        if model not in TheoryHelper.valid_np_models:
            raise ValueError(
                f"Model choice {model} is not a supported model. Valid choices are {TheoryHelper.valid_np_models}"
            )

        signal_samples = self.datagroups.procGroups["signal_samples"]
        self.np_hist_name = self.corr_hist_name.replace("Corr", "FlavDepNP")
        self.np_hist = self.datagroups.getHistsForProcAndSyst(
            signal_samples[0], self.np_hist_name
        )

        self.scale_hist_name = self.corr_hist_name.replace("Corr", "PtDepScales")

        var_name = model
        var_name = var_name.replace("binned_", "")
        var_name = var_name.replace("_Correlated", "")

        if (
            not any(var_name in x for x in self.np_hist.axes[self.syst_ax])
            and model != "LatticeEigvars"
        ):
            raise ValueError(
                f"NP model choice was '{model}' but did not find corresponding variations in the histogram"
            )

        self.np_model = model

    def add_gamma_np_uncertainties(self):
        if (
            self.np_model == "LatticeEigvars"
        ):  # new SCETlib NP model, using lattice central values and constrained eigenvariations
            lattice_vals = [
                "lambda2_nu0.0696-lambda4_nu0.0122-lambda_inf_nu1.1Ext",
                "lambda2_nu0.1044-lambda4_nu0.0026-lambda_inf_nu2.1Ext",
                "lambda2_nu0.1153-lambda4_nu0.0032-lambda_inf_nu1.6Ext",
                "lambda2_nu0.0587-lambda4_nu0.0116-lambda_inf_nu1.6Ext",
                "lambda2_nu0.0873-lambda4_nu0.0092",
                "lambda2_nu0.0867-lambda4_nu0.0056",
            ]
            if not all([x in self.corr_hist.axes[self.syst_ax] for x in lattice_vals]):
                raise ValueError(
                    f"Using the lattice NP model, but could not find the 3 Eigenvariations for gamma in hist {self.corr_hist_name}"
                )

            gamma_nuisance_names = [
                "scetlibNPgammaEigvar1",
                "scetlibNPgammaEigvar2",
                "scetlibNPgammaEigvar3",
            ]
            var_names = [
                f"{name}{direction}"
                for name in gamma_nuisance_names
                for direction in ["Up", "Down"]
            ]
            var_vals = lattice_vals
        else:
            # Since "c_nu = 0.1 is the central value, it doesn't show up in the name"
            gamma_vals = list(
                filter(
                    lambda x: x in self.corr_hist.axes[self.syst_ax],
                    ["c_nu-0.1-omega_nu0.5", "omega_nu0.5", "c_nu-0.25", "c_nu-0.25"],
                )
            )

            if len(gamma_vals) != 2:
                raise ValueError(
                    f"Failed to find consistent variation for gamma NP in hist {self.corr_hist_name}"
                )

            gamma_nuisance_name = "scetlibNPgamma"

            var_vals = gamma_vals
            var_names = [f"{gamma_nuisance_name}Down", f"{gamma_nuisance_name}Up"]

        logger.debug(f"Adding gamma uncertainties from syst entries {var_vals}")

        processesZ = ["single_v_samples"]
        processesW = ["single_v_samples"]
        processes = processesW if self.label == "W" else processesZ

        self.datagroups.addSystematic(
            self.corr_hist_name,
            processes=processes,
            passToFakes=self.propagate_to_fakes,
            systAxes=[self.syst_ax],
            preOp=lambda h: h[{self.syst_ax: var_vals}],
            outNames=var_names,
            groups=["resumNonpert", "resum", "pTModeling", "theory", "theory_qcd"],
            name="scetlibNP",
        )

    def add_resum_scale_uncertainty(self, name_append):
        obs = self.datagroups.fit_axes[:]
        if not obs:
            raise ValueError(
                "Failed to find the observable names for the resummation uncertainties"
            )

        theory_hist = self.datagroups.getHistsForProcAndSyst(
            self.samples[0],  # theory_hist_name #FIXME
        )
        resumscale_nuisances = match_str_axis_entries(
            theory_hist.axes[self.syst_ax],
            [
                "^nuB.*",
                "nuS.*",
                "^muB.*",
                "^muS.*",
            ],
        )

        expanded_samples = self.datagroups.getProcGroupNames(self.samples)
        # syst_ax = "vars" # FIXME

        self.datagroups.addSystematic(
            theory_hist,
            processes=self.samples,
            groups=["resumScale", "resum", "pTModeling", "theory", "theory_qcd"],
            passToFakes=self.propagate_to_fakes,
            # skipEntries=[{syst_ax: x} for x in both_exclude + tnp_nuisances], # FIXME
            systAxes=["downUpVar"],  # Is added by the preOpMap
            preOpMap={
                s: lambda h: hh.syst_min_and_max_env_hist(
                    h, obs, "vars", resumscale_nuisances
                )
                for s in expanded_samples
            },
            outNames=[
                f"scetlibResumScale{name_append}Up",
                f"scetlibResumScale{name_append}Down",
            ],
            name=f"resumScale{name_append}",
            baseName=f"resumScale{name_append}_",
        )
        # TODO: check if this is actually the proper treatment of these uncertainties
        self.datagroups.addSystematic(
            theory_hist,
            processes=self.samples,
            groups=["resumScale", "resum", "pTModeling", "theory", "theory_qcd"],
            passToFakes=self.propagate_to_fakes,
            systAxes=["vars"],
            preOpMap={
                s: lambda h: h[
                    {
                        "vars": [
                            "kappaFO0.5-kappaf2.",
                            "kappaFO2.-kappaf0.5",
                            "mufdown",
                            "mufup",
                        ]
                    }
                ]
                for s in expanded_samples
            },
            outNames=[
                f"scetlib_kappa{name_append}Up",
                f"scetlib_kappa{name_append}Down",
                f"scetlib_muF{name_append}Up",
                f"scetlib_muF{name_append}Down",
            ],
            name=f"resumFOScale{name_append}",
            baseName=f"resumScale{name_append}_",  # TODO: check if intented or should it be 'resumFOScale' instead?
        )

    def add_correlated_np_uncertainties(self):
        if self.np_model == "LatticeEigvars":
            np_map = {
                "lambda2": ["0.0", "0.5"],
                "delta_lambda2": ["0.105", "0.145"],
                "lambda4": ["0.01", "0.16"],
            }
        elif "Lambda" in self.np_model:
            np_map = {
                "Lambda2": [
                    "-0.25",
                    "0.25",
                ],
                "Delta_Lambda2": [
                    "-0.02",
                    "0.02",
                ],
                "Lambda4": [".01", ".16"],
            }
        else:
            np_map = {
                "Omega": ["0.", "0.8"],
                "Delta_Omega": ["-0.02", "0.02"],
            }

        if "Delta" not in self.np_model and self.np_model != "LatticeEigvars":
            to_remove = list(filter(lambda x: "Delta" in x, np_map.keys())) + list(
                filter(lambda x: "delta" in x, np_map.keys())
            )
            for k in to_remove:
                np_map.pop(k)

        def operation(h, entries):
            return h[{self.syst_ax: entries}]

        for nuisance, vals in np_map.items():
            entries = [nuisance + v for v in vals]
            rename = f"scetlibNP{nuisance}"
            # operation = lambda h : h[{self.syst_ax : entries}]
            self.datagroups.addSystematic(
                self.corr_hist_name,
                processes=["single_v_samples"],
                groups=["resumNonpert", "resum", "pTModeling", "theory", "theory_qcd"],
                systAxes=[self.syst_ax],
                passToFakes=self.propagate_to_fakes,
                preOp=operation,
                preOpArgs=dict(entries=entries),
                outNames=[f"{rename}Down", f"{rename}Up"],
                name=rename,
            )

    def add_uncorrelated_np_uncertainties(self):
        if self.np_model == "LatticeEigvars":
            np_map = {
                "lambda2": ["0.0", "0.5"],
                "delta_lambda2": ["0.105", "0.145"],
                "lambda4": ["0.01", "0.16"],
            }
        elif "Lambda" in self.np_model:
            np_map = {
                "Lambda2": [
                    "-0.25",
                    "0.25",
                ],
                "Delta_Lambda2": [
                    "-0.02",
                    "0.02",
                ],
                "Lambda4": [".01", ".16"],
            }
        else:
            np_map = {
                "Omega": ["0.", "0.8"],
                "Delta_Omega": ["-0.02", "0.02"],
            }

        if "Delta" not in self.np_model and self.np_model != "LatticeEigvars":
            to_remove = list(filter(lambda x: "Delta" in x, np_map.keys())) + list(
                filter(lambda x: "delta" in x, np_map.keys())
            )
            for k in to_remove:
                np_map.pop(k)

        for label, vals in np_map.items():
            if not all(label + v in self.np_hist.axes[self.syst_ax] for v in vals):
                tmpvals = [
                    x.replace(label, "")
                    for x in self.np_hist.axes[self.syst_ax]
                    if re.match(rf"^{label}\d+", x)
                ]
                if tmpvals:
                    logger.warning(
                        f"Using variations {tmpvals} rather than default values {vals}!"
                    )
                    np_map[label] = tmpvals
                else:
                    raise ValueError(
                        f"Failed to find all vars {vals} for var {label} in hist {self.np_hist_name}"
                    )

        binned = "binned" in self.np_model
        gen_axes = ["absYVgenNP", "chargeVgenNP"]
        sum_axes = [] if binned else ["absYVgenNP"]
        syst_axes = (
            ["absYVgenNP", "chargeVgenNP", self.syst_ax]
            if binned
            else ["chargeVgenNP", self.syst_ax]
        )
        operation = lambda h, entries: syst_tools.hist_to_variations(
            h[{self.syst_ax: [h.axes[self.syst_ax][0], *entries]}],
            gen_axes=gen_axes,
            sum_axes=sum_axes,
        )
        for sample_group in self.samples:
            if not self.datagroups.procGroups.get(sample_group, None):
                continue
            label = self.sample_label(sample_group)
            for nuisance, vals in np_map.items():
                entries = [nuisance + v for v in vals]
                rename = f"scetlibNP{label}{nuisance}"
                self.datagroups.addSystematic(
                    self.np_hist_name,
                    processes=[sample_group],
                    groups=[
                        "resumNonpert",
                        "resum",
                        "pTModeling",
                        "theory",
                        "theory_qcd",
                    ],
                    systAxes=syst_axes,
                    passToFakes=self.propagate_to_fakes,
                    preOp=operation,
                    preOpArgs={"entries": entries},
                    # outNames=[f"{rename}Down", f"{rename}Up"] if not binned else None,
                    systNameReplace=[
                        (entries[1], f"{rename}Up"),
                        (entries[0], f"{rename}Down"),
                    ],
                    skipEntries=[{self.syst_ax: ["central", "pdf0"]}],
                    name=rename,
                )

    def add_pdf_uncertainty(self, operation=None, scale=-1.0):
        pdf = self.datagroups.args_from_metadata("pdfs")[0]
        pdfInfo = theory_tools.pdf_info_map("ZmumuPostVFP", pdf)
        pdfName = pdfInfo["name"]
        scale = (
            scale
            if scale != -1.0
            else theory_tools.pdf_inflation_factor(pdfInfo, self.args.noi)
        )
        pdf_hist = pdfName

        if self.pdf_from_corr:
            pdf_corr_hist = f"{self.corr_hist_name.replace("Corr", "pdfvars_Corr")}"
            if pdf_corr_hist.replace(
                "_Corr", ""
            ) not in self.datagroups.args_from_metadata("theoryCorr"):
                raise RuntimeError(
                    f"PDF correction histogram {pdf_corr_hist} not found in metadata. "
                    "Cannot add PDF uncertainty from corrections!"
                )
            pdf_hist = pdf_corr_hist

        if self.from_hels:
            pdf_hist += "ByHelicity"

        logger.info(f"Using PDF hist {pdf_hist}, apply scaling of {scale}")

        pdf_ax = self.syst_ax if self.pdf_from_corr else "pdfVar"
        symHessian = pdfInfo["combine"] == "symHessian"

        processes = ["single_v_samples"]

        pdf_args = dict(
            processes=processes,
            mirror=True if symHessian else False,
            groups=[pdfName, f"{pdfName}NoAlphaS", "theory", "theory_qcd"],
            passToFakes=self.propagate_to_fakes,
            preOpMap=operation,
            scale=pdfInfo.get("scale", 1) * scale,
            symmetrize=self.pdf_symmetrize,
            systAxes=[pdf_ax],
        )
        if self.pdf_from_corr:
            self.datagroups.addSystematic(
                pdf_hist,
                outNames=[""]
                + theory_tools.pdfNamesAsymHessian(pdfInfo["entries"], pdfset=pdfName)[
                    1:
                ],
                **pdf_args,
            )
        else:
            self.datagroups.addSystematic(
                pdf_hist, skipEntries=[{pdf_ax: "^pdf0[a-z]*"}], **pdf_args
            )
            if pdfName == "pdfHERAPDF20":

                self.datagroups.addSystematic(
                    pdf_hist.replace("pdfHERAPDF20", "pdfHERAPDF20ext"),
                    skipEntries=[
                        {pdf_ax: "^pdf(0|[6-8])[a-z]*"}
                    ],  # exclude 0, 6 and above
                    **pdf_args,
                )

                tmp_pdf_args = pdf_args.copy()
                tmp_pdf_args["mirror"] = True
                self.datagroups.addSystematic(
                    pdf_hist.replace("pdfHERAPDF20", "pdfHERAPDF20ext"),
                    skipEntries=[
                        {pdf_ax: "^(?!pdf[6-8][a-z]*)"}
                    ],  # exclude everything but 6-8
                    **tmp_pdf_args,
                )

    def add_pdf_alphas_variation(self, noi=False):
        pdf = self.datagroups.args_from_metadata("pdfs")[0]
        pdfInfo = theory_tools.pdf_info_map("ZmumuPostVFP", pdf)
        pdfName = pdfInfo["name"]
        as_range = pdfInfo["alphasRange"]

        if self.as_from_corr:
            asname = f"{self.corr_hist_name.replace("Corr", "pdfas_Corr")}"
            # alphaS from correction histograms only available for some pdf sets,
            # so fall back to CT18Z for other sets
            if asname.replace("_Corr", "") not in self.datagroups.args_from_metadata(
                "theoryCorr"
            ):
                asname = "scetlib_dyturbo_CT18Z_N3p0LL_N2LO_pdfas_Corr"
                if asname.replace("_Corr", "") in self.datagroups.args_from_metadata(
                    "theoryCorr"
                ):
                    logger.warning(
                        f"AlphaS correction histogram {asname} not found in theoryCorrs. "
                        "Falling back to default alphaS corrections scetlib_dyturbo_CT18Z_N3p0LL_N2LO_pdfasCorr."
                    )
                    pdf = "ct18z"
                    pdfInfo = theory_tools.pdf_info_map("ZmumuPostVFP", pdf)
                    pdfName = pdfInfo["name"]
                    as_range = pdfInfo["alphasRange"]
                    as_range = theory_tools.pdfMap[pdf]["alphasRange"]
                else:
                    raise RuntimeError(
                        f"AlphaS correction histogram {asname} not found in theoryCorrs. "
                        "Cannot add alphaS uncertainty!"
                    )
        else:
            asname = f"{pdfName}alphaS{as_range}"

        if self.as_from_corr and self.from_hels and not asname.endswith("ByHelicity"):
            asname += "ByHelicity"

        input_variation = int(as_range) * 10 ** (-len(as_range))
        target_variation = 0.002 if noi else 0.0015
        scale = target_variation / input_variation

        as_replace = [("as", "pdfAlphaS")]
        if as_range == "002":
            as_replace.extend([("0116", "Down"), ("0120", "Up")])
        elif as_range == "001":
            as_replace.extend([("0117", "Down"), ("0119", "Up")])
        else:
            raise RuntimeError("Unsupported alphaS range for PDF set!")

        as_args = dict(
            histname=asname,
            processes=["single_v_samples"],
            mirror=False,
            noi=noi,
            noConstraint=noi,
            groups=[pdfName],
            systAxes=["vars" if self.as_from_corr else "alphasVar"],
            scale=scale,
            symmetrize="average" if noi else self.pdf_symmetrize,
            passToFakes=self.propagate_to_fakes,
        )
        if not noi:
            as_args["groups"].extend([f"{pdfName}AlphaS", "theory", "theory_qcd"])
        if self.as_from_corr:
            as_args["outNames"] = ["", "pdfAlphaSDown", "pdfAlphaSUp"]
        else:
            as_args["systNameReplace"] = as_replace
            as_args["skipEntries"] = [{"alphasVar": "as0118"}]
        logger.info(f"Using alphaS variation {asname}, applying scaling of {scale}")
        self.datagroups.addSystematic(**as_args)

    def add_transition_fo_scale_uncertainties(self, transition=True, scale=True):
        for sample_group in self.samples:
            if not self.datagroups.procGroups.get(sample_group, None):
                continue

            name_append = self.sample_label(sample_group)

            sel_vars = []
            outNames = []

            if transition:
                sel_vars.extend(
                    ["transition_points0.2_0.35_1.0", "transition_points0.2_0.75_1.0"]
                )
                outNames.extend(
                    [
                        f"resumTransition{name_append}Down",
                        f"resumTransition{name_append}Up",
                    ]
                )

            if scale:
                sel_vars.extend(
                    ["renorm_scale_pt20_envelope_Down", "renorm_scale_pt20_envelope_Up"]
                )
                resum_fo_name = "resumFOScale"
                if not self.correlate_fo_scale:
                    resum_fo_name += name_append
                outNames.extend([f"{resum_fo_name}Down", f"{resum_fo_name}Up"])

            if not sel_vars:
                # nothing to do
                continue

            self.datagroups.addSystematic(
                self.corr_hist_name,
                processes=[sample_group],
                groups=[
                    "resumTransitionFOScale",
                    "resum",
                    "pTModeling",
                    "theory",
                    "theory_qcd",
                ],
                systAxes=["vars"],
                symmetrize=self.theory_symmetrize,
                passToFakes=self.propagate_to_fakes,
                preOp=lambda h: h[{"vars": sel_vars}],
                outNames=outNames,
                name=f"resumTransitionFOScale{name_append}",
            )

    def add_quark_mass_vars(self, from_minnlo=True):
        pdfs = self.datagroups.args_from_metadata("pdfs")
        theory_corrs = self.datagroups.args_from_metadata("theoryCorr")

        from_minnlo = not (
            "scetlib_dyturboMSHT20mcrange" in theory_corrs
            and "scetlib_dyturboMSHT20mcrange" in theory_corrs
            and "msht20" in pdfs[0]
        )

        if from_minnlo:
            if (
                "msht20mbrange" in pdfs
                and "msht20mcrange" in pdfs
                and pdfs[0] not in ["msht20", "msht20mbrange", "msht20mcrange"]
            ):
                raise ValueError(
                    "Using the mass variation sets from MiNNLO requires MSHT20 as the central set"
                )
            elif (
                "msht20mbrange_renorm" not in pdfs or "msht20mcrange_renorm" not in pdfs
            ):
                raise ValueError(
                    "Must include the msht20mb(c)range pdf sets to take the mass variation from MiNNLO"
                )
        elif not (
            "msht20" in pdfs[0]
            and "scetlib_dyturboMSHT20mbrange" in theory_corrs
            and "scetlib_dyturboMSHT20mcrange" in theory_corrs
        ):
            raise ValueError(
                "In order to take the mb(c) mass unc. from SCETlib+DYTurbo, you need to include those corr files and use MSHT20 as central PDF"
            )

        if from_minnlo:
            if self.from_hels:
                bhist = "pdfMSHT20mbrangeByHelicity"
            else:
                bhist = "pdfMSHT20mbrange"
        else:
            bhist = "scetlib_dyturboMSHT20mbrangeCorr"
        syst_ax = "pdfVar" if from_minnlo else "vars"

        self.datagroups.addSystematic(
            bhist,
            processes=self.samples,
            systAxes=[syst_ax],
            symmetrize=self.pdf_symmetrize,
            groups=["bcQuarkMass", "pTModeling", "theory", "theory_qcd"],
            passToFakes=self.propagate_to_fakes,
            outNames=[
                "",
                "pdfMSHT20mbrangeDown",
            ]
            + [""] * 4
            + ["pdfMSHT20mbrangeUp"],
        )

        self.datagroups.addSystematic(
            bhist.replace("brange", "crange"),
            processes=self.samples,
            systAxes=[syst_ax],
            symmetrize=self.pdf_symmetrize,
            groups=["bcQuarkMass", "pTModeling", "theory", "theory_qcd"],
            passToFakes=self.propagate_to_fakes,
            outNames=[
                "",
                "pdfMSHT20mcrangeDown",
            ]
            + [""] * 6
            + ["pdfMSHT20mcrangeUp"],
        )
