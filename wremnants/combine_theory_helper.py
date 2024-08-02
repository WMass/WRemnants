from utilities import logging,common, boostHistHelpers as hh
from utilities.io_tools import input_tools
from wremnants import syst_tools,theory_tools
import numpy as np
import re
import hist

logger = logging.child_logger(__name__)

class TheoryHelper(object):
    valid_np_models = ["Lambda", "Omega", "Delta_Lambda", "Delta_Omega", "binned_Omega", "none"]
    def __init__(self, card_tool, args, hasNonsigSamples=False):
        toCheck = ['signal_samples', 'signal_samples_inctau', 'single_v_samples']
        if hasNonsigSamples:
            toCheck.extend(['single_v_nonsig_samples', 'wtau_samples'])
        for group in toCheck:
            if group not in card_tool.procGroups:
                raise ValueError(f"Must define '{group}' procGroup in CardTool for theory uncertainties")
        
        self.card_tool = card_tool
        corr_hists = input_tools.args_from_metadata(self.card_tool, "theoryCorr")
        self.corr_hist_name = (corr_hists[0]+"Corr") if corr_hists else None
        self.syst_ax = "vars"
        self.corr_hist = None
        self.resumUnc = None
        self.np_model = "Delta_Lambda"
        self.pdf_from_corr = False
        self.scale_pdf_unc = -1.
        self.mirror_tnp = True
        self.minnlo_unc = 'byHelicityPt'
        self.skipFromSignal = False
        self.args = args

    def sample_label(self, sample_group):
        if sample_group not in self.card_tool.procGroups:
            raise ValueError(f"Failed to find sample group {sample_group} in predefined groups")
        if not self.card_tool.procGroups[sample_group]:
            logger.warning(f"Sample group {sample_group} is empty")

        return self.card_tool.procGroups[sample_group][0][0] 

    def configure(self, resumUnc, np_model,
            transitionUnc = True,
            propagate_to_fakes=True, 
            tnp_scale=1.,
            mirror_tnp=True,
            as_from_corr=True,
            pdf_from_corr=False,
            pdf_operation=None,
            scale_pdf_unc=-1.,
            minnlo_unc='byHelicityPt'):

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
        self.samples = []
        self.skipFromSignal = False

    def add_all_theory_unc(self, samples, skipFromSignal=False):
        self.samples = samples
        self.skipFromSignal = skipFromSignal
        self.add_nonpert_unc(model=self.np_model)
        self.add_resum_unc(scale=self.tnp_scale)
        self.add_pdf_uncertainty(operation=self.pdf_operation, scale=self.scale_pdf_unc)
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
            raise ValueError("Cannot add resummation uncertainties. No theory correction was applied!")

        if input_tools.args_from_metadata(self.card_tool, "theoryCorrAltOnly"):
            raise ValueError("The theory correction was only applied as an alternate hist. Using it for systs isn't well defined!")

        signal_samples = self.card_tool.procGroups['signal_samples']
        self.corr_hist = self.card_tool.getHistsForProcAndSyst(signal_samples[0], self.corr_hist_name)

        if resumUnc.startswith("tnp"):
            self.tnp_nuisance_names = ['b_qg',
                                   'b_qqDS',
                                   'b_qqS',
                                   'b_qqV',
                                   'b_qqbarV',
                                   'gamma_cusp',
                                   'gamma_mu_q',
                                   'gamma_nu',
                                   'h_qqV',
                                   's'
            ]

            self.tnp_nuisances = self.card_tool.match_str_axis_entries(self.corr_hist.axes[self.syst_ax], self.tnp_nuisance_names)
                                    
            if not self.tnp_nuisances:
                raise ValueError(f"Did not find TNP uncertainties in hist {self.corr_hist_name}")

        self.resumUnc = resumUnc
        
    def add_resum_unc(self, scale=1):
        if not self.resumUnc:
            logger.warning("No resummation uncertainty will be applied!")
            return

        if self.resumUnc.startswith("tnp"):
            self.add_resum_tnp_unc(scale)

            fo_scale = self.resumUnc == "tnp"
            self.add_transition_fo_scale_uncertainties(transition = self.transitionUnc, scale = fo_scale)

            if self.resumUnc == "tnp_minnlo":
                for sample_group in self.samples:
                    if self.card_tool.procGroups.get(sample_group, None):
                        # add sigma -1 uncertainty from minnlo for pt>27 GeV
                        self.add_minnlo_scale_uncertainty(sample_group, extra_name = "highpt", rebin_pt=common.ptV_binning[::2], helicities_to_exclude=range(0, 8), pt_min=27.)
        elif self.resumUnc == "scale":
            # two sets of nuisances, one binned in ~10% quantiles, and one inclusive in pt
            # to avoid underestimating the correlated part of the uncertainty
            self.add_scetlib_dyturbo_scale_uncertainty(extra_name = "fine", rebin_pt=common.ptV_binning[::2], transition = self.transitionUnc)
            self.add_scetlib_dyturbo_scale_uncertainty(extra_name = "inclusive", rebin_pt=[common.ptV_binning[0], common.ptV_binning[-1]], transition = self.transitionUnc)


        if self.minnlo_unc and self.minnlo_unc not in ["none", None]:
            # sigma_-1 uncertainty is covered by scetlib-dyturbo uncertainties if they are used
            helicities_to_exclude = None if self.resumUnc == "minnlo" else [-1]
            for sample_group in self.samples:
                if self.card_tool.procGroups.get(sample_group, None):
                    # two sets of nuisances, one binned in ~10% quantiles, and one inclusive in pt
                    # to avoid underestimating the correlated part of the uncertainty
                    # (but scale it down to avoid double counting)
                    if self.args.muRmuFPolVar:
                        # orthogonality of chebychev polynomials means that there is no
                        # double counting with fully correlated uncertainty
                        scale_inclusive = 1.
                    else:
                        fine_pt_binning = common.ptV_binning[::2]
                        nptfine = len(fine_pt_binning) - 1
                        scale_inclusive = np.sqrt((nptfine - 1)/nptfine)

                        self.add_minnlo_scale_uncertainty(sample_group, extra_name = "fine", rebin_pt=fine_pt_binning, helicities_to_exclude=helicities_to_exclude)

                    if not (self.args.muRmuFPolVar and self.args.muRmuFFullyCorr):
                        self.add_minnlo_scale_uncertainty(sample_group, extra_name = "inclusive", rebin_pt=[common.ptV_binning[0], common.ptV_binning[-1]], helicities_to_exclude=helicities_to_exclude, scale = scale_inclusive)

            # additional uncertainty for effect of shower and intrinsic kt on angular coeffs
            self.add_helicity_shower_kt_uncertainty()

    def add_minnlo_scale_uncertainty(self, sample_group, extra_name="", use_hel_hist=True, rebin_pt=None, helicities_to_exclude=None, pt_min = None, scale = 1.0):
        if not sample_group or sample_group not in self.card_tool.procGroups:
            logger.warning(f"Skipping QCD scale syst '{self.minnlo_unc}' for group '{sample_group}.' No process to apply it to")
            return
            
        helicity = "Helicity" in self.minnlo_unc
        pt_binned = "Pt" in self.minnlo_unc
        scale_hist = "qcdScale" if not (helicity or use_hel_hist) else "qcdScaleByHelicity"
        if "helicity" in scale_hist.lower():
            use_hel_hist = True

        # All possible syst_axes
        # TODO: Move the axes to common and refer to axis_chargeVgen etc by their name attribute, not just
        # assuming the name is unchanged
        obs = self.card_tool.fit_axes
        pt_ax = "ptVgen" if "ptVgen" not in obs else "ptVgenAlt"

        syst_axes = [pt_ax, "vars"]
        syst_ax_labels = ["PtV", "var"]
        format_with_values = ["edges", "center"]

        group_name = f"QCDscale{self.sample_label(sample_group)}"
        base_name = f"{group_name}{extra_name}"

        skip_entries = []
        preop_map = {}

        # skip nominal
        skip_entries.append({"vars" : "nominal"})

        # skip pythia shower and kt variations since they are handled elsewhere
        skip_entries.append({"vars" : "pythia_shower_kt"})

        if helicities_to_exclude:
            for helicity in helicities_to_exclude:
                skip_entries.append({"vars" : f"helicity_{helicity}_Down"})
                skip_entries.append({"vars" : f"helicity_{helicity}_Up"})

        # NOTE: The map needs to be keyed on the base procs not the group names, which is
        # admittedly a bit nasty
        expanded_samples = self.card_tool.getProcNames([sample_group])
        logger.debug(f"using {scale_hist} histogram for QCD scale systematics")
        logger.debug(f"expanded_samples: {expanded_samples}")

        preop_map = {}
        preop_args = {}

        if pt_binned:
            signal_samples = self.card_tool.procGroups['signal_samples']
            binning = np.array(rebin_pt) if rebin_pt else None

            hscale = self.card_tool.getHistsForProcAndSyst(signal_samples[0], scale_hist)
            # A bit janky, but refer to the original ptVgen ax since the alt hasn't been added yet
            orig_binning = hscale.axes[pt_ax.replace("Alt", "")].edges
            if not (np.isclose(binning[0], orig_binning[0]) and np.isclose(binning[-1], orig_binning[-1])):
                if len(binning) == 2:
                    binning = np.array((orig_binning[0], orig_binning[-1]))
                else:
                    binning = binning[(binning >= orig_binning[0]-1e-5) & (binning <= orig_binning[-1]+1e-6)]
                logger.warning(f"Adjusting requested binning to {binning}")

            if not hh.compatibleBins(orig_binning, binning):
                logger.warning(f"Requested binning {binning} is not compatible with hist binning {orig_binning}. Will not rebin!")
                binning = orig_binning

            if pt_min is not None:
                pt_idx = np.argmax(binning >= pt_min)
                skip_entries.extend([{pt_ax : complex(0, x)} for x in binning[:pt_idx]])

            func = syst_tools.gen_hist_to_variations if pt_ax == "ptVgenAlt" else syst_tools.hist_to_variations
            preop_map = {proc : func for proc in expanded_samples}
            preop_args["gen_axes"] = [pt_ax]
            preop_args["rebin_axes"] = [pt_ax]
            preop_args["rebin_edges"] = [binning]
            if pt_ax == "ptVgenAlt":
                preop_args["gen_obs"] = ["ptVgen"]

        # Skip MiNNLO unc. 
        if self.resumUnc and not (pt_binned or helicity):
            logger.warning("Without pT or helicity splitting, only the SCETlib uncertainty will be applied!")
        else:
            #FIXME Maybe put W and Z nuisances in the same group
            group_name += f"MiNNLO"
            self.card_tool.addSystematic(scale_hist,
                preOpMap=preop_map,
                preOpArgs=preop_args,
                symmetrize = "quadratic",
                processes=[sample_group],
                group=group_name,
                splitGroup={"QCDscale": ".*", "angularCoeffs" : ".*", "theory": ".*"},
                systAxes=syst_axes,
                labelsByAxis=syst_ax_labels,
                skipEntries=skip_entries,
                baseName=base_name+"_",
                formatWithValue=format_with_values,
                passToFakes=self.propagate_to_fakes,
                scale = scale,
                rename=base_name, # Needed to allow it to be called multiple times
            )

    def add_helicity_shower_kt_uncertainty(self):
        # select the proper variation and project over gen pt unless it is one of the fit variables
        if "ptVgen" in self.card_tool.fit_axes:
            op = lambda h: h[{self.syst_ax : ["pythia_shower_kt"]}]
        else:
            op = lambda h: h[{"ptVgen" : hist.sum, self.syst_ax : ["pythia_shower_kt"]}]

        self.card_tool.addSystematic(name="qcdScaleByHelicity",
            processes=['wtau_samples', 'single_v_nonsig_samples'] if self.skipFromSignal else ['single_v_samples'],
            passToFakes=self.propagate_to_fakes,
            systAxes=[self.syst_ax],
            preOp=op,
            group="helicity_shower_kt",
            splitGroup={"angularCoeffs": ".*", "theory": ".*"},
            rename="helicity_shower_kt",
            mirror=True,
        )

    def add_scetlib_dyturbo_scale_uncertainty(self, extra_name="", transition = True, rebin_pt=None):
        obs = self.card_tool.fit_axes[:]
        pt_ax = "ptVgen" if "ptVgen" not in obs else "ptVgenAlt"

        binning = np.array(rebin_pt) if rebin_pt else None

        signal_samples = self.card_tool.procGroups['signal_samples']
        hscale = self.card_tool.getHistsForProcAndSyst(signal_samples[0], self.scale_hist_name)
        # A bit janky, but refer to the original ptVgen ax since the alt hasn't been added yet
        orig_binning = hscale.axes[pt_ax.replace("Alt", "")].edges
        if not hh.compatibleBins(orig_binning, binning):
            logger.warning(f"Requested binning {binning} is not compatible with hist binning {orig_binning}. Will not rebin!")
            binning = orig_binning

        for sample_group in self.samples:
            if not self.card_tool.procGroups.get(sample_group, None):
                continue

            name_append = self.sample_label(sample_group)
            name_append += extra_name

            # skip nominal
            skip_entries = []
            skip_entries.append({"vars" : "pdf0"})

            # choose the correct variations depending on whether transition variations are included
            if transition:
                sel_vars = ["renorm_fact_resum_transition_scale_envelope_Down", "renorm_fact_resum_transition_scale_envelope_Up"]
            else:
                sel_vars = ["renorm_fact_resum_scale_envelope_Down", "renorm_fact_resum_scale_envelope_Up"]


            syst_axes = [pt_ax, "vars"]
            syst_ax_labels = ["PtV", "var"]
            format_with_values = ["edges", "center"]

            def preop_func(h, *args, **kwargs):
                hsel = h[{"vars" : ["pdf0"] + sel_vars}]
                func = syst_tools.gen_hist_to_variations if pt_ax == "ptVgenAlt" else syst_tools.hist_to_variations
                return func(hsel, *args, **kwargs)

            preop_args = {}
            preop_args["gen_axes"] = [pt_ax]
            preop_args["rebin_axes"] = [pt_ax]
            preop_args["rebin_edges"] = [binning]

            if pt_ax == "ptVgenAlt":
                preop_args["gen_obs"] = ["ptVgen"]

            self.card_tool.addSystematic(name=self.scale_hist_name,
                processes=[sample_group],
                group="resumTransitionFOScale",
                splitGroup={"resum": ".*", "pTModeling" : ".*", "theory": ".*"},
                systAxes=[pt_ax, "vars"],
                symmetrize = "quadratic",
                passToFakes=self.propagate_to_fakes,
                preOp = preop_func,
                preOpArgs = preop_args,
                skipEntries = skip_entries,
                labelsByAxis=syst_ax_labels,
                baseName=name_append+"_",
                formatWithValue=format_with_values,
                rename=name_append, # Needed to allow it to be called multiple times
            )

    def set_propagate_to_fakes(self, to_fakes):
        self.propagate_to_fakes = to_fakes

    def add_resum_tnp_unc(self, magnitude, scale=1):
        syst_ax = self.corr_hist.axes[self.syst_ax]

        tnp_has_mirror = ("+" in self.tnp_nuisances[0] and self.tnp_nuisances[0].replace("+", "-") in syst_ax) or \
                            ("-" in self.tnp_nuisances[0] and self.tnp_nuisances[0].replace("-", "") in syst_ax)
        if not tnp_has_mirror and not self.mirror_tnp:
            logger.warning("Up or down variation missing in TNP histogram. Will use mirroring")
            self.mirror_tnp = True

        central_var = syst_ax[0]
        
        tnp_magnitudes = ["2.5", "0.5", "1."]
        name_replace = [(f"-{x}", "Down") for x in tnp_magnitudes]+[(x, "Up") for x in tnp_magnitudes]

        logger.debug(f"Selected TNP nuisances: {self.tnp_nuisance_names}")

        self.card_tool.addSystematic(name=self.corr_hist_name,
            processes=['wtau_samples', 'single_v_nonsig_samples'] if self.skipFromSignal else ['single_v_samples'],
            group="resumTNP",
            splitGroup={"resum": ".*", "pTModeling" : ".*", "theory": ".*"},
            systAxes=["vars"],
            passToFakes=self.propagate_to_fakes,
            systNameReplace=name_replace,
            preOp=lambda h: h[{self.syst_ax : [central_var, *self.tnp_nuisances]}],
            mirror=self.mirror_tnp,
            scale=scale,
            skipEntries=[{self.syst_ax : central_var},],
            rename=f"resumTNP",
            systNamePrepend=f"resumTNP_",
        )

    def add_nonpert_unc(self, model):
        if not self.resumUnc:
            return

        self.set_np_model(model)
        if self.np_model:
            self.add_gamma_np_uncertainties()
            self.add_uncorrelated_np_uncertainties()
        else:
            logger.warning("Will not add any nonperturbative uncertainty!")

    def set_np_model(self, model):
        if model in ["none", None]:
            self.np_model = None
            return

        if model not in TheoryHelper.valid_np_models:
            raise ValueError(f"Model choice {model} is not a supported model. Valid choices are {TheoryHelper.valid_np_models}")
    
        signal_samples = self.card_tool.procGroups['signal_samples']
        self.np_hist_name = self.corr_hist_name.replace("Corr", "FlavDepNP")
        self.np_hist = self.card_tool.getHistsForProcAndSyst(signal_samples[0], self.np_hist_name)

        self.scale_hist_name = self.corr_hist_name.replace("Corr", "PtDepScales")

        var_name = model.replace("binned_", "")

        if not any(var_name in x for x in self.np_hist.axes[self.syst_ax]):
            raise ValueError(f"NP model choice was '{model}' but did not find corresponding variations in the histogram")

        self.np_model = model

    def add_gamma_np_uncertainties(self):
        # Since "c_nu = 0.1 is the central value, it doesn't show up in the name"
        gamma_vals = list(filter(lambda x: x in self.corr_hist.axes[self.syst_ax], 
            ["c_nu-0.1-omega_nu0.5", "omega_nu0.5", "c_nu-0.25", "c_nu-0.25"]))

        if len(gamma_vals) != 2:
            raise ValueError(f"Failed to find consistent variation for gamma NP in hist {self.corr_hist_name}")

        gamma_nuisance_name = "scetlibNPgamma"

        var_vals = gamma_vals
        var_names = [f"{gamma_nuisance_name}Down", f"{gamma_nuisance_name}Up"]

        logger.debug(f"Adding gamma uncertainties from syst entries {gamma_vals}")


        self.card_tool.addSystematic(name=self.corr_hist_name,
            processes=['wtau_samples', 'single_v_nonsig_samples'] if self.skipFromSignal else ['single_v_samples'],
            passToFakes=self.propagate_to_fakes,
            systAxes=[self.syst_ax],
            preOp=lambda h: h[{self.syst_ax : var_vals}],
            outNames=var_names,
            group="resumNonpert",
            splitGroup={"resum": ".*", "pTModeling" : ".*", "theory": ".*"},
            rename="scetlibNP",
        )

    def add_resum_scale_uncertainty():
        obs = self.card_tool.fit_axes[:]
        if not obs:
            raise ValueError("Failed to find the observable names for the resummation uncertainties")

        theory_hist = self.card_tool.getHistsForProcAndSyst(self.samples[0], theory_hist_name)
        resumscale_nuisances = match_str_axis_entries(h.axes[syst_ax], ["^nuB.*", "nuS.*", "^muB.*", "^muS.*",])

        expanded_samples = card_tool.datagroups.getProcNames(self.samples)
        syst_ax = "vars"

        card_tool.addSystematic(name=theory_hist,
            processes=self.samples,
            group="resumScale",
            splitGroup={"resum": ".*", "pTModeling" : ".*", "theory": ".*"},
            passToFakes=to_fakes,
            skipEntries=[{syst_ax : x} for x in both_exclude+tnp_nuisances],
            systAxes=["downUpVar"], # Is added by the preOpMap
            preOpMap={s : lambda h: hh.syst_min_and_max_env_hist(h, obs, "vars", resumscale_nuisances) for s in expanded_samples},
            outNames=[f"scetlibResumScale{name_append}Up", f"scetlibResumScale{name_append}Down"],
            rename=f"resumScale{name_append}",
            systNamePrepend=f"resumScale{name_append}_",
        )
        #TODO: check if this is actually the proper treatment of these uncertainties
        card_tool.addSystematic(name=theory_hist,
            processes=self.samples,
            group="resumScale",
            splitGroup={"resum": ".*", "pTModeling" : ".*", "theory": ".*"},
            passToFakes=to_fakes,
            systAxes=["vars"],
            preOpMap={s : lambda h: h[{"vars" : ["kappaFO0.5-kappaf2.", "kappaFO2.-kappaf0.5", "mufdown", "mufup",]}] for s in expanded_samples},
            outNames=[f"scetlib_kappa{name_append}Up", f"scetlib_kappa{name_append}Down", f"scetlib_muF{name_append}Up", f"scetlib_muF{name_append}Down"],
            rename=f"resumFOScale{name_append}",
            systNamePrepend=f"resumScale{name_append}_",
        )

    def add_uncorrelated_np_uncertainties(self):
        np_map = {
            "Lambda2" : ["-0.25", "0.25",],
            "Delta_Lambda2" : ["-0.02", "0.02",],
            "Lambda4" : [".01", ".16"],
        } if "Lambda" in self.np_model else {
            "Omega" : ["0.", "0.8"],
            "Delta_Omega" : ["-0.02", "0.02"],
        }

        if "Delta" not in self.np_model:
            to_remove = list(filter(lambda x: "Delta" in x, np_map.keys()))
            for k in to_remove:
                np_map.pop(k)

        central_var = self.np_hist.axes[self.syst_ax][0]

        for label,vals in np_map.items():
            if not all(label+v in self.np_hist.axes[self.syst_ax] for v in vals):
                tmpvals = [x.replace(label, "") for x in self.np_hist.axes[self.syst_ax] if re.match(f"^{label}\d+", x)]
                if tmpvals:
                    logger.warning(f"Using variations {tmpvals} rather than default values {vals}!")
                    np_map[label] = tmpvals
                else:
                    raise ValueError(f"Failed to find all vars {vals} for var {label} in hist {self.np_hist_name}")


        binned = "binned" in self.np_model
        gen_axes = ["absYVgenNP", "chargeVgenNP"]
        sum_axes = [] if binned else ["absYVgenNP"]
        syst_axes = ["absYVgenNP", "chargeVgenNP", self.syst_ax] if binned else ["chargeVgenNP", self.syst_ax]
        operation=lambda h,entries: syst_tools.hist_to_variations(h[{self.syst_ax : [central_var, *entries]}], gen_axes=gen_axes, sum_axes=sum_axes)
        for sample_group in self.samples:
            if not self.card_tool.procGroups.get(sample_group, None):
                continue
            label = self.sample_label(sample_group)
            for nuisance,vals in np_map.items():
                entries = [nuisance+v for v in vals]
                rename = f"scetlibNP{label}{nuisance}"
                self.card_tool.addSystematic(name=self.np_hist_name,
                    processes=[sample_group],
                    group="resumNonpert",
                    splitGroup={"resum": ".*", "pTModeling" : ".*", "theory": ".*"},
                    systAxes=syst_axes,
                    passToFakes=self.propagate_to_fakes,
                    preOp=operation,
                    preOpArgs={"entries": entries},
                    # outNames=[f"{rename}Down", f"{rename}Up"] if not binned else None,
                    systNameReplace=[(entries[1], f"{rename}Up"), (entries[0], f"{rename}Down"), ],
                    skipEntries=[{self.syst_ax : central_var}],
                    rename=rename,
                )

    def add_pdf_uncertainty(self, operation=None, scale=-1.):
        pdf = input_tools.args_from_metadata(self.card_tool, "pdfs")[0]
        pdfInfo = theory_tools.pdf_info_map("ZmumuPostVFP", pdf)
        pdfName = pdfInfo["name"]
        scale = scale if scale != -1. else pdfInfo["inflationFactor"]
        pdf_hist = pdfName
        pdf_corr_hist = f"scetlib_dyturbo{pdf.upper().replace('AN3LO', 'an3lo')}VarsCorr" 
        symmetrize = "quadratic"

        if self.pdf_from_corr:
            theory_unc = input_tools.args_from_metadata(self.card_tool, "theoryCorr")
            if not theory_unc:
                logger.error("Can not add resummation uncertainties. No theory correction was applied!")
            if pdf_hist[:-4] not in theory_unc:
                logger.error(f"Did not find {pdf_hist} correction in file! Cannot use SCETlib+DYTurbo PDF uncertainties")
            pdf_hist = pdf_corr_hist

        logger.info(f"Using PDF hist {pdf_hist}, apply scaling of {scale}")

        pdf_ax = self.syst_ax if self.pdf_from_corr else "pdfVar"
        symHessian = pdfInfo["combine"] == "symHessian"
        pdf_args = dict(
            processes=['wtau_samples', 'single_v_nonsig_samples'] if self.skipFromSignal else ['single_v_samples'],
            mirror=True if symHessian else False,
            group=pdfName,
            splitGroup={f"{pdfName}NoAlphaS": '.*', "theory": ".*"},
            passToFakes=self.propagate_to_fakes,
            preOpMap=operation,
            scale=pdfInfo.get("scale", 1)*scale,
            symmetrize=symmetrize,
            systAxes=[pdf_ax],
        )
        if self.pdf_from_corr:
            self.card_tool.addSystematic(pdf_hist, 
                outNames=[""]+theory_tools.pdfNamesAsymHessian(pdfInfo['entries'], pdfset=pdfName)[1:],
                **pdf_args
            )
        else:
            self.card_tool.addSystematic(pdf_hist, 
                skipEntries=[{pdf_ax : "^pdf0[a-z]*"}],
                **pdf_args
            )
            if pdfName == 'pdfHERAPDF20':
                self.card_tool.addSystematic(pdf_hist+'ext',
                    skipEntries=[{pdf_ax : "^pdf0[a-z]*"}],
                    processes=['wtau_samples', 'single_v_nonsig_samples'] if self.skipFromSignal else ['single_v_samples'],
                    mirror=True,
                    group=pdfName,
                    splitGroup={f"{pdfName}NoAlphaS": '.*', "theory": ".*"},
                    passToFakes=self.propagate_to_fakes,
                    preOpMap=operation,
                    scale=pdfInfo.get("scale", 1)*scale,
                    symmetrize=symmetrize,
                    systAxes=[pdf_ax],
                )

        asRange = pdfInfo['alphasRange']
        asname = f"{pdfName}alphaS{asRange}" if not self.as_from_corr else pdf_corr_hist.replace("Vars", "_pdfas")
        as_replace = [("as", "pdfAlphaS")]+[("0116", "Down"), ("0120", "Up")] if asRange == "002" else [("0117", "Down"), ("0119", "Up")]
        as_args = dict(name=asname,
            processes=['wtau_samples', 'single_v_nonsig_samples'] if self.skipFromSignal else ['single_v_samples'],
            mirror=False,
            group=pdfName,
            splitGroup={f"{pdfName}AlphaS": '.*', "theory": ".*"},
            systAxes=["vars" if self.as_from_corr else "alphasVar"],
            scale=(0.75 if asRange == "002" else 1.5)*scale,
            symmetrize=symmetrize,
            passToFakes=self.propagate_to_fakes,
        )
        if self.as_from_corr:
            as_args["outNames"] = ['', "pdfAlphaSDown", "pdfAlphaSUp"]
        else:
            as_args["systNameReplace"] = as_replace
            as_args['skipEntries'] = [{"alphasVar" : "as0118"}]
        self.card_tool.addSystematic(**as_args)

    def add_transition_fo_scale_uncertainties(self, transition = True, scale=True):
        obs = self.card_tool.fit_axes[:]

        for sample_group in self.samples:
            if not self.card_tool.procGroups.get(sample_group, None):
                continue

            name_append = self.sample_label(sample_group)

            sel_vars = []
            outNames = []

            if transition:
                sel_vars.extend(["transition_points0.2_0.35_1.0", "transition_points0.2_0.75_1.0"])
                outNames.extend([f"resumTransition{name_append}Down", f"resumTransition{name_append}Up"])

            if scale:
                sel_vars.extend(["renorm_scale_pt20_envelope_Down", "renorm_scale_pt20_envelope_Up"])
                outNames.extend([f"resumFOScale{name_append}Down", f"resumFOScale{name_append}Up"])

            if not sel_vars:
                # nothing to do
                continue

            self.card_tool.addSystematic(name=self.corr_hist_name,
                processes=[sample_group],
                group="resumTransitionFOScale",
                splitGroup={"resum": ".*", "pTModeling" : ".*", "theory": ".*"},
                systAxes=["vars"],
                symmetrize = "quadratic",
                passToFakes=self.propagate_to_fakes,
                preOp = lambda h: h[{"vars" : sel_vars}],
                outNames=outNames,
                rename=f"resumTransitionFOScale{name_append}",
            )

    def add_quark_mass_vars(self, from_minnlo=True):
        pdfs = input_tools.args_from_metadata(self.card_tool, "pdfs")
        theory_corrs = input_tools.args_from_metadata(self.card_tool, "theoryCorr")

        from_minnlo = not ("scetlib_dyturboMSHT20mcrange" in theory_corrs and "scetlib_dyturboMSHT20mcrange" in theory_corrs and "msht20" in pdfs[0])

        if from_minnlo:
            if "msht20mbrange" in pdfs and "msht20mcrange" in pdfs and pdfs[0] not in ["msht20", "msht20mbrange", "msht20mcrange"]:
                raise ValueError("Using the mass variation sets from MiNNLO requires MSHT20 as the central set")
            elif "msht20mbrange_renorm" not in pdfs or "msht20mcrange_renorm" not in pdfs:
                raise ValueError("Must include the msht20mb(c)range pdf sets to take the mass variation from MiNNLO")
        elif not ("msht20" in pdfs[0] and "scetlib_dyturboMSHT20mbrange" in theory_corrs and "scetlib_dyturboMSHT20mcrange" in theory_corrs):
            raise ValueError("In order to take the mb(c) mass unc. from SCETlib+DYTurbo, you need to include those corr files and use MSHT20 as central PDF")
            
        bhist = "pdfMSHT20mbrange" if from_minnlo else "scetlib_dyturboMSHT20mbrangeCorr"
        syst_ax = "pdfVar" if from_minnlo else "vars"

        self.card_tool.addSystematic(name=bhist,
            processes=self.samples,
            systAxes=[syst_ax],
            symmetrize = "quadratic",
            group="bcQuarkMass",
            splitGroup={"pTModeling" : ".*", "theory": ".*"},
            passToFakes=self.propagate_to_fakes,
            outNames=["", "pdfMSHT20mbrangeDown",]+[""]*4+["pdfMSHT20mbrangeUp"],
        )
        
        self.card_tool.addSystematic(name=bhist.replace("brange", "crange"),
            processes=self.samples,
            systAxes=[syst_ax],
            symmetrize = "quadratic",
            group="bcQuarkMass",
            splitGroup={"pTModeling" : ".*", "theory": ".*"},
            passToFakes=self.propagate_to_fakes,
            outNames=["", "pdfMSHT20mcrangeDown",]+[""]*6+["pdfMSHT20mcrangeUp"],
        )


