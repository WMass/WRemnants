import itertools
import math
import os
import pickle
import re

import h5py
import hist
import lz4.frame
import numpy as np
import pandas as pd

import wums
from utilities.io_tools import input_tools
from utilities.styles import styles
from wremnants import histselections as sel
from wremnants.datasets.datagroup import Datagroup
from wums import boostHistHelpers as hh
from wums import logging

logger = logging.child_logger(__name__)


class Datagroups(object):
    mode_map = {
        "w_z_gen_dists.py": "vgen",
        "mz_dilepton.py": "z_dilepton",
        "mz_wlike_with_mu_eta_pt.py": "z_wlike",
        "mw_with_mu_eta_pt.py": "w_mass",
        "mw_lowPU.py": "w_lowpu",
        "mz_lowPU.py": "z_lowpu",
    }

    lumi_uncertainties = {
        "2016RreVFP": 1.012,
        "2016PostVFP": 1.012,
        "2017": 1.023,
        "2017G": 1.019,
        "2017H": 1.017,
        "2018": 1.025,
    }

    def __init__(self, infile, mode=None, xnorm=False, **kwargs):
        if infile.endswith(".pkl.lz4"):
            with lz4.frame.open(infile) as f:
                self.results = pickle.load(f)
        elif infile.endswith(".hdf5"):
            logger.info("Load input file")
            h5file = h5py.File(infile, "r")
            self.results = input_tools.load_results_h5py(h5file)
        else:
            raise ValueError(f"{infile} has unsupported file type")

        if mode is None:
            analysis_script = os.path.basename(self.getScriptCommand().split()[0])
            self.mode = Datagroups.analysisLabel(analysis_script)
        else:
            if mode not in Datagroups.mode_map.values():
                raise ValueError(
                    f"Unrecognized mode '{mode}.' Must be one of {set(Datagroups.mode_map.values())}"
                )
            self.mode = mode
        logger.info(f"Set mode to {self.mode}")

        args = self.getMetaInfo()["args"]
        self.flavor = args.get("flavor", None)
        self.era = args.get("era", None)
        self.lumi_uncertainty = Datagroups.lumi_uncertainties.get(self.era, None)

        self.groups = {}
        self.procGroups = {}  # groups of groups for convenent definition of systematics
        self.nominalName = "nominal"
        self.rebinOp = None
        self.rebinBeforeSelection = False
        self.globalAction = None
        self.unconstrainedProcesses = []
        self.fakeName = "Fake" + (f"_{self.flavor}" if self.flavor is not None else "")
        self.dataName = "Data"
        self.gen_axes = {}
        self.fit_axes = []
        self.fakerate_axes = ["pt", "eta", "charge"]

        self.setGenAxes()

        if "lowpu" in self.mode:
            from wremnants.datasets.datagroupsLowPU import (
                make_datagroups_lowPU as make_datagroups,
            )
        else:
            from wremnants.datasets.datagroups2016 import (
                make_datagroups_2016 as make_datagroups,
            )

        make_datagroups(self, **kwargs)

        self.lumi = sum([value.get("lumi", 0) for key, value in self.results.items()])
        if self.lumi > 0:
            logger.info(f"Integrated luminosity from data: {self.lumi}/fb")
        else:
            self.lumi = 1
            logger.warning(
                f"No data process was selected, normalizing MC to {self.lumi }/fb"
            )

        self.lumiScale = 1
        self.lumiScaleVarianceLinearly = []

        self.channel = "ch0"
        self.excludeSyst = None
        self.keepSyst = None
        self.absorbSyst = None
        self.explicitSyst = None
        self.customSystMapping = {}

        # if the histograms should be normalized to cross section (otherwise expected events)
        self.xnorm = xnorm

        self.writer = None

    def get_members_from_results(self, startswith=[], not_startswith=[], is_data=False):
        dsets = {
            k: v for k, v in self.results.items() if type(v) == dict and "dataset" in v
        }
        if is_data:
            dsets = {
                k: v for k, v in dsets.items() if v["dataset"].get("is_data", False)
            }
        else:
            dsets = {
                k: v for k, v in dsets.items() if not v["dataset"].get("is_data", False)
            }
        if type(startswith) == str:
            startswith = [startswith]
        if len(startswith) > 0:
            dsets = {
                k: v
                for k, v in dsets.items()
                if any([v["dataset"]["name"].startswith(x) for x in startswith])
            }
        if type(not_startswith) == str:
            not_startswith = [not_startswith]
        if len(not_startswith) > 0:
            dsets = {
                k: v
                for k, v in dsets.items()
                if not any([v["dataset"]["name"].startswith(x) for x in not_startswith])
            }
        return dsets

    def addGroup(self, name, **kwargs):
        group = Datagroup(name, **kwargs)
        self.groups[name] = group

    def deleteGroups(self, names):
        for n in names:
            self.deleteGroup(n)

    def deleteGroup(self, name):
        if name in self.groups.keys():
            del self.groups[name]
        else:
            logger.warning(f"Try to delete group '{name}' but did not find this group.")

    def copyGroup(self, group_name, new_name, member_filter=None):
        self.groups[new_name] = self.groups[group_name].copy(new_name, member_filter)

    def selectGroups(self, selections):
        new_groupnames = []
        for selection in selections:
            new_groupnames += list(
                filter(lambda x, s=selection: x == s, self.groups.keys())
            )

        # remove duplicates selected by multiple filters
        return list(set(new_groupnames))

    def mergeGroups(self, groups, new_name):
        groups_to_merge = []
        for g in groups:
            if g in self.groups:
                groups_to_merge.append(g)
            else:
                logger.warning(
                    f"Did not find group {g}. continue without merging it to new group {new_name}."
                )
        if len(groups_to_merge) < 1:
            logger.warning(f"No groups to be merged. continue without merging.")
            return
        if new_name != groups_to_merge[0]:
            self.copyGroup(groups_to_merge[0], new_name)
        self.groups[new_name].label = styles.process_labels.get(new_name, new_name)
        self.groups[new_name].color = styles.process_colors.get(new_name, "grey")
        for group in groups_to_merge[1:]:
            self.groups[new_name].addMembers(
                self.groups[group].members,
                member_operations=self.groups[group].memberOp,
            )
        self.deleteGroups([g for g in groups_to_merge if g != new_name])

    def filterGroups(self, filters):
        if filters is None:
            return

        if isinstance(filters, str):
            filters = [filters]

        if isinstance(filters, list):
            new_groupnames = self.selectGroups(filters)
        else:
            new_groupnames = list(filter(filters, self.groups.keys()))

        diff = list(self.groups.keys() - set(new_groupnames))
        if diff:
            logger.info(
                f"Datagroups.filterGroups : filtered out following groups: {diff}"
            )

        self.groups = {key: self.groups[key] for key in new_groupnames}

        if len(self.groups) == 0:
            logger.warning(
                f"Filtered groups using '{filters}' but didn't find any match. Continue without any group."
            )

    def excludeGroups(self, excludes):
        if excludes is None:
            return

        if isinstance(excludes, str):
            excludes = [excludes]

        if isinstance(excludes, list):
            new_groupnames = list(
                filter(lambda x: x not in self.selectGroups(excludes), self.groups)
            )
        else:
            new_groupnames = list(filter(excludes, self.groups.keys()))

        diff = list(self.groups.keys() - set(new_groupnames))
        if diff:
            logger.info(
                f"Datagroups.excludeGroups: filtered out following groups: {diff}"
            )

        self.groups = {key: self.groups[key] for key in new_groupnames}

        if len(self.groups) == 0:
            logger.warning(
                f"Excluded all groups using '{excludes}'. Continue without any group."
            )

    def set_histselectors(
        self,
        group_names,
        histToRead="nominal",
        fake_processes=None,
        mode="extended1D",
        smoothing_mode="full",
        smoothingOrderFakerate=3,
        smoothingOrderSpectrum=3,
        smoothingPolynomialSpectrum="power",
        integrate_shapecorrection_x=True,  # integrate the abcd x-axis or not, only relevant for extended2D
        simultaneousABCD=False,
        forceGlobalScaleFakes=None,
        mcCorr=["pt", "eta"],
        abcdExplicitAxisEdges={},
        **kwargs,
    ):
        logger.info(f"Set histselector")
        if self.mode[0] != "w":
            return  # histselectors only implemented for single lepton (with fakes)
        auxiliary_info = {"ABCDmode": mode}
        signalselector = sel.SignalSelectorABCD
        scale = 1
        if mode == "extended1D":
            scale = 0.85
            fakeselector = sel.FakeSelector1DExtendedABCD
        elif mode == "extended2D":
            scale = 1.15
            fakeselector = sel.FakeSelector2DExtendedABCD
            auxiliary_info["integrate_shapecorrection_x"] = integrate_shapecorrection_x
            if smoothing_mode == "fakerate" and not integrate_shapecorrection_x:
                auxiliary_info.update(
                    dict(
                        smooth_shapecorrection=True,
                        interpolate_x=True,
                        rebin_x="automatic",
                    )
                )
            else:
                auxiliary_info.update(
                    dict(
                        smooth_shapecorrection=False, interpolate_x=False, rebin_x=None
                    )
                )
        elif mode == "extrapolate":
            fakeselector = sel.FakeSelectorExtrapolateABCD
        elif mode == "simple":
            scale = 0.85
            if simultaneousABCD:
                fakeselector = sel.FakeSelectorSimultaneousABCD
            else:
                fakeselector = sel.FakeSelectorSimpleABCD
        elif mode == "mc":
            pass
        else:
            raise RuntimeError(f"Unknown mode {mode} for fakerate estimation")
        if forceGlobalScaleFakes is not None:
            scale = forceGlobalScaleFakes
        fake_processes = [self.fakeName] if fake_processes is None else fake_processes
        for i, g in enumerate(group_names):
            members = self.groups[g].members[:]
            if len(members) == 0:
                raise RuntimeError(f"No member found for group {g}")
            base_member = members[0].name
            h = self.results[base_member]["output"][histToRead].get()
            if g in fake_processes and mode.lower() != "mc":
                self.groups[g].histselector = fakeselector(
                    h,
                    global_scalefactor=scale,
                    fakerate_axes=self.fakerate_axes,
                    smoothing_mode=smoothing_mode,
                    smoothing_order_fakerate=smoothingOrderFakerate,
                    smoothing_order_spectrum=smoothingOrderSpectrum,
                    smoothing_polynomial_spectrum=smoothingPolynomialSpectrum,
                    abcdExplicitAxisEdges=abcdExplicitAxisEdges,
                    **auxiliary_info,
                    **kwargs,
                )
                if (
                    mode in ["simple", "extended1D", "extended2D"]
                    and forceGlobalScaleFakes is None
                    and (len(mcCorr) == 0 or mcCorr[0] not in ["none", None])
                ):
                    # set QCD MC nonclosure corrections
                    histname_qcd_mc = f"QCDmuEnrichPt15_{self.era}"
                    if histname_qcd_mc not in self.results:
                        logger.warning(
                            f"Dataset '{histname_qcd_mc}' not in results, continue without fake correction"
                        )
                        return
                    if "unweighted" not in self.results[histname_qcd_mc]["output"]:
                        logger.warning(
                            "Histogram 'unweighted' not found, continue without fake correction"
                        )
                        return
                    hQCD = self.results[histname_qcd_mc]["output"]["unweighted"].get()
                    self.groups[g].histselector.set_correction(hQCD, axes_names=mcCorr)
            else:
                self.groups[g].histselector = signalselector(
                    h, fakerate_axes=self.fakerate_axes, **auxiliary_info, **kwargs
                )

    def setGlobalAction(self, action):
        # To be used for applying a selection, rebinning, etc.
        if self.globalAction is None:
            self.globalAction = action
        else:
            self.globalAction = lambda h, old_action=self.globalAction: action(
                old_action(h)
            )

    def setMemberOp(self, group_name, ops):
        group = self.groups[group_name]
        if not isinstance(ops, list):
            ops = [ops for i in range(len(group.members))]
        # To be used for applying a selection, rebinning, etc.
        if group.memberOp is None:
            group.memberOp = ops
        else:
            group.memberOp = [
                lambda h, old=old: op(old(h)) for op, old in zip(ops, group.memberOp)
            ]

    def setRebinOp(self, action):
        # To be used for applying a selection, rebinning, etc.
        if self.rebinOp is None:
            self.rebinOp = action
        else:
            self.rebinOp = lambda h, old_action=self.rebinOp: action(old_action(h))

    def setNominalName(self, name):
        self.nominalName = name

    def processScaleFactor(self, proc):
        if proc.is_data or proc.xsec is None:
            return 1
        scale = proc.xsec / proc.weight_sum
        if not self.xnorm:
            scale *= self.lumi * 1000
        return scale

    def getMetaInfo(self):
        if "meta_info" not in self.results and "meta_data" not in self.results:
            raise ValueError("Did not find meta data in results file")
        return (
            self.results["meta_info"]
            if "meta_info" in self.results
            else self.results["meta_data"]
        )

    def args_from_metadata(self, arg):
        meta_data = self.getMetaInfo()
        if "args" not in meta_data.keys():
            raise IOError(
                f"The argument {arg} was not found in the metadata, maybe you run on an obsolete file."
            )
        elif arg not in meta_data["args"].keys():
            raise IOError(
                f"Did not find the argument {arg} in the meta_data dict. Maybe it is an outdated option"
            )

        return meta_data["args"][arg]

    def getScriptCommand(self):
        meta_info = self.getMetaInfo()
        return meta_info["command"]

    # remove a histogram that is loaded into memory from a proxy object
    def release_results(self, histname):
        for result in self.results.values():
            if "output" not in result:
                continue
            res = result["output"]
            if histname in res:
                res[histname].release()

    # for reading pickle files
    # as a reminder, the ND hists with tensor axes in the pickle files are organized as
    # pickle[procName]["output"][baseName] where
    ## procName are grouped into datagroups
    ## baseName takes values such as "nominal"
    def loadHistsForDatagroups(
        self,
        baseName,
        syst,
        procsToRead=None,
        label=None,
        nominalIfMissing=True,
        applySelection=True,
        forceNonzero=False,
        preOpMap=None,
        preOpArgs={},
        excludeProcs=None,
        forceToNominal=[],
        sumFakesPartial=True,
    ):
        logger.debug("Calling loadHistsForDatagroups()")
        logger.debug(f"The basename and syst is: {baseName}, {syst}")
        logger.debug(
            f"The procsToRead and excludedProcs are: {procsToRead}, {excludeProcs}"
        )
        if not label:
            label = syst if syst else baseName
        # this line is annoying for the theory agnostic, too many processes for signal
        logger.debug(
            f"In loadHistsForDatagroups(): for hist {syst} procsToRead = {procsToRead}"
        )

        if not procsToRead:
            if excludeProcs:
                procsToRead = list(
                    filter(lambda x: x not in excludeProcs, self.groups.keys())
                )
            else:
                procsToRead = list(self.groups.keys())

        foundExact = False

        # If fakes are present do them as last group, and when running on prompt group build the sum to be used for the fakes.
        # This makes the code faster and avoid possible bugs related to reading again the same processes
        # NOTE:
        # To speed up even more, one could directly use the per-group sum already computed for each group,
        # but this would need to assume that fakes effectively had all the single processes in each group as members
        # (usually it will be the case, but it is more difficult to handle in a fully general way and without bugs)
        histForFake = (
            None  # to store the data-MC sums used for the fakes, for each syst
        )
        if sumFakesPartial and self.fakeName in procsToRead:
            procsToReadSort = [x for x in procsToRead if x != self.fakeName] + [
                self.fakeName
            ]
            hasFake = True
            fakesMembers = [m.name for m in self.groups[self.fakeName].members]
            fakesMembersWithSyst = []
            logger.debug(f"Has fake members: {fakesMembers}")
        else:
            hasFake = False
            procsToReadSort = [x for x in procsToRead]
        # Note: if 'hasFake' is kept as False (but Fake exists), the original behaviour for which Fake reads everything again is restored
        for procName in procsToReadSort:
            logger.debug(f"Reading group {procName}")

            if procName not in self.groups.keys():
                raise RuntimeError(
                    f"Group {procName} not known. Defined groups are {list(self.groups.keys())}."
                )
            group = self.groups[procName]

            group.hists[label] = None

            for i, member in enumerate(group.members):
                if (
                    sumFakesPartial
                    and procName == self.fakeName
                    and member.name in fakesMembersWithSyst
                ):
                    # if we are here this process has been already used to build the fakes when running for other groups
                    continue
                logger.debug(f"Looking at group member {member.name}")
                read_syst = syst
                if member.name in forceToNominal:
                    read_syst = ""
                    logger.debug(
                        f"Forcing group member {member.name} to read the nominal hist for syst {syst}"
                    )
                try:
                    h = self.readHist(baseName, member, read_syst)
                    foundExact = True
                except ValueError as e:
                    if nominalIfMissing:
                        logger.info(
                            f"{str(e)}. Using nominal hist {self.nominalName} instead"
                        )
                        h = self.readHist(self.nominalName, member, "")
                    else:
                        logger.warning(str(e))
                        continue

                h_id = id(h)
                logger.debug(f"Hist axes are {h.axes.name}")

                if group.memberOp:
                    if group.memberOp[i] is not None:
                        logger.debug(
                            f"Apply operation to member {i}: {member.name}/{procName}"
                        )
                        h = group.memberOp[i](h)
                    else:
                        logger.debug(
                            f"No operation for member {i}: {member.name}/{procName}"
                        )

                if preOpMap and member.name in preOpMap:
                    logger.debug(
                        f"Applying preOp to {member.name}/{procName} after loading"
                    )
                    h = preOpMap[member.name](h, **preOpArgs)

                sum_axes = [x for x in self.sum_gen_axes if x in h.axes.name]
                if len(sum_axes) > 0:
                    # sum over remaining axes (avoid integrating over fit axes & fakerate axes)
                    logger.debug(f"Sum over axes {sum_axes}")
                    h = h.project(*[x for x in h.axes.name if x not in sum_axes])
                    logger.debug(f"Hist axes are now {h.axes.name}")

                if h_id == id(h):
                    logger.debug(f"Make explicit copy")
                    h = h.copy()

                if self.globalAction:
                    logger.debug("Applying global action")
                    h = self.globalAction(h)

                if forceNonzero:
                    logger.debug("force non zero")
                    h = hh.clipNegativeVals(h, createNew=False)

                scale = self.processScaleFactor(member)
                if group.scale:
                    scale *= group.scale(member)

                # When scaling yields by a luminosity factor, select whether to scale the variance linearly (e.g. for extrapolation studies) or quadratically (default).
                if not np.isclose(self.lumiScale, 1, rtol=0, atol=1e-6) and (
                    (
                        procName == self.dataName
                        and "data" in self.lumiScaleVarianceLinearly
                    )
                    or (
                        procName != self.dataName
                        and "mc" in self.lumiScaleVarianceLinearly
                    )
                ):
                    logger.debug(
                        f"Scale {procName} hist by {self.lumiScale} as a multiplicative luminosity factor, with variance scaled linearly"
                    )
                    h = hh.scaleHist(
                        h, self.lumiScale, createNew=False, scaleVarianceLinearly=True
                    )
                else:
                    scale *= self.lumiScale

                if not np.isclose(scale, 1, rtol=0, atol=1e-10):
                    logger.debug(f"Scale hist with {scale}")
                    h = hh.scaleHist(h, scale, createNew=False)

                hasPartialSumForFake = False
                if hasFake and procName != self.fakeName:
                    if member.name in fakesMembers:
                        logger.debug("Make partial sums for fakes")
                        if member.name not in fakesMembersWithSyst:
                            fakesMembersWithSyst.append(member.name)
                        hasPartialSumForFake = True
                        # apply the correct scale for fakes
                        scaleProcForFake = self.groups[self.fakeName].scale(member)
                        logger.debug(
                            f"Summing hist {read_syst} for {member.name} to {self.fakeName} with scale = {scaleProcForFake}"
                        )
                        hProcForFake = scaleProcForFake * h
                        histForFake = (
                            hh.addHists(histForFake, hProcForFake, createNew=False)
                            if histForFake
                            else hProcForFake
                        )

                # The following must be done when the group is not Fake, or when the previous part for fakes was not done
                # For fake this essentially happens when the process doesn't have the syst, so that the nominal is used
                if procName != self.fakeName or (
                    procName == self.fakeName and not hasPartialSumForFake
                ):
                    if procName == self.fakeName:
                        logger.debug(
                            f"Summing nominal hist instead of {syst} to {self.fakeName} for {member.name}"
                        )
                    else:
                        logger.debug(
                            f"Summing {read_syst} to {procName} for {member.name}"
                        )

                    group.hists[label] = (
                        hh.addHists(group.hists[label], h, createNew=False)
                        if group.hists[label]
                        else h
                    )

            if not nominalIfMissing and group.hists[label] is None:
                continue

            # now sum to fakes the partial sums which where not already done before
            # (group.hists[label] contains only the contribution from nominal histograms).
            # Then continue with the rest of the code as usual
            if hasFake and procName == self.fakeName:
                if histForFake is not None:
                    group.hists[label] = (
                        hh.addHists(group.hists[label], histForFake, createNew=False)
                        if group.hists[label]
                        else histForFake
                    )

            if self.rebinOp and self.rebinBeforeSelection:
                logger.debug(f"Apply rebin operation for process {procName}")
                group.hists[label] = self.rebinOp(group.hists[label])

            if group.histselector is not None:
                if not applySelection:
                    logger.warning(
                        f"Selection requested for process {procName} but applySelection=False, thus it will be ignored"
                    )
                elif label in group.hists.keys() and group.hists[label] is not None:
                    group.hists[label] = group.histselector.get_hist(
                        group.hists[label], is_nominal=(label == self.nominalName)
                    )
                else:
                    raise RuntimeError("Failed to apply selection")

            if self.rebinOp and not self.rebinBeforeSelection:
                logger.debug(f"Apply rebin operation for process {procName}")
                group.hists[label] = self.rebinOp(group.hists[label])

        # Avoid situation where the nominal is read for all processes for this syst
        if nominalIfMissing and not foundExact:
            raise ValueError(f"Did not find systematic {syst} for any processes!")

    def getNames(self, matches=[], exclude=False, match_exact=False):
        # This method returns the names from the defined groups, unless one selects further.
        listOfNames = list(x for x in self.groups.keys())
        if not matches:
            return listOfNames
        else:
            # matches uses regular expressions with search (and can be inverted when exclude is true),
            # thus a string will match if the process name contains that string anywhere inside it
            if exclude:
                return list(
                    filter(
                        lambda x: all([re.search(expr, x) is None for expr in matches]),
                        listOfNames,
                    )
                )
            elif match_exact:
                return [x for x in listOfNames if x in matches]
            else:
                return list(
                    filter(
                        lambda x: any([re.search(expr, x) for expr in matches]),
                        listOfNames,
                    )
                )

    def getProcNames(self, to_expand=[], exclude_group=[]):
        procs = []
        if not to_expand:
            to_expand = self.groups.keys()
        for group_name in to_expand:
            if group_name not in self.groups:
                raise ValueError(
                    f"Trying to expand unknown group {group_name}. Valid groups are {list(self.groups.keys())}"
                )
            if group_name not in exclude_group:
                for member in self.groups[group_name].members:
                    # protection against duplicates in the output list, they may arise from fakes
                    if member.name not in procs:
                        procs.append(member.name)
        return procs

    def sortByYields(self, histName, nominalName="nominal"):
        def get_sum(h):
            return h.sum() if not hasattr(h.sum(), "value") else h.sum().value

        self.groups = dict(
            sorted(
                self.groups.items(),
                key=lambda x: (
                    get_sum(
                        x[1].hists[histName if histName in x[1].hists else nominalName]
                    )
                    if nominalName in x[1].hists or histName in x[1].hists
                    else 0
                ),
                reverse=True,
            )
        )

    def getDatagroupsForHist(self, histName):
        filled = {}
        for k, v in self.groups.items():
            if histName in v:
                filled[k] = v
        return filled

    def resultsDict(self):
        return self.results

    def addSummedProc(
        self,
        refname,
        name,
        label=None,
        color=None,
        exclude=["Data"],
        relabel=None,
        procsToRead=None,
        reload=False,
        rename=None,
        action=None,
        actionArgs={},
        actionRequiresRef=False,
        **kwargs,
    ):
        if reload:
            self.loadHistsForDatagroups(
                refname,
                syst=name,
                label=rename,
                excludeProcs=exclude,
                procsToRead=procsToRead,
                **kwargs,
            )

        if not rename:
            rename = name

        self.addGroup(
            rename,
            label=label,
            color=color,
            members=[],
        )
        tosum = []
        procs = procsToRead if procsToRead else self.groups.keys()
        for proc in filter(lambda x: x not in exclude + [rename], procs):
            h = self.groups[proc].hists[rename if reload else name]
            if not h:
                raise ValueError(
                    f"Failed to find hist for proc {proc}, histname {name}"
                )
            if action:
                logger.debug(f"Applying action in addSummedProc! Before sum {h.sum()}")
                if actionRequiresRef:
                    actionArgs["hnom"] = self.groups[proc].hists[refname]
                h = action(h, **actionArgs)
                logger.debug(f"After action sum {h.sum()}")
            tosum.append(h)
        histname = refname if not relabel else relabel
        self.groups[rename].hists[histname] = hh.sumHists(tosum)

    def setSelectOp(self, op, processes=None):
        if processes is None:
            procs = self.groups
        else:
            procs = [processes] if isinstance(processes, str) else processes

        for proc in procs:
            if proc not in self.groups.keys():
                raise ValueError(f"In setSelectOp(): process {proc} not found")
            self.groups[proc].selectOp = op

    def setGenAxes(
        self,
        gen_axes_names=None,
        sum_gen_axes=None,
        base_group=None,
        histToReadAxes="xnorm",
    ):
        # gen_axes_names are the axes names to be recognized as gen axes, e.g. for the unfolding
        # sum_gen_axes are all gen axes names that are potentially in the produced histogram and integrated over if not used
        if isinstance(gen_axes_names, str):
            gen_axes_names = [gen_axes_names]
        if isinstance(sum_gen_axes, str):
            sum_gen_axes = [sum_gen_axes]

        # infer all gen axes from metadata
        try:
            args = self.getMetaInfo()["args"]
        except ValueError:
            logger.warning("No meta data found so no gen axes could be auto set")
            return

        self.all_gen_axes = args.get("unfoldingAxes", [])
        self.all_gen_axes = [n for n in self.all_gen_axes]

        self.gen_axes_names = (
            list(gen_axes_names) if gen_axes_names != None else self.all_gen_axes
        )
        self.sum_gen_axes = (
            list(sum_gen_axes) if sum_gen_axes != None else self.all_gen_axes
        )

        logger.debug(f"Gen axes names are now {self.gen_axes_names}")

        # set actual hist axes objects to be stored in metadata for post processing/plots/...
        for group_name, group in self.groups.items():
            if group_name != base_group:
                continue
            unfolding_hist = self.getHistForUnfolding(
                group_name,
                member_filter=lambda x: not x.name.endswith("OOA"),
                histToReadAxes=histToReadAxes,
            )
            if unfolding_hist is None:
                continue
            self.gen_axes[group_name[0]] = [
                ax for ax in unfolding_hist.axes if ax.name in self.gen_axes_names
            ]

        logger.debug(f"New gen axes are: {self.gen_axes}")

    def getGenBinIndices(self, axes=None):
        gen_bins = []
        for axis in axes:
            gen_bin_list = [i for i in range(axis.size)]
            if axis.traits.underflow:
                gen_bin_list.append(hist.underflow)
            if axis.traits.overflow:
                gen_bin_list.append(hist.overflow)
            gen_bins.append(gen_bin_list)
        return gen_bins

    def getHistForUnfolding(
        self, group_name, member_filter=None, histToReadAxes="xnorm"
    ):
        if group_name not in self.groups.keys():
            raise RuntimeError(
                f"Base group {group_name} not found in groups {self.groups.keys()}!"
            )
        base_members = self.groups[group_name].members[:]
        if member_filter is not None:
            base_member_idx = [
                i for i, m in enumerate(base_members) if member_filter(m)
            ][0]
        else:
            base_member_idx = 0

        base_member = base_members[base_member_idx]

        if histToReadAxes not in self.results[base_member.name]["output"]:
            logger.warning(
                f"Results for member {base_member.name} does not include histogram {histToReadAxes}. Found {self.results[base_member.name]['output'].keys()}"
            )
            return None
        nominal_hist = self.results[base_member.name]["output"][histToReadAxes].get()

        if self.groups[group_name].memberOp is not None:
            base_member_op = self.groups[group_name].memberOp[base_member_idx]
            return base_member_op(nominal_hist)
        else:
            return nominal_hist

    def getPOINames(self, gen_bin_indices, axes_names, base_name, flow=True):
        poi_names = []
        for indices in itertools.product(*gen_bin_indices):
            poi_name = base_name
            for idx, var in zip(indices, axes_names):
                if idx in [hist.overflow, hist.underflow] and not flow:
                    break
                elif idx == hist.underflow:
                    idx_str = "U"
                elif idx == hist.overflow:
                    idx_str = "O"
                else:
                    idx_str = str(idx)
                poi_name += f"_{var}{idx_str}"
            else:
                poi_names.append(poi_name)

        return poi_names

    def defineSignalBinsUnfolding(
        self,
        group_name,
        new_name=None,
        member_filter=None,
        histToReadAxes="xnorm",
        axesNamesToRead=None,
        fitvar=[],
        disable_flow_fit_axes=True,
    ):
        nominal_hist = self.getHistForUnfolding(
            group_name, member_filter, histToReadAxes
        )
        if axesNamesToRead is None:
            axesNamesToRead = self.gen_axes_names

        axesToRead = [nominal_hist.axes[n] for n in axesNamesToRead]

        # if a gen var and fit var are the same we have to extand the axis before splitting into gen bin contributions
        expand_vars = [x for x in axesNamesToRead if x in fitvar]
        if len(expand_vars):
            expand_vars_rename = [f"{x}_2" for x in expand_vars]
            expandOp = lambda h, vars_exp=expand_vars, vars_exp_rename=expand_vars_rename: hh.expand_hist_by_duplicate_axes(
                h, vars_exp, vars_exp_rename
            )
            self.setMemberOp(group_name, expandOp)
        else:
            expand_vars_rename = axesNamesToRead

        if disable_flow_fit_axes:
            # turn off flow for axes that are fit and used to define new groups, otherwise groups with empty histogram for the flow bins would be added
            for a in expand_vars:
                idx = axesNamesToRead.index(a)
                ax = axesToRead[idx]
                axesToRead[idx] = hh.disableAxisFlow(ax)

        self.gen_axes[new_name] = axesToRead
        logger.debug(f"New gen axes are: {self.gen_axes}")

        gen_bin_indices = self.getGenBinIndices(axesToRead)

        for indices, proc_name in zip(
            itertools.product(*gen_bin_indices),
            self.getPOINames(
                gen_bin_indices,
                axesNamesToRead,
                base_name=group_name if new_name is None else new_name,
            ),
        ):
            logger.debug(f"Now at {proc_name} with indices {indices}")
            self.copyGroup(group_name, proc_name, member_filter=member_filter)

            memberOp = lambda h, indices=indices, genvars=expand_vars_rename: h[
                {var: i for var, i in zip(genvars, indices)}
            ]
            self.setMemberOp(proc_name, memberOp)
            self.unconstrainedProcesses.append(proc_name)

    def select_xnorm_groups(self, select_groups=None, base_name="xnorm"):
        # only keep members and groups where xnorm is defined
        logger.info(
            "Select xnorm groups" + (f" {select_groups}" if select_groups else "")
        )
        if select_groups is not None:
            if isinstance(select_groups, str):
                select_groups = [select_groups]
            self.deleteGroups([g for g in self.groups.keys() if g not in select_groups])
        elif self.fakeName in self.groups:
            self.deleteGroup(self.fakeName)
        toDel_groups = []
        for g_name, group in self.groups.items():
            toDel_members = []
            for member in group.members:
                if member.name not in self.results.keys():
                    raise RuntimeError(
                        f"The member {member.name} of group {g_name} was not found in the results!"
                    )
                if base_name not in self.results[member.name]["output"].keys():
                    logger.debug(
                        f"Member {member.name} has no {base_name} and will be deleted"
                    )
                    toDel_members.append(member)
            if len(toDel_members) == len(group.members):
                logger.warning(
                    f"All members of group {g_name} have no xnorm and the group will be deleted"
                )
                toDel_groups.append(g_name)
            else:
                group.deleteMembers(toDel_members)
        self.deleteGroups(toDel_groups)

    def make_yields_df(self, histName, procs, action=lambda x: x, norm_proc=None):
        def sum_and_unc(h):
            if not hasattr(h.sum(), "value"):
                return (h.sum(), None)
            else:
                return (h.sum().value, math.sqrt(h.sum().variance))

        df = pd.DataFrame(
            [
                (k, *sum_and_unc(action(v.hists[histName])))
                for k, v in self.groups.items()
                if k in procs
            ],
            columns=["Process", "Yield", "Uncertainty"],
        )

        if norm_proc and norm_proc in self.groups:
            hist = action(self.groups[norm_proc].hists[histName])
            denom = hist.sum() if not hasattr(hist.sum(), "value") else hist.sum().value
            df[f"Ratio to {norm_proc} (%)"] = df["Yield"] / denom * 100

        return df

    def set_rebin_action(
        self,
        axes,
        ax_lim=[],
        ax_rebin=[],
        ax_absval=[],
        rebin_before_selection=False,
        rename=True,
    ):
        if len(ax_lim):
            if not all(x.real == 0 or x.imag == 0 for x in ax_lim):
                raise ValueError(
                    "In set_rebin_action(): ax_lim only accepts pure real or imaginary numbers"
                )
            if any(x.imag == 0 and (x.real % 1) != 0.0 for x in ax_lim):
                raise ValueError(
                    "In set_rebin_action(): ax_lim requires real numbers to be of integer type"
                )

        self.rebinBeforeSelection = rebin_before_selection

        for a in hh.get_rebin_actions(
            axes,
            ax_lim=ax_lim,
            ax_rebin=ax_rebin,
            ax_absval=ax_absval,
            rename=rename,
        ):
            self.setRebinOp(a)

    def readHist(self, baseName, proc, syst):
        output = self.results[proc.name]["output"]
        histname = self.histName(baseName, proc.name, syst)
        logger.debug(
            f"Reading hist {histname} for proc/group {proc.name} and syst '{syst}'"
        )
        if histname not in output:
            raise ValueError(f"Histogram {histname} not found for process {proc.name}")

        h = output[histname]
        if isinstance(h, wums.ioutils.H5PickleProxy):
            h = h.get()

        return h

    def addProcessGroup(self, name, startsWith=[], excludeMatch=[]):
        procFilter = lambda x: (
            any(x.startswith(init) for init in startsWith) or len(startsWith) == 0
        ) and all(excl not in x for excl in excludeMatch)

        self.procGroups[name] = self.filteredProcesses(procFilter)
        if not self.procGroups[name]:
            logger.warning(
                f"Did not match any processes to filter for group {name}! Valid procs are {self.groups.keys()}"
            )

    def expandProcesses(self, processes):
        if type(processes) == str:
            processes = [processes]

        return [x for y in processes for x in self.expandProcess(y)]

    def expandProcess(self, process):
        return self.procGroups.get(process, [process])

    def getProcGroupNames(self, grouped_procs):
        expanded_procs = []
        for group in grouped_procs:
            procs = self.expandProcess(group)
            for ungrouped in procs:
                expanded_procs.extend(self.getProcNames([ungrouped]))

        return expanded_procs

    def isData(self, procName, onlyData=False):
        if onlyData:
            return all([x.is_data for x in self.groups[procName].members])
        else:
            return any([x.is_data for x in self.groups[procName].members])

    def isMC(self, procName):
        return not self.isData(procName)

    def getProcesses(self):
        return list(self.groups.keys())

    def filteredProcesses(self, filterExpr):
        return list(filter(filterExpr, self.groups.keys()))

    def allMCProcesses(self):
        return self.filteredProcesses(lambda x: self.isMC(x))

    def predictedProcesses(self):
        return self.filteredProcesses(lambda x: x != self.dataName)

    def histName(self, baseName, procName="", syst=""):
        return Datagroups.histName(
            baseName, procName, syst, nominalName=self.nominalName
        )

    ## Functions to customize systs to be added in card, mainly for tests
    def setCustomSystForCard(self, exclude=None, keep=None, absorb=None, explicit=None):
        for regex, name in zip(
            (keep, exclude, absorb, explicit),
            ("keepSyst", "excludeSyst", "absorbSyst", "explicitSyst"),
        ):
            if regex in self.customSystMapping:
                regex = self.customSystMapping[regex]
            if regex:
                setattr(self, name, re.compile(regex))

    def setCustomSystGroupMapping(self, mapping):
        self.customSystMapping = mapping

    def isExcludedNuisance(self, name):
        # note, re.match search for a match from the beginning, so if x="test" x.match("mytestPDF1") will NOT match
        # might use re.search instead to be able to match from anywhere inside the name
        if self.excludeSyst != None and self.excludeSyst.match(name):
            if self.keepSyst != None and self.keepSyst.match(name):
                return False
            else:
                logger.info(f"   Excluding nuisance: {name}")
                return True
        else:
            return False

    def isAbsorbedNuisance(self, name):
        # note, re.match search for a match from the beginning, so if x="test" x.match("mytestPDF1") will NOT match
        # might use re.search instead to be able to match from anywhere inside the name
        if self.absorbSyst != None and self.absorbSyst.match(name):
            if self.explicitSyst != None and self.explicitSyst.match(name):
                return False
            else:
                logger.info(f"   Absorbing nuisance in covariance: {name}")
                return True
        else:
            return False

    # Read a specific hist, useful if you need to check info about the file
    def getHistsForProcAndSyst(self, proc, syst, nominal_name=None, **kwargs):
        if nominal_name is None:
            nominal_name = self.nominalName
        self.loadHistsForDatagroups(
            baseName=nominal_name, syst=syst, label="syst", procsToRead=[proc], **kwargs
        )
        return self.groups[proc].hists["syst"]

    def addNominalHistograms(
        self,
        forceNonzero=False,
        real_data=False,
        exclude_bin_by_bin_stat=None,
        bin_by_bin_stat_scale=1.0,
        fitresult_data=None,
        masked=False,
        masked_flow_axes=[],
    ):
        if self.writer is None:
            raise RuntimeError("Writer must be defined to add nominal histograms")

        self.loadHistsForDatagroups(
            baseName=self.nominalName,
            syst=self.nominalName,
            procsToRead=self.groups.keys(),
            label=self.nominalName,
            forceNonzero=forceNonzero,
            sumFakesPartial=True,
        )

        for i, proc in enumerate(self.predictedProcesses()):
            logger.info(f"Add process {proc} in channel {self.channel}")

            # nominal histograms of prediction
            norm_proc_hist = self.groups[proc].hists[self.nominalName]

            if norm_proc_hist.axes.name != self.fit_axes:
                norm_proc_hist = norm_proc_hist.project(*self.fit_axes)

            if (
                exclude_bin_by_bin_stat is not None
                and proc in self.procGroups[exclude_bin_by_bin_stat]
            ):
                norm_proc_hist.variances(flow=True)[...] *= 0
            elif bin_by_bin_stat_scale != 1:
                norm_proc_hist.variances(flow=True)[...] = (
                    norm_proc_hist.variances(flow=True) * bin_by_bin_stat_scale**2
                )

            if len(masked_flow_axes) > 0:
                self.axes_disable_flow = [
                    n
                    for n in norm_proc_hist.axes.name
                    if n not in masked_flow_axes and n != "helicitySig"
                ]
                norm_proc_hist = hh.disableFlow(norm_proc_hist, self.axes_disable_flow)

            if self.channel not in self.writer.channels:
                self.writer.add_channel(
                    axes=norm_proc_hist.axes,
                    name=self.channel,
                    masked=masked,
                    flow=len(masked_flow_axes) > 0,
                )

            self.writer.add_process(
                norm_proc_hist,
                proc,
                self.channel,
                signal=proc in self.unconstrainedProcesses,
            )

        # add metadata to channel info
        self.writer.channels[self.channel].update(
            {
                "era": self.era,
                "flavor": self.flavor,
                "lumi": self.lumi,
            }
        )

        if masked:
            # no data histogram for masked channel
            return

        if fitresult_data is not None:
            fitresult_axes = [n for n in fitresult_data.axes.name]
            if fitresult_axes != self.fit_axes:
                raise RuntimeError(
                    f"The axes of the fitresult {fitresult_axes} are different from the fit axes {self.fit_axes} but can't be re-ordered as they are according to their covariance matrix. Please choose the fit axes accordingly."
                )

            data_obs_hist = fitresult_data
        elif real_data:
            data_obs_hist = self.groups[self.dataName].hists[self.nominalName]
        else:
            data_obs_hist = hh.sumHists(
                [
                    self.groups[proc].hists[self.nominalName]
                    for proc in self.predictedProcesses()
                ]
            )

        logger.info(f"Add data histogram")
        if data_obs_hist.axes.name != self.fit_axes:
            data_obs_hist = data_obs_hist.project(*self.fit_axes)
        self.writer.add_data(data_obs_hist, self.channel)

    def addNormSystematic(self, norm, **kwargs):
        self.addSystematic(
            preOp=hh.scaleHist, preOpArgs={"scale": norm}, mirror=True, **kwargs
        )

    def addSystematic(
        self,
        histname=None,
        name=None,
        nominalName=None,
        processes=None,
        noi=False,
        noConstraint=False,
        mirror=False,
        symmetrize="average",
        preOp=None,
        preOpMap=None,
        preOpArgs={},
        passToFakes=False,
        forceNonzero=False,
        applySelection=True,
        scale=1,
        group=None,
        groups=[],
        splitGroup=None,
        action=None,
        actionArgs={},
        actionRequiresNomi=False,
        **kwargs,
    ):
        """
        'preOp': Operation that is applied on each member before members are summed up to groups and before the selection is performed
        'action': Operation that is applied after everything else
        """

        if group is not None:
            groups = [*groups, group]
        if splitGroup is not None:
            # precompile splitGroup expressions for better performance
            splitGroup = {g: re.compile(v) for g, v in splitGroup.items()}

        nominalName = self.nominalName if nominalName is None else nominalName

        if preOp and preOpMap:
            raise ValueError("Only one of preOp and preOpMap args are allowed")
        if histname is None:
            histname = nominalName
        if name is None:
            if histname == nominalName:
                raise RuntimeError(
                    "The systematic is based on the nominal histogram, a name must be specified."
                )
            else:
                name = histname

        logger.info(f"Now in channel {self.channel} at shape systematic group {name}")

        if self.isExcludedNuisance(name):
            return

        if isinstance(processes, str):
            processes = [processes]
        # Need to make an explicit copy of the array before appending
        procs_to_add = [
            x for x in (self.allMCProcesses() if processes is None else processes)
        ]
        procs_to_add = self.expandProcesses(procs_to_add)

        if preOp:
            preOpMap = {
                n: preOp
                for n in set(
                    [m.name for g in procs_to_add for m in self.groups[g].members]
                )
            }

        if passToFakes and self.fakeName not in procs_to_add:
            procs_to_add.append(self.fakeName)

        # protection when the input list is empty because of filters but the systematic is built reading the nominal
        # since the nominal reads all filtered processes regardless whether a systematic is passed to them or not
        # this can happen when creating new systs by scaling of the nominal histogram
        if not len(procs_to_add):
            return

        # Needed to avoid always reading the variation for the fakes, even for procs not specified
        forceToNominal = [
            x
            for x in self.getProcNames()
            if x
            not in self.getProcNames(
                [
                    p
                    for g in procs_to_add
                    for p in self.expandProcesses(g)
                    if p != self.fakeName
                ]
            )
        ]

        self.loadHistsForDatagroups(
            nominalName,
            histname,
            label="syst",
            procsToRead=procs_to_add,
            forceNonzero=forceNonzero and name != "qcdScaleByHelicity",
            preOpMap=preOpMap,
            preOpArgs=preOpArgs,
            applySelection=applySelection,
            forceToNominal=forceToNominal,
            sumFakesPartial=True,
        )

        for proc in procs_to_add:
            logger.debug(f"Now at proc {proc}!")

            hvar = self.groups[proc].hists["syst"]

            if action is not None:
                if actionRequiresNomi:
                    hnom = self.groups[proc].hists[self.nominalName]
                    hvar = action(hvar, hnom, **actionArgs)
                else:
                    hvar = action(hvar, **actionArgs)

            var_map = self.systHists(name, hvar, **kwargs)

            for var_name, hists in var_map.items():
                matched_groups = groups[:]
                if splitGroup is not None:
                    matched_groups.extend(
                        [
                            grp
                            for grp, matchre in splitGroup.items()
                            if matchre.match(var_name)
                        ]
                    )

                if hasattr(self, "axes_disable_flow") and len(self.axes_disable_flow):
                    if isinstance(hists, hist.Hist):
                        hists = hh.disableFlow(hists, self.axes_disable_flow)
                    else:
                        hists = [
                            hh.disableFlow(h, self.axes_disable_flow) for h in hists
                        ]

                logger.debug(f"Add systematic {var_name}")
                self.writer.add_systematic(
                    hists,
                    var_name,
                    proc,
                    self.channel,
                    groups=matched_groups,
                    mirror=mirror,
                    symmetrize=symmetrize,
                    kfactor=scale,
                    noi=noi,
                    constrained=not noConstraint,
                    add_to_data_covariance=self.isAbsorbedNuisance(name),
                )

    def systHists(
        self,
        name,
        hvar,
        systAxes=[],
        systAxesFlow=[],
        labelsByAxis=None,
        skipEntries=None,
        baseName="",
        systNameReplace=None,
        formatWithValue=None,
        outNames=[],
        isPoiHistDecorr=False,
    ):
        if name == self.nominalName or len(systAxes) == 0:
            if hvar.axes.name != self.fit_axes:
                hvar = hvar.project(*self.fit_axes)
            return {name: hvar}
        axNames = systAxes[:]
        axLabels = labelsByAxis[:] if labelsByAxis is not None else systAxes[:]

        if "downUpVar" in axNames and axNames[-1] != "downUpVar":
            logger.warning(
                "Axis 'downUpVar' detected, but not specified as trailing axis, this may lead to issues in pairing the up/down variation"
            )

        if not all([n in hvar.axes.name for n in axNames]):
            raise ValueError(
                f"Failed to find axis names {str(axNames)} in hist for syst {name}."
                f"Axes in hist are {str(hvar.axes.name)}"
            )

        # Converting to a list because otherwise if you print it for debugging you loose it
        def systIndexForAxis(axis, flow=False):
            if type(axis) == hist.axis.StrCategory:
                bins = [x for x in axis]
            else:
                bins = [a for a in range(axis.size)]
            if flow and axis.traits.underflow:
                bins = [hist.underflow, *bins]
            if flow and axis.traits.overflow:
                bins = [*bins, hist.overflow]
            return bins

        entries = list(
            itertools.product(
                *[
                    systIndexForAxis(hvar.axes[ax], flow=ax in systAxesFlow)
                    for ax in axNames
                ]
            )
        )

        def skipEntryDictToArray(h, skipEntry, axes):
            naxes = len(axes)

            if type(skipEntry) == dict:
                skipEntryArr = np.full(naxes, -1, dtype=object)
                nother_ax = h.ndim - naxes
                for k, v in skipEntry.items():
                    if k not in h.axes.name:
                        raise ValueError(
                            f"Invalid skipEntry expression {k} : {v}. Axis {k} is not in hist!"
                        )
                    idx = (
                        h.axes.name.index(k) - nother_ax
                    )  # Offset by the number of other axes, require that syst axes are the trailing ones
                    if idx < 0:
                        raise ValueError(
                            f"Invalid skip entry! Axis {k} was found in position {idx+nother_ax} of {h.ndim} axes, but {naxes} syst axes were expected"
                        )
                    skipEntryArr[idx] = v
                logger.debug(
                    f"Expanded skipEntry for is {skipEntryArr}. Syst axes are {h.axes.name[-naxes:]}"
                )
            elif isinstance(skipEntry, (bool, int, float, str)):
                skipEntryArr = (skipEntry,)
            elif type(skipEntry) not in (np.array, list, tuple):
                raise ValueError(
                    f"Unexpected format for skipEntry. Must be either dict, sequence, or scalar type. found {type(skipEntry)}"
                )
            else:
                skipEntryArr = skipEntry

            if len(skipEntryArr) != naxes:
                raise ValueError(
                    "skipEntry tuple must have the same dimensions as the number of syst axes. "
                    f"found {naxes} systematics and len(skipEntry) = {len(skipEntryArr)}."
                )
            return skipEntryArr

        def expandSkipEntries(h, syst, skipEntries, axes):
            updated_skip = []
            for skipEntry in skipEntries:
                skipEntry = skipEntryDictToArray(h, skipEntry, axes)
                # The lookup is handled by passing an imaginary number,
                # so detect these and then call the bin lookup on them
                # np.iscomplex returns false for 0.j, but still want to detect that
                to_lookup = np.array([isinstance(x, complex) for x in skipEntry])
                skip_arr = np.array(skipEntry, dtype=object)
                if to_lookup.any():
                    naxes = len(axes)
                    bin_lookup = np.array(
                        [
                            ax.index(x.imag)
                            for x, ax in zip(skipEntry, h.axes[-naxes:])
                            if isinstance(x, complex)
                        ]
                    )
                    # Need to loop here rather than using skip_arr.real because the dtype is object to allow strings
                    skip_arr = np.array([a.real for a in skip_arr])
                    skip_arr[to_lookup] += bin_lookup
                updated_skip.append(skip_arr)

            return updated_skip

        # TODO: Really would be better to use the axis names, not just indices
        def excludeSystEntry(entry, entries_to_skip):
            # Check if the entry in the hist matches one of the entries in entries_to_skip, across all axes
            # Can use -1 to exclude all values of an axis
            def match_entry(curr_entry, to_skip):
                if isinstance(to_skip, list) and all(
                    isinstance(x, str) for x in to_skip
                ):
                    return any(match_entry(curr_entry, m) for m in to_skip)
                return (
                    to_skip == -1
                    or curr_entry == to_skip
                    or re.match(str(to_skip), str(curr_entry))
                )

            for skipEntry in entries_to_skip:
                if all(match_entry(e, m) for e, m in zip(entry, skipEntry)):
                    return True
            # If no matches were found for any of the entries_to_skip possibilities
            return False

        def systLabelForAxis(axLabel, entry, axis, formatWithValue=None):
            if type(axis) == hist.axis.StrCategory:
                if entry in axis:
                    return entry
                else:
                    raise ValueError(
                        f"Did not find label {entry} in categorical axis {axis}"
                    )
            if axLabel == "downUpVar":
                return "Up" if entry else "Down"
            if "{i}" in axLabel:
                return axLabel.format(i=entry)

            if formatWithValue:
                if formatWithValue == "center":
                    entry = axis.centers[entry]
                elif formatWithValue == "low":
                    entry = axis.edges[:-1][entry]
                elif formatWithValue == "high":
                    entry = axis.edges[1:][entry]
                elif formatWithValue == "edges":
                    low = axis.edges[entry]
                    high = axis.edges[entry + 1]
                    lowstr = (
                        f"{low:0.1f}".replace(".", "p")
                        if not low.is_integer()
                        else str(int(low))
                    )
                    highstr = (
                        f"{high:0.1f}".replace(".", "p")
                        if not high.is_integer()
                        else str(int(high))
                    )
                    entry = f"{lowstr}_{highstr}"
                else:
                    raise ValueError(
                        f"Invalid formatWithValue choice {formatWithValue}."
                    )

            if type(entry) in [float, np.float64]:
                entry = (
                    f"{entry:0.1f}".replace(".", "p")
                    if not entry.is_integer()
                    else str(int(entry))
                )
            elif entry == hist.underflow:
                entry = "U"
            elif entry == hist.overflow:
                entry = "O"

            return f"{axLabel}{entry}"

        outNames = outNames[:]
        if len(outNames) == 0:
            if skipEntries is not None:
                skipEntries = expandSkipEntries(hvar, name, skipEntries, axNames)

            for entry in entries:
                if skipEntries and excludeSystEntry(entry, skipEntries):
                    outNames.append("")
                else:
                    fwv = formatWithValue
                    iname = baseName + "".join(
                        [
                            systLabelForAxis(
                                al, entry[i], hvar.axes[ax], fwv[i] if fwv else fwv
                            )
                            for i, (al, ax) in enumerate(zip(axLabels, axNames))
                        ]
                    )
                    if systNameReplace is not None:
                        for rep in systNameReplace:
                            iname = iname.replace(*rep)
                            logger.debug(f"Replacement {rep} yields new name {iname}")
                    outNames.append(iname)
            if not len(outNames):
                raise RuntimeError(f"Did not find any valid variations for syst {name}")

        variations = [
            hvar[{ax: binnum for ax, binnum in zip(axNames, entry)}]
            for entry in entries
        ]

        if len(variations) != len(outNames):
            raise RuntimeError(
                f"The number of variations doesn't match the number of names."
                f"Found {len(outNames)} names and {len(variations)} variations."
            )

        var_map = {
            n: var.project(*self.fit_axes) if var.axes.name != self.fit_axes else var
            for n, var in zip(outNames, variations)
            if n
        }

        # pair all up/down histograms, otherwise single histogram for mirroring
        # NB: with decorrelated axis, Up/Down might not be at the end, must search them within the string
        result = {}
        for key in var_map.keys():
            if not key:
                continue
            if isPoiHistDecorr:
                # use Down for first search: less likely to be present on its own in the name
                if "Down" in key:
                    base_key, tail_key = key.split("Down", 1)
                    key_up = base_key + "Up" + tail_key
                    if key_up in outNames:
                        result[base_key + tail_key] = (var_map[key_up], var_map[key])
                    else:
                        result[key] = var_map[key]
                elif "Up" in key:
                    base_key, tail_key = key.split("Up", 1)
                    key_down = base_key + "Down" + tail_key
                    if key_down in outNames:
                        continue
                    else:
                        result[key] = var_map[key]
                else:
                    result[key] = var_map[key]
            else:
                if key.endswith("Up"):
                    base_key = key[:-2]
                    key_down = base_key + "Down"
                    if key_down in outNames:
                        result[base_key] = (var_map[key], var_map[key_down])
                    else:
                        result[key] = var_map[key]
                elif key.endswith("Down"):
                    if key[:-4] + "Up" in outNames:
                        continue
                    else:
                        result[key] = var_map[key]
                else:
                    result[key] = var_map[key]
        return result

    def addPseudodataHistogramFakes(
        self, pseudodata, pseudodataGroups, forceNonzero=False
    ):
        pseudodataGroups.nominalName = self.nominalName
        pseudodataGroups.rebinOp = self.rebinOp
        pseudodataGroups.rebinBeforeSelection = self.rebinBeforeSelection
        pseudodataGroups.lumiScale = self.lumiScale
        pseudodataGroups.lumiScaleVarianceLinearly = self.lumiScaleVarianceLinearly

        processes = [
            x
            for x in pseudodataGroups.groups.keys()
            if x != self.dataName and self.pseudoDataProcsRegexp.match(x)
        ]
        processes = self.expandProcesses(processes)

        processesFromNomi = [
            x
            for x in pseudodataGroups.groups.keys()
            if x != self.dataName and not self.pseudoDataProcsRegexp.match(x)
        ]

        if pseudodata in ["closure", "truthMC"]:
            # get the nonclosure for fakes/multijet background from QCD MC
            pseudodataGroups.loadHistsForDatagroups(
                baseName=self.nominalName,
                syst=self.nominalName,
                label="syst",
                procsToRead=pseudodataGroups.groups.keys(),
                forceNonzero=forceNonzero,
                sumFakesPartial=False,
                applySelection=False,
            )
            procDict = pseudodataGroups.groups
            gTruth = procDict["QCDTruth"]
            hTruth = gTruth.histselector.get_hist(gTruth.hists["syst"])

            # now load the nominal histograms
            # only load nominal histograms that are not already loaded
            processesFromNomiToLoad = [
                proc
                for proc in self.groups.keys()
                if self.nominalName not in self.groups[proc].hists
            ]
            if len(processesFromNomiToLoad):
                self.loadHistsForDatagroups(
                    baseName=self.nominalName,
                    syst=self.nominalName,
                    procsToRead=processesFromNomiToLoad,
                    forceNonzero=forceNonzero,
                )
            if "QCD" not in procDict:
                # use truth MC as QCD
                logger.info(f"Have MC QCD truth {hTruth.sum()}")
                hFake = hTruth
            else:
                # compute the nonclosure correction
                gPred = procDict["QCD"]
                hPred = gPred.histselector.get_hist(gPred.hists["syst"])
                logger.info(
                    f"Have MC QCD truth {hTruth.sum()} and predicted {hPred.sum()}"
                )
                histCorr = hh.divideHists(hTruth, hPred)

                # apply the nonclosure to fakes derived from data
                hFake = self.groups[self.fakeName].hists[self.nominalName]
                if any([a not in hFake.axes for a in histCorr.axes]):
                    # TODO: Make if work for arbitrary axes (maybe as an action when loading nominal histogram, before fakerate axes are integrated e.g. in mt fit)
                    raise NotImplementedError(
                        f"The multijet closure test is not implemented for arbitrary axes, the required axes are {histCorr.axes.name}"
                    )
                hFake = hh.multiplyHists(hFake, histCorr)

                # apply variances from hCorr to fakes to account for stat uncertainty
                hFakeNominal = self.groups[self.getFakeName()].hists[self.nominalName]
                hFakeNominal.variances(flow=True)[...] = hFake.variances(flow=True)
                self.groups[self.getFakeName()].hists[self.nominalName] = hFakeNominal

            # done, now sum all histograms
            hists_data = [
                self.groups[x].hists[self.nominalName]
                for x in self.predictedProcesses()
                if x != self.fakeName
            ]
            hdata = hh.sumHists([*hists_data, hFake]) if len(hists_data) > 0 else hFake

        elif pseudodata in ["dataClosure", "mcClosure"]:
            # build the pseudodata by adding the nonclosure

            # build the pseudodata by adding the nonclosure
            # first load the nonclosure
            if pseudodata == "dataClosure":
                pseudodataGroups.loadHistsForDatagroups(
                    baseName=self.nominalName,
                    syst=self.nominalName,
                    label=pseudodata,
                    procsToRead=[self.fakeName],
                    forceNonzero=forceNonzero,
                    sumFakesPartial=not self.simultaneousABCD,
                    applySelection=False,
                )
                hist_fake = pseudodataGroups.groups[self.fakeName].hists[pseudodata]
            elif pseudodata == "mcClosure":
                hist_fake = pseudodataGroups.results[f"QCDmuEnrichPt15_{self.era}"][
                    "output"
                ]["unweighted"].get()

            fakeselector = self.groups[self.fakeName].histselector

            _0, _1 = fakeselector.calculate_fullABCD_smoothed(
                hist_fake, signal_region=True
            )
            params_d = fakeselector.spectrum_regressor.params
            cov_d = fakeselector.spectrum_regressor.cov

            hist_fake = hh.scaleHist(hist_fake, fakeselector.global_scalefactor)
            _0, _1 = fakeselector.calculate_fullABCD_smoothed(hist_fake)
            params = fakeselector.spectrum_regressor.params
            cov = fakeselector.spectrum_regressor.cov

            # add the nonclosure by adding the difference of the parameters
            fakeselector.spectrum_regressor.external_params = params_d - params
            # load the pseudodata including the nonclosure
            self.loadHistsForDatagroups(
                baseName=self.nominalName,
                syst=self.nominalName,
                label=pseudodata,
                procsToRead=[x for x in self.groups.keys() if x != self.getDataName()],
                forceNonzero=forceNonzero,
            )
            # adding the pseudodata
            hdata = hh.sumHists(
                [
                    self.groups[x].hists[pseudodata]
                    for x in self.groups.keys()
                    if x != self.getDataName()
                ]
            )

            # remove the parameter offset again
            fakeselector.spectrum_regressor.external_params = None
            # add the covariance matrix from the nonclosure to the model
            fakeselector.external_cov = cov + cov_d
        elif pseudodata.split("-")[0] in ["simple", "extended1D", "extended2D"]:
            # pseudodata for fakes using data with different fake estimation, change the selection but still keep the nominal histogram
            parts = pseudodata.split("-")
            if len(parts) == 2:
                pseudoDataMode, pseudoDataSmoothingMode = parts
            else:
                pseudoDataMode = pseudodata
                pseudoDataSmoothingMode = "full"

            pseudodataGroups.set_histselectors(
                pseudodataGroups.getNames(),
                self.nominalName,
                mode=pseudoDataMode,
                smoothing_mode=pseudoDataSmoothingMode,
                smoothingOrderFakerate=3,
                integrate_x=True,
                mcCorr=[None],
            )

            pseudodataGroups.loadHistsForDatagroups(
                baseName=self.nominalName,
                syst=self.nominalName,
                label=pseudodata,
                procsToRead=pseudodataGroups.groups.keys(),
                forceNonzero=forceNonzero,
            )
            procDict = pseudodataGroups.groups
            hists = [
                procDict[proc].hists[pseudodata]
                for proc in pseudodataGroups.groups.keys()
            ]

            # add BBB stat on top of nominal
            hist_fake = self.groups[self.getFakeName()].hists[self.nominalName]
            hist_fake.variances(flow=True)[...] = (
                pseudodataGroups.groups[self.getFakeName()]
                .hists[pseudodata]
                .variances(flow=True)
            )
            self.groups[self.getFakeName()].hists[self.nominalName] = hist_fake
            # now add possible processes from nominal
            logger.warning(
                f"Making pseudodata summing these processes: {pseudodataGroups.groups.keys()}"
            )

            # done, now sum all histograms
            hdata = hh.sumHists(hists)

        logger.info(f"Write pseudodata {pseudodata}")
        if hdata.axes.name != self.fit_axes:
            hdata = hdata.project(*self.fit_axes)
        self.writer.add_pseudodata(hdata, pseudodata, self.channel)

    def addPseudodataHistograms(
        self,
        pseudodataGroups,
        pseudodata,
        pseudodata_axes=[None],
        idxs=[None],
        pseudoDataProcsRegexp=".*",
        forceNonzero=False,
    ):

        pseudodataGroups.nominalName = self.nominalName
        pseudodataGroups.rebinOp = self.rebinOp
        pseudodataGroups.rebinBeforeSelection = self.rebinBeforeSelection
        pseudodataGroups.lumiScale = self.lumiScale
        pseudodataGroups.lumiScaleVarianceLinearly = self.lumiScaleVarianceLinearly

        pseudoDataAxes = pseudodata_axes[:]
        if len(pseudodata) != len(pseudodata_axes):
            if len(pseudodata_axes) == 1:
                pseudoDataAxes = pseudodata_axes * len(pseudodata)
            else:
                raise RuntimeError(
                    f"Found {len(pseudodata)} histograms for pseudodata but {len(pseudodata_axes)} corresponding axes, need either the same number or exactly 1 axis to be specified."
                )

        idxs = [int(idx) if idx is not None and idx.isdigit() else idx for idx in idxs]
        if len(pseudodata) == 1:
            pseudoDataIdxs = [idxs]
        elif len(pseudodata) > 1:
            if len(idxs) == 1:
                pseudoDataIdxs = [[idxs[0]]] * len(pseudodata)
            elif len(pseudodata) == len(idxs):
                pseudoDataIdxs = [[idxs[i]] for i in range(len(idxs))]
            else:
                raise RuntimeError(
                    f"""Found {len(pseudodata)} histograms for pseudodata but {len(idxs)} corresponding indices,
                    need either 1 histogram or exactly 1 index or the same number of histograms and indices to be specified."""
                )

        # name for the pseudodata set to be written into the output file
        pseudoDataProcsRegexp = re.compile(pseudoDataProcsRegexp)

        processes = [
            x
            for x in pseudodataGroups.groups.keys()
            if x != self.dataName and pseudoDataProcsRegexp.match(x)
        ]
        processes = self.expandProcesses(processes)

        processesFromNomi = [
            x
            for x in pseudodataGroups.groups.keys()
            if x != self.dataName and not pseudoDataProcsRegexp.match(x)
        ]

        for idx, p in enumerate(pseudodata):
            pseudodataGroups.loadHistsForDatagroups(
                baseName=self.nominalName,
                syst=p,
                label=p,
                procsToRead=processes,
                forceNonzero=forceNonzero,
            )
            hists = [
                pseudodataGroups.groups[proc].hists[p]
                for proc in processes
                if proc not in processesFromNomi
            ]
            # now add possible processes from nominal
            logger.warning(f"Making pseudodata summing these processes: {processes}")
            if len(processesFromNomi):
                # only load nominal histograms that are not already loaded
                processesFromNomiToLoad = [
                    proc
                    for proc in processesFromNomi
                    if self.nominalName not in self.groups[proc].hists
                ]
                if len(processesFromNomiToLoad):
                    logger.warning(
                        f"These processes are taken from nominal datagroups: {processesFromNomiToLoad}"
                    )
                    self.loadHistsForDatagroups(
                        baseName=self.nominalName,
                        syst=self.nominalName,
                        procsToRead=processesFromNomiToLoad,
                        forceNonzero=forceNonzero,
                    )
                hists.extend(
                    [
                        self.groups[proc].hists[self.nominalName]
                        for proc in processesFromNomi
                    ]
                )
            # done, now sum all histograms
            hdata = hh.sumHists(hists)
            if pseudoDataAxes[idx] is None:
                extra_ax = [ax for ax in hdata.axes.name if ax not in self.fit_axes]
                if len(extra_ax) > 0 and extra_ax[-1] in [
                    "vars",
                    "systIdx",
                    "tensor_axis_0",
                ]:
                    pseudoDataAxes[idx] = extra_ax[-1]
                    logger.info(f"Setting pseudoDataSystAx[{idx}] to {extra_ax[-1]}")
                    if pseudoDataIdxs[idx] == [None]:
                        pseudoDataIdxs[idx] = [0]
                        logger.info(f"Setting pseudoDataIdxs[{idx}] to {[0]}")
            if (
                pseudoDataAxes[idx] is not None
                and pseudoDataAxes[idx] not in hdata.axes.name
            ):
                raise RuntimeError(
                    f"Pseudodata axis {pseudoDataAxes[idx]} not found in {hdata.axes.name}."
                )

            if pseudoDataAxes[idx] is not None:
                pseudo_axis = hdata.axes[pseudoDataAxes[idx]]

                if (
                    len(pseudoDataIdxs[idx]) == 1
                    and pseudoDataIdxs[idx][0] is not None
                    and int(pseudoDataIdxs[idx][0]) == -1
                ):
                    pseudoDataIdxs[idx] = pseudo_axis

                for syst_idx in pseudoDataIdxs[idx]:
                    _idx = 0 if syst_idx is None else syst_idx

                    if type(pseudo_axis) == hist.axis.StrCategory:
                        syst_bin = (
                            pseudo_axis.bin(_idx) if type(_idx) == int else str(_idx)
                        )
                    else:
                        syst_bin = (
                            str(pseudo_axis.index(_idx))
                            if type(_idx) == int
                            else str(_idx)
                        )
                    name = f"{p}_{pseudoDataAxes[idx]}{f'_{syst_bin}' if syst_idx not in [None, 0] else ''}"

                    logger.info(f"Write pseudodata {name}")

                    h = hdata[{pseudoDataAxes[idx]: _idx}]
                    if h.axes.name != self.fit_axes:
                        h = h.project(*self.fit_axes)

                    if self.channel not in self.writer.channels:
                        self.writer.add_channel(axes=h.axes, name=self.channel)

                    self.writer.add_pseudodata(h, name, self.channel)
            else:
                # pseudodata from alternative histogram that has no syst axis
                logger.info(f"Write pseudodata {p}")
                if hdata.axes.name != self.fit_axes:
                    hdata = hdata.project(*self.fit_axes)

                if self.channel not in self.writer.channels:
                    self.writer.add_channel(axes=hdata.axes, name=self.channel)

                self.writer.add_pseudodata(hdata, p, self.channel)

    def addPseudodataHistogramsFitInput(
        self,
        pseudodata,
        pseudodataFitInput,
        pseudoDataFitInputChannel,
        pseudodataFitInputDownUp,
    ):
        channel = pseudoDataFitInputChannel
        for idx, p in enumerate(pseudodata):
            if p == "nominal":
                phist = pseudodataFitInput.nominal_hists[channel]
            elif p == "syst":
                phist = pseudodataFitInput.syst_hists[channel][
                    {"DownUp": pseudodataFitInputDownUp}
                ]
            else:
                raise ValueError(
                    "For pseudodata fit input the only valid names are 'nominal' and 'syst'."
                )
            logger.info(f"Write pseudodata {p}")
            if phist.axes.name != self.fit_axes:
                phist = phist.project(*self.fit_axes)
            self.writer.add_pseudodata(phist, p, self.channel)

    @staticmethod
    def histName(baseName, procName="", syst=""):
        if baseName != "x" and (syst == ""):
            return baseName
        if baseName in ["", "x"] and syst:
            return syst
        if syst[: len(baseName)] == baseName:
            return syst
        return "_".join([baseName, syst])

    @staticmethod
    def analysisLabel(filename):
        if filename not in Datagroups.mode_map:
            logger.warning(
                f"Unrecognized analysis script {filename}! Expected one of {Datagroups.mode_map.keys()}"
            )
            return filename.replace(".py", "")
        else:
            return Datagroups.mode_map[filename]
