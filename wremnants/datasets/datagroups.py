from utilities import boostHistHelpers as hh,common,logging
from utilities.io_tools import input_tools
from utilities.styles import styles
import lz4.frame
import pickle
import h5py
import narf
import ROOT
import re
import os
import itertools
import functools
import hist
import pandas as pd
import math
import numpy as np

from wremnants.datasets.datagroup import Datagroup
from wremnants import histselections as sel

logger = logging.child_logger(__name__)

class Datagroups(object):
    mode_map = {
        "w_z_gen_dists.py" : "vgen",
        "mz_dilepton.py" : "z_dilepton",
        "mz_wlike_with_mu_eta_pt.py" : "z_wlike",
        "mw_with_mu_eta_pt.py" : "w_mass",
        "mw_lowPU.py" : "w_lowpu",
        "mz_lowPU.py" : "z_lowpu",
    }

    def __init__(self, infile, mode=None, **kwargs):
        self.h5file = None
        self.rtfile = None
        if infile.endswith(".pkl.lz4"):
            with lz4.frame.open(infile) as f:
                self.results = pickle.load(f)
        elif infile.endswith(".hdf5"):
            logger.info("Load input file")
            self.h5file = h5py.File(infile, "r")
            self.results = input_tools.load_results_h5py(self.h5file)
        else:
            raise ValueError(f"{infile} has unsupported file type")

        if mode == None:
            analysis_script = os.path.basename(self.getScriptCommand().split()[0])
            self.mode = Datagroups.analysisLabel(analysis_script)
        else:
            if mode not in self.mode_map.values():
                raise ValueError(f"Unrecognized mode '{mode}.' Must be one of {set(mode_map.values())}")
            self.mode = mode
        logger.info(f"Set mode to {self.mode}")

        try:
            args = self.getMetaInfo()["args"]
            self.flavor = args.get("flavor", None)
        except ValueError as e:
            logger.warning(e)
            self.flavor = None

        self.groups = {}
        self.nominalName = "nominal"
        self.rebinOp = None
        self.rebinBeforeSelection = False
        self.globalAction = None
        self.unconstrainedProcesses = []
        self.fakeName = "Fake" + (f"_{self.flavor}" if self.flavor is not None else '')
        self.dataName = "Data"
        self.gen_axes = {}
        self.fakerate_axes = ["pt", "eta", "charge"]

        self.setGenAxes()

        if "lowpu" in self.mode:
            from wremnants.datasets.datagroupsLowPU import make_datagroups_lowPU as make_datagroups
            self.era = "2017H"
        else:
            from wremnants.datasets.datagroups2016 import make_datagroups_2016 as make_datagroups
            self.era = "2016postVFP"

        make_datagroups(self, **kwargs)

        self.lumi = sum([value.get("lumi", 0) for key, value in self.results.items()])
        if self.lumi > 0:
            logger.info(f"Integrated luminosity from data: {self.lumi}/fb")
        else:
            self.lumi = 1
            logger.warning(f"No data process was selected, normalizing MC to {self.lumi }/fb")

    def get_members_from_results(self, startswith=[], not_startswith=[], is_data=False):
        dsets = {k: v for k, v in self.results.items() if type(v) == dict and "dataset" in v}
        if is_data:
            dsets = {k: v for k, v in dsets.items() if v["dataset"].get("is_data", False)}
        else:
            dsets = {k: v for k, v in dsets.items() if not v["dataset"].get("is_data", False)}
        if type(startswith) == str:
            startswith = [startswith]
        if len(startswith) > 0:
            dsets = {k: v for k, v in dsets.items() if any([v["dataset"]["name"].startswith(x) for x in startswith])}
        if type(not_startswith) == str:
            not_startswith = [not_startswith]
        if len(not_startswith) > 0:
            dsets = {k: v for k, v in dsets.items() if not any([v["dataset"]["name"].startswith(x) for x in not_startswith])}
        return dsets

    def __del__(self):
        if self.h5file:
            self.h5file.close()
        if self.rtfile:
            self.rtfile.Close()

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
            new_groupnames += list(filter(lambda x, s=selection: x == s, self.groups.keys()))

        # remove duplicates selected by multiple filters
        return list(set(new_groupnames))

    def mergeGroups(self, groups, new_name):
        groups_to_merge=[]
        for g in groups:
            if g in self.groups:
                groups_to_merge.append(g)
            else:
                logger.warning(f"Did not find group {g}. continue without merging it to new group {new_name}.")
        if len(groups_to_merge) < 1:
            logger.warning(f"No groups to be merged. continue without merging.")
            return
        if new_name != groups_to_merge[0]:
            self.copyGroup(groups_to_merge[0], new_name)
        self.groups[new_name].label = styles.process_labels.get(new_name, new_name)
        self.groups[new_name].color = styles.process_colors.get(new_name, "grey")
        for group in groups_to_merge[1:]:            
            self.groups[new_name].addMembers(self.groups[group].members, member_operations=self.groups[group].memberOp)
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
            logger.info(f"Datagroups.filterGroups : filtered out following groups: {diff}")

        self.groups = {key: self.groups[key] for key in new_groupnames}

        if len(self.groups) == 0:
            logger.warning(f"Filtered groups using '{filters}' but didn't find any match. Continue without any group.")

    def excludeGroups(self, excludes):
        if excludes is None:
            return

        if isinstance(excludes, str):
            excludes = [excludes]

        if isinstance(excludes, list):
            new_groupnames = list(filter(lambda x: x not in self.selectGroups(excludes), self.groups))
        else:
            new_groupnames = list(filter(excludes, self.groups.keys()))

        diff = list(self.groups.keys() - set(new_groupnames))
        if diff:
            logger.info(f"Datagroups.excludeGroups: filtered out following groups: {diff}")

        self.groups = {key: self.groups[key] for key in new_groupnames}
        
        if len(self.groups) == 0:
            logger.warning(f"Excluded all groups using '{excludes}'. Continue without any group.")

    def set_histselectors(
        self,
        group_names,
        histToRead="nominal",
        fake_processes=None,
        mode="extended1D",
        smoothing_mode="full",
        smoothingOrderFakerate=2,
        integrate_shapecorrection_x=True, # integrate the abcd x-axis or not, only relevant for extended2D
        simultaneousABCD=False,
        forceGlobalScaleFakes=None,
        mcCorr=["pt","eta"],
        **kwargs
    ):
        logger.info(f"Set histselector")
        if self.mode[0] != "w":
            return # histselectors only implemented for single lepton (with fakes)
        auxiliary_info={}
        signalselector = sel.SignalSelectorABCD
        scale = 1
        if mode == "extended1D":
            scale = 0.85
            fakeselector = sel.FakeSelector1DExtendedABCD
        elif mode == "extended2D":
            scale = 1.15
            fakeselector = sel.FakeSelector2DExtendedABCD
            auxiliary_info["integrate_shapecorrection_x"]=integrate_shapecorrection_x
            if smoothing_mode == "fakerate" and not integrate_shapecorrection_x:
                auxiliary_info.update(dict(smooth_shapecorrection=True, interpolate_x=True, rebin_x="automatic"))
            else:
                auxiliary_info.update(dict(smooth_shapecorrection=False, interpolate_x=False, rebin_x=None))
        elif mode == "extrapolate":
            fakeselector = sel.FakeSelectorExtrapolateABCD
        elif mode == "simple":
            scale = 0.85
            if simultaneousABCD:
                fakeselector = sel.FakeSelectorSimultaneousABCD
            else:
                fakeselector = sel.FakeSelectorSimpleABCD
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
            if g in fake_processes:
                self.groups[g].histselector = fakeselector(
                    h,
                    global_scalefactor=scale,
                    fakerate_axes=self.fakerate_axes,
                    smoothing_mode=smoothing_mode,
                    smoothing_order_fakerate=smoothingOrderFakerate,
                    **auxiliary_info, **kwargs
                    )
                if mode in ["simple", "extended1D", "extended2D"] and forceGlobalScaleFakes is None and (len(mcCorr)==0 or mcCorr[0] not in ["none", None]):
                    # set QCD MC nonclosure corrections
                    if "QCDmuEnrichPt15PostVFP" not in self.results:
                        logger.warning("Dataset 'QCDmuEnrichPt15PostVFP' not in results, continue without fake correction")
                        return
                    if "unweighted" not in self.results["QCDmuEnrichPt15PostVFP"]["output"]:
                        logger.warning("Histogram 'unweighted' not found, continue without fake correction")
                        return
                    hQCD = self.results["QCDmuEnrichPt15PostVFP"]["output"]["unweighted"].get()
                    self.groups[g].histselector.set_correction(hQCD, axes_names=mcCorr)
            else:
                self.groups[g].histselector = signalselector(h, fakerate_axes=self.fakerate_axes, **kwargs)

    def setGlobalAction(self, action):
        # To be used for applying a selection, rebinning, etc.
        if self.globalAction is None:
            self.globalAction = action
        else:
            self.globalAction = lambda h, old_action=self.globalAction: action(old_action(h))

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
        return self.lumi*1000*proc.xsec/proc.weight_sum

    def getMetaInfo(self):
        if self.results:
            if "meta_info" not in self.results and "meta_data" not in self.results:
                raise ValueError("Did not find meta data in results file")
            return self.results["meta_info"] if "meta_info" in self.results else self.results["meta_data"]
        raise NotImplementedError("Currently can't access meta data as dict for ROOT file")

    def getScriptCommand(self):
        if self.rtfile:
            return self.rtfile.Get("meta_info/command").GetTitle()
        else:
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
    def loadHistsForDatagroups(self, 
        baseName, syst, procsToRead=None, label=None, nominalIfMissing=True, 
        applySelection=True, forceNonzero=False, preOpMap=None, preOpArgs={}, 
        scaleToNewLumi=1, lumiScaleVarianceLinearly=[], excludeProcs=None, forceToNominal=[], sumFakesPartial=True,
    ):
        logger.debug("Calling loadHistsForDatagroups()")
        logger.debug(f"The basename and syst is: {baseName}, {syst}")
        logger.debug(f"The procsToRead and excludedProcs are: {procsToRead}, {excludeProcs}")
        if not label:
            label = syst if syst else baseName
        # this line is annoying for the theory agnostic, too many processes for signal
        logger.debug(f"In loadHistsForDatagroups(): for hist {syst} procsToRead = {procsToRead}")

        if not procsToRead:
            if excludeProcs:
                procsToRead = list(filter(lambda x: x not in excludeProcs, self.groups.keys()))
            else:
                procsToRead = list(self.groups.keys())

        foundExact = False

        # If fakes are present do them as last group, and when running on prompt group build the sum to be used for the fakes.
        # This makes the code faster and avoid possible bugs related to reading again the same processes
        # NOTE:
        # To speed up even more, one could directly use the per-group sum already computed for each group,
        # but this would need to assume that fakes effectively had all the single processes in each group as members
        # (usually it will be the case, but it is more difficult to handle in a fully general way and without bugs)
        histForFake = None # to store the data-MC sums used for the fakes, for each syst
        if sumFakesPartial and self.fakeName in procsToRead:
            procsToReadSort = [x for x in procsToRead if x != self.fakeName] + [self.fakeName]
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
                raise RuntimeError(f"Group {procName} not known. Defined groups are {list(self.groups.keys())}.")
            group = self.groups[procName]

            group.hists[label] = None

            for i, member in enumerate(group.members):
                if sumFakesPartial and procName == self.fakeName and member.name in fakesMembersWithSyst:
                    # if we are here this process has been already used to build the fakes when running for other groups
                    continue
                logger.debug(f"Looking at group member {member.name}")
                read_syst = syst
                if member.name in forceToNominal:
                    read_syst = ""
                    logger.debug(f"Forcing group member {member.name} to read the nominal hist for syst {syst}")
                try:
                    h = self.readHist(baseName, member, procName, read_syst)
                    foundExact = True
                except ValueError as e:
                    if nominalIfMissing:
                        logger.info(f"{str(e)}. Using nominal hist {self.nominalName} instead")
                        h = self.readHist(self.nominalName, member, procName, "")
                    else:
                        logger.warning(str(e))
                        continue

                h_id = id(h)
                logger.debug(f"Hist axes are {h.axes.name}")

                if group.memberOp:
                    if group.memberOp[i] is not None:
                        logger.debug(f"Apply operation to member {i}: {member.name}/{procName}")
                        h = group.memberOp[i](h)
                    else:
                        logger.debug(f"No operation for member {i}: {member.name}/{procName}")

                if preOpMap and member.name in preOpMap:
                    logger.debug(f"Applying action to {member.name}/{procName} after loading")
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
                if not np.isclose(scaleToNewLumi, 1, rtol=0, atol=1e-6) and ((procName == self.dataName and "data" in lumiScaleVarianceLinearly) or (procName != self.dataName and "mc" in lumiScaleVarianceLinearly)):
                        logger.warning(f"Scale {procName} hist by {scaleToNewLumi} as a multiplicative luminosity factor, with variance scaled linearly")
                        h = hh.scaleHist(h, scaleToNewLumi, createNew=False, scaleVarianceLinearly=True)
                else:
                    scale *= scaleToNewLumi

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
                        logger.debug(f"Summing hist {read_syst} for {member.name} to {self.fakeName} with scale = {scaleProcForFake}")
                        hProcForFake = scaleProcForFake * h
                        histForFake = hh.addHists(histForFake, hProcForFake, createNew=False) if histForFake else hProcForFake

                # The following must be done when the group is not Fake, or when the previous part for fakes was not done
                # For fake this essentially happens when the process doesn't have the syst, so that the nominal is used
                if procName != self.fakeName or (procName == self.fakeName and not hasPartialSumForFake):
                    if procName == self.fakeName:
                        logger.debug(f"Summing nominal hist instead of {syst} to {self.fakeName} for {member.name}")
                    else:
                        logger.debug(f"Summing {read_syst} to {procName} for {member.name}")

                    group.hists[label] = hh.addHists(group.hists[label], h, createNew=False) if group.hists[label] else h

            if not nominalIfMissing and group.hists[label] is None:
                continue

            # now sum to fakes the partial sums which where not already done before
            # (group.hists[label] contains only the contribution from nominal histograms).
            # Then continue with the rest of the code as usual
            if hasFake and procName == self.fakeName:
                if histForFake is not None:
                    group.hists[label] = hh.addHists(group.hists[label], histForFake, createNew=False) if group.hists[label] else histForFake

            if self.rebinOp and self.rebinBeforeSelection:
                logger.debug(f"Apply rebin operation for process {procName}")
                group.hists[label] = self.rebinOp(group.hists[label])

            if group.histselector is not None:
                if not applySelection:
                    logger.warning(f"Selection requested for process {procName} but applySelection=False, thus it will be ignored")
                elif label in group.hists.keys() and group.hists[label] is not None:
                    group.hists[label] = group.histselector.get_hist(group.hists[label], is_nominal=(label==self.nominalName))
                else:
                    raise RuntimeError("Failed to apply selection")

            if self.rebinOp and not self.rebinBeforeSelection:
                logger.debug(f"Apply rebin operation for process {procName}")
                group.hists[label] = self.rebinOp(group.hists[label])

        # Avoid situation where the nominal is read for all processes for this syst
        if nominalIfMissing and not foundExact:
            raise ValueError(f"Did not find systematic {syst} for any processes!")

    def getDatagroups(self):
        return self.groups

    def getNames(self, matches=[], exclude=False, match_exact=False):
        # This method returns the names from the defined groups, unless one selects further.
        listOfNames = list(x for x in self.groups.keys())
        if not matches:
            return listOfNames
        else:
            # matches uses regular expressions with search (and can be inverted when exclude is true),
            # thus a string will match if the process name contains that string anywhere inside it
            if exclude:
                return list(filter(lambda x: all([re.search(expr, x) is None for expr in matches]), listOfNames))
            elif match_exact:
                return [x for x in listOfNames if x in matches]
            else:
                return list(filter(lambda x: any([re.search(expr, x) for expr in matches]), listOfNames))
              
    def getProcNames(self, to_expand=[], exclude_group=[]):
        procs = []
        if not to_expand:
            to_expand = self.groups.keys()
        for group_name in to_expand:
            if group_name not in self.groups:
                raise ValueError(f"Trying to expand unknown group {group_name}. Valid groups are {list(self.groups.keys())}")
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
            sorted(self.groups.items(), key=lambda x: get_sum(
                x[1].hists[histName if histName in x[1].hists else nominalName])
                    if nominalName in x[1].hists or histName in x[1].hists else 0,
                reverse=True)
        )

    def getDatagroupsForHist(self, histName):
        filled = {}
        for k, v in self.groups.items():
            if histName in v:
                filled[k] = v
        return filled

    def resultsDict(self):
        return self.results

    def addSummedProc(self, refname, name, label=None, color=None, exclude=["Data"], relabel=None, 
            procsToRead=None, reload=False, rename=None, action=None, actionArgs={}, actionRequiresRef=False, **kwargs):
        if reload:
            self.loadHistsForDatagroups(refname, syst=name, excludeProcs=exclude,
                procsToRead=procsToRead, **kwargs)

        if not rename:
            rename = name
        self.addGroup(rename,
            label=label,
            color=color,
            members=[],
        )
        tosum = []
        procs = procsToRead if procsToRead else self.groups.keys()
        for proc in filter(lambda x: x not in exclude+[rename], procs):
            h = self.groups[proc].hists[name]
            if not h:
                raise ValueError(f"Failed to find hist for proc {proc}, histname {name}")
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
        if processes == None:
            procs = self.groups
        else:
            procs = [processes] if isinstance(processes, str) else processes

        for proc in procs:
            if proc not in self.groups.keys():
                raise ValueError(f"In setSelectOp(): process {proc} not found")
            self.groups[proc].selectOp = op

    def setGenAxes(self, gen_axes_names=None, sum_gen_axes=None):
        # gen_axes_names are the axes names to be recognized as gen axes, e.g. for the unfolding
        # sum_gen_axes are all gen axes names that are potentially in the produced histogram and integrated over if not used
        if isinstance(gen_axes_names, str):
            gen_axes_names = [gen_axes_names]
        if isinstance(sum_gen_axes, str):
            sum_gen_axes = [sum_gen_axes]

        # infer all gen axes from metadata
        try:
            args = self.getMetaInfo()["args"]
        except ValueError as e:
            logger.warning("No meta data found so no gen axes could be auto set")
            return

        self.all_gen_axes = args.get("genAxes", [])

        if self.mode[0] == "w":
            self.all_gen_axes = ["qGen", *self.all_gen_axes]

        self.gen_axes_names = list(gen_axes_names) if gen_axes_names != None else self.all_gen_axes
        self.sum_gen_axes = list(sum_gen_axes) if sum_gen_axes != None else self.all_gen_axes

        logger.debug(f"Gen axes names are now {self.gen_axes_names}")

    def getGenBinIndices(self, h, axesToRead=None):
        gen_bins = []
        for gen_axis in (self.gen_axes_names if axesToRead is None else axesToRead):
            if gen_axis not in h.axes.name:
                raise RuntimeError(f"Gen axis '{gen_axis}' not found in histogram axes '{h.axes.name}'!")

            gen_bin_list = [i for i in range(h.axes[gen_axis].size)]
            if h.axes[gen_axis].traits.underflow:
                gen_bin_list.append(hist.underflow)
            if h.axes[gen_axis].traits.overflow:
                gen_bin_list.append(hist.overflow)
            gen_bins.append(gen_bin_list)
        return gen_bins

    def defineSignalBinsUnfolding(self, group_name, new_name=None, member_filter=None, histToReadAxes="xnorm", axesToRead=None):
        if group_name not in self.groups.keys():
            raise RuntimeError(f"Base group {group_name} not found in groups {self.groups.keys()}!")
        if axesToRead is None:
            axesToRead = self.gen_axes_names
        base_members = self.groups[group_name].members[:]
        if member_filter is not None:
            base_members = [m for m in filter(lambda x, f=member_filter: f(x), base_members)]            

        if histToReadAxes not in self.results[base_members[0].name]["output"]:
            raise ValueError(f"Results for member {base_members[0].name} does not include xnorm. Found {self.results[base_members[0].name]['output'].keys()}")
        nominal_hist = self.results[base_members[0].name]["output"][histToReadAxes].get()

        self.gen_axes[new_name] = [ax for ax in nominal_hist.axes if ax.name in axesToRead]
        logger.debug(f"New gen axes are: {self.gen_axes}")

        gen_bin_indices = self.getGenBinIndices(nominal_hist, axesToRead=axesToRead)

        for indices in itertools.product(*gen_bin_indices):

            proc_name = group_name if new_name is None else new_name
            for idx, var in zip(indices, axesToRead):
                if idx == hist.underflow:
                    idx_str = "U"
                elif idx == hist.overflow:
                    idx_str = "O"
                else:
                    idx_str = str(idx)
                proc_name += f"_{var}{idx_str}"


            self.copyGroup(group_name, proc_name, member_filter=member_filter)

            memberOp = lambda x, indices=indices, genvars=axesToRead: x[{var : i for var, i in zip(genvars, indices)}]
            self.groups[proc_name].memberOp = [memberOp for m in base_members]

            self.unconstrainedProcesses.append(proc_name)

    def select_xnorm_groups(self, select_groups=None):
        # only keep members and groups where xnorm is defined
        logger.info("Select xnorm groups"+(f" {select_groups}" if select_groups else ""))
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
                    raise RuntimeError(f"The member {member.name} of group {g_name} was not found in the results!")
                if "xnorm" not in self.results[member.name]["output"].keys():
                    logger.debug(f"Member {member.name} has no xnorm and will be deleted")
                    toDel_members.append(member)
            if len(toDel_members) == len(group.members):
                logger.warning(f"All members of group {g_name} have no xnorm and the group will be deleted")
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

        df = pd.DataFrame([(k, *sum_and_unc(action(v.hists[histName]))) for k,v in self.groups.items() if k in procs], 
                columns=["Process", "Yield", "Uncertainty"])

        if norm_proc and norm_proc in self.groups:
            hist = action(self.groups[norm_proc].hists[histName])
            denom = hist.sum() if not hasattr(hist.sum(), "value") else hist.sum().value
            df[f"Ratio to {norm_proc} (%)"] = df["Yield"]/denom*100
            
        return df

    def set_rebin_action(self, axes, ax_lim=[], ax_rebin=[], ax_absval=[], rebin_before_selection=False, rename=True):
        self.rebinBeforeSelection = rebin_before_selection

        for a in hh.get_rebin_actions(axes, ax_lim=ax_lim, ax_rebin=ax_rebin, ax_absval=ax_absval, rename=rename):
            self.setRebinOp(a)

    def readHist(self, baseName, proc, group, syst):
        output = self.results[proc.name]["output"]
        histname = self.histName(baseName, proc.name, syst)
        logger.debug(f"Reading hist {histname} for proc/group {proc.name}/{group} and syst '{syst}'")
        if histname not in output:
            raise ValueError(f"Histogram {histname} not found for process {proc.name}")

        h = output[histname]
        if isinstance(h, narf.ioutils.H5PickleProxy):
            h = h.get()

        return h

    def histName(self, baseName, procName="", syst=""):
        return Datagroups.histName(baseName, procName, syst, nominalName=self.nominalName)

    @staticmethod
    def histName(baseName, procName="", syst=""):
        if baseName != "x" and (syst == ""):
            return baseName
        if baseName in ["", "x"] and syst:
            return syst
        if syst[:len(baseName)] == baseName:
            return syst
        return "_".join([baseName,syst])
    
    @staticmethod
    def analysisLabel(filename):
        if filename not in Datagroups.mode_map:
            raise ValueError(f"Unrecognized analysis script {filename}! Expected one of {Datagroups.mode_map.keys()}")

        return Datagroups.mode_map[filename]
