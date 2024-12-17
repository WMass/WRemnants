import itertools
import re

import hist
import numpy as np

from utilities import boostHistHelpers as hh
from utilities import logging

logger = logging.child_logger(__name__)


def notImplemented(operation="Unknown"):
    raise NotImplementedError(f"Required operation '{operation}' is not implemented!")


def checkFiniteBinValues(h, hname, flow=True, throw=True):
    # one could use this function without raising error to count how many occurrences there are
    if np.all(np.isfinite(h.values(flow=flow))):
        return 0
    else:
        errMsg = f"One or more NAN or Inf values encountered in {hname} histogram"
        logger.error(errMsg)
        if throw:
            raise RuntimeError(errMsg)
        return 1


class CardTool(object):
    def __init__(self, outpath="./", xnorm=False, real_data=False):

        self.skipHist = (
            False  # don't produce/write histograms, file with them already exists
        )
        self.systematics = {}
        self.fakeEstimate = None
        self.cardGroups = {}
        self.nominalName = "nominal"
        self.datagroups = None
        self.pseudodata_datagroups = None
        self.unconstrainedProcesses = None
        self.noStatUncProcesses = []
        self.buildHistNameFunc = None
        self.histName = "x"
        self.nominalDim = None
        self.pseudoData = None
        self.pseudoDataAxes = None
        self.pseudoDataIdxs = None
        self.pseudoDataName = None
        self.pseudoDataProcsRegexp = None
        self.pseudodataFitInput = None
        self.excludeSyst = None
        self.keepSyst = (
            None  # to override previous one with exceptions for special cases
        )
        self.lumiScale = 1.0
        self.lumiScaleVarianceLinearly = []
        self.fit_axes = None
        self.xnorm = xnorm
        self.real_data = real_data
        self.charge_ax = "charge"
        self.procGroups = {}
        self.binByBinStatScale = 1.0
        self.exponentialTransform = False

    def getProcNames(self, grouped_procs):
        expanded_procs = []
        for group in grouped_procs:
            procs = self.expandProcess(group)
            for ungrouped in procs:
                expanded_procs.extend(self.datagroups.getProcNames([ungrouped]))

        return expanded_procs

    def addProcessGroup(self, name, procFilter):
        self.procGroups[name] = self.filteredProcesses(procFilter)
        if not self.procGroups[name]:
            logger.warning(
                f"Did not match any processes to filter for group {name}! Valid procs are {self.datagroups.groups.keys()}"
            )

    def expandProcesses(self, processes):
        if type(processes) == str:
            processes = [processes]

        return [x for y in processes for x in self.expandProcess(y)]

    def expandProcess(self, process):
        return self.procGroups.get(process, [process])

    def skipHistograms(self):
        self.skipHist = True
        if len(self.noStatUncProcesses):
            logger.info(
                "Attention: histograms are not saved according to input options, thus statistical uncertainty won't be zeroed"
            )

    def setFitAxes(self, axes):
        self.fit_axes = axes[:]

    def setProcsNoStatUnc(self, procs, resetList=True):
        if self.skipHist:
            logger.info(
                "Attention: trying to set statistical uncertainty to 0 for some processes, but histograms won't be saved according to input options"
            )
        if resetList:
            self.noStatUncProcesses = []
        if isinstance(procs, str):
            self.noStatUncProcesses.append(procs)
        elif isinstance(procs, list):
            self.noStatUncProcesses.extend(procs)
        else:
            raise ValueError(
                "In setNoStatUncForProcs(): expecting string or list argument"
            )

    def setLumiScale(self, lumiScale, lumiScaleVarianceLinearly=[]):
        self.lumiScale = lumiScale
        self.lumiScaleVarianceLinearly = lumiScaleVarianceLinearly

    def getProcsNoStatUnc(self):
        return self.noStatUncProcesses

    ## Functions to customize systs to be added in card, mainly for tests
    def setCustomSystForCard(self, exclude=None, keep=None):
        for regex, name in zip((keep, exclude), ("keepSyst", "excludeSyst")):
            if hasattr(self, "customSystMapping"):
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

    def setFakeName(self, name):
        self.datagroups.fakeName = name

    def getFakeName(self):
        return self.datagroups.fakeName

    def setDataName(self, name):
        self.datagroups.dataName = name

    def getDataName(self):
        return self.datagroups.dataName

    def setPseudodata(
        self,
        pseudodata,
        pseudodata_axes=[None],
        idxs=[None],
        pseudoDataProcsRegexp=".*",
    ):
        self.pseudoData = pseudodata[:]
        self.pseudoDataAxes = pseudodata_axes[:]
        self.pseudoDataProcsRegexp = re.compile(pseudoDataProcsRegexp)
        if len(pseudodata) != len(pseudodata_axes):
            if len(pseudodata_axes) == 1:
                self.pseudoDataAxes = pseudodata_axes * len(pseudodata)
            else:
                raise RuntimeError(
                    f"Found {len(pseudodata)} histograms for pseudodata but {len(pseudodata_axes)} corresponding axes, need either the same number or exactly 1 axis to be specified."
                )
        idxs = [int(idx) if idx is not None and idx.isdigit() else idx for idx in idxs]
        if len(pseudodata) == 1:
            self.pseudoDataIdxs = [idxs]
        elif len(pseudodata) > 1:
            if len(idxs) == 1:
                self.pseudoDataIdxs = [[idxs[0]]] * len(pseudodata)
            elif len(pseudodata) == len(idxs):
                self.pseudoDataIdxs = [[idxs[i]] for i in range(len(idxs))]
            else:
                raise RuntimeError(
                    f"""Found {len(pseudodata)} histograms for pseudodata but {len(idxs)} corresponding indices,
                    need either 1 histogram or exactly 1 index or the same number of histograms and indices to be specified."""
                )
        # name for the pseudodata set to be written into the output file
        self.pseudoDataName = [
            f"{n}{f'_{a}' if a is not None else ''}"
            for n, a in zip(self.pseudoData, self.pseudoDataAxes)
        ]

    def setDatagroups(self, datagroups, resetGroups=False):
        self.datagroups = datagroups
        if self.pseudodata_datagroups is None:
            self.pseudodata_datagroups = datagroups
        self.unconstrainedProcesses = datagroups.unconstrainedProcesses
        if self.nominalName:
            self.datagroups.setNominalName(self.nominalName)
        if datagroups.mode == "vgen":
            self.charge_ax = "chargeVgen"

    def setPseudodataDatagroups(self, datagroups):
        self.pseudodata_datagroups = datagroups
        if self.nominalName:
            self.pseudodata_datagroups.setNominalName(self.nominalName)

    def predictedProcesses(self):
        return list(
            filter(lambda x: x != self.getDataName(), self.datagroups.groups.keys())
        )

    def setHistName(self, histName):
        self.histName = histName

    def setNominalName(self, histName):
        self.nominalName = histName
        if self.datagroups:
            self.datagroups.setNominalName(histName)

    def setBinByBinStatScale(self, scale):
        self.binByBinStatScale = scale

    def setExponentialTransform(self, transform=True):
        self.exponentialTransform = transform

    # by default this returns True also for Fake since it has Data in the list of members
    # then self.isMC negates this one and thus will only include pure MC processes
    def isData(self, procName, onlyData=False):
        if onlyData:
            return all([x.is_data for x in self.datagroups.groups[procName].members])
        else:
            return any([x.is_data for x in self.datagroups.groups[procName].members])

    def isMC(self, procName):
        return not self.isData(procName)

    def addFakeEstimate(self, estimate):
        self.fakeEstimate = estimate

    def getProcesses(self):
        return list(self.datagroups.groups.keys())

    def filteredProcesses(self, filterExpr):
        return list(filter(filterExpr, self.datagroups.groups.keys()))

    def allMCProcesses(self):
        return self.filteredProcesses(lambda x: self.isMC(x))

    def precompile_splitGroupDict(self, group, splitGroup):
        # precompile splitGroup expressions for better performance
        splitGroupDict = {g: re.compile(v) for g, v in splitGroup.items()}
        # add the group with everything if not there already
        if group not in splitGroupDict:
            splitGroupDict[group] = re.compile(".*")
        return splitGroupDict

    # preOp is a function to apply per process, preOpMap can be used with a dict for a speratate function for each process,
    #   it is executed before summing the processes. Arguments can be specified with preOpArgs
    # action will be applied to the sum of all the individual samples contributing, arguments can be specified with actionArgs
    def addSystematic(
        self,
        name,
        nominalName=None,
        systAxes=[],
        systAxesFlow=[],
        outNames=None,
        skipEntries=None,
        labelsByAxis=None,
        baseName="",
        mirror=False,
        mirrorDownVarEqualToUp=False,
        mirrorDownVarEqualToNomi=False,
        symmetrize="average",
        scale=1,
        processes=None,
        group=None,
        noi=False,
        noConstraint=False,
        noProfile=False,
        preOp=None,
        preOpMap=None,
        preOpArgs={},
        action=None,
        actionArgs={},
        actionRequiresNomi=False,
        selectionArgs={},
        systNameReplace=[],
        systNamePrepend=None,
        groupFilter=None,
        passToFakes=False,
        rename=None,
        splitGroup={},
        formatWithValue=None,
        customizeNuisanceAttributes={},
        applySelection=True,
    ):
        logger.debug(f"Add systematic {name}")
        # note: setting Up=Down seems to be pathological for the moment, it might be due to the interpolation in the fit
        # for now better not to use the options, although it might be useful to keep it implemented
        if mirrorDownVarEqualToUp or mirrorDownVarEqualToNomi:
            raise ValueError(
                "mirrorDownVarEqualToUp and mirrorDownVarEqualToNomi currently lead to pathological results in the fit, please keep them False"
            )

        if not mirror and (mirrorDownVarEqualToUp or mirrorDownVarEqualToNomi):
            raise ValueError(
                "mirrorDownVarEqualToUp and mirrorDownVarEqualToNomi requires mirror=True"
            )

        if mirrorDownVarEqualToUp and mirrorDownVarEqualToNomi:
            raise ValueError(
                "mirrorDownVarEqualToUp and mirrorDownVarEqualToNomi cannot be both True"
            )

        if symmetrize not in [None, "average", "conservative", "linear", "quadratic"]:
            raise ValueError(
                "Invalid option for 'symmetrize'.  Valid options are None, 'average' 'conservative', 'linear', and 'quadratic'"
            )

        if preOp and preOpMap:
            raise ValueError("Only one of preOp and preOpMap args are allowed")

        if nominalName is None:
            nominalName = self.nominalName

        if isinstance(processes, str):
            processes = [processes]
        # Need to make an explicit copy of the array before appending
        procs_to_add = [
            x for x in (self.allMCProcesses() if processes is None else processes)
        ]
        procs_to_add = self.expandProcesses(procs_to_add)

        if preOp:
            preOpMap = {
                name: preOp
                for name in set(
                    [
                        m.name
                        for g in procs_to_add
                        for m in self.datagroups.groups[g].members
                    ]
                )
            }

        if passToFakes and self.getFakeName() not in procs_to_add:
            procs_to_add.append(self.getFakeName())

        # protection when the input list is empty because of filters but the systematic is built reading the nominal
        # since the nominal reads all filtered processes regardless whether a systematic is passed to them or not
        # this can happen when creating new systs by scaling of the nominal histogram
        if not len(procs_to_add):
            return

        if name == self.nominalName:
            logger.debug(f"Defining syst {rename} from nominal histogram")

        self.systematics.update(
            {
                name if not rename else rename: {
                    "outNames": [] if not outNames else outNames,
                    "outNamesFinal": [] if not outNames else outNames,
                    "baseName": baseName,
                    "nominalName": nominalName,
                    "processes": procs_to_add,
                    "systAxes": systAxes,
                    "systAxesFlow": systAxesFlow,
                    "labelsByAxis": systAxes if not labelsByAxis else labelsByAxis,
                    "group": group,
                    "noi": noi,
                    "groupFilter": groupFilter,
                    "splitGroup": self.precompile_splitGroupDict(group, splitGroup),
                    "scale": scale,
                    "customizeNuisanceAttributes": customizeNuisanceAttributes,
                    "mirror": mirror,
                    "mirrorDownVarEqualToUp": mirrorDownVarEqualToUp,
                    "mirrorDownVarEqualToNomi": mirrorDownVarEqualToNomi,
                    "symmetrize": symmetrize,
                    "preOpMap": preOpMap,
                    "preOpArgs": preOpArgs,
                    "action": action,
                    "actionArgs": actionArgs,
                    "actionRequiresNomi": actionRequiresNomi,
                    "applySelection": applySelection,
                    "systNameReplace": systNameReplace,
                    "noConstraint": noConstraint,
                    "noProfile": noProfile,
                    "skipEntries": [] if not skipEntries else skipEntries,
                    "name": name,
                    "systNamePrepend": systNamePrepend,
                    "formatWithValue": formatWithValue,
                }
            }
        )

    # Read a specific hist, useful if you need to check info about the file
    def getHistsForProcAndSyst(self, proc, syst, nominal_name=None):
        if nominal_name is None:
            nominal_name = self.nominalName
        if not self.datagroups:
            raise RuntimeError(
                "No datagroups defined! Must call setDatagroups before accessing histograms"
            )
        self.datagroups.loadHistsForDatagroups(
            baseName=nominal_name,
            syst=syst,
            label="syst",
            procsToRead=[proc],
            scaleToNewLumi=self.lumiScale,
            lumiScaleVarianceLinearly=self.lumiScaleVarianceLinearly,
        )
        return self.datagroups.getDatagroups()[proc].hists["syst"]

    def getNominalHistForSignal(self):
        signal_samples = self.procGroups["signal_samples"]
        return self.getHistsForProcAndSyst(signal_samples[0], self.nominalName)

    def setMirrorForSyst(self, syst, mirror=True):
        self.systematics[syst]["mirror"] = mirror

    def systIndexForAxis(self, axis, flow=False):
        if type(axis) == hist.axis.StrCategory:
            bins = [x for x in axis]
        else:
            bins = [a for a in range(axis.size)]
        if flow and axis.traits.underflow:
            bins = [hist.underflow, *bins]
        if flow and axis.traits.overflow:
            bins = [*bins, hist.overflow]
        return bins

    def systLabelForAxis(self, axLabel, entry, axis, formatWithValue=None):
        if type(axis) == hist.axis.StrCategory:
            if entry in axis:
                return entry
            else:
                raise ValueError(
                    f"Did not find label {entry} in categorical axis {axis}"
                )
        if axLabel == "mirror":
            return (
                "Down" if entry else "Up"
            )  # first entry is the original, call it Up since it is usually defined by an actual scaling up of something (e.g. efficiencies)
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
                raise ValueError(f"Invalid formatWithValue choice {formatWithValue}.")

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

    # TODO: Really would be better to use the axis names, not just indices
    def excludeSystEntry(self, entry, entries_to_skip):
        # Check if the entry in the hist matches one of the entries in entries_to_skip, across all axes
        # Can use -1 to exclude all values of an axis
        def match_entry(curr_entry, to_skip):
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

    def skipEntryDictToArray(self, h, skipEntry, syst):
        nsyst = len(self.systematics[syst]["systAxes"])
        if "mirror" in h.axes.name:
            nsyst += 1

        if type(skipEntry) == dict:
            skipEntryArr = np.full(nsyst, -1, dtype=object)
            nother_ax = h.ndim - nsyst
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
                        f"Invalid skip entry! Axis {k} was found in position {idx+nother_ax} of {h.ndim} axes, but {nsyst} syst axes were expected"
                    )
                skipEntryArr[idx] = v
            logger.debug(
                f"Expanded skipEntry for syst {syst} is {skipEntryArr}. Syst axes are {h.axes.name[-nsyst:]}"
            )
        elif isinstance(skipEntry, (bool, int, float, str)):
            skipEntryArr = (skipEntry,)
        elif type(skipEntry) not in (np.array, list, tuple):
            raise ValueError(
                f"Unexpected format for skipEntry. Must be either dict, sequence, or scalar type. found {type(skipEntry)}"
            )
        else:
            skipEntryArr = skipEntry

        if (
            self.systematics[syst]["mirror"]
            and "mirror" not in h.axes.name
            and skipEntryArr[-1] == -1
        ):
            skipEntryArr = skipEntryArr[:-1]

        if len(skipEntryArr) != nsyst:
            raise ValueError(
                "skipEntry tuple must have the same dimensions as the number of syst axes. "
                f"found {nsyst} systematics and len(skipEntry) = {len(skipEntryArr)}."
            )

        return skipEntryArr

    def expandSkipEntries(self, h, syst, skipEntries):
        updated_skip = []
        for skipEntry in skipEntries:
            skipEntry = self.skipEntryDictToArray(h, skipEntry, syst)
            # The lookup is handled by passing an imaginary number,
            # so detect these and then call the bin lookup on them
            # np.iscomplex returns false for 0.j, but still want to detect that
            to_lookup = np.array([isinstance(x, complex) for x in skipEntry])
            skip_arr = np.array(skipEntry, dtype=object)
            if to_lookup.any():
                nsyst = (
                    len(self.systematics[syst]["systAxes"])
                    + self.systematics[syst]["mirror"]
                )
                bin_lookup = np.array(
                    [
                        ax.index(x.imag)
                        for x, ax in zip(skipEntry, h.axes[-nsyst:])
                        if isinstance(x, complex)
                    ]
                )
                # Need to loop here rather than using skip_arr.real because the dtype is object to allow strings
                skip_arr = np.array([a.real for a in skip_arr])
                skip_arr[to_lookup] += bin_lookup
            updated_skip.append(skip_arr)

        return updated_skip

    def systHists(self, hvar, syst, hnom):
        if syst == self.nominalName:
            return {self.nominalName: hvar}

        systInfo = self.systematics[syst]
        systAxes = systInfo["systAxes"]
        systAxesLabels = systInfo.get("labelsByAxis", systAxes)

        # Jan: moved above the mirror action, as this action can cause mirroring
        if systInfo["action"]:
            if systInfo["actionRequiresNomi"]:
                hvar = systInfo["action"](hvar, hnom, **systInfo["actionArgs"])
            else:
                hvar = systInfo["action"](hvar, **systInfo["actionArgs"])

        if len(systAxes) == 0:
            return {syst: hvar}

        axNames = systAxes[:]
        axLabels = systAxesLabels[:]
        if hvar.axes[-1].name == "mirror":
            axNames.append("mirror")
            axLabels.append("mirror")

        if not all([name in hvar.axes.name for name in axNames]):
            raise ValueError(
                f"Failed to find axis names {str(axNames)} in hist for syst {syst}. "
                f"Axes in hist are {str(hvar.axes.name)}"
            )

        # Converting to a list becasue otherwise if you print it for debugging you loose it
        entries = list(
            itertools.product(
                *[
                    self.systIndexForAxis(
                        hvar.axes[ax], flow=ax in systInfo["systAxesFlow"]
                    )
                    for ax in axNames
                ]
            )
        )

        if len(systInfo["outNames"]) == 0:
            skipEntries = (
                None
                if "skipEntries" not in systInfo
                else self.expandSkipEntries(hvar, syst, systInfo["skipEntries"])
            )
            for entry in entries:
                if skipEntries and self.excludeSystEntry(entry, skipEntries):
                    systInfo["outNames"].append("")
                else:
                    name = systInfo["baseName"]
                    fwv = systInfo["formatWithValue"]
                    if fwv:
                        if "mirror" in axLabels:
                            fwv.append(None)
                    name += "".join(
                        [
                            self.systLabelForAxis(
                                al, entry[i], hvar.axes[ax], fwv[i] if fwv else fwv
                            )
                            for i, (al, ax) in enumerate(zip(axLabels, axNames))
                        ]
                    )
                    if "systNameReplace" in systInfo and systInfo["systNameReplace"]:
                        for rep in systInfo["systNameReplace"]:
                            name = name.replace(*rep)
                            logger.debug(f"Replacement {rep} yields new name {name}")
                    if (
                        name
                        and "systNamePrepend" in systInfo
                        and systInfo["systNamePrepend"]
                    ):
                        name = systInfo["systNamePrepend"] + name
                    # Obviously there is a nicer way to do this...
                    if "Up" in name:
                        name = name.replace("Up", "") + "Up"
                    elif "Down" in name:
                        name = name.replace("Down", "") + "Down"
                    systInfo["outNames"].append(name)
            if not len(systInfo["outNames"]):
                raise RuntimeError(f"Did not find any valid variations for syst {syst}")

        variations = [
            hvar[{ax: binnum for ax, binnum in zip(axNames, entry)}]
            for entry in entries
        ]

        if hvar.axes[-1].name == "mirror" and len(variations) == 2 * len(
            systInfo["outNames"]
        ):
            systInfo["outNames"] = [
                n + d for n in systInfo["outNames"] for d in ["Up", "Down"]
            ]
        elif len(variations) != len(systInfo["outNames"]):
            logger.warning(
                f"The number of variations doesn't match the number of names for "
                f"syst {syst}. Found {len(systInfo['outNames'])} names and {len(variations)} variations."
            )
        return {
            name: var for name, var in zip(systInfo["outNames"], variations) if name
        }

    def variationName(self, proc, name):
        if name == self.nominalName:
            return f"{self.histName}_{proc}"
        else:
            return f"{self.histName}_{proc}_{name}"

    def getBoostHistByCharge(self, h, q):
        return h[
            {
                self.charge_ax: (
                    h.axes[self.charge_ax].index(q) if q != "sum" else hist.sum
                )
            }
        ]

    def loadPseudodataFakes(self, datagroups, forceNonzero=False):
        # get the nonclosure for fakes/multijet background from QCD MC
        datagroups.loadHistsForDatagroups(
            baseName=self.nominalName,
            syst=self.nominalName,
            label="syst",
            procsToRead=datagroups.groups.keys(),
            scaleToNewLumi=self.lumiScale,
            lumiScaleVarianceLinearly=self.lumiScaleVarianceLinearly,
            forceNonzero=forceNonzero,
            sumFakesPartial=False,
            applySelection=False,
        )
        procDict = datagroups.getDatagroups()
        gTruth = procDict["QCDTruth"]
        hTruth = gTruth.histselector.get_hist(gTruth.hists["syst"])

        # now load the nominal histograms
        # only load nominal histograms that are not already loaded
        procDictFromNomi = self.datagroups.getDatagroups()
        processesFromNomiToLoad = [
            proc
            for proc in self.datagroups.groups.keys()
            if self.nominalName not in procDictFromNomi[proc].hists
        ]
        if len(processesFromNomiToLoad):
            self.datagroups.loadHistsForDatagroups(
                baseName=self.nominalName,
                syst=self.nominalName,
                procsToRead=processesFromNomiToLoad,
                scaleToNewLumi=self.lumiScale,
                lumiScaleVarianceLinearly=self.lumiScaleVarianceLinearly,
                forceNonzero=forceNonzero,
                sumFakesPartial=True,
            )

        if "QCD" not in procDict:
            # use truth MC as QCD
            logger.info(f"Have MC QCD truth {hTruth.sum()}")
            hFake = hTruth
        else:
            # compute the nonclosure correction
            gPred = procDict["QCD"]
            hPred = gPred.histselector.get_hist(gPred.hists["syst"])
            logger.info(f"Have MC QCD truth {hTruth.sum()} and predicted {hPred.sum()}")
            histCorr = hh.divideHists(hTruth, hPred)

            # apply the nonclosure to fakes derived from data
            hFake = procDictFromNomi[self.datagroups.fakeName].hists[self.nominalName]
            if any([a not in hFake.axes for a in histCorr.axes]):
                # TODO: Make if work for arbitrary axes (maybe as an action when loading nominal histogram, before fakerate axes are integrated e.g. in mt fit)
                raise NotImplementedError(
                    f"The multijet closure test is not implemented for arbitrary axes, the required axes are {histCorr.axes.name}"
                )
            hFake = hh.multiplyHists(hFake, histCorr)

            # apply variances from hCorr to fakes to account for stat uncertainty
            hFakeNominal = procDictFromNomi[self.getFakeName()].hists[self.nominalName]
            hFakeNominal.variances(flow=True)[...] = hFake.variances(flow=True)
            procDictFromNomi[self.getFakeName()].hists[self.nominalName] = hFakeNominal

        # done, now sum all histograms
        hists_data = [
            procDictFromNomi[x].hists[self.nominalName]
            for x in self.predictedProcesses()
            if x != self.getFakeName()
        ]
        hdata = hh.sumHists([*hists_data, hFake]) if len(hists_data) > 0 else hFake

        return hdata

    def loadPseudodata(self, forceNonzero=False):

        hdatas = []

        if self.pseudodataFitInput:
            channel = self.pseudoDataFitInputChannel
            for idx, pseudoData in enumerate(self.pseudoData):
                if pseudoData == "nominal":
                    phist = self.pseudodataFitInput.nominal_hists[channel]
                elif pseudoData == "syst":
                    phist = self.pseudodataFitInput.syst_hists[channel][
                        {"DownUp": self.pseudodataFitInputDownUp}
                    ]
                else:
                    raise ValueError(
                        "For pseudodata fit input the only valid names are 'nominal' and 'syst'."
                    )

                hdatas.append(phist)
            return hdatas

        datagroups = self.pseudodata_datagroups
        processes = [
            x
            for x in datagroups.groups.keys()
            if x != self.getDataName() and self.pseudoDataProcsRegexp.match(x)
        ]
        processes = self.expandProcesses(processes)

        processesFromNomi = [
            x
            for x in datagroups.groups.keys()
            if x != self.getDataName() and not self.pseudoDataProcsRegexp.match(x)
        ]
        for idx, pseudoData in enumerate(self.pseudoData):
            if pseudoData in ["closure", "truthMC"]:
                # pseudodata for fakes [using closure from QCD MC as correction, using QCD MC as prediction]
                if pseudoData == "truthMC":
                    datagroups.deleteGroup("QCD")
                hdatas.append(
                    self.loadPseudodataFakes(datagroups, forceNonzero=forceNonzero)
                )
                continue

            if pseudoData in ["dataClosure", "mcClosure"]:
                # build the pseudodata by adding the nonclosure

                # build the pseudodata by adding the nonclosure
                # first load the nonclosure
                if pseudoData == "dataClosure":
                    datagroups.loadHistsForDatagroups(
                        baseName=self.nominalName,
                        syst=self.nominalName,
                        label=pseudoData,
                        procsToRead=[self.getFakeName()],
                        scaleToNewLumi=self.lumiScale,
                        lumiScaleVarianceLinearly=self.lumiScaleVarianceLinearly,
                        forceNonzero=forceNonzero,
                        sumFakesPartial=True,
                        applySelection=False,
                    )
                    hist_fake = datagroups.getDatagroups()[self.getFakeName()].hists[
                        pseudoData
                    ]
                elif pseudoData == "mcClosure":
                    hist_fake = datagroups.results["QCDmuEnrichPt15PostVFP"]["output"][
                        "unweighted"
                    ].get()

                fakeselector = self.datagroups.getDatagroups()[
                    self.getFakeName()
                ].histselector

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
                self.datagroups.loadHistsForDatagroups(
                    baseName=self.nominalName,
                    syst=self.nominalName,
                    label=pseudoData,
                    procsToRead=[
                        x
                        for x in self.datagroups.groups.keys()
                        if x != self.getDataName()
                    ],
                    scaleToNewLumi=self.lumiScale,
                    lumiScaleVarianceLinearly=self.lumiScaleVarianceLinearly,
                    forceNonzero=forceNonzero,
                    sumFakesPartial=True,
                )
                # adding the pseudodata
                hdata = hh.sumHists(
                    [
                        self.datagroups.getDatagroups()[x].hists[pseudoData]
                        for x in self.datagroups.groups.keys()
                        if x != self.getDataName()
                    ]
                )
                hdatas.append(hdata)

                # remove the parameter offset again
                fakeselector.spectrum_regressor.external_params = None
                # add the covariance matrix from the nonclosure to the model
                fakeselector.external_cov = cov + cov_d

                continue

            elif pseudoData.split("-")[0] in ["simple", "extended1D", "extended2D"]:
                # pseudodata for fakes using data with different fake estimation, change the selection but still keep the nominal histogram
                parts = pseudoData.split("-")
                if len(parts) == 2:
                    pseudoDataMode, pseudoDataSmoothingMode = parts
                else:
                    pseudoDataMode = pseudoData
                    pseudoDataSmoothingMode = "full"

                datagroups.set_histselectors(
                    datagroups.getNames(),
                    self.nominalName,
                    mode=pseudoDataMode,
                    smoothing_mode=pseudoDataSmoothingMode,
                    smoothingOrderFakerate=3,
                    integrate_x=True,
                    mcCorr=[None],
                )
                syst = self.nominalName
            else:
                syst = pseudoData

            datagroups.loadHistsForDatagroups(
                baseName=self.nominalName,
                syst=syst,
                label=pseudoData,
                procsToRead=processes,
                scaleToNewLumi=self.lumiScale,
                lumiScaleVarianceLinearly=self.lumiScaleVarianceLinearly,
                forceNonzero=forceNonzero,
                sumFakesPartial=True,
            )
            procDict = datagroups.getDatagroups()
            hists = [
                procDict[proc].hists[pseudoData]
                for proc in processes
                if proc not in processesFromNomi
            ]

            if pseudoData.split("-")[0] in ["simple", "extended1D", "extended2D"]:
                # add BBB stat on top of nominal
                hist_fake = self.datagroups.getDatagroups()[self.getFakeName()].hists[
                    self.nominalName
                ]
                hist_fake.variances(flow=True)[...] = (
                    datagroups.getDatagroups()[self.getFakeName()]
                    .hists[pseudoData]
                    .variances(flow=True)
                )
                self.datagroups.getDatagroups()[self.getFakeName()].hists[
                    self.nominalName
                ] = hist_fake

            # now add possible processes from nominal
            logger.warning(f"Making pseudodata summing these processes: {processes}")
            if len(processesFromNomi):
                # only load nominal histograms that are not already loaded
                datagroupsFromNomi = self.datagroups
                procDictFromNomi = datagroupsFromNomi.getDatagroups()
                processesFromNomiToLoad = [
                    proc
                    for proc in processesFromNomi
                    if self.nominalName not in procDictFromNomi[proc].hists
                ]
                if len(processesFromNomiToLoad):
                    logger.warning(
                        f"These processes are taken from nominal datagroups: {processesFromNomiToLoad}"
                    )
                    datagroupsFromNomi.loadHistsForDatagroups(
                        baseName=self.nominalName,
                        syst=self.nominalName,
                        procsToRead=processesFromNomiToLoad,
                        scaleToNewLumi=self.lumiScale,
                        lumiScaleVarianceLinearly=self.lumiScaleVarianceLinearly,
                        forceNonzero=forceNonzero,
                        sumFakesPartial=True,
                    )
                procDictFromNomi = datagroupsFromNomi.getDatagroups()
                hists.extend(
                    [
                        procDictFromNomi[proc].hists[self.nominalName]
                        for proc in processesFromNomi
                    ]
                )
            # done, now sum all histograms
            hdata = hh.sumHists(hists)
            if self.pseudoDataAxes[idx] is None:
                extra_ax = [ax for ax in hdata.axes.name if ax not in self.fit_axes]
                if len(extra_ax) > 0 and extra_ax[-1] in [
                    "vars",
                    "systIdx",
                    "tensor_axis_0",
                ]:
                    self.pseudoDataAxes[idx] = extra_ax[-1]
                    logger.info(f"Setting pseudoDataSystAx[{idx}] to {extra_ax[-1]}")
                    if self.pseudoDataIdxs[idx] == [None]:
                        self.pseudoDataIdxs[idx] = [0]
                        logger.info(f"Setting pseudoDataIdxs[{idx}] to {[0]}")
            if (
                self.pseudoDataAxes[idx] is not None
                and self.pseudoDataAxes[idx] not in hdata.axes.name
            ):
                raise RuntimeError(
                    f"Pseudodata axis {self.pseudoDataAxes[idx]} not found in {hdata.axes.name}."
                )
            hdatas.append(hdata)

        return hdatas

    def setUnconstrainedProcs(self, procs):
        self.unconstrainedProcesses = procs

    def processLabels(self, procs=None):
        nondata = np.array(self.predictedProcesses() if procs is None else procs)
        labels = np.arange(len(nondata)) + 1
        issig = np.isin(nondata, self.unconstrainedProcesses)
        labels[issig] = -np.arange(np.count_nonzero(issig)) - 1
        return labels

    def match_str_axis_entries(self, str_axis, match_re):
        return [x for x in str_axis if any(re.match(r, x) for r in match_re)]
