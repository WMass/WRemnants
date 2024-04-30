from collections import OrderedDict
from wremnants import histselections as sel
from utilities import boostHistHelpers as hh, common, logging
from utilities.io_tools import output_tools
import narf
import ROOT
import uproot
import time
import numpy as np
import os
import pathlib
import itertools
import re
import hist
import copy
import math

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
    def __init__(self, outpath="./", xnorm=False, simultaneousABCD=False, real_data=False):
    
        self.skipHist = False # don't produce/write histograms, file with them already exists
        self.outfile = None
        self.systematics = {}
        self.lnNSystematics = {}
        self.predictedProcs = []
        self.fakeEstimate = None
        self.channels = ["inclusive"]
        self.cardContent = {}
        self.cardGroups = {}
        self.cardSumGroups = {} # POI sum groups
        self.nominalTemplate = f"{pathlib.Path(__file__).parent}/../scripts/combine/Templates/datacard.txt"
        self.spacing = 28
        self.systTypeSpacing = 16
        self.procColumnsSpacing = 20
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
        self.excludeSyst = None
        self.writeByCharge = True
        self.unroll = False # unroll final histogram before writing to root
        self.keepSyst = None # to override previous one with exceptions for special cases
        self.lumiScale = 1.
        self.lumiScaleVarianceLinearly = []
        self.fit_axes = None
        self.xnorm = xnorm
        self.simultaneousABCD = simultaneousABCD
        self.real_data = real_data
        self.absolutePathShapeFileInCard = False
        self.excludeProcessForChannel = {} # can be used to exclue some POI when runnig a specific name (use case, force gen and reco charges to match)
        self.chargeIdDict = {"minus" : {"val" : -1, "id" : "q0", "badId" : "q1"},
                             "plus"  : {"val" : 1., "id" : "q1", "badId" : "q0"},
                             "inclusive" : {"val" : "sum", "id" : "none", "badId" : None},
                             }
        self.charge_ax = "charge"
        self.procGroups = {}

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
            logger.warning(f"Did not match any processes to filter for group {name}! Valid procs are {self.datagroups.groups.keys()}")

    def expandProcesses(self, processes):
        if type(processes) == str:
            processes = [processes]

        return [x for y in processes for x in self.expandProcess(y)]

    def expandProcess(self, process):
        return self.procGroups.get(process, [process])

    def skipHistograms(self):
        self.skipHist = True
        if len(self.noStatUncProcesses):
            logger.info("Attention: histograms are not saved according to input options, thus statistical uncertainty won't be zeroed")

    def setExcludeProcessForChannel(self, channel, POIregexp, canUpdate=False):
        if canUpdate or channel not in self.excludeProcessForChannel.keys():
            self.excludeProcessForChannel[channel] = re.compile(POIregexp)
            
    def setFitAxes(self, axes):
        self.fit_axes = axes[:]

    def setProcsNoStatUnc(self, procs, resetList=True):
        if self.skipHist:
            logger.info("Attention: trying to set statistical uncertainty to 0 for some processes, but histograms won't be saved according to input options")
        if resetList:
            self.noStatUncProcesses = []
        if isinstance(procs, str):
            self.noStatUncProcesses.append(procs)
        elif isinstance(procs, list):
            self.noStatUncProcesses.extend(procs)
        else:
            raise ValueError("In setNoStatUncForProcs(): expecting string or list argument")
    
    def setLumiScale(self, lumiScale, lumiScaleVarianceLinearly=[]):
        self.lumiScale = lumiScale
        self.lumiScaleVarianceLinearly = lumiScaleVarianceLinearly

    def setAbsolutePathShapeInCard(self, setRelative=False):
        self.absolutePathShapeFileInCard = False if setRelative else True
        
    def getProcsNoStatUnc(self):
        return self.noStatUncProcesses
        
    ## Functions to customize systs to be added in card, mainly for tests
    def setCustomSystForCard(self, exclude=None, keep=None):
        for regex,name in zip((keep, exclude), ("keepSyst", "excludeSyst")):
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

    def setPseudodata(self, pseudodata, pseudodata_axes=[None], idxs = [None], pseudoDataProcsRegexp=".*"):
        self.pseudoData = pseudodata[:]
        self.pseudoDataAxes = pseudodata_axes[:]        
        self.pseudoDataProcsRegexp = re.compile(pseudoDataProcsRegexp)
        if len(pseudodata) != len(pseudodata_axes):
            if len(pseudodata_axes) == 1:
                self.pseudoDataAxes = pseudodata_axes*len(pseudodata)
            else:
                raise RuntimeError(f"Found {len(pseudodata)} histograms for pseudodata but {len(pseudodata_axes)} corresponding axes, need either the same number or exactly 1 axis to be specified.")
        idxs = [int(idx) if idx is not None and idx.isdigit() else idx for idx in idxs]
        if len(pseudodata) == 1:
            self.pseudoDataIdxs = [idxs]
        elif len(pseudodata) > 1:
            if len(idxs) == 1:
                self.pseudoDataIdxs = [ [idxs[0]]]*len(pseudodata)
            elif len(pseudodata) == len(idxs):
                self.pseudoDataIdxs = [ [idxs[i]] for i in range(len(idxs))]
            else:
                raise RuntimeError(f"""Found {len(pseudodata)} histograms for pseudodata but {len(idxs)} corresponding indices, 
                    need either 1 histogram or exactly 1 index or the same number of histograms and indices to be specified.""")
        # name for the pseudodata set to be written into the output file
        self.pseudoDataName = [ f"{n}{f'_{a}' if a is not None else ''}" for n, a in zip(self.pseudoData, self.pseudoDataAxes)]


    # Needs to be increased from default for long proc names
    def setSpacing(self, spacing):
        self.spacing = spacing
        
    def setProcColumnsSpacing(self, spacing):
        self.procColumnsSpacing = spacing

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

    def setChannels(self, channels):
        self.channels = channels
        
    def setWriteByCharge(self, writeByCharge):
        self.writeByCharge = writeByCharge

    def setNominalTemplate(self, template):
        if not os.path.abspath(template):
            raise IOError(f"Template file {template} is not a valid file")
        self.nominalTemplate = template

    def predictedProcesses(self):
        if self.predictedProcs:
            return self.predictedProcs
        return list(filter(lambda x: x != self.getDataName(), self.datagroups.groups.keys()))

    def setHistName(self, histName):
        self.histName = histName

    def setNominalName(self, histName):
        self.nominalName = histName
        if self.datagroups:
            self.datagroups.setNominalName(histName)

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

    def addLnNSystematic(self, name, size, processes, group=None, groupFilter=None, splitGroup={}):
        if not self.isExcludedNuisance(name):
            self.lnNSystematics.update({name : {"size" : size,
                                                "processes" : self.expandProcesses(processes),
                                                "group" : group,
                                                "groupFilter" : groupFilter,
                                                "splitGroup" : self.precompile_splitGroupDict(group, splitGroup) 
                                                }
            })

    # preOp is a function to apply per process, preOpMap can be used with a dict for a speratate function for each process, 
    #   it is executed before summing the processes. Arguments can be specified with preOpArgs 
    # action will be applied to the sum of all the individual samples contributing, arguments can be specified with actionArgs
    def addSystematic(self, name, systAxes=[], systAxesFlow=[], outNames=None, skipEntries=None, labelsByAxis=None, 
                      baseName="", mirror=False, mirrorDownVarEqualToUp=False, mirrorDownVarEqualToNomi=False, symmetrize = "average",
                      scale=1, processes=None, group=None, noi=False, noConstraint=False, noProfile=False,
                      preOp=None, preOpMap=None, preOpArgs={}, action=None, actionArgs={}, actionRequiresNomi=False, selectionArgs={},
                      systNameReplace=[], systNamePrepend=None, groupFilter=None, passToFakes=False,
                      rename=None, splitGroup={}, formatWithValue=None,
                      customizeNuisanceAttributes={}, applySelection=True,
                      ):
        logger.debug(f"Add systematic {name}")
        # note: setting Up=Down seems to be pathological for the moment, it might be due to the interpolation in the fit
        # for now better not to use the options, although it might be useful to keep it implemented
        if mirrorDownVarEqualToUp or mirrorDownVarEqualToNomi:
            raise ValueError("mirrorDownVarEqualToUp and mirrorDownVarEqualToNomi currently lead to pathological results in the fit, please keep them False")

        if not mirror and (mirrorDownVarEqualToUp or mirrorDownVarEqualToNomi):
            raise ValueError("mirrorDownVarEqualToUp and mirrorDownVarEqualToNomi requires mirror=True")

        if mirrorDownVarEqualToUp and mirrorDownVarEqualToNomi:
            raise ValueError("mirrorDownVarEqualToUp and mirrorDownVarEqualToNomi cannot be both True")

        if symmetrize not in [None, "average", "conservative", "linear", "quadratic"]:
            raise ValueError("Invalid option for 'symmetrize'.  Valid options are None, 'average' 'conservative', 'linear', and 'quadratic'")

        if preOp and preOpMap:
            raise ValueError("Only one of preOp and preOpMap args are allowed")

        if isinstance(processes, str):
            processes = [processes]
        # Need to make an explicit copy of the array before appending
        procs_to_add = [x for x in (self.allMCProcesses() if processes is None else processes)]
        procs_to_add = self.expandProcesses(procs_to_add)

        if preOp:
            preOpMap = {name : preOp for name in set([m.name for g in procs_to_add for m in self.datagroups.groups[g].members])}

        if passToFakes and self.getFakeName() not in procs_to_add and not self.simultaneousABCD:
            procs_to_add.append(self.getFakeName())

        # protection when the input list is empty because of filters but the systematic is built reading the nominal
        # since the nominal reads all filtered processes regardless whether a systematic is passed to them or not
        # this can happen when creating new systs by scaling of the nominal histogram
        if not len(procs_to_add):
            return

        if name == self.nominalName:
            logger.debug(f"Defining syst {rename} from nominal histogram")
        
        self.systematics.update({
            name if not rename else rename : {
                "outNames" : [] if not outNames else outNames,
                "outNamesFinal" : [] if not outNames else outNames,
                "baseName" : baseName,
                "processes" : procs_to_add,
                "systAxes" : systAxes,
                "systAxesFlow" : systAxesFlow,
                "labelsByAxis" : systAxes if not labelsByAxis else labelsByAxis,
                "group" : group,
                "noi": noi,
                "groupFilter" : groupFilter,
                "splitGroup" : self.precompile_splitGroupDict(group, splitGroup), 
                "scale" : scale,
                "customizeNuisanceAttributes" : customizeNuisanceAttributes,
                "mirror" : mirror,
                "mirrorDownVarEqualToUp" : mirrorDownVarEqualToUp,
                "mirrorDownVarEqualToNomi" : mirrorDownVarEqualToNomi,
                "symmetrize" : symmetrize,
                "preOpMap" : preOpMap,
                "preOpArgs" : preOpArgs,
                "action" : action,
                "actionArgs" : actionArgs,
                "actionRequiresNomi" : actionRequiresNomi,
                "applySelection" : applySelection,
                "systNameReplace" : systNameReplace,
                "noConstraint" : noConstraint,
                "noProfile" : noProfile,
                "skipEntries" : [] if not skipEntries else skipEntries,
                "name" : name,
                "systNamePrepend" : systNamePrepend,
                "formatWithValue" : formatWithValue,
            }
        })

    # Read a specific hist, useful if you need to check info about the file
    def getHistsForProcAndSyst(self, proc, syst):
        if not self.datagroups:
            raise RuntimeError("No datagroups defined! Must call setDatagroups before accessing histograms")
        self.datagroups.loadHistsForDatagroups(
            baseName=self.nominalName, syst=syst, label="syst",
            procsToRead=[proc],
            scaleToNewLumi=self.lumiScale,
            lumiScaleVarianceLinearly=self.lumiScaleVarianceLinearly)
        return self.datagroups.getDatagroups()[proc].hists["syst"]

    def getNominalHistForSignal(self):
        signal_samples = self.procGroups['signal_samples']
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
                raise ValueError(f"Did not find label {entry} in categorical axis {axis}")
        if axLabel == "mirror":
            return 'Down' if entry else 'Up' # first entry is the original, call it Up since it is usually defined by an actual scaling up of something (e.g. efficiencies)
        if axLabel == "downUpVar":
            return 'Up' if entry else 'Down'
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
                high = axis.edges[entry+1]
                lowstr = f"{low:0.1f}".replace(".", "p") if not low.is_integer() else str(int(low))
                highstr = f"{high:0.1f}".replace(".", "p") if not high.is_integer() else str(int(high))
                entry = f"{lowstr}_{highstr}"
            else:
                raise ValueError(f"Invalid formatWithValue choice {formatWithValue}.")

        if type(entry) in [float, np.float64]:
            entry = f"{entry:0.1f}".replace(".", "p") if not entry.is_integer() else str(int(entry))
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
            return to_skip == -1 or curr_entry == to_skip or re.match(str(to_skip), str(curr_entry))

        for skipEntry in entries_to_skip:
            if all(match_entry(e,m) for e,m in zip(entry, skipEntry)):
                return True
        # If no matches were found for any of the entries_to_skip possibilities
        return False

    def skipEntryDictToArray(self, h, skipEntry, syst):
        nsyst = len(self.systematics[syst]["systAxes"])
        if "mirror" in h.axes.name:
            nsyst += 1

        if type(skipEntry) == dict:
            skipEntryArr = np.full(nsyst, -1, dtype=object)
            nother_ax = h.ndim-nsyst
            for k,v in skipEntry.items():
                if k not in h.axes.name:
                    raise ValueError(f"Invalid skipEntry expression {k} : {v}. Axis {k} is not in hist!")
                idx = h.axes.name.index(k)-nother_ax # Offset by the number of other axes, require that syst axes are the trailing ones
                if idx < 0:
                    raise ValueError(f"Invalid skip entry! Axis {k} was found in position {idx+nother_ax} of {h.ndim} axes, but {nsyst} syst axes were expected")
                skipEntryArr[idx] = v
            logger.debug(f"Expanded skipEntry for syst {syst} is {skipEntryArr}. Syst axes are {h.axes.name[-nsyst:]}")
        elif isinstance(skipEntry, (bool, int, float, str)):
            skipEntryArr = (skipEntry,)
        elif type(skipEntry) not in (np.array, list, tuple):
            raise ValueError(f"Unexpected format for skipEntry. Must be either dict, sequence, or scalar type. found {type(skipEntry)}")
        else:
            skipEntryArr = skipEntry

        if self.systematics[syst]["mirror"] and "mirror" not in h.axes.name and skipEntryArr[-1] == -1:
            skipEntryArr = skipEntryArr[:-1]

        if len(skipEntryArr) != nsyst:
            raise ValueError("skipEntry tuple must have the same dimensions as the number of syst axes. " \
                f"found {nsyst} systematics and len(skipEntry) = {len(skipEntryArr)}.") 

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
                nsyst = len(self.systematics[syst]["systAxes"])+self.systematics[syst]["mirror"]
                bin_lookup = np.array([ax.index(x.imag) for x, ax in 
                    zip(skipEntry, h.axes[-nsyst:]) if isinstance(x, complex)])
                # Need to loop here rather than using skip_arr.real because the dtype is object to allow strings
                skip_arr = np.array([a.real for a in skip_arr])
                skip_arr[to_lookup] += bin_lookup
            updated_skip.append(skip_arr)

        return updated_skip

    def systHists(self, hvar, syst, hnom):
        if syst == self.nominalName:
            return {self.nominalName : hvar}

        systInfo = self.systematics[syst] 
        systAxes = systInfo["systAxes"]
        systAxesLabels = systInfo.get("labelsByAxis", systAxes)

        # Jan: moved above the mirror action, as this action can cause mirroring
        if systInfo["action"]:
            if systInfo["actionRequiresNomi"]:
                hvar = systInfo["action"](hvar, hnom, **systInfo["actionArgs"])
            else:
                hvar = systInfo["action"](hvar, **systInfo["actionArgs"])
        if self.outfile:
            self.outfile.cd() # needed to restore the current directory in case the action opens a new root file

        axNames = systAxes[:]
        axLabels = systAxesLabels[:]
        if hvar.axes[-1].name == "mirror":
            axNames.append("mirror")
            axLabels.append("mirror")

        if not all([name in hvar.axes.name for name in axNames]):
            raise ValueError(f"Failed to find axis names {str(axNames)} in hist for syst {syst}. " \
                f"Axes in hist are {str(hvar.axes.name)}")

        # Converting to a list becasue otherwise if you print it for debugging you loose it
        entries = list(itertools.product(*[self.systIndexForAxis(hvar.axes[ax], flow=ax in systInfo["systAxesFlow"]) for ax in axNames]))

        if len(systInfo["outNames"]) == 0:
            skipEntries = None if "skipEntries" not in systInfo else self.expandSkipEntries(hvar, syst, systInfo["skipEntries"])
            for entry in entries:
                if skipEntries and self.excludeSystEntry(entry, skipEntries):
                    systInfo["outNames"].append("")
                else:
                    name = systInfo["baseName"]
                    fwv = systInfo["formatWithValue"]
                    if fwv:
                        if "mirror" in axLabels:
                            fwv.append(None)
                    name += "".join([self.systLabelForAxis(al, entry[i], hvar.axes[ax], fwv[i] if fwv else fwv) for i,(al,ax) in enumerate(zip(axLabels,axNames))])
                    if "systNameReplace" in systInfo and systInfo["systNameReplace"]:
                        for rep in systInfo["systNameReplace"]:
                            name = name.replace(*rep)
                            logger.debug(f"Replacement {rep} yields new name {name}")
                    if name and "systNamePrepend" in systInfo and systInfo["systNamePrepend"]:
                        name = systInfo["systNamePrepend"]+name
                    # Obviously there is a nicer way to do this...
                    if "Up" in name:
                        name = name.replace("Up", "")+"Up"
                    elif "Down" in name:
                        name = name.replace("Down", "")+"Down"
                    systInfo["outNames"].append(name)
            if not len(systInfo["outNames"]):
                raise RuntimeError(f"Did not find any valid variations for syst {syst}")

        variations = [hvar[{ax : binnum for ax,binnum in zip(axNames, entry)}] for entry in entries]
        if hvar.axes[-1].name == "mirror" and len(variations) == 2*len(systInfo["outNames"]):
            systInfo["outNames"] = [n + d for n in systInfo["outNames"] for d in ["Up", "Down"]]
        elif len(variations) != len(systInfo["outNames"]):
            logger.warning(f"The number of variations doesn't match the number of names for "
                f"syst {syst}. Found {len(systInfo['outNames'])} names and {len(variations)} variations.")

        return {name : var for name,var in zip(systInfo["outNames"], variations) if name}

    def getLogk(self, hvar, hnom, kfac=1., logkepsilon=math.log(1e-3)):
        # check if there is a sign flip between systematic and nominal
        _logk = kfac*np.log(hvar.values()/hnom.values())
        _logk_view = np.where(np.equal(np.sign(hnom.values()*hvar.values()),1), _logk, logkepsilon*np.ones_like(_logk))
        return _logk_view

    def symmetrize(self, var_map, hnom, symmetrize=None):
        if symmetrize is None:
            # nothing to do
            return var_map

        var_map_out = {}

        for var, hvar in var_map.items():
            if not np.all(np.isfinite(hvar.values())):
                raise RuntimeError(f"{len(hvar.values())-sum(np.isfinite(hvar.values()))} NaN or Inf values encountered in systematic {var}!")

            if symmetrize is not None and var.endswith("Up"):
                varbase = var.removesuffix("Up")

                varup = var
                vardown = varbase + "Down"

                hvarup = hvar
                hvardown = var_map[vardown]

                logkup = self.getLogk(hvarup, hnom)
                logkdown = -self.getLogk(hvardown, hnom)

                if symmetrize in ["conservative", "average"]:
                    if symmetrize=="conservative":
                        # symmetrize by largest magnitude of up and down variations
                        logk = np.where(np.abs(logkup) > np.abs(logkdown), logkup, logkdown)
                    elif symmetrize=="average":
                        # symmetrize by average of up and down variations
                        logk = 0.5*(logkup + logkdown)

                    # reuse histograms to avoid copies
                    # up and down variations are explicitly produced in this case
                    hvarup.values()[...] = hnom.values()*np.exp(logk)
                    hvardown.values()[...] = hnom.values()*np.exp(-logk)

                    var_map_out[varup] = hvarup
                    var_map_out[vardown] = hvardown

                elif symmetrize in ["linear", "quadratic"]:
                    # "linear" corresponds to a piecewise linear dependence of logk on theta
                    # while "quadratic" corresponds to a quadratic dependence and leads
                    # to a large variance
                    diff_fact = np.sqrt(3.) if symmetrize=="quadratic" else 1.

                    # split asymmetric variation into two symmetric variations
                    logkavg = 0.5*(logkup + logkdown)
                    logkdiff =  0.5*diff_fact*(logkup - logkdown)

                    varavg = varbase + "SymAvg"
                    vardiff = varbase + "SymDiff"

                    # reuse histograms to minimize copies
                    hvaravgup = hvarup
                    hvaravgdown = hvardown

                    hvardiffup = hvarup.copy()
                    hvardiffdown = hvardown.copy()

                    hvaravgup.values()[...] = hnom.values()*np.exp(logkavg)
                    hvaravgdown.values()[...] = hnom.values()*np.exp(-logkavg)

                    hvardiffup.values()[...] = hnom.values()*np.exp(logkdiff)
                    hvardiffdown.values()[...] = hnom.values()*np.exp(-logkdiff)

                    varavgup = varavg + "Up"
                    varavgdown = varavg + "Down"

                    var_map_out[varavgup] = hvaravgup
                    var_map_out[varavgdown] = hvaravgdown

                    vardiffup = vardiff + "Up"
                    vardiffdown = vardiff + "Down"

                    var_map_out[vardiffup] = hvardiffup
                    var_map_out[vardiffdown] = hvardiffdown
            elif symmetrize is not None and var.endswith("Down"):
                # this is already handled above
                pass
            else:
                var_map_out[var] = hvar

        return var_map_out


    def variationName(self, proc, name):
        if name == self.nominalName:
            return f"{self.histName}_{proc}"
        else:
            return f"{self.histName}_{proc}_{name}"

    def getBoostHistByCharge(self, h, q):
        return h[{self.charge_ax : h.axes[self.charge_ax].index(q) if q != "sum" else hist.sum}]

    def checkSysts(self, var_map, proc, thresh=0.25, skipSameSide=False, skipOneAsNomi=False):
        #if self.check_variations:
        var_names = set([name.replace("Up", "").replace("Down", "") for name in var_map.keys() if name])
        if len(var_names) != len(var_map.keys())/2:
            raise ValueError(f"Invalid syst names for process {proc}! Expected an up/down variation for each syst. "
                f"Found systs {var_names} and outNames {var_map.keys()}")
        # for wmass some systs are only expected to affect a reco charge, but the syst for other charge might still exist and be used
        # although one expects it to be same as nominal. The following check would trigger on this case with spurious warnings
        # so there is some customization based on what one expects to silent some noisy warnings

        for name in sorted(var_names):
            hnom = self.datagroups.groups[proc].hists[self.nominalName]
            up = var_map[name+"Up"]
            down = var_map[name+"Down"]
            nCellsWithoutOverflows = np.product(hnom.shape)

            checkFiniteBinValues(up, name+"Up")
            checkFiniteBinValues(down, name+"Down")

            if not skipSameSide:
                try:
                    up_relsign = np.sign(up.values(flow=False)-hnom.values(flow=False))
                except ValueError as e:
                    logger.error(f"Incompatible shapes between up {up.shape} and nominal {hnom.shape} for syst {name}")
                    raise e
                down_relsign = np.sign(down.values(flow=False)-hnom.values(flow=False))
                # protect against yields very close to nominal, for which it can be sign != 0 but should be treated as 0
                # was necessary for Fake and effStat
                vars_sameside = (up_relsign != 0) & (up_relsign == down_relsign) & np.logical_not(np.isclose(up.values(flow=False), hnom.values(flow=False), rtol=1e-07, atol=1e-08))
                perc_sameside = np.count_nonzero(vars_sameside)/nCellsWithoutOverflows 
                if perc_sameside > thresh:
                    logger.warning(f"{perc_sameside:.1%} bins are one sided for syst {name} and process {proc}!")
            # check variations are not same as nominal
            # it evaluates absolute(a - b) <= (atol + rtol * absolute(b))
            up_nBinsSystSameAsNomi = np.count_nonzero(np.isclose(up.values(flow=False), hnom.values(flow=False), rtol=1e-07, atol=1e-08))/nCellsWithoutOverflows
            down_nBinsSystSameAsNomi = np.count_nonzero(np.isclose(down.values(flow=False), hnom.values(flow=False), rtol=1e-06, atol=1e-08))/nCellsWithoutOverflows
            # check against 100% bins equal to nominal, the tolerances in np.isclose should already catch bad cases with 1.0 != 1.0 because of numerical imprecisions
            varEqNomiThreshold = 1.0
            if up_nBinsSystSameAsNomi >= varEqNomiThreshold or down_nBinsSystSameAsNomi >= varEqNomiThreshold:
                if not skipOneAsNomi or (up_nBinsSystSameAsNomi >= varEqNomiThreshold and down_nBinsSystSameAsNomi >= varEqNomiThreshold):
                    logger.warning(f"syst {name} has Up/Down variation with {up_nBinsSystSameAsNomi:.1%}/{down_nBinsSystSameAsNomi:.1%} of bins equal to nominal")


    def writeForProcess(self, h, proc, syst, check_systs=True):
        hnom = None
        systInfo = None
        if syst != self.nominalName:
            systInfo = self.systematics[syst]
            procDict = self.datagroups.getDatagroups()
            hnom = procDict[proc].hists[self.nominalName]
            if systInfo["mirror"]:
                h = hh.extendHistByMirror(h, hnom,
                                          downAsUp=systInfo["mirrorDownVarEqualToUp"],
                                          downAsNomi=systInfo["mirrorDownVarEqualToNomi"])            

        logger.info(f"   {syst} for process {proc}")
        var_map = self.systHists(h, syst, hnom)

        if syst != self.nominalName:
            if not systInfo["mirror"] and systInfo["symmetrize"] is not None:
                var_map = self.symmetrize(var_map, hnom, symmetrize = systInfo["symmetrize"])

            # since the list of variations may have been modified, we need to
            # resynchronize it
            systInfo["outNamesFinal"] = list(var_map.keys())

            if check_systs:
                self.checkSysts(var_map, proc,
                                skipSameSide=systInfo["mirrorDownVarEqualToUp"],
                                skipOneAsNomi=systInfo["mirrorDownVarEqualToNomi"])

        setZeroStatUnc = False
        if proc in self.noStatUncProcesses:
            logger.warning(f"Zeroing statistical uncertainty for process {proc}")
            setZeroStatUnc = True
        # this is a big loop a bit slow, but it might be mainly the hist->root conversion and writing into the root file
        for name, var in var_map.items():
            if name != "":
                self.writeHist(var, proc, name, setZeroStatUnc=setZeroStatUnc, hnomi=hnom)

    def loadPseudodataFakes(self, datagroups, forceNonzero=False):
        # get the nonclosure for fakes/multijet background from QCD MC
        datagroups.loadHistsForDatagroups(
            baseName=self.nominalName, syst=self.nominalName, label="syst",
            procsToRead=datagroups.groups.keys(),
            scaleToNewLumi=self.lumiScale,
            lumiScaleVarianceLinearly=self.lumiScaleVarianceLinearly,
            forceNonzero=forceNonzero,
            sumFakesPartial=False,
            applySelection=False 
            )
        procDict = datagroups.getDatagroups()
        gTruth = procDict["QCDTruth"]
        hTruth = gTruth.histselector.get_hist(gTruth.hists["syst"])

        # now load the nominal histograms
        self.datagroups.loadHistsForDatagroups(
            baseName=self.nominalName, syst=self.nominalName,
            procsToRead=self.datagroups.groups.keys(),
            label=self.nominalName, 
            scaleToNewLumi=self.lumiScale,
            lumiScaleVarianceLinearly=self.lumiScaleVarianceLinearly,
            forceNonzero=forceNonzero,
            sumFakesPartial=not self.simultaneousABCD)
        procDictFromNomi = self.datagroups.getDatagroups()

        if "QCD" not in procDict:
            # use truth MC as QCD
            logger.info(f"Have MC QCD truth {hTruth.sum()}")
            hFake = hTruth
        elif self.simultaneousABCD:
            # TODO: Make if work for simultaneous ABCD fit, apply the correction in the signal region (D) only
            raise NotImplementedError("The multijet closure test is not implemented for simultaneous ABCD fit")
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
                raise NotImplementedError(f"The multijet closure test is not implemented for arbitrary axes, the required axes are {histCorr.axes.name}")
            hFake = hh.multiplyHists(hFake, histCorr)

            # apply variances from hCorr to fakes to account for stat uncertainty
            hFakeNominal = procDictFromNomi[self.getFakeName()].hists[self.nominalName]
            hFakeNominal.variances(flow=True)[...] = hFake.variances(flow=True)
            procDictFromNomi[self.getFakeName()].hists[self.nominalName] = hFakeNominal

        # done, now sum all histograms
        hists_data = [procDictFromNomi[x].hists[self.nominalName] for x in self.predictedProcesses() if x != self.getFakeName()]
        hdata = hh.sumHists([*hists_data, hFake]) if len(hists_data)>0 else hFake

        return hdata

    def loadPseudodata(self, forceNonzero=False):
        datagroups = self.pseudodata_datagroups
        processes = [x for x in datagroups.groups.keys() if x != self.getDataName() and self.pseudoDataProcsRegexp.match(x)]
        processes = self.expandProcesses(processes)

        processesFromNomi = [x for x in datagroups.groups.keys() if x != self.getDataName() and not self.pseudoDataProcsRegexp.match(x)]
        hdatas = []
        for idx, pseudoData in enumerate(self.pseudoData):
            if pseudoData in ["closure", "truthMC"]:
                # pseudodata for fakes [using closure from QCD MC as correction, using QCD MC as prediction]
                if pseudoData == "truthMC":
                    datagroups.deleteGroup("QCD")
                hdatas.append(self.loadPseudodataFakes(datagroups, forceNonzero=forceNonzero))
                continue

            if pseudoData in ["simple", "extended1D", "extended2D"]:
                # pseudodata for fakes using data with different fake estimation, change the selection but still keep the nominal histogram
                datagroups.set_histselectors(datagroups.getNames(), self.nominalName, 
                    mode=pseudoData,
                    simultaneousABCD=self.simultaneousABCD,
                )
                syst=self.nominalName
            else:
                syst=pseudoData
 
            datagroups.loadHistsForDatagroups(
                baseName=self.nominalName, syst=syst, label=pseudoData,
                procsToRead=processes,
                scaleToNewLumi=self.lumiScale,
                lumiScaleVarianceLinearly=self.lumiScaleVarianceLinearly,
                forceNonzero=forceNonzero,
                sumFakesPartial=not self.simultaneousABCD)
            procDict = datagroups.getDatagroups()
            hists = [procDict[proc].hists[pseudoData] for proc in processes if proc not in processesFromNomi]

            # now add possible processes from nominal
            logger.warning(f"Making pseudodata summing these processes: {processes}")
            if len(processesFromNomi):
                logger.warning(f"These processes are taken from nominal datagroups: {processesFromNomi}")
                datagroupsFromNomi = self.datagroups
                datagroupsFromNomi.loadHistsForDatagroups(
                    baseName=self.nominalName, syst=self.nominalName,
                    procsToRead=processesFromNomi, 
                    label=pseudoData,
                    scaleToNewLumi=self.lumiScale,
                    lumiScaleVarianceLinearly=self.lumiScaleVarianceLinearly,
                    forceNonzero=forceNonzero,
                    sumFakesPartial=not self.simultaneousABCD)
                procDictFromNomi = datagroupsFromNomi.getDatagroups()
                hists.extend([procDictFromNomi[proc].hists[pseudoData] for proc in processesFromNomi])
            # done, now sum all histograms
            hdata = hh.sumHists(hists)
            if self.pseudoDataAxes[idx] is None:
                extra_ax = [ax for ax in hdata.axes.name if ax not in self.fit_axes]
                if len(extra_ax) > 0 and extra_ax[-1] in ["vars", "systIdx", "tensor_axis_0"]:
                    self.pseudoDataAxes[idx] = extra_ax[-1]
                    logger.info(f"Setting pseudoDataSystAx[{idx}] to {extra_ax[-1]}")
                    if self.pseudoDataIdxs[idx] == [None]:
                        self.pseudoDataIdxs[idx] = [0]
                        logger.info(f"Setting pseudoDataIdxs[{idx}] to {[0]}")
            if self.pseudoDataAxes[idx] is not None and self.pseudoDataAxes[idx] not in hdata.axes.name:
                raise RuntimeError(f"Pseudodata axis {self.pseudoDataAxes[idx]} not found in {hdata.axes.name}.")
            hdatas.append(hdata)

        return hdatas

    def addPseudodata(self):
        if len(self.pseudoData) > 1 or len(self.pseudoDataIdxs) > 1:
            raise RuntimeError(f"Mutliple pseudo data sets from different histograms or indices is not supported in the root writer.")
        hdata = self.loadPseudodata()[0]
        pseudoData = self.pseudoData[0]
        pseudoDataAxis = self.pseudoDataAxes[0]
        pseudoDataIdx = self.pseudoDataIdxs[0] if self.pseudoDataIdxs[0] is not None else 0
        if pseudoDataAxis is not None:
            hdata = hdata[{pseudoDataAxis : pseudoDataIdx }] 
        self.writeHist(hdata, self.getDataName(), pseudoData+"_sum")
        procDict = self.pseudodata_datagroups.getDatagroups()
        if self.getFakeName() in procDict:
            self.writeHist(procDict[self.getFakeName()].hists[pseudoData], self.getFakeName(), pseudoData+"_sum")

    def writeForProcesses(self, syst, processes, label, check_systs=True):
        logger.info("-"*50)
        logger.info(f"Preparing to write systematic {syst}")
        for process in processes:
            hvar = self.datagroups.groups[process].hists[label]
            if not hvar:
                raise RuntimeError(f"Failed to load hist for process {process}, systematic {syst}")
            self.writeForProcess(hvar, process, syst, check_systs=check_systs)
        if syst != self.nominalName:
            self.fillCardWithSyst(syst)

    def setOutfile(self, outfile):
        if type(outfile) == str:
            if self.skipHist:
                self.outfile = outfile # only store name, file will not be used and doesn't need to be opened
            else:
                self.outfile = ROOT.TFile(outfile, "recreate")
                self.outfile.cd()
        else:
            self.outfile = outfile
            self.outfile.cd()

    def setOutput(self, outfolder, basename):
        self.outfolder = outfolder
        if not os.path.isdir(self.outfolder):
            os.makedirs(self.outfolder)
        suffix = f"_{self.datagroups.flavor}" if self.datagroups.flavor else ""
        if self.xnorm:
            suffix += '_xnorm'

        self.cardName = (f"{self.outfolder}/{basename}_{{chan}}{suffix}.txt")
        self.setOutfile(os.path.abspath(f"{self.outfolder}/{basename}CombineInput{suffix}.root"))
            
    def writeOutput(self, args=None, forceNonzero=False, check_systs=True):
        self.datagroups.loadHistsForDatagroups(
            baseName=self.nominalName, syst=self.nominalName,
            procsToRead=self.datagroups.groups.keys(),
            label=self.nominalName, 
            scaleToNewLumi=self.lumiScale,
            lumiScaleVarianceLinearly=self.lumiScaleVarianceLinearly,
            forceNonzero=forceNonzero,
            sumFakesPartial=not self.simultaneousABCD)
        
        self.writeForProcesses(self.nominalName, processes=self.datagroups.groups.keys(), label=self.nominalName, check_systs=check_systs)
        self.loadNominalCard()

        if not self.real_data and not self.xnorm:
            # If real data is not explicitly requested, use pseudodata instead (but still store read data in root writer)
            self.setPseudodata([self.nominalName])
        if self.pseudoData and not self.xnorm:
            self.addPseudodata()

        self.writeLnNSystematics()
        for syst in self.systematics.keys():
            if self.isExcludedNuisance(syst): continue
            systMap = self.systematics[syst]
            systName = syst if not systMap["name"] else systMap["name"]
            processes = systMap["processes"]
            # Needed to avoid always reading the variation for the fakes, even for procs not specified
            forceToNominal=[x for x in self.datagroups.getProcNames() if x not in 
                self.datagroups.getProcNames([p for g in processes for p in self.expandProcesses(g) if p != self.getFakeName()])]
            self.datagroups.loadHistsForDatagroups(
                self.nominalName, systName, label="syst",
                procsToRead=processes, 
                forceNonzero=forceNonzero and systName != "qcdScaleByHelicity",
                preOpMap=systMap["preOpMap"], preOpArgs=systMap["preOpArgs"], applySelection=systMap["applySelection"],
                scaleToNewLumi=self.lumiScale,
                lumiScaleVarianceLinearly=self.lumiScaleVarianceLinearly,
                forceToNominal=forceToNominal,
                sumFakesPartial=not self.simultaneousABCD
            )
            self.writeForProcesses(syst, label="syst", processes=processes, check_systs=check_systs)

        output_tools.writeMetaInfoToRootFile(self.outfile, exclude_diff='notebooks', args=args)
        if self.skipHist:
            logger.info("Histograms will not be written because 'skipHist' flag is set to True")
        logger.info(f"Writing text/root cards to {self.outfile}")
        self.writeCard()
        
    def match_str_axis_entries(self, str_axis, match_re):
        return [x for x in str_axis if any(re.match(r, x) for r in match_re)]

    def writeCard(self):
        for chan in self.channels:
            with open(self.cardName.format(chan=chan), "w") as card:
                card.write(self.cardContent[chan])
                card.write("\n")
                card.write(self.cardGroups[chan])
                card.write(self.writePOISumGroupToText())

    def addSystToGroup(self, groupName, chan, members, groupLabel="group"):
        group_expr = f"{groupName} {groupLabel} ="
        if group_expr in self.cardGroups[chan]:
            idx = self.cardGroups[chan].index(group_expr)+len(group_expr)
            self.cardGroups[chan] = self.cardGroups[chan][:idx] + " " + members + self.cardGroups[chan][idx:]
        else:
            self.cardGroups[chan] += f"\n{group_expr} {members}"                                              

    def addPOISumGroups(self, gen_axes=None, additional_axes=None, genCharge=None):
        if gen_axes is None:
            gen_axes = self.datagroups.gen_axes_names.copy()
        if additional_axes is not None:
            gen_axes += additional_axes
        # if only one or none gen axes, it is already included as main POI and no sumGroups are needed
        if len(gen_axes) <= 1:
            return
        # make a sum group for each gen axis
        axes_combinations = gen_axes
        # also include combinations of axes in case there are more than 2 axes
        for n in range(2, len(self.datagroups.gen_axes_names)):
            axes_combinations += [k for k in itertools.combinations(self.datagroups.gen_axes_names, n)]
        for axes in axes_combinations:
            logger.debug(f"Add sum group for {axes}{' with ' + genCharge if genCharge else ''}")

            if isinstance(axes, str):
                axes = [axes]

            pois_axis = [x for x in self.unconstrainedProcesses if all([a in x for a in axes])]

            # in case of multiple base processes (e.g. in simultaneous unfoldings) loop over all base processes
            base_processes = set(map(lambda x: x.split("_")[0], pois_axis))
            for base_process in base_processes:
                pois = [x for x in pois_axis if base_process in x.split("_")]

                sum_groups = set(["_".join([a + p.split(a)[1].split("_")[0] for a in axes]) for p in pois])

                for sum_group in sorted(sum_groups):
                    membersList = [p for p in pois if all([g in p.split("_") for g in sum_group.split("_")])]
                    sum_group_name = f"{base_process}_{sum_group}"
                    if genCharge is not None:                
                        membersList = list(filter(lambda x: genCharge in x, membersList))
                        sum_group_name += f"_{genCharge}"
                    if len(membersList):                            
                        self.addPOISumGroup(sum_group_name, membersList)
                        
    def addPOISumGroup(self, groupName, members):
        if groupName in self.cardSumGroups:
            self.cardSumGroups[groupName].append(members)
        else:
            self.cardSumGroups[groupName] = members

    def writePOISumGroupToText(self, groupLabel="sumGroup"):
        # newName sumGroup = poi_bin1 poi_bin2 poi_bin3
        text = ""
        for groupName, membersList in self.cardSumGroups.items():
            members = " ".join(membersList)
            logger.debug(f"Write POI sum group {groupName} with members {members}")
            text += f"\n{groupName} {groupLabel} = {members}"
        return text
        
    def writeLnNSystematics(self):
        nondata = self.predictedProcesses()
        nondata_chan = {chan: nondata.copy() for chan in self.channels}
        for chan in self.excludeProcessForChannel.keys():
            nondata_chan[chan] = list(filter(lambda x: not self.excludeProcessForChannel[chan].match(x), nondata))
        # exit this function when a syst is applied to no process (can happen when some are excluded)
        for name,info in self.lnNSystematics.items():
            if self.isExcludedNuisance(name): continue
            if all(x not in info["processes"] for x in nondata):
                logger.warning(f"Skipping syst {name}, procs to apply it to would be {info['processes']}, and predicted processes are {nondata}")
                return
            group = info["group"]
            groupFilter = info["groupFilter"]
            for chan in self.channels:
                nameChan = name.replace("CHANNEL",chan)
                include = [(str(info["size"]) if x in info["processes"] else "-").ljust(self.procColumnsSpacing) for x in nondata_chan[chan]]
                self.cardContent[chan] += f'{nameChan.ljust(self.spacing)} lnN{" "*(self.systTypeSpacing-2)} {"".join(include)}\n'
                if group and len(list(filter(groupFilter, [nameChan]))):
                    self.addSystToGroup(group, chan, nameChan)

    def fillCardWithSyst(self, syst):
        # note: this function doesn't act on all systematics all at once
        # but rather it deals with all those coming from each call to CardTool.addSystematics
        systInfo = self.systematics[syst]
        scale = systInfo["scale"]
        procs = systInfo["processes"]
        group = systInfo["group"]
        groupFilter = systInfo["groupFilter"]
        label = "group" if not systInfo["noi"] else "noiGroup"
        nondata = self.predictedProcesses()
        nondata_chan = {chan: nondata.copy() for chan in self.channels}
        for chan in self.excludeProcessForChannel.keys():
            nondata_chan[chan] = list(filter(lambda x: not self.excludeProcessForChannel[chan].match(x), nondata))
            
        names = [x[:-2] if "Up" in x[-2:] else (x[:-4] if "Down" in x[-4:] else x) 
                    for x in filter(lambda x: x != "", systInfo["outNamesFinal"])]
        # exit this function when a syst is applied to no process (can happen when some are excluded)
        if all(x not in procs for x in nondata):
            return 0
        
        include_chan = {}
        for chan in nondata_chan.keys():
            include_chan[chan] = [(str(scale) if x in procs else "-").ljust(self.procColumnsSpacing) for x in nondata_chan[chan]]
            # remove unnecessary trailing spaces after the last element
            include_chan[chan][-1] = include_chan[chan][-1].rstrip()
                
        shape = "shapeNoConstraint" if systInfo["noConstraint"] else "shape"
            
        splitGroupDict = systInfo["splitGroup"]

        # Deduplicate while keeping order
        systNames = list(dict.fromkeys(names))

        systnamesPruned = [s for s in systNames if not self.isExcludedNuisance(s)]
        systNames = systnamesPruned[:]
        for chan in self.channels:
            systNamesChan = [x.replace("CHANNEL",chan) for x in systNames]
            for systname in systNamesChan:
                systShape = shape
                include_line = include_chan[chan]
                if systInfo["customizeNuisanceAttributes"]:
                    for regexpCustom in systInfo["customizeNuisanceAttributes"]:
                        if re.match(regexpCustom, systname):
                            keys = list(systInfo["customizeNuisanceAttributes"][regexpCustom].keys())
                            if "scale" in keys:
                                include_line = [(str(systInfo["customizeNuisanceAttributes"][regexpCustom]["scale"]) if x in procs else "-").ljust(self.procColumnsSpacing) for x in nondata_chan[chan]]
                                include_line[-1] = include_line[-1].rstrip()
                            if "shapeType" in keys:
                                systShape = systInfo["customizeNuisanceAttributes"][regexpCustom]["shapeType"]
                self.cardContent[chan] += f"{systname.ljust(self.spacing)} {systShape.ljust(self.systTypeSpacing)} {''.join(include_line)}\n"
            # unlike for LnN systs, here it is simpler to act on the list of these systs to form groups, rather than doing it syst by syst 
            if group:
                systNamesForGroupPruned = systNamesChan[:]
                systNamesForGroup = list(systNamesForGroupPruned if not groupFilter else filter(groupFilter, systNamesForGroupPruned))
                if len(systNamesForGroup):
                    for subgroup, matchre in splitGroupDict.items():
                        systNamesForSubgroup = list(filter(lambda x: matchre.match(x),systNamesForGroup))
                        if len(systNamesForSubgroup):
                            members = " ".join(systNamesForSubgroup)
                            self.addSystToGroup(subgroup, chan, members, groupLabel=label)

    def setUnconstrainedProcs(self, procs):
        self.unconstrainedProcesses = procs

    def processLabels(self, procs=None):
        nondata = np.array(self.predictedProcesses() if procs is None else procs)
        labels = np.arange(len(nondata))+1
        issig = np.isin(nondata, self.unconstrainedProcesses)
        labels[issig] = -np.arange(np.count_nonzero(issig))-1
        return labels

    def loadNominalCard(self):
        procs = self.predictedProcesses()
        nprocs = len(procs)
        for chan in self.channels:
            if chan in self.excludeProcessForChannel.keys():
                procs = list(filter(lambda x: not self.excludeProcessForChannel[chan].match(x), self.predictedProcesses()))
                nprocs = len(procs)
            args = {
                "channel" :  chan,
                "channelPerProc" : chan.ljust(self.procColumnsSpacing)*nprocs,
                "processes" : " ".join([x.ljust(self.procColumnsSpacing) for x in procs]),
                "labels" : "".join([str(x).ljust(self.procColumnsSpacing) for x in self.processLabels(procs)]),
                # Could write out the proper normalizations pretty easily
                "rates" : "-1".ljust(self.procColumnsSpacing)*nprocs,
                "inputfile" : self.outfile if type(self.outfile) == str  else self.outfile.GetName(),
                "dataName" : self.getDataName(),
                "histName" : self.histName,
                "pseudodataHist" : f"{self.histName}_{self.getDataName()}_{self.pseudoData[0]}_sum" if self.pseudoData else f"{self.histName}_{self.getDataName()}"
            }
            if not self.absolutePathShapeFileInCard:
                # use the relative path because absolute paths are slow in text2hdf5.py conversion
                args["inputfile"] = os.path.basename(args["inputfile"])

            self.cardContent[chan] = output_tools.readTemplate(self.nominalTemplate, args)
            self.cardGroups[chan] = ""
            
    def writeHistByCharge(self, h, name):
        for charge in self.channels:
            q = self.chargeIdDict[charge]["val"]
            hout = narf.hist_to_root(self.getBoostHistByCharge(h, q))
            hout.SetName(name.replace("CHANNEL",charge)+f"_{charge}")
            hout.Write()
        
    def writeHistWithCharges(self, h, name):
        hout = narf.hist_to_root(h)
        hout.SetName(f"{name}_{self.channels[0]}" if self.channels else name)
        hout.Write()
    
    def writeHist(self, h, proc, syst, setZeroStatUnc=False, hnomi=None):
        if self.skipHist:
            return
        if self.fit_axes:
            axes = self.fit_axes[:]
            if self.simultaneousABCD and not self.xnorm:
                if common.passIsoName not in axes:
                    axes.append(common.passIsoName)
                if self.nameMT not in axes:
                    axes.append(self.nameMT)
            # don't project h into itself when axes to project are all axes
            if any (ax not in h.axes.name for ax in axes):
                logger.error("Request to project some axes not present in the histogram")
                raise ValueError(f"Histogram has {h.axes.name} but requested axes for projection are {axes}")
            if len(axes) < len(h.axes.name):
                logger.debug(f"Projecting {h.axes.name} into {axes}")
                h = h.project(*axes)

        if self.unroll:
            logger.debug(f"Unrolling histogram")
            h = hh.unrolledHist(h, axes)

        if not self.nominalDim:
            self.nominalDim = h.ndim
            if self.nominalDim-self.writeByCharge > 3:
                raise ValueError(f"Cannot write hists with > 3 dimensions as combinetf does not accept THn. Axes are {h.axes.name}")

        if h.ndim != self.nominalDim:
            raise ValueError(f"Histogram {proc}/{syst} does not have the correct dimensions. Found {h.ndim}, expected {self.nominalDim}")

        if setZeroStatUnc:
            h.variances(flow=True)[...] = 0.

        # make sub directories for each process or return existing sub directory
        directory = self.outfile.mkdir(proc, proc, True)
        directory.cd()

        name = self.variationName(proc, syst)

        if self.writeByCharge:
            self.writeHistByCharge(h, name)
        else:
            self.writeHistWithCharges(h, name)
        self.outfile.cd()
