from collections import OrderedDict
from wremnants import histselections as sel
from wremnants.combine_helpers import setSimultaneousABCD
from utilities import boostHistHelpers as hh, common, output_tools, logging
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

logger = logging.child_logger(__name__)

def notImplemented(operation="Unknown"):
    raise NotImplementedError(f"Required operation '{operation}' is not implemented!")

class CardTool(object):
    def __init__(self, outpath="./", xnorm=False, ABCD=False):
    
        self.skipHist = False # don't produce/write histograms, file with them already exists
        self.outfile = None
        self.systematics = {}
        self.lnNSystematics = {}
        self.predictedProcs = []
        self.fakeEstimate = None
        self.channels = ["plus", "minus"]
        self.cardContent = {}
        self.cardGroups = {}
        self.cardSumGroups = {} # POI sum groups
        self.nominalTemplate = f"{pathlib.Path(__file__).parent}/../scripts/combine/Templates/datacard.txt"
        self.spacing = 28
        self.systTypeSpacing = 16
        self.procColumnsSpacing = 30
        self.nominalName = "nominal"
        self.datagroups = None
        self.pseudodata_datagroups = None
        self.unconstrainedProcesses = None
        self.noStatUncProcesses = []
        self.buildHistNameFunc = None
        self.histName = "x"
        self.nominalDim = None
        self.pseudoData = None
        self.pseudoDataIdx = None
        self.pseudoDataProcsRegexp = None
        self.excludeSyst = None
        self.writeByCharge = True
        self.unroll = False # unroll final histogram before writing to root
        self.keepSyst = None # to override previous one with exceptions for special cases
        self.lumiScale = 1.
        self.project = None
        self.fakerateAxes = ["pt", "eta", "charge"]
        self.xnorm = xnorm
        self.ABCD = ABCD
        self.absolutePathShapeFileInCard = False
        self.excludeProcessForChannel = {} # can be used to exclue some POI when runnig a specific name (use case, force gen and reco charges to match)
        self.signalProcesses = []
        self.singleVBackground = []
        self.chargeIdDict = {"minus" : {"val" : -1, "id" : "q0", "badId" : "q1"},
                             "plus"  : {"val" : 1., "id" : "q1", "badId" : "q0"},
                             "inclusive" : {"val" : "sum", "id" : "none", "badId" : None},
                             }
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
            
    def setProjectionAxes(self, project):
        self.project = project

    def setFakerateAxes(self, fakerate_axes=["pt", "eta", "charge"]):
        self.fakerateAxes = fakerate_axes
        
    def getFakerateIntegrationAxes(self):
        return [x for x in self.project if x not in self.fakerateAxes]

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
    
    def setLumiScale(self, lumiScale):
        self.lumiScale = lumiScale

    def setAbsolutePathShapeInCard(self, setRelative=False):
        self.absolutePathShapeFileInCard = False if setRelative else True
        
    def getProcsNoStatUnc(self):
        return self.noStatUncProcesses
        
    ## Functions to customize systs to be added in card, mainly for tests
    def setCustomSystForCard(self, exclude=None, keep=None):
        if exclude: self.excludeSyst = re.compile(exclude)
        if keep:    self.keepSyst    = re.compile(keep)
        
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

    def setPseudodata(self, pseudodata, idx = 0, pseudoDataProcsRegexp=".*"):
        self.pseudoData = pseudodata
        self.pseudoDataIdx = idx if not idx.isdigit() else int(idx)
        self.pseudoDataProcsRegexp = re.compile(pseudoDataProcsRegexp)
        
    # Needs to be increased from default for long proc names
    def setSpacing(self, spacing):
        self.spacing = spacing
        
    def setProcColumnsSpacing(self, spacing):
        self.procColumnsSpacing = spacing

    def setDatagroups(self, datagroups, resetGroups=False):
        self.datagroups = datagroups
        self.unconstrainedProcesses = datagroups.unconstrainedProcesses
        if self.nominalName:
            self.datagroups.setNominalName(self.nominalName)
        
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

    def addLnNSystematic(self, name, size, processes, group=None, groupFilter=None):
        if not self.isExcludedNuisance(name):
            self.lnNSystematics.update({name : {"size" : size,
                                                "processes" : self.expandProcesses(processes),
                                                "group" : group,
                                                "groupFilter" : groupFilter}
            })

    # action will be applied to the sum of all the individual samples contributing, actionMap should be used
    # to apply a separate action per process. this is needed for example for the scale uncertainty split
    # by pt or helicity
    # action takes place after mirroring
    # use doActionBeforeMirror to do something before it instead (so the mirroring will act on the modified histogram)
    # decorrelateByBin is to customize eta-pt decorrelation: pass dictionary with {axisName: [bin edges]}
    def addSystematic(self, name, systAxes=[], outNames=None, skipEntries=None, labelsByAxis=None, 
                      baseName="", mirror=False, mirrorDownVarEqualToUp=False, mirrorDownVarEqualToNomi=False,
                      scale=1, processes=None, group=None, noi=False, noConstraint=False, noProfile=False,
                      action=None, doActionBeforeMirror=False, actionArgs={}, actionMap={},
                      systNameReplace=[], systNamePrepend=None, groupFilter=None, passToFakes=False,
                      rename=None, splitGroup={}, decorrelateByBin={},
                      ):
        # note: setting Up=Down seems to be pathological for the moment, it might be due to the interpolation in the fit
        # for now better not to use the options, although it might be useful to keep it implemented
        if mirrorDownVarEqualToUp or mirrorDownVarEqualToNomi:
            raise ValueError("mirrorDownVarEqualToUp and mirrorDownVarEqualToNomi currently lead to pathological results in the fit, please keep them False")
        
        if isinstance(processes, str):
            processes = [processes]
        # Need to make an explicit copy of the array before appending
        procs_to_add = [x for x in (self.allMCProcesses() if processes is None else processes)]
        procs_to_add = self.expandProcesses(procs_to_add)
        if passToFakes and self.getFakeName() not in procs_to_add and not self.ABCD:
            procs_to_add.append(self.getFakeName())

        if not mirror and (mirrorDownVarEqualToUp or mirrorDownVarEqualToNomi):
            raise ValueError("mirrorDownVarEqualToUp and mirrorDownVarEqualToNomi requires mirror=True")

        if mirrorDownVarEqualToUp and mirrorDownVarEqualToNomi:
            raise ValueError("mirrorDownVarEqualToUp and mirrorDownVarEqualToNomi cannot be both True")
            
        if action and actionMap:
            raise ValueError("Only one of action and actionMap args are allowed")

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
                "baseName" : baseName,
                "processes" : procs_to_add,
                "systAxes" : systAxes,
                "labelsByAxis" : systAxes if not labelsByAxis else labelsByAxis,
                "group" : group,
                "noi": noi,
                "groupFilter" : groupFilter,
                "splitGroup" : splitGroup if len(splitGroup) else {group : ".*"}, # dummy dictionary if splitGroup=None, to allow for uniform treatment
                "scale" : scale,
                "mirror" : mirror,
                "mirrorDownVarEqualToUp" : mirrorDownVarEqualToUp,
                "mirrorDownVarEqualToNomi" : mirrorDownVarEqualToNomi,
                "action" : action,
                "doActionBeforeMirror" : doActionBeforeMirror,
                "actionMap" : actionMap,
                "actionArgs" : actionArgs,
                "systNameReplace" : systNameReplace,
                "noConstraint" : noConstraint,
                "noProfile" : noProfile,
                "skipEntries" : [] if not skipEntries else skipEntries,
                "name" : name,
                "decorrByBin": decorrelateByBin,
                "systNamePrepend" : systNamePrepend,
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
            fakerateIntegrationAxes=self.getFakerateIntegrationAxes())
        return self.datagroups.getDatagroups()[proc].hists["syst"]
        
    def setMirrorForSyst(self, syst, mirror=True):
        self.systematics[syst]["mirror"] = mirror

    def systLabelForAxis(self, axLabel, entry, axis):
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
        return axLabel+str(entry)

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
        if self.systematics[syst]["mirror"]:
            nsyst += 1

        if type(skipEntry) == dict:
            skipEntryArr = np.full(nsyst, -1, dtype=object)
            nother_ax = h.ndim-nsyst
            for k,v in skipEntry.items():
                idx = h.axes.name.index(k)-nother_ax # Offset by the number of other axes, require that syst axes are the trailing ones
                skipEntryArr[idx] = v
            logger.debug(f"Expanded skipEntry for syst {syst} is {skipEntryArr}. Syst axes are {h.axes.name[-nsyst:]}")
        elif type(skipEntry) not in (np.array, list, tuple):
            raise ValueError(f"Unexpected format for skipEntry. Must be either dict or sequence. found {type(skipEntry)}")
        else:
            skipEntryArr = skipEntry

        if len(skipEntryArr) != nsyst:
            raise ValueError("skipEntry tuple must have the same dimensions as the number of syst axes. " \
                f"found {nsyst} systematics and len(skipEntry) = {len(skipEntry)}.") 

        return skipEntryArr

    def expandSkipEntries(self, h, syst, skipEntries):
        updated_skip = []
        for skipEntry in skipEntries:
            skipEntry = self.skipEntryDictToArray(h, skipEntry, syst)
            # The lookup is handled by passing an imaginary number,
            # so detect these and then call the bin lookup on them
            # np.iscomplex returns false for 0.j, but still want to detect that
            to_lookup = np.array([isinstance(x, complex) for x in skipEntry])
            skip_arr = np.array(skipEntry)
            if to_lookup.any():
                nsyst = len(self.systematics[syst]["systAxes"])+self.systematics[syst]["mirror"]
                bin_lookup = np.array([ax.index(x.imag) for x, ax in 
                    zip(skipEntry, h.axes[-nsyst:]) if isinstance(x, complex)])
                # Need to loop here rather than using skip_arr.real because the dtype is object to allow strings
                skip_arr = np.array([a.real for a in skip_arr])
                skip_arr[to_lookup] += bin_lookup
            updated_skip.append(skip_arr)

        return updated_skip

    def systHists(self, hvar, syst):
        if syst == self.nominalName:
            return {self.nominalName : hvar}

        systInfo = self.systematics[syst] 
        systAxes = systInfo["systAxes"]
        systAxesLabels = systInfo.get("labelsByAxis", systAxes)

        # Jan: moved above the mirror action, as this action can cause mirroring
        if systInfo["action"] and not systInfo["doActionBeforeMirror"]:
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

        axes = [hvar.axes[ax] for ax in axNames]

        # Converting to a list becasue otherwise if you print it for debugging you loose it
        entries = list(itertools.product(*[[x for x in ax] if type(ax) == hist.axis.StrCategory else range(ax.size) for ax in axes]))

        if len(systInfo["outNames"]) == 0:
            skipEntries = None if "skipEntries" not in systInfo else self.expandSkipEntries(hvar, syst, systInfo["skipEntries"])
            for entry in entries:
                if skipEntries and self.excludeSystEntry(entry, skipEntries):
                    systInfo["outNames"].append("")
                else:
                    name = systInfo["baseName"]
                    name += "".join([self.systLabelForAxis(al, entry[i], ax) for i,(al,ax) in enumerate(zip(axLabels,axes))])
                    if "systNameReplace" in systInfo and systInfo["systNameReplace"]:
                        for rep in systInfo["systNameReplace"]:
                            name = name.replace(*rep)
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

    def variationName(self, proc, name):
        if name == self.nominalName:
            return f"{self.histName}_{proc}"
        else:
            return f"{self.histName}_{proc}_{name}"

    def getBoostHistByCharge(self, h, q):
        return h[{"charge" : h.axes["charge"].index(q) if q != "sum" else hist.sum}]

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
            # varEqNomiThreshold = 1.0 - 5./CellsWithoutOverflows # at least 5 bins different, but sensible choice depends on how many cells we have,
            # perhaps just better to check against 100%, the tolerances in np.isclose should already catch bad cases with 1.0 != 1.0 because of numerical imprecisions
            varEqNomiThreshold = 1.0
            if up_nBinsSystSameAsNomi >= varEqNomiThreshold or down_nBinsSystSameAsNomi >= varEqNomiThreshold:
                if not skipOneAsNomi or (up_nBinsSystSameAsNomi >= varEqNomiThreshold and down_nBinsSystSameAsNomi >= varEqNomiThreshold):
                    logger.warning(f"syst {name} has Up/Down variation with {up_nBinsSystSameAsNomi:.1%}/{down_nBinsSystSameAsNomi:.1%} of bins equal to nominal")
                    
    def makeDecorrelatedSystNuisances(self, systNames, dcbb={}):
        # NOTE: the dictionary is supposed to have only a single key
        for k in dcbb.keys():
            systNamesTmp = []
            decorrSystLabel = dcbb[k]["label"]
            if k == "xy":
                for ix in range(len(dcbb[k]["edges"][0])-1):
                    for iy in range(len(dcbb[k]["edges"][1])-1):
                        systNamesTmp.extend([f"{x}_{decorrSystLabel[0]}{ix}{decorrSystLabel[1]}{iy}" for x in systNames])
            else:
                for ik in range(len(dcbb[k]["edges"])-1):
                    systNamesTmp.extend([f"{x}_{decorrSystLabel}{ik}" for x in systNames])
            return systNamesTmp
            #logger.error(f"systNames = {systNames}")

    def makeDecorrelatedSystHistograms(self, h, hnomi, name, decorrByBinDict):
        # s = hist.tag.Slicer()
        # TODO: using an alternative version using boost directly may be faster, but for now this is just for testing
        # decorrByBinDict is a dictionary as
        # decorrByBinDict = {"x": {"label" : "eta",
        #                          "edges" : [round(-2.4+i*0.4,1) for i in range(13)],}
        #                   }
        # for a 2D decorrelation the values of each key are arrays of size 2
        # decorrByBinDict = {"xy": {"label" : ["eta", "pt"],
        #                          "edges" : [ [etaEdges], [ptEdges] ]}
        #                   }
        # TODO: could also do charge decorrelation by using the z axis if present
        ret = {}
        hnomiroot = narf.hist_to_root(hnomi)
        hsystroot = narf.hist_to_root(h)
        for ax in decorrByBinDict.keys():
            decorrDict = decorrByBinDict[ax]
            decorrSystLabel = decorrDict["label"]
            logger.info(f"Decorrelating syst {name} by {ax} bins using {decorrSystLabel}")
            if ax != "xy":
                for ibin in range(len(decorrDict["edges"]) -1):
                    upDown = "Up" if name.endswith("Up") else "Down" if name.endswith("Down") else ""
                    newname = name[:-len(upDown)] if len(upDown) else name[:]
                    newname = f"{newname}_{decorrSystLabel}{ibin}{upDown}"
                    logger.debug(f"Decorrelating syst {name} by {ax} bins: preparing new histogram {newname}")
                    # FIXME: do not assume we can only have eta and pt axes (or eta-pt when implemented)
                    # bin ID are retrieved adding some small constants to bin edges, low/high edge is increased/decreased
                    if ax == "x":
                        xbinLow = hnomiroot.GetXaxis().FindFixBin(decorrDict["edges"][ibin]+0.001)
                        xbinHigh = hnomiroot.GetXaxis().FindFixBin(decorrDict["edges"][ibin+1]-0.001)
                        ybinLow = 1
                        ybinHigh = hnomiroot.GetNbinsY()
                    elif ax == "y":
                        xbinLow = 1
                        xbinHigh = hnomiroot.GetNbinsX()
                        ybinLow = hnomiroot.GetYaxis().FindFixBin(decorrDict["edges"][ibin]+0.001)
                        ybinHigh = hnomiroot.GetYaxis().FindFixBin(decorrDict["edges"][ibin+1]-0.001)
                    hsystrootDecorr = copy.deepcopy(hnomiroot.Clone(newname))
                    ROOT.wrem.fillTH3fromTH3part(hsystrootDecorr, hsystroot,
                                                 xbinLow, ybinLow, 1,
                                                 xbinHigh, ybinHigh, hnomiroot.GetNbinsZ())
                    ret[newname] = narf.root_to_hist(hsystrootDecorr, axis_names = hnomi.axes.name)
            else:
                edgesX = decorrDict["edges"][0]
                edgesY = decorrDict["edges"][1]
                for ix in range(len(edgesX)-1):
                    for iy in range(len(edgesY)-1):
                        xbinLow = hnomiroot.GetXaxis().FindFixBin(edgesX[ix]+0.001)
                        xbinHigh = hnomiroot.GetXaxis().FindFixBin(edgesX[ix+1]-0.001)
                        ybinLow = hnomiroot.GetYaxis().FindFixBin(edgesY[iy]+0.001)
                        ybinHigh = hnomiroot.GetYaxis().FindFixBin(edgesY[iy+1]-0.001)
                        upDown = "Up" if name.endswith("Up") else "Down" if name.endswith("Down") else ""
                        newname = name[:-len(upDown)] if len(upDown) else name[:]
                        newname = f"{newname}_{decorrSystLabel[0]}{ix}{decorrSystLabel[1]}{iy}{upDown}"
                        logger.warning(f"Decorrelating syst {name} by {ax} bins: preparing new histogram {newname}")
                        hsystrootDecorr = copy.deepcopy(hnomiroot.Clone(newname))
                        ROOT.wrem.fillTH3fromTH3part(hsystrootDecorr, hsystroot,
                                                     xbinLow, ybinLow, 1,
                                                     xbinHigh, ybinHigh, hnomiroot.GetNbinsZ())
                        ret[newname] = narf.root_to_hist(hsystrootDecorr, axis_names = hnomi.axes.name)
        return ret

                
    def writeForProcess(self, h, proc, syst, check_systs=True):
        decorrelateByBin = {}
        hnom = None
        systInfo = None
        if syst != self.nominalName:
            systInfo = self.systematics[syst]
            procDict = self.datagroups.getDatagroups()
            hnom = procDict[proc].hists[self.nominalName]
            if systInfo["doActionBeforeMirror"] and systInfo["action"]:
                h = systInfo["action"](h, **systInfo["actionArgs"])
                self.outfile.cd() # needed to restore the current directory in case the action opens a new root file
            if systInfo["mirror"]:
                h = hh.extendHistByMirror(h, hnom,
                                          downAsUp=systInfo["mirrorDownVarEqualToUp"],
                                          downAsNomi=systInfo["mirrorDownVarEqualToNomi"])
            if systInfo["decorrByBin"]:
                decorrelateByBin = systInfo["decorrByBin"]
        logger.info(f"   {syst} for process {proc}")
        var_map = self.systHists(h, syst)
        if check_systs and syst != self.nominalName:
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
                self.writeHist(var, proc, name, setZeroStatUnc=setZeroStatUnc,
                               decorrByBin=decorrelateByBin, hnomi=hnom)

    def addPseudodata(self, processes, processesFromNomi=[]):
        datagroups = self.datagroups if not self.pseudodata_datagroups else self.pseudodata_datagroups
        processes = self.expandProcesses(processes)
        datagroups.loadHistsForDatagroups(
            baseName=self.nominalName, syst=self.pseudoData, label=self.pseudoData,
            procsToRead=processes,
            scaleToNewLumi=self.lumiScale, 
            fakerateIntegrationAxes=self.getFakerateIntegrationAxes())
        procDict = datagroups.getDatagroups()
        hists = [procDict[proc].hists[self.pseudoData] for proc in processes if proc not in processesFromNomi]
        # now add possible processes from nominal
        logger.warning(f"Making pseudodata summing these processes: {processes}")
        if len(processesFromNomi):
            logger.warning(f"These processes are taken from nominal datagroups: {processesFromNomi}")
            datagroupsFromNomi = self.datagroups
            datagroupsFromNomi.loadHistsForDatagroups(
                baseName=self.pseudoData, syst=self.nominalName, label=self.pseudoData,
                procsToRead=processesFromNomi,
                scaleToNewLumi=self.lumiScale,
                fakerateIntegrationAxes=self.getFakerateIntegrationAxes())
            procDictFromNomi = datagroupsFromNomi.getDatagroups()
            hists.extend([procDictFromNomi[proc].hists[self.pseudoData] for proc in processesFromNomi])
        # done, now sum all histograms
        hdata = hh.sumHists(hists)
        # Kind of hacky, but in case the alt hist has uncertainties
        for systAxName in ["systIdx", "tensor_axis_0", "vars"]:
            if systAxName in [ax.name for ax in hdata.axes]:
                hdata = hdata[{systAxName : self.pseudoDataIdx }] 

        self.writeHist(hdata, self.getDataName(), self.pseudoData+"_sum")

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

    def setOutput(self, outfolder, fitvars=[], doStatOnly=False, postfix=None):
        if self.datagroups.wmass:
            prefix = "WMass"
        elif self.datagroups.wlike:
            prefix = "ZMassWLike"
        else:
            prefix = "ZMassDilepton"
        if self.datagroups.lowPU:
            prefix += "_lowPU"
        tag = prefix+"_"+"_".join(fitvars)
        if doStatOnly:
            tag += "_statOnly"
        if self.datagroups.flavor:
            tag += f"_{self.datagroups.flavor}"
        if postfix is not None:
            tag += f"_{postfix}"
        self.outfolder = f"{outfolder}/{tag}/"
        if not os.path.isdir(self.outfolder):
            os.makedirs(self.outfolder)
        suffix = f"_{self.datagroups.flavor}" if self.datagroups.flavor else ""
        if self.xnorm:
            suffix += '_xnorm'

        self.cardName = (f"{self.outfolder}/{prefix}_{{chan}}{suffix}.txt")
        self.setOutfile(os.path.abspath(f"{self.outfolder}/{prefix}CombineInput{suffix}.root"))

    def writeOutput(self, args=None, forceNonzero=True, check_systs=True):
        self.datagroups.loadHistsForDatagroups(
            baseName=self.nominalName, syst=self.nominalName,
            procsToRead=self.datagroups.groups.keys(),
            label=self.nominalName, 
            scaleToNewLumi=self.lumiScale, 
            forceNonzero=forceNonzero,
            sumFakesPartial=not self.ABCD,
            fakerateIntegrationAxes=self.getFakerateIntegrationAxes())
        if self.ABCD and not self.xnorm:
            setSimultaneousABCD(self)
        
        self.writeForProcesses(self.nominalName, processes=self.datagroups.groups.keys(), label=self.nominalName, check_systs=check_systs)
        self.loadNominalCard()
        if self.pseudoData and not self.xnorm:
            self.addPseudodata([x for x in self.datagroups.groups.keys() if x != self.getDataName()],
                               [x for x in self.datagroups.groups.keys() if x != self.getDataName() and not self.pseudoDataProcsRegexp.match(x)])

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
                preOpMap=systMap["actionMap"], preOpArgs=systMap["actionArgs"],
                forceToNominal=forceToNominal,
                scaleToNewLumi=self.lumiScale,
                fakerateIntegrationAxes=self.getFakerateIntegrationAxes(),
            )
            self.writeForProcesses(syst, label="syst", processes=processes, check_systs=check_systs)

        output_tools.writeMetaInfoToRootFile(self.outfile, exclude_diff='notebooks', args=args)
        if self.skipHist:
            logger.info("Histograms will not be written because 'skipHist' flag is set to True")
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
            gen_axes = self.datagroups.gen_axes.copy()
        if additional_axes is not None:
            gen_axes += additional_axes
        # if only one or none gen axes, it is already included as main POI and no sumGroups are needed
        if len(gen_axes) <= 1:
            return
        # make a sum group for each gen axis
        axes_combinations = gen_axes
        # also include combinations of axes in case there are more than 2 axes
        for n in range(2, len(self.datagroups.gen_axes)):
            axes_combinations += [k for k in itertools.combinations(self.datagroups.gen_axes, n)]
        for axes in axes_combinations:
            logger.debug(f"Add sum group for {axes}")

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
                include = [(str(info["size"]) if x in info["processes"] else "-").ljust(self.procColumnsSpacing) for x in nondata_chan[chan]]
                self.cardContent[chan] += f'{name.ljust(self.spacing)} lnN{" "*(self.systTypeSpacing-2)} {"".join(include)}\n'
                if group and len(list(filter(groupFilter, [name]))):
                    self.addSystToGroup(group, chan, name)

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
                    for x in filter(lambda x: x != "", systInfo["outNames"])]
        # exit this function when a syst is applied to no process (can happen when some are excluded)
        if all(x not in procs for x in nondata):
            return 0
        
        #include = [(str(scale) if x in procs else "-").ljust(self.procColumnsSpacing) for x in nondata]
        include_chan = {}
        for chan in nondata_chan.keys():
            include_chan[chan] = [(str(scale) if x in procs else "-").ljust(self.procColumnsSpacing) for x in nondata_chan[chan]]

        splitGroupDict = systInfo["splitGroup"]
        shape = "shape" if not systInfo["noConstraint"] else "shapeNoConstraint"

        # Deduplicate while keeping order
        systNames = list(dict.fromkeys(names))

        # if decorrelating by bin, define new nuisances
        if systInfo["decorrByBin"]:
            systNamesTmp = self.makeDecorrelatedSystNuisances(systNames, systInfo["decorrByBin"])
            systNames = systNamesTmp[:]
                
        systnamesPruned = [s for s in systNames if not self.isExcludedNuisance(s)]
        systNames = systnamesPruned[:]
        for chan in self.channels:
            for systname in systNames:
                shape = "shape" if not systInfo["noConstraint"] else "shapeNoConstraint"
                # do not write systs which should only apply to other charge, to simplify card
                self.cardContent[chan] += f"{systname.ljust(self.spacing)} {shape.ljust(self.systTypeSpacing)} {''.join(include_chan[chan])}\n"
            # unlike for LnN systs, here it is simpler to act on the list of these systs to form groups, rather than doing it syst by syst 
            if group:
                systNamesForGroupPruned = systNames[:]
                systNamesForGroup = list(systNamesForGroupPruned if not groupFilter else filter(groupFilter, systNamesForGroupPruned))
                if len(systNamesForGroup):
                    for subgroup in splitGroupDict.keys():
                        matchre = re.compile(splitGroupDict[subgroup])
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
                "pseudodataHist" : f"{self.histName}_{self.getDataName()}_{self.pseudoData}_sum" if self.pseudoData else f"{self.histName}_{self.getDataName()}"
            }
            if not self.absolutePathShapeFileInCard:
                # use the relative path because absolute paths are slow in text2hdf5.py conversion
                args["inputfile"] = os.path.basename(args["inputfile"])

            self.cardContent[chan] = output_tools.readTemplate(self.nominalTemplate, args)
            self.cardGroups[chan] = ""
            
    def writeHistByCharge(self, h, name, decorrCharge=False):
        for charge in self.channels:
            q = self.chargeIdDict[charge]["val"]
            hout = narf.hist_to_root(self.getBoostHistByCharge(h, q))
            hout.SetName(name+f"_{charge}")
            hout.Write()
        
    def writeHistWithCharges(self, h, name):
        hout = narf.hist_to_root(h)
        hout.SetName(f"{name}_{self.channels[0]}")
        hout.Write()
    
    def writeHist(self, h, proc, syst, setZeroStatUnc=False, decorrByBin={}, hnomi=None):
        if self.skipHist:
            return
        if self.project:
            axes = self.project[:]
            if "charge" in h.axes.name and "charge" not in axes and not self.xnorm :
                axes.append("charge")
            if self.ABCD and not self.xnorm:
                if self.nameMT not in axes:
                    axes.append(self.nameMT)
                if common.passIsoName not in axes:
                    axes.append(common.passIsoName)
            # don't project h into itself when axes to project are all axes
            if any (ax not in h.axes.name for ax in axes):
                logger.error("Request to project some axes not present in the histogram")
                raise ValueError(f"Histogram has {h.axes.name} but requested axes for projection are {axes}")
            if len(axes) < len(h.axes.name):
                logger.debug(f"Projecting {h.axes.name} into {axes}")
                h = h.project(*axes)

        if self.unroll:
            logger.debug(f"Unrolling histogram")
            h = sel.unrolledHist(h, axes)

        if not self.nominalDim:
            self.nominalDim = h.ndim
            if self.nominalDim-self.writeByCharge > 3:
                raise ValueError("Cannot write hists with > 3 dimensions as combinetf does not accept THn")

        if h.ndim != self.nominalDim:
            raise ValueError(f"Histogram {proc}/{syst} does not have the correct dimensions. Found {h.ndim}, expected {self.nominalDim}")

        if setZeroStatUnc:
            h.variances(flow=True)[...] = 0.

        # make sub directories for each process or return existing sub directory
        directory = self.outfile.mkdir(proc, proc, True)
        directory.cd()

        name = self.variationName(proc, syst)

        hists = {name: h} # always keep original variation in output file for checks
        if decorrByBin:
            hists.update(self.makeDecorrelatedSystHistograms(h, hnomi, syst, decorrByBin))

        for hname, histo in hists.items():
            if self.writeByCharge:
                self.writeHistByCharge(histo, hname)
            else:
                self.writeHistWithCharges(histo, hname)
        self.outfile.cd()
