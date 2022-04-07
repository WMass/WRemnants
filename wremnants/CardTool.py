from collections import OrderedDict
from . import boostHistHelpers as hh
from . import OutputTools
import narf
import logging
import ROOT
import uproot
import time
import numpy as np
import collections.abc
import os
import itertools

logging.basicConfig(level=logging.INFO)

def notImplemented(operation="Unknown"):
    raise NotImplementedError(f"Required operation '{operation}' is not implemented!")

class CardTool(object):
    def __init__(self, cardName="card.txt"):
        self.cardName = cardName
        self.systematics = {}
        self.lnNSystematics = {}
        self.procDict = OrderedDict()
        self.predictedProcs = []
        self.fakeEstimate = None
        self.addMirrorForSyst = {}
        self.channels = ["plus", "minus"]
        self.cardContent = {}
        self.cardGroups = ""
        self.nominalTemplate = ""
        self.spacing = 28
        self.fakeName = "Fake"
        self.dataName = "Data"
        self.nominalName = "nominal"
        self.datagroups = None
        self.unconstrainedProcesses = None
        self.buildHistNameFunc = None
        self.histName = "x"
        #self.loadArgs = {"operation" : "self.loadProcesses (reading hists from file)"}

    # Function call to load hists for processes (e.g., read from a ROOT file)
    # Extra args will be passed to each call
    def setLoadDatagroups(self, datagroups, extraArgs={}):
        self.datagroups = datagroups
        self.loadArgs = extraArgs
    
    def getFakeName(self):
        return self.fakeName

    # Needs to be increased from default for long proc names
    def setSpacing(self, spacing):
        self.spacing = spacing

    def setDataName(self, name):
        self.dataName = name

    def setDatagroups(self, datagroups):
        self.datagroups = datagroups 

    def setNominalTemplate(self, template):
        if not os.path.isfile(template):
            raise IOError(f"Template file {template} is not a valid file")
        self.nominalTemplate = template

    def predictedProcesses(self):
        if self.predictedProcs:
            return self.predictedProces
        return list(filter(lambda x: x != self.dataName, self.procDict.keys()))

    def setHistName(self, histName):
        self.histName = histName

    def isData(self, procName):
        return any([x.is_data for x in self.datagroups.groups[procName]["members"]])

    def isMC(self, procName):
        return not self.isData(procName)

    def addFakeEstimate(self, estimate):
        self.fakeEstimate = estimate

    def setProcesses(self, processes):
        self.procDict = OrderedDict([ (proc, {}) for proc in processes])

    def filteredProcesses(self, filterExpr):
        return list(filter(filterExpr, self.datagroups.processes()))

    def allMCProcesses(self):
        return self.filteredProcesses(lambda x: self.isMC(x))

    def mirrorNames(self, baseName, size, offset=0):
        names = [""]*offset + [f"{baseName.format(i=i%size)}{'Up' if i % 2 else 'Down'}" for i in range(size*2)]
        return names

    def addLnNSystematic(self, name, size, processes):
        self.lnNSystematics.update({name : {"size" : size, "processes" : processes}})

    def addSystematic(self, name, systAxes, outNames=None, skipEntries=None, labelsByAxis=None, 
                        baseName="", mirror=False, scale=1, processes=None, group=None, noConstraint=False,
                        action=None, actionArgs={}, systNameReplace=[], groupFilter=None):
        if not processes:
            processes = self.predictedProcesses()
        self.systematics.update({
            name : { "outNames" : [] if not outNames else outNames,
                "baseName" : baseName,
                "processes" : processes,
                "systAxes" : systAxes,
                "labelsByAxis" : systAxes if not labelsByAxis else labelsByAxis,
                "group" : group,
                "groupFilter" : groupFilter,
                "scale" : scale,
                "mirror" : mirror,
                "action" : action,
                "actionArgs" : actionArgs,
                "systNameReplace" : systNameReplace,
                "noConstraint" : noConstraint,
                "skipEntries" : [] if not skipEntries else skipEntries
            }
        })

    def setMirrorForSyst(self, syst, mirror=True):
        self.systematics[syst]["mirror"] = mirror

    def systLabelForAxis(self, axLabel, entry):
        if axLabel == "mirror" or axLabel == "downUpVar":
            return 'Up' if entry else 'Down'
        if "{i}" in axLabel:
            return axLabel.format(i=entry)
        return axLabel+str(entry)

    def excludeSystEntry(self, entry, skipEntries):
        for skipEntry in skipEntries:
            # Can use -1 to exclude all values of an axis
            if all(y == -1 or x == y for x,y in zip(entry, skipEntry)):
                return True
        return False

    def systHists(self, hvar, syst):
        if syst == self.nominalName:
            return ([self.nominalName], [hvar])

        systInfo = self.systematics[syst] 
        systAxes = systInfo["systAxes"]
        systAxesLabels = systAxes
        if "labelsByAxis" in systInfo:
            systAxesLabels = systInfo["labelsByAxis"]

        axNames = systAxes[:]
        axLabels = systAxesLabels[:]
        if hvar.axes[-1].name == "mirror":
            axNames.append("mirror")
            axLabels.append("mirror")

        if systInfo["action"]:
            hvar = systInfo["action"](hvar, **systInfo["actionArgs"])

        if not all([name in [ax.name for ax in hvar.axes] for name in axNames]):
            raise ValueError(f"Failed to find axis names '{str(systAxes)} in hist. " \
                f"Axes in hist are {str([ax.name for ax in hvar.axes])}")
        entries = list(itertools.product(*[range(hvar.axes[ax].size) for ax in axNames]))
        
        if len(systInfo["outNames"]) == 0:
            for entry in entries:
                if "skipEntries" in systInfo and self.excludeSystEntry(entry, systInfo["skipEntries"]):
                    systInfo["outNames"].append("")
                else:
                    name = systInfo["baseName"]
                    name += "".join([self.systLabelForAxis(al, entry[i]) for i,al in enumerate(axLabels)])
                    if "systNameReplace" in systInfo and systInfo["systNameReplace"]:
                        for rep in systInfo["systNameReplace"]:
                            name = name.replace(*rep)
                    # Obviously there is a nicer way to do this...
                    if "Up" in name:
                        name = name.replace("Up", "")+"Up"
                    elif "Down" in name:
                        name = name.replace("Down", "")+"Down"
                    systInfo["outNames"].append(name)

        variations = []
        for entry in entries:
            sel = {ax : binnum for ax,binnum in zip(axNames, entry)}
            variations.append(hvar[sel])
            #print(sel)
            #print("Sum is", hvar[sel].sum())
        return systInfo["outNames"], variations            

    def variationName(self, proc, name):
        proc = proc if proc != self.dataName else "data_obs"
        if name == self.nominalName:
            return f"{self.histName}_{proc}"
        else:
            return f"{self.histName}_{proc}_{name}"

    # This logic used to be more complicated, leaving the function here for now even
    # though it's trivial
    def addMirror(self, h, proc, syst):
        return syst != self.nominalName and self.systematics[syst]["mirror"]

    def writeForProcess(self, h, proc, syst):
        if self.addMirror(h, proc, syst):
            hnom = self.procDict[proc][self.nominalName]
            h = hh.extendHistByMirror(h, hnom)
        # Otherwise this is a processes not affected by the variation, don't write it out,
        # it's only needed for the fake subtraction
        logging.info(f"Writing systematic {syst} for process {proc}")
        var_names, variations = self.systHists(h, syst) 
        if len(var_names) != len(variations):
            logging.warning("The number of variations doesn't match the number of names for "
                f"process {proc}, syst {syst}. Found {len(var_names)} names and {len(variations)} variations.")
        for name, var in zip(var_names, variations):
            if name != "":
                self.writeHist(var, self.variationName(proc, name))

    def writeForProcesses(self, syst, processes, label):
        for process in processes:
            hvar = self.procDict[process][label]
            if not hvar:
                raise RuntimeError(f"Failed to load hist for process {process}, systematic {syst}")
            self.writeForProcess(hvar, process, syst)
        if syst != self.nominalName:
            self.fillCardWithSyst(syst)

    def setOutfile(self, outfile):
        if type(outfile) == str:
            self.outfile = ROOT.TFile(outfile, "recreate")
            #self.outfile = uproot.recreate(outfile)
        else:
            self.outfile = outfile

    def writeOutput(self):
        self.datagroups.loadHistsForDatagroups(
            baseName=self.histName, syst=self.nominalName, label=self.nominalName)
        self.procDict = self.datagroups.getDatagroups()
        self.writeForProcesses(self.nominalName, processes=self.procDict.keys(), label=self.nominalName)
        self.loadNominalCard()
        self.writeLnNSystematics()
        for syst in self.systematics.keys():
            processes=self.systematics[syst]["processes"]
            self.datagroups.loadHistsForDatagroups(self.histName, syst, label="syst",
                procsToRead=processes)
            self.writeForProcesses(syst, label="syst", processes=processes)
        self.writeCard()

    def writeCard(self):
        for chan in self.channels:
            with open(self.cardName.format(chan=chan), "w") as card:
                card.write(self.cardContent[chan])
                card.write("\n")
                card.write(self.cardGroups)

    def writeLnNSystematics(self):
        nondata = self.predictedProcesses()
        for name,info in self.lnNSystematics.items():
            include = [(str(info["size"]) if x in info["processes"] else "-").ljust(self.spacing) for x in nondata]
            for chan in self.channels:
                self.cardContent[chan] += f'{name.ljust(self.spacing)}lnN{" "*(self.spacing-3)}{"".join(include)}\n'

    def fillCardWithSyst(self, syst):
        systInfo = self.systematics[syst]
        scale = systInfo["scale"]
        procs = systInfo["processes"]
        nondata = self.predictedProcesses()
        names = [x[:-2] if "Up" in x[-2:] else (x[:-4] if "Down" in x[-4:] else x) 
                    for x in filter(lambda x: x != "", systInfo["outNames"])]
        include = [(str(scale) if x in procs else "-").ljust(self.spacing) for x in nondata]

        # Deduplicate while keeping order
        systNames = list(dict.fromkeys(names))
        for systname in systNames:
            shape = "shape" if not systInfo["noConstraint"] else "shapeNoConstraint"
            for chan in self.channels:
                self.cardContent[chan] += f"{systname.ljust(self.spacing)}{shape.ljust(self.spacing)}{''.join(include)}\n"

        group = systInfo["group"]
        if group:
            # TODO: Make more general
            label = "group" if not systInfo["noConstraint"] else "noiGroup"
            filt = systInfo["groupFilter"]
            members = " ".join(systNames if not filt else filter(filt, systNames))
            group_expr = f"{group} {label} ="
            if group_expr in self.cardGroups:
                idx = self.cardGroups.index(group_expr)+len(group_expr)
                self.cardGroups = self.cardGroups[:idx] + " " + members + " " + self.cardGroups[idx:]
            else:
                self.cardGroups += f"\n{group_expr} {members}"

    def setUnconstrainedProcs(self, procs):
        self.unconstrainedProcesses = procs

    def processLabels(self):
        nondata = np.array(self.predictedProcesses())
        labels = np.arange(len(nondata))+1
        issig = np.isin(nondata, self.unconstrainedProcesses)
        labels[issig] = -np.arange(np.count_nonzero(issig))-1
        return labels

    def loadNominalCard(self):
        procs = self.predictedProcesses()
        nprocs = len(procs)
        for chan in self.channels:
            args = {
                "channel" :  chan,
                "channelPerProc" : chan.ljust(self.spacing)*nprocs,
                "processes" : "".join([x.ljust(self.spacing) for x in procs]),
                "labels" : "".join([str(x).ljust(self.spacing) for x in self.processLabels()]),
                # Could write out the proper normalizations pretty easily
                "rates" : "-1".ljust(self.spacing)*nprocs,
                #"inputfile" : self.outfile.file_path,
                "inputfile" : self.outfile.GetName(),
            }
            self.cardContent[chan] = OutputTools.readTemplate(self.nominalTemplate, args)
        
    def writeHistByCharge(self, h, name, charges=["plus", "minus"]):
        for i,charge in enumerate(self.channels):
            hout = narf.hist_to_root(h[{"charge" : i}])
            hout.SetName(name+f"_{charge}")
            hout.Write()

    def writeHistWithCharges(self, h, name):
        print("As boost", h.sum())
        hout = narf.hist_to_root(h)
        #self.outfile[name] = h
        hout.SetName(name)
        hout.Write()
    
    def writeHist(self, h, name):
        if self.channels[0] != "all":
            self.writeHistByCharge(h, name)
        else:
            self.writeHistWithCharges(h, name)
