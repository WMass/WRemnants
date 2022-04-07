from wremnants import boostHistHelpers as hh
from wremnants import histselections as sel
from wremnants.datasets import datasets2016
import logging
import lz4.frame
import pickle
import narf
#import uproot
import ROOT

class datagroups(object):
    def __init__(self, infile, combine=False):
        self.combine = combine
        if ".root" not in infile[-5:]:
            with lz4.frame.open(infile) as f:
                self.results = pickle.load(f)
            self.rtfile = None
        else:
            self.rtfile = ROOT.TFile.Open(infile)
            self.results = None

        if self.datasets:
            self.data = [x for x in self.datasets.values() if x.is_data]

        self.groups = {}
        self.lumi = 1 if not self.results else sum([self.results[x.name]["lumi"] for x in self.data if x.name in self.results])
        self.nominalName = "nominal"

    def processScaleFactor(self, proc):
        if proc.is_data:
            return 1
        return self.lumi*1000*proc.xsec/self.results[proc.name]["weight_sum"]

    def setHists(self, baseName, syst, procsToRead=None, label=None, nominalIfMissing=True, 
            selectSignal=True, forceNonzero=True):
        if not label:
            label = syst if syst else baseName
        if not procsToRead:
            procsToRead = self.groups.keys()

        for procName in procsToRead:
            group = self.groups[procName]
            group[label] = None

            for member in group["members"]:
                scale = group["scale"] if "scale" in group else None
                try:
                    h = self.readHist(baseName, member, syst, scaleOp=scale, forceNonzero=forceNonzero)
                except ValueError as e:
                    if nominalIfMissing:
                        h = self.readHist(baseName, member, self.nominalName, scaleOp=scale, forceNonzero=forceNonzero)
                    else:
                        logging.warning(str(e))
                        continue
                group[label] = h if not group[label] else hh.addHists(h, group[label])
            if selectSignal and group[label] and "signalOp" in group and group["signalOp"]:
                group[label] = group["signalOp"](group[label])

    #TODO: Better organize to avoid duplicated code
    def setHistsCombine(self, baseName, syst, channel, procsToRead=None, label=None):
        #TODO Set axis names properly
        if baseName == "x":
            axisNames=["eta", "pt"]

        if not label:
            label = syst
        if not procsToRead:
            procsToRead = self.groups.keys()

        for procName in procsToRead:
            group = self.groups[procName]
            group[label] = None
            name = self.histNameCombine(procName, baseName, syst, channel)
            rthist = self.rtfile.Get(name)
            if not rthist:
                raise RuntimeError(f"Failed to load hist {name} from file")
            group[label] = narf.root_to_hist(rthist, axis_names=axisNames)

    def histNameCombine(self, procName, baseName, syst, channel):
        name = f"{baseName}_{procName}"
        if syst != "nominal":
            name += "_"+syst
        if channel:
            name += "_"+channel
        return name

    def loadHistsForDatagroups(self, baseName, syst, procsToRead=None, channel="", label="", nominalIfMissing=True,
            selectSignal=True, forceNonzero=True):
        if self.rtfile and self.combine:
            self.setHistsCombine(baseName, syst, channel, procsToRead, label)
        else:
            self.setHists(baseName, syst, procsToRead, label, nominalIfMissing, selectSignal, forceNonzero)

    def getDatagroups(self):
        return self.groups

    def resultsDict(self):
        return self.results

    def processes(self):
        return self.groups.keys()

    def addUncorrectedProc(self, refname, name="uncorr", label="Uncorrected", color="red", exclude=["Data"]):
        self.loadHistsForDatagroups(refname, syst=name)
        self.groups[name] = dict(
            label=label,
            color=color,
            members=[],
        )
        self.groups[name][refname] = sum([self.groups[x][name] for x in self.groups.keys() if x not in exclude+[name]])

class datagroups2016(datagroups):
    def __init__(self, infile, combine=False, wlike=False):
        self.datasets = {x.name : x for x in datasets2016.getDatasets()}
        super().__init__(infile, combine)
        self.groups =  {
            "Data" : dict(
                members = [self.datasets["dataPostVFP"]],
                color = "black",
                label = "Data",
                signalOp = sel.signalHistWmass if not wlike else None,
            ),
            "Zmumu" : dict(
                members = [self.datasets["ZmumuPostVFP"]],
                label = r"Z$\to\mu\mu$",
                color = "lightblue",
                signalOp = sel.signalHistWmass if not wlike else None,
            ),   
            "Ztautau" : dict(
                members = [self.datasets["ZtautauPostVFP"]],
                label = r"Z$\to\tau\tau$",
                color = "darkblue",
                signalOp = sel.signalHistWmass if not wlike else None,
            ), 
        }
        if not wlike:
            self.groups.update({
                "Fake" : dict(
                    members = list(self.datasets.values()),
                    scale = lambda x: 1. if x.is_data else -1,
                    label = "Nonprompt",
                    color = "grey",
                    signalOp = sel.fakeHistABCD,
                ),
                "Wtau" : dict(
                    members = [self.datasets["WminustaunuPostVFP"], self.datasets["WplustaunuPostVFP"]],
                    label = r"W$^{\pm}\to\tau\nu$",
                    color = "orange",
                    signalOp = sel.signalHistWmass,
                ),
                "Wmunu" : dict(
                    members = [self.datasets["WminusmunuPostVFP"], self.datasets["WplusmunuPostVFP"]],
                    label = r"W$^{\pm}\to\mu\nu$",
                    color = "darkred",
                    signalOp = sel.signalHistWmass,
                ),
                "Top" : dict(
                    members = [self.datasets["TTSemileptonicPostVFP"], self.datasets["TTLeptonicPostVFP"]],
                    label = "Top",
                    color = "green",
                    signalOp = sel.signalHistWmass,
                ), 
                "Diboson" : dict(
                    members = [self.datasets["WWPostVFP"]],
                    label = "Diboson",
                    color = "pink",
                    signalOp = sel.signalHistWmass,
                ), 
            })
        else:
            self.groups["Other"] = dict(
                members = [x for x in self.datasets.values() if not x.is_data and x.name not in ["ZmumuPostVFP", "ZtautauPostVFP"]],
                label = "Other",
                color = "grey",
            )

    def histName(self, baseName, procName, syst):
        # This is kind of hacky to deal with the different naming from combine
        if baseName != "x" and (syst == "" or syst == self.nominalName):
            return baseName
        if (baseName == "" or baseName == "x") and syst:
            return syst
        return "_".join([baseName,syst])
    
    def readHist(self, baseName, proc, syst, scaleOp=None, forceNonzero=True):
        output = self.results[proc.name]["output"]
        histname = self.histName(baseName, proc.name, syst)
        if histname not in output:
            raise ValueError(f"Histogram {histname} not found for process {proc.name}")
        h = output[histname]
        if forceNonzero:
            h = hh.clipNegativeVals(h)
        scale = self.processScaleFactor(proc)
        if scaleOp:
            scale = scale*scaleOp(proc)
        return h*scale

