from wremnants import boostHistHelpers as hh
from wremnants import histselections as sel
from wremnants.datasets import datasetsLowPU
from wremnants.datasets.datagroups import datagroups
import logging
import lz4.frame
import pickle
import narf
import ROOT
import hist

class datagroupsLowPU(datagroups):
    def __init__(self, infile, combine=False):
        self.datasets = {x.name : x for x in datasetsLowPU.getDatasets()}
        super().__init__(infile, combine)
        self.lumi = 0.199 # TODO More digits
        self.groups = dict(
            Top=dict(
                members = [self.datasets[x] for x in ["TTTo2L2Nu", "TTToSemiLeptonic"]],
                label = "Top",
                color="green",
                signalOp = sel.signalHistLowPileupW,
            ),
            Fake=dict(
                members = list(self.datasets.values()),
                #scale = lambda x: 0.13*(1. if x.is_data else -1),
                scale = lambda x: 2.3*(1. if x.is_data else -1),
                label = "Nonprompt",
                color = "grey",
                signalOp = sel.fakeHistIsoRegion,
            ),
            Diboson=dict(
                members = [self.datasets[x] for x in ["WZTo3LNu", "WWTo2L2Nu"]],
                label="Diboson",
                color="pink",
                signalOp = sel.signalHistLowPileupW,
            ),
            Zmumu=dict(
                members=[self.datasets["DYmumu_MiNNLO"]],
                label=r"Z$\to\mu\mu$",
                color="lightblue",
                signalOp = sel.signalHistLowPileupW,
            ),
            Wmunu=dict(
                members=[self.datasets[x] for x in ["WplusJetsToMuNu", "WminusJetsToMuNu"]],
                label=r"W$^{\pm}\to\mu\nu$",
                color="darkred",
                signalOp = sel.signalHistLowPileupW,
            ),
            Wtaunu=dict(
                members=[self.datasets[x] for x in ["WminusJetsToTauNu", "WplusJetsToTauNu"]],
                label=r"W$^{\pm}\to\tau\nu$",
                color="darkblue",
                signalOp = sel.signalHistLowPileupW,
            ),
            Data=dict(
                members=[self.datasets["singlemuon"]],
                label="Data",
                color="black",
                signalOp = sel.signalHistLowPileupW,
            ),
        )

    def processScaleFactor(self, proc):
        if proc.is_data:
            return 1
        return self.lumi

    def histName(self, baseName, proc, syst):
        if proc in ["WplusJetsToMuNu", "WminusJetsToMuNu"] and "gen" not in baseName:
            baseName = baseName.replace("reco", "gen_reco")
        base = f"{baseName}_{proc}"
        return base if syst == "nominal" else f"{base}_{syst}_syst"

    def readHist(self, baseName, proc, syst, scaleOp=None, forceNonzero=True):
        axisNames = None
        readname = self.histName(baseName, proc.name, syst)
        if "mt_reco_pf" in readname:
            axisNames = ["qTreco", "iso", "charge", "mt"]
        elif "mt_gen_reco_pf" in readname:
            axisNames = ["qTreco", "qTgen", "iso", "charge", "mt"]
        if syst != "nominal":
            axisNames.append("systAx")

        rthist = self.rtfile.Get(readname)
        if not rthist:
            raise ValueError(f"Histogram {readname} not found for process {proc.name}")
        h = narf.root_to_hist(rthist, axis_names=axisNames)
        # TODO: this is a hack. For this to be smoothly treated, qTgen should be the first axis
        if "qTgen" in axisNames:
            s = hist.tag.Slicer()
            h = h[{"qTgen" : s[::hist.sum]}]

        if forceNonzero:
            h = hh.clipNegativeVals(h)
        scale = self.processScaleFactor(proc)
        if scaleOp:
            scale = scale*scaleOp(proc)
        return h*scale

