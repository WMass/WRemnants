from utilities import logging,common, boostHistHelpers as hh
from utilities.io_tools import input_tools
from wremnants import syst_tools,theory_tools
import narf.ioutils
import numpy as np
import re
import hist
import h5py

logger = logging.child_logger(__name__)

class TheoryAgnosticHelper(object):
    def __init__(self, card_tool, externalArgs=None):
        self.card_tool = card_tool        
        toCheck = ['signal_samples', 'signal_samples_noOutAcc']
        for group in toCheck:
            if group not in self.card_tool.procGroups:
                raise ValueError(f"Must define '{group}' procGroup in CardTool for theory agnostic fit")
        self.args = externalArgs
        self.label = "Label"
        self.passSystToFakes = True
        self.separateOutOfAccSignal = False
        self.poi_axes = []
        
    def configure_polVar(self,
                         label,
                         passSystToFakes,
                         hasSeparateOutOfAcceptanceSignal):
        self.label = label
        self.passSystToFakes = passSystToFakes
        self.separateOutOfAccSignal = hasSeparateOutOfAcceptanceSignal

    def configure_normVar(self,
                          label,
                          passSystToFakes,
                          poi_axes):
        self.label = label
        self.passSystToFakes = passSystToFakes
        self.poi_axes = poi_axes
        
    def add_theoryAgnostic_polVar_uncertainty(self):
        coeffs = ["UL"] + [f"A{i}" for i in range(5)]
        groupName = f"polVar{self.label}"
        for genVcharge in ["minus", "plus"]:
            for coeffKey in coeffs:
                self.card_tool.addSystematic(f"theoryAgnosticWithPol_{coeffKey}_{genVcharge}",
                                       group=groupName,
                                       mirror=False,
                                       symmetrize="average" if self.args.symmetrizePolVar else None,
                                       passToFakes=False if self.args.noPolVarOnFake else self.passSystToFakes,
                                       processes=["signal_samples_noOutAcc" if self.separateOutOfAccSignal else "signal_samples"],
                                       baseName=f"{groupName}_{coeffKey}_{genVcharge}_",
                                       noConstraint=True,
                                       systAxes=["nPolVarSyst", "downUpVar"], 
                                       labelsByAxis=["v", "downUpVar"],
                                       #splitGroup={f"{groupName}_{coeffKey}" : f"{groupName}_{coeffKey}"}
                                       )

    def add_muRmuF_polVar_uncertainty(self, isTheoryAgnostic):
        coeffs = [f"A{i}" for i in range(8)]
        signalGroupName = "muRmuFPolVarW" if self.label == "W" else "muRmuFPolVarZ"
        nonSignalGroupName = "muRmuFPolVarZ" if self.label == "W" else "muRmuFPolVarW"
        signalProcesses = "signal_samples_inctau_OOA" if isTheoryAgnostic else "signal_samples_inctau"
        nonsignalProcesses = "nonsignal_samples_inctau_OOA" if isTheoryAgnostic else "nonsignal_samples_inctau"
        ProcessNames = [signalGroupName,nonSignalGroupName]
        Processes = [signalProcesses,nonsignalProcesses]
        if self.args.muRmuFFullyCorr:
            signal_expanded_samples = self.card_tool.getProcNames([signalProcesses])
            nonsignal_expanded_samples = self.card_tool.getProcNames([nonsignalProcesses])
            func = syst_tools.muRmuFFullyCorrelatedVariations
            signal_preop_map = {proc : func for proc in signal_expanded_samples}
            nonsignal_preop_map = {proc : func for proc in nonsignal_expanded_samples}
            preop_args = {}
            PreOpMaps = [signal_preop_map,nonsignal_preop_map]
        for Index in range(0,len(ProcessNames)):
            for coeffKey in coeffs:
                self.card_tool.addSystematic(f"muRmuFPolVar_{coeffKey}",
                                       group=ProcessNames[Index],
                                       mirror=False,
                                       passToFakes=self.passSystToFakes,
                                       processes=[Processes[Index]],
                                       baseName=f"{ProcessNames[Index]}_{coeffKey}_",
                                       noConstraint=False,
                                       systAxes=["nPolVarSyst", "downUpVar"], 
                                       labelsByAxis=["v", "downUpVar"],
                                       rename=f"{ProcessNames[Index]}_{coeffKey}",
                                       #splitGroup={f"{groupName}_{coeffKey}" : f"{groupName}_{coeffKey}"}
                                       )
                if self.args.muRmuFFullyCorr:
                    self.card_tool.addSystematic(f"muRmuFPolVar_{coeffKey}",
                                           preOpMap=PreOpMaps[Index],
                                           preOpArgs=preop_args,
                                           group=ProcessNames[Index],
                                           mirror=False,
                                           passToFakes=self.passSystToFakes,
                                           processes=[Processes[Index]],
                                           baseName=f"{ProcessNames[Index]}_{coeffKey}_FullyCorr_",
                                           noConstraint=False,
                                           systAxes=["nPolVarSyst", "downUpVar"], 
                                           labelsByAxis=["v", "downUpVar"],
                                           rename=f"{ProcessNames[Index]}_{coeffKey}_FullyCorr",
                                           #splitGroup={f"{groupName}_{coeffKey}" : f"{groupName}_{coeffKey}"}
                                           )
        

    def add_theoryAgnostic_normVar_uncertainty(self, flow=True):

        common_noi_args = dict(
            group = f"normXsec{self.label}",
            passToFakes = self.passSystToFakes
        )
        
        # open file with theory bands
        with h5py.File(f"{common.data_dir}/angularCoefficients/theoryband_variations.hdf5", "r") as ff:
            scale_hists = narf.ioutils.pickle_load_h5py(ff["theorybands"])
            for tt in scale_hists:
                scale_hists[tt].values()[...,1] = scale_hists[tt].values()[...,1]-np.ones_like(scale_hists[tt].values()[...,1])
                scale_hists[tt].values()[...,0] = np.where(scale_hists[tt].values()[...,0]>0,scale_hists[tt].values()[...,0]*(-1.0),-np.ones_like(scale_hists[tt].values()[...,0])+scale_hists[tt].values()[...,0])

        # First do in acceptance bins, then OOA later (for OOA we need to group bins into macro regions)
        nuisanceBaseName = f"norm{self.label}"
        for sign in ["plus", "minus"]:
            self.card_tool.addSystematic("yieldsTheoryAgnostic",
                                rename=f"{nuisanceBaseName}{sign}",
                                **common_noi_args,
                                mirror=False,
                                symmetrize = "quadratic",
                                systAxes=self.poi_axes+["downUpVar"],
                                processes=["signal_samples"],
                                baseName=f"{nuisanceBaseName}{sign}_",
                                noConstraint=True if self.args.priorNormXsec < 0 else False,
                                scale=1,
                                formatWithValue=[None,None,"low",None],
                                #customizeNuisanceAttributes={".*AngCoeff4" : {"scale" : 1, "shapeType": "shapeNoConstraint"}},
                                labelsByAxis=["PtV", "YVBin", "Helicity","downUpVar"],
                                systAxesFlow=[], # only bins in acceptance in this call
                                skipEntries=[{"helicitySig" : [6,7,8]}], # removing last three indices out of 9 (0,1,...,7,8) corresponding to A5,6,7
                                preOpMap={
                                    m.name: (
                                        lambda h, scale_hist=scale_hists[m.name]: hh.addHists(h[{ax: hist.tag.Slicer()[::hist.sum] for ax in self.poi_axes}], hh.multiplyHists(hh.addGenericAxis(h,common.down_up_axis, flow=flow), hh.rescaleBandVariation(scale_hist,self.args.theoryAgnosticBandSize),flow=flow))) if sign in m.name else (lambda h: h[{ax: hist.tag.Slicer()[::hist.sum] for ax in self.poi_axes}]) for g in self.card_tool.procGroups["signal_samples"] for m in self.card_tool.datagroups.groups[g].members},
                                )
            # now OOA
            nuisanceBaseNameOOA = f"{nuisanceBaseName}OOA_"
            # TODO: implement a loop to generalize it
            #
            # ptV OOA, yV in acceptance, integrate helicities 
            self.card_tool.addSystematic("yieldsTheoryAgnostic",
                                rename=f"yieldsTheoryAgnostic_OOA_ptV_{self.label}{sign}",
                                **common_noi_args,
                                mirror=True,
                                scale=0.5,
                                processes=["signal_samples"],
                                baseName=f"{nuisanceBaseNameOOA}{sign}_",
                                noConstraint=True if self.args.priorNormXsec < 0 else False,
                                systAxes=["ptVgenSig","helicitySig"],
                                formatWithValue=[None,"low"],
                                labelsByAxis=["PtVBin","Helicity"],
                                systAxesFlow=["ptVgenSig"], # this can activate nuisances on overflow bins, mainly just ptV and yV since the helicity axis has no overflow bins
                                skipEntries=[{"helicitySig" : [6,7,8]}], # removing last three indices out of 9 (0,1,...,7,8) corresponding to A5,6,7
                                preOpMap={
                                    m.name: (lambda h: hh.addHists(h[{ax: hist.tag.Slicer()[::hist.sum] for ax in self.poi_axes}],
                                                                    h[{"ptVgenSig": hist.tag.Slicer()[hist.overflow:],
                                                                        "absYVgenSig": hist.tag.Slicer()[0:h.axes["absYVgenSig"].size:hist.sum]}],
                                                                    scale2=self.args.scaleNormXsecHistYields)
                                                ) if sign in m.name else (lambda h: h[{ax: hist.tag.Slicer()[::hist.sum] for ax in self.poi_axes}]) for g in self.card_tool.procGroups["signal_samples"] for m in self.card_tool.datagroups.groups[g].members
                                },
                                )
            # ptV in acceptance, yV OOA, integrate helicities
            self.card_tool.addSystematic("yieldsTheoryAgnostic",
                                rename=f"yieldsTheoryAgnostic_OOA_yV_{self.label}{sign}",
                                **common_noi_args,
                                mirror=True,
                                scale=0.5,
                                processes=["signal_samples"],
                                baseName=f"{nuisanceBaseNameOOA}{sign}_",
                                noConstraint=True if self.args.priorNormXsec < 0 else False,
                                systAxes=["absYVgenSig","helicitySig"],
                                formatWithValue=[None,"low"],
                                labelsByAxis=["YVBin","Helicity"],
                                systAxesFlow=["absYVgenSig"], # this can activate nuisances on overflow bins, mainly just ptV and yV since the helicity axis has no overflow bins
                                skipEntries=[{"helicitySig" : [6,7,8]}], # removing last three indices out of 9 (0,1,...,7,8) corresponding to A5,6,7
                                preOpMap={
                                    m.name: (lambda h: hh.addHists(h[{ax: hist.tag.Slicer()[::hist.sum] for ax in self.poi_axes}],
                                                                    h[{"ptVgenSig": hist.tag.Slicer()[0:h.axes["ptVgenSig"].size:hist.sum],
                                                                        "absYVgenSig": hist.tag.Slicer()[hist.overflow:]}],
                                                                    scale2=self.args.scaleNormXsecHistYields)
                                                ) if sign in m.name else (lambda h: h[{ax: hist.tag.Slicer()[::hist.sum] for ax in self.poi_axes}]) for g in self.card_tool.procGroups["signal_samples"] for m in self.card_tool.datagroups.groups[g].members
                                },
                                )
            # ptV OOA and yV OOA, integrate helicities
            self.card_tool.addSystematic("yieldsTheoryAgnostic",
                                rename=f"yieldsTheoryAgnostic_OOA_ptVyV_{self.label}{sign}",
                                **common_noi_args,
                                mirror=True,
                                scale=0.5,
                                processes=["signal_samples"],
                                baseName=f"{nuisanceBaseNameOOA}{sign}_",
                                noConstraint=True if self.args.priorNormXsec < 0 else False,
                                systAxes=["ptVgenSig", "absYVgenSig","helicitySig"],
                                formatWithValue=[None,None,"low"],
                                labelsByAxis=["PtVBin", "YVBin", "Helicity"],
                                systAxesFlow=["ptVgenSig", "absYVgenSig"], # this can activate nuisances on overflow bins, mainly just ptV and yV since the helicity axis has no overflow bins
                                skipEntries=[{"helicitySig" : [6,7,8]}], # removing last three indices out of 9 (0,1,...,7,8) corresponding to A5,6,7
                                preOpMap={
                                    m.name: (lambda h: hh.addHists(h[{ax: hist.tag.Slicer()[::hist.sum] for ax in self.poi_axes}],
                                                                    h[{"ptVgenSig": hist.tag.Slicer()[hist.overflow:],
                                                                        "absYVgenSig": hist.tag.Slicer()[hist.overflow:]}],
                                                                    scale2=self.args.scaleNormXsecHistYields)
                                                ) if sign in m.name else (lambda h: h[{ax: hist.tag.Slicer()[::hist.sum] for ax in self.poi_axes}]) for g in self.card_tool.procGroups["signal_samples"] for m in self.card_tool.datagroups.groups[g].members
                                },
                                )

    def add_theoryAgnostic_uncertainty(self):
        isTheoryAgnostic = self.args.analysisMode in ["theoryAgnosticNormVar", "theoryAgnosticPolVar"]
        if isTheoryAgnostic:
            if self.args.analysisMode == "theoryAgnosticPolVar":
                self.add_theoryAgnostic_polVar_uncertainty()
            else:
                self.add_theoryAgnostic_normVar_uncertainty()
        if self.args.muRmuFPolVar == True:
            self.add_muRmuF_polVar_uncertainty(isTheoryAgnostic)
