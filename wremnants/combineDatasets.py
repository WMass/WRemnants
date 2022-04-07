from collections import OrderedDict
from . import boostHistHelpers as hh
import narf
import logging
import hist
import numpy as np

def processDict():
    procs = OrderedDict(
            Wmunu_plus={
                "label" : r"W$\to\mu\nu$",
                "color" : "darkred",
            },
            Wmunu_minus = {
                "label" : "_nolegend_",
                "color" : "darkred",
            },
            Wtaunu_plus = {
                "label" : r"W$\to\tau\nu$",
                "color" : "darkblue",
            },
            Wtaunu_minus = {
                "label" : "_nolegend_",
                "color" : "darkblue"
            },
            Ztautau={
                "label" : r"Z$\to\tau\tau$",
                "color" : "orange",
            },
            Zmumu={
                "label" : r"Z$\to\mu\mu$",
                "color" : "lightblue",
            },
            Top={
                "label" : "Top",
                "color" : "green",
            },
            Diboson={
                "label" : "VV",
                "color" : "purple",
            },
            data_obs={
                "label" : "Data",
                "color" : "black",
            },
    )
    return procs

def processDictLowPileup(signalPtSplit=[0, 9, 20, 37, 60, 89, r"$\infty$"], removeUnsplit=False):
    redshades = ["#ffcccc", "#ff9999", "#ff6666", "#ff3333", "#ff0000", "#cc0000", "#990000", "#800000", "#660000", ]

    processes = OrderedDict(
        TTTo2L2Nu={
            "label" : r"t$\bar{t}$",
            "color" : "green",
        },
        TTToSemiLeptonic={
            "label" : "_no_label_",
            "color" : "green",
        },
        WZTo3LNu={
            "label" : "Diboson",
            "color" : "purple",
        },
        WWTo2L2Nu={
            "label" : "_no_label_",
            "color" : "pink",
        },
        DYmumu_MiNNLO = {
            "label" : r"Z$\to\mu\mu$",
            "color" : "lightblue",
        },
        WplusJetsToTauNu={
            "label" : r"W$^{\pm}\to\tau\nu$",
            "color" : "darkblue",
        },
        WminusJetsToTauNu={
            "label" : "_no_label_",
            "color" : "darkred",
        },
        WminusJetsToMuNu={
            "label" : "_no_label_",
            "color" : "darkred",
        },
        WplusJetsToMuNu={
            "label" : r"W$^{\pm}\to\mu\nu$",
            "color" : "darkred",
        },
        data_obs={
            "label" : "Data",
            "color" : "black",
        },
    )
    for i,ptlow in enumerate(signalPtSplit[:-1]):
        #for proc in ["WminusJetsToMuNu", "WplusJetsToMuNu", "WminusJetsToTauNu", "WplusJetsToTauNu"]:
        for proc in ["WminusJetsToMuNu", "WplusJetsToMuNu", ]:
            name = f"{proc}__PtBin{i}"
            processes[name] = dict(
                label=processes[proc]["label"] + r" (p$_{T}^{\mathrm{V}} \in$ "+f"[{ptlow}, {signalPtSplit[i+1]}])",
                color=redshades[i if i % 2 else len(redshades)-1-int(i/2)],
                name=processes[proc]["name"] if name in processes[proc] else name
            )
    if removeUnsplit:
        processes.pop("WminusJetsToMuNu")
        processes.pop("WplusJetsToMuNu")
        #processes.pop("WminusJetsToTauNu")
        #processes.pop("WplusJetsToTauNu")

    return processes

# TODO: Refactor these functions to not duplicate code
def fillLowPileupProcDictFromRoot(procDict, rtfile, syst, label="", obs="mt_reco_pf", 
        axis_names=["qTreco", "iso", "charge", "mt"], makeVariable={}, processes=[]):
    if label == "":
        label = syst
    for k,v in procDict.items():
        obsname = obs
        axes = axis_names[:]
        shouldRead = not processes or k in processes
        # Would be better if this wasn't so hacky
        name = k.split('__')[0] if "data_obs" not in k else "singlemuon"
        if syst != "nominal":
            name = f"{name}_{syst}_syst"
        if "WminusJetsToMuNu" in k or "WplusJetsToMuNu" in k:
            obsname = obs.replace("reco", "gen_reco")
            axes.insert(1, "qTgen")
        histname = f"{obsname}_{name}"
        rhist = rtfile.Get(histname) if shouldRead else None
        if not rhist:
            if shouldRead:
                logging.warning(f"No match for histogram '{histname}' found in file")
            continue
        if rhist.GetNdimensions() == len(axes)+1:
            axes.append("systIdx")
        bhist = narf.root_to_hist(rhist, axis_names=axes)
        axes = []
        if makeVariable:
            convertVariable = False
            for axis in bhist.axes:
                name = axis.name
                ax = axis
                if name in makeVariable:
                    ax = hist.axis.Variable(makeVariable[name], name=name)
                    convertVariable = True
                axes.append(ax)
            if convertVariable:
                hnew = hist.Hist(*axes, storage=hist.storage.Weight())
                hnew[...] = np.stack((bhist.values(), bhist.variances()), axis=-1)
                bhist = hnew
        # Horrible hack
        if "__PtBin" in k:
            b = int(k[-1])
            bhist = bhist[{"qTgen" : b}].copy()
        v.update({label : bhist})

def fillProcDictFromRoot(procDict, rtfile, syst, label="", processes=[],
        axis_names=["eta", "pt", "charge", "isoMt"]):
    if label == "":
        label = syst

    for k,v in procDict.items():
        shouldRead = not processes or k in processes
        # Combine wants this name
        proc = k if k != "data_obs" else "data"
        histname = f"{syst}__{proc}"
        rhist = rtfile.Get(histname) if shouldRead else None
        if not rhist:
            if shouldRead:
                logging.warning(f"No match for histogram '{syst}__{proc}' found in file")
            v.update({label : None})
            continue
        axes = axis_names[:]
        if rhist.GetNdimensions() == len(axes)+1:
            axes.append("systIdx")
        try:
            v.update({label : narf.root_to_hist(
                rhist, axis_names=axes
            )})
        except ValueError as e:
            logging.error(f"Failed to initiate hist {histname}")
            raise e

