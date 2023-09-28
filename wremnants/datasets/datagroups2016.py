from utilities import boostHistHelpers as hh, logging
from wremnants import histselections as sel

logger = logging.child_logger(__name__)
    
def make_datagroups_2016(dg, combine=False, pseudodata_pdfset = None, applySelection=True, excludeGroups=None, filterGroups=None):
    # reset datagroups
    dg.groups = {}

    if dg.wmass and applySelection:
        sigOp = sel.signalHistWmass
        fakeOp = sel.fakeHistABCD
    else:
        sigOp = None
        fakeOp = None

    dg.addGroup("Data",
        members = dg.getSafeListFromDataset(["dataPostVFP"]),
        color = "black",
        label = "Data",
        selectOp = sigOp,
    )
    dg.addGroup("Zmumu",
        members = list(filter(lambda y: y.group == "Zmumu", dg.datasets.values())),
        #label = r"Z$\to\mu\mu$ (N$^{3}LL+NNLO)$",
        label = r"Z$\to\mu\mu$",
        color = "lightblue",
        selectOp = sigOp,
    ) 
    dg.addGroup("Ztautau",
        members = list(filter(lambda y: y.group == "Ztautau", dg.datasets.values())),
        label = r"Z$\to\tau\tau$",
        color = "darkblue",
        selectOp = sigOp,
    )

    if pseudodata_pdfset and dg.combine:
        dg.addGroup(f"pdf{pseudodata_pdfset.upper()}_sum",
            label = f"pdf{pseudodata_pdfset.upper()}",
            color = "dimgray"
        )
    if dg.wmass:
        dg.addGroup("Wmunu",
            members = list(filter(lambda y: y.group == "Wmunu", dg.datasets.values())),
            label = r"W$^{\pm}\to\mu\nu$",
            color = "darkred",
            selectOp = sigOp,
        )
        dg.addGroup("Wtaunu",
            members = list(filter(lambda y: y.group == "Wtaunu", dg.datasets.values())),
            label = r"W$^{\pm}\to\tau\nu$",
            color = "orange",
            selectOp = sigOp,
        )
        dg.addGroup("DYlowMass",
            members = list(filter(lambda y: y.group == "DYlowMass", dg.datasets.values())),
            label = r"Z$\to\mu\mu$, $10<m<50$ GeV",
            color = "deepskyblue",
            selectOp = sigOp,
        )
        dg.addGroup("Top",
            members = list(filter(lambda y: y.group == "Top", dg.datasets.values())),
            label = "Top",
            color = "green",
            selectOp = sigOp,
        )
        dg.addGroup("Diboson",
            members = list(filter(lambda y: y.group == "Diboson", dg.datasets.values())),
            label = "Diboson",
            color = "pink",
            selectOp = sigOp,
        )
        dg.addGroup("QCD",
            members = list(filter(lambda y: y.group == "QCD", dg.datasets.values())),
            label = "QCD MC",
            color = "grey",
            selectOp = sigOp,
        )   
    else:
        dg.addGroup("Other",
            members = [x for x in dg.datasets.values() if not x.is_data and x.group not in ["Zmumu", "Ztautau"] and x.group != "QCD"],
            label = "Other",
            color = "grey",
        )

    dg.filterGroups(filterGroups)
    dg.excludeGroups(excludeGroups)

    if dg.wmass:
        # add all processes to the fake contributions after filtered and excluded groups
        dg.addGroup("Fake",
            members = [member for sublist in [v.members for k, v in dg.groups.items() if k != "QCD"] for member in sublist],
            scale = lambda x: 1. if x.is_data else -1,
            label = "Nonprompt",
            color = "grey",
            selectOp = fakeOp,
        )
        dg.filterGroups(filterGroups)
        dg.excludeGroups(excludeGroups)

    return dg
