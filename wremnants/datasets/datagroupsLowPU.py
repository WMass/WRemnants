from utilities import boostHistHelpers as hh, logging
from wremnants import histselections as sel
import hist

logger = logging.child_logger(__name__)

def make_datagroups_lowPU(dg, combine=False, excludeGroups=None, filterGroups=None, applySelection=True):
    # reset datagroups
    dg.groups = {}

    if dg.wmass and applySelection:
        sigOp = sel.signalHistWmass
        fakeOp = sel.fakeHistABCD
    else:
        sigOp = None
        fakeOp = None

    # data
    if dg.flavor == "mu" or dg.flavor == "mumu":  
        dg.addGroup("Data",
            members = list(filter(lambda y: y.name == "singlemuon", dg.datasets.values())),
            label="Data",
            color = "black",
            selectOp = sigOp,
        )
        dg.addGroup("Zmumu",
            members = list(filter(lambda y: y.group == "Zmumu", dg.datasets.values())),
            label = r"Z$\to\mu\mu$",
            color = "lightblue",
            selectOp = sigOp,
        ) 
        if dg.wmass:
            dg.addGroup("Wmunu",
                members = list(filter(lambda y: y.group == "Wmunu", dg.datasets.values())),
                label = r"W$^{\pm}\to\mu\nu$",
                color = "darkred",
                selectOp = sigOp,
            )

    if dg.flavor == "e" or dg.flavor == "ee":  
        dg.addGroup("Data",
            members = list(filter(lambda y: y.name == "singleelectron", dg.datasets.values())),
            label="Data",
            color = "black",
            selectOp = sigOp,
        )
        dg.addGroup("Zee",
            members = list(filter(lambda y: y.group == "Zee", dg.datasets.values())),
            label = r"Z$\to ee$",
            color = "lightblue",
            selectOp = sigOp,
        ) 
        if dg.wmass:
            dg.addGroup("Wenu",
                members = list(filter(lambda y: y.group == "Wenu", dg.datasets.values())),
                label = r"W$^{\pm}\to e\nu$",
                color = "darkred",
                selectOp = sigOp,
            )


    dg.addGroup("Ztautau",
        members = list(filter(lambda y: y.group == "Ztautau", dg.datasets.values())),
        label = r"Z$\to\tau\tau$",
        color = "darkblue",
        selectOp = sigOp,
    )


    if dg.wmass:
        dg.addGroup("Wtaunu",
            members = list(filter(lambda y: y.group == "Wtaunu", dg.datasets.values())),
            label = r"W$^{\pm}\to\tau\nu$",
            color = "orange",
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
    else:
        dg.addGroup("Other",
            members = [x for x in dg.datasets.values() if not x.is_data and x.group not in ["Zmumu", "Zee", "Ztautau", "QCD"]],
            label = "Other",
            color = "grey",
        )


    dg.filterGroups(filterGroups)
    dg.excludeGroups(excludeGroups)

    if dg.wmass:
        # add all processes to the fake contributions after filtered and excluded groups
        dg.addGroup(dg.fakeName,
            members = [member for sublist in [v.members for k, v in dg.groups.items() if k != "QCD"] for member in sublist],
            scale = lambda x: 1. if x.is_data else -1,
            label = "Nonprompt",
            color = "grey",
            selectOp = fakeOp,
        )
        dg.filterGroups(filterGroups)
        dg.excludeGroups(excludeGroups)


    return dg
