from utilities import boostHistHelpers as hh, logging

logger = logging.child_logger(__name__)

def make_datagroups_lowPU(dg, combine=False, excludeGroups=None, filterGroups=None):
    # reset datagroups
    dg.groups = {}

    dg.addGroup("Data",
        members = dg.get_members_from_results(is_data=True),
    )
    dg.addGroup("Ztautau",
        members = dg.get_members_from_results(startswith="Ztautau"),
    )

    if dg.flavor == "mu" or dg.flavor == "mumu":  
        dg.addGroup("Zmumu",
            members = dg.get_members_from_results(startswith="Zmumu"),
        ) 
        if dg.mode == "w_lowpu":
            dg.addGroup("Wmunu",
                members = dg.get_members_from_results(startswith=["Wplusmunu", "Wminusmunu"]),
            )

    if dg.flavor == "e" or dg.flavor == "ee":  
        dg.addGroup("Zee",
            members = dg.get_members_from_results(startswith="Zee"),
        ) 
        if dg.mode == "w_lowpu":
            dg.addGroup("Wenu",
                members = dg.get_members_from_results(startswith=["Wplusenu", "Wminusenu"]),
            )

    if dg.mode == "w_lowpu":
        dg.addGroup("Wtaunu",
            members = dg.get_members_from_results(startswith=["Wplustaunu", "Wminustaunu"]),
        )
        dg.addGroup("Top",
            members = dg.get_members_from_results(startswith=["Top", "SingleT", "TT"]),
        )
        dg.addGroup("Diboson",
            members = dg.get_members_from_results(startswith=["Diboson", "WW", "WZ", "ZZ"]),
        )
    else:
        dg.addGroup("Other",
            members = dg.get_members_from_results(not_startswith=["Zmumu", "Zee", "Ztautau", "QCD"]),
        )

    dg.filterGroups(filterGroups)
    dg.excludeGroups(excludeGroups)

    if dg.mode == "w_lowpu":
        # add all processes to the fake contributions after filtered and excluded groups
        dg.addGroup(dg.fakeName,
            members = [member for sublist in [v.members for k, v in dg.groups.items() if k != "QCD"] for member in sublist],
            scale = lambda x: 1. if x.is_data else -1,
        )
        dg.filterGroups(filterGroups)
        dg.excludeGroups(excludeGroups)


    return dg
