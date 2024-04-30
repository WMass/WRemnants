from utilities import boostHistHelpers as hh, logging

logger = logging.child_logger(__name__)
    
def make_datagroups_2016(dg, combine=False, pseudodata_pdfset = None, excludeGroups=None, filterGroups=None):
    # reset datagroups
    dg.groups = {}

    dg.addGroup("Data",
        members = dg.get_members_from_results(is_data=True),
    )
    dg.addGroup("Zmumu",
        members = dg.get_members_from_results(startswith=["Zmumu"]),
    ) 
    dg.addGroup("Ztautau",
        members = dg.get_members_from_results(startswith=["Ztautau"]),
    )
    dg.addGroup("PhotonInduced",
        members = dg.get_members_from_results(startswith=["GG", "QG"]),
    )

    if pseudodata_pdfset and dg.combine:
        dg.addGroup(f"pdf{pseudodata_pdfset.upper()}_sum",
            label = f"pdf{pseudodata_pdfset.upper()}",
            color = "dimgray"
        )
    if dg.mode in ["vgen", "w_mass"]:
        dg.addGroup("Wmunu",
            members = dg.get_members_from_results(startswith=["Wplusmunu", "Wminusmunu", "Wmunu"]),
        )
        dg.addGroup("Wtaunu",
            members = dg.get_members_from_results(startswith=["Wplustaunu", "Wminustaunu"]),
        )
        dg.addGroup("DYlowMass",
            members = dg.get_members_from_results(startswith=["DYlowMass", "DYJetsToMuMuMass10to50"]),
        )
        dg.addGroup("Top",
            members = dg.get_members_from_results(startswith=["Top", "SingleT", "TT"]),
        )
        dg.addGroup("Diboson",
            members = dg.get_members_from_results(startswith=["Diboson", "WW", "WZ", "ZZ"]),
        )
        dg.addGroup("QCD",
            members = dg.get_members_from_results(startswith=["QCD"]),
        )   
    else:
        dg.addGroup("Other",
            members = dg.get_members_from_results(not_startswith=["Zmumu", "Ztautau", "QCD", "GG", "QG"]),
        )

    dg.filterGroups(filterGroups)
    dg.excludeGroups(excludeGroups)

    if dg.mode == "w_mass":
        # add all processes to the fake contributions after filtered and excluded groups
        dg.addGroup("Fake",
            members = [member for sublist in [v.members for k, v in dg.groups.items() if k != "QCD"] for member in sublist],
            scale = lambda x: 1. if x.is_data else -1,
        )
        dg.filterGroups(filterGroups)
        dg.excludeGroups(excludeGroups)

    return dg
