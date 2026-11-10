"""
Define the datagroups objects with individual datagroup objects and it's members.
This one is for the single muon and dimuon analysis at high PU used for 2016 postVFP analysis but it may be used also for other, similar analyses.
"""

from wums import logging

logger = logging.child_logger(__name__)

# Groups of several processes, by the prefixes of their member names. They pool the
# MC statistics of their components into one template per bin; each systematic keeps
# acting on the components it is defined for (Datagroups.resolveProcesses).
merged_groups = {
    "Z": {"Zmumu": ["Zmumu"], "Ztautau": ["Ztautau"]},
    "W": {
        "Wmunu": ["Wplusmunu", "Wminusmunu", "Wmunu"],
        "Wtaunu": ["Wplustaunu", "Wminustaunu"],
    },
    "TopVV": {
        "Top": ["Top", "SingleT", "TT"],
        "Diboson": ["Diboson", "WW", "WZ", "ZZ"],
    },
    # gamma q -> V q' and gamma gamma -> l l, with separate normalization uncertainties
    "PhotonInduced": {"PhotonInducedQG": ["QG"], "PhotonInducedGG": ["GG"]},
}


def add_merged_group(dg, name, split=False):
    """The group `name` of merged_groups, or with `split` its components as groups"""
    components = {
        c: dg.get_members_from_results(startswith=prefixes)
        for c, prefixes in merged_groups[name].items()
    }
    if split:
        for c, members in components.items():
            dg.addGroup(c, members=members)
        return
    dg.addGroup(
        name,
        members={k: v for members in components.values() for k, v in members.items()},
        components={c: list(members) for c, members in components.items() if members},
    )


def make_datagroups_2016(
    dg,
    pseudodata_pdfset=None,
    excludeGroups=None,
    filterGroups=None,
):
    # reset datagroups
    dg.groups = {}

    # merged groups whose components are groups of their own: Datagroups splitGroups
    # (True for all), e.g. in the fits where the muon channel is the signal and the tau
    # one a background, and W and Z for the gen-level fits
    to_split = dg.splitGroups if not isinstance(dg.splitGroups, bool) else []
    split = lambda name: (
        dg.splitGroups is True
        or name in to_split
        or (dg.mode == "vgen" and name in ["W", "Z"])
    )

    dg.addGroup(
        "Data",
        members=dg.get_members_from_results(is_data=True),
    )
    add_merged_group(dg, "Z", split=split("Z"))
    add_merged_group(dg, "PhotonInduced", split=split("PhotonInduced"))

    if pseudodata_pdfset:
        dg.addGroup(
            f"pdf{pseudodata_pdfset.upper()}_sum",
            label=f"pdf{pseudodata_pdfset.upper()}",
            color="dimgray",
        )
    if dg.mode in ["vgen", "w_mass"]:
        add_merged_group(dg, "W", split=split("W"))
        add_merged_group(dg, "TopVV", split=split("TopVV"))
        dg.addGroup(
            "QCD",
            members=dg.get_members_from_results(startswith=["QCD"]),
        )
    else:
        dg.addGroup(
            "Other",
            members=dg.get_members_from_results(
                not_startswith=["Zmumu", "Ztautau", "QCD", "GG", "QG"]
            ),
        )

    dg.filterGroups(filterGroups)
    dg.excludeGroups(excludeGroups)

    if dg.mode == "w_mass":
        # add all processes to the fake contributions after filtered and excluded groups
        dg.addGroup(
            "Fake",
            members=[
                member
                for sublist in [v.members for k, v in dg.groups.items() if k != "QCD"]
                for member in sublist
            ],
            scale=lambda x: 1.0 if x.is_data else -1,
        )
        dg.filterGroups(filterGroups)
        dg.excludeGroups(excludeGroups)

    return dg
