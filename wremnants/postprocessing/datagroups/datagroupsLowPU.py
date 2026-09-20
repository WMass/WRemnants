"""
Define the datagroups objects with individual datagroup objects and it's members.
This one is for the single muon/electron and dimuon/electron analysis at low PU used for 2017 analysis but it may be used also for other, similar analyses.
"""

from wums import logging

logger = logging.child_logger(__name__)


def make_datagroups_lowPU(dg, excludeGroups=None, filterGroups=None):
    # reset datagroups
    dg.groups = {}

    def add_group_if_nonempty(name, **kwargs):
        members = dg.get_members_from_results(**kwargs)
        if members:
            dg.addGroup(name, members=members)

    dg.addGroup(
        "Data",
        members=dg.get_members_from_results(is_data=True),
    )
    add_group_if_nonempty("Ztautau", startswith="Ztautau")

    if dg.flavor in ["mu", "mumu"]:
        add_group_if_nonempty("Zmumu", startswith="Zmumu")
        if dg.mode in ["w_lowpu", "met_lowpu"]:
            add_group_if_nonempty(
                "Wmunu",
                startswith=["Wplusmunu", "Wminusmunu", "Wmunu"],
            )

    if dg.flavor in ["e", "ee"]:
        add_group_if_nonempty("Zee", startswith="Zee")
        if dg.mode in ["w_lowpu", "met_lowpu"]:
            add_group_if_nonempty(
                "Wenu",
                startswith=["Wplusenu", "Wminusenu", "Wenu"],
            )

    if dg.mode in ["w_lowpu", "met_lowpu"]:
        add_group_if_nonempty(
            "Wtaunu",
            startswith=["Wplustaunu", "Wminustaunu", "Wtaunu"],
        )
        add_group_if_nonempty("Top", startswith=["Top", "SingleT", "TT"])
        add_group_if_nonempty("Diboson", startswith=["Diboson", "WW", "WZ", "ZZ"])
        add_group_if_nonempty("QCD", startswith="QCD")
    else:
        dg.addGroup(
            "Other",
            members=dg.get_members_from_results(
                not_startswith=["Zmumu", "Zee", "Ztautau", "QCD"]
            ),
        )

    dg.filterGroups(filterGroups)
    dg.excludeGroups(excludeGroups)

    if dg.mode == "w_lowpu":
        # add all processes to the fake contributions after filtered and excluded groups
        dg.addGroup(
            dg.fakeName,
            members=[
                member
                for sublist in [v.members for k, v in dg.groups.items() if k != "QCD"]
                for member in sublist
            ],
            scale=lambda x: 1.0 if x.is_data else -1,
        )
        dg.filterGroups(filterGroups)
        dg.excludeGroups(excludeGroups)

        # QCD MC and the data-driven nonprompt (Fake_mu/Fake_e) estimate both
        # model the same physics (the data-driven estimate is not subtracting
        # off QCD, see the exclusion above), so having both in the same plot
        # or fit double-counts that background. Require the caller to pick one
        # via --excludeProcs/--filterProcs rather than silently stacking both.
        if "QCD" in dg.groups and dg.fakeName in dg.groups:
            raise RuntimeError(
                "Both 'QCD' (MC) and "
                f"'{dg.fakeName}' (data-driven nonprompt) groups are present. "
                "These are two alternative estimates of the same background and "
                "must not be used together: exclude one with --excludeProcs."
            )

    return dg
