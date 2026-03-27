from wums import logging

logger = logging.child_logger(__name__)


def make_datagroups_btojpsik(
    dg, combine=False, pseudodata_pdfset=None, excludeGroups=None, filterGroups=None
):
    # reset datagroups
    dg.groups = {}

    dg.addGroup(
        "Data",
        members=dg.get_members_from_results(is_data=True),
    )
    dg.addGroup(
        "BuToJpsiK",
        members=dg.get_members_from_results(startswith=["BuToJpsiK"]),
    )
    dg.addGroup(
        "signalBuToJpsiK",
        members=dg.get_members_from_results(startswith=["signalBuToJpsiK"]),
    )
    dg.addGroup(
        "BuToJpsiPi",
        members=dg.get_members_from_results(startswith=["BuToJpsiPi"]),
    )
    dg.addGroup(
        "Other",
        members=dg.get_members_from_results(
            not_startswith=["signalBuToJpsiK", "BuToJpsiK", "BuToJpsiPi"]
        ),
    )

    dg.filterGroups(filterGroups)
    dg.excludeGroups(excludeGroups)
