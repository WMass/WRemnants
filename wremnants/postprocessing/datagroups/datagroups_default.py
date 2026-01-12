from wums import logging

logger = logging.child_logger(__name__)


def make_datagroups(
    dg, combine=False, pseudodata_pdfset=None, excludeGroups=None, filterGroups=None
):
    # reset datagroups
    dg.groups = {}

    dg.addGroup(
        "Data",
        members=dg.get_members_from_results(is_data=True),
    )
    dg.addGroup(
        "Simulation",
        members=dg.get_members_from_results(is_data=False),
    )

    dg.filterGroups(filterGroups)
    dg.excludeGroups(excludeGroups)
