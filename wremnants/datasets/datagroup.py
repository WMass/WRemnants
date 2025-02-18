from copy import deepcopy

from utilities.styles import styles
from wums import logging

logger = logging.child_logger(__name__)


class Datagroup_member(object):
    def __init__(self, name, result):
        self.name = name
        self.xsec = result["dataset"]["xsec"]
        self.weight_sum = result["weight_sum"]
        self.is_data = result["dataset"].get("is_data", False)
        self.lumi = result.get("lumi", 0)


class Datagroup(object):

    def __init__(
        self, name, members={}, scale=None, memberOp=None, label=None, color=None
    ):
        self.name = name
        self.scale = scale
        self.label = styles.process_labels.get(name, name) if label is None else label
        self.color = styles.process_colors.get(name, "grey") if color is None else color
        if type(members) == dict:
            self.members = [Datagroup_member(k, v) for k, v in members.items()]
        else:
            self.members = members

        self.histselector = None  # selector object to perform selection operation
        self.memberOp = memberOp  # list of operations that is applied on single members

        self.hists = {}  # list of histograms processed from narf datasets

    def copy(self, new_name, member_filter=None):
        x = deepcopy(self)
        x.name = new_name

        if member_filter:
            # Invert the member filter and exclude those members
            x.deleteMembers(
                [m for m in filter(lambda x, f=member_filter: not f(x), x.members)]
            )

        logger.debug(
            f"Make a copy of group {self.name}, named {new_name} with members {[m.name for m in x.members]}"
        )
        return x

    def addMembers(self, members, member_operations=None):
        if callable(member_operations) or member_operations is None:
            for m in members:
                self.addMember(m, member_operations)
        elif len(member_operations) == len(members):
            for m, o in zip(members, member_operations):
                self.addMember(m, o)
        else:
            raise RuntimeError(
                "'member_operations' has to be a string or a list with the same length as 'members'!"
            )

    def addMember(self, member, member_operation=None):
        # adds a member to the existing members of a given group

        # add member operation
        if self.memberOp is None:
            self.memberOp = [None] * len(self.members)

        self.memberOp.append(deepcopy(member_operation))
        self.members.append(member)

    def deleteMembers(self, members=None):
        if members is None:
            # delete all members
            members = self.members[:]
        for m in members:
            self.deleteMember(m)

    def deleteMember(self, member):
        # deletes a process from the list of members of a given group

        if member not in [m for m in self.members]:
            logger.warning(
                f"The member {member.name} can not be found in the group {self.name}! Do nothing here."
            )
            return

        logger.debug(f"Delete member {member.name}!")

        mask = [m != member for m in self.members]
        if self.memberOp is not None:
            self.memberOp = [v for (v, i) in zip(self.memberOp, mask) if i]

        self.members = [m for (m, i) in zip(self.members, mask) if i]
