import hist

from wums import boostHistHelpers as hh
from wums import logging

logger = logging.child_logger(__name__)

_passIsoName = "passIso"
_passMTName = "passMT"

_passIso = {_passIsoName: True}
_failIso = {_passIsoName: False}
_passMT = {_passMTName: True}
_failMT = {_passMTName: False}


def get_mt_selection(
    h, thresholdMT=40.0, axis_name_mt="mt", integrateLowMT=True, integrateHighMT=False
):
    if axis_name_mt in h.axes.name:
        s = hist.tag.Slicer()
        high = h.axes[axis_name_mt].index(thresholdMT + 0.001)
        failMT = s[: high : hist.sum] if integrateLowMT else s[:high:]
        passMT = s[high :: hist.sum] if integrateHighMT else s[high:]
        nameMT = axis_name_mt
    else:
        failMT = 0
        passMT = 1
        nameMT = _passMTName

    return nameMT, failMT, passMT


def fakeHistABCD(
    h,
    thresholdMT=40.0,
    fakerate_integration_axes=[],
    axis_name_mt="mt",
    integrateLowMT=True,
    integrateHighMT=False,
):
    # integrateMT=False keeps the mT axis in the returned histogram (can be used to have fakes vs mT)
    nameMT, failMT, passMT = get_mt_selection(
        h, thresholdMT, axis_name_mt, integrateLowMT, integrateHighMT
    )

    if any(a in h.axes.name for a in fakerate_integration_axes):
        fakerate_axes = [
            n
            for n in h.axes.name
            if n not in [*fakerate_integration_axes, _passIsoName, nameMT]
        ]
        hPassIsoFailMT = h[{**_passIso, nameMT: failMT}].project(*fakerate_axes)
        hFailIsoFailMT = h[{**_failIso, nameMT: failMT}].project(*fakerate_axes)
    else:
        hPassIsoFailMT = h[{**_passIso, nameMT: failMT}]
        hFailIsoFailMT = h[{**_failIso, nameMT: failMT}]

    hFRF = hh.divideHists(hPassIsoFailMT, hFailIsoFailMT, cutoff=1, createNew=True)

    return hh.multiplyHists(hFRF, h[{**_failIso, nameMT: passMT}])


def signalHistWmass(
    h,
    thresholdMT=40.0,
    charge=None,
    passIso=True,
    passMT=True,
    axis_name_mt="mt",
    integrateLowMT=True,
    integrateHighMT=False,
    genBin=None,
):
    if genBin != None:
        h = h[{"recoil_gen": genBin}]

    nameMT, fMT, pMT = get_mt_selection(
        h, thresholdMT, axis_name_mt, integrateLowMT, integrateHighMT
    )

    sel = {_passIsoName: passIso, nameMT: pMT if passMT else fMT}
    if charge in [-1, 1]:
        sel.update({"charge": -1j if charge < 0 else 1j})

    # remove ax slice if the ax does not exist
    for key in sel.copy().keys():
        if not key in [ax.name for ax in h.axes]:
            del sel[key]

    return h[sel]


# the following are utility wrapper functions for signalHistWmass with proper region selection
def histWmass_failMT_passIso(h, thresholdMT=40.0, charge=None):
    return signalHistWmass(h, thresholdMT, charge, True, False)


def histWmass_failMT_failIso(h, thresholdMT=40.0, charge=None):
    return signalHistWmass(h, thresholdMT, charge, False, False)


def histWmass_passMT_failIso(h, thresholdMT=40.0, charge=None):
    return signalHistWmass(h, thresholdMT, charge, False, True)


def histWmass_passMT_passIso(h, thresholdMT=40.0, charge=None):
    return signalHistWmass(h, thresholdMT, charge, True, True)
