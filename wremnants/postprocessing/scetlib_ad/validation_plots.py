"""Shared plot conventions for the scetlib_ad validation scripts.

The four validation scripts used to each hardcode their own ratio-panel range,
which made the same residual look very different depending on which script drew
it -- a 8e-4 reco residual filled an auto-zoomed frame while a 2.4e-3 gen
residual sat at under half height in a fixed +-0.5% one, so the better number
looked worse. One helper, used by all of them.
"""

import numpy as np


def ratio_range(*ratios, floor=1.0e-3, pad=1.3, cap=0.5):
    """Symmetric ``[lo, hi]`` ratio-panel range that actually contains the data.

    ``floor`` keeps a perfect ratio from collapsing onto a zero-height axis;
    ``pad`` leaves headroom above the largest deviation; ``cap`` bounds the
    window so one pathological bin cannot flatten everything else into a line.

    Returns ``(range, clipped)``. ``clipped`` is True when the data exceeds the
    capped window, i.e. when part of the curve will be drawn outside the panel
    -- the caller is expected to say so out loud. That case is not hypothetical:
    running the absolute comparison against the wrong reference put the model at
    3e-5 of it, and with a fixed +-3% panel the model's curve simply was not
    drawn. The plot showed the reference's own flat 1.000 self-ratio and read as
    perfect agreement.
    """
    dev = 0.0
    for r in ratios:
        r = np.asarray(r, dtype=float)
        finite = r[np.isfinite(r)]
        if finite.size:
            dev = max(dev, float(np.max(np.abs(finite - 1.0))))
    want = max(pad * dev, floor)
    clipped = want > cap
    half = min(want, cap)
    return [1.0 - half, 1.0 + half], clipped


def warn_if_clipped(clipped, dev_desc="the model/reference ratio"):
    """Print a loud, unmissable note when a curve leaves the ratio panel."""
    if clipped:
        print(
            f"    *** WARNING: {dev_desc} leaves the ratio panel, so part of "
            "the curve is NOT DRAWN. A flat line in that panel is the "
            "reference's own self-ratio, not agreement. Check the printed "
            "numbers, not the picture. ***",
            flush=True,
        )
