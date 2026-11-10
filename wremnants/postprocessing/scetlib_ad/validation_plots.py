"""Shared ratio-panel conventions for the scetlib_ad validation scripts."""

import numpy as np


def ratio_range(*ratios, floor=1.0e-3, pad=1.3, cap=0.5):
    """Symmetric ``[lo, hi]`` ratio-panel range that contains the data.

    ``floor`` is the minimum half-width, ``pad`` the headroom factor above the
    largest deviation from 1, ``cap`` the maximum half-width.

    Returns ``(range, clipped)``; ``clipped`` is True when the data exceeds the
    capped window, so part of the curve is not drawn and the caller must warn
    (a flat line in the panel would otherwise read as perfect agreement).
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
