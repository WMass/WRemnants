#!/usr/bin/env python
"""Plot the qop_reco/qop_gen distribution from the MC-shifted/smeared
reference and the analytic-splines / ONNX reweight tests written by
``scripts/histmakers/w_z_muonresponse.py --testHelpers``.

Two figures are produced:

* ``<out>_smear.{pdf,png}`` overlays nominal, MC-smeared (truth),
  splines-reweight and ONNX-reweight; ratio panel against MC truth.
* ``<out>_scale.{pdf,png}`` same layout for the scale shift.

The input is the HDF5 produced by the histmaker. By default it
aggregates over all MC processes in the file (skipping data). A
specific process group can be selected with ``--process``.

Usage:
    python plot_muonresponse_reweight_closure.py \\
        --input w_z_muonresponse_..._maxFiles_*.hdf5 \\
        --output muonresponse_closure
"""

import argparse
import os
import sys
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Repo root, so the wremnants/wums imports work without an env setup.
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)
from wremnants.utilities.io_tools import base_io  # noqa: E402

SMEAR_PLOT = {
    "title": "Smear closure (qop_reco / qop_gen)",
    "nominal": "hist_qopr",
    "truth": "hist_qopr_smearedmulti",
    "truth_label": "MC smeared (sample, multi)",
    "variants": [
        ("hist_qopr_smeared_weight", "splines reweight"),
        ("hist_qopr_smeared_weight_onnx", "ONNX reweight"),
        ("hist_qopr_transformed", "splines transform"),
    ],
}

SCALE_PLOT = {
    "title": "Scale closure (qop_reco / qop_gen)",
    "nominal": "hist_qopr",
    "truth": "hist_qopr_shifted",
    "truth_label": "MC shifted",
    "variants": [
        ("hist_qopr_scaled_weight", "splines reweight"),
        ("hist_qopr_scaled_weight_onnx", "ONNX reweight"),
    ],
}


def load_aggregated_hists(path, process=None):
    """Return a dict {hist_name: 1D-projected-on-qopr ``hist.Hist``},
    summed over the requested MC processes."""
    # Pre-discover every histogram the plots care about.
    wanted = set()
    for plot in (SMEAR_PLOT, SCALE_PLOT):
        wanted.add(plot["nominal"])
        wanted.add(plot["truth"])
        for h, _ in plot["variants"]:
            wanted.add(h)

    # Keep the HDF5 open while resolving the lazy H5PickleProxy entries.
    with h5py.File(path, "r") as f:
        results = base_io.load_results_h5py(f)
        procs = [
            p
            for p in results
            if isinstance(results[p], dict) and "output" in results[p]
        ]
        if process is not None:
            procs = [p for p in procs if process in p]
        if not procs:
            raise SystemExit(f"No MC processes found in {path} (filter={process!r}).")

        summed = {}
        for proc in procs:
            out = results[proc]["output"]
            for name in wanted:
                if name not in out:
                    continue
                h = out[name].get() if hasattr(out[name], "get") else out[name]
                # Project on qopr axis only.
                h1 = h.project("qopr")
                summed[name] = h1 if name not in summed else summed[name] + h1

    missing = wanted - summed.keys()
    if missing:
        warnings.warn(f"missing histograms (will be skipped): {sorted(missing)}")
    return summed, procs


def _values_with_err(h):
    vals = h.values()
    var = h.variances() if h.variances() is not None else np.zeros_like(vals)
    err = np.sqrt(np.clip(var, 0.0, None))
    return vals, err


def _auto_ratio_ylim(ratios_in_view, pad=1.3, floor=0.005, ceiling=0.5):
    """Pick a symmetric y-range around 1 that contains every ratio
    value in ``ratios_in_view`` (a list of 1-D numpy arrays already
    restricted to the visible x-range and statistically-populated
    bins).  Padded by ``pad`` and clamped to [floor, ceiling]."""
    devs = []
    for r in ratios_in_view:
        r = r[np.isfinite(r)]
        if r.size:
            devs.append(np.max(np.abs(r - 1.0)))
    if not devs:
        return (1.0 - ceiling, 1.0 + ceiling)
    half = float(np.clip(pad * max(devs), floor, ceiling))
    return (1.0 - half, 1.0 + half)


def plot_one(
    summed, plot_spec, output_prefix, yscale="log", xlim=(0.95, 1.05), ratio_ylim=None
):
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(7.5, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.05},
    )

    nominal = summed.get(plot_spec["nominal"])
    truth = summed.get(plot_spec["truth"])
    if nominal is None or truth is None:
        warnings.warn(
            f"Skipping {plot_spec['title']!r}: missing"
            f" nominal={plot_spec['nominal']} or truth={plot_spec['truth']}"
        )
        plt.close(fig)
        return

    # The MC-smeared reference (``hist_qopr_smearedmulti``) is filled
    # with multiple smear replicas per muon (``SmearingHelperSimpleMulti``
    # at ``nreps=100`` in the histmaker), so its integral is N_reps ×
    # the nominal integral. Rescale to match the nominal integral so
    # the shape and the ratio-to-truth panel are comparable across
    # variants. For single-sample reference hists this is a no-op
    # (ratio ≈ 1).
    nominal_int = nominal.values().sum()
    truth_v_raw = truth.values()
    truth_int = truth_v_raw.sum()
    truth_scale = nominal_int / truth_int if truth_int > 0.0 else 1.0
    if not np.isclose(truth_scale, 1.0, atol=1e-3):
        print(
            f"  [{plot_spec['title']}] scaling truth hist"
            f" {plot_spec['truth']!r} by {truth_scale:.6g}"
            f" (integral {truth_int:.4g} -> {truth_int * truth_scale:.4g})"
        )

    edges = nominal.axes["qopr"].edges

    def step_arr(ax, v, label, color, ls="-"):
        ax.step(
            edges,
            np.concatenate([[v[0]], v]),
            where="pre",
            label=label,
            color=color,
            linestyle=ls,
            linewidth=1.4,
        )

    def step(ax, h, label, color, ls="-"):
        v, _ = _values_with_err(h)
        step_arr(ax, v, label, color, ls=ls)

    step(ax_top, nominal, "nominal", "black", ls=":")
    truth_v = truth_v_raw * truth_scale
    step_arr(ax_top, truth_v, plot_spec["truth_label"], "black")

    truth_safe = np.where(truth_v > 0.0, truth_v, np.nan)

    # Bin-center mask for "what is in the visible x-range" + a soft
    # statistical mask (only bins where the truth bulk is well-populated
    # contribute to the auto y-range -- avoids the ratio fitting to
    # noise in the far tails).
    centers = 0.5 * (edges[:-1] + edges[1:])
    in_view = (centers >= xlim[0]) & (centers <= xlim[1])
    truth_max = float(np.nanmax(truth_v)) if truth_v.size else 0.0
    well_populated = truth_v >= 0.01 * truth_max  # >= 1% of peak

    palette = ["tab:red", "tab:blue", "tab:green", "tab:purple"]
    ratios_in_view = []
    for (name, label), color in zip(plot_spec["variants"], palette):
        h = summed.get(name)
        if h is None:
            continue
        step(ax_top, h, label, color)
        v, _ = _values_with_err(h)
        ratio = v / truth_safe
        ax_bot.step(
            edges,
            np.concatenate([[ratio[0]], ratio]),
            where="pre",
            color=color,
            linewidth=1.2,
        )
        ratios_in_view.append(ratio[in_view & well_populated])

    ax_top.set_yscale(yscale)
    ax_top.set_ylabel("entries / bin")
    ax_top.set_title(plot_spec["title"])
    ax_top.legend(loc="upper right", fontsize=9)
    ax_top.grid(alpha=0.3)
    # Focus on the closure-relevant region around 1.0.
    ax_top.set_xlim(*xlim)
    if yscale == "log":
        ax_top.set_ylim(bottom=max(1e-3, nominal.values().max() * 1e-5))
    else:
        ax_top.set_ylim(bottom=0.0)

    ax_bot.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    ax_bot.set_ylabel("variant / truth")
    ax_bot.set_xlabel("qop_reco / qop_gen")
    if ratio_ylim is None:
        rlo, rhi = _auto_ratio_ylim(ratios_in_view)
    else:
        rlo, rhi = ratio_ylim
    ax_bot.set_ylim(rlo, rhi)
    ax_bot.grid(alpha=0.3)

    for ext in ("pdf", "png"):
        out = f"{output_prefix}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"wrote {out}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        "-i",
        required=True,
        help="HDF5 from w_z_muonresponse.py --testHelpers",
    )
    p.add_argument(
        "--output",
        "-o",
        default="muonresponse_closure",
        help="Output filename prefix (default: %(default)s)",
    )
    p.add_argument(
        "--process",
        default=None,
        help="Only aggregate over processes whose name contains this string.",
    )
    p.add_argument(
        "--ratio-ylim",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="Override the ratio panel y-limits (default: auto-zoomed to "
        "max |ratio-1| over well-populated bins in the visible x-range, "
        "clamped to ±[0.5%%, 50%%]).",
    )
    args = p.parse_args()

    summed, procs = load_aggregated_hists(args.input, process=args.process)
    print(f"aggregated over {len(procs)} process(es): {procs}")

    for ys in ("log", "lin"):
        ys_mpl = "log" if ys == "log" else "linear"
        plot_one(
            summed,
            SMEAR_PLOT,
            f"{args.output}_smear_{ys}",
            yscale=ys_mpl,
            ratio_ylim=args.ratio_ylim,
        )
        plot_one(
            summed,
            SCALE_PLOT,
            f"{args.output}_scale_{ys}",
            yscale=ys_mpl,
            ratio_ylim=args.ratio_ylim,
        )


if __name__ == "__main__":
    main()
