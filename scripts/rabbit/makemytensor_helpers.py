from pathlib import Path

import h5py
import hist
import matplotlib.pyplot as plt
import numpy as np

from utilities.io_tools import input_tools
from wums import boostHistHelpers as hh


def load_histogram(_filename: str, _dataset: str):
    h5file = h5py.File(_filename, "r")
    results = input_tools.load_results_h5py(h5file)

    h = results[_dataset]["output"]["nominal_HistToFit"].get()
    hvar = results[_dataset]["output"]["nominal_muonScaleSyst_responseWeights"].get()

    combined = None
    for proc in [
        "data2018Acharmonium",
        "data2018Bcharmonium",
        "data2018Ccharmonium",
        "data2018Dcharmonium",
    ]:
        hd = results[proc]["output"]["nominal_HistToFit"].get()
        combined = (
            hh.addHists(combined, hd.copy(), createNew=False) if combined else hd.copy()
        )

        # eta = "bkmm_jpsimc_kaon1eta"
        # unc = "unc"
        # du = "downUpVar"

        # check if any variation depends on eta for a given unc bin:
        # for eta_idx in [0, 6, 10, 20]:
        #    vals = hvar[{eta: eta_idx, unc: 0, du: 1}].values().sum()
        #    print(eta_idx, vals)

    hdata = combined
    return h, hdata, hvar


def rebin_histogram(h: hist.Hist, rebinning):
    result = h
    for axis, bins in rebinning.items():
        if axis not in result.axes.name:
            continue
        factor = result.axes[axis].size // bins
        result = result[{axis: hist.rebin(factor)}]
    return result


def rebin_variation_unc_axis(
    variation_hist: hist.Hist,
    eta_axis_name: str,
    unc_axis_name: str,
    labels,
    eta_rebin_factor: int,
):
    # The unc axis is ordered as (eta_idx * n_labels + label_idx), per
    # make_jpsi_crctn_unc_helper: each fine-eta bin gets its own A/e/M entry.
    # When eta is rebinned, we must rebuild unc to match the coarse eta bins by
    # summing the matching (eta, unc) pairs for the fine bins inside each coarse bin.
    if eta_axis_name not in variation_hist.axes.name:
        raise ValueError(
            f"Eta axis '{eta_axis_name}' not found in variation histogram."
        )
    if unc_axis_name not in variation_hist.axes.name:
        raise ValueError(
            f"Unc axis '{unc_axis_name}' not found in variation histogram."
        )
    if eta_rebin_factor <= 0:
        raise ValueError("eta_rebin_factor must be positive.")

    eta_axis = variation_hist.axes[eta_axis_name]
    unc_axis = variation_hist.axes[unc_axis_name]
    labels = tuple(labels)

    fine_eta_bins = eta_axis.size
    if fine_eta_bins % eta_rebin_factor != 0:
        raise ValueError(
            f"Eta axis size {fine_eta_bins} is not divisible by rebin factor {eta_rebin_factor}."
        )
    coarse_eta_bins = fine_eta_bins // eta_rebin_factor
    expected_unc_bins = fine_eta_bins * len(labels)
    if unc_axis.size != expected_unc_bins:
        raise ValueError(
            f"Unc axis size {unc_axis.size} does not match expected "
            f"{expected_unc_bins} (fine eta bins * labels)."
        )

    new_eta_axis = hist.axis.Regular(
        coarse_eta_bins,
        eta_axis.edges[0],
        eta_axis.edges[-1],
        name=eta_axis.name,
        underflow=eta_axis.traits.underflow,
        overflow=eta_axis.traits.overflow,
        growth=eta_axis.traits.growth,
        circular=eta_axis.traits.circular,
    )
    new_unc_axis = hist.axis.Integer(
        0,
        coarse_eta_bins * len(labels),
        name=unc_axis.name,
        underflow=unc_axis.traits.underflow,
        overflow=unc_axis.traits.overflow,
        growth=unc_axis.traits.growth,
    )

    axes = []
    for ax in variation_hist.axes:
        if ax.name == eta_axis_name:
            axes.append(new_eta_axis)
        elif ax.name == unc_axis_name:
            axes.append(new_unc_axis)
        else:
            axes.append(ax)

    new_hist = hist.Hist(*axes, storage=variation_hist.storage_type())
    name_to_idx = {ax.name: idx for idx, ax in enumerate(new_hist.axes)}

    for coarse_eta_idx in range(coarse_eta_bins):
        fine_start = coarse_eta_idx * eta_rebin_factor
        fine_end = fine_start + eta_rebin_factor
        for label_idx, _ in enumerate(labels):
            new_unc_bin = coarse_eta_idx * len(labels) + label_idx
            summed = None
            for fine_eta_idx in range(fine_start, fine_end):
                fine_unc_bin = fine_eta_idx * len(labels) + label_idx
                selection = {eta_axis_name: fine_eta_idx, unc_axis_name: fine_unc_bin}
                sliced = collapse_axes(
                    variation_hist[selection], [eta_axis_name, unc_axis_name]
                )
                summed = (
                    hh.addHists(summed, sliced, createNew=False)
                    if summed is not None
                    else sliced.copy()
                )

            index = [slice(None)] * len(new_hist.axes)
            index[name_to_idx[eta_axis_name]] = coarse_eta_idx
            index[name_to_idx[unc_axis_name]] = new_unc_bin
            index = tuple(index)

            new_hist.values()[index] = summed.values()
            if summed.variances() is not None and new_hist.variances() is not None:
                new_hist.variances()[index] = summed.variances()

    return new_hist


def assert_matching_axes(hists, ref, label="histogram"):
    ref_axes = ref.axes
    for name, h in hists.items():
        if h.axes.name != ref_axes.name:
            raise RuntimeError(
                f"{label} '{name}' axes {h.axes.name} do not match reference axes {ref_axes.name}."
            )
        if not all(np.allclose(a, ref_axes[i]) for i, a in enumerate(h.axes)):
            raise RuntimeError(
                f"{label} '{name}' axes edges do not match reference axes."
            )


def collapse_axes(h: hist.Hist, drop_axes):
    result = h
    for name in drop_axes:
        if name in result.axes.name:
            result = result[{name: hist.sum}]
    return result


def _project_to_curvature(h: hist.Hist, keep_axes):
    drop_axes = [name for name in h.axes.name if name not in keep_axes]
    return collapse_axes(h, drop_axes)


def _safe_ratio(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator, dtype=float),
        where=denominator != 0,
    )


def _reorder_hist_axes(h: hist.Hist, target_axis_order):
    current_names = tuple(h.axes.name)
    desired = tuple(target_axis_order)
    if current_names == desired:
        return h
    if set(current_names) != set(desired):
        raise RuntimeError(
            f"Cannot reorder histogram axes; mismatch between {current_names} and {desired}"
        )
    name_to_idx = {name: idx for idx, name in enumerate(current_names)}
    perm = [name_to_idx[name] for name in desired]
    axes = [h.axes[name] for name in desired]
    reordered = hist.Hist(*axes, storage=h._storage_type())
    reordered.values()[...] = np.transpose(h.values(), perm)
    variances = h.variances()
    if variances is not None:
        reordered.variances()[...] = np.transpose(variances, perm)
    return reordered


def _resolve_plot_labels(args):
    raw = args.plot_curvature_response
    if raw is None:
        return None
    if len(raw) == 0:
        return tuple(args.systematic_labels)
    invalid = [label for label in raw if label not in args.systematic_labels]
    if invalid:
        raise ValueError(
            f"Unknown systematic(s) requested for plotting: {', '.join(invalid)}. "
            f"Valid options are: {', '.join(args.systematic_labels)}."
        )
    return tuple(raw)


def plot_curvature_response(signal_hist, variation_hist, args, labels_to_plot):
    curvature_axis_name = args.curvature_axis
    charge_axis_name = args.charge_axis
    eta_axis_name = args.eta_axis

    if curvature_axis_name not in signal_hist.axes.name:
        print(
            f"[plot] Axis '{curvature_axis_name}' not found, skipping curvature plot."
        )
        return
    if charge_axis_name not in signal_hist.axes.name:
        print(f"[plot] Axis '{charge_axis_name}' not found, skipping curvature plot.")
        return
    if eta_axis_name not in signal_hist.axes.name:
        print(f"[plot] Axis '{eta_axis_name}' not found, skipping curvature plot.")
        return

    plot_dir = Path(args.plot_output or args.tensor_output)
    plot_dir.mkdir(parents=True, exist_ok=True)

    curvature_axis = signal_hist.axes[curvature_axis_name]
    charge_axis = signal_hist.axes[charge_axis_name]
    eta_axis = signal_hist.axes[eta_axis_name]

    keep_axes = {curvature_axis_name, charge_axis_name}
    n_charge_bins = len(charge_axis.centers)
    scale = args.plot_curvature_scale

    full_label_order = tuple(args.systematic_labels)

    for eta_idx, eta_center in enumerate(eta_axis.centers):
        eta_selection = {eta_axis_name: eta_idx}
        nominal_proj = _project_to_curvature(signal_hist[eta_selection], keep_axes)

        fig, ax = plt.subplots()
        plotted_any = False

        for label in labels_to_plot:
            offset = full_label_order.index(label)
            systematic_bin = len(full_label_order) * eta_idx + offset
            up_selection = {
                args.systematic_axis: systematic_bin,
                args.variation_axis: args.variation_up_index,
                eta_axis_name: eta_idx,
            }
            down_selection = {
                args.systematic_axis: systematic_bin,
                args.variation_axis: args.variation_down_index,
                eta_axis_name: eta_idx,
            }

            up_proj = _project_to_curvature(variation_hist[up_selection], keep_axes)
            down_proj = _project_to_curvature(variation_hist[down_selection], keep_axes)
            for charge_idx, charge_center in enumerate(charge_axis.centers):
                nom_curve = nominal_proj[{charge_axis_name: charge_idx}]
                up_curve = up_proj[{charge_axis_name: charge_idx}]
                down_curve = down_proj[{charge_axis_name: charge_idx}]

                nom_vals = nom_curve.values()
                if not np.any(nom_vals):
                    print(
                        "warning: did not find nominal values for systematic bin "
                        f"{systematic_bin} in eta bin {eta_idx} and charge bin {charge_idx}"
                    )
                    continue

                up_ratio = _safe_ratio(up_curve.values(), nom_vals)
                down_ratio = _safe_ratio(down_curve.values(), nom_vals)
                if scale != 1.0:
                    up_ratio = 1.0 + (up_ratio - 1.0) * scale
                    down_ratio = 1.0 + (down_ratio - 1.0) * scale

                color_idx = (offset * n_charge_bins + charge_idx) % 10
                color = f"C{color_idx}"
                label_base = f"{label} q={charge_center:g}"
                ax.plot(
                    curvature_axis.centers,
                    up_ratio,
                    color=color,
                    linestyle="-",
                    label=f"{label_base} up",
                )
                ax.plot(
                    curvature_axis.centers,
                    down_ratio,
                    color=color,
                    linestyle="--",
                    label=f"{label_base} down",
                )
                plotted_any = True

        if not plotted_any:
            plt.close(fig)
            continue

        ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
        ax.set_xlabel(f"{curvature_axis_name} (k)")
        ax.set_ylabel("k'/k")
        ax.set_title(f"Curvature response, eta bin {eta_idx} (center={eta_center:.2f})")

        ax.legend(ncol=2, fontsize="small", loc="upper right")

        counts_lines = []
        for charge_idx, charge_center in enumerate(charge_axis.centers):
            nom_curve = nominal_proj[{charge_axis_name: charge_idx}]
            counts = nom_curve.values()
            count_str = ", ".join(f"{int(round(val))}" for val in counts)
            counts_lines.append(f"q={charge_center:g}: {count_str}")

        if counts_lines:
            text = "Counts/bin\n" + "\n".join(counts_lines)
            ax.text(
                0.02,
                0.02,
                text,
                transform=ax.transAxes,
                fontsize="x-small",
                va="bottom",
                ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, linewidth=0),
            )

        fig.tight_layout()
        param = f"_{label}" if len(labels_to_plot) == 1 else ""
        outpath = plot_dir / f"curvature_response_eta{eta_idx}{param}.png"
        fig.savefig(outpath)
        plt.close(fig)


def plot_variation_projection(
    signal_hist,
    variation_hist,
    args,
    axis_name,
    eta_only=None,
    mode="yield",
    bin_axis=None,
):
    charge_axis_name = args.charge_axis
    eta_axis_name = args.eta_axis

    if not args.plot_output:
        raise ValueError(
            "--plot-output is required when using --plot-variation or --plot-variation-ratio."
        )

    if axis_name not in signal_hist.axes.name:
        print(f"[plot] Axis '{axis_name}' not found, skipping variation plot.")
        return
    if charge_axis_name not in signal_hist.axes.name:
        print(f"[plot] Axis '{charge_axis_name}' not found, skipping variation plot.")
        return
    if eta_axis_name not in signal_hist.axes.name:
        print(f"[plot] Axis '{eta_axis_name}' not found, skipping variation plot.")
        return
    if bin_axis is not None:
        if bin_axis == axis_name:
            raise ValueError(
                "--bin-plot-variation must be different from --plot-variation axis."
            )
        if bin_axis == charge_axis_name:
            raise ValueError("--bin-plot-variation cannot be the charge axis.")
        if bin_axis == eta_axis_name:
            raise ValueError("--bin-plot-variation cannot be the eta axis.")
        if bin_axis not in signal_hist.axes.name:
            print(f"[plot] Axis '{bin_axis}' not found, skipping variation plot.")
            return
    if bin_axis is None and args.overlay_bin_variations:
        raise ValueError("--overlay-bin-variations requires --bin-plot-variation.")

    plot_dir = Path(args.plot_output)
    plot_dir.mkdir(parents=True, exist_ok=True)

    axis = signal_hist.axes[axis_name]
    charge_axis = signal_hist.axes[charge_axis_name]
    eta_axis = signal_hist.axes[eta_axis_name]
    bin_axis_obj = signal_hist.axes[bin_axis] if bin_axis is not None else None

    keep_axes = {axis_name, charge_axis_name}
    if bin_axis is not None:
        keep_axes.add(bin_axis)
    n_charge_bins = len(charge_axis.centers)
    labels = tuple(args.systematic_labels)

    eta_indices = range(len(eta_axis.centers))
    if eta_only is not None:
        if eta_only < 0 or eta_only >= len(eta_axis.centers):
            raise ValueError(
                f"Requested eta bin {eta_only} outside valid range 0..{len(eta_axis.centers) - 1}."
            )
        eta_indices = [eta_only]

    bin_msg = ""
    if bin_axis is not None:
        bin_msg = f" in each '{bin_axis}' bin ({len(bin_axis_obj.centers)})"
    print(
        f"[plot] Variation {mode} projections for axis '{axis_name}' "
        f"across {len(eta_indices)} eta bin(s){bin_msg}."
    )

    for eta_idx in eta_indices:
        eta_selection = {eta_axis_name: eta_idx}
        nominal_proj = _project_to_curvature(signal_hist[eta_selection], keep_axes)

        for label in labels:
            offset = labels.index(label)
            systematic_bin = len(labels) * eta_idx + offset

            up_selection = {
                args.systematic_axis: systematic_bin,
                args.variation_axis: args.variation_up_index,
                eta_axis_name: eta_idx,
            }
            down_selection = {
                args.systematic_axis: systematic_bin,
                args.variation_axis: args.variation_down_index,
                eta_axis_name: eta_idx,
            }

            up_proj = _project_to_curvature(variation_hist[up_selection], keep_axes)
            down_proj = _project_to_curvature(variation_hist[down_selection], keep_axes)

            for charge_idx, charge_center in enumerate(charge_axis.centers):
                if args.debug_variation:
                    nom_curve_dbg = nominal_proj[{charge_axis_name: charge_idx}]
                    up_curve_dbg = up_proj[{charge_axis_name: charge_idx}]
                    down_curve_dbg = down_proj[{charge_axis_name: charge_idx}]
                    nom_vals_dbg = nom_curve_dbg.values()
                    up_vals_dbg = up_curve_dbg.values()
                    down_vals_dbg = down_curve_dbg.values()
                    if np.any(nom_vals_dbg):
                        max_up = np.max(np.abs(up_vals_dbg - nom_vals_dbg))
                        max_down = np.max(np.abs(down_vals_dbg - nom_vals_dbg))
                        print(
                            "[debug] eta "
                            f"{eta_idx} label {label} q {charge_center:g} "
                            f"nom_sum {nom_vals_dbg.sum():.6g} "
                            f"up_sum {up_vals_dbg.sum():.6g} "
                            f"down_sum {down_vals_dbg.sum():.6g} "
                            f"max|up-nom| {max_up:.6g} "
                            f"max|down-nom| {max_down:.6g}"
                        )
                bin_indices = [None]
                if bin_axis_obj is not None:
                    bin_indices = range(len(bin_axis_obj.centers))

                def _plot_curves(ax, nom_vals, up_vals, down_vals, legend_prefix):
                    if mode == "ratio":
                        ax.plot(
                            axis.centers,
                            np.ones_like(nom_vals, dtype=float),
                            color="C0",
                            linestyle="-",
                            label=f"{legend_prefix} nom",
                        )
                        ax.plot(
                            axis.centers,
                            up_vals,
                            color="C1",
                            linestyle="--",
                            label=f"{legend_prefix} up",
                        )
                        ax.plot(
                            axis.centers,
                            down_vals,
                            color="C2",
                            linestyle=":",
                            label=f"{legend_prefix} down",
                        )
                        ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
                        ax.set_ylabel("Variation / Nominal")
                    else:
                        ax.plot(
                            axis.centers,
                            nom_vals,
                            color="C0",
                            linestyle="-",
                            label=f"{legend_prefix} nom",
                        )
                        ax.plot(
                            axis.centers,
                            up_vals,
                            color="C1",
                            linestyle="--",
                            label=f"{legend_prefix} up",
                        )
                        ax.plot(
                            axis.centers,
                            down_vals,
                            color="C2",
                            linestyle=":",
                            label=f"{legend_prefix} down",
                        )
                        ax.set_ylabel("Yield")

                if bin_axis_obj is not None and args.overlay_bin_variations:
                    fig, ax = plt.subplots()
                    plotted_any = False
                    for bin_idx in bin_indices:
                        selection = {charge_axis_name: charge_idx, bin_axis: bin_idx}
                        nom_curve = nominal_proj[selection]
                        up_curve = up_proj[selection]
                        down_curve = down_proj[selection]

                        nom_vals = nom_curve.values()
                        if not np.any(nom_vals):
                            continue
                        scale = args.plot_variation_scale
                        if mode == "ratio":
                            up_vals = _safe_ratio(up_curve.values(), nom_vals)
                            down_vals = _safe_ratio(down_curve.values(), nom_vals)
                            if scale != 1.0:
                                up_vals = 1.0 + (up_vals - 1.0) * scale
                                down_vals = 1.0 + (down_vals - 1.0) * scale
                        else:
                            up_vals = up_curve.values()
                            down_vals = down_curve.values()
                            if scale != 1.0:
                                up_vals = nom_vals + (up_vals - nom_vals) * scale
                                down_vals = nom_vals + (down_vals - nom_vals) * scale
                        _plot_curves(
                            ax, nom_vals, up_vals, down_vals, f"{bin_axis}{bin_idx}"
                        )
                        plotted_any = True

                    if not plotted_any:
                        plt.close(fig)
                        continue

                    ax.set_xlabel(axis_name)
                    ax.set_title(
                        f"Variation {mode} projection {label}, eta bin {eta_idx} "
                        f"(center={eta_axis.centers[eta_idx]:.2f}), q={charge_center:g}"
                    )
                    ax.legend(fontsize="small", loc="upper right")
                    fig.tight_layout()
                    suffix = "ratio" if mode == "ratio" else "yield"
                    outpath = plot_dir / (
                        f"variation_{suffix}_{axis_name}_eta{eta_idx}_{label}_q{charge_center:g}_{bin_axis}overlay.png"
                    )
                    fig.savefig(outpath)
                    plt.close(fig)
                else:
                    for bin_idx in bin_indices:
                        fig, ax = plt.subplots()
                        selection = {charge_axis_name: charge_idx}
                        title_extra = ""
                        file_extra = ""
                        if bin_axis_obj is not None:
                            selection[bin_axis] = bin_idx
                            title_extra = f", {bin_axis} bin {bin_idx}"
                            file_extra = f"_{bin_axis}{bin_idx}"

                        nom_curve = nominal_proj[selection]
                        up_curve = up_proj[selection]
                        down_curve = down_proj[selection]

                        nom_vals = nom_curve.values()
                        if not np.any(nom_vals):
                            plt.close(fig)
                            continue
                        scale = args.plot_variation_scale
                        if mode == "ratio":
                            up_vals = _safe_ratio(up_curve.values(), nom_vals)
                            down_vals = _safe_ratio(down_curve.values(), nom_vals)
                            if scale != 1.0:
                                up_vals = 1.0 + (up_vals - 1.0) * scale
                                down_vals = 1.0 + (down_vals - 1.0) * scale
                        else:
                            up_vals = up_curve.values()
                            down_vals = down_curve.values()
                            if scale != 1.0:
                                up_vals = nom_vals + (up_vals - nom_vals) * scale
                                down_vals = nom_vals + (down_vals - nom_vals) * scale

                        _plot_curves(ax, nom_vals, up_vals, down_vals, "")

                        ax.set_xlabel(axis_name)
                        ax.set_title(
                            f"Variation {mode} projection {label}, eta bin {eta_idx} "
                            f"(center={eta_axis.centers[eta_idx]:.2f}), q={charge_center:g}{title_extra}"
                        )
                        ax.legend(fontsize="small", loc="upper right")

                        fig.tight_layout()
                        suffix = "ratio" if mode == "ratio" else "yield"
                        outpath = plot_dir / (
                            f"variation_{suffix}_{axis_name}_eta{eta_idx}_{label}_q{charge_center:g}{file_extra}.png"
                        )
                        fig.savefig(outpath)
                        plt.close(fig)
