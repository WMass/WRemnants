# On results of fiducial inclusive cross sections and their ratios
# make a summary plot with different theory predictions
# make a latex summary table with the breakdown of uncertainties

import math

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, Polygon
from scipy.stats import chi2

import rabbit.io_tools
from wremnants.utilities import parsing
from wremnants.utilities.io_tools import tex_tools

from wums import logging, output_tools, plot_tools  # isort: skip

parser = parsing.plot_parser()
parser.add_argument("infile", type=str, help="Rabbit fitresult file")
parser.add_argument(
    "--pdfFiles",
    type=str,
    nargs="*",
    default=[],
    help="Rabbit fitresult files with alternative pdf predictions",
)
parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="Path to config file for style formatting",
)
parser.add_argument(
    "--full", action="store_true", default=False, help="full phase space results"
)
parser.add_argument(
    "--scaleToNewLumi",
    action="store_true",
    default=False,
    help="scale the results of SMP-20-004 to updated lumi",
)
parser.add_argument(
    "--includeSMP20004",
    action="store_true",
    default=False,
    help="include data points from SMP-20-004 (JHEP04(2025)162)",
)
parser.add_argument(
    "--theoryFiles",
    type=str,
    nargs="*",
    default=[],
    help="Theory prediction files in LABEL:filepath format. "
    "Supports DYTURBO 1D txt files and NNLOjet cross.dat files. "
    "Multiple files with the same label are combined into one prediction set.",
)
args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

config = plot_tools.load_config(args.config)

grouping = getattr(config, "nuisance_grouping", {}).get("xsecs", None)
translate_label = getattr(config, "systematics_labels", {})

fitresult, meta = rabbit.io_tools.get_fitresult(args.infile, meta=True)
result = fitresult["mappings"]

pdf_results = {}
comp_result = {}
for pdf_file in args.pdfFiles:
    pdf_name = pdf_file.split("/")[-2].split("_")[-1]
    pdf_name = pdf_name.upper().replace("AN3LO", "an3lo")

    logger.info(f"Now at PDF {pdf_name}")

    pdf_result, pdf_meta = rabbit.io_tools.get_fitresult(pdf_file, meta=True)

    pdf_mapping = pdf_result["mappings"]

    if "CompositeMapping" in pdf_mapping.keys():
        comp_result[pdf_name] = pdf_mapping["CompositeMapping"]
    else:
        pdf_results[pdf_name] = pdf_mapping

all_1d_pdf_names = list(pdf_results.keys()) + [
    k for k in comp_result.keys() if k != "TOTAL"
]
nPDFs = len(all_1d_pdf_names)
pdf_colors = {
    "CT18": "#2ca02c",
    "CT18Z": "#E42536",
    "PDF4LHC21": "#9467bd",
    "MSHT20": "#7f7f7f",
    "MSHT20an3lo": "#8c564b",
    "NNPDF31": "#e377c2",
    "NNPDF40": "#17becf",
}

# mapping, channel, bin
identifier = "_masked"
if args.full:
    identifier = f"_full{identifier}"
xsec_keys = [
    (
        r"$\mathrm{W}^{-}$",
        f"Project ch1{identifier} qGen",
        f"ch1{identifier}",
        {"qGen": 0},
    ),
    (
        r"$\mathrm{W}^{+}$",
        f"Project ch1{identifier} qGen",
        f"ch1{identifier}",
        {"qGen": 1},
    ),
    (r"$\mathrm{W}$", f"Project ch1{identifier}", f"ch1{identifier}", None),
    (r"$\mathrm{Z}$", f"Project ch0{identifier}", f"ch0{identifier}", None),
    (
        r"$\mathrm{W}^{+}/\mathrm{W}^{-}$",
        (
            f"Ratio ch1{identifier} ch1{identifier} qGen:1,ptGen:sum,absEtaGen:sum qGen:0,ptGen:sum,absEtaGen:sum",
            f"Ratio ch1{identifier} ch1{identifier} qGen:1 qGen:0",
        ),
        f"ch1{identifier}",
        None,
    ),
    (
        r"$\mathrm{W/Z}$",
        (
            f"Ratio ch1{identifier} ch0{identifier} qGen:sum,ptGen:sum,absEtaGen:sum ptVGen:sum,absYVGen:sum",
            f"Ratio ch1{identifier} ch0{identifier} qGen:sum count:sum",
        ),
        f"ch1{identifier}_ch0{identifier}",
        None,
    ),
]

smp_20_004 = {
    r"$\mathrm{W}^{-}$": [8670, 215.63858652847824],
    r"$\mathrm{W}^{+}$": [11800, 288.09720581775866],
    r"$\mathrm{W}$": [20480, 499.8999899979995],
    r"$\mathrm{Z}$": [1952, 48.63126566315131],
}

if args.scaleToNewLumi:
    # old lumi uncertainty 2.3%
    # new lumi uncertainty 0.9%
    for k, v in smp_20_004.items():
        smp_20_004[k] = [v[0], v[0] * ((v[1] / v[0]) ** 2 - 0.023**2 + 0.009**2) ** 0.5]

smp_20_004[r"$\mathrm{W}^{+}/\mathrm{W}^{-}$"] = [1.3615, 0.009570788891204319]
smp_20_004[r"$\mathrm{W/Z}$"] = [10.491, 0.0864002314811714]


# Map xsec_key display names to internal process keys
_name_to_proc = {
    r"$\mathrm{W}^{-}$": "wm",
    r"$\mathrm{W}^{+}$": "wp",
    r"$\mathrm{W}$": "w",
    r"$\mathrm{Z}$": "z",
    r"$\mathrm{W}^{+}/\mathrm{W}^{-}$": "wp_wm",
    r"$\mathrm{W/Z}$": "wz",
}

theory_colors = {
    "DYTURBO": "#ff7f0e",
    "NNLOjet": "#1f77b4",
    "SCETlib+DYTurbo": "#9467bd",
}


def _detect_process(path):
    """Detect W-/W+/Z from file path (filename or parent directories)."""
    parts = path.replace("\\", "/").split("/")
    for part in reversed(parts):
        p = part.lower()
        if p.startswith("results_wm") or p.startswith("wm") or "wminus" in p:
            return "wm"
        if p.startswith("results_wp") or p.startswith("wp") or "wplus" in p:
            return "wp"
        if (
            (p.startswith("results_z") or p.startswith("z"))
            and "wm" not in p
            and "wp" not in p
        ):
            return "z"
    raise ValueError(f"Cannot detect process (z/wm/wp) from path: {path}")


def _parse_dyturbo_1d(path):
    """Read total cross section from DYTURBO 1D rapidity file (last summary line).
    Returns (central_fb, unc_up_fb, unc_dn_fb); no scale variations in 1D files."""
    last = None
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                last = s
    cols = last.split()
    return float(cols[2]), 0.0, 0.0


def _parse_nnlojet_cross(path):
    """Parse NNLOjet cross.dat and return (central_fb, unc_up_fb, unc_dn_fb).
    Scale uncertainty is symmetric (quadrature of muR and muF half-ranges)."""
    labels, data = None, None
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("#labels:"):
                labels = s[len("#labels:") :].split()
            elif not s.startswith("#") and s:
                data = np.array(s.split(), dtype=float)
    col = {lbl.split("[")[0]: i for i, lbl in enumerate(labels)}
    vals = np.array([data[col[f"tot_scale0{i+1}"]] for i in range(7)])
    central = vals[3]  # scale04 = (muR=1, muF=1)
    delta_muR = (vals[5] - vals[1]) / 2  # (scale06 - scale02) / 2
    delta_muF = (vals[2] - vals[4]) / 2  # (scale03 - scale05) / 2
    unc = np.sqrt(delta_muR**2 + delta_muF**2)
    return central, unc, unc


def _parse_matrix_radish(path):
    """Parse MATRIX+RadISH pT distribution file and return (central_fb, unc_up_fb, unc_dn_fb).
    Columns: bin_idx, central, stat, scale_down, stat, scale_up, stat.
    Total cross section = sum of bins × bin width (1 GeV). Last row is a duplicate, skipped.
    """
    data = np.loadtxt(path)
    bins = data[:-1]  # drop last duplicate row
    bin_width = bins[1, 0] - bins[0, 0]  # should be 1 GeV
    central = np.sum(bins[:, 1]) * bin_width
    down = np.sum(bins[:, 3]) * bin_width
    up = np.sum(bins[:, 5]) * bin_width
    return central, up - central, central - down


def _parse_theory_file(path):
    with open(path) as f:
        first = f.readline().strip()
    if first.startswith("#labels: tot_scale01"):
        return _parse_nnlojet_cross(path)
    elif first.startswith("#ylo yhi PDF0"):
        return _parse_dyturbo_1d(path)
    elif not first.startswith("#"):
        # No header: assume MATRIX_RadISH 7-column pT distribution
        cols = first.split()
        if len(cols) == 7:
            return _parse_matrix_radish(path)
    raise ValueError(f"Unknown theory file format: {path}")


_SCETLIB_SCALE_VARS = [
    "mufdown",
    "mufup",
    "kappaFO0.5-kappaf2.",
    "kappaFO2.-kappaf0.5",
    "mufdown-kappaFO0.5-kappaf2.",
    "mufup-kappaFO2.-kappaf0.5",
]


def _parse_scetlib_pkl(path):
    """Parse a SCETlib+DYTurbo pkl.lz4 correction file.

    Returns dict: proc -> (central_pb, unc_up_pb, unc_dn_pb).
    CorrZ files produce {'z': ...}, CorrW files produce {'wm': ..., 'wp': ...}.
    Scale uncertainty = asymmetric quadrature of the 6 scale variation deviations.
    """
    import pickle

    import lz4.frame

    with lz4.frame.open(path, "rb") as f:
        data = pickle.load(f)

    results = {}
    for boson_key in [k for k in data if k in ("Z", "W")]:
        sub = data[boson_key]
        hist_key = next(k for k in sub if k.endswith("_hist"))
        h = sub[hist_key]
        vars_list = list(h.axes["vars"])
        vals = h.view(flow=True)["value"]
        # Axes: Q(underflow+1data+overflow), absY(no underflow, +overflow),
        #       qT(underflow+data+overflow), charge(no flow), vars(no underflow, +overflow)
        # Q data bin is always index 1 in the flow array.
        q_slice = 1

        def _sum_proc(charge_idx, var_idx):
            return float(np.sum(vals[q_slice, :, :, charge_idx, var_idx]))

        pdf0_idx = vars_list.index("pdf0")

        if boson_key == "Z":
            central = _sum_proc(0, pdf0_idx)
            deltas = np.array(
                [
                    _sum_proc(0, vars_list.index(sv)) - central
                    for sv in _SCETLIB_SCALE_VARS
                    if sv in vars_list
                ]
            )
            unc_up = float(np.sqrt(np.sum(np.where(deltas > 0, deltas, 0) ** 2)))
            unc_dn = float(np.sqrt(np.sum(np.where(deltas < 0, deltas, 0) ** 2)))
            results["z"] = (central, unc_up, unc_dn)
        elif boson_key == "W":
            # charge axis: index 0 = W- (charge < 0), index 1 = W+
            for ci, proc in [(0, "wm"), (1, "wp")]:
                central = _sum_proc(ci, pdf0_idx)
                deltas = np.array(
                    [
                        _sum_proc(ci, vars_list.index(sv)) - central
                        for sv in _SCETLIB_SCALE_VARS
                        if sv in vars_list
                    ]
                )
                unc_up = float(np.sqrt(np.sum(np.where(deltas > 0, deltas, 0) ** 2)))
                unc_dn = float(np.sqrt(np.sum(np.where(deltas < 0, deltas, 0) ** 2)))
                results[proc] = (central, unc_up, unc_dn)

    return results


# Build theory_predictions: label -> proc -> (central_pb, unc_up_pb, unc_dn_pb)
theory_predictions = {}
for tf in args.theoryFiles:
    label, path = tf.split(":", 1)
    if path.endswith(".pkl.lz4"):
        # SCETlib+DYTurbo: one file may contain multiple processes
        multi = _parse_scetlib_pkl(path)
        for proc, (c, u, d) in multi.items():
            theory_predictions.setdefault(label, {})[proc] = (c, u, d)
    else:
        proc = _detect_process(path)
        central_fb, unc_up_fb, unc_dn_fb = _parse_theory_file(path)
        theory_predictions.setdefault(label, {})[proc] = (
            central_fb / 1e3,
            unc_up_fb / 1e3,
            unc_dn_fb / 1e3,
        )

# Compute derived quantities (W, W+/W-, W/Z) from per-process values
for preds in theory_predictions.values():
    if "wm" in preds and "wp" in preds:
        wm, wm_u, wm_d = preds["wm"]
        wp, wp_u, wp_d = preds["wp"]
        preds["w"] = (wm + wp, np.sqrt(wm_u**2 + wp_u**2), np.sqrt(wm_d**2 + wp_d**2))
        r = wp / wm
        preds["wp_wm"] = (
            r,
            (
                r * np.sqrt((wp_u / wp) ** 2 + (wm_u / wm) ** 2)
                if wp > 0 and wm > 0
                else 0.0
            ),
            (
                r * np.sqrt((wp_d / wp) ** 2 + (wm_d / wm) ** 2)
                if wp > 0 and wm > 0
                else 0.0
            ),
        )
    if "w" in preds and "z" in preds:
        w, w_u, w_d = preds["w"]
        z, z_u, z_d = preds["z"]
        r = w / z
        preds["wz"] = (
            r,
            r * np.sqrt((w_u / w) ** 2 + (z_u / z) ** 2) if w > 0 and z > 0 else 0.0,
            r * np.sqrt((w_d / w) ** 2 + (z_d / z) ** 2) if w > 0 and z > 0 else 0.0,
        )

nTheory = len(theory_predictions)
nMeas = 1

lumi = meta["meta_info_input"]["channel_info"]["ch0"]["lumi"]

custom_order = [
    "Total",
    "stat",
    "binByBinStat",
    "luminosity",
    "Fake",
    "CMS_background",
    "muon_eff_syst",
    "muon_eff_stat",
    "prefire",
    "muonCalibration",
    "recoil",
    "pdfCT18Z",
    "angularCoeffs",
    "pTModeling",
    "theory_ew",
    "massAndWidths",
]

dfs = []
for name, mappings, channel, selection in xsec_keys:
    if len(mappings) == 2:
        mapping = mappings[0]
    else:
        mapping = mappings

    hp = result[mapping]["channels"][channel]["hist_prefit_inclusive"].get()
    h1 = result[mapping]["channels"][channel]["hist_postfit_inclusive"].get()
    hi = result[mapping]["channels"][channel][
        "hist_postfit_inclusive_gaussian_global_impacts_grouped"
    ].get()
    if selection is not None:
        hp = hp[selection]
        h1 = h1[selection]
        hi = hi[selection]
    if getattr(h1, "axes", False) and "yield" in h1.axes.name:
        hp = hp[{"yield": hist.sum}]
        h1 = h1[{"yield": hist.sum}]
        hi = hi[{"yield": hist.sum}]

    prefit = hp.value
    prefit_error = hp.variance**0.5

    value = h1.value
    error = h1.variance**0.5

    impacts = hi.values()

    labels = np.array(hi.axes["impacts"])
    mask = np.isin(labels, grouping)

    labels = labels[mask]
    impacts = impacts[mask]

    # if np.sum(impacts**2) ** 0.5 / error - 1 > 10e-10:
    #     raise RuntimeError(
    #         f"Sources don't add up to total error, got a difference of {np.sum(impacts**2)**0.5/error - 1}"
    #     )

    labels = np.append(labels, "Total")
    impacts = np.append(impacts, error)

    df = pd.DataFrame(np.array(impacts, dtype=np.float64).T, columns=["impact"])

    df["label"] = labels

    for label, combine_labels in {
        "binByBinStat": ["binByBinStatW", "binByBinStatZ", "binByBinStat"],
        "theory_ew": ["theory_ew", "massShift", "sin2thetaZ", "widthW", "widthZ"],
    }.items():

        subset = df[df["label"].isin(combine_labels)]
        subset = subset.fillna(0)
        combined = np.sqrt((subset[["impact"]] ** 2).sum())

        # df = df.drop(columns=["binByBinStatW", "binByBinStatZ", "massShift", "sin2thetaZ", "widthW", "widthZ"])

        new_row = pd.DataFrame(
            {
                "label": [label],
                "impact": [combined["impact"]],
            }
        )

        # Remove old rows and append the new one
        df = df[~df["label"].isin(combine_labels)]
        df = pd.concat([df, new_row], ignore_index=True)

    df["name"] = name
    df["value"] = value

    df["prefit"] = prefit
    df["prefit_error"] = prefit_error

    for pdf_name, pdf_res in pdf_results.items():
        if len(mappings) == 2:
            if mappings[0].replace(identifier, "") not in pdf_res.keys():
                mapping = mappings[1]
            else:
                mapping = mappings[0]
        else:
            mapping = mappings

        mapping = mapping.replace(identifier, "")
        channel_mappings = pdf_res[mapping]["channels"][channel.replace(identifier, "")]
        hr = channel_mappings["hist_prefit_inclusive"].get()
        hr_impacts = channel_mappings[
            "hist_prefit_inclusive_global_impacts_grouped"
        ].get()

        if selection is not None:
            hr = hr[selection]
            hr_impacts = hr_impacts[selection]
        if getattr(hr, "axes", False) and "yield" in hr.axes.name:
            hr = hr[{"yield": hist.sum}]
            hr_impacts = hr_impacts[{"yield": hist.sum}]

        df[pdf_name] = hr.value
        df[f"{pdf_name}_error"] = hr.variance**0.5
        df[f"{pdf_name}_pdf"] = hr_impacts[{"impacts": f"pdf{pdf_name}"}]

    for pdf_name, comp_res in comp_result.items():
        if pdf_name == "TOTAL":
            continue

        candidates = [mappings] if not isinstance(mappings, tuple) else list(mappings)
        channel_data = None
        for m in candidates:
            ckey = f"{m.replace(identifier, '')} {channel.replace(identifier, '')}"
            if ckey in comp_res["channels"]:
                channel_data = comp_res["channels"][ckey]
                break
        if channel_data is None:
            continue
        fittype = (
            "postfit" if "hist_postfit_inclusive" in channel_data.keys() else "prefit"
        )
        hr = channel_data[f"hist_{fittype}_inclusive"].get()

        if selection is not None:
            hr = hr[selection]
        if getattr(hr, "axes", False) and "yield" in hr.axes.name:
            hr = hr[{"yield": hist.sum}]

        df[pdf_name] = hr.value
        df[f"{pdf_name}_error"] = hr.variance**0.5

        if f"hist_{fittype}_inclusive_global_impacts_grouped" in channel_data:
            hr_impacts = channel_data[
                f"hist_{fittype}_inclusive_global_impacts_grouped"
            ].get()
            if selection is not None:
                hr_impacts = hr_impacts[selection]
            if getattr(hr_impacts, "axes", False) and "yield" in hr_impacts.axes.name:
                hr_impacts = hr_impacts[{"yield": hist.sum}]
            df[f"{pdf_name}_pdf"] = hr_impacts[{"impacts": f"pdf{pdf_name}"}]

    # Convert 'labels' column to categorical with the custom order
    df["label"] = pd.Categorical(df["label"], categories=custom_order, ordered=True)

    df["source"] = df["label"].apply(lambda l: translate_label.get(l, l))

    df = df.sort_values("label", ascending=False)

    dfs.append(df)

df = pd.concat(dfs)

logger.debug(df)

names = [k[0] for k in xsec_keys]

outdir = output_tools.make_plot_dir(args.outpath, eoscp=args.eoscp)

# make latex table
outname = "summary_table"
if args.postfix:
    outname += f"_{args.postfix}"

df_t = df.copy()
relative = True  # compute relative uncertainty
percentage = True  # numbers in percentage

if relative:
    df_t["impact"] /= df_t["value"]
if percentage:
    df_t["impact"] *= 100

# sorting
cat_dtype = pd.CategoricalDtype(categories=names, ordered=True)
df_t["name"] = df_t["name"].astype(cat_dtype)

tex_tools.make_latex_table(
    df_t,
    output_dir=outdir,
    output_name=outname,
    column_title=None,
    caption="Uncertainties in percentage.",
    label="",
    sublabel="",
    column_name="name",
    row_name="source",
    cell_columns=["impact"],
    cell_format=lambda x: f"${round(x,2)}$",
    sort="impact",
)

# make plot 1D
hep.style.use(hep.style.ROOT)

plt.clf()
fig = plt.figure()
fig.subplots_adjust(left=0.15, right=0.99, top=0.99, bottom=0.125)
ax = fig.add_subplot(111)

# x axis range
lo, hi = 0.97, 1.115

# totals = []
# stats = []
norms = []
_theory_labels_added = set()
for i, name in enumerate(names[::-1]):
    df_g = df.loc[df["name"] == name]

    norm = df_g["value"].values[0]
    total = df_g.loc[df_g["label"] == "Total"]["impact"].values[0]
    stat = df_g.loc[df_g["label"] == "stat"]["impact"].values[0]
    total_rel = total / norm
    stat_rel = stat / norm

    prefit = df_g["prefit"].values[0] / norm
    prefit_err = df_g["prefit_error"].values[0] / norm

    norms.append(norm)
    # totals.append(total)
    # stats.append(stat)

    x1 = ax.bar(
        1.0, height=1, bottom=i, width=2 * total_rel, color="silver"  # , label="Total"
    )
    x2 = ax.bar(
        1.0, height=1, bottom=i, width=2 * stat_rel, color="gold"  # , label="Stat"
    )

    # ax.errorbar([prefit], [i+0.5], xerr=prefit_err, color="red", marker="o", label="Prefit" if i ==0 else None)

    nSlots = nPDFs + nTheory + nMeas
    j = 0
    for j, pdf_name in enumerate(all_1d_pdf_names):
        pdf_value = df_g[pdf_name].values[0] / norm
        pdf_error = df_g[f"{pdf_name}_error"].values[0] / norm
        ax.errorbar(
            [pdf_value],
            [i + 1 - (j + 1) / nSlots],
            xerr=pdf_error,
            color=pdf_colors[pdf_name],
            marker="o",
            label=pdf_name if i == 0 else None,
        )
        if f"{pdf_name}_pdf" in df_g.columns:
            pdf_error_pdf = df_g[f"{pdf_name}_pdf"].values[0] / norm
            ax.errorbar(
                [pdf_value],
                [i + 1 - (j + 1) / nSlots],
                xerr=pdf_error_pdf,
                color=pdf_colors[pdf_name],
                capsize=5,
                capthick=2,
                marker="o",
            )

    proc = _name_to_proc.get(name)
    for k, (theo_label, preds) in enumerate(theory_predictions.items()):
        if proc not in preds:
            continue
        central_pb, unc_up_pb, unc_dn_pb = preds[proc]
        color = next(
            (v for key, v in theory_colors.items() if theo_label.startswith(key)),
            f"C{k + 5}",
        )
        legend_label = theo_label if theo_label not in _theory_labels_added else None
        if legend_label is not None:
            _theory_labels_added.add(theo_label)
        has_unc = unc_up_pb > 0 or unc_dn_pb > 0
        ax.errorbar(
            [central_pb / norm],
            [i + 1 - (nPDFs + k + 1) / nSlots],
            xerr=[[unc_dn_pb / norm], [unc_up_pb / norm]] if has_unc else None,
            color=color,
            marker="^",
            label=legend_label,
        )

    # cross sections from SMP-20-004
    if args.includeSMP20004:
        ax.errorbar(
            smp_20_004[name][0] / norm,
            i + 0.5,
            xerr=smp_20_004[name][1] / norm,
            color="black",
            marker="x",
            label="JHEP04(2025)162" if i == 0 else None,
        )

    # round to two significant digits in total uncertainty
    sig_digi = 2 - int(math.floor(math.log10(abs(total)))) - 1

    if sig_digi <= 0:
        norm = int(norm)
        total = int(total)
    else:
        norm = round(norm, sig_digi)
        total = round(total, sig_digi)

    ax.text(
        lo + 0.005,
        i + 0.5,
        name,
        fontsize=20,
        verticalalignment="bottom",
        horizontalalignment="left",
    )
    title = rf"${norm} \pm {total}"
    if "/" in name:
        title += "$"
    else:
        title += r"\,\mathrm{pb}$"
    ax.text(
        hi - 0.04,
        i + 0.5,
        title,
        fontsize=20,
        verticalalignment="bottom",
        horizontalalignment="left",
    )

# ax.text(
#     hi - 0.04,
#     len(names) + 0.5,
#     r"$\mathrm{Measured} \pm {unc}$",
#     fontsize=20,
#     verticalalignment="bottom",
#     horizontalalignment="left",
# )

x0 = ax.plot([1.0, 1.0], [0, len(norms)], color="black")
ax.plot([lo, hi], [len(norms), len(norms)], color="black")

p = Polygon(
    [[0, 0], [0, 0], [0, 0], [0, 0]],
    facecolor="silver",
    linestyle="solid",
    edgecolor="black",
    linewidth=2,
    alpha=0.6,
)

p.outer_color = "grey"
p.outer_alpha = 0.5
p.inner_color = "gold"
p.inner_alpha = 1

extra_handles = [(p,)]

extra_labels = ["Measurement"]

leg = plot_tools.addLegend(
    ax,
    ncols=args.legCols,
    text_size="small",
    bbox_to_anchor=None,
    loc="upper left",
    reverse=False,
    markerfirst=True,
    labelcolor="black",
    extra_handles=extra_handles,
    extra_labels=extra_labels,
    extra_entries_first=False,
    custom_handlers=(["doubleband"]),
    padding_loc="auto",
)

# add a link to the legend entry
for text in leg.get_texts():
    if "JHEP04(2025)162" in text.get_text():
        text.set_url("https://doi.org/10.1007/JHEP04(2025)162")
        # Optional: Make it look like a link
        # text.set_color("blue")

ax.set_xlim([lo, hi])
ax.set_ylim([0, len(norms) + 2])

ax.set_xlabel("Prediction / Measurement")

# Disable ticks on the top and right axes
ax.tick_params(top=False)

# Disable y-axis labels and ticks
plt.gca().set_yticklabels([])
plt.gca().set_yticks([])

plot_tools.add_cms_decor(ax, args.cmsDecor, data=True, lumi=lumi, loc=args.logoPos)

outname = "summary"
if args.full:
    outname = f"total_{outname}"
if args.postfix:
    outname += f"_{args.postfix}"
plot_tools.save_pdf_and_png(outdir, outname)

xsec_summary = {}
for name, mappings, channel, selection in xsec_keys:
    df_g = df.loc[df["name"] == name]
    value = df_g["value"].values[0]
    total = df_g.loc[df_g["label"] == "Total"]["impact"].values[0]
    stat = df_g.loc[df_g["label"] == "stat"]["impact"].values[0]

    entry = {
        "value": value,
        "stat": stat,
        "total": total,
    }

    for pdf_name in all_1d_pdf_names:
        if pdf_name in df_g.columns:
            pdf_entry = {
                "value": df_g[pdf_name].values[0],
                "total": df_g[f"{pdf_name}_error"].values[0],
            }
            if f"{pdf_name}_pdf" in df_g.columns:
                pdf_entry["pdf"] = df_g[f"{pdf_name}_pdf"].values[0]
            entry[pdf_name] = pdf_entry

    if args.includeSMP20004:
        entry["JHEP04(2025)162"] = {
            "value": smp_20_004[name][0],
            "total": smp_20_004[name][1],
        }

    xsec_summary[name] = entry

output_tools.write_index_and_log(
    outdir,
    outname,
    analysis_meta_info={
        "CombinetfOutput": meta["meta_info"],
        "xsec_summary": xsec_summary,
    },
    args=args,
)


# make plot 2D ellipses


def plot_cov_ellipse(cov, pos, nstd=1, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the covariance matrix (`cov`)
    centered at position `pos`.
    """
    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort eigenvalues and eigenvectors
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # Compute angle in degrees
    theta = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

    prob = chi2.cdf(nstd**2, df=1)  # Get the probability of the 1D n-sigma
    scale = np.sqrt(chi2.ppf(prob, df=2))  # Find the 2D equivalent scale

    # Width and height are "2*nstd" standard deviations
    width, height = 2 * scale * np.sqrt(eigvals)

    # Create ellipse
    ellipse = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    return ellipse


plot_2d_configs = {
    False: (  # fiducial phase space
        ("WpWm", xsec_keys[0], xsec_keys[1], "pb", (3065, 3200), (3975, 4200)),
        ("WZ", xsec_keys[2], xsec_keys[3], "pb", (7050, 7375), (585, 615)),
        ("R", xsec_keys[4], xsec_keys[5], None, (1.297, 1.304), (11.9, 12.2)),
    ),
    True: (  # total phase space
        ("WpWm", xsec_keys[0], xsec_keys[1], "pb", (7600, 9600), (10200, 12600)),
        ("WZ", xsec_keys[2], xsec_keys[3], "pb", (18000, 22000), (1720, 2100)),
        ("R", xsec_keys[4], xsec_keys[5], None, (1.32, 1.41), (10.1, 10.85)),
    ),
}

for name, channel0, channel1, unit, xlim, ylim in plot_2d_configs[args.full]:

    plt.clf()
    fig = plt.figure()
    fig.subplots_adjust(left=0.15, right=0.99, top=0.99, bottom=0.125)
    ax = fig.add_subplot(111)

    for pdf_name, result in comp_result.items():

        if len(channel0[1]) == 2:
            mappings0 = channel0[1]
            mappings1 = channel1[1]

            if args.full and pdf_name != "TOTAL":
                mapping0 = mappings0[1]
                mapping1 = mappings1[1]
            else:
                mapping0 = mappings0[0]
                mapping1 = mappings1[0]
        else:
            mapping0 = channel0[1]
            mapping1 = channel1[1]

        ckey0 = mapping0 + " " + channel0[2]
        ckey1 = mapping1 + " " + channel1[2]

        found_x = False
        found_y = False
        ibin = 0
        for k, r in result["channels"].items():
            fittype = "postfit" if f"hist_postfit_inclusive" in r.keys() else "prefit"

            hi = r[f"hist_{fittype}_inclusive"].get()
            if getattr(hi, "axes", False) and "yield" in hi.axes.name:
                hi = hi[{"yield": hist.sum}]

            if k == ckey0 or k == ckey0.replace(identifier, ""):
                sel = channel0[-1]
                found_x = True
                if sel is not None:
                    x = hi[sel].value
                    ix = ibin + [i for i in sel.values()][0]
                else:
                    x = hi.value
                    ix = ibin

            if k == ckey1 or k == ckey1.replace(identifier, ""):
                sel = channel1[-1]
                found_y = True
                if sel is not None:
                    y = hi[sel].value
                    iy = ibin + [i for i in sel.values()][0]
                else:
                    y = hi.value
                    iy = ibin

            ibin += hi.size if hasattr(hi, "size") else 1

        if not found_x or not found_y:
            raise RuntimeError(f"Not found x {found_x} or y {found_y}")

        cov = result[f"hist_{fittype}_inclusive_cov"].get().values()
        cov = cov[np.ix_([ix, iy], [ix, iy])]

        # for pos, cov in zip(points, covs):
        if fittype == "postfit":
            ell = plot_cov_ellipse(
                cov,
                np.array([x, y]),
                nstd=1,
                edgecolor="none",
                facecolor="grey",
                label="Measurement",
            )
            ax.add_patch(ell)
            ax.plot(x, y, color="black", marker="P")
        else:
            icol = pdf_colors[pdf_name]
            ell = plot_cov_ellipse(
                cov,
                np.array([x, y]),
                nstd=1,
                edgecolor=icol,
                facecolor="none",
                linewidth=2,
                label=pdf_name,
            )
            ax.add_patch(ell)
            ax.plot(x, y, color=icol, marker="o", alpha=0)

    if args.includeSMP20004:
        # cross sections from SMP-20-004 — correlations unknown, show as 2D error bars
        x = smp_20_004[channel0[0]][0]
        x_err = smp_20_004[channel0[0]][1]
        y = smp_20_004[channel1[0]][0]
        y_err = smp_20_004[channel1[0]][1]

        ax.errorbar(
            x,
            y,
            xerr=x_err,
            yerr=y_err,
            fmt="x",
            color="black",
            ecolor="black",
            capsize=4,
            capthick=1.5,
            label="JHEP04(2025)162",
        )
    if xlim is not None:
        ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        yrange = ylim[1] - ylim[0]

        ax.set_ylim(ylim[0], ylim[1] + yrange * 0.25)

    if unit is not None:
        plt.xlabel(rf"$\sigma({channel0[0].replace('$', '')})$ [pb]")
        plt.ylabel(rf"$\sigma({channel1[0].replace('$', '')})$ [pb]")
    else:
        plt.xlabel(channel0[0])
        plt.ylabel(channel1[0])
    # plt.title("2D Covariance Ellipses")
    # plt.grid(True)
    # plt.show()

    plot_tools.addLegend(
        ax,
        ncols=args.legCols,
        text_size="small",
        bbox_to_anchor=None,
        loc="upper left",
        reverse=False,
        # markerfirst=True,
        labelcolor="black",
        # extra_handles=extra_handles,
        # extra_labels=extra_labels,
        # extra_entries_first=False,
        # custom_handlers=(
        #     ["doubleband"]
        # ),
        padding_loc="auto",
    )

    plot_tools.add_cms_decor(ax, args.cmsDecor, data=True, loc=args.logoPos)

    outname = f"summary_2D_{name}"
    if args.full:
        outname = f"total_{outname}"
    if args.postfix:
        outname += f"_{args.postfix}"
    plot_tools.save_pdf_and_png(outdir, outname)

    output_tools.write_index_and_log(
        outdir,
        outname,
        analysis_meta_info={"CombinetfOutput": meta["meta_info"]},
        args=args,
    )


if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
    output_tools.copy_to_eos(args.outpath, args.outfolder)
