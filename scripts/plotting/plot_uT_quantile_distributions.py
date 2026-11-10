"""Plot the recoUT (uT) distributions used to construct the in-situ uT
quantiles, from the ``utQuantileInput_{failIso,failHLT}`` histograms emitted by
mz_dilepton.py --makeUTQuantileHists.

Three levels of granularity (folder nesting matches the levels):
  1. <plotdir>/<cat>/                          uT projection (integrate eta, pt)
  2. <plotdir>/<cat>/eta<i>/                    uT per eta slice (integrate pt)
  3. <plotdir>/<cat>/eta<i>/pt<j>/              uT per (eta, pt) cell

failHLT carries a charge axis -> each plot is split into q+/q-; failIso is
charge-inclusive. On the level-3 per-cell plots the Q=7 quantile boundaries
(equal-population edges of the per-cell recoUT CDF, exactly as
make_quantile_helper computes them) are overlaid as dashed lines, so one can
judge whether the quantile binning is sensible.

The histogram is loaded once and all slices are looped internally (the file is
~1 GB, so per-slice subprocess calls are not viable).
"""

import argparse
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from wremnants.postprocessing.datagroups.datagroups import Datagroups
from wums import logging, output_tools, plot_tools

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--inputFile", type=str, required=True, help="histmaker hdf5"
    )
    parser.add_argument(
        "--plotdir", type=str, required=True, help="output base directory"
    )
    parser.add_argument(
        "--process", type=str, default="Zmumu", help="process group to plot"
    )
    parser.add_argument(
        "--nQuantiles", type=int, default=7, help="number of uT quantiles"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=["failIso", "failHLT"],
        help="categories to plot (failIso is charge-inclusive)",
    )
    parser.add_argument("--eoscp", action="store_true")
    return parser.parse_args()


CHARGE_META = {-1: ("minus", r"$\mu^{-}$"), +1: ("plus", r"$\mu^{+}$")}


def quantile_edges(counts, ut_edges, n_q):
    """Equal-population uT quantile boundaries for one cell, from the CDF of the
    binned recoUT counts -- the same construction as make_quantile_helper
    (cumsum along the target axis, normalised). Returns the n_q-1 interior
    boundary uT values, or None if the cell has no positive yield."""
    total = counts.sum()
    if not (total > 0):
        return None
    cdf = np.cumsum(counts) / total
    edges = []
    for k in range(1, n_q):
        target = k / n_q
        j = int(np.searchsorted(cdf, target))
        j = min(max(j, 0), len(ut_edges) - 2)
        # linear interpolation within the crossing bin for a smooth edge
        c0 = cdf[j - 1] if j > 0 else 0.0
        c1 = cdf[j]
        frac = 0.0 if c1 == c0 else (target - c0) / (c1 - c0)
        edges.append(ut_edges[j] + frac * (ut_edges[j + 1] - ut_edges[j]))
    return edges


def draw_uT(out_dir, name, ut_edges, counts, annotations, q_edges=None):
    """One recoUT distribution as a step histogram; optional quantile lines."""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.step(
        ut_edges,
        np.concatenate([counts[:1], counts]),
        where="pre",
        color="black",
    )
    if q_edges:
        for xe in q_edges:
            ax.axvline(xe, linestyle="--", color="tab:red", linewidth=0.8, zorder=0)
    ax.set_xlabel(r"$u_T$ [GeV]")
    ax.set_ylabel("Events / bin")
    ax.set_xlim(ut_edges[0], ut_edges[-1])
    ax.set_ylim(bottom=0)
    for i, txt in enumerate(annotations):
        ax.text(
            0.04,
            0.92 - 0.07 * i,
            txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=17,
        )
    plot_tools.add_decor(ax, "CMS", "Simulation", data=False, lumi=None, loc=0)
    plot_tools.save_pdf_and_png(out_dir, name)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = output_tools.make_plot_dir(*args.plotdir.rsplit("/", 1), eoscp=args.eoscp)

    dg = Datagroups(args.inputFile)
    for cat in args.categories:
        hname = f"utQuantileInput_{cat}"
        dg.loadHistsForDatagroups(hname, procsToRead=[args.process], syst="")
        h = dg.groups[args.process].hists[hname]
        axn = [a.name for a in h.axes]
        assert axn[:3] == ["recoUT", "eta", "pt"], axn
        charge_dep = "charge" in axn

        ut_ax, eta_ax, pt_ax = h.axes["recoUT"], h.axes["eta"], h.axes["pt"]
        ut_edges = np.asarray(ut_ax.edges)
        # view: (recoUT, eta, pt[, charge]); charge bins map to -1, +1
        v = h.values()
        charges = (
            [(j, int(round(c))) for j, c in enumerate(h.axes["charge"].centers)]
            if charge_dep
            else [(None, None)]
        )

        cat_dir = os.path.join(outdir, cat)

        def cell_counts(eta_sl, pt_sl, j_q):
            """recoUT counts for an (eta, pt) selection and charge index.
            eta_sl / pt_sl are either an int bin or None (= integrate)."""
            sel = v
            # axis order is (recoUT, eta, pt[, charge])
            if charge_dep and j_q is not None:
                sel = sel[..., j_q]
            # now (recoUT, eta, pt); reduce eta then pt
            if eta_sl is None:
                sel = sel.sum(axis=1)
            else:
                sel = sel[:, eta_sl, :]
            if pt_sl is None:
                sel = sel.sum(axis=1)
            else:
                sel = sel[:, pt_sl]
            return sel  # 1D over recoUT

        for j_q, qval in charges:
            qtag, qlab = (
                CHARGE_META.get(qval, ("incl", "")) if charge_dep else ("incl", "")
            )
            qsuffix = f"_q{qtag}" if charge_dep else ""
            qann = [qlab] if charge_dep else []

            # ---- Level 1: projection over eta, pt ----
            draw_uT(
                cat_dir,
                f"recoUT_proj{qsuffix}",
                ut_edges,
                cell_counts(None, None, j_q),
                [f"{cat}", *qann, "all $\\eta$, $p_T$"],
            )

            # ---- Level 2 & 3 ----
            for i_eta in range(eta_ax.size):
                eta_lo, eta_hi = eta_ax.edges[i_eta], eta_ax.edges[i_eta + 1]
                eta_ann = f"${eta_lo:+.2f} \\leq \\eta < {eta_hi:+.2f}$"
                eta_dir = os.path.join(cat_dir, f"eta{i_eta:02d}")
                # Level 2: integrate pt
                draw_uT(
                    eta_dir,
                    f"recoUT_eta{i_eta:02d}{qsuffix}",
                    ut_edges,
                    cell_counts(i_eta, None, j_q),
                    [f"{cat}", *qann, eta_ann, "all $p_T$"],
                )
                # Level 3: per (eta, pt) cell, with quantile boundaries
                pt_dir = os.path.join(eta_dir, "pt_cells")
                for j_pt in range(pt_ax.size):
                    pt_lo, pt_hi = pt_ax.edges[j_pt], pt_ax.edges[j_pt + 1]
                    pt_ann = f"${pt_lo:.0f} \\leq p_T < {pt_hi:.0f}$ GeV"
                    counts = cell_counts(i_eta, j_pt, j_q)
                    draw_uT(
                        pt_dir,
                        f"recoUT_eta{i_eta:02d}_pt{j_pt:02d}{qsuffix}",
                        ut_edges,
                        counts,
                        [f"{cat}", *qann, eta_ann, pt_ann],
                        q_edges=quantile_edges(counts, ut_edges, args.nQuantiles),
                    )
                output_tools.write_index_and_log(pt_dir, "index")
                output_tools.write_index_and_log(eta_dir, "index")
            output_tools.write_index_and_log(cat_dir, "index")
        logger.info(f"Wrote {cat} uT-quantile distribution plots under {cat_dir}")

    if output_tools.is_eosuser_path(args.plotdir) and args.eoscp:
        output_tools.copy_to_eos(outdir, args.plotdir)


if __name__ == "__main__":
    main()
