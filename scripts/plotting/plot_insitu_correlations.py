"""Overview plots of the postfit correlations between the in-situ muon
efficiency Chebyshev coefficients.

rabbit_plot_cov.py --params draws one labelled cell per parameter, which is
unreadable beyond O(50) parameters; the full in-situ matrix has 2112. This
script covers the complementary, aggregate view:

  * the full 2112x2112 correlation matrix as a heatmap, ordered
    step -> eta -> charge -> coefficient so the block-diagonal structure is
    the message, plus a zoom on the first few blocks (which are only 8-24
    wide, hence invisible in the full view);
  * the strongest off-diagonal correlation within each (step, eta) block as a
    function of eta, which is where the near-degeneracies live.

Use rabbit_plot_cov.py for the per-eta blocks where individual coefficients
still need to be identifiable.
"""

import argparse
import datetime
import os
import re

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.ticker import NullLocator

import rabbit.io_tools
from wums import logging, output_tools, plot_tools

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

STEPS = ["ID", "HLT", "Iso"]
STEP_LABELS = {
    "ID": r"$\varepsilon^{\mathrm{ID}}$",
    "HLT": r"$\varepsilon^{\mathrm{trig}}$",
    "Iso": r"$\varepsilon^{\mathrm{iso}}$",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "infile", type=str, help="rabbit fitresults hdf5 with a Hessian"
    )
    parser.add_argument("--outpath", type=str, default=None)
    parser.add_argument("--postfix", type=str, default="")
    parser.add_argument(
        "--result", type=str, default=None, help="fitresults key (e.g. 'asimov')"
    )
    parser.add_argument("--title", type=str, default="CMS")
    parser.add_argument("--subtitle", type=str, default="Preliminary")
    parser.add_argument("--titlePos", type=int, default=0)
    parser.add_argument("--nEta", type=int, default=48)
    parser.add_argument("--etaRange", type=float, nargs=2, default=[-2.4, 2.4])
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def default_outpath(script_path):
    stem = os.path.splitext(os.path.basename(script_path))[0]
    today = datetime.date.today().strftime("%y%m%d")
    return os.path.expanduser(f"~/public_html/AlphaS/{today}_{stem}/")


def load_correlations(infile, result):
    """Return (corr, names) restricted to the in-situ coefficients."""
    fitresult, meta = rabbit.io_tools.get_fitresult(infile, result, meta=True)
    cov = fitresult["cov"].get().values()
    names = np.array([n for n in fitresult["parms"].get().axes["parms"]])
    sel = np.array([i for i, n in enumerate(names) if n.startswith("effInsitu")])
    if not len(sel):
        raise RuntimeError("No effInsitu* parameters found in the fit result")
    cov = cov[np.ix_(sel, sel)]
    sigma = np.sqrt(np.diag(cov))
    return cov / np.outer(sigma, sigma), names[sel], meta


def decode(names):
    """(step, eta, charge) per parameter name."""
    step = np.array([n.split("_")[0].replace("effInsitu", "") for n in names])
    eta = np.array([int(re.search(r"_eta(\d+)", n).group(1)) for n in names])
    m = [re.search(r"_q(minus|plus)", n) for n in names]
    charge = np.array([x.group(1) if x else "none" for x in m])
    return step, eta, charge


def ordered_index(step, eta, charge, names):
    """step -> eta -> charge -> coefficient, so blocks are contiguous."""
    key = [(STEPS.index(s), e, c, n) for s, e, c, n in zip(step, eta, charge, names)]
    return np.array(sorted(range(len(key)), key=lambda i: key[i]))


def plot_matrix(outdir, corr, step, eta, order, args, stem, nShow=None):
    """Heatmap of the ordered correlation matrix. With nShow set, only the
    first nShow ordered parameters are drawn: the per-(step, eta) blocks are
    just 8-24 wide out of 2112, so they are not resolvable in the full view.
    The full view is annotated by step, the zoom by eta bin.
    """
    idx = order[:nShow] if nShow else order
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(
        corr[np.ix_(idx, idx)],
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
        origin="upper",
    )

    # group the drawn parameters by step (full view) or eta bin (zoom, which
    # sits inside a single step), then rule and label the group boundaries
    groups = step[idx] if nShow is None else eta[idx]
    labels = []
    for g in dict.fromkeys(groups):
        pos = np.where(groups == g)[0]
        ax.axhline(pos[-1] + 0.5, color="k", lw=0.8)
        ax.axvline(pos[-1] + 0.5, color="k", lw=0.8)
        labels.append(
            (pos.mean(), STEP_LABELS[g] if nShow is None else rf"$\eta_{{{g}}}$")
        )

    ax.set_xticks([p for p, _ in labels])
    ax.set_yticks([p for p, _ in labels])
    ax.set_xticklabels([l for _, l in labels])
    ax.set_yticklabels([l for _, l in labels])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("in-situ efficiency coefficient")
    ax.set_ylabel("in-situ efficiency coefficient")
    fig.colorbar(im, ax=ax, label="correlation")

    plot_tools.add_decor(
        ax, args.title, args.subtitle, data=True, lumi=None, loc=args.titlePos
    )
    name = "_".join(filter(None, [stem, args.postfix]))
    plot_tools.save_pdf_and_png(outdir, name)
    output_tools.write_logfile(outdir, name, args=args, wd=os.path.dirname(__file__))


def plot_max_corr_vs_eta(outdir, corr, step, eta, args, stem):
    edges = np.linspace(args.etaRange[0], args.etaRange[1], args.nEta + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])

    fig, ax = plt.subplots(figsize=(8, 6))
    for s in STEPS:
        y = []
        for b in range(args.nEta):
            idx = np.where((step == s) & (eta == b))[0]
            if len(idx) < 2:
                y.append(np.nan)
                continue
            block = corr[np.ix_(idx, idx)]
            off = block[~np.eye(len(idx), dtype=bool)]
            y.append(np.abs(off).max())
        ax.plot(centers, y, marker="o", ms=3, label=STEP_LABELS[s])

    ax.set_xlabel(r"probe $\eta$")
    ax.set_ylabel(r"max $|\rho|$ within $(\mathrm{step},\eta)$ block")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower center", ncol=3)
    plot_tools.add_decor(
        ax, args.title, args.subtitle, data=True, lumi=None, loc=args.titlePos
    )
    name = "_".join(filter(None, [stem, args.postfix]))
    plot_tools.save_pdf_and_png(outdir, name)
    output_tools.write_logfile(outdir, name, args=args, wd=os.path.dirname(__file__))


def main():
    args = parse_args()
    logging.setup_logger(__file__, 4 if args.debug else 3)

    corr, names, _ = load_correlations(args.infile, args.result)
    step, eta, charge = decode(names)
    logger.info(
        f"{len(names)} in-situ coefficients: "
        + ", ".join(f"{s}={np.sum(step == s)}" for s in STEPS)
    )

    outdir = output_tools.make_plot_dir(args.outpath or default_outpath(__file__))
    logger.info(f"Writing plots to {outdir}")

    order = ordered_index(step, eta, charge, names)
    plot_matrix(outdir, corr, step, eta, order, args, "insitu_corr_full")
    plot_matrix(outdir, corr, step, eta, order, args, "insitu_corr_zoom", nShow=64)
    plot_max_corr_vs_eta(outdir, corr, step, eta, args, "insitu_corr_max_vs_eta")


if __name__ == "__main__":
    main()
