"""Plot the SCETlib NP form factors — CS γ_ν^NP(b_T) and TMD F_eff(b_T, y).

This is a PURE plotting library: it takes physical λ values (the two parameter
dicts the model uses) and draws the two NP functions. It knows nothing about
fitresults — where the λ come from (a new continuous-λ fit, an old
template-based fit, or hand-picked values to test) is the caller's job. The
companion reader :mod:`fitresult_lambdas` turns a fitresults HDF5 into the λ
sets / toy ensembles this module consumes; ``main()`` below glues the two for
convenience but the plot functions stay reader-agnostic.

The curves call the same form factors the fit integrates
(:func:`btgrid_tf.F_eff_tf` / :func:`btgrid_tf.gamma_nu_NP_tf`), driven by the
``np_model`` / ``np_model_nu`` strings, so a plotted curve is exactly the model
the fit used.

A "λ set" is the pair of dicts :class:`NPLambdas` carries:

    eff = {lambda_inf, lambda2, lambda4, lambda6, delta_lambda2, np_model}   (F_eff / TMD)
    gnu = {lambda_inf_nu, lambda2_nu, lambda4_nu, np_model_nu}                (γ_ν / CS)

These map 1:1 onto the form-factor keyword arguments. A band is drawn from
a list of λ-set "toys" handed in by the caller (this module just takes
percentiles of the resulting curves); it never samples and never sees a
covariance.

CLI (plot a raw λ set, no fit involved)::

    python -m wremnants.postprocessing.scetlib_np.np_function_plots \\
        --lambda2 0.4 --lambda4 0.4 --lambda2_nu 0.15 \\
        --np-model tanh_6 --np-model-nu tanh_2 -o /tmp/np.png

CLI (from a fitresults: prefit dashed, postfit solid + 68% band)::

    python -m wremnants.postprocessing.scetlib_np.np_function_plots \\
        --from-fitresult <fitresults.hdf5> -o /tmp/np.png
"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from wremnants.postprocessing.scetlib_np import btgrid_tf

# λ split across the two NP sectors, with sensible "all knobs off" defaults so a
# bare CLI call still draws something. Mirrors param_model.{GNU,EFF}_PARAMS.
GNU_PARAMS = ("lambda2_nu", "lambda4_nu", "lambda_inf_nu")
EFF_PARAMS = ("lambda2", "lambda4", "lambda6", "delta_lambda2", "lambda_inf")


@dataclass
class NPLambdas:
    """One physical λ point: the two form-factor parameter dicts.

    ``eff`` / ``gnu`` hold exactly the kwargs ``btgrid_tf.F_eff_tf`` /
    ``gamma_nu_NP_tf`` expect (numeric λ + the ``np_model`` / ``np_model_nu``
    string). Build one from a fit via :mod:`fitresult_lambdas`, or by hand.
    """

    eff: dict
    gnu: dict

    @classmethod
    def from_flat(cls, values, np_model, np_model_nu):
        """Build from a flat name->value mapping (the param-model λ names)."""
        eff = {k: float(values[k]) for k in EFF_PARAMS if k in values}
        gnu = {k: float(values[k]) for k in GNU_PARAMS if k in values}
        eff["np_model"] = np_model
        gnu["np_model_nu"] = np_model_nu
        return cls(eff=eff, gnu=gnu)


@dataclass
class Series:
    """A curve to draw: a λ set, styling, and optional toys for an error band."""

    label: str
    lam: NPLambdas
    color: Optional[str] = None
    linestyle: str = "-"
    lw: float = 2.0
    toys: Optional[List[NPLambdas]] = None  # band drawn from these if present
    band_pct: Tuple[float, float] = (16.0, 84.0)


def gamma_nu_curve(bT, gnu):
    """γ_ν^NP(b_T) for one gnu dict (CS sector)."""
    return np.asarray(btgrid_tf.gamma_nu_NP_tf(bT, **gnu), dtype=float)


def f_eff_curve(bT, y, eff):
    """F_eff(y, b_T) for one eff dict at rapidity ``y`` (TMD sector)."""
    return np.asarray(btgrid_tf.F_eff_tf(y, bT, **eff), dtype=float)


def _band(curves, pct):
    """(lo, hi) percentile envelope across a stack of curves, or None if empty."""
    if not len(curves):
        return None
    stack = np.asarray(curves)
    return np.percentile(stack, pct[0], axis=0), np.percentile(stack, pct[1], axis=0)


_CORNERS = {
    "upper right": (0.97, 0.97, "top", "right"),
    "upper left": (0.03, 0.97, "top", "left"),
    "lower left": (0.03, 0.03, "bottom", "left"),
    "lower right": (0.97, 0.03, "bottom", "right"),
}


def _param_inset(ax, lam, sector, corner="upper right"):
    """Small text box listing the λ that drive the panel."""
    if sector == "gnu":
        lines = [
            rf"$\lambda_2^\nu = {lam.gnu.get('lambda2_nu', 0):+.4f}$",
            rf"$\lambda_4^\nu = {lam.gnu.get('lambda4_nu', 0):+.4f}$",
            rf"$\lambda_\infty^\nu = {lam.gnu.get('lambda_inf_nu', 0):+.4f}$",
            rf"model: {lam.gnu.get('np_model_nu', '?')}",
        ]
    else:
        lines = [
            rf"$\lambda_2 = {lam.eff.get('lambda2', 0):+.4f}$",
            rf"$\lambda_4 = {lam.eff.get('lambda4', 0):+.4f}$",
            rf"$\delta\lambda_2 = {lam.eff.get('delta_lambda2', 0):+.4f}$",
            rf"$\lambda_6 = {lam.eff.get('lambda6', 0):+.4f}$",
            rf"$\lambda_\infty = {lam.eff.get('lambda_inf', 0):+.4f}$",
            rf"model: {lam.eff.get('np_model', '?')}",
        ]
    box = dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.85)
    x, y, va, ha = _CORNERS[corner]
    ax.text(
        x,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=8,
        va=va,
        ha=ha,
        bbox=box,
    )


def plot_np_functions(
    series: Sequence[Series],
    *,
    y_values: Sequence[float] = (0.0, 2.5),
    bT_max: float = 4.0,
    n_points: int = 401,
    outpath: str,
    inset_from: Optional[Series] = None,
    f_ymax: Optional[float] = None,
):
    """Draw the two NP form factors for one or more λ sets.

    Parameters
    ----------
    series
        Curves to overlay. The first series with ``toys`` set draws a
        percentile band on each panel. Pure: the caller supplies the toys.
    y_values
        Rapidity values for the TMD panel (F_eff depends on y; γ_ν does not).
    bT_max, n_points
        b_T grid for the curves [GeV^-1].
    inset_from
        Series whose λ are written into the per-panel parameter box (defaults
        to the last series — typically the postfit point).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bT = np.linspace(0.0, bT_max, n_points)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.5))

    auto_colors = [c for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]]
    cmap_tmd = plt.cm.viridis

    # NP factors are evaluated at the bare b_T grid (this grid's b* prescription
    # is the identity, b_bar == b_T; see param_model / base.conf b0_over_bmax=0).
    # For a b*-frozen grid the caller would need to map b_T -> b_bar first.
    # F_eff can run away at large b_T for λ4 < 0 toys, so we scale the TMD panel
    # to the line curves rather than let a runaway band tail set the autoscale.
    line_fmax = 0.0

    for si, s in enumerate(series):
        color = s.color or auto_colors[si % len(auto_colors)]

        # ---- CS panel: γ_ν^NP(b_T) (no y dependence) ----
        axL.plot(
            bT,
            gamma_nu_curve(bT, s.lam.gnu),
            label=s.label,
            color=color,
            lw=s.lw,
            ls=s.linestyle,
        )
        if s.toys:
            band = _band([gamma_nu_curve(bT, t.gnu) for t in s.toys], s.band_pct)
            if band is not None:
                axL.fill_between(
                    bT,
                    band[0],
                    band[1],
                    color=color,
                    alpha=0.22,
                    label=f"{s.label} {int(s.band_pct[1]-s.band_pct[0])}% band",
                )

        # ---- TMD panel: F_eff(b_T, y) per requested y ----
        n_y = len(y_values)
        for yi, y in enumerate(y_values):
            shade = 0.25 + 0.6 * (yi / max(n_y - 1, 1))
            yc = color if n_y == 1 else cmap_tmd(shade)
            line = f_eff_curve(bT, y, s.lam.eff)
            line_fmax = max(line_fmax, float(np.nanmax(line)))
            axR.plot(
                bT, line, color=yc, lw=s.lw, ls=s.linestyle, label=f"{s.label}, y={y:g}"
            )
            if s.toys:
                band = _band([f_eff_curve(bT, y, t.eff) for t in s.toys], s.band_pct)
                if band is not None:
                    axR.fill_between(bT, band[0], band[1], color=yc, alpha=0.18)

    axL.axhline(0, color="k", lw=0.5)
    axL.set_xlabel(r"$b_T$ [GeV$^{-1}$]")
    axL.set_ylabel(r"$\tilde\gamma_\nu^{\rm NP}(b_T)$")
    axL.set_title(
        r"CS rapidity anomalous dimension $\tilde\gamma_\nu^{\rm NP}(b_T)$", fontsize=11
    )
    axL.legend(loc="lower left", fontsize=8)
    axL.grid(alpha=0.3)

    axR.set_xlabel(r"$b_T$ [GeV$^{-1}$]")
    axR.set_ylabel(r"$F_{\rm eff}(b_T, y)$")
    axR.set_title(r"TMD-effective NP factor $F_{\rm eff}(b_T, y)$", fontsize=11)
    axR.legend(loc="upper right", fontsize=8)
    axR.grid(alpha=0.3)

    # Scale the TMD panel to the line curves so a runaway band tail (bare F_eff
    # for λ4 < 0) doesn't dominate the autoscale.
    top = f_ymax if f_ymax is not None else max(1.1, 1.2 * line_fmax)
    axR.set_ylim(0.0, top)

    # Param boxes diagonally opposite each panel's legend so they don't collide:
    # CS legend lower-left -> box upper-right; TMD legend upper-right -> box lower-left.
    inset = inset_from if inset_from is not None else (series[-1] if series else None)
    if inset is not None:
        _param_inset(axL, inset.lam, "gnu", corner="upper right")
        _param_inset(axR, inset.lam, "eff", corner="lower left")

    os.makedirs(os.path.dirname(os.path.abspath(outpath)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"Wrote {outpath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Defaults for a bare CLI call: SCETlib "knobs off" plus the conventional model
# strings (override with --np-model / --np-model-nu and the --lambda* flags).
_CLI_DEFAULTS = dict(
    lambda2=0.0,
    lambda4=0.0,
    lambda6=0.0,
    delta_lambda2=0.0,
    lambda_inf=1.0,
    lambda2_nu=0.0,
    lambda4_nu=0.0,
    lambda_inf_nu=1.0,
)


def make_parser():
    p = argparse.ArgumentParser(
        description="Plot SCETlib NP form factors γ_ν^NP(b_T) and F_eff(b_T,y).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_argument_group("input (pick one mode)")
    src.add_argument(
        "--from-fitresult",
        default=None,
        help="fitresults HDF5: plot prefit (dashed) + postfit (solid + band). "
        "Uses fitresult_lambdas to read λ / sample the band.",
    )
    src.add_argument(
        "--result", default=None, help="results group suffix for --from-fitresult."
    )
    src.add_argument(
        "--n-toys", type=int, default=500, help="band toys (--from-fitresult)."
    )
    src.add_argument("--seed", type=int, default=0, help="band RNG seed.")

    raw = p.add_argument_group("raw λ mode (no fit)")
    for k, v in _CLI_DEFAULTS.items():
        raw.add_argument(f"--{k}", type=float, default=None, help=f"default {v}")
    raw.add_argument("--np-model", default="tanh_6", help="F_eff model string.")
    raw.add_argument("--np-model-nu", default="tanh_2", help="γ_ν model string.")

    p.add_argument(
        "--y",
        type=float,
        nargs="+",
        default=[0.0, 2.5],
        help="rapidity values for the TMD panel.",
    )
    p.add_argument("--bT-max", type=float, default=4.0)
    p.add_argument(
        "--f-ymax",
        type=float,
        default=None,
        help="fixed upper y-limit for the TMD panel (default: auto "
        "from the line curves; clips runaway negative-λ band tails).",
    )
    p.add_argument("--label", default="input")
    p.add_argument("--outpath", "-o", required=True)
    return p


def main(argv=None):
    args = make_parser().parse_args(argv)

    if args.from_fitresult:
        # Reading lives in the reader module; the plotter stays pure.
        from wremnants.postprocessing.scetlib_np import fitresult_lambdas as frl

        series = frl.plot_series_from_fitresult(
            args.from_fitresult,
            result=args.result,
            n_toys=args.n_toys,
            seed=args.seed,
        )
    else:
        vals = {
            k: (getattr(args, k) if getattr(args, k) is not None else v)
            for k, v in _CLI_DEFAULTS.items()
        }
        lam = NPLambdas.from_flat(vals, args.np_model, args.np_model_nu)
        series = [Series(label=args.label, lam=lam, color="C3")]

    plot_np_functions(
        series,
        y_values=args.y,
        bT_max=args.bT_max,
        outpath=args.outpath,
        f_ymax=args.f_ymax,
    )


if __name__ == "__main__":
    main()
