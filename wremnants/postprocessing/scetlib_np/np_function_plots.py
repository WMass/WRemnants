"""Plot the SCETlib NP form factors: CS γ_ν^NP(b_T) and TMD F_eff(b_T, y).

A pure plotting library: takes physical λ values (the two parameter dicts the
model uses) and draws the two NP functions. Where the λ come from (a new
continuous-λ fit, an old template-based fit, or hand-picked values) is the
caller's job. The companion reader :mod:`fitresult_lambdas` turns a fitresults
HDF5 into the λ sets / toy ensembles this module consumes; ``main()`` glues the
two together, but the plot functions stay reader-agnostic.

The curves call the same form factors the fit integrates
(:func:`btgrid_tf.F_eff_tf` / :func:`btgrid_tf.gamma_nu_NP_tf`), driven by the
``np_model`` / ``np_model_nu`` strings, so a plotted curve is the fit's model.

A "λ set" is the pair of dicts :class:`NPLambdas` carries:

    eff = {lambda_inf, lambda2, lambda4, lambda6, delta_lambda2, np_model}   (F_eff / TMD)
    gnu = {lambda_inf_nu, lambda2_nu, lambda4_nu, np_model_nu}                (γ_ν / CS)

These map 1:1 onto the form-factor keyword arguments. A band is drawn from a
caller-supplied list of λ-set "toys" (this module takes percentiles of the
resulting curves); it never samples and never sees a covariance.

CLI (plot a raw λ set, no fit involved)::

    python -m wremnants.postprocessing.scetlib_np.np_function_plots \\
        --lambdas lambda2=0.4,lambda4=0.4,lambda2_nu=0.15 \\
        --np-model tanh_6 --np-model-nu tanh_2 -o /tmp/np.png

The ``--lambdas`` names must be λ the chosen models actually use (e.g. ``lambda6``
needs ``--np-model tanh_6``); naming an inert λ is a hard error rather than a
silently-ignored value. Unset λ stay at their defaults (NP-unit point).

CLI (from a fitresults: prefit dashed, postfit solid + 68% band)::

    python -m wremnants.postprocessing.scetlib_np.np_function_plots \\
        --fitresult <fitresults.hdf5> -o /tmp/np.png
"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from wremnants.postprocessing.scetlib_np import btgrid_tf
from wremnants.postprocessing.scetlib_np.params import (
    EFF_PARAMS,
    GNU_PARAMS,
    active_params,
    parse_lambda_overrides,
    split_eff_gnu,
)


@dataclass
class NPLambdas:
    """One physical λ point: the two form-factor parameter dicts.

    ``eff`` / ``gnu`` hold the kwargs ``btgrid_tf.F_eff_tf`` / ``gamma_nu_NP_tf``
    expect (numeric λ + ``np_model`` / ``np_model_nu``). Build via
    :mod:`fitresult_lambdas` or by hand.
    """

    eff: dict
    gnu: dict

    @classmethod
    def from_flat(cls, values, np_model, np_model_nu):
        """Build from a flat name->value mapping (the param-model λ names)."""
        eff, gnu = split_eff_gnu(values)
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


# LaTeX labels per λ name, for the parameter inset.
_EFF_LABELS = {
    "lambda2": r"\lambda_2",
    "lambda4": r"\lambda_4",
    "delta_lambda2": r"\delta\lambda_2",
    "lambda6": r"\lambda_6",
    "lambda_inf": r"\lambda_\infty",
}
_GNU_LABELS = {
    "lambda2_nu": r"\lambda_2^\nu",
    "lambda4_nu": r"\lambda_4^\nu",
    "lambda6_nu": r"\lambda_6^\nu",
    "lambda_inf_nu": r"\lambda_\infty^\nu",
}


def _param_inset(ax, lam, sector, corner="upper right"):
    """Small text box listing only the λ that drive the panel for its model."""
    if sector == "gnu":
        model = lam.gnu.get("np_model_nu", "?")
        active = active_params(np_model_nu=model)
        labels, src, order = _GNU_LABELS, lam.gnu, GNU_PARAMS
    else:
        model = lam.eff.get("np_model", "?")
        active = active_params(np_model=model)
        labels, src, order = _EFF_LABELS, lam.eff, EFF_PARAMS
    lines = [
        rf"${labels[k]} = {src.get(k, 0):+.4f}$" for k in order if k in active
    ]
    lines.append(rf"model: {model}")
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
    y_values: Sequence[float] = (0.0, 2.5, 5.0),
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
        Curves to overlay. Each series with ``toys`` draws a percentile band on
        its panels; the caller supplies the toys.
    y_values
        Rapidity values for the TMD panel (F_eff depends on y; γ_ν does not).
    bT_max, n_points
        b_T grid for the curves [GeV^-1].
    inset_from
        Series whose λ fill the per-panel parameter box (default: last series,
        typically the postfit point).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bT = np.linspace(0.0, bT_max, n_points)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.5))

    auto_colors = [c for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]]
    cmap_tmd = plt.cm.viridis

    # NP factors evaluated at the bare b_T grid (this grid's b* prescription is
    # the identity, b_bar == b_T; see param_model / base.conf b0_over_bmax=0); a
    # b*-frozen grid would need b_T -> b_bar mapped first. F_eff runs away at large
    # b_T for λ4 < 0 toys, so the TMD panel scales to the line curves (below), not
    # a runaway band tail.
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

    # Scale to the line curves so a runaway band tail (bare F_eff, λ4 < 0) can't
    # dominate the autoscale.
    top = f_ymax if f_ymax is not None else max(1.1, 1.2 * line_fmax)
    axR.set_ylim(0.0, top)

    # Param boxes diagonally opposite each panel's legend to avoid collisions:
    # CS legend lower-left -> box upper-right; TMD upper-right -> box lower-left.
    inset = inset_from if inset_from is not None else (series[-1] if series else None)
    if inset is not None:
        _param_inset(axL, inset.lam, "gnu", corner="upper right")
        _param_inset(axR, inset.lam, "eff", corner="lower left")

    # Allow --outpath to be a directory (trailing slash or no extension): append
    # a default filename rather than erroring on a bare ".png".
    if outpath.endswith(("/", os.sep)) or os.path.isdir(outpath) or not os.path.splitext(outpath)[1]:
        outpath = os.path.join(outpath, "np_functions.png")
    os.makedirs(os.path.dirname(os.path.abspath(outpath)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"Wrote {outpath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Baseline λ for the raw mode: SCETlib "knobs off" (NP-unit point). --lambdas
# overrides any subset; unset λ stay here. Carries all ALL_PARAMS so the
# form-factor kwargs are complete regardless of the chosen model.
_DEFAULT_LAMBDAS = dict(
    lambda2=0.0,
    lambda4=0.0,
    lambda6=0.0,
    delta_lambda2=0.0,
    lambda_inf=1.0,
    lambda2_nu=0.0,
    lambda4_nu=0.0,
    lambda6_nu=0.0,
    lambda_inf_nu=1.0,
)


def make_parser():
    p = argparse.ArgumentParser(
        description="Plot SCETlib NP form factors γ_ν^NP(b_T) and F_eff(b_T,y).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_argument_group("input (pick one mode)")
    src.add_argument(
        "--fitresult",
        default=None,
        help="fitresults HDF5: plot prefit (dashed) + postfit (solid + band). "
        "Uses fitresult_lambdas to read λ / sample the band.",
    )
    src.add_argument(
        "--result", default=None, help="results group suffix for --fitresult."
    )
    src.add_argument(
        "--n-toys", type=int, default=500, help="band toys (--fitresult)."
    )
    src.add_argument("--seed", type=int, default=0, help="band RNG seed.")

    raw = p.add_argument_group("raw λ mode (no fit)")
    raw.add_argument(
        "--lambdas",
        default=None,
        help="λ overrides 'name=val,...' (e.g. lambda2=0.4,lambda2_nu=0.15); unset "
        "λ stay at the NP-unit defaults. Names must be λ the chosen models use "
        "(lambda6/lambda6_nu need tanh_6); an inert λ is a hard error.",
    )
    raw.add_argument("--np-model", default="tanh_2", help="F_eff model string.")
    raw.add_argument("--np-model-nu", default="tanh_2", help="γ_ν model string.")

    p.add_argument(
        "--y",
        type=float,
        nargs="+",
        default=[0.0, 2.5, 5.0],
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
    parser = make_parser()
    args = parser.parse_args(argv)

    if args.fitresult:
        # Reading lives in the reader module; the plotter stays pure.
        from wremnants.postprocessing.scetlib_np import fitresult_lambdas as frl

        series = frl.plot_series_from_fitresult(
            args.fitresult,
            result=args.result,
            n_toys=args.n_toys,
            seed=args.seed,
        )
    else:
        try:
            overrides = parse_lambda_overrides(args.lambdas)
        except ValueError as e:
            parser.error(str(e))
        active = active_params(args.np_model, args.np_model_nu)
        inert = [k for k in overrides if k not in active]
        if inert:
            parser.error(
                "--lambdas: "
                + ", ".join(inert)
                + f" not used by np_model={args.np_model} / "
                + f"np_model_nu={args.np_model_nu} (active: "
                + ", ".join(k for k in (*EFF_PARAMS, *GNU_PARAMS) if k in active)
                + ")"
            )
        vals = {**_DEFAULT_LAMBDAS, **overrides}
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
