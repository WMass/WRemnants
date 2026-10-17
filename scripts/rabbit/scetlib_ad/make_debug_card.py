#!/usr/bin/env python3
"""Build a gen-level rabbit card straight from a SCETlib autodiff cache.

The card's single channel IS the cache's (qT, |Y|) grid, its signal template is
sigma_gen at the cache anchor, and its data is sigma_gen at whatever point you
ask for. That makes it a self-contained closure test of
``SCETlibADParamModel``: everything the card knows comes from the same backend
the model fits with, so a failure to recover the injected point is a bug in the
model, not a mismatch between two predictions.

It deliberately does NOT replace the real gen-level sigmaUL card
(``feedRabbitSigmaUL.py``); it exists so the plumbing can be exercised before a
production cache is built.

    source <scetlib-cms>/setup.sh
    python scripts/rabbit/scetlib_ad/make_debug_card.py \
        --conf  <...>/matched.conf --cache <...>/cache_debug.npz \
        --inject np_gnu_lambda2=0.12,alphas=0.1195 \
        -o /tmp/scetlib_ad_debug
"""

import argparse
import os
import sys

import hist
import numpy as np

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

from rabbit import tensorwriter  # noqa: E402
from wremnants.postprocessing.scetlib_ad import params as adp  # noqa: E402
from wremnants.postprocessing.scetlib_ad.response import (  # noqa: E402
    CORR_CONFIG_META_KEY,
    NP_ANCHOR_META_KEY,
)
from wremnants.postprocessing.scetlib_ad.xsec_backend import (  # noqa: E402
    ScetlibADXsec,
)

QT_AXIS = "ptVGen"
ABSY_AXIS = "absYVGen"
PROC = "Zmumu"
CHANNEL = "chSigmaUL"


def gen_axes_from_bins(bins):
    """Recover the (qT, |Y|) product grid and the Q window from the cache bins."""
    Q = np.unique(bins[:, 0:2], axis=0)
    if Q.shape[0] != 1:
        raise SystemExit(f"expected a single Q bin in the cache, got {Q}")
    y = np.unique(bins[:, 2:4], axis=0)
    t = np.unique(bins[:, 4:6], axis=0)
    y = y[np.argsort(y[:, 0])]
    t = t[np.argsort(t[:, 0])]
    if y.shape[0] * t.shape[0] != bins.shape[0]:
        raise SystemExit(
            "the cache bins are not a full (qT x |Y|) product grid; this debug "
            "card maker only handles product grids"
        )
    return (
        [
            (QT_AXIS, np.concatenate([t[:, 0], t[-1:, 1]])),
            (ABSY_AXIS, np.concatenate([y[:, 0], y[-1:, 1]])),
        ],
        float(Q[0, 0]),
        float(Q[0, 1]),
    )


def parse_kv(spec):
    if not spec:
        return {}
    out = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        k, v = item.split("=", 1)
        out[k.strip()] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--conf", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--outname", default="debug_card")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument(
        "--inject",
        default=None,
        help="SCETlib parameter values the DATA is generated at, "
        "'np_gnu_lambda2=0.12,alphas=0.1195'. Default: the anchor (Asimov).",
    )
    ap.add_argument(
        "--yield-total",
        type=float,
        default=1.0e6,
        help="scale sigma_gen to this total number of events, which is what sets "
        "the Poisson stat power (and hence the recovered uncertainties)",
    )
    args = ap.parse_args()

    core = ScetlibADXsec(args.conf, args.cache, threads=args.threads)
    gen_axes, q_lo, q_hi = gen_axes_from_bins(core.bins)
    fold = core.fold_for(gen_axes, q_lo, q_hi)
    shape = fold.gen_shape

    def sigma(p):
        vals, _ = core.values_and_jacobian(p)
        return fold(np.asarray(vals, dtype=np.float64)).reshape(shape)

    nominal = sigma(core.anchor)
    if np.any(nominal <= 0):
        bad = int((nominal <= 0).sum())
        raise SystemExit(
            f"{bad} bin(s) of sigma_gen at the anchor are non-positive; a Poisson "
            f"card cannot be built from them. Restrict the cache's qT range."
        )
    scale = args.yield_total / nominal.sum()

    p_data = core.anchor.copy()
    injected = parse_kv(args.inject)
    for name, val in injected.items():
        if name not in core.param_names:
            raise SystemExit(
                f"--inject: {name!r} is not a cache parameter {core.param_names}"
            )
        p_data[core.param_names.index(name)] = val
    data = sigma(p_data)

    axes = [
        hist.axis.Variable(edges, name=name, overflow=False, underflow=False)
        for name, edges in gen_axes
    ]

    def as_hist(values, weighted):
        h = hist.Hist(
            *axes,
            storage=hist.storage.Weight() if weighted else hist.storage.Double(),
        )
        if weighted:
            h.values()[...] = values * scale
            # Poisson-equivalent variance for an Asimov-style template.
            h.variances()[...] = values * scale
        else:
            h.values()[...] = values * scale
        return h

    writer = tensorwriter.TensorWriter()
    writer.add_channel(axes, CHANNEL)
    writer.add_data(as_hist(data, weighted=False), CHANNEL)
    writer.add_process(as_hist(nominal, weighted=True), PROC, CHANNEL, signal=True)

    # The anchor metadata a real card carries, so the debug card exercises the
    # model's anchor path rather than skipping it.
    #
    # On a real card these entries describe the theory CORRECTION its templates
    # were reweighted with, which is a different SCETlib artefact than the cache.
    # Here there is no correction at all -- the template IS sigma_gen at the
    # cache anchor -- so the cache's own runcard is, self-consistently, the
    # correction: same calculation, same central values, by construction. That
    # is exactly what makes this a closure test of the model and not of two
    # predictions agreeing.
    eff, gnu = {}, {}
    for sname, value in zip(core.param_names, core.anchor):
        try:
            rname = adp.rabbit_name(sname)
        except KeyError:
            continue
        if sname.startswith("np_eff_"):
            eff[rname] = float(value)
        elif sname.startswith("np_gnu_"):
            gnu[rname] = float(value)
    conf_np = _np_models_from_conf(core.conf)
    eff["np_model"] = conf_np[0]
    gnu["np_model_nu"] = conf_np[1]
    meta = {
        NP_ANCHOR_META_KEY: {"Z": {"eff_params": eff, "gnu_params": gnu}},
        CORR_CONFIG_META_KEY: {
            "Z": {
                "tag": "debug_card_from_cache",
                "basename": os.path.basename(args.conf),
                "config": {
                    section: {k.lower(): str(v) for k, v in core.conf[section].items()}
                    for section in core.conf.sections()
                },
                "applied_to_nominal": True,
            }
        },
    }

    os.makedirs(args.outdir, exist_ok=True)
    writer.write(
        outfolder=args.outdir, outfilename=args.outname + ".hdf5", meta_data_dict=meta
    )
    out = os.path.join(args.outdir, args.outname + ".hdf5")
    print(f"\nwrote {out}")
    print(
        f"  channel {CHANNEL}: {shape[0]} x {shape[1]} = {int(np.prod(shape))} bins "
        f"({gen_axes[0][0]}, {gen_axes[1][0]}), Q [{q_lo:g}, {q_hi:g}]"
    )
    print(f"  template = sigma_gen(anchor), scaled to {args.yield_total:g} events")
    print(f"  data     = sigma_gen({'anchor' if not injected else injected})")
    if injected:
        rel = np.max(np.abs(data / nominal - 1.0))
        print(f"  injected shape change: max |data/nominal - 1| = {rel:.3%}")


def _np_models_from_conf(conf):
    """(np_model, np_model_nu) from the runcard, for the lambda_central block."""
    npsec = conf["Nonperturbative"] if conf.has_section("Nonperturbative") else {}
    return (
        npsec.get("np_model", "tanh_2"),
        npsec.get("np_model_nu", "tanh_2"),
    )


if __name__ == "__main__":
    main()
