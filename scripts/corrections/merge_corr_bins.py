"""Adaptive merging of the bins of a MiNNLO reference where the correction by helicity is ill-defined.

The correction by helicity (CentralCorrByHelicityHelper) weights an event by
sum_h sigma_h^corr P_h(Omega) / sum_h sigma_h^MiNNLO P_h(Omega). In a bin of the MiNNLO reference the denominator is
sigma_UL (1 + cos^2 theta + sum_i A_i P_i(theta, phi)); it is ill-defined where sigma_UL <= 0 or where the bracket is
not positive at all angles, which happens in sparsely populated bins (high |Y|, high pT, high mass).

The reference keeps its rectangular grid; bins are grouped, and the correction is to be computed on the groups
(MiNNLO and the prediction summed over the same groups) and assigned to every bin of a group. A group passes if
sigma_UL > 0, N_eff = sigma_UL^2 / var(sigma_UL) >= nmin and the minimum of the bracket over all angles (computed
exactly) is >= bmin. Bins are merged only where needed, first along the first merge axis (default pT), then along both
merge axes (pT and mass); the other axes (|Y|, charge) are never merged. At each step the worst failing group
(sigma_UL <= 0 first, then the smallest N_eff) is merged with the neighbour that gives a passing group of the smallest
N_eff, or else with the populated neighbour of the smallest N_eff; ties go to the lowest cell number. Empty bins (no
events) are not merged for their own sake; a failing group crosses them only when it has no populated neighbour. Bins
outside the generated mass range of the samples (--massRange) are not merged at all: MiNNLO is empty there because of
the generation cut, and a group extending there would add predicted cross section without MiNNLO events to its ratio.

Output: an npz file with the group label and the status (STATUS_CODES) of every bin, both arrays over the
non-helicity axes, the axis edges, the number of failing groups and the provenance (meta, JSON). The script fails if
groups still fail the criteria, unless --allowFailing. The correction step has to treat the empty and outside bins
explicitly.
"""

import argparse
import hashlib
import itertools
import json
import os

import numpy as np

from wremnants.utilities import common
from wremnants.utilities.io_tools import input_tools
from wums import logging, output_tools

SCHEMA_VERSION = 2
STATUS_CODES = {"pass": 0, "empty": 1, "outside": 2, "fail": 3}
HELICITY_CONVENTION = (
    "helicity -1: sigma_UL, 0..7: sigma_UL * A_0..A_7 (csAngularMoments); "
    "bracket 1 + cos^2 theta + sum_i A_i P_i with the P_i of csAngularFactors"
)


def bracket_min(coeffs):
    """Minimum over all angles of 1 + cos^2 theta + sum_i A_i P_i(theta, phi), with the polynomials of
    csAngularFactors, for angular coefficients coeffs (..., 8). With n = (sin theta cos phi, sin theta sin phi,
    cos theta) the bracket is 1 + A0 / 2 + n^T Q n + l^T n. Its minimum on the unit sphere is the maximum over t >= 0
    of lambda_0 - t - sum_i g_i^2 / (d_i + t), with the eigenvalues lambda_i (ascending) of Q, d_i = lambda_i - lambda_0
    and g = V^T l / 2, reached at the root of sum_i g_i^2 / (d_i + t)^2 = 1 (or t -> 0). Every t gives a lower bound,
    so the residual of the bisection can only lower the result."""
    a = np.moveaxis(np.asarray(coeffs, dtype=float), -1, 0)
    q = np.stack(
        [
            np.stack([a[2] / 2, a[5], a[1]], axis=-1),
            np.stack([a[5], -a[2] / 2, a[6]], axis=-1),
            np.stack([a[1], a[6], 1 - 1.5 * a[0]], axis=-1),
        ],
        axis=-2,
    )
    lam, vec = np.linalg.eigh(q)
    lin = np.stack([a[3], a[7], a[4]], axis=-1)
    g2 = (np.einsum("...ji,...j->...i", vec, lin) / 2) ** 2
    d = lam - lam[..., :1]

    def terms(t, power):
        # sum_i g_i^2 / (d_i + t)^power; vanishing g_i are left out (d_0 + t = 0 at t = 0)
        den = (d + t[..., None]) ** power
        return np.sum(np.divide(g2, den, out=np.zeros_like(g2), where=g2 > 0), axis=-1)

    lo = np.zeros(lam.shape[:-1])
    hi = np.sqrt(g2.sum(axis=-1))  # the root is below: there the sum is <= 1
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        above = terms(mid, 2) > 1
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    return 1 + a[0] / 2 + lam[..., 0] - hi - terms(hi, 1)


def group_stats(s, var):
    """(sigma_UL, N_eff, minimum of the angular bracket) of summed helicity cross sections s (..., 9) and UL variances
    var (...); the minimum is -inf where sigma_UL <= 0"""
    s, var = np.asarray(s, dtype=float), np.asarray(var, dtype=float)
    ul = s[..., 0]
    pos = ul > 0
    neff = np.where(pos & (var > 0), ul**2 / np.where(var > 0, var, 1.0), 0.0)
    coeffs = s[..., 1:] / np.where(pos, ul, 1.0)[..., None]
    return ul, neff, np.where(pos, bracket_min(coeffs), -np.inf)


def passes(x, nmin, bmin):
    return (x[0] > 0) & (x[1] >= nmin) & (x[2] >= bmin)


def is_empty(s, var):
    # no events: not failing, crossed only to reach populated bins
    return (s[..., 0] == 0) & (var == 0)


def merge_plane(vals, ulvar, active, nmin, bmin):
    """Group the cells of a 2D plane (first merge axis, second merge axis); returns the group label of every cell.
    Inactive cells are left as they are."""
    n0, n1 = vals.shape[:2]
    ncell = n0 * n1
    parent = np.arange(ncell)
    # cells are numbered with the first merge axis running fastest, c = i0 + n0 * i1
    ssum = vals.reshape(ncell, -1, order="F").astype(float)
    svar = ulvar.reshape(-1, order="F").astype(float)
    active = active.reshape(-1, order="F")
    members = {c: [c] for c in range(ncell)}
    st = dict(enumerate(zip(*group_stats(ssum, svar))))

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    def neighbours(root, both_axes):
        out = set()
        steps = (
            ((1, 0), (-1, 0)) if not both_axes else ((1, 0), (-1, 0), (0, 1), (0, -1))
        )
        for c in members[root]:
            i1, i0 = divmod(c, n0)
            for d0, d1 in steps:
                j0, j1 = i0 + d0, i1 + d1
                if 0 <= j0 < n0 and 0 <= j1 < n1 and active[j1 * n0 + j0]:
                    rr = find(j1 * n0 + j0)
                    if rr != root:
                        out.add(rr)
        return sorted(out)

    for both_axes in [False, True]:
        stuck = set()
        while True:
            failing = [
                r
                for r in st
                if active[r]
                and not passes(st[r], nmin, bmin)
                and r not in stuck
                and not is_empty(ssum[r], svar[r])
            ]
            if not failing:
                break
            r = min(failing, key=lambda x: (st[x][0] > 0, st[x][1], x))
            nb = neighbours(r, both_axes)
            if not nb:
                # e.g. a whole line along the first axis in the first phase; retried in the second phase
                stuck.add(r)
                continue
            trial = group_stats(ssum[r] + ssum[nb], svar[r] + svar[nb])
            ok = passes(trial, nmin, bmin)
            populated = [m for m in nb if not is_empty(ssum[m], svar[m])]
            if ok.any():
                n = min((trial[1][k], m) for k, m in enumerate(nb) if ok[k])[1]
            elif populated:
                n = min(populated, key=lambda m: (st[m][1], m))
            else:
                n = nb[0]  # only empty neighbours: cross them
            a, b = (r, n) if len(members[r]) >= len(members[n]) else (n, r)
            parent[b] = a
            ssum[a] += ssum[b]
            svar[a] += svar[b]
            members[a] += members.pop(b)
            st.pop(b)
            st[a] = tuple(x[()] for x in group_stats(ssum[a], svar[a]))
            stuck.discard(a)
    return np.array([find(c) for c in range(ncell)]).reshape(n1, n0).T


def group_status(vals, ulvar, labels, active, nmin, bmin):
    """Status (STATUS_CODES) of every bin, recomputed from the bins of its group, and the number of failing groups"""
    flat = labels.ravel()
    ng = flat.max() + 1
    ssum = np.zeros((ng, vals.shape[-1]))
    np.add.at(ssum, flat, vals.reshape(-1, vals.shape[-1]))
    svar = np.bincount(flat, weights=ulvar.ravel(), minlength=ng)
    gactive = np.bincount(flat, weights=active.ravel().astype(float), minlength=ng) > 0
    gstatus = np.select(
        [~gactive, is_empty(ssum, svar), passes(group_stats(ssum, svar), nmin, bmin)],
        [STATUS_CODES["outside"], STATUS_CODES["empty"], STATUS_CODES["pass"]],
        STATUS_CODES["fail"],
    )
    return gstatus[labels], int((gstatus == STATUS_CODES["fail"]).sum())


def merge_groups(h, merge_axes, nmin, bmin, ranges=None, helicity_axis="helicity"):
    """Group labels over the non-helicity axes of h, merging only along the two merge_axes and only bins within
    ranges ({axis name: (low, high)}); returns the labels, the axis names, the status of every bin and the number of
    groups that still fail the criteria."""
    names = [n for n in h.axes.name if n != helicity_axis]
    hel = h.axes.name.index(helicity_axis)
    vals0 = np.moveaxis(h.values(), hel, -1)
    ulvar0 = np.moveaxis(h.variances(), hel, -1)[..., 0]
    active0 = np.ones(ulvar0.shape, dtype=bool)
    for name, (low, high) in (ranges or {}).items():
        edges = h.axes[name].edges
        shape = [1] * len(names)
        shape[names.index(name)] = -1
        active0 &= ((edges[:-1] >= low) & (edges[1:] <= high)).reshape(shape)
    i0, i1 = names.index(merge_axes[0]), names.index(merge_axes[1])
    others = [i for i in range(len(names)) if i not in (i0, i1)]
    order = [i0, i1, *others]
    vals = np.transpose(vals0, (*order, len(names)))
    ulvar = np.transpose(ulvar0, order)
    active = np.transpose(active0, order)
    labels = np.zeros(ulvar.shape, dtype=np.int64)
    offset = 0
    for idx in itertools.product(
        *[range(ulvar.shape[k]) for k in range(2, ulvar.ndim)]
    ):
        sl = (slice(None), slice(None), *idx)
        lab = merge_plane(vals[sl], ulvar[sl], active[sl], nmin, bmin)
        _, inv = np.unique(lab, return_inverse=True)
        labels[sl] = inv.reshape(lab.shape) + offset
        offset += inv.max() + 1
    labels = np.transpose(labels, np.argsort(order))
    status, nfail = group_status(vals0, ulvar0, labels, active0, nmin, bmin)
    return labels, names, status, nfail


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--minnloFile",
        required=True,
        help="MiNNLO gen file with the helicity cross sections",
    )
    parser.add_argument(
        "--minnloh", default="nominal_gen", help="histogram with a helicity axis"
    )
    parser.add_argument(
        "--minnloProcs",
        nargs="+",
        default=["Zmumu"],
        help="MiNNLO samples, read as <name>_<era>",
    )
    parser.add_argument("--eras", nargs="+", default=["13TeVGen"])
    parser.add_argument(
        "--noCombineTau", action="store_true", help="Do not add the tau samples"
    )
    parser.add_argument(
        "--mergeAxes",
        nargs=2,
        default=["ptVgen", "massVgen"],
        help="Axes along which bins are merged, the first one first",
    )
    parser.add_argument(
        "--nmin",
        type=float,
        default=100.0,
        help="Minimum effective number of events per group",
    )
    parser.add_argument(
        "--bmin",
        type=float,
        default=0.1,
        help="Minimum of the angular bracket per group",
    )
    parser.add_argument(
        "--massRange",
        nargs=2,
        type=float,
        default=None,
        help="Generated mass range of the samples (massVgen); bins outside are not merged, "
        "e.g. 50 13000 for a reference of the m > 50 GeV samples only",
    )
    parser.add_argument(
        "--allowFailing",
        action="store_true",
        help="Write the output even if groups still fail the criteria (status 'fail')",
    )
    parser.add_argument(
        "-o", "--outfile", required=True, help="Output npz file with the group labels"
    )
    parser.add_argument("-v", "--verbose", type=int, default=3, choices=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    logger = logging.setup_logger(__file__, args.verbose)

    combine_with_tau = not args.noCombineTau
    input_tools.check_minnlo_procs(args.minnloProcs, combine_with_tau)
    h = None
    for proc in args.minnloProcs:
        hproc = input_tools.read_mu_hist_combine_tau(
            args.minnloFile,
            proc,
            args.minnloh,
            eras=args.eras,
            combine_with_tau=combine_with_tau,
        )
        h = hproc if h is None else h + hproc

    ranges = {"massVgen": args.massRange} if args.massRange else None
    labels, names, status, nfail = merge_groups(
        h, args.mergeAxes, args.nmin, args.bmin, ranges
    )
    sizes = np.bincount(labels.ravel())
    merged = sizes[labels] > 1
    counts = ", ".join(
        f"{int((status == code).sum())} {name}" for name, code in STATUS_CODES.items()
    )
    logger.info(
        f"{labels.size} bins -> {len(sizes)} groups (nmin {args.nmin:g}, bmin {args.bmin:g}), {int(merged.sum())} "
        f"bins ({100 * merged.mean():.1f}%) in merged groups, largest group {sizes.max()} bins; bins by status: "
        f"{counts}; {nfail} groups failing after merging"
    )
    if nfail and not args.allowFailing:
        raise RuntimeError(
            f"{nfail} groups still fail the criteria; --allowFailing writes them with status 'fail'"
        )

    with open(args.minnloFile, "rb") as f:
        sha256 = hashlib.file_digest(f, "sha256").hexdigest()
    meta = {
        "schema_version": SCHEMA_VERSION,
        "minnlo_file": os.path.abspath(args.minnloFile),
        "minnlo_file_sha256": sha256,
        "minnlo_hist": args.minnloh,
        "minnlo_procs": args.minnloProcs,
        "eras": args.eras,
        "combine_with_tau": combine_with_tau,
        "merge_axes": args.mergeAxes,
        "mass_range": args.massRange,
        "nmin": args.nmin,
        "bmin": args.bmin,
        "nfail": nfail,
        "status_codes": STATUS_CODES,
        "helicity_convention": HELICITY_CONVENTION,
        "meta_info": output_tools.make_meta_info_dict(args=args, wd=common.base_dir),
    }
    np.savez(
        args.outfile,
        labels=labels,
        status=status,
        axes=np.array(names),
        **{f"edges_{n}": h.axes[n].edges for n in names},
        nmin=args.nmin,
        bmin=args.bmin,
        nfail=nfail,
        **({"massRange": args.massRange} if args.massRange else {}),
        meta=json.dumps(meta, default=str),
    )
    logger.info(f"Wrote {args.outfile}")


if __name__ == "__main__":
    main()
