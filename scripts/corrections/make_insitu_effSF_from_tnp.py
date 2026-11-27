import argparse
import pickle

import lz4.frame
import numpy as np
import ROOT

from wremnants.production.muon_efficiencies_insitu import (
    insitu_delta,
    insitu_eff_steps,
    insitu_n_coeff_pt,
    insitu_n_coeff_ut,
    insitu_n_eta,
    insitu_parameter_labels,
    insitu_pt_range,
    insitu_step_group,
    insitu_ut_range,
)
from wremnants.utilities import common
from wums import logging, output_tools

# Express the external tag-and-probe scale factors in the in-situ Chebyshev
# basis, producing the same label-keyed theta_central pkl that
# make_insitu_effSF.py writes from a fit result. The histmakers consume it
# through --insituSFFile unchanged, so this is an alternative *linearisation
# point*: instead of starting from the in-situ fit, start from the SFs the
# default mW analysis applies.
#
# The projection is exact, not approximate. SF = 1 + P(theta) is linear in
# theta, and the smoothed 3D SFs were themselves produced by smoothing with a
# 3rd-order polynomial in pt and 2nd-order in uT -- the same function space our
# basis spans -- so the least-squares solve reproduces them to ~1e-16. Dropping
# to a smaller basis breaks that immediately (3x2 leaves residuals of 3e-2),
# which is the control that shows the agreement is real and not a degenerate
# fit.
#
# Sources, chosen to match what muon_efficiencies_smooth actually applies:
#   trigger, iso : smoothSF3D_*.pkl.lz4, 3D in (eta, pt, uT). Their uT axis
#                  spans exactly insitu_ut_range and their 48 eta bins are ours.
#   idip         : allSmooth_*.root, SF_nomiAndAlt_GtoH_idip_<charge>, 2D in
#                  (eta, pt) -- idip has no uT dependence and is absent from the
#                  3D file, which only smoothed the steps that need it.

parser = argparse.ArgumentParser()
parser.add_argument(
    "--smoothSF3DFile",
    required=True,
    help="smoothSF3D_*.pkl.lz4 with the 3D (eta, pt, uT) scale factors",
)
parser.add_argument(
    "--smoothSF2DFile",
    required=True,
    help="allSmooth_*.root with the 2D SFs, used for idip",
)
parser.add_argument(
    "--isoKey",
    default="smoothSF3D_iso",
    help="iso key in the 3D file: smoothSF3D_iso (triggering muon) or "
    "smoothSF3D_isonotrig (no trigger requirement). Our in-situ iso leg is "
    "measured on Z dilepton, so which one is appropriate is a physics choice.",
)
parser.add_argument("--outpath", type=str, default="./")
parser.add_argument("-p", "--postfix", type=str, default=None)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

logger = logging.setup_logger("make_insitu_effSF_from_tnp", 4 if args.debug else 3)

PT_LO, PT_HI = insitu_pt_range
UT_LO, UT_HI = insitu_ut_range


def cheb(x, n):
    """Chebyshev T_0..T_{n-1} on x, matching eval_leg in the C++ helper."""
    T = [np.ones_like(x), x]
    while len(T) < n:
        T.append(2.0 * x * T[-1] - T[-2])
    return np.stack(T[:n], axis=-1)


def project(design, target):
    """Least-squares coefficients and the residual, so the caller can check."""
    theta, *_ = np.linalg.lstsq(design, target, rcond=None)
    return theta, float(np.abs(design @ theta - target).max())


with lz4.frame.open(args.smoothSF3DFile, "rb") as f:
    sf3d = pickle.load(f)

theta_central = {}
resid = {step: [] for step in insitu_eff_steps}

# ---- trigger and iso: 3D, full pt x uT basis -------------------------------
for step, keys in (
    ("trigger", {"plus": "smoothSF3D_triggerplus", "minus": "smoothSF3D_triggerminus"}),
    ("iso", {None: args.isoKey}),
):
    grp = insitu_step_group[step]
    for qtag, key in keys.items():
        if key not in sf3d:
            raise KeyError(f"{key} not in {args.smoothSF3DFile}: {sorted(sf3d)}")
        h = sf3d[key]
        pt_c, ut_c = h.axes["pt"].centers, h.axes["ut"].centers
        mp = (pt_c >= PT_LO) & (pt_c <= PT_HI)
        mu = (ut_c >= UT_LO) & (ut_c <= UT_HI)
        # index 0 of the eigenvariation axis is the nominal
        vals = h.values()[:, mp][:, :, mu][..., 0]
        xp = 2.0 * (pt_c[mp] - PT_LO) / (PT_HI - PT_LO) - 1.0
        xu = 2.0 * (ut_c[mu] - UT_LO) / (UT_HI - UT_LO) - 1.0
        Bp, Bu = cheb(xp, insitu_n_coeff_pt), cheb(xu, insitu_n_coeff_ut)
        # outer product in the same (cPt, cUt) order as insitu_parameter_labels
        design = (Bp[:, None, :, None] * Bu[None, :, None, :]).reshape(
            len(xp) * len(xu), insitu_n_coeff_pt * insitu_n_coeff_ut
        )
        for b in range(insitu_n_eta):
            theta, r = project(design, (vals[b] - 1.0).reshape(-1))
            resid[step].append(r)
            for k in range(insitu_n_coeff_pt):
                for m in range(insitu_n_coeff_ut):
                    qpart = "" if qtag is None else f"_q{qtag}"
                    lbl = f"{grp}_eta{b}{qpart}_cPt{k}_cUt{m}"
                    theta_central[lbl] = float(theta[k * insitu_n_coeff_ut + m])

# ---- idip: 2D, pt coefficients only ----------------------------------------
grp = insitu_step_group["idip"]
froot = ROOT.TFile.Open(args.smoothSF2DFile)
for qtag in ("plus", "minus"):
    h = froot.Get(f"SF_nomiAndAlt_GtoH_idip_{qtag}")
    if not h:
        raise KeyError(f"SF_nomiAndAlt_GtoH_idip_{qtag} not in {args.smoothSF2DFile}")
    ay, n_pt = h.GetYaxis(), h.GetYaxis().GetNbins()
    pt_c = np.array([ay.GetBinCenter(j + 1) for j in range(n_pt)])
    mp = (pt_c >= PT_LO) & (pt_c <= PT_HI)
    xp = 2.0 * (pt_c[mp] - PT_LO) / (PT_HI - PT_LO) - 1.0
    design = cheb(xp, insitu_n_coeff_pt)
    for b in range(insitu_n_eta):
        # z bin 1 is the nominal of the nomiAndAlt axis
        y = np.array([h.GetBinContent(b + 1, j + 1, 1) for j in range(n_pt)])[mp]
        theta, r = project(design, y - 1.0)
        resid["idip"].append(r)
        for k in range(insitu_n_coeff_pt):
            theta_central[f"{grp}_eta{b}_q{qtag}_cPt{k}"] = float(theta[k])
froot.Close()

# ---- verify against the expected label set ---------------------------------
expected = insitu_parameter_labels(insitu_n_eta, insitu_n_coeff_pt, insitu_n_coeff_ut)
missing = [lbl for lbl in expected if lbl not in theta_central]
extra = [lbl for lbl in theta_central if lbl not in expected]
if missing or extra:
    raise KeyError(
        f"label mismatch: {len(missing)} missing (e.g. {missing[:3]}), "
        f"{len(extra)} unexpected (e.g. {extra[:3]})"
    )
theta_central = {lbl: theta_central[lbl] for lbl in expected}

logger.info(f"Projected external SFs onto the in-situ basis, {len(expected)} coeffs:")
for step in insitu_eff_steps:
    r = resid[step]
    vals = [
        abs(v)
        for lbl, v in theta_central.items()
        if lbl.startswith(insitu_step_group[step])
    ]
    logger.info(
        f"  {step:8s}: max|residual| median {np.median(r):.3e} worst {np.max(r):.3e}"
        f"   |theta| max {max(vals):.4g}"
    )

filename = "insitu_effSF"
if args.postfix:
    filename += f"_{args.postfix}"
outfile = f"{args.outpath}/{filename}.pkl.lz4"
output_dict = {
    "theta_central": theta_central,
    "delta": insitu_delta,
    "steps": insitu_eff_steps,
    "n_eta": insitu_n_eta,
}
output_tools.write_lz4_pkl_output(outfile, "insitu", output_dict, common.base_dir, args)
logger.info(f"Wrote T&P-derived in-situ central SF to {outfile}")
