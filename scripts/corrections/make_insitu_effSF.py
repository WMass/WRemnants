import argparse

from rabbit.io_tools import get_fitresult, get_pulls_and_constraints
from wremnants.production.muon_efficiencies_insitu import (
    insitu_delta,
    insitu_eff_steps,
    insitu_n_coeff_pt,
    insitu_n_coeff_ut,
    insitu_parameter_labels,
    insitu_step_group,
    load_insitu_central,
)
from wremnants.utilities import common
from wums import logging, output_tools

# Iterative in-situ muon-efficiency fit (M10): read a rabbit fit result, extract
# the post-fit values n_hat of the unconstrained in-situ Chebyshev nuisances,
# and accumulate the central coefficients
#
#     theta_central^{n+1}_c = theta_central^n_c + delta * n_hat_c
#
# (delta = insitu_delta, the variation-step folded into the linearised tensor).
# The output pkl (a label-keyed theta_central dict) is fed back to the next
# histmaker run via --insituSFFile, which reweights the MC by the central SF and
# re-linearises the variations around it. Converged when max|delta*n_hat| -> 0.
#
# Iteration 0 has no --prevSFFile (theta_central^0 = 0, == SF = 1).

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--inputFile", type=str, required=True, help="rabbit fitresults.hdf5"
)
parser.add_argument(
    "--prevSFFile",
    type=str,
    default=None,
    help="theta_central pkl from the previous iteration (the file passed as "
    "--insituSFFile to the histmaker that produced this fit). None -> zeros.",
)
parser.add_argument("--outpath", type=str, default="./")
parser.add_argument(
    "-p", "--postfix", type=str, default=None, help="postfix for the output name"
)
parser.add_argument(
    "--result",
    type=str,
    default=None,
    help="named fit result to read (passed to rabbit get_fitresult)",
)
parser.add_argument(
    "--eoscp",
    action="store_true",
    help="copy folder to eos with xrdcp rather than using the mount",
)
parser.add_argument("--debug", action="store_true", help="print debug output")
args = parser.parse_args()

logger = logging.setup_logger("make_insitu_effSF", 4 if args.debug else 3)

# Post-fit nuisance values n_hat for every in-situ coefficient.
fitresult = get_fitresult(args.inputFile, result=args.result)
labels, pulls, _ = get_pulls_and_constraints(fitresult)
prefixes = tuple(insitu_step_group.values())  # effInsituID / HLT / Iso
nhat = {
    str(lbl): float(val)
    for lbl, val in zip(labels, pulls)
    if str(lbl).startswith(prefixes)
}
n_sf = len(nhat)
if n_sf == 0:
    raise ValueError(
        f"No effInsitu* nuisances found in {args.inputFile}; is this an "
        "in-situ fit result?"
    )

# Infer n_eta from the coefficient count (same arithmetic as
# rabbit_helpers.add_muon_insitu_efficiency_systs.relabel).
denom = insitu_n_coeff_pt * (2 + 3 * insitu_n_coeff_ut)
n_eta, rem = divmod(n_sf, denom)
if rem != 0:
    raise ValueError(
        f"{n_sf} effInsitu* nuisances not divisible by kPt*(2+3*kUt)={denom}"
    )
expected = insitu_parameter_labels(n_eta, insitu_n_coeff_pt, insitu_n_coeff_ut)
missing = [lbl for lbl in expected if lbl not in nhat]
if missing:
    raise KeyError(
        f"fit result missing {len(missing)} in-situ nuisances, e.g. {missing[:3]}"
    )

# Accumulate around the previous central value.
prev_arr = load_insitu_central(
    args.prevSFFile, n_eta, insitu_n_coeff_pt, insitu_n_coeff_ut
)
prev = dict(zip(expected, prev_arr))
theta_new = {lbl: prev[lbl] + insitu_delta * nhat[lbl] for lbl in expected}

# Convergence report: the per-iteration step is delta*n_hat; converged at ~0.
logger.info(f"In-situ SF iteration step (delta*n_hat), {n_sf} coeffs, eta={n_eta}:")
overall = 0.0
for step, grp in insitu_step_group.items():
    steps_delta = [
        abs(insitu_delta * nhat[lbl]) for lbl in expected if lbl.startswith(grp)
    ]
    smax = max(steps_delta) if steps_delta else 0.0
    smean = (sum(steps_delta) / len(steps_delta)) if steps_delta else 0.0
    overall = max(overall, smax)
    logger.info(
        f"  {step:8s}: max|delta*n_hat|={smax:.4g}  mean={smean:.4g}  "
        f"({sum(s > 1e-3 for s in steps_delta)} coeffs > 1e-3)"
    )
logger.info(f"  overall  : max|delta*n_hat|={overall:.4g}  (converged when ~0)")

filename = "insitu_effSF"
if args.postfix:
    filename += f"_{args.postfix}"
outfile = f"{args.outpath}/{filename}.pkl.lz4"

output_dict = {
    "theta_central": theta_new,
    "delta": insitu_delta,
    "steps": insitu_eff_steps,
    "n_eta": n_eta,
}
output_tools.write_lz4_pkl_output(outfile, "insitu", output_dict, common.base_dir, args)
logger.info(f"Wrote accumulated in-situ central SF to {outfile}")
