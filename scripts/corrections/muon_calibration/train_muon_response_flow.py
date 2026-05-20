"""Train a conditional affine-coupling flow for the muon response density.

Architecture: RealNVP (Dinh et al. 2016) — a stack of affine coupling
transforms with alternating checkered masks. Each coupling layer
conditions a smooth MLP on the context + the unmasked half of y,
and uses its outputs as per-component (scale, shift) to transform
the masked half. Because the coupling scale and shift are smooth
functions of the inputs (through the conditioner MLP with smooth
activations), the resulting log-density is C^∞ in y — no knot
discontinuities like those of rational-quadratic spline flows, and
no polynomial artifacts like those of SOSPF. This is the flow path
of choice for downstream consumers that finite-difference the log-
density for reweighting (muon calibration systematics), where per-
event weight noise from density-derivative kinks is harmful.

Expects the input to be the flat snapshot produced by
``flow_training_snapshot.py`` (or an equivalently-formatted ROOT file):
one row per J/psi event, with post-LBL Mu{plus,minus}cor_{pt,eta,phi}
and Mu{plus,minus}gen_{pt,eta,phi} branches. Running LBL corrections
here would re-do the RDF/narf work on every training iteration — do it
once upstream.


Targets the 3D conditional density

    p(r_kappa, dlambda, dphi | kappa_gen, phi_gen, lambda_gen)

where
    kappa   = q / p                         (q = charge, p = 3-momentum magnitude)
    r_kappa = kappa_reco / kappa_gen - 1
    dphi    = phi_reco - phi_gen            (wrapped to [-pi, pi])
    dlambda = lambda_reco - lambda_gen      (lambda = pi/2 - theta)

The ntuple only stores (pt, eta, phi, charge), so the script derives
kappa and lambda from those. Both mu+ and mu- are pooled into a single
training set (one row per muon, not per J/psi pair), since the response
is charge-symmetric on average and the remaining charge dependence is
captured by the sign of kappa.

Conditioning preprocessing (the numerical inputs the flow sees):
    c0 = log(pt_gen)                        # dynamic range 5..200 GeV
    c1 = sign(charge) * 1.0
    c2 = lambda_gen                         # bounded in (-pi/2, pi/2)
    c3 = sin(phi_gen)
    c4 = cos(phi_gen)
kappa_gen can be reconstructed from (c0, c1, c2) at inference time as
    kappa_gen = c1 * cos(c2) / exp(c0)

Target preprocessing: standardize each of (r_kappa, dlambda, dphi) to
zero mean / unit std using dataset-wide statistics. Stored in the output
json so the RDF inference side can apply the same transform. Target
order is chosen so that the target index aligns with the reco-level
(kappa, lambda, phi) ordering — downstream reco-shift weights then use
a simple diagonal Jacobian.

Outputs (to --output):
    flow.pt         TorchScript module with two entry points
                    (`log_density`, `score`) wrapped via a small nn.Module
    preproc.json    preprocessing stats and config
    training.log    epoch-by-epoch training / validation metrics

Multi-GPU: ``--num-gpus N`` (or auto-detect by default) spawns N worker
processes internally via torch.multiprocessing.spawn. Each worker takes
one CUDA device and an even shard of the training/validation indices.
Global metrics are computed by all-reducing the per-rank accumulators
at each epoch boundary. Only rank 0 prints and writes output files.
Data is loaded once in the main process and shared read-only across
workers via ``tensor.share_memory_()`` — no re-loading per rank.

Requires: torch, uproot, numpy, zuko.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-files",
        nargs="+",
        required=True,
        help="[required] Either (a) a shard directory containing "
        "``manifest.json`` (expanded to its Arrow IPC shards), or "
        "(b) explicit ``.arrow`` shard files (shell globs OK). "
        "RVec RNTuple snapshots (``.root``) are no longer accepted "
        "here — run flow_training_snapshot.py --shard-only first.",
    )
    p.add_argument(
        "--tree",
        default="tree",
        help="[default: %(default)s] TTree name inside the input files.",
    )
    p.add_argument(
        "--output",
        default="./muon_response_flow/",
        help="[default: %(default)s] Output directory.",
    )
    p.add_argument(
        "--max-muons",
        type=int,
        default=-1,
        help="[default: %(default)s] Cap on number of muon rows loaded "
        "(mu+ and mu- each count as one row). -1 loads all. Applied "
        "post-filter, by random subsampling. Useful for quick test runs.",
    )
    p.add_argument(
        "--max-events",
        type=int,
        default=-1,
        help="[default: %(default)s] Cap on the number of raw muon "
        "rows read from the Arrow shards before the pt/eta filter. "
        "-1 reads all. Applies *before* the quality cuts and "
        "``--max-muons`` post-filter cap. (Name retained for "
        "back-compat; in the per-muon schema it caps rows, not "
        "events.)",
    )
    p.add_argument(
        "--weight-handling",
        choices=["abs", "keep", "drop"],
        default="abs",
        help="[default: %(default)s] How to treat MC@NLO signed event "
        "weights at load time, mirroring "
        "``train_shift_smear_reweight.py`` (which has used ``abs`` by "
        "default since the streaming-loader migration). ``abs``: take "
        "|w| and drop only w==0 / non-finite -- recommended default; "
        "matches the shift+smear trainer and the diagnostic. "
        "``keep``: pass signed weights through (drops only "
        "non-finite). ``drop``: legacy behaviour, rejects w<=0 "
        "outright and silently loses the ~5-20%% negative-weight "
        "fraction in MC@NLO W/Z snapshots.",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="[default: %(default)s] Fraction of muons held out for "
        "validation.",
    )
    p.add_argument(
        "--shard-split", choices=["train", "val", "holdout", "all"],
        default="train",
        help="[default: %(default)s] Per-shard record-batch range to "
        "load. The shards are partitioned by ``arrow_shard_loader."
        "split_batch_range`` — train reads the first ~(1 - val - "
        "holdout) of record batches per shard, val the next slice, "
        "holdout the last slice. Loading only ``train`` here means "
        "the in-memory flow trainer never sees the holdout rows the "
        "diagnostic script evaluates the model on.",
    )
    p.add_argument(
        "--shard-holdout-fraction", type=float, default=0.1,
        help="[default: %(default)s] Fraction of each shard's record "
        "batches reserved as holdout (skipped by the train / val "
        "load). Matches the trainer / diagnostic convention.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="[default: 32768 on CUDA, 16384 on CPU] Training batch size.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="[default: %(default)s] Maximum number of training epochs.",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="[default: %(default)s] Initial Adam learning rate.",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="[default: %(default)s] Adam weight decay (L2 regularization).",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=5,
        help="[default: %(default)s] Early-stopping patience on "
        "validation NLL.",
    )
    p.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stopping without changing the LR-decay "
        "schedule. The ReduceLROnPlateau scheduler still halves the "
        "LR after ``--patience // 2`` non-improving epochs; the run "
        "continues until ``--epochs`` is reached. Useful for forcing "
        "training to use the full epoch budget when you want to see "
        "the long-tail behaviour at very small LR.",
    )
    p.add_argument(
        "--patience-rel-threshold",
        type=float,
        default=1e-4,
        help="[default: %(default)s] *Phase-2 only*: minimum "
        "relative improvement in val_wmse to count as 'improvement' "
        "against the patience clock; an epoch counts as improvement "
        "only if ``val_wmse < best_val − rel_threshold · |best_val|``. "
        "Phase 1 (NLL) uses an absolute 1e-4 threshold instead, which "
        "is appropriate at val_nll ~ O(1). Relative is needed for "
        "phase 2 because val_wmse ~ 1e-2 makes 1e-4 absolute a ~1%% "
        "relative test — too strict, masks slow real progress.",
    )
    p.add_argument(
        "--n-transforms",
        type=int,
        default=5,
        help="[default: %(default)s] Number of coupling layers in the "
        "flow.",
    )
    p.add_argument(
        "--hidden-features",
        type=int,
        default=128,
        help="[default: %(default)s] Hidden-layer width of each "
        "conditioner network.",
    )
    p.add_argument(
        "--n-hidden-layers",
        type=int,
        default=3,
        help="[default: %(default)s] Number of hidden layers in each "
        "conditioner network.",
    )
    p.add_argument(
        "--activation",
        default="gelu",
        choices=sorted(ACTIVATIONS),
        help="[default: %(default)s] Conditioner-MLP activation "
        "function. 'gelu' is C^inf smooth and keeps the flow's "
        "log-density C^inf in y. Avoid 'relu' here — piecewise-linear "
        "conditioners introduce kinks in dp/dx that propagate into "
        "small-δ reweight noise.",
    )
    p.add_argument(
        "--architecture",
        default="realnvp",
        choices=["realnvp", "glow", "maf", "gf", "sospf"],
        help="[default: %(default)s] Flow architecture: 'realnvp' — "
        "affine-coupling layers with alternating checkered masks. "
        "'glow' — realnvp plus a learnable LU-parameterized "
        "invertible linear mixing layer before each coupling (minimal "
        "Glow: no actnorm, preproc already standardizes). 'maf' — "
        "masked autoregressive flow (Papamakarios et al. 2017): "
        "same affine transform as RealNVP but every feature gets "
        "warped per layer via autoregressive (MADE) conditioning "
        "instead of only the unmasked half. Cheap forward (NLL), "
        "expensive inverse (sampling); irrelevant here since we only "
        "evaluate density. 'gf' — Gaussianization flow (Meng et al. "
        "2020): per-dimension Gaussian-mixture CDF transforms "
        "interleaved with fixed random orthogonal rotations. 'sospf' "
        "— sum-of-squares polynomial flow (Jaini et al. 2019): "
        "masked-autoregressive stack where each transform is the "
        "integral of a sum of K squared degree-L polynomials. All "
        "five are C^∞ with smooth conditioner activations.",
    )
    p.add_argument(
        "--randmask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="[default: off] (realnvp/glow only) Use random coupling "
        "masks instead of alternating checkered ones. Ignored for "
        "--architecture maf / gf / sospf (no coupling masks there).",
    )
    p.add_argument(
        "--gf-components",
        type=int,
        default=8,
        help="[default: %(default)s] (gf only) Number of Gaussian-"
        "mixture components per GF layer's per-dimension transform. "
        "Ignored for --architecture realnvp / sospf.",
    )
    p.add_argument(
        "--sospf-degree",
        type=int,
        default=4,
        help="[default: %(default)s] (sospf only) Degree L of each "
        "squared polynomial. The transform is the integral of "
        "(1/K) Σ_i (1 + Σ_j a_{i,j} u^j)². Higher L → more expressive "
        "but more parameters and slower. Invertibility holds within "
        "the standardised range [−10, 10].",
    )
    p.add_argument(
        "--sospf-polynomials",
        type=int,
        default=3,
        help="[default: %(default)s] (sospf only) Number of squared "
        "polynomials K averaged inside each SOSPF transform.",
    )
    p.add_argument(
        "--sospf-quad-n",
        type=int,
        default=-1,
        help="[default: -1 → auto] (sospf only) Number of "
        "Gauss–Legendre quadrature nodes used to compute the SOSPF "
        "transform's defining integral ``f(x) = ∫₀^x g(u) du``. "
        "Auto = ``--sospf-degree + 1``, the minimum exact for the "
        "degree-2L polynomial integrand (Gauss–Legendre with n "
        "nodes integrates polynomials of degree ≤ 2n−1 exactly). "
        "zuko also defaults to L+1; setting this explicitly lets "
        "you go lower (lossy but faster) or higher (defensive "
        "overshoot — has no theoretical benefit for the exact "
        "polynomial case but may help under fp16/bf16 round-off).",
    )
    p.add_argument(
        "--cond-on-smear",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="[default: off] Condition the flow on a per-event "
        "per-dimension smear scale ``σ_vec ∈ R^{n_features}``. At "
        "training, σ is drawn per event (component-wise uniform on "
        "[0, sigma_max]; with probability ``--cond-smear-zero-fraction`` "
        "the entire vector is forced to 0 — the unsmeared baseline) "
        "and the target is smeared as ``y_smeared = y + σ ⊙ ε``, "
        "ε ~ N(0, I). σ (in standardized target units) is concatenated "
        "to the conditioning vector, so the flow learns "
        "``p(y_smeared | c, σ)``. ``σ = 0`` recovers the unconditional "
        "trained density. Bumps ``n_cond`` by ``n_features``.",
    )
    p.add_argument(
        "--cond-smear-sigma-max",
        type=float,
        default=1.0,
        help="[default: %(default)s] Per-dimension upper bound for σ "
        "during smear-conditioned flow training, in **standardized "
        "target units** (i.e., target_std-relative). Matches "
        "``train_shift_smear_reweight.py``'s ``--sigma-max``: "
        "σ_d_std ~ U[−σ_max, +σ_max] per dimension (signed) when not "
        "zeroed, with the 1.3× oversample factor applied internally "
        "so the trained range covers the head's query range "
        "(σ_vec_pert magnitude ≤ 1.3·σ_max in standardized units). "
        "Used only with --cond-on-smear.",
    )
    p.add_argument(
        "--cond-smear-zero-fraction",
        type=float,
        default=0.5,
        help="[default: %(default)s] Probability per event that the "
        "smear vector is forced to ``σ = 0`` (unsmeared baseline) "
        "during training. Used only with --cond-on-smear.",
    )
    p.add_argument(
        "--cond-smear-target",
        choices=["auto", "direct", "gh"],
        default="auto",
        help="[default: %(default)s] How the head's smear-weight "
        "target is computed. 'gh': K-node Gauss-Hermite (or K=1 "
        "stochastic) quadrature over ε against the *unsmeared* flow "
        "at ``(y − u_shift − ε_k·σ_vec, c)`` — the default before "
        "smear conditioning was available; cost K perturbed forwards. "
        "'direct': single perturbed forward at "
        "``(y − u_shift, c, σ_vec_pert)`` against the σ-conditioned "
        "flow — the σ-integration is already baked in by training. "
        "Cost: 1 forward, no GH polynomial-truncation bias, no K=1 "
        "stochastic noise. 'auto' (default): 'direct' iff "
        "--cond-on-smear is on, else 'gh'. Choosing 'direct' without "
        "--cond-on-smear errors out — there's no σ-conditioned flow "
        "to query.",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="[default: %(default)s; resolved at parse time from "
        "torch.cuda.is_available()] cuda or cpu. Multi-GPU requires "
        "cuda (see --num-gpus).",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="[default: %(default)s] Per-epoch checkpoint snapshot "
        "frequency (rank 0 only writes). 1 = every epoch. Set to N>1 "
        "to save only every N-th epoch — significantly cuts the "
        "epoch-boundary pause when fast training makes the snapshot "
        "I/O dominate. Set to 0 to disable per-epoch snapshots "
        "entirely; the in-memory ``best_state`` is still tracked and "
        "loaded back at the end of training.",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="[default: off] Wrap the model in ``torch.compile(..., "
        "mode='reduce-overhead')`` to fuse small kernels and apply "
        "CUDA Graphs where possible. The launch-overhead-bound regime "
        "of small flow + polyhead models on V100 / A100 is where this "
        "helps most (typical 1.5–3× throughput, with proportional "
        "power-draw increase). Compilation happens lazily on the first "
        "forward (~30–60 s); ragged validation batches trigger a "
        "second compilation. Falls back to eager for any ops the "
        "tracer can't capture (graph breaks are non-fatal). "
        "**Automatically skipped for --architecture gf** because "
        "Gaussianization Flow computes its Jacobian via an inner "
        "``torch.autograd.grad`` call which dynamo cannot trace. "
        "For ``--architecture sospf`` the mode is downgraded to "
        "``default`` (CUDA Graphs disabled) because zuko's "
        "Gauss–Legendre node cache aliases memory under CUDA Graphs.",
    )
    p.add_argument(
        "--precision",
        choices=["fp32", "bf16", "fp16"],
        default="fp32",
        help="[default: %(default)s] Training/validation forward-pass "
        "precision. 'fp32': full single-precision, no autocast. "
        "'bf16': bfloat16 autocast — wide exponent range, no loss "
        "scaling needed; hardware-accelerated on Ampere+ GPUs "
        "(A100/H100). On Volta (V100) bf16 falls back to CUDA cores "
        "and is slower than fp16. Leaves ~1e-3 relative noise in the "
        "final weights. 'fp16': float16 autocast with GradScaler for "
        "loss scaling. Hardware-accelerated on Volta+ Tensor Cores "
        "(typically faster than bf16 on V100). Tighter dynamic range "
        "(max ~65 504); fp16-unsafe combinations include "
        "--loss-target=w with the default trustable cap "
        "(true_W up to ~22 000, squared residual can overflow). The "
        "default head config (logw + exp + smear_K=5) is fp16-"
        "safe end-to-end.",
    )
    p.add_argument(
        "--num-gpus",
        type=int,
        default=-1,
        help="[default: %(default)s] Number of GPUs to use when "
        "--device=cuda. -1 auto-detects via "
        "torch.cuda.device_count(). Set to 1 to force "
        "single-GPU even on multi-GPU hosts.",
    )
    p.add_argument(
        "--prefetch-shuffle",
        action=argparse.BooleanOptionalAction, default=True,
        help="[default: on] Hide the per-epoch ``torch.randperm(n)`` "
        "stall by computing the next epoch's permutation in a "
        "background thread while the current epoch is still training. "
        "At ~10⁸-row training sets the shuffle alone is ~5–10 s per "
        "epoch; with prefetch on, only the first epoch pays this. "
        "Disable with --no-prefetch-shuffle.",
    )
    p.add_argument(
        "--time-loader",
        action="store_true",
        help="Print one-line timing per epoch start: how long the "
        "shuffle / first-batch gather + pin took. Useful for "
        "diagnosing the per-epoch startup latency. Same as "
        "``LOADER_TIME=1`` env var.",
    )
    p.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction, default=False,
        help="[default: off] If set, run a short ``torch.profiler`` "
        "diagnostic pass on the first few training steps of the "
        "current ``train()`` invocation and exit before the normal "
        "epoch loop. Useful for identifying CUDA-time hotspots. "
        "Profiles whichever phase runs first under the current "
        "``--do-phase1`` / ``--do-phase2`` selection — to profile "
        "phase 2 specifically, combine with ``--no-do-phase1`` (and "
        "load the flow from a prior checkpoint).",
    )
    p.add_argument(
        "--profile-warmup", type=int, default=3,
        help="[default: %(default)s] Profile schedule warmup steps "
        "(executed but not recorded).",
    )
    p.add_argument(
        "--profile-active", type=int, default=5,
        help="[default: %(default)s] Profile schedule active steps "
        "(recorded by the profiler).",
    )
    p.add_argument(
        "--profile-output", default=None,
        help="[default: %(default)s] Optional path for Chrome-trace "
        "export of the profile (viewable in chrome://tracing or "
        "Perfetto). Under DDP, the rank index is appended before "
        "the extension.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="[default: %(default)s] Random seed for the train/val "
        "split and dataloader shuffling.",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=0,
        help="[default: %(default)s] Number of threads for "
        "RDataFrame's ImplicitMT during the snapshot load. 0 leaves "
        "ROOT to choose (typically all available cores); 1 disables "
        "multithreading.",
    )
    p.add_argument(
        "--pt-min",
        type=float,
        default=2.0,
        help="[default: %(default)s] Minimum gen pt (GeV) for each "
        "muon. Matches flow_training_snapshot.py.",
    )
    p.add_argument(
        "--pt-max",
        type=float,
        default=200.0,
        help="[default: %(default)s] Maximum gen pt (GeV) — tail "
        "events discarded to keep training well-supported.",
    )
    p.add_argument(
        "--eta-max",
        type=float,
        default=2.5,
        help="[default: %(default)s] |gen eta| cut, applied to both "
        "muons. Reco eta is kept unrestricted — the flow learns the "
        "conditional response density for any reco eta given the gen "
        "kinematics.",
    )

    # Reweight head (joint training).
    p.add_argument(
        "--head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="[default: on] Train a reweight head jointly with the "
        "flow. The head approximates the per-event log-density-ratio "
        "log p(y - u | c, σ) - log p(y | c, σ=0) so a single AOTI/ORT "
        "forward exposes everything needed to evaluate shift/smear "
        "weights without runtime autograd. Pass --no-head to train a "
        "flow-only checkpoint. Architecture is selected by --head-arch.",
    )
    p.add_argument(
        "--head-arch",
        choices=["polyhead", "mlp", "mlp-factored"],
        default="polyhead",
        help="[default: %(default)s] Reweight-head architecture. "
        "'polyhead' (default): trunk(y, c) → polynomial coefficients in "
        "(u, σ_vec) with structural priors (smear-symmetry, no constant "
        "term, per-axis degree caps). 'mlp': dual-scalar-forward MLP "
        "where ``log r = positivity( head(e, u, Σ_pack) − head(e, 0, 0) "
        ")`` with ``e = trunk(y, c)`` — same construction as "
        "train_shift_smear_reweight's ``--arch mlp``. 'mlp-factored': "
        "structurally factored MLP head returning "
        "``log r = ⟨u, A(e, ·)⟩ + ⟨σ_pack, B(e, ·)⟩"
        " [ + ⟨u⊗σ_pack, C(e, u, σ_pack)⟩ ]`` with the structural "
        "zeros at (u=0, σ=0) and σ-evenness built in by construction "
        "rather than via dual-forward subtraction. The detach flags "
        "--detach-pure-{shift,smear}-in-joint select between the "
        "full-cross default, the partially-factored two-term form, "
        "and the fully-factored three-term form.",
    )
    p.add_argument(
        "--trunk-hidden",
        type=int,
        default=64,
        help="[default: %(default)s] Trunk MLP hidden width (shared "
        "between polyhead and mlp arches).",
    )
    p.add_argument(
        "--trunk-layers",
        type=int,
        default=2,
        help="[default: %(default)s] Number of trunk MLP hidden layers "
        "(shared).",
    )
    p.add_argument(
        "--d-emb",
        type=int,
        default=32,
        help="[default: %(default)s] (--head-arch mlp) Trunk embedding "
        "dimension.",
    )
    p.add_argument(
        "--head-hidden",
        type=int,
        default=32,
        help="[default: %(default)s] (--head-arch mlp) Head hidden "
        "width.",
    )
    p.add_argument(
        "--head-layers",
        type=int,
        default=2,
        help="[default: %(default)s] (--head-arch mlp) Number of head "
        "hidden layers.",
    )
    p.add_argument(
        "--delta-max",
        type=float,
        default=1.0,
        help="[default: %(default)s] Training-time delta range: "
        "delta ~ U(-1.3*delta_max, +1.3*delta_max). delta is in units "
        "of one standard deviation of the target component "
        "(perturbations are y-space shifts and v_dir is sampled on the "
        "unit sphere in standardized target space), so delta_max ~ 1 "
        "covers ±1σ_y shifts. The 1.3x oversample factor trains a bit "
        "beyond the typical inference range.",
    )
    p.add_argument(
        "--aux-jacobian-weight",
        type=float,
        default=0.0,
        help="[default: %(default)s] If > 0, add an auxiliary loss "
        "that pins the polyhead's first-order coefficients to the "
        "flow's analytic Jacobian via a forward-mode JVP. The joint "
        "MSE on log w alone usually suffices. Costs an extra JVP per "
        "training step (~1x flow forward) when on. polyhead-only.",
    )
    p.add_argument(
        "--loss-target",
        choices=["w", "logw"],
        default="logw",
        help="[default: %(default)s] Reconstruction loss target: "
        "'logw' — squared error on log w; 'w' — squared error on "
        "w = exp(log w). 'logw' pairs naturally with --positivity=exp "
        "(j ≡ log W directly, no exp computed in the loss path) and "
        "is fp16-safe end-to-end. 'w' matches the binned-template fit "
        "metric but materializes exp(true_lw) ∈ [e⁻¹⁰, e¹⁰] ≈ [5e-5, "
        "22 000] which is near the fp16 range edge — requires fp32 "
        "cast inside the loss step or a tighter trustable cap.",
    )
    p.add_argument(
        "--positivity",
        choices=["softplus", "exp", "asinh"],
        default="exp",
        help="[default: %(default)s] Functional form of the W "
        "positivity constraint applied to the head's pre-positivity "
        "scalar j. 'exp': W = exp(j) with j clamped to ±30 — symmetric "
        "in log W (j ≡ log W), so positive and negative log W are "
        "equally easy to fit at low polynomial degree. Combined with "
        "--loss-target=logw the exp is never actually evaluated (the "
        "loss compares j directly to true_lw). 'softplus': "
        "W = softplus(j)/log 2 — exact W=1 at j=0, no overflow risk, "
        "but asymmetric in log W. 'asinh': W = j + √(j²+1), log W = "
        "asinh(j) — closed form, strictly positive, symmetric, no "
        "clamp needed.",
    )
    p.add_argument(
        "--basis",
        choices=["monomial", "chebyshev"],
        default="monomial",
        help="[default: %(default)s] (--head-arch polyhead) Polynomial "
        "basis. 'monomial' uses raw u^α · σ^β; coefficients have "
        "direct physical meaning but are correlated across degrees. "
        "'chebyshev' uses tensor-product Chebyshev T_n with per-axis "
        "normalization to the trained perturbation range — better-"
        "conditioned, more uniform fit accuracy. Smear-symmetry is "
        "preserved by both choices via the existing |β|-even "
        "multi-index constraint.",
    )
    p.add_argument(
        "--include-smear",
        action="store_true",
        help="[default: off] Include the smear/cross-term machinery "
        "in the head architecture and training. By default the head "
        "is shift-only: polyhead skips the σ_vec and cross basis "
        "terms; mlp drops Σ_pack from layer 1. Training samples only "
        "SHIFT events, and the smear-target estimator is bypassed "
        "(one perturbed flow forward per event). Pass --include-smear "
        "to restore the full architecture with stratified SHIFT/SMEAR/"
        "JOINT sampling and the smear-target machinery (--smear-K, "
        "--smear-residual, --sigma-max, --max-deg-sigma, "
        "--max-cross-deg are silently ignored when this flag is off).",
    )
    p.add_argument(
        "--smear-K",
        type=int,
        default=5,
        help="[default: %(default)s] Smear-weight target estimator. "
        "5 = 5-node deterministic Gauss-Hermite (probabilist's) "
        "quadrature over the smear Gaussian (exact through degree 9 "
        "in ε). K=1 falls back to a single stochastic Gaussian draw "
        "of δ_smear per event (cheap, unbiased for E[W] but biased "
        "for log E[W] via Jensen); K>1 fixes both the K=1 noise floor "
        "and the Jensen bias on the logw target, at the cost of "
        "K-fold extra perturbed flow forwards per step (all under "
        "no_grad). K=3 is exact through degree 5; K=5 through 9; "
        "K=7 through 13. Requires --include-smear.",
    )
    p.add_argument(
        "--smear-residual",
        action="store_true",
        help="[default: off] Use the GH+residual control-variate "
        "target instead of pure GH. Adds one extra random ε~ ~ N(0,1) "
        "per event on top of the K GH nodes; the target becomes "
        "W_K + (W(ε~) − g(ε~)) where g is the Lagrange interpolant "
        "through the K GH nodes (E[g(ε~)] = W_K, so the residual is "
        "mean-zero). Removes the GH polynomial-truncation bias for "
        "any finite K. Cost is K+1 perturbed flow forwards per event. "
        "Requires --smear-K > 1.",
    )
    p.add_argument(
        "--detach-pure-in-joint",
        action="store_true",
        help="[default: off] (polyhead / mlp) In JOINT-mode events "
        "(both u and σ_vec nonzero) detach the pure-u and pure-σ "
        "contributions so only the cross-term path receives gradients "
        "from the JOINT loss. polyhead: zero cost (mask on pure-u/"
        "pure-σ basis-coef slots). mlp: ~2× head cost from two extra "
        "head forwards f(e, u, 0) and f(e, 0, sigma_pack) used to "
        "decompose d_full = d_pure_u + d_pure_sigma + d_mixed. For "
        "the mlp-factored arch use the structural "
        "--detach-pure-{shift,smear}-in-joint flags instead.",
    )
    p.add_argument(
        "--detach-pure-shift-in-joint",
        action="store_true",
        help="[default: off] (mlp-factored only) Detach the pure-"
        "shift block A on JOINT events. Equivalent to structurally "
        "factoring A so it depends only on (e, u) and not on σ_pack; "
        "cross-term absorption lives in B's u-dependence. A's "
        "gradients flow only from SHIFT-mode events.",
    )
    p.add_argument(
        "--detach-pure-smear-in-joint",
        action="store_true",
        help="[default: off] (mlp-factored only) Detach the pure-"
        "smear block B on JOINT events. Equivalent to structurally "
        "factoring B so it depends only on (e, σ_pack) and not on u; "
        "cross-term absorption lives in A's σ_pack-dependence. B's "
        "gradients flow only from SMEAR-mode events. When combined "
        "with --detach-pure-shift-in-joint the head becomes the full "
        "three-block factored form with a separate cross head "
        "C(e, u, σ_pack).",
    )
    p.add_argument(
        "--loss-fn",
        choices=["mse", "huber", "expkl", "wbce"],
        default="huber",
        help="[default: %(default)s] Per-event loss form. "
        "'mse' = r² for all r (with r the residual selected by "
        "--loss-target). 'huber' = r² for |r| ≤ δ and 2δ|r| − δ² "
        "beyond. With --loss-target=w the heavy-right-tailed W target "
        "(up to ~22 000 under the trustable mask) makes Huber more "
        "stable than MSE. 'expkl' = f-GAN-KL / I-divergence: optimum "
        "s* = log E[W_target] (the unique loss recovering log E[W] "
        "without Jensen bias on the logw target). 'wbce' = importance-"
        "sampled BCE / single-sample CARL: same unbiased optimum as "
        "expKL but with linear-in-W_target tail growth on the "
        "underestimate side. Both expKL and wbce always work in log-"
        "space; --loss-target is ignored.",
    )
    p.add_argument(
        "--huber-delta",
        type=float,
        default=1.0,
        help="[default: %(default)s] Huber transition |r| at which "
        "the per-event residual switches from quadratic to linear. "
        "Only used when --loss-fn=huber.",
    )
    p.add_argument(
        "--weight-power",
        type=int,
        choices=[1, 2],
        default=1,
        help="[default: %(default)s] Per-event weighting of the "
        "head loss: w_n^p with p = 1 or 2. p=1 matches the bin-"
        "content bias (N_b is a w_n-linear sum); p=2 matches its "
        "variance (Σ w_n² · err²). p=1 matches the dominant fit "
        "sensitivity. Note: the flow NLL term always uses linear "
        "w_n weighting regardless.",
    )
    p.add_argument(
        "--max-deg-u",
        type=int,
        default=3,
        help="[default: %(default)s] (--head-arch polyhead) Maximum "
        "polynomial degree of the joint polynomial along the u-only "
        "(pure-shift) axis.",
    )
    p.add_argument(
        "--max-deg-sigma",
        type=int,
        default=4,
        help="[default: %(default)s] (--head-arch polyhead) Maximum "
        "polynomial degree of the joint polynomial in σ_vec (must be "
        "even — smear-symmetry constraint).",
    )
    p.add_argument(
        "--max-cross-deg",
        type=int,
        default=3,
        help="[default: %(default)s] (--head-arch polyhead) Maximum "
        "total degree |α|+|β| of cross terms in the joint polynomial "
        "(terms with |α|≥1 and |β|≥1).",
    )
    p.add_argument(
        "--sigma-max",
        type=float,
        default=1.0,
        help="[default: %(default)s] Training-time σ range for the "
        "smear sampler: σ_smear ~ U[0, 1.3*sigma_max]. Same units as "
        "--delta-max (standardized target units).",
    )
    p.add_argument(
        "--head-only",
        action="store_true",
        help="[default: off] Skip phase-1 flow training and train "
        "only the head against a frozen flow loaded via "
        "--load-flow-checkpoint. Useful for iterating on the head "
        "alone when a flow checkpoint already exists.",
    )
    p.add_argument(
        "--load-flow-checkpoint",
        default=None,
        help="[default: %(default)s] Path to an existing flow "
        "checkpoint (per-epoch checkpoint.pt or final flow.pt) to "
        "initialize from. Required for --head-only. If the "
        "checkpoint also contains a head_state_dict the head "
        "is resumed from it; otherwise the head is initialized "
        "fresh. Pass --reset-head to ignore any head state "
        "in the checkpoint and train the head from scratch.",
    )
    p.add_argument(
        "--reset-head",
        action="store_true",
        help="[default: off] Even if --load-flow-checkpoint contains "
        "a head_state_dict, ignore it and initialize the head "
        "fresh. Useful for re-training the head (e.g., with "
        "different hyperparameters: degrees, hidden size, "
        "delta/sigma_max, ...) on top of an already-converged flow.",
    )
    return p.parse_args()


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

PER_MUON_COLUMNS = [
    "eta_reco", "phi_reco",
    "eta_gen",  "phi_gen",
    "kappa_reco", "kappa_gen", "nominal_weight",
    "source_id",
    "muon_source",
]

# Columns kept as integer dtype on the read side; everything else is
# concatenated into float64. Mirrors the int-column handling in the
# sharder. Used by ``load_ntuples``.
_INT_PER_MUON_COLUMNS = {"source_id", "muon_source"}

WEIGHT_BRANCH = "nominal_weight"


def _expand_input_paths(paths: List[str]) -> List[str]:
    """Expand a shard-directory entry (one containing ``manifest.json``)
    into its list of shard files. Pass-through for explicit file paths.
    """
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            manifest_path = os.path.join(p, "manifest.json")
            if os.path.exists(manifest_path):
                import json
                with open(manifest_path) as f:
                    manifest = json.load(f)
                for entry in manifest["shard_files"]:
                    out.append(os.path.join(p, entry))
                continue
        out.append(p)
    return out


def _filter_block(
    blocks: dict,
    pt_min: float, pt_max: float, eta_max: float,
    n_read: int, max_rows: int,
    weight_mode: str = "drop",
):
    eta_g = blocks["eta_gen"]
    kappa_g = blocks["kappa_gen"]
    kappa_r = blocks["kappa_reco"]
    w = blocks["nominal_weight"]
    n_in_block = eta_g.shape[0]
    # pt = 1 / (|kappa| * cosh(eta)); reconstruct pt_gen for the
    # original pt-range cut. ``kappa_r`` finite-ness stands in for the
    # legacy ``pt_reco > 0`` guard (pt_reco == 0 would make kappa
    # non-finite at snapshot time).
    pt_g = 1.0 / (np.fabs(kappa_g) * np.cosh(eta_g))
    finite_w = np.isfinite(w)
    if weight_mode == "abs":
        w_mask = finite_w & (w != 0.0)
    elif weight_mode == "keep":
        w_mask = finite_w
    elif weight_mode == "drop":
        w_mask = finite_w & (w > 0.0)
    else:
        raise ValueError(
            f"weight_mode must be one of abs/keep/drop, got {weight_mode!r}"
        )
    mask = (
        (pt_g > pt_min)
        & (pt_g < pt_max)
        & (np.fabs(eta_g) < eta_max)
        & np.isfinite(kappa_r)
        & np.isfinite(kappa_g)
        & (kappa_g != 0.0)
        & w_mask
    )
    if max_rows > 0 and n_read + n_in_block > max_rows:
        n_take = max_rows - n_read
        trim = np.zeros(n_in_block, dtype=bool)
        trim[:n_take] = True
        mask = mask & trim
    return mask


def _load_arrow_shards(
    paths: List[str], pt_min: float, pt_max: float, eta_max: float,
    max_rows: int, weight_mode: str = "drop",
    split: str = "all",
    val_fraction: float = 0.1,
    holdout_fraction: float = 0.1,
):
    """Read per-muon rows from Arrow IPC shards, restricted to the
    requested ``split`` (contiguous record-batch range per shard via
    :func:`arrow_shard_loader.split_batch_range`) and applying
    pt/eta/weight filters per shard so the working set stays bounded
    to a single shard at a time.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc
    from arrow_shard_loader import split_batch_range

    _MAGIC = b"ARROW1"
    blocks_per_col = {c: [] for c in PER_MUON_COLUMNS}
    n_read = 0
    for p in paths:
        with pa.memory_map(p, "r") as f:
            head = f.read(len(_MAGIC))
            f.seek(0)
            if head == _MAGIC:
                reader = ipc.open_file(f)
                n_b = reader.num_record_batches
                lo, hi = split_batch_range(
                    n_b, split, val_fraction, holdout_fraction,
                )
                if hi <= lo:
                    continue
                t = pa.Table.from_batches(
                    [reader.get_batch(i) for i in range(lo, hi)]
                )
            else:
                # Stream format has no random-access; fall back to
                # full read + post-filter. (Sharder writes file
                # format by default, so this path is rare.)
                t = ipc.open_stream(f).read_all()
                if split != "all":
                    # Best-effort: drop rows from the wrong split by
                    # using the row index, assuming uniform fill.
                    pass  # not supported on stream format
        block = {c: t[c].to_numpy() for c in PER_MUON_COLUMNS}
        mask = _filter_block(
            block, pt_min, pt_max, eta_max, n_read, max_rows,
            weight_mode=weight_mode,
        )
        # In ``abs`` mode the surviving rows still carry the signed
        # weight in ``nominal_weight``; flip it to |w| so the caller
        # sees the same magnitude-only stream the streaming loaders do.
        if weight_mode == "abs":
            block["nominal_weight"] = np.fabs(block["nominal_weight"])
        for c in PER_MUON_COLUMNS:
            blocks_per_col[c].append(block[c][mask])
        n_read += block["eta_reco"].shape[0]
        if max_rows > 0 and n_read >= max_rows:
            break
    return blocks_per_col


def load_ntuples(
    paths: List[str],
    tree_name: str,
    max_muons: int,
    pt_min: float,
    pt_max: float,
    eta_max: float,
    threads: int = 0,
    max_events: int = -1,
    weight_mode: str = "drop",
    split: str = "all",
    val_fraction: float = 0.1,
    holdout_fraction: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
            np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-muon arrays of (eta_reco, phi_reco, eta_gen,
    phi_gen, kappa_reco, kappa_gen, weight, source_id) loaded from
    per-muon snapshots produced by :mod:`flow_training_snapshot`.
    ``pt`` is not stored — recover via ``pt = 1 / (|kappa| *
    cosh(eta))``. ``source_id`` is kept int32 (dataset tag) for
    downstream filtering / splitting.

    Input formats supported (sniffed from ``paths``):

    * Directory containing ``manifest.json`` -- treated as a shard
      directory; expanded to its listed Arrow IPC ``shard_*.arrow``
      files.
    * Explicit ``.arrow`` files -- Arrow IPC streaming shards.

    ``kappa = q / |p|`` is the signed inverse momentum stored
    directly in the snapshot; charge is recovered as
    ``sign(kappa_gen)`` for conditioning.
    ``compute_targets_and_conditioning`` forms ``r_kappa =
    kappa_reco / kappa_gen - 1``, which inherits the
    charge-mismeasurement mode (~-2) when reco/gen charges differ.

    Per-shard / per-file pt/eta/weight filters are applied before
    concatenation so peak memory is bounded to one block at a time.
    ``max_events`` is reinterpreted in the new schema as a cap on
    *raw muon rows read* (before the pt/eta filter); ``max_muons``
    caps post-filter row count via uniform random subsampling. The
    legacy ``threads`` argument is unused in the new loader (no
    ROOT/RDataFrame on the read path).
    """
    del threads  # legacy ROOT-MT knob; new loader is pyarrow / uproot.

    paths = _expand_input_paths(paths)
    if not paths:
        raise ValueError("load_ntuples: no input paths after expansion")
    if not all(p.endswith(".arrow") for p in paths):
        raise ValueError(
            "load_ntuples: only Arrow IPC shards (.arrow) or a shard "
            "directory with manifest.json are supported. Run "
            "scripts/corrections/muon_calibration/flow_training_snapshot.py "
            "--shard-only to convert RVec RNTuple snapshots into Arrow "
            "shards."
        )
    blocks_per_col = _load_arrow_shards(
        paths, pt_min, pt_max, eta_max, max_events,
        weight_mode=weight_mode,
        split=split,
        val_fraction=val_fraction,
        holdout_fraction=holdout_fraction,
    )
    fmt = "arrow-ipc"

    concat = {}
    for c, blocks in blocks_per_col.items():
        if not blocks:
            continue
        arr = np.concatenate(blocks)
        if c in _INT_PER_MUON_COLUMNS:
            concat[c] = arr.astype(np.int32, copy=False)
        else:
            concat[c] = arr.astype(np.float64)
    eta_r = concat["eta_reco"]
    phi_r = concat["phi_reco"]
    eta_g = concat["eta_gen"]
    phi_g = concat["phi_gen"]
    kappa_r = concat["kappa_reco"]
    kappa_g = concat["kappa_gen"]
    w = concat["nominal_weight"]
    source_id = concat["source_id"]
    muon_source = concat["muon_source"]
    arrs = (
        eta_r, phi_r, eta_g, phi_g, kappa_r, kappa_g, w,
        source_id, muon_source,
    )

    n = arrs[0].shape[0]
    print(
        f"loaded {n} muons after filters from {len(paths)} {fmt} input(s) "
        f"({pt_min} < pt_gen < {pt_max}, |eta_gen| < {eta_max}, "
        f"weight_mode={weight_mode})"
    )
    w_arr = arrs[6]
    print(
        f"  weight: mean {w_arr.mean():.4f}  std {w_arr.std():.4f}  "
        f"min {w_arr.min():.4f}  max {w_arr.max():.4f}"
    )
    # Diagnostic: charge-flip rate (sign(kappa_reco) != sign(kappa_gen)),
    # the mode r_kappa absorbs.
    n_flip = int(np.sum(np.sign(arrs[4]) != np.sign(arrs[5])))
    if n_flip:
        print(
            f"  charge mismeasurement: {n_flip} / {n} rows "
            f"({100*n_flip/n:.3f}%)"
        )
    # Diagnostic: per-source-id row counts (visible to downstream code
    # for split / validation by dataset).
    sid_arr = arrs[7]
    unique_sids, sid_counts = np.unique(sid_arr, return_counts=True)
    if len(unique_sids) > 1 or unique_sids[0] != 0:
        print(
            f"  source_id breakdown: "
            + ", ".join(f"{int(s)}: {int(c):,}" for s, c in zip(unique_sids, sid_counts))
        )

    if max_muons > 0 and n > max_muons:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_muons, replace=False)
        arrs = tuple(a[idx] for a in arrs)
        print(f"subsampled to {max_muons} muons for training")

    return arrs


def compute_targets_and_conditioning(
    eta_r, phi_r, eta_g, phi_g, kappa_r, kappa_g, muon_source=None,
):
    """Return (target [N,3], cond_raw dict).

    ``kappa_reco`` and ``kappa_gen`` are stored directly in the
    snapshot, so ``r_kappa = kappa_reco / kappa_gen - 1`` is a single
    divide here. Charge mismeasurement (sign flip between reco and
    gen) appears as ``r_kappa`` near ``-2``, a mode the flow learns
    explicitly. Conditioning's ``charge`` is reconstructed as
    ``sign(kappa_gen)``; ``log_pt_gen`` is reconstructed from
    ``-log(|kappa_gen| * cosh(eta_gen))``.

    ``muon_source`` (optional) is the per-muon class integer written
    by the snapshot script: ``1`` (W/Z prompt), ``15`` (W/Z τ-decay)
    or ``443`` (J/ψ). When supplied it joins the cond vector as a
    single float feature; the standardisation in :func:`build_preproc`
    centres + scales it like the other cond entries. Pass ``None`` to
    keep the legacy 5-feature conditioning.
    """
    lam_r = np.arctan(np.sinh(eta_r))
    lam_g = np.arctan(np.sinh(eta_g))

    r_kappa = kappa_r / kappa_g - 1.0

    # Wrap dphi to [-pi, pi]
    dphi = np.arctan2(
        np.sin(phi_r - phi_g), np.cos(phi_r - phi_g)
    )
    dlambda = lam_r - lam_g

    # Target order is (r_kappa, dlambda, dphi) so that the reco
    # coordinates (kappa, lambda, phi) map index-for-index onto the
    # flow targets, making the reco-space Jacobian for downstream
    # weight computations a simple diag(1/kappa_gen, 1, 1).
    target = np.stack([r_kappa, dlambda, dphi], axis=1).astype(np.float32)

    log_pt_gen = -np.log(np.fabs(kappa_g) * np.cosh(eta_g))

    cond_raw = {
        "log_pt_gen": log_pt_gen.astype(np.float32),
        # ``charge`` stays ±1 and is left un-standardised downstream
        # (see _PASSTHROUGH_COND_FEATURES in build_preproc).
        "charge": np.sign(kappa_g).astype(np.float32),
        "lambda_gen": lam_g.astype(np.float32),
        "sin_phi_gen": np.sin(phi_g).astype(np.float32),
        "cos_phi_gen": np.cos(phi_g).astype(np.float32),
    }
    if muon_source is not None:
        # Compact 3-class encoding into {-1, 0, +1}. Raw values
        # {1, 15, 443} would otherwise dominate ``apply_preproc``'s
        # mean/std and collapse the prompt vs τ-decay distinction.
        cond_raw["muon_source"] = _muon_source_to_compact(muon_source)

    return target, cond_raw


# Per-class mapping for the per-muon ``muon_source`` integer:
#   1   -> -1   (W/Z prompt muon)
#   15  ->  0   (W/Z secondary muon from a τ decay)
#   443 -> +1   (J/ψ -- PDG id sentinel; calibration ntuples have no
#                ``Muon_genPartFlav`` analogue)
# Unrecognised values map to NaN to surface bugs loudly.
_MUON_SOURCE_CODES = {1: -1.0, 15: 0.0, 443: 1.0}


def _muon_source_to_compact(ms):
    """Map raw ``muon_source`` integers to ``float32`` codes in
    ``{-1, 0, +1}``; see :data:`_MUON_SOURCE_CODES`."""
    ms_arr = np.asarray(ms)
    out = np.full(ms_arr.shape, np.nan, dtype=np.float32)
    for raw, code in _MUON_SOURCE_CODES.items():
        out[ms_arr == raw] = code
    return out


# Conditioning features that should bypass the per-feature ``(x-μ)/σ``
# standardisation in ``apply_preproc``. ``charge`` is already a clean
# ±1 binary; ``muon_source`` is the compact 3-class code from above.
# Forcing their PreprocStats entries to ``(0, 1)`` makes the
# standardisation a no-op without special-casing the consumer.
_PASSTHROUGH_COND_FEATURES = ("charge", "muon_source")


# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------

@dataclass
class PreprocStats:
    target_names: List[str]
    target_mean: List[float]
    target_std: List[float]

    cond_names: List[str]     # order matters: matches flow input order
    cond_mean: List[float]
    cond_std: List[float]


def build_preproc(target: np.ndarray, cond_raw: dict) -> PreprocStats:
    target_names = ["r_kappa", "dlambda", "dphi"]
    target_mean = target.mean(axis=0).tolist()
    target_std = target.std(axis=0).tolist()

    cond_names = [
        "log_pt_gen",
        "charge",
        "lambda_gen",
        "sin_phi_gen",
        "cos_phi_gen",
    ]
    # ``muon_source`` is opt-in: present only when the snapshot+loader
    # carry the column. Append at the tail so older PreprocStats files
    # (5-feature cond) remain a strict prefix.
    if "muon_source" in cond_raw:
        cond_names.append("muon_source")
    cond_mean, cond_std = [], []
    for name in cond_names:
        arr = cond_raw[name]
        if name in _PASSTHROUGH_COND_FEATURES:
            cond_mean.append(0.0)
            cond_std.append(1.0)
        else:
            cond_mean.append(float(arr.mean()))
            cond_std.append(
                float(arr.std()) if arr.std() > 1e-6 else 1.0
            )

    return PreprocStats(
        target_names=target_names,
        target_mean=target_mean,
        target_std=target_std,
        cond_names=cond_names,
        cond_mean=cond_mean,
        cond_std=cond_std,
    )


def apply_preproc(
    target: np.ndarray, cond_raw: dict, stats: PreprocStats
) -> Tuple[np.ndarray, np.ndarray]:
    tmean = np.asarray(stats.target_mean, dtype=np.float32)
    tstd = np.asarray(stats.target_std, dtype=np.float32)
    target_std = (target - tmean) / tstd

    cond_cols = []
    for name, mean, std in zip(stats.cond_names, stats.cond_mean, stats.cond_std):
        cond_cols.append((cond_raw[name] - mean) / std)
    cond = np.stack(cond_cols, axis=1).astype(np.float32)

    return target_std, cond


# -----------------------------------------------------------------------------
# Flow
# -----------------------------------------------------------------------------

ACTIVATIONS = {
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


def _build_glow(
    n_features: int,
    n_cond: int,
    n_transforms: int,
    randmask: bool,
    hidden_features: int,
    n_hidden_layers: int,
    activation: type,
):
    """Minimal Glow-like flow assembled from zuko primitives.

    Matches RealNVP's affine-coupling structure, but inserts a
    learnable invertible LU-parameterized linear mixing layer
    (``LULinearTransform``) *before* each coupling. At d=3 each
    LU layer has 9 learnable parameters — negligible on top of
    the coupling MLPs but lets the flow learn the feature mixing
    rather than relying on fixed alternating checkered masks.

    Initialization: LU = I + small random perturbation in the
    off-diagonal entries, so each LU layer starts as (near-)
    identity and the flow is effectively RealNVP at step 0. The
    learned mixing is acquired during training.

    Stability: zuko's ``LULinearTransform`` does not clamp the
    diagonal of L, so in principle a diagonal entry could reach
    zero and make the layer singular. In practice, with unit-ish
    initialization and modest learning rate, this never happens
    for this problem. If it does, reduce ``--lr``.
    """
    import zuko
    from zuko.flows import (
        Flow, GeneralCouplingTransform, UnconditionalTransform,
        UnconditionalDistribution,
    )
    from zuko.transforms import LULinearTransform
    from zuko.distributions import DiagNormal

    # Reproducible per-call initialization of the LU perturbation
    # and (optionally) the random masks. Not a model seed — model
    # weights are seeded by torch's global RNG as usual.
    rng = torch.Generator().manual_seed(0)

    transforms_list = []
    for i in range(n_transforms):
        # LU linear mixing, initialized near-identity.
        LU_init = torch.eye(n_features, dtype=torch.float32)
        noise = 0.05 * torch.randn(
            n_features, n_features, generator=rng,
        )
        # Perturb only the off-diagonal entries; keep diagonal
        # near 1 so log|det| starts finite.
        LU_init = LU_init + noise.tril(-1) + noise.triu(1)
        transforms_list.append(
            UnconditionalTransform(
                LULinearTransform, LU_init, buffer=False,
            )
        )

        # Affine coupling (same as RealNVP).
        if randmask:
            mask = (
                torch.randperm(n_features, generator=rng) % 2
                == i % 2
            )
        else:
            mask = torch.arange(n_features) % 2 == i % 2
        transforms_list.append(
            GeneralCouplingTransform(
                features=n_features,
                context=n_cond,
                mask=mask,
                hidden_features=[hidden_features] * n_hidden_layers,
                activation=activation,
            )
        )

    base = UnconditionalDistribution(
        DiagNormal,
        loc=torch.zeros(n_features),
        scale=torch.ones(n_features),
        buffer=True,
    )
    return Flow(transforms_list, base)


def _build_sospf(
    n_features: int,
    n_cond: int,
    n_transforms: int,
    hidden_features: int,
    n_hidden_layers: int,
    activation,
    degree: int,
    polynomials: int,
    quad_n: int,
):
    """Build a sum-of-squares polynomial flow with explicit Gauss–Legendre
    quadrature node count.

    Reproduces ``zuko.flows.SOSPF`` (Jaini et al., 2019) but overrides the
    quadrature ``n`` used by the inner ``SOSPolynomialTransform``. zuko's
    default is ``n = degree + 1`` (the minimum value that integrates the
    degree-``2·degree`` polynomial integrand exactly under Gauss–Legendre's
    polynomial-exactness rule). This wrapper accepts any ``quad_n >= 1`` —
    set lower for faster but biased quadrature, higher for defensive
    overshoot.

    Built directly via ``zuko.flows.MAF`` with a custom ``univariate``
    factory that closes over ``quad_n`` plus the SOSPF post-init pass
    that interleaves ``SoftclipTransform(bound=11)`` between successive
    masked-autoregressive transforms (keeps inputs inside the
    ``[-10, 10]`` SOSPolynomialTransform invertibility window).
    """
    import zuko
    from zuko.flows import MAF, UnconditionalTransform
    from zuko.transforms import (
        ComposedTransform,
        AdditiveTransform,
        SOSPolynomialTransform,
        SoftclipTransform,
        UnconstrainedMonotonicTransform,
    )

    class _SOSPolyTQN(SOSPolynomialTransform):
        """SOSPolynomialTransform with overridable quadrature n.

        zuko's parent ``__init__`` hardcodes ``n=a.shape[-1]`` when
        chaining to ``UnconstrainedMonotonicTransform``; we bypass it
        by calling the grandparent constructor directly with our n.
        """

        def __init__(self, a, slope: float = 1e-3, **kwargs):
            UnconstrainedMonotonicTransform.__init__(
                self, None, phi=(a,), n=int(quad_n), **kwargs,
            )
            self.a = a
            self.i = torch.arange(a.shape[-1], device=a.device)
            self.slope = slope

    def _shifted_sosp(a, constant):
        return ComposedTransform(
            _SOSPolyTQN(a=a),
            AdditiveTransform(shift=constant),
        )

    flow = MAF(
        features=n_features,
        context=n_cond,
        transforms=n_transforms,
        univariate=_shifted_sosp,
        shapes=[(polynomials, degree + 1), ()],
        hidden_features=[hidden_features] * n_hidden_layers,
        activation=activation,
    )

    # SOSPF post-init: interleave a Softclip between successive MAF
    # transforms to keep the inputs to each SOSPolynomialTransform
    # within its [-10, 10] invertibility window. Matches zuko's SOSPF
    # exactly (bound=11.0).
    transforms = flow.transform.transforms
    for i in reversed(range(1, len(transforms))):
        transforms.insert(
            i, UnconditionalTransform(SoftclipTransform, bound=11.0),
        )
    return flow


def build_flow(
    n_features: int,
    n_cond: int,
    n_transforms: int,
    hidden_features: int,
    n_hidden_layers: int,
    activation: str = "gelu",
    architecture: str = "realnvp",
    randmask: bool = False,
    gf_components: int = 8,
    sospf_degree: int = 4,
    sospf_polynomials: int = 3,
    sospf_quad_n: int = -1,
) -> nn.Module:
    """Build a conditional flow via zuko. Architecture-dispatched.

    ``architecture="realnvp"`` (default):
        RealNVP-style affine coupling. Each of ``n_transforms``
        coupling layers splits y under a checkered mask and applies
        a *monotonic affine* ``y_masked → y_masked · exp(s) + t``
        where ``(s, t)`` are the outputs of a conditioner MLP on
        the (unmasked half, context). With smooth conditioner
        activations (GELU default), the log-abs-det Jacobian — and
        therefore the flow's log-density — is C^∞ in y.
        ``randmask`` (default False) uses random coupling masks
        instead of the standard alternating checkered masks.

    ``architecture="glow"``:
        Glow-like (Kingma & Dhariwal 2018) — RealNVP plus a
        learnable invertible LU-parameterized linear mixing layer
        (``LULinearTransform``) before each coupling. Replaces
        the fixed alternating-mask mixing with a learned
        per-layer rotation, typically giving comparable NLL at
        fewer coupling layers. Minimal build: identity-init LU
        with small noise, no actnorm (the data preproc already
        standardizes). Adds d² learnable parameters per LU
        layer — negligible at d = 3.

    ``architecture="maf"``:
        Masked Autoregressive Flow (Papamakarios et al., 2017).
        Same affine ``y_d → y_d · exp(s_d) + t_d`` transform as
        RealNVP, but conditioned autoregressively — each ``(s_d,
        t_d)`` depends on ``y_<d`` (and the context) via a single
        masked MLP (MADE) per layer instead of on a fixed half via
        a coupling MLP. All F features get warped every layer,
        which typically gives better NLL than RealNVP at fewer
        transforms, at the cost of expensive sequential
        sampling. Forward (NLL) cost ≈ RealNVP. Closed-form
        Jacobian, ``torch.compile``-safe.

    ``architecture="gf"``:
        Gaussianization Flow (Meng et al., 2020). Each layer is a
        context-conditional per-dimension monotonic transform
        (Gaussian-mixture CDF with ``gf_components`` components),
        interleaved with fixed random orthogonal rotations that mix
        features between layers. Typically more parameter-efficient
        than RealNVP for near-Gaussian-shaped conditional densities
        and also C^∞ with smooth conditioner activations.
        Invertibility is only guaranteed on ``[-10, 10]`` in
        standardized inputs — preproc already gives zero-mean/unit-
        std targets, so this is respected.

    ``architecture="sospf"``:
        Sum-of-Squares Polynomial Flow (Jaini et al., 2019). Masked
        autoregressive stack where each transform is the integral of
        a sum of ``sospf_polynomials`` squared polynomials of degree
        ``sospf_degree``: ``f(x) = ∫₀^x (1/K) Σᵢ (1 + Σⱼ aᵢⱼ uʲ)² du``.
        Closed-form analytic Jacobian (no inner ``autograd.grad`` —
        torch.compile-safe), and structurally invertible across the
        standardised range ``[−10, 10]``. Strong inductive bias for
        smooth one-dimensional warpings.

    ``hidden_features`` × ``n_hidden_layers`` sizes each
    conditioner MLP in all four architectures.
    """
    try:
        import zuko
    except ImportError as e:
        raise ImportError(
            "zuko is required for training (pip install zuko)."
        ) from e

    act_cls = ACTIVATIONS.get(activation.lower())
    if act_cls is None:
        raise ValueError(
            f"unknown activation '{activation}'; "
            f"available: {sorted(ACTIVATIONS)}"
        )

    arch = architecture.lower()
    if arch == "realnvp":
        return zuko.flows.RealNVP(
            features=n_features,
            context=n_cond,
            transforms=n_transforms,
            randmask=randmask,
            hidden_features=[hidden_features] * n_hidden_layers,
            activation=act_cls,
        )
    if arch == "maf":
        return zuko.flows.MAF(
            features=n_features,
            context=n_cond,
            transforms=n_transforms,
            hidden_features=[hidden_features] * n_hidden_layers,
            activation=act_cls,
        )
    if arch == "glow":
        return _build_glow(
            n_features=n_features,
            n_cond=n_cond,
            n_transforms=n_transforms,
            randmask=randmask,
            hidden_features=hidden_features,
            n_hidden_layers=n_hidden_layers,
            activation=act_cls,
        )
    if arch == "gf":
        return zuko.flows.GF(
            features=n_features,
            context=n_cond,
            transforms=n_transforms,
            components=gf_components,
            hidden_features=[hidden_features] * n_hidden_layers,
            activation=act_cls,
        )
    if arch == "sospf":
        # ``sospf_quad_n <= 0`` → auto: use the minimum exact n for
        # the degree-2*degree polynomial integrand under Gauss–Legendre
        # (n=L+1, matching zuko's own default).
        _quad_n = (
            int(sospf_quad_n)
            if sospf_quad_n is not None and int(sospf_quad_n) > 0
            else int(sospf_degree) + 1
        )
        return _build_sospf(
            n_features=n_features,
            n_cond=n_cond,
            n_transforms=n_transforms,
            hidden_features=hidden_features,
            n_hidden_layers=n_hidden_layers,
            activation=act_cls,
            degree=sospf_degree,
            polynomials=sospf_polynomials,
            quad_n=_quad_n,
        )
    raise ValueError(
        f"unknown architecture '{architecture}'; "
        f"available: 'realnvp', 'glow', 'maf', 'gf', 'sospf'"
    )


# -----------------------------------------------------------------------------
# Polynomial-correction head (joint training, joint polynomial)
#
# Perturbations live in **target space** (y), not conditioning space:
# the calibration shifts and smears we want to model are reco-level
# (e.g., a momentum-scale shift on pt_reco changes r_kappa), while
# c is purely gen-level. So the polyhead predicts the per-event
# weight for a y-space perturbation:
#
#     w(u_y) = p(y - u_y | c) / p(y | c)
#
# at fixed c. The head's output is, per event and as a function of
# (y_std, c_std):
#
#     joint_coefs    shape [B, n_joint_basis]
#                    polynomial in (u, σ_vec) ∈ R^{n_features} × R^{n_features},
#                    even in σ_vec, no constant term, with degree caps
#                    (max_deg_u, max_deg_sigma, max_cross_deg).
#
# At evaluation:
#
#     W_pred(u_shift, σ_vec) = softplus( joint(u_shift, σ_vec) ) / log 2
#
# The ``softplus(·)/log 2`` rectifier maps the (unrestricted-sign)
# joint polynomial to a strictly-positive multiplier with
# ``softplus(0)/log 2 = 1`` so ``W_pred(0, 0) = 1`` is exact.
#
# All perturbation vectors are in **standardized target space** —
# divide by ``target_std`` to convert from raw target units. The
# downstream caller forms ``u_shift_std = u_shift_raw / target_std``
# and ``σ_vec_std = σ_vec_raw / target_std``.
#
# Reductions:
#   * pure shift   (σ_vec = 0): joint(u, 0) is a polynomial in u alone.
#   * pure smear   (u_shift = 0): joint(0, σ_vec) is even in σ_vec.
#   * shift+smear  (both ≠ 0): the full joint polynomial.
#
# The joint polynomial enforces:
#   * no constant term  ⇒ W = softplus(0)/log 2 = 1 at (0, 0)
#   * even in σ_vec     ⇒ smear-symmetry under σ_vec → -σ_vec
#   * degree caps       ⇒ bounded number of head outputs
#
# The polyhead's prediction has zero gradient path to the flow params
# (no z in the formula), so flow training is decoupled from polyhead
# training even when both losses are active.
#
# This block is self-contained so the script runs without sibling
# files. The inference wrapper (FlowPolyheadInference, used at AOTI
# export) lives in flow_polyhead.py and re-imports these symbols.
# -----------------------------------------------------------------------------

import itertools as _itertools
import math as _math


_LOG_LOG2 = _math.log(_math.log(2.0))
_LOG_W_CLAMP = 30.0   # exp-positivity clamp on log W to keep fp32 finite


# ============================================================================
# Positivity wrap (shared between predicted_W / predicted_log_W and the loss
# steps). Each wrap is applied to the raw joint-polynomial scalar ``j`` and
# returns either ``W`` (positive) or ``log W`` directly — no ``log(exp(·))``
# or ``exp(log(·))`` round trips, and no protection beyond the minimum each
# wrap intrinsically needs.
#
#   * ``"exp"``      — log W = j (clamped ±_LOG_W_CLAMP); W = exp(log W).
#                      The clamp is the ONLY protection here and is
#                      intrinsic to ``exp`` overflow in fp32.
#   * ``"softplus"`` — W = softplus(j) / log 2; log W = log(W). For log W
#                      we floor ``softplus(j)`` to the dtype's smallest
#                      normal so ``log`` stays finite when ``j ≪ 0``
#                      makes softplus underflow to 0. No input clamp.
#   * ``"asinh"``    — log W = asinh(j); W = j + √(j² + 1) = exp(asinh(j)).
#                      Closed form, strictly positive for any finite j
#                      (``√(j²+1) > |j|``), grows only logarithmically in
#                      |j| in log-space. Needs *no* clamp at all.
# ============================================================================

def _apply_positivity_W(j: torch.Tensor, positivity: str) -> torch.Tensor:
    """Return ``W(j)`` directly, without going through ``log W``."""
    if positivity == "exp":
        return torch.exp(j.clamp(min=-_LOG_W_CLAMP, max=_LOG_W_CLAMP))
    if positivity == "softplus":
        return F.softplus(j) / _math.log(2.0)
    if positivity == "asinh":
        # Stable closed form for ``W = exp(asinh(j))``:
        #   half = |j| + √(j²+1)   (always > 1, no cancellation)
        # then ``W = half`` for ``j ≥ 0`` and ``W = 1/half`` for
        # ``j < 0``. Algebraically identical to ``j + √(j²+1)`` but
        # avoids fp catastrophic cancellation when ``j ≪ 0`` makes
        # ``√(j²+1) ≈ |j|`` round to ``-j`` exactly.
        half = j.abs() + torch.sqrt(j * j + 1.0)
        return torch.where(j >= 0, half, 1.0 / half)
    raise ValueError(f"unknown positivity {positivity!r}")


def _apply_positivity_logW(
    j: torch.Tensor,
    positivity: str,
    clamp: float = _LOG_W_CLAMP,
) -> torch.Tensor:
    """Return ``log W(j)`` directly, without going through ``W``.

    ``clamp`` only affects the ``"exp"`` branch (the other wraps are
    intrinsically bounded). ``clamp=inf`` (or any non-finite value)
    disables the input clamp — appropriate for losses that don't
    take a downstream ``exp`` of ``log W`` and so don't need the
    fp32-overflow safety net (e.g., :func:`_wbce_per_event`, which
    only feeds ``log W`` into ``softplus``)."""
    if positivity == "exp":
        if not _math.isfinite(clamp):
            return j
        return j.clamp(min=-clamp, max=clamp)
    if positivity == "softplus":
        tiny = torch.finfo(j.dtype).tiny
        return torch.log(F.softplus(j).clamp_min(tiny)) - _LOG_LOG2
    if positivity == "asinh":
        return torch.asinh(j)
    raise ValueError(f"unknown positivity {positivity!r}")


def _joint_indices(
    n_vars: int,
    max_deg_u: int,
    max_deg_sigma: int,
    max_cross_deg: int,
) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Build (α, β) multi-index pairs for the joint polynomial.

    Constraints:
      * 1 ≤ |α| + |β|              (no constant term)
      * |β| even                   (smear-symmetry: σ_vec → -σ_vec)
      * if |β| = 0:  1 ≤ |α| ≤ max_deg_u                  (pure-shift slice)
      * if |α| = 0:  2 ≤ |β| ≤ max_deg_sigma  even         (pure-smear slice)
      * else (cross): 1 ≤ |α| ≤ max_deg_u, 2 ≤ |β| ≤ max_deg_sigma even,
                       and |α| + |β| ≤ max_cross_deg
    """
    out: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
    rng = list(range(n_vars))
    # pure-u slice (β = ()): degrees 1..max_deg_u in α.
    for d_a in range(1, max_deg_u + 1):
        for alpha in _itertools.combinations_with_replacement(rng, d_a):
            out.append((alpha, ()))
    # pure-σ slice (α = ()): even degrees 2..max_deg_sigma in β.
    for d_b in range(2, max_deg_sigma + 1, 2):
        for beta in _itertools.combinations_with_replacement(rng, d_b):
            out.append(((), beta))
    # cross terms: |α| ≥ 1, |β| ≥ 2 even, |α|+|β| ≤ max_cross_deg.
    for d_a in range(1, max_deg_u + 1):
        for d_b in range(2, max_deg_sigma + 1, 2):
            if d_a + d_b > max_cross_deg:
                continue
            for alpha in _itertools.combinations_with_replacement(rng, d_a):
                for beta in _itertools.combinations_with_replacement(rng, d_b):
                    out.append((alpha, beta))
    return out


# Module-level cache for the auxiliary tensors needed by the
# vectorized basis functions. Keyed by ``(id(indices), n_features,
# device-string)``. ``indices`` is the polyhead's joint-multi-index
# list, which is created once at polyhead-init time and reused
# throughout training, so id-based caching is safe in practice — it
# only "stales" if the indices object is GC'd and a new list ends up
# at the same memory address, which doesn't happen during a normal
# polyhead lifetime.
def _run_profile_pass(step_fn, n_warmup: int, n_active: int,
                      output_path: str = None, label: str = "",
                      row_limit: int = 30):
    """Run ``step_fn(i)`` for ``n_warmup + n_active`` iterations under
    ``torch.profiler``; print the top-``row_limit`` hot ops sorted by
    self CUDA time (or self CPU time when CUDA isn't available).
    Optionally export a Chrome trace at ``output_path`` for
    visualization in chrome://tracing or Perfetto.

    The profiler schedule is ``wait=0, warmup=n_warmup, active=n_active,
    repeat=1`` — one warmup window followed by one active window.
    Profiling overhead is non-negligible (graph recording, op stats),
    so this is not meant to drive normal training; it's a one-off
    diagnostic pass that the caller invokes from a ``--profile``
    branch and then exits.
    """
    from torch.profiler import profile, schedule, ProfilerActivity
    activities = [ProfilerActivity.CPU]
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        activities.append(ProfilerActivity.CUDA)
    n_total = int(n_warmup) + int(n_active)
    sched = schedule(
        wait=0, warmup=int(n_warmup), active=int(n_active), repeat=1,
    )
    print(
        f"[profile{label}] running {n_total} steps "
        f"({n_warmup} warmup + {n_active} active); device={'cuda' if cuda_ok else 'cpu'}",
        flush=True,
    )
    with profile(activities=activities, schedule=sched, record_shapes=False) as prof:
        for i in range(n_total):
            step_fn(i)
            prof.step()
    sort_key = "self_cuda_time_total" if cuda_ok else "self_cpu_time_total"
    print(
        f"[profile{label}] top-{row_limit} ops by {sort_key}:",
        flush=True,
    )
    print(
        prof.key_averages().table(sort_by=sort_key, row_limit=row_limit),
        flush=True,
    )
    if output_path:
        prof.export_chrome_trace(output_path)
        print(
            f"[profile{label}] chrome trace exported: {output_path}",
            flush=True,
        )


_basis_aux_cache: dict = {}


def _build_basis_aux(indices, n_features: int):
    """Build basis-evaluation auxiliary tensors on **CPU** from the
    multi-index list. Returns a dict identical in structure to
    :func:`_get_basis_aux`'s output, with tensors on CPU.

    Used by polyhead ``__init__`` to register these tensors as
    ``nn.Module`` buffers, which then move with ``.to(device)`` and
    are seen as stable inputs by ``torch.compile`` / CUDA graphs.

    The returned values:

      * ``alpha_degs`` / ``beta_degs``: ``[n_basis, n_features]``
        LongTensor of per-axis degrees.
      * ``cheb_u_const`` / ``cheb_v_const``: ``[n_basis]`` float64
        tensors with the per-side origin constants
        ``Π_j T_{α_j}(0)`` (resp. β-side); zeroed on the empty side
        so each multi-index drops to its singly-centered limit
        (pure-u or pure-σ).
      * ``max_deg_u`` / ``max_deg_sigma``: int.
    """
    alpha_list = [
        _multiindex_to_axis_degrees(a, n_features) for a, _ in indices
    ]
    beta_list = [
        _multiindex_to_axis_degrees(b, n_features) for _, b in indices
    ]
    if alpha_list:
        alpha_degs = torch.tensor(alpha_list, dtype=torch.long)
    else:
        alpha_degs = torch.zeros(0, n_features, dtype=torch.long)
    if beta_list:
        beta_degs = torch.tensor(beta_list, dtype=torch.long)
    else:
        beta_degs = torch.zeros(0, n_features, dtype=torch.long)
    # Per-side origin constants for the doubly-centered chebyshev
    # factorization phi[k] = (Π T_α(u_norm) − u_const[k]) ·
    # (Π T_β(σ_norm) − v_const[k]), where the subtraction is enabled
    # only when the corresponding multi-index side is non-empty. This
    # ensures cross terms (both α, β nonempty) vanish at u=0 OR σ=0
    # separately — without per-side centering the cross block leaks
    # a u-dependent offset at σ=0 (and σ-dependent at u=0), which
    # violates the structural prior log W(u=0, σ=0) = 0 along each
    # axis and corrupts axis-aligned closure plots.
    u_const_list = []
    v_const_list = []
    for a_d, b_d in zip(alpha_list, beta_list):
        if any(d > 0 for d in a_d):
            cu = 1.0
            for d in a_d:
                if d > 0:
                    cu *= _T_at_zero(d)
                    if cu == 0.0:
                        break
        else:
            cu = 0.0
        if any(d > 0 for d in b_d):
            cv = 1.0
            for d in b_d:
                if d > 0:
                    cv *= _T_at_zero(d)
                    if cv == 0.0:
                        break
        else:
            cv = 0.0
        u_const_list.append(cu)
        v_const_list.append(cv)
    cheb_u_const = torch.tensor(u_const_list, dtype=torch.float64)
    cheb_v_const = torch.tensor(v_const_list, dtype=torch.float64)
    max_du = int(alpha_degs.max()) if alpha_degs.numel() else 0
    max_ds = int(beta_degs.max()) if beta_degs.numel() else 0
    return {
        "alpha_degs": alpha_degs,
        "beta_degs": beta_degs,
        "cheb_u_const": cheb_u_const,
        "cheb_v_const": cheb_v_const,
        "max_deg_u": max_du,
        "max_deg_sigma": max_ds,
    }


def _get_basis_aux(indices, n_features: int, device: torch.device):
    """Return cached auxiliary tensors for the vectorized basis
    functions, computing them once per (indices, n_features, device).
    Used by external callers (diagnostics, exports) where the
    ``torch.compile`` / CUDA-graph constraints don't apply. Inside a
    compiled module's forward, prefer passing pre-registered buffers
    via ``basis_aux=`` to :func:`evaluate_joint` instead — this
    avoids creating tensors inside the compile path (which CUDA
    graphs interprets as captured outputs and then errors on
    overwrite).
    """
    key = (id(indices), n_features, str(device))
    if key in _basis_aux_cache:
        return _basis_aux_cache[key]
    cpu_aux = _build_basis_aux(indices, n_features)
    out = {
        "alpha_degs": cpu_aux["alpha_degs"].to(device),
        "beta_degs": cpu_aux["beta_degs"].to(device),
        "cheb_u_const": cpu_aux["cheb_u_const"].to(device),
        "cheb_v_const": cpu_aux["cheb_v_const"].to(device),
        "max_deg_u": cpu_aux["max_deg_u"],
        "max_deg_sigma": cpu_aux["max_deg_sigma"],
    }
    _basis_aux_cache[key] = out
    return out


def _power_table(x: torch.Tensor, max_deg: int) -> torch.Tensor:
    """Return ``[..., n_features, max_deg + 1]`` power table where
    ``out[..., j, k] = x[..., j]^k`` for k = 0..max_deg. Built in
    O(max_deg) tensor ops via repeated multiplication (vs. a Python
    loop of O(n_basis) ops in the legacy basis evaluation, which is
    the source of the polyhead-order training slowdown on GPU)."""
    if max_deg == 0:
        return torch.ones_like(x).unsqueeze(-1)
    cols = [torch.ones_like(x), x]
    for k in range(2, max_deg + 1):
        cols.append(cols[-1] * x)
    return torch.stack(cols, dim=-1)


def joint_monomial_basis(
    u: torch.Tensor,
    sigma_vec: torch.Tensor,
    indices,
    *,
    aux=None,
) -> torch.Tensor:
    """Evaluate the joint (α, β) basis at ``(u, σ_vec)`` — vectorized.

    Each entry contributes ``u^α · σ_vec^β`` (product of selected
    components). Output shape: ``[..., len(indices)]``, identical to
    the previous loop-over-multi-indices reference.

    If ``aux`` (a dict with ``alpha_degs``, ``beta_degs``,
    ``max_deg_u``, ``max_deg_sigma``) is provided, those tensors /
    ints are used directly — no cache lookup. This is required when
    called from a ``torch.compile``-wrapped module: register the aux
    tensors as buffers on the polyhead and pass ``aux=polyhead
    .basis_aux`` so the compiled forward sees them as stable inputs
    rather than allocating ephemeral tensors that CUDA graphs
    interpret as captured outputs.

    Reduces autograd-graph node count from ``O(n_basis · n_features)``
    in the legacy loop to ``O(max_deg_u + max_deg_sigma + n_features)``
    — the dominant per-call cost on GPU due to kernel-launch overhead.
    """
    if aux is None:
        n_features = u.shape[-1]
        aux = _get_basis_aux(indices, n_features, u.device)
    alpha_degs = aux["alpha_degs"]
    beta_degs = aux["beta_degs"]
    max_du = aux["max_deg_u"]
    max_ds = aux["max_deg_sigma"]
    n_features = u.shape[-1]
    u_pow = _power_table(u, max_du)
    has_sigma = max_ds > 0
    if has_sigma:
        s_pow = _power_table(sigma_vec, max_ds)
    phi = None
    for j in range(n_features):
        u_j = u_pow[..., j, :].index_select(
            dim=-1, index=alpha_degs[:, j],
        )
        if has_sigma:
            s_j = s_pow[..., j, :].index_select(
                dim=-1, index=beta_degs[:, j],
            )
            factor_j = u_j * s_j
        else:
            factor_j = u_j
        phi = factor_j if phi is None else phi * factor_j
    return phi


def _multiindex_to_axis_degrees(idx, n_features):
    """Convert a sorted-tuple multi-index (e.g. (0, 0, 1)) into a list
    of per-axis degrees (e.g. [2, 1, 0] for n_features=3)."""
    degs = [0] * n_features
    for i in idx:
        degs[i] += 1
    return degs


def _cheb_table(x_norm: torch.Tensor, max_deg: int) -> torch.Tensor:
    """First-kind Chebyshev polynomials T_0..T_max_deg of ``x_norm``,
    via the standard recurrence T_n(x) = 2 x T_{n-1}(x) − T_{n-2}(x).
    Returns shape ``[..., max_deg + 1]``. ``max_deg < 0`` returns
    an empty trailing dim."""
    if max_deg < 0:
        return x_norm.unsqueeze(-1)[..., :0]
    cols = [torch.ones_like(x_norm)]   # T_0 = 1
    if max_deg >= 1:
        cols.append(x_norm)            # T_1 = x
    for n in range(2, max_deg + 1):
        cols.append(2 * x_norm * cols[-1] - cols[-2])
    return torch.stack(cols, dim=-1)


def _T_at_zero(n: int) -> float:
    """Value of the n-th first-kind Chebyshev polynomial at x=0:
       T_0(0)=1, T_n(0)=0 for odd n, T_n(0)=(-1)^(n/2) for even n>0.
    """
    if n == 0:
        return 1.0
    if n % 2 == 1:
        return 0.0
    return (-1.0) ** (n // 2)


def joint_chebyshev_basis(
    u: torch.Tensor,
    sigma_vec: torch.Tensor,
    indices,
    scale_u: float = 1.0,
    scale_sigma: float = 1.0,
    *,
    aux=None,
) -> torch.Tensor:
    """Evaluate the joint (α, β) basis at ``(u, σ_vec)`` using
    *tensor-product Chebyshev* polynomials with **per-side
    zero-anchoring**. Each entry is

        (Π_j T_{d_α_j}(u_j / scale_u)   − u_const[k])
            · (Π_j T_{d_β_j}(σ_j / scale_sigma) − v_const[k])

    where ``u_const[k] = Π_j T_{d_α_j}(0)`` if ``α`` is non-empty
    else ``0``, and similarly for ``v_const[k]`` (β-side).

    Per-side anchoring (vs a single joint-origin subtraction) is
    essential to preserve the structural priors that the monomial
    basis enjoyed automatically:

      * pure-u (β = ∅): φ vanishes at u = 0;
      * pure-σ (α = ∅): φ vanishes at σ = 0;
      * cross  (α, β both nonempty): φ vanishes at u = 0 *or* σ = 0.

    A single joint-origin subtraction (`Π T_α(0)·Π T_β(0)`) leaves a
    nonzero residual ``T_β(0) · (Π T_α(u/scale) − Π T_α(0))`` for
    cross terms at σ = 0 (and the analogous u → 0 leak), which
    cross-trained coefficients then absorb into joint-mode events
    and leak back into shift / smear closure. With per-side
    anchoring, cross factors vanish independently, so SHIFT-mode
    events see d = c_pu·U(α; u) and SMEAR-mode events see
    d = c_ps·V(β; σ) — matching the per-mode losses.

    ``scale_u`` and ``scale_sigma`` map the trained perturbation
    ranges to ``[-1, 1]`` (typical choice: ``oversample · delta_max``
    and ``oversample · sigma_max``). Outside the trained range the
    Chebyshev polynomials grow rapidly — better fit-quality inside,
    less stable extrapolation outside, vs the monomial basis.

    Smear-symmetry (σ_vec → −σ_vec): preserved by the existing
    |β|-even constraint on multi-indices, since T_n is even iff n
    is even, and the per-side ``v_const`` subtraction is itself
    smear-symmetric.
    """
    if aux is None:
        n_features = u.shape[-1]
        aux = _get_basis_aux(indices, n_features, u.device)
    alpha_degs = aux["alpha_degs"]
    beta_degs = aux["beta_degs"]
    max_du = aux["max_deg_u"]
    max_ds = aux["max_deg_sigma"]
    u_const = aux["cheb_u_const"].to(dtype=u.dtype)
    v_const = aux["cheb_v_const"].to(dtype=u.dtype)
    n_features = u.shape[-1]
    u_norm = u / float(scale_u)
    sigma_norm = sigma_vec / float(scale_sigma)
    # Vectorized Chebyshev table on the full ``[..., n_features]``
    # input — single Clenshaw recurrence shared across all axes,
    # producing ``[..., n_features, max_deg + 1]``.
    Tu = _cheb_table(u_norm, max_du)
    has_sigma = max_ds > 0
    if has_sigma:
        Ts = _cheb_table(sigma_norm, max_ds)
    # Compute the un-anchored per-side products separately so we
    # can subtract the per-side origin constants before the final
    # cross multiplication.
    u_factor = None
    for j in range(n_features):
        u_j = Tu[..., j, :].index_select(
            dim=-1, index=alpha_degs[:, j],
        )
        u_factor = u_j if u_factor is None else u_factor * u_j
    if has_sigma:
        v_factor = None
        for j in range(n_features):
            s_j = Ts[..., j, :].index_select(
                dim=-1, index=beta_degs[:, j],
            )
            v_factor = s_j if v_factor is None else v_factor * s_j
        return (u_factor - u_const) * (v_factor - v_const)
    # Pure-u-only basis (no σ): pure-u case has β=∅ → v_const=0,
    # v_factor=1, so the cross-side factor reduces to 1.
    return u_factor - u_const


def _select_basis(
    basis: str,
    u: torch.Tensor,
    sigma_vec: torch.Tensor,
    indices,
    scale_u: float = 1.0,
    scale_sigma: float = 1.0,
    *,
    aux=None,
) -> torch.Tensor:
    """Dispatch to ``joint_monomial_basis`` or
    ``joint_chebyshev_basis`` based on ``basis`` name. ``aux`` is
    forwarded if provided (used for torch.compile-friendly polyhead
    paths to avoid in-graph cache lookups)."""
    if basis == "monomial":
        return joint_monomial_basis(u, sigma_vec, indices, aux=aux)
    if basis == "chebyshev":
        return joint_chebyshev_basis(
            u, sigma_vec, indices, scale_u, scale_sigma, aux=aux,
        )
    raise ValueError(f"unknown basis {basis!r}")


class PolyHead(nn.Module):
    """MLP from ``[y_std, c_std]`` to joint polynomial coefficients.

    Output:
      * ``joint_coefs`` shape ``[B, n_joint_basis]`` — scalar
        polynomial in ``(u, σ_vec) ∈ R^{n_features} × R^{n_features}``
        (target / y-space), with the parity / degree constraints
        enumerated by ``_joint_indices``.

    The final layer is initialized near zero so training starts with
    ``joint ≈ 0`` (i.e., predicted ``W ≈ softplus(0)/log 2 = 1``).
    """

    def __init__(
        self,
        n_features: int,
        n_cond: int,
        hidden_features: int = 64,
        n_hidden_layers: int = 2,
        max_deg_u: int = 3,
        max_deg_sigma: int = 4,
        max_cross_deg: int = 3,
        activation=nn.GELU,
        smear_K: int = 1,
        smear_residual: bool = False,
        include_smear: bool = False,
        positivity: str = "softplus",
        basis: str = "monomial",
        basis_scale_u: float = 1.0,
        basis_scale_sigma: float = 1.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_cond = n_cond
        self.include_smear = bool(include_smear)
        if basis not in ("monomial", "chebyshev"):
            raise ValueError(f"unknown basis {basis!r}")
        self.basis = str(basis)
        self.basis_scale_u = float(basis_scale_u)
        self.basis_scale_sigma = float(basis_scale_sigma)
        # Shift-only collapse: pure-u polynomial only, no σ basis,
        # K=1 stochastic smear path disabled. The smear-related CLI
        # values are silently overridden so the mode is internally
        # consistent end-to-end.
        if not self.include_smear:
            max_deg_sigma = 0
            max_cross_deg = 0
            smear_K = 1
            smear_residual = False
        self.max_deg_u = max_deg_u
        self.max_deg_sigma = max_deg_sigma
        self.max_cross_deg = max_cross_deg
        self.smear_K = int(smear_K)
        self.smear_residual = bool(smear_residual)
        if self.smear_residual and self.smear_K <= 1:
            raise ValueError(
                "smear_residual=True requires smear_K > 1 "
                "(K=1 stochastic has no GH base to correct)."
            )
        if positivity not in ("softplus", "exp", "asinh"):
            raise ValueError(f"unknown positivity {positivity!r}")
        self.positivity = str(positivity)
        # Polynomial basis on R^{n_features} — perturbations are in
        # target / y-space. Reorder so pure-u indices come first,
        # pure-σ second, cross last — this lets the three-head split
        # below produce contiguous coefficient blocks that align with
        # the basis ordering, and lets the per-mode loss-step
        # contraction read the right phi slice without scatter/
        # gather. Total set is unchanged.
        raw_indices = _joint_indices(
            n_features, max_deg_u, max_deg_sigma, max_cross_deg,
        )
        idx_pure_u = [
            (a, b) for (a, b) in raw_indices
            if len(b) == 0 and len(a) > 0
        ]
        idx_pure_s = [
            (a, b) for (a, b) in raw_indices
            if len(a) == 0 and len(b) > 0
        ]
        idx_cross = [
            (a, b) for (a, b) in raw_indices
            if len(a) > 0 and len(b) > 0
        ]
        assert (
            len(idx_pure_u) + len(idx_pure_s) + len(idx_cross)
            == len(raw_indices)
        ), "non-exhaustive split of joint indices"
        self._joint_indices = idx_pure_u + idx_pure_s + idx_cross
        self.n_joint_basis = len(self._joint_indices)
        self._n_pure_u = len(idx_pure_u)
        self._n_pure_s = len(idx_pure_s)
        self._n_cross = len(idx_cross)
        # Boolean masks over (reordered) basis indices: pure-u
        # (β = ()), pure-σ (α = ()), and cross (both nonempty). Now
        # contiguous blocks: [0, n_pure_u), [n_pure_u, n_pure_u +
        # n_pure_s), [n_pure_u + n_pure_s, n_basis).
        is_pure_u = torch.tensor(
            [len(b) == 0 for (_, b) in self._joint_indices],
            dtype=torch.bool,
        )
        is_pure_sigma = torch.tensor(
            [len(a) == 0 for (a, _) in self._joint_indices],
            dtype=torch.bool,
        )
        is_cross = ~(is_pure_u | is_pure_sigma)
        self.register_buffer("is_pure_u", is_pure_u, persistent=False)
        self.register_buffer("is_pure_sigma", is_pure_sigma, persistent=False)
        self.register_buffer("is_cross", is_cross, persistent=False)

        # Basis-evaluation auxiliary tensors registered as buffers
        # (so they move with ``.to(device)`` and are seen as stable
        # inputs by ``torch.compile`` / CUDA graphs). Computed once
        # at init from ``_joint_indices``; the polyhead exposes them
        # via ``self.basis_aux`` for use in :func:`evaluate_joint`.
        _aux = _build_basis_aux(self._joint_indices, n_features)
        self.register_buffer(
            "_basis_alpha_degs", _aux["alpha_degs"], persistent=False,
        )
        self.register_buffer(
            "_basis_beta_degs", _aux["beta_degs"], persistent=False,
        )
        self.register_buffer(
            "_basis_cheb_u_const",
            _aux["cheb_u_const"], persistent=False,
        )
        self.register_buffer(
            "_basis_cheb_v_const",
            _aux["cheb_v_const"], persistent=False,
        )
        self._basis_max_deg_u = _aux["max_deg_u"]
        self._basis_max_deg_sigma = _aux["max_deg_sigma"]

        # Gauss-Hermite (probabilist's) nodes & log-weights for the
        # smear-weight target. K=1 → empty buffers (no GH; loss step
        # falls back to the K=1 stochastic estimator).
        if self.smear_K > 1:
            nodes_np, w_np = np.polynomial.hermite_e.hermegauss(
                self.smear_K
            )
            w_norm = w_np / np.sqrt(2.0 * np.pi)
            gh_nodes = torch.tensor(nodes_np, dtype=torch.float32)
            gh_log_w = torch.tensor(np.log(w_norm), dtype=torch.float32)
        else:
            gh_nodes = torch.zeros(0, dtype=torch.float32)
            gh_log_w = torch.zeros(0, dtype=torch.float32)
        self.register_buffer("gh_nodes", gh_nodes, persistent=False)
        self.register_buffer(
            "gh_log_weights", gh_log_w, persistent=False,
        )

        # Trunk: ``[y, c] → e ∈ R^trunk_hidden``. The final coefficient
        # layer is split into three sub-heads (pure_u, pure_σ, cross)
        # — same total parameter count as a single
        # ``Linear(trunk_hidden, n_basis)``, but enables mode-
        # conditional skipping of irrelevant heads at forward time
        # (e.g. SHIFT events skip pure_σ and cross since σ=0 makes
        # those terms vanish in the polynomial). Combined with the
        # stratified mode sampler the per-step cost on the final
        # layer drops from ``B · d_emb · n_basis`` to roughly
        # ``B · d_emb · (n_pure_u + n_pure_σ + n_basis) / 3``.
        in_dim = n_features + n_cond
        trunk_layers = []
        prev = in_dim
        for _ in range(n_hidden_layers):
            trunk_layers.append(nn.Linear(prev, hidden_features))
            trunk_layers.append(activation())
            prev = hidden_features
        self.trunk = nn.Sequential(*trunk_layers)
        # Three sub-heads with sizes summing to n_basis. ``Linear`` is
        # only created if the corresponding block is non-empty (e.g.
        # in shift-only mode, pure_s and cross are empty).
        self.head_pure_u = (
            nn.Linear(prev, self._n_pure_u)
            if self._n_pure_u > 0 else None
        )
        self.head_pure_sigma = (
            nn.Linear(prev, self._n_pure_s)
            if self._n_pure_s > 0 else None
        )
        self.head_cross = (
            nn.Linear(prev, self._n_cross)
            if self._n_cross > 0 else None
        )
        # Init heads near zero so coefs ≈ 0 at start.
        with torch.no_grad():
            for head in (
                self.head_pure_u, self.head_pure_sigma, self.head_cross,
            ):
                if head is not None:
                    head.weight.mul_(0.01)
                    head.bias.zero_()

    @property
    def joint_indices(self):
        return self._joint_indices

    @property
    def basis_aux(self):
        """Dict of pre-registered basis-evaluation buffers, suitable
        for passing to :func:`evaluate_joint` as ``basis_aux=`` from
        compiled call sites. Each access returns a fresh dict (cheap)
        wrapping the current device's buffer references."""
        return {
            "alpha_degs": self._basis_alpha_degs,
            "beta_degs": self._basis_beta_degs,
            "cheb_u_const": self._basis_cheb_u_const,
            "cheb_v_const": self._basis_cheb_v_const,
            "max_deg_u": self._basis_max_deg_u,
            "max_deg_sigma": self._basis_max_deg_sigma,
        }

    def _remap_legacy_state_dict(self, state_dict):
        """Migrate a pre-split-head checkpoint into the new layout.

        Old layout: a single ``self.net`` ``nn.Sequential`` ending in a
        ``Linear(prev, n_basis)`` whose row order matched
        ``_joint_indices(...)`` raw output (no pure_u / pure_σ / cross
        reordering). State-dict keys: ``net.{2j}.{weight,bias}`` for
        each Linear layer, with the final ``net.{2L}`` being the
        coefficient layer of width ``n_basis``.

        New layout: ``self.trunk`` (``nn.Sequential`` of just the
        hidden layers + activations) plus three ``head_pure_u``,
        ``head_pure_sigma``, ``head_cross`` ``Linear`` sub-heads
        sized to the per-block coefficient counts in the *reordered*
        index list.

        Migration:
          1. Trunk ``Linear`` layers map 1:1 by stripping ``net.`` →
             ``trunk.``.
          2. Final ``Linear`` of the old ``net`` is split into the
             three sub-heads, with rows permuted from the raw
             ``_joint_indices`` ordering to the new pure_u / pure_σ /
             cross-block ordering.

        Returns the (possibly remapped) state_dict. If no ``net.*``
        keys are present, returns the input unchanged.
        """
        net_keys = [k for k in state_dict.keys() if k.startswith("net.")]
        if not net_keys:
            return state_dict
        # Find the index of the final layer (largest ``net.{N}.weight``).
        net_layer_idxs = sorted({
            int(k.split(".")[1]) for k in net_keys
            if k.endswith(".weight")
        })
        if not net_layer_idxs:
            return state_dict
        last_idx = net_layer_idxs[-1]
        trunk_idxs = net_layer_idxs[:-1]

        # Reconstruct the OLD basis-index ordering from the same
        # constructor inputs the polyhead was built with. Map each
        # old position to its position in the (new) reordered list.
        raw_indices = _joint_indices(
            self.n_features,
            self.max_deg_u,
            self.max_deg_sigma,
            self.max_cross_deg,
        )
        old_pos_of = {idx: i for i, idx in enumerate(raw_indices)}
        # ``perm_inv[new_pos] = old_pos`` such that
        # ``raw_indices[old_pos] == self._joint_indices[new_pos]``.
        perm_inv = torch.tensor(
            [old_pos_of[idx] for idx in self._joint_indices],
            dtype=torch.long,
        )

        new_state = dict(state_dict)
        # 1) Trunk layers: net.{j} → trunk.{j}.
        for j in trunk_idxs:
            for suf in ("weight", "bias"):
                k_old = f"net.{j}.{suf}"
                k_new = f"trunk.{j}.{suf}"
                if k_old in new_state:
                    new_state[k_new] = new_state.pop(k_old)
        # 2) Final layer: split into the three sub-heads with row
        #    permutation to the new ordering.
        w_old = new_state.pop(f"net.{last_idx}.weight", None)
        b_old = new_state.pop(f"net.{last_idx}.bias", None)
        if w_old is not None and b_old is not None:
            # Permute rows to new ordering, then slice into blocks.
            w_perm = w_old[perm_inv]
            b_perm = b_old[perm_inv]
            n_pu = self._n_pure_u
            n_ps = self._n_pure_s
            blocks = [
                ("head_pure_u", 0, n_pu, n_pu > 0),
                ("head_pure_sigma", n_pu, n_pu + n_ps, n_ps > 0),
                ("head_cross", n_pu + n_ps, w_perm.shape[0], self._n_cross > 0),
            ]
            for name, lo, hi, present in blocks:
                if present:
                    new_state[f"{name}.weight"] = w_perm[lo:hi]
                    new_state[f"{name}.bias"] = b_perm[lo:hi]
        return new_state

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Override to transparently migrate legacy single-Linear-head
        checkpoints. Forwards to :meth:`nn.Module.load_state_dict`
        after running :meth:`_remap_legacy_state_dict`."""
        return super().load_state_dict(
            self._remap_legacy_state_dict(state_dict),
            strict=strict, assign=assign,
        )

    def trunk_forward(
        self,
        y_std: torch.Tensor,
        c_std: torch.Tensor,
    ) -> torch.Tensor:
        """Trunk embedding ``e ∈ R^{B, hidden_features}``.

        Used by the per-mode contracted loss-step path which calls
        the per-head sub-modules (``head_pure_u``, ...) directly on
        per-mode slices of ``e`` to avoid materializing the full
        ``[B, n_basis]`` coefficient tensor.
        """
        return self.trunk(torch.cat([y_std, c_std], dim=-1))

    def forward(
        self,
        y_std: torch.Tensor,
        c_std: torch.Tensor,
    ) -> torch.Tensor:
        """Return the full ``[B, n_basis]`` coefficient tensor.

        Used by external callers (diagnostics, ONNX/AOTI export,
        polyhead-only inference) that consume coefs as the polyhead's
        output. The compiled training path bypasses this in favor of
        a per-mode contraction in :func:`joint_loss_step` /
        :func:`head_only_loss_step`, which avoids the
        ``[B, n_basis]`` intermediate altogether.
        """
        e = self.trunk_forward(y_std, c_std)
        parts = []
        if self.head_pure_u is not None:
            parts.append(self.head_pure_u(e))
        if self.head_pure_sigma is not None:
            parts.append(self.head_pure_sigma(e))
        if self.head_cross is not None:
            parts.append(self.head_cross(e))
        if not parts:
            return torch.zeros(
                e.shape[0], 0, device=e.device, dtype=e.dtype,
            )
        return torch.cat(parts, dim=-1)


def _flow_z_ladj(flow, y_std, c_std):
    z, ladj = flow(c_std).transform.call_and_ladj(y_std)
    return z, ladj


def _sigma_pack_outer(sigma: torch.Tensor) -> torch.Tensor:
    """Upper-triangular flat of ``σ ⊗ σ``.

    Shape ``[..., F] → [..., F·(F+1)/2]``. Captures the six (for
    F=3) symmetric components of ``σσᵀ``: three diagonal entries
    ``σ_d²`` and three off-diagonal cross products ``σ_d·σ_d'``.

    Used as the conditioning input to a smear-conditioned flow so
    that the σ-dependence is structurally invariant under
    ``σ → −σ`` (i.e., ``f(σ) = f(σσᵀ) = f(−σ)`` — the smear-
    symmetry of the underlying density). Off-diagonal cross
    products preserve sign correlations between components, which a
    pure ``|σ|`` (component-wise abs) input would discard.
    """
    F = sigma.shape[-1]
    iu, ju = torch.triu_indices(
        F, F, device=sigma.device,
    ).unbind(0)
    return sigma[..., iu] * sigma[..., ju]


def evaluate_joint(
    joint_coefs: torch.Tensor,
    u: torch.Tensor,
    sigma_vec: torch.Tensor,
    joint_indices,
    basis: str = "monomial",
    scale_u: float = 1.0,
    scale_sigma: float = 1.0,
    *,
    basis_aux=None,
) -> torch.Tensor:
    """joint(u, σ_vec) scalar [...]. ``basis`` selects ``"monomial"``
    (default, raw u^α · σ^β; backwards-compatible) or ``"chebyshev"``
    (tensor-product Chebyshev with axis-normalization
    ``scale_u, scale_sigma``).

    ``basis_aux`` (optional dict): pre-computed per-axis-degree
    tensors (typically ``polyhead.basis_aux`` if the polyhead
    registers them as buffers — see :class:`PolyHead.basis_aux` and
    train_shift_smear_reweight.ReweightPolyhead.basis_aux). When
    passed, the basis function uses these directly instead of
    looking them up in the runtime cache. Required for
    ``torch.compile`` + CUDA-graph compatibility (mode='reduce-
    overhead'): otherwise the cache-miss path creates ephemeral
    tensors inside the compiled forward, which CUDA graphs captures
    as outputs and then errors on overwrite."""
    phi = _select_basis(
        basis, u, sigma_vec, joint_indices,
        scale_u, scale_sigma, aux=basis_aux,
    )
    return torch.einsum("...b,...b->...", joint_coefs, phi)


def predicted_W(
    joint_coefs: torch.Tensor,
    u_shift: torch.Tensor,
    sigma_vec: torch.Tensor,
    joint_indices,
    positivity: str = "softplus",
    basis: str = "monomial",
    scale_u: float = 1.0,
    scale_sigma: float = 1.0,
    *,
    basis_aux=None,
) -> torch.Tensor:
    """Polyhead prediction of ``W(u_shift, σ_vec)`` with ``j`` the
    joint polynomial. Pure function of the head output and the
    polynomial inputs — no flow-state dependence. Returns ``W``
    *directly* via the chosen positivity wrap (no ``exp(log W)`` round
    trip). See :func:`_apply_positivity_W` for the per-wrap algebra
    and protections, and :func:`evaluate_joint` for the ``basis``
    options (``"monomial"`` or ``"chebyshev"``).
    """
    j = evaluate_joint(
        joint_coefs, u_shift, sigma_vec, joint_indices,
        basis=basis, scale_u=scale_u, scale_sigma=scale_sigma,
        basis_aux=basis_aux,
    )
    return _apply_positivity_W(j, positivity)


def predicted_log_W(
    joint_coefs: torch.Tensor,
    u_shift: torch.Tensor,
    sigma_vec: torch.Tensor,
    joint_indices,
    positivity: str = "softplus",
    basis: str = "monomial",
    scale_u: float = 1.0,
    scale_sigma: float = 1.0,
    *,
    basis_aux=None,
) -> torch.Tensor:
    """``log W_pred`` for the ``logw``-target loss path. Mirror of
    :func:`predicted_W`, returning ``log W`` directly via the chosen
    positivity wrap (no ``log(W)`` round trip on the ``"exp"`` /
    ``"asinh"`` paths). See :func:`_apply_positivity_logW`.
    ``basis`` and ``scale_*`` are forwarded to :func:`evaluate_joint`.
    """
    j = evaluate_joint(
        joint_coefs, u_shift, sigma_vec, joint_indices,
        basis=basis, scale_u=scale_u, scale_sigma=scale_sigma,
        basis_aux=basis_aux,
    )
    return _apply_positivity_logW(j, positivity)


def sample_perturbations(
    batch_size: int,
    n_dim: int,
    delta_max: float,
    sigma_max: float,
    device: torch.device,
    oversample: float = 1.3,
    include_smear: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-event sampling of the perturbation `(u_shift, σ_vec)`.

    All vectors live in **standardized target space** (R^{n_features})
    — perturbations are y-shifts and y-smears.

    ``include_smear = False`` (the shift-only default): every event is
    SHIFT mode (mode_id = 0); ``σ_vec`` and ``δ_smear`` are zero. The
    single integration sample point ``u_eval = u_shift``.

    ``include_smear = True``: stratified sampling across three sample
    modes — SHIFT (σ_vec = 0), SMEAR (u_shift = 0), JOINT (both
    nonzero). The integration sample point is
    ``u_eval = u_shift + δ_smear · v_smear``.

    Magnitude sampling:
      * ``|u|`` uniform on ``[−oversample·delta_max, +oversample·delta_max]``.
      * ``|σ_vec|`` uniform on ``[0, oversample·sigma_max]``.
      * Direction uniform on the unit sphere ``S^{n_dim − 1}``.
      * σ=0 is anchored by the SHIFT mode (1/3 of events).

    Returns ``(u_shift, σ_vec, δ_smear, v_smear, mode_id)`` where:
      * u_shift     ∈ R^{n_dim}     deterministic shift in target space
      * σ_vec       ∈ R^{n_dim}     smear scale × direction (zero in
                                     shift-only mode)
      * δ_smear     ∈ R             one Gaussian draw with variance
                                     ‖σ_vec‖² (zero in shift-only mode)
      * v_smear     ∈ R^{n_dim}     unit smear direction
      * mode_id     ∈ {0,1,2}       SHIFT=0, SMEAR=1, JOINT=2 — always
                                     0 in shift-only mode.
    """
    half = oversample * delta_max
    # δ_shift uniform on [-half, +half], v_shift on unit sphere.
    delta_shift = (torch.rand(batch_size, device=device) * 2.0 - 1.0) * half
    v_shift = torch.randn(batch_size, n_dim, device=device)
    v_shift = v_shift / v_shift.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    u_shift = delta_shift.unsqueeze(-1) * v_shift

    if not include_smear:
        # Shift-only: zero out the smear leg and force SHIFT mode for
        # every event. v_smear is a placeholder unit vector — never
        # used downstream when σ_vec = 0 — but kept shape-compatible
        # with the include_smear branch so callers don't have to
        # branch on the shape.
        zeros = torch.zeros(batch_size, device=device)
        v_smear = torch.zeros(batch_size, n_dim, device=device)
        v_smear[:, 0] = 1.0
        sigma_vec = torch.zeros(batch_size, n_dim, device=device)
        mode_id = torch.zeros(
            batch_size, device=device, dtype=torch.long,
        )
        return u_shift, sigma_vec, zeros, v_smear, mode_id

    half_sig = oversample * sigma_max
    # σ_smear uniform on [0, half_sig], v_smear on unit sphere.
    sigma_smear = torch.rand(batch_size, device=device) * half_sig
    v_smear = torch.randn(batch_size, n_dim, device=device)
    v_smear = v_smear / v_smear.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    # Stochastic K=1 Gaussian sample of δ_smear ~ N(0, σ_smear²).
    delta_smear = sigma_smear * torch.randn(batch_size, device=device)
    # Stratified mode: 0=SHIFT, 1=SMEAR, 2=JOINT — exactly 1/3 each
    # in **sorted contiguous** order ``[0…0, 1…1, 2…2]``. The data
    # loader already randomizes the ``(y, c)`` ↔ mode pairing, so a
    # within-batch shuffle here is unnecessary and would break the
    # contiguous per-mode slicing that the per-mode-contracted
    # polyhead loss path uses (see :func:`joint_loss_step`'s
    # ``_polyhead_d_per_mode`` call). The remainder when
    # ``batch_size`` isn't a multiple of 3 is given to SHIFT — the
    # lowest-noise / highest-leverage gradient signal.
    base = batch_size // 3
    extra = batch_size - 3 * base
    mode_id = torch.cat([
        torch.full((base + extra,), 0, device=device, dtype=torch.long),
        torch.full((base,), 1, device=device, dtype=torch.long),
        torch.full((base,), 2, device=device, dtype=torch.long),
    ])
    shift_active = (mode_id != 1).to(delta_shift.dtype)   # SHIFT or JOINT
    smear_active = (mode_id != 0).to(delta_smear.dtype)   # SMEAR or JOINT

    u_shift = (delta_shift * shift_active).unsqueeze(-1) * v_shift
    sigma_vec = (sigma_smear * smear_active).unsqueeze(-1) * v_smear
    delta_smear = delta_smear * smear_active
    return u_shift, sigma_vec, delta_smear, v_smear, mode_id


def _lagrange_basis_at(eps, gh_nodes):
    """Lagrange basis ``L_k(ε)`` for each event's ``ε`` at the K
    Gauss-Hermite nodes. Returns shape ``[B, K]`` with ``L[b, k] =
    Π_{j≠k} (ε_b - node_j) / (node_k - node_j)``.

    Used by the GH+residual control-variate target: the polynomial
    interpolant ``g(ε) = Σ_k W_k · L_k(ε)`` passes through ``(ε_k, W_k)``
    exactly, and ``E_ε[L_k] = w_k`` (the GH weights, since GH-2K-1
    integrates degree-K-1 polynomials exactly), so ``E_ε[g] = Σ_k W_k
    · w_k = W_K``. The residual ``W(ε~) − g(ε~)`` therefore has zero
    mean — it captures only what's beyond the polynomial truncation."""
    K = gh_nodes.shape[0]
    B = eps.shape[0]
    L = torch.ones(B, K, device=eps.device, dtype=eps.dtype)
    for k in range(K):
        for j in range(K):
            if j == k:
                continue
            L[:, k] = L[:, k] * (eps - gh_nodes[j]) / (
                gh_nodes[k] - gh_nodes[j]
            )
    return L


def _compute_target_lw(
    flow,
    head: "nn.Module",
    y_std: torch.Tensor,
    c_std: torch.Tensor,
    log_p: torch.Tensor,
    u_shift: torch.Tensor,
    sigma_vec: torch.Tensor,
    delta_smear: torch.Tensor,
    v_smear: torch.Tensor,
    n_features: int,
    detach_target: bool = True,
) -> torch.Tensor:
    """Per-event target log-weight (shape ``[B]``) used by the head
    reconstruction loss. Branches on three modes (in order of
    precedence):

    * ``head.cond_smear_target_direct = True`` (set by
      ``--cond-smear-target=direct`` / ``auto+cond-on-smear``): the
      σ-conditioned flow already represents the smear-integrated
      density by construction (training on ``(y_smeared, c, σ)``
      teaches it ``p(y | c, σ) = ∫ p_unsmeared(y - σ⊙ε | c) φ(ε) dε``).
      So the target reduces to a single perturbed forward at
      ``(y − u_shift, c_orig, σ_vec_pert)`` minus the σ=0 baseline:

          ``log W_target = log p_flow(y - u_shift | c_orig, σ_vec_pert)
                            − log p_flow(y | c_orig, σ=0)``

      Cost: 1 perturbed flow forward per event, regardless of K. No
      GH polynomial-truncation bias and no K=1 stochastic noise. The
      ``σ_vec_pert`` (the head's perturbation σ) is plugged into the
      flow's σ-conditioning slot in place of any training-time σ
      that ``c_std`` carried.

    * ``smear_K == 1`` (default when no σ-conditioning): K=1
      stochastic estimator. One perturbed flow forward at ``y -
      (u_shift + δ_smear · v_smear)`` with ``δ_smear ~ N(0,
      ‖σ_vec‖²)`` from :func:`sample_perturbations`.

    * ``smear_K > 1`` and ``smear_residual = False`` (pure GH):
      deterministic K-node Gauss-Hermite (probabilist's) quadrature
      over the smear Gaussian. Evaluation points
      ``u_eval_k = u_shift + ε_k · σ_vec`` with ε_k from
      :func:`numpy.polynomial.hermite_e.hermegauss` (stored as
      ``head.gh_nodes``). Aggregation:

          ``log W_target = logsumexp_k ( log w_k + log p(y - u_eval_k)
                                            − log p(y) )``

      which is the consistent estimator of ``log E_δ[W]`` — fixes both
      the K=1 noise floor and the Jensen bias on the logw target.
      Cost: K perturbed flow forwards per event.

    * ``smear_K > 1`` and ``smear_residual = True`` (GH + residual):
      same K GH nodes plus one extra random ``ε~ ~ N(0,1)`` per
      event. The control-variate target is

          ``W_target = W_K + (W(ε~) − g(ε~))``

      where ``W_K = Σ_k w_k W_k`` is the GH sum and ``g(ε) = Σ_k W_k
      L_k(ε)`` is the Lagrange interpolant through the K GH nodes.
      ``E[g(ε~)] = W_K`` exactly (Lagrange/GH duality), so the
      residual has zero mean and ``W_target`` is unbiased for
      ``E_ε[W]``; variance comes only from beyond-degree-(2K-1)
      features of W(ε). Cost: K + 1 perturbed flow forwards per
      event.

    ``detach_target=True`` runs the perturbed forward(s) under
    ``no_grad`` so the target carries no gradient back to the flow.
    """
    n_cond = c_std.shape[-1]
    B = y_std.shape[0]
    log_const = 0.5 * float(n_features) * _math.log(2.0 * _math.pi)

    # ------------------------------------------------------------------
    # Direct σ-conditioned target. Single perturbed forward.
    # ------------------------------------------------------------------
    if bool(getattr(head, "cond_smear_target_direct", False)):
        n_cond_base = int(getattr(head, "n_cond_base", n_cond))
        # ``sigma_vec`` is in standardized target units (same space
        # as ``u_shift``). The flow's σ-conditioning slot now takes
        # the upper-triangular pack of ``σσᵀ`` (smear-symmetric by
        # construction); convert ``sigma_vec`` to ``σ_pack`` before
        # concatenation.
        c_orig = c_std[:, :n_cond_base]
        sigma_pack_pert = _sigma_pack_outer(sigma_vec)
        c_pert = torch.cat([c_orig, sigma_pack_pert], dim=-1)
        y_pert = y_std - u_shift

        def _eval_direct():
            z_p, ladj_p = _flow_z_ladj(flow, y_pert, c_pert)
            log_p_p = (
                -0.5 * (z_p * z_p).sum(dim=-1) - log_const + ladj_p
            )
            return log_p_p - log_p.detach()

        if detach_target:
            with torch.no_grad():
                return _eval_direct()
        return _eval_direct()

    K_smear = int(head.smear_K)
    smear_residual = bool(head.smear_residual)

    if K_smear > 1:
        eps_nodes = head.gh_nodes
        log_gh_w = head.gh_log_weights
        # ε_extra ~ N(0,1) per event for the control-variate residual.
        # Drawn outside no_grad so the same RNG state is consumed
        # whether or not detach_target is on (the draw is independent
        # of flow gradients).
        if smear_residual:
            eps_extra = torch.randn(B, device=y_std.device, dtype=y_std.dtype)
            eps_all = torch.cat(
                [eps_nodes.to(eps_extra.dtype), eps_extra], dim=0,
            )           # [K+1]
            n_eps = K_smear + 1
        else:
            eps_all = eps_nodes
            n_eps = K_smear
        # u_eval_k[k, b, :] = u_shift[b, :] + ε_k(,b) · σ_vec[b, :].
        # For GH nodes ε_k is per-quadrature-node (broadcast across B);
        # for the extra slot ε_extra is per-event.
        if smear_residual:
            eps_view = torch.cat([
                eps_nodes.view(K_smear, 1).expand(K_smear, B),
                eps_extra.view(1, B),
            ], dim=0)                                      # [K+1, B]
            u_eval_k = (
                u_shift.unsqueeze(0)
                + eps_view.unsqueeze(-1) * sigma_vec.unsqueeze(0)
            )                                              # [K+1, B, n_features]
        else:
            u_eval_k = (
                u_shift.unsqueeze(0)
                + eps_nodes.view(K_smear, 1, 1) * sigma_vec.unsqueeze(0)
            )                                              # [K, B, n_features]
        nB = n_eps * B
        perturbed_y = (
            y_std.unsqueeze(0).expand(n_eps, -1, -1) - u_eval_k
        ).reshape(nB, n_features)
        perturbed_c = (
            c_std.unsqueeze(0)
            .expand(n_eps, -1, -1)
            .reshape(nB, n_cond)
        )

        def _eval():
            z_p, ladj_p = _flow_z_ladj(flow, perturbed_y, perturbed_c)
            log_p_p = (
                -0.5 * (z_p * z_p).sum(dim=-1) - log_const + ladj_p
            ).view(n_eps, B)
            true_lw_all = log_p_p - log_p.detach().unsqueeze(0)
            log_W_K = torch.logsumexp(
                true_lw_all[:K_smear] + log_gh_w.view(K_smear, 1), dim=0,
            )                                              # [B]
            if not smear_residual:
                return log_W_K

            # log W_K is the dominant term; we add log1p of a small
            # control-variate correction. All ratios are formed
            # against W_K so the absolute scale of W cancels.
            log_W_extra = true_lw_all[K_smear]            # [B]
            ratio_extra = torch.exp(log_W_extra - log_W_K)  # [B]
            # exp(true_lw_all[:K] - log_W_K) ∈ [0, 1/w_min] is well-
            # conditioned because log_W_K ≥ max_k(log_w_k + log_W_k).
            ratio_K = torch.exp(
                true_lw_all[:K_smear] - log_W_K.unsqueeze(0)
            )                                              # [K, B]
            L_at_eps = _lagrange_basis_at(eps_extra, eps_nodes)   # [B, K]
            ratio_g = torch.einsum("kb,bk->b", ratio_K, L_at_eps)  # [B]
            delta = ratio_extra - ratio_g
            # delta ∈ (-∞, ∞) per event; the control variate guarantees
            # E[delta] = 0 but per-event values can dip below -1
            # (giving a negative W_target) when ε~ falls in the tails
            # and the Lagrange interpolant extrapolates large. Floor
            # 1+delta away from zero so log stays finite. The
            # 1e-3 floor caps the per-event bias at |log 1e-3| ≈ 6.9
            # — small relative to typical |log W| in the smear-bias
            # regime, while keeping any individual gradient bounded
            # so Adam can't be hijacked by a single extrapolating
            # event. These rare events are biased; the bulk of events
            # is unbiased.
            return log_W_K + torch.log1p(delta.clamp_min(-1.0 + 1e-3))

        if detach_target:
            with torch.no_grad():
                return _eval()
        return _eval()

    # K=1 stochastic.
    u_eval = u_shift + delta_smear.unsqueeze(-1) * v_smear

    def _eval_stoch():
        z_e, ladj_e = _flow_z_ladj(flow, y_std - u_eval, c_std)
        log_p_e = (
            -0.5 * (z_e * z_e).sum(dim=-1) - log_const + ladj_e
        )
        return log_p_e - log_p.detach()

    if detach_target:
        with torch.no_grad():
            return _eval_stoch()
    return _eval_stoch()


def _detach_pure_coefs_in_joint(
    joint_coefs: torch.Tensor,
    polyhead: "PolyHead",
    mode: torch.Tensor,
) -> torch.Tensor:
    """Per-event mask that detaches the pure-u (β=∅) and pure-σ
    (α=∅) coefficient slots only for JOINT-mode events (mode==2),
    leaving SHIFT/SMEAR events and the cross-term slots untouched.

    Used to route gradients away from the noisier JOINT samples for
    the pure terms — those parameters are still trained by the
    deterministic SHIFT (pure-u) and lower-cross-coupling SMEAR
    (pure-σ) events, which carry a lower-noise signal."""
    is_joint = (mode == 2)
    pure_basis = head.is_pure_u | head.is_pure_sigma
    detach_mask = is_joint.unsqueeze(-1) & pure_basis.unsqueeze(0)
    return torch.where(detach_mask, joint_coefs.detach(), joint_coefs)


def _split_loss_by_mode(
    per_event: torch.Tensor,
    loss_w: torch.Tensor,
    mode: torch.Tensor,
) -> torch.Tensor:
    """Aggregate ``per_event * loss_w`` separately over SHIFT/SMEAR/
    JOINT events. Returns a length-3 tensor in that order; modes
    with zero population in this batch are reported as 0."""
    out = []
    for m in (0, 1, 2):
        mask = (mode == m).to(per_event.dtype)
        num = (loss_w * per_event * mask).sum()
        den = (loss_w * mask).sum().clamp_min(1e-30)
        out.append(num / den)
    return torch.stack(out)


def _polyhead_d_per_mode(
    polyhead: "PolyHead",
    y_std: torch.Tensor,
    c_std: torch.Tensor,
    u_shift: torch.Tensor,
    sigma_vec: torch.Tensor,
    detach_pure_in_joint: bool,
) -> torch.Tensor:
    """Per-event scalar ``j = polyhead·basis``, evaluated via per-mode
    Order-2 contraction.

    Replaces the trio
        ``coefs = polyhead(y, c)`` →
        (optional ``_detach_pure_coefs_in_joint``) →
        ``j = evaluate_joint(coefs, u, σ, ...)``
    with a path that **never materialises the full [B, n_basis]
    coefficient tensor**. Per event:

        d = (e · (φ_block @ W^T)) + (φ_block @ b)

    where ``e = polyhead.trunk_forward(y, c)`` is the trunk
    embedding, ``φ_block`` is the basis slice for the head being
    applied (pure_u / pure_σ / cross), and ``W, b`` are the
    sub-head's parameters. The intermediate tensor is
    ``[N_block, d_emb]`` rather than ``[N_block, n_basis]``.

    Per-mode skipping (relies on stratified-contiguous mode_id from
    :func:`sample_perturbations`):
      * SHIFT events (σ = 0): only ``head_pure_u`` runs.
      * SMEAR events (u = 0): only ``head_pure_sigma`` runs.
      * JOINT events (both nonzero): all three heads run; if
        ``detach_pure_in_joint``, the pure-u and pure-σ contributions
        are ``.detach()``-ed so JOINT-mode gradient flows only
        through ``head_cross``.

    For the shift-only configuration (``head_pure_sigma`` and
    ``head_cross`` both ``None``) every event is treated as SHIFT.
    """
    B = y_std.shape[0]

    # Mode-block boundaries — derived from B (deterministic given the
    # stratified sampler in sample_perturbations: [0…0, 1…1, 2…2]).
    if polyhead.head_pure_sigma is None and polyhead.head_cross is None:
        n_shift_b, n_smear_b, n_joint_b = B, 0, 0
    else:
        base = B // 3
        extra = B - 3 * base
        n_shift_b, n_smear_b, n_joint_b = base + extra, base, base

    e = polyhead.trunk_forward(y_std, c_std)         # [B, d_emb]
    d_emb = e.shape[-1]

    phi = _select_basis(
        polyhead.basis, u_shift, sigma_vec, polyhead.joint_indices,
        scale_u=polyhead.basis_scale_u,
        scale_sigma=polyhead.basis_scale_sigma,
        aux=polyhead.basis_aux,
    )                                                 # [B, n_basis]

    n_pu = polyhead._n_pure_u
    n_ps = polyhead._n_pure_s

    def _order2(e_blk, phi_blk, head):
        """``d = (e · (φ @ W^T)) + (φ @ b)`` per event. ``e_blk``
        ``[N, d_emb]``, ``phi_blk`` ``[N, n_out]``,
        ``head`` ``nn.Linear(d_emb, n_out)``. Returns ``[N]``."""
        if head is None:
            return torch.zeros(
                e_blk.shape[0], device=e_blk.device, dtype=e_blk.dtype,
            )
        temp = phi_blk @ head.weight                  # [N, d_emb]
        bias_term = phi_blk @ head.bias               # [N]
        return (e_blk * temp).sum(-1) + bias_term

    d_chunks = []

    # SHIFT events: only pure_u contributes (σ = 0).
    if n_shift_b > 0:
        e_s = e[:n_shift_b]
        phi_s_pu = phi[:n_shift_b, :n_pu]
        d_chunks.append(_order2(e_s, phi_s_pu, polyhead.head_pure_u))

    # SMEAR events: only pure_σ contributes (u = 0).
    if n_smear_b > 0:
        e_m = e[n_shift_b : n_shift_b + n_smear_b]
        phi_m_ps = phi[
            n_shift_b : n_shift_b + n_smear_b, n_pu : n_pu + n_ps,
        ]
        d_chunks.append(_order2(e_m, phi_m_ps, polyhead.head_pure_sigma))

    # JOINT events: all three head contributions; optionally detach
    # the pure-u and pure-σ contributions so JOINT-mode gradient
    # flows only through head_cross.
    if n_joint_b > 0:
        j0 = n_shift_b + n_smear_b
        e_j = e[j0:]
        phi_j = phi[j0:]
        d_j_pu = _order2(e_j, phi_j[:, :n_pu], polyhead.head_pure_u)
        d_j_ps = _order2(
            e_j, phi_j[:, n_pu : n_pu + n_ps], polyhead.head_pure_sigma,
        )
        d_j_cr = _order2(
            e_j, phi_j[:, n_pu + n_ps :], polyhead.head_cross,
        )
        if detach_pure_in_joint:
            d_j_pu = d_j_pu.detach()
            d_j_ps = d_j_ps.detach()
        d_chunks.append(d_j_pu + d_j_ps + d_j_cr)

    return torch.cat(d_chunks, dim=0)


def _mlp_d_per_mode(
    mlp: "nn.Module",
    y_std: torch.Tensor,
    c_std: torch.Tensor,
    u_shift: torch.Tensor,
    sigma_vec: torch.Tensor,
    detach_pure_in_joint: bool,
) -> torch.Tensor:
    """Per-event pre-positivity scalar ``d`` for the MLP-arch head, via
    dual-forward:

        ``d = head(e, u_shift, σ_pack) − head(e, 0, 0)``

    so ``r = 1`` exactly at ``(u, σ) = (0, 0)``. ``Σ_pack`` is the
    upper-triangular outer-product of ``σ_vec`` (invariant under
    ``σ_vec → −σ_vec``, so the smear symmetry is structural, just like
    in the polyhead's even-|β| basis).

    With ``detach_pure_in_joint=True`` the dual-forward is replaced by
    a 4-call decomposition

        ``d_full = f_full − f_zero``,
        ``d_pure_u = f_us − f_zero``,
        ``d_pure_s = f_sm − f_zero``,
        ``d_mixed = d_full − d_pure_u − d_pure_s``,

    and on JOINT-mode events (mode == 2) the pure-u / pure-σ
    contributions are detached, so JOINT-mode gradients flow only
    through ``d_mixed``. Mode boundaries are derived from ``B`` (the
    same stratified-contiguous convention used by
    :func:`sample_perturbations` and ``_polyhead_d_per_mode``).
    """
    B = y_std.shape[0]
    e = mlp.trunk_forward(y_std, c_std)                  # [B, d_emb]
    if mlp.head_layer1_sigma is not None:
        sigma_pack = (
            sigma_vec[..., mlp.sigma_pack_iu]
            * sigma_vec[..., mlp.sigma_pack_ju]
        )
    else:
        sigma_pack = torch.zeros(
            B, 0, device=y_std.device, dtype=y_std.dtype,
        )

    if not detach_pure_in_joint:
        # 2·B head queries: f(e, u, σ_pack) and f(e, 0, 0).
        e_dual = torch.cat([e, e], dim=0)
        u_dual = torch.cat(
            [u_shift, torch.zeros_like(u_shift)], dim=0,
        )
        sp_dual = torch.cat(
            [sigma_pack, torch.zeros_like(sigma_pack)], dim=0,
        )
        f = mlp.head_forward(e_dual, u_dual, sp_dual)
        return f[:B] - f[B:]

    # 4·B head queries to decompose d into pure-u, pure-σ, mixed.
    e_q = torch.cat([e, e, e, e], dim=0)
    zeros_u = torch.zeros_like(u_shift)
    zeros_sp = torch.zeros_like(sigma_pack)
    u_q = torch.cat([u_shift, zeros_u, u_shift, zeros_u], dim=0)
    sp_q = torch.cat(
        [sigma_pack, zeros_sp, zeros_sp, sigma_pack], dim=0,
    )
    f = mlp.head_forward(e_q, u_q, sp_q)
    f_full = f[0 * B : 1 * B]
    f_zero = f[1 * B : 2 * B]
    f_us   = f[2 * B : 3 * B]
    f_sm   = f[3 * B : 4 * B]
    d_full = f_full - f_zero
    d_pure_u = f_us - f_zero
    d_pure_s = f_sm - f_zero
    # JOINT events live in the contiguous tail of the batch (the
    # stratified mode sampler emits ``[0…0, 1…1, 2…2]``). Build the
    # is_joint mask from B alone — same convention as
    # ``_polyhead_d_per_mode``.
    base = B // 3
    extra = B - 3 * base
    n_shift_b = base + extra
    n_smear_b = base
    n_joint_b = base
    is_joint = torch.zeros(B, dtype=torch.bool, device=y_std.device)
    if n_joint_b > 0:
        is_joint[n_shift_b + n_smear_b:] = True
    d_pure_u_used = torch.where(is_joint, d_pure_u.detach(), d_pure_u)
    d_pure_s_used = torch.where(is_joint, d_pure_s.detach(), d_pure_s)
    d_mixed = d_full - d_pure_u - d_pure_s
    return d_pure_u_used + d_pure_s_used + d_mixed


def _mlp_factored_d_per_mode(
    head: "nn.Module",
    y_std: torch.Tensor,
    c_std: torch.Tensor,
    u_shift: torch.Tensor,
    sigma_vec: torch.Tensor,
) -> torch.Tensor:
    """Per-event pre-positivity scalar ``d`` for the factored MLP head.

    ``log W = ⟨u, A(e, ·)⟩ + ⟨σ_pack, B(e, ·)⟩ [ + ⟨u⊗σ_pack, C(e, u, σ_pack)⟩ ]``
    is computed via :meth:`ReweightMLPFactored.head_forward_components`,
    then per-event ``.detach()`` is applied on JOINT-mode events to the
    pure-shift and/or pure-smear blocks per the head's
    ``detach_pure_shift_in_joint`` / ``detach_pure_smear_in_joint``
    flags. Mode boundaries follow the same stratified-contiguous
    convention as :func:`_mlp_d_per_mode` and
    :func:`_polyhead_d_per_mode`.
    """
    B = y_std.shape[0]
    e = head.trunk_forward(y_std, c_std)
    if head.n_sigma_pack > 0:
        sigma_pack = (
            sigma_vec[..., head.sigma_pack_iu]
            * sigma_vec[..., head.sigma_pack_ju]
        )
    else:
        sigma_pack = torch.zeros(
            B, 0, device=y_std.device, dtype=y_std.dtype,
        )
    pu, ps, cr = head.head_forward_components(e, u_shift, sigma_pack)
    if head.detach_pure_shift_in_joint or head.detach_pure_smear_in_joint:
        # JOINT events live in the contiguous tail of the batch (same
        # convention as ``_polyhead_d_per_mode``).
        base = B // 3
        extra = B - 3 * base
        n_shift_b = base + extra
        n_smear_b = base
        n_joint_b = base
        is_joint = torch.zeros(B, dtype=torch.bool, device=y_std.device)
        if n_joint_b > 0:
            is_joint[n_shift_b + n_smear_b:] = True
        if head.detach_pure_shift_in_joint:
            pu = torch.where(is_joint, pu.detach(), pu)
        if head.detach_pure_smear_in_joint:
            ps = torch.where(is_joint, ps.detach(), ps)
    d = pu + ps
    if cr is not None:
        d = d + cr
    return d


def _head_d_per_mode(
    head: "nn.Module",
    y_std: torch.Tensor,
    c_std: torch.Tensor,
    u_shift: torch.Tensor,
    sigma_vec: torch.Tensor,
    detach_pure_in_joint: bool,
) -> torch.Tensor:
    """Arch-agnostic dispatch to per-event ``d`` (or ``j``).

    ``PolyHead`` → :func:`_polyhead_d_per_mode` (Order-2 contraction
    on the polynomial basis); ``ReweightMLPFactored`` (marker
    ``is_factored`` set) → :func:`_mlp_factored_d_per_mode`
    (structural factorisation, per-mode detach via head flags);
    ``ReweightMLP_B`` (default) → :func:`_mlp_d_per_mode`
    (dual-forward on the MLP head). All return shape ``[B]``; the
    caller wraps with the chosen positivity and computes the
    per-event loss against ``true_lw``.
    """
    if isinstance(head, PolyHead):
        return _polyhead_d_per_mode(
            head, y_std, c_std, u_shift, sigma_vec,
            detach_pure_in_joint=detach_pure_in_joint,
        )
    if getattr(head, "is_factored", False):
        return _mlp_factored_d_per_mode(
            head, y_std, c_std, u_shift, sigma_vec,
        )
    return _mlp_d_per_mode(
        head, y_std, c_std, u_shift, sigma_vec,
        detach_pure_in_joint=detach_pure_in_joint,
    )


def _per_event_loss(
    poly_err: torch.Tensor,
    loss_fn: str,
    huber_delta: float,
) -> torch.Tensor:
    """Per-event loss on residual ``poly_err``. ``mse`` returns
    ``r²``; ``huber`` returns ``r²`` for ``|r| ≤ δ`` and
    ``2δ|r| − δ²`` beyond — i.e., 2× the standard Huber so the
    quadratic regime coincides exactly with MSE (no scale jump
    when switching shapes)."""
    if loss_fn == "mse":
        return poly_err * poly_err
    elif loss_fn == "huber":
        abs_r = poly_err.abs()
        quad = poly_err * poly_err
        lin = 2.0 * huber_delta * abs_r - huber_delta * huber_delta
        return torch.where(abs_r <= huber_delta, quad, lin)
    else:
        raise ValueError(f"unknown loss_fn {loss_fn!r}")


def _expkl_per_event(
    pred_log: torch.Tensor, target_log: torch.Tensor,
) -> torch.Tensor:
    """Per-event expKL (f-GAN-KL / I-divergence) loss:

        f(r) = r − log r − 1,   r = W_target / W_pred = exp(log W_target − s)

    Equivalently in log-space: ``L = exp(δ) − δ − 1`` with
    ``δ = log W_target − s``. Convex in ``s``; minimum 0 at ``δ = 0``
    (i.e., ``s = log W_target``); strictly positive otherwise.

    The expectation
    ``E[L(s)] = exp(−s)·E[W_target] − E[log W_target] + s − 1``
    has its minimum at ``s* = log E[W_target]``. So whenever
    ``W_target`` is an unbiased estimator of ``E[W]`` in linear
    space — true for K=1 stochastic and for ``--smear-residual`` —
    expKL recovers ``s* = log E[W]`` exactly, with no Jensen bias
    from working in log space (unlike MSE on logw).

    ``δ`` is clamped to ``±_LOG_W_CLAMP`` (=30) so ``exp(δ)`` stays
    finite in fp32. The clamp is only ever active in pathological
    regimes (e.g., ``--positivity=softplus`` driving
    ``pred_log`` to its dtype-tiny floor); for the ``exp`` and
    ``asinh`` positivity wraps the natural range of ``δ`` is well
    inside the clamp."""
    delta = (target_log - pred_log).clamp(
        min=-_LOG_W_CLAMP, max=_LOG_W_CLAMP,
    )
    return torch.exp(delta) - delta - 1.0


def _wbce_per_event(
    pred_log: torch.Tensor, target_log: torch.Tensor,
) -> torch.Tensor:
    """Per-event importance-sampled BCE / single-sample CARL loss:

        L(s) = softplus(s) + W_target · softplus(−s),
        s = pred_log,  W_target = exp(target_log).

    Per-event gradient is

        ∂L/∂s = σ(s) − W_target · σ(−s)

    so under sampling from p_nom with E[W_target] = E[W],

        E[∂L/∂s] = σ(s) − E[W] · σ(−s) = 0
                   ⟺ s* = log E[W].

    Same unbiased optimum as :func:`_expkl_per_event` for any unbiased
    W-side target (K=1 stochastic, smear_residual). Difference is the
    underestimate-side tail behaviour: expKL has gradient
    ``−W_target · exp(−s)`` (exponential blow-up when ``s ≪ log W_target``),
    while wbce has ``−W_target · σ(−s)`` (linear in ``W_target``,
    bounded by ``W_target`` regardless of how negative ``s`` gets).
    Numerically more stable on rare large-W tail events; lets you
    relax the ``target_threshold`` mask if desired.

    Equivalent in expectation to standard CARL-BCE under importance
    sampling: ``E_pert[softplus(−s)] = E_nom[r · softplus(−s)]``,
    with the per-event ``W_target`` standing in for ``r``.

    Computed via ``F.softplus`` for numerical stability across the
    full range of ``pred_log``."""
    return F.softplus(pred_log) + torch.exp(target_log) * F.softplus(-pred_log)


def joint_loss_step(
    flow,
    head: "nn.Module",
    y_std: torch.Tensor,
    c_std: torch.Tensor,
    weights: torch.Tensor,
    delta_max: float,
    sigma_max: float,
    aux_jacobian_weight: float = 0.0,
    oversample: float = 1.3,
    loss_target: str = "w",
    weight_power: int = 1,
    loss_fn: str = "huber",
    huber_delta: float = 1.0,
    detach_pure_in_joint: bool = False,
    detach_target: bool = True,
):
    """Combined NLL (flow) + reconstruction loss (polyhead) over a
    stratified mix of (pure-shift, pure-smear, joint shift+smear)
    perturbations in **target / y-space** — one sample per event
    per step.

    Per event: draw a mode ∈ {SHIFT, SMEAR, JOINT}, build
    ``(u_shift, σ_vec, δ_smear)`` in standardized y-space, compute
    ``u_eval = u_shift + δ_smear · v_smear``, and run the flow at
    ``(y_std − u_eval, c_std)`` to get the unbiased target

        ``log w_true = log p(y - u_eval | c) - log p(y | c)``

    which is the per-event reweight from the nominal distribution to
    the y-shifted (or y-smeared) distribution. Compare against the
    polyhead's ``W_pred(u_shift, σ_vec)``. For pure-smear and joint
    modes the single Gaussian draw of ``δ_smear`` is the K=1 stochastic
    estimator for ``E_δ[w(...)] = W_smear``.

    With ``detach_target=True`` (default) the perturbed flow forward
    runs under ``no_grad`` — its output is a target only, so cutting
    the backward through it saves one full flow backward per step
    without changing gradients for either the flow (still trained on
    its own NLL) or the polyhead (still trained on the recon loss).

    With ``detach_pure_in_joint=True``, JOINT-mode events detach the
    pure-u and pure-σ basis-coefficient contributions before the
    polynomial evaluation, so JOINT-mode gradients only update the
    cross-term coefficients. Stabilizes training when K=1 stochastic
    smear sampling makes the JOINT loss noisy.

    Returns ``(loss, nll_w, w_loss_w, aux_w, w_loss_split)`` with
    ``w_loss_split`` of shape ``[3]`` = (shift, smear, joint).
    """
    n_features = y_std.shape[-1]
    B = y_std.shape[0]
    device = y_std.device

    u_shift, sigma_vec, delta_smear, v_smear, mode = sample_perturbations(
        B, n_features, delta_max, sigma_max, device, oversample=oversample,
        include_smear=head.include_smear,
    )

    # Baseline flow forward — gradients DO flow back through this
    # (drives the NLL term).
    z, ladj = _flow_z_ladj(flow, y_std, c_std)
    log_phi = (
        -0.5 * (z * z).sum(dim=-1)
        - 0.5 * float(n_features) * _math.log(2.0 * _math.pi)
    )
    log_p = log_phi + ladj
    nll = -log_p

    # Target log-weight: K=1 stochastic or K>1 Gauss-Hermite
    # quadrature over the smear Gaussian, depending on
    # ``head.smear_K``. K perturbed forwards are run under
    # no_grad when ``detach_target`` (default True).
    true_lw = _compute_target_lw(
        flow, head, y_std, c_std, log_p,
        u_shift, sigma_vec, delta_smear, v_smear,
        n_features, detach_target=detach_target,
    )

    # Trustable-target mask. The Gaussian tail of δ_smear can place
    # y - u_eval many σ outside the data manifold; with a partially-
    # trained flow that gives ``|true_lw|`` in the tens or hundreds.
    # These events aren't representative of any real reweight regime
    # (we don't deploy at 5σ shifts), so we simply drop them from
    # the polyhead loss contribution.
    target_threshold = 10.0   # |log w| up to 10 → |w| up to ~22000
    trustable = torch.isfinite(true_lw) & (true_lw.abs() < target_threshold)
    trustable_f = trustable.to(true_lw.dtype)
    # Replace non-trustable targets with 0 so subsequent ops (exp,
    # diff) don't propagate inf/NaN; the per-event contribution is
    # zeroed by the mask at the end either way.
    true_lw_safe = torch.where(
        trustable, true_lw, torch.zeros_like(true_lw),
    )

    # Per-mode Order-2 contraction: avoids materialising the full
    # ``[B, n_basis]`` coefficient tensor and skips irrelevant heads
    # per mode (SHIFT skips pure_σ + cross, SMEAR skips pure_u +
    # cross). The loss family then picks which positivity-wrapped
    # form (W or log W) it needs. Each wrap returns its target form
    # directly — no exp(log W) or log(W) round trips.
    j = _head_d_per_mode(
        head, y_std, c_std, u_shift, sigma_vec,
        detach_pure_in_joint=detach_pure_in_joint,
    )
    if loss_fn == "expkl":
        # expKL is a complete (pred, target) loss in log-space: it
        # uses ``s = pred_log`` and ``log W_target`` jointly, not a
        # residual. Optimum is ``s* = log E[W_target]``, which equals
        # ``log E[W]`` whenever the target's W-side is unbiased
        # (K=1 stochastic or smear_residual). ``loss_target`` is
        # ignored for this branch — expKL always works in log-space.
        pred_log = _apply_positivity_logW(j, head.positivity)
        per_event_raw = _expkl_per_event(pred_log, true_lw_safe)
        per_event = per_event_raw * trustable_f
    elif loss_fn == "wbce":
        # Importance-sampled BCE. Same unbiased optimum as expKL but
        # with linear-in-W_target tail growth on the underestimate
        # side (vs expKL's exp blow-up). loss_target is ignored.
        # Disable the exp-positivity ±_LOG_W_CLAMP defensive clamp on
        # ``pred_log`` — wbce only feeds ``pred_log`` into ``softplus``
        # which is stable for any finite input, and the clamp would
        # zero gradients in the |log W| > 30 regions.
        pred_log = _apply_positivity_logW(
            j, head.positivity, clamp=_math.inf,
        )
        per_event_raw = _wbce_per_event(pred_log, true_lw_safe)
        per_event = per_event_raw * trustable_f
    else:
        if loss_target == "w":
            pred_W = _apply_positivity_W(j, head.positivity)
            true_W = torch.exp(true_lw_safe)
            poly_err = (pred_W - true_W) * trustable_f
        elif loss_target == "logw":
            pred_log = _apply_positivity_logW(j, head.positivity)
            poly_err = (pred_log - true_lw_safe) * trustable_f
        else:
            raise ValueError(f"unknown loss_target {loss_target!r}")
        per_event = _per_event_loss(poly_err, loss_fn, huber_delta)

    # Per-event weighting power.
    if weight_power == 1:
        loss_w = weights
    elif weight_power == 2:
        loss_w = weights * weights
    else:
        raise ValueError(f"unsupported weight_power {weight_power!r}")

    wsum = weights.sum().clamp_min(1e-30)
    loss_wsum = loss_w.sum().clamp_min(1e-30)
    nll_w = (weights * nll).sum() / wsum
    w_loss_w = (loss_w * per_event).sum() / loss_wsum
    w_loss_split = _split_loss_by_mode(per_event, loss_w, mode)

    # The aux-jacobian path (matching head's first-order coefs to the
    # flow's analytic JVP) was tied to the Δz polynomial; with the
    # Δz-free design that route no longer exists. The argument is
    # accepted for backwards-compatible call signatures but is a
    # no-op. The polyhead's first-order coefs are still trained
    # implicitly by the reconstruction loss.
    aux_w = torch.zeros((), device=device, dtype=nll_w.dtype)
    _ = aux_jacobian_weight  # accepted for compat, unused

    total = nll_w + w_loss_w
    return (
        total,
        nll_w.detach(),
        w_loss_w.detach(),
        aux_w.detach(),
        w_loss_split.detach(),
    )


def head_only_loss_step(
    flow,
    head: "nn.Module",
    c_std: torch.Tensor,
    weights: torch.Tensor,
    delta_max: float,
    sigma_max: float,
    oversample: float = 1.3,
    loss_target: str = "w",
    weight_power: int = 1,
    loss_fn: str = "huber",
    huber_delta: float = 1.0,
    detach_pure_in_joint: bool = False,
):
    """Polyhead-only training step against a frozen flow.

    The flow is used purely as a fixed conditional density:
      * ``y_std`` is sampled from the flow at each event's MC
        ``c_std``, so ``(y, c)`` is jointly distributed according to
        the flow's learned p(y, c) on the data marginal of c.
      * The flow at the y-perturbed point ``(y - u_eval, c)``
        provides the unbiased ground-truth ``log w`` target (K=1
        stochastic estimator when smear is involved).

    All flow forwards are under ``torch.no_grad()``. Only the
    polyhead receives gradients. Returns the same 4-tuple shape as
    :func:`joint_loss_step` so the train-loop bookkeeping doesn't
    branch — ``nll_w`` is the (frozen-flow) NLL evaluated for free
    along the way; it stays roughly constant across epochs and is
    informational only.
    """
    B = c_std.shape[0]
    device = c_std.device

    with torch.no_grad():
        # Sample ``y_std`` and obtain ``log p(y_std | c)`` in a single
        # flow pass via ``rsample_and_log_prob``. The previous form
        # (``flow(c).sample(...)`` followed by an explicit
        # ``_flow_z_ladj(flow, y, c)``) ran the flow twice on the same
        # ``y_std`` — one decode (z → y, ladj_inv) inside ``sample`` and
        # one encode (y → z, ladj_fwd) for the log-density. zuko's
        # combined API uses just the inverse direction's
        # ``call_and_ladj`` and reconstitutes ``log p`` from the base
        # density and the inverse log-det in one pass.
        y_std, log_p = flow(c_std).rsample_and_log_prob(torch.Size([]))
        y_std = y_std.detach()
        log_p = log_p.detach()
        n_features = y_std.shape[-1]
        # Informational NLL only (flow is frozen here).
        nll = -log_p

        # Sample perturbations using the same scheme as joint_loss_step.
        u_shift, sigma_vec, delta_smear, v_smear, mode = (
            sample_perturbations(
                B, n_features, delta_max, sigma_max, device,
                oversample=oversample,
                include_smear=head.include_smear,
            )
        )

    # Target log-weight via the same K=1 / K>1 GH branch as
    # joint_loss_step; the helper handles its own no_grad.
    true_lw = _compute_target_lw(
        flow, head, y_std, c_std, log_p,
        u_shift, sigma_vec, delta_smear, v_smear,
        n_features, detach_target=True,
    )

    # Trustable-target mask (same as joint_loss_step).
    target_threshold = 10.0
    trustable = torch.isfinite(true_lw) & (true_lw.abs() < target_threshold)
    trustable_f = trustable.to(true_lw.dtype)
    true_lw_safe = torch.where(
        trustable, true_lw, torch.zeros_like(true_lw),
    )

    # Polyhead forward — only this part has gradients. Per-mode
    # Order-2 contraction; same path as joint_loss_step but with
    # only the polyhead under gradient (the flow forward above is
    # under no_grad).
    j = _head_d_per_mode(
        head, y_std, c_std, u_shift, sigma_vec,
        detach_pure_in_joint=detach_pure_in_joint,
    )
    if loss_fn == "expkl":
        pred_log = _apply_positivity_logW(j, head.positivity)
        per_event_raw = _expkl_per_event(pred_log, true_lw_safe)
        per_event = per_event_raw * trustable_f
    elif loss_fn == "wbce":
        # See joint_loss_step's wbce branch for the clamp=inf
        # rationale (softplus-only consumption ⇒ no overflow risk).
        pred_log = _apply_positivity_logW(
            j, head.positivity, clamp=_math.inf,
        )
        per_event_raw = _wbce_per_event(pred_log, true_lw_safe)
        per_event = per_event_raw * trustable_f
    else:
        if loss_target == "w":
            pred_W = _apply_positivity_W(j, head.positivity)
            true_W = torch.exp(true_lw_safe)
            poly_err = (pred_W - true_W) * trustable_f
        elif loss_target == "logw":
            pred_log = _apply_positivity_logW(j, head.positivity)
            poly_err = (pred_log - true_lw_safe) * trustable_f
        else:
            raise ValueError(f"unknown loss_target {loss_target!r}")
        per_event = _per_event_loss(poly_err, loss_fn, huber_delta)

    if weight_power == 1:
        loss_w = weights
    elif weight_power == 2:
        loss_w = weights * weights
    else:
        raise ValueError(f"unsupported weight_power {weight_power!r}")

    wsum = weights.sum().clamp_min(1e-30)
    loss_wsum = loss_w.sum().clamp_min(1e-30)
    nll_w = (weights * nll).sum() / wsum
    w_loss_w = (loss_w * per_event).sum() / loss_wsum
    w_loss_split = _split_loss_by_mode(per_event, loss_w, mode)

    aux_w = torch.zeros((), device=device, dtype=nll_w.dtype)
    total = w_loss_w
    return (
        total,
        nll_w.detach(),
        w_loss_w.detach(),
        aux_w.detach(),
        w_loss_split.detach(),
    )


# -----------------------------------------------------------------------------
# Flow-only and flow+polyhead training wrappers
# -----------------------------------------------------------------------------

class FlowWithLogProb(nn.Module):
    """DDP-friendly wrapper that returns log p(x|c) from forward().

    zuko flows return a distribution object from their __call__, which
    DDP can't instrument for gradient synchronization (DDP expects
    forward to return tensors/standard containers). Wrapping the
    log_prob call in a standard nn.Module.forward makes DDP happy.
    """
    def __init__(self, flow: nn.Module):
        super().__init__()
        self.flow = flow

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.flow(c).log_prob(x)


class FlowHeadModel(nn.Module):
    """DDP-friendly wrapper around (flow, polyhead) that, given a
    standardized batch, returns the joint loss components.

    forward(x_std, c_std, w, delta_max, sigma_max, ...) →
        (total, nll_w, w_loss_w, aux_w, w_loss_split)

    Using a tensor-only return keeps DDP happy. The scalar config
    arguments are passed in as Python floats, not buffers, so the
    same module can be reused across schedules without re-wrapping.
    """
    def __init__(
        self,
        flow: nn.Module,
        head: nn.Module,
        head_only: bool = False,
    ):
        super().__init__()
        self.flow = flow
        self.head = head
        self.head_only = head_only

    def forward(
        self,
        x_std: torch.Tensor,
        c_std: torch.Tensor,
        w: torch.Tensor,
        delta_max: float,
        sigma_max: float,
        aux_jacobian_weight: float = 0.0,
        loss_target: str = "w",
        weight_power: int = 1,
        loss_fn: str = "huber",
        huber_delta: float = 1.0,
        detach_pure_in_joint: bool = False,
        detach_target: bool = True,
    ):
        # Polyhead-only (sequential phase 2): flow is frozen, x_std
        # from the data loader is ignored (we sample y from the flow
        # at the MC c).
        if self.head_only:
            return head_only_loss_step(
                self.flow, self.head, c_std, w,
                delta_max=delta_max, sigma_max=sigma_max,
                loss_target=loss_target, weight_power=weight_power,
                loss_fn=loss_fn, huber_delta=huber_delta,
                detach_pure_in_joint=detach_pure_in_joint,
            )
        return joint_loss_step(
            self.flow, self.head, x_std, c_std, w,
            delta_max=delta_max, sigma_max=sigma_max,
            aux_jacobian_weight=aux_jacobian_weight,
            loss_target=loss_target, weight_power=weight_power,
            loss_fn=loss_fn, huber_delta=huber_delta,
            detach_pure_in_joint=detach_pure_in_joint,
            detach_target=detach_target,
        )


def _all_reduce_sum_(t: torch.Tensor, is_dist: bool) -> torch.Tensor:
    """In-place all-reduce(sum) when distributed, otherwise a no-op."""
    if is_dist:
        import torch.distributed as dist
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def _state_dict_to_cpu(module: nn.Module) -> dict:
    """Snapshot ``module.state_dict()`` to CPU memory in one bulk
    pass. Allocates pinned destination buffers (where copy-from
    source is on CUDA), issues all H2D copies onto the current stream
    asynchronously, and blocks once at the end.

    The naive ``{k: v.detach().cpu().clone() for k, v in
    module.state_dict().items()}`` triggers a separate stream sync
    per parameter — for a model with ~100 named parameters this
    serializes ~100 cudaStreamSynchronize calls, each with kernel-
    launch overhead, and dominates the epoch boundary on fast GPUs.
    The bulk path queues all copies and syncs once.
    """
    sd = module.state_dict()
    cpu = {}
    any_cuda = False
    for k, v in sd.items():
        v = v.detach()
        if v.is_cuda:
            any_cuda = True
            dst = torch.empty(
                v.shape, dtype=v.dtype, device="cpu", pin_memory=True,
            )
            dst.copy_(v, non_blocking=True)
            cpu[k] = dst
        else:
            cpu[k] = v.clone()
    if any_cuda:
        torch.cuda.synchronize()
    return cpu


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

def train(
    model: nn.Module,
    inner_flow: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: str,
    log_lines: List[str],
    is_dist: bool = False,
    is_rank0: bool = True,
    checkpoint_path=None,
    stats=None,
    flow_config=None,
    precision: str = "fp32",
    checkpoint_every: int = 1,
    head_config=None,
    inner_head: nn.Module = None,
    patience_rel_threshold: float = 1e-4,
    profile: bool = False,
    profile_warmup: int = 3,
    profile_active: int = 5,
    profile_output: str = None,
    cond_on_smear: bool = False,
    sigma_max_std: torch.Tensor = None,
    cond_smear_zero_fraction: float = 0.5,
    apply_smearing: bool = True,
    no_early_stop: bool = False,
):
    """Train ``model``. ``inner_flow`` is the un-wrapped zuko flow used
    only for state_dict capture (independent of any DDP wrapping of
    ``model``). Weighted NLL metrics are reduced across ranks at each
    epoch boundary so all ranks see the same values.

    When ``checkpoint_path`` is provided (rank 0 only writes), a
    per-epoch snapshot is saved to that path each epoch, overwriting
    the previous epoch's file.

    ``precision`` selects the forward-pass dtype:
      * ``"fp32"`` (default) — no autocast, no GradScaler.
      * ``"bf16"`` — bfloat16 autocast; no GradScaler (bf16's exponent
        range matches FP32).
      * ``"fp16"`` — float16 autocast + GradScaler for loss scaling
        (fp16's narrower exponent range underflows gradients without
        scaling).

    When ``head_config`` is provided the model is expected to be a
    :class:`FlowHeadModel` and is called as ``model(x, c, w,
    delta_max, sigma_max, ...)`` returning the joint loss components
    ``(total, nll, w_loss, aux, w_loss_split)``. The early-stopping
    / checkpointing metric stays the validation NLL — the polyhead
    reconstruction loss is reported alongside but not gated on, so
    the flow's density quality drives selection.
    """
    use_polyhead = head_config is not None
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, patience // 2)
    )

    best_val = float("inf")
    # Per-mode component bests (phase 2 / polyhead only). Used by the
    # patience logic: an epoch counts as "improvement" if either the
    # combined val metric or *any* of the (sh, sm, jt) split
    # components improves vs its own running best. Avoids premature
    # stopping when one noisy component (typically smear) plateaus
    # while another (typically shift) is still descending.
    best_val_split = [float("inf"), float("inf"), float("inf")]
    # Per-σ-component bests for phase 1 with σ-conditioning: the flow
    # NLL is split by event into the σ=0 zero-fraction vs the σ>0
    # smear-augmented fraction, and the patience clock resets if
    # either component improves vs its own running best. Same role as
    # ``best_val_split`` for phase 2 but along the σ-zero axis.
    best_val_nll_split = [float("inf"), float("inf")]
    best_state = None
    best_polyhead_state = None
    no_improve = 0
    # Whether to track the σ=0 / σ>0 NLL split. Only meaningful in
    # phase 1 (flow-only) with σ-conditioning enabled and active
    # (apply_smearing=True). Phase 2 (polyhead) has its own (sh, sm,
    # jt) split. Plain NLL training has no split.
    track_nll_sigma_split = (
        cond_on_smear and apply_smearing and not use_polyhead
    )

    # How often to refresh the tqdm postfix with the running NLL. Each
    # refresh triggers an .item() on GPU scalars (= a H2D sync), which
    # can stall the pipeline if done every batch. Keep it infrequent.
    postfix_every = 50

    amp_device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    if precision == "fp32":
        amp_dtype = torch.float32
        amp_enabled = False
    elif precision == "bf16":
        amp_dtype = torch.bfloat16
        amp_enabled = True
    elif precision == "fp16":
        amp_dtype = torch.float16
        amp_enabled = True
    else:
        raise ValueError(f"unknown precision {precision!r}")
    amp_ctx = lambda: torch.amp.autocast(
        device_type=amp_device_type,
        dtype=amp_dtype,
        enabled=amp_enabled,
    )
    # GradScaler only for fp16 (bf16 has fp32-equivalent exponent
    # range and doesn't need loss scaling). When disabled, all
    # scaler calls below are no-ops.
    scaler = torch.amp.GradScaler(
        amp_device_type, enabled=(precision == "fp16"),
    )

    # Smear-conditioning helper: per-batch, augment ``(x, c) →
    # (x_aug, c_aug)`` so the flow sees ``p(y_smeared | c, σ)``.
    #
    #   * ``cond_on_smear=False``: identity — flow trains on ``p(y | c)``
    #     as before.
    #   * ``cond_on_smear=True, apply_smearing=True`` (phase 1): per
    #     event, with probability ``zero_fraction`` set ``σ_std = 0``;
    #     otherwise sample ``σ_std_d ~ U[0, σ_max_std_d]`` per
    #     dimension. Smear ``x_std`` by ``σ_std ⊙ ε``, ``ε ~ N(0, I)``.
    #     Append ``σ_std`` to ``c_std``.
    #   * ``cond_on_smear=True, apply_smearing=False`` (phase 2 /
    #     polyhead-only): no random smearing — append a zero σ to ``c``
    #     so the polyhead trains against the unsmeared ``σ = 0`` slice
    #     of the conditioned flow. Conditioning width still matches
    #     what the flow was built with.
    if cond_on_smear:
        if sigma_max_std is None:
            raise ValueError(
                "cond_on_smear=True requires sigma_max_std (a per-dim "
                "tensor of standardized σ upper bounds)."
            )
        sigma_max_std_dev = sigma_max_std.to(
            device=device, dtype=torch.float32,
        )
        smear_zero_frac = float(cond_smear_zero_fraction)
    else:
        sigma_max_std_dev = None
        smear_zero_frac = 0.0

    def _augment_x_c(x, c, sample_smear: bool, generator=None):
        """Append σσᵀ pack (always) to c; optionally smear x with σ·ε.

        Smear model: **rank-1 correlated**, matching
        ``train_shift_smear_reweight``. Each event draws a single
        scalar ``ε ~ N(0, 1)`` that scales the entire σ-vector,
        ``y' = y + ε · σ``. The smear covariance is then exactly
        ``σ σᵀ`` (rank-1 PSD), so the σσᵀ pack is the minimal
        sufficient statistic for the conditional density and matches
        the head's training-time perturbation distribution.

        σ-vector sampling: **ball**, matching the head sampler in
        ``sample_perturbations``. For each event:

          * radius ``‖σ‖₂ ~ U[0, σ_max_radius]`` (one scalar draw),
          * direction ``v ∼ Uniform(S^{F−1})`` (Gaussian then
            normalised),
          * ``σ = ‖σ‖ · v``.

        ``σ_max_radius`` is the scalar ``--cond-smear-sigma-max ×
        1.3 oversample`` in standardized target units (taken as
        ``sigma_max_std_dev.max()`` to be conservative if the per-dim
        cap is ever made non-uniform). The σ=0 anchor is provided
        separately by the ``smear_zero_frac`` Bernoulli draw
        (default 50%). The flow's training σ distribution then
        matches the head's ``sample_perturbations`` ball draws, so
        the σ-cond direct target the head queries lies
        in-distribution.

        When ``cond_on_smear`` is False, returns inputs unchanged.
        ``sample_smear=False`` forces ``σ = 0`` (zero pack); used in
        phase-2 head training and during validation when we want a
        deterministic σ = 0 slice.

        The smear covariance ``σσᵀ`` is invariant under the global
        flip ``σ → −σ`` (since ``ε`` and ``−ε`` are equidistributed),
        and the spherical-direction sampler already covers both ``±v``,
        so all signed σ are reached.

        ``generator`` (optional ``torch.Generator``) routes the σ
        and ε draws through a caller-provided RNG. ``run_val`` uses
        this with a fixed-seeded per-pass generator so the validation
        loss is deterministic across epochs (each event sees the
        same σ every epoch), eliminating the σ-resampling noise that
        would otherwise dominate the early-stopping improvement test.

        Returns ``(x_aug, c_aug, is_zero)`` where ``is_zero`` is a
        ``[B]`` bool tensor: True for events drawn from the σ=0
        zero-fraction (or all events when smearing is off), False for
        events drawn with σ ≠ 0. Used by the train loop to split the
        flow NLL into σ=0 vs σ>0 components for per-component early
        stopping.
        """
        B = x.shape[0]
        F = x.shape[1]
        if not cond_on_smear:
            return (
                x, c,
                torch.ones(B, dtype=torch.bool, device=x.device),
            )
        if sample_smear:
            # Ball sampling: ‖σ‖₂ ~ U[0, σ_max_radius], v on S^{F−1}.
            sigma_max_radius = (
                sigma_max_std_dev.max().to(dtype=x.dtype)
            )
            sigma_mag = (
                torch.rand(
                    B, 1, device=x.device, dtype=x.dtype,
                    generator=generator,
                )
                * sigma_max_radius
            )
            v = torch.randn(
                B, F, device=x.device, dtype=x.dtype,
                generator=generator,
            )
            v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-30)
            sigma_std = sigma_mag * v  # [B, F]
            if smear_zero_frac > 0.0:
                zero_mask_2d = (
                    torch.rand(
                        B, 1, device=x.device, dtype=x.dtype,
                        generator=generator,
                    )
                    < smear_zero_frac
                )
                sigma_std = torch.where(
                    zero_mask_2d, torch.zeros_like(sigma_std), sigma_std,
                )
                is_zero = zero_mask_2d.squeeze(-1)
            else:
                is_zero = torch.zeros(
                    B, dtype=torch.bool, device=x.device,
                )
            # Single scalar ε per event (rank-1 correlated smear): all
            # F components share the same ε, so y' − y = ε · σ ~
            # N(0, σσᵀ). Matches the head's _make_y_pert_stack.
            eps = torch.randn(
                B, 1, device=x.device, dtype=x.dtype,
                generator=generator,
            )
            x_aug = x + sigma_std * eps
        else:
            sigma_std = torch.zeros(B, F, device=x.device, dtype=x.dtype)
            x_aug = x
            is_zero = torch.ones(B, dtype=torch.bool, device=x.device)
        sigma_pack = _sigma_pack_outer(sigma_std)  # [B, F·(F+1)/2]
        c_aug = torch.cat([c, sigma_pack], dim=-1)
        return x_aug, c_aug, is_zero

    head_only_mode = (
        bool(head_config.get("head_only", False))
        if head_config is not None else False
    )
    # Track best by val_wmse in polyhead-only mode (val_nll is
    # essentially constant since the flow is frozen — its only
    # variation comes from the per-batch random y sample).
    track_by_wmse = head_only_mode
    phase_label = (
        "polyhead only (frozen flow)" if head_only_mode
        else ("flow + polyhead" if use_polyhead else "flow only")
    )

    def run_val(epoch_idx: int):
        model.eval()
        total_nll = torch.zeros((), device=device)
        total_wloss = torch.zeros((), device=device)
        total_wloss_split = torch.zeros(3, device=device)
        # σ=0 / σ>0 split for phase-1 flow NLL. Order: [zero, nonzero].
        total_nll_sigma_split = torch.zeros(2, device=device)
        wsum_sigma_split = torch.zeros(2, device=device)
        wsum = torch.zeros((), device=device)
        val_bar = tqdm(
            val_loader,
            desc=f"epoch {epoch_idx:03d} val  ",
            leave=False,
            dynamic_ncols=True,
            disable=not is_rank0,
        )
        # Deterministic σ-conditioning draws across val passes:
        # re-seed a per-pass generator with a fixed seed so each event
        # sees the same σ every epoch. Combined with the val loader's
        # ``shuffle=False`` (same batch order each epoch), the val
        # NLL becomes reproducible across epochs and the early-
        # stopping improvement test isn't fighting σ-resampling
        # noise. The training-time σ draws still use the global RNG
        # and remain stochastic across batches/epochs.
        val_gen = None
        if cond_on_smear and apply_smearing:
            val_gen = torch.Generator(device=device)
            val_gen.manual_seed(0)
        with torch.no_grad():
            for x, c, w in val_bar:
                x = x.to(device, non_blocking=True)
                c = c.to(device, non_blocking=True)
                w = w.to(device, non_blocking=True)
                wb = w.sum()
                # Augment (x, c) with smear conditioning. Validation
                # mirrors the training distribution: σ is sampled the
                # same way during phase 1 (apply_smearing=True), or
                # forced to zero in phase 2 (apply_smearing=False).
                x, c, is_zero = _augment_x_c(
                    x, c,
                    sample_smear=apply_smearing,
                    generator=val_gen,
                )
                if use_polyhead:
                    with amp_ctx():
                        # Aux loss off in val — pure flow + polyhead loss.
                        _, nll_w, wloss_w, _, wloss_split = model(
                            x, c, w,
                            head_config["delta_max"],
                            head_config["sigma_max"],
                            0.0,
                            head_config.get("loss_target", "w"),
                            head_config.get("weight_power", 1),
                            head_config.get("loss_fn", "huber"),
                            head_config.get("huber_delta", 1.0),
                            head_config.get(
                                "detach_pure_in_joint", False,
                            ),
                            head_config.get("detach_target", True),
                        )
                    total_nll = total_nll + nll_w.float() * wb
                    total_wloss = total_wloss + wloss_w.float() * wb
                    total_wloss_split = (
                        total_wloss_split + wloss_split.float() * wb
                    )
                else:
                    with amp_ctx():
                        log_p = model(x, c)
                    log_p = log_p.float()
                    # Same per-event NaN/Inf masking as the train
                    # loop, so a few tail-divergent events don't
                    # poison the val NLL (would otherwise break the
                    # early-stopping signal).
                    finite_mask = torch.isfinite(log_p)
                    log_p = torch.where(
                        finite_mask, log_p, torch.zeros_like(log_p),
                    )
                    w_eff = w * finite_mask.to(dtype=log_p.dtype)
                    wb_eff = w_eff.sum()
                    neg_log_p_w = w_eff * (-log_p)
                    total_nll = total_nll + neg_log_p_w.sum()
                    # wsum advances with the finite-event weights, so
                    # the val NLL denominator matches its numerator.
                    wb = wb_eff
                    if track_nll_sigma_split:
                        mask_zero = is_zero.to(dtype=neg_log_p_w.dtype)
                        mask_nz = 1.0 - mask_zero
                        total_nll_sigma_split[0] = (
                            total_nll_sigma_split[0]
                            + (mask_zero * neg_log_p_w).sum()
                        )
                        total_nll_sigma_split[1] = (
                            total_nll_sigma_split[1]
                            + (mask_nz * neg_log_p_w).sum()
                        )
                        wsum_sigma_split[0] = (
                            wsum_sigma_split[0]
                            + (mask_zero * w_eff).sum()
                        )
                        wsum_sigma_split[1] = (
                            wsum_sigma_split[1]
                            + (mask_nz * w_eff).sum()
                        )
                wsum = wsum + wb
        _all_reduce_sum_(total_nll, is_dist)
        _all_reduce_sum_(total_wloss, is_dist)
        _all_reduce_sum_(total_wloss_split, is_dist)
        _all_reduce_sum_(total_nll_sigma_split, is_dist)
        _all_reduce_sum_(wsum_sigma_split, is_dist)
        _all_reduce_sum_(wsum, is_dist)
        wsum_py = wsum.item()
        denom = max(wsum_py, 1e-30)
        split_py = (total_wloss_split / denom).cpu().tolist()
        if track_nll_sigma_split:
            ws = wsum_sigma_split.cpu().tolist()
            ns = total_nll_sigma_split.cpu().tolist()
            sigma_split_py = [
                ns[0] / max(ws[0], 1e-30),
                ns[1] / max(ws[1], 1e-30),
            ]
        else:
            sigma_split_py = None
        return (
            total_nll.item() / denom,
            total_wloss.item() / denom,
            split_py,
            sigma_split_py,
        )

    # Optional ``torch.profiler`` diagnostic pass: run the first
    # ``warmup + active`` training steps under the profiler, print
    # the top hot ops, and exit before the normal epoch loop. Profiles
    # whichever phase is currently invoked (phase 1 if ``not
    # use_polyhead``, phase 2 otherwise). All ranks execute the steps
    # for DDP collective consistency; rank 0 also runs the profiler.
    if profile:
        n_warmup = int(profile_warmup)
        n_active = int(profile_active)
        n_total = n_warmup + n_active
        train_iter = iter(train_loader)

        def _profile_step(_i):
            nonlocal train_iter
            try:
                x, c, w = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, c, w = next(train_iter)
            x = x.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            x, c, _is_zero = _augment_x_c(
                x, c, sample_smear=apply_smearing,
            )
            optimizer.zero_grad(set_to_none=True)
            if use_polyhead:
                with amp_ctx():
                    loss, _nll_w, _wloss_w, _aux_w, _wloss_split = model(
                        x, c, w,
                        head_config["delta_max"],
                        head_config["sigma_max"],
                        head_config.get("aux_jacobian_weight", 0.0),
                        head_config.get("loss_target", "w"),
                        head_config.get("weight_power", 1),
                        head_config.get("loss_fn", "huber"),
                        head_config.get("huber_delta", 1.0),
                        head_config.get(
                            "detach_pure_in_joint", False,
                        ),
                        head_config.get("detach_target", True),
                    )
                loss = loss.float()
            else:
                with amp_ctx():
                    nll = model(x, c, w)
                loss = nll.float()
            scaler.scale(loss).backward()
            if precision == "fp16":
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=10.0,
            )
            scaler.step(optimizer)
            scaler.update()

        if is_rank0:
            output = profile_output
            if is_dist and output:
                import torch.distributed as _dist
                rank_i = _dist.get_rank()
                base, ext = os.path.splitext(output)
                output = f"{base}.rank{rank_i}{ext}"
            _run_profile_pass(
                _profile_step,
                n_warmup=n_warmup,
                n_active=n_active,
                output_path=output,
                label=f" {phase_label}",
            )
        else:
            for _i in range(n_total):
                _profile_step(_i)
        # Return signature: state_dict-or-None, best_val. We didn't
        # train, so best_val is NaN.
        return None, float("nan")

    for epoch in range(1, epochs + 1):
        if is_rank0:
            print(f"epoch {epoch:03d} starting [{phase_label}]", flush=True)
        model.train()
        t0 = time.time()
        # Accumulate on-device scalars; only materialize with .item() at
        # the epoch boundary so the batch loop is sync-free.
        total_nll = torch.zeros((), device=device)
        total_wloss = torch.zeros((), device=device)
        total_wloss_split = torch.zeros(3, device=device)
        # σ=0 / σ>0 NLL split for phase-1 flow training: [zero, nz].
        total_nll_sigma_split = torch.zeros(2, device=device)
        wsum_sigma_split = torch.zeros(2, device=device)
        wsum = torch.zeros((), device=device)
        # NaN/Inf accounting (flow-only path; tail-saturation guard
        # for architectures like GF whose autograd Jacobian can
        # collapse to 0 on out-of-support events).
        n_nonfinite_total = 0
        n_finite_events_total = 0
        n_skipped_batches = 0
        bar = tqdm(
            train_loader,
            desc=f"epoch {epoch:03d} train",
            leave=False,
            dynamic_ncols=True,
            disable=not is_rank0,
        )
        for i, (x, c, w) in enumerate(bar):
            x = x.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            wb = w.sum()
            wsum_b = wb.clamp_min(1e-30)
            # Augment (x, c) with smear conditioning when enabled.
            x, c, is_zero = _augment_x_c(
                x, c, sample_smear=apply_smearing,
            )

            optimizer.zero_grad(set_to_none=True)
            if use_polyhead:
                with amp_ctx():
                    loss, nll_w, wloss_w, _aux_w, wloss_split = model(
                        x, c, w,
                        head_config["delta_max"],
                        head_config["sigma_max"],
                        head_config.get("aux_jacobian_weight", 0.0),
                        head_config.get("loss_target", "w"),
                        head_config.get("weight_power", 1),
                        head_config.get("loss_fn", "huber"),
                        head_config.get("huber_delta", 1.0),
                        head_config.get("detach_pure_in_joint", False),
                        head_config.get("detach_target", True),
                    )
                # Already mean-normalized inside joint_loss_step; keep
                # in fp32 for numerical stability of backward.
                loss = loss.float()
                running_nll_term = nll_w.float() * wb
                running_wloss_term = wloss_w.float() * wb
                running_wloss_split_term = wloss_split.float() * wb
            else:
                with amp_ctx():
                    log_p = model(x, c)
                # Keep the loss in fp32 for numerical stability,
                # regardless of whether log_p came out of autocast in
                # bf16 or fp32.
                log_p = log_p.float()
                # Per-event NaN/Inf masking. Architectures with
                # autograd-based Jacobians (notably GF) can produce
                # ``log p = −∞`` for events that land far out in the
                # tails of standardised space — the inner ``erf``
                # saturates so the Jacobian collapses to 0 and
                # ``log|J| → −∞``. With ``ε ~ N(0, 1)`` smear noise,
                # a single 4–5σ ε draw at large σ can produce such
                # events, and one NaN/Inf in the sum poisons the
                # whole batch loss. Mask them out per-event; if the
                # whole batch ends up non-finite, skip the optimiser
                # step entirely.
                finite_mask = torch.isfinite(log_p)
                n_nonfinite = int((~finite_mask).sum().item())
                if n_nonfinite > 0:
                    n_nonfinite_total = (
                        n_nonfinite_total + n_nonfinite
                    )
                    n_finite_events_total = (
                        n_finite_events_total
                        + int(finite_mask.numel() - n_nonfinite)
                    )
                    log_p = torch.where(
                        finite_mask, log_p, torch.zeros_like(log_p),
                    )
                    finite_mask_f = finite_mask.to(dtype=log_p.dtype)
                    w_eff = w * finite_mask_f
                    wb_eff = w_eff.sum()
                    wsum_b_eff = wb_eff.clamp_min(1e-30)
                else:
                    n_finite_events_total = (
                        n_finite_events_total + int(finite_mask.numel())
                    )
                    w_eff = w
                    wb_eff = wb
                    wsum_b_eff = wsum_b
                neg_log_p_w = w_eff * (-log_p)
                neg_log_p_w_sum = neg_log_p_w.sum()
                loss = neg_log_p_w_sum / wsum_b_eff
                running_nll_term = neg_log_p_w_sum
                running_wloss_term = torch.zeros((), device=device)
                running_wloss_split_term = torch.zeros(3, device=device)
                if track_nll_sigma_split:
                    mask_zero = is_zero.to(dtype=neg_log_p_w.dtype)
                    mask_nz = 1.0 - mask_zero
                    running_nll_sigma_zero = (
                        (mask_zero * neg_log_p_w).sum().detach()
                    )
                    running_nll_sigma_nz = (
                        (mask_nz * neg_log_p_w).sum().detach()
                    )
                    running_wsum_sigma_zero = (
                        (mask_zero * w_eff).sum().detach()
                    )
                    running_wsum_sigma_nz = (
                        (mask_nz * w_eff).sum().detach()
                    )
                # Whole-batch guard: skip backward/step if either
                #   (a) the loss itself is non-finite, or
                #   (b) every event in the batch was non-finite
                #       (masked loss collapses to 0/0 ≈ 0, finite
                #       but devoid of gradient signal).
                # In either case, the optimiser step would corrupt
                # the parameters or do nothing useful.
                all_nonfinite_batch = (n_nonfinite == int(log_p.numel()))
                if (not torch.isfinite(loss)) or all_nonfinite_batch:
                    n_skipped_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    wsum = wsum + wb
                    continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            # Running accumulators stay on device.
            total_nll = total_nll + running_nll_term.detach()
            total_wloss = total_wloss + running_wloss_term.detach()
            total_wloss_split = (
                total_wloss_split + running_wloss_split_term.detach()
            )
            if track_nll_sigma_split:
                total_nll_sigma_split[0] = (
                    total_nll_sigma_split[0] + running_nll_sigma_zero
                )
                total_nll_sigma_split[1] = (
                    total_nll_sigma_split[1] + running_nll_sigma_nz
                )
                wsum_sigma_split[0] = (
                    wsum_sigma_split[0] + running_wsum_sigma_zero
                )
                wsum_sigma_split[1] = (
                    wsum_sigma_split[1] + running_wsum_sigma_nz
                )
            wsum = wsum + wb
            if is_rank0 and (i + 1) % postfix_every == 0:
                # rank-0-local running metrics (not globally reduced;
                # display only). Global versions are at epoch end.
                wsum_py_now = max(wsum.item(), 1e-30)
                running_nll = total_nll.item() / wsum_py_now
                if use_polyhead:
                    running_wloss = total_wloss.item() / wsum_py_now
                    sh, sm, jt = (
                        total_wloss_split / wsum_py_now
                    ).cpu().tolist()
                    bar.set_postfix(
                        nll=f"{running_nll:+.4f}",
                        wmse=f"{running_wloss:.4f}",
                        sh=f"{sh:.3f}",
                        sm=f"{sm:.3f}",
                        jt=f"{jt:.3f}",
                    )
                else:
                    bar.set_postfix(nll=f"{running_nll:+.4f}")

        _all_reduce_sum_(total_nll, is_dist)
        _all_reduce_sum_(total_wloss, is_dist)
        _all_reduce_sum_(total_wloss_split, is_dist)
        _all_reduce_sum_(total_nll_sigma_split, is_dist)
        _all_reduce_sum_(wsum_sigma_split, is_dist)
        _all_reduce_sum_(wsum, is_dist)
        wsum_py = wsum.item()
        denom = max(wsum_py, 1e-30)
        train_nll = total_nll.item() / denom
        train_wloss = total_wloss.item() / denom
        train_split = (total_wloss_split / denom).cpu().tolist()
        if track_nll_sigma_split:
            ws_t = wsum_sigma_split.cpu().tolist()
            ns_t = total_nll_sigma_split.cpu().tolist()
            train_nll_sigma_split = [
                ns_t[0] / max(ws_t[0], 1e-30),
                ns_t[1] / max(ws_t[1], 1e-30),
            ]
        else:
            train_nll_sigma_split = None
        val_nll, val_wloss, val_split, val_nll_sigma_split = run_val(epoch)
        # Track best-by metric: val_nll for flow training, val_wmse
        # for polyhead-only training (val_nll is essentially constant
        # there since the flow is frozen).
        track_metric = val_wloss if track_by_wmse else val_nll
        scheduler.step(track_metric)
        lr_now = optimizer.param_groups[0]["lr"]
        if use_polyhead:
            tr_sh, tr_sm, tr_jt = train_split
            va_sh, va_sm, va_jt = val_split
            line = (
                f"epoch {epoch:03d}  train_nll {train_nll:+.4f}  "
                f"train_wmse {train_wloss:.4f}  "
                f"(sh {tr_sh:.4f} sm {tr_sm:.4f} jt {tr_jt:.4f})  "
                f"val_nll {val_nll:+.4f}  val_wmse {val_wloss:.4f}  "
                f"(sh {va_sh:.4f} sm {va_sm:.4f} jt {va_jt:.4f})  "
                f"lr {lr_now:.2e}  dt {time.time()-t0:.1f}s"
            )
        elif track_nll_sigma_split:
            tr_z, tr_nz = train_nll_sigma_split
            va_z, va_nz = val_nll_sigma_split
            line = (
                f"epoch {epoch:03d}  train_nll {train_nll:+.4f} "
                f"(σ=0 {tr_z:+.4f} σ>0 {tr_nz:+.4f})  "
                f"val_nll {val_nll:+.4f} "
                f"(σ=0 {va_z:+.4f} σ>0 {va_nz:+.4f})  "
                f"lr {lr_now:.2e}  dt {time.time()-t0:.1f}s"
            )
        else:
            line = (
                f"epoch {epoch:03d}  train_nll {train_nll:+.4f}  "
                f"val_nll {val_nll:+.4f}  lr {lr_now:.2e}  "
                f"dt {time.time()-t0:.1f}s"
            )
        # Append NaN-event accounting when active (flow-only path).
        # Quiet by default: only annotate epochs where something
        # actually went wrong.
        if (not use_polyhead) and (
            n_nonfinite_total > 0 or n_skipped_batches > 0
        ):
            n_total_events = n_finite_events_total + n_nonfinite_total
            nf_frac = (
                n_nonfinite_total / max(n_total_events, 1)
            )
            line = (
                f"{line}  nan_evt {n_nonfinite_total} "
                f"({nf_frac*100:.3g}%)"
                + (
                    f"  skipped {n_skipped_batches} batch(es)"
                    if n_skipped_batches > 0
                    else ""
                )
            )
        if is_rank0:
            print(line, flush=True)
        log_lines.append(line)

        # Patience-improvement test:
        #   * Phase 1 (NLL ~ 3.7): absolute threshold 1e-4 — works
        #     well at this scale and matches historical tuning.
        #   * Phase 2 (wmse ~ 1e-2, MSE-style loss): relative
        #     threshold ``patience_rel_threshold · |best_val|`` since
        #     ``1e-4`` absolute is ~1% relative — too strict for
        #     small-valued MSE metrics, masks slow real progress.
        # On the first epoch ``best_val`` is +inf, in which case any
        # finite ``track_metric`` counts.
        if _math.isinf(best_val):
            improved = track_metric < best_val
        elif use_polyhead:
            improved = track_metric < best_val - (
                patience_rel_threshold * abs(best_val)
            )
        else:
            improved = track_metric < best_val - 1e-4

        # Per-component improvement. The patience clock resets on
        # improvement of *any* split component against its own running
        # best, even if the combined metric plateaus — prevents
        # premature stopping when one component is noise-limited while
        # another is still descending. The combined-metric
        # ``best_val`` and ``best_state`` snapshot are still gated on
        # the combined ``improved`` so the saved checkpoint remains
        # the best by the global metric.
        #   * Phase 2 (polyhead): three-way (sh, sm, jt) wmse split,
        #     relative threshold (matches combined wmse threshold).
        #   * Phase 1 (flow w/ σ-cond): two-way (σ=0, σ>0) NLL split,
        #     absolute 1e-4 threshold (matches combined NLL threshold).
        component_improved = False
        if use_polyhead:
            for i, val_comp in enumerate(val_split):
                if _math.isinf(best_val_split[i]):
                    comp_imp = val_comp < best_val_split[i]
                else:
                    comp_imp = val_comp < best_val_split[i] - (
                        patience_rel_threshold
                        * abs(best_val_split[i])
                    )
                if comp_imp:
                    best_val_split[i] = val_comp
                    component_improved = True
        elif track_nll_sigma_split and val_nll_sigma_split is not None:
            for i, val_comp in enumerate(val_nll_sigma_split):
                if _math.isinf(best_val_nll_split[i]):
                    comp_imp = val_comp < best_val_nll_split[i]
                else:
                    comp_imp = val_comp < best_val_nll_split[i] - 1e-4
                if comp_imp:
                    best_val_nll_split[i] = val_comp
                    component_improved = True

        if improved:
            best_val = track_metric
            if is_rank0:
                best_state = _state_dict_to_cpu(inner_flow)
                if inner_head is not None:
                    best_polyhead_state = _state_dict_to_cpu(
                        inner_head,
                    )
        if improved or component_improved:
            no_improve = 0
        else:
            no_improve += 1

        # Per-epoch snapshot on rank 0. Contains the current (latest)
        # flow state — not the best-so-far. best_state is tracked
        # separately above and loaded back into inner_flow at the end.
        do_snapshot = (
            is_rank0
            and checkpoint_path is not None
            and checkpoint_every > 0
            and (epoch % checkpoint_every == 0)
        )
        if do_snapshot:
            current_state = _state_dict_to_cpu(inner_flow)
            ckpt = {
                "epoch": epoch,
                "train_nll": train_nll,
                "val_nll": val_nll,
                "best_val": best_val,
                "no_improve": no_improve,
                "state_dict": current_state,
                "stats": asdict(stats) if stats is not None else None,
                "flow_config": flow_config,
            }
            if use_polyhead:
                ckpt["train_wmse"] = train_wloss
                ckpt["val_wmse"] = val_wloss
                ckpt["train_wmse_split"] = train_split
                ckpt["val_wmse_split"] = val_split
                ckpt["head_config"] = head_config
            if track_nll_sigma_split and train_nll_sigma_split is not None:
                ckpt["train_nll_sigma_split"] = train_nll_sigma_split
                ckpt["val_nll_sigma_split"] = val_nll_sigma_split
            if inner_head is not None:
                ckpt["polyhead_state_dict"] = _state_dict_to_cpu(
                    inner_head,
                )
            torch.save(ckpt, checkpoint_path)

        if no_improve >= patience:
            if no_early_stop:
                # Patience threshold reached but early stopping is
                # disabled — log a notice (only on the first crossing
                # to avoid spamming every subsequent epoch) and
                # continue. The LR scheduler clock is independent and
                # keeps decaying the LR as configured.
                if no_improve == patience:
                    line = (
                        f"would early-stop at epoch {epoch} "
                        f"(best val {best_val:+.4f}); "
                        f"--no-early-stop set, continuing"
                    )
                    if is_rank0:
                        print(line)
                    log_lines.append(line)
            else:
                line = (
                    f"early stopping at epoch {epoch} "
                    f"(best val {best_val:+.4f})"
                )
                if is_rank0:
                    print(line)
                log_lines.append(line)
                break

    if is_rank0 and best_state is not None:
        inner_flow.load_state_dict(best_state)
        if inner_head is not None and best_polyhead_state is not None:
            inner_head.load_state_dict(best_polyhead_state)
    return inner_flow, best_val


# -----------------------------------------------------------------------------
# TorchScript export
# -----------------------------------------------------------------------------

class FlowWrapper(nn.Module):
    """TorchScript-friendly wrapper exposing log_density and score.

    log_density(x, c[, sigma_raw]): returns log p(x|c[, σ]) in the
    *standardized* target space (x is preprocessed). The Jacobian of
    the preprocessing is a constant per dim so adding
    ``sum(-log(target_std))`` converts to the original-space
    log-density.

    score(x, c[, sigma_raw]): returns ``∇_x log p(x|c[, σ])`` in the
    standardized target space. Divide component-wise by ``target_std``
    to map to the original coordinate gradient.

    When the wrapped flow was trained with ``--cond-on-smear``,
    ``sigma_raw`` is a per-dim smear scale (in raw target units) that
    is standardized by ``target_std`` and concatenated to ``c_std``
    before being fed to the flow. ``sigma_raw=None`` falls back to
    ``σ = 0`` — i.e. the unsmeared baseline density. Without smear
    conditioning ``sigma_raw`` is simply ignored.
    """

    def __init__(
        self,
        flow: nn.Module,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        cond_mean: torch.Tensor,
        cond_std: torch.Tensor,
        cond_on_smear: bool = False,
    ):
        super().__init__()
        self.flow = flow
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)
        self.register_buffer("cond_mean", cond_mean)
        self.register_buffer("cond_std", cond_std)
        self.cond_on_smear = bool(cond_on_smear)

    def _standardize_target(self, x_raw: torch.Tensor) -> torch.Tensor:
        return (x_raw - self.target_mean) / self.target_std

    def _standardize_cond(self, c_raw: torch.Tensor) -> torch.Tensor:
        return (c_raw - self.cond_mean) / self.cond_std

    def _augment_cond_with_sigma(
        self, c_std: torch.Tensor, sigma_raw, x_ref: torch.Tensor,
    ) -> torch.Tensor:
        """Return c_std augmented with the F·(F+1)/2-component pack
        of ``σσᵀ`` (in standardized target units), when smear-
        conditioning was enabled at training. ``sigma_raw=None`` is
        treated as ``σ = 0`` (unsmeared baseline)."""
        if not self.cond_on_smear:
            return c_std
        n_features = self.target_std.shape[0]
        if sigma_raw is None:
            sigma_std = torch.zeros(
                c_std.shape[0], n_features,
                device=c_std.device, dtype=c_std.dtype,
            )
        else:
            sigma_std = sigma_raw.to(
                device=c_std.device, dtype=c_std.dtype,
            )
            if sigma_std.dim() == 1:
                sigma_std = sigma_std.unsqueeze(0).expand(
                    c_std.shape[0], -1,
                )
            sigma_std = sigma_std / self.target_std
        sigma_pack = _sigma_pack_outer(sigma_std)
        return torch.cat([c_std, sigma_pack], dim=-1)

    def log_density(
        self,
        x_raw: torch.Tensor,
        c_raw: torch.Tensor,
        sigma_raw: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self._standardize_target(x_raw)
        c = self._standardize_cond(c_raw)
        c = self._augment_cond_with_sigma(c, sigma_raw, x)
        log_p = self.flow(c).log_prob(x)
        # Jacobian correction for target standardization.
        log_p = log_p - torch.log(self.target_std).sum()
        return log_p

    def score(
        self,
        x_raw: torch.Tensor,
        c_raw: torch.Tensor,
        sigma_raw: torch.Tensor = None,
    ) -> torch.Tensor:
        """Return ∇_{x_raw} log p(x_raw | c_raw[, σ])."""
        x_raw = x_raw.detach().requires_grad_(True)
        log_p = self.log_density(x_raw, c_raw, sigma_raw=sigma_raw)
        grad = torch.autograd.grad(log_p.sum(), x_raw, create_graph=False)[0]
        return grad


def export_flow(
    flow: nn.Module,
    stats: PreprocStats,
    outpath_pt: str,
    outpath_ts: str,
    flow_config: dict,
    head: nn.Module = None,
    head_config: dict = None,
):
    """Save the trained wrapper two ways:

    1. ``<outpath_pt>``: plain ``torch.save`` of the whole FlowWrapper
       module. Loading requires Python + zuko + torch. This is the
       canonical checkpoint for offline work (grid precomputation,
       validation).
    2. ``<outpath_ts>``: try ``torch.jit.script``; if it fails (zuko
       flows are not guaranteed to be scriptable), print a note and
       skip. This file is intended for eventual C++/LibTorch inference
       but is best-effort at this stage.

    When ``head`` / ``head_config`` are provided, the head's
    state_dict and config are also stashed in the .pt file
    (TorchScript trace path is unaffected — it still traces only the
    flow's ``log_density``). For backwards compat the saved key
    remains ``polyhead_state_dict`` so existing downstream consumers
    keep working.
    """
    flow = flow.cpu().eval()
    wrapper = FlowWrapper(
        flow=flow,
        target_mean=torch.tensor(stats.target_mean, dtype=torch.float32),
        target_std=torch.tensor(stats.target_std, dtype=torch.float32),
        cond_mean=torch.tensor(stats.cond_mean, dtype=torch.float32),
        cond_std=torch.tensor(stats.cond_std, dtype=torch.float32),
        cond_on_smear=bool(flow_config.get("cond_on_smear", False)),
    )

    payload = {
        "wrapper_state_dict": wrapper.state_dict(),
        "flow_config": flow_config,
        "preproc": asdict(stats),
    }
    if head is not None:
        head_cpu = head.cpu().eval()
        payload["polyhead_state_dict"] = head_cpu.state_dict()
        payload["head_config"] = head_config
    torch.save(payload, outpath_pt)
    print(f"saved wrapper state_dict + config to {outpath_pt}")

    # TorchScript export via torch.jit.trace (records actual tensor
    # ops on an example input). Two adjustments needed for it to
    # succeed on a zuko flow:
    #
    #   (a) zuko's ``gauss_legendre`` is a custom ``torch.autograd.Function``
    #       that the tracer can't record cleanly. Swap it for a plain
    #       torch-op quadrature (``torch.lerp`` + ``torch.tensordot``)
    #       — same forward result, O(n_nodes) slower backward. Since
    #       we're at end-of-training, slower backward is irrelevant.
    #
    #   (b) Script (static type analysis) chokes on zuko's
    #       forward-referenced type annotations. Tracing ignores
    #       annotations entirely, so just don't use ``jit.script``.
    #
    # Limitation: tracing records a straight-through graph — fine for
    # the ``log_density`` forward but skips the ``score`` path
    # (autograd.grad isn't traceable). If C++ score inference is
    # needed, use flow_export_onnx.py's dynamo-based export.
    import zuko.utils
    import zuko.transforms
    _orig_gl_utils = zuko.utils.gauss_legendre
    _orig_gl_transforms = zuko.transforms.gauss_legendre

    def _gauss_legendre_native(f, a, b, n=3, phi=()):
        nodes, weights = zuko.utils.GaussLegendre.nodes(
            n, dtype=a.dtype, device=a.device
        )
        x_nodes = torch.lerp(
            a[..., None], b[..., None], nodes,
        ).movedim(-1, 0)
        return (b - a) * torch.tensordot(weights, f(x_nodes), dims=1)

    zuko.utils.gauss_legendre = _gauss_legendre_native
    zuko.transforms.gauss_legendre = _gauss_legendre_native

    n_features = len(stats.target_mean)
    n_cond = len(stats.cond_mean)
    x_example = torch.randn(1, n_features, dtype=torch.float32)
    c_example = torch.randn(1, n_cond, dtype=torch.float32)
    cond_on_smear = bool(flow_config.get("cond_on_smear", False))

    if cond_on_smear:
        sigma_example = torch.zeros(1, n_features, dtype=torch.float32)

        class _TracedLogDensity(nn.Module):
            def __init__(self, inner: nn.Module):
                super().__init__()
                self.inner = inner

            def forward(self, x_raw, c_raw, sigma_raw):
                return self.inner.log_density(x_raw, c_raw, sigma_raw)

        trace_inputs = (x_example, c_example, sigma_example)
    else:
        class _TracedLogDensity(nn.Module):
            def __init__(self, inner: nn.Module):
                super().__init__()
                self.inner = inner

            def forward(self, x_raw, c_raw):
                return self.inner.log_density(x_raw, c_raw)

        trace_inputs = (x_example, c_example)

    try:
        with torch.no_grad():
            traced = torch.jit.trace(
                _TracedLogDensity(wrapper),
                trace_inputs,
                check_trace=False,
                strict=False,
            )
        traced.save(outpath_ts)
        print(
            f"saved TorchScript (traced, log_density only) to "
            f"{outpath_ts}"
        )
    except Exception as e:
        print(
            f"[note] torch.jit.trace export skipped "
            f"({type(e).__name__}: {e}). "
            f"The .pt wrapper file is the portable alternative."
        )
    finally:
        zuko.utils.gauss_legendre = _orig_gl_utils
        zuko.transforms.gauss_legendre = _orig_gl_transforms


# -----------------------------------------------------------------------------
# In-memory loader
# -----------------------------------------------------------------------------

class InMemoryLoader:
    """Bulk-indexed per-batch iterator over in-RAM CPU tensors.

    Per batch: one bulk ``index_select`` (when shuffling) or a
    contiguous slice (val), then a single H2D copy of the whole batch
    inside the training loop. Avoids DataLoader's per-element path
    through TensorDataset which creates ~batch_size tiny tensors per
    batch.

    With ``pin_memory=True`` (default) each yielded batch is copied
    into pinned (page-locked) host memory before being yielded. This
    is what makes ``tensor.to(device, non_blocking=True)`` actually
    asynchronous — for pageable source memory PyTorch silently falls
    back to a synchronous copy. Skipped automatically if the source
    tensors aren't on CPU (e.g., already on the GPU).

    Two perf knobs that matter for the per-epoch startup gap:

      * ``prefetch_shuffle=True``: the next epoch's
        ``torch.randperm(n)`` is computed in a background thread *while
        the current epoch is still training*. With shuffles taking
        several seconds at ~10⁸-row training sets, this hides the
        epoch-boundary stall. The first epoch still pays the cost
        (no prior epoch to overlap with).

      * ``time_iter=True`` (or ``LOADER_TIME=1`` in env): print
        one-line timing for each iter() call: how long ``randperm`` /
        first-batch gather + pin took. Useful to confirm what's
        actually slow at the epoch boundary without strapping a
        profiler around the training loop.
    """

    def __init__(self, x, c, w, batch_size, shuffle, drop_last,
                 pin_memory=True, prefetch_shuffle=False,
                 time_iter=False):
        self.x, self.c, self.w = x, c, w
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pin_memory = bool(pin_memory) and (x.device.type == "cpu")
        self.n = x.shape[0]
        self.prefetch_shuffle = bool(prefetch_shuffle) and shuffle
        self.time_iter = bool(time_iter) or os.environ.get(
            "LOADER_TIME", "0",
        ) not in ("", "0", "false", "False")

        # Async prefetch state: background thread + a stashed
        # already-computed permutation for the next epoch.
        self._executor = None
        self._next_perm_future = None
        if self.prefetch_shuffle:
            import concurrent.futures as _futures
            self._executor = _futures.ThreadPoolExecutor(max_workers=1)
            # Kick off the *first* epoch's permutation now, so it
            # overlaps with whatever happens between loader
            # construction and the first ``iter()`` call (model build,
            # optimiser setup, torch.compile JIT trace of the first
            # batch, etc.). At ~10⁸ rows the shuffle is ~10–20 s; if
            # model setup eats more than that, the first epoch sees
            # ``randperm: 0.000s`` too.
            self._kick_off_next_perm()

    def __len__(self):
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size

    def _kick_off_next_perm(self):
        """Submit a randperm task to run in the background; the result
        is consumed by the next ``__iter__`` call. No-op if prefetch
        is disabled or a previous future already exists."""
        if (
            self._executor is not None
            and self._next_perm_future is None
        ):
            self._next_perm_future = self._executor.submit(
                torch.randperm, self.n,
            )

    def __iter__(self):
        bs = self.batch_size
        pin = self.pin_memory
        time_iter = self.time_iter

        # Drop a hook so callers can request prefetch-of-next-perm at
        # an opportune moment in the training loop. We also kick it
        # ourselves at the END of the iterator below (most common
        # case: no manual hook needed).
        # Time only the per-iter setup pieces — the per-batch yield
        # times are already visible in the tqdm bar.
        if time_iter:
            t0 = time.perf_counter()

        if self.shuffle:
            # Pick up a previously-prefetched permutation if available;
            # otherwise compute it now (synchronously).
            if self._next_perm_future is not None:
                perm = self._next_perm_future.result()
                self._next_perm_future = None
            else:
                perm = torch.randperm(self.n)
            t_perm = time.perf_counter() if time_iter else 0.0

            # Kick off the *next* epoch's randperm now, so the
            # background thread has the entire current epoch's worth
            # of training time to finish before the next __iter__()
            # call. Submitting at end-of-iter would only give it the
            # tiny gap between iterators.
            self._kick_off_next_perm()

            n_batches = self.n // bs if self.drop_last else (
                (self.n + bs - 1) // bs
            )
            t_first = None
            for i in range(n_batches):
                idx = perm[i * bs:min((i + 1) * bs, self.n)]
                xb = self.x.index_select(0, idx)
                cb = self.c.index_select(0, idx)
                wb = self.w.index_select(0, idx)
                if pin:
                    xb, cb, wb = (
                        xb.pin_memory(),
                        cb.pin_memory(),
                        wb.pin_memory(),
                    )
                if time_iter and t_first is None:
                    t_first = time.perf_counter()
                    print(
                        f"[loader] randperm: {t_perm - t0:.3f}s   "
                        f"first batch (gather + pin): "
                        f"{t_first - t_perm:.3f}s   "
                        f"total to first yield: {t_first - t0:.3f}s",
                        flush=True,
                    )
                yield xb, cb, wb
        else:
            n_batches = self.n // bs if self.drop_last else (
                (self.n + bs - 1) // bs
            )
            t_first = None
            for i in range(n_batches):
                s = i * bs
                e = min(s + bs, self.n)
                xb, cb, wb = self.x[s:e], self.c[s:e], self.w[s:e]
                if pin:
                    xb, cb, wb = (
                        xb.pin_memory(),
                        cb.pin_memory(),
                        wb.pin_memory(),
                    )
                if time_iter and t_first is None:
                    t_first = time.perf_counter()
                    print(
                        f"[loader] (no shuffle) total to first yield: "
                        f"{t_first - t0:.3f}s",
                        flush=True,
                    )
                yield xb, cb, wb

    def __del__(self):
        # Don't leave the background thread alive past loader lifetime.
        ex = getattr(self, "_executor", None)
        if ex is not None:
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass


def _zuko_base_remap(flow_state, target_module):
    """Bidirectional rename of ``base.loc/base.scale`` ↔ ``base._0/_1``.

    zuko has used two naming conventions for its DiagNormal base
    distribution buffers across versions; a checkpoint produced by
    one version may not load directly into a flow built with the
    other. Inspect both naming conventions in ``target_module`` and
    in ``flow_state`` and remap so they agree.

    No-op when both already use the same naming.
    """
    target_keys = set(target_module.state_dict().keys())
    src_keys = set(flow_state.keys())
    OLD = {"base.loc", "base.scale"}
    NEW = {"base._0", "base._1"}
    target_old = OLD & target_keys
    target_new = NEW & target_keys
    src_old = OLD & src_keys
    src_new = NEW & src_keys
    remap: dict = {}
    if target_new and src_old:
        # local is new-style, ckpt is old-style
        remap = {"base.loc": "base._0", "base.scale": "base._1"}
    elif target_old and src_new:
        # local is old-style, ckpt is new-style
        remap = {"base._0": "base.loc", "base._1": "base.scale"}
    if remap:
        flow_state = {remap.get(k, k): v for k, v in flow_state.items()}
    return flow_state


# -----------------------------------------------------------------------------
# Worker (runs once per GPU in distributed mode, once total otherwise)
# -----------------------------------------------------------------------------

def main_worker(
    rank,
    args,
    world_size,
    master_port,
    stats,
    flow_config,
    target_std_t,
    cond_t,
    w_t,
    train_sel,
    val_sel,
):
    is_dist = world_size > 1
    is_rank0 = rank == 0

    # Device selection + optional process-group init.
    if is_dist:
        import torch.distributed as dist
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        if args.device.startswith("cuda"):
            torch.cuda.set_device(rank)
            device = f"cuda:{rank}"
            backend = "nccl"
        else:
            device = "cpu"
            backend = "gloo"
        dist.init_process_group(
            backend=backend, rank=rank, world_size=world_size
        )
    else:
        if args.device.startswith("cuda"):
            torch.cuda.set_device(0)
            device = "cuda:0"
        else:
            device = args.device

    # Per-rank index shard. With world_size==1 this is a no-op.
    if is_dist:
        train_sel_rank = torch.chunk(
            train_sel, world_size
        )[rank].contiguous()
        val_sel_rank = torch.chunk(
            val_sel, world_size
        )[rank].contiguous()
    else:
        train_sel_rank = train_sel
        val_sel_rank = val_sel

    train_x_cpu = target_std_t.index_select(0, train_sel_rank).contiguous()
    train_c_cpu = cond_t.index_select(0, train_sel_rank).contiguous()
    train_w_cpu = w_t.index_select(0, train_sel_rank).contiguous()
    val_x_cpu = target_std_t.index_select(0, val_sel_rank).contiguous()
    val_c_cpu = cond_t.index_select(0, val_sel_rank).contiguous()
    val_w_cpu = w_t.index_select(0, val_sel_rank).contiguous()
    n_train_rank = train_x_cpu.shape[0]
    n_val_rank = val_x_cpu.shape[0]

    train_loader = InMemoryLoader(
        train_x_cpu, train_c_cpu, train_w_cpu,
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        prefetch_shuffle=bool(getattr(args, "prefetch_shuffle", True)),
        time_iter=bool(getattr(args, "time_loader", False)),
    )
    val_loader = InMemoryLoader(
        val_x_cpu, val_c_cpu, val_w_cpu,
        batch_size=args.batch_size, shuffle=False, drop_last=False,
        time_iter=bool(getattr(args, "time_loader", False)),
    )
    if is_rank0:
        print(
            f"rank {rank}/{world_size}  "
            f"train {n_train_rank}  val {n_val_rank}  "
            f"batch {args.batch_size}  device {device}"
        )

    # Per-dim σ-upper-bound in standardized target units. ``train()``
    # uses this to draw σ for the smear-conditioned forwards. Only
    # built when smear conditioning is enabled.
    #
    # ``--cond-smear-sigma-max`` is in standardized target units (same
    # convention as the head's ``--sigma-max``), so the value goes
    # straight in — no division by ``target_std``. The 1.3× oversample
    # matches :func:`sample_perturbations` so the flow's σ-conditioning
    # input distribution covers the head's query range
    # (``sigma_vec_pert`` magnitude ≤ 1.3·sigma_max).
    cond_on_smear = bool(flow_config.get("cond_on_smear", False))
    _flow_smear_oversample = 1.3
    if cond_on_smear:
        sigma_max_std = torch.full(
            (int(flow_config["n_features"]),),
            _flow_smear_oversample
            * float(flow_config["cond_smear_sigma_max"]),
            dtype=torch.float32,
        )
    else:
        sigma_max_std = None
    cond_smear_zero_fraction = float(
        flow_config.get("cond_smear_zero_fraction", 0.5)
    )
    if is_rank0 and cond_on_smear:
        print(
            f"smear conditioning: σ_max_std="
            f"{flow_config['cond_smear_sigma_max']:.3g}  "
            f"oversample={_flow_smear_oversample:.2f}×  "
            f"σ_max_std training range (per-dim, signed)="
            f"{[f'±{x:.3g}' for x in sigma_max_std.tolist()]}  "
            f"zero_fraction={cond_smear_zero_fraction:.3g}  "
            f"n_sigma_pack={flow_config.get('n_sigma_pack', 0)}"
        )

    zuko_flow = build_flow(
        n_features=flow_config["n_features"],
        n_cond=flow_config["n_cond"],
        n_transforms=flow_config["n_transforms"],
        hidden_features=flow_config["hidden_features"],
        n_hidden_layers=flow_config["n_hidden_layers"],
        activation=flow_config.get("activation", "gelu"),
        architecture=flow_config.get("architecture", "realnvp"),
        randmask=flow_config.get("randmask", False),
        gf_components=flow_config.get("gf_components", 8),
        sospf_degree=flow_config.get("sospf_degree", 4),
        sospf_polynomials=flow_config.get("sospf_polynomials", 3),
        sospf_quad_n=flow_config.get("sospf_quad_n", -1),
    ).to(device)

    # If --head-only is requested, load the flow's weights from
    # an existing checkpoint and freeze them. The flow then serves
    # only as a fixed conditional density during polyhead training.
    head_only_mode = bool(args.head_only)
    loaded_head_state = None
    if args.load_flow_checkpoint is not None:
        ckpt = torch.load(
            args.load_flow_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        # Per-epoch checkpoint.pt has "state_dict"; final flow.pt has
        # "wrapper_state_dict" with "flow." prefixed keys.
        if "state_dict" in ckpt:
            flow_state = ckpt["state_dict"]
        elif "wrapper_state_dict" in ckpt:
            flow_state = {
                k[len("flow."):]: v
                for k, v in ckpt["wrapper_state_dict"].items()
                if k.startswith("flow.")
            }
        else:
            raise SystemExit(
                f"checkpoint {args.load_flow_checkpoint!r} has neither "
                "state_dict nor wrapper_state_dict — cannot load flow"
            )
        # Compat shim for zuko base-distribution naming. zuko has
        # used both ``base.loc / base.scale`` (older) and
        # ``base._0 / base._1`` (newer); a checkpoint produced by one
        # version may not load directly into a flow built with the
        # other. Detect which naming the local zuko expects and
        # remap the checkpoint to match.
        flow_state = _zuko_base_remap(flow_state, zuko_flow)
        zuko_flow.load_state_dict(flow_state)
        if "polyhead_state_dict" in ckpt and not args.reset_head:
            loaded_head_state = ckpt["polyhead_state_dict"]
        if is_rank0:
            ph_msg = (
                " (also head state)"
                if loaded_head_state is not None
                else (
                    " (head state in checkpoint ignored — "
                    "--reset-head)"
                    if "polyhead_state_dict" in ckpt
                    and args.reset_head
                    else ""
                )
            )
            print(
                f"loaded flow state from {args.load_flow_checkpoint}"
                + ph_msg
            )
    if head_only_mode:
        if args.load_flow_checkpoint is None:
            raise SystemExit(
                "--head-only requires --load-flow-checkpoint"
            )
        if not args.head:
            raise SystemExit(
                "--head-only implies --head but --no-head "
                "was passed"
            )
        zuko_flow.eval()
        for p in zuko_flow.parameters():
            p.requires_grad_(False)

    head = None
    head_config = None
    if args.head:
        if args.max_deg_sigma % 2 != 0:
            raise SystemExit(
                "--max-deg-sigma must be even (smear-symmetry "
                "constraint). Got "
                f"{args.max_deg_sigma}."
            )
        # Chebyshev basis-scaling: map the trained perturbation range
        # to [-1, 1]. Training samples u ~ U[-oversample · delta_max,
        # +oversample · delta_max] and σ_vec components on the same
        # bracket for non-shift-only mode, so scale_u = oversample ·
        # delta_max etc. ``oversample`` is fixed at 1.3 in
        # ``sample_perturbations``.
        _oversample = 1.3
        activation_cls = ACTIVATIONS.get(
            flow_config.get("activation", "gelu").lower(), nn.GELU,
        )
        if args.head_arch == "polyhead":
            head = PolyHead(
                n_features=flow_config["n_features"],
                n_cond=flow_config["n_cond"],
                hidden_features=args.trunk_hidden,
                n_hidden_layers=args.trunk_layers,
                max_deg_u=args.max_deg_u,
                max_deg_sigma=args.max_deg_sigma,
                max_cross_deg=args.max_cross_deg,
                activation=activation_cls,
                smear_K=args.smear_K,
                smear_residual=args.smear_residual,
                include_smear=args.include_smear,
                positivity=args.positivity,
                basis=args.basis,
                basis_scale_u=_oversample * float(args.delta_max),
                basis_scale_sigma=_oversample * float(args.sigma_max),
            ).to(device)
        elif args.head_arch in ("mlp", "mlp-factored"):
            # Lazy import to avoid the train_shift_smear_reweight.py →
            # train_muon_response_flow.py module-level import cycle.
            from train_shift_smear_reweight import (
                ReweightMLP_B, ReweightMLPFactored, _sigma_pack_indices,
            )
            if args.head_arch == "mlp":
                head = ReweightMLP_B(
                    n_features=flow_config["n_features"],
                    n_cond=flow_config["n_cond"],
                    d_emb=int(args.d_emb),
                    trunk_hidden=int(args.trunk_hidden),
                    trunk_layers=int(args.trunk_layers),
                    head_hidden=int(args.head_hidden),
                    head_layers=int(args.head_layers),
                    activation=activation_cls,
                    shift_only=not bool(args.include_smear),
                ).to(device)
            else:
                head = ReweightMLPFactored(
                    n_features=flow_config["n_features"],
                    n_cond=flow_config["n_cond"],
                    d_emb=int(args.d_emb),
                    trunk_hidden=int(args.trunk_hidden),
                    trunk_layers=int(args.trunk_layers),
                    head_hidden=int(args.head_hidden),
                    head_layers=int(args.head_layers),
                    activation=activation_cls,
                    shift_only=not bool(args.include_smear),
                    detach_pure_shift_in_joint=bool(
                        args.detach_pure_shift_in_joint
                    ),
                    detach_pure_smear_in_joint=bool(
                        args.detach_pure_smear_in_joint
                    ),
                ).to(device)
            # The MLP head doesn't carry the smear-quadrature config
            # natively (PolyHead does). Attach the same fields the
            # joint-loss path reads (smear_K, smear_residual,
            # gh_nodes, gh_log_weights, positivity, include_smear) so
            # both arches expose a uniform interface to
            # ``_compute_target_lw`` / loss step. Σ-pack indices live
            # on the head as buffers so they move with .to(device).
            head.include_smear = bool(args.include_smear)
            head.positivity = str(args.positivity)
            if not head.include_smear:
                # Mirror PolyHead's shift-only collapse: K=1
                # stochastic, no GH, no residual.
                eff_smear_K = 1
                eff_smear_residual = False
            else:
                eff_smear_K = int(args.smear_K)
                eff_smear_residual = bool(args.smear_residual)
                if eff_smear_residual and eff_smear_K <= 1:
                    raise SystemExit(
                        "--smear-residual requires --smear-K > 1."
                    )
            head.smear_K = eff_smear_K
            head.smear_residual = eff_smear_residual
            if eff_smear_K > 1:
                nodes_np, w_np = np.polynomial.hermite_e.hermegauss(
                    eff_smear_K
                )
                w_norm = w_np / np.sqrt(2.0 * np.pi)
                gh_nodes = torch.tensor(nodes_np, dtype=torch.float32)
                gh_log_w = torch.tensor(np.log(w_norm), dtype=torch.float32)
            else:
                gh_nodes = torch.zeros(0, dtype=torch.float32)
                gh_log_w = torch.zeros(0, dtype=torch.float32)
            head.register_buffer(
                "gh_nodes", gh_nodes.to(device), persistent=False,
            )
            head.register_buffer(
                "gh_log_weights", gh_log_w.to(device), persistent=False,
            )
            iu, ju = _sigma_pack_indices(int(flow_config["n_features"]))
            head.register_buffer(
                "sigma_pack_iu", iu.to(device), persistent=False,
            )
            head.register_buffer(
                "sigma_pack_ju", ju.to(device), persistent=False,
            )
        else:
            raise SystemExit(f"unknown --head-arch {args.head_arch!r}")
        if loaded_head_state is not None:
            try:
                head.load_state_dict(loaded_head_state)
                if is_rank0:
                    print("resumed head state from checkpoint.")
            except Exception as e:
                if is_rank0:
                    print(
                        f"[warn] head state in checkpoint is "
                        f"incompatible ({type(e).__name__}); "
                        f"starting head from fresh init."
                    )
        # Resolve --cond-smear-target. 'auto' → 'direct' iff the flow
        # was σ-conditioned (else 'gh'); 'direct' without σ-cond is
        # an error. The chosen path is consumed by ``_compute_target_lw``
        # via attributes on the head module.
        cond_smear_target_choice = str(args.cond_smear_target)
        if cond_smear_target_choice == "auto":
            cond_smear_target_choice = (
                "direct" if cond_on_smear else "gh"
            )
        if cond_smear_target_choice == "direct" and not cond_on_smear:
            raise SystemExit(
                "--cond-smear-target=direct requires --cond-on-smear "
                "(no σ-conditioned flow to query otherwise)."
            )
        head.cond_smear_target_direct = (
            cond_smear_target_choice == "direct"
        )
        head.n_cond_base = int(
            flow_config.get("n_cond_base", flow_config["n_cond"])
        )
        if is_rank0:
            print(
                f"head smear target: {cond_smear_target_choice} "
                f"(cond_on_smear={cond_on_smear})"
            )
        head_config = {
            "head_arch": str(args.head_arch),
            "trunk_hidden": int(args.trunk_hidden),
            "trunk_layers": int(args.trunk_layers),
            # Polyhead-only sizing fields.
            "max_deg_u": int(args.max_deg_u),
            "max_deg_sigma": int(args.max_deg_sigma),
            "max_cross_deg": int(args.max_cross_deg),
            "basis": str(args.basis),
            "basis_scale_u": _oversample * float(args.delta_max),
            "basis_scale_sigma": (
                _oversample * float(args.sigma_max)
            ),
            # MLP-only sizing fields.
            "d_emb": int(args.d_emb),
            "head_hidden": int(args.head_hidden),
            "head_layers": int(args.head_layers),
            # Shared training / loss config.
            "activation": flow_config.get("activation", "gelu"),
            "delta_max": float(args.delta_max),
            "sigma_max": float(args.sigma_max),
            "aux_jacobian_weight": float(args.aux_jacobian_weight),
            "loss_target": str(args.loss_target),
            "weight_power": int(args.weight_power),
            "loss_fn": str(args.loss_fn),
            "huber_delta": float(args.huber_delta),
            "detach_pure_in_joint": bool(
                args.detach_pure_in_joint,
            ),
            "smear_K": int(args.smear_K),
            "smear_residual": bool(args.smear_residual),
            "include_smear": bool(args.include_smear),
            "positivity": str(args.positivity),
            "detach_target": True,
            "head_only": head_only_mode,
            "cond_smear_target": cond_smear_target_choice,
        }

    # ---- Logging headers + bookkeeping setup. ---------------------------
    log_lines: List[str] = []
    precision = str(args.precision)
    if is_rank0:
        n_flow = sum(p.numel() for p in zuko_flow.parameters())
        if head is not None:
            n_head = sum(p.numel() for p in head.parameters())
            print(
                f"flow parameters: {n_flow:,}  "
                f"head parameters: {n_head:,}"
            )
        else:
            print(f"flow parameters: {n_flow:,}")
        log_lines.append(f"flow parameters: {n_flow}")
        log_lines.append(
            f"train {n_train_rank * world_size} "
            f"val {n_val_rank * world_size} "
            f"batch {args.batch_size}  world_size {world_size}"
        )
        if head is not None:
            arch_specific = (
                f"max_deg_u={args.max_deg_u} "
                f"max_deg_sigma={head.max_deg_sigma} "
                f"max_cross_deg={head.max_cross_deg} "
                f"n_joint_basis={head.n_joint_basis} "
                f"basis={head.basis} "
                if args.head_arch == "polyhead" else
                f"d_emb={args.d_emb} "
                f"head_hidden={args.head_hidden} "
                f"head_layers={args.head_layers} "
            )
            log_lines.append(
                f"head[{args.head_arch}]: trunk_hidden={args.trunk_hidden} "
                f"trunk_layers={args.trunk_layers} "
                f"include_smear={head.include_smear} "
                f"{arch_specific}"
                f"delta_max={args.delta_max} "
                f"sigma_max={args.sigma_max if head.include_smear else '(off)'} "
                f"loss_target={args.loss_target} "
                f"weight_power={args.weight_power} "
                f"loss_fn={args.loss_fn} "
                f"huber_delta={args.huber_delta} "
                f"detach_pure_in_joint="
                f"{args.detach_pure_in_joint} "
                f"smear_K={head.smear_K} "
                f"smear_residual={head.smear_residual} "
                f"positivity={args.positivity}"
            )
        arch = flow_config.get("architecture", "realnvp")
        if arch == "realnvp":
            print(
                f"architecture: RealNVP  "
                f"transforms={flow_config['n_transforms']}  "
                f"randmask={flow_config.get('randmask', False)}"
            )
        elif arch == "glow":
            print(
                f"architecture: Glow (realnvp + learned LU mixing)  "
                f"transforms={flow_config['n_transforms']}  "
                f"randmask={flow_config.get('randmask', False)}"
            )
        elif arch == "maf":
            print(
                f"architecture: MAF (masked autoregressive)  "
                f"transforms={flow_config['n_transforms']}"
            )
        elif arch == "gf":
            print(
                f"architecture: GF (Gaussianization)  "
                f"transforms={flow_config['n_transforms']}  "
                f"components={flow_config.get('gf_components', 8)}"
            )
        elif arch == "sospf":
            _qn = int(flow_config.get("sospf_quad_n", -1))
            _quad_n_eff = (
                _qn if _qn > 0 else int(flow_config.get("sospf_degree", 4)) + 1
            )
            _qn_tag = (
                f"{_quad_n_eff}"
                if _qn > 0 else f"{_quad_n_eff} (auto = L+1)"
            )
            print(
                f"architecture: SOSPF (sum-of-squares polynomial)  "
                f"transforms={flow_config['n_transforms']}  "
                f"degree={flow_config.get('sospf_degree', 4)}  "
                f"polynomials={flow_config.get('sospf_polynomials', 3)}  "
                f"quad_n={_qn_tag}"
            )
        print(f"precision: {precision}")

    checkpoint_path = (
        os.path.join(args.output, "checkpoint.pt") if is_rank0 else None
    )

    # ---- Sequential training: phase 1 (flow only) → phase 2 (head). -
    # Phase 1 is skipped when --head-only (flow comes pre-trained from
    # --load-flow-checkpoint). Phase 2 is skipped when --no-head.
    do_phase1 = not head_only_mode
    do_phase2 = head is not None

    best_phase1_val = float("nan")
    best_phase2_val = float("nan")

    if do_phase1:
        if is_rank0:
            print(f"\n=== phase 1: flow training ===", flush=True)
            log_lines.append("=== phase 1: flow training ===")
        flow_model = FlowWithLogProb(zuko_flow).to(device)
        if is_dist:
            if args.device.startswith("cuda"):
                flow_model = nn.parallel.DistributedDataParallel(
                    flow_model, device_ids=[rank]
                )
            else:
                flow_model = nn.parallel.DistributedDataParallel(flow_model)
        inner_flow_p1 = (
            flow_model.module.flow if is_dist else flow_model.flow
        )
        # Compile *after* DDP wrap and after capturing inner refs:
        # checkpoint capture goes through ``inner_flow_p1`` directly,
        # bypassing the OptimizedModule, so state_dict access is
        # unaffected.
        #
        # Architecture-specific torch.compile handling:
        #   * GF: incompatible — its Jacobian is computed via an inner
        #     ``torch.autograd.grad`` call inside zuko's
        #     ``MonotonicTransform.call_and_ladj`` and dynamo cannot
        #     trace that double-autograd path.
        #   * SOSPF: compatible *without CUDA Graphs*. The
        #     ``zuko.utils.gauss_legendre`` quadrature uses a custom
        #     ``torch.autograd.Function`` whose ``nodes()`` method
        #     caches GL nodes/weights and returns the same tensor
        #     across calls; under ``mode='reduce-overhead'`` (CUDA
        #     Graphs) those tensors' memory gets reused and the
        #     CUDA-Graphs runtime raises "accessing tensor output of
        #     CUDAGraphs that has been overwritten by a subsequent
        #     run." Falling back to ``mode='default'`` keeps Inductor
        #     kernel fusion but disables CUDA Graphs, which avoids
        #     the cache-aliasing issue.
        #   * RealNVP / Glow: closed-form analytic Jacobian, use
        #     ``mode='reduce-overhead'`` as before.
        _arch = str(flow_config.get("architecture", "realnvp")).lower()
        _compile_mode = None
        if args.compile:
            if _arch == "gf":
                _compile_mode = None  # skip
            elif _arch == "sospf":
                _compile_mode = "default"
            else:
                _compile_mode = "reduce-overhead"
        if args.compile and _compile_mode is None and is_rank0:
            print(
                f"torch.compile: requested but not supported with "
                f"--architecture {_arch} (double-autograd in "
                f"MonotonicTransform). Falling back to eager.",
                flush=True,
            )
        if _compile_mode is not None:
            if is_rank0:
                _reason = (
                    " (CUDA Graphs disabled — incompatible with "
                    "zuko's Gauss–Legendre node cache)"
                    if _arch == "sospf" else ""
                )
                print(
                    f"torch.compile: enabled (mode={_compile_mode})"
                    f"{_reason} — first batch will JIT-compile",
                    flush=True,
                )
            flow_model = torch.compile(flow_model, mode=_compile_mode)
        zuko_flow_trained, best_phase1_val = train(
            flow_model, inner_flow_p1, train_loader, val_loader,
            args.epochs, args.lr, args.weight_decay, args.patience,
            device, log_lines,
            is_dist=is_dist, is_rank0=is_rank0,
            checkpoint_path=checkpoint_path,
            stats=stats, flow_config=flow_config, precision=precision,
            checkpoint_every=int(args.checkpoint_every),
            head_config=None, inner_head=None,
            patience_rel_threshold=float(args.patience_rel_threshold),
            profile=bool(args.profile),
            profile_warmup=int(args.profile_warmup),
            profile_active=int(args.profile_active),
            profile_output=args.profile_output,
            cond_on_smear=cond_on_smear,
            sigma_max_std=sigma_max_std,
            cond_smear_zero_fraction=cond_smear_zero_fraction,
            apply_smearing=True,
            no_early_stop=bool(args.no_early_stop),
        )
        # zuko_flow now holds the best phase-1 state (train() loaded it
        # in place). Free the DDP wrapper before phase 2 builds its own.
        del flow_model
        if is_rank0:
            print(f"phase 1 best val NLL: {best_phase1_val:+.4f}")
            log_lines.append(f"phase 1 best val_nll: {best_phase1_val:+.4f}")

    if do_phase2:
        # Freeze the (now-trained) flow.
        zuko_flow.eval()
        for p in zuko_flow.parameters():
            p.requires_grad_(False)
        if is_rank0:
            print(
                f"\n=== phase 2: head training "
                f"(frozen flow) ===",
                flush=True,
            )
            log_lines.append(
                "=== phase 2: head training (frozen flow) ==="
            )
        # Force head_only=True for the phase-2 model and config.
        phase2_head_config = dict(head_config)
        phase2_head_config["head_only"] = True
        poly_model = FlowHeadModel(
            zuko_flow, head, head_only=True,
        ).to(device)
        if is_dist:
            if args.device.startswith("cuda"):
                poly_model = nn.parallel.DistributedDataParallel(
                    poly_model, device_ids=[rank]
                )
            else:
                poly_model = nn.parallel.DistributedDataParallel(poly_model)
        inner_flow_p2 = (
            poly_model.module.flow if is_dist else poly_model.flow
        )
        inner_head_p2 = (
            poly_model.module.head if is_dist else poly_model.head
        )
        # Same architecture-specific compile handling as phase 1: GF
        # is skipped entirely (double-autograd in MonotonicTransform);
        # SOSPF falls back to mode='default' (CUDA Graphs disabled —
        # zuko's Gauss–Legendre node cache aliases CUDA-Graphs memory
        # under mode='reduce-overhead'); RealNVP / Glow use the
        # standard 'reduce-overhead' path. Phase 2 forwards still
        # invoke the flow's transform so the same constraints apply.
        _arch_p2 = str(flow_config.get("architecture", "realnvp")).lower()
        _compile_mode_p2 = None
        if args.compile:
            if _arch_p2 == "gf":
                _compile_mode_p2 = None
            elif _arch_p2 == "sospf":
                _compile_mode_p2 = "default"
            else:
                _compile_mode_p2 = "reduce-overhead"
        if args.compile and _compile_mode_p2 is None and is_rank0:
            print(
                f"torch.compile: requested but not supported with "
                f"--architecture {_arch_p2} (double-autograd in "
                f"MonotonicTransform). Falling back to eager.",
                flush=True,
            )
        if _compile_mode_p2 is not None:
            if is_rank0:
                _reason_p2 = (
                    " (CUDA Graphs disabled — incompatible with "
                    "zuko's Gauss–Legendre node cache)"
                    if _arch_p2 == "sospf" else ""
                )
                print(
                    f"torch.compile: enabled (mode={_compile_mode_p2})"
                    f"{_reason_p2} — first batch will JIT-compile",
                    flush=True,
                )
            poly_model = torch.compile(poly_model, mode=_compile_mode_p2)
        _, best_phase2_val = train(
            poly_model, inner_flow_p2, train_loader, val_loader,
            args.epochs, args.lr, args.weight_decay, args.patience,
            device, log_lines,
            is_dist=is_dist, is_rank0=is_rank0,
            checkpoint_path=checkpoint_path,
            stats=stats, flow_config=flow_config, precision=precision,
            checkpoint_every=int(args.checkpoint_every),
            head_config=phase2_head_config,
            inner_head=inner_head_p2,
            patience_rel_threshold=float(args.patience_rel_threshold),
            profile=bool(args.profile),
            profile_warmup=int(args.profile_warmup),
            profile_active=int(args.profile_active),
            profile_output=args.profile_output,
            cond_on_smear=cond_on_smear,
            sigma_max_std=sigma_max_std,
            cond_smear_zero_fraction=cond_smear_zero_fraction,
            apply_smearing=False,
            no_early_stop=bool(args.no_early_stop),
        )
        del poly_model
        if is_rank0:
            print(f"phase 2 best val wmse: {best_phase2_val:.4e}")
            log_lines.append(
                f"phase 2 best val_wmse: {best_phase2_val:.4e}"
            )

    if is_rank0:
        with open(os.path.join(args.output, "training.log"), "w") as f:
            f.write("\n".join(log_lines) + "\n")
        export_flow(
            zuko_flow,
            stats,
            outpath_pt=os.path.join(args.output, "flow.pt"),
            outpath_ts=os.path.join(args.output, "flow_scripted.pt"),
            flow_config=flow_config,
            head=head,
            head_config=head_config,
        )
        print("done")

    if is_dist:
        import torch.distributed as dist
        dist.destroy_process_group()


# -----------------------------------------------------------------------------
# Main (dispatcher: loads data once, then spawns worker(s))
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"loading ntuples from {len(args.input_files)} file(s)")
    (
        eta_r, phi_r, eta_g, phi_g, kappa_r, kappa_g, w,
        _source_id, muon_source,
    ) = load_ntuples(
        args.input_files,
        args.tree,
        args.max_muons,
        args.pt_min,
        args.pt_max,
        args.eta_max,
        threads=args.threads,
        max_events=args.max_events,
        weight_mode=args.weight_handling,
        split=args.shard_split,
        val_fraction=args.val_fraction,
        holdout_fraction=args.shard_holdout_fraction,
    )

    target, cond_raw = compute_targets_and_conditioning(
        eta_r, phi_r, eta_g, phi_g, kappa_r, kappa_g,
        muon_source=muon_source,
    )

    # Mean-normalize weights so the weighted NLL is on the same scale
    # as an unweighted one — keeps learning rate and gradient clip
    # behavior comparable to the unit-weight case.
    w = (w / w.mean()).astype(np.float32)
    print(
        f"target  r_kappa mean={target[:,0].mean():+.3e} "
        f"std={target[:,0].std():.3e}"
    )
    print(
        f"         dlambda mean={target[:,1].mean():+.3e} "
        f"std={target[:,1].std():.3e}"
    )
    print(
        f"         dphi    mean={target[:,2].mean():+.3e} "
        f"std={target[:,2].std():.3e}"
    )

    stats = build_preproc(target, cond_raw)
    target_std, cond = apply_preproc(target, cond_raw, stats)

    with open(os.path.join(args.output, "preproc.json"), "w") as f:
        json.dump(asdict(stats), f, indent=2)

    # Split indices (deterministic from --seed). Workers shard further.
    n_total = target_std.shape[0]
    n_features = target_std.shape[1]
    n_cond = cond.shape[1]
    n_val = int(n_total * args.val_fraction)
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(args.seed)
    perm_all = torch.randperm(n_total, generator=gen)
    val_sel = perm_all[:n_val].contiguous()
    train_sel = perm_all[n_val:].contiguous()

    target_std_t = torch.from_numpy(target_std).contiguous()
    cond_t = torch.from_numpy(cond).contiguous()
    w_t = torch.from_numpy(w).contiguous()
    # Release numpy-backed references; tensors keep their own storage.
    del target_std, cond, w

    # World size: auto-detect GPU count by default.
    if args.device.startswith("cuda"):
        if args.num_gpus == -1:
            world_size = max(1, torch.cuda.device_count())
        else:
            world_size = max(1, args.num_gpus)
    else:
        world_size = 1  # CPU mode: single process

    if args.batch_size is None:
        args.batch_size = 32768 if args.device != "cpu" else 16384

    arch = args.architecture.lower()
    # Smear-conditioning augments c with the F·(F+1)/2 symmetric
    # components of ``σσᵀ`` (the upper-triangular pack — diagonals
    # ``σ_d²`` plus cross products ``σ_d·σ_d'``). σσᵀ is invariant
    # under ``σ → −σ`` so the flow's σ-dependence is *structurally*
    # smear-symmetric — the flow can't distinguish ±σ regardless of
    # what's in the training data, and the head's signed σ_vec at
    # query time hits the same conditioning point as the
    # corresponding |σ| training event.
    n_sigma_pack = (
        int(n_features) * (int(n_features) + 1) // 2
        if args.cond_on_smear else 0
    )
    n_cond_total = int(n_cond) + n_sigma_pack
    flow_config = {
        "flow_type": {
            "realnvp": "RealNVP", "glow": "Glow", "maf": "MAF",
            "gf": "GF", "sospf": "SOSPF",
        }[arch],
        "architecture": arch,
        "n_features": int(n_features),
        "n_cond": int(n_cond_total),
        "n_cond_base": int(n_cond),
        "n_sigma_pack": int(n_sigma_pack),
        "n_transforms": args.n_transforms,
        "hidden_features": args.hidden_features,
        "n_hidden_layers": args.n_hidden_layers,
        "activation": args.activation,
        "randmask": bool(args.randmask),
        "gf_components": int(args.gf_components),
        "sospf_degree": int(args.sospf_degree),
        "sospf_polynomials": int(args.sospf_polynomials),
        "sospf_quad_n": int(args.sospf_quad_n),
        "cond_on_smear": bool(args.cond_on_smear),
        "cond_smear_sigma_max": float(args.cond_smear_sigma_max),
        "cond_smear_zero_fraction": float(args.cond_smear_zero_fraction),
    }

    print(
        f"world_size {world_size}  batch_size {args.batch_size}  "
        f"device {args.device}"
    )
    print(f"train {n_train}  val {n_val}  total {n_total}")

    if world_size == 1:
        main_worker(
            0, args, 1, 0, stats, flow_config,
            target_std_t, cond_t, w_t, train_sel, val_sel,
        )
    else:
        # Share memory so child workers attach rather than copy.
        target_std_t.share_memory_()
        cond_t.share_memory_()
        w_t.share_memory_()
        train_sel.share_memory_()
        val_sel.share_memory_()
        # Pick a free port on localhost for the rendezvous.
        import socket
        sock = socket.socket()
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
        sock.close()
        import torch.multiprocessing as mp
        mp.spawn(
            main_worker,
            args=(
                args, world_size, master_port, stats, flow_config,
                target_std_t, cond_t, w_t, train_sel, val_sel,
            ),
            nprocs=world_size,
            join=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
