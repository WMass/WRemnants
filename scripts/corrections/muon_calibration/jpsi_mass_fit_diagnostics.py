"""Diagnostic plots for the J/ψ unbinned mass-fit calibration.

Inputs:
  --checkpoint <run>/checkpoint_best.pt   model + stats + θ_scale + θ_smear
  --fisher     <run>/fisher_info.pt       Hessian + covariance (optional;
                                          omit → θ ±σ bands skipped)
  --shards     <run>/shards/              per-bucket Arrow files

Outputs (under --output, default ``<checkpoint_dir>/diagnostics/``):
  1. mll_closure_{eta,rho,cosalpha}.png  (+ mc_closure_{eta,rho,cosalpha}.png)
     m_ll histograms (data + MC, weighted) with overlaid model curves
     (signal + Bernstein backgrounds + total mixture). One panel figure per
     slice dimension — columns are the slices: inclusive + |η_+| (eta), and
     the conditional ρ (pt asymmetry) and cos α (opening angle) tertiles.
  2. theta_scale_vs_eta.png
     A, e, M per η-bin with ±1σ error bars (from --fisher: fisher_info.pt or
     bootstrap_cov.pt).
  3. theta_smear_vs_eta.png
     effective a, c per η-bin with ±1σ error bars (from --fisher).
  4. covariance_correlation.png
     full joint covariance (signed-log) + correlation heatmaps over
     θ_scale + the active θ_smear, with the scale/smear block separator.
     (Legacy θ_scale-only files fall back to fisher_correlation.png.)
  5. mll_pulls_inclusive.png  + mll_pulls_eta{0..3}.png
     Per-bin (data − model)/√model histograms; expect ~N(0,1).

The ±1σ bands and the matrix plot require a covariance file via --fisher
(fisher_info.pt from --fisher-info, or bootstrap_cov.pt from --bootstrap).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from typing import List

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — must precede pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from jpsi_mass_arrow_loader import (  # noqa: E402
    JpsiMassArrowLoader,
    discover_shards,
)
from jpsi_mass_model import (  # noqa: E402
    JpsiMassMixtureModel, _event_mll, N_THETA_SCALE_PM, N_THETA_SMEAR_PM,
    SMEAR_VAR_SCALE_A, SMEAR_VAR_SCALE_C, THETA_SCALE_REF,
)
from train_jpsi_mass_fit import _move_batch, _stats_from_dict  # noqa: E402


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_model_from_checkpoint(checkpoint_path: str, device: str):
    """Rebuild the trained model from a checkpoint dict."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    stats = _stats_from_dict(ckpt["stats"])
    model = JpsiMassMixtureModel(
        m_lo=stats.m_lo,
        m_hi=stats.m_hi,
        mll_log_scale=stats.mll_log_scale,
        mll_mean=stats.mll_mean,
        mll_std=stats.mll_std,
        y_event_mean=torch.from_numpy(stats.y_event_mean),
        y_event_std_tensor=torch.from_numpy(stats.y_event_std),
        muon_kin_mean=torch.from_numpy(stats.muon_kin_mean),
        muon_kin_std_tensor=torch.from_numpy(stats.muon_kin_std),
        flow_arch=args.get("flow_arch", "gf"),
        flow_n_transforms=args["flow_n_transforms"],
        flow_hidden_features=args["flow_hidden"],
        flow_n_hidden_layers=args["flow_n_hidden"],
        flow_gf_components=args["gf_components"],
        flow_nsf_bins=args.get("nsf_bins", 8),
        mlp_hidden=args["mlp_hidden"],
        mlp_n_layers=args["mlp_n_layers"],
        smearing_enabled=not args.get("disable_smearing", False),
        scale_enabled=not args.get("disable_scale", False),
        qop_floor_frac=args.get("qop_floor_frac", 0.0),
        smear_fit_params=args.get("smear_fit_params", "both"),
        scale_fit_params=args.get("scale_fit_params", "AM"),
        smear_flow_steps=args.get("smear_flow_steps", 1),
        smear_operator=args.get("smear_operator", "pf_ode"),
        n_gh_nodes=args.get("n_gh_nodes", 8),
        jacobian_form=args.get("jacobian_form", "softlog"),
        smear_param_form=args.get("smear_param_form", "linear"),
        norm_correction=args.get("norm_correction", "none"),
        background_enabled=not bool(args.get("no_background", False)),
        theta_mode=("mlp" if args.get("theta_mlp", False) else "binned"),
        theta_mlp_hidden=args.get("theta_mlp_hidden", 32),
        theta_mlp_layers=args.get("theta_mlp_layers", 2),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, stats, args, ckpt


# ---------------------------------------------------------------------------
# Model evaluation on a loader: per-event signal/bkg densities on a grid
# ---------------------------------------------------------------------------


@torch.no_grad()
@torch.no_grad()
def _tilt_density_on_grid(
    model, batch, idx, m_centers_dev, *, chunk_events: int = 2048,
    n_iter: int = 2,
) -> torch.Tensor:
    """``[len(idx), n_grid]`` log p_s(m_grid | c_e, θ_fit) — the #2 direct-eval
    signal density (``_continuity_logp``) evaluated at each grid mass, exactly
    what the stage-2 fit optimises. ``m_centers_dev`` is the physical bin grid.
    Per event, pt scales as ``pt·(m_grid/m_obs)`` (pt∝m at fixed conditioning)."""
    n = idx.shape[0]
    G = m_centers_dev.shape[0]
    eta = batch["eta_pm"][idx]; q = batch["q_pm"][idx]; b = batch["b_pm"][idx]
    phi = batch["phi_pm"][idx]
    mk = batch["muon_kin_std"][idx]; pt = batch["pt_pm"][idx]
    m_obs = batch["mll"][idx]
    out = torch.empty((n, G), device=m_centers_dev.device, dtype=mk.dtype)
    for start in range(0, n, max(1, chunk_events)):
        end = min(start + chunk_events, n); sub = end - start
        mg = m_centers_dev.view(1, G).expand(sub, G)                # [sub,G]
        scale = (mg / m_obs[start:end].view(sub, 1)).unsqueeze(-1)  # [sub,G,1]
        rep = lambda x: x[start:end].unsqueeze(1).expand(
            sub, G, *x.shape[1:]).reshape(sub * G, *x.shape[1:])
        pt_g = (pt[start:end].unsqueeze(1) * scale).reshape(sub * G, 2)
        lp = model._continuity_logp(
            mg.reshape(-1), rep(mk), pt_g, rep(eta), rep(phi), rep(q), rep(b),
            n_iter=n_iter)
        out[start:end] = lp.reshape(sub, G)
    return out


@torch.no_grad()
def _nominal_density_on_grid(model, batch, idx, m_centers_dev, *, chunk_events=4096):
    """``[len(idx), n_grid]`` log p₀(m_grid | c) — the *nominal* (θ=0) flow
    density, i.e. the stage-1 template with no scale/smear correction. Point
    evaluations of the frozen flow at the grid masses."""
    n = idx.shape[0]; G = m_centers_dev.shape[0]
    mk = batch["muon_kin_std"][idx]
    out = torch.empty((n, G), device=m_centers_dev.device, dtype=mk.dtype)
    for start in range(0, n, max(1, chunk_events)):
        end = min(start + chunk_events, n); sub = end - start
        mg = m_centers_dev.view(1, G).expand(sub, G).reshape(-1)
        mke = mk[start:end].unsqueeze(1).expand(sub, G, mk.shape[-1]).reshape(sub * G, -1)
        out[start:end] = model.log_p_nominal(mg, mke).reshape(sub, G)
    return out


@torch.no_grad()
def _injected_raw_theta(model, inject_scale_phys, inject_smear_phys):
    """Convert physical injection values to RAW θ tensors (shape ``[n_eta, 3]``
    and ``[n_eta, 2]``) that reproduce them through the model's parameterisation.
    Components that were NOT injected are filled with zero physical, i.e. the
    truth in the closure is θ=0 for those (not the fitted value).

    * scale: physical (A, e, M) = θ · THETA_SCALE_REF (linear), so the raw θ
      that gives the injected physical value is ``inject_phys / REF``.
    * smear:
        - ``smear_param_form='linear'``: effective(θ) = θ; raw = inject_phys /
          SMEAR_VAR_SCALE.
        - ``smear_param_form='softplus'``: effective(θ) = softplus(θ); invert
          via raw = log(exp(eff) − 1) (= softplus⁻¹). For zero (un-injected)
          physical, eff is clipped to 1e-10 so the raw value is large-negative
          (~ −23) with softplus(.) ≈ 0 — effectively the no-smear identity."""
    n_eta = model.theta_scale.shape[0]
    if inject_scale_phys is None:
        inject_scale_phys = np.zeros((n_eta, 3), dtype=np.float64)
    if inject_smear_phys is None:
        inject_smear_phys = np.zeros((n_eta, 2), dtype=np.float64)
    ref = np.asarray(THETA_SCALE_REF, dtype=np.float64)
    raw_scale = torch.tensor(inject_scale_phys / ref[None, :], dtype=torch.float32)
    scale = np.asarray([SMEAR_VAR_SCALE_A, SMEAR_VAR_SCALE_C], dtype=np.float64)
    eff = inject_smear_phys / scale[None, :]   # target effective θ_smear
    if getattr(model, "smear_param_form", "linear") == "softplus":
        # softplus⁻¹(y) = log(exp(y) − 1) = log(expm1(y)); clip eff at 1e-10 so
        # zero-injection columns give a large-negative raw value (softplus→0).
        eff_safe = np.clip(eff, 1e-10, None)
        raw_smear = torch.tensor(np.log(np.expm1(eff_safe)), dtype=torch.float32)
    else:
        raw_smear = torch.tensor(eff, dtype=torch.float32)
    return raw_scale, raw_smear


def _degeneracy_eigbasis(k_moments):
    """Global (A,e) and (a,c) degeneracy eigenbases from the curvature moments.

    Built from the O(1)-space Gauss–Newton Fisher of each fitted pair, with the
    response weighted by the SAME reference scales the parameters carry (so the
    eigenvectors live in the O(1) θ coordinates the fit moves in):

      scale (θ_A, θ_e): response (REF_A·1, REF_e·k)  → M_s
      smear (θ_a, θ_c): response (SCALE_A·1, SCALE_C·k²) → M_c

    using the global moments ⟨k⟩,⟨k²⟩,⟨k⁴⟩. (Using the bare correlation
    [[1,ρ],[ρ,1]] instead — a unit-diagonal approximation — slightly rotates
    the eigenvectors and can spuriously put a (θ_A=θ_e) injection at stiff=0;
    the REF-weighted diagonal is the correct measured/degenerate split.) The
    eigenvector with the LARGE eigenvalue is the STIFF (well-measured)
    combination, the small one is the SLOPPY (degenerate) one.

    ``k_moments``: [n_eta, 4] sums (N, Σk, Σk², Σk⁴). Returns
    ``(E_scale, λ_scale, E_smear, λ_smear)`` with each ``E[:, 0]`` = stiff,
    ``E[:, 1]`` = sloppy and λ NORMALISED so the sloppy eigenvalue = 1 (so
    λ_stiff is the stiff/sloppy information ratio); or None if unavailable."""
    if k_moments is None:
        return None
    km = np.asarray(k_moments, dtype=np.float64).reshape(-1, 4)
    Ntot = max(float(km[:, 0].sum()), 1.0)
    k1, k2, k4 = km[:, 1].sum() / Ntot, km[:, 2].sum() / Ntot, km[:, 3].sum() / Ntot
    rA, re_, _ = THETA_SCALE_REF
    sA, sC = SMEAR_VAR_SCALE_A, SMEAR_VAR_SCALE_C
    M_s = np.array([[rA * rA,        -rA * re_ * k1],
                    [-rA * re_ * k1,  re_ * re_ * k2]], dtype=np.float64)
    M_c = np.array([[sA * sA,         sA * sC * k2],
                    [sA * sC * k2,    sC * sC * k4]], dtype=np.float64)

    def _eb(M):
        lam, E = np.linalg.eigh(M)        # ascending
        E, lam = E[:, ::-1], lam[::-1]    # → (stiff, sloppy)
        lam = lam / max(lam[1], 1e-300)   # normalise: sloppy = 1
        for j in range(2):                # sign: dominant component positive
            if E[np.argmax(np.abs(E[:, j])), j] < 0:
                E[:, j] = -E[:, j]
        return E, lam

    Es, ls = _eb(M_s)
    Ec, lc = _eb(M_c)
    return Es, ls, Ec, lc


def _project_whitened(phys, ref, E):
    """Physical (A,e)/(a,c) → O(1) (÷ref) → project onto eigenvectors E.

    ``phys`` last axis is size 2 (any leading shape); ``ref`` is the (2,) O(1)
    reference scale; ``E`` is [2, 2] with columns (stiff, sloppy). Returns the
    same shape with the last axis = (stiff, sloppy) dimensionless coordinates."""
    o1 = np.asarray(phys, dtype=np.float64) / np.asarray(ref, dtype=np.float64)
    return o1 @ E


class _override_theta:
    """Context manager that temporarily sets the model's θ_scale / θ_smear to
    the supplied raw values and restores on exit. Used to evaluate the model
    density at the INJECTED θ values for the validation-closure plots —
    showing where the fitted curve *should* converge to."""

    def __init__(self, model, raw_scale=None, raw_smear=None):
        self.model = model
        self.raw_scale = raw_scale
        self.raw_smear = raw_smear
        self._saved_scale = None
        self._saved_smear = None

    def __enter__(self):
        if self.raw_scale is not None:
            self._saved_scale = self.model.theta_scale.detach().clone()
            with torch.no_grad():
                self.model.theta_scale.copy_(
                    self.raw_scale.to(self.model.theta_scale.device,
                                      self.model.theta_scale.dtype))
        if self.raw_smear is not None:
            self._saved_smear = self.model.theta_smear.detach().clone()
            with torch.no_grad():
                self.model.theta_smear.copy_(
                    self.raw_smear.to(self.model.theta_smear.device,
                                      self.model.theta_smear.dtype))
        return self

    def __exit__(self, *_):
        if self._saved_scale is not None:
            with torch.no_grad():
                self.model.theta_scale.copy_(self._saved_scale)
        if self._saved_smear is not None:
            with torch.no_grad():
                self.model.theta_smear.copy_(self._saved_smear)


def _continuity_mc_fold(model, ptm, etam, phim, qm, bm):
    """Fold MC reco at the fitted θ — the empirical template the model signal
    curve should reproduce. The per-muon PHYSICAL fold: apply the fitted scale
    δqop and an independent Gaussian σ_qop smear to each muon's pt, then
    RECOMPUTE m_ll from the folded 4-vectors. The smear σ_qop uses the fitted
    width coefficients directly via ``model.fold_sigma_qop_pm`` (the signed
    (a, c) clipped at 0). This is the exact operation the continuity density
    approximates (linearized advection + the deterministic width stretch), so
    the closure overlay exposes those approximations. (ρ is a flow condition,
    not histogrammed/binned here, so it is not recomputed — no effect on the
    m_ll closure plots.)"""
    pt_cur = ptm
    if model.scale_enabled:
        AeM = model._scale_AeM_pm(etam, phim, bm)
        dqop_s = model._delta_qop_analytic(AeM, ptm, etam, qm)
        pt_cur = model._apply_scale_pt(ptm, etam, qm, dqop_s, sign=+1.0)
    if model.smearing_enabled:
        sig = model.fold_sigma_qop_pm(pt_cur, etam, phim, bm)
        pt_cur = model.apply_smear_pt(pt_cur, etam, qm, sig, torch.randn_like(sig))
    return _event_mll(pt_cur, etam, phim).detach()


@torch.no_grad()
def evaluate_predictions(
    model: JpsiMassMixtureModel,
    loader: JpsiMassArrowLoader,
    device: str,
    m_centers: torch.Tensor,
    m_grid_std: torch.Tensor,
    bin_width: float,
    *,
    chunk_events: int = 4096,
    max_events: int = 0,
    progress: bool = True,
    seed: int = 42,
    n_iter: int = 2,
    mc_as_data: bool = False,
):
    """Stream the loader once; collect per-event aggregates AND per-event
    × per-bin signal densities for the model curves.

    The signal grid density is the **#2 direct-eval** ``_tilt_density_on_grid``
    (the same forward-folded flow density the fit optimises) and the MC
    comparison is the **per-muon physical fold** (``_continuity_mc_fold``:
    scale δqop + Gaussian σ_qop smear, m_ll recomputed).

    Per data event: store m_ll, w, |η_+|, MLP outputs ``f = (f_0, f_1, f_s)``,
    and ``pred_signal_data[e, j] = p_flow(m_bin_j | y_e, fitted θ_scale+θ_smear)``.
    Per MC event: store m_fold (forward-fold of scale+smear at the fitted
    nuisances), w, |η_+|, and
    ``pred_signal_mc[e, j] = p_flow(m_bin_j | y_e, fitted θ_scale+θ_smear)``.

    Cost: O((N_data + N_mc) × n_grid) flow forwards — bounded by
    ``--grid-chunk-events`` per call and ``--max-events`` per pass.

    Returns dict (1-D unless noted):
      mll_data, w_data, eta_data           — per data event
      f_data                                — [N_data, 3]
      pred_signal_data                      — [N_data, n_grid]
      mll_mc_fold, w_mc, eta_mc            — per MC event (scale+smear fold)
      pred_signal_mc                        — [N_mc, n_grid]
      bin_width                              — scalar passed-through
    """
    # Save + seed RNG so the smearing-kernel ε is deterministic across
    # diagnostic runs.
    cpu_state = torch.random.get_rng_state()
    cuda_state = None
    cuda_avail = device.startswith("cuda") and torch.cuda.is_available()
    if cuda_avail:
        try:
            cuda_state = torch.cuda.get_rng_state(device)
        except Exception:
            cuda_state = None
    torch.manual_seed(seed)
    if cuda_avail:
        torch.cuda.manual_seed_all(seed)

    m_grid_std_dev = m_grid_std.to(device)
    m_centers_dev = m_centers.to(device)
    n_grid = m_centers.shape[0]
    print("  (continuity model: signal curve = #2 direct-eval tilt; "
          "MC = per-muon scale+smear fold with m_ll recomputed)")
    if mc_as_data:
        print("  (validation: simulation routed through the data branch as "
              "pseudo-data for the m_ll closure / pulls, in addition to the "
              "MC-branch closure)")

    def _sig_grid(idx):
        return _tilt_density_on_grid(
            model, batch, idx, m_centers_dev, chunk_events=chunk_events,
            n_iter=n_iter)

    out = {
        "mll_data": [], "w_data": [], "eta_data": [], "f_data": [],
        "pred_signal_data": [],
        "mll_mc_fold": [], "w_mc": [], "eta_mc": [],
        "pred_signal_mc": [],
        # Conditional slice variables (besides |η₊|) for the m_ll closure plots:
        # ρ (pt asymmetry, = muon_kin[-1]) and cos α (3-D opening angle). Both
        # are functions of the conditioning, computed exactly as in
        # plot_param_sensitivity, per data + MC event.
        "rho_data": [], "cosalpha_data": [],
        "rho_mc": [], "cosalpha_mc": [],
        # continuity only: the nominal (θ=0, unshifted/unsmeared) MC + flow,
        # to overlay the stage-1 closure alongside the folded stage-2 one.
        "mll_mc_nominal": [], "pred_nominal_mc": [],
        # validation-with-injection only: the INJECTED pseudo-data m_ll (the
        # closure target the fold should reproduce). Distinct from the nominal
        # whenever a θ/smear injection was replayed into the loader.
        "mll_mc_pseudodata": [],
        # validation-with-injection only: signal density on the grid evaluated
        # at the INJECTED θ values (the closure target curve — where the
        # fitted-θ density should converge to if the fit recovers the truth).
        "pred_signal_mc_at_inj": [],
    }
    # Convert the loader's replayed injection (physical) to RAW θ tensors that
    # produce them through the model — used below to evaluate the model
    # density at the truth.
    inject_scale_phys = getattr(loader, "inject_theta_scale", None)
    inject_smear_phys = getattr(loader, "inject_theta_smear", None)
    has_inj = inject_scale_phys is not None or inject_smear_phys is not None
    if has_inj:
        # Full per-η-bin PHYSICAL injection (0 for non-injected terms — the truth
        # there is θ=0). Used to SET the model's effective per-muon θ for the
        # 'flow at injected θ' curve (works for binned AND MLP).
        n_eta = model.theta_scale.shape[0]
        inj_scale_full = (np.asarray(inject_scale_phys, dtype=np.float64)
                          if inject_scale_phys is not None else np.zeros((n_eta, 3)))
        inj_smear_full = (np.asarray(inject_smear_phys, dtype=np.float64)
                          if inject_smear_phys is not None else np.zeros((n_eta, 2)))

    total_events = 0
    bar = tqdm(loader, desc="eval", disable=not progress, unit="batch")
    try:
        for batch in bar:
            if max_events > 0 and total_events >= max_events:
                break
            batch = _move_batch(batch, device)
            is_data = batch["is_data_mask"]
            is_mc = ~is_data
            # Validation (MC-closure) checkpoints fit θ on simulation as
            # pseudo-data, so route MC through the data branch too. The MC
            # branch always uses the simulation rows, so in that mode the same
            # events feed both closures.
            data_sel = is_mc if mc_as_data else is_data
            mc_sel = is_mc

            if bool(data_sel.any()):
                data_idx = data_sel.nonzero(as_tuple=True)[0]
                if getattr(model, "background_enabled", True):
                    f = model.f_data(batch["muon_kin_std"][data_idx])  # [n_data, 3]
                else:
                    # --no-background: data branch is pure signal. The MLP is
                    # bypassed in data_nll_continuity; mirror that here so the
                    # mixture plot doesn't show whatever the random-init MLP
                    # happens to output. f ≡ [0, 0, 1] → no Bernstein bkg.
                    f = torch.zeros((data_idx.numel(), 3), device=batch["muon_kin_std"].device,
                                    dtype=batch["muon_kin_std"].dtype)
                    f[:, 2] = 1.0
                # Signal density at every bin centre for every data event at the
                # fitted θ: tilt (continuity) or θ-conditioned flow (legacy).
                log_p_grid = _sig_grid(data_idx)  # [n_data, n_grid] log-density (1/GeV)
                out["mll_data"].append(batch["mll"][data_idx].cpu().numpy())
                out["w_data"].append(batch["w"][data_idx].cpu().numpy())
                out["eta_data"].append(batch["eta_pm"][data_idx, 0].cpu().numpy())
                _rho_d, _cosa_d = _rho_cosalpha_np(
                    batch["pt_pm"][data_idx], batch["eta_pm"][data_idx],
                    batch["phi_pm"][data_idx])
                out["rho_data"].append(_rho_d)
                out["cosalpha_data"].append(_cosa_d)
                out["f_data"].append(f.cpu().numpy())
                out["pred_signal_data"].append(log_p_grid.exp().cpu().numpy())

            if bool(mc_sel.any()):
                mc_idx = mc_sel.nonzero(as_tuple=True)[0]
                # Directly shift+smear the MC at the *fitted* θ — the empirical
                # template the model signal curve should reproduce. Use the
                # NOMINAL (pre-injection) pt: in validation the loader replaces
                # pt_pm with the injected/smeared pt (so mll = _event_mll(pt_pm)
                # is consistent), so folding pt_pm again would DOUBLE the
                # injection. pt_pm_nominal is the un-injected pt (== pt_pm when
                # no injection / older loaders without the field).
                ptm = batch.get("pt_pm_nominal", batch["pt_pm"])[mc_idx]
                etam = batch["eta_pm"][mc_idx]
                phim = batch["phi_pm"][mc_idx]
                qm = batch["q_pm"][mc_idx]
                bm = batch["b_pm"][mc_idx]
                mll_fold = _continuity_mc_fold(model, ptm, etam, phim, qm, bm)
                # Signal density on the grid for every MC event (#2 tilt).
                log_p_grid_mc = _sig_grid(mc_idx)  # [n_mc, n_grid]
                out["mll_mc_fold"].append(mll_fold.cpu().numpy())
                out["w_mc"].append(batch["w"][mc_idx].cpu().numpy())
                out["eta_mc"].append(batch["eta_pm"][mc_idx, 0].cpu().numpy())
                # Slice ρ/cos α by the conditioning the model sees (batch pt_pm
                # = injected pt in validation), matching the data side so a given
                # bin compares the same physical region. cos α is pt-independent.
                _rho_m, _cosa_m = _rho_cosalpha_np(
                    batch["pt_pm"][mc_idx], etam, phim)
                out["rho_mc"].append(_rho_m)
                out["cosalpha_mc"].append(_cosa_m)
                out["pred_signal_mc"].append(log_p_grid_mc.exp().cpu().numpy())
                # Closure target: signal density at the INJECTED θ values.
                # Temporarily SET the model's effective per-muon θ to the
                # injection, evaluate the tilt density, restore. Uses
                # _set_theta_output so it works for the MLP too (the old
                # _override_theta set only the binned tensors → no-op for
                # --theta-mlp, leaving this curve meaningless). The fitted curve
                # should converge to this when the closure is good.
                if has_inj:
                    with _set_theta_output(model, inj_scale_full, inj_smear_full):
                        log_p_grid_mc_inj = _sig_grid(mc_idx)
                    out["pred_signal_mc_at_inj"].append(
                        log_p_grid_mc_inj.exp().cpu().numpy())
                # nominal (θ=0): the TRUE un-injected reco mass, recomputed from
                # the (un-injected) per-muon pt — NOT batch["mll"], which carries
                # the replayed validation injection. Plus the untilted flow p₀.
                out["mll_mc_nominal"].append(
                    _event_mll(ptm, etam, phim).cpu().numpy())
                # The injected pseudo-data m_ll (= nominal + replayed injection),
                # i.e. the closure target the fold should reproduce. Equals the
                # nominal when no injection was replayed.
                out["mll_mc_pseudodata"].append(batch["mll"][mc_idx].cpu().numpy())
                out["pred_nominal_mc"].append(
                    _nominal_density_on_grid(
                        model, batch, mc_idx, m_centers_dev,
                        chunk_events=chunk_events).exp().cpu().numpy())

            total_events += int(batch["mll"].shape[0])
            bar.set_postfix_str(f"n_events={total_events:,}")
    finally:
        bar.close()
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)

    for k, lst in out.items():
        if lst:
            out[k] = np.concatenate(lst, axis=0)
        else:
            if k == "f_data":
                out[k] = np.zeros((0, 3))
            elif k in ("pred_signal_data", "pred_signal_mc", "pred_nominal_mc",
                       "pred_signal_mc_at_inj"):
                out[k] = np.zeros((0, n_grid))
            else:
                out[k] = np.zeros((0,))
    out["bin_width"] = bin_width
    out["continuity"] = True
    out["mc_as_data"] = mc_as_data
    # Whether the loader replayed a validation injection into batch["mll"] — so
    # the closure plot draws the injected pseudo-data curve only when it is
    # genuinely distinct from the (un-injected) nominal.
    out["injected"] = bool(
        getattr(loader, "inject_theta_scale", None) is not None
        or getattr(loader, "inject_theta_smear", None) is not None)
    return out


def _bernstein_bin_integrals(m_lo: float, m_hi: float, m_edges: np.ndarray):
    """∫_bin p_0 dm and ∫_bin p_1 dm, closed form.

    p_0(m) = 2(1 − u)/width, p_1(m) = 2u/width, u = (m − m_lo)/width
    so ∫(1 − u) du = u − u²/2 and ∫u du = u²/2 ⇒ each per-bin integral
    is 2·ΔF(u). Returns ``(I_0_per_bin, I_1_per_bin)`` each shape
    ``[len(m_edges) − 1]``.
    """
    width = m_hi - m_lo
    u = (m_edges - m_lo) / width
    F0 = lambda x: x - 0.5 * x * x  # noqa: E731
    F1 = lambda x: 0.5 * x * x      # noqa: E731
    return 2.0 * (F0(u[1:]) - F0(u[:-1])), 2.0 * (F1(u[1:]) - F1(u[:-1]))


def _model_pred_histograms(
    evals, m_edges: np.ndarray, m_lo: float, m_hi: float,
    slice_mask_data,
):
    """``(signal, bkg0, bkg1)`` per-bin predicted counts for one slice.

    signal[j] = Δm · Σ_data w_e · f_s(y_e) · p_flow(m_bin_j | y_e, σ_e)
              (grid eval over data events — flow density at each bin centre)

    bkg_k[j] = (Σ_data w_e · f_k(y_e)) · ∫_bin_j p_k(m) dm
              (analytic Bernstein integrals × MLP mixture weights)
    """
    n_bins = len(m_edges) - 1
    bin_width = float(m_edges[1] - m_edges[0])
    sig = np.zeros(n_bins)
    bk0 = np.zeros(n_bins)
    bk1 = np.zeros(n_bins)

    if evals["mll_data"].size and slice_mask_data.any():
        w_d = evals["w_data"][slice_mask_data]
        f_d = evals["f_data"][slice_mask_data]
        pred = evals["pred_signal_data"][slice_mask_data]  # [n_d, n_bins]
        # Signal: Δm · Σ_e w_e · f_s(y_e) · p_flow_e_at_bin
        weights = (w_d * f_d[:, 2])[:, None]  # [n_d, 1]
        sig = bin_width * (pred * weights).sum(axis=0)
        # Bernstein bkg analytic.
        sum_f0_w = float((w_d * f_d[:, 0]).sum())
        sum_f1_w = float((w_d * f_d[:, 1]).sum())
        I0_bin, I1_bin = _bernstein_bin_integrals(m_lo, m_hi, m_edges)
        bk0 = sum_f0_w * I0_bin
        bk1 = sum_f1_w * I1_bin

    return sig, bk0, bk1


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _save_fig(fig, output_dir: str, stem: str, formats=("png", "pdf"), dpi: int = 110):
    """Write ``fig`` as ``<stem>.<ext>`` for each requested format.
    Returns the list of written paths (for the diagnostic stdout line).
    """
    out_paths = []
    for ext in formats:
        path = os.path.join(output_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=dpi)
        out_paths.append(path)
    plt.close(fig)
    return out_paths


def _chi2_compat_zero(theta: np.ndarray, cov: np.ndarray):
    """χ² for the compatibility of ``theta`` with zero given covariance ``cov``,
    via the eigendecomposition restricted to the POSITIVE-variance directions:
    ``χ² = Σ_{λ_k>0} (v_kᵀθ)² / λ_k``, ``dof = #{λ_k > 0}``. This is a proper
    (≥0) generalised χ² that drops both zero-variance (rank-deficient / pinv)
    and any negative-eigenvalue directions (a non-PD Hessian covariance), so it
    is well-defined for every estimator. Returns ``(chi2, dof, p_value)``."""
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    w, V = np.linalg.eigh(cov)
    wmax = float(w.max()) if w.size else 0.0
    tol = cov.shape[0] * np.finfo(np.float64).eps * max(wmax, 0.0)
    pos = w > tol
    dof = int(pos.sum())
    chi2 = 0.0
    if dof > 0:
        proj = V[:, pos].T @ theta            # θ along the positive-variance dirs
        chi2 = float(np.sum(proj * proj / w[pos]))
    p = float("nan")
    if dof > 0 and np.isfinite(chi2):
        try:
            from scipy.stats import chi2 as _chi2dist
            p = float(_chi2dist.sf(chi2, dof))
        except Exception:
            try:
                from scipy.special import gammaincc
                p = float(gammaincc(dof / 2.0, chi2 / 2.0))
            except Exception:
                p = float("nan")
    return chi2, dof, p


def _rho_cosalpha_np(pt_pm, eta_pm, phi_pm):
    """(ρ, cos α) per event as numpy, from the per-muon pt/η/φ — the same
    conditional quantities used to slice the param-sensitivity plots. ρ is the
    pt asymmetry (= muon_kin[-1]); cos α is the 3-D opening angle (pt cancels):
        cos α = (cos Δφ + sinh η₊ sinh η₋) / (cosh η₊ cosh η₋)."""
    pt = pt_pm.cpu().numpy()
    e = eta_pm.cpu().numpy()
    p = phi_pm.cpu().numpy()
    rho = (pt[:, 0] - pt[:, 1]) / (pt[:, 0] + pt[:, 1])
    num = np.cos(p[:, 0] - p[:, 1]) + np.sinh(e[:, 0]) * np.sinh(e[:, 1])
    cosa = num / (np.cosh(e[:, 0]) * np.cosh(e[:, 1]))
    return rho, cosa


def _select_slice(eta_abs: np.ndarray, slice_def):
    """Boolean mask for a |η| slice (lo, hi) or None for inclusive."""
    if slice_def is None:
        return np.ones_like(eta_abs, dtype=bool)
    lo, hi = slice_def
    return (eta_abs >= lo) & (eta_abs < hi)


def _closure_slice_dims(evals, eta_slice_edges):
    """Slice dimensions shared by the m_ll- and MC-closure PANEL plots. Returns
    a list of ``(prefix, label, fmt, data_vals, mc_vals, columns)`` where
    ``columns`` is the per-panel ``[(tag, slice_def), ...]`` of that figure:
      • ``eta``      — |η₊| with the passed fixed edges, plus a leading
                       inclusive panel (slice_def=None).
      • ``rho``      — ρ (pt asymmetry) adaptive tertiles.
      • ``cosalpha`` — cos α (3-D opening angle) adaptive tertiles.
    ρ / cos α are the conditional discriminants used by plot_param_sensitivity
    (ρ → M; cos α → A vs e and a vs c). Each figure is one slice dimension; the
    panels are its slices."""
    def _tertiles(d, mc):
        v = d if (d is not None and d.size) else mc
        if v is None or v.size == 0:
            return None
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        e = np.unique(np.percentile(v, [0.0, 100.0 / 3.0, 200.0 / 3.0, 100.0]))
        return e if e.size >= 2 else None

    abseta_d = np.abs(evals["eta_data"])
    abseta_m = np.abs(evals["eta_mc"])
    eta_cols = [("inclusive", None)] + [
        (f"eta{i}", (eta_slice_edges[i], eta_slice_edges[i + 1]))
        for i in range(len(eta_slice_edges) - 1)
    ]
    dims = [("eta", "|η₊|", ".1f", abseta_d, abseta_m, eta_cols)]
    for key, prefix, label, fmt in (
        ("rho", "rho", "ρ", ".2f"),
        ("cosalpha", "cosalpha", "cos α", ".3f"),
    ):
        dv = evals.get(f"{key}_data", np.zeros((0,)))
        mv = evals.get(f"{key}_mc", np.zeros((0,)))
        edges = _tertiles(dv, mv)
        if edges is None:
            continue
        cols = [(f"{prefix}{i}", (edges[i], edges[i + 1]))
                for i in range(len(edges) - 1)]
        dims.append((prefix, label, fmt, dv, mv, cols))
    return dims


def plot_mll_closure(
    evals, m_centers_np, eta_slice_edges, m_lo: float, m_hi: float,
    output_dir: str,
):
    """Plot data + forward-folded-MC histograms with overlaid model curves.
    One PANEL FIGURE per slice dimension (columns = slices): inclusive + |η_+|
    (fixed edges), ρ (pt asymmetry) tertiles, and cos α (3-D opening angle)
    tertiles — the conditional discriminants of plot_param_sensitivity (ρ → M;
    cos α → A vs e and a vs c). The model signal curve is the flow density
    evaluated on a per-event × per-bin grid at the fitted θ_scale + θ_smear; the
    green MC curve is the MC forward-folded (scale+smear) at the same fitted
    nuisances — independent estimates of the signal."""
    m_edges = np.concatenate([
        [m_centers_np[0] - evals["bin_width"] / 2],
        m_centers_np[:-1] + evals["bin_width"] / 2,
        [m_centers_np[-1] + evals["bin_width"] / 2],
    ])
    # Validation runs fit θ on simulation as pseudo-data — label accordingly.
    pseudo = bool(evals.get("mc_as_data", False))
    data_label = "MC (pseudo-data)" if pseudo else "data"
    cont = bool(evals.get("continuity", False))

    def _draw_panel(ax, axr, data_mask, mc_mask):
        """Draw one slice into (main, ratio) axes; return (ymax, has_curves)."""
        # Data hist.
        if data_mask.any():
            data_hist, _ = np.histogram(
                evals["mll_data"][data_mask], bins=m_edges,
                weights=evals["w_data"][data_mask])
            ax.errorbar(m_centers_np, data_hist, yerr=np.sqrt(np.abs(data_hist)),
                        fmt="o", color="k", markersize=3, label=data_label, zorder=3)
        else:
            data_hist = np.zeros(m_centers_np.shape[0])
        # Forward-folded MC, scaled to the data signal weight in the slice.
        if mc_mask.any():
            mc_hist_raw, _ = np.histogram(
                evals["mll_mc_fold"][mc_mask], bins=m_edges,
                weights=evals["w_mc"][mc_mask])
            w_mc_sum = float(evals["w_mc"][mc_mask].sum())
            scale = 0.0
            if data_mask.any() and w_mc_sum > 0:
                w_d = evals["w_data"][data_mask]
                f_d = evals["f_data"][data_mask]
                scale = float((w_d * f_d[:, 2]).sum()) / w_mc_sum
            mc_hist = mc_hist_raw * scale
            mc_label = ("MC (shifted+smeared, scaled to signal wt)" if cont
                        else "MC (scale+smear folded, scaled to signal wt)")
            ax.step(m_edges[:-1], mc_hist, where="post", color="C2", lw=1.2,
                    label=mc_label)
        else:
            mc_hist = np.zeros(m_centers_np.shape[0])
        # Model components — analytic bkg + grid-eval signal (tilt / flow).
        signal, bkg0, bkg1 = _model_pred_histograms(
            evals, m_edges, m_lo, m_hi, data_mask)
        total = signal + bkg0 + bkg1
        sig_label = ("model signal (tilt p₀·e^δ at fitted θ)" if cont
                     else "model signal (flow at fitted scale+smear)")
        if total.sum() > 0:
            ax.plot(m_centers_np, total, color="C0", lw=1.5, label="model total")
            ax.plot(m_centers_np, signal, color="C1", lw=1.0, ls="--",
                    label=sig_label)
            ax.fill_between(m_centers_np, 0, bkg0 + bkg1, alpha=0.3, color="C3",
                            label="model bkg (Bernstein)")
            with np.errstate(divide="ignore", invalid="ignore"):
                denom = np.where(total > 0, total, np.nan)
                ratio = data_hist / denom
                ratio_err = np.sqrt(np.abs(data_hist)) / denom
            axr.errorbar(m_centers_np, ratio, yerr=ratio_err, fmt="o",
                         color="k", markersize=3)
        axr.axhline(1.0, color="C0", lw=1)
        ymax = max(float((data_hist + np.sqrt(np.abs(data_hist))).max()),
                   float(mc_hist.max()), float(total.max()))
        return ymax, total.sum() > 0

    for prefix, label, fmt, dv, mv, cols in _closure_slice_dims(
            evals, eta_slice_edges):
        ncol = len(cols)
        fig, axes = plt.subplots(
            2, ncol, figsize=(max(6.0, 4.3 * ncol), 6.2), squeeze=False,
            sharex="col", gridspec_kw={"height_ratios": [3, 1]})
        leg_ax = None
        for ci, (tag, slice_def) in enumerate(cols):
            ax, axr = axes[0, ci], axes[1, ci]
            data_mask = _select_slice(dv, slice_def) if dv.size else np.zeros((0,), bool)
            mc_mask = _select_slice(mv, slice_def) if mv.size else np.zeros((0,), bool)
            if data_mask.sum() == 0 and mc_mask.sum() == 0:
                ax.set_visible(False); axr.set_visible(False); continue
            ymax, ok = _draw_panel(ax, axr, data_mask, mc_mask)
            if ok and leg_ax is None:
                leg_ax = ax
            ttl = ("inclusive" if slice_def is None
                   else f"{label} ∈ [{slice_def[0]:{fmt}}, {slice_def[1]:{fmt}}]")
            ax.set_title(ttl, fontsize=9)
            if ymax > 0:
                ax.set_ylim(0, ymax * 1.25)
            if ci == 0:
                ax.set_ylabel("events / bin (weighted)")
                axr.set_ylabel(f"{'pseudo-data' if pseudo else 'data'} / model")
            axr.set_xlabel("m_ll [GeV]"); axr.set_ylim(0.6, 1.4)
        if leg_ax is not None:
            h, l = leg_ax.get_legend_handles_labels()
            fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.945),
                       ncol=len(l), fontsize=8, framealpha=0.9)
        fig.suptitle(f"m_ll closure{' (MC pseudo-data)' if pseudo else ''} "
                     f"— slices of {label}", y=0.998)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        for p in _save_fig(fig, output_dir, f"mll_closure_{prefix}"):
            print(f"  wrote {p}")


def plot_theta_vs_eta(
    theta: np.ndarray,       # [n_eta, n_comp]
    sigma: "np.ndarray | None",  # [n_eta, n_comp] or None
    component_names: List[str],
    name: str,
    eta_edges: np.ndarray,    # [n_eta + 1]
    output_dir: str,
    edm: "float | None" = None,
    chi2_info=None,
    ref: "np.ndarray | None" = None,   # [n_eta, n_comp] reference (e.g. injected)
    band: "np.ndarray | None" = None,  # [n_eta, n_comp] φ-std for shaded band
    slices: "np.ndarray | None" = None,    # [n_eta, n_slices, n_comp] φ-slices
    slice_labels: "list | None" = None,    # length n_slices, e.g. ['φ=0', ...]
):
    n_eta, n_comp = theta.shape
    eta_centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])

    fig, axes = plt.subplots(n_comp, 1, sharex=True, figsize=(8, 2.5 * n_comp))
    if n_comp == 1:
        axes = [axes]
    slice_colors = ["C0", "C1", "C2", "C4", "C5"]   # skip C3 (reserved for ref)
    for i, ax in enumerate(axes):
        # φ-spread band (MLP mode only — `band` carries the std over φ at each η).
        if band is not None:
            ax.fill_between(
                eta_centers, theta[:, i] - band[:, i], theta[:, i] + band[:, i],
                color="0.5", alpha=0.18, label="±1σ over φ", linewidth=0)
        # Individual φ slices (faint coloured lines).
        if slices is not None:
            n_slices = slices.shape[1]
            for s_i in range(n_slices):
                lab = (slice_labels[s_i] if slice_labels and s_i < len(slice_labels)
                       else f"slice {s_i}")
                ax.plot(eta_centers, slices[:, s_i, i],
                        color=slice_colors[s_i % len(slice_colors)],
                        lw=0.9, alpha=0.6, label=lab)
        # Main: error bars (binned θ has ±σ) or φ-mean line (MLP).
        main_label = "fit" if band is None else "fit (φ-mean)"
        if sigma is not None:
            ax.errorbar(
                eta_centers, theta[:, i], yerr=sigma[:, i],
                fmt="o", color="k", markersize=4, capsize=2, label=main_label,
            )
        else:
            ax.plot(eta_centers, theta[:, i], "o-", color="k", markersize=4,
                    label=main_label)
        ax.axhline(0, color="0.5", lw=0.8, ls=":")
        if ref is not None:
            ax.plot(eta_centers, ref[:, i], color="C3", ls="--", lw=1.3,
                    label="injected")
        ax.set_ylabel(component_names[i])
        ax.grid(True, alpha=0.3)
    if ref is not None or slices is not None or band is not None:
        axes[0].legend(loc="best", fontsize=7, ncol=max(1,
            (1 + int(ref is not None) + int(band is not None)
             + (slices.shape[1] if slices is not None else 0)) // 4 + 1))
    axes[-1].set_xlabel("η-bin center")
    title = name
    if chi2_info is not None:
        chi2, dof, p = chi2_info
        cmp = "injected" if ref is not None else "0"
        title += (f"   (vs {cmp}: χ²/dof = {chi2:.1f}/{dof} = {chi2 / max(dof, 1):.2f}, "
                  f"p = {p:.3g})")
    axes[0].set_title(title)
    if edm is not None:
        fig.text(0.995, 0.005, f"EDM = {edm:.2e}", ha="right", va="bottom",
                 fontsize=8, color="0.4")
    fig.tight_layout()
    for p in _save_fig(fig, output_dir, name):
        print(f"  wrote {p}")


def plot_fisher_correlation(cov: np.ndarray, output_dir: str):
    """72×72 correlation heatmap, with η-bin grid lines + (A,e,M) tick labels."""
    n = cov.shape[0]
    d = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(d, d)
    corr = np.where(np.isfinite(corr), corr, 0.0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    # η-bin grid lines (3 cols per η-bin).
    for k in range(3, n, 3):
        ax.axhline(k - 0.5, color="k", lw=0.3, alpha=0.4)
        ax.axvline(k - 0.5, color="k", lw=0.3, alpha=0.4)
    ax.set_xticks(np.arange(1, n, 6))
    ax.set_xticklabels([f"η{j//3}" for j in range(1, n, 6)], fontsize=7, rotation=90)
    ax.set_yticks(np.arange(1, n, 6))
    ax.set_yticklabels([f"η{j//3}" for j in range(1, n, 6)], fontsize=7)
    ax.set_title("Fisher correlation matrix (A, e, M per η-bin)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    for p in _save_fig(fig, output_dir, "fisher_correlation"):
        print(f"  wrote {p}")


def plot_cov_corr(cov: np.ndarray, labels, n_scale: int, output_dir: str,
                  edm: "float | None" = None):
    """Side-by-side covariance (symmetric-log) + correlation ([-1,1]) heatmaps of
    the FULL joint parameter covariance (θ_scale + the active θ_smear), with the
    scale/smear block separator and sparse per-parameter tick labels.

    The covariance mixes units across blocks (A, e, M, smear), spanning many
    orders of magnitude, so it is shown on a signed-log (SymLog) colour scale;
    the correlation is the dimensionless, directly-readable companion.
    """
    import matplotlib.colors as mcolors

    n = cov.shape[0]
    d = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(d, d)
    corr = np.where(np.isfinite(corr), corr, 0.0)

    fig, (axc, axr) = plt.subplots(1, 2, figsize=(15, 6.5))
    amax = float(np.abs(cov).max()) or 1.0
    norm = mcolors.SymLogNorm(linthresh=amax * 1e-6, vmin=-amax, vmax=amax, base=10)
    im0 = axc.imshow(cov, cmap="RdBu_r", norm=norm)
    axc.set_title("covariance (signed log)")
    fig.colorbar(im0, ax=axc, fraction=0.046, pad=0.04)
    im1 = axr.imshow(corr, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    axr.set_title("correlation")
    fig.colorbar(im1, ax=axr, fraction=0.046, pad=0.04)

    # Sparse ticks straight from the parameter labels (~16 across the axis).
    if labels is not None and len(labels) == n:
        step = max(1, n // 16)
        ticks = list(range(0, n, step))
        tlabels = [labels[i] for i in ticks]
    else:
        ticks, tlabels = [], []
    for ax in (axc, axr):
        if 0 < n_scale < n:   # separate the θ_scale and θ_smear blocks
            ax.axhline(n_scale - 0.5, color="k", lw=1.0)
            ax.axvline(n_scale - 0.5, color="k", lw=1.0)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tlabels, fontsize=6, rotation=90)
        ax.set_yticks(ticks)
        ax.set_yticklabels(tlabels, fontsize=6)
    suptitle = "parameter covariance / correlation (θ_scale + active θ_smear)"
    if edm is not None:
        suptitle += f"    EDM = {edm:.2e}"
    fig.suptitle(suptitle)
    fig.tight_layout()
    for p in _save_fig(fig, output_dir, "covariance_correlation"):
        print(f"  wrote {p}")


def plot_mc_closure(
    evals, m_centers_np, eta_slice_edges, output_dir: str,
):
    """MC closure: forward-folded-MC histogram (points) vs flow-density
    curve (line), both at the fitted scale + smearing. One PANEL FIGURE per
    slice dimension (columns = slices): inclusive + |η_+| (fixed edges), ρ (pt
    asymmetry) tertiles, cos α (3-D opening angle) tertiles.

    Empirical: histogram of m_fold_e — the MC reco forward-folded through
    scale then smearing at the fitted θ_scale / θ_smear — weighted by w_mc.

    Model curve: ``Δm · Σ_e w_e · p_flow(m_bin_j | y_e, fitted θ_scale+θ_smear)`` —
    per-event × per-bin flow density at the fitted nuisances, summed over
    MC events. The two should agree when the flow has correctly learned the
    forward-fold distribution; this is the analogue of the dashed orange
    "model signal" curve on the data plot (no Bernstein bkg, MC is
    signal-only).
    """
    bin_width = float(m_centers_np[1] - m_centers_np[0])
    m_edges = np.concatenate([
        [m_centers_np[0] - bin_width / 2],
        m_centers_np[:-1] + bin_width / 2,
        [m_centers_np[-1] + bin_width / 2],
    ])
    cont = bool(evals.get("continuity", False))
    injected = bool(evals.get("injected", False))

    def _draw_panel(ax, axr, mc_mask):
        """Draw one MC-closure slice into (main, ratio) axes; return ymax."""
        mc_hist, _ = np.histogram(
            evals["mll_mc_fold"][mc_mask], bins=m_edges,
            weights=evals["w_mc"][mc_mask])
        w = evals["w_mc"][mc_mask][:, None]
        model_curve = bin_width * (evals["pred_signal_mc"][mc_mask] * w).sum(axis=0)
        mc_label = ("MC (shifted+smeared, fitted θ)" if cont
                    else "MC (scale+smear folded)")
        model_label = ("flow (folded, tilt at fitted θ)" if cont
                       else "flow (at fitted scale+smear)")
        has_nom = cont and evals.get("mll_mc_nominal", np.zeros((0,))).size > 0
        if has_nom:
            nom_hist, _ = np.histogram(
                evals["mll_mc_nominal"][mc_mask], bins=m_edges,
                weights=evals["w_mc"][mc_mask])
            nom_curve = bin_width * (evals["pred_nominal_mc"][mc_mask] * w).sum(axis=0)
        show_pseudo = (cont and injected
                       and evals.get("mll_mc_pseudodata", np.zeros((0,))).size > 0)
        if show_pseudo:
            pseudo_hist, _ = np.histogram(
                evals["mll_mc_pseudodata"][mc_mask], bins=m_edges,
                weights=evals["w_mc"][mc_mask])
        show_inj_curve = (cont and injected
                          and evals.get("pred_signal_mc_at_inj", np.zeros((0,))).size > 0)
        if show_inj_curve:
            inj_curve = bin_width * (
                evals["pred_signal_mc_at_inj"][mc_mask] * w).sum(axis=0)

        if has_nom:
            ax.step(m_edges[:-1], nom_hist, where="post", color="0.6", lw=1.0,
                    label="MC (nominal, θ=0)", zorder=2)
            ax.plot(m_centers_np, nom_curve, color="C0", ls=":", lw=1.3,
                    label="flow (nominal p₀, θ=0)", zorder=2)
        if show_pseudo:
            ax.step(m_edges[:-1], pseudo_hist, where="post", color="C2", lw=1.3,
                    label="pseudo-data (injected θ)", zorder=2)
        ax.errorbar(m_centers_np, mc_hist, yerr=np.sqrt(np.abs(mc_hist)),
                    fmt="o", color="k", markersize=3, label=mc_label, zorder=3)
        ax.plot(m_centers_np, model_curve, color="C1", ls="--", lw=1.5,
                label=model_label)
        if show_inj_curve:
            ax.plot(m_centers_np, inj_curve, color="C3", ls="-.", lw=1.3,
                    label="flow (at INJECTED θ)", zorder=2)

        # Ratio panel: everything relative to the folded flow (the model).
        denom = np.where(model_curve > 0, model_curve, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = mc_hist / denom
            ratio_err = np.sqrt(np.abs(mc_hist)) / denom
        axr.errorbar(m_centers_np, ratio, yerr=ratio_err, fmt="o",
                     color="k", markersize=3, zorder=3)
        if show_pseudo:
            with np.errstate(divide="ignore", invalid="ignore"):
                axr.step(m_edges[:-1], pseudo_hist / denom, where="post",
                         color="C2", lw=1.3, zorder=2)
        if show_inj_curve:
            with np.errstate(divide="ignore", invalid="ignore"):
                axr.plot(m_centers_np, inj_curve / denom, color="C3", ls="-.",
                         lw=1.3, zorder=2)
        if has_nom:
            with np.errstate(divide="ignore", invalid="ignore"):
                axr.step(m_edges[:-1], nom_hist / denom, where="post",
                         color="0.6", lw=1.0, zorder=2)
                axr.plot(m_centers_np, nom_curve / denom, color="C0", ls=":",
                         lw=1.3, zorder=2)
        axr.axhline(1.0, color="C1", lw=1)   # folded flow (model) = reference
        axr.set_ylim(0.6, 1.4)
        return max(float((mc_hist + np.sqrt(np.abs(mc_hist))).max()),
                   float(model_curve.max()),
                   float(nom_hist.max()) if has_nom else 0.0,
                   float(nom_curve.max()) if has_nom else 0.0,
                   float(pseudo_hist.max()) if show_pseudo else 0.0,
                   float(inj_curve.max()) if show_inj_curve else 0.0)

    for prefix, label, fmt, _dv, mv, cols in _closure_slice_dims(
            evals, eta_slice_edges):
        ncol = len(cols)
        fig, axes = plt.subplots(
            2, ncol, figsize=(max(6.0, 4.3 * ncol), 6.2), squeeze=False,
            sharex="col", gridspec_kw={"height_ratios": [3, 1]})
        leg_ax = None
        for ci, (tag, slice_def) in enumerate(cols):
            ax, axr = axes[0, ci], axes[1, ci]
            mc_mask = _select_slice(mv, slice_def) if mv.size else np.zeros((0,), bool)
            if mc_mask.sum() == 0:
                ax.set_visible(False); axr.set_visible(False); continue
            ymax = _draw_panel(ax, axr, mc_mask)
            if leg_ax is None:
                leg_ax = ax
            ttl = ("inclusive" if slice_def is None
                   else f"{label} ∈ [{slice_def[0]:{fmt}}, {slice_def[1]:{fmt}}]")
            ax.set_title(ttl, fontsize=9)
            if ymax > 0:
                ax.set_ylim(0, ymax * 1.25)
            if ci == 0:
                ax.set_ylabel("events / bin (weighted)")
                axr.set_ylabel("ratio to folded flow")
            axr.set_xlabel("m_ll [GeV]")
        if leg_ax is not None:
            h, l = leg_ax.get_legend_handles_labels()
            fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.945),
                       ncol=min(len(l), 4), fontsize=8, framealpha=0.9)
        fig.suptitle(f"MC closure — slices of {label}", y=0.998)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        for p in _save_fig(fig, output_dir, f"mc_closure_{prefix}"):
            print(f"  wrote {p}")


def plot_pulls(
    evals, m_centers_np, eta_slice_edges, m_lo: float, m_hi: float,
    output_dir: str,
):
    """Per-bin (data − model)/√model pulls; expect ~N(0,1) if model is OK."""
    pseudo = bool(evals.get("mc_as_data", False))
    label = "pseudo-data" if pseudo else "data"
    m_edges = np.concatenate([
        [m_centers_np[0] - evals["bin_width"] / 2],
        m_centers_np[:-1] + evals["bin_width"] / 2,
        [m_centers_np[-1] + evals["bin_width"] / 2],
    ])

    slices = [("inclusive", None)] + [
        (f"eta{i}", (eta_slice_edges[i], eta_slice_edges[i + 1]))
        for i in range(len(eta_slice_edges) - 1)
    ]

    for tag, slice_def in slices:
        data_mask = (
            _select_slice(np.abs(evals["eta_data"]), slice_def)
            if evals["eta_data"].size else np.zeros((0,), bool)
        )
        mc_mask = (
            _select_slice(np.abs(evals["eta_mc"]), slice_def)
            if evals["eta_mc"].size else np.zeros((0,), bool)
        )
        if data_mask.sum() == 0:
            continue

        w_sel = evals["w_data"][data_mask]
        data_hist, _ = np.histogram(
            evals["mll_data"][data_mask], bins=m_edges, weights=w_sel,
        )
        signal_curve, p0_curve, p1_curve = _model_pred_histograms(
            evals, m_edges, m_lo, m_hi, data_mask,
        )
        total_curve = signal_curve + p0_curve + p1_curve

        with np.errstate(divide="ignore", invalid="ignore"):
            pulls = (data_hist - total_curve) / np.sqrt(
                np.where(total_curve > 0, total_curve, np.nan)
            )
        pulls_finite = pulls[np.isfinite(pulls)]

        fig, (ax_b, ax_h) = plt.subplots(1, 2, figsize=(10, 4))
        ax_b.stem(m_centers_np, pulls, markerfmt="ko", basefmt="grey", linefmt="k-")
        ax_b.axhline(0, color="grey", lw=0.5)
        ax_b.set_xlabel("m_ll [GeV]")
        ax_b.set_ylabel(f"({label} − model) / √model")
        ax_b.set_title(
            f"per-bin pulls{' (MC pseudo-data)' if pseudo else ''} — {tag}"
            + (f"  |η₊| ∈ [{slice_def[0]:.1f}, {slice_def[1]:.1f}]" if slice_def else "")
        )

        ax_h.hist(pulls_finite, bins=20, range=(-5, 5),
                  histtype="step", color="k", lw=1.2)
        # Overlay N(0,1) reference scaled to integral=n_bins.
        x = np.linspace(-5, 5, 200)
        ax_h.plot(
            x, len(pulls_finite) * 10 / 20 * np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi),
            color="C0", lw=1, label="N(0,1)",
        )
        ax_h.set_xlabel("pull")
        ax_h.set_ylabel("bins")
        ax_h.legend(fontsize=8)
        ax_h.text(
            0.05, 0.95,
            f"mean={np.nanmean(pulls_finite):+.2f}\n"
            f"std={np.nanstd(pulls_finite):.2f}\n"
            f"n={len(pulls_finite)}",
            transform=ax_h.transAxes, va="top", fontsize=8,
        )

        fig.tight_layout()
        for p in _save_fig(fig, output_dir, f"mll_pulls_{tag}"):
            print(f"  wrote {p}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="jpsi_mass_fit_diagnostics",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to checkpoint_best.pt from train_jpsi_mass_fit.")
    p.add_argument("--shards", required=True,
                   help="Path to the shard directory (or list).")
    p.add_argument("--fisher", default=None,
                   help="Path to fisher_info.pt (optional; enables θ_scale ±σ "
                   "bands and the correlation heatmap).")
    p.add_argument("--output", default=None,
                   help="Output directory (default: <checkpoint_dir>/diagnostics/).")
    p.add_argument("--device", default=("cuda:0" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--batch-size", type=int, default=65536,
                   help="Loader batch size. Bigger is fine here — no "
                   "backward pass, only forward + collect to host.")
    p.add_argument("--n-mll-bins", type=int, default=50,
                   help="Number of m_ll bins. Cost of the per-event "
                   "grid eval scales linearly in this.")
    p.add_argument("--grid-chunk-events", type=int, default=4096,
                   help="Cap on events processed per flow call inside "
                   "the per-event × per-bin grid eval — bounds memory "
                   "(chunk_events × n_mll_bins expanded inputs per call).")
    p.add_argument("--max-events", type=int, default=0,
                   help="Stop after this many events (0 = run to end). "
                   "Bounds the grid-eval cost (O(N × n_mll_bins) flow "
                   "forwards) on full-statistics shards.")
    p.add_argument("--split", default="holdout", choices=("train", "val", "holdout", "all"),
                   help="Which loader split to evaluate on. 'holdout' is the "
                   "untouched-by-training default and the canonical choice.")
    p.add_argument("--eval-seed", type=int, default=42,
                   help="Fixed seed for the smearing-kernel ε on MC events. "
                   "Keeps the predicted-signal histogram deterministic across "
                   "diagnostic runs.")
    p.add_argument("--continuity-n-iter", type=int, default=2,
                   help="Fixed-point iterations for the #2 source solve.")
    p.add_argument("--param-shift", type=float, default=3.0,
                   help="Representative shift (in units of each parameter's "
                   "reference scale: THETA_SCALE_REF for A/e/M, SMEAR_VAR_SCALE "
                   "for a/c) used for the parameter-sensitivity overlay curves. "
                   "Default 3.0 → A±3e-4, e±3e-3, M±3e-5, a±3e-7, c±6e-5 — large "
                   "enough that the (otherwise sub-MeV) A/e/M peak shifts are "
                   "visible. The reference scales already roughly equalise the "
                   "inclusive m_ll effect across A/e/M (REF ratio ≈ k̄), so a "
                   "single global factor keeps them comparable.")
    p.add_argument("--no-param-sensitivity", action="store_true",
                   help="Skip the parameter-sensitivity slice plots "
                   "(param_sensitivity_*). They re-iterate the loader and do "
                   "~2·n_fitted extra grid evaluations per event, which is the "
                   "slowest diagnostic for the qop smear operator — pair with "
                   "--max-events for a quick look.")
    return p.parse_args(argv)


@contextlib.contextmanager
def _set_theta_output(model, scale_phys=None, smear_phys=None):
    """Temporarily SET the model's per-muon parameter output to fixed per-η-bin
    PHYSICAL values (masked to the fitted terms), for BOTH binned and MLP θ —
    used for the 'flow at injected θ' closure-target curve. Replaces
    ``_override_theta``, which sets the binned tensors and is a NO-OP under
    ``--theta-mlp`` (so that overlay was previously meaningless for MLP fits).
    ``scale_phys`` is [n_eta, 3] = (A, e, M); ``smear_phys`` is [n_eta, 2] =
    (a, c). The smear is converted to the O(1) coefficient (÷ SMEAR_VAR_SCALE)
    since ``_smear_ac_pm`` returns O(1) ac. The model's masks are applied so
    non-fitted terms are exactly 0 (matching the fit's parameterisation)."""
    dev = model.m_lo.device
    orig_s, orig_c = model._scale_AeM_pm, model._smear_ac_pm
    if scale_phys is not None:
        S = torch.as_tensor(scale_phys, dtype=torch.float32, device=dev)  # [n_eta,3]
        model._scale_AeM_pm = lambda e, p, b: S[b] * model.scale_param_mask
    if smear_phys is not None:
        sc = torch.tensor([SMEAR_VAR_SCALE_A, SMEAR_VAR_SCALE_C],
                          dtype=torch.float32, device=dev)
        C = torch.as_tensor(smear_phys, dtype=torch.float32, device=dev) / sc  # O(1)
        model._smear_ac_pm = lambda e, p, b: C[b] * model.smear_param_mask
    try:
        yield
    finally:
        if scale_phys is not None:
            del model.__dict__["_scale_AeM_pm"]
        if smear_phys is not None:
            del model.__dict__["_smear_ac_pm"]


@contextlib.contextmanager
def _shift_theta_output(model, dscale=None, dsmear=None):
    """Temporarily ADD a physical (A,e,M)/(a,c) shift to the model's per-muon
    parameter output. Works for BOTH binned and MLP θ because it wraps the
    output methods (``_scale_AeM_pm`` / ``_smear_ac_pm``), shifting the effective
    θ however it is produced — unlike ``_override_theta``, which only sets the
    binned tensors and is a no-op under ``--theta-mlp``. ``dscale`` is a physical
    [A, e, M] shift; ``dsmear`` a physical [a, c] shift (converted to the O(1)
    coefficient via SMEAR_VAR_SCALE before being added to the O(1) ac output)."""
    dev = model.m_lo.device
    set_s, set_c = dscale is not None, dsmear is not None
    orig_s, orig_c = model._scale_AeM_pm, model._smear_ac_pm
    if set_s:
        ds = torch.as_tensor(dscale, dtype=torch.float32, device=dev)
        model._scale_AeM_pm = lambda e, p, b: orig_s(e, p, b) + ds
    if set_c:
        sc = torch.tensor([SMEAR_VAR_SCALE_A, SMEAR_VAR_SCALE_C],
                          dtype=torch.float32, device=dev)
        dc = torch.as_tensor(dsmear, dtype=torch.float32, device=dev) / sc
        model._smear_ac_pm = lambda e, p, b: orig_c(e, p, b) + dc
    try:
        yield
    finally:
        if set_s:
            del model.__dict__["_scale_AeM_pm"]
        if set_c:
            del model.__dict__["_smear_ac_pm"]


@torch.no_grad()
def plot_param_sensitivity(model, loader, stats, m_centers, out_dir, *,
                           shift_scale=1.0, max_events=0, chunk_events=2048,
                           n_iter=2, device="cpu", mc_as_data=False, progress=True):
    """Per-slice m_ll closure with overlays of the model signal density at ±a
    representative shift of each FITTED parameter — to expose parameter
    sensitivities and degeneracies. The data (pseudo-data in validation) is
    histogrammed per slice; the model density at the fitted θ and at θ_fit ± Δ
    (one parameter at a time) are slice-weighted-averaged and overlaid.

    Slice variables are CONDITIONAL quantities (or functions of them) — the flow
    conditions on muon_kin = (η_±, φ_±, ρ), so slicing in those keeps the
    conditioning fixed within a slice and the closure cleanly isolates the model
    behaviour. (Slicing in a NON-conditional variable such as pt_avg — the
    absolute pt scale, which the leak-free conditioning deliberately omits —
    would conflate the flow's pt-marginalisation with the θ-effects.) The
    discriminating slices:
      |η|max — reference (the existing closure axis), from η_±;
      ρ = (pt₊−pt₋)/(pt₊+pt₋) — charge-odd → isolates M (mass effect ∝ pt₊−pt₋);
      cos α = (cosΔφ + sinhη₊ sinhη₋)/(coshη₊ coshη₋) — the 3-D opening angle.
              The pt CANCELS, so it is a pure function of (η_±, Δφ) ⊂ muon_kin;
              and at fixed m, m² = 2p₊p₋(1−cosα) ⇒ α fixes p₊p₋, i.e. the pt
              scale (k̄) — so it separates A vs e (peak shift ∝ A−e·k̄) and
              a vs c (mass-variance ∝ a·pt²+c) using ONLY conditional info.

    Slice edges are adaptive (tertiles) per variable. The shift Δ per parameter
    is ``shift_scale`` × its reference scale (THETA_SCALE_REF for A/e/M,
    SMEAR_VAR_SCALE for a/c)."""
    G = m_centers.shape[0]
    mdev = m_centers.to(device)
    mc = m_centers.cpu().numpy()
    dmb = float(m_centers[1] - m_centers[0])
    m_edges = np.concatenate([[mc[0] - dmb / 2], mc[:-1] + dmb / 2, [mc[-1] + dmb / 2]])

    # ± representative shift per FITTED parameter.
    ref_s = THETA_SCALE_REF
    ref_c = (SMEAR_VAR_SCALE_A, SMEAR_VAR_SCALE_C)
    pal = {"A": "C0", "e": "C1", "M": "C2", "a": "C5", "c": "C6"}
    shifts = []  # (label, color, '+'/'-', dscale|None, dsmear|None)
    for j, nm in enumerate("AeM"):
        if model.scale_enabled and nm in model.scale_fit_params:
            for s in (1, -1):
                d = [0.0, 0.0, 0.0]; d[j] = s * shift_scale * ref_s[j]
                shifts.append((nm, pal[nm], "+" if s > 0 else "-", d, None))
    for j, nm in enumerate("ac"):
        if model.smearing_enabled and model.smear_fit_params in ("both", nm):
            for s in (1, -1):
                d = [0.0, 0.0]; d[j] = s * shift_scale * ref_c[j]
                shifts.append((nm, pal[nm], "+" if s > 0 else "-", None, d))
    if not shifts:
        print("  no fitted parameters → skipping param-sensitivity plots")
        return

    def _cos_alpha(b):
        e, p = b["eta_pm"], b["phi_pm"]
        num = torch.cos(p[:, 0] - p[:, 1]) + torch.sinh(e[:, 0]) * torch.sinh(e[:, 1])
        return num / (torch.cosh(e[:, 0]) * torch.cosh(e[:, 1]))

    # All slice variables are conditional (or functions of muon_kin).
    slice_vars = {
        "abseta": (lambda b: b["eta_pm"].abs().max(1).values, "|η|max"),
        "rho": (lambda b: (b["pt_pm"][:, 0] - b["pt_pm"][:, 1])
                / (b["pt_pm"][:, 0] + b["pt_pm"][:, 1]), "ρ"),
        "cosalpha": (_cos_alpha, "cos α (opening angle)"),
    }

    def _select(b):
        s = (~b["is_data_mask"]) if mc_as_data else b["is_data_mask"]
        return s.nonzero(as_tuple=True)[0]

    # Pre-pass: collect the slice-variable values to set adaptive (tertile) edges
    # — cheap (no flow evals), and robust to each variable's unknown range.
    vals = {v: [] for v in slice_vars}
    seen = 0
    for batch in loader:
        if max_events > 0 and seen >= max_events:
            break
        batch = _move_batch(batch, device)
        idx = _select(batch)
        if idx.numel() == 0:
            continue
        seen += int(idx.numel())
        for v, (fn, _) in slice_vars.items():
            vals[v].append(fn(batch)[idx].cpu().numpy())
    if seen == 0:
        print("  no events selected → skipping param-sensitivity plots")
        return
    edges = {}
    for v in slice_vars:
        a = np.concatenate(vals[v])
        e = np.quantile(a, [0.0, 1.0 / 3, 2.0 / 3, 1.0])
        e[0] -= 1e-6; e[-1] += 1e-6   # include the extremes
        edges[v] = e
    del vals

    slice_specs = {v: (slice_vars[v][0], edges[v], slice_vars[v][1])
                   for v in slice_vars}
    acc = {v: [{"data": np.zeros(G), "fit": np.zeros(G), "w": 0.0,
                "sh": {i: np.zeros(G) for i in range(len(shifts))}}
               for _ in range(len(edges[v]) - 1)]
           for v in slice_vars}

    seen = 0
    bar = tqdm(loader, desc="param-sens", disable=not progress, unit="batch")
    for batch in bar:
        if max_events > 0 and seen >= max_events:
            break
        batch = _move_batch(batch, device)
        sel = (~batch["is_data_mask"]) if mc_as_data else batch["is_data_mask"]
        idx = sel.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        seen += int(idx.numel())
        mll = batch["mll"][idx].cpu().numpy()
        w = batch["w"][idx].cpu().numpy()
        p_fit = torch.exp(_tilt_density_on_grid(
            model, batch, idx, mdev, chunk_events=chunk_events, n_iter=n_iter)).cpu().numpy()
        p_sh = []
        for _, _, _, ds, dc in shifts:
            with _shift_theta_output(model, ds, dc):
                p_sh.append(torch.exp(_tilt_density_on_grid(
                    model, batch, idx, mdev, chunk_events=chunk_events,
                    n_iter=n_iter)).cpu().numpy())
        for v, (fn, edges, _) in slice_specs.items():
            val = fn(batch)[idx].cpu().numpy()
            for si in range(len(edges) - 1):
                sm = (val >= edges[si]) & (val < edges[si + 1])
                if not sm.any():
                    continue
                a = acc[v][si]
                a["data"] += np.histogram(mll[sm], bins=m_edges, weights=w[sm])[0]
                a["fit"] += (p_fit[sm] * w[sm, None]).sum(0)
                a["w"] += float(w[sm].sum())
                for i in range(len(shifts)):
                    a["sh"][i] += (p_sh[i][sm] * w[sm, None]).sum(0)
    bar.close()

    for v, (fn, edges, xlabel) in slice_specs.items():
        ns = len(edges) - 1
        # Two rows per slice: density (top) + ratio-to-fit (bottom). The ratio
        # panel makes the (small) shift sensitivity and the data/fit closure
        # legible in fractional terms — shift/fit isolates each parameter's
        # effect and data/fit shows the residual mis-closure.
        fig, axes = plt.subplots(
            2, ns, figsize=(4.8 * ns, 4.6), squeeze=False, sharex="col",
            gridspec_kw={"height_ratios": [3, 1]})
        for si in range(ns):
            ax, axr, a = axes[0, si], axes[1, si], acc[v][si]
            if a["w"] <= 0:
                ax.set_visible(False); axr.set_visible(False); continue
            fit_cnt = a["fit"] * dmb                  # model expected counts
            ax.step(mc, a["data"], where="mid", color="k", lw=1.2,
                    label="pseudo-data" if mc_as_data else "data")
            ax.plot(mc, fit_cnt, color="0.3", lw=2.0, label="model (fit)")
            for i, (nm, col, sgn, _, _) in enumerate(shifts):
                ax.plot(mc, a["sh"][i] * dmb, color=col, lw=1.0,
                        ls="--" if sgn == "+" else ":", alpha=0.85,
                        label=f"{nm}{sgn}Δ")
            hi = "∞" if edges[si + 1] > 1e8 else f"{edges[si + 1]:.2f}"
            ax.set_title(f"{xlabel} ∈ [{edges[si]:.2f}, {hi})", fontsize=9)
            ax.grid(alpha=0.3)
            if si == 0:
                ax.legend(fontsize=6, ncol=2)
                axr.set_ylabel("ratio / fit", fontsize=8)
            # --- ratio panel: everything ÷ the fitted model ---
            fok = fit_cnt > 0
            r_data = np.divide(a["data"], fit_cnt,
                               out=np.full_like(fit_cnt, np.nan), where=fok)
            axr.axhline(1.0, color="0.3", lw=1.2)
            axr.step(mc, r_data, where="mid", color="k", lw=1.0)
            r_all = [r_data]
            for i, (nm, col, sgn, _, _) in enumerate(shifts):
                r_sh = np.divide(a["sh"][i], a["fit"],
                                 out=np.full_like(a["fit"], np.nan), where=fok)
                axr.plot(mc, r_sh, color=col, lw=1.0,
                         ls="--" if sgn == "+" else ":", alpha=0.85)
                r_all.append(r_sh)
            # Robust y-range over the data-supported region (ignore empty tails).
            sup = a["data"] > 0.02 * (a["data"].max() if a["data"].max() > 0 else 1.0)
            rr = np.concatenate([r[sup] for r in r_all]) if sup.any() else np.array([])
            rr = rr[np.isfinite(rr)]
            if rr.size:
                lo, hi_r = np.percentile(rr, [1, 99])
                pad = 0.1 * max(hi_r - lo, 1e-3)
                axr.set_ylim(max(0.0, lo - pad), hi_r + pad)
            axr.set_xlabel("m_ll [GeV]"); axr.grid(alpha=0.3)
        fig.suptitle(f"parameter sensitivity — slices of {xlabel} "
                     f"(Δ = {shift_scale:g}× ref scale)")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"param_sensitivity_{v}.{ext}"), dpi=110)
        plt.close(fig)
        print(f"  wrote param_sensitivity_{v} ({ns} slices)")


def main() -> int:
    args = parse_args()
    out_dir = args.output or os.path.join(
        os.path.dirname(args.checkpoint), "diagnostics",
    )
    os.makedirs(out_dir, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU.")
        device = "cpu"

    print(f"loading checkpoint: {args.checkpoint}")
    model, stats, train_args, ckpt = load_model_from_checkpoint(args.checkpoint, device)

    # Loader.
    shard_files = discover_shards([args.shards])
    if not shard_files:
        print(f"error: no .arrow shards found under {args.shards!r}", file=sys.stderr)
        return 1
    # Validation (MC-closure) checkpoints fit θ on simulation as pseudo-data;
    # auto-detect so the m_ll closure / pulls treat simulation as the data
    # branch (otherwise those plots are empty — there are no is_data rows).
    mc_as_data = bool(train_args.get("validation", False))
    if mc_as_data:
        print("checkpoint was trained with --validation: routing simulation "
              "through the data branch as pseudo-data for the m_ll closure / pulls")

    # Injected θ_scale shift (validation closure with a non-zero target): read
    # the values the fit was trained with, replay the same m_ll injection in the
    # pseudo-data, and use them as the χ² reference + dashed line on the θ plot.
    inject_np = None
    inject_smear_np = None
    if mc_as_data:
        n_eta = len(stats.eta_edges) - 1
        ia = float(train_args.get("inject_A", 0.0) or 0.0)
        ie = float(train_args.get("inject_e", 0.0) or 0.0)
        im = float(train_args.get("inject_M", 0.0) or 0.0)
        if ia or ie or im:
            inject_np = np.zeros((n_eta, 3), dtype=np.float64)
            inject_np[:, 0] = ia; inject_np[:, 1] = ie; inject_np[:, 2] = im
            print(f"checkpoint injected θ_scale (A,e,M)=({ia:g},{ie:g},{im:g}) — "
                  f"replaying the pseudo-data injection; closure target = injected")
        isa = float(train_args.get("inject_a", 0.0) or 0.0)
        isc = float(train_args.get("inject_c", 0.0) or 0.0)
        if isa or isc:
            # --inject-a/-c are PHYSICAL σ²_qop coefficients — used directly.
            inject_smear_np = np.zeros((n_eta, 2), dtype=np.float64)
            inject_smear_np[:, 0] = isa
            inject_smear_np[:, 1] = isc
            print(f"checkpoint injected smear (a,c)=({isa:g},{isc:g}) — replaying "
                  f"the per-muon qop fold into the pseudo-data")

    print(f"found {len(shard_files)} shard(s); split={args.split}")
    loader = JpsiMassArrowLoader(
        shard_files, stats,
        batch_size=args.batch_size,
        split=args.split,
        val_fraction=float(train_args.get("val_fraction", 0.10)),
        holdout_fraction=float(train_args.get("holdout_fraction", 0.05)),
        drop_last=False,
        inject_theta_scale=inject_np,
        inject_theta_smear=inject_smear_np,
        inject_seed=int(train_args.get("inject_smear_seed", 12345)),
    )

    # m_ll grid.
    m_edges = torch.linspace(stats.m_lo, stats.m_hi, args.n_mll_bins + 1)
    bin_width = float((m_edges[1] - m_edges[0]).item())
    m_centers = 0.5 * (m_edges[:-1] + m_edges[1:])
    m_centers_np = m_centers.cpu().numpy()
    m_grid_std = (m_centers - stats.mll_mean) / stats.mll_std

    # η slices for the per-slice closure plots.
    eta_slice_edges = np.array([0.0, 0.6, 1.2, 1.8, 2.4])

    print(
        f"evaluating model on the loader (batch_size={args.batch_size}, "
        f"n_mll_bins={args.n_mll_bins}, "
        f"grid_chunk_events={args.grid_chunk_events}"
        f"{', max_events=' + str(args.max_events) if args.max_events else ''})..."
    )
    evals = evaluate_predictions(
        model, loader, device, m_centers, m_grid_std, bin_width,
        chunk_events=args.grid_chunk_events,
        max_events=args.max_events,
        progress=True,
        seed=args.eval_seed,
        n_iter=args.continuity_n_iter,
        mc_as_data=mc_as_data,
    )
    if mc_as_data:
        print(
            f"  collected {evals['mll_data'].shape[0]} MC pseudo-data events "
            f"(same simulation also drives the MC-branch closure)"
        )
    else:
        print(
            f"  collected {evals['mll_data'].shape[0]} data events, "
            f"{evals['mll_mc_fold'].shape[0]} MC events"
        )

    # Plot 1: m_ll closure.
    print("plotting m_ll closure...")
    plot_mll_closure(
        evals, m_centers_np, eta_slice_edges,
        stats.m_lo, stats.m_hi, out_dir,
    )

    # Fisher info → ±1σ for θ_scale (and θ_smear, when present).
    sigma_scale = None
    sigma_smear = None
    cov_scale_flat = None   # 72×72 θ_scale covariance block (for the χ² test)
    edm = None
    if args.fisher and os.path.exists(args.fisher):
        f = torch.load(args.fisher, weights_only=False)
        if f.get("n_negative_eig", 0) not in (0, None):
            print(f"  warning: fisher_info.pt has {f['n_negative_eig']} "
                  f"non-positive eigenvalue(s) (min={f.get('min_eig', float('nan')):.2e}) "
                  f"— σ bands for affected params are unreliable.")
        edm = f.get("edm")
        if edm is not None:
            print(f"  fit EDM (½ gᵀV g) = {edm:.3e}")
        cov_pt = f.get("covariance_24_3_24_3")
        if cov_pt is not None:
            cov_scale_flat = cov_pt.reshape(72, 72).cpu().numpy()
            sigma_scale = np.sqrt(np.maximum(np.diag(cov_scale_flat), 0.0)).reshape(24, 3)
        ss = f.get("sigma_smear_eff_24_2")
        if ss is not None:
            sigma_smear = ss.cpu().numpy()
        # Covariance + correlation matrix over the FULL joint parameter set
        # (θ_scale + active θ_smear); fall back to the θ_scale-only correlation
        # for legacy files that store only the 24×3×24×3 scale block.
        full_cov = f.get("covariance")
        if full_cov is not None:
            print("plotting covariance / correlation matrix...")
            plot_cov_corr(full_cov.detach().cpu().numpy(), f.get("labels"),
                          int(f.get("n_scale", 72)), out_dir, edm=edm)
        elif cov_scale_flat is not None:
            print("plotting correlation matrix (θ_scale block)...")
            plot_fisher_correlation(cov_scale_flat, out_dir)
        else:
            print("  warning: no covariance in the file; skipping matrix plot.")
    else:
        print("no fisher_info.pt → skipping θ_scale ±σ bands + correlation plot.")

    # Plots 2, 3: θ vs η — only for the *enabled* nuisances (a disabled one
    # is an inert, fixed parameter; plotting it would be misleading).
    print("plotting θ vs η...")
    # 'mlp' θ: sample the continuous ThetaNet at the η-bin centres (φ=0 slice)
    # so the same per-bin plot shows the learned function; no ±σ (no Fisher).
    mlp_scale_grid = mlp_smear_grid = None
    mlp_scale_band = mlp_smear_band = None
    mlp_scale_slices = mlp_smear_slices = None
    mlp_scale_avg_samples = mlp_smear_avg_samples = None
    mlp_slice_labels = None
    if model.theta_mode == "mlp":
        # Sample the ThetaNet on a 2D (η-centre, φ) grid: 16 points uniformly
        # spaced over [0, 2π) for the φ-average and ±std band, plus 4 cardinal
        # slices {0, π/2, π, −π/2} for the overlaid curves. Both muons are
        # given the same (η, φ) since the ThetaNet is per-muon (the symmetry
        # is implicit). The conditioner sees (cos φ, sin φ), so uniform φ on
        # the circle gives an exact mean + std with no statistical noise.
        centers = 0.5 * (np.asarray(stats.eta_edges[:-1]) + np.asarray(stats.eta_edges[1:]))
        centers_t = torch.as_tensor(centers, dtype=torch.float32, device=device)
        n_eta_c = int(centers_t.shape[0])
        # φ-average sampling (16 uniform points → exact integral over the circle).
        n_phi_avg = 16
        phi_avg = torch.linspace(
            0.0, 2 * np.pi * (1.0 - 1.0 / n_phi_avg), n_phi_avg,
            dtype=torch.float32, device=device)
        # Overlaid φ-slices for the reader.
        phi_slice_vals = torch.tensor(
            [0.0, np.pi / 2, np.pi, -np.pi / 2],
            dtype=torch.float32, device=device)
        mlp_slice_labels = ["φ=0", "φ=π/2", "φ=π", "φ=−π/2"]
        all_phi = torch.cat([phi_avg, phi_slice_vals])
        n_phi_all = int(all_phi.shape[0])
        # Build the (n_eta * n_phi, 2) per-muon η/φ grids (both muons share
        # (η, φ) — the ThetaNet's output for muon 0 is what gets plotted).
        eta_grid = centers_t[:, None, None].expand(
            n_eta_c, n_phi_all, 2).reshape(-1, 2)
        phi_grid = all_phi[None, :, None].expand(
            n_eta_c, n_phi_all, 2).reshape(-1, 2)
        with torch.no_grad():
            AeM_g, ac_g = model.theta_net(eta_grid, phi_grid)
        # Keep muon-0 output and reshape back to (n_eta, n_phi_all, n_comp).
        AeM_g = AeM_g[:, 0, :].view(n_eta_c, n_phi_all, 3)              # physical scale (×scale_ref inside the net)
        # Effective smear: route the raw MLP (a, c) through
        # _smear_raw_to_effective so the softplus positivity reparam (when
        # --smear-param-form softplus) is applied here too — same operator the
        # model uses internally for the MLP smear branch in _smear_ac_pm.
        # Then multiply by SMEAR_VAR_SCALE for the physical units.
        smear_scale = ac_g.new_tensor([SMEAR_VAR_SCALE_A, SMEAR_VAR_SCALE_C])
        ac_eff = model._smear_raw_to_effective(ac_g[:, 0, :])           # softplus(ac)·mask if 'softplus', else ac·mask
        ac_phys = (ac_eff * smear_scale).view(n_eta_c, n_phi_all, 2)
        # Split: first n_phi_avg are the φ-average grid; rest are the slices.
        AeM_avg, AeM_slc = AeM_g[:, :n_phi_avg, :], AeM_g[:, n_phi_avg:, :]
        ac_avg,  ac_slc  = ac_phys[:, :n_phi_avg, :], ac_phys[:, n_phi_avg:, :]
        mlp_scale_grid = AeM_avg.mean(dim=1).cpu().numpy()              # [n_eta, 3]  φ-mean
        mlp_scale_band = AeM_avg.std(dim=1).cpu().numpy()               # [n_eta, 3]  φ-std
        mlp_smear_grid = ac_avg.mean(dim=1).cpu().numpy()               # [n_eta, 2]
        mlp_smear_band = ac_avg.std(dim=1).cpu().numpy()                # [n_eta, 2]
        mlp_scale_slices = AeM_slc.cpu().numpy()                        # [n_eta, 4, 3]
        mlp_smear_slices = ac_slc.cpu().numpy()                         # [n_eta, 4, 2]
        # Raw φ-samples (physical) kept for the whitened-basis band/slices —
        # the φ-std must be computed AFTER projecting onto the eigenbasis (a
        # linear combination's spread ≠ the combination of per-component stds).
        mlp_scale_avg_samples = AeM_avg.cpu().numpy()                   # [n_eta, n_phi, 3]
        mlp_smear_avg_samples = ac_avg.cpu().numpy()                    # [n_eta, n_phi, 2]
        sigma_scale = sigma_smear = None
    if model.scale_enabled:
        # binned θ_scale is the O(1) fit param → ×THETA_SCALE_REF for physical
        # (A,e,M); the MLP grid is already physical (scale_ref inside the net).
        theta_scale = (mlp_scale_grid if mlp_scale_grid is not None
                       else ckpt.get("theta_scale", model.theta_scale.detach()).cpu().numpy()
                       * np.asarray(THETA_SCALE_REF))
        # χ² for compatibility of all θ_scale (A,e,M over the η bins) with the
        # reference (the injected values if a shift was injected, else 0), using
        # the full θ_scale covariance block (correlations included).
        ref = inject_np if inject_np is not None else None
        chi2_info = None
        if cov_scale_flat is not None:
            resid = theta_scale.reshape(-1)
            if ref is not None:
                resid = resid - ref.reshape(-1)
            chi2, dof, pval = _chi2_compat_zero(resid, cov_scale_flat)
            chi2_info = (chi2, dof, pval)
            tgt = "injected" if ref is not None else "0"
            print(f"  θ_scale compatibility with {tgt}: χ²/dof = {chi2:.1f}/{dof} = "
                  f"{chi2 / max(dof, 1):.2f}, p = {pval:.3g}")
        plot_theta_vs_eta(
            theta_scale, sigma_scale, ["A", "e [GeV]", "M"],
            "theta_scale_vs_eta", stats.eta_edges, out_dir, edm=edm,
            chi2_info=chi2_info, ref=ref,
            band=mlp_scale_band, slices=mlp_scale_slices,
            slice_labels=mlp_slice_labels,
        )
    else:
        print("  --disable-scale: skipping theta_scale_vs_eta")
    if model.smearing_enabled:
        # PHYSICAL qop-variance coefficients (a, c) (σ²_qop = a + c·k²); MLP mode
        # samples the net. effective_theta_smear() already applies SMEAR_VAR_SCALE.
        theta_smear_eff = (mlp_smear_grid if mlp_smear_grid is not None
                           else model.effective_theta_smear().detach().cpu().numpy())
        plot_theta_vs_eta(
            theta_smear_eff, sigma_smear, ["a [qop²]", "c [qop²·GeV²]"],
            "theta_smear_vs_eta", stats.eta_edges, out_dir, edm=edm,
            ref=(inject_smear_np if inject_smear_np is not None else None),
            band=mlp_smear_band, slices=mlp_smear_slices,
            slice_labels=mlp_slice_labels,
        )
    else:
        print("  --disable-smearing: skipping theta_smear_vs_eta")

    # Plots 3b/3c: closure in the DEGENERACY-WHITENED basis. (A,e) and (a,c)
    # are each near-degenerate over the J/ψ pt range, so the m_ll likelihood
    # constrains only the STIFF combination; the orthogonal SLOPPY combination
    # drifts. Projecting the fitted + injected θ onto the (stiff, sloppy)
    # eigenvectors separates "what J/ψ can measure" (stiff — should close) from
    # "what it cannot" (sloppy — large spread, may not close), which is exactly
    # the right way to read the closure when fitting both members of a pair.
    eigb = _degeneracy_eigbasis(getattr(stats, "k_moments", None))
    if eigb is None:
        print("  no k_moments in stats → skipping whitened-basis closure plots")
    else:
        E_s, l_s, E_c, l_c = eigb
        print("  degeneracy eigenbasis (global, O(1) θ coords; info ratio stiff/sloppy):")
        print("    scale (θ_A,θ_e): stiff×%.0f vec=[%+.3f,%+.3f]  sloppy vec=[%+.3f,%+.3f]"
              % (l_s[0], E_s[0, 0], E_s[1, 0], E_s[0, 1], E_s[1, 1]))
        print("    smear (θ_a,θ_c): stiff×%.0f vec=[%+.3f,%+.3f]  sloppy vec=[%+.3f,%+.3f]"
              % (l_c[0], E_c[0, 0], E_c[1, 0], E_c[0, 1], E_c[1, 1]))

        def _whitened_plot(phys_grid, phys_samples, phys_slices, ref_phys,
                           inj_phys, refE, lam, name, unit_pair):
            """Project a 2-component (A,e)/(a,c) set onto the eigenbasis and plot
            stiff/sloppy closure. ``phys_grid`` [n_eta,2]; ``phys_samples``
            [n_eta,n_phi,2] or None (→ band); ``phys_slices`` [n_eta,n_s,2] or
            None; ``inj_phys`` [n_eta,2] or None."""
            grid_w = _project_whitened(phys_grid, ref_phys, refE)        # [n_eta,2]
            band_w = None
            if phys_samples is not None:
                samp_w = _project_whitened(phys_samples, ref_phys, refE)  # [n_eta,n_phi,2]
                grid_w = samp_w.mean(axis=1)
                band_w = samp_w.std(axis=1)
            slices_w = (None if phys_slices is None
                        else _project_whitened(phys_slices, ref_phys, refE))
            ref_w = None if inj_phys is None else _project_whitened(inj_phys, ref_phys, refE)
            names = [f"STIFF (measured, info×{lam[0]:.0f})",
                     f"SLOPPY (degenerate, info×1)"]
            plot_theta_vs_eta(
                grid_w, None, names, name, stats.eta_edges, out_dir, edm=edm,
                ref=ref_w, band=band_w, slices=slices_w,
                slice_labels=mlp_slice_labels)

        scale_pair_fit = ("A" in model.scale_fit_params and "e" in model.scale_fit_params)
        if model.scale_enabled and scale_pair_fit:
            sg = (mlp_scale_grid[:, :2] if mlp_scale_grid is not None
                  else (ckpt.get("theta_scale", model.theta_scale.detach()).cpu().numpy()
                        * np.asarray(THETA_SCALE_REF))[:, :2])
            ss = None if mlp_scale_avg_samples is None else mlp_scale_avg_samples[:, :, :2]
            sl = None if mlp_scale_slices is None else mlp_scale_slices[:, :, :2]
            ij = None if inject_np is None else inject_np[:, :2]
            _whitened_plot(sg, ss, sl, THETA_SCALE_REF[:2], ij, E_s, l_s,
                           "theta_scale_whitened_vs_eta", ("A", "e"))
        if model.smearing_enabled and model.smear_fit_params == "both":
            cg = (mlp_smear_grid if mlp_smear_grid is not None
                  else model.effective_theta_smear().detach().cpu().numpy())
            cs = mlp_smear_avg_samples
            csl = mlp_smear_slices
            ij = inject_smear_np
            _whitened_plot(cg, cs, csl, [SMEAR_VAR_SCALE_A, SMEAR_VAR_SCALE_C],
                           ij, E_c, l_c, "theta_smear_whitened_vs_eta", ("a", "c"))

    # Plot 5: pulls.
    print("plotting per-bin pulls...")
    plot_pulls(
        evals, m_centers_np, eta_slice_edges,
        stats.m_lo, stats.m_hi, out_dir,
    )

    # Plot 6: MC closure (forward-folded MC vs flow density curve, both at
    # the fitted scale + smearing). Re-uses pred_signal_mc — no second pass.
    print("plotting MC closure...")
    plot_mc_closure(evals, m_centers_np, eta_slice_edges, out_dir)

    # Plot 7: parameter-sensitivity slices — model density at ±shifts of each
    # fitted param, in slices of |η| / ρ / pt_avg chosen to break degeneracies.
    # Re-iterates the loader (own pass; extra grid evals per shift).
    if not bool(getattr(args, "no_param_sensitivity", False)):
        print("plotting parameter-sensitivity slices...")
        plot_param_sensitivity(
            model, loader, stats, m_centers, out_dir,
            shift_scale=args.param_shift, max_events=args.max_events,
            chunk_events=args.grid_chunk_events, n_iter=args.continuity_n_iter,
            device=device, mc_as_data=mc_as_data,
            progress=getattr(args, "progress", True))

    print(f"\nall diagnostics written under {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
