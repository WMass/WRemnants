"""Per-event Arrow IPC loader for the unbinned J/ψ mass calibration fit.

Streams batches of mixed standardised + raw tensors from one or more
Arrow files written by ``jpsi_mass_fit_snapshot.py``. The η-bin
look-up uses the same 24 bins the existing ``make_jpsi_crctn_helper``
consumes, read once from the J/ψ calibration ROOT file at trainer
startup.

Tensor schema produced per batch:

  Observed (reco) flow inputs — standardised:
    mll            ``[B]``      raw m_ll [GeV]
    mll_std        ``[B]``      standardised m_ll
    y_event_std    ``[B, 7]``   standardised (y_ll, ln pt_ll, cos φ_ll,
                                sin φ_ll, cos θ*, sin φ*, cos φ*) — LEGACY,
                                emitted but unused by the model
    muon_kin_std   ``[B, 7]``   standardised (η_+, η_-, cos φ_+, sin φ_+,
                                cos φ_-, sin φ_-, ρ) — conditioning for BOTH the
                                signal flow (+ θ) and the background-fraction MLP
                                (ρ = (pt_+−pt_-)/(pt_++pt_-))

  Raw per-muon kinematics — float32, used by T_scale / T_smear:
    pt_pm        ``[B, 2]``   reco pt_+, pt_-  [GeV]
    eta_pm       ``[B, 2]``   reco η_+, η_-
    phi_pm       ``[B, 2]``   reco φ_+, φ_-
    q_pm         ``[B, 2]``   ±1

  Bookkeeping:
    b_pm           ``[B, 2]``   long (η-bin index of (+, −) muons)
    is_data_mask   ``[B]``      bool
    w              ``[B]``      float32 (nominal_weight for MC, 1 for data)

Standardisation uses fixed per-column ``mean`` / ``std`` tensors
provided at construction (computed once over the full dataset and
saved alongside the checkpoint). ``q_±`` and ``η_±`` are kept on
their physical scale by setting their ``mean = 0, std = 1`` in the
stats — both are bounded scalars that the network reads better in
their natural units.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, List, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from torch.utils.data import IterableDataset


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JpsiMassPreprocStats:
    """Standardisation stats + Bernstein window + η-bin edges."""

    # Raw → standardised offsets/scales (float32 arrays).
    mll_mean: float
    mll_std: float
    y_event_mean: np.ndarray   # shape [7]
    y_event_std: np.ndarray    # shape [7]
    muon_kin_mean: np.ndarray  # shape [7]
    muon_kin_std: np.ndarray   # shape [7]
    # η-bin edges (uniform 24 bins from -2.4 to +2.4 by default).
    eta_edges: np.ndarray      # shape [25]
    # Window kept for downstream consumers (loader doesn't filter again).
    m_lo: float
    m_hi: float
    # Per-η-bin curvature (k = 1/pt) moment SUMS over muons, used by the model
    # to build the degeneracy-whitening preconditioner (A↔e via ⟨k⟩, a↔c via
    # ⟨k²⟩,⟨k⁴⟩). Shape [n_eta, 4] = (N, Σk, Σk², Σk⁴). None for stats loaded
    # from an older stats.json (whitening then falls back to disabled).
    k_moments: np.ndarray | None = None

    @property
    def mll_log_scale(self) -> float:
        return float(np.log(self.mll_std))


# Per-event raw column names (in the order they appear in the snapshot).
_RAW_COLUMNS = (
    "mll",
    "yll",
    "ptll",
    "cosPhill",
    "sinPhill",
    "cosThetaStarll",
    "sinPhiStarll",
    "cosPhiStarll",
    "pt_plus",
    "eta_plus",
    "phi_plus",
    "q_plus",
    "pt_minus",
    "eta_minus",
    "phi_minus",
    "q_minus",
    "nominal_weight",
    "is_data",
    "source_id",
)


# Per-event derived feature ordering (matches jpsi_mass_model.N_Y_EVENT / N_MUON_KIN).
#
# Two feature blocks:
#   y_event  — dilepton-level kinematics. LEGACY: still computed/emitted, but
#              the model no longer consumes it (it includes pt_ll, which would
#              re-pin m_ll if it conditioned the signal flow).
#   muon_kin — the conditioning for BOTH the signal flow (+ θ) and the
#              background-fraction MLP: per-muon (η, φ) plus the pt asymmetry
#              ρ = (pt_+ − pt_-)/(pt_+ + pt_-). These span 5 of the 6 dimuon
#              DOF (η_±, φ_±, ρ), leaving the pt *scale* ↔ m_ll free, so the
#              flow's target is not leaked. φ is encoded as (cos, sin) per muon
#              to avoid wrap/boundary issues and keep φ_± recoverable (detector
#              φ-response is not azimuthally symmetric).
_Y_EVENT_FEATURES = (
    "yll",
    "log_ptll",
    "cosPhill",
    "sinPhill",
    "cosThetaStarll",
    "sinPhiStarll",
    "cosPhiStarll",
)
_MUON_KIN_FEATURES = (
    "eta_plus",
    "eta_minus",
    "cosPhi_plus",
    "sinPhi_plus",
    "cosPhi_minus",
    "sinPhi_minus",
    "rho",
)
# Features kept on their natural scale (mean=0, std=1 passthrough). The
# real-valued ρ is standardised; η_± and all cos/sin are passthrough.
_PASSTHROUGH_FEATURES = (
    "eta_plus",
    "eta_minus",
    "cosPhi_plus",
    "sinPhi_plus",
    "cosPhi_minus",
    "sinPhi_minus",
    "cosPhill",
    "sinPhill",
    "cosThetaStarll",
    "sinPhiStarll",
    "cosPhiStarll",
)


def _per_event_features(cols: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive ``(y_event_raw [N,7], muon_kin_raw [N,8], extras_raw [N,*])``
    from the snapshot's raw columns.
    """
    log_ptll = np.log(cols["ptll"].astype(np.float64, copy=False)).astype(np.float32)
    eta_plus = cols["eta_plus"]
    eta_minus = cols["eta_minus"]
    pt_plus = cols["pt_plus"].astype(np.float64, copy=False)
    pt_minus = cols["pt_minus"].astype(np.float64, copy=False)
    phi_plus = cols["phi_plus"].astype(np.float64, copy=False)
    phi_minus = cols["phi_minus"].astype(np.float64, copy=False)
    # pt asymmetry ρ ∈ (−1, 1); φ as (cos, sin) per muon (wrap-free, φ±
    # recoverable for the φ-dependent detector response).
    rho = ((pt_plus - pt_minus) / (pt_plus + pt_minus)).astype(np.float32)

    # y_event — dilepton kinematics, MLP (background-fraction) conditioning.
    y_event = np.stack(
        [
            cols["yll"],
            log_ptll,
            cols["cosPhill"],
            cols["sinPhill"],
            cols["cosThetaStarll"],
            cols["sinPhiStarll"],
            cols["cosPhiStarll"],
        ],
        axis=1,
    ).astype(np.float32)

    # muon_kin — signal-flow conditioning: (η_±, cos/sin φ_±, ρ). Leak-free
    # (pt scale ↔ m_ll left free), φ_± recoverable.
    muon_kin = np.stack(
        [
            eta_plus,
            eta_minus,
            np.cos(phi_plus).astype(np.float32),
            np.sin(phi_plus).astype(np.float32),
            np.cos(phi_minus).astype(np.float32),
            np.sin(phi_minus).astype(np.float32),
            rho,
        ],
        axis=1,
    ).astype(np.float32)

    return y_event, muon_kin


# ---------------------------------------------------------------------------
# Stats computation (single pass over all shards)
# ---------------------------------------------------------------------------


def compute_jpsi_mass_stats(
    shard_files: Sequence[str],
    *,
    m_lo: float,
    m_hi: float,
    eta_edges: np.ndarray | None = None,
) -> JpsiMassPreprocStats:
    """Stream over the snapshot shards, compute per-column mean/std.

    Stats are computed *unweighted* — calibration-fit standardisation
    just needs the input ranges to be roughly normalised. The trainer
    later applies the per-event ``nominal_weight`` in the loss.
    Passthrough features (charge, η, the cos/sin angle features) get
    mean=0, std=1 so they pass through unchanged.
    """
    if eta_edges is None:
        eta_edges = np.linspace(-2.4, 2.4, 25, dtype=np.float64)

    mll_sum = mll_sq = 0.0
    n_y = len(_Y_EVENT_FEATURES)
    n_k = len(_MUON_KIN_FEATURES)
    y_sum = np.zeros(n_y, dtype=np.float64)
    y_sq = np.zeros(n_y, dtype=np.float64)
    k_sum = np.zeros(n_k, dtype=np.float64)
    k_sq = np.zeros(n_k, dtype=np.float64)
    n_rows = 0
    # Per-η-bin curvature moments (k = 1/pt), accumulated over BOTH muons of
    # every event (each assigned to its own η bin). Columns: (N, Σk, Σk², Σk⁴).
    n_eta = len(eta_edges) - 1
    kmom = np.zeros((n_eta, 4), dtype=np.float64)

    for path in shard_files:
        with pa.OSFile(path, "rb") as src:
            reader = ipc.open_file(src)
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                cols = {c: batch.column(c).to_numpy() for c in _RAW_COLUMNS}
                m = cols["mll"].astype(np.float64, copy=False)
                mll_sum += float(m.sum())
                mll_sq += float((m * m).sum())
                y_event, muon_kin = _per_event_features(cols)
                y_sum += y_event.sum(axis=0)
                y_sq += (y_event.astype(np.float64) ** 2).sum(axis=0)
                k_sum += muon_kin.sum(axis=0)
                k_sq += (muon_kin.astype(np.float64) ** 2).sum(axis=0)
                n_rows += int(len(m))
                # Curvature moments per η-bin, stacking + and − muons.
                k_pm = 1.0 / np.concatenate([
                    cols["pt_plus"].astype(np.float64, copy=False),
                    cols["pt_minus"].astype(np.float64, copy=False)])
                eta_pm = np.concatenate([cols["eta_plus"], cols["eta_minus"]])
                b_pm = _bucketize_eta(eta_pm, eta_edges)
                k2 = k_pm * k_pm
                np.add.at(kmom[:, 0], b_pm, 1.0)
                np.add.at(kmom[:, 1], b_pm, k_pm)
                np.add.at(kmom[:, 2], b_pm, k2)
                np.add.at(kmom[:, 3], b_pm, k2 * k2)

    if n_rows == 0:
        raise RuntimeError(f"no rows found across shards {list(shard_files)!r}")

    def _mean_std(s, sq, n):
        mu = s / n
        var = np.maximum(sq / n - mu * mu, 1e-12)
        return mu.astype(np.float32), np.sqrt(var).astype(np.float32)

    mll_mean = mll_sum / n_rows
    mll_var = max(mll_sq / n_rows - mll_mean * mll_mean, 1e-12)
    mll_std = float(np.sqrt(mll_var))

    y_mean, y_std = _mean_std(y_sum, y_sq, n_rows)
    k_mean, k_std = _mean_std(k_sum, k_sq, n_rows)

    # Force passthrough features.
    for i, name in enumerate(_Y_EVENT_FEATURES):
        if name in _PASSTHROUGH_FEATURES:
            y_mean[i] = 0.0
            y_std[i] = 1.0
    for i, name in enumerate(_MUON_KIN_FEATURES):
        if name in _PASSTHROUGH_FEATURES:
            k_mean[i] = 0.0
            k_std[i] = 1.0

    return JpsiMassPreprocStats(
        mll_mean=float(mll_mean),
        mll_std=mll_std,
        y_event_mean=y_mean,
        y_event_std=y_std,
        muon_kin_mean=k_mean,
        muon_kin_std=k_std,
        eta_edges=np.asarray(eta_edges, dtype=np.float64),
        m_lo=float(m_lo),
        m_hi=float(m_hi),
        k_moments=kmom,
    )


# ---------------------------------------------------------------------------
# Per-batch derivation
# ---------------------------------------------------------------------------


def _standardise(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def _bucketize_eta(eta: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Per-muon η-bin index in {0, …, 23}. Out-of-range values clamp."""
    idx = np.searchsorted(edges[1:-1], eta, side="right")
    return np.clip(idx, 0, len(edges) - 2).astype(np.int64)


def _scale_advection_np(mll, pt_pm, q_pm, b_pm, theta_inj):
    """Analytic advective m_ll shift Σ_μ v_μ·θ_inj[b_μ] for an injected
    θ_scale, matching the model's ``_continuity_response`` (per muon
    v = (−m/2, (m/2)/pt, −(m/2)·q·pt) for (A, e, M)). ``theta_inj`` is
    ``[n_eta, 3]``; inputs are ``[N]`` / ``[N, 2]``. Returns ``[N]``."""
    k = 1.0 / pt_pm                                   # [N, 2]
    A = theta_inj[b_pm, 0]; e = theta_inj[b_pm, 1]; M = theta_inj[b_pm, 2]  # [N,2]
    mh = 0.5 * mll[:, None]                            # [N, 1]
    s = (-mh * A + mh * k * e - mh * q_pm * pt_pm * M).sum(axis=1)
    return s.astype(np.float32)


def _scale_inject_rho_np(pt_pm, eta_pm, q_pm, b_pm, theta_inj):
    """ρ after applying the injected θ_scale (forward, truth→obs) to the muon
    pt — matching the model's ``_delta_qop_analytic`` + ``_apply_scale_pt``
    (sign=+1, floor inactive for the small injected shift). Returns ``[N]``.
    The fit's ``_scale_source_rho_std`` un-applies the same scale, so ρ_src → ρ_MC."""
    sinth = 1.0 / np.cosh(eta_pm)                     # [N,2]
    k = 1.0 / pt_pm
    A = theta_inj[b_pm, 0]; e = theta_inj[b_pm, 1]; M = theta_inj[b_pm, 2]
    dqop = q_pm * sinth * ((A - e * k) * k + q_pm * M)
    qop = q_pm * sinth / pt_pm
    pt_inj = q_pm * sinth / (qop + dqop)              # forward
    return ((pt_inj[:, 0] - pt_inj[:, 1]) /
            (pt_inj[:, 0] + pt_inj[:, 1])).astype(np.float32)


_MUON_MASS_GEV = 0.1056583755


def _event_mll_np(pt_pm, eta_pm, phi_pm):
    """Two-body invariant mass ``[N]`` for muons of mass ``_MUON_MASS_GEV`` —
    numpy twin of ``jpsi_mass_model._event_mll``. Inputs ``[N, 2]``."""
    px = pt_pm * np.cos(phi_pm)
    py = pt_pm * np.sin(phi_pm)
    pz = pt_pm * np.sinh(eta_pm)
    E = np.sqrt(px * px + py * py + pz * pz + _MUON_MASS_GEV * _MUON_MASS_GEV)
    Etot = E.sum(1); Px = px.sum(1); Py = py.sum(1); Pz = pz.sum(1)
    m2 = Etot * Etot - (Px * Px + Py * Py + Pz * Pz)
    return np.sqrt(np.clip(m2, 1e-12, None))


def _inject_pt_np(pt_pm, eta_pm, q_pm, b_pm, scale_inj, smear_inj, rng,
                  qop_floor_frac: float = 0.25):
    """Apply the injected θ_scale (deterministic δqop) THEN the θ_smear (per-muon
    Gaussian qop kick) to the per-muon pt, entirely in qop/pt space, with the
    sign-preserving qop floor at each step — mirroring the model's forward
    operators (``_delta_qop_analytic`` + ``_apply_scale_pt`` for the scale,
    ``apply_smear_pt`` for the smear). Returns the injected pt ``[N, 2]``.

    Working in pt space (rather than the old additive m_ll shifts) lets the
    caller propagate a fully SELF-CONSISTENT pseudo-data event — smeared pt,
    ``mll = _event_mll(pt)``, and ρ(pt) — so every downstream consumer (both
    smear operators, the per-muon fold, the diagnostics) sees coherent observed
    quantities with no rescaling, exactly as for real data."""
    sinth = 1.0 / np.cosh(eta_pm)                          # [N,2]
    pt_cur = pt_pm.astype(np.float64, copy=True)

    def _floor_to_pt(qop_new, qop_ref):
        if qop_floor_frac > 0.0:
            s = np.sign(qop_ref)
            qop_new = s * np.maximum(qop_new * s, qop_floor_frac * np.abs(qop_ref))
        return q_pm * sinth / qop_new

    if scale_inj is not None:
        k = 1.0 / pt_cur
        A = scale_inj[b_pm, 0]; e = scale_inj[b_pm, 1]; M = scale_inj[b_pm, 2]
        dqop = q_pm * sinth * ((A - e * k) * k + q_pm * M)
        qop_ref = q_pm * sinth / pt_cur
        pt_cur = _floor_to_pt(qop_ref + dqop, qop_ref)
    if smear_inj is not None:
        k2 = (1.0 / pt_cur) ** 2
        vq = smear_inj[b_pm, 0] + smear_inj[b_pm, 1] * k2   # σ²_qop = a + c·k²
        sig = np.sqrt(np.clip(vq, 0.0, None))
        eps = rng.standard_normal(pt_pm.shape)
        qop_ref = q_pm * sinth / pt_cur
        pt_cur = _floor_to_pt(qop_ref + sig * eps, qop_ref)
    return pt_cur.astype(np.float32)


def _smear_inject_dmll_np(mll, pt_pm, eta_pm, phi_pm, q_pm, b_pm, smear_inj, rng,
                          qop_floor_frac: float = 0.25):
    """Δm_ll ``[N]`` from injecting a per-muon qop Gaussian smear at the injected
    qop-variance coefficients ``smear_inj`` ([n_eta, 2] = (a, c)): exactly the
    validation-plot fold path (``fold_sigma_qop_pm`` + ``apply_smear_pt`` +
    ``_event_mll``) — the COMBINED variance ``σ²_qop = a + c·k²`` clipped at 0
    (an injected smear can only broaden), an INDEPENDENT Gaussian qop kick per
    muon, then recompute m_ll. ρ (a flow condition) is left untouched, as in the
    fold. Returned as the mass CHANGE so it composes additively with the scale
    advection."""
    sinth = 1.0 / np.cosh(eta_pm)                          # [N,2]
    k2 = (1.0 / pt_pm) ** 2                                # [N,2]
    vq = smear_inj[b_pm, 0] + smear_inj[b_pm, 1] * k2      # σ²_qop = a + c·k² [N,2]
    sig = np.sqrt(np.clip(vq, 0.0, None))                  # σ_qop [N,2]
    eps = rng.standard_normal(pt_pm.shape).astype(pt_pm.dtype)
    qop = q_pm * sinth / pt_pm
    qop_new = qop + sig * eps
    # sign-preserving floor, mirroring the model's qop→pt inversion guard.
    s = np.sign(qop)
    qop_new = s * np.maximum(qop_new * s, qop_floor_frac * np.abs(qop))
    pt_new = q_pm * sinth / qop_new
    dm = _event_mll_np(pt_new, eta_pm, phi_pm) - _event_mll_np(pt_pm, eta_pm, phi_pm)
    return dm.astype(np.float32)


def _batch_tensors(
    cols: dict,
    stats: JpsiMassPreprocStats,
    inject_theta_scale: "np.ndarray | None" = None,
    inject_theta_smear: "np.ndarray | None" = None,
    rng: "np.random.Generator | None" = None,
) -> dict[str, torch.Tensor]:
    """Build the tensor batch from one Arrow record batch's columns.

    ``inject_theta_scale`` ([n_eta, 3], validation closure only): the MC
    (``is_data == 0``) m_ll is shifted by the advective scale shift at that
    injected θ_scale, so the (pseudo-)data look as if they carried that
    calibration and the fit should recover it.

    ``inject_theta_smear`` ([n_eta, 2] = (a, c), validation closure only): the MC
    m_ll additionally gets the per-muon qop Gaussian smear at those injected
    width coefficients (same fold path as the validation plots; needs ``rng``).
    """
    y_event, muon_kin = _per_event_features(cols)

    b_plus = _bucketize_eta(cols["eta_plus"], stats.eta_edges)
    b_minus = _bucketize_eta(cols["eta_minus"], stats.eta_edges)
    b_pm = np.stack([b_plus, b_minus], axis=1)

    pt_pm = np.stack([cols["pt_plus"], cols["pt_minus"]], axis=1).astype(np.float32)
    eta_pm = np.stack([cols["eta_plus"], cols["eta_minus"]], axis=1).astype(np.float32)
    phi_pm = np.stack([cols["phi_plus"], cols["phi_minus"]], axis=1).astype(np.float32)
    q_pm = np.stack([cols["q_plus"], cols["q_minus"]], axis=1).astype(np.float32)

    is_data_mask = (cols["is_data"].astype(np.uint8) != 0)

    mll = cols["mll"].astype(np.float32)
    if inject_theta_scale is not None or inject_theta_smear is not None:
        # Validation closure: apply the injected scale + smear to the per-muon pt
        # in qop space (MC rows only) and propagate FULLY CONSISTENT observed
        # quantities — smeared pt, mll = _event_mll(pt), ρ = ρ(pt) — so the
        # pseudo-data is coherent exactly like real data and every downstream
        # consumer (both smear operators, the fold, the diagnostics) uses it
        # directly with no rescaling. (Previously the smear was injected as an
        # additive m_ll shift with pt left at its nominal value, leaving mll and
        # pt_pm inconsistent; the qop operator, which reconstructs the mass from
        # pt, then saw the nominal mass and missed the injected smear.)
        mc = ~is_data_mask
        pt_inj = _inject_pt_np(
            pt_pm, eta_pm, q_pm, b_pm, inject_theta_scale, inject_theta_smear, rng)
        mll_inj = _event_mll_np(pt_inj, eta_pm, phi_pm).astype(np.float32)
        rho_inj = ((pt_inj[:, 0] - pt_inj[:, 1]) /
                   (pt_inj[:, 0] + pt_inj[:, 1])).astype(np.float32)
        mll = np.where(mc, mll_inj, mll).astype(np.float32)
        pt_pm = np.where(mc[:, None], pt_inj, pt_pm).astype(np.float32)
        muon_kin[:, -1] = np.where(mc, rho_inj, muon_kin[:, -1])
    mll_std = ((mll - stats.mll_mean) / stats.mll_std).astype(np.float32)

    y_event_std = _standardise(y_event, stats.y_event_mean, stats.y_event_std)
    muon_kin_std = _standardise(muon_kin, stats.muon_kin_mean, stats.muon_kin_std)

    w = cols["nominal_weight"].astype(np.float32, copy=False)
    # Enforce the m_ll window on the (possibly injection-perturbed) mass: zero
    # the weight for events pushed outside [m_lo, m_hi] by the injection so the
    # fit does not see them. Bernstein-d1 evaluates to NEGATIVE values outside
    # the window (linear u-extrapolation) and the flow's log p₀ extrapolates
    # past its training range — without this cut, those events hit the
    # `log(p_mix.clamp_min(eps))` floor at NLL ≈ +log(1/eps) ≈ 69 per event
    # AND give the data-branch MLP a strong incentive to grow f_bkg to absorb
    # the pollution via the Bernstein basis's large positive value just past
    # the boundary. For REAL data the snapshot already cut to the window so
    # this is a no-op (in_window all True). For injected MC pseudo-data it
    # zero-weights ~0.06% of central / ~3% of |η|>1.8 events; both the
    # weighted numerator AND denominator drop them, so the mean NLL is
    # computed correctly over the in-window subset.
    in_window = ((mll >= float(stats.m_lo)) & (mll <= float(stats.m_hi)))
    w = w * in_window.astype(np.float32)

    return {
        "mll": torch.from_numpy(mll),
        "mll_std": torch.from_numpy(mll_std),
        "y_event_std": torch.from_numpy(y_event_std),
        "muon_kin_std": torch.from_numpy(muon_kin_std),
        "b_pm": torch.from_numpy(b_pm),
        "is_data_mask": torch.from_numpy(is_data_mask),
        "w": torch.from_numpy(w),
        "pt_pm": torch.from_numpy(pt_pm),
        "eta_pm": torch.from_numpy(eta_pm),
        "phi_pm": torch.from_numpy(phi_pm),
        "q_pm": torch.from_numpy(q_pm),
    }


# ---------------------------------------------------------------------------
# IterableDataset
# ---------------------------------------------------------------------------


class JpsiMassArrowLoader(IterableDataset):
    """Single-process per-event Arrow IPC loader.

    Iterates over ``shard_files`` in order, yielding fixed-size
    batches of pre-standardised tensors. Splits the input record
    batches into ``train`` / ``val`` / ``holdout`` slices by
    contiguous record-batch index (the sharder upstream is
    responsible for global shuffle of rows; the loader does no
    additional shuffling for the v1).

    For DDP, instantiate one loader per rank with the same
    ``shard_files`` and ``world_size, rank`` set accordingly; shards
    are dealt out round-robin so each rank reads a disjoint subset.
    """

    _SPLITS = ("train", "val", "holdout", "all")

    def __init__(
        self,
        shard_files: Sequence[str],
        stats: JpsiMassPreprocStats,
        *,
        batch_size: int = 65536,
        split: str = "train",
        val_fraction: float = 0.1,
        holdout_fraction: float = 0.05,
        drop_last: bool = True,
        world_size: int = 1,
        rank: int = 0,
        pin_memory: bool = False,
        half: "int | None" = None,
        inject_theta_scale: "np.ndarray | None" = None,
        inject_theta_smear: "np.ndarray | None" = None,
        inject_seed: int = 12345,
    ):
        if split not in self._SPLITS:
            raise ValueError(f"split must be one of {self._SPLITS}, got {split!r}")
        if half not in (None, 0, 1):
            raise ValueError(f"half must be None, 0, or 1, got {half!r}")
        self.shard_files = list(shard_files)
        self.my_shards = self.shard_files[rank::world_size]
        self.stats = stats
        self.batch_size = int(batch_size)
        self.split = split
        self.val_fraction = float(val_fraction)
        self.holdout_fraction = float(holdout_fraction)
        self.drop_last = bool(drop_last)
        self.pin_memory = bool(pin_memory)
        self.world_size = int(world_size)
        self.rank = int(rank)
        # Deterministic disjoint event half (0/1) selected before the
        # train/val/holdout slice; None = all events. Used by the MC-closure
        # validation mode to give stage 1 and stage 2 disjoint simulation.
        self.half = half
        # Injected θ_scale [n_eta, 3] (validation closure): shifts the MC m_ll
        # by the advective scale shift so the (pseudo-)data carry that
        # calibration; None = no injection.
        self.inject_theta_scale = (
            np.asarray(inject_theta_scale, dtype=np.float64)
            if inject_theta_scale is not None else None)
        # Injected smear width coefficients [n_eta, 2] = (a, c) (validation
        # closure): a per-muon qop Gaussian kick on the MC m_ll via the same
        # fold path as the validation plots. Stochastic → seeded per __iter__
        # (inject_seed) so the pseudo-data realisation is reproducible.
        self.inject_theta_smear = (
            np.asarray(inject_theta_smear, dtype=np.float64)
            if inject_theta_smear is not None else None)
        self.inject_seed = int(inject_seed)

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _split_range(n: int, val_frac: float, holdout_frac: float, which: str):
        """Per-shard ROW window for ``which`` split.

        Previously split by record-batch index, which broke when the
        sharder wrote one big record batch per shard: ``round(1*0.1)=0``
        gave a permanently-empty val set. Splitting by row works for
        any shard layout (one big batch or many small ones).
        """
        n_holdout = int(round(n * holdout_frac))
        n_val = int(round(n * val_frac))
        n_train = n - n_val - n_holdout
        if which == "train":
            return 0, n_train
        if which == "val":
            return n_train, n_train + n_val
        if which == "holdout":
            return n_train + n_val, n
        return 0, n  # 'all'

    # -- iteration ------------------------------------------------------

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        accum: dict[str, list[np.ndarray]] = {c: [] for c in _RAW_COLUMNS}
        accum_n = 0
        # Reseed each pass so the injected smear realisation is reproducible
        # (and identical across epochs → a fixed pseudo-data set).
        rng = np.random.default_rng(self.inject_seed)

        for path in self.my_shards:
            with pa.OSFile(path, "rb") as src:
                reader = ipc.open_file(src)
                # Read the whole shard as one Arrow Table. For our shard
                # sizes (~10⁴-10⁵ rows × ~20 float32 cols ≈ a few MB)
                # this is cheap; zero-copy where possible because Arrow
                # IPC is memory-mapped on disk.
                table = reader.read_all()
                if self.half is not None:
                    # Deterministic disjoint half by row-index parity (half 0 =
                    # even rows, half 1 = odd). The sharder pre-shuffles rows
                    # globally, so parity is a clean ~50/50 mix. Applied before
                    # the split below so each half carries its own
                    # train/val/holdout windows.
                    idx = np.arange(self.half, table.num_rows, 2)
                    table = table.take(pa.array(idx))
                n_rows = table.num_rows
                start_row, stop_row = self._split_range(
                    n_rows, self.val_fraction, self.holdout_fraction, self.split,
                )
                if start_row >= stop_row:
                    continue
                sub = table.slice(start_row, stop_row - start_row)
                # Iterate in record batches of at most ``batch_size`` rows
                # so the cross-shard accumulator can stitch them into the
                # caller's requested mini-batch size.
                for batch in sub.to_batches(max_chunksize=self.batch_size):
                    for c in _RAW_COLUMNS:
                        accum[c].append(batch.column(c).to_numpy())
                    accum_n += batch.num_rows
                    while accum_n >= self.batch_size:
                        cols = {c: np.concatenate(accum[c]) for c in _RAW_COLUMNS}
                        emit = {c: cols[c][: self.batch_size] for c in _RAW_COLUMNS}
                        rem = {c: cols[c][self.batch_size :] for c in _RAW_COLUMNS}
                        accum = {c: [rem[c]] for c in _RAW_COLUMNS}
                        accum_n -= self.batch_size
                        yield _batch_tensors(
                            emit, self.stats, self.inject_theta_scale,
                            self.inject_theta_smear, rng)

        # Final partial batch.
        if accum_n > 0 and not self.drop_last:
            cols = {c: np.concatenate(accum[c]) for c in _RAW_COLUMNS}
            yield _batch_tensors(
                cols, self.stats, self.inject_theta_scale,
                self.inject_theta_smear, rng)


# ---------------------------------------------------------------------------
# Convenience for the trainer
# ---------------------------------------------------------------------------


def discover_shards(paths: Sequence[str]) -> List[str]:
    """Expand ``paths`` (files or directories) to a flat list of
    Arrow IPC files."""
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.endswith(".arrow"):
                    out.append(os.path.join(p, name))
        else:
            out.append(p)
    return out
