"""SCETlibNPParamModel — continuous-λ rabbit ParamModel for SCETlib NP.

Architecture (template-style fit):

    P(b, g)      = R_raw(b, g) / N_gen(g)               (normalized response)
    σ_reco(λ; b) = Σ_g  P(b, g) · σ_gen(λ; g)
    ratio(b)     = σ_reco(λ; b) / σ_reco(λ_central; b)
    rnorm        = ratio per reco bin, broadcast over signal proc; ones elsewhere.

The raw response counts R_raw(b, g) carry the MC's absolute gen spectrum, which
is theory-dependent. A response matrix should encode only the gen→reco
*mapping*, so each gen column is normalized by the gen-total N_gen(g) →
P(b, g) = efficiency × migration (theory-independent). N_gen(g) is the xnorm
"postfsr" histogram from the unfolding output — the generated fiducial yield
per gen bin BEFORE reco selection — loaded alongside R by response_matrix.
Then σ_gen(λ) is folded through P. NB normalizing instead by the reco-passing
marginal Σ_b R_raw(b, g) is wrong: R is post-reco-selection so that marginal
already includes efficiency, and dividing by it cancels efficiency
(migration-only), which closes far worse — efficiency is not flat in gen bin.
(If the gen-total hist is absent, σ_gen(λ_c) is used as a proxy for N_gen, but
then σ_gen cancels in σ_reco(λ_c) = R_raw·1 and the λ_central closure can't
test the integral.)

R_raw is the (reco × gen) response matrix loaded from the upstream unfolding
histmaker output (a separate hdf5 from the fit-tensor input).

λ_central is read from the fit-tensor's meta_info_input via the upstream
SCETlib correction pkl (see :mod:`scetlib_lambda_central`).

σ_gen(λ; g) is evaluated on the btgrid then integrated over Q (arctan_Q²
Simpson) and rebinned (Simpson) onto the unfolding hist's gen edges
(ptVGen, absYVGen). The absYVGen-side rebin folds the signed btgrid Y axis
into |Y| bins (NP is Y-symmetric: F_eff depends on Y², γ_ν^NP doesn't depend
on Y at all).

The 8 v1 parameters (all factorisable through the current btgrid):

    γ_ν^NP (CS-side):       lambda2_nu, lambda4_nu, lambda_inf_nu
    F_eff  (TMD-effective): lambda2, lambda4, lambda6, delta_lambda2, lambda_inf

The np_model and np_model_nu strings are fixed at construction (from
λ_central). All λ values are TF Variables — differentiable in the fit.
"""

import json
import os
import re
from typing import Mapping, Optional

import numpy as np
import tensorflow as tf

from rabbit.param_models.param_model import ParamModel
from wremnants.postprocessing.scetlib_np import (
    btgrid_cache,
)
from wremnants.postprocessing.scetlib_np import btgrid_integrate as fz_int
from wremnants.postprocessing.scetlib_np import btgrid_tf as fz_tf
from wremnants.postprocessing.scetlib_np import lambda_central as scetlib_lambda_central
from wremnants.postprocessing.scetlib_np import response_matrix as fz_R
from wremnants.utilities import common as wrem_common

# Default fixed-order inputs for the NP-independent nonsingular term
# σ_ns = DYTurbo − SCETlib_singular, resolved relative to this package via
# wrem_common.data_dir (= <WRemnants>/wremnants-data/data). These are the CT18Z
# N3+0LL fixed-order pieces; both are NP-independent (same for any λ tune).
_NONSING_FO_SING_DEFAULT = os.path.join(
    wrem_common.data_dir,
    "TheoryCorrections",
    "inclusive_Z_COM13_CT18Z_N3+0LL_lattice_lambda4bugfix_fine_nnlo_sing_combined.pkl",
)
_NONSING_DYTURBO_DEFAULT = os.path.join(
    wrem_common.data_dir,
    "TheoryCorrections",
    "results_z-2d-nnlo-vj-CT18ZNNLO-{scale}-scetlibmatch.txt",
)

# Ordered list of the v1 continuous λ. CS-side first, then TMD-effective.
GNU_PARAMS = ("lambda2_nu", "lambda4_nu", "lambda_inf_nu")
EFF_PARAMS = ("lambda2", "lambda4", "lambda6", "delta_lambda2", "lambda_inf")
ALL_PARAMS = GNU_PARAMS + EFF_PARAMS


# Theorist-recommended Gaussian prior widths for the SCETlib NP λ parameters.
# Source: NP-NP discussion slide, central values 2026-05, plus a wide
# in-house default for delta_lambda2 (the theorist hasn't quoted a width
# for it; 0 ± 0.2 is comfortably wider than its expected scale).
#
#     λ₂^ν       = 0.15 ± 0.10
#     Λ₂         = 0.40 ⁺⁰·⁶₋₀.₄   (asymmetric)
#     Λ₄         = 0.40 ⁺⁰·⁶₋₀.₄   (asymmetric)
#     δ Λ₂       = 0.00 ± 0.20      (in-house wide default; not from the slide)
#
# Asymmetric uncertainties (Λ₂, Λ₄) are approximated by a symmetric Gaussian
# with σ = (σ⁺ + σ⁻) / 2. Slightly conservative on the upper side, slightly
# loose on the lower side. A future patch can implement a split-Gaussian if
# needed.
#
# Any λ not listed here gets σ = NaN by default → no prior, floats free.
# Currently those are: lambda_inf, lambda_inf_nu, lambda4_nu, lambda6.
# They are expected to be FROZEN via rabbit's --freezeParameters until
# the theorist provides priors for them.
THEORIST_PRIOR_SIGMAS = {
    "lambda2_nu": 0.10,
    "lambda2": 0.50,  # 0.4 ⁺⁰·⁶₋₀.₄ -> symmetric average
    "lambda4": 0.50,  # 0.4 ⁺⁰·⁶₋₀.₄ -> symmetric average
    "delta_lambda2": 0.20,  # 0 ± 0.20 wide default (no theorist value yet)
}


def _load_lambda_central_file(path):
    """Load a λ_central override from a JSON or YAML file.

    The file must decode to a dict with ``eff_params`` and ``gnu_params``
    sub-dicts (same shape as :func:`scetlib_lambda_central.read_lambda_central`).
    Format is chosen by extension (``.yaml``/``.yml`` → YAML, else JSON);
    YAML's loader also accepts JSON, so this is forgiving either way.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SCETLIB_NP_LAMBDA_CENTRAL_FILE points to a missing file: {path!r}"
        )
    with open(path) as f:
        text = f.read()
    if path.lower().endswith((".yaml", ".yml")):
        import yaml

        data = yaml.safe_load(text)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"SCETLIB_NP_LAMBDA_CENTRAL_FILE {path!r} is not valid JSON; got {exc}"
            ) from exc
    if (
        not isinstance(data, dict)
        or "eff_params" not in data
        or "gnu_params" not in data
    ):
        raise ValueError(
            f"SCETLIB_NP_LAMBDA_CENTRAL_FILE {path!r} must decode to a dict with "
            f"'eff_params' and 'gnu_params' keys; got {type(data).__name__} "
            f"with keys {list(data) if isinstance(data, dict) else '<n/a>'}."
        )
    return data


def _crop_R_to_fit(R, R_reco_axes, fit_reco_axes, tol=1e-9):
    """Crop R's trailing reco bins so its reco shape matches the fit.

    R's reco binning is typically a superset of the fit's (e.g. R has one
    extra overflow ptll bin past the fit's last edge). For each reco axis,
    require that R's leading edges match the fit's edges; crop R along that
    axis to keep only the matching bins.
    """
    if len(R_reco_axes) != len(fit_reco_axes):
        raise ValueError(
            f"Reco axis count mismatch: R has {len(R_reco_axes)}, "
            f"fit has {len(fit_reco_axes)}"
        )
    for (rname, redges), (fname, fedges) in zip(R_reco_axes, fit_reco_axes):
        if rname != fname:
            raise ValueError(f"Reco axis name mismatch: R={rname!r} vs fit={fname!r}")
        fnb = len(fedges)
        if len(redges) < fnb:
            raise ValueError(
                f"Reco axis {rname}: R has {len(redges)-1} bins, fit needs "
                f"{fnb-1}. R is missing edges."
            )
        if not np.allclose(redges[:fnb], fedges, atol=tol):
            raise ValueError(
                f"Reco axis {rname}: leading R edges don't match fit edges. "
                f"R[:{fnb}]={list(redges[:fnb])} vs fit={list(fedges)}"
            )
    slices = tuple(slice(0, len(fedges) - 1) for (_, fedges) in fit_reco_axes)
    # Keep all gen axes (the remaining axes of R).
    slices += (slice(None),) * (R.ndim - len(fit_reco_axes))
    return R[slices]


def _bin_sum_matrix(src_centers, target_edges, tol=1e-6):
    """(N_target, N_src) 0/1 matrix that SUMS bin-integrated source bins whose
    centre falls in each target bin. Source bins outside all target bins are
    dropped — a natural truncation to the target range (qT>ptVGen_max, |Y|>absY_max)."""
    src = np.asarray(src_centers, dtype=np.float64)
    edges = np.asarray(target_edges, dtype=np.float64)
    W = np.zeros((edges.size - 1, src.size), dtype=np.float64)
    for i in range(edges.size - 1):
        m = (src >= edges[i] - tol) & (src <= edges[i + 1] + tol)
        W[i, m] = 1.0
    return W


def compute_nonsingular_gen(
    fo_sing_path,
    dyturbo_path,
    gen_axes_meta,
    charge=0,
    q_lo=60.0,
    q_hi=120.0,
    qt_cutoff=1.0,
    dyturbo_axes=("Q", "Y", "qT"),
):
    """Nonsingular FO term on the model gen grid (NptVGen, NabsYVGen).

    The fixed-order/DYTurbo matching adds a NP-INDEPENDENT piece to σ_gen:
        σ_gen^matched(λ) = σ_gen^resum(λ) + σ_ns ,
    where the nonsingular is read straight from the original fixed-order inputs:
        σ_ns = (DYTurbo fixed order) − (SCETlib singular fixed order)
    — exactly the ``-hfo_sing + hfo`` that ``read_matched_scetlib_hist`` forms.
    ``fo_sing_path`` is the SCETlib singular ``…_nnlo_sing…combined.pkl``;
    ``dyturbo_path`` is the DYTurbo fixed-order ``results_…scetlibmatch.txt``
    (use ``{scale}`` → mur1-muf1 for the central). The nonsingular is zeroed below
    ``qt_cutoff`` (as make_theory_corr does), Q-windowed to [q_lo, q_hi], |Y|-folded,
    and projected onto the coarse (ptVGen, absYVGen) gen bins by SUMMING the
    bin-integrated native bins.
    """
    from wremnants.utilities.io_tools import input_tools
    from wums import boostHistHelpers as hh

    def _central(h):
        if "vars" in h.axes.name:
            names = list(h.axes["vars"])
            idx = 0
            for c in ("central", "pdf0", "nominal"):
                if c in names:
                    idx = names.index(c)
                    break
            h = h[{"vars": idx}]
        return h

    # SCETlib singular fixed order, and DYTurbo fixed order, from their own files.
    dyturbo_path = (
        dyturbo_path.format(scale="mur1-muf1")
        if "{scale}" in dyturbo_path
        else dyturbo_path
    )
    hfo_sing = _central(input_tools.read_scetlib_hist(fo_sing_path, charge=charge))
    hfo = input_tools.read_dyturbo_hist(
        [dyturbo_path], axes=list(dyturbo_axes), charge=charge
    )
    if "vars" in hfo.axes.name:
        hfo = _central(hfo)

    # Align shared physics axes (DYTurbo is coarser), then σ_ns = DYTurbo − singular.
    for ax in ("Y", "Q", "qT"):
        if ax in set(hfo.axes.name) & set(hfo_sing.axes.name):
            hfo, hfo_sing = hh.rebinHistsToCommon([hfo, hfo_sing], ax)
    nonsing_h = hh.addHists(-1.0 * hfo_sing, hfo, flow=False, by_ax_name=False)

    if "charge" in nonsing_h.axes.name:
        nonsing_h = nonsing_h[{"charge": sum}]
    # Q-window: slice(...,sum) sums ONLY the in-range Q bins (no underflow leak).
    Qe = np.asarray(nonsing_h.axes["Q"].edges, dtype=np.float64)
    qi = int(np.argmin(np.abs(Qe - q_lo)))
    qj = int(np.argmin(np.abs(Qe - q_hi)))
    nonsing_h = nonsing_h[{"Q": slice(qi, qj, sum)}]
    nonsing_h = hh.makeAbsHist(nonsing_h, "Y")  # signed Y -> |Y|

    qT_c = np.asarray(nonsing_h.axes["qT"].centers, dtype=np.float64)
    absY_c = np.asarray(nonsing_h.axes["absY"].centers, dtype=np.float64)
    v = nonsing_h.project("qT", "absY").values(flow=False)  # (qT, absY)
    v[qT_c < qt_cutoff, :] = 0.0  # zero the nonsingular below the cutoff

    ptV_edges = np.asarray(gen_axes_meta[0][1], dtype=np.float64)
    absY_edges = np.asarray(gen_axes_meta[1][1], dtype=np.float64)
    Wp = _bin_sum_matrix(qT_c, ptV_edges)  # (NptVGen, NqT)
    Wa = _bin_sum_matrix(absY_c, absY_edges)  # (NabsYVGen, NabsYsrc)
    return Wp @ v @ Wa.T  # (NptVGen, NabsYVGen)


class SCETlibNPParamModel(ParamModel):

    def __init__(
        self,
        indata,
        unfolding_hdf5_path: str,
        btgrid_dir: str,
        lambda_central: Optional[Mapping] = None,
        signal_proc: str = "Zmumu",
        Q_lo: float = 60.0,
        Q_hi: float = 120.0,
        poi_params: Optional[tuple] = (),
        prior_sigmas: Optional[Mapping] = None,
        include_nonsingular: bool = True,
        nonsingular_fo_sing: str = _NONSING_FO_SING_DEFAULT,
        nonsingular_dyturbo: str = _NONSING_DYTURBO_DEFAULT,
        nonsingular_qt_cutoff: float = 1.0,
        **kwargs,
    ):
        """Construct the ParamModel.

        Parameters
        ----------
        indata
            rabbit's input-data structure (passed by ``ph.load_models``).
        unfolding_hdf5_path
            Path to the upstream histmaker output containing
            ``nominal_postfsr_yieldsUnfolding`` for R.
        btgrid_dir
            Directory of the SCETlib bT-grid shards (fineall).
        lambda_central
            Dict with two sub-dicts ``eff_params`` and ``gnu_params`` (same
            shape as returned by :func:`scetlib_lambda_central.read_lambda_central`).
            By default λ_central is auto-detected from ``indata.metadata``;
            pass this only to override or to support hand-built indata that
            lacks metadata.
        signal_proc
            Name of the signal process whose reco yields get the per-bin
            ratio. Other processes get factor 1.
        Q_lo, Q_hi
            Z mass window for the Q-integration on the btgrid.
        poi_params
            Tuple of parameter names (subset of ``ALL_PARAMS``) to treat as
            POIs (reported as POIs in the fit output). The rest are reported
            as model nuisances (npou). The POI vs POU split is independent
            of the prior assignment (see ``prior_sigmas``).
        prior_sigmas
            Per-name override dict for the Gaussian prior σ on each
            parameter. Defaults come from ``THEORIST_PRIOR_SIGMAS``:

                lambda2_nu : 0.10
                lambda2    : 0.50   (symmetric approx of +0.6/-0.4)
                lambda4    : 0.50   (symmetric approx of +0.6/-0.4)

            All other params default to ``NaN`` → no prior, float free; in
            practice they are expected to be frozen with rabbit's
            ``--freezeParameters`` until the theorist provides priors for
            them. Pass ``np.nan`` here to free a constrained param, or a
            finite value to add a prior on one that defaults to NaN.
            Only consumed when the fitter is invoked with
            ``--paramModelPriors``; otherwise everything floats free.
            Prior mean for each param is ``self.xparamdefault`` (the
            runcard's λ_central).
        """
        self.indata = indata

        # ---- Double-counting guard
        # If the histmaker baked discrete NP κ-template variations into the
        # input HDF5, those systs describe the same physics as our continuous
        # λ POUs. Running both → double-counting (the discrete syst absorbs
        # whatever shape variation our ParamModel should describe). Warn
        # loudly if any such systs are present and unfrozen.
        self._check_discrete_np_double_counting(kwargs.get("freezeParameters"))

        # ---- λ_central
        # Three sources of λ_central, in priority order:
        #   1. ``lambda_central`` constructor arg (explicit dict).
        #   2. ``SCETLIB_NP_LAMBDA_CENTRAL_FILE`` env var — path to a JSON or
        #      YAML file with ``eff_params`` and ``gnu_params``. Overrides the
        #      metadata auto-detect; useful when the upstream SCETlib pkl isn't
        #      accessible (e.g. a colleague's input).
        #   3. Auto-detect from the fit hdf5's theoryCorr → upstream pkl.
        env_lc_file = os.environ.get("SCETLIB_NP_LAMBDA_CENTRAL_FILE", "").strip()
        if lambda_central is None and env_lc_file:
            lambda_central = _load_lambda_central_file(env_lc_file)
            print(
                f"[SCETlibNPParamModel] λ_central from file {env_lc_file!r}",
                flush=True,
            )
        if lambda_central is None:
            # Auto-detect from indata.metadata (loaded by rabbit's
            # FitInputData from the input HDF5's "meta" group).
            indata_meta = getattr(indata, "metadata", None) or {}
            if not indata_meta:
                raise ValueError(
                    "SCETlibNPParamModel: indata has no metadata; pass "
                    "lambda_central explicitly or use an indata that "
                    "carries metadata."
                )
            lambda_central = scetlib_lambda_central.read_lambda_central_from_meta(
                indata_meta, _source="indata.metadata"
            )
            print(
                f"[SCETlibNPParamModel] λ_central auto-detected from indata.metadata",
                flush=True,
            )
        print(f"[SCETlibNPParamModel] λ_central:", flush=True)
        for key, value in lambda_central.items():
            print(f"  {key} = {value!r}", flush=True)

        self.eff_central = dict(lambda_central["eff_params"])
        self.gnu_central = dict(lambda_central["gnu_params"])
        self.np_model = self.eff_central["np_model"]
        self.np_model_nu = self.gnu_central["np_model_nu"]

        # ---- btgrid + dense layout
        grid = btgrid_cache.load(btgrid_dir)
        idx_map = fz_int.dense_index_map(grid["bins"])
        self.Q_unique = idx_map["Q_unique"]
        self.Y_unique = idx_map["Y_unique"]
        self.qT_unique = idx_map["qT_unique"]
        self.flat_idx = tf.constant(idx_map["flat_idx"], dtype=tf.int64)

        # Cache btgrid arrays as TF constants.
        self.bT = tf.constant(grid["bT"], dtype=fz_tf.DTYPE)
        self.b_bar = tf.constant(grid["b_bar"], dtype=fz_tf.DTYPE)
        self.I_pert = tf.constant(grid["I_pert"][0], dtype=fz_tf.DTYPE)  # (Nbins, Nbt)
        self.C_nu = tf.constant(grid["C_nu"][0], dtype=fz_tf.DTYPE)

        # Per-bin qT and Y (from the bin tuple), for reconstruct_batch_tf.
        bins = grid["bins"]
        self.qT_per_bin = tf.constant(
            np.array([b[2] for b in bins], dtype=np.float64), dtype=fz_tf.DTYPE
        )
        Y_pb_np = np.array([b[1] for b in bins], dtype=np.float64)
        self.Y_per_bin = tf.constant(Y_pb_np, dtype=fz_tf.DTYPE)

        # F_eff depends on the bin only through Y (not Q or qT), and Y takes few
        # distinct values across the grid. Precompute the unique-Y map so
        # reconstruct_batch_tf evaluates the NP transcendentals on NY rows and
        # gathers, instead of recomputing identical rows for every (Q, qT).
        Y_feff_unique_np, Y_feff_inv_np = np.unique(Y_pb_np, return_inverse=True)
        self.Y_feff_unique = tf.constant(Y_feff_unique_np, dtype=fz_tf.DTYPE)
        self.Y_feff_inverse_idx = tf.constant(
            Y_feff_inv_np.reshape(-1).astype(np.int32), dtype=tf.int32
        )

        # Precompute the bT·J0(qT·bT) kernel (λ-independent).
        self.bT_J0_kernel = fz_tf.build_bT_J0_kernel(self.qT_per_bin, self.bT)
        self.bT_simpson_w = tf.constant(
            fz_tf.simpson_weights(np.asarray(self.bT)), dtype=fz_tf.DTYPE
        )

        # ---- Q-integration weights (arctan_Q² Simpson on Z mass window).
        self.Q_weights = tf.constant(
            fz_int.q_integrate_weights(self.Q_unique, Q_lo, Q_hi),
            dtype=fz_tf.DTYPE,
        )

        # ---- R matrix
        R_info = fz_R.load_R(unfolding_hdf5_path)
        # The fit-tensor's reco binning may differ from R's by trailing
        # overflow bins (e.g. R has ptll [0, …, 44, 100] while the fit ends
        # at 44). Crop R's trailing bins so the reco shape matches.
        fit_reco_axes = self._fit_reco_axes(indata)
        R_arr = _crop_R_to_fit(R_info["R"], R_info["reco_axes"], fit_reco_axes)
        # Tighten the metadata to match the cropped R.
        self.reco_shape = R_arr.shape[: len(fit_reco_axes)]
        self.gen_shape = R_arr.shape[len(fit_reco_axes) :]
        N_reco = int(np.prod(self.reco_shape))
        N_gen = int(np.prod(self.gen_shape))
        # Raw response counts; normalized to a response below.
        self._R_raw = tf.constant(R_arr.reshape(N_reco, N_gen), dtype=fz_tf.DTYPE)
        # Gen-total denominator N_gen(g) from the xnorm hist ("postfsr"): the
        # generated fiducial yield per gen bin (pre-reco-selection). Dividing R
        # by this gives the theory-independent efficiency×migration response.
        # Falls back to the σ_gen(λ_c) proxy if the gen-total isn't in the file.
        if R_info.get("N_gen") is not None:
            self._N_gen_flat = tf.constant(
                R_info["N_gen"].reshape(-1), dtype=fz_tf.DTYPE
            )
        else:
            self._N_gen_flat = None
            print(
                "[SCETlibNPParamModel] WARNING: no gen-total hist in unfolding "
                "output; falling back to σ_gen(λ_c) as the response normalizer "
                "(circular closure — see param_model docstring).",
                flush=True,
            )
        self._reco_axes_meta = [
            (name, fit_axes[1])
            for (name, fit_axes) in zip(
                [a[0] for a in R_info["reco_axes"]],
                fit_reco_axes,
            )
        ]
        self._gen_axes_meta = R_info["gen_axes"]

        # ---- Rebin weights: btgrid (NY signed) → (NabsYVGen) via |Y| folding
        # and (NqT) → (NptVGen).
        absY_edges = self._gen_axes_meta[1][1]  # absYVGen edges
        ptVGen_edges = self._gen_axes_meta[0][1]  # ptVGen edges

        # |Y| folding: σ(Y) is symmetric in Y so the absY-bin integral is
        # 2·∫_{absY_lo}^{absY_hi} σ(Y) dY. Use Y >= 0 source samples and
        # multiply by 2.
        Y_pos_mask = self.Y_unique >= 0
        Y_pos = self.Y_unique[Y_pos_mask]
        absY_rebin_pos = fz_int.rebin_weights(Y_pos, absY_edges, name="absY")
        # Pad to full NY: zero on negative-Y columns.
        W_absY = np.zeros((absY_edges.size - 1, self.Y_unique.size), dtype=np.float64)
        W_absY[:, Y_pos_mask] = 2.0 * absY_rebin_pos
        self.W_absY = tf.constant(W_absY, dtype=fz_tf.DTYPE)

        # qT rebin: btgrid qT (signed nonneg, NqT=141) → ptVGen edges (e.g. 20 bins, 0-44).
        # Anything past ptVGen_max is out of fit range; we drop it silently for now
        # (events with gen qT > ptVGen_max are routed through R's overflow which is
        # not present in our materialised R — see plan doc, dyturbo-handoff item).
        self.W_ptVGen = tf.constant(
            fz_int.rebin_weights(self.qT_unique, ptVGen_edges, name="ptVGen"),
            dtype=fz_tf.DTYPE,
        )

        # ---- Normalize the response, then cache σ_reco(λ_central).
        # A response matrix must encode only the gen→reco *mapping*, not the
        # MC's absolute gen spectrum. Normalize each gen column by the gen-total
        # N_gen(g) (the xnorm "postfsr" hist — generated fiducial yield before
        # reco selection) → P(b|g) = eff×migration (theory-independent):
        #     P(b|g)        = R_raw(b,g) / N_gen(g)
        #     σ_reco(λ;b)   = Σ_g P(b|g) · σ_gen(λ;g)
        #     σ_reco(λ_c;b) = Σ_g P(b|g) · σ_gen(λ_c;g)
        # NB the reco-passing marginal Σ_b R_raw(b,g) is the WRONG normalizer:
        # it already includes efficiency (R is post-reco-selection), so dividing
        # by it cancels efficiency (migration-only) — closes far worse, ε is not
        # flat in gen bin. We use the true gen-total N_gen instead. Because
        # σ_reco_central then depends on σ_gen(λ_c) (it does NOT collapse to
        # R_raw·1), the λ_central closure is a genuine test of the integral.
        # Fallback: if the gen-total hist is absent, use σ_gen(λ_c) as a proxy
        # for N_gen (∝ σ_gen^MC) — keeps efficiency but makes the closure
        # circular (σ_gen cancels to R_raw·1).
        # Native-binning Q-integrated reconstruction (NY, NqT) on the signed-Y /
        # qT grid, BEFORE the |Y|-fold and qT-rebin — exposed so the native-binning
        # validation can compare it to the SCETlib reference / numpy factorize
        # without the projection layer.
        self.sigma_YqT_central = self._sigma_YqT_native_at(
            self.eff_central, self.gnu_central
        )
        # ---- Optional fixed-order/DYTurbo nonsingular term (NP-independent).
        # σ_gen^matched(λ) = σ_gen^resum(λ) + σ_ns. Added at GEN level, so it folds
        # through the same response R as the resummed piece. Because rnorm is a
        # ratio, this correctly DILUTES the NP variation where the FO dominates
        # (high qT). σ_ns is a constant (no λ dependence).
        self.include_nonsingular = bool(include_nonsingular)
        if self.include_nonsingular:
            _dy0 = (
                nonsingular_dyturbo.format(scale="mur1-muf1")
                if (nonsingular_dyturbo and "{scale}" in nonsingular_dyturbo)
                else nonsingular_dyturbo
            )
            missing = [
                p for p in (nonsingular_fo_sing, _dy0) if not (p and os.path.exists(p))
            ]
            if missing:
                raise FileNotFoundError(
                    "include_nonsingular=True needs the fixed-order inputs for "
                    "σ_ns = DYTurbo − SCETlib_singular, but these are missing:\n  "
                    + "\n  ".join(missing)
                    + "\nThey live under wremnants-data/data/TheoryCorrections (the "
                    "SCETlib singular …_nnlo_sing…combined.pkl and the DYTurbo "
                    "results_…scetlibmatch.txt). Pass nonsingular_fo_sing / "
                    "nonsingular_dyturbo, or set include_nonsingular=False for resum-only."
                )
            sigma_ns_np = compute_nonsingular_gen(
                nonsingular_fo_sing,
                nonsingular_dyturbo,
                self._gen_axes_meta,
                q_lo=Q_lo,
                q_hi=Q_hi,
                qt_cutoff=nonsingular_qt_cutoff,
            )
            if sigma_ns_np.shape != tuple(self.gen_shape):
                raise ValueError(
                    f"nonsingular gen shape {sigma_ns_np.shape} != model gen shape "
                    f"{tuple(self.gen_shape)}"
                )
            self.sigma_ns = tf.constant(sigma_ns_np, dtype=fz_tf.DTYPE)
        else:
            self.sigma_ns = tf.zeros(self.gen_shape, dtype=fz_tf.DTYPE)
        sigma_gen_central = self._sigma_gen_at(self.eff_central, self.gnu_central)
        # The pure gen-level integral (NptVGen, NabsYVGen), BEFORE folding through
        # the response — used by the gen-level validation to test the integral
        # in isolation (no R).
        self.sigma_gen_central = sigma_gen_central
        gen_flat = tf.reshape(sigma_gen_central, [-1])
        if tf.reduce_any(gen_flat <= 0).numpy():
            n_bad = int(tf.reduce_sum(tf.cast(gen_flat <= 0, tf.int32)))
            raise ValueError(
                f"SCETlibNPParamModel: {n_bad} gen bins have non-positive "
                f"σ_gen(λ_central); cannot normalize / fold the response."
            )
        N_gen = self._N_gen_flat if self._N_gen_flat is not None else gen_flat
        # Guard empty gen bins (no generated events): leave column at 0.
        safe_N_gen = tf.where(N_gen > 0, N_gen, tf.ones_like(N_gen))
        self.R = self._R_raw / safe_N_gen[tf.newaxis, :]  # P(b|g) = R_raw / N_gen
        self.sigma_reco_central = tf.linalg.matvec(self.R, gen_flat)  # Σ_g P·σ_gen(λ_c)
        if tf.reduce_any(self.sigma_reco_central <= 0).numpy():
            n_bad = int(tf.reduce_sum(tf.cast(self.sigma_reco_central <= 0, tf.int32)))
            raise ValueError(
                f"SCETlibNPParamModel: {n_bad} reco bins have non-positive "
                f"σ_reco(λ_central). Likely a binning mismatch between R and "
                f"the fit-tensor reco axes."
            )

        # ---- Process index: signal column gets the ratio, others get 1.
        procs = [p.decode() if isinstance(p, bytes) else str(p) for p in indata.procs]
        if signal_proc not in procs:
            raise ValueError(
                f"SCETlibNPParamModel: signal_proc={signal_proc!r} not in "
                f"indata.procs={procs[:10]}..."
            )
        self.signal_proc_idx = procs.index(signal_proc)
        self.nproc = indata.nproc

        # ---- ParamModel registration (POIs first, then NOUs).
        poi_params = tuple(poi_params or ())
        nou_params = tuple(p for p in ALL_PARAMS if p not in poi_params)
        self._param_order = poi_params + nou_params
        self.npoi = len(poi_params)
        self.npou = len(nou_params)
        self.params = np.array([p.encode() for p in self._param_order])

        # Defaults: λ_central values per parameter. Optionally overridden by
        # the ``SCETLIB_NP_XPARAMDEFAULT`` env var — comma-separated
        # ``name=value`` pairs (for closure tests where the data-generating
        # / fit-start point should differ from the card's λ_central).
        central_lookup = {**self.eff_central, **self.gnu_central}
        defaults = np.array(
            [central_lookup[p] for p in self._param_order], dtype=np.float64
        )

        env_override = os.environ.get("SCETLIB_NP_XPARAMDEFAULT", "").strip()
        if env_override:
            overrides = dict(
                tuple(s.split("=")) for s in env_override.split(",") if s.strip()
            )
            for name, val in overrides.items():
                name = name.strip()
                if name not in self._param_order:
                    raise KeyError(f"SCETLIB_NP_XPARAMDEFAULT: unknown param {name!r}")
                i = self._param_order.index(name)
                defaults[i] = float(val)
            print(
                f"[SCETlibNPParamModel] xparamdefault overridden: {dict(zip(self._param_order, defaults))}"
            )
        # rabbit's set_param_default expects an internal-storage convention
        # where POIs (npoi entries) are SQRT(value) if not allowNegativeParam.
        # For our λ which can in principle be tiny / zero (delta_lambda2),
        # default to allowNegativeParam=True so the stored value == λ directly.
        self.allowNegativeParam = True
        self.is_linear = False
        self.xparamdefault = tf.constant(defaults, dtype=indata.dtype)

        # Gaussian priors (consumed by rabbit's Fitter when --paramModelPriors
        # is set; ignored otherwise). Default σ values come from
        # ``THEORIST_PRIOR_SIGMAS`` (lambda2_nu, lambda2, lambda4 — the only
        # params the theorist provides widths for as of 2026-05).
        # All other params default to σ = NaN → no prior, float free; in
        # practice those should be frozen via rabbit's --freezeParameters
        # until the theorist gives priors for them.
        # The ``prior_sigmas`` kwarg is a per-name override dict; pass NaN to
        # force a parameter free, or a finite value to add / change a prior.
        # The mean of each prior is λ_central (i.e. self.xparamdefault).
        prior_sigmas = dict(prior_sigmas or {})
        sigmas_arr = np.empty(self.nparams, dtype=np.float64)
        for i, p in enumerate(self._param_order):
            if p in prior_sigmas:
                sigmas_arr[i] = float(prior_sigmas[p])  # explicit override (may be NaN)
            elif p in THEORIST_PRIOR_SIGMAS:
                sigmas_arr[i] = THEORIST_PRIOR_SIGMAS[p]  # theorist recommendation
            else:
                sigmas_arr[i] = np.nan  # free (expected to be frozen)
        self.prior_sigmas = sigmas_arr
        # prior_means defaults to xparamdefault if not set, so don't store
        # redundantly — Fitter will fall back to xparamdefault.

    # =========================================================================
    # Helpers
    # =========================================================================

    # Substrings (case-insensitive) that mark indata.systs as discrete
    # NP-template variations of one of our 8 continuous λ parameters.
    # Matching is case-insensitive because the histmaker uses inconsistent
    # casing (canonical names are uppercase Lambda, but some configurations
    # serialize them lowercase).
    #
    # Canonical names (see theory_variation_labels.py):
    #   chargeVgenNP0scetlibNPZLambda2          → catches "scetlibnpzlambda"
    #   chargeVgenNP0scetlibNPZLambda4          → catches "scetlibnpzlambda"
    #   chargeVgenNP0scetlibNPZDelta_Lambda2    → catches "scetlibnpzdelta"
    #   chargeVgenNP0scetlibNPLambda2  (W-side) → catches "scetlibnplambda"
    #   chargeVgenNP0scetlibNPLambda4  (W-side) → catches "scetlibnplambda"
    #   chargeVgenNP0scetlibNPDelta_Lambda2     → catches "scetlibnpdelta"
    #   scetlibNPgamma                          → catches "scetlibnpgamma"
    #   scetlibNPgammaEigvar{1,2,3}             → catches "scetlibnpgamma"
    #   scetlibNPgammaLambda{2,4,Inf}           → catches "scetlibnpgamma"
    _DISCRETE_NP_PATTERNS = (
        "scetlibnpzlambda",  # Z-side Lambda2 / Lambda4 templates
        "scetlibnpzdelta",  # Z-side Delta_Lambda2 template
        "scetlibnplambda",  # W-side Lambda2 / Lambda4 templates
        "scetlibnpdelta",  # W-side Delta_Lambda2 template
        "scetlibnpgamma",  # all γ_ν templates (Lambda2/4/Inf, Eigvar1/2/3, "gamma")
    )

    def _check_discrete_np_double_counting(self, freeze_patterns):
        """Detect indata systs that overlap with our continuous λ POUs.

        Three outcomes:
          - The discrete NP syst is **absent** from ``indata.systs`` entirely
            (histmaker didn't include it): nothing to do, silent return.
          - The discrete NP syst is **present and matched** by
            ``freeze_patterns``: it's frozen at central → no double-counting,
            silent return.
          - The discrete NP syst is **present and unfrozen**: it overlaps
            with one of our continuous λ POUs and will absorb shape variation
            the ParamModel should describe → print a loud banner with the
            exact freeze args to add.
        """

        systs = getattr(self.indata, "systs", None)
        if systs is None or len(systs) == 0:
            return
        syst_names = [s.decode() if isinstance(s, bytes) else str(s) for s in systs]

        # Case-insensitive substring match: canonical names use uppercase
        # Lambda but some histmaker outputs lowercase the names.
        conflicting = [
            s
            for s in syst_names
            if any(pat in s.lower() for pat in self._DISCRETE_NP_PATTERNS)
        ]
        if not conflicting:
            return  # not in indata.systs at all → nothing to warn about

        # Which of those are NOT already covered by a user-supplied freeze
        # pattern (exact match or anchored regex)?
        patterns = list(freeze_patterns or [])
        unfrozen = []
        for s in conflicting:
            covered = False
            for pat in patterns:
                if pat == s:
                    covered = True
                    break
                try:
                    if re.fullmatch(pat, s):
                        covered = True
                        break
                except re.error:
                    continue
            if not covered:
                unfrozen.append(s)

        if not unfrozen:
            return  # all conflicting systs are already frozen by the user

        print(
            "\n"
            "===================================================================\n"
            "[SCETlibNPParamModel] DOUBLE-COUNTING WARNING\n"
            "===================================================================\n"
            f"Detected {len(unfrozen)} discrete NP κ-template syst(s) in the\n"
            "input HDF5 that describe the same physics as this ParamModel's\n"
            "continuous λ parameters. Running both leads to double-counting:\n"
            "the discrete syst absorbs shape variation that the ParamModel\n"
            "should describe (the indata syst will show a spurious pull, and\n"
            "the postfit λ values are not what the data actually prefers).\n\n"
            "Unfrozen conflicting systs:\n"
            + "\n".join(f"    {s}" for s in unfrozen)
            + "\n\n"
            "Fix by adding to --freezeParameters, e.g.:\n"
            "    --freezeParameters '.*scetlibNPZ.*lambda.*' "
            "'.*scetlibNPgammaLambda.*' ...\n"
            "or list them explicitly:\n"
            "    --freezeParameters " + " ".join(repr(s) for s in unfrozen) + "\n"
            "===================================================================",
            flush=True,
        )

    def _fit_reco_axes(self, indata):
        """Read the (name, edges) of each reco axis from the (single) channel.

        Multi-channel support is deferred to v2; this raises if there's >1
        non-masked channel.
        """
        non_masked = [
            (name, info)
            for name, info in indata.channel_info.items()
            if not info.get("masked", False)
        ]
        if len(non_masked) != 1:
            raise NotImplementedError(
                f"SCETlibNPParamModel v1 supports a single non-masked channel; "
                f"got {len(non_masked)}: {[n for n, _ in non_masked]}"
            )
        _, info = non_masked[0]
        return [
            (ax.name, np.asarray(ax.edges, dtype=np.float64)) for ax in info["axes"]
        ]

    # =========================================================================
    # σ_gen evaluation
    # =========================================================================

    def _sigma_YqT_native_at(self, eff_params, gnu_params):
        """Reconstruct σ(λ) on the btgrid and Q-integrate, returning the result
        in the btgrid's *native* binning: shape (NY, NqT) on the signed-Y /
        qT grid (Y_unique, qT_unique), BEFORE the |Y|-fold and qT-rebin. This
        is the object that the native-binning validation compares against the
        SCETlib spectrum reference (curve 1) and the numpy `factorize` (curve 2)."""
        # 1. Reconstruct σ on the btgrid's flat (Nbins,) layout.
        sigma_flat = fz_tf.reconstruct_batch_tf(
            qT_per_bin=self.qT_per_bin,
            bT=self.bT,
            I_pert=self.I_pert,
            C_nu=self.C_nu,
            b_bar=self.b_bar,
            Y_per_bin=self.Y_per_bin,
            eff_params={k: v for k, v in eff_params.items() if k != "np_model"},
            gnu_params={k: v for k, v in gnu_params.items() if k != "np_model_nu"},
            np_model=self.np_model,
            np_model_nu=self.np_model_nu,
            bT_J0_kernel=self.bT_J0_kernel,
            bT_simpson_weights=self.bT_simpson_w,
            Y_unique=self.Y_feff_unique,
            Y_inverse_idx=self.Y_feff_inverse_idx,
        )
        # 2. Sparse → dense (NQ, NY, NqT). Missing cells get 0.
        sigma_dense = fz_int.sparse_to_dense_tf(sigma_flat, self.flat_idx)
        # 3. Integrate over Q (arctan_Q² Simpson) → (NY, NqT).
        return fz_int.integrate_over_Q_tf(sigma_dense, self.Q_weights)

    def _sigma_gen_at(self, eff_params, gnu_params):
        """Evaluate σ_gen(λ) on R's gen binning. Returns shape (NptVGen, NabsYVGen)."""
        sigma_YqT = self._sigma_YqT_native_at(eff_params, gnu_params)
        # 4. Rebin Y (signed) → absYVGen (|Y|-folded): (NabsYVGen, NqT).
        sigma_absY_qT = fz_int.rebin_axis_tf(sigma_YqT, axis=0, weights=self.W_absY)
        # 5. Rebin qT → ptVGen: (NabsYVGen, NptVGen).
        sigma_absY_ptV = fz_int.rebin_axis_tf(
            sigma_absY_qT, axis=1, weights=self.W_ptVGen
        )
        # 6. Reorder to (NptVGen, NabsYVGen) to match R's gen axis order.
        sigma_resum = tf.transpose(sigma_absY_ptV, perm=[1, 0])
        # 7. Add the NP-independent fixed-order/DYTurbo nonsingular (zeros if off).
        return sigma_resum + self.sigma_ns

    # =========================================================================
    # λ-vector helpers
    # =========================================================================

    def _eff_gnu_from_array(self, lambdas_np):
        """Helper: numpy 8-vector → (eff_params dict, gnu_params dict)."""
        eff = {"np_model": self.np_model}
        gnu = {"np_model_nu": self.np_model_nu}
        for i, name in enumerate(self._param_order):
            v = float(lambdas_np[i])
            if name in EFF_PARAMS:
                eff[name] = v
            elif name in GNU_PARAMS:
                gnu[name] = v
            else:
                raise KeyError(name)
        return eff, gnu

    # =========================================================================
    # compute
    # =========================================================================

    def _unpack_params(self, param):
        """Map flat param tensor to eff/gnu dicts in canonical order."""
        eff_params = {"np_model": self.np_model}
        gnu_params = {"np_model_nu": self.np_model_nu}
        # param values are stored directly (allowNegativeParam=True).
        for i, name in enumerate(self._param_order):
            v = param[i]
            if name in EFF_PARAMS:
                eff_params[name] = v
            elif name in GNU_PARAMS:
                gnu_params[name] = v
            else:
                raise KeyError(name)
        return eff_params, gnu_params

    def compute(self, param, full=False):
        """Return per-(bin, proc) scaling tensor.

        Shape: (N_reco, N_proc). Signal-proc column carries the per-reco-bin
        ratio σ_reco(λ; b) / σ_reco(λ_central; b); other columns are 1.
        """
        import time

        t_start = time.perf_counter()
        eff_params, gnu_params = self._unpack_params(param)
        sigma_gen = self._sigma_gen_at(eff_params, gnu_params)
        # Fold σ_gen(λ) through the normalized migration response P (self.R).
        gen_flat = tf.reshape(sigma_gen, [-1])
        sigma_reco = tf.linalg.matvec(self.R, gen_flat)  # (N_reco,)
        ratio = sigma_reco / self.sigma_reco_central  # (N_reco,)

        # Build (N_reco, N_proc) scaling: ones except signal column.
        N_reco = int(self.sigma_reco_central.shape[0])
        col_ones = tf.ones([N_reco, self.nproc], dtype=self.indata.dtype)
        # Place ratio into the signal column.
        ratio_col = tf.cast(tf.reshape(ratio, [N_reco, 1]), self.indata.dtype)
        # mask: one-hot vector for signal_proc_idx.
        mask = tf.one_hot(
            self.signal_proc_idx, self.nproc, dtype=self.indata.dtype
        )  # (N_proc,)
        # rnorm = ones + (ratio - 1) * one_hot_signal
        # → (ratio at signal col, 1 elsewhere)
        rnorm = col_ones + (ratio_col - 1.0) * mask[tf.newaxis, :]

        # ---- Diagnostic timing ----
        if not hasattr(self, "_n_compute_calls"):
            self._n_compute_calls = 0
            self._t_compute_total = 0.0
            import atexit

            atexit.register(self._print_compute_summary)
        self._n_compute_calls += 1
        self._t_compute_total += time.perf_counter() - t_start
        if self._n_compute_calls % 100 == 0:
            n, t = self._n_compute_calls, self._t_compute_total
            print(
                f"[SCETlibNPParamModel] compute() running: {n} calls, "
                f"total {t:.1f}s, mean {t*1000/n:.1f} ms/call",
                flush=True,
            )
        return rnorm

    def _print_compute_summary(self):
        """Print final tally of compute() calls and time. Atexit hook."""
        n = getattr(self, "_n_compute_calls", 0)
        t = getattr(self, "_t_compute_total", 0.0)
        if n == 0:
            return
        print(
            f"[SCETlibNPParamModel] compute() FINAL: {n} calls, "
            f"total {t:.2f}s wall in compute(), "
            f"mean {t*1000/n:.2f} ms/call",
            flush=True,
        )
