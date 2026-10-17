"""ASWZ lattice CS-kernel chi2 with the kernel computed by SCETlib AT EVERY STEP (no theory tables).

Adds 1/2 * (chi2_lat - offset) to the NLL, where

    chi2_lat = min_k1  r^T C^-1 r ,     r_i = gamma_zeta(b_i, mu = 2 GeV; p_full) + k1 a_i / b_i - y_i

over the 21 ASWZ per-ensemble lattice points (arXiv:2402.06725). The difference to the earlier table-based term
(``lattice_cs_chi2.LatticeCSChi2``, now removed) is where gamma_zeta comes from:

* gamma_zeta = gamma_nu / 2 is SCETlib's own ``ad::gamma_nu_resummed`` -- the function the cross section's AD kernel
  calls for the beam and soft rapidity evolution -- through ``DrellYan.gamma_nu_points`` and
  ``scetlib_tf.ScetlibGammaNuTF`` (scetlib-cms branch ``gamma-nu-points``). Perturbative AND NP part, value, gradient
  and Hessian exact (clad), at every step.
* Its fitted inputs come from ``p_full = SCETlibADParamModel.scetlib_full_vector_tf(params)``: the SAME tensor the
  cross section is evaluated at, in the physical (offset-applied) frame rabbit hands the model, never the blinded
  internal ``Fitter.x``, and through one theta -> physical map (no second implementation of it). With the default
  ``pert=frozen`` only the CS NP lambdas are taken from it (alpha_s and the CS TNPs held at a fixed reference); with
  ``pert=live`` alpha_s, the CS-kernel TNPs (tnp_gamma_cusp, tnp_gamma_nu) and the CS NP lambdas are all exactly what
  the Z prediction sees. See PERTURBATIVE PART.
* The kernel's coefficients are a snapshot of the loaded calculation's, checked BITWISE against the coefficients stored
  in the first loaded cache rule (``gamma_nu_snapshot_status() == 1``). Hence the same RGE as the cache: SCETlib's
  ANALYTIC running and evolution (the AD kernel has no other), where the old table used the exact RGE (<= 1.9e-3 in
  gamma_zeta, -8.5 % in d gamma_zeta / d alpha_s on the plateau; 261007-lattice-term-design).
* Only DATA is read: ``lattice_aswz_data.npz`` (points, b_T, a, ensembles, covariance; provenance in the .json next
  to it). It is NOT in this repository; see THE LATTICE DATA.

k1 (lattice-spacing artefact, k1 a/b_T) is PROFILED ANALYTICALLY: chi2 = r0^T M r0, M = W - (W v)(W v)^T / (v^T W v),
W = C^-1, v = a/b, r0 = r(k1 = 0). Identical to a free k1 parameter (a rabbit regularizer cannot own parameters).

Covariance C = C_stat + sum_g delta_g delta_g^T (each delta_g a point shift = a profiled unit-Gaussian nuisance), all
computed AT LOAD from SCETlib, at the load-time reference (see PERTURBATIVE PART), with lattice-only refits of
(lambda2_nu, lambda4_nu) on the stat covariance, k1 profiled:

    Jnf        n_f scheme. Alternative kernel (``gamma_nu_points(nf=4, mu_match=nfmatch)``: n_f = 4 in beta, cusp and
               boundary, alpha_s^(4)(nfmatch) := alpha_s^(5)(nfmatch), the coupling IDENTIFIED at nfmatch):
                 nfscheme=full (default): the whole kernel in n_f = 4 (boundary at mu0 and evolution to 2 GeV),
                   alt = z4(2 GeV); with nfmatch=4.18 (default) that is "n_f = 4 below m_b, coupling identified at
                   m_b(m_b)".
                 nfscheme=evolve: our n_f = 5 kernel at mu = nfswitch, evolved to 2 GeV with n_f = 4,
                   alt = z5(nfswitch) + [z4(2 GeV) - z4(nfswitch)]; nfswitch defaults to nfmatch. nfmatch=1: the
                   SCETlib-native "n_f identified at 1 GeV" (LATFROZ_V1); nfswitch=1 nfmatch=4.18: the old table's
                   convention (n_f = 4 evolution 1 -> 2 GeV with alpha_s^(4) from m_b(m_b), identified instead of
                   decoupled; LATFROZ_V2). nfswitch needs nfscheme=evolve given explicitly.
               The shift alt - nominal is computed once at the load-time reference (the NP part cancels exactly).
               d = lambda_hat(alt) - lambda_hat(nominal), delta = J_NP d.
    Jbt        b_T window: d = lambda_hat(b_T >= 0.2 fm) - lambda_hat(all), delta = J_NP d. Opt-in only
               (``syst=Jnf+Jbt``): dropped from the default by Luca 2026-10-08.
    direct_nf  the n_f group as the direct point shift alt - nominal (instead of Jnf).
    none       stat only.
    default    ``Jnf``.

J_NP is SCETlib's Jacobian d gamma_zeta / d(lambda2_nu, lambda4_nu) at the nominal lattice-only fit. No mu0 variation
and no k-form (Luca 2026-10-06/07: missing higher orders are the TNPs' job; k1 a/b_T is the lattice authors'
prescription).

``offset=min`` (default): the lattice-only chi2_min at the reference with the final covariance, so the term is
1/2 Delta chi2. ``ydata=asimov``: y = gamma_zeta(reference) + k1asimov a/b (closure; use ``offset=0``).

PERTURBATIVE PART: ``pert=frozen alphas_frozen=0.1168`` (default): the kernel compared with the lattice points is
evaluated at a FIXED reference -- the param model's anchor with alpha_s(mZ) := alphas_frozen (default 0.1168, the
value the ASWZ matching corresponds to: alpha_s^(4)(2 GeV) = 0.293) and the CS TNPs (tnp_gamma_cusp, tnp_gamma_nu)
:= 0 -- except the CS NP lambdas (np_gnu_lambda2, np_gnu_lambda4), which come from p_full. Every other entry,
lambda_inf_nu included, stays at that reference. The lattice then constrains only the CS NP parameters: its gradient
and Hessian with respect to alpha_s and the TNPs are exactly zero, and all load-time quantities (n_f shift, refits,
J_NP, covariance, offset) are computed at that reference. The cross section is untouched (alpha_s and the TNPs float
there as usual). The term's value then depends on public parameters only and may be printed. ``pert=live`` (opt-in)
evaluates the kernel at p_full, so alpha_s and the CS TNPs move the lattice comparison exactly as they move the cross
section; its load-time reference is the param model's anchor.

DEFAULTS (Luca 2026-10-09): a bare ``-r ...LatticeCSTerm ...LatticeCSTermMapping`` is the LATFROZ_V3 nominal,

    pert=frozen alphas_frozen=0.1168 syst=Jnf nfmatch=4.18 nfscheme=full offset=min

(WRemnantsHelpers studies/lattice-cs-kernel, task 261008-latfroz-nf-variants; the b_T window was dropped 2026-10-08,
V3 chosen 2026-10-09). Everything else stays available as an explicit option.

REPRODUCING OLDER FITS: commands written before 2026-10-09 relied on the old defaults pert=live syst=Jnf+Jbt
nfmatch=1 nfscheme=evolve. To rerun one, add what it left out: the LATB5/LATB8/SAT* era (``syst=Jnf+Jbt
offset=min``) needs ``pert=live nfmatch=1 nfscheme=evolve``; LATFROZ_V1 (``nfmatch=1.0``) and LATFROZ_V2
(``nfmatch=4.18 nfswitch=1.0``) need ``nfscheme=evolve`` (V1 would otherwise silently become the full-n_f=4 scheme at
1 GeV; V2 is refused). LATFROZ_V3's own command is unchanged by the new defaults.

THE LATTICE DATA are private files from the ASWZ authors (arXiv:2402.06725) and are not distributed: never commit
them, or a copy or reformatting of them, to a repository. They live on the group data area, as a sibling of the NanoAOD
directory that ``dataset_tools.getDataPath()`` returns (``WREMNANTS_DATA_PATH`` if set, else the per-host default; it
points at the NanoAOD directory itself, and the lattice directory is its sibling):

    <dirname(getDataPath())>/lattice/ASWZ2024/lattice_aswz_data.npz   (+ lattice_aswz_data.json, the provenance)
    MIT:  /scratch/submit/cms/wmass/lattice/ASWZ2024/      (getDataPath() = /scratch/submit/cms/wmass/NanoAOD)
    CERN: /scratch/shared/lattice/ASWZ2024/                (getDataPath() = /scratch/shared/NanoAOD)

``author_files/`` in the same directory holds the authors' original CSVs the .npz was built from. ``data=<npz>`` on
the -r line overrides the default path. A missing file is an error (there is no fallback copy); access is restricted
to the analysis group, so ask for a copy rather than fetching one from elsewhere.

THE TABLE-BASED TERM ``lattice_cs_chi2.LatticeCSChi2`` and its inputs (``data/lattice_aswz_inputs.npz``) were removed
in 1b3732bb (superseded by this term); ``git show 3ef3efb9`` has their last version.

COMPOSITE MODELS (the saturated test): when the fitter's param model is a rabbit ``CompositeParamModel`` with exactly
one SCETlibADParamModel among its direct submodels, the SCETlib submodel's own [poi | pou] vector is rebuilt from the
composite layout by running the composite class's own ``compute`` on recorders (``submodel_vector``), so the term
follows the composite's permutation instead of re-deriving it. Anything else is refused.

THE exp(2 tau) COMPENSATION: rabbit multiplies every regularizer penalty by exp(2 tau) (tau = --regularizationStrength);
this term divides it back out with the live ``fitter.tau``, found on the call stack at ``set_expectations``. If no
fitter is found there and no ``tau=`` is on the -r line, arming FAILS: the penalty would otherwise go in unscaled and
come out exp(2 tau) too strong (e^16 at tau = 8). Pass ``tau=<--regularizationStrength>`` explicitly then; offline /
standalone evaluation (``param_model_override``, no fitter) must pass ``tau=`` too (``tau=0`` for the bare
1/2 (chi2 - offset)).

BLINDING: nothing evaluated at the live parameter vector is printed, logged or raised (gamma_zeta at the points, chi2,
k1_hat, residuals, derivatives). Only load-time quantities at the public reference are printed.

Invoke (composes with the wall; each on its own -r):

    rabbit_fit.py ... --paramModel wremnants.postprocessing.scetlib_ad.SCETlibADParamModel ... \\
      -r wremnants.postprocessing.scetlib_ad.np_damping_wall.NPDampingWall \\
         wremnants.postprocessing.scetlib_ad.np_damping_wall.NPDampingMapping margin=0 \\
      -r wremnants.postprocessing.scetlib_ad.lattice_cs_term.LatticeCSTerm \\
         wremnants.postprocessing.scetlib_ad.lattice_cs_term.LatticeCSTermMapping \\
         [pert=frozen|live] [alphas_frozen=0.1168] [syst=Jnf|Jnf+Jbt|direct_nf|none] [nfmatch=4.18]
         [nfscheme=full|evolve] [nfswitch=<GeV>] [offset=min|<float>] [ydata=lattice|asimov] [k1asimov=0.2]
         [tau=<float>] [data=<npz>]

Requires a SCETlib build with ``DrellYan.gamma_nu_points`` (scetlib-cms >= 6ab371a on branch gamma-nu-points).
"""

import inspect
import os

import numpy as np

LATTICE_SUBDIR = os.path.join("lattice", "ASWZ2024")
LATTICE_FILE = "lattice_aswz_data.npz"
MU_LATTICE = 2.0  # GeV: the ASWZ MSbar scale
NF_ALT = 4  # the lattice's flavour number
DEFAULT_NF_MATCH = (
    4.18  # GeV, m_b(m_b): alpha_s^(4) identified with alpha_s^(5) here (V3)
)
BT_WINDOW_FM = 0.2
DEFAULT_SYST = "Jnf"  # the b_T window (Jbt) is opt-in since 2026-10-08
SYST_COMPONENTS = ("Jnf", "Jbt", "direct_nf", "none")
FIT_LAMBDAS = ("np_gnu_lambda2", "np_gnu_lambda4")  # SCETlib names refit at load
NF_SCHEMES = ("evolve", "full")
DEFAULT_NF_SCHEME = "full"  # the whole kernel in n_f = 4 (V3)
PERT_MODES = ("live", "frozen")
DEFAULT_PERT = "frozen"  # the lattice constrains the CS NP lambdas only (V3)
DEFAULT_ALPHAS_FROZEN = (
    0.1168  # ASWZ: alpha_s^(4)(2 GeV) = 0.293 <-> alpha_s(mZ) = 0.1168 (m_b decoupling)
)
FROZEN_TNPS = (
    "tnp_gamma_cusp",
    "tnp_gamma_nu",
)  # the CS-kernel TNPs, set to 0 in the frozen reference
_KEYS = (
    "data",
    "syst",
    "offset",
    "tau",
    "ydata",
    "k1asimov",
    "nfmatch",
    "nfswitch",
    "nfscheme",
    "pert",
    "alphas_frozen",
    "rules",
)


def default_data_path():
    """``<parent of getDataPath()>/lattice/ASWZ2024/lattice_aswz_data.npz``: the lattice directory is a sibling of the
    NanoAOD directory ``getDataPath()`` returns (``WREMNANTS_DATA_PATH`` or the per-host default).
    """
    # lazy: dataset_tools imports ROOT, XRootD and narf, which a fit that never arms this term should not pay for
    from wremnants.production.datasets.dataset_tools import getDataPath

    try:
        nano = getDataPath()
    except RuntimeError as e:
        raise RuntimeError(
            f"LatticeCSTerm: cannot locate the lattice data: {e} (WREMNANTS_DATA_PATH is the NanoAOD directory; the "
            f"lattice data are expected at <its parent>/{LATTICE_SUBDIR}/{LATTICE_FILE}), or give data=<npz>"
        ) from e
    return os.path.join(
        os.path.dirname(os.path.normpath(nano)), LATTICE_SUBDIR, LATTICE_FILE
    )


def resolve_data_path(data=None):
    """The lattice .npz to read: ``data`` if given, else ``default_data_path()``. Raises FileNotFoundError, naming the
    path, ``WREMNANTS_DATA_PATH`` and ``data=``, if it does not exist (the data are private and never in the repo).
    """
    explicit = data not in (None, "")
    path = str(data) if explicit else default_data_path()
    if not os.path.isfile(path):
        origin = (
            "given as data="
            if explicit
            else "the default <dirname(getDataPath())>/"
            f"{LATTICE_SUBDIR}/{LATTICE_FILE}; WREMNANTS_DATA_PATH="
            f"{os.environ.get('WREMNANTS_DATA_PATH', '<unset, per-host default>')}"
        )
        raise FileNotFoundError(
            f"LatticeCSTerm: lattice data file not found: {path} ({origin}). The ASWZ lattice data are private author "
            "files, not in the repository; they live on the group data area as a sibling of the NanoAOD directory "
            f"(MIT: /scratch/submit/cms/wmass/{LATTICE_SUBDIR}/). Point WREMNANTS_DATA_PATH at a NanoAOD directory "
            "whose parent holds lattice/ASWZ2024/, or pass data=<path to lattice_aswz_data.npz> on the -r line."
        )
    return path


def frozen_reference(names, p_anchor, alphas_frozen):
    """The pert=frozen reference: the anchor with alpha_s := alphas_frozen and the CS TNPs := 0."""
    p = np.array(p_anchor, float).copy()
    p[list(names).index("alphas")] = float(alphas_frozen)
    for n in FROZEN_TNPS:
        p[list(names).index(n)] = 0.0
    return p


def _import_gamma_nu_tf():
    try:
        from scetlib_tf import ScetlibGammaNuTF
    except ImportError as e:
        raise ImportError(
            "lattice_cs_term needs scetlib_tf.ScetlibGammaNuTF and DrellYan.gamma_nu_points (scetlib-cms branch "
            "gamma-nu-points, >= 6ab371a). Run with that build (agent_setup.sh --scetlib <its build dir>). "
            f"Original error: {e}"
        ) from e
    return ScetlibGammaNuTF


def _profile_matrix(cov, v):
    """M with chi2 = r0^T M r0 = min_k1 (r0 + k1 v)^T C^-1 (r0 + k1 v)."""
    W = np.linalg.inv(cov)
    W = 0.5 * (W + W.T)
    Wv = W @ v
    M = W - np.outer(Wv, Wv) / float(v @ Wv)
    return 0.5 * (M + M.T), Wv, float(v @ Wv)


def _syst_components(syst):
    comps = [c for c in str(syst).split("+") if c]
    bad = [c for c in comps if c not in SYST_COMPONENTS]
    if bad or not comps:
        raise ValueError(f"syst={syst!r}: components must be from {SYST_COMPONENTS}")
    if "none" in comps and len(comps) > 1:
        raise ValueError(f"syst={syst!r}: 'none' cannot be combined")
    if "Jnf" in comps and "direct_nf" in comps:
        raise ValueError(f"syst={syst!r}: give the n_f group once (Jnf or direct_nf)")
    return [c for c in comps if c != "none"]


class LatticeCSNativeCore:
    """The numpy side: data, the SCETlib kernel, the load-time refits and the covariance.

    ``sing``: the resummed SCETlib piece the cross section replays (rules loaded); ``p_ref``: the full SCETlib vector at
    the anchor (alpha_s, TNPs, lambda_inf_nu of the param model's anchor), at which every load-time quantity is
    computed. Nothing here depends on a fitted value.
    """

    def __init__(
        self,
        sing,
        p_ref,
        data=None,
        syst=DEFAULT_SYST,
        nf_match=DEFAULT_NF_MATCH,
        ydata="lattice",
        k1asimov=0.2,
        require_rules=True,
        nf_switch=None,
        nf_scheme=DEFAULT_NF_SCHEME,
    ):
        data = resolve_data_path(data)
        ScetlibGammaNuTF = _import_gamma_nu_tf()
        d = np.load(data)
        self.data_path = data
        self.b_fm = np.array(d["b_fm"], float)
        self.a_fm = np.array(d["a_fm"], float)
        self.y_lattice = np.array(d["y"], float)
        self.cov_stat = np.array(d["cov_stat"], float)
        self.fm_to_gevinv = float(d["fm_to_gevinv"])
        self.mu = float(d["mu_gev"]) if "mu_gev" in d else MU_LATTICE
        self.bT = self.b_fm * self.fm_to_gevinv
        self.v = self.a_fm / self.b_fm
        self.sing = sing
        self.gz = ScetlibGammaNuTF(sing, self.bT, mu=self.mu)
        if require_rules and self.gz.snapshot_status != 1:
            raise ValueError(
                "LatticeCSTerm: the SCETlib kernel snapshot could not be checked against loaded cache rules "
                f"(status {self.gz.snapshot_status}); refusing (rules=any overrides, for offline tests only)"
            )
        self.names = list(self.gz.param_names)
        self.p_ref = np.array(p_ref, float)
        if self.p_ref.shape != (len(self.names),):
            raise ValueError(
                "LatticeCSTerm: p_ref does not match the SCETlib parameter vector"
            )
        self.ilam = [self.names.index(n) for n in FIT_LAMBDAS]
        self.syst = str(syst)
        self.nf_match = float(nf_match)
        self.nf_switch = self.nf_match if nf_switch is None else float(nf_switch)
        if nf_scheme not in NF_SCHEMES:
            raise ValueError(f"nf_scheme={nf_scheme!r} ({'|'.join(NF_SCHEMES)})")
        self.nf_scheme = nf_scheme
        self.ydata = ydata
        if ydata == "asimov":
            self.k1asimov = float(k1asimov)
            self.y = self.zeta(self.p_ref) + self.k1asimov * self.v
        elif ydata == "lattice":
            self.y = self.y_lattice.copy()
        else:
            raise ValueError(f"ydata={ydata!r} (lattice|asimov)")

        # ---- n_f-scheme alternative, SCETlib-native (a point shift; the NP part cancels exactly)
        self.nf_shift = self._nf_shift(self.nf_match, self.nf_switch, self.nf_scheme)

        # ---- lattice-only refits on the stat covariance (k1 profiled), at the anchor
        all_pts = np.arange(len(self.y))
        win = np.flatnonzero(self.b_fm >= BT_WINDOW_FM - 1e-12)
        M_stat = _profile_matrix(self.cov_stat, self.v)[0]
        self.fit_nominal = self.refit(M_stat, all_pts)
        comps = _syst_components(syst)
        self.syst_rows, self.syst_info = [], {}
        lam0 = self.fit_nominal["lam"]
        J0 = self.zeta(self._p_lam(lam0), order=1)[1][:, self.ilam]
        self.J_np = J0
        for c in comps:
            if c == "Jnf":
                f = self.refit(M_stat, all_pts, shift=self.nf_shift)
                dl = f["lam"] - lam0
                self.syst_rows.append(J0 @ dl)
                self.syst_info[c] = dict(dlam=dl.tolist(), chi2min=f["chi2"])
            elif c == "Jbt":
                Mw = _profile_matrix(self.cov_stat[np.ix_(win, win)], self.v[win])[0]
                f = self.refit(Mw, win)
                dl = f["lam"] - lam0
                self.syst_rows.append(J0 @ dl)
                self.syst_info[c] = dict(
                    dlam=dl.tolist(), chi2min=f["chi2"], npts=int(len(win))
                )
            elif c == "direct_nf":
                self.syst_rows.append(self.nf_shift.copy())
                self.syst_info[c] = dict(
                    shift_range=[float(self.nf_shift.min()), float(self.nf_shift.max())]
                )
        self.cov = self.cov_stat + sum(
            (np.outer(s, s) for s in self.syst_rows), np.zeros_like(self.cov_stat)
        )
        self.M, self.Wv, self.vWv = _profile_matrix(self.cov, self.v)
        self.fit_final = self.refit(self.M, all_pts)
        self.chi2_min = self.fit_final["chi2"]

    # ------------------------------------------------------------------ kernel
    def zeta(self, p, order=0, mu=None, nf=0, mu_match=0.0):
        """gamma_zeta = gamma_nu / 2 at the points; order 1/2 adds the Jacobian / Hessians (same units)."""
        r = self.sing.gamma_nu_points(
            self.bT,
            self.mu if mu is None else mu,
            np.ascontiguousarray(p, float),
            order,
            nf,
            mu_match,
        )
        out = [0.5 * np.asarray(r["value"])]
        if order >= 1:
            out.append(0.5 * np.asarray(r["grad"]))
        if order >= 2:
            out.append(0.5 * np.asarray(r["hess"]))
        return out[0] if order == 0 else tuple(out)

    def _nf_shift(self, mu_match, mu_switch=None, scheme="evolve"):
        """alt - nominal at p_ref, with z4 = the n_f = 4 kernel whose coupling is identified at mu_match:

        evolve: alt = z5(mu_switch) + [z4(mu) - z4(mu_switch)]  (n_f = 4 evolution mu_switch -> mu; mu_switch defaults
                to mu_match, the original single-scale form)
        full:   alt = z4(mu)                                    (boundary and evolution in n_f = 4)
        """
        p = self.p_ref
        z5_mu = self.zeta(p)
        z4_mu = self.zeta(p, nf=NF_ALT, mu_match=mu_match)
        if scheme == "full":
            return z4_mu - z5_mu
        if scheme != "evolve":
            raise ValueError(f"nf scheme {scheme!r} ({'|'.join(NF_SCHEMES)})")
        mu_switch = mu_match if mu_switch is None else mu_switch
        z5_s = self.zeta(p, mu=mu_switch)
        z4_s = self.zeta(p, mu=mu_switch, nf=NF_ALT, mu_match=mu_match)
        return (z5_s + z4_mu - z4_s) - z5_mu

    def _p_lam(self, lam):
        p = self.p_ref.copy()
        p[self.ilam] = lam
        return p

    # ------------------------------------------------------------------ chi2 (numpy)
    def chi2_full(self, p, M=None, idx=None, shift=None):
        """chi2 (k1 profiled) at a full SCETlib vector p; M / idx / shift default to the term's own."""
        z = self.zeta(p)
        if shift is not None:
            z = z + shift
        idx = slice(None) if idx is None else idx
        r = (z - self.y)[idx]
        M = self.M if M is None else M
        return float(r @ M @ r)

    def k1hat(self, p):
        r = self.zeta(p) - self.y
        return float(-(self.Wv @ r) / self.vWv)

    def refit(self, M, idx, shift=None, x0=None):
        """Lattice-only minimum in (lambda2_nu, lambda4_nu) at p_ref otherwise (exact SCETlib grad / Hessian)."""
        from scipy.optimize import minimize

        sh = np.zeros(len(self.y)) if shift is None else shift

        def parts(x):
            z, J, H = self.zeta(self._p_lam(x), order=2)
            r = (z + sh - self.y)[idx]
            Jl = J[idx][:, self.ilam]
            Hl = H[idx][:, self.ilam][:, :, self.ilam]
            Mr = M @ r
            return (
                float(r @ Mr),
                2.0 * Jl.T @ Mr,
                2.0 * (Jl.T @ M @ Jl + np.einsum("n,nij->ij", Mr, Hl)),
            )

        x = np.array(self.p_ref[self.ilam] if x0 is None else x0, float)
        res = minimize(
            lambda q: parts(q)[0],
            x,
            jac=lambda q: parts(q)[1],
            hess=lambda q: parts(q)[2],
            method="trust-exact",
            options=dict(gtol=1e-10, maxiter=200),
        )
        f, g, h = parts(res.x)
        return dict(
            lam=np.array(res.x),
            chi2=f,
            grad=g,
            hess=h,
            nit=int(res.nit),
            success=bool(res.success),
        )

    def summary(self):
        """Load-time numbers at the public anchor (alpha_s, TNPs of the anchor): safe to print under blinding."""
        fn, ff = self.fit_nominal, self.fit_final
        cov_gn = np.linalg.inv(0.5 * ff["hess"])
        return dict(
            data=self.data_path,
            syst=self.syst,
            nf_match=self.nf_match,
            nf_switch=self.nf_switch,
            nf_scheme=self.nf_scheme,
            ydata=self.ydata,
            anchor_alphas=float(self.p_ref[self.names.index("alphas")]),
            nominal_stat_fit=dict(lam=fn["lam"].tolist(), chi2=fn["chi2"]),
            final_fit=dict(
                lam=ff["lam"].tolist(),
                chi2=ff["chi2"],
                sigma=np.sqrt(np.diag(cov_gn)).tolist(),
                rho=float(cov_gn[0, 1] / np.sqrt(cov_gn[0, 0] * cov_gn[1, 1])),
            ),
            nf_shift_range=[float(self.nf_shift.min()), float(self.nf_shift.max())],
            syst_info=self.syst_info,
            chi2_min=self.chi2_min,
        )


def _make_mapping_class():
    from rabbit.mappings.mapping import BaseMapping

    class LatticeCSTermMapping(BaseMapping):
        """Carries the -r options (key=value) to LatticeCSTerm; see the module docstring."""

        def __init__(self, indata, key, **kw):
            super().__init__(indata, key)
            self.indata = indata
            self.data = resolve_data_path(kw.get("data"))  # fail before any fit work
            self.syst = kw.get("syst", DEFAULT_SYST)
            _syst_components(self.syst)
            self.offset = kw.get("offset", "min")
            self.tau = kw.get("tau")
            self.ydata = kw.get("ydata", "lattice")
            if self.ydata not in ("lattice", "asimov"):
                raise ValueError(
                    f"LatticeCSTermMapping: ydata={self.ydata!r} (lattice|asimov)"
                )
            self.k1asimov = float(kw.get("k1asimov", 0.2))
            self.nfmatch = float(kw.get("nfmatch", DEFAULT_NF_MATCH))
            self.nfswitch = (
                float(kw["nfswitch"]) if kw.get("nfswitch") is not None else None
            )
            self.nfscheme = kw.get("nfscheme", DEFAULT_NF_SCHEME)
            if self.nfscheme not in NF_SCHEMES:
                raise ValueError(
                    f"LatticeCSTermMapping: nfscheme={self.nfscheme!r} ({'|'.join(NF_SCHEMES)})"
                )
            if self.nfscheme == "full" and self.nfswitch is not None:
                raise ValueError(
                    "LatticeCSTermMapping: nfswitch has no meaning with nfscheme=full "
                    "(the default); give nfscheme=evolve explicitly"
                )
            self.pert = kw.get("pert", DEFAULT_PERT)
            if self.pert not in PERT_MODES:
                raise ValueError(
                    f"LatticeCSTermMapping: pert={self.pert!r} ({'|'.join(PERT_MODES)})"
                )
            if self.pert == "live" and kw.get("alphas_frozen") is not None:
                raise ValueError(
                    "LatticeCSTermMapping: alphas_frozen needs pert=frozen"
                )
            self.alphas_frozen = float(kw.get("alphas_frozen", DEFAULT_ALPHAS_FROZEN))
            self.rules = kw.get("rules", "require")
            if self.rules not in ("require", "any"):
                raise ValueError(
                    f"LatticeCSTermMapping: rules={self.rules!r} (require|any)"
                )

        @classmethod
        def parse_args(cls, indata, *args):
            kw = {}
            for a in args:
                if "=" not in a:
                    raise ValueError(
                        f"LatticeCSTermMapping: args are key=value, got {a!r}"
                    )
                k, v = a.split("=", 1)
                if k not in _KEYS:
                    raise ValueError(
                        f"LatticeCSTermMapping: unknown key {k!r}; use {_KEYS}"
                    )
                kw[k] = float(v) if k == "tau" else v
            return cls(indata, " ".join([cls.__name__, *args]), **kw)

    return LatticeCSTermMapping


def _find_fitter():
    """The rabbit Fitter calling set_expectations (Fitter.arm_regularizers), or None."""
    fr = inspect.currentframe()
    try:
        f = fr.f_back
        for _ in range(8):
            if f is None:
                return None
            obj = f.f_locals.get("self")
            if obj is not None and hasattr(obj, "tau") and hasattr(obj, "regularizers"):
                return obj
            f = f.f_back
    finally:
        del fr
    return None


def _is_scetlib_model(m):
    return hasattr(m, "scetlib_full_vector_tf")


def resolve_scetlib_model(pm):
    """(the SCETlibADParamModel, its index in pm.param_models or None) for a plain or composite param model.

    A rabbit ``CompositeParamModel`` (the saturated test wraps the analysis model in one) is accepted when EXACTLY ONE
    of its direct submodels carries ``scetlib_full_vector_tf``; anything else (none, two, a composite nested inside a
    composite) is refused rather than guessed.
    """
    if _is_scetlib_model(pm):
        return pm, None
    subs = getattr(pm, "param_models", None)
    if subs is None:
        raise ValueError(
            f"LatticeCSTerm: the fitter's param model ({type(pm).__name__}) is neither SCETlibADParamModel nor a "
            "CompositeParamModel containing one"
        )
    hits = [i for i, m in enumerate(subs) if _is_scetlib_model(m)]
    nested = [i for i, m in enumerate(subs) if hasattr(m, "param_models")]
    if len(hits) != 1 or nested:
        raise ValueError(
            f"LatticeCSTerm: composite param model with {len(hits)} SCETlib submodel(s) and {len(nested)} nested "
            "composite(s); exactly one SCETlib submodel and no nesting is supported"
        )
    return subs[hits[0]], hits[0]


class _Recorder:
    """Stands in for one submodel inside a call of the composite's OWN compute(): records the vector it is handed."""

    def __init__(self, m):
        self.npoi, self.npou, self.nparams = m.npoi, m.npou, m.nparams
        self.seen = None

    def compute(self, mparam, full=False):
        self.seen = mparam
        return 1.0


def submodel_vector(composite, k, param):
    """The native [poi | pou] vector the composite hands submodel ``k``, obtained by running the composite class's
    own ``compute`` on recorders (so the permutation is the composite's, not a re-derivation of it).
    """
    import types

    recs = [_Recorder(m) for m in composite.param_models]
    proxy = types.SimpleNamespace(
        param_models=recs, npoi=composite.npoi, npou=composite.npou
    )
    type(composite).compute(proxy, param)
    return recs[k].seen


def _make_regularizer_class():
    import tensorflow as tf

    from rabbit.regularization.regularizer import Regularizer

    class LatticeCSTerm(Regularizer):
        """1/2 (chi2_lat - offset) / exp(2 tau), gamma_zeta from SCETlib at p_full; see the module docstring."""

        needs_observables = False
        # Offline use (no fitter on the stack): set to an object with the param-model interface.
        param_model_override = None

        def __init__(self, mapping, dtype):
            super().__init__(mapping, dtype)
            self.dtype = dtype
            self.mapping = mapping
            self.data = getattr(mapping, "data", None)
            self.syst = getattr(mapping, "syst", DEFAULT_SYST)
            self.offset_arg = getattr(mapping, "offset", "min")
            self.tau_arg = getattr(mapping, "tau", None)
            self.ydata = getattr(mapping, "ydata", "lattice")
            self.k1asimov = float(getattr(mapping, "k1asimov", 0.2))
            self.nfmatch = float(getattr(mapping, "nfmatch", DEFAULT_NF_MATCH))
            self.nfswitch = getattr(mapping, "nfswitch", None)
            self.nfscheme = getattr(mapping, "nfscheme", DEFAULT_NF_SCHEME)
            self.pert = getattr(mapping, "pert", DEFAULT_PERT)
            self.alphas_frozen = float(
                getattr(mapping, "alphas_frozen", DEFAULT_ALPHAS_FROZEN)
            )
            self.rules = getattr(mapping, "rules", "require")
            self.core = None
            self._core_backend = None
            self._pm = None
            self._sub = None
            self._k = None
            self._tau = None

        def __deepcopy__(self, memo):
            # rabbit deep-copies fitters (saturated model, toys); the core holds pybind handles and TF constants,
            # all immutable after arming, so they are shared rather than copied.
            new = self.__class__.__new__(self.__class__)
            memo[id(self)] = new
            new.__dict__.update(self.__dict__)
            return new

        def _build(self, pm):
            sub, k = resolve_scetlib_model(pm)
            self._pm, self._k = pm, k
            self._sub = sub
            self._npm = int(pm.nparams)
            if self.core is not None and self._core_backend is sub.core:
                return  # same SCETlib calculation (e.g. a deepcopy for the saturated test): load-time work is reused
            pm = sub
            sing = pm.core.tf_fn._sing
            if list(sing.gradient_param_names()) != list(pm.scetlib_names):
                raise ValueError(
                    "LatticeCSTerm: SCETlib parameter registry differs from the param model's"
                )
            p_ref = np.array(pm._p_base_anchor, float)
            if self.pert == "frozen":
                p_ref = frozen_reference(pm.scetlib_names, p_ref, self.alphas_frozen)
            self.core = LatticeCSNativeCore(
                sing,
                p_ref,
                data=self.data,
                syst=self.syst,
                nf_match=self.nfmatch,
                ydata=self.ydata,
                k1asimov=self.k1asimov,
                require_rules=self.rules == "require",
                nf_switch=self.nfswitch,
                nf_scheme=self.nfscheme,
            )
            if self.pert == "frozen":
                live = np.zeros(len(p_ref))
                live[self.core.ilam] = 1.0
                # p_kernel = p_frz (1 - m) + m p_full: a constant 0/1 product, so every frozen entry has an exactly
                # zero first and second derivative and the live ones pass through unchanged (bitwise)
                self._tf_pfrz = tf.constant(p_ref * (1.0 - live), dtype=self.dtype)
                self._tf_mlive = tf.constant(live, dtype=self.dtype)
                frozen_dep = [
                    n
                    for n in pm.scetlib_names
                    if n.startswith("np_gnu_") and n not in FIT_LAMBDAS
                ]
                pert_txt = (
                    f"pert=frozen: kernel at alpha_s(mZ) = {self.alphas_frozen:g}, CS TNPs "
                    f"{list(FROZEN_TNPS)} = 0, {frozen_dep} at the anchor; only {list(FIT_LAMBDAS)} from p_full"
                )
            else:
                pert_txt = "pert=live: alpha_s, CS TNPs and CS lambdas from p_full"
            if self.offset_arg == "min":
                self.offset = float(self.core.chi2_min)
            else:
                self.offset = float(self.offset_arg)
            c = lambda a: tf.constant(
                np.asarray(a, float), dtype=self.dtype
            )  # noqa: E731
            self.tf_y, self.tf_M = c(self.core.y), c(self.core.M)
            self._gz = self.core.gz
            self._core_backend = pm.core
            s = self.core.summary()
            print(
                f"[LatticeCSTerm] {len(self.core.y)} ASWZ points ({self.ydata}), data {self.data}; gamma_zeta from "
                f"SCETlib DrellYan.gamma_nu_points (snapshot status {self.core.gz.snapshot_status}: "
                f"{'== loaded rules, bitwise' if self.core.gz.snapshot_status == 1 else 'NOT checked against rules'})"
                f", mu = {self.core.mu} GeV, p_full = param model scetlib_full_vector_tf; {pert_txt}; k1 profiled. "
                f"Load-time (reference alpha_s {s['anchor_alphas']:g}, reference TNPs): "
                f"syst={self.syst} {s['syst_info']}, n_f alternative ({self.core.nf_scheme}"
                + (
                    f", n_f = 4 evolution from {self.core.nf_switch:g} GeV"
                    if self.core.nf_scheme == "evolve"
                    else ", whole kernel n_f = 4"
                )
                + f", coupling identified at {self.nfmatch:g} GeV; shift "
                f"{s['nf_shift_range'][0]:+.5f}..{s['nf_shift_range'][1]:+.5f}); stat-only lattice fit "
                f"lambda=({s['nominal_stat_fit']['lam'][0]:.5f}, {s['nominal_stat_fit']['lam'][1]:.6f}) chi2 "
                f"{s['nominal_stat_fit']['chi2']:.4f}; final chi2_min {s['chi2_min']:.6f}; offset = {self.offset:.6f}",
                flush=True,
            )

        def set_expectations(self, initial_params, initial_observables, parms=None):
            fitter = _find_fitter()
            pm = getattr(fitter, "param_model", None) if fitter is not None else None
            if pm is None:
                pm = self.param_model_override
            if pm is None:
                raise ValueError(
                    "LatticeCSTerm: no param model (no fitter on the stack, no param_model_override)"
                )
            if self.core is None or self._pm is not pm:
                self._build(pm)
            if parms is not None:
                names = np.asarray(parms).astype(str)
                idx = np.asarray(
                    self.sub_vector(tf.range(len(names), dtype=tf.float64))
                ).astype(int)
                if list(names[idx]) != list(np.asarray(self._sub.params).astype(str)):
                    raise ValueError(
                        "LatticeCSTerm: the SCETlib submodel's parameters are not where the layout puts them"
                    )
            if fitter is None and self.tau_arg is None:
                # without the live fitter.tau the penalty would go in unscaled, and rabbit's exp(2 tau) would make
                # it exp(2 tau) too strong (e^16 at tau = 8): refuse rather than guess
                raise ValueError(
                    "LatticeCSTerm: no rabbit Fitter found on the call stack, so the live fitter.tau is unknown and "
                    "the exp(2 tau) that rabbit multiplies every regularizer penalty by cannot be divided out. Pass "
                    "tau=<the fit's --regularizationStrength> on the -r line (offline / standalone use: tau=0 for "
                    "the unscaled 1/2 (chi2 - offset))."
                )
            tau = fitter.tau if fitter is not None else None
            if tau is not None:
                if (
                    self.tau_arg is not None
                    and abs(float(tau.numpy()) - self.tau_arg) > 1e-12
                ):
                    raise ValueError(
                        f"LatticeCSTerm: tau={self.tau_arg} on the -r line but fitter.tau = {float(tau.numpy())}"
                    )
                self._tau = tau
                src = f"fitter.tau (live variable, now {float(tau.numpy()):g})"
            else:
                self._tau = None
                src = f"tau={self.tau_arg} from the -r line"
            print(
                f"[LatticeCSTerm] armed on {type(self._pm).__name__}"
                + (
                    ""
                    if self._k is None
                    else f" (SCETlib submodel #{self._k}, composite permutation)"
                )
                + f": get_x()[:{self._npm}] -> scetlib_full_vector_tf; exp(2 tau) "
                f"compensation from {src}",
                flush=True,
            )

        def sub_vector(self, params):
            """The SCETlib submodel's own fit vector, from get_x() (plain model: its prefix)."""
            if self._k is None:
                return params[: self._npm]
            return submodel_vector(self._pm, self._k, params[: self._npm])

        def p_full_tf(self, params):
            return self._sub.scetlib_full_vector_tf(self.sub_vector(params))

        def p_kernel_tf(self, params):
            """The vector the lattice kernel is evaluated at: p_full (live), or the frozen reference with the CS NP
            lambdas from p_full (frozen)."""
            p = self.p_full_tf(params)
            if self.pert == "frozen":
                p = self._tf_pfrz + self._tf_mlive * p
            return p

        def chi2_tf(self, params):
            z = 0.5 * self._gz(self.p_kernel_tf(params))
            r = z - self.tf_y
            return tf.tensordot(r, tf.linalg.matvec(self.tf_M, r), 1)

        def compute_nll_penalty(self, params, observables):
            half = 0.5 * (
                self.chi2_tf(params) - tf.constant(self.offset, dtype=self.dtype)
            )
            if self._tau is not None:
                return half * tf.exp(-2.0 * tf.cast(self._tau, self.dtype))
            if self.tau_arg is not None:
                return half * tf.constant(np.exp(-2.0 * self.tau_arg), dtype=self.dtype)
            return half

    return LatticeCSTerm


def __getattr__(name):
    if name == "LatticeCSTermMapping":
        cls = _make_mapping_class()
        globals()[name] = cls
        return cls
    if name == "LatticeCSTerm":
        cls = _make_regularizer_class()
        globals()[name] = cls
        return cls
    raise AttributeError(name)
