#!/usr/bin/env python3
"""Checks of wremnants/postprocessing/scetlib_ad/lattice_cs_term.py (the lattice CS-kernel term whose kernel SCETlib
computes at every step). Needs a SCETlib build with DrellYan.gamma_nu_points (scetlib-cms branch gamma-nu-points);
skips (exit 0) otherwise. No big cache: the calculation is configured live from a runcard, and a small cache can be
given to exercise the bitwise snapshot check against real rules.

    python scripts/tests/test_lattice_cs_term.py [--conf <runcard>] [--cache <small cache.npz>]

  1. the data file is the ASWZ per-ensemble layout of its .json provenance (21 points, 6/7/8 per ensemble at
     a = 0.15/0.12/0.09 fm, b_T = k a, mu = 2 GeV), with a symmetric, positive-definite covariance that is block-diagonal
     across ensembles;
  2. the stat-only lattice-only fit on SCETlib's kernel converges (gradient ~ 0) and the J-mapped systematics
     leave its minimum unchanged (Woodbury);
  3. the term's chi2 == the explicit k1 fit of the same residuals (k1 floated by least squares);
  4. TF gradient of the penalty == central finite differences; the parameter the kernel does not depend on has an
     exactly zero gradient;
  5. exp(2 tau) compensation (tau= on the -r line);
  6. a rabbit CompositeParamModel (as the saturated test builds it), in both submodel orders: the term hands SCETlib
     exactly the vector the composite's own compute() hands the SCETlib submodel, and its chi2 equals the plain
     model's at the same physical point, bitwise; two SCETlib submodels are refused;
  7. the bare default term == the explicit LATFROZ_V3 options (pert=frozen alphas_frozen=0.1168 syst=Jnf
     nfmatch=4.18 nfscheme=full offset=min), bitwise: n_f shift, covariance, M, offset, reference, refits, and the
     penalty's value, gradient and Hessian at a displaced point; its load-time numbers == those the LATFROZ_V3 fit
     logged (WRemnantsHelpers studies/lattice-cs-kernel/261008-latfroz-nf-variants/logs/LATFROZ_V3.log);
     checks 2-6 run on the old defaults given explicitly (pert=live syst=Jnf+Jbt nfmatch=1 nfscheme=evolve);
  8. pert=frozen: the kernel sees alpha_s = alphas_frozen and TNPs = 0 whatever the fit vector says (chi2 == the numpy
     chi2 at the frozen reference with the lambdas of p_full); gradient AND Hessian with respect to alpha_s and the TNP
     exactly zero; lambda gradient == FD; the load-time reference is the frozen one;
  9. n_f-scheme alternatives (nfscheme / nfswitch / nfmatch): the shift == the explicit gamma_nu_points composition,
     bitwise; the evolve shifts are flat in b_T; the old-table conventions (n_f = 4 evolution 1 -> 2 GeV with alpha_s^(4)
     from m_b; whole kernel n_f = 4 from m_b) agree with the exact-RGE, 3-loop-decoupled numpy tables of the removed
     table-based term (lattice_aswz_inputs.npz, read from git history at 3ef3efb9; skipped without git) within 6 %
     (analytic vs exact RGE, identification vs decoupling);
 10. the mis-scale guard: with no fitter on the stack and no tau= the term refuses to arm (the penalty would go in
     unscaled and rabbit's exp(2 tau) would make it e^(2 tau) too strong); tau=0 gives the bare 1/2 (chi2 - offset).
 11. the data location: the ASWZ data are private and not in the repository; the default is
     <dirname(getDataPath())>/lattice/ASWZ2024/lattice_aswz_data.npz (MIT: /scratch/submit/cms/wmass/lattice/ASWZ2024/)
     and the term reads it from there; with WREMNANTS_DATA_PATH pointing at an empty directory, or a missing data=, the
     term refuses with a FileNotFoundError naming the path, WREMNANTS_DATA_PATH and data=.
Exits 1 on any failure."""

import argparse
import io
import os
import socket
import subprocess
import sys
import tempfile
import types

import numpy as np

CONF = (
    "/ceph/submit/data/group/cms/store/user/lavezzo/alphaS/scetlib_ad_caches/"
    "pdf62_y35_260921_y25/cache.conf"
)
FAIL = []
# the pre-2026-10-09 defaults (LATB8 era), given explicitly where the checks exercise the live path
OLD_DEFAULTS = dict(pert="live", syst="Jnf+Jbt", nfmatch=1.0, nfscheme="evolve")
# the LATFROZ_V3 -r options (261008-latfroz-nf-variants/cmds/LATFROZ_V3.cmd), the defaults since 2026-10-09
V3 = dict(
    syst="Jnf",
    offset="min",
    pert="frozen",
    alphas_frozen=0.1168,
    nfmatch=4.18,
    nfscheme="full",
)
# load-time numbers the LATFROZ_V3 fit printed (reference alpha_s 0.1168, TNPs 0)
V3_LOG = dict(
    dlam=[-0.016468738386292647, 0.0007696907729429249],
    chi2min_alt=6.892003701755524,
    shift_range=(-0.02754, 0.00253),
    stat_lam=(0.18795, -0.006088),
    chi2_min=6.682179,
)
# the removed table-based term's inputs (exact-RGE numpy tables), kept in git history only
OLD_TABLES = (
    "3ef3efb9",
    "wremnants/postprocessing/scetlib_ad/data/lattice_aswz_inputs.npz",
)


def check(name, ok, msg=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {msg}", flush=True)
    if not ok:
        FAIL.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", default=CONF)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()
    try:
        import scetlib_tf

        scetlib_tf.ScetlibGammaNuTF  # noqa: B018
    except (ImportError, AttributeError) as e:
        print(f"[SKIP] no ScetlibGammaNuTF in this SCETlib build ({e})")
        return 0
    import tensorflow as tf

    from wremnants.postprocessing.scetlib_ad import lattice_cs_term as LT
    from wremnants.postprocessing.scetlib_ad import xsec_backend as xb

    conf = (
        args.conf
        if args.cache is None
        else args.cache.replace("cache.npz", "cache.conf")
    )
    _, sigma = xb.configure(conf, threads=4)
    sing, nons = sigma.sub_pieces()
    if args.cache:
        n_eig = sum(
            1
            for n in xb.ScetlibADXsec.cache_param_names(args.cache)
            if n.startswith("pdf_eig")
        )
        sing.set_pdf_eig_params(n_eig)
        nons.set_pdf_eig_params(n_eig)
        fn = scetlib_tf.ScetlibCachedXsecTF.load(args.cache, sing, nons)
    else:
        fn = types.SimpleNamespace(_sing=sing)
    names = list(sing.gradient_param_names())
    anchor = np.array(sing.gradient_central(), float)

    fit = [
        ("alphaS", "alphas", 0.118, 0.002),
        ("lambda2_nu", "np_gnu_lambda2", 0.15, 0.1),
        ("lambda4_nu", "np_gnu_lambda4", 0.0, 0.5),
        ("resumTNP_gamma_nu", "tnp_gamma_nu", 0.0, 1.0),
        (
            "lambda2",
            "np_eff_lambda2",
            float(anchor[names.index("np_eff_lambda2")]),
            0.5,
        ),
    ]
    idx = np.array([names.index(f[1]) for f in fit])
    c0 = np.array([f[2] for f in fit])
    c1 = np.array([f[3] for f in fit])
    held = anchor.copy()
    held[idx] = 0.0
    S = np.zeros((len(names), len(fit)))
    S[idx, np.arange(len(fit))] = 1.0

    class PM:
        params = np.array([f[0].encode() for f in fit])
        nparams = len(fit)
        npoi, npou = 1, len(fit) - 1
        xparamdefault = tf.zeros([len(fit)], dtype=tf.float64)
        allowNegativeParam = True
        is_linear = False
        seen = None

        def compute(self, param, full=False):
            PM.seen = param
            return tf.ones([], dtype=tf.float64)

        scetlib_names = names
        core = types.SimpleNamespace(tf_fn=fn)
        _p_base_anchor = anchor.copy()

        @staticmethod
        def scetlib_full_vector_tf(param):
            p = tf.constant(c0) + tf.constant(c1) * tf.cast(
                param[: len(fit)], tf.float64
            )
            return tf.constant(held) + tf.linalg.matvec(tf.constant(S), p)

    def term(**kw):
        m = LT.LatticeCSTermMapping.parse_args(
            types.SimpleNamespace(channel_info={}, procs=[]),
            *[f"{k}={v}" for k, v in kw.items()],
        )
        t = LT.LatticeCSTerm(m, tf.float64)
        t.param_model_override = PM()
        t.set_expectations(None, None, parms=PM.params)
        return t

    rules = "require" if args.cache else "any"
    T = term(tau=8.0, rules=rules, **OLD_DEFAULTS)
    c = T.core

    d = np.load(c.data_path)
    ens = np.asarray(d["ens"])
    k = np.concatenate([np.arange(1, n + 1) for n in (6, 7, 8)])
    blk = ens[:, None] == ens[None, :]
    cs = c.cov_stat
    check(
        "1. data file: ASWZ layout (21 points, 6/7/8, a, b_T = k a, mu), covariance",
        len(c.y) == 21
        and list(np.bincount(ens)) == [6, 7, 8]
        and np.array_equal(np.unique(c.a_fm[ens == 0]), [0.15])
        and np.array_equal(np.unique(c.a_fm[ens == 1]), [0.12])
        and np.array_equal(np.unique(c.a_fm[ens == 2]), [0.09])
        and np.allclose(c.b_fm, k * c.a_fm, rtol=0, atol=1e-12)
        and c.mu == 2.0
        and abs(c.fm_to_gevinv * 0.1973269804 - 1) < 1e-9
        and np.array_equal(cs, cs.T)
        and bool(np.all(cs[~blk] == 0.0))
        and bool(np.all(np.linalg.eigvalsh(cs) > 0)),
    )
    fnom = c.fit_nominal
    check(
        "2. stat-only lattice fit converged",
        bool(np.max(np.abs(fnom["grad"])) < 1e-7),
        f"|grad| {np.max(np.abs(fnom['grad'])):.1e}, chi2 {fnom['chi2']:.4f}, lambda {fnom['lam']}",
    )
    dmin = float(np.max(np.abs(c.fit_final["lam"] - fnom["lam"])))
    check("2. J-mapped systematics keep the minimum", dmin < 1e-7, f"{dmin:.1e}")

    th = np.array([0.5, -0.8, 0.012, 0.6, 0.3])
    p = held + S @ (c0 + c1 * th)
    r0 = c.zeta(p) - c.y
    W = np.linalg.inv(c.cov)
    k1 = -(c.v @ W @ r0) / (c.v @ W @ c.v)
    r = r0 + k1 * c.v
    chi2_fit = float(r @ W @ r)
    chi2_term = float(T.chi2_tf(tf.constant(th)))
    check(
        "3. chi2 == explicit k1 fit",
        abs(chi2_term - chi2_fit) < 1e-10 * max(1, chi2_fit),
        f"{chi2_term - chi2_fit:+.1e}",
    )

    x = tf.Variable(th)
    with tf.GradientTape() as tape:
        pen = T.compute_nll_penalty(x, None)
    g = tape.gradient(pen, x).numpy()
    steps = np.array([1e-3, 1e-3, 1e-5, 1e-2, 1e-2])
    fd = np.zeros(len(th))

    def cd(i, h):
        e = np.eye(len(th))[i] * h
        up = float(T.chi2_tf(tf.constant(th + e)))
        dn = float(T.chi2_tf(tf.constant(th - e)))
        return (up - dn) / (2 * h) * 0.5 * np.exp(-16.0)

    for i, h in enumerate(steps):
        fd[i] = (4 * cd(i, h / 2) - cd(i, h)) / 3  # Richardson
    rel = float(np.max(np.abs(g[:4] - fd[:4]) / np.max(np.abs(fd[:4]))))
    check("4. TF gradient == FD", rel < 1e-8, f"rel {rel:.1e}")
    check("4. no kernel dependence -> exactly zero gradient", g[4] == 0.0)
    half = 0.5 * (chi2_term - T.offset)
    check(
        "5. penalty == 1/2 (chi2 - offset) exp(-2 tau)",
        abs(float(pen) - half * np.exp(-16.0)) <= 1e-14 * abs(half * np.exp(-16.0)),
    )
    from rabbit.param_models.param_model import CompositeParamModel

    class Toy:
        npoi, npou, nparams = 2, 1, 3
        params = np.array([b"sat0", b"sat1", b"toynui"])
        xparamdefault = tf.zeros([3], dtype=tf.float64)
        allowNegativeParam = True
        is_linear = True

        def compute(self, param, full=False):
            return tf.ones([], dtype=tf.float64)

    chi2_plain = float(T.chi2_tf(tf.constant(th)))
    for order in ("scetlib first", "scetlib last"):
        subs = [PM(), Toy()] if order == "scetlib first" else [Toy(), PM()]
        comp = CompositeParamModel(subs)
        cnames = np.asarray(comp.params).astype(str)
        parms = np.concatenate([cnames, ["theta0", "theta1"]])
        xc = np.zeros(len(parms))
        for j, f_ in enumerate(fit):
            xc[list(cnames).index(f_[0])] = th[j]
        xc[list(cnames).index("sat0")] = (
            0.37  # values the SCETlib vector must NOT pick up
        )
        xc[list(cnames).index("toynui")] = -1.3
        m = LT.LatticeCSTermMapping.parse_args(
            types.SimpleNamespace(channel_info={}, procs=[]),
            *[f"{k_}={v_}" for k_, v_ in OLD_DEFAULTS.items()],
            "tau=8.0",
            f"rules={rules}",
        )
        Tc = LT.LatticeCSTerm(m, tf.float64)
        Tc.param_model_override = comp
        Tc.set_expectations(None, None, parms=parms)
        comp.compute(tf.constant(xc[: comp.nparams]))
        mine = Tc.sub_vector(tf.constant(xc)).numpy()
        check(
            f"6. composite ({order}): vector == what compute() hands the submodel",
            np.array_equal(mine, PM.seen.numpy()) and np.array_equal(mine, th),
        )
        chi2_c = float(Tc.chi2_tf(tf.constant(xc)))
        check(
            f"6. composite ({order}): chi2 == plain model's, bitwise",
            chi2_c == chi2_plain,
            f"{chi2_c - chi2_plain:+.1e}",
        )
        check(
            f"6. composite ({order}): load-time numbers identical",
            Tc.core.chi2_min == T.core.chi2_min,
        )
    try:
        Tb = LT.LatticeCSTerm(m, tf.float64)
        Tb.param_model_override = CompositeParamModel([PM(), PM()])
        Tb.set_expectations(None, None, parms=None)
        check("6. two SCETlib submodels refused", False)
    except ValueError:
        check("6. two SCETlib submodels refused", True)

    # ---- 7. the bare default == LATFROZ_V3, bitwise
    TD = term(tau=8.0, rules=rules)
    TV3 = term(tau=8.0, rules=rules, **V3)

    def vgh(t):
        xv = tf.Variable(th)
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                v = t.compute_nll_penalty(xv, None)
            gv = t1.gradient(v, xv)
        return float(v), gv.numpy(), t2.jacobian(gv, xv).numpy()

    vD, gD, HD = vgh(TD)
    vV, gV, HV = vgh(TV3)
    cD, cV = TD.core, TV3.core
    check(
        "7. default options resolve to V3",
        (TD.pert, TD.syst, TD.alphas_frozen, TD.nfmatch, TD.nfscheme, TD.offset_arg)
        == ("frozen", "Jnf", 0.1168, 4.18, "full", "min")
        and TD.nfswitch is None
        and cD.nf_switch == 4.18,
    )
    check(
        "7. default == V3: n_f shift, cov, M, offset, reference, refits, bitwise",
        np.array_equal(cD.nf_shift, cV.nf_shift)
        and np.array_equal(cD.cov, cV.cov)
        and np.array_equal(cD.M, cV.M)
        and TD.offset == TV3.offset
        and np.array_equal(cD.p_ref, cV.p_ref)
        and np.array_equal(cD.fit_nominal["lam"], cV.fit_nominal["lam"])
        and np.array_equal(cD.fit_final["lam"], cV.fit_final["lam"])
        and cD.syst_info == cV.syst_info,
    )
    check(
        "7. default == V3: penalty value, gradient, Hessian at a point, bitwise",
        vD == vV and np.array_equal(gD, gV) and np.array_equal(HD, HV),
        f"value {vD:.6e}",
    )
    sD = cD.summary()
    dl = np.array(sD["syst_info"]["Jnf"]["dlam"])
    rows = [
        ("Jnf dlam", float(np.max(np.abs(dl - V3_LOG["dlam"]))), 1e-12),
        (
            "Jnf chi2min",
            abs(sD["syst_info"]["Jnf"]["chi2min"] - V3_LOG["chi2min_alt"]),
            1e-10,
        ),
        (
            "shift range",
            float(
                np.max(np.abs(np.array(sD["nf_shift_range"]) - V3_LOG["shift_range"]))
            ),
            5e-6,
        ),
        (
            "stat-only lambda",
            float(
                np.max(
                    np.abs(np.array(sD["nominal_stat_fit"]["lam"]) - V3_LOG["stat_lam"])
                )
            ),
            5e-6,
        ),
        ("chi2_min", abs(sD["chi2_min"] - V3_LOG["chi2_min"]), 5e-7),
    ]
    check(
        "7. default load-time numbers == the LATFROZ_V3 fit's log (to its printed digits)",
        all(d_ <= tol for _, d_, tol in rows),
        "; ".join(f"{n_} {d_:.1e}" for n_, d_, _ in rows),
    )

    # ---- 8. pert=frozen
    afz = 0.1168
    TF_ = term(syst="Jnf", tau=8.0, rules=rules, pert="frozen", alphas_frozen=afz)
    cf = TF_.core
    pf_ref = anchor.copy()
    pf_ref[names.index("alphas")] = afz
    for n in ("tnp_gamma_cusp", "tnp_gamma_nu"):
        pf_ref[names.index(n)] = 0.0
    check(
        "8. frozen: load-time reference == anchor with alpha_s 0.1168, CS TNPs 0",
        np.array_equal(cf.p_ref, pf_ref),
    )
    pk = held + S @ (
        c0 + c1 * th
    )  # the live p_full at th (alpha_s, TNP off the reference)
    pexp = pf_ref.copy()
    for n in ("np_gnu_lambda2", "np_gnu_lambda4"):
        pexp[names.index(n)] = pk[names.index(n)]
    chi2_np = cf.chi2_full(pexp)
    chi2_tf_ = float(TF_.chi2_tf(tf.constant(th)))
    check(
        "8. frozen: chi2 == numpy chi2 at the frozen reference + p_full lambdas",
        abs(chi2_tf_ - chi2_np) < 1e-12 * max(1.0, chi2_np),
        f"{chi2_tf_ - chi2_np:+.1e}",
    )
    th2 = th.copy()
    th2[0] += 1.7  # alpha_s
    th2[3] -= 2.1  # TNP_nu
    check(
        "8. frozen: chi2 independent of alpha_s and the TNP, bitwise",
        float(TF_.chi2_tf(tf.constant(th2))) == chi2_tf_,
    )
    xv = tf.Variable(th)
    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
            pf = TF_.compute_nll_penalty(xv, None)
        gf = t1.gradient(pf, xv)
    Hf = t2.jacobian(gf, xv).numpy()
    gf = gf.numpy()
    check(
        "8. frozen: gradient wrt alpha_s, TNP_nu (and TMD lambda2) exactly 0",
        gf[0] == 0.0 and gf[3] == 0.0 and gf[4] == 0.0,
        f"{gf[[0, 3, 4]]}",
    )
    check(
        "8. frozen: Hessian rows/cols of alpha_s, TNP_nu exactly 0",
        bool(np.all(Hf[[0, 3, 4], :] == 0.0) and np.all(Hf[:, [0, 3, 4]] == 0.0)),
    )

    def cdf(i, h):
        e = np.eye(len(th))[i] * h
        up = float(TF_.chi2_tf(tf.constant(th + e)))
        dn = float(TF_.chi2_tf(tf.constant(th - e)))
        return (up - dn) / (2 * h) * 0.5 * np.exp(-16.0)

    fdf = np.array(
        [(4 * cdf(i, h / 2) - cdf(i, h)) / 3 for i, h in ((1, 1e-3), (2, 1e-5))]
    )
    relf = float(np.max(np.abs(gf[[1, 2]] - fdf) / np.max(np.abs(fdf))))
    check("8. frozen: lambda gradient == FD", relf < 1e-8, f"rel {relf:.1e}")
    try:
        term(syst="Jnf", rules=rules, pert="live", alphas_frozen=afz)
        check("8. alphas_frozen without pert=frozen refused", False)
    except ValueError:
        check("8. alphas_frozen without pert=frozen refused", True)

    # ---- 9. n_f-scheme alternatives
    MB = 4.18
    pr = cf.p_ref

    def z(mu, nf=0, mm=0.0):
        return 0.5 * np.asarray(sing.gamma_nu_points(cf.bT, mu, pr, 0, nf, mm)["value"])

    z5 = z(2.0)
    variants = {
        "V1 evolve, switch = match = 1 GeV": (
            dict(nfmatch=1.0, nfscheme="evolve"),
            z(1.0) + z(2.0, 4, 1.0) - z(1.0, 4, 1.0) - z5,
        ),
        "V2 evolve, switch 1 GeV, match m_b": (
            dict(nfmatch=MB, nfswitch=1.0, nfscheme="evolve"),
            z(1.0) + z(2.0, 4, MB) - z(1.0, 4, MB) - z5,
        ),
        "V3 full n_f = 4, match m_b": (
            dict(nfmatch=MB, nfscheme="full"),
            z(2.0, 4, MB) - z5,
        ),
        "V3lit evolve, switch = match = m_b": (
            dict(nfmatch=MB, nfscheme="evolve"),
            z(MB) + z(2.0, 4, MB) - z(MB, 4, MB) - z5,
        ),
    }
    for lab, (kw, ref) in variants.items():
        Tv = term(
            syst="Jnf", rules=rules, tau=0.0, pert="frozen", alphas_frozen=afz, **kw
        )
        check(
            f"9. {lab}: shift == explicit composition, bitwise",
            np.array_equal(Tv.core.nf_shift, ref),
            f"range {ref.min():+.5f}..{ref.max():+.5f}",
        )
        if "evolve" in lab:
            spread = float(np.ptp(ref))
            check(f"9. {lab}: flat in b_T", spread < 1e-12, f"{spread:.1e}")
    # independent: the shipped exact-RGE numpy tables (260923-conventions-map our_cs_kernel), alpha_s 0.118, TNPs 0
    T118 = term(
        syst="Jnf",
        rules=rules,
        tau=0.0,
        pert="frozen",
        alphas_frozen=0.118,
        nfmatch=MB,
        nfswitch=1.0,
        nfscheme="evolve",
    )
    T118f = term(
        syst="Jnf",
        rules=rules,
        tau=0.0,
        pert="frozen",
        alphas_frozen=0.118,
        nfmatch=MB,
        nfscheme="full",
    )
    try:
        blob = subprocess.run(
            [
                "git",
                "-C",
                os.path.dirname(os.path.abspath(LT.__file__)),
                "show",
                ":".join(OLD_TABLES),
            ],
            capture_output=True,
            check=True,
        ).stdout
        old = np.load(io.BytesIO(blob))
    except (OSError, subprocess.CalledProcessError, ValueError) as e:
        old = None
        print(f"[SKIP] 9. exact-RGE table comparison: no {':'.join(OLD_TABLES)} ({e})")
    if old is not None:
        tab_v2 = old["pert_nf5_mu1match"] - old["pert"]
        tab_v3 = old["pert_nf4"] - old["pert"]
        r2 = float(np.max(np.abs(T118.core.nf_shift / tab_v2 - 1)))
        r3 = float(
            np.max(np.abs(T118f.core.nf_shift - tab_v3)) / np.max(np.abs(tab_v3))
        )
        check(
            "9. V2 vs exact-RGE decoupled numpy table (alpha_s 0.118)",
            r2 < 0.06,
            f"max rel {r2:.3f}",
        )
        check(
            "9. V3 vs exact-RGE decoupled numpy table (alpha_s 0.118)",
            r3 < 0.06,
            f"max rel (to max|shift|) {r3:.3f}",
        )

    try:
        term(syst="Jnf", rules=rules, nfmatch=MB, nfswitch=1.0)
        check("9. nfswitch with the default nfscheme=full refused", False)
    except ValueError:
        check("9. nfswitch with the default nfscheme=full refused", True)

    # ---- 10. mis-scale guard: no fitter and no tau= -> refuse to arm
    mg = LT.LatticeCSTermMapping.parse_args(
        types.SimpleNamespace(channel_info={}, procs=[]), f"rules={rules}"
    )
    Tg = LT.LatticeCSTerm(mg, tf.float64)
    Tg.param_model_override = PM()
    try:
        Tg.set_expectations(None, None, parms=PM.params)
        check("10. no fitter, no tau= -> refused", False)
    except ValueError as e:
        check(
            "10. no fitter, no tau= -> refused (message names tau=)",
            "tau=" in str(e) and "regularizationStrength" in str(e),
        )
    T0 = term(tau=0.0, rules=rules)
    v0 = float(T0.compute_nll_penalty(tf.constant(th), None))
    check(
        "10. tau=0 -> penalty == 1/2 (chi2 - offset), unscaled",
        v0 == 0.5 * (float(T0.chi2_tf(tf.constant(th))) - T0.offset),
    )

    # ---- 11. data location: the default (sibling of the NanoAOD dir) and the missing-file error
    dflt = LT.default_data_path()
    check(
        "11. default data path = <dirname(getDataPath())>/lattice/ASWZ2024/lattice_aswz_data.npz, read by the term",
        c.data_path == dflt
        and dflt.endswith(os.path.join("lattice", "ASWZ2024", "lattice_aswz_data.npz"))
        and os.path.isfile(dflt)
        and (
            "WREMNANTS_DATA_PATH" in os.environ
            or dflt
            == "/scratch/submit/cms/wmass/lattice/ASWZ2024/lattice_aswz_data.npz"
            or not socket.getfqdn().endswith(".mit.edu")
        ),
        dflt,
    )
    saved = os.environ.get("WREMNANTS_DATA_PATH")
    with tempfile.TemporaryDirectory() as tmp:
        nano = os.path.join(tmp, "NanoAOD")
        os.mkdir(nano)
        os.environ["WREMNANTS_DATA_PATH"] = nano
        try:
            want = os.path.join(tmp, "lattice", "ASWZ2024", "lattice_aswz_data.npz")
            check(
                "11. WREMNANTS_DATA_PATH moves the default",
                LT.default_data_path() == want,
            )
            for label, kw in (
                ("default", {}),
                ("data=", {"data": os.path.join(tmp, "x.npz")}),
            ):
                try:
                    term(tau=0.0, rules=rules, **kw)
                    check(f"11. missing data file ({label}) refused", False)
                except FileNotFoundError as e:
                    msg = str(e)
                    check(
                        f"11. missing data file ({label}) refused, message names path, WREMNANTS_DATA_PATH, data=",
                        (want if not kw else kw["data"]) in msg
                        and "WREMNANTS_DATA_PATH" in msg
                        and "data=" in msg,
                    )
        finally:
            if saved is None:
                os.environ.pop("WREMNANTS_DATA_PATH", None)
            else:
                os.environ["WREMNANTS_DATA_PATH"] = saved

    print("ALL PASS" if not FAIL else f"FAILED: {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
