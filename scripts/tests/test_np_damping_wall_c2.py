#!/usr/bin/env python3
"""Checks of the C^2 ramp (``smooth=c2``, the default since 2026-10-08) of
wremnants/postprocessing/scetlib_ad/np_damping_wall.py.

No cache, no card: a stand-in ``indata`` carrying the card-A correction runcard
(tanh_2 / tanh_2, the y25 cache's anchors) and a gen |Y| axis to 2.5.

    python scripts/tests/test_np_damping_wall_c2.py

  1. c2_ramp: value, first and second derivative continuous at 0 and delta
     (analytic one-sided limits AND one-sided finite differences); P'' == 2 for
     x >= delta; P == x^2 - delta x + delta^2/3 there; P == 0 for x <= 0.
  2. TF autodiff of c2_ramp (nested GradientTape) == the analytic P', P'' on
     both sides of both knots.
  3. The per-condition raw ramp width is delta / scale with the 261006-diagnosis
     scales at b_max = 12.6 GeV^-1 (L2: 3.15e-6 GeV^2, lambda2_nu: 6.30e-6
     GeV^2, cubic: 5.95e-8 GeV^6, lambda4_nu: 3.97e-8 GeV^4).
  4. The full NPDampingWall through the -r parse path (smooth=c2): penalty ==
     the numpy reference built from damping_conditions + c2_ramp; TF gradient ==
     central finite differences; TF Hessian-vector product == finite
     differences of the TF gradient. Evaluated at points where faces are
     violated inside and beyond the ramp.
  5. The default is C^2: no smooth= on tanh_2 gives exactly the smooth=c2
     wall, bitwise. smooth=relu2 returns exactly the relu^2 sum, bitwise, and
     is bitwise equal to the pre-2026-10-08 default (no smooth=) of the module
     as of 2f1c3df4, loaded from git.
  6. 1D equilibrium of -g x + k P(x): x* = g/(2k) + delta/2 for g >= k delta,
     sqrt(g delta / k) below.
  7. Refusals: delta=/bmax= with smooth=relu2; a bad smooth= value; delta <= 0;
     an EXPLICIT smooth=c2 on tanh_6 (its interior discriminants have no
     constant scale).
  8. The DEFAULT on tanh_6 (TMD and CS) arms without error: relu^2 on the
     interior discriminants, C^2 on every other condition, penalty == the mixed
     numpy reference, and one INFO line naming the relu^2 conditions.
Exits 1 on any failure."""

import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import types

import numpy as np

FAIL = []


def check(name, ok, msg=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {msg}", flush=True)
    if not ok:
        FAIL.append(name)


def fake_indata(np_model="tanh_2", np_model_nu="tanh_2"):
    npsec = {
        "np_model_nu": np_model_nu,
        "lambda2_nu": "0.15",
        "lambda4_nu": "0.",
        "lambda_inf_nu": "2.",
        "b0_over_bmax_nu": "1.",
        "np_model": np_model,
        "lambda2": "0.4",
        "delta_lambda2": "0.0",
        "lambda4": "0.4",
        "lambda_inf": "1.",
    }
    if np_model == "tanh_6":
        npsec["lambda6"] = "0.1"
    if np_model_nu == "tanh_6":
        npsec["lambda6_nu"] = "0.1"
    axis = types.SimpleNamespace(
        name="absYVGen",
        edges=np.array([0, 0.15, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2, 2.5]),
    )
    return types.SimpleNamespace(
        metadata={
            "scetlib_corr_config": {
                "Z": {"tag": "fake_cardA", "config": {"Nonperturbative": npsec}}
            }
        },
        channel_info={"ch0": {"axes": [axis]}},
        auxiliary=None,
        procs=np.array(["Z"]),
    )


def load_old_module(rev="2f1c3df4"):
    """np_damping_wall.py as of ``rev`` (the last relu^2-default commit).

    Loaded from ``git show`` into a fresh module, so its default can be
    compared bitwise against today's ``smooth=relu2``. None if git fails.
    """
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = "wremnants/postprocessing/scetlib_ad/np_damping_wall.py"
    try:
        src = subprocess.run(
            ["git", "-C", repo, "show", f"{rev}:{path}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except Exception as e:  # noqa: BLE001
        print("   git show failed:", e)
        return None
    spec = importlib.util.spec_from_loader("np_damping_wall_old", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, f"{rev}:{path}", "exec"), mod.__dict__)
    return mod


def analytic(x, d):
    """P, P', P'' of the C^2 ramp, written piecewise.

    Independent of c2_ramp's branch-free form.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.where(
        x <= 0, 0.0, np.where(x < d, x**3 / (3 * d), x * x - d * x + d * d / 3)
    )
    p1 = np.where(x <= 0, 0.0, np.where(x < d, x * x / d, 2 * x - d))
    p2 = np.where(x <= 0, 0.0, np.where(x < d, 2 * x / d, 2.0))
    return p, p1, p2


def main():
    import tensorflow as tf

    from wremnants.postprocessing.scetlib_ad import np_damping_wall as W

    # ---- 1. continuity of the ramp itself
    for d in (1e-3, 3.15e-6, 5.95e-8):
        xs = np.concatenate([np.linspace(-2 * d, 3 * d, 1001), [d * 10, d * 1e3]])
        p_ref, _, _ = analytic(xs, d)
        p = W.c2_ramp(xs, d)
        check(
            f"c2_ramp == piecewise P (d={d:g})",
            np.allclose(p, p_ref, rtol=1e-12, atol=1e-30 * d * d),
        )
        check(f"P == 0 for x <= 0 (d={d:g})", np.all(p[xs <= 0] == 0.0))
        big = xs >= d
        check(
            f"P == x^2 - d x + d^2/3 for x >= d (d={d:g})",
            np.allclose(p[big], xs[big] ** 2 - d * xs[big] + d * d / 3, rtol=1e-12),
        )
        # one-sided limits at the knots, analytic
        for knot in (0.0, d):
            eps = d * 1e-9
            lo = analytic(knot - eps, d)
            hi = analytic(knot + eps, d)
            for k, nm in enumerate(("P", "P'", "P''")):
                scale = (d * d, d, 1.0)[k]
                check(
                    f"{nm} continuous at x={'0' if knot == 0 else 'd'} (d={d:g})",
                    abs(float(lo[k]) - float(hi[k])) < 1e-6 * scale,
                    f"left {float(lo[k]):.6g} right {float(hi[k]):.6g}",
                )
        # one-sided finite differences of c2_ramp's own values (step h = 1e-4 d)
        h = 1e-4 * d
        for knot in (0.0, d):
            f = lambda x: float(W.c2_ramp(np.float64(x), d))  # noqa: E731
            dl = (f(knot) - f(knot - h)) / h
            dr = (f(knot + h) - f(knot)) / h
            d2l = (f(knot) - 2 * f(knot - h) + f(knot - 2 * h)) / h**2
            d2r = (f(knot + 2 * h) - 2 * f(knot + h) + f(knot)) / h**2
            check(
                f"FD slope continuous at {'0' if knot == 0 else 'd'} (d={d:g})",
                abs(dl - dr) < 1e-3 * d,
                f"{dl:.6g} vs {dr:.6g}",
            )
            check(
                f"FD curvature continuous at {'0' if knot == 0 else 'd'} (d={d:g})",
                abs(d2l - d2r) < 2e-3 * 2,
                f"{d2l:.6g} vs {d2r:.6g}",
            )

    # ---- 2. TF autodiff of the ramp
    d = 3.15e-6
    xs = np.array(
        [-d, 1e-3 * d, 0.5 * d, (1 - 1e-6) * d, (1 + 1e-6) * d, 2 * d, 100 * d]
    )
    x = tf.Variable(xs, dtype=tf.float64)
    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
            p = W.c2_ramp(x, d, maximum=tf.maximum, minimum=tf.minimum)
            s = tf.reduce_sum(p)
        g = t1.gradient(s, x)
    hdiag = t2.gradient(g, x)
    p_ref, p1_ref, p2_ref = analytic(xs, d)
    check("TF P == analytic", np.allclose(p.numpy(), p_ref, rtol=1e-12, atol=1e-30))
    check("TF P' == analytic", np.allclose(g.numpy(), p1_ref, rtol=1e-10, atol=1e-20))
    check(
        "TF P'' == analytic",
        np.allclose(hdiag.numpy(), p2_ref, rtol=1e-8, atol=1e-12),
        f"{hdiag.numpy()} vs {p2_ref}",
    )

    # ---- 3. per-condition raw widths
    conds = W.damping_conditions("tanh_2", "tanh_2", 2.5)
    widths = {c.label: 1e-3 / c.scale for c in conds}
    exp = {
        "lambda2 + delta_lambda2*Y^2 >= 0 at |Y|=2.5 (TMD small-b turn-on)": 1e-3
        / (2 * 12.6**2),
        "lambda2_nu >= 0 (CS small-b turn-on)": 1e-3 / 12.6**2,
        "3*lambda_inf^2*lambda4 + L2^3 >= 0 at |Y|=2.5 (TMD large-b)": 1e-3
        / (2 * 12.6**4 / 3),
        "lambda4_nu >= 0 (CS large-b leading)": 1e-3 / 12.6**4,
    }
    for lab, v in exp.items():
        check(
            f"raw width {lab}",
            np.isclose(widths[lab], v, rtol=1e-12),
            f"{widths[lab]:.4g}",
        )
    print(
        "   raw C^2 widths at delta=1e-3, b_max=12.6:",
        {k[:40]: f"{v:.3g}" for k, v in widths.items()},
    )

    # ---- 4./5. full wall through the -r path
    indata = fake_indata()
    names = np.array(
        [
            "alphaS",
            "lambda2",
            "lambda4",
            "delta_lambda2",
            "lambda2_nu",
            "lambda4_nu",
            "pdfEig0",
        ]
    )
    Map = W.NPDampingMapping
    Wall = W.NPDampingWall

    def make(*args):
        m = Map.parse_args(indata, *args)
        w = Wall(m, tf.float64)
        w.set_expectations(None, None, parms=names)
        return m, w

    # theta points: physical = anchor + width*theta. lambda2 = 0.4 + 0.5 t,
    # delta_lambda2 = 0.5 t, lambda2_nu = 0.15 + 0.1 t, lambda4_nu = 0.5 t.
    # Violations chosen across and beyond the ramps.
    def theta_for(l2, dl2, l4, l2nu, l4nu):
        return np.array(
            [
                0.3,
                (l2 - 0.4) / 0.5,
                (l4 - 0.4) / 0.5,
                dl2 / 0.5,
                (l2nu - 0.15) / 0.1,
                l4nu / 0.5,
                -0.2,
            ]
        )

    pts = {
        # L2(2.5) = 0.026 + 6.25 dl2: -1.5e-6 (inside its 3.15e-6 ramp) or
        # -2e-5 (beyond it)
        "L2(2.5) in ramp": theta_for(0.026, (-0.026 - 1.5e-6) / 6.25, 0.05, 0.1, 1e-3),
        "L2(2.5) beyond ramp + l4nu in ramp": theta_for(
            0.026, (-0.026 - 2e-5) / 6.25, 0.05, 0.1, -2e-8
        ),
        "several faces beyond": theta_for(-1e-4, -0.01, -0.3, -1e-4, -1e-6),
        "interior (no penalty)": theta_for(0.3, 0.01, 0.2, 0.2, 0.1),
    }

    def numpy_penalty(theta, smooth, delta=1e-3):
        v = {
            "lambda2": 0.4 + 0.5 * theta[1],
            "lambda4": 0.4 + 0.5 * theta[2],
            "delta_lambda2": 0.5 * theta[3],
            "lambda2_nu": 0.15 + 0.1 * theta[4],
            "lambda4_nu": 0.5 * theta[5],
            "lambda_inf": 1.0,
            "lambda_inf_nu": 2.0,
        }
        tot = 0.0
        for c in W.damping_conditions("tanh_2", "tanh_2", 2.5):
            if c.names in (("lambda_inf",), ("lambda_inf_nu",)):
                continue  # held -> dropped by the wall
            x = c.bound - c.value(v, W.numpy_relu2)
            tot += float(
                W.numpy_relu2(x) if smooth == "relu2" else W.c2_ramp(x, delta / c.scale)
            )
        return tot

    check(
        "test points violate where intended",
        numpy_penalty(pts["L2(2.5) in ramp"], "relu2") > 0
        and numpy_penalty(pts["L2(2.5) beyond ramp + l4nu in ramp"], "relu2") > 0
        and numpy_penalty(pts["interior (no penalty)"], "relu2") == 0,
    )
    _, wc2 = make("margin=0", "smooth=c2")
    _, wc2b = make("margin=0", "smooth=c2", "delta=1e-2", "bmax=10")
    _, wdef = make("margin=0")
    _, wrel = make("margin=0", "smooth=relu2")
    check(
        "default mapping smooth == c2 (not explicit); relu2 explicit",
        wdef.smooth == "c2"
        and not wdef.smooth_explicit
        and wrel.smooth == "relu2"
        and wrel.smooth_explicit
        and wc2.smooth_explicit,
    )
    check(
        "default tanh_2: C^2 on every active condition",
        all(fn is not None for fn in wdef._outer) and len(wdef._outer) > 0,
    )

    old_mod = load_old_module()
    check("old (2f1c3df4) module loaded from git", old_mod is not None)
    if old_mod is not None:
        wold = old_mod.NPDampingWall(
            old_mod.NPDampingMapping.parse_args(indata, "margin=0"), tf.float64
        )
        wold.set_expectations(None, None, parms=names)
        check("old default was relu2", wold.smooth == "relu2")
    check(
        "c2 mapping parsed",
        wc2.smooth == "c2" and wc2.delta == 1e-3 and wc2.bmax == 12.6,
    )
    check("delta=/bmax= parsed", wc2b.delta == 1e-2 and wc2b.bmax == 10.0)

    def tf_pen(w, theta):
        return float(
            w.compute_nll_penalty(tf.constant(theta, tf.float64), None).numpy()
        )

    def tf_grad_hvp(w, theta, vec):
        x = tf.Variable(theta, dtype=tf.float64)
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                p = w.compute_nll_penalty(x, None)
            g = t1.gradient(p, x)
            gv = tf.reduce_sum(g * tf.constant(vec, tf.float64))
        hv = t2.gradient(gv, x)
        return g.numpy(), hv.numpy()

    rng = np.random.default_rng(1)
    for lab, th in pts.items():
        # relu2 default: bitwise vs the explicit relu^2 graph used before this change
        x = tf.constant(th, tf.float64)
        vals = wdef._physical(x)
        old = float(
            tf.add_n([c.penalty(vals, wdef._relu2_tf) for c in wdef._active]).numpy()
        )
        check(
            f"[{lab}] smooth=relu2 == relu2 sum, bitwise",
            tf_pen(wrel, th) == old,
        )
        if old_mod is not None:
            check(
                f"[{lab}] smooth=relu2 == old (2f1c3df4) default, bitwise",
                tf_pen(wrel, th) == tf_pen(wold, th),
                f"{tf_pen(wrel, th)!r} vs {tf_pen(wold, th)!r}",
            )
        check(
            f"[{lab}] default == explicit smooth=c2, bitwise",
            tf_pen(wdef, th) == tf_pen(wc2, th),
        )
        check(
            f"[{lab}] relu2 == numpy reference",
            np.isclose(
                tf_pen(wrel, th), numpy_penalty(th, "relu2"), rtol=1e-12, atol=1e-300
            ),
        )
        pc2 = tf_pen(wc2, th)
        check(
            f"[{lab}] c2 penalty == numpy reference",
            np.isclose(pc2, numpy_penalty(th, "c2"), rtol=1e-11, atol=1e-300),
            f"{pc2:.6g}",
        )
        vec = rng.normal(size=th.size)
        g, hv = tf_grad_hvp(wc2, th, vec)
        # central FD of the penalty along each coordinate; the step is far
        # below the smallest active ramp
        h = 1e-11
        fd = np.array(
            [
                (tf_pen(wc2, th + h * e) - tf_pen(wc2, th - h * e)) / (2 * h)
                for e in np.eye(th.size)
            ]
        )
        ok = np.abs(g - fd).max() <= 1e-5 * np.abs(g).max() + 1e-14
        check(
            f"[{lab}] c2 gradient == central FD",
            ok,
            f"max|g-fd| {np.abs(g - fd).max():.3g}, max|g| {np.abs(g).max():.3g}",
        )
        gp, _ = tf_grad_hvp(wc2, th + h * vec, vec)
        gm, _ = tf_grad_hvp(wc2, th - h * vec, vec)
        hfd = (gp - gm) / (2 * h)
        ok = np.abs(hv - hfd).max() <= 1e-4 * np.abs(hv).max() + 1e-12
        check(
            f"[{lab}] c2 HVP == FD of gradient",
            ok,
            f"max|hv-fd| {np.abs(hv - hfd).max():.3g}, max|hv| {np.abs(hv).max():.3g}",
        )

    # ---- 6. 1D equilibrium of -g x + k P(x)
    from scipy.optimize import minimize_scalar

    k = np.exp(16.0)
    d = 3.15e-6
    for g in (16.3, 100.0, 1.0):
        res = minimize_scalar(
            lambda x: -g * x + k * float(W.c2_ramp(x, d)),
            bounds=(0, 1e-3),
            method="bounded",
            options={"xatol": 1e-16},
        )
        xs = g / (2 * k) + d / 2 if g >= k * d else np.sqrt(g * d / k)
        check(
            f"equilibrium g={g:g} (k d = {k * d:.3g})",
            np.isclose(res.x, xs, rtol=1e-4),
            f"{res.x:.5g} vs {xs:.5g}",
        )

    # ---- 7. refusals
    def raises(fn, exc):
        try:
            fn()
        except exc:
            return True
        except Exception as e:  # noqa: BLE001
            print("   unexpected", type(e).__name__, e)
            return False
        return False

    check(
        "delta= with smooth=relu2 refused",
        raises(
            lambda: Map.parse_args(indata, "smooth=relu2", "delta=1e-3"), ValueError
        ),
    )
    check(
        "bmax= with smooth=relu2 refused",
        raises(lambda: Map.parse_args(indata, "smooth=relu2", "bmax=12"), ValueError),
    )
    m = Map.parse_args(indata, "delta=2e-3", "bmax=11")
    check(
        "delta=/bmax= accepted with the default (c2)",
        m.smooth == "c2" and m.delta == 2e-3 and m.bmax == 11.0,
    )
    check(
        "smooth=cubic refused",
        raises(lambda: Map.parse_args(indata, "smooth=cubic"), ValueError),
    )
    check(
        "delta=0 refused",
        raises(lambda: Map.parse_args(indata, "smooth=c2", "delta=0"), ValueError),
    )
    ind6 = fake_indata("tanh_6", "tanh_2")
    names6 = np.array(list(names) + ["lambda6"])

    def t6():
        w = Wall(Map.parse_args(ind6, "smooth=c2"), tf.float64)
        w.set_expectations(None, None, parms=names6)

    check("explicit smooth=c2 on tanh_6 (TMD) refused", raises(t6, NotImplementedError))

    # ---- 8. the DEFAULT on tanh_6: relu^2 on the discriminants, C^2 elsewhere
    names66 = np.array(list(names) + ["lambda6", "lambda6_nu"])
    for forms in (("tanh_6", "tanh_2"), ("tanh_2", "tanh_6"), ("tanh_6", "tanh_6")):
        ind = fake_indata(*forms)
        parms = names6 if forms == ("tanh_6", "tanh_2") else names66
        tag = f"np_model={forms[0]}, np_model_nu={forms[1]}"

        def arm(*args, ind=ind, parms=parms):
            w = Wall(Map.parse_args(ind, *args), tf.float64)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                w.set_expectations(None, None, parms=parms)
            return w, buf.getvalue()

        def t6c2(arm=arm):
            arm("smooth=c2")

        check(f"[{tag}] explicit smooth=c2 refused", raises(t6c2, NotImplementedError))
        try:
            w6, out = arm("margin=0")
        except Exception as e:  # noqa: BLE001
            check(f"[{tag}] default arms", False, f"{type(e).__name__}: {e}")
            continue
        check(f"[{tag}] default arms", w6.smooth == "c2" and not w6.smooth_explicit)
        disc = [c for c in w6._active if c.scale is None]
        n_disc = (forms[0] == "tanh_6") * 2 + (forms[1] == "tanh_6")
        check(
            f"[{tag}] relu^2 on exactly the {n_disc} discriminant(s), C^2 elsewhere",
            len(disc) == n_disc
            and all(
                (fn is None) == (c.scale is None)
                for c, fn in zip(w6._active, w6._outer)
            ),
        )
        info = [ln for ln in out.splitlines() if "INFO" in ln]
        check(
            f"[{tag}] one INFO line naming the relu^2 conditions",
            len(info) == 1
            and all(c.label in info[0] for c in disc)
            and "(default)" in info[0],
            info[0] if info else "(none)",
        )
        print("   " + info[0] if info else "   (no INFO line)")
        # penalty == mixed numpy reference (relu^2 on scale None, C^2 else)
        wr6, _ = arm("margin=0", "smooth=relu2")
        idx = {n: i for i, n in enumerate(parms)}
        if forms[1] == "tanh_6":
            # ONLY the CS interior discriminant violated: lambda4_nu = -0.1,
            # lambda2_nu = 0.15, lambda6_nu = 0.01 -> 4*0.15*0.01 - 0.01 = -4e-3.
            # The default must charge exactly relu^2 of it.
            th = np.zeros(parms.size)
            th[idx["lambda4_nu"]] = -0.1 / 0.5
            th[idx["lambda6_nu"]] = (0.01 - 0.1) / 0.1
            got = tf_pen(w6, th)
            check(
                f"[{tag}] CS discriminant alone violated: default == relu^2 of it",
                np.isclose(got, 4e-3**2, rtol=1e-9) and got == tf_pen(wr6, th),
                f"{got!r} vs {4e-3 ** 2!r}",
            )
        rng6 = np.random.default_rng(6)
        for trial in range(4):
            th = rng6.normal(scale=1.5, size=parms.size)
            v = {
                n: float(W.physical_from_theta(w6.inputs["specs"][n], th[idx[n]]))
                for n in w6._idx
            }
            v.update(w6._held)
            ref = 0.0
            for c in w6._active:
                x = c.bound - c.value(v, W.numpy_relu2)
                ref += float(
                    W.numpy_relu2(x)
                    if c.scale is None
                    else W.c2_ramp(x, w6.delta / c.scale)
                )
            got = tf_pen(w6, th)
            check(
                f"[{tag}] trial {trial}: default penalty == mixed numpy reference",
                np.isclose(got, ref, rtol=1e-11, atol=1e-300) and np.isfinite(got),
                f"{got:.6g} vs {ref:.6g}",
            )
            got_r = tf_pen(wr6, th)
            ref_r = sum(
                float(W.numpy_relu2(c.bound - c.value(v, W.numpy_relu2)))
                for c in wr6._active
            )
            check(
                f"[{tag}] trial {trial}: smooth=relu2 == relu^2 numpy reference",
                np.isclose(got_r, ref_r, rtol=1e-11, atol=1e-300),
            )

    print(f"\n{len(FAIL)} failure(s)" + (f": {FAIL}" if FAIL else ""))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
