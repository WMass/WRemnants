"""Export a ``train_shift_smear_reweight.py --arch polyhead`` checkpoint
to AOTI (``.pt2``) and ONNX (``.onnx``) plus a small JSON sidecar
that the C++ benchmarks consume to do polynomial evaluation outside
the model graph.

Inference signature

    forward(y_raw, c_raw) -> joint_coefs

i.e., trunk-only — the polynomial evaluation at any (u, σ_vec) is
done outside the package by the deployment-side ``evaluate_joint``
(C++ or python). The trunk forward dominates per-event NN cost and
amortizes across many (u, σ_vec) variations per event, which is the
realistic deployment pattern for the shift+smear reweight.

The sidecar JSON ``<output>.indices.json`` carries the structural
priors needed to evaluate the polynomial:

    {
      "n_features": 3,
      "n_basis":    <int>,
      "max_deg_u":  <int>,
      "max_deg_sigma": <int>,
      "basis":      "monomial" | "chebyshev",
      "basis_scale_u":  <float>,
      "basis_scale_sigma": <float>,
      "alpha_degs": [[k_u_0, ..., k_u_{d-1}] for each basis index],
      "beta_degs":  [[k_s_0, ..., k_s_{d-1}] for each basis index],
      "target_std": [...]   # for converting raw u/σ -> standardized
    }

Run::

    python shift_smear_reweight_export.py \
        --checkpoint /path/to/checkpoint.pt \
        --output     /path/to/polyhead.pt2 \
        --onnx-output /path/to/polyhead.onnx
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch._inductor import config as ind_cfg

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from shift_smear_reweight_diagnostics import (  # noqa: E402
    load_model_from_checkpoint,
)
from train_muon_response_flow import (  # noqa: E402
    _multiindex_to_axis_degrees,
    evaluate_joint,
)


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

class ReweightPolyheadInference(nn.Module):
    """``forward(y_raw, c_raw) → joint_coefs`` with preproc baked in.

    Pure tensor ops; torch.export and ONNX both trace it directly.
    Dynamic batch is supported via ``torch.export.Dim``. The polynomial
    evaluation at ``(u, σ_vec)`` is done outside the package.
    """

    def __init__(
        self,
        polyhead: nn.Module,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        cond_mean: torch.Tensor,
        cond_std: torch.Tensor,
    ):
        super().__init__()
        self.polyhead = polyhead
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)
        self.register_buffer("cond_mean", cond_mean)
        self.register_buffer("cond_std", cond_std)

    def forward(
        self, y_raw: torch.Tensor, c_raw: torch.Tensor,
    ) -> torch.Tensor:
        y_std = (y_raw - self.target_mean) / self.target_std
        c_std = (c_raw - self.cond_mean) / self.cond_std
        return self.polyhead(y_std, c_std)


class CombinedInference(nn.Module):
    """``forward(y_raw, c_raw, u, sigma) → d`` with the polynomial
    evaluation folded into the graph.

    Trunk amortizes: one trunk forward per ``(y, c)`` event, the
    coefs are broadcast across the ``N_var`` perturbations and the
    polynomial sum is evaluated in-graph via :func:`evaluate_joint`.

    Shapes (all float32):
        y_raw  [B, F]
        c_raw  [B, n_cond]
        u      [B, N_var, F]
        sigma  [B, N_var, F]
        d      [B, N_var]

    Both batch dimensions can be marked dynamic at export time so a
    single deployed graph handles any (B, N_var) at runtime.
    """

    def __init__(
        self,
        polyhead: nn.Module,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        cond_mean: torch.Tensor,
        cond_std: torch.Tensor,
    ):
        super().__init__()
        self.polyhead = polyhead
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)
        self.register_buffer("cond_mean", cond_mean)
        self.register_buffer("cond_std", cond_std)

    def forward(
        self,
        y_raw: torch.Tensor,
        c_raw: torch.Tensor,
        u: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        y_std = (y_raw - self.target_mean) / self.target_std
        c_std = (c_raw - self.cond_mean) / self.cond_std
        coefs = self.polyhead(y_std, c_std)              # [B, n_basis]
        # Broadcast coefs across N_var without materializing a copy.
        coefs_b = coefs.unsqueeze(1).expand(-1, u.shape[1], -1)
        return evaluate_joint(
            coefs_b, u, sigma,
            self.polyhead.joint_indices,
            basis=str(getattr(self.polyhead, "basis", "monomial")),
            scale_u=float(getattr(self.polyhead, "basis_scale_u", 1.0)),
            scale_sigma=float(
                getattr(self.polyhead, "basis_scale_sigma", 1.0)
            ),
            basis_aux=self.polyhead.basis_aux,
        )


# ---------------------------------------------------------------------------
# Helpers (shared with flow_polyhead_export)
# ---------------------------------------------------------------------------

def _decompose_linear(ep):
    aten = torch.ops.aten

    def _linear_decomp(input, weight, bias=None):
        if bias is None:
            return aten.mm.default(input, weight.permute(1, 0))
        return aten.addmm.default(bias, input, weight.permute(1, 0))

    return ep.run_decompositions({aten.linear.default: _linear_decomp})


def _to_tuple(out):
    if isinstance(out, (tuple, list)):
        return tuple(out)
    return (out,)


def _validate_aoti(wrapper, runner, n_features, n_cond, B, n_events, tol):
    rng = np.random.default_rng(0)
    n_events = max(B, ((n_events + B - 1) // B) * B)
    y = torch.from_numpy(
        rng.standard_normal((n_events, n_features)).astype(np.float32)
    )
    c = torch.from_numpy(
        rng.standard_normal((n_events, n_cond)).astype(np.float32)
    )
    eager_chunks, aoti_chunks = [], []
    with torch.no_grad():
        for s in range(0, n_events, B):
            yi, ci = y[s:s + B], c[s:s + B]
            eager_chunks.append(_to_tuple(wrapper(yi, ci))[0])
            aoti_chunks.append(_to_tuple(runner(yi, ci))[0])
    et, at = torch.cat(eager_chunks), torch.cat(aoti_chunks)
    max_abs = (et - at).abs().max().item()
    denom = max(et.abs().max().item(), 1e-30)
    rel = max_abs / denom
    status = "OK" if rel < tol else "MISMATCH"
    print(
        f"[validate-aoti] joint_coefs: max |eager-aoti|={max_abs:.3e} "
        f"rel={rel:.2e}  {status}"
    )
    return rel < tol


def _validate_onnx_combined(
    combined, onnx_path, n_features, n_cond, n_events, tol,
    n_var: int = 8,
):
    """Compare eager vs ORT for the (y, c, u, σ) → d combined graph."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[validate-onnx-combined] onnxruntime not installed — skipping.")
        return True
    sess = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"],
    )
    rng = np.random.default_rng(2)
    y_v = rng.standard_normal((n_events, n_features)).astype(np.float32)
    c_v = rng.standard_normal((n_events, n_cond)).astype(np.float32)
    u_v = rng.uniform(-1.0, 1.0,
                      (n_events, n_var, n_features)).astype(np.float32)
    s_v = rng.uniform(0.0, 1.0,
                      (n_events, n_var, n_features)).astype(np.float32)
    with torch.no_grad():
        ref = combined(
            torch.from_numpy(y_v), torch.from_numpy(c_v),
            torch.from_numpy(u_v), torch.from_numpy(s_v),
        ).numpy()
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]
    feeds = dict(zip(in_names, [y_v, c_v, u_v, s_v]))
    ort_out = sess.run([out_names[0]], feeds)[0]
    max_abs = float(np.abs(ref - ort_out).max())
    denom = max(float(np.abs(ref).max()), 1e-30)
    rel = max_abs / denom
    status = "OK" if rel < tol else "MISMATCH"
    print(
        f"[validate-onnx-combined] d over N={n_events} N_var={n_var}: "
        f"max |eager-ort|={max_abs:.3e} rel={rel:.2e}  {status}"
    )
    return rel < tol


def _validate_onnx(
    wrapper, onnx_path, n_features, n_cond, n_events, tol,
):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[validate-onnx] onnxruntime not installed — skipping.")
        return True
    sess = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"],
    )
    rng = np.random.default_rng(1)
    y_v = rng.standard_normal((n_events, n_features)).astype(np.float32)
    c_v = rng.standard_normal((n_events, n_cond)).astype(np.float32)
    with torch.no_grad():
        ref = wrapper(
            torch.from_numpy(y_v), torch.from_numpy(c_v),
        ).numpy()
    out_names = [o.name for o in sess.get_outputs()]
    in_names = [i.name for i in sess.get_inputs()]
    feeds = {in_names[0]: y_v, in_names[1]: c_v}
    ort_out = sess.run([out_names[0]], feeds)[0]
    max_abs = float(np.abs(ref - ort_out).max())
    denom = max(float(np.abs(ref).max()), 1e-30)
    rel = max_abs / denom
    status = "OK" if rel < tol else "MISMATCH"
    print(
        f"[validate-onnx] joint_coefs over N={n_events}: "
        f"max |eager-ort|={max_abs:.3e} rel={rel:.2e}  {status}"
    )
    return rel < tol


# ---------------------------------------------------------------------------
# Sidecar — joint_indices in flat per-axis-degree form
# ---------------------------------------------------------------------------

def _emit_indices_sidecar(polyhead, stats, out_path):
    """Write a JSON file describing the polynomial basis so the C++
    side can evaluate it without re-implementing the index logic.

    For each basis index i, ``alpha_degs[i, j]`` is the degree of the
    j-th u-axis monomial factor (0 if axis j absent) and similarly
    for ``beta_degs``. This is the same layout used by
    ``train_muon_response_flow._build_basis_aux``.
    """
    n_features = int(polyhead.n_features)
    aa, bb = [], []
    for (a, b) in polyhead._joint_indices:
        aa.append(_multiindex_to_axis_degrees(a, n_features))
        bb.append(_multiindex_to_axis_degrees(b, n_features))
    payload = {
        "n_features":     n_features,
        "n_basis":        int(polyhead.n_basis),
        "n_pure_u":       int(polyhead._n_pure_u),
        "n_pure_sigma":   int(polyhead._n_pure_s),
        "n_cross":        int(polyhead._n_cross),
        "max_deg_u":      int(polyhead.max_deg_u),
        "max_deg_sigma":  int(polyhead.max_deg_sigma),
        "max_cross_deg":  int(polyhead.max_cross_deg),
        "basis":          str(getattr(polyhead, "basis", "monomial")),
        "basis_scale_u":  float(
            getattr(polyhead, "basis_scale_u", 1.0)
        ),
        "basis_scale_sigma": float(
            getattr(polyhead, "basis_scale_sigma", 1.0)
        ),
        "alpha_degs":     [list(map(int, row)) for row in aa],
        "beta_degs":      [list(map(int, row)) for row in bb],
        "target_std":     [float(x) for x in stats.target_std],
        "target_mean":    [float(x) for x in stats.target_mean],
        "cond_std":       [float(x) for x in stats.cond_std],
        "cond_mean":      [float(x) for x in stats.cond_mean],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_path} (n_basis={payload['n_basis']})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to a shift_smear_reweight checkpoint with "
        "model_config.arch == 'polyhead'.",
    )
    p.add_argument(
        "--output", default=None,
        help="AOTI .pt2 output path. Pass empty string or skip to "
        "disable AOTI export. Default: alongside the checkpoint as "
        "shift_smear_reweight_polyhead.pt2.",
    )
    p.add_argument(
        "--onnx-output", default=None,
        help="ONNX .onnx output path for the trunk-only graph "
        "(forward(y, c) -> coefs). Default: alongside the "
        "checkpoint as shift_smear_reweight_polyhead.onnx.",
    )
    p.add_argument(
        "--combined-onnx-output", default=None,
        help="ONNX .onnx output path for the combined graph "
        "(forward(y, c, u, sigma) -> d) with the polynomial "
        "evaluation folded in. Default: alongside the checkpoint as "
        "shift_smear_reweight_polyhead_combined.onnx. Pass empty "
        "string to disable.",
    )
    p.add_argument(
        "--combined-output", default=None,
        help="AOTI .pt2 output path for the combined graph "
        "(forward(y, c, u, sigma) -> d). Compiled with dynamic batch "
        "and dynamic N_var dimensions so a single package handles "
        "any (B, N_var) at runtime. Default: alongside the "
        "checkpoint as shift_smear_reweight_polyhead_combined.pt2. "
        "Pass empty string to disable.",
    )
    p.add_argument(
        "--indices-output", default=None,
        help="JSON sidecar path. Default: derived from --output by "
        "replacing the trailing extension with '.indices.json'.",
    )
    p.add_argument(
        "--batch", type=int, default=1,
        help="Static batch size baked into AOTI when "
        "--no-dynamic-batch is in effect.",
    )
    p.add_argument(
        "--dynamic-batch",
        action=argparse.BooleanOptionalAction, default=False,
        help="Compile a dynamic-batch AOTI package. Off by default "
        "(static batch=1 is the lowest-latency narf path).",
    )
    p.add_argument(
        "--max-autotune",
        action=argparse.BooleanOptionalAction, default=True,
    )
    p.add_argument(
        "--freeze",
        action=argparse.BooleanOptionalAction, default=True,
    )
    p.add_argument(
        "--prepack",
        action=argparse.BooleanOptionalAction, default=True,
    )
    p.add_argument(
        "--link-libtorch",
        action=argparse.BooleanOptionalAction, default=True,
        help="If False, AOTI compiles the wrapper.so without "
        "libtorch.so / libtorch_cpu.so as DT_NEEDED, leaving only "
        "the C-ABI aoti_torch_* shim symbols (~20) plus libgomp/"
        "glibc as runtime deps. The shim must be resolved at app "
        "link time — either from libtorch_cpu (already needed for "
        "AOTIModelPackageLoader at the harness level), or from a "
        "minimal hand-rolled implementation. Tradeoff: with "
        "link_libtorch=False inductor's vec-ISA dry-compile fails "
        "(its probe code calls .exp() → Sleef_*, which isn't "
        "linkable without libtorch_cpu) so pick_vec_isa() falls "
        "back to INVALID and the autotuner is restricted to the "
        "scalar reference micro-gemm. On this trunk that's roughly "
        "5x slower than the AVX2 FP32Vec path. Use only if "
        "deployment-size constraints outweigh the per-event cost.",
    )
    p.add_argument(
        "--package-cpp-only",
        action=argparse.BooleanOptionalAction, default=False,
        help="Emit the AOTI .pt2 as source code (.cpp + .h + "
        ".weights.o + CMakeLists.txt) instead of a precompiled "
        ".so. Combined with --no-link-libtorch this gives full "
        "control over the link line — e.g. statically link a "
        "custom shim, set custom CFLAGS, target an embedded "
        "toolchain. Off by default.",
    )
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--validate-n", type=int, default=128)
    p.add_argument("--validate-tol", type=float, default=1e-4)
    p.add_argument(
        "--bench", action="store_true",
        help="After export, time AOTI runner forward at the static "
        "batch (single core).",
    )
    p.add_argument("--bench-duration", type=float, default=2.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _default_paths(args):
    base = os.path.dirname(os.path.abspath(args.checkpoint))
    if args.output is None:
        args.output = os.path.join(
            base, "shift_smear_reweight_polyhead.pt2",
        )
    if args.onnx_output is None:
        args.onnx_output = os.path.join(
            base, "shift_smear_reweight_polyhead.onnx",
        )
    if args.combined_onnx_output is None:
        args.combined_onnx_output = os.path.join(
            base, "shift_smear_reweight_polyhead_combined.onnx",
        )
    if args.combined_output is None:
        args.combined_output = os.path.join(
            base, "shift_smear_reweight_polyhead_combined.pt2",
        )
    if args.indices_output is None:
        args.indices_output = os.path.join(
            base, "shift_smear_reweight_polyhead.indices.json",
        )


def _build_wrapper(checkpoint_path):
    """Load checkpoint via the diagnostic loader (handles both old and
    current model_config schemas) and wrap it for export.
    """
    from train_muon_response_flow import PreprocStats
    model, arch, stats, _ = load_model_from_checkpoint(
        checkpoint_path, device="cpu",
    )
    if arch != "polyhead":
        raise SystemExit(
            f"--checkpoint arch={arch!r}; only 'polyhead' is supported "
            "by this exporter. For 'mlp' / 'mlp-factored' arches use "
            "the combined-MLP exporter "
            "(scripts/corrections/muon_calibration/"
            "train_shift_reweight_mlp_export.py for legacy shift-only "
            "MLP; or the bench_mlp_combined_export.py pattern adapted "
            "to whichever head class the checkpoint carries — "
            "ReweightMLP_B for 'mlp' or ReweightMLPFactored for "
            "'mlp-factored')."
        )
    if stats is None:
        # Fall back to preproc.json next to the checkpoint.
        ck_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        pp = os.path.join(ck_dir, "preproc.json")
        if not os.path.exists(pp):
            raise SystemExit(
                f"checkpoint has no PreprocStats and no preproc.json "
                f"at {pp}"
            )
        with open(pp) as f:
            stats = PreprocStats(**json.load(f))
    target_mean = torch.tensor(
        list(stats.target_mean), dtype=torch.float32,
    )
    target_std = torch.tensor(
        list(stats.target_std), dtype=torch.float32,
    )
    cond_mean = torch.tensor(
        list(stats.cond_mean), dtype=torch.float32,
    )
    cond_std = torch.tensor(
        list(stats.cond_std), dtype=torch.float32,
    )
    wrapper = ReweightPolyheadInference(
        polyhead=model.eval(),
        target_mean=target_mean,
        target_std=target_std,
        cond_mean=cond_mean,
        cond_std=cond_std,
    ).eval()
    n_features = int(target_mean.shape[0])
    n_cond = int(cond_mean.shape[0])
    return wrapper, model, stats, n_features, n_cond


def main():
    args = parse_args()
    _default_paths(args)

    if args.freeze:
        ind_cfg.freezing = True
    if args.prepack:
        ind_cfg.cpp.weight_prepack = True
    if args.max_autotune:
        ind_cfg.max_autotune = True
        ind_cfg.max_autotune_gemm = True
    # AOTI runtime-dependency knobs. ``link_libtorch=False`` strips
    # libtorch / libtorch_cpu from the wrapper.so's DT_NEEDED, leaving
    # only the ~20 aoti_torch_* C-shim symbols plus glibc/libstdc++/
    # libgomp. ``package_cpp_only=True`` emits source + CMakeLists in
    # the .pt2 instead of a compiled .so so the caller can choose how
    # to satisfy the shim (e.g. static-link a custom implementation).
    if not args.link_libtorch:
        ind_cfg.aot_inductor.link_libtorch = False
    if args.package_cpp_only:
        ind_cfg.aot_inductor.package_cpp_only = True
        # Pair with embed_kernel_binary so the .weights.o is bundled.
        ind_cfg.aot_inductor.embed_kernel_binary = True
    print(
        f"inductor: freezing={ind_cfg.freezing} "
        f"weight_prepack={ind_cfg.cpp.weight_prepack} "
        f"max_autotune={getattr(ind_cfg, 'max_autotune', False)} "
        f"link_libtorch={ind_cfg.aot_inductor.link_libtorch} "
        f"package_cpp_only={getattr(ind_cfg.aot_inductor, 'package_cpp_only', False)}"
    )

    print(f"loading checkpoint {args.checkpoint}")
    wrapper, polyhead, stats, n_features, n_cond = _build_wrapper(
        args.checkpoint,
    )
    print(
        f"polyhead: n_basis={polyhead.n_basis} "
        f"(pu={polyhead._n_pure_u} ps={polyhead._n_pure_s} "
        f"cr={polyhead._n_cross}) "
        f"max_deg_u={polyhead.max_deg_u} "
        f"max_deg_sigma={polyhead.max_deg_sigma} "
        f"max_cross_deg={polyhead.max_cross_deg} "
        f"basis={getattr(polyhead, 'basis', 'monomial')}"
    )

    # Always emit the sidecar (cheap; needed for both ORT and AOTI).
    _emit_indices_sidecar(polyhead, stats, args.indices_output)

    B = args.batch
    B_trace = max(B, 2) if args.dynamic_batch else B
    y = torch.randn(B_trace, n_features)
    c = torch.randn(B_trace, n_cond)
    with torch.no_grad():
        ref = wrapper(y, c)
    print(f"eager: joint_coefs={tuple(ref.shape)}")

    # ---------- AOTI ----------
    if args.output:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output)) or ".",
            exist_ok=True,
        )
        if args.dynamic_batch:
            batch_dim = torch.export.Dim("batch", min=1, max=2**20)
            dynamic_shapes = {
                "y_raw": {0: batch_dim},
                "c_raw": {0: batch_dim},
            }
            print(
                f"\ntorch.export.export (dynamic batch, B_example="
                f"{B_trace}) ..."
            )
        else:
            dynamic_shapes = None
            print(f"\ntorch.export.export (static B={B}) ...")
        ep = torch.export.export(
            wrapper, (y, c), dynamic_shapes=dynamic_shapes,
        )
        ep = _decompose_linear(ep)
        print(f"aoti_compile_and_package -> {args.output}")
        with torch.no_grad():
            torch._inductor.aoti_compile_and_package(
                ep, package_path=args.output,
            )
        size_mb = os.path.getsize(args.output) / 1e6
        print(f" OK ({size_mb:.2f} MB)")

        if args.package_cpp_only:
            # cpp-only .pt2 has no precompiled wrapper.so — caller
            # builds it via the bundled CMakeLists. Skip in-process
            # validation; round-trip must be done with the built
            # artifact externally.
            print(
                "package_cpp_only=True: skipping in-process AOTI "
                "round-trip (no .so in package; build via CMake)."
            )
        elif not args.link_libtorch:
            # ``aoti_load_package`` dlopens the wrapper.so with
            # RTLD_LOCAL by default, so the aoti_torch_* shim
            # symbols (provided by libtorch_cpu) aren't visible
            # unless we promote libtorch_cpu to RTLD_GLOBAL first.
            # Do so explicitly so users can still validate the
            # exported package round-trips correctly.
            import ctypes
            import torch as _torch
            _libcpu = os.path.join(
                os.path.dirname(_torch.__file__), "lib",
                "libtorch_cpu.so",
            )
            ctypes.CDLL(_libcpu, mode=ctypes.RTLD_GLOBAL)
            runner = torch._inductor.aoti_load_package(args.output)
            out = _to_tuple(runner(y, c))[0]
            diff = (out - ref).abs().max().item()
            print(f"loaded; max |aoti - eager| joint_coefs = {diff:.2e}")
            if args.validate_n > 0:
                _validate_aoti(
                    wrapper, runner, n_features, n_cond, B,
                    args.validate_n, args.validate_tol,
                )
        else:
            runner = torch._inductor.aoti_load_package(args.output)
            out = _to_tuple(runner(y, c))[0]
            diff = (out - ref).abs().max().item()
            print(f"loaded; max |aoti - eager| joint_coefs = {diff:.2e}")

            if args.validate_n > 0:
                _validate_aoti(
                    wrapper, runner, n_features, n_cond, B,
                    args.validate_n, args.validate_tol,
                )

        if args.bench:
            try:
                os.sched_setaffinity(0, {0})
            except Exception:
                pass
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            for _ in range(20):
                runner(y, c)
            n = 0
            t0 = time.perf_counter()
            deadline = t0 + args.bench_duration
            while True:
                runner(y, c)
                n += 1
                if n >= 5 and time.perf_counter() >= deadline:
                    break
            elapsed = time.perf_counter() - t0
            ms_b = 1000.0 * elapsed / n
            us_e = 1e6 * elapsed / n / B
            ev_s = n * B / elapsed
            print(
                f"\naoti bench (B={B}, 1 core): "
                f"{ms_b:.3f} ms/call  {us_e:.3f} us/event  "
                f"{ev_s:.1f} ev/s  ({n} iters)"
            )

    # ---------- ONNX ----------
    if args.onnx_output:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.onnx_output)) or ".",
            exist_ok=True,
        )
        dynamic_axes = None
        # Export ONNX with dynamic batch by default — trivial for ORT.
        dynamic_axes = {
            "y_raw":      {0: "batch"},
            "c_raw":      {0: "batch"},
            "joint_coefs": {0: "batch"},
        }
        print(
            f"\ntorch.onnx.export (dynamic batch, B_example={B_trace}, "
            f"opset={args.opset}) -> {args.onnx_output}"
        )
        torch.onnx.export(
            wrapper, (y, c), args.onnx_output,
            input_names=["y_raw", "c_raw"],
            output_names=["joint_coefs"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
            export_params=True,
        )
        size_mb = os.path.getsize(args.onnx_output) / 1e6
        print(f" OK ({size_mb:.2f} MB)")

        if args.validate_n > 0:
            _validate_onnx(
                wrapper, args.onnx_output, n_features, n_cond,
                args.validate_n, args.validate_tol,
            )

    # ---------- Combined AOTI / ONNX share the same wrapper ----------
    combined = None
    if args.combined_output or args.combined_onnx_output:
        combined = CombinedInference(
            polyhead=polyhead,
            target_mean=wrapper.target_mean,
            target_std=wrapper.target_std,
            cond_mean=wrapper.cond_mean,
            cond_std=wrapper.cond_std,
        ).eval()
        # Eager smoke check at a non-trivial (B, N_var). Both dims
        # must be > 1 to avoid torch.export specializing them.
        B_ex, N_ex = max(B_trace, 2), 4
        y_e = torch.randn(B_ex, n_features)
        c_e = torch.randn(B_ex, n_cond)
        u_e = torch.randn(B_ex, N_ex, n_features)
        s_e = torch.randn(B_ex, N_ex, n_features)
        with torch.no_grad():
            d_e = combined(y_e, c_e, u_e, s_e)
        print(
            f"\ncombined eager: d={tuple(d_e.shape)} "
            f"(B={B_ex}, N_var={N_ex})"
        )

    # ---------- Combined AOTI .pt2 ----------
    if args.combined_output:
        os.makedirs(
            os.path.dirname(
                os.path.abspath(args.combined_output)
            ) or ".",
            exist_ok=True,
        )
        batch_dim = torch.export.Dim("batch", min=1, max=2**20)
        nvar_dim  = torch.export.Dim("nvar",  min=1, max=2**20)
        dynamic_shapes_combined = {
            "y_raw":  {0: batch_dim},
            "c_raw":  {0: batch_dim},
            "u":      {0: batch_dim, 1: nvar_dim},
            "sigma":  {0: batch_dim, 1: nvar_dim},
        }
        print(
            f"torch.export.export combined "
            f"(dynamic batch+nvar, B={B_ex}, N_var={N_ex}) ..."
        )
        ep = torch.export.export(
            combined, (y_e, c_e, u_e, s_e),
            dynamic_shapes=dynamic_shapes_combined,
        )
        ep = _decompose_linear(ep)
        print(f"aoti_compile_and_package -> {args.combined_output}")
        with torch.no_grad():
            torch._inductor.aoti_compile_and_package(
                ep, package_path=args.combined_output,
            )
        size_mb = os.path.getsize(args.combined_output) / 1e6
        print(f" OK ({size_mb:.2f} MB)")

        if args.package_cpp_only:
            print(
                "package_cpp_only=True: skipping in-process combined "
                "AOTI round-trip (no .so in package; build via CMake)."
            )
            runner = None
        else:
            if not args.link_libtorch:
                # See trunk export: promote libtorch_cpu to RTLD_GLOBAL
                # so the wrapper.so's aoti_torch_* shim symbols
                # resolve at dlopen time when libtorch isn't in
                # DT_NEEDED.
                import ctypes
                import torch as _torch
                _libcpu = os.path.join(
                    os.path.dirname(_torch.__file__), "lib",
                    "libtorch_cpu.so",
                )
                ctypes.CDLL(_libcpu, mode=ctypes.RTLD_GLOBAL)
            runner = torch._inductor.aoti_load_package(args.combined_output)
            out = _to_tuple(runner(y_e, c_e, u_e, s_e))[0]
            diff = (out - d_e).abs().max().item()
            print(
                f"loaded; max |aoti - eager| d = {diff:.2e}"
            )

        if args.validate_n > 0 and runner is not None:
            # Match the existing combined-ONNX validation: random
            # (y, c, u, σ) at N_var=8.
            rng = np.random.default_rng(3)
            n_use = args.validate_n
            n_var = 8
            y_v = torch.from_numpy(rng.standard_normal(
                (n_use, n_features)).astype(np.float32))
            c_v = torch.from_numpy(rng.standard_normal(
                (n_use, n_cond)).astype(np.float32))
            u_v = torch.from_numpy(rng.uniform(
                -1.0, 1.0, (n_use, n_var, n_features)
            ).astype(np.float32))
            s_v = torch.from_numpy(rng.uniform(
                0.0, 1.0, (n_use, n_var, n_features)
            ).astype(np.float32))
            with torch.no_grad():
                ref = combined(y_v, c_v, u_v, s_v)
                aoti_out = _to_tuple(runner(y_v, c_v, u_v, s_v))[0]
            max_abs = (ref - aoti_out).abs().max().item()
            denom = max(ref.abs().max().item(), 1e-30)
            rel = max_abs / denom
            status = (
                "OK" if rel < args.validate_tol else "MISMATCH"
            )
            print(
                f"[validate-aoti-combined] d over N={n_use} "
                f"N_var={n_var}: max |eager-aoti|={max_abs:.3e} "
                f"rel={rel:.2e}  {status}"
            )

    # ---------- Combined ONNX (trunk + poly) ----------
    if args.combined_onnx_output:
        os.makedirs(
            os.path.dirname(
                os.path.abspath(args.combined_onnx_output)
            ) or ".",
            exist_ok=True,
        )
        dynamic_axes_combined = {
            "y_raw":  {0: "batch"},
            "c_raw":  {0: "batch"},
            "u":      {0: "batch", 1: "nvar"},
            "sigma":  {0: "batch", 1: "nvar"},
            "d":      {0: "batch", 1: "nvar"},
        }
        print(
            f"torch.onnx.export combined (dynamic batch+nvar, "
            f"B={B_ex} N_var={N_ex}, opset={args.opset}) -> "
            f"{args.combined_onnx_output}"
        )
        torch.onnx.export(
            combined, (y_e, c_e, u_e, s_e), args.combined_onnx_output,
            input_names=["y_raw", "c_raw", "u", "sigma"],
            output_names=["d"],
            dynamic_axes=dynamic_axes_combined,
            opset_version=args.opset,
            do_constant_folding=True,
            export_params=True,
        )
        size_mb = os.path.getsize(args.combined_onnx_output) / 1e6
        print(f" OK ({size_mb:.2f} MB)")

        if args.validate_n > 0:
            _validate_onnx_combined(
                combined, args.combined_onnx_output, n_features,
                n_cond, args.validate_n, args.validate_tol,
            )


if __name__ == "__main__":
    main()
