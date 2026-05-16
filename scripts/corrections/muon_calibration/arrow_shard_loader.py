"""Streaming per-batch iterator over Arrow IPC training shards.

Companion to the snapshot-side sharder
(``wremnants/production/arrow_shard_export.py`` /
``flow_training_snapshot.py --shard-only``): that step produces
LZ4-compressed Arrow IPC stream files (one record batch ~= 16k rows)
carrying the per-muon training schema. The trainers used to
``load_ntuples`` the entire dataset into RAM up front; at the scale
of the merged J/psi + W/Z snapshot that's tens of GB of resident
data and minutes of cold-start latency. This module streams instead:
a single pre-pass computes the preprocessing mean/std, then each
training epoch reads record batches on demand from disk.

Lives alongside the training scripts in
``scripts/corrections/muon_calibration/`` so the trainer ecosystem
stays standalone (no ``wremnants`` package import on the read path).
The exposed API mirrors :class:`InMemoryLoader` in
``train_muon_response_flow.py`` so the trainer's training/validation
loops don't have to change. ``compute_stats_streaming`` is the
one-pass helper for ``PreprocStats`` + a global ``weight_mean``.
"""

from __future__ import annotations

import json
import os
from typing import List, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
import torch.utils.data as torch_data


# ---------------------------------------------------------------------------
# Per-muon schema (fp32 columns the snapshot script writes, plus int32
# ``source_id`` which we don't currently feed to the flow but is
# preserved on the shards).
# ---------------------------------------------------------------------------

_RAW_FEATURE_COLUMNS = (
    "eta_reco", "phi_reco", "eta_gen", "phi_gen",
    "kappa_reco", "kappa_gen", "nominal_weight",
)


_TARGET_NAMES = ("r_kappa", "dlambda", "dphi")
_COND_NAMES = (
    "log_pt_gen", "charge", "lambda_gen", "sin_phi_gen", "cos_phi_gen",
)


_WEIGHT_MODES = ("abs", "keep", "drop")


class TimedLoader:
    """Wrap a data loader and emit per-iteration timing for the first
    ``n_print`` batches. Splits each step into:

    * ``loader_wait`` — time the trainer thread blocked on
      ``next(loader_iter)``; this is the worker-queue wait. Large
      values mean the workers can't keep up.
    * ``trainer_step`` — time between successive ``next`` calls, i.e.
      everything the trainer does between batches (H2D copy + model
      forward + backward + optimizer step + any GPU sync). Large
      values mean the consumer is the bottleneck and worker CPU%
      can't be improved by adding more workers.

    Counts are emitted once at the start of each ``iter()`` so they
    appear once per epoch (or once per validation pass).
    """

    def __init__(self, loader, n_print: int = 20, label: str = "loader"):
        self.loader = loader
        self.n_print = int(n_print)
        self.label = str(label)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        import time
        n_print = self.n_print
        label = self.label
        it = iter(self.loader)
        idx = 0
        if n_print > 0:
            print(
                f"[{label}] data-pipeline profile (first {n_print} batches):"
            )
        while True:
            t_req = time.perf_counter()
            try:
                item = next(it)
            except StopIteration:
                return
            t_got = time.perf_counter()
            loader_wait_ms = (t_got - t_req) * 1e3
            # Time the consumer's hold on this batch by sampling
            # ``perf_counter`` right before yielding and right after
            # the generator resumes (= consumer asked for the next
            # batch). The print happens after the consumer is done
            # with this iteration, so output appears one step
            # delayed relative to the trainer's progress bar — fine
            # for a diagnostic.
            t_yield = time.perf_counter()
            yield item
            t_resume = time.perf_counter()
            trainer_step_ms = (t_resume - t_yield) * 1e3
            if idx < n_print:
                print(
                    f"[{label}] iter {idx:>3d}  "
                    f"loader_wait={loader_wait_ms:7.2f}ms  "
                    f"trainer_step={trainer_step_ms:7.2f}ms"
                )
            idx += 1


class StepProfiler:
    """Time labelled checkpoints inside a training step.

    Companion to :class:`TimedLoader`: ``TimedLoader`` splits the
    epoch into (loader_wait, trainer_step); this class splits the
    ``trainer_step`` into its components — H2D copy, forward,
    backward, optimizer, bookkeeping — so you can localise where the
    main-process budget is being spent.

    On CUDA, calls ``torch.cuda.synchronize()`` between marks so
    that async kernel queues are attributed to the right section.
    Sync itself costs ~tens of microseconds and is only inserted
    when profiling, so it doesn't slow steady-state training.
    """

    def __init__(self, enabled: bool, label: str, device: str = "cpu"):
        import time as _time
        self.enabled = bool(enabled)
        self.label = str(label)
        self.use_cuda = (
            self.enabled
            and isinstance(device, str)
            and device.startswith("cuda")
            and torch.cuda.is_available()
        )
        self._time = _time
        self._marks: list = []

    def start(self):
        """Reset and capture the start timestamp. Call once per step
        before any work."""
        if not self.enabled:
            return
        if self.use_cuda:
            torch.cuda.synchronize()
        self._marks = [("__start", self._time.perf_counter())]

    def mark(self, section: str):
        """Record the end of a section. Inserts ``cuda.synchronize``
        so async GPU work issued during the section is included."""
        if not self.enabled:
            return
        if self.use_cuda:
            torch.cuda.synchronize()
        self._marks.append((section, self._time.perf_counter()))

    def report(self, iter_idx: int):
        """Emit one line summarising this step's sections."""
        if not self.enabled or len(self._marks) < 2:
            return
        deltas = []
        prev = self._marks[0][1]
        for name, t in self._marks[1:]:
            deltas.append(f"{name}={(t - prev) * 1e3:6.2f}ms")
            prev = t
        total = (self._marks[-1][1] - self._marks[0][1]) * 1e3
        print(
            f"[{self.label}] iter {iter_idx:>3d}  total={total:6.2f}ms  "
            + "  ".join(deltas)
        )


def dataloader_worker_init(worker_id: int):
    """``DataLoader(worker_init_fn=...)`` hook.

    Pin each DataLoader worker subprocess to a single CPU thread for
    numpy / BLAS / pyarrow / torch ops. The wmassdevrolling container
    inherits ``OMP_NUM_THREADS`` from the entry-point env (often
    ~all cores, see memory note); without this hook every one of the
    N DataLoader workers spins up an N-way thread pool of its own,
    and the resulting N×N oversubscription gives each worker ~1/N
    of a single core — the signature of "every worker at ~25 %
    CPU" we hit at scale. Single-threaded workers + many of them is
    the right shape for this pipeline (the parallelism comes from
    the worker count, not threads per worker).
    """
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    import torch as _torch
    _torch.set_num_threads(1)
    try:
        _torch.set_num_interop_threads(1)
    except RuntimeError:
        # Already set — interop threads can't be changed after the
        # first torch op in the worker.
        pass


def _apply_weight_mode(w, mode):
    """Return ``(w_out, keep_mask)`` after applying the weight policy.

    * ``abs``  (default): take ``|w|`` and drop rows with w == 0 or
      non-finite. Loses the destructive-interference signal of
      MC@NLO-style signed samples but keeps the magnitude information
      and makes the trainer behave as if every row had positive
      weight. Matches the historical J/psi training where weights
      were always positive.
    * ``keep``: pass ``w`` through unchanged (signed). Only drops
      non-finite. Use this for an unbiased weighted-NLL fit where
      negative weights cancel positive ones in expectation.
    * ``drop``: drop rows with w <= 0 or non-finite. Legacy
      behaviour; loses ~5% of W/Z rows in MC@NLO samples.
    """
    finite = np.isfinite(w)
    if mode == "abs":
        keep = finite & (w != 0.0)
        return np.fabs(w), keep
    if mode == "keep":
        return w, finite
    if mode == "drop":
        return w, finite & (w > 0.0)
    raise ValueError(f"weight_mode must be one of {_WEIGHT_MODES}, got {mode!r}")


def resolve_shard_files(
    input_files: Sequence[str], return_counts: bool = False,
):
    """Expand the trainer's ``--input-files`` contract to a flat list
    of ``.arrow`` shard paths. Accepts:

    * a directory containing ``manifest.json`` (auto-expanded);
    * a list of explicit ``.arrow`` shard files;

    Anything else (notably ``.root``) raises.

    When ``return_counts=True`` returns ``(paths, row_counts_or_None)``.
    ``row_counts`` is a per-shard list of row counts (same length as
    ``paths``) read from the manifest's ``shard_row_counts`` block —
    avoids the trainer having to walk every record batch later just
    to estimate ``__len__``. ``None`` if any input came from an
    explicit ``.arrow`` path (no manifest) or the manifest is
    missing the counts.
    """
    paths: List[str] = []
    counts: List[int] = []
    have_counts = True
    for p in input_files:
        if os.path.isdir(p):
            manifest = os.path.join(p, "manifest.json")
            if not os.path.exists(manifest):
                raise FileNotFoundError(
                    f"resolve_shard_files: {p!r} is a directory but "
                    f"contains no manifest.json"
                )
            with open(manifest) as f:
                m = json.load(f)
            count_map = {
                int(entry["shard"]): int(entry["n_rows"])
                for entry in m.get("shard_row_counts", [])
            }
            for i, fn in enumerate(m["shard_files"]):
                paths.append(os.path.join(p, fn))
                if i in count_map:
                    counts.append(count_map[i])
                else:
                    counts.append(0)
                    have_counts = False
        elif p.endswith(".arrow"):
            paths.append(p)
            counts.append(0)
            have_counts = False
        else:
            raise ValueError(
                f"resolve_shard_files: only .arrow shards or a shard "
                f"directory are supported; got {p!r}"
            )
    if return_counts:
        return paths, (counts if have_counts else None)
    return paths


_ARROW_FILE_MAGIC = b"ARROW1"


def _open_ipc(path: str):
    """Open an Arrow IPC shard for reading. Auto-detects file vs
    stream format by sniffing the magic header — the sharder writes
    file-format now (random-access via footer index), but older
    stream-format shards from earlier runs are still supported.
    Returns ``(memory_map_handle, reader)``. The caller owns both;
    closing the reader does not close the underlying mmap.
    """
    src = pa.memory_map(path, "r")
    head = src.read(len(_ARROW_FILE_MAGIC))
    src.seek(0)
    if head == _ARROW_FILE_MAGIC:
        return src, ipc.open_file(src)
    return src, ipc.open_stream(src)


def _iter_record_batches(reader):
    """Uniform iteration interface over file or stream readers.
    ``open_file`` returns a ``RecordBatchFileReader`` with
    ``num_record_batches`` + ``get_batch(i)``; ``open_stream``
    returns a ``RecordBatchStreamReader`` that's itself iterable.
    """
    if isinstance(reader, ipc.RecordBatchFileReader):
        for i in range(reader.num_record_batches):
            yield reader.get_batch(i)
    else:
        yield from reader


def _num_record_batches(reader) -> int:
    if isinstance(reader, ipc.RecordBatchFileReader):
        return reader.num_record_batches
    # Stream readers don't expose a count without walking; fall back.
    return -1


def count_rows(shard_files: Sequence[str]) -> List[int]:
    """Row count per shard. For file-format shards, reads only the
    footer index (cheap). For stream-format shards, walks the whole
    file (cost on the order of one full scan)."""
    counts: List[int] = []
    for p in shard_files:
        src, r = _open_ipc(p)
        try:
            if isinstance(r, ipc.RecordBatchFileReader):
                # Sum num_rows per batch — read_all() would
                # materialize the whole table.
                n = sum(
                    r.get_batch(i).num_rows
                    for i in range(r.num_record_batches)
                )
            else:
                n = r.read_all().num_rows
            counts.append(n)
        finally:
            if hasattr(r, "close"):
                r.close()
            src.close()
    return counts


# ---------------------------------------------------------------------------
# Per-batch numpy compute: same identities as
# ``compute_targets_and_conditioning`` and ``apply_preproc`` in
# ``train_muon_response_flow.py``, fused into one pass for streaming.
# ---------------------------------------------------------------------------


def _per_batch_target_cond(cols: dict):
    """Return (target [N, 3], cond [N, 5], weight [N]) in fp32 from
    the raw fp32 columns of one Arrow record batch.

    Mirrors :func:`compute_targets_and_conditioning` exactly:

    * ``r_kappa = kappa_reco / kappa_gen - 1``
    * ``dlambda = arctan(sinh(eta_reco)) - arctan(sinh(eta_gen))``
    * ``dphi = atan2(sin(phi_r - phi_g), cos(phi_r - phi_g))``
    * conditioning: log_pt_gen (from kappa_gen+eta_gen),
      sign(kappa_gen), lambda_gen, sin/cos(phi_gen).
    """
    eta_r = cols["eta_reco"]
    phi_r = cols["phi_reco"]
    eta_g = cols["eta_gen"]
    phi_g = cols["phi_gen"]
    kappa_r = cols["kappa_reco"]
    kappa_g = cols["kappa_gen"]

    lam_r = np.arctan(np.sinh(eta_r))
    lam_g = np.arctan(np.sinh(eta_g))

    r_kappa = kappa_r / kappa_g - 1.0
    dphi = np.arctan2(
        np.sin(phi_r - phi_g), np.cos(phi_r - phi_g)
    )
    dlambda = lam_r - lam_g

    target = np.stack([r_kappa, dlambda, dphi], axis=1).astype(np.float32)

    log_pt_gen = -np.log(np.fabs(kappa_g) * np.cosh(eta_g))
    cond = np.stack(
        [
            log_pt_gen,
            np.sign(kappa_g),
            lam_g,
            np.sin(phi_g),
            np.cos(phi_g),
        ],
        axis=1,
    ).astype(np.float32)

    w = cols["nominal_weight"].astype(np.float32, copy=False)

    return target, cond, w


def _read_raw_columns(batch: pa.RecordBatch) -> dict:
    return {c: batch.column(c).to_numpy() for c in _RAW_FEATURE_COLUMNS}


# ---------------------------------------------------------------------------
# Single-pass preproc stats + weight mean
# ---------------------------------------------------------------------------


def _stats_chunk(arg):
    """Worker: read a (shard, [batch_start, batch_stop), weight_mode)
    slice and return partial (n_kept, t_sum, t_sq, c_sum, c_sq,
    w_sum, abs_w_sum, n_filt). ``abs_w_sum`` is the sum of ``|w|``
    over the kept rows — used as the normalisation scale by the
    streaming loader regardless of weight_mode, so signed-weight
    training (``mode=keep``) doesn't divide by a near-zero mean(w)
    when positive and negative contributions cancel.
    """
    path, batch_start, batch_stop, weight_mode = arg
    n_target = len(_TARGET_NAMES)
    n_cond = len(_COND_NAMES)
    t_sum = np.zeros(n_target, dtype=np.float64)
    t_sq = np.zeros(n_target, dtype=np.float64)
    c_sum = np.zeros(n_cond, dtype=np.float64)
    c_sq = np.zeros(n_cond, dtype=np.float64)
    w_sum = 0.0
    abs_w_sum = 0.0
    n_kept = 0
    n_filt = 0

    src, reader = _open_ipc(path)
    try:
        if isinstance(reader, ipc.RecordBatchFileReader):
            indices = range(batch_start, min(batch_stop, reader.num_record_batches))
            batches = (reader.get_batch(i) for i in indices)
        else:
            # Stream: walk sequentially, only accumulate in range.
            def _walk():
                for i, b in enumerate(reader):
                    if i < batch_start:
                        continue
                    if i >= batch_stop:
                        return
                    yield b
            batches = _walk()

        for batch in batches:
            cols = _read_raw_columns(batch)
            target, cond, w = _per_batch_target_cond(cols)
            target_finite = (
                np.isfinite(target).all(axis=1)
                & np.isfinite(cond).all(axis=1)
            )
            w_out, w_keep = _apply_weight_mode(w, weight_mode)
            keep = target_finite & w_keep
            if not keep.all():
                target = target[keep]
                cond = cond[keep]
                w_out = w_out[keep]
                n_filt += int(np.size(keep) - keep.sum())
            if target.shape[0] == 0:
                continue
            t_sum += target.sum(axis=0, dtype=np.float64)
            t_sq += np.square(target, dtype=np.float64).sum(axis=0)
            c_sum += cond.sum(axis=0, dtype=np.float64)
            c_sq += np.square(cond, dtype=np.float64).sum(axis=0)
            w_sum += float(w_out.sum(dtype=np.float64))
            abs_w_sum += float(np.fabs(w_out, dtype=np.float64).sum())
            n_kept += target.shape[0]
    finally:
        if hasattr(reader, "close"): reader.close()
        src.close()
    return n_kept, t_sum, t_sq, c_sum, c_sq, w_sum, abs_w_sum, n_filt


def _enumerate_chunks(shard_files, batches_per_chunk: int, weight_mode: str):
    """Build a list of (shard, batch_start, batch_stop, weight_mode)
    work items. File-format shards expose ``num_record_batches``
    (cheap, footer read); stream-format shards have to be walked to
    count, which we do once here. Each shard's batches are split
    into ``ceil(n_batches / batches_per_chunk)`` chunks."""
    out = []
    for path in shard_files:
        src, reader = _open_ipc(path)
        try:
            n_b = _num_record_batches(reader)
            if n_b < 0:
                # Stream — count by walking (cheap-ish; metadata only).
                n_b = sum(1 for _ in reader)
        finally:
            if hasattr(reader, "close"): reader.close()
            src.close()
        if n_b == 0:
            continue
        for start in range(0, n_b, batches_per_chunk):
            stop = min(start + batches_per_chunk, n_b)
            out.append((path, start, stop, weight_mode))
    return out


def _compute_stats_robust(
    shard_files: Sequence[str],
    sample_rows: int,
    weight_mode: str,
    progress: bool,
):
    """Sample-based robust location + scale: ``median`` for the
    location, ``1.4826 * MAD`` for the scale. Reads up to
    ``sample_rows`` filtered rows sequentially across the shard list
    (the sharder has already globally shuffled the rows, so the
    first ``sample_rows`` across shards is a random sample of the
    full dataset). 1 M rows is plenty for stable median + MAD on the
    schema we use.
    """
    from train_muon_response_flow import PreprocStats

    if progress:
        print(
            f"computing robust preproc stats: median + 1.4826·MAD from "
            f"a {sample_rows:,}-row sample, weight_mode={weight_mode!r}"
        )

    t_chunks: List[np.ndarray] = []
    c_chunks: List[np.ndarray] = []
    w_chunks: List[np.ndarray] = []
    n_total = 0
    n_filt = 0
    for path in shard_files:
        if n_total >= sample_rows:
            break
        src, reader = _open_ipc(path)
        try:
            for batch in _iter_record_batches(reader):
                if n_total >= sample_rows:
                    break
                cols = _read_raw_columns(batch)
                target, cond, w = _per_batch_target_cond(cols)
                target_finite = (
                    np.isfinite(target).all(axis=1)
                    & np.isfinite(cond).all(axis=1)
                )
                w_out, w_keep = _apply_weight_mode(w, weight_mode)
                keep = target_finite & w_keep
                if not keep.all():
                    target = target[keep]
                    cond = cond[keep]
                    w_out = w_out[keep]
                    n_filt += int(np.size(keep) - keep.sum())
                if target.shape[0] == 0:
                    continue
                if n_total + target.shape[0] > sample_rows:
                    take = sample_rows - n_total
                    target = target[:take]
                    cond = cond[:take]
                    w_out = w_out[:take]
                t_chunks.append(target)
                c_chunks.append(cond)
                w_chunks.append(w_out)
                n_total += target.shape[0]
        finally:
            if hasattr(reader, "close"):
                reader.close()
            src.close()

    if n_total == 0:
        raise RuntimeError("compute_stats_streaming(robust): zero rows survived filters")

    target_arr = np.concatenate(t_chunks, axis=0)
    cond_arr = np.concatenate(c_chunks, axis=0)
    w_arr = np.concatenate(w_chunks, axis=0)

    # Median + 1.4826·MAD per column. fp64 for the median computation
    # — np.median internally does a partition over fp64 if input is
    # fp64; cast once to avoid repeating it per axis.
    t_arr64 = target_arr.astype(np.float64, copy=False)
    t_median = np.median(t_arr64, axis=0)
    t_mad = np.median(np.fabs(t_arr64 - t_median), axis=0)
    t_std = 1.4826 * t_mad
    t_std = np.where(t_std > 1e-6, t_std, 1.0)

    c_arr64 = cond_arr.astype(np.float64, copy=False)
    c_median = np.median(c_arr64, axis=0)
    c_mad = np.median(np.fabs(c_arr64 - c_median), axis=0)
    c_std = 1.4826 * c_mad
    c_std = np.where(c_std > 1e-6, c_std, 1.0)

    abs_w_mean = float(np.fabs(w_arr.astype(np.float64, copy=False)).mean())

    if progress:
        print(
            f"  sampled {n_total:,} rows"
            + (f" (dropped {n_filt:,} non-finite or filtered)" if n_filt else "")
            + f"; |weight| mean = {abs_w_mean:.4g}"
        )
        print(f"  target median: {t_median.tolist()}")
        print(f"  target 1.4826·MAD: {t_std.tolist()}")
        print(f"  cond median: {c_median.tolist()}")
        print(f"  cond 1.4826·MAD: {c_std.tolist()}")

    return (
        PreprocStats(
            target_names=list(_TARGET_NAMES),
            target_mean=t_median.tolist(),
            target_std=t_std.tolist(),
            cond_names=list(_COND_NAMES),
            cond_mean=c_median.tolist(),
            cond_std=c_std.tolist(),
        ),
        abs_w_mean,
    )


def compute_stats_streaming(
    shard_files: Sequence[str],
    max_rows: int = -1,
    n_workers: int = 0,
    batches_per_chunk: int = 32,
    progress: bool = True,
    weight_mode: str = "abs",
    robust: bool = True,
    robust_sample_rows: int = 20_000_000,
):
    """One pass over the shards; returns ``(PreprocStats, weight_mean)``.

    Accumulates per-column ``sum`` and ``sum_of_squares`` in float64
    so a few billion fp32 rows stay numerically clean.

    Parallelism: when ``n_workers > 1`` (default: ``os.cpu_count()``)
    record-batch chunks are dispatched to a ``ProcessPoolExecutor``.
    Threads were tried first but pyarrow IPC read + ``to_numpy()`` hold
    the GIL through more of the hot path than the docs suggest, so
    threaded scaling tops out around 5-6 cores even on a 192-thread
    box; separate Python interpreters sidestep the GIL entirely.
    ``batches_per_chunk`` controls the granularity (more chunks =
    better load balance, slightly more pickling overhead).

    ``max_rows > 0`` caps the scan to roughly that many rows total
    (across all workers); the precise cap is checked per-chunk so a
    bit of overshoot is possible.
    """
    # ``PreprocStats`` lives in the sibling trainer script.
    from train_muon_response_flow import PreprocStats
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    if weight_mode not in _WEIGHT_MODES:
        raise ValueError(
            f"weight_mode must be one of {_WEIGHT_MODES}, got {weight_mode!r}"
        )
    if robust:
        # Dispatch to the sample-based median + 1.4826·MAD path.
        # Tail mass (e.g. r_kappa charge-mismeasurement peak) doesn't
        # move the median, so the resulting scale reflects the bulk
        # width — what you actually want when the trainer's δ=1
        # perturbation is meant to span ~1 typical-event width.
        return _compute_stats_robust(
            shard_files, int(robust_sample_rows), weight_mode, progress,
        )
    if n_workers <= 0:
        n_workers = max(1, (os.cpu_count() or 1))

    if progress:
        print(
            f"computing preproc stats: {len(shard_files)} shard(s), "
            f"{n_workers} process(es), batches_per_chunk={batches_per_chunk}, "
            f"weight_mode={weight_mode!r}"
            + (f", capped at {max_rows:,} rows" if max_rows > 0 else "")
        )

    tasks = _enumerate_chunks(shard_files, batches_per_chunk, weight_mode)
    if not tasks:
        raise RuntimeError("compute_stats_streaming: no record batches found")

    n_target = len(_TARGET_NAMES)
    n_cond = len(_COND_NAMES)
    t_sum = np.zeros(n_target, dtype=np.float64)
    t_sq = np.zeros(n_target, dtype=np.float64)
    c_sum = np.zeros(n_cond, dtype=np.float64)
    c_sq = np.zeros(n_cond, dtype=np.float64)
    w_sum = 0.0
    abs_w_sum = 0.0
    n_total = 0
    n_filt = 0

    # ``fork`` is cheaper than ``spawn`` and safe here: the trainer
    # entry point doesn't import ROOT (or any other library that
    # dislikes being forked through), so the worker's address space
    # only needs numpy / pyarrow which fork cleanly. Each worker
    # returns a tiny (sums + scalar) payload, so the per-chunk
    # pickle round-trip is negligible.
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        for partial in ex.map(_stats_chunk, tasks):
            (p_n, p_ts, p_tsq, p_cs, p_csq, p_ws, p_abs_ws, p_filt) = partial
            t_sum += p_ts
            t_sq += p_tsq
            c_sum += p_cs
            c_sq += p_csq
            w_sum += p_ws
            abs_w_sum += p_abs_ws
            n_total += p_n
            n_filt += p_filt
            if max_rows > 0 and n_total >= max_rows:
                # Best-effort early stop: cancel remaining chunks by
                # exiting the with-block (in-flight workers finish,
                # we just stop accumulating).
                break

    if n_total == 0:
        raise RuntimeError("compute_stats_streaming: zero rows survived filters")

    t_mean = t_sum / n_total
    t_var = np.maximum(t_sq / n_total - t_mean * t_mean, 0.0)
    t_std = np.sqrt(t_var)
    # Guard against perfectly-constant columns (var=0) so apply_preproc
    # doesn't divide by zero.
    t_std = np.where(t_std > 1e-6, t_std, 1.0)

    c_mean = c_sum / n_total
    c_var = np.maximum(c_sq / n_total - c_mean * c_mean, 0.0)
    c_std = np.sqrt(c_var)
    c_std = np.where(c_std > 1e-6, c_std, 1.0)

    # Normalisation scale: mean(|w|). Robust to MC@NLO-style signed
    # weights (where mean(w) can be near zero from cancellation) while
    # equivalent to mean(w) for ``mode=abs`` (where w is already
    # non-negative).
    weight_mean = abs_w_sum / n_total

    if progress:
        drop_label = {
            "abs":  "w==0 or non-finite",
            "keep": "non-finite",
            "drop": "w<=0 or non-finite",
        }[weight_mode]
        print(
            f"  scanned {n_total:,} rows"
            + (f" (dropped {n_filt:,} {drop_label})" if n_filt else "")
            + f"; |weight| mean = {weight_mean:.4g}"
            + (f", signed weight mean = {w_sum/n_total:.4g}"
               if weight_mode == "keep" else "")
        )

    return (
        PreprocStats(
            target_names=list(_TARGET_NAMES),
            target_mean=t_mean.tolist(),
            target_std=t_std.tolist(),
            cond_names=list(_COND_NAMES),
            cond_mean=c_mean.tolist(),
            cond_std=c_std.tolist(),
        ),
        float(weight_mean),
    )


# ---------------------------------------------------------------------------
# Streaming loader
# ---------------------------------------------------------------------------


class ArrowShardLoader(torch_data.IterableDataset):
    """Per-epoch streaming iterator over Arrow IPC shards.

    Yields ``(x, c, w)`` torch tensors per training batch. Matches the
    ``InMemoryLoader`` API (``__len__``, ``__iter__``, ``pin_memory``)
    so the trainer's epoch loop can switch from in-RAM to streaming
    by swapping the loader object. Also inherits
    :class:`torch.utils.data.IterableDataset`, so the same instance
    can be wrapped in a ``DataLoader`` with ``num_workers > 0``: each
    worker process gets its own shard subset (via ``get_worker_info``),
    bypasses the GIL bottleneck of the in-process producer thread,
    and shared-memory delivers tensors to the consumer.

    * **Per-rank partition**: shards are dealt out round-robin to
      ranks via ``shard_files[rank::world_size]``.
    * **Train/val split**: each record batch is sliced contiguously
      — first ``val_fraction`` of rows go to val, rest to train. The
      shards are already globally shuffled by the bucket-shuffle
      pass, so this is unbiased.
    * **Per-epoch shuffle**: shard read-order is permuted; within
      each emitted training batch (accumulated from many record
      batches) rows are permuted again before yielding. Strong
      enough that the trainer doesn't need to do additional
      shuffling.
    * **Preproc on-the-fly**: ``compute_targets_and_conditioning``
      + ``apply_preproc`` are applied per record batch.
      ``nominal_weight`` is divided by the global ``weight_mean`` so
      the mean-normalised weights match the in-RAM trainer's
      behaviour.
    * **Prefetch thread**: a single background thread runs the
      decompress + numpy preproc + pin-memory chain for the next
      ``prefetch`` batches while the trainer is busy on the current
      one (``prefetch=0`` disables and falls back to fully
      sequential iteration). pyarrow + numpy + ``Tensor.pin_memory``
      all release the GIL, so the prefetch overlaps cleanly with
      GPU compute.
    """

    def __init__(
        self,
        shard_files: Sequence[str],
        stats,                                 # PreprocStats
        weight_mean: float,
        *,
        world_size: int,
        rank: int,
        batch_size: int,
        split: str = "train",                  # "train" | "val" | "all"
        val_fraction: float = 0.1,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
        seed: int = 0,
        prefetch: int = 2,
        weight_mode: str = "abs",
        shard_row_counts: Sequence[int] | None = None,
        num_workers_hint: int = 1,
    ):
        if split not in ("train", "val", "all"):
            raise ValueError(f"split must be train|val|all, got {split!r}")
        if weight_mode not in _WEIGHT_MODES:
            raise ValueError(
                f"weight_mode must be one of {_WEIGHT_MODES}, got {weight_mode!r}"
            )
        self.shard_files = list(shard_files)
        self.my_shards = self.shard_files[rank::world_size]
        if shard_row_counts is not None:
            shard_row_counts = list(shard_row_counts)
            if len(shard_row_counts) != len(self.shard_files):
                raise ValueError(
                    f"shard_row_counts has {len(shard_row_counts)} entries "
                    f"but {len(self.shard_files)} shards were given"
                )
            self._my_row_counts = shard_row_counts[rank::world_size]
        else:
            self._my_row_counts = None
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.batch_size = int(batch_size)
        self.split = split
        self.val_fraction = float(val_fraction)
        self.shuffle = bool(shuffle)
        self.pin_memory = bool(pin_memory) and torch.cuda.is_available()
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.prefetch = max(0, int(prefetch))
        self.weight_mode = weight_mode
        self.num_workers_hint = max(1, int(num_workers_hint))
        self._epoch = 0
        # Per-record-batch layout. Lazily filled on the first
        # DataLoader-worker iteration (or anywhere that calls
        # ``_ensure_layout``); not needed for the standalone path.
        self._layout = None

        self._tmean = np.asarray(stats.target_mean, dtype=np.float32)
        self._tstd = np.asarray(stats.target_std, dtype=np.float32)
        self._cmean = np.asarray(stats.cond_mean, dtype=np.float32)
        self._cstd = np.asarray(stats.cond_std, dtype=np.float32)
        self._weight_mean = float(weight_mean)

        # Estimate length from the per-shard row counts. Used only by
        # the tqdm progress bar; correctness doesn't depend on it.
        # When caller passed ``shard_row_counts`` (e.g. populated
        # from manifest.json), use those directly — otherwise we'd
        # walk every record batch of every shard just to count rows,
        # which can take tens of seconds at scale.
        if self._my_row_counts is not None:
            per_shard_rows = self._my_row_counts
        else:
            per_shard_rows = (
                count_rows(self.my_shards) if self.my_shards else []
            )
        my_rows = sum(per_shard_rows)
        frac = (
            1.0 - self.val_fraction
            if split == "train"
            else self.val_fraction
            if split == "val"
            else 1.0
        )
        my_rows_split = int(my_rows * frac)
        # Always round up: with multiple DataLoader workers the
        # actual yielded count can EXCEED a floor-divided estimate,
        # which triggers PyTorch's "Length of IterableDataset was X
        # but Y samples have been fetched" warning. Also add a
        # ``+num_workers_hint`` cushion because each worker emits
        # its own tail partial batch when ``drop_last=False`` (and a
        # nearly-full one when ``drop_last=True``), so the
        # multi-worker total can exceed the single-process estimate
        # by up to ``num_workers``. tqdm reaches 100 % a handful of
        # iterations before the epoch finishes — fine.
        ceil_len = (
            (my_rows_split + self.batch_size - 1) // self.batch_size
        )
        self._approx_len = max(
            ceil_len + self.num_workers_hint,
            1,
        )

    def __len__(self):
        return self._approx_len

    def _pin(self, t: torch.Tensor) -> torch.Tensor:
        return t.pin_memory() if self.pin_memory else t

    def _yield(self, target, cond, w):
        x_t = self._pin(torch.from_numpy(np.ascontiguousarray(target)))
        c_t = self._pin(torch.from_numpy(np.ascontiguousarray(cond)))
        w_t = self._pin(torch.from_numpy(np.ascontiguousarray(w)))
        return x_t, c_t, w_t

    def _ensure_layout(self):
        """Populate ``self._layout`` (list of ``(shard_idx, batch_idx)``
        pairs across this rank's shards). Reads only IPC footers /
        message counts — no record-batch decompression. Cached after
        the first call; cheap to compute even at 32 shards × 4 500
        batches each (~ms)."""
        if self._layout is not None:
            return
        layout = []
        for s_idx, path in enumerate(self.my_shards):
            src, reader = _open_ipc(path)
            try:
                n_b = _num_record_batches(reader)
                if n_b < 0:
                    # Stream format — count by walking metadata.
                    n_b = sum(1 for _ in reader)
            finally:
                if hasattr(reader, "close"):
                    reader.close()
                src.close()
            for b in range(n_b):
                layout.append((s_idx, b))
        self._layout = layout

    def _partition_for_worker(self, worker_id: int, num_workers: int):
        """Return ``[(shard_idx, [batch_idx, ...]), ...]`` for this
        worker. Contiguous partition of the full record-batch layout
        across ``num_workers`` workers, then grouped by shard so each
        chunk only opens its shard once."""
        self._ensure_layout()
        total = len(self._layout)
        start = (total * worker_id) // num_workers
        stop = (total * (worker_id + 1)) // num_workers
        my_layout = self._layout[start:stop]
        chunks = []
        cur_shard = None
        cur_batches = None
        for s_idx, b_idx in my_layout:
            if s_idx != cur_shard:
                if cur_batches:
                    chunks.append((cur_shard, cur_batches))
                cur_shard = s_idx
                cur_batches = [b_idx]
            else:
                cur_batches.append(b_idx)
        if cur_batches:
            chunks.append((cur_shard, cur_batches))
        return chunks

    def __iter__(self):
        # Detect being called inside a ``DataLoader`` worker process
        # (``num_workers > 0``). When inside one we partition the
        # rank's shard slice further across workers, reseed the
        # per-epoch RNG so each worker gets independent shuffles,
        # and skip the internal pin_memory + prefetch — DataLoader
        # does both at the worker output stage. Outside a DataLoader
        # (or with ``num_workers == 0``) the original single-producer
        # path is used.
        wi = torch_data.get_worker_info()
        if wi is not None:
            # Partition at the record-batch level so DataLoader can
            # scale past ``n_shards`` workers. Each worker gets a
            # contiguous slice of the rank's full (shard_idx, batch_idx)
            # layout, grouped back into per-shard chunks so we only
            # open each shard once per worker per epoch. Requires
            # file-format shards for efficient random-access; for
            # stream-format the loop walks the whole stream and skips
            # batches not in the assigned set, which is wasteful but
            # functional.
            chunks = self._partition_for_worker(wi.id, wi.num_workers)
            orig_seed = self.seed
            orig_pin = self.pin_memory
            try:
                self.seed = (orig_seed * 1_000_003) ^ (wi.id + 1)
                self.pin_memory = False
                yield from self._iter_sync(chunks=chunks)
                return
            finally:
                self.seed = orig_seed
                self.pin_memory = orig_pin

        if self.prefetch <= 0:
            yield from self._iter_sync()
            return

        # Background-thread prefetch. The producer runs the same
        # body as ``_iter_sync`` and pushes ready tensors onto a
        # bounded queue; the trainer's epoch loop consumes them via
        # the yields below. With ``maxsize = prefetch`` the producer
        # stays at most ``prefetch`` batches ahead — enough to hide
        # CPU decompress + numpy preproc behind the previous step's
        # GPU compute, without runaway memory growth.
        import queue
        import threading

        q: "queue.Queue" = queue.Queue(maxsize=self.prefetch)
        sentinel = object()
        producer_exc: list = [None]
        stop_event = threading.Event()

        def _producer():
            try:
                for batch in self._iter_sync():
                    if stop_event.is_set():
                        return
                    # ``put`` blocks once the queue hits its cap —
                    # that's the backpressure.
                    q.put(batch)
            except BaseException as e:
                producer_exc[0] = e
            finally:
                q.put(sentinel)

        t = threading.Thread(
            target=_producer,
            name=f"ArrowShardLoader-prefetch-r{self.rank}",
            daemon=True,
        )
        t.start()
        try:
            while True:
                item = q.get()
                if item is sentinel:
                    break
                yield item
            if producer_exc[0] is not None:
                raise producer_exc[0]
        finally:
            # Make sure the producer wakes up if the consumer
            # abandoned the iterator early (e.g. break).
            stop_event.set()
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
            t.join(timeout=5.0)

    def _iter_sync(self, chunks=None):
        """Iterate one epoch's worth of training batches.

        ``chunks`` is a list of ``(shard_idx, batch_indices_or_None)``
        pairs. ``None`` means "process every record batch in this
        shard" (the standalone path). When provided (DataLoader
        worker path), only the listed record batches are fetched —
        random-access for file-format shards, walk-and-skip for
        stream-format.
        """
        epoch = self._epoch
        self._epoch += 1
        # Distinct RNG per (rank, epoch) — independent shuffles across
        # workers but reproducible per (seed, rank, epoch).
        rng = np.random.default_rng(
            (self.seed * 1_000_003)
            ^ (self.rank * 7919 + 1)
            ^ (epoch * 2_654_435_761)
        )

        if chunks is None:
            chunks = [(i, None) for i in range(len(self.my_shards))]
        else:
            # Defensive copy — we shuffle this list in place below.
            chunks = list(chunks)
        if self.shuffle:
            rng.shuffle(chunks)

        # Accumulator: list of arrays per column-group; merged when
        # we have enough rows to emit one training batch.
        buf_t: List[np.ndarray] = []
        buf_c: List[np.ndarray] = []
        buf_w: List[np.ndarray] = []
        buf_n = 0
        bs = self.batch_size

        def _flush_one():
            """Concatenate the accumulator, take ``bs`` rows (shuffled),
            stash the remainder, yield."""
            nonlocal buf_t, buf_c, buf_w, buf_n
            target_cat = np.concatenate(buf_t, axis=0)
            cond_cat = np.concatenate(buf_c, axis=0)
            w_cat = np.concatenate(buf_w, axis=0)
            if self.shuffle:
                perm = rng.permutation(target_cat.shape[0])
                target_cat = target_cat[perm]
                cond_cat = cond_cat[perm]
                w_cat = w_cat[perm]
            x = target_cat[:bs]
            c = cond_cat[:bs]
            ww = w_cat[:bs]
            rem_t = target_cat[bs:]
            rem_c = cond_cat[bs:]
            rem_w = w_cat[bs:]
            buf_t = [rem_t] if rem_t.shape[0] else []
            buf_c = [rem_c] if rem_c.shape[0] else []
            buf_w = [rem_w] if rem_w.shape[0] else []
            buf_n = rem_t.shape[0]
            return self._yield(x, c, ww)

        for shard_idx, batch_indices in chunks:
            path = self.my_shards[shard_idx]
            src, reader = _open_ipc(path)
            try:
                if batch_indices is None:
                    batches_iter = _iter_record_batches(reader)
                elif isinstance(reader, ipc.RecordBatchFileReader):
                    # Random-access by index. Sort to keep the on-disk
                    # read pattern sequential.
                    batches_iter = (
                        reader.get_batch(bi) for bi in sorted(batch_indices)
                    )
                else:
                    # Stream format — walk the whole stream and pick
                    # only the assigned indices. Wasteful (each worker
                    # re-decompresses earlier batches just to skip
                    # them), but functional.
                    idx_set = set(batch_indices)
                    batches_iter = (
                        b for i, b in enumerate(reader) if i in idx_set
                    )
                for batch in batches_iter:
                    cols = _read_raw_columns(batch)
                    target, cond, w = _per_batch_target_cond(cols)
                    n = target.shape[0]
                    if n == 0:
                        continue

                    # Train/val split (contiguous per record batch).
                    n_val = int(n * self.val_fraction)
                    if self.split == "train":
                        target = target[n_val:]
                        cond = cond[n_val:]
                        w = w[n_val:]
                    elif self.split == "val":
                        target = target[:n_val]
                        cond = cond[:n_val]
                        w = w[:n_val]
                    n = target.shape[0]
                    if n == 0:
                        continue

                    # Filter rows: target/cond must be finite, weight
                    # mode applies its own keep mask (abs: drop w==0
                    # & non-finite; keep: drop only non-finite; drop:
                    # drop w<=0 & non-finite).
                    target_finite = (
                        np.isfinite(target).all(axis=1)
                        & np.isfinite(cond).all(axis=1)
                    )
                    w_out, w_keep = _apply_weight_mode(w, self.weight_mode)
                    keep = target_finite & w_keep
                    if not keep.all():
                        target = target[keep]
                        cond = cond[keep]
                        w_out = w_out[keep]

                    if target.shape[0] == 0:
                        continue

                    # Standardise target + cond; normalise weight by
                    # global mean(|w|).
                    target = (target - self._tmean) / self._tstd
                    cond = (cond - self._cmean) / self._cstd
                    w = w_out / self._weight_mean

                    buf_t.append(target)
                    buf_c.append(cond)
                    buf_w.append(w)
                    buf_n += target.shape[0]

                    while buf_n >= bs:
                        yield _flush_one()
            finally:
                if hasattr(reader, "close"): reader.close()
                src.close()

        if buf_n > 0 and not self.drop_last:
            target_cat = np.concatenate(buf_t, axis=0)
            cond_cat = np.concatenate(buf_c, axis=0)
            w_cat = np.concatenate(buf_w, axis=0)
            if self.shuffle:
                perm = rng.permutation(target_cat.shape[0])
                target_cat = target_cat[perm]
                cond_cat = cond_cat[perm]
                w_cat = w_cat[perm]
            yield self._yield(target_cat, cond_cat, w_cat)
