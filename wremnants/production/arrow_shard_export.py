"""Bucket-shuffle + Arrow-IPC shard export from one or more RNTuple
per-muon snapshots.

Each input snapshot is an RNTuple where every row is one *event*
carrying a ``ROOT::RVec<float>`` of length N (number of selected
muons in that event) per output column. Snapshots produced by
``flow_training_snapshot.py`` in ``--source jpsi`` mode have N == 2
per row; ``--source wz`` mode has variable N. All snapshots passed in
``snapshot_paths`` must share the same per-event column schema (which
this module then flattens to per-muon rows).

Two-phase parallel pipeline (``multiprocessing.Pool`` of W workers):

  Phase 1 — disjoint-entry-range scan + per-bucket dispatch.
    For each input snapshot S, each phase-1 task reads a worker-slot
    slice of S's entry range in ``step_rows``-row windows via
    ``uproot.arrays()`` (``iterate`` is avoided because uproot 5.7
    silently ignores ``entry_stop`` for RNTuple). The ragged
    ``RVec<float>`` columns are flattened to per-muon arrays;
    ``bucket_id`` is computed *per muon* via a SplitMix64 hash of
    ``(source_id << 56) | (entry << 8) | muon_idx_in_event``. Per-
    muon rows are dispatched into per-(source, worker, bucket) raw
    fp32 binary intermediates under ``tmp_dir``.

  Phase 2 — bucket-aligned merge + shuffle + Arrow IPC write.
    Each shard worker w is assigned buckets ``range(w, B, W)``
    (round-robin). For each assigned bucket b, it concatenates every
    ``s*_w*_b{b}.bin`` produced in phase 1, applies an in-RAM uniform
    permutation, and writes the rows to its output Arrow IPC stream
    in fixed-size record batches (``batch_rows``).

After phase 2 the (source × worker × bucket) intermediates are
deleted; the final shards plus a ``manifest.json`` describing the
layout remain.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from typing import Sequence

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import uproot

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_GOLDEN_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)


def _splitmix64_bucket(
    source_id: int,
    entries: np.ndarray,
    muon_idxs: np.ndarray,
    seed: int,
    n_buckets: int,
) -> np.ndarray:
    """SplitMix64 finaliser, vectorised over numpy uint64.

    Mirrors the C++ helper that the previous bucket_id-Define used in
    RDF, but is now computed at shard time per *muon* — combining
    source, entry, and muon-index-in-event into a single 64-bit key.
    Reproducible per (source_id, entry, muon_idx, seed, n_buckets).
    """
    src = np.uint64(source_id) << np.uint64(56)
    x = (
        src
        | (entries.astype(np.uint64) << np.uint64(8))
        | (muon_idxs.astype(np.uint64) & np.uint64(0xFF))
    )
    x = (x + np.uint64(seed) + _GOLDEN_GAMMA) & _MASK64
    x = ((x ^ (x >> np.uint64(30))) * _MIX1) & _MASK64
    x = ((x ^ (x >> np.uint64(27))) * _MIX2) & _MASK64
    x = x ^ (x >> np.uint64(31))
    return (x % np.uint64(n_buckets)).astype(np.int32)


# ---------------------------------------------------------------------------
# Intermediate row dtype
# ---------------------------------------------------------------------------


def _intermediate_row_dtype(branches, int_columns, int64_columns):
    """Build the numpy *structured* dtype used for the per-(source,
    worker, bucket) phase-1 intermediates. Each branch is stored at
    its native width — fp32 for ordinary columns, int32 for tag
    columns, int64 for high-precision id columns. Packed (no
    padding), so ``rec.tobytes()`` / ``np.fromfile(p, dtype=...)``
    round-trip bit-for-bit."""
    int_set = set(int_columns)
    int64_set = set(int64_columns)
    fields = []
    for c in branches:
        if c in int64_set:
            fields.append((c, np.int64))
        elif c in int_set:
            fields.append((c, np.int32))
        else:
            fields.append((c, np.float32))
    return np.dtype(fields)


# ---------------------------------------------------------------------------
# Phase 1: parallel scan + per-bucket dispatch
# ---------------------------------------------------------------------------


def _phase1_worker(arg_tuple):
    (
        source_id,
        worker_id,
        n_workers,
        snapshot_path,
        tree_name,
        branches,
        n_buckets,
        tmp_dir,
        flush_threshold_bytes,
        step_rows,
        seed,
        int_columns,
        int64_columns,
    ) = arg_tuple
    int_set = set(int_columns)
    int64_set = set(int64_columns)
    row_dtype = _intermediate_row_dtype(
        branches,
        int_columns,
        int64_columns,
    )

    src = uproot.open(snapshot_path)[tree_name]
    n_entries = int(src.num_entries)
    start = (n_entries * worker_id) // n_workers
    stop = (n_entries * (worker_id + 1)) // n_workers
    if stop <= start:
        return source_id, worker_id, 0, 0

    bufs: list[list[np.ndarray]] = [[] for _ in range(n_buckets)]
    bytes_in_bufs = 0

    def flush_all():
        nonlocal bytes_in_bufs
        for b in range(n_buckets):
            if not bufs[b]:
                continue
            path = os.path.join(
                tmp_dir,
                f"s{source_id:02d}_w{worker_id:04d}_b{b:05d}.bin",
            )
            with open(path, "ab") as f:
                for arr in bufs[b]:
                    f.write(arr.tobytes())
            bufs[b].clear()
        bytes_in_bufs = 0

    cur = start
    n_events_read = 0
    n_muons_emitted = 0
    while cur < stop:
        chunk_stop = min(cur + step_rows, stop)
        # awkward returns ListOffsetArray<float32> per RVec column.
        # arrays() honors both entry_start and entry_stop for RNTuple.
        chunk = src.arrays(
            list(branches),
            entry_start=cur,
            entry_stop=chunk_stop,
        )
        # Counts per event from the first column; all output columns
        # are aligned per the snapshot contract (same RVec length).
        counts = ak.num(chunk[branches[0]]).to_numpy().astype(np.int64)
        n_events = int(counts.shape[0])
        n_events_read += n_events
        if n_events == 0:
            cur = chunk_stop
            continue

        total = int(counts.sum())
        if total == 0:
            cur = chunk_stop
            continue
        n_muons_emitted += total

        # Per-muon entry index (global within source) and within-event
        # muon index.
        offsets = np.empty(n_events + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        event_idx_flat = np.repeat(
            np.arange(cur, cur + n_events, dtype=np.int64),
            counts,
        )
        muon_idx_flat = np.arange(total, dtype=np.int64) - np.repeat(
            offsets[:-1], counts
        )

        bids = _splitmix64_bucket(
            source_id,
            event_idx_flat,
            muon_idx_flat,
            seed,
            n_buckets,
        )

        # Pack each RVec column into the per-row structured record at
        # its native width. No bit-views or value-casts beyond the
        # final dtype of each field — int64 columns survive bit-exact.
        flat = np.empty(total, dtype=row_dtype)
        for c in branches:
            arr = ak.flatten(chunk[c]).to_numpy()
            if c in int64_set:
                flat[c] = arr.astype(np.int64, copy=False)
            elif c in int_set:
                flat[c] = arr.astype(np.int32, copy=False)
            else:
                flat[c] = arr.astype(np.float32, copy=False)

        # Dispatch per bucket appearing in this chunk.
        for b in np.unique(bids):
            mask = bids == b
            rows = flat[mask]
            bufs[int(b)].append(rows)
            bytes_in_bufs += rows.nbytes
        if bytes_in_bufs >= flush_threshold_bytes:
            flush_all()
        cur = chunk_stop
    flush_all()
    return source_id, worker_id, n_events_read, n_muons_emitted


# ---------------------------------------------------------------------------
# Phase 2: parallel merge + shuffle + Arrow IPC write
# ---------------------------------------------------------------------------


def _phase2_worker(arg_tuple):
    (
        worker_id,
        n_workers_phase2,
        n_buckets,
        n_workers_phase1,
        n_sources,
        branches,
        int_columns,
        int64_columns,
        tmp_dir,
        shard_dir,
        batch_rows,
        seed,
    ) = arg_tuple
    int_set = set(int_columns)
    int64_set = set(int64_columns)
    row_dtype = _intermediate_row_dtype(
        branches,
        int_columns,
        int64_columns,
    )

    my_buckets = list(range(worker_id, n_buckets, n_workers_phase2))

    # ``int_columns`` -> ``pa.int32()``, ``int64_columns`` ->
    # ``pa.int64()``, rest -> ``pa.float32()``. The phase-1
    # intermediates are a packed structured array of the same widths,
    # so per-field extraction is dtype-clean (no value-cast / bit-view
    # gymnastics) at Arrow write time.
    def _arrow_type(c):
        if c in int64_set:
            return pa.int64()
        if c in int_set:
            return pa.int32()
        return pa.float32()

    schema = pa.schema([(c, _arrow_type(c)) for c in branches])
    write_opts = ipc.IpcWriteOptions(
        compression="lz4_frame",
        use_threads=False,
    )
    rng = np.random.default_rng(
        int(seed) ^ (worker_id * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    )

    out_path = os.path.join(shard_dir, f"shard_{worker_id:05d}.arrow")
    tmp_out = out_path + ".tmp"

    n_rows_total = 0
    n_record_batches = 0
    bucket_row_counts: list[tuple[int, int]] = []

    # Arrow IPC ``new_file`` (not ``new_stream``) so each shard has a
    # footer record-batch index — lets downstream readers fetch
    # individual record batches without walking the whole shard
    # sequentially, which enables intra-shard parallelism in the
    # trainer's stats warmup.
    with pa.OSFile(tmp_out, "wb") as out_f, ipc.new_file(
        out_f, schema, options=write_opts
    ) as writer:
        for b in my_buckets:
            chunks: list[np.ndarray] = []
            for s in range(n_sources):
                for w in range(n_workers_phase1):
                    p = os.path.join(
                        tmp_dir,
                        f"s{s:02d}_w{w:04d}_b{b:05d}.bin",
                    )
                    if os.path.exists(p) and os.path.getsize(p) > 0:
                        arr = np.fromfile(p, dtype=row_dtype)
                        chunks.append(arr)
            if not chunks:
                bucket_row_counts.append((b, 0))
                continue

            rows = np.concatenate(chunks)
            del chunks
            perm = rng.permutation(len(rows))
            rows = rows[perm]
            n_rows_in_bucket = len(rows)
            bucket_row_counts.append((b, n_rows_in_bucket))

            for off in range(0, n_rows_in_bucket, batch_rows):
                sub = rows[off : off + batch_rows]
                arrays = []
                for c in branches:
                    # ``sub[c]`` is already the field's native dtype;
                    # ``ascontiguousarray`` only copies if the slice
                    # isn't already C-contiguous (it is here since
                    # ``rows`` was reordered by integer indexing).
                    col = np.ascontiguousarray(sub[c])
                    arrays.append(pa.array(col, type=_arrow_type(c)))
                batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
                writer.write_batch(batch)
                n_record_batches += 1

            n_rows_total += n_rows_in_bucket

            # Free the bucket's intermediates as soon as it's
            # consumed — keeps peak disk usage bounded.
            for s in range(n_sources):
                for w in range(n_workers_phase1):
                    p = os.path.join(
                        tmp_dir,
                        f"s{s:02d}_w{w:04d}_b{b:05d}.bin",
                    )
                    if os.path.exists(p):
                        os.remove(p)

    os.replace(tmp_out, out_path)
    return worker_id, n_rows_total, n_record_batches, bucket_row_counts


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_sharding_pass(
    snapshot_paths: Sequence[str],
    tree_name: str,
    branches: Sequence[str],
    n_buckets: int,
    n_workers: int,
    shard_dir: str,
    batch_rows: int = 16384,
    seed: int = 42,
    tmp_dir: str | None = None,
    flush_threshold_bytes: int = 64 * 1024 * 1024,
    step_rows: int = 2_000_000,
    int_columns: Sequence[str] = (),
    n_shards: int | None = None,
) -> dict:
    """Run the bucket-shuffle + Arrow IPC sharding pass over one or
    more RNTuple per-muon snapshots.

    Parallelism is split across two phases:

    * **phase 1** uses ``n_workers`` processes; each reads a disjoint
      entry slice from every input snapshot and routes rows into
      per-(source, worker, bucket) intermediate files. This is the
      I/O-heavy part of the pipeline (xrootd / disk read +
      decompression), so ``n_workers`` is typically set to the
      number of CPU cores.
    * **phase 2** uses ``n_shards`` processes; each handles a
      round-robin subset of buckets and writes exactly one output
      Arrow IPC shard file. Arrow IPC files cannot be written
      concurrently by multiple workers, so phase 2 is capped at
      one worker per shard. ``n_shards`` defaults to ``n_workers``
      for back-compat.

    ``int_columns`` lists branch names to write as ``pa.int32()`` in
    the Arrow IPC output (everything else is ``pa.float32()``). The
    per-(worker, bucket) phase-1 intermediates are a *packed
    structured numpy array* whose fields match the per-branch Arrow
    types one-to-one (fp32 / int32, with int64 reserved for future
    callers), so every column survives the disk round-trip at its
    native width with no value-casts or bit-views needed.

    Returns a dict matching the on-disk ``manifest.json`` payload.
    """
    branches = list(branches)
    int_columns = list(int_columns)
    int64_columns: list[str] = []  # reserved for future use; structured
    # intermediate already supports per-field int64 if a caller wires
    # it in. No int64 column is written by the current snapshot.
    unknown_int = [c for c in int_columns if c not in branches]
    if unknown_int:
        raise ValueError(f"int_columns refers to unknown branches: {unknown_int}")
    n_cols = len(branches)
    sources = list(snapshot_paths)
    n_sources = len(sources)
    if n_sources == 0:
        raise ValueError("run_sharding_pass: snapshot_paths is empty")
    if n_shards is None:
        n_shards = n_workers
    n_shards = int(n_shards)
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")

    os.makedirs(shard_dir, exist_ok=True)
    cleanup_tmp = tmp_dir is None
    if tmp_dir is None:
        parent = os.path.dirname(os.path.abspath(shard_dir)) or "."
        tmp_dir = tempfile.mkdtemp(prefix="arrow_shard_", dir=parent)
    else:
        os.makedirs(tmp_dir, exist_ok=True)

    # Per-source num_entries (one quick uproot.open).
    source_entries: list[int] = []
    for sp in sources:
        with uproot.open(sp) as f:
            source_entries.append(int(f[tree_name].num_entries))

    print(f"arrow-shard pass")
    for i, (sp, ne) in enumerate(zip(sources, source_entries)):
        print(f"  source[{i:02d}]: {sp}  ({ne:,} events)")
    print(f"  tree (RNTuple) name:     {tree_name!r}")
    print(f"  shard_dir:               {shard_dir}")
    print(f"  tmp_dir:                 {tmp_dir}")
    print(f"  branches:                {n_cols} cols  {branches}")
    if int_columns:
        print(f"  int32 columns:           {int_columns}")
    print(f"  n_buckets:               {n_buckets}")
    print(f"  phase-1 workers:         {n_workers}")
    print(f"  output shards:           {n_shards}")
    print(f"  shard rows/record batch: {batch_rows:,}")
    print(f"  shuffle seed:            {seed}")

    try:
        # ---- Phase 1 ----
        print(
            f"\nphase 1: parallel scan + per-(source, worker, bucket) " f"intermediates"
        )
        t0 = time.time()
        phase1_args = [
            (
                src_id,
                w,
                n_workers,
                sources[src_id],
                tree_name,
                branches,
                n_buckets,
                tmp_dir,
                flush_threshold_bytes,
                step_rows,
                seed,
                int_columns,
                int64_columns,
            )
            for src_id in range(n_sources)
            for w in range(n_workers)
        ]
        with mp.Pool(n_workers) as pool:
            p1_results = pool.map(_phase1_worker, phase1_args)
        t1 = time.time()
        n_events = sum(r[2] for r in p1_results)
        n_muons = sum(r[3] for r in p1_results)
        print(
            f"  scanned {n_events:,} events -> {n_muons:,} muons "
            f"across {n_sources} source(s) x {n_workers} workers "
            f"in {t1 - t0:.1f}s"
        )

        # ---- Phase 2 ----
        print(
            f"\nphase 2: parallel merge + shuffle + Arrow IPC + LZ4 write"
            f" ({n_shards} shard writer(s))"
        )
        t2 = time.time()
        # Phase 2 fans out by output shard (one writer per Arrow file
        # — they can't be concurrently appended to). Each shard
        # worker iterates ``range(s, n_buckets, n_shards)`` and reads
        # the per-(source, phase1_worker, bucket) intermediates that
        # phase 1 deposited.
        phase2_args = [
            (
                s,
                n_shards,
                n_buckets,
                n_workers,
                n_sources,
                branches,
                int_columns,
                int64_columns,
                tmp_dir,
                shard_dir,
                batch_rows,
                seed,
            )
            for s in range(n_shards)
        ]
        with mp.Pool(n_shards) as pool:
            p2_results = pool.map(_phase2_worker, phase2_args)
        t3 = time.time()
        total_rows = sum(r[1] for r in p2_results)
        total_batches = sum(r[2] for r in p2_results)
        per_shard = [(r[0], r[1], r[2]) for r in p2_results]
        per_shard.sort(key=lambda x: x[0])
        for w, rows, batches in per_shard:
            print(
                f"  shard_{w:05d}.arrow: {rows:>12,} rows  "
                f"{batches:>5} record batches"
            )
        print(
            f"  total: {total_rows:,} rows in {total_batches} batches  "
            f"({t3 - t2:.1f}s)"
        )
        print(f"\ntotal sharding wall: {t3 - t0:.1f}s")

        # Collect any per-snapshot side-car label files written by the
        # snapshot stage (``<basename>.source_meta.json``). Each entry
        # maps a *data-column* source_id (the integer stamped into the
        # ``source_id`` branch) to a human-readable sample name. Multiple
        # snapshots can contribute multiple ids (e.g. the J/psi snapshot
        # tags Pt0to8 vs Pt8toInf as base+1 / base). Last-write-wins on
        # collisions; this is just for display.
        source_labels: dict[str, str] = {}
        for sp in sources:
            root_path, _ = os.path.splitext(sp)
            meta_path = root_path + ".source_meta.json"
            if not os.path.exists(meta_path):
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"  warning: failed to parse {meta_path}: {exc!r}")
                continue
            for entry in meta.get("entries", []):
                try:
                    sid = int(entry["source_id"])
                    name = str(entry["sample_name"])
                except (KeyError, TypeError, ValueError) as exc:
                    print(
                        f"  warning: skipping malformed entry in "
                        f"{meta_path}: {entry!r} ({exc!r})"
                    )
                    continue
                source_labels[str(sid)] = name
        if source_labels:
            print(
                f"  source_id labels (from side-cars): "
                + ", ".join(
                    f"{k}={v!r}"
                    for k, v in sorted(source_labels.items(), key=lambda kv: int(kv[0]))
                )
            )

        manifest = {
            "n_shards": n_shards,
            "n_buckets": n_buckets,
            "total_rows": int(total_rows),
            "total_events_read": int(n_events),
            "batch_rows": int(batch_rows),
            "seed": int(seed),
            "compression": "lz4_frame",
            "schema": [
                {
                    "name": c,
                    "type": "int32" if c in set(int_columns) else "float32",
                }
                for c in branches
            ],
            "shard_files": [f"shard_{s:05d}.arrow" for s in range(n_shards)],
            "shard_row_counts": [
                {"shard": w, "n_rows": rows} for w, rows, _ in per_shard
            ],
            "sources": [
                {
                    "source_id": i,
                    "snapshot": os.path.abspath(sp),
                    "n_events": ne,
                }
                for i, (sp, ne) in enumerate(zip(sources, source_entries))
            ],
            "source_labels": source_labels,
            "tree_name": tree_name,
        }
        manifest_path = os.path.join(shard_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"manifest: {manifest_path}")
        return manifest
    finally:
        if cleanup_tmp and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
