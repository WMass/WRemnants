"""Bucket-shuffle sharder for the J/ψ unbinned mass-fit ROOT snapshots.

Reads one or more per-event ROOT files (TTree or RNTuple) produced by
``jpsi_mass_fit_snapshot.py`` (typically one MC and one data file),
streams them in chunks via uproot, assigns each row to one of
``n_buckets`` via a SplitMix64 hash of ``(file_idx, row_idx, seed)``,
optionally shuffles within each bucket, and writes one Arrow IPC file
per non-empty bucket. The output bucket files are then the trainer's
shard inputs: each bucket contains rows from *all* input sources,
randomly interleaved, so every training mini-batch is automatically a
mix of MC + data.

Why this matters here even more than for the per-muon flow snapshot:

  * MC mixes Pt0to8 + Pt8toInf samples with different kinematic
    coverage — sequential reads bias early batches toward one regime.
  * Data spans multiple data-taking eras (Runs F-post / G / H for the
    current defaults) whose detector conditions differ.
  * The two-branch NLL needs data *and* MC in every mini-batch so all
    of (flow, MLP, θ_scale surrogate, θ_smear) see signal each step;
    pure-MC or pure-data batches give biased Adam updates per group.

The reader is streaming (chunked) so memory is O(``step_rows`` × n_cols)
during the scan, plus O(bucket_total_rows × n_cols) for the bucket
buffers before the per-bucket Arrow write. For J/ψ event-counts this
fits comfortably; adopt ``arrow_shard_export.py``'s spill-to-disk
approach if input scales demand it.
"""

from __future__ import annotations

import os
from typing import List, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import uproot


# SplitMix64 constants — same as wremnants.production.arrow_shard_export.
_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_GOLDEN_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)


def splitmix64_bucket(
    file_idx: int,
    row_idxs: np.ndarray,
    seed: int,
    n_buckets: int,
) -> np.ndarray:
    """SplitMix64 finaliser, vectorised over numpy uint64.

    Reproducible per ``(file_idx, row_idx, seed)``. ``file_idx`` lives
    in the top 16 bits, ``row_idx`` in the lower 48 — supports up to
    2⁴⁸ rows per file before collisions, plenty for any J/ψ run.
    """
    src = np.uint64(file_idx) << np.uint64(48)
    x = src | (row_idxs.astype(np.uint64) & np.uint64((1 << 48) - 1))
    x = (x + np.uint64(seed) + _GOLDEN_GAMMA) & _MASK64
    x = ((x ^ (x >> np.uint64(30))) * _MIX1) & _MASK64
    x = ((x ^ (x >> np.uint64(27))) * _MIX2) & _MASK64
    x = x ^ (x >> np.uint64(31))
    return (x % np.uint64(n_buckets)).astype(np.int32)


def _numpy_to_arrow_type(arr: np.ndarray) -> "pa.DataType":
    """Map a numpy dtype to a pyarrow type, with explicit handling of
    the integer / unsigned-char branches the snapshot writes."""
    if arr.dtype == np.uint8:
        return pa.uint8()
    if arr.dtype == np.int32:
        return pa.int32()
    if arr.dtype == np.int64:
        return pa.int64()
    if arr.dtype == np.float32:
        return pa.float32()
    if arr.dtype == np.float64:
        return pa.float64()
    # Fallback: let pyarrow infer.
    return pa.array(arr).type


def shard_jpsi_mass_inputs(
    input_paths: Sequence[str],
    output_dir: str,
    *,
    n_buckets: int = 64,
    seed: int = 0,
    shuffle_within_bucket: bool = True,
    prefix: str = "jpsi_shard",
    tree_name: str = "tree",
    step_rows: int = 100_000,
) -> List[str]:
    """Bucket-shuffle ``input_paths`` (ROOT files) into ``n_buckets``
    Arrow IPC shards.

    Returns the list of output paths (sorted by bucket id; empty
    buckets are skipped). All inputs are required to share the same
    column set (the snapshot enforces this by construction — MC + data
    run through the same OUTPUT_BRANCHES list).
    """
    if not input_paths:
        raise ValueError("input_paths must be non-empty")
    os.makedirs(output_dir, exist_ok=True)

    # Phase 1: stream inputs, dispatch rows to per-bucket per-column buffers.
    columns: list[str] | None = None
    col_types: dict[str, "pa.DataType"] | None = None
    # bucket -> column -> list[np.ndarray]
    bucket_cols: dict[int, dict[str, list[np.ndarray]]] = {}

    total_rows = 0
    per_file_rows = []
    for file_idx, path in enumerate(input_paths):
        with uproot.open(path) as f:
            tree = f[tree_name]
            file_rows = int(tree.num_entries)
            per_file_rows.append(file_rows)
            total_rows += file_rows

            if columns is None:
                # Use the order of fields uproot exposes; preserved
                # downstream as the Arrow schema column order.
                columns = list(tree.keys())
                bucket_cols = {
                    b: {c: [] for c in columns} for b in range(n_buckets)
                }
            else:
                this_cols = list(tree.keys())
                if set(this_cols) != set(columns):
                    raise ValueError(
                        f"column-set mismatch on {path!r}: extra "
                        f"{set(this_cols) - set(columns)}, missing "
                        f"{set(columns) - set(this_cols)}"
                    )

            # Stream the file in step_rows chunks.
            row_offset = 0
            while row_offset < file_rows:
                chunk_stop = min(row_offset + step_rows, file_rows)
                chunk = tree.arrays(
                    columns,
                    entry_start=row_offset,
                    entry_stop=chunk_stop,
                    library="np",
                )
                n = chunk[columns[0]].shape[0]
                # First chunk seen → record column dtypes for the Arrow schema.
                if col_types is None:
                    col_types = {
                        c: _numpy_to_arrow_type(chunk[c]) for c in columns
                    }
                row_idxs = np.arange(row_offset, row_offset + n, dtype=np.uint64)
                bids = splitmix64_bucket(file_idx, row_idxs, seed, n_buckets)
                for b in np.unique(bids):
                    mask = bids == b
                    for c in columns:
                        bucket_cols[int(b)][c].append(chunk[c][mask])
                row_offset = chunk_stop

    print(
        f"sharder: read {total_rows:,} rows from {len(input_paths)} file(s) "
        f"into {n_buckets} buckets"
    )
    for i, (p, n) in enumerate(zip(input_paths, per_file_rows)):
        print(f"  file {i}: {n:,} rows  {p}")

    # Phase 2: per bucket — concat, shuffle, write Arrow IPC.
    schema = pa.schema([(c, col_types[c]) for c in columns])
    rng = np.random.default_rng(seed)
    out_paths: list[str] = []
    bucket_counts: list[int] = []
    for b in range(n_buckets):
        if not bucket_cols[b][columns[0]]:
            bucket_counts.append(0)
            continue
        cols_concat = {c: np.concatenate(bucket_cols[b][c]) for c in columns}
        n_rows = len(cols_concat[columns[0]])
        if shuffle_within_bucket:
            perm = rng.permutation(n_rows)
            cols_concat = {c: cols_concat[c][perm] for c in columns}
        arrs = {c: pa.array(cols_concat[c], type=col_types[c]) for c in columns}
        table = pa.table(arrs, schema=schema)
        out_path = os.path.join(output_dir, f"{prefix}_{b:04d}.arrow")
        with pa.OSFile(out_path, "wb") as sink:
            writer = ipc.new_file(sink, schema)
            writer.write_table(table)
            writer.close()
        out_paths.append(out_path)
        bucket_counts.append(n_rows)

    n_nonempty = sum(1 for c in bucket_counts if c > 0)
    counts_nonzero = [c for c in bucket_counts if c > 0]
    mean = np.mean(counts_nonzero) if counts_nonzero else 0.0
    rms = np.std(counts_nonzero) if counts_nonzero else 0.0
    print(
        f"sharder: wrote {n_nonempty}/{n_buckets} non-empty shards to "
        f"{output_dir!r} (mean {mean:.0f}, rms {rms:.0f} rows/bucket)"
    )
    return out_paths
