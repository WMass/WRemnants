"""Produce a flat per-muon snapshot of MC for normalizing-flow training.

Two source modes share a single output schema; running both modes
yields snapshots that can be merged and shuffled together by the
sharding step.

  --source jpsi  Custom J/psi ideal-geometry ntuples (default). The
                 LBL muon calibration is applied to obtain post-
                 correction Mu{plus,minus}cor_{pt,eta,phi}; each event
                 row carries an RVec of length 2, one per muon.

  --source wz    Standard NanoAOD W/Z MC, enumerated via
                 ``wremnants.production.datasets.dataset_tools.getDatasets``
                 (filtered to processes in
                 ``wremnants.utilities.samples.wprocs + zprocs``). The
                 ``vetoMuonsPre && genMatchedMuons`` selection is
                 applied (gen-matched, loose veto preselection — same
                 mask used by w_z_muonresponse.py for its response
                 histograms). Each event row carries an RVec of the N
                 selected muons.

Both modes write the same unified per-event RVec schema (one row per
event; the sharder flattens to per-muon rows):

  eta_reco[N]     ROOT::RVec<float>   post-correction reco eta
  phi_reco[N]     ROOT::RVec<float>   post-correction reco phi
  eta_gen[N]      ROOT::RVec<float>   matched gen eta
  phi_gen[N]      ROOT::RVec<float>   matched gen phi
  kappa_reco[N]   ROOT::RVec<float>   q_reco / (pt_reco * cosh(eta_reco))
  kappa_gen[N]    ROOT::RVec<float>   q_gen  / (pt_gen  * cosh(eta_gen))
  nominal_weight[N] ROOT::RVec<float> event weight, replicated per muon
  source_id[N]    ROOT::RVec<int>     dataset tag (--source-id base; W/Z
                                      auto-increments per dataset)

``kappa = q / |p|`` is the signed inverse momentum (``q`` the muon
charge, ``|p| = pt * cosh(eta)``); it factors charge and ``|p|`` into
the single per-muon scalar the flow's ``r_kappa = kappa_reco/kappa_gen
- 1`` target consumes directly. ``sign(kappa)`` recovers the muon
charge when needed; ``pt = 1 / (|kappa| * cosh(eta))`` recovers pt.

For J/psi: N == 2, the RVec is built explicitly from the per-event
Mu{plus,minus}{cor,gen}_{pt,eta,phi} columns; kappa_reco / kappa_gen
have signs ``{+, -}`` by construction (sign-labelled muon legs).

For W/Z: N == popcount(selMuons), variable per event; the signs are
``Muon_correctedCharge[selMuons]`` and the sign of
``GenPart_pdgId[Muon_genPartIdx[selMuons]]`` respectively. They may
differ when reco mismeasures the curvature; the resulting sign flip
in ``kappa_reco`` makes ``r_kappa = -2`` and lets the flow learn the
charge-mismeasurement mode explicitly.

Outputs
-------
  --source jpsi  --output FILE                Single .root file.
  --source wz    --output-dir DIR             One .root per W/Z dataset
                                              (dir/<dataset>.root).

The sharding pass is not invoked from this script when running in
either snapshot mode. After producing one or more snapshots, invoke
``--shard-only --inputs file1.root [file2.root ...]`` to run the
bucket-shuffle + Arrow IPC sharding pipeline from
:mod:`wremnants.production.arrow_shard_export` over the union of
inputs. Bucket assignment is computed at shard time from
``(source_id, entry, muon_idx_in_event)``, so reproducibility depends
on the ordering of ``--inputs``.

Bare invocation (no mode flag) runs the full pipeline end-to-end:
``--source jpsi`` -> ``--source wz`` -> ``--shard-only`` using the
default output locations.
"""

import argparse
import json
import os
import sys
import time
from typing import List


def _source_meta_path(snapshot_path: str) -> str:
    """Return the path to the ``<basename>.source_meta.json`` side-car
    next to a snapshot file. Sits beside the snapshot so the sharder
    can find it without an out-of-band registry; if absent, the
    sharder simply skips that source in the manifest's label map."""
    root, _ = os.path.splitext(snapshot_path)
    return root + ".source_meta.json"


def _write_source_meta(snapshot_path: str, entries: List[dict]) -> None:
    """Write ``{"entries": [{"source_id": int, "sample_name": str}, ...]}``
    next to ``snapshot_path``. The sharder picks these up to build
    ``manifest.json``'s ``source_labels`` map, which downstream tools
    (e.g. ``shift_smear_reweight_diagnostics``) use to display readable
    legend labels in per-source plots."""
    meta_path = _source_meta_path(snapshot_path)
    payload = {"entries": list(entries)}
    with open(meta_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote source-id label side-car: {meta_path}")

# Heavy imports (ROOT, wremnants, narf, pyxrootd, …) are deferred to
# inside the ``run_jpsi_snapshot`` / ``run_wz_snapshot`` /
# ``run_shard_only`` functions. The orchestrator path (bare
# invocation, no mode flag) never touches them.
#
# **Import order matters** inside each run_* function:
#   1. ``import XRootD.client`` first — pins pyxrootd's bundled
#      libssl-1.1 + libcrypto-1.1 in the global symbol table so that
#      pyarrow (transitively pulled in by wremnants via hist /
#      uproot) can't later shadow ``CRYPTO_THREAD_*`` with its
#      OpenSSL-3 libcrypto and crash the xrootd TLS handshake in
#      ``OPENSSL_init_ssl``.
#   2. ``import ROOT`` second — fine because the installed pyxrootd
#      is pinned to ``xrootd<6`` (v5), so its ``libXrdCl.so.3``
#      matches the one ROOT bundles. With xrootd v6 the SONAMEs
#      diverged (pyxrootd shipped ``libXrdCl.so.6``) and ROOT's
#      ``TNetXNGFile::Open`` cross-dispatched into pyxrootd's
#      vtables and segfaulted — that's why the xrootd downgrade
#      matters and the ``XRootD.client``-first rule below is now
#      safe.


DEFAULT_JPSI_INPUT_PATHS = [
    # Pt8toInf samples -> source_id = base (e.g. 0).
    "root://eoscms.cern.ch//store/group/phys_smp/ec/bendavid/muoncal/"
    "JPsiToMuMu_Pt8toInf-pythia8/MuonGunUL2016_v722_"
    "RecJpsiPythiaPhotosPt8toInf_quality_novtx_noconstraint_idealgeom/",
    "root://eoscms.cern.ch//store/group/phys_smp/ec/bendavid/muoncal/"
    "JPsiToMuMu_Pt8toInf-pythia8/MuonGunUL2016_v722_"
    "RecJpsiPythiaPhotosPt8toInfExt1_quality_novtx_noconstraint_idealgeom/",
    # Pt0to8 samples -> source_id = base + 1.
    "root://eoscms.cern.ch//store/group/phys_smp/ec/bendavid/muoncal/"
    "JPsiToMuMu_Pt0to8-pythia8/MuonGunUL2016_v722_"
    "RecJpsiPythiaPhotosPt0to8_quality_novtx_noconstraint_idealgeom/",
    "root://eoscms.cern.ch//store/group/phys_smp/ec/bendavid/muoncal/"
    "JPsiToMuMu_Pt0to8-pythia8/MuonGunUL2016_v722_"
    "RecJpsiPythiaPhotosPt0to8Ext1_quality_novtx_noconstraint_idealgeom/",
]


def _jpsi_sample_source_offset(path: str) -> int:
    """Per-sample offset added to ``--source-id`` for J/psi.

    The two 8toInf samples share offset 0 (the historical default
    behaviour). The two 0to8 samples get offset 1 so they remain
    distinguishable in the ``source_id`` column. Anything else falls
    back to 0 — safer than failing on an unrecognised path. Mirrors
    the C++ matcher used by ``DefinePerSample`` at snapshot time.
    """
    return 1 if "Pt0to8" in path else 0


# Per-event RVec columns written for both --source modes. Trainer
# consumes the flattened (per-muon) rows after the sharding pass.
OUTPUT_BRANCHES = [
    "eta_reco",
    "phi_reco",
    "eta_gen",
    "phi_gen",
    "kappa_reco",
    "kappa_gen",
    "nominal_weight",
    "source_id",
    # Per-muon "muon source" -- ``Muon_genPartFlav`` (1 = prompt W/Z,
    # 15 = secondary from a τ decay) on the W/Z branch, hardcoded to
    # 443 (J/ψ PDG id) on the J/ψ branch. Used as a conditioning input
    # in the flow and shift+smear reweight training so the model can
    # learn class-dependent response shapes (e.g. the slightly biased
    # gen-matching geometry of τ-decay muons).
    "muon_source",
]

# Columns written as int32 (everything else is fp32). Carried through
# the sharder so downstream code can filter/split rows by source.
INT_BRANCHES = ("source_id", "muon_source")


class _HelpFmt(
    argparse.RawDescriptionHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
):
    """Keep the module docstring's raw newlines, and append "(default:
    ...)" to every option's help line automatically.
    """


def _default_shard_workers(hard_cap: int = 32):
    """Choose a phase-1 worker count that's unlikely to exceed the
    cgroup ``pids.max`` budget on a shared host.

    ``os.cpu_count()`` returns the *machine-wide* core count (256+ on
    a typical box) but the per-cgroup PID limit is usually 2048 with a
    few hundred PIDs already in flight from the user's session.
    Spawning ``cpu_count`` workers -- each its own process plus a few
    helper threads -- routinely hits ``EAGAIN`` on ``os.fork()`` at
    ``mp.Pool.__init__``. Cap the default to ``hard_cap=32`` (proven
    to fit under the typical 2048-PID budget) while still allowing
    bigger pools via ``--shard-workers N`` on hosts with a larger
    budget.
    """
    cpu = os.cpu_count() or 8
    # Best-effort cgroup pids.max read (returns "max" or an integer).
    pids_max = None
    for path in ("/sys/fs/cgroup/pids.max", "/sys/fs/cgroup/pids/pids.max"):
        try:
            with open(path) as f:
                raw = f.read().strip()
            if raw and raw != "max":
                pids_max = int(raw)
                break
        except (FileNotFoundError, ValueError, OSError):
            continue
    cgroup_cap = max(1, (pids_max // 8) if pids_max else hard_cap)
    return max(1, min(cpu, hard_cap, cgroup_cap))


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_HelpFmt,
    )
    mode = p.add_mutually_exclusive_group(required=False)
    mode.add_argument(
        "--source",
        choices=["jpsi", "wz"],
        help="Snapshot mode: 'jpsi' reads custom J/psi ideal-geometry "
        "ntuples (--input-paths); 'wz' enumerates W/Z MC datasets via "
        "wremnants getDatasets() (--data-path, --era). If neither this "
        "nor --shard-only is given, the script runs the full pipeline "
        "end-to-end (jpsi snapshot, wz snapshot, then shard) by "
        "re-invoking itself once per step in a subprocess (avoids the "
        "pyarrow + pyxrootd OpenSSL ABI clash that segfaults when "
        "both libraries coexist in one process).",
    )
    mode.add_argument(
        "--shard-only",
        action="store_true",
        help="Skip snapshot creation; run only the bucket-shuffle + "
        "Arrow IPC sharding pass over --inputs (one or more snapshot "
        "files produced by previous --source invocations).",
    )

    # ---- J/psi-source inputs ----------------------------------------
    p.add_argument(
        "--input-paths",
        nargs="+",
        default=DEFAULT_JPSI_INPUT_PATHS,
        metavar="PATH",
        help="(--source jpsi only) Top-level xrootd/posix paths or "
        "explicit .root files. Each entry is resolved via "
        "wremnants.production.datasets.dataset_tools.buildFileList. "
        "The built-in default is the two ideal-geometry J/psi MC "
        "samples used by jpsi_module_corrections.py.",
    )
    p.add_argument(
        "--input-tree",
        default="tree",
        help="(--source jpsi only) TTree name inside the J/psi input "
        "files.",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=-1,
        help="(--source jpsi only) Cap on number of input .root files "
        "after resolution. -1 uses all.",
    )

    # ---- W/Z-source inputs ------------------------------------------
    p.add_argument(
        "--data-path",
        default=None,
        help="(--source wz only) Base path passed to getDatasets() for "
        "NanoAOD discovery. None lets wremnants pick the default.",
    )
    p.add_argument(
        "--era",
        default="2016PostVFP",
        help="Era label. Used by pileup/vertex helpers (both modes) "
        "and for W/Z dataset selection.",
    )
    p.add_argument(
        "--nano-version",
        default="v9",
        help="(--source wz only) nanoVersion passed to getDatasets().",
    )
    p.add_argument(
        "--wz-max-files",
        type=int,
        default=-1,
        help="(--source wz only) Cap on input files per dataset (via "
        "getDatasets(maxFiles=)). -1 uses all.",
    )
    p.add_argument(
        "--filter-procs",
        nargs="*",
        default=None,
        help="(--source wz only) Optional list of dataset name "
        "substrings to KEEP in addition to the default W/Z filter. "
        "If unset, all samples.wprocs + samples.zprocs are kept "
        "(modulo --exclude-tau-procs).",
    )
    p.add_argument(
        "--exclude-tau-procs",
        action="store_true",
        help="(--source wz only) Drop the Wτν / Z→ττ samples from the "
        "W/Z snapshot. By default they are *kept*: the per-muon "
        "selection ``Muon_genPartFlav == 1 || == 15`` accepts both "
        "prompt and τ-decay muons, and the ``muon_source`` column "
        "records which one each muon is so the trainer can condition "
        "on it.",
    )

    # ---- W/Z calibration knobs (forwarded to define_corrected_muons) -
    p.add_argument(
        "--muonCorrMC",
        default="idealMC_lbltruth",
        help="(--source wz only) MC muon correction type. Default "
        "matches w_z_muonresponse.py.",
    )
    p.add_argument(
        "--muonCorrData",
        default="massfit",
        help="(--source wz only) Data muon correction type. Unused on "
        "MC-only snapshots; kept for define_corrected_muons API.",
    )
    p.add_argument(
        "--muonScaleVariation",
        default="smearingWeightsGaus",
        help="(--source wz only) Scale variation method (forwarded).",
    )
    p.add_argument(
        "--biasCalibration",
        default=None,
        help="(--source wz only) bias calibration mode. None disables "
        "the bias-helper branch in define_corrected_muons.",
    )
    p.add_argument(
        "--smearing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the data/MC resolution-matching smearing helper "
        "to the reco muons of *both* the J/ψ and W/Z snapshots. On by "
        "default so the two samples are directly comparable in the "
        "target distributions (same LBL + cvhideal calibration chain, "
        "same smearing). Pass --no-smearing to drop the smearing on "
        "both paths.",
    )

    # ---- source_id stamp ------------------------------------------
    p.add_argument(
        "--source-id",
        type=int,
        default=None,
        help="Integer stamped into the ``source_id`` column. "
        "--source jpsi: stamps every row with this value. "
        "--source wz: this is the base; dataset i (in getDatasets() "
        "order) gets ``source_id = base + i``. None auto-selects: 0 "
        "for --source jpsi, 100 for --source wz (avoids the default "
        "J/psi vs W/Z collision in an end-to-end run).",
    )

    # ---- Output ------------------------------------------------------
    p.add_argument(
        "--output",
        default=None,
        help="(--source jpsi only) Output .root file path. None falls "
        "back to flow_training_snapshot_jpsi.root in the cwd.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="(--source wz only) Output directory; one "
        "<dataset_name>.root is written per dataset. None falls back "
        "to ./flow_training_snapshot_wz/.",
    )
    p.add_argument(
        "--output-tree",
        default="tree",
        help="RNTuple name inside each output file.",
    )
    p.add_argument(
        "--snapshot-format",
        choices=["rntuple", "ttree"],
        default="rntuple",
        help="Output snapshot format. The sharding step requires "
        "RNTuple; TTree kept as a legacy path.",
    )

    # ---- Runtime knobs ----------------------------------------------
    p.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Number of threads for ImplicitMT. 0 leaves ROOT to choose.",
    )
    p.add_argument(
        "--pt-min",
        type=float,
        default=2.0,
        help="(--source jpsi only) Reject events where either gen pt "
        "is below this (GeV).",
    )
    p.add_argument(
        "--eta-max",
        type=float,
        default=2.5,
        help="(--source jpsi only) |gen eta| cut, applied to both "
        "muons.",
    )
    p.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        default=True,
        help="Disable the RDF progress bar (jpsi mode only).",
    )

    # ---- Sharding (--shard-only mode + post-snapshot shortcut) -----
    p.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="(--shard-only only) One or more snapshot .root files "
        "to merge + shard. None auto-detects "
        "./flow_training_snapshot_jpsi.root + "
        "./flow_training_snapshot_wz/*.root produced by the default "
        "--source jpsi / --source wz invocations.",
    )
    p.add_argument(
        "--n-buckets",
        type=int,
        default=1024,
        help="Bucket count for the shard pass. Per-bucket RAM during "
        "phase 2 is ~ total_rows / n_buckets.",
    )
    p.add_argument(
        "--shard-workers",
        type=int,
        default=_default_shard_workers(),
        help="Phase-1 parallel readers for the shard pass (xrootd / "
        "disk read + decompression). Default: an auto-selected value "
        "capped against the cgroup ``pids.max`` budget so the worker "
        "pool can actually be forked (each worker is a separate "
        "process plus a couple of threads; high values like "
        "``os.cpu_count() == 256`` typically exceed the per-cgroup "
        "PID limit on a shared host). Phase 2 is separately capped at "
        "``--shard-count`` writers (Arrow IPC files can't be "
        "concurrently appended to).",
    )
    p.add_argument(
        "--shard-count",
        type=int,
        default=32,
        help="Number of output Arrow IPC shard files. Each is "
        "written by one phase-2 worker; intra-shard parallelism "
        "comes from phase 1's larger pool.",
    )
    p.add_argument(
        "--shard-output-dir",
        default=None,
        help="(--shard-only only) Directory for Arrow IPC shards + "
        "manifest.json. None falls back to <inputs[0] dir>/shards.",
    )
    p.add_argument(
        "--shard-rows-per-batch",
        type=int,
        default=16384,
        help="Rows per Arrow record batch in each shard.",
    )
    p.add_argument(
        "--shard-seed",
        type=int,
        default=42,
        help="Seed for the SplitMix64 bucket hash (applied to "
        "(source_id, entry, muon_idx)) and the in-bucket shuffle "
        "permutations.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# J/psi: explicit two-muon RVec build from the side-by-side per-event columns
# ---------------------------------------------------------------------------


def _declare_jpsi_sample_helper():
    """JIT-declare the per-sample source_id matcher used by
    ``DefinePerSample`` in J/psi mode. The matcher inspects the chain
    link string (which embeds the input file URL) and returns
    ``base + 1`` for Pt0to8 samples and ``base`` for everything else
    (Pt8toInf or unrecognised). Idempotent — repeated declarations
    are no-ops in cling.
    """
    import ROOT
    ROOT.gInterpreter.Declare(
        """
        #ifndef _FLOW_JPSI_SAMPLE_HELPER_DECLARED
        #define _FLOW_JPSI_SAMPLE_HELPER_DECLARED
        namespace _flow_jpsi {
            inline int sample_source_id(
                const ROOT::RDF::RSampleInfo& info, int base
            ) {
                const std::string s = info.AsString();
                if (s.find("Pt0to8") != std::string::npos) return base + 1;
                return base;
            }
        }
        #endif
        """
    )


def _define_jpsi_rvecs(df, source_id_base: int, smearing_helper=None):
    """Define the unified per-muon RVec schema on a J/psi RDataFrame.

    Each event becomes a 2-element RVec (mu+ first, mu- second).
    The kappa pair has signs ``{+, -}`` by construction (sign-labelled
    legs); the trainer recovers gen charge as ``sign(kappa_gen)`` and
    pt as ``1 / (|kappa| * cosh(eta))``. ``source_id`` is assigned
    per-sample via ``DefinePerSample`` from the chain link path:
    Pt0to8 samples get ``source_id_base + 1``, everything else
    (Pt8toInf or unrecognised) gets ``source_id_base``.

    When ``smearing_helper`` is provided the reco pt is smeared with
    the same per-(η, pt) resolution-matching helper used on the W/Z
    branch so the two snapshots are directly comparable. The smear
    affects only pt; (η, φ) and the gen leg are untouched. The seed
    triplet is the J/ψ ntuple's own ``(run, lumi, event)`` branches
    (note ``lumi``, not the NanoAOD ``luminosityBlock`` -- the J/ψ
    calibration ntuples use the shorter name) so the same physical
    event always gets the same smear across re-runs.
    """
    _declare_jpsi_sample_helper()
    df = df.DefinePerSample(
        "sample_source_id",
        f"_flow_jpsi::sample_source_id(rdfsampleinfo_, {int(source_id_base)})",
    )
    df = df.Define(
        "eta_reco",
        "ROOT::VecOps::RVec<float>{"
        " static_cast<float>(Mupluscor_eta),"
        " static_cast<float>(Muminuscor_eta)}",
    )
    df = df.Define(
        "phi_reco",
        "ROOT::VecOps::RVec<float>{"
        " static_cast<float>(Mupluscor_phi),"
        " static_cast<float>(Muminuscor_phi)}",
    )
    df = df.Define(
        "eta_gen",
        "ROOT::VecOps::RVec<float>{"
        " static_cast<float>(Muplusgen_eta),"
        " static_cast<float>(Muminusgen_eta)}",
    )
    df = df.Define(
        "phi_gen",
        "ROOT::VecOps::RVec<float>{"
        " static_cast<float>(Muplusgen_phi),"
        " static_cast<float>(Muminusgen_phi)}",
    )
    # kappa = q / |p| = q / (pt * cosh(eta)). Sign-labelled legs give
    # the {+1, -1} signs; |p| reuses the same pt/eta we just defined.
    # When the smearing helper is requested, smear the LBL-corrected
    # pt RVec with the same data/MC resolution-matching helper used on
    # the W/Z branch (only pt is smeared; (η, φ) are unchanged).
    if smearing_helper is not None:
        df = df.Define(
            "_jpsi_recopt_pre_smear",
            "ROOT::VecOps::RVec<float>{"
            " static_cast<float>(Mupluscor_pt),"
            " static_cast<float>(Muminuscor_pt)}",
        )
        df = df.Define(
            "_jpsi_recopt_smeared",
            smearing_helper,
            [
                # The J/ψ calibration ntuples expose ``lumi`` (not the
                # NanoAOD ``luminosityBlock``); ``run`` and ``event``
                # match the standard names.
                "run", "lumi", "event",
                "_jpsi_recopt_pre_smear", "eta_reco",
            ],
        )
        df = df.Define(
            "kappa_reco",
            "ROOT::VecOps::RVec<float>{"
            " +1.0f / (_jpsi_recopt_smeared[0] * std::cosh(eta_reco[0])),"
            " -1.0f / (_jpsi_recopt_smeared[1] * std::cosh(eta_reco[1]))}",
        )
    else:
        df = df.Define(
            "kappa_reco",
            "ROOT::VecOps::RVec<float>{"
            " +1.0f / (static_cast<float>(Mupluscor_pt) "
            "          * std::cosh(static_cast<float>(Mupluscor_eta))),"
            " -1.0f / (static_cast<float>(Muminuscor_pt) "
            "          * std::cosh(static_cast<float>(Muminuscor_eta)))}",
        )
    df = df.Define(
        "kappa_gen",
        "ROOT::VecOps::RVec<float>{"
        " +1.0f / (static_cast<float>(Muplusgen_pt) "
        "          * std::cosh(static_cast<float>(Muplusgen_eta))),"
        " -1.0f / (static_cast<float>(Muminusgen_pt) "
        "          * std::cosh(static_cast<float>(Muminusgen_eta)))}",
    )
    df = df.Define(
        "nominal_weight",
        "ROOT::VecOps::RVec<float>{"
        " static_cast<float>(nominal_weight_event),"
        " static_cast<float>(nominal_weight_event)}",
    )
    df = df.Define(
        "source_id",
        "ROOT::VecOps::RVec<int>{sample_source_id, sample_source_id}",
    )
    # Hardcode the J/ψ PDG id (443) as the per-muon ``muon_source``;
    # there is no ``Muon_genPartFlav`` equivalent on the calibration
    # ntuples, and a sentinel value distinguishes J/ψ from the
    # ``Muon_genPartFlav`` values (1 / 15) written on the W/Z branch.
    df = df.Define(
        "muon_source",
        "ROOT::VecOps::RVec<int>{443, 443}",
    )
    return df


def resolve_jpsi_input_paths(paths: List[str]) -> List[str]:
    # Caller (run_jpsi_snapshot) has already imported ROOT, so
    # pyxrootd loaded here through dataset_tools comes second and
    # ROOT's libXrdCl wins the symbol resolution.
    from wremnants.production.datasets import dataset_tools
    out: List[str] = []
    for p in paths:
        if p.lower().endswith(".root"):
            out.append(p)
            continue
        found = dataset_tools.buildFileList(p)
        if not found:
            print(
                f"warning: no .root files found under {p}", file=sys.stderr
            )
        out.extend(found)
    return out


def run_jpsi_snapshot(args) -> str:
    # ``XRootD.client`` first: pins pyxrootd's bundled libssl-1.1 +
    # libcrypto-1.1 in the global symbol table before pyarrow's
    # libcrypto-3 (pulled in via ``wremnants.production.muon_calibration``
    # -> hist / uproot) can shadow them. Requires xrootd pip package
    # < 6 so that pyxrootd's libXrdCl matches ROOT's (both v5).
    import XRootD.client  # noqa: F401
    import ROOT
    import wremnants
    import wremnants.production.muon_calibration
    import wremnants.production.pileup
    import wremnants.production.vertex

    out = args.output or "flow_training_snapshot_jpsi.root"
    out_dir = os.path.dirname(os.path.abspath(out))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print("resolving J/psi input paths")
    files = resolve_jpsi_input_paths(args.input_paths)
    if not files:
        print("error: no input files resolved", file=sys.stderr)
        return ""
    if args.max_files > 0 and len(files) > args.max_files:
        print(f"  capping from {len(files)} to {args.max_files} files")
        files = files[: args.max_files]
    print(f"  {len(files)} file(s) to read")

    # IMT is process-global; skip re-enabling if a previous step
    # already turned it on (bare-invocation pipeline runs all three
    # run_* in one process).
    if not ROOT.ROOT.IsImplicitMTEnabled():
        if args.threads == 0:
            ROOT.ROOT.EnableImplicitMT()
        elif args.threads > 1:
            ROOT.ROOT.EnableImplicitMT(args.threads)

    print("building RDF and applying LBL corrections")
    t0 = time.time()
    df = ROOT.ROOT.RDataFrame(args.input_tree, files)
    if args.progress:
        ROOT.ROOT.RDF.Experimental.AddProgressBar(df)

    helper = (
        wremnants.production.muon_calibration.make_muon_calibration_helper_single()
    )
    df = (
        wremnants.production.muon_calibration
        .define_lbl_corrections_jpsi_calibration_ntuples(df, helper)
    )

    pileup_helper = wremnants.production.pileup.make_pileup_helper(era=args.era)
    vertex_helper = wremnants.production.vertex.make_vertex_helper(era=args.era)
    df = df.DefinePerSample("weight", "1.0")
    df = df.Define("weight_pu", pileup_helper, ["Pileup_nTrueInt"])
    df = df.Define("weight_vtx", vertex_helper, ["Jpsigen_z", "Pileup_nTrueInt"])
    df = df.Define("nominal_weight_event", "weight*weight_pu*weight_vtx")

    df = df.Filter(
        f"std::fabs(Muplusgen_eta) < {args.eta_max} && "
        f"std::fabs(Muminusgen_eta) < {args.eta_max}",
        "eta_acceptance",
    )
    df = df.Filter(
        f"Muplusgen_pt > {args.pt_min} && Muminusgen_pt > {args.pt_min} && "
        f"Mupluscor_pt > 0. && Muminuscor_pt > 0.",
        "pt_positivity_and_gen_minimum",
    )

    source_id_base = 0 if args.source_id is None else int(args.source_id)
    if args.smearing:
        smearing_helper, _ = (
            wremnants.production.muon_calibration.make_muon_smearing_helpers()
        )
        print("  applying resolution smearing helper to J/ψ reco pt")
    else:
        smearing_helper = None
    df = _define_jpsi_rvecs(
        df,
        source_id_base=source_id_base,
        smearing_helper=smearing_helper,
    )
    # Print the planned per-sample assignment so the user can verify
    # the input files matched the expected Pt0to8 / Pt8toInf buckets.
    n_0to8 = sum("Pt0to8" in p for p in files)
    n_other = len(files) - n_0to8
    print(
        f"  source_id assignments (DefinePerSample):\n"
        f"    Pt8toInf / other -> {source_id_base}   ({n_other} file(s))\n"
        f"    Pt0to8           -> {source_id_base + 1}   ({n_0to8} file(s))"
    )

    snapshot_options = ROOT.RDF.RSnapshotOptions()
    if args.snapshot_format == "rntuple":
        snapshot_options.fOutputFormat = (
            ROOT.RDF.ESnapshotOutputFormat.kRNTuple
        )
        snapshot_options.fCompressionAlgorithm = (
            ROOT.RCompressionSetting.EAlgorithm.kLZ4
        )
        snapshot_options.fCompressionLevel = 1
        print(f"snapshotting to RNTuple LZ4 -> {out}")
    else:
        snapshot_options.fCompressionAlgorithm = (
            ROOT.RCompressionSetting.EAlgorithm.kZSTD
        )
        snapshot_options.fCompressionLevel = 5
        print(f"snapshotting to TTree ZSTD -> {out}")

    cols_vec = ROOT.std.vector("string")()
    for c in OUTPUT_BRANCHES:
        cols_vec.push_back(c)
    df.Snapshot(args.output_tree, out, cols_vec, snapshot_options)

    # Persist the source_id -> sample mapping next to the snapshot so
    # the sharder can merge it into manifest.json. The matcher tags
    # ``Pt0to8`` files with ``base+1`` and everything else with ``base``;
    # skip the +1 entry when no Pt0to8 files were used.
    jpsi_entries = [
        {"source_id": source_id_base, "sample_name": "J/psi (pT>8 GeV)"},
    ]
    if n_0to8 > 0:
        jpsi_entries.append(
            {"source_id": source_id_base + 1,
             "sample_name": "J/psi (pT<8 GeV)"}
        )
    _write_source_meta(out, jpsi_entries)

    dt = time.time() - t0
    print(f"snapshot done in {dt:.1f}s")
    print(f"output file size: {os.path.getsize(out) / 1e6:.1f} MB")
    return out


# ---------------------------------------------------------------------------
# W/Z: gen-matched vetoMuonsPre selection on standard NanoAOD MC
# ---------------------------------------------------------------------------


def _define_wz_rvecs(df, dataset, args, calib_helpers, source_id: int):
    import wremnants.production.muon_calibration
    import wremnants.production.muon_selections
    """Define the unified per-muon RVec schema on a W/Z RDataFrame.

    Applies the same selMuons mask as ``w_z_muonresponse.py``
    (vetoMuonsPre && genMatchedMuons) and projects per-muon RVecs from
    Muon_corrected* and the matched GenPart_* arrays.
    """
    pileup_helper, vertex_helper, mc_calibration_helper, mc_jpsi_crctn_helper, \
        smearing_helper, bias_helper = calib_helpers

    df = df.Define("weight", "std::copysign(1.0, genWeight)")
    df = df.Define("weight_pu", pileup_helper, ["Pileup_nTrueInt"])
    df = df.Define("weight_vtx", vertex_helper, ["GenVtx_z", "Pileup_nTrueInt"])
    df = df.Define("nominal_weight_event", "weight*weight_pu*weight_vtx")

    df = wremnants.production.muon_calibration.define_corrected_muons(
        df,
        mc_calibration_helper,
        mc_jpsi_crctn_helper,
        args,
        dataset,
        smearing_helper,
        bias_helper,
    )
    df = wremnants.production.muon_selections.select_veto_muons(df, nMuons=-1)
    # Keep both prompt (genPartFlav == 1) and τ-decay (== 15) muons.
    # The per-muon ``Muon_genPartFlav`` is carried through as the
    # ``muon_source`` column so the trainer can condition on the class
    # (and the model can learn the slightly different gen-matching
    # geometry / softer pT spectrum of τ-decay muons).
    df = df.Define(
        "genMatchedMuons",
        "Muon_genPartFlav == 1 || Muon_genPartFlav == 15",
    )
    df = df.Define("selMuons", "vetoMuonsPre && genMatchedMuons")

    df = df.Define("selMuons_genPartIdx", "Muon_genPartIdx[selMuons]")
    df = df.Define(
        "selMuons_genPdgId",
        "Take(GenPart_pdgId, selMuons_genPartIdx)",
    )

    df = df.Define(
        "eta_reco",
        "ROOT::VecOps::RVec<float>(Muon_correctedEta[selMuons])",
    )
    df = df.Define(
        "phi_reco",
        "ROOT::VecOps::RVec<float>(Muon_correctedPhi[selMuons])",
    )
    df = df.Define(
        "eta_gen",
        "ROOT::VecOps::RVec<float>(Take(GenPart_eta, selMuons_genPartIdx))",
    )
    df = df.Define(
        "phi_gen",
        "ROOT::VecOps::RVec<float>(Take(GenPart_phi, selMuons_genPartIdx))",
    )

    # kappa = q / |p| = q / (pt * cosh(eta)). Reco uses
    # Muon_correctedCharge (which may mismeasure); gen uses the sign
    # of the matched GenPart_pdgId. For mismeasured muons,
    # sign(kappa_reco) != sign(kappa_gen), which the trainer's
    # r_kappa = kappa_reco/kappa_gen - 1 sees as a ~-2 mode.
    df = df.Define(
        "kappa_reco",
        "ROOT::VecOps::RVec<float>("
        "  ROOT::VecOps::RVec<float>(Muon_correctedCharge[selMuons])"
        "  / (ROOT::VecOps::RVec<float>(Muon_correctedPt[selMuons])"
        "     * cosh(eta_reco))"
        ")",
    )
    df = df.Define(
        "kappa_gen",
        "ROOT::VecOps::RVec<float>("
        "  (-1.0f*(selMuons_genPdgId > 0) + 1.0f*(selMuons_genPdgId < 0))"
        "  / (ROOT::VecOps::RVec<float>(Take(GenPart_pt, selMuons_genPartIdx))"
        "     * cosh(eta_gen))"
        ")",
    )

    df = df.Define(
        "nominal_weight",
        "ROOT::VecOps::RVec<float>(eta_reco.size(), "
        " static_cast<float>(nominal_weight_event))",
    )
    df = df.Define(
        "source_id",
        f"ROOT::VecOps::RVec<int>(eta_reco.size(), {int(source_id)})",
    )
    # Per-muon ``muon_source`` = ``Muon_genPartFlav`` for the selected
    # rows (1 = prompt, 15 = from a τ decay), cast to int. Mirrors the
    # J/ψ branch which writes 443 (J/ψ PDG id) instead.
    df = df.Define(
        "muon_source",
        "ROOT::VecOps::RVec<int>(Muon_genPartFlav[selMuons])",
    )

    return df


def run_wz_snapshot(args) -> List[str]:
    # ``XRootD.client`` first — see run_jpsi_snapshot for rationale.
    import XRootD.client  # noqa: F401
    import ROOT
    import wremnants
    import wremnants.production.muon_calibration
    import wremnants.production.pileup
    import wremnants.production.vertex
    from wremnants.production.datasets.dataset_tools import getDatasets
    from wremnants.utilities import common, samples

    out_dir = args.output_dir or "flow_training_snapshot_wz"
    os.makedirs(out_dir, exist_ok=True)

    # IMT is process-global; skip re-enabling if a previous step
    # already turned it on (bare-invocation pipeline runs all three
    # run_* in one process).
    if not ROOT.ROOT.IsImplicitMTEnabled():
        if args.threads == 0:
            ROOT.ROOT.EnableImplicitMT()
        elif args.threads > 1:
            ROOT.ROOT.EnableImplicitMT(args.threads)

    # Filter: keep only W/Z MC. --filter-procs (if set) is an additional
    # name-substring whitelist on top of samples.wprocs + samples.zprocs.
    # Tau-decay V samples (Wτν, Z→ττ) are now included by default; the
    # muon selection (``Muon_genPartFlav == 1 || == 15``) accepts both
    # prompt and τ-decay muons, and the per-muon ``muon_source`` column
    # records which one each muon is. ``--exclude-tau-procs`` drops the
    # τ samples if that's what you want.
    wz_names = set(samples.wprocs) | set(samples.zprocs)
    if args.exclude_tau_procs:
        tau_names = set(samples.wprocs_tau_minnlo) | set(
            samples.zprocs_tau_minnlo
        ) | set(samples.wprocs_tau_minnlo_2017G) | set(
            samples.zprocs_tau_minnlo_2017G
        )
        wz_names -= tau_names
    datasets = getDatasets(
        maxFiles=args.wz_max_files,
        filt=args.filter_procs,
        excl=None,
        extended=True,
        nanoVersion=args.nano_version,
        base_path=args.data_path,
    )
    datasets = [
        d for d in datasets
        if d.name in wz_names and not d.is_data
    ]
    if not datasets:
        print("error: no W/Z MC datasets matched", file=sys.stderr)
        return []
    print(f"selected {len(datasets)} W/Z MC dataset(s):")
    for d in datasets:
        print(f"  {d.name}  ({len(d.filepaths)} file(s))")

    # Calibration / smearing / bias helpers (W/Z analysis path).
    calib_filepaths = common.calib_filepaths
    # ``make_jpsi_crctn_helpers`` returns 2 values when
    # ``make_uncertainty_helper=False`` and 4 when ``True``.
    (
        mc_jpsi_crctn_helper,
        _,
    ) = wremnants.production.muon_calibration.make_jpsi_crctn_helpers(
        calib_filepaths,
        muon_corr_mc=args.muonCorrMC,
        muon_corr_data=args.muonCorrData,
        scale_var_method=args.muonScaleVariation,
        scale_A=0.0,
        scale_e=0.0,
        scale_M=0.0,
        make_uncertainty_helper=False,
    )
    (
        mc_calibration_helper,
        _,
        _,
    ) = wremnants.production.muon_calibration.make_muon_calibration_helpers(args)
    if args.smearing:
        smearing_helper, _ = (
            wremnants.production.muon_calibration.make_muon_smearing_helpers()
        )
    else:
        smearing_helper = None
    bias_helper = None  # --biasCalibration off by default

    pileup_helper = wremnants.production.pileup.make_pileup_helper(era=args.era)
    vertex_helper = wremnants.production.vertex.make_vertex_helper(era=args.era)
    calib_helpers = (
        pileup_helper, vertex_helper, mc_calibration_helper,
        mc_jpsi_crctn_helper, smearing_helper, bias_helper,
    )

    # Build all per-dataset graphs first, hold a TChain + RDF +
    # Snapshot result per dataset, then run all event loops in one
    # narf pass.
    chains = []
    dfs = []
    snapshot_handles = []
    output_paths = []

    snapshot_options = ROOT.RDF.RSnapshotOptions()
    if args.snapshot_format == "rntuple":
        snapshot_options.fOutputFormat = (
            ROOT.RDF.ESnapshotOutputFormat.kRNTuple
        )
        snapshot_options.fCompressionAlgorithm = (
            ROOT.RCompressionSetting.EAlgorithm.kLZ4
        )
        snapshot_options.fCompressionLevel = 1
    else:
        snapshot_options.fCompressionAlgorithm = (
            ROOT.RCompressionSetting.EAlgorithm.kZSTD
        )
        snapshot_options.fCompressionLevel = 5
    snapshot_options.fLazy = True

    cols_vec = ROOT.std.vector("string")()
    for c in OUTPUT_BRANCHES:
        cols_vec.push_back(c)

    # End-to-end-friendly default: jpsi mode stamps source_id=0,
    # wz mode bases at 100 so they don't collide.
    base_source_id = 100 if args.source_id is None else int(args.source_id)
    for i, dataset in enumerate(datasets):
        chain = ROOT.TChain("Events")
        for fpath in dataset.filepaths:
            chain.Add(fpath)
        chains.append(chain)
        df = ROOT.ROOT.RDataFrame(chain)
        sid = base_source_id + i
        df = _define_wz_rvecs(
            df, dataset, args, calib_helpers, source_id=sid,
        )
        out_path = os.path.join(out_dir, f"{dataset.name}.root")
        snap = df.Snapshot(args.output_tree, out_path, cols_vec, snapshot_options)
        dfs.append(df)
        snapshot_handles.append(snap)
        output_paths.append(out_path)
        # Persist source_id -> dataset.name next to each W/Z snapshot.
        # Safe to write up-front (before the event loop) since the
        # side-car carries metadata only — no event counts.
        _write_source_meta(
            out_path,
            [{"source_id": int(sid), "sample_name": str(dataset.name)}],
        )
        print(
            f"queued snapshot graph: {dataset.name} "
            f"(source_id={sid}) -> {out_path}"
        )

    print(f"\nrunning event loops across {len(dfs)} dataset(s)")
    t0 = time.time()
    # Deferred narf import: keeps pandas/pyarrow (pulled in by narf)
    # out of the process until *after* getDatasets()'s pyxrootd
    # handshake has succeeded. From here on, file I/O goes through
    # ROOT's xrootd plugin during the event loop, which doesn't share
    # symbol state with pyxrootd's libssl.
    import narf  # noqa: F401  (side effect: registers narf:: with cling)
    interval = 1 if sys.stdout.isatty() else 5 * 60
    # PyROOT won't auto-convert a Python list of post-Define
    # ``RInterface<...>`` nodes into the C++ ``vector<RNode>``
    # overload — build it explicitly with ``AsRNode``.
    rnode_vec = ROOT.std.vector("ROOT::RDF::RNode")()
    for df in dfs:
        rnode_vec.push_back(ROOT.RDF.AsRNode(df))
    ROOT.narf.RunGraphsWithProgressBar(rnode_vec, 1000, interval)
    # Touch each Snapshot handle so any lazy book-keeping is flushed.
    for h in snapshot_handles:
        h.GetValue()
    dt = time.time() - t0
    print(f"all snapshots done in {dt:.1f}s")
    for p in output_paths:
        if os.path.exists(p):
            print(f"  {p}  ({os.path.getsize(p) / 1e6:.1f} MB)")
        else:
            print(f"  {p}  (MISSING)")
    return output_paths


# ---------------------------------------------------------------------------
# Shard-only mode
# ---------------------------------------------------------------------------


def _autodetect_shard_inputs() -> List[str]:
    """Pick up the J/psi snapshot file + every W/Z snapshot in the
    default output locations used by --source jpsi / --source wz.
    Empty list if nothing's present.
    """
    found: List[str] = []
    jpsi_default = "flow_training_snapshot_jpsi.root"
    if os.path.isfile(jpsi_default):
        found.append(jpsi_default)
    wz_default_dir = "flow_training_snapshot_wz"
    if os.path.isdir(wz_default_dir):
        for fname in sorted(os.listdir(wz_default_dir)):
            if fname.endswith(".root"):
                found.append(os.path.join(wz_default_dir, fname))
    return found


def run_shard_only(args) -> int:
    inputs = list(args.inputs) if args.inputs else _autodetect_shard_inputs()
    if not inputs:
        print(
            "error: --shard-only requires --inputs (no default "
            "snapshots found in cwd: looked for "
            "./flow_training_snapshot_jpsi.root and "
            "./flow_training_snapshot_wz/*.root)",
            file=sys.stderr,
        )
        return 1
    if not args.inputs:
        print(f"auto-detected {len(inputs)} snapshot input(s):")
        for p in inputs:
            print(f"  {p}")
    args.inputs = inputs
    missing = [p for p in args.inputs if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"error: input not found: {p}", file=sys.stderr)
        return 1

    shard_dir = args.shard_output_dir
    if shard_dir is None:
        shard_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.inputs[0])) or ".",
            "shards",
        )

    from wremnants.production.arrow_shard_export import run_sharding_pass

    run_sharding_pass(
        snapshot_paths=list(args.inputs),
        tree_name=args.output_tree,
        branches=list(OUTPUT_BRANCHES),
        int_columns=list(INT_BRANCHES),
        n_buckets=int(args.n_buckets),
        n_workers=int(args.shard_workers),
        n_shards=int(args.shard_count),
        shard_dir=shard_dir,
        batch_rows=int(args.shard_rows_per_batch),
        seed=int(args.shard_seed),
    )
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    if args.shard_only:
        return run_shard_only(args)
    if args.source == "jpsi":
        ok = run_jpsi_snapshot(args)
        return 0 if ok else 1
    if args.source == "wz":
        paths = run_wz_snapshot(args)
        return 0 if paths else 1
    # Bare invocation: run all three steps sequentially in this
    # process. Each ``run_*`` function imports ``XRootD.client``
    # first to pin pyxrootd's libssl-1.1 before pyarrow loads
    # libcrypto-3, so the OpenSSL clash doesn't fire. With xrootd
    # pinned to v5 (pip ``xrootd<6``), the libXrdCl conflict that
    # previously forced subprocess isolation is gone too.
    print("=" * 70)
    print("end-to-end pipeline (no mode flag given):")
    print("  step 1: J/psi snapshot     -> flow_training_snapshot_jpsi.root")
    print("  step 2: W/Z snapshot       -> flow_training_snapshot_wz/*.root")
    print("  step 3: Arrow IPC sharding -> shards/")
    print("=" * 70)

    print("\n[step 1/3] J/psi snapshot")
    if not run_jpsi_snapshot(args):
        print("error: J/psi snapshot failed", file=sys.stderr)
        return 1

    print("\n[step 2/3] W/Z snapshot")
    if not run_wz_snapshot(args):
        print("error: W/Z snapshot failed", file=sys.stderr)
        return 1

    print("\n[step 3/3] Arrow IPC sharding")
    return run_shard_only(args)


if __name__ == "__main__":
    sys.exit(main())
