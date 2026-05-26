"""Per-event J/ψ snapshot for the unbinned calibration fit.

Variant of ``flow_training_snapshot.py`` that writes **one row per
event** with the dilepton-level variables the new fit consumes
(``m_ll``, ``y_ll``, ``p_T^ll``, ``φ_ll``, ``cos θ*``, ``φ*``) plus
the per-muon kinematics needed to route θ to the right η-bins at
training time. Outputs an Arrow IPC file directly (no intermediate
RNTuple → sharder hop) since the per-event row count is small enough
to keep the pipeline single-file.

Both MC and data flow through the **same** LBL-corrected J/ψ
calibration ntuple schema (``Mupluscor_pt`` / ``Muminuscor_pt`` /
``Mupluscor_eta`` / ...). The MC ntuples additionally expose
gen-level branches (``Muplusgen_pt``, ``Jpsigen_mass``, ...). The
data path zero-fills the gen-level snapshot columns and uses unit
weights; everything else (HLT, η, pt, J/ψ-pt, mass window) is shared.

Schema (one row per event):

    mll, yll, ptll, cosPhill, sinPhill,
    cosThetaStarll, sinThetaStarll, sinPhiStarll, cosPhiStarll,
    pt_plus, eta_plus, phi_plus, q_plus,
    pt_minus, eta_minus, phi_minus, q_minus,
    nominal_weight    # MC weight × pileup × vertex (≡ 1 for data)
    is_data           # uint8 (0 = MC, 1 = data)
    source_id         # int32

No gen-level columns: the asymmetric design (Option II) treats
``θ_smear`` as the *residual* smearing applied on top of the LBL-
corrected MC reco, so data and MC consume identical reco columns.

The window cut ``m_lo ≤ m_ll ≤ m_hi`` is applied at this stage.

Run::

    python jpsi_mass_fit_snapshot.py mc --output /path/jpsi_mc.root ...
    python jpsi_mass_fit_snapshot.py data --output /path/jpsi_data.root ...
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

# Default J/ψ data ntuples (LBL-corrected, 2016 Charmonium Runs F-post / G / H).
DEFAULT_JPSI_DATA_INPUT_PATHS = [
    "root://eoscms.cern.ch//store/group/phys_smp/ec/bendavid/muoncal/"
    "Charmonium/MuonGunUL2016_v725_RecDataJPsiFpost_quality_novtx_noconstraint/",
    "root://eoscms.cern.ch//store/group/phys_smp/ec/bendavid/muoncal/"
    "Charmonium/MuonGunUL2016_v725_RecDataJPsiG_quality_novtx_noconstraint/",
    "root://eoscms.cern.ch//store/group/phys_smp/ec/bendavid/muoncal/"
    "Charmonium/MuonGunUL2016_v725_RecDataJPsiH_quality_novtx_noconstraint/",
]


# Per-event output columns.
OUTPUT_BRANCHES = [
    "mll",
    "yll",
    "ptll",
    "cosPhill",
    "sinPhill",
    "cosThetaStarll",
    "sinThetaStarll",
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
]

INT_BRANCHES = ("source_id",)
U8_BRANCHES = ("is_data",)


# ---------------------------------------------------------------------------
# Per-event dilepton + CS-frame defines, shared by MC and data paths
# ---------------------------------------------------------------------------


def _define_jpsi_event_vars(
    df,
    *,
    plus_pt: str,
    plus_eta: str,
    plus_phi: str,
    minus_pt: str,
    minus_eta: str,
    minus_phi: str,
    mu_mass_gev: float = 0.1056583755,
):
    """Define the per-event dilepton + Collins-Soper observables.

    The input column names parameterise which scalar pt / η / φ
    columns to read for the μ+ and μ− legs — letting the same helper
    serve both the MC (LBL-corrected ``Mu±cor_*``) and data
    (MuOnia / dedicated J/ψ stream) ntuples.
    """
    # μ+ / μ− 4-vectors.
    df = df.Define(
        "mu_plus_lv",
        f"ROOT::Math::PtEtaPhiMVector("
        f"  static_cast<double>({plus_pt}),"
        f"  static_cast<double>({plus_eta}),"
        f"  static_cast<double>({plus_phi}),"
        f"  {mu_mass_gev})",
    )
    df = df.Define(
        "mu_minus_lv",
        f"ROOT::Math::PtEtaPhiMVector("
        f"  static_cast<double>({minus_pt}),"
        f"  static_cast<double>({minus_eta}),"
        f"  static_cast<double>({minus_phi}),"
        f"  {mu_mass_gev})",
    )
    df = df.Define("ll_lv", "mu_plus_lv + mu_minus_lv")
    df = df.Define("mll", "static_cast<float>(ll_lv.M())")
    df = df.Define("yll", "static_cast<float>(ll_lv.Rapidity())")
    df = df.Define("ptll", "static_cast<float>(ll_lv.Pt())")
    df = df.Define("phill", "static_cast<double>(ll_lv.Phi())")
    df = df.Define("cosPhill", "static_cast<float>(std::cos(phill))")
    df = df.Define("sinPhill", "static_cast<float>(std::sin(phill))")
    # CS variables — antilepton is μ+, lepton is μ−. csSineCosThetaPhi
    # returns a CSVars struct with sintheta, costheta, sinphi, cosphi.
    df = df.Define("_csvars", "wrem::csSineCosThetaPhi(mu_plus_lv, mu_minus_lv)")
    df = df.Define("cosThetaStarll", "static_cast<float>(_csvars.costheta)")
    df = df.Define("sinThetaStarll", "static_cast<float>(_csvars.sintheta)")
    df = df.Define("cosPhiStarll", "static_cast<float>(_csvars.cosphi)")
    df = df.Define("sinPhiStarll", "static_cast<float>(_csvars.sinphi)")
    # Per-muon reco scalars (η-bin look-up + T_scale / T_smear inputs).
    df = df.Define("pt_plus", f"static_cast<float>({plus_pt})")
    df = df.Define("eta_plus", f"static_cast<float>({plus_eta})")
    df = df.Define("phi_plus", f"static_cast<float>({plus_phi})")
    df = df.Define("pt_minus", f"static_cast<float>({minus_pt})")
    df = df.Define("eta_minus", f"static_cast<float>({minus_eta})")
    df = df.Define("phi_minus", f"static_cast<float>({minus_phi})")
    df = df.Define("q_plus", "static_cast<float>(+1.0)")
    df = df.Define("q_minus", "static_cast<float>(-1.0)")
    return df


def _apply_window_cut(df, m_lo: float, m_hi: float):
    return df.Filter(
        f"mll >= {float(m_lo)}f && mll <= {float(m_hi)}f",
        "mll_window",
    )


# ---------------------------------------------------------------------------
# Common pipeline (MC + data share the LBL J/ψ calibration ntuple schema)
# ---------------------------------------------------------------------------


def _setup_jit_and_files(args, kind: str) -> tuple[object, list[str], float]:
    """Common preamble: pyxrootd-before-pyarrow, ROOT imports, JIT
    headers, input-path resolution, IMT enable. Returns ``(ROOT, files,
    t0)`` and prints progress lines.
    """
    # pyxrootd ahead of pyarrow: same ordering rule as the existing
    # snapshot to avoid the libcrypto-3 vs libcrypto-1.1 clash.
    import ROOT
    import XRootD.client  # noqa: F401

    from scripts.corrections.muon_calibration.flow_training_snapshot import (
        resolve_jpsi_input_paths,
    )

    # Make CS variables available to the RDataFrame JIT.
    import narf.clingutils as clingutils
    cs_header = os.path.join(
        os.environ["WREM_BASE"], "wremnants/production/include/csVariables.hpp"
    )
    clingutils.Declare(f'#include "{cs_header}"')

    print(f"resolving J/ψ {kind} input paths")
    files = resolve_jpsi_input_paths(args.input_paths)
    if not files:
        print("error: no input files resolved", file=sys.stderr)
        return ROOT, [], 0.0
    if args.max_files > 0 and len(files) > args.max_files:
        print(f"  capping from {len(files)} to {args.max_files} files")
        files = files[: args.max_files]
    print(f"  {len(files)} file(s) to read")

    if not ROOT.ROOT.IsImplicitMTEnabled():
        if args.threads == 0:
            ROOT.ROOT.EnableImplicitMT()
        elif args.threads > 1:
            ROOT.ROOT.EnableImplicitMT(args.threads)

    t0 = time.time()
    return ROOT, files, t0


def _print_branch_list(df, tree_name: str) -> None:
    """List the input columns visible on ``df``. Used by ``--list-branches``."""
    cols = sorted(str(c) for c in df.GetColumnNames())
    hlt = [c for c in cols if any(s in c for s in ("HLT", "Trig", "trig", "L1"))]
    print(f"tree {tree_name!r}: {len(cols)} columns")
    print(f"\nHLT / Trig / L1 matches: {len(hlt)}")
    for c in hlt:
        print(f"  {c}")
    print(f"\nALL {len(cols)} columns:")
    for c in cols:
        print(f"  {c}")


def _apply_reco_selection(df, args):
    """Apply the reco-level cuts shared by MC and data.

    Order matters for the RDF cut-flow report: cheap-and-discriminating
    cuts go first. HLT (Bool) → η acceptance → both-muons pt → leading
    pt. The J/ψ-pt and mass-window cuts are applied separately AFTER
    the event-level kinematics are defined.

    HLT is only skipped when ``--hlt-path ""`` is passed explicitly (the
    intended way to disable for MC gun-style productions that carry no
    HLT bits). A non-empty ``--hlt-path`` whose branch is missing from
    the tree will fail loudly at RDF JIT time, not silently pass through.
    """
    if args.hlt_path:
        # Allow multiple OR-ed HLT branches. Wrap in static_cast<bool>
        # so the named filter's bool-return check passes regardless of
        # the branch type (the J/ψ LBL ntuples store HLT_* as int, not
        # Bool_t) — the ``||`` alone is not enough when there's only one
        # path in the list (the join is a no-op then).
        expr = " || ".join(args.hlt_path)
        df = df.Filter(
            f"static_cast<bool>({expr})", f"hlt_OR_{len(args.hlt_path)}"
        )
    df = df.Filter(
        f"std::fabs(Mupluscor_eta) < {args.eta_max}f && "
        f"std::fabs(Muminuscor_eta) < {args.eta_max}f",
        "reco_eta_acceptance",
    )
    df = df.Filter(
        f"Mupluscor_pt > {args.muon_pt_min_both}f && "
        f"Muminuscor_pt > {args.muon_pt_min_both}f",
        "reco_muon_pt_lower",
    )
    df = df.Filter(
        f"std::max(Mupluscor_pt, Muminuscor_pt) > {args.muon_pt_min_leading}f",
        "reco_muon_pt_leading",
    )
    return df


# ---------------------------------------------------------------------------
# MC path
# ---------------------------------------------------------------------------


def run_jpsi_mass_snapshot_mc(args) -> str:
    """Snapshot per-event observables from the J/ψ MC calibration ntuples."""
    ROOT, files, t0 = _setup_jit_and_files(args, kind="MC")
    if not files:
        return ""

    import wremnants.production.muon_calibration
    import wremnants.production.pileup
    import wremnants.production.vertex

    # Resolve correction-file default lazily — see CLI helptext for why.
    if args.correction_file is None:
        args.correction_file = (
            wremnants.production.muon_calibration.data_dir
            + "/calibration/correctionResults_v718_idealgeom_gensim.root"
        )

    out = args.output or "jpsi_mass_fit_snapshot_mc.root"
    out_dir = os.path.dirname(os.path.abspath(out)) or "."
    os.makedirs(out_dir, exist_ok=True)

    print(f"building RDF and applying LBL corrections from {args.correction_file}")
    df = ROOT.ROOT.RDataFrame(args.input_tree, files)
    if args.progress:
        ROOT.ROOT.RDF.Experimental.AddProgressBar(df)

    if args.list_branches:
        _print_branch_list(df, args.input_tree)
        return ""

    helper = wremnants.production.muon_calibration.make_muon_calibration_helper_single(
        filename=args.correction_file
    )
    df = wremnants.production.muon_calibration.define_lbl_corrections_jpsi_calibration_ntuples(
        df, helper
    )

    pileup_helper = wremnants.production.pileup.make_pileup_helper(era=args.era)
    vertex_helper = wremnants.production.vertex.make_vertex_helper(era=args.era)
    df = df.DefinePerSample("weight", "1.0")
    df = df.Define("weight_pu", pileup_helper, ["Pileup_nTrueInt"])
    df = df.Define("weight_vtx", vertex_helper, ["Jpsigen_z", "Pileup_nTrueInt"])
    df = df.Define("nominal_weight", "static_cast<float>(weight*weight_pu*weight_vtx)")

    # Gen matching (MC only): Jpsigen_mass > 0 indicates a matched truth J/ψ.
    df = df.Filter("Jpsigen_mass > 0.f", "gen_match")

    df = _apply_reco_selection(df, args)

    df = _define_jpsi_event_vars(
        df,
        plus_pt="Mupluscor_pt",
        plus_eta="Mupluscor_eta",
        plus_phi="Mupluscor_phi",
        minus_pt="Muminuscor_pt",
        minus_eta="Muminuscor_eta",
        minus_phi="Muminuscor_phi",
    )
    df = df.Filter(f"ptll > {args.ptll_min}f", "reco_ptll")
    df = _apply_window_cut(df, args.m_lo, args.m_hi)

    df = df.Define("is_data", "static_cast<unsigned char>(0)")
    # Per-sample source_id offset (Pt0to8 → +1, Pt8toInf / other → +0),
    # mirroring the convention used by flow_training_snapshot.py.
    from scripts.corrections.muon_calibration.flow_training_snapshot import (
        _declare_jpsi_sample_helper,
    )
    _declare_jpsi_sample_helper()
    df = df.DefinePerSample(
        "source_id",
        f"_flow_jpsi::sample_source_id(rdfsampleinfo_, {int(args.source_id or 0)})",
    )

    return _snapshot_to_root(
        df, out, t0,
        tree_name=args.output_tree, snapshot_format=args.snapshot_format,
    )


# ---------------------------------------------------------------------------
# Data path — same LBL ntuple schema, no gen, unit weight
# ---------------------------------------------------------------------------


def run_jpsi_mass_snapshot_data(args) -> str:
    """Snapshot per-event observables from J/ψ data calibration ntuples.

    Same LBL-corrected ntuple schema as MC: ``Mupluscor_pt``,
    ``Muminuscor_pt``, ``Mupluscor_eta`` etc., the HLT trigger branch
    named by ``--hlt-path``. Per-event ``nominal_weight = 1`` and
    ``is_data = 1`` / ``source_id`` are stamped. Reco selection,
    J/ψ-pt and mass-window cuts are identical to the MC path.
    """
    ROOT, files, t0 = _setup_jit_and_files(args, kind="data")
    if not files:
        return ""

    import wremnants.production.muon_calibration

    if args.correction_file is None:
        args.correction_file = (
            wremnants.production.muon_calibration.data_dir
            + "/calibration/correctionResults_v721_recjpsidata.root"
        )

    out = args.output or "jpsi_mass_fit_snapshot_data.root"
    out_dir = os.path.dirname(os.path.abspath(out)) or "."
    os.makedirs(out_dir, exist_ok=True)

    print(f"building RDF and applying LBL corrections from {args.correction_file}")
    df = ROOT.ROOT.RDataFrame(args.input_tree, files)
    if args.progress:
        ROOT.ROOT.RDF.Experimental.AddProgressBar(df)

    if args.list_branches:
        _print_branch_list(df, args.input_tree)
        return ""

    helper = wremnants.production.muon_calibration.make_muon_calibration_helper_single(
        filename=args.correction_file
    )
    df = wremnants.production.muon_calibration.define_lbl_corrections_jpsi_calibration_ntuples(
        df, helper
    )

    # No pileup / vertex / MC weight for data.
    df = df.Define("nominal_weight", "static_cast<float>(1.0)")

    df = _apply_reco_selection(df, args)

    df = _define_jpsi_event_vars(
        df,
        plus_pt="Mupluscor_pt",
        plus_eta="Mupluscor_eta",
        plus_phi="Mupluscor_phi",
        minus_pt="Muminuscor_pt",
        minus_eta="Muminuscor_eta",
        minus_phi="Muminuscor_phi",
    )
    df = df.Filter(f"ptll > {args.ptll_min}f", "reco_ptll")
    df = _apply_window_cut(df, args.m_lo, args.m_hi)

    df = df.Define("is_data", "static_cast<unsigned char>(1)")
    df = df.Define("source_id", f"static_cast<int>({int(args.source_id or 1)})")

    return _snapshot_to_root(
        df, out, t0,
        tree_name=args.output_tree, snapshot_format=args.snapshot_format,
    )


# ---------------------------------------------------------------------------
# Snapshot finalisation: pull columns from RDF, write Arrow IPC
# ---------------------------------------------------------------------------


def _snapshot_to_root(
    df, out_path: str, t0_start: float, *,
    tree_name: str = "tree", snapshot_format: str = "rntuple",
) -> str:
    """Materialise the OUTPUT_BRANCHES via ``df.Snapshot`` to a ROOT file.

    Single RDF action → single event loop, streaming write. Matches the
    flow/shift_reweight snapshot pattern (RNTuple + LZ4 by default).
    The sharder reads this file via uproot to bucket-shuffle rows into
    per-bucket Arrow IPC shards that the trainer's loader consumes.
    """
    import ROOT  # imported lazily — XRootD.client must come first.

    snapshot_options = ROOT.RDF.RSnapshotOptions()
    if snapshot_format == "rntuple":
        snapshot_options.fOutputFormat = ROOT.RDF.ESnapshotOutputFormat.kRNTuple
        snapshot_options.fCompressionAlgorithm = (
            ROOT.RCompressionSetting.EAlgorithm.kLZ4
        )
        snapshot_options.fCompressionLevel = 1
        fmt_label = "RNTuple LZ4"
    elif snapshot_format == "ttree":
        snapshot_options.fCompressionAlgorithm = (
            ROOT.RCompressionSetting.EAlgorithm.kZSTD
        )
        snapshot_options.fCompressionLevel = 5
        fmt_label = "TTree ZSTD"
    else:
        raise ValueError(
            f"snapshot_format must be 'rntuple' or 'ttree', got {snapshot_format!r}"
        )

    cols_vec = ROOT.std.vector("string")()
    for c in OUTPUT_BRANCHES:
        cols_vec.push_back(c)

    print(
        f"snapshotting {len(OUTPUT_BRANCHES)} per-event observables to "
        f"{fmt_label} -> {out_path}"
    )
    df.Snapshot(tree_name, out_path, cols_vec, snapshot_options)

    print(f"output file size: {os.path.getsize(out_path) / 1e6:.1f} MB")
    print(f"snapshot done in {time.time() - t0_start:.1f}s")
    return out_path


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


class _HelpFmt(
    argparse.RawDescriptionHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
):
    pass


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="jpsi_mass_fit_snapshot",
        description=__doc__,
        formatter_class=_HelpFmt,
    )
    # Subparsers optional: bare invocation re-parses as ``all`` below.
    subs = p.add_subparsers(dest="mode", required=False)

    # Default MC samples: the same J/ψ ideal-geometry samples used by
    # flow_training_snapshot.py / train_shift_smear_reweight.py.
    # flow_training_snapshot.py is stdlib-only at module level, so this
    # import is pyarrow-safe (unlike importing from wremnants.*).
    from scripts.corrections.muon_calibration.flow_training_snapshot import (
        DEFAULT_JPSI_INPUT_PATHS,
    )

    p_mc = subs.add_parser(
        "mc",
        help="snapshot J/ψ MC calibration ntuples",
        formatter_class=_HelpFmt,
    )
    p_mc.add_argument(
        "input_paths",
        nargs="*",
        default=DEFAULT_JPSI_INPUT_PATHS,
        help="One or more xrootd/posix paths or .root files (J/ψ LBL "
        "calibration ntuples). Directories are walked recursively. "
        "Defaults to the four ideal-geometry J/ψ MC samples used by "
        "flow_training_snapshot.py / train_shift_smear_reweight.py.",
    )
    p_mc.add_argument("--input-tree", default="tree", help="TTree name.")
    p_mc.add_argument("--era", default="2016PostVFP",
                      help="Used by the pileup/vertex weight helpers.")
    p_mc.add_argument("--output", default=None,
                      help="Output ROOT path "
                      "(None → jpsi_mass_fit_snapshot_mc.root in CWD).")
    p_mc.add_argument("--output-tree", default="tree",
                      help="Name of the output (R)NTuple inside the ROOT file.")
    p_mc.add_argument(
        "--snapshot-format", choices=("rntuple", "ttree"), default="rntuple",
        help="Snapshot output format. RNTuple = the modern columnar "
        "format used by the flow/shift_reweight snapshots; uproot reads "
        "both transparently in the sharder.",
    )
    p_mc.add_argument("--max-files", type=int, default=0,
                      help="Cap on number of input .root files (0 = all).")
    p_mc.add_argument("--threads", type=int, default=0,
                      help="ROOT::EnableImplicitMT thread count "
                      "(0 = ROOT default, 1 = serial).")
    p_mc.add_argument("--progress", default=True,
                      action=argparse.BooleanOptionalAction,
                      help="Show RDF progress bar during the scan (--no-progress to disable).")
    p_mc.add_argument(
        "--list-branches", action="store_true",
        help="Dump the input tree's column names (via df.GetColumnNames()) "
        "and exit. Useful for picking the right --hlt-path.",
    )
    p_mc.add_argument(
        "--correction-file", default=None,
        help="LBL/CVH correction file passed to "
        "make_muon_calibration_helper_single. Defaults to "
        "<wremnants-data>/data/calibration/correctionResults_v718_idealgeom_gensim.root "
        "(resolved at run time so wremnants isn't imported during arg-parsing — "
        "early wremnants imports pull pyarrow's OpenSSL-3 ahead of "
        "pyxrootd's OpenSSL-1.1 and segfault the snapshot).",
    )
    p_mc.add_argument(
        "--hlt-path", nargs="*", default=["HLT_Dimuon20_Jpsi"],
        help="One or more HLT branch names; joined with '||' into a "
        "single Filter expression. The OR also casts each operand to "
        "bool, so int-typed flags work. Pass ``--hlt-path`` with no "
        "values to disable; missing branch names fail loud at RDF JIT.",
    )
    p_mc.add_argument(
        "--eta-max", type=float, default=2.4,
        help="Reco-level: |Mupluscor_eta| < eta_max && |Muminuscor_eta| < eta_max.",
    )
    p_mc.add_argument(
        "--muon-pt-min-both", type=float, default=6.2,
        help="Reco-level: both muons must have pt above this threshold [GeV].",
    )
    p_mc.add_argument(
        "--muon-pt-min-leading", type=float, default=13.2,
        help="Reco-level: max(pt_+, pt_-) above this threshold [GeV].",
    )
    p_mc.add_argument(
        "--ptll-min", type=float, default=8.2,
        help="Reco-level: J/ψ pt above this threshold [GeV].",
    )
    p_mc.add_argument("--m-lo", type=float, default=2.92, dest="m_lo",
                      help="Lower edge of the J/ψ mass window [GeV].")
    p_mc.add_argument("--m-hi", type=float, default=3.28, dest="m_hi",
                      help="Upper edge of the J/ψ mass window [GeV].")
    p_mc.add_argument(
        "--source-id",
        type=int,
        default=0,
        help="int32 tag stamped on every emitted row.",
    )

    p_data = subs.add_parser(
        "data",
        help="snapshot J/ψ data calibration ntuples",
        formatter_class=_HelpFmt,
    )
    p_data.add_argument(
        "input_paths",
        nargs="*",
        default=DEFAULT_JPSI_DATA_INPUT_PATHS,
        help="One or more xrootd/posix paths or .root files (J/ψ LBL "
        "calibration ntuples for data). Directories are walked recursively. "
        "Defaults to the 2016 Charmonium Runs F-post / G / H samples.",
    )
    p_data.add_argument("--input-tree", default="tree", help="TTree name.")
    p_data.add_argument("--output", default=None,
                        help="Output ROOT path "
                        "(None → jpsi_mass_fit_snapshot_data.root in CWD).")
    p_data.add_argument("--output-tree", default="tree",
                        help="Name of the output (R)NTuple inside the ROOT file.")
    p_data.add_argument(
        "--snapshot-format", choices=("rntuple", "ttree"), default="rntuple",
        help="Snapshot output format (see mc subcommand for details).",
    )
    p_data.add_argument("--max-files", type=int, default=0,
                        help="Cap on number of input .root files (0 = all).")
    p_data.add_argument("--threads", type=int, default=0,
                        help="ROOT::EnableImplicitMT thread count "
                        "(0 = ROOT default, 1 = serial).")
    p_data.add_argument("--progress", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Show RDF progress bar during the scan (--no-progress to disable).")
    p_data.add_argument(
        "--list-branches", action="store_true",
        help="Dump the input tree's column names (via df.GetColumnNames()) "
        "and exit.",
    )
    p_data.add_argument(
        "--correction-file", default=None,
        help="LBL/CVH correction file passed to "
        "make_muon_calibration_helper_single. Defaults to "
        "<wremnants-data>/data/calibration/correctionResults_v721_recjpsidata.root "
        "(resolved at run time — see the MC arg's docstring for the "
        "pyxrootd/pyarrow rationale).",
    )
    p_data.add_argument(
        "--hlt-path", nargs="*", default=["HLT_Dimuon20_Jpsi"],
        help="One or more HLT branch names; joined with '||' into a "
        "single Filter expression. The OR also casts each operand to "
        "bool, so int-typed flags work. Pass ``--hlt-path`` with no "
        "values to disable; missing branch names fail loud at RDF JIT.",
    )
    p_data.add_argument(
        "--eta-max", type=float, default=2.4,
        help="Reco-level: |Mupluscor_eta| < eta_max && |Muminuscor_eta| < eta_max.",
    )
    p_data.add_argument(
        "--muon-pt-min-both", type=float, default=6.2,
        help="Reco-level: both muons must have pt above this threshold [GeV].",
    )
    p_data.add_argument(
        "--muon-pt-min-leading", type=float, default=13.2,
        help="Reco-level: max(pt_+, pt_-) above this threshold [GeV].",
    )
    p_data.add_argument(
        "--ptll-min", type=float, default=8.2,
        help="Reco-level: J/ψ pt above this threshold [GeV].",
    )
    p_data.add_argument("--m-lo", type=float, default=2.92, dest="m_lo",
                        help="Lower edge of the J/ψ mass window [GeV].")
    p_data.add_argument("--m-hi", type=float, default=3.28, dest="m_hi",
                        help="Upper edge of the J/ψ mass window [GeV].")
    p_data.add_argument(
        "--source-id",
        type=int,
        default=1,
        help="int32 tag stamped on every emitted row.",
    )

    p_shard = subs.add_parser(
        "shard",
        help="Bucket-shuffle MC + data ROOT snapshots into mixed Arrow "
        "shards for joint training (one shard per bucket; rows from all "
        "inputs interleaved + shuffled).",
        formatter_class=_HelpFmt,
    )
    p_shard.add_argument(
        "inputs",
        nargs="+",
        help="ROOT files produced by the mc/data subcommands. Pass MC "
        "and data together to get mixed-source shards.",
    )
    p_shard.add_argument(
        "--output", required=True,
        help="Output directory for per-bucket Arrow files "
        "(jpsi_shard_NNNN.arrow).",
    )
    p_shard.add_argument(
        "--tree-name", default="tree",
        help="Name of the input (R)NTuple inside the ROOT files.",
    )
    p_shard.add_argument(
        "--n-buckets", type=int, default=64,
        help="Bucket count → output shard count.",
    )
    p_shard.add_argument(
        "--seed", type=int, default=0,
        help="SplitMix64 seed for the bucket hash + within-bucket shuffle.",
    )
    p_shard.add_argument(
        "--no-shuffle", action="store_true",
        help="Disable within-bucket shuffle (for debugging — bucketing "
        "alone already mixes sources but preserves their relative order).",
    )

    # `all`: orchestrator that runs mc → data → shard with sensible defaults.
    p_all = subs.add_parser(
        "all",
        help="Run mc + data + shard in sequence (default when no "
        "subcommand is given).",
        formatter_class=_HelpFmt,
    )
    p_all.add_argument(
        "--output", default="runs/jpsi_v1",
        help="Output directory. Receives jpsi_mc.root, jpsi_data.root, "
        "and shards/jpsi_shard_NNNN.arrow.",
    )
    p_all.add_argument("--input-tree", default="tree",
                       help="TTree name on the LBL ntuples (same for MC + data).")
    p_all.add_argument("--output-tree", default="tree",
                       help="Name of the (R)NTuple inside the intermediate ROOT files.")
    p_all.add_argument(
        "--snapshot-format", choices=("rntuple", "ttree"), default="rntuple",
        help="Intermediate snapshot format (see mc subcommand for details).",
    )
    p_all.add_argument("--era", default="2016PostVFP",
                       help="Used by the MC pileup/vertex helpers.")
    p_all.add_argument("--max-files", type=int, default=0,
                       help="Per-source cap on input .root files (0 = all).")
    p_all.add_argument("--threads", type=int, default=0,
                       help="ROOT::EnableImplicitMT thread count "
                       "(0 = ROOT default, 1 = serial).")
    p_all.add_argument("--progress", default=True,
                       action=argparse.BooleanOptionalAction,
                       help="Show RDF progress bar for both snapshots (--no-progress to disable).")
    p_all.add_argument(
        "--mc-input-paths", nargs="*", default=DEFAULT_JPSI_INPUT_PATHS,
        help="Override MC sample paths (defaults: the four ideal-geometry "
        "J/ψ MC samples).",
    )
    p_all.add_argument(
        "--data-input-paths", nargs="*", default=DEFAULT_JPSI_DATA_INPUT_PATHS,
        help="Override data sample paths (defaults: 2016 Charmonium F-post/G/H).",
    )
    p_all.add_argument(
        "--mc-correction-file", default=None,
        help="LBL correction file for MC. Lazy default = "
        "<wremnants-data>/data/calibration/correctionResults_v718_idealgeom_gensim.root.",
    )
    p_all.add_argument(
        "--data-correction-file", default=None,
        help="LBL correction file for data. Lazy default = "
        "<wremnants-data>/data/calibration/correctionResults_v721_recjpsidata.root.",
    )
    p_all.add_argument(
        "--hlt-path", nargs="*", default=["HLT_Dimuon20_Jpsi"],
        help="HLT branch(es) — same for MC and data.",
    )
    p_all.add_argument(
        "--eta-max", type=float, default=2.4,
        help="Reco-level: |Mupluscor_eta| < eta_max && |Muminuscor_eta| < eta_max.",
    )
    p_all.add_argument(
        "--muon-pt-min-both", type=float, default=6.2,
        help="Reco-level: both muons must have pt above this threshold [GeV].",
    )
    p_all.add_argument(
        "--muon-pt-min-leading", type=float, default=13.2,
        help="Reco-level: max(pt_+, pt_-) above this threshold [GeV].",
    )
    p_all.add_argument(
        "--ptll-min", type=float, default=8.2,
        help="Reco-level: J/ψ pt above this threshold [GeV].",
    )
    p_all.add_argument(
        "--m-lo", type=float, default=2.92, dest="m_lo",
        help="Lower edge of the J/ψ mass window [GeV].",
    )
    p_all.add_argument(
        "--m-hi", type=float, default=3.28, dest="m_hi",
        help="Upper edge of the J/ψ mass window [GeV].",
    )
    p_all.add_argument("--n-buckets", type=int, default=64,
                       help="Sharder bucket count.")
    p_all.add_argument("--seed", type=int, default=0,
                       help="SplitMix64 seed for the bucket hash + shuffle.")
    p_all.add_argument(
        "--skip-mc", action="store_true",
        help="Skip the MC snapshot step (use existing jpsi_mc.root).",
    )
    p_all.add_argument(
        "--skip-data", action="store_true",
        help="Skip the data snapshot step (use existing jpsi_data.root).",
    )
    p_all.add_argument(
        "--skip-shard", action="store_true",
        help="Skip the sharder step.",
    )

    # Insert the implicit ``all`` subcommand when none was given, so
    # ``python jpsi_mass_fit_snapshot.py [--flags...]`` dispatches to
    # the orchestrator with the flags forwarded. ``--help`` /``-h``
    # alone is preserved as a request for the top-level help.
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    _SUBCOMMANDS = {"mc", "data", "shard", "all"}
    _has_sub = any(a in _SUBCOMMANDS for a in argv if not a.startswith("-"))
    _just_help = bool(argv) and set(argv).issubset({"-h", "--help"})
    if not _has_sub and not _just_help:
        argv = ["all"] + argv
    return p.parse_args(argv)


def _run_shard_step(inputs, output_dir, *, n_buckets, seed, tree_name="tree"):
    """Dispatch helper — defers the pyarrow/uproot-heavy sharder import."""
    from scripts.corrections.muon_calibration.jpsi_mass_sharder import (
        shard_jpsi_mass_inputs,
    )
    shard_jpsi_mass_inputs(
        inputs, output_dir,
        n_buckets=n_buckets, seed=seed, shuffle_within_bucket=True,
        tree_name=tree_name,
    )


def run_all_pipeline(args) -> int:
    """End-to-end: snapshot MC → snapshot data → bucket-shuffle shard.

    Builds per-stage Namespace objects whose defaults mirror the
    ``mc``/``data``/``shard`` subparser defaults. If the per-mode
    defaults change, update this function in lockstep — the per-mode
    subparsers' values are the source of truth.
    """
    import argparse as _ap

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    mc_root = os.path.join(out_dir, "jpsi_mc.root")
    data_root = os.path.join(out_dir, "jpsi_data.root")
    shard_dir = os.path.join(out_dir, "shards")

    # Common kinematic-cut / window args lifted from the orchestrator's CLI.
    cut_kw = dict(
        input_tree=args.input_tree,
        output_tree=args.output_tree,
        snapshot_format=args.snapshot_format,
        max_files=args.max_files,
        threads=args.threads,
        progress=args.progress,
        list_branches=False,
        hlt_path=list(args.hlt_path),
        eta_max=args.eta_max,
        muon_pt_min_both=args.muon_pt_min_both,
        muon_pt_min_leading=args.muon_pt_min_leading,
        ptll_min=args.ptll_min,
        m_lo=args.m_lo,
        m_hi=args.m_hi,
    )

    if not args.skip_mc:
        print(f"\n=== [1/3] MC snapshot → {mc_root} ===")
        mc_args = _ap.Namespace(
            mode="mc",
            input_paths=list(args.mc_input_paths),
            era=args.era,
            output=mc_root,
            correction_file=args.mc_correction_file,
            source_id=0,
            **cut_kw,
        )
        run_jpsi_mass_snapshot_mc(mc_args)
    else:
        print(f"\n=== [1/3] MC snapshot SKIPPED (reusing {mc_root}) ===")

    if not args.skip_data:
        print(f"\n=== [2/3] data snapshot → {data_root} ===")
        data_args = _ap.Namespace(
            mode="data",
            input_paths=list(args.data_input_paths),
            output=data_root,
            correction_file=args.data_correction_file,
            source_id=1,
            **cut_kw,
        )
        run_jpsi_mass_snapshot_data(data_args)
    else:
        print(f"\n=== [2/3] data snapshot SKIPPED (reusing {data_root}) ===")

    if not args.skip_shard:
        print(f"\n=== [3/3] shard → {shard_dir} ===")
        _run_shard_step(
            [mc_root, data_root], shard_dir,
            n_buckets=args.n_buckets, seed=args.seed,
            tree_name=args.output_tree,
        )
    else:
        print("\n=== [3/3] shard SKIPPED ===")

    print(f"\n=== pipeline done; shards under {shard_dir} ===")
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "mc":
        run_jpsi_mass_snapshot_mc(args)
    elif args.mode == "data":
        run_jpsi_mass_snapshot_data(args)
    elif args.mode == "shard":
        _run_shard_step(
            args.inputs, args.output,
            n_buckets=args.n_buckets, seed=args.seed,
            tree_name=args.tree_name,
        )
    elif args.mode == "all":
        return run_all_pipeline(args)
    else:
        raise SystemExit(f"unknown mode {args.mode!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
