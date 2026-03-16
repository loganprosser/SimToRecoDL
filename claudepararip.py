#!/usr/bin/env python3
"""
Parallel single-file ROOT → numpy cache builder.

Reads one ROOT file using multiple workers (each opens its own TFile handle),
extracts per-sim-track features and regression targets, applies optional cuts,
and writes a compact .npz.

Usage:
    python root_to_cache.py [path/to/file.root]

Memory strategy:
  - Each worker reads a slice of tree entries and returns only the kept rows
    as small numpy arrays (typically a few MB each).
  - Workers are capped at N_WORKERS to avoid opening too many TFile handles.
  - The main process concatenates results once at the end.
  - Peak RAM ≈ (N_WORKERS × per-chunk data) + final concat.
    For a typical RelVal file this is well under 1 GB.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────

ROOT_FILE = (
    "/data2/segmentlinking/CMSSW_12_5_0_pre3/"
    "RelValTTbar_14TeV_CMSSW_12_5_0_pre3/event_1000.root"
)
TREE_PATH = "trackingNtuple/tree"
OUTDIR = "./track_cache"

# Workers: each opens its own TFile, so memory scales with this.
# 32 is a good sweet spot on a 92-core box — ROOT I/O is the bottleneck,
# not CPU, and 32 concurrent file handles is plenty.  Tune up if your
# storage is very fast (NVMe RAID) or down if RAM is tight.
N_WORKERS = 32

# Entries per worker chunk.  Smaller = better load balance but more overhead.
# None = auto (total_entries / (N_WORKERS * 4), clamped to [50, 2000]).
CHUNK_SIZE = None

# Per-sim-track input features the network will see
FEATURE_BRANCHES = [
    "sim_px", "sim_py", "sim_pz", "sim_pt",
    "sim_eta", "sim_phi", "sim_q",
    "sim_nValid", "sim_nPixel", "sim_nStrip",
    "sim_nLay", "sim_nPixelLay", "sim_n3DLay",
    "sim_nTrackerHits", "sim_nRecoClusters",
]

# Branches needed to build regression targets
TARGET_SOURCE_BRANCHES = [
    "sim_q", "sim_pca_pt", "sim_pca_eta",
    "sim_pca_lambda", "sim_pca_phi",
    "sim_pca_dxy", "sim_pca_dz",
]

# "qoverpt_lambda_phi_dxy_dz" | "pt_eta_phi_dxy_dz" | "pt_lambda_phi_dxy_dz"
TARGET_MODE = "qoverpt_lambda_phi_dxy_dz"

# Optional quality cuts (set to None to disable)
MIN_SIM_PT: Optional[float] = None
MAX_ABS_SIM_ETA: Optional[float] = None
DROP_NONFINITE = True


# ── Helpers (must be top-level for pickling) ─────────────────────────────────

def _vec_to_f32(vec) -> np.ndarray:
    """Convert a ROOT std::vector to a float32 numpy array."""
    n = vec.size()
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = vec[i]
    return out


def _find_tree(tfile, path: str):
    """Try direct Get, then recursive search."""
    obj = tfile.Get(path)
    if obj and obj.InheritsFrom("TTree"):
        return obj

    target = path.split("/")[-1]

    def _recurse(d):
        keys = d.GetListOfKeys()
        if not keys:
            return None
        for k in keys:
            child = d.Get(k.GetName())
            if not child:
                continue
            if child.GetName() == target and child.InheritsFrom("TTree"):
                return child
            if child.InheritsFrom("TDirectory"):
                found = _recurse(child)
                if found:
                    return found
        return None

    return _recurse(tfile)


def _build_targets(src: dict, mode: str) -> Tuple[list, np.ndarray]:
    """Build (n_sim, n_targets) target array from source branch dict."""
    q   = src["sim_q"].astype(np.float32)
    pt  = src["sim_pca_pt"]
    lam = src["sim_pca_lambda"]
    phi = src["sim_pca_phi"]
    dxy = src["sim_pca_dxy"]
    dz  = src["sim_pca_dz"]

    if mode == "qoverpt_lambda_phi_dxy_dz":
        with np.errstate(divide="ignore", invalid="ignore"):
            qoverpt = q / pt
        cols = ["target_qoverpt", "target_lambda", "target_phi", "target_dxy", "target_dz"]
        return cols, np.column_stack([qoverpt, lam, phi, dxy, dz])

    eta = src["sim_pca_eta"]
    if mode == "pt_eta_phi_dxy_dz":
        cols = ["target_pt", "target_eta", "target_phi", "target_dxy", "target_dz"]
        return cols, np.column_stack([pt, eta, phi, dxy, dz])

    if mode == "pt_lambda_phi_dxy_dz":
        cols = ["target_pt", "target_lambda", "target_phi", "target_dxy", "target_dz"]
        return cols, np.column_stack([pt, lam, phi, dxy, dz])

    raise ValueError(f"Unknown TARGET_MODE: {mode}")


def _get_label_cols(mode: str) -> list:
    """Return target column names without needing data."""
    if mode == "qoverpt_lambda_phi_dxy_dz":
        return ["target_qoverpt", "target_lambda", "target_phi", "target_dxy", "target_dz"]
    if mode == "pt_eta_phi_dxy_dz":
        return ["target_pt", "target_eta", "target_phi", "target_dxy", "target_dz"]
    if mode == "pt_lambda_phi_dxy_dz":
        return ["target_pt", "target_lambda", "target_phi", "target_dxy", "target_dz"]
    raise ValueError(f"Unknown TARGET_MODE: {mode}")


# ── Worker function ──────────────────────────────────────────────────────────

def _process_chunk(args: Tuple) -> dict:
    """
    Worker: opens its own TFile, reads entries [start, end), returns numpy arrays.

    Returns dict with keys: n_entries, n_sim, n_kept, X, Y
    X and Y are float32 arrays of just the kept rows (small).
    """
    import ROOT as _ROOT  # each spawned worker needs its own import

    (root_path, tree_path, start, end,
     feature_branches, target_source_branches, target_mode,
     drop_nonfinite, min_sim_pt, max_abs_sim_eta) = args

    tfile = _ROOT.TFile.Open(root_path)
    if not tfile or tfile.IsZombie():
        raise RuntimeError(f"Worker cannot open: {root_path}")

    tree = _find_tree(tfile, tree_path)
    if tree is None:
        tfile.Close()
        raise RuntimeError(f"TTree '{tree_path}' not found in {root_path}")

    all_branches = set(feature_branches) | set(target_source_branches)

    # Activate only what we need
    tree.SetBranchStatus("*", 0)
    for b in all_branches:
        tree.SetBranchStatus(b, 1)

    n_feat = len(feature_branches)
    n_lab = len(_get_label_cols(target_mode))

    feat_chunks = []
    label_chunks = []
    total_sim = 0

    for i_entry in range(start, end):
        tree.GetEntry(i_entry)

        # Read branches
        src = {}
        n_sim = None
        for b in all_branches:
            vec = getattr(tree, b)
            arr = _vec_to_f32(vec)
            src[b] = arr
            if n_sim is None:
                n_sim = len(arr)

        if n_sim == 0:
            continue
        total_sim += n_sim

        feat_matrix = np.column_stack([src[b] for b in feature_branches])
        _, label_matrix = _build_targets(src, target_mode)

        # Vectorized cuts
        keep = np.ones(n_sim, dtype=bool)

        if min_sim_pt is not None:
            keep &= src["sim_pt"] >= min_sim_pt

        if max_abs_sim_eta is not None:
            keep &= np.abs(src["sim_eta"]) <= max_abs_sim_eta

        if drop_nonfinite:
            keep &= (
                np.isfinite(feat_matrix).all(axis=1)
                & np.isfinite(label_matrix).all(axis=1)
            )

        if keep.any():
            feat_chunks.append(feat_matrix[keep])
            label_chunks.append(label_matrix[keep])

    tfile.Close()

    if feat_chunks:
        X = np.concatenate(feat_chunks, axis=0)
        Y = np.concatenate(label_chunks, axis=0)
    else:
        X = np.empty((0, n_feat), dtype=np.float32)
        Y = np.empty((0, n_lab), dtype=np.float32)

    return {
        "n_entries": end - start,
        "n_sim": total_sim,
        "n_kept": X.shape[0],
        "X": X,
        "Y": Y,
    }


# ── Orchestrator ─────────────────────────────────────────────────────────────

def get_n_entries(root_path: str, tree_path: str) -> int:
    """Quick open just to get entry count."""
    import ROOT as _ROOT
    tfile = _ROOT.TFile.Open(root_path)
    if not tfile or tfile.IsZombie():
        raise RuntimeError(f"Cannot open: {root_path}")
    tree = _find_tree(tfile, tree_path)
    if tree is None:
        tfile.Close()
        raise RuntimeError(f"TTree '{tree_path}' not found")
    n = int(tree.GetEntries())
    tfile.Close()
    return n


def extract_parallel(root_path: str) -> Tuple[np.ndarray, np.ndarray, list, list, dict]:
    n_entries = get_n_entries(root_path, TREE_PATH)
    if n_entries == 0:
        label_cols = _get_label_cols(TARGET_MODE)
        return (np.empty((0, len(FEATURE_BRANCHES)), dtype=np.float32),
                np.empty((0, len(label_cols)), dtype=np.float32),
                list(FEATURE_BRANCHES), label_cols, {"n_entries": 0})

    # Auto chunk size: aim for ~4 chunks per worker for good load balance
    if CHUNK_SIZE is not None:
        chunk = int(CHUNK_SIZE)
    else:
        chunk = max(50, min(2000, n_entries // (N_WORKERS * 4)))

    # Build chunk ranges
    ranges = []
    for s in range(0, n_entries, chunk):
        ranges.append((s, min(s + chunk, n_entries)))

    n_chunks = len(ranges)
    print(f"  {n_entries:,} entries → {n_chunks} chunks of ~{chunk} entries, {N_WORKERS} workers")

    # Build worker args (all picklable plain types)
    worker_args = [
        (root_path, TREE_PATH, s, e,
         tuple(FEATURE_BRANCHES), tuple(TARGET_SOURCE_BRANCHES), TARGET_MODE,
         DROP_NONFINITE, MIN_SIM_PT, MAX_ABS_SIM_ETA)
        for s, e in ranges
    ]

    # Run with process pool — use 'spawn' so ROOT state doesn't leak across fork
    t0 = time.time()
    results = []
    done = 0
    total_kept = 0
    total_sim = 0
    progress_interval = max(1, n_chunks // 20)

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=N_WORKERS) as pool:
        for result in pool.imap_unordered(_process_chunk, worker_args):
            results.append(result)
            done += 1
            total_kept += result["n_kept"]
            total_sim += result["n_sim"]

            if done % progress_interval == 0 or done == n_chunks:
                elapsed = time.time() - t0
                entries_done = sum(r["n_entries"] for r in results)
                rate = entries_done / elapsed if elapsed > 0 else 0
                print(
                    f"\r  chunks {done}/{n_chunks} | "
                    f"tracks seen: {total_sim:,} | "
                    f"kept: {total_kept:,} | "
                    f"{rate:,.0f} entries/s | "
                    f"{elapsed:.1f}s elapsed",
                    end="", flush=True,
                )

    print()
    elapsed = time.time() - t0

    # Concatenate — each result["X"] is just the kept rows, typically small
    label_cols = _get_label_cols(TARGET_MODE)

    Xs = [r["X"] for r in results if r["n_kept"] > 0]
    Ys = [r["Y"] for r in results if r["n_kept"] > 0]
    del results  # free before concat

    if Xs:
        X = np.concatenate(Xs, axis=0)
        Y = np.concatenate(Ys, axis=0)
        del Xs, Ys
    else:
        X = np.empty((0, len(FEATURE_BRANCHES)), dtype=np.float32)
        Y = np.empty((0, len(label_cols)), dtype=np.float32)

    stats = {
        "n_entries": n_entries,
        "n_total_sim_tracks": total_sim,
        "n_kept": int(X.shape[0]),
        "n_workers": N_WORKERS,
        "n_chunks": n_chunks,
        "chunk_size": chunk,
        "elapsed_sec": round(elapsed, 2),
    }

    return X, Y, list(FEATURE_BRANCHES), label_cols, stats


# ── Save ─────────────────────────────────────────────────────────────────────

def save(X: np.ndarray, Y: np.ndarray,
         feature_cols: list, label_cols: list) -> None:
    """
    Save a single .npz with the matched feature/target arrays.

    Loading:
        d = np.load("track_cache/track_data.npz")
        X, Y = d["X"], d["Y"]
        print(d["feature_columns"])
        print(d["label_columns"])
    """
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / "track_data.npz"

    np.savez_compressed(
        npz_path,
        X=X,
        Y=Y,
        feature_columns=np.array(feature_cols, dtype=str),
        label_columns=np.array(label_cols, dtype=str),
    )

    size_mb = npz_path.stat().st_size / 1e6
    print(f"Saved → {npz_path}  ({size_mb:.1f} MB compressed)")
    print(f"  X: {X.shape}  ({', '.join(feature_cols)})")
    print(f"  Y: {Y.shape}  ({', '.join(label_cols)})")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    root_path = sys.argv[1] if len(sys.argv) > 1 else ROOT_FILE

    print(f"ROOT file : {root_path}")
    print(f"Tree      : {TREE_PATH}")
    print(f"Targets   : {TARGET_MODE}")
    print(f"Features  : {len(FEATURE_BRANCHES)} branches")
    print(f"Workers   : {N_WORKERS}")
    print(f"Output dir: {OUTDIR}\n")

    X, Y, feat_cols, label_cols, stats = extract_parallel(root_path)

    print(f"\nExtraction stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if X.shape[0] == 0:
        print("\nNo rows survived filtering — nothing to save.")
        return

    save(X, Y, feat_cols, label_cols)
    print("\nDone.")


if __name__ == "__main__":
    main()