#!/usr/bin/env python3
"""
Filter track cache for simple feed-forward network training.

Loads track_data.npz from the cache builder, applies:
  1. eta cut:  |sim_eta| <= ETA_MAX  (default 1.0)
  2. completeness check: drop any rows that have zeros in ALL hit-count
     columns (nValid, nPixel, nStrip, etc.) — these are tracks with no
     actual detector hits, not useful for training.

Prints a diagnostic summary of what's in the data before and after filtering,
then saves a clean .npz ready for a simple dense network.

Usage:
    python filter_tracks.py [path/to/track_data.npz]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# ── Configuration ────────────────────────────────────────────────────────────

INPUT_NPZ = "./track_cache/track_data.npz"
OUTPUT_NPZ = "./track_cache/track_data_filtered.npz"

# Eta cut: keep only |sim_eta| <= ETA_MAX
ETA_MAX = 1.0

# Hit-count columns — if ALL of these are zero for a track, it has no real
# detector info and we drop it.  Set to empty list to skip this cut.
HIT_COUNT_FEATURES = [
    "sim_nValid", "sim_nPixel", "sim_nStrip",
    "sim_nLay", "sim_nPixelLay", "sim_n3DLay",
    "sim_nTrackerHits", "sim_nRecoClusters",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def print_summary(X: np.ndarray, Y: np.ndarray,
                  feat_cols: list[str], label_cols: list[str],
                  title: str) -> None:
    """Print a quick statistical overview of the dataset."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"  {X.shape[0]:,} samples × {X.shape[1]} features → {Y.shape[1]} targets")
    print(f"{'=' * 60}")

    print(f"\n  Features:")
    for i, name in enumerate(feat_cols):
        col = X[:, i]
        print(
            f"    {name:>22s}  "
            f"min={col.min():12.4f}  "
            f"max={col.max():12.4f}  "
            f"mean={col.mean():12.4f}  "
            f"std={col.std():10.4f}  "
            f"nans={np.isnan(col).sum()}"
        )

    print(f"\n  Targets:")
    for i, name in enumerate(label_cols):
        col = Y[:, i]
        print(
            f"    {name:>22s}  "
            f"min={col.min():12.4f}  "
            f"max={col.max():12.4f}  "
            f"mean={col.mean():12.4f}  "
            f"std={col.std():10.4f}  "
            f"nans={np.isnan(col).sum()}"
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_NPZ

    print(f"Loading {input_path} ...")
    d = np.load(input_path, allow_pickle=False)
    X = d["X"]
    Y = d["Y"]
    feat_cols = list(d["feature_columns"])
    label_cols = list(d["label_columns"])

    n_before = X.shape[0]
    print(f"Loaded {n_before:,} samples")

    # Show raw data summary
    print_summary(X, Y, feat_cols, label_cols, "RAW DATA (before filtering)")

    # ── Build feature column index lookup ────────────────────────
    col_idx = {name: i for i, name in enumerate(feat_cols)}

    # ── Cut 1: |sim_eta| <= ETA_MAX ──────────────────────────────
    if "sim_eta" not in col_idx:
        print(f"\nWARNING: 'sim_eta' not in features, skipping eta cut")
        keep = np.ones(n_before, dtype=bool)
    else:
        eta = X[:, col_idx["sim_eta"]]
        keep = np.abs(eta) <= ETA_MAX
        n_cut_eta = int((~keep).sum())
        print(f"\nEta cut |sim_eta| <= {ETA_MAX}: dropping {n_cut_eta:,} samples")

    # ── Cut 2: drop tracks with zero in ALL hit-count columns ────
    hit_indices = [col_idx[c] for c in HIT_COUNT_FEATURES if c in col_idx]
    if hit_indices:
        hit_cols = X[:, hit_indices]
        has_hits = (hit_cols != 0).any(axis=1)
        n_cut_hits = int((~has_hits & keep).sum())
        keep &= has_hits
        print(f"Hit completeness cut: dropping {n_cut_hits:,} samples with all-zero hit counts")
    else:
        print("No hit-count columns found, skipping completeness cut")

    # ── Apply ────────────────────────────────────────────────────
    X_out = X[keep]
    Y_out = Y[keep]

    n_after = X_out.shape[0]
    n_dropped = n_before - n_after
    print(f"\nKept {n_after:,} / {n_before:,} samples ({100*n_after/n_before:.1f}%), dropped {n_dropped:,}")

    if n_after == 0:
        print("Nothing left after filtering — check your cuts.")
        return

    # Show filtered data summary
    print_summary(X_out, Y_out, feat_cols, label_cols, "FILTERED DATA")

    # ── Save ─────────────────────────────────────────────────────
    out_path = Path(OUTPUT_NPZ)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        X=X_out,
        Y=Y_out,
        feature_columns=np.array(feat_cols, dtype=str),
        label_columns=np.array(label_cols, dtype=str),
    )

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved → {out_path}  ({size_mb:.1f} MB)")
    print(f"  X: {X_out.shape}  Y: {Y_out.shape}")

    # Quick sanity check for training readiness
    print(f"\n{'=' * 60}")
    print(f"  TRAINING READINESS CHECK")
    print(f"{'=' * 60}")
    print(f"  Fixed input size:  {X_out.shape[1]} features per sample ✓")
    print(f"  Fixed output size: {Y_out.shape[1]} targets per sample ✓")
    print(f"  NaN in features:   {np.isnan(X_out).sum()}")
    print(f"  NaN in targets:    {np.isnan(Y_out).sum()}")
    print(f"  Inf in features:   {np.isinf(X_out).sum()}")
    print(f"  Inf in targets:    {np.isinf(Y_out).sum()}")
    print(f"  dtype:             {X_out.dtype}")
    print(f"\n  Load in your training script:")
    print(f"    d = np.load('{out_path}')")
    print(f"    X, Y = d['X'], d['Y']")


if __name__ == "__main__":
    main()