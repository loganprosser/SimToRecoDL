#!/usr/bin/env python3
"""
Build a row-aligned track cache from all ROOT files in one directory.

Edit the CONSTANTS block below, then run:
    python root_to_track_cache_constants.py

This script:
- finds every .root file in INPUT_DIR
- reads one tree from each file
- matches reco tracks to sim tracks using the configured match branches
- stores a row-wise cache (parquet if available, otherwise HDF5)
- stores PyTorch-ready feature/label arrays as .npy files

Design goal:
- keep configuration in one place
- keep ROOT extraction separate from later ML work
- allow parallel processing across files and within large files via chunking
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import ROOT  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "PyROOT is required for this script. Load your ROOT environment first."
    ) from exc


# =========================
# CONSTANTS: EDIT THESE
# =========================
INPUT_DIR = "."                       # directory containing your ROOT files
FILE_GLOB = "*.root"                  # which ROOT files to include
TREE_PATH = "trackingNtuple/tree"     # full tree path inside each ROOT file
OUTDIR = "./track_cache"              # where cache + arrays will be written

# Parallelism
N_WORKERS = max(1, (os.cpu_count() or 1) // 2)
CHUNK_SIZE = 200                       # entries per job inside each file

# Matching
MATCH_INDEX_BRANCH = "trk_bestSimTrkIdx"
MATCH_SHARE_BRANCH = "trk_bestSimTrkShareFrac"
MIN_SHARE_FRAC = 0.75
KEEP_UNMATCHED = False

# Columns to save
RECO_BRANCHES = [
    "trk_pt",
    "trk_eta",
    "trk_phi",
]

SIM_BRANCHES = [
    "sim_pt",
    "sim_eta",
    "sim_phi",
    "sim_qoverp",
    "sim_dxy",
]

# Output behavior
SORT_BY = ["source_file", "event_id", "reco_idx"]
PREFER_PARQUET = True


# ---------- ROOT helpers ----------

def find_object_in_file(root_file, path: str):
    obj = root_file.Get(path)
    if obj:
        return obj

    target_name = path.split("/")[-1]

    def _search_dir(d):
        keys = d.GetListOfKeys()
        if not keys:
            return None
        for key in keys:
            name = key.GetName()
            child = d.Get(name)
            if not child:
                continue
            if child.GetName() == target_name:
                return child
            if child.InheritsFrom("TDirectory"):
                found = _search_dir(child)
                if found:
                    return found
        return None

    return _search_dir(root_file)


def open_tree(root_path: str, tree_path: str):
    f = ROOT.TFile.Open(root_path)
    if not f or f.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {root_path}")

    tree = find_object_in_file(f, tree_path)
    if not tree or not tree.InheritsFrom("TTree"):
        top_keys = []
        lk = f.GetListOfKeys()
        if lk:
            for k in lk:
                top_keys.append(k.GetName())
        f.Close()
        raise RuntimeError(
            f"Could not find TTree '{tree_path}' in {root_path}. Top-level keys: {top_keys[:30]}"
        )
    return f, tree


def branch_to_numpy_float(x) -> np.ndarray:
    n = x.size()
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = float(x[i])
    return out


def branch_to_numpy_int(x) -> np.ndarray:
    n = x.size()
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        out[i] = int(x[i])
    return out


# ---------- Worker logic ----------

def process_chunk(args: Tuple) -> Dict[str, np.ndarray]:
    (
        root_path,
        tree_path,
        start,
        end,
        reco_branches,
        sim_branches,
        match_index_branch,
        match_share_branch,
        min_share_frac,
        keep_unmatched,
    ) = args

    f, tree = open_tree(root_path, tree_path)

    rows: Dict[str, List[object]] = {
        "source_file": [],
        "entry": [],
        "event_id": [],
        "reco_idx": [],
        "sim_idx": [],
        "match_share_frac": [],
        "is_matched": [],
    }
    for b in reco_branches:
        rows[f"reco__{b}"] = []
    for b in sim_branches:
        rows[f"sim__{b}"] = []

    for entry_idx in range(start, end):
        tree.GetEntry(entry_idx)

        try:
            match_idx = branch_to_numpy_int(getattr(tree, match_index_branch))
            share_frac = branch_to_numpy_float(getattr(tree, match_share_branch))
        except AttributeError as exc:
            f.Close()
            raise RuntimeError(
                f"Missing matching branch in {root_path}: {exc}. "
                f"Expected '{match_index_branch}' and '{match_share_branch}'."
            ) from exc

        reco_payload = {}
        for b in reco_branches:
            try:
                reco_payload[b] = branch_to_numpy_float(getattr(tree, b))
            except AttributeError as exc:
                f.Close()
                raise RuntimeError(f"Missing reco branch '{b}' in {root_path}") from exc

        sim_payload = {}
        for b in sim_branches:
            try:
                sim_payload[b] = branch_to_numpy_float(getattr(tree, b))
            except AttributeError as exc:
                f.Close()
                raise RuntimeError(f"Missing sim branch '{b}' in {root_path}") from exc

        n_reco = len(match_idx)
        if n_reco == 0:
            continue

        event_id = entry_idx
        for candidate in ("event", "event_id", "evt"):
            if hasattr(tree, candidate):
                try:
                    event_id = int(getattr(tree, candidate))
                    break
                except Exception:
                    pass

        for i_reco in range(n_reco):
            j_sim = int(match_idx[i_reco])
            frac = float(share_frac[i_reco])
            matched = (j_sim >= 0) and (frac >= min_share_frac)

            if (not matched) and (not keep_unmatched):
                continue

            rows["source_file"].append(str(root_path))
            rows["entry"].append(int(entry_idx))
            rows["event_id"].append(int(event_id))
            rows["reco_idx"].append(int(i_reco))
            rows["sim_idx"].append(int(j_sim if matched else -1))
            rows["match_share_frac"].append(frac)
            rows["is_matched"].append(int(matched))

            for b in reco_branches:
                rows[f"reco__{b}"].append(float(reco_payload[b][i_reco]))

            for b in sim_branches:
                if matched and 0 <= j_sim < len(sim_payload[b]):
                    rows[f"sim__{b}"].append(float(sim_payload[b][j_sim]))
                else:
                    rows[f"sim__{b}"].append(np.nan)

    f.Close()

    out = {"source_file": np.asarray(rows["source_file"], dtype=object)}
    for key, values in rows.items():
        if key == "source_file":
            continue
        if key in {"entry", "event_id", "reco_idx", "sim_idx", "is_matched"}:
            out[key] = np.asarray(values, dtype=np.int64)
        else:
            out[key] = np.asarray(values, dtype=np.float32)
    return out


# ---------- Merge / save ----------

def concat_chunks(chunks: Sequence[Dict[str, np.ndarray]]) -> pd.DataFrame:
    if not chunks:
        return pd.DataFrame()
    keys = list(chunks[0].keys())
    merged = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in keys}
    return pd.DataFrame(merged)


def save_row_cache(df: pd.DataFrame, outdir: Path, prefer_parquet: bool) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    if prefer_parquet:
        try:
            path = outdir / "track_rows.parquet"
            df.to_parquet(path, index=False)
            return path
        except Exception:
            pass

    path = outdir / "track_rows.h5"
    df.to_hdf(path, key="tracks", mode="w", format="table")
    return path


def build_xy(df: pd.DataFrame, reco_branches: Sequence[str], sim_branches: Sequence[str]):
    feature_cols = [f"reco__{b}" for b in reco_branches]
    label_cols = [f"sim__{b}" for b in sim_branches]

    if df.empty:
        return (
            np.empty((0, len(feature_cols)), dtype=np.float32),
            np.empty((0, len(label_cols)), dtype=np.float32),
            feature_cols,
            label_cols,
        )

    xy = df[feature_cols + label_cols].copy()
    finite_mask = np.isfinite(xy.to_numpy(dtype=np.float32)).all(axis=1)
    xy = xy.loc[finite_mask]

    X = xy[feature_cols].to_numpy(dtype=np.float32)
    Y = xy[label_cols].to_numpy(dtype=np.float32)
    return X, Y, feature_cols, label_cols


# ---------- Job planning ----------

def discover_root_files(input_dir: str, file_glob: str) -> List[str]:
    files = sorted(str(p.resolve()) for p in Path(input_dir).glob(file_glob) if p.is_file())
    return files


def make_jobs(
    input_files: Sequence[str],
    tree_path: str,
    reco_branches: Sequence[str],
    sim_branches: Sequence[str],
    match_index_branch: str,
    match_share_branch: str,
    min_share_frac: float,
    keep_unmatched: bool,
    chunk_size: int,
) -> List[Tuple]:
    jobs: List[Tuple] = []
    for root_path in input_files:
        f, tree = open_tree(root_path, tree_path)
        n_entries = int(tree.GetEntries())
        f.Close()

        for start in range(0, n_entries, chunk_size):
            end = min(start + chunk_size, n_entries)
            jobs.append(
                (
                    root_path,
                    tree_path,
                    start,
                    end,
                    tuple(reco_branches),
                    tuple(sim_branches),
                    match_index_branch,
                    match_share_branch,
                    float(min_share_frac),
                    bool(keep_unmatched),
                )
            )
    return jobs


def main() -> None:
    input_files = discover_root_files(INPUT_DIR, FILE_GLOB)
    if not input_files:
        raise RuntimeError(f"No ROOT files matched {FILE_GLOB!r} in directory {INPUT_DIR!r}")

    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    jobs = make_jobs(
        input_files=input_files,
        tree_path=TREE_PATH,
        reco_branches=RECO_BRANCHES,
        sim_branches=SIM_BRANCHES,
        match_index_branch=MATCH_INDEX_BRANCH,
        match_share_branch=MATCH_SHARE_BRANCH,
        min_share_frac=MIN_SHARE_FRAC,
        keep_unmatched=KEEP_UNMATCHED,
        chunk_size=CHUNK_SIZE,
    )

    print(f"Found {len(input_files)} ROOT file(s) in {Path(INPUT_DIR).resolve()}")
    print(f"Planned {len(jobs)} job(s) with {N_WORKERS} worker(s)")
    print("Files:")
    for f in input_files:
        print(f"  - {f}")

    chunks = []
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(process_chunk, job) for job in jobs]
        for fut in as_completed(futures):
            chunks.append(fut.result())
            done += 1
            if done % max(1, len(jobs) // 20) == 0 or done == len(jobs):
                print(f"Finished {done}/{len(jobs)} jobs")

    df = concat_chunks(chunks)
    if df.empty:
        print("No rows were extracted.")
        return

    sort_cols = [c for c in SORT_BY if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    row_cache_path = save_row_cache(df, outdir=outdir, prefer_parquet=PREFER_PARQUET)
    print(f"Saved row cache to {row_cache_path}")

    matched_df = df[df["is_matched"] == 1].copy()
    X, Y, feature_cols, label_cols = build_xy(
        matched_df,
        reco_branches=RECO_BRANCHES,
        sim_branches=SIM_BRANCHES,
    )

    np.save(outdir / "features.npy", X)
    np.save(outdir / "labels.npy", Y)

    metadata = {
        "input_dir": str(Path(INPUT_DIR).resolve()),
        "file_glob": FILE_GLOB,
        "input_files": input_files,
        "tree": TREE_PATH,
        "n_files": len(input_files),
        "n_rows_total": int(len(df)),
        "n_rows_matched": int((df["is_matched"] == 1).sum()),
        "n_features": int(X.shape[1]),
        "n_labels": int(Y.shape[1]),
        "feature_columns": feature_cols,
        "label_columns": label_cols,
        "reco_branches": list(RECO_BRANCHES),
        "sim_branches": list(SIM_BRANCHES),
        "min_share_frac": float(MIN_SHARE_FRAC),
        "match_index_branch": MATCH_INDEX_BRANCH,
        "match_share_branch": MATCH_SHARE_BRANCH,
        "chunk_size": int(CHUNK_SIZE),
        "n_workers": int(N_WORKERS),
        "keep_unmatched": bool(KEEP_UNMATCHED),
    }
    with open(outdir / "column_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved X to {outdir / 'features.npy'} with shape {X.shape}")
    print(f"Saved Y to {outdir / 'labels.npy'} with shape {Y.shape}")
    print(f"Saved metadata to {outdir / 'column_metadata.json'}")


if __name__ == "__main__":
    main()
