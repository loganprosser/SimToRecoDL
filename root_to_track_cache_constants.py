#!/usr/bin/env python3
"""
Build a row-aligned SIM-track cache from all ROOT files in one directory.

Edit the CONSTANTS block below, then run:
    python root_to_track_cache_constants.py

This script:
- finds every .root file in INPUT_DIR
- reads one tree from each file
- loops over SIM tracks only (no reco matching)
- stores a row-wise cache (parquet if available, otherwise HDF5)
- stores PyTorch-ready feature/label arrays as .npy files

Design goal:
- keep configuration in one place
- keep ROOT extraction separate from later ML work
- allow parallel processing across files and within large files via chunking
"""

from __future__ import annotations

import json
import math
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
INPUT_DIR = "/data2/segmentlinking/CMSSW_12_5_0_pre3/RelValTTbar_14TeV_CMSSW_12_5_0_pre3"                       # directory containing your ROOT files
FILE_GLOB = "*.root"                  # which ROOT files to include
TREE_PATH = "trackingNtuple/tree"     # full tree path inside each ROOT file
OUTDIR = "./track_cache"              # where cache + arrays will be written

# Parallelism
N_WORKERS = 16
CHUNK_SIZE = 1000                     # entries per job inside each file

# Input columns: these are the per-sim-track features the network will see
INPUT_BRANCHES = [
    "sim_px",
    "sim_py",
    "sim_pz",
    "sim_pt",
    "sim_eta",
    "sim_phi",
    "sim_q",
    "sim_nValid",
    "sim_nPixel",
    "sim_nStrip",
    "sim_nLay",
    "sim_nPixelLay",
    "sim_n3DLay",
    "sim_nTrackerHits",
    "sim_nRecoClusters",
]

# Target mode:
#   "qoverpt_lambda_phi_dxy_dz"  -> [q/pt, lambda, phi, dxy, dz]
#   "pt_eta_phi_dxy_dz"          -> [pt, eta, phi, dxy, dz]
#   "pt_lambda_phi_dxy_dz"       -> [pt, lambda, phi, dxy, dz]
TARGET_MODE = "qoverpt_lambda_phi_dxy_dz"

# These are the source branches used to build targets.
# Leave them here even if your final label vector is only length 5.
TARGET_SOURCE_BRANCHES = [
    "sim_q",
    "sim_pca_pt",
    "sim_pca_eta",
    "sim_pca_lambda",
    "sim_pca_phi",
    "sim_pca_dxy",
    "sim_pca_dz",
]

# Optional cuts / cleaning
DROP_NONFINITE = True
MIN_SIM_PT = None          # e.g. 0.5
MAX_ABS_SIM_ETA = None     # e.g. 2.5

# Output behavior
SORT_BY = ["source_file", "event_id", "sim_idx"]
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


def get_event_id(tree, entry_idx: int) -> int:
    for candidate in ("event", "event_id", "evt"):
        if hasattr(tree, candidate):
            try:
                return int(getattr(tree, candidate))
            except Exception:
                pass
    return int(entry_idx)


def build_target_row(payload: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    q = payload["sim_q"].astype(np.float32)
    pt = payload["sim_pca_pt"].astype(np.float32)
    eta = payload["sim_pca_eta"].astype(np.float32)
    lam = payload["sim_pca_lambda"].astype(np.float32)
    phi = payload["sim_pca_phi"].astype(np.float32)
    dxy = payload["sim_pca_dxy"].astype(np.float32)
    dz = payload["sim_pca_dz"].astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        qoverpt = q / pt

    if TARGET_MODE == "qoverpt_lambda_phi_dxy_dz":
        return {
            "target_qoverpt": qoverpt,
            "target_lambda": lam,
            "target_phi": phi,
            "target_dxy": dxy,
            "target_dz": dz,
        }

    if TARGET_MODE == "pt_eta_phi_dxy_dz":
        return {
            "target_pt": pt,
            "target_eta": eta,
            "target_phi": phi,
            "target_dxy": dxy,
            "target_dz": dz,
        }

    if TARGET_MODE == "pt_lambda_phi_dxy_dz":
        return {
            "target_pt": pt,
            "target_lambda": lam,
            "target_phi": phi,
            "target_dxy": dxy,
            "target_dz": dz,
        }

    raise ValueError(f"Unknown TARGET_MODE: {TARGET_MODE}")


# ---------- Worker logic ----------

def process_chunk(args: Tuple) -> Dict[str, np.ndarray]:
    (
        root_path,
        tree_path,
        start,
        end,
        input_branches,
        target_source_branches,
        target_mode,
        drop_nonfinite,
        min_sim_pt,
        max_abs_sim_eta,
    ) = args

    global TARGET_MODE
    TARGET_MODE = target_mode

    f, tree = open_tree(root_path, tree_path)

    rows: Dict[str, List[object]] = {
        "source_file": [],
        "entry": [],
        "event_id": [],
        "sim_idx": [],
    }

    for b in input_branches:
        rows[b] = []

    target_names = {
        "qoverpt_lambda_phi_dxy_dz": [
            "target_qoverpt", "target_lambda", "target_phi", "target_dxy", "target_dz"
        ],
        "pt_eta_phi_dxy_dz": [
            "target_pt", "target_eta", "target_phi", "target_dxy", "target_dz"
        ],
        "pt_lambda_phi_dxy_dz": [
            "target_pt", "target_lambda", "target_phi", "target_dxy", "target_dz"
        ],
    }[target_mode]

    for name in target_names:
        rows[name] = []

    needed_branches = set(input_branches) | set(target_source_branches)

    for entry_idx in range(start, end):
        tree.GetEntry(entry_idx)
        event_id = get_event_id(tree, entry_idx)

        payload: Dict[str, np.ndarray] = {}

        for b in needed_branches:
            if not hasattr(tree, b):
                f.Close()
                raise RuntimeError(f"Missing branch '{b}' in {root_path}")

            obj = getattr(tree, b)

            if b == "sim_q":
                payload[b] = branch_to_numpy_int(obj).astype(np.float32)
            else:
                payload[b] = branch_to_numpy_float(obj)

        if "sim_pt" not in payload:
            f.Close()
            raise RuntimeError(f"Missing required branch 'sim_pt' in {root_path}")

        n_sim = len(payload["sim_pt"])
        if n_sim == 0:
            continue

        # Basic consistency check: all branches should have same per-event length
        for b, arr in payload.items():
            if len(arr) != n_sim:
                f.Close()
                raise RuntimeError(
                    f"Length mismatch in {root_path}, entry {entry_idx}: "
                    f"branch '{b}' has len {len(arr)} but sim_pt has len {n_sim}"
                )

        target_payload = build_target_row(payload)

        for i_sim in range(n_sim):
            # optional cuts
            if min_sim_pt is not None and float(payload["sim_pt"][i_sim]) < float(min_sim_pt):
                continue

            if max_abs_sim_eta is not None and abs(float(payload["sim_eta"][i_sim])) > float(max_abs_sim_eta):
                continue

            row_values = []

            for b in input_branches:
                row_values.append(float(payload[b][i_sim]))

            for name in target_names:
                row_values.append(float(target_payload[name][i_sim]))

            if drop_nonfinite and not np.isfinite(np.asarray(row_values, dtype=np.float32)).all():
                continue

            rows["source_file"].append(str(root_path))
            rows["entry"].append(int(entry_idx))
            rows["event_id"].append(int(event_id))
            rows["sim_idx"].append(int(i_sim))

            for b in input_branches:
                rows[b].append(float(payload[b][i_sim]))

            for name in target_names:
                rows[name].append(float(target_payload[name][i_sim]))

    f.Close()

    out = {"source_file": np.asarray(rows["source_file"], dtype=object)}
    for key, values in rows.items():
        if key == "source_file":
            continue
        if key in {"entry", "event_id", "sim_idx"}:
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


def get_label_columns(target_mode: str) -> List[str]:
    if target_mode == "qoverpt_lambda_phi_dxy_dz":
        return ["target_qoverpt", "target_lambda", "target_phi", "target_dxy", "target_dz"]
    if target_mode == "pt_eta_phi_dxy_dz":
        return ["target_pt", "target_eta", "target_phi", "target_dxy", "target_dz"]
    if target_mode == "pt_lambda_phi_dxy_dz":
        return ["target_pt", "target_lambda", "target_phi", "target_dxy", "target_dz"]
    raise ValueError(f"Unknown TARGET_MODE: {target_mode}")


def build_xy(df: pd.DataFrame, input_branches: Sequence[str], target_mode: str):
    feature_cols = list(input_branches)
    label_cols = get_label_columns(target_mode)

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
    input_branches: Sequence[str],
    target_source_branches: Sequence[str],
    target_mode: str,
    chunk_size: int,
    drop_nonfinite: bool,
    min_sim_pt,
    max_abs_sim_eta,
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
                    tuple(input_branches),
                    tuple(target_source_branches),
                    target_mode,
                    bool(drop_nonfinite),
                    min_sim_pt,
                    max_abs_sim_eta,
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
        input_branches=INPUT_BRANCHES,
        target_source_branches=TARGET_SOURCE_BRANCHES,
        target_mode=TARGET_MODE,
        chunk_size=CHUNK_SIZE,
        drop_nonfinite=DROP_NONFINITE,
        min_sim_pt=MIN_SIM_PT,
        max_abs_sim_eta=MAX_ABS_SIM_ETA,
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

    X, Y, feature_cols, label_cols = build_xy(
        df,
        input_branches=INPUT_BRANCHES,
        target_mode=TARGET_MODE,
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
        "n_features": int(X.shape[1]),
        "n_labels": int(Y.shape[1]),
        "feature_columns": feature_cols,
        "label_columns": label_cols,
        "input_branches": list(INPUT_BRANCHES),
        "target_mode": TARGET_MODE,
        "target_source_branches": list(TARGET_SOURCE_BRANCHES),
        "chunk_size": int(CHUNK_SIZE),
        "n_workers": int(N_WORKERS),
        "drop_nonfinite": bool(DROP_NONFINITE),
        "min_sim_pt": MIN_SIM_PT,
        "max_abs_sim_eta": MAX_ABS_SIM_ETA,
    }
    with open(outdir / "column_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved X to {outdir / 'features.npy'} with shape {X.shape}")
    print(f"Saved Y to {outdir / 'labels.npy'} with shape {Y.shape}")
    print(f"Saved metadata to {outdir / 'column_metadata.json'}")


if __name__ == "__main__":
    main()