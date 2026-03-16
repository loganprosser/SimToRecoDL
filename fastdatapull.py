#!/usr/bin/env python3
"""
Build a row-aligned SIM-track cache from all ROOT files in one directory
without exploding RAM usage.

Run:
    python root_to_track_cache_streaming.py

This version is designed to keep memory low by:
- doing a first pass that only COUNTS kept rows per chunk
- allocating final output arrays once as memory-mapped .npy files
- doing a second pass that writes chunk results directly into those files
- never concatenating all chunks in RAM
- optionally writing row-cache shards per chunk instead of one giant DataFrame

Outputs:
- features.npy
- labels.npy
- column_metadata.json
- optional row shards in OUTDIR/row_shards/

Notes:
- This uses .npy files, not .npz, because .npz is not friendly for true incremental appends.
- .npy files created here are memory-mapped and written incrementally.
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

try:
    import ROOT  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "PyROOT is required for this script. Load your ROOT environment first."
    ) from exc


# =========================
# CONSTANTS: EDIT THESE
# =========================
INPUT_DIR = "/data2/segmentlinking/CMSSW_12_5_0_pre3/RelValTTbar_14TeV_CMSSW_12_5_0_pre3"
FILE_GLOB = "*.root"
TREE_PATH = "trackingNtuple/tree"
OUTDIR = "./track_cache"

# Parallelism
N_WORKERS = 16

# Set to None for smart auto chunking
CHUNK_SIZE = None

# Heuristic controls for auto chunking
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 5000
TARGET_JOBS_PER_WORKER = 6
TARGET_TOTAL_JOBS_CAP = 20

# Progress printing
PROGRESS_PRINT_EVERY_SEC = 5.0

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
MIN_SIM_PT = None
MAX_ABS_SIM_ETA = None

# Output behavior
SORT_WITHIN_CHUNK = False  # global sort would require heavy extra work; keep False for lowest RAM
WRITE_ROW_SHARDS = False   # True = writes chunk-level parquet/hdf shards
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


def build_target_arrays(payload: Dict[str, np.ndarray], target_mode: str) -> List[np.ndarray]:
    q = payload["sim_q"].astype(np.float32)
    pt = payload["sim_pca_pt"].astype(np.float32)
    eta = payload["sim_pca_eta"].astype(np.float32)
    lam = payload["sim_pca_lambda"].astype(np.float32)
    phi = payload["sim_pca_phi"].astype(np.float32)
    dxy = payload["sim_pca_dxy"].astype(np.float32)
    dz = payload["sim_pca_dz"].astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        qoverpt = q / pt

    if target_mode == "qoverpt_lambda_phi_dxy_dz":
        return [qoverpt, lam, phi, dxy, dz]

    if target_mode == "pt_eta_phi_dxy_dz":
        return [pt, eta, phi, dxy, dz]

    if target_mode == "pt_lambda_phi_dxy_dz":
        return [pt, lam, phi, dxy, dz]

    raise ValueError(f"Unknown TARGET_MODE: {target_mode}")


def get_label_columns(target_mode: str) -> List[str]:
    if target_mode == "qoverpt_lambda_phi_dxy_dz":
        return ["target_qoverpt", "target_lambda", "target_phi", "target_dxy", "target_dz"]
    if target_mode == "pt_eta_phi_dxy_dz":
        return ["target_pt", "target_eta", "target_phi", "target_dxy", "target_dz"]
    if target_mode == "pt_lambda_phi_dxy_dz":
        return ["target_pt", "target_lambda", "target_phi", "target_dxy", "target_dz"]
    raise ValueError(f"Unknown TARGET_MODE: {target_mode}")


# ---------- Job planning ----------

def discover_root_files(input_dir: str, file_glob: str) -> List[str]:
    return sorted(str(p.resolve()) for p in Path(input_dir).glob(file_glob) if p.is_file())


def collect_file_entry_counts(input_files: Sequence[str], tree_path: str) -> List[Dict[str, Any]]:
    info: List[Dict[str, Any]] = []
    for root_path in input_files:
        f, tree = open_tree(root_path, tree_path)
        n_entries = int(tree.GetEntries())
        f.Close()
        info.append(
            {
                "root_path": root_path,
                "n_entries": n_entries,
                "basename": Path(root_path).name,
            }
        )
    return info


def choose_chunk_size(
    file_infos: Sequence[Dict[str, Any]],
    n_workers: int,
    user_chunk_size,
    min_chunk_size: int,
    max_chunk_size: int,
    target_jobs_per_worker: int,
    target_total_jobs_cap: int,
) -> int:
    if user_chunk_size is not None:
        return int(user_chunk_size)

    total_entries = sum(int(x["n_entries"]) for x in file_infos)
    n_files = len(file_infos)

    if total_entries <= 0:
        return min_chunk_size

    desired_jobs_floor = max(n_files, n_workers * target_jobs_per_worker)
    desired_jobs_cap = max(desired_jobs_floor, n_workers * target_total_jobs_cap)

    chunk_from_floor = math.ceil(total_entries / desired_jobs_floor)
    chunk_from_cap = math.ceil(total_entries / desired_jobs_cap)

    raw_chunk = max(chunk_from_cap, min(chunk_from_floor, chunk_from_cap * 2))
    chunk = max(min_chunk_size, min(max_chunk_size, raw_chunk))
    return int(chunk)


def make_jobs(
    file_infos: Sequence[Dict[str, Any]],
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

    for file_id, finfo in enumerate(file_infos):
        root_path = finfo["root_path"]
        n_entries = int(finfo["n_entries"])
        basename = finfo["basename"]

        for start in range(0, n_entries, chunk_size):
            end = min(start + chunk_size, n_entries)
            jobs.append(
                (
                    {
                        "job_id": len(jobs),
                        "file_id": file_id,
                        "root_path": root_path,
                        "basename": basename,
                        "n_entries_in_file": n_entries,
                        "start": start,
                        "end": end,
                    },
                    tree_path,
                    tuple(input_branches),
                    tuple(target_source_branches),
                    target_mode,
                    bool(drop_nonfinite),
                    min_sim_pt,
                    max_abs_sim_eta,
                )
            )

    return jobs


# ---------- Worker logic ----------

def process_chunk_count(args: Tuple) -> Dict[str, Any]:
    (
        job_info,
        tree_path,
        input_branches,
        target_source_branches,
        target_mode,
        drop_nonfinite,
        min_sim_pt,
        max_abs_sim_eta,
    ) = args

    root_path = job_info["root_path"]
    start = job_info["start"]
    end = job_info["end"]

    f, tree = open_tree(root_path, tree_path)

    needed_branches = set(input_branches) | set(target_source_branches)
    kept_rows = 0

    for entry_idx in range(start, end):
        tree.GetEntry(entry_idx)

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

        n_sim = len(payload["sim_pt"])
        if n_sim == 0:
            continue

        for b, arr in payload.items():
            if len(arr) != n_sim:
                f.Close()
                raise RuntimeError(
                    f"Length mismatch in {root_path}, entry {entry_idx}: "
                    f"branch '{b}' has len {len(arr)} but sim_pt has len {n_sim}"
                )

        target_arrays = build_target_arrays(payload, target_mode)

        for i_sim in range(n_sim):
            if min_sim_pt is not None and float(payload["sim_pt"][i_sim]) < float(min_sim_pt):
                continue
            if max_abs_sim_eta is not None and abs(float(payload["sim_eta"][i_sim])) > float(max_abs_sim_eta):
                continue

            if drop_nonfinite:
                vals = [float(payload[b][i_sim]) for b in input_branches]
                vals.extend(float(arr[i_sim]) for arr in target_arrays)
                if not np.isfinite(np.asarray(vals, dtype=np.float32)).all():
                    continue

            kept_rows += 1

    f.Close()

    return {
        "job_id": int(job_info["job_id"]),
        "basename": job_info["basename"],
        "start": int(start),
        "end": int(end),
        "chunk_entries": int(end - start),
        "n_rows": int(kept_rows),
    }


def process_chunk_data(args: Tuple) -> Dict[str, Any]:
    (
        job_info,
        tree_path,
        input_branches,
        target_source_branches,
        target_mode,
        drop_nonfinite,
        min_sim_pt,
        max_abs_sim_eta,
    ) = args

    root_path = job_info["root_path"]
    start = job_info["start"]
    end = job_info["end"]

    f, tree = open_tree(root_path, tree_path)

    needed_branches = set(input_branches) | set(target_source_branches)
    n_features = len(input_branches)
    n_labels = len(get_label_columns(target_mode))

    feature_rows: List[List[float]] = []
    label_rows: List[List[float]] = []

    meta_rows = {
        "source_file": [],
        "entry": [],
        "event_id": [],
        "sim_idx": [],
    }

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

        n_sim = len(payload["sim_pt"])
        if n_sim == 0:
            continue

        for b, arr in payload.items():
            if len(arr) != n_sim:
                f.Close()
                raise RuntimeError(
                    f"Length mismatch in {root_path}, entry {entry_idx}: "
                    f"branch '{b}' has len {len(arr)} but sim_pt has len {n_sim}"
                )

        target_arrays = build_target_arrays(payload, target_mode)

        for i_sim in range(n_sim):
            if min_sim_pt is not None and float(payload["sim_pt"][i_sim]) < float(min_sim_pt):
                continue
            if max_abs_sim_eta is not None and abs(float(payload["sim_eta"][i_sim])) > float(max_abs_sim_eta):
                continue

            feat = [float(payload[b][i_sim]) for b in input_branches]
            lab = [float(arr[i_sim]) for arr in target_arrays]

            if drop_nonfinite:
                if not np.isfinite(np.asarray(feat + lab, dtype=np.float32)).all():
                    continue

            feature_rows.append(feat)
            label_rows.append(lab)

            if WRITE_ROW_SHARDS:
                meta_rows["source_file"].append(str(root_path))
                meta_rows["entry"].append(int(entry_idx))
                meta_rows["event_id"].append(int(event_id))
                meta_rows["sim_idx"].append(int(i_sim))

    f.Close()

    if feature_rows:
        X = np.asarray(feature_rows, dtype=np.float32)
        Y = np.asarray(label_rows, dtype=np.float32)
    else:
        X = np.empty((0, n_features), dtype=np.float32)
        Y = np.empty((0, n_labels), dtype=np.float32)

    result = {
        "job_id": int(job_info["job_id"]),
        "basename": job_info["basename"],
        "start": int(start),
        "end": int(end),
        "n_rows": int(X.shape[0]),
        "X": X,
        "Y": Y,
    }

    if WRITE_ROW_SHARDS:
        result["meta"] = meta_rows

    return result


# ---------- Row-shard writer ----------

def write_row_shard(
    *,
    outdir: Path,
    job_id: int,
    X: np.ndarray,
    Y: np.ndarray,
    meta: Dict[str, List[Any]],
    feature_cols: Sequence[str],
    label_cols: Sequence[str],
) -> Path:
    shard_dir = outdir / "row_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    cols: Dict[str, Any] = {
        "source_file": np.asarray(meta["source_file"], dtype=object),
        "entry": np.asarray(meta["entry"], dtype=np.int64),
        "event_id": np.asarray(meta["event_id"], dtype=np.int64),
        "sim_idx": np.asarray(meta["sim_idx"], dtype=np.int64),
    }

    for i, c in enumerate(feature_cols):
        cols[c] = X[:, i] if X.size else np.empty((0,), dtype=np.float32)

    for i, c in enumerate(label_cols):
        cols[c] = Y[:, i] if Y.size else np.empty((0,), dtype=np.float32)

    df = pd.DataFrame(cols)

    if SORT_WITHIN_CHUNK and not df.empty:
        df = df.sort_values(["source_file", "event_id", "sim_idx"], kind="stable").reset_index(drop=True)

    if PREFER_PARQUET:
        try:
            path = shard_dir / f"rows_job_{job_id:06d}.parquet"
            df.to_parquet(path, index=False)
            return path
        except Exception:
            pass

    path = shard_dir / f"rows_job_{job_id:06d}.h5"
    df.to_hdf(path, key="tracks", mode="w", format="table")
    return path


# ---------- Progress helpers ----------

def format_seconds(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"


def print_progress(
    *,
    phase: str,
    done_jobs: int,
    total_jobs: int,
    done_entries: int,
    total_entries: int,
    total_rows: int,
    start_time: float,
    file_done_entries: Dict[str, int],
    file_total_entries: Dict[str, int],
):
    elapsed = time.time() - start_time
    frac_jobs = (done_jobs / total_jobs) if total_jobs else 1.0
    frac_entries = (done_entries / total_entries) if total_entries else 1.0

    if done_entries > 0 and elapsed > 0:
        entry_rate = done_entries / elapsed
        eta = (total_entries - done_entries) / entry_rate if entry_rate > 0 else math.inf
    else:
        entry_rate = 0.0
        eta = math.inf

    print(
        f"[{phase}] jobs {done_jobs}/{total_jobs} ({100.0*frac_jobs:5.1f}%) | "
        f"entries {done_entries}/{total_entries} ({100.0*frac_entries:5.1f}%) | "
        f"rows {total_rows:,} | "
        f"rate {entry_rate:,.1f} entries/s | "
        f"elapsed {format_seconds(elapsed)} | "
        f"ETA {format_seconds(eta)}"
    )

    incomplete = []
    for bname, total in file_total_entries.items():
        done = file_done_entries.get(bname, 0)
        if done < total:
            incomplete.append((done / total if total else 1.0, bname, done, total))

    incomplete.sort(key=lambda x: x[0])
    if incomplete:
        print(f"[{phase}] slowest/incomplete files:")
        for _, bname, done, total in incomplete[:5]:
            print(f"    {bname}: {done}/{total} entries ({100.0*done/total:5.1f}%)")


# ---------- Main ----------

def main() -> None:
    input_files = discover_root_files(INPUT_DIR, FILE_GLOB)
    if not input_files:
        raise RuntimeError(f"No ROOT files matched {FILE_GLOB!r} in directory {INPUT_DIR!r}")

    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning ROOT files in {Path(INPUT_DIR).resolve()} ...")
    file_infos = collect_file_entry_counts(input_files, TREE_PATH)

    total_entries = sum(int(x["n_entries"]) for x in file_infos)

    auto_chunk_size = choose_chunk_size(
        file_infos=file_infos,
        n_workers=N_WORKERS,
        user_chunk_size=CHUNK_SIZE,
        min_chunk_size=MIN_CHUNK_SIZE,
        max_chunk_size=MAX_CHUNK_SIZE,
        target_jobs_per_worker=TARGET_JOBS_PER_WORKER,
        target_total_jobs_cap=TARGET_TOTAL_JOBS_CAP,
    )

    jobs = make_jobs(
        file_infos=file_infos,
        tree_path=TREE_PATH,
        input_branches=INPUT_BRANCHES,
        target_source_branches=TARGET_SOURCE_BRANCHES,
        target_mode=TARGET_MODE,
        chunk_size=auto_chunk_size,
        drop_nonfinite=DROP_NONFINITE,
        min_sim_pt=MIN_SIM_PT,
        max_abs_sim_eta=MAX_ABS_SIM_ETA,
    )

    print(f"Found {len(input_files)} ROOT file(s)")
    print(f"Total tree entries across all files: {total_entries:,}")
    print(f"N_WORKERS = {N_WORKERS}")
    print(f"Chunk size = {auto_chunk_size:,} entries/job")
    print(f"Planned {len(jobs)} job(s)")
    print("Files:")
    for finfo in file_infos:
        print(f"  - {finfo['basename']}: {finfo['n_entries']:,} entries")

    feature_cols = list(INPUT_BRANCHES)
    label_cols = get_label_columns(TARGET_MODE)

    file_total_entries: Dict[str, int] = {
        finfo["basename"]: int(finfo["n_entries"]) for finfo in file_infos
    }

    # =========================================================
    # PASS 1: count rows per chunk
    # =========================================================
    print("\n=== PASS 1: counting surviving rows per chunk ===")
    count_results: Dict[int, Dict[str, Any]] = {}
    done_jobs = 0
    done_entries = 0
    total_rows = 0
    file_done_entries: Dict[str, int] = defaultdict(int)
    t0 = time.time()
    last_progress_time = 0.0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        future_to_job = {ex.submit(process_chunk_count, job): job for job in jobs}

        for fut in as_completed(future_to_job):
            original_job = future_to_job[fut]
            job_info = original_job[0]
            basename = job_info["basename"]
            start = job_info["start"]
            end = job_info["end"]

            try:
                result = fut.result()
            except Exception as exc:
                raise RuntimeError(
                    f"Count pass failed for file={basename}, entries=[{start}, {end}): {exc}"
                ) from exc

            count_results[result["job_id"]] = result
            done_jobs += 1
            done_entries += int(result["chunk_entries"])
            total_rows += int(result["n_rows"])
            file_done_entries[basename] += int(result["chunk_entries"])

            now = time.time()
            if (now - last_progress_time) >= PROGRESS_PRINT_EVERY_SEC or done_jobs == len(jobs):
                print_progress(
                    phase="count",
                    done_jobs=done_jobs,
                    total_jobs=len(jobs),
                    done_entries=done_entries,
                    total_entries=total_entries,
                    total_rows=total_rows,
                    start_time=t0,
                    file_done_entries=file_done_entries,
                    file_total_entries=file_total_entries,
                )
                last_progress_time = now

    n_total_rows = sum(count_results[job_info[0]["job_id"]]["n_rows"] for job_info in jobs)
    print(f"\nTotal kept rows after filtering: {n_total_rows:,}")

    if n_total_rows == 0:
        print("No rows were extracted.")
        return

    # Build deterministic offsets in original job order
    job_offsets: Dict[int, Tuple[int, int]] = {}
    cursor = 0
    for job in jobs:
        job_id = job[0]["job_id"]
        n_rows = int(count_results[job_id]["n_rows"])
        job_offsets[job_id] = (cursor, cursor + n_rows)
        cursor += n_rows

    # =========================================================
    # Allocate final outputs as memory-mapped .npy
    # =========================================================
    features_path = outdir / "features.npy"
    labels_path = outdir / "labels.npy"

    print("\nAllocating final memory-mapped output arrays ...")
    X_mem = open_memmap(
        filename=features_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_total_rows, len(feature_cols)),
    )
    Y_mem = open_memmap(
        filename=labels_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_total_rows, len(label_cols)),
    )

    # =========================================================
    # PASS 2: fill final arrays chunk by chunk
    # =========================================================
    print("\n=== PASS 2: extracting data and writing directly to disk-backed arrays ===")
    done_jobs = 0
    done_entries = 0
    written_rows = 0
    file_done_entries = defaultdict(int)
    t1 = time.time()
    last_progress_time = 0.0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        future_to_job = {ex.submit(process_chunk_data, job): job for job in jobs}

        for fut in as_completed(future_to_job):
            original_job = future_to_job[fut]
            job_info = original_job[0]
            job_id = int(job_info["job_id"])
            basename = job_info["basename"]
            start = job_info["start"]
            end = job_info["end"]

            try:
                result = fut.result()
            except Exception as exc:
                raise RuntimeError(
                    f"Data pass failed for file={basename}, entries=[{start}, {end}): {exc}"
                ) from exc

            expected_rows = int(count_results[job_id]["n_rows"])
            got_rows = int(result["n_rows"])
            if expected_rows != got_rows:
                raise RuntimeError(
                    f"Row count mismatch in job {job_id}: count pass expected {expected_rows}, "
                    f"data pass produced {got_rows}"
                )

            lo, hi = job_offsets[job_id]
            if got_rows > 0:
                X_mem[lo:hi, :] = result["X"]
                Y_mem[lo:hi, :] = result["Y"]

                if WRITE_ROW_SHARDS:
                    write_row_shard(
                        outdir=outdir,
                        job_id=job_id,
                        X=result["X"],
                        Y=result["Y"],
                        meta=result["meta"],
                        feature_cols=feature_cols,
                        label_cols=label_cols,
                    )

            done_jobs += 1
            done_entries += int(end - start)
            written_rows += got_rows
            file_done_entries[basename] += int(end - start)

            # free worker-returned arrays ASAP
            del result
            gc.collect()

            now = time.time()
            if (now - last_progress_time) >= PROGRESS_PRINT_EVERY_SEC or done_jobs == len(jobs):
                print_progress(
                    phase="write",
                    done_jobs=done_jobs,
                    total_jobs=len(jobs),
                    done_entries=done_entries,
                    total_entries=total_entries,
                    total_rows=written_rows,
                    start_time=t1,
                    file_done_entries=file_done_entries,
                    file_total_entries=file_total_entries,
                )
                last_progress_time = now

    X_mem.flush()
    Y_mem.flush()
    del X_mem
    del Y_mem
    gc.collect()

    metadata = {
        "input_dir": str(Path(INPUT_DIR).resolve()),
        "file_glob": FILE_GLOB,
        "input_files": input_files,
        "tree": TREE_PATH,
        "n_files": len(input_files),
        "n_tree_entries_total": int(total_entries),
        "n_rows_total": int(n_total_rows),
        "n_features": len(feature_cols),
        "n_labels": len(label_cols),
        "feature_columns": feature_cols,
        "label_columns": label_cols,
        "input_branches": list(INPUT_BRANCHES),
        "target_mode": TARGET_MODE,
        "target_source_branches": list(TARGET_SOURCE_BRANCHES),
        "chunk_size": int(auto_chunk_size),
        "n_workers": int(N_WORKERS),
        "drop_nonfinite": bool(DROP_NONFINITE),
        "min_sim_pt": MIN_SIM_PT,
        "max_abs_sim_eta": MAX_ABS_SIM_ETA,
        "min_chunk_size": int(MIN_CHUNK_SIZE),
        "max_chunk_size": int(MAX_CHUNK_SIZE),
        "target_jobs_per_worker": int(TARGET_JOBS_PER_WORKER),
        "write_row_shards": bool(WRITE_ROW_SHARDS),
        "prefer_parquet": bool(PREFER_PARQUET),
        "sort_within_chunk": bool(SORT_WITHIN_CHUNK),
        "features_file": str(features_path.resolve()),
        "labels_file": str(labels_path.resolve()),
        "array_format_note": "Arrays are .npy files written incrementally via numpy memmap.",
    }

    with open(outdir / "column_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone.")
    print(f"Saved X to {features_path} with shape ({n_total_rows}, {len(feature_cols)})")
    print(f"Saved Y to {labels_path} with shape ({n_total_rows}, {len(label_cols)})")
    print(f"Saved metadata to {outdir / 'column_metadata.json'}")
    if WRITE_ROW_SHARDS:
        print(f"Saved row shards under {outdir / 'row_shards'}")


if __name__ == "__main__":
    main()