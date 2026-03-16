#!/usr/bin/env python3
"""
Preprocess track cache for NN training.

Behavior:
- load from track_data.npz if present, otherwise features.npy + labels.npy
- filter to "good" samples in a target range
- if features are variable-length, keep only the most common input length
- split into train / val / test
- compute normalization stats from TRAIN only
- save processed dataset for later training

Usage:
    python preprocess_track_cache.py

Edit CONFIG below if needed.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np


# ==========================================================
# CONFIG
# ==========================================================
DATA_DIR = Path("track_cache")

# load preference:
# 1) DATA_DIR / "track_data.npz"
# 2) DATA_DIR / "features.npy" and DATA_DIR / "labels.npy"
NPZ_NAME = "track_data.npz"
FEATURES_NAME = "features.npy"
LABELS_NAME = "labels.npy"

# keys inside NPZ if using track_data.npz
FEATURE_KEY_CANDIDATES = ["X", "features", "x"]
LABEL_KEY_CANDIDATES = ["y", "labels", "target", "targets"]

SEED = 0

TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10

# good-sample target filter
GOOD_MIN = -1.0
GOOD_MAX = 1.0

# which target columns to apply the range cut to
# None = all target columns
# example: [0] if you only want first parameter in [-1,1]
TARGET_COLS_TO_FILTER = None

SAVE_NORMALIZED = True

OUTPUT_NAME = "processed_most_common_bucket.npz"


# ==========================================================
# HELPERS
# ==========================================================
def find_first_key(data, candidates, kind):
    for k in candidates:
        if k in data.files:
            return k
    raise KeyError(f"Could not find {kind} key in NPZ. Available keys: {data.files}")


def load_data(data_dir: Path):
    npz_path = data_dir / NPZ_NAME
    feat_path = data_dir / FEATURES_NAME
    lab_path = data_dir / LABELS_NAME

    if npz_path.exists():
        print(f"Loading NPZ: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)

        print("\nKeys in NPZ:")
        for k in data.files:
            print(f"  {k}: shape={data[k].shape}, dtype={data[k].dtype}")

        x_key = find_first_key(data, FEATURE_KEY_CANDIDATES, "feature")
        y_key = find_first_key(data, LABEL_KEY_CANDIDATES, "label")

        X = data[x_key]
        y = np.asarray(data[y_key], dtype=np.float32)

        print(f"\nUsing NPZ keys:")
        print(f"  features = {x_key}")
        print(f"  labels   = {y_key}")
        return X, y

    if feat_path.exists() and lab_path.exists():
        print(f"Loading NPY files:")
        print(f"  {feat_path}")
        print(f"  {lab_path}")
        X = np.load(feat_path, allow_pickle=True)
        y = np.load(lab_path, allow_pickle=True).astype(np.float32)

        print(f"\nLoaded:")
        print(f"  X: shape={X.shape}, dtype={X.dtype}")
        print(f"  y: shape={y.shape}, dtype={y.dtype}")
        return X, y

    raise FileNotFoundError(
        f"Could not find {npz_path} or the pair "
        f"{feat_path} / {lab_path}"
    )


def is_ragged_object_array(X):
    return isinstance(X, np.ndarray) and X.dtype == object


def get_input_lengths(X):
    if isinstance(X, np.ndarray) and X.dtype != object:
        if X.ndim != 2:
            raise ValueError(f"Expected dense X to be 2D, got shape {X.shape}")
        return np.full(X.shape[0], X.shape[1], dtype=np.int32)

    lengths = []
    for i, row in enumerate(X):
        arr = np.asarray(row)
        if arr.ndim == 0:
            raise ValueError(f"Sample {i} is scalar-like; expected vector-like")
        lengths.append(arr.reshape(-1).shape[0])
    return np.asarray(lengths, dtype=np.int32)


def summarize_lengths(lengths):
    counts = Counter(lengths.tolist())
    print("\nInput-size distribution:")
    for length, count in counts.most_common(15):
        print(f"  length {length:5d}: {count:10d} samples")
    return counts


def finite_mask_dense(X, y):
    return np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)


def finite_mask_ragged(X, y):
    mask = np.isfinite(y).all(axis=1).copy()
    for i in range(len(X)):
        xi = np.asarray(X[i], dtype=np.float32).reshape(-1)
        if not np.isfinite(xi).all():
            mask[i] = False
    return mask


def target_range_mask(y, cols, lo, hi):
    if y.ndim != 2:
        raise ValueError(f"Expected y to be 2D, got shape {y.shape}")

    if cols is None:
        cols = list(range(y.shape[1]))

    mask = np.ones(y.shape[0], dtype=bool)
    for c in cols:
        mask &= (y[:, c] >= lo) & (y[:, c] <= hi)
    return mask


def convert_bucket_to_dense_2d(X_bucket):
    rows = [np.asarray(x, dtype=np.float32).reshape(-1) for x in X_bucket]
    return np.stack(rows, axis=0).astype(np.float32)


def split_data(X, y, seed=0):
    n = y.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    X = X[perm]
    y = y[perm]

    n_train = int(TRAIN_FRAC * n)
    n_val = int(VAL_FRAC * n)

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]

    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_stats(arr):
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(arr, mean, std):
    return ((arr - mean) / std).astype(np.float32)


# ==========================================================
# MAIN
# ==========================================================
def main():
    if not np.isclose(TRAIN_FRAC + VAL_FRAC + TEST_FRAC, 1.0):
        raise ValueError("TRAIN_FRAC + VAL_FRAC + TEST_FRAC must sum to 1")

    X_raw, y = load_data(DATA_DIR)

    if y.ndim != 2:
        raise ValueError(f"Expected y to be 2D, got shape {y.shape}")

    if len(X_raw) != len(y):
        raise ValueError(f"X/y length mismatch: {len(X_raw)} vs {len(y)}")

    n0 = len(y)
    lengths = get_input_lengths(X_raw)
    summarize_lengths(lengths)

    # -----------------------------
    # finite filtering
    # -----------------------------
    if is_ragged_object_array(X_raw):
        mask = finite_mask_ragged(X_raw, y)
    else:
        X_dense0 = np.asarray(X_raw, dtype=np.float32)
        mask = finite_mask_dense(X_dense0, y)

    print(f"\nFinite rows kept: {mask.sum():,} / {n0:,}")

    # -----------------------------
    # target-range filtering
    # -----------------------------
    tmask = target_range_mask(
        y,
        cols=TARGET_COLS_TO_FILTER,
        lo=GOOD_MIN,
        hi=GOOD_MAX,
    )
    print(f"Target-range rows kept [{GOOD_MIN}, {GOOD_MAX}]: {tmask.sum():,} / {n0:,}")

    mask &= tmask

    X_raw = X_raw[mask]
    y = y[mask]
    lengths = lengths[mask]

    print(f"\nRows after all filtering: {len(y):,}")

    if len(y) < 10:
        raise RuntimeError("Too few rows left after filtering")

    # -----------------------------
    # most common bucket
    # -----------------------------
    counts = Counter(lengths.tolist())
    bucket_len, bucket_count = counts.most_common(1)[0]

    print(f"\nKeeping most common bucket:")
    print(f"  input length = {bucket_len}")
    print(f"  samples      = {bucket_count:,}")

    keep = (lengths == bucket_len)
    X_bucket = X_raw[keep]
    y_bucket = y[keep]

    # convert to dense 2D if needed
    if is_ragged_object_array(X_bucket):
        X = convert_bucket_to_dense_2d(X_bucket)
    else:
        X = np.asarray(X_bucket, dtype=np.float32)

    print(f"\nFinal bucketed dataset:")
    print(f"  X shape = {X.shape}")
    print(f"  y shape = {y_bucket.shape}")

    # -----------------------------
    # split
    # -----------------------------
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y_bucket, seed=SEED)

    print("\nSplit sizes:")
    print(f"  train = {len(y_train):,}")
    print(f"  val   = {len(y_val):,}")
    print(f"  test  = {len(y_test):,}")

    # -----------------------------
    # normalization from TRAIN only
    # -----------------------------
    x_mean, x_std = compute_stats(X_train)
    y_mean, y_std = compute_stats(y_train)

    save_dict = {
        "X_train": X_train.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "X_val": X_val.astype(np.float32),
        "y_val": y_val.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "input_dim": np.array([X.shape[1]], dtype=np.int32),
        "output_dim": np.array([y_bucket.shape[1]], dtype=np.int32),
        "bucket_length": np.array([bucket_len], dtype=np.int32),
    }

    if SAVE_NORMALIZED:
        save_dict["X_train_norm"] = standardize(X_train, x_mean, x_std)
        save_dict["X_val_norm"] = standardize(X_val, x_mean, x_std)
        save_dict["X_test_norm"] = standardize(X_test, x_mean, x_std)

        save_dict["y_train_norm"] = standardize(y_train, y_mean, y_std)
        save_dict["y_val_norm"] = standardize(y_val, y_mean, y_std)
        save_dict["y_test_norm"] = standardize(y_test, y_mean, y_std)

    out_path = DATA_DIR / OUTPUT_NAME
    np.savez_compressed(out_path, **save_dict)

    meta = {
        "data_dir": str(DATA_DIR),
        "seed": SEED,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
        "good_min": GOOD_MIN,
        "good_max": GOOD_MAX,
        "target_cols_to_filter": TARGET_COLS_TO_FILTER,
        "n_rows_before_filter": int(n0),
        "n_rows_after_filter": int(len(y)),
        "bucket_length": int(bucket_len),
        "bucket_samples": int(len(y_bucket)),
        "input_dim": int(X.shape[1]),
        "output_dim": int(y_bucket.shape[1]),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
    }

    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved processed dataset: {out_path}")
    print(f"Saved metadata:          {meta_path}")


if __name__ == "__main__":
    main()