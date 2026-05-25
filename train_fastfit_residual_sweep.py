import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import train_fastfit_residual_auto as base
from helpers import format_epoch_report, save_model_checkpoint
from helpers_data import DEFAULT_TARGET_COLS, set_seed
from helpers_vis import (
    compute_target_histogram_overlap,
    compute_target_scatter_linearity,
    make_training_history_plots,
)
from loss import hetero_gaussian_nll_with_phi, hetero_huber_corr_loss
from model import HeteroTrackNet

DEFAULT_DATA_PATH = base.DEFAULT_DATA_PATH
DEFAULT_OUTPUT_DIR = "auto_fastfit_residual_sweep2"
DEFAULT_CACHE_DIR = "auto_fastfit_residual"
DEFAULT_DEVICE_SLOTS = base.DEFAULT_DEVICE_SLOTS
DEFAULT_MAX_CONCURRENT = base.DEFAULT_MAX_CONCURRENT
DEFAULT_DATALOADER_WORKERS = base.DEFAULT_DATALOADER_WORKERS
DEFAULT_FASTFIT_PRECOMPUTE_WORKERS = base.DEFAULT_FASTFIT_PRECOMPUTE_WORKERS
DEFAULT_BATCH_SIZE = base.BATCH_SIZE
DEFAULT_SEED = base.SEED
DEFAULT_VAL_FRACTION = base.VAL_FRACTION
DEFAULT_MODES = ",".join(base.TRAINING_MODES)
TARGET_KEY_ORDER = tuple(DEFAULT_TARGET_COLS)
DXY_TARGET = "pca_dxy"
DZ_TARGET = "pca_dz"
PHI_TARGET = "pca_phi"


@dataclass(frozen=True)
class SweepConfig:
    name: str
    hidden_layers: tuple[int, ...]
    lr: float
    epochs: int
    loss_name: str
    batchnorm: bool = False
    dropout: float = 0.0
    activation: str = "relu"
    min_lr: float = 1.0e-5


DEFAULT_CONFIGS = [
    SweepConfig(
        name="baseline_relu_512x512x256_lr1e3",
        hidden_layers=(512, 512, 256),
        lr=1.0e-3,
        epochs=280,
        loss_name="gaussian_nll",
    ),
    SweepConfig(
        name="small_silu_256x256x128_lr1e3",
        hidden_layers=(256, 256, 128),
        lr=1.0e-3,
        epochs=240,
        loss_name="gaussian_nll",
        activation="silu",
    ),
    SweepConfig(
        name="wide_gelu_768x512x256_lr5e4",
        hidden_layers=(768, 512, 256),
        lr=5.0e-4,
        epochs=320,
        loss_name="gaussian_nll",
        activation="gelu",
    ),
    SweepConfig(
        name="bn_drop_relu_512x512x256_lr7e4",
        hidden_layers=(512, 512, 256),
        lr=7.0e-4,
        epochs=280,
        loss_name="gaussian_nll",
        batchnorm=True,
        dropout=0.05,
    ),
    SweepConfig(
        name="robust_relu_512x512x256_lr1e3",
        hidden_layers=(512, 512, 256),
        lr=1.0e-3,
        epochs=240,
        loss_name="hetero_huber_corr",
    ),
    SweepConfig(
        name="robust_small_silu_256x256x128_lr7e4",
        hidden_layers=(256, 256, 128),
        lr=7.0e-4,
        epochs=220,
        loss_name="hetero_huber_corr",
        activation="silu",
    ),
    SweepConfig(
        name="medium_silu_384x384x192_lr8e4",
        hidden_layers=(384, 384, 192),
        lr=8.0e-4,
        epochs=260,
        loss_name="gaussian_nll",
        activation="silu",
    ),
    SweepConfig(
        name="deep_gelu_512x512x512x256_lr6e4",
        hidden_layers=(512, 512, 512, 256),
        lr=6.0e-4,
        epochs=320,
        loss_name="gaussian_nll",
        activation="gelu",
    ),
    SweepConfig(
        name="robust_medium_silu_384x256x256_lr7e4",
        hidden_layers=(384, 256, 256),
        lr=7.0e-4,
        epochs=240,
        loss_name="hetero_huber_corr",
        activation="silu",
    ),
    SweepConfig(
        name="robust_bn_relu_512x512x256_lr7e4",
        hidden_layers=(512, 512, 256),
        lr=7.0e-4,
        epochs=260,
        loss_name="hetero_huber_corr",
        batchnorm=True,
        dropout=0.05,
    ),
]

CONFIG_MAP = {cfg.name: cfg for cfg in DEFAULT_CONFIGS}
LOSS_FNS = {
    "gaussian_nll": hetero_gaussian_nll_with_phi,
    "hetero_huber_corr": hetero_huber_corr_loss,
}
ACTIVATIONS = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overnight multi-GPU sweep for fast-fit residual training."
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to the CSV to train on.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for sweep outputs.")
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="Directory holding reusable fast-fit and rotated/raw processed caches.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--dataloader-workers", type=int, default=DEFAULT_DATALOADER_WORKERS)
    parser.add_argument(
        "--fastfit-precompute-workers",
        type=int,
        default=DEFAULT_FASTFIT_PRECOMPUTE_WORKERS,
    )
    parser.add_argument("--show-plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print-final-samples", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--device-slots", default=DEFAULT_DEVICE_SLOTS)
    parser.add_argument("--device", default="", help="Device override for a single run.")
    parser.add_argument(
        "--modes",
        default=DEFAULT_MODES,
        help="Comma-separated modes to run. Supported: canonical, raw.",
    )
    parser.add_argument(
        "--configs",
        default=",".join(cfg.name for cfg in DEFAULT_CONFIGS),
        help="Comma-separated config names to run.",
    )
    parser.add_argument("--epochs-override", type=int, default=0, help="Force all configs to this epoch count.")
    parser.add_argument(
        "--single-mode",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--single-config",
        default="",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def parse_modes(modes_arg: str) -> list[str]:
    return base.parse_modes(modes_arg)


def parse_configs(configs_arg: str) -> list[SweepConfig]:
    names = [name.strip() for name in configs_arg.split(",") if name.strip()]
    if not names:
        raise ValueError("At least one config must be requested.")
    unknown = [name for name in names if name not in CONFIG_MAP]
    if unknown:
        raise ValueError(f"Unknown configs: {unknown}")
    return [CONFIG_MAP[name] for name in names]


def get_config_with_overrides(config: SweepConfig, args) -> SweepConfig:
    if args.epochs_override > 0 and args.epochs_override != config.epochs:
        return SweepConfig(**{**asdict(config), "epochs": int(args.epochs_override)})
    return config


def resolve_device(device_override: str = ""):
    return base.resolve_device(device_override)


def parse_device_slots(device_slots: str) -> list[str]:
    return base.parse_device_slots(device_slots)


def build_processed_cache_prefix(csv_path: Path, mode: str, seed: int, val_fraction: float) -> str:
    fastfit_prefix = base.build_fastfit_cache_prefix(csv_path)
    val_text = f"{val_fraction:.6f}".replace(".", "p")
    return f"{fastfit_prefix}_{mode}_seed{seed}_val{val_text}"


def _tensor_from_array(values: np.ndarray) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def _build_bundle_from_cache_arrays(payload: dict, batch_size: int, dataloader_workers: int, device, seed: int):
    x_train_t = _tensor_from_array(payload["x_train"])
    x_val_t = _tensor_from_array(payload["x_val"])
    y_train_t = _tensor_from_array(payload["y_train"])
    y_val_t = _tensor_from_array(payload["y_val"])
    rot_train_t = _tensor_from_array(payload["rot_train"])
    rot_val_t = _tensor_from_array(payload["rot_val"])
    baseline_train_t = _tensor_from_array(payload["baseline_train"])
    baseline_val_t = _tensor_from_array(payload["baseline_val"])

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        TensorDataset(x_train_t, y_train_t, rot_train_t, baseline_train_t),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=dataloader_workers,
    )
    val_loader = DataLoader(
        TensorDataset(x_val_t, y_val_t, rot_val_t, baseline_val_t),
        batch_size=batch_size,
        shuffle=False,
        generator=generator,
        num_workers=dataloader_workers,
    )

    return base.FastFitDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        x_train=x_train_t,
        x_val=x_val_t,
        y_train=y_train_t,
        y_val=y_val_t,
        rot_train=rot_train_t,
        rot_val=rot_val_t,
        x_mean=payload["x_mean"],
        x_std=payload["x_std"],
        y_mean=payload["y_mean"],
        y_std=payload["y_std"],
        y_mean_t=torch.tensor(payload["y_mean"], dtype=torch.float32, device=device),
        y_std_t=torch.tensor(payload["y_std"], dtype=torch.float32, device=device),
        feature_cols=payload["feature_cols"],
        target_cols=payload["target_cols"],
        fast_fit_cols=payload["fast_fit_cols"],
        phi_index=payload["phi_index"],
        rotation_source=payload["rotation_source"],
        training_mode=payload["training_mode"],
        n_fastfit_failures=payload["n_fastfit_failures"],
    )


def load_or_create_processed_bundle(
    csv_path: Path,
    cache_root: Path,
    batch_size: int,
    seed: int,
    device,
    val_fraction: float,
    dataloader_workers: int,
    training_mode: str,
    fastfit_precompute_workers: int,
):
    processed_cache_dir = cache_root / "processed_cache"
    processed_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_prefix = build_processed_cache_prefix(csv_path, training_mode, seed, val_fraction)
    cache_npz = processed_cache_dir / f"{cache_prefix}.npz"
    cache_meta = processed_cache_dir / f"{cache_prefix}.json"

    source_stat = csv_path.stat()
    expected_meta = {
        "csv_path": str(csv_path.resolve()),
        "csv_size": int(source_stat.st_size),
        "csv_mtime_ns": int(source_stat.st_mtime_ns),
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "training_mode": training_mode,
        "targets": list(DEFAULT_TARGET_COLS),
    }

    if cache_npz.exists() and cache_meta.exists():
        try:
            saved_meta = json.loads(cache_meta.read_text(encoding="utf-8"))
            if saved_meta == expected_meta:
                cached = np.load(cache_npz, allow_pickle=True)
                payload = {
                    "x_train": cached["x_train"].astype(np.float32),
                    "x_val": cached["x_val"].astype(np.float32),
                    "y_train": cached["y_train"].astype(np.float32),
                    "y_val": cached["y_val"].astype(np.float32),
                    "rot_train": cached["rot_train"].astype(np.float32),
                    "rot_val": cached["rot_val"].astype(np.float32),
                    "baseline_train": cached["baseline_train"].astype(np.float32),
                    "baseline_val": cached["baseline_val"].astype(np.float32),
                    "x_mean": cached["x_mean"].astype(np.float32),
                    "x_std": cached["x_std"].astype(np.float32),
                    "y_mean": cached["y_mean"].astype(np.float32),
                    "y_std": cached["y_std"].astype(np.float32),
                    "feature_cols": cached["feature_cols"].tolist(),
                    "target_cols": cached["target_cols"].tolist(),
                    "fast_fit_cols": cached["fast_fit_cols"].tolist(),
                    "phi_index": int(cached["phi_index"][0]),
                    "rotation_source": str(cached["rotation_source"][0]),
                    "training_mode": str(cached["training_mode"][0]),
                    "n_fastfit_failures": int(cached["n_fastfit_failures"][0]),
                }
                print(f"Loaded processed cache: {cache_npz}")
                return _build_bundle_from_cache_arrays(payload, batch_size, dataloader_workers, device, seed)
            print(f"Processed cache metadata mismatch, rebuilding: {cache_npz}")
        except Exception as exc:
            print(f"Failed to load processed cache, rebuilding {cache_npz}: {exc}")

    df = pd.read_csv(csv_path)
    feature_cols = base.detect_feature_cols(df.columns.tolist())
    if not feature_cols:
        raise ValueError(f"No supported legacy hit columns found in {csv_path}")

    target_cols = list(DEFAULT_TARGET_COLS)
    phi_index = target_cols.index("pca_phi")

    x_raw = df[feature_cols].to_numpy(dtype=np.float32)
    y_truth_raw = df[target_cols].to_numpy(dtype=np.float32)
    hit_groups = base.build_hit_groups(feature_cols)

    fast_fit_raw, n_fastfit_failures, _ = base.load_or_create_fastfit_cache(
        csv_path=csv_path,
        output_dir=cache_root,
        x_raw=x_raw,
        y_truth_raw=y_truth_raw,
        hit_groups=hit_groups,
        worker_count=max(1, int(fastfit_precompute_workers)),
    )

    x_proc = x_raw.copy()
    x_proc[x_proc == -999.0] = 0.0

    rotation_angles, has_rotation_anchor = base.compute_first_valid_hit_angles(x_proc, hit_groups)
    if training_mode == "canonical":
        x_proc = base.rotate_hit_xy_features(x_proc, hit_groups, rotation_angles)
        y_truth_proc = base.rotate_phi_column_to_canonical(y_truth_raw, phi_index, rotation_angles)
        fast_fit_proc = base.rotate_phi_column_to_canonical(fast_fit_raw, phi_index, rotation_angles)
        rotation_source = "first_valid_hit_xy_angle"
        print("========== Canonical phi rotation check ==========")
        print("Rotation source: first valid hit xy angle")
        print(f"Rows with a valid rotation anchor: {int(has_rotation_anchor.sum()):,}/{len(x_proc):,}")
        print("==================================================")
    else:
        y_truth_proc = y_truth_raw.copy()
        fast_fit_proc = fast_fit_raw.copy()
        rotation_angles = np.zeros(len(x_proc), dtype=np.float32)
        rotation_source = "none"

    residual_targets = base.build_residual_targets(y_truth_proc, fast_fit_proc, phi_index)
    x_full = np.concatenate([x_proc, fast_fit_proc], axis=1)

    n_rows = len(x_full)
    n_val = int(n_rows * val_fraction)
    rng = np.random.default_rng(seed=seed)
    indices = rng.permutation(n_rows)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    x_train = x_full[train_idx]
    x_val = x_full[val_idx]
    y_train = residual_targets[train_idx]
    y_val = residual_targets[val_idx]
    rot_train = rotation_angles[train_idx]
    rot_val = rotation_angles[val_idx]
    baseline_train = fast_fit_proc[train_idx]
    baseline_val = fast_fit_proc[val_idx]

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    x_train = (x_train - x_mean) / x_std
    x_val = (x_val - x_mean) / x_std

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std[y_std < 1e-8] = 1.0
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std

    np.savez_compressed(
        cache_npz,
        x_train=x_train.astype(np.float32),
        x_val=x_val.astype(np.float32),
        y_train=y_train.astype(np.float32),
        y_val=y_val.astype(np.float32),
        rot_train=rot_train.astype(np.float32),
        rot_val=rot_val.astype(np.float32),
        baseline_train=baseline_train.astype(np.float32),
        baseline_val=baseline_val.astype(np.float32),
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        y_mean=y_mean.astype(np.float32),
        y_std=y_std.astype(np.float32),
        feature_cols=np.asarray(feature_cols),
        target_cols=np.asarray(target_cols),
        fast_fit_cols=np.asarray(base.FAST_FIT_COLS),
        phi_index=np.asarray([phi_index], dtype=np.int64),
        rotation_source=np.asarray([rotation_source]),
        training_mode=np.asarray([training_mode]),
        n_fastfit_failures=np.asarray([n_fastfit_failures], dtype=np.int64),
    )
    cache_meta.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
    print(f"Saved processed cache: {cache_npz}")

    payload = {
        "x_train": x_train.astype(np.float32),
        "x_val": x_val.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "y_val": y_val.astype(np.float32),
        "rot_train": rot_train.astype(np.float32),
        "rot_val": rot_val.astype(np.float32),
        "baseline_train": baseline_train.astype(np.float32),
        "baseline_val": baseline_val.astype(np.float32),
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "fast_fit_cols": list(base.FAST_FIT_COLS),
        "phi_index": phi_index,
        "rotation_source": rotation_source,
        "training_mode": training_mode,
        "n_fastfit_failures": int(n_fastfit_failures),
    }
    return _build_bundle_from_cache_arrays(payload, batch_size, dataloader_workers, device, seed)


def build_checkpoint_metadata(data, input_dim: int, csv_path: Path, args, config: SweepConfig, report_text=None):
    metadata = {
        "target_cols": data.target_cols,
        "feature_cols": data.feature_cols + data.fast_fit_cols,
        "raw_hit_feature_cols": data.feature_cols,
        "fast_fit_feature_cols": data.fast_fit_cols,
        "model_type": "HeteroTrackNet",
        "input_dim": input_dim,
        "output_dim": len(data.target_cols),
        "y_mean": data.y_mean,
        "y_std": data.y_std,
        "x_mean": data.x_mean,
        "x_std": data.x_std,
        "hidden_layers": list(config.hidden_layers),
        "use_batchnorm": config.batchnorm,
        "dropout": config.dropout,
        "activation": config.activation,
        "batch_size": args.batch_size,
        "dataloader_workers": args.dataloader_workers,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "criterion": config.loss_name,
        "lr": config.lr,
        "epochs": config.epochs,
        "training_mode": data.training_mode,
        "canonical_phi": data.training_mode == "canonical",
        "canonical_rotation_source": data.rotation_source,
        "fast_fit_baseline": "peter_linearized_3d_helix_fit",
        "fast_fit_cache_dir": str(args.cache_dir),
        "target_definition": "truth_minus_fast_fit_residual",
        "source_csv": str(csv_path),
        "n_fastfit_failures": int(data.n_fastfit_failures),
        "config_name": config.name,
        "loss_name": config.loss_name,
    }
    if report_text is not None:
        metadata["report_text"] = report_text
    return metadata


def collect_metric_scores(y_true, y_pred, target_cols):
    scatter_scores = {}
    overlap_scores = {}
    for idx, name in enumerate(target_cols):
        scatter_scores[name] = compute_target_scatter_linearity(
            y_true=y_true,
            y_pred=y_pred,
            target_index=idx,
        )
        overlap_scores[name] = {
            "hist_overlap": compute_target_histogram_overlap(
                y_true=y_true,
                y_pred=y_pred,
                target_index=idx,
                target_cols=target_cols,
                bins=100,
            )
        }
    return scatter_scores, overlap_scores


def format_metric_report(scatter_scores, overlap_scores):
    lines = ["   Plot-quality scores:"]
    lines.append("      Scatter linearity:")
    for name, metric in scatter_scores.items():
        lines.append(
            f"         {name}: score={metric['score']:.6f} | "
            f"corr={metric['corr']:.6f} | slope={metric['slope']:.6f} | "
            f"intercept_penalty={metric['intercept_penalty']:.6f}"
        )
    lines.append("      Histogram overlap:")
    for name, metric in overlap_scores.items():
        lines.append(f"         {name}: overlap={metric['hist_overlap']:.6f}")
    return "\n".join(lines)


def _mean_for_targets(values_by_name, target_names):
    values = [float(values_by_name[name]) for name in target_names if name in values_by_name]
    if not values:
        return float("nan")
    return float(np.mean(values))


def summarize_snapshot_metrics(
    *,
    val_loss: float,
    per_target_mae: np.ndarray,
    per_target_rmse: np.ndarray,
    scatter_scores: dict,
    overlap_scores: dict,
    fast_scatter: dict,
    fast_overlap: dict,
    target_cols: list[str],
):
    mae_by_name = {name: float(per_target_mae[idx]) for idx, name in enumerate(target_cols)}
    rmse_by_name = {name: float(per_target_rmse[idx]) for idx, name in enumerate(target_cols)}
    scatter_by_name = {name: float(scatter_scores[name]["score"]) for name in target_cols}
    overlap_by_name = {name: float(overlap_scores[name]["hist_overlap"]) for name in target_cols}
    fast_scatter_by_name = {name: float(fast_scatter[name]["score"]) for name in target_cols}
    fast_overlap_by_name = {name: float(fast_overlap[name]["hist_overlap"]) for name in target_cols}

    no_dxy_targets = [name for name in target_cols if name != DXY_TARGET]
    no_phi_targets = [name for name in target_cols if name != PHI_TARGET]

    summary = {
        "val_loss": float(val_loss),
        "mean_mae": float(np.mean(per_target_mae)),
        "mean_rmse": float(np.mean(per_target_rmse)),
        "mean_mae_no_dxy": _mean_for_targets(mae_by_name, no_dxy_targets),
        "mean_rmse_no_dxy": _mean_for_targets(rmse_by_name, no_dxy_targets),
        "mean_mae_no_phi": _mean_for_targets(mae_by_name, no_phi_targets),
        "mean_rmse_no_phi": _mean_for_targets(rmse_by_name, no_phi_targets),
        "mean_scatter_score": _mean_for_targets(scatter_by_name, target_cols),
        "mean_scatter_no_dxy": _mean_for_targets(scatter_by_name, no_dxy_targets),
        "mean_hist_overlap": _mean_for_targets(overlap_by_name, target_cols),
        "mean_overlap_no_dxy": _mean_for_targets(overlap_by_name, no_dxy_targets),
        "mean_scatter_delta_vs_fastfit": _mean_for_targets(
            {name: scatter_by_name[name] - fast_scatter_by_name[name] for name in target_cols},
            target_cols,
        ),
        "mean_overlap_delta_vs_fastfit": _mean_for_targets(
            {name: overlap_by_name[name] - fast_overlap_by_name[name] for name in target_cols},
            target_cols,
        ),
    }

    for name in target_cols:
        summary[f"{name}_mae"] = mae_by_name[name]
        summary[f"{name}_rmse"] = rmse_by_name[name]
        summary[f"{name}_scatter"] = scatter_by_name[name]
        summary[f"{name}_overlap"] = overlap_by_name[name]
        summary[f"{name}_scatter_delta_vs_fastfit"] = scatter_by_name[name] - fast_scatter_by_name[name]
        summary[f"{name}_overlap_delta_vs_fastfit"] = overlap_by_name[name] - fast_overlap_by_name[name]

    return summary


def flatten_snapshot_metrics(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def build_baseline_comparison_report(y_true, y_pred, y_fast, target_cols):
    model_scatter, model_overlap = collect_metric_scores(y_true, y_pred, target_cols)
    fast_scatter, fast_overlap = collect_metric_scores(y_true, y_fast, target_cols)
    lines = ["   Baseline vs model comparison:"]
    lines.append("      Scatter score deltas (model - fast fit):")
    for name in target_cols:
        delta = model_scatter[name]["score"] - fast_scatter[name]["score"]
        lines.append(
            f"         {name}: model={model_scatter[name]['score']:.6f} | "
            f"fast={fast_scatter[name]['score']:.6f} | delta={delta:.6f}"
        )
    lines.append("      Histogram overlap deltas (model - fast fit):")
    for name in target_cols:
        delta = model_overlap[name]["hist_overlap"] - fast_overlap[name]["hist_overlap"]
        lines.append(
            f"         {name}: model={model_overlap[name]['hist_overlap']:.6f} | "
            f"fast={fast_overlap[name]['hist_overlap']:.6f} | delta={delta:.6f}"
        )
    return "\n".join(lines), model_scatter, model_overlap, fast_scatter, fast_overlap


def make_mode_val_diagnostic_plots(model, data, device, output_dir: Path, prefix: str, figure_label: str, show: bool):
    return base.make_mode_val_diagnostic_plots(
        model=model,
        val_loader=data.val_loader,
        device=device,
        y_mean_t=data.y_mean_t,
        y_std_t=data.y_std_t,
        target_cols=data.target_cols,
        phi_index=data.phi_index,
        training_mode=data.training_mode,
        output_dir=str(output_dir),
        prefix=prefix,
        bins=100,
        density=True,
        show=show,
        scatter_max_points=base.DIAGNOSTIC_SCATTER_MAX_POINTS,
        central_fraction=base.DIAGNOSTIC_CENTRAL_FRACTION,
        figure_label=figure_label,
    )


def collect_predictions(model, data, device):
    return base.collect_predictions_targets_and_sigma(
        model=model,
        val_loader=data.val_loader,
        device=device,
        y_mean_t=data.y_mean_t,
        y_std_t=data.y_std_t,
        phi_index=data.phi_index,
        training_mode=data.training_mode,
    )


def save_snapshot_outputs(model, data, device, output_dir: Path, report_prefix: str, show: bool):
    plot_paths = make_mode_val_diagnostic_plots(
        model=model,
        data=data,
        device=device,
        output_dir=output_dir,
        prefix=data.training_mode,
        figure_label=report_prefix,
        show=show,
    )
    y_pred, y_true, y_fast, _ = collect_predictions(model, data, device)
    plot_paths.update(
        base.make_fastfit_baseline_comparison_plots(
            y_true=y_true,
            y_pred=y_pred,
            y_fast=y_fast,
            target_cols=data.target_cols,
            output_dir=str(output_dir),
            prefix=data.training_mode,
            show=show,
            max_points=base.DIAGNOSTIC_SCATTER_MAX_POINTS,
            central_fraction=base.DIAGNOSTIC_CENTRAL_FRACTION,
            figure_label=report_prefix,
        )
    )
    return plot_paths, y_pred, y_true, y_fast


def config_run_dir(output_root: Path, mode: str, config: SweepConfig) -> Path:
    return output_root / mode / config.loss_name / config.name


def write_config_file(run_dir: Path, config: SweepConfig):
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def train_one_experiment(csv_path: Path, output_root: Path, cache_root: Path, args, device, training_mode: str, config: SweepConfig):
    run_dir = config_run_dir(output_root, training_mode, config)
    plot_dir = run_dir / "plots"
    final_plot_dir = plot_dir / "final"
    best_overall_plot_dir = plot_dir / "best_overall"
    best_val_loss_plot_dir = plot_dir / "best_val_loss"
    save_dir = run_dir / "saves"
    final_plot_dir.mkdir(parents=True, exist_ok=True)
    best_overall_plot_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss_plot_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    write_config_file(run_dir, config)

    data = load_or_create_processed_bundle(
        csv_path=csv_path,
        cache_root=cache_root,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        val_fraction=args.val_fraction,
        dataloader_workers=args.dataloader_workers,
        training_mode=training_mode,
        fastfit_precompute_workers=args.fastfit_precompute_workers,
    )

    input_dim = data.x_train.shape[1]
    model = HeteroTrackNet(
        input_dim=input_dim,
        hidden_layers=list(config.hidden_layers),
        output_dim=len(data.target_cols),
        use_batchnorm=config.batchnorm,
        dropout=config.dropout,
        activation=ACTIVATIONS[config.activation],
    ).to(device)

    criterion = LOSS_FNS[config.loss_name]
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=min(config.min_lr, config.lr * 0.1),
    )

    training_history = {"epoch": [], "train_loss": [], "val_loss": [], "val_mean_mae": [], "val_mean_rmse": [], "learning_rate": []}
    best_val_loss = float("inf")
    best_val_epoch = 0
    best_val_loss_plot_paths = {}
    best_val_loss_report_path = ""
    best_val_snapshot_metrics = {}
    best_overall_scatter_score = -float("inf")
    best_overall_scatter_epoch = 0
    best_overall_plot_paths = {}
    best_overall_report_path = ""
    best_overall_checkpoint_path = ""
    best_overall_snapshot_metrics = {}

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for xb, yb, _, _ in data.train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            mu, logvar = model(xb)
            loss = criterion(
                yb,
                mu,
                logvar,
                phi_index=data.phi_index,
                target_weights=base.TARGET_WEIGHTS,
                mean_weights=base.MEAN_WEIGHTS,
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(data.train_loader.dataset)

        model.eval()
        val_loss = 0.0
        total_val_mae = torch.zeros(len(data.target_cols), device=device)
        total_val_sq = torch.zeros(len(data.target_cols), device=device)
        total_count = 0
        pred_parts = []
        true_parts = []
        fast_parts = []

        with torch.no_grad():
            for xb, yb, rb, fastb in data.val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                rb = rb.to(device)
                fastb = fastb.to(device)

                mu, logvar = model(xb)
                loss = criterion(
                    yb,
                    mu,
                    logvar,
                    phi_index=data.phi_index,
                    target_weights=base.TARGET_WEIGHTS,
                    mean_weights=base.MEAN_WEIGHTS,
                )
                val_loss += loss.item() * xb.size(0)

                pred_resid_phys = mu * data.y_std_t + data.y_mean_t
                true_resid_phys = yb * data.y_std_t + data.y_mean_t
                pred_proc = base.reconstruct_full_targets(fastb, pred_resid_phys, data.phi_index)
                true_proc = base.reconstruct_full_targets(fastb, true_resid_phys, data.phi_index)

                pred_phys = base.maybe_recover_canonical_phi(pred_proc, rb, data.phi_index, training_mode)
                true_phys = base.maybe_recover_canonical_phi(true_proc, rb, data.phi_index, training_mode)
                fast_phys = base.maybe_recover_canonical_phi(fastb, rb, data.phi_index, training_mode)

                diff = pred_phys - true_phys
                if data.phi_index is not None:
                    diff[:, data.phi_index] = base.wrap_angle_torch(diff[:, data.phi_index])
                total_val_mae += diff.abs().sum(dim=0)
                total_val_sq += (diff ** 2).sum(dim=0)
                total_count += xb.size(0)

                pred_parts.append(pred_phys.detach().cpu())
                true_parts.append(true_phys.detach().cpu())
                fast_parts.append(fast_phys.detach().cpu())

        val_loss /= len(data.val_loader.dataset)
        per_target_mae = (total_val_mae / total_count).detach().cpu().numpy()
        per_target_rmse = np.sqrt((total_val_sq / total_count).detach().cpu().numpy())
        overall_val_mae = float(per_target_mae.mean())
        overall_val_rmse = float(per_target_rmse.mean())
        current_lr = optimizer.param_groups[0]["lr"]

        y_pred = torch.cat(pred_parts, dim=0).numpy()
        y_true = torch.cat(true_parts, dim=0).numpy()
        y_fast = torch.cat(fast_parts, dim=0).numpy()

        scatter_scores, overlap_scores = collect_metric_scores(y_true, y_pred, data.target_cols)
        baseline_report, model_scatter, model_overlap, fast_scatter, fast_overlap = build_baseline_comparison_report(
            y_true,
            y_pred,
            y_fast,
            data.target_cols,
        )
        mean_scatter_score = float(np.mean([metric["score"] for metric in scatter_scores.values()]))
        mean_hist_overlap = float(np.mean([metric["hist_overlap"] for metric in overlap_scores.values()]))
        overlap_target = compute_target_histogram_overlap(
            y_true,
            y_pred,
            base.OVERLAP_TARGET_INDEX,
            data.target_cols,
            bins=100,
        )

        report = format_epoch_report(
            epoch,
            config.epochs,
            train_loss,
            val_loss,
            overall_val_mae,
            overall_val_rmse,
            per_target_mae,
            per_target_rmse,
            data.target_cols,
        )
        overall_report = (
            f"   Sweep model-selection scores:\n"
            f"      mean_scatter_score: {mean_scatter_score:.6f}\n"
            f"      mean_hist_overlap: {mean_hist_overlap:.6f}\n"
            f"      {data.target_cols[base.OVERLAP_TARGET_INDEX]} overlap: {overlap_target:.6f}"
        )
        full_report = f"{report}\n{overall_report}\n{format_metric_report(scatter_scores, overlap_scores)}\n{baseline_report}"
        snapshot_metrics = summarize_snapshot_metrics(
            val_loss=val_loss,
            per_target_mae=per_target_mae,
            per_target_rmse=per_target_rmse,
            scatter_scores=model_scatter,
            overlap_scores=model_overlap,
            fast_scatter=fast_scatter,
            fast_overlap=fast_overlap,
            target_cols=data.target_cols,
        )

        training_history["epoch"].append(epoch + 1)
        training_history["train_loss"].append(train_loss)
        training_history["val_loss"].append(val_loss)
        training_history["val_mean_mae"].append(overall_val_mae)
        training_history["val_mean_rmse"].append(overall_val_rmse)
        training_history["learning_rate"].append(current_lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch + 1
            best_val_snapshot_metrics = snapshot_metrics.copy()
            metadata = build_checkpoint_metadata(data, input_dim, csv_path, args, config, report_text=full_report)
            metadata.update({"checkpoint_type": "best_val_loss", "val_loss": float(val_loss), "mode": training_mode})
            save_model_checkpoint(str(save_dir / "best_val_loss.pt"), model, optimizer, scheduler, epoch + 1, metadata)
            best_val_loss_plot_paths, _, _, _ = save_snapshot_outputs(
                model=model,
                data=data,
                device=device,
                output_dir=best_val_loss_plot_dir,
                report_prefix=f"{training_mode} | {config.name} | best_val_loss",
                show=False,
            )
            best_val_loss_report_path = str(best_val_loss_plot_dir / "best_val_loss_training_report.txt")
            Path(best_val_loss_report_path).write_text(
                "BEST VAL LOSS MODEL REPORT\n"
                + "=" * 80
                + f"\nmode: {training_mode}\nconfig: {config.name}\nepoch: {epoch + 1}\nmetric_value: {float(val_loss):.6f}\n\n{full_report}\n",
                encoding="utf-8",
            )

        if mean_scatter_score > best_overall_scatter_score:
            best_overall_scatter_score = mean_scatter_score
            best_overall_scatter_epoch = epoch + 1
            best_overall_checkpoint_path = str(save_dir / "best_overall.pt")
            best_overall_snapshot_metrics = snapshot_metrics.copy()
            metadata = build_checkpoint_metadata(data, input_dim, csv_path, args, config, report_text=full_report)
            metadata.update(
                {
                    "checkpoint_type": base.BEST_OVERALL_SCATTER_TAG,
                    "metric_tag": base.BEST_OVERALL_SCATTER_TAG,
                    "metric_value": float(mean_scatter_score),
                    "mode": training_mode,
                }
            )
            save_model_checkpoint(best_overall_checkpoint_path, model, optimizer, scheduler, epoch + 1, metadata)
            best_overall_plot_paths, _, _, _ = save_snapshot_outputs(
                model=model,
                data=data,
                device=device,
                output_dir=best_overall_plot_dir,
                report_prefix=f"{training_mode} | {config.name} | best_overall",
                show=False,
            )
            best_overall_report_path = str(best_overall_plot_dir / "best_overall_training_report.txt")
            Path(best_overall_report_path).write_text(
                "BEST OVERALL MODEL REPORT\n"
                + "=" * 80
                + f"\nmode: {training_mode}\nconfig: {config.name}\nepoch: {epoch + 1}\nmetric_value: {float(mean_scatter_score):.6f}\n\n{full_report}\n",
                encoding="utf-8",
            )

        print(
            f"[{training_mode} | {config.name}] Epoch {epoch + 1}/{config.epochs} | "
            f"train {train_loss:.6f} | val {val_loss:.6f} | mean MAE {overall_val_mae:.6f} | "
            f"mean RMSE {overall_val_rmse:.6f} | mean scatter {mean_scatter_score:.6f} | "
            f"mean overlap {mean_hist_overlap:.6f}"
        )
        scheduler.step()

    history_plot_paths = make_training_history_plots(
        training_history,
        output_dir=str(final_plot_dir),
        prefix=training_mode,
        show=False,
        figure_label=f"{training_mode} | {config.name} | final",
    )

    y_pred, y_true, y_fast, _ = collect_predictions(model, data, device)
    baseline_report, model_scatter, model_overlap, fast_scatter, fast_overlap = build_baseline_comparison_report(
        y_true,
        y_pred,
        y_fast,
        data.target_cols,
    )
    final_per_target_mae = np.mean(np.abs(y_pred - y_true), axis=0)
    final_per_target_rmse = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))
    final_mean_scatter_score = float(np.mean([metric["score"] for metric in model_scatter.values()]))
    final_mean_hist_overlap = float(np.mean([metric["hist_overlap"] for metric in model_overlap.values()]))
    final_snapshot_metrics = summarize_snapshot_metrics(
        val_loss=float(training_history["val_loss"][-1]),
        per_target_mae=final_per_target_mae,
        per_target_rmse=final_per_target_rmse,
        scatter_scores=model_scatter,
        overlap_scores=model_overlap,
        fast_scatter=fast_scatter,
        fast_overlap=fast_overlap,
        target_cols=data.target_cols,
    )
    final_epoch_report = format_epoch_report(
        config.epochs - 1,
        config.epochs,
        training_history["train_loss"][-1],
        training_history["val_loss"][-1],
        training_history["val_mean_mae"][-1],
        training_history["val_mean_rmse"][-1],
        final_per_target_mae,
        final_per_target_rmse,
        data.target_cols,
    )
    final_full_report = (
        f"{final_epoch_report}\n"
        f"   Sweep model-selection scores:\n"
        f"      mean_scatter_score: {final_mean_scatter_score:.6f}\n"
        f"      mean_hist_overlap: {final_mean_hist_overlap:.6f}\n"
        f"{format_metric_report(model_scatter, model_overlap)}\n"
        f"{baseline_report}"
    )

    final_metadata = build_checkpoint_metadata(data, input_dim, csv_path, args, config, report_text=final_full_report)
    final_metadata.update(
        {
            "checkpoint_type": "final_model",
            "best_val_loss": float(best_val_loss),
            "best_val_epoch": best_val_epoch,
            "mode": training_mode,
        }
    )
    save_model_checkpoint(str(save_dir / "final_model.pt"), model, optimizer, scheduler, config.epochs, final_metadata)

    final_plot_paths, _, _, _ = save_snapshot_outputs(
        model=model,
        data=data,
        device=device,
        output_dir=final_plot_dir,
        report_prefix=f"{training_mode} | {config.name} | final",
        show=args.show_plots,
    )
    final_report_path = final_plot_dir / "final_training_report.txt"
    final_report_path.write_text(final_full_report + "\n", encoding="utf-8")

    with open(final_plot_dir / "fastfit_vs_model_report.txt", "w", encoding="utf-8") as handle:
        handle.write("FAST FIT VS MODEL REPORT\n")
        handle.write("=" * 80 + "\n")
        handle.write(baseline_report)
        handle.write("\n\nModel scatter scores\n")
        for name, metric in model_scatter.items():
            handle.write(
                f"{name}: score={metric['score']:.6f} corr={metric['corr']:.6f} slope={metric['slope']:.6f}\n"
            )
        handle.write("\nModel histogram overlap\n")
        for name, metric in model_overlap.items():
            handle.write(f"{name}: overlap={metric['hist_overlap']:.6f}\n")
        handle.write("\nFast-fit scatter scores\n")
        for name, metric in fast_scatter.items():
            handle.write(
                f"{name}: score={metric['score']:.6f} corr={metric['corr']:.6f} slope={metric['slope']:.6f}\n"
            )
        handle.write("\nFast-fit histogram overlap\n")
        for name, metric in fast_overlap.items():
            handle.write(f"{name}: overlap={metric['hist_overlap']:.6f}\n")

    if args.print_final_samples:
        base.print_fastfit_final_validation_samples(
            model,
            data.val_loader,
            device,
            data.y_mean_t,
            data.y_std_t,
            data.target_cols,
            data.phi_index,
            training_mode,
            num_examples=5,
        )

    mode_row = {
        "mode": training_mode,
        "config_name": config.name,
        "loss_name": config.loss_name,
        "activation": config.activation,
        "hidden_layers": json.dumps(list(config.hidden_layers)),
        "batchnorm": bool(config.batchnorm),
        "dropout": float(config.dropout),
        "lr": float(config.lr),
        "epochs": int(config.epochs),
        "csv_path": str(csv_path),
        "rows": int(len(data.x_train) + len(data.x_val)),
        "train_rows": int(len(data.x_train)),
        "val_rows": int(len(data.x_val)),
        "input_dim": int(input_dim),
        "best_val_loss": float(best_val_loss),
        "best_val_epoch": int(best_val_epoch),
        "final_val_loss": float(training_history["val_loss"][-1]),
        "final_val_mean_mae": float(training_history["val_mean_mae"][-1]),
        "final_val_mean_rmse": float(training_history["val_mean_rmse"][-1]),
        "best_overall_scatter_epoch": int(best_overall_scatter_epoch),
        "best_overall_scatter_score": float(best_overall_scatter_score),
        "run_dir": str(run_dir),
        "final_plot_dir": str(final_plot_dir),
        "best_val_plot_dir": str(best_val_loss_plot_dir),
        "best_overall_plot_dir": str(best_overall_plot_dir),
        "best_val_checkpoint_path": str(save_dir / "best_val_loss.pt"),
        "best_overall_checkpoint_path": best_overall_checkpoint_path,
        "final_checkpoint_path": str(save_dir / "final_model.pt"),
        "final_report_path": str(final_report_path),
        "best_val_report_path": best_val_loss_report_path,
        "best_overall_report_path": best_overall_report_path,
        "history_plot_count": int(len(history_plot_paths)),
        "final_plot_count": int(len(final_plot_paths)),
        "best_val_plot_count": int(len(best_val_loss_plot_paths)),
        "best_overall_plot_count": int(len(best_overall_plot_paths)),
        "n_fastfit_failures": int(data.n_fastfit_failures),
    }
    mode_row.update(flatten_snapshot_metrics("best_val", best_val_snapshot_metrics))
    mode_row.update(flatten_snapshot_metrics("best_overall", best_overall_snapshot_metrics))
    mode_row.update(flatten_snapshot_metrics("final", final_snapshot_metrics))
    pd.DataFrame([mode_row]).to_csv(run_dir / "run_summary.csv", index=False)
    return mode_row


def build_subprocess_command(args, mode: str, config_name: str, device_name: str) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-mode",
        mode,
        "--single-config",
        config_name,
        "--data-path",
        str(args.data_path),
        "--output-dir",
        str(args.output_dir),
        "--cache-dir",
        str(args.cache_dir),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--val-fraction",
        str(args.val_fraction),
        "--device",
        device_name,
        "--dataloader-workers",
        str(args.dataloader_workers),
        "--fastfit-precompute-workers",
        str(args.fastfit_precompute_workers),
        "--epochs-override",
        str(args.epochs_override),
        "--no-show-plots",
        "--no-print-final-samples",
    ]
    if args.show_plots:
        cmd[-2] = "--show-plots"
    if args.print_final_samples:
        cmd[-1] = "--print-final-samples"
    return cmd


def run_experiments_concurrently(args, output_root: Path, experiments: list[tuple[str, SweepConfig]]):
    device_slots = parse_device_slots(args.device_slots)
    max_parallel = max(1, min(args.max_concurrent, len(device_slots)))
    free_slots = device_slots[:max_parallel]
    pending = list(experiments)
    running = []

    print(f"Using device slots: {free_slots}")

    while pending or running:
        while pending and free_slots:
            mode, config = pending.pop(0)
            device_name = free_slots.pop(0)
            run_dir = config_run_dir(output_root, mode, config)
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "train.log"
            status_path = run_dir / "run_status.txt"
            status_path.write_text(f"running on {device_name}\n", encoding="utf-8")
            log_handle = open(log_path, "w", encoding="utf-8")
            cmd = build_subprocess_command(args, mode, config.name, device_name)
            print(f"Launching {mode} | {config.name} on {device_name}")
            proc = subprocess.Popen(
                cmd,
                cwd=str(Path(__file__).resolve().parent),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running.append(
                {
                    "process": proc,
                    "log_handle": log_handle,
                    "device_name": device_name,
                    "mode": mode,
                    "config_name": config.name,
                    "status_path": status_path,
                }
            )

        time.sleep(2.0)
        still_running = []
        for item in running:
            retcode = item["process"].poll()
            if retcode is None:
                still_running.append(item)
                continue
            item["log_handle"].close()
            item["status_path"].write_text(
                f"finished rc={retcode} on {item['device_name']}\n",
                encoding="utf-8",
            )
            free_slots.append(item["device_name"])
            if retcode == 0:
                print(f"Finished {item['mode']} | {item['config_name']} on {item['device_name']}")
            else:
                print(
                    f"Run failed for {item['mode']} | {item['config_name']} on {item['device_name']} "
                    f"with rc={retcode}"
                )
        running = still_running


def rank_summary_rows(
    df: pd.DataFrame,
    *,
    lower_is_better: list[str],
    higher_is_better: list[str],
    tie_breakers: list[tuple[str, bool]],
) -> pd.DataFrame:
    ranked = df.copy()
    rank_columns = []

    for column in lower_is_better:
        rank_col = f"rank_{column}"
        ranked[rank_col] = ranked[column].rank(method="min", ascending=True)
        rank_columns.append(rank_col)

    for column in higher_is_better:
        rank_col = f"rank_{column}"
        ranked[rank_col] = ranked[column].rank(method="min", ascending=False)
        rank_columns.append(rank_col)

    ranked["rank_score"] = ranked[rank_columns].mean(axis=1)

    sort_columns = ["rank_score"]
    ascending = [True]
    for column, high_is_better in tie_breakers:
        sort_columns.append(column)
        ascending.append(not high_is_better)

    ranked = ranked.sort_values(by=sort_columns, ascending=ascending).reset_index(drop=True)
    ranked.insert(0, "overall_rank", np.arange(1, len(ranked) + 1))
    return ranked


def rank_balanced_final_rows(df: pd.DataFrame) -> pd.DataFrame:
    return rank_summary_rows(
        df,
        lower_is_better=[
            "final_mean_mae",
            "final_mean_rmse",
            "final_pca_dxy_rmse",
            "final_pca_dz_rmse",
            "final_mean_rmse_no_dxy",
        ],
        higher_is_better=[
            "final_mean_overlap_delta_vs_fastfit",
            "final_mean_hist_overlap",
            "final_mean_scatter_score",
        ],
        tie_breakers=[
            ("final_mean_rmse", False),
            ("final_pca_dxy_rmse", False),
            ("final_mean_overlap_delta_vs_fastfit", True),
        ],
    )


def rank_best_val_physics_rows(df: pd.DataFrame) -> pd.DataFrame:
    return rank_summary_rows(
        df,
        lower_is_better=[
            "best_val_val_loss",
            "best_val_mean_mae",
            "best_val_mean_rmse",
            "best_val_pca_dxy_rmse",
            "best_val_pca_dz_rmse",
        ],
        higher_is_better=[
            "best_val_mean_hist_overlap",
        ],
        tie_breakers=[
            ("best_val_mean_rmse", False),
            ("best_val_pca_dxy_rmse", False),
            ("best_val_mean_hist_overlap", True),
        ],
    )


def rank_fastfit_gain_rows(df: pd.DataFrame) -> pd.DataFrame:
    return rank_summary_rows(
        df,
        lower_is_better=[
            "final_pca_dxy_rmse",
            "final_pca_dz_rmse",
        ],
        higher_is_better=[
            "final_mean_overlap_delta_vs_fastfit",
            "final_mean_scatter_delta_vs_fastfit",
            "final_pca_dxy_overlap_delta_vs_fastfit",
            "final_pca_dz_overlap_delta_vs_fastfit",
        ],
        tie_breakers=[
            ("final_mean_overlap_delta_vs_fastfit", True),
            ("final_pca_dxy_overlap_delta_vs_fastfit", True),
            ("final_mean_rmse", False),
        ],
    )


def rank_shape_rows(df: pd.DataFrame) -> pd.DataFrame:
    return rank_summary_rows(
        df,
        lower_is_better=[],
        higher_is_better=[
            "final_mean_scatter_score",
            "final_mean_hist_overlap",
            "final_mean_scatter_no_dxy",
            "final_mean_overlap_no_dxy",
        ],
        tie_breakers=[
            ("final_mean_scatter_score", True),
            ("final_mean_hist_overlap", True),
            ("final_mean_rmse", False),
        ],
    )


def rank_legacy_rows(df: pd.DataFrame) -> pd.DataFrame:
    return rank_summary_rows(
        df,
        lower_is_better=["best_val_loss", "final_val_mean_mae", "final_val_mean_rmse"],
        higher_is_better=["final_mean_scatter_score", "final_mean_hist_overlap"],
        tie_breakers=[
            ("final_mean_scatter_score", True),
            ("final_mean_hist_overlap", True),
            ("best_val_loss", False),
        ],
    )


def write_summary_text(summary_path: Path, title: str, ranked: pd.DataFrame, metric_prefix: str):
    lines = [title, "=" * len(title), ""]
    for _, row in ranked.iterrows():
        final_prefix = metric_prefix
        lines.append(
            f"#{int(row['overall_rank'])} | mode={row['mode']} | config={row['config_name']} | "
            f"loss={row['loss_name']} | "
            f"{final_prefix}_mae={row[f'{final_prefix}_mean_mae']:.6f} | "
            f"{final_prefix}_rmse={row[f'{final_prefix}_mean_rmse']:.6f} | "
            f"{final_prefix}_dxy_rmse={row[f'{final_prefix}_pca_dxy_rmse']:.6f} | "
            f"{final_prefix}_dz_rmse={row[f'{final_prefix}_pca_dz_rmse']:.6f} | "
            f"{final_prefix}_scatter={row[f'{final_prefix}_mean_scatter_score']:.6f} | "
            f"{final_prefix}_overlap={row[f'{final_prefix}_mean_hist_overlap']:.6f}"
        )
        if f"{final_prefix}_mean_overlap_delta_vs_fastfit" in row:
            lines.append(
                f"   delta_vs_fastfit: mean_overlap={row[f'{final_prefix}_mean_overlap_delta_vs_fastfit']:.6f} | "
                f"mean_scatter={row[f'{final_prefix}_mean_scatter_delta_vs_fastfit']:.6f} | "
                f"dxy_overlap={row[f'{final_prefix}_pca_dxy_overlap_delta_vs_fastfit']:.6f}"
            )
        lines.append(f"run_dir: {row['run_dir']}")
        lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def summarize_sweep(output_root: Path):
    rows = []
    for run_summary_path in output_root.glob("*/*/*/run_summary.csv"):
        rows.extend(pd.read_csv(run_summary_path).to_dict(orient="records"))
    if not rows:
        return

    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    all_df = pd.DataFrame(rows)
    all_df.to_csv(summary_dir / "all_runs.csv", index=False)

    summary_specs = [
        ("ranked_balanced_final", "SWEEP2 BALANCED FINAL RANKING", rank_balanced_final_rows, "final"),
        ("ranked_best_val_physics", "SWEEP2 BEST-VAL PHYSICS RANKING", rank_best_val_physics_rows, "best_val"),
        ("ranked_fastfit_gain", "SWEEP2 FASTFIT-GAIN RANKING", rank_fastfit_gain_rows, "final"),
        ("ranked_shape", "SWEEP2 SHAPE RANKING", rank_shape_rows, "final"),
        ("ranked_legacy", "LEGACY MIXED RANKING", rank_legacy_rows, "final"),
    ]

    for stem, title, ranker, metric_prefix in summary_specs:
        ranked_all = ranker(all_df)
        ranked_all.to_csv(summary_dir / f"{stem}.csv", index=False)
        write_summary_text(summary_dir / f"{stem}.txt", title, ranked_all, metric_prefix)

    for mode, mode_df in all_df.groupby("mode"):
        mode_df = mode_df.reset_index(drop=True)
        for stem, title, ranker, metric_prefix in summary_specs:
            ranked_mode = ranker(mode_df)
            ranked_mode.to_csv(summary_dir / f"{stem}_{mode}.csv", index=False)
            write_summary_text(
                summary_dir / f"{stem}_{mode}.txt",
                f"{title} | {mode.upper()}",
                ranked_mode,
                metric_prefix,
            )

    for loss_name, loss_df in all_df.groupby("loss_name"):
        safe_loss_name = loss_name.replace("/", "_")
        ranked_loss = rank_balanced_final_rows(loss_df.reset_index(drop=True))
        ranked_loss.to_csv(summary_dir / f"ranked_balanced_final_{safe_loss_name}.csv", index=False)
        write_summary_text(
            summary_dir / f"ranked_balanced_final_{safe_loss_name}.txt",
            f"SWEEP2 BALANCED FINAL RANKING | LOSS={loss_name}",
            ranked_loss,
            "final",
        )


def prepare_mode_caches(csv_path: Path, cache_root: Path, args, modes: list[str]):
    cpu_device = torch.device("cpu")
    for mode in modes:
        print(f"Preparing shared cache for mode: {mode}")
        load_or_create_processed_bundle(
            csv_path=csv_path,
            cache_root=cache_root,
            batch_size=1,
            seed=args.seed,
            device=cpu_device,
            val_fraction=args.val_fraction,
            dataloader_workers=0,
            training_mode=mode,
            fastfit_precompute_workers=args.fastfit_precompute_workers,
        )


def main():
    args = parse_args()
    output_root = Path(args.output_dir)
    cache_root = Path(args.cache_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.data_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV does not exist: {csv_path}")

    if args.single_mode and args.single_config:
        device = resolve_device(args.device)
        set_seed(args.seed)
        config = get_config_with_overrides(CONFIG_MAP[args.single_config], args)
        print(f"Device set to {device}")
        print(f"Training mode: {args.single_mode}")
        print(f"Config: {config.name}")
        train_one_experiment(csv_path, output_root, cache_root, args, device, args.single_mode, config)
        return

    modes = parse_modes(args.modes)
    configs = [get_config_with_overrides(cfg, args) for cfg in parse_configs(args.configs)]
    experiments = [(mode, config) for mode in modes for config in configs]
    prepare_mode_caches(csv_path, cache_root, args, modes)

    if args.max_concurrent <= 1 or len(experiments) == 1:
        device = resolve_device(args.device)
        set_seed(args.seed)
        print(f"Device set to {device}")
        for mode, config in experiments:
            print(f"Training mode: {mode} | Config: {config.name}")
            train_one_experiment(csv_path, output_root, cache_root, args, device, mode, config)
    else:
        run_experiments_concurrently(args, output_root, experiments)

    summarize_sweep(output_root)


if __name__ == "__main__":
    main()
