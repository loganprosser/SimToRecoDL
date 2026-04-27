import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from helpers import (
    format_epoch_report,
    save_golden_model,
    save_model_checkpoint,
    write_final_golden_summary,
)
from helpers_canonical_phi import recover_phi_from_canonical
from helpers_data import DEFAULT_TARGET_COLS, build_hit_feature_cols, set_seed
from helpers_vis import (
    compute_plot_quality_scores,
    compute_target_histogram_overlap,
    format_plot_quality_report,
    make_training_history_plots,
    plot_distance_distribution,
    plot_overlap_distributions,
    plot_pred_vs_true_scatter,
    plot_pull_distributions,
)
from loss import hetero_gaussian_nll_with_phi
from model import HeteroTrackNet

DEFAULT_DATA_PATH = "/nfs/cms/tracktrigger/logan/root/simvrico/SimToRecoDL/outputCSVs/filtered_particles.csv"
DEFAULT_OUTPUT_DIR = "auto_fastfit_residual"
DEFAULT_MAX_CONCURRENT = 2
DEFAULT_DEVICE_SLOTS = "cuda:0=1,cuda:1=1"
DEFAULT_DATALOADER_WORKERS = 0
DEFAULT_TRACK_OVERLAP_GOLDEN = False
TRAINING_MODES = ("canonical", "raw")

EPOCHS = 750
BATCH_SIZE = 256
HIDDEN_LAYERS = [512, 512, 256]
CRITERION = hetero_gaussian_nll_with_phi
TARGET_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
MEAN_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
BATCH_NORM = False
DROPOUT = 0.0
SEED = 42
VAL_FRACTION = 0.2
OVERLAP_TARGET_INDEX = 3
DIAGNOSTIC_CENTRAL_FRACTION = 0.99
DIAGNOSTIC_SCATTER_MAX_POINTS = None
GOLDEN_SCATTER_PREFIX = "best_scatter_linear_"
GOLDEN_OVERLAP_PREFIX = "best_overlap_cover_"
BEST_OVERALL_SCATTER_TAG = "best_overall_scatter"
BEST_OVERALL_OVERLAP_TAG = "best_overall_overlap"

FIELD_ORDER = {"x": 0, "y": 1, "z": 2, "r": 3, "mask": 4}
FAST_FIT_COLS = ["fast_pca_c", "fast_pca_eta", "fast_pca_phi", "fast_pca_dxy", "fast_pca_dz"]

B_FIELD_T = 3.811
PCA_C_TO_R_INV = 0.003 * B_FIELD_T


@dataclass
class FastFitDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    x_train: torch.Tensor
    x_val: torch.Tensor
    y_train: torch.Tensor
    y_val: torch.Tensor
    rot_train: torch.Tensor
    rot_val: torch.Tensor
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    y_mean_t: torch.Tensor
    y_std_t: torch.Tensor
    feature_cols: list[str]
    target_cols: list[str]
    fast_fit_cols: list[str]
    phi_index: int | None
    rotation_source: str
    training_mode: str
    n_fastfit_failures: int


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train fast-fit residual DNNs in canonical and raw frames."
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to the CSV to train on.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for runs.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--dataloader-workers", type=int, default=DEFAULT_DATALOADER_WORKERS)
    parser.add_argument("--show-plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print-final-samples", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--track-overlap-golden",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TRACK_OVERLAP_GOLDEN,
    )
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--device-slots", default=DEFAULT_DEVICE_SLOTS)
    parser.add_argument("--device", default="", help="Device override for a single training run.")
    parser.add_argument(
        "--modes",
        default=",".join(TRAINING_MODES),
        help="Comma-separated training modes to run. Supported: canonical, raw.",
    )
    parser.add_argument("--single-mode", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def wrap_angle_np(values):
    return np.arctan2(np.sin(values), np.cos(values))


def wrap_angle_torch(values):
    return torch.atan2(torch.sin(values), torch.cos(values))


def detect_feature_cols(columns: list[str]) -> list[str]:
    legacy_cols = []
    legacy_col_set = set(columns)
    candidate_cols = build_hit_feature_cols(n_layers=6)
    if all(col in legacy_col_set for col in candidate_cols):
        for col in candidate_cols:
            parts = col.split("_")
            bucket = int(parts[1])
            field = parts[2]
            legacy_cols.append((bucket, FIELD_ORDER[field], col))
        legacy_cols.sort()
        return [col for _, _, col in legacy_cols]
    return []


def build_hit_groups(feature_cols: list[str], n_layers: int = 6) -> list[dict]:
    col_to_idx = {col: idx for idx, col in enumerate(feature_cols)}
    groups = []
    for layer in range(1, n_layers + 1):
        groups.append(
            {
                "x": col_to_idx[f"hit_{layer}_x"],
                "y": col_to_idx[f"hit_{layer}_y"],
                "z": col_to_idx[f"hit_{layer}_z"],
                "mask": col_to_idx[f"hit_{layer}_mask"],
            }
        )
    return groups


def compute_first_valid_hit_angles(x: np.ndarray, hit_groups: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    angles = np.zeros(x.shape[0], dtype=np.float32)
    assigned = np.zeros(x.shape[0], dtype=bool)
    for cols in hit_groups:
        x_vals = x[:, cols["x"]]
        y_vals = x[:, cols["y"]]
        valid = x[:, cols["mask"]] > 0.5
        use = valid & ~assigned
        angles[use] = np.arctan2(y_vals[use], x_vals[use])
        assigned[use] = True
    return angles, assigned


def rotate_hit_xy_features(x: np.ndarray, hit_groups: list[dict], rotation_angles: np.ndarray) -> np.ndarray:
    x_rot = x.copy()
    cos_a = np.cos(rotation_angles)
    sin_a = np.sin(rotation_angles)
    for cols in hit_groups:
        x_old = x[:, cols["x"]].copy()
        y_old = x[:, cols["y"]].copy()
        x_rot[:, cols["x"]] = cos_a * x_old + sin_a * y_old
        x_rot[:, cols["y"]] = -sin_a * x_old + cos_a * y_old
    return x_rot


def rotate_phi_column_to_canonical(values: np.ndarray, phi_index: int | None, rotation_angles: np.ndarray) -> np.ndarray:
    out = values.copy()
    if phi_index is not None:
        out[:, phi_index] = wrap_angle_np(out[:, phi_index] - rotation_angles)
    return out


def _prepare_sigma_values(hit_sigma, n_hits, name):
    if np.isscalar(hit_sigma):
        sigma_values = np.full(n_hits, float(hit_sigma), dtype=float)
    else:
        sigma_values = np.asarray(hit_sigma, dtype=float)
        if sigma_values.shape != (n_hits,):
            raise ValueError(f"{name} must be a scalar or one value per hit.")
    if np.any(sigma_values <= 0.0):
        raise ValueError(f"{name} must be strictly positive.")
    return sigma_values


def infer_pca_c_sign(xy_hits, x_center, y_center):
    center = np.array([x_center, y_center], dtype=float)
    cross_values = []
    for hit_idx in range(len(xy_hits) - 1):
        tangent = xy_hits[hit_idx + 1] - xy_hits[hit_idx]
        radius_vector = xy_hits[hit_idx] - center
        cross_values.append(tangent[0] * radius_vector[1] - tangent[1] * radius_vector[0])
    mean_cross = np.mean(cross_values)
    if np.isclose(mean_cross, 0.0):
        return 1.0
    return np.sign(mean_cross)


def compute_signed_arc_lengths(x_values, y_values, x_center, y_center, radius):
    raw_angles = np.arctan2(y_values - y_center, x_values - x_center)
    unwrapped_angles = np.unwrap(raw_angles)
    arc_lengths = radius * (unwrapped_angles - unwrapped_angles[0])
    flip_applied = False
    if len(arc_lengths) > 1 and np.mean(np.diff(arc_lengths)) < 0.0:
        arc_lengths *= -1.0
        flip_applied = True
    return arc_lengths, unwrapped_angles, flip_applied


def estimate_s_pca(x_center, y_center, radius, unwrapped_angles, flip_applied):
    center_norm = np.hypot(x_center, y_center)
    if np.isclose(center_norm, 0.0):
        return 0.0
    x_pca = x_center - radius * x_center / center_norm
    y_pca = y_center - radius * y_center / center_norm
    pca_angle_raw = np.arctan2(y_pca - y_center, x_pca - x_center)
    mean_angle = np.mean(unwrapped_angles)
    angle_shift = 2.0 * np.pi * np.round((mean_angle - pca_angle_raw) / (2.0 * np.pi))
    pca_angle_unwrapped = pca_angle_raw + angle_shift
    s_pca = radius * (pca_angle_unwrapped - unwrapped_angles[0])
    if flip_applied:
        s_pca *= -1.0
    return s_pca


def fit_helix_xyz_pca_linearized(xyz_hits, hit_sigma_xy=1.0, hit_sigma_z=1.0):
    x_values = xyz_hits[:, 0]
    y_values = xyz_hits[:, 1]
    z_values = xyz_hits[:, 2]
    n_hits = len(x_values)

    sigma_xy_values = _prepare_sigma_values(hit_sigma_xy, n_hits, "hit_sigma_xy")
    sigma_z_values = _prepare_sigma_values(hit_sigma_z, n_hits, "hit_sigma_z")

    design_xy = np.column_stack([x_values, y_values, np.ones_like(x_values)])
    target_xy = -(x_values**2 + y_values**2)

    weighted_design_xy = design_xy / sigma_xy_values[:, None]
    weighted_target_xy = target_xy / sigma_xy_values
    coeffs_xy, *_ = np.linalg.lstsq(weighted_design_xy, weighted_target_xy, rcond=None)

    a_coef, b_coef, c_coef = coeffs_xy
    x_center = -0.5 * a_coef
    y_center = -0.5 * b_coef
    radius_sq = x_center**2 + y_center**2 - c_coef
    if radius_sq <= 0.0:
        raise ValueError("Degenerate linearized fit: non-positive radius^2.")

    radius = np.sqrt(radius_sq)
    abs_pca_c = 1.0 / (PCA_C_TO_R_INV * radius)
    pca_phi_geom = wrap_angle_np(np.array([np.arctan2(x_center, -y_center)]))[0]
    pca_dxy_geom = radius - np.hypot(x_center, y_center)

    pca_c_sign = infer_pca_c_sign(xyz_hits[:, :2], x_center, y_center)
    if pca_c_sign < 0.0:
        pca_phi = wrap_angle_np(np.array([pca_phi_geom + np.pi]))[0]
        pca_dxy = -pca_dxy_geom
    else:
        pca_phi = pca_phi_geom
        pca_dxy = pca_dxy_geom

    arc_lengths, unwrapped_angles, flip_applied = compute_signed_arc_lengths(
        x_values, y_values, x_center, y_center, radius
    )
    s_pca = estimate_s_pca(x_center, y_center, radius, unwrapped_angles, flip_applied)
    arc_lengths_from_pca = arc_lengths - s_pca

    design_z = np.column_stack([np.ones_like(arc_lengths_from_pca), arc_lengths_from_pca])
    weighted_design_z = design_z / sigma_z_values[:, None]
    weighted_target_z = z_values / sigma_z_values
    coeffs_z, *_ = np.linalg.lstsq(weighted_design_z, weighted_target_z, rcond=None)
    pca_dz, tan_lambda = coeffs_z
    pca_eta = np.arcsinh(tan_lambda)

    return np.array([pca_c_sign * abs_pca_c, pca_eta, pca_phi, pca_dxy, pca_dz], dtype=np.float32)


def compute_fast_fit_targets(x_raw: np.ndarray, y_truth: np.ndarray, hit_groups: list[dict], sentinel_value: float = -999.0):
    fast = np.zeros_like(y_truth, dtype=np.float32)
    failures = 0
    for row_idx in range(x_raw.shape[0]):
        xyz_hits = []
        row = x_raw[row_idx]
        for cols in hit_groups:
            if row[cols["mask"]] <= 0.5:
                continue
            x_value = row[cols["x"]]
            y_value = row[cols["y"]]
            z_value = row[cols["z"]]
            if x_value == sentinel_value or y_value == sentinel_value or z_value == sentinel_value:
                continue
            xyz_hits.append((x_value, y_value, z_value))
        if len(xyz_hits) < 3:
            fast[row_idx] = y_truth[row_idx]
            failures += 1
            continue
        try:
            fast[row_idx] = fit_helix_xyz_pca_linearized(np.asarray(xyz_hits, dtype=float))
        except Exception:
            # Keep the run trainable even if the deterministic fit fails on a rare row.
            fast[row_idx] = y_truth[row_idx]
            failures += 1
    return fast, failures


def build_residual_targets(truth_values: np.ndarray, baseline_values: np.ndarray, phi_index: int | None):
    residual = truth_values - baseline_values
    if phi_index is not None:
        residual[:, phi_index] = wrap_angle_np(residual[:, phi_index])
    return residual


def reconstruct_full_targets(baseline_values: torch.Tensor, residual_values: torch.Tensor, phi_index: int | None):
    out = baseline_values + residual_values
    if phi_index is not None:
        out = out.clone()
        out[:, phi_index] = wrap_angle_torch(out[:, phi_index])
    return out


def maybe_recover_canonical_phi(values: torch.Tensor, rotation_angles: torch.Tensor, phi_index: int | None, training_mode: str):
    if training_mode != "canonical" or phi_index is None:
        return values
    return recover_phi_from_canonical(values, rotation_angles, phi_index)


def load_fastfit_track_data(
    csv_path: Path,
    batch_size: int,
    seed: int,
    device,
    val_fraction: float,
    dataloader_workers: int,
    training_mode: str,
) -> FastFitDataBundle:
    df = pd.read_csv(csv_path)
    target_cols = list(DEFAULT_TARGET_COLS)
    feature_cols = detect_feature_cols(df.columns.tolist())
    if not feature_cols:
        raise ValueError(f"No supported legacy hit columns found in {csv_path}")
    phi_index = target_cols.index("pca_phi")

    x_raw = df[feature_cols].to_numpy(dtype=np.float32)
    y_truth_raw = df[target_cols].to_numpy(dtype=np.float32)
    hit_groups = build_hit_groups(feature_cols)

    fast_fit_raw, n_fastfit_failures = compute_fast_fit_targets(x_raw, y_truth_raw, hit_groups)

    x_proc = x_raw.copy()
    x_proc[x_proc == -999.0] = 0.0

    rotation_angles, has_rotation_anchor = compute_first_valid_hit_angles(x_proc, hit_groups)
    if training_mode == "canonical":
        x_proc = rotate_hit_xy_features(x_proc, hit_groups, rotation_angles)
        y_truth_proc = rotate_phi_column_to_canonical(y_truth_raw, phi_index, rotation_angles)
        fast_fit_proc = rotate_phi_column_to_canonical(fast_fit_raw, phi_index, rotation_angles)
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

    residual_targets = build_residual_targets(y_truth_proc, fast_fit_proc, phi_index)
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

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    rot_train_t = torch.tensor(rot_train, dtype=torch.float32)
    rot_val_t = torch.tensor(rot_val, dtype=torch.float32)
    baseline_train_t = torch.tensor(baseline_train, dtype=torch.float32)
    baseline_val_t = torch.tensor(baseline_val, dtype=torch.float32)

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

    return FastFitDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        x_train=x_train_t,
        x_val=x_val_t,
        y_train=y_train_t,
        y_val=y_val_t,
        rot_train=rot_train_t,
        rot_val=rot_val_t,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        y_mean_t=torch.tensor(y_mean, dtype=torch.float32, device=device),
        y_std_t=torch.tensor(y_std, dtype=torch.float32, device=device),
        feature_cols=feature_cols,
        target_cols=target_cols,
        fast_fit_cols=FAST_FIT_COLS,
        phi_index=phi_index,
        rotation_source=rotation_source,
        training_mode=training_mode,
        n_fastfit_failures=n_fastfit_failures,
    )


def get_device():
    return torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )


def resolve_device(device_override: str = ""):
    if device_override:
        device = torch.device(device_override)
    else:
        device = get_device()
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def parse_device_slots(device_slots: str) -> list[str]:
    if device_slots.strip():
        slots = []
        for part in device_slots.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" in part:
                device_name, count_text = part.split("=", 1)
                count = int(count_text)
            else:
                device_name = part
                count = 1
            slots.extend([device_name.strip()] * max(count, 0))
        return slots
    if torch.cuda.is_available():
        return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
    return ["cpu"]


def parse_modes(modes_arg: str) -> list[str]:
    modes = [mode.strip() for mode in modes_arg.split(",") if mode.strip()]
    invalid = [mode for mode in modes if mode not in TRAINING_MODES]
    if invalid:
        raise ValueError(f"Unsupported modes: {invalid}. Supported: {TRAINING_MODES}")
    if not modes:
        raise ValueError("At least one mode must be requested.")
    return modes


def build_subprocess_command(args, mode: str, device_name: str) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-mode",
        mode,
        "--data-path",
        str(args.data_path),
        "--output-dir",
        args.output_dir,
        "--epochs",
        str(args.epochs),
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
        "--no-show-plots",
        "--no-print-final-samples",
    ]
    if args.show_plots:
        cmd[-2] = "--show-plots"
    if args.print_final_samples:
        cmd[-1] = "--print-final-samples"
    if args.track_overlap_golden:
        cmd.append("--track-overlap-golden")
    return cmd


def run_modes_concurrently(args, modes: list[str], auto_root: Path):
    device_slots = parse_device_slots(args.device_slots)
    max_parallel = max(1, min(args.max_concurrent, len(device_slots)))
    free_slots = device_slots[:max_parallel]
    pending = list(modes)
    running = []

    print(f"Using device slots: {free_slots}")

    while pending or running:
        while pending and free_slots:
            mode = pending.pop(0)
            device_name = free_slots.pop(0)
            run_dir = auto_root / mode
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "train.log"
            status_path = run_dir / "run_status.txt"
            status_path.write_text(f"running on {device_name}\n", encoding="utf-8")
            log_handle = open(log_path, "w", encoding="utf-8")
            cmd = build_subprocess_command(args, mode, device_name)
            print(f"Launching {mode} on {device_name}")
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
                print(f"Finished {item['mode']} on {item['device_name']}")
            else:
                print(f"Run failed for {item['mode']} on {item['device_name']} with rc={retcode}")
        running = still_running


def get_golden_plot_location(base_plot_dir: Path, metric_name: str):
    for metric_prefix in (GOLDEN_SCATTER_PREFIX, GOLDEN_OVERLAP_PREFIX):
        if metric_name.startswith(metric_prefix):
            target_name = metric_name[len(metric_prefix):]
            file_prefix = metric_prefix.rstrip("_")
            return base_plot_dir / target_name, file_prefix
    return base_plot_dir, metric_name


def build_checkpoint_metadata(data: FastFitDataBundle, input_dim: int, csv_path: Path, args, report_text=None):
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
        "hidden_layers": HIDDEN_LAYERS,
        "use_batchnorm": BATCH_NORM,
        "dropout": DROPOUT,
        "activation": "ReLU",
        "batch_size": args.batch_size,
        "dataloader_workers": args.dataloader_workers,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "criterion": CRITERION.__name__,
        "target_weights": TARGET_WEIGHTS.tolist(),
        "mean_weights": MEAN_WEIGHTS.tolist(),
        "overlap_target_index": OVERLAP_TARGET_INDEX,
        "overlap_target_name": data.target_cols[OVERLAP_TARGET_INDEX],
        "training_mode": data.training_mode,
        "canonical_phi": data.training_mode == "canonical",
        "canonical_rotation_source": data.rotation_source,
        "fast_fit_baseline": "peter_linearized_3d_helix_fit",
        "target_definition": "truth_minus_fast_fit_residual",
        "source_csv": str(csv_path),
        "n_fastfit_failures": int(data.n_fastfit_failures),
    }
    if report_text is not None:
        metadata["report_text"] = report_text
    return metadata


def collect_predictions_targets_and_sigma(model, val_loader, device, y_mean_t, y_std_t, phi_index, training_mode):
    model.eval()
    all_pred = []
    all_true = []
    all_fast = []
    all_sigma = []
    saw_logvar = False

    with torch.no_grad():
        for xb, yb, rb, fastb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            rb = rb.to(device)
            fastb = fastb.to(device)

            pred_resid_norm, logvar = model(xb)
            pred_resid_phys = pred_resid_norm * y_std_t + y_mean_t
            true_resid_phys = yb * y_std_t + y_mean_t

            pred_proc = reconstruct_full_targets(fastb, pred_resid_phys, phi_index)
            true_proc = reconstruct_full_targets(fastb, true_resid_phys, phi_index)
            fast_proc = fastb

            pred_phys = maybe_recover_canonical_phi(pred_proc, rb, phi_index, training_mode)
            true_phys = maybe_recover_canonical_phi(true_proc, rb, phi_index, training_mode)
            fast_phys = maybe_recover_canonical_phi(fast_proc, rb, phi_index, training_mode)

            all_pred.append(pred_phys.detach().cpu())
            all_true.append(true_phys.detach().cpu())
            all_fast.append(fast_phys.detach().cpu())

            if logvar is not None:
                saw_logvar = True
                sigma_phys = torch.exp(0.5 * logvar) * y_std_t
                all_sigma.append(sigma_phys.detach().cpu())

    y_pred = torch.cat(all_pred, dim=0).numpy()
    y_true = torch.cat(all_true, dim=0).numpy()
    y_fast = torch.cat(all_fast, dim=0).numpy()
    y_sigma = torch.cat(all_sigma, dim=0).numpy() if saw_logvar else None
    return y_pred, y_true, y_fast, y_sigma


def make_mode_val_diagnostic_plots(
    model,
    val_loader,
    device,
    y_mean_t,
    y_std_t,
    target_cols,
    phi_index,
    training_mode,
    output_dir="plots",
    prefix="val",
    bins=100,
    density=True,
    show=False,
    scatter_max_points=5000,
    central_fraction=0.99,
    figure_label=None,
):
    y_pred, y_true, _, y_sigma = collect_predictions_targets_and_sigma(
        model=model,
        val_loader=val_loader,
        device=device,
        y_mean_t=y_mean_t,
        y_std_t=y_std_t,
        phi_index=phi_index,
        training_mode=training_mode,
    )

    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "overlap": os.path.join(output_dir, f"{prefix}_overlap.png"),
        "scatter": os.path.join(output_dir, f"{prefix}_scatter_pred_vs_actual.png"),
        "distance": os.path.join(output_dir, f"{prefix}_distance.png"),
    }
    if y_sigma is not None:
        paths["pull"] = os.path.join(output_dir, f"{prefix}_pull.png")

    plot_overlap_distributions(
        y_true=y_true,
        y_pred=y_pred,
        target_cols=target_cols,
        bins=bins,
        density=density,
        save_path=paths["overlap"],
        show=show,
        central_fraction=central_fraction,
        figure_label=figure_label,
    )
    plot_pred_vs_true_scatter(
        y_true=y_true,
        y_pred=y_pred,
        target_cols=target_cols,
        save_path=paths["scatter"],
        show=show,
        max_points=scatter_max_points,
        central_fraction=central_fraction,
        figure_label=figure_label,
    )
    plot_distance_distribution(
        y_true=y_true,
        y_pred=y_pred,
        y_sigma=y_sigma,
        target_cols=target_cols,
        phi_index=phi_index,
        bins=bins,
        density=density,
        save_path=paths["distance"],
        show=show,
        figure_label=figure_label,
    )
    if y_sigma is not None:
        plot_pull_distributions(
            y_true=y_true,
            y_pred=y_pred,
            y_sigma=y_sigma,
            target_cols=target_cols,
            phi_index=phi_index,
            bins=bins,
            density=density,
            save_path=paths["pull"],
            show=show,
            figure_label=figure_label,
        )
    return paths


def make_fastfit_baseline_comparison_plots(
    y_true,
    y_pred,
    y_fast,
    target_cols,
    output_dir,
    prefix,
    show=False,
    max_points=5000,
    seed=42,
    figure_label=None,
):
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    scatter_path = os.path.join(output_dir, f"{prefix}_fastfit_vs_model_scatter.png")
    overlap_path = os.path.join(output_dir, f"{prefix}_fastfit_vs_model_overlap.png")

    n_targets = len(target_cols)
    n_rows = len(y_true)
    if max_points is not None and n_rows > max_points:
        rng = np.random.default_rng(seed=seed)
        plot_idx = rng.choice(n_rows, size=max_points, replace=False)
    else:
        plot_idx = np.arange(n_rows)

    fig, axes = plt.subplots(2, n_targets, figsize=(5 * n_targets, 8))
    axes = np.asarray(axes).reshape(2, n_targets)
    for row_idx, (row_name, row_pred) in enumerate((("Fast fit", y_fast), ("Model", y_pred))):
        for col_idx, name in enumerate(target_cols):
            ax = axes[row_idx, col_idx]
            true_vals = y_true[plot_idx, col_idx]
            pred_vals = row_pred[plot_idx, col_idx]
            mask = np.isfinite(true_vals) & np.isfinite(pred_vals)
            true_vals = true_vals[mask]
            pred_vals = pred_vals[mask]
            if len(true_vals) == 0:
                continue
            lo = min(true_vals.min(), pred_vals.min())
            hi = max(true_vals.max(), pred_vals.max())
            if np.isclose(lo, hi):
                lo -= 0.5
                hi += 0.5
            ax.scatter(true_vals, pred_vals, s=8, alpha=0.35)
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="red", linewidth=1)
            ax.set_title(f"{name} | {row_name}")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
    if figure_label:
        fig.text(0.995, 0.005, str(figure_label), ha="right", va="bottom", fontsize=6, alpha=0.65)
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 4))
    axes = np.atleast_1d(axes)
    for col_idx, name in enumerate(target_cols):
        ax = axes[col_idx]
        true_vals = y_true[:, col_idx]
        fast_vals = y_fast[:, col_idx]
        pred_vals = y_pred[:, col_idx]
        mask_true = np.isfinite(true_vals)
        mask_fast = np.isfinite(fast_vals)
        mask_pred = np.isfinite(pred_vals)
        merged = np.concatenate([true_vals[mask_true], fast_vals[mask_fast], pred_vals[mask_pred]])
        if len(merged) == 0:
            continue
        lo, hi = np.quantile(merged, [0.005, 0.995])
        if np.isclose(lo, hi):
            lo -= 0.5
            hi += 0.5
        bins = np.linspace(lo, hi, 100)
        ax.hist(true_vals[mask_true], bins=bins, alpha=0.4, density=True, label="Actual")
        ax.hist(fast_vals[mask_fast], bins=bins, alpha=0.4, density=True, label="Fast fit")
        ax.hist(pred_vals[mask_pred], bins=bins, alpha=0.4, density=True, label="Model")
        ax.set_title(name)
        ax.legend()
    if figure_label:
        fig.text(0.995, 0.005, str(figure_label), ha="right", va="bottom", fontsize=6, alpha=0.65)
    plt.tight_layout()
    plt.savefig(overlap_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"scatter_compare": scatter_path, "overlap_compare": overlap_path}


def print_fastfit_final_validation_samples(
    model,
    val_loader,
    device,
    y_mean_t,
    y_std_t,
    target_cols,
    phi_index,
    training_mode,
    num_examples=10,
):
    print("\n" + "=" * 80)
    print("FINAL VALIDATION SAMPLES")
    print("=" * 80)

    model.eval()
    shown = 0
    with torch.no_grad():
        for xb, yb, rb, fastb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            rb = rb.to(device)
            fastb = fastb.to(device)

            pred_resid_norm, logvar = model(xb)
            pred_resid_phys = pred_resid_norm * y_std_t + y_mean_t
            true_resid_phys = yb * y_std_t + y_mean_t

            pred_proc = reconstruct_full_targets(fastb, pred_resid_phys, phi_index)
            true_proc = reconstruct_full_targets(fastb, true_resid_phys, phi_index)
            fast_proc = fastb

            pred_phys = maybe_recover_canonical_phi(pred_proc, rb, phi_index, training_mode)
            true_phys = maybe_recover_canonical_phi(true_proc, rb, phi_index, training_mode)
            fast_phys = maybe_recover_canonical_phi(fast_proc, rb, phi_index, training_mode)

            err = pred_phys - true_phys
            if phi_index is not None:
                err[:, phi_index] = wrap_angle_torch(err[:, phi_index])

            std_phys = None
            if logvar is not None:
                std_phys = torch.exp(0.5 * logvar) * y_std_t

            for i in range(xb.size(0)):
                print(f"\nValidation example {shown + 1}")
                print(f"  training_mode = {training_mode}")
                print(f"  canonical_rotation = {rb[i].item(): .6f}")
                for j, name in enumerate(target_cols):
                    line = (
                        f"  {name:8s} | "
                        f"true = {true_phys[i, j].item(): .6f} | "
                        f"fast = {fast_phys[i, j].item(): .6f} | "
                        f"pred = {pred_phys[i, j].item(): .6f} | "
                        f"error = {err[i, j].item(): .6f}"
                    )
                    if std_phys is not None:
                        line += f" | uncertainty_std = {std_phys[i, j].item(): .6f}"
                    print(line)

                shown += 1
                if shown >= num_examples:
                    return


def build_baseline_comparison_report(y_true, y_pred, y_fast, target_cols):
    model_scores = compute_plot_quality_scores(y_true=y_true, y_pred=y_pred, target_cols=target_cols, bins=100)
    fast_scores = compute_plot_quality_scores(y_true=y_true, y_pred=y_fast, target_cols=target_cols, bins=100)
    lines = ["   Baseline vs model comparison:"]
    lines.append("      Scatter score deltas (model - fast fit):")
    for name in target_cols:
        delta = model_scores["scatter"][name]["score"] - fast_scores["scatter"][name]["score"]
        lines.append(
            f"         {name}: model={model_scores['scatter'][name]['score']:.6f} | "
            f"fast={fast_scores['scatter'][name]['score']:.6f} | delta={delta:.6f}"
        )
    lines.append("      Overlap score deltas (model - fast fit):")
    for name in target_cols:
        delta = model_scores["overlap"][name]["score"] - fast_scores["overlap"][name]["score"]
        lines.append(
            f"         {name}: model={model_scores['overlap'][name]['score']:.6f} | "
            f"fast={fast_scores['overlap'][name]['score']:.6f} | delta={delta:.6f}"
        )
    return "\n".join(lines), model_scores, fast_scores


def summarize_mode_outputs(auto_root: Path, modes: list[str]):
    rows = []
    metric_rows = []
    for mode in modes:
        run_dir = auto_root / mode
        run_summary_path = run_dir / "run_summary.csv"
        metric_summary_path = run_dir / "metric_summary.csv"
        if run_summary_path.exists():
            rows.extend(pd.read_csv(run_summary_path).to_dict(orient="records"))
        if metric_summary_path.exists():
            metric_rows.extend(pd.read_csv(metric_summary_path).to_dict(orient="records"))
    if rows:
        pd.DataFrame(rows).to_csv(auto_root / "mode_run_summary.csv", index=False)
    if metric_rows:
        pd.DataFrame(metric_rows).to_csv(auto_root / "mode_metric_summary.csv", index=False)


def train_one_mode(csv_path: Path, auto_root: Path, args, device, training_mode: str):
    run_dir = auto_root / training_mode
    plot_dir = run_dir / "plots"
    final_plot_dir = plot_dir / "final"
    best_dxy_plot_dir = plot_dir / "best_dxy_corr"
    golden_model_dir = run_dir / "goldenmodels"
    save_dir = run_dir / "saves"
    golden_summary_file = run_dir / "golden_summary.txt"
    os.makedirs(final_plot_dir, exist_ok=True)
    os.makedirs(best_dxy_plot_dir, exist_ok=True)
    os.makedirs(golden_model_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    data = load_fastfit_track_data(
        csv_path=csv_path,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        val_fraction=args.val_fraction,
        dataloader_workers=args.dataloader_workers,
        training_mode=training_mode,
    )
    input_dim = data.x_train.shape[1]
    model = HeteroTrackNet(
        input_dim=input_dim,
        hidden_layers=HIDDEN_LAYERS,
        output_dim=len(data.target_cols),
        use_batchnorm=BATCH_NORM,
        dropout=DROPOUT,
        activation=nn.ReLU,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    training_history = {"epoch": [], "train_loss": [], "val_loss": [], "val_mean_mae": [], "val_mean_rmse": [], "learning_rate": []}
    best_vals = {}
    best_reports = {}
    best_epochs = {}
    best_vals[BEST_OVERALL_SCATTER_TAG] = -float("inf")
    if args.track_overlap_golden:
        best_vals[BEST_OVERALL_OVERLAP_TAG] = -float("inf")
    for name in data.target_cols:
        best_vals[f"{GOLDEN_SCATTER_PREFIX}{name}"] = -float("inf")
        if args.track_overlap_golden:
            best_vals[f"{GOLDEN_OVERLAP_PREFIX}{name}"] = -float("inf")

    best_val_loss = float("inf")
    best_val_epoch = 0
    best_model_paths = {}
    best_plot_report_paths = {}
    best_overall_scatter_score = -float("inf")
    best_overall_scatter_epoch = 0
    best_overall_plot_paths = {}
    best_overall_overlap_score = -float("inf")
    best_overall_overlap_epoch = 0

    def save_golden(metric_tag, metric_value, metric_details, epoch, report, overlap_report, plot_quality_report, baseline_report):
        full_report = f"{report}\n{overlap_report}\n{plot_quality_report}\n{baseline_report}"
        metadata = build_checkpoint_metadata(data, input_dim, csv_path, args, report_text=full_report)
        metadata.update(
            {
                "metric_tag": metric_tag,
                "metric_value": float(metric_value),
                "plot_quality_metric": metric_details,
                "mode": training_mode,
            }
        )
        save_golden_model(model, optimizer, scheduler, metric_tag, metric_value, epoch, full_report, str(golden_model_dir), metadata)
        best_reports[metric_tag] = full_report
        best_epochs[metric_tag] = epoch + 1
        best_model_paths[metric_tag] = str(golden_model_dir / f"{metric_tag}.pt")
        golden_output_dir, golden_file_prefix = get_golden_plot_location(plot_dir, metric_tag)
        os.makedirs(golden_output_dir, exist_ok=True)
        report_path = golden_output_dir / f"{golden_file_prefix}_training_report.txt"
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write("GOLDEN TRAINING REPORT\n")
            handle.write("=" * 80 + "\n")
            handle.write(f"metric_tag: {metric_tag}\n")
            handle.write(f"epoch: {epoch + 1}\n")
            handle.write(f"metric_value: {float(metric_value):.6f}\n\n")
            handle.write(full_report)
            handle.write("\n")
        best_plot_report_paths[metric_tag] = str(report_path)

    def save_best_overall_plots(metric_tag, epoch, metric_value, report, overlap_report, plot_quality_report, baseline_report):
        nonlocal best_overall_plot_paths
        best_overall_plot_paths = make_mode_val_diagnostic_plots(
            model=model,
            val_loader=data.val_loader,
            device=device,
            y_mean_t=data.y_mean_t,
            y_std_t=data.y_std_t,
            target_cols=data.target_cols,
            phi_index=data.phi_index,
            training_mode=training_mode,
            output_dir=str(best_dxy_plot_dir),
            prefix=training_mode,
            bins=100,
            density=True,
            show=False,
            scatter_max_points=DIAGNOSTIC_SCATTER_MAX_POINTS,
            central_fraction=DIAGNOSTIC_CENTRAL_FRACTION,
            figure_label=f"{training_mode} | {metric_tag}",
        )
        y_pred, y_true, y_fast, _ = collect_predictions_targets_and_sigma(
            model=model,
            val_loader=data.val_loader,
            device=device,
            y_mean_t=data.y_mean_t,
            y_std_t=data.y_std_t,
            phi_index=data.phi_index,
            training_mode=training_mode,
        )
        best_overall_plot_paths.update(
            make_fastfit_baseline_comparison_plots(
                y_true=y_true,
                y_pred=y_pred,
                y_fast=y_fast,
                target_cols=data.target_cols,
                output_dir=str(best_dxy_plot_dir),
                prefix=training_mode,
                show=False,
                max_points=DIAGNOSTIC_SCATTER_MAX_POINTS,
                figure_label=f"{training_mode} | {metric_tag}",
            )
        )
        report_path = best_dxy_plot_dir / f"{metric_tag}_training_report.txt"
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write("BEST OVERALL MODEL SNAPSHOT\n")
            handle.write("=" * 80 + "\n")
            handle.write(f"mode: {training_mode}\n")
            handle.write(f"metric_tag: {metric_tag}\n")
            handle.write(f"epoch: {epoch + 1}\n")
            handle.write(f"metric_value: {float(metric_value):.6f}\n\n")
            handle.write(report)
            handle.write("\n")
            handle.write(overlap_report)
            handle.write("\n")
            handle.write(plot_quality_report)
            handle.write("\n")
            handle.write(baseline_report)
            handle.write("\n")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for xb, yb, _, _ in data.train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            mu, logvar = model(xb)
            loss = CRITERION(
                yb,
                mu,
                logvar,
                phi_index=data.phi_index,
                target_weights=TARGET_WEIGHTS,
                mean_weights=MEAN_WEIGHTS,
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

        overlap_pred_parts = []
        overlap_true_parts = []
        overlap_fast_parts = []
        with torch.no_grad():
            for xb, yb, rb, fastb in data.val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                rb = rb.to(device)
                fastb = fastb.to(device)

                mu, logvar = model(xb)
                loss = CRITERION(
                    yb,
                    mu,
                    logvar,
                    phi_index=data.phi_index,
                    target_weights=TARGET_WEIGHTS,
                    mean_weights=MEAN_WEIGHTS,
                )
                val_loss += loss.item() * xb.size(0)

                pred_resid_phys = mu * data.y_std_t + data.y_mean_t
                true_resid_phys = yb * data.y_std_t + data.y_mean_t

                pred_proc = reconstruct_full_targets(fastb, pred_resid_phys, data.phi_index)
                true_proc = reconstruct_full_targets(fastb, true_resid_phys, data.phi_index)
                fast_proc = fastb

                pred_phys = maybe_recover_canonical_phi(pred_proc, rb, data.phi_index, training_mode)
                true_phys = maybe_recover_canonical_phi(true_proc, rb, data.phi_index, training_mode)
                fast_phys = maybe_recover_canonical_phi(fast_proc, rb, data.phi_index, training_mode)

                diff = pred_phys - true_phys
                diff[:, data.phi_index] = wrap_angle_torch(diff[:, data.phi_index])
                total_val_mae += diff.abs().sum(dim=0)
                total_val_sq += (diff ** 2).sum(dim=0)
                total_count += xb.size(0)

                overlap_pred_parts.append(pred_phys.detach().cpu())
                overlap_true_parts.append(true_phys.detach().cpu())
                overlap_fast_parts.append(fast_phys.detach().cpu())

        val_loss /= len(data.val_loader.dataset)
        per_target_mae = (total_val_mae / total_count).detach().cpu().numpy()
        per_target_rmse = np.sqrt((total_val_sq / total_count).detach().cpu().numpy())
        overall_val_mae = float(per_target_mae.mean())
        overall_val_rmse = float(per_target_rmse.mean())
        current_lr = optimizer.param_groups[0]["lr"]

        overlap_pred = torch.cat(overlap_pred_parts, dim=0).numpy()
        overlap_true = torch.cat(overlap_true_parts, dim=0).numpy()
        overlap_fast = torch.cat(overlap_fast_parts, dim=0).numpy()

        target_overlap = compute_target_histogram_overlap(overlap_true, overlap_pred, OVERLAP_TARGET_INDEX, data.target_cols, bins=100)
        plot_quality_scores = compute_plot_quality_scores(y_true=overlap_true, y_pred=overlap_pred, target_cols=data.target_cols, bins=100)
        plot_quality_report = format_plot_quality_report(plot_quality_scores)
        baseline_report, _, _ = build_baseline_comparison_report(overlap_true, overlap_pred, overlap_fast, data.target_cols)
        overall_scatter_score = float(np.mean([plot_quality_scores["scatter"][name]["score"] for name in data.target_cols]))
        overall_overlap_score = float(np.mean([plot_quality_scores["overlap"][name]["score"] for name in data.target_cols]))
        report = format_epoch_report(epoch, args.epochs, train_loss, val_loss, overall_val_mae, overall_val_rmse, per_target_mae, per_target_rmse, data.target_cols)
        overlap_report = f"   Overlap {data.target_cols[OVERLAP_TARGET_INDEX]}: {target_overlap:.6f} | MAE: {float(per_target_mae[OVERLAP_TARGET_INDEX]):.6f}"
        overall_report = (
            f"   Overall model-selection scores:\n"
            f"      mean_scatter_score: {overall_scatter_score:.6f}\n"
            f"      mean_overlap_score: {overall_overlap_score:.6f}"
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
            metadata = build_checkpoint_metadata(data, input_dim, csv_path, args, report_text=f"{report}\n{overlap_report}\n{plot_quality_report}\n{baseline_report}")
            metadata.update({"checkpoint_type": "best_val_loss", "val_loss": float(val_loss), "mode": training_mode})
            save_model_checkpoint(str(save_dir / "best_val_loss.pt"), model, optimizer, scheduler, epoch + 1, metadata)

        if overall_scatter_score > best_overall_scatter_score:
            best_overall_scatter_score = overall_scatter_score
            best_overall_scatter_epoch = epoch + 1
            save_golden(
                BEST_OVERALL_SCATTER_TAG,
                overall_scatter_score,
                {"score": overall_scatter_score},
                epoch,
                report,
                f"{overlap_report}\n{overall_report}",
                plot_quality_report,
                baseline_report,
            )
            save_best_overall_plots(
                BEST_OVERALL_SCATTER_TAG,
                epoch,
                overall_scatter_score,
                report,
                f"{overlap_report}\n{overall_report}",
                plot_quality_report,
                baseline_report,
            )

        if args.track_overlap_golden and overall_overlap_score > best_overall_overlap_score:
            best_overall_overlap_score = overall_overlap_score
            best_overall_overlap_epoch = epoch + 1
            save_golden(
                BEST_OVERALL_OVERLAP_TAG,
                overall_overlap_score,
                {"score": overall_overlap_score},
                epoch,
                report,
                f"{overlap_report}\n{overall_report}",
                plot_quality_report,
                baseline_report,
            )

        for target_name in data.target_cols:
            scatter_tag = f"{GOLDEN_SCATTER_PREFIX}{target_name}"
            scatter_metric = plot_quality_scores["scatter"][target_name]
            scatter_score = scatter_metric["score"]
            if scatter_score > best_vals[scatter_tag]:
                best_vals[scatter_tag] = scatter_score
                save_golden(scatter_tag, scatter_score, scatter_metric, epoch, report, f"{overlap_report}\n{overall_report}", plot_quality_report, baseline_report)

            if args.track_overlap_golden:
                overlap_tag = f"{GOLDEN_OVERLAP_PREFIX}{target_name}"
                overlap_metric = plot_quality_scores["overlap"][target_name]
                overlap_score = overlap_metric["score"]
                if overlap_score > best_vals[overlap_tag]:
                    best_vals[overlap_tag] = overlap_score
                    save_golden(overlap_tag, overlap_score, overlap_metric, epoch, report, f"{overlap_report}\n{overall_report}", plot_quality_report, baseline_report)

        print(
            f"[{training_mode}] Epoch {epoch + 1}/{args.epochs} | train {train_loss:.6f} | "
            f"val {val_loss:.6f} | mean MAE {overall_val_mae:.6f} | mean RMSE {overall_val_rmse:.6f} | "
            f"{data.target_cols[OVERLAP_TARGET_INDEX]} overlap {target_overlap:.6f} | "
            f"overall scatter {overall_scatter_score:.6f}"
        )
        scheduler.step()

    final_metadata = build_checkpoint_metadata(data, input_dim, csv_path, args, report_text="final_model")
    final_metadata.update({"checkpoint_type": "final_model", "best_val_loss": float(best_val_loss), "best_val_epoch": best_val_epoch, "mode": training_mode})
    save_model_checkpoint(str(save_dir / "final_model.pt"), model, optimizer, scheduler, args.epochs, final_metadata)

    history_plot_paths = make_training_history_plots(
        training_history,
        output_dir=str(final_plot_dir),
        prefix=training_mode,
        show=False,
        figure_label=f"{training_mode} | final",
    )
    val_plot_paths = make_mode_val_diagnostic_plots(
        model=model,
        val_loader=data.val_loader,
        device=device,
        y_mean_t=data.y_mean_t,
        y_std_t=data.y_std_t,
        target_cols=data.target_cols,
        phi_index=data.phi_index,
        training_mode=training_mode,
        output_dir=str(final_plot_dir),
        prefix=training_mode,
        bins=100,
        density=True,
        show=args.show_plots,
        scatter_max_points=DIAGNOSTIC_SCATTER_MAX_POINTS,
        central_fraction=DIAGNOSTIC_CENTRAL_FRACTION,
        figure_label=f"{training_mode} | final",
    )

    y_pred, y_true, y_fast, _ = collect_predictions_targets_and_sigma(
        model=model,
        val_loader=data.val_loader,
        device=device,
        y_mean_t=data.y_mean_t,
        y_std_t=data.y_std_t,
        phi_index=data.phi_index,
        training_mode=training_mode,
    )
    compare_plot_paths = make_fastfit_baseline_comparison_plots(
        y_true=y_true,
        y_pred=y_pred,
        y_fast=y_fast,
        target_cols=data.target_cols,
        output_dir=str(final_plot_dir),
        prefix=training_mode,
        show=args.show_plots,
        max_points=DIAGNOSTIC_SCATTER_MAX_POINTS,
        figure_label=f"{training_mode} | final",
    )
    baseline_report, model_scores, fast_scores = build_baseline_comparison_report(y_true, y_pred, y_fast, data.target_cols)
    with open(final_plot_dir / "fastfit_vs_model_report.txt", "w", encoding="utf-8") as handle:
        handle.write("FAST FIT VS MODEL REPORT\n")
        handle.write("=" * 80 + "\n")
        handle.write(baseline_report)
        handle.write("\n\nModel plot quality scores\n")
        handle.write(format_plot_quality_report(model_scores))
        handle.write("\n\nFast-fit plot quality scores\n")
        handle.write(format_plot_quality_report(fast_scores))
        handle.write("\n")

    if args.print_final_samples:
        print_fastfit_final_validation_samples(
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

    write_final_golden_summary(str(golden_summary_file), best_reports, best_vals)

    metric_rows = []
    for metric_tag, value in best_vals.items():
        family = "scatter" if metric_tag.startswith(GOLDEN_SCATTER_PREFIX) else "overlap"
        target = metric_tag.split("_")[-1]
        metric_rows.append(
            {
                "mode": training_mode,
                "csv_path": str(csv_path),
                "metric_tag": metric_tag,
                "metric_family": family,
                "target": target,
                "best_score": float(value),
                "best_epoch": int(best_epochs.get(metric_tag, 0)),
                "checkpoint_path": best_model_paths.get(metric_tag, ""),
                "report_path": best_plot_report_paths.get(metric_tag, ""),
            }
        )

    run_summary = {
        "mode": training_mode,
        "csv_path": str(csv_path),
        "rows": int(len(data.x_train) + len(data.x_val)),
        "train_rows": int(len(data.x_train)),
        "val_rows": int(len(data.x_val)),
        "input_dim": input_dim,
        "best_val_loss": float(best_val_loss),
        "best_val_epoch": int(best_val_epoch),
        "final_val_loss": float(training_history["val_loss"][-1]),
        "final_val_mean_mae": float(training_history["val_mean_mae"][-1]),
        "final_val_mean_rmse": float(training_history["val_mean_rmse"][-1]),
        "run_dir": str(run_dir),
        "final_plot_dir": str(final_plot_dir),
        "best_dxy_corr_plot_dir": str(best_dxy_plot_dir),
        "best_overall_scatter_epoch": int(best_overall_scatter_epoch),
        "best_overall_scatter_score": float(best_overall_scatter_score),
        "best_overall_overlap_epoch": int(best_overall_overlap_epoch),
        "best_overall_overlap_score": float(best_overall_overlap_score),
        "best_val_checkpoint_path": str(save_dir / "best_val_loss.pt"),
        "final_checkpoint_path": str(save_dir / "final_model.pt"),
        "history_plot_count": len(history_plot_paths),
        "val_plot_count": len(val_plot_paths),
        "compare_plot_count": len(compare_plot_paths),
        "best_overall_plot_count": len(best_overall_plot_paths),
        "n_fastfit_failures": int(data.n_fastfit_failures),
    }
    pd.DataFrame(metric_rows).to_csv(run_dir / "metric_summary.csv", index=False)
    pd.DataFrame([run_summary]).to_csv(run_dir / "run_summary.csv", index=False)
    return run_summary, metric_rows


def main():
    args = parse_args()
    auto_root = Path(args.output_dir)
    auto_root.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.data_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV does not exist: {csv_path}")

    if args.single_mode:
        device = resolve_device(args.device)
        set_seed(args.seed)
        print(f"Device set to {device}")
        print(f"Training mode: {args.single_mode}")
        train_one_mode(csv_path, auto_root, args, device, args.single_mode)
        return

    modes = parse_modes(args.modes)
    if args.max_concurrent <= 1 or len(modes) == 1:
        device = resolve_device(args.device)
        set_seed(args.seed)
        print(f"Device set to {device}")
        for mode in modes:
            print(f"Training mode: {mode}")
            train_one_mode(csv_path, auto_root, args, device, mode)
    else:
        run_modes_concurrently(args, modes, auto_root)

    summarize_mode_outputs(auto_root, modes)


if __name__ == "__main__":
    main()
