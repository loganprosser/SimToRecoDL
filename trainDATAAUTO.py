import argparse
import os
import re
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
    wrapped_angle_diff,
    write_final_golden_summary,
)
from helpers_canonical_phi import (
    denormalize_and_recover_phi,
    make_canonical_val_diagnostic_plots,
    print_canonical_final_validation_samples,
)
from helpers_data import DEFAULT_TARGET_COLS, build_hit_feature_cols, set_seed
from helpers_vis import (
    compute_plot_quality_scores,
    compute_target_histogram_overlap,
    format_plot_quality_report,
    make_training_history_plots,
)
from loss import hetero_gaussian_nll_with_phi
from model import HeteroTrackNet


DEFAULT_DATA_DIR = "/nfs/cms/tracktrigger/logan/root/simvrico/SimToRecoDL/outputCSVs/outputCSVs_hittype_compare"
DEFAULT_OUTPUT_DIR = "auto"
DEFAULT_MAX_CONCURRENT = 2
DEFAULT_DEVICE_SLOTS = ""
DEFAULT_TRACK_OVERLAP_GOLDEN = False
FIELD_ORDER = {"x": 0, "y": 1, "z": 2, "r": 3, "mask": 4}
BUCKET_RE = re.compile(r"^ht(?P<hit_type>-?\d+)_bucket_(?P<bucket>\d+)_(?P<field>x|y|z|r|mask)$")
LEGACY_RE = re.compile(r"^hit_(?P<bucket>\d+)_(?P<field>x|y|z|r|mask)$")

EPOCHS = 750
BATCH_SIZE = 256
HIDDEN_LAYERS = [512, 512, 256]
CRITERION = hetero_gaussian_nll_with_phi
TARGET_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
MEAN_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
BATCH_NORM = False
DROPOUT = 0.0
SEED = 42
VAL_FRACTION = 0.2
OVERLAP_TARGET_INDEX = 3
DIAGNOSTIC_CENTRAL_FRACTION = 0.99      #second zoom in plot how much to show
DIAGNOSTIC_SCATTER_MAX_POINTS = None
GOLDEN_SCATTER_PREFIX = "best_scatter_linear_"
GOLDEN_OVERLAP_PREFIX = "best_overlap_cover_"


@dataclass
class AutoDataBundle:
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
    phi_index: int | None
    rotation_source: str
    feature_schema: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train canonical-phi style models over every hit-type comparison CSV and compare results."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory containing comparison CSVs.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for auto runs.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--show-plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print-final-samples", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--track-overlap-golden",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TRACK_OVERLAP_GOLDEN,
        help="Track and save best overlap-based golden checkpoints.",
    )
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT, help="Maximum number of datasets to train at once.")
    parser.add_argument(
        "--device-slots",
        default=DEFAULT_DEVICE_SLOTS,
        help="Comma-separated device slot counts like cuda:0=2,cuda:1=1. Default is one slot per visible CUDA device.",
    )
    parser.add_argument("--device", default="", help="Device override for a single training run, e.g. cuda:0 or cpu.")
    parser.add_argument("--single-dataset", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def discover_dataset_paths(data_dir: str) -> list[Path]:
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"Data directory does not exist: {base}")
    paths = []
    skipped = []
    for path in sorted(base.glob("*.csv")):
        stem_lower = path.stem.lower()
        if "summary" in stem_lower or stem_lower.startswith("auto_"):
            continue
        try:
            columns = pd.read_csv(path, nrows=0).columns.tolist()
            feature_cols = detect_feature_cols(columns)
            if feature_cols:
                paths.append(path)
            else:
                skipped.append(path.name)
        except Exception as exc:
            skipped.append(f"{path.name} ({exc})")
    if not paths:
        raise FileNotFoundError(f"No dataset CSVs found in {base}")
    if skipped:
        print("Skipping unsupported CSVs:")
        for name in skipped:
            print(f"  - {name}")
    return paths


def detect_feature_cols(columns: list[str]) -> list[str]:
    bucket_matches = []
    for col in columns:
        bucket_match = BUCKET_RE.match(col)
        if bucket_match:
            hit_type = int(bucket_match.group("hit_type"))
            bucket = int(bucket_match.group("bucket"))
            field = bucket_match.group("field")
            bucket_matches.append((0, hit_type, bucket, FIELD_ORDER[field], col))

    if bucket_matches:
        bucket_matches.sort()
        return [col for _, _, _, _, col in bucket_matches]

    legacy_cols = []
    legacy_col_set = set(columns)
    candidate_cols = build_hit_feature_cols(n_layers=6)
    if all(col in legacy_col_set for col in candidate_cols):
        for col in candidate_cols:
            match = LEGACY_RE.match(col)
            bucket = int(match.group("bucket"))
            field = match.group("field")
            legacy_cols.append((bucket, FIELD_ORDER[field], col))
        legacy_cols.sort()
        return [col for _, _, col in legacy_cols]

    return []


def detect_feature_schema(feature_cols: list[str]) -> str:
    if feature_cols and all(BUCKET_RE.match(col) for col in feature_cols):
        return "bucketed"
    if feature_cols and all(LEGACY_RE.match(col) for col in feature_cols):
        return "legacy"
    raise ValueError("Unable to determine feature schema from feature columns.")


def build_bucket_groups(feature_cols: list[str]) -> list[dict]:
    groups = {}
    for idx, col in enumerate(feature_cols):
        bucket_match = BUCKET_RE.match(col)
        if bucket_match:
            key = (int(bucket_match.group("hit_type")), int(bucket_match.group("bucket")))
            groups.setdefault(key, {})[bucket_match.group("field")] = idx
            continue

        legacy_match = LEGACY_RE.match(col)
        if legacy_match:
            key = (999, int(legacy_match.group("bucket")))
            groups.setdefault(key, {})[legacy_match.group("field")] = idx
    return [
        {"x": fields["x"], "y": fields["y"], "mask": fields.get("mask")}
        for _, fields in sorted(groups.items())
        if "x" in fields and "y" in fields
    ]


def wrap_angle_np(values):
    return np.arctan2(np.sin(values), np.cos(values))


def compute_first_valid_hit_angles(x: np.ndarray, bucket_groups: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    angles = np.zeros(x.shape[0], dtype=np.float32)
    assigned = np.zeros(x.shape[0], dtype=bool)
    for cols in bucket_groups:
        x_vals = x[:, cols["x"]]
        y_vals = x[:, cols["y"]]
        if cols["mask"] is None:
            valid = np.isfinite(x_vals) & np.isfinite(y_vals) & ((x_vals != 0.0) | (y_vals != 0.0))
        else:
            valid = x[:, cols["mask"]] > 0.5
        use = valid & ~assigned
        angles[use] = np.arctan2(y_vals[use], x_vals[use])
        assigned[use] = True
    return angles, assigned


def rotate_hit_xy_features(x: np.ndarray, bucket_groups: list[dict], rotation_angles: np.ndarray) -> np.ndarray:
    x_rot = x.copy()
    cos_a = np.cos(rotation_angles)
    sin_a = np.sin(rotation_angles)
    for cols in bucket_groups:
        x_old = x[:, cols["x"]].copy()
        y_old = x[:, cols["y"]].copy()
        x_rot[:, cols["x"]] = cos_a * x_old + sin_a * y_old
        x_rot[:, cols["y"]] = -sin_a * x_old + cos_a * y_old
    return x_rot


def rotate_phi_targets_to_canonical(y: np.ndarray, target_cols: list[str], rotation_angles: np.ndarray) -> np.ndarray:
    y_rot = y.copy()
    if "pca_phi" in target_cols:
        phi_index = target_cols.index("pca_phi")
        y_rot[:, phi_index] = wrap_angle_np(y[:, phi_index] - rotation_angles)
    return y_rot


def load_auto_track_data(csv_path: Path, batch_size: int, seed: int, device, val_fraction: float) -> AutoDataBundle:
    df = pd.read_csv(csv_path)
    target_cols = list(DEFAULT_TARGET_COLS)
    feature_cols = detect_feature_cols(df.columns.tolist())
    if not feature_cols:
        raise ValueError(f"No supported feature columns found in {csv_path}")
    feature_schema = detect_feature_schema(feature_cols)
    phi_index = target_cols.index("pca_phi")

    x = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_cols].to_numpy(dtype=np.float32)

    x[x == -999.0] = 0.0
    bucket_groups = build_bucket_groups(feature_cols)
    rotation_angles, _ = compute_first_valid_hit_angles(x, bucket_groups)
    x = rotate_hit_xy_features(x, bucket_groups, rotation_angles)
    y = rotate_phi_targets_to_canonical(y, target_cols, rotation_angles)

    n_rows = len(x)
    n_val = int(n_rows * val_fraction)
    rng = np.random.default_rng(seed=seed)
    indices = rng.permutation(n_rows)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    x_train = x[train_idx]
    x_val = x[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    rot_train = rotation_angles[train_idx]
    rot_val = rotation_angles[val_idx]

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

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(x_train_t, y_train_t, rot_train_t),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        TensorDataset(x_val_t, y_val_t, rot_val_t),
        batch_size=batch_size,
        shuffle=False,
        generator=generator,
    )

    return AutoDataBundle(
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
        phi_index=phi_index,
        rotation_source="first_valid_bucket_xy_angle",
        feature_schema=feature_schema,
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


def build_subprocess_command(args, csv_path: Path, device_name: str) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-dataset",
        str(csv_path),
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
        "--no-show-plots",
        "--no-print-final-samples",
    ]
    if args.show_plots:
        cmd[-2] = "--show-plots"
    if args.print_final_samples:
        cmd[-1] = "--print-final-samples"
    return cmd


def aggregate_run_outputs(auto_root: Path, dataset_paths: list[Path]):
    run_summaries = []
    all_metric_rows = []
    failed_rows = []

    for csv_path in dataset_paths:
        dataset_name = csv_path.stem
        run_dir = auto_root / dataset_name
        run_summary_path = run_dir / "run_summary.csv"
        metric_summary_path = run_dir / "metric_summary.csv"
        status_path = run_dir / "run_status.txt"

        if run_summary_path.exists() and metric_summary_path.exists():
            run_summaries.extend(pd.read_csv(run_summary_path).to_dict(orient="records"))
            all_metric_rows.extend(pd.read_csv(metric_summary_path).to_dict(orient="records"))
            continue

        status_text = ""
        if status_path.exists():
            status_text = status_path.read_text(encoding="utf-8").strip()
        failed_rows.append(
            {
                "dataset": dataset_name,
                "csv_path": str(csv_path),
                "run_dir": str(run_dir),
                "status": status_text or "missing run outputs",
            }
        )

    if run_summaries:
        run_summary_df = pd.DataFrame(run_summaries).sort_values("dataset")
        metric_summary_df = pd.DataFrame(all_metric_rows).sort_values(["metric_family", "target", "best_score"], ascending=[True, True, False])
        best_by_target_df = (
            metric_summary_df.sort_values(["metric_family", "target", "best_score"], ascending=[True, True, False])
            .groupby(["metric_family", "target"], as_index=False)
            .first()
        )
        global_best_models_df = best_by_target_df[
            [
                "metric_family",
                "target",
                "dataset",
                "csv_path",
                "best_score",
                "best_epoch",
                "checkpoint_path",
                "report_path",
                "feature_schema",
            ]
        ].copy()

        run_summary_df.to_csv(auto_root / "auto_run_summary.csv", index=False)
        metric_summary_df.to_csv(auto_root / "auto_metric_summary.csv", index=False)
        best_by_target_df.to_csv(auto_root / "auto_best_by_target.csv", index=False)
        global_best_models_df.to_csv(auto_root / "auto_best_models_summary.csv", index=False)
        print(f"Saved run summary to {auto_root / 'auto_run_summary.csv'}")
        print(f"Saved metric summary to {auto_root / 'auto_metric_summary.csv'}")
        print(f"Saved best-by-target summary to {auto_root / 'auto_best_by_target.csv'}")
        print(f"Saved best-model summary to {auto_root / 'auto_best_models_summary.csv'}")

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(auto_root / "auto_failed_runs.csv", index=False)
        print(f"Saved failed-run summary to {auto_root / 'auto_failed_runs.csv'}")


def run_datasets_concurrently(args, dataset_paths: list[Path], auto_root: Path):
    device_slots = parse_device_slots(args.device_slots)
    max_parallel = max(1, min(args.max_concurrent, len(device_slots)))
    free_slots = device_slots[:max_parallel]
    pending = list(dataset_paths)
    running = []

    print(f"Using device slots: {free_slots}")

    while pending or running:
        while pending and free_slots:
            csv_path = pending.pop(0)
            dataset_name = csv_path.stem
            device_name = free_slots.pop(0)
            run_dir = auto_root / dataset_name
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "train.log"
            status_path = run_dir / "run_status.txt"
            status_path.write_text(f"running on {device_name}\n", encoding="utf-8")
            log_handle = open(log_path, "w", encoding="utf-8")
            cmd = build_subprocess_command(args, csv_path, device_name)
            print(f"Launching {dataset_name} on {device_name}")
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
                    "dataset_name": dataset_name,
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
                print(f"Finished {item['dataset_name']} on {item['device_name']}")
            else:
                print(f"Run failed for {item['dataset_name']} on {item['device_name']} with rc={retcode}")
        running = still_running


def get_golden_plot_location(base_plot_dir: Path, metric_name: str):
    for metric_prefix in (GOLDEN_SCATTER_PREFIX, GOLDEN_OVERLAP_PREFIX):
        if metric_name.startswith(metric_prefix):
            target_name = metric_name[len(metric_prefix):]
            file_prefix = metric_prefix.rstrip("_")
            return base_plot_dir / target_name, file_prefix
    return base_plot_dir, metric_name


def build_checkpoint_metadata(data: AutoDataBundle, input_dim: int, csv_path: Path, args, report_text=None):
    metadata = {
        "target_cols": data.target_cols,
        "feature_cols": data.feature_cols,
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
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "criterion": CRITERION.__name__,
        "target_weights": TARGET_WEIGHTS.tolist(),
        "mean_weights": MEAN_WEIGHTS.tolist(),
        "overlap_target_index": OVERLAP_TARGET_INDEX,
        "overlap_target_name": data.target_cols[OVERLAP_TARGET_INDEX],
        "canonical_phi": True,
        "canonical_rotation_source": data.rotation_source,
        "source_csv": str(csv_path),
        "feature_schema": data.feature_schema,
    }
    if report_text is not None:
        metadata["report_text"] = report_text
    return metadata


def train_one_dataset(csv_path: Path, auto_root: Path, args, device):
    dataset_name = csv_path.stem
    run_dir = auto_root / dataset_name
    plot_dir = run_dir / "plots"
    golden_model_dir = run_dir / "goldenmodels"
    save_dir = run_dir / "saves"
    golden_summary_file = run_dir / "golden_summary.txt"
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(golden_model_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    data = load_auto_track_data(csv_path, batch_size=args.batch_size, seed=args.seed, device=device, val_fraction=args.val_fraction)
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
    for name in data.target_cols:
        best_vals[f"{GOLDEN_SCATTER_PREFIX}{name}"] = -float("inf")
        if args.track_overlap_golden:
            best_vals[f"{GOLDEN_OVERLAP_PREFIX}{name}"] = -float("inf")

    best_val_loss = float("inf")
    best_val_epoch = 0
    best_model_paths = {}
    best_plot_report_paths = {}

    def save_golden(metric_tag, metric_value, metric_details, epoch, report, overlap_report, plot_quality_report):
        full_report = f"{report}\n{overlap_report}\n{plot_quality_report}"
        metadata = build_checkpoint_metadata(data, input_dim, csv_path, args, report_text=full_report)
        metadata.update({"metric_tag": metric_tag, "metric_value": float(metric_value), "plot_quality_metric": metric_details, "dataset_name": dataset_name})
        save_golden_model(model, optimizer, scheduler, metric_tag, metric_value, epoch, full_report, str(golden_model_dir), metadata)
        best_reports[metric_tag] = full_report
        best_epochs[metric_tag] = epoch + 1
        best_model_paths[metric_tag] = str(golden_model_dir / f"{metric_tag}.pt")
        golden_output_dir, golden_file_prefix = get_golden_plot_location(plot_dir, metric_tag)
        os.makedirs(golden_output_dir, exist_ok=True)
        report_path = golden_output_dir / f"{golden_file_prefix}_training_report.txt"
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write("GOLDEN PLOT-QUALITY TRAINING REPORT\n")
            handle.write("=" * 80 + "\n")
            handle.write(f"metric_tag: {metric_tag}\n")
            handle.write(f"epoch: {epoch + 1}\n")
            handle.write(f"metric_value: {float(metric_value):.6f}\n\n")
            handle.write(full_report)
            handle.write("\n")
        best_plot_report_paths[metric_tag] = str(report_path)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for xb, yb, rb in data.train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            mu, logvar = model(xb)
            loss = CRITERION(yb, mu, logvar, phi_index=data.phi_index, target_weights=TARGET_WEIGHTS, mean_weights=MEAN_WEIGHTS)
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
        with torch.no_grad():
            for xb, yb, rb in data.val_loader:
                xb, yb, rb = xb.to(device), yb.to(device), rb.to(device)
                mu, logvar = model(xb)
                loss = CRITERION(yb, mu, logvar, phi_index=data.phi_index, target_weights=TARGET_WEIGHTS, mean_weights=MEAN_WEIGHTS)
                val_loss += loss.item() * xb.size(0)

                mu_phys = denormalize_and_recover_phi(mu, rb, data.y_mean_t, data.y_std_t, data.phi_index)
                yb_phys = denormalize_and_recover_phi(yb, rb, data.y_mean_t, data.y_std_t, data.phi_index)
                diff = mu_phys - yb_phys
                diff[:, data.phi_index] = wrapped_angle_diff(mu_phys[:, data.phi_index], yb_phys[:, data.phi_index])
                total_val_mae += diff.abs().sum(dim=0)
                total_val_sq += (diff ** 2).sum(dim=0)
                total_count += xb.size(0)
                overlap_pred_parts.append(mu_phys.detach().cpu())
                overlap_true_parts.append(yb_phys.detach().cpu())

        val_loss /= len(data.val_loader.dataset)
        per_target_mae = (total_val_mae / total_count).detach().cpu().numpy()
        per_target_rmse = np.sqrt((total_val_sq / total_count).detach().cpu().numpy())
        overall_val_mae = float(per_target_mae.mean())
        overall_val_rmse = float(per_target_rmse.mean())
        current_lr = optimizer.param_groups[0]["lr"]
        overlap_pred = torch.cat(overlap_pred_parts, dim=0).numpy()
        overlap_true = torch.cat(overlap_true_parts, dim=0).numpy()
        overlap_pred[:, data.phi_index] = np.arctan2(np.sin(overlap_pred[:, data.phi_index]), np.cos(overlap_pred[:, data.phi_index]))
        overlap_true[:, data.phi_index] = np.arctan2(np.sin(overlap_true[:, data.phi_index]), np.cos(overlap_true[:, data.phi_index]))
        target_overlap = compute_target_histogram_overlap(overlap_true, overlap_pred, OVERLAP_TARGET_INDEX, data.target_cols, bins=100)
        plot_quality_scores = compute_plot_quality_scores(y_true=overlap_true, y_pred=overlap_pred, target_cols=data.target_cols, bins=100)
        plot_quality_report = format_plot_quality_report(plot_quality_scores)
        report = format_epoch_report(epoch, args.epochs, train_loss, val_loss, overall_val_mae, overall_val_rmse, per_target_mae, per_target_rmse, data.target_cols)
        overlap_report = f"   Overlap {data.target_cols[OVERLAP_TARGET_INDEX]}: {target_overlap:.6f} | MAE: {float(per_target_mae[OVERLAP_TARGET_INDEX]):.6f}"

        training_history["epoch"].append(epoch + 1)
        training_history["train_loss"].append(train_loss)
        training_history["val_loss"].append(val_loss)
        training_history["val_mean_mae"].append(overall_val_mae)
        training_history["val_mean_rmse"].append(overall_val_rmse)
        training_history["learning_rate"].append(current_lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch + 1
            metadata = build_checkpoint_metadata(data, input_dim, csv_path, args, report_text=f"{report}\n{overlap_report}\n{plot_quality_report}")
            metadata.update({"dataset_name": dataset_name, "checkpoint_type": "best_val_loss", "val_loss": float(val_loss)})
            save_model_checkpoint(str(save_dir / "best_val_loss.pt"), model, optimizer, scheduler, epoch + 1, metadata)

        for target_name in data.target_cols:
            scatter_tag = f"{GOLDEN_SCATTER_PREFIX}{target_name}"
            scatter_metric = plot_quality_scores["scatter"][target_name]
            scatter_score = scatter_metric["score"]
            if scatter_score > best_vals[scatter_tag]:
                best_vals[scatter_tag] = scatter_score
                save_golden(scatter_tag, scatter_score, scatter_metric, epoch, report, overlap_report, plot_quality_report)

            if args.track_overlap_golden:
                overlap_tag = f"{GOLDEN_OVERLAP_PREFIX}{target_name}"
                overlap_metric = plot_quality_scores["overlap"][target_name]
                overlap_score = overlap_metric["score"]
                if overlap_score > best_vals[overlap_tag]:
                    best_vals[overlap_tag] = overlap_score
                    save_golden(overlap_tag, overlap_score, overlap_metric, epoch, report, overlap_report, plot_quality_report)

        print(
            f"[{dataset_name}] Epoch {epoch + 1}/{args.epochs} | train {train_loss:.6f} | "
            f"val {val_loss:.6f} | mean MAE {overall_val_mae:.6f} | mean RMSE {overall_val_rmse:.6f} | "
            f"{data.target_cols[OVERLAP_TARGET_INDEX]} overlap {target_overlap:.6f}"
        )
        scheduler.step()

    final_metadata = build_checkpoint_metadata(data, input_dim, csv_path, args, report_text="final_model")
    final_metadata.update({"dataset_name": dataset_name, "checkpoint_type": "final_model", "best_val_loss": float(best_val_loss), "best_val_epoch": best_val_epoch})
    save_model_checkpoint(str(save_dir / "final_model.pt"), model, optimizer, scheduler, args.epochs, final_metadata)

    history_plot_paths = make_training_history_plots(
        training_history,
        output_dir=str(plot_dir),
        prefix=dataset_name,
        show=False,
        figure_label=dataset_name,
    )
    val_plot_paths = make_canonical_val_diagnostic_plots(
        model=model,
        val_loader=data.val_loader,
        device=device,
        y_mean_t=data.y_mean_t,
        y_std_t=data.y_std_t,
        target_cols=data.target_cols,
        phi_index=data.phi_index,
        output_dir=str(plot_dir),
        prefix=dataset_name,
        bins=100,
        density=True,
        show=args.show_plots,
        scatter_max_points=DIAGNOSTIC_SCATTER_MAX_POINTS,
        central_fraction=DIAGNOSTIC_CENTRAL_FRACTION,
        figure_label=dataset_name,
    )
    if args.print_final_samples:
        print_canonical_final_validation_samples(model, data.val_loader, device, data.y_mean_t, data.y_std_t, data.target_cols, data.phi_index, num_examples=5)

    write_final_golden_summary(str(golden_summary_file), best_reports, best_vals)

    metric_rows = []
    for metric_tag, value in best_vals.items():
        family = "scatter" if metric_tag.startswith(GOLDEN_SCATTER_PREFIX) else "overlap"
        target = metric_tag.split("_")[-1]
        metric_rows.append(
            {
                "dataset": dataset_name,
                "csv_path": str(csv_path),
                "metric_tag": metric_tag,
                "metric_family": family,
                "target": target,
                "best_score": float(value),
                "best_epoch": int(best_epochs.get(metric_tag, 0)),
                "checkpoint_path": best_model_paths.get(metric_tag, ""),
                "report_path": best_plot_report_paths.get(metric_tag, ""),
                "feature_schema": data.feature_schema,
            }
        )

    run_summary = {
        "dataset": dataset_name,
        "csv_path": str(csv_path),
        "rows": int(len(data.x_train) + len(data.x_val)),
        "train_rows": int(len(data.x_train)),
        "val_rows": int(len(data.x_val)),
        "input_dim": input_dim,
        "feature_schema": data.feature_schema,
        "best_val_loss": float(best_val_loss),
        "best_val_epoch": int(best_val_epoch),
        "final_val_loss": float(training_history["val_loss"][-1]),
        "final_val_mean_mae": float(training_history["val_mean_mae"][-1]),
        "final_val_mean_rmse": float(training_history["val_mean_rmse"][-1]),
        "run_dir": str(run_dir),
        "best_val_checkpoint_path": str(save_dir / "best_val_loss.pt"),
        "final_checkpoint_path": str(save_dir / "final_model.pt"),
        "history_plot_count": len(history_plot_paths),
        "val_plot_count": len(val_plot_paths),
    }
    pd.DataFrame(metric_rows).to_csv(run_dir / "metric_summary.csv", index=False)
    pd.DataFrame([run_summary]).to_csv(run_dir / "run_summary.csv", index=False)
    return run_summary, metric_rows


def main():
    args = parse_args()
    auto_root = Path(args.output_dir)
    auto_root.mkdir(parents=True, exist_ok=True)

    if args.single_dataset:
        device = resolve_device(args.device)
        set_seed(args.seed)
        print(f"Device set to {device}")
        csv_path = Path(args.single_dataset)
        summary, metric_rows = train_one_dataset(csv_path, auto_root, args, device)
        run_dir = auto_root / csv_path.stem
        pd.DataFrame(metric_rows).to_csv(run_dir / "metric_summary.csv", index=False)
        pd.DataFrame([summary]).to_csv(run_dir / "run_summary.csv", index=False)
        return

    dataset_paths = discover_dataset_paths(args.data_dir)
    if args.max_concurrent <= 1:
        device = resolve_device(args.device)
        set_seed(args.seed)
        print(f"Device set to {device}")
        for csv_path in dataset_paths:
            train_one_dataset(csv_path, auto_root, args, device)
    else:
        run_datasets_concurrently(args, dataset_paths, auto_root)

    aggregate_run_outputs(auto_root, dataset_paths)


if __name__ == "__main__":
    main()
