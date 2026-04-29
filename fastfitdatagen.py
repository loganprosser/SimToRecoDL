import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from helpers_data import DEFAULT_DATA_PATH, set_seed
from helpers_vis import (
    compute_plot_quality_scores,
    compute_target_histogram_overlap,
    format_plot_quality_report,
    plot_distance_distribution,
    plot_overlap_distributions,
    plot_pred_vs_true_scatter,
    plot_residual_distributions,
)
from train_fastfit_residual_auto import (
    DIAGNOSTIC_CENTRAL_FRACTION,
    DIAGNOSTIC_SCATTER_MAX_POINTS,
    OVERLAP_TARGET_INDEX,
    load_fastfit_track_data,
    maybe_recover_canonical_phi,
    reconstruct_full_targets,
)

DEFAULT_OUTPUT_DIR = "fastfit_compare"
DEFAULT_FASTFIT_PRECOMPUTE_WORKERS = 16
DEFAULT_DATALOADER_WORKERS = 0
TRAINING_MODES = ("canonical", "raw")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate canonical/raw validation plots using the deterministic fast-fit baseline only."
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to the CSV to evaluate.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for plots and reports.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--dataloader-workers", type=int, default=DEFAULT_DATALOADER_WORKERS)
    parser.add_argument(
        "--fastfit-precompute-workers",
        type=int,
        default=DEFAULT_FASTFIT_PRECOMPUTE_WORKERS,
        help="CPU worker count for one-time fast-fit cache generation.",
    )
    parser.add_argument("--show-plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print-final-samples", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--modes",
        default="canonical",
        help="Comma-separated modes to evaluate. Supported: canonical, raw.",
    )
    return parser.parse_args()


def parse_modes(modes_arg: str) -> list[str]:
    modes = [mode.strip() for mode in modes_arg.split(",") if mode.strip()]
    invalid = [mode for mode in modes if mode not in TRAINING_MODES]
    if invalid:
        raise ValueError(f"Unsupported modes: {invalid}. Supported: {TRAINING_MODES}")
    if not modes:
        raise ValueError("At least one mode must be requested.")
    return modes


def get_device():
    return torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )


def collect_fastfit_predictions_and_truth(data, device):
    all_true = []
    all_fast = []

    with torch.no_grad():
        for _, yb, rb, fastb in data.val_loader:
            yb = yb.to(device)
            rb = rb.to(device)
            fastb = fastb.to(device)

            true_resid_phys = yb * data.y_std_t + data.y_mean_t
            true_proc = reconstruct_full_targets(fastb, true_resid_phys, data.phi_index)
            fast_proc = fastb

            true_phys = maybe_recover_canonical_phi(true_proc, rb, data.phi_index, data.training_mode)
            fast_phys = maybe_recover_canonical_phi(fast_proc, rb, data.phi_index, data.training_mode)

            all_true.append(true_phys.detach().cpu())
            all_fast.append(fast_phys.detach().cpu())

    y_true = torch.cat(all_true, dim=0).numpy()
    y_fast = torch.cat(all_fast, dim=0).numpy()
    return y_true, y_fast


def compute_residuals(y_true, y_pred, phi_index):
    residuals = y_pred - y_true
    if phi_index is not None:
        residuals[:, phi_index] = np.arctan2(
            np.sin(residuals[:, phi_index]),
            np.cos(residuals[:, phi_index]),
        )
    return residuals


def make_fastfit_diagnostic_plots(
    y_true,
    y_fast,
    target_cols,
    phi_index,
    output_dir,
    prefix,
    show,
    figure_label,
):
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "overlap": os.path.join(output_dir, f"{prefix}_overlap.png"),
        "scatter": os.path.join(output_dir, f"{prefix}_scatter_pred_vs_actual.png"),
        "residual": os.path.join(output_dir, f"{prefix}_residual.png"),
        "distance": os.path.join(output_dir, f"{prefix}_distance.png"),
    }

    plot_overlap_distributions(
        y_true=y_true,
        y_pred=y_fast,
        target_cols=target_cols,
        bins=100,
        density=True,
        save_path=paths["overlap"],
        show=show,
        central_fraction=DIAGNOSTIC_CENTRAL_FRACTION,
        figure_label=figure_label,
    )
    plot_pred_vs_true_scatter(
        y_true=y_true,
        y_pred=y_fast,
        target_cols=target_cols,
        save_path=paths["scatter"],
        show=show,
        max_points=DIAGNOSTIC_SCATTER_MAX_POINTS,
        central_fraction=DIAGNOSTIC_CENTRAL_FRACTION,
        figure_label=figure_label,
    )
    plot_residual_distributions(
        y_true=y_true,
        y_pred=y_fast,
        target_cols=target_cols,
        phi_index=phi_index,
        bins=100,
        density=True,
        save_path=paths["residual"],
        show=show,
        figure_label=figure_label,
    )
    plot_distance_distribution(
        y_true=y_true,
        y_pred=y_fast,
        target_cols=target_cols,
        phi_index=phi_index,
        bins=100,
        density=True,
        save_path=paths["distance"],
        show=show,
        figure_label=figure_label,
    )
    return paths


def build_fastfit_report(data, y_true, y_fast):
    residuals = compute_residuals(y_true, y_fast, data.phi_index)
    per_target_mae = np.mean(np.abs(residuals), axis=0)
    per_target_rmse = np.sqrt(np.mean(residuals ** 2, axis=0))
    mean_mae = float(per_target_mae.mean())
    mean_rmse = float(per_target_rmse.mean())

    target_overlap = compute_target_histogram_overlap(
        y_true=y_true,
        y_pred=y_fast,
        target_index=OVERLAP_TARGET_INDEX,
        target_cols=data.target_cols,
        bins=100,
    )
    plot_quality_scores = compute_plot_quality_scores(
        y_true=y_true,
        y_pred=y_fast,
        target_cols=data.target_cols,
        bins=100,
    )
    overall_scatter_score = float(
        np.mean([plot_quality_scores["scatter"][name]["score"] for name in data.target_cols])
    )
    overall_overlap_score = float(
        np.mean([plot_quality_scores["overlap"][name]["score"] for name in data.target_cols])
    )

    lines = [
        f"mode: {data.training_mode}",
        f"rows: {len(data.x_train) + len(data.x_val)}",
        f"train_rows: {len(data.x_train)}",
        f"val_rows: {len(data.x_val)}",
        f"n_fastfit_failures: {data.n_fastfit_failures}",
        f"rotation_source: {data.rotation_source}",
        "",
        "Fast-fit validation metrics:",
        f"   mean_mae: {mean_mae:.6f}",
        f"   mean_rmse: {mean_rmse:.6f}",
        f"   overlap_{data.target_cols[OVERLAP_TARGET_INDEX]}: {target_overlap:.6f}",
        f"   mean_scatter_score: {overall_scatter_score:.6f}",
        f"   mean_overlap_score: {overall_overlap_score:.6f}",
        "",
        "Per-target metrics:",
    ]
    for idx, name in enumerate(data.target_cols):
        lines.append(
            f"   {name}: mae={float(per_target_mae[idx]):.6f} | rmse={float(per_target_rmse[idx]):.6f}"
        )
    lines.extend(["", format_plot_quality_report(plot_quality_scores)])

    summary = {
        "mode": data.training_mode,
        "rows": int(len(data.x_train) + len(data.x_val)),
        "train_rows": int(len(data.x_train)),
        "val_rows": int(len(data.x_val)),
        "mean_mae": mean_mae,
        "mean_rmse": mean_rmse,
        "overlap_target": data.target_cols[OVERLAP_TARGET_INDEX],
        "overlap_score": float(target_overlap),
        "mean_scatter_score": overall_scatter_score,
        "mean_overlap_score": overall_overlap_score,
        "n_fastfit_failures": int(data.n_fastfit_failures),
        "rotation_source": data.rotation_source,
    }
    return "\n".join(lines), summary


def print_final_samples(data, y_true, y_fast, num_examples=5):
    print("\n" + "=" * 80)
    print("FAST-FIT VALIDATION SAMPLES")
    print("=" * 80)
    residuals = compute_residuals(y_true.copy(), y_fast.copy(), data.phi_index)
    for row_idx in range(min(num_examples, len(y_true))):
        print(f"\nValidation example {row_idx + 1}")
        for col_idx, name in enumerate(data.target_cols):
            print(
                f"  {name:8s} | "
                f"true = {y_true[row_idx, col_idx]: .6f} | "
                f"fast = {y_fast[row_idx, col_idx]: .6f} | "
                f"error = {residuals[row_idx, col_idx]: .6f}"
            )


def run_mode(csv_path: Path, output_root: Path, args, device, mode: str):
    run_dir = output_root / mode
    final_plot_dir = run_dir / "plots" / "final"
    final_plot_dir.mkdir(parents=True, exist_ok=True)

    data = load_fastfit_track_data(
        csv_path=csv_path,
        output_dir=output_root,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        val_fraction=args.val_fraction,
        dataloader_workers=args.dataloader_workers,
        training_mode=mode,
        fastfit_precompute_workers=args.fastfit_precompute_workers,
    )
    y_true, y_fast = collect_fastfit_predictions_and_truth(data, device)
    plot_paths = make_fastfit_diagnostic_plots(
        y_true=y_true,
        y_fast=y_fast,
        target_cols=data.target_cols,
        phi_index=data.phi_index,
        output_dir=str(final_plot_dir),
        prefix="final",
        show=args.show_plots,
        figure_label=f"{mode} | fastfit final",
    )
    report_text, summary = build_fastfit_report(data, y_true, y_fast)

    report_path = final_plot_dir / "final_training_report.txt"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("FAST-FIT BASELINE REPORT\n")
        handle.write("=" * 80 + "\n")
        handle.write(report_text)
        handle.write("\n")

    if args.print_final_samples:
        print_final_samples(data, y_true, y_fast)

    pd.DataFrame(
        [
            {
                "mode": mode,
                "csv_path": str(csv_path),
                "snapshot": "final",
                "metric_tag": "fastfit_baseline",
                "score": float(summary["mean_scatter_score"]),
                "epoch": 0,
                "checkpoint_path": "",
                "report_path": str(report_path),
                "plot_dir": str(final_plot_dir),
            }
        ]
    ).to_csv(run_dir / "model_summary.csv", index=False)

    summary.update(
        {
            "csv_path": str(csv_path),
            "run_dir": str(run_dir),
            "final_plot_dir": str(final_plot_dir),
            "final_plot_count": len(plot_paths),
            "report_path": str(report_path),
        }
    )
    pd.DataFrame([summary]).to_csv(run_dir / "run_summary.csv", index=False)
    return summary


def main():
    args = parse_args()
    modes = parse_modes(args.modes)
    csv_path = Path(args.data_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV does not exist: {csv_path}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    device = get_device()
    set_seed(args.seed)
    print(f"Device set to {device}")

    summaries = []
    for mode in modes:
        print(f"Generating fast-fit plots for mode: {mode}")
        summaries.append(run_mode(csv_path, output_root, args, device, mode))

    pd.DataFrame(summaries).to_csv(output_root / "mode_run_summary.csv", index=False)


if __name__ == "__main__":
    main()
