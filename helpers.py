import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import random

def angle_diff(pred, target):
    return torch.atan2(
        torch.sin(pred - target),
        torch.cos(pred - target)
    )

def print_final_validation_samples(
    model,
    val_loader,
    device,
    denormalize_targets,
    y_mean_t,
    y_std_t,
    TARGET_COLS,
    PHI_INDEX,
    wrapped_angle_diff,
    num_examples=10
):
    print("\n" + "="*80)
    print("FINAL VALIDATION SAMPLES")
    print("="*80)

    model.eval()
    shown = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            mu, logvar = model(xb)

            # std in normalized space
            std = torch.exp(0.5 * logvar)

            # de-normalize
            mu_phys = denormalize_targets(mu, y_mean_t, y_std_t)
            yb_phys = denormalize_targets(yb, y_mean_t, y_std_t)

            # uncertainty in physical units
            std_phys = std * y_std_t

            # error in physical units
            err = mu_phys - yb_phys

            # wrap phi error
            err[:, PHI_INDEX] = wrapped_angle_diff(
                mu_phys[:, PHI_INDEX],
                yb_phys[:, PHI_INDEX]
            )

            for i in range(xb.size(0)):
                print(f"\nValidation example {shown + 1}")
                for j, name in enumerate(TARGET_COLS):
                    print(
                        f"  {name:8s} | "
                        f"true = {yb_phys[i, j].item(): .6f} | "
                        f"pred = {mu_phys[i, j].item(): .6f} | "
                        f"error = {err[i, j].item(): .6f} | "
                        f"uncertainty_std = {std_phys[i, j].item(): .6f}"
                    )

                shown += 1
                if shown >= num_examples:
                    return

# ===== Math helpers =====
def wrapped_angle_diff(pred, target):
    return torch.atan2(torch.sin(pred - target), torch.cos(pred - target))


def denormalize_targets(y_norm, y_mean_t, y_std_t):
    return y_norm * y_std_t + y_mean_t


# ===== Logging / reporting =====
def format_epoch_report(
    epoch,
    total_epochs,
    train_loss,
    val_loss,
    overall_val_mae,
    overall_val_rmse,
    per_target_mae,
    per_target_rmse,
    target_cols
):
    lines = []

    lines.append(
        f"EPOCH {epoch + 1}/{total_epochs} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"Val Mean MAE: {overall_val_mae:.6f} | "
        f"Val Mean RMSE: {overall_val_rmse:.6f}"
    )

    lines.append("   Per-target MAE:")
    for name, val in zip(target_cols, per_target_mae):
        lines.append(f"      {name}: {val:.6f}")

    lines.append("   Per-target RMSE:")
    for name, val in zip(target_cols, per_target_rmse):
        lines.append(f"      {name}: {val:.6f}")

    return "\n".join(lines)

# ===== Golden model tracking =====
def save_golden_model(
    model,
    optimizer,
    scheduler,
    metric_tag,
    metric_value,
    epoch,
    report_text,
    save_dir,
    metadata
):
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{metric_tag}.pt")

    torch.save(
        {
            "metric_tag": metric_tag,
            "metric_value": float(metric_value),
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            **metadata
        },
        save_path
    )


def write_final_golden_summary(summary_file, best_reports, best_values):
    with open(summary_file, "w") as f:
        f.write("FINAL BEST RESULTS BY METRIC\n")
        f.write("=" * 100 + "\n\n")

        for metric_name in best_reports:
            f.write(f"{metric_name}: {best_values[metric_name]:.6f}\n")
            f.write(best_reports[metric_name] + "\n")
            f.write("\n" + "-" * 100 + "\n\n")