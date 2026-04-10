import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


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
        # ===== Ploting distributions helpers ======
# ===== Ploting distributions helpers ======
def collect_val_predictions_and_targets(
    model,
    val_loader,
    device,
    y_mean_t,
    y_std_t,
    phi_index,
    denormalize_targets,
):
    """
    Run model on val_loader and return de-normalized predictions + targets
    as numpy arrays with shape [N, n_targets].
    """
    model.eval()

    all_pred = []
    all_true = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            out = model(xb)

            # handle both model types
            if isinstance(out, tuple):
                mu = out[0]
            else:
                mu = out

            # de-normalize using the actual function signature
            mu_phys = denormalize_targets(mu, y_mean_t, y_std_t)
            yb_phys = denormalize_targets(yb, y_mean_t, y_std_t)

            # wrap phi into [-pi, pi]
            mu_phys = mu_phys.clone()
            yb_phys = yb_phys.clone()

            mu_phys[:, phi_index] = torch.atan2(
                torch.sin(mu_phys[:, phi_index]),
                torch.cos(mu_phys[:, phi_index])
            )
            yb_phys[:, phi_index] = torch.atan2(
                torch.sin(yb_phys[:, phi_index]),
                torch.cos(yb_phys[:, phi_index])
            )

            all_pred.append(mu_phys.detach().cpu())
            all_true.append(yb_phys.detach().cpu())

    all_pred = torch.cat(all_pred, dim=0).numpy()
    all_true = torch.cat(all_true, dim=0).numpy()

    return all_pred, all_true

def plot_pred_vs_true_distributions(
    y_true,
    y_pred,
    target_cols,
    bins=100,
    density=True,
    save_path=None,
    show=True,
):
    """
    Make a multi-panel histogram comparing predicted vs actual distributions.
    """
    n_targets = len(target_cols)

    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 4))

    if n_targets == 1:
        axes = [axes]

    for i, name in enumerate(target_cols):
        ax = axes[i]

        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]

        vmin = min(true_vals.min(), pred_vals.min())
        vmax = max(true_vals.max(), pred_vals.max())

        # manual bounds tuning
        if i == 0:
            vmin, vmax = -0.05, 0.05
        elif i == 3:
            vmin, vmax = -0.005, 0.005

        bin_edges = np.linspace(vmin, vmax, bins + 1)

        ax.hist(true_vals, bins=bin_edges, alpha=0.5, label="Actual", density=density)
        ax.hist(pred_vals, bins=bin_edges, alpha=0.5, label="Predicted", density=density)

        ax.set_title(name)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density" if density else "Count")
        ax.legend()

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def make_val_distribution_plots(
    model,
    val_loader,
    device,
    y_mean_t,
    y_std_t,
    target_cols,
    phi_index,
    denormalize_targets,
    save_path=None,
    bins=100,
    density=True,
    show=True,
):
    """
    Wrapper:
    1) collect predictions/targets
    2) plot distributions
    """
    y_pred, y_true = collect_val_predictions_and_targets(
        model=model,
        val_loader=val_loader,
        device=device,
        y_mean_t=y_mean_t,
        y_std_t=y_std_t,
        phi_index=phi_index,
        denormalize_targets=denormalize_targets,
    )

    plot_pred_vs_true_distributions(
        y_true=y_true,
        y_pred=y_pred,
        target_cols=target_cols,
        bins=bins,
        density=density,
        save_path=save_path,
        show=show,
    )