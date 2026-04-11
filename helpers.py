import os
import torch


def angle_diff(pred, target):
    return torch.atan2(
        torch.sin(pred - target),
        torch.cos(pred - target)
    )

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

    checkpoint = build_model_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch + 1,
        metadata={
            "metric_tag": metric_tag,
            "metric_value": float(metric_value),
            "report_text": report_text,
            **metadata,
        },
    )

    torch.save(checkpoint, save_path)


def build_model_checkpoint(
    model,
    optimizer=None,
    scheduler=None,
    epoch=None,
    metadata=None,
):
    checkpoint = {
        "checkpoint_version": 1,
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if metadata is not None:
        checkpoint.update(metadata)

    return checkpoint


def save_model_checkpoint(
    save_path,
    model,
    optimizer=None,
    scheduler=None,
    epoch=None,
    metadata=None,
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    checkpoint = build_model_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        metadata=metadata,
    )
    torch.save(checkpoint, save_path)


def write_final_golden_summary(summary_file, best_reports, best_values):
    with open(summary_file, "w") as f:
        f.write("FINAL BEST RESULTS BY METRIC\n")
        f.write("=" * 100 + "\n\n")

        for metric_name in best_reports:
            f.write(f"{metric_name}: {best_values[metric_name]:.6f}\n")
            f.write(best_reports[metric_name] + "\n")
            f.write("\n" + "-" * 100 + "\n\n")
