import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from helpers import format_epoch_report, save_model_checkpoint, wrapped_angle_diff
from helpers_canonical_phi import (
    denormalize_and_recover_phi,
    load_canonical_phi_track_data,
    make_canonical_val_diagnostic_plots,
    print_canonical_data_shapes,
    print_canonical_final_validation_samples,
)
from helpers_data import DEFAULT_DATA_PATH, set_seed
from helpers_vis import (
    compute_plot_quality_scores,
    compute_target_histogram_overlap,
    format_plot_quality_report,
    make_training_history_plots,
)
from loss import hetero_gaussian_nll_with_phi
from model import HeteroTrackNet

# ====== Running Constants =======
DATA_PATH = DEFAULT_DATA_PATH
OUTPUT_DIR = "canonicalphi_compare"
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
BEST_OVERALL_SCATTER_TAG = "best_overall_scatter"

# ====== Running Flags =======
CHECK_SHAPE = False
CHECK_MASK_COUNTS = False
TEST_TRAIN = False
TRAIN = True
PRINT_FINAL_VAL_SAMPLES = True
SHOW_FINAL_PLOTS = True


device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Device set to {device}")
set_seed(SEED)


def build_checkpoint_metadata(data, input_dim, report_text=None):
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
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "criterion": CRITERION.__name__,
        "target_weights": TARGET_WEIGHTS.tolist(),
        "mean_weights": MEAN_WEIGHTS.tolist(),
        "overlap_target_index": OVERLAP_TARGET_INDEX,
        "overlap_target_name": data.target_cols[OVERLAP_TARGET_INDEX],
        "canonical_phi": True,
        "canonical_rotation_source": data.rotation_source,
        "canonical_rotation_sign": "inputs_xy_and_target_phi_minus_rotation",
        "source_csv": DATA_PATH,
    }
    if report_text is not None:
        metadata["report_text"] = report_text
    return metadata


def collect_val_arrays(model, data):
    model.eval()
    pred_parts = []
    true_parts = []
    with torch.no_grad():
        for xb, yb, rb in data.val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            rb = rb.to(device)
            mu, _ = model(xb)
            mu_phys = denormalize_and_recover_phi(mu, rb, data.y_mean_t, data.y_std_t, data.phi_index)
            yb_phys = denormalize_and_recover_phi(yb, rb, data.y_mean_t, data.y_std_t, data.phi_index)
            pred_parts.append(mu_phys.detach().cpu())
            true_parts.append(yb_phys.detach().cpu())
    y_pred = torch.cat(pred_parts, dim=0).numpy()
    y_true = torch.cat(true_parts, dim=0).numpy()
    return y_pred, y_true


def save_snapshot_outputs(model, data, output_dir, prefix, metric_value, epoch, report_text, show):
    plot_paths = make_canonical_val_diagnostic_plots(
        model=model,
        val_loader=data.val_loader,
        device=device,
        y_mean_t=data.y_mean_t,
        y_std_t=data.y_std_t,
        target_cols=data.target_cols,
        phi_index=data.phi_index,
        output_dir=output_dir,
        prefix=prefix,
        bins=100,
        density=True,
        show=show,
        scatter_max_points=DIAGNOSTIC_SCATTER_MAX_POINTS,
        central_fraction=DIAGNOSTIC_CENTRAL_FRACTION,
        figure_label=prefix,
    )
    report_path = os.path.join(output_dir, f"{prefix}_training_report.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(f"{prefix.upper()} MODEL REPORT\n")
        handle.write("=" * 80 + "\n")
        handle.write(f"epoch: {epoch}\n")
        handle.write(f"metric_value: {float(metric_value):.6f}\n\n")
        handle.write(report_text)
        handle.write("\n")
    return plot_paths, report_path


data = load_canonical_phi_track_data(
    csv_path=DATA_PATH,
    batch_size=BATCH_SIZE,
    seed=SEED,
    device=device,
    val_fraction=VAL_FRACTION,
    print_mask_counts=CHECK_MASK_COUNTS,
)

if CHECK_SHAPE:
    print_canonical_data_shapes(data)

input_dim = data.x_train.shape[1]
model = HeteroTrackNet(
    input_dim=input_dim,
    hidden_layers=HIDDEN_LAYERS,
    output_dim=len(data.target_cols),
    use_batchnorm=BATCH_NORM,
    dropout=DROPOUT,
    activation=nn.ReLU,
).to(device)
print(model)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

if TEST_TRAIN:
    xb, yb, _ = next(iter(data.train_loader))
    xb, yb = xb.to(device), yb.to(device)
    mu, logvar = model(xb)
    loss = CRITERION(
        yb,
        mu,
        logvar,
        phi_index=data.phi_index,
        target_weights=TARGET_WEIGHTS,
        mean_weights=MEAN_WEIGHTS,
    )
    print("mu shape:", mu.shape)
    print("logvar shape:", logvar.shape)
    print("initial loss:", loss.item())

training_history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "val_mean_mae": [],
    "val_mean_rmse": [],
    "learning_rate": [],
}

run_dir = OUTPUT_DIR
plot_dir = os.path.join(run_dir, "plots")
final_plot_dir = os.path.join(plot_dir, "final")
best_overall_plot_dir = os.path.join(plot_dir, "best_overall")
best_val_loss_plot_dir = os.path.join(plot_dir, "best_val_loss")
save_dir = os.path.join(run_dir, "saves")
os.makedirs(final_plot_dir, exist_ok=True)
os.makedirs(best_overall_plot_dir, exist_ok=True)
os.makedirs(best_val_loss_plot_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

best_val_loss = float("inf")
best_val_epoch = 0
best_val_loss_plot_paths = {}
best_val_loss_report_path = ""
best_overall_scatter_score = -float("inf")
best_overall_scatter_epoch = 0
best_overall_plot_paths = {}
best_overall_report_path = ""
best_overall_checkpoint_path = ""

if TRAIN:
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb, _ in data.train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
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

        with torch.no_grad():
            for xb, yb, rb in data.val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                rb = rb.to(device)
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
        target_overlap = compute_target_histogram_overlap(
            y_true=overlap_true,
            y_pred=overlap_pred,
            target_index=OVERLAP_TARGET_INDEX,
            target_cols=data.target_cols,
            bins=100,
        )
        plot_quality_scores = compute_plot_quality_scores(
            y_true=overlap_true,
            y_pred=overlap_pred,
            target_cols=data.target_cols,
            bins=100,
        )
        plot_quality_report = format_plot_quality_report(plot_quality_scores)
        overall_scatter_score = float(np.mean([plot_quality_scores["scatter"][name]["score"] for name in data.target_cols]))
        overall_overlap_score = float(np.mean([plot_quality_scores["overlap"][name]["score"] for name in data.target_cols]))
        report = format_epoch_report(
            epoch,
            EPOCHS,
            train_loss,
            val_loss,
            overall_val_mae,
            overall_val_rmse,
            per_target_mae,
            per_target_rmse,
            data.target_cols,
        )
        overlap_report = (
            f"   Overlap {data.target_cols[OVERLAP_TARGET_INDEX]}: "
            f"{target_overlap:.6f} | MAE: {float(per_target_mae[OVERLAP_TARGET_INDEX]):.6f}"
        )
        overall_report = (
            f"   Overall model-selection scores:\n"
            f"      mean_scatter_score: {overall_scatter_score:.6f}\n"
            f"      mean_overlap_score: {overall_overlap_score:.6f}"
        )
        full_report = f"{report}\n{overlap_report}\n{overall_report}\n{plot_quality_report}"

        training_history["epoch"].append(epoch + 1)
        training_history["train_loss"].append(train_loss)
        training_history["val_loss"].append(val_loss)
        training_history["val_mean_mae"].append(overall_val_mae)
        training_history["val_mean_rmse"].append(overall_val_rmse)
        training_history["learning_rate"].append(current_lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch + 1
            metadata = build_checkpoint_metadata(data, input_dim, report_text=full_report)
            metadata.update({"checkpoint_type": "best_val_loss", "val_loss": float(val_loss)})
            save_model_checkpoint(os.path.join(save_dir, "best_val_loss.pt"), model, optimizer, scheduler, epoch + 1, metadata)
            best_val_loss_plot_paths, best_val_loss_report_path = save_snapshot_outputs(
                model,
                data,
                best_val_loss_plot_dir,
                "best_val_loss",
                epoch + 1,
                float(val_loss),
                full_report,
                False,
            )

        if overall_scatter_score > best_overall_scatter_score:
            best_overall_scatter_score = overall_scatter_score
            best_overall_scatter_epoch = epoch + 1
            best_overall_checkpoint_path = os.path.join(save_dir, "best_overall.pt")
            metadata = build_checkpoint_metadata(data, input_dim, report_text=full_report)
            metadata.update(
                {
                    "checkpoint_type": BEST_OVERALL_SCATTER_TAG,
                    "metric_tag": BEST_OVERALL_SCATTER_TAG,
                    "metric_value": float(overall_scatter_score),
                }
            )
            save_model_checkpoint(best_overall_checkpoint_path, model, optimizer, scheduler, epoch + 1, metadata)
            best_overall_plot_paths, best_overall_report_path = save_snapshot_outputs(
                model,
                data,
                best_overall_plot_dir,
                "best_overall",
                epoch + 1,
                overall_scatter_score,
                full_report,
                False,
            )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train {train_loss:.6f} | val {val_loss:.6f} | "
            f"mean MAE {overall_val_mae:.6f} | mean RMSE {overall_val_rmse:.6f} | "
            f"{data.target_cols[OVERLAP_TARGET_INDEX]} overlap {target_overlap:.6f} | "
            f"overall scatter {overall_scatter_score:.6f}"
        )
        scheduler.step()

final_y_pred, final_y_true = collect_val_arrays(model, data)
final_plot_quality_scores = compute_plot_quality_scores(
    y_true=final_y_true,
    y_pred=final_y_pred,
    target_cols=data.target_cols,
    bins=100,
)
final_target_overlap = compute_target_histogram_overlap(
    y_true=final_y_true,
    y_pred=final_y_pred,
    target_index=OVERLAP_TARGET_INDEX,
    target_cols=data.target_cols,
    bins=100,
)
final_per_target_mae = np.mean(np.abs(final_y_pred - final_y_true), axis=0)
final_per_target_rmse = np.sqrt(np.mean((final_y_pred - final_y_true) ** 2, axis=0))
final_overall_scatter_score = float(np.mean([final_plot_quality_scores["scatter"][name]["score"] for name in data.target_cols]))
final_overall_overlap_score = float(np.mean([final_plot_quality_scores["overlap"][name]["score"] for name in data.target_cols]))
final_epoch_report = format_epoch_report(
    EPOCHS - 1,
    EPOCHS,
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
    f"   Overlap {data.target_cols[OVERLAP_TARGET_INDEX]}: {final_target_overlap:.6f} | MAE: {float(final_per_target_mae[OVERLAP_TARGET_INDEX]):.6f}\n"
    f"   Overall model-selection scores:\n"
    f"      mean_scatter_score: {final_overall_scatter_score:.6f}\n"
    f"      mean_overlap_score: {final_overall_overlap_score:.6f}\n"
    f"{format_plot_quality_report(final_plot_quality_scores)}"
)

final_metadata = build_checkpoint_metadata(data, input_dim, report_text=final_full_report)
final_metadata.update({"checkpoint_type": "final_model", "best_val_loss": float(best_val_loss), "best_val_epoch": best_val_epoch})
save_model_checkpoint(os.path.join(save_dir, "final_model.pt"), model, optimizer, scheduler, EPOCHS, final_metadata)

history_plot_paths = make_training_history_plots(
    history=training_history,
    output_dir=final_plot_dir,
    prefix="canonicalphi",
    show=False,
    figure_label="canonicalphi | final",
)
final_plot_paths, final_report_path = save_snapshot_outputs(
    model,
    data,
    final_plot_dir,
    "final",
    EPOCHS,
    final_overall_scatter_score,
    final_full_report,
    SHOW_FINAL_PLOTS,
)

if PRINT_FINAL_VAL_SAMPLES:
    print_canonical_final_validation_samples(
        model,
        data.val_loader,
        device,
        data.y_mean_t,
        data.y_std_t,
        data.target_cols,
        data.phi_index,
        num_examples=5,
    )

model_rows = [
    {
        "mode": "canonical",
        "csv_path": DATA_PATH,
        "snapshot": "best_val_loss",
        "metric_tag": "best_val_loss",
        "score": float(best_val_loss),
        "epoch": int(best_val_epoch),
        "checkpoint_path": os.path.join(save_dir, "best_val_loss.pt"),
        "report_path": best_val_loss_report_path,
        "plot_dir": best_val_loss_plot_dir,
    },
    {
        "mode": "canonical",
        "csv_path": DATA_PATH,
        "snapshot": "best_overall",
        "metric_tag": BEST_OVERALL_SCATTER_TAG,
        "score": float(best_overall_scatter_score),
        "epoch": int(best_overall_scatter_epoch),
        "checkpoint_path": best_overall_checkpoint_path,
        "report_path": best_overall_report_path,
        "plot_dir": best_overall_plot_dir,
    },
    {
        "mode": "canonical",
        "csv_path": DATA_PATH,
        "snapshot": "final",
        "metric_tag": "final_model",
        "score": float(final_overall_scatter_score),
        "epoch": int(EPOCHS),
        "checkpoint_path": os.path.join(save_dir, "final_model.pt"),
        "report_path": final_report_path,
        "plot_dir": final_plot_dir,
    },
]

run_summary = {
    "mode": "canonical",
    "csv_path": DATA_PATH,
    "rows": int(len(data.x_train) + len(data.x_val)),
    "train_rows": int(len(data.x_train)),
    "val_rows": int(len(data.x_val)),
    "input_dim": input_dim,
    "best_val_loss": float(best_val_loss),
    "best_val_epoch": int(best_val_epoch),
    "final_val_loss": float(training_history["val_loss"][-1]),
    "final_val_mean_mae": float(training_history["val_mean_mae"][-1]),
    "final_val_mean_rmse": float(training_history["val_mean_rmse"][-1]),
    "run_dir": run_dir,
    "final_plot_dir": final_plot_dir,
    "best_overall_plot_dir": best_overall_plot_dir,
    "best_val_loss_plot_dir": best_val_loss_plot_dir,
    "best_val_loss_plot_count": len(best_val_loss_plot_paths),
    "best_overall_scatter_epoch": int(best_overall_scatter_epoch),
    "best_overall_scatter_score": float(best_overall_scatter_score),
    "best_val_checkpoint_path": os.path.join(save_dir, "best_val_loss.pt"),
    "best_overall_checkpoint_path": best_overall_checkpoint_path,
    "final_checkpoint_path": os.path.join(save_dir, "final_model.pt"),
    "history_plot_count": len(history_plot_paths),
    "final_plot_count": len(final_plot_paths),
    "best_overall_plot_count": len(best_overall_plot_paths),
}

os.makedirs(run_dir, exist_ok=True)
pd.DataFrame(model_rows).to_csv(os.path.join(run_dir, "model_summary.csv"), index=False)
pd.DataFrame([run_summary]).to_csv(os.path.join(run_dir, "run_summary.csv"), index=False)
