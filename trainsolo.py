import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from helpers import (
    denormalize_targets,
    format_epoch_report,
    save_model_checkpoint,
)
from helpers_data import load_track_data, print_data_shapes, set_seed
from helpers_vis import (
    compute_target_histogram_overlap,
    make_training_history_plots,
    make_val_diagnostic_plots,
    plot_overlap_history,
    print_final_validation_samples,
)
from loss import hetero_gaussian_nll_with_phi
from model import HeteroTrackNet

# TODO maybe use another togglable filter becasue overlap can be bias include like a density matching penality instead of straight overlap



# ====== Running Constants =======
EPOCHS = 2000
BATCH_SIZE = 256
HIDDEN_LAYERS = [2048, 2048, 1024, 512]
TARGET_COLS = ["pca_dxy"]
CRITERION = hetero_gaussian_nll_with_phi

# ====== Running Flags =======
CHECK_SHAPE = False
TEST_TRAIN = False
TRAIN = True
PRINT_FINAL_VAL_SAMPLES = True
SAVE_BEST_MODELS = True
PLOT_VAL_DISTRIBUTIONS = True
PLOT_TRAINING_HISTORY = True
PLOT_OVERLAP_HISTORY = True

# ====== Save settings ======
SAVE_DIR = "solomodelsRUN3"
PLOT_DIR = "plotsSOLO"
PLOT_PREFIX = "solo_pca_dxy"
OVERLAP_TARGET_INDEX = 0

# ===== Picking Device ========
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Device set to {device}")


# ==== Setting Seed =====
SEED = 42
set_seed(SEED)


# ====== Load and prepare data =======
data = load_track_data(
    batch_size=BATCH_SIZE,
    seed=SEED,
    device=device,
    target_cols=TARGET_COLS,
)

train_loader = data.train_loader
val_loader = data.val_loader
X_train = data.x_train
X_val = data.x_val
Y_train = data.y_train
Y_val = data.y_val
x_mean = data.x_mean
x_std = data.x_std
y_mean = data.y_mean
y_std = data.y_std
y_mean_t = data.y_mean_t
y_std_t = data.y_std_t
FEATURE_COLS = data.feature_cols
PHI_INDEX = data.phi_index

if CHECK_SHAPE:
    print_data_shapes(data)


# ===== Training ======
input_dim = X_train.shape[1]
output_dim = len(TARGET_COLS)

model = HeteroTrackNet(
    input_dim=input_dim,
    hidden_layers=HIDDEN_LAYERS,
    output_dim=output_dim,
    use_batchnorm=True,
    dropout=0.10,
    activation=nn.ReLU,
)
model.to(device)
print(model)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)


def build_checkpoint_metadata(report_text=None):
    metadata = {
        "target_cols": TARGET_COLS,
        "feature_cols": FEATURE_COLS,
        "model_type": "HeteroTrackNet",
        "input_dim": input_dim,
        "output_dim": output_dim,
        "y_mean": y_mean,
        "y_std": y_std,
        "x_mean": x_mean,
        "x_std": x_std,
        "hidden_layers": HIDDEN_LAYERS,
        "use_batchnorm": True,
        "dropout": 0.10,
        "activation": "ReLU",
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "val_fraction": 0.2,
        "criterion": CRITERION.__name__,
        "overlap_target_index": OVERLAP_TARGET_INDEX,
        "overlap_target_name": TARGET_COLS[OVERLAP_TARGET_INDEX],
    }

    if report_text is not None:
        metadata["report_text"] = report_text

    return metadata


def save_best_model_and_plots(metric_tag, metric_value, epoch, report_text):
    model_path = os.path.join(SAVE_DIR, f"{metric_tag}.pt")
    report_path = os.path.join(PLOT_DIR, f"{metric_tag}_training_report.txt")

    metadata = build_checkpoint_metadata(report_text=report_text)
    metadata.update(
        {
            "metric_tag": metric_tag,
            "metric_value": float(metric_value),
        }
    )

    save_model_checkpoint(
        save_path=model_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch + 1,
        metadata=metadata,
    )

    plot_paths = make_val_diagnostic_plots(
        model=model,
        val_loader=val_loader,
        device=device,
        y_mean_t=y_mean_t,
        y_std_t=y_std_t,
        target_cols=TARGET_COLS,
        phi_index=PHI_INDEX,
        output_dir=PLOT_DIR,
        prefix=metric_tag,
        bins=100,
        density=True,
        show=False,
    )

    os.makedirs(PLOT_DIR, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"{metric_tag.upper()} TRAINING REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"target_name: {TARGET_COLS[OVERLAP_TARGET_INDEX]}\n")
        f.write(f"epoch: {epoch + 1}\n")
        f.write(f"metric_value: {metric_value:.6f}\n")
        f.write("\n")
        f.write(report_text)
        f.write("\n")

    print(f"   Saved model: {model_path}")
    print(f"   Saved report: {report_path}")
    print("   Saved plots:")
    for plot_name, plot_path in plot_paths.items():
        print(f"      {plot_name}: {plot_path}")


if TEST_TRAIN:
    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)

    mu, logvar = model(xb)

    print("mu shape:", mu.shape)
    print("logvar shape:", logvar.shape)
    print("target shape:", yb.shape)

    loss = CRITERION(yb, mu, logvar, phi_index=PHI_INDEX)
    print("initial loss:", loss.item())


training_history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "val_mean_mae": [],
    "val_mean_rmse": [],
    "learning_rate": [],
}
overlap_history = {
    "epoch": [],
    "overlap": [],
    "mae": [],
}

best_vals = {
    "best_val_loss": float("inf"),
    "best_mae_pca_dxy": float("inf"),
    "best_rmse_pca_dxy": float("inf"),
    "best_overlap_pca_dxy": -float("inf"),
}

if TRAIN:
    if SAVE_BEST_MODELS:
        os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()

            mu, logvar = model(xb)
            loss = CRITERION(yb, mu, logvar, phi_index=PHI_INDEX)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0

        total_val_mae = torch.zeros(output_dim, device=device)
        total_val_sq = torch.zeros(output_dim, device=device)
        total_count = 0
        overlap_pred_parts = []
        overlap_true_parts = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)

                mu, logvar = model(xb)
                loss = CRITERION(yb, mu, logvar, phi_index=PHI_INDEX)
                val_loss += loss.item() * xb.size(0)

                mu_phys = denormalize_targets(mu, y_mean_t, y_std_t)
                yb_phys = denormalize_targets(yb, y_mean_t, y_std_t)

                diff = mu_phys - yb_phys

                total_val_mae += diff.abs().sum(dim=0)
                total_val_sq += (diff ** 2).sum(dim=0)
                total_count += xb.size(0)

                overlap_pred_parts.append(mu_phys.detach().cpu())
                overlap_true_parts.append(yb_phys.detach().cpu())

        val_loss /= len(val_loader.dataset)

        per_target_mae = (total_val_mae / total_count).detach().cpu().numpy()
        per_target_rmse = np.sqrt((total_val_sq / total_count).detach().cpu().numpy())

        overall_val_mae = per_target_mae.mean()
        overall_val_rmse = per_target_rmse.mean()
        current_lr = optimizer.param_groups[0]["lr"]

        overlap_pred = torch.cat(overlap_pred_parts, dim=0).numpy()
        overlap_true = torch.cat(overlap_true_parts, dim=0).numpy()
        target_overlap = compute_target_histogram_overlap(
            y_true=overlap_true,
            y_pred=overlap_pred,
            target_index=OVERLAP_TARGET_INDEX,
            target_cols=TARGET_COLS,
            bins=100,
        )
        target_mae = float(per_target_mae[OVERLAP_TARGET_INDEX])
        target_rmse = float(per_target_rmse[OVERLAP_TARGET_INDEX])

        training_history["epoch"].append(epoch + 1)
        training_history["train_loss"].append(train_loss)
        training_history["val_loss"].append(val_loss)
        training_history["val_mean_mae"].append(overall_val_mae)
        training_history["val_mean_rmse"].append(overall_val_rmse)
        training_history["learning_rate"].append(current_lr)

        overlap_history["epoch"].append(epoch + 1)
        overlap_history["overlap"].append(target_overlap)
        overlap_history["mae"].append(target_mae)

        report = format_epoch_report(
            epoch,
            EPOCHS,
            train_loss,
            val_loss,
            overall_val_mae,
            overall_val_rmse,
            per_target_mae,
            per_target_rmse,
            TARGET_COLS,
        )
        overlap_report = f"   Overlap pca_dxy: {target_overlap:.6f} | MAE: {target_mae:.6f}"
        full_report = f"{report}\n{overlap_report}"

        print(report)
        print(overlap_report)

        if SAVE_BEST_MODELS:
            if val_loss < best_vals["best_val_loss"]:
                best_vals["best_val_loss"] = val_loss
                print(f"   New best solo val loss: {val_loss:.6f} at epoch {epoch + 1}")
                save_best_model_and_plots("best_val_loss_pca_dxy", val_loss, epoch, full_report)

            if target_mae < best_vals["best_mae_pca_dxy"]:
                best_vals["best_mae_pca_dxy"] = target_mae
                print(f"   New best solo pca_dxy MAE: {target_mae:.6f} at epoch {epoch + 1}")
                save_best_model_and_plots("best_mae_pca_dxy", target_mae, epoch, full_report)

            if target_rmse < best_vals["best_rmse_pca_dxy"]:
                best_vals["best_rmse_pca_dxy"] = target_rmse
                print(f"   New best solo pca_dxy RMSE: {target_rmse:.6f} at epoch {epoch + 1}")
                save_best_model_and_plots("best_rmse_pca_dxy", target_rmse, epoch, full_report)

            if target_overlap > best_vals["best_overlap_pca_dxy"]:
                best_vals["best_overlap_pca_dxy"] = target_overlap
                print(f"   New best solo pca_dxy overlap: {target_overlap:.6f} at epoch {epoch + 1}")
                save_best_model_and_plots(
                    "best_overlap_pca_dxy",
                    target_overlap,
                    epoch,
                    full_report,
                )

        scheduler.step()

if PRINT_FINAL_VAL_SAMPLES:
    print_final_validation_samples(
        model,
        val_loader,
        device,
        y_mean_t,
        y_std_t,
        TARGET_COLS,
        PHI_INDEX,
        num_examples=5,
    )

if PLOT_TRAINING_HISTORY:
    history_plot_paths = make_training_history_plots(
        history=training_history,
        output_dir=PLOT_DIR,
        prefix=PLOT_PREFIX,
        show=False,
    )
    if history_plot_paths:
        print("========== Saved training history plots: ==========")
        for plot_name, plot_path in history_plot_paths.items():
            print(f"  {plot_name}: {plot_path}")

if PLOT_OVERLAP_HISTORY:
    if overlap_history["epoch"]:
        overlap_history_path = os.path.join(PLOT_DIR, f"{PLOT_PREFIX}_overlap_over_time.png")
        plot_overlap_history(
            history=overlap_history,
            target_name=TARGET_COLS[OVERLAP_TARGET_INDEX],
            save_path=overlap_history_path,
            show=False,
        )
        print("========== Saved overlap history plot: ==========")
        print(f"  overlap_history: {overlap_history_path}")
    else:
        print("Skipping overlap history plot because no epochs were recorded.")

if PLOT_VAL_DISTRIBUTIONS:
    plot_paths = make_val_diagnostic_plots(
        model=model,
        val_loader=val_loader,
        device=device,
        y_mean_t=y_mean_t,
        y_std_t=y_std_t,
        target_cols=TARGET_COLS,
        phi_index=PHI_INDEX,
        output_dir=PLOT_DIR,
        prefix=PLOT_PREFIX,
        bins=100,
        density=True,
        show=True,
    )
    print("========== Saved validation diagnostic plots: ==========")
    for plot_name, plot_path in plot_paths.items():
        print(f"  {plot_name}: {plot_path}")
