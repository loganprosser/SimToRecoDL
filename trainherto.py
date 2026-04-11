import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from model import HeteroTrackNet
from loss import paper_hetero_loss, hetero_gaussian_nll_with_phi, hetero_gaussian_nll_with_phi_relative
from helpers_data import load_track_data, print_data_shapes, set_seed
from helpers import (
        wrapped_angle_diff,
        denormalize_targets,
        format_epoch_report,
        save_golden_model,
        save_model_checkpoint,
        write_final_golden_summary,
    )
from helpers_vis import (
    compute_target_histogram_overlap,
    make_training_history_plots,
    make_val_diagnostic_plots,
    plot_overlap_history,
    print_final_validation_samples,
)
#TODO fix bashrc script on classe machine keeps getting hung on something not sure whta
# TODO use a different learning funciton or play with rate as we go on
# TODO get a shit ton of data and see if we can acomplish double descent??

# ====== Running Constants =======
EPOCHS = 2000
TARGET_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
MEAN_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
# know that 1 is prob too high of a weighting since this loss is HUGE at small values
LAMBDA_REL = torch.tensor([.25, .25, .25, .25, .25])



# set TARGET_WEIGHTS = None if you want default weighting i.e. [1,1,1,1,1]

BATCH_SIZE = 256
#HIDDEN_LAYERS = [2048, 2048, 1024, 512] # new layers try to get double descent!!!! #[256, 256, 64]
HIDDEN_LAYERS = [2048, 2048, 1024, 512] 
CRITERION = hetero_gaussian_nll_with_phi_relative # paper_hetero_loss, hetero_gaussian_nll_with_phi, hetero_gaussian_nll_with_phi_relative

# ====== Running Flags =======
CHECK_SHAPE = False
TEST_TRAIN = False
TRAIN = True
PRINT_FINAL_VAL_SAMPLES = True
TRACK_GOLDEN = True
PLOT_VAL_DISTRIBUTIONS = True
PLOT_TRAINING_HISTORY = True
TRACK_BEST_OVERLAP = True
PLOT_OVERLAP_HISTORY = True

# ====== Overlap tracking settings ======
OVERLAP_TARGET_INDEX = 3
OVERLAP_MODEL_DIR = "maxoverlapd0"

# ====== Golden model settings ======
GOLDEN_MODEL_DIR = "RELgoldenmodels"
GOLDEN_SUMMARY_FILE = "RELgoldeniteration.txt"
PLOT_DIR = "RELplots"
PLOT_PREFIX = "relative_loss_hetero"

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
data = load_track_data(batch_size=BATCH_SIZE, seed=SEED, device=device)

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
TARGET_COLS = data.target_cols
PHI_INDEX = data.phi_index

if not 0 <= OVERLAP_TARGET_INDEX < len(TARGET_COLS):
    raise ValueError(f"OVERLAP_TARGET_INDEX must be in [0, {len(TARGET_COLS) - 1}]")

# ====== CHECK SHAPES ======
if CHECK_SHAPE:
    print_data_shapes(data)


# ===== Training ======
input_dim = X_train.shape[1]

# hidden_layers=[1024, 1024, 512, 256]
model = HeteroTrackNet(
    input_dim=input_dim,
    hidden_layers=HIDDEN_LAYERS,
    output_dim=5,
    use_batchnorm=True,
    dropout=0.10,
    activation=nn.ReLU
)
model.to(device)

print(model)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
# scheduler decreases the learnring rate as we go on with cosine decay
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

# TODO Fix and put these in helpers tommorow: 
# === temp helper functions =====


# ====== trial forward pass ======
if TEST_TRAIN:
    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)

    mu, logvar = model(xb)

    print("mu shape:", mu.shape)
    print("logvar shape:", logvar.shape)
    print("target shape:", yb.shape)

    loss = CRITERION(
        yb,
        mu,
        logvar,
        phi_index=PHI_INDEX,
        target_weights=TARGET_WEIGHTS,
        mean_weights=MEAN_WEIGHTS
        )
    
    print("initial loss:", loss.item())


# ===== Training loop =====
#EPOCHS = EPOCHS
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


def build_checkpoint_metadata(report_text=None):
    metadata = {
        "target_cols": TARGET_COLS,
        "feature_cols": FEATURE_COLS,
        "model_type": "HeteroTrackNet",
        "input_dim": input_dim,
        "output_dim": len(TARGET_COLS),
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
        "target_weights": TARGET_WEIGHTS.tolist() if TARGET_WEIGHTS is not None else None,
        "mean_weights": MEAN_WEIGHTS.tolist() if MEAN_WEIGHTS is not None else None,
        "overlap_target_index": OVERLAP_TARGET_INDEX,
        "overlap_target_name": TARGET_COLS[OVERLAP_TARGET_INDEX],
    }

    if report_text is not None:
        metadata["report_text"] = report_text

    return metadata

if TRAIN:

    if TRACK_GOLDEN:
        os.makedirs(GOLDEN_MODEL_DIR, exist_ok=True)

        best_vals = {
            "best_val_loss": float("inf"),
            "best_mean_mae": float("inf"),
            "best_mean_rmse": float("inf"),
        }

        for name in TARGET_COLS:
            best_vals[f"best_mae_{name}"] = float("inf")
            best_vals[f"best_rmse_{name}"] = float("inf")

        best_reports = {}

    if TRACK_BEST_OVERLAP:
        os.makedirs(OVERLAP_MODEL_DIR, exist_ok=True)
        best_overlap = {
            "overlap": -float("inf"),
            "mae": float("inf"),
            "epoch": 0,
        }

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()

            mu, logvar = model(xb)
            loss = CRITERION(
                yb,
                mu,
                logvar,
                phi_index=PHI_INDEX,
                target_weights=TARGET_WEIGHTS,
                mean_weights=MEAN_WEIGHTS
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0

        total_val_mae = torch.zeros(len(TARGET_COLS), device=device)
        total_val_sq = torch.zeros(len(TARGET_COLS), device=device)
        total_count = 0
        overlap_pred_parts = []
        overlap_true_parts = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)

                mu, logvar = model(xb)

                loss = CRITERION(
                    yb,
                    mu,
                    logvar,
                    phi_index=PHI_INDEX,
                    target_weights=TARGET_WEIGHTS,
                    mean_weights=MEAN_WEIGHTS
                )
                val_loss += loss.item() * xb.size(0)

                mu_phys = denormalize_targets(mu, y_mean_t, y_std_t)
                yb_phys = denormalize_targets(yb, y_mean_t, y_std_t)

                diff = mu_phys - yb_phys

                diff[:, PHI_INDEX] = wrapped_angle_diff(
                    mu_phys[:, PHI_INDEX],
                    yb_phys[:, PHI_INDEX]
                )

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
        if OVERLAP_TARGET_INDEX == PHI_INDEX:
            overlap_pred[:, PHI_INDEX] = np.arctan2(
                np.sin(overlap_pred[:, PHI_INDEX]),
                np.cos(overlap_pred[:, PHI_INDEX])
            )
            overlap_true[:, PHI_INDEX] = np.arctan2(
                np.sin(overlap_true[:, PHI_INDEX]),
                np.cos(overlap_true[:, PHI_INDEX])
            )
        target_overlap = compute_target_histogram_overlap(
            y_true=overlap_true,
            y_pred=overlap_pred,
            target_index=OVERLAP_TARGET_INDEX,
            target_cols=TARGET_COLS,
            bins=100,
        )
        target_mae = float(per_target_mae[OVERLAP_TARGET_INDEX])

        training_history["epoch"].append(epoch + 1)
        training_history["train_loss"].append(train_loss)
        training_history["val_loss"].append(val_loss)
        training_history["val_mean_mae"].append(overall_val_mae)
        training_history["val_mean_rmse"].append(overall_val_rmse)
        training_history["learning_rate"].append(current_lr)

        overlap_history["epoch"].append(epoch + 1)
        overlap_history["overlap"].append(target_overlap)
        overlap_history["mae"].append(target_mae)

        # ===== build report string =====
        report = format_epoch_report(
            epoch,
            EPOCHS,
            train_loss,
            val_loss,
            overall_val_mae,
            overall_val_rmse,
            per_target_mae,
            per_target_rmse,
            TARGET_COLS
        )

        overlap_report = (
            f"   Overlap {TARGET_COLS[OVERLAP_TARGET_INDEX]}: "
            f"{target_overlap:.6f} | MAE: {target_mae:.6f}"
        )

        print(report)
        print(overlap_report)

        # ===== GOLDEN TRACKING =====
        if TRACK_GOLDEN:

            def save(name, value):
                save_golden_model(
                    model,
                    optimizer,
                    scheduler,
                    name,
                    value,
                    epoch,
                    report,
                    GOLDEN_MODEL_DIR,
                    build_checkpoint_metadata(report_text=report)
                )
                best_reports[name] = report

            # overall
            if val_loss < best_vals["best_val_loss"]:
                best_vals["best_val_loss"] = val_loss
                save("best_val_loss", val_loss)

            if overall_val_mae < best_vals["best_mean_mae"]:
                best_vals["best_mean_mae"] = overall_val_mae
                save("best_mean_mae", overall_val_mae)

            if overall_val_rmse < best_vals["best_mean_rmse"]:
                best_vals["best_mean_rmse"] = overall_val_rmse
                save("best_mean_rmse", overall_val_rmse)

            # per target
            for i, name in enumerate(TARGET_COLS):
                if per_target_mae[i] < best_vals[f"best_mae_{name}"]:
                    best_vals[f"best_mae_{name}"] = per_target_mae[i]
                    save(f"best_mae_{name}", per_target_mae[i])

                if per_target_rmse[i] < best_vals[f"best_rmse_{name}"]:
                    best_vals[f"best_rmse_{name}"] = per_target_rmse[i]
                    save(f"best_rmse_{name}", per_target_rmse[i])

        if TRACK_BEST_OVERLAP:
            overlap_improved = (
                target_overlap > best_overlap["overlap"]
                or (
                    target_overlap == best_overlap["overlap"]
                    and target_mae < best_overlap["mae"]
                )
            )

            if overlap_improved:
                best_overlap["overlap"] = target_overlap
                best_overlap["mae"] = target_mae
                best_overlap["epoch"] = epoch + 1

                overlap_target_name = TARGET_COLS[OVERLAP_TARGET_INDEX]
                overlap_tag = f"best_overlap_{overlap_target_name}"
                overlap_model_path = os.path.join(OVERLAP_MODEL_DIR, f"{overlap_tag}.pt")
                overlap_report_path = os.path.join(PLOT_DIR, f"{overlap_tag}_training_report.txt")

                metadata = build_checkpoint_metadata(report_text=f"{report}\n{overlap_report}")
                metadata.update(
                    {
                        "metric_tag": overlap_tag,
                        "metric_value": float(target_overlap),
                        "overlap": float(target_overlap),
                        "overlap_mae": float(target_mae),
                    }
                )

                save_model_checkpoint(
                    save_path=overlap_model_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    metadata=metadata,
                )

                overlap_plot_paths = make_val_diagnostic_plots(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    y_mean_t=y_mean_t,
                    y_std_t=y_std_t,
                    target_cols=TARGET_COLS,
                    phi_index=PHI_INDEX,
                    output_dir=PLOT_DIR,
                    prefix=overlap_tag,
                    bins=100,
                    density=True,
                    show=False,
                )

                os.makedirs(PLOT_DIR, exist_ok=True)
                with open(overlap_report_path, "w") as f:
                    f.write("BEST OVERLAP TRAINING REPORT\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"target_index: {OVERLAP_TARGET_INDEX}\n")
                    f.write(f"target_name: {overlap_target_name}\n")
                    f.write(f"epoch: {epoch + 1}\n")
                    f.write(f"overlap: {target_overlap:.6f}\n")
                    f.write(f"mae: {target_mae:.6f}\n")
                    f.write("\n")
                    f.write(report)
                    f.write("\n")
                    f.write(overlap_report)
                    f.write("\n")

                print(
                    f"   New best overlap model for {overlap_target_name}: "
                    f"{target_overlap:.6f} at epoch {epoch + 1}"
                )
                print(f"   Saved overlap model: {overlap_model_path}")
                print(f"   Saved overlap report: {overlap_report_path}")
                print("   Saved best overlap plots:")
                for plot_name, plot_path in overlap_plot_paths.items():
                    print(f"      {plot_name}: {plot_path}")

        scheduler.step()

    # ===== write FINAL summary ONLY ONCE =====
    if TRACK_GOLDEN:
        write_final_golden_summary(GOLDEN_SUMMARY_FILE, best_reports, best_vals)

if PRINT_FINAL_VAL_SAMPLES:
    print_final_validation_samples(
        model, val_loader, device,
        y_mean_t, y_std_t,
        TARGET_COLS, PHI_INDEX,
        num_examples=5
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
        overlap_target_name = TARGET_COLS[OVERLAP_TARGET_INDEX]
        overlap_history_path = os.path.join(
            PLOT_DIR,
            f"{PLOT_PREFIX}_overlap_{overlap_target_name}_over_time.png"
        )
        plot_overlap_history(
            history=overlap_history,
            target_name=overlap_target_name,
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
        show=True
    )
    print("========== Saved validation diagnostic plots: ==========")
    for plot_name, plot_path in plot_paths.items():
        print(f"  {plot_name}: {plot_path}")
