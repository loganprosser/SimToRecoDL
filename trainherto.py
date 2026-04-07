import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import random

from model import HeteroTrackNet
from loss import paper_hetero_loss, hetero_gaussian_nll_with_phi
from helpers import print_final_validation_samples

#TODO fix bashrc script on classe machine keeps getting hung on something not sure whta
# TODO use a different learning funciton or play with rate as we go on
# TODO get a shit ton of data and see if we can acomplish double descent??

# ====== Running Constants =======
EPOCHS = 10000
TARGET_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0], dtype=torch.float32)
MEAN_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 1.0, 3.0])

# set TARGET_WEIGHTS = None if you want default weighting i.e. [1,1,1,1,1]

BATCH_SIZE = 256
HIDDEN_LAYERS = [512, 512, 128]

CRITERION = hetero_gaussian_nll_with_phi

# ====== Running Flags =======
CHECK_SHAPE = False
TEST_TRAIN = False
TRAIN = True
PRINT_FINAL_VAL_SAMPLES = True
TRACK_GOLDEN = True

# ====== Golden model settings ======
GOLDEN_MODEL_DIR = "goldenmodels"
GOLDEN_SUMMARY_FILE = "goldeniteration.txt"

# ===== Picking Device ========
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Device set to {device}")


# ==== Setting Seed =====
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ====== import data from the csv =======
path = "/nfs/cms/tracktrigger/logan/root/simvrico/SimToRecoDL/outputCSVs/filtered_particles.csv"
df = pd.read_csv(path)


# ===== Input and Output Data =====
TARGET_COLS = [
    "pca_c",
    "pca_eta",
    "pca_phi",
    "pca_dxy",
    "pca_dz"
]

FEATURE_COLS = []

for j in range(1, 7):
    FEATURE_COLS += [
        f"hit_{j}_x",
        f"hit_{j}_y",
        f"hit_{j}_z",
        f"hit_{j}_r",
        f"hit_{j}_mask"
    ]


# ====== PULL NUMPY ARRAYS ======
X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
Y = df[TARGET_COLS].to_numpy(dtype=np.float32)


# ====== REPLACE SENTINEL VALUES WITH 0 ======
X[X == -999.0] = 0.0


# ====== TRAIN / VAL SPLIT ======
n = len(X)
val_fraction = 0.2
n_val = int(n * val_fraction)

rng = np.random.default_rng(seed=SEED)
indices = rng.permutation(n)

val_idx = indices[:n_val]
train_idx = indices[n_val:]

X_train = X[train_idx]
X_val   = X[val_idx]
Y_train = Y[train_idx]
Y_val   = Y[val_idx]

# ==== Normalize inputs from train only ====
x_mean = X_train.mean(axis=0)
x_std = X_train.std(axis=0)

# avoid divide-by-zero for constant columns
x_std[x_std < 1e-8] = 1.0

# normalize
X_train = (X_train - x_mean) / x_std
X_val   = (X_val   - x_mean) / x_std

# ====== OUTPUT NORMALIZATION STATS FROM TRAIN ONLY ======
y_mean = Y_train.mean(axis=0)
y_std = Y_train.std(axis=0)

y_std[y_std < 1e-8] = 1.0

Y_train = (Y_train - y_mean) / y_std
Y_val   = (Y_val   - y_mean) / y_std


# ====== CONVERT TO TENSORS ======
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val   = torch.tensor(X_val, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
Y_val   = torch.tensor(Y_val, dtype=torch.float32)

train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)


# ====== DataLoaders ======
BATCH_SIZE = BATCH_SIZE

g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, generator=g)

PHI_INDEX = TARGET_COLS.index("pca_phi")

# store normalization stats as tensors on device for de-normalizing predictions
y_mean_t = torch.tensor(y_mean, dtype=torch.float32, device=device)
y_std_t = torch.tensor(y_std, dtype=torch.float32, device=device)

def wrapped_angle_diff(pred, target):
    return torch.atan2(torch.sin(pred - target), torch.cos(pred - target))

def denormalize_targets(y_norm):
    return y_norm * y_std_t + y_mean_t

# ====== CHECK SHAPES ======
if CHECK_SHAPE:
    print("X_train shape:", X_train.shape)
    print("X_val shape:  ", X_val.shape)
    print("Y_train shape:", Y_train.shape)
    print("Y_val shape:  ", Y_val.shape)

    xb, yb = next(iter(train_loader))
    print("batch X shape:", xb.shape)
    print("batch Y shape:", yb.shape)


# ===== Training ======
input_dim = X_train.shape[1]

# hidden_layers=[1024, 1024, 512, 256]
model = HeteroTrackNet(
    input_dim=input_dim,
    hidden_layers=HIDDEN_LAYERS,
    output_dim=5,
    use_batchnorm=True,
    dropout=0.00,
    activation=nn.ReLU
)
model.to(device)

print(model)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
# scheduler decreases the learnring rate as we go on with cosine decay
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

# TODO Fix and put these in helpers tommorow: 
# === temp helper functions =====
def denormalize_targets(y_norm):
    return y_norm * y_std_t + y_mean_t

def format_epoch_report(
    epoch,
    total_epochs,
    train_loss,
    val_loss,
    overall_val_mae,
    overall_val_mse,
    overall_val_rmse,
    per_target_mae,
    per_target_mse,
    per_target_rmse,
    target_cols
):
    lines = []

    lines.append(
        f"EPOCH {epoch + 1}/{total_epochs} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"Val Mean MAE: {overall_val_mae:.6f} | "
        f"Val Mean MSE: {overall_val_mse:.6f} | "
        f"Val Mean RMSE: {overall_val_rmse:.6f}"
    )

    lines.append("   Per-target MAE:")
    for name, val in zip(target_cols, per_target_mae):
        lines.append(f"      {name}: {val:.6f}")

    lines.append("   Per-target MSE:")
    for name, val in zip(target_cols, per_target_mse):
        lines.append(f"      {name}: {val:.6f}")

    lines.append("   Per-target RMSE:")
    for name, val in zip(target_cols, per_target_rmse):
        lines.append(f"      {name}: {val:.6f}")

    return "\n".join(lines)


def save_golden_model(metric_tag, metric_value, epoch, report_text):
    os.makedirs(GOLDEN_MODEL_DIR, exist_ok=True)

    save_path = os.path.join(GOLDEN_MODEL_DIR, f"{metric_tag}.pt")

    torch.save(
        {
            "metric_tag": metric_tag,
            "metric_value": float(metric_value),
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "target_cols": TARGET_COLS,
            "feature_cols": FEATURE_COLS,
            "y_mean": y_mean,
            "y_std": y_std,
            "x_mean": x_mean,
            "x_std": x_std,
            "hidden_layers": HIDDEN_LAYERS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "report_text": report_text,
        },
        save_path
    )


def write_final_golden_summary(best_reports, best_values):
    with open(GOLDEN_SUMMARY_FILE, "w") as f:
        f.write("FINAL BEST RESULTS BY METRIC\n")
        f.write("=" * 100 + "\n\n")

        for metric_name in best_reports:
            f.write(f"{metric_name}: {best_values[metric_name]:.6f}\n")
            f.write(best_reports[metric_name] + "\n")
            f.write("\n" + "-" * 100 + "\n\n")
            



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
TARGET_WEIGHTS = TARGET_WEIGHTS

if TRAIN:

    if TRACK_GOLDEN:
        import os
        os.makedirs("goldenmodels", exist_ok=True)

        best_vals = {
            "val_loss": float("inf"),
            "mean_mae": float("inf"),
            "mean_mse": float("inf"),
            "mean_rmse": float("inf"),
        }

        for name in TARGET_COLS:
            best_vals[f"mae_{name}"] = float("inf")
            best_vals[f"mse_{name}"] = float("inf")
            best_vals[f"rmse_{name}"] = float("inf")

        best_reports = {}

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

                mu_phys = denormalize_targets(mu)
                yb_phys = denormalize_targets(yb)

                diff = mu_phys - yb_phys

                diff[:, PHI_INDEX] = wrapped_angle_diff(
                    mu_phys[:, PHI_INDEX],
                    yb_phys[:, PHI_INDEX]
                )

                total_val_mae += diff.abs().sum(dim=0)
                total_val_sq += (diff ** 2).sum(dim=0)
                total_count += xb.size(0)

        val_loss /= len(val_loader.dataset)

        per_target_mae = (total_val_mae / total_count).detach().cpu().numpy()
        per_target_mse = (total_val_sq / total_count).detach().cpu().numpy()
        per_target_rmse = np.sqrt(per_target_mse)

        overall_val_mae = per_target_mae.mean()
        overall_val_mse = per_target_mse.mean()
        overall_val_rmse = per_target_rmse.mean()

        # ===== build report string =====
        report = (
            f"EPOCH {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Val Mean MAE: {overall_val_mae:.6f} | Val Mean MSE: {overall_val_mse:.6f} | Val Mean RMSE: {overall_val_rmse:.6f}\n"
            "   Per-target MAE:\n" +
            "\n".join([f"      {n}: {v:.6f}" for n, v in zip(TARGET_COLS, per_target_mae)]) +
            "\n   Per-target RMSE:\n" +
            "\n".join([f"      {n}: {v:.6f}" for n, v in zip(TARGET_COLS, per_target_rmse)])
        )

        print(report)

        # ===== GOLDEN TRACKING =====
        if TRACK_GOLDEN:

            def save(name, value):
                save_golden_model(name, value, epoch, report)
                best_reports[name] = report

            # overall
            if val_loss < best_vals["val_loss"]:
                best_vals["val_loss"] = val_loss
                save("best_val_loss", val_loss)

            if overall_val_mae < best_vals["mean_mae"]:
                best_vals["mean_mae"] = overall_val_mae
                save("best_mean_mae", overall_val_mae)

            if overall_val_mse < best_vals["mean_mse"]:
                best_vals["mean_mse"] = overall_val_mse
                save("best_mean_mse", overall_val_mse)

            if overall_val_rmse < best_vals["mean_rmse"]:
                best_vals["mean_rmse"] = overall_val_rmse
                save("best_mean_rmse", overall_val_rmse)

            # per target
            for i, name in enumerate(TARGET_COLS):
                if per_target_mae[i] < best_vals[f"mae_{name}"]:
                    best_vals[f"mae_{name}"] = per_target_mae[i]
                    save(f"best_mae_{name}", per_target_mae[i])

                if per_target_mse[i] < best_vals[f"mse_{name}"]:
                    best_vals[f"mse_{name}"] = per_target_mse[i]
                    save(f"best_mse_{name}", per_target_mse[i])

                if per_target_rmse[i] < best_vals[f"rmse_{name}"]:
                    best_vals[f"rmse_{name}"] = per_target_rmse[i]
                    save(f"best_rmse_{name}", per_target_rmse[i])

        scheduler.step()

    # ===== write FINAL summary ONLY ONCE =====
    if TRACK_GOLDEN:
        with open("goldeniteration.txt", "w") as f:
            for k in best_reports:
                f.write(f"{k}\n")
                f.write(best_reports[k] + "\n\n")
                
if PRINT_FINAL_VAL_SAMPLES:
    print_final_validation_samples(
        model, val_loader, device,
        denormalize_targets, y_std_t,
        TARGET_COLS, PHI_INDEX,
        wrapped_angle_diff,
        num_examples=4
    )
    
    