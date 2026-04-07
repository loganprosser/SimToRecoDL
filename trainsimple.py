import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib
import pandas as pd
import random

from model import SimpleTrackNet, TestTrackNet, HeteroTrackNet
from helpers import print_final_validation_samples

# ===== Constants ======
EPOCHS = 100


# ====== Running flags ======
PRINT_FINAL_VAL_SAMPLES = False # not working need sigma for the funciton



# ===== Picking Device ========
device = torch.device(
    "mps" if torch.backends.mps.is_available() 
    else "cuda" if torch.cuda.is_available() 
    else "cpu"
    
)
print(F"Device set to {device}")

# ==== Setting Seed =====
SEED = 42

# Python + NumPy
random.seed(SEED)
np.random.seed(SEED)

# PyTorch (CPU always works)
torch.manual_seed(SEED)

# CUDA (ONLY if available)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Determinism settings (only affects CUDA backend)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ====== import data from the csv =======
path = "/nfs/cms/tracktrigger/logan/root/simvrico/SimToRecoDL/outputCSVs/filtered_particles.csv"
df = pd.read_csv(path)

# ===== Input and Output Data =====

TARGET_COLS = [
    "pca_c",     # q / pca_pt (curvature proxy)
    "pca_eta",
    "pca_phi",
    "pca_dxy",
    "pca_dz"
]

PHI_INDEX = TARGET_COLS.index("pca_phi")

FEATURE_COLS = []

for j in range(1, 7):  # since N_LAYERS = 6
    FEATURE_COLS += [
        f"hit_{j}_x",
        f"hit_{j}_y",
        f"hit_{j}_z",
        f"hit_{j}_r",
        f"hit_{j}_mask"
    ]

def wrapped_angle_diff(pred, target):
    return torch.atan2(torch.sin(pred - target), torch.cos(pred - target))

# ====== PULL NUMPY ARRAYS ======
X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
Y = df[TARGET_COLS].to_numpy(dtype=np.float32)

# ====== REPLACE SENTINEL VALUES WITH 0 ======
# only the coordinate/r columns should have -999, but this safely replaces any that remain
# removes the mask of the data so the network doesnt get thrown off can normailze too but zero might work
X[X == -999.0] = 0.0

# ====== TRAIN / VAL SPLIT ======
n = len(X)
val_fraction = 0.2
n_val = int(n * val_fraction)

# shuffle indices
rng = np.random.default_rng(seed=SEED)
indices = rng.permutation(n)

val_idx = indices[:n_val]
train_idx = indices[n_val:]

X_train = X[train_idx]
X_val   = X[val_idx]
Y_train = Y[train_idx]
Y_val   = Y[val_idx]

# ==== Normalize inputs ====
x_mean = X_train.mean(axis=0)
x_std = X_train.std(axis=0)
x_std[x_std < 1e-8] = 1.0

X_train = (X_train - x_mean) / x_std
X_val   = (X_val   - x_mean) / x_std

# ==== Normalize targets ====
y_mean = Y_train.mean(axis=0)
y_std = Y_train.std(axis=0)
y_std[y_std < 1e-8] = 1.0

Y_train = (Y_train - y_mean) / y_std
Y_val   = (Y_val   - y_mean) / y_std

# tensors for denorm
y_mean_t = torch.tensor(y_mean, dtype=torch.float32, device=device)
y_std_t  = torch.tensor(y_std, dtype=torch.float32, device=device)

def denormalize_targets(y_norm):
    return y_norm * y_std_t + y_mean_t


# ====== CONVERT TO TENSORS ======
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val   = torch.tensor(X_val, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
Y_val   = torch.tensor(Y_val, dtype=torch.float32)

train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)

# ====== DataLoaders ======
# data loader puts data into the network in chunks
# gives a batch size at a time (used for gradient update)
# 256 and 80 epochs works pretty well
# TODO understand why we have such a large dependancy on this
BATCH_SIZE = 256 #not sure why this effects model so much idk??

# random seed for dataloader
g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, generator=g)


# ====== CHECK SHAPES ======
CHECK_SHAPE = False
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
# === Init Model =====

#model = TestTrackNet(input_dim=input_dim, hidden_dim=64, output_dim=5)
model = SimpleTrackNet(
    input_dim=input_dim,
    hidden_layers=[512, 512, 128],   #128, 128, 64]. [256, 256, 64]
    use_batchnorm=False,
    dropout=0.00,
    activation=nn.ReLU
)


model.to(device)

print(model)


# ====== Set up the Loss and optimizer =======

criterion = nn.MSELoss() # use for simple models
#optimizer = optim.Adam(model.parameters(), lr=1e-3) # use for simple models

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)


# ====== trial forward pass ======
TEST_TRAIN = False
if TEST_TRAIN:
    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)
    preds = model(xb)

    print("pred shape:", preds.shape)
    print("target shape:", yb.shape)

    loss = criterion(preds, yb)
    print("initial loss:", loss.item())

# ===== Training loop =====
EPOCHS = EPOCHS

for epoch in range(EPOCHS):
    # ===== TRAIN ======
    model.train()
    train_loss = 0.0
    
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * xb.size(0)
        
    train_loss /= len(train_loader.dataset)
    
    # ==== VALIDATION ====
    
    model.eval()
    val_loss = 0.0

    total_val_mae = torch.zeros(len(TARGET_COLS), device=device)
    total_val_sq  = torch.zeros(len(TARGET_COLS), device=device)
    total_count = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)

            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item() * xb.size(0)

            # ==== DENORMALIZE (if using normalization) ====
            preds_phys = denormalize_targets(preds)
            yb_phys    = denormalize_targets(yb)

            diff = preds_phys - yb_phys

            # wrap phi correctly
            diff[:, PHI_INDEX] = wrapped_angle_diff(
                preds_phys[:, PHI_INDEX],
                yb_phys[:, PHI_INDEX]
            )

            total_val_mae += diff.abs().sum(dim=0)
            total_val_sq  += (diff ** 2).sum(dim=0)
            total_count   += xb.size(0)

    val_loss /= len(val_loader.dataset)

    per_target_mae = (total_val_mae / total_count).detach().cpu().numpy()
    per_target_rmse = torch.sqrt(total_val_sq / total_count).detach().cpu().numpy()

    overall_val_mae = per_target_mae.mean()
    overall_val_rmse = per_target_rmse.mean()
    
    print(f"EPOCH {epoch + 1:2d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Mean MAE: {overall_val_mae:.6f} | Val Mean RMSE: {overall_val_rmse:.6f}")

    print("   Per-target MAE:")
    for name, val in zip(TARGET_COLS, per_target_mae):
        print(f"      {name}: {val:.6f}")

    print("   Per-target RMSE:")
    for name, val in zip(TARGET_COLS, per_target_rmse):
        print(f"      {name}: {val:.6f}")
    
if PRINT_FINAL_VAL_SAMPLES:
    print_final_validation_samples(
        model, val_loader, device,
        denormalize_targets, y_std_t,
        TARGET_COLS, PHI_INDEX,
        wrapped_angle_diff,
        num_examples=4
    )