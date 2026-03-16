#!/usr/bin/env python3
"""
Simple dense network for track parameter regression.

Predicts 5 PCA track parameters from 15 sim-track features.
All the knobs you'd want to tweak are at the top.

Usage:
    python claudeNN.py [path/to/track_data_filtered.npz]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these
# ═════════════════════════════════════════════════════════════════════════════

DATA_PATH = "./track_cache/track_data_filtered.npz"

# ── Network architecture ────────────────────────────────────────────────────
# Each entry is the width of one hidden layer.  Add/remove entries to
# change depth.  Examples:
#   [128, 128]              — 2 hidden layers, 128 wide
#   [256, 256, 128, 64]     — 4 hidden layers, tapering
#   [512, 512, 512]         — 3 hidden layers, 512 wide
HIDDEN_LAYERS = [256, 256, 128]

ACTIVATION = "relu"          # "relu", "gelu", "silu", "tanh", "leaky_relu"
DROPOUT = 0.1                # dropout between hidden layers, try 0.1–0.3
BATCH_NORM = True            # batch norm between layers

# ── Training ────────────────────────────────────────────────────────────────
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4          # L2 regularization strength
EPOCHS = 50
OPTIMIZER = "adam"            # "adam", "adamw", "sgd"
SCHEDULER = "cosine"         # "cosine", "step", "none"
STEP_LR_EVERY = 15           # only used if SCHEDULER = "step"
STEP_LR_GAMMA = 0.5          # only used if SCHEDULER = "step"

# ── Data ────────────────────────────────────────────────────────────────────
VAL_FRACTION = 0.2           # fraction held out for validation
SHUFFLE_SEED = 42
NORMALIZE_INPUTS = True      # z-score normalize features
NORMALIZE_TARGETS = True     # z-score normalize targets (predictions get
                             # un-normalized for reporting)

# ── Output ──────────────────────────────────────────────────────────────────
SAVE_MODEL = True
MODEL_PATH = "./track_cache/model.pt"
PRINT_EVERY = 1              # print metrics every N epochs


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL
# ═════════════════════════════════════════════════════════════════════════════

def get_activation(name: str) -> nn.Module:
    return {
        "relu":       nn.ReLU,
        "gelu":       nn.GELU,
        "silu":       nn.SiLU,
        "tanh":       nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
    }[name]()


class TrackNet(nn.Module):
    """Simple feed-forward regression network, built from config."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()

        layers: list[nn.Module] = []
        prev = n_in

        for width in HIDDEN_LAYERS:
            layers.append(nn.Linear(prev, width))
            if BATCH_NORM:
                layers.append(nn.BatchNorm1d(width))
            layers.append(get_activation(ACTIVATION))
            if DROPOUT > 0:
                layers.append(nn.Dropout(DROPOUT))
            prev = width

        # Final linear head — no activation, raw regression output
        layers.append(nn.Linear(prev, n_out))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_data(path: str, device: torch.device):
    d = np.load(path, allow_pickle=False)
    X = d["X"].astype(np.float32)
    Y = d["Y"].astype(np.float32)
    feat_cols = list(d["feature_columns"])
    label_cols = list(d["label_columns"])

    # Shuffle and split
    n = X.shape[0]
    rng = np.random.default_rng(SHUFFLE_SEED)
    idx = rng.permutation(n)
    split = int(n * (1 - VAL_FRACTION))

    X_train, X_val = X[idx[:split]], X[idx[split:]]
    Y_train, Y_val = Y[idx[:split]], Y[idx[split:]]

    # Normalize
    x_mean = x_std = y_mean = y_std = None

    if NORMALIZE_INPUTS:
        x_mean = X_train.mean(axis=0)
        x_std = X_train.std(axis=0)
        x_std[x_std < 1e-8] = 1.0  # avoid div-by-zero for constant cols
        X_train = (X_train - x_mean) / x_std
        X_val = (X_val - x_mean) / x_std

    if NORMALIZE_TARGETS:
        y_mean = Y_train.mean(axis=0)
        y_std = Y_train.std(axis=0)
        y_std[y_std < 1e-8] = 1.0
        Y_train = (Y_train - y_mean) / y_std
        Y_val = (Y_val - y_mean) / y_std

    # To tensors
    Xt = torch.from_numpy(X_train).to(device)
    Yt = torch.from_numpy(Y_train).to(device)
    Xv = torch.from_numpy(X_val).to(device)
    Yv = torch.from_numpy(Y_val).to(device)

    train_ds = TensorDataset(Xt, Yt)
    val_ds = TensorDataset(Xv, Yv)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

    norm = {
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": y_mean, "y_std": y_std,
    }

    return train_dl, val_dl, feat_cols, label_cols, norm


# ═════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    opts = {
        "adam":  lambda: torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                          weight_decay=WEIGHT_DECAY),
        "adamw": lambda: torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                           weight_decay=WEIGHT_DECAY),
        "sgd":   lambda: torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
                                         momentum=0.9, weight_decay=WEIGHT_DECAY),
    }
    return opts[OPTIMIZER]()


def make_scheduler(optimizer):
    if SCHEDULER == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    if SCHEDULER == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=STEP_LR_EVERY,
                                                gamma=STEP_LR_GAMMA)
    return None


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.shape[0]
        n += xb.shape[0]
    return total_loss / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        total_loss += loss.item() * xb.shape[0]
        n += xb.shape[0]
    return total_loss / n


@torch.no_grad()
def per_target_metrics(model, loader, label_cols, norm, device):
    """Compute MAE and RMSE per target in original (un-normalized) units."""
    model.eval()
    all_pred = []
    all_true = []
    for xb, yb in loader:
        all_pred.append(model(xb).cpu().numpy())
        all_true.append(yb.cpu().numpy())

    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)

    # Un-normalize if needed
    if norm["y_mean"] is not None:
        pred = pred * norm["y_std"] + norm["y_mean"]
        true = true * norm["y_std"] + norm["y_mean"]

    print(f"\n  {'target':>22s}  {'MAE':>12s}  {'RMSE':>12s}")
    print(f"  {'─'*22}  {'─'*12}  {'─'*12}")
    for i, name in enumerate(label_cols):
        diff = pred[:, i] - true[:, i]
        mae = np.abs(diff).mean()
        rmse = np.sqrt((diff ** 2).mean())
        print(f"  {name:>22s}  {mae:12.6f}  {rmse:12.6f}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Device: {device}")
    print(f"Data:   {data_path}\n")

    # Load
    train_dl, val_dl, feat_cols, label_cols, norm = load_data(data_path, device)
    n_in = len(feat_cols)
    n_out = len(label_cols)
    n_train = len(train_dl.dataset)
    n_val = len(val_dl.dataset)

    print(f"Samples: {n_train:,} train, {n_val:,} val")
    print(f"Input:   {n_in} features → {feat_cols}")
    print(f"Output:  {n_out} targets  → {label_cols}")

    # Model
    model = TrackNet(n_in, n_out).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel:  {n_params:,} parameters")
    print(model)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer)

    print(f"\nOptimizer:  {OPTIMIZER}  lr={LEARNING_RATE}  wd={WEIGHT_DECAY}")
    print(f"Scheduler:  {SCHEDULER}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs:     {EPOCHS}\n")

    # Train
    best_val = float("inf")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        t_epoch = time.time()

        train_loss = train_one_epoch(model, train_dl, criterion, optimizer, device)
        val_loss = validate(model, val_dl, criterion, device)

        if scheduler is not None:
            scheduler.step()

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            if SAVE_MODEL:
                torch.save({
                    "model_state": model.state_dict(),
                    "config": {
                        "n_in": n_in, "n_out": n_out,
                        "hidden_layers": HIDDEN_LAYERS,
                        "activation": ACTIVATION,
                        "dropout": DROPOUT,
                        "batch_norm": BATCH_NORM,
                    },
                    "norm": norm,
                    "feature_columns": feat_cols,
                    "label_columns": label_cols,
                    "epoch": epoch,
                    "val_loss": val_loss,
                }, MODEL_PATH)

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            dt = time.time() - t_epoch
            star = " *" if improved else ""
            print(
                f"  epoch {epoch:3d}/{EPOCHS} | "
                f"train {train_loss:.6f} | "
                f"val {val_loss:.6f}{star} | "
                f"lr {lr:.2e} | "
                f"{dt:.1f}s"
            )

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best val loss: {best_val:.6f}")

    # Per-target breakdown on validation set
    if SAVE_MODEL:
        ckpt = torch.load(MODEL_PATH, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    per_target_metrics(model, val_dl, label_cols, norm, device)

    if SAVE_MODEL:
        print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()