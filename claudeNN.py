#!/usr/bin/env python3
"""
Fast dense network for track parameter regression.

Predicts 5 PCA track parameters from 10 sim-track features.

Usage:
    python claudeNN_fast.py [path/to/track_data_filtered.npz]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = "./track_data_etaM1to1.npz"

HIDDEN_LAYERS = [256, 256, 128]
ACTIVATION = "relu"          # "relu", "gelu", "silu", "tanh", "leaky_relu"
DROPOUT = 0.0                # set 0 for max speed first
BATCH_NORM = True

BATCH_SIZE = 16384           # try 8192 / 16384 / 32768 depending on VRAM
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
OPTIMIZER = "adamw"          # adamw is a good default
SCHEDULER = "cosine"         # "cosine", "step", "none"
STEP_LR_EVERY = 15
STEP_LR_GAMMA = 0.5

VAL_FRACTION = 0.2
SHUFFLE_SEED = 42
NORMALIZE_INPUTS = True
NORMALIZE_TARGETS = True

SAVE_MODEL = True
MODEL_PATH = "./track_cache/model.pt"
PRINT_EVERY = 1

# DataLoader speed knobs
NUM_WORKERS = min(8, os.cpu_count() or 1)
PIN_MEMORY = True
PERSISTENT_WORKERS = NUM_WORKERS > 0
PREFETCH_FACTOR = 4 if NUM_WORKERS > 0 else None

# CUDA speed knobs
USE_AMP = True               # mixed precision on CUDA
USE_COMPILE = True           # torch.compile for PyTorch 2.x


# ============================================================================
# MODEL
# ============================================================================

def get_activation(name: str) -> nn.Module:
    return {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
    }[name]()


class TrackNet(nn.Module):
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

        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# DATA
# ============================================================================

def load_data(path: str):
    d = np.load(path, allow_pickle=True)

    X = d["X"].astype(np.float32, copy=False)
    Y = d["Y"].astype(np.float32, copy=False)
    feat_cols = list(d["feature_names"])
    label_cols = list(d["label_names"])

    n = X.shape[0]
    rng = np.random.default_rng(SHUFFLE_SEED)
    idx = rng.permutation(n)
    split = int(n * (1 - VAL_FRACTION))

    train_idx = idx[:split]
    val_idx = idx[split:]

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_val = X[val_idx]
    Y_val = Y[val_idx]

    x_mean = x_std = y_mean = y_std = None

    if NORMALIZE_INPUTS:
        x_mean = X_train.mean(axis=0)
        x_std = X_train.std(axis=0)
        x_std[x_std < 1e-8] = 1.0
        X_train = (X_train - x_mean) / x_std
        X_val = (X_val - x_mean) / x_std

    if NORMALIZE_TARGETS:
        y_mean = Y_train.mean(axis=0)
        y_std = Y_train.std(axis=0)
        y_std[y_std < 1e-8] = 1.0
        Y_train = (Y_train - y_mean) / y_std
        Y_val = (Y_val - y_mean) / y_std

    # Keep tensors on CPU. Move batch-by-batch to GPU in training loop.
    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(Y_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(Y_val),
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
    )

    norm = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }

    return train_dl, val_dl, feat_cols, label_cols, norm


# ============================================================================
# OPTIMIZER / SCHEDULER
# ============================================================================

def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    opts = {
        "adam": lambda: torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        ),
        "adamw": lambda: torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        ),
        "sgd": lambda: torch.optim.SGD(
            model.parameters(),
            lr=LEARNING_RATE,
            momentum=0.9,
            weight_decay=WEIGHT_DECAY,
        ),
    }
    return opts[OPTIMIZER]()


def make_scheduler(optimizer):
    if SCHEDULER == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    if SCHEDULER == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=STEP_LR_EVERY,
            gamma=STEP_LR_GAMMA,
        )
    return None


# ============================================================================
# TRAIN / VALIDATE
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    model.train()
    total_loss = 0.0
    n = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * xb.shape[0]
        n += xb.shape[0]

    return total_loss / n


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    n = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(xb)
                loss = criterion(pred, yb)
        else:
            pred = model(xb)
            loss = criterion(pred, yb)

        total_loss += loss.item() * xb.shape[0]
        n += xb.shape[0]

    return total_loss / n


@torch.no_grad()
def per_target_metrics(model, loader, label_cols, norm, device, use_amp):
    model.eval()
    all_pred = []
    all_true = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(xb)
        else:
            pred = model(xb)

        all_pred.append(pred.cpu().numpy())
        all_true.append(yb.numpy())

    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)

    if norm["y_mean"] is not None:
        pred = pred * norm["y_std"] + norm["y_mean"]
        true = true * norm["y_std"] + norm["y_mean"]

    print(f"\n  {'target':>22s}  {'MAE':>12s}  {'RMSE':>12s}")
    print(f"  {'-'*22}  {'-'*12}  {'-'*12}")
    for i, name in enumerate(label_cols):
        diff = pred[:, i] - true[:, i]
        mae = np.abs(diff).mean()
        rmse = np.sqrt((diff ** 2).mean())
        print(f"  {name:>22s}  {mae:12.6f}  {rmse:12.6f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Data:   {data_path}")

    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    train_dl, val_dl, feat_cols, label_cols, norm = load_data(data_path)

    n_in = len(feat_cols)
    n_out = len(label_cols)
    n_train = len(train_dl.dataset)
    n_val = len(val_dl.dataset)

    print(f"\nSamples: {n_train:,} train, {n_val:,} val")
    print(f"Input:   {n_in} features")
    print(f"Output:  {n_out} targets")

    model = TrackNet(n_in, n_out).to(device)

    if USE_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer)

    use_amp = USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"\nOptimizer:  {OPTIMIZER}")
    print(f"Scheduler:  {SCHEDULER}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Workers:    {NUM_WORKERS}")
    print(f"AMP:        {use_amp}")
    print(f"Compile:    {USE_COMPILE and hasattr(torch, 'compile')}")
    print(f"Epochs:     {EPOCHS}\n")

    best_val = float("inf")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        t_epoch = time.time()

        train_loss = train_one_epoch(
            model, train_dl, criterion, optimizer, device, scaler, use_amp
        )
        val_loss = validate(model, val_dl, criterion, device, use_amp)

        if scheduler is not None:
            scheduler.step()

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            if SAVE_MODEL:
                Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": {
                            "n_in": n_in,
                            "n_out": n_out,
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
                    },
                    MODEL_PATH,
                )

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            dt = time.time() - t_epoch
            star = " *" if improved else ""
            print(
                f"epoch {epoch:3d}/{EPOCHS} | "
                f"train {train_loss:.6f} | "
                f"val {val_loss:.6f}{star} | "
                f"lr {lr:.2e} | "
                f"{dt:.1f}s"
            )

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best val loss: {best_val:.6f}")

    if SAVE_MODEL:
        ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

    per_target_metrics(model, val_dl, label_cols, norm, device, use_amp)

    if SAVE_MODEL:
        print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()