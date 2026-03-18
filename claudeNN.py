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
ACTIVATION = "silu"          # "relu", "gelu", "silu", "tanh", "leaky_relu"
DROPOUT = 0.0                # set 0 for max speed first
BATCH_NORM = True

BATCH_SIZE = 32768           # try 8192 / 16384 / 32768 depending on VRAM
LEARNING_RATE = 1e-3

# Regularization
WEIGHT_DECAY = 0.0           # optimizer-level decay
L1_LAMBDA = 0.0              # explicit L1 penalty in loss
L2_LAMBDA = 0.0              # explicit L2 penalty in loss
REG_ON_BIAS = False          # usually False
REG_ON_BATCHNORM = False     # usually False



EPOCHS = 5
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
NUM_WORKERS = 0             #min(8, os.cpu_count() or 1)
PIN_MEMORY = True
PERSISTENT_WORKERS = False
PREFETCH_FACTOR = None

# CUDA speed knobs
USE_AMP = True               # mixed precision on CUDA
USE_COMPILE = False           # torch.compile for PyTorch 2.x

# ==== HELPERS ======
def regularization_loss(
    model: nn.Module,
    l1_lambda: float = 0.0,
    l2_lambda: float = 0.25,
    reg_on_bias: bool = False,
    reg_on_batchnorm: bool = False,
) -> torch.Tensor:
    reg_loss = None

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Skip bias unless requested
        if not reg_on_bias and name.endswith(".bias"):
            continue

        # Skip batchnorm params unless requested
        if not reg_on_batchnorm and ("bn" in name.lower() or "batchnorm" in name.lower()):
            continue

        term = 0.0

        if l1_lambda > 0:
            term = term + l1_lambda * param.abs().sum()

        if l2_lambda > 0:
            term = term + l2_lambda * param.pow(2).sum()

        if term != 0.0:
            reg_loss = term if reg_loss is None else reg_loss + term

    if reg_loss is None:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)

    return reg_loss

def get_activation(name: str) -> nn.Module:
    return {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
    }[name]()


# ============================================================================
# MODEL
# ============================================================================

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
        self.net = nn.Sequential(*layers) #links together so we can just call self.net() for foward pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TrackLossInd(nn.Module):
    
    def __init__(self, label_cols, norm, delta = 1.0, target_weights: dict[str, float] | None = None,):
        super().__init__()
        # small delta → switches to linear earlier → more robust to outliers
        # large delta → more quadratic behavior → closer to MSE
        
        self.label_cols = label_cols
        self.delta = delta
        self.idx = {name: i for i, name in enumerate(label_cols)}
        self.norm = norm
        
        default_weights = {
            "sim_pca_pt": 1.0,
            "sim_pca_eta": 1.0,
            "sim_pca_phi": 1.0,
            "sim_pca_dxy": 1.0,
            "sim_pca_dz": 1.0,
        }
        
        if target_weights is not None:
            default_weights.update(target_weights)
        self.target_weights = default_weights
        
        # what does this norm do?
        y_mean = norm["y_mean"]
        y_std = norm["y_std"]

        if y_mean is not None and y_std is not None:
            self.register_buffer("y_mean", torch.tensor(y_mean, dtype=torch.float32))
            self.register_buffer("y_std", torch.tensor(y_std, dtype=torch.float32))
            self.has_target_norm = True
        else:
            self.y_mean = None
            self.y_std = None
            self.has_target_norm = False
    
    def smooth_l1(self, diff: torch.Tensor) -> torch.Tensor:
        abs_diff = diff.abs()
        return torch.where(
            abs_diff < self.delta,
            0.5 * diff.pow(2) / self.delta,
            abs_diff - 0.5 * self.delta,
        )

    def denorm_component(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        if not self.has_target_norm:
            return x
        return x * self.y_std[idx] + self.y_mean[idx]

    def wrapped_phi_diff(self, pred_phi: torch.Tensor, true_phi: torch.Tensor) -> torch.Tensor:
        return torch.atan2(
            torch.sin(pred_phi - true_phi),
            torch.cos(pred_phi - true_phi),
        )
    
    def component_losses(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        losses = {}
        
        # pt
        if "sim_pca_pt" in self.idx:
            i = self.idx["sim_pca_pt"]
            diff = pred[:, i] - target[:, i]
            losses["sim_pca_pt"] = self.smooth_l1(diff).mean()

        # eta
        if "sim_pca_eta" in self.idx:
            i = self.idx["sim_pca_eta"]
            diff = pred[:, i] - target[:, i]
            losses["sim_pca_eta"] = self.smooth_l1(diff).mean()

        # phi (wrapped angular residual)
        if "sim_pca_phi" in self.idx:
            i = self.idx["sim_pca_phi"]

            pred_phi = self.denorm_component(pred[:, i], i)
            true_phi = self.denorm_component(target[:, i], i)

            diff_phi = self.wrapped_phi_diff(pred_phi, true_phi)

            # put back into normalized units if targets are normalized
            if self.has_target_norm:
                diff_phi = diff_phi / self.y_std[i]

            losses["sim_pca_phi"] = self.smooth_l1(diff_phi).mean()

        # dxy
        if "sim_pca_dxy" in self.idx:
            i = self.idx["sim_pca_dxy"]
            diff = pred[:, i] - target[:, i]
            losses["sim_pca_dxy"] = self.smooth_l1(diff).mean()

        # dz
        if "sim_pca_dz" in self.idx:
            i = self.idx["sim_pca_dz"]
            diff = pred[:, i] - target[:, i]
            losses["sim_pca_dz"] = self.smooth_l1(diff).mean()

        return losses
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = self.component_losses(pred, target)

        total = 0.0
        for name, loss_val in losses.items():
            total = total + self.target_weights.get(name, 1.0) * loss_val

        return total
    
    


class TrackLoss(nn.Module):
    def __init__(
        self,
        label_cols,
        norm,
        delta: float = 1.0,
        target_weights: dict[str, float] | None = None,
    ):
        super().__init__()
        self.label_cols = label_cols
        self.delta = delta

        weights = torch.ones(len(label_cols), dtype=torch.float32)
        if target_weights is not None:
            for i, name in enumerate(label_cols):
                if name in target_weights:
                    weights[i] = float(target_weights[name])

        self.register_buffer("weights", weights)

        self.phi_idx = (
            label_cols.index("sim_pca_phi")
            if "sim_pca_phi" in label_cols
            else None
        )

        y_mean = norm["y_mean"]
        y_std = norm["y_std"]

        if y_mean is not None and y_std is not None:
            self.register_buffer("y_mean", torch.tensor(y_mean, dtype=torch.float32))
            self.register_buffer("y_std", torch.tensor(y_std, dtype=torch.float32))
            self.has_target_norm = True
        else:
            self.y_mean = None
            self.y_std = None
            self.has_target_norm = False

    def smooth_l1_per_element(self, diff: torch.Tensor) -> torch.Tensor:
        abs_diff = diff.abs()
        return torch.where(
            abs_diff < self.delta,
            0.5 * diff**2 / self.delta,
            abs_diff - 0.5 * self.delta,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target

        if self.phi_idx is not None:
            if self.has_target_norm:
                pred_phi = (
                    pred[:, self.phi_idx] * self.y_std[self.phi_idx]
                    + self.y_mean[self.phi_idx]
                )
                true_phi = (
                    target[:, self.phi_idx] * self.y_std[self.phi_idx]
                    + self.y_mean[self.phi_idx]
                )
            else:
                pred_phi = pred[:, self.phi_idx]
                true_phi = target[:, self.phi_idx]

            phi_diff = torch.atan2(
                torch.sin(pred_phi - true_phi),
                torch.cos(pred_phi - true_phi),
            )

            if self.has_target_norm:
                phi_diff = phi_diff / self.y_std[self.phi_idx]

            diff = diff.clone()
            diff[:, self.phi_idx] = phi_diff

        loss_per_elem = self.smooth_l1_per_element(diff)
        weighted = loss_per_elem * self.weights.view(1, -1)
        return weighted.mean()

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
    total_base_loss = 0.0
    total_reg_loss = 0.0
    n = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(xb)
                base_loss = criterion(pred, yb)
                reg_loss = regularization_loss(
                    model,
                    l1_lambda=L1_LAMBDA,
                    l2_lambda=L2_LAMBDA,
                    reg_on_bias=REG_ON_BIAS,
                    reg_on_batchnorm=REG_ON_BATCHNORM,
                )
                loss = base_loss + reg_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(xb)
            base_loss = criterion(pred, yb)
            reg_loss = regularization_loss(
                model,
                l1_lambda=L1_LAMBDA,
                l2_lambda=L2_LAMBDA,
                reg_on_bias=REG_ON_BIAS,
                reg_on_batchnorm=REG_ON_BATCHNORM,
            )
            loss = base_loss + reg_loss
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * xb.shape[0]
        total_base_loss += base_loss.item() * xb.shape[0]
        total_reg_loss += reg_loss.item() * xb.shape[0]
        n += xb.shape[0]

    return total_loss / n, total_base_loss / n, total_reg_loss / n


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
def per_target_loss_breakdown(model, loader, criterion, device, use_amp):
    model.eval()

    raw_totals = {name: 0.0 for name in criterion.label_cols}
    weighted_totals = {name: 0.0 for name in criterion.label_cols}
    n = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(xb)
                losses = criterion.component_losses(pred, yb)
        else:
            pred = model(xb)
            losses = criterion.component_losses(pred, yb)

        batch_n = xb.shape[0]
        for name, val in losses.items():
            raw_totals[name] += val.item() * batch_n
            weighted_totals[name] += criterion.target_weights.get(name, 1.0) * val.item() * batch_n
        n += batch_n

    print("\nPer-target val loss breakdown:")
    print(f"  {'target':>22s}  {'raw':>12s}  {'weighted':>12s}  {'weight':>8s}")
    print(f"  {'-'*22}  {'-'*12}  {'-'*12}  {'-'*8}")
    for name in criterion.label_cols:
        raw = raw_totals[name] / n
        weighted = weighted_totals[name] / n
        weight = criterion.target_weights.get(name, 1.0)
        print(f"  {name:>22s}  {raw:12.6f}  {weighted:12.6f}  {weight:8.3f}")

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

        if name == "sim_pca_phi":
            diff = np.arctan2(np.sin(diff), np.cos(diff))

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

    criterion = TrackLossInd(
        label_cols=label_cols,
        norm=norm,
        delta=1.0,
        target_weights={
            "sim_pca_pt": 1.0,
            "sim_pca_eta": 1.0,
            "sim_pca_phi": 1.0,
            "sim_pca_dxy": 1.0,
            "sim_pca_dz": 1.0,
        },
    ).to(device)

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
    
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"L1 lambda:    {L1_LAMBDA}")
    print(f"L2 lambda:    {L2_LAMBDA}")
    print(f"Reg bias:     {REG_ON_BIAS}")
    print(f"Reg BN:       {REG_ON_BATCHNORM}")

    best_val = float("inf")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        t_epoch = time.time()
        
        train_loss, train_base_loss, train_reg_loss = train_one_epoch(
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
                f"base {train_base_loss:.6f} | "
                f"reg {train_reg_loss:.6f} | "
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
    per_target_loss_breakdown(model, val_dl, criterion, device, use_amp)
    
    if SAVE_MODEL:
        print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
    