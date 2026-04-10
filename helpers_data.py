import random
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_DATA_PATH = "/nfs/cms/tracktrigger/logan/root/simvrico/SimToRecoDL/outputCSVs/filtered_particles.csv"
DEFAULT_TARGET_COLS = [
    "pca_c",
    "pca_eta",
    "pca_phi",
    "pca_dxy",
    "pca_dz",
]


@dataclass
class TrackDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    x_train: torch.Tensor
    x_val: torch.Tensor
    y_train: torch.Tensor
    y_val: torch.Tensor
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    y_mean_t: torch.Tensor
    y_std_t: torch.Tensor
    feature_cols: List[str]
    target_cols: List[str]
    phi_index: int

    @property
    def input_dim(self):
        return self.x_train.shape[1]


def build_hit_feature_cols(n_layers=6):
    feature_cols = []
    for j in range(1, n_layers + 1):
        feature_cols += [
            f"hit_{j}_x",
            f"hit_{j}_y",
            f"hit_{j}_z",
            f"hit_{j}_r",
            f"hit_{j}_mask",
        ]
    return feature_cols


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_track_data(
    csv_path=DEFAULT_DATA_PATH,
    batch_size=256,
    seed=42,
    device=None,
    val_fraction=0.2,
    target_cols=None,
    feature_cols=None,
    n_layers=6,
    sentinel_value=-999.0,
    sentinel_replacement=0.0,
):
    if device is None:
        device = torch.device("cpu")

    target_cols = list(target_cols or DEFAULT_TARGET_COLS)
    feature_cols = list(feature_cols or build_hit_feature_cols(n_layers))
    phi_index = target_cols.index("pca_phi")

    df = pd.read_csv(csv_path)
    x = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_cols].to_numpy(dtype=np.float32)

    x[x == sentinel_value] = sentinel_replacement

    n_rows = len(x)
    n_val = int(n_rows * val_fraction)
    rng = np.random.default_rng(seed=seed)
    indices = rng.permutation(n_rows)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    x_train = x[train_idx]
    x_val = x[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std[x_std < 1e-8] = 1.0

    x_train = (x_train - x_mean) / x_std
    x_val = (x_val - x_mean) / x_std

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std[y_std < 1e-8] = 1.0

    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(x_train_t, y_train_t)
    val_dataset = TensorDataset(x_val_t, y_val_t)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=generator,
    )

    y_mean_device = torch.tensor(y_mean, dtype=torch.float32, device=device)
    y_std_device = torch.tensor(y_std, dtype=torch.float32, device=device)

    return TrackDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        x_train=x_train_t,
        x_val=x_val_t,
        y_train=y_train_t,
        y_val=y_val_t,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        y_mean_t=y_mean_device,
        y_std_t=y_std_device,
        feature_cols=feature_cols,
        target_cols=target_cols,
        phi_index=phi_index,
    )


def print_data_shapes(data):
    print("X_train shape:", data.x_train.shape)
    print("X_val shape:  ", data.x_val.shape)
    print("Y_train shape:", data.y_train.shape)
    print("Y_val shape:  ", data.y_val.shape)

    xb, yb = next(iter(data.train_loader))
    print("batch X shape:", xb.shape)
    print("batch Y shape:", yb.shape)
