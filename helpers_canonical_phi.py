from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from helpers import denormalize_targets, wrapped_angle_diff
from helpers_data import DEFAULT_DATA_PATH, DEFAULT_TARGET_COLS, build_hit_feature_cols
from helpers_vis import (
    plot_distance_distribution,
    plot_overlap_distributions,
    plot_pred_vs_true_scatter,
    plot_pull_distributions,
    predict_mu_and_logvar,
)


@dataclass
class CanonicalPhiDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    x_train: torch.Tensor
    x_val: torch.Tensor
    y_train: torch.Tensor
    y_val: torch.Tensor
    rot_train: torch.Tensor
    rot_val: torch.Tensor
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    y_mean_t: torch.Tensor
    y_std_t: torch.Tensor
    feature_cols: List[str]
    target_cols: List[str]
    phi_index: Optional[int]
    rotation_source: str

    @property
    def input_dim(self):
        return self.x_train.shape[1]


def wrap_angle_np(values):
    return np.arctan2(np.sin(values), np.cos(values))


def wrap_angle_torch(values):
    return torch.atan2(torch.sin(values), torch.cos(values))


def _count_raw_masks_and_sentinels(df, feature_cols, x, sentinel_value, sentinel_replacement):
    sentinel_count = int((x == sentinel_value).sum())
    mask_cols = [col for col in feature_cols if col.endswith("_mask")]

    print("========== Raw mask/sentinel data check ==========")
    print(f"Rows loaded: {len(df):,}")
    print(f"Feature columns: {len(feature_cols):,}")
    print(f"Mask columns found: {len(mask_cols):,}")

    if mask_cols:
        mask_values = df[mask_cols].to_numpy(dtype=np.float32)
        present_hits = int((mask_values == 1).sum())
        missing_hits = int((mask_values == 0).sum())
        other_mask_values = int(mask_values.size - present_hits - missing_hits)

        print(f"Mask entries total: {mask_values.size:,}")
        print(f"Mask entries == 1: {present_hits:,}")
        print(f"Mask entries == 0: {missing_hits:,}")
        print(f"Mask entries other: {other_mask_values:,}")

        for col in mask_cols:
            values = df[col].to_numpy(dtype=np.float32)
            print(
                f"  {col}: ones={int((values == 1).sum()):,}, "
                f"zeros={int((values == 0).sum()):,}"
            )
    else:
        print("No *_mask feature columns found.")

    print(
        f"Sentinel values ({sentinel_value}) replaced with "
        f"{sentinel_replacement}: {sentinel_count:,}"
    )
    print("==================================================")


def _hit_indices(feature_cols, n_layers):
    col_to_idx = {col: idx for idx, col in enumerate(feature_cols)}
    indices = []

    for layer in range(1, n_layers + 1):
        indices.append(
            {
                "x": col_to_idx[f"hit_{layer}_x"],
                "y": col_to_idx[f"hit_{layer}_y"],
                "mask": col_to_idx.get(f"hit_{layer}_mask"),
            }
        )

    return indices


def compute_first_valid_hit_angles(x, feature_cols, n_layers=6):
    hit_indices = _hit_indices(feature_cols, n_layers)
    angles = np.zeros(x.shape[0], dtype=np.float32)
    assigned = np.zeros(x.shape[0], dtype=bool)

    for cols in hit_indices:
        x_vals = x[:, cols["x"]]
        y_vals = x[:, cols["y"]]

        if cols["mask"] is None:
            valid = np.isfinite(x_vals) & np.isfinite(y_vals) & ((x_vals != 0.0) | (y_vals != 0.0))
        else:
            valid = x[:, cols["mask"]] > 0.5

        use = valid & ~assigned
        angles[use] = np.arctan2(y_vals[use], x_vals[use])
        assigned[use] = True

    return angles, assigned


def rotate_hit_xy_features(x, feature_cols, rotation_angles, n_layers=6):
    x_rot = x.copy()
    cos_a = np.cos(rotation_angles)
    sin_a = np.sin(rotation_angles)

    for cols in _hit_indices(feature_cols, n_layers):
        x_idx = cols["x"]
        y_idx = cols["y"]
        x_old = x[:, x_idx].copy()
        y_old = x[:, y_idx].copy()

        x_rot[:, x_idx] = cos_a * x_old + sin_a * y_old
        x_rot[:, y_idx] = -sin_a * x_old + cos_a * y_old

    return x_rot


def rotate_phi_targets_to_canonical(y, target_cols, rotation_angles):
    y_rot = y.copy()

    if "pca_phi" in target_cols:
        phi_index = target_cols.index("pca_phi")
        y_rot[:, phi_index] = wrap_angle_np(y[:, phi_index] - rotation_angles)

    return y_rot


def recover_phi_from_canonical(y_phys, rotation_angles, phi_index):
    if phi_index is None:
        return y_phys

    recovered = y_phys.clone()
    recovered[:, phi_index] = wrap_angle_torch(recovered[:, phi_index] + rotation_angles)
    return recovered


def load_canonical_phi_track_data(
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
    print_mask_counts=False,
):
    if device is None:
        device = torch.device("cpu")

    target_cols = list(target_cols or DEFAULT_TARGET_COLS)
    feature_cols = list(feature_cols or build_hit_feature_cols(n_layers))
    phi_index = target_cols.index("pca_phi") if "pca_phi" in target_cols else None

    df = pd.read_csv(csv_path)
    x = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_cols].to_numpy(dtype=np.float32)

    if print_mask_counts:
        _count_raw_masks_and_sentinels(
            df=df,
            feature_cols=feature_cols,
            x=x,
            sentinel_value=sentinel_value,
            sentinel_replacement=sentinel_replacement,
        )

    x[x == sentinel_value] = sentinel_replacement

    rotation_angles, has_rotation_anchor = compute_first_valid_hit_angles(
        x=x,
        feature_cols=feature_cols,
        n_layers=n_layers,
    )
    x = rotate_hit_xy_features(
        x=x,
        feature_cols=feature_cols,
        rotation_angles=rotation_angles,
        n_layers=n_layers,
    )
    y = rotate_phi_targets_to_canonical(
        y=y,
        target_cols=target_cols,
        rotation_angles=rotation_angles,
    )

    print("========== Canonical phi rotation check ==========")
    print("Rotation source: first valid hit xy angle")
    print(f"Rows with a valid rotation anchor: {int(has_rotation_anchor.sum()):,}/{len(x):,}")
    print("==================================================")

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
    rot_train = rotation_angles[train_idx]
    rot_val = rotation_angles[val_idx]

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
    rot_train_t = torch.tensor(rot_train, dtype=torch.float32)
    rot_val_t = torch.tensor(rot_val, dtype=torch.float32)

    train_dataset = TensorDataset(x_train_t, y_train_t, rot_train_t)
    val_dataset = TensorDataset(x_val_t, y_val_t, rot_val_t)

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

    return CanonicalPhiDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        x_train=x_train_t,
        x_val=x_val_t,
        y_train=y_train_t,
        y_val=y_val_t,
        rot_train=rot_train_t,
        rot_val=rot_val_t,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        y_mean_t=y_mean_device,
        y_std_t=y_std_device,
        feature_cols=feature_cols,
        target_cols=target_cols,
        phi_index=phi_index,
        rotation_source="first_valid_hit_xy_angle",
    )


def print_canonical_data_shapes(data):
    print("X_train shape:", data.x_train.shape)
    print("X_val shape:  ", data.x_val.shape)
    print("Y_train shape:", data.y_train.shape)
    print("Y_val shape:  ", data.y_val.shape)
    print("rot_train shape:", data.rot_train.shape)
    print("rot_val shape:  ", data.rot_val.shape)

    xb, yb, rb = next(iter(data.train_loader))
    print("batch X shape:", xb.shape)
    print("batch Y shape:", yb.shape)
    print("batch rotation shape:", rb.shape)


def denormalize_and_recover_phi(y_norm, rotation_angles, y_mean_t, y_std_t, phi_index):
    y_phys = denormalize_targets(y_norm, y_mean_t, y_std_t)
    return recover_phi_from_canonical(y_phys, rotation_angles, phi_index)


def collect_canonical_val_predictions_targets_and_sigma(
    model,
    val_loader,
    device,
    y_mean_t,
    y_std_t,
    phi_index,
):
    model.eval()
    all_pred = []
    all_true = []
    all_sigma = []
    saw_logvar = False

    with torch.no_grad():
        for xb, yb, rb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            rb = rb.to(device)

            pred, logvar = predict_mu_and_logvar(model, xb)

            pred_phys = denormalize_and_recover_phi(pred, rb, y_mean_t, y_std_t, phi_index)
            yb_phys = denormalize_and_recover_phi(yb, rb, y_mean_t, y_std_t, phi_index)

            all_pred.append(pred_phys.detach().cpu())
            all_true.append(yb_phys.detach().cpu())

            if logvar is not None:
                saw_logvar = True
                sigma_phys = torch.exp(0.5 * logvar) * y_std_t
                all_sigma.append(sigma_phys.detach().cpu())

    y_pred = torch.cat(all_pred, dim=0).numpy()
    y_true = torch.cat(all_true, dim=0).numpy()
    y_sigma = torch.cat(all_sigma, dim=0).numpy() if saw_logvar else None

    return y_pred, y_true, y_sigma


def make_canonical_val_diagnostic_plots(
    model,
    val_loader,
    device,
    y_mean_t,
    y_std_t,
    target_cols,
    phi_index,
    output_dir="plots",
    prefix="val",
    bins=100,
    density=True,
    show=False,
    axis_limits=None,
    scatter_max_points=5000,
    central_fraction=0.99,
    figure_label=None,
):
    import os

    y_pred, y_true, y_sigma = collect_canonical_val_predictions_targets_and_sigma(
        model=model,
        val_loader=val_loader,
        device=device,
        y_mean_t=y_mean_t,
        y_std_t=y_std_t,
        phi_index=phi_index,
    )

    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "overlap": os.path.join(output_dir, f"{prefix}_overlap.png"),
        "scatter": os.path.join(output_dir, f"{prefix}_scatter_pred_vs_actual.png"),
        "distance": os.path.join(output_dir, f"{prefix}_distance.png"),
    }

    if y_sigma is not None:
        paths["pull"] = os.path.join(output_dir, f"{prefix}_pull.png")

    plot_overlap_distributions(
        y_true=y_true,
        y_pred=y_pred,
        target_cols=target_cols,
        bins=bins,
        density=density,
        save_path=paths["overlap"],
        show=show,
        axis_limits=axis_limits,
        central_fraction=central_fraction,
        figure_label=figure_label,
    )

    plot_pred_vs_true_scatter(
        y_true=y_true,
        y_pred=y_pred,
        target_cols=target_cols,
        save_path=paths["scatter"],
        show=show,
        max_points=scatter_max_points,
        central_fraction=central_fraction,
        figure_label=figure_label,
    )

    if y_sigma is not None:
        plot_pull_distributions(
            y_true=y_true,
            y_pred=y_pred,
            y_sigma=y_sigma,
            target_cols=target_cols,
            phi_index=phi_index,
            bins=bins,
            density=density,
            save_path=paths["pull"],
            show=show,
            figure_label=figure_label,
        )
    else:
        print("Skipping pull plot because this model did not return sigma/logvar.")

    plot_distance_distribution(
        y_true=y_true,
        y_pred=y_pred,
        y_sigma=y_sigma,
        target_cols=target_cols,
        phi_index=phi_index,
        bins=bins,
        density=density,
        save_path=paths["distance"],
        show=show,
        figure_label=figure_label,
    )

    return paths


def print_canonical_final_validation_samples(
    model,
    val_loader,
    device,
    y_mean_t,
    y_std_t,
    target_cols,
    phi_index,
    num_examples=10,
):
    print("\n" + "=" * 80)
    print("FINAL VALIDATION SAMPLES")
    print("=" * 80)

    model.eval()
    shown = 0

    with torch.no_grad():
        for xb, yb, rb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            rb = rb.to(device)

            pred, logvar = predict_mu_and_logvar(model, xb)

            pred_phys = denormalize_and_recover_phi(pred, rb, y_mean_t, y_std_t, phi_index)
            yb_phys = denormalize_and_recover_phi(yb, rb, y_mean_t, y_std_t, phi_index)

            err = pred_phys - yb_phys
            if phi_index is not None:
                err[:, phi_index] = wrapped_angle_diff(
                    pred_phys[:, phi_index],
                    yb_phys[:, phi_index],
                )

            std_phys = None
            if logvar is not None:
                std_phys = torch.exp(0.5 * logvar) * y_std_t

            for i in range(xb.size(0)):
                print(f"\nValidation example {shown + 1}")
                print(f"  canonical_rotation = {rb[i].item(): .6f}")
                for j, name in enumerate(target_cols):
                    line = (
                        f"  {name:8s} | "
                        f"true = {yb_phys[i, j].item(): .6f} | "
                        f"pred = {pred_phys[i, j].item(): .6f} | "
                        f"error = {err[i, j].item(): .6f}"
                    )
                    if std_phys is not None:
                        line += f" | uncertainty_std = {std_phys[i, j].item(): .6f}"
                    print(line)

                shown += 1
                if shown >= num_examples:
                    return
