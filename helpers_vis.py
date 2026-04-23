import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from helpers import denormalize_targets, wrapped_angle_diff


def _add_figure_label(fig, figure_label):
    if not figure_label:
        return

    fig.text(
        0.995,
        0.005,
        str(figure_label),
        ha="right",
        va="bottom",
        fontsize=6,
        alpha=0.65,
    )


def unpack_model_output(output):
    if isinstance(output, (tuple, list)):
        pred = output[0]
        logvar = output[1] if len(output) > 1 else None
        return pred, logvar

    return output, None


def predict_mu_and_logvar(model, xb):
    return unpack_model_output(model(xb))


def wrap_phi_column(values, phi_index):
    if phi_index is None:
        return values

    values = values.clone()
    values[:, phi_index] = torch.atan2(
        torch.sin(values[:, phi_index]),
        torch.cos(values[:, phi_index])
    )
    return values


def print_final_validation_samples(
    model,
    val_loader,
    device,
    y_mean_t,
    y_std_t,
    target_cols,
    phi_index,
    num_examples=10
):
    print("\n" + "=" * 80)
    print("FINAL VALIDATION SAMPLES")
    print("=" * 80)

    model.eval()
    shown = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred, logvar = predict_mu_and_logvar(model, xb)

            pred_phys = denormalize_targets(pred, y_mean_t, y_std_t)
            yb_phys = denormalize_targets(yb, y_mean_t, y_std_t)

            err = pred_phys - yb_phys
            if phi_index is not None:
                err[:, phi_index] = wrapped_angle_diff(
                    pred_phys[:, phi_index],
                    yb_phys[:, phi_index]
                )

            std_phys = None
            if logvar is not None:
                std_phys = torch.exp(0.5 * logvar) * y_std_t

            for i in range(xb.size(0)):
                print(f"\nValidation example {shown + 1}")
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


def collect_val_predictions_and_targets(
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

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred, _ = predict_mu_and_logvar(model, xb)

            pred_phys = denormalize_targets(pred, y_mean_t, y_std_t)
            yb_phys = denormalize_targets(yb, y_mean_t, y_std_t)

            pred_phys = wrap_phi_column(pred_phys, phi_index)
            yb_phys = wrap_phi_column(yb_phys, phi_index)

            all_pred.append(pred_phys.detach().cpu())
            all_true.append(yb_phys.detach().cpu())

    y_pred = torch.cat(all_pred, dim=0).numpy()
    y_true = torch.cat(all_true, dim=0).numpy()

    return y_pred, y_true


def collect_val_predictions_targets_and_sigma(
    model,
    val_loader,
    device,
    y_mean_t,
    y_std_t,
    phi_index,
):
    y_pred, y_true = collect_val_predictions_and_targets(
        model=model,
        val_loader=val_loader,
        device=device,
        y_mean_t=y_mean_t,
        y_std_t=y_std_t,
        phi_index=phi_index,
    )

    model.eval()
    all_sigma = []

    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            _, logvar = predict_mu_and_logvar(model, xb)

            if logvar is None:
                return y_pred, y_true, None

            sigma_phys = torch.exp(0.5 * logvar) * y_std_t
            all_sigma.append(sigma_phys.detach().cpu())

    y_sigma = torch.cat(all_sigma, dim=0).numpy()
    return y_pred, y_true, y_sigma


def phi_wrapped_residuals(y_pred, y_true, phi_index):
    residuals = y_pred - y_true
    if phi_index is not None:
        residuals[:, phi_index] = np.arctan2(
            np.sin(residuals[:, phi_index]),
            np.cos(residuals[:, phi_index])
        )
    return residuals


def _finite_values(values):
    values = np.asarray(values)
    return values[np.isfinite(values)]


def _pad_range(vmin, vmax, pad_fraction=0.02):
    if vmin == vmax:
        return vmin - 0.5, vmax + 0.5

    pad = (vmax - vmin) * pad_fraction
    return vmin - pad, vmax + pad


def _central_range(values, central_fraction=0.99):
    values = _finite_values(values)

    if len(values) == 0:
        return -0.5, 0.5

    central_fraction = float(central_fraction)
    central_fraction = min(max(central_fraction, 0.0), 1.0)

    if central_fraction >= 1.0:
        return float(values.min()), float(values.max())

    tail_fraction = (1.0 - central_fraction) / 2.0
    vmin, vmax = np.quantile(values, [tail_fraction, 1.0 - tail_fraction])
    return float(vmin), float(vmax)


def _target_plot_values(y_true, y_pred, target_index):
    true_vals = _finite_values(y_true[:, target_index])
    pred_vals = _finite_values(y_pred[:, target_index])
    return np.concatenate([true_vals, pred_vals])


def _target_axis_range(y_true, y_pred, target_index, central_fraction=1.0, axis_limit=None, padded=True):
    if axis_limit is not None:
        vmin, vmax = axis_limit
    else:
        values = _target_plot_values(y_true, y_pred, target_index)
        vmin, vmax = _central_range(values, central_fraction=central_fraction)

    if padded:
        return _pad_range(vmin, vmax)

    if vmin == vmax:
        return vmin - 0.5, vmax + 0.5

    return vmin, vmax


def _finite_pair_mask(true_vals, pred_vals):
    return np.isfinite(true_vals) & np.isfinite(pred_vals)


def get_overlap_plot_range(y_true, y_pred, target_index, target_name=None, axis_limits=None):
    axis_limits = axis_limits or {}
    axis_limit = axis_limits.get(target_name)
    return _target_axis_range(
        y_true=y_true,
        y_pred=y_pred,
        target_index=target_index,
        central_fraction=1.0,
        axis_limit=axis_limit,
        padded=False,
    )


def compute_target_histogram_overlap(
    y_true,
    y_pred,
    target_index,
    target_cols,
    bins=100,
    axis_limits=None,
):
    target_name = target_cols[target_index]
    vmin, vmax = get_overlap_plot_range(
        y_true=y_true,
        y_pred=y_pred,
        target_index=target_index,
        target_name=target_name,
        axis_limits=axis_limits,
    )
    bin_edges = np.linspace(vmin, vmax, bins + 1)

    true_hist, _ = np.histogram(y_true[:, target_index], bins=bin_edges, density=True)
    pred_hist, _ = np.histogram(y_pred[:, target_index], bins=bin_edges, density=True)

    true_hist = np.nan_to_num(true_hist, nan=0.0, posinf=0.0, neginf=0.0)
    pred_hist = np.nan_to_num(pred_hist, nan=0.0, posinf=0.0, neginf=0.0)

    bin_widths = np.diff(bin_edges)
    overlap = np.sum(np.minimum(true_hist, pred_hist) * bin_widths)

    return float(overlap)


def _safe_std(values):
    if len(values) < 2:
        return 0.0
    return float(np.std(values))


def compute_target_scatter_linearity(
    y_true,
    y_pred,
    target_index,
):
    true_vals = y_true[:, target_index]
    pred_vals = y_pred[:, target_index]
    mask = _finite_pair_mask(true_vals, pred_vals)
    true_vals = true_vals[mask]
    pred_vals = pred_vals[mask]

    if len(true_vals) < 3:
        return {
            "score": -float("inf"),
            "corr": 0.0,
            "slope": 0.0,
            "intercept": 0.0,
            "slope_penalty": float("inf"),
            "intercept_penalty": float("inf"),
            "n": int(len(true_vals)),
        }

    true_std = _safe_std(true_vals)
    pred_std = _safe_std(pred_vals)

    if true_std == 0.0 or pred_std == 0.0:
        corr = 0.0
        slope = 0.0
    else:
        corr = float(np.corrcoef(true_vals, pred_vals)[0, 1])
        slope = float(np.cov(true_vals, pred_vals, ddof=0)[0, 1] / (true_std ** 2))

    intercept = float(pred_vals.mean() - slope * true_vals.mean())
    value_scale = max(true_std, pred_std, 1e-12)
    slope_penalty = abs(np.log(max(abs(slope), 1e-12)))
    intercept_penalty = abs(intercept) / value_scale

    score = corr - 0.20 * slope_penalty - 0.10 * intercept_penalty

    return {
        "score": float(score),
        "corr": float(corr),
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_penalty": float(slope_penalty),
        "intercept_penalty": float(intercept_penalty),
        "n": int(len(true_vals)),
    }


def compute_target_overlap_coverage(
    y_true,
    y_pred,
    target_index,
    target_cols,
    bins=100,
    axis_limits=None,
):
    target_name = target_cols[target_index]
    true_vals = _finite_values(y_true[:, target_index])
    pred_vals = _finite_values(y_pred[:, target_index])

    if len(true_vals) < 3 or len(pred_vals) < 3:
        return {
            "score": -float("inf"),
            "overlap": 0.0,
            "mean_penalty": float("inf"),
            "std_penalty": float("inf"),
            "spread_penalty": float("inf"),
            "n_true": int(len(true_vals)),
            "n_pred": int(len(pred_vals)),
        }

    overlap = compute_target_histogram_overlap(
        y_true=y_true,
        y_pred=y_pred,
        target_index=target_index,
        target_cols=target_cols,
        bins=bins,
        axis_limits=axis_limits,
    )

    true_std = max(_safe_std(true_vals), 1e-12)
    pred_std = max(_safe_std(pred_vals), 1e-12)
    mean_penalty = abs(float(pred_vals.mean() - true_vals.mean())) / true_std
    std_penalty = abs(np.log(pred_std / true_std))

    true_q_low, true_q_high = np.quantile(true_vals, [0.005, 0.995])
    pred_q_low, pred_q_high = np.quantile(pred_vals, [0.005, 0.995])
    true_spread = max(float(true_q_high - true_q_low), 1e-12)
    pred_spread = max(float(pred_q_high - pred_q_low), 1e-12)
    spread_penalty = abs(np.log(pred_spread / true_spread))

    score = overlap - 0.10 * mean_penalty - 0.20 * std_penalty - 0.20 * spread_penalty

    return {
        "score": float(score),
        "overlap": float(overlap),
        "mean_penalty": float(mean_penalty),
        "std_penalty": float(std_penalty),
        "spread_penalty": float(spread_penalty),
        "n_true": int(len(true_vals)),
        "n_pred": int(len(pred_vals)),
    }


def compute_plot_quality_scores(
    y_true,
    y_pred,
    target_cols,
    bins=100,
    axis_limits=None,
):
    scores = {
        "scatter": {},
        "overlap": {},
    }

    for i, name in enumerate(target_cols):
        scores["scatter"][name] = compute_target_scatter_linearity(
            y_true=y_true,
            y_pred=y_pred,
            target_index=i,
        )
        scores["overlap"][name] = compute_target_overlap_coverage(
            y_true=y_true,
            y_pred=y_pred,
            target_index=i,
            target_cols=target_cols,
            bins=bins,
            axis_limits=axis_limits,
        )

    return scores


def format_plot_quality_report(scores):
    lines = ["   Plot-quality scores:"]

    lines.append("      Scatter linearity:")
    for name, metric in scores["scatter"].items():
        lines.append(
            f"         {name}: score={metric['score']:.6f} | "
            f"corr={metric['corr']:.6f} | slope={metric['slope']:.6f} | "
            f"intercept_penalty={metric['intercept_penalty']:.6f}"
        )

    lines.append("      Overlap coverage:")
    for name, metric in scores["overlap"].items():
        lines.append(
            f"         {name}: score={metric['score']:.6f} | "
            f"overlap={metric['overlap']:.6f} | mean_pen={metric['mean_penalty']:.6f} | "
            f"std_pen={metric['std_penalty']:.6f} | spread_pen={metric['spread_penalty']:.6f}"
        )

    return "\n".join(lines)


def plot_overlap_distributions(
    y_true,
    y_pred,
    target_cols,
    bins=100,
    density=True,
    save_path=None,
    show=True,
    axis_limits=None,
    central_fraction=0.99,
    figure_label=None,
):
    n_targets = len(target_cols)
    fig, axes = plt.subplots(2, n_targets, figsize=(5 * n_targets, 8))
    axes = np.asarray(axes).reshape(2, n_targets)

    axis_limits = axis_limits or {}
    rows = [
        ("All finite data", 1.0),
        (f"Central {central_fraction * 100:.0f}%", central_fraction),
    ]

    for row_idx, (row_label, row_fraction) in enumerate(rows):
        for i, name in enumerate(target_cols):
            ax = axes[row_idx, i]

            true_vals = _finite_values(y_true[:, i])
            pred_vals = _finite_values(y_pred[:, i])
            axis_limit = axis_limits.get(name)
            vmin, vmax = _target_axis_range(
                y_true=y_true,
                y_pred=y_pred,
                target_index=i,
                central_fraction=row_fraction,
                axis_limit=axis_limit,
                padded=False,
            )

            if row_fraction < 1.0:
                true_vals = true_vals[(true_vals >= vmin) & (true_vals <= vmax)]
                pred_vals = pred_vals[(pred_vals >= vmin) & (pred_vals <= vmax)]

            bin_edges = np.linspace(vmin, vmax, bins + 1)

            ax.hist(true_vals, bins=bin_edges, alpha=0.5, label="Actual", density=density)
            ax.hist(pred_vals, bins=bin_edges, alpha=0.5, label="Predicted", density=density)

            ax.set_xlim(*_pad_range(vmin, vmax))
            ax.set_title(f"{name} - {row_label}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density" if density else "Count")
            ax.legend()

    _add_figure_label(fig, figure_label)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_pred_vs_true_distributions(*args, **kwargs):
    return plot_overlap_distributions(*args, **kwargs)


def plot_pred_vs_true_scatter(
    y_true,
    y_pred,
    target_cols,
    save_path=None,
    show=True,
    max_points=5000,
    seed=42,
    central_fraction=0.99,
    figure_label=None,
):
    n_targets = len(target_cols)
    fig, axes = plt.subplots(2, n_targets, figsize=(5 * n_targets, 8))
    axes = np.asarray(axes).reshape(2, n_targets)

    n_rows = len(y_true)
    if max_points is not None and n_rows > max_points:
        rng = np.random.default_rng(seed=seed)
        plot_idx = rng.choice(n_rows, size=max_points, replace=False)
    else:
        plot_idx = np.arange(n_rows)

    rows = [
        ("All finite data", 1.0),
        (f"Central {central_fraction * 100:.0f}%", central_fraction),
    ]

    for row_idx, (row_label, row_fraction) in enumerate(rows):
        for i, name in enumerate(target_cols):
            ax = axes[row_idx, i]
            true_vals = y_true[plot_idx, i]
            pred_vals = y_pred[plot_idx, i]
            pair_mask = _finite_pair_mask(true_vals, pred_vals)
            true_vals = true_vals[pair_mask]
            pred_vals = pred_vals[pair_mask]

            vmin, vmax = _target_axis_range(
                y_true=y_true,
                y_pred=y_pred,
                target_index=i,
                central_fraction=row_fraction,
                padded=False,
            )

            if row_fraction < 1.0:
                central_mask = (
                    (true_vals >= vmin)
                    & (true_vals <= vmax)
                    & (pred_vals >= vmin)
                    & (pred_vals <= vmax)
                )
                true_vals = true_vals[central_mask]
                pred_vals = pred_vals[central_mask]

            x_min, x_max = _pad_range(vmin, vmax)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(x_min, x_max)

            ax.scatter(true_vals, pred_vals, s=5, alpha=0.25, linewidths=0)
            ax.plot([vmin, vmax], [vmin, vmax], color="black", linewidth=1.0)
            ax.set_title(f"{name} - {row_label}")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")

    _add_figure_label(fig, figure_label)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_pull_distributions(
    y_true,
    y_pred,
    y_sigma,
    target_cols,
    phi_index,
    bins=100,
    density=True,
    save_path=None,
    show=True,
    figure_label=None,
):
    if y_sigma is None:
        print("Skipping pull plot because this model did not return sigma/logvar.")
        return

    residuals = phi_wrapped_residuals(y_pred, y_true, phi_index)
    sigma = np.maximum(y_sigma, 1e-12)
    pulls = residuals / sigma

    n_targets = len(target_cols)
    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 4))

    if n_targets == 1:
        axes = [axes]

    for i, name in enumerate(target_cols):
        ax = axes[i]
        vals = pulls[:, i]
        finite_vals = vals[np.isfinite(vals)]


        ax.hist(finite_vals, bins=bins, alpha=0.75, density=density)
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.axvline(-1.0, color="gray", linewidth=1.0, linestyle="--")
        ax.axvline(1.0, color="gray", linewidth=1.0, linestyle="--")
        
        # if name == "pca_dxy":
        #     ax.set_xlim(-100, 100)
        #     # optional: match y scaling too
        #     # ax.set_ylim(0, some_value)
        
        ax.set_title(f"{name} pull")
        ax.set_xlabel("(pred - actual) / sigma")
        ax.set_ylabel("Density" if density else "Count")

    _add_figure_label(fig, figure_label)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_distance_distribution(
    y_true,
    y_pred,
    target_cols,
    phi_index,
    y_sigma=None,
    bins=100,
    density=True,
    save_path=None,
    show=True,
    figure_label=None,
):
    residuals = phi_wrapped_residuals(y_pred, y_true, phi_index)

    if y_sigma is not None:
        sigma = np.maximum(y_sigma, 1e-12)
        values = np.sqrt(np.sum((residuals / sigma) ** 2, axis=1))
        xlabel = "sqrt(sum(pull^2))"
        title = "Normalized prediction distance"
    else:
        values = np.sqrt(np.sum(residuals ** 2, axis=1))
        xlabel = "sqrt(sum((pred - actual)^2))"
        title = "Physical residual distance"

    finite_values = values[np.isfinite(values)]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist(finite_values, bins=bins, alpha=0.75, density=density)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density" if density else "Count")

    if y_sigma is not None:
        expected = np.sqrt(len(target_cols))
        ax.axvline(expected, color="black", linewidth=1.0, linestyle="--", label="sqrt(n targets)")
        ax.legend()

    _add_figure_label(fig, figure_label)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def make_val_overlap_plot(
    model,
    val_loader,
    device,
    y_mean_t,
    y_std_t,
    target_cols,
    phi_index,
    save_path=None,
    bins=100,
    density=True,
    show=True,
    axis_limits=None,
):
    y_pred, y_true = collect_val_predictions_and_targets(
        model=model,
        val_loader=val_loader,
        device=device,
        y_mean_t=y_mean_t,
        y_std_t=y_std_t,
        phi_index=phi_index,
    )

    plot_overlap_distributions(
        y_true=y_true,
        y_pred=y_pred,
        target_cols=target_cols,
        bins=bins,
        density=density,
        save_path=save_path,
        show=show,
        axis_limits=axis_limits,
    )


def make_val_distribution_plots(*args, **kwargs):
    return make_val_overlap_plot(*args, **kwargs)


def make_val_diagnostic_plots(
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
):
    y_pred, y_true, y_sigma = collect_val_predictions_targets_and_sigma(
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
    )

    plot_pred_vs_true_scatter(
        y_true=y_true,
        y_pred=y_pred,
        target_cols=target_cols,
        save_path=paths["scatter"],
        show=show,
        max_points=scatter_max_points,
        central_fraction=central_fraction,
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
    )

    return paths


def plot_training_performance(
    history,
    save_path=None,
    show=True,
    figure_label=None,
):
    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    max_loss = 5

    train_loss = np.clip(history["train_loss"], None, max_loss)
    val_loss = np.clip(history["val_loss"], None, max_loss)

    axes[0].plot(epochs, train_loss, label="Train loss")
    axes[0].plot(epochs, val_loss, label="Val loss")
    
    axes[0].set_title("Loss over time")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["val_mean_mae"], label="Val mean MAE")
    axes[1].plot(epochs, history["val_mean_rmse"], label="Val mean RMSE")
    axes[1].set_title("Validation error over time")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Error")
    axes[1].legend()

    _add_figure_label(fig, figure_label)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_learning_rate_history(
    history,
    save_path=None,
    show=True,
    figure_label=None,
):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.plot(history["epoch"], history["learning_rate"])
    ax.set_title("Learning rate over time")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")

    _add_figure_label(fig, figure_label)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def make_training_history_plots(
    history,
    output_dir="plots",
    prefix="training",
    show=False,
    figure_label=None,
):
    if not history["epoch"]:
        print("Skipping training history plots because no epochs were recorded.")
        return {}

    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "performance": os.path.join(output_dir, f"{prefix}_performance_over_time.png"),
        "learning_rate": os.path.join(output_dir, f"{prefix}_learning_rate_over_time.png"),
    }

    plot_training_performance(
        history=history,
        save_path=paths["performance"],
        show=show,
        figure_label=figure_label,
    )
    plot_learning_rate_history(
        history=history,
        save_path=paths["learning_rate"],
        show=show,
        figure_label=figure_label,
    )

    return paths


def plot_overlap_history(
    history,
    target_name,
    save_path=None,
    show=True,
):
    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["overlap"], label="Histogram overlap")
    axes[0].set_title(f"{target_name} overlap over time")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Overlap")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend()

    axes[1].plot(epochs, history["mae"], label="MAE")
    axes[1].set_title(f"{target_name} MAE over time")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
