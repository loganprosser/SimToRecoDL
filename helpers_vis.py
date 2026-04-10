import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from helpers import denormalize_targets, wrapped_angle_diff


def unpack_model_output(output):
    if isinstance(output, (tuple, list)):
        pred = output[0]
        logvar = output[1] if len(output) > 1 else None
        return pred, logvar

    return output, None


def predict_mu_and_logvar(model, xb):
    return unpack_model_output(model(xb))


def wrap_phi_column(values, phi_index):
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
    residuals[:, phi_index] = np.arctan2(
        np.sin(residuals[:, phi_index]),
        np.cos(residuals[:, phi_index])
    )
    return residuals


def plot_overlap_distributions(
    y_true,
    y_pred,
    target_cols,
    bins=100,
    density=True,
    save_path=None,
    show=True,
    axis_limits=None,
):
    n_targets = len(target_cols)
    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 4))

    if n_targets == 1:
        axes = [axes]

    axis_limits = axis_limits or {}

    for i, name in enumerate(target_cols):
        ax = axes[i]

        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]

        vmin = min(true_vals.min(), pred_vals.min())
        vmax = max(true_vals.max(), pred_vals.max())
        
        if i == 0:
            vmin, vmax = -.1,.1
        
        elif i == 3:
            vmin, vmax = -.005, .005
        

        if name in axis_limits:
            vmin, vmax = axis_limits[name]

        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5

        bin_edges = np.linspace(vmin, vmax, bins + 1)

        ax.hist(true_vals, bins=bin_edges, alpha=0.5, label="Actual", density=density)
        ax.hist(pred_vals, bins=bin_edges, alpha=0.5, label="Predicted", density=density)

        ax.set_title(name)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density" if density else "Count")
        ax.legend()

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
):
    n_targets = len(target_cols)
    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 4))

    if n_targets == 1:
        axes = [axes]

    n_rows = len(y_true)
    if n_rows > max_points:
        rng = np.random.default_rng(seed=seed)
        plot_idx = rng.choice(n_rows, size=max_points, replace=False)
    else:
        plot_idx = np.arange(n_rows)

    for i, name in enumerate(target_cols):
        ax = axes[i]
        true_vals = y_true[plot_idx, i]
        pred_vals = y_pred[plot_idx, i]

        vmin = min(true_vals.min(), pred_vals.min())
        vmax = max(true_vals.max(), pred_vals.max())
        
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
            
        if i == 3:
            vmin, vmax = -0.005, 0.005
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
        else:
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
            
        ax.scatter(true_vals, pred_vals, s=5, alpha=0.25, linewidths=0)
        ax.plot([vmin, vmax], [vmin, vmax], color="black", linewidth=1.0)
        ax.set_title(name)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")

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
        
        # if i == 3:
        #     finite_vals = np.clip(finite_vals, -5, 5)  # example bounds
        #     ax.set_xlim(-5, 5)

        ax.hist(finite_vals, bins=bins, alpha=0.75, density=density)
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.axvline(-1.0, color="gray", linewidth=1.0, linestyle="--")
        ax.axvline(1.0, color="gray", linewidth=1.0, linestyle="--")
        ax.set_title(f"{name} pull")
        ax.set_xlabel("(pred - actual) / sigma")
        ax.set_ylabel("Density" if density else "Count")

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
    )

    plot_pred_vs_true_scatter(
        y_true=y_true,
        y_pred=y_pred,
        target_cols=target_cols,
        save_path=paths["scatter"],
        show=show,
        max_points=scatter_max_points,
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
):
    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["val_loss"], label="Val loss")
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
):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.plot(history["epoch"], history["learning_rate"])
    ax.set_title("Learning rate over time")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")

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
    )
    plot_learning_rate_history(
        history=history,
        save_path=paths["learning_rate"],
        show=show,
    )

    return paths
