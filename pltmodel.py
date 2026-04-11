import os
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import HeteroTrackNet, SimpleTrackNet, SimpleTrackNetNOBNDROP, TestTrackNet
from helpers_vis import make_val_diagnostic_plots

# ===== Defaults =====
DEFAULT_CSV_PATH = "/nfs/cms/tracktrigger/logan/root/simvrico/SimToRecoDL/outputCSVs/filtered_particles.csv"
DEFAULT_BATCH_SIZE = 256
DEFAULT_PATTERN = "*.pt"

'''
Call this with:

python pltmodel.py \
  --model-dir goldenmodelsRUN3 \
  --recursive \
  --pattern "*.pt"


'''


# ===== Device =====
def get_device():
    return torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )


# ===== Seed =====
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


# ===== Data prep =====
def build_val_loader_from_checkpoint(
    csv_path,
    checkpoint,
    batch_size,
):
    feature_cols = checkpoint["feature_cols"]
    target_cols = checkpoint["target_cols"]

    x_mean = np.array(checkpoint["x_mean"], dtype=np.float32)
    x_std = np.array(checkpoint["x_std"], dtype=np.float32)
    y_mean = np.array(checkpoint["y_mean"], dtype=np.float32)
    y_std = np.array(checkpoint["y_std"], dtype=np.float32)

    seed = int(checkpoint.get("seed", 42))

    df = pd.read_csv(csv_path)

    X = df[feature_cols].to_numpy(dtype=np.float32)
    Y = df[target_cols].to_numpy(dtype=np.float32)

    # match training preprocessing
    X[X == -999.0] = 0.0

    n = len(X)
    val_fraction = float(checkpoint.get("val_fraction", 0.2))
    n_val = int(n * val_fraction)

    rng = np.random.default_rng(seed=seed)
    indices = rng.permutation(n)

    val_idx = indices[:n_val]

    X_val = X[val_idx]
    Y_val = Y[val_idx]

    # normalize using saved train stats from checkpoint
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)

    X_val = (X_val - x_mean) / x_std
    Y_val = (Y_val - y_mean) / y_std

    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32)

    val_dataset = TensorDataset(X_val, Y_val)

    g = torch.Generator()
    g.manual_seed(seed)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=g
    )

    return val_loader, target_cols, y_mean, y_std


# ===== Model rebuild =====
def activation_from_checkpoint(checkpoint):
    activation_name = checkpoint.get("activation", "ReLU")
    activations = {
        "ReLU": nn.ReLU,
        "Tanh": nn.Tanh,
        "Sigmoid": nn.Sigmoid,
        "GELU": nn.GELU,
        "LeakyReLU": nn.LeakyReLU,
    }

    if activation_name not in activations:
        raise ValueError(f"Unsupported activation in checkpoint: {activation_name}")

    return activations[activation_name]


def build_model_from_checkpoint(checkpoint, device):
    model_type = checkpoint.get("model_type", "HeteroTrackNet")
    feature_cols = checkpoint.get("feature_cols")
    target_cols = checkpoint.get("target_cols", [])

    if "input_dim" in checkpoint:
        input_dim = int(checkpoint["input_dim"])
    elif feature_cols is not None:
        input_dim = len(feature_cols)
    else:
        raise KeyError("Checkpoint must include input_dim or feature_cols.")

    output_dim = int(checkpoint.get("output_dim", len(target_cols) or 5))
    hidden_layers = checkpoint.get("hidden_layers", [64, 64])
    activation = activation_from_checkpoint(checkpoint)
    is_legacy_hetero = model_type == "HeteroTrackNet" and "model_type" not in checkpoint

    common_kwargs = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "activation": activation,
    }

    if model_type == "HeteroTrackNet":
        model = HeteroTrackNet(
            hidden_layers=hidden_layers,
            use_batchnorm=bool(checkpoint.get("use_batchnorm", True if is_legacy_hetero else False)),
            dropout=float(checkpoint.get("dropout", 0.10 if is_legacy_hetero else 0.0)),
            **common_kwargs,
        )
    elif model_type == "SimpleTrackNet":
        model = SimpleTrackNet(
            hidden_layers=hidden_layers,
            use_batchnorm=bool(checkpoint.get("use_batchnorm", False)),
            dropout=float(checkpoint.get("dropout", 0.0)),
            **common_kwargs,
        )
    elif model_type == "SimpleTrackNetNOBNDROP":
        model = SimpleTrackNetNOBNDROP(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            output_dim=output_dim,
            activation=activation,
        )
    elif model_type == "TestTrackNet":
        hidden_dim = int(checkpoint.get("hidden_dim", hidden_layers[0] if hidden_layers else 64))
        model = TestTrackNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
    else:
        raise ValueError(f"Unsupported model_type in checkpoint: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ===== Plot one checkpoint =====
def plot_one_model(
    model_path,
    csv_path,
    output_dir,
    batch_size,
    device,
    bins,
    density,
    show,
):
    print(f"\nLoading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    seed = int(checkpoint.get("seed", 42))
    set_seed(seed)

    model = build_model_from_checkpoint(checkpoint, device)

    val_loader, target_cols, y_mean, y_std = build_val_loader_from_checkpoint(
        csv_path=csv_path,
        checkpoint=checkpoint,
        batch_size=batch_size,
    )

    y_mean_t = torch.tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t = torch.tensor(y_std, dtype=torch.float32, device=device)

    phi_index = target_cols.index("pca_phi") if "pca_phi" in target_cols else None

    model_name = Path(model_path).stem

    plot_paths = make_val_diagnostic_plots(
        model=model,
        val_loader=val_loader,
        device=device,
        y_mean_t=y_mean_t,
        y_std_t=y_std_t,
        target_cols=target_cols,
        phi_index=phi_index,
        output_dir=output_dir,
        prefix=model_name,
        bins=bins,
        density=density,
        show=show,
    )

    print("Saved plots:")
    for plot_name, plot_path in plot_paths.items():
        print(f"  {plot_name}: {plot_path}")


# ===== File finding =====
def find_model_files(model_path=None, model_dir=None, recursive=False, pattern="*.pt"):
    found = []

    if model_path is not None:
        if os.path.isfile(model_path):
            found.append(model_path)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_dir is not None:
        p = Path(model_dir)
        if not p.is_dir():
            raise NotADirectoryError(f"Model directory not found: {model_dir}")

        if recursive:
            found.extend([str(x) for x in p.rglob(pattern)])
        else:
            found.extend([str(x) for x in p.glob(pattern)])

    found = sorted(set(found))

    if not found:
        raise FileNotFoundError("No model files found with the given path/directory settings.")

    return found


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot val predicted vs true distributions for one saved model or many."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-path", type=str, help="Path to a single .pt checkpoint")
    group.add_argument("--model-dir", type=str, help="Directory containing .pt checkpoints")

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search model-dir"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_PATTERN,
        help="Glob pattern for checkpoint discovery, default: *.pt"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=DEFAULT_CSV_PATH,
        help="Path to filtered_particles.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots_saved_models",
        help="Directory to save plots"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for val loader"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Number of histogram bins"
    )
    parser.add_argument(
        "--no-density",
        action="store_true",
        help="Plot counts instead of density"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Device set to {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    model_files = find_model_files(
        model_path=args.model_path,
        model_dir=args.model_dir,
        recursive=args.recursive,
        pattern=args.pattern,
    )

    print("\nFound model files:")
    for path in model_files:
        print(f"  {path}")

    for model_file in model_files:
        try:
            plot_one_model(
                model_path=model_file,
                csv_path=args.csv_path,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                device=device,
                bins=args.bins,
                density=not args.no_density,
                show=args.show,
            )
        except Exception as e:
            print(f"\nFAILED on {model_file}")
            print(f"Reason: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
