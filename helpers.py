import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import random

def angle_diff(pred, target):
    return torch.atan2(
        torch.sin(pred - target),
        torch.cos(pred - target)
    )

def print_final_validation_samples(
    model,
    val_loader,
    device,
    denormalize_targets,
    y_std_t,
    TARGET_COLS,
    PHI_INDEX,
    wrapped_angle_diff,
    num_examples=10
):
    print("\n" + "="*80)
    print("FINAL VALIDATION SAMPLES")
    print("="*80)

    model.eval()
    shown = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            mu, logvar = model(xb)

            # std in normalized space
            std = torch.exp(0.5 * logvar)

            # de-normalize
            mu_phys = denormalize_targets(mu)
            yb_phys = denormalize_targets(yb)

            # uncertainty in physical units
            std_phys = std * y_std_t

            # error in physical units
            err = mu_phys - yb_phys

            # wrap phi error
            err[:, PHI_INDEX] = wrapped_angle_diff(
                mu_phys[:, PHI_INDEX],
                yb_phys[:, PHI_INDEX]
            )

            for i in range(xb.size(0)):
                print(f"\nValidation example {shown + 1}")
                for j, name in enumerate(TARGET_COLS):
                    print(
                        f"  {name:8s} | "
                        f"true = {yb_phys[i, j].item(): .6f} | "
                        f"pred = {mu_phys[i, j].item(): .6f} | "
                        f"error = {err[i, j].item(): .6f} | "
                        f"uncertainty_std = {std_phys[i, j].item(): .6f}"
                    )

                shown += 1
                if shown >= num_examples:
                    return