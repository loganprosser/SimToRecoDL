import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib
import pandas as pd

import torch
import torch.nn as nn

def angle_diff(pred, target):
    return torch.atan2(
        torch.sin(pred - target),
        torch.cos(pred - target)
    )

def paper_hetero_loss(y, mu, logvar, phi_idx=2, clamp_min=-10.0, clamp_max=5.0, target_weights=None):
    logvar = torch.clamp(logvar, clamp_min, clamp_max)

    # ----- Mean term: pure squared error -----
    residual = y - mu
    residual = residual.clone()
    residual[:, phi_idx] = angle_diff(mu[:, phi_idx], y[:, phi_idx])
    mse = residual ** 2

    # ----- Variance term: NLL with detached mean -----
    mu_detached = mu.detach()
    residual_detached = y - mu_detached
    residual_detached = residual_detached.clone()
    residual_detached[:, phi_idx] = angle_diff(mu_detached[:, phi_idx], y[:, phi_idx])

    nll = 0.5 * (
        logvar + residual_detached ** 2 * torch.exp(-logvar)
    )

    if target_weights is not None:
        target_weights = target_weights.to(y.device)
        mse = mse * target_weights
        nll = nll * target_weights
    
    
    return mse.mean() + nll.mean()



def bad_hetero_loss(y, mu, logvar):
    # mean prediction loss
    mse = (y - mu) ** 2

    # variance term (can be off by the scalars doesnt matter for optimizaion)
    var_loss = logvar + ((y.detach() - mu.detach()) ** 2) * torch.exp(-logvar)

    # combine both parts
    return mse.mean() + 0.5 * var_loss.mean()

def actual_herto_loss(y, mu, logvar):
    logvar = torch.clamp(logvar, min=-5, max=5)
    return (logvar + (y - mu)**2 * torch.exp(-logvar)).mean()