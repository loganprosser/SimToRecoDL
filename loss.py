import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib
import pandas as pd

import torch
import torch.nn as nn

from helpers import angle_diff

def huber_loss_with_phi(
    y,
    pred,
    phi_index=None,
    delta=0.5,
    target_weights=None,
    lambda_corr=0.20,
    lambda_logstd=1.0,
    lambda_mean=0.05,
    lambda_l2pred=0.05,
    lambda_res_corr=0.20,
    eps=1e-8,
):
    diff = pred - y

    if phi_index is not None:
        diff = diff.clone()
        diff[:, phi_index] = angle_diff(pred[:, phi_index], y[:, phi_index])

    # -------------------------
    # Huber
    # -------------------------
    abs_diff = diff.abs()
    huber = torch.where(
        abs_diff <= delta,
        0.5 * diff ** 2,
        delta * (abs_diff - 0.5 * delta),
    )

    # 🔥 ADD THIS (Step 5)
    mag_weight = (y.abs() / (y.abs().mean() + eps)).clamp(min=0.5, max=5.0)
    huber = huber * mag_weight

    if target_weights is not None:
        target_weights = target_weights.to(y.device, dtype=y.dtype).view(1, -1)
        huber = huber * target_weights

    huber_loss = huber.mean()

    # -------------------------
    # Stats
    # -------------------------
    if phi_index is not None and pred.shape[1] > 1:
        keep = [i for i in range(pred.shape[1]) if i != phi_index]
        pred_stats = pred[:, keep]
        y_stats = y[:, keep]
    else:
        pred_stats = pred
        y_stats = y

    pred_mean = pred_stats.mean(dim=0)
    y_mean = y_stats.mean(dim=0)

    pred_centered = pred_stats - pred_mean
    y_centered = y_stats - y_mean

    pred_std = torch.sqrt((pred_centered**2).mean(dim=0) + eps)
    y_std = torch.sqrt((y_centered**2).mean(dim=0) + eps)

    # -------------------------
    # Correlation
    # -------------------------
    corr = (pred_centered * y_centered).mean(dim=0) / (pred_std * y_std)
    corr = torch.clamp(corr, -1.0, 1.0)
    corr_loss = (1.0 - corr).mean()

    # -------------------------
    # Log std
    # -------------------------
    logstd_loss = ((torch.log(pred_std + eps) - torch.log(y_std + eps)) ** 2).mean()

    # -------------------------
    # Mean
    # -------------------------
    mean_loss = ((pred_mean - y_mean) ** 2).mean()

    # -------------------------
    # L2 pred
    # -------------------------
    l2pred_loss = (pred_stats ** 2).mean()

    # -------------------------
    # 🔥 ADD THIS (Step 6)
    # Residual correlation
    # -------------------------
    residual = pred_stats - y_stats
    res_centered = residual - residual.mean(dim=0)

    res_std = torch.sqrt((res_centered**2).mean(dim=0) + eps)

    res_corr = (res_centered * y_centered).mean(dim=0) / (res_std * y_std)
    res_corr_loss = (res_corr ** 2).mean()

    return (
        huber_loss
        + lambda_corr * corr_loss
        + lambda_logstd * logstd_loss
        + lambda_mean * mean_loss
        + lambda_l2pred * l2pred_loss
        + lambda_res_corr * res_corr_loss
    ) 
    
def hetero_huber_corr_loss(
    y,
    mu,
    logvar,
    phi_index=None,
    delta=1.0,
    min_logvar=-6.0,
    max_logvar=4.0,
    target_weights=None,
    mean_weights=None,
    lambda_corr=0.25,
    lambda_tail=0.0,
    tail_power=1.0,
    eps=1e-8,
):
    """
    Robust heteroscedastic multi-target loss with optional correlation term.

    Inputs
    ------
    y       : [B, D] true targets
    mu      : [B, D] predicted means
    logvar  : [B, D] predicted log variances

    phi_index:
        Optional column index for wrapped angular target.

    delta:
        Huber threshold. If targets are normalized, 0.5 to 1.0 is a good start.

    lambda_corr:
        Weight on correlation penalty. Higher = stronger push to track target shape.

    lambda_tail:
        Optional extra weighting for larger-|y| examples so dense near-zero regions
        do not dominate as much.

    tail_power:
        Controls how strongly large-|y| examples are emphasized.

    Returns
    -------
    scalar loss
    """

    # keep predicted variance from becoming absurd
    logvar = torch.clamp(logvar, min=min_logvar, max=max_logvar)

    # residuals
    diff = y - mu

    # wrap phi residual if needed
    if phi_index is not None:
        diff = diff.clone()
        diff[:, phi_index] = angle_diff(mu[:, phi_index], y[:, phi_index])

    # -------------------------
    # 1. robust Huber data term
    # -------------------------
    abs_diff = diff.abs()
    huber = torch.where(
        abs_diff <= delta,
        0.5 * diff ** 2,
        delta * (abs_diff - 0.5 * delta),
    )

    # optional per-target weighting on mean-fit part
    if mean_weights is not None:
        mean_weights = mean_weights.to(y.device, dtype=y.dtype).view(1, -1)
    else:
        mean_weights = 1.0

    # -------------------------
    # 2. heteroscedastic weighting
    # -------------------------
    # This still predicts uncertainty.
    # exp(0.5 * logvar) would be sigma.
    hetero_loss = 0.5 * logvar + mean_weights * huber * torch.exp(-logvar)

    # optional extra emphasis on larger-|y| values
    if lambda_tail > 0.0:
        tail_scale = 1.0 + lambda_tail * (y.abs() ** tail_power)
        hetero_loss = hetero_loss * tail_scale

    # optional per-target weighting on total target contribution
    if target_weights is not None:
        target_weights = target_weights.to(y.device, dtype=y.dtype).view(1, -1)
        hetero_loss = hetero_loss * target_weights

    hetero_loss = hetero_loss.mean()

    # -------------------------
    # 3. correlation penalty on mu
    # -------------------------
    # We do this on mean predictions only, not variance.
    # For wrapped phi, raw Pearson correlation is usually not meaningful,
    # so we exclude that dimension from correlation if there are other dims.
    if phi_index is not None and mu.shape[1] > 1:
        keep = [i for i in range(mu.shape[1]) if i != phi_index]
        mu_corr = mu[:, keep]
        y_corr = y[:, keep]
    else:
        mu_corr = mu
        y_corr = y

    mu_centered = mu_corr - mu_corr.mean(dim=0, keepdim=True)
    y_centered = y_corr - y_corr.mean(dim=0, keepdim=True)

    mu_std = torch.sqrt((mu_centered ** 2).mean(dim=0) + eps)
    y_std = torch.sqrt((y_centered ** 2).mean(dim=0) + eps)

    corr = (mu_centered * y_centered).mean(dim=0) / (mu_std * y_std)
    corr = torch.clamp(corr, -1.0, 1.0)

    corr_loss = (1.0 - corr).mean()

    # final combined loss
    return hetero_loss + lambda_corr * corr_loss



def hetero_gaussian_nll_with_phi_relative(
    y,
    mu,
    logvar,
    phi_index=None,
    min_logvar=-10.0,
    max_logvar=10.0,
    target_weights=None,
    mean_weights=None,
    lambda_rel=1.0,
    eps=1e-6,
):
    # lambda_rel is a weighting tensor here for how much we wieght the relative_loss for each component in standard order as always
    
    logvar = torch.clamp(logvar, min=min_logvar, max=max_logvar)

    diff = y - mu

    if phi_index is not None:
        diff = diff.clone()
        diff[:, phi_index] = angle_diff(mu[:, phi_index], y[:, phi_index])

    sq_error = diff ** 2

    if mean_weights is not None:
        mean_weights = mean_weights.to(y.device).view(1, -1)
    else:
        mean_weights = 1.0

    hetero_loss = 0.5 * (
        logvar + mean_weights * sq_error * torch.exp(-logvar)
    )

    # can also change this loss to not be squared if its too dominating
    scale = torch.abs(y) + eps
    relative_loss = sq_error / scale

    if not torch.is_tensor(lambda_rel):
        lambda_rel = torch.tensor(lambda_rel, device=y.device, dtype=y.dtype)

    if lambda_rel.ndim == 0:
        lambda_rel = lambda_rel.view(1, 1)
    else:
        lambda_rel = lambda_rel.to(y.device, dtype=y.dtype).view(1, -1)

    total_loss = hetero_loss + lambda_rel * relative_loss

    if target_weights is not None:
        target_weights = target_weights.to(y.device).view(1, -1)
        total_loss = total_loss * target_weights

    return total_loss.mean()

def hetero_gaussian_nll_with_phi(
    y,
    mu,
    logvar,
    phi_index=None,
    min_logvar=-10.0,
    max_logvar=10.0,
    target_weights=None,
    mean_weights=None
):
    logvar = torch.clamp(logvar, min=min_logvar, max=max_logvar)

    diff = y - mu

    if phi_index is not None:
        diff = diff.clone()
        diff[:, phi_index] = angle_diff(mu[:, phi_index], y[:, phi_index])

    sq_error = diff ** 2
    
    if mean_weights is not None:
        mean_weights = mean_weights.to(y.device).view(1,-1)
    else:
        mean_weights = 1.0
    
    loss = 0.5 * (
        logvar + mean_weights * sq_error * torch.exp(-logvar)
    )

    if target_weights is not None: # currently weights both mean and variacnce equally maybe change?
        target_weights = target_weights.to(y.device)
        loss = loss * target_weights

    return loss.mean()


def paper_hetero_loss(y, mu, logvar, phi_idx=2, clamp_min=-10.0, clamp_max=5.0, target_weights=None):
    # this loss kinda sucks ass easily beaten by plain MSE
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
