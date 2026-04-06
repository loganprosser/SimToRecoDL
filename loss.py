import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib
import pandas as pd

import torch
import torch.nn as nn


def hetero_loss(y, mu, logvar):
    # mean prediction loss
    mse = (y - mu) ** 2

    # variance term (can be off by the scalars doesnt matter for optimizaion)
    var_loss = logvar + ((y.detach() - mu.detach()) ** 2) * torch.exp(-logvar)

    # combine both parts
    return mse.mean() + 0.5 * var_loss.mean()

def actual_herto_loss(y, mu, logvar):
    logvar = torch.clamp(logvar, min=-5, max=5)
    return (logvar + (y - mu)**2 * torch.exp(-logvar)).mean()