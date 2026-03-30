import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib
import pandas as pd


class SimpleTrackNet(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=5):
        super().__init__()
        
        self.net == nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    
