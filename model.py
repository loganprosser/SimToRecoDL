import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib
import pandas as pd

class SimpleTrackNet(nn.Module):
    def __init__(self, input_dim, hidden_layers=None, output_dim=5, activation=nn.ReLU):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [64, 64]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class TestTrackNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=5):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    
