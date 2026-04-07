import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib
import pandas as pd

import torch
import torch.nn as nn

#TODO impliment and test recurrent neural network (GRU vs LSTM plain RNN?)
#TODO implement and test transformer based NN 


class HeteroTrackNet(nn.Module):
    # TODO impliment L1 and L2 regularization, stopping early, 
    # learns mu and sigma assuming a diagonal variance matrix (no covariance)
    def __init__(
        self,
        input_dim,
        hidden_layers=None,
        output_dim=5,
        activation=nn.ReLU,
        use_batchnorm=False,
        dropout=0.0
    ):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [64, 64]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(activation())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        self.mu_head = nn.Linear(prev_dim, output_dim)
        self.logvar_head = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        features = self.backbone(x)
        mu = self.mu_head(features)
        logvar = self.logvar_head(features) #added detach() to go with paper???
        return mu, logvar

class SimpleTrackNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_layers=None,
        output_dim=5,
        activation=nn.ReLU,
        use_batchnorm=False,
        dropout=0.0
    ):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [64, 64]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(activation())

            if dropout > 0:
                layers.append(nn.Dropout(dropout)) # dropout and batchnorm automatically hav ethis built in so we dont nee to worry about wriiting stuff for eval vs train pytroch does it all internally

            prev_dim = hidden_dim

        # final output layer (no activation, no BN, no dropout)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SimpleTrackNetNOBNDROP(nn.Module):
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
    
