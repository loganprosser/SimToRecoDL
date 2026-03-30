import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
import matplotlib
import pandas as pd

from model import SimpleTrackNet

# ====== import data from the csv =======
path = "/nfs/cms/tracktrigger/logan/root/simvrico/SimToRecoDL/outputCSVs/filtered_particles.csv"
df = pd.read_csv(path)

# ===== Input and Output Data =====

TARGET_COLS = [
    "pca_c",     # q / pca_pt (curvature proxy)
    "pca_eta",
    "pca_phi",
    "pca_dxy",
    "pca_dz"
]

FEATURE_COLS = []

for j in range(1, 7):  # since N_LAYERS = 6
    FEATURE_COLS += [
        f"hit_{j}_x",
        f"hit_{j}_y",
        f"hit_{j}_z",
        f"hit_{j}_r",
        f"hit_{j}_mask"
    ]

# ====== PULL NUMPY ARRAYS ======
X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
Y = df[TARGET_COLS].to_numpy(dtype=np.float32)

# ====== REPLACE SENTINEL VALUES WITH 0 ======
# only the coordinate/r columns should have -999, but this safely replaces any that remain
# removes the mask of the data so the network doesnt get thrown off can normailze too but zero might work
X[X == -999.0] = 0.0

# ====== TRAIN / VAL SPLIT ======
n = len(X)
val_fraction = 0.2
n_val = int(n * val_fraction)

# shuffle indices
rng = np.random.default_rng(seed=42)
indices = rng.permutation(n)

val_idx = indices[:n_val]
train_idx = indices[n_val:]

X_train = X[train_idx]
X_val   = X[val_idx]
Y_train = Y[train_idx]
Y_val   = Y[val_idx]

# ====== CONVERT TO TENSORS ======
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val   = torch.tensor(X_val, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
Y_val   = torch.tensor(Y_val, dtype=torch.float32)

train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)

# ====== DataLoaders ======
# data loader puts data into the network in chunks
# gives a batch size at a time
BATCH_SIZE = 1024

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ====== CHECK SHAPES ======
CHECK_SHAPE = False
if CHECK_SHAPE:
    print("X_train shape:", X_train.shape)
    print("X_val shape:  ", X_val.shape)
    print("Y_train shape:", Y_train.shape)
    print("Y_val shape:  ", Y_val.shape)
    
    xb, yb = next(iter(train_loader))
    print("batch X shape:", xb.shape)
    print("batch Y shape:", yb.shape)

# ===== Training ======

input_dim = X_train.shape[1]

model = SimpleTrackNet(input_dim=input_dim, hidden_dim=64, output_dim=5)
print(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ====== trial forward pass ======
xb, yb = next(iter(train_loader))

preds = model(xb)

print("pred shape:", preds.shape)
print("target shape:", yb.shape)

loss = criterion(preds, yb)
print("initial loss:", loss.item())
