import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import random

from model import HeteroTrackNet
from loss import paper_hetero_loss

# ====== Running Flags =======
CHECK_SHAPE = False
TEST_TRAIN = False
TRAIN = True


# ===== Picking Device ========
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Device set to {device}")


# ==== Setting Seed =====
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ====== import data from the csv =======
path = "/nfs/cms/tracktrigger/logan/root/simvrico/SimToRecoDL/outputCSVs/filtered_particles.csv"
df = pd.read_csv(path)


# ===== Input and Output Data =====
TARGET_COLS = [
    "pca_c",
    "pca_eta",
    "pca_phi",
    "pca_dxy",
    "pca_dz"
]

FEATURE_COLS = []

for j in range(1, 7):
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
X[X == -999.0] = 0.0


# ====== TRAIN / VAL SPLIT ======
n = len(X)
val_fraction = 0.2
n_val = int(n * val_fraction)

rng = np.random.default_rng(seed=SEED)
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
BATCH_SIZE = 256

g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, generator=g)


# ====== CHECK SHAPES ======
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

model = HeteroTrackNet(
    input_dim=input_dim,
    hidden_layers=[256, 256, 64],
    output_dim=5,
    use_batchnorm=True,
    dropout=0.00,
    activation=nn.ReLU
)
model.to(device)

print(model)

optimizer = optim.Adam(model.parameters(), lr=1e-3)


# ====== trial forward pass ======
if TEST_TRAIN:
    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)

    mu, logvar = model(xb)

    print("mu shape:", mu.shape)
    print("logvar shape:", logvar.shape)
    print("target shape:", yb.shape)

    loss = paper_hetero_loss(yb, mu, logvar)
    print("initial loss:", loss.item())


# ===== Training loop =====
EPOCHS = 80

if TRAIN:
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()

            mu, logvar = model(xb)
            loss = paper_hetero_loss(yb, mu, logvar)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)

                mu, logvar = model(xb)
                loss = paper_hetero_loss(yb, mu, logvar)

                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"EPOCH {epoch + 1:2d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")