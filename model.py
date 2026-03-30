import torch
import numpy as np
import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split


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
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# ====== CONVERT TO TENSORS ======
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val   = torch.tensor(X_val, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
Y_val   = torch.tensor(Y_val, dtype=torch.float32)

# ====== CHECK SHAPES ======
print("X_train shape:", X_train.shape)
print("X_val shape:  ", X_val.shape)
print("Y_train shape:", Y_train.shape)
print("Y_val shape:  ", Y_val.shape)

