import torch
import numpy
import matplotlib
import pandas as pd

# ====== import data from the csv =======
path = "/nfs/cms/tracktrigger/logan/root/simvrico/SimToRecoDL/outputCSVs/filtered_particles.csv"
df = pd.read(path)

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

# ==== Set X and Y values =====
X = df[FEATURE_COLS].values
Y = df[TARGET_COLS].values

print(X.shape)  # (N, 30)
print(Y.shape)  # (N, 5)




