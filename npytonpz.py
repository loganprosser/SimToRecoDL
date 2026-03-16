import numpy as np
from pathlib import Path

outdir = Path("track_cache")   # change if needed

X = np.load(outdir / "features.npy", mmap_mode="r")
Y = np.load(outdir / "labels.npy", mmap_mode="r")

np.savez_compressed(
    outdir / "dataset.npz",
    features=X,
    labels=Y
)

print("Created dataset.npz")