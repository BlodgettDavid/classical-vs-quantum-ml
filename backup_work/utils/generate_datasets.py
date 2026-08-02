# src/utils/generate_datasets.py
# -------------------------------------------------
# Dataset generation script for benchmarks.
# Produces breast cancer and parity datasets (4D, 6D, clean + stressed).
# -------------------------------------------------

import pandas as pd
import numpy as np
import itertools
from sklearn.datasets import load_breast_cancer
import os

# -------------------------------------------------
# Resolve absolute path to the data folder
# -------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------------------------
# 1. Breast cancer dataset
# -------------------------------------------------
def save_breast_cancer(noise_scale=0.1):
    """
    Save the breast cancer dataset.
    Optionally adds Gaussian noise to features if noise_scale > 0.
    """
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    if noise_scale is not None and noise_scale > 0:
        features = df.drop(columns=["target"]).astype(float)
        noisy = features + noise_scale * np.random.randn(*features.shape)
        df = pd.DataFrame(noisy, columns=data.feature_names)
        df["target"] = data.target

    filename = "breast_cancer.csv"
    csv_path = os.path.join(DATA_DIR, filename)
    df.to_csv(csv_path, index=False)
    print(f"[generate_datasets] wrote {filename} to: {csv_path}")

# -------------------------------------------------
# 2. Parity dataset generator
# -------------------------------------------------
def save_parity(n_bits, stressed=False, noise_scale=0.1):
    """
    Generate an n-bit parity dataset.
    Each row is a binary vector of length n_bits with target = sum(x) % 2.
    If stressed=True, add Gaussian noise to features (scaled by noise_scale).
    """
    data = []
    for bits in itertools.product([0, 1], repeat=n_bits):
        x = list(bits)
        y = sum(x) % 2
        data.append(x + [y])

    df = pd.DataFrame(data, columns=[f"x{i+1}" for i in range(n_bits)] + ["target"])

    if stressed:
        features = df.drop(columns=["target"]).astype(float)
        noisy = features + noise_scale * np.random.randn(*features.shape)
        df = pd.DataFrame(noisy, columns=[f"x{i+1}" for i in range(n_bits)])
        df["target"] = [sum(bits) % 2 for bits in itertools.product([0, 1], repeat=n_bits)]

    filename = f"parity{n_bits}d{'_stressed' if stressed else ''}.csv"
    csv_path = os.path.join(DATA_DIR, filename)
    df.to_csv(csv_path, index=False)
    print(f"[generate_datasets] wrote {filename} to: {csv_path}")

# -------------------------------------------------
# Main execution
# -------------------------------------------------
if __name__ == "__main__":
    save_breast_cancer()  # default noise_scale=0.1
    save_parity(4, stressed=True)   # default noise_scale=0.1
    save_parity(6, stressed=True)
    save_parity(4, stressed=False)
    save_parity(6, stressed=False)
