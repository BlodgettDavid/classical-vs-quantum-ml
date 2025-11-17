# src/config/generate_datasets.py

import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification
import os

# Resolve absolute path to the data folder
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)

def save_breast_cancer():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    df.to_csv(os.path.join(DATA_DIR, "breast_cancer.csv"), index=False)
    print("Saved: breast_cancer.csv")

def save_xor():
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1,
                               class_sep=2.0, random_state=42)
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["target"] = y
    df.to_csv(os.path.join(DATA_DIR, "xor_dataset.csv"), index=False)
    print("Saved: xor_dataset.csv")

def save_parity():
    data = []
    for a in [0, 1]:
        for b in [0, 1]:
            for c in [0, 1]:
                x = [a, b, c]
                y = sum(x) % 2  # parity: even=0, odd=1
                data.append(x + [y])
    df = pd.DataFrame(data, columns=["x1", "x2", "x3", "target"])
    df.to_csv(os.path.join(DATA_DIR, "parity_dataset.csv"), index=False)
    print("Saved: parity_dataset.csv")

if __name__ == "__main__":
    save_breast_cancer()
    save_xor()
    save_parity()
