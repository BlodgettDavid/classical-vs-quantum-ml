# src/phase1/SVM_Parity.py
# Phase 1: Classical SVM benchmark on Parity Datasets

import sys, os, time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
from sklearn.model_selection import train_test_split
from datetime import datetime, timezone
from itertools import product

# -------------------------------
# Setup paths
# -------------------------------
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(ROOT_DIR, "..", "..", "src")
if os.path.abspath(SRC_PATH) not in sys.path:
    sys.path.append(os.path.abspath(SRC_PATH))

from utils.logger import log_results
from utils.classical_visualizer import (
    plot_confusion_matrix,
    plot_kernel_matrix
)
from utils.classical_evaluator import evaluate_model
from utils.config_loader import load_config

def generate_parity_32_samples(num_bits=4):
    """Generates a balanced 32-sample Parity dataset (2 complete cycles of 2^4 binary vectors)."""
    base_X = np.array(list(product([0, 1], repeat=num_bits)))
    base_y = np.sum(base_X, axis=1) % 2
    X = np.vstack([base_X, base_X])
    y = np.concatenate([base_y, base_y])
    return X, y

def run_svm_parity():
    # -------------------------------
    # 1. Load parameters from config
    # -------------------------------
    cfg = load_config("classical_svm.yaml", dataset_key="parity4d")
    dataset = cfg.get("dataset", "parity4d_stressed")

    split_ratio = cfg.get("split_ratio", 0.25)
    random_state = cfg.get("random_state", 42)

    model_params = cfg.get("model_params", {})
    kernel = model_params.get("kernel", "rbf")
    C = model_params.get("C", 10.0)
    gamma = model_params.get("gamma", "scale")
    degree = model_params.get("degree", 4)

    # -------------------------------
    # 2. Load dataset
    # -------------------------------
    DATA_DIR = os.path.join(ROOT_DIR, "..", "..", "data")
    data_path = os.path.join(DATA_DIR, f"{dataset}.csv")
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        target_col = "target" if "target" in df.columns else df.columns[-1]
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
    else:
        # Fallback generator guaranteeing 32 total samples
        X, y = generate_parity_32_samples(num_bits=4)

    # -------------------------------
    # 3. Train/test split (24 train / 8 test)
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=random_state, stratify=y
    )

    # -------------------------------
    # 4. Instantiate Classical SVM
    # -------------------------------
    if kernel == "poly":
        model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, random_state=random_state)
    else:
        model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=random_state)

    # -------------------------------
    # 5. Evaluate Model
    # -------------------------------
    metrics = evaluate_model(
        model, X_train, y_train, X_test, y_test, label=f"SVM_Parity_{dataset}"
    )

    # Metadata enrichment
    metrics["dataset"] = dataset
    metrics["split_ratio"] = split_ratio
    metrics["random_state"] = random_state
    metrics["kernel"] = kernel
    metrics["C"] = C
    metrics["gamma"] = gamma
    metrics["degree"] = degree if kernel == "poly" else 0
    metrics["pca_components"] = 0  # Raw feature space used
    metrics["n_train_samples"] = len(X_train)
    metrics["n_test_samples"] = len(X_test)

    # Schema alignment for results.csv
    metrics["backend"] = "CPU_Classical"
    metrics["num_qubits"] = 0
    metrics["circuit_depth"] = 0
    metrics["feature_map"] = "N/A"
    metrics["reps"] = 0
    metrics["entanglement"] = "N/A"

    # -------------------------------
    # 6. Visualizations
    # -------------------------------
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    plot_start_time = time.perf_counter()

    # Classical Kernel Matrix Heatmap (NxN)
    if kernel == "poly":
        gram_matrix = polynomial_kernel(X_train, X_train, degree=degree)
    else:
        # Default RBF Kernel scale conversion
        gamma_val = 1.0 / X_train.shape[1] if gamma == "scale" else gamma
        gram_matrix = rbf_kernel(X_train, X_train, gamma=gamma_val)

    kernel_filename = f"svm_{dataset}_{kernel}_kernel_{timestamp}.png"
    plot_kernel_matrix(
        gram_matrix,
        title=f"Classical Kernel Matrix ({dataset.upper()}, {kernel.upper()} kernel)",
        save=True,
        show=False,
        filename=kernel_filename
    )

    # Confusion Matrix (2x2)
    y_pred = model.predict(X_test)
    cm_filename = f"svm_{dataset}_{kernel}_cm_{timestamp}.png"
    plot_confusion_matrix(
        y_test, 
        y_pred,
        title=f"Confusion Matrix ({dataset.upper()}, {kernel.upper()} kernel)",
        save=True,
        show=False,
        filename=cm_filename
    )

    # Record plotting runtime BEFORE logging
    metrics["plotting_runtime"] = round(time.perf_counter() - plot_start_time, 4)

    # -------------------------------
    # 7. Log Results
    # -------------------------------
    log_results(metrics)

    print(f"\n=== Classical SVM Parity Results ({dataset}) ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return metrics

if __name__ == "__main__":
    run_svm_parity()