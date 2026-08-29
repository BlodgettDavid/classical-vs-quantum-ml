# QSVM_BreastCancer_PCA.py
# Phase 3: Quantum SVM benchmark on PCA-reduced breast cancer dataset (config-driven)

import sys, os, time
from datetime import datetime, timezone
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------------
# 0. Setup paths + imports
# -------------------------------
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(ROOT_DIR, "..", "..", "src")
sys.path.append(os.path.abspath(SRC_PATH))

from utils.logger import log_results
from utils.quantum_visualizer import plot_qsvm_decision_boundary, plot_quantum_kernel_matrix
from utils.data_loader import load_dataset_from_config
from utils.quantum_evaluator import evaluate_quantum_model

# Qiskit imports
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

# -------------------------------
# 1. Load dataset + config
# -------------------------------
df, cfg = load_dataset_from_config()
X = df.drop(columns=["target"]).values
y = df["target"].values

split_ratio = cfg["global"]["split_ratio"]
random_state = cfg["global"]["random_state"]
pca_components = cfg["global"]["pca_components"]

# -------------------------------
# 2. Train/test split + scaling
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=split_ratio, random_state=random_state
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 3. PCA reduction
# -------------------------------
pca = PCA(n_components=pca_components, random_state=random_state)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# -------------------------------
# 4. Setup QSVM from config
# -------------------------------
feature_map_name = cfg["qsvm"]["feature_map"]
reps = cfg["qsvm"]["reps"]
entanglement = cfg["qsvm"]["entanglement"]

if feature_map_name == "ZZFeatureMap":
    feature_map = ZZFeatureMap(
        feature_dimension=pca_components,
        reps=reps,
        entanglement=entanglement
    )
elif feature_map_name == "PauliFeatureMap":
    feature_map = PauliFeatureMap(
        feature_dimension=pca_components,
        reps=reps,
        entanglement=entanglement
    )
else:
    raise ValueError(f"Unsupported feature map: {feature_map_name}")

quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
qsvc = QSVC(quantum_kernel=quantum_kernel)

# -------------------------------
# 5. Train + evaluate QSVM
# -------------------------------
metrics = evaluate_quantum_model(
    qsvc, X_train, y_train, X_test, y_test,
    label="QSVM_BreastCancer_PCA",
    feature_map=feature_map,
    backend=cfg["qsvm"]["backend"]
)

# -------------------------------
# 6. Add reproducibility + quantum hyperparameters
# -------------------------------
metrics["dataset"] = cfg["dataset"]
metrics["split_ratio"] = split_ratio
metrics["random_state"] = random_state
metrics["feature_map"] = feature_map_name
metrics["reps"] = reps
metrics["entanglement"] = entanglement
metrics["backend"] = cfg["qsvm"]["backend"]
metrics["pca_components"] = pca_components

log_results(metrics)

print("\n=== QSVM Breast Cancer PCA Results ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

# -------------------------------
# 7. Visualization (optional)
# -------------------------------
timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

# --- Decision Boundary ---
plot_filename = (
    f"{metrics['model'].lower()}_"
    f"{metrics['dataset']}_"
    f"{feature_map_name.lower()}_"
    f"{timestamp}.png"
)

plot_qsvm_decision_boundary(
    qsvc,
    X_test,
    y_test,
    title=f"QSVM Breast Cancer PCA ({pca_components} comps, {feature_map_name})",
    filename=plot_filename,
    do_pca=False,   # PCA already applied in modeling → DO NOT apply again
    save=True,
    show=False
)

# --- Quantum Kernel Matrix ---
subset_size = min(len(X_test), max(10, int(0.25 * len(X_test))))
subset = X_test[:subset_size]
kernel_matrix = quantum_kernel.evaluate(subset)

kernel_filename = (
    f"{metrics['model'].lower()}_"
    f"{metrics['dataset']}_kernel_"
    f"{timestamp}.png"
)

plot_quantum_kernel_matrix(
    kernel_matrix,
    title=f"Quantum Kernel Matrix PCA ({feature_map_name}, {subset_size} samples)",
    filename=kernel_filename,
    save=True,
    show=False
)