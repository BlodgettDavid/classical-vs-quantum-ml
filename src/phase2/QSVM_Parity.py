# QSVM_Parity.py
# Phase 2: Quantum SVM benchmark on parity dataset (ZZ feature map, reps=2, linear entanglement, no PCA)

import sys, os
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime, timezone
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

# Ensure src/ is in sys.path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(ROOT_DIR, "..", "..", "src")
sys.path.append(os.path.abspath(SRC_PATH))

from utils.logger import log_results
from utils.quantum_visualizer import plot_qsvm_decision_boundary, plot_quantum_kernel_matrix
from utils.data_loader import load_dataset_from_config
from utils.quantum_evaluator import evaluate_quantum_model

# -------------------------------
# 1. Load dataset + config
# -------------------------------
df, cfg = load_dataset_from_config()
X = df.drop(columns=["target"]).values
y = df["target"].values

split_ratio = cfg["global"]["split_ratio"]
random_state = cfg["global"]["random_state"]

# -------------------------------
# 2. Train/test split (config-driven)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=split_ratio, random_state=random_state
)

# -------------------------------
# 3. Setup QSVM with config-driven feature map
# -------------------------------
feature_map_name = cfg["qsvm"]["feature_map"]
if feature_map_name == "ZZFeatureMap":
    feature_map = ZZFeatureMap(
        feature_dimension=X.shape[1],
        reps=cfg["qsvm"]["reps"],
        entanglement=cfg["qsvm"]["entanglement"]
    )
elif feature_map_name == "PauliFeatureMap":
    feature_map = PauliFeatureMap(
        feature_dimension=X.shape[1],
        reps=cfg["qsvm"]["reps"],
        entanglement=cfg["qsvm"]["entanglement"]
    )
else:
    raise ValueError(f"Unsupported feature map: {feature_map_name}")

quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
qsvc = QSVC(quantum_kernel=quantum_kernel)

# -------------------------------
# 4. Train + evaluate QSVM
# -------------------------------
metrics = evaluate_quantum_model(
    qsvc, X_train, y_train, X_test, y_test,
    label="QSVM_Parity",
    feature_map=feature_map,
    backend=cfg["qsvm"]["backend"]
)


metrics["dataset"] = cfg["dataset"]
metrics["split_ratio"] = split_ratio
metrics["random_state"] = random_state

# Quantum-specific fields
metrics["feature_map"] = feature_map_name
metrics["reps"] = cfg["qsvm"]["reps"]
metrics["entanglement"] = cfg["qsvm"]["entanglement"]
metrics["backend"] = cfg["qsvm"]["backend"]

# PCA is not used in parity datasets
metrics["pca_components"] = 0

log_results(metrics)

print("\n=== QSVM Parity Results ===")
for k, v in metrics.items():
    print(f"{k}: {v}")


# -------------------------------
# 5a. Visualize decision boundary
# -------------------------------

timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

plot_filename = f"{metrics['model'].lower()}_{metrics['dataset']}_{feature_map_name.lower()}_{timestamp}.png"
plot_qsvm_decision_boundary(
    qsvc, 
    X_test, 
    y_test,
    title=f"QSVM Parity ({metrics['dataset']}, {feature_map_name})",
    save=True,      # always save plots for reproducibility
    show=False,      # disable interactive popup for batch runs
    filename=plot_filename,
    do_pca=True   # Parity dataset needs PCA in visualization
)


# -------------------------------
# 5b. Visualize quantum kernel matrix
# -------------------------------
subset_size = min(len(X_test), max(10, int(0.25 * len(X_test))))
subset = X_test[:subset_size]

kernel_matrix = quantum_kernel.evaluate(subset)
kernel_filename = f"{metrics['model'].lower()}_{metrics['dataset']}_kernel_{timestamp}.png"

plot_quantum_kernel_matrix(
    kernel_matrix,
    title=f"Quantum Kernel Matrix ({metrics['dataset']}, {feature_map_name}, {subset_size} samples)",
    filename=kernel_filename,
    save=True,      # always save for reproducibility
    show=False      # disable interactive popup for batch runs
)