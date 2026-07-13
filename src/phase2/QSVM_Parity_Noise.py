# QSVM_Parity_Noise.py
# Phase 2: Quantum SVM benchmark on parity datasets with synthetic noise
# -------------------------------------------------
# This script benchmarks a Quantum SVM (QSVC) on parity datasets,
# but runs with a synthetic depolarizing noise model applied.

import sys
import os
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from datetime import datetime, timezone

# Qiskit imports
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel

# -------------------------------------------------
# Ensure src/ is in sys.path for root-level execution
# -------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(ROOT_DIR, "..", "..", "src")
sys.path.append(os.path.abspath(SRC_PATH))

# Modular imports
from utils.logger import log_results
from utils.visualizer import plot_projected_decision_boundary
from utils.data_loader import load_dataset_from_config

# -------------------------------------------------
# 1. Load dataset (via config)
# -------------------------------------------------
df, cfg = load_dataset_from_config()

X = df.drop(columns=["target"]).values
y = df["target"].values

# Fixed 50/50 split for parity experiments
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# -------------------------------------------------
# 2. Build synthetic noise model
# -------------------------------------------------
noise_model = NoiseModel()
error_1q = depolarizing_error(0.01, 1)   # 1% error on single-qubit gates
error_2q = depolarizing_error(0.02, 2)   # 2% error on two-qubit gates

noise_model.add_all_qubit_quantum_error(error_1q, ['u1','u2','u3'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

simulator = AerSimulator(noise_model=noise_model)

# -------------------------------------------------
# 3. Setup QSVM with noisy kernel
# -------------------------------------------------
feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2, entanglement="linear")
quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=simulator)

qsvc = SVC(kernel=quantum_kernel.evaluate)

# -------------------------------------------------
# 4. Train QSVM (noisy)
# -------------------------------------------------
start = time.time()
qsvc.fit(X_train, y_train)
training_time = round(time.time() - start, 4)

# -------------------------------------------------
# 5. Evaluate
# -------------------------------------------------
y_train_pred = qsvc.predict(X_train)
y_test_pred = qsvc.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
generalization_gap = round(train_accuracy - test_accuracy, 4)

metrics = {
    "model": "QSVM_Parity_Noise",
    "dataset": cfg["dataset"],
    "accuracy": test_accuracy,
    "train_accuracy": train_accuracy,
    "generalization_gap": generalization_gap,
    "training_time": training_time
}
log_results(metrics)

print("\n=== QSVM Parity (Noisy) Results ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

# -------------------------------------------------
# 6. Visualize decision boundary
# -------------------------------------------------
timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
plot_filename = f"{metrics['model'].lower()}_{metrics['dataset']}_pca_projection_{timestamp}.png"

plot_projected_decision_boundary(
    qsvc,
    X_test,
    y_test,
    title="QSVM Parity with Synthetic Noise (PCA Projection)",
    filename=plot_filename
)
