# QSVM_Parity.py
# Phase 2: Quantum SVM benchmark on parity dataset (ZZ feature map, reps=1, no PCA)

import sys, os, time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from datetime import datetime, timezone
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

# Ensure src/ is in sys.path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(ROOT_DIR, "..", "..", "src")
sys.path.append(os.path.abspath(SRC_PATH))

from utils.logger import log_results
from utils.quantum_visualizer import plot_qsvm_decision_boundary
from utils.data_loader import load_dataset_from_config

# 1. Load dataset (via config)
df, cfg = load_dataset_from_config()
X = df.drop(columns=["target"]).values
y = df["target"].values

# 2. Train/test split (50/50 for parity)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 3. Setup QSVM with explicit ZZ feature map + FidelityQuantumKernel
feature_map = ZZFeatureMap(feature_dimension=X.shape[1], reps=1)
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
qsvc = QSVC(quantum_kernel=quantum_kernel)

# 4. Train QSVM
start = time.time()
qsvc.fit(X_train, y_train)
training_time = round(time.time() - start, 4)

# 5. Evaluate
y_train_pred = qsvc.predict(X_train)
y_test_pred = qsvc.predict(X_test)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
generalization_gap = round(train_accuracy - test_accuracy, 4)

metrics = {
    "model": "QSVM_Parity",
    "dataset": cfg["dataset"],
    "kernel": "ZZFeatureMap",
    "accuracy": test_accuracy,
    "train_accuracy": train_accuracy,
    "generalization_gap": generalization_gap,
    "training_time": training_time
}
log_results(metrics)

print("\n=== QSVM Parity Results ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

# 6. Visualize decision boundary
timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
plot_filename = f"{metrics['model'].lower()}_{metrics['dataset']}_zz_{timestamp}.png"
plot_qsvm_decision_boundary(
    qsvc, X_test, y_test,
    title=f"QSVM Parity (ZZ Feature Map, PCA Projection)",
    filename=plot_filename
)
