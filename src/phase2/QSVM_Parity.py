# QSVM_Parity.py
# Phase 2: Quantum SVM benchmark on parity datasets
# -------------------------------------------------
# This script benchmarks a Quantum SVM (QSVC) on parity datasets.
# In qiskit-machine-learning v0.8.4, QSVC builds its own kernel internally,
# so we instantiate QSVC() directly without QuantumKernel.

import sys
import os
import time
import numpy as np
from qiskit_machine_learning.algorithms import QSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from datetime import datetime, timezone

# -------------------------------------------------
# Ensure src/ is in sys.path for root-level execution
# -------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(ROOT_DIR, "..", "..", "src")
sys.path.append(os.path.abspath(SRC_PATH))

# Modular imports
from utils.logger import log_results
from utils.visualizer import plot_projected_decision_boundary
from utils.data_loader import load_dataset_from_config   # NEW unified loader

# -------------------------------------------------
# 1. Load dataset (via config)
# -------------------------------------------------
df, cfg = load_dataset_from_config()

X = df.drop(columns=["target"]).values
y = df["target"].values

# Fixed 50/50 split for parity experiments
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# -------------------------------------------------
# 2. Setup QSVM (simplified for v0.8.4)
# -------------------------------------------------
# Backend and shots are not directly configurable in this version.
# QSVC builds its own kernel internally.
qsvc = QSVC()

# -------------------------------------------------
# 3. Train QSVM
# -------------------------------------------------
start = time.time()
qsvc.fit(X_train, y_train)
training_time = round(time.time() - start, 4)

# -------------------------------------------------
# 4. Evaluate
# -------------------------------------------------
y_train_pred = qsvc.predict(X_train)
y_test_pred = qsvc.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
generalization_gap = round(train_accuracy - test_accuracy, 4)

metrics = {
    "model": "QSVM_Parity",
    "dataset": cfg["dataset"],   # use dataset name from config
    "accuracy": test_accuracy,
    "train_accuracy": train_accuracy,
    "generalization_gap": generalization_gap,
    "training_time": training_time
}
log_results(metrics)

print("\n=== QSVM Parity Results ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

# -------------------------------------------------
# 5. Visualize decision boundary
# -------------------------------------------------
timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
plot_filename = f"{metrics['model'].lower()}_{metrics['dataset']}_pca_projection_{timestamp}.png"

plot_projected_decision_boundary(
    qsvc,
    X_test,
    y_test,
    title="QSVM Parity (PCA Projection)",
    filename=plot_filename
)