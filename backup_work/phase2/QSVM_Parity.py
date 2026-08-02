# QSVM.py
# -----------------------------------------------
# General QSVM benchmark script
# Dataset choice is controlled by config/config.yaml
# Options: breast_cancer, parity4d, parity4d_stressed, parity6d, parity6d_stressed
# -----------------------------------------------

import sys
import os
import time
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from datetime import datetime, timezone

from qiskit_machine_learning.algorithms import QSVC

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

# Fixed 50/50 split for benchmarks
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# -------------------------------------------------
# 2. Setup QSVM (canonical v0.8.4 usage)
# -------------------------------------------------
qsvc = QSVC()

# -------------------------------------------------
# 3. Train QSVM
# -------------------------------------------------
start = time.time()
print("before qsvc.fit")
qsvc.fit(X_train, y_train)
training_time = round(time.time() - start, 4)

# -------------------------------------------------
# 4. Evaluate
# -------------------------------------------------
print("before qsvc.predict(X_train)")
y_train_pred = qsvc.predict(X_train)
y_test_pred = qsvc.predict(X_test)
print("after qsvc.predict(X_train)")

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
generalization_gap = round(train_accuracy - test_accuracy, 4)

metrics = {
    "model": "QSVM",
    "dataset": cfg["dataset"],
    "accuracy": test_accuracy,
    "train_accuracy": train_accuracy,
    "generalization_gap": generalization_gap,
    "training_time": training_time
}
log_results(metrics)

print("\n=== QSVM Results ===")
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
    title=f"QSVM ({cfg['dataset']} PCA Projection)",
    filename=plot_filename
)