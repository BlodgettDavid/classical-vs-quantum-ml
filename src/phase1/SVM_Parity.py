# SVM_Parity.py
# Phase 1: Classical SVM benchmark on parity dataset

import sys
import os
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Ensure src/ is in sys.path for root-level execution
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(ROOT_DIR, "..", "..", "src")
sys.path.append(os.path.abspath(SRC_PATH))

# Modular imports
from utils.logger import log_results
from utils.visualizer import plot_projected_decision_boundary
from utils.data_loader import load_dataset_from_config   # NEW unified loader

# -------------------------------
# 1. Load dataset (via config)
# -------------------------------
df, cfg = load_dataset_from_config()

X = df.drop(columns=["target"]).values
y = df["target"].values

# -------------------------------
# 2. Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# -------------------------------
# 3. Train classical SVM (kernel from config)
# -------------------------------
kernel = cfg["svm"]["kernel"]
model = SVC(kernel=kernel, C=cfg["svm"]["C"], gamma=cfg["svm"]["gamma"])

start = time.time()
model.fit(X_train, y_train)
training_time = round(time.time() - start, 4)

# -------------------------------
# 4. Evaluate
# -------------------------------
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
generalization_gap = round(train_accuracy - test_accuracy, 4)

metrics = {
    "model": "SVM_Parity",
    "dataset": cfg["dataset"],   # use dataset name from config
    "accuracy": test_accuracy,
    "train_accuracy": train_accuracy,
    "generalization_gap": generalization_gap,
    "training_time": training_time
}
log_results(metrics)

print("\n=== Classical SVM Parity Results ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

# -------------------------------
# 5. Visualize decision boundary
# -------------------------------
plot_projected_decision_boundary(model, X_test, y_test, title="Classical SVM Parity (PCA Projection)")
