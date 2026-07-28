



# QSVM_BreastCancer.py
# Phase 3: Quantum SVM benchmark on breast cancer dataset

import sys
import os
import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure src/ is in sys.path for root-level execution
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(ROOT_DIR, "..", "..", "src")
sys.path.append(os.path.abspath(SRC_PATH))

# Modular imports
from utils.logger import log_results
from utils.visualizer import plot_projected_decision_boundary
from utils.data_loader import load_dataset_from_config   # unified loader

# Qiskit imports
from qiskit_machine_learning.algorithms import QSVC
#db: added because we are going to specify 
#a kernel to use so that we avoid the
#default SamplerV1 kernel  and hopefully
#things speed up
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel


# -------------------------------
# 1. Load dataset (via config)
# -------------------------------
df, cfg = load_dataset_from_config()

X = df.drop(columns=["target"]).values
y = df["target"].values

# -------------------------------
# 2. Train/test split + scaling
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 3. Setup QSVM
# -------------------------------
# QSVC builds its own kernel internally in v0.8.4

#db: added because we are going to specify 
#a kernel to use so that we avoid the
#default SamplerV1 kernel  and hopefully
#things speed up
feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=1)
print("after feature map")
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
print("after quantum_kernel")
qsvc = QSVC(quantum_kernel=quantum_kernel)
print("after qsvc")
#qsvc = QSVC()

# -------------------------------
# 4. Train QSVM
# -------------------------------
start = time.time()
qsvc.fit(X_train, y_train)
print("after qsvc.fit")
training_time = round(time.time() - start, 4)

# -------------------------------
# 5. Evaluate
# -------------------------------
y_train_pred = qsvc.predict(X_train)
print("after qsvc qsvc.predict y on train data")
y_test_pred = qsvc.predict(X_test)
print("after qsvc qsvc.predict y on test data")

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
generalization_gap = round(train_accuracy - test_accuracy, 4)

metrics = {
    "model": "QSVM_BreastCancer",
    "dataset": cfg["dataset"],   # use dataset name from config
    "accuracy": test_accuracy,
    "train_accuracy": train_accuracy,
    "generalization_gap": generalization_gap,
    "training_time": training_time
}
log_results(metrics)

print("\n=== QSVM Breast Cancer Results ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

# -------------------------------
# 6. Visualize decision boundary
# -------------------------------
plot_projected_decision_boundary(qsvc, X_test, y_test, title="QSVM Breast Cancer (PCA Projection)")
