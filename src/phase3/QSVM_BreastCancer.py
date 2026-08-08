# QSVM_BreastCancer.py
# Phase 3: Quantum SVM benchmark on breast cancer dataset

import sys, os, time
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 0. Setup paths + imports
# -------------------------------
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(ROOT_DIR, "..", "..", "src")
sys.path.append(os.path.abspath(SRC_PATH))

from utils.logger import log_results
from utils.quantum_visualizer import plot_qsvm_decision_boundary, plot_quantum_kernel_matrix

# Qiskit imports
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

# -------------------------------
# 1. Configurable parameters
# -------------------------------
feature_map_type = "ZZ"   # options: "ZZ", "Pauli"
feature_map_reps = 1      # number of repetitions in feature map

# -------------------------------
# 2. Load dataset (direct CSV read)
# -------------------------------
DATA_DIR = os.path.join(ROOT_DIR, "..", "..", "data")
df = pd.read_csv(os.path.join(DATA_DIR, "breast_cancer.csv"))

X = df.drop(columns=["target"]).values
y = df["target"].values

# -------------------------------
# 3. Train/test split + scaling
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 4. Setup QSVM
# -------------------------------
if feature_map_type == "ZZ":
    feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=feature_map_reps)
elif feature_map_type == "Pauli":
    feature_map = PauliFeatureMap(feature_dimension=X_train.shape[1], reps=feature_map_reps)
else:
    raise ValueError(f"Unsupported feature_map_type: {feature_map_type}")

quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
qsvc = QSVC(quantum_kernel=quantum_kernel)

# -------------------------------
# 5. Train QSVM
# -------------------------------
start = time.time()
try:
    qsvc.fit(X_train, y_train)
    training_time = round(time.time() - start, 4)
except Exception as e:
    print(f"QSVM training failed: {e}")
    training_time = None

# -------------------------------
# 6. Evaluate
# -------------------------------
if training_time is not None:
    y_train_pred = qsvc.predict(X_train)
    y_test_pred = qsvc.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    generalization_gap = round(train_accuracy - test_accuracy, 4)
else:
    train_accuracy = test_accuracy = generalization_gap = None

timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")

metrics = {
    "model": "QSVM_BreastCancer",
    "dataset": "breast_cancer",
    "kernel": f"{feature_map_type}FeatureMap+FidelityQuantumKernel",
    "accuracy": test_accuracy,
    "train_accuracy": train_accuracy,
    "generalization_gap": generalization_gap,
    "training_time": training_time,
    "timestamp": timestamp
}
log_results(metrics)

print("\n=== QSVM Breast Cancer Results ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

# -------------------------------
# 7. Visualize
# -------------------------------
# Decision boundary
plot_filename = f"QSVM_BreastCancer_{feature_map_type}_{timestamp}.png"
plot_path = os.path.join(ROOT_DIR, "..", "..", "plots", plot_filename)

plot_qsvm_decision_boundary(
    qsvc,
    X_test,
    y_test,
    title=f"QSVM Breast Cancer ({feature_map_type} FeatureMap, PCA Projection)",
    filename=plot_path
)

# Kernel matrix heatmap
kernel_matrix = quantum_kernel.evaluate(X_test)
plot_quantum_kernel_matrix(
    kernel_matrix,
    title=f"Quantum Kernel Matrix ({feature_map_type} FeatureMap)",
    filename=f"QSVM_BreastCancer_{feature_map_type}_KernelMatrix_{timestamp}.png"
)