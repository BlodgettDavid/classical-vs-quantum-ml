# SVM_BreastCancer.py
# Phase 1: Classical SVM benchmark on breast cancer dataset

import sys
import os
import time
from datetime import datetime
import pandas as pd
from sklearn.svm import SVC
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
from utils.classical_visualizer import plot_projected_decision_boundary

# -------------------------------
# 1. Load dataset (direct CSV read)
# -------------------------------
DATA_DIR = os.path.join(ROOT_DIR, "..", "..", "data")
df = pd.read_csv(os.path.join(DATA_DIR, "breast_cancer.csv"))

X = df.drop(columns=["target"]).values
y = df["target"].values

# -------------------------------
# 2. Train/test split + scaling
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 3. Train classical SVM
# -------------------------------
kernel = "sigmoid"   # change here if you want rbf, poly, etc.
model = SVC(kernel=kernel)

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

# Generate timestamp for logging + plots
timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")

metrics = {
    "model": "SVM_BreastCancer",
    "dataset": "breast_cancer",
    "kernel": kernel,
    "accuracy": test_accuracy,
    "train_accuracy": train_accuracy,
    "generalization_gap": generalization_gap,
    "training_time": training_time,
    "timestamp": timestamp
}

# -------------------------------
# 4b. Log + print results
# -------------------------------
log_results(metrics)

print("\n=== Classical SVM Breast Cancer Results ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

# -------------------------------
# 5. Visualize decision boundary
# -------------------------------
plot_filename = f"SVM_BreastCancer_{kernel}_{timestamp}.png"
plot_path = os.path.join(ROOT_DIR, "..", "..", "plots", plot_filename)

plot_projected_decision_boundary(
    model,
    X_test,
    y_test,
    title=f"Classical SVM Breast Cancer ({kernel} kernel, PCA Projection)",
    filename=plot_path,
    surrogate_kernel=kernel   # pass kernel explicitly
)