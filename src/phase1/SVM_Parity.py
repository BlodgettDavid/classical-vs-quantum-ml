# svm_parity.py
# Phase 1: Classical SVM benchmark on parity dataset

import sys, os
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from datetime import datetime, timezone

# -------------------------------
# Setup paths
# -------------------------------
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(ROOT_DIR, "..", "..", "src")
sys.path.append(os.path.abspath(SRC_PATH))

from utils.logger import log_results
from utils.classical_visualizer import (
    plot_projected_decision_boundary,
    plot_confusion_matrix
)
from utils.config_loader import load_config
from utils.classical_evaluator import evaluate_model

# -------------------------------
# Load dataset + config
# -------------------------------
cfg = load_config()
dataset = cfg["dataset"]

DATA_DIR = os.path.join(ROOT_DIR, "..", "..", "data")
df = pd.read_csv(os.path.join(DATA_DIR, f"{dataset}.csv"))

X = df.drop(columns=["target"]).values
y = df["target"].values

kernel = cfg["svm"]["kernel"]
C = cfg["svm"]["C"]
gamma = cfg["svm"]["gamma"]
degree = cfg["svm"].get("degree", 3)
split_ratio = cfg["global"]["split_ratio"]
random_state = cfg["global"]["random_state"]

# -------------------------------
# Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=split_ratio, random_state=random_state
)

# -------------------------------
# Train classical SVM
# -------------------------------
if kernel == "poly":
    model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
else:
    model = SVC(kernel=kernel, C=C, gamma=gamma)

# -------------------------------
# Evaluate
# -------------------------------
metrics = evaluate_model(model, X_train, y_train, X_test, y_test, label="SVM_Parity")
metrics["dataset"] = dataset
metrics["kernel"] = kernel
metrics["split_ratio"] = split_ratio
metrics["random_state"] = random_state
if kernel == "poly":
    metrics["degree"] = degree

log_results(metrics)

print("\n=== Classical SVM Parity Results ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

# -------------------------------
# Visualize
# -------------------------------
timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
# Decision boundary
boundary_filename = f"{metrics['model'].lower()}_{metrics['dataset']}_{kernel}_{timestamp}.png"

plot_projected_decision_boundary(
    X_test, 
    y_test,
    title=f"Classical SVM Parity ({dataset}, {kernel} kernel)",
    save=True,       # always save plots for reproducibility
    show=False,      # disable interactive popup for batch runs
    filename=boundary_filename,
    surrogate_kernel=kernel,  # mirror the kernel choice
    do_pca=True      # parity dataset needs PCA in visualization
)

# Confusion matrix
y_pred = model.predict(X_test)
cm_filename = f"{metrics['model'].lower()}_{metrics['dataset']}_{kernel}_cm_{timestamp}.png"
plot_confusion_matrix(
    y_test, 
    y_pred,
    title=f"Confusion Matrix ({dataset}, {kernel} kernel)",
    save=True,       # always save plots for reproducibility
    show=False,      # disable interactive popup for batch runs
    filename=cm_filename
)