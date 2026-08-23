# SVM_BreastCancer.py
# Phase 1: Classical SVM benchmark on breast cancer dataset

import sys, os
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timezone

# -------------------------------
# Setup paths
# -------------------------------
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(ROOT_DIR, "..", "..", "src")
sys.path.append(os.path.abspath(SRC_PATH))

from utils.logger import log_results
from utils.classical_visualizer import plot_projected_decision_boundary
from utils.classical_evaluator import evaluate_model
from utils.config_loader import load_config

# -------------------------------
# Load parameters from config
# -------------------------------
cfg = load_config()
dataset = cfg["dataset"]

# Global controls
split_ratio = cfg["global"]["split_ratio"]
random_state = cfg["global"]["random_state"]

# Classical SVM hyperparameters
kernel = cfg["svm"]["kernel"]
C = cfg["svm"]["C"]
gamma = cfg["svm"]["gamma"]
degree = cfg["svm"].get("degree", 3)

# -------------------------------
# Load dataset
# -------------------------------
DATA_DIR = os.path.join(ROOT_DIR, "..", "..", "data")
df = pd.read_csv(os.path.join(DATA_DIR, f"{dataset}.csv"))
X = df.drop(columns=["target"]).values
y = df["target"].values

# -------------------------------
# Train/test split + scaling
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=split_ratio, random_state=random_state
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

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
metrics = evaluate_model(model, X_train, y_train, X_test, y_test, label="SVM_BreastCancer")

# Always log all 8 reproducibility parameters
metrics["dataset"] = dataset
metrics["split_ratio"] = split_ratio
metrics["random_state"] = random_state
metrics["kernel"] = kernel
metrics["C"] = C
metrics["gamma"] = gamma
metrics["degree"] = degree if kernel == "poly" else 0
metrics["pca_components"] = 0  # always 0 for non-PCA scripts

log_results(metrics)

print("\n=== Classical SVM Breast Cancer Results ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

# -------------------------------
# Visualize
# -------------------------------
timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
plot_filename = f"{metrics['model'].lower()}_{metrics['dataset']}_{metrics['kernel']}_{timestamp}.png"
plot_projected_decision_boundary(
    model, X_test, y_test,
    title=f"Classical SVM Breast Cancer ({dataset}, {kernel} kernel)",
    filename=plot_filename
)