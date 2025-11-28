# SVM_BreastCancer_PCA.py
# Phase 1 (Exploratory): Classical SVM benchmark on PCA-reduced breast cancer dataset

import sys, os, time
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Ensure src/ is in sys.path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(ROOT_DIR, "..", "..", "src")
sys.path.append(os.path.abspath(SRC_PATH))

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
# 2. Train/test split + scaling
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 3. PCA reduction
# -------------------------------
pca = PCA(n_components=4)  # reduce to 4 principal components
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# -------------------------------
# 4. Train classical SVM
# -------------------------------
clf = SVC(kernel="linear")

start = time.time()
clf.fit(X_train, y_train)
training_time = round(time.time() - start, 4)

# -------------------------------
# 5. Evaluate
# -------------------------------
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
generalization_gap = round(train_accuracy - test_accuracy, 4)

metrics = {
    "model": "SVM_BreastCancer_PCA",
    "dataset": cfg["dataset"],   # use dataset name from config
    "accuracy": test_accuracy,
    "train_accuracy": train_accuracy,
    "generalization_gap": generalization_gap,
    "training_time": training_time
}
log_results(metrics)

print("\n=== SVM Breast Cancer PCA Results ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

plot_projected_decision_boundary(clf, X_test, y_test, title="SVM Breast Cancer (PCA Projection)")
