# SVM_BreastCancer_PCA.py
# Phase 1: Classical SVM benchmark on PCA-reduced breast cancer dataset

import sys, os, time
from datetime import datetime
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
# 3. PCA reduction
# -------------------------------
n_components = 4   # configurable later
pca = PCA(n_components=n_components, random_state=42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# -------------------------------
# 4. Train classical SVM
# -------------------------------
kernel = "sigmoid"   # change manually: "linear", "rbf", "poly", "sigmoid"
clf = SVC(kernel=kernel, C=1.0, gamma="scale", degree=3)

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

timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")

metrics = {
    "model": f"SVM_BreastCancer_PCA_{n_components}c",
    "dataset": "breast_cancer",
    "kernel": kernel,
    "accuracy": test_accuracy,
    "train_accuracy": train_accuracy,
    "generalization_gap": generalization_gap,
    "training_time": training_time,
    "timestamp": timestamp
}

# -------------------------------
# 5b. Log + print results
# -------------------------------
log_results(metrics)

print(f"\n=== {metrics['model']} Results ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

# -------------------------------
# 6. Visualize
# -------------------------------
plot_filename = f"SVM_BreastCancer_PCA_{kernel}_{timestamp}.png"
plot_path = os.path.join(ROOT_DIR, "..", "..", "plots", plot_filename)

plot_projected_decision_boundary(
    clf,
    X_test,
    y_test,
    title=f"Classical SVM Breast Cancer (PCA {n_components} comps, {kernel} kernel)",
    filename=plot_path,
    surrogate_kernel=kernel   # consistent with classical_visualizer API
)
