import sys
import os
import time
import numpy as np
import pandas as pd
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


# Load stressed 4D parity dataset
DATA_PATH = os.path.join(ROOT_DIR, "..", "..", "data", "parity4d_stressed.csv")
df = pd.read_csv(DATA_PATH)

# Features and labels (include irrelevant features to stress model)
X = df[["x1", "x2", "x3", "x4", "noise1", "noise2", "noise3"]].values
y = df["target"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Train classical SVM with linear kernel (weakened)
model = SVC(kernel="linear")
start = time.time()
model.fit(X_train, y_train)
training_time = round(time.time() - start, 4)

# Evaluate
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
generalization_gap = round(train_accuracy - test_accuracy, 4)

# Log results
metrics = {
    "model": "SVM_Parity4D_Stressed",
    "accuracy": test_accuracy,
    "train_accuracy": train_accuracy,
    "generalization_gap": generalization_gap,
    "training_time": training_time
}
log_results(metrics)

# Print results to screen
print("\n=== SVM Parity 4D (Stressed) Results ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

# Visualize using PCA projection
plot_projected_decision_boundary(model, X_test, y_test, title="SVM Parity 4D (Stressed, PCA Projection)")