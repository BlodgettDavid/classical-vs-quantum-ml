import sys
import os
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Ensure src/ is in sys.path for root-level execution
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(ROOT_DIR, "..", "..", "src")
sys.path.append(os.path.abspath(SRC_PATH))

from utils.logger import log_results
from utils.visualizer import plot_decision_boundary

# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 1, 1, 0])

# Train classical SVM with RBF kernel
model = SVC(kernel="rbf", gamma="auto")
start = time.time()
model.fit(X, y)
training_time = round(time.time() - start, 4)

# Evaluate
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
train_accuracy = accuracy  # No test split for XOR
generalization_gap = 0.0

# Log results
metrics = {
    "model": "SVM_XOR",
    "accuracy": accuracy,
    "train_accuracy": train_accuracy,
    "generalization_gap": generalization_gap,
    "training_time": training_time
}
log_results(metrics)

# Visualize
plot_decision_boundary(model, X, y, title="SVM XOR Decision Boundary")
