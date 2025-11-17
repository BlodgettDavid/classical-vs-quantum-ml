# src/phase1/SVM_BreastCancer.py

import sys
import os

# Ensure src/ is in sys.path for root-level execution
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(ROOT_DIR, "..", "..", "src")
sys.path.append(os.path.abspath(SRC_PATH))

import yaml
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils.data_loader import load_dataset
from utils.evaluator import evaluate_model
from utils.logger import log_results

# Load config.yaml
CONFIG_PATH = os.path.join(SRC_PATH, "config", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Extract config values
dataset_name = config["dataset"]
test_size = config["test_size"]
random_state = config["random_state"]
svm_params = config["svm"]

# Load and prepare data
df = load_dataset(dataset_name)
X = df.drop("target", axis=1).values
y = df["target"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, random_state=random_state
)

# Train classical SVM
svm_model = SVC(kernel=svm_params["kernel"], C=svm_params["C"], gamma=svm_params["gamma"])
metrics = evaluate_model(svm_model, X_train, y_train, X_test, y_test, label="SVM Breast Cancer")

# Log results
log_results(metrics)