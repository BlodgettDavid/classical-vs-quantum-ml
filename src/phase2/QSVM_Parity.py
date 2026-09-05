# src/phase2/QSVM_Parity.py
# Phase 2: Quantum QSVM benchmark on Parity Datasets

import os
import sys
import yaml
import time
import psutil
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel, FidelityQuantumKernel
from qiskit_aer import AerSimulator

# Setup repository paths
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(ROOT_DIR, "..")
if os.path.abspath(SRC_PATH) not in sys.path:
    sys.path.append(os.path.abspath(SRC_PATH))

from utils.logger import log_results
from utils.quantum_visualizer import plot_quantum_kernel_matrix


def load_config(config_filename="quantum_svm.yaml", dataset_key="parity4d"):
    project_root = os.path.abspath(os.path.join(ROOT_DIR, "..", ".."))
    config_path = os.path.join(project_root, "config", config_filename)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config["datasets"][dataset_key]


def generate_parity_32_samples(num_bits=4):
    """Generates a balanced 32-sample Parity dataset matching Phase 1's size."""
    base_X = np.array(list(product([0, 1], repeat=num_bits)))
    base_y = np.sum(base_X, axis=1) % 2
    X = np.vstack([base_X, base_X])
    y = np.concatenate([base_y, base_y])
    return X, y


def build_feature_map(fm_config, num_qubits):
    name = fm_config["name"]
    reps = fm_config.get("reps", 2)
    entanglement = fm_config.get("entanglement", "full")

    if name == "ZZFeatureMap":
        return ZZFeatureMap(feature_dimension=num_qubits, reps=reps, entanglement=entanglement)
    elif name == "PauliFeatureMap":
        paulis = fm_config.get("paulis", ["Z", "ZZ"])
        return PauliFeatureMap(feature_dimension=num_qubits, reps=reps, entanglement=entanglement, paulis=paulis)
    else:
        raise ValueError(f"Unsupported feature map: {name}")


def get_quantum_kernel(feature_map, backend_config):
    backend_type = backend_config.get("type", "statevector")
    shots = backend_config.get("shots", None)

    if backend_type == "statevector" or shots is None:
        return FidelityStatevectorKernel(feature_map=feature_map)
    else:
        backend = AerSimulator()
        return FidelityQuantumKernel(feature_map=feature_map, fidelity_shots=shots)


def run_parity_benchmark():
    cfg = load_config("quantum_svm.yaml", "parity4d")

    dataset_base = cfg.get("dataset_name", "parity4d_stressed")
    split_ratio = cfg.get("split_ratio", 0.25)
    random_state = cfg.get("random_state", 42)
    c_val = cfg["svm_params"]["c_val"]
    feature_map_configs = cfg["quantum_params"]["feature_maps"]
    backend_config = cfg["backend_config"]

    num_bits = 4
    DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, "..", "..", "data"))
    data_path = os.path.join(DATA_DIR, f"{dataset_base}.csv")

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        target_col = "target" if "target" in df.columns else df.columns[-1]
        X_raw = df.drop(columns=[target_col]).values
        y = df[target_col].values
    else:
        X_raw, y = generate_parity_32_samples(num_bits=num_bits)

    # 32 total samples split into 24 train / 8 test
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=split_ratio, random_state=random_state, stratify=y
    )

    for fm_cfg in feature_map_configs:
        fm_name = fm_cfg["name"]
        reps = fm_cfg.get("reps", 2)
        entanglement = fm_cfg.get("entanglement", "full")

        print(f"\n--- QSVM_Parity_{dataset_base}_{fm_name} ---")

        feature_map = build_feature_map(fm_cfg, num_qubits=num_bits)
        qkernel = get_quantum_kernel(feature_map, backend_config)

        # 1. Training & matrix build time
        start_train = time.perf_counter()
        matrix_train = qkernel.evaluate(x_vec=X_train)
        qsvm = SVC(kernel="precomputed", C=c_val)
        qsvm.fit(matrix_train, y_train)
        train_runtime = round(time.perf_counter() - start_train, 4)

        # 2. Prediction time
        start_pred = time.perf_counter()
        matrix_test = qkernel.evaluate(x_vec=X_test, y_vec=X_train)
        y_test_pred = qsvm.predict(matrix_test)
        predict_runtime = round(time.perf_counter() - start_pred, 4)

        y_train_pred = qsvm.predict(matrix_train)

        # Classification metrics
        acc = accuracy_score(y_test, y_test_pred)
        train_acc = accuracy_score(y_train, y_train_pred)
        prec = precision_score(y_test, y_test_pred, average="binary", zero_division=0)
        rec = recall_score(y_test, y_test_pred, average="binary", zero_division=0)
        f1 = f1_score(y_test, y_test_pred, average="binary", zero_division=0)

        cm = confusion_matrix(y_test, y_test_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        process = psutil.Process()
        mem_mb = round(process.memory_info().rss / (1024 ** 2), 2)
        cpu_pct = round(psutil.cpu_percent(interval=None), 2)

        # 3. Kernel plot & plotting runtime
        timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        kernel_filename = f"qsvm_{dataset_base}_{fm_name.lower()}_kernel_{timestamp_str}.png"

        plot_start = time.perf_counter()
        plot_quantum_kernel_matrix(
            matrix_train,
            title=f"Quantum Kernel Matrix ({dataset_base}, {fm_name})",
            filename=kernel_filename,
            save=True,
            show=False
        )
        plotting_runtime = round(time.perf_counter() - plot_start, 4)

        # Metrics payload
        metrics = {
            "model": f"QSVM_Parity_{fm_name}",
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "train_accuracy": round(train_acc, 4),
            "generalization_gap": round(train_acc - acc, 4),
            "training_runtime": train_runtime,
            "prediction_runtime": predict_runtime,
            "feasibility": True,
            "memory_MB": mem_mb,
            "cpu_percent": cpu_pct,
            "support_vectors": len(getattr(qsvm, "support_", [])),
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "backend": backend_config.get("type", "statevector"),
            "dataset": dataset_base,
            "kernel": "quantum_kernel",
            "split_ratio": split_ratio,
            "random_state": random_state,
            "num_qubits": feature_map.num_qubits,
            "circuit_depth": feature_map.decompose().depth(),
            "feature_map": fm_name,
            "reps": reps,
            "entanglement": entanglement,
            "pca_components": 0,
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "plotting_runtime": plotting_runtime,
        }

        # CSV log
        log_results(metrics)

        print(f"\n=== Quantum QSVM Parity Results ({dataset_base} - {fm_name}) ===")
        for k, v in metrics.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    run_parity_benchmark()