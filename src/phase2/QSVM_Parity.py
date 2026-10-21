# src/phase2/QSVM_Parity.py
# Phase 2: Quantum QSVM benchmark on Parity Datasets (Supports Standalone & Harness execution)

import os
import sys
import yaml
import time
import psutil
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from itertools import product

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel, FidelityQuantumKernel

# Setup repository paths
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(ROOT_DIR, "..")
if os.path.abspath(SRC_PATH) not in sys.path:
    sys.path.append(os.path.abspath(SRC_PATH))

from utils.logger import log_results
from utils.quantum_visualizer import plot_quantum_kernel_matrix, plot_quantum_confusion_matrix, plot_qsvm_decision_boundary


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
    reps = fm_config.get("reps", 1)
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
        return FidelityQuantumKernel(feature_map=feature_map, fidelity_shots=shots)

def evaluate_quantum_parity_experiment(C_val=None, gamma_val=None, stage_label=None, timestamp=None, output_dir=None):
    cfg = load_config("quantum_svm.yaml", "parity4d")

    dataset_base = cfg.get("dataset_name", "parity4d_stressed")
    split_ratio = cfg.get("split_ratio", 0.25)
    random_state = cfg.get("random_state", 42)
    scale_data = cfg.get("scale_data", True)
    
    # Dynamic PCA components configuration handling (supporting lists for sweeps vs single values for standalone)
    pca_list = cfg.get("pca_components", [0])
    n_components = pca_list[0] if isinstance(pca_list, list) and len(pca_list) > 0 else 0

    # Allow override from sweep runner, otherwise fall back to config
    c_val = C_val if C_val is not None else cfg["svm_params"]["C"]
    
    feature_map_configs = cfg["quantum_params"]["feature_maps"]
    backend_config = cfg["backend_config"]

    # Use synchronized timestamp if provided
    run_timestamp = timestamp if timestamp else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

    # Determine label suffix
    if stage_label:
        label_suffix = f"_{stage_label}"
        model_name = f"QSVM_Parity_{dataset_base}_{stage_label}"
    else:
        label_suffix = "_baseline"
        model_name = f"QSVM_Parity_{dataset_base}"

    project_root = os.path.abspath(os.path.join(ROOT_DIR, "..", ".."))
    
    # Symmetric output directory routing for runs vs sweeps
    if not output_dir:
        if stage_label:
            output_dir = os.path.join(project_root, "plots", "sweeps", "quantum", "QSVM_Parity", run_timestamp)
        else:
            output_dir = os.path.join(project_root, "plots", "runs", "quantum", "QSVM_Parity", run_timestamp)

    os.makedirs(output_dir, exist_ok=True)

    num_bits = 4
    data_path = os.path.join(project_root, "data", f"{dataset_base}.csv")

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        target_col = "target" if "target" in df.columns else df.columns[-1]
        X_raw = df.drop(columns=[target_col]).values
        y = df[target_col].values
    else:
        X_raw, y = generate_parity_32_samples(num_bits=num_bits)

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=split_ratio, random_state=random_state, stratify=y
    )

    if scale_data:
        scaler = MinMaxScaler(feature_range=(0, np.pi))
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # For the sweep, we typically evaluate the primary feature map config (e.g., first one or ZZFeatureMap)
    fm_cfg = feature_map_configs[0]
    fm_name = fm_cfg["name"]
    reps = fm_cfg.get("reps", 1)
    entanglement = fm_cfg.get("entanglement", "full")

    print(f"\n--- {model_name} [Run: {run_timestamp}] ---")

    feature_map = build_feature_map(fm_cfg, num_qubits=num_bits)
    qkernel = get_quantum_kernel(feature_map, backend_config)

    start_train = time.perf_counter()
    matrix_train = qkernel.evaluate(x_vec=X_train)
    qsvm = SVC(kernel="precomputed", C=c_val)
    qsvm.fit(matrix_train, y_train)
    train_runtime = round(time.perf_counter() - start_train, 4)

    start_pred = time.perf_counter()
    matrix_test = qkernel.evaluate(x_vec=X_test, y_vec=X_train)
    y_test_pred = qsvm.predict(matrix_test)
    predict_runtime = round(time.perf_counter() - start_pred, 4)

    y_train_pred = qsvm.predict(matrix_train)

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

    kernel_filename = f"qsvm_{dataset_base}{label_suffix}_{fm_name.lower()}_kernel_{run_timestamp}.png"
    cm_filename = f"qsvm_{dataset_base}{label_suffix}_{fm_name.lower()}_cm_{run_timestamp}.png"

    plot_start = time.perf_counter()
    
    # ---------------------------------------------------------------------------
    # EDUCATIONAL NOTE FOR STUDENTS:
    # Parity Decision Boundary Visualization & Limitations
    # 
    # 1. Dimensionality Reduction Constraint:
    #    The parity dataset is 4-dimensional (living on the corners of a 4D hypercube). 
    #    Projecting this down to 2D for visualization causes information loss and overlap.
    # 
    # 2. Interpreting the Boundary Plot:
    #    Unlike continuous datasets, parity logic is discrete and entangled, leading 
    #    to fragmented or complex visualization boundaries in projected 2D spaces.
    # 
    # 3. Learning Takeaway:
    #    Do not confuse 2D projection limits with quantum kernel failure! The QSVM 
    #    operates in the high-dimensional Hilbert space mapped by the quantum circuit.
    # ---------------------------------------------------------------------------

    plot_quantum_kernel_matrix(
        matrix_train, 
        title=f"Quantum Kernel Matrix ({dataset_base}{label_suffix}, {fm_name})", 
        filename=kernel_filename, 
        output_dir=output_dir,
        save=True, 
        show=False
    )
    plot_quantum_confusion_matrix(
        y_test, 
        y_test_pred, 
        title=f"QSVM Parity ({fm_name}) Confusion Matrix", 
        filename=cm_filename, 
        output_dir=output_dir,
        save=True, 
        show=False
    )

    # Only generate the decision boundary plot for standalone runs,
    # keeping automated sweeps fast, clean, and un-cluttered.
    # ---------------------------------------------------------------------------
    # EDUCATIONAL NOTE FOR STUDENTS:
    # Quantum Parity Decision Boundary Visualization & Limitations
    # 
    # 1. Dimensionality Reduction Constraint:
    #    The parity dataset is 4-dimensional (living on the corners of a 4D hypercube). 
    #    To plot a 2D decision boundary, PCA (or projection) compresses 4D down to 2D, 
    #    inevitably causing information loss and class overlap in the visualization space.
    # 
    # 2. Interpreting the Quantum Boundary Plot:
    #    Unlike continuous datasets, parity logic is discrete and entangled. Furthermore, 
    #    the QSVM classifies data using a quantum kernel evaluated in Hilbert space. 
    #    Projecting this back down to a 2D mesh grid often results in fragmented or 
    #    complex boundaries.
    # 
    # 3. Learning Takeaway:
    #    Do not confuse 2D projection limits with quantum kernel failure! The QSVM 
    #    successfully separates the data using quantum feature maps in high-dimensional 
    #    Hilbert space, even if its 2D shadow looks messy or speckled.
    # ---------------------------------------------------------------------------
    if not stage_label:
        boundary_filename = f"qsvm_{dataset_base}_baseline_{fm_name.lower()}_boundary_{run_timestamp}.png"
        plot_qsvm_decision_boundary(
            qsvm, 
            X_train, 
            y_train, 
            qkernel=qkernel, 
            title=f"QSVM Parity Projected Decision Boundary ({fm_name})", 
            filename=boundary_filename, 
            output_dir=output_dir,
            save=True, 
            show=False
        )
    plotting_runtime = round(time.perf_counter() - plot_start, 4)

    metrics = {
        "model": model_name,
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
        "pca_components": n_components,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "plotting_runtime": plotting_runtime,
        "timestamp": run_timestamp,
        "C": c_val,
        "gamma": "N/A"
    }

    log_results(metrics)
    
    print(f"=== Results for {model_name} ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return metrics

if __name__ == "__main__":
    print(f"\n--- Running Quantum QSVM Parity Standalone ---")
    evaluate_quantum_parity_experiment()