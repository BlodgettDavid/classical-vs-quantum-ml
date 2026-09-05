import os
import sys
import time
import psutil
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel, FidelityQuantumKernel
from qiskit_aer import AerSimulator

# Setup repository paths matching Phase 1
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(ROOT_DIR, "..", "..", "src")
if os.path.abspath(SRC_PATH) not in sys.path:
    sys.path.append(os.path.abspath(SRC_PATH))

from utils.config_loader import load_config
from utils.logger import log_results
from utils.quantum_visualizer import plot_quantum_kernel_matrix


def build_feature_map(fm_config, num_qubits):
    """Factory function to build Qiskit feature maps based on YAML config."""
    name = fm_config["name"]
    reps = fm_config.get("reps", 2)
    entanglement = fm_config.get("entanglement", "full")

    if name == "ZZFeatureMap":
        return ZZFeatureMap(
            feature_dimension=num_qubits,
            reps=reps,
            entanglement=entanglement
        )
    elif name == "PauliFeatureMap":
        paulis = fm_config.get("paulis", ["Z", "ZZ"])
        return PauliFeatureMap(
            feature_dimension=num_qubits,
            reps=reps,
            entanglement=entanglement,
            paulis=paulis
        )
    else:
        raise ValueError(f"Unsupported feature map: {name}")


def get_quantum_kernel(feature_map, backend_config):
    """Instantiates exact statevector or shot-based quantum kernel."""
    backend_type = backend_config.get("type", "statevector")
    shots = backend_config.get("shots", None)

    if backend_type == "statevector" or shots is None:
        return FidelityStatevectorKernel(feature_map=feature_map)
    else:
        backend = AerSimulator()
        return FidelityQuantumKernel(feature_map=feature_map, fidelity_shots=shots)


def run_breast_cancer_pca_quantum():
    # 1. Load configuration matching Phase 1 loader pattern
    cfg = load_config("quantum_svm.yaml", dataset_key="breast_cancer")

    dataset = cfg.get("dataset", "breast_cancer")
    data_dir = os.path.join(ROOT_DIR, "..", "..", "data")
    data_path = os.path.join(data_dir, f"{dataset}.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Breast cancer dataset not found at: {data_path}")

    df = pd.read_csv(data_path)

    # Resolve target column dynamically
    target_col = "target" if "target" in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # Extract hyperparameter configs
    split_ratio = cfg.get("split_ratio", 0.3)
    random_state = cfg.get("random_state", 42)
    scale_data = cfg.get("scale_data", True)

    c_val = cfg.get("svm_params", {}).get("c_val", 1.0)
    pca_components_list = cfg.get("pca_components", [2, 4])
    if isinstance(pca_components_list, int):
        pca_components_list = [pca_components_list]

    quantum_params = cfg.get("quantum_params", {})
    feature_map_configs = quantum_params.get("feature_maps", [])
    backend_config = cfg.get("backend_config", {"type": "statevector"})

    for n_components in pca_components_list:
        print(f"\n--- Running Quantum QSVM on {dataset} (PCA={n_components}) ---")

        # 2. Train / Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split_ratio, random_state=random_state, stratify=y
        )

        # 3. Scaling and Dimensionality Reduction
        if scale_data:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        pca = PCA(n_components=n_components, random_state=random_state)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        for fm_cfg in feature_map_configs:
            fm_name = fm_cfg["name"]
            reps = fm_cfg.get("reps", 2)
            entanglement = fm_cfg.get("entanglement", "full")

            print(f"\n--- QSVM_BreastCancer_PCA{n_components}_{fm_name} ---")

            # 4. Build Quantum Feature Map & Kernel
            feature_map = build_feature_map(fm_cfg, num_qubits=n_components)
            qkernel = get_quantum_kernel(feature_map, backend_config)

            # Training runtime & precomputed matrix fit
            start_train = time.perf_counter()
            matrix_train = qkernel.evaluate(x_vec=X_train_pca)
            qsvm = SVC(kernel="precomputed", C=c_val)
            qsvm.fit(matrix_train, y_train)
            train_runtime = round(time.perf_counter() - start_train, 4)

            # Prediction runtime
            start_pred = time.perf_counter()
            matrix_test = qkernel.evaluate(x_vec=X_test_pca, y_vec=X_train_pca)
            y_test_pred = qsvm.predict(matrix_test)
            predict_runtime = round(time.perf_counter() - start_pred, 4)

            y_train_pred = qsvm.predict(matrix_train)

            # 5. Evaluate Metrics
            acc = accuracy_score(y_test, y_test_pred)
            train_acc = accuracy_score(y_train, y_train_pred)
            prec = precision_score(y_test, y_test_pred, average="binary", zero_division=0)
            rec = recall_score(y_test, y_test_pred, average="binary", zero_division=0)
            f1 = f1_score(y_test, y_test_pred, average="binary", zero_division=0)

            cm = confusion_matrix(y_test, y_test_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

            process = psutil.Process()
            mem_mb = round(process.memory_info().rss / (1024 ** 2), 2)
            cpu_pct = round(psutil.cpu_percent(interval=None), 1)

            # 6. Generate Kernel Visualizations
            timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
            kernel_filename = f"qsvm_breastcancer_{dataset}_pca{n_components}_{fm_name.lower()}_kernel_{timestamp_str}.png"

            plot_start = time.perf_counter()
            plot_quantum_kernel_matrix(
                matrix_train,
                title=f"Quantum Kernel Matrix ({dataset} PCA={n_components}, {fm_name})",
                filename=kernel_filename,
                save=True,
                show=False
            )
            plotting_runtime = round(time.perf_counter() - plot_start, 4)

            # Enriched metric dictionary aligned with exact schema requirements
            metrics = {
                "model": f"QSVM_BreastCancer_PCA{n_components}_{fm_name}",
                "dataset": f"{dataset}_pca{n_components}",
                "backend": backend_config.get("type", "statevector"),
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "train_accuracy": round(train_acc, 4),
                "generalization_gap": round(train_acc - acc, 4),
                "training_runtime": train_runtime,
                "prediction_runtime": predict_runtime,
                "plotting_runtime": plotting_runtime,
                "feasibility": True,
                "memory_MB": mem_mb,
                "cpu_percent": cpu_pct,
                "support_vectors": int(len(getattr(qsvm, "support_", []))),
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn),
                "kernel": "quantum_kernel",
                "C": c_val,
                "gamma": "N/A",
                "degree": 0,
                "split_ratio": split_ratio,
                "random_state": random_state,
                "pca_components": n_components,
                "n_train_samples": len(X_train_pca),
                "n_test_samples": len(X_test_pca),
                "num_qubits": feature_map.num_qubits,
                "circuit_depth": feature_map.decompose().depth(),
                "feature_map": fm_name,
                "reps": reps,
                "entanglement": entanglement,
            }

            # 7. Log Results & Print Key-Value Output
            log_results(metrics)

            print(f"\n=== Results for {dataset} PCA={n_components} ({fm_name}) ===")
            for k, v in metrics.items():
                print(f"{k}: {v}")


if __name__ == "__main__":
    run_breast_cancer_pca_quantum()