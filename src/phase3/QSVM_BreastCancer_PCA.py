# src/phase1/QSVM_BreastCancer_PCA.py

import os
import sys
import time
import pandas as pd
from datetime import datetime, timezone

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC


from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel, FidelityQuantumKernel
from qiskit_aer import AerSimulator

import numpy as np


# Setup repository paths
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(ROOT_DIR, "..", "..", "src")
if os.path.abspath(SRC_PATH) not in sys.path:
    sys.path.append(os.path.abspath(SRC_PATH))

from utils.config_loader import load_config
from utils.quantum_evaluator import evaluate_quantum_model
from utils.logger import log_results
from utils.quantum_visualizer import (
    plot_quantum_kernel_matrix,
    plot_quantum_confusion_matrix,
    plot_qsvm_decision_boundary,
)


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

    # Extract class_weight from svm_params (defaults to None if missing)
    class_weight = cfg.get("svm_params", {}).get("class_weight", None)
    

    pca_components_list = cfg.get("pca_components", [2, 4])
    if isinstance(pca_components_list, int):
        pca_components_list = [pca_components_list]

    quantum_params = cfg.get("quantum_params", {})
    feature_map_configs = quantum_params.get("feature_maps", [])
    backend_config = cfg.get("backend_config", {"type": "statevector"})
    backend_type = backend_config.get("type", "statevector")

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

        # 2. Rescale PCA features to [0, pi] for quantum feature map phase encoding
        quantum_scaler = MinMaxScaler(feature_range=(0, np.pi))
        X_train_pca = quantum_scaler.fit_transform(X_train_pca)
        X_test_pca = quantum_scaler.transform(X_test_pca)



        for fm_cfg in feature_map_configs:
            fm_name = fm_cfg["name"]
            reps = fm_cfg.get("reps", 2)
            entanglement = fm_cfg.get("entanglement", "full")

            model_label = f"QSVM_BreastCancer_PCA{n_components}_{fm_name}"

            # 4. Build Quantum Feature Map & Kernel
            feature_map = build_feature_map(fm_cfg, num_qubits=n_components)
            qkernel = get_quantum_kernel(feature_map, backend_config)
            #qsvm = SVC(kernel="precomputed", C=c_val)

            # Initialize SVC solver with precomputed quantum kernel matrix
            qsvm = SVC(kernel="precomputed", C=c_val, class_weight=class_weight)


            # 5. Evaluate Quantum Model (Execution benchmarked in isolation)
            eval_res = evaluate_quantum_model(
                qsvm=qsvm,
                qkernel=qkernel,
                X_train=X_train_pca,
                y_train=y_train,
                X_test=X_test_pca,
                y_test=y_test,
                label=model_label,
                feature_map=feature_map,
                backend_name=backend_type,
            )

            # 6. Generate Complete Visualization Suite (Timed independently for plotting_runtime)
            timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
            viz_cfg = cfg.get("visualization", {})
            plot_start = time.perf_counter()

            # A. Kernel Matrix Heatmap
            kernel_filename = f"qsvm_breastcancer_{dataset}_pca{n_components}_{fm_name.lower()}_kernel_{timestamp_str}.png"
            plot_quantum_kernel_matrix(
                eval_res["matrix_train"],
                title=f"Quantum Kernel Matrix ({dataset} PCA={n_components}, {fm_name})",
                filename=kernel_filename,
                save=True,
                show=False
            )

            # B. Categorical Confusion Matrix Heatmap
            cm_filename = f"qsvm_breastcancer_{dataset}_pca{n_components}_{fm_name.lower()}_cm_{timestamp_str}.png"
            plot_quantum_confusion_matrix(
                y_true=y_test,
                y_pred=eval_res["y_pred"],
                title=f"Quantum Confusion Matrix ({dataset} PCA={n_components}, {fm_name})",
                filename=cm_filename,
                save=True,
                show=False
            )

            # C. Decision Boundary Mesh Plot
            if viz_cfg.get("plot_decision_boundary", True):
                grid_steps = viz_cfg.get("grid_steps", 25)
                boundary_filename = f"qsvm_breastcancer_{dataset}_pca{n_components}_{fm_name.lower()}_{timestamp_str}.png"

                class QuantumSVCWrapper:
                    def __init__(self, model, kernel, X_train):
                        self.model = model
                        self.kernel = kernel
                        self.X_train = X_train

                    def decision_function(self, X_grid):
                        K_grid = self.kernel.evaluate(x_vec=X_grid, y_vec=self.X_train)
                        return self.model.decision_function(K_grid)

                qsvm_wrapper = QuantumSVCWrapper(qsvm, qkernel, X_train_pca)

                plot_qsvm_decision_boundary(
                    qsvc=qsvm_wrapper,
                    X=X_train_pca,
                    y=y_train,
                    title=f"Quantum SVM Breast Cancer ({n_components}D PCA, {fm_name})",
                    filename=boundary_filename,
                    do_pca=(n_components > 2),
                    grid_steps=grid_steps,
                    save=True,
                    show=False
                )

            total_plotting_runtime = round(time.perf_counter() - plot_start, 4)

            # 7. Construct Enriched Metrics Dictionary
            metrics = {
                "model": model_label,
                "dataset": f"{dataset}_pca{n_components}",
                "backend": backend_type,
                "accuracy": eval_res["accuracy"],
                "precision": eval_res["precision"],
                "recall": eval_res["recall"],
                "f1_score": eval_res["f1_score"],
                "train_accuracy": eval_res["train_accuracy"],
                "generalization_gap": eval_res["generalization_gap"],
                "training_runtime": eval_res["training_runtime"],
                "prediction_runtime": eval_res["prediction_runtime"],
                "plotting_runtime": total_plotting_runtime,
                "feasibility": eval_res["feasibility"],
                "memory_MB": eval_res["memory_MB"],
                "cpu_percent": eval_res["cpu_percent"],
                "support_vectors": eval_res["support_vectors"],
                "TP": eval_res["TP"],
                "FP": eval_res["FP"],
                "TN": eval_res["TN"],
                "FN": eval_res["FN"],
                "kernel": "quantum_kernel",
                "C": c_val,
                "gamma": "N/A",
                "degree": 0,
                "split_ratio": split_ratio,
                "random_state": random_state,
                "pca_components": n_components,
                "n_train_samples": len(X_train_pca),
                "n_test_samples": len(X_test_pca),
                "num_qubits": eval_res["num_qubits"],
                "circuit_depth": eval_res["circuit_depth"],
                "feature_map": fm_name,
                "reps": reps,
                "entanglement": entanglement,
            }

            # 8. Log Results & Print Output
            log_results(metrics)

            print(f"\n=== Results for {dataset} PCA={n_components} ({fm_name}) ===")
            for k, v in metrics.items():
                print(f"{k}: {v}")


if __name__ == "__main__":
    run_breast_cancer_pca_quantum()