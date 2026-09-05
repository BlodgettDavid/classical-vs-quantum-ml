# src/phase1/SVM_BreastCancer_PCA.py

import os
import sys
import pandas as pd
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Setup repository paths
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(ROOT_DIR, "..", "..", "src")
if os.path.abspath(SRC_PATH) not in sys.path:
    sys.path.append(os.path.abspath(SRC_PATH))

from utils.config_loader import load_config
from utils.classical_evaluator import evaluate_model
from utils.logger import log_results
from utils.classical_visualizer import (
    plot_projected_decision_boundary,
    plot_confusion_matrix,
)

def run_breast_cancer_pca_baseline():
    # 1. Load configuration specifically for breast_cancer
    cfg = load_config("classical_svm.yaml", dataset_key="breast_cancer")
    
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

    # Extract hyperparameter configs directly from nested schema
    split_ratio = cfg.get("split_ratio", 0.3)
    random_state = cfg.get("random_state", 42)
    scale_data = cfg.get("scale_data", True)
    
    model_params = cfg.get("model_params", {})
    kernel = model_params.get("kernel", "rbf")
    C = model_params.get("C", 1.0)
    gamma = model_params.get("gamma", "scale")
    degree = model_params.get("degree", 0)

    # Extract array of components to test (e.g. [2, 4])
    pca_components_list = cfg.get("pca_components", [2, 4])
    if isinstance(pca_components_list, int):
        pca_components_list = [pca_components_list]

    for n_components in pca_components_list:
        print(f"\n--- Running Classical SVM on {dataset} (PCA={n_components}) ---")

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

        # 4. Instantiate Classical SVM
        model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, random_state=random_state)

        # 5. Evaluate Model
        model_name = f"SVM_BreastCancer_PCA{n_components}"
        eval_metrics = evaluate_model(
            model, X_train_pca, y_train, X_test_pca, y_test, label=model_name
        )

        # 6. Generate Visualizations
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        boundary_filename = f"svm_breastcancer_pca_{dataset}_pca{n_components}_{kernel}_{timestamp}.png"
        
        _, plotting_runtime = plot_projected_decision_boundary(
            model,
            X_test_pca,
            y_test,
            title=f"Classical SVM Breast Cancer ({n_components}D PCA, {kernel} kernel)",
            save=True,
            show=False,
            filename=boundary_filename,
            do_pca=(n_components > 2),
            grid_steps=50,
        )

        y_pred = model.predict(X_test_pca)
        cm_filename = f"svm_breastcancer_pca_{dataset}_pca{n_components}_{kernel}_cm_{timestamp}.png"
        plot_confusion_matrix(
            y_test,
            y_pred,
            title=f"Confusion Matrix ({dataset} PCA{n_components}, {kernel} kernel)",
            save=True,
            show=False,
            filename=cm_filename,
        )

        # 7. Construct Enriched Metrics Dictionary explicitly matching required schema
        metrics = {
            "model": model_name,
            "dataset": f"{dataset}_pca{n_components}",
            "backend": "CPU_Classical",
            "accuracy": round(eval_metrics.get("accuracy", 0.0), 4),
            "precision": round(eval_metrics.get("precision", 0.0), 4),
            "recall": round(eval_metrics.get("recall", 0.0), 4),
            "f1_score": round(eval_metrics.get("f1_score", 0.0), 4),
            "train_accuracy": round(eval_metrics.get("train_accuracy", 0.0), 4),
            "generalization_gap": round(
                eval_metrics.get("train_accuracy", 0.0) - eval_metrics.get("accuracy", 0.0), 4
            ),
            "training_runtime": round(eval_metrics.get("training_runtime", 0.0), 4),
            "prediction_runtime": round(eval_metrics.get("prediction_runtime", 0.0), 4),
            "plotting_runtime": round(plotting_runtime, 4),
            "feasibility": True,
            "memory_MB": round(eval_metrics.get("memory_MB", 0.0), 2),
            "cpu_percent": round(eval_metrics.get("cpu_percent", 0.0), 1),
            "support_vectors": int(len(model.support_)),
            "TP": int(eval_metrics.get("TP", 0)),
            "FP": int(eval_metrics.get("FP", 0)),
            "TN": int(eval_metrics.get("TN", 0)),
            "FN": int(eval_metrics.get("FN", 0)),
            "kernel": kernel,
            "C": C,
            "gamma": str(gamma),
            "degree": degree,
            "split_ratio": split_ratio,
            "random_state": random_state,
            "pca_components": n_components,
            "n_train_samples": len(X_train_pca),
            "n_test_samples": len(X_test_pca),
            "num_qubits": 0,
            "circuit_depth": 0,
            "feature_map": "N/A",
            "reps": 0,
            "entanglement": "N/A",
        }

        # 8. Log Enriched Metrics
        log_results(metrics)

        print(f"=== Results for PCA={n_components} ===")
        for k, v in metrics.items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    run_breast_cancer_pca_baseline()