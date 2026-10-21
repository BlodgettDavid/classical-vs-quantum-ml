# src/phase1/SVM_BreastCancer_PCA.py

import os
import sys
import time
import pandas as pd
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

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
    plot_kernel_matrix,
)

def evaluate_breast_cancer_experiment(n_components: int = 2, C_val: float = None, gamma_val = None, stage_label: str = None, timestamp: str = None, output_dir: str = None) -> dict:
    # 1. Load configuration specifically for breast_cancer
    cfg = load_config("classical_svm.yaml", dataset_key="breast_cancer")
    
    dataset = cfg.get("dataset_name", cfg.get("dataset", "breast_cancer"))
    data_dir = os.path.join(ROOT_DIR, "..", "..", "data")
    data_path = os.path.join(data_dir, f"{dataset}.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Breast cancer dataset not found at: {data_path}")

    df = pd.read_csv(data_path)

    # Resolve target column dynamically
    target_col = "target" if "target" in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    split_ratio = cfg.get("split_ratio", 0.3)
    random_state = cfg.get("random_state", 42)
    scale_data = cfg.get("scale_data", True)
    
    svm_params = cfg.get("svm_params", cfg.get("model_params", {}))
    kernel = svm_params.get("kernel", "rbf")
    degree = svm_params.get("degree", 0)
    
    # Fallbacks if arguments aren't provided by a harness/sweep
    C_val = C_val if C_val is not None else svm_params.get("C", 10.0)
    gamma_val = gamma_val if gamma_val is not None else svm_params.get("gamma", "scale")
    
    viz_cfg = cfg.get("visualization", {})
    grid_steps = viz_cfg.get("grid_steps", 100)

    if stage_label:
        label_suffix = f"_{stage_label}"
    else:
        label_suffix = "_baseline"

    model_name = f"SVM_BreastCancer_PCA{label_suffix}"

    print(f"\n--- Running Classical SVM on {dataset} (PCA={n_components}, C={C_val}) ---")

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

    # 4. Instantiate Classical SVM using passed sweep parameters
    model = SVC(kernel=kernel, C=C_val, gamma=gamma_val, degree=degree, random_state=random_state)

    # 5. Evaluate Model
    eval_metrics = evaluate_model(
        model, X_train_pca, y_train, X_test_pca, y_test, label=model_name
    )

    # 6. Generate Visualizations routed to timestamped directory under plots/runs/classical
    run_timestamp = timestamp if timestamp else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    
    if not output_dir:
        output_dir = os.path.join(ROOT_DIR, "..", "..", "plots", "runs", "classical", "SVM_BreastCancer_PCA", run_timestamp)
    os.makedirs(output_dir, exist_ok=True)

    plot_start_time = time.perf_counter()

    if not stage_label:
        # Decision Boundary Plot
        boundary_filename = f"svm_breastcancer_{dataset}{label_suffix}_pca{n_components}_{kernel}_{run_timestamp}.png"
        plot_projected_decision_boundary(
            model,
            X_test_pca,
            y_test,
            title=f"Classical SVM Breast Cancer ({n_components}D PCA, {kernel.upper()} kernel)",
            save=True,
            show=False,
            filename=boundary_filename,
            do_pca=(n_components > 2),
            grid_steps=grid_steps,
        )
        # Move decision boundary plot from default plots/runs to output_dir if needed
        default_plots_dir = os.path.join(ROOT_DIR, "..", "..", "plots", "runs")
        default_boundary_path = os.path.join(default_plots_dir, boundary_filename)
        target_boundary_path = os.path.join(output_dir, boundary_filename)
        if os.path.exists(default_boundary_path):
            os.replace(default_boundary_path, target_boundary_path)

    # Classical Kernel Matrix Heatmap (NxN)
    if kernel == "poly":
        gram_matrix = polynomial_kernel(X_train_pca, X_train_pca, degree=degree)
    else:
        gamma_resolved = 1.0 / X_train_pca.shape[1] if gamma_val == "scale" else gamma_val
        gram_matrix = rbf_kernel(X_train_pca, X_train_pca, gamma=gamma_resolved)

    kernel_filename = f"svm_breastcancer_{dataset}{label_suffix}_pca{n_components}_{kernel}_kernel_{run_timestamp}.png"
    plot_kernel_matrix(
        gram_matrix,
        title=f"Classical Kernel Matrix ({dataset.upper()} PCA{n_components}, {kernel.upper()} kernel)",
        save=True,
        show=False,
        filename=kernel_filename,
        output_dir=output_dir
    )

    # Confusion Matrix
    y_pred = model.predict(X_test_pca)
    cm_filename = f"svm_breastcancer_{dataset}{label_suffix}_pca{n_components}_{kernel}_cm_{run_timestamp}.png"
    plot_confusion_matrix(
        y_test,
        y_pred,
        title=f"Confusion Matrix ({dataset.upper()} PCA{n_components}, {kernel.upper()} kernel)",
        save=True,
        show=False,
        filename=cm_filename,
        output_dir=output_dir
    )

    total_plotting_runtime = round(time.perf_counter() - plot_start_time, 4)

    # 7. Construct Enriched Metrics Dictionary
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
        "plotting_runtime": total_plotting_runtime,
        "feasibility": True,
        "memory_MB": round(eval_metrics.get("memory_MB", 0.0), 2),
        "cpu_percent": round(eval_metrics.get("cpu_percent", 0.0), 2),
        "support_vectors": int(eval_metrics.get("support_vectors", 0)),
        "TP": int(eval_metrics.get("TP", 0)),
        "FP": int(eval_metrics.get("FP", 0)),
        "TN": int(eval_metrics.get("TN", 0)),
        "FN": int(eval_metrics.get("FN", 0)),
        "kernel": kernel,
        "C": C_val,
        "gamma": str(gamma_val),
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
        "timestamp": run_timestamp
    }

    log_results(metrics)
    
    print(f"=== Results for {model_name} ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return metrics

if __name__ == "__main__":
    print(f"\n--- Running Classical SVM Breast Cancer PCA Standalone ---")
    cfg = load_config("classical_svm.yaml", dataset_key="breast_cancer")
    pca_list = cfg.get("pca_components", [2])
    default_pca = pca_list[0] if isinstance(pca_list, list) and len(pca_list) > 0 else 2
    evaluate_breast_cancer_experiment(n_components=default_pca)