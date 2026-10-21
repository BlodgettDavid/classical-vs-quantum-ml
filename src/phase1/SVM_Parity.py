# src/phase1/SVM_Parity.py
# Phase 1: Pristine Classical SVM benchmark on Parity Datasets (Supports Standalone & Harness execution)

import sys, os, time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
from sklearn.model_selection import train_test_split
from datetime import datetime, timezone
from itertools import product

# Setup paths
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(ROOT_DIR, "..", "..", "src")
if os.path.abspath(SRC_PATH) not in sys.path:
    sys.path.append(os.path.abspath(SRC_PATH))

from utils.logger import log_results
from utils.classical_visualizer import (
    plot_confusion_matrix,
    plot_kernel_matrix,
    plot_projected_decision_boundary
)
from utils.classical_evaluator import evaluate_model
from utils.config_loader import load_config

def generate_parity_32_samples(num_bits=4):
    """Generates a balanced 32-sample Parity dataset (2 complete cycles of 2^4 binary vectors)."""
    base_X = np.array(list(product([0, 1], repeat=num_bits)))
    base_y = np.sum(base_X, axis=1) % 2
    X = np.vstack([base_X, base_X])
    y = np.concatenate([base_y, base_y])
    return X, y




def evaluate_parity_experiment(C_val=None, gamma_val=None, stage_label=None, timestamp=None, output_dir=None):
    """
    Core execution pipeline. Can be called standalone or invoked iteratively by a harness.
    """
    # 1. Load parameters from unified config
    cfg = load_config("classical_svm.yaml", dataset_key="parity4d")
    dataset = cfg.get("dataset_name", "parity4d_stressed")

    split_ratio = cfg.get("split_ratio", 0.25)
    random_state = cfg.get("random_state", 42)

    svm_params = cfg.get("svm_params", {})
    kernel = svm_params.get("kernel", "rbf")
    
    # Fallback to config defaults if arguments aren't provided by a harness
    C = C_val if C_val is not None else svm_params.get("C", 10.0)
    gamma = gamma_val if gamma_val is not None else svm_params.get("gamma", "scale")
    degree = svm_params.get("degree", 4)

    # Determine logging label suffix and artifact run identifier
    if stage_label:
        label_suffix = f"_{stage_label}"
    else:
        label_suffix = "_baseline"

    model_name = f"SVM_Parity_{dataset}{label_suffix}"

    # 2. Load dataset
    DATA_DIR = os.path.join(ROOT_DIR, "..", "..", "data")
    data_path = os.path.join(DATA_DIR, f"{dataset}.csv")
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        target_col = "target" if "target" in df.columns else df.columns[-1]
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
    else:
        X, y = generate_parity_32_samples(num_bits=4)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=random_state, stratify=y
    )

    # 4. Model Training & Evaluation
    model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=random_state)
    metrics = evaluate_model(
        model, X_train, y_train, X_test, y_test, label=model_name
    )

    # Enriched metadata
    metrics["dataset"] = dataset
    metrics["split_ratio"] = split_ratio
    metrics["random_state"] = random_state
    metrics["kernel"] = kernel
    metrics["C"] = C
    metrics["gamma"] = gamma
    metrics["degree"] = degree if kernel == "poly" else 0
    metrics["pca_components"] = 0
    metrics["n_train_samples"] = len(X_train)
    metrics["n_test_samples"] = len(X_test)
    metrics["backend"] = "CPU_Classical"
    metrics["num_qubits"] = 0
    metrics["circuit_depth"] = 0
    metrics["feature_map"] = "N/A"
    metrics["reps"] = 0
    metrics["entanglement"] = "N/A"
    metrics["generalization_gap"] = round(metrics["train_accuracy"] - metrics["accuracy"], 4)

    # Visualizations
    # Use passed timestamp if available, otherwise generate a new one
    run_timestamp = timestamp if timestamp else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

    # Fallback output directory for standalone execution with a timestamped subfolder
    if not output_dir:
        output_dir = os.path.join(ROOT_DIR, "..", "..", "plots", "runs", "classical", "SVM_Parity", run_timestamp)

    os.makedirs(output_dir, exist_ok=True)

    plot_start_time = time.perf_counter()

    # ---------------------------------------------------------------------------
    # EDUCATIONAL NOTE FOR STUDENTS:
    # Parity Decision Boundary Visualization & Limitations
    # 
    # 1. Dimensionality Reduction Constraint:
    #    The parity dataset is 4-dimensional (living on the corners of a 4D hypercube). 
    #    To plot a 2D decision boundary, PCA must compress 4D down to 2D. This projection 
    #    inevitably causes information loss and class overlap in the visualization space.
    # 
    # 2. Interpreting the Boundary Plot:
    #    Unlike smooth continuous datasets (like Breast Cancer), parity logic is 
    #    non-linearly entangled and discrete. Therefore, the resulting mesh-grid plot 
    #    will look fragmented, speckled, or overlapping. 
    # 
    # 3. Learning Takeaway:
    #    Do not confuse visualization projection limits with model failure! The SVM 
    #    operates and finds separating hyperplanes in the high-dimensional kernel space, 
    #    even if its 2D shadow looks complex or messy.
    # ---------------------------------------------------------------------------
    
    if not stage_label:
        boundary_filename = f"svm_{dataset}{label_suffix}_{kernel}_boundary_{run_timestamp}.png"
        plot_projected_decision_boundary(
            model,
            X_test,
            y_test,
            title=f"Classical SVM Parity Decision Boundary ({kernel.upper()} kernel)",
            save=True,
            show=False,
            filename=boundary_filename,
            do_pca=True,
            grid_steps=100,
        )
        # Move decision boundary plot from default plots directory to output_dir
        default_plots_dir = os.path.join(ROOT_DIR, "..", "..", "plots", "runs")
        default_boundary_path = os.path.join(default_plots_dir, boundary_filename)
        target_boundary_path = os.path.join(output_dir, boundary_filename)
        if os.path.exists(default_boundary_path):
            os.replace(default_boundary_path, target_boundary_path)

    gamma_val_calc = 1.0 / X_train.shape[1] if gamma == "scale" else gamma
    gram_matrix = rbf_kernel(X_train, X_train, gamma=gamma_val_calc)

    kernel_filename = f"svm_{dataset}{label_suffix}_{kernel}_kernel_{run_timestamp}.png"
    plot_kernel_matrix(
        gram_matrix, 
        title=f"Kernel Matrix ({dataset}{label_suffix}, {kernel.upper()} kernel)", 
        save=True, 
        show=False, 
        filename=kernel_filename,
        output_dir=output_dir
    )

    y_pred = model.predict(X_test)
    cm_filename = f"svm_{dataset}{label_suffix}_{kernel}_cm_{run_timestamp}.png"
    plot_confusion_matrix(
        y_test, 
        y_pred, 
        title=f"Confusion Matrix ({dataset}{label_suffix}, {kernel.upper()} kernel)", 
        save=True, 
        show=False, 
        filename=cm_filename,
        output_dir=output_dir
    )

    metrics["plotting_runtime"] = round(time.perf_counter() - plot_start_time, 4)
    log_results(metrics)

    print(f"=== Results for {dataset}{label_suffix} ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return metrics

if __name__ == "__main__":
    # Pathway A: Standalone Execution
    print(f"\n--- Running Classical SVM Standalone ---")
    evaluate_parity_experiment()