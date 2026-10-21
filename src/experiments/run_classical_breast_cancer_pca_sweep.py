# src/experiments/run_classical_breast_cancer_pca_sweep.py

import os
import sys
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(ROOT_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from utils.sweep_visualizer import generate_sweep_summaries
from phase1.SVM_BreastCancer_PCA import evaluate_breast_cancer_experiment
from utils.config_loader import load_config

def run_classical_breast_cancer_sweep():
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%dT%H-%M-%SZ")
    
    cfg = load_config("classical_svm.yaml", dataset_key="breast_cancer")
    dataset = cfg.get("dataset_name", cfg.get("dataset", "breast_cancer"))
    
    # Define hyperparameter sweep stages for Breast Cancer PCA
    classical_stages = [
        {"stage": "Stage1", "pca": 2,  "C": 0.1,    "gamma": 0.01,  "desc": "Underfitting Baseline (PCA=2)"},
        {"stage": "Stage2", "pca": 2,  "C": 1.0,    "gamma": 0.10,  "desc": "Moderate Regularization (PCA=2)"},
        {"stage": "Stage3", "pca": 4,  "C": 10.0,   "gamma": 1.00,  "desc": "Higher Dimension (PCA=4)"},
        {"stage": "Stage4", "pca": 4,  "C": 1000.0, "gamma": 10.0,  "desc": "High Variance (PCA=4)"},
        {"stage": "Stage5", "pca": 8,  "C": 100000.0, "gamma": 100.0, "desc": "Max Complexity (PCA=8)"}
    ]

    print(f"\n======================================================================")
    print(f"STARTING CLASSICAL BREAST CANCER PCA SWEEP [Timestamp: {timestamp}]")
    print(f"======================================================================")

    results_summary = []
    
    # Define the sweep-specific output directory for individual stage artifacts
    sweep_output_dir = os.path.join(ROOT_DIR, "plots", "sweeps", "classical", "SVM_BreastCancer_PCA", timestamp)
    os.makedirs(sweep_output_dir, exist_ok=True)

    for step in classical_stages:
        stage_name = step["stage"]
        n_components = step["pca"]
        C_val = step["C"]
        gamma_val = step["gamma"]
        desc = step["desc"]

        print(f"\n--- Executing {stage_name}: PCA={n_components}, C={C_val}, gamma={gamma_val} ({desc}) ---")
        
        metrics = evaluate_breast_cancer_experiment(
            n_components=n_components,
            C_val=C_val,
            gamma_val=gamma_val,
            stage_label=stage_name,
            timestamp=timestamp,
            output_dir=sweep_output_dir
        )
        
        results_summary.append({
            "timestamp": timestamp,
            "model": f"SVM_BreastCancer_PCA_{stage_name}",
            "stage": stage_name,
            "C": C_val,
            "gamma": gamma_val,
            "train_accuracy": metrics.get("train_accuracy"),
            "accuracy": metrics.get("accuracy"),
            "generalization_gap": metrics.get("generalization_gap")
        })

    results_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    df_sweep = pd.DataFrame(results_summary)
    
    master_path = os.path.join(results_dir, "results_classical_sweeps.csv")
    dataset_path = os.path.join(results_dir, "results_classical_breast_cancer_sweeps.csv")
    
    for path in [master_path, dataset_path]:
        if os.path.exists(path):
            existing_df = pd.read_csv(path)
            df_combined = pd.concat([existing_df, df_sweep], ignore_index=True).drop_duplicates()
            df_combined.to_csv(path, index=False)
        else:
            df_sweep.to_csv(path, index=False)

    print(f"\n======================================================================")
    print(f"CLASSICAL BREAST CANCER SWEEP COMPLETE. Logging successful.")
    print(f"======================================================================\n")

    generate_sweep_summaries(
        csv_path=dataset_path, 
        output_dir=sweep_output_dir,
        family_name="SVM_BreastCancer_PCA",
        timestamp_filter=timestamp
    )

if __name__ == "__main__":
    run_classical_breast_cancer_sweep()