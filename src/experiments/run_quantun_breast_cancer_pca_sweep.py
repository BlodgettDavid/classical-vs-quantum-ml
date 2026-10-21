# src/experiments/run_quantum_breast_cancer_pca_sweep.py

import os
import sys
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(ROOT_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from utils.sweep_visualizer import generate_sweep_summaries
from phase3.QSVM_BreastCancer_PCA import evaluate_quantum_breast_cancer_experiment
from utils.config_loader import load_config

def run_quantum_breast_cancer_sweep():
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%dT%H-%M-%SZ")
    
    cfg = load_config("quantum_svm.yaml", dataset_key="breast_cancer")
    dataset = cfg.get("dataset_name", cfg.get("dataset", "breast_cancer"))
    
    # Define quantum hyperparameter sweep stages for Breast Cancer PCA
    quantum_stages = [
        {"stage": "Stage1", "pca": 2,  "C": 0.1,    "desc": "Quantum Underfitting Baseline (PCA=2)"},
        {"stage": "Stage2", "pca": 2,  "C": 1.0,    "desc": "Quantum Kernel Curvature (PCA=2)"},
        {"stage": "Stage3", "pca": 4,  "C": 10.0,   "desc": "Optimal Quantum Limit (PCA=4)"},
        {"stage": "Stage4", "pca": 4,  "C": 1000.0, "desc": "Quantum Over-reaching (PCA=4)"},
        {"stage": "Stage5", "pca": 8,  "C": 100000.0, "desc": "QUANTUM OVERSHOOT COLLAPSE (PCA=8)"}
    ]

    print(f"\n======================================================================")
    print(f"STARTING QUANTUM BREAST CANCER PCA SWEEP [Timestamp: {timestamp}]")
    print(f"======================================================================")

    results_summary = []

    for step in quantum_stages:
        stage_name = step["stage"]
        n_components = step["pca"]
        C_val = step["C"]
        desc = step["desc"]

        print(f"\n--- Executing {stage_name}: PCA={n_components}, C={C_val} ({desc}) ---")
        
        metrics = evaluate_quantum_breast_cancer_experiment(
            n_components=n_components,
            C_val=C_val,
            stage_label=stage_name,
            timestamp=timestamp
        )
        
        results_summary.append({
            "timestamp": timestamp,
            "model": f"QSVM_BreastCancer_PCA_{stage_name}",
            "stage": stage_name,
            "C": C_val,
            "gamma": "N/A",
            "train_accuracy": metrics.get("train_accuracy"),
            "accuracy": metrics.get("accuracy"),
            "generalization_gap": metrics.get("generalization_gap")
        })

    results_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    df_sweep = pd.DataFrame(results_summary)
    
    master_path = os.path.join(results_dir, "results_quantum_sweeps.csv")
    dataset_path = os.path.join(results_dir, "results_quantum_breast_cancer_sweeps.csv")
    
    for path in [master_path, dataset_path]:
        if os.path.exists(path):
            existing_df = pd.read_csv(path)
            df_combined = pd.concat([existing_df, df_sweep], ignore_index=True).drop_duplicates()
            df_combined.to_csv(path, index=False)
        else:
            df_sweep.to_csv(path, index=False)

    print(f"\n======================================================================")
    print(f"QUANTUM BREAST CANCER SWEEP COMPLETE. Logging successful.")
    print(f"======================================================================\n")

    output_dir = os.path.join(ROOT_DIR, "plots", "sweeps", "quantum", "QSVM_BreastCancer_PCA", timestamp)
    generate_sweep_summaries(
        csv_path=dataset_path, 
        output_dir=output_dir,
        family_name="QSVM_BreastCancer_PCA",
        timestamp_filter=timestamp
    )

if __name__ == "__main__":
    run_quantum_breast_cancer_sweep()