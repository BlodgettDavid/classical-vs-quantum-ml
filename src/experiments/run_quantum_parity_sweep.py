# src/experiments/run_quantum_parity_sweep.py
# Experiment Harness: Quantum QSVM Parity Sweep

import sys, os
import pandas as pd

# Setup paths to ensure clean imports from src/
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(ROOT_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from utils.sweep_visualizer import generate_sweep_summaries
# Adjust import below if your quantum parity evaluation function has a distinct module/name name
from phase2.QSVM_Parity import evaluate_quantum_parity_experiment

def run_quantum_sweep():
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%dT%H-%M-%SZ")
    
    # Define the corresponding quantum hyperparameter sweep stages
    quantum_stages = [
        {"stage": "Stage1", "C": 0.1,    "gamma": 0.01,  "desc": "Quantum Underfitting Baseline"},
        {"stage": "Stage2", "C": 1.0,    "gamma": 0.10,  "desc": "Quantum Kernel Curvature"},
        {"stage": "Stage3", "C": 10.0,   "gamma": 1.00,  "desc": "Optimal Quantum Limit"},
        {"stage": "Stage4", "C": 1000.0, "gamma": 10.0,  "desc": "Quantum Over-reaching"},
        {"stage": "Stage5", "C": 100000.0, "gamma": 100.0, "desc": "QUANTUM OVERSHOOT COLLAPSE"}
    ]

    print(f"\n======================================================================")
    print(f"STARTING QUANTUM QSVM PARITY SWEEP [Timestamp: {timestamp}]")
    print(f"======================================================================")

    results_summary = []

    for step in quantum_stages:
        stage_name = step["stage"]
        C_val = step["C"]
        gamma_val = step["gamma"]
        desc = step["desc"]

        print(f"\n--- Executing {stage_name}: C={C_val}, gamma={gamma_val} ({desc}) ---")
        
        # Invoke quantum evaluation function
        metrics = evaluate_quantum_parity_experiment(C_val=C_val, gamma_val=gamma_val, stage_label=f"QSVM_Parity_{stage_name}", timestamp=timestamp)
        
        results_summary.append({
            "timestamp": timestamp,
            "model": f"QSVM_Parity_{stage_name}",
            "stage": stage_name,
            "C": C_val,
            "gamma": gamma_val,
            "train_accuracy": metrics.get("train_accuracy"),
            "accuracy": metrics.get("accuracy"),
            "generalization_gap": metrics.get("generalization_gap")
        })

    # Save to fine-grained Quantum CSV logs
    results_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    df_sweep = pd.DataFrame(results_summary)
    
    master_path = os.path.join(results_dir, "results_quantum_sweeps.csv")
    dataset_path = os.path.join(results_dir, "results_quantum_parity_sweeps.csv")
    
    for path in [master_path, dataset_path]:
        if os.path.exists(path):
            existing_df = pd.read_csv(path)
            df_combined = pd.concat([existing_df, df_sweep], ignore_index=True).drop_duplicates()
            df_combined.to_csv(path, index=False)
        else:
            df_sweep.to_csv(path, index=False)

    print(f"\n======================================================================")
    print(f"QUANTUM SWEEP COMPLETE. Logging to fine-grained CSVs successful.")
    print(f"======================================================================\n")

    # Generate summary plots routed directly to plots/sweeps/quantum/QSVM_Parity/<timestamp>/
    generate_sweep_summaries(
        csv_path=dataset_path, 
        output_dir=os.path.join(ROOT_DIR, "plots", "sweeps", "quantum", "QSVM_Parity", timestamp),
        family_name="QSVM_Parity",
        timestamp_filter=timestamp
    )

if __name__ == "__main__":
    run_quantum_sweep()