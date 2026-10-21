import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


# Compute project root relative to this script (src/analysis/ -> root)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT_DIR / "results"
BASE_PLOTS_DIR = ROOT_DIR / "plots" / "sweep_comparisons"

# Generate a timestamp for this execution batch (e.g., 2026-10-19T01-30-00Z)
BATCH_TIMESTAMP = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
TIMESTAMPED_PLOTS_DIR = BASE_PLOTS_DIR / BATCH_TIMESTAMP

DATASETS = {
    "parity": {
        "classical_csv": RESULTS_DIR / "results_classical_parity_sweeps.csv",
        "quantum_csv": RESULTS_DIR / "results_quantum_parity_sweeps.csv",
        "output_dir": TIMESTAMPED_PLOTS_DIR / "parity"
    },
    "breast_cancer": {
        "classical_csv": RESULTS_DIR / "results_classical_breast_cancer_sweeps.csv",
        "quantum_csv": RESULTS_DIR / "results_quantum_breast_cancer_sweeps.csv",
        "output_dir": TIMESTAMPED_PLOTS_DIR / "breast_cancer"
    }
}

def select_run(model_name, df, dataset_name):
    """Handles auto-selection or interactive CLI selection if multiple timestamps exist."""
    if df is None or df.empty:
        return None
    
    timestamps = sorted(df['timestamp'].unique(), reverse=True)
    
    if len(timestamps) == 1:
        print(f"[INFO] [{dataset_name.upper()}] Only one {model_name} sweep found ({timestamps[0]}). Auto-selecting.")
        return timestamps[0]
    
    print(f"\n[SELECTION REQUIRED] Multiple {model_name} sweeps found for {dataset_name}:")
    for idx, ts in enumerate(timestamps, 1):
        print(f"  {idx}) {ts}{' [Latest]' if idx == 1 else ''}")
    
    while True:
        try:
            choice = input(f"Select {model_name} run [1-{len(timestamps)}, default=1]: ").strip()
            if not choice:
                return timestamps[0]
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(timestamps):
                return timestamps[choice_idx]
        except ValueError:
            pass
        print("[ERROR] Invalid selection. Please try again.")

def generate_comparison_report(dataset_name, paths):
    print(f"\n--- Processing Comparison Summary: {dataset_name.upper()} ---")
    
    c_csv = paths["classical_csv"]
    q_csv = paths["quantum_csv"]
    out_dir = paths["output_dir"]
    
    # Graceful missing data checks
    missing = []
    if not c_csv.exists(): missing.append(str(c_csv))
    if not q_csv.exists(): missing.append(str(q_csv))
    
    if missing:
        print(f"[WARNING] Missing sweep logs for '{dataset_name}': {missing}")
        print(f"[WARNING] Skipping comparison summary for {dataset_name} (partial or missing data).")
        return

    df_c = pd.read_csv(c_csv)
    df_q = pd.read_csv(q_csv)
    
    # Select specific runs independently
    ts_c = select_run("Classical", df_c, dataset_name)
    ts_q = select_run("Quantum", df_q, dataset_name)
    
    if not ts_c or not ts_q:
        print(f"[WARNING] Could not resolve runs for {dataset_name}. Skipping.")
        return
        
    df_c_filtered = df_c[df_c['timestamp'] == ts_c].sort_values('stage')
    df_q_filtered = df_q[df_q['timestamp'] == ts_q].sort_values('stage')
    
    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Plotting Side-by-Side Comparison (Accuracy Progression)
    plt.figure(figsize=(10, 6))
    stages = df_c_filtered['stage'].tolist()
    
    plt.plot(stages, df_c_filtered['accuracy'], marker='o', linestyle='-', label=f'Classical SVM ({ts_c[:10]})', color='royalblue', linewidth=2)
    plt.plot(stages, df_q_filtered['accuracy'], marker='s', linestyle='--', label=f'Quantum QSVM ({ts_q[:10]})', color='darkorange', linewidth=2)
    
    plt.title(f'Classical vs. Quantum Benchmark: {dataset_name.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    plt.xlabel('Sweep Stages (Increasing Complexity/Parameters)', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    output_path = out_dir / f"comparison_{dataset_name}_accuracy_trend.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"[SUCCESS] Generated comparison report saved to: {output_path}")

def main():
    print("==================================================")
    print("  MLOps Cross-Model Comparison Summary Generator  ")
    print("==================================================")
    
    for dataset_name, paths in DATASETS.items():
        generate_comparison_report(dataset_name, paths)
        
    print("\n[INFO] All comparison summary tasks completed.")

if __name__ == "__main__":
    main()