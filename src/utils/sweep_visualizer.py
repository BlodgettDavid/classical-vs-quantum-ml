# src/utils/sweep_visualizer.py
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

def generate_sweep_summaries(csv_path: str, output_dir: str, family_name: str, timestamp_filter: str = None):
    """
    Reads a specific fine-grained sweep results CSV, extracts multi-stage metrics,
    and generates professional comparative trend charts saved in the precise target directory.
    """
    if not os.path.exists(csv_path):
        print(f"[!] Target sweep CSV not found: '{csv_path}'")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[!] Error reading {csv_path}: {e}")
        return

    if df.empty or "model" not in df.columns:
        print(f"[!] CSV file {csv_path} is empty or lacks 'model' column.")
        return

    # Filter by timestamp if provided
    if timestamp_filter and "timestamp" in df.columns:
        df["timestamp_str"] = df["timestamp"].astype(str)
        df = df[df["timestamp_str"].str.contains(timestamp_filter, na=False)]

    if df.empty:
        print("[!] No matching sweep data found for the given filter.")
        return

    # Extract stage numbers
    df["stage_num"] = df["model"].str.extract(r"Stage(\d+)").astype(int)
    df_sorted = df.sort_values("stage_num").drop_duplicates(subset=["stage_num"])

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    stages = []
    for _, row in df_sorted.iterrows():
        c_val = row.get("C", "N/A")
        stages.append(f"Stage {row['stage_num']}\n(C={c_val})")

    ax.plot(stages, df_sorted["train_accuracy"], marker='o', linestyle='-', linewidth=2.5, label="Training Accuracy", color="#2b5c8f")
    ax.plot(stages, df_sorted["accuracy"], marker='s', linestyle='--', linewidth=2.5, label="Testing Accuracy", color="#d95f02")
    
    if "generalization_gap" in df_sorted.columns:
        ax.plot(stages, df_sorted["generalization_gap"], marker='^', linestyle=':', linewidth=2.5, label="Generalization Gap", color="#7570b3")

    run_label = timestamp_filter if timestamp_filter else "Latest"
    ax.set_title(f"Sweep Trend: {family_name} [Run: {run_label}]", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Experiment Stages & Hyperparameters", fontsize=11)
    ax.set_ylabel("Metric Value / Accuracy", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="center left", frameon=True)

    # Include timestamp in filename if available for absolute clarity
    run_suffix = f"_{timestamp_filter}" if timestamp_filter else ""
    out_path = os.path.join(output_dir, f"sweep_trend_{family_name}{run_suffix}.png")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved sweep trend summary plot: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Targeted Sweep Visualizer")
    parser.add_argument("--csv", type=str, required=True, help="Path to the fine-grained sweep results CSV")
    parser.add_argument("--output", type=str, required=True, help="Target directory for output plots")
    parser.add_argument("--family", type=str, required=True, help="Experiment family name (e.g., SVM_Parity)")
    parser.add_argument("--timestamp", type=str, default=None, help="Filter by specific timestamp string")
    
    args = parser.parse_args()
    generate_sweep_summaries(args.csv, args.output, args.family, timestamp_filter=args.timestamp)