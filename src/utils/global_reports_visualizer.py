import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dynamically compute project root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
PLOTS_ROOT = os.path.join(ROOT_DIR, "plots", "global_reports")


def load_targeted_results(
    results_dir: str = RESULTS_DIR, 
    dataset_filter: str = None, 
    model_filter: str = None, 
    timestamp_filter: str = None
) -> pd.DataFrame:
    """Finds, concatenates, and filters CSV benchmark logs based on explicit user criteria."""
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        print(f"[!] No CSV result files found in directory: '{results_dir}'")
        return pd.DataFrame()

    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if not df.empty and ("dataset" in df.columns or "dataset_name" in df.columns):
                dfs.append(df)
        except Exception as e:
            print(f"[!] Error reading {file}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined_df = pd.concat(dfs, ignore_index=True)

    # Standardize column mappings
    column_mapping = {
        "dataset": "dataset_name",
        "model": "model_type",
        "accuracy": "test_accuracy",
        "training_runtime": "train_time_sec",
        "support_vectors": "num_support_vectors",
    }
    df_clean = combined_df.rename(columns=column_mapping)

    # Handle timestamp mapping
    if "timestamp" in df_clean.columns:
        df_clean["timestamp_str"] = df_clean["timestamp"].astype(str)
        df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"], format="ISO8601", errors="coerce")

    # --- FINE-GRAINED CLI FILTERING ---
    if dataset_filter:
        df_clean = df_clean[df_clean["dataset_name"].str.contains(dataset_filter, case=False, na=False)]
    
    if model_filter:
        df_clean = df_clean[df_clean["model_type"].str.contains(model_filter, case=False, na=False)]

    if timestamp_filter:
        # Isolate exactly the user-specified run batch
        df_clean = df_clean[df_clean["timestamp_str"].str.contains(timestamp_filter, na=False)]
    elif "timestamp" in df_clean.columns:
        # Fallback default if no timestamp is provided: keep the latest run per configuration
        df_clean = df_clean.sort_values("timestamp").groupby(["dataset_name", "model_type"], as_index=False).last()

    return df_clean


def plot_accuracy_vs_runtime(df: pd.DataFrame, output_dir: str):
    """Generates a scatter plot comparing Test Accuracy vs Execution Runtime."""
    if df.empty or "test_accuracy" not in df.columns:
        return

    df_plot = df.copy()
    runtime_series = df_plot.get("train_time_sec", df_plot.get("kernel_time_sec", pd.Series(0, index=df_plot.index)))
    df_plot["runtime_plot"] = runtime_series.clip(lower=1e-4)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df_plot,
        x="runtime_plot",
        y="test_accuracy",
        hue="model_type",
        style="dataset_name",
        s=130,
        alpha=0.9,
        ax=ax
    )

    ax.set_xscale("log")
    ax.set_title("Model Tradeoff: Test Accuracy vs. Runtime (Log Scale)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Execution / Training Runtime (Seconds, Log Scale)", fontsize=11)
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., frameon=True)
    
    out_path = os.path.join(output_dir, "summary_accuracy_vs_runtime.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_generalization_gap(df: pd.DataFrame, output_dir: str):
    """Generates a bar chart visualizing Generalization Gap (Train Acc - Test Acc)."""
    if df.empty or "train_accuracy" not in df.columns or "test_accuracy" not in df.columns:
        return

    df_plot = df.copy()
    if "generalization_gap" in df_plot.columns:
        df_plot["gen_gap"] = df_plot["generalization_gap"]
    else:
        df_plot["gen_gap"] = df_plot["train_accuracy"] - df_plot["test_accuracy"]

    df_plot["short_label"] = df_plot["dataset_name"].astype(str) + "\n" + df_plot["model_type"].astype(str)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=df_plot,
        x="short_label",
        y="gen_gap",
        hue="model_type",
        dodge=False,
        ax=ax
    )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Generalization Gap Across Models & Datasets (Train Acc - Test Acc)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Dataset & Model Configuration", fontsize=11)
    ax.set_ylabel("Generalization Gap", fontsize=11)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., frameon=True)

    out_path = os.path.join(output_dir, "summary_generalization_gap.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_support_vectors(df: pd.DataFrame, output_dir: str):
    """Generates a bar chart comparing the number of Support Vectors per model."""
    if df.empty or "num_support_vectors" not in df.columns:
        return

    df_plot = df.copy()
    df_plot["short_label"] = df_plot["dataset_name"].astype(str) + "\n" + df_plot["model_type"].astype(str)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=df_plot,
        x="short_label",
        y="num_support_vectors",
        hue="model_type",
        dodge=False,
        ax=ax
    )

    ax.set_title("Support Vector Count Comparison Across Runs", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Dataset & Model Configuration", fontsize=11)
    ax.set_ylabel("Number of Support Vectors", fontsize=11)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., frameon=True)

    out_path = os.path.join(output_dir, "summary_support_vectors.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {out_path}")


def generate_targeted_summary(dataset_filter=None, model_filter=None, timestamp_filter=None):
    """Main execution entry point with argument-driven filtering and dynamic subfolder routing."""
    # Mirror the sweep layout: plots/summaries/<timestamp>/
    run_timestamp = timestamp_filter if timestamp_filter else pd.Timestamp.now().strftime("%Y-%m-%dT%H-%M-%SZ")
    output_dir = os.path.join(PLOTS_ROOT, run_timestamp)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_targeted_results(RESULTS_DIR, dataset_filter, model_filter, timestamp_filter)
    if df.empty:
        print("[SUMMARY VISUALIZER] No matching data found for the given filters. Skipping plot generation.")
        return

    plot_accuracy_vs_runtime(df, output_dir)
    plot_generalization_gap(df, output_dir)
    plot_support_vectors(df, output_dir)
    print(f"[+] Fine-grained summary plots successfully saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-grained Targeted MLOps Summary Visualizer")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by dataset name (e.g., parity)")
    parser.add_argument("--model", type=str, default=None, help="Filter by model type (e.g., QSVM or SVM)")
    parser.add_argument("--timestamp", type=str, default=None, help="Target a specific run timestamp batch")
    
    args = parser.parse_args()
    generate_targeted_summary(
        dataset_filter=args.dataset, 
        model_filter=args.model, 
        timestamp_filter=args.timestamp
    )