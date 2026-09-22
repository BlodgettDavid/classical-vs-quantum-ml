import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "results"
PLOTS_DIR = "plots"


def load_all_results(results_dir: str = RESULTS_DIR) -> pd.DataFrame:
    """Finds and concatenates all CSV benchmark logs into a unified DataFrame."""
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        print(f"[!] No CSV result files found in directory: '{results_dir}'")
        return pd.DataFrame()

    dfs = []
    for file in csv_files:
        # Exclude sub-group files to prevent duplicate entries
        if os.path.basename(file) in ["results_breast_cancer.csv", "results_parity.csv"]:
            continue
            
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
    return combined_df.rename(columns=column_mapping)


def plot_accuracy_vs_runtime(df: pd.DataFrame, output_dir: str = PLOTS_DIR):
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
    
    # Place legend strictly outside plot area
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., frameon=True)
    
    out_path = os.path.join(output_dir, "summary_accuracy_vs_runtime.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_generalization_gap(df: pd.DataFrame, output_dir: str = PLOTS_DIR):
    """Generates a bar chart visualizing Generalization Gap (Train Acc - Test Acc)."""
    if df.empty or "train_accuracy" not in df.columns or "test_accuracy" not in df.columns:
        return

    df_plot = df.copy()
    if "generalization_gap" in df_plot.columns:
        df_plot["gen_gap"] = df_plot["generalization_gap"]
    else:
        df_plot["gen_gap"] = df_plot["train_accuracy"] - df_plot["test_accuracy"]

    # Shorten multiline labels to prevent axis overlapping
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
    
    # Place legend strictly outside plot area
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., frameon=True)

    out_path = os.path.join(output_dir, "summary_generalization_gap.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_support_vectors(df: pd.DataFrame, output_dir: str = PLOTS_DIR):
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
    
    # Place legend strictly outside plot area
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., frameon=True)

    out_path = os.path.join(output_dir, "summary_support_vectors.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {out_path}")


def generate_all_summary_plots(results_dir="results", output_dir="plots"):
    """Main execution entry point."""
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_all_results(results_dir)
    if df.empty:
        print("[SUMMARY VISUALIZER] Skipping plot generation (no result CSVs found).")
        return

    plot_accuracy_vs_runtime(df, output_dir)
    plot_generalization_gap(df, output_dir)
    plot_support_vectors(df, output_dir)
    print("[SUMMARY VISUALIZER] All summary plots updated successfully.")


if __name__ == "__main__":
    generate_all_summary_plots()