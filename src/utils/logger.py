# src/utils/logger.py

import os
import csv
from datetime import datetime, timezone

# Defined master column order matching the repository results schema
FIELDNAMES = [
    "model", "dataset", "backend",
    "accuracy", "precision", "recall", "f1_score", "train_accuracy", "generalization_gap",
    "training_runtime", "prediction_runtime", "plotting_runtime",
    "feasibility", "memory_MB", "cpu_percent", "support_vectors",
    "TP", "FP", "TN", "FN",
    "kernel", "C", "gamma", "degree",
    "split_ratio", "random_state", "pca_components",
    "n_train_samples", "n_test_samples",
    "num_qubits", "circuit_depth", "feature_map", "reps", "entanglement",
    "timestamp"
]

def _repo_root_from_utils() -> str:
    """utils/ lives at src/utils/, so root is two levels up."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _get_dataset_group(dataset_name: str) -> str:
    """Categorizes raw dataset names into target CSV file groups."""
    dataset_str = str(dataset_name).lower()
    if "parity" in dataset_str:
        return "parity"
    elif "breast" in dataset_str or "cancer" in dataset_str:
        return "breast_cancer"
    else:
        return "other"


def log_results(metrics: dict, filename: str = "results/results.csv") -> str:
    """
    Append metrics to the main results CSV and automatically route a copy to
    a dataset-specific CSV (e.g., results/results_breast_cancer.csv).
    
    Uses FIELDNAMES to guarantee strict column alignment across all scripts.
    Returns the absolute path to the main written file.
    """
    root = _repo_root_from_utils()
    master_filepath = os.path.join(root, filename)

    # Ensure parent directory exists
    results_dir = os.path.dirname(master_filepath)
    os.makedirs(results_dir, exist_ok=True)

    # Ensure ISO UTC timestamp
    row_data = {
        **metrics,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ"),
    }

    # Determine dataset-specific filename
    dataset_group = _get_dataset_group(metrics.get("dataset", "other"))
    group_filepath = os.path.join(results_dir, f"results_{dataset_group}.csv")

    # Helper function to append row using DictWriter
    def _write_row(target_path: str):
        file_exists = os.path.isfile(target_path) and os.path.getsize(target_path) > 0
        with open(target_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=FIELDNAMES,
                extrasaction="ignore"  # Safely ignores unexpected auxiliary keys
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)

    # Write to master file and dataset-specific file
    _write_row(master_filepath)
    _write_row(group_filepath)

    print(f"\n[log_results] wrote row to master: {master_filepath}")
    print(f"[log_results] wrote row to group:  {group_filepath}")

    return master_filepath