# utils/logger.py
import os
import csv
from datetime import datetime

def _repo_root_from_utils() -> str:
    # utils/ lives at src/utils/, so root is two levels up
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

#def log_results(metrics: dict, filename: str = "results.csv") -> str:
def log_results(metrics: dict, filename: str = "results/results.csv") -> str:
    """
    Append metrics to a single CSV at the repo root.
    Returns the absolute path to the written file.
    """
    root = _repo_root_from_utils()
    filepath = os.path.join(root, filename)

    # Ensure stable keys and add timestamp
    metrics = {**metrics, "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z"}

    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

    # Also echo to console (durable feedback for students)
    print("\n[log_results] wrote row to:", filepath)
    return filepath
