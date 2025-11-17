# src/utils/logger.py

import os
import csv
import psutil
import time

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

def log_results(metrics):
    # Dynamically name results file based on model label
    label = metrics["model"]
    filename = f"results_{label.replace(' ', '_')}.csv"
    
    RESULTS_PATH = os.path.join(ROOT_DIR, "..", "..", "data", filename)

    # Add memory usage
    process = psutil.Process(os.getpid())
    memory_MB = process.memory_info().rss / 1024 / 1024
    metrics["memory_MB"] = round(memory_MB, 2)

    # Add backend info if missing
    if "backend" not in metrics:
        metrics["backend"] = "N/A"

    # Write header if file doesn't exist
    write_header = not os.path.exists(RESULTS_PATH)
    with open(RESULTS_PATH, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "model", "accuracy", "train_accuracy", "generalization_gap",
            "training_time", "memory_MB", "backend"
        ])
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)

    print(f"✅ Logged results to: {os.path.abspath(RESULTS_PATH)}")
