# src/utils/data_loader.py
# -------------------------------------------------
# Centralized dataset loader for benchmarks.
# Reads dataset choice from config.yaml and loads the corresponding CSV.
# Supports breast cancer and parity datasets (4D, 6D, clean + stressed).
# -------------------------------------------------

import os
import yaml
import pandas as pd

# Resolve absolute paths (repo root)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "config.yaml")

def load_config():
    """Load experiment configuration from config.yaml."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def dataset_to_filename(name: str) -> str:
    """Map dataset key from config.yaml to actual CSV filename."""
    mapping = {
        "breast_cancer": "breast_cancer.csv",
        "parity4d": "parity4d.csv",
        "parity4d_stressed": "parity4d_stressed.csv",
        "parity6d": "parity6d.csv",
        "parity6d_stressed": "parity6d_stressed.csv",
    }
    if name not in mapping:
        raise ValueError(f"Unknown dataset '{name}'. Valid keys: {list(mapping.keys())}")
    return os.path.join(DATA_DIR, mapping[name])

def load_dataset_from_config():
    """
    Load dataset specified in config.yaml.
    Returns (DataFrame, config dict).
    """
    cfg = load_config()
    ds_key = cfg.get("dataset", "parity6d_stressed")
    csv_path = dataset_to_filename(ds_key)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}. Generate it first.")
    df = pd.read_csv(csv_path)
    return df, cfg

def list_available_datasets():
    """List all CSV files currently available in the data directory."""
    return [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
