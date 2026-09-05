# src/utils/data_loader.py
import os
import pandas as pd
from src.utils.config_loader import load_config

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")

def dataset_to_filename(name: str) -> str:
    """Map dataset key to actual CSV filename dynamically or via explicit overrides."""
    # Append .csv if not present
    filename = name if name.endswith(".csv") else f"{name}.csv"
    csv_path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(csv_path):
        available = list_available_datasets()
        raise FileNotFoundError(
            f"Dataset '{name}' not found at {csv_path}. Available datasets: {available}"
        )
    return csv_path

def load_dataset_from_config(config_name="classical_svm.yaml"):
    """
    Loads dataset specified in the targeted config file.
    Returns (DataFrame, config dict).
    """
    cfg = load_config(config_name)
    ds_key = cfg.get("dataset", "breast_cancer")
    csv_path = dataset_to_filename(ds_key)
        
    df = pd.read_csv(csv_path)
    return df, cfg

def list_available_datasets():
    """List all CSV files currently available in the data directory."""
    if not os.path.exists(DATA_DIR):
        return []
    return [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]