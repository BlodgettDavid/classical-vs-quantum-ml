# src/utils/data_loader.py

import os
import pandas as pd

# Resolve absolute path to the data folder
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

def load_dataset(name):
    """
    Load a dataset by name (without .csv extension).
    Example: load_dataset("xor_dataset") → returns DataFrame
    """
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)

def list_available_datasets():
    """
    List all CSV files available in the data directory.
    """
    return [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
