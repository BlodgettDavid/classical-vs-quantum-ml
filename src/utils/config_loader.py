# src/utils/config_loader.py
import os
import yaml

def load_config(config_name="classical_svm.yaml", dataset_key=None):
    """
    Loads a configuration YAML file from the config/ directory.
    
    Dynamically routes dataset keys through 'classical_svm', 'datasets', or top-level.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    if not config_name.endswith((".yaml", ".yml")):
        config_name += ".yaml"
        
    if os.path.isabs(config_name):
        config_path = config_name
    else:
        config_path = os.path.join(repo_root, "config", config_name)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    if dataset_key:
        # Check 'datasets' section first, then 'classical_svm', then fallback to root
        section = (
            full_config.get("datasets") 
            or full_config.get("classical_svm") 
            or full_config
        )
        if dataset_key not in section:
            raise KeyError(f"Dataset key '{dataset_key}' not found in configuration file '{config_name}'.")
        return section[dataset_key]

    return full_config