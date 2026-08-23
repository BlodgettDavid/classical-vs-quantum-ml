# src/utils/config_loader.py
import os
import yaml

def load_config(config_path=None):
    """
    Loads the central config.yaml file and returns it as a dict.
    Default path: project_root/config/config.yaml
    """
    if config_path is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        config_path = os.path.join(repo_root, "config", "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config