# test_templates.py
from src.utils.config_loader import load_config

def test_template_loading():
    # Paths are relative to the 'config/' directory as handled by load_config()
    templates = [
        "templates/classical_svm_parity.yaml",
        "templates/classical_svm_breastcancer.yaml",
        "templates/quantum_svm_parity.yaml",
        "templates/quantum_svm_breastcancer.yaml"
    ]

    print("--- Verifying CS 402 Experiment Templates ---")
    for template in templates:
        try:
            cfg = load_config(template)
            dataset = cfg.get("dataset_name")
            print(f" [PASS] Successfully parsed: config/{template}")
            print(f"        Target Dataset -> {dataset}")
        except Exception as e:
            print(f" [FAIL] Error parsing {template}: {e}")
            raise

    print("---------------------------------------------")
    print("All four templates loaded and validated successfully!")

if __name__ == "__main__":
    test_template_loading()