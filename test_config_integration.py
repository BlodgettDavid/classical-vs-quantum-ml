# test_config_integration.py
from src.utils.config_loader import load_config

def verify_template_interface():
    print("--- Testing config_loader.py Interface Integration ---\n")

    # 1. Test Classical Parity Template
    cfg_cl_parity = load_config("templates/classical_svm_parity.yaml")
    assert cfg_cl_parity["dataset_name"] == "parity4d_stressed"
    assert cfg_cl_parity["svm_params"]["kernel"] == "rbf"
    print(" [OK] Classical Parity template interface verified.")

    # 2. Test Classical Breast Cancer Template
    cfg_cl_bc = load_config("templates/classical_svm_breastcancer.yaml")
    assert cfg_cl_bc["dataset_name"] == "breast_cancer"
    assert cfg_cl_bc["svm_params"]["kernel"] == "linear"
    print(" [OK] Classical Breast Cancer template interface verified.")

    # 3. Test Quantum Parity Template
    cfg_q_parity = load_config("templates/quantum_svm_parity.yaml")
    assert cfg_q_parity["dataset_name"] == "parity4d_stressed"
    assert cfg_q_parity["quantum_params"]["feature_map"] == "ZZFeatureMap"
    print(" [OK] Quantum Parity template interface verified.")

    # 4. Test Quantum Breast Cancer Template
    cfg_q_bc = load_config("templates/quantum_svm_breastcancer.yaml")
    assert cfg_q_bc["dataset_name"] == "breast_cancer"
    assert cfg_q_bc["quantum_params"]["feature_map"] == "PauliFeatureMap"
    print(" [OK] Quantum Breast Cancer template interface verified.")

    print("\n----------------------------------------------------")
    print("All templates successfully interfaced via config_loader.py!")

if __name__ == "__main__":
    verify_template_interface()