# 🔬 Quantum vs Classical SVM Benchmarking Pipeline

This repository provides a clean, reproducible, end-to-end MLOps benchmarking pipeline comparing classical Support Vector Machines (SVM) with Quantum Support Vector Machines (QSVM) using Qiskit Machine Learning. The project is designed for students, educators, and researchers to explore quantum kernel methods, dataset structures (parity vs. real-world clinical data), feature mapping, and performance trade-offs under rigorous experimental conditions.

## 🎯 Project Goals & Regimes

This pipeline evaluates three primary regimes of quantum vs. classical performance:

1. Quantum Advantage / Challenge
   Evaluates non-linear parity structures (e.g., parity4d_stressed, parity6d_stressed) where classical RBF kernel SVMs struggle without specialized higher-order feature transformations.

2. Quantum Neutrality
   QSVM achieves comparable test accuracy to classical SVM after PCA reduces feature dimensionality on real-world datasets (e.g., breast_cancer.csv, wine_top6.csv).

3. Resource Trade-off (Quantum Disadvantage)
   QSVM exhibits significantly higher runtime latency (seconds to minutes vs. milliseconds) and varying support vector densities compared to classical SVM.

*All quantum evaluations execute on Qiskit statevector simulators—no physical quantum hardware is required.*

## 📁 Repository Structure

classical-vs-quantum-ml/
├── .devcontainer/
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── LICENSE
├── README.md
├── requirements.txt
├── config/
│   ├── classical_svm.yaml
│   └── quantum_svm.yaml
├── templates/
│   ├── classical_svm_breastcancer.yaml
│   ├── classical_svm_parity.yaml
│   ├── quantum_svm_breastcancer.yaml
│   └── quantum_svm_parity.yaml
├── data/
│   ├── breast_cancer.csv
│   ├── parity3d.csv
│   ├── parity4d.csv
│   ├── parity4d_stressed.csv
│   ├── parity6d.csv
│   ├── parity6d_stressed.csv
│   └── wine_top6.csv
├── plots/
│   ├── global_reports/
│   ├── runs/
│   └── sweeps/
├── results/
│   └── .gitkeep (Stores generated execution CSV logs)
├── src/
│   ├── __init__.py
│   ├── analysis/
│   ├── experiments/
│   ├── phase1/
│   ├── phase2/
│   ├── phase3/
│   └── utils/
└── tests/

## ⚙️ Environment Setup & Execution

To ensure complete environment parity and eliminate local dependency drift across Windows and Mac operating systems, this project utilizes **VS Code Dev Containers** powered by Docker.

### Option A: Recommended (VS Code Dev Containers)
1. Ensure you have Docker Desktop installed and running on your host machine (Windows or macOS).
2. Install the Dev Containers extension in VS Code.
3. Open the repository folder in VS Code. When prompted ("Folder contains a Dev Container configuration. Open in Container?"), click Reopen in Container.
   (Alternatively, press `F1` and select `Dev Containers: Reopen in Container`).
4. Once the bottom-left corner indicates "Dev Container: classical vs. quantum ml", open the integrated terminal (`Ctrl + \`` on Windows/Linux or `Cmd + \`` on Mac) which will be active directly inside the standardized Linux environment at `/app`.

### Option B: Manual Local Setup (Python 3.11)
If running locally without Docker, ensure Python 3.11 is installed to guarantee binary wheel compatibility with scipy, qiskit, and symengine:

git clone https://github.com/your-username/classical-vs-quantum-ml.git
cd classical-vs-quantum-ml

# Create and activate virtual environment
python -m venv .venv311
# On Windows: .venv311\Scripts\Activate
# On Mac/Linux: source .venv311/bin/activate

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

## ▶️ Execution & Workflow Commands

All scripts should be executed as modules from the root directory to preserve relative import paths.

### 1. Phase 1: Classical Single Runs
python -m src.phase1.SVM_Parity
python -m src.phase1.SVM_BreastCancer_PCA

### 2. Phase 2 & 3: Quantum Single Runs
python -m src.phase2.QSVM_Parity
python -m src.phase3.QSVM_BreastCancer_PCA

### 3. Hyperparameter Sweeps
Run multi-configuration grid searches across kernels, parameters, and feature maps:
python -m src.experiments.run_classical_parity_sweep
python -m src.experiments.run_classical_breast_cancer_pca_sweep
python -m src.experiments.run_quantum_parity_sweep
python -m src.experiments.run_quantun_breast_cancer_pca_sweep

### 4. Cross-Sweep & Global Aggregation Reports
Generate cross-experiment visualizations and high-level benchmarking figures across all completed runs:
python -m src.analysis.generate_sweep_comparisons
python -m src.utils.global_reports_visualizer

## 🧪 Schema Guardrails & Validation

To ensure configuration parsing and template interfaces function correctly, execute the integration test suite:

python test_config_integration.py
python test_templates.py
python -m tests.verify_csv_headers

## 🎓 Key Learning Outcomes

- Model Comparison: Compare classical and quantum kernel architectures under standardized evaluation metrics.
- Dimensionality Reduction: Analyze the impact of PCA pre-processing on quantum statevector simulation scalability.
- Modular Pipeline Architecture: Gain experience with config-driven experiment tracking, automated logging, and decoupled visualization layers.
- Containerized Parity: Experience reproducible development workflows using Docker and VS Code Dev Containers.