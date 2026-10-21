# 🔬 Quantum vs Classical SVM Benchmarking Pipeline

This repository provides a clean, reproducible, end-to-end MLOps benchmarking pipeline comparing classical Support Vector Machines (SVM) with Quantum Support Vector Machines (QSVM) using Qiskit Machine Learning. The project is designed for students, educators, and researchers to explore quantum kernel methods, dataset structures (parity vs. real-world clinical data), feature mapping, and performance trade-offs under rigorous experimental conditions.

## 🎯 Project Goals & Regimes

This pipeline evaluates three primary regimes of quantum vs. classical performance:

1. **Quantum Advantage / Challenge**
   Evaluates non-linear parity structures (e.g., parity4d_stressed, parity6d_stressed) where classical RBF kernel SVMs struggle without specialized higher-order feature transformations.

2. **Quantum Neutrality**
   QSVM achieves comparable test accuracy to classical SVM after PCA reduces feature dimensionality on real-world datasets (e.g., breast_cancer.csv, wine_top6.csv).

3. **Resource Trade-off (Quantum Disadvantage)**
   QSVM exhibits significantly higher runtime latency (seconds to minutes vs. milliseconds) and varying support vector densities compared to classical SVM.

*All quantum evaluations execute on Qiskit statevector simulators—no physical quantum hardware is required.*

## 📁 Repository Structure

classical-vs-quantum-ml/
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
│   │   ├── classical/
│   │   │   ├── SVM_BreastCancer_PCA/
│   │   │   └── SVM_Parity/
│   │   └── quantum/
│   │       ├── QSVM_BreastCancer_PCA/
│   │       └── QSVM_Parity/
│   ├── sweeps/
│   │   ├── classical/
│   │   │   ├── SVM_BreastCancer_PCA/
│   │   │   └── SVM_Parity/
│   │   └── quantum/
│   │       ├── QSVM_BreastCancer_PCA/
│   │       └── QSVM_Parity/
│   └── sweep_comparisons/
├── results/
│   └── .gitkeep (Stores generated execution CSV logs)
├── src/
│   ├── __init__.py
│   ├── analysis/
│   │   └── generate_sweep_comparisons.py
│   ├── experiments/
│   │   ├── run_classical_breast_cancer_pca_sweep.py
│   │   ├── run_classical_parity_sweep.py
│   │   ├── run_quantum_parity_sweep.py
│   │   └── run_quantun_breast_cancer_pca_sweep.py
│   ├── phase1/
│   │   ├── __init__.py
│   │   ├── SVM_BreastCancer_PCA.py
│   │   └── SVM_Parity.py
│   ├── phase2/
│   │   ├── __init__.py
│   │   └── QSVM_Parity.py
│   ├── phase3/
│   │   ├── __init__.py
│   │   └── QSVM_BreastCancer_PCA.py
│   └── utils/
│       ├── classical_evaluator.py
│       ├── classical_visualizer.py
│       ├── config_loader.py
│       ├── data_loader.py
│       ├── generate_datasets.py
│       ├── global_reports_visualizer.py
│       ├── logger.py
│       ├── quantum_evaluator.py
│       ├── quantum_visualizer.py
│       └── sweep_visualizer.py
└── tests/
    └── verify_csv_headers.py

## ⚙️ Environment Setup

This project uses Python 3.11 to ensure full binary wheel compatibility with scipy, qiskit, and symengine.

### 1. Clone the repository

git clone https://github.com/your-username/classical-vs-quantum-ml.git
cd classical-vs-quantum-ml

### 2. Create and activate a Python 3.11 Virtual Environment

C:\Users\<YourUser>\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv311
.venv311\Scripts\Activate

### 3. Install dependencies

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

To ensure generated result CSV files adhere strictly to the target schema defined in logger.py, execute the verification suite:

python -m tests.verify_csv_headers

## 🐳 Docker Deployment (Future Iterations)

Container configurations are provided in the root directory (Dockerfile, docker-compose.yml, .dockerignore). They run on python:3.11-slim and mount local data and plot paths into container volumes.

## 🎓 Key Learning Outcomes

- Model Comparison: Compare classical and quantum kernel architectures under standardized evaluation metrics.
- Dimensionality Reduction: Analyze the impact of PCA pre-processing on quantum statevector simulation scalability.
- Modular Pipeline Architecture: Gain experience with config-driven experiment tracking, automated logging, and decoupled visualization layers.