# 🔬 Quantum vs. Classical SVM Benchmarking with Qiskit

This repository provides a clean, reproducible benchmarking pipeline comparing classical Support Vector Machines (SVM) to Quantum Support Vector Machines (QSVM) using Qiskit Machine Learning. The project is designed for teaching and research environments where students explore when quantum models help, when they fail, and how preprocessing affects performance.

---

## 📦 Pipeline Overview

The project includes:

- **Classical SVM baselines** (Parity datasets and Breast Cancer dataset)
- **Quantum SVM experiments** using Qiskit’s Sampler-based `QSVC`
- **PCA and non-PCA comparisons** to demonstrate feasibility limits
- **Unified logging** to `results.csv`
- **Reproducible plots** saved to the `plots/` directory
- **Clear separation** between student-facing code and instructor-only assets

---

## 🎯 Project Goals

- Demonstrate the strengths and weaknesses of QSVMs on real datasets
- Show how PCA affects quantum kernel feasibility
- Provide a transparent, reproducible benchmark pipeline for students
- Stay fully simulator-based to avoid hardware costs or queue delays

---

## 📁 Repository Structure

```text
src/
├── phase1/        # Classical SVM experiments
├── phase2/        # QSVM experiments on Parity datasets
├── phase3/        # QSVM experiments on Breast Cancer dataset
└── utils/         # Shared loaders, loggers, visualizers, and dataset tools

config/            # YAML configuration files for dataset selection
data/              # Public datasets used by the pipeline
plots/             # Generated visualizations (ignored in .gitignore)
results.csv        # Logged experiment results (ignored in .gitignore)
instructor_only/   # Private instructor assets (ignored in .gitignore)
```

---

## ▶️ Running Experiments

Each experiment is executed as a Python module:

```bash
python -m src.phase1.SVM_Parity
python -m src.phase2.QSVM_Parity
python -m src.phase1.SVM_BreastCancer
python -m src.phase1.SVM_BreastCancer_PCA
python -m src.phase3.QSVM_BreastCancer
python -m src.phase3.QSVM_BreastCancer_PCA
```

---

## 🛠️ Environment Setup (Windows)

**1. Clone the repository:**
```powershell
git clone [https://github.com/your-username/qiskitdev.git](https://github.com/your-username/qiskitdev.git)
cd qiskitdev
```

**2. Create and activate a virtual environment:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**3. Install dependencies:**
```powershell
pip install -r requirements.txt
```

---

## 🎓 Key Learning Outcomes

- Understand why QSVMs fail on high-dimensional data
- Observe how PCA enables QSVM feasibility (but not necessarily accuracy)
- Compare classical and quantum models on equal footing
- Learn reproducible ML/QML experiment design

---

*This repository is intended for educational use and is fully simulator-based. **No quantum hardware access is required.***