# 🔬 Quantum vs Classical SVM Benchmarking (Windows)

This repository provides a clean, reproducible benchmarking pipeline comparing classical Support Vector Machines (SVM) with Quantum Support Vector Machines (QSVM) using Qiskit Machine Learning. The project is designed for students, educators, and researchers who want to understand when quantum kernel methods help, when they fail, and how preprocessing such as PCA affects feasibility.

---

## 🎯 Purpose of the Project

This project demonstrates three regimes of quantum machine learning performance:

1. **Quantum Advantage / Challenge**  
   Evaluates non-linear parity structures (e.g., `parity4d_stressed`) where classical RBF kernel SVMs fail completely without specific higher-order feature maps.

2. **Quantum Neutrality**  
   QSVM achieves comparable test accuracy to classical SVM after PCA reduces feature dimensionality on real-world datasets.

3. **Quantum Disadvantage (Resource Trade-off)**  
   QSVM exhibits significantly higher runtime latency (5-8 seconds vs sub-second) and requires a high density of support vectors compared to classical SVM.

All experiments run entirely on Qiskit statevector simulators. No quantum hardware access is required.

---

## 📁 Repository Structure

src/
  phase1/     Classical SVM experiments (Parity, Breast Cancer PCA)
  phase2/     QSVM experiments on Parity datasets
  phase3/     QSVM experiments on Breast Cancer datasets (PCA)
  utils/      Shared loaders, evaluators, loggers, and visualization tools
              ├─ classical_evaluator.py & classical_visualizer.py
              ├─ quantum_evaluator.py & quantum_visualizer.py
              └─ summary_visualizer.py (Aggregated metric visualizer)

config/       YAML configurations (classical_svm.yaml, quantum_svm.yaml)
data/         Public and synthetic datasets (parity and breast cancer)
plots/        Auto-generated decision boundaries, confusion matrices, and summary plots
results/      Logged experiment CSV results (results.csv, dataset-specific CSVs)
tests/        CSV header schema validation scripts

---

## ▶️ Running the Software (Windows)

### 1. Clone the repository
git clone https://github.com/your-username/classical-vs-quantum-svm.git
cd classical-vs-quantum-svm

### 2. Activate virtual environment
.venv311\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Run experiments as Python modules
python -m src.phase1.SVM_Parity
python -m src.phase2.QSVM_Parity
python -m src.phase1.SVM_BreastCancer_PCA
python -m src.phase3.QSVM_BreastCancer_PCA

### 5. Run aggregated summary visualizer
python -m src.utils.summary_visualizer

---

## 📈 Summary Visualizations

To visualize aggregated results across experiment runs, run the summary visualizer:

python -m src.utils.summary_visualizer

This tool reads results/results.csv and outputs comparative figures directly to plots/:
- summary_accuracy_vs_runtime.png (Accuracy vs. Runtime on log scale)
- summary_generalization_gap.png (Train vs. Test accuracy difference)
- summary_support_vectors.png (Support vector count across configurations)

---

## ⚙ Config-driven Experiments

Experiments are managed via separate YAML configuration files:
- config/classical_svm.yaml — Classical SVM kernels, gamma settings, C parameters, and train/test splits.
- config/quantum_svm.yaml — Quantum feature map selection (ZZFeatureMap, PauliFeatureMap), repetitions, entanglement schemes, and simulator backends.

---

## 🧩 Evaluators & Visualizers

- utils/classical_evaluator.py & utils/classical_visualizer.py → Metrics, decision boundaries, and confusion matrices for classical SVM.
- utils/quantum_evaluator.py & utils/quantum_visualizer.py → Metrics, quantum kernel heatmaps, and confusion matrices for QSVM.
- utils/logger.py → Standardized schema enforcement writing to results/ CSV files.
- utils/summary_visualizer.py → Standalone summary visualizer.

---

## 🧪 Tests & Schema Guardrails

Run schema validation to verify CSV result alignment with the active logger definitions:

python -m tests.verify_csv_headers

---

## 🎓 Key Learning Outcomes

- Compare classical and quantum kernel models on equal footing.
- Observe trade-offs between Hilbert space feature maps and execution runtime.
- Understand how dimensionality reduction (PCA) impacts quantum simulation feasibility.
- Maintain clean, modular experiment structures and config-driven pipelines.

---

This project is intended for educational use and is fully simulator-based.