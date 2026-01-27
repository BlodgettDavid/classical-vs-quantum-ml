# 🔬 Quantum vs Classical SVM Benchmarking (Windows)

This repository provides a clean, reproducible benchmarking pipeline comparing classical Support Vector Machines (SVM) with Quantum Support Vector Machines (QSVM) using Qiskit Machine Learning. The project is designed for students, educators, and researchers who want to understand when quantum kernel methods help, when they fail, and how preprocessing such as PCA affects feasibility.

---

## 🎯 Purpose of the Project

This project demonstrates three regimes of quantum machine learning performance:

1. **Quantum Advantage**  
   QSVM outperforms classical SVM on structured parity datasets.

2. **Quantum Neutrality**  
   QSVM performs similarly to classical SVM after PCA reduces dimensionality.

3. **Quantum Disadvantage**  
   QSVM becomes infeasible or unstable on high-dimensional real-world data without preprocessing.

All experiments run entirely on simulators. No quantum hardware access is required.

---

## 📁 Repository Structure

\`\`\`
src/
  phase1/   Classical SVM experiments (Parity, Breast Cancer, Breast Cancer PCA)
  phase2/   QSVM experiments on Parity datasets
  phase3/   QSVM experiments on Breast Cancer datasets
  utils/    Shared loaders, evaluators, loggers, and visualization tools

config/     YAML configuration for dataset selection
data/       Public datasets (parity and breast cancer)
plots/      Auto-generated plots (ignored by git)
results.csv Logged experiment results
\`\`\`

---

## ▶️ Running the Software (Windows)

### 1. Clone the repository
\`\`\`cmd
git clone https://github.com/your-username/classical-vs-quantum-svm.git
cd classical-vs-quantum-svm
\`\`\`

### 2. Create and activate a virtual environment
\`\`\`cmd
python -m venv .venv
.venv\Scripts\activate
\`\`\`

### 3. Install dependencies
\`\`\`cmd
pip install -r requirements.txt
\`\`\`

### 4. Run experiments as Python modules
\`\`\`cmd
python -m src.phase1.SVM_Parity
python -m src.phase2.QSVM_Parity
python -m src.phase1.SVM_BreastCancer
python -m src.phase1.SVM_BreastCancer_PCA
python -m src.phase3.QSVM_BreastCancer
python -m src.phase3.QSVM_BreastCancer_PCA
\`\`\`

---

## 📊 Summary of Results

### **Parity Datasets (4D, 4D stressed, 6D, 6D stressed)**
- Classical SVM struggles.
- QSVM performs significantly better.
- This is a clear example of **quantum advantage**.

### **Breast Cancer Dataset (No PCA)**
- Classical SVM performs well.
- QSVM is not feasible due to dimensionality.
- This demonstrates **quantum disadvantage**.

### **Breast Cancer Dataset (With PCA)**
- Both classical SVM and QSVM run successfully.
- Accuracy is comparable.
- This demonstrates **quantum neutrality**.

---

## 🎓 Key Learning Outcomes

- Understand when quantum kernels help and when they fail.
- See how PCA enables QSVM feasibility on real datasets.
- Compare classical and quantum models on equal footing.
- Learn reproducible ML and QML experiment design.
- Run all experiments on Windows using Qiskit simulators.

---

This project is intended for educational use and is fully simulator-based.