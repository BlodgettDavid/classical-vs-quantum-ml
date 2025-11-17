# Data Folder

This folder contains datasets used across hybrid QML phases.

## Input Datasets

- `breast_cancer.csv`: Real-world classification dataset from sklearn, used in Phase 1 and Phase 3.
- `xor_dataset.csv`: Synthetic binary classification dataset for Phase 1 and Phase 2.
- `parity_dataset.csv`: Synthetic parity dataset for Phase 1 and Phase 2.
- `parity4d_dataset.csv`: Extended parity dataset (4D version), used in Phase 2 and Phase 3.
- `parity4d_stressed.csv`: Variant of the 4D parity dataset with added noise/stress conditions.

## Generated Results (not versioned in Git)

- `results_SVM_Breast_Cancer.csv`: Output results from SVM experiments on breast cancer dataset.
- `results_SVM_Parity.csv`: Output results from SVM experiments on parity dataset.
- `results_SVM_Parity4D.csv`: Output results from SVM experiments on 4D parity dataset.
- `results_SVM_Parity4D_Stressed.csv`: Output results from SVM experiments on stressed 4D parity dataset.

## Notes

- All datasets are preprocessed (scaled, encoded) before use.
- Synthetic datasets are generated using `sklearn.datasets.make_classification()` or custom logic.
- Breast cancer data sourced from `sklearn.datasets.load_breast_cancer()` and exported to CSV.
