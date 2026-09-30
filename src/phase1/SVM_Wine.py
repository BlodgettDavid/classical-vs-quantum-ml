# src/phase1/SVM_Wine.py
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from src.utils.config_loader import load_config
from src.utils.data_loader import load_dataset_by_key
from src.utils.classical_evaluator import evaluate_model
from src.utils.classical_visualizer import plot_projected_decision_boundary, plot_confusion_matrix
from src.utils.logger import log_results

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def run_classical_wine_experiment():
    print("=" * 60)
    print("   PHASE 1: Classical SVM Baseline - Wine Top 6 Dataset")
    print("=" * 60)

    # 1. Load configuration for classical_svm
    config = load_config("classical_svm.yaml")
    wine_cfg = config.get("classical_svm", {}).get("wine_top6", {})
    
    dataset_key = wine_cfg.get("dataset", "wine_top6")
    split_ratio = wine_cfg.get("split_ratio", 0.3)
    random_state = wine_cfg.get("random_state", 42)
    scale_data = wine_cfg.get("scale_data", True)
    pca_components_list = wine_cfg.get("pca_components", [0, 2])
    
    model_params = wine_cfg.get("model_params", {})
    kernel_type = model_params.get("kernel", "rbf")
    c_val = model_params.get("C", 1.0)
    gamma_val = model_params.get("gamma", "scale")
    degree_val = model_params.get("degree", 0)

    vis_cfg = wine_cfg.get("visualization", {})
    output_dir = os.path.join(ROOT_DIR, vis_cfg.get("output_dir", "plots"))

    # 2. Load dataset via helper utility
    print(f"\n[+] Loading dataset '{dataset_key}'...")
    X, y = load_dataset_by_key(dataset_key)
    print(f"    Raw features shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")

    # Iterate over configured PCA dimensions (0 = raw 6D features, >0 = PCA reduced)
    for pca_comp in pca_components_list:
        exp_label = f"Wine (Raw 6D)" if pca_comp == 0 else f"Wine (PCA {pca_comp}D)"
        print(f"\n---> Running Experiment: {exp_label}")

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split_ratio, random_state=random_state, stratify=y
        )

        # Preprocessing: Standard Scaling
        scaler = StandardScaler()
        if scale_data:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values

        # Optional Dimensionality Reduction
        if pca_comp > 0:
            pca = PCA(n_components=pca_comp, random_state=random_state)
            X_train_proc = pca.fit_transform(X_train_scaled)
            X_test_proc = pca.transform(X_test_scaled)
            explained_var = sum(pca.explained_variance_ratio_) * 100
            print(f"    PCA applied ({pca_comp} components). Variance retained: {explained_var:.2f}%")
        else:
            X_train_proc = X_train_scaled
            X_test_proc = X_test_scaled

        # Initialize Classical SVM
        clf = SVC(
            kernel=kernel_type,
            C=c_val,
            gamma=gamma_val,
            degree=degree_val,
            random_state=random_state
        )

        # Evaluate performance metrics using evaluate_model
        metrics = evaluate_model(
            clf, X_train_proc, y_train, X_test_proc, y_test, label=exp_label
        )
        
        print(f"    Results for {exp_label}:")
        print(f"      - Accuracy:  {metrics['accuracy']:.4f}")
        print(f"      - Precision: {metrics['precision']:.4f}")
        print(f"      - Recall:    {metrics['recall']:.4f}")
        print(f"      - F1 Score:  {metrics['f1_score']:.4f}")
        print(f"      - Fit Time:  {metrics['training_runtime']:.4f} sec")

        # Fully mapped log payload matching results.csv columns
        log_payload = {
            "model": f"Classical SVM ({kernel_type.upper()})",
            "dataset": f"{dataset_key}_pca{pca_comp}" if pca_comp > 0 else dataset_key,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "train_accuracy": metrics.get("train_accuracy", 0.0),
            "generalization_gap": metrics.get("generalization_gap", 0.0),
            "training_runtime": metrics["training_runtime"],
            "prediction_runtime": metrics.get("prediction_runtime", 0.0),
            "support_vectors": metrics.get("support_vectors", 0),
            "kernel": kernel_type,
            "C": c_val,
            "gamma": gamma_val,
            "pca_components": pca_comp
        }
        log_results(log_payload)
        print("    [✓] Results successfully logged to results/results.csv")

        # Predictions for confusion matrix visualizer
        y_pred = clf.predict(X_test_proc)

        # Visualizations
        if vis_cfg.get("plot_confusion_matrix", True):
            cm_filename = f"cm_classical_wine_pca{pca_comp}.png" if pca_comp > 0 else "cm_classical_wine_raw.png"
            plot_confusion_matrix(
                y_true=y_test, 
                y_pred=y_pred,
                title=f"Classical SVM Confusion Matrix - {exp_label}",
                save=True,
                show=False,
                filename=cm_filename
            )

        if vis_cfg.get("plot_decision_boundary", True) and pca_comp == 2:
            boundary_filename = f"decision_boundary_classical_wine_pca2.png"
            plot_projected_decision_boundary(
                model=clf,
                X=X_train_proc,
                y=y_train,
                title=f"Classical SVM Decision Boundary - Wine (PCA 2D)",
                save=True,
                show=False,
                filename=boundary_filename,
                do_pca=False
            )

    print("\n[+] Phase 1 Wine experiments complete.\n")

if __name__ == "__main__":
    run_classical_wine_experiment()