"""
KNN_BreastCancer.py
Phase 1: Classical kNN baseline on Breast Cancer dataset
"""

import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from src.utils import data_loader, classical_evaluator, classical_visualizer, config_loader, logger

def main():
    # 1. Load configuration
    config = config_loader.load_config("config/classical_knn.yaml")

    # 2. Load dataset (DataFrame + config)
    df, cfg = data_loader.load_dataset_from_config()

    # 3. Split into train/test
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["dataset"]["test_size"],
        random_state=config["dataset"]["random_state"]
    )

    # 4. Train classical kNN
    knn = KNeighborsClassifier(
        n_neighbors=config["model"]["n_neighbors"],
        weights=config["model"]["weights"],
        metric=config["model"]["metric"]
    )
    knn.fit(X_train, y_train)

    # 5. Evaluate
    results = classical_evaluator.evaluate_model(
        knn,
        X_train, y_train,
        X_test, y_test,
        label="Classical kNN"
    )

    # 6. Log results
    log_entry = {
        "model_name": "classical_knn",
        "dataset": config["dataset"]["name"],
        **results
    }
    logger.log_results(log_entry)

    # 7. Visualize (optional)
    classical_visualizer.plot_confusion_matrix(
        y_test,
        knn.predict(X_test),
        title="Classical kNN - Breast Cancer"
    )

if __name__ == "__main__":
    main()