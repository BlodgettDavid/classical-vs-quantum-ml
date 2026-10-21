# src/utils/quantum_visualizer.py
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for batch execution
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


def _repo_root_from_utils() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _sanitize_filename(title: str) -> str:
    return (
        title.lower()
             .replace("(", "")
             .replace(")", "")
             .replace(":", "")
             .replace("/", "_")
             .replace(" ", "_")
             + ".png"
    )


def plot_quantum_kernel_matrix(
    kernel_matrix: np.ndarray,
    title: str = "Quantum Kernel Matrix",
    save: bool = True,
    show: bool = True,
    filename: str | None = None,
    output_dir: str | None = None
) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(kernel_matrix, interpolation="nearest", cmap="viridis")
    fig.colorbar(cax, ax=ax)
    ax.set_title(title)
    fig.tight_layout()

    saved_path = ""
    if save:
        root = _repo_root_from_utils()
        target_dir = output_dir if output_dir else os.path.join(root, "plots", "sweeps")

        os.makedirs(target_dir, exist_ok=True)
        fname = filename if filename else _sanitize_filename(title)
        saved_path = os.path.join(target_dir, fname)
        fig.savefig(saved_path, dpi=120)
        print(f"[quantum_visualizer] saved kernel matrix to: {saved_path}")

    if show:
        plt.show()

    plt.close(fig)

    return saved_path


def plot_quantum_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Quantum Confusion Matrix",
    save: bool = True,
    show: bool = True,
    filename: str | None = None,
    output_dir: str | None = None
) -> str:
    """
    Renders a 2x2 categorical confusion matrix heatmap with cell count annotations.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(5, 4.5))
    cax = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(cax, ax=ax)

    total = np.sum(cm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = (count / total * 100) if total > 0 else 0
            ax.text(
                j, i, f"{count}\n({pct:.1f}%)",
                ha="center", va="center",
                color="white" if cm[i, j] > (cm.max() / 2.0) else "black",
                fontsize=11, fontweight="bold"
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Class 0", "Class 1"])
    ax.set_yticklabels(["Class 0", "Class 1"])
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title(title)
    fig.tight_layout()

    saved_path = ""
    if save:
        root = _repo_root_from_utils()
        target_dir = output_dir if output_dir else os.path.join(root, "plots", "sweeps")

        os.makedirs(target_dir, exist_ok=True)
        fname = filename if filename else _sanitize_filename(title)
        saved_path = os.path.join(target_dir, fname)
        fig.savefig(saved_path, dpi=120)
        print(f"[quantum_visualizer] saved confusion matrix to: {saved_path}")

    if show:
        plt.show()

    plt.close(fig)

    return saved_path


def plot_qsvm_decision_boundary(
    model=None,
    X: np.ndarray = None,
    y: np.ndarray = None,
    title: str = "Quantum SVM Decision Boundary",
    save: bool = True,
    show: bool = True,
    filename: str | None = None,
    output_dir: str | None = None,
    qsvc=None,
    do_pca: bool = False,
    grid_steps: int = 50,
    **kwargs
) -> str:
    """
    Renders the decision boundary for a quantum model/classifier on 2D feature data (e.g., PCA components).
    """
    # Fallback if qsvc is passed instead of model
    classifier = model if model is not None else qsvc

    fig, ax = plt.subplots(figsize=(7, 6))

    # Determine meshgrid boundaries with padding
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.05),
        np.arange(y_min, y_max, 0.05)
    )

    # Evaluate the classifier across the grid
    try:
        Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    except Exception:
        Z = np.zeros(xx.ravel().shape)

    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    # Scatter plot of actual data points
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.coolwarm, s=40)
    ax.set_title(title)
    ax.set_xlabel("PCA Feature 1", fontweight="bold")
    ax.set_ylabel("PCA Feature 2", fontweight="bold")
    fig.tight_layout()

    saved_path = ""
    if save:
        root = _repo_root_from_utils()
        target_dir = output_dir if output_dir else os.path.join(root, "plots", "sweeps")

        os.makedirs(target_dir, exist_ok=True)
        fname = filename if filename else _sanitize_filename(title)
        saved_path = os.path.join(target_dir, fname)
        fig.savefig(saved_path, dpi=120)
        print(f"[quantum_visualizer] saved decision boundary to: {saved_path}")

    if show:
        plt.show()

    plt.close(fig)

    return saved_path