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


def plot_qsvm_decision_boundary(
    qsvc, 
    X: np.ndarray, 
    y: np.ndarray,
    title: str = "QSVM Decision Boundary",
    save: bool = True,
    show: bool = True,
    filename: str | None = None,
    do_pca: bool = True,
    grid_steps: int = 25
) -> tuple[str, float]:
    """
    Plots QSVM decision scores in 2D space and returns (saved_path, plotting_runtime).
    """
    start_time = time.perf_counter()
    saved_path = ""

    # 1. Coordinate reduction strategy
    if do_pca and X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X)
    else:
        if X.shape[1] != 2 and not do_pca:
            raise ValueError("X must have exactly 2 dimensions when do_pca=False")
        X2 = X[:, :2] if X.shape[1] >= 2 else X

        class IdentityPCA:
            def inverse_transform(self, pts: np.ndarray) -> np.ndarray:
                if X.shape[1] > 2:
                    padded = np.zeros((pts.shape[0], X.shape[1]))
                    padded[:, :2] = pts
                    return padded
                return pts

        pca = IdentityPCA()

    # 2. Mesh grid construction
    margin_x = (X2[:, 0].max() - X2[:, 0].min()) * 0.05
    margin_y = (X2[:, 1].max() - X2[:, 1].min()) * 0.05

    x_min = max(0.0, X2[:, 0].min() - margin_x)
    x_max = min(np.pi, X2[:, 0].max() + margin_x)
    y_min = max(0.0, X2[:, 1].min() - margin_y)
    y_max = min(np.pi, X2[:, 1].max() + margin_y)

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_steps),
        np.linspace(y_min, y_max, grid_steps)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # 3. Decision score evaluation
    grid_original = pca.inverse_transform(grid_points)
    grid_original = np.clip(grid_original, 0.0, np.pi)
    
    Z = qsvc.decision_function(grid_original).reshape(xx.shape)

    # Compute execution time before using it in title formatting
    plotting_runtime = round(time.perf_counter() - start_time, 4)

    # 4. Explicit Object-Oriented Figure Rendering
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.35, colors=["#FFAAAA", "#AAAAFF"])
    ax.scatter(X2[:, 0], X2[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors="k", s=24)
    ax.set_title(f"{title}\n[Plotting Latency: {plotting_runtime}s]")
    fig.tight_layout()

    # 5. Save & Cleanup logic
    if save:
        root = _repo_root_from_utils()
        plots_dir = os.path.join(root, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        fname = filename if filename else _sanitize_filename(title)
        saved_path = os.path.join(plots_dir, fname)
        fig.savefig(saved_path, dpi=120)
        print(f"[quantum_visualizer] saved boundary plot to: {saved_path}")

    print(f"[quantum_visualizer] Boundary latency ({grid_steps}x{grid_steps} mesh): {plotting_runtime:.4f}s")

    if show:
        plt.show()
    
    plt.close(fig)

    return saved_path, plotting_runtime


def plot_quantum_kernel_matrix(
    kernel_matrix: np.ndarray,
    title: str = "Quantum Kernel Matrix",
    save: bool = True,
    show: bool = True,
    filename: str | None = None
) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(kernel_matrix, interpolation="nearest", cmap="viridis")
    fig.colorbar(cax, ax=ax)
    ax.set_title(title)
    fig.tight_layout()

    saved_path = ""
    if save:
        root = _repo_root_from_utils()
        plots_dir = os.path.join(root, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        fname = filename if filename else _sanitize_filename(title)
        saved_path = os.path.join(plots_dir, fname)
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
    filename: str | None = None
) -> str:
    """
    Renders a 2x2 categorical confusion matrix heatmap with cell count annotations.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(5, 4.5))
    cax = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(cax, ax=ax)

    # Annotate counts and percentages inside cells
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
        plots_dir = os.path.join(root, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        fname = filename if filename else _sanitize_filename(title)
        saved_path = os.path.join(plots_dir, fname)
        fig.savefig(saved_path, dpi=120)
        print(f"[quantum_visualizer] saved confusion matrix to: {saved_path}")

    if show:
        plt.show()

    plt.close(fig)

    return saved_path