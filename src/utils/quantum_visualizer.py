# src/utils/quantum_visualizer.py
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for batch execution
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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
    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_steps),
        np.linspace(y_min, y_max, grid_steps)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # 3. Decision score evaluation
    grid_original = pca.inverse_transform(grid_points)
    Z = qsvc.decision_function(grid_original).reshape(xx.shape)

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
    
    plt.close(fig)  # Guarantees zero figure leaks in batch processing

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