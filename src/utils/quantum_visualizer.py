# quantum_visualizer.py
import os
import numpy as np
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
    qsvc, X, y,
    title="QSVM Decision Boundary (PCA Projection)",
    save: bool = True,
    show: bool = True,
    filename: str = None
) -> str:
    """
    Plot QSVM decision scores projected into 2D PCA space with contour lines.
    Always applies PCA internally for visualization.
    """

    # 1) PCA projection
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)

    # 2) Mesh grid in PCA space
    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # 3) Map grid back to original space
    grid_original = pca.inverse_transform(grid_points)

    # 4) Decision scores from QSVM
    Z = qsvc.decision_function(grid_original)
    Z = Z.reshape(xx.shape)

    # 5) Plot contour + scatter
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.3, colors=["#FFAAAA", "#AAAAFF"])
    plt.scatter(X2[:, 0], X2[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors="k")
    plt.title(title)
    plt.tight_layout()

    saved_path = ""
    if save:
        root = _repo_root_from_utils()
        plots_dir = os.path.join(root, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        fname = filename if filename else _sanitize_filename(title)
        saved_path = os.path.join(plots_dir, fname)

        plt.savefig(saved_path, dpi=120)
        print(f"[quantum_visualizer] saved plot to: {saved_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return saved_path


def plot_quantum_kernel_matrix(
    kernel_matrix,
    title="Quantum Kernel Matrix",
    save: bool = True,
    show: bool = True,
    filename: str = None
) -> str:
    """
    Plot a heatmap of the quantum kernel matrix.
    """

    plt.figure(figsize=(6, 5))
    plt.imshow(kernel_matrix, interpolation="nearest", cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()

    saved_path = ""
    if save:
        root = _repo_root_from_utils()
        plots_dir = os.path.join(root, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        fname = filename if filename else _sanitize_filename(title)
        saved_path = os.path.join(plots_dir, fname)

        plt.savefig(saved_path, dpi=120)
        print(f"[quantum_visualizer] saved kernel matrix to: {saved_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return saved_path
