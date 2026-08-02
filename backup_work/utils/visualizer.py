# utils/visualizer.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def _repo_root_from_utils() -> str:
    # utils/ lives at src/utils/, so root is two levels up
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

def plot_projected_decision_boundary(model, X, y, title="Decision Boundary",
                                     save: bool = True, show: bool = True, filename: str = None) -> str:
    """
    Projects X to 2D via PCA, plots decision boundary + points, and optionally saves/shows the figure.
    Returns the absolute path to the saved image if save=True, else an empty string.
    """
    # 1) PCA to 2D
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)
    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5

    # 2) Mesh for decision boundary
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # 3) Predict on the 2D grid using a surrogate:
    #    Fit a light 2D classifier on the projected training space
    #    (many scikit/Qiskit models don't accept arbitrary PCA-projected inputs directly).
    #    We refit a simple 2D SVM surrogate on X2 vs y to visualize the boundary.
    from sklearn.svm import SVC
    surrogate = SVC(kernel="linear")
    surrogate.fit(X2, y)
    Z = surrogate.predict(grid).reshape(xx.shape)
    
    # 4) Plot
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.35)
    plt.scatter(X2[:, 0], X2[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=24)
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
        print(f"[visualizer] saved plot to: {saved_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return saved_path
