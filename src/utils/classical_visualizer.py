# utils/visualizer.py
import os
import numpy as np
import matplotlib.pyplot as plt

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


def plot_projected_decision_boundary(
    model, X, y,
    title="Decision Boundary (PCA Projection)",
    save: bool = True,
    show: bool = True,
    filename: str = None,
    surrogate_kernel: str = "linear"
) -> str:
    """
    Visualizes a decision boundary by always projecting X to 2D with PCA.
    PCA is applied for plotting regardless of training preprocessing.
    The surrogate classifier is trained only for visualization, not for evaluation.
    Returns the absolute path to the saved image if save=True, else an empty string.
    """

    # 1) Dimensionality reduction (always PCA)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)

    # 2) Mesh for decision boundary
    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # 3) Surrogate classifier for visualization only
    from sklearn.svm import SVC
    surrogate = SVC(kernel=surrogate_kernel)
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