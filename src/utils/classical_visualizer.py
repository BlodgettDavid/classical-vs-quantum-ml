# utils/classical_visualizer.py
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
    X, 
    y,
    title="Decision Boundary",
    save: bool = True,
    show: bool = True,
    filename: str = None,
    surrogate_kernel: str = "linear",
    do_pca: bool = True
) -> str:
    """
    Visualizes a decision boundary. If use_pca=True, project X to 2D with PCA.
    If use_pca=False, assume X is already 2D (e.g., PCA done in modeling).
    """
    if do_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X)
    else:
        if X.shape[1] != 2:
            raise ValueError("X must be 2D if use_pca=False")
        X2 = X

    # Mesh grid
    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Surrogate classifier
    from sklearn.svm import SVC
    surrogate = SVC(kernel=surrogate_kernel)
    surrogate.fit(X2, y)
    Z = surrogate.predict(grid).reshape(xx.shape)

    # Plot
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


def plot_confusion_matrix(
    y_true, 
    y_pred,
    title="Confusion Matrix",
    save: bool = True,
    show: bool = True,
    filename: str = None
) -> str:
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    saved_path = ""
    if save:
        root = _repo_root_from_utils()
        plots_dir = os.path.join(root, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        fname = filename if filename else _sanitize_filename(title)
        saved_path = os.path.join(plots_dir, fname)
        plt.savefig(saved_path, dpi=120)
        print(f"[classical_visualizer] saved confusion matrix to: {saved_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return saved_path