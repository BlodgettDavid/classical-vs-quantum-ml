# utils/classical_visualizer.py
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Prevent GUI window creation in headless/batch execution
import matplotlib.pyplot as plt

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


def plot_projected_decision_boundary(
    model,
    X: np.ndarray, 
    y: np.ndarray,
    title: str = "Decision Boundary",
    save: bool = True,
    show: bool = True,
    filename: str = None,
    do_pca: bool = True,
    grid_steps: int = 100
) -> tuple[str, float]:
    """
    Visualizes the decision boundary of a trained classical model and tracks grid evaluation latency.
    """
    start_time = time.perf_counter()
    saved_path = ""

    # 1. Coordinate projection for visualization space
    if do_pca or X.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X)
    else:
        X2 = X[:, :2] if X.shape[1] >= 2 else X
        class IdentityPCA:
            def inverse_transform(self, pts: np.ndarray) -> np.ndarray:
                if X.shape[1] > 2:
                    padded = np.zeros((pts.shape[0], X.shape[1]))
                    padded[:, :2] = pts
                    return padded
                return pts
        pca = IdentityPCA()

    # 2. Mesh grid generation
    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_steps),
        np.linspace(y_min, y_max, grid_steps)
    )
    grid_2d = np.c_[xx.ravel(), yy.ravel()]

    # 3. Predict using the actual trained model instance
    grid_original = pca.inverse_transform(grid_2d)
    
    if hasattr(model, "decision_function"):
        Z = model.decision_function(grid_original).reshape(xx.shape)
    else:
        Z = model.predict(grid_original).reshape(xx.shape)

    plotting_runtime = round(time.perf_counter() - start_time, 4)

    # 4. Figure rendering
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.35)
    ax.scatter(X2[:, 0], X2[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=24)
    ax.set_title(f"{title}\n[Plotting Latency: {plotting_runtime}s]")
    fig.tight_layout()

    # 5. Export plot artifact
    if save:
        root = _repo_root_from_utils()
        plots_dir = os.path.join(root, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        fname = filename if filename else _sanitize_filename(title)
        saved_path = os.path.join(plots_dir, fname)
        fig.savefig(saved_path, dpi=120)
        print(f"[classical_visualizer] saved boundary plot to: {saved_path}")

    print(f"[classical_visualizer] Plotting latency ({grid_steps}x{grid_steps} mesh): {plotting_runtime}s")

    if show:
        plt.show()
    
    plt.close(fig)

    return saved_path, plotting_runtime


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

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()

    saved_path = ""
    if save:
        root = _repo_root_from_utils()
        plots_dir = os.path.join(root, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        fname = filename if filename else _sanitize_filename(title)
        saved_path = os.path.join(plots_dir, fname)
        fig.savefig(saved_path, dpi=120)
        print(f"[classical_visualizer] saved confusion matrix to: {saved_path}")

    if show:
        plt.show()
    
    plt.close(fig)

    return saved_path