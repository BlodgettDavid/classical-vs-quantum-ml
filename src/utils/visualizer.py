from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    # Only works for 2D input
    if X.shape[1] != 2:
        raise ValueError("X must be 2D for decision boundary visualization.")

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors="k")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.show()

def plot_3d_decision_boundary(model, X, y, title="3D Decision Boundary"):
    # Only works for 3D input
    if X.shape[1] != 3:
        raise ValueError("X must be 3D for 3D decision boundary visualization.")

    # Create 3D mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

    # Coarse grid to avoid memory overload
    xx, yy, zz = np.meshgrid(
        np.linspace(x_min, x_max, 30),
        np.linspace(y_min, y_max, 30),
        np.linspace(z_min, z_max, 30)
    )
    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    Z = model.predict(grid)

    # Plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.coolwarm, edgecolors="k", s=50)
    ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2], c=Z, cmap=plt.cm.coolwarm, alpha=0.1, s=5)

    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    plt.tight_layout()
    plt.show()
    
'''
handle 4D and any higher‑dimensional input cleanly
'''

def plot_projected_decision_boundary(model, X, y, title="SVM Projected Decision Boundary"):
    # Reduce to 3D using PCA (works for any input dimension)
    pca = PCA(n_components=3)
    X_proj = pca.fit_transform(X)

    # Predict labels using the full feature set
    y_pred = model.predict(X)

    # Plot projected data
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        X_proj[:, 0], X_proj[:, 1], X_proj[:, 2],
        c=y_pred, cmap=plt.cm.coolwarm, edgecolors="k", s=50
    )

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.tight_layout()
    plt.show()