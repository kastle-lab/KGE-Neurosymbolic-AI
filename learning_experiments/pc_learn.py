from load_data import load_data
import os, re, random, torch, joblib
import numpy as np
import pandas as pd

from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D

# scikit-learn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap.umap_ as umap
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# plotting imports
import matplotlib
matplotlib.use("Agg")  # always use headless backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

seed = 42
E, E_parents, E3_mean, entity_labels, init_vals = load_data("~/fry_program/data/E-5000/5000_E_entities.csv")

def get_or_compute_pca(E, n_components=2, seed=42,
                       cache_path="./learning_experiments/pca_cache.joblib"):
    """
    Loads PCA + transformed coordinates from disk if available.
    Otherwise computes and optionally saves them.
    """

    use_cache = bool(cache_path)

    if use_cache and os.path.exists(cache_path):
        print(f"[PCA] Loading cached PCA from {cache_path}")
        data = joblib.load(cache_path)
        return data["coords"], data["pca"]

    print("[PCA] Computing PCA...")
    pca = PCA(n_components=n_components, random_state=seed)
    coords = pca.fit_transform(E)

    if use_cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        joblib.dump({"coords": coords, "pca": pca}, cache_path)
        print(f"[PCA] Saved PCA cache → {cache_path}")

    return coords, pca

def make_mlp(hidden=(128, 64), seed=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="tanh",
            solver="adam",
            max_iter=5000,
            random_state=seed
        ))
    ])

def plot_DR_result(reductions, ordered_init_vals, basename, outdir=None):

    if outdir:
        os.makedirs(outdir, exist_ok=True)

    vals = np.asarray(ordered_init_vals, dtype=float)
    mean_val = np.nanmean(vals)
    valid = ~np.isnan(vals)

    cmap_grad = LinearSegmentedColormap.from_list(
        "blue_red", ["dodgerblue", "red"]
    )

    V = vals.reshape(-1, 1)

    for method, coords in reductions.items():

        if coords.shape[0] != len(vals):
            raise ValueError("Reduction rows don't match initialization values!")

        d = coords.shape[1]
        if d < 2:
            raise ValueError("Need at least 2 components.")

        # ======================================================
        # 2D PAIRWISE PLOTS (original behavior, generalized)
        # ======================================================
        for i, j in combinations(range(d), 2):

            x = coords[:, i]
            y = coords[:, j]

            mlp_vxy = make_mlp(hidden=(128, 128))
            mlp_vxy.fit(
                V[valid],
                np.column_stack([x[valid], y[valid]])
            )

            v_line = np.linspace(
                vals[valid].min(),
                vals[valid].max(),
                600
            ).reshape(-1, 1)

            xy_pred = mlp_vxy.predict(v_line)

            fig, ax = plt.subplots(figsize=(10, 8))

            sc = ax.scatter(
                x, y,
                c=vals,
                cmap=cmap_grad,
                alpha=0.3,
                s=60,
                edgecolor=(0, 0, 0, 0.2),
                linewidths=0.2
            )

            ax.plot(
                xy_pred[:, 0],
                xy_pred[:, 1],
                color="purple",
                linewidth=3,
                label="MLP: init_vals → (PC)"
            )

            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Initialization Value")

            ax.set_title(f"PC{i+1} vs PC{j+1}")
            ax.set_xlabel(f"PC{i+1}")
            ax.set_ylabel(f"PC{j+1}")
            ax.legend()

            plt.tight_layout()

            if outdir:
                path = os.path.join(
                    outdir,
                    f"PC{i+1}_PC{j+1}.svg"
                )
                plt.savefig(path, bbox_inches="tight")

            plt.close(fig)

        # ======================================================
        # 3D PLOT (PC1, PC2, PC3 only)
        # ======================================================
        if d >= 3:

            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

            fig = plt.figure(figsize=(11, 9))
            ax = fig.add_subplot(111, projection="3d")

            sc = ax.scatter(
                x, y, z,
                c=vals,
                cmap=cmap_grad,
                alpha=0.35,
                s=40
            )

            # ----------------------------------
            # 1D CURVE: init_vals → (x,y,z)
            # ----------------------------------
            mlp_curve = make_mlp(hidden=(128, 128))
            mlp_curve.fit(
                V[valid],
                np.column_stack([x[valid], y[valid], z[valid]])
            )

            v_line = np.linspace(
                vals[valid].min(),
                vals[valid].max(),
                600
            ).reshape(-1, 1)

            xyz_curve = mlp_curve.predict(v_line)

            ax.plot(
                xyz_curve[:, 0],
                xyz_curve[:, 1],
                xyz_curve[:, 2],
                color="purple",
                linewidth=3,
                label="1D curve: init_vals → (PC1,PC2,PC3)"
            )

            # ----------------------------------
            # 2D SURFACE: (x,y) → z
            # ----------------------------------
            XY = np.column_stack([x, y])

            mlp_surface = make_mlp(hidden=(128, 128))
            mlp_surface.fit(XY[valid], z[valid])

            grid_n = 40
            gx = np.linspace(x.min(), x.max(), grid_n)
            gy = np.linspace(y.min(), y.max(), grid_n)
            GX, GY = np.meshgrid(gx, gy)

            Z_pred = mlp_surface.predict(
                np.column_stack([GX.ravel(), GY.ravel()])
            ).reshape(GX.shape)

            ax.plot_surface(
                GX, GY, Z_pred,
                alpha=0.35,
                color="purple",
                linewidth=0,
                antialiased=True
            )

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            ax.set_title(f"{method} — 3D Curve + Surface")
            ax.legend()

            cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
            cbar.set_label("Initialization Value")

            if outdir:
                path = os.path.join(
                    outdir,
                    f"PC1_PC2_PC3_3D.svg"
                )
                plt.savefig(path, bbox_inches="tight")

            plt.close(fig)
            
            # save 3D data for manim plotting
            np.savez(
                "./learning_experiments/manim_3d_data.npz",
                points=np.column_stack([x, y, z]),
                curve=xyz_curve,
                surface_X=GX,
                surface_Y=GY,
                surface_Z=Z_pred,
                vals=vals
            )
        
pca_coords, pca_model = get_or_compute_pca(
    E,
    n_components=3,
    seed=seed,
    cache_path=""
)

reductions = {"PCA": pca_coords}

plot_DR_result(reductions, init_vals, "PCA", "./learning_experiments/plots")