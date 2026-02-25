import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BASE_DIR = os.environ.get(
    "SLURM_SUBMIT_DIR",
    os.path.dirname(os.path.abspath(__file__))
)

def visualize_pca(experiment_folder, outdir="plots"):
    exp_dir = os.path.join(BASE_DIR, "version2", experiment_folder)

    pca_cache = os.path.join(
        exp_dir,
        f"{experiment_folder}_pca.joblib"
    )

    embeddings_csv = os.path.join(
        exp_dir,
        "embeddings_and_labels.csv"
    )

    if not os.path.exists(pca_cache):
        raise FileNotFoundError(pca_cache)

    if not os.path.exists(embeddings_csv):
        raise FileNotFoundError(embeddings_csv)

    print("[VIS] Loading PCA cache...")
    pca_data = joblib.load(pca_cache)

    coords = pca_data["coords"]

    print("[VIS] Loading entity order...")
    df = pd.read_csv(embeddings_csv)

    # init_vals already aligned to embedding rows
    vals = np.asarray(pca_data["init_vals"], dtype=float)

    if len(vals) != coords.shape[0]:
        raise ValueError("Init values do not align with PCA rows")

    mean_val = np.nanmean(vals)
    valid = ~np.isnan(vals)

    # ---------------------------------------
    # Plot
    # ---------------------------------------
    os.makedirs(outdir, exist_ok=True)

    cmap_grad = LinearSegmentedColormap.from_list(
        "blue_red",
        ["dodgerblue", "red"]
    )

    x = coords[:, 0]
    y = coords[:, 1]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Points without init values
    ax.scatter(
        x[~valid],
        y[~valid],
        color="lightgray",
        alpha=0.25,
        s=60,
        label="No init value"
    )

    # Points with init values
    sc = ax.scatter(
        x[valid],
        y[valid],
        c=vals[valid],
        cmap=cmap_grad,
        alpha=0.6,
        s=80,
        edgecolor=(0, 0, 0, 0.3),
        linewidths=0.4,
        label="Init value"
    )

    # Centered numeric labels
    for i in np.where(valid)[0]:
        txt_color = "white" if vals[i] >= mean_val else "black"

        ax.text(
            x[i],
            y[i],
            f"{vals[i]:.0f}",
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
            color=txt_color
        )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Initialization Value")

    ax.set_title("PCA: PC1 vs PC2")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()

    plt.tight_layout()

    out_path = os.path.join(outdir, "PCA_PC1_PC2.svg")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[VIS] Saved → {out_path}")
