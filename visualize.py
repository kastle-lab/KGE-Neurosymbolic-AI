import os
import re
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

person_node_color = "black"
window_node_color = "green"
missed_node_color = "yellow"

BASE_DIR = os.environ.get(
    "SLURM_SUBMIT_DIR",
    os.path.dirname(os.path.abspath(__file__))
)

def visualize_pca(
    experiment_folder,
    outdir=None,
    show=False,
):
    exp_dir = os.path.join(BASE_DIR, experiment_folder)

    if outdir is None:
        outdir = os.path.join(exp_dir, "plots")

    os.makedirs(outdir, exist_ok=True)

    pca_cache = os.path.join(
        exp_dir,
        f"{experiment_folder}_pca.joblib"
    )

    if not os.path.exists(pca_cache):
        raise FileNotFoundError(f"PCA cache not found: {pca_cache}")

    print("[VIS] Loading PCA cache...")
    pca_data = joblib.load(pca_cache)

    coords = np.asarray(pca_data["coords"])
    entity_labels = list(pca_data["entities"])

    if coords.shape[1] < 3:
        raise ValueError(
            f"Expected at least 3 PCA components, but got coords shape {coords.shape}"
        )

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    cmap_grad = LinearSegmentedColormap.from_list(
        "blue_red",
        ["dodgerblue", "red"]
    )

    v_idx = []
    v_vals = []
    person_idx = []
    window_idx = []
    other_idx = []

    for i, label in enumerate(entity_labels):
        label_str = str(label)

        if label_str.lower().startswith("person"):
            person_idx.append(i)

        elif label_str.lower().startswith("window"):
            window_idx.append(i)

        else:
            m = re.fullmatch(r"v(\d+)", label_str)
            if m:
                v_idx.append(i)
                v_vals.append(int(m.group(1)))
            else:
                other_idx.append(i)

    v_idx = np.array(v_idx, dtype=int)
    v_vals = np.array(v_vals, dtype=float)
    person_idx = np.array(person_idx, dtype=int)
    window_idx = np.array(window_idx, dtype=int)
    other_idx = np.array(other_idx, dtype=int)

    all_idx = np.arange(len(entity_labels), dtype=int)

    def plot_variant(name, keep_mask):
        keep_mask = np.array(keep_mask, dtype=int)

        # -------------------------------------------------
        # 1D PCA PLOT
        # -------------------------------------------------
        fig1, ax1 = plt.subplots(figsize=(12, 3.5))

        idx = np.intersect1d(other_idx, keep_mask)
        if idx.size:
            ax1.scatter(
                x[idx],
                np.zeros_like(x[idx]),
                color=missed_node_color,
                alpha=0.25,
                s=55,
                label="other"
            )

        idx = np.intersect1d(person_idx, keep_mask)
        if idx.size:
            ax1.scatter(
                x[idx],
                np.zeros_like(x[idx]),
                color=person_node_color,
                alpha=0.8,
                s=70,
                label="person"
            )

        idx = np.intersect1d(window_idx, keep_mask)
        if idx.size:
            ax1.scatter(
                x[idx],
                np.zeros_like(x[idx]),
                color=window_node_color,
                alpha=0.8,
                s=70,
                label="window"
            )

        idx = np.intersect1d(v_idx, keep_mask)
        if idx.size:
            vals = v_vals[np.isin(v_idx, idx)]
            sc1 = ax1.scatter(
                x[idx],
                np.zeros_like(x[idx]),
                c=vals,
                cmap=cmap_grad,
                alpha=0.9,
                s=85
            )
            cbar = plt.colorbar(sc1, ax=ax1)
            cbar.set_label("v value")

        ax1.set_title(f"{experiment_folder} — {name} (1D PCA)")
        ax1.set_xlabel("PC1")
        ax1.set_yticks([])
        ax1.axhline(0, linewidth=1, alpha=0.3)
        plt.tight_layout()

        path1 = os.path.join(outdir, f"{experiment_folder}_{name}_1D.svg")
        plt.savefig(path1, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig1)

        # -------------------------------------------------
        # PC1 VS ENCODED VALUE PLOT
        # -------------------------------------------------
        figv, axv = plt.subplots(figsize=(10, 8))

        idx = np.intersect1d(v_idx, keep_mask)
        if idx.size:
            vals = v_vals[np.isin(v_idx, idx)]
            scv = axv.scatter(
                x[idx],
                vals,
                c=vals,
                cmap=cmap_grad,
                alpha=0.9,
                s=85
            )
            cbar = plt.colorbar(scv, ax=axv)
            cbar.set_label("v value")

        axv.set_title(f"{experiment_folder} — {name} (PC1 vs v value)")
        axv.set_xlabel("PC1")
        axv.set_ylabel("Encoded Value")
        plt.tight_layout()

        pathv = os.path.join(outdir, f"{experiment_folder}_{name}_PC1_vs_value.svg")
        plt.savefig(pathv, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(figv)

        # -------------------------------------------------
        # 2D PCA PLOT
        # -------------------------------------------------
        fig2, ax2 = plt.subplots(figsize=(10, 8))

        idx = np.intersect1d(other_idx, keep_mask)
        if idx.size:
            ax2.scatter(
                x[idx], y[idx],
                color=missed_node_color,
                alpha=0.25,
                s=55,
                label="other"
            )

        idx = np.intersect1d(person_idx, keep_mask)
        if idx.size:
            ax2.scatter(
                x[idx], y[idx],
                color=person_node_color,
                alpha=0.8,
                s=70,
                label="person"
            )

        idx = np.intersect1d(window_idx, keep_mask)
        if idx.size:
            ax2.scatter(
                x[idx], y[idx],
                color=window_node_color,
                alpha=0.8,
                s=70,
                label="window"
            )

        idx = np.intersect1d(v_idx, keep_mask)
        if idx.size:
            vals = v_vals[np.isin(v_idx, idx)]
            sc2 = ax2.scatter(
                x[idx], y[idx],
                c=vals,
                cmap=cmap_grad,
                alpha=0.9,
                s=85
            )
            cbar = plt.colorbar(sc2, ax=ax2)
            cbar.set_label("v value")

        ax2.set_title(f"{experiment_folder} — {name} (2D PCA)")
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        plt.tight_layout()

        path2 = os.path.join(outdir, f"{experiment_folder}_{name}_2D.svg")
        plt.savefig(path2, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig2)

        # -------------------------------------------------
        # 3D PCA PLOT
        # -------------------------------------------------
        fig3 = plt.figure(figsize=(11, 9))
        ax3 = fig3.add_subplot(111, projection="3d")

        idx = np.intersect1d(other_idx, keep_mask)
        if idx.size:
            ax3.scatter(
                x[idx], y[idx], z[idx],
                color=missed_node_color,
                alpha=0.18,
                s=35
            )

        idx = np.intersect1d(person_idx, keep_mask)
        if idx.size:
            ax3.scatter(
                x[idx], y[idx], z[idx],
                color=person_node_color,
                alpha=0.85,
                s=45
            )

        idx = np.intersect1d(window_idx, keep_mask)
        if idx.size:
            ax3.scatter(
                x[idx], y[idx], z[idx],
                color=window_node_color,
                alpha=0.85,
                s=45
            )

        idx = np.intersect1d(v_idx, keep_mask)
        if idx.size:
            vals = v_vals[np.isin(v_idx, idx)]
            sc3 = ax3.scatter(
                x[idx], y[idx], z[idx],
                c=vals,
                cmap=cmap_grad,
                alpha=0.9,
                s=55
            )
            cbar = plt.colorbar(sc3, ax=ax3, shrink=0.6)
            cbar.set_label("v value")

        ax3.set_title(f"{experiment_folder} — {name} (3D PCA)")
        ax3.set_xlabel("PC1")
        ax3.set_ylabel("PC2")
        ax3.set_zlabel("PC3")
        plt.tight_layout()

        path3 = os.path.join(outdir, f"{experiment_folder}_{name}_3D.svg")
        plt.savefig(path3, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig3)

        print(f"[VIS] Saved: {path1}")
        print(f"[VIS] Saved: {pathv}")
        print(f"[VIS] Saved: {path2}")
        print(f"[VIS] Saved: {path3}")

    plot_variant("full", all_idx)

    no_person = np.setdiff1d(all_idx, person_idx)
    plot_variant("no_person", no_person)

    no_person_window = np.setdiff1d(no_person, window_idx)
    plot_variant("no_person_window", no_person_window)