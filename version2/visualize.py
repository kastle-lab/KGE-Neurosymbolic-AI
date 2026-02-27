import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

person_node_color = "green"
window_node_color = "black"
missed_node_color = "yellow"

BASE_DIR = os.environ.get(
    "SLURM_SUBMIT_DIR",
    os.path.dirname(os.path.abspath(__file__))
)

def _safe_mkdir(p):
    if p:
        os.makedirs(p, exist_ok=True)

def _annotate_points_2d(ax, x, y, labels, idx, max_labels=150, fontsize=7):
    """
    Annotate a subset of points to avoid unreadable clutter.
    - If len(idx) <= max_labels: annotate all in idx
    - Else: annotate evenly-spaced subset of idx of size max_labels
    """
    if idx.size == 0:
        return

    if idx.size > max_labels:
        pick = np.linspace(0, idx.size - 1, max_labels).astype(int)
        idx = idx[pick]

    for i in idx:
        ax.text(
            float(x[i]), float(y[i]),
            str(labels[i]),
            fontsize=fontsize,
            alpha=0.85
        )

def _annotate_points_3d(ax, x, y, z, labels, idx, max_labels=80, fontsize=7):
    if idx.size == 0:
        return

    if idx.size > max_labels:
        pick = np.linspace(0, idx.size - 1, max_labels).astype(int)
        idx = idx[pick]

    for i in idx:
        ax.text(
            float(x[i]), float(y[i]), float(z[i]),
            str(labels[i]),
            fontsize=fontsize,
            alpha=0.85
        )

def visualize_pca(
    experiment_folder,
    outdir=None,
    show=False,
    v_range=(0, 99)
):
    """
    Visualize PCA results for a given experiment.

    Produces:
        - full plots
        - no person nodes
        - no person + no window nodes

    Saves into:
        version2/<experiment_folder>/plots/
    """

    exp_dir = os.path.join(BASE_DIR, "version2", experiment_folder)

    # Default output dir inside experiment folder
    if outdir is None:
        outdir = os.path.join(exp_dir, "plots")

    os.makedirs(outdir, exist_ok=True)

    pca_cache = os.path.join(
        exp_dir,
        f"{experiment_folder}_pca.joblib"
    )

    if not os.path.exists(pca_cache):
        print(f"[VIS] Expected PCA cache not found:")
        print(f"       {pca_cache}")
        print(f"[VIS] Available experiment folders:")
        print(os.listdir(os.path.join(BASE_DIR, "version2")))
        raise FileNotFoundError(pca_cache)

    print("[VIS] Loading PCA cache...")
    pca_data = joblib.load(pca_cache)

    coords = np.asarray(pca_data["coords"])
    entity_labels = list(pca_data["entities"])

    cmap_grad = LinearSegmentedColormap.from_list(
        "blue_red",
        ["dodgerblue", "red"]
    )

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    v_idx = []
    v_vals = []
    person_idx = []
    window_idx = []
    other_idx = []

    for i, label in enumerate(entity_labels):
        if label.startswith("person"):
            person_idx.append(i)
        elif label.startswith("Window"):
            window_idx.append(i)
        elif label.startswith("v"):
            try:
                num = int(label[1:])
                v_idx.append(i)
                v_vals.append(num)
            except:
                other_idx.append(i)
        else:
            other_idx.append(i)

    v_idx = np.array(v_idx)
    v_vals = np.array(v_vals)
    person_idx = np.array(person_idx)
    window_idx = np.array(window_idx)
    other_idx = np.array(other_idx)

    def plot_variant(name, keep_mask):

        # 2D
        fig2, ax2 = plt.subplots(figsize=(10, 8))

        idx = np.intersect1d(other_idx, keep_mask)
        if idx.size:
            ax2.scatter(x[idx], y[idx], color=missed_node_color, alpha=0.25, s=55)

        idx = np.intersect1d(person_idx, keep_mask)
        if idx.size:
            ax2.scatter(x[idx], y[idx], color=person_node_color, alpha=0.8, s=70)

        idx = np.intersect1d(window_idx, keep_mask)
        if idx.size:
            ax2.scatter(x[idx], y[idx], color=window_node_color, alpha=0.8, s=70)

        idx = np.intersect1d(v_idx, keep_mask)
        if idx.size:
            vals = v_vals[np.isin(v_idx, idx)]
            sc = ax2.scatter(
                x[idx], y[idx],
                c=vals,
                cmap=cmap_grad,
                alpha=0.9,
                s=85
            )
            cbar = plt.colorbar(sc, ax=ax2)
            cbar.set_label("v index")

        ax2.set_title(f"{experiment_folder} — {name} (2D)")
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        plt.tight_layout()

        path2 = os.path.join(outdir, f"{experiment_folder}_{name}_2D.svg")
        plt.savefig(path2, bbox_inches="tight")
        if show: plt.show()
        plt.close(fig2)

        # 3D
        fig3 = plt.figure(figsize=(11, 9))
        ax3 = fig3.add_subplot(111, projection="3d")

        idx = np.intersect1d(other_idx, keep_mask)
        if idx.size:
            ax3.scatter(x[idx], y[idx], z[idx], color=missed_node_color, alpha=0.18, s=35)

        idx = np.intersect1d(person_idx, keep_mask)
        if idx.size:
            ax3.scatter(x[idx], y[idx], z[idx], color=person_node_color, alpha=0.85, s=45)

        idx = np.intersect1d(window_idx, keep_mask)
        if idx.size:
            ax3.scatter(x[idx], y[idx], z[idx], color=window_node_color, alpha=0.85, s=45)

        idx = np.intersect1d(v_idx, keep_mask)
        if idx.size:
            vals = v_vals[np.isin(v_idx, idx)]
            sc = ax3.scatter(
                x[idx], y[idx], z[idx],
                c=vals,
                cmap=cmap_grad,
                alpha=0.9,
                s=55
            )
            cbar = plt.colorbar(sc, ax=ax3, shrink=0.6)
            cbar.set_label("v index")

        ax3.set_title(f"{experiment_folder} — {name} (3D)")
        ax3.set_xlabel("PC1")
        ax3.set_ylabel("PC2")
        ax3.set_zlabel("PC3")

        plt.tight_layout()

        path3 = os.path.join(outdir, f"{experiment_folder}_{name}_3D.svg")
        plt.savefig(path3, bbox_inches="tight")
        if show: plt.show()
        plt.close(fig3)

        print(f"[VIS] Saved: {path2}")
        print(f"[VIS] Saved: {path3}")

    all_idx = np.arange(len(entity_labels))

    plot_variant("full", all_idx)
    no_person = np.setdiff1d(all_idx, person_idx)
    plot_variant("no_person", no_person)
    no_person_window = np.setdiff1d(no_person, window_idx)
    plot_variant("no_person_window", no_person_window)
    
# Example usage:
# visualize_pca("my_experiment", outdir="./learning_experiments/plots", show=False)