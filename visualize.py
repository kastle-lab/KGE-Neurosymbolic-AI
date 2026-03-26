import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

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

def _parse_label_groups(entity_labels):
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

    return (
        np.array(v_idx, dtype=int),
        np.array(v_vals, dtype=float),
        np.array(person_idx, dtype=int),
        np.array(window_idx, dtype=int),
        np.array(other_idx, dtype=int),
    )

def _plot_family(
    coords,
    entity_labels,
    experiment_folder,
    outdir,
    show=False,
    filename_prefix="all_embeddings",
    title_prefix="All Embeddings PCA"
):
    coords = np.asarray(coords)
    entity_labels = list(entity_labels)

    if coords.shape[1] < 3:
        raise ValueError(f"Expected at least 3 PCA components, got shape {coords.shape}")

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    cmap_grad = LinearSegmentedColormap.from_list(
        "blue_red",
        ["dodgerblue", "red"]
    )

    v_idx, v_vals, person_idx, window_idx, other_idx = _parse_label_groups(entity_labels)
    all_idx = np.arange(len(entity_labels), dtype=int)

    def plot_variant(name, keep_mask):
        keep_mask = np.array(keep_mask, dtype=int)

        # -------------------------------------------------
        # 1D PCA
        # -------------------------------------------------
        fig1, ax1 = plt.subplots(figsize=(12, 3.5))

        idx = np.intersect1d(other_idx, keep_mask)
        if idx.size:
            ax1.scatter(
                x[idx],
                np.zeros_like(x[idx]),
                color=missed_node_color,
                alpha=0.25,
                s=55
            )

        idx = np.intersect1d(person_idx, keep_mask)
        if idx.size:
            ax1.scatter(
                x[idx],
                np.zeros_like(x[idx]),
                color=person_node_color,
                alpha=0.8,
                s=70
            )

        idx = np.intersect1d(window_idx, keep_mask)
        if idx.size:
            ax1.scatter(
                x[idx],
                np.zeros_like(x[idx]),
                color=window_node_color,
                alpha=0.85,
                s=70
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

        ax1.set_title(f"{experiment_folder} — {title_prefix} — {name} (1D)")
        ax1.set_xlabel("PC1")
        ax1.set_yticks([])
        ax1.axhline(0, linewidth=1, alpha=0.3)
        plt.tight_layout()

        path1 = os.path.join(outdir, f"{experiment_folder}_{filename_prefix}_{name}_1D.svg")
        plt.savefig(path1, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig1)

        # -------------------------------------------------
        # PC1 vs encoded v value
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

        axv.set_title(f"{experiment_folder} — {title_prefix} — {name} (PC1 vs v value)")
        axv.set_xlabel("PC1")
        axv.set_ylabel("Encoded Value")
        plt.tight_layout()

        pathv = os.path.join(outdir, f"{experiment_folder}_{filename_prefix}_{name}_PC1_vs_value.svg")
        plt.savefig(pathv, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(figv)

        # -------------------------------------------------
        # 2D PCA
        # -------------------------------------------------
        fig2, ax2 = plt.subplots(figsize=(10, 8))

        idx = np.intersect1d(other_idx, keep_mask)
        if idx.size:
            ax2.scatter(
                x[idx], y[idx],
                color=missed_node_color,
                alpha=0.25,
                s=55
            )

        idx = np.intersect1d(person_idx, keep_mask)
        if idx.size:
            ax2.scatter(
                x[idx], y[idx],
                color=person_node_color,
                alpha=0.8,
                s=70
            )

        idx = np.intersect1d(window_idx, keep_mask)
        if idx.size:
            ax2.scatter(
                x[idx], y[idx],
                color=window_node_color,
                alpha=0.85,
                s=70
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

        ax2.set_title(f"{experiment_folder} — {title_prefix} — {name} (2D)")
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        plt.tight_layout()

        path2 = os.path.join(outdir, f"{experiment_folder}_{filename_prefix}_{name}_2D.svg")
        plt.savefig(path2, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig2)

        # -------------------------------------------------
        # 3D PCA
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
                alpha=0.9,
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

        ax3.set_title(f"{experiment_folder} — {title_prefix} — {name} (3D)")
        ax3.set_xlabel("PC1")
        ax3.set_ylabel("PC2")
        ax3.set_zlabel("PC3")
        plt.tight_layout()

        path3 = os.path.join(outdir, f"{experiment_folder}_{filename_prefix}_{name}_3D.svg")
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

def _load_init_only_pca_from_csv(experiment_folder, n_components=3, seed=42):
    exp_dir = os.path.join(BASE_DIR, experiment_folder)
    csv_path = os.path.join(exp_dir, "embeddings_and_labels.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Embeddings CSV not found: {csv_path}")

    print(f"[VIS] Loading embeddings CSV for init-only PCA: {csv_path}")
    df = pd.read_csv(csv_path)

    if "initial_value" not in df.columns:
        raise KeyError("CSV does not contain 'initial_value' column")

    if "entity" not in df.columns:
        raise KeyError("CSV does not contain 'entity' column")

    df["initial_value"] = pd.to_numeric(df["initial_value"], errors="coerce")
    init_df = df.dropna(subset=["initial_value"]).copy()
    init_df = init_df.sort_values(by="initial_value").reset_index(drop=True)

    embedding_columns = [c for c in init_df.columns if c.startswith("dim_")]
    if not embedding_columns:
        raise ValueError("No embedding columns found starting with 'dim_'")

    X = init_df[embedding_columns].to_numpy(dtype=float)

    if X.shape[0] == 0:
        raise ValueError("No rows remained after filtering to numeric initial_value")

    print(f"[VIS] Init-only PCA rows: {X.shape[0]}, dim: {X.shape[1]}")

    pca = PCA(n_components=n_components, random_state=seed)
    coords = pca.fit_transform(X)

    entity_labels = init_df["entity"].astype(str).tolist()
    init_vals = init_df["initial_value"].to_numpy(dtype=float)

    return coords, entity_labels, init_vals

def _plot_init_only_family(
    coords,
    entity_labels,
    init_vals,
    experiment_folder,
    outdir,
    show=False,
    filename_prefix="init_only",
    title_prefix="Init-Only PCA"
):
    coords = np.asarray(coords)
    entity_labels = list(entity_labels)
    init_vals = np.asarray(init_vals, dtype=float)

    if coords.shape[1] < 3:
        raise ValueError(f"Expected at least 3 PCA components, got shape {coords.shape}")

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    cmap_grad = LinearSegmentedColormap.from_list(
        "blue_red",
        ["dodgerblue", "red"]
    )

    v_idx, v_vals, person_idx, window_idx, other_idx = _parse_label_groups(entity_labels)
    all_idx = np.arange(len(entity_labels), dtype=int)

    def plot_variant(name, keep_mask):
        keep_mask = np.array(keep_mask, dtype=int)

        # 1D
        fig1, ax1 = plt.subplots(figsize=(12, 3.5))

        idx = np.intersect1d(other_idx, keep_mask)
        if idx.size:
            ax1.scatter(x[idx], np.zeros_like(x[idx]), color=missed_node_color, alpha=0.25, s=55)

        idx = np.intersect1d(person_idx, keep_mask)
        if idx.size:
            ax1.scatter(x[idx], np.zeros_like(x[idx]), color=person_node_color, alpha=0.8, s=70)

        idx = np.intersect1d(window_idx, keep_mask)
        if idx.size:
            ax1.scatter(x[idx], np.zeros_like(x[idx]), color=window_node_color, alpha=0.85, s=70)

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

        ax1.set_title(f"{experiment_folder} — {title_prefix} — {name} (1D)")
        ax1.set_xlabel("PC1")
        ax1.set_yticks([])
        ax1.axhline(0, linewidth=1, alpha=0.3)
        plt.tight_layout()

        path1 = os.path.join(outdir, f"{experiment_folder}_{filename_prefix}_{name}_1D.svg")
        plt.savefig(path1, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig1)

        # PC1 vs initial_value
        figiv, axiv = plt.subplots(figsize=(10, 8))

        idx = np.intersect1d(v_idx, keep_mask)
        if idx.size:
            vals = v_vals[np.isin(v_idx, idx)]
            ax_y = init_vals[idx]
            sciv = axiv.scatter(
                x[idx],
                ax_y,
                c=vals,
                cmap=cmap_grad,
                alpha=0.9,
                s=85
            )
            cbar = plt.colorbar(sciv, ax=axiv)
            cbar.set_label("v value")

        axiv.set_title(f"{experiment_folder} — {title_prefix} — {name} (PC1 vs initial_value)")
        axiv.set_xlabel("PC1")
        axiv.set_ylabel("initial_value")
        plt.tight_layout()

        pathiv = os.path.join(outdir, f"{experiment_folder}_{filename_prefix}_{name}_PC1_vs_init.svg")
        plt.savefig(pathiv, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(figiv)

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
            ax2.scatter(x[idx], y[idx], color=window_node_color, alpha=0.85, s=70)

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

        ax2.set_title(f"{experiment_folder} — {title_prefix} — {name} (2D)")
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        plt.tight_layout()

        path2 = os.path.join(outdir, f"{experiment_folder}_{filename_prefix}_{name}_2D.svg")
        plt.savefig(path2, bbox_inches="tight")
        if show:
            plt.show()
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
            ax3.scatter(x[idx], y[idx], z[idx], color=window_node_color, alpha=0.9, s=45)

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

        ax3.set_title(f"{experiment_folder} — {title_prefix} — {name} (3D)")
        ax3.set_xlabel("PC1")
        ax3.set_ylabel("PC2")
        ax3.set_zlabel("PC3")
        plt.tight_layout()

        path3 = os.path.join(outdir, f"{experiment_folder}_{filename_prefix}_{name}_3D.svg")
        plt.savefig(path3, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig3)

        print(f"[VIS] Saved: {path1}")
        print(f"[VIS] Saved: {pathiv}")
        print(f"[VIS] Saved: {path2}")
        print(f"[VIS] Saved: {path3}")

    plot_variant("full", all_idx)

    no_person = np.setdiff1d(all_idx, person_idx)
    plot_variant("no_person", no_person)

    no_person_window = np.setdiff1d(no_person, window_idx)
    plot_variant("no_person_window", no_person_window)

def visualize_pca(
    experiment_folder,
    outdir=None,
    show=False,
    seed=42
):
    exp_dir = os.path.join(BASE_DIR, experiment_folder)

    if outdir is None:
        outdir = os.path.join(exp_dir, "plots")

    _safe_mkdir(outdir)

    # -------------------------------------------------
    # FAMILY 1: cached PCA over all embeddings
    # -------------------------------------------------
    pca_cache = os.path.join(exp_dir, f"{experiment_folder}_pca.joblib")

    if not os.path.exists(pca_cache):
        raise FileNotFoundError(f"PCA cache not found: {pca_cache}")

    print("[VIS] Loading cached PCA (all embeddings)...")
    pca_data = joblib.load(pca_cache)

    coords_all = np.asarray(pca_data["coords"])
    labels_all = list(pca_data["entities"])

    _plot_family(
        coords=coords_all,
        entity_labels=labels_all,
        experiment_folder=experiment_folder,
        outdir=outdir,
        show=show,
        filename_prefix="all_embeddings",
        title_prefix="All Embeddings PCA"
    )

    # -------------------------------------------------
    # FAMILY 2: fresh PCA over init-only rows from CSV
    # -------------------------------------------------
    coords_init, labels_init, init_vals = _load_init_only_pca_from_csv(
        experiment_folder=experiment_folder,
        n_components=3,
        seed=seed
    )

    _plot_init_only_family(
        coords=coords_init,
        entity_labels=labels_init,
        init_vals=init_vals,
        experiment_folder=experiment_folder,
        outdir=outdir,
        show=show,
        filename_prefix="init_only",
        title_prefix="Init-Only PCA"
    )