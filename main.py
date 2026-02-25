# ======================================
# ===============IMPORTS ===============
# ======================================

# standard imports 
import os, re, random

# math imports
import numpy as np
import pandas as pd
import torch

# plotting imports
import matplotlib
matplotlib.use("Agg")  # always use headless backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# DR imports
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap.umap_ as umap
from pacmap import PaCMAP  # not used yet

# pykeen imports
from pykeen.pipeline import pipeline as pykeen_pipeline
from pykeen.triples import TriplesFactory
from pykeen.utils import set_random_seed

# =======================================
# ============== EMBEDDING ==============
# =======================================

def load_KG(path: str):
    """
    Load a KG TSV file and its associated .val file.

    If the user passes 'mygraph' → loads mygraph.tsv and mygraph.tsv.val
    If the user passes 'mygraph.tsv' → resolves to mygraph.tsv and mygraph.tsv.val
    """

    # remove .tsv if present
    base = os.path.splitext(path)[0]

    tsv_path = f"{base}.tsv"
    val_path = f"{base}.tsv.val"

    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"KG file not found: {tsv_path}")

    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Value file not found: {val_path}")

    # load KG triples (TSV)
    kg_df = pd.read_csv(tsv_path, sep="\t", header=None, names=["head", "relation", "tail"])

    # load initialization values
    init_vals = pd.read_csv(val_path, sep="\t", header=None, names=["entity", "value"])

    return kg_df, init_vals        


def process_embeddings(knowledge_graph, init_vals, model, seed, epochs, dimensions):
    """
    Train a PyKEEN model on the given KG and align .val values
    with PyKEEN's internal entity ordering.
    """

    # prep KG for embedding
    tf = TriplesFactory.from_labeled_triples(
        knowledge_graph[["head", "relation", "tail"]].values
    )

    # split for testing and validation
    training, testing, validation = tf.split([0.8, 0.1, 0.1])

    print(f"[INFO] Training {model}...")

    result = pykeen_pipeline(
        training=training,
        testing=testing,
        validation=validation,
        model=model,
        random_seed=seed,
        model_kwargs={"embedding_dim": dimensions},
        training_kwargs={"num_epochs": epochs},
    )
    print(f"[INFO] Training complete!")

    # entity ordering used by PyKEEN
    embedding_order = list(result.training.entity_to_id.keys())

    # map entities → init values
    val_dict = dict(zip(init_vals["entity"], init_vals["value"]))

    # reorder the init values to match PyKEEN's entity order
    ordered_init_vals = [
        val_dict.get(entity, None)
        for entity in embedding_order
    ]

    # warn if any .val entities were missing
    missing_e_vals = [
        e for e in embedding_order
        if re.match(r"^E\d+$", e) and e not in val_dict
    ]

    if missing_e_vals:
        print("[WARN] Missing .val entries for entities:", missing_e_vals)


    return result, ordered_init_vals, embedding_order

# ======================================================
# ============== DIMENSIONALITY REDUCTION ==============
# ======================================================

def prep_for_DR(pykeen_result, only_entities=False, return_names=False):
    """
    Extract the embedding matrix from a PyKEEN result.

    Args:
        pykeen_result : PyKEEN PipelineResult
        only_entities : If True, return only embeddings whose names match ^E\d+$
        return_names  : If True, also return the list of entity names (in order)
    """

    # Get embedding matrix (rows follow entity_to_id ordering)
    emb = pykeen_result.model.entity_representations[0]().detach().cpu().numpy()

    # THE CORRECT WAY TO GET ENTITY NAMES
    names = list(pykeen_result.training.entity_to_id.keys())

    # If not filtering, return everything
    if not only_entities:
        if return_names:
            return emb, names
        return emb

    # -------- filter only the E### entities ----------
    mask = [bool(re.match(r"^E\d+$", name)) for name in names]

    filtered_emb = emb[mask]
    filtered_names = [name for name, keep in zip(names, mask) if keep]

    if return_names:
        return filtered_emb, filtered_names

    return filtered_emb

def compute_reductions(entity_emb, seed=42):
    """Return dictionary of DR results for PCA, t-SNE, UMAP."""

    reduced_tSNE = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=30,
        max_iter=1000,
    ).fit_transform(entity_emb)

    reduced_UMAP = umap.UMAP(
        n_components=2,
        random_state=seed,
    ).fit_transform(entity_emb)

    reduced_PCA = PCA(
        n_components=3,
        random_state=seed,
    ).fit_transform(entity_emb)

    # reduced_PaCMAP = PaCMAP(random_state=seed, n_neighbors=5).fit_transform(entity_emb)

    return {
        "PCA": reduced_PCA,
        "tSNE": reduced_tSNE,
        "UMAP": reduced_UMAP,
        # "PaCMAP": reduced_PaCMAP
    }


# ======================================
# ============== PLOTTING ==============
# ======================================

def plot_DR_result(reductions, ordered_init_vals, basename, outdir=None):
    """
    Plot gradient-colored DR results for each reduction method.

    Args:
        reductions         : dict {"PCA": arr(N,2), "tSNE": arr(N,2), "UMAP": arr(N,2)}
        ordered_init_vals  : list/array of floats (length N)
        basename           : base name for output files
        outdir             : directory to save SVGs (created if missing)
    """

    if outdir:
        os.makedirs(outdir, exist_ok=True)

    vals = np.asarray(ordered_init_vals, dtype=float)
    mean_val = np.nanmean(vals)

    cmap_grad = LinearSegmentedColormap.from_list("blue_red", ["dodgerblue", "red"])

    for method, coords in reductions.items():

        if coords.shape[0] != len(vals):
            raise ValueError("Reduction rows don't match initialization values!")

        x = coords[:, 0]
        y = coords[:, 1]

        fig, ax = plt.subplots(figsize=(10, 8))

        sc = ax.scatter(
            x, y,
            c=vals,
            cmap=cmap_grad,
            s=60,
            edgecolor="k",
            linewidths=0.4,
        )

        for xi, yi, v in zip(x, y, vals):
            if not np.isnan(v):
                ax.text(
                    xi, yi, f"{v:.3f}",
                    ha="center", va="center",
                    fontsize=2.5,
                    color="white" if v > mean_val else "black",
                )

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Initialization Value")

        ax.set_title(f"{basename} — {method} — Gradient")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

        plt.tight_layout()

        if outdir:
            outpath = os.path.join(outdir, f"{basename}_{method}.svg")
            plt.savefig(outpath, format="svg", bbox_inches="tight")
            print("[Saved]", outpath)

        plt.close(fig)

# ======================================
# ============== PIPELINE ==============
# ======================================

def run_pipeline(
    kg_path,
    model="TransE",
    seed=123,
    epochs=200,
    dim=128,
    only_entities=False,
    outdir="plots",
    basename=None,
):
    """
    Full end-to-end pipeline:
      1. Load KG + .val file
      2. Train PyKEEN embedding model
      3. Extract embeddings
      4. Compute PCA, t-SNE, UMAP
      5. Plot DR results (gradient)
    
    Args:
        kg_path        : base path to KG (e.g., "arithmetic" or "arithmetic.tsv")
        model          : PyKEEN model name string
        seed           : random seed for reproducibility
        epochs         : number of training epochs
        dim            : embedding dimensionality
        only_entities  : if True, only return E### embeddings
        outdir         : output directory for SVG plots
        basename       : base name for plot files; if None, derived from kg_path

    Returns:
        result         : PyKEEN PipelineResult
        emb            : embedding matrix (possibly filtered)
        reductions     : dict of {"PCA": arr, "tSNE": arr, "UMAP": arr}
        ordered_vals   : initialization values aligned to emb
        names          : names aligned to emb
    """

    print("=====================================")
    print("         RUNNING FULL PIPELINE       ")
    print("=====================================")

    # -----------------------------------
    # 1. Load KG + val
    # -----------------------------------
    print("[STEP 1] Loading KG...")
    knowledge_graph, init_vals = load_KG(kg_path)

    # -----------------------------------
    # 2. Train embeddings
    # -----------------------------------
    print("[STEP 2] Training embeddings...")
    result, ordered_init_vals, full_order = process_embeddings(
        knowledge_graph,
        init_vals,
        model=model,
        seed=seed,
        epochs=epochs,
        dimensions=dim,
    )

    # -----------------------------------
    # 3. Extract embeddings (optionally only E###)
    # -----------------------------------
    print("[STEP 3] Extracting embedding matrix...")
    emb, names = prep_for_DR(result, only_entities=only_entities, return_names=True)

    # align init vals to filtered embedding order
    if only_entities:
        filtered_vals = []
        val_dict = dict(zip(full_order, ordered_init_vals))
        for name in names:
            filtered_vals.append(val_dict.get(name, None))
        ordered_vals = np.array(filtered_vals)
    else:
        ordered_vals = np.array(ordered_init_vals)
            
    # -----------------------------------
    # SAVE E-ENTITY ARTIFACTS AS A SINGLE CSV
    # -----------------------------------
    if only_entities:
        save_base = os.path.join(outdir, f"{basename}_E_entities.csv")

        # build a combined dataframe
        d = emb.shape[1]
        df = pd.DataFrame(emb, columns=[f"dim_{i}" for i in range(d)])
        df.insert(0, "entity", names)
        df["initial_value"] = ordered_vals

        # save CSV
        df.to_csv(save_base, index=False)
        print("[Saved E-entity combined CSV]", save_base)

    # -----------------------------------
    # 4. Compute dimensionality reductions
    # -----------------------------------
    print("[STEP 4] Computing DR reductions (PCA, t-SNE, UMAP)...")
    reductions = compute_reductions(emb, seed=seed)

    # -----------------------------------
    # 5. Plot
    # -----------------------------------
    print("[STEP 5] Plotting results...")
    os.makedirs(outdir, exist_ok=True)

    # derive basename if not supplied
    if basename is None:
        basename = os.path.splitext(os.path.basename(f"{kg_path}_{model}"))[0]

    plot_DR_result(
        reductions,
        ordered_vals,
        basename=basename,
        outdir=outdir,
    )

    print("=====================================")
    print("     PIPELINE COMPLETE — SAVED TO     ")
    print(f"          {os.path.abspath(outdir)}")
    print("=====================================")

    return result, emb, reductions, ordered_vals, names


if __name__ == "__main__":

    run_pipeline(
        kg_path="./KGs/E-1000_uniform_d-8_p-16_pairwise_e3",
        model="complex",
        seed=42,
        epochs=100,
        dim=300,
        only_entities=True,
        outdir="data",
        # basename="E-100_uniform_d-8_p-4_pairwise_e3_MuRE_d300_e100"  # optional custom name
    )
