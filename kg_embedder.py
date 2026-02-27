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

BASE_DIR = os.environ.get("SLURM_SUBMIT_DIR", os.path.dirname(os.path.abspath(__file__)))

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

    # return the kg dataframe and the assigned init values
    return kg_df, init_vals        

def process_embeddings(knowledge_graph, model, seed, epochs, dimensions):
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

    return result

# ======================================================
# ============== DIMENSIONALITY REDUCTION ==============
# ======================================================

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
# ============== PIPELINE ==============
# ======================================

def create_embeddings(
    experiment_folder,
    model="mure",
    seed=42,
    epochs=200,
    dim=128
):
    """
    (Don't read these comments they're basically useless)
    Full end-to-end pipeline (not really this is a lie):
      1. Load KG + .val file
      2. Train PyKEEN embedding model
      3. Extract embeddings
      4. Compute PCA, t-SNE, UMAP
      5. Stop reading
      6. Plot DR results (gradient)
    
    Args:
        kg_path        : base path to KG (e.g., "arithmetic" or "arithmetic.tsv")
        model          : PyKEEN model name string
        seed           : random seed for reproducibility
        epochs         : number of training epochs
        dim            : embedding dimensionality

    Returns (nothing anymore don't read this!):
        result         : PyKEEN PipelineResult
        emb            : embedding matrix (possibly filtered)
        reductions     : dict of {"PCA": arr, "tSNE": arr, "UMAP": arr}
        ordered_vals   : initialization values aligned to emb
        names          : names aligned to emb
    """

    print("\n=====================================")
    print("          CREATING EMEBDDINGS        ")
    print("=====================================")

    # -----------------------------------
    # 1. Load KG + vals
    # -----------------------------------
    kg_base = os.path.join(BASE_DIR, experiment_folder, "kg")

    print("[STEP 1] Loading KG...")
    knowledge_graph, init_vals = load_KG(kg_base)

    # -----------------------------------
    # 2. Train embeddings
    # -----------------------------------
    print("[STEP 2] Training embeddings...")
    result= process_embeddings(
        knowledge_graph,
        model=model,
        seed=seed,
        epochs=epochs,
        dimensions=dim,
    )

    # -----------------------------------
    # 3. Extract embeddings
    # -----------------------------------
    print("[STEP 3] Extracting embedding matrix...")
    
    # Get embedding matrix (rows follow entity_to_id ordering) - all the embedded vectors
    emb = result.model.entity_representations[0]().detach().cpu().numpy()
    
    # entity ordering used by PyKEEN - all the entity label in the order pykeen results
    embedding_order = list(result.training.entity_to_id.keys())

    # map entity labels to corresponding init values
    val_dict = dict(zip(init_vals["entity"], init_vals["value"]))

    # reorder the init values to match PyKEEN's entity order
    # for each entity label in the list of entity labels, if the entity in the vals dict matches, add the entity and the label so the order matches the order of vectors explicitly
    ordered_vals = [
        val_dict.get(entity, None)
        for entity in embedding_order
    ]
                
    # -----------------------------------
    # SAVE EMBEDDINGS INTO EXPERIMENT DIR
    # -----------------------------------
    # path stuff
    exp_dir = os.path.join(BASE_DIR, experiment_folder)
    os.makedirs(exp_dir, exist_ok=True)
    save_path = os.path.join(exp_dir, "embeddings_and_labels.csv")

    # size of the matrix horizontally aka the number of vectors
    d = emb.shape[1] 
    
    # fill in the dataframe with the vectors
    df = pd.DataFrame(emb, columns=[f"dim_{i}" for i in range(d)])
    
    # insert in the 0th position the entity labels for each vector in a column
    df.insert(0, "entity", embedding_order) 

    df["initial_value"] = ordered_vals

    df.to_csv(save_path, index=False)
    print("[Saved embeddings CSV]", save_path)


