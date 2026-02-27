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
from sklearn.decomposition import PCA
import joblib
import os

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

BASE_DIR = os.environ.get(
    "SLURM_SUBMIT_DIR",
    os.path.dirname(os.path.abspath(__file__))
)

def run_pca_for_experiment(
    experiment_folder,
    n_components=3,
    seed=42,
    overwrite=True,
):
    """
    Load embeddings CSV from version2/experiment_folder,
    compute PCA, and save PCA coords + model.
    """

    exp_dir = os.path.join(BASE_DIR, "version2", experiment_folder)

    embeddings_csv = os.path.join(
        exp_dir,
        f"embeddings_and_labels.csv"
    )

    pca_cache = os.path.join(
        exp_dir,
        f"{experiment_folder}_pca.joblib"
    )

    if not os.path.exists(embeddings_csv):
        raise FileNotFoundError(f"Embeddings not found: {embeddings_csv}")

    if os.path.exists(pca_cache) and not overwrite:
        print(f"[PCA] Loading cached PCA → {pca_cache}")
        data = joblib.load(pca_cache)
        return data["coords"], data["pca"]

    print("[PCA] Loading embeddings for PCA...")
    dataframe = load_data(embeddings_csv)
    
    # extract embeddings from dataframe
    E = dataframe.drop(columns=["entity", "initial_value"]).values

    # sanity check - print embedding shape
    N, D = E.shape
    print(f"Embeddings loaded: N={N}, D={D}")
    
    # grab assigned values from dataframe
    # init_vals = df["initial_value"].values - these aren't relevant with person hasAge stuff now since the nodes themselves encode the age e.g. v20 -> 20
    
    print("[PCA] Computing PCA...")
    pca = PCA(n_components=n_components, random_state=seed)
    coords = pca.fit_transform(E)

    joblib.dump(
        {
            "coords": coords,
            "pca": pca,
            "entities": dataframe["entity"].values.tolist()
        },
        pca_cache,
    )

    print(f"[PCA] Saved PCA cache → {pca_cache}")

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

def load_data(csv_path):

    # =====================================================
    # LOAD + SORT
    # =====================================================
    print("Loading:", csv_path)
    df = pd.read_csv(csv_path)
        
    return df