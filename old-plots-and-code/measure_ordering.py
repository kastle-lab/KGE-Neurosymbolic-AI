import os
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA, KernelPCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, kendalltau

BASE_DIR = os.environ.get(
    "SLURM_SUBMIT_DIR",
    os.path.dirname(os.path.abspath(__file__))
)

def load_sorted_embeddings(experiment_folder):
    exp_dir = os.path.join(BASE_DIR, experiment_folder)

    csv_path = os.path.join(exp_dir, "embeddings_and_labels.csv")
    print(f"Processing Embeddings Path: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Embeddings not found: {csv_path}")

    df = pd.read_csv(csv_path)

    df["initial_value"] = pd.to_numeric(df["initial_value"], errors="coerce")
    node_vectors = df.dropna(subset=["initial_value"]).copy()
    node_vectors = node_vectors.sort_values(by="initial_value").reset_index(drop=True)

    embedding_columns = [c for c in node_vectors.columns if c.startswith("dim_")]
    if not embedding_columns:
        raise ValueError("No embedding columns found starting with 'dim_'")

    X = node_vectors[embedding_columns].to_numpy(dtype=float)
    y_order = node_vectors["initial_value"].to_numpy(dtype=float)

    return node_vectors, X, y_order, embedding_columns


def measure_ordering_from_scalar(coord_1d):
    """
    Measure whether a learned 1D coordinate preserves the sorted row order.
    Since rows are already sorted by initial_value, perfect ordering means coord_1d is increasing.
    """
    coord_1d = np.asarray(coord_1d).ravel()
    idx_true = np.arange(len(coord_1d))

    monotonic_violations = int(np.sum(np.diff(coord_1d) < 0))

    rho, rho_p = spearmanr(idx_true, coord_1d)
    tau, tau_p = kendalltau(idx_true, coord_1d)

    inversions = 0
    n = len(coord_1d)
    for i in range(n):
        for j in range(i + 1, n):
            if coord_1d[i] > coord_1d[j]:
                inversions += 1

    total_pairs = n * (n - 1) // 2
    inversion_rate = inversions / total_pairs if total_pairs > 0 else 0.0

    return {
        "monotonic_violations": monotonic_violations,
        "spearman_rho": float(abs(rho)),
        "spearman_p": float(rho_p),
        "kendall_tau": float(abs(tau)),
        "kendall_p": float(tau_p),
        "pairwise_inversions": inversions,
        "pairwise_inversion_rate": float(inversion_rate),
    }


def linear_pca_score(X):
    pca = PCA(n_components=1, random_state=42)
    X_1d = pca.fit_transform(X)
    X_back = pca.inverse_transform(X_1d)

    residuals = np.linalg.norm(X - X_back, axis=1)

    return {
        "model": pca,
        "coord_1d": X_1d.ravel(),
        "reconstructed": X_back,
        "residuals": residuals,
        "mean_distance": float(np.mean(residuals)),
        "max_distance": float(np.max(residuals)),
        "explained_variance_ratio_1d": float(pca.explained_variance_ratio_[0]),
        "ordering": measure_ordering_from_scalar(X_1d.ravel()),
    }


def kernel_pca_score(X, kernel="rbf", gamma=None, degree=3, coef0=1.0, alpha=1e-3):
    """
    Nonlinear 1D projection using Kernel PCA.
    fit_inverse_transform=True gives an approximate reconstruction back to input space.
    """
    kpca = KernelPCA(
        n_components=1,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
        fit_inverse_transform=True,
        alpha=alpha,
        eigen_solver="auto",
        random_state=42
    )

    X_1d = kpca.fit_transform(X)
    X_back = kpca.inverse_transform(X_1d)

    residuals = np.linalg.norm(X - X_back, axis=1)

    return {
        "model": kpca,
        "coord_1d": X_1d.ravel(),
        "reconstructed": X_back,
        "residuals": residuals,
        "mean_distance": float(np.mean(residuals)),
        "max_distance": float(np.max(residuals)),
        "ordering": measure_ordering_from_scalar(X_1d.ravel()),
    }


def fit_mlp_curve(X, hidden_layer_sizes=(128, 64), max_iter=5000):
    n = len(X)
    t = np.linspace(0.0, 1.0, n).reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="tanh",
        solver="adam",
        max_iter=max_iter,
        random_state=42
    )

    mlp.fit(t, X_scaled)

    X_fit_scaled = mlp.predict(t)
    X_fit = scaler.inverse_transform(X_fit_scaled)

    residuals = np.linalg.norm(X - X_fit, axis=1)

    return {
        "model": mlp,
        "scaler": scaler,
        "t_train": t.ravel(),
        "fitted_curve_points": X_fit,
        "residuals": residuals,
        "mean_distance": float(np.mean(residuals)),
        "max_distance": float(np.max(residuals)),
    }


def project_points_to_mlp_curve(X, mlp, scaler, grid_size=4000):
    t_grid = np.linspace(0.0, 1.0, grid_size).reshape(-1, 1)
    curve_scaled = mlp.predict(t_grid)
    curve = scaler.inverse_transform(curve_scaled)

    projected_t = []
    projected_points = []
    distances = []

    for x in X:
        dists = np.linalg.norm(curve - x, axis=1)
        idx = np.argmin(dists)
        projected_t.append(float(t_grid[idx, 0]))
        projected_points.append(curve[idx])
        distances.append(float(dists[idx]))

    projected_t = np.array(projected_t)
    projected_points = np.array(projected_points)
    distances = np.array(distances)

    return {
        "projected_t": projected_t,
        "projected_points": projected_points,
        "distances": distances,
        "mean_distance": float(np.mean(distances)),
        "max_distance": float(np.max(distances)),
        "ordering": measure_ordering_from_scalar(projected_t),
    }


def measure_all_methods(
    experiment_folder,
    kpca_kernel="rbf",
    kpca_gamma=None,
    kpca_degree=3,
    kpca_coef0=1.0,
    kpca_alpha=1e-3
):
    node_vectors, X, y_order, embedding_columns = load_sorted_embeddings(experiment_folder)

    print(f"Loaded {len(X)} ordered nodes")
    print(f"Embedding dimension count: {len(embedding_columns)}")

    # PCA
    pca_results = linear_pca_score(X)

    # Kernel PCA
    kpca_results = kernel_pca_score(
        X,
        kernel=kpca_kernel,
        gamma=kpca_gamma,
        degree=kpca_degree,
        coef0=kpca_coef0,
        alpha=kpca_alpha
    )

    # MLP curve
    mlp_results = fit_mlp_curve(X)
    mlp_proj = project_points_to_mlp_curve(
        X,
        mlp_results["model"],
        mlp_results["scaler"]
    )

    print("\n=== PCA ===")
    print("Explained variance ratio:", pca_results["explained_variance_ratio_1d"])
    print("Mean distance to PCA line:", pca_results["mean_distance"])
    print("Ordering:", pca_results["ordering"])

    print("\n=== KERNEL PCA ===")
    print("Kernel:", kpca_kernel)
    print("Gamma:", kpca_gamma)
    print("Mean distance to KPCA inverse image:", kpca_results["mean_distance"])
    print("Ordering:", kpca_results["ordering"])

    print("\n=== MLP CURVE ===")
    print("Mean distance to learned curve:", mlp_proj["mean_distance"])
    print("Ordering:", mlp_proj["ordering"])

    out_df = node_vectors.copy()

    out_df["pca_coord_1d"] = pca_results["coord_1d"]
    out_df["pca_distance"] = pca_results["residuals"]

    out_df["kpca_coord_1d"] = kpca_results["coord_1d"]
    out_df["kpca_distance"] = kpca_results["residuals"]

    out_df["mlp_projected_t"] = mlp_proj["projected_t"]
    out_df["mlp_distance"] = mlp_proj["distances"]

    out_path = os.path.join(BASE_DIR, experiment_folder, "manifold_ordering_results.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved results to: {out_path}")

    return {
        "pca": pca_results,
        "kpca": kpca_results,
        "mlp": mlp_results,
        "mlp_projection": mlp_proj,
        "output_csv": out_path,
    }

def write_markdown_report(
    experiment_folder,
    pca_results,
    kpca_results,
    mlp_proj,
    output_path,
    kpca_kernel,
    kpca_gamma
):
    def fmt(d):
        return f"{d:.6f}" if isinstance(d, float) else str(d)

    md = []

    md.append(f"# Manifold Ordering Report\n")
    md.append(f"**Experiment Folder:** `{experiment_folder}`\n")

    md.append(f"---\n")

    # PCA
    md.append(f"## PCA\n")
    md.append(f"- Explained Variance Ratio (1D): `{fmt(pca_results['explained_variance_ratio_1d'])}`\n")
    md.append(f"- Mean Distance to Line: `{fmt(pca_results['mean_distance'])}`\n")

    md.append(f"\n### Ordering Metrics\n")
    for k, v in pca_results["ordering"].items():
        md.append(f"- {k}: `{fmt(v)}`\n")

    md.append(f"\n---\n")

    # Kernel PCA
    md.append(f"## Kernel PCA\n")
    md.append(f"- Kernel: `{kpca_kernel}`\n")
    md.append(f"- Gamma: `{kpca_gamma}`\n")
    md.append(f"- Mean Distance to Inverse Image: `{fmt(kpca_results['mean_distance'])}`\n")

    md.append(f"\n### Ordering Metrics\n")
    for k, v in kpca_results["ordering"].items():
        md.append(f"- {k}: `{fmt(v)}`\n")

    md.append(f"\n---\n")

    # MLP
    md.append(f"## MLP Curve\n")
    md.append(f"- Mean Distance to Curve: `{fmt(mlp_proj['mean_distance'])}`\n")

    md.append(f"\n### Ordering Metrics\n")
    for k, v in mlp_proj["ordering"].items():
        md.append(f"- {k}: `{fmt(v)}`\n")

    md.append(f"\n---\n")

    # Quick interpretation section
    md.append(f"## Interpretation\n")

    md.append(f"- PCA linearity is low if explained variance is small.\n")
    md.append(f"- Compare PCA vs KPCA distances to detect nonlinearity.\n")
    md.append(f"- MLP curve tests if a parametric 1D manifold exists.\n")
    md.append(f"- Spearman ≈ global ordering quality.\n")
    md.append(f"- Kendall ≈ pairwise correctness.\n")

    with open(output_path, "w") as f:
        f.write("\n".join(md))

    print(f"Markdown report saved to: {output_path}")
    
    

def measure(experiment_folder):

    results = measure_all_methods(
        experiment_folder,
        kpca_kernel="rbf",
        kpca_gamma=None
    )

    md_path = os.path.join(
        BASE_DIR,
        experiment_folder,
        "manifold_report.md"
    )

    write_markdown_report(
        experiment_folder,
        results["pca"],
        results["kpca"],
        results["mlp_projection"],
        md_path,
        kpca_kernel="rbf",
        kpca_gamma=None
    )

if __name__ == "__main__":
    measure(experiment_folder="100ages_500people_depth_4_TransE")
    
    

