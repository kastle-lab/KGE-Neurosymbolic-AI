import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, kendalltau
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from load_vectors import load_age_vectors

def load_vals(run_path):
    vals_path = run_path / "kg.tsv.val"
    vals = {}

    with open(vals_path, "r") as f:
        for line in f:
            entity, value = line.strip().split("\t")
            vals[entity] = float(value)

    return vals

def load_ordered_age_embeddings(run_path):
    age_vectors = load_age_vectors(
        run_path=run_path,
        relation_filter="hasAge",
        v_prefix="v",
    )

    vals = load_vals(run_path)

    rows = []

    for _, item in age_vectors.items():
        v_node = item["v_node"]

        if v_node not in vals:
            continue

        rows.append({
            "v_node": v_node,
            "v_id": item["v_id"],
            "true_age": vals[v_node],   # ← THIS IS THE FIX
            "v_vec": item["v_vec"],
        })

    rows = sorted(rows, key=lambda x: x["true_age"])

    X = np.stack([row["v_vec"] for row in rows])
    y = np.array([row["true_age"] for row in rows], dtype=float)

    return rows, X, y

def measure_ordering(coord_1d, true_age):
    coord_1d = np.asarray(coord_1d).ravel()
    true_age = np.asarray(true_age).ravel()

    rho, rho_p = spearmanr(true_age, coord_1d)
    tau, tau_p = kendalltau(true_age, coord_1d)

    # Flip orientation if the learned coordinate runs backwards.
    if rho < 0:
        coord_1d = -coord_1d
        rho, rho_p = spearmanr(true_age, coord_1d)
        tau, tau_p = kendalltau(true_age, coord_1d)

    monotonic_violations = int(np.sum(np.diff(coord_1d) < 0))

    inversions = 0
    n = len(coord_1d)

    for i in range(n):
        for j in range(i + 1, n):
            if coord_1d[i] > coord_1d[j]:
                inversions += 1

    total_pairs = n * (n - 1) // 2
    inversion_rate = inversions / total_pairs if total_pairs > 0 else 0.0

    return {
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "kendall_tau": float(tau),
        "kendall_p": float(tau_p),
        "monotonic_violations": monotonic_violations,
        "pairwise_inversions": inversions,
        "pairwise_inversion_rate": float(inversion_rate),
        "oriented_coord": coord_1d,
    }


def linear_pca_projection(X, true_age):
    pca = PCA(n_components=1, random_state=42)

    coord = pca.fit_transform(X).ravel()
    reconstructed = pca.inverse_transform(coord.reshape(-1, 1))
    distances = np.linalg.norm(X - reconstructed, axis=1)

    ordering = measure_ordering(coord, true_age)

    return {
        "method": "pca",
        "coord": ordering["oriented_coord"],
        "distance": distances,
        "mean_distance": float(np.mean(distances)),
        "max_distance": float(np.max(distances)),
        "explained_variance_ratio_1d": float(pca.explained_variance_ratio_[0]),
        "ordering": ordering,
    }


def kernel_pca_projection(X, true_age):
    kpca = KernelPCA(
        n_components=1,
        kernel="rbf",
        gamma=None,
        fit_inverse_transform=True,
        alpha=1e-3,
        random_state=42,
    )

    coord = kpca.fit_transform(X).ravel()
    reconstructed = kpca.inverse_transform(coord.reshape(-1, 1))
    distances = np.linalg.norm(X - reconstructed, axis=1)

    ordering = measure_ordering(coord, true_age)

    return {
        "method": "kernel_pca",
        "coord": ordering["oriented_coord"],
        "distance": distances,
        "mean_distance": float(np.mean(distances)),
        "max_distance": float(np.max(distances)),
        "explained_variance_ratio_1d": None,
        "ordering": ordering,
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
        random_state=42,
    )

    mlp.fit(t, X_scaled)

    return mlp, scaler


def project_to_mlp_curve(X, true_age, grid_size=4000):
    mlp, scaler = fit_mlp_curve(X)

    t_grid = np.linspace(0.0, 1.0, grid_size).reshape(-1, 1)

    curve_scaled = mlp.predict(t_grid)
    curve = scaler.inverse_transform(curve_scaled)

    projected_t = []
    projected_distances = []

    for x in X:
        dists = np.linalg.norm(curve - x, axis=1)
        idx = int(np.argmin(dists))

        projected_t.append(float(t_grid[idx, 0]))
        projected_distances.append(float(dists[idx]))

    projected_t = np.array(projected_t)
    projected_distances = np.array(projected_distances)

    ordering = measure_ordering(projected_t, true_age)

    return {
        "method": "mlp_curve",
        "coord": ordering["oriented_coord"],
        "distance": projected_distances,
        "mean_distance": float(np.mean(projected_distances)),
        "max_distance": float(np.max(projected_distances)),
        "explained_variance_ratio_1d": None,
        "ordering": ordering,
    }


def result_row(run_label, result, n_age_nodes):
    ordering = result["ordering"]

    return {
        "run": run_label,
        "method": result["method"],
        "n_age_nodes": n_age_nodes,

        "spearman_rho": ordering["spearman_rho"],
        "spearman_p": ordering["spearman_p"],

        "kendall_tau": ordering["kendall_tau"],
        "kendall_p": ordering["kendall_p"],

        "monotonic_violations": ordering["monotonic_violations"],
        "pairwise_inversions": ordering["pairwise_inversions"],
        "pairwise_inversion_rate": ordering["pairwise_inversion_rate"],

        "mean_distance_to_function": result["mean_distance"],
        "max_distance_to_function": result["max_distance"],

        "explained_variance_ratio_1d": result["explained_variance_ratio_1d"],
    }


def evaluate_run(basepath, run_folder, run_label, output_dir):
    run_path = Path(basepath) / "runs" / run_folder

    rows, X, true_age = load_ordered_age_embeddings(run_path)

    results = [
        linear_pca_projection(X, true_age),
        kernel_pca_projection(X, true_age),
        project_to_mlp_curve(X, true_age),
    ]

    detail_df = pd.DataFrame({
        "run": run_label,
        "v_node": [row["v_node"] for row in rows],
        "v_id": [row["v_id"] for row in rows],
        "true_age": true_age,
    })

    summary_rows = []

    for result in results:
        method = result["method"]

        detail_df[f"{method}_coord"] = result["coord"]
        detail_df[f"{method}_distance"] = result["distance"]

        summary_rows.append(
            result_row(
                run_label=run_label,
                result=result,
                n_age_nodes=len(true_age),
            )
        )

    detail_path = output_dir / f"age_manifold_details_{run_label}.csv"
    detail_df.to_csv(detail_path, index=False)

    return summary_rows


def analyze_age_manifold(basepath, removal_start=2, removal_stop=8, single_run=None):
    
    basepath = Path(basepath)
    output_dir = basepath / "age_manifold_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    if single_run is not None:
        runs = [(single_run, single_run)]
    else:
        runs = [("original", "original")]

        for n in range(removal_start, removal_stop + 1):
            runs.append((f"every_{n}_removed", f"e{n}"))

    all_summary_rows = []

    for run_folder, run_label in runs:
        print(f"Analyzing age-node manifold for {run_label}")

        summary_rows = evaluate_run(
            basepath=basepath,
            run_folder=run_folder,
            run_label=run_label,
            output_dir=output_dir,
        )

        all_summary_rows.extend(summary_rows)

    summary_df = pd.DataFrame(all_summary_rows)
    
    summary_df = summary_df.rename(columns={
        "n_age_nodes": "n",
        "spearman_rho": "rho",
        "spearman_p": "rho_p",
        "kendall_tau": "tau",
        "kendall_p": "tau_p",
        "monotonic_violations": "mono_v",
        "pairwise_inversions": "inv",
        "pairwise_inversion_rate": "inv_rate",
        "mean_distance_to_function": "mean_d",
        "max_distance_to_function": "max_d",
        "explained_variance_ratio_1d": "EVR_1D",
    })
    
    # drop kernel PCA rows
    summary_df = summary_df[summary_df["method"] != "kernel_pca"]

    # drop columns
    summary_df = summary_df.drop(columns=["run", "n"])
    
    for col in summary_df.columns:
        if summary_df[col].dtype in ["float64", "float32"]:
            summary_df[col] = summary_df[col].round(4)

    summary_path = output_dir / "age_manifold_summary.csv"
    summary_df.to_csv(summary_path, index=False)


    print(f"\nSaved summary to: {summary_path}")
    print(summary_df.to_string(index=False))

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--single-run",
        default=None,
        help="Analyze one run folder inside basepath/runs, e.g. emb",
    )

    parser.add_argument(
        "--basepath",
        required=True,
        help="Experiment folder, e.g. ./100people or ./500people",
    )

    parser.add_argument(
        "--removal-start",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--removal-stop",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    analyze_age_manifold(
        basepath=args.basepath,
        removal_start=args.removal_start,
        removal_stop=args.removal_stop,
        single_run=args.single_run,
    )

