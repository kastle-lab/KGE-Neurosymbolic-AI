from pathlib import Path
import argparse
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from pykeen.triples import TriplesFactory
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path("final_results/age-node-mlp-curves")
BACKGROUND = "#fffdf7"

AGE_CMAP = LinearSegmentedColormap.from_list(
    "age_blue_red",
    ["dodgerblue", "red"],
)

WINDOW_NAMES = {
    "with_windows": "With Windowing",
    "without_windows": "Without Windowing",
}


def discover_populations(root):
    root = Path(root)

    folders = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.isdigit()
        and (path / "kg_manifest.csv").exists()
        and (path / "runs").is_dir()
    ]

    folders.sort(key=lambda path: int(path.name))

    if not folders:
        raise FileNotFoundError(
            f"No population folders found under {root}"
        )

    return folders


def load_manifest(population):
    manifest = pd.read_csv(
        population / "kg_manifest.csv"
    )

    if (
        "label" in manifest.columns
        and "run" not in manifest.columns
    ):
        manifest = manifest.rename(
            columns={"label": "run"}
        )

    required = {
        "run",
        "window_condition",
        "removal_percent",
    }

    missing = required - set(manifest.columns)

    if missing:
        raise ValueError(
            f"Manifest is missing columns: {sorted(missing)}"
        )

    return manifest.sort_values(
        [
            "window_condition",
            "removal_percent",
            "run",
        ]
    )


def load_model(path):
    try:
        model = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        model = torch.load(
            path,
            map_location="cpu",
        )

    model = model.to("cpu")
    model.eval()

    return model


def load_literal_values(run_path):
    path = run_path / "kg.tsv.val"
    values = {}

    if not path.exists():
        return values

    with path.open(encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")

            if len(parts) != 2:
                continue

            try:
                values[parts[0]] = float(parts[1])
            except ValueError:
                pass

    return values


def age_from_label(label):
    match = re.fullmatch(
        r"v(-?\d+(?:\.\d+)?)",
        label,
    )

    if match is None:
        return None

    return float(match.group(1))


def load_age_nodes(run_path):
    triples = TriplesFactory.from_path_binary(
        run_path / "training_triples"
    )

    model = load_model(
        run_path / "trained_model.pkl"
    )

    with torch.no_grad():
        embeddings = (
            model.entity_representations[0](
                indices=None
            )
            .detach()
            .cpu()
            .numpy()
        )

    literal_values = load_literal_values(
        run_path
    )

    rows = []

    for label, entity_id in (
        triples.entity_to_id.items()
    ):
        label = str(label)

        if not label.startswith("v"):
            continue

        age = literal_values.get(
            label,
            age_from_label(label),
        )

        if age is not None:
            rows.append(
                (
                    label,
                    float(age),
                    embeddings[int(entity_id)],
                )
            )

    rows.sort(
        key=lambda row: row[0]
    )

    if len(rows) < 4:
        raise ValueError(
            f"Only {len(rows)} age nodes found in {run_path}"
        )

    vectors = np.vstack(
        [
            row[2]
            for row in rows
        ]
    ).astype(float)

    ages = np.asarray(
        [
            row[1]
            for row in rows
        ],
        dtype=float,
    )

    return vectors, ages


def fit_mlp_curve(
    vectors,
    hidden_layers,
    max_iter,
    grid_size,
    seed,
):
    """
    Fit a smooth nonlinear curve using only embedding geometry.

    The embeddings are first projected to 2D PCA. The MLP then
    learns a mapping from the first PCA coordinate to the complete
    2D age-node layout.
    """
    embeddings = StandardScaler().fit_transform(
        vectors
    )

    node_xy = PCA(
        n_components=2
    ).fit_transform(
        embeddings
    )

    xy_scaler = StandardScaler()

    fit_xy = xy_scaler.fit_transform(
        node_xy
    )

    latent = fit_xy[:, 0]

    latent = (
        (latent - latent.min())
        / (latent.max() - latent.min())
    )

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="tanh",
        solver="lbfgs",
        alpha=1e-5,
        max_iter=max_iter,
        random_state=seed,
    )

    mlp.fit(
        latent.reshape(-1, 1),
        fit_xy,
    )

    grid = np.linspace(
        0.0,
        1.0,
        grid_size,
    )

    curve_scaled = mlp.predict(
        grid.reshape(-1, 1)
    )

    curve_xy = xy_scaler.inverse_transform(
        curve_scaled
    )

    return node_xy, curve_xy, grid


def draw_gradient_curve(axis, curve_xy):
    segments = np.stack(
        [
            curve_xy[:-1],
            curve_xy[1:],
        ],
        axis=1,
    )

    line = LineCollection(
        segments,
        cmap=AGE_CMAP,
        norm=Normalize(
            vmin=0.0,
            vmax=1.0,
        ),
        linewidths=4.5,
        zorder=2,
    )

    line.set_array(
        np.linspace(
            0.0,
            1.0,
            len(segments),
        )
    )

    axis.add_collection(
        line
    )


def plot_run(
    vectors,
    ages,
    title,
    output_path,
    args,
):
    node_xy, curve_xy, grid = fit_mlp_curve(
        vectors=vectors,
        hidden_layers=tuple(
            args.hidden_layers
        ),
        max_iter=args.max_iter,
        grid_size=args.grid_size,
        seed=args.seed,
    )

    # Find each node's nearest point on the curve.
    nearest_indices = np.argmin(
        (
            (
                node_xy[:, None, :]
                - curve_xy[None, :, :]
            )
            ** 2
        ).sum(axis=2),
        axis=1,
    )

    # Curve direction is arbitrary. Use age only after fitting
    # to orient DodgerBlue toward lower ages.
    rho = spearmanr(
        ages,
        grid[nearest_indices],
    ).statistic

    if np.isfinite(rho) and rho < 0:
        curve_xy = curve_xy[::-1]

    age_norm = Normalize(
        vmin=float(ages.min()),
        vmax=float(ages.max()),
    )

    figure, axis = plt.subplots(
        figsize=(10, 8),
        facecolor=BACKGROUND,
    )

    axis.set_facecolor(
        BACKGROUND
    )

    # Soft dark outline behind the gradient curve.
    axis.plot(
        curve_xy[:, 0],
        curve_xy[:, 1],
        color="black",
        linewidth=7,
        alpha=0.18,
        zorder=1,
    )

    draw_gradient_curve(
        axis,
        curve_xy,
    )

    points = axis.scatter(
        node_xy[:, 0],
        node_xy[:, 1],
        c=ages,
        cmap=AGE_CMAP,
        norm=age_norm,
        s=80,
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )

    axis.set_title(
        title,
        pad=12,
    )

    axis.set_xlabel(
        "Age-node embedding PC1"
    )

    axis.set_ylabel(
        "Age-node embedding PC2"
    )

    axis.grid(
        alpha=0.22,
        linewidth=0.5,
    )

    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    colorbar = figure.colorbar(
        points,
        ax=axis,
        pad=0.03,
    )

    colorbar.set_label(
        "Ground-truth age"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=args.dpi,
        bbox_inches="tight",
        facecolor=BACKGROUND,
    )

    plt.close(
        figure
    )


def process_population(
    population,
    output_dir,
    args,
):
    population_size = int(
        population.name
    )

    manifest = load_manifest(
        population
    )

    for row in manifest.itertuples(
        index=False
    ):
        condition = str(
            row.window_condition
        )

        removal = float(
            row.removal_percent
        )

        run_path = (
            population
            / "runs"
            / str(row.run)
        )

        print(
            f"Plotting population {population_size}, "
            f"{condition}, {removal:g}% removed"
        )

        vectors, ages = load_age_nodes(
            run_path
        )

        removal_text = (
            f"{removal:g}"
            .replace(".", "p")
        )

        output_path = output_dir / (
            f"population_{population_size}_"
            f"{condition}_"
            f"removed_{removal_text}pct.png"
        )

        title = (
            "MLP Curve Through Age-Node Embeddings\n"
            f"Population {population_size} · "
            f"{WINDOW_NAMES.get(condition, condition)} · "
            f"{removal:g}% removed"
        )

        plot_run(
            vectors=vectors,
            ages=ages,
            title=title,
            output_path=output_path,
            args=args,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--basepath",
        nargs="+",
    )

    parser.add_argument(
        "--root",
        default=".",
    )

    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
    )

    parser.add_argument(
        "--hidden-layers",
        nargs="+",
        type=int,
        default=[64, 64],
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=8000,
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    if args.basepath:
        populations = [
            Path(path)
            for path in args.basepath
        ]
    else:
        populations = discover_populations(
            args.root
        )

    print(
        "Population folders: "
        + ", ".join(
            str(path)
            for path in populations
        )
    )

    output_dir = Path(
        args.output
    )

    for population in populations:
        process_population(
            population=population,
            output_dir=output_dir,
            args=args,
        )

    print(
        f"Saved plots to {output_dir}"
    )


if __name__ == "__main__":
    main()