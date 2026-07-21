from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pykeen.triples import TriplesFactory
from scipy.stats import kendalltau, spearmanr
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# ============================================================
# Input and output configuration
# ============================================================

MANIFEST_FILE = "kg_manifest.csv"

DEFAULT_OUTPUT_DIRECTORY = Path("final_results")

MARKDOWN_FILENAME = "age-monotonicity.md"
CSV_FILENAME = "age-monotonicity.csv"
PLOT_DIRECTORY_NAME = "age-monotonicity-plots"

AGE_PREFIX = "v"

WINDOW_LABELS = {
    "with_windows": "With Windowing",
    "without_windows": "Without Windowing",
}

WINDOW_ORDER = {
    "with_windows": 0,
    "without_windows": 1,
}

PLOT_BACKGROUND = "#fffdf7"


# ============================================================
# General helpers
# ============================================================

def numeric_suffix(label, prefix=AGE_PREFIX):
    """
    Extract a numeric suffix from a label.

    Examples:
        v18     -> 18.0
        v-2     -> -2.0
        v18.5   -> 18.5
    """
    if label is None:
        return None

    match = re.fullmatch(
        rf"{re.escape(prefix)}(-?\d+(?:\.\d+)?)",
        str(label).strip(),
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return float(match.group(1))


def read_csv(path):
    """Read and validate a non-empty CSV file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Required file was not found: {path}"
        )

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(
            f"CSV contains no rows: {path}"
        )

    return dataframe


def load_torch_model(model_path):
    """
    Load a saved PyTorch model on the CPU.

    The fallback supports PyTorch versions that do not accept
    the weights_only argument.
    """
    model_path = Path(model_path)

    try:
        model = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        model = torch.load(
            model_path,
            map_location="cpu",
        )

    model = model.to("cpu")
    model.eval()

    return model


# ============================================================
# Population and manifest discovery
# ============================================================

def load_manifest(population_directory):
    """Load and validate kg_manifest.csv for one population."""
    population_directory = Path(population_directory)
    manifest_path = population_directory / MANIFEST_FILE

    manifest = read_csv(manifest_path)

    if "label" in manifest.columns:
        manifest = manifest.rename(
            columns={"label": "run"}
        )

    required_columns = {
        "run",
        "window_condition",
        "removal_percent",
    }

    missing_columns = (
        required_columns - set(manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{manifest_path} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    manifest = manifest.copy()

    manifest["run"] = (
        manifest["run"]
        .astype(str)
    )

    manifest["window_condition"] = (
        manifest["window_condition"]
        .astype(str)
    )

    manifest["removal_percent"] = pd.to_numeric(
        manifest["removal_percent"],
        errors="raise",
    )

    unexpected_conditions = (
        set(manifest["window_condition"])
        - set(WINDOW_LABELS)
    )

    if unexpected_conditions:
        raise ValueError(
            f"Unexpected window conditions in {manifest_path}: "
            f"{sorted(unexpected_conditions)}"
        )

    duplicate_conditions = manifest.duplicated(
        [
            "window_condition",
            "removal_percent",
        ],
        keep=False,
    )

    if duplicate_conditions.any():
        duplicates = manifest.loc[
            duplicate_conditions,
            [
                "run",
                "window_condition",
                "removal_percent",
            ],
        ]

        raise ValueError(
            "Multiple runs were found for the same window condition "
            "and removal percentage. This script expects one run per "
            "condition:\n"
            f"{duplicates.to_string(index=False)}"
        )

    manifest["_window_order"] = (
        manifest["window_condition"]
        .map(WINDOW_ORDER)
    )

    return manifest.sort_values(
        [
            "_window_order",
            "removal_percent",
            "run",
        ]
    ).reset_index(drop=True)


def discover_population_directories(root="."):
    """
    Discover numeric population folders such as:

        100/
        200/
        500/
    """
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(
            f"Root directory does not exist: {root}"
        )

    directories = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.isdigit()
        and (path / MANIFEST_FILE).exists()
        and (path / "runs").is_dir()
    ]

    directories.sort(
        key=lambda path: int(path.name)
    )

    if not directories:
        raise ValueError(
            f"No numeric population experiment folders were "
            f"found under {root}."
        )

    return directories


# ============================================================
# Age-node values and embeddings
# ============================================================

def load_value_map(run_path):
    """
    Load numeric literal values from run_path/kg.tsv.val.

    Expected format:

        v18<TAB>18
        v19<TAB>19

    If the file is unavailable, numeric suffixes such as v18 are
    used as the age values instead.
    """
    run_path = Path(run_path)
    values_path = run_path / "kg.tsv.val"

    if not values_path.exists():
        return {}

    values = {}

    with values_path.open(
        "r",
        encoding="utf-8",
    ) as infile:

        for line_number, line in enumerate(
            infile,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 2:
                raise ValueError(
                    f"Malformed row in {values_path} at line "
                    f"{line_number}: expected two columns."
                )

            entity, value = parts

            try:
                values[entity] = float(value)
            except ValueError:
                # Allow a possible first-line header.
                if line_number == 1:
                    continue

                raise ValueError(
                    f"Invalid numeric value in {values_path} at "
                    f"line {line_number}: {value!r}"
                )

    return values


def load_age_embedding_data(run_path):
    """
    Load all numeric age-node embeddings from one trained run.

    The embeddings are not sorted according to age. They are sorted
    only by node label to keep the input order deterministic.

    Numeric ages are returned separately and are used only after the
    PCA and MLP coordinates have been learned.
    """
    run_path = Path(run_path)

    triples_path = run_path / "training_triples"
    model_path = run_path / "trained_model.pkl"

    if not triples_path.exists():
        raise FileNotFoundError(
            f"Training triples were not found: {triples_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model was not found: {model_path}"
        )

    triples_factory = TriplesFactory.from_path_binary(
        triples_path
    )

    model = load_torch_model(model_path)

    with torch.no_grad():
        entity_embeddings = (
            model.entity_representations[0](indices=None)
            .detach()
            .cpu()
            .numpy()
        )

    value_map = load_value_map(run_path)

    rows = []

    for entity_label, entity_id in (
        triples_factory.entity_to_id.items()
    ):
        entity_label = str(entity_label)

        if not entity_label.startswith(AGE_PREFIX):
            continue

        if entity_label in value_map:
            true_age = value_map[entity_label]
        else:
            true_age = numeric_suffix(
                entity_label,
                AGE_PREFIX,
            )

        if true_age is None:
            continue

        rows.append(
            {
                "v_node": entity_label,
                "v_id": int(entity_id),
                "true_age": float(true_age),
                "v_vec": np.asarray(
                    entity_embeddings[int(entity_id)],
                    dtype=float,
                ),
            }
        )

    # Important:
    # do not sort using true_age, because that would give the MLP
    # the correct age order indirectly.
    rows.sort(
        key=lambda row: row["v_node"]
    )

    if len(rows) < 3:
        raise ValueError(
            f"At least three numeric age nodes are required in "
            f"{run_path}; found {len(rows)}."
        )

    vectors = np.vstack(
        [
            row["v_vec"]
            for row in rows
        ]
    )

    true_ages = np.asarray(
        [
            row["true_age"]
            for row in rows
        ],
        dtype=float,
    )

    if len(np.unique(true_ages)) < 2:
        raise ValueError(
            f"Age nodes in {run_path} do not contain at least "
            f"two distinct numeric values."
        )

    return rows, vectors, true_ages


# ============================================================
# Monotonicity evaluation
# ============================================================

def measure_ordering(coordinate, true_age):
    """
    Compare a learned one-dimensional coordinate with true age.

    Spearman rho and Kendall tau-b measure ordering rather than
    numeric error.

    A one-dimensional axis has arbitrary direction, so the sign is
    reversed when necessary before reporting the correlations.
    """
    coordinate = np.asarray(
        coordinate,
        dtype=float,
    ).ravel()

    true_age = np.asarray(
        true_age,
        dtype=float,
    ).ravel()

    if len(coordinate) != len(true_age):
        raise ValueError(
            "Coordinate and true-age arrays have different lengths."
        )

    valid = (
        np.isfinite(coordinate)
        & np.isfinite(true_age)
    )

    coordinate = coordinate[valid]
    true_age = true_age[valid]

    if (
        len(coordinate) < 2
        or len(np.unique(coordinate)) < 2
        or len(np.unique(true_age)) < 2
    ):
        return {
            "rho": np.nan,
            "rho_p": np.nan,
            "tau": np.nan,
            "tau_p": np.nan,
            "coordinate": coordinate,
        }

    rho, rho_p = spearmanr(
        true_age,
        coordinate,
    )

    tau, tau_p = kendalltau(
        true_age,
        coordinate,
        variant="b",
    )

    # The learned axis can point from old to young or young to old.
    # Sign orientation is not meaningful, so orient it toward
    # increasing age for reporting.
    should_reverse = (
        np.isfinite(rho)
        and rho < 0
    )

    if (
        not should_reverse
        and np.isfinite(rho)
        and np.isclose(rho, 0)
        and np.isfinite(tau)
        and tau < 0
    ):
        should_reverse = True

    if should_reverse:
        coordinate = -coordinate

        rho, rho_p = spearmanr(
            true_age,
            coordinate,
        )

        tau, tau_p = kendalltau(
            true_age,
            coordinate,
            variant="b",
        )

    return {
        "rho": float(rho),
        "rho_p": float(rho_p),
        "tau": float(tau),
        "tau_p": float(tau_p),
        "coordinate": coordinate,
    }


# ============================================================
# Unsupervised 1D PCA
# ============================================================

def evaluate_pca_1d(vectors, true_age):
    """
    Learn an unsupervised one-dimensional linear PCA coordinate.

    True ages are not used to fit PCA. They are used only afterward
    to calculate rank correlations.
    """
    vectors = np.asarray(
        vectors,
        dtype=float,
    )

    pca = PCA(
        n_components=1,
    )

    coordinate = pca.fit_transform(
        vectors
    ).ravel()

    ordering = measure_ordering(
        coordinate=coordinate,
        true_age=true_age,
    )

    return {
        "rho": ordering["rho"],
        "rho_p": ordering["rho_p"],
        "tau": ordering["tau"],
        "tau_p": ordering["tau_p"],
        "coordinate": ordering["coordinate"],
        "explained_variance": float(
            pca.explained_variance_ratio_[0]
        ),
    }


# ============================================================
# Unsupervised nonlinear MLP principal curve
# ============================================================

def scale_to_unit_interval(values):
    """Scale a one-dimensional coordinate to [0, 1]."""
    values = np.asarray(
        values,
        dtype=float,
    ).ravel()

    minimum = float(np.min(values))
    maximum = float(np.max(values))

    if np.isclose(minimum, maximum):
        return np.zeros_like(values)

    return (
        (values - minimum)
        / (maximum - minimum)
    )


def fit_unsupervised_mlp_curve(
    vectors,
    hidden_layer_sizes=(16,),
    max_iter=5000,
    grid_size=4000,
    refinement_steps=8,
    alpha=0.1,
    tolerance=1e-4,
    seed=42,
):
    """
    Learn a nonlinear one-dimensional principal curve using only
    the age-node embeddings.

    Numeric ages are not used during fitting.

    Process:

      1. Standardize embedding dimensions.
      2. Initialize a 1D coordinate using PCA.
      3. Train an MLP that maps the 1D coordinate to embedding space.
      4. Project each embedding onto its nearest location on the curve.
      5. Use the projected curve locations as updated coordinates.
      6. Repeat until stable or until refinement_steps is reached.

    The final curve parameter is the unsupervised MLP coordinate.
    """
    vectors = np.asarray(
        vectors,
        dtype=float,
    )

    if vectors.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional embedding matrix, "
            f"got shape {vectors.shape}."
        )

    if len(vectors) < 3:
        raise ValueError(
            "At least three age-node embeddings are required."
        )

    scaler = StandardScaler()

    scaled_vectors = scaler.fit_transform(
        vectors
    )

    # PCA provides an unsupervised initial coordinate.
    initial_coordinate = PCA(
        n_components=1,
    ).fit_transform(
        scaled_vectors
    ).ravel()

    coordinate = scale_to_unit_interval(
        initial_coordinate
    )

    parameter_grid = np.linspace(
        0.0,
        1.0,
        grid_size,
    )

    final_curve = None
    final_distances = None
    refinement_count = 0

    for refinement_index in range(
        refinement_steps
    ):
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="tanh",

            # More stable than L-BFGS for this repeatedly refitted
            # multi-output principal-curve model.
            solver="adam",

            # Stronger regularization discourages an unnecessarily
            # complicated or looping curve.
            alpha=alpha,

            learning_rate_init=1e-3,
            max_iter=max_iter,
            tol=1e-5,
            n_iter_no_change=200,

            random_state=seed + refinement_index,
        )

        # Learn:
        #
        #     unsupervised 1D coordinate -> embedding vector
        #
        # No age values are passed here.
        model.fit(
            coordinate.reshape(-1, 1),
            scaled_vectors,
        )

        curve = model.predict(
            parameter_grid.reshape(-1, 1)
        )

        new_coordinate = np.empty(
            len(scaled_vectors),
            dtype=float,
        )

        projected_distances = np.empty(
            len(scaled_vectors),
            dtype=float,
        )

        for vector_index, vector in enumerate(
            scaled_vectors
        ):
            squared_distances = np.sum(
                (curve - vector) ** 2,
                axis=1,
            )

            nearest_index = int(
                np.argmin(squared_distances)
            )

            new_coordinate[vector_index] = (
                parameter_grid[nearest_index]
            )

            projected_distances[vector_index] = float(
                np.sqrt(
                    squared_distances[nearest_index]
                )
            )

        # Keep orientation consistent between refinement iterations.
        # This does not use true age.
        if (
            np.std(coordinate) > 0
            and np.std(new_coordinate) > 0
        ):
            orientation_correlation = np.corrcoef(
                coordinate,
                new_coordinate,
            )[0, 1]

            if (
                np.isfinite(orientation_correlation)
                and orientation_correlation < 0
            ):
                new_coordinate = (
                    1.0 - new_coordinate
                )

        maximum_change = float(
            np.max(
                np.abs(
                    new_coordinate - coordinate
                )
            )
        )

        coordinate = new_coordinate
        final_curve = curve
        final_distances = projected_distances
        refinement_count = refinement_index + 1

        if maximum_change < tolerance:
            break

    if final_curve is None or final_distances is None:
        raise RuntimeError(
            "The MLP principal-curve procedure did not complete."
        )

    return {
        "coordinate": coordinate,
        "curve": final_curve,
        "distance": final_distances,
        "mean_distance": float(
            np.mean(final_distances)
        ),
        "max_distance": float(
            np.max(final_distances)
        ),
        "refinement_count": refinement_count,
    }


def evaluate_mlp_monotonicity(
    vectors,
    true_age,
    hidden_layer_sizes=(16,),
    max_iter=5000,
    grid_size=4000,
    refinement_steps=8,
    alpha=0.1,
    tolerance=1e-4,
    seed=42,
):
    """
    Learn an unsupervised nonlinear MLP principal curve and then
    compare its 1D coordinate with ground-truth age.

    True age is used only in measure_ordering().
    """
    curve_result = fit_unsupervised_mlp_curve(
        vectors=vectors,
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        grid_size=grid_size,
        refinement_steps=refinement_steps,
        alpha=alpha,
        tolerance=tolerance,
        seed=seed,
    )

    ordering = measure_ordering(
        coordinate=curve_result["coordinate"],
        true_age=true_age,
    )

    return {
        "rho": ordering["rho"],
        "rho_p": ordering["rho_p"],
        "tau": ordering["tau"],
        "tau_p": ordering["tau_p"],
        "coordinate": ordering["coordinate"],
        "mean_distance": curve_result["mean_distance"],
        "max_distance": curve_result["max_distance"],
        "refinement_count": curve_result[
            "refinement_count"
        ],
    }


# ============================================================
# Per-run evaluation
# ============================================================

def evaluate_run(
    population_size,
    run_path,
    run_label,
    window_condition,
    removal_percent,
    hidden_layer_sizes=(16,),
    max_iter=5000,
    grid_size=4000,
    refinement_steps=8,
    mlp_alpha=0.1,
    tolerance=1e-4,
    seed=42,
):
    """
    Evaluate linear PCA monotonicity and nonlinear MLP principal-curve
    monotonicity for one trained embedding run.
    """
    _, vectors, true_age = load_age_embedding_data(
        run_path
    )

    pca_result = evaluate_pca_1d(
        vectors=vectors,
        true_age=true_age,
    )

    mlp_result = evaluate_mlp_monotonicity(
        vectors=vectors,
        true_age=true_age,
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        grid_size=grid_size,
        refinement_steps=refinement_steps,
        alpha=mlp_alpha,
        tolerance=tolerance,
        seed=seed,
    )

    return {
        "population_size": int(
            population_size
        ),
        "run": str(
            run_label
        ),
        "window_condition": str(
            window_condition
        ),
        "removal_percent": float(
            removal_percent
        ),
        "n_age_nodes": int(
            len(true_age)
        ),

        "pca_rho": pca_result["rho"],
        "pca_rho_p": pca_result["rho_p"],
        "pca_tau": pca_result["tau"],
        "pca_tau_p": pca_result["tau_p"],
        "pca_explained_variance": (
            pca_result["explained_variance"]
        ),

        "mlp_rho": mlp_result["rho"],
        "mlp_rho_p": mlp_result["rho_p"],
        "mlp_tau": mlp_result["tau"],
        "mlp_tau_p": mlp_result["tau_p"],

        # Diagnostic only; not part of the main monotonicity table.
        "mlp_mean_curve_distance": (
            mlp_result["mean_distance"]
        ),
        "mlp_max_curve_distance": (
            mlp_result["max_distance"]
        ),
        "mlp_refinement_count": (
            mlp_result["refinement_count"]
        ),
    }


def evaluate_population_directory(
    population_directory,
    hidden_layer_sizes=(16,),
    max_iter=5000,
    grid_size=4000,
    refinement_steps=8,
    mlp_alpha=0.1,
    tolerance=1e-4,
    seed=42,
):
    """Evaluate every manifest run for one population folder."""
    population_directory = Path(
        population_directory
    )

    population_size = int(
        population_directory.name
    )

    manifest = load_manifest(
        population_directory
    )

    rows = []

    for manifest_row in manifest.itertuples(
        index=False
    ):
        run_label = str(
            manifest_row.run
        )

        run_path = (
            population_directory
            / "runs"
            / run_label
        )

        if not run_path.is_dir():
            raise FileNotFoundError(
                f"Run directory was not found: {run_path}"
            )

        print(
            f"Analyzing population {population_size}, "
            f"{manifest_row.window_condition}, "
            f"{float(manifest_row.removal_percent):g}% removed"
        )

        rows.append(
            evaluate_run(
                population_size=population_size,
                run_path=run_path,
                run_label=run_label,
                window_condition=(
                    manifest_row.window_condition
                ),
                removal_percent=(
                    manifest_row.removal_percent
                ),
                hidden_layer_sizes=(
                    hidden_layer_sizes
                ),
                max_iter=max_iter,
                grid_size=grid_size,
                refinement_steps=refinement_steps,
                mlp_alpha=mlp_alpha,
                tolerance=tolerance,
                seed=seed,
            )
        )

    return rows


# ============================================================
# Output formatting
# ============================================================

def format_number(value, digits=4):
    if pd.isna(value):
        return "—"

    return f"{float(value):.{digits}f}"


def format_percent(value):
    if pd.isna(value):
        return "—"

    return f"{float(value):g}%"


def escape_markdown(value):
    return (
        str(value)
        .replace("|", "\\|")
        .replace("\n", " ")
    )


def dataframe_to_markdown(dataframe):
    """Convert a DataFrame to Markdown without requiring tabulate."""
    if dataframe.empty:
        return "_No cases are available._\n"

    headers = [
        escape_markdown(column)
        for column in dataframe.columns
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(
            ["---"] * len(headers)
        ) + " |",
    ]

    for row in dataframe.itertuples(
        index=False,
        name=None,
    ):
        lines.append(
            "| "
            + " | ".join(
                escape_markdown(value)
                for value in row
            )
            + " |"
        )

    return "\n".join(lines) + "\n"


def create_markdown_table(
    results,
    window_condition,
):
    """Create one of the two requested report tables."""
    table_rows = results.loc[
        results["window_condition"]
        == window_condition
    ].copy()

    table_rows = table_rows.sort_values(
        [
            "population_size",
            "removal_percent",
        ],
        ascending=[
            False,
            True,
        ],
    )

    return pd.DataFrame(
        {
            "Population Size": (
                table_rows["population_size"]
                .astype(int)
            ),
            "Removal %": (
                table_rows["removal_percent"]
                .map(format_percent)
            ),
            "PCA Spearman ρ": (
                table_rows["pca_rho"]
                .map(format_number)
            ),
            "PCA Kendall τb": (
                table_rows["pca_tau"]
                .map(format_number)
            ),
            "MLP Spearman ρ": (
                table_rows["mlp_rho"]
                .map(format_number)
            ),
            "MLP Kendall τb": (
                table_rows["mlp_tau"]
                .map(format_number)
            ),
        }
    )


def create_csv_table(results):
    """Create a combined machine-readable table."""
    ordered = results.copy()

    ordered["_window_order"] = (
        ordered["window_condition"]
        .map(WINDOW_ORDER)
    )

    ordered = ordered.sort_values(
        [
            "_window_order",
            "population_size",
            "removal_percent",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )

    return pd.DataFrame(
        {
            "Windowing": (
                ordered["window_condition"]
                .map(WINDOW_LABELS)
            ),
            "Population Size": (
                ordered["population_size"]
                .astype(int)
            ),
            "Removal %": (
                ordered["removal_percent"]
            ),
            "PCA Spearman rho": (
                ordered["pca_rho"]
            ),
            "PCA Spearman p": (
                ordered["pca_rho_p"]
            ),
            "PCA Kendall tau-b": (
                ordered["pca_tau"]
            ),
            "PCA Kendall p": (
                ordered["pca_tau_p"]
            ),
            "PCA Explained Variance 1D": (
                ordered["pca_explained_variance"]
            ),
            "MLP Spearman rho": (
                ordered["mlp_rho"]
            ),
            "MLP Spearman p": (
                ordered["mlp_rho_p"]
            ),
            "MLP Kendall tau-b": (
                ordered["mlp_tau"]
            ),
            "MLP Kendall p": (
                ordered["mlp_tau_p"]
            ),
            "MLP Mean Curve Distance": (
                ordered["mlp_mean_curve_distance"]
            ),
            "MLP Max Curve Distance": (
                ordered["mlp_max_curve_distance"]
            ),
            "MLP Refinement Count": (
                ordered["mlp_refinement_count"]
                .astype(int)
            ),
        }
    )


# ============================================================
# Plotting
# ============================================================

def configure_plot_axis(axis):
    """Apply consistent plot styling."""
    axis.set_facecolor(PLOT_BACKGROUND)

    axis.grid(
        True,
        linewidth=0.45,
        alpha=0.30,
    )

    for spine in axis.spines.values():
        spine.set_alpha(0.45)


def plot_monotonicity_by_removal(
    results,
    window_condition,
    output_path,
    dpi=300,
):
    """
    Plot PCA and MLP rank correlations across removal percentages.
    """
    window_results = results.loc[
        results["window_condition"]
        == window_condition
    ].copy()

    if window_results.empty:
        return

    population_sizes = sorted(
        window_results["population_size"].unique(),
        reverse=True,
    )

    default_colors = plt.rcParams[
        "axes.prop_cycle"
    ].by_key()["color"]

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(15, 6),
        sharex=True,
        sharey=True,
        facecolor=PLOT_BACKGROUND,
    )

    metric_definitions = (
        (
            "pca_rho",
            "mlp_rho",
            "Spearman ρ",
        ),
        (
            "pca_tau",
            "mlp_tau",
            "Kendall τb",
        ),
    )

    for axis, (
        pca_column,
        mlp_column,
        metric_label,
    ) in zip(axes, metric_definitions):

        configure_plot_axis(axis)

        for population_index, population_size in enumerate(
            population_sizes
        ):
            population_results = window_results.loc[
                window_results["population_size"]
                == population_size
            ].sort_values("removal_percent")

            color = default_colors[
                population_index % len(default_colors)
            ]

            axis.plot(
                population_results["removal_percent"],
                population_results[pca_column],
                marker="o",
                linestyle="-",
                linewidth=2.1,
                markersize=6,
                color=color,
                label=(
                    f"PCA — population "
                    f"{population_size}"
                ),
            )

            axis.plot(
                population_results["removal_percent"],
                population_results[mlp_column],
                marker="s",
                linestyle="--",
                linewidth=2.1,
                markersize=6,
                color=color,
                label=(
                    f"MLP — population "
                    f"{population_size}"
                ),
            )

        axis.set_title(metric_label)

        axis.set_xlabel(
            "Removed hasAge relations (%)"
        )

        axis.set_ylabel(
            "Rank correlation"
        )

        axis.set_ylim(
            -0.05,
            1.05,
        )

        removal_levels = sorted(
            window_results[
                "removal_percent"
            ].unique()
        )

        axis.set_xticks(
            removal_levels
        )

    handles, labels = (
        axes[0].get_legend_handles_labels()
    )

    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    figure.suptitle(
        "Age-Embedding Monotonicity Across Removal Levels — "
        f"{WINDOW_LABELS[window_condition]}",
        fontsize=14,
    )

    figure.tight_layout(
        rect=(0, 0.10, 1, 0.94)
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )

    plt.close(figure)


def plot_method_gap_by_removal(
    results,
    window_condition,
    output_path,
    dpi=300,
):
    """
    Plot the difference between MLP and PCA monotonicity.

    Difference:

        MLP correlation - PCA correlation

    Positive values mean the MLP curve has stronger monotonicity.
    Negative values mean PCA has stronger monotonicity.
    """
    window_results = results.loc[
        results["window_condition"]
        == window_condition
    ].copy()

    if window_results.empty:
        return

    window_results["rho_gap"] = (
        window_results["mlp_rho"]
        - window_results["pca_rho"]
    )

    window_results["tau_gap"] = (
        window_results["mlp_tau"]
        - window_results["pca_tau"]
    )

    population_sizes = sorted(
        window_results["population_size"].unique(),
        reverse=True,
    )

    default_colors = plt.rcParams[
        "axes.prop_cycle"
    ].by_key()["color"]

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(15, 6),
        sharex=True,
        facecolor=PLOT_BACKGROUND,
    )

    metric_definitions = (
        (
            "rho_gap",
            "Δ Spearman ρ: MLP − PCA",
        ),
        (
            "tau_gap",
            "Δ Kendall τb: MLP − PCA",
        ),
    )

    for axis, (
        gap_column,
        metric_label,
    ) in zip(axes, metric_definitions):

        configure_plot_axis(axis)

        for population_index, population_size in enumerate(
            population_sizes
        ):
            population_results = window_results.loc[
                window_results["population_size"]
                == population_size
            ].sort_values("removal_percent")

            color = default_colors[
                population_index % len(default_colors)
            ]

            axis.plot(
                population_results["removal_percent"],
                population_results[gap_column],
                marker="o",
                linestyle="-",
                linewidth=2.1,
                markersize=6,
                color=color,
                label=f"Population {population_size}",
            )

        axis.axhline(
            0,
            linestyle="--",
            linewidth=1.2,
            color="black",
            alpha=0.65,
        )

        axis.set_title(
            metric_label
        )

        axis.set_xlabel(
            "Removed hasAge relations (%)"
        )

        axis.set_ylabel(
            "Difference in rank correlation"
        )

        removal_levels = sorted(
            window_results[
                "removal_percent"
            ].unique()
        )

        axis.set_xticks(
            removal_levels
        )

        axis.text(
            0.02,
            0.97,
            "Above zero: MLP is more monotonic\n"
            "Below zero: PCA is more monotonic",
            transform=axis.transAxes,
            verticalalignment="top",
            fontsize=8,
        )

    handles, labels = (
        axes[0].get_legend_handles_labels()
    )

    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=max(
            1,
            len(population_sizes),
        ),
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    figure.suptitle(
        "Difference Between MLP and PCA Monotonicity — "
        f"{WINDOW_LABELS[window_condition]}",
        fontsize=14,
    )

    figure.tight_layout(
        rect=(0, 0.10, 1, 0.94)
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )

    plt.close(figure)


def generate_monotonicity_plots(
    results,
    output_directory,
    dpi=300,
):
    """Generate all four plots."""
    output_directory = Path(
        output_directory
    )

    plot_directory = (
        output_directory
        / PLOT_DIRECTORY_NAME
    )

    plot_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_paths = {}

    for window_condition in (
        "with_windows",
        "without_windows",
    ):
        monotonicity_path = (
            plot_directory
            / f"{window_condition}_monotonicity.png"
        )

        gap_path = (
            plot_directory
            / f"{window_condition}_pca_mlp_gap.png"
        )

        plot_monotonicity_by_removal(
            results=results,
            window_condition=window_condition,
            output_path=monotonicity_path,
            dpi=dpi,
        )

        plot_method_gap_by_removal(
            results=results,
            window_condition=window_condition,
            output_path=gap_path,
            dpi=dpi,
        )

        plot_paths[
            f"{window_condition}_monotonicity"
        ] = monotonicity_path

        plot_paths[
            f"{window_condition}_gap"
        ] = gap_path

    return plot_paths


# ============================================================
# Report writing
# ============================================================

def write_outputs(
    results,
    output_directory=DEFAULT_OUTPUT_DIRECTORY,
):
    """Write Markdown, CSV, and plot references."""
    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    markdown_path = (
        output_directory
        / MARKDOWN_FILENAME
    )

    csv_path = (
        output_directory
        / CSV_FILENAME
    )

    create_csv_table(results).to_csv(
        csv_path,
        index=False,
    )

    with markdown_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        output_file.write(
            "# Age-Node Embedding Monotonicity\n\n"
        )

        output_file.write(
            "Spearman’s ρ and Kendall’s τb compare "
            "ground-truth age ordering with one-dimensional "
            "coordinates learned from age-node embeddings.\n\n"
        )

        output_file.write(
            "Neither PCA nor the MLP principal curve uses numeric "
            "age while fitting its one-dimensional coordinate. "
            "True ages are used only afterward to calculate rank "
            "correlations.\n\n"
        )

        output_file.write(
            "PCA measures ordering along the dominant unsupervised "
            "linear direction. The MLP principal curve measures "
            "ordering along an unsupervised nonlinear one-dimensional "
            "curve initialized from PCA and refined through nearest-"
            "curve projection.\n\n"
        )

        output_file.write(
            "One-dimensional coordinates have arbitrary direction. "
            "Their signs are therefore oriented toward increasing age "
            "before reporting the correlations.\n\n"
        )

        output_file.write(
            "In the gap plots, the difference is "
            "`MLP correlation − PCA correlation`. Positive values "
            "indicate stronger MLP monotonicity; negative values "
            "indicate stronger PCA monotonicity.\n\n"
        )

        for window_condition in (
            "with_windows",
            "without_windows",
        ):
            output_file.write(
                f"## {WINDOW_LABELS[window_condition]}\n\n"
            )

            table = create_markdown_table(
                results=results,
                window_condition=window_condition,
            )

            output_file.write(
                dataframe_to_markdown(table)
            )

            output_file.write("\n")

            monotonicity_image = (
                Path(PLOT_DIRECTORY_NAME)
                / (
                    f"{window_condition}_"
                    "monotonicity.png"
                )
            )

            gap_image = (
                Path(PLOT_DIRECTORY_NAME)
                / (
                    f"{window_condition}_"
                    "pca_mlp_gap.png"
                )
            )

            output_file.write(
                "### Monotonicity Across Removal Levels\n\n"
            )

            output_file.write(
                f"![{WINDOW_LABELS[window_condition]} "
                f"monotonicity]"
                f"({monotonicity_image.as_posix()})\n\n"
            )

            output_file.write(
                "### Difference Between PCA and MLP\n\n"
            )

            output_file.write(
                f"![{WINDOW_LABELS[window_condition]} "
                f"PCA and MLP difference]"
                f"({gap_image.as_posix()})\n\n"
            )

    print(
        f"Saved Markdown report to: {markdown_path}"
    )

    print(
        f"Saved CSV results to: {csv_path}"
    )

    print(
        "Saved plots to: "
        f"{output_directory / PLOT_DIRECTORY_NAME}"
    )


# ============================================================
# Complete analysis pipeline
# ============================================================

def analyze_age_monotonicity(
    population_directories,
    output_directory=DEFAULT_OUTPUT_DIRECTORY,
    hidden_layer_sizes=(16,),
    max_iter=5000,
    grid_size=4000,
    refinement_steps=8,
    mlp_alpha=0.1,
    tolerance=1e-4,
    seed=42,
    dpi=300,
):
    """Run the complete monotonicity analysis."""
    all_rows = []

    for population_directory in (
        population_directories
    ):
        all_rows.extend(
            evaluate_population_directory(
                population_directory=(
                    population_directory
                ),
                hidden_layer_sizes=(
                    hidden_layer_sizes
                ),
                max_iter=max_iter,
                grid_size=grid_size,
                refinement_steps=refinement_steps,
                mlp_alpha=mlp_alpha,
                tolerance=tolerance,
                seed=seed,
            )
        )

    results = pd.DataFrame(
        all_rows
    )

    if results.empty:
        raise ValueError(
            "No monotonicity results were generated."
        )

    results["_window_order"] = (
        results["window_condition"]
        .map(WINDOW_ORDER)
    )

    results = results.sort_values(
        [
            "_window_order",
            "population_size",
            "removal_percent",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)

    results = results.drop(
        columns="_window_order"
    )

    generate_monotonicity_plots(
        results=results,
        output_directory=output_directory,
        dpi=dpi,
    )

    write_outputs(
        results=results,
        output_directory=output_directory,
    )

    return results


# ============================================================
# Command-line interface
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Measure age-node embedding monotonicity using "
            "unsupervised one-dimensional PCA and an unsupervised "
            "nonlinear MLP principal curve."
        )
    )

    parser.add_argument(
        "--basepath",
        nargs="+",
        help=(
            "Population experiment folders, such as "
            "--basepath 100 200 500. When omitted, numeric "
            "population folders are discovered automatically."
        ),
    )

    parser.add_argument(
        "--root",
        default=".",
        help=(
            "Root directory used for automatic population-folder "
            "discovery. Default: current directory."
        ),
    )

    parser.add_argument(
        "--output-directory",
        default=str(
            DEFAULT_OUTPUT_DIRECTORY
        ),
        help=(
            "Output directory. "
            f"Default: {DEFAULT_OUTPUT_DIRECTORY}"
        ),
    )

    parser.add_argument(
        "--hidden-layers",
        nargs="+",
        type=int,
        default=[16],
        help=(
            "MLP principal-curve hidden-layer sizes. "
            "Default: 16."
        ),
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
        help=(
            "Maximum MLP optimizer iterations per refinement. "
            "Default: 5000."
        ),
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        default=4000,
        help=(
            "Number of sampled locations along the MLP curve. "
            "Default: 4000."
        ),
    )

    parser.add_argument(
        "--refinement-steps",
        type=int,
        default=8,
        help=(
            "Maximum number of MLP curve-refinement iterations. "
            "Default: 8."
        ),
    )

    parser.add_argument(
        "--mlp-alpha",
        type=float,
        default=0.1,
        help=(
            "MLP L2 regularization strength. "
            "Default: 0.1."
        ),
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help=(
            "Coordinate-change convergence tolerance. "
            "Default: 0.0001."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed. Default: 42."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help=(
            "Plot resolution. Default: 300."
        ),
    )

    return parser


def main():
    args = build_parser().parse_args()

    if args.basepath:
        population_directories = [
            Path(path)
            for path in args.basepath
        ]
    else:
        population_directories = (
            discover_population_directories(
                args.root
            )
        )

        print(
            "Automatically discovered population folders: "
            + ", ".join(
                str(path)
                for path in population_directories
            )
        )

    analyze_age_monotonicity(
        population_directories=(
            population_directories
        ),
        output_directory=(
            args.output_directory
        ),
        hidden_layer_sizes=tuple(
            args.hidden_layers
        ),
        max_iter=args.max_iter,
        grid_size=args.grid_size,
        refinement_steps=args.refinement_steps,
        mlp_alpha=args.mlp_alpha,
        tolerance=args.tolerance,
        seed=args.seed,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()