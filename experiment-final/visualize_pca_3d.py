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

from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from pykeen.triples import TriplesFactory
from sklearn.decomposition import PCA


# ============================================================
# Data configuration
# ============================================================

RELATION = "hasAge"
PERSON_PREFIX = "person"
AGE_PREFIX = "v"

WINDOW_NAMES = {
    "with_windows": "With windows",
    "without_windows": "Without windows",
}

WINDOW_ORDER = {
    "with_windows": 0,
    "without_windows": 1,
}


# ============================================================
# Appearance
# ============================================================

BACKGROUND_COLOR = "#fffdf7"
GRID_COLOR = "#d8d3c8"

PERSON_QUERY_LINE_WIDTH = 1.1
QUERY_AGE_LINE_WIDTH = 1.5

PERSON_NODE_SIZE = 15
AGE_NODE_SIZE = 62

TRIANGLE_LENGTH_FRACTION = 0.018
TRIANGLE_WIDTH_FRACTION = 0.012
TRIANGLE_LINE_WIDTH = 1.4

DEFAULT_ELEVATION = 24
DEFAULT_AZIMUTH = -58


# ============================================================
# Label helpers
# ============================================================

def suffix_number(label, prefix):
    """
    Extract the numeric suffix from labels such as:

        person42
        v18
        v-2
        v18.5
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


def sorted_labels(labels, prefix):
    """Sort labels by their numeric suffix."""

    def sort_key(label):
        number = suffix_number(label, prefix)

        return (
            number is None,
            number if number is not None else float("inf"),
            str(label),
        )

    return sorted(labels, key=sort_key)


# ============================================================
# MuRE model loading
# ============================================================

def get_mure_parameters(model):
    """Extract MuRE entity and relation parameters."""
    if len(model.entity_representations) < 1:
        raise ValueError(
            "The loaded model does not contain entity representations."
        )

    if len(model.relation_representations) < 2:
        raise ValueError(
            "The loaded model does not expose both MuRE relation "
            "representations required for q = h * R + r."
        )

    with torch.no_grad():
        entity_embeddings = (
            model.entity_representations[0](indices=None)
            .detach()
            .cpu()
            .numpy()
        )

        relation_embeddings = (
            model.relation_representations[0](indices=None)
            .detach()
            .cpu()
            .numpy()
        )

        relation_specific_embeddings = (
            model.relation_representations[1](indices=None)
            .detach()
            .cpu()
            .numpy()
        )

    return {
        "entity_embeddings": entity_embeddings,
        "relation_embeddings": relation_embeddings,
        "relation_specific_embeddings": relation_specific_embeddings,
    }


def load_run_vectors(run_path):
    """
    Load person, query-point, and age-node vectors from one trained run.
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

    triples_factory = TriplesFactory.from_path_binary(triples_path)

    model = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    model = model.to("cpu")
    model.eval()

    parameters = get_mure_parameters(model)

    if RELATION not in triples_factory.relation_to_id:
        raise ValueError(
            f"Relation {RELATION!r} was not found in {run_path}."
        )

    relation_id = int(
        triples_factory.relation_to_id[RELATION]
    )

    relation_translation = parameters[
        "relation_embeddings"
    ][relation_id]

    relation_scale = parameters[
        "relation_specific_embeddings"
    ][relation_id]

    person_labels = [
        label
        for label in triples_factory.entity_to_id
        if str(label).startswith(PERSON_PREFIX)
    ]

    person_labels = sorted_labels(
        person_labels,
        PERSON_PREFIX,
    )

    age_labels = [
        label
        for label in triples_factory.entity_to_id
        if str(label).startswith(AGE_PREFIX)
        and suffix_number(label, AGE_PREFIX) is not None
    ]

    age_labels = sorted_labels(
        age_labels,
        AGE_PREFIX,
    )

    if not person_labels:
        raise ValueError(
            f"No person entities were found in {run_path}."
        )

    if not age_labels:
        raise ValueError(
            f"No numeric age entities were found in {run_path}."
        )

    person_vectors = {}

    for person_label in person_labels:
        person_id = int(
            triples_factory.entity_to_id[person_label]
        )

        person_vectors[person_label] = parameters[
            "entity_embeddings"
        ][person_id]

    age_vectors = {}

    for age_label in age_labels:
        age_id = int(
            triples_factory.entity_to_id[age_label]
        )

        age_vectors[age_label] = parameters[
            "entity_embeddings"
        ][age_id]

    query_vectors = {}

    for person_label, person_vector in person_vectors.items():
        query_vectors[person_label] = (
            person_vector * relation_scale
            + relation_translation
        )

    return person_vectors, query_vectors, age_vectors


# ============================================================
# Ground-truth mappings
# ============================================================

def load_person_to_age(tsv_path):
    """
    Read person-to-age ground truth from an original graph TSV.
    """
    tsv_path = Path(tsv_path)
    person_to_age = {}

    with tsv_path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.rstrip("\r\n")

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 3:
                raise ValueError(
                    f"Malformed triple in {tsv_path} at line "
                    f"{line_number}: expected three columns."
                )

            head, relation, tail = parts

            if line_number == 1:
                possible_header = [head, relation, tail]

                if possible_header in (
                    ["?sub", "?pred", "?val"],
                    ["sub", "pred", "val"],
                ):
                    continue

            if relation != RELATION:
                continue

            if (
                head in person_to_age
                and person_to_age[head] != tail
            ):
                raise ValueError(
                    f"Conflicting {RELATION} values for "
                    f"{head!r} in {tsv_path}."
                )

            person_to_age[head] = tail

    if not person_to_age:
        raise ValueError(
            f"No {RELATION!r} triples were found in {tsv_path}."
        )

    return person_to_age


def find_original_tsv(
    population_directory,
    window_condition,
):
    """Find the original zero-removal graph for a window condition."""
    population_directory = Path(population_directory)
    kg_directory = population_directory / "kgs"

    exact_path = (
        kg_directory
        / f"{window_condition}_original.tsv"
    )

    if exact_path.exists():
        return exact_path

    matches = sorted(
        kg_directory.glob(
            f"{window_condition}*original*.tsv"
        )
    )

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            f"Could not find an original TSV for "
            f"{window_condition!r} under {kg_directory}."
        )

    raise ValueError(
        f"Multiple possible original TSV files were found "
        f"for {window_condition!r}: {matches}"
    )


# ============================================================
# Manifest and directory discovery
# ============================================================

def load_manifest(population_directory):
    """Load and validate kg_manifest.csv."""
    population_directory = Path(population_directory)
    manifest_path = population_directory / "kg_manifest.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest was not found: {manifest_path}"
        )

    manifest = pd.read_csv(manifest_path)

    if "label" in manifest.columns:
        run_column = "label"
    elif "run" in manifest.columns:
        run_column = "run"
    else:
        raise ValueError(
            f"{manifest_path} requires either a "
            "'label' or 'run' column."
        )

    required_columns = {
        run_column,
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

    manifest = manifest.rename(
        columns={run_column: "run"}
    ).copy()

    manifest["run"] = manifest["run"].astype(str)

    manifest["window_condition"] = (
        manifest["window_condition"].astype(str)
    )

    manifest["removal_percent"] = pd.to_numeric(
        manifest["removal_percent"],
        errors="raise",
    )

    manifest = manifest.loc[
        manifest["window_condition"].isin(WINDOW_NAMES)
    ].copy()

    if manifest.empty:
        raise ValueError(
            f"No with_windows or without_windows runs "
            f"were found in {manifest_path}."
        )

    manifest["_window_order"] = (
        manifest["window_condition"].map(WINDOW_ORDER)
    )

    return manifest.sort_values(
        [
            "_window_order",
            "removal_percent",
            "run",
        ]
    ).reset_index(drop=True)


def discover_population_directories(
    basepath,
    requested_populations=None,
):
    """
    Find numeric experiment folders such as 100/, 200/, and 500/.
    """
    basepath = Path(basepath)

    if requested_populations:
        population_directories = [
            basepath / str(population)
            for population in requested_populations
        ]
    else:
        population_directories = [
            path
            for path in basepath.iterdir()
            if path.is_dir()
            and path.name.isdigit()
            and (path / "kg_manifest.csv").exists()
            and (path / "runs").is_dir()
        ]

        population_directories = sorted(
            population_directories,
            key=lambda path: int(path.name),
        )

    if not population_directories:
        raise ValueError(
            f"No population directories were found under {basepath}."
        )

    for population_directory in population_directories:
        if not population_directory.is_dir():
            raise FileNotFoundError(
                f"Population directory was not found: "
                f"{population_directory}"
            )

    return population_directories


# ============================================================
# Joint 3D PCA
# ============================================================

def perform_joint_pca_3d(
    person_vectors,
    query_vectors,
    age_vectors,
):
    """
    Fit one three-component PCA model jointly across person,
    query-point, and age-node vectors.
    """
    person_labels = list(person_vectors)
    age_labels = list(age_vectors)

    person_matrix = np.vstack(
        [
            person_vectors[label]
            for label in person_labels
        ]
    )

    query_matrix = np.vstack(
        [
            query_vectors[label]
            for label in person_labels
        ]
    )

    age_matrix = np.vstack(
        [
            age_vectors[label]
            for label in age_labels
        ]
    )

    combined_matrix = np.vstack(
        [
            person_matrix,
            query_matrix,
            age_matrix,
        ]
    )

    if combined_matrix.shape[1] < 3:
        raise ValueError(
            "The embeddings must have at least three dimensions "
            "for 3D PCA."
        )

    pca = PCA(n_components=3)
    reduced = pca.fit_transform(combined_matrix)

    number_of_people = len(person_labels)
    number_of_ages = len(age_labels)

    person_coordinates = reduced[
        :number_of_people
    ]

    query_coordinates = reduced[
        number_of_people:
        number_of_people * 2
    ]

    age_coordinates = reduced[
        number_of_people * 2:
        number_of_people * 2 + number_of_ages
    ]

    return {
        "person_labels": person_labels,
        "age_labels": age_labels,
        "person_coordinates": person_coordinates,
        "query_coordinates": query_coordinates,
        "age_coordinates": age_coordinates,
        "explained_variance": pca.explained_variance_ratio_,
    }


# ============================================================
# 3D geometry helpers
# ============================================================

def coordinate_span(*coordinate_arrays):
    """Return the overall plotting span across three dimensions."""
    all_coordinates = np.vstack(coordinate_arrays)

    ranges = np.ptp(
        all_coordinates,
        axis=0,
    )

    span = float(np.max(ranges))

    if span <= 0:
        return 1.0

    return span


def add_oriented_triangle(
    axis,
    person_position,
    query_position,
    edge_color,
    length,
    width,
):
    """
    Draw an outlined 3D triangle at the query point.

    The triangle tip is located at the query point and points along
    the person-to-query direction.
    """
    person_position = np.asarray(
        person_position,
        dtype=float,
    )

    query_position = np.asarray(
        query_position,
        dtype=float,
    )

    direction = query_position - person_position
    direction_norm = np.linalg.norm(direction)

    if direction_norm <= 1e-12:
        direction_unit = np.array(
            [1.0, 0.0, 0.0]
        )
    else:
        direction_unit = direction / direction_norm

    # Choose a reference vector that is not parallel to the
    # person-to-query direction.
    reference = np.array(
        [0.0, 0.0, 1.0]
    )

    if abs(np.dot(direction_unit, reference)) > 0.90:
        reference = np.array(
            [0.0, 1.0, 0.0]
        )

    side_vector = np.cross(
        direction_unit,
        reference,
    )

    side_norm = np.linalg.norm(side_vector)

    if side_norm <= 1e-12:
        side_vector = np.array(
            [0.0, 1.0, 0.0]
        )
    else:
        side_vector = side_vector / side_norm

    tip = query_position

    base_center = (
        query_position
        - direction_unit * length
    )

    base_left = (
        base_center
        + side_vector * width
    )

    base_right = (
        base_center
        - side_vector * width
    )

    triangle = Poly3DCollection(
        [[tip, base_left, base_right]],
        facecolors=BACKGROUND_COLOR,
        edgecolors=[edge_color],
        linewidths=TRIANGLE_LINE_WIDTH,
        alpha=0.98,
    )

    axis.add_collection3d(triangle)


def set_axes_equal_3d(axis):
    """
    Give all three PCA axes the same physical scale.
    """
    x_limits = axis.get_xlim3d()
    y_limits = axis.get_ylim3d()
    z_limits = axis.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    plot_radius = 0.5 * max(
        x_range,
        y_range,
        z_range,
    )

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    axis.set_xlim3d(
        x_middle - plot_radius,
        x_middle + plot_radius,
    )

    axis.set_ylim3d(
        y_middle - plot_radius,
        y_middle + plot_radius,
    )

    axis.set_zlim3d(
        z_middle - plot_radius,
        z_middle + plot_radius,
    )


# ============================================================
# Plotting
# ============================================================

def plot_run_3d(
    population_size,
    window_condition,
    removal_percent,
    person_vectors,
    query_vectors,
    age_vectors,
    truth_person_to_age,
    output_path,
    dpi=300,
    elevation=DEFAULT_ELEVATION,
    azimuth=DEFAULT_AZIMUTH,
):
    """Generate one joint three-dimensional PCA visualization."""
    reduced = perform_joint_pca_3d(
        person_vectors=person_vectors,
        query_vectors=query_vectors,
        age_vectors=age_vectors,
    )

    person_labels = reduced["person_labels"]
    age_labels = reduced["age_labels"]

    person_coordinates = reduced[
        "person_coordinates"
    ]

    query_coordinates = reduced[
        "query_coordinates"
    ]

    age_coordinates = reduced[
        "age_coordinates"
    ]

    explained_variance = reduced[
        "explained_variance"
    ]

    person_index = {
        label: index
        for index, label in enumerate(person_labels)
    }

    age_index = {
        label: index
        for index, label in enumerate(age_labels)
    }

    age_values = np.asarray(
        [
            suffix_number(label, AGE_PREFIX)
            for label in age_labels
        ],
        dtype=float,
    )

    age_min = float(age_values.min())
    age_max = float(age_values.max())

    if np.isclose(age_min, age_max):
        age_min -= 0.5
        age_max += 0.5

    age_normalizer = Normalize(
        vmin=age_min,
        vmax=age_max,
    )

    # Age nodes and query-point outlines.
    age_colormap = LinearSegmentedColormap.from_list(
        "age_gradient",
        [
            "dodgerblue",
            "red",
        ],
    )

    # Person nodes: younger people black, older people light gray.
    person_colormap = LinearSegmentedColormap.from_list(
        "person_age_gradient",
        [
            "black",
            "lightgray",
        ],
    )

    figure = plt.figure(
        figsize=(15, 12),
        facecolor=BACKGROUND_COLOR,
    )

    axis = figure.add_subplot(
        111,
        projection="3d",
    )

    axis.set_facecolor(BACKGROUND_COLOR)

    pane_color = to_rgba(
        BACKGROUND_COLOR,
        alpha=1.0,
    )

    axis.xaxis.set_pane_color(pane_color)
    axis.yaxis.set_pane_color(pane_color)
    axis.zaxis.set_pane_color(pane_color)

    grid_rgba = to_rgba(
        GRID_COLOR,
        alpha=0.45,
    )

    for dimension_axis in (
        axis.xaxis,
        axis.yaxis,
        axis.zaxis,
    ):
        dimension_axis._axinfo["grid"]["color"] = grid_rgba
        dimension_axis._axinfo["grid"]["linewidth"] = 0.45

    # --------------------------------------------------------
    # Determine colors from each person's true age.
    # --------------------------------------------------------
    person_colors = []
    query_outline_colors = []

    for person_label in person_labels:
        true_age_label = truth_person_to_age.get(
            person_label
        )

        true_age_value = suffix_number(
            true_age_label,
            AGE_PREFIX,
        )

        if true_age_value is None:
            person_colors.append("gray")
            query_outline_colors.append("gray")
        else:
            normalized_age = age_normalizer(
                true_age_value
            )

            person_colors.append(
                person_colormap(normalized_age)
            )

            query_outline_colors.append(
                age_colormap(normalized_age)
            )

    # --------------------------------------------------------
    # Solid person-to-query lines.
    # --------------------------------------------------------
    person_query_segments = [
        [
            person_coordinates[index],
            query_coordinates[index],
        ]
        for index in range(len(person_labels))
    ]

    person_query_collection = Line3DCollection(
        person_query_segments,
        colors="#383838",
        linewidths=PERSON_QUERY_LINE_WIDTH,
        alpha=0.34,
        linestyles="solid",
    )

    axis.add_collection3d(
        person_query_collection
    )

    # --------------------------------------------------------
    # Dotted query-to-true-age lines.
    # --------------------------------------------------------
    truth_segments = []
    truth_colors = []

    missing_truth_mappings = 0
    missing_true_age_nodes = 0

    for person_label in person_labels:
        true_age_label = truth_person_to_age.get(
            person_label
        )

        if true_age_label is None:
            missing_truth_mappings += 1
            continue

        if true_age_label not in age_index:
            missing_true_age_nodes += 1
            continue

        person_position = person_index[
            person_label
        ]

        age_position = age_index[
            true_age_label
        ]

        truth_segments.append(
            [
                query_coordinates[
                    person_position
                ],
                age_coordinates[
                    age_position
                ],
            ]
        )

        truth_colors.append(
            age_colormap(
                age_normalizer(
                    age_values[
                        age_position
                    ]
                )
            )
        )

    if truth_segments:
        truth_collection = Line3DCollection(
            truth_segments,
            colors=truth_colors,
            linewidths=QUERY_AGE_LINE_WIDTH,
            alpha=0.43,
            linestyles="dotted",
        )

        axis.add_collection3d(
            truth_collection
        )

    # --------------------------------------------------------
    # Person nodes.
    # --------------------------------------------------------
    axis.scatter(
        person_coordinates[:, 0],
        person_coordinates[:, 1],
        person_coordinates[:, 2],
        s=PERSON_NODE_SIZE,
        marker="o",
        c=person_colors,
        edgecolors="#202020",
        linewidths=0.35,
        alpha=0.94,
        depthshade=False,
    )

    # --------------------------------------------------------
    # Oriented query-point triangles.
    # --------------------------------------------------------
    overall_span = coordinate_span(
        person_coordinates,
        query_coordinates,
        age_coordinates,
    )

    triangle_length = (
        overall_span
        * TRIANGLE_LENGTH_FRACTION
    )

    triangle_width = (
        overall_span
        * TRIANGLE_WIDTH_FRACTION
    )

    for index, person_label in enumerate(person_labels):
        add_oriented_triangle(
            axis=axis,
            person_position=person_coordinates[index],
            query_position=query_coordinates[index],
            edge_color=query_outline_colors[index],
            length=triangle_length,
            width=triangle_width,
        )

    # --------------------------------------------------------
    # Age nodes.
    # --------------------------------------------------------
    age_scatter = axis.scatter(
        age_coordinates[:, 0],
        age_coordinates[:, 1],
        age_coordinates[:, 2],
        s=AGE_NODE_SIZE,
        marker="o",
        c=age_values,
        cmap=age_colormap,
        norm=age_normalizer,
        edgecolors="#202020",
        linewidths=0.55,
        alpha=0.98,
        depthshade=False,
    )

    colorbar = figure.colorbar(
        age_scatter,
        ax=axis,
        pad=0.08,
        shrink=0.72,
    )

    colorbar.set_label("Age")
    colorbar.ax.set_facecolor(BACKGROUND_COLOR)

    removal_text = (
        f"{float(removal_percent):g}%"
    )

    axis.set_title(
        f"Population {population_size} | "
        f"{WINDOW_NAMES[window_condition]} | "
        f"{removal_text} hasAge removed",
        pad=20,
    )

    axis.set_xlabel(
        f"PC1 "
        f"({explained_variance[0] * 100:.1f}% variance)",
        labelpad=10,
    )

    axis.set_ylabel(
        f"PC2 "
        f"({explained_variance[1] * 100:.1f}% variance)",
        labelpad=10,
    )

    axis.set_zlabel(
        f"PC3 "
        f"({explained_variance[2] * 100:.1f}% variance)",
        labelpad=10,
    )

    axis.view_init(
        elev=elevation,
        azim=azimuth,
    )

    set_axes_equal_3d(axis)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="gray",
            markeredgecolor="#202020",
            markersize=5,
            label="Person node",
        ),
        Line2D(
            [0],
            [0],
            marker=">",
            linestyle="none",
            markerfacecolor=BACKGROUND_COLOR,
            markeredgecolor="dodgerblue",
            markeredgewidth=1.4,
            markersize=6,
            label="Oriented query point",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="red",
            markeredgecolor="#202020",
            markersize=6,
            label="Age node",
        ),
        Line2D(
            [0, 1],
            [0, 0],
            color="#383838",
            linewidth=PERSON_QUERY_LINE_WIDTH,
            linestyle="solid",
            label="Person → query",
        ),
        Line2D(
            [0, 1],
            [0, 0],
            color="dodgerblue",
            linewidth=QUERY_AGE_LINE_WIDTH,
            linestyle="dotted",
            label="Query → true age",
        ),
    ]

    legend = axis.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=8,
        frameon=True,
        facecolor=BACKGROUND_COLOR,
        edgecolor=GRID_COLOR,
        framealpha=0.96,
    )

    legend.get_frame().set_linewidth(0.7)

    notes = []

    if missing_truth_mappings:
        notes.append(
            f"{missing_truth_mappings} people had no "
            "ground-truth age mapping"
        )

    if missing_true_age_nodes:
        notes.append(
            f"{missing_true_age_nodes} true-age nodes "
            "were absent from this run's vocabulary"
        )

    if notes:
        figure.text(
            0.02,
            0.02,
            "; ".join(notes),
            fontsize=7,
            color="#404040",
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


# ============================================================
# Batch generation
# ============================================================

def generate_visualizations(
    basepath=".",
    populations=None,
    output_directory="visualizations",
    dpi=300,
    elevation=DEFAULT_ELEVATION,
    azimuth=DEFAULT_AZIMUTH,
):
    """
    Generate one 3D PCA PNG for every population and manifest run.
    """
    basepath = Path(basepath)
    output_directory = Path(output_directory)

    population_directories = discover_population_directories(
        basepath=basepath,
        requested_populations=populations,
    )

    generated_count = 0
    failures = []

    for population_directory in population_directories:
        population_size = int(
            population_directory.name
        )

        manifest = load_manifest(
            population_directory
        )

        truth_cache = {}

        for row in manifest.itertuples(index=False):
            run_name = str(row.run)
            window_condition = str(
                row.window_condition
            )
            removal_percent = float(
                row.removal_percent
            )

            run_path = (
                population_directory
                / "runs"
                / run_name
            )

            try:
                if not run_path.is_dir():
                    raise FileNotFoundError(
                        f"Run directory was not found: "
                        f"{run_path}"
                    )

                if window_condition not in truth_cache:
                    truth_tsv_path = find_original_tsv(
                        population_directory,
                        window_condition,
                    )

                    truth_cache[
                        window_condition
                    ] = load_person_to_age(
                        truth_tsv_path
                    )

                (
                    person_vectors,
                    query_vectors,
                    age_vectors,
                ) = load_run_vectors(run_path)

                output_filename = (
                    f"population_{population_size}_"
                    f"{window_condition}_"
                    f"removed_{removal_percent:g}pct_"
                    f"pca_3d.png"
                )

                output_path = (
                    output_directory
                    / output_filename
                )

                plot_run_3d(
                    population_size=population_size,
                    window_condition=window_condition,
                    removal_percent=removal_percent,
                    person_vectors=person_vectors,
                    query_vectors=query_vectors,
                    age_vectors=age_vectors,
                    truth_person_to_age=truth_cache[
                        window_condition
                    ],
                    output_path=output_path,
                    dpi=dpi,
                    elevation=elevation,
                    azimuth=azimuth,
                )

                generated_count += 1
                print(f"Saved: {output_path}")

            except Exception as error:
                failures.append(
                    (
                        run_path,
                        str(error),
                    )
                )

                print(f"FAILED: {run_path}")
                print(f"  {error}")

    print()
    print(
        f"Generated {generated_count} 3D visualization(s) in: "
        f"{output_directory}"
    )

    if failures:
        print(
            f"{len(failures)} run(s) failed:"
        )

        for run_path, message in failures:
            print(
                f"  - {run_path}: {message}"
            )


# ============================================================
# Command-line interface
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate joint 3D PCA visualizations of MuRE "
            "person nodes, hasAge query points, and age nodes."
        )
    )

    parser.add_argument(
        "--basepath",
        default=".",
        help=(
            "Project folder containing population folders such "
            "as 100/, 200/, and 500/. Default: current folder."
        ),
    )

    parser.add_argument(
        "--populations",
        nargs="*",
        help=(
            "Optional population folders. Example: "
            "--populations 100 200 500. Numeric folders are "
            "discovered automatically when omitted."
        ),
    )

    parser.add_argument(
        "--output-directory",
        default="visualizations",
        help=(
            "Output folder for PNG files. "
            "Default: visualizations/"
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution. Default: 300.",
    )

    parser.add_argument(
        "--elevation",
        type=float,
        default=DEFAULT_ELEVATION,
        help=(
            "Vertical viewing angle in degrees. "
            f"Default: {DEFAULT_ELEVATION}."
        ),
    )

    parser.add_argument(
        "--azimuth",
        type=float,
        default=DEFAULT_AZIMUTH,
        help=(
            "Horizontal viewing angle in degrees. "
            f"Default: {DEFAULT_AZIMUTH}."
        ),
    )

    return parser


def main():
    args = build_parser().parse_args()

    generate_visualizations(
        basepath=args.basepath,
        populations=args.populations,
        output_directory=args.output_directory,
        dpi=args.dpi,
        elevation=args.elevation,
        azimuth=args.azimuth,
    )


if __name__ == "__main__":
    main()