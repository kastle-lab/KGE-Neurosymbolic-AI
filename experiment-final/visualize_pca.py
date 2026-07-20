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

from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from pykeen.triples import TriplesFactory
from sklearn.decomposition import PCA


RELATION = "hasAge"
PERSON_PREFIX = "person"
AGE_PREFIX = "v"

BACKGROUND_COLOR = "#fffdf7"

PERSON_QUERY_LINE_WIDTH = 0.9
QUERY_AGE_LINE_WIDTH = 1.25

PERSON_NODE_SIZE = 13
QUERY_POINT_SIZE = 32
AGE_NODE_SIZE = 60

WINDOW_NAMES = {
    "with_windows": "With windows",
    "without_windows": "Without windows",
}

WINDOW_ORDER = {
    "with_windows": 0,
    "without_windows": 1,
}


def suffix_number(label, prefix):
    """
    Extract the numeric suffix from labels such as:

        person42
        v18
        v-2
        v18.5
    """
    match = re.fullmatch(
        rf"{re.escape(prefix)}(-?\d+(?:\.\d+)?)",
        str(label).strip(),
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return float(match.group(1))


def sorted_labels(labels, prefix):
    """Sort person and age labels by their numeric suffix."""

    def sort_key(label):
        value = suffix_number(label, prefix)

        return (
            value is None,
            value if value is not None else float("inf"),
            str(label),
        )

    return sorted(labels, key=sort_key)


def get_mure_parameters(model):
    """
    Extract MuRE entity and relation parameters.

    MuRE query points are calculated using:

        query = head * relation_scale + relation_translation
    """
    if len(model.entity_representations) < 1:
        raise ValueError(
            "The loaded model does not contain entity representations."
        )

    if len(model.relation_representations) < 2:
        raise ValueError(
            "The loaded model does not expose the two MuRE relation "
            "representations needed for q = h * R + r."
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
    Load one trained MuRE run.

    Returns:

        person_vectors
        query_vectors
        age_vectors
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


def load_person_to_age(tsv_path):
    """
    Read the original person-to-age mapping from a TSV graph.

    The original graph is used as ground truth even when the evaluated
    run has removed some hasAge relations.
    """
    tsv_path = Path(tsv_path)
    person_to_age = {}

    with tsv_path.open(
        "r",
        encoding="utf-8",
    ) as infile:

        for line_number, line in enumerate(
            infile,
            start=1,
        ):
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
            f"No {RELATION!r} triples were found in "
            f"{tsv_path}."
        )

    return person_to_age


def load_manifest(population_directory):
    """Load the run list from kg_manifest.csv."""
    population_directory = Path(population_directory)
    manifest_path = (
        population_directory
        / "kg_manifest.csv"
    )

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
        required_columns
        - set(manifest.columns)
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
        manifest["window_condition"]
        .astype(str)
    )

    manifest["removal_percent"] = pd.to_numeric(
        manifest["removal_percent"],
        errors="raise",
    )

    manifest = manifest.loc[
        manifest["window_condition"].isin(
            WINDOW_NAMES
        )
    ].copy()

    if manifest.empty:
        raise ValueError(
            f"No with_windows or without_windows runs "
            f"were found in {manifest_path}."
        )

    manifest["_window_order"] = (
        manifest["window_condition"]
        .map(WINDOW_ORDER)
    )

    manifest = manifest.sort_values(
        [
            "_window_order",
            "removal_percent",
            "run",
        ]
    ).reset_index(drop=True)

    return manifest


def find_original_tsv(
    population_directory,
    window_condition,
):
    """
    Find the original 0%-removal TSV for a window condition.
    """
    population_directory = Path(
        population_directory
    )

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
            f"Could not find the original TSV for "
            f"{window_condition!r} under "
            f"{kg_directory}."
        )

    raise ValueError(
        f"Multiple possible original TSV files were "
        f"found for {window_condition!r}: {matches}"
    )


def perform_joint_pca(
    person_vectors,
    query_vectors,
    age_vectors,
):
    """
    Fit one PCA model jointly across:

        person embeddings
        query-point embeddings
        age-node embeddings

    This keeps all plotted objects in the same PCA coordinate system.
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

    if combined_matrix.shape[1] < 2:
        raise ValueError(
            "The embeddings must have at least two "
            "dimensions for 2D PCA."
        )

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(
        combined_matrix
    )

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
        "explained_variance": (
            pca.explained_variance_ratio_
        ),
    }


def plot_run(
    population_size,
    window_condition,
    removal_percent,
    person_vectors,
    query_vectors,
    age_vectors,
    truth_person_to_age,
    output_path,
    dpi=300,
):
    """
    Generate one joint 2D PCA visualization.

    Person nodes:
        Small circles colored from black to light gray according to age.

    Query points:
        Hollow triangles outlined with the corresponding age color.
        Each triangle points from its person node toward its query point.

    Age nodes:
        Colored from DodgerBlue for lower ages to red for higher ages.

    Lines:
        Solid person-to-query lines.
        Dotted query-to-ground-truth-age lines.
    """
    reduced = perform_joint_pca(
        person_vectors=person_vectors,
        query_vectors=query_vectors,
        age_vectors=age_vectors,
    )

    person_labels = reduced["person_labels"]
    age_labels = reduced["age_labels"]

    person_coordinates = reduced["person_coordinates"]
    query_coordinates = reduced["query_coordinates"]
    age_coordinates = reduced["age_coordinates"]

    explained_variance = reduced["explained_variance"]

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

    # Prevent Normalize from receiving an identical minimum and maximum.
    if np.isclose(age_min, age_max):
        age_min -= 0.5
        age_max += 0.5

    age_normalizer = Normalize(
        vmin=age_min,
        vmax=age_max,
    )

    # Age-node and query-outline gradient.
    age_colormap = LinearSegmentedColormap.from_list(
        "age_gradient",
        [
            "dodgerblue",
            "red",
        ],
    )

    # Person-node gradient.
    # Younger people are black; older people approach light gray.
    person_colormap = LinearSegmentedColormap.from_list(
        "person_age_gradient",
        [
            "black",
            "lightgray",
        ],
    )

    figure, axis = plt.subplots(
        figsize=(14, 10),
        facecolor=BACKGROUND_COLOR,
    )

    axis.set_facecolor(BACKGROUND_COLOR)

    # ------------------------------------------------------------
    # Determine each person's true age and corresponding colors.
    # ------------------------------------------------------------
    person_age_values = []
    person_colors = []
    query_outline_colors = []

    for person_label in person_labels:
        true_age_label = truth_person_to_age.get(person_label)
        true_age_value = suffix_number(true_age_label, AGE_PREFIX)

        person_age_values.append(true_age_value)

        if true_age_value is None:
            person_colors.append("gray")
            query_outline_colors.append("gray")
        else:
            normalized_age = age_normalizer(true_age_value)

            person_colors.append(
                person_colormap(normalized_age)
            )

            query_outline_colors.append(
                age_colormap(normalized_age)
            )

    # ------------------------------------------------------------
    # Solid person -> query lines.
    # ------------------------------------------------------------
    person_query_segments = np.stack(
        [
            person_coordinates,
            query_coordinates,
        ],
        axis=1,
    )

    person_query_collection = LineCollection(
        person_query_segments,
        colors="#383838",
        linewidths=PERSON_QUERY_LINE_WIDTH,
        alpha=0.30,
        linestyles="solid",
        capstyle="round",
        joinstyle="round",
        zorder=1,
    )

    axis.add_collection(person_query_collection)

    # ------------------------------------------------------------
    # Dotted query -> true-age lines.
    # ------------------------------------------------------------
    truth_segments = []
    truth_colors = []

    missing_true_age_nodes = 0
    missing_truth_mappings = 0

    for person_label in person_labels:
        true_age_label = truth_person_to_age.get(person_label)

        if true_age_label is None:
            missing_truth_mappings += 1
            continue

        if true_age_label not in age_index:
            missing_true_age_nodes += 1
            continue

        person_position = person_index[person_label]
        age_position = age_index[true_age_label]

        truth_segments.append(
            [
                query_coordinates[person_position],
                age_coordinates[age_position],
            ]
        )

        truth_colors.append(
            age_colormap(
                age_normalizer(
                    age_values[age_position]
                )
            )
        )

    if truth_segments:
        truth_collection = LineCollection(
            truth_segments,
            colors=truth_colors,
            linewidths=QUERY_AGE_LINE_WIDTH,
            alpha=0.38,
            linestyles="dotted",
            capstyle="round",
            joinstyle="round",
            zorder=1,
        )

        axis.add_collection(truth_collection)

    # ------------------------------------------------------------
    # Person nodes.
    # ------------------------------------------------------------
    axis.scatter(
        person_coordinates[:, 0],
        person_coordinates[:, 1],
        s=PERSON_NODE_SIZE,
        marker="o",
        c=person_colors,
        edgecolors="#202020",
        linewidths=0.35,
        alpha=0.92,
        zorder=3,
    )

    # ------------------------------------------------------------
    # Query points.
    #
    # A separate marker is drawn for each person because every
    # triangle needs a different rotation and outline color.
    # ------------------------------------------------------------
    for index, person_label in enumerate(person_labels):
        person_xy = person_coordinates[index]
        query_xy = query_coordinates[index]

        direction = query_xy - person_xy

        if np.allclose(direction, 0):
            angle_degrees = 0.0
        else:
            angle_degrees = np.degrees(
                np.arctan2(
                    direction[1],
                    direction[0],
                )
            )

        # The ">" marker initially points to the right. Rotate it into
        # the person-to-query direction.
        triangle_marker = MarkerStyle(">").transformed(
            Affine2D().rotate_deg(angle_degrees)
        )

        axis.scatter(
            [query_xy[0]],
            [query_xy[1]],
            s=QUERY_POINT_SIZE,
            marker=triangle_marker,
            facecolors=BACKGROUND_COLOR,
            edgecolors=[query_outline_colors[index]],
            linewidths=1.25,
            alpha=0.98,
            zorder=5,
        )

    # ------------------------------------------------------------
    # Age nodes.
    # ------------------------------------------------------------
    age_scatter = axis.scatter(
        age_coordinates[:, 0],
        age_coordinates[:, 1],
        s=AGE_NODE_SIZE,
        marker="o",
        c=age_values,
        cmap=age_colormap,
        norm=age_normalizer,
        edgecolors="#202020",
        linewidths=0.55,
        alpha=0.97,
        zorder=6,
    )

    colorbar = figure.colorbar(
        age_scatter,
        ax=axis,
        pad=0.015,
    )

    colorbar.set_label("Age")

    colorbar.ax.set_facecolor(BACKGROUND_COLOR)

    removal_text = f"{float(removal_percent):g}%"

    axis.set_title(
        f"Population {population_size} | "
        f"{WINDOW_NAMES[window_condition]} | "
        f"{removal_text} hasAge removed"
    )

    axis.set_xlabel(
        f"PC1 "
        f"({explained_variance[0] * 100:.1f}% variance)"
    )

    axis.set_ylabel(
        f"PC2 "
        f"({explained_variance[1] * 100:.1f}% variance)"
    )

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
            markeredgewidth=1.25,
            markersize=6,
            label="Query point",
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
        loc="best",
        fontsize=8,
        frameon=True,
        facecolor=BACKGROUND_COLOR,
        edgecolor="#d8d3c8",
        framealpha=0.95,
    )

    legend.get_frame().set_linewidth(0.7)

    axis.grid(
        True,
        linewidth=0.35,
        color="#d8d3c8",
        alpha=0.35,
    )

    axis.margins(0.05)

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
        axis.text(
            0.01,
            0.01,
            "; ".join(notes),
            transform=axis.transAxes,
            fontsize=7,
            verticalalignment="bottom",
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

def discover_population_directories(
    basepath,
    requested_populations=None,
):
    """
    Find population folders such as:

        100/
        200/
        500/
    """
    basepath = Path(basepath)

    if requested_populations:
        population_directories = [
            basepath / str(population)
            for population
            in requested_populations
        ]
    else:
        population_directories = [
            path
            for path in basepath.iterdir()
            if path.is_dir()
            and path.name.isdigit()
            and (
                path
                / "kg_manifest.csv"
            ).exists()
            and (
                path
                / "runs"
            ).is_dir()
        ]

        population_directories = sorted(
            population_directories,
            key=lambda path: int(path.name),
        )

    if not population_directories:
        raise ValueError(
            f"No population directories were found "
            f"under {basepath}."
        )

    for population_directory in (
        population_directories
    ):
        if not population_directory.is_dir():
            raise FileNotFoundError(
                f"Population directory was not found: "
                f"{population_directory}"
            )

    return population_directories


def generate_visualizations(
    basepath=".",
    populations=None,
    output_directory="visualizations",
    dpi=300,
):
    """
    Generate one PNG for every population and every run listed in
    that population's kg_manifest.csv.
    """
    basepath = Path(basepath)

    output_directory = Path(
        output_directory
    )

    population_directories = (
        discover_population_directories(
            basepath=basepath,
            requested_populations=populations,
        )
    )

    generated_count = 0
    failures = []

    for population_directory in (
        population_directories
    ):
        population_size = int(
            population_directory.name
        )

        manifest = load_manifest(
            population_directory
        )

        truth_cache = {}

        for row in manifest.itertuples(
            index=False
        ):
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
                    truth_tsv_path = (
                        find_original_tsv(
                            population_directory,
                            window_condition,
                        )
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
                ) = load_run_vectors(
                    run_path
                )

                output_filename = (
                    f"population_{population_size}_"
                    f"{window_condition}_"
                    f"removed_"
                    f"{removal_percent:g}pct_"
                    f"pca.png"
                )

                output_path = (
                    output_directory
                    / output_filename
                )

                plot_run(
                    population_size=population_size,
                    window_condition=window_condition,
                    removal_percent=removal_percent,
                    person_vectors=person_vectors,
                    query_vectors=query_vectors,
                    age_vectors=age_vectors,
                    truth_person_to_age=(
                        truth_cache[
                            window_condition
                        ]
                    ),
                    output_path=output_path,
                    dpi=dpi,
                )

                generated_count += 1

                print(
                    f"Saved: {output_path}"
                )

            except Exception as error:
                failures.append(
                    (
                        run_path,
                        str(error),
                    )
                )

                print(
                    f"FAILED: {run_path}"
                )

                print(
                    f"  {error}"
                )

    print()

    print(
        f"Generated {generated_count} "
        f"visualization(s) in: "
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


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate joint 2D PCA visualizations "
            "of MuRE person nodes, hasAge query "
            "points, and age nodes."
        )
    )

    parser.add_argument(
        "--basepath",
        default=".",
        help=(
            "Project folder containing population "
            "folders such as 100/, 200/, and 500/. "
            "Default: current folder."
        ),
    )

    parser.add_argument(
        "--populations",
        nargs="*",
        help=(
            "Optional population folders. Example: "
            "--populations 100 200 500. If omitted, "
            "numeric population folders are discovered "
            "automatically."
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

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    generate_visualizations(
        basepath=args.basepath,
        populations=args.populations,
        output_directory=(
            args.output_directory
        ),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()