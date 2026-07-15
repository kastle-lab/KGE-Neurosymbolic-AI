# analyze_with_learning.py

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import pandas as pd

from load_vectors import load_age_vectors, load_person_vectors
from learn_function import learn_all_positions


METADATA_COLUMNS = (
    "window_condition",
    "removal_percent",
    "removed_relation",
    "nested_removals",
)


def parse_age_node(value) -> float | None:
    """Convert an age node such as v42 into 42.0."""
    if value is None or pd.isna(value):
        return None

    label = str(value).strip()
    match = re.fullmatch(r"[vV](-?\d+(?:\.\d+)?)", label)

    if match is not None:
        return float(match.group(1))

    try:
        return float(label)
    except ValueError:
        return None


def load_manifest(basepath: str | Path) -> pd.DataFrame:
    """Load and validate kg_manifest.csv."""
    manifest_path = Path(basepath) / "kg_manifest.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing manifest: {manifest_path}. Start from the KG step first."
        )

    manifest = pd.read_csv(manifest_path)
    required = {"label", "window_condition", "removal_percent"}
    missing = required - set(manifest.columns)

    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    if manifest["label"].duplicated().any():
        raise ValueError("Manifest contains duplicate run labels.")

    manifest["label"] = manifest["label"].astype(str).str.strip()
    manifest["window_condition"] = (
        manifest["window_condition"].astype(str).str.strip()
    )
    manifest["removal_percent"] = pd.to_numeric(
        manifest["removal_percent"], errors="raise"
    )

    return manifest


def manifest_row_for_run(manifest: pd.DataFrame, run_label: str) -> pd.Series:
    matches = manifest.loc[manifest["label"] == str(run_label)]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one manifest row for {run_label!r}; "
            f"found {len(matches)}."
        )

    return matches.iloc[0]


def resolve_run_labels(
    manifest: pd.DataFrame,
    run_labels: Sequence[str] | None,
) -> list[str]:
    """Resolve which embedding runs should be analyzed."""
    available = manifest["label"].astype(str).tolist()

    if run_labels is None:
        return available

    requested = [str(label) for label in run_labels]
    unknown = sorted(set(requested) - set(available))

    if unknown:
        raise ValueError(f"Unknown run labels: {unknown}")

    return requested


def validate_run_paths(
    basepath: str | Path,
    run_labels: Sequence[str],
) -> None:
    """Verify that all requested embedding run folders exist."""
    basepath = Path(basepath)
    missing = [
        basepath / "runs" / label
        for label in run_labels
        if not (basepath / "runs" / label).is_dir()
    ]

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing embedding run folders:\n{formatted}")


def metadata_for_run(manifest: pd.DataFrame, run_label: str) -> dict:
    """Retrieve manifest metadata for one run."""
    record = manifest_row_for_run(manifest, run_label)
    metadata = {}

    for column in METADATA_COLUMNS:
        source_column = "relation" if column == "removed_relation" else column

        if source_column in record.index:
            metadata[column] = record[source_column]

    return metadata


def original_manifest_row_for_condition(
    manifest: pd.DataFrame,
    window_condition: str,
) -> pd.Series:
    """Return the 0%-removal manifest row for one window condition."""
    matches = manifest.loc[
        (manifest["window_condition"] == str(window_condition))
        & (manifest["removal_percent"] == 0)
    ]

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one original 0% run for window condition "
            f"{window_condition!r}; found {len(matches)}."
        )

    return matches.iloc[0]


def get_manifest_kg_path(
    basepath: str | Path,
    manifest_row: pd.Series,
) -> Path:
    """Resolve the KG TSV path from a manifest row."""
    basepath = Path(basepath)

    if "tsv_path" in manifest_row.index:
        raw_path = manifest_row["tsv_path"]
    elif "tsv_filename" in manifest_row.index:
        raw_path = manifest_row["tsv_filename"]
    else:
        raise ValueError(
            "Manifest must contain either 'tsv_path' or 'tsv_filename'."
        )

    kg_path = Path(str(raw_path))

    if not kg_path.is_absolute():
        kg_path = basepath / kg_path

    return kg_path


def read_person_truth_from_kg(
    kg_path: str | Path,
    relation_string: str = "hasAge",
) -> dict[str, tuple[str, float]]:
    """
    Read person -> (age node, numeric age) mappings directly from a KG TSV.

    For an original KG this is the complete truth mapping.
    For a removal KG this contains only the hasAge triples still present.
    """
    kg_path = Path(kg_path)

    if not kg_path.exists():
        raise FileNotFoundError(f"KG not found: {kg_path}")

    truth: dict[str, tuple[str, float]] = {}

    with kg_path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            stripped = line.rstrip("\r\n")

            if not stripped:
                continue

            parts = stripped.split("\t")

            if len(parts) != 3:
                raise ValueError(
                    f"Malformed triple in {kg_path} at line {line_number}: "
                    f"expected 3 tab-separated values, found {len(parts)}."
                )

            head, relation, tail = parts

            if relation.strip() != relation_string:
                continue

            person = head.strip()
            age_node = tail.strip()
            numeric_age = parse_age_node(age_node)

            if numeric_age is None:
                raise ValueError(
                    f"Could not parse age node {age_node!r} in {kg_path} "
                    f"at line {line_number}."
                )

            if person in truth and truth[person] != (age_node, numeric_age):
                raise ValueError(
                    f"Conflicting ages for {person!r} in {kg_path}: "
                    f"{truth[person][0]!r} and {age_node!r}."
                )

            truth[person] = (age_node, numeric_age)

    return truth


def relation_for_manifest_row(manifest_row: pd.Series) -> str:
    if "relation" in manifest_row.index and not pd.isna(manifest_row["relation"]):
        return str(manifest_row["relation"]).strip()

    return "hasAge"


def build_truth_for_condition(
    basepath: str | Path,
    manifest: pd.DataFrame,
    window_condition: str,
) -> dict[str, tuple[str, float]]:
    """Load authoritative truth from the original KG for one condition."""
    original_row = original_manifest_row_for_condition(
        manifest=manifest,
        window_condition=window_condition,
    )
    relation_string = relation_for_manifest_row(original_row)
    kg_path = get_manifest_kg_path(basepath, original_row)
    truth = read_person_truth_from_kg(kg_path, relation_string)

    if not truth:
        raise ValueError(
            f"No {relation_string!r} triples were found in original KG {kg_path}."
        )

    return truth


def build_run_mapping(
    basepath: str | Path,
    manifest: pd.DataFrame,
    run_label: str,
) -> dict[str, tuple[str, float]]:
    """Load the hasAge triples that are actually present in one evaluated KG."""
    run_row = manifest_row_for_run(manifest, run_label)
    relation_string = relation_for_manifest_row(run_row)
    kg_path = get_manifest_kg_path(basepath, run_row)

    return read_person_truth_from_kg(
        kg_path=kg_path,
        relation_string=relation_string,
    )


def validate_window_truth_matches(
    truth_by_condition: dict[str, dict[str, tuple[str, float]]],
) -> None:
    """Confirm both original KGs contain identical person-to-age truth."""
    if len(truth_by_condition) < 2:
        return

    conditions = list(truth_by_condition)
    reference_condition = conditions[0]
    reference_truth = truth_by_condition[reference_condition]

    for condition in conditions[1:]:
        comparison_truth = truth_by_condition[condition]

        if comparison_truth == reference_truth:
            continue

        reference_people = set(reference_truth)
        comparison_people = set(comparison_truth)

        missing_people = sorted(reference_people - comparison_people)[:10]
        extra_people = sorted(comparison_people - reference_people)[:10]
        conflicting_people = sorted(
            person
            for person in reference_people & comparison_people
            if reference_truth[person] != comparison_truth[person]
        )[:10]

        raise ValueError(
            "The original with-windows and without-windows KGs do not contain "
            "the same person-to-age truth.\n"
            f"Reference condition: {reference_condition}\n"
            f"Comparison condition: {condition}\n"
            f"Missing people: {missing_people}\n"
            f"Extra people: {extra_people}\n"
            f"Conflicting people: {conflicting_people}"
        )


def evaluate_learning_run(
    evaluated_run_path,
    run_label,
    truth_person_to_age,
    run_person_to_age,
    person_model_type="ridge",
    metadata=None,
):
    """Evaluate the learned affine/regression method for one embedding run."""
    metadata = metadata or {}

    person_vectors = load_person_vectors(
        run_path=evaluated_run_path,
        original_index=None,
        relation_filter="hasAge",
        head_prefix="person",
    )
    age_vectors = load_age_vectors(
        run_path=evaluated_run_path,
        relation_filter="hasAge",
        v_prefix="v",
    )

    result = learn_all_positions(
        age_vectors=age_vectors,
        person_vectors=person_vectors,
        v_prefix="v",
        person_model_type=person_model_type,
    )

    rows = []
    ordered_positions = result["person_positions"]["ordered"]

    for position_row in ordered_positions:
        person = str(position_row["person"]).strip()
        truth = truth_person_to_age.get(person)

        if truth is None:
            raise ValueError(
                f"Could not determine true age for {person!r}; "
                "person was not found in the original KG truth mapping."
            )

        true_v, true_age = truth
        run_truth = run_person_to_age.get(person)
        run_v, run_age = run_truth if run_truth is not None else (None, None)

        predicted_age = float(position_row["predicted_v_num"])
        error = predicted_age - true_age

        rows.append(
            {
                "run": run_label,
                "person": person,
                "true_v": true_v,
                "true_age": true_age,
                "run_v": run_v,
                "run_age": run_age,
                "ground_truth_missing_in_run": person not in run_person_to_age,
                "predicted_age": predicted_age,
                "error": error,
                "abs_error": abs(error),
                "axis_position": position_row["axis_position"],
                "nearest_known_person": position_row["nearest_known_person"],
                "nearest_known_age_node": position_row["nearest_known_age_node"],
                "nearest_known_v_num": position_row["nearest_known_v_num"],
                "nearest_known_distance": position_row["nearest_known_distance"],
                **metadata,
            }
        )

    if not rows:
        raise RuntimeError(
            f"Learning evaluation produced no rows for {run_label!r}."
        )

    return pd.DataFrame(rows)


def summarize_runs(
    person_df: pd.DataFrame,
    count_column_name: str = "n_people",
) -> pd.DataFrame:
    """Produce one learned-regression summary row per run."""
    metadata_columns = [
        column for column in METADATA_COLUMNS if column in person_df.columns
    ]
    output_columns = [
        "run",
        *metadata_columns,
        count_column_name,
        "best_person",
        "best_true_age",
        "best_pred_age",
        "best_error",
        "best_abs_error",
        "worst_person",
        "worst_true_age",
        "worst_pred_age",
        "worst_error",
        "worst_abs_error",
        "mean_abs_error",
        "median_abs_error",
        "max_abs_error",
        "mean_error",
        "std_error",
        "std_abs_error",
    ]

    if person_df.empty:
        return pd.DataFrame(columns=output_columns)

    rows = []

    for run, group in person_df.groupby("run", sort=False, dropna=False):
        best = group.loc[group["abs_error"].idxmin()]
        worst = group.loc[group["abs_error"].idxmax()]

        summary_row = {
            "run": run,
            count_column_name: len(group),
            "best_person": best["person"],
            "best_true_age": best["true_age"],
            "best_pred_age": best["predicted_age"],
            "best_error": best["error"],
            "best_abs_error": best["abs_error"],
            "worst_person": worst["person"],
            "worst_true_age": worst["true_age"],
            "worst_pred_age": worst["predicted_age"],
            "worst_error": worst["error"],
            "worst_abs_error": worst["abs_error"],
            "mean_abs_error": group["abs_error"].mean(),
            "median_abs_error": group["abs_error"].median(),
            "max_abs_error": group["abs_error"].max(),
            "mean_error": group["error"].mean(),
            "std_error": group["error"].std(),
            "std_abs_error": group["abs_error"].std(),
        }

        for column in metadata_columns:
            values = group[column].dropna().drop_duplicates().tolist()

            if len(values) > 1:
                raise ValueError(
                    f"Conflicting {column!r} values in run {run!r}: {values}"
                )

            summary_row[column] = values[0] if values else None

        rows.append(summary_row)

    result = pd.DataFrame(rows, columns=output_columns)
    sort_columns = [
        column
        for column in ("removal_percent", "window_condition", "run")
        if column in result.columns
    ]

    if sort_columns:
        result = result.sort_values(
            sort_columns,
            ascending=True,
            na_position="last",
        ).reset_index(drop=True)

    return result


def save_dataframe(
    dataframe: pd.DataFrame,
    output_path,
    description: str,
) -> None:
    """Save a dataframe when an output path is supplied."""
    if output_path is None:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)

    print(f"Saved {description} to: {output_path}")


def analyze_with_learning(
    basepath,
    run_labels=None,
    person_model_type="ridge",
    person_output=None,
    run_output=None,
    missing_run_output=None,
):
    """Evaluate learned regression for all paired runs."""
    basepath = Path(basepath)
    manifest = load_manifest(basepath)
    labels = resolve_run_labels(manifest, run_labels)
    validate_run_paths(basepath, labels)

    window_conditions = (
        manifest["window_condition"].drop_duplicates().astype(str).tolist()
    )
    truth_by_condition = {
        condition: build_truth_for_condition(
            basepath=basepath,
            manifest=manifest,
            window_condition=condition,
        )
        for condition in window_conditions
    }

    validate_window_truth_matches(truth_by_condition)
    dataframes = []

    for run_label in labels:
        metadata = metadata_for_run(manifest, run_label)
        window_condition = str(metadata["window_condition"])
        removal_percent = float(metadata["removal_percent"])

        print(
            f"Evaluating learned regression for {run_label} "
            f"({window_condition}, {removal_percent:g}% removed)"
        )

        run_person_to_age = build_run_mapping(
            basepath=basepath,
            manifest=manifest,
            run_label=run_label,
        )

        dataframe = evaluate_learning_run(
            evaluated_run_path=str(basepath / "runs" / run_label),
            run_label=run_label,
            truth_person_to_age=truth_by_condition[window_condition],
            run_person_to_age=run_person_to_age,
            person_model_type=person_model_type,
            metadata=metadata,
        )
        dataframes.append(dataframe)

    if not dataframes:
        raise RuntimeError("No learned-regression runs were evaluated.")

    person_df = pd.concat(dataframes, ignore_index=True)
    full_summary_df = summarize_runs(person_df, count_column_name="n_people")

    missing_person_df = person_df.loc[
        person_df["ground_truth_missing_in_run"]
    ].copy()
    missing_summary_df = summarize_runs(
        missing_person_df,
        count_column_name="n_missing_people",
    )

    save_dataframe(
        person_df,
        person_output,
        "learned per-person predictions",
    )
    save_dataframe(
        full_summary_df,
        run_output,
        "learned full-run summary",
    )
    save_dataframe(
        missing_summary_df,
        missing_run_output,
        "learned missing-only summary",
    )

    return person_df, full_summary_df, missing_summary_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze learned regression predictions for paired KG runs."
    )
    parser.add_argument("--basepath", required=True)
    parser.add_argument("--run-labels", nargs="+", default=None)
    parser.add_argument(
        "--model-type",
        default="ridge",
        choices=["ridge", "mlp"],
    )
    parser.add_argument("--person-output", default=None)
    parser.add_argument("--run-output", default=None)
    parser.add_argument("--missing-run-output", default=None)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    basepath = Path(args.basepath)

    person_df, full_summary_df, missing_summary_df = analyze_with_learning(
        basepath=basepath,
        run_labels=args.run_labels,
        person_model_type=args.model_type,
        person_output=(
            args.person_output
            or basepath / "learning_person_predictions.csv"
        ),
        run_output=(
            args.run_output
            or basepath / "learning_run_summary.csv"
        ),
        missing_run_output=(
            args.missing_run_output
            or basepath / "learning_run_summary_missing_only.csv"
        ),
    )

    print("\nFull learned-regression summary:")

    if full_summary_df.empty:
        print("No full-run summary rows.")
    else:
        print(full_summary_df.to_string(index=False))

    print("\nMissing-only learned-regression summary:")

    if missing_summary_df.empty:
        print("No missing-relation rows.")
    else:
        print(missing_summary_df.to_string(index=False))