# analyze_data.py

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "run",
    "person",
    "candidate_v_num",
    "distance",
    "true_v_num",
    "ground_truth_missing_in_run",
}

OPTIONAL_METADATA_COLUMNS = (
    "experiment",
    "window_condition",
    "removal_percent",
    "removed_relation",
    "nested_removals",
)


def validate_columns(df: pd.DataFrame) -> None:
    """
    Verify that the query-point CSV contains all required columns.
    """

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns in query-point CSV: "
            f"{sorted(missing)}"
        )


def parse_boolean(value) -> bool:
    """
    Convert common CSV boolean representations into actual booleans.

    This avoids the Python behavior where bool("False") is True.
    """

    if isinstance(value, bool):
        return value

    if pd.isna(value):
        raise ValueError("Boolean value cannot be empty.")

    if isinstance(value, (int, float)):
        if value == 1:
            return True

        if value == 0:
            return False

    normalized = str(value).strip().lower()

    if normalized in {
        "true",
        "t",
        "yes",
        "y",
        "1",
    }:
        return True

    if normalized in {
        "false",
        "f",
        "no",
        "n",
        "0",
    }:
        return False

    raise ValueError(
        f"Could not interpret boolean value: {value!r}"
    )


def get_single_group_value(
    group: pd.DataFrame,
    column: str,
    run: str,
    person: str | None = None,
):
    """
    Return a metadata value that should be constant within a group.

    Raises an error if the same run or person contains conflicting metadata.
    """

    values = (
        group[column]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    if len(values) > 1:
        location = f"run {run!r}"

        if person is not None:
            location += f", person {person!r}"

        raise ValueError(
            f"Column {column!r} has conflicting values for "
            f"{location}: {values}"
        )

    if not values:
        return None

    return values[0]


def add_manifest_metadata(
    df: pd.DataFrame,
    csv_path: str | Path,
) -> pd.DataFrame:
    """
    Add experiment metadata from kg_manifest.csv when it is not already
    present in the query-point CSV.
    """

    csv_path = Path(csv_path)
    manifest_path = csv_path.parent / "kg_manifest.csv"

    if not manifest_path.exists():
        print(
            "Warning: manifest not found; continuing without "
            f"additional run metadata: {manifest_path}"
        )
        return df

    manifest_df = pd.read_csv(manifest_path)

    if "label" not in manifest_df.columns:
        raise ValueError(
            f"Manifest is missing the 'label' column: "
            f"{manifest_path}"
        )

    manifest_df = manifest_df.rename(
        columns={
            "label": "run",
            "relation": "removed_relation",
        }
    )

    if manifest_df["run"].duplicated().any():
        duplicated_runs = (
            manifest_df.loc[
                manifest_df["run"].duplicated(),
                "run",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Manifest contains duplicate run labels: "
            f"{duplicated_runs}"
        )

    metadata_columns = [
        column
        for column in (
            "window_condition",
            "removal_percent",
            "removed_relation",
            "nested_removals",
        )
        if (
            column in manifest_df.columns
            and column not in df.columns
        )
    ]

    if not metadata_columns:
        return df

    result = df.copy()

    result["run"] = (
        result["run"]
        .astype(str)
        .str.strip()
    )

    manifest_df["run"] = (
        manifest_df["run"]
        .astype(str)
        .str.strip()
    )

    return result.merge(
        manifest_df[
            [
                "run",
                *metadata_columns,
            ]
        ],
        on="run",
        how="left",
        validate="many_to_one",
    )


def parse_age_node(value) -> float | None:
    """
    Convert an age-node label into a numeric age.

    Supported examples:

        v42
        v42.5
        42
        42.5
    """

    if value is None or pd.isna(value):
        return None

    label = str(value).strip()

    if not label:
        return None

    match = re.fullmatch(
        r"[vV](-?\d+(?:\.\d+)?)",
        label,
    )

    if match is not None:
        return float(match.group(1))

    try:
        return float(label)
    except ValueError:
        return None


def read_person_truth_from_kg(
    tsv_path: str | Path,
    relation_string: str,
) -> dict[str, float]:
    """
    Read the authoritative person-to-age mapping from an original KG.

    Only triples whose relation equals relation_string are considered.
    """

    tsv_path = Path(tsv_path)

    if not tsv_path.exists():
        raise FileNotFoundError(
            f"Original KG not found: {tsv_path}"
        )

    person_to_age: dict[str, float] = {}

    with tsv_path.open(
        "r",
        encoding="utf-8",
    ) as infile:
        for line_number, line in enumerate(
            infile,
            start=1,
        ):
            stripped = line.rstrip("\r\n")

            if not stripped:
                continue

            parts = stripped.split("\t")

            if len(parts) != 3:
                raise ValueError(
                    f"Malformed triple in {tsv_path} at line "
                    f"{line_number}: expected 3 tab-separated values, "
                    f"found {len(parts)}."
                )

            head, relation, tail = parts

            if relation.strip() != relation_string:
                continue

            person = head.strip()
            age = parse_age_node(tail)

            if age is None:
                raise ValueError(
                    f"Could not parse age node {tail!r} in "
                    f"{tsv_path} at line {line_number}."
                )

            if (
                person in person_to_age
                and person_to_age[person] != age
            ):
                raise ValueError(
                    f"Person {person!r} has conflicting ages in "
                    f"{tsv_path}: {person_to_age[person]} and {age}."
                )

            person_to_age[person] = age

    if not person_to_age:
        raise ValueError(
            f"No triples with relation {relation_string!r} "
            f"were found in {tsv_path}."
        )

    return person_to_age


def recover_true_v_num(
    df: pd.DataFrame,
    csv_path: str | Path,
) -> pd.DataFrame:
    """
    Recover missing true_v_num values from the original KGs listed in
    kg_manifest.csv.

    The query-point evaluator may leave true_v_num blank for some or all
    candidate rows. The canonical original KG still contains the authoritative
    person-to-age mapping, so this function fills those missing values before
    the query-point predictions are analyzed.

    The with-windows and without-windows original KGs are also checked to make
    sure they contain the same person-to-age truth.
    """

    result = df.copy()

    existing_truth = pd.to_numeric(
        result["true_v_num"],
        errors="coerce",
    )

    missing_mask = existing_truth.isna()

    if not missing_mask.any():
        result["true_v_num"] = existing_truth
        return result

    csv_path = Path(csv_path)
    basepath = csv_path.parent
    manifest_path = basepath / "kg_manifest.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(
            "The query-point CSV contains missing true_v_num values, "
            "but the KG manifest could not be found:\n"
            f"  {manifest_path}"
        )

    manifest = pd.read_csv(manifest_path)

    required_manifest_columns = {
        "label",
        "tsv_path",
        "window_condition",
        "removal_percent",
        "relation",
    }

    missing_manifest_columns = (
        required_manifest_columns
        - set(manifest.columns)
    )

    if missing_manifest_columns:
        raise ValueError(
            "Manifest is missing columns required to recover true ages: "
            f"{sorted(missing_manifest_columns)}"
        )

    manifest["removal_percent"] = pd.to_numeric(
        manifest["removal_percent"],
        errors="coerce",
    )

    original_rows = manifest.loc[
        manifest["removal_percent"] == 0
    ].copy()

    if original_rows.empty:
        raise ValueError(
            "kg_manifest.csv contains no original runs with "
            "removal_percent equal to 0."
        )

    truth_maps: list[
        tuple[
            str,
            str,
            dict[str, float],
        ]
    ] = []

    for row in original_rows.itertuples(index=False):
        condition = str(row.window_condition)
        relation_string = str(row.relation)

        relative_kg_path = Path(str(row.tsv_path))

        if relative_kg_path.is_absolute():
            kg_path = relative_kg_path
        else:
            kg_path = basepath / relative_kg_path

        truth_map = read_person_truth_from_kg(
            tsv_path=kg_path,
            relation_string=relation_string,
        )

        truth_maps.append(
            (
                condition,
                str(kg_path),
                truth_map,
            )
        )

    (
        reference_condition,
        reference_path,
        canonical_truth,
    ) = truth_maps[0]

    for (
        condition,
        kg_path,
        truth_map,
    ) in truth_maps[1:]:
        if truth_map == canonical_truth:
            continue

        canonical_people = set(
            canonical_truth
        )

        comparison_people = set(
            truth_map
        )

        missing_people = sorted(
            canonical_people - comparison_people
        )[:10]

        extra_people = sorted(
            comparison_people - canonical_people
        )[:10]

        conflicting_people = sorted(
            person
            for person in (
                canonical_people
                & comparison_people
            )
            if (
                canonical_truth[person]
                != truth_map[person]
            )
        )[:10]

        raise ValueError(
            "The original with-windows and without-windows KGs "
            "do not contain the same person-to-age mapping.\n"
            f"Reference condition: {reference_condition}\n"
            f"Reference KG: {reference_path}\n"
            f"Comparison condition: {condition}\n"
            f"Comparison KG: {kg_path}\n"
            f"People missing from comparison: {missing_people}\n"
            f"Extra people in comparison: {extra_people}\n"
            f"People with conflicting ages: {conflicting_people}"
        )

    person_keys = (
        result["person"]
        .astype(str)
        .str.strip()
    )

    recovered_truth = person_keys.map(
        canonical_truth
    )

    conflict_mask = (
        existing_truth.notna()
        & recovered_truth.notna()
        & (
            (
                existing_truth
                - recovered_truth
            ).abs()
            > 1e-9
        )
    )

    if conflict_mask.any():
        conflict_examples = result.loc[
            conflict_mask,
            [
                "run",
                "person",
                "true_v_num",
            ],
        ].head(10).copy()

        conflict_examples[
            "canonical_true_v_num"
        ] = (
            recovered_truth.loc[
                conflict_mask
            ]
            .head(10)
            .values
        )

        raise ValueError(
            "Existing true_v_num values conflict with the "
            "original KG truth:\n"
            f"{conflict_examples.to_string(index=False)}"
        )

    result["true_v_num"] = (
        existing_truth.fillna(
            recovered_truth
        )
    )

    unresolved_mask = (
        result["true_v_num"].isna()
    )

    if unresolved_mask.any():
        unresolved_people = (
            result.loc[
                unresolved_mask,
                "person",
            ]
            .astype(str)
            .drop_duplicates()
            .head(20)
            .tolist()
        )

        raise ValueError(
            "Could not recover true ages for some people from "
            "the original KGs. Example people:\n"
            f"  {unresolved_people}"
        )

    recovered_count = int(
        (
            missing_mask
            & result["true_v_num"].notna()
        ).sum()
    )

    unique_people_recovered = int(
        result.loc[
            missing_mask,
            "person",
        ]
        .astype(str)
        .nunique()
    )

    print(
        f"Recovered {recovered_count} missing true_v_num values "
        f"for {unique_people_recovered} people from the original KG."
    )

    result["true_v_num"] = pd.to_numeric(
        result["true_v_num"],
        errors="raise",
    )

    return result


def prepare_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize the query-point dataframe before analysis.
    """

    result = df.copy()

    for column in (
        "candidate_v_num",
        "distance",
        "true_v_num",
    ):
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        )

    required_numeric_columns = [
        "candidate_v_num",
        "distance",
        "true_v_num",
    ]

    numeric_null_columns = [
        column
        for column in required_numeric_columns
        if result[column].isna().any()
    ]

    if numeric_null_columns:
        raise ValueError(
            "Required numeric columns still contain missing values: "
            f"{numeric_null_columns}"
        )

    result[
        "ground_truth_missing_in_run"
    ] = result[
        "ground_truth_missing_in_run"
    ].apply(parse_boolean)

    if "removal_percent" in result.columns:
        result["removal_percent"] = pd.to_numeric(
            result["removal_percent"],
            errors="coerce",
        )

    if "nested_removals" in result.columns:
        non_null_mask = (
            result["nested_removals"].notna()
        )

        result.loc[
            non_null_mask,
            "nested_removals",
        ] = result.loc[
            non_null_mask,
            "nested_removals",
        ].apply(parse_boolean)

    return result


def build_person_predictions(
    df: pd.DataFrame,
    max_k: int,
) -> pd.DataFrame:
    """
    Create one query-point prediction row for each person in each run.
    """

    if max_k < 1:
        raise ValueError(
            "max_k must be at least 1."
        )

    metadata_columns = [
        column
        for column in OPTIONAL_METADATA_COLUMNS
        if column in df.columns
    ]

    rows = []

    grouped = df.groupby(
        [
            "run",
            "person",
        ],
        sort=False,
        dropna=False,
    )

    for (
        run,
        person,
    ), group in grouped:
        group = (
            group
            .sort_values(
                "distance",
                ascending=True,
            )
            .reset_index(drop=True)
        )

        true_ages = (
            group["true_v_num"]
            .dropna()
            .astype(float)
            .unique()
        )

        if len(true_ages) != 1:
            raise ValueError(
                f"{run} / {person} must have exactly one true age; "
                f"found {true_ages.tolist()}."
            )

        missing_values = (
            group[
                "ground_truth_missing_in_run"
            ]
            .drop_duplicates()
            .tolist()
        )

        if len(missing_values) != 1:
            raise ValueError(
                f"{run} / {person} has conflicting "
                "ground_truth_missing_in_run values: "
                f"{missing_values}"
            )

        top_ages = (
            group["candidate_v_num"]
            .head(max_k)
            .astype(float)
            .tolist()
        )

        if len(top_ages) < max_k:
            raise ValueError(
                f"{run} / {person} has only "
                f"{len(top_ages)} candidates; need {max_k}."
            )

        true_age = float(
            true_ages[0]
        )

        top1_prediction = float(
            top_ages[0]
        )

        top1_error = (
            top1_prediction
            - true_age
        )

        row = {
            "run": run,
            "person": person,
            "true_age": true_age,
            "ground_truth_missing_in_run": bool(
                missing_values[0]
            ),
            "top1_pred_age": top1_prediction,
            "top1_error": top1_error,
            "top1_abs_error": abs(
                top1_error
            ),
        }

        for column in metadata_columns:
            row[column] = get_single_group_value(
                group=group,
                column=column,
                run=str(run),
                person=str(person),
            )

        for k in range(
            1,
            max_k + 1,
        ):
            prediction = (
                sum(top_ages[:k])
                / k
            )

            error = (
                prediction
                - true_age
            )

            row[
                f"top{k}_avg_pred_age"
            ] = prediction

            row[
                f"top{k}_avg_error"
            ] = error

            row[
                f"top{k}_avg_abs_error"
            ] = abs(error)

        rows.append(row)

    if not rows:
        raise ValueError(
            "No query-point person predictions were produced."
        )

    return pd.DataFrame(rows)


def summary_columns(
    max_k: int,
    count_column_name: str,
    metadata_columns: list[str],
) -> list[str]:
    """
    Construct the output columns for run-level summaries.
    """

    columns = [
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
        "mean_top1_abs_error",
        "median_top1_abs_error",
        "max_top1_abs_error",
        "mean_top1_error",
        "std_top1_error",
        "std_top1_abs_error",
    ]

    for k in range(
        1,
        max_k + 1,
    ):
        columns.extend(
            [
                f"mean_top{k}_avg_abs_error",
                f"median_top{k}_avg_abs_error",
                f"std_top{k}_avg_error",
                f"std_top{k}_avg_abs_error",
            ]
        )

    return columns


def summarize_runs(
    person_df: pd.DataFrame,
    max_k: int,
    count_column_name: str = "n_people",
) -> pd.DataFrame:
    """
    Produce one summary row for each KG run.
    """

    metadata_columns = [
        column
        for column in OPTIONAL_METADATA_COLUMNS
        if column in person_df.columns
    ]

    output_columns = summary_columns(
        max_k=max_k,
        count_column_name=count_column_name,
        metadata_columns=metadata_columns,
    )

    if person_df.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    rows = []

    for run, group in person_df.groupby(
        "run",
        sort=False,
        dropna=False,
    ):
        best = group.loc[
            group[
                "top1_abs_error"
            ].idxmin()
        ]

        worst = group.loc[
            group[
                "top1_abs_error"
            ].idxmax()
        ]

        row = {
            "run": run,
            count_column_name: len(group),

            "best_person": best["person"],
            "best_true_age": best["true_age"],
            "best_pred_age": best["top1_pred_age"],
            "best_error": best["top1_error"],
            "best_abs_error": best["top1_abs_error"],

            "worst_person": worst["person"],
            "worst_true_age": worst["true_age"],
            "worst_pred_age": worst["top1_pred_age"],
            "worst_error": worst["top1_error"],
            "worst_abs_error": worst["top1_abs_error"],

            "mean_top1_abs_error": (
                group[
                    "top1_abs_error"
                ].mean()
            ),

            "median_top1_abs_error": (
                group[
                    "top1_abs_error"
                ].median()
            ),

            "max_top1_abs_error": (
                group[
                    "top1_abs_error"
                ].max()
            ),

            "mean_top1_error": (
                group[
                    "top1_error"
                ].mean()
            ),

            "std_top1_error": (
                group[
                    "top1_error"
                ].std()
            ),

            "std_top1_abs_error": (
                group[
                    "top1_abs_error"
                ].std()
            ),
        }

        for column in metadata_columns:
            row[column] = get_single_group_value(
                group=group,
                column=column,
                run=str(run),
            )

        for k in range(
            1,
            max_k + 1,
        ):
            abs_error_column = (
                f"top{k}_avg_abs_error"
            )

            error_column = (
                f"top{k}_avg_error"
            )

            row[
                f"mean_top{k}_avg_abs_error"
            ] = group[
                abs_error_column
            ].mean()

            row[
                f"median_top{k}_avg_abs_error"
            ] = group[
                abs_error_column
            ].median()

            row[
                f"std_top{k}_avg_error"
            ] = group[
                error_column
            ].std()

            row[
                f"std_top{k}_avg_abs_error"
            ] = group[
                abs_error_column
            ].std()

        rows.append(row)

    result = pd.DataFrame(
        rows,
        columns=output_columns,
    )

    sort_columns = [
        column
        for column in (
            "removal_percent",
            "window_condition",
            "run",
        )
        if column in result.columns
    ]

    if sort_columns:
        result = (
            result
            .sort_values(
                sort_columns,
                ascending=True,
                na_position="last",
            )
            .reset_index(drop=True)
        )

    return result


def save_dataframe(
    df: pd.DataFrame,
    output_path: str | Path | None,
    description: str,
) -> None:
    """
    Save a dataframe and create its parent directory if needed.
    """

    if output_path is None:
        return

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )

    print(
        f"Saved {description} to: {path}"
    )


def analyze_data(
    csv_path,
    person_output=None,
    run_output=None,
    missing_run_output=None,
    max_k=10,
):
    """
    Analyze query-point predictions for every with-windows and
    without-windows KG run.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Query-point CSV not found: {csv_path}"
        )

    df = pd.read_csv(
        csv_path
    )

    if df.empty:
        raise ValueError(
            f"Query-point CSV contains no rows: {csv_path}"
        )

    validate_columns(df)

    df = add_manifest_metadata(
        df=df,
        csv_path=csv_path,
    )

    # This must happen before prepare_dataframe() and before
    # build_person_predictions().
    df = recover_true_v_num(
        df=df,
        csv_path=csv_path,
    )

    df = prepare_dataframe(
        df
    )

    person_df = build_person_predictions(
        df=df,
        max_k=max_k,
    )

    full_summary_df = summarize_runs(
        person_df=person_df,
        max_k=max_k,
        count_column_name="n_people",
    )

    missing_person_df = person_df.loc[
        person_df[
            "ground_truth_missing_in_run"
        ]
    ].copy()

    missing_summary_df = summarize_runs(
        person_df=missing_person_df,
        max_k=max_k,
        count_column_name="n_missing_people",
    )

    save_dataframe(
        df=person_df,
        output_path=person_output,
        description=(
            "query-point per-person predictions"
        ),
    )

    save_dataframe(
        df=full_summary_df,
        output_path=run_output,
        description=(
            "query-point full-run summary"
        ),
    )

    save_dataframe(
        df=missing_summary_df,
        output_path=missing_run_output,
        description=(
            "query-point missing-only summary"
        ),
    )

    return (
        person_df,
        full_summary_df,
        missing_summary_df,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze query-point age predictions for paired "
            "with-windows and without-windows KG runs."
        )
    )

    parser.add_argument(
        "--csv",
        required=True,
        help=(
            "Path to query_point_distances_all_runs.csv."
        ),
    )

    parser.add_argument(
        "--person-output",
        default=None,
        help=(
            "Output path for per-person predictions."
        ),
    )

    parser.add_argument(
        "--run-output",
        default=None,
        help=(
            "Output path for the full run summary."
        ),
    )

    parser.add_argument(
        "--missing-run-output",
        default=None,
        help=(
            "Output path for the missing-only run summary."
        ),
    )

    parser.add_argument(
        "--max-k",
        type=int,
        default=10,
    )

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    csv_path = Path(
        args.csv
    )

    output_directory = (
        csv_path.parent
    )

    person_output = (
        args.person_output
        or (
            output_directory
            / "person_year_predictions.csv"
        )
    )

    run_output = (
        args.run_output
        or (
            output_directory
            / "run_year_summary.csv"
        )
    )

    missing_run_output = (
        args.missing_run_output
        or (
            output_directory
            / "run_year_summary_missing_only.csv"
        )
    )

    (
        person_df,
        full_summary_df,
        missing_summary_df,
    ) = analyze_data(
        csv_path=csv_path,
        person_output=person_output,
        run_output=run_output,
        missing_run_output=missing_run_output,
        max_k=args.max_k,
    )

    print(
        "\nFull run summary:"
    )

    if full_summary_df.empty:
        print(
            "No full-run summary rows were produced."
        )
    else:
        print(
            full_summary_df.to_string(
                index=False
            )
        )

    print(
        "\nMissing-only run summary:"
    )

    if missing_summary_df.empty:
        print(
            "No missing-relation rows."
        )
    else:
        print(
            missing_summary_df.to_string(
                index=False
            )
        )