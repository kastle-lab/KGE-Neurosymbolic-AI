# compare_methods.py

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"CSV contains no rows: {path}"
        )

    return df


def parse_boolean(value) -> bool:
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        raise ValueError(
            "Boolean value cannot be empty."
        )

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


def bootstrap_mean_ci(
    differences: np.ndarray,
    seed: int = 42,
    n_bootstrap: int = 5000,
) -> tuple[float, float]:
    if len(differences) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(
        seed
    )

    sample_indices = rng.integers(
        0,
        len(differences),
        size=(
            n_bootstrap,
            len(differences),
        ),
    )

    means = differences[
        sample_indices
    ].mean(axis=1)

    return (
        float(
            np.percentile(
                means,
                2.5,
            )
        ),
        float(
            np.percentile(
                means,
                97.5,
            )
        ),
    )


def paired_statistics(
    first: pd.Series,
    second: pd.Series,
    seed: int = 42,
) -> dict:
    first_values = pd.to_numeric(
        first,
        errors="raise",
    ).to_numpy(float)

    second_values = pd.to_numeric(
        second,
        errors="raise",
    ).to_numpy(float)

    if len(first_values) != len(second_values):
        raise ValueError(
            "Paired arrays have unequal lengths."
        )

    differences = (
        first_values
        - second_values
    )

    n = len(
        differences
    )

    if n >= 2:
        paired_t_p_value = float(
            ttest_rel(
                first_values,
                second_values,
                nan_policy="raise",
            ).pvalue
        )
    else:
        paired_t_p_value = np.nan

    if (
        n >= 1
        and not np.allclose(
            differences,
            0,
        )
    ):
        try:
            wilcoxon_p_value = float(
                wilcoxon(
                    differences
                ).pvalue
            )
        except ValueError:
            wilcoxon_p_value = np.nan
    else:
        wilcoxon_p_value = (
            1.0
            if n >= 1
            else np.nan
        )

    difference_std = (
        float(
            np.std(
                differences,
                ddof=1,
            )
        )
        if n >= 2
        else np.nan
    )

    cohen_dz = (
        float(
            np.mean(
                differences
            )
            / difference_std
        )
        if (
            n >= 2
            and difference_std != 0
        )
        else np.nan
    )

    ci_low, ci_high = (
        bootstrap_mean_ci(
            differences,
            seed=seed,
        )
    )

    return {
        "n_pairs": n,

        "mean_difference": (
            float(
                np.mean(
                    differences
                )
            )
            if n
            else np.nan
        ),

        "median_difference": (
            float(
                np.median(
                    differences
                )
            )
            if n
            else np.nan
        ),

        "std_difference": difference_std,
        "paired_t_p_value": paired_t_p_value,
        "wilcoxon_p_value": wilcoxon_p_value,
        "cohen_dz": cohen_dz,
        "bootstrap_mean_diff_ci_low": ci_low,
        "bootstrap_mean_diff_ci_high": ci_high,
    }


def load_manifest(
    csv_path: str | Path,
) -> pd.DataFrame:
    basepath = Path(
        csv_path
    ).parent

    manifest_path = (
        basepath
        / "kg_manifest.csv"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path
    )

    required_columns = {
        "label",
        "window_condition",
        "removal_percent",
    }

    missing = (
        required_columns
        - set(
            manifest.columns
        )
    )

    if missing:
        raise ValueError(
            "Manifest is missing required columns: "
            f"{sorted(missing)}"
        )

    if manifest[
        "label"
    ].duplicated().any():
        raise ValueError(
            "Manifest contains duplicate run labels."
        )

    manifest["label"] = (
        manifest["label"]
        .astype(str)
        .str.strip()
    )

    manifest[
        "removal_percent"
    ] = pd.to_numeric(
        manifest[
            "removal_percent"
        ],
        errors="raise",
    )

    return manifest


def attach_manifest(
    df: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    if "run" not in df.columns:
        return df

    result = df.copy()

    result["run"] = (
        result["run"]
        .astype(str)
        .str.strip()
    )

    metadata = (
        manifest.rename(
            columns={
                "label": "run",
                "relation": (
                    "removed_relation"
                ),
            }
        )
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
            column
            in metadata.columns
            and column
            not in result.columns
        )
    ]

    if not metadata_columns:
        return result

    return result.merge(
        metadata[
            [
                "run",
                *metadata_columns,
            ]
        ],
        on="run",
        how="left",
        validate="many_to_one",
    )


def get_manifest_kg_path(
    basepath: Path,
    row: pd.Series,
) -> Path:
    if "tsv_path" in row.index:
        raw_path = row[
            "tsv_path"
        ]
    elif "tsv_filename" in row.index:
        raw_path = row[
            "tsv_filename"
        ]
    else:
        raise ValueError(
            "Manifest must contain either "
            "'tsv_path' or 'tsv_filename'."
        )

    kg_path = Path(
        str(
            raw_path
        )
    )

    if not kg_path.is_absolute():
        kg_path = (
            basepath
            / kg_path
        )

    return kg_path


def read_people_with_relation(
    kg_path: str | Path,
    relation_string: str,
) -> set[str]:
    kg_path = Path(
        kg_path
    )

    if not kg_path.exists():
        raise FileNotFoundError(
            f"KG file not found: {kg_path}"
        )

    people = set()

    with kg_path.open(
        "r",
        encoding="utf-8",
    ) as infile:
        for line_number, line in enumerate(
            infile,
            start=1,
        ):
            stripped = line.rstrip(
                "\r\n"
            )

            if not stripped:
                continue

            parts = stripped.split(
                "\t"
            )

            if len(parts) != 3:
                raise ValueError(
                    f"Malformed triple in {kg_path} "
                    f"at line {line_number}."
                )

            head, relation, _ = parts

            if (
                relation.strip()
                == relation_string
            ):
                people.add(
                    head.strip()
                )

    return people


def build_missing_map(
    manifest: pd.DataFrame,
    basepath: Path,
) -> dict[
    tuple[str, str],
    bool,
]:
    """
    Determine missing hasAge status from the actual KG files.

    Returns:
        (run, person) -> is_missing
    """

    missing_map = {}

    conditions = (
        manifest[
            "window_condition"
        ]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    for condition in conditions:
        original_rows = manifest.loc[
            (
                manifest[
                    "window_condition"
                ].astype(str)
                == condition
            )
            & (
                manifest[
                    "removal_percent"
                ]
                == 0
            )
        ]

        if len(
            original_rows
        ) != 1:
            raise ValueError(
                "Expected one original run for "
                f"{condition!r}; found "
                f"{len(original_rows)}."
            )

        original_row = (
            original_rows.iloc[0]
        )

        relation_string = (
            str(
                original_row[
                    "relation"
                ]
            )
            if (
                "relation"
                in original_row.index
            )
            else "hasAge"
        )

        original_path = (
            get_manifest_kg_path(
                basepath,
                original_row,
            )
        )

        original_people = (
            read_people_with_relation(
                original_path,
                relation_string,
            )
        )

        condition_rows = manifest.loc[
            manifest[
                "window_condition"
            ].astype(str)
            == condition
        ]

        for _, run_row in (
            condition_rows.iterrows()
        ):
            run_label = str(
                run_row[
                    "label"
                ]
            )

            run_path = (
                get_manifest_kg_path(
                    basepath,
                    run_row,
                )
            )

            run_people = (
                read_people_with_relation(
                    run_path,
                    relation_string,
                )
            )

            for person in original_people:
                missing_map[
                    (
                        run_label,
                        person,
                    )
                ] = (
                    person
                    not in run_people
                )

    return missing_map


def apply_authoritative_missing_flags(
    df: pd.DataFrame,
    missing_map: dict[
        tuple[str, str],
        bool,
    ],
) -> pd.DataFrame:
    result = df.copy()

    keys = list(
        zip(
            result[
                "run"
            ].astype(str),
            result[
                "person"
            ].astype(str),
        )
    )

    authoritative_flags = [
        missing_map.get(
            (
                run.strip(),
                person.strip(),
            )
        )
        for run, person in keys
    ]

    unresolved = [
        (
            run,
            person,
        )
        for (
            run,
            person,
        ), flag in zip(
            keys,
            authoritative_flags,
        )
        if flag is None
    ]

    if unresolved:
        raise ValueError(
            "Could not determine missing status "
            "from KG files for examples: "
            f"{unresolved[:20]}"
        )

    result[
        "ground_truth_missing_in_run"
    ] = authoritative_flags

    return result


def compare_methods(
    query_person_csv,
    learning_person_csv,
    output_csv,
    missing_only=False,
    seed=42,
):
    query = read_csv(
        query_person_csv
    )

    learning = read_csv(
        learning_person_csv
    )

    manifest = load_manifest(
        query_person_csv
    )

    basepath = Path(
        query_person_csv
    ).parent

    query = attach_manifest(
        query,
        manifest,
    )

    learning = attach_manifest(
        learning,
        manifest,
    )

    missing_map = build_missing_map(
        manifest=manifest,
        basepath=basepath,
    )

    query = apply_authoritative_missing_flags(
        query,
        missing_map,
    )

    learning = apply_authoritative_missing_flags(
        learning,
        missing_map,
    )

    required_query = {
        "run",
        "person",
        "true_age",
        "ground_truth_missing_in_run",
        "top1_abs_error",
    }

    required_learning = {
        "run",
        "person",
        "true_age",
        "ground_truth_missing_in_run",
        "abs_error",
    }

    query_missing_columns = (
        required_query
        - set(
            query.columns
        )
    )

    learning_missing_columns = (
        required_learning
        - set(
            learning.columns
        )
    )

    if query_missing_columns:
        raise ValueError(
            "Query prediction CSV is missing: "
            f"{sorted(query_missing_columns)}"
        )

    if learning_missing_columns:
        raise ValueError(
            "Learning prediction CSV is missing: "
            f"{sorted(learning_missing_columns)}"
        )

    query_columns = [
        "run",
        "person",
        "true_age",
        "ground_truth_missing_in_run",
        "top1_abs_error",
        *[
            column
            for column in (
                "window_condition",
                "removal_percent",
                "removed_relation",
                "nested_removals",
            )
            if column in query.columns
        ],
    ]

    learning_columns = [
        "run",
        "person",
        "true_age",
        "ground_truth_missing_in_run",
        "abs_error",
    ]

    merged = query[
        query_columns
    ].merge(
        learning[
            learning_columns
        ],
        on=[
            "run",
            "person",
            "true_age",
        ],
        how="inner",
        suffixes=(
            "_query",
            "_learning",
        ),
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError(
            "No matching query-point and "
            "learning prediction rows."
        )

    merged[
        "ground_truth_missing_in_run"
    ] = merged[
        "ground_truth_missing_in_run_query"
    ]

    if missing_only:
        merged = merged.loc[
            merged[
                "ground_truth_missing_in_run"
            ]
        ].copy()

    rows = []

    for run, group in merged.groupby(
        "run",
        sort=False,
    ):
        stats = paired_statistics(
            group[
                "top1_abs_error"
            ],
            group[
                "abs_error"
            ],
            seed=seed,
        )

        row = {
            "run": run,
            "missing_only": bool(
                missing_only
            ),
            "n_pairs": stats[
                "n_pairs"
            ],

            "mean_query_abs_error": (
                group[
                    "top1_abs_error"
                ].mean()
            ),

            "std_query_abs_error": (
                group[
                    "top1_abs_error"
                ].std()
            ),

            "mean_learning_abs_error": (
                group[
                    "abs_error"
                ].mean()
            ),

            "std_learning_abs_error": (
                group[
                    "abs_error"
                ].std()
            ),

            "mean_error_difference_query_minus_learning": (
                stats[
                    "mean_difference"
                ]
            ),

            "median_error_difference_query_minus_learning": (
                stats[
                    "median_difference"
                ]
            ),

            "std_error_difference_query_minus_learning": (
                stats[
                    "std_difference"
                ]
            ),

            "paired_t_p_value": (
                stats[
                    "paired_t_p_value"
                ]
            ),

            "wilcoxon_p_value": (
                stats[
                    "wilcoxon_p_value"
                ]
            ),

            "cohen_dz_query_minus_learning": (
                stats[
                    "cohen_dz"
                ]
            ),

            "bootstrap_mean_diff_ci_low": (
                stats[
                    "bootstrap_mean_diff_ci_low"
                ]
            ),

            "bootstrap_mean_diff_ci_high": (
                stats[
                    "bootstrap_mean_diff_ci_high"
                ]
            ),
        }

        query_mean = row[
            "mean_query_abs_error"
        ]

        learning_mean = row[
            "mean_learning_abs_error"
        ]

        if np.isclose(
            query_mean,
            learning_mean,
        ):
            row[
                "better_method_by_mean"
            ] = "tie"

        elif query_mean < learning_mean:
            row[
                "better_method_by_mean"
            ] = "query_point"

        else:
            row[
                "better_method_by_mean"
            ] = "learned_regression"

        for column in (
            "window_condition",
            "removal_percent",
            "removed_relation",
            "nested_removals",
        ):
            if column in group.columns:
                values = (
                    group[column]
                    .dropna()
                    .drop_duplicates()
                    .tolist()
                )

                row[column] = (
                    values[0]
                    if values
                    else None
                )

        rows.append(
            row
        )

    result = pd.DataFrame(
        rows
    )

    if not result.empty:
        sort_columns = [
            column
            for column in (
                "removal_percent",
                "window_condition",
                "run",
            )
            if column in result.columns
        ]

        result = (
            result
            .sort_values(
                sort_columns
            )
            .reset_index(
                drop=True
            )
        )

    output_path = Path(
        output_csv
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved method comparison to: "
        f"{output_path}"
    )

    return result


def compare_window_conditions(
    person_csv,
    method,
    output_csv,
    missing_only=False,
    seed=42,
):
    df = read_csv(
        person_csv
    )

    manifest = load_manifest(
        person_csv
    )

    basepath = Path(
        person_csv
    ).parent

    df = attach_manifest(
        df,
        manifest,
    )

    missing_map = build_missing_map(
        manifest=manifest,
        basepath=basepath,
    )

    df = apply_authoritative_missing_flags(
        df,
        missing_map,
    )

    if method == "query_point":
        error_column = (
            "top1_abs_error"
        )

    elif method == "learned_regression":
        error_column = (
            "abs_error"
        )

    else:
        raise ValueError(
            "method must be 'query_point' "
            "or 'learned_regression'."
        )

    required = {
        "person",
        "true_age",
        "window_condition",
        "removal_percent",
        "ground_truth_missing_in_run",
        error_column,
    }

    missing_columns = (
        required
        - set(
            df.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Cannot compare window conditions; "
            f"missing columns: "
            f"{sorted(missing_columns)}"
        )

    if missing_only:
        df = df.loc[
            df[
                "ground_truth_missing_in_run"
            ]
        ].copy()

    with_windows = df.loc[
        df[
            "window_condition"
        ]
        == "with_windows",
        [
            "person",
            "true_age",
            "removal_percent",
            "ground_truth_missing_in_run",
            error_column,
            "run",
        ],
    ].rename(
        columns={
            error_column: (
                "with_windows_abs_error"
            ),
            "run": (
                "with_windows_run"
            ),
            "ground_truth_missing_in_run": (
                "missing_with_windows"
            ),
        }
    )

    without_windows = df.loc[
        df[
            "window_condition"
        ]
        == "without_windows",
        [
            "person",
            "true_age",
            "removal_percent",
            "ground_truth_missing_in_run",
            error_column,
            "run",
        ],
    ].rename(
        columns={
            error_column: (
                "without_windows_abs_error"
            ),
            "run": (
                "without_windows_run"
            ),
            "ground_truth_missing_in_run": (
                "missing_without_windows"
            ),
        }
    )

    paired = with_windows.merge(
        without_windows,
        on=[
            "person",
            "true_age",
            "removal_percent",
        ],
        how="inner",
        validate="one_to_one",
    )

    if paired.empty:
        raise ValueError(
            "No paired with-windows and "
            "without-windows rows were found."
        )

    rows = []

    for removal_percent, group in (
        paired.groupby(
            "removal_percent",
            sort=True,
        )
    ):
        stats = paired_statistics(
            group[
                "with_windows_abs_error"
            ],
            group[
                "without_windows_abs_error"
            ],
            seed=seed,
        )

        with_mean = group[
            "with_windows_abs_error"
        ].mean()

        without_mean = group[
            "without_windows_abs_error"
        ].mean()

        if np.isclose(
            with_mean,
            without_mean,
        ):
            better = "tie"

        elif with_mean < without_mean:
            better = "with_windows"

        else:
            better = "without_windows"

        rows.append(
            {
                "method": method,
                "removal_percent": (
                    removal_percent
                ),
                "missing_only": bool(
                    missing_only
                ),
                "n_pairs": stats[
                    "n_pairs"
                ],

                "with_windows_run": (
                    group[
                        "with_windows_run"
                    ].iloc[0]
                ),

                "without_windows_run": (
                    group[
                        "without_windows_run"
                    ].iloc[0]
                ),

                "mean_with_windows_abs_error": (
                    with_mean
                ),

                "std_with_windows_abs_error": (
                    group[
                        "with_windows_abs_error"
                    ].std()
                ),

                "mean_without_windows_abs_error": (
                    without_mean
                ),

                "std_without_windows_abs_error": (
                    group[
                        "without_windows_abs_error"
                    ].std()
                ),

                "mean_error_difference_with_minus_without": (
                    stats[
                        "mean_difference"
                    ]
                ),

                "median_error_difference_with_minus_without": (
                    stats[
                        "median_difference"
                    ]
                ),

                "std_error_difference_with_minus_without": (
                    stats[
                        "std_difference"
                    ]
                ),

                "better_window_condition_by_mean": (
                    better
                ),

                "paired_t_p_value": (
                    stats[
                        "paired_t_p_value"
                    ]
                ),

                "wilcoxon_p_value": (
                    stats[
                        "wilcoxon_p_value"
                    ]
                ),

                "cohen_dz_with_minus_without": (
                    stats[
                        "cohen_dz"
                    ]
                ),

                "bootstrap_mean_diff_ci_low": (
                    stats[
                        "bootstrap_mean_diff_ci_low"
                    ]
                ),

                "bootstrap_mean_diff_ci_high": (
                    stats[
                        "bootstrap_mean_diff_ci_high"
                    ]
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if not result.empty:
        result = (
            result
            .sort_values(
                "removal_percent"
            )
            .reset_index(
                drop=True
            )
        )

    output_path = Path(
        output_csv
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print(
        "Saved window-condition "
        f"comparison to: {output_path}"
    )

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    subparsers = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )

    method_parser = (
        subparsers.add_parser(
            "methods"
        )
    )

    method_parser.add_argument(
        "--query-person-csv",
        required=True,
    )

    method_parser.add_argument(
        "--learning-person-csv",
        required=True,
    )

    method_parser.add_argument(
        "--output-csv",
        required=True,
    )

    method_parser.add_argument(
        "--missing-only",
        action="store_true",
    )

    window_parser = (
        subparsers.add_parser(
            "windows"
        )
    )

    window_parser.add_argument(
        "--person-csv",
        required=True,
    )

    window_parser.add_argument(
        "--method",
        required=True,
        choices=[
            "query_point",
            "learned_regression",
        ],
    )

    window_parser.add_argument(
        "--output-csv",
        required=True,
    )

    window_parser.add_argument(
        "--missing-only",
        action="store_true",
    )

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.command == "methods":
        compare_methods(
            query_person_csv=(
                args.query_person_csv
            ),
            learning_person_csv=(
                args.learning_person_csv
            ),
            output_csv=(
                args.output_csv
            ),
            missing_only=(
                args.missing_only
            ),
        )

    else:
        compare_window_conditions(
            person_csv=(
                args.person_csv
            ),
            method=(
                args.method
            ),
            output_csv=(
                args.output_csv
            ),
            missing_only=(
                args.missing_only
            ),
        )