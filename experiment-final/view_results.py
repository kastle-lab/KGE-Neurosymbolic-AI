from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


QUERY_FILE = "person_year_predictions.csv"
LEARNING_FILE = "learning_person_predictions.csv"
MANIFEST_FILE = "kg_manifest.csv"
DEFAULT_OUTPUT = Path("final_results") / "qp-lr-compare.md"

WINDOW_LABELS = {
    "with_windows": "With Windowing",
    "without_windows": "Without Windowing",
}

def write_comparison_csv(
    comparison: pd.DataFrame,
    output_csv: str | Path = Path("final_results") / "qp-lr-compare.csv",
) -> None:
    """Write the windowed and windowless results to one CSV file."""

    results = comparison.copy()

    window_order = {
        "with_windows": 0,
        "without_windows": 1,
    }

    results["_window_order"] = results["window_condition"].map(window_order)

    results = results.sort_values(
        ["_window_order", "population_size", "removal_percent"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    csv_table = pd.DataFrame(
        {
            "Windowing": results["window_condition"].map(WINDOW_LABELS),
            "Population Size": results["population_size"].astype(int),
            "Removal %": results["removal_percent"].map(format_percent),
            "QP MAE/SD (years)": [
                f"{format_number(mae)}, {format_number(sd)}"
                for mae, sd in zip(
                    results["query_mae"],
                    results["query_sd"],
                )
            ],
            "Learned Regression MAE/SD (years)": [
                f"{format_number(mae)}, {format_number(sd)}"
                for mae, sd in zip(
                    results["learning_mae"],
                    results["learning_sd"],
                )
            ],
        }
    )

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_table.to_csv(output_csv, index=False)

    print(f"Saved CSV report to: {output_csv}")

def read_csv(path: Path) -> pd.DataFrame:
    """Read a non-empty CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"CSV contains no rows: {path}")

    return df


def as_bool(value) -> bool:
    """Convert common CSV boolean representations to bool."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if pd.isna(value):
        raise ValueError("Boolean value cannot be empty.")

    text = str(value).strip().lower()

    if text in {"true", "t", "yes", "y", "1", "1.0"}:
        return True

    if text in {"false", "f", "no", "n", "0", "0.0"}:
        return False

    raise ValueError(f"Could not interpret boolean value: {value!r}")


def read_manifest(basepath: Path) -> pd.DataFrame:
    """Load and validate the experiment manifest."""
    manifest = read_csv(basepath / MANIFEST_FILE)

    required = {"label", "window_condition", "removal_percent"}
    missing = required - set(manifest.columns)

    if missing:
        raise ValueError(
            f"{basepath / MANIFEST_FILE} is missing columns: {sorted(missing)}"
        )

    manifest = manifest.rename(columns={"label": "run"}).copy()
    manifest["run"] = manifest["run"].astype(str)
    manifest["removal_percent"] = pd.to_numeric(
        manifest["removal_percent"],
        errors="raise",
    )

    duplicate_runs = manifest["run"].duplicated(keep=False)
    if duplicate_runs.any():
        runs = sorted(manifest.loc[duplicate_runs, "run"].unique())
        raise ValueError(f"Manifest contains duplicate run labels: {runs}")

    return manifest


def attach_metadata(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    """Attach window and removal metadata to prediction rows."""
    if "run" not in predictions.columns:
        raise ValueError(f"{source_name} is missing the 'run' column.")

    predictions = predictions.copy()
    predictions["run"] = predictions["run"].astype(str)

    metadata_columns = ["run", "window_condition", "removal_percent"]

    # If metadata already exists in the prediction file, remove it before
    # merging so the manifest remains the single source of truth.
    predictions = predictions.drop(
        columns=[
            column
            for column in ("window_condition", "removal_percent")
            if column in predictions.columns
        ]
    )

    predictions = predictions.merge(
        manifest[metadata_columns],
        on="run",
        how="left",
        validate="many_to_one",
    )

    missing_metadata = predictions[
        ["window_condition", "removal_percent"]
    ].isna().any(axis=1)

    if missing_metadata.any():
        unmatched_runs = sorted(
            predictions.loc[missing_metadata, "run"].astype(str).unique()
        )
        raise ValueError(
            f"{source_name} contains runs not found in the manifest: "
            f"{unmatched_runs}"
        )

    predictions["removal_percent"] = pd.to_numeric(
        predictions["removal_percent"],
        errors="raise",
    )

    return predictions


def determine_population_size(
    query: pd.DataFrame,
    learning: pd.DataFrame,
    basepath: Path,
) -> int:
    """Determine and validate the number of unique people per experiment run."""
    required = {"run", "person"}

    for name, df in (
        (QUERY_FILE, query),
        (LEARNING_FILE, learning),
    ):
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{basepath / name} is missing columns: {sorted(missing)}"
            )

    query_sizes = query.groupby("run")["person"].nunique()
    learning_sizes = learning.groupby("run")["person"].nunique()

    if query_sizes.nunique() != 1:
        raise ValueError(
            f"Query runs in {basepath} have inconsistent population sizes: "
            f"{query_sizes.to_dict()}"
        )

    if learning_sizes.nunique() != 1:
        raise ValueError(
            f"Learned-regression runs in {basepath} have inconsistent "
            f"population sizes: {learning_sizes.to_dict()}"
        )

    query_size = int(query_sizes.iloc[0])
    learning_size = int(learning_sizes.iloc[0])

    if query_size != learning_size:
        raise ValueError(
            f"Query and learned-regression population sizes differ in "
            f"{basepath}: {query_size} vs {learning_size}"
        )

    return query_size


def load_experiment(basepath: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one experiment directory."""
    basepath = Path(basepath)

    manifest = read_manifest(basepath)

    query = attach_metadata(
        read_csv(basepath / QUERY_FILE),
        manifest,
        QUERY_FILE,
    )

    learning = attach_metadata(
        read_csv(basepath / LEARNING_FILE),
        manifest,
        LEARNING_FILE,
    )

    required_query = {
        "run",
        "person",
        "top1_abs_error",
        "window_condition",
        "removal_percent",
    }
    required_learning = {
        "run",
        "person",
        "abs_error",
        "window_condition",
        "removal_percent",
    }

    missing_query = required_query - set(query.columns)
    missing_learning = required_learning - set(learning.columns)

    if missing_query:
        raise ValueError(
            f"{basepath / QUERY_FILE} is missing columns: "
            f"{sorted(missing_query)}"
        )

    if missing_learning:
        raise ValueError(
            f"{basepath / LEARNING_FILE} is missing columns: "
            f"{sorted(missing_learning)}"
        )

    population_size = determine_population_size(query, learning, basepath)

    query["population_size"] = population_size
    learning["population_size"] = population_size

    query["top1_abs_error"] = pd.to_numeric(
        query["top1_abs_error"],
        errors="raise",
    )
    learning["abs_error"] = pd.to_numeric(
        learning["abs_error"],
        errors="raise",
    )

    return query, learning


def aggregate_metrics(
    df: pd.DataFrame,
    error_column: str,
    prefix: str,
) -> pd.DataFrame:
    """
    Calculate mean absolute error and sample standard deviation.

    pandas GroupBy.std() uses ddof=1, so the reported SD is the sample
    standard deviation of absolute errors.
    """
    group_columns = [
        "population_size",
        "window_condition",
        "removal_percent",
    ]

    return (
        df.groupby(group_columns, as_index=False)
        .agg(
            **{
                f"{prefix}_mae": (error_column, "mean"),
                f"{prefix}_sd": (error_column, "std"),
            }
        )
    )


def build_comparison(
    query: pd.DataFrame,
    learning: pd.DataFrame,
) -> pd.DataFrame:
    """Combine query-point and learned-regression summary statistics."""
    query_metrics = aggregate_metrics(
        query,
        error_column="top1_abs_error",
        prefix="query",
    )

    learning_metrics = aggregate_metrics(
        learning,
        error_column="abs_error",
        prefix="learning",
    )

    key_columns = [
        "population_size",
        "window_condition",
        "removal_percent",
    ]

    comparison = query_metrics.merge(
        learning_metrics,
        on=key_columns,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = comparison["_merge"] != "both"
    if unmatched.any():
        mismatches = comparison.loc[unmatched, key_columns + ["_merge"]]
        raise ValueError(
            "Query-point and learned-regression conditions do not match:\n"
            f"{mismatches.to_string(index=False)}"
        )

    comparison = comparison.drop(columns="_merge")

    unexpected_conditions = (
        set(comparison["window_condition"]) - set(WINDOW_LABELS)
    )
    if unexpected_conditions:
        raise ValueError(
            "Unexpected window_condition values: "
            f"{sorted(unexpected_conditions)}"
        )

    return comparison.sort_values(
        ["window_condition", "population_size", "removal_percent"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def format_number(value, digits: int = 3) -> str:
    """Format numeric report values."""
    if pd.isna(value):
        return "—"

    return f"{float(value):.{digits}f}"


def format_percent(value) -> str:
    """Format removal percentage."""
    if pd.isna(value):
        return "—"

    return f"{float(value):g}%"


def markdown_table(df: pd.DataFrame) -> str:
    """Convert a DataFrame to Markdown."""
    if df.empty:
        return "_No matching cases are available._\n"

    return df.to_markdown(index=False) + "\n"


def make_output_table(results: pd.DataFrame) -> pd.DataFrame:
    """
    Create the final four-column table.

    Each method cell is formatted as:
        MAE, SD
    """
    results = results.sort_values(
        ["population_size", "removal_percent"],
        ascending=[False, True],
    ).reset_index(drop=True)

    query_mae_sd = [
        f"{format_number(mae)}, {format_number(sd)}"
        for mae, sd in zip(results["query_mae"], results["query_sd"])
    ]

    learning_mae_sd = [
        f"{format_number(mae)}, {format_number(sd)}"
        for mae, sd in zip(results["learning_mae"], results["learning_sd"])
    ]

    return pd.DataFrame(
        {
            "Population Size": results["population_size"].astype(int),
            "Removal %": results["removal_percent"].map(format_percent),
            "QP MAE/SD (years)": query_mae_sd,
            "Learned Regression MAE/SD (years)": learning_mae_sd,
        }
    )


def generate_report(
    basepaths: list[str | Path],
    output_md: str | Path = DEFAULT_OUTPUT,
) -> None:
    """Load all experiment folders and write the two final tables."""
    loaded = [load_experiment(path) for path in basepaths]

    query = pd.concat(
        [query_df for query_df, _ in loaded],
        ignore_index=True,
    )
    learning = pd.concat(
        [learning_df for _, learning_df in loaded],
        ignore_index=True,
    )

    comparison = build_comparison(query, learning)

    write_comparison_csv(comparison) 
    
    output_md = Path(output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    with output_md.open("w", encoding="utf-8") as file:
        file.write("# Query-Point and Learned-Regression Comparison\n\n")
        file.write(
            "Each method column is shown as `MAE, SD`, measured in years, "
            "for the full evaluated population.\n\n"
        )

        for condition in ("with_windows", "without_windows"):
            title = WINDOW_LABELS[condition]
            results = comparison.loc[
                comparison["window_condition"] == condition
            ].copy()

            file.write(f"## {title}\n\n")
            file.write(markdown_table(make_output_table(results)))
            file.write("\n")

    print(f"Saved final comparison report to: {output_md}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate two Markdown tables comparing query-point and "
            "learned-regression MAE and SD with and without windowing."
        )
    )

    parser.add_argument(
        "--basepath",
        nargs="+",
        required=True,
        help=(
            "One or more experiment folders containing "
            f"{QUERY_FILE}, {LEARNING_FILE}, and {MANIFEST_FILE}."
        ),
    )

    parser.add_argument(
        "--output-md",
        default=str(DEFAULT_OUTPUT),
        help=(
            "Output Markdown path. "
            f"Default: {DEFAULT_OUTPUT.as_posix()}"
        ),
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    generate_report(args.basepath, args.output_md)


if __name__ == "__main__":
    main()
