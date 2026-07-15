from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RESULT_FILES = [
    "run_year_summary.csv",
    "run_year_summary_missing_only.csv",
    "learning_run_summary.csv",
    "learning_run_summary_missing_only.csv",
    "method_comparison.csv",
    "method_comparison_missing_only.csv",
    "query_window_comparison.csv",
    "query_window_comparison_missing_only.csv",
    "learning_window_comparison.csv",
    "learning_window_comparison_missing_only.csv",
]


def read_csv_if_exists(path: str | Path):
    path = Path(path)
    if not path.exists():
        print(f"Warning: missing file: {path}")
        return None

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_manifest(basepath: str | Path):
    path = Path(basepath) / "kg_manifest.csv"
    manifest = read_csv_if_exists(path)
    if manifest is None or manifest.empty:
        return manifest

    if "label" not in manifest.columns:
        raise ValueError(f"Manifest is missing 'label': {path}")

    return manifest.rename(
        columns={"label": "run", "relation": "removed_relation"}
    )


def attach_manifest_metadata(df, manifest):
    if df is None or df.empty or manifest is None or manifest.empty:
        return df
    if "run" not in df.columns:
        return df

    metadata = [
        column
        for column in (
            "window_condition",
            "removal_percent",
            "removed_relation",
            "nested_removals",
            "seed",
            "tsv_path",
        )
        if column in manifest.columns and column not in df.columns
    ]

    if not metadata:
        return df

    return df.merge(
        manifest[["run", *metadata]],
        on="run",
        how="left",
        validate="many_to_one",
    )


def sort_runs(df):
    if df is None or df.empty:
        return df

    result = df.copy()
    sort_columns = [
        column
        for column in ("removal_percent", "window_condition", "run")
        if column in result.columns
    ]
    return (
        result.sort_values(sort_columns).reset_index(drop=True)
        if sort_columns
        else result
    )


def df_to_md(df, columns=None, round_digits=4):
    if df is None or df.empty:
        return "_No data available._\n"

    output = df.copy()
    if columns is not None:
        available = [column for column in columns if column in output.columns]
        if available:
            output = output[available]

    numeric = output.select_dtypes(include="number").columns
    output[numeric] = output[numeric].round(round_digits)
    return output.to_markdown(index=False) + "\n"


def write_section(file, title, df, columns=None, description=None):
    file.write(f"\n## {title}\n\n")
    if description:
        file.write(description.strip() + "\n\n")
    file.write(df_to_md(df, columns=columns))


def generate_report(basepath, output_md):
    basepath = Path(basepath)
    output_md = Path(output_md)
    manifest = read_manifest(basepath)

    data = {
        filename: read_csv_if_exists(basepath / filename)
        for filename in RESULT_FILES
    }

    for filename in (
        "run_year_summary.csv",
        "run_year_summary_missing_only.csv",
        "learning_run_summary.csv",
        "learning_run_summary_missing_only.csv",
        "method_comparison.csv",
        "method_comparison_missing_only.csv",
    ):
        data[filename] = sort_runs(
            attach_manifest_metadata(data[filename], manifest)
        )

    query_columns = [
        "run",
        "window_condition",
        "removal_percent",
        "n_people",
        "n_missing_people",
        "mean_top1_abs_error",
        "median_top1_abs_error",
        "std_top1_abs_error",
        "best_person",
        "best_abs_error",
        "worst_person",
        "worst_abs_error",
    ]

    learning_columns = [
        "run",
        "window_condition",
        "removal_percent",
        "n_people",
        "n_missing_people",
        "mean_abs_error",
        "median_abs_error",
        "std_abs_error",
        "best_person",
        "best_abs_error",
        "worst_person",
        "worst_abs_error",
    ]

    method_columns = [
        "run",
        "window_condition",
        "removal_percent",
        "n_pairs",
        "missing_only",
        "mean_query_abs_error",
        "mean_learning_abs_error",
        "mean_error_difference_query_minus_learning",
        "better_method_by_mean",
        "paired_t_p_value",
        "wilcoxon_p_value",
        "cohen_dz_query_minus_learning",
        "bootstrap_mean_diff_ci_low",
        "bootstrap_mean_diff_ci_high",
    ]

    window_columns = [
        "method",
        "removal_percent",
        "n_pairs",
        "missing_only",
        "mean_with_windows_abs_error",
        "mean_without_windows_abs_error",
        "mean_error_difference_with_minus_without",
        "better_window_condition_by_mean",
        "paired_t_p_value",
        "wilcoxon_p_value",
        "cohen_dz_with_minus_without",
        "bootstrap_mean_diff_ci_low",
        "bootstrap_mean_diff_ci_high",
    ]

    output_md.parent.mkdir(parents=True, exist_ok=True)

    with output_md.open("w", encoding="utf-8") as file:
        file.write("# Knowledge Graph Windowing and Age-Removal Experiment\n\n")
        file.write(f"Base path: `{basepath}`\n\n")
        file.write(
            "The experiment begins with one KG containing window structure. "
            "A paired windowless KG is derived by removing only window-related "
            "triples. The same randomly selected `hasAge` triples are then "
            "removed from both conditions at every percentage, allowing a "
            "controlled with-windows versus without-windows comparison.\n"
        )

        file.write("\n## Files Used\n\n")
        file.write(
            f"- `kg_manifest.csv` — "
            f"{'found' if (basepath / 'kg_manifest.csv').exists() else 'missing'}\n"
        )
        file.write(
            f"- `hasAge_removal_plan.csv` — "
            f"{'found' if (basepath / 'hasAge_removal_plan.csv').exists() else 'missing'}\n"
        )
        for filename in RESULT_FILES:
            file.write(
                f"- `{filename}` — "
                f"{'found' if (basepath / filename).exists() else 'missing'}\n"
            )

        write_section(
            file,
            "Experiment Runs",
            sort_runs(manifest),
            columns=[
                "run",
                "window_condition",
                "removal_percent",
                "removed_relation",
                "nested_removals",
                "seed",
                "tsv_path",
            ],
        )

        write_section(
            file,
            "Query-Point Method — All People",
            data["run_year_summary.csv"],
            query_columns,
        )
        write_section(
            file,
            "Query-Point Method — Missing Relations Only",
            data["run_year_summary_missing_only.csv"],
            query_columns,
        )
        write_section(
            file,
            "Learned Regression Method — All People",
            data["learning_run_summary.csv"],
            learning_columns,
        )
        write_section(
            file,
            "Learned Regression Method — Missing Relations Only",
            data["learning_run_summary_missing_only.csv"],
            learning_columns,
        )
        write_section(
            file,
            "Query Point vs Learned Regression — All People",
            data["method_comparison.csv"],
            method_columns,
        )
        write_section(
            file,
            "Query Point vs Learned Regression — Missing Relations Only",
            data["method_comparison_missing_only.csv"],
            method_columns,
        )
        write_section(
            file,
            "Window Comparison — Query Point, All People",
            data["query_window_comparison.csv"],
            window_columns,
        )
        write_section(
            file,
            "Window Comparison — Query Point, Missing Relations Only",
            data["query_window_comparison_missing_only.csv"],
            window_columns,
        )
        write_section(
            file,
            "Window Comparison — Learned Regression, All People",
            data["learning_window_comparison.csv"],
            window_columns,
        )
        write_section(
            file,
            "Window Comparison — Learned Regression, Missing Relations Only",
            data["learning_window_comparison_missing_only.csv"],
            window_columns,
        )

        file.write("\n## Interpretation Notes\n\n")
        file.write(
            "- Positive `mean_error_difference_query_minus_learning` means "
            "query-point error was larger, favoring learned regression.\n"
        )
        file.write(
            "- Positive `mean_error_difference_with_minus_without` means the "
            "with-windows condition had larger error, favoring the windowless "
            "condition. Negative values favor the with-windows condition.\n"
        )
        file.write(
            "- Missing-only tables include only people whose exact `hasAge` "
            "triple was removed at that percentage.\n"
        )
        file.write(
            "- With nested removal enabled, each larger percentage includes "
            "every removal made at all smaller percentages.\n"
        )

    print(f"Saved Markdown report to: {output_md}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basepath", required=True)
    parser.add_argument("--output-md", default=None)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    output = args.output_md or Path(args.basepath) / "results_report.md"
    generate_report(args.basepath, output)
