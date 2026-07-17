from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


QUERY_PERSON_FILE = "person_year_predictions.csv"
LEARNING_PERSON_FILE = "learning_person_predictions.csv"
MANIFEST_FILE = "kg_manifest.csv"

POPULATIONS = (
    (
        "full",
        "Full Population",
        "All people in the evaluated graph.",
    ),
    (
        "retained",
        "Retained `hasAge` Relations",
        "Only people whose `hasAge` triple remains present in the evaluated run.",
    ),
    (
        "removed",
        "Removed `hasAge` Relations",
        "Only people whose `hasAge` triple was removed from the evaluated run.",
    ),
)

STRUCTURE_NAMES = {
    "with_windows": "Windowed",
    "without_windows": "Windowless",
}

METHOD_NAMES = {
    "query_point": "Query Point",
    "learned_regression": "Learned Regression",
    "tie": "Tie",
}


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    dataframe = pd.read_csv(path)
    if dataframe.empty:
        raise ValueError(f"CSV contains no rows: {path}")
    return dataframe


def parse_boolean(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if pd.isna(value):
        raise ValueError("Boolean value cannot be empty.")

    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False

    normalized = str(value).strip().lower()
    if normalized in {"true", "t", "yes", "y", "1"}:
        return True
    if normalized in {"false", "f", "no", "n", "0"}:
        return False

    raise ValueError(f"Could not interpret boolean value: {value!r}")


def read_manifest(basepath: str | Path) -> pd.DataFrame:
    path = Path(basepath) / MANIFEST_FILE
    manifest = read_csv(path)

    required = {"label", "window_condition", "removal_percent"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing)}"
        )

    if manifest["label"].duplicated().any():
        raise ValueError("Manifest contains duplicate run labels.")

    manifest = manifest.rename(
        columns={"label": "run", "relation": "removed_relation"}
    )
    manifest["run"] = manifest["run"].astype(str)
    manifest["removal_percent"] = pd.to_numeric(
        manifest["removal_percent"], errors="raise"
    )
    return manifest


def attach_manifest_metadata(
    dataframe: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    if "run" not in dataframe.columns:
        raise ValueError("Prediction data is missing the 'run' column.")

    result = dataframe.copy()
    result["run"] = result["run"].astype(str)

    metadata_columns = [
        column
        for column in (
            "window_condition",
            "removal_percent",
            "removed_relation",
            "nested_removals",
            "seed",
        )
        if column in manifest.columns and column not in result.columns
    ]

    if metadata_columns:
        result = result.merge(
            manifest[["run", *metadata_columns]],
            on="run",
            how="left",
            validate="many_to_one",
        )

    if "ground_truth_missing_in_run" not in result.columns:
        raise ValueError(
            "Prediction data is missing 'ground_truth_missing_in_run'."
        )

    result["ground_truth_missing_in_run"] = result[
        "ground_truth_missing_in_run"
    ].apply(parse_boolean)

    if "removal_percent" in result.columns:
        result["removal_percent"] = pd.to_numeric(
            result["removal_percent"], errors="raise"
        )

    return result


def validate_query_predictions(dataframe: pd.DataFrame) -> None:
    required = {
        "run",
        "person",
        "true_age",
        "ground_truth_missing_in_run",
        "top1_abs_error",
    }
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"Query-point predictions are missing: {sorted(missing)}"
        )


def validate_learning_predictions(dataframe: pd.DataFrame) -> None:
    required = {
        "run",
        "person",
        "true_age",
        "ground_truth_missing_in_run",
        "abs_error",
    }
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"Learned-regression predictions are missing: {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# Population handling
# ---------------------------------------------------------------------------


def select_population(
    dataframe: pd.DataFrame,
    population: str,
) -> pd.DataFrame:
    if population == "full":
        return dataframe.copy()
    if population == "retained":
        return dataframe.loc[
            ~dataframe["ground_truth_missing_in_run"]
        ].copy()
    if population == "removed":
        return dataframe.loc[
            dataframe["ground_truth_missing_in_run"]
        ].copy()

    raise ValueError(f"Unknown population: {population!r}")


def single_value(group: pd.DataFrame, column: str, context: str):
    values = group[column].dropna().drop_duplicates().tolist()
    if len(values) > 1:
        raise ValueError(
            f"Conflicting values for {column!r} in {context}: {values}"
        )
    return values[0] if values else None


def sort_results(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe

    sort_columns = [
        column
        for column in ("removal_percent", "window_condition", "run")
        if column in dataframe.columns
    ]
    return dataframe.sort_values(sort_columns).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Descriptive summaries
# ---------------------------------------------------------------------------


def summarize_prediction_runs(
    dataframe: pd.DataFrame,
    error_column: str,
) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame()

    rows: list[dict] = []

    for run, group in dataframe.groupby("run", sort=False, dropna=False):
        errors = pd.to_numeric(group[error_column], errors="raise")
        best_index = errors.idxmin()
        worst_index = errors.idxmax()

        row = {
            "run": run,
            "window_condition": single_value(
                group, "window_condition", f"run {run!r}"
            ),
            "removal_percent": single_value(
                group, "removal_percent", f"run {run!r}"
            ),
            "n_cases": len(group),
            "mean_abs_error": errors.mean(),
            "median_abs_error": errors.median(),
            "std_abs_error": errors.std(),
            "best_person": group.loc[best_index, "person"],
            "best_abs_error": errors.loc[best_index],
            "worst_person": group.loc[worst_index, "person"],
            "worst_abs_error": errors.loc[worst_index],
        }
        rows.append(row)

    return sort_results(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------


def bootstrap_mean_ci(
    differences: np.ndarray,
    seed: int = 42,
    n_bootstrap: int = 5000,
) -> tuple[float, float]:
    if len(differences) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0,
        len(differences),
        size=(n_bootstrap, len(differences)),
    )
    means = differences[sample_indices].mean(axis=1)
    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def paired_statistics(
    first: pd.Series,
    second: pd.Series,
    seed: int = 42,
) -> dict:
    first_values = pd.to_numeric(first, errors="raise").to_numpy(float)
    second_values = pd.to_numeric(second, errors="raise").to_numpy(float)

    if len(first_values) != len(second_values):
        raise ValueError("Paired arrays have unequal lengths.")

    differences = first_values - second_values
    n_pairs = len(differences)

    if n_pairs >= 2:
        paired_t_p_value = float(
            ttest_rel(
                first_values,
                second_values,
                nan_policy="raise",
            ).pvalue
        )
    else:
        paired_t_p_value = np.nan

    if n_pairs >= 1 and not np.allclose(differences, 0):
        try:
            wilcoxon_p_value = float(wilcoxon(differences).pvalue)
        except ValueError:
            wilcoxon_p_value = np.nan
    else:
        wilcoxon_p_value = 1.0 if n_pairs >= 1 else np.nan

    difference_std = (
        float(np.std(differences, ddof=1))
        if n_pairs >= 2
        else np.nan
    )
    cohen_dz = (
        float(np.mean(differences) / difference_std)
        if n_pairs >= 2 and difference_std != 0
        else np.nan
    )

    ci_low, ci_high = bootstrap_mean_ci(differences, seed=seed)

    return {
        "n_pairs": n_pairs,
        "mean_difference": (
            float(np.mean(differences)) if n_pairs else np.nan
        ),
        "paired_t_p_value": paired_t_p_value,
        "wilcoxon_p_value": wilcoxon_p_value,
        "cohen_dz": cohen_dz,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def compare_recovery_methods(
    query: pd.DataFrame,
    learning: pd.DataFrame,
    population: str,
    seed: int,
) -> pd.DataFrame:
    query_columns = [
        "run",
        "person",
        "true_age",
        "ground_truth_missing_in_run",
        "top1_abs_error",
        "window_condition",
        "removal_percent",
    ]
    learning_columns = [
        "run",
        "person",
        "true_age",
        "ground_truth_missing_in_run",
        "abs_error",
    ]

    merged = query[query_columns].merge(
        learning[learning_columns],
        on=["run", "person", "true_age"],
        how="inner",
        suffixes=("_query", "_learning"),
        validate="one_to_one",
    )

    query_missing = merged["ground_truth_missing_in_run_query"]
    learning_missing = merged["ground_truth_missing_in_run_learning"]
    if not (query_missing == learning_missing).all():
        raise ValueError(
            "Query-point and learned-regression predictions disagree "
            "about which hasAge relations are missing."
        )

    merged["ground_truth_missing_in_run"] = query_missing
    merged = select_population(merged, population)

    if merged.empty:
        return pd.DataFrame()

    rows: list[dict] = []

    for run, group in merged.groupby("run", sort=False, dropna=False):
        stats = paired_statistics(
            group["top1_abs_error"],
            group["abs_error"],
            seed=seed,
        )

        query_mean = group["top1_abs_error"].mean()
        learning_mean = group["abs_error"].mean()

        if np.isclose(query_mean, learning_mean):
            better = "tie"
        elif query_mean < learning_mean:
            better = "query_point"
        else:
            better = "learned_regression"

        rows.append(
            {
                "run": run,
                "window_condition": single_value(
                    group, "window_condition", f"run {run!r}"
                ),
                "removal_percent": single_value(
                    group, "removal_percent", f"run {run!r}"
                ),
                "n_pairs": stats["n_pairs"],
                "query_mae": query_mean,
                "regression_mae": learning_mean,
                "mae_difference_query_minus_regression": stats[
                    "mean_difference"
                ],
                "lower_mae_method": better,
                "paired_t_p_value": stats["paired_t_p_value"],
                "wilcoxon_p_value": stats["wilcoxon_p_value"],
                "cohen_dz": stats["cohen_dz"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
            }
        )

    return sort_results(pd.DataFrame(rows))


def compare_window_structures(
    dataframe: pd.DataFrame,
    error_column: str,
    method_name: str,
    population: str,
    seed: int,
) -> pd.DataFrame:
    selected = select_population(dataframe, population)
    if selected.empty:
        return pd.DataFrame()

    windowed = selected.loc[
        selected["window_condition"] == "with_windows",
        [
            "person",
            "true_age",
            "removal_percent",
            "ground_truth_missing_in_run",
            error_column,
        ],
    ].rename(
        columns={
            error_column: "windowed_abs_error",
            "ground_truth_missing_in_run": "missing_windowed",
        }
    )

    windowless = selected.loc[
        selected["window_condition"] == "without_windows",
        [
            "person",
            "true_age",
            "removal_percent",
            "ground_truth_missing_in_run",
            error_column,
        ],
    ].rename(
        columns={
            error_column: "windowless_abs_error",
            "ground_truth_missing_in_run": "missing_windowless",
        }
    )

    paired = windowed.merge(
        windowless,
        on=["person", "true_age", "removal_percent"],
        how="inner",
        validate="one_to_one",
    )

    if paired.empty:
        return pd.DataFrame()

    if not (
        paired["missing_windowed"] == paired["missing_windowless"]
    ).all():
        raise ValueError(
            "Windowed and windowless runs do not share the same removed "
            "hasAge relations."
        )

    rows: list[dict] = []

    for removal_percent, group in paired.groupby(
        "removal_percent", sort=True
    ):
        stats = paired_statistics(
            group["windowed_abs_error"],
            group["windowless_abs_error"],
            seed=seed,
        )

        windowed_mean = group["windowed_abs_error"].mean()
        windowless_mean = group["windowless_abs_error"].mean()

        if np.isclose(windowed_mean, windowless_mean):
            better = "tie"
        elif windowed_mean < windowless_mean:
            better = "with_windows"
        else:
            better = "without_windows"

        rows.append(
            {
                "method": method_name,
                "removal_percent": removal_percent,
                "n_pairs": stats["n_pairs"],
                "windowed_mae": windowed_mean,
                "windowless_mae": windowless_mean,
                "mae_difference_windowed_minus_windowless": stats[
                    "mean_difference"
                ],
                "lower_mae_structure": better,
                "paired_t_p_value": stats["paired_t_p_value"],
                "wilcoxon_p_value": stats["wilcoxon_p_value"],
                "cohen_dz": stats["cohen_dz"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
            }
        )

    return pd.DataFrame(rows).sort_values("removal_percent").reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------


def format_percent(value) -> str:
    if pd.isna(value):
        return "—"
    numeric = float(value)
    if numeric.is_integer():
        return f"{int(numeric)}%"
    return f"{numeric:g}%"


def format_error(value) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.3f}"


def format_effect(value) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.3f}"


def format_p_value(value) -> str:
    if pd.isna(value):
        return "—"

    numeric = float(value)
    if numeric < 0.0001:
        return "<0.0001"
    return f"{numeric:.4f}"


def format_ci(low, high) -> str:
    if pd.isna(low) or pd.isna(high):
        return "—"
    return f"[{float(low):.3f}, {float(high):.3f}]"


def structure_name(value) -> str:
    return STRUCTURE_NAMES.get(str(value), str(value))


def method_name(value) -> str:
    return METHOD_NAMES.get(str(value), str(value))


def structure_result_name(value) -> str:
    mapping = {
        "with_windows": "Windowed",
        "without_windows": "Windowless",
        "tie": "Tie",
    }
    return mapping.get(str(value), str(value))


def summary_markdown(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "_No cases are available for this population._\n"

    display = pd.DataFrame(
        {
            "Structure": dataframe["window_condition"].map(structure_name),
            "Removed": dataframe["removal_percent"].map(format_percent),
            "N": dataframe["n_cases"].astype(int),
            "MAE (years)": dataframe["mean_abs_error"].map(format_error),
            "Median AE": dataframe["median_abs_error"].map(format_error),
            "SD of AE": dataframe["std_abs_error"].map(format_error),
            "Best case": [
                f"{person} ({format_error(error)})"
                for person, error in zip(
                    dataframe["best_person"],
                    dataframe["best_abs_error"],
                )
            ],
            "Worst case": [
                f"{person} ({format_error(error)})"
                for person, error in zip(
                    dataframe["worst_person"],
                    dataframe["worst_abs_error"],
                )
            ],
        }
    )
    return display.to_markdown(index=False) + "\n"


def method_comparison_markdown(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "_No cases are available for this population._\n"

    display = pd.DataFrame(
        {
            "Structure": dataframe["window_condition"].map(structure_name),
            "Removed": dataframe["removal_percent"].map(format_percent),
            "N": dataframe["n_pairs"].astype(int),
            "Query MAE": dataframe["query_mae"].map(format_error),
            "Regression MAE": dataframe["regression_mae"].map(format_error),
            "ΔMAE (Q − R)": dataframe[
                "mae_difference_query_minus_regression"
            ].map(format_error),
            "Lower MAE": dataframe["lower_mae_method"].map(method_name),
            "Paired t p": dataframe["paired_t_p_value"].map(format_p_value),
            "Wilcoxon p": dataframe["wilcoxon_p_value"].map(format_p_value),
            "Cohen's dz": dataframe["cohen_dz"].map(format_effect),
            "95% bootstrap CI": [
                format_ci(low, high)
                for low, high in zip(
                    dataframe["ci_low"], dataframe["ci_high"]
                )
            ],
        }
    )
    return display.to_markdown(index=False) + "\n"


def window_comparison_markdown(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "_No cases are available for this population._\n"

    display = pd.DataFrame(
        {
            "Removed": dataframe["removal_percent"].map(format_percent),
            "N": dataframe["n_pairs"].astype(int),
            "Windowed MAE": dataframe["windowed_mae"].map(format_error),
            "Windowless MAE": dataframe["windowless_mae"].map(format_error),
            "ΔMAE (W − WL)": dataframe[
                "mae_difference_windowed_minus_windowless"
            ].map(format_error),
            "Lower MAE": dataframe["lower_mae_structure"].map(
                structure_result_name
            ),
            "Paired t p": dataframe["paired_t_p_value"].map(format_p_value),
            "Wilcoxon p": dataframe["wilcoxon_p_value"].map(format_p_value),
            "Cohen's dz": dataframe["cohen_dz"].map(format_effect),
            "95% bootstrap CI": [
                format_ci(low, high)
                for low, high in zip(
                    dataframe["ci_low"], dataframe["ci_high"]
                )
            ],
        }
    )
    return display.to_markdown(index=False) + "\n"


def conditions_markdown(
    query: pd.DataFrame,
    manifest: pd.DataFrame,
) -> str:
    counts = (
        query.groupby("run", sort=False)
        .agg(
            total_people=("person", "size"),
            removed_relations=("ground_truth_missing_in_run", "sum"),
        )
        .reset_index()
    )
    counts["retained_relations"] = (
        counts["total_people"] - counts["removed_relations"]
    )

    table = manifest.merge(counts, on="run", how="left")
    table = sort_results(table)

    display = pd.DataFrame(
        {
            "Structure": table["window_condition"].map(structure_name),
            "Removed": table["removal_percent"].map(format_percent),
            "Total people": table["total_people"].astype(int),
            "Retained `hasAge`": table["retained_relations"].astype(int),
            "Removed `hasAge`": table["removed_relations"].astype(int),
        }
    )
    return display.to_markdown(index=False) + "\n"


def write_population_sections(
    file,
    results: dict[str, pd.DataFrame],
    heading_level: int,
) -> None:
    marker = "#" * heading_level
    for key, title, description in POPULATIONS:
        file.write(f"\n{marker} {title}\n\n")
        file.write(description + "\n\n")
        file.write(summary_markdown(results[key]))


def generate_report(
    basepath: str | Path,
    output_md: str | Path,
    seed: int = 42,
) -> None:
    basepath = Path(basepath)
    output_md = Path(output_md)

    manifest = read_manifest(basepath)
    query = attach_manifest_metadata(
        read_csv(basepath / QUERY_PERSON_FILE), manifest
    )
    learning = attach_manifest_metadata(
        read_csv(basepath / LEARNING_PERSON_FILE), manifest
    )

    validate_query_predictions(query)
    validate_learning_predictions(learning)

    query_summaries = {
        key: summarize_prediction_runs(
            select_population(query, key), "top1_abs_error"
        )
        for key, _, _ in POPULATIONS
    }
    learning_summaries = {
        key: summarize_prediction_runs(
            select_population(learning, key), "abs_error"
        )
        for key, _, _ in POPULATIONS
    }
    method_comparisons = {
        key: compare_recovery_methods(
            query,
            learning,
            population=key,
            seed=seed,
        )
        for key, _, _ in POPULATIONS
    }
    query_window_comparisons = {
        key: compare_window_structures(
            query,
            error_column="top1_abs_error",
            method_name="query_point",
            population=key,
            seed=seed,
        )
        for key, _, _ in POPULATIONS
    }
    learning_window_comparisons = {
        key: compare_window_structures(
            learning,
            error_column="abs_error",
            method_name="learned_regression",
            population=key,
            seed=seed,
        )
        for key, _, _ in POPULATIONS
    }

    output_md.parent.mkdir(parents=True, exist_ok=True)

    with output_md.open("w", encoding="utf-8") as file:
        file.write("# MuRE Numeric-Literal Preservation Experiment\n\n")
        file.write(
            "This report summarizes age-recovery performance across paired "
            "windowed and windowless knowledge graphs. The same `hasAge` "
            "relations are removed from both structural conditions at each "
            "removal level.\n"
        )

        file.write("\n## Population Definitions\n\n")
        file.write(
            "- **Full Population:** all people in the evaluated graph.\n"
            "- **Retained `hasAge` Relations:** only people whose `hasAge` "
            "triple remains present in that run.\n"
            "- **Removed `hasAge` Relations:** only people whose `hasAge` "
            "triple was removed from that run.\n\n"
            "Every person has a `hasAge` relation in the original 0% graph. "
            "Here, *retained* and *removed* refer to whether that relation is "
            "present in the specific evaluated run.\n"
        )

        file.write("\n## Experimental Conditions\n\n")
        file.write(conditions_markdown(query, manifest))

        file.write("\n## Query-Point Recovery\n")
        write_population_sections(
            file,
            query_summaries,
            heading_level=3,
        )

        file.write("\n## Learned-Regression Recovery\n")
        write_population_sections(
            file,
            learning_summaries,
            heading_level=3,
        )

        file.write("\n## Recovery-Method Comparison\n\n")
        file.write(
            "`ΔMAE (Q − R)` is query-point MAE minus learned-regression "
            "MAE. Positive values favor learned regression; negative values "
            "favor query-point recovery.\n"
        )
        for key, title, description in POPULATIONS:
            file.write(f"\n### {title}\n\n")
            file.write(description + "\n\n")
            file.write(method_comparison_markdown(method_comparisons[key]))

        file.write("\n## Effect of Window Structure\n\n")
        file.write(
            "`ΔMAE (W − WL)` is windowed MAE minus windowless MAE. "
            "Positive values favor the windowless graph; negative values "
            "favor the windowed graph.\n"
        )

        file.write("\n### Query-Point Recovery\n")
        for key, title, description in POPULATIONS:
            file.write(f"\n#### {title}\n\n")
            file.write(description + "\n\n")
            file.write(
                window_comparison_markdown(
                    query_window_comparisons[key]
                )
            )

        file.write("\n### Learned-Regression Recovery\n")
        for key, title, description in POPULATIONS:
            file.write(f"\n#### {title}\n\n")
            file.write(description + "\n\n")
            file.write(
                window_comparison_markdown(
                    learning_window_comparisons[key]
                )
            )

        file.write("\n## Metric Notes\n\n")
        file.write(
            "- **AE:** absolute error, measured in years.\n"
            "- **MAE:** mean absolute error. Lower values indicate more "
            "accurate predictions.\n"
            "- **SD:** standard deviation of the absolute errors.\n"
            "- **N:** number of people or paired observations in the table row.\n"
            "- **Paired t p:** p-value from the paired t-test.\n"
            "- **Wilcoxon p:** p-value from the Wilcoxon signed-rank test.\n"
            "- **Cohen's dz:** standardized effect size for paired data.\n"
            "- **95% bootstrap CI:** bootstrap confidence interval for the "
            "mean paired difference.\n"
            "- **Top-1:** the nearest age entity selected by query-point "
            "recovery.\n"
        )

        file.write("\n## Nested Removal\n\n")
        file.write(
            "Removal sets are cumulative. If `R15` is the set of `hasAge` "
            "relations removed at 15%, then:\n\n"
            "```text\n"
            "R15 ⊆ R30 ⊆ R45 ⊆ R60 ⊆ R75\n"
            "```\n\n"
            "Equivalently, the complete triple sets of the evaluated graphs "
            "shrink in the opposite direction:\n\n"
            "```text\n"
            "KG0 ⊇ KG15 ⊇ KG30 ⊇ KG45 ⊇ KG60 ⊇ KG75\n"
            "```\n"
        )

    print(f"Saved Markdown report to: {output_md}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a research-focused Markdown report from per-person "
            "query-point and learned-regression predictions."
        )
    )
    parser.add_argument("--basepath", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    output = args.output_md or Path(args.basepath) / "results_report.md"
    generate_report(
        basepath=args.basepath,
        output_md=output,
        seed=args.seed,
    )
