from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import ttest_rel, wilcoxon


QUERY_FILE = "person_year_predictions.csv"
LEARNING_FILE = "learning_person_predictions.csv"
MANIFEST_FILE = "kg_manifest.csv"

DEFAULT_OUTPUT_DIRECTORY = Path("final_results")
REPORT_FILENAME = "qp-lr-method-tests.md"
CSV_FILENAME = "qp-lr-method-tests.csv"
PLOT_DIRECTORY_NAME = "qp-lr-method-plots"

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
# File loading and validation
# ============================================================

def read_csv(path: str | Path) -> pd.DataFrame:
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


def read_manifest(basepath: str | Path) -> pd.DataFrame:
    """Load one population folder's experiment manifest."""
    basepath = Path(basepath)
    manifest_path = basepath / MANIFEST_FILE

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

    duplicate_runs = manifest["run"].duplicated(
        keep=False
    )

    if duplicate_runs.any():
        duplicated = sorted(
            manifest.loc[
                duplicate_runs,
                "run",
            ].unique()
        )

        raise ValueError(
            f"Manifest contains duplicate run labels: "
            f"{duplicated}"
        )

    unknown_windows = (
        set(manifest["window_condition"])
        - set(WINDOW_LABELS)
    )

    if unknown_windows:
        raise ValueError(
            "Unexpected window_condition values in "
            f"{manifest_path}: {sorted(unknown_windows)}"
        )

    return manifest


def attach_manifest_metadata(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    source_path: Path,
) -> pd.DataFrame:
    """
    Attach window condition and removal percentage to predictions.

    The manifest is treated as the source of truth.
    """
    if "run" not in predictions.columns:
        raise ValueError(
            f"{source_path} is missing the 'run' column."
        )

    predictions = predictions.copy()

    predictions["run"] = (
        predictions["run"]
        .astype(str)
    )

    predictions = predictions.drop(
        columns=[
            column
            for column in (
                "window_condition",
                "removal_percent",
            )
            if column in predictions.columns
        ]
    )

    predictions = predictions.merge(
        manifest[
            [
                "run",
                "window_condition",
                "removal_percent",
            ]
        ],
        on="run",
        how="left",
        validate="many_to_one",
    )

    missing_metadata = predictions[
        [
            "window_condition",
            "removal_percent",
        ]
    ].isna().any(axis=1)

    if missing_metadata.any():
        unmatched_runs = sorted(
            predictions.loc[
                missing_metadata,
                "run",
            ].unique()
        )

        raise ValueError(
            f"{source_path} contains runs that are not "
            f"present in the manifest: {unmatched_runs}"
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
    """Determine the number of unique people in each run."""
    for source_name, dataframe in (
        (QUERY_FILE, query),
        (LEARNING_FILE, learning),
    ):
        required = {"run", "person"}
        missing = required - set(dataframe.columns)

        if missing:
            raise ValueError(
                f"{basepath / source_name} is missing "
                f"columns: {sorted(missing)}"
            )

    query_sizes = (
        query.groupby("run")["person"]
        .nunique()
    )

    learning_sizes = (
        learning.groupby("run")["person"]
        .nunique()
    )

    if query_sizes.nunique() != 1:
        raise ValueError(
            f"Query-point runs in {basepath} have "
            f"inconsistent population sizes: "
            f"{query_sizes.to_dict()}"
        )

    if learning_sizes.nunique() != 1:
        raise ValueError(
            f"Learned-regression runs in {basepath} have "
            f"inconsistent population sizes: "
            f"{learning_sizes.to_dict()}"
        )

    query_size = int(query_sizes.iloc[0])
    learning_size = int(learning_sizes.iloc[0])

    if query_size != learning_size:
        raise ValueError(
            f"Query-point and learned-regression population "
            f"sizes differ in {basepath}: "
            f"{query_size} versus {learning_size}"
        )

    return query_size


def load_experiment(
    basepath: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one population-size experiment folder."""
    basepath = Path(basepath)

    manifest = read_manifest(basepath)

    query_path = basepath / QUERY_FILE
    learning_path = basepath / LEARNING_FILE

    query = attach_manifest_metadata(
        predictions=read_csv(query_path),
        manifest=manifest,
        source_path=query_path,
    )

    learning = attach_manifest_metadata(
        predictions=read_csv(learning_path),
        manifest=manifest,
        source_path=learning_path,
    )

    required_query_columns = {
        "run",
        "person",
        "true_age",
        "top1_abs_error",
        "window_condition",
        "removal_percent",
    }

    required_learning_columns = {
        "run",
        "person",
        "true_age",
        "abs_error",
        "window_condition",
        "removal_percent",
    }

    missing_query = (
        required_query_columns - set(query.columns)
    )

    missing_learning = (
        required_learning_columns - set(learning.columns)
    )

    if missing_query:
        raise ValueError(
            f"{query_path} is missing columns: "
            f"{sorted(missing_query)}"
        )

    if missing_learning:
        raise ValueError(
            f"{learning_path} is missing columns: "
            f"{sorted(missing_learning)}"
        )

    population_size = determine_population_size(
        query=query,
        learning=learning,
        basepath=basepath,
    )

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


def discover_experiment_directories(
    root: str | Path = ".",
) -> list[Path]:
    """
    Discover numeric population folders such as 100/, 200/, and 500/.
    """
    root = Path(root)

    directories = []

    for path in root.iterdir():
        if not path.is_dir():
            continue

        if not path.name.isdigit():
            continue

        required_files = (
            path / QUERY_FILE,
            path / LEARNING_FILE,
            path / MANIFEST_FILE,
        )

        if all(file_path.exists() for file_path in required_files):
            directories.append(path)

    directories.sort(
        key=lambda path: int(path.name)
    )

    if not directories:
        raise ValueError(
            f"No population experiment directories were "
            f"found under {root}."
        )

    return directories


# ============================================================
# Pair QP and learned-regression errors
# ============================================================

def pair_method_errors(
    query: pd.DataFrame,
    learning: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pair query-point and learned-regression errors for the same person
    in the same trained run.
    """
    query_columns = [
        "population_size",
        "run",
        "person",
        "true_age",
        "top1_abs_error",
        "window_condition",
        "removal_percent",
    ]

    learning_columns = [
        "population_size",
        "run",
        "person",
        "true_age",
        "abs_error",
        "window_condition",
        "removal_percent",
    ]

    paired = query[query_columns].merge(
        learning[learning_columns],
        on=[
            "population_size",
            "run",
            "person",
        ],
        how="inner",
        suffixes=("_qp", "_lr"),
        validate="one_to_one",
    )

    if len(paired) != len(query) or len(paired) != len(learning):
        raise ValueError(
            "Query-point and learned-regression rows could "
            "not all be paired one-to-one."
        )

    same_window = (
        paired["window_condition_qp"]
        == paired["window_condition_lr"]
    )

    same_removal = np.isclose(
        paired["removal_percent_qp"],
        paired["removal_percent_lr"],
    )

    if not same_window.all():
        raise ValueError(
            "The two prediction files disagree about "
            "window conditions."
        )

    if not same_removal.all():
        raise ValueError(
            "The two prediction files disagree about "
            "removal percentages."
        )

    paired["window_condition"] = (
        paired["window_condition_qp"]
    )

    paired["removal_percent"] = (
        paired["removal_percent_qp"]
    )

    paired["qp_error"] = pd.to_numeric(
        paired["top1_abs_error"],
        errors="raise",
    )

    paired["lr_error"] = pd.to_numeric(
        paired["abs_error"],
        errors="raise",
    )

    paired["difference"] = (
        paired["qp_error"]
        - paired["lr_error"]
    )

    return paired


# ============================================================
# Statistical analysis
# ============================================================

def bootstrap_mean_ci(
    differences: np.ndarray,
    seed: int = 42,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for the mean paired difference.

    Difference is defined as:

        QP absolute error - LR absolute error
    """
    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        np.isfinite(differences)
    ]

    number_of_cases = len(differences)

    if number_of_cases == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    bootstrap_means = np.empty(
        n_bootstrap,
        dtype=float,
    )

    batch_size = 500

    for start in range(
        0,
        n_bootstrap,
        batch_size,
    ):
        stop = min(
            start + batch_size,
            n_bootstrap,
        )

        current_batch_size = stop - start

        indices = rng.integers(
            low=0,
            high=number_of_cases,
            size=(
                current_batch_size,
                number_of_cases,
            ),
        )

        bootstrap_means[start:stop] = (
            differences[indices]
            .mean(axis=1)
        )

    alpha = 1.0 - confidence

    lower_percentile = (
        100.0 * alpha / 2.0
    )

    upper_percentile = (
        100.0 * (1.0 - alpha / 2.0)
    )

    lower = float(
        np.percentile(
            bootstrap_means,
            lower_percentile,
        )
    )

    upper = float(
        np.percentile(
            bootstrap_means,
            upper_percentile,
        )
    )

    return lower, upper


def calculate_paired_statistics(
    qp_errors,
    lr_errors,
    seed: int = 42,
    n_bootstrap: int = 5000,
) -> dict:
    """Calculate descriptive and paired inferential statistics."""
    qp_errors = np.asarray(
        qp_errors,
        dtype=float,
    )

    lr_errors = np.asarray(
        lr_errors,
        dtype=float,
    )

    valid = (
        np.isfinite(qp_errors)
        & np.isfinite(lr_errors)
    )

    qp_errors = qp_errors[valid]
    lr_errors = lr_errors[valid]

    if len(qp_errors) != len(lr_errors):
        raise ValueError(
            "Paired arrays have unequal lengths."
        )

    number_of_pairs = len(qp_errors)

    if number_of_pairs == 0:
        return {
            "n": 0,
            "qp_mae": np.nan,
            "qp_sd": np.nan,
            "lr_mae": np.nan,
            "lr_sd": np.nan,
            "mean_difference": np.nan,
            "median_difference": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "t_statistic": np.nan,
            "t_p": np.nan,
            "wilcoxon_statistic": np.nan,
            "wilcoxon_p": np.nan,
            "cohen_dz": np.nan,
            "better_method": "No cases",
        }

    differences = qp_errors - lr_errors

    qp_mae = float(np.mean(qp_errors))
    lr_mae = float(np.mean(lr_errors))

    qp_sd = (
        float(np.std(qp_errors, ddof=1))
        if number_of_pairs >= 2
        else np.nan
    )

    lr_sd = (
        float(np.std(lr_errors, ddof=1))
        if number_of_pairs >= 2
        else np.nan
    )

    mean_difference = float(
        np.mean(differences)
    )

    median_difference = float(
        np.median(differences)
    )

    ci_low, ci_high = bootstrap_mean_ci(
        differences=differences,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )

    if (
        number_of_pairs >= 2
        and not np.allclose(differences, 0)
    ):
        t_result = ttest_rel(
            qp_errors,
            lr_errors,
            nan_policy="omit",
        )

        t_statistic = float(
            t_result.statistic
        )

        t_p = float(
            t_result.pvalue
        )
    elif np.allclose(differences, 0):
        t_statistic = 0.0
        t_p = 1.0
    else:
        t_statistic = np.nan
        t_p = np.nan

    if np.allclose(differences, 0):
        wilcoxon_statistic = 0.0
        wilcoxon_p = 1.0
    else:
        try:
            wilcoxon_result = wilcoxon(
                differences,
                alternative="two-sided",
                zero_method="wilcox",
            )

            wilcoxon_statistic = float(
                wilcoxon_result.statistic
            )

            wilcoxon_p = float(
                wilcoxon_result.pvalue
            )

        except ValueError:
            wilcoxon_statistic = np.nan
            wilcoxon_p = np.nan

    difference_sd = (
        float(np.std(differences, ddof=1))
        if number_of_pairs >= 2
        else np.nan
    )

    if (
        number_of_pairs >= 2
        and np.isfinite(difference_sd)
        and difference_sd > 0
    ):
        cohen_dz = (
            mean_difference
            / difference_sd
        )
    else:
        cohen_dz = np.nan

    if np.isclose(qp_mae, lr_mae):
        better_method = "Tie"
    elif qp_mae < lr_mae:
        better_method = "Query Point"
    else:
        better_method = "Learned Regression"

    return {
        "qp_mae": qp_mae,
        "qp_sd": qp_sd,
        "lr_mae": lr_mae,
        "lr_sd": lr_sd,
        "mean_difference": mean_difference,
        "median_difference": median_difference,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "t_statistic": t_statistic,
        "t_p": t_p,
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_p": wilcoxon_p,
        "cohen_dz": cohen_dz,
        "better_method": better_method,
    }


def summarize_comparisons(
    paired: pd.DataFrame,
    seed: int = 42,
    n_bootstrap: int = 5000,
) -> pd.DataFrame:
    """
    Calculate one QP-versus-LR comparison for every combination of:

        population size
        window condition
        removal percentage
    """
    rows = []

    group_columns = [
        "population_size",
        "window_condition",
        "removal_percent",
    ]

    grouped = paired.groupby(
        group_columns,
        sort=False,
    )

    for (
        population_size,
        window_condition,
        removal_percent,
    ), group in grouped:

        statistics = calculate_paired_statistics(
            qp_errors=group["qp_error"],
            lr_errors=group["lr_error"],
            seed=seed,
            n_bootstrap=n_bootstrap,
        )

        rows.append(
            {
                "population_size": int(
                    population_size
                ),
                "window_condition": (
                    window_condition
                ),
                "removal_percent": float(
                    removal_percent
                ),
                **statistics,
            }
        )

    summary = pd.DataFrame(rows)

    summary["_window_order"] = (
        summary["window_condition"]
        .map(WINDOW_ORDER)
    )

    summary = summary.sort_values(
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

    return summary.drop(
        columns="_window_order"
    )


# ============================================================
# Formatting
# ============================================================

def format_number(
    value,
    digits: int = 3,
) -> str:
    if pd.isna(value):
        return "—"

    return f"{float(value):.{digits}f}"


def format_p_value(value) -> str:
    if pd.isna(value):
        return "—"

    value = float(value)

    if value < 0.0001:
        return "<0.0001"

    return f"{value:.4f}"


def format_percent(value) -> str:
    if pd.isna(value):
        return "—"

    return f"{float(value):g}%"


def format_ci(low, high) -> str:
    if pd.isna(low) or pd.isna(high):
        return "—"

    return (
        f"[{format_number(low)}, "
        f"{format_number(high)}]"
    )


def markdown_escape(value) -> str:
    return (
        str(value)
        .replace("|", "\\|")
        .replace("\n", " ")
    )


def dataframe_to_markdown(
    dataframe: pd.DataFrame,
) -> str:
    """Convert a DataFrame to Markdown without requiring tabulate."""
    if dataframe.empty:
        return "_No cases are available._\n"

    headers = [
        markdown_escape(column)
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
        values = [
            markdown_escape(value)
            for value in row
        ]

        lines.append(
            "| " + " | ".join(values) + " |"
        )

    return "\n".join(lines) + "\n"


def create_markdown_table(
    summary: pd.DataFrame,
    window_condition: str,
) -> pd.DataFrame:
    """Create one formatted publication-style table."""
    results = summary.loc[
        summary["window_condition"]
        == window_condition
    ].copy()

    results = results.sort_values(
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
                results["population_size"]
                .astype(int)
            ),
            "Removal %": (
                results["removal_percent"]
                .map(format_percent)
            ),
            "QP MAE/SD (years)": [
                (
                    f"{format_number(mae)}, "
                    f"{format_number(sd)}"
                )
                for mae, sd in zip(
                    results["qp_mae"],
                    results["qp_sd"],
                )
            ],
            "LR MAE/SD (years)": [
                (
                    f"{format_number(mae)}, "
                    f"{format_number(sd)}"
                )
                for mae, sd in zip(
                    results["lr_mae"],
                    results["lr_sd"],
                )
            ],
            "ΔMAE QP−LR (years)": (
                results["mean_difference"]
                .map(format_number)
            ),
            "95% Bootstrap CI": [
                format_ci(low, high)
                for low, high in zip(
                    results["ci_low"],
                    results["ci_high"],
                )
            ],
            "Paired t p": (
                results["t_p"]
                .map(format_p_value)
            ),
            "Wilcoxon p": (
                results["wilcoxon_p"]
                .map(format_p_value)
            ),
            "Cohen's dz": (
                results["cohen_dz"]
                .map(format_number)
            ),
            "Lower MAE": (
                results["better_method"]
            ),
        }
    )


def create_csv_table(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create a machine-readable CSV table."""
    ordered = summary.copy()

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
    ).reset_index(drop=True)

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
            "QP MAE (years)": (
                ordered["qp_mae"]
            ),
            "QP SD (years)": (
                ordered["qp_sd"]
            ),
            "Learned Regression MAE (years)": (
                ordered["lr_mae"]
            ),
            "Learned Regression SD (years)": (
                ordered["lr_sd"]
            ),
            "Delta MAE QP-LR (years)": (
                ordered["mean_difference"]
            ),
            "Median Difference QP-LR (years)": (
                ordered["median_difference"]
            ),
            "95% CI Low": (
                ordered["ci_low"]
            ),
            "95% CI High": (
                ordered["ci_high"]
            ),
            "Paired t Statistic": (
                ordered["t_statistic"]
            ),
            "Paired t p": (
                ordered["t_p"]
            ),
            "Wilcoxon Statistic": (
                ordered["wilcoxon_statistic"]
            ),
            "Wilcoxon p": (
                ordered["wilcoxon_p"]
            ),
            "Cohen's dz": (
                ordered["cohen_dz"]
            ),
            "Lower MAE": (
                ordered["better_method"]
            ),
        }
    )


# ============================================================
# Plots
# ============================================================

def configure_plot(axis):
    """Apply common plot styling."""
    axis.set_facecolor(PLOT_BACKGROUND)

    axis.grid(
        True,
        linewidth=0.45,
        alpha=0.30,
    )

    for spine in axis.spines.values():
        spine.set_alpha(0.45)


def plot_mae_by_removal(
    summary: pd.DataFrame,
    window_condition: str,
    output_path: str | Path,
    dpi: int = 300,
) -> None:
    """
    Plot QP and learned-regression MAE across removal percentages.

    Each population size receives one color. QP is solid and learned
    regression is dashed.
    """
    results = summary.loc[
        summary["window_condition"]
        == window_condition
    ].copy()

    if results.empty:
        return

    population_sizes = sorted(
        results["population_size"].unique(),
        reverse=True,
    )

    default_colors = plt.rcParams[
        "axes.prop_cycle"
    ].by_key()["color"]

    figure, axis = plt.subplots(
        figsize=(11, 7),
        facecolor=PLOT_BACKGROUND,
    )

    configure_plot(axis)

    for index, population_size in enumerate(
        population_sizes
    ):
        population_results = results.loc[
            results["population_size"]
            == population_size
        ].sort_values("removal_percent")

        color = default_colors[
            index % len(default_colors)
        ]

        axis.plot(
            population_results["removal_percent"],
            population_results["qp_mae"],
            marker="o",
            linestyle="-",
            linewidth=2.0,
            markersize=5,
            color=color,
            label=f"QP — population {population_size}",
        )

        axis.plot(
            population_results["removal_percent"],
            population_results["lr_mae"],
            marker="s",
            linestyle="--",
            linewidth=2.0,
            markersize=5,
            color=color,
            label=f"LR — population {population_size}",
        )

    removal_levels = sorted(
        results["removal_percent"].unique()
    )

    axis.set_xticks(removal_levels)

    axis.set_xlabel(
        "Removed hasAge relations (%)"
    )

    axis.set_ylabel(
        "Mean absolute error (years)"
    )

    axis.set_title(
        f"QP and Learned-Regression Performance — "
        f"{WINDOW_LABELS[window_condition]}"
    )

    axis.legend(
        fontsize=8,
        ncol=2,
        frameon=True,
    )

    figure.tight_layout()

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


def plot_difference_by_removal(
    summary: pd.DataFrame,
    window_condition: str,
    output_path: str | Path,
    dpi: int = 300,
) -> None:
    """
    Plot QP MAE minus learned-regression MAE.

    Positive values favor learned regression.
    Negative values favor query point.
    """
    results = summary.loc[
        summary["window_condition"]
        == window_condition
    ].copy()

    if results.empty:
        return

    population_sizes = sorted(
        results["population_size"].unique(),
        reverse=True,
    )

    default_colors = plt.rcParams[
        "axes.prop_cycle"
    ].by_key()["color"]

    figure, axis = plt.subplots(
        figsize=(11, 7),
        facecolor=PLOT_BACKGROUND,
    )

    configure_plot(axis)

    for index, population_size in enumerate(
        population_sizes
    ):
        population_results = results.loc[
            results["population_size"]
            == population_size
        ].sort_values("removal_percent")

        color = default_colors[
            index % len(default_colors)
        ]

        differences = (
            population_results["mean_difference"]
            .to_numpy(dtype=float)
        )

        ci_low = (
            population_results["ci_low"]
            .to_numpy(dtype=float)
        )

        ci_high = (
            population_results["ci_high"]
            .to_numpy(dtype=float)
        )

        lower_error = differences - ci_low
        upper_error = ci_high - differences

        lower_error = np.maximum(
            lower_error,
            0,
        )

        upper_error = np.maximum(
            upper_error,
            0,
        )

        axis.errorbar(
            population_results["removal_percent"],
            differences,
            yerr=np.vstack(
                [
                    lower_error,
                    upper_error,
                ]
            ),
            marker="o",
            linestyle="-",
            linewidth=2.0,
            markersize=5,
            capsize=4,
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

    removal_levels = sorted(
        results["removal_percent"].unique()
    )

    axis.set_xticks(removal_levels)

    axis.set_xlabel(
        "Removed hasAge relations (%)"
    )

    axis.set_ylabel(
        "ΔMAE: QP − learned regression (years)"
    )

    axis.set_title(
        f"Relative Method Performance — "
        f"{WINDOW_LABELS[window_condition]}"
    )

    axis.legend(
        fontsize=8,
        frameon=True,
    )

    axis.text(
        0.01,
        0.98,
        "Above zero: learned regression has lower MAE\n"
        "Below zero: query point has lower MAE",
        transform=axis.transAxes,
        verticalalignment="top",
        fontsize=8,
    )

    figure.tight_layout()

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


def generate_plots(
    summary: pd.DataFrame,
    plot_directory: str | Path,
    dpi: int = 300,
) -> dict[str, Path]:
    """Generate all four comparison plots."""
    plot_directory = Path(plot_directory)

    plot_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {}

    for window_condition in (
        "with_windows",
        "without_windows",
    ):
        mae_path = (
            plot_directory
            / f"{window_condition}_mae.png"
        )

        difference_path = (
            plot_directory
            / f"{window_condition}_difference.png"
        )

        plot_mae_by_removal(
            summary=summary,
            window_condition=window_condition,
            output_path=mae_path,
            dpi=dpi,
        )

        plot_difference_by_removal(
            summary=summary,
            window_condition=window_condition,
            output_path=difference_path,
            dpi=dpi,
        )

        paths[
            f"{window_condition}_mae"
        ] = mae_path

        paths[
            f"{window_condition}_difference"
        ] = difference_path

    return paths


# ============================================================
# Report writing
# ============================================================

def write_report(
    summary: pd.DataFrame,
    output_directory: str | Path,
    plot_paths: dict[str, Path],
    n_bootstrap: int,
) -> tuple[Path, Path]:
    """Write the Markdown report and combined CSV."""
    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        output_directory
        / REPORT_FILENAME
    )

    csv_path = (
        output_directory
        / CSV_FILENAME
    )

    csv_table = create_csv_table(summary)
    csv_table.to_csv(csv_path, index=False)

    plot_directory_name = (
        Path(PLOT_DIRECTORY_NAME)
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "# Query-Point vs. Learned-Regression "
            "Statistical Comparison\n\n"
        )

        file.write(
            "This report compares query-point and "
            "learned-regression absolute errors for the "
            "same people under each population size, removal "
            "percentage, and window condition.\n\n"
        )

        file.write(
            "`ΔMAE QP−LR` is query-point MAE minus "
            "learned-regression MAE. Positive values favor "
            "learned regression; negative values favor the "
            "query-point method.\n\n"
        )

        file.write(
            f"The confidence intervals use "
            f"{n_bootstrap:,} bootstrap samples. "
            "The paired t-test and Wilcoxon signed-rank test "
            "are two-sided. P-values are unadjusted for "
            "multiple comparisons.\n\n"
        )

        for window_condition in (
            "with_windows",
            "without_windows",
        ):
            window_title = WINDOW_LABELS[
                window_condition
            ]

            table = create_markdown_table(
                summary=summary,
                window_condition=window_condition,
            )

            file.write(
                f"## {window_title}\n\n"
            )

            file.write(
                dataframe_to_markdown(table)
            )

            file.write("\n")

            mae_relative_path = (
                plot_directory_name
                / f"{window_condition}_mae.png"
            )

            difference_relative_path = (
                plot_directory_name
                / f"{window_condition}_difference.png"
            )

            file.write(
                f"### MAE Across Removal Levels\n\n"
            )

            file.write(
                f"![{window_title} MAE comparison]"
                f"({mae_relative_path.as_posix()})\n\n"
            )

            file.write(
                "### Difference Between Methods\n\n"
            )

            file.write(
                f"![{window_title} method difference]"
                f"({difference_relative_path.as_posix()})\n\n"
            )

        file.write(
            "## Metric Notes\n\n"
        )

        file.write(
            "- **QP MAE/SD:** mean and sample standard "
            "deviation of query-point absolute errors.\n"
            "- **LR MAE/SD:** mean and sample standard "
            "deviation of learned-regression absolute errors.\n"
            "- **95% Bootstrap CI:** confidence interval for "
            "the mean paired difference, QP error minus LR "
            "error.\n"
            "- **Paired t p:** tests whether the mean paired "
            "difference is zero.\n"
            "- **Wilcoxon p:** signed-rank robustness test for "
            "the paired differences.\n"
            "- **Cohen's dz:** mean paired difference divided "
            "by the sample standard deviation of the paired "
            "differences.\n"
        )

    return report_path, csv_path


# ============================================================
# Main pipeline
# ============================================================

def generate_method_comparison(
    basepaths: list[str | Path],
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    seed: int = 42,
    n_bootstrap: int = 5000,
    dpi: int = 300,
) -> None:
    """Run the complete standalone comparison analysis."""
    loaded = [
        load_experiment(basepath)
        for basepath in basepaths
    ]

    query = pd.concat(
        [
            query_dataframe
            for query_dataframe, _
            in loaded
        ],
        ignore_index=True,
    )

    learning = pd.concat(
        [
            learning_dataframe
            for _, learning_dataframe
            in loaded
        ],
        ignore_index=True,
    )

    paired = pair_method_errors(
        query=query,
        learning=learning,
    )

    summary = summarize_comparisons(
        paired=paired,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )

    output_directory = Path(
        output_directory
    )

    plot_directory = (
        output_directory
        / PLOT_DIRECTORY_NAME
    )

    plot_paths = generate_plots(
        summary=summary,
        plot_directory=plot_directory,
        dpi=dpi,
    )

    report_path, csv_path = write_report(
        summary=summary,
        output_directory=output_directory,
        plot_paths=plot_paths,
        n_bootstrap=n_bootstrap,
    )

    print(
        f"Saved Markdown report to: {report_path}"
    )

    print(
        f"Saved CSV results to: {csv_path}"
    )

    print(
        f"Saved plots to: {plot_directory}"
    )


# ============================================================
# Command-line interface
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate standalone paired statistical "
            "comparisons between query-point and "
            "learned-regression age recovery."
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
            "Root folder used for automatic population-folder "
            "discovery. Default: current folder."
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
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for bootstrap confidence intervals. "
            "Default: 42."
        ),
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=5000,
        help=(
            "Number of bootstrap samples. Default: 5000."
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


def main() -> None:
    args = build_parser().parse_args()

    if args.basepath:
        basepaths = [
            Path(path)
            for path in args.basepath
        ]
    else:
        basepaths = discover_experiment_directories(
            args.root
        )

        print(
            "Automatically discovered population folders: "
            + ", ".join(
                str(path)
                for path in basepaths
            )
        )

    generate_method_comparison(
        basepaths=basepaths,
        output_directory=args.output_directory,
        seed=args.seed,
        n_bootstrap=args.bootstrap,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()