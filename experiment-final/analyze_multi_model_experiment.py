from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import shutil
import zlib
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import friedmanchisquare, ttest_rel, wilcoxon


QUERY_FILE = "person_year_predictions.csv"
LEARNING_FILE = "learning_person_predictions.csv"
MANIFEST_FILE = "kg_manifest.csv"
SHARED_HASH_FILE = "shared_kg_hashes.csv"

DEFAULT_ROOT = Path("model_comparison_500_shared_run1")
DEFAULT_OUTPUT = Path("final_results/model_comparison_500_shared_run1_comprehensive")

REPORT_FILE = "comprehensive-model-analysis.md"
PROVENANCE_FILE = "analysis-provenance.json"
CONDITION_SUMMARY_FILE = "condition-summary.csv"
PERFORMANCE_MATRIX_FILE = "performance-matrix.csv"
CONFIGURATION_RANKINGS_FILE = "configuration-rankings.csv"
ROBUSTNESS_FILE = "robustness-summary.csv"
BEST_CONFIGURATIONS_FILE = "best-configurations.csv"
METHOD_COMPARISONS_FILE = "method-comparisons.csv"
WINDOW_COMPARISONS_FILE = "window-comparisons.csv"
MODEL_OMNIBUS_FILE = "model-omnibus-tests.csv"
MODEL_PAIRWISE_FILE = "model-pairwise-tests.csv"
MODEL_WIN_SUMMARY_FILE = "model-win-summary.csv"
PLOT_DIRECTORY_NAME = "comprehensive-model-plots"

WINDOW_LABELS = {
    "with_windows": "With Windowing",
    "without_windows": "Without Windowing",
}
METHOD_LABELS = {
    "query_point": "Query Point",
    "learned_regression": "Learned Regression",
}
SUBSET_LABELS = {
    "all": "All People",
    "missing_only": "Missing hasAge Only",
    "retained_only": "Retained hasAge Only",
}
MODEL_LABELS = {
    "transe": "TransE",
    "distmult": "DistMult",
    "transd": "TransD",
    "transr": "TransR",
    "mure": "MuRE",
}

WINDOW_ORDER = {"with_windows": 0, "without_windows": 1}
METHOD_ORDER = {"query_point": 0, "learned_regression": 1}
SUBSET_ORDER = {"all": 0, "missing_only": 1, "retained_only": 2}
MODEL_ORDER = {"transe": 0, "distmult": 1, "transd": 2, "transr": 3, "mure": 4}
REPORT_SUBSETS = ("all", "missing_only")


# ============================================================
# General utilities
# ============================================================


def read_csv(path: str | Path, allow_empty: bool = False) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required file was not found: {path}")
    dataframe = pd.read_csv(path)
    if dataframe.empty and not allow_empty:
        raise ValueError(f"CSV contains no rows: {path}")
    return dataframe


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        raise ValueError("Boolean value cannot be empty.")
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "t", "yes", "y", "1"}:
        return True
    if normalized in {"false", "f", "no", "n", "0"}:
        return False
    raise ValueError(f"Could not interpret boolean value: {value!r}")


def normalize_model(value: str) -> str:
    return str(value).strip().lower().replace("-", "").replace("_", "")


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, str(model))


def ordered_models(values: Iterable[str]) -> list[str]:
    return sorted(values, key=lambda value: (MODEL_ORDER.get(value, 999), value))


def stable_seed(seed: int, values: Iterable[object]) -> int:
    token = "|".join(map(str, values)).encode("utf-8")
    return (seed + zlib.crc32(token)) % (2**32)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as infile:
        for block in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_output(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(
                f"Output directory already contains files: {path}\n"
                "Choose another --output-directory or rerun with --force."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def fmt(value, digits: int = 3) -> str:
    return "—" if pd.isna(value) else f"{float(value):.{digits}f}"


def fmt_percent(value) -> str:
    return "—" if pd.isna(value) else f"{float(value):g}%"


def fmt_p(value) -> str:
    if pd.isna(value):
        return "—"
    value = float(value)
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def markdown_escape(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "_No cases are available._\n"
    headers = [markdown_escape(column) for column in dataframe.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in dataframe.itertuples(index=False, name=None):
        lines.append(
            "| " + " | ".join(markdown_escape(value) for value in row) + " |"
        )
    return "\n".join(lines) + "\n"


# ============================================================
# Model-folder discovery, loading, and shared-KG validation
# ============================================================


def discover_model_folders(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Experiment root not found: {root}")

    folders = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and all(
            (path / filename).exists()
            for filename in (QUERY_FILE, LEARNING_FILE, MANIFEST_FILE)
        )
    ]
    folders.sort(
        key=lambda path: (
            MODEL_ORDER.get(normalize_model(path.name), 999),
            path.name.lower(),
        )
    )

    if not folders:
        raise ValueError(
            f"No completed model folders were found beneath {root}. "
            f"Each model folder must contain {QUERY_FILE}, {LEARNING_FILE}, "
            f"and {MANIFEST_FILE}."
        )
    return folders


def read_manifest(basepath: str | Path) -> pd.DataFrame:
    basepath = Path(basepath)
    path = basepath / MANIFEST_FILE
    manifest = read_csv(path)

    if "label" in manifest.columns:
        manifest = manifest.rename(columns={"label": "run"})

    required = {
        "run",
        "tsv_path",
        "window_condition",
        "removal_percent",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    manifest = manifest.copy()
    manifest["run"] = manifest["run"].astype(str).str.strip()
    manifest["window_condition"] = (
        manifest["window_condition"].astype(str).str.strip()
    )
    manifest["removal_percent"] = pd.to_numeric(
        manifest["removal_percent"], errors="raise"
    )

    if manifest["run"].duplicated().any():
        duplicated = sorted(
            manifest.loc[manifest["run"].duplicated(keep=False), "run"].unique()
        )
        raise ValueError(f"Manifest contains duplicate run labels: {duplicated}")

    unknown_windows = set(manifest["window_condition"]) - set(WINDOW_LABELS)
    if unknown_windows:
        raise ValueError(
            f"Unexpected window conditions in {path}: {sorted(unknown_windows)}"
        )

    manifest["resolved_tsv_path"] = manifest["tsv_path"].map(
        lambda raw: str(
            (Path(raw) if Path(raw).is_absolute() else basepath / Path(raw)).resolve()
        )
    )
    return manifest


def resolve_model(manifest: pd.DataFrame, basepath: Path) -> str:
    if "embedding_model" in manifest.columns:
        values = (
            manifest["embedding_model"]
            .dropna()
            .astype(str)
            .map(normalize_model)
            .unique()
        )
        if len(values) > 1:
            raise ValueError(
                f"{basepath / MANIFEST_FILE} contains multiple embedding models: "
                f"{values.tolist()}"
            )
        if len(values) == 1:
            return str(values[0])
    return normalize_model(basepath.name)


def validate_shared_manifests(
    model_manifests: dict[str, pd.DataFrame],
    verify_hashes: bool,
) -> pd.DataFrame:
    models = ordered_models(model_manifests)
    reference_model = models[0]
    reference = model_manifests[reference_model].copy()

    comparison_columns = [
        "run",
        "window_condition",
        "removal_percent",
        "resolved_tsv_path",
    ]
    reference_records = reference[comparison_columns].sort_values("run").reset_index(
        drop=True
    )

    for model in models[1:]:
        current = (
            model_manifests[model][comparison_columns]
            .sort_values("run")
            .reset_index(drop=True)
        )
        if not current.equals(reference_records):
            merged = reference_records.merge(
                current,
                on="run",
                how="outer",
                suffixes=(f"_{reference_model}", f"_{model}"),
                indicator=True,
            )
            mismatches = merged.loc[
                (merged["_merge"] != "both")
                | (
                    merged[f"window_condition_{reference_model}"]
                    != merged[f"window_condition_{model}"]
                )
                | ~np.isclose(
                    merged[f"removal_percent_{reference_model}"],
                    merged[f"removal_percent_{model}"],
                    equal_nan=True,
                )
                | (
                    merged[f"resolved_tsv_path_{reference_model}"]
                    != merged[f"resolved_tsv_path_{model}"]
                )
            ]
            raise ValueError(
                "Model manifests do not point to the exact same shared KG files.\n"
                f"Reference model: {model_label(reference_model)}\n"
                f"Comparison model: {model_label(model)}\n"
                f"Examples:\n{mismatches.head(10).to_string(index=False)}"
            )

    unique_paths = reference_records["resolved_tsv_path"].drop_duplicates().tolist()
    missing_paths = [path for path in unique_paths if not Path(path).exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Shared KG files referenced by the manifests are missing:\n"
            + "\n".join(f"  - {path}" for path in missing_paths)
        )

    hashes = {}
    if verify_hashes:
        for path in unique_paths:
            hashes[path] = sha256_file(path)

    provenance = reference_records.copy()
    provenance["sha256"] = (
        provenance["resolved_tsv_path"].map(hashes)
        if verify_hashes
        else None
    )
    return provenance


def attach_manifest_metadata(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    source_path: Path,
) -> pd.DataFrame:
    if "run" not in predictions.columns:
        raise ValueError(f"{source_path} is missing the 'run' column.")

    predictions = predictions.copy()
    predictions["run"] = predictions["run"].astype(str).str.strip()
    predictions = predictions.drop(
        columns=[
            column
            for column in ("window_condition", "removal_percent")
            if column in predictions.columns
        ]
    )
    predictions = predictions.merge(
        manifest[["run", "window_condition", "removal_percent"]],
        on="run",
        how="left",
        validate="many_to_one",
    )

    missing_metadata = predictions[["window_condition", "removal_percent"]].isna().any(
        axis=1
    )
    if missing_metadata.any():
        unmatched = sorted(predictions.loc[missing_metadata, "run"].unique())
        raise ValueError(
            f"{source_path} contains runs that are absent from its manifest: "
            f"{unmatched}"
        )
    return predictions


def determine_population_size(
    query: pd.DataFrame,
    learning: pd.DataFrame,
    basepath: Path,
) -> int:
    for source_name, dataframe in ((QUERY_FILE, query), (LEARNING_FILE, learning)):
        required = {"run", "person"}
        missing = required - set(dataframe.columns)
        if missing:
            raise ValueError(
                f"{basepath / source_name} is missing columns: {sorted(missing)}"
            )

    query_sizes = query.groupby("run")["person"].nunique()
    learning_sizes = learning.groupby("run")["person"].nunique()

    if query_sizes.nunique() != 1:
        raise ValueError(
            f"Query-point runs in {basepath} have inconsistent population sizes: "
            f"{query_sizes.to_dict()}"
        )
    if learning_sizes.nunique() != 1:
        raise ValueError(
            f"Learned-regression runs in {basepath} have inconsistent population "
            f"sizes: {learning_sizes.to_dict()}"
        )

    query_size = int(query_sizes.iloc[0])
    learning_size = int(learning_sizes.iloc[0])
    if query_size != learning_size:
        raise ValueError(
            f"QP and LR population sizes differ in {basepath}: "
            f"{query_size} versus {learning_size}"
        )
    return query_size


def standardize_predictions(
    dataframe: pd.DataFrame,
    method: str,
    model: str,
    population_size: int,
    source_path: Path,
) -> pd.DataFrame:
    if method == "query_point":
        required = {
            "run",
            "person",
            "true_age",
            "ground_truth_missing_in_run",
            "window_condition",
            "removal_percent",
            "top1_abs_error",
        }
        prediction_column = "top1_pred_age"
        signed_error_column = "top1_error"
        absolute_error_column = "top1_abs_error"
    else:
        required = {
            "run",
            "person",
            "true_age",
            "ground_truth_missing_in_run",
            "window_condition",
            "removal_percent",
            "abs_error",
        }
        prediction_column = "predicted_age"
        signed_error_column = "error"
        absolute_error_column = "abs_error"

    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"{source_path} is missing columns: {sorted(missing)}")

    result = dataframe[
        [
            "run",
            "person",
            "true_age",
            "ground_truth_missing_in_run",
            "window_condition",
            "removal_percent",
            *(
                [prediction_column]
                if prediction_column in dataframe.columns
                else []
            ),
            *(
                [signed_error_column]
                if signed_error_column in dataframe.columns
                else []
            ),
            absolute_error_column,
        ]
    ].copy()

    result = result.rename(columns={absolute_error_column: "abs_error"})
    if prediction_column in result.columns:
        result = result.rename(columns={prediction_column: "predicted_age"})
    if signed_error_column in result.columns:
        result = result.rename(columns={signed_error_column: "signed_error"})

    result["run"] = result["run"].astype(str).str.strip()
    result["person"] = result["person"].astype(str).str.strip()
    result["true_age"] = pd.to_numeric(result["true_age"], errors="raise")
    result["abs_error"] = pd.to_numeric(result["abs_error"], errors="raise")
    result["removal_percent"] = pd.to_numeric(
        result["removal_percent"], errors="raise"
    )
    result["ground_truth_missing_in_run"] = result[
        "ground_truth_missing_in_run"
    ].map(parse_bool)

    if "predicted_age" in result.columns:
        result["predicted_age"] = pd.to_numeric(
            result["predicted_age"], errors="raise"
        )
    else:
        result["predicted_age"] = np.nan

    if "signed_error" in result.columns:
        result["signed_error"] = pd.to_numeric(
            result["signed_error"], errors="raise"
        )
    else:
        result["signed_error"] = result["predicted_age"] - result["true_age"]

    result["model"] = model
    result["method"] = method
    result["population_size"] = population_size

    keys = [
        "model",
        "method",
        "window_condition",
        "removal_percent",
        "person",
    ]
    if result.duplicated(keys).any():
        duplicates = result.loc[result.duplicated(keys, keep=False), keys].head(10)
        raise ValueError(
            f"Duplicate prediction rows were found in {source_path}:\n"
            f"{duplicates.to_string(index=False)}"
        )
    return result


def load_model_folder(
    basepath: str | Path,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    basepath = Path(basepath)
    manifest = read_manifest(basepath)
    model = resolve_model(manifest, basepath)

    query_path = basepath / QUERY_FILE
    learning_path = basepath / LEARNING_FILE

    query = attach_manifest_metadata(read_csv(query_path), manifest, query_path)
    learning = attach_manifest_metadata(read_csv(learning_path), manifest, learning_path)
    population_size = determine_population_size(query, learning, basepath)

    standardized = pd.concat(
        [
            standardize_predictions(
                query,
                "query_point",
                model,
                population_size,
                query_path,
            ),
            standardize_predictions(
                learning,
                "learned_regression",
                model,
                population_size,
                learning_path,
            ),
        ],
        ignore_index=True,
    )
    return model, manifest, standardized


def validate_prediction_alignment(dataframe: pd.DataFrame) -> None:
    models = ordered_models(dataframe["model"].unique())
    if len(models) < 2:
        raise ValueError("At least two embedding models are required.")

    population_sizes = dataframe["population_size"].unique().tolist()
    if len(population_sizes) != 1:
        raise ValueError(
            f"Model folders contain different population sizes: {population_sizes}"
        )

    expected_conditions = set(
        dataframe.loc[dataframe["model"] == models[0], [
            "method",
            "window_condition",
            "removal_percent",
        ]].itertuples(index=False, name=None)
    )
    for model in models[1:]:
        conditions = set(
            dataframe.loc[dataframe["model"] == model, [
                "method",
                "window_condition",
                "removal_percent",
            ]].itertuples(index=False, name=None)
        )
        if conditions != expected_conditions:
            raise ValueError(
                f"{model_label(model)} has different analysis conditions from "
                f"{model_label(models[0])}.\n"
                f"Missing: {sorted(expected_conditions - conditions)}\n"
                f"Extra: {sorted(conditions - expected_conditions)}"
            )

    cross_model_keys = ["method", "window_condition", "removal_percent", "person"]
    if dataframe.groupby(cross_model_keys)["true_age"].nunique().gt(1).any():
        raise ValueError("True ages disagree across model folders.")
    if (
        dataframe.groupby(cross_model_keys)["ground_truth_missing_in_run"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise ValueError(
            "Missingness differs across models. The models were not evaluated on "
            "the same removal plan."
        )

    person_sets = (
        dataframe.groupby(["model", "method", "window_condition", "removal_percent"])[
            "person"
        ]
        .agg(lambda values: frozenset(values))
        .reset_index()
    )
    for condition, group in person_sets.groupby(
        ["method", "window_condition", "removal_percent"], sort=False
    ):
        if group["person"].nunique() != 1:
            raise ValueError(
                "Models do not contain the same people under condition "
                f"{condition}."
            )

    within_model = dataframe.groupby(
        ["model", "window_condition", "removal_percent", "person"]
    )
    if within_model["true_age"].nunique().gt(1).any():
        raise ValueError("QP and LR disagree about true ages within a model.")
    if within_model["ground_truth_missing_in_run"].nunique().gt(1).any():
        raise ValueError("QP and LR disagree about missingness within a model.")


# ============================================================
# Subsets and descriptive summaries
# ============================================================


def add_subsets(dataframe: pd.DataFrame) -> pd.DataFrame:
    all_people = dataframe.assign(subset="all")
    missing = dataframe.loc[dataframe["ground_truth_missing_in_run"]].assign(
        subset="missing_only"
    )
    retained = dataframe.loc[~dataframe["ground_truth_missing_in_run"]].assign(
        subset="retained_only"
    )
    return pd.concat([all_people, missing, retained], ignore_index=True)


def summarize_conditions(dataframe: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "subset",
        "model",
        "method",
        "window_condition",
        "removal_percent",
        "population_size",
    ]
    summary = (
        dataframe.groupby(group_columns, sort=False)
        .agg(
            n=("abs_error", "size"),
            mae=("abs_error", "mean"),
            sd_abs_error=("abs_error", "std"),
            median_abs_error=("abs_error", "median"),
            q25_abs_error=("abs_error", lambda values: values.quantile(0.25)),
            q75_abs_error=("abs_error", lambda values: values.quantile(0.75)),
            max_abs_error=("abs_error", "max"),
            rmse=("signed_error", lambda values: math.sqrt(np.mean(np.square(values)))),
            mean_signed_error=("signed_error", "mean"),
            sd_signed_error=("signed_error", "std"),
        )
        .reset_index()
    )

    model_condition = ["subset", "method", "window_condition", "removal_percent"]
    summary["model_rank"] = summary.groupby(model_condition)["mae"].rank(
        method="min"
    )
    summary["model_is_best"] = summary["model_rank"] == 1

    configuration_condition = ["subset", "removal_percent"]
    summary["configuration_rank"] = summary.groupby(configuration_condition)["mae"].rank(
        method="min"
    )
    summary["configuration_is_best"] = summary["configuration_rank"] == 1
    summary["model_label"] = summary["model"].map(model_label)
    summary["method_label"] = summary["method"].map(METHOD_LABELS)
    summary["window_label"] = summary["window_condition"].map(WINDOW_LABELS)
    summary["subset_label"] = summary["subset"].map(SUBSET_LABELS)

    summary["_subset"] = summary["subset"].map(SUBSET_ORDER)
    summary["_method"] = summary["method"].map(METHOD_ORDER)
    summary["_window"] = summary["window_condition"].map(WINDOW_ORDER)
    summary["_model"] = summary["model"].map(MODEL_ORDER).fillna(999)
    summary = summary.sort_values(
        ["_subset", "_window", "removal_percent", "_method", "_model"]
    ).reset_index(drop=True)
    return summary.drop(columns=["_subset", "_method", "_window", "_model"])


def build_performance_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    matrix = summary.pivot_table(
        index=["subset", "window_condition", "removal_percent"],
        columns=["model", "method"],
        values="mae",
        aggfunc="first",
    )
    matrix.columns = [f"{model}_{method}_mae" for model, method in matrix.columns]
    return matrix.reset_index()


def build_configuration_rankings(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "subset",
        "removal_percent",
        "configuration_rank",
        "model",
        "model_label",
        "method",
        "method_label",
        "window_condition",
        "window_label",
        "n",
        "mae",
        "sd_abs_error",
        "median_abs_error",
        "rmse",
        "mean_signed_error",
    ]
    result = summary[columns].copy()
    result["_subset"] = result["subset"].map(SUBSET_ORDER)
    result = result.sort_values(
        ["_subset", "removal_percent", "configuration_rank", "mae"]
    ).reset_index(drop=True)
    return result.drop(columns="_subset")


def build_best_configurations(configuration_rankings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subset, removal), group in configuration_rankings.groupby(
        ["subset", "removal_percent"], sort=False
    ):
        ordered = group.sort_values(["mae", "configuration_rank"])
        best = ordered.iloc[0]
        runner_up = ordered.iloc[1] if len(ordered) > 1 else None
        rows.append(
            {
                "subset": subset,
                "removal_percent": float(removal),
                "best_model": best["model"],
                "best_model_label": best["model_label"],
                "best_method": best["method"],
                "best_method_label": best["method_label"],
                "best_window_condition": best["window_condition"],
                "best_window_label": best["window_label"],
                "best_mae": float(best["mae"]),
                "runner_up_model": None if runner_up is None else runner_up["model"],
                "runner_up_model_label": None
                if runner_up is None
                else runner_up["model_label"],
                "runner_up_method": None if runner_up is None else runner_up["method"],
                "runner_up_method_label": None
                if runner_up is None
                else runner_up["method_label"],
                "runner_up_window_condition": None
                if runner_up is None
                else runner_up["window_condition"],
                "runner_up_window_label": None
                if runner_up is None
                else runner_up["window_label"],
                "runner_up_mae": np.nan if runner_up is None else float(runner_up["mae"]),
                "margin_to_runner_up": np.nan
                if runner_up is None
                else float(runner_up["mae"] - best["mae"]),
            }
        )
    result = pd.DataFrame(rows)
    result["_subset"] = result["subset"].map(SUBSET_ORDER)
    result = result.sort_values(["_subset", "removal_percent"]).reset_index(drop=True)
    return result.drop(columns="_subset")


def normalized_auc(x: Sequence[float], y: Sequence[float]) -> float:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    order = np.argsort(x_values)
    x_values = x_values[order]
    y_values = y_values[order]
    if len(x_values) == 0:
        return np.nan
    if len(x_values) == 1 or np.isclose(x_values[-1], x_values[0]):
        return float(y_values.mean())
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(y_values, x_values) / (x_values[-1] - x_values[0]))


def summarize_robustness(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["subset", "model", "method", "window_condition"]

    for keys, group in summary.groupby(group_columns, sort=False):
        subset, model, method, window = keys
        group = group.sort_values("removal_percent")
        removals = group["removal_percent"].to_numpy(dtype=float)
        maes = group["mae"].to_numpy(dtype=float)
        ranks = group["model_rank"].to_numpy(dtype=float)

        slope_per_percent = (
            float(np.polyfit(removals, maes, 1)[0]) if len(group) >= 2 else np.nan
        )
        baseline = float(maes[0])
        final = float(maes[-1])

        rows.append(
            {
                "subset": subset,
                "model": model,
                "model_label": model_label(model),
                "method": method,
                "method_label": METHOD_LABELS[method],
                "window_condition": window,
                "window_label": WINDOW_LABELS[window],
                "n_removal_levels": len(group),
                "minimum_removal_percent": float(removals[0]),
                "maximum_removal_percent": float(removals[-1]),
                "normalized_auc_mae": normalized_auc(removals, maes),
                "mean_mae": float(maes.mean()),
                "baseline_mae": baseline,
                "final_mae": final,
                "absolute_mae_change": final - baseline,
                "relative_mae_change_percent": (
                    np.nan
                    if np.isclose(baseline, 0)
                    else 100.0 * (final - baseline) / baseline
                ),
                "slope_years_per_10pct_removal": 10.0 * slope_per_percent,
                "mean_model_rank": float(ranks.mean()),
                "median_model_rank": float(np.median(ranks)),
                "first_place_finishes": int(np.sum(ranks == 1)),
            }
        )

    result = pd.DataFrame(rows)
    result["robustness_rank_within_method_window"] = result.groupby(
        ["subset", "method", "window_condition"]
    )["normalized_auc_mae"].rank(method="min")
    result["overall_configuration_rank"] = result.groupby("subset")[
        "normalized_auc_mae"
    ].rank(method="min")

    result["_subset"] = result["subset"].map(SUBSET_ORDER)
    result["_method"] = result["method"].map(METHOD_ORDER)
    result["_window"] = result["window_condition"].map(WINDOW_ORDER)
    result["_model"] = result["model"].map(MODEL_ORDER).fillna(999)
    result = result.sort_values(
        [
            "_subset",
            "overall_configuration_rank",
            "_method",
            "_window",
            "_model",
        ]
    ).reset_index(drop=True)
    return result.drop(columns=["_subset", "_method", "_window", "_model"])


# ============================================================
# Paired statistics and multiple-comparison correction
# ============================================================


def bootstrap_mean_ci(
    differences: np.ndarray,
    seed: int,
    n_bootstrap: int,
) -> tuple[float, float]:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    batch_size = 500

    for start in range(0, n_bootstrap, batch_size):
        stop = min(start + batch_size, n_bootstrap)
        indices = rng.integers(
            0,
            len(differences),
            size=(stop - start, len(differences)),
        )
        means[start:stop] = differences[indices].mean(axis=1)

    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def paired_statistics(
    errors_a,
    errors_b,
    seed: int,
    n_bootstrap: int,
) -> dict:
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    valid = np.isfinite(errors_a) & np.isfinite(errors_b)
    errors_a = errors_a[valid]
    errors_b = errors_b[valid]

    if len(errors_a) != len(errors_b):
        raise ValueError("Paired arrays have unequal lengths.")

    n = len(errors_a)
    if n == 0:
        return {
            "n": 0,
            "a_mae": np.nan,
            "b_mae": np.nan,
            "mean_difference_a_minus_b": np.nan,
            "median_difference_a_minus_b": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "t_statistic": np.nan,
            "t_p": np.nan,
            "wilcoxon_statistic": np.nan,
            "wilcoxon_p": np.nan,
            "cohen_dz": np.nan,
            "a_win_rate": np.nan,
            "b_win_rate": np.nan,
            "tie_rate": np.nan,
            "lower_mae": "No cases",
        }

    differences = errors_a - errors_b
    a_mae = float(errors_a.mean())
    b_mae = float(errors_b.mean())
    ci_low, ci_high = bootstrap_mean_ci(differences, seed, n_bootstrap)

    if np.allclose(differences, 0):
        t_statistic, t_p = 0.0, 1.0
        wilcoxon_statistic, wilcoxon_p = 0.0, 1.0
    else:
        if n >= 2:
            difference_sd_for_test = float(np.std(differences, ddof=1))
            if np.isclose(difference_sd_for_test, 0):
                t_statistic = math.copysign(np.inf, float(differences.mean()))
                t_p = 0.0
            else:
                t_result = ttest_rel(errors_a, errors_b, nan_policy="omit")
                t_statistic = float(t_result.statistic)
                t_p = float(t_result.pvalue)
        else:
            t_statistic, t_p = np.nan, np.nan

        try:
            wilcoxon_result = wilcoxon(
                differences,
                alternative="two-sided",
                zero_method="wilcox",
            )
            wilcoxon_statistic = float(wilcoxon_result.statistic)
            wilcoxon_p = float(wilcoxon_result.pvalue)
        except ValueError:
            wilcoxon_statistic, wilcoxon_p = np.nan, np.nan

    difference_sd = float(np.std(differences, ddof=1)) if n >= 2 else np.nan
    cohen_dz = (
        float(differences.mean() / difference_sd)
        if np.isfinite(difference_sd) and difference_sd > 0
        else np.nan
    )

    a_win_rate = float(np.mean(errors_a < errors_b))
    b_win_rate = float(np.mean(errors_b < errors_a))
    tie_rate = float(np.mean(np.isclose(errors_a, errors_b)))

    if np.isclose(a_mae, b_mae):
        lower_mae = "Tie"
    elif a_mae < b_mae:
        lower_mae = "A"
    else:
        lower_mae = "B"

    return {
        "n": n,
        "a_mae": a_mae,
        "b_mae": b_mae,
        "mean_difference_a_minus_b": float(differences.mean()),
        "median_difference_a_minus_b": float(np.median(differences)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "t_statistic": t_statistic,
        "t_p": t_p,
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_p": wilcoxon_p,
        "cohen_dz": cohen_dz,
        "a_win_rate": a_win_rate,
        "b_win_rate": b_win_rate,
        "tie_rate": tie_rate,
        "lower_mae": lower_mae,
    }


def holm_adjust(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    ordered = values.dropna().astype(float).sort_values()
    running = 0.0
    number = len(ordered)

    for rank, (index, p_value) in enumerate(ordered.items(), start=1):
        adjusted = min(1.0, (number - rank + 1) * p_value)
        running = max(running, adjusted)
        result.loc[index] = running
    return result


def pair_two_groups(
    group_a: pd.DataFrame,
    group_b: pd.DataFrame,
    context: str,
) -> pd.DataFrame:
    a = group_a[["person", "true_age", "abs_error"]].rename(
        columns={"true_age": "true_age_a", "abs_error": "error_a"}
    )
    b = group_b[["person", "true_age", "abs_error"]].rename(
        columns={"true_age": "true_age_b", "abs_error": "error_b"}
    )
    paired = a.merge(b, on="person", how="inner", validate="one_to_one")

    if len(paired) != len(a) or len(paired) != len(b):
        raise ValueError(f"Could not pair every person for {context}.")
    if not np.allclose(paired["true_age_a"], paired["true_age_b"]):
        raise ValueError(f"True ages disagree for {context}.")
    return paired


# ============================================================
# QP-versus-LR, window, and model comparisons
# ============================================================


def compare_methods(
    dataframe: pd.DataFrame,
    seed: int,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows = []
    conditions = [
        "subset",
        "model",
        "window_condition",
        "removal_percent",
        "population_size",
    ]

    for condition, group in dataframe.groupby(conditions, sort=False):
        subset, model, window, removal, population_size = condition
        qp = group.loc[group["method"] == "query_point"]
        lr = group.loc[group["method"] == "learned_regression"]
        paired = pair_two_groups(
            qp,
            lr,
            f"QP versus LR under {condition}",
        )
        statistics = paired_statistics(
            paired["error_a"],
            paired["error_b"],
            stable_seed(seed, ["method", *condition]),
            n_bootstrap,
        )
        lower = statistics.pop("lower_mae")
        lower_label = (
            METHOD_LABELS["query_point"]
            if lower == "A"
            else METHOD_LABELS["learned_regression"]
            if lower == "B"
            else lower
        )
        rows.append(
            {
                "subset": subset,
                "model": model,
                "model_label": model_label(model),
                "window_condition": window,
                "window_label": WINDOW_LABELS[window],
                "removal_percent": float(removal),
                "population_size": int(population_size),
                "method_a": "query_point",
                "method_a_label": METHOD_LABELS["query_point"],
                "method_b": "learned_regression",
                "method_b_label": METHOD_LABELS["learned_regression"],
                **statistics,
                "lower_mae_method": lower_label,
            }
        )

    result = pd.DataFrame(rows)
    result = result.rename(
        columns={
            "a_mae": "query_point_mae",
            "b_mae": "learned_regression_mae",
            "mean_difference_a_minus_b": "mean_difference_qp_minus_lr",
            "median_difference_a_minus_b": "median_difference_qp_minus_lr",
            "a_win_rate": "query_point_person_win_rate",
            "b_win_rate": "learned_regression_person_win_rate",
        }
    )
    correction_groups = ["subset", "window_condition", "removal_percent"]
    result["t_p_holm"] = result.groupby(correction_groups)["t_p"].transform(
        holm_adjust
    )
    result["wilcoxon_p_holm"] = result.groupby(correction_groups)[
        "wilcoxon_p"
    ].transform(holm_adjust)
    return result.sort_values(
        ["subset", "window_condition", "removal_percent", "model"]
    ).reset_index(drop=True)


def compare_windows(
    dataframe: pd.DataFrame,
    seed: int,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows = []
    conditions = [
        "subset",
        "model",
        "method",
        "removal_percent",
        "population_size",
    ]

    for condition, group in dataframe.groupby(conditions, sort=False):
        subset, model, method, removal, population_size = condition
        without_windows = group.loc[group["window_condition"] == "without_windows"]
        with_windows = group.loc[group["window_condition"] == "with_windows"]
        paired = pair_two_groups(
            without_windows,
            with_windows,
            f"without versus with windowing under {condition}",
        )
        statistics = paired_statistics(
            paired["error_a"],
            paired["error_b"],
            stable_seed(seed, ["window", *condition]),
            n_bootstrap,
        )
        lower = statistics.pop("lower_mae")
        lower_label = (
            WINDOW_LABELS["without_windows"]
            if lower == "A"
            else WINDOW_LABELS["with_windows"]
            if lower == "B"
            else lower
        )
        rows.append(
            {
                "subset": subset,
                "model": model,
                "model_label": model_label(model),
                "method": method,
                "method_label": METHOD_LABELS[method],
                "removal_percent": float(removal),
                "population_size": int(population_size),
                "window_a": "without_windows",
                "window_a_label": WINDOW_LABELS["without_windows"],
                "window_b": "with_windows",
                "window_b_label": WINDOW_LABELS["with_windows"],
                **statistics,
                "lower_mae_window_condition": lower_label,
            }
        )

    result = pd.DataFrame(rows)
    result = result.rename(
        columns={
            "a_mae": "without_windows_mae",
            "b_mae": "with_windows_mae",
            "mean_difference_a_minus_b": "mean_difference_without_minus_with",
            "median_difference_a_minus_b": "median_difference_without_minus_with",
            "a_win_rate": "without_windows_person_win_rate",
            "b_win_rate": "with_windows_person_win_rate",
        }
    )
    correction_groups = ["subset", "method", "removal_percent"]
    result["t_p_holm"] = result.groupby(correction_groups)["t_p"].transform(
        holm_adjust
    )
    result["wilcoxon_p_holm"] = result.groupby(correction_groups)[
        "wilcoxon_p"
    ].transform(holm_adjust)
    return result.sort_values(
        ["subset", "method", "removal_percent", "model"]
    ).reset_index(drop=True)


def compare_models(
    dataframe: pd.DataFrame,
    seed: int,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    omnibus_rows = []
    pairwise_rows = []
    conditions = [
        "subset",
        "method",
        "window_condition",
        "removal_percent",
        "population_size",
    ]

    for condition, group in dataframe.groupby(conditions, sort=False):
        subset, method, window, removal, population_size = condition
        models = ordered_models(group["model"].unique())

        wide = group.pivot(index="person", columns="model", values="abs_error")
        wide = wide[models]
        if wide.isna().any().any():
            raise ValueError(
                f"Models could not be paired for every person under {condition}."
            )

        if len(models) >= 3 and len(wide) >= 2:
            if np.allclose(wide.to_numpy(), wide.iloc[:, [0]].to_numpy()):
                statistic, p_value = 0.0, 1.0
            else:
                friedman = friedmanchisquare(
                    *[wide[model].to_numpy(dtype=float) for model in models]
                )
                statistic = float(friedman.statistic)
                p_value = float(friedman.pvalue)
            kendalls_w = statistic / (len(wide) * (len(models) - 1))
        else:
            statistic, p_value, kendalls_w = np.nan, np.nan, np.nan

        omnibus_rows.append(
            {
                "subset": subset,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "window_condition": window,
                "window_label": WINDOW_LABELS[window],
                "removal_percent": float(removal),
                "population_size": int(population_size),
                "n_people": len(wide),
                "n_models": len(models),
                "friedman_chi_square": statistic,
                "friedman_p": p_value,
                "kendalls_w": kendalls_w,
            }
        )

        for model_a, model_b in itertools.combinations(models, 2):
            statistics = paired_statistics(
                wide[model_a],
                wide[model_b],
                stable_seed(seed, ["model", *condition, model_a, model_b]),
                n_bootstrap,
            )
            lower = statistics.pop("lower_mae")
            lower_label = (
                model_label(model_a)
                if lower == "A"
                else model_label(model_b)
                if lower == "B"
                else lower
            )
            pairwise_rows.append(
                {
                    "subset": subset,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "window_condition": window,
                    "window_label": WINDOW_LABELS[window],
                    "removal_percent": float(removal),
                    "population_size": int(population_size),
                    "model_a": model_a,
                    "model_a_label": model_label(model_a),
                    "model_b": model_b,
                    "model_b_label": model_label(model_b),
                    **statistics,
                    "lower_mae_model": lower_label,
                }
            )

    omnibus = pd.DataFrame(omnibus_rows)
    pairwise = pd.DataFrame(pairwise_rows)

    if not pairwise.empty:
        correction_groups = [
            "subset",
            "method",
            "window_condition",
            "removal_percent",
        ]
        pairwise["t_p_holm"] = pairwise.groupby(correction_groups)["t_p"].transform(
            holm_adjust
        )
        pairwise["wilcoxon_p_holm"] = pairwise.groupby(correction_groups)[
            "wilcoxon_p"
        ].transform(holm_adjust)

    return (
        omnibus.sort_values(
            ["subset", "method", "window_condition", "removal_percent"]
        ).reset_index(drop=True),
        pairwise.sort_values(
            [
                "subset",
                "method",
                "window_condition",
                "removal_percent",
                "model_a",
                "model_b",
            ]
        ).reset_index(drop=True),
    )


def summarize_model_wins(
    summary: pd.DataFrame,
    model_pairwise: pd.DataFrame,
) -> pd.DataFrame:
    models = ordered_models(summary["model"].unique())
    rows = []

    for subset in summary["subset"].drop_duplicates():
        for model in models:
            model_summary = summary.loc[
                (summary["subset"] == subset) & (summary["model"] == model)
            ]
            comparisons = model_pairwise.loc[model_pairwise["subset"] == subset]
            model_comparisons = comparisons.loc[
                (comparisons["model_a"] == model) | (comparisons["model_b"] == model)
            ]

            significant = model_comparisons.loc[
                model_comparisons["wilcoxon_p_holm"] < 0.05
            ]
            significant_wins = int(
                (significant["lower_mae_model"] == model_label(model)).sum()
            )
            significant_losses = int(
                (
                    significant["lower_mae_model"].notna()
                    & (significant["lower_mae_model"] != model_label(model))
                    & (significant["lower_mae_model"] != "Tie")
                ).sum()
            )

            rows.append(
                {
                    "subset": subset,
                    "model": model,
                    "model_label": model_label(model),
                    "conditions": len(model_summary),
                    "first_place_conditions": int(model_summary["model_is_best"].sum()),
                    "mean_model_rank": float(model_summary["model_rank"].mean()),
                    "mean_mae": float(model_summary["mae"].mean()),
                    "pairwise_tests": len(model_comparisons),
                    "significant_pairwise_wins": significant_wins,
                    "significant_pairwise_losses": significant_losses,
                    "significant_win_loss_balance": significant_wins
                    - significant_losses,
                }
            )

    result = pd.DataFrame(rows)
    result["_subset"] = result["subset"].map(SUBSET_ORDER)
    result["_model"] = result["model"].map(MODEL_ORDER).fillna(999)
    result = result.sort_values(
        ["_subset", "mean_model_rank", "mean_mae", "_model"]
    ).reset_index(drop=True)
    return result.drop(columns=["_subset", "_model"])


# ============================================================
# Tables for the Markdown report
# ============================================================


def performance_markdown_table(
    summary: pd.DataFrame,
    subset: str,
    window_condition: str,
) -> pd.DataFrame:
    results = summary.loc[
        (summary["subset"] == subset)
        & (summary["window_condition"] == window_condition)
    ]
    if results.empty:
        return pd.DataFrame()

    models = ordered_models(results["model"].unique())
    rows = []
    for removal, group in results.groupby("removal_percent", sort=True):
        row = {"Removal %": fmt_percent(removal)}
        for model in models:
            model_group = group.loc[group["model"] == model]
            qp = model_group.loc[model_group["method"] == "query_point"]
            lr = model_group.loc[model_group["method"] == "learned_regression"]
            if qp.empty or lr.empty:
                row[model_label(model)] = "—"
            else:
                row[model_label(model)] = (
                    f"{fmt(qp.iloc[0]['mae'])} / {fmt(lr.iloc[0]['mae'])}"
                )

        best = group.sort_values("mae").iloc[0]
        row["Best Configuration"] = (
            f"{best['model_label']} · {best['method_label']} · {fmt(best['mae'])}"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def delta_matrix_table(
    comparisons: pd.DataFrame,
    subset: str,
    category: str,
    fixed_value: str,
) -> pd.DataFrame:
    if category == "method":
        results = comparisons.loc[
            (comparisons["subset"] == subset)
            & (comparisons["window_condition"] == fixed_value)
        ]
        value_column = "mean_difference_qp_minus_lr"
    elif category == "window":
        results = comparisons.loc[
            (comparisons["subset"] == subset)
            & (comparisons["method"] == fixed_value)
        ]
        value_column = "mean_difference_without_minus_with"
    else:
        raise ValueError(category)

    if results.empty:
        return pd.DataFrame()

    pivot = results.pivot(index="model", columns="removal_percent", values=value_column)
    pivot = pivot.reindex(ordered_models(pivot.index))
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    pivot.index = [model_label(model) for model in pivot.index]
    pivot.columns = [fmt_percent(value) for value in pivot.columns]
    formatted = pivot.map(fmt) if hasattr(pivot, "map") else pivot.applymap(fmt)
    return formatted.reset_index(names="Model")


def best_configurations_markdown_table(
    best_configurations: pd.DataFrame,
    subset: str,
) -> pd.DataFrame:
    results = best_configurations.loc[best_configurations["subset"] == subset]
    return pd.DataFrame(
        {
            "Removal %": results["removal_percent"].map(fmt_percent),
            "Best Model": results["best_model_label"],
            "Method": results["best_method_label"],
            "Windowing": results["best_window_label"],
            "MAE": results["best_mae"].map(fmt),
            "Runner-Up": [
                (
                    "—"
                    if pd.isna(model)
                    else f"{model} · {method} · {window}"
                )
                for model, method, window in zip(
                    results["runner_up_model_label"],
                    results["runner_up_method_label"],
                    results["runner_up_window_label"],
                )
            ],
            "Margin": results["margin_to_runner_up"].map(fmt),
        }
    )


def robustness_markdown_table(
    robustness: pd.DataFrame,
    subset: str,
    limit: int = 12,
) -> pd.DataFrame:
    results = (
        robustness.loc[robustness["subset"] == subset]
        .sort_values(["overall_configuration_rank", "normalized_auc_mae"])
        .head(limit)
    )
    return pd.DataFrame(
        {
            "Rank": results["overall_configuration_rank"].map(lambda value: int(value)),
            "Model": results["model_label"],
            "Method": results["method_label"],
            "Windowing": results["window_label"],
            "Normalized AUC MAE": results["normalized_auc_mae"].map(fmt),
            "First MAE": results["baseline_mae"].map(fmt),
            "Final MAE": results["final_mae"].map(fmt),
            "Δ MAE": results["absolute_mae_change"].map(fmt),
            "Slope / 10%": results["slope_years_per_10pct_removal"].map(fmt),
            "Mean Model Rank": results["mean_model_rank"].map(fmt),
        }
    )


def model_omnibus_markdown_table(
    omnibus: pd.DataFrame,
    subset: str,
) -> pd.DataFrame:
    results = omnibus.loc[omnibus["subset"] == subset].copy()
    results = results.sort_values(
        ["window_condition", "method", "removal_percent"]
    )
    return pd.DataFrame(
        {
            "Windowing": results["window_label"],
            "Method": results["method_label"],
            "Removal %": results["removal_percent"].map(fmt_percent),
            "Friedman χ²": results["friedman_chi_square"].map(fmt),
            "p": results["friedman_p"].map(fmt_p),
            "Kendall's W": results["kendalls_w"].map(fmt),
        }
    )


# ============================================================
# Visualizations
# ============================================================


def save_figure(figure, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_model_lines(
    summary: pd.DataFrame,
    subset: str,
    method: str,
    window_condition: str,
    output_path: Path,
    dpi: int,
) -> None:
    results = summary.loc[
        (summary["subset"] == subset)
        & (summary["method"] == method)
        & (summary["window_condition"] == window_condition)
    ]
    if results.empty:
        return

    figure, axis = plt.subplots(figsize=(10, 6.5))
    for model in ordered_models(results["model"].unique()):
        model_results = results.loc[results["model"] == model].sort_values(
            "removal_percent"
        )
        axis.plot(
            model_results["removal_percent"],
            model_results["mae"],
            marker="o",
            linewidth=2,
            label=model_label(model),
        )

    axis.set_xticks(sorted(results["removal_percent"].unique()))
    axis.set_xlabel("Removed hasAge relations (%)")
    axis.set_ylabel("Mean absolute error (years)")
    axis.set_title(
        f"{METHOD_LABELS[method]} — {WINDOW_LABELS[window_condition]} — "
        f"{SUBSET_LABELS[subset]}"
    )
    axis.grid(True, linewidth=0.45, alpha=0.30)
    axis.legend(frameon=True)
    save_figure(figure, output_path, dpi)


def annotated_heatmap(
    matrix: pd.DataFrame,
    title: str,
    colorbar_label: str,
    output_path: Path,
    dpi: int,
    center_zero: bool = False,
) -> None:
    if matrix.empty:
        return

    values = matrix.to_numpy(dtype=float)
    figure, axis = plt.subplots(
        figsize=(max(8, 1.25 * len(matrix.columns)), max(4.5, 0.75 * len(matrix.index)))
    )

    finite = values[np.isfinite(values)]
    if center_zero and len(finite):
        maximum = max(abs(float(finite.min())), abs(float(finite.max())))
        norm = TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum) if maximum > 0 else None
    else:
        norm = None

    image = axis.imshow(values, aspect="auto", norm=norm)
    axis.set_xticks(np.arange(len(matrix.columns)))
    axis.set_xticklabels([fmt_percent(value) for value in matrix.columns])
    axis.set_yticks(np.arange(len(matrix.index)))
    axis.set_yticklabels([model_label(value) for value in matrix.index])
    axis.set_xlabel("Removed hasAge relations (%)")
    axis.set_ylabel("Embedding model")
    axis.set_title(title)

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            if np.isfinite(value):
                axis.text(
                    column_index,
                    row_index,
                    fmt(value, 2),
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)
    save_figure(figure, output_path, dpi)


def plot_method_delta_heatmap(
    method_comparisons: pd.DataFrame,
    subset: str,
    window_condition: str,
    output_path: Path,
    dpi: int,
) -> None:
    results = method_comparisons.loc[
        (method_comparisons["subset"] == subset)
        & (method_comparisons["window_condition"] == window_condition)
    ]
    matrix = results.pivot(
        index="model",
        columns="removal_percent",
        values="mean_difference_qp_minus_lr",
    )
    matrix = matrix.reindex(ordered_models(matrix.index))
    annotated_heatmap(
        matrix,
        (
            f"QP − LR MAE — {WINDOW_LABELS[window_condition]} — "
            f"{SUBSET_LABELS[subset]}"
        ),
        "ΔMAE QP − LR (years); positive favors LR",
        output_path,
        dpi,
        center_zero=True,
    )


def plot_window_delta_heatmap(
    window_comparisons: pd.DataFrame,
    subset: str,
    method: str,
    output_path: Path,
    dpi: int,
) -> None:
    results = window_comparisons.loc[
        (window_comparisons["subset"] == subset)
        & (window_comparisons["method"] == method)
    ]
    matrix = results.pivot(
        index="model",
        columns="removal_percent",
        values="mean_difference_without_minus_with",
    )
    matrix = matrix.reindex(ordered_models(matrix.index))
    annotated_heatmap(
        matrix,
        (
            f"Without − With Windowing MAE — {METHOD_LABELS[method]} — "
            f"{SUBSET_LABELS[subset]}"
        ),
        "ΔMAE without − with (years); positive favors windowing",
        output_path,
        dpi,
        center_zero=True,
    )


def plot_auc_configuration_heatmap(
    robustness: pd.DataFrame,
    subset: str,
    output_path: Path,
    dpi: int,
) -> None:
    results = robustness.loc[robustness["subset"] == subset].copy()
    results["configuration"] = (
        results["method"].map(METHOD_LABELS)
        + " — "
        + results["window_condition"].map(WINDOW_LABELS)
    )
    matrix = results.pivot(
        index="model",
        columns="configuration",
        values="normalized_auc_mae",
    )
    matrix = matrix.reindex(ordered_models(matrix.index))
    matrix = matrix.reindex(
        [
            "Query Point — With Windowing",
            "Query Point — Without Windowing",
            "Learned Regression — With Windowing",
            "Learned Regression — Without Windowing",
        ],
        axis=1,
    )

    values = matrix.to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(12, 5.5))
    image = axis.imshow(values, aspect="auto")
    axis.set_xticks(np.arange(len(matrix.columns)))
    axis.set_xticklabels(matrix.columns, rotation=20, ha="right")
    axis.set_yticks(np.arange(len(matrix.index)))
    axis.set_yticklabels([model_label(value) for value in matrix.index])
    axis.set_title(f"Robustness Across Removal Levels — {SUBSET_LABELS[subset]}")
    axis.set_ylabel("Embedding model")

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            if np.isfinite(value):
                axis.text(
                    column_index,
                    row_index,
                    fmt(value, 2),
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Normalized AUC of MAE; lower is better")
    save_figure(figure, output_path, dpi)


def plot_top_configurations(
    summary: pd.DataFrame,
    robustness: pd.DataFrame,
    subset: str,
    output_path: Path,
    dpi: int,
    top_n: int = 6,
) -> None:
    top = (
        robustness.loc[robustness["subset"] == subset]
        .sort_values("normalized_auc_mae")
        .head(top_n)
    )
    if top.empty:
        return

    figure, axis = plt.subplots(figsize=(11, 7))
    for row in top.itertuples(index=False):
        results = summary.loc[
            (summary["subset"] == subset)
            & (summary["model"] == row.model)
            & (summary["method"] == row.method)
            & (summary["window_condition"] == row.window_condition)
        ].sort_values("removal_percent")
        axis.plot(
            results["removal_percent"],
            results["mae"],
            marker="o",
            linewidth=2,
            label=(
                f"{row.model_label} — {row.method_label} — {row.window_label}"
            ),
        )

    axis.set_xticks(sorted(summary.loc[summary["subset"] == subset, "removal_percent"].unique()))
    axis.set_xlabel("Removed hasAge relations (%)")
    axis.set_ylabel("Mean absolute error (years)")
    axis.set_title(f"Top Robust Configurations — {SUBSET_LABELS[subset]}")
    axis.grid(True, linewidth=0.45, alpha=0.30)
    axis.legend(frameon=True, fontsize=8)
    save_figure(figure, output_path, dpi)


def generate_plots(
    summary: pd.DataFrame,
    method_comparisons: pd.DataFrame,
    window_comparisons: pd.DataFrame,
    robustness: pd.DataFrame,
    plot_directory: Path,
    dpi: int,
) -> dict[str, Path]:
    plot_directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for subset in REPORT_SUBSETS:
        if summary.loc[summary["subset"] == subset].empty:
            continue

        for method in METHOD_LABELS:
            for window in WINDOW_LABELS:
                key = f"{subset}_{method}_{window}_model_lines"
                path = plot_directory / f"{key}.png"
                plot_model_lines(summary, subset, method, window, path, dpi)
                paths[key] = path

        for window in WINDOW_LABELS:
            key = f"{subset}_{window}_method_delta"
            path = plot_directory / f"{key}.png"
            plot_method_delta_heatmap(
                method_comparisons, subset, window, path, dpi
            )
            paths[key] = path

        for method in METHOD_LABELS:
            key = f"{subset}_{method}_window_delta"
            path = plot_directory / f"{key}.png"
            plot_window_delta_heatmap(
                window_comparisons, subset, method, path, dpi
            )
            paths[key] = path

        auc_key = f"{subset}_auc_configuration_heatmap"
        auc_path = plot_directory / f"{auc_key}.png"
        plot_auc_configuration_heatmap(robustness, subset, auc_path, dpi)
        paths[auc_key] = auc_path

        top_key = f"{subset}_top_configurations"
        top_path = plot_directory / f"{top_key}.png"
        plot_top_configurations(summary, robustness, subset, top_path, dpi)
        paths[top_key] = top_path

    return paths


# ============================================================
# Report insights and writing
# ============================================================


def count_comparison_outcomes(
    comparisons: pd.DataFrame,
    winner_column: str,
    winner_a: str,
    winner_b: str,
) -> dict[str, int]:
    return {
        "a": int((comparisons[winner_column] == winner_a).sum()),
        "b": int((comparisons[winner_column] == winner_b).sum()),
        "ties": int((comparisons[winner_column] == "Tie").sum()),
        "significant_a": int(
            (
                (comparisons[winner_column] == winner_a)
                & (comparisons["wilcoxon_p_holm"] < 0.05)
            ).sum()
        ),
        "significant_b": int(
            (
                (comparisons[winner_column] == winner_b)
                & (comparisons["wilcoxon_p_holm"] < 0.05)
            ).sum()
        ),
    }


def write_report(
    summary: pd.DataFrame,
    best_configurations: pd.DataFrame,
    robustness: pd.DataFrame,
    method_comparisons: pd.DataFrame,
    window_comparisons: pd.DataFrame,
    model_omnibus: pd.DataFrame,
    model_win_summary: pd.DataFrame,
    shared_provenance: pd.DataFrame,
    plot_paths: dict[str, Path],
    output_directory: Path,
    n_bootstrap: int,
) -> Path:
    report_path = output_directory / REPORT_FILE
    models = ordered_models(summary["model"].unique())
    population_size = int(summary["population_size"].iloc[0])
    removal_levels = sorted(summary["removal_percent"].unique())

    with report_path.open("w", encoding="utf-8") as file:
        file.write("# Comprehensive Embedding-Model Analysis\n\n")
        file.write(
            f"This report compares {', '.join(model_label(model) for model in models)} "
            f"on the same {population_size}-person KG conditions at removal levels "
            f"{', '.join(fmt_percent(value) for value in removal_levels)}. It evaluates "
            "embedding model, Query Point versus Learned Regression, and with-windowing "
            "versus without-windowing effects.\n\n"
        )

        file.write("## Data Integrity and Experimental Pairing\n\n")
        file.write(
            f"All model manifests reference the same {len(shared_provenance)} resolved KG "
            "files for the same run labels. Person identities, true ages, and removed-versus-"
            "retained `hasAge` membership were also checked across every model before any "
            "statistics were calculated.\n\n"
        )
        file.write(
            "All paired tests operate on the same people within the same KG condition. "
            f"Confidence intervals use {n_bootstrap:,} bootstrap samples. Pairwise tests "
            "include paired t-tests, Wilcoxon signed-rank tests, Cohen's $d_z$, person-level "
            "win rates, and Holm-adjusted p-values. Model-wide tests use the Friedman test "
            "with Kendall's $W$.\n\n"
        )

        for subset in REPORT_SUBSETS:
            subset_summary = summary.loc[summary["subset"] == subset]
            if subset_summary.empty:
                continue

            file.write(f"## {SUBSET_LABELS[subset]}\n\n")

            best_robust = (
                robustness.loc[robustness["subset"] == subset]
                .sort_values("normalized_auc_mae")
                .iloc[0]
            )
            model_wins = (
                model_win_summary.loc[model_win_summary["subset"] == subset]
                .sort_values(["mean_model_rank", "mean_mae"])
                .iloc[0]
            )
            method_outcomes = count_comparison_outcomes(
                method_comparisons.loc[method_comparisons["subset"] == subset],
                "lower_mae_method",
                METHOD_LABELS["query_point"],
                METHOD_LABELS["learned_regression"],
            )
            window_outcomes = count_comparison_outcomes(
                window_comparisons.loc[window_comparisons["subset"] == subset],
                "lower_mae_window_condition",
                WINDOW_LABELS["without_windows"],
                WINDOW_LABELS["with_windows"],
            )
            omnibus_subset = model_omnibus.loc[model_omnibus["subset"] == subset]
            significant_omnibus = int((omnibus_subset["friedman_p"] < 0.05).sum())

            file.write("### High-Level Findings\n\n")
            file.write(
                f"- The lowest normalized MAE area under the removal curve was produced by "
                f"**{best_robust['model_label']} with {best_robust['method_label']} and "
                f"{best_robust['window_label']}** "
                f"({fmt(best_robust['normalized_auc_mae'])} years).\n"
            )
            file.write(
                f"- **{model_wins['model_label']}** had the strongest average model rank "
                f"({fmt(model_wins['mean_model_rank'])}) across method, window, and removal "
                "conditions.\n"
            )
            file.write(
                f"- Query Point had lower MAE in {method_outcomes['a']} model-condition "
                f"comparisons; Learned Regression had lower MAE in {method_outcomes['b']}. "
                f"After Holm correction, the corresponding significant counts were "
                f"{method_outcomes['significant_a']} and {method_outcomes['significant_b']}.\n"
            )
            file.write(
                f"- Without windowing had lower MAE in {window_outcomes['a']} comparisons; "
                f"with windowing had lower MAE in {window_outcomes['b']}. After Holm "
                f"correction, the significant counts were {window_outcomes['significant_a']} "
                f"and {window_outcomes['significant_b']}.\n"
            )
            file.write(
                f"- The Friedman model comparison was significant in {significant_omnibus} "
                f"of {len(omnibus_subset)} method-window-removal conditions.\n\n"
            )

            file.write("### Core Performance Matrices\n\n")
            file.write(
                "Each model cell is `QP MAE / LR MAE` in years. Lower values are better.\n\n"
            )
            for window in WINDOW_LABELS:
                file.write(f"#### {WINDOW_LABELS[window]}\n\n")
                file.write(
                    dataframe_to_markdown(
                        performance_markdown_table(summary, subset, window)
                    )
                )
                file.write("\n")

            file.write("### Best Overall Configuration at Each Removal Level\n\n")
            file.write(
                dataframe_to_markdown(
                    best_configurations_markdown_table(best_configurations, subset)
                )
            )
            file.write("\n")

            file.write("### Robustness Across Removal Levels\n\n")
            file.write(
                "Normalized AUC summarizes MAE across the observed removal range; lower is "
                "better. The missing-only subset begins at the first nonempty removal level.\n\n"
            )
            file.write(
                dataframe_to_markdown(robustness_markdown_table(robustness, subset))
            )
            file.write("\n")

            auc_plot = plot_paths.get(f"{subset}_auc_configuration_heatmap")
            if auc_plot:
                file.write(
                    f"![Robustness heatmap]({PLOT_DIRECTORY_NAME}/{auc_plot.name})\n\n"
                )
            top_plot = plot_paths.get(f"{subset}_top_configurations")
            if top_plot:
                file.write(
                    f"![Top configurations]({PLOT_DIRECTORY_NAME}/{top_plot.name})\n\n"
                )

            file.write("### Query Point Versus Learned Regression\n\n")
            file.write(
                "The matrices report `QP MAE − LR MAE`; positive values favor Learned "
                "Regression, and negative values favor Query Point.\n\n"
            )
            for window in WINDOW_LABELS:
                file.write(f"#### {WINDOW_LABELS[window]}\n\n")
                file.write(
                    dataframe_to_markdown(
                        delta_matrix_table(
                            method_comparisons,
                            subset,
                            category="method",
                            fixed_value=window,
                        )
                    )
                )
                file.write("\n")
                plot = plot_paths.get(f"{subset}_{window}_method_delta")
                if plot:
                    file.write(
                        f"![QP versus LR heatmap]({PLOT_DIRECTORY_NAME}/{plot.name})\n\n"
                    )

            file.write("### Window-Structure Effect\n\n")
            file.write(
                "The matrices report `without-windowing MAE − with-windowing MAE`; positive "
                "values favor retaining the window structure, and negative values favor "
                "removing it.\n\n"
            )
            for method in METHOD_LABELS:
                file.write(f"#### {METHOD_LABELS[method]}\n\n")
                file.write(
                    dataframe_to_markdown(
                        delta_matrix_table(
                            window_comparisons,
                            subset,
                            category="window",
                            fixed_value=method,
                        )
                    )
                )
                file.write("\n")
                plot = plot_paths.get(f"{subset}_{method}_window_delta")
                if plot:
                    file.write(
                        f"![Window effect heatmap]({PLOT_DIRECTORY_NAME}/{plot.name})\n\n"
                    )

            file.write("### Model-Specific Performance Curves\n\n")
            for method in METHOD_LABELS:
                for window in WINDOW_LABELS:
                    plot = plot_paths.get(f"{subset}_{method}_{window}_model_lines")
                    if not plot:
                        continue
                    file.write(
                        f"#### {METHOD_LABELS[method]} — {WINDOW_LABELS[window]}\n\n"
                    )
                    file.write(
                        f"![Model performance curves]({PLOT_DIRECTORY_NAME}/{plot.name})\n\n"
                    )

            file.write("### Omnibus Differences Among Embedding Models\n\n")
            file.write(
                dataframe_to_markdown(
                    model_omnibus_markdown_table(model_omnibus, subset)
                )
            )
            file.write("\n")

        file.write("## Additional Machine-Readable Results\n\n")
        file.write(
            "The retained-only subset and all full-resolution pairwise tests are included in "
            "the CSV outputs even though the report emphasizes all-person and missing-only "
            "results.\n\n"
        )
        file.write(f"- `{CONDITION_SUMMARY_FILE}`: full descriptive results.\n")
        file.write(f"- `{PERFORMANCE_MATRIX_FILE}`: wide MAE matrix.\n")
        file.write(
            f"- `{CONFIGURATION_RANKINGS_FILE}`: every model-method-window configuration "
            "ranked within each removal level.\n"
        )
        file.write(f"- `{ROBUSTNESS_FILE}`: AUC, degradation, slope, and ranks.\n")
        file.write(f"- `{BEST_CONFIGURATIONS_FILE}`: winner and runner-up by removal.\n")
        file.write(f"- `{METHOD_COMPARISONS_FILE}`: paired QP-versus-LR tests.\n")
        file.write(f"- `{WINDOW_COMPARISONS_FILE}`: paired window-ablation tests.\n")
        file.write(f"- `{MODEL_OMNIBUS_FILE}`: Friedman tests and Kendall's W.\n")
        file.write(f"- `{MODEL_PAIRWISE_FILE}`: all paired model-versus-model tests.\n")
        file.write(f"- `{MODEL_WIN_SUMMARY_FILE}`: model win/loss and rank summary.\n")
        file.write(f"- `{PROVENANCE_FILE}`: input folders and shared-KG verification.\n\n")

        file.write("## Metric Notes\n\n")
        file.write("- **MAE:** mean absolute age-prediction error in years.\n")
        file.write("- **RMSE:** root mean squared signed error in years.\n")
        file.write("- **QP − LR:** positive values mean Learned Regression has lower MAE.\n")
        file.write(
            "- **Without − With:** positive values mean the with-windowing condition has "
            "lower MAE.\n"
        )
        file.write(
            "- **Normalized AUC MAE:** trapezoidal area under MAE across removal percentage, "
            "divided by the observed removal range; lower is better.\n"
        )
        file.write(
            "- **Cohen's dz:** mean paired difference divided by its sample standard "
            "deviation.\n"
        )
        file.write(
            "- **Kendall's W:** effect size for the Friedman repeated-measures comparison "
            "among embedding models.\n"
        )
        file.write(
            "- **Holm-adjusted p:** familywise-error correction applied within the relevant "
            "method, window, removal, and subset comparison family.\n"
        )

    return report_path


# ============================================================
# Main analysis pipeline
# ============================================================


def generate_analysis(
    model_folders: list[str | Path],
    output_directory: str | Path,
    seed: int = 42,
    n_bootstrap: int = 5000,
    dpi: int = 300,
    force: bool = False,
    verify_hashes: bool = True,
) -> None:
    output_directory = Path(output_directory)
    prepare_output(output_directory, force)

    loaded = [load_model_folder(path) for path in model_folders]
    model_manifests = {model: manifest for model, manifest, _ in loaded}

    if len(model_manifests) != len(loaded):
        models = [model for model, _, _ in loaded]
        raise ValueError(f"Duplicate model folders were supplied: {models}")

    shared_provenance = validate_shared_manifests(
        model_manifests=model_manifests,
        verify_hashes=verify_hashes,
    )

    combined = pd.concat([dataframe for _, _, dataframe in loaded], ignore_index=True)
    validate_prediction_alignment(combined)
    data = add_subsets(combined)

    summary = summarize_conditions(data)
    performance_matrix = build_performance_matrix(summary)
    configuration_rankings = build_configuration_rankings(summary)
    best_configurations = build_best_configurations(configuration_rankings)
    robustness = summarize_robustness(summary)

    method_comparisons = compare_methods(data, seed, n_bootstrap)
    window_comparisons = compare_windows(data, seed, n_bootstrap)
    model_omnibus, model_pairwise = compare_models(data, seed, n_bootstrap)
    model_win_summary = summarize_model_wins(summary, model_pairwise)

    summary.to_csv(output_directory / CONDITION_SUMMARY_FILE, index=False)
    performance_matrix.to_csv(output_directory / PERFORMANCE_MATRIX_FILE, index=False)
    configuration_rankings.to_csv(
        output_directory / CONFIGURATION_RANKINGS_FILE, index=False
    )
    robustness.to_csv(output_directory / ROBUSTNESS_FILE, index=False)
    best_configurations.to_csv(
        output_directory / BEST_CONFIGURATIONS_FILE, index=False
    )
    method_comparisons.to_csv(
        output_directory / METHOD_COMPARISONS_FILE, index=False
    )
    window_comparisons.to_csv(
        output_directory / WINDOW_COMPARISONS_FILE, index=False
    )
    model_omnibus.to_csv(output_directory / MODEL_OMNIBUS_FILE, index=False)
    model_pairwise.to_csv(output_directory / MODEL_PAIRWISE_FILE, index=False)
    model_win_summary.to_csv(output_directory / MODEL_WIN_SUMMARY_FILE, index=False)

    plots = generate_plots(
        summary=summary,
        method_comparisons=method_comparisons,
        window_comparisons=window_comparisons,
        robustness=robustness,
        plot_directory=output_directory / PLOT_DIRECTORY_NAME,
        dpi=dpi,
    )

    provenance_payload = {
        "model_folders": [str(Path(path).resolve()) for path in model_folders],
        "models": ordered_models(model_manifests),
        "shared_kg_verification": {
            "same_resolved_paths_across_models": True,
            "sha256_checked": verify_hashes,
            "kg_files": json.loads(shared_provenance.to_json(orient="records")),
        },
        "population_size": int(summary["population_size"].iloc[0]),
        "removal_percentages": sorted(
            float(value) for value in summary["removal_percent"].unique()
        ),
        "bootstrap_samples": n_bootstrap,
        "seed": seed,
    }
    with (output_directory / PROVENANCE_FILE).open("w", encoding="utf-8") as outfile:
        json.dump(provenance_payload, outfile, indent=2)

    report = write_report(
        summary=summary,
        best_configurations=best_configurations,
        robustness=robustness,
        method_comparisons=method_comparisons,
        window_comparisons=window_comparisons,
        model_omnibus=model_omnibus,
        model_win_summary=model_win_summary,
        shared_provenance=shared_provenance,
        plot_paths=plots,
        output_directory=output_directory,
        n_bootstrap=n_bootstrap,
    )

    print("Validated that every model references the same shared KG files.")
    print("Loaded model folders: " + ", ".join(str(path) for path in model_folders))
    print(f"Saved report: {report}")
    print(f"Saved tables: {output_directory}")
    print(f"Saved plots: {output_directory / PLOT_DIRECTORY_NAME}")


# ============================================================
# Command-line interface
# ============================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Comprehensively compare embedding models, Query Point versus Learned "
            "Regression, and with-windowing versus without-windowing conditions."
        )
    )
    parser.add_argument(
        "--basepath",
        nargs="+",
        help=(
            "Explicit completed model folders. When omitted, model folders are "
            "discovered beneath --root."
        ),
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help=f"Shared multi-model experiment root. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--output-directory",
        default=str(DEFAULT_OUTPUT),
        help=f"Output directory. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-kg-hash-check",
        action="store_true",
        help=(
            "Still require identical resolved KG paths across manifests, but skip "
            "recomputing SHA-256 hashes."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_folders = (
        [Path(path) for path in args.basepath]
        if args.basepath
        else discover_model_folders(args.root)
    )

    if not args.basepath:
        print(
            "Automatically discovered model folders: "
            + ", ".join(str(path) for path in model_folders)
        )

    generate_analysis(
        model_folders=model_folders,
        output_directory=args.output_directory,
        seed=args.seed,
        n_bootstrap=args.bootstrap,
        dpi=args.dpi,
        force=args.force,
        verify_hashes=not args.skip_kg_hash_check,
    )


if __name__ == "__main__":
    main()
