from __future__ import annotations

import argparse
import random
import re
import shutil
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from kg_creator import create_kg
from kg_modifier import (
    create_paired_percentage_variants,
    discover_window_entities,
    read_kg,
    remove_windows,
)
from embeddings_helper import KastleEmbeddings
from evaluate_query_points import evaluate_query_point_run
from analyze_data import analyze_data
from analyze_with_learning import analyze_with_learning
from compare_methods import compare_methods, compare_window_conditions
from view_results import generate_report


STEPS = (
    "kg",
    "embeddings",
    "query-points",
    "analysis",
    "comparisons",
    "report",
)

MANIFEST_FILENAME = "kg_manifest.csv"
REMOVAL_PLAN_FILENAME = "hasAge_removal_plan.csv"
GLOBAL_ARGS = None


@dataclass(frozen=True)
class KGRun:
    label: str
    tsv_path: Path
    window_condition: str
    removal_percent: float
    relation: str
    seed: int
    nested_removals: bool
    sort_order: int


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def percentage_slug(value: float) -> str:
    text = f"{value:.10g}".replace(".", "p")
    return f"{text}pct"


def safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return token or "relation"


def get_removal_percentages(args) -> list[float]:
    if args.n_variations < 1:
        raise ValueError("--n-variations must be at least 1.")
    if args.removal_percent_step <= 0:
        raise ValueError("--removal-percent-step must be greater than 0.")

    percentages = [
        args.removal_percent_step * index
        for index in range(1, args.n_variations + 1)
    ]

    if percentages[-1] > 100:
        raise ValueError(
            f"Largest removal percentage is {percentages[-1]:g}%, above 100%."
        )

    return percentages


def build_run_plan(args) -> list[KGRun]:
    basepath = Path(args.basepath)
    kg_dir = basepath / "kgs"
    percentages = get_removal_percentages(args)
    relation_token = safe_token(args.relation)
    nested = not args.independent_removals

    runs: list[KGRun] = []
    sort_order = 0

    for condition in ("with_windows", "without_windows"):
        runs.append(
            KGRun(
                label=f"{condition}_original",
                tsv_path=kg_dir / f"{condition}_original.tsv",
                window_condition=condition,
                removal_percent=0.0,
                relation=args.relation,
                seed=args.seed,
                nested_removals=nested,
                sort_order=sort_order,
            )
        )
        sort_order += 1

    for percentage in percentages:
        slug = percentage_slug(percentage)
        for condition in ("with_windows", "without_windows"):
            runs.append(
                KGRun(
                    label=f"{condition}_removed_{slug}",
                    tsv_path=(
                        kg_dir
                        / f"{condition}_{relation_token}_removed_{slug}.tsv"
                    ),
                    window_condition=condition,
                    removal_percent=percentage,
                    relation=args.relation,
                    seed=args.seed,
                    nested_removals=nested,
                    sort_order=sort_order,
                )
            )
            sort_order += 1

    return runs


def save_manifest(args, runs: Sequence[KGRun]) -> Path:
    basepath = Path(args.basepath)
    manifest_path = basepath / MANIFEST_FILENAME

    rows = []
    for run in runs:
        rows.append(
            {
                "label": run.label,
                "tsv_path": str(run.tsv_path.relative_to(basepath)),
                "window_condition": run.window_condition,
                "removal_percent": run.removal_percent,
                "relation": run.relation,
                "seed": run.seed,
                "nested_removals": run.nested_removals,
                "sort_order": run.sort_order,
            }
        )

    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"Saved KG manifest: {manifest_path}")
    return manifest_path


def load_manifest(args) -> list[KGRun]:
    basepath = Path(args.basepath)
    manifest_path = basepath / MANIFEST_FILENAME

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Start from --start-step kg first."
        )

    manifest = pd.read_csv(manifest_path)
    required = {
        "label",
        "tsv_path",
        "window_condition",
        "removal_percent",
        "relation",
        "seed",
        "nested_removals",
        "sort_order",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing)}"
        )

    if manifest["label"].duplicated().any():
        raise ValueError("Manifest contains duplicate run labels.")

    manifest = manifest.sort_values("sort_order")
    runs = [
        KGRun(
            label=str(row.label),
            tsv_path=basepath / str(row.tsv_path),
            window_condition=str(row.window_condition),
            removal_percent=float(row.removal_percent),
            relation=str(row.relation),
            seed=int(row.seed),
            nested_removals=str(row.nested_removals).lower()
            in {"true", "1", "yes"},
            sort_order=int(row.sort_order),
        )
        for row in manifest.itertuples(index=False)
    ]

    relation_values = {run.relation for run in runs}
    if len(relation_values) == 1:
        args.relation = next(iter(relation_values))

    return runs


def validate_tsv(path: str | Path) -> None:
    read_kg(path)


def _baseline_run(runs: Sequence[KGRun], condition: str) -> KGRun:
    matches = [
        run
        for run in runs
        if run.window_condition == condition and run.removal_percent == 0
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one original run for {condition}; found {len(matches)}."
        )
    return matches[0]


def _relation_set(path: Path, relation: str) -> set[tuple[str, str, str]]:
    return {
        triple
        for triple in read_kg(path)
        if triple[1] == relation
    }


def validate_kg_runs(args, runs: Sequence[KGRun]) -> None:
    print("\nValidating paired KG files")

    for run in runs:
        validate_tsv(run.tsv_path)
        print(
            f"  OK: {run.label} "
            f"({run.window_condition}, {run.removal_percent:g}%)"
        )

    with_original = _baseline_run(runs, "with_windows")
    without_original = _baseline_run(runs, "without_windows")

    with_targets = _relation_set(with_original.tsv_path, args.relation)
    without_targets = _relation_set(without_original.tsv_path, args.relation)
    if with_targets != without_targets:
        raise ValueError(
            "Original with-windows and without-windows KGs do not share the "
            f"same {args.relation} triples."
        )

    without_triples = read_kg(without_original.tsv_path)
    window_entities = discover_window_entities(
        without_triples,
        window_relation=args.window_relation,
        window_prefixes=args.window_prefix,
    )
    if window_entities or any(
        relation == args.window_relation
        for _, relation, _ in without_triples
    ):
        raise ValueError(
            "The without-windows baseline still contains window structure."
        )

    percentages = sorted(
        {
            run.removal_percent
            for run in runs
            if run.removal_percent > 0
        }
    )

    for percentage in percentages:
        with_run = next(
            run
            for run in runs
            if run.window_condition == "with_windows"
            and run.removal_percent == percentage
        )
        without_run = next(
            run
            for run in runs
            if run.window_condition == "without_windows"
            and run.removal_percent == percentage
        )

        removed_with = with_targets - _relation_set(
            with_run.tsv_path,
            args.relation,
        )
        removed_without = without_targets - _relation_set(
            without_run.tsv_path,
            args.relation,
        )

        if removed_with != removed_without:
            raise ValueError(
                f"Paired {percentage:g}% runs removed different "
                f"{args.relation} triples."
            )


def clear_generated_outputs(basepath: Path) -> None:
    for relative in (
        "kgs",
        "runs",
        MANIFEST_FILENAME,
        REMOVAL_PLAN_FILENAME,
        "query_point_distances_all_runs.csv",
        "person_year_predictions.csv",
        "run_year_summary.csv",
        "run_year_summary_missing_only.csv",
        "learning_person_predictions.csv",
        "learning_run_summary.csv",
        "learning_run_summary_missing_only.csv",
        "method_comparison.csv",
        "method_comparison_missing_only.csv",
        "query_window_comparison.csv",
        "query_window_comparison_missing_only.csv",
        "learning_window_comparison.csv",
        "learning_window_comparison_missing_only.csv",
        "results_report.md",
        "kg.tsv",
    ):
        path = basepath / relative
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def create_kgs(args) -> list[KGRun]:
    basepath = Path(args.basepath)
    ensure_dir(basepath)

    if (basepath / MANIFEST_FILENAME).exists() and not args.force:
        raise FileExistsError(
            f"Experiment outputs already exist in {basepath}. Use --force to "
            "regenerate from the KG step."
        )

    if args.force:
        clear_generated_outputs(basepath)

    runs = build_run_plan(args)
    ensure_dir(basepath / "kgs")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("\nCreating one source KG with window structure")
    create_kg(
        n_vertices=args.n_vertices,
        n_people=args.n_people,
        window_depth=args.window_depth,
        decimal_precision=args.decimal_precision,
        high=args.high,
        low=args.low,
        experiment_folder=str(basepath),
        include_windowing=True,
    )

    generated = basepath / "kg.tsv"
    with_original = _baseline_run(runs, "with_windows")
    without_original = _baseline_run(runs, "without_windows")

    if not generated.exists():
        raise FileNotFoundError(
            f"create_kg() did not create expected file: {generated}"
        )

    generated.replace(with_original.tsv_path)
    print(f"Preserved with-windows baseline: {with_original.tsv_path}")

    print("\nCreating paired windowless baseline")
    remove_windows(
        input_path=with_original.tsv_path,
        output_path=without_original.tsv_path,
        window_relation=args.window_relation,
        window_prefixes=args.window_prefix,
    )

    percentages = get_removal_percentages(args)
    with_outputs = [
        next(
            run.tsv_path
            for run in runs
            if run.window_condition == "with_windows"
            and run.removal_percent == percentage
        )
        for percentage in percentages
    ]
    without_outputs = [
        next(
            run.tsv_path
            for run in runs
            if run.window_condition == "without_windows"
            and run.removal_percent == percentage
        )
        for percentage in percentages
    ]

    print("\nCreating paired random hasAge-removal variants")
    create_paired_percentage_variants(
        with_windows_path=with_original.tsv_path,
        without_windows_path=without_original.tsv_path,
        percentages=percentages,
        with_windows_output_paths=with_outputs,
        without_windows_output_paths=without_outputs,
        relation_string=args.relation,
        seed=args.seed,
        nested=not args.independent_removals,
        removal_plan_path=basepath / REMOVAL_PLAN_FILENAME,
    )

    save_manifest(args, runs)
    validate_kg_runs(args, runs)
    return runs


def init_worker(args) -> None:
    global GLOBAL_ARGS
    GLOBAL_ARGS = args


def run_embedding_job(run: KGRun) -> None:
    global GLOBAL_ARGS
    if GLOBAL_ARGS is None:
        raise RuntimeError("Embedding worker was not initialized.")

    args = GLOBAL_ARGS
    torch.set_num_threads(1)
    run_folder = Path(args.basepath) / "runs" / run.label

    if run_folder.exists() and args.force:
        shutil.rmtree(run_folder)
    ensure_dir(run_folder)

    print(f"Creating embeddings for {run.label}")
    experiment = KastleEmbeddings(
        seed=args.seed,
        experiment_folder=str(run_folder),
        tsv_path=str(run.tsv_path),
        emb_model=args.embedding_model,
        emb_epochs=args.embedding_epochs,
        emb_dimensions=args.embedding_dimensions,
        pca=False,
        tsne=False,
        umap=False,
        dr_components=2,
        plot_name=run.label,
    )
    experiment.create_pykeen_embeddings()


def _embedding_folder_has_output(path: Path) -> bool:
    if not path.is_dir():
        return False
    ignored = {"query_point_distances.csv"}
    return any(
        child.is_file() and child.name not in ignored
        for child in path.rglob("*")
    )


def validate_embedding_runs(args, runs: Sequence[KGRun]) -> None:
    missing = []
    for run in runs:
        path = Path(args.basepath) / "runs" / run.label
        if not _embedding_folder_has_output(path):
            missing.append(path)

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(
            f"Missing or empty embedding run folders:\n{formatted}"
        )


def run_embeddings(args, runs: Sequence[KGRun]) -> None:
    validate_kg_runs(args, runs)

    if args.processes <= 1:
        init_worker(args)
        for run in runs:
            run_embedding_job(run)
    else:
        with Pool(
            processes=args.processes,
            initializer=init_worker,
            initargs=(args,),
        ) as pool:
            pool.map(run_embedding_job, runs)

    validate_embedding_runs(args, runs)


def run_query_point_evaluations(args, runs):
    basepath = Path(args.basepath)

    all_dfs = []

    for run in runs:
        original_candidates = [
            candidate
            for candidate in runs
            if (
                candidate.window_condition == run.window_condition
                and candidate.removal_percent == 0
            )
        ]

        if len(original_candidates) != 1:
            raise ValueError(
                "Expected exactly one original run for "
                f"{run.window_condition!r}; "
                f"found {len(original_candidates)}."
            )

        original_run = original_candidates[0]

        print(
            f"\nEvaluating query points for {run.label}"
        )

        original_run_path = (
            basepath
            / "runs"
            / original_run.label
        )

        evaluated_run_path = (
            basepath
            / "runs"
            / run.label
        )

        output_csv_path = (
            evaluated_run_path
            / "query_point_distances.csv"
        )

        df = evaluate_query_point_run(
            original_run_path=str(
                original_run_path
            ),
            evaluated_run_path=str(
                evaluated_run_path
            ),

            # Truth comes from the corresponding 0%-removal KG.
            original_tsv_path=str(
                original_run.tsv_path
            ),

            # Missingness comes from the current evaluated KG.
            evaluated_tsv_path=str(
                run.tsv_path
            ),

            output_csv_path=str(
                output_csv_path
            ),
        )

        df["experiment"] = (
            args.experiment_name
        )

        df["run"] = run.label

        df["window_condition"] = (
            run.window_condition
        )

        df["removal_percent"] = (
            run.removal_percent
        )

        df["removed_relation"] = (
            args.relation
        )

        df["nested_removals"] = (
            not args.independent_removals
        )

        all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError(
            "Query-point evaluation produced no results."
        )

    full_df = pd.concat(
        all_dfs,
        ignore_index=True,
    )

    full_output_path = (
        basepath
        / "query_point_distances_all_runs.csv"
    )

    full_df.to_csv(
        full_output_path,
        index=False,
    )

    print(
        "\nSaved all query-point evaluations to: "
        f"{full_output_path}"
    )

    return full_df


def validate_query_output(args) -> Path:
    path = Path(args.basepath) / "query_point_distances_all_runs.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Query-point result file is empty: {path}")
    return path


def run_analyses(args, runs: Sequence[KGRun]) -> None:
    basepath = Path(args.basepath)
    query_csv = validate_query_output(args)

    analyze_data(
        csv_path=query_csv,
        person_output=basepath / "person_year_predictions.csv",
        run_output=basepath / "run_year_summary.csv",
        missing_run_output=basepath / "run_year_summary_missing_only.csv",
        max_k=args.max_k,
    )

    analyze_with_learning(
        basepath=basepath,
        run_labels=[run.label for run in runs],
        person_model_type=args.regression_model,
        person_output=basepath / "learning_person_predictions.csv",
        run_output=basepath / "learning_run_summary.csv",
        missing_run_output=basepath
        / "learning_run_summary_missing_only.csv",
    )


def validate_analysis_outputs(args) -> None:
    basepath = Path(args.basepath)
    required = [
        basepath / "person_year_predictions.csv",
        basepath / "learning_person_predictions.csv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing analysis outputs:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )


def run_comparisons(args) -> None:
    basepath = Path(args.basepath)
    validate_analysis_outputs(args)

    query_person = basepath / "person_year_predictions.csv"
    learning_person = basepath / "learning_person_predictions.csv"

    compare_methods(
        query_person,
        learning_person,
        basepath / "method_comparison.csv",
        missing_only=False,
        seed=args.seed,
    )
    compare_methods(
        query_person,
        learning_person,
        basepath / "method_comparison_missing_only.csv",
        missing_only=True,
        seed=args.seed,
    )

    compare_window_conditions(
        query_person,
        "query_point",
        basepath / "query_window_comparison.csv",
        missing_only=False,
        seed=args.seed,
    )
    compare_window_conditions(
        query_person,
        "query_point",
        basepath / "query_window_comparison_missing_only.csv",
        missing_only=True,
        seed=args.seed,
    )
    compare_window_conditions(
        learning_person,
        "learned_regression",
        basepath / "learning_window_comparison.csv",
        missing_only=False,
        seed=args.seed,
    )
    compare_window_conditions(
        learning_person,
        "learned_regression",
        basepath / "learning_window_comparison_missing_only.csv",
        missing_only=True,
        seed=args.seed,
    )


def validate_comparison_outputs(args) -> None:
    basepath = Path(args.basepath)
    required = [
        basepath / "method_comparison.csv",
        basepath / "method_comparison_missing_only.csv",
        basepath / "query_window_comparison.csv",
        basepath / "query_window_comparison_missing_only.csv",
        basepath / "learning_window_comparison.csv",
        basepath / "learning_window_comparison_missing_only.csv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing comparison outputs:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )


def step_range(args) -> tuple[int, int]:
    start = STEPS.index(args.start_step)
    stop = STEPS.index(args.stop_after)
    if start > stop:
        raise ValueError("--start-step occurs after --stop-after.")
    return start, stop


def should_run(step: str, start: int, stop: int) -> bool:
    index = STEPS.index(step)
    return start <= index <= stop


def run_pipeline(args) -> None:
    basepath = Path(args.basepath)
    ensure_dir(basepath)

    if args.experiment_name is None:
        args.experiment_name = basepath.name

    start, stop = step_range(args)

    print(f"Experiment:       {args.experiment_name}")
    print(f"Base path:        {basepath}")
    print(f"Starting step:    {args.start_step}")
    print(f"Stopping after:   {args.stop_after}")

    runs = None

    if should_run("kg", start, stop):
        print("\nStep 1 — Creating paired KG conditions")
        runs = create_kgs(args)

    if runs is None:
        runs = load_manifest(args)

    if should_run("embeddings", start, stop):
        print("\nStep 2 — Creating embeddings")
        run_embeddings(args, runs)

    if should_run("query-points", start, stop):
        print("\nStep 3 — Running query-point evaluations")
        run_query_point_evaluations(args, runs)

    if should_run("analysis", start, stop):
        print("\nStep 4 — Running method analyses")
        run_analyses(args, runs)

    if should_run("comparisons", start, stop):
        print("\nStep 5 — Running paired comparisons")
        run_comparisons(args)

    if should_run("report", start, stop):
        print("\nStep 6 — Generating report")
        validate_comparison_outputs(args)
        generate_report(
            basepath=basepath,
            output_md=basepath / "results_report.md",
        )

    print("\nDone.")
    print(f"Results folder: {basepath}")
    report = basepath / "results_report.md"
    if report.exists():
        print(f"Report: {report}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-people", type=int, default=100)
    parser.add_argument("--n-vertices", type=int, default=100)
    parser.add_argument("--basepath", default=None)
    parser.add_argument("--experiment-name", default=None)

    parser.add_argument("--embedding-dimensions", type=int, default=300)
    parser.add_argument("--embedding-epochs", type=int, default=100)
    parser.add_argument("--embedding-model", default="mure")

    parser.add_argument("--window-depth", type=int, default=4)
    parser.add_argument("--window-relation", default="inWindow")
    parser.add_argument(
        "--window-prefix",
        action="append",
        default=None,
        help=(
            "Window entity prefix. Repeat for multiple prefixes. Defaults to "
            "Window_, window_, and window."
        ),
    )

    parser.add_argument("--decimal-precision", type=int, default=0)
    parser.add_argument("--high", type=int, default=100)
    parser.add_argument("--low", type=int, default=1)

    parser.add_argument("--relation", default="hasAge")
    parser.add_argument("--n-variations", type=int, default=5)
    parser.add_argument("--removal-percent-step", type=float, default=15.0)
    parser.add_argument("--independent-removals", action="store_true")

    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument(
        "--regression-model",
        default="ridge",
        choices=["ridge", "mlp"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processes", type=int, default=8)

    parser.add_argument(
        "--start-step",
        choices=STEPS,
        default="kg",
    )
    parser.add_argument(
        "--stop-after",
        choices=STEPS,
        default="report",
    )
    parser.add_argument("--force", action="store_true")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.basepath is None:
        args.basepath = f"./{args.n_people}people_window_comparison"

    if args.window_prefix is None:
        args.window_prefix = ["Window_", "window_", "window"]

    run_pipeline(args)
