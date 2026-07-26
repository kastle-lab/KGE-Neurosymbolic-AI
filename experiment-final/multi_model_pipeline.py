from __future__ import annotations

import argparse
import copy
import hashlib
import os
import random
import shutil
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

import pipeline as core
from embeddings_helper import KastleEmbeddings


DEFAULT_MODELS = ("transe", "distmult", "transd", "transr", "mure")
SHARED_DIRECTORY_NAME = "shared_kgs"
HASH_FILENAME = "shared_kg_hashes.csv"


@dataclass(frozen=True)
class EmbeddingJob:
    model: str
    run_label: str
    tsv_path: str
    run_folder: str
    seed: int
    epochs: int
    dimensions: int
    force: bool
    resume: bool


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_model_name(value: str) -> str:
    token = core.safe_token(str(value)).lower()
    if not token:
        raise ValueError(f"Invalid model name: {value!r}")
    return token


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as infile:
        for block in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def embedding_folder_has_output(path: str | Path) -> bool:
    path = Path(path)
    if not path.is_dir():
        return False

    ignored = {"query_point_distances.csv"}
    return any(
        child.is_file() and child.name not in ignored
        for child in path.rglob("*")
    )


def train_embedding_job(job: EmbeddingJob) -> str:
    """Train one model on one already-generated KG."""
    run_folder = Path(job.run_folder)

    if embedding_folder_has_output(run_folder):
        if job.resume:
            return f"SKIPPED existing: {job.model}/{job.run_label}"
        if not job.force:
            raise FileExistsError(
                f"Embedding output already exists: {run_folder}"
            )

    if run_folder.exists() and job.force:
        shutil.rmtree(run_folder)

    ensure_dir(run_folder)

    # Keep each worker from spawning a large internal CPU thread pool.
    torch.set_num_threads(1)
    random.seed(job.seed)
    np.random.seed(job.seed)
    torch.manual_seed(job.seed)

    print(f"Training {job.model}: {job.run_label}", flush=True)

    experiment = KastleEmbeddings(
        seed=job.seed,
        experiment_folder=str(run_folder),
        tsv_path=job.tsv_path,
        emb_model=job.model,
        emb_epochs=job.epochs,
        emb_dimensions=job.dimensions,
        pca=False,
        tsne=False,
        umap=False,
        dr_components=2,
        plot_name=f"{job.model}_{job.run_label}",
    )
    experiment.create_pykeen_embeddings()

    if not embedding_folder_has_output(run_folder):
        raise RuntimeError(
            f"Embedding job completed without producing output: {run_folder}"
        )

    return f"DONE: {job.model}/{job.run_label}"


def prepare_root(args) -> tuple[Path, Path]:
    root = Path(args.output_root).resolve()
    shared_dir = root / SHARED_DIRECTORY_NAME

    if args.force and args.resume:
        raise ValueError("Use either --force or --resume, not both.")

    if root.exists() and not args.force and not args.resume:
        raise FileExistsError(
            f"Output root already exists: {root}\n"
            "Choose another --output-root, use --resume, or use --force."
        )

    if args.force and root.exists():
        shutil.rmtree(root)

    ensure_dir(root)
    ensure_dir(shared_dir)
    return root, shared_dir


def build_shared_args(args, shared_dir: Path):
    shared_args = copy.copy(args)
    shared_args.basepath = str(shared_dir)
    shared_args.experiment_name = f"{args.n_people}people_shared_kg"
    shared_args.embedding_model = "shared_only"
    shared_args.processes = 1
    shared_args.start_step = "kg"
    shared_args.stop_after = "kg"
    shared_args.force = False
    return shared_args


def create_or_load_shared_kgs(args, shared_dir: Path) -> list[core.KGRun]:
    shared_args = build_shared_args(args, shared_dir)
    manifest_path = shared_dir / core.MANIFEST_FILENAME

    if args.resume and manifest_path.exists():
        print(f"Reusing shared KGs from: {shared_dir}")
        runs = core.load_manifest(shared_args)
        core.validate_kg_runs(shared_args, runs)
        return runs

    if args.resume and not manifest_path.exists() and any(shared_dir.iterdir()):
        raise FileNotFoundError(
            f"Cannot resume: shared KG manifest is missing from {shared_dir}"
        )

    print("\nGenerating the shared KG set exactly once")
    runs = core.create_kgs(shared_args)
    write_shared_hashes(shared_dir, runs)
    return runs


def write_shared_hashes(
    shared_dir: Path,
    runs: Sequence[core.KGRun],
) -> Path:
    rows = [
        {
            "run": run.label,
            "window_condition": run.window_condition,
            "removal_percent": run.removal_percent,
            "tsv_path": str(run.tsv_path.resolve()),
            "sha256": sha256_file(run.tsv_path),
        }
        for run in runs
    ]

    output_path = shared_dir / HASH_FILENAME
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved shared-KG hashes: {output_path}")
    return output_path


def write_model_manifest(
    model_dir: Path,
    shared_runs: Sequence[core.KGRun],
    model: str,
    experiment_name: str,
) -> Path:
    """
    Write a model-local manifest whose TSV paths point directly to the shared
    KG files. No model receives a private regenerated KG copy.
    """
    ensure_dir(model_dir)

    rows = []
    for run in shared_runs:
        rows.append(
            {
                "label": run.label,
                "tsv_path": str(run.tsv_path.resolve()),
                "experiment": experiment_name,
                "embedding_model": model,
                "window_condition": run.window_condition,
                "removal_percent": run.removal_percent,
                "relation": run.relation,
                "seed": run.seed,
                "nested_removals": run.nested_removals,
                "sort_order": run.sort_order,
            }
        )

    manifest_path = model_dir / core.MANIFEST_FILENAME
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def copy_shared_provenance(shared_dir: Path, model_dir: Path) -> None:
    for filename in (
        core.REMOVAL_PLAN_FILENAME,
        HASH_FILENAME,
    ):
        source = shared_dir / filename
        destination = model_dir / filename
        if source.exists():
            shutil.copy2(source, destination)


def prepare_models(
    args,
    root: Path,
    shared_dir: Path,
    shared_runs: Sequence[core.KGRun],
):
    prepared = []
    seen = set()

    for raw_model in args.models:
        model = safe_model_name(raw_model)
        if model in seen:
            raise ValueError(f"Duplicate model requested: {model}")
        seen.add(model)

        model_dir = root / model
        experiment_name = f"{args.n_people}people_{model}"

        write_model_manifest(
            model_dir=model_dir,
            shared_runs=shared_runs,
            model=model,
            experiment_name=experiment_name,
        )
        copy_shared_provenance(shared_dir, model_dir)

        model_args = copy.copy(args)
        model_args.basepath = str(model_dir)
        model_args.experiment_name = experiment_name
        model_args.embedding_model = model
        model_args.processes = 1
        model_args.force = False

        runs = core.load_manifest(model_args)
        core.validate_kg_runs(model_args, runs)
        prepared.append((model, model_dir, model_args, runs))

    return prepared


def build_embedding_jobs(args, prepared_models) -> list[EmbeddingJob]:
    jobs = []

    # Interleave models by KG condition. With five workers, the first batch
    # trains all five models on the same shared KG condition concurrently.
    run_count = len(prepared_models[0][3])
    if any(len(item[3]) != run_count for item in prepared_models):
        raise ValueError("Models do not have matching run manifests.")

    for run_index in range(run_count):
        reference_run = prepared_models[0][3][run_index]

        for model, model_dir, _, runs in prepared_models:
            run = runs[run_index]

            if run.label != reference_run.label:
                raise ValueError(
                    "Model manifests have different run ordering: "
                    f"{reference_run.label!r} versus {run.label!r}."
                )

            if run.tsv_path.resolve() != reference_run.tsv_path.resolve():
                raise ValueError(
                    f"Model {model!r} is not pointing at the shared KG for "
                    f"run {run.label!r}."
                )

            jobs.append(
                EmbeddingJob(
                    model=model,
                    run_label=run.label,
                    tsv_path=str(run.tsv_path.resolve()),
                    run_folder=str(model_dir / "runs" / run.label),
                    seed=args.seed,
                    epochs=args.embedding_epochs,
                    dimensions=args.embedding_dimensions,
                    force=args.force,
                    resume=args.resume,
                )
            )

    return jobs


def run_parallel_embeddings(args, prepared_models) -> None:
    jobs = build_embedding_jobs(args, prepared_models)

    process_count = min(args.embedding_processes, len(jobs))
    if process_count < 1:
        raise ValueError("--embedding-processes must be at least 1.")

    print(
        f"\nTraining {len(jobs)} model/KG embedding jobs "
        f"with {process_count} parallel workers"
    )

    if process_count == 1:
        for job in jobs:
            print(train_embedding_job(job))
    else:
        # spawn is safer than fork for PyTorch/PyKEEN workloads.
        context = torch.multiprocessing.get_context("spawn")
        with context.Pool(processes=process_count) as pool:
            for message in pool.imap_unordered(train_embedding_job, jobs):
                print(message, flush=True)

    for _, _, model_args, runs in prepared_models:
        core.validate_embedding_runs(model_args, runs)


def csv_has_rows(path: str | Path) -> bool:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        return not pd.read_csv(path, nrows=1).empty
    except Exception:
        return False


def run_downstream_steps(args, prepared_models) -> None:
    if args.stop_after == "embeddings":
        return

    for model, model_dir, model_args, runs in prepared_models:
        query_output = model_dir / "query_point_distances_all_runs.csv"

        if args.resume and csv_has_rows(query_output):
            print(f"\nSKIPPED existing query-point output for {model}")
        else:
            print(f"\nQuery-point evaluation for {model}")
            core.run_query_point_evaluations(model_args, runs)

        if args.stop_after == "query-points":
            continue

        analysis_outputs = [
            model_dir / "person_year_predictions.csv",
            model_dir / "run_year_summary.csv",
            model_dir / "run_year_summary_missing_only.csv",
            model_dir / "learning_person_predictions.csv",
            model_dir / "learning_run_summary.csv",
            model_dir / "learning_run_summary_missing_only.csv",
        ]

        if args.resume and all(csv_has_rows(path) for path in analysis_outputs):
            print(f"SKIPPED existing analysis outputs for {model}")
        else:
            print(f"\nAnalysis for {model}")
            core.run_analyses(model_args, runs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one shared KG set, train multiple embedding models on "
            "those exact KGs in parallel, and create per-model analysis data."
        )
    )

    parser.add_argument(
        "--output-root",
        default="model_comparison_500_run1",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
    )

    parser.add_argument("--n-people", type=int, default=500)
    parser.add_argument("--n-vertices", type=int, default=500)
    parser.add_argument("--embedding-dimensions", type=int, default=300)
    parser.add_argument("--embedding-epochs", type=int, default=100)

    parser.add_argument("--window-depth", type=int, default=4)
    parser.add_argument("--window-relation", default="inWindow")
    parser.add_argument(
        "--window-prefix",
        action="append",
        default=None,
    )

    parser.add_argument("--decimal-precision", type=int, default=0)
    parser.add_argument("--high", type=int, default=100)
    parser.add_argument("--low", type=int, default=1)

    parser.add_argument("--relation", default="hasAge")
    parser.add_argument(
        "--removal-percentages",
        type=float,
        nargs="+",
        default=[25.0, 50.0, 75.0, 99.0],
    )
    # Kept because core.get_removal_percentages() expects these fallback fields.
    parser.add_argument("--n-variations", type=int, default=5)
    parser.add_argument("--removal-percent-step", type=float, default=15.0)
    parser.add_argument("--independent-removals", action="store_true")

    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument(
        "--regression-model",
        choices=["ridge", "mlp"],
        default="ridge",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--embedding-processes",
        type=int,
        default=min(5, os.cpu_count() or 1),
        help=(
            "Number of model/KG training jobs to run at once. "
            "Reduce this if GPU memory or RAM is exhausted."
        ),
    )
    parser.add_argument(
        "--stop-after",
        choices=["embeddings", "query-points", "analysis"],
        default="analysis",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse the shared KGs and skip completed embedding folders.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rebuild the entire output root.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.window_prefix is None:
        args.window_prefix = ["Window_", "window_", "window"]

    root, shared_dir = prepare_root(args)
    shared_runs = create_or_load_shared_kgs(args, shared_dir)
    prepared_models = prepare_models(
        args=args,
        root=root,
        shared_dir=shared_dir,
        shared_runs=shared_runs,
    )

    run_parallel_embeddings(args, prepared_models)
    run_downstream_steps(args, prepared_models)

    print("\nDone.")
    print(f"Shared KGs: {shared_dir}")
    print(f"Model outputs: {root}")


if __name__ == "__main__":
    main()
