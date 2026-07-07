# pipeline.py

import argparse
from pathlib import Path
from multiprocessing import Pool

import torch
import pandas as pd

from kg_creator import create_kg
from kg_modifier import modify_kg
from embeddings_helper import KastleEmbeddings
from evaluate_query_points import evaluate_query_point_run

from analyze_data import analyze_data
from analyze_with_learning import analyze_with_learning
from compare_methods import compare_methods
from view_results import generate_report


GLOBAL_ARGS = None 


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def create_KGs(args):
    basepath = Path(args.basepath)
    ensure_dir(basepath)

    create_kg(
        n_vertices=args.n_vertices,
        n_people=args.n_people,
        window_depth=args.window_depth,
        decimal_precision=args.decimal_precision,
        high=args.high,
        low=args.low,
        experiment_folder=str(basepath),
        include_windowing=args.include_windowing,
    )

    for n in range(args.removal_start, args.removal_stop + 1):
        modify_kg(
            path=str(basepath),
            n=n,
            relation_string=args.relation,
        )

    kg_files = [("original", str(basepath / "kg.tsv"))]

    for n in range(args.removal_start, args.removal_stop + 1):
        kg_files.append(
            (
                f"every_{n}_removed",
                str(basepath / f"every_{n}_removed_kg.tsv"),
            )
        )

    return kg_files


def init_worker(args):
    global GLOBAL_ARGS
    GLOBAL_ARGS = args


def run_embedding_job(job):
    global GLOBAL_ARGS

    torch.set_num_threads(1)

    args = GLOBAL_ARGS
    basepath = Path(args.basepath)

    label, tsv_path = job
    run_folder = basepath / "runs" / label

    experiment = KastleEmbeddings(
        seed=args.seed,
        experiment_folder=str(run_folder),
        tsv_path=tsv_path,
        emb_model=args.embedding_model,
        emb_epochs=args.embedding_epochs,
        emb_dimensions=args.embedding_dimensions,
        pca=False,
        tsne=False,
        umap=False,
        dr_components=2,
        plot_name=label,
    )

    experiment.create_pykeen_embeddings()


def run_query_point_evaluations(args):
    basepath = Path(args.basepath)

    run_labels = ["original"] + [
        f"every_{n}_removed"
        for n in range(args.removal_start, args.removal_stop + 1)
    ]

    all_dfs = []

    for run_label in run_labels:
        print(f"\nEvaluating query points for {run_label}")

        df = evaluate_query_point_run(
            original_run_path=str(basepath / "runs" / "original"),
            evaluated_run_path=str(basepath / "runs" / run_label),
            output_csv_path=str(
                basepath / "runs" / run_label / "query_point_distances.csv"
            ),
        )

        df["experiment"] = args.experiment_name
        df["run"] = run_label

        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)

    full_output_path = basepath / "query_point_distances_all_runs.csv"
    full_df.to_csv(full_output_path, index=False)

    print(f"\nSaved all query point evaluations to: {full_output_path}")

    return full_df


def run_all_analyses(args):
    basepath = Path(args.basepath)

    query_distances_csv = basepath / "query_point_distances_all_runs.csv"

    query_person_csv = basepath / "person_year_predictions.csv"
    query_summary_csv = basepath / "run_year_summary.csv"
    query_missing_summary_csv = basepath / "run_year_summary_missing_only.csv"

    learning_person_csv = basepath / "learning_person_predictions.csv"
    learning_summary_csv = basepath / "learning_run_summary.csv"
    learning_missing_summary_csv = basepath / "learning_run_summary_missing_only.csv"

    comparison_csv = basepath / "method_comparison.csv"
    comparison_missing_csv = basepath / "method_comparison_missing_only.csv"

    print("\nAnalyzing query-point predictions")

    analyze_data(
        csv_path=str(query_distances_csv),
        person_output=str(query_person_csv),
        run_output=str(query_summary_csv),
        missing_run_output=str(query_missing_summary_csv),
        max_k=args.max_k,
    )

    print("\nAnalyzing learned regression predictions")

    analyze_with_learning(
        basepath=str(basepath),
        removal_start=args.removal_start,
        removal_stop=args.removal_stop,
        person_model_type=args.regression_model,
        person_output=str(learning_person_csv),
        run_output=str(learning_summary_csv),
        missing_run_output=str(learning_missing_summary_csv),
    )

    print("\nComparing methods — all people")

    compare_methods(
        query_person_csv=str(query_person_csv),
        learning_person_csv=str(learning_person_csv),
        output_csv=str(comparison_csv),
        missing_only=False,
    )

    print("\nComparing methods — missing only")

    compare_methods(
        query_person_csv=str(query_person_csv),
        learning_person_csv=str(learning_person_csv),
        output_csv=str(comparison_missing_csv),
        missing_only=True,
    )

    print("\nGenerating Markdown report")

    generate_report(
        basepath=str(basepath),
        output_md=str(basepath / "results_report.md"),
    )


def run_pipeline(args):
    basepath = Path(args.basepath)
    ensure_dir(basepath)

    if args.experiment_name is None:
        args.experiment_name = basepath.name

    if not args.skip_kg:
        print("\nStep 1 — Creating KGs")
        kg_files = create_KGs(args)
    else:
        print("\nSkipping KG creation")
        kg_files = [("original", str(basepath / "kg.tsv"))]
        for n in range(args.removal_start, args.removal_stop + 1):
            kg_files.append(
                (
                    f"every_{n}_removed",
                    str(basepath / f"every_{n}_removed_kg.tsv"),
                )
            )

    if not args.skip_embeddings:
        print("\nStep 2 — Creating embeddings")

        if args.processes <= 1:
            init_worker(args)
            for job in kg_files:
                run_embedding_job(job)
        else:
            with Pool(
                processes=args.processes,
                initializer=init_worker,
                initargs=(args,),
            ) as pool:
                pool.map(run_embedding_job, kg_files)
    else:
        print("\nSkipping embedding creation")

    if not args.skip_query_points:
        print("\nStep 3 — Running query-point evaluations")
        run_query_point_evaluations(args)
    else:
        print("\nSkipping query-point evaluation")

    if not args.skip_analysis:
        print("\nStep 4 — Running analyses")
        run_all_analyses(args)
    else:
        print("\nSkipping analyses")

    print("\nDone.")
    print(f"Results folder: {basepath}")
    print(f"Report: {basepath / 'results_report.md'}")


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-people", type=int, default=100)
    parser.add_argument("--n-vertices", type=int, default=100)

    parser.add_argument(
        "--basepath",
        default=None,
        help="Output experiment folder. Defaults to ./<n_people>people.",
    )

    parser.add_argument("--experiment-name", default=None)

    parser.add_argument("--embedding-dimensions", type=int, default=300)
    parser.add_argument("--embedding-epochs", type=int, default=100)
    parser.add_argument("--embedding-model", default="mure")

    parser.add_argument("--window-depth", type=int, default=4)
    parser.add_argument("--decimal-precision", type=int, default=0)
    parser.add_argument("--high", type=int, default=100)
    parser.add_argument("--low", type=int, default=1)

    parser.add_argument("--removal-start", type=int, default=2)
    parser.add_argument("--removal-stop", type=int, default=8)
    parser.add_argument("--relation", default="hasAge")

    parser.add_argument("--max-k", type=int, default=10)

    parser.add_argument(
        "--regression-model",
        default="ridge",
        choices=["ridge", "mlp"],
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processes", type=int, default=8)

    parser.add_argument("--skip-kg", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-query-points", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")

    parser.add_argument(
        "--include-windowing",
        action="store_true",
        help="Include inWindow/window hierarchy triples.",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.basepath is None:
        args.basepath = f"./{args.n_people}people"

    run_pipeline(args)