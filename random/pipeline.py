# pipeline.py

import argparse
from pathlib import Path

import pandas as pd

from embeddings_helper import KastleEmbeddings

# TODO: implement these modules/functions next.
from query_points import run_query_point_method
from regression_method import run_regression_method
from compare_methods import compare_methods
from view_results import generate_report


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_input_tsv(input_tsv, output_tsv):
    df = pd.read_csv(input_tsv, sep="\t", header=None)

    if df.shape[1] < 3:
        raise ValueError("Input TSV must have at least 3 columns: head, relation, tail")

    df = df.iloc[:, :3]
    df.columns = ["head", "relation", "tail"]

    df.to_csv(output_tsv, sep="\t", header=False, index=False)

    return df


def validate_graph(df, entity_prefix, property_relation, property_prefix):
    relation_edges = df[df["relation"] == property_relation]

    matching_edges = relation_edges[
        relation_edges["head"].astype(str).str.startswith(entity_prefix)
        & relation_edges["tail"].astype(str).str.startswith(property_prefix)
    ]

    print("\nGraph summary")
    print(f"Total triples: {len(df)}")
    print(f"Property relation: {property_relation}")
    print(f"Entity prefix: {entity_prefix}")
    print(f"Property prefix: {property_prefix}")
    print(f"Matching property edges: {len(matching_edges)}")

    if matching_edges.empty:
        raise ValueError(
            "No matching property edges found. "
            "Check --entity-prefix, --property-relation, and --property-prefix."
        )

    return matching_edges


def create_embeddings(
    tsv_path,
    run_path,
    embedding_model,
    embedding_epochs,
    embedding_dimensions,
    seed,
):
    ensure_dir(run_path)

    experiment = KastleEmbeddings(
        seed=seed,
        experiment_folder=str(run_path),
        tsv_path=str(tsv_path),
        emb_model=embedding_model,
        emb_epochs=embedding_epochs,
        emb_dimensions=embedding_dimensions,
        pca=False,
        tsne=False,
        umap=False,
        dr_components=2,
        plot_name="embeddings",
    )

    experiment.create_pykeen_embeddings()


def run_pipeline(args):
    output_dir = Path(args.output_dir)
    runs_dir = output_dir / "runs"
    run_path = runs_dir / "original"

    ensure_dir(output_dir)
    ensure_dir(runs_dir)

    normalized_tsv = output_dir / "input_graph.tsv"

    print("\nStep 1 — Normalize input TSV")
    df = normalize_input_tsv(
        input_tsv=args.input_tsv,
        output_tsv=normalized_tsv,
    )

    print("\nStep 2 — Validate graph")
    validate_graph(
        df=df,
        entity_prefix=args.entity_prefix,
        property_relation=args.property_relation,
        property_prefix=args.property_prefix,
    )

    print("\nStep 3 — Create embeddings")
    create_embeddings(
        tsv_path=normalized_tsv,
        run_path=run_path,
        embedding_model=args.embedding_model,
        embedding_epochs=args.embedding_epochs,
        embedding_dimensions=args.embedding_dimensions,
        seed=args.seed,
    )

    print("\nStep 4 — Query-point method")
    query_person_csv = output_dir / "query_point_predictions.csv"
    query_summary_csv = output_dir / "query_point_summary.csv"

    run_query_point_method(
        run_path=str(run_path),
        input_tsv=str(normalized_tsv),
        entity_prefix=args.entity_prefix,
        property_relation=args.property_relation,
        property_prefix=args.property_prefix,
        output_csv=str(query_person_csv),
        summary_csv=str(query_summary_csv),
        max_k=args.max_k,
    )

    print("\nStep 5 — Regression method")
    regression_person_csv = output_dir / "regression_predictions.csv"
    regression_summary_csv = output_dir / "regression_summary.csv"

    run_regression_method(
        run_path=str(run_path),
        input_tsv=str(normalized_tsv),
        entity_prefix=args.entity_prefix,
        property_relation=args.property_relation,
        property_prefix=args.property_prefix,
        model_type=args.regression_model,
        output_csv=str(regression_person_csv),
        summary_csv=str(regression_summary_csv),
    )

    print("\nStep 6 — Compare methods")
    comparison_csv = output_dir / "method_comparison.csv"

    compare_methods(
        query_person_csv=str(query_person_csv),
        learning_person_csv=str(regression_person_csv),
        output_csv=str(comparison_csv),
        missing_only=False,
    )

    print("\nStep 7 — Generate Markdown report")
    report_md = output_dir / "results_report.md"

    generate_report(
        basepath=str(output_dir),
        output_md=str(report_md),
    )

    print("\nDone.")
    print(f"Output directory: {output_dir}")
    print(f"Report: {report_md}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-tsv",
        required=True,
        help="Input KG TSV with columns: head, relation, tail.",
    )

    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Directory where outputs will be written.",
    )

    parser.add_argument(
        "--entity-prefix",
        required=True,
        help="Prefix identifying entities to evaluate.",
    )

    parser.add_argument(
        "--property-relation",
        required=True,
        help="Relation connecting entity to property.",
    )

    parser.add_argument(
        "--property-prefix",
        required=True,
        help="Prefix identifying property/value nodes.",
    )

    parser.add_argument(
        "--embedding-model",
        default="mure",
    )

    parser.add_argument(
        "--embedding-epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--regression-model",
        default="ridge",
        choices=["ridge", "mlp"],
    )

    parser.add_argument(
        "--max-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()
    run_pipeline(args)