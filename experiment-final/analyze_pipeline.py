import argparse
import subprocess
from pathlib import Path


def run_command(command):
    print("\nRunning:")
    print(" ".join(command))

    result = subprocess.run(command)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")


def analyze_pipeline(basepath, max_k=10):
    basepath = Path(basepath)

    query_distances_csv = basepath / "query_point_distances_all_runs.csv"

    query_person_output = basepath / "person_year_predictions.csv"
    query_run_output = basepath / "run_year_summary.csv"
    query_missing_run_output = basepath / "run_year_summary_missing_only.csv"

    learning_person_output = basepath / "learning_person_predictions.csv"
    learning_run_output = basepath / "learning_run_summary.csv"
    learning_missing_run_output = basepath / "learning_run_summary_missing_only.csv"

    comparison_output = basepath / "method_comparison.csv"
    comparison_missing_output = basepath / "method_comparison_missing_only.csv"

    if not query_distances_csv.exists():
        raise FileNotFoundError(
            f"Could not find {query_distances_csv}. "
            "Run pipeline.py first to generate query_point_distances_all_runs.csv."
        )

    # ----------------------------
    # Analyze query-point method
    # ----------------------------
    run_command([
        "python",
        "analyze_data.py",
        "--csv",
        str(query_distances_csv),
        "--person-output",
        str(query_person_output),
        "--run-output",
        str(query_run_output),
        "--missing-run-output",
        str(query_missing_run_output),
        "--max-k",
        str(max_k),
    ])

    # ----------------------------
    # Analyze learned affine/ridge method
    # ----------------------------
    run_command([
        "python",
        "analyze_with_learning.py",
        "--basepath",
        str(basepath),
        "--model-type",
        "ridge",
        "--person-output",
        str(learning_person_output),
        "--run-output",
        str(learning_run_output),
        "--missing-run-output",
        str(learning_missing_run_output),
    ])

    # ----------------------------
    # Compare query vs learned ridge
    # ----------------------------
    run_command([
        "python",
        "compare_methods.py",
        "--query-person-csv",
        str(query_person_output),
        "--learning-person-csv",
        str(learning_person_output),
        "--output-csv",
        str(comparison_output),
        "--missing-only-output-csv",
        str(comparison_missing_output),
    ])

    print("\nDone. Outputs:")
    print(f"Query per-person:        {query_person_output}")
    print(f"Query summary:           {query_run_output}")
    print(f"Query missing summary:   {query_missing_run_output}")
    print(f"Learning per-person:     {learning_person_output}")
    print(f"Learning summary:        {learning_run_output}")
    print(f"Learning missing summary:{learning_missing_run_output}")
    print(f"Comparison:              {comparison_output}")
    print(f"Missing-only comparison: {comparison_missing_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--basepath",
        required=True,
        help="Experiment folder, e.g. ./100people or ./500people",
    )

    parser.add_argument(
        "--max-k",
        type=int,
        default=10,
        help="Top-k averaging limit for query-point method.",
    )

    args = parser.parse_args()

    analyze_pipeline(
        basepath=args.basepath,
        max_k=args.max_k,
    )