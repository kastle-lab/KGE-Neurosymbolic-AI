# view_results.py
# Generates a Markdown report from experiment CSV outputs.

import argparse
from pathlib import Path

import pandas as pd


def read_csv_if_exists(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    print(f"Warning: missing file: {path}")
    return None


def df_to_md(df, columns=None, round_digits=4):
    if df is None or df.empty:
        return "_No data available._\n"

    if columns is not None:
        columns = [c for c in columns if c in df.columns]
        df = df[columns]

    return df.round(round_digits).to_markdown(index=False) + "\n"


def write_section(f, title, df, columns=None):
    f.write(f"\n## {title}\n\n")
    f.write(df_to_md(df, columns=columns))


def generate_report(basepath, output_md):
    basepath = Path(basepath)

    query_all = read_csv_if_exists(basepath / "run_year_summary.csv")
    query_missing = read_csv_if_exists(basepath / "run_year_summary_missing_only.csv")

    learning_all = read_csv_if_exists(basepath / "learning_run_summary.csv")
    learning_missing = read_csv_if_exists(basepath / "learning_run_summary_missing_only.csv")

    comparison_all = read_csv_if_exists(basepath / "method_comparison.csv")
    comparison_missing = read_csv_if_exists(basepath / "method_comparison_missing_only.csv")

    query_cols = [
        "run",
        "n_people",
        "n_missing_people",
        "mean_top1_abs_error",
        "median_top1_abs_error",
        "mean_top1_error",
        "std_top1_error",
        "std_top1_abs_error",
        "best_person",
        "best_true_age",
        "best_pred_age",
        "best_abs_error",
        "worst_person",
        "worst_true_age",
        "worst_pred_age",
        "worst_abs_error",
    ]

    learning_cols = [
        "run",
        "n_people",
        "n_missing_people",
        "mean_abs_error",
        "median_abs_error",
        "std_abs_error",
        "max_abs_error",
        "mean_error",
        "std_error",
        "best_person",
        "best_true_age",
        "best_pred_age",
        "best_abs_error",
        "worst_person",
        "worst_true_age",
        "worst_pred_age",
        "worst_abs_error",
    ]

    comparison_cols = [
        "run",
        "n_pairs",
        "missing_only",
        "mean_query_abs_error",
        "std_query_abs_error",
        "mean_learning_abs_error",
        "std_learning_abs_error",
        "mean_error_difference_query_minus_learning",
        "median_error_difference_query_minus_learning",
        "std_error_difference_query_minus_learning",
        "better_method_by_mean",
        "paired_t_p_value",
        "wilcoxon_p_value",
        "cohen_dz_query_minus_learning",
        "bootstrap_mean_diff_ci_low",
        "bootstrap_mean_diff_ci_high",
    ]

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("# Knowledge Graph Age Prediction Results\n\n")
        f.write(f"Base path: `{basepath}`\n\n")

        f.write("## Result files used\n\n")
        for name in [
            "run_year_summary.csv",
            "run_year_summary_missing_only.csv",
            "learning_run_summary.csv",
            "learning_run_summary_missing_only.csv",
            "method_comparison.csv",
            "method_comparison_missing_only.csv",
        ]:
            status = "found" if (basepath / name).exists() else "missing"
            f.write(f"- `{name}` — {status}\n")

        f.write("\n---\n")

        write_section(f, "Query-Point Method — All People", query_all, query_cols)
        write_section(f, "Query-Point Method — Missing Relations Only", query_missing, query_cols)

        write_section(f, "Learned Regression Method — All People", learning_all, learning_cols)
        write_section(f, "Learned Regression Method — Missing Relations Only", learning_missing, learning_cols)

        write_section(f, "Method Comparison — All People", comparison_all, comparison_cols)
        write_section(f, "Method Comparison — Missing Relations Only", comparison_missing, comparison_cols)

        f.write("\n## Notes\n\n")
        f.write(
            "- Positive `mean_error_difference_query_minus_learning` means the query-point method had larger error, "
            "so learned regression performed better.\n"
        )
        f.write(
            "- Negative `mean_error_difference_query_minus_learning` means learned regression had larger error, "
            "so query-point prediction performed better.\n"
        )
        f.write(
            "- Standard deviation columns report variability across people within each run.\n"
        )

    print(f"Saved Markdown report to: {output_md}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--basepath",
        default="./500people",
        help="Experiment folder containing summary CSV files.",
    )

    parser.add_argument(
        "--output-md",
        default=None,
        help="Output Markdown path. Defaults to <basepath>/results_report.md.",
    )

    args = parser.parse_args()

    output_md = args.output_md
    if output_md is None:
        output_md = Path(args.basepath) / "results_report.md"

    generate_report(
        basepath=args.basepath,
        output_md=output_md,
    )