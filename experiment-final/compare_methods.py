import argparse
import numpy as np
import pandas as pd

from scipy.stats import ttest_rel, wilcoxon


def cohen_dz(diff):
    diff = np.asarray(diff, dtype=float)

    if len(diff) < 2:
        return np.nan

    std = np.std(diff, ddof=1)

    if std == 0:
        return np.nan

    return np.mean(diff) / std


def paired_bootstrap_ci(diff, n_boot=10000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)

    if len(diff) == 0:
        return np.nan, np.nan

    boot_means = []

    for _ in range(n_boot):
        sample = rng.choice(diff, size=len(diff), replace=True)
        boot_means.append(np.mean(sample))

    low = np.percentile(boot_means, 100 * alpha / 2)
    high = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return low, high


def compare_methods(
    query_person_csv,
    learning_person_csv,
    output_csv,
    missing_only=False,
):
    query_df = pd.read_csv(query_person_csv)
    learning_df = pd.read_csv(learning_person_csv)

    query_required = {
        "run",
        "person",
        "true_age",
        "ground_truth_missing_in_run",
        "top1_abs_error",
    }

    learning_required = {
        "run",
        "person",
        "true_age",
        "ground_truth_missing_in_run",
        "abs_error",
    }

    missing_query = query_required - set(query_df.columns)
    missing_learning = learning_required - set(learning_df.columns)

    if missing_query:
        raise ValueError(f"Query CSV missing columns: {missing_query}")

    if missing_learning:
        raise ValueError(f"Learning CSV missing columns: {missing_learning}")

    query_df = query_df.rename(columns={
        "top1_abs_error": "query_abs_error",
        "top1_pred_age": "query_pred_age",
    })

    learning_df = learning_df.rename(columns={
        "abs_error": "learning_abs_error",
        "predicted_age": "learning_pred_age",
    })

    keep_query = [
        "run",
        "person",
        "true_age",
        "ground_truth_missing_in_run",
        "query_abs_error",
    ]

    if "query_pred_age" in query_df.columns:
        keep_query.append("query_pred_age")

    keep_learning = [
        "run",
        "person",
        "learning_abs_error",
    ]

    if "learning_pred_age" in learning_df.columns:
        keep_learning.append("learning_pred_age")

    merged = pd.merge(
        query_df[keep_query],
        learning_df[keep_learning],
        on=["run", "person"],
        how="inner",
    )

    if missing_only:
        merged = merged[merged["ground_truth_missing_in_run"] == True].copy()

    rows = []

    for run, group in merged.groupby("run"):
        q = group["query_abs_error"].astype(float)
        l = group["learning_abs_error"].astype(float)

        # Positive diff means query-point method has larger error.
        # So positive means learning regression is better.
        diff = q - l

        mean_query = q.mean()
        mean_learning = l.mean()

        mean_diff = diff.mean()
        median_diff = diff.median()
                
        std_diff = diff.std(ddof=1)
        std_query_abs_error = q.std(ddof=1)
        std_learning_abs_error = l.std(ddof=1)

        better_method = (
            "learning"
            if mean_learning < mean_query
            else "query_points"
            if mean_query < mean_learning
            else "tie"
        )

        try:
            t_stat, t_p = ttest_rel(q, l)
        except Exception:
            t_stat, t_p = np.nan, np.nan

        try:
            w_stat, w_p = wilcoxon(q, l, zero_method="wilcox")
        except Exception:
            w_stat, w_p = np.nan, np.nan

        ci_low, ci_high = paired_bootstrap_ci(diff)

        rows.append({
            "run": run,
            "n_pairs": len(group),
            "missing_only": missing_only,

            "mean_query_abs_error": mean_query,
            "mean_learning_abs_error": mean_learning,

            "mean_error_difference_query_minus_learning": mean_diff,
            "median_error_difference_query_minus_learning": median_diff,

            "better_method_by_mean": better_method,

            "paired_t_stat": t_stat,
            "paired_t_p_value": t_p,

            "wilcoxon_stat": w_stat,
            "wilcoxon_p_value": w_p,

            "cohen_dz_query_minus_learning": cohen_dz(diff),

            "bootstrap_mean_diff_ci_low": ci_low,
            "bootstrap_mean_diff_ci_high": ci_high,
                        
            "std_query_abs_error": std_query_abs_error,
            "std_learning_abs_error": std_learning_abs_error,

            "std_error_difference_query_minus_learning": std_diff,
            
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_csv, index=False)

    print(f"Saved comparison to: {output_csv}")
    print(result_df.to_string(index=False))

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query-person-csv",
        default="./500people/person_year_predictions.csv",
    )

    parser.add_argument(
        "--learning-person-csv",
        default="./500people/learning_person_predictions.csv",
    )

    parser.add_argument(
        "--output-csv",
        default="./500people/method_comparison.csv",
    )

    parser.add_argument(
        "--missing-only-output-csv",
        default="./500people/method_comparison_missing_only.csv",
    )

    args = parser.parse_args()

    print("\nComparing all people:")
    compare_methods(
        query_person_csv=args.query_person_csv,
        learning_person_csv=args.learning_person_csv,
        output_csv=args.output_csv,
        missing_only=False,
    )

    print("\nComparing only missing relations:")
    compare_methods(
        query_person_csv=args.query_person_csv,
        learning_person_csv=args.learning_person_csv,
        output_csv=args.missing_only_output_csv,
        missing_only=True,
    )