import argparse
import pandas as pd


REQUIRED_COLUMNS = {
    "run",
    "person",
    "candidate_v_num",
    "distance",
    "true_v_num",
    "ground_truth_missing_in_run",
}


def validate_columns(df):
    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")


def build_person_predictions(df, max_k):
    rows = []

    for (run, person), group in df.groupby(["run", "person"]):
        group = group.sort_values("distance").reset_index(drop=True)

        true_age = float(group["true_v_num"].iloc[0])
        top_ages = group["candidate_v_num"].head(max_k).astype(float).tolist()

        if len(top_ages) < max_k:
            raise ValueError(
                f"{run} / {person} has only {len(top_ages)} candidates; need {max_k}"
            )

        row = {
            "run": run,
            "person": person,
            "true_age": true_age,
            "ground_truth_missing_in_run": bool(group["ground_truth_missing_in_run"].iloc[0]),

            "top1_pred_age": top_ages[0],
            "top1_error": top_ages[0] - true_age,
            "top1_abs_error": abs(top_ages[0] - true_age),
        }

        for k in range(1, max_k + 1):
            pred_avg = sum(top_ages[:k]) / k

            row[f"top{k}_avg_pred_age"] = pred_avg
            row[f"top{k}_avg_error"] = pred_avg - true_age
            row[f"top{k}_avg_abs_error"] = abs(pred_avg - true_age)

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_runs(person_df, max_k, count_column_name="n_people"):
    summary_rows = []

    for run, group in person_df.groupby("run"):
        best = group.sort_values("top1_abs_error", ascending=True).iloc[0]
        worst = group.sort_values("top1_abs_error", ascending=False).iloc[0]

        row = {
            "run": run,
            count_column_name: len(group),

            "best_person": best["person"],
            "best_true_age": best["true_age"],
            "best_pred_age": best["top1_pred_age"],
            "best_error": best["top1_error"],
            "best_abs_error": best["top1_abs_error"],

            "worst_person": worst["person"],
            "worst_true_age": worst["true_age"],
            "worst_pred_age": worst["top1_pred_age"],
            "worst_error": worst["top1_error"],
            "worst_abs_error": worst["top1_abs_error"],

            "mean_top1_abs_error": group["top1_abs_error"].mean(),
            "median_top1_abs_error": group["top1_abs_error"].median(),
        }

        for k in range(1, max_k + 1):
            row[f"mean_top{k}_avg_abs_error"] = group[f"top{k}_avg_abs_error"].mean()

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def analyze_data(
    csv_path,
    person_output=None,
    run_output=None,
    missing_run_output=None,
    max_k=10,
):
    df = pd.read_csv(csv_path)
    validate_columns(df)

    person_df = build_person_predictions(df, max_k=max_k)

    full_summary_df = summarize_runs(
        person_df=person_df,
        max_k=max_k,
        count_column_name="n_people",
    )

    missing_person_df = person_df[
        person_df["ground_truth_missing_in_run"] == True
    ].copy()

    missing_summary_df = summarize_runs(
        person_df=missing_person_df,
        max_k=max_k,
        count_column_name="n_missing_people",
    )

    if person_output is not None:
        person_df.to_csv(person_output, index=False)
        print(f"Saved per-person predictions to: {person_output}")

    if run_output is not None:
        full_summary_df.to_csv(run_output, index=False)
        print(f"Saved full run summary to: {run_output}")

    if missing_run_output is not None:
        missing_summary_df.to_csv(missing_run_output, index=False)
        print(f"Saved missing-only run summary to: {missing_run_output}")

    return person_df, full_summary_df, missing_summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to query_point_distances_all_runs.csv",
    )

    parser.add_argument(
        "--person-output",
        default="person_year_predictions.csv",
    )

    parser.add_argument(
        "--run-output",
        default="run_year_summary.csv",
    )

    parser.add_argument(
        "--missing-run-output",
        default="run_year_summary_missing_only.csv",
    )

    parser.add_argument(
        "--max-k",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    person_df, full_summary_df, missing_summary_df = analyze_data(
        csv_path=args.csv,
        person_output=args.person_output,
        run_output=args.run_output,
        missing_run_output=args.missing_run_output,
        max_k=args.max_k,
    )

    print("\nFull run summary:")
    print(full_summary_df.to_string(index=False))

    print("\nMissing-only run summary:")
    print(missing_summary_df.to_string(index=False))