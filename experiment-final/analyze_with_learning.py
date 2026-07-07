import argparse

import pandas as pd

from load_vectors import load_person_vectors, load_age_vectors, load_hasAge_bundle
from learn_function import learn_all_positions


def v_num(v_label):
    if v_label is None:
        return None

    label = str(v_label)

    if not label.startswith("v"):
        return None

    try:
        return int(label[1:])
    except ValueError:
        return None


def build_truth(original_run_path):
    original_bundle = load_hasAge_bundle(original_run_path)
    return original_bundle["person_to_v"]


def evaluate_learning_run(
    original_run_path,
    evaluated_run_path,
    run_label,
    person_model_type="ridge",
):
    truth_person_to_v = build_truth(original_run_path)

    person_vectors = load_person_vectors(
        run_path=evaluated_run_path,
        original_index=None,
        relation_filter="hasAge",
        head_prefix="person",
    )

    age_vectors = load_age_vectors(
        run_path=evaluated_run_path,
        relation_filter="hasAge",
        v_prefix="v",
    )

    result = learn_all_positions(
        age_vectors=age_vectors,
        person_vectors=person_vectors,
        v_prefix="v",
        person_model_type=person_model_type,
    )

    rows = []

    for row in result["person_positions"]["ordered"]:
        person = row["person"]

        true_v = truth_person_to_v.get(person)
        true_age = v_num(true_v)

        run_v = row["known_age_node"]
        run_age = row["known_v_num"]

        pred_age = float(row["predicted_v_num"])
        error = pred_age - true_age
        abs_error = abs(error)

        rows.append({
            "run": run_label,
            "person": person,

            "true_v": true_v,
            "true_age": true_age,

            "run_v": run_v,
            "run_age": run_age,
            "ground_truth_missing_in_run": bool(row["exists"] == False),

            "predicted_age": pred_age,
            "error": error,
            "abs_error": abs_error,

            "axis_position": row["axis_position"],
            "nearest_known_person": row["nearest_known_person"],
            "nearest_known_age_node": row["nearest_known_age_node"],
            "nearest_known_v_num": row["nearest_known_v_num"],
            "nearest_known_distance": row["nearest_known_distance"],
        })

    return pd.DataFrame(rows)


def summarize_runs(person_df, count_column_name="n_people"):
    summary_rows = []

    for run, group in person_df.groupby("run"):
        best = group.sort_values("abs_error", ascending=True).iloc[0]
        worst = group.sort_values("abs_error", ascending=False).iloc[0]

        summary_rows.append({
            "run": run,
            count_column_name: len(group),

            "best_person": best["person"],
            "best_true_age": best["true_age"],
            "best_pred_age": best["predicted_age"],
            "best_error": best["error"],
            "best_abs_error": best["abs_error"],

            "worst_person": worst["person"],
            "worst_true_age": worst["true_age"],
            "worst_pred_age": worst["predicted_age"],
            "worst_error": worst["error"],
            "worst_abs_error": worst["abs_error"],

            "mean_abs_error": group["abs_error"].mean(),
            "median_abs_error": group["abs_error"].median(),
            "max_abs_error": group["abs_error"].max(),
            "mean_error": group["error"].mean(),
                        
            "std_error": group["error"].std(),
            "std_abs_error": group["abs_error"].std(),
        })

    return pd.DataFrame(summary_rows)


def analyze_with_learning(
    basepath,
    removal_start=2,
    removal_stop=8,
    person_model_type="ridge",
    person_output=None,
    run_output=None,
    missing_run_output=None,
):
    original_run_path = f"{basepath}/runs/original"

    run_labels = ["original"] + [
        f"every_{n}_removed"
        for n in range(removal_start, removal_stop + 1)
    ]

    dfs = []

    for run_label in run_labels:
        evaluated_run_path = f"{basepath}/runs/{run_label}"

        print(f"Evaluating learned regression for {run_label}")

        df = evaluate_learning_run(
            original_run_path=original_run_path,
            evaluated_run_path=evaluated_run_path,
            run_label=run_label,
            person_model_type=person_model_type,
        )

        dfs.append(df)

    person_df = pd.concat(dfs, ignore_index=True)

    full_summary_df = summarize_runs(
        person_df=person_df,
        count_column_name="n_people",
    )

    missing_person_df = person_df[
        person_df["ground_truth_missing_in_run"] == True
    ].copy()

    missing_summary_df = summarize_runs(
        person_df=missing_person_df,
        count_column_name="n_missing_people",
    )

    if person_output is not None:
        person_df.to_csv(person_output, index=False)
        print(f"Saved learned per-person predictions to: {person_output}")

    if run_output is not None:
        full_summary_df.to_csv(run_output, index=False)
        print(f"Saved learned full run summary to: {run_output}")

    if missing_run_output is not None:
        missing_summary_df.to_csv(missing_run_output, index=False)
        print(f"Saved learned missing-only summary to: {missing_run_output}")

    return person_df, full_summary_df, missing_summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--basepath",
        default="./500people",
        help="Base experiment folder containing runs/original and runs/every_n_removed.",
    )

    parser.add_argument(
        "--removal-start",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--removal-stop",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--model-type",
        default="ridge",
        choices=["ridge", "mlp"],
    )

    parser.add_argument(
        "--person-output",
        default="./500people/learning_person_predictions.csv",
    )

    parser.add_argument(
        "--run-output",
        default="./500people/learning_run_summary.csv",
    )

    parser.add_argument(
        "--missing-run-output",
        default="./500people/learning_run_summary_missing_only.csv",
    )

    args = parser.parse_args()

    person_df, full_summary_df, missing_summary_df = analyze_with_learning(
        basepath=args.basepath,
        removal_start=args.removal_start,
        removal_stop=args.removal_stop,
        person_model_type=args.model_type,
        person_output=args.person_output,
        run_output=args.run_output,
        missing_run_output=args.missing_run_output,
    )

    print("\nFull learned-regression summary:")
    print(full_summary_df.to_string(index=False))

    print("\nMissing-only learned-regression summary:")
    print(missing_summary_df.to_string(index=False))