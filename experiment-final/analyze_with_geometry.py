import argparse
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge


def make_bool(series):
    if series.dtype == bool:
        return series

    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def build_distance_matrix(run_df):
    pivot = run_df.pivot_table(
        index="person",
        columns="candidate_v_num",
        values="distance",
        aggfunc="first",
    )

    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    meta = (
        run_df
        .groupby("person")
        .agg(
            true_age=("true_v_num", "first"),
            ground_truth_missing_in_run=("ground_truth_missing_in_run", "first"),
        )
    )

    meta["ground_truth_missing_in_run"] = make_bool(meta["ground_truth_missing_in_run"])

    pivot = pivot.loc[meta.index]

    return pivot, meta


def fit_predict_distance_regression(X, y, train_mask, model_type="ridge"):
    if model_type == "ridge":
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0),
        )

    elif model_type == "kernel_ridge":
        model = make_pipeline(
            StandardScaler(),
            KernelRidge(alpha=1.0, kernel="rbf", gamma=None),
        )

    else:
        raise ValueError("model_type must be ridge or kernel_ridge")

    model.fit(X[train_mask], y[train_mask])
    return model.predict(X)


def softknn_predict(X, candidate_ages, train_mask, y):
    candidate_ages = np.asarray(candidate_ages, dtype=float)

    taus = np.logspace(-2, 2, 50)
    best_tau = None
    best_mae = float("inf")

    for tau in taus:
        weights = np.exp(-(X ** 2) / tau)
        preds = (weights @ candidate_ages) / weights.sum(axis=1)

        mae = np.mean(np.abs(preds[train_mask] - y[train_mask]))

        if mae < best_mae:
            best_mae = mae
            best_tau = tau

    weights = np.exp(-(X ** 2) / best_tau)
    preds = (weights @ candidate_ages) / weights.sum(axis=1)

    return preds


def summarize_runs(person_df, count_column_name="n_people"):
    rows = []

    for run, group in person_df.groupby("run"):
        best = group.sort_values("abs_error").iloc[0]
        worst = group.sort_values("abs_error", ascending=False).iloc[0]

        rows.append({
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
        })

    return pd.DataFrame(rows)


def analyze_geometry(
    csv_path,
    person_output,
    run_output,
    missing_run_output,
    model_type="ridge",
):
    df = pd.read_csv(csv_path)

    required = {
        "run",
        "person",
        "candidate_v_num",
        "distance",
        "true_v_num",
        "ground_truth_missing_in_run",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    all_rows = []

    for run, run_df in df.groupby("run"):
        print(f"Evaluating MuRE-distance geometry model for {run}")

        X_df, meta = build_distance_matrix(run_df)

        X = X_df.to_numpy(dtype=float)
        y = meta["true_age"].to_numpy(dtype=float)
        train_mask = ~meta["ground_truth_missing_in_run"].to_numpy(dtype=bool)

        if train_mask.sum() < 2:
            raise ValueError(f"{run} has fewer than two known relations to train on.")

        if model_type in ["ridge", "kernel_ridge"]:
            preds = fit_predict_distance_regression(
                X=X,
                y=y,
                train_mask=train_mask,
                model_type=model_type,
            )

        elif model_type == "softknn":
            preds = softknn_predict(
                X=X,
                candidate_ages=X_df.columns.to_numpy(dtype=float),
                train_mask=train_mask,
                y=y,
            )

        else:
            raise ValueError("model_type must be ridge, kernel_ridge, or softknn")

        for person, true_age, missing_flag, pred in zip(
            X_df.index,
            y,
            meta["ground_truth_missing_in_run"],
            preds,
        ):
            error = pred - true_age

            all_rows.append({
                "run": run,
                "person": person,
                "true_age": true_age,
                "ground_truth_missing_in_run": bool(missing_flag),
                "predicted_age": pred,
                "error": error,
                "abs_error": abs(error),
                "model_type": model_type,
            })

    person_df = pd.DataFrame(all_rows)

    full_summary_df = summarize_runs(person_df, count_column_name="n_people")

    missing_person_df = person_df[
        person_df["ground_truth_missing_in_run"] == True
    ].copy()

    missing_summary_df = summarize_runs(
        missing_person_df,
        count_column_name="n_missing_people",
    )

    person_df.to_csv(person_output, index=False)
    full_summary_df.to_csv(run_output, index=False)
    missing_summary_df.to_csv(missing_run_output, index=False)

    print(f"Saved geometry per-person predictions to: {person_output}")
    print(f"Saved geometry full summary to: {run_output}")
    print(f"Saved geometry missing-only summary to: {missing_run_output}")

    return person_df, full_summary_df, missing_summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        default="./500people/query_point_distances_all_runs.csv",
    )

    parser.add_argument(
        "--model-type",
        default="ridge",
        choices=["ridge", "kernel_ridge", "softknn"],
    )

    parser.add_argument(
        "--person-output",
        default="./500people/geometry_person_predictions.csv",
    )

    parser.add_argument(
        "--run-output",
        default="./500people/geometry_run_summary.csv",
    )

    parser.add_argument(
        "--missing-run-output",
        default="./500people/geometry_run_summary_missing_only.csv",
    )

    args = parser.parse_args()

    person_df, full_summary_df, missing_summary_df = analyze_geometry(
        csv_path=args.csv,
        person_output=args.person_output,
        run_output=args.run_output,
        missing_run_output=args.missing_run_output,
        model_type=args.model_type,
    )

    print("\nFull geometry summary:")
    print(full_summary_df.to_string(index=False))

    print("\nMissing-only geometry summary:")
    print(missing_summary_df.to_string(index=False))