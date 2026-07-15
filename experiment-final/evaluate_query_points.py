import re
import numpy as np
import pandas as pd

from load_vectors import load_hasAge_bundle
from compute_distances import compute_query_points

def load_person_to_v_from_tsv(
    tsv_path,
    relation_string="hasAge",
):
    person_to_v = {}

    with open(tsv_path, "r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.rstrip("\r\n")

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 3:
                raise ValueError(
                    f"Malformed triple in {tsv_path} at line "
                    f"{line_number}: expected 3 columns."
                )

            head, relation, tail = parts

            if relation != relation_string:
                continue

            if head in person_to_v and person_to_v[head] != tail:
                raise ValueError(
                    f"Conflicting {relation_string} values for "
                    f"{head!r}: {person_to_v[head]!r} and {tail!r}."
                )

            person_to_v[head] = tail

    if not person_to_v:
        raise ValueError(
            f"No {relation_string!r} triples found in {tsv_path}."
        )

    return person_to_v

def v_num(v_label):
    if v_label is None:
        return None

    match = re.search(
        r"[vV](-?\d+(?:\.\d+)?)$",
        str(v_label).strip(),
    )

    if match is None:
        return None

    value = float(match.group(1))

    return int(value) if value.is_integer() else value


def compare_all_distances_to_truth(
    query_points,
    v_embeddings,
    truth_person_to_v,
    run_person_to_v,
):
    rows = []

    for person_label, q_vec in query_points.items():
        true_v = truth_person_to_v.get(person_label)
        true_v_number = v_num(true_v)

        run_v = run_person_to_v.get(person_label)

        exists_in_run = person_label in run_person_to_v
        ground_truth_missing_in_run = (
            true_v is not None
            and person_label not in run_person_to_v
        )

        for v_label, a_vec in v_embeddings.items():
            pred_v_number = v_num(v_label)
            distance = np.linalg.norm(q_vec - a_vec)

            rows.append({
                "person": person_label,

                "candidate_v": v_label,
                "candidate_v_num": pred_v_number,
                "distance": distance,

                "true_v": true_v,
                "true_v_num": true_v_number,

                "exists_in_run": exists_in_run,
                "run_v": run_v,
                "run_v_num": v_num(run_v),

                "is_ground_truth": v_label == true_v,
                "ground_truth_missing_in_run": ground_truth_missing_in_run,

                "absolute_error": (
                    None
                    if pred_v_number is None or true_v_number is None
                    else abs(pred_v_number - true_v_number)
                ),

                "signed_error": (
                    None
                    if pred_v_number is None or true_v_number is None
                    else pred_v_number - true_v_number
                ),
            })

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by=["person", "distance"],
        ascending=[True, True],
    ).reset_index(drop=True)

    df["distance_rank_for_person"] = (
        df.groupby("person")
        .cumcount()
        + 1
    )

    return df


def evaluate_query_point_run(
    original_run_path,
    evaluated_run_path,
    original_tsv_path,
    evaluated_tsv_path,
    output_csv_path=None,
):
    
    original_bundle = load_hasAge_bundle(original_run_path)
    evaluated_bundle = load_hasAge_bundle(evaluated_run_path)

    truth_person_to_v = load_person_to_v_from_tsv(
        original_tsv_path,
        relation_string="hasAge",
    )

    run_person_to_v = load_person_to_v_from_tsv(
        evaluated_tsv_path,
        relation_string="hasAge",
    )

    v_embeddings = evaluated_bundle["v_embeddings"]
    person_embeddings = evaluated_bundle["person_embeddings"]

    r_vec = evaluated_bundle["hasAge_relation"]["r_vec"]
    R_vec = evaluated_bundle["hasAge_relation"]["R_vec"]

    query_points = compute_query_points(
        person_embeddings=person_embeddings,
        r_vec=r_vec,
        R_vec=R_vec,
    )

    df = compare_all_distances_to_truth(
        query_points=query_points,
        v_embeddings=v_embeddings,
        truth_person_to_v=truth_person_to_v,
        run_person_to_v=run_person_to_v,
    )

    if output_csv_path is not None:
        df.to_csv(output_csv_path, index=False)
        print(f"Saved all query point distances to: {output_csv_path}")

    return df