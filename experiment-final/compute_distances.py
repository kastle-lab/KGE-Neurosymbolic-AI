import re
import numpy as np
import pandas as pd


def v_num(v_label):
    if v_label is None:
        return None

    m = re.search(r"v(\d+)$", str(v_label))

    if m is None:
        return None

    return int(m.group(1))


def top_k_ages_for_each_person(query_points, v_embeddings, k=5):
    results = {}

    for person_label, q_vec in query_points.items():
        scored = []

        for v_label, a_vec in v_embeddings.items():
            d = np.linalg.norm(q_vec - a_vec)

            scored.append({
                "v_label": v_label,
                "v_num": v_num(v_label),
                "distance": d,
            })

        scored = sorted(scored, key=lambda x: x["distance"])

        results[person_label] = scored[:k]

    return results


def compare_top_k_to_truth(query_points, v_embeddings, person_to_v, k=5):
    top_k = top_k_ages_for_each_person(
        query_points=query_points,
        v_embeddings=v_embeddings,
        k=k,
    )

    rows = []

    for person_label, candidates in top_k.items():
        true_v = person_to_v.get(person_label)
        true_v_number = v_num(true_v)

        for rank, candidate in enumerate(candidates, start=1):
            pred_v = candidate["v_label"]
            pred_v_number = candidate["v_num"]

            rows.append({
                "person": person_label,
                "rank": rank,
                "predicted_v": pred_v,
                "predicted_v_num": pred_v_number,
                "distance": candidate["distance"],
                "true_v": true_v,
                "true_v_num": true_v_number,
                "is_truth": pred_v == true_v,
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

    return pd.DataFrame(rows)

def compute_query_point(p_vec, r_vec, R_vec):
    return p_vec * R_vec + r_vec


def compute_query_points(person_embeddings, r_vec, R_vec):
    query_points = {}

    for person_label, p_vec in person_embeddings.items():
        query_points[person_label] = compute_query_point(p_vec, r_vec, R_vec)

    return query_points


def compute_distances_to_ages(query_points, v_embeddings):
    distances = {}

    for person_label, q_vec in query_points.items():
        distances[person_label] = {}

        for v_label, a_vec in v_embeddings.items():
            d = np.linalg.norm(q_vec - a_vec)
            distances[person_label][v_label] = d

    return distances


def nearest_age_for_each_person(query_points, v_embeddings):
    nearest = {}

    for person_label, q_vec in query_points.items():
        best_v = None
        best_d = float("inf")

        for v_label, a_vec in v_embeddings.items():
            d = np.linalg.norm(q_vec - a_vec)

            if d < best_d:
                best_d = d
                best_v = v_label

        nearest[person_label] = {
            "person": person_label,
            "nearest_v": best_v,
            "distance": best_d,
        }

    return nearest