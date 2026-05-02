import re
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor

def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _v_number(v_label, v_prefix="v"):
    if v_label is None:
        return None

    match = re.search(rf"{re.escape(v_prefix)}(\d+)$", str(v_label))
    if match is None:
        return None

    return int(match.group(1))


def _stack_vectors(items, vec_key):
    return np.stack([
        _to_numpy(item[vec_key])
        for item in items
    ])


def learn_age_vector_axis(age_vectors, v_prefix="v"):
    """
    Learns/provides ordered scalar positions for v nodes.

    Since v nodes are already ordered v1...vn, their axis position is exact.
    """

    rows = []

    for idx, item in age_vectors.items():
        v_node = item["v_node"]
        v_num = _v_number(v_node, v_prefix=v_prefix)

        if v_num is None:
            continue

        rows.append({
            "source_index": idx,
            "v_node": v_node,
            "v_id": item["v_id"],
            "v_vec": item["v_vec"],
            "v_num": v_num,
        })

    rows = sorted(rows, key=lambda x: x["v_num"])

    n = len(rows)

    if n == 0:
        return {
            "positions": {},
            "ordered": [],
            "axis_min": None,
            "axis_max": None,
        }

    for rank, row in enumerate(rows):
        row["rank"] = rank
        row["axis_position"] = 0.0 if n == 1 else rank / (n - 1)

    positions = {
        row["source_index"]: row
        for row in rows
    }

    return {
        "positions": positions,
        "ordered": rows,
        "axis_min": 0.0,
        "axis_max": 1.0,
    }


def learn_person_age_axis(
    person_vectors,
    v_prefix="v",
    model_type="ridge",
    hidden_layer_sizes=(128, 64),
    alpha=1.0,
    max_iter=5000,
    random_state=42,
):
    """
    Learns a 1D age-like coordinate for person vectors.

    Known-age people are anchors.
    Missing-age people are projected onto the learned latent age axis.

    Returns every person with:
        - predicted_v_num
        - axis_position
        - nearest_known_person
        - nearest_known_distance
    """

    all_items = []
    known_items = []

    for idx, item in person_vectors.items():
        triple = item["triple_name"]
        tail_label = triple[2] if len(triple) > 2 else None
        v_num = _v_number(tail_label, v_prefix=v_prefix)

        row = {
            "source_index": idx,
            "person": triple[0],
            "known_age_node": tail_label if item["exists"] else None,
            "known_v_num": v_num if item["exists"] else None,
            "exists": item["exists"],
            "head_id": item["head_id"],
            "query_vec": item["query_vec"],
            "raw_item": item,
        }

        all_items.append(row)

        if item["exists"] and v_num is not None:
            known_items.append(row)

    if len(known_items) < 2:
        raise ValueError("Need at least two known-age person nodes to learn an age axis.")

    X_known = _stack_vectors(known_items, "query_vec")
    y_known = np.array([
        row["known_v_num"]
        for row in known_items
    ], dtype=float)

    X_all = _stack_vectors(all_items, "query_vec")

    if model_type == "ridge":
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=alpha)
        )

    elif model_type == "mlp":
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation="tanh",
                solver="adam",
                max_iter=max_iter,
                random_state=random_state,
            )
        )

    else:
        raise ValueError("model_type must be either 'ridge' or 'mlp'.")

    model.fit(X_known, y_known)

    predicted_v_nums = model.predict(X_all)

    min_v = float(np.min(y_known))
    max_v = float(np.max(y_known))

    if max_v == min_v:
        axis_positions = np.zeros_like(predicted_v_nums)
    else:
        axis_positions = (predicted_v_nums - min_v) / (max_v - min_v)

    # nearest known person in learned scalar space
    known_axis = model.predict(X_known)

    for row, pred_v, axis_pos in zip(all_items, predicted_v_nums, axis_positions):
        scalar_dists = np.abs(known_axis - pred_v)
        nearest_idx = int(np.argmin(scalar_dists))
        nearest = known_items[nearest_idx]

        row["predicted_v_num"] = float(pred_v)
        row["axis_position"] = float(axis_pos)
        row["nearest_known_person"] = nearest["person"]
        row["nearest_known_age_node"] = nearest["known_age_node"]
        row["nearest_known_v_num"] = nearest["known_v_num"]
        row["nearest_known_distance"] = float(scalar_dists[nearest_idx])

    ordered = sorted(all_items, key=lambda x: x["axis_position"])

    positions = {
        row["source_index"]: row
        for row in ordered
    }

    return {
        "model": model,
        "positions": positions,
        "ordered": ordered,
        "known_count": len(known_items),
        "total_count": len(all_items),
        "axis_min_v": min_v,
        "axis_max_v": max_v,
    }


def learn_all_positions(
    age_vectors,
    person_vectors,
    v_prefix="v",
    person_model_type="ridge",
):
    """
    Main function.

    Returns projected 1D positions for both:
        - age_vectors, ordered exactly by v1...vn
        - person_vectors, learned from known-age anchors
    """

    age_result = learn_age_vector_axis(
        age_vectors=age_vectors,
        v_prefix=v_prefix,
    )

    person_result = learn_person_age_axis(
        person_vectors=person_vectors,
        v_prefix=v_prefix,
        model_type=person_model_type,
    )

    return {
        "age_positions": age_result,
        "person_positions": person_result,
    }


def print_age_positions(result):
    for row in result["age_positions"]["ordered"]:
        print(
            row["v_node"],
            "rank=", row["rank"],
            "axis=", round(row["axis_position"], 4),
            "v_id=", row["v_id"],
        )


def print_person_positions(result):
    for row in result["person_positions"]["ordered"]:
        print(
            row["person"],
            "known=", row["known_age_node"],
            "predicted_v=", round(row["predicted_v_num"], 4),
            "axis=", round(row["axis_position"], 4),
            "nearest_known=",
            row["nearest_known_person"],
            row["nearest_known_age_node"],
            "distance=",
            round(row["nearest_known_distance"], 4),
        )