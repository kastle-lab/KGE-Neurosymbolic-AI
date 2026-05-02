from pathlib import Path
import torch
from pykeen.triples import TriplesFactory

def get_mure_params(model):
    return {
        "entity_embeddings": model.entity_representations[0](indices=None).detach().cpu().numpy(),
        "relation_embeddings": model.relation_representations[0](indices=None).detach().cpu().numpy(),
        "relation_specific_embeddings": model.relation_representations[1](indices=None).detach().cpu().numpy(),
    }

def compute_query_point(h_vec, r_vec, R_vec):
    return h_vec * R_vec + r_vec


def load_model(run_path):
    run_path = Path(run_path)
    
    tf = TriplesFactory.from_path_binary(run_path / "training_triples")
    model = torch.load(run_path / "trained_model.pkl", weights_only=False)
    params = get_mure_params(model)
    
    return params

def load_person_vectors(
    run_path,
    original_index=None,
    relation_filter="hasAge",
    head_prefix="person",
):
    run_path = Path(run_path)

    tf = TriplesFactory.from_path_binary(run_path / "training_triples")
    model = torch.load(run_path / "trained_model.pkl", weights_only=False)
    params = get_mure_params(model)

    # Map head/person -> matching age triple
    age_triples_by_head = {}

    for triple in tf.mapped_triples:
        h_id, r_id, t_id = map(int, triple.tolist())

        h_label = tf.entity_id_to_label[h_id]
        r_label = tf.relation_id_to_label[r_id]
        t_label = tf.entity_id_to_label[t_id]

        if relation_filter is not None and r_label != relation_filter:
            continue

        if head_prefix is not None and not h_label.startswith(head_prefix):
            continue

        age_triples_by_head[h_label] = {
            "head_id": h_id,
            "relation_id": r_id,
            "tail_id": t_id,
            "triple_name": [h_label, r_label, t_label],
        }

    results = {}

    if original_index is None:
        triple_index = 0

        # Iterate over all person entities, not just existing hasAge triples
        for h_id, h_label in tf.entity_id_to_label.items():
            if head_prefix is not None and not h_label.startswith(head_prefix):
                continue

            h_vec = params["entity_embeddings"][h_id]

            if h_label in age_triples_by_head:
                info = age_triples_by_head[h_label]

                r_id = info["relation_id"]
                t_id = info["tail_id"]

                r_vec = params["relation_embeddings"][r_id]
                R_vec = params["relation_specific_embeddings"][r_id]
                t_vec = params["entity_embeddings"][t_id]
                q_vec = compute_query_point(h_vec, r_vec, R_vec)

                results[triple_index] = {
                    "triple_index": triple_index,
                    "triple_name": info["triple_name"],
                    "exists": True,
                    "head_id": h_id,
                    "relation_id": r_id,
                    "tail_id": t_id,
                    "head_vec": h_vec,
                    "query_vec": q_vec,
                    "tail_vec": t_vec,
                }

            else:
                # Even if the age triple is missing, we can still compute
                # where this person points under the hasAge relation.
                r_id = tf.relation_to_id[relation_filter]

                r_vec = params["relation_embeddings"][r_id]
                R_vec = params["relation_specific_embeddings"][r_id]
                q_vec = compute_query_point(h_vec, r_vec, R_vec)

                results[triple_index] = {
                    "triple_index": triple_index,
                    "triple_name": [h_label, relation_filter, None],
                    "exists": False,
                    "head_id": h_id,
                    "relation_id": r_id,
                    "tail_id": None,
                    "head_vec": h_vec,
                    "query_vec": q_vec,
                    "tail_vec": None,
                }
                
            triple_index += 1

        return results

def load_age_vectors(
    run_path,
    relation_filter="hasAge",
    v_prefix="v",
):
    run_path = Path(run_path)

    tf = TriplesFactory.from_path_binary(run_path / "training_triples")
    model = torch.load(run_path / "trained_model.pkl", weights_only=False)
    params = get_mure_params(model)

    def v_sort_key(label):
        try:
            return int(label[len(v_prefix):])
        except ValueError:
            return float("inf")

    v_labels = [
        label
        for label in tf.entity_to_id.keys()
        if label.startswith(v_prefix)
    ]

    v_labels = sorted(v_labels, key=v_sort_key)

    results = {}

    for age_index, v_label in enumerate(v_labels):
        v_id = tf.entity_to_id[v_label]

        results[age_index] = {
            "age_index": age_index,
            "v_node": v_label,
            "v_id": v_id,
            "v_vec": params["entity_embeddings"][v_id],
            "exists": True,
        }

    return results

def load_hasAge_bundle(
    run_path,
    relation_filter="hasAge",
    person_prefix="person",
    v_prefix="v",
):
    run_path = Path(run_path)

    tf = TriplesFactory.from_path_binary(run_path / "training_triples")
    model = torch.load(run_path / "trained_model.pkl", weights_only=False)
    params = get_mure_params(model)

    def numeric_suffix_sort_key(label, prefix):
        try:
            return int(label[len(prefix):])
        except ValueError:
            return float("inf")

    # ----------------------------
    # v node embeddings
    # ----------------------------
    v_embeddings = {}

    v_labels = [
        label
        for label in tf.entity_to_id.keys()
        if label.startswith(v_prefix)
    ]

    v_labels = sorted(
        v_labels,
        key=lambda label: numeric_suffix_sort_key(label, v_prefix)
    )

    for v_label in v_labels:
        v_id = tf.entity_to_id[v_label]
        v_embeddings[v_label] = params["entity_embeddings"][v_id]

    # ----------------------------
    # person embeddings
    # ----------------------------
    person_embeddings = {}

    person_labels = [
        label
        for label in tf.entity_to_id.keys()
        if label.startswith(person_prefix)
    ]

    person_labels = sorted(
        person_labels,
        key=lambda label: numeric_suffix_sort_key(label, person_prefix)
    )

    for person_label in person_labels:
        person_id = tf.entity_to_id[person_label]
        person_embeddings[person_label] = params["entity_embeddings"][person_id]

    # ----------------------------
    # person -> v mappings
    # only for existing hasAge triples
    # ----------------------------
    person_to_v = {}

    for triple in tf.mapped_triples:
        h_id, r_id, t_id = map(int, triple.tolist())

        h_label = tf.entity_id_to_label[h_id]
        r_label = tf.relation_id_to_label[r_id]
        t_label = tf.entity_id_to_label[t_id]

        if r_label != relation_filter:
            continue

        if not h_label.startswith(person_prefix):
            continue

        if not t_label.startswith(v_prefix):
            continue

        person_to_v[h_label] = t_label

    # ----------------------------
    # hasAge relation vectors
    # ----------------------------
    relation_id = tf.relation_to_id[relation_filter]

    hasAge_relation = {
        "relation_id": relation_id,
        "r_vec": params["relation_embeddings"][relation_id],
        "R_vec": params["relation_specific_embeddings"][relation_id],
    }

    return {
        "v_embeddings": v_embeddings,
        "person_embeddings": person_embeddings,
        "person_to_v": person_to_v,
        "hasAge_relation": hasAge_relation,
    }
    
    