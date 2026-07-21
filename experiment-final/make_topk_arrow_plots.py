from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from pykeen.triples import TriplesFactory
from sklearn.decomposition import PCA


RELATION = "hasAge"
PERSON_PREFIX = "person"
AGE_PREFIX = "v"
BACKGROUND_COLOR = "#fffdf7"

WINDOW_NAMES = {
    "with_windows": "With windows",
    "without_windows": "Without windows",
}
WINDOW_ORDER = {"with_windows": 0, "without_windows": 1}


def suffix_number(label, prefix):
    m = re.fullmatch(rf"{re.escape(prefix)}(-?\d+(?:\.\d+)?)", str(label).strip(), flags=re.IGNORECASE)
    return None if m is None else float(m.group(1))


def sorted_labels(labels, prefix):
    def key(label):
        n = suffix_number(label, prefix)
        return (n is None, n if n is not None else float("inf"), str(label))
    return sorted(labels, key=key)


def get_mure_parameters(model):
    with torch.no_grad():
        entity_embeddings = model.entity_representations[0](indices=None).detach().cpu().numpy()
        relation_embeddings = model.relation_representations[0](indices=None).detach().cpu().numpy()
        relation_specific_embeddings = model.relation_representations[1](indices=None).detach().cpu().numpy()
    return {
        "entity_embeddings": entity_embeddings,
        "relation_embeddings": relation_embeddings,
        "relation_specific_embeddings": relation_specific_embeddings,
    }


def load_run_vectors(run_path):
    run_path = Path(run_path)
    triples_factory = TriplesFactory.from_path_binary(run_path / "training_triples")
    model = torch.load(run_path / "trained_model.pkl", map_location="cpu", weights_only=False).to("cpu")
    model.eval()

    params = get_mure_parameters(model)
    relation_id = int(triples_factory.relation_to_id[RELATION])
    rel_t = params["relation_embeddings"][relation_id]
    rel_s = params["relation_specific_embeddings"][relation_id]

    person_labels = sorted_labels([x for x in triples_factory.entity_to_id if str(x).startswith(PERSON_PREFIX)], PERSON_PREFIX)
    age_labels = sorted_labels(
        [x for x in triples_factory.entity_to_id if str(x).startswith(AGE_PREFIX) and suffix_number(x, AGE_PREFIX) is not None],
        AGE_PREFIX,
    )

    person_vectors = {
        p: params["entity_embeddings"][int(triples_factory.entity_to_id[p])]
        for p in person_labels
    }
    age_vectors = {
        a: params["entity_embeddings"][int(triples_factory.entity_to_id[a])]
        for a in age_labels
    }
    query_vectors = {p: person_vectors[p] * rel_s + rel_t for p in person_labels}
    return person_vectors, query_vectors, age_vectors


def load_person_to_age(tsv_path):
    person_to_age = {}
    with Path(tsv_path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"Malformed triple at line {i} in {tsv_path}")
            h, r, t = parts
            if i == 1 and [h, r, t] in (["?sub", "?pred", "?val"], ["sub", "pred", "val"]):
                continue
            if r == RELATION:
                person_to_age[h] = t
    if not person_to_age:
        raise ValueError(f"No {RELATION} triples found in {tsv_path}")
    return person_to_age


def load_manifest(pop_dir):
    manifest = pd.read_csv(Path(pop_dir) / "kg_manifest.csv")
    run_col = "label" if "label" in manifest.columns else "run"
    manifest = manifest.rename(columns={run_col: "run"}).copy()
    manifest["window_condition"] = manifest["window_condition"].astype(str)
    manifest["removal_percent"] = pd.to_numeric(manifest["removal_percent"], errors="raise")
    manifest = manifest.loc[manifest["window_condition"].isin(WINDOW_NAMES)].copy()
    manifest["_window_order"] = manifest["window_condition"].map(WINDOW_ORDER)
    return manifest.sort_values(["_window_order", "removal_percent", "run"]).reset_index(drop=True)


def find_original_tsv(pop_dir, window_condition):
    kg_dir = Path(pop_dir) / "kgs"
    exact = kg_dir / f"{window_condition}_original.tsv"
    if exact.exists():
        return exact
    matches = sorted(kg_dir.glob(f"{window_condition}*original*.tsv"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Could not find original TSV for {window_condition} under {kg_dir}")
    raise ValueError(f"Multiple original TSV matches for {window_condition}: {matches}")


def discover_population_directories(basepath, requested=None):
    basepath = Path(basepath)
    if requested:
        dirs = [basepath / str(x) for x in requested]
    else:
        dirs = sorted(
            [
                p for p in basepath.iterdir()
                if p.is_dir() and p.name.isdigit() and (p / "kg_manifest.csv").exists() and (p / "runs").is_dir()
            ],
            key=lambda p: int(p.name),
        )
    if not dirs:
        raise ValueError(f"No population directories found under {basepath}")
    return dirs


def evaluate_triples(person_vectors, query_vectors, age_vectors, truth_person_to_age):
    age_labels = list(age_vectors)
    age_matrix = np.vstack([age_vectors[a] for a in age_labels])
    age_index = {a: i for i, a in enumerate(age_labels)}

    rows = []
    for person, q in query_vectors.items():
        true_age = truth_person_to_age.get(person)
        if true_age not in age_index:
            continue
        dists = np.linalg.norm(age_matrix - q, axis=1)
        order = np.argsort(dists)
        true_idx = age_index[true_age]
        pred_idx = int(order[0])
        true_rank = int(np.where(order == true_idx)[0][0]) + 1
        rows.append({
            "person": person,
            "true_age": true_age,
            "pred_age": age_labels[pred_idx],
            "true_rank": true_rank,
            "true_distance": float(dists[true_idx]),
            "pred_distance": float(dists[pred_idx]),
            "pred_age_gap": abs((suffix_number(age_labels[pred_idx], AGE_PREFIX) or 0) - (suffix_number(true_age, AGE_PREFIX) or 0)),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["true_rank", "true_distance", "pred_age_gap", "person"]).reset_index(drop=True)


def perform_joint_pca(person_vectors, query_vectors, age_vectors):
    person_labels = list(person_vectors)
    age_labels = list(age_vectors)

    person_matrix = np.vstack([person_vectors[p] for p in person_labels])
    query_matrix = np.vstack([query_vectors[p] for p in person_labels])
    age_matrix = np.vstack([age_vectors[a] for a in age_labels])

    combined = np.vstack([person_matrix, query_matrix, age_matrix])
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(combined)

    n_people, n_ages = len(person_labels), len(age_labels)
    return {
        "person_labels": person_labels,
        "age_labels": age_labels,
        "person_xy": reduced[:n_people],
        "query_xy": reduced[n_people:n_people * 2],
        "age_xy": reduced[n_people * 2:n_people * 2 + n_ages],
        "explained_variance": pca.explained_variance_ratio_,
    }


def plot_topk(
    population_size,
    window_condition,
    removal_percent,
    run_name,
    person_vectors,
    query_vectors,
    age_vectors,
    truth_person_to_age,
    k,
    output_path,
    dpi=300,
):
    eval_df = evaluate_triples(person_vectors, query_vectors, age_vectors, truth_person_to_age)
    if eval_df.empty:
        print(f"SKIPPED: {run_name} (no evaluable triples)")
        return False

    top_df = eval_df.head(min(k, len(eval_df))).copy()
    reduced = perform_joint_pca(person_vectors, query_vectors, age_vectors)

    person_labels = reduced["person_labels"]
    age_labels = reduced["age_labels"]
    person_xy = reduced["person_xy"]
    query_xy = reduced["query_xy"]
    age_xy = reduced["age_xy"]
    ev = reduced["explained_variance"]

    person_idx = {p: i for i, p in enumerate(person_labels)}
    age_idx = {a: i for i, a in enumerate(age_labels)}

    age_values = np.array([suffix_number(a, AGE_PREFIX) for a in age_labels], dtype=float)
    age_min, age_max = float(age_values.min()), float(age_values.max())
    if np.isclose(age_min, age_max):
        age_min -= 0.5
        age_max += 0.5

    age_norm = Normalize(vmin=age_min, vmax=age_max)
    age_cmap = LinearSegmentedColormap.from_list("age_gradient", ["dodgerblue", "red"])
    person_cmap = LinearSegmentedColormap.from_list("person_gradient", ["black", "lightgray"])

    fig, ax = plt.subplots(figsize=(12, 9), facecolor=BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    # background age nodes for context
    ax.scatter(
        age_xy[:, 0], age_xy[:, 1],
        s=24, c=age_values, cmap=age_cmap, norm=age_norm,
        edgecolors="none", alpha=0.18, zorder=1
    )

    pq_segments, qa_segments, qa_colors = [], [], []
    selected_true_ages = set()

    for row in top_df.itertuples(index=False):
        pi, ai = person_idx[row.person], age_idx[row.true_age]
        selected_true_ages.add(row.true_age)
        pq_segments.append([person_xy[pi], query_xy[pi]])
        qa_segments.append([query_xy[pi], age_xy[ai]])
        qa_colors.append(age_cmap(age_norm(suffix_number(row.true_age, AGE_PREFIX))))

    if pq_segments:
        ax.add_collection(LineCollection(pq_segments, colors="#383838", linewidths=1.3, alpha=0.55, zorder=2))
    if qa_segments:
        ax.add_collection(LineCollection(qa_segments, colors=qa_colors, linewidths=1.7, linestyles="dotted", alpha=0.75, zorder=2))

    # selected people + queries
    for row in top_df.itertuples(index=False):
        pi = person_idx[row.person]
        ai = age_idx[row.true_age]
        true_age_val = suffix_number(row.true_age, AGE_PREFIX)
        color_person = person_cmap(age_norm(true_age_val))
        color_query = age_cmap(age_norm(true_age_val))

        px, py = person_xy[pi]
        qx, qy = query_xy[pi]
        ax.scatter([px], [py], s=52, marker="o", c=[color_person], edgecolors="#202020", linewidths=0.6, zorder=4)

        direction = np.array([qx - px, qy - py])
        angle = 0.0 if np.allclose(direction, 0) else np.degrees(np.arctan2(direction[1], direction[0]))
        marker = MarkerStyle(">").transformed(Affine2D().rotate_deg(angle))
        ax.scatter([qx], [qy], s=110, marker=marker, facecolors=BACKGROUND_COLOR, edgecolors=[color_query], linewidths=1.45, zorder=5)

        ax.annotate(
            row.person,
            (px, py),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="#222222",
            zorder=6,
        )

    # selected true age nodes
    for age_label in sorted(selected_true_ages, key=lambda x: suffix_number(x, AGE_PREFIX)):
        ai = age_idx[age_label]
        val = suffix_number(age_label, AGE_PREFIX)
        ax.scatter(
            [age_xy[ai, 0]], [age_xy[ai, 1]],
            s=115, marker="o", c=[age_cmap(age_norm(val))],
            edgecolors="#202020", linewidths=0.75, zorder=6
        )
        ax.annotate(
            age_label,
            (age_xy[ai, 0], age_xy[ai, 1]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="#222222",
            zorder=7,
        )

    summary_lines = [
        f"{i+1}. {r.person} → {r.true_age} | rank={r.true_rank} | dist={r.true_distance:.4f}"
        for i, r in enumerate(top_df.itertuples(index=False))
    ]
    ax.text(
        1.02, 0.98,
        "Top triples\n(by true-age rank, then distance)\n\n" + "\n".join(summary_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=BACKGROUND_COLOR, edgecolor="#d8d3c8", alpha=0.97),
    )

    legend = ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="gray", markeredgecolor="#202020", markersize=6, label="Selected person"),
            Line2D([0], [0], marker=">", linestyle="none", markerfacecolor=BACKGROUND_COLOR, markeredgecolor="dodgerblue", markersize=7, label="Selected query"),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="red", markeredgecolor="#202020", markersize=7, label="Selected true age"),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="lightgray", markeredgecolor="none", alpha=0.4, markersize=6, label="All age nodes (context)"),
            Line2D([0, 1], [0, 0], color="#383838", linewidth=1.3, label="Person → query"),
            Line2D([0, 1], [0, 0], color="dodgerblue", linewidth=1.7, linestyle="dotted", label="Query → true age"),
        ],
        loc="lower left",
        fontsize=8,
        frameon=True,
        facecolor=BACKGROUND_COLOR,
        edgecolor="#d8d3c8",
        framealpha=0.95,
    )
    legend.get_frame().set_linewidth(0.7)

    removal_text = f"{float(removal_percent):g}%"
    ax.set_title(
        f"Population {population_size} | {WINDOW_NAMES[window_condition]} | {removal_text} hasAge removed\n"
        f"Run: {run_name} | Top {len(top_df)} triple{'s' if len(top_df) != 1 else ''}"
    )
    ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}% variance)")
    ax.grid(True, linewidth=0.35, color="#d8d3c8", alpha=0.35)
    ax.margins(0.08)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def make_all_plots(basepath=".", populations=None, ks=(1, 5, 10), output_dir="final_results/arrow_plots", dpi=300):
    pop_dirs = discover_population_directories(basepath, populations)
    output_dir = Path(output_dir)
    made, failed = 0, []

    for pop_dir in pop_dirs:
        pop_size = int(pop_dir.name)
        manifest = load_manifest(pop_dir)
        truth_cache = {}

        for row in manifest.itertuples(index=False):
            run_name = str(row.run)
            window_condition = str(row.window_condition)
            removal_percent = float(row.removal_percent)
            run_path = pop_dir / "runs" / run_name

            try:
                if window_condition not in truth_cache:
                    truth_cache[window_condition] = load_person_to_age(find_original_tsv(pop_dir, window_condition))

                person_vectors, query_vectors, age_vectors = load_run_vectors(run_path)

                for k in ks:
                    subdir = output_dir / f"{k}_triple" if k == 1 else output_dir / f"{k}_triples"
                    filename = (
                        f"population_{pop_size}_"
                        f"{window_condition}_"
                        f"removed_{removal_percent:g}pct_"
                        f"{run_name}_top_{k}.png"
                    )
                    ok = plot_topk(
                        population_size=pop_size,
                        window_condition=window_condition,
                        removal_percent=removal_percent,
                        run_name=run_name,
                        person_vectors=person_vectors,
                        query_vectors=query_vectors,
                        age_vectors=age_vectors,
                        truth_person_to_age=truth_cache[window_condition],
                        k=k,
                        output_path=subdir / filename,
                        dpi=dpi,
                    )
                    if ok:
                        made += 1
                        print(f"Saved: {subdir / filename}")

            except Exception as e:
                failed.append((str(run_path), str(e)))
                print(f"FAILED: {run_path}\n  {e}")

    print(f"\nGenerated {made} plot(s) in {output_dir}")
    if failed:
        print(f"{len(failed)} run(s) failed:")
        for run_path, msg in failed:
            print(f"  - {run_path}: {msg}")


def build_parser():
    p = argparse.ArgumentParser(description="Generate top-k arrow plots (top 1, 5, 10) for MuRE hasAge runs.")
    p.add_argument("--basepath", default=".", help="Folder containing population folders such as 100/, 200/, 500/")
    p.add_argument("--populations", nargs="*", help="Optional population folders, e.g. --populations 100 200 500")
    p.add_argument("--ks", nargs="*", type=int, default=[1, 5, 10], help="Top-k values to plot. Default: 1 5 10")
    p.add_argument("--output-dir", default="final_results/arrow_plots", help="Output directory")
    p.add_argument("--dpi", type=int, default=300, help="Image DPI")
    return p


def main():
    args = build_parser().parse_args()
    make_all_plots(
        basepath=args.basepath,
        populations=args.populations,
        ks=tuple(args.ks),
        output_dir=args.output_dir,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()