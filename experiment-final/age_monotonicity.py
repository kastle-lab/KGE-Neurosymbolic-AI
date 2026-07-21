from __future__ import annotations

import argparse, re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from pykeen.triples import TriplesFactory
from scipy.stats import kendalltau, spearmanr
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


MANIFEST = "kg_manifest.csv"
OUT = Path("final_results")
BG = "#fffdf7"
CMAP = LinearSegmentedColormap.from_list("age_blue_red", ["dodgerblue", "red"])
WINDOW = {"with_windows": "With Windowing", "without_windows": "Without Windowing"}
WORDER = {"with_windows": 0, "without_windows": 1}


def scale01(x):
    x = np.asarray(x, float).ravel()
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if np.isclose(lo, hi) else (x - lo) / (hi - lo)


def load_model(path):
    try:
        model = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        model = torch.load(path, map_location="cpu")
    model = model.to("cpu")
    model.eval()
    return model


def age_from_label(label):
    m = re.fullmatch(r"v(-?\d+(?:\.\d+)?)", str(label))
    return None if m is None else float(m.group(1))


def load_values(run):
    path, values = run / "kg.tsv.val", {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            try:
                values[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return values


def load_age_nodes(run):
    tf = TriplesFactory.from_path_binary(run / "training_triples")
    model = load_model(run / "trained_model.pkl")
    with torch.no_grad():
        emb = model.entity_representations[0](indices=None).detach().cpu().numpy()
    values, rows = load_values(run), []
    for label, idx in tf.entity_to_id.items():
        label = str(label)
        if not label.startswith("v"):
            continue
        age = values.get(label, age_from_label(label))
        if age is not None:
            rows.append((label, float(age), np.asarray(emb[int(idx)], float)))
    rows.sort(key=lambda x: x[0])  # deterministic, never sorted by age
    if len(rows) < 4:
        raise ValueError(f"Only {len(rows)} age nodes found in {run}")
    return np.vstack([x[2] for x in rows]), np.array([x[1] for x in rows])


def discover(root):
    root = Path(root)
    paths = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()
         and (p / MANIFEST).exists() and (p / "runs").is_dir()],
        key=lambda p: int(p.name),
    )
    if not paths:
        raise FileNotFoundError(f"No population folders found under {root}")
    return paths


def load_manifest(pop):
    df = pd.read_csv(pop / MANIFEST)
    if "label" in df and "run" not in df:
        df = df.rename(columns={"label": "run"})
    needed = {"run", "window_condition", "removal_percent"}
    if missing := needed - set(df.columns):
        raise ValueError(f"{pop / MANIFEST} missing columns: {sorted(missing)}")
    return df.sort_values(["window_condition", "removal_percent", "run"])


def orient_coordinate(coord, ages):
    coord = np.asarray(coord, float)
    rho = spearmanr(ages, coord).statistic
    tau = kendalltau(ages, coord, variant="b").statistic
    flip = np.isfinite(rho) and (rho < 0 or (np.isclose(rho, 0) and np.isfinite(tau) and tau < 0))
    return (1.0 - coord if flip else coord), flip


def correlations(coord, ages):
    rho, rp = spearmanr(ages, coord)
    tau, tp = kendalltau(ages, coord, variant="b")
    return float(rho), float(rp), float(tau), float(tp)


def fit_2d_curve(vectors, ages, args):
    """Fit one regularized MLP in PCA-2 space; reuse it for metrics and plotting."""
    scaled = StandardScaler().fit_transform(np.asarray(vectors, float))
    pca = PCA(n_components=2)
    node_xy = pca.fit_transform(scaled)

    xy_scaler = StandardScaler()
    fit_xy = xy_scaler.fit_transform(node_xy)
    initial_t = scale01(fit_xy[:, 0])

    mlp = MLPRegressor(
        hidden_layer_sizes=tuple(args.hidden_layers),
        activation="tanh",
        solver="lbfgs",
        alpha=args.mlp_alpha,
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    mlp.fit(initial_t[:, None], fit_xy)

    grid = np.linspace(0, 1, args.grid_size)
    curve_fit = mlp.predict(grid[:, None])
    nearest = np.argmin(((fit_xy[:, None, :] - curve_fit[None, :, :]) ** 2).sum(2), axis=1)
    mlp_t, flip = orient_coordinate(grid[nearest], ages)
    if flip:
        curve_fit = curve_fit[::-1]

    curve_xy = xy_scaler.inverse_transform(curve_fit)
    pca_t, _ = orient_coordinate(scale01(node_xy[:, 0]), ages)
    pca_rho, pca_rp, pca_tau, pca_tp = correlations(pca_t, ages)
    mlp_rho, mlp_rp, mlp_tau, mlp_tp = correlations(mlp_t, ages)
    distances = np.sqrt(((fit_xy - curve_fit[(len(grid) - 1 - nearest) if flip else nearest]) ** 2).sum(1))

    return {
        "node_xy": node_xy,
        "curve_xy": curve_xy,
        "pca_rho": pca_rho, "pca_rho_p": pca_rp,
        "pca_tau": pca_tau, "pca_tau_p": pca_tp,
        "mlp_rho": mlp_rho, "mlp_rho_p": mlp_rp,
        "mlp_tau": mlp_tau, "mlp_tau_p": mlp_tp,
        "pca_pc1_variance": float(pca.explained_variance_ratio_[0]),
        "pca_2d_variance": float(pca.explained_variance_ratio_.sum()),
        "mlp_mean_curve_distance": float(distances.mean()),
        "mlp_max_curve_distance": float(distances.max()),
    }


def gradient_curve(ax, xy):
    seg = np.stack([xy[:-1], xy[1:]], axis=1)
    line = LineCollection(seg, cmap=CMAP, norm=Normalize(0, 1), linewidths=4.5, zorder=2)
    line.set_array(np.linspace(0, 1, len(seg)))
    ax.add_collection(line)


def plot_curve(result, ages, title, path, dpi):
    xy, curve = result["node_xy"], result["curve_xy"]
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot(curve[:, 0], curve[:, 1], color="black", linewidth=7, alpha=.18, zorder=1)
    gradient_curve(ax, curve)
    points = ax.scatter(
        xy[:, 0], xy[:, 1], c=ages, cmap=CMAP,
        norm=Normalize(float(ages.min()), float(ages.max())),
        s=80, edgecolors="white", linewidths=.8, zorder=3,
    )
    ax.set(
        title=title + f"\nMLP ρ={result['mlp_rho']:.3f}, τb={result['mlp_tau']:.3f}",
        xlabel="Age-node embedding PC1", ylabel="Age-node embedding PC2",
    )
    ax.grid(alpha=.22, linewidth=.5)
    ax.set_aspect("equal", adjustable="box")
    cb = fig.colorbar(points, ax=ax, pad=.03)
    cb.set_label("Ground-truth age (nodes)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_summary(df, condition, path, gap=False, dpi=300):
    data = df[df.window_condition == condition]
    if data.empty:
        return
    pops = sorted(data.population_size.unique(), reverse=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, facecolor=BG)
    metrics = [("rho", "Spearman ρ"), ("tau", "Kendall τb")]
    for ax, (suffix, label) in zip(axes, metrics):
        ax.set_facecolor(BG)
        ax.grid(True, linewidth=.45, alpha=.3)
        for i, pop in enumerate(pops):
            x = data[data.population_size == pop].sort_values("removal_percent")
            color = colors[i % len(colors)]
            if gap:
                y = x[f"mlp_{suffix}"] - x[f"pca_{suffix}"]
                ax.plot(x.removal_percent, y, "o-", color=color, linewidth=2, label=f"Population {pop}")
            else:
                ax.plot(x.removal_percent, x[f"pca_{suffix}"], "o-", color=color, linewidth=2, label=f"PCA — {pop}")
                ax.plot(x.removal_percent, x[f"mlp_{suffix}"], "s--", color=color, linewidth=2, label=f"MLP — {pop}")
        if gap:
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            ax.set_ylabel("MLP − PCA")
        else:
            ax.set_ylim(-.05, 1.05)
            ax.set_ylabel("Rank correlation")
        ax.set_title(("Δ " if gap else "") + label)
        ax.set_xlabel("Removed hasAge relations (%)")
        ax.set_xticks(sorted(data.removal_percent.unique()))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(1, min(4, len(labels))), fontsize=8)
    fig.suptitle(
        ("MLP − PCA Monotonicity — " if gap else "Age-Embedding Monotonicity — ") + WINDOW[condition],
        fontsize=14,
    )
    fig.tight_layout(rect=(0, .1, 1, .94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def markdown_table(df, condition):
    x = df[df.window_condition == condition].sort_values(["population_size", "removal_percent"], ascending=[False, True])
    cols = [
        ("Population", "population_size", lambda v: str(int(v))),
        ("Removal %", "removal_percent", lambda v: f"{v:g}%"),
        ("PCA Spearman ρ", "pca_rho", lambda v: f"{v:.4f}"),
        ("PCA Kendall τb", "pca_tau", lambda v: f"{v:.4f}"),
        ("MLP Spearman ρ", "mlp_rho", lambda v: f"{v:.4f}"),
        ("MLP Kendall τb", "mlp_tau", lambda v: f"{v:.4f}"),
    ]
    lines = ["| " + " | ".join(c[0] for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(fmt(row[key]) for _, key, fmt in cols) + " |" for _, row in x.iterrows()]
    return "\n".join(lines)


def write_report(df, out):
    df.to_csv(out / "age-monotonicity.csv", index=False)
    parts = [
        "# Age-Node Embedding Monotonicity\n",
        "Age-node embeddings are standardized once and projected into one shared PCA-2 space per run. "
        "PCA statistics use PC1. One regularized MLP maps normalized PC1 to the PCA-2 coordinates; "
        "each node's nearest position on that same fitted curve supplies the MLP coordinate used for "
        "Spearman and Kendall statistics. The exact same curve is drawn in the per-run visualizations.\n",
        "Ground-truth age is used only after fitting to orient each one-dimensional coordinate toward increasing age.\n",
    ]
    for condition in ("with_windows", "without_windows"):
        parts += [
            f"## {WINDOW[condition]}\n",
            markdown_table(df, condition) + "\n",
            f"![Monotonicity](age-monotonicity-plots/{condition}_monotonicity.png)\n",
            f"![MLP minus PCA](age-monotonicity-plots/{condition}_gap.png)\n",
        ]
    (out / "age-monotonicity.md").write_text("\n".join(parts), encoding="utf-8")


def analyze(populations, args):
    out, rows = Path(args.output_directory), []
    curve_dir, summary_dir = out / "age-node-mlp-curves", out / "age-monotonicity-plots"
    out.mkdir(parents=True, exist_ok=True)

    for pop in populations:
        size = int(pop.name)
        for row in load_manifest(pop).itertuples(index=False):
            condition, removal, run_label = str(row.window_condition), float(row.removal_percent), str(row.run)
            run = pop / "runs" / run_label
            print(f"Analyzing population {size}, {condition}, {removal:g}% removed")
            vectors, ages = load_age_nodes(run)
            result = fit_2d_curve(vectors, ages, args)  # one fit, reused below
            rows.append({
                "population_size": size, "run": run_label, "window_condition": condition,
                "removal_percent": removal, "n_age_nodes": len(ages),
                **{k: v for k, v in result.items() if not isinstance(v, np.ndarray)},
            })
            name = f"population_{size}_{condition}_removed_{str(removal).replace('.', 'p')}pct.png"
            plot_curve(
                result, ages,
                f"Regularized MLP Curve in PCA-2\nPopulation {size} · {WINDOW.get(condition, condition)} · {removal:g}% removed",
                curve_dir / name, args.dpi,
            )

    df = pd.DataFrame(rows)
    df["_w"] = df.window_condition.map(WORDER)
    df = df.sort_values(["_w", "population_size", "removal_percent"], ascending=[True, False, True]).drop(columns="_w")
    for condition in ("with_windows", "without_windows"):
        plot_summary(df, condition, summary_dir / f"{condition}_monotonicity.png", dpi=args.dpi)
        plot_summary(df, condition, summary_dir / f"{condition}_gap.png", gap=True, dpi=args.dpi)
    write_report(df, out)
    print(f"Saved report, CSV, summary plots, and per-run curves to {out}")
    return df


def parser():
    p = argparse.ArgumentParser(description="PCA-2 age-node monotonicity using one regularized MLP curve for both metrics and visualization.")
    p.add_argument("--basepath", nargs="+")
    p.add_argument("--root", default=".")
    p.add_argument("--output-directory", default=str(OUT))
    p.add_argument("--hidden-layers", nargs="+", type=int, default=[8], help="Keep small to avoid spiky curves. Default: 8")
    p.add_argument("--mlp-alpha", type=float, default=.1, help="L2 regularization. Default: 0.1")
    p.add_argument("--max-iter", type=int, default=5000)
    p.add_argument("--grid-size", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=300)
    return p


def main():
    args = parser().parse_args()
    populations = [Path(x) for x in args.basepath] if args.basepath else discover(args.root)
    print("Population folders:", ", ".join(map(str, populations)))
    analyze(populations, args)


if __name__ == "__main__":
    main()