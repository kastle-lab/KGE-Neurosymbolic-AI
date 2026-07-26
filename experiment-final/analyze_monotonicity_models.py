from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pykeen.triples import TriplesFactory
from scipy.stats import kendalltau, spearmanr
from sklearn.decomposition import PCA

DEFAULT_MODELS = ("transe", "distmult", "transd", "transr", "mure")
DEFAULT_OUTPUT_DIRECTORY = Path("final_results/monotonicity_analysis")
MANIFEST_FILE = "kg_manifest.csv"
HASH_FILE = "shared_kg_hashes.csv"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
    return df


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_suffix(label: str, prefix: str) -> float:
    try:
        return int(label[len(prefix):])
    except ValueError:
        return float("inf")


def find_run_path(model_root: Path, run_label: str) -> Path:
    path = model_root / "runs" / run_label
    if not path.exists():
        raise FileNotFoundError(f"Missing run directory: {path}")
    return path


def load_entity_embedding_matrix(run_path: Path):
    tf = TriplesFactory.from_path_binary(run_path / "training_triples")
    model = torch.load(run_path / "trained_model.pkl", weights_only=False)
    embeddings = (
        model.entity_representations[0](indices=None)
        .detach()
        .cpu()
        .numpy()
    )
    return tf, embeddings


def extract_age_embeddings(run_path: Path, v_prefix: str = "v"):
    tf, entity_embeddings = load_entity_embedding_matrix(run_path)

    labels = [label for label in tf.entity_to_id if label.startswith(v_prefix)]
    labels = sorted(labels, key=lambda x: numeric_suffix(x, v_prefix))

    if len(labels) < 2:
        raise ValueError(f"Need at least 2 age literal nodes in {run_path}; found {len(labels)}")

    ages, vectors = [], []
    for label in labels:
        suffix = label[len(v_prefix):]
        try:
            age = float(suffix)
        except ValueError as exc:
            raise ValueError(
                f"Could not parse numeric age from literal node '{label}' in {run_path}"
            ) from exc
        ages.append(age)
        vectors.append(entity_embeddings[tf.entity_to_id[label]])

    return np.asarray(ages, dtype=float), np.asarray(vectors, dtype=float), labels


def compute_monotonicity(ages: np.ndarray, vectors: np.ndarray):
    pc1 = PCA(n_components=1).fit_transform(vectors).ravel()

    raw_spearman = spearmanr(ages, pc1)
    raw_kendall = kendalltau(ages, pc1)

    rho = float(raw_spearman.statistic)
    tau = float(raw_kendall.statistic)
    sign = -1.0 if np.isfinite(rho) and rho < 0 else 1.0
    oriented_pc1 = pc1 * sign

    oriented_spearman = spearmanr(ages, oriented_pc1)
    oriented_kendall = kendalltau(ages, oriented_pc1)

    summary = {
        "n_age_literals": int(len(ages)),
        "spearman_rho_raw": rho,
        "spearman_p_raw": float(raw_spearman.pvalue),
        "kendall_tau_raw": tau,
        "kendall_p_raw": float(raw_kendall.pvalue),
        "spearman_abs": abs(rho),
        "kendall_abs": abs(tau),
        "spearman_oriented": float(oriented_spearman.statistic),
        "kendall_oriented": float(oriented_kendall.statistic),
    }

    points = pd.DataFrame({
        "age": ages,
        "pc1_raw": pc1,
        "pc1_oriented": oriented_pc1,
    })
    return summary, points


def load_manifest(shared_root: Path) -> pd.DataFrame:
    manifest = read_csv(shared_root / MANIFEST_FILE)
    if "label" in manifest.columns and "run" not in manifest.columns:
        manifest = manifest.rename(columns={"label": "run"})

    required = {"run", "window_condition", "removal_percent"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"{shared_root / MANIFEST_FILE} missing columns: {sorted(missing)}")

    manifest = manifest.copy()
    manifest["run"] = manifest["run"].astype(str)
    manifest["window_condition"] = manifest["window_condition"].astype(str)
    manifest["removal_percent"] = pd.to_numeric(manifest["removal_percent"], errors="raise")
    return manifest


def infer_models(root: Path) -> list[str]:
    models = [name for name in DEFAULT_MODELS if (root / name / "runs").exists()]
    if not models:
        models = sorted(
            path.name for path in root.iterdir()
            if path.is_dir() and (path / "runs").exists()
        )
    if not models:
        raise ValueError(f"No model directories found under {root}")
    return models


def verify_shared_kg_hashes(root: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    shared_root = root / "shared_kgs"
    hash_path = shared_root / HASH_FILE

    path_column = next((c for c in ("tsv_path", "path", "kg_path") if c in manifest.columns), None)
    if path_column is None:
        return pd.DataFrame()

    records = []
    for row in manifest.itertuples(index=False):
        run = str(getattr(row, "run"))
        raw_path = Path(str(getattr(row, path_column)))
        kg_path = raw_path if raw_path.is_absolute() else (shared_root / raw_path)
        if not kg_path.exists():
            alt = root / raw_path
            if alt.exists():
                kg_path = alt
        if not kg_path.exists():
            raise FileNotFoundError(f"Shared KG for {run} not found: {raw_path}")
        records.append({"run": run, "kg_path": str(kg_path.resolve()), "sha256": sha256(kg_path)})

    calculated = pd.DataFrame(records)

    if hash_path.exists():
        existing = pd.read_csv(hash_path)
        if {"run", "sha256"}.issubset(existing.columns):
            merged = calculated.merge(
                existing[["run", "sha256"]], on="run", suffixes=("_calculated", "_recorded"), how="left"
            )
            mismatch = merged["sha256_recorded"].notna() & (
                merged["sha256_calculated"] != merged["sha256_recorded"]
            )
            if mismatch.any():
                raise ValueError(
                    f"Shared KG hash mismatch for runs: {merged.loc[mismatch, 'run'].tolist()}"
                )
    return calculated


def analyze_experiment(
    root: Path,
    output_directory: Path,
    models: list[str] | None = None,
    v_prefix: str = "v",
    force: bool = False,
    dpi: int = 300,
) -> None:
    if output_directory.exists() and any(output_directory.iterdir()) and not force:
        raise FileExistsError(
            f"Output directory is not empty: {output_directory}\nUse --force to overwrite analysis outputs."
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    plots_dir = output_directory / "plots"
    pc1_dir = plots_dir / "pc1"
    plots_dir.mkdir(parents=True, exist_ok=True)
    pc1_dir.mkdir(parents=True, exist_ok=True)

    shared_root = root / "shared_kgs"
    manifest = load_manifest(shared_root)
    calculated_hashes = verify_shared_kg_hashes(root, manifest)
    models = models or infer_models(root)

    summary_rows = []
    point_frames = []
    total_jobs = len(models) * len(manifest)
    job_index = 0

    for model_name in models:
        model_root = root / model_name
        for row in manifest.itertuples(index=False):
            job_index += 1
            run_label = str(row.run)
            window_condition = str(row.window_condition)
            removal_percent = float(row.removal_percent)
            print(f"[{job_index}/{total_jobs}] {model_name}: {run_label}")

            run_path = find_run_path(model_root, run_label)
            ages, vectors, labels = extract_age_embeddings(run_path, v_prefix=v_prefix)
            metrics, points = compute_monotonicity(ages, vectors)

            summary_rows.append({
                "model": model_name,
                "run": run_label,
                "window_condition": window_condition,
                "removal_percent": removal_percent,
                **metrics,
            })

            points = points.copy()
            points["v_node"] = labels
            points["model"] = model_name
            points["run"] = run_label
            points["window_condition"] = window_condition
            points["removal_percent"] = removal_percent
            point_frames.append(points)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(points["age"], points["pc1_oriented"], s=18)
            ax.set_xlabel("True numeric age")
            ax.set_ylabel("Oriented PCA component 1")
            ax.set_title(
                f"{model_name} — {window_condition} — {removal_percent:g}% removal\n"
                f"Spearman |ρ|={metrics['spearman_abs']:.3f}, "
                f"Kendall |τ|={metrics['kendall_abs']:.3f}"
            )
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(pc1_dir / f"{model_name}_{run_label}.png", dpi=dpi, bbox_inches="tight")
            plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    points = pd.concat(point_frames, ignore_index=True)
    summary.to_csv(output_directory / "monotonicity_results.csv", index=False)
    points.to_csv(output_directory / "pc1_age_coordinates.csv", index=False)

    rankings = (
        summary.groupby("model", as_index=False)
        .agg(
            mean_spearman_abs=("spearman_abs", "mean"),
            median_spearman_abs=("spearman_abs", "median"),
            mean_kendall_abs=("kendall_abs", "mean"),
            median_kendall_abs=("kendall_abs", "median"),
        )
    )
    rankings["combined_score"] = (
        rankings["mean_spearman_abs"] + rankings["mean_kendall_abs"]
    ) / 2.0
    rankings = rankings.sort_values(
        ["combined_score", "mean_spearman_abs", "mean_kendall_abs"], ascending=False
    ).reset_index(drop=True)
    rankings["overall_rank"] = np.arange(1, len(rankings) + 1)
    rankings.to_csv(output_directory / "monotonicity_rankings.csv", index=False)

    window_rows = []
    if {"with_windows", "without_windows"}.issubset(summary["window_condition"].unique()):
        wide = summary.pivot_table(
            index=["model", "removal_percent"],
            columns="window_condition",
            values=["spearman_abs", "kendall_abs"],
        )
        for (model, removal), row in wide.iterrows():
            try:
                with_rho = row[("spearman_abs", "with_windows")]
                without_rho = row[("spearman_abs", "without_windows")]
                with_tau = row[("kendall_abs", "with_windows")]
                without_tau = row[("kendall_abs", "without_windows")]
            except KeyError:
                continue
            if any(pd.isna(v) for v in (with_rho, without_rho, with_tau, without_tau)):
                continue
            window_rows.append({
                "model": model,
                "removal_percent": removal,
                "with_windows_spearman_abs": with_rho,
                "without_windows_spearman_abs": without_rho,
                "delta_spearman_with_minus_without": with_rho - without_rho,
                "with_windows_kendall_abs": with_tau,
                "without_windows_kendall_abs": without_tau,
                "delta_kendall_with_minus_without": with_tau - without_tau,
            })

    window_effects = pd.DataFrame(window_rows)
    window_effects.to_csv(output_directory / "window_effects.csv", index=False)

    robustness_rows = []
    for (model, window), group in summary.groupby(["model", "window_condition"]):
        group = group.sort_values("removal_percent")
        x = group["removal_percent"].to_numpy(dtype=float)
        rho = group["spearman_abs"].to_numpy(dtype=float)
        tau = group["kendall_abs"].to_numpy(dtype=float)

        rho_slope = np.polyfit(x, rho, 1)[0] * 10 if len(group) >= 2 else np.nan
        tau_slope = np.polyfit(x, tau, 1)[0] * 10 if len(group) >= 2 else np.nan
        rho_auc = np.trapz(rho, x) / (x.max() - x.min()) if x.max() > x.min() else rho[0]
        tau_auc = np.trapz(tau, x) / (x.max() - x.min()) if x.max() > x.min() else tau[0]

        robustness_rows.append({
            "model": model,
            "window_condition": window,
            "initial_removal_percent": float(x[0]),
            "final_removal_percent": float(x[-1]),
            "initial_spearman_abs": float(rho[0]),
            "final_spearman_abs": float(rho[-1]),
            "spearman_degradation": float(rho[0] - rho[-1]),
            "spearman_slope_per_10pct": float(rho_slope),
            "spearman_normalized_auc": float(rho_auc),
            "initial_kendall_abs": float(tau[0]),
            "final_kendall_abs": float(tau[-1]),
            "kendall_degradation": float(tau[0] - tau[-1]),
            "kendall_slope_per_10pct": float(tau_slope),
            "kendall_normalized_auc": float(tau_auc),
        })

    robustness = pd.DataFrame(robustness_rows)
    robustness["combined_auc"] = (
        robustness["spearman_normalized_auc"] + robustness["kendall_normalized_auc"]
    ) / 2.0
    robustness = robustness.sort_values("combined_auc", ascending=False).reset_index(drop=True)
    robustness.to_csv(output_directory / "removal_robustness.csv", index=False)

    for metric, ylabel, filename in (
        ("spearman_abs", "Absolute Spearman ρ", "spearman_by_removal.png"),
        ("kendall_abs", "Absolute Kendall τ", "kendall_by_removal.png"),
    ):
        for window in sorted(summary["window_condition"].unique()):
            data = summary[summary["window_condition"] == window]
            if data.empty:
                continue
            fig, ax = plt.subplots(figsize=(9, 6))
            for model in models:
                g = data[data["model"] == model].sort_values("removal_percent")
                if g.empty:
                    continue
                ax.plot(g["removal_percent"], g[metric], marker="o", label=model)
            ax.set_xlabel("Removed hasAge relations (%)")
            ax.set_ylabel(ylabel)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{ylabel} by Removal Level — {window}")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(plots_dir / f"{window}_{filename}", dpi=dpi, bbox_inches="tight")
            plt.close(fig)

    if not window_effects.empty:
        for metric, title, filename in (
            ("delta_spearman_with_minus_without", "Window Effect on Spearman Monotonicity", "window_effect_spearman.png"),
            ("delta_kendall_with_minus_without", "Window Effect on Kendall Monotonicity", "window_effect_kendall.png"),
        ):
            fig, ax = plt.subplots(figsize=(9, 6))
            for model in models:
                g = window_effects[window_effects["model"] == model].sort_values("removal_percent")
                if g.empty:
                    continue
                ax.plot(g["removal_percent"], g[metric], marker="o", label=model)
            ax.axhline(0, linestyle="--", linewidth=1)
            ax.set_xlabel("Removed hasAge relations (%)")
            ax.set_ylabel("With windows − without windows")
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(plots_dir / filename, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

    report_path = output_directory / "monotonicity_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Cross-Model Numeric Monotonicity Analysis\n\n")
        f.write(
            "For each trained checkpoint, PCA was fit to the embedding vectors of the numeric literal nodes only (`v*`). "
            "The first principal component was then compared with the ground-truth numeric values using Spearman's rho and Kendall's tau.\n\n"
        )
        f.write(
            "Because PCA component sign is arbitrary, absolute correlations are used as the primary preservation metrics. "
            "PC1 is sign-oriented only for visualization.\n\n"
        )
        f.write("## Overall Model Ranking\n\n")
        f.write(rankings.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## Per-Run Results\n\n")
        cols = ["model", "window_condition", "removal_percent", "n_age_literals", "spearman_abs", "kendall_abs"]
        ordered = summary.sort_values(["window_condition", "removal_percent", "model"])
        f.write(ordered[cols].to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## Removal Robustness\n\n")
        f.write(
            robustness[[
                "model", "window_condition", "spearman_degradation", "kendall_degradation",
                "spearman_normalized_auc", "kendall_normalized_auc", "combined_auc"
            ]].to_markdown(index=False, floatfmt=".4f")
        )
        if not window_effects.empty:
            f.write("\n\n## Window Effects\n\n")
            f.write(window_effects.sort_values(["removal_percent", "model"]).to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## Interpretation\n\n")
        f.write(
            "- Values near 1 indicate strong preservation of numeric order.\n"
            "- Values near 0 indicate little monotonic relationship between PC1 and ground-truth numeric values.\n"
            "- Positive window-effect values mean the window-materialized KG preserved ordering better than its matched non-windowed KG.\n"
            "- Smaller degradation and larger normalized AUC indicate better robustness as direct `hasAge` supervision is removed.\n"
        )

    provenance = {
        "root": str(root.resolve()),
        "models": models,
        "manifest": str((shared_root / MANIFEST_FILE).resolve()),
        "shared_kg_hash_verification": calculated_hashes.to_dict(orient="records") if not calculated_hashes.empty else [],
        "primary_metric": "absolute Spearman rho and absolute Kendall tau",
        "pca_scope": "age literal embeddings only",
    }
    with (output_directory / "analysis_provenance.json").open("w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    print(f"Saved results to: {output_directory}")
    print(f"Overall best model: {rankings.iloc[0]['model']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare numeric monotonicity across embedding models using 1-D PCA of age literal embeddings plus Spearman and Kendall metrics."
        )
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-directory", default=str(DEFAULT_OUTPUT_DIRECTORY))
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--v-prefix", default="v")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    analyze_experiment(
        root=Path(args.root),
        output_directory=Path(args.output_directory),
        models=args.models,
        v_prefix=args.v_prefix,
        force=args.force,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
