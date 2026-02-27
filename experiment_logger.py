import os
from datetime import datetime

BASE_DIR = os.environ.get(
    "SLURM_SUBMIT_DIR",
    os.getcwd()
)

def write_experiment_markdown(
    experiment_folder,
    kg_params,
    embed_params,
    pca_params
):
    """
    Writes experiment_config.md inside:
    /<experiment_folder>/
    """

    exp_dir = os.path.join(BASE_DIR, experiment_folder)
    os.makedirs(exp_dir, exist_ok=True)

    md_path = os.path.join(exp_dir, "experiment_config.md")

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    md = f"""# Experiment Configuration

## Experiment
**Folder:** `{experiment_folder}`  
**Timestamp:** {now}

---

## KG Creation

| Parameter | Value |
|----------|------|
"""

    for k, v in kg_params.items():
        md += f"| {k} | {v} |\n"

    md += """
---

## Embedding

| Parameter | Value |
|----------|------|
"""

    for k, v in embed_params.items():
        md += f"| {k} | {v} |\n"

    md += """
---

## PCA

| Parameter | Value |
|----------|------|
"""

    for k, v in pca_params.items():
        md += f"| {k} | {v} |\n"

    md += """

---

## Notes

- Generated automatically by pipeline
- Ensures experiment reproducibility
"""

    with open(md_path, "w") as f:
        f.write(md)

    print(f"[LOG] Experiment config written → {md_path}")