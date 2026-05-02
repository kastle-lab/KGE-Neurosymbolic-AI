import torch
import pandas as pd

from multiprocessing import Pool

from kg_creator import create_kg
from kg_modifier import modify_kg
from embeddings_helper import KastleEmbeddings
from evaluate_query_points import evaluate_query_point_run


# ##############################
# experiment settings
# ##############################

n_people = 100
n_vertices = 100

basepath = f"./{n_people}people"

embedding_dimensions = 300
embedding_epochs = 100
embedding_model = "mure"
experiment_basename = f"{n_people}people"

window_depth = 4
decimal_precision = 0
high = 100
low = 1

removal_range = range(2, 9)


# ##############################
# create KGs
# ##############################

def create_KGs():
    create_kg(
        n_vertices=n_vertices,
        n_people=n_people,
        window_depth=window_depth,
        decimal_precision=decimal_precision,
        high=high,
        low=low,
        experiment_folder=basepath,
    )

    for n in removal_range:
        modify_kg(
            path=basepath,
            n=n,
            relation_string="hasAge",
        )

    kg_files = [("original", f"{basepath}/kg.tsv")]

    for n in removal_range:
        kg_files.append(
            (f"every_{n}_removed", f"{basepath}/every_{n}_removed_kg.tsv")
        )

    return kg_files

# ##############################
# create embeddings
# ##############################

def run_embedding_job(job):
    torch.set_num_threads(1)

    label, tsv_path = job
    run_folder = f"{basepath}/runs/{label}"

    experiment = KastleEmbeddings(
        seed=42,
        experiment_folder=run_folder,
        tsv_path=tsv_path,
        emb_model=embedding_model,
        emb_epochs=embedding_epochs,
        emb_dimensions=embedding_dimensions,
        pca=False,
        tsne=False,
        umap=False,
        dr_components=2,
        plot_name=label,
    )

    experiment.create_pykeen_embeddings()

# ##############################
# run measurements
# ##############################

def run_query_point_evaluations():
    run_labels = ["original"] + [
        f"every_{n}_removed"
        for n in removal_range
    ]

    all_dfs = []

    for run_label in run_labels:
        print(f"\nEvaluating query points for {run_label}")

        df = evaluate_query_point_run(
            original_run_path=f"{basepath}/runs/original",
            evaluated_run_path=f"{basepath}/runs/{run_label}",
            output_csv_path=f"{basepath}/runs/{run_label}/query_point_distances.csv",
        )

        df["experiment"] = experiment_basename
        df["run"] = run_label

        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)

    full_output_path = f"{basepath}/query_point_distances_all_runs.csv"
    full_df.to_csv(full_output_path, index=False)

    print(f"\nSaved all query point evaluations to: {full_output_path}")

    return full_df


if __name__ == "__main__":

    # Create KGs.
     kg_files = create_KGs()

    # Generate embeddings.
     with Pool(processes=8) as pool:
         pool.map(run_embedding_job, kg_files)

    # Run query-point evaluation on embeddings.
    full_df = run_query_point_evaluations()