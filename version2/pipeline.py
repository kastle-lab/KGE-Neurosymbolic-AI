# KG creator
from kg_creator import create_kg
from kg_embedder import create_embeddings
from pca_runner import run_pca_for_experiment
from visualize import visualize_pca
from experiment_logger import write_experiment_markdown

n_people = 100
n_vertices = 1000

window_depth = 4
experiment_folder = f"{n_vertices}ages_{n_people}people"
decimal_precision = 0
high = 100
low = 0

model = "mure"
seed = 42
epochs = 100
dim = 300

# create KG
print("Creating knowledge graph...")
create_kg(n_vertices=n_vertices, window_depth=window_depth, experiment_folder=experiment_folder, decimal_precision=decimal_precision, high=high, low=low, n_people=n_people)
    
# KG embedder
print("Creating embeddings...")
create_embeddings(experiment_folder=experiment_folder, model=model, seed=seed, epochs=epochs, dim=dim)

# PCA on embeddings 
print("Running PCA...")
coords, pca_model = run_pca_for_experiment(experiment_folder=experiment_folder, n_components=3, seed=42)

# visualizations on PCA data
print("Creating visualizations...")
visualize_pca(experiment_folder)

write_experiment_markdown(
    experiment_folder=experiment_folder,
    kg_params={
        "n_vertices": n_vertices,
        "window_depth": window_depth,
        "decimal_precision": decimal_precision,
        "high": high,
        "low": low,
        "n_people": n_people
    },
    embed_params={
        "model": model,
        "seed": seed,
        "epochs": epochs,
        "dim": dim
    },
    pca_params={
        "n_components": 3,
        "seed": seed
    }
)