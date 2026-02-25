# KG creator
from kg_creator import create_kg
from kg_embedder import create_embeddings
from pca_runner import run_pca_for_experiment
from visualize import visualize_pca

n_vertices = 100
window_depth = 4
experiment_folder = "100v_200p"
decimal_precision=0
high=100
low=0

model = "mure"
seed = 42
epochs=100
dim = 300

create_kg(n_vertices=n_vertices, window_depth=window_depth, experiment_folder=experiment_folder, decimal_precision=decimal_precision, high=high, low=low)
    
# KG embedder
create_embeddings(experiment_folder=experiment_folder, model=model, seed=seed, epochs=epochs, dim=dim)

# PCA on embeddings 
coords, pca_model = run_pca_for_experiment(
    experiment_folder=experiment_folder,
    n_components=3,
    seed=42
)
# visualizations on PCA data
visualize_pca(experiment_folder)
