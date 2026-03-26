# KG creator
from kg_creator import create_kg
from kg_embedder import create_embeddings
from pca_runner import run_pca_for_experiment
from visualize import visualize_pca
from experiment_logger import write_experiment_markdown
from measure_ordering import measure

def run_full_experiment(settings, step=4):

    # KG creator
    from kg_creator import create_kg
    from kg_embedder import create_embeddings
    from pca_runner import run_pca_for_experiment
    from visualize import visualize_pca
    from experiment_logger import write_experiment_markdown

    n_people = settings["n_people"]
    n_vertices = settings["n_vertices"]

    window_depth = settings["window_depth"]
    decimal_precision = settings["decimal_precision"]
    high = settings["high"]
    low = settings["low"]

    model = settings["model"]
    seed = settings["seed"]
    epochs = settings["epochs"]
    dim = settings["dim"]
    custom_name_addon = settings["custom_name_addon"]

    experiment_folder = f"{n_vertices}ages_{n_people}people{custom_name_addon}"

    # using steps here to avoid recreating data 
    if step <= 1:
        # create KG
        print("Creating knowledge graph...")
        create_kg(n_vertices=n_vertices, window_depth=window_depth, experiment_folder=experiment_folder, decimal_precision=decimal_precision, high=high, low=low, n_people=n_people)
        
    if step <= 2:
        # KG embedder
        print("Creating embeddings...")
        create_embeddings(experiment_folder=experiment_folder, model=model, seed=seed, epochs=epochs, dim=dim)

    if step <= 3:
        # PCA on embeddings 
        print("Running PCA...")
        coords, pca_model = run_pca_for_experiment(experiment_folder=experiment_folder, n_components=3, seed=seed)
     
    if step <= 4:
        # visualizations on PCA data
        print("Creating visualizations...")
        visualize_pca(experiment_folder)
        
    if step <= 5:
        print("Measuring error...")
        measure(experiment_folder)
     
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
    
base_settings_depth4 = {
    "window_depth": 4,
    "decimal_precision": 0,
    "high": 100,
    "low": 0,
    "seed": 42,
    "epochs": 100,
    "dim": 300
}

models = ["RGCN", "DistMult", "MuRE", "TransE", "TransD", "DistMultLiteral"]

experiments = [
    {"n_vertices": 100,  "n_people": 500},
    {"n_vertices": 100,  "n_people": 5000}
]

for model in models:
    for exp in experiments:

        settings = base_settings_depth4.copy()
        settings.update(exp)

        settings["model"] = model
        settings["custom_name_addon"] = f"_depth_4_{model}"

        print(f"\nRunning: {settings['n_vertices']} ages, {settings['n_people']} people | depth 4 | {model}")

        run_full_experiment(settings)