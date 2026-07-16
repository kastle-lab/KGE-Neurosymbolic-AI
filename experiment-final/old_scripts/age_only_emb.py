from embeddings_helper import KastleEmbeddings

# ##############################
# experiment settings
# ##############################

basepath = "./100ages_only"

embedding_dimensions = 300
embedding_epochs = 100
embedding_model = "mure"

label = "emb"

tsv_path = f"{basepath}/kg.tsv"
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