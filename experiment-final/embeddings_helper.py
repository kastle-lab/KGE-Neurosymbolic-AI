# main code or something 

# I think what I'll do is make a class with some functions like embed, plot, etc and 
# try to be as general and useful in the future as possible

# standard imports
import os, re, joblib, json, hashlib, random, torch 
import pandas as pd
import numpy as np

# plotting imports
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")  # always use headless backend

# pykeen imports
from pykeen.pipeline import pipeline as pykeen_pipeline
from pykeen.triples import TriplesFactory
from pykeen.utils import set_random_seed

# dimensionality reduction imports
import umap.umap_ as umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class KastleEmbeddings:
    
    def __init__(self, seed, experiment_folder, **kwargs):
        random.seed(seed)
        torch.manual_seed(seed)
         
        # misc parameters
        self.seed = seed
        self.experiment_folder = experiment_folder
        self.tsv_path = kwargs.get("tsv_path")
        
        # load tsv
        self.triples_dataframe = pd.read_csv(kwargs.get("tsv_path"), sep="\t", header=None, names=["head", "relation", "tail"])
        
        # remove header row if present 
        if len(self.triples_dataframe) > 0:
            first_row = list(self.triples_dataframe.iloc[0].astype(str))
            if first_row == ["?sub", "?pred", "?val"] or first_row == ["sub", "pred", "val"]:
                self.triples_dataframe = self.triples_dataframe.iloc[1:].reset_index(drop=True)
                
        # print friendly helpful message so guests feel welcomed
        print(f"Triples converted to dataframe: {self.triples_dataframe}")
        
        # embedding parameters
        self.emb_model = kwargs.get("emb_model")
        self.emb_epochs = kwargs.get("emb_epochs")
        self.emb_dimensions = kwargs.get("emb_dimensions")
        
        # dimensionality reduction parameters
        self.tsne = kwargs.get("tsne")
        self.umap = kwargs.get("umap")
        self.pca = kwargs.get("pca")
        self.dr_components = kwargs.get("dr_components")
        
        # plot/visualization parameters
        self.plot_name = kwargs.get("plot_name")
        
        # parameters created by class functions but that are useful to have access to
        self.pykeen_result_full = kwargs.get("pykeen_result")
        self.entity_embeddings = kwargs.get("entity_embeddings")
        self.relation_embeddings = kwargs.get("relation_embeddings")
        self.dr_results = kwargs.get("dr_results")

        self.entity_labels = kwargs.get("entity_labels")
        self.relation_labels = kwargs.get("relation_labels")
        self.unique_entities = kwargs.get("unique_entities")
        self.entity_to_color = kwargs.get("entity_to_color")

        os.makedirs(self.experiment_folder, exist_ok=True)
        
    def create_pykeen_embeddings(self):
        
        # format dataframe for pykeen
        triples = self.triples_dataframe[["head", "relation", "tail"]].astype(str).values
        
        # this might just be for debug purposes but let's print again
        print(f"\nTriples: {triples}")
        
        # create triples factory object
        tf = TriplesFactory.from_labeled_triples(triples)

        # split into training, testing, validation 
        training, testing, validation = tf.split([0.8, 0.1, 0.1])
        
        # just in case these change for some reason idk redundancy is good sometimes no?
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # create embeddings using pykeen pipeline function TODO: look into parameter optimizer
        result = pykeen_pipeline(
            training=training,
            testing=testing,
            validation=validation,
            model=self.emb_model,
            random_seed=self.seed,
            model_kwargs={"embedding_dim": self.emb_dimensions},
            training_kwargs={"num_epochs": self.emb_epochs},
        )

        print(f"Pykeen embeddings done: {result}")
        
        # set pykeen_result parameter
        self.pykeen_result_full = result
        self.entity_embeddings = result.model.entity_representations[0](indices=None).detach().cpu().numpy()
        self.relation_embeddings = result.model.relation_representations[0](indices=None).detach().cpu().numpy()

        # recover relation labels in embedding-row order
        relation_to_id = training.relation_to_id
        self.relation_labels = [None] * len(relation_to_id)

        for label, idx in relation_to_id.items():
            self.relation_labels[idx] = str(label).strip()
                
        # recover entity labels in embedding-row order
        entity_to_id = training.entity_to_id
        self.entity_labels = [None] * len(entity_to_id)

        for label, idx in entity_to_id.items():
            self.entity_labels[idx] = str(label).strip()

        self.unique_entities = sorted(set(self.entity_labels))

        cmap = plt.cm.get_cmap("tab20", len(self.unique_entities))
        self.entity_to_color = {
            entity: cmap(i)
            for i, entity in enumerate(self.unique_entities)
        }
        
        self.save_experiment_output()
        return result

    def reduce_dimensions(self):
    
        reduced_pca = None
        reduced_tsne = None
        reduced_umap = None
        
        if self.pca:
            print("Calculating PCA")
            reduced_pca = PCA(
                n_components=self.dr_components,
                random_state=self.seed

            ).fit_transform(self.entity_embeddings)
        
        if self.tsne:
            print("Calculating t-SNE")
            reduced_tsne = TSNE(
                n_components=self.dr_components,
                random_state=self.seed,
                perplexity=30,
                max_iter=1000,
            ).fit_transform(self.entity_embeddings) 

        if self.umap:
            print("Calculating UMAP")
            reduced_umap = umap.UMAP(
                n_components=self.dr_components,
                random_state=self.seed,
            ).fit_transform(self.entity_embeddings)
        
        self.dr_results = { "pca": reduced_pca, "tsne": reduced_tsne, "umap": reduced_umap }

        self.save_experiment_output()      
          
        print(f"Dimensionality reduction complete: {self.dr_results}")
        
        return self.dr_results   

    def create_plots(self):

        def _plot_helper(coords, suffix):
            coords = np.asarray(coords)

            if coords.ndim == 1:
                coords = coords.reshape(-1, 1)

            if coords.ndim != 2:
                raise ValueError(f"Expected 1D or 2D coords array, got shape {coords.shape}")

            n_dims = coords.shape[1]

            if n_dims < 1:
                raise ValueError("coords must have at least 1 component")

            if n_dims == 1:
                x = coords[:, 0]

                fig, ax = plt.subplots()

                for entity in self.unique_entities:
                    indices = [i for i, label in enumerate(self.entity_labels) if label == entity]

                    ax.scatter(
                        x[indices],
                        np.zeros(len(indices)),
                        color=self.entity_to_color[entity],
                        alpha=0.7
                    )

                ax.set_xlabel("Component 1")
                ax.set_yticks([])
                ax.set_title(f"{self.plot_name}_{suffix}_1")
                
                filename = f"{self.plot_name}_{suffix}_1.png"

            elif n_dims == 2:
                x, y = coords[:, 0], coords[:, 1]

                fig, ax = plt.subplots()

                for entity in self.unique_entities:
                    indices = [i for i, label in enumerate(self.entity_labels) if label == entity]

                    ax.scatter(
                        x[indices],
                        y[indices],
                        color=self.entity_to_color[entity],
                        alpha=0.7
                    )

                ax.set_xlabel("Component 1")
                ax.set_ylabel("Component 2")
                ax.set_title(f"{self.plot_name}_{suffix}_2")
                
                filename = f"{self.plot_name}_{suffix}_2.png"

            else:
                x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")

                for entity in self.unique_entities:
                    indices = [i for i, label in enumerate(self.entity_labels) if label == entity]

                    ax.scatter(
                        x[indices],
                        y[indices],
                        z[indices],
                        color=self.entity_to_color[entity],
                        alpha=0.7
                    )

                ax.set_xlabel("Component 1")
                ax.set_ylabel("Component 2")
                ax.set_zlabel("Component 3")
                ax.set_title(f"{self.plot_name}_{suffix}_3")
                
                filename = f"{self.plot_name}_{suffix}_3.png"
            os.makedirs(self.experiment_folder, exist_ok=True)
            fig.savefig(os.path.join(self.experiment_folder, filename), bbox_inches="tight", dpi=300)
            plt.close(fig)

        if self.pca:
            print("Generating PCA plot")
            coords = np.asarray(self.dr_results["pca"])
            _plot_helper(coords, suffix="PCA")

        if self.tsne:
            print("Generating t-SNE plot")
            coords = np.asarray(self.dr_results["tsne"])
            _plot_helper(coords, suffix="t-SNE")

        if self.umap:
            print("Generating UMAP plot")
            coords = np.asarray(self.dr_results["umap"])
            _plot_helper(coords, suffix="UMAP")
            
    # misc helpers
    def save_experiment_output(self):
        self.pykeen_result_full.save_to_directory(self.experiment_folder)
        