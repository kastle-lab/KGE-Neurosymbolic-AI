## windower.py
Creates a synthetic knowledge graph of entities using assigned random values to generate graph structure of windows as a binary tree and less than structure (pairwise, sequential, etc)

## main.py
Creates embeddings from the synthetic knowledge graph output by windower.py and creates visualizations using PCA, UMAP and t-SNE

## learning_experiments/load_data.py
Helper to load data saved in the KG construction and dimensionality reduction steps

## learning_experiments/pc_learn.py
Uses the PCA data to fit a curve using a multilayer perceptron