# Visualizations 

Most of these are plots we made early on in in this work and do not appear in the final paper; however, they are interesting and useful to understanding the motivations behind certain choices made in our experimental design. 

--- 

## PC1 vs Initialization Value

![PC1-vs-init-val](./visualizations/PCA1-vs-Init-val.svg)

This figure shows the principle component of the embeddings of the "age" nodes plotted against the value the age node was instantiated used for materializing the KG but then subsequently discarded. The image shows a clear relationship between the topological arrangement of the embedding vectors with the values intially chosen, demonstrating that the embedding process has preserved a notion of montonicity.

## Monotonicity Preservation Across Models 

![montonicity-preservation-modeles](./model_comparison_200people_100nodes_monotonicity/plots/with_windows_kendall_by_removal.png)

![montonicity-preservation-modeles](./model_comparison_200people_100nodes_monotonicity/plots/without_windows_kendall_by_removal.png)

These figures show how monotonicity is preserved using Kendall's Tau as a metric both with and without our windowing procedure across several popular KGE models.

![montonicity-preservation-modeles](./model_comparison_200people_100nodes_monotonicity/plots/pc1/distmult_with_windows_original.png)
![montonicity-preservation-modeles](./model_comparison_200people_100nodes_monotonicity/plots/pc1/distmult_without_windows_original.png)

MuRE performs better with consistency. Interestingly, DistMult's performance is significantly affected by windowing.

![montonicity-preservation-modeles](./model_comparison_200people_100nodes_monotonicity/plots/pc1/transr_with_windows_removed_75pct.png)
![montonicity-preservation-modeles](./model_comparison_200people_100nodes_monotonicity/plots/pc1/transr_without_windows_removed_75pct.png)

TransR is particularly affected by windowing at 75% removal.

---

## MLP Curve fit to Principle Components 1 and 2

![PC1-PC2-MLP-curve](./age-node-mlp-curves/population_500_with_windows_removed_0p0pct.png)

This figure shows principle components 1 and 2 of the age vectors with an MLP curve learned in the two-dimensional PCA. This step is mainly to visually demonstrate that a notion of monotonicity is preserved in the embedding space and is recoverable using principle component analysis and a multi-layer perceptron.

The metrics (Spearman's rho and Kendall's tau-b) are comparisons of ordering preserved in PC1 and MLP+2D-PCA both between the ordering in the ground truth. 


--- 

## Relation Topology in 2D-PCA

![Relation-Arrows](./visualizations/population_500_with_windows_removed_0pct_pca.png)

This figure shows the person and age nodes along with the relational vectors (R and r) "applied" (R $\odot$ h + r) to the person node with a dotted line to that person's respective age. 

This visualizes the MuRE scoring function in 2D-PCA, demonstrating how relations are preserved in the embedding space using our synthetic knowledge graph structure.
