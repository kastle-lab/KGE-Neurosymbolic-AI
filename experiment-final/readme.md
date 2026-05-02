# Investigating Latent Representations under Structured Sampling and Constraints

`pipeline.py` contains all the code needed to create the KGs, embeddings, and collects data for the distance from the query vector `h ⊙ R + r`.

`analyze_pipeline.py` performs calculates on the data to assemble a table of averages for the query method and the learned mapping method. Calls compare_methods to demonstrate performance between methods.