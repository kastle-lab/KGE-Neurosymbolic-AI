# Manifold Ordering Report

**Experiment Folder:** `100ages_500people_depth_4_RGCN`

---

## PCA

- Explained Variance Ratio (1D): `0.271568`

- Mean Distance to Line: `5.793065`


### Ordering Metrics

- monotonic_violations: `47`

- spearman_rho: `0.080108`

- spearman_p: `0.428196`

- kendall_tau: `0.053333`

- kendall_p: `0.431737`

- pairwise_inversions: `2343`

- pairwise_inversion_rate: `0.473333`


---

## Kernel PCA

- Kernel: `rbf`

- Gamma: `None`

- Mean Distance to Inverse Image: `5.799906`


### Ordering Metrics

- monotonic_violations: `48`

- spearman_rho: `0.082112`

- spearman_p: `0.416693`

- kendall_tau: `0.054141`

- kendall_p: `0.424792`

- pairwise_inversions: `2341`

- pairwise_inversion_rate: `0.472929`


---

## MLP Curve

- Mean Distance to Curve: `6.047400`


### Ordering Metrics

- monotonic_violations: `46`

- spearman_rho: `0.895906`

- spearman_p: `0.000000`

- kendall_tau: `0.708034`

- kendall_p: `0.000000`

- pairwise_inversions: `711`

- pairwise_inversion_rate: `0.143636`


---

## Interpretation

- PCA linearity is low if explained variance is small.

- Compare PCA vs KPCA distances to detect nonlinearity.

- MLP curve tests if a parametric 1D manifold exists.

- Spearman ≈ global ordering quality.

- Kendall ≈ pairwise correctness.
