# Manifold Ordering Report

**Experiment Folder:** `100ages_5000people_depth_4_TransD`

---

## PCA

- Explained Variance Ratio (1D): `0.058079`

- Mean Distance to Line: `0.947781`


### Ordering Metrics

- monotonic_violations: `49`

- spearman_rho: `0.927117`

- spearman_p: `0.000000`

- kendall_tau: `0.761616`

- kendall_p: `0.000000`

- pairwise_inversions: `590`

- pairwise_inversion_rate: `0.119192`


---

## Kernel PCA

- Kernel: `rbf`

- Gamma: `None`

- Mean Distance to Inverse Image: `0.966954`


### Ordering Metrics

- monotonic_violations: `50`

- spearman_rho: `0.927117`

- spearman_p: `0.000000`

- kendall_tau: `0.761616`

- kendall_p: `0.000000`

- pairwise_inversions: `4360`

- pairwise_inversion_rate: `0.880808`


---

## MLP Curve

- Mean Distance to Curve: `0.949028`


### Ordering Metrics

- monotonic_violations: `47`

- spearman_rho: `0.951181`

- spearman_p: `0.000000`

- kendall_tau: `0.806609`

- kendall_p: `0.000000`

- pairwise_inversions: `470`

- pairwise_inversion_rate: `0.094949`


---

## Interpretation

- PCA linearity is low if explained variance is small.

- Compare PCA vs KPCA distances to detect nonlinearity.

- MLP curve tests if a parametric 1D manifold exists.

- Spearman ≈ global ordering quality.

- Kendall ≈ pairwise correctness.
