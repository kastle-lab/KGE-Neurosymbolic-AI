# Manifold Ordering Report

**Experiment Folder:** `100ages_5000people_depth_4_TransE`

---

## PCA

- Explained Variance Ratio (1D): `0.040626`

- Mean Distance to Line: `0.825518`


### Ordering Metrics

- monotonic_violations: `50`

- spearman_rho: `0.949079`

- spearman_p: `0.000000`

- kendall_tau: `0.798384`

- kendall_p: `0.000000`

- pairwise_inversions: `4451`

- pairwise_inversion_rate: `0.899192`


---

## Kernel PCA

- Kernel: `rbf`

- Gamma: `None`

- Mean Distance to Inverse Image: `0.839246`


### Ordering Metrics

- monotonic_violations: `49`

- spearman_rho: `0.949079`

- spearman_p: `0.000000`

- kendall_tau: `0.798384`

- kendall_p: `0.000000`

- pairwise_inversions: `499`

- pairwise_inversion_rate: `0.100808`


---

## MLP Curve

- Mean Distance to Curve: `0.826081`


### Ordering Metrics

- monotonic_violations: `48`

- spearman_rho: `0.971285`

- spearman_p: `0.000000`

- kendall_tau: `0.849515`

- kendall_p: `0.000000`

- pairwise_inversions: `369`

- pairwise_inversion_rate: `0.074545`


---

## Interpretation

- PCA linearity is low if explained variance is small.

- Compare PCA vs KPCA distances to detect nonlinearity.

- MLP curve tests if a parametric 1D manifold exists.

- Spearman ≈ global ordering quality.

- Kendall ≈ pairwise correctness.
