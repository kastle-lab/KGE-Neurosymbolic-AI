# Manifold Ordering Report

**Experiment Folder:** `100ages_5000people_depth_4_DistMult`

---

## PCA

- Explained Variance Ratio (1D): `0.027511`

- Mean Distance to Line: `0.771391`


### Ordering Metrics

- monotonic_violations: `42`

- spearman_rho: `0.011341`

- spearman_p: `0.910832`

- kendall_tau: `0.004848`

- kendall_p: `0.943020`

- pairwise_inversions: `2487`

- pairwise_inversion_rate: `0.502424`


---

## Kernel PCA

- Kernel: `rbf`

- Gamma: `None`

- Mean Distance to Inverse Image: `0.780916`


### Ordering Metrics

- monotonic_violations: `42`

- spearman_rho: `0.012205`

- spearman_p: `0.904070`

- kendall_tau: `0.004444`

- kendall_p: `0.947761`

- pairwise_inversions: `2486`

- pairwise_inversion_rate: `0.502222`


---

## MLP Curve

- Mean Distance to Curve: `0.775843`


### Ordering Metrics

- monotonic_violations: `45`

- spearman_rho: `0.870248`

- spearman_p: `0.000000`

- kendall_tau: `0.673852`

- kendall_p: `0.000000`

- pairwise_inversions: `790`

- pairwise_inversion_rate: `0.159596`


---

## Interpretation

- PCA linearity is low if explained variance is small.

- Compare PCA vs KPCA distances to detect nonlinearity.

- MLP curve tests if a parametric 1D manifold exists.

- Spearman ≈ global ordering quality.

- Kendall ≈ pairwise correctness.
