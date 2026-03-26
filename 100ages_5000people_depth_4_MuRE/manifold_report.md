# Manifold Ordering Report

**Experiment Folder:** `100ages_5000people_depth_4_MuRE`

---

## PCA

- Explained Variance Ratio (1D): `0.159457`

- Mean Distance to Line: `1.507925`


### Ordering Metrics

- monotonic_violations: `56`

- spearman_rho: `0.971653`

- spearman_p: `0.000000`

- kendall_tau: `0.856566`

- kendall_p: `0.000000`

- pairwise_inversions: `4595`

- pairwise_inversion_rate: `0.928283`


---

## Kernel PCA

- Kernel: `rbf`

- Gamma: `None`

- Mean Distance to Inverse Image: `1.525370`


### Ordering Metrics

- monotonic_violations: `43`

- spearman_rho: `0.972157`

- spearman_p: `0.000000`

- kendall_tau: `0.858586`

- kendall_p: `0.000000`

- pairwise_inversions: `350`

- pairwise_inversion_rate: `0.070707`


---

## MLP Curve

- Mean Distance to Curve: `1.510429`


### Ordering Metrics

- monotonic_violations: `44`

- spearman_rho: `0.979047`

- spearman_p: `0.000000`

- kendall_tau: `0.874432`

- kendall_p: `0.000000`

- pairwise_inversions: `306`

- pairwise_inversion_rate: `0.061818`


---

## Interpretation

- PCA linearity is low if explained variance is small.

- Compare PCA vs KPCA distances to detect nonlinearity.

- MLP curve tests if a parametric 1D manifold exists.

- Spearman ≈ global ordering quality.

- Kendall ≈ pairwise correctness.
