# Manifold Ordering Report

**Experiment Folder:** `100ages_500people_depth_4_DistMult`

---

## PCA

- Explained Variance Ratio (1D): `0.152428`

- Mean Distance to Line: `0.489046`


### Ordering Metrics

- monotonic_violations: `45`

- spearman_rho: `0.679508`

- spearman_p: `0.000000`

- kendall_tau: `0.416162`

- kendall_p: `0.000000`

- pairwise_inversions: `1445`

- pairwise_inversion_rate: `0.291919`


---

## Kernel PCA

- Kernel: `rbf`

- Gamma: `None`

- Mean Distance to Inverse Image: `0.520507`


### Ordering Metrics

- monotonic_violations: `54`

- spearman_rho: `0.679508`

- spearman_p: `0.000000`

- kendall_tau: `0.416162`

- kendall_p: `0.000000`

- pairwise_inversions: `3505`

- pairwise_inversion_rate: `0.708081`


---

## MLP Curve

- Mean Distance to Curve: `0.495162`


### Ordering Metrics

- monotonic_violations: `43`

- spearman_rho: `0.783113`

- spearman_p: `0.000000`

- kendall_tau: `0.556985`

- kendall_p: `0.000000`

- pairwise_inversions: `1040`

- pairwise_inversion_rate: `0.210101`


---

## Interpretation

- PCA linearity is low if explained variance is small.

- Compare PCA vs KPCA distances to detect nonlinearity.

- MLP curve tests if a parametric 1D manifold exists.

- Spearman ≈ global ordering quality.

- Kendall ≈ pairwise correctness.
