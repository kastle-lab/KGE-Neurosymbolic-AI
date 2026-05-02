# Manifold Ordering Report

**Experiment Folder:** `100ages_500people_depth_4_MuRE`

---

## PCA

- Explained Variance Ratio (1D): `0.461072`

- Mean Distance to Line: `0.851809`


### Ordering Metrics

- monotonic_violations: `52`

- spearman_rho: `0.988035`

- spearman_p: `0.000000`

- kendall_tau: `0.910303`

- kendall_p: `0.000000`

- pairwise_inversions: `4728`

- pairwise_inversion_rate: `0.955152`


---

## Kernel PCA

- Kernel: `rbf`

- Gamma: `None`

- Mean Distance to Inverse Image: `0.878453`


### Ordering Metrics

- monotonic_violations: `52`

- spearman_rho: `0.988083`

- spearman_p: `0.000000`

- kendall_tau: `0.911111`

- kendall_p: `0.000000`

- pairwise_inversions: `4730`

- pairwise_inversion_rate: `0.955556`


---

## MLP Curve

- Mean Distance to Curve: `0.852989`


### Ordering Metrics

- monotonic_violations: `47`

- spearman_rho: `0.989313`

- spearman_p: `0.000000`

- kendall_tau: `0.915336`

- kendall_p: `0.000000`

- pairwise_inversions: `209`

- pairwise_inversion_rate: `0.042222`


---

## Interpretation

- PCA linearity is low if explained variance is small.

- Compare PCA vs KPCA distances to detect nonlinearity.

- MLP curve tests if a parametric 1D manifold exists.

- Spearman ≈ global ordering quality.

- Kendall ≈ pairwise correctness.
