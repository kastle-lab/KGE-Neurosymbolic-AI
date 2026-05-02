# Manifold Ordering Report

**Experiment Folder:** `100ages_5000people_depth_4_RGCN`

---

## PCA

- Explained Variance Ratio (1D): `0.227901`

- Mean Distance to Line: `14.969914`


### Ordering Metrics

- monotonic_violations: `50`

- spearman_rho: `0.213993`

- spearman_p: `0.032528`

- kendall_tau: `0.137374`

- kendall_p: `0.042855`

- pairwise_inversions: `2815`

- pairwise_inversion_rate: `0.568687`


---

## Kernel PCA

- Kernel: `rbf`

- Gamma: `None`

- Mean Distance to Inverse Image: `15.004853`


### Ordering Metrics

- monotonic_violations: `48`

- spearman_rho: `0.219562`

- spearman_p: `0.028173`

- kendall_tau: `0.141414`

- kendall_p: `0.037098`

- pairwise_inversions: `2125`

- pairwise_inversion_rate: `0.429293`


---

## MLP Curve

- Mean Distance to Curve: `15.798580`


### Ordering Metrics

- monotonic_violations: `49`

- spearman_rho: `0.932747`

- spearman_p: `0.000000`

- kendall_tau: `0.770549`

- kendall_p: `0.000000`

- pairwise_inversions: `555`

- pairwise_inversion_rate: `0.112121`


---

## Interpretation

- PCA linearity is low if explained variance is small.

- Compare PCA vs KPCA distances to detect nonlinearity.

- MLP curve tests if a parametric 1D manifold exists.

- Spearman ≈ global ordering quality.

- Kendall ≈ pairwise correctness.
