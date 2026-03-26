# Manifold Ordering Report

**Experiment Folder:** `100ages_500people_depth_4_TransE`

---

## PCA

- Explained Variance Ratio (1D): `0.067967`

- Mean Distance to Line: `0.895224`


### Ordering Metrics

- monotonic_violations: `42`

- spearman_rho: `0.975638`

- spearman_p: `0.000000`

- kendall_tau: `0.869899`

- kendall_p: `0.000000`

- pairwise_inversions: `322`

- pairwise_inversion_rate: `0.065051`


---

## Kernel PCA

- Kernel: `rbf`

- Gamma: `None`

- Mean Distance to Inverse Image: `0.915847`


### Ordering Metrics

- monotonic_violations: `42`

- spearman_rho: `0.975638`

- spearman_p: `0.000000`

- kendall_tau: `0.869899`

- kendall_p: `0.000000`

- pairwise_inversions: `322`

- pairwise_inversion_rate: `0.065051`


---

## MLP Curve

- Mean Distance to Curve: `0.895592`


### Ordering Metrics

- monotonic_violations: `40`

- spearman_rho: `0.984620`

- spearman_p: `0.000000`

- kendall_tau: `0.895130`

- kendall_p: `0.000000`

- pairwise_inversions: `259`

- pairwise_inversion_rate: `0.052323`


---

## Interpretation

- PCA linearity is low if explained variance is small.

- Compare PCA vs KPCA distances to detect nonlinearity.

- MLP curve tests if a parametric 1D manifold exists.

- Spearman ≈ global ordering quality.

- Kendall ≈ pairwise correctness.
