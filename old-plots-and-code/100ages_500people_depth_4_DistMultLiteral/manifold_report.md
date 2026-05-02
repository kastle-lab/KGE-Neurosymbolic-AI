# Manifold Ordering Report

**Experiment Folder:** `100ages_500people_depth_4_DistMultLiteral`

---

## PCA

- Explained Variance Ratio (1D): `0.296953`

- Mean Distance to Line: `2.917376`


### Ordering Metrics

- monotonic_violations: `46`

- spearman_rho: `0.255002`

- spearman_p: `0.010454`

- kendall_tau: `0.174141`

- kendall_p: `0.010254`

- pairwise_inversions: `2044`

- pairwise_inversion_rate: `0.412929`


---

## Kernel PCA

- Kernel: `rbf`

- Gamma: `None`

- Mean Distance to Inverse Image: `2.920110`


### Ordering Metrics

- monotonic_violations: `53`

- spearman_rho: `0.254665`

- spearman_p: `0.010560`

- kendall_tau: `0.174949`

- kendall_p: `0.009907`

- pairwise_inversions: `2908`

- pairwise_inversion_rate: `0.587475`


---

## MLP Curve

- Mean Distance to Curve: `3.214989`


### Ordering Metrics

- monotonic_violations: `46`

- spearman_rho: `0.479952`

- spearman_p: `0.000000`

- kendall_tau: `0.342345`

- kendall_p: `0.000001`

- pairwise_inversions: `1445`

- pairwise_inversion_rate: `0.291919`


---

## Interpretation

- PCA linearity is low if explained variance is small.

- Compare PCA vs KPCA distances to detect nonlinearity.

- MLP curve tests if a parametric 1D manifold exists.

- Spearman ≈ global ordering quality.

- Kendall ≈ pairwise correctness.
