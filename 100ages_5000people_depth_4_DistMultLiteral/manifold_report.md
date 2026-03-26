# Manifold Ordering Report

**Experiment Folder:** `100ages_5000people_depth_4_DistMultLiteral`

---

## PCA

- Explained Variance Ratio (1D): `0.326809`

- Mean Distance to Line: `4.023697`


### Ordering Metrics

- monotonic_violations: `51`

- spearman_rho: `0.134941`

- spearman_p: `0.180707`

- kendall_tau: `0.092525`

- kendall_p: `0.172575`

- pairwise_inversions: `2246`

- pairwise_inversion_rate: `0.453737`


---

## Kernel PCA

- Kernel: `rbf`

- Gamma: `None`

- Mean Distance to Inverse Image: `4.024824`


### Ordering Metrics

- monotonic_violations: `51`

- spearman_rho: `0.135506`

- spearman_p: `0.178875`

- kendall_tau: `0.093333`

- kendall_p: `0.168856`

- pairwise_inversions: `2244`

- pairwise_inversion_rate: `0.453333`


---

## MLP Curve

- Mean Distance to Curve: `4.639095`


### Ordering Metrics

- monotonic_violations: `42`

- spearman_rho: `0.241913`

- spearman_p: `0.015315`

- kendall_tau: `0.170984`

- kendall_p: `0.018061`

- pairwise_inversions: `1678`

- pairwise_inversion_rate: `0.338990`


---

## Interpretation

- PCA linearity is low if explained variance is small.

- Compare PCA vs KPCA distances to detect nonlinearity.

- MLP curve tests if a parametric 1D manifold exists.

- Spearman ≈ global ordering quality.

- Kendall ≈ pairwise correctness.
