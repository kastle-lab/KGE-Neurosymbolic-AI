# Manifold Ordering Report

**Experiment Folder:** `100ages_500people_depth_4_TransD`

---

## PCA

- Explained Variance Ratio (1D): `0.300206`

- Mean Distance to Line: `0.814822`


### Ordering Metrics

- monotonic_violations: `45`

- spearman_rho: `0.983546`

- spearman_p: `0.000000`

- kendall_tau: `0.889293`

- kendall_p: `0.000000`

- pairwise_inversions: `274`

- pairwise_inversion_rate: `0.055354`


---

## Kernel PCA

- Kernel: `rbf`

- Gamma: `None`

- Mean Distance to Inverse Image: `0.852891`


### Ordering Metrics

- monotonic_violations: `54`

- spearman_rho: `0.983582`

- spearman_p: `0.000000`

- kendall_tau: `0.889697`

- kendall_p: `0.000000`

- pairwise_inversions: `4677`

- pairwise_inversion_rate: `0.944848`


---

## MLP Curve

- Mean Distance to Curve: `0.815464`


### Ordering Metrics

- monotonic_violations: `44`

- spearman_rho: `0.984338`

- spearman_p: `0.000000`

- kendall_tau: `0.891313`

- kendall_p: `0.000000`

- pairwise_inversions: `269`

- pairwise_inversion_rate: `0.054343`


---

## Interpretation

- PCA linearity is low if explained variance is small.

- Compare PCA vs KPCA distances to detect nonlinearity.

- MLP curve tests if a parametric 1D manifold exists.

- Spearman ≈ global ordering quality.

- Kendall ≈ pairwise correctness.
