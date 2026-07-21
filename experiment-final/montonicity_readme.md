# Age-Node Embedding Monotonicity

This script measures how well learned age-node embeddings preserve the numerical ordering of age.

It compares two one-dimensional representations:

1. **PCA PC1**, used as a linear baseline.
2. **A regularized nonlinear MLP curve**, fitted in the two-dimensional PCA space.

The same MLP curve is used for both:

- the reported Spearman and Kendall monotonicity statistics;
- the per-run curve visualizations.

There is no separate visualization-only model.

---

## What the script analyzes

For every experiment run, the script:

1. Loads the trained PyKEEN entity embeddings.
2. Selects numeric age nodes such as `v18`, `v19`, and `v20`.
3. Retrieves each node's ground-truth numerical age.
4. Standardizes the original embedding dimensions.
5. projects the age-node embeddings into PCA components PC1 and PC2.
6. Fits a smooth nonlinear MLP curve through that PCA-2 representation.
7. Assigns each age node a one-dimensional position along the fitted curve.
8. Compares that curve position with ground-truth age using:
   - Spearman's rank correlation;
   - Kendall's tau-b.
9. Generates:
   - a CSV results file;
   - a Markdown report;
   - summary plots;
   - one PCA-2 curve visualization for every run.

---

## Important interpretation

The MLP curve is learned in the **two-dimensional PC1-PC2 projection**, not in the complete original embedding space.

The workflow is:

```text
Original embedding vectors
        ↓
Standardization
        ↓
PCA projection to PC1 and PC2
        ↓
Regularized MLP curve in PCA-2
        ↓
Nearest curve position for each node
        ↓
Spearman and Kendall comparison with true age
```

---

## Age-node detection

Age nodes are expected to begin with `v`.

Examples:

```text
v18
v25
v67
v18.5
v-2
```

The script first attempts to load numerical values from:

```text
kg.tsv.val
```

Expected format:

```text
v18    18
v19    19
v20    20
```

The separator must be a tab.

When a node is not present in `kg.tsv.val`, the script attempts to extract the number directly from its label.

For example:

```text
v42 → 42
```

At least four valid age nodes are required for each run.

---

## How PCA is used

The original age-node embeddings may have many dimensions.

The script first standardizes every embedding dimension:

```python
scaled = StandardScaler().fit_transform(vectors)
```

It then fits one two-component PCA model:

```python
pca = PCA(n_components=2)
node_xy = pca.fit_transform(scaled)
```

Each age node is represented by:

```text
PC1 coordinate
PC2 coordinate
```

These coordinates are used for both the MLP fit and the visualization.

### PCA baseline

The PCA monotonicity baseline uses the node's PC1 coordinate.

The script compares PC1 ordering with ground-truth age using:

- Spearman's rho;
- Kendall's tau-b.

PCA does not use age while calculating PC1.

---

## How the MLP curve works

The MLP learns a smooth curve in the PC1-PC2 plane.

### 1. Initial one-dimensional coordinate

Each node's initial curve coordinate is its normalized PC1 position:

```text
minimum PC1 → 0
maximum PC1 → 1
```

This coordinate is called `t`.

### 2. MLP input and output

The MLP learns:

```text
t → predicted PC1 and PC2 position
```

Mathematically:

```text
f(t) = (predicted PC1, predicted PC2)
```

The MLP does not receive age as an input or training target.

### 3. Curve sampling

The fitted MLP is evaluated at many evenly spaced values between 0 and 1:

```python
grid = np.linspace(0, 1, grid_size)
curve = mlp.predict(grid[:, None])
```

With the default grid size, the displayed curve contains 2,000 sampled locations.

### 4. Projecting nodes onto the curve

For each age node, the script finds the nearest sampled curve location in the standardized PCA-2 space.

That nearest location supplies the node's final MLP curve coordinate.

The MLP statistics therefore compare:

```text
ground-truth age
```

with:

```text
nearest position along the fitted MLP curve
```

---

## Avoiding overfitting

The earlier visualization model used a large network and extremely weak regularization, which could create sharp hooks and spiky curves.

This script uses restrained defaults:

```text
Hidden layers: 8
Activation: tanh
Solver: L-BFGS
L2 regularization alpha: 0.1
Maximum iterations: 5000
```

The small hidden layer limits curve complexity.

The `tanh` activation produces a smooth continuous mapping.

The L2 regularization discourages unnecessary bending and extreme weights.

These settings are intended to capture broad nonlinear structure without attempting to pass through every individual node.

---

## Use of ground-truth age

Ground-truth age is **not used to fit PCA or the MLP curve**.

It is used only after fitting for:

1. calculating Spearman and Kendall statistics;
2. determining which end of the one-dimensional coordinate should be treated as younger versus older;
3. coloring the plotted age nodes.

A one-dimensional coordinate has arbitrary direction.

For example, a valid learned coordinate could run:

```text
old → young
```

instead of:

```text
young → old
```

The script checks the correlation with age and reverses the orientation when necessary.

This does not change the curve's geometry or its ordering quality. It only gives the reported coordinate a consistent young-to-old direction.

---

## Curve and node colors

The plotted nodes are colored using their actual ground-truth ages:

```text
blue → lower age
red → higher age
```

The curve gradient represents normalized progress along the oriented curve:

```text
blue end → beginning of the young-to-old curve
red end → end of the young-to-old curve
```

The curve color is not an exact predicted age.

Therefore:

- node color represents actual age;
- curve color represents relative curve position.

---

## Monotonicity metrics

### Spearman's rho

Spearman's rho measures whether two variables have the same rank ordering.

Values range from `-1` to `1`.

After orientation:

```text
1.0   = perfect increasing age order
0.0   = no rank relationship
near 0 = weak preservation of age order
```

Because the coordinate direction is corrected after fitting, reported correlations should normally be nonnegative.

### Kendall's tau-b

Kendall's tau-b measures pairwise ordering agreement and handles tied values.

Values range from `-1` to `1`.

After orientation:

```text
1.0 = every comparable pair is ordered correctly
0.0 = no consistent pairwise ordering
```

Kendall's tau-b is often more conservative than Spearman's rho.

---

## Curve-distance diagnostics

The CSV also includes:

```text
mlp_mean_curve_distance
mlp_max_curve_distance
```

These are distances measured in the standardized PCA-2 fitting space.

### Mean curve distance

The average distance from an age node to its nearest point on the fitted MLP curve.

Lower values indicate that the curve is generally close to the nodes.

### Maximum curve distance

The largest node-to-curve distance.

This can reveal an age node that lies far from the main fitted trajectory.

These are geometric diagnostics, not age-prediction errors.

---

## PCA variance columns

The CSV includes:

```text
pca_pc1_variance
pca_2d_variance
```

### `pca_pc1_variance`

The proportion of standardized embedding variance explained by PC1.

### `pca_2d_variance`

The combined proportion explained by PC1 and PC2.

Because the MLP is fitted only in PCA-2, this value helps indicate how much of the original embedding geometry is represented in the analysis.

A low PCA-2 explained-variance value means that substantial embedding structure exists outside the two displayed components.

---

## Command-line options

```text
--basepath
```

One or more population directories to analyze.

Example:

```bash
--basepath 100 200 500
```

---

```text
--root
```

Directory searched when population folders are discovered automatically.

Default:

```text
.
```

---

```text
--output-directory
```

Directory where the report, CSV, and plots are written.

Default:

```text
final_results
```

---

```text
--hidden-layers
```

MLP hidden-layer sizes.

Default:

```text
8
```

Example with two hidden layers:

```bash
--hidden-layers 8 8
```

Larger networks can produce more flexible curves but increase the risk of overfitting and spikes.

---

```text
--mlp-alpha
```

L2 regularization strength.

Default:

```text
0.1
```

Higher values produce simpler and usually smoother curves.

Lower values allow the MLP to bend more closely around individual nodes.

---

```text
--max-iter
```

Maximum MLP optimizer iterations.

Default:

```text
5000
```

---

```text
--grid-size
```

Number of sampled locations along the fitted curve.

Default:

```text
2000
```

A larger value produces a more finely sampled line but does not make the learned model itself more complex.

---

```text
--seed
```

Random seed used by the MLP.

Default:

```text
42
```

---

```text
--dpi
```

Image resolution.

Default:

```text
300
```

---

## Recommended command

```bash
python age_monotonicity_pca2.py \
  --basepath 100 200 300 400 500 \
  --hidden-layers 8 \
  --mlp-alpha 0.1 \
  --max-iter 5000 \
  --grid-size 2000 \
  --seed 42 \
  --dpi 300
```

---

## Output structure

The default output structure is:

```text
final_results/
├── age-monotonicity.csv
├── age-monotonicity.md
├── age-monotonicity-plots/
│   ├── with_windows_monotonicity.png
│   ├── with_windows_gap.png
│   ├── without_windows_monotonicity.png
│   └── without_windows_gap.png
└── age-node-mlp-curves/
    ├── population_100_with_windows_removed_0p0pct.png
    ├── population_100_with_windows_removed_15p0pct.png
    ├── population_100_with_windows_removed_30p0pct.png
    └── ...
```

---

## Filename meanings

Example:

```text
population_500_with_windows_removed_15p0pct.png
```

means:

```text
population_500  → experiment containing 500 people
with_windows    → windowing condition
removed_15p0pct → 15.0% of hasAge relations removed
```

The letter `p` replaces the decimal point in the filename:

```text
15p0pct → 15.0%
12p5pct → 12.5%
0p0pct  → 0.0%
```

This is only filename formatting.

It does not indicate a different experimental condition.

---

## CSV columns

The CSV contains one row per run.

### Run information

```text
population_size
run
window_condition
removal_percent
n_age_nodes
```

### PCA monotonicity

```text
pca_rho
pca_rho_p
pca_tau
pca_tau_p
```

### MLP monotonicity

```text
mlp_rho
mlp_rho_p
mlp_tau
mlp_tau_p
```

### PCA explained variance

```text
pca_pc1_variance
pca_2d_variance
```

### Curve-distance diagnostics

```text
mlp_mean_curve_distance
mlp_max_curve_distance
```

---

## Summary plots

For each window condition, the script creates two summary figures.

### Monotonicity plot

Compares PCA and MLP rank correlations across removal levels.

The left panel shows:

```text
Spearman's rho
```

The right panel shows:

```text
Kendall's tau-b
```

Solid lines with circular markers represent PCA.

Dashed lines with square markers represent the MLP curve.

### Gap plot

Shows:

```text
MLP correlation − PCA correlation
```

Interpretation:

```text
above zero → MLP preserves age ordering better
below zero → PCA preserves age ordering better
zero       → equal measured monotonicity
```

---

## Per-run curve plots

Each run receives a visualization showing:

- age-node embeddings in PC1-PC2;
- the regularized MLP curve;
- ground-truth age coloring;
- the run's MLP Spearman rho;
- the run's MLP Kendall tau-b.

The plotted curve is the same curve used to calculate the MLP statistics.

A smooth curve indicates that the MLP found a broad nonlinear trajectory in PCA-2.

A nearly straight curve indicates that the nonlinear model found little useful curvature beyond PC1.

A sharply bending or spiky curve may indicate:

- insufficient regularization;
- an MLP that is too large;
- too few age nodes;
- irregular PCA-2 geometry;
- or genuine nonlinear structure.

Before interpreting spikes as meaningful geometry, increase `--mlp-alpha` or reduce the hidden-layer size.

---

## Suggested regularization adjustments

### Curve is too spiky

Try:

```bash
--hidden-layers 4 --mlp-alpha 0.5
```

or:

```bash
--hidden-layers 8 --mlp-alpha 1.0
```

### Curve is too flat

Try:

```bash
--hidden-layers 8 --mlp-alpha 0.03
```

### More flexibility

Try:

```bash
--hidden-layers 12 --mlp-alpha 0.1
```

Only change model settings consistently across all runs being compared.

---

## Methodological limitations

### PCA-2 information loss

The MLP uses only PC1 and PC2.

Embedding information contained in PC3 and later components is not included in the MLP analysis.

The `pca_2d_variance` column should be considered when interpreting the results.

### PC1-based parameterization

The MLP input is initialized directly from PC1.

The learned curve is therefore constrained to progress according to PC1.

It is best interpreted as a smooth nonlinear relationship between PC1 and PC2, approximately:

```text
PC2 = g(PC1)
```

It cannot properly represent a loop or a trajectory that repeatedly doubles back along PC1.

### Unsupervised curve fitting

Age values do not influence the curve's shape.

The curve measures whether age ordering emerges from the geometry learned by the embedding model.

It is not a supervised age-prediction model.

### Correlation is not numerical accuracy

Spearman and Kendall measure ordering.

They do not measure whether a node's curve position predicts its exact age in years.

A model can have high monotonicity while still being poorly calibrated numerically.

### Curve gradient

The curve's blue-to-red gradient represents normalized progress along the curve.

It should not be interpreted as an exact continuous age estimate.

---

## Reproducibility

For comparable experiments:

- use the same script version;
- use the same hidden-layer architecture;
- use the same regularization value;
- use the same random seed;
- use the same grid size;
- use the same preprocessing;
- avoid tuning MLP settings separately for individual runs.

The default seed is:

```text
42
```

Age-node rows are sorted by node label only for deterministic loading.

They are never sorted by numerical age before PCA or MLP fitting.

---

## Summary

This script asks:

> Do the age-node embeddings form a one-dimensional age-ordered trajectory in their first two principal components?

It answers this using:

- PC1 as a linear baseline;
- a smooth regularized MLP curve as a nonlinear alternative;
- Spearman's rho and Kendall's tau-b as ordering metrics;
- one shared curve for both numerical evaluation and visualization.