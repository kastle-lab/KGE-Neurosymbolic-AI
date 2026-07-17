# MuRE Numeric-Literal Preservation Experiment

## Repository Structure

The readme describes the pipeline inside `experiment-final` and its imports. The files inside `old-plots-and-code` though important to motivating the research question through visualizations (PCA) of various graph embeddings, are not directly pertinent to the experiment and its results.

`pipeline.py` and `pipeline.ipynb` are essentially the same though `pipeline.py` runs the embedding steps in parallel. The Jupyter notebook is easier to read.

## Experiment Overview

Knowledge graph embeddings learn vector representations from the structure of a graph. Most work on knowledge graph embeddings focuses on predicting missing entities or relationships. This project instead explores a related question:

> How does the topology of a knowledge graph affect the numerical information preserved in its learned embeddings?

The experiments examine whether age information remains recoverable after the graph is structurally modified and increasing numbers of explicit `hasAge` relationships are removed.

Window relationships are the primary structural modification examined in the current study. A graph containing this additional structure is compared with a corresponding graph in which the window-related triples have been removed.

The project is exploratory rather than intended as a broad benchmark. The current experiments use:

- The **MuRE** embedding model
- Synthetic graphs containing **100, 200, and 500 people**
- Numeric age values connected through `hasAge`
- Windowed and windowless graph structures
- `hasAge` removal levels of **0%, 15%, 30%, 45%, 60%, and 75%**
- Query-point and learned-regression recovery methods

The broader goal is to empirically investigate what happens to the information represented in embedding space when a controlled mathematical change is made to the graph structure.

---

## Experimental Design

Each synthetic knowledge graph contains people connected to numeric age values through `hasAge` relationships.

```text
person1  hasAge  age24
person2  hasAge  age37
person3  hasAge  age51
```

Two structural versions of each graph are evaluated.

### Windowed Graph

The windowed graph contains the additional window relationships used to organize the numeric portion of the graph.

### Windowless Graph

The windowless graph is created by removing only the window-related triples from the corresponding windowed graph.

The two versions therefore contain the same people, ages, and non-window relationships.

### Removal of `hasAge`

Increasing percentages of `hasAge` relationships are removed:

```text
0%, 15%, 30%, 45%, 60%, 75%
```

The removals are cumulative. Every relationship removed in a lower-removal condition remains removed in each higher-removal condition.

If `R15` represents the set of relationships removed at 15%, then:

```text
R15 ⊆ R30 ⊆ R45 ⊆ R60 ⊆ R75
```

The complete triple sets of the resulting graphs shrink in the opposite direction:

```text
KG0 ⊇ KG15 ⊇ KG30 ⊇ KG45 ⊇ KG60 ⊇ KG75
```

The same `hasAge` relationships are removed from the windowed and windowless graphs at each percentage. This allows the two graph structures to be compared without changing which people have missing age relationships.

Each resulting graph is embedded independently using MuRE.

---

## MuRE

**MuRE** stands for **Multi-Relational Euclidean** embedding.

For a query such as:

```text
(person42, hasAge, ?)
```

MuRE transforms the embedding of the head entity using the learned representation of the relation:

```text
R ⊙ h + r
```

where:

- `h` is the embedding of the head entity
- `R` is a relation-specific scaling vector
- `r` is a relation-specific translation vector
- `⊙` represents element-by-element multiplication

The transformed vector is used by both age-recovery methods.

---

## Recovery Methods

### Query-Point Recovery

The query-point method compares the transformed person vector with the embeddings of the possible age entities.

The nearest age entity is selected as the prediction.

This method asks:

> Which age entity is closest to the transformed person embedding?

Only the highest-ranked candidate is used, so this is referred to as a **top-1 prediction**.

### Learned-Regression Recovery

The learned-regression method trains a regression model to map transformed embedding vectors directly to numeric ages.

```text
Transformed embedding → predicted age
```

This method asks:

> What numeric value is represented by the transformed embedding?

Unlike query-point recovery, learned regression does not require the correct age entity to be the nearest entity in embedding space.

The two methods therefore evaluate different aspects of literal preservation:

```text
Query point:
Can the embedding retrieve the correct symbolic age entity?

Learned regression:
Can the numeric age be decoded from the embedding?
```

---

## Evaluation Populations

Results are reported for three populations.

### Full Population

Includes every person in the evaluated graph.

This combines people whose `hasAge` relationships remain present with people whose relationships were removed.

### Retained `hasAge` Relations

Includes only people whose `hasAge` relationship remains present in the evaluated graph.

These are the cases where the original age relationship is still available during embedding training.

### Removed `hasAge` Relations

Includes only people whose `hasAge` relationship was removed from the evaluated graph.

These cases more directly measure whether age information can be recovered from the remaining graph structure after the explicit relationship is unavailable.

Every person has a `hasAge` relationship in the original 0% graph. The terms **retained** and **removed** refer to whether that relationship is present in the specific experimental run being evaluated.

At 0% removal:

- The Full Population and Retained populations are identical.
- The Removed population contains no cases.

---

# Understanding the Results

## AE

**AE** means **Absolute Error**.

```text
AE = |predicted age - true age|
```

Lower values indicate more accurate predictions.

Predicting five years above the correct age and five years below the correct age both produce an AE of five years.

## MAE

**MAE** means **Mean Absolute Error**.

It is the average AE across all people in the evaluated population.

```text
MAE = average of the absolute errors
```

An MAE of `2.5` means that predictions differ from the true ages by approximately 2.5 years on average.

Lower MAE is better.

## Median AE

**Median AE** is the middle absolute error after all errors are placed in order.

It is less affected by a small number of unusually large errors than MAE.

Comparing MAE with Median AE can help show whether a small number of poor predictions are increasing the average.

## SD of AE

**SD** means **Standard Deviation**.

`SD of AE` measures how much the individual absolute errors vary.

- A lower SD indicates more consistent errors.
- A higher SD indicates greater variation between people.

## N

`N` is the number of observations included in a row.

In summary tables, it is the number of people evaluated.

In comparison tables, it is the number of paired observations used in the statistical comparison.

## Best Case and Worst Case

The summary tables report the people with the smallest and largest absolute errors.

```text
person42 (0.000)
```

means that `person42` had an absolute error of zero.

These values show the observed range of performance but should not be interpreted as overall measures of a method.

## Top-1

**Top-1** refers to the single highest-ranked prediction.

For query-point recovery, this is the age entity nearest to the transformed person vector.

---

# Recovery-Method Comparison

The recovery-method tables compare query-point recovery with learned regression for the same people.

## Query MAE

The MAE produced by query-point recovery.

## Regression MAE

The MAE produced by learned-regression recovery.

## ΔMAE (Q − R)

`Δ` means **difference** or **change**.

`ΔMAE (Q − R)` is calculated as:

```text
Query MAE - Regression MAE
```

Interpretation:

```text
Positive value → learned regression has lower error
Negative value → query point has lower error
Zero           → both methods have equal MAE
```

Here:

- `Q` means Query Point
- `R` means Learned Regression

## Lower MAE

Identifies the recovery method with the lower observed MAE.

This is a descriptive comparison of the means. The statistical tests provide additional information about the reliability and magnitude of the difference.

---

# Window-Structure Comparison

The window-comparison tables compare the windowed and windowless graph structures using the same recovery method and the same people.

## Windowed MAE

The MAE produced using the graph containing window relationships.

## Windowless MAE

The MAE produced using the graph without window relationships.

## ΔMAE (W − WL)

`ΔMAE (W − WL)` is calculated as:

```text
Windowed MAE - Windowless MAE
```

Interpretation:

```text
Positive value → the windowless graph has lower error
Negative value → the windowed graph has lower error
Zero           → both structures have equal MAE
```

Here:

- `W` means Windowed
- `WL` means Windowless

## Lower MAE

Identifies the graph structure with the lower observed MAE.

---

# Statistical Measures

The statistical comparisons are paired because the same people are evaluated under both conditions.

For example, the query-point and learned-regression errors for one person form a pair. The windowed and windowless errors for that same person also form a pair.

## Paired t p

`Paired t p` is the p-value produced by a **paired t-test**.

The test evaluates whether the average paired difference is distinguishable from zero.

A commonly used threshold is:

```text
p < 0.05
```

A small p-value provides evidence that the observed mean difference is not zero under the assumptions of the test.

The p-value does not describe the size or practical importance of the difference.

## Wilcoxon p

`Wilcoxon p` is the p-value produced by the **Wilcoxon signed-rank test**.

This is a non-parametric paired test. It is included because prediction-error differences may not follow a normal distribution.

The paired t-test examines the mean paired difference, while the Wilcoxon test considers the ranked direction and magnitude of the paired differences.

## Cohen's dz

**Cohen's `dz`** is a standardized effect-size measure for paired observations.

It describes the size of the paired difference relative to the variability of those differences.

Common approximate guidelines are:

| Absolute `dz` | Approximate interpretation |
|---:|---|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |

These are general guidelines rather than strict boundaries.

The sign of `dz` follows the direction of the difference column used in the table.

For `ΔMAE (Q − R)`:

```text
Positive dz → favors learned regression
Negative dz → favors query point
```

For `ΔMAE (W − WL)`:

```text
Positive dz → favors the windowless graph
Negative dz → favors the windowed graph
```

## 95% Bootstrap CI

**CI** means **Confidence Interval**.

The **95% bootstrap CI** estimates a range for the mean paired difference by repeatedly resampling the paired observations.

```text
[lower bound, upper bound]
```

Interpretation:

```text
Entire interval above zero → positive mean difference
Entire interval below zero → negative mean difference
Interval includes zero     → the direction is less certain
```

A narrower interval indicates a more precise estimate of the mean difference.

---

## Reading the Measures Together

The reported values describe different parts of the result:

- MAE describes average prediction accuracy.
- Median AE describes the middle prediction error.
- SD describes consistency across people.
- The paired t-test and Wilcoxon test evaluate statistical evidence.
- Cohen's `dz` describes effect size.
- The bootstrap CI describes uncertainty around the mean difference.

No single value should be treated as the complete interpretation of a comparison.

---

# Report Organization

The generated result report is organized into the following sections:

1. **Experimental Conditions**
2. **Query-Point Recovery**
   - Full Population
   - Retained `hasAge` Relations
   - Removed `hasAge` Relations
3. **Learned-Regression Recovery**
   - Full Population
   - Retained `hasAge` Relations
   - Removed `hasAge` Relations
4. **Recovery-Method Comparison**
5. **Effect of Window Structure**
6. **Metric Notes**
7. **Nested Removal**

This organization separates:

- Descriptive performance for each recovery method
- Comparisons between recovery methods
- Comparisons between graph structures
- Results for relationships that remain present and relationships that were removed

---

# Future Work

The current experiments use MuRE only.

Future work will apply the same methodology to additional knowledge graph embedding models to determine which observed behaviors are specific to MuRE and which are more general properties of embedding-based representations.

Another direction concerns pairwise ordering relationships between numeric values.

A complete pairwise construction creates an ordering relationship between every pair of numeric values.

For `n` numeric values, this produces:

```text
n(n - 1) / 2
```

relationships.

For five ordered values:

```text
n1 < n2
n1 < n3
n1 < n4
n1 < n5
n2 < n3
n2 < n4
n2 < n5
n3 < n4
n3 < n5
n4 < n5
```

This construction grows quadratically as the number of values increases and may become impractical for larger graphs.

Future experiments will investigate whether similar information can be preserved by creating pairwise relationships only within local blocks or neighborhoods.

For example, a block size of five would create complete pairwise ordering within each group:

```text
Block 1:
n1, n2, n3, n4, n5

Block 2:
n6, n7, n8, n9, n10
```

Different block sizes, such as 5, 10, or 20, can then be evaluated.

The purpose is to examine the relationship between:

- The amount of ordering information added to the graph
- The number of additional relationships created
- The computational cost of the construction
- The numerical information preserved in the embeddings

Other structural representations of numeric information may also be investigated.

The broader objective is to better understand how knowledge graph topology affects the representation and recovery of numeric literals in embedding space.