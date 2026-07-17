# MuRE Numeric-Literal Preservation Experiment

This report summarizes age-recovery performance across paired windowed and windowless knowledge graphs. The same `hasAge` relations are removed from both structural conditions at each removal level.

## Population Definitions

- **Full Population:** all people in the evaluated graph.
- **Retained `hasAge` Relations:** only people whose `hasAge` triple remains present in that run.
- **Removed `hasAge` Relations:** only people whose `hasAge` triple was removed from that run.

Every person has a `hasAge` relation in the original 0% graph. Here, *retained* and *removed* refer to whether that relation is present in the specific evaluated run.

## Experimental Conditions

| Structure   | Removed   |   Total people |   Retained `hasAge` |   Removed `hasAge` |
|:------------|:----------|---------------:|--------------------:|-------------------:|
| Windowed    | 0%        |            200 |                 200 |                  0 |
| Windowless  | 0%        |            200 |                 200 |                  0 |
| Windowed    | 15%       |            200 |                 170 |                 30 |
| Windowless  | 15%       |            200 |                 170 |                 30 |
| Windowed    | 30%       |            200 |                 140 |                 60 |
| Windowless  | 30%       |            200 |                 140 |                 60 |
| Windowed    | 45%       |            200 |                 110 |                 90 |
| Windowless  | 45%       |            200 |                 110 |                 90 |
| Windowed    | 60%       |            200 |                  80 |                120 |
| Windowless  | 60%       |            200 |                  80 |                120 |
| Windowed    | 75%       |            200 |                  50 |                150 |
| Windowless  | 75%       |            200 |                  50 |                150 |

## Query-Point Recovery

### Full Population

All people in the evaluated graph.

| Structure   | Removed   |   N |   MAE (years) |   Median AE |   SD of AE | Best case         | Worst case         |
|:------------|:----------|----:|--------------:|------------:|-----------:|:------------------|:-------------------|
| Windowed    | 0%        | 200 |         1.29  |         0   |      3.74  | person10 (0.000)  | person137 (21.000) |
| Windowless  | 0%        | 200 |         0.935 |         0   |      2.572 | person0 (0.000)   | person174 (12.000) |
| Windowed    | 15%       | 200 |         1.92  |         0   |      4.139 | person0 (0.000)   | person142 (21.000) |
| Windowless  | 15%       | 200 |         1.995 |         0   |      4.053 | person0 (0.000)   | person187 (22.000) |
| Windowed    | 30%       | 200 |         2.905 |         0   |      5.074 | person0 (0.000)   | person140 (32.000) |
| Windowless  | 30%       | 200 |         3.635 |         0   |      5.557 | person0 (0.000)   | person96 (25.000)  |
| Windowed    | 45%       | 200 |         4.48  |         2   |      5.545 | person0 (0.000)   | person81 (28.000)  |
| Windowless  | 45%       | 200 |         4.2   |         0.5 |      6.724 | person0 (0.000)   | person186 (34.000) |
| Windowed    | 60%       | 200 |         5.835 |         3   |      6.914 | person0 (0.000)   | person96 (37.000)  |
| Windowless  | 60%       | 200 |         5.76  |         3   |      6.585 | person0 (0.000)   | person147 (25.000) |
| Windowed    | 75%       | 200 |         8.73  |         7   |      8.397 | person0 (0.000)   | person187 (39.000) |
| Windowless  | 75%       | 200 |         7.735 |         6.5 |      6.968 | person100 (0.000) | person187 (28.000) |

### Retained `hasAge` Relations

Only people whose `hasAge` triple remains present in the evaluated run.

| Structure   | Removed   |   N |   MAE (years) |   Median AE |   SD of AE | Best case         | Worst case         |
|:------------|:----------|----:|--------------:|------------:|-----------:|:------------------|:-------------------|
| Windowed    | 0%        | 200 |         1.29  |         0   |      3.74  | person10 (0.000)  | person137 (21.000) |
| Windowless  | 0%        | 200 |         0.935 |         0   |      2.572 | person0 (0.000)   | person174 (12.000) |
| Windowed    | 15%       | 170 |         0.924 |         0   |      2.715 | person0 (0.000)   | person177 (15.000) |
| Windowless  | 15%       | 170 |         1.035 |         0   |      2.931 | person0 (0.000)   | person187 (22.000) |
| Windowed    | 30%       | 140 |         1.271 |         0   |      3.55  | person0 (0.000)   | person133 (20.000) |
| Windowless  | 30%       | 140 |         1.65  |         0   |      3.574 | person0 (0.000)   | person103 (19.000) |
| Windowed    | 45%       | 110 |         1.827 |         0   |      3.692 | person0 (0.000)   | person119 (15.000) |
| Windowless  | 45%       | 110 |         1.409 |         0   |      4.423 | person0 (0.000)   | person69 (23.000)  |
| Windowed    | 60%       |  80 |         2.125 |         0   |      4.726 | person0 (0.000)   | person126 (25.000) |
| Windowless  | 60%       |  80 |         1.887 |         0   |      4.115 | person0 (0.000)   | person159 (17.000) |
| Windowed    | 75%       |  50 |         3.84  |         0.5 |      6.476 | person0 (0.000)   | person176 (26.000) |
| Windowless  | 75%       |  50 |         3.26  |         0   |      5.114 | person100 (0.000) | person130 (18.000) |

### Removed `hasAge` Relations

Only people whose `hasAge` triple was removed from the evaluated run.

| Structure   | Removed   |   N |   MAE (years) |   Median AE |   SD of AE | Best case         | Worst case         |
|:------------|:----------|----:|--------------:|------------:|-----------:|:------------------|:-------------------|
| Windowed    | 15%       |  30 |         7.567 |         6   |      5.981 | person32 (0.000)  | person142 (21.000) |
| Windowless  | 15%       |  30 |         7.433 |         6.5 |      5.164 | person101 (0.000) | person96 (17.000)  |
| Windowed    | 30%       |  60 |         6.717 |         5   |      6.003 | person101 (0.000) | person140 (32.000) |
| Windowless  | 30%       |  60 |         8.267 |         6   |      6.548 | person72 (0.000)  | person96 (25.000)  |
| Windowed    | 45%       |  90 |         7.722 |         6   |      5.72  | person101 (0.000) | person81 (28.000)  |
| Windowless  | 45%       |  90 |         7.611 |         4   |      7.465 | person101 (0.000) | person186 (34.000) |
| Windowed    | 60%       | 120 |         8.308 |         7   |      7.048 | person121 (0.000) | person96 (37.000)  |
| Windowless  | 60%       | 120 |         8.342 |         6   |      6.668 | person129 (0.000) | person147 (25.000) |
| Windowed    | 75%       | 150 |        10.36  |         9   |      8.348 | person129 (0.000) | person187 (39.000) |
| Windowless  | 75%       | 150 |         9.227 |         8   |      6.877 | person129 (0.000) | person187 (28.000) |

## Learned-Regression Recovery

### Full Population

All people in the evaluated graph.

| Structure   | Removed   |   N |   MAE (years) |   Median AE |   SD of AE | Best case         | Worst case         |
|:------------|:----------|----:|--------------:|------------:|-----------:|:------------------|:-------------------|
| Windowed    | 0%        | 200 |         0.471 |       0.364 |      0.518 | person194 (0.002) | person1 (5.839)    |
| Windowless  | 0%        | 200 |         0.46  |       0.349 |      0.416 | person16 (0.000)  | person1 (3.301)    |
| Windowed    | 15%       | 200 |         0.835 |       0.365 |      1.619 | person45 (0.000)  | person96 (10.247)  |
| Windowless  | 15%       | 200 |         0.728 |       0.276 |      1.463 | person143 (0.000) | person16 (8.602)   |
| Windowed    | 30%       | 200 |         1.313 |       0.318 |      2.257 | person25 (0.013)  | person98 (12.422)  |
| Windowless  | 30%       | 200 |         1.239 |       0.322 |      2.319 | person60 (0.004)  | person72 (15.432)  |
| Windowed    | 45%       | 200 |         1.834 |       0.276 |      2.741 | person93 (0.002)  | person180 (15.098) |
| Windowless  | 45%       | 200 |         2.203 |       0.254 |      3.168 | person139 (0.001) | person72 (15.636)  |
| Windowed    | 60%       | 200 |         2.022 |       0.683 |      2.531 | person40 (0.003)  | person71 (12.520)  |
| Windowless  | 60%       | 200 |         2.134 |       0.598 |      2.758 | person49 (0.001)  | person121 (11.144) |
| Windowed    | 75%       | 200 |         2.446 |       1.733 |      2.492 | person136 (0.000) | person191 (12.633) |
| Windowless  | 75%       | 200 |         2.218 |       1.208 |      2.388 | person197 (0.001) | person177 (9.380)  |

### Retained `hasAge` Relations

Only people whose `hasAge` triple remains present in the evaluated run.

| Structure   | Removed   |   N |   MAE (years) |   Median AE |   SD of AE | Best case         | Worst case        |
|:------------|:----------|----:|--------------:|------------:|-----------:|:------------------|:------------------|
| Windowed    | 0%        | 200 |         0.471 |       0.364 |      0.518 | person194 (0.002) | person1 (5.839)   |
| Windowless  | 0%        | 200 |         0.46  |       0.349 |      0.416 | person16 (0.000)  | person1 (3.301)   |
| Windowed    | 15%       | 170 |         0.327 |       0.278 |      0.25  | person45 (0.000)  | person178 (1.253) |
| Windowless  | 15%       | 170 |         0.274 |       0.25  |      0.198 | person143 (0.000) | person173 (0.918) |
| Windowed    | 30%       | 140 |         0.282 |       0.209 |      0.32  | person25 (0.013)  | person1 (3.317)   |
| Windowless  | 30%       | 140 |         0.251 |       0.229 |      0.163 | person60 (0.004)  | person189 (0.761) |
| Windowed    | 45%       | 110 |         0.16  |       0.131 |      0.133 | person93 (0.002)  | person160 (0.717) |
| Windowless  | 45%       | 110 |         0.128 |       0.105 |      0.111 | person139 (0.001) | person1 (0.836)   |
| Windowed    | 60%       |  80 |         0.057 |       0.054 |      0.038 | person40 (0.003)  | person93 (0.148)  |
| Windowless  | 60%       |  80 |         0.063 |       0.049 |      0.047 | person49 (0.001)  | person52 (0.181)  |
| Windowed    | 75%       |  50 |         0.031 |       0.027 |      0.021 | person136 (0.000) | person43 (0.087)  |
| Windowless  | 75%       |  50 |         0.034 |       0.029 |      0.028 | person197 (0.001) | person52 (0.118)  |

### Removed `hasAge` Relations

Only people whose `hasAge` triple was removed from the evaluated run.

| Structure   | Removed   |   N |   MAE (years) |   Median AE |   SD of AE | Best case         | Worst case         |
|:------------|:----------|----:|--------------:|------------:|-----------:|:------------------|:-------------------|
| Windowed    | 15%       |  30 |         3.714 |       3.423 |      2.745 | person26 (0.118)  | person96 (10.247)  |
| Windowless  | 15%       |  30 |         3.302 |       3.171 |      2.531 | person156 (0.104) | person16 (8.602)   |
| Windowed    | 30%       |  60 |         3.717 |       2.802 |      2.923 | person32 (0.062)  | person98 (12.422)  |
| Windowless  | 30%       |  60 |         3.545 |       2.713 |      3.218 | person75 (0.019)  | person72 (15.432)  |
| Windowed    | 45%       |  90 |         3.88  |       3.519 |      3.013 | person68 (0.016)  | person180 (15.098) |
| Windowless  | 45%       |  90 |         4.738 |       4.091 |      3.256 | person30 (0.010)  | person72 (15.636)  |
| Windowed    | 60%       | 120 |         3.331 |       2.976 |      2.528 | person76 (0.037)  | person71 (12.520)  |
| Windowless  | 60%       | 120 |         3.515 |       2.844 |      2.813 | person167 (0.058) | person121 (11.144) |
| Windowed    | 75%       | 150 |         3.25  |       2.861 |      2.384 | person186 (0.028) | person191 (12.633) |
| Windowless  | 75%       | 150 |         2.946 |       2.227 |      2.341 | person90 (0.007)  | person177 (9.380)  |

## Recovery-Method Comparison

`ΔMAE (Q − R)` is query-point MAE minus learned-regression MAE. Positive values favor learned regression; negative values favor query-point recovery.

### Full Population

All people in the evaluated graph.

| Structure   | Removed   |   N |   Query MAE |   Regression MAE |   ΔMAE (Q − R) | Lower MAE          | Paired t p   | Wilcoxon p   |   Cohen's dz | 95% bootstrap CI   |
|:------------|:----------|----:|------------:|-----------------:|---------------:|:-------------------|:-------------|:-------------|-------------:|:-------------------|
| Windowed    | 0%        | 200 |       1.29  |            0.471 |          0.819 | Learned Regression | 0.0022       | <0.0001      |        0.219 | [0.328, 1.362]     |
| Windowless  | 0%        | 200 |       0.935 |            0.46  |          0.475 | Learned Regression | 0.0102       | <0.0001      |        0.183 | [0.133, 0.855]     |
| Windowed    | 15%       | 200 |       1.92  |            0.835 |          1.085 | Learned Regression | <0.0001      | 0.0124       |        0.313 | [0.608, 1.597]     |
| Windowless  | 15%       | 200 |       1.995 |            0.728 |          1.267 | Learned Regression | <0.0001      | 0.0256       |        0.331 | [0.756, 1.831]     |
| Windowed    | 30%       | 200 |       2.905 |            1.313 |          1.592 | Learned Regression | <0.0001      | 0.9854       |        0.337 | [0.973, 2.271]     |
| Windowless  | 30%       | 200 |       3.635 |            1.239 |          2.396 | Learned Regression | <0.0001      | 0.0003       |        0.466 | [1.705, 3.144]     |
| Windowed    | 45%       | 200 |       4.48  |            1.834 |          2.646 | Learned Regression | <0.0001      | <0.0001      |        0.496 | [1.924, 3.409]     |
| Windowless  | 45%       | 200 |       4.2   |            2.203 |          1.997 | Learned Regression | <0.0001      | 0.5788       |        0.302 | [1.115, 2.929]     |
| Windowed    | 60%       | 200 |       5.835 |            2.022 |          3.813 | Learned Regression | <0.0001      | <0.0001      |        0.58  | [2.939, 4.735]     |
| Windowless  | 60%       | 200 |       5.76  |            2.134 |          3.626 | Learned Regression | <0.0001      | <0.0001      |        0.584 | [2.799, 4.470]     |
| Windowed    | 75%       | 200 |       8.73  |            2.446 |          6.284 | Learned Regression | <0.0001      | <0.0001      |        0.739 | [5.133, 7.484]     |
| Windowless  | 75%       | 200 |       7.735 |            2.218 |          5.517 | Learned Regression | <0.0001      | <0.0001      |        0.806 | [4.571, 6.474]     |

### Retained `hasAge` Relations

Only people whose `hasAge` triple remains present in the evaluated run.

| Structure   | Removed   |   N |   Query MAE |   Regression MAE |   ΔMAE (Q − R) | Lower MAE          | Paired t p   | Wilcoxon p   |   Cohen's dz | 95% bootstrap CI   |
|:------------|:----------|----:|------------:|-----------------:|---------------:|:-------------------|:-------------|:-------------|-------------:|:-------------------|
| Windowed    | 0%        | 200 |       1.29  |            0.471 |          0.819 | Learned Regression | 0.0022       | <0.0001      |        0.219 | [0.328, 1.362]     |
| Windowless  | 0%        | 200 |       0.935 |            0.46  |          0.475 | Learned Regression | 0.0102       | <0.0001      |        0.183 | [0.133, 0.855]     |
| Windowed    | 15%       | 170 |       0.924 |            0.327 |          0.597 | Learned Regression | 0.0050       | <0.0001      |        0.218 | [0.224, 1.041]     |
| Windowless  | 15%       | 170 |       1.035 |            0.274 |          0.762 | Learned Regression | 0.0009       | <0.0001      |        0.259 | [0.345, 1.215]     |
| Windowed    | 30%       | 140 |       1.271 |            0.282 |          0.989 | Learned Regression | 0.0010       | <0.0001      |        0.285 | [0.437, 1.616]     |
| Windowless  | 30%       | 140 |       1.65  |            0.251 |          1.399 | Learned Regression | <0.0001      | 0.7916       |        0.388 | [0.832, 2.024]     |
| Windowed    | 45%       | 110 |       1.827 |            0.16  |          1.667 | Learned Regression | <0.0001      | 0.4233       |        0.451 | [1.014, 2.391]     |
| Windowless  | 45%       | 110 |       1.409 |            0.128 |          1.281 | Learned Regression | 0.0028       | 0.0003       |        0.292 | [0.533, 2.129]     |
| Windowed    | 60%       |  80 |       2.125 |            0.057 |          2.068 | Learned Regression | 0.0002       | 0.8742       |        0.438 | [1.122, 3.173]     |
| Windowless  | 60%       |  80 |       1.887 |            0.063 |          1.824 | Learned Regression | 0.0002       | 0.4719       |        0.443 | [0.949, 2.778]     |
| Windowed    | 75%       |  50 |       3.84  |            0.031 |          3.809 | Learned Regression | 0.0001       | 0.0021       |        0.588 | [2.185, 5.593]     |
| Windowless  | 75%       |  50 |       3.26  |            0.034 |          3.226 | Learned Regression | <0.0001      | 0.0506       |        0.631 | [1.849, 4.726]     |

### Removed `hasAge` Relations

Only people whose `hasAge` triple was removed from the evaluated run.

| Structure   | Removed   |   N |   Query MAE |   Regression MAE |   ΔMAE (Q − R) | Lower MAE          | Paired t p   | Wilcoxon p   |   Cohen's dz | 95% bootstrap CI   |
|:------------|:----------|----:|------------:|-----------------:|---------------:|:-------------------|:-------------|:-------------|-------------:|:-------------------|
| Windowed    | 15%       |  30 |       7.567 |            3.714 |          3.853 | Learned Regression | 0.0005       | 0.0007       |        0.71  | [1.993, 5.828]     |
| Windowless  | 15%       |  30 |       7.433 |            3.302 |          4.131 | Learned Regression | 0.0013       | 0.0054       |        0.649 | [1.887, 6.362]     |
| Windowed    | 30%       |  60 |       6.717 |            3.717 |          3     | Learned Regression | 0.0009       | 0.0011       |        0.452 | [1.456, 4.726]     |
| Windowless  | 30%       |  60 |       8.267 |            3.545 |          4.721 | Learned Regression | <0.0001      | <0.0001      |        0.663 | [2.897, 6.551]     |
| Windowed    | 45%       |  90 |       7.722 |            3.88  |          3.843 | Learned Regression | <0.0001      | <0.0001      |        0.578 | [2.540, 5.264]     |
| Windowless  | 45%       |  90 |       7.611 |            4.738 |          2.873 | Learned Regression | 0.0019       | 0.0229       |        0.337 | [1.172, 4.699]     |
| Windowed    | 60%       | 120 |       8.308 |            3.331 |          4.977 | Learned Regression | <0.0001      | <0.0001      |        0.677 | [3.708, 6.326]     |
| Windowless  | 60%       | 120 |       8.342 |            3.515 |          4.827 | Learned Regression | <0.0001      | <0.0001      |        0.686 | [3.566, 6.096]     |
| Windowed    | 75%       | 150 |      10.36  |            3.25  |          7.11  | Learned Regression | <0.0001      | <0.0001      |        0.795 | [5.735, 8.528]     |
| Windowless  | 75%       | 150 |       9.227 |            2.946 |          6.281 | Learned Regression | <0.0001      | <0.0001      |        0.874 | [5.182, 7.469]     |

## Effect of Window Structure

`ΔMAE (W − WL)` is windowed MAE minus windowless MAE. Positive values favor the windowless graph; negative values favor the windowed graph.

### Query-Point Recovery

#### Full Population

All people in the evaluated graph.

| Removed   |   N |   Windowed MAE |   Windowless MAE |   ΔMAE (W − WL) | Lower MAE   |   Paired t p |   Wilcoxon p |   Cohen's dz | 95% bootstrap CI   |
|:----------|----:|---------------:|-----------------:|----------------:|:------------|-------------:|-------------:|-------------:|:-------------------|
| 0%        | 200 |          1.29  |            0.935 |           0.355 | Windowless  |       0.2426 |       0.5103 |        0.083 | [-0.220, 0.970]    |
| 15%       | 200 |          1.92  |            1.995 |          -0.075 | Windowed    |       0.812  |       0.7705 |       -0.017 | [-0.705, 0.540]    |
| 30%       | 200 |          2.905 |            3.635 |          -0.73  | Windowed    |       0.1011 |       0.061  |       -0.116 | [-1.605, 0.130]    |
| 45%       | 200 |          4.48  |            4.2   |           0.28  | Windowless  |       0.6039 |       0.4251 |        0.037 | [-0.770, 1.325]    |
| 60%       | 200 |          5.835 |            5.76  |           0.075 | Windowless  |       0.8897 |       0.9809 |        0.01  | [-0.975, 1.120]    |
| 75%       | 200 |          8.73  |            7.735 |           0.995 | Windowless  |       0.1364 |       0.2334 |        0.106 | [-0.295, 2.320]    |

#### Retained `hasAge` Relations

Only people whose `hasAge` triple remains present in the evaluated run.

| Removed   |   N |   Windowed MAE |   Windowless MAE |   ΔMAE (W − WL) | Lower MAE   |   Paired t p |   Wilcoxon p |   Cohen's dz | 95% bootstrap CI   |
|:----------|----:|---------------:|-----------------:|----------------:|:------------|-------------:|-------------:|-------------:|:-------------------|
| 0%        | 200 |          1.29  |            0.935 |           0.355 | Windowless  |       0.2426 |       0.5103 |        0.083 | [-0.220, 0.970]    |
| 15%       | 170 |          0.924 |            1.035 |          -0.112 | Windowed    |       0.7171 |       0.7303 |       -0.028 | [-0.700, 0.488]    |
| 30%       | 140 |          1.271 |            1.65  |          -0.379 | Windowed    |       0.3399 |       0.2541 |       -0.081 | [-1.164, 0.407]    |
| 45%       | 110 |          1.827 |            1.409 |           0.418 | Windowless  |       0.445  |       0.2648 |        0.073 | [-0.636, 1.473]    |
| 60%       |  80 |          2.125 |            1.887 |           0.237 | Windowless  |       0.7174 |       0.8188 |        0.041 | [-1.075, 1.550]    |
| 75%       |  50 |          3.84  |            3.26  |           0.58  | Windowless  |       0.575  |       0.3932 |        0.08  | [-1.400, 2.600]    |

#### Removed `hasAge` Relations

Only people whose `hasAge` triple was removed from the evaluated run.

| Removed   |   N |   Windowed MAE |   Windowless MAE |   ΔMAE (W − WL) | Lower MAE   |   Paired t p |   Wilcoxon p |   Cohen's dz | 95% bootstrap CI   |
|:----------|----:|---------------:|-----------------:|----------------:|:------------|-------------:|-------------:|-------------:|:-------------------|
| 15%       |  30 |          7.567 |            7.433 |           0.133 | Windowless  |       0.9112 |       0.9904 |        0.021 | [-2.133, 2.434]    |
| 30%       |  60 |          6.717 |            8.267 |          -1.55  | Windowed    |       0.1845 |       0.1647 |       -0.173 | [-3.850, 0.717]    |
| 45%       |  90 |          7.722 |            7.611 |           0.111 | Windowless  |       0.9116 |       0.8249 |        0.012 | [-1.811, 2.044]    |
| 60%       | 120 |          8.308 |            8.342 |          -0.033 | Windowed    |       0.9664 |       0.9068 |       -0.004 | [-1.558, 1.567]    |
| 75%       | 150 |         10.36  |            9.227 |           1.133 | Windowless  |       0.1689 |       0.3165 |        0.113 | [-0.447, 2.733]    |

### Learned-Regression Recovery

#### Full Population

All people in the evaluated graph.

| Removed   |   N |   Windowed MAE |   Windowless MAE |   ΔMAE (W − WL) | Lower MAE   |   Paired t p |   Wilcoxon p |   Cohen's dz | 95% bootstrap CI   |
|:----------|----:|---------------:|-----------------:|----------------:|:------------|-------------:|-------------:|-------------:|:-------------------|
| 0%        | 200 |          0.471 |            0.46  |           0.011 | Windowless  |       0.7741 |       0.7978 |        0.02  | [-0.066, 0.087]    |
| 15%       | 200 |          0.835 |            0.728 |           0.107 | Windowless  |       0.3668 |       0.0201 |        0.064 | [-0.107, 0.345]    |
| 30%       | 200 |          1.313 |            1.239 |           0.074 | Windowless  |       0.5843 |       0.9951 |        0.039 | [-0.190, 0.340]    |
| 45%       | 200 |          1.834 |            2.203 |          -0.369 | Windowed    |       0.074  |       0.4537 |       -0.127 | [-0.786, 0.024]    |
| 60%       | 200 |          2.022 |            2.134 |          -0.113 | Windowed    |       0.5976 |       0.829  |       -0.037 | [-0.539, 0.297]    |
| 75%       | 200 |          2.446 |            2.218 |           0.228 | Windowless  |       0.2596 |       0.3019 |        0.08  | [-0.175, 0.632]    |

#### Retained `hasAge` Relations

Only people whose `hasAge` triple remains present in the evaluated run.

| Removed   |   N |   Windowed MAE |   Windowless MAE |   ΔMAE (W − WL) | Lower MAE   |   Paired t p |   Wilcoxon p |   Cohen's dz | 95% bootstrap CI   |
|:----------|----:|---------------:|-----------------:|----------------:|:------------|-------------:|-------------:|-------------:|:-------------------|
| 0%        | 200 |          0.471 |            0.46  |           0.011 | Windowless  |       0.7741 |       0.7978 |        0.02  | [-0.066, 0.087]    |
| 15%       | 170 |          0.327 |            0.274 |           0.053 | Windowless  |       0.0333 |       0.0513 |        0.165 | [0.006, 0.101]     |
| 30%       | 140 |          0.282 |            0.251 |           0.032 | Windowless  |       0.3147 |       0.7981 |        0.085 | [-0.023, 0.098]    |
| 45%       | 110 |          0.16  |            0.128 |           0.032 | Windowless  |       0.047  |       0.0911 |        0.192 | [0.001, 0.063]     |
| 60%       |  80 |          0.057 |            0.063 |          -0.006 | Windowed    |       0.3656 |       0.4013 |       -0.102 | [-0.020, 0.007]    |
| 75%       |  50 |          0.031 |            0.034 |          -0.003 | Windowed    |       0.5402 |       0.7161 |       -0.087 | [-0.012, 0.006]    |

#### Removed `hasAge` Relations

Only people whose `hasAge` triple was removed from the evaluated run.

| Removed   |   N |   Windowed MAE |   Windowless MAE |   ΔMAE (W − WL) | Lower MAE   |   Paired t p |   Wilcoxon p |   Cohen's dz | 95% bootstrap CI   |
|:----------|----:|---------------:|-----------------:|----------------:|:------------|-------------:|-------------:|-------------:|:-------------------|
| 15%       |  30 |          3.714 |            3.302 |           0.412 | Windowless  |       0.6034 |       0.4645 |        0.096 | [-1.088, 1.951]    |
| 30%       |  60 |          3.717 |            3.545 |           0.171 | Windowless  |       0.7005 |       0.8598 |        0.05  | [-0.677, 1.021]    |
| 45%       |  90 |          3.88  |            4.738 |          -0.858 | Windowed    |       0.0607 |       0.1019 |       -0.2   | [-1.727, 0.015]    |
| 60%       | 120 |          3.331 |            3.515 |          -0.184 | Windowed    |       0.6067 |       0.8238 |       -0.047 | [-0.862, 0.520]    |
| 75%       | 150 |          3.25  |            2.946 |           0.305 | Windowless  |       0.2584 |       0.2474 |        0.093 | [-0.218, 0.851]    |

## Metric Notes

- **AE:** absolute error, measured in years.
- **MAE:** mean absolute error. Lower values indicate more accurate predictions.
- **SD:** standard deviation of the absolute errors.
- **N:** number of people or paired observations in the table row.
- **Paired t p:** p-value from the paired t-test.
- **Wilcoxon p:** p-value from the Wilcoxon signed-rank test.
- **Cohen's dz:** standardized effect size for paired data.
- **95% bootstrap CI:** bootstrap confidence interval for the mean paired difference.
- **Top-1:** the nearest age entity selected by query-point recovery.

## Nested Removal

Removal sets are cumulative. If `R15` is the set of `hasAge` relations removed at 15%, then:

```text
R15 ⊆ R30 ⊆ R45 ⊆ R60 ⊆ R75
```

Equivalently, the complete triple sets of the evaluated graphs shrink in the opposite direction:

```text
KG0 ⊇ KG15 ⊇ KG30 ⊇ KG45 ⊇ KG60 ⊇ KG75
```
