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
| Windowed    | 0%        |            100 |                 100 |                  0 |
| Windowless  | 0%        |            100 |                 100 |                  0 |
| Windowed    | 15%       |            100 |                  85 |                 15 |
| Windowless  | 15%       |            100 |                  85 |                 15 |
| Windowed    | 30%       |            100 |                  70 |                 30 |
| Windowless  | 30%       |            100 |                  70 |                 30 |
| Windowed    | 45%       |            100 |                  55 |                 45 |
| Windowless  | 45%       |            100 |                  55 |                 45 |
| Windowed    | 60%       |            100 |                  40 |                 60 |
| Windowless  | 60%       |            100 |                  40 |                 60 |
| Windowed    | 75%       |            100 |                  25 |                 75 |
| Windowless  | 75%       |            100 |                  25 |                 75 |

## Query-Point Recovery

### Full Population

All people in the evaluated graph.

| Structure   | Removed   |   N |   MAE (years) |   Median AE |   SD of AE | Best case        | Worst case        |
|:------------|:----------|----:|--------------:|------------:|-----------:|:-----------------|:------------------|
| Windowed    | 0%        | 100 |          2.79 |         0   |      5.47  | person0 (0.000)  | person69 (26.000) |
| Windowless  | 0%        | 100 |          1.98 |         0   |      3.913 | person0 (0.000)  | person27 (19.000) |
| Windowed    | 15%       | 100 |          3.49 |         0   |      5.663 | person0 (0.000)  | person68 (29.000) |
| Windowless  | 15%       | 100 |          4.64 |         0   |      7.542 | person1 (0.000)  | person43 (41.000) |
| Windowed    | 30%       | 100 |          6.2  |         3   |      7.333 | person0 (0.000)  | person9 (26.000)  |
| Windowless  | 30%       | 100 |          7.71 |         7   |      6.696 | person12 (0.000) | person17 (30.000) |
| Windowed    | 45%       | 100 |          8.62 |         6   |      8.43  | person16 (0.000) | person41 (30.000) |
| Windowless  | 45%       | 100 |          8.85 |         5.5 |     11.139 | person1 (0.000)  | person0 (49.000)  |
| Windowed    | 60%       | 100 |         12.02 |        11   |     10.111 | person2 (0.000)  | person24 (36.000) |
| Windowless  | 60%       | 100 |          7.71 |         6   |      7.32  | person1 (0.000)  | person82 (28.000) |
| Windowed    | 75%       | 100 |         12.41 |         8   |     12.256 | person12 (0.000) | person63 (47.000) |
| Windowless  | 75%       | 100 |         13.54 |        10   |     12.463 | person13 (0.000) | person19 (49.000) |

### Retained `hasAge` Relations

Only people whose `hasAge` triple remains present in the evaluated run.

| Structure   | Removed   |   N |   MAE (years) |   Median AE |   SD of AE | Best case        | Worst case        |
|:------------|:----------|----:|--------------:|------------:|-----------:|:-----------------|:------------------|
| Windowed    | 0%        | 100 |         2.79  |         0   |      5.47  | person0 (0.000)  | person69 (26.000) |
| Windowless  | 0%        | 100 |         1.98  |         0   |      3.913 | person0 (0.000)  | person27 (19.000) |
| Windowed    | 15%       |  85 |         2.165 |         0   |      3.706 | person0 (0.000)  | person16 (15.000) |
| Windowless  | 15%       |  85 |         3.165 |         0   |      5.042 | person1 (0.000)  | person97 (22.000) |
| Windowed    | 30%       |  70 |         4.486 |         0   |      6.538 | person0 (0.000)  | person86 (24.000) |
| Windowless  | 30%       |  70 |         6.471 |         4.5 |      6.248 | person12 (0.000) | person46 (28.000) |
| Windowed    | 45%       |  55 |         6.036 |         3   |      7.391 | person16 (0.000) | person41 (30.000) |
| Windowless  | 45%       |  55 |         5.418 |         0   |      8.062 | person1 (0.000)  | person48 (32.000) |
| Windowed    | 60%       |  40 |         7.925 |         4   |      8.639 | person27 (0.000) | person23 (27.000) |
| Windowless  | 60%       |  40 |         4.125 |         1.5 |      5.703 | person1 (0.000)  | person36 (24.000) |
| Windowed    | 75%       |  25 |         5.12  |         3   |      5.988 | person13 (0.000) | person30 (22.000) |
| Windowless  | 75%       |  25 |         5.56  |         2   |      7.136 | person13 (0.000) | person47 (27.000) |

### Removed `hasAge` Relations

Only people whose `hasAge` triple was removed from the evaluated run.

| Structure   | Removed   |   N |   MAE (years) |   Median AE |   SD of AE | Best case        | Worst case        |
|:------------|:----------|----:|--------------:|------------:|-----------:|:-----------------|:------------------|
| Windowed    | 15%       |  15 |        11     |        10   |      8.561 | person63 (1.000) | person68 (29.000) |
| Windowless  | 15%       |  15 |        13     |        10   |     12.711 | person17 (1.000) | person43 (41.000) |
| Windowed    | 30%       |  30 |        10.2   |         9   |      7.631 | person11 (0.000) | person9 (26.000)  |
| Windowless  | 30%       |  30 |        10.6   |        11   |      6.916 | person20 (1.000) | person17 (30.000) |
| Windowed    | 45%       |  45 |        11.778 |        11   |      8.621 | person83 (0.000) | person95 (30.000) |
| Windowless  | 45%       |  45 |        13.044 |        10   |     12.91  | person15 (0.000) | person0 (49.000)  |
| Windowed    | 60%       |  60 |        14.75  |        12.5 |     10.162 | person2 (0.000)  | person24 (36.000) |
| Windowless  | 60%       |  60 |        10.1   |         9   |      7.341 | person14 (0.000) | person82 (28.000) |
| Windowed    | 75%       |  75 |        14.84  |        10   |     12.86  | person12 (0.000) | person63 (47.000) |
| Windowless  | 75%       |  75 |        16.2   |        15   |     12.751 | person66 (0.000) | person19 (49.000) |

## Learned-Regression Recovery

### Full Population

All people in the evaluated graph.

| Structure   | Removed   |   N |   MAE (years) |   Median AE |   SD of AE | Best case        | Worst case        |
|:------------|:----------|----:|--------------:|------------:|-----------:|:-----------------|:------------------|
| Windowed    | 0%        | 100 |         0.209 |       0.131 |      0.374 | person87 (0.002) | person1 (3.596)   |
| Windowless  | 0%        | 100 |         0.248 |       0.181 |      0.215 | person28 (0.001) | person15 (0.961)  |
| Windowed    | 15%       | 100 |         0.765 |       0.152 |      1.844 | person74 (0.003) | person77 (12.432) |
| Windowless  | 15%       | 100 |         0.683 |       0.162 |      1.647 | person98 (0.001) | person68 (10.945) |
| Windowed    | 30%       | 100 |         1.107 |       0.11  |      2.158 | person79 (0.000) | person77 (11.051) |
| Windowless  | 30%       | 100 |         2.197 |       0.203 |      3.927 | person69 (0.002) | person26 (18.274) |
| Windowed    | 45%       | 100 |         2.163 |       0.163 |      3.066 | person71 (0.002) | person50 (11.062) |
| Windowless  | 45%       | 100 |         1.88  |       0.182 |      2.987 | person21 (0.001) | person76 (11.936) |
| Windowed    | 60%       | 100 |         2.62  |       1.531 |      3.257 | person72 (0.000) | person31 (14.334) |
| Windowless  | 60%       | 100 |         2.643 |       0.967 |      3.371 | person44 (0.003) | person69 (14.109) |
| Windowed    | 75%       | 100 |         3.321 |       2.202 |      3.473 | person36 (0.000) | person26 (13.421) |
| Windowless  | 75%       | 100 |         3.545 |       2.236 |      3.776 | person70 (0.000) | person69 (15.377) |

### Retained `hasAge` Relations

Only people whose `hasAge` triple remains present in the evaluated run.

| Structure   | Removed   |   N |   MAE (years) |   Median AE |   SD of AE | Best case        | Worst case       |
|:------------|:----------|----:|--------------:|------------:|-----------:|:-----------------|:-----------------|
| Windowed    | 0%        | 100 |         0.209 |       0.131 |      0.374 | person87 (0.002) | person1 (3.596)  |
| Windowless  | 0%        | 100 |         0.248 |       0.181 |      0.215 | person28 (0.001) | person15 (0.961) |
| Windowed    | 15%       |  85 |         0.153 |       0.13  |      0.128 | person74 (0.003) | person81 (0.665) |
| Windowless  | 15%       |  85 |         0.168 |       0.136 |      0.145 | person98 (0.001) | person94 (0.770) |
| Windowed    | 30%       |  70 |         0.084 |       0.076 |      0.068 | person79 (0.000) | person97 (0.262) |
| Windowless  | 30%       |  70 |         0.16  |       0.148 |      0.12  | person69 (0.002) | person50 (0.613) |
| Windowed    | 45%       |  55 |         0.09  |       0.082 |      0.059 | person71 (0.002) | person47 (0.266) |
| Windowless  | 45%       |  55 |         0.083 |       0.061 |      0.075 | person21 (0.001) | person57 (0.308) |
| Windowed    | 60%       |  40 |         0.054 |       0.052 |      0.038 | person72 (0.000) | person33 (0.169) |
| Windowless  | 60%       |  40 |         0.059 |       0.039 |      0.054 | person44 (0.003) | person34 (0.235) |
| Windowed    | 75%       |  25 |         0.049 |       0.044 |      0.031 | person36 (0.000) | person30 (0.117) |
| Windowless  | 75%       |  25 |         0.034 |       0.024 |      0.032 | person70 (0.000) | person7 (0.135)  |

### Removed `hasAge` Relations

Only people whose `hasAge` triple was removed from the evaluated run.

| Structure   | Removed   |   N |   MAE (years) |   Median AE |   SD of AE | Best case        | Worst case        |
|:------------|:----------|----:|--------------:|------------:|-----------:|:-----------------|:------------------|
| Windowed    | 15%       |  15 |         4.235 |       3.67  |      2.961 | person17 (0.976) | person77 (12.432) |
| Windowless  | 15%       |  15 |         3.597 |       2.616 |      2.888 | person78 (0.007) | person68 (10.945) |
| Windowed    | 30%       |  30 |         3.495 |       3.012 |      2.731 | person22 (0.342) | person77 (11.051) |
| Windowless  | 30%       |  30 |         6.952 |       5.614 |      4.38  | person91 (0.845) | person26 (18.274) |
| Windowed    | 45%       |  45 |         4.696 |       3.915 |      3.036 | person82 (0.098) | person50 (11.062) |
| Windowless  | 45%       |  45 |         4.077 |       2.993 |      3.33  | person63 (0.007) | person76 (11.936) |
| Windowed    | 60%       |  60 |         4.33  |       3.779 |      3.218 | person43 (0.096) | person31 (14.334) |
| Windowless  | 60%       |  60 |         4.366 |       3.789 |      3.395 | person51 (0.007) | person69 (14.109) |
| Windowed    | 75%       |  75 |         4.412 |       3.409 |      3.364 | person93 (0.076) | person26 (13.421) |
| Windowless  | 75%       |  75 |         4.716 |       3.813 |      3.677 | person82 (0.003) | person69 (15.377) |

## Recovery-Method Comparison

`ΔMAE (Q − R)` is query-point MAE minus learned-regression MAE. Positive values favor learned regression; negative values favor query-point recovery.

### Full Population

All people in the evaluated graph.

| Structure   | Removed   |   N |   Query MAE |   Regression MAE |   ΔMAE (Q − R) | Lower MAE          | Paired t p   | Wilcoxon p   |   Cohen's dz | 95% bootstrap CI   |
|:------------|:----------|----:|------------:|-----------------:|---------------:|:-------------------|:-------------|:-------------|-------------:|:-------------------|
| Windowed    | 0%        | 100 |        2.79 |            0.209 |          2.581 | Learned Regression | <0.0001      | 0.7129       |        0.478 | [1.592, 3.646]     |
| Windowless  | 0%        | 100 |        1.98 |            0.248 |          1.732 | Learned Regression | <0.0001      | 0.9015       |        0.443 | [1.028, 2.561]     |
| Windowed    | 15%       | 100 |        3.49 |            0.765 |          2.725 | Learned Regression | <0.0001      | 0.0077       |        0.519 | [1.751, 3.787]     |
| Windowless  | 15%       | 100 |        4.64 |            0.683 |          3.957 | Learned Regression | <0.0001      | 0.0018       |        0.538 | [2.581, 5.490]     |
| Windowed    | 30%       | 100 |        6.2  |            1.107 |          5.093 | Learned Regression | <0.0001      | <0.0001      |        0.751 | [3.791, 6.393]     |
| Windowless  | 30%       | 100 |        7.71 |            2.197 |          5.513 | Learned Regression | <0.0001      | <0.0001      |        0.788 | [4.181, 6.876]     |
| Windowed    | 45%       | 100 |        8.62 |            2.163 |          6.457 | Learned Regression | <0.0001      | <0.0001      |        0.8   | [4.873, 8.051]     |
| Windowless  | 45%       | 100 |        8.85 |            1.88  |          6.97  | Learned Regression | <0.0001      | <0.0001      |        0.641 | [4.969, 9.097]     |
| Windowed    | 60%       | 100 |       12.02 |            2.62  |          9.4   | Learned Regression | <0.0001      | <0.0001      |        0.951 | [7.432, 11.296]    |
| Windowless  | 60%       | 100 |        7.71 |            2.643 |          5.067 | Learned Regression | <0.0001      | <0.0001      |        0.677 | [3.627, 6.586]     |
| Windowed    | 75%       | 100 |       12.41 |            3.321 |          9.089 | Learned Regression | <0.0001      | <0.0001      |        0.774 | [6.781, 11.410]    |
| Windowless  | 75%       | 100 |       13.54 |            3.545 |          9.995 | Learned Regression | <0.0001      | <0.0001      |        0.81  | [7.576, 12.484]    |

### Retained `hasAge` Relations

Only people whose `hasAge` triple remains present in the evaluated run.

| Structure   | Removed   |   N |   Query MAE |   Regression MAE |   ΔMAE (Q − R) | Lower MAE          | Paired t p   | Wilcoxon p   |   Cohen's dz | 95% bootstrap CI   |
|:------------|:----------|----:|------------:|-----------------:|---------------:|:-------------------|:-------------|:-------------|-------------:|:-------------------|
| Windowed    | 0%        | 100 |       2.79  |            0.209 |          2.581 | Learned Regression | <0.0001      | 0.7129       |        0.478 | [1.592, 3.646]     |
| Windowless  | 0%        | 100 |       1.98  |            0.248 |          1.732 | Learned Regression | <0.0001      | 0.9015       |        0.443 | [1.028, 2.561]     |
| Windowed    | 15%       |  85 |       2.165 |            0.153 |          2.012 | Learned Regression | <0.0001      | 0.0823       |        0.544 | [1.270, 2.829]     |
| Windowless  | 15%       |  85 |       3.165 |            0.168 |          2.996 | Learned Regression | <0.0001      | 0.0280       |        0.593 | [2.011, 4.081]     |
| Windowed    | 30%       |  70 |       4.486 |            0.084 |          4.402 | Learned Regression | <0.0001      | 0.0033       |        0.673 | [2.983, 6.028]     |
| Windowless  | 30%       |  70 |       6.471 |            0.16  |          6.312 | Learned Regression | <0.0001      | <0.0001      |        1.01  | [4.856, 7.773]     |
| Windowed    | 45%       |  55 |       6.036 |            0.09  |          5.947 | Learned Regression | <0.0001      | <0.0001      |        0.803 | [4.092, 8.020]     |
| Windowless  | 45%       |  55 |       5.418 |            0.083 |          5.335 | Learned Regression | <0.0001      | 0.0050       |        0.663 | [3.330, 7.557]     |
| Windowed    | 60%       |  40 |       7.925 |            0.054 |          7.871 | Learned Regression | <0.0001      | <0.0001      |        0.911 | [5.346, 10.636]    |
| Windowless  | 60%       |  40 |       4.125 |            0.059 |          4.066 | Learned Regression | <0.0001      | 0.0009       |        0.713 | [2.493, 5.937]     |
| Windowed    | 75%       |  25 |       5.12  |            0.049 |          5.071 | Learned Regression | 0.0003       | 0.0009       |        0.849 | [2.870, 7.541]     |
| Windowless  | 75%       |  25 |       5.56  |            0.034 |          5.526 | Learned Regression | 0.0007       | 0.0219       |        0.774 | [2.883, 8.528]     |

### Removed `hasAge` Relations

Only people whose `hasAge` triple was removed from the evaluated run.

| Structure   | Removed   |   N |   Query MAE |   Regression MAE |   ΔMAE (Q − R) | Lower MAE          | Paired t p   | Wilcoxon p   |   Cohen's dz | 95% bootstrap CI   |
|:------------|:----------|----:|------------:|-----------------:|---------------:|:-------------------|:-------------|:-------------|-------------:|:-------------------|
| Windowed    | 15%       |  15 |      11     |            4.235 |          6.765 | Learned Regression | 0.0163       | 0.0125       |        0.705 | [2.325, 11.610]    |
| Windowless  | 15%       |  15 |      13     |            3.597 |          9.403 | Learned Regression | 0.0199       | 0.0302       |        0.678 | [2.718, 16.247]    |
| Windowed    | 30%       |  30 |      10.2   |            3.495 |          6.705 | Learned Regression | <0.0001      | 0.0001       |        0.937 | [4.112, 9.227]     |
| Windowless  | 30%       |  30 |      10.6   |            6.952 |          3.648 | Learned Regression | 0.0228       | 0.0248       |        0.439 | [0.665, 6.662]     |
| Windowed    | 45%       |  45 |      11.778 |            4.696 |          7.082 | Learned Regression | <0.0001      | <0.0001      |        0.799 | [4.549, 9.646]     |
| Windowless  | 45%       |  45 |      13.044 |            4.077 |          8.967 | Learned Regression | <0.0001      | <0.0001      |        0.671 | [5.184, 13.050]    |
| Windowed    | 60%       |  60 |      14.75  |            4.33  |         10.42  | Learned Regression | <0.0001      | <0.0001      |        0.984 | [7.751, 13.109]    |
| Windowless  | 60%       |  60 |      10.1   |            4.366 |          5.734 | Learned Regression | <0.0001      | <0.0001      |        0.679 | [3.563, 7.909]     |
| Windowed    | 75%       |  75 |      14.84  |            4.412 |         10.428 | Learned Regression | <0.0001      | <0.0001      |        0.81  | [7.704, 13.318]    |
| Windowless  | 75%       |  75 |      16.2   |            4.716 |         11.484 | Learned Regression | <0.0001      | <0.0001      |        0.86  | [8.523, 14.489]    |

## Effect of Window Structure

`ΔMAE (W − WL)` is windowed MAE minus windowless MAE. Positive values favor the windowless graph; negative values favor the windowed graph.

### Query-Point Recovery

#### Full Population

All people in the evaluated graph.

| Removed   |   N |   Windowed MAE |   Windowless MAE |   ΔMAE (W − WL) | Lower MAE   | Paired t p   |   Wilcoxon p |   Cohen's dz | 95% bootstrap CI   |
|:----------|----:|---------------:|-----------------:|----------------:|:------------|:-------------|-------------:|-------------:|:-------------------|
| 0%        | 100 |           2.79 |             1.98 |            0.81 | Windowless  | 0.1670       |       0.2499 |        0.139 | [-0.300, 1.940]    |
| 15%       | 100 |           3.49 |             4.64 |           -1.15 | Windowed    | 0.1776       |       0.13   |       -0.136 | [-2.820, 0.500]    |
| 30%       | 100 |           6.2  |             7.71 |           -1.51 | Windowed    | 0.1413       |       0.1483 |       -0.148 | [-3.500, 0.450]    |
| 45%       | 100 |           8.62 |             8.85 |           -0.23 | Windowed    | 0.8310       |       0.7041 |       -0.021 | [-2.350, 1.850]    |
| 60%       | 100 |          12.02 |             7.71 |            4.31 | Windowless  | <0.0001      |       0.0001 |        0.418 | [2.290, 6.320]     |
| 75%       | 100 |          12.41 |            13.54 |           -1.13 | Windowed    | 0.2686       |       0.9727 |       -0.111 | [-3.250, 0.750]    |

#### Retained `hasAge` Relations

Only people whose `hasAge` triple remains present in the evaluated run.

| Removed   |   N |   Windowed MAE |   Windowless MAE |   ΔMAE (W − WL) | Lower MAE   |   Paired t p |   Wilcoxon p |   Cohen's dz | 95% bootstrap CI   |
|:----------|----:|---------------:|-----------------:|----------------:|:------------|-------------:|-------------:|-------------:|:-------------------|
| 0%        | 100 |          2.79  |            1.98  |           0.81  | Windowless  |       0.167  |       0.2499 |        0.139 | [-0.300, 1.940]    |
| 15%       |  85 |          2.165 |            3.165 |          -1     | Windowed    |       0.1462 |       0.1485 |       -0.159 | [-2.318, 0.294]    |
| 30%       |  70 |          4.486 |            6.471 |          -1.986 | Windowed    |       0.095  |       0.0754 |       -0.202 | [-4.229, 0.371]    |
| 45%       |  55 |          6.036 |            5.418 |           0.618 | Windowless  |       0.6303 |       0.8362 |        0.065 | [-1.836, 3.164]    |
| 60%       |  40 |          7.925 |            4.125 |           3.8   | Windowless  |       0.0218 |       0.0291 |        0.378 | [0.775, 6.975]     |
| 75%       |  25 |          5.12  |            5.56  |          -0.44  | Windowed    |       0.8055 |       0.9303 |       -0.05  | [-4.160, 2.920]    |

#### Removed `hasAge` Relations

Only people whose `hasAge` triple was removed from the evaluated run.

| Removed   |   N |   Windowed MAE |   Windowless MAE |   ΔMAE (W − WL) | Lower MAE   |   Paired t p |   Wilcoxon p |   Cohen's dz | 95% bootstrap CI   |
|:----------|----:|---------------:|-----------------:|----------------:|:------------|-------------:|-------------:|-------------:|:-------------------|
| 15%       |  15 |         11     |           13     |          -2     | Windowed    |       0.6441 |       0.5287 |       -0.122 | [-9.935, 6.000]    |
| 30%       |  30 |         10.2   |           10.6   |          -0.4   | Windowed    |       0.8447 |       0.9364 |       -0.036 | [-4.333, 3.467]    |
| 45%       |  45 |         11.778 |           13.044 |          -1.267 | Windowed    |       0.4885 |       0.4677 |       -0.104 | [-4.778, 2.222]    |
| 60%       |  60 |         14.75  |           10.1   |           4.65  | Windowless  |       0.0011 |       0.0014 |        0.441 | [2.050, 7.367]     |
| 75%       |  75 |         14.84  |           16.2   |          -1.36  | Windowed    |       0.2704 |       0.9738 |       -0.128 | [-3.840, 0.880]    |

### Learned-Regression Recovery

#### Full Population

All people in the evaluated graph.

| Removed   |   N |   Windowed MAE |   Windowless MAE |   ΔMAE (W − WL) | Lower MAE   |   Paired t p | Wilcoxon p   |   Cohen's dz | 95% bootstrap CI   |
|:----------|----:|---------------:|-----------------:|----------------:|:------------|-------------:|:-------------|-------------:|:-------------------|
| 0%        | 100 |          0.209 |            0.248 |          -0.039 | Windowed    |       0.2889 | 0.0136       |       -0.107 | [-0.101, 0.038]    |
| 15%       | 100 |          0.765 |            0.683 |           0.082 | Windowless  |       0.628  | 0.5752       |        0.049 | [-0.249, 0.416]    |
| 30%       | 100 |          1.107 |            2.197 |          -1.09  | Windowed    |       0.0013 | <0.0001      |       -0.332 | [-1.734, -0.473]   |
| 45%       | 100 |          2.163 |            1.88  |           0.283 | Windowless  |       0.3356 | 0.0591       |        0.097 | [-0.299, 0.856]    |
| 60%       | 100 |          2.62  |            2.643 |          -0.023 | Windowed    |       0.9458 | 0.5799       |       -0.007 | [-0.707, 0.641]    |
| 75%       | 100 |          3.321 |            3.545 |          -0.224 | Windowed    |       0.6088 | 0.9260       |       -0.051 | [-1.087, 0.608]    |

#### Retained `hasAge` Relations

Only people whose `hasAge` triple remains present in the evaluated run.

| Removed   |   N |   Windowed MAE |   Windowless MAE |   ΔMAE (W − WL) | Lower MAE   | Paired t p   | Wilcoxon p   |   Cohen's dz | 95% bootstrap CI   |
|:----------|----:|---------------:|-----------------:|----------------:|:------------|:-------------|:-------------|-------------:|:-------------------|
| 0%        | 100 |          0.209 |            0.248 |          -0.039 | Windowed    | 0.2889       | 0.0136       |       -0.107 | [-0.101, 0.038]    |
| 15%       |  85 |          0.153 |            0.168 |          -0.016 | Windowed    | 0.3896       | 0.8079       |       -0.094 | [-0.052, 0.018]    |
| 30%       |  70 |          0.084 |            0.16  |          -0.076 | Windowed    | <0.0001      | <0.0001      |       -0.543 | [-0.109, -0.045]   |
| 45%       |  55 |          0.09  |            0.083 |           0.007 | Windowless  | 0.5860       | 0.5027       |        0.074 | [-0.019, 0.031]    |
| 60%       |  40 |          0.054 |            0.059 |          -0.005 | Windowed    | 0.6495       | 0.8160       |       -0.072 | [-0.028, 0.016]    |
| 75%       |  25 |          0.049 |            0.034 |           0.015 | Windowless  | 0.0875       | 0.0851       |        0.356 | [-0.001, 0.032]    |

#### Removed `hasAge` Relations

Only people whose `hasAge` triple was removed from the evaluated run.

| Removed   |   N |   Windowed MAE |   Windowless MAE |   ΔMAE (W − WL) | Lower MAE   |   Paired t p |   Wilcoxon p |   Cohen's dz | 95% bootstrap CI   |
|:----------|----:|---------------:|-----------------:|----------------:|:------------|-------------:|-------------:|-------------:|:-------------------|
| 15%       |  15 |          4.235 |            3.597 |           0.638 | Windowless  |       0.5868 |       0.4212 |        0.144 | [-1.638, 2.688]    |
| 30%       |  30 |          3.495 |            6.952 |          -3.457 | Windowed    |       0.0014 |       0.0008 |       -0.647 | [-5.387, -1.631]   |
| 45%       |  45 |          4.696 |            4.077 |           0.619 | Windowless  |       0.3453 |       0.1563 |        0.142 | [-0.678, 1.850]    |
| 60%       |  60 |          4.33  |            4.366 |          -0.035 | Windowed    |       0.9508 |       0.8024 |       -0.008 | [-1.152, 1.050]    |
| 75%       |  75 |          4.412 |            4.716 |          -0.304 | Windowed    |       0.6035 |       0.7634 |       -0.06  | [-1.467, 0.826]    |

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
