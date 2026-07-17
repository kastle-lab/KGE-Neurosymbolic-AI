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
| Windowed    | 0%        |            500 |                 500 |                  0 |
| Windowless  | 0%        |            500 |                 500 |                  0 |
| Windowed    | 15%       |            500 |                 425 |                 75 |
| Windowless  | 15%       |            500 |                 425 |                 75 |
| Windowed    | 30%       |            500 |                 350 |                150 |
| Windowless  | 30%       |            500 |                 350 |                150 |
| Windowed    | 45%       |            500 |                 275 |                225 |
| Windowless  | 45%       |            500 |                 275 |                225 |
| Windowed    | 60%       |            500 |                 200 |                300 |
| Windowless  | 60%       |            500 |                 200 |                300 |
| Windowed    | 75%       |            500 |                 125 |                375 |
| Windowless  | 75%       |            500 |                 125 |                375 |

## Query-Point Recovery

### Full Population

All people in the evaluated graph.

| Structure   | Removed   |   N |   MAE (years) |   Median AE |   SD of AE | Best case         | Worst case         |
|:------------|:----------|----:|--------------:|------------:|-----------:|:------------------|:-------------------|
| Windowed    | 0%        | 500 |         1.632 |           0 |      3.333 | person0 (0.000)   | person28 (18.000)  |
| Windowless  | 0%        | 500 |         1.53  |           0 |      3.333 | person0 (0.000)   | person80 (17.000)  |
| Windowed    | 15%       | 500 |         2.698 |           0 |      4.91  | person1 (0.000)   | person211 (27.000) |
| Windowless  | 15%       | 500 |         2.428 |           0 |      4.384 | person10 (0.000)  | person323 (22.000) |
| Windowed    | 30%       | 500 |         3.53  |           1 |      5.101 | person0 (0.000)   | person157 (29.000) |
| Windowless  | 30%       | 500 |         3.452 |           0 |      5.71  | person0 (0.000)   | person226 (48.000) |
| Windowed    | 45%       | 500 |         4.982 |           2 |      6.6   | person101 (0.000) | person103 (35.000) |
| Windowless  | 45%       | 500 |         4.438 |           2 |      5.653 | person0 (0.000)   | person495 (25.000) |
| Windowed    | 60%       | 500 |         5.16  |           3 |      5.822 | person100 (0.000) | person204 (39.000) |
| Windowless  | 60%       | 500 |         5.082 |           3 |      5.498 | person100 (0.000) | person261 (26.000) |
| Windowed    | 75%       | 500 |         8.754 |           7 |      8.203 | person101 (0.000) | person115 (45.000) |
| Windowless  | 75%       | 500 |         7.694 |           5 |      7.764 | person101 (0.000) | person451 (41.000) |

### Retained `hasAge` Relations

Only people whose `hasAge` triple remains present in the evaluated run.

| Structure   | Removed   |   N |   MAE (years) |   Median AE |   SD of AE | Best case         | Worst case         |
|:------------|:----------|----:|--------------:|------------:|-----------:|:------------------|:-------------------|
| Windowed    | 0%        | 500 |         1.632 |           0 |      3.333 | person0 (0.000)   | person28 (18.000)  |
| Windowless  | 0%        | 500 |         1.53  |           0 |      3.333 | person0 (0.000)   | person80 (17.000)  |
| Windowed    | 15%       | 425 |         2.228 |           0 |      4.598 | person101 (0.000) | person211 (27.000) |
| Windowless  | 15%       | 425 |         1.779 |           0 |      3.861 | person10 (0.000)  | person323 (22.000) |
| Windowed    | 30%       | 350 |         2.154 |           0 |      4.095 | person0 (0.000)   | person431 (20.000) |
| Windowless  | 30%       | 350 |         2.263 |           0 |      5.317 | person0 (0.000)   | person226 (48.000) |
| Windowed    | 45%       | 275 |         2.56  |           0 |      5.134 | person101 (0.000) | person87 (27.000)  |
| Windowless  | 45%       | 275 |         2.418 |           0 |      4.655 | person101 (0.000) | person495 (25.000) |
| Windowed    | 60%       | 200 |         2.725 |           0 |      4.764 | person100 (0.000) | person451 (31.000) |
| Windowless  | 60%       | 200 |         2.605 |           0 |      4.846 | person100 (0.000) | person261 (26.000) |
| Windowed    | 75%       | 125 |         4.616 |           0 |      6.762 | person101 (0.000) | person483 (27.000) |
| Windowless  | 75%       | 125 |         3.624 |           0 |      6.136 | person101 (0.000) | person361 (32.000) |

### Removed `hasAge` Relations

Only people whose `hasAge` triple was removed from the evaluated run.

| Structure   | Removed   |   N |   MAE (years) |   Median AE |   SD of AE | Best case         | Worst case         |
|:------------|:----------|----:|--------------:|------------:|-----------:|:------------------|:-------------------|
| Windowed    | 15%       |  75 |         5.36  |           3 |      5.744 | person1 (0.000)   | person397 (22.000) |
| Windowless  | 15%       |  75 |         6.107 |           5 |      5.298 | person145 (0.000) | person24 (21.000)  |
| Windowed    | 30%       | 150 |         6.74  |           5 |      5.75  | person134 (0.000) | person157 (29.000) |
| Windowless  | 30%       | 150 |         6.227 |           5 |      5.649 | person108 (0.000) | person161 (34.000) |
| Windowed    | 45%       | 225 |         7.942 |           6 |      6.985 | person108 (0.000) | person103 (35.000) |
| Windowless  | 45%       | 225 |         6.907 |           6 |      5.792 | person0 (0.000)   | person394 (24.000) |
| Windowed    | 60%       | 300 |         6.783 |           5 |      5.903 | person108 (0.000) | person204 (39.000) |
| Windowless  | 60%       | 300 |         6.733 |           6 |      5.289 | person113 (0.000) | person156 (24.000) |
| Windowed    | 75%       | 375 |        10.133 |           8 |      8.185 | person121 (0.000) | person115 (45.000) |
| Windowless  | 75%       | 375 |         9.051 |           7 |      7.781 | person139 (0.000) | person451 (41.000) |

## Learned-Regression Recovery

### Full Population

All people in the evaluated graph.

| Structure   | Removed   |   N |   MAE (years) |   Median AE |   SD of AE | Best case         | Worst case         |
|:------------|:----------|----:|--------------:|------------:|-----------:|:------------------|:-------------------|
| Windowed    | 0%        | 500 |         0.929 |       0.815 |      0.726 | person124 (0.002) | person6 (3.674)    |
| Windowless  | 0%        | 500 |         0.865 |       0.713 |      0.682 | person280 (0.001) | person431 (3.993)  |
| Windowed    | 15%       | 500 |         0.871 |       0.677 |      0.801 | person289 (0.004) | person20 (6.566)   |
| Windowless  | 15%       | 500 |         1.043 |       0.781 |      1.039 | person326 (0.000) | person200 (7.750)  |
| Windowed    | 30%       | 500 |         1.355 |       0.693 |      1.768 | person57 (0.002)  | person79 (11.595)  |
| Windowless  | 30%       | 500 |         1.223 |       0.686 |      1.535 | person88 (0.001)  | person301 (8.673)  |
| Windowed    | 45%       | 500 |         1.603 |       0.597 |      2.132 | person485 (0.002) | person9 (12.554)   |
| Windowless  | 45%       | 500 |         1.811 |       0.561 |      2.436 | person194 (0.001) | person470 (12.874) |
| Windowed    | 60%       | 500 |         1.896 |       0.928 |      2.268 | person440 (0.001) | person316 (10.648) |
| Windowless  | 60%       | 500 |         1.698 |       0.731 |      2.074 | person46 (0.000)  | person376 (10.128) |
| Windowed    | 75%       | 500 |         1.736 |       1.246 |      1.81  | person314 (0.000) | person334 (8.814)  |
| Windowless  | 75%       | 500 |         1.79  |       1.256 |      1.934 | person131 (0.000) | person139 (10.198) |

### Retained `hasAge` Relations

Only people whose `hasAge` triple remains present in the evaluated run.

| Structure   | Removed   |   N |   MAE (years) |   Median AE |   SD of AE | Best case         | Worst case        |
|:------------|:----------|----:|--------------:|------------:|-----------:|:------------------|:------------------|
| Windowed    | 0%        | 500 |         0.929 |       0.815 |      0.726 | person124 (0.002) | person6 (3.674)   |
| Windowless  | 0%        | 500 |         0.865 |       0.713 |      0.682 | person280 (0.001) | person431 (3.993) |
| Windowed    | 15%       | 425 |         0.696 |       0.58  |      0.52  | person289 (0.004) | person399 (2.852) |
| Windowless  | 15%       | 425 |         0.829 |       0.705 |      0.629 | person326 (0.000) | person37 (3.701)  |
| Windowed    | 30%       | 350 |         0.594 |       0.462 |      0.485 | person57 (0.002)  | person448 (2.569) |
| Windowless  | 30%       | 350 |         0.61  |       0.532 |      0.468 | person88 (0.001)  | person164 (2.484) |
| Windowed    | 45%       | 275 |         0.341 |       0.276 |      0.258 | person485 (0.002) | person263 (1.542) |
| Windowless  | 45%       | 275 |         0.287 |       0.247 |      0.229 | person194 (0.001) | person348 (1.179) |
| Windowed    | 60%       | 200 |         0.076 |       0.064 |      0.057 | person440 (0.001) | person401 (0.285) |
| Windowless  | 60%       | 200 |         0.056 |       0.05  |      0.04  | person46 (0.000)  | person61 (0.210)  |
| Windowed    | 75%       | 125 |         0.023 |       0.02  |      0.017 | person314 (0.000) | person70 (0.076)  |
| Windowless  | 75%       | 125 |         0.019 |       0.014 |      0.015 | person131 (0.000) | person261 (0.078) |

### Removed `hasAge` Relations

Only people whose `hasAge` triple was removed from the evaluated run.

| Structure   | Removed   |   N |   MAE (years) |   Median AE |   SD of AE | Best case         | Worst case         |
|:------------|:----------|----:|--------------:|------------:|-----------:|:------------------|:-------------------|
| Windowed    | 15%       |  75 |         1.863 |       1.708 |      1.264 | person24 (0.060)  | person20 (6.566)   |
| Windowless  | 15%       |  75 |         2.255 |       1.711 |      1.805 | person430 (0.011) | person200 (7.750)  |
| Windowed    | 30%       | 150 |         3.132 |       2.548 |      2.318 | person223 (0.002) | person79 (11.595)  |
| Windowless  | 30%       | 150 |         2.654 |       2.264 |      2.105 | person119 (0.009) | person301 (8.673)  |
| Windowed    | 45%       | 225 |         3.145 |       2.661 |      2.387 | person200 (0.031) | person9 (12.554)   |
| Windowless  | 45%       | 225 |         3.674 |       3.056 |      2.612 | person246 (0.016) | person470 (12.874) |
| Windowed    | 60%       | 300 |         3.11  |       2.592 |      2.212 | person405 (0.077) | person316 (10.648) |
| Windowless  | 60%       | 300 |         2.793 |       2.363 |      2.042 | person196 (0.024) | person376 (10.128) |
| Windowed    | 75%       | 375 |         2.307 |       1.9   |      1.751 | person5 (0.030)   | person334 (8.814)  |
| Windowless  | 75%       | 375 |         2.38  |       1.842 |      1.896 | person77 (0.001)  | person139 (10.198) |

## Recovery-Method Comparison

`ΔMAE (Q − R)` is query-point MAE minus learned-regression MAE. Positive values favor learned regression; negative values favor query-point recovery.

### Full Population

All people in the evaluated graph.

| Structure   | Removed   |   N |   Query MAE |   Regression MAE |   ΔMAE (Q − R) | Lower MAE          | Paired t p   | Wilcoxon p   |   Cohen's dz | 95% bootstrap CI   |
|:------------|:----------|----:|------------:|-----------------:|---------------:|:-------------------|:-------------|:-------------|-------------:|:-------------------|
| Windowed    | 0%        | 500 |       1.632 |            0.929 |          0.703 | Learned Regression | <0.0001      | 0.0029       |        0.207 | [0.418, 1.011]     |
| Windowless  | 0%        | 500 |       1.53  |            0.865 |          0.665 | Learned Regression | <0.0001      | <0.0001      |        0.196 | [0.378, 0.966]     |
| Windowed    | 15%       | 500 |       2.698 |            0.871 |          1.827 | Learned Regression | <0.0001      | 0.0096       |        0.378 | [1.416, 2.268]     |
| Windowless  | 15%       | 500 |       2.428 |            1.043 |          1.385 | Learned Regression | <0.0001      | 0.5514       |        0.318 | [1.001, 1.778]     |
| Windowed    | 30%       | 500 |       3.53  |            1.355 |          2.175 | Learned Regression | <0.0001      | <0.0001      |        0.443 | [1.758, 2.617]     |
| Windowless  | 30%       | 500 |       3.452 |            1.223 |          2.229 | Learned Regression | <0.0001      | <0.0001      |        0.398 | [1.758, 2.730]     |
| Windowed    | 45%       | 500 |       4.982 |            1.603 |          3.379 | Learned Regression | <0.0001      | <0.0001      |        0.521 | [2.811, 3.960]     |
| Windowless  | 45%       | 500 |       4.438 |            1.811 |          2.627 | Learned Regression | <0.0001      | <0.0001      |        0.469 | [2.155, 3.126]     |
| Windowed    | 60%       | 500 |       5.16  |            1.896 |          3.264 | Learned Regression | <0.0001      | <0.0001      |        0.545 | [2.763, 3.796]     |
| Windowless  | 60%       | 500 |       5.082 |            1.698 |          3.384 | Learned Regression | <0.0001      | <0.0001      |        0.623 | [2.915, 3.871]     |
| Windowed    | 75%       | 500 |       8.754 |            1.736 |          7.018 | Learned Regression | <0.0001      | <0.0001      |        0.871 | [6.322, 7.751]     |
| Windowless  | 75%       | 500 |       7.694 |            1.79  |          5.904 | Learned Regression | <0.0001      | <0.0001      |        0.744 | [5.210, 6.598]     |

### Retained `hasAge` Relations

Only people whose `hasAge` triple remains present in the evaluated run.

| Structure   | Removed   |   N |   Query MAE |   Regression MAE |   ΔMAE (Q − R) | Lower MAE          | Paired t p   | Wilcoxon p   |   Cohen's dz | 95% bootstrap CI   |
|:------------|:----------|----:|------------:|-----------------:|---------------:|:-------------------|:-------------|:-------------|-------------:|:-------------------|
| Windowed    | 0%        | 500 |       1.632 |            0.929 |          0.703 | Learned Regression | <0.0001      | 0.0029       |        0.207 | [0.418, 1.011]     |
| Windowless  | 0%        | 500 |       1.53  |            0.865 |          0.665 | Learned Regression | <0.0001      | <0.0001      |        0.196 | [0.378, 0.966]     |
| Windowed    | 15%       | 425 |       2.228 |            0.696 |          1.532 | Learned Regression | <0.0001      | 0.7819       |        0.331 | [1.105, 1.983]     |
| Windowless  | 15%       | 425 |       1.779 |            0.829 |          0.95  | Learned Regression | <0.0001      | 0.0004       |        0.241 | [0.593, 1.333]     |
| Windowed    | 30%       | 350 |       2.154 |            0.594 |          1.561 | Learned Regression | <0.0001      | 0.5627       |        0.379 | [1.122, 2.013]     |
| Windowless  | 30%       | 350 |       2.263 |            0.61  |          1.653 | Learned Regression | <0.0001      | 0.7002       |        0.314 | [1.120, 2.236]     |
| Windowed    | 45%       | 275 |       2.56  |            0.341 |          2.219 | Learned Regression | <0.0001      | 0.7377       |        0.433 | [1.645, 2.832]     |
| Windowless  | 45%       | 275 |       2.418 |            0.287 |          2.131 | Learned Regression | <0.0001      | 0.3001       |        0.458 | [1.578, 2.694]     |
| Windowed    | 60%       | 200 |       2.725 |            0.076 |          2.649 | Learned Regression | <0.0001      | 0.0002       |        0.557 | [2.022, 3.362]     |
| Windowless  | 60%       | 200 |       2.605 |            0.056 |          2.549 | Learned Regression | <0.0001      | 0.0867       |        0.526 | [1.890, 3.214]     |
| Windowed    | 75%       | 125 |       4.616 |            0.023 |          4.593 | Learned Regression | <0.0001      | 0.0003       |        0.679 | [3.464, 5.778]     |
| Windowless  | 75%       | 125 |       3.624 |            0.019 |          3.605 | Learned Regression | <0.0001      | 0.0007       |        0.588 | [2.572, 4.725]     |

### Removed `hasAge` Relations

Only people whose `hasAge` triple was removed from the evaluated run.

| Structure   | Removed   |   N |   Query MAE |   Regression MAE |   ΔMAE (Q − R) | Lower MAE          | Paired t p   | Wilcoxon p   |   Cohen's dz | 95% bootstrap CI   |
|:------------|:----------|----:|------------:|-----------------:|---------------:|:-------------------|:-------------|:-------------|-------------:|:-------------------|
| Windowed    | 15%       |  75 |       5.36  |            1.863 |          3.497 | Learned Regression | <0.0001      | <0.0001      |        0.624 | [2.291, 4.827]     |
| Windowless  | 15%       |  75 |       6.107 |            2.255 |          3.852 | Learned Regression | <0.0001      | <0.0001      |        0.685 | [2.653, 5.119]     |
| Windowed    | 30%       | 150 |       6.74  |            3.132 |          3.608 | Learned Regression | <0.0001      | <0.0001      |        0.584 | [2.649, 4.622]     |
| Windowless  | 30%       | 150 |       6.227 |            2.654 |          3.572 | Learned Regression | <0.0001      | <0.0001      |        0.58  | [2.584, 4.534]     |
| Windowed    | 45%       | 225 |       7.942 |            3.145 |          4.797 | Learned Regression | <0.0001      | <0.0001      |        0.629 | [3.839, 5.786]     |
| Windowless  | 45%       | 225 |       6.907 |            3.674 |          3.233 | Learned Regression | <0.0001      | <0.0001      |        0.495 | [2.387, 4.113]     |
| Windowed    | 60%       | 300 |       6.783 |            3.11  |          3.674 | Learned Regression | <0.0001      | <0.0001      |        0.551 | [2.922, 4.394]     |
| Windowless  | 60%       | 300 |       6.733 |            2.793 |          3.941 | Learned Regression | <0.0001      | <0.0001      |        0.688 | [3.284, 4.597]     |
| Windowed    | 75%       | 375 |      10.133 |            2.307 |          7.827 | Learned Regression | <0.0001      | <0.0001      |        0.943 | [7.020, 8.676]     |
| Windowless  | 75%       | 375 |       9.051 |            2.38  |          6.67  | Learned Regression | <0.0001      | <0.0001      |        0.802 | [5.884, 7.535]     |

## Effect of Window Structure

`ΔMAE (W − WL)` is windowed MAE minus windowless MAE. Positive values favor the windowless graph; negative values favor the windowed graph.

### Query-Point Recovery

#### Full Population

All people in the evaluated graph.

| Removed   |   N |   Windowed MAE |   Windowless MAE |   ΔMAE (W − WL) | Lower MAE   |   Paired t p |   Wilcoxon p |   Cohen's dz | 95% bootstrap CI   |
|:----------|----:|---------------:|-----------------:|----------------:|:------------|-------------:|-------------:|-------------:|:-------------------|
| 0%        | 500 |          1.632 |            1.53  |           0.102 | Windowless  |       0.6304 |       0.6599 |        0.022 | [-0.304, 0.520]    |
| 15%       | 500 |          2.698 |            2.428 |           0.27  | Windowless  |       0.3292 |       0.5543 |        0.044 | [-0.282, 0.820]    |
| 30%       | 500 |          3.53  |            3.452 |           0.078 | Windowless  |       0.8106 |       0.2692 |        0.011 | [-0.580, 0.710]    |
| 45%       | 500 |          4.982 |            4.438 |           0.544 | Windowless  |       0.1164 |       0.259  |        0.07  | [-0.130, 1.214]    |
| 60%       | 500 |          5.16  |            5.082 |           0.078 | Windowless  |       0.8159 |       0.9449 |        0.01  | [-0.574, 0.734]    |
| 75%       | 500 |          8.754 |            7.694 |           1.06  | Windowless  |       0.0156 |       0.047  |        0.108 | [0.232, 1.940]     |

#### Retained `hasAge` Relations

Only people whose `hasAge` triple remains present in the evaluated run.

| Removed   |   N |   Windowed MAE |   Windowless MAE |   ΔMAE (W − WL) | Lower MAE   |   Paired t p |   Wilcoxon p |   Cohen's dz | 95% bootstrap CI   |
|:----------|----:|---------------:|-----------------:|----------------:|:------------|-------------:|-------------:|-------------:|:-------------------|
| 0%        | 500 |          1.632 |            1.53  |           0.102 | Windowless  |       0.6304 |       0.6599 |        0.022 | [-0.304, 0.520]    |
| 15%       | 425 |          2.228 |            1.779 |           0.449 | Windowless  |       0.1234 |       0.1834 |        0.075 | [-0.104, 1.000]    |
| 30%       | 350 |          2.154 |            2.263 |          -0.109 | Windowed    |       0.7709 |       0.6844 |       -0.016 | [-0.886, 0.640]    |
| 45%       | 275 |          2.56  |            2.418 |           0.142 | Windowless  |       0.7389 |       0.9314 |        0.02  | [-0.673, 0.993]    |
| 60%       | 200 |          2.725 |            2.605 |           0.12  | Windowless  |       0.8084 |       0.6852 |        0.017 | [-0.855, 1.090]    |
| 75%       | 125 |          4.616 |            3.624 |           0.992 | Windowless  |       0.207  |       0.3269 |        0.113 | [-0.536, 2.560]    |

#### Removed `hasAge` Relations

Only people whose `hasAge` triple was removed from the evaluated run.

| Removed   |   N |   Windowed MAE |   Windowless MAE |   ΔMAE (W − WL) | Lower MAE   |   Paired t p |   Wilcoxon p |   Cohen's dz | 95% bootstrap CI   |
|:----------|----:|---------------:|-----------------:|----------------:|:------------|-------------:|-------------:|-------------:|:-------------------|
| 15%       |  75 |          5.36  |            6.107 |          -0.747 | Windowed    |       0.3638 |       0.2695 |       -0.106 | [-2.347, 0.840]    |
| 30%       | 150 |          6.74  |            6.227 |           0.513 | Windowless  |       0.4305 |       0.1882 |        0.065 | [-0.747, 1.787]    |
| 45%       | 225 |          7.942 |            6.907 |           1.036 | Windowless  |       0.0685 |       0.1249 |        0.122 | [-0.044, 2.151]    |
| 60%       | 300 |          6.783 |            6.733 |           0.05  | Windowless  |       0.9119 |       0.808  |        0.006 | [-0.813, 0.910]    |
| 75%       | 375 |         10.133 |            9.051 |           1.083 | Windowless  |       0.0386 |       0.0862 |        0.107 | [0.083, 2.107]     |

### Learned-Regression Recovery

#### Full Population

All people in the evaluated graph.

| Removed   |   N |   Windowed MAE |   Windowless MAE |   ΔMAE (W − WL) | Lower MAE   |   Paired t p |   Wilcoxon p |   Cohen's dz | 95% bootstrap CI   |
|:----------|----:|---------------:|-----------------:|----------------:|:------------|-------------:|-------------:|-------------:|:-------------------|
| 0%        | 500 |          0.929 |            0.865 |           0.064 | Windowless  |       0.1446 |       0.1437 |        0.065 | [-0.020, 0.146]    |
| 15%       | 500 |          0.871 |            1.043 |          -0.172 | Windowed    |       0.0008 |       0.006  |       -0.15  | [-0.269, -0.071]   |
| 30%       | 500 |          1.355 |            1.223 |           0.132 | Windowless  |       0.1083 |       0.3952 |        0.072 | [-0.028, 0.294]    |
| 45%       | 500 |          1.603 |            1.811 |          -0.209 | Windowed    |       0.0558 |       0.8947 |       -0.086 | [-0.429, 0.000]    |
| 60%       | 500 |          1.896 |            1.698 |           0.198 | Windowless  |       0.0519 |       0.0106 |        0.087 | [-0.008, 0.398]    |
| 75%       | 500 |          1.736 |            1.79  |          -0.054 | Windowed    |       0.5746 |       0.4619 |       -0.025 | [-0.242, 0.140]    |

#### Retained `hasAge` Relations

Only people whose `hasAge` triple remains present in the evaluated run.

| Removed   |   N |   Windowed MAE |   Windowless MAE |   ΔMAE (W − WL) | Lower MAE   | Paired t p   |   Wilcoxon p |   Cohen's dz | 95% bootstrap CI   |
|:----------|----:|---------------:|-----------------:|----------------:|:------------|:-------------|-------------:|-------------:|:-------------------|
| 0%        | 500 |          0.929 |            0.865 |           0.064 | Windowless  | 0.1446       |       0.1437 |        0.065 | [-0.020, 0.146]    |
| 15%       | 425 |          0.696 |            0.829 |          -0.133 | Windowed    | 0.0007       |       0.0035 |       -0.165 | [-0.209, -0.057]   |
| 30%       | 350 |          0.594 |            0.61  |          -0.016 | Windowed    | 0.6598       |       0.4011 |       -0.024 | [-0.087, 0.055]    |
| 45%       | 275 |          0.341 |            0.287 |           0.053 | Windowless  | 0.0107       |       0.0228 |        0.155 | [0.013, 0.095]     |
| 60%       | 200 |          0.076 |            0.056 |           0.02  | Windowless  | <0.0001      |       0.0003 |        0.291 | [0.010, 0.029]     |
| 75%       | 125 |          0.023 |            0.019 |           0.004 | Windowless  | 0.0227       |       0.0677 |        0.206 | [0.001, 0.008]     |

#### Removed `hasAge` Relations

Only people whose `hasAge` triple was removed from the evaluated run.

| Removed   |   N |   Windowed MAE |   Windowless MAE |   ΔMAE (W − WL) | Lower MAE   |   Paired t p |   Wilcoxon p |   Cohen's dz | 95% bootstrap CI   |
|:----------|----:|---------------:|-----------------:|----------------:|:------------|-------------:|-------------:|-------------:|:-------------------|
| 15%       |  75 |          1.863 |            2.255 |          -0.392 | Windowed    |       0.1334 |       0.2475 |       -0.175 | [-0.917, 0.119]    |
| 30%       | 150 |          3.132 |            2.654 |           0.477 | Windowless  |       0.0668 |       0.0692 |        0.151 | [-0.026, 0.989]    |
| 45%       | 225 |          3.145 |            3.674 |          -0.529 | Windowed    |       0.028  |       0.0476 |       -0.147 | [-0.996, -0.048]   |
| 60%       | 300 |          3.11  |            2.793 |           0.317 | Windowless  |       0.062  |       0.0836 |        0.108 | [-0.007, 0.646]    |
| 75%       | 375 |          2.307 |            2.38  |          -0.074 | Windowed    |       0.5673 |       0.536  |       -0.03  | [-0.325, 0.180]    |

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
