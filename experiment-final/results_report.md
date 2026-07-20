# MuRE Numeric-Literal Preservation Experiment

This report combines the requested population sizes and compares paired windowed and windowless graphs. Error values and confidence intervals are measured in years.

## Population Definitions

- **Full Population:** all people.
- **Retained `hasAge` Relations:** people whose `hasAge` triple remains in the run.
- **Removed `hasAge` Relations:** people whose `hasAge` triple was removed.

At 0% removal, Full and Retained are identical and Removed has no cases.

## Experimental Conditions

|   Population Size | Structure   | Removed   |   Total People |   Retained `hasAge` |   Removed `hasAge` |
|------------------:|:------------|:----------|---------------:|--------------------:|-------------------:|
|               100 | Windowed    | 0%        |            100 |                 100 |                  0 |
|               100 | Windowless  | 0%        |            100 |                 100 |                  0 |
|               100 | Windowed    | 15%       |            100 |                  85 |                 15 |
|               100 | Windowless  | 15%       |            100 |                  85 |                 15 |
|               100 | Windowed    | 30%       |            100 |                  70 |                 30 |
|               100 | Windowless  | 30%       |            100 |                  70 |                 30 |
|               100 | Windowed    | 45%       |            100 |                  55 |                 45 |
|               100 | Windowless  | 45%       |            100 |                  55 |                 45 |
|               100 | Windowed    | 60%       |            100 |                  40 |                 60 |
|               100 | Windowless  | 60%       |            100 |                  40 |                 60 |
|               100 | Windowed    | 75%       |            100 |                  25 |                 75 |
|               100 | Windowless  | 75%       |            100 |                  25 |                 75 |
|               200 | Windowed    | 0%        |            200 |                 200 |                  0 |
|               200 | Windowless  | 0%        |            200 |                 200 |                  0 |
|               200 | Windowed    | 15%       |            200 |                 170 |                 30 |
|               200 | Windowless  | 15%       |            200 |                 170 |                 30 |
|               200 | Windowed    | 30%       |            200 |                 140 |                 60 |
|               200 | Windowless  | 30%       |            200 |                 140 |                 60 |
|               200 | Windowed    | 45%       |            200 |                 110 |                 90 |
|               200 | Windowless  | 45%       |            200 |                 110 |                 90 |
|               200 | Windowed    | 60%       |            200 |                  80 |                120 |
|               200 | Windowless  | 60%       |            200 |                  80 |                120 |
|               200 | Windowed    | 75%       |            200 |                  50 |                150 |
|               200 | Windowless  | 75%       |            200 |                  50 |                150 |
|               500 | Windowed    | 0%        |            500 |                 500 |                  0 |
|               500 | Windowless  | 0%        |            500 |                 500 |                  0 |
|               500 | Windowed    | 15%       |            500 |                 425 |                 75 |
|               500 | Windowless  | 15%       |            500 |                 425 |                 75 |
|               500 | Windowed    | 30%       |            500 |                 350 |                150 |
|               500 | Windowless  | 30%       |            500 |                 350 |                150 |
|               500 | Windowed    | 45%       |            500 |                 275 |                225 |
|               500 | Windowless  | 45%       |            500 |                 275 |                225 |
|               500 | Windowed    | 60%       |            500 |                 200 |                300 |
|               500 | Windowless  | 60%       |            500 |                 200 |                300 |
|               500 | Windowed    | 75%       |            500 |                 125 |                375 |
|               500 | Windowless  | 75%       |            500 |                 125 |                375 |

## Query-Point Recovery

### Full Population

All people in the evaluated graph.

|   Population Size | Structure   | Removed   |   N |   MAE (years) |   Median AE (years) |   SD AE (years) | Best case               | Worst case               |
|------------------:|:------------|:----------|----:|--------------:|--------------------:|----------------:|:------------------------|:-------------------------|
|               100 | Windowed    | 0%        | 100 |         2.79  |                 0   |           5.47  | person0 (0.000 years)   | person69 (26.000 years)  |
|               100 | Windowless  | 0%        | 100 |         1.98  |                 0   |           3.913 | person0 (0.000 years)   | person27 (19.000 years)  |
|               100 | Windowed    | 15%       | 100 |         3.49  |                 0   |           5.663 | person0 (0.000 years)   | person68 (29.000 years)  |
|               100 | Windowless  | 15%       | 100 |         4.64  |                 0   |           7.542 | person1 (0.000 years)   | person43 (41.000 years)  |
|               100 | Windowed    | 30%       | 100 |         6.2   |                 3   |           7.333 | person0 (0.000 years)   | person9 (26.000 years)   |
|               100 | Windowless  | 30%       | 100 |         7.71  |                 7   |           6.696 | person12 (0.000 years)  | person17 (30.000 years)  |
|               100 | Windowed    | 45%       | 100 |         8.62  |                 6   |           8.43  | person16 (0.000 years)  | person41 (30.000 years)  |
|               100 | Windowless  | 45%       | 100 |         8.85  |                 5.5 |          11.139 | person1 (0.000 years)   | person0 (49.000 years)   |
|               100 | Windowed    | 60%       | 100 |        12.02  |                11   |          10.111 | person2 (0.000 years)   | person24 (36.000 years)  |
|               100 | Windowless  | 60%       | 100 |         7.71  |                 6   |           7.32  | person1 (0.000 years)   | person82 (28.000 years)  |
|               100 | Windowed    | 75%       | 100 |        12.41  |                 8   |          12.256 | person12 (0.000 years)  | person63 (47.000 years)  |
|               100 | Windowless  | 75%       | 100 |        13.54  |                10   |          12.463 | person13 (0.000 years)  | person19 (49.000 years)  |
|               200 | Windowed    | 0%        | 200 |         1.29  |                 0   |           3.74  | person10 (0.000 years)  | person137 (21.000 years) |
|               200 | Windowless  | 0%        | 200 |         0.935 |                 0   |           2.572 | person0 (0.000 years)   | person174 (12.000 years) |
|               200 | Windowed    | 15%       | 200 |         1.92  |                 0   |           4.139 | person0 (0.000 years)   | person142 (21.000 years) |
|               200 | Windowless  | 15%       | 200 |         1.995 |                 0   |           4.053 | person0 (0.000 years)   | person187 (22.000 years) |
|               200 | Windowed    | 30%       | 200 |         2.905 |                 0   |           5.074 | person0 (0.000 years)   | person140 (32.000 years) |
|               200 | Windowless  | 30%       | 200 |         3.635 |                 0   |           5.557 | person0 (0.000 years)   | person96 (25.000 years)  |
|               200 | Windowed    | 45%       | 200 |         4.48  |                 2   |           5.545 | person0 (0.000 years)   | person81 (28.000 years)  |
|               200 | Windowless  | 45%       | 200 |         4.2   |                 0.5 |           6.724 | person0 (0.000 years)   | person186 (34.000 years) |
|               200 | Windowed    | 60%       | 200 |         5.835 |                 3   |           6.914 | person0 (0.000 years)   | person96 (37.000 years)  |
|               200 | Windowless  | 60%       | 200 |         5.76  |                 3   |           6.585 | person0 (0.000 years)   | person147 (25.000 years) |
|               200 | Windowed    | 75%       | 200 |         8.73  |                 7   |           8.397 | person0 (0.000 years)   | person187 (39.000 years) |
|               200 | Windowless  | 75%       | 200 |         7.735 |                 6.5 |           6.968 | person100 (0.000 years) | person187 (28.000 years) |
|               500 | Windowed    | 0%        | 500 |         1.632 |                 0   |           3.333 | person0 (0.000 years)   | person28 (18.000 years)  |
|               500 | Windowless  | 0%        | 500 |         1.53  |                 0   |           3.333 | person0 (0.000 years)   | person80 (17.000 years)  |
|               500 | Windowed    | 15%       | 500 |         2.698 |                 0   |           4.91  | person1 (0.000 years)   | person211 (27.000 years) |
|               500 | Windowless  | 15%       | 500 |         2.428 |                 0   |           4.384 | person10 (0.000 years)  | person323 (22.000 years) |
|               500 | Windowed    | 30%       | 500 |         3.53  |                 1   |           5.101 | person0 (0.000 years)   | person157 (29.000 years) |
|               500 | Windowless  | 30%       | 500 |         3.452 |                 0   |           5.71  | person0 (0.000 years)   | person226 (48.000 years) |
|               500 | Windowed    | 45%       | 500 |         4.982 |                 2   |           6.6   | person101 (0.000 years) | person103 (35.000 years) |
|               500 | Windowless  | 45%       | 500 |         4.438 |                 2   |           5.653 | person0 (0.000 years)   | person495 (25.000 years) |
|               500 | Windowed    | 60%       | 500 |         5.16  |                 3   |           5.822 | person100 (0.000 years) | person204 (39.000 years) |
|               500 | Windowless  | 60%       | 500 |         5.082 |                 3   |           5.498 | person100 (0.000 years) | person261 (26.000 years) |
|               500 | Windowed    | 75%       | 500 |         8.754 |                 7   |           8.203 | person101 (0.000 years) | person115 (45.000 years) |
|               500 | Windowless  | 75%       | 500 |         7.694 |                 5   |           7.764 | person101 (0.000 years) | person451 (41.000 years) |

### Retained `hasAge` Relations

Only people whose `hasAge` triple remains in the evaluated run.

|   Population Size | Structure   | Removed   |   N |   MAE (years) |   Median AE (years) |   SD AE (years) | Best case               | Worst case               |
|------------------:|:------------|:----------|----:|--------------:|--------------------:|----------------:|:------------------------|:-------------------------|
|               100 | Windowed    | 0%        | 100 |         2.79  |                 0   |           5.47  | person0 (0.000 years)   | person69 (26.000 years)  |
|               100 | Windowless  | 0%        | 100 |         1.98  |                 0   |           3.913 | person0 (0.000 years)   | person27 (19.000 years)  |
|               100 | Windowed    | 15%       |  85 |         2.165 |                 0   |           3.706 | person0 (0.000 years)   | person16 (15.000 years)  |
|               100 | Windowless  | 15%       |  85 |         3.165 |                 0   |           5.042 | person1 (0.000 years)   | person97 (22.000 years)  |
|               100 | Windowed    | 30%       |  70 |         4.486 |                 0   |           6.538 | person0 (0.000 years)   | person86 (24.000 years)  |
|               100 | Windowless  | 30%       |  70 |         6.471 |                 4.5 |           6.248 | person12 (0.000 years)  | person46 (28.000 years)  |
|               100 | Windowed    | 45%       |  55 |         6.036 |                 3   |           7.391 | person16 (0.000 years)  | person41 (30.000 years)  |
|               100 | Windowless  | 45%       |  55 |         5.418 |                 0   |           8.062 | person1 (0.000 years)   | person48 (32.000 years)  |
|               100 | Windowed    | 60%       |  40 |         7.925 |                 4   |           8.639 | person27 (0.000 years)  | person23 (27.000 years)  |
|               100 | Windowless  | 60%       |  40 |         4.125 |                 1.5 |           5.703 | person1 (0.000 years)   | person36 (24.000 years)  |
|               100 | Windowed    | 75%       |  25 |         5.12  |                 3   |           5.988 | person13 (0.000 years)  | person30 (22.000 years)  |
|               100 | Windowless  | 75%       |  25 |         5.56  |                 2   |           7.136 | person13 (0.000 years)  | person47 (27.000 years)  |
|               200 | Windowed    | 0%        | 200 |         1.29  |                 0   |           3.74  | person10 (0.000 years)  | person137 (21.000 years) |
|               200 | Windowless  | 0%        | 200 |         0.935 |                 0   |           2.572 | person0 (0.000 years)   | person174 (12.000 years) |
|               200 | Windowed    | 15%       | 170 |         0.924 |                 0   |           2.715 | person0 (0.000 years)   | person177 (15.000 years) |
|               200 | Windowless  | 15%       | 170 |         1.035 |                 0   |           2.931 | person0 (0.000 years)   | person187 (22.000 years) |
|               200 | Windowed    | 30%       | 140 |         1.271 |                 0   |           3.55  | person0 (0.000 years)   | person133 (20.000 years) |
|               200 | Windowless  | 30%       | 140 |         1.65  |                 0   |           3.574 | person0 (0.000 years)   | person103 (19.000 years) |
|               200 | Windowed    | 45%       | 110 |         1.827 |                 0   |           3.692 | person0 (0.000 years)   | person119 (15.000 years) |
|               200 | Windowless  | 45%       | 110 |         1.409 |                 0   |           4.423 | person0 (0.000 years)   | person69 (23.000 years)  |
|               200 | Windowed    | 60%       |  80 |         2.125 |                 0   |           4.726 | person0 (0.000 years)   | person126 (25.000 years) |
|               200 | Windowless  | 60%       |  80 |         1.887 |                 0   |           4.115 | person0 (0.000 years)   | person159 (17.000 years) |
|               200 | Windowed    | 75%       |  50 |         3.84  |                 0.5 |           6.476 | person0 (0.000 years)   | person176 (26.000 years) |
|               200 | Windowless  | 75%       |  50 |         3.26  |                 0   |           5.114 | person100 (0.000 years) | person130 (18.000 years) |
|               500 | Windowed    | 0%        | 500 |         1.632 |                 0   |           3.333 | person0 (0.000 years)   | person28 (18.000 years)  |
|               500 | Windowless  | 0%        | 500 |         1.53  |                 0   |           3.333 | person0 (0.000 years)   | person80 (17.000 years)  |
|               500 | Windowed    | 15%       | 425 |         2.228 |                 0   |           4.598 | person101 (0.000 years) | person211 (27.000 years) |
|               500 | Windowless  | 15%       | 425 |         1.779 |                 0   |           3.861 | person10 (0.000 years)  | person323 (22.000 years) |
|               500 | Windowed    | 30%       | 350 |         2.154 |                 0   |           4.095 | person0 (0.000 years)   | person431 (20.000 years) |
|               500 | Windowless  | 30%       | 350 |         2.263 |                 0   |           5.317 | person0 (0.000 years)   | person226 (48.000 years) |
|               500 | Windowed    | 45%       | 275 |         2.56  |                 0   |           5.134 | person101 (0.000 years) | person87 (27.000 years)  |
|               500 | Windowless  | 45%       | 275 |         2.418 |                 0   |           4.655 | person101 (0.000 years) | person495 (25.000 years) |
|               500 | Windowed    | 60%       | 200 |         2.725 |                 0   |           4.764 | person100 (0.000 years) | person451 (31.000 years) |
|               500 | Windowless  | 60%       | 200 |         2.605 |                 0   |           4.846 | person100 (0.000 years) | person261 (26.000 years) |
|               500 | Windowed    | 75%       | 125 |         4.616 |                 0   |           6.762 | person101 (0.000 years) | person483 (27.000 years) |
|               500 | Windowless  | 75%       | 125 |         3.624 |                 0   |           6.136 | person101 (0.000 years) | person361 (32.000 years) |

### Removed `hasAge` Relations

Only people whose `hasAge` triple was removed from the evaluated run.

|   Population Size | Structure   | Removed   |   N |   MAE (years) |   Median AE (years) |   SD AE (years) | Best case               | Worst case               |
|------------------:|:------------|:----------|----:|--------------:|--------------------:|----------------:|:------------------------|:-------------------------|
|               100 | Windowed    | 15%       |  15 |        11     |                10   |           8.561 | person63 (1.000 years)  | person68 (29.000 years)  |
|               100 | Windowless  | 15%       |  15 |        13     |                10   |          12.711 | person17 (1.000 years)  | person43 (41.000 years)  |
|               100 | Windowed    | 30%       |  30 |        10.2   |                 9   |           7.631 | person11 (0.000 years)  | person9 (26.000 years)   |
|               100 | Windowless  | 30%       |  30 |        10.6   |                11   |           6.916 | person20 (1.000 years)  | person17 (30.000 years)  |
|               100 | Windowed    | 45%       |  45 |        11.778 |                11   |           8.621 | person83 (0.000 years)  | person95 (30.000 years)  |
|               100 | Windowless  | 45%       |  45 |        13.044 |                10   |          12.91  | person15 (0.000 years)  | person0 (49.000 years)   |
|               100 | Windowed    | 60%       |  60 |        14.75  |                12.5 |          10.162 | person2 (0.000 years)   | person24 (36.000 years)  |
|               100 | Windowless  | 60%       |  60 |        10.1   |                 9   |           7.341 | person14 (0.000 years)  | person82 (28.000 years)  |
|               100 | Windowed    | 75%       |  75 |        14.84  |                10   |          12.86  | person12 (0.000 years)  | person63 (47.000 years)  |
|               100 | Windowless  | 75%       |  75 |        16.2   |                15   |          12.751 | person66 (0.000 years)  | person19 (49.000 years)  |
|               200 | Windowed    | 15%       |  30 |         7.567 |                 6   |           5.981 | person32 (0.000 years)  | person142 (21.000 years) |
|               200 | Windowless  | 15%       |  30 |         7.433 |                 6.5 |           5.164 | person101 (0.000 years) | person96 (17.000 years)  |
|               200 | Windowed    | 30%       |  60 |         6.717 |                 5   |           6.003 | person101 (0.000 years) | person140 (32.000 years) |
|               200 | Windowless  | 30%       |  60 |         8.267 |                 6   |           6.548 | person72 (0.000 years)  | person96 (25.000 years)  |
|               200 | Windowed    | 45%       |  90 |         7.722 |                 6   |           5.72  | person101 (0.000 years) | person81 (28.000 years)  |
|               200 | Windowless  | 45%       |  90 |         7.611 |                 4   |           7.465 | person101 (0.000 years) | person186 (34.000 years) |
|               200 | Windowed    | 60%       | 120 |         8.308 |                 7   |           7.048 | person121 (0.000 years) | person96 (37.000 years)  |
|               200 | Windowless  | 60%       | 120 |         8.342 |                 6   |           6.668 | person129 (0.000 years) | person147 (25.000 years) |
|               200 | Windowed    | 75%       | 150 |        10.36  |                 9   |           8.348 | person129 (0.000 years) | person187 (39.000 years) |
|               200 | Windowless  | 75%       | 150 |         9.227 |                 8   |           6.877 | person129 (0.000 years) | person187 (28.000 years) |
|               500 | Windowed    | 15%       |  75 |         5.36  |                 3   |           5.744 | person1 (0.000 years)   | person397 (22.000 years) |
|               500 | Windowless  | 15%       |  75 |         6.107 |                 5   |           5.298 | person145 (0.000 years) | person24 (21.000 years)  |
|               500 | Windowed    | 30%       | 150 |         6.74  |                 5   |           5.75  | person134 (0.000 years) | person157 (29.000 years) |
|               500 | Windowless  | 30%       | 150 |         6.227 |                 5   |           5.649 | person108 (0.000 years) | person161 (34.000 years) |
|               500 | Windowed    | 45%       | 225 |         7.942 |                 6   |           6.985 | person108 (0.000 years) | person103 (35.000 years) |
|               500 | Windowless  | 45%       | 225 |         6.907 |                 6   |           5.792 | person0 (0.000 years)   | person394 (24.000 years) |
|               500 | Windowed    | 60%       | 300 |         6.783 |                 5   |           5.903 | person108 (0.000 years) | person204 (39.000 years) |
|               500 | Windowless  | 60%       | 300 |         6.733 |                 6   |           5.289 | person113 (0.000 years) | person156 (24.000 years) |
|               500 | Windowed    | 75%       | 375 |        10.133 |                 8   |           8.185 | person121 (0.000 years) | person115 (45.000 years) |
|               500 | Windowless  | 75%       | 375 |         9.051 |                 7   |           7.781 | person139 (0.000 years) | person451 (41.000 years) |

## Learned-Regression Recovery

### Full Population

All people in the evaluated graph.

|   Population Size | Structure   | Removed   |   N |   MAE (years) |   Median AE (years) |   SD AE (years) | Best case               | Worst case               |
|------------------:|:------------|:----------|----:|--------------:|--------------------:|----------------:|:------------------------|:-------------------------|
|               100 | Windowed    | 0%        | 100 |         0.209 |               0.131 |           0.374 | person87 (0.002 years)  | person1 (3.596 years)    |
|               100 | Windowless  | 0%        | 100 |         0.248 |               0.181 |           0.215 | person28 (0.001 years)  | person15 (0.961 years)   |
|               100 | Windowed    | 15%       | 100 |         0.765 |               0.152 |           1.844 | person74 (0.003 years)  | person77 (12.432 years)  |
|               100 | Windowless  | 15%       | 100 |         0.683 |               0.162 |           1.647 | person98 (0.001 years)  | person68 (10.945 years)  |
|               100 | Windowed    | 30%       | 100 |         1.107 |               0.11  |           2.158 | person79 (0.000 years)  | person77 (11.051 years)  |
|               100 | Windowless  | 30%       | 100 |         2.197 |               0.203 |           3.927 | person69 (0.002 years)  | person26 (18.274 years)  |
|               100 | Windowed    | 45%       | 100 |         2.163 |               0.163 |           3.066 | person71 (0.002 years)  | person50 (11.062 years)  |
|               100 | Windowless  | 45%       | 100 |         1.88  |               0.182 |           2.987 | person21 (0.001 years)  | person76 (11.936 years)  |
|               100 | Windowed    | 60%       | 100 |         2.62  |               1.531 |           3.257 | person72 (0.000 years)  | person31 (14.334 years)  |
|               100 | Windowless  | 60%       | 100 |         2.643 |               0.967 |           3.371 | person44 (0.003 years)  | person69 (14.109 years)  |
|               100 | Windowed    | 75%       | 100 |         3.321 |               2.202 |           3.473 | person36 (0.000 years)  | person26 (13.421 years)  |
|               100 | Windowless  | 75%       | 100 |         3.545 |               2.236 |           3.776 | person70 (0.000 years)  | person69 (15.377 years)  |
|               200 | Windowed    | 0%        | 200 |         0.471 |               0.364 |           0.518 | person194 (0.002 years) | person1 (5.839 years)    |
|               200 | Windowless  | 0%        | 200 |         0.46  |               0.349 |           0.416 | person16 (0.000 years)  | person1 (3.301 years)    |
|               200 | Windowed    | 15%       | 200 |         0.835 |               0.365 |           1.619 | person45 (0.000 years)  | person96 (10.247 years)  |
|               200 | Windowless  | 15%       | 200 |         0.728 |               0.276 |           1.463 | person143 (0.000 years) | person16 (8.602 years)   |
|               200 | Windowed    | 30%       | 200 |         1.313 |               0.318 |           2.257 | person25 (0.013 years)  | person98 (12.422 years)  |
|               200 | Windowless  | 30%       | 200 |         1.239 |               0.322 |           2.319 | person60 (0.004 years)  | person72 (15.432 years)  |
|               200 | Windowed    | 45%       | 200 |         1.834 |               0.276 |           2.741 | person93 (0.002 years)  | person180 (15.098 years) |
|               200 | Windowless  | 45%       | 200 |         2.203 |               0.254 |           3.168 | person139 (0.001 years) | person72 (15.636 years)  |
|               200 | Windowed    | 60%       | 200 |         2.022 |               0.683 |           2.531 | person40 (0.003 years)  | person71 (12.520 years)  |
|               200 | Windowless  | 60%       | 200 |         2.134 |               0.598 |           2.758 | person49 (0.001 years)  | person121 (11.144 years) |
|               200 | Windowed    | 75%       | 200 |         2.446 |               1.733 |           2.492 | person136 (0.000 years) | person191 (12.633 years) |
|               200 | Windowless  | 75%       | 200 |         2.218 |               1.208 |           2.388 | person197 (0.001 years) | person177 (9.380 years)  |
|               500 | Windowed    | 0%        | 500 |         0.929 |               0.815 |           0.726 | person124 (0.002 years) | person6 (3.674 years)    |
|               500 | Windowless  | 0%        | 500 |         0.865 |               0.713 |           0.682 | person280 (0.001 years) | person431 (3.993 years)  |
|               500 | Windowed    | 15%       | 500 |         0.871 |               0.677 |           0.801 | person289 (0.004 years) | person20 (6.566 years)   |
|               500 | Windowless  | 15%       | 500 |         1.043 |               0.781 |           1.039 | person326 (0.000 years) | person200 (7.750 years)  |
|               500 | Windowed    | 30%       | 500 |         1.355 |               0.693 |           1.768 | person57 (0.002 years)  | person79 (11.595 years)  |
|               500 | Windowless  | 30%       | 500 |         1.223 |               0.686 |           1.535 | person88 (0.001 years)  | person301 (8.673 years)  |
|               500 | Windowed    | 45%       | 500 |         1.603 |               0.597 |           2.132 | person485 (0.002 years) | person9 (12.554 years)   |
|               500 | Windowless  | 45%       | 500 |         1.811 |               0.561 |           2.436 | person194 (0.001 years) | person470 (12.874 years) |
|               500 | Windowed    | 60%       | 500 |         1.896 |               0.928 |           2.268 | person440 (0.001 years) | person316 (10.648 years) |
|               500 | Windowless  | 60%       | 500 |         1.698 |               0.731 |           2.074 | person46 (0.000 years)  | person376 (10.128 years) |
|               500 | Windowed    | 75%       | 500 |         1.736 |               1.246 |           1.81  | person314 (0.000 years) | person334 (8.814 years)  |
|               500 | Windowless  | 75%       | 500 |         1.79  |               1.256 |           1.934 | person131 (0.000 years) | person139 (10.198 years) |

### Retained `hasAge` Relations

Only people whose `hasAge` triple remains in the evaluated run.

|   Population Size | Structure   | Removed   |   N |   MAE (years) |   Median AE (years) |   SD AE (years) | Best case               | Worst case              |
|------------------:|:------------|:----------|----:|--------------:|--------------------:|----------------:|:------------------------|:------------------------|
|               100 | Windowed    | 0%        | 100 |         0.209 |               0.131 |           0.374 | person87 (0.002 years)  | person1 (3.596 years)   |
|               100 | Windowless  | 0%        | 100 |         0.248 |               0.181 |           0.215 | person28 (0.001 years)  | person15 (0.961 years)  |
|               100 | Windowed    | 15%       |  85 |         0.153 |               0.13  |           0.128 | person74 (0.003 years)  | person81 (0.665 years)  |
|               100 | Windowless  | 15%       |  85 |         0.168 |               0.136 |           0.145 | person98 (0.001 years)  | person94 (0.770 years)  |
|               100 | Windowed    | 30%       |  70 |         0.084 |               0.076 |           0.068 | person79 (0.000 years)  | person97 (0.262 years)  |
|               100 | Windowless  | 30%       |  70 |         0.16  |               0.148 |           0.12  | person69 (0.002 years)  | person50 (0.613 years)  |
|               100 | Windowed    | 45%       |  55 |         0.09  |               0.082 |           0.059 | person71 (0.002 years)  | person47 (0.266 years)  |
|               100 | Windowless  | 45%       |  55 |         0.083 |               0.061 |           0.075 | person21 (0.001 years)  | person57 (0.308 years)  |
|               100 | Windowed    | 60%       |  40 |         0.054 |               0.052 |           0.038 | person72 (0.000 years)  | person33 (0.169 years)  |
|               100 | Windowless  | 60%       |  40 |         0.059 |               0.039 |           0.054 | person44 (0.003 years)  | person34 (0.235 years)  |
|               100 | Windowed    | 75%       |  25 |         0.049 |               0.044 |           0.031 | person36 (0.000 years)  | person30 (0.117 years)  |
|               100 | Windowless  | 75%       |  25 |         0.034 |               0.024 |           0.032 | person70 (0.000 years)  | person7 (0.135 years)   |
|               200 | Windowed    | 0%        | 200 |         0.471 |               0.364 |           0.518 | person194 (0.002 years) | person1 (5.839 years)   |
|               200 | Windowless  | 0%        | 200 |         0.46  |               0.349 |           0.416 | person16 (0.000 years)  | person1 (3.301 years)   |
|               200 | Windowed    | 15%       | 170 |         0.327 |               0.278 |           0.25  | person45 (0.000 years)  | person178 (1.253 years) |
|               200 | Windowless  | 15%       | 170 |         0.274 |               0.25  |           0.198 | person143 (0.000 years) | person173 (0.918 years) |
|               200 | Windowed    | 30%       | 140 |         0.282 |               0.209 |           0.32  | person25 (0.013 years)  | person1 (3.317 years)   |
|               200 | Windowless  | 30%       | 140 |         0.251 |               0.229 |           0.163 | person60 (0.004 years)  | person189 (0.761 years) |
|               200 | Windowed    | 45%       | 110 |         0.16  |               0.131 |           0.133 | person93 (0.002 years)  | person160 (0.717 years) |
|               200 | Windowless  | 45%       | 110 |         0.128 |               0.105 |           0.111 | person139 (0.001 years) | person1 (0.836 years)   |
|               200 | Windowed    | 60%       |  80 |         0.057 |               0.054 |           0.038 | person40 (0.003 years)  | person93 (0.148 years)  |
|               200 | Windowless  | 60%       |  80 |         0.063 |               0.049 |           0.047 | person49 (0.001 years)  | person52 (0.181 years)  |
|               200 | Windowed    | 75%       |  50 |         0.031 |               0.027 |           0.021 | person136 (0.000 years) | person43 (0.087 years)  |
|               200 | Windowless  | 75%       |  50 |         0.034 |               0.029 |           0.028 | person197 (0.001 years) | person52 (0.118 years)  |
|               500 | Windowed    | 0%        | 500 |         0.929 |               0.815 |           0.726 | person124 (0.002 years) | person6 (3.674 years)   |
|               500 | Windowless  | 0%        | 500 |         0.865 |               0.713 |           0.682 | person280 (0.001 years) | person431 (3.993 years) |
|               500 | Windowed    | 15%       | 425 |         0.696 |               0.58  |           0.52  | person289 (0.004 years) | person399 (2.852 years) |
|               500 | Windowless  | 15%       | 425 |         0.829 |               0.705 |           0.629 | person326 (0.000 years) | person37 (3.701 years)  |
|               500 | Windowed    | 30%       | 350 |         0.594 |               0.462 |           0.485 | person57 (0.002 years)  | person448 (2.569 years) |
|               500 | Windowless  | 30%       | 350 |         0.61  |               0.532 |           0.468 | person88 (0.001 years)  | person164 (2.484 years) |
|               500 | Windowed    | 45%       | 275 |         0.341 |               0.276 |           0.258 | person485 (0.002 years) | person263 (1.542 years) |
|               500 | Windowless  | 45%       | 275 |         0.287 |               0.247 |           0.229 | person194 (0.001 years) | person348 (1.179 years) |
|               500 | Windowed    | 60%       | 200 |         0.076 |               0.064 |           0.057 | person440 (0.001 years) | person401 (0.285 years) |
|               500 | Windowless  | 60%       | 200 |         0.056 |               0.05  |           0.04  | person46 (0.000 years)  | person61 (0.210 years)  |
|               500 | Windowed    | 75%       | 125 |         0.023 |               0.02  |           0.017 | person314 (0.000 years) | person70 (0.076 years)  |
|               500 | Windowless  | 75%       | 125 |         0.019 |               0.014 |           0.015 | person131 (0.000 years) | person261 (0.078 years) |

### Removed `hasAge` Relations

Only people whose `hasAge` triple was removed from the evaluated run.

|   Population Size | Structure   | Removed   |   N |   MAE (years) |   Median AE (years) |   SD AE (years) | Best case               | Worst case               |
|------------------:|:------------|:----------|----:|--------------:|--------------------:|----------------:|:------------------------|:-------------------------|
|               100 | Windowed    | 15%       |  15 |         4.235 |               3.67  |           2.961 | person17 (0.976 years)  | person77 (12.432 years)  |
|               100 | Windowless  | 15%       |  15 |         3.597 |               2.616 |           2.888 | person78 (0.007 years)  | person68 (10.945 years)  |
|               100 | Windowed    | 30%       |  30 |         3.495 |               3.012 |           2.731 | person22 (0.342 years)  | person77 (11.051 years)  |
|               100 | Windowless  | 30%       |  30 |         6.952 |               5.614 |           4.38  | person91 (0.845 years)  | person26 (18.274 years)  |
|               100 | Windowed    | 45%       |  45 |         4.696 |               3.915 |           3.036 | person82 (0.098 years)  | person50 (11.062 years)  |
|               100 | Windowless  | 45%       |  45 |         4.077 |               2.993 |           3.33  | person63 (0.007 years)  | person76 (11.936 years)  |
|               100 | Windowed    | 60%       |  60 |         4.33  |               3.779 |           3.218 | person43 (0.096 years)  | person31 (14.334 years)  |
|               100 | Windowless  | 60%       |  60 |         4.366 |               3.789 |           3.395 | person51 (0.007 years)  | person69 (14.109 years)  |
|               100 | Windowed    | 75%       |  75 |         4.412 |               3.409 |           3.364 | person93 (0.076 years)  | person26 (13.421 years)  |
|               100 | Windowless  | 75%       |  75 |         4.716 |               3.813 |           3.677 | person82 (0.003 years)  | person69 (15.377 years)  |
|               200 | Windowed    | 15%       |  30 |         3.714 |               3.423 |           2.745 | person26 (0.118 years)  | person96 (10.247 years)  |
|               200 | Windowless  | 15%       |  30 |         3.302 |               3.171 |           2.531 | person156 (0.104 years) | person16 (8.602 years)   |
|               200 | Windowed    | 30%       |  60 |         3.717 |               2.802 |           2.923 | person32 (0.062 years)  | person98 (12.422 years)  |
|               200 | Windowless  | 30%       |  60 |         3.545 |               2.713 |           3.218 | person75 (0.019 years)  | person72 (15.432 years)  |
|               200 | Windowed    | 45%       |  90 |         3.88  |               3.519 |           3.013 | person68 (0.016 years)  | person180 (15.098 years) |
|               200 | Windowless  | 45%       |  90 |         4.738 |               4.091 |           3.256 | person30 (0.010 years)  | person72 (15.636 years)  |
|               200 | Windowed    | 60%       | 120 |         3.331 |               2.976 |           2.528 | person76 (0.037 years)  | person71 (12.520 years)  |
|               200 | Windowless  | 60%       | 120 |         3.515 |               2.844 |           2.813 | person167 (0.058 years) | person121 (11.144 years) |
|               200 | Windowed    | 75%       | 150 |         3.25  |               2.861 |           2.384 | person186 (0.028 years) | person191 (12.633 years) |
|               200 | Windowless  | 75%       | 150 |         2.946 |               2.227 |           2.341 | person90 (0.007 years)  | person177 (9.380 years)  |
|               500 | Windowed    | 15%       |  75 |         1.863 |               1.708 |           1.264 | person24 (0.060 years)  | person20 (6.566 years)   |
|               500 | Windowless  | 15%       |  75 |         2.255 |               1.711 |           1.805 | person430 (0.011 years) | person200 (7.750 years)  |
|               500 | Windowed    | 30%       | 150 |         3.132 |               2.548 |           2.318 | person223 (0.002 years) | person79 (11.595 years)  |
|               500 | Windowless  | 30%       | 150 |         2.654 |               2.264 |           2.105 | person119 (0.009 years) | person301 (8.673 years)  |
|               500 | Windowed    | 45%       | 225 |         3.145 |               2.661 |           2.387 | person200 (0.031 years) | person9 (12.554 years)   |
|               500 | Windowless  | 45%       | 225 |         3.674 |               3.056 |           2.612 | person246 (0.016 years) | person470 (12.874 years) |
|               500 | Windowed    | 60%       | 300 |         3.11  |               2.592 |           2.212 | person405 (0.077 years) | person316 (10.648 years) |
|               500 | Windowless  | 60%       | 300 |         2.793 |               2.363 |           2.042 | person196 (0.024 years) | person376 (10.128 years) |
|               500 | Windowed    | 75%       | 375 |         2.307 |               1.9   |           1.751 | person5 (0.030 years)   | person334 (8.814 years)  |
|               500 | Windowless  | 75%       | 375 |         2.38  |               1.842 |           1.896 | person77 (0.001 years)  | person139 (10.198 years) |

## Recovery-Method Comparison

`ΔMAE (Q − R)` is query-point MAE minus learned-regression MAE. Positive values favor learned regression.

### Full Population

All people in the evaluated graph.

|   Population Size | Structure   | Removed   |   N |   Query MAE (years) |   Regression MAE (years) |   ΔMAE (Q − R) (years) |   Median ΔAE (Q − R) (years) | Lower MAE          | Paired t p   | Wilcoxon p   |   Cohen's dz | 95% bootstrap CI (years)   |
|------------------:|:------------|:----------|----:|--------------------:|-------------------------:|-----------------------:|-----------------------------:|:-------------------|:-------------|:-------------|-------------:|:---------------------------|
|               100 | Windowed    | 0%        | 100 |               2.79  |                    0.209 |                  2.581 |                       -0.076 | Learned Regression | <0.0001      | 0.7129       |        0.478 | [1.592, 3.646]             |
|               100 | Windowless  | 0%        | 100 |               1.98  |                    0.248 |                  1.732 |                       -0.086 | Learned Regression | <0.0001      | 0.9015       |        0.443 | [1.028, 2.561]             |
|               100 | Windowed    | 15%       | 100 |               3.49  |                    0.765 |                  2.725 |                       -0.026 | Learned Regression | <0.0001      | 0.0077       |        0.519 | [1.751, 3.787]             |
|               100 | Windowless  | 15%       | 100 |               4.64  |                    0.683 |                  3.957 |                       -0.019 | Learned Regression | <0.0001      | 0.0018       |        0.538 | [2.581, 5.490]             |
|               100 | Windowed    | 30%       | 100 |               6.2   |                    1.107 |                  5.093 |                        1.605 | Learned Regression | <0.0001      | <0.0001      |        0.751 | [3.791, 6.393]             |
|               100 | Windowless  | 30%       | 100 |               7.71  |                    2.197 |                  5.513 |                        5.191 | Learned Regression | <0.0001      | <0.0001      |        0.788 | [4.181, 6.876]             |
|               100 | Windowed    | 45%       | 100 |               8.62  |                    2.163 |                  6.457 |                        3.275 | Learned Regression | <0.0001      | <0.0001      |        0.8   | [4.873, 8.051]             |
|               100 | Windowless  | 45%       | 100 |               8.85  |                    1.88  |                  6.97  |                        1.99  | Learned Regression | <0.0001      | <0.0001      |        0.641 | [4.969, 9.097]             |
|               100 | Windowed    | 60%       | 100 |              12.02  |                    2.62  |                  9.4   |                        7.839 | Learned Regression | <0.0001      | <0.0001      |        0.951 | [7.432, 11.296]            |
|               100 | Windowless  | 60%       | 100 |               7.71  |                    2.643 |                  5.067 |                        2.969 | Learned Regression | <0.0001      | <0.0001      |        0.677 | [3.627, 6.586]             |
|               100 | Windowed    | 75%       | 100 |              12.41  |                    3.321 |                  9.089 |                        5.306 | Learned Regression | <0.0001      | <0.0001      |        0.774 | [6.781, 11.410]            |
|               100 | Windowless  | 75%       | 100 |              13.54  |                    3.545 |                  9.995 |                        7.371 | Learned Regression | <0.0001      | <0.0001      |        0.81  | [7.576, 12.484]            |
|               200 | Windowed    | 0%        | 200 |               1.29  |                    0.471 |                  0.819 |                       -0.289 | Learned Regression | 0.0022       | <0.0001      |        0.219 | [0.328, 1.362]             |
|               200 | Windowless  | 0%        | 200 |               0.935 |                    0.46  |                  0.475 |                       -0.27  | Learned Regression | 0.0102       | <0.0001      |        0.183 | [0.133, 0.855]             |
|               200 | Windowed    | 15%       | 200 |               1.92  |                    0.835 |                  1.085 |                       -0.173 | Learned Regression | <0.0001      | 0.0124       |        0.313 | [0.608, 1.597]             |
|               200 | Windowless  | 15%       | 200 |               1.995 |                    0.728 |                  1.267 |                       -0.19  | Learned Regression | <0.0001      | 0.0256       |        0.331 | [0.756, 1.831]             |
|               200 | Windowed    | 30%       | 200 |               2.905 |                    1.313 |                  1.592 |                       -0.144 | Learned Regression | <0.0001      | 0.9854       |        0.337 | [0.973, 2.271]             |
|               200 | Windowless  | 30%       | 200 |               3.635 |                    1.239 |                  2.396 |                       -0.089 | Learned Regression | <0.0001      | 0.0003       |        0.466 | [1.705, 3.144]             |
|               200 | Windowed    | 45%       | 200 |               4.48  |                    1.834 |                  2.646 |                       -0.033 | Learned Regression | <0.0001      | <0.0001      |        0.496 | [1.924, 3.409]             |
|               200 | Windowless  | 45%       | 200 |               4.2   |                    2.203 |                  1.997 |                       -0.068 | Learned Regression | <0.0001      | 0.5788       |        0.302 | [1.115, 2.929]             |
|               200 | Windowed    | 60%       | 200 |               5.835 |                    2.022 |                  3.813 |                        0.924 | Learned Regression | <0.0001      | <0.0001      |        0.58  | [2.939, 4.735]             |
|               200 | Windowless  | 60%       | 200 |               5.76  |                    2.134 |                  3.626 |                        0.576 | Learned Regression | <0.0001      | <0.0001      |        0.584 | [2.799, 4.470]             |
|               200 | Windowed    | 75%       | 200 |               8.73  |                    2.446 |                  6.284 |                        3.938 | Learned Regression | <0.0001      | <0.0001      |        0.739 | [5.133, 7.484]             |
|               200 | Windowless  | 75%       | 200 |               7.735 |                    2.218 |                  5.517 |                        3.773 | Learned Regression | <0.0001      | <0.0001      |        0.806 | [4.571, 6.474]             |
|               500 | Windowed    | 0%        | 500 |               1.632 |                    0.929 |                  0.703 |                       -0.442 | Learned Regression | <0.0001      | 0.0029       |        0.207 | [0.418, 1.011]             |
|               500 | Windowless  | 0%        | 500 |               1.53  |                    0.865 |                  0.665 |                       -0.461 | Learned Regression | <0.0001      | <0.0001      |        0.196 | [0.378, 0.966]             |
|               500 | Windowed    | 15%       | 500 |               2.698 |                    0.871 |                  1.827 |                       -0.221 | Learned Regression | <0.0001      | 0.0096       |        0.378 | [1.416, 2.268]             |
|               500 | Windowless  | 15%       | 500 |               2.428 |                    1.043 |                  1.385 |                       -0.375 | Learned Regression | <0.0001      | 0.5514       |        0.318 | [1.001, 1.778]             |
|               500 | Windowed    | 30%       | 500 |               3.53  |                    1.355 |                  2.175 |                       -0.135 | Learned Regression | <0.0001      | <0.0001      |        0.443 | [1.758, 2.617]             |
|               500 | Windowless  | 30%       | 500 |               3.452 |                    1.223 |                  2.229 |                       -0.128 | Learned Regression | <0.0001      | <0.0001      |        0.398 | [1.758, 2.730]             |
|               500 | Windowed    | 45%       | 500 |               4.982 |                    1.603 |                  3.379 |                       -0.032 | Learned Regression | <0.0001      | <0.0001      |        0.521 | [2.811, 3.960]             |
|               500 | Windowless  | 45%       | 500 |               4.438 |                    1.811 |                  2.627 |                       -0.043 | Learned Regression | <0.0001      | <0.0001      |        0.469 | [2.155, 3.126]             |
|               500 | Windowed    | 60%       | 500 |               5.16  |                    1.896 |                  3.264 |                        1.145 | Learned Regression | <0.0001      | <0.0001      |        0.545 | [2.763, 3.796]             |
|               500 | Windowless  | 60%       | 500 |               5.082 |                    1.698 |                  3.384 |                        1.473 | Learned Regression | <0.0001      | <0.0001      |        0.623 | [2.915, 3.871]             |
|               500 | Windowed    | 75%       | 500 |               8.754 |                    1.736 |                  7.018 |                        4.967 | Learned Regression | <0.0001      | <0.0001      |        0.871 | [6.322, 7.751]             |
|               500 | Windowless  | 75%       | 500 |               7.694 |                    1.79  |                  5.904 |                        3.515 | Learned Regression | <0.0001      | <0.0001      |        0.744 | [5.210, 6.598]             |

### Retained `hasAge` Relations

Only people whose `hasAge` triple remains in the evaluated run.

|   Population Size | Structure   | Removed   |   N |   Query MAE (years) |   Regression MAE (years) |   ΔMAE (Q − R) (years) |   Median ΔAE (Q − R) (years) | Lower MAE          | Paired t p   | Wilcoxon p   |   Cohen's dz | 95% bootstrap CI (years)   |
|------------------:|:------------|:----------|----:|--------------------:|-------------------------:|-----------------------:|-----------------------------:|:-------------------|:-------------|:-------------|-------------:|:---------------------------|
|               100 | Windowed    | 0%        | 100 |               2.79  |                    0.209 |                  2.581 |                       -0.076 | Learned Regression | <0.0001      | 0.7129       |        0.478 | [1.592, 3.646]             |
|               100 | Windowless  | 0%        | 100 |               1.98  |                    0.248 |                  1.732 |                       -0.086 | Learned Regression | <0.0001      | 0.9015       |        0.443 | [1.028, 2.561]             |
|               100 | Windowed    | 15%       |  85 |               2.165 |                    0.153 |                  2.012 |                       -0.037 | Learned Regression | <0.0001      | 0.0823       |        0.544 | [1.270, 2.829]             |
|               100 | Windowless  | 15%       |  85 |               3.165 |                    0.168 |                  2.996 |                       -0.043 | Learned Regression | <0.0001      | 0.0280       |        0.593 | [2.011, 4.081]             |
|               100 | Windowed    | 30%       |  70 |               4.486 |                    0.084 |                  4.402 |                       -0.009 | Learned Regression | <0.0001      | 0.0033       |        0.673 | [2.983, 6.028]             |
|               100 | Windowless  | 30%       |  70 |               6.471 |                    0.16  |                  6.312 |                        4.361 | Learned Regression | <0.0001      | <0.0001      |        1.01  | [4.856, 7.773]             |
|               100 | Windowed    | 45%       |  55 |               6.036 |                    0.09  |                  5.947 |                        2.967 | Learned Regression | <0.0001      | <0.0001      |        0.803 | [4.092, 8.020]             |
|               100 | Windowless  | 45%       |  55 |               5.418 |                    0.083 |                  5.335 |                       -0.009 | Learned Regression | <0.0001      | 0.0050       |        0.663 | [3.330, 7.557]             |
|               100 | Windowed    | 60%       |  40 |               7.925 |                    0.054 |                  7.871 |                        3.953 | Learned Regression | <0.0001      | <0.0001      |        0.911 | [5.346, 10.636]            |
|               100 | Windowless  | 60%       |  40 |               4.125 |                    0.059 |                  4.066 |                        1.378 | Learned Regression | <0.0001      | 0.0009       |        0.713 | [2.493, 5.937]             |
|               100 | Windowed    | 75%       |  25 |               5.12  |                    0.049 |                  5.071 |                        2.96  | Learned Regression | 0.0003       | 0.0009       |        0.849 | [2.870, 7.541]             |
|               100 | Windowless  | 75%       |  25 |               5.56  |                    0.034 |                  5.526 |                        1.999 | Learned Regression | 0.0007       | 0.0219       |        0.774 | [2.883, 8.528]             |
|               200 | Windowed    | 0%        | 200 |               1.29  |                    0.471 |                  0.819 |                       -0.289 | Learned Regression | 0.0022       | <0.0001      |        0.219 | [0.328, 1.362]             |
|               200 | Windowless  | 0%        | 200 |               0.935 |                    0.46  |                  0.475 |                       -0.27  | Learned Regression | 0.0102       | <0.0001      |        0.183 | [0.133, 0.855]             |
|               200 | Windowed    | 15%       | 170 |               0.924 |                    0.327 |                  0.597 |                       -0.211 | Learned Regression | 0.0050       | <0.0001      |        0.218 | [0.224, 1.041]             |
|               200 | Windowless  | 15%       | 170 |               1.035 |                    0.274 |                  0.762 |                       -0.197 | Learned Regression | 0.0009       | <0.0001      |        0.259 | [0.345, 1.215]             |
|               200 | Windowed    | 30%       | 140 |               1.271 |                    0.282 |                  0.989 |                       -0.168 | Learned Regression | 0.0010       | <0.0001      |        0.285 | [0.437, 1.616]             |
|               200 | Windowless  | 30%       | 140 |               1.65  |                    0.251 |                  1.399 |                       -0.135 | Learned Regression | <0.0001      | 0.7916       |        0.388 | [0.832, 2.024]             |
|               200 | Windowed    | 45%       | 110 |               1.827 |                    0.16  |                  1.667 |                       -0.089 | Learned Regression | <0.0001      | 0.4233       |        0.451 | [1.014, 2.391]             |
|               200 | Windowless  | 45%       | 110 |               1.409 |                    0.128 |                  1.281 |                       -0.088 | Learned Regression | 0.0028       | 0.0003       |        0.292 | [0.533, 2.129]             |
|               200 | Windowed    | 60%       |  80 |               2.125 |                    0.057 |                  2.068 |                       -0.025 | Learned Regression | 0.0002       | 0.8742       |        0.438 | [1.122, 3.173]             |
|               200 | Windowless  | 60%       |  80 |               1.887 |                    0.063 |                  1.824 |                       -0.038 | Learned Regression | 0.0002       | 0.4719       |        0.443 | [0.949, 2.778]             |
|               200 | Windowed    | 75%       |  50 |               3.84  |                    0.031 |                  3.809 |                        0.469 | Learned Regression | 0.0001       | 0.0021       |        0.588 | [2.185, 5.593]             |
|               200 | Windowless  | 75%       |  50 |               3.26  |                    0.034 |                  3.226 |                       -0.004 | Learned Regression | <0.0001      | 0.0506       |        0.631 | [1.849, 4.726]             |
|               500 | Windowed    | 0%        | 500 |               1.632 |                    0.929 |                  0.703 |                       -0.442 | Learned Regression | <0.0001      | 0.0029       |        0.207 | [0.418, 1.011]             |
|               500 | Windowless  | 0%        | 500 |               1.53  |                    0.865 |                  0.665 |                       -0.461 | Learned Regression | <0.0001      | <0.0001      |        0.196 | [0.378, 0.966]             |
|               500 | Windowed    | 15%       | 425 |               2.228 |                    0.696 |                  1.532 |                       -0.279 | Learned Regression | <0.0001      | 0.7819       |        0.331 | [1.105, 1.983]             |
|               500 | Windowless  | 15%       | 425 |               1.779 |                    0.829 |                  0.95  |                       -0.468 | Learned Regression | <0.0001      | 0.0004       |        0.241 | [0.593, 1.333]             |
|               500 | Windowed    | 30%       | 350 |               2.154 |                    0.594 |                  1.561 |                       -0.258 | Learned Regression | <0.0001      | 0.5627       |        0.379 | [1.122, 2.013]             |
|               500 | Windowless  | 30%       | 350 |               2.263 |                    0.61  |                  1.653 |                       -0.221 | Learned Regression | <0.0001      | 0.7002       |        0.314 | [1.120, 2.236]             |
|               500 | Windowed    | 45%       | 275 |               2.56  |                    0.341 |                  2.219 |                       -0.149 | Learned Regression | <0.0001      | 0.7377       |        0.433 | [1.645, 2.832]             |
|               500 | Windowless  | 45%       | 275 |               2.418 |                    0.287 |                  2.131 |                       -0.118 | Learned Regression | <0.0001      | 0.3001       |        0.458 | [1.578, 2.694]             |
|               500 | Windowed    | 60%       | 200 |               2.725 |                    0.076 |                  2.649 |                       -0.014 | Learned Regression | <0.0001      | 0.0002       |        0.557 | [2.022, 3.362]             |
|               500 | Windowless  | 60%       | 200 |               2.605 |                    0.056 |                  2.549 |                       -0.022 | Learned Regression | <0.0001      | 0.0867       |        0.526 | [1.890, 3.214]             |
|               500 | Windowed    | 75%       | 125 |               4.616 |                    0.023 |                  4.593 |                       -0.004 | Learned Regression | <0.0001      | 0.0003       |        0.679 | [3.464, 5.778]             |
|               500 | Windowless  | 75%       | 125 |               3.624 |                    0.019 |                  3.605 |                       -0.004 | Learned Regression | <0.0001      | 0.0007       |        0.588 | [2.572, 4.725]             |

### Removed `hasAge` Relations

Only people whose `hasAge` triple was removed from the evaluated run.

|   Population Size | Structure   | Removed   |   N |   Query MAE (years) |   Regression MAE (years) |   ΔMAE (Q − R) (years) |   Median ΔAE (Q − R) (years) | Lower MAE          | Paired t p   | Wilcoxon p   |   Cohen's dz | 95% bootstrap CI (years)   |
|------------------:|:------------|:----------|----:|--------------------:|-------------------------:|-----------------------:|-----------------------------:|:-------------------|:-------------|:-------------|-------------:|:---------------------------|
|               100 | Windowed    | 15%       |  15 |              11     |                    4.235 |                  6.765 |                        4.981 | Learned Regression | 0.0163       | 0.0125       |        0.705 | [2.325, 11.610]            |
|               100 | Windowless  | 15%       |  15 |              13     |                    3.597 |                  9.403 |                        5.244 | Learned Regression | 0.0199       | 0.0302       |        0.678 | [2.718, 16.247]            |
|               100 | Windowed    | 30%       |  30 |              10.2   |                    3.495 |                  6.705 |                        6.702 | Learned Regression | <0.0001      | 0.0001       |        0.937 | [4.112, 9.227]             |
|               100 | Windowless  | 30%       |  30 |              10.6   |                    6.952 |                  3.648 |                        5.581 | Learned Regression | 0.0228       | 0.0248       |        0.439 | [0.665, 6.662]             |
|               100 | Windowed    | 45%       |  45 |              11.778 |                    4.696 |                  7.082 |                        5.938 | Learned Regression | <0.0001      | <0.0001      |        0.799 | [4.549, 9.646]             |
|               100 | Windowless  | 45%       |  45 |              13.044 |                    4.077 |                  8.967 |                        4.408 | Learned Regression | <0.0001      | <0.0001      |        0.671 | [5.184, 13.050]            |
|               100 | Windowed    | 60%       |  60 |              14.75  |                    4.33  |                 10.42  |                        8.573 | Learned Regression | <0.0001      | <0.0001      |        0.984 | [7.751, 13.109]            |
|               100 | Windowless  | 60%       |  60 |              10.1   |                    4.366 |                  5.734 |                        7.113 | Learned Regression | <0.0001      | <0.0001      |        0.679 | [3.563, 7.909]             |
|               100 | Windowed    | 75%       |  75 |              14.84  |                    4.412 |                 10.428 |                        6.756 | Learned Regression | <0.0001      | <0.0001      |        0.81  | [7.704, 13.318]            |
|               100 | Windowless  | 75%       |  75 |              16.2   |                    4.716 |                 11.484 |                        8.415 | Learned Regression | <0.0001      | <0.0001      |        0.86  | [8.523, 14.489]            |
|               200 | Windowed    | 15%       |  30 |               7.567 |                    3.714 |                  3.853 |                        2.481 | Learned Regression | 0.0005       | 0.0007       |        0.71  | [1.993, 5.828]             |
|               200 | Windowless  | 15%       |  30 |               7.433 |                    3.302 |                  4.131 |                        3.459 | Learned Regression | 0.0013       | 0.0054       |        0.649 | [1.887, 6.362]             |
|               200 | Windowed    | 30%       |  60 |               6.717 |                    3.717 |                  3     |                        2.551 | Learned Regression | 0.0009       | 0.0011       |        0.452 | [1.456, 4.726]             |
|               200 | Windowless  | 30%       |  60 |               8.267 |                    3.545 |                  4.721 |                        3.803 | Learned Regression | <0.0001      | <0.0001      |        0.663 | [2.897, 6.551]             |
|               200 | Windowed    | 45%       |  90 |               7.722 |                    3.88  |                  3.843 |                        3.565 | Learned Regression | <0.0001      | <0.0001      |        0.578 | [2.540, 5.264]             |
|               200 | Windowless  | 45%       |  90 |               7.611 |                    4.738 |                  2.873 |                        1.254 | Learned Regression | 0.0019       | 0.0229       |        0.337 | [1.172, 4.699]             |
|               200 | Windowed    | 60%       | 120 |               8.308 |                    3.331 |                  4.977 |                        3.683 | Learned Regression | <0.0001      | <0.0001      |        0.677 | [3.708, 6.326]             |
|               200 | Windowless  | 60%       | 120 |               8.342 |                    3.515 |                  4.827 |                        3.847 | Learned Regression | <0.0001      | <0.0001      |        0.686 | [3.566, 6.096]             |
|               200 | Windowed    | 75%       | 150 |              10.36  |                    3.25  |                  7.11  |                        4.877 | Learned Regression | <0.0001      | <0.0001      |        0.795 | [5.735, 8.528]             |
|               200 | Windowless  | 75%       | 150 |               9.227 |                    2.946 |                  6.281 |                        5.157 | Learned Regression | <0.0001      | <0.0001      |        0.874 | [5.182, 7.469]             |
|               500 | Windowed    | 15%       |  75 |               5.36  |                    1.863 |                  3.497 |                        1.639 | Learned Regression | <0.0001      | <0.0001      |        0.624 | [2.291, 4.827]             |
|               500 | Windowless  | 15%       |  75 |               6.107 |                    2.255 |                  3.852 |                        2.526 | Learned Regression | <0.0001      | <0.0001      |        0.685 | [2.653, 5.119]             |
|               500 | Windowed    | 30%       | 150 |               6.74  |                    3.132 |                  3.608 |                        2.188 | Learned Regression | <0.0001      | <0.0001      |        0.584 | [2.649, 4.622]             |
|               500 | Windowless  | 30%       | 150 |               6.227 |                    2.654 |                  3.572 |                        2.339 | Learned Regression | <0.0001      | <0.0001      |        0.58  | [2.584, 4.534]             |
|               500 | Windowed    | 45%       | 225 |               7.942 |                    3.145 |                  4.797 |                        3.039 | Learned Regression | <0.0001      | <0.0001      |        0.629 | [3.839, 5.786]             |
|               500 | Windowless  | 45%       | 225 |               6.907 |                    3.674 |                  3.233 |                        2.332 | Learned Regression | <0.0001      | <0.0001      |        0.495 | [2.387, 4.113]             |
|               500 | Windowed    | 60%       | 300 |               6.783 |                    3.11  |                  3.674 |                        2.528 | Learned Regression | <0.0001      | <0.0001      |        0.551 | [2.922, 4.394]             |
|               500 | Windowless  | 60%       | 300 |               6.733 |                    2.793 |                  3.941 |                        3.261 | Learned Regression | <0.0001      | <0.0001      |        0.688 | [3.284, 4.597]             |
|               500 | Windowed    | 75%       | 375 |              10.133 |                    2.307 |                  7.827 |                        5.845 | Learned Regression | <0.0001      | <0.0001      |        0.943 | [7.020, 8.676]             |
|               500 | Windowless  | 75%       | 375 |               9.051 |                    2.38  |                  6.67  |                        4.761 | Learned Regression | <0.0001      | <0.0001      |        0.802 | [5.884, 7.535]             |

## Effect of Window Structure

`ΔMAE (W − WL)` is windowed MAE minus windowless MAE. Positive values favor the windowless graph.

### Query-Point Recovery

#### Full Population

All people in the evaluated graph.

|   Population Size | Removed   |   N |   Windowed MAE (years) |   Windowless MAE (years) |   ΔMAE (W − WL) (years) |   Median ΔAE (W − WL) (years) | Lower MAE   | Paired t p   |   Wilcoxon p |   Cohen's dz | 95% bootstrap CI (years)   |
|------------------:|:----------|----:|-----------------------:|-------------------------:|------------------------:|------------------------------:|:------------|:-------------|-------------:|-------------:|:---------------------------|
|               100 | 0%        | 100 |                  2.79  |                    1.98  |                   0.81  |                             0 | Windowless  | 0.1670       |       0.2499 |        0.139 | [-0.300, 1.940]            |
|               100 | 15%       | 100 |                  3.49  |                    4.64  |                  -1.15  |                             0 | Windowed    | 0.1776       |       0.13   |       -0.136 | [-2.820, 0.500]            |
|               100 | 30%       | 100 |                  6.2   |                    7.71  |                  -1.51  |                             0 | Windowed    | 0.1413       |       0.1483 |       -0.148 | [-3.500, 0.450]            |
|               100 | 45%       | 100 |                  8.62  |                    8.85  |                  -0.23  |                             0 | Windowed    | 0.8310       |       0.7041 |       -0.021 | [-2.350, 1.850]            |
|               100 | 60%       | 100 |                 12.02  |                    7.71  |                   4.31  |                             4 | Windowless  | <0.0001      |       0.0001 |        0.418 | [2.290, 6.320]             |
|               100 | 75%       | 100 |                 12.41  |                   13.54  |                  -1.13  |                             0 | Windowed    | 0.2686       |       0.9727 |       -0.111 | [-3.250, 0.750]            |
|               200 | 0%        | 200 |                  1.29  |                    0.935 |                   0.355 |                             0 | Windowless  | 0.2426       |       0.5103 |        0.083 | [-0.220, 0.970]            |
|               200 | 15%       | 200 |                  1.92  |                    1.995 |                  -0.075 |                             0 | Windowed    | 0.8120       |       0.7705 |       -0.017 | [-0.705, 0.540]            |
|               200 | 30%       | 200 |                  2.905 |                    3.635 |                  -0.73  |                             0 | Windowed    | 0.1011       |       0.061  |       -0.116 | [-1.605, 0.130]            |
|               200 | 45%       | 200 |                  4.48  |                    4.2   |                   0.28  |                             0 | Windowless  | 0.6039       |       0.4251 |        0.037 | [-0.770, 1.325]            |
|               200 | 60%       | 200 |                  5.835 |                    5.76  |                   0.075 |                             0 | Windowless  | 0.8897       |       0.9809 |        0.01  | [-0.975, 1.120]            |
|               200 | 75%       | 200 |                  8.73  |                    7.735 |                   0.995 |                             0 | Windowless  | 0.1364       |       0.2334 |        0.106 | [-0.295, 2.320]            |
|               500 | 0%        | 500 |                  1.632 |                    1.53  |                   0.102 |                             0 | Windowless  | 0.6304       |       0.6599 |        0.022 | [-0.304, 0.520]            |
|               500 | 15%       | 500 |                  2.698 |                    2.428 |                   0.27  |                             0 | Windowless  | 0.3292       |       0.5543 |        0.044 | [-0.282, 0.820]            |
|               500 | 30%       | 500 |                  3.53  |                    3.452 |                   0.078 |                             0 | Windowless  | 0.8106       |       0.2692 |        0.011 | [-0.580, 0.710]            |
|               500 | 45%       | 500 |                  4.982 |                    4.438 |                   0.544 |                             0 | Windowless  | 0.1164       |       0.259  |        0.07  | [-0.130, 1.214]            |
|               500 | 60%       | 500 |                  5.16  |                    5.082 |                   0.078 |                             0 | Windowless  | 0.8159       |       0.9449 |        0.01  | [-0.574, 0.734]            |
|               500 | 75%       | 500 |                  8.754 |                    7.694 |                   1.06  |                             0 | Windowless  | 0.0156       |       0.047  |        0.108 | [0.232, 1.940]             |

#### Retained `hasAge` Relations

Only people whose `hasAge` triple remains in the evaluated run.

|   Population Size | Removed   |   N |   Windowed MAE (years) |   Windowless MAE (years) |   ΔMAE (W − WL) (years) |   Median ΔAE (W − WL) (years) | Lower MAE   |   Paired t p |   Wilcoxon p |   Cohen's dz | 95% bootstrap CI (years)   |
|------------------:|:----------|----:|-----------------------:|-------------------------:|------------------------:|------------------------------:|:------------|-------------:|-------------:|-------------:|:---------------------------|
|               100 | 0%        | 100 |                  2.79  |                    1.98  |                   0.81  |                             0 | Windowless  |       0.167  |       0.2499 |        0.139 | [-0.300, 1.940]            |
|               100 | 15%       |  85 |                  2.165 |                    3.165 |                  -1     |                             0 | Windowed    |       0.1462 |       0.1485 |       -0.159 | [-2.318, 0.294]            |
|               100 | 30%       |  70 |                  4.486 |                    6.471 |                  -1.986 |                             0 | Windowed    |       0.095  |       0.0754 |       -0.202 | [-4.229, 0.371]            |
|               100 | 45%       |  55 |                  6.036 |                    5.418 |                   0.618 |                             0 | Windowless  |       0.6303 |       0.8362 |        0.065 | [-1.836, 3.164]            |
|               100 | 60%       |  40 |                  7.925 |                    4.125 |                   3.8   |                             2 | Windowless  |       0.0218 |       0.0291 |        0.378 | [0.775, 6.975]             |
|               100 | 75%       |  25 |                  5.12  |                    5.56  |                  -0.44  |                             0 | Windowed    |       0.8055 |       0.9303 |       -0.05  | [-4.160, 2.920]            |
|               200 | 0%        | 200 |                  1.29  |                    0.935 |                   0.355 |                             0 | Windowless  |       0.2426 |       0.5103 |        0.083 | [-0.220, 0.970]            |
|               200 | 15%       | 170 |                  0.924 |                    1.035 |                  -0.112 |                             0 | Windowed    |       0.7171 |       0.7303 |       -0.028 | [-0.700, 0.488]            |
|               200 | 30%       | 140 |                  1.271 |                    1.65  |                  -0.379 |                             0 | Windowed    |       0.3399 |       0.2541 |       -0.081 | [-1.164, 0.407]            |
|               200 | 45%       | 110 |                  1.827 |                    1.409 |                   0.418 |                             0 | Windowless  |       0.445  |       0.2648 |        0.073 | [-0.636, 1.473]            |
|               200 | 60%       |  80 |                  2.125 |                    1.887 |                   0.237 |                             0 | Windowless  |       0.7174 |       0.8188 |        0.041 | [-1.075, 1.550]            |
|               200 | 75%       |  50 |                  3.84  |                    3.26  |                   0.58  |                             0 | Windowless  |       0.575  |       0.3932 |        0.08  | [-1.400, 2.600]            |
|               500 | 0%        | 500 |                  1.632 |                    1.53  |                   0.102 |                             0 | Windowless  |       0.6304 |       0.6599 |        0.022 | [-0.304, 0.520]            |
|               500 | 15%       | 425 |                  2.228 |                    1.779 |                   0.449 |                             0 | Windowless  |       0.1234 |       0.1834 |        0.075 | [-0.104, 1.000]            |
|               500 | 30%       | 350 |                  2.154 |                    2.263 |                  -0.109 |                             0 | Windowed    |       0.7709 |       0.6844 |       -0.016 | [-0.886, 0.640]            |
|               500 | 45%       | 275 |                  2.56  |                    2.418 |                   0.142 |                             0 | Windowless  |       0.7389 |       0.9314 |        0.02  | [-0.673, 0.993]            |
|               500 | 60%       | 200 |                  2.725 |                    2.605 |                   0.12  |                             0 | Windowless  |       0.8084 |       0.6852 |        0.017 | [-0.855, 1.090]            |
|               500 | 75%       | 125 |                  4.616 |                    3.624 |                   0.992 |                             0 | Windowless  |       0.207  |       0.3269 |        0.113 | [-0.536, 2.560]            |

#### Removed `hasAge` Relations

Only people whose `hasAge` triple was removed from the evaluated run.

|   Population Size | Removed   |   N |   Windowed MAE (years) |   Windowless MAE (years) |   ΔMAE (W − WL) (years) |   Median ΔAE (W − WL) (years) | Lower MAE   |   Paired t p |   Wilcoxon p |   Cohen's dz | 95% bootstrap CI (years)   |
|------------------:|:----------|----:|-----------------------:|-------------------------:|------------------------:|------------------------------:|:------------|-------------:|-------------:|-------------:|:---------------------------|
|               100 | 15%       |  15 |                 11     |                   13     |                  -2     |                           0   | Windowed    |       0.6441 |       0.5287 |       -0.122 | [-9.935, 6.000]            |
|               100 | 30%       |  30 |                 10.2   |                   10.6   |                  -0.4   |                          -1   | Windowed    |       0.8447 |       0.9364 |       -0.036 | [-4.333, 3.467]            |
|               100 | 45%       |  45 |                 11.778 |                   13.044 |                  -1.267 |                          -1   | Windowed    |       0.4885 |       0.4677 |       -0.104 | [-4.778, 2.222]            |
|               100 | 60%       |  60 |                 14.75  |                   10.1   |                   4.65  |                           6   | Windowless  |       0.0011 |       0.0014 |        0.441 | [2.050, 7.367]             |
|               100 | 75%       |  75 |                 14.84  |                   16.2   |                  -1.36  |                           0   | Windowed    |       0.2704 |       0.9738 |       -0.128 | [-3.840, 0.880]            |
|               200 | 15%       |  30 |                  7.567 |                    7.433 |                   0.133 |                           0.5 | Windowless  |       0.9112 |       0.9904 |        0.021 | [-2.133, 2.434]            |
|               200 | 30%       |  60 |                  6.717 |                    8.267 |                  -1.55  |                           0   | Windowed    |       0.1845 |       0.1647 |       -0.173 | [-3.850, 0.717]            |
|               200 | 45%       |  90 |                  7.722 |                    7.611 |                   0.111 |                           0   | Windowless  |       0.9116 |       0.8249 |        0.012 | [-1.811, 2.044]            |
|               200 | 60%       | 120 |                  8.308 |                    8.342 |                  -0.033 |                           0   | Windowed    |       0.9664 |       0.9068 |       -0.004 | [-1.558, 1.567]            |
|               200 | 75%       | 150 |                 10.36  |                    9.227 |                   1.133 |                           0   | Windowless  |       0.1689 |       0.3165 |        0.113 | [-0.447, 2.733]            |
|               500 | 15%       |  75 |                  5.36  |                    6.107 |                  -0.747 |                          -1   | Windowed    |       0.3638 |       0.2695 |       -0.106 | [-2.347, 0.840]            |
|               500 | 30%       | 150 |                  6.74  |                    6.227 |                   0.513 |                           1   | Windowless  |       0.4305 |       0.1882 |        0.065 | [-0.747, 1.787]            |
|               500 | 45%       | 225 |                  7.942 |                    6.907 |                   1.036 |                           0   | Windowless  |       0.0685 |       0.1249 |        0.122 | [-0.044, 2.151]            |
|               500 | 60%       | 300 |                  6.783 |                    6.733 |                   0.05  |                           0   | Windowless  |       0.9119 |       0.808  |        0.006 | [-0.813, 0.910]            |
|               500 | 75%       | 375 |                 10.133 |                    9.051 |                   1.083 |                           0   | Windowless  |       0.0386 |       0.0862 |        0.107 | [0.083, 2.107]             |

### Learned-Regression Recovery

#### Full Population

All people in the evaluated graph.

|   Population Size | Removed   |   N |   Windowed MAE (years) |   Windowless MAE (years) |   ΔMAE (W − WL) (years) |   Median ΔAE (W − WL) (years) | Lower MAE   |   Paired t p | Wilcoxon p   |   Cohen's dz | 95% bootstrap CI (years)   |
|------------------:|:----------|----:|-----------------------:|-------------------------:|------------------------:|------------------------------:|:------------|-------------:|:-------------|-------------:|:---------------------------|
|               100 | 0%        | 100 |                  0.209 |                    0.248 |                  -0.039 |                        -0.046 | Windowed    |       0.2889 | 0.0136       |       -0.107 | [-0.101, 0.038]            |
|               100 | 15%       | 100 |                  0.765 |                    0.683 |                   0.082 |                         0.017 | Windowless  |       0.628  | 0.5752       |        0.049 | [-0.249, 0.416]            |
|               100 | 30%       | 100 |                  1.107 |                    2.197 |                  -1.09  |                        -0.101 | Windowed    |       0.0013 | <0.0001      |       -0.332 | [-1.734, -0.473]           |
|               100 | 45%       | 100 |                  2.163 |                    1.88  |                   0.283 |                         0.034 | Windowless  |       0.3356 | 0.0591       |        0.097 | [-0.299, 0.856]            |
|               100 | 60%       | 100 |                  2.62  |                    2.643 |                  -0.023 |                         0.016 | Windowed    |       0.9458 | 0.5799       |       -0.007 | [-0.707, 0.641]            |
|               100 | 75%       | 100 |                  3.321 |                    3.545 |                  -0.224 |                         0.018 | Windowed    |       0.6088 | 0.9260       |       -0.051 | [-1.087, 0.608]            |
|               200 | 0%        | 200 |                  0.471 |                    0.46  |                   0.011 |                         0.008 | Windowless  |       0.7741 | 0.7978       |        0.02  | [-0.066, 0.087]            |
|               200 | 15%       | 200 |                  0.835 |                    0.728 |                   0.107 |                         0.051 | Windowless  |       0.3668 | 0.0201       |        0.064 | [-0.107, 0.345]            |
|               200 | 30%       | 200 |                  1.313 |                    1.239 |                   0.074 |                        -0.009 | Windowless  |       0.5843 | 0.9951       |        0.039 | [-0.190, 0.340]            |
|               200 | 45%       | 200 |                  1.834 |                    2.203 |                  -0.369 |                        -0.003 | Windowed    |       0.074  | 0.4537       |       -0.127 | [-0.786, 0.024]            |
|               200 | 60%       | 200 |                  2.022 |                    2.134 |                  -0.113 |                        -0.001 | Windowed    |       0.5976 | 0.8290       |       -0.037 | [-0.539, 0.297]            |
|               200 | 75%       | 200 |                  2.446 |                    2.218 |                   0.228 |                         0.01  | Windowless  |       0.2596 | 0.3019       |        0.08  | [-0.175, 0.632]            |
|               500 | 0%        | 500 |                  0.929 |                    0.865 |                   0.064 |                         0.037 | Windowless  |       0.1446 | 0.1437       |        0.065 | [-0.020, 0.146]            |
|               500 | 15%       | 500 |                  0.871 |                    1.043 |                  -0.172 |                        -0.093 | Windowed    |       0.0008 | 0.0060       |       -0.15  | [-0.269, -0.071]           |
|               500 | 30%       | 500 |                  1.355 |                    1.223 |                   0.132 |                        -0.002 | Windowless  |       0.1083 | 0.3952       |        0.072 | [-0.028, 0.294]            |
|               500 | 45%       | 500 |                  1.603 |                    1.811 |                  -0.209 |                         0.028 | Windowed    |       0.0558 | 0.8947       |       -0.086 | [-0.429, 0.000]            |
|               500 | 60%       | 500 |                  1.896 |                    1.698 |                   0.198 |                         0.022 | Windowless  |       0.0519 | 0.0106       |        0.087 | [-0.008, 0.398]            |
|               500 | 75%       | 500 |                  1.736 |                    1.79  |                  -0.054 |                        -0.005 | Windowed    |       0.5746 | 0.4619       |       -0.025 | [-0.242, 0.140]            |

#### Retained `hasAge` Relations

Only people whose `hasAge` triple remains in the evaluated run.

|   Population Size | Removed   |   N |   Windowed MAE (years) |   Windowless MAE (years) |   ΔMAE (W − WL) (years) |   Median ΔAE (W − WL) (years) | Lower MAE   | Paired t p   | Wilcoxon p   |   Cohen's dz | 95% bootstrap CI (years)   |
|------------------:|:----------|----:|-----------------------:|-------------------------:|------------------------:|------------------------------:|:------------|:-------------|:-------------|-------------:|:---------------------------|
|               100 | 0%        | 100 |                  0.209 |                    0.248 |                  -0.039 |                        -0.046 | Windowed    | 0.2889       | 0.0136       |       -0.107 | [-0.101, 0.038]            |
|               100 | 15%       |  85 |                  0.153 |                    0.168 |                  -0.016 |                         0.007 | Windowed    | 0.3896       | 0.8079       |       -0.094 | [-0.052, 0.018]            |
|               100 | 30%       |  70 |                  0.084 |                    0.16  |                  -0.076 |                        -0.057 | Windowed    | <0.0001      | <0.0001      |       -0.543 | [-0.109, -0.045]           |
|               100 | 45%       |  55 |                  0.09  |                    0.083 |                   0.007 |                         0.01  | Windowless  | 0.5860       | 0.5027       |        0.074 | [-0.019, 0.031]            |
|               100 | 60%       |  40 |                  0.054 |                    0.059 |                  -0.005 |                         0.006 | Windowed    | 0.6495       | 0.8160       |       -0.072 | [-0.028, 0.016]            |
|               100 | 75%       |  25 |                  0.049 |                    0.034 |                   0.015 |                         0.017 | Windowless  | 0.0875       | 0.0851       |        0.356 | [-0.001, 0.032]            |
|               200 | 0%        | 200 |                  0.471 |                    0.46  |                   0.011 |                         0.008 | Windowless  | 0.7741       | 0.7978       |        0.02  | [-0.066, 0.087]            |
|               200 | 15%       | 170 |                  0.327 |                    0.274 |                   0.053 |                         0.025 | Windowless  | 0.0333       | 0.0513       |        0.165 | [0.006, 0.101]             |
|               200 | 30%       | 140 |                  0.282 |                    0.251 |                   0.032 |                         0.003 | Windowless  | 0.3147       | 0.7981       |        0.085 | [-0.023, 0.098]            |
|               200 | 45%       | 110 |                  0.16  |                    0.128 |                   0.032 |                         0.018 | Windowless  | 0.0470       | 0.0911       |        0.192 | [0.001, 0.063]             |
|               200 | 60%       |  80 |                  0.057 |                    0.063 |                  -0.006 |                        -0.001 | Windowed    | 0.3656       | 0.4013       |       -0.102 | [-0.020, 0.007]            |
|               200 | 75%       |  50 |                  0.031 |                    0.034 |                  -0.003 |                         0     | Windowed    | 0.5402       | 0.7161       |       -0.087 | [-0.012, 0.006]            |
|               500 | 0%        | 500 |                  0.929 |                    0.865 |                   0.064 |                         0.037 | Windowless  | 0.1446       | 0.1437       |        0.065 | [-0.020, 0.146]            |
|               500 | 15%       | 425 |                  0.696 |                    0.829 |                  -0.133 |                        -0.097 | Windowed    | 0.0007       | 0.0035       |       -0.165 | [-0.209, -0.057]           |
|               500 | 30%       | 350 |                  0.594 |                    0.61  |                  -0.016 |                        -0.045 | Windowed    | 0.6598       | 0.4011       |       -0.024 | [-0.087, 0.055]            |
|               500 | 45%       | 275 |                  0.341 |                    0.287 |                   0.053 |                         0.043 | Windowless  | 0.0107       | 0.0228       |        0.155 | [0.013, 0.095]             |
|               500 | 60%       | 200 |                  0.076 |                    0.056 |                   0.02  |                         0.01  | Windowless  | <0.0001      | 0.0003       |        0.291 | [0.010, 0.029]             |
|               500 | 75%       | 125 |                  0.023 |                    0.019 |                   0.004 |                         0.001 | Windowless  | 0.0227       | 0.0677       |        0.206 | [0.001, 0.008]             |

#### Removed `hasAge` Relations

Only people whose `hasAge` triple was removed from the evaluated run.

|   Population Size | Removed   |   N |   Windowed MAE (years) |   Windowless MAE (years) |   ΔMAE (W − WL) (years) |   Median ΔAE (W − WL) (years) | Lower MAE   |   Paired t p |   Wilcoxon p |   Cohen's dz | 95% bootstrap CI (years)   |
|------------------:|:----------|----:|-----------------------:|-------------------------:|------------------------:|------------------------------:|:------------|-------------:|-------------:|-------------:|:---------------------------|
|               100 | 15%       |  15 |                  4.235 |                    3.597 |                   0.638 |                         1.259 | Windowless  |       0.5868 |       0.4212 |        0.144 | [-1.638, 2.688]            |
|               100 | 30%       |  30 |                  3.495 |                    6.952 |                  -3.457 |                        -2.171 | Windowed    |       0.0014 |       0.0008 |       -0.647 | [-5.387, -1.631]           |
|               100 | 45%       |  45 |                  4.696 |                    4.077 |                   0.619 |                         1.741 | Windowless  |       0.3453 |       0.1563 |        0.142 | [-0.678, 1.850]            |
|               100 | 60%       |  60 |                  4.33  |                    4.366 |                  -0.035 |                         0.769 | Windowed    |       0.9508 |       0.8024 |       -0.008 | [-1.152, 1.050]            |
|               100 | 75%       |  75 |                  4.412 |                    4.716 |                  -0.304 |                         0.67  | Windowed    |       0.6035 |       0.7634 |       -0.06  | [-1.467, 0.826]            |
|               200 | 15%       |  30 |                  3.714 |                    3.302 |                   0.412 |                         0.573 | Windowless  |       0.6034 |       0.4645 |        0.096 | [-1.088, 1.951]            |
|               200 | 30%       |  60 |                  3.717 |                    3.545 |                   0.171 |                        -0.175 | Windowless  |       0.7005 |       0.8598 |        0.05  | [-0.677, 1.021]            |
|               200 | 45%       |  90 |                  3.88  |                    4.738 |                  -0.858 |                        -0.51  | Windowed    |       0.0607 |       0.1019 |       -0.2   | [-1.727, 0.015]            |
|               200 | 60%       | 120 |                  3.331 |                    3.515 |                  -0.184 |                         0.073 | Windowed    |       0.6067 |       0.8238 |       -0.047 | [-0.862, 0.520]            |
|               200 | 75%       | 150 |                  3.25  |                    2.946 |                   0.305 |                         0.285 | Windowless  |       0.2584 |       0.2474 |        0.093 | [-0.218, 0.851]            |
|               500 | 15%       |  75 |                  1.863 |                    2.255 |                  -0.392 |                         0.031 | Windowed    |       0.1334 |       0.2475 |       -0.175 | [-0.917, 0.119]            |
|               500 | 30%       | 150 |                  3.132 |                    2.654 |                   0.477 |                         0.515 | Windowless  |       0.0668 |       0.0692 |        0.151 | [-0.026, 0.989]            |
|               500 | 45%       | 225 |                  3.145 |                    3.674 |                  -0.529 |                        -0.246 | Windowed    |       0.028  |       0.0476 |       -0.147 | [-0.996, -0.048]           |
|               500 | 60%       | 300 |                  3.11  |                    2.793 |                   0.317 |                         0.224 | Windowless  |       0.062  |       0.0836 |        0.108 | [-0.007, 0.646]            |
|               500 | 75%       | 375 |                  2.307 |                    2.38  |                  -0.074 |                        -0.126 | Windowed    |       0.5673 |       0.536  |       -0.03  | [-0.325, 0.180]            |

## Recovered-Age Monotonicity

Spearman’s ρ and Kendall’s τb compare ground-truth age order with recovered scalar order. They measure ordering, not error in years.

### Query-Point Ordering

#### Full Population

All people in the evaluated graph.

|   Population Size | Structure   | Removed   |   N |   Unique Ages | True-Age Range (years)   |   Spearman ρ | Spearman p   |   Kendall τb | Kendall p   |
|------------------:|:------------|:----------|----:|--------------:|:-------------------------|-------------:|:-------------|-------------:|:------------|
|               100 | Windowed    | 0%        | 100 |            68 | 2–99                     |        0.98  | <0.0001      |        0.927 | <0.0001     |
|               100 | Windowless  | 0%        | 100 |            68 | 2–99                     |        0.987 | <0.0001      |        0.938 | <0.0001     |
|               100 | Windowed    | 15%       | 100 |            68 | 2–99                     |        0.973 | <0.0001      |        0.897 | <0.0001     |
|               100 | Windowless  | 15%       | 100 |            68 | 2–99                     |        0.945 | <0.0001      |        0.844 | <0.0001     |
|               100 | Windowed    | 30%       | 100 |            68 | 2–99                     |        0.951 | <0.0001      |        0.841 | <0.0001     |
|               100 | Windowless  | 30%       | 100 |            68 | 2–99                     |        0.927 | <0.0001      |        0.802 | <0.0001     |
|               100 | Windowed    | 45%       | 100 |            68 | 2–99                     |        0.914 | <0.0001      |        0.791 | <0.0001     |
|               100 | Windowless  | 45%       | 100 |            68 | 2–99                     |        0.911 | <0.0001      |        0.787 | <0.0001     |
|               100 | Windowed    | 60%       | 100 |            68 | 2–99                     |        0.863 | <0.0001      |        0.698 | <0.0001     |
|               100 | Windowless  | 60%       | 100 |            68 | 2–99                     |        0.931 | <0.0001      |        0.807 | <0.0001     |
|               100 | Windowed    | 75%       | 100 |            68 | 2–99                     |        0.885 | <0.0001      |        0.756 | <0.0001     |
|               100 | Windowless  | 75%       | 100 |            68 | 2–99                     |        0.747 | <0.0001      |        0.582 | <0.0001     |
|               200 | Windowed    | 0%        | 200 |            88 | 2–99                     |        0.99  | <0.0001      |        0.954 | <0.0001     |
|               200 | Windowless  | 0%        | 200 |            88 | 2–99                     |        0.995 | <0.0001      |        0.968 | <0.0001     |
|               200 | Windowed    | 15%       | 200 |            88 | 2–99                     |        0.987 | <0.0001      |        0.936 | <0.0001     |
|               200 | Windowless  | 15%       | 200 |            88 | 2–99                     |        0.986 | <0.0001      |        0.93  | <0.0001     |
|               200 | Windowed    | 30%       | 200 |            88 | 2–99                     |        0.977 | <0.0001      |        0.906 | <0.0001     |
|               200 | Windowless  | 30%       | 200 |            88 | 2–99                     |        0.97  | <0.0001      |        0.889 | <0.0001     |
|               200 | Windowed    | 45%       | 200 |            88 | 2–99                     |        0.967 | <0.0001      |        0.871 | <0.0001     |
|               200 | Windowless  | 45%       | 200 |            88 | 2–99                     |        0.965 | <0.0001      |        0.872 | <0.0001     |
|               200 | Windowed    | 60%       | 200 |            88 | 2–99                     |        0.959 | <0.0001      |        0.859 | <0.0001     |
|               200 | Windowless  | 60%       | 200 |            88 | 2–99                     |        0.962 | <0.0001      |        0.854 | <0.0001     |
|               200 | Windowed    | 75%       | 200 |            88 | 2–99                     |        0.908 | <0.0001      |        0.764 | <0.0001     |
|               200 | Windowless  | 75%       | 200 |            88 | 2–99                     |        0.916 | <0.0001      |        0.771 | <0.0001     |
|               500 | Windowed    | 0%        | 500 |            98 | 2–99                     |        0.992 | <0.0001      |        0.946 | <0.0001     |
|               500 | Windowless  | 0%        | 500 |            98 | 2–99                     |        0.992 | <0.0001      |        0.949 | <0.0001     |
|               500 | Windowed    | 15%       | 500 |            98 | 2–99                     |        0.981 | <0.0001      |        0.913 | <0.0001     |
|               500 | Windowless  | 15%       | 500 |            98 | 2–99                     |        0.986 | <0.0001      |        0.927 | <0.0001     |
|               500 | Windowed    | 30%       | 500 |            98 | 2–99                     |        0.978 | <0.0001      |        0.897 | <0.0001     |
|               500 | Windowless  | 30%       | 500 |            98 | 2–99                     |        0.973 | <0.0001      |        0.893 | <0.0001     |
|               500 | Windowed    | 45%       | 500 |            98 | 2–99                     |        0.961 | <0.0001      |        0.854 | <0.0001     |
|               500 | Windowless  | 45%       | 500 |            98 | 2–99                     |        0.969 | <0.0001      |        0.874 | <0.0001     |
|               500 | Windowed    | 60%       | 500 |            98 | 2–99                     |        0.961 | <0.0001      |        0.848 | <0.0001     |
|               500 | Windowless  | 60%       | 500 |            98 | 2–99                     |        0.965 | <0.0001      |        0.855 | <0.0001     |
|               500 | Windowed    | 75%       | 500 |            98 | 2–99                     |        0.924 | <0.0001      |        0.785 | <0.0001     |
|               500 | Windowless  | 75%       | 500 |            98 | 2–99                     |        0.951 | <0.0001      |        0.829 | <0.0001     |

#### Retained `hasAge` Relations

Only people whose `hasAge` triple remains in the evaluated run.

|   Population Size | Structure   | Removed   |   N |   Unique Ages | True-Age Range (years)   |   Spearman ρ | Spearman p   |   Kendall τb | Kendall p   |
|------------------:|:------------|:----------|----:|--------------:|:-------------------------|-------------:|:-------------|-------------:|:------------|
|               100 | Windowed    | 0%        | 100 |            68 | 2–99                     |        0.98  | <0.0001      |        0.927 | <0.0001     |
|               100 | Windowless  | 0%        | 100 |            68 | 2–99                     |        0.987 | <0.0001      |        0.938 | <0.0001     |
|               100 | Windowed    | 15%       |  85 |            60 | 2–99                     |        0.989 | <0.0001      |        0.939 | <0.0001     |
|               100 | Windowless  | 15%       |  85 |            60 | 2–99                     |        0.976 | <0.0001      |        0.898 | <0.0001     |
|               100 | Windowed    | 30%       |  70 |            49 | 2–99                     |        0.967 | <0.0001      |        0.883 | <0.0001     |
|               100 | Windowless  | 30%       |  70 |            49 | 2–99                     |        0.944 | <0.0001      |        0.836 | <0.0001     |
|               100 | Windowed    | 45%       |  55 |            43 | 2–99                     |        0.929 | <0.0001      |        0.827 | <0.0001     |
|               100 | Windowless  | 45%       |  55 |            43 | 2–99                     |        0.951 | <0.0001      |        0.863 | <0.0001     |
|               100 | Windowed    | 60%       |  40 |            35 | 2–97                     |        0.87  | <0.0001      |        0.711 | <0.0001     |
|               100 | Windowless  | 60%       |  40 |            35 | 2–97                     |        0.979 | <0.0001      |        0.914 | <0.0001     |
|               100 | Windowed    | 75%       |  25 |            24 | 4–97                     |        0.933 | <0.0001      |        0.846 | <0.0001     |
|               100 | Windowless  | 75%       |  25 |            24 | 4–97                     |        0.919 | <0.0001      |        0.798 | <0.0001     |
|               200 | Windowed    | 0%        | 200 |            88 | 2–99                     |        0.99  | <0.0001      |        0.954 | <0.0001     |
|               200 | Windowless  | 0%        | 200 |            88 | 2–99                     |        0.995 | <0.0001      |        0.968 | <0.0001     |
|               200 | Windowed    | 15%       | 170 |            82 | 2–99                     |        0.996 | <0.0001      |        0.97  | <0.0001     |
|               200 | Windowless  | 15%       | 170 |            82 | 2–99                     |        0.993 | <0.0001      |        0.962 | <0.0001     |
|               200 | Windowed    | 30%       | 140 |            75 | 2–99                     |        0.991 | <0.0001      |        0.955 | <0.0001     |
|               200 | Windowless  | 30%       | 140 |            75 | 2–99                     |        0.989 | <0.0001      |        0.946 | <0.0001     |
|               200 | Windowed    | 45%       | 110 |            67 | 2–99                     |        0.99  | <0.0001      |        0.946 | <0.0001     |
|               200 | Windowless  | 45%       | 110 |            67 | 2–99                     |        0.987 | <0.0001      |        0.951 | <0.0001     |
|               200 | Windowed    | 60%       |  80 |            55 | 2–99                     |        0.986 | <0.0001      |        0.944 | <0.0001     |
|               200 | Windowless  | 60%       |  80 |            55 | 2–99                     |        0.987 | <0.0001      |        0.944 | <0.0001     |
|               200 | Windowed    | 75%       |  50 |            39 | 2–97                     |        0.962 | <0.0001      |        0.888 | <0.0001     |
|               200 | Windowless  | 75%       |  50 |            39 | 2–97                     |        0.958 | <0.0001      |        0.872 | <0.0001     |
|               500 | Windowed    | 0%        | 500 |            98 | 2–99                     |        0.992 | <0.0001      |        0.946 | <0.0001     |
|               500 | Windowless  | 0%        | 500 |            98 | 2–99                     |        0.992 | <0.0001      |        0.949 | <0.0001     |
|               500 | Windowed    | 15%       | 425 |            98 | 2–99                     |        0.985 | <0.0001      |        0.927 | <0.0001     |
|               500 | Windowless  | 15%       | 425 |            98 | 2–99                     |        0.99  | <0.0001      |        0.944 | <0.0001     |
|               500 | Windowed    | 30%       | 350 |            93 | 2–99                     |        0.987 | <0.0001      |        0.934 | <0.0001     |
|               500 | Windowless  | 30%       | 350 |            93 | 2–99                     |        0.977 | <0.0001      |        0.92  | <0.0001     |
|               500 | Windowed    | 45%       | 275 |            92 | 2–99                     |        0.979 | <0.0001      |        0.913 | <0.0001     |
|               500 | Windowless  | 45%       | 275 |            92 | 2–99                     |        0.984 | <0.0001      |        0.924 | <0.0001     |
|               500 | Windowed    | 60%       | 200 |            87 | 2–99                     |        0.981 | <0.0001      |        0.913 | <0.0001     |
|               500 | Windowless  | 60%       | 200 |            87 | 2–99                     |        0.98  | <0.0001      |        0.914 | <0.0001     |
|               500 | Windowed    | 75%       | 125 |            70 | 2–99                     |        0.963 | <0.0001      |        0.862 | <0.0001     |
|               500 | Windowless  | 75%       | 125 |            70 | 2–99                     |        0.972 | <0.0001      |        0.896 | <0.0001     |

#### Removed `hasAge` Relations

Only people whose `hasAge` triple was removed from the evaluated run.

|   Population Size | Structure   | Removed   |   N |   Unique Ages | True-Age Range (years)   |   Spearman ρ | Spearman p   |   Kendall τb | Kendall p   |
|------------------:|:------------|:----------|----:|--------------:|:-------------------------|-------------:|:-------------|-------------:|:------------|
|               100 | Windowed    | 15%       |  15 |            12 | 3–91                     |        0.924 | <0.0001      |        0.822 | <0.0001     |
|               100 | Windowless  | 15%       |  15 |            12 | 3–91                     |        0.363 | 0.1841       |        0.228 | 0.2736      |
|               100 | Windowed    | 30%       |  30 |            26 | 3–97                     |        0.918 | <0.0001      |        0.79  | <0.0001     |
|               100 | Windowless  | 30%       |  30 |            26 | 3–97                     |        0.876 | <0.0001      |        0.739 | <0.0001     |
|               100 | Windowed    | 45%       |  45 |            38 | 3–97                     |        0.892 | <0.0001      |        0.758 | <0.0001     |
|               100 | Windowless  | 45%       |  45 |            38 | 3–97                     |        0.87  | <0.0001      |        0.721 | <0.0001     |
|               100 | Windowed    | 60%       |  60 |            46 | 3–99                     |        0.87  | <0.0001      |        0.72  | <0.0001     |
|               100 | Windowless  | 60%       |  60 |            46 | 3–99                     |        0.901 | <0.0001      |        0.76  | <0.0001     |
|               100 | Windowed    | 75%       |  75 |            56 | 2–99                     |        0.872 | <0.0001      |        0.744 | <0.0001     |
|               100 | Windowless  | 75%       |  75 |            56 | 2–99                     |        0.678 | <0.0001      |        0.516 | <0.0001     |
|               200 | Windowed    | 15%       |  30 |            24 | 3–97                     |        0.932 | <0.0001      |        0.807 | <0.0001     |
|               200 | Windowless  | 15%       |  30 |            24 | 3–97                     |        0.95  | <0.0001      |        0.842 | <0.0001     |
|               200 | Windowed    | 30%       |  60 |            42 | 2–97                     |        0.949 | <0.0001      |        0.846 | <0.0001     |
|               200 | Windowless  | 30%       |  60 |            42 | 2–97                     |        0.936 | <0.0001      |        0.826 | <0.0001     |
|               200 | Windowed    | 45%       |  90 |            56 | 2–97                     |        0.947 | <0.0001      |        0.826 | <0.0001     |
|               200 | Windowless  | 45%       |  90 |            56 | 2–97                     |        0.937 | <0.0001      |        0.808 | <0.0001     |
|               200 | Windowed    | 60%       | 120 |            65 | 2–99                     |        0.939 | <0.0001      |        0.824 | <0.0001     |
|               200 | Windowless  | 60%       | 120 |            65 | 2–99                     |        0.947 | <0.0001      |        0.821 | <0.0001     |
|               200 | Windowed    | 75%       | 150 |            77 | 2–99                     |        0.894 | <0.0001      |        0.743 | <0.0001     |
|               200 | Windowless  | 75%       | 150 |            77 | 2–99                     |        0.904 | <0.0001      |        0.75  | <0.0001     |
|               500 | Windowed    | 15%       |  75 |            52 | 2–96                     |        0.968 | <0.0001      |        0.863 | <0.0001     |
|               500 | Windowless  | 15%       |  75 |            52 | 2–96                     |        0.962 | <0.0001      |        0.854 | <0.0001     |
|               500 | Windowed    | 30%       | 150 |            77 | 2–97                     |        0.956 | <0.0001      |        0.841 | <0.0001     |
|               500 | Windowless  | 30%       | 150 |            77 | 2–97                     |        0.959 | <0.0001      |        0.847 | <0.0001     |
|               500 | Windowed    | 45%       | 225 |            85 | 2–99                     |        0.937 | <0.0001      |        0.8   | <0.0001     |
|               500 | Windowless  | 45%       | 225 |            85 | 2–99                     |        0.952 | <0.0001      |        0.831 | <0.0001     |
|               500 | Windowed    | 60%       | 300 |            93 | 2–99                     |        0.947 | <0.0001      |        0.811 | <0.0001     |
|               500 | Windowless  | 60%       | 300 |            93 | 2–99                     |        0.954 | <0.0001      |        0.829 | <0.0001     |
|               500 | Windowed    | 75%       | 375 |            96 | 2–99                     |        0.911 | <0.0001      |        0.766 | <0.0001     |
|               500 | Windowless  | 75%       | 375 |            96 | 2–99                     |        0.947 | <0.0001      |        0.819 | <0.0001     |

### Learned Age-Axis Ordering

#### Full Population

All people in the evaluated graph.

|   Population Size | Structure   | Removed   |   N |   Unique Ages | True-Age Range (years)   |   Spearman ρ | Spearman p   |   Kendall τb | Kendall p   |
|------------------:|:------------|:----------|----:|--------------:|:-------------------------|-------------:|:-------------|-------------:|:------------|
|               100 | Windowed    | 0%        | 100 |            68 | 2–99                     |        1     | <0.0001      |        0.995 | <0.0001     |
|               100 | Windowless  | 0%        | 100 |            68 | 2–99                     |        1     | <0.0001      |        0.996 | <0.0001     |
|               100 | Windowed    | 15%       | 100 |            68 | 2–99                     |        0.997 | <0.0001      |        0.976 | <0.0001     |
|               100 | Windowless  | 15%       | 100 |            68 | 2–99                     |        0.998 | <0.0001      |        0.979 | <0.0001     |
|               100 | Windowed    | 30%       | 100 |            68 | 2–99                     |        0.996 | <0.0001      |        0.964 | <0.0001     |
|               100 | Windowless  | 30%       | 100 |            68 | 2–99                     |        0.99  | <0.0001      |        0.931 | <0.0001     |
|               100 | Windowed    | 45%       | 100 |            68 | 2–99                     |        0.992 | <0.0001      |        0.936 | <0.0001     |
|               100 | Windowless  | 45%       | 100 |            68 | 2–99                     |        0.993 | <0.0001      |        0.945 | <0.0001     |
|               100 | Windowed    | 60%       | 100 |            68 | 2–99                     |        0.99  | <0.0001      |        0.93  | <0.0001     |
|               100 | Windowless  | 60%       | 100 |            68 | 2–99                     |        0.99  | <0.0001      |        0.922 | <0.0001     |
|               100 | Windowed    | 75%       | 100 |            68 | 2–99                     |        0.989 | <0.0001      |        0.919 | <0.0001     |
|               100 | Windowless  | 75%       | 100 |            68 | 2–99                     |        0.986 | <0.0001      |        0.907 | <0.0001     |
|               200 | Windowed    | 0%        | 200 |            88 | 2–99                     |        1     | <0.0001      |        0.99  | <0.0001     |
|               200 | Windowless  | 0%        | 200 |            88 | 2–99                     |        1     | <0.0001      |        0.991 | <0.0001     |
|               200 | Windowed    | 15%       | 200 |            88 | 2–99                     |        0.998 | <0.0001      |        0.976 | <0.0001     |
|               200 | Windowless  | 15%       | 200 |            88 | 2–99                     |        0.998 | <0.0001      |        0.975 | <0.0001     |
|               200 | Windowed    | 30%       | 200 |            88 | 2–99                     |        0.996 | <0.0001      |        0.961 | <0.0001     |
|               200 | Windowless  | 30%       | 200 |            88 | 2–99                     |        0.997 | <0.0001      |        0.966 | <0.0001     |
|               200 | Windowed    | 45%       | 200 |            88 | 2–99                     |        0.993 | <0.0001      |        0.942 | <0.0001     |
|               200 | Windowless  | 45%       | 200 |            88 | 2–99                     |        0.992 | <0.0001      |        0.932 | <0.0001     |
|               200 | Windowed    | 60%       | 200 |            88 | 2–99                     |        0.995 | <0.0001      |        0.945 | <0.0001     |
|               200 | Windowless  | 60%       | 200 |            88 | 2–99                     |        0.993 | <0.0001      |        0.937 | <0.0001     |
|               200 | Windowed    | 75%       | 200 |            88 | 2–99                     |        0.993 | <0.0001      |        0.935 | <0.0001     |
|               200 | Windowless  | 75%       | 200 |            88 | 2–99                     |        0.993 | <0.0001      |        0.937 | <0.0001     |
|               500 | Windowed    | 0%        | 500 |            98 | 2–99                     |        0.999 | <0.0001      |        0.978 | <0.0001     |
|               500 | Windowless  | 0%        | 500 |            98 | 2–99                     |        0.999 | <0.0001      |        0.98  | <0.0001     |
|               500 | Windowed    | 15%       | 500 |            98 | 2–99                     |        0.999 | <0.0001      |        0.979 | <0.0001     |
|               500 | Windowless  | 15%       | 500 |            98 | 2–99                     |        0.999 | <0.0001      |        0.973 | <0.0001     |
|               500 | Windowed    | 30%       | 500 |            98 | 2–99                     |        0.997 | <0.0001      |        0.962 | <0.0001     |
|               500 | Windowless  | 30%       | 500 |            98 | 2–99                     |        0.998 | <0.0001      |        0.966 | <0.0001     |
|               500 | Windowed    | 45%       | 500 |            98 | 2–99                     |        0.996 | <0.0001      |        0.953 | <0.0001     |
|               500 | Windowless  | 45%       | 500 |            98 | 2–99                     |        0.995 | <0.0001      |        0.949 | <0.0001     |
|               500 | Windowed    | 60%       | 500 |            98 | 2–99                     |        0.995 | <0.0001      |        0.946 | <0.0001     |
|               500 | Windowless  | 60%       | 500 |            98 | 2–99                     |        0.996 | <0.0001      |        0.95  | <0.0001     |
|               500 | Windowed    | 75%       | 500 |            98 | 2–99                     |        0.996 | <0.0001      |        0.951 | <0.0001     |
|               500 | Windowless  | 75%       | 500 |            98 | 2–99                     |        0.996 | <0.0001      |        0.951 | <0.0001     |

#### Retained `hasAge` Relations

Only people whose `hasAge` triple remains in the evaluated run.

|   Population Size | Structure   | Removed   |   N |   Unique Ages | True-Age Range (years)   |   Spearman ρ | Spearman p   |   Kendall τb | Kendall p   |
|------------------:|:------------|:----------|----:|--------------:|:-------------------------|-------------:|:-------------|-------------:|:------------|
|               100 | Windowed    | 0%        | 100 |            68 | 2–99                     |        1     | <0.0001      |        0.995 | <0.0001     |
|               100 | Windowless  | 0%        | 100 |            68 | 2–99                     |        1     | <0.0001      |        0.996 | <0.0001     |
|               100 | Windowed    | 15%       |  85 |            60 | 2–99                     |        1     | <0.0001      |        0.996 | <0.0001     |
|               100 | Windowless  | 15%       |  85 |            60 | 2–99                     |        1     | <0.0001      |        0.995 | <0.0001     |
|               100 | Windowed    | 30%       |  70 |            49 | 2–99                     |        1     | <0.0001      |        0.995 | <0.0001     |
|               100 | Windowless  | 30%       |  70 |            49 | 2–99                     |        1     | <0.0001      |        0.995 | <0.0001     |
|               100 | Windowed    | 45%       |  55 |            43 | 2–99                     |        1     | <0.0001      |        0.996 | <0.0001     |
|               100 | Windowless  | 45%       |  55 |            43 | 2–99                     |        1     | <0.0001      |        0.996 | <0.0001     |
|               100 | Windowed    | 60%       |  40 |            35 | 2–97                     |        1     | <0.0001      |        0.997 | <0.0001     |
|               100 | Windowless  | 60%       |  40 |            35 | 2–97                     |        1     | <0.0001      |        0.997 | <0.0001     |
|               100 | Windowed    | 75%       |  25 |            24 | 4–97                     |        1     | <0.0001      |        0.998 | <0.0001     |
|               100 | Windowless  | 75%       |  25 |            24 | 4–97                     |        1     | <0.0001      |        0.998 | <0.0001     |
|               200 | Windowed    | 0%        | 200 |            88 | 2–99                     |        1     | <0.0001      |        0.99  | <0.0001     |
|               200 | Windowless  | 0%        | 200 |            88 | 2–99                     |        1     | <0.0001      |        0.991 | <0.0001     |
|               200 | Windowed    | 15%       | 170 |            82 | 2–99                     |        1     | <0.0001      |        0.993 | <0.0001     |
|               200 | Windowless  | 15%       | 170 |            82 | 2–99                     |        1     | <0.0001      |        0.995 | <0.0001     |
|               200 | Windowed    | 30%       | 140 |            75 | 2–99                     |        1     | <0.0001      |        0.994 | <0.0001     |
|               200 | Windowless  | 30%       | 140 |            75 | 2–99                     |        1     | <0.0001      |        0.995 | <0.0001     |
|               200 | Windowed    | 45%       | 110 |            67 | 2–99                     |        1     | <0.0001      |        0.995 | <0.0001     |
|               200 | Windowless  | 45%       | 110 |            67 | 2–99                     |        1     | <0.0001      |        0.995 | <0.0001     |
|               200 | Windowed    | 60%       |  80 |            55 | 2–99                     |        1     | <0.0001      |        0.995 | <0.0001     |
|               200 | Windowless  | 60%       |  80 |            55 | 2–99                     |        1     | <0.0001      |        0.995 | <0.0001     |
|               200 | Windowed    | 75%       |  50 |            39 | 2–97                     |        1     | <0.0001      |        0.995 | <0.0001     |
|               200 | Windowless  | 75%       |  50 |            39 | 2–97                     |        1     | <0.0001      |        0.995 | <0.0001     |
|               500 | Windowed    | 0%        | 500 |            98 | 2–99                     |        0.999 | <0.0001      |        0.978 | <0.0001     |
|               500 | Windowless  | 0%        | 500 |            98 | 2–99                     |        0.999 | <0.0001      |        0.98  | <0.0001     |
|               500 | Windowed    | 15%       | 425 |            98 | 2–99                     |        0.999 | <0.0001      |        0.985 | <0.0001     |
|               500 | Windowless  | 15%       | 425 |            98 | 2–99                     |        0.999 | <0.0001      |        0.98  | <0.0001     |
|               500 | Windowed    | 30%       | 350 |            93 | 2–99                     |        1     | <0.0001      |        0.987 | <0.0001     |
|               500 | Windowless  | 30%       | 350 |            93 | 2–99                     |        1     | <0.0001      |        0.986 | <0.0001     |
|               500 | Windowed    | 45%       | 275 |            92 | 2–99                     |        1     | <0.0001      |        0.993 | <0.0001     |
|               500 | Windowless  | 45%       | 275 |            92 | 2–99                     |        1     | <0.0001      |        0.994 | <0.0001     |
|               500 | Windowed    | 60%       | 200 |            87 | 2–99                     |        1     | <0.0001      |        0.995 | <0.0001     |
|               500 | Windowless  | 60%       | 200 |            87 | 2–99                     |        1     | <0.0001      |        0.995 | <0.0001     |
|               500 | Windowed    | 75%       | 125 |            70 | 2–99                     |        1     | <0.0001      |        0.995 | <0.0001     |
|               500 | Windowless  | 75%       | 125 |            70 | 2–99                     |        1     | <0.0001      |        0.995 | <0.0001     |

#### Removed `hasAge` Relations

Only people whose `hasAge` triple was removed from the evaluated run.

|   Population Size | Structure   | Removed   |   N |   Unique Ages | True-Age Range (years)   |   Spearman ρ | Spearman p   |   Kendall τb | Kendall p   |
|------------------:|:------------|:----------|----:|--------------:|:-------------------------|-------------:|:-------------|-------------:|:------------|
|               100 | Windowed    | 15%       |  15 |            12 | 3–91                     |        0.983 | <0.0001      |        0.928 | <0.0001     |
|               100 | Windowless  | 15%       |  15 |            12 | 3–91                     |        0.976 | <0.0001      |        0.908 | <0.0001     |
|               100 | Windowed    | 30%       |  30 |            26 | 3–97                     |        0.984 | <0.0001      |        0.926 | <0.0001     |
|               100 | Windowless  | 30%       |  30 |            26 | 3–97                     |        0.975 | <0.0001      |        0.88  | <0.0001     |
|               100 | Windowed    | 45%       |  45 |            38 | 3–97                     |        0.981 | <0.0001      |        0.889 | <0.0001     |
|               100 | Windowless  | 45%       |  45 |            38 | 3–97                     |        0.977 | <0.0001      |        0.893 | <0.0001     |
|               100 | Windowed    | 60%       |  60 |            46 | 3–99                     |        0.986 | <0.0001      |        0.91  | <0.0001     |
|               100 | Windowless  | 60%       |  60 |            46 | 3–99                     |        0.982 | <0.0001      |        0.888 | <0.0001     |
|               100 | Windowed    | 75%       |  75 |            56 | 2–99                     |        0.986 | <0.0001      |        0.909 | <0.0001     |
|               100 | Windowless  | 75%       |  75 |            56 | 2–99                     |        0.982 | <0.0001      |        0.89  | <0.0001     |
|               200 | Windowed    | 15%       |  30 |            24 | 3–97                     |        0.982 | <0.0001      |        0.905 | <0.0001     |
|               200 | Windowless  | 15%       |  30 |            24 | 3–97                     |        0.982 | <0.0001      |        0.9   | <0.0001     |
|               200 | Windowed    | 30%       |  60 |            42 | 2–97                     |        0.983 | <0.0001      |        0.897 | <0.0001     |
|               200 | Windowless  | 30%       |  60 |            42 | 2–97                     |        0.989 | <0.0001      |        0.921 | <0.0001     |
|               200 | Windowed    | 45%       |  90 |            56 | 2–97                     |        0.988 | <0.0001      |        0.913 | <0.0001     |
|               200 | Windowless  | 45%       |  90 |            56 | 2–97                     |        0.982 | <0.0001      |        0.885 | <0.0001     |
|               200 | Windowed    | 60%       | 120 |            65 | 2–99                     |        0.991 | <0.0001      |        0.924 | <0.0001     |
|               200 | Windowless  | 60%       | 120 |            65 | 2–99                     |        0.989 | <0.0001      |        0.914 | <0.0001     |
|               200 | Windowed    | 75%       | 150 |            77 | 2–99                     |        0.992 | <0.0001      |        0.926 | <0.0001     |
|               200 | Windowless  | 75%       | 150 |            77 | 2–99                     |        0.991 | <0.0001      |        0.925 | <0.0001     |
|               500 | Windowed    | 15%       |  75 |            52 | 2–96                     |        0.996 | <0.0001      |        0.961 | <0.0001     |
|               500 | Windowless  | 15%       |  75 |            52 | 2–96                     |        0.995 | <0.0001      |        0.951 | <0.0001     |
|               500 | Windowed    | 30%       | 150 |            77 | 2–97                     |        0.991 | <0.0001      |        0.925 | <0.0001     |
|               500 | Windowless  | 30%       | 150 |            77 | 2–97                     |        0.993 | <0.0001      |        0.931 | <0.0001     |
|               500 | Windowed    | 45%       | 225 |            85 | 2–99                     |        0.991 | <0.0001      |        0.92  | <0.0001     |
|               500 | Windowless  | 45%       | 225 |            85 | 2–99                     |        0.99  | <0.0001      |        0.918 | <0.0001     |
|               500 | Windowed    | 60%       | 300 |            93 | 2–99                     |        0.992 | <0.0001      |        0.925 | <0.0001     |
|               500 | Windowless  | 60%       | 300 |            93 | 2–99                     |        0.993 | <0.0001      |        0.932 | <0.0001     |
|               500 | Windowed    | 75%       | 375 |            96 | 2–99                     |        0.995 | <0.0001      |        0.942 | <0.0001     |
|               500 | Windowless  | 75%       | 375 |            96 | 2–99                     |        0.995 | <0.0001      |        0.942 | <0.0001     |

## Metric Notes

- **AE:** absolute error in years. **MAE:** mean absolute error. **SD:** standard deviation. **N:** cases or pairs.
- **Spearman ρ:** rank correlation based on monotonic association. **Kendall τb:** pairwise rank agreement adjusted for ties.
- **Paired t p / Wilcoxon p:** paired-test p-values. **Cohen’s dz:** paired effect size. **95% bootstrap CI:** interval for the mean paired difference.
- Retained learned-regression rows describe fit on available age anchors; removed rows describe recovery for ages not used as anchors.

## Nested Removal

Removal sets are cumulative: `R15 ⊆ R30 ⊆ R45 ⊆ R60 ⊆ R75`. Complete KG triple sets shrink oppositely: `KG0 ⊇ KG15 ⊇ KG30 ⊇ KG45 ⊇ KG60 ⊇ KG75`.
