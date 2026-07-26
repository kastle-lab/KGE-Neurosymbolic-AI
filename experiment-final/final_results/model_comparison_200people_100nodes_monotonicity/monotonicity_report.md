# Cross-Model Numeric Monotonicity Analysis

For each trained checkpoint, PCA was fit to the embedding vectors of the numeric literal nodes only (`v*`). The first principal component was then compared with the ground-truth numeric values using Spearman's rho and Kendall's tau.

Because PCA component sign is arbitrary, absolute correlations are used as the primary preservation metrics. PC1 is sign-oriented only for visualization.

## Overall Model Ranking

| model    |   mean_spearman_abs |   median_spearman_abs |   mean_kendall_abs |   median_kendall_abs |   combined_score |   overall_rank |
|:---------|--------------------:|----------------------:|-------------------:|---------------------:|-----------------:|---------------:|
| mure     |              0.9870 |                0.9876 |             0.9128 |               0.9141 |           0.9499 |              1 |
| transd   |              0.9806 |                0.9794 |             0.8830 |               0.8786 |           0.9318 |              2 |
| transe   |              0.9746 |                0.9743 |             0.8659 |               0.8661 |           0.9203 |              3 |
| distmult |              0.4267 |                0.4461 |             0.2939 |               0.2992 |           0.3603 |              4 |
| transr   |              0.2344 |                0.1668 |             0.1628 |               0.1093 |           0.1986 |              5 |

## Per-Run Results

| model    | window_condition   |   removal_percent |   n_age_literals |   spearman_abs |   kendall_abs |
|:---------|:-------------------|------------------:|-----------------:|---------------:|--------------:|
| distmult | with_windows       |            0.0000 |              100 |         0.7780 |        0.5394 |
| mure     | with_windows       |            0.0000 |              100 |         0.9909 |        0.9305 |
| transd   | with_windows       |            0.0000 |              100 |         0.9779 |        0.8735 |
| transe   | with_windows       |            0.0000 |              100 |         0.9710 |        0.8598 |
| transr   | with_windows       |            0.0000 |              100 |         0.1818 |        0.1095 |
| distmult | with_windows       |           25.0000 |              100 |         0.7419 |        0.4994 |
| mure     | with_windows       |           25.0000 |              100 |         0.9863 |        0.9103 |
| transd   | with_windows       |           25.0000 |              100 |         0.9805 |        0.8832 |
| transe   | with_windows       |           25.0000 |              100 |         0.9790 |        0.8760 |
| transr   | with_windows       |           25.0000 |              100 |         0.1517 |        0.1091 |
| distmult | with_windows       |           50.0000 |              100 |         0.7509 |        0.5071 |
| mure     | with_windows       |           50.0000 |              100 |         0.9886 |        0.9180 |
| transd   | with_windows       |           50.0000 |              100 |         0.9776 |        0.8715 |
| transe   | with_windows       |           50.0000 |              100 |         0.9743 |        0.8675 |
| transr   | with_windows       |           50.0000 |              100 |         0.0788 |        0.0529 |
| distmult | with_windows       |           75.0000 |              100 |         0.7830 |        0.5455 |
| mure     | with_windows       |           75.0000 |              100 |         0.9817 |        0.8933 |
| transd   | with_windows       |           75.0000 |              100 |         0.9788 |        0.8735 |
| transe   | with_windows       |           75.0000 |              100 |         0.9789 |        0.8764 |
| transr   | with_windows       |           75.0000 |              100 |         0.8402 |        0.6368 |
| distmult | with_windows       |           99.0000 |              100 |         0.8045 |        0.5822 |
| mure     | with_windows       |           99.0000 |              100 |         0.9866 |        0.9135 |
| transd   | with_windows       |           99.0000 |              100 |         0.9783 |        0.8719 |
| transe   | with_windows       |           99.0000 |              100 |         0.9743 |        0.8594 |
| transr   | with_windows       |           99.0000 |              100 |         0.0220 |        0.0174 |
| distmult | without_windows    |            0.0000 |              100 |         0.0684 |        0.0428 |
| mure     | without_windows    |            0.0000 |              100 |         0.9915 |        0.9285 |
| transd   | without_windows    |            0.0000 |              100 |         0.9864 |        0.9042 |
| transe   | without_windows    |            0.0000 |              100 |         0.9686 |        0.8493 |
| transr   | without_windows    |            0.0000 |              100 |         0.3210 |        0.2170 |
| distmult | without_windows    |           25.0000 |              100 |         0.0541 |        0.0364 |
| mure     | without_windows    |           25.0000 |              100 |         0.9879 |        0.9143 |
| transd   | without_windows    |           25.0000 |              100 |         0.9852 |        0.8978 |
| transe   | without_windows    |           25.0000 |              100 |         0.9735 |        0.8590 |
| transr   | without_windows    |           25.0000 |              100 |         0.0586 |        0.0424 |
| distmult | without_windows    |           50.0000 |              100 |         0.0752 |        0.0461 |
| mure     | without_windows    |           50.0000 |              100 |         0.9795 |        0.8857 |
| transd   | without_windows    |           50.0000 |              100 |         0.9773 |        0.8739 |
| transe   | without_windows    |           50.0000 |              100 |         0.9767 |        0.8743 |
| transr   | without_windows    |           50.0000 |              100 |         0.3525 |        0.2404 |
| distmult | without_windows    |           75.0000 |              100 |         0.1503 |        0.0990 |
| mure     | without_windows    |           75.0000 |              100 |         0.9895 |        0.9196 |
| transd   | without_windows    |           75.0000 |              100 |         0.9800 |        0.8840 |
| transe   | without_windows    |           75.0000 |              100 |         0.9760 |        0.8727 |
| transr   | without_windows    |           75.0000 |              100 |         0.2483 |        0.1446 |
| distmult | without_windows    |           99.0000 |              100 |         0.0611 |        0.0416 |
| mure     | without_windows    |           99.0000 |              100 |         0.9873 |        0.9139 |
| transd   | without_windows    |           99.0000 |              100 |         0.9841 |        0.8962 |
| transe   | without_windows    |           99.0000 |              100 |         0.9737 |        0.8646 |
| transr   | without_windows    |           99.0000 |              100 |         0.0891 |        0.0582 |

## Removal Robustness

| model    | window_condition   |   spearman_degradation |   kendall_degradation |   spearman_normalized_auc |   kendall_normalized_auc |   combined_auc |
|:---------|:-------------------|-----------------------:|----------------------:|--------------------------:|-------------------------:|---------------:|
| mure     | with_windows       |                 0.0043 |                0.0170 |                    0.9864 |                   0.9110 |         0.9487 |
| mure     | without_windows    |                 0.0042 |                0.0145 |                    0.9865 |                   0.9101 |         0.9483 |
| transd   | without_windows    |                 0.0022 |                0.0081 |                    0.9819 |                   0.8890 |         0.9355 |
| transd   | with_windows       |                -0.0004 |                0.0016 |                    0.9788 |                   0.8753 |         0.9270 |
| transe   | with_windows       |                -0.0033 |                0.0004 |                    0.9762 |                   0.8699 |         0.9230 |
| transe   | without_windows    |                -0.0051 |               -0.0154 |                    0.9743 |                   0.8657 |         0.9200 |
| distmult | with_windows       |                -0.0265 |               -0.0428 |                    0.7665 |                   0.5278 |         0.6472 |
| transr   | with_windows       |                 0.1598 |                0.0921 |                    0.2918 |                   0.2144 |         0.2531 |
| transr   | without_windows    |                 0.2319 |                0.1588 |                    0.2166 |                   0.1417 |         0.1791 |
| distmult | without_windows    |                 0.0073 |                0.0012 |                    0.0859 |                   0.0558 |         0.0708 |

## Window Effects

| model    |   removal_percent |   with_windows_spearman_abs |   without_windows_spearman_abs |   delta_spearman_with_minus_without |   with_windows_kendall_abs |   without_windows_kendall_abs |   delta_kendall_with_minus_without |
|:---------|------------------:|----------------------------:|-------------------------------:|------------------------------------:|---------------------------:|------------------------------:|-----------------------------------:|
| distmult |            0.0000 |                      0.7780 |                         0.0684 |                              0.7096 |                     0.5394 |                        0.0428 |                             0.4966 |
| mure     |            0.0000 |                      0.9909 |                         0.9915 |                             -0.0006 |                     0.9305 |                        0.9285 |                             0.0020 |
| transd   |            0.0000 |                      0.9779 |                         0.9864 |                             -0.0085 |                     0.8735 |                        0.9042 |                            -0.0307 |
| transe   |            0.0000 |                      0.9710 |                         0.9686 |                              0.0024 |                     0.8598 |                        0.8493 |                             0.0105 |
| transr   |            0.0000 |                      0.1818 |                         0.3210 |                             -0.1391 |                     0.1095 |                        0.2170 |                            -0.1075 |
| distmult |           25.0000 |                      0.7419 |                         0.0541 |                              0.6877 |                     0.4994 |                        0.0364 |                             0.4630 |
| mure     |           25.0000 |                      0.9863 |                         0.9879 |                             -0.0015 |                     0.9103 |                        0.9143 |                            -0.0040 |
| transd   |           25.0000 |                      0.9805 |                         0.9852 |                             -0.0046 |                     0.8832 |                        0.8978 |                            -0.0145 |
| transe   |           25.0000 |                      0.9790 |                         0.9735 |                              0.0056 |                     0.8760 |                        0.8590 |                             0.0170 |
| transr   |           25.0000 |                      0.1517 |                         0.0586 |                              0.0931 |                     0.1091 |                        0.0424 |                             0.0667 |
| distmult |           50.0000 |                      0.7509 |                         0.0752 |                              0.6757 |                     0.5071 |                        0.0461 |                             0.4610 |
| mure     |           50.0000 |                      0.9886 |                         0.9795 |                              0.0091 |                     0.9180 |                        0.8857 |                             0.0323 |
| transd   |           50.0000 |                      0.9776 |                         0.9773 |                              0.0003 |                     0.8715 |                        0.8739 |                            -0.0024 |
| transe   |           50.0000 |                      0.9743 |                         0.9767 |                             -0.0025 |                     0.8675 |                        0.8743 |                            -0.0069 |
| transr   |           50.0000 |                      0.0788 |                         0.3525 |                             -0.2737 |                     0.0529 |                        0.2404 |                            -0.1875 |
| distmult |           75.0000 |                      0.7830 |                         0.1503 |                              0.6328 |                     0.5455 |                        0.0990 |                             0.4465 |
| mure     |           75.0000 |                      0.9817 |                         0.9895 |                             -0.0078 |                     0.8933 |                        0.9196 |                            -0.0263 |
| transd   |           75.0000 |                      0.9788 |                         0.9800 |                             -0.0013 |                     0.8735 |                        0.8840 |                            -0.0105 |
| transe   |           75.0000 |                      0.9789 |                         0.9760 |                              0.0029 |                     0.8764 |                        0.8727 |                             0.0036 |
| transr   |           75.0000 |                      0.8402 |                         0.2483 |                              0.5919 |                     0.6368 |                        0.1446 |                             0.4921 |
| distmult |           99.0000 |                      0.8045 |                         0.0611 |                              0.7434 |                     0.5822 |                        0.0416 |                             0.5406 |
| mure     |           99.0000 |                      0.9866 |                         0.9873 |                             -0.0007 |                     0.9135 |                        0.9139 |                            -0.0004 |
| transd   |           99.0000 |                      0.9783 |                         0.9841 |                             -0.0058 |                     0.8719 |                        0.8962 |                            -0.0242 |
| transe   |           99.0000 |                      0.9743 |                         0.9737 |                              0.0006 |                     0.8594 |                        0.8646 |                            -0.0053 |
| transr   |           99.0000 |                      0.0220 |                         0.0891 |                             -0.0671 |                     0.0174 |                        0.0582 |                            -0.0408 |

## Interpretation

- Values near 1 indicate strong preservation of numeric order.
- Values near 0 indicate little monotonic relationship between PC1 and ground-truth numeric values.
- Positive window-effect values mean the window-materialized KG preserved ordering better than its matched non-windowed KG.
- Smaller degradation and larger normalized AUC indicate better robustness as direct `hasAge` supervision is removed.
