# README

Each results folder (100people, 500people, no_windows_100people) contains a `results_report.md` file. That will have all the data found in the CSVs in one place. 


# Experiment Overview

## Pipeline
ps
pipeline.py contains the following workflow:

- Create KG with `n` people 
- Create modified KGs by remove `k` intervals of `p` percent of `hasAge` relationships in. E.g. if `k=5` and `p=3`, the script will create modified KGs 15% removed, 30% removed and 45% removed, and keep the original with 0% removed.
- Create modified KGs for all existing KGs at this step (including the hasAge removed modified KGs) by removing all `window` triples.
- Generate embeddings of all KGs both original and modified. 
- Calculate accuracy of relation vectors to their corresponding object node in the ground truth. 
- Learn affine mapping from relation vectors to numeric literals correspondng to the "value" (age) of the node. I.e. instead of comparing distance from relation vector to the corresponding objec node in the ground truth e.g. `(Person_19, hasAge, 30yrs)`, we learn a function from the relation vector `hasAge` applied to the person `(R odot h + r)` to the numeric literal `30`.
- Methods are analyzed and error is reported as the difference between the predicted age (using relation vector and learned function) to the age in the ground truth (in the KG)
- Methods are then compared.

## Metrics 