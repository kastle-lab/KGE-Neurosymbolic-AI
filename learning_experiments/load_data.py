import numpy as np
import pandas as pd
import os
import re

def load_data(csv_path):

    # helper for sorting list according to creation order (E0, E1, E2, ..., En)
    def extract_num(e):
        m = re.search(r"(\d+)$", str(e))
        return int(m.group(1)) if m else -1

    # =====================================================
    # LOAD + SORT
    # =====================================================
    print("Loading:", csv_path)
    df = pd.read_csv(csv_path)

    # remove the E part of the entity name into new column for sorting 
    df["entity_num"] = df["entity"].apply(extract_num)

    # sort entities based on labels
    df = df.sort_values("entity_num").reset_index(drop=True)

    # remove the entity label from dataframe after sort
    df = df.drop(columns=["entity_num"])

    # grab initialization values from dataframe
    init_vals = df["initial_value"].values

    #grab entites labels
    entity_labels = df["entity"].values
    
    # grab embedding matrix
    E = df.drop(columns=["entity", "initial_value"]).values

    # output embedding matrix shape
    N, D = E.shape
    print(f"Embeddings loaded: N={N}, D={D}")

    # calculate the number of triplets that can be analyzed
    num_triplets = N // 3

    # grab triplets starting at zeroth index
    triplets = [(3*t, 3*t+1, 3*t+2) for t in range(num_triplets)]

    # build X/Y
    X = []
    Y = []
    
    for (i0, i1, i2) in triplets:
        X.append(np.concatenate([E[i0], E[i1]]))
        Y.append(E[i2])
        
    # X = a matrix concat of E0 and E1
    E_parents = np.array(X)

    # Y = our 'average' E2 (initialized with the average values of E0 and E1)
    E3_mean = np.array(Y)
    
    E = np.array(E)
    
    return E, E_parents, E3_mean, entity_labels, init_vals, 