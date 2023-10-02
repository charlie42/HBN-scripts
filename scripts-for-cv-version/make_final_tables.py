import pandas as pd
import numpy as np
import os, datetime
import json
from joblib import load

import dsutils
from dsutils.file_utils import write_dict_of_dfs_to_csv


def make_dfs_for_saturation_plots(perf_dict):
    """"
    Make dataframes for saturation plots (learning curves) for each diag
    from performance dict. 
    Dict format:
    diag: {
        "perf_on_features": {
            n_features: {
                "auc": float,
            }
        }
    }
    Output format:
    Columns: 
        - "Number of features" (from max to min)
        - "All assessments" (auc on n features)
    """
    
    diags = perf_dict.keys()

    dfs = {}

    for diag in diags:
        dfs[diag] = pd.DataFrame()
        for n_features in perf_dict[diag]["perf_on_features"].keys():
            dfs[diag].loc[n_features, "All assessments"] = np.mean(perf_dict[diag]["perf_on_features"][n_features]["auc"])
        dfs[diag].index.name = "Number of features"
        dfs[diag] = dfs[diag].sort_index(ascending=False)

    return dfs

if __name__ == "__main__":

    ### Consensus diags ###

    # Read data
    perf_dict = load("input/scores_objects_lr_debug.joblib")

    dfs_for_saturation_plots = make_dfs_for_saturation_plots(perf_dict)
    print(dfs_for_saturation_plots)
    write_dict_of_dfs_to_csv(dfs_for_saturation_plots, "output/cv/saturation_dfs", index=True)