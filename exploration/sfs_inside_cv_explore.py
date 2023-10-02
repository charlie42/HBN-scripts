import joblib
import numpy as np
import pandas as pd
import yaml

if __name__ == '__main__':
    cv_dict = joblib.load("input/rs_objects_lr_debug.joblib")
    print(cv_dict)
    

    # How mayn features "model" step uses?

    # print(type(cv_dict))
    # print(cv_dict.keys())
    # key = list(cv_dict.keys())[1]
    # print(cv_dict[key].keys())
    # print(cv_dict[key]["test_roc_auc"])
    # print(cv_dict[key]["test_sensitivity"])
    # print(cv_dict[key]["test_specificity"])
    # print(type(cv_dict[key]["estimator"][0]))
    # print(cv_dict[key]["estimator"][0].best_estimator_.named_steps.keys())
    # print(type(cv_dict[key]["estimator"][0].best_estimator_.named_steps["selector"]))
    # print(cv_dict[key]["estimator"][0].best_estimator_.named_steps["selector"].get_metric_dict(confidence_interval=0.95))
    # sfs = cv_dict[key]["estimator"][0].best_estimator_.named_steps["selector"]

    # # Get subsets
    # subsets = sfs.subsets_
    # print(subsets)

    #print()

    #for key in cv_dict.keys():
        #print(key)
        #print(cv_dict[key].keys())
        #print(cv_dict[key]["test_score"])
        #print(cv_dict[key]["estimator"])