import pandas as pd
import numpy as np
import os, datetime
import json
import math
from joblib import load

import dsutils
from dsutils.file_utils import write_dict_of_dfs_to_csv

import sys, os, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from data_reading import DataReader

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

DIAGNOSIS_DICT = {
    'Diag.Major Depressive Disorder': 'MDD',
    'Diag.Autism Spectrum Disorder': 'ASD',
    'Diag.Enuresis': 'Enuresis',
    'Diag.ADHD-Combined Type': 'ADHD-C',
    'Diag.Social Anxiety (Social Phobia)': 'SAD',
    'Diag.Generalized Anxiety Disorder': 'GAD',
    'Diag.Oppositional Defiant Disorder': 'ODD',
    'Diag.Any Diag': 'Any',
    'Diag.No Diagnosis Given': 'None',
    'Diag.Separation Anxiety': 'SA',
    'Diag.ADHD-Inattentive Type': 'ADHD-I',
    'Diag.Specific Learning Disorder with Impairment in Mathematics': 'SLD-Math',
    'Diag.Specific Learning Disorder with Impairment in Mathematics (test)': 'SLD-Math (test)',
    'Diag.Language Disorder': 'Language',
    'Diag.Specific Phobia': 'Phobia',
    'Diag.Specific Learning Disorder with Impairment in Reading': 'SLD-Reading',
    'Diag.Specific Learning Disorder with Impairment in Written Expression': 'SLD-Writing',
    'Diag.Specific Learning Disorder with Impairment in Reading (test)': 'SLD-Reading (test)',
    'Diag.Specific Learning Disorder with Impairment in Written Expression (test)': 'SLD-Writing (test)',
    'Diag.Other Specified Anxiety Disorder': 'Other Anxiety',
    'Diag.Processing Speed Deficit (test)': 'PS (test)',
    'Diag.Borderline Intellectual Functioning (test)': 'BIF (test)',
    'Diag.Intellectual Disability-Borderline (test)': 'BIF (test)',
    'Diag.NVLD (test)': 'NVLD (test)',
    'Diag.NVLD without reading condition (test)': 'NVLD no read (test)',
}

def parse_perf_data(data_dict):
    #print(data_dict)
    #print(data_dict.keys())

    for diag in data_dict:
        #print(diag)
        #print(data_dict[diag].keys())
        pass
    

def make_dfs_for_saturation_plots(perf_dict):
    """
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


def make_df_for_manual_vs_cv_ml(perf_dict, manual_scoring_df):
    """
    Make dataframe for with CV ML scors for each diagnosis
    from performance dict. 
    Dict format:
    diag: {
        "perf_on_features": {
            n_features: {
                "auc": [float],
            }
        }
    }
    Output format:
    Columns: 
        - "AUROC"
    Index:
        - Diagnosis
    """
    
    diags = perf_dict.keys()
    dict_for_df = []
    n_checked = max(perf_dict[list(diags)[0]]["perf_on_features"])

    for diag in perf_dict:
        if diag in manual_scoring_df.index:
            n = manual_scoring_df.loc[diag, "N in best scale"]
            best_scale = manual_scoring_df.loc[diag, "Best scale"]
            score_of_best_scale = manual_scoring_df.loc[diag, "AUC"]

            if n > n_checked:
                n = n_checked

            row = {
                "Diag": diag,
                "N in best scale": n,
                "Best scale": best_scale,
                "Score of best scale": score_of_best_scale,
                "CV ML scores at N": perf_dict[diag]["perf_on_features"][n]["auc"],
                "CV sum scores at N": perf_dict[diag]["perf_on_features"][n]["auc_sum_score"],
                "Mean ML score at N": np.mean(perf_dict[diag]["perf_on_features"][n]["auc"]),
                "Median ML score at N": np.median(perf_dict[diag]["perf_on_features"][n]["auc"]),
                "Std ML score at N": np.std(perf_dict[diag]["perf_on_features"][n]["auc"]),
                "Mean sum score at N": np.mean(perf_dict[diag]["perf_on_features"][n]["auc_sum_score"]),
                "Min ML score at N": np.min(perf_dict[diag]["perf_on_features"][n]["auc"]),
                "Min sum score at N": np.min(perf_dict[diag]["perf_on_features"][n]["auc_sum_score"]),
            }
            dict_for_df.append(row)    
    
    df = pd.DataFrame.from_dict(dict_for_df).set_index("Diag")
    df["Delta ML"] = df["Mean ML score at N"] - df["Score of best scale"]
    #df = df.sort_values("Median ML score at N")
    df = df.sort_values("Delta ML")

    # DEBUG
    print("AVERAGE SD: ", df["Std ML score at N"].mean())

    return df

def make_sens_spec_table(perf_dict):

    print("DEBUG")
    print("/DEBUG")

    print("DEBUG sens spec cv")
    for diag in perf_dict:
        n = math.ceil(np.mean(perf_dict[diag]["opt_ns"]))
        opt_thresh_last = perf_dict[diag]["perf_on_features"][n]["opt_thresh"][-1]
        print(diag)
        #if diag == "Diag.Generalized Anxiety Disorder":
            #print(diag, perf_dict[diag]["perf_on_features"][10]["opt_thresh"])
            #print(perf_dict[diag]["n_positives"])
            #print("Last fold opt threshold: ", opt_thresh_last)
        takeClosest = lambda num,collection:min(collection,key=lambda x:abs(x-num))
        thresh = 0.2
        opt_thresh = takeClosest(np.mean(perf_dict[diag]["perf_on_features"][n]["opt_thresh"]), perf_dict[diag]["perf_on_features"][n]["spec_sens_dict"][0].keys())
        #opt_thresh = np.quantile(perf_dict[diag]["perf_on_features"][n]["opt_thresh"], 0.5)
        print("opt thresh", opt_thresh)
        #print(perf_dict[diag]["perf_on_features"][n]["spec_sens_dict"][0].keys())
        
        thresh =  opt_thresh
        for i in range(0, len(perf_dict[diag]["perf_on_features"][n]["spec_sens_dict"])):
            print(
                i, " FOLD: ", 
                "\nsens spec at avg opt thresh", perf_dict[diag]["perf_on_features"][n]["spec_sens_dict"][i][thresh], 
                "\nopt thresh for this fold", perf_dict[diag]["perf_on_features"][n]["opt_thresh"][i],
                #"\nsens spec at this fold opt thresh", perf_dict[diag]["perf_on_features"][n]["spec_sens_dict"][i][perf_dict[diag]["perf_on_features"][n]["opt_thresh"][i]], 
                #"\nsens spec dict", perf_dict[diag]["perf_on_features"][n]["spec_sens_dict"][i],
                "\n n positives in this fold", perf_dict[diag]["n_positives"][i]["y_test"],
                #"\n", perf_dict[diag]["perf_on_features"][n]["auc"][i]
                )
        #print("TEST SET: ", perf_dict[diag]["sens_test_set"], perf_dict[diag]["spec_test_set"])
        #print("LAST FOLD: ", perf_dict[diag]["perf_on_features"][n]["spec_sens_dict"][-1][0.5])
        print(
            "TEST SET: ", 
            #perf_dict[diag]["sens_spec_test_set"],
            perf_dict[diag]["sens_spec_test_set"][thresh],
            #perf_dict[diag]["auc_test_set"]
            #"\n n positives test set", perf_dict[diag]["n_positives_test_set"]
        )

        


    rows = []
    for diag in perf_dict:
        n = math.ceil(np.mean(perf_dict[diag]["opt_ns"]))
        auc = perf_dict[diag]["auc_test_set"]
        sens = perf_dict[diag]["sens_test_set"]
        spec = perf_dict[diag]["spec_test_set"]
        row = [diag, n, auc, sens, spec]
        rows.append(row)
    result_df = pd.DataFrame(
        rows, 
        columns=["Diag", "N features", "AUC", "Sensitivity", "Specificity"]
    ).sort_values("AUC", ascending=False) 
    return result_df

if __name__ == "__main__":

    #### Original Assessment Set ####
    
    # Read data
    perf_dict = load("input/scores_objects_lr_debug_false_non_learning.joblib")
    
    # Make tables for saturation plots
    dfs_for_saturation_plots = make_dfs_for_saturation_plots(perf_dict)
    write_dict_of_dfs_to_csv(dfs_for_saturation_plots, "output/cv/saturation_dfs", index=True)

    # Make tables for mean manual vs ML
    data_reader = DataReader()
    manual_scoring_df = data_reader.read_data(data_type="manual_scoring", only_manual=True).set_index("Diag")

    # Make table for cv ML
    manual_vs_cv_ml_df = make_df_for_manual_vs_cv_ml(perf_dict, manual_scoring_df)
    manual_vs_cv_ml_df = manual_vs_cv_ml_df.rename(index=DIAGNOSIS_DICT)
    manual_vs_cv_ml_df.to_csv("output/cv/manual_vs_cv_ml.csv", float_format='%.3f')

    sens_spec_table = make_sens_spec_table(perf_dict)
    sens_spec_table.to_csv("output/cv/sens_spec.csv", float_format='%.3f')

    #### Learning Assessment Set ####
    # Read data
    perf_dict = load("input/scores_objects_lr_debug_false_learning.joblib")
    
    # Make tables for saturation plots
    dfs_for_saturation_plots = make_dfs_for_saturation_plots(perf_dict)
    write_dict_of_dfs_to_csv(dfs_for_saturation_plots, "output/cv/saturation_dfs_learning", index=True)

    # Make tables for mean manual vs ML
    data_reader = DataReader()
    manual_scoring_df = data_reader.read_data(data_type="manual_scoring", only_manual=True).set_index("Diag")

    # Make table for cv ML
    manual_vs_cv_ml_df = make_df_for_manual_vs_cv_ml(perf_dict, manual_scoring_df)
    manual_vs_cv_ml_df = manual_vs_cv_ml_df.rename(index=DIAGNOSIS_DICT)
    manual_vs_cv_ml_df.to_csv("output/cv/manual_vs_cv_ml_learning.csv", float_format='%.3f')

    sens_spec_table = make_sens_spec_table(perf_dict)
    sens_spec_table.to_csv("output/cv/sens_spec_learning.csv", float_format='%.3f')

    #### Non-proprietary Assessment Set ####
    # Read data
    perf_dict = load("input/scores_objects_lr_debug_false_free.joblib")
    
    # Make tables for saturation plots
    dfs_for_saturation_plots = make_dfs_for_saturation_plots(perf_dict)
    write_dict_of_dfs_to_csv(dfs_for_saturation_plots, "output/cv/saturation_dfs_free", index=True)

    # Make tables for mean manual vs ML
    data_reader = DataReader()
    manual_scoring_df = data_reader.read_data(data_type="manual_scoring", only_manual=True).set_index("Diag")

    # Make table for cv ML
    manual_vs_cv_ml_df = make_df_for_manual_vs_cv_ml(perf_dict, manual_scoring_df)
    manual_vs_cv_ml_df = manual_vs_cv_ml_df.rename(index=DIAGNOSIS_DICT)
    manual_vs_cv_ml_df.to_csv("output/cv/manual_vs_cv_ml_free.csv", float_format='%.3f')

    sens_spec_table = make_sens_spec_table(perf_dict)
    sens_spec_table.to_csv("output/cv/sens_spec_free.csv", float_format='%.3f')