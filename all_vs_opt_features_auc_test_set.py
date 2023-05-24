import pandas as pd
import os, datetime
import json

from helpers import get_newest_non_empty_dir_in_dir

def read_dict_from_txt_file(path):
    return json.load(open(path))

def remove_prefix_from_diag(diag):
    if diag.startswith("Diag."):
        return diag[5:]
    else:
        return diag
    
def read_performance_table():
    path_all_features = "../diagnosis_predictor_data/reports/evaluate_original_models/"
    dir_all_features_all_assessments = get_newest_non_empty_dir_in_dir(path_all_features, ["only_free_assessments__0"]) + "performance_table_all_features.csv"
    dir_all_featrures_free_assessments = get_newest_non_empty_dir_in_dir(path_all_features, ["only_free_assessments__1"]) + "performance_table_all_features.csv"
    
    perf_all_features_all_assessments = pd.read_csv(dir_all_features_all_assessments)[["Diag", "ROC AUC"]]
    perf_all_features_free_assessments = pd.read_csv(dir_all_featrures_free_assessments)[["Diag", "ROC AUC"]]

    path_optimal_features = "../diagnosis_predictor_data/reports/evaluate_models_on_feature_subsets/"
    dir_optimal_features_all_assessments = get_newest_non_empty_dir_in_dir(path_optimal_features, ["only_free_assessments__0"]) + "auc-sens-spec-on-subsets-test-set-optimal-threshold-optimal-nb-features.csv"
    dir_optimal_features_free_assessments = get_newest_non_empty_dir_in_dir(path_optimal_features, ["only_free_assessments__1"]) + "auc-sens-spec-on-subsets-test-set-optimal-threshold-optimal-nb-features.csv"

    perf_optimal_features_all_assessments = pd.read_csv(dir_optimal_features_all_assessments, index_col=0)[["Diagnosis", "Number of features", "AUC"]]
    perf_optimal_features_free_assessments = pd.read_csv(dir_optimal_features_free_assessments, index_col=0)[["Diagnosis", "Number of features", "AUC"]]

    # Merge all, add free or all to column names
    perf_all_features_all_assessments.columns = ["Diag", "ROC AUC all features all assessments"]
    perf_all_features_free_assessments.columns = ["Diag", "ROC AUC all features free assessments"]
    perf_optimal_features_all_assessments.columns = ["Diag", "Number of features all assessments", "ROC AUC optimal features all assessments"]
    perf_optimal_features_free_assessments.columns = ["Diag", "Number of features free assessments", "ROC AUC optimal features free assessments"]

    result = perf_all_features_all_assessments.merge(perf_all_features_free_assessments, on="Diag")
    result = result.merge(perf_optimal_features_all_assessments, on="Diag")
    result = result.merge(perf_optimal_features_free_assessments, on="Diag")

    # Remove decimal points from numbers of features
    result["Number of features all assessments"] = result["Number of features all assessments"].astype(int)
    result["Number of features free assessments"] = result["Number of features free assessments"].astype(int)

    print(result)
    return result

def main():
    performance = read_performance_table()

    # Remove prefix from diagnosis names in table
    performance["Diag"] = performance["Diag"].apply(remove_prefix_from_diag)

    performance.to_csv("output/compare_auc_at_optimal_nb_features_vs_all_features.csv", float_format='%.3f', index=False)

if __name__ == "__main__":
    main()