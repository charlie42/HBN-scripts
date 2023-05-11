import pandas as pd
import os, datetime
import json

def get_newest_non_empty_dir_in_dir(path):
    dir_names = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    non_empty_dir_names = [d for d in dir_names if len(os.listdir(path+d)) > 0]
    # Find non-empty dir with the latest timestamp, dir name format: 2023-01-05 11.03.00___first_dropped_assessment__ICU_P___other_diag_as_input__0___debug_mode__True
    
    timestamps = [d.split("___")[0] for d in non_empty_dir_names]
    timestamps = [datetime.datetime.strptime(t, "%Y-%m-%d %H.%M.%S") for t in timestamps]
    newest_dir_name = non_empty_dir_names[timestamps.index(max(timestamps))]
    return path + newest_dir_name + "/"

def read_dict_from_txt_file(path):
    return json.load(open(path))

def remove_prefix_from_diag(diag):
    if diag.startswith("Diag."):
        return diag[5:]
    else:
        return diag

path_all_features = "../diagnosis_predictor_data/reports/evaluate_original_models/"
dir_all_features = get_newest_non_empty_dir_in_dir(path_all_features) + "performance_table_all_features.csv"
path_optimal_features = "../diagnosis_predictor_data/reports/evaluate_models_on_feature_subsets/"
dir_optimal_features = get_newest_non_empty_dir_in_dir(path_optimal_features) + "auc-on-subsets-test-set-optimal-threshold.csv"

optimal_nb_features = read_dict_from_txt_file(get_newest_non_empty_dir_in_dir(path_optimal_features) + "optimal-nb-features.txt")

all_features_performance = pd.read_csv(dir_all_features)
print("all features")
print(all_features_performance)
optimal_features_performance = pd.read_csv(dir_optimal_features)
print("optimal features")
print(optimal_features_performance)

diags = [x for x in optimal_features_performance.columns if x.startswith("Diag.")]

result_table = []
for diag in diags:
    opt_nb_features = optimal_nb_features[diag]
    value_at_opt_nb_features = optimal_features_performance[optimal_features_performance["Number of features"] == opt_nb_features][diag].values[0]
    value_all_features = all_features_performance[all_features_performance["Diag"] == diag]["ROC AUC"].values[0]
    result_table.append([remove_prefix_from_diag(diag), opt_nb_features, value_all_features, value_at_opt_nb_features])

result_table = pd.DataFrame(result_table, columns=["Diagnosis", "Optimal number of features", "AUCROC with all features", "AUROC at optimal number of features"])
print(result_table)
result_table.to_csv("output/compare_auc_at_optimal_nb_features_vs_all_features.csv", index=False)