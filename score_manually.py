import pandas as pd
import os, datetime
from sklearn.metrics import roc_auc_score

# Read prepared data

def get_newest_non_empty_dir_in_dir(path):
    dir_names = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    non_empty_dir_names = [d for d in dir_names if len(os.listdir(path+d)) > 0]
    # Find non-empty dir with the latest timestamp, dir name format: 2023-01-05 11.03.00___first_dropped_assessment__ICU_P___other_diag_as_input__0___debug_mode__True
    
    timestamps = [d.split("___")[0] for d in non_empty_dir_names]
    timestamps = [datetime.datetime.strptime(t, "%Y-%m-%d %H.%M.%S") for t in timestamps]
    newest_dir_name = non_empty_dir_names[timestamps.index(max(timestamps))]
    return path + newest_dir_name + "/"

def get_list_of_analysed_diags():
    path = get_newest_non_empty_dir_in_dir("../diagnosis_predictor_data/reports/evaluate_models_on_feature_subsets/")
    report = pd.read_csv(path + "auc-on-subsets-test-set-optimal-threshold.csv")
    return [x for x in report.columns if x.startswith("Diag.")]

path = get_newest_non_empty_dir_in_dir("../diagnosis_predictor_data/data/make_dataset/")
total_scores_data = pd.read_csv(path+'total_scores_w_impairment.csv')
subscale_scores_data = pd.read_csv(path+'subscale_scores_w_impairment.csv')

diags = get_list_of_analysed_diags()
print(diags)

print("total_scores_data: ", total_scores_data.shape)
print("subscale_scores_data: ", subscale_scores_data.shape)

score_cols = ([x for x in total_scores_data.columns if not x.startswith("Diag") and not x.endswith("WAS_MISSING") and not "Barratt" in x and not "preg_symp" in x] 
    + [x for x in subscale_scores_data.columns if not x.startswith("Diag") and not x.endswith("WAS_MISSING")  and not "Barratt" in x and not "preg_symp" in x])

for diag_col in diags:
    best_auc = 0
    best_score_col = ""
    for score_col in score_cols:
        # Find best scale for diag    
        if "Total" in score_col:
            auc = roc_auc_score(total_scores_data[diag_col], total_scores_data[score_col])    
        else:
            auc = roc_auc_score(subscale_scores_data[diag_col], subscale_scores_data[score_col])
        if auc > best_auc:
            #print(auc, best_auc)
            best_auc = auc
            best_score_col = score_col
        #print(diag_col, score_col, auc)
        
    print(diag_col, best_score_col, best_auc)