import pandas as pd
import os, datetime
from sklearn.metrics import roc_auc_score

def get_newest_non_empty_dir_in_dir(path):
    dir_names = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    non_empty_dir_names = [d for d in dir_names if len(os.listdir(path+d)) > 0]
    # Filter only those with first_dropped_assessment in name (not single assessment)
    non_empty_dir_names = [d for d in non_empty_dir_names if "first_dropped_assessment" in d]
    # Find non-empty dir with the latest timestamp, dir name format: 2023-01-05 11.03.00___first_dropped_assessment__ICU_P___other_diag_as_input__0___debug_mode__True
    
    timestamps = [d.split("___")[0] for d in non_empty_dir_names]
    timestamps = [datetime.datetime.strptime(t, "%Y-%m-%d %H.%M.%S") for t in timestamps]
    newest_dir_name = non_empty_dir_names[timestamps.index(max(timestamps))]
    return path + newest_dir_name + "/"

def get_list_of_analysed_diags():
    path = get_newest_non_empty_dir_in_dir("../../diagnosis_predictor_data/reports/evaluate_models_on_feature_subsets/")
    report = pd.read_csv(path + "auc-on-subsets-test-set-optimal-threshold.csv")
    return [x for x in report.columns if x.startswith("Diag.")]

def find_auc_for_score_col(score_col):
    if "Total" in score_col:
        auc = roc_auc_score(total_scores_data[diag_col], total_scores_data[score_col])    
    else:
        auc = roc_auc_score(subscale_scores_data[diag_col], subscale_scores_data[score_col])
    return auc

# Read prepared data

path = get_newest_non_empty_dir_in_dir("../../diagnosis_predictor_data/data/make_dataset/")
total_scores_data = pd.read_csv(path+'total_scores.csv')
subscale_scores_data = pd.read_csv(path+'subscale_scores.csv')

diags = get_list_of_analysed_diags()
print(diags)

print("total_scores_data: ", total_scores_data.shape)
print("subscale_scores_data: ", subscale_scores_data.shape)

score_cols = ([x for x in total_scores_data.columns if not x.startswith("Diag") and not x.endswith("WAS_MISSING") and not "Barratt" in x and not "preg_symp" in x] 
    + [x for x in subscale_scores_data.columns if not x.startswith("Diag") and not x.endswith("WAS_MISSING")  and not "Barratt" in x and not "preg_symp" in x])
print(score_cols)

# Get manual AUC scores for each diag and each scale/subscale

best_score = []
all_scores = {}
for diag_col in diags:
    scores_for_diag = []
    for score_col in score_cols:
        auc = find_auc_for_score_col(score_col)
        scores_for_diag.append(auc)
    all_scores[diag_col] = scores_for_diag

all_scores_df = pd.DataFrame.from_dict(all_scores, orient='index', columns=score_cols)
print(all_scores_df)

# Make df with best scores for each diag (cols: Diag,Best score,AUC) 

best_scores = []
for diag_col in diags:
    best_score = all_scores_df.loc[diag_col].max()
    best_score_col = all_scores_df.loc[diag_col].idxmax()
    best_scores.append([diag_col, best_score_col, best_score])
best_scores_df = pd.DataFrame(best_scores, columns=["Diag","Best score","AUC"])
print(best_scores_df)

best_scores_df.to_csv("output/best_scores.csv", index=False)

# Compare with ML scores

path = get_newest_non_empty_dir_in_dir("../../diagnosis_predictor_data/reports/evaluate_models_on_feature_subsets/")
print("Reading report from: ", path)
ml_scores = pd.read_csv(path + "auc-on-subsets-test-set-optimal-threshold.csv")

numbers_of_items = {"SCQ,SCQ_Total": 40,
                    "ASSQ,ASSQ_Total": 27,
                    "ARI_P,ARI_P_Total_Score": 7,
                    "SWAN,SWAN_Total": 18,
                    "SRS,SRS_Total_T": 65,
                    "CBCL,CBCL_Total_T": 112,
                    "SWAN,SWAN_HY": 9,
                    "SWAN,SWAN_IN": 9}

ml_scores_at_num_features = {}
for diag_col in diags:
    best_score = best_scores_df[best_scores_df["Diag"] == diag_col]["AUC"].values[0]
    best_score_subscale = best_scores_df[best_scores_df["Diag"] == diag_col]["Best score"].values[0]
    number_of_items_in_best_subscale = numbers_of_items[best_score_subscale] if best_score_subscale in numbers_of_items else 10
    ml_score_at_number_of_items_of_best_subscale = ml_scores[ml_scores["Number of features"] == number_of_items_in_best_subscale][diag_col].values[0]
    # Find number of items needed to reach performance of the best subscale    
    number_of_items_for_ml_score_of_best_subscale = ml_scores[ml_scores[diag_col] >= best_score]["Number of features"].min()
    ml_scores_at_num_features[diag_col] = [best_score_subscale, best_score, number_of_items_in_best_subscale, ml_score_at_number_of_items_of_best_subscale, number_of_items_for_ml_score_of_best_subscale]

ml_scores_at_num_features_df = pd.DataFrame.from_dict(ml_scores_at_num_features, orient='index', columns=["Best subscale", "Best subscale score", "Number of items", "ML score", "Number of items to reach best subscale"])

all_scores_df.T.to_csv("output/subscale_scores.csv")
ml_scores_at_num_features_df.to_csv("output/subsale_scores_vs_ml.csv")