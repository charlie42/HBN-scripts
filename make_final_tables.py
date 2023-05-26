import pandas as pd
import os, datetime
import json

from helpers import get_newest_non_empty_dir_in_dir

def read_data_eval_orig():
    # Read report tables to visualize
    path_eval_orig = "../diagnosis_predictor_data/reports/evaluate_original_models/"
    dir_eval_orig_all = get_newest_non_empty_dir_in_dir(path_eval_orig, ["first_assessment_to_drop", # several assessments were used as opposed to one single assessment
                                                                    "only_free_assessments__0",
                                                                    #"debug_mode__False"
                                                                    ])
    dir_eval_orig_free = get_newest_non_empty_dir_in_dir(path_eval_orig, ["first_assessment_to_drop", # several assessments were used as opposed to one single assessment
                                                                    "only_free_assessments__1",
                                                                    #"debug_mode__False"
                                                                    ])
    
    return dir_eval_orig_all, dir_eval_orig_free

def read_data_eval_subsets():

    path_eval_subsets = "../diagnosis_predictor_data/reports/evaluate_models_on_feature_subsets/"
    dir_eval_subsets_all = get_newest_non_empty_dir_in_dir(path_eval_subsets, ["first_assessment_to_drop", # several assessments were used as opposed to one single assessment
                                                                  "only_free_assessments__0",
                                                                  #"debug_mode__False"
                                                                  ])
    dir_eval_subsets_free = get_newest_non_empty_dir_in_dir(path_eval_subsets, ["first_assessment_to_drop", # several assessments were used as opposed to one single assessment
                                                                  "only_free_assessments__1",
                                                                  #"debug_mode__False"
                                                                  ])

    return dir_eval_subsets_all, dir_eval_subsets_free

def make_df_to_plot_eval_orig(dir_eval_orig_all, dir_eval_orig_free):
    # Read ROC AUC on all features on test set and for only healthy controls, for all assessments and free assessments
    eval_orig_all_df = pd.read_csv(dir_eval_orig_all + "performance_table_all_features.csv", index_col=0)
    eval_orig_free_df = pd.read_csv(dir_eval_orig_free + "performance_table_all_features.csv", index_col=0)

    # Each row is diagnosis (index), plot ROC AUC column and ROC AUC healthy controls column. 
    # Plot both for all assessments and free assessments on the same plot, add number of positive examples to each diagnosis
    eval_orig_all_df = eval_orig_all_df[["ROC AUC", "ROC AUC Healthy Controls", "# of Positive Examples"]]
    eval_orig_free_df = eval_orig_free_df[["ROC AUC", "ROC AUC Healthy Controls"]]
    eval_orig_all_df.columns = ["AUC all features all assessments", "AUC all features healthy controls all assessments", "# of Positive Examples"]
    eval_orig_free_df.columns = ["AUC all features free assessments", "AUC all features healthy controls free assessments"]
    eval_orig_df = eval_orig_all_df.merge(eval_orig_free_df, left_index=True, right_index=True).sort_values(by="AUC all features all assessments", ascending=False)

    return eval_orig_df

def make_df_to_plot_eval_subsets(dir_eval_subsets_all, dir_eval_subsets_free):
    df_opt_features_all = pd.read_csv(dir_eval_subsets_all + "auc-sens-spec-on-subsets-test-set-optimal-threshold-optimal-nb-features.csv", index_col=1)
    df_opt_features_free = pd.read_csv(dir_eval_subsets_free + "auc-sens-spec-on-subsets-test-set-optimal-threshold-optimal-nb-features.csv", index_col=1)

    df_manual = pd.read_csv("output/manual_scoring_analysis/manual_subsale_scores_vs_ml.csv", index_col=0)

    # Make one dataset with all info
    df_opt_features_all = df_opt_features_all[["AUC", "Number of features"]]
    df_opt_features_all.columns = ["AUC optimal features all assessments", "Optimal # of features all assessments"]
    df_opt_features_free = df_opt_features_free[["AUC", "Number of features"]]
    df_opt_features_free.columns = ["AUC optimal features free assessments", "Optimal # of features free assessments"]
    df_opt_features = df_opt_features_all.merge(df_opt_features_free, left_index=True, right_index=True)

    # Add manual scoring
    df_opt_features = df_opt_features.merge(df_manual, left_index=True, right_index=True).sort_values(by="Best subscale score", ascending=False)

    return df_opt_features

def main():
    dir_eval_orig_all, dir_eval_orig_free = read_data_eval_orig()
    dir_eval_subsets_all, dir_eval_subsets_free = read_data_eval_subsets()
    print("Reading reports from: ", dir_eval_orig_all, dir_eval_orig_free)
    print("Reading reports from: ", dir_eval_subsets_all, dir_eval_subsets_free)

    eval_orig_df = make_df_to_plot_eval_orig(dir_eval_orig_all, dir_eval_orig_free)
    eval_subsets_df = make_df_to_plot_eval_subsets(dir_eval_subsets_all, dir_eval_subsets_free)
    compare_orig_vs_subsets_df = eval_orig_df.merge(eval_subsets_df, left_index=True, right_index=True)

    eval_orig_df.to_csv("output/eval_orig.csv")
    eval_subsets_df.to_csv("output/eval_subsets.csv")
    compare_orig_vs_subsets_df.to_csv("output/compare_orig_vs_subsets.csv")
if __name__ == "__main__":
    main()