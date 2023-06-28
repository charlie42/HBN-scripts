import pandas as pd
import os, datetime
import json

from helpers import get_newest_non_empty_dir_in_dir

def get_data_dir(dir, params, file_filter = None):
    # Read report tables to visualize
    print("Reading input data from: ", dir, " with params: ", params, " and file filters: ", file_filter)

    path = "../diagnosis_predictor_data/reports/" + dir + "/"
    dir = get_newest_non_empty_dir_in_dir(path, params, file_filter)
    
    return dir

def make_df_to_plot_eval_orig(dir_eval_orig_all, dir_eval_orig_free = None):
    # Read ROC AUC on all features on test set and for only healthy controls, for all assessments and free assessments
    eval_orig_all_df = pd.read_csv(dir_eval_orig_all + "performance_table_all_features.csv", index_col=0)
    # Each row is diagnosis (index), plot ROC AUC column and ROC AUC healthy controls column. 
    # Plot both for all assessments and free assessments on the same plot, add number of positive examples to each diagnosis
    eval_orig_all_df = eval_orig_all_df[["ROC AUC", "ROC AUC Mean CV", "ROC AUC Healthy Controls", "# of Positive Examples"]]
    eval_orig_all_df.columns = ["AUC all features all assessments", "AUC CV all features all assessments", "AUC all features healthy controls all assessments", "# of Positive Examples all assessments"]
    
    if dir_eval_orig_free:
        eval_orig_free_df = pd.read_csv(dir_eval_orig_free + "performance_table_all_features.csv", index_col=0)
        eval_orig_free_df = eval_orig_free_df[["ROC AUC", "ROC AUC Mean CV", "ROC AUC Healthy Controls", "# of Positive Examples"]]
        eval_orig_free_df.columns = ["AUC all features free assessments", "AUC CV all features free assessments", "AUC all features healthy controls free assessments", "# of Positive Examples free assessments"]
        
        eval_orig_df = eval_orig_all_df.merge(eval_orig_free_df, left_index=True, right_index=True).sort_values(by="AUC CV all features all assessments", ascending=False)
    else:
        eval_orig_df = eval_orig_all_df.sort_values(by="AUC CV all features all assessments", ascending=False)

    return eval_orig_df

def make_df_to_plot_eval_subsets(dir_eval_subsets_all, dir_eval_subsets_free = None):
    df_opt_features_all = pd.read_csv(dir_eval_subsets_all + "auc-sens-spec-on-subsets-test-set-optimal-threshold-optimal-nb-features.csv", index_col=1)
    
    df_manual = pd.read_csv("output/manual_scoring_analysis/manual_subsale_scores_vs_ml.csv", index_col=0)

    # Make one dataset with all info
    df_opt_features_all = df_opt_features_all[["AUC", "Number of features"]]
    df_opt_features_all.columns = ["AUC optimal features all assessments", "Optimal # of features all assessments"]
    
    if dir_eval_subsets_free:
        df_opt_features_free = pd.read_csv(dir_eval_subsets_free + "auc-sens-spec-on-subsets-test-set-optimal-threshold-optimal-nb-features.csv", index_col=1)
        df_opt_features_free = df_opt_features_free[["AUC", "Number of features"]]
        df_opt_features_free.columns = ["AUC optimal features free assessments", "Optimal # of features free assessments"]

        df_opt_features = df_opt_features_all.merge(df_opt_features_free, left_index=True, right_index=True)
    else:
        df_opt_features = df_opt_features_all
    
    print("df_opt_features", df_opt_features, "df_manual", df_manual)
    # Add manual scoring if consensus diag (not learning)
    if "(test)" not in df_opt_features.index[0]:
        df_opt_features = df_opt_features.merge(df_manual, left_index=True, right_index=True).sort_values(by="Best subscale score", ascending=False)

    return df_opt_features

def make_df_ds_stats(dir_all, dir_free = None):
    
    ds_stats_df_all_assessments = pd.read_csv(dir_all + "dataset_stats.csv", index_col=0)

    ds_stats_df_all_assessments = ds_stats_df_all_assessments.T[["n_rows_full_ds", "n_input_cols"]]
    ds_stats_df_all_assessments.columns = ["# rows full dataset all assessments", "# input features all assessments"]

    if dir_free:
        ds_stats_df_free_assessments = pd.read_csv(dir_free + "dataset_stats.csv", index_col=0)
        ds_stats_df_free_assessments = ds_stats_df_free_assessments.T[["n_rows_full_ds", "n_input_cols"]]
        ds_stats_df_free_assessments.columns = ["# rows full dataset free assessments", "# input features free assessments"]
        #Merge
        ds_stats_df = ds_stats_df_all_assessments.merge(ds_stats_df_free_assessments, left_index=True, right_index=True)
    else:
        ds_stats_df = ds_stats_df_all_assessments

    return ds_stats_df

report_dirs = {
    "eval_orig": "evaluate_original_models",
    "eval_subsets": "evaluate_models_on_feature_subsets",
    "make_ds": "create_datasets",
}

params = {
    "multiple_assessments": "first_assessment_to_drop",
    "all_assessments": "only_free_assessments__0",
    "free_assessments": "only_free_assessments__1",
    "learning_diags": "learning?__1",
    "consensus_diags": "learning?__0",
}

file_filters = {
    "stats_file": "dataset_stats.csv"
}

data_paths = {
    "eval_orig": get_data_dir(
        dir = report_dirs["eval_orig"],
        params = [params["multiple_assessments"], params["all_assessments"], params["consensus_diags"]]
    ),
    "eval_subsets": get_data_dir(
        dir = report_dirs["eval_subsets"],
        params = [params["multiple_assessments"], params["all_assessments"], params["consensus_diags"]]
    ),
    "make_ds": get_data_dir(
        dir = report_dirs["make_ds"],
        params = [params["multiple_assessments"], params["all_assessments"], params["consensus_diags"]],
        file_filter = file_filters["stats_file"]
    ),
    "eval_orig_learning": get_data_dir(
        dir = report_dirs["eval_orig"],
        params = [params["multiple_assessments"], params["all_assessments"], params["learning_diags"]]
    ),
    "eval_subsets_learning": get_data_dir(
        dir = report_dirs["eval_subsets"],
        params = [params["multiple_assessments"], params["all_assessments"], params["learning_diags"]]
    ),
    "make_ds_learning": get_data_dir(
        dir = report_dirs["make_ds"],
        params = [params["multiple_assessments"], params["all_assessments"], params["learning_diags"]],
        file_filter = file_filters["stats_file"]
    ),
    "eval_orig_learning_free": get_data_dir(
        dir = report_dirs["eval_orig"],
        params = [params["multiple_assessments"], params["free_assessments"], params["learning_diags"]]
    ),
    "eval_subsets_learning_free": get_data_dir(
        dir = report_dirs["eval_subsets"],
        params = [params["multiple_assessments"], params["free_assessments"], params["learning_diags"]]
    ),
    "make_ds_learning_free": get_data_dir(
        dir = report_dirs["make_ds"],
        params = [params["multiple_assessments"], params["free_assessments"], params["learning_diags"]],
        file_filter = file_filters["stats_file"]
    ),
}

def main():
    ### Consensus digas ###

    eval_orig_df = make_df_to_plot_eval_orig(data_paths["eval_orig"])
    
    eval_subsets_df = make_df_to_plot_eval_subsets(data_paths["eval_subsets"])
    
    ds_stats_df = make_df_ds_stats(data_paths["make_ds"])
    
    
    compare_orig_vs_subsets_df = eval_orig_df.merge(eval_subsets_df, left_index=True, right_index=True)
    
    # Add total # of features and examples to compare_orig_vs_subsets_df
    for col in ds_stats_df.columns:
        compare_orig_vs_subsets_df[col] = ds_stats_df[col].values[0]
    

    eval_orig_df.to_csv("output/eval_orig.csv")
    eval_subsets_df.to_csv("output/eval_subsets.csv")
    compare_orig_vs_subsets_df.to_csv("output/compare_orig_vs_subsets.csv")

    ### Learning diags ###

    eval_orig_learning_df = make_df_to_plot_eval_orig(data_paths["eval_orig_learning"], data_paths["eval_orig_learning_free"])
    
    eval_subsets_learning_df = make_df_to_plot_eval_subsets(data_paths["eval_subsets_learning"], data_paths["eval_subsets_learning_free"])
    print("eval_subsets_learning_df", eval_subsets_learning_df)
    ds_stats_learning_df = make_df_ds_stats(data_paths["make_ds_learning"], data_paths["make_ds_learning_free"])

    compare_orig_vs_subsets_learning_df = eval_orig_learning_df.merge(eval_subsets_learning_df, left_index=True, right_index=True)
    # Add total # of features and examples to compare_orig_vs_subsets_df
    for col in ds_stats_learning_df.columns:
        compare_orig_vs_subsets_learning_df[col] = ds_stats_learning_df[col].values[0]
    print("compare_orig_vs_subsets_learning_df", compare_orig_vs_subsets_learning_df)

    eval_orig_learning_df.to_csv("output/eval_orig_learning.csv")
    eval_subsets_learning_df.to_csv("output/eval_subsets_learning.csv")
    compare_orig_vs_subsets_learning_df.to_csv("output/compare_orig_vs_subsets_learning.csv")


if __name__ == "__main__":
    main()