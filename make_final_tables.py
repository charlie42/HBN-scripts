import pandas as pd
import os, datetime
import json

from data_reading import DataReader

def make_df_to_plot_eval_orig(df, df_free = None):
    
    # Each row is diagnosis (index), plot ROC AUC column and ROC AUC healthy controls column. 
    # Plot both for all assessments and free assessments on the same plot, add number of positive examples to each diagnosis
    df = df[["ROC AUC", "ROC AUC Mean CV", "ROC AUC Healthy Controls", "# of Positive Examples"]]
    df.columns = ["AUC all features all assessments", "AUC CV all features all assessments", "AUC all features healthy controls all assessments", "# of Positive Examples all assessments"]
    
    if df_free is not None:
        df_free = df_free[["ROC AUC", "ROC AUC Mean CV", "ROC AUC Healthy Controls", "# of Positive Examples"]]
        df_free.columns = ["AUC all features free assessments", "AUC CV all features free assessments", "AUC all features healthy controls free assessments", "# of Positive Examples free assessments"]
        
        df = df.merge(df_free, left_index=True, right_index=True).sort_values(by="AUC CV all features all assessments", ascending=False)
    
    df = df.sort_values(by="AUC CV all features all assessments", ascending=False)

    return df

def make_df_to_plot_eval_subsets(df_manual, df, df_free = None):

    # Make one dataset with all info
    df = df[["AUC", "Number of features"]]
    df.columns = ["AUC optimal features all assessments", "Optimal # of features all assessments"]
    
    if df_free is not None:
        df_free = df_free[["AUC", "Number of features"]]
        df_free.columns = ["AUC optimal features free assessments", "Optimal # of features free assessments"]

        df = df.merge(df_free, left_index=True, right_index=True)

    # Add manual scoring if consensus diag (not learning)
    if "(test)" not in df.index[0]:
        df = df.merge(df_manual, left_index=True, right_index=True).sort_values(by="Best subscale score", ascending=False)

    return df

def make_df_ds_stats(df, df_free = None):

    df = df.T[["n_rows_full_ds", "n_input_cols"]]
    df.columns = ["# rows full dataset all assessments", "# input features all assessments"]

    if df_free is not None:
        df_free = df_free.T[["n_rows_full_ds", "n_input_cols"]]
        df_free.columns = ["# rows full dataset free assessments", "# input features free assessments"]
        #Merge
        df = df.merge(df_free, left_index=True, right_index=True)
    
    return df

def find_max_n_features(df_all):
    # Find max number of features (max in index)
    return df_all.index.max()

def make_df_learning_improvements(df_all, df_learning_NIH, df_learning_no_NIH, all_diags = True):
    # Get row with index 27 for each diag (columns are diag names) for all dataframes, merge into one, index = diags, columns = "original (df_all), more assessments (df_learning_no_NIH), NIH (df_learning_NIH)"
    max_n_features = find_max_n_features(df_all)

    df_all = df_all.loc[max_n_features]
    df_learning_NIH = df_learning_NIH.loc[max_n_features]
    df_learning_no_NIH = df_learning_no_NIH.loc[max_n_features]

    df = pd.concat([df_all, df_learning_no_NIH, df_learning_NIH], axis=1)
    df.columns = ["original", "more assessments", "NIH"]
    df = df.sort_values(by="original", ascending=False)

    if not all_diags: # Only get test-based diags (have "(test)" in the name)
        df = df[df.index.str.contains("(test)")]

    return df


def make_dfs(data_reader):
    # Read data
    dfs = {}

    # Consensus and learning
    for data_type in ["item_lvl", "eval_orig", "eval_subsets", "make_ds", "eval_subsets_one_subset"]:
        file_filter = "eval_orig_test_set_file" if data_type == "eval_orig" else ""
        dfs[data_type+"_consensus_and_learning_all"] = data_reader.read_data(data_type = data_type, 
                                                params = ["multiple_assessments", "all_assessments", "learning_and_consensus_diags"],
                                                file_filter = file_filter)
        dfs[data_type+"_consensus_and_learning_free"] = data_reader.read_data(data_type = data_type, 
                                                params = ["multiple_assessments", "free_assessments", "learning_and_consensus_diags"],
                                                file_filter = file_filter)
    # Learning
    for data_type in ["eval_subsets_one_subset"]:
        dfs[data_type+"_learning_NIH"] = data_reader.read_data(data_type = data_type, 
                                                params = ["multiple_assessments", "all_assessments", "only_learning_diags", "NIH"],
                                                file_filter = file_filter)
        dfs[data_type+"_learning_no_NIH"] = data_reader.read_data(data_type = data_type, 
                                                params = ["multiple_assessments", "all_assessments", "only_learning_diags", "no_NIH"],
                                                file_filter = file_filter)
    
    # Manual scoring
    dfs["manual_scoring"] = data_reader.read_data(data_type="manual_scoring")

    return dfs

if __name__ == "__main__":

    ### Consensus diags ###

    data_reader = DataReader()

    # Read data
    dfs = make_dfs(data_reader)

    eval_orig_df = make_df_to_plot_eval_orig(dfs["eval_orig_consensus_and_learning_all"], dfs["eval_orig_consensus_and_learning_free"])
    eval_subsets_df = make_df_to_plot_eval_subsets(dfs["manual_scoring"], dfs["eval_subsets_consensus_and_learning_all"], dfs["eval_subsets_consensus_and_learning_free"])
    ds_stats_df = make_df_ds_stats(dfs["make_ds_consensus_and_learning_all"], dfs["make_ds_consensus_and_learning_free"])
    
    compare_orig_vs_subsets_df = eval_orig_df.merge(eval_subsets_df, left_index=True, right_index=True)
    
    # Add total # of features and examples to compare_orig_vs_subsets_df
    for col in ds_stats_df.columns:
        compare_orig_vs_subsets_df[col] = ds_stats_df[col].values[0]
    

    eval_orig_df.to_csv("output/eval_orig.csv")
    eval_subsets_df.to_csv("output/eval_subsets.csv")
    compare_orig_vs_subsets_df.to_csv("output/compare_orig_vs_subsets.csv")

    ### Learning diags improvements ###
    learning_improvement_df = make_df_learning_improvements(dfs["eval_subsets_one_subset_consensus_and_learning_all"], dfs["eval_subsets_one_subset_learning_NIH"], dfs["eval_subsets_one_subset_learning_no_NIH"], all_diags = False)
    learning_improvement_all_diags_df = make_df_learning_improvements(dfs["eval_subsets_one_subset_consensus_and_learning_all"], dfs["eval_subsets_one_subset_learning_NIH"], dfs["eval_subsets_one_subset_learning_no_NIH"], all_diags = True)
    
    learning_improvement_df.to_csv("output/learning_improvements.csv")
    learning_improvement_all_diags_df.to_csv("output/learning_improvements_all_diags.csv")
