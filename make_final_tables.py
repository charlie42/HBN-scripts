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

def read_data_eval_orig_learning():
    # Read report tables to visualize
    path_eval_orig = "../learning_diagnosis_predictor_data/reports/evaluate_original_models/"

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

def read_data_eval_subsets_learning():
    path_eval_subsets = "../learning_diagnosis_predictor_data/reports/evaluate_models_on_feature_subsets/"

    dir_eval_subsets_all = get_newest_non_empty_dir_in_dir(path_eval_subsets, ["first_assessment_to_drop", # several assessments were used as opposed to one single assessment
                                                                  "only_free_assessments__0",
                                                                  #"debug_mode__False"
                                                                  ])
    dir_eval_subsets_free = get_newest_non_empty_dir_in_dir(path_eval_subsets, ["first_assessment_to_drop", # several assessments were used as opposed to one single assessment
                                                                  "only_free_assessments__1", 
                                                                  #"debug_mode__False"
                                                                  ])

    return dir_eval_subsets_all, dir_eval_subsets_free

def read_data_make_ds():
    
    path = "../diagnosis_predictor_data/reports/create_datasets/"

    dir_all = get_newest_non_empty_dir_in_dir(path, ["first_assessment_to_drop", # several assessments were used as opposed to one single assessment
                                                        "only_free_assessments__0",
                                                        #"debug_mode__False"
                                                        ], 
                                                        "dataset_stats.csv") # Need to check if this file was created (some dirs were generated by
                                                                            # running the script with only_assessment_distribution=1, which does not create this file)
    dir_free = get_newest_non_empty_dir_in_dir(path, ["first_assessment_to_drop", # several assessments were used as opposed to one single assessment
                                                        "only_free_assessments__1",
                                                        #"debug_mode__False"
                                                        ],
                                                        "dataset_stats.csv")
    
    return dir_all, dir_free

def read_data_make_ds_learning():

    path = "../learning_diagnosis_predictor_data/reports/create_datasets/"

    dir_all = get_newest_non_empty_dir_in_dir(path, ["first_assessment_to_drop", # several assessments were used as opposed to one single assessment
                                                        "only_free_assessments__0",
                                                        #"debug_mode__False"
                                                        ], 
                                                        "dataset_stats.csv") # Need to check if this file was created (some dirs were generated by
                                                                            # running the script with only_assessment_distribution=1, which does not create this file)
    dir_free = get_newest_non_empty_dir_in_dir(path, ["first_assessment_to_drop", # several assessments were used as opposed to one single assessment
                                                        "only_free_assessments__1", 
                                                        #"debug_mode__False"
                                                        ],
                                                        "dataset_stats.csv")
    
    return dir_all, dir_free

def read_data_coefs_asd():
    path = "../diagnosis_predictor_data/reports/evaluate_models_on_feature_subsets/"

    dir_all = get_newest_non_empty_dir_in_dir(path, ["first_assessment_to_drop", # several assessments were used as opposed to one single assessment
                                                        "only_free_assessments__0",
                                                        #"debug_mode__False"
                                                        ]) # Need to check if this file was created (some dirs were generated by
                                                                            # running the script with only_assessment_distribution=1, which does not create this file)
    dir_free = get_newest_non_empty_dir_in_dir(path, ["first_assessment_to_drop", # several assessments were used as opposed to one single assessment
                                                        "only_free_assessments__1", 
                                                        #"debug_mode__False"
                                                        ])
    
    dir_all_coefs = dir_all + "feature-subsets/Diag.Autism Spectrum Disorder.txt"
    dir_free_coefs = dir_free + "feature-subsets/Diag.Autism Spectrum Disorder.txt"

    dir_nb_opt_all = dir_all + "optimal-nb-features.txt"
    dir_nb_opt_free = dir_free + "optimal-nb-features.txt"
    
    return dir_all_coefs, dir_free_coefs, dir_nb_opt_all, dir_nb_opt_free

def read_data_coefs_RD():
    path = "../learning_diagnosis_predictor_data/reports/evaluate_models_on_feature_subsets/"

    dir_all = get_newest_non_empty_dir_in_dir(path, ["first_assessment_to_drop", # several assessments were used as opposed to one single assessment
                                                        "only_free_assessments__0",
                                                        #"debug_mode__False"
                                                        ]) # Need to check if this file was created (some dirs were generated by
                                                                            # running the script with only_assessment_distribution=1, which does not create this file)
    dir_free = get_newest_non_empty_dir_in_dir(path, ["first_assessment_to_drop", # several assessments were used as opposed to one single assessment
                                                        "only_free_assessments__1", 
                                                        #"debug_mode__False"
                                                        ])
    
    dir_all_coefs = dir_all + "feature-subsets/New Diag.Specific Learning Disorder with Impairment in Reading.txt"
    dir_free_coefs = dir_free + "feature-subsets/New Diag.Specific Learning Disorder with Impairment in Reading.txt"

    dir_nb_opt_all = dir_all + "optimal-nb-features.txt"
    dir_nb_opt_free = dir_free + "optimal-nb-features.txt"
    
    
    return dir_all_coefs, dir_free_coefs, dir_nb_opt_all, dir_nb_opt_free

def read_data_coefs_CIS():
    path_coef = "../diagnosis_predictor_PCIAT_data/reports/identify_feature_subsets/"
    path_nb_opt = "../diagnosis_predictor_PCIAT_data/reports/evaluate_models_on_feature_subsets/"

    dir_all_coef = get_newest_non_empty_dir_in_dir(path_coef, ["first_assessment_to_drop", # several assessments were used as opposed to one single assessment
                                                        "only_free_assessments__0",
                                                        #"debug_mode__False"
                                                        ]) # Need to check if this file was created (some dirs were generated by
                                                                            # running the script with only_assessment_distribution=1, which does not create this file)
    dir_free_coef = get_newest_non_empty_dir_in_dir(path_coef, ["first_assessment_to_drop", # several assessments were used as opposed to one single assessment
                                                        "only_free_assessments__0", # TODO: change to 1 when done
                                                        #"debug_mode__False"
                                                        ])
    
    dir_all_nb_opt = get_newest_non_empty_dir_in_dir(path_nb_opt, ["first_assessment_to_drop", # several assessments were used as opposed to one single assessment
                                                        "only_free_assessments__0",
                                                        #"debug_mode__False"
                                                        ]) # Need to check if this file was created (some dirs were generated by
                                                                            # running the script with only_assessment_distribution=1, which does not create this file)
    dir_free_nb_opt = get_newest_non_empty_dir_in_dir(path_nb_opt, ["first_assessment_to_drop", # several assessments were used as opposed to one single assessment
                                                        "only_free_assessments__0", # TODO: change to 1 when done
                                                        #"debug_mode__False"
                                                        ])
    
    dir_all_coefs = dir_all_coef + "feature-subsets/CIS_P,CIS_P_Score.txt"
    dir_free_coefs = dir_free_coef + "feature-subsets/CIS_P,CIS_P_Score.txt"

    dir_nb_opt_all = dir_all_nb_opt + "optimal-nb-features.txt"
    dir_nb_opt_free = dir_free_nb_opt + "optimal-nb-features.txt"
    
    
    return dir_all_coefs, dir_free_coefs, dir_nb_opt_all, dir_nb_opt_free


def make_df_to_plot_eval_orig(dir_eval_orig_all, dir_eval_orig_free):
    # Read ROC AUC on all features on test set and for only healthy controls, for all assessments and free assessments
    eval_orig_all_df = pd.read_csv(dir_eval_orig_all + "performance_table_all_features.csv", index_col=0)
    eval_orig_free_df = pd.read_csv(dir_eval_orig_free + "performance_table_all_features.csv", index_col=0)

    # Each row is diagnosis (index), plot ROC AUC column and ROC AUC healthy controls column. 
    # Plot both for all assessments and free assessments on the same plot, add number of positive examples to each diagnosis
    eval_orig_all_df = eval_orig_all_df[["ROC AUC", "ROC AUC Mean CV", "ROC AUC Healthy Controls", "# of Positive Examples"]]
    eval_orig_free_df = eval_orig_free_df[["ROC AUC", "ROC AUC Mean CV", "ROC AUC Healthy Controls", "# of Positive Examples"]]
    eval_orig_all_df.columns = ["AUC all features all assessments", "AUC CV all features all assessments", "AUC all features healthy controls all assessments", "# of Positive Examples all assessments"]
    eval_orig_free_df.columns = ["AUC all features free assessments", "AUC CV all features free assessments", "AUC all features healthy controls free assessments", "# of Positive Examples free assessments"]
    eval_orig_df = eval_orig_all_df.merge(eval_orig_free_df, left_index=True, right_index=True).sort_values(by="AUC CV all features all assessments", ascending=False)

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

def make_df_to_plot_eval_subsets_learning(dir_eval_subsets_all, dir_eval_subsets_free):
    df_opt_features_all = pd.read_csv(dir_eval_subsets_all + "auc-sens-spec-on-subsets-test-set-optimal-threshold-optimal-nb-features.csv", index_col=1)
    df_opt_features_free = pd.read_csv(dir_eval_subsets_free + "auc-sens-spec-on-subsets-test-set-optimal-threshold-optimal-nb-features.csv", index_col=1)

    #df_manual = pd.read_csv("output/manual_scoring_analysis/manual_subsale_scores_vs_ml.csv", index_col=0)

    # Make one dataset with all info
    df_opt_features_all = df_opt_features_all[["AUC", "Number of features"]]
    df_opt_features_all.columns = ["AUC optimal features all assessments", "Optimal # of features all assessments"]
    df_opt_features_free = df_opt_features_free[["AUC", "Number of features"]]
    df_opt_features_free.columns = ["AUC optimal features free assessments", "Optimal # of features free assessments"]
    df_opt_features = df_opt_features_all.merge(df_opt_features_free, left_index=True, right_index=True)

    # Add manual scoring
    #df_opt_features = df_opt_features.merge(df_manual, left_index=True, right_index=True).sort_values(by="Best subscale score", ascending=False)

    return df_opt_features

def make_df_ds_stats(dir_all, dir_free):
    
    ds_stats_df_all_assessments = pd.read_csv(dir_all + "dataset_stats.csv", index_col=0)
    ds_stats_df_free_assessments = pd.read_csv(dir_free + "dataset_stats.csv", index_col=0)

    ds_stats_df_all_assessments = ds_stats_df_all_assessments.T[["n_rows_full_ds", "n_input_cols"]]
    ds_stats_df_all_assessments.columns = ["# rows full dataset all assessments", "# input features all assessments"]

    ds_stats_df_free_assessments = ds_stats_df_free_assessments.T[["n_rows_full_ds", "n_input_cols"]]
    ds_stats_df_free_assessments.columns = ["# rows full dataset free assessments", "# input features free assessments"]

    #Merge
    ds_stats_df = ds_stats_df_all_assessments.merge(ds_stats_df_free_assessments, left_index=True, right_index=True)

    return ds_stats_df

def merge_tables(eval_orig_df, eval_subsets_df, ds_stats_df):
    compare_orig_vs_subsets_df = eval_orig_df.merge(eval_subsets_df, left_index=True, right_index=True)
    # Add total # of features and examples to compare_orig_vs_subsets_df
    for col in ds_stats_df.columns:
        compare_orig_vs_subsets_df[col] = ds_stats_df[col].values[0]

    return compare_orig_vs_subsets_df

def make_df_coefs(dir_coefs_all, dir_coefs_free, dir_nb_opt_all, dir_nb_opt_free, diag):
    print(diag)
    import json

    coefs_dict_all = json.load(open(dir_coefs_all))
    coefs_dict_free = json.load(open(dir_coefs_free))

    # Get optimal nb of features: read dict in dir_nb_opt_asd_all and dir_nb_opt_asd_free and get optimal nb of features for diag
    with open(dir_nb_opt_all, "r") as f:
        nb_opt_dict_all = eval(f.read())
    with open(dir_nb_opt_free, "r") as f:
        nb_opt_dict_free = eval(f.read())

    nb_opt = nb_opt_dict_all[diag]
    nb_opt_free = nb_opt_dict_free[diag]

    print(coefs_dict_all.keys())
    

    # Read subsets at nb_opt
    if diag == "CIS_P,CIS_P_Score": # :TODO when PCIAT code has feature-subsets in eval_subsets, remove condition

        #import joblib
        #coefs_dict_all = joblib.load("../diagnosis_predictor_PCIAT_data/reports/evaluate_models_on_feature_subsets/2023-05-30 12.18.19___first_assessment_to_drop__SCARED_SR___use_other_outputs_as_input__0___only_free_assessments__0___debug_mode__True/feature-subsets.joblib")
        #print("22 KEYS:", coefs_dict_all["22"].keys())

        coefs_all_text = coefs_dict_all[str(nb_opt)] 
        coefs_free_text = coefs_dict_free[str(nb_opt_free)]
    else:
        coefs_all_text = coefs_dict_all[str(nb_opt)][1] #coefs_dict_all[0] has AUROC value, coefs_dict_all[1] has coefs
        coefs_free_text = coefs_dict_free[str(nb_opt_free)][1]

    print(coefs_all_text, coefs_free_text)

    # Parse item and coefs, format: "(0.64*) C3SR,C3SR_33: 33. I have trouble with reading. - 0=Not true at all (Never, Seldom), 1=Just a little true (Occasionally), 2=Pretty much true (Often, Quite a bit), 3=Very much true (Very often, Very frequently)"
    coefs_all = {}
    for line in coefs_all_text:
        line_list = line.split("*)")
        print(line, line_list)
        coef = float(line_list[0][1:])
        item = line_list[1][:100] + "..."
        coefs_all[item] = coef

    coefs_free = {}
    for line in coefs_free_text:
        line_list = line.split("*)")
        coef = float(line_list[0][1:])
        item = line_list[1][:100] + "..."
        coefs_free[item] = coef

    # Make df
    coefs_all_df = pd.DataFrame.from_dict(coefs_all, orient="index", columns=["Coefficient"])
    coefs_all_df.index.name = "Item"

    coefs_free_df = pd.DataFrame.from_dict(coefs_free, orient="index", columns=["Coefficient"])
    coefs_free_df.index.name = "Item"

    # Sort by coef
    coefs_all_df = coefs_all_df.sort_values(by="Coefficient", ascending=False)
    coefs_free_df = coefs_free_df.sort_values(by="Coefficient", ascending=False)

    return coefs_all_df, coefs_free_df

def main():
    dir_eval_orig_all, dir_eval_orig_free = read_data_eval_orig()
    dir_eval_subsets_all, dir_eval_subsets_free = read_data_eval_subsets()
    dir_make_ds_all, dir_make_ds_free = read_data_make_ds()
    dir_coefs_asd_all, dir_coefs_asd_free, dir_nb_opt_asd_all, dir_nb_opt_asd_free = read_data_coefs_asd()

    dir_eval_orig_all_learning, dir_eval_orig_free_learning = read_data_eval_orig_learning()
    dir_eval_subsets_all_learning, dir_eval_subsets_free_learning = read_data_eval_subsets_learning()
    dir_make_ds_all_learning, dir_make_ds_free_learning = read_data_make_ds_learning()
    dir_coefs_RD_all, dir_coefs_RD_free, dir_nb_opt_RD_all, dir_nb_opt_RD_free = read_data_coefs_RD()

    dir_coefs_CIS_all, dir_coefs_CIS_free, dir_nb_opt_CIS_all, dir_nb_opt_CIS_free = read_data_coefs_CIS()

    print("Reading reports from: ", dir_eval_orig_all, dir_eval_orig_free)
    print("\nReading reports from: ", dir_eval_subsets_all, dir_eval_subsets_free)
    print("\nReading reports from: ", dir_make_ds_all, dir_make_ds_free)
    print("\nReading reports from: ", dir_coefs_asd_all, dir_coefs_asd_free, dir_nb_opt_asd_all, dir_nb_opt_asd_free)

    print("\n\nReading reports from: ", dir_eval_orig_all_learning, dir_eval_orig_free_learning)
    print("\nReading reports from: ", dir_eval_subsets_all_learning, dir_eval_subsets_free_learning)
    print("\nReading reports from: ", dir_make_ds_all_learning, dir_make_ds_free_learning)
    print("\nReading reports from: ", dir_coefs_RD_all, dir_coefs_RD_free, dir_nb_opt_RD_all, dir_nb_opt_RD_free)

    eval_orig_df = make_df_to_plot_eval_orig(dir_eval_orig_all, dir_eval_orig_free)
    eval_subsets_df = make_df_to_plot_eval_subsets(dir_eval_subsets_all, dir_eval_subsets_free)
    ds_stats_df = make_df_ds_stats(dir_make_ds_all, dir_make_ds_free)
    coefs_all_df, coefs_free_df = make_df_coefs(dir_coefs_asd_all, dir_coefs_asd_free, dir_nb_opt_asd_all, dir_nb_opt_asd_free,
                             "Diag.Autism Spectrum Disorder")

    eval_orig_learning_df = make_df_to_plot_eval_orig(dir_eval_orig_all_learning, dir_eval_orig_free_learning)
    eval_subsets_learning_df = make_df_to_plot_eval_subsets_learning(dir_eval_subsets_all_learning, dir_eval_subsets_free_learning)
    ds_stats_learning_df = make_df_ds_stats(dir_make_ds_all_learning, dir_make_ds_free_learning)
    coefs_learning_all_df, coefs_learning_free_df = make_df_coefs(dir_coefs_RD_all, dir_coefs_RD_free, dir_nb_opt_RD_all, dir_nb_opt_RD_free, 
                                      "New Diag.Specific Learning Disorder with Impairment in Reading")
    
    coefs_CIS_all_df, coefs_CIS_free_df = make_df_coefs(dir_coefs_CIS_all, dir_coefs_CIS_free, dir_nb_opt_CIS_all, dir_nb_opt_CIS_free, 
                                      "CIS_P,CIS_P_Score")
    
    compare_orig_vs_subsets_df = merge_tables(eval_orig_df, eval_subsets_df, ds_stats_df)

    compare_orig_vs_subsets_learning_df = merge_tables(eval_orig_learning_df, eval_subsets_learning_df, ds_stats_learning_df)

    eval_orig_df.to_csv("output/eval_orig.csv")
    eval_subsets_df.to_csv("output/eval_subsets.csv")
    compare_orig_vs_subsets_df.to_csv("output/compare_orig_vs_subsets.csv")
    coefs_all_df.to_csv("output/coefs_all.csv")
    coefs_free_df.to_csv("output/coefs_free.csv")

    eval_orig_learning_df.to_csv("output/eval_orig_learning.csv")
    eval_subsets_learning_df.to_csv("output/eval_subsets_learning.csv")
    compare_orig_vs_subsets_learning_df.to_csv("output/compare_orig_vs_subsets_learning.csv")
    coefs_learning_all_df.to_csv("output/coefs_learning_all.csv")
    coefs_learning_free_df.to_csv("output/coefs_learning_free.csv")

    coefs_CIS_all_df.to_csv("output/coefs_CIS_all.csv")
    coefs_CIS_free_df.to_csv("output/coefs_CIS_free.csv")


if __name__ == "__main__":
    main()