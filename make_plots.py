from helpers import get_newest_non_empty_dir_in_dir, make_dir_if_not_exists

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

diagnosis_dict = {
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
        'Diag.Language Disorder': 'Language',
        'Diag.Specific Phobia': 'Phobia',
        'Diag.Specific Learning Disorder with Impairment in Reading': 'SLD-Reading',
        'Diag.Specific Learning Disorder with Impairment in Written Expression': 'SLD-Writing',
        'Diag.Other Specified Anxiety Disorder': 'Other Anxiety'
    }

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
    eval_orig_all_df.columns = ["ROC AUC all assessments", "ROC AUC healthy controls all assessments", "# of Positive Examples"]
    eval_orig_free_df.columns = ["ROC AUC free assessments", "ROC AUC healthy controls free assessments"]
    eval_orig_df = eval_orig_all_df.merge(eval_orig_free_df, left_index=True, right_index=True).sort_values(by="ROC AUC all assessments", ascending=False)

    # Rephrase diags to shorter names
    eval_orig_df = eval_orig_df.rename(index=diagnosis_dict)

    return eval_orig_df

def make_df_to_plot_eval_subsets(dir_eval_subsets_all, dir_eval_subsets_free):
    pass

def plot_eval_orig(eval_orig_df):
    # Plot on same plot, one point for each diagnosis, connext diagnoses with lines
    plt.figure(figsize=(10, 5))
    plt.title("ROC AUC for all features")
    plt.plot(eval_orig_df["ROC AUC all assessments"], label="all assessments", marker="o", linestyle="", color="blue")
    plt.plot(eval_orig_df["ROC AUC free assessments"], label="free assessments", marker="*", linestyle="", color="blue")
    plt.plot(eval_orig_df["ROC AUC healthy controls all assessments"], label="healthy controls all assessments", marker="o", linestyle="", color="red")
    #plt.plot(eval_orig_df["ROC AUC healthy controls free assessments"], label="healthy controls free assessments", marker="o", linestyle="--", color="green")
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc="lower right")
    plt.ylim([0.5, 1.0])

    # Add bars with the number of positive examples to each diagnosis
    ax2 = plt.twinx()
    ax2.bar(eval_orig_df.index, eval_orig_df["# of Positive Examples"], alpha=0.2, color="black")
    ax2.set_ylabel("# of Positive Examples", color="black")
    ax2.tick_params(axis='y', labelcolor="black")

    output_dir = "output/viz/"
    make_dir_if_not_exists(output_dir)
    plt.savefig(output_dir + "ROC_AUC_all_features.png", bbox_inches="tight", dpi=600)

def plot_eval_subsets(eval_subsets_df):
    pass

def main():
    dir_eval_orig_all, dir_eval_orig_free = read_data_eval_orig()
    print("Reading reports from: ", dir_eval_orig_all, dir_eval_orig_free)

    eval_orig_df = make_df_to_plot_eval_orig(dir_eval_orig_all, dir_eval_orig_free)
    plot_eval_orig(eval_orig_df)

    dir_eval_subsets_all, dir_eval_subsets_free = read_data_eval_subsets()
    print("Reading reports from: ", dir_eval_subsets_all, dir_eval_subsets_free)

    eval_subsets_df = make_df_to_plot_eval_subsets(dir_eval_subsets_all, dir_eval_subsets_free)
    plot_eval_subsets(eval_subsets_df)

if __name__ == "__main__":
    main()



