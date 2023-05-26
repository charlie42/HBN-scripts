from helpers import get_newest_non_empty_dir_in_dir, make_dir_if_not_exists

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from make_final_tables import read_data_eval_subsets

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
        'New Diag.Specific Learning Disorder with Impairment in Mathematics': 'SLD-Math',
        'Diag.Language Disorder': 'Language',
        'Diag.Specific Phobia': 'Phobia',
        'Diag.Specific Learning Disorder with Impairment in Reading': 'SLD-Reading',
        'Diag.Specific Learning Disorder with Impairment in Written Expression': 'SLD-Writing',
        'New Diag.Specific Learning Disorder with Impairment in Reading': 'SLD-Reading',
        'New Diag.Specific Learning Disorder with Impairment in Written Expression': 'SLD-Writing',
        'Diag.Other Specified Anxiety Disorder': 'Other Anxiety',
        'New Diag.Processing Speed Deficit': 'LD-PS',
        'New Diag.Borderline Intellectual Functioning': 'LD-BIF',
    }

def make_df_to_plot_thresholds(dir_eval_subsets_all, filename):

    df_opt_features_all = pd.read_csv(dir_eval_subsets_all + "sens-spec-on-subsets-test-set-optimal-nb-features/" + filename, 
                                      index_col=0)

    return df_opt_features_all

def plot_eval_orig(eval_orig_df):
    # Plot on same plot, one point for each diagnosis, connext diagnoses with lines
    plt.figure(figsize=(10, 5))
    plt.title("ROC AUC for all features")
    plt.plot(eval_orig_df["AUC all features all assessments"], label="test set all assessments", marker="o", linestyle="", color="lightblue")
    plt.plot(eval_orig_df["AUC CV all features all assessments"], label="CV all assessments", marker="o", linestyle="", color="blue")
    plt.plot(eval_orig_df["AUC all features free assessments"], label="test set free assessments", marker="o", linestyle="", color="lightblue", markerfacecolor='none')
    plt.plot(eval_orig_df["AUC CV all features free assessments"], label="CV free assessments", marker="o", linestyle="", color="blue", markerfacecolor='none')
    plt.plot(eval_orig_df["AUC all features healthy controls all assessments"], label="test set healthy controls all assessments", marker="o", linestyle="", color="lightcoral")
    plt.xticks(rotation=45, ha="right", size=8)
    plt.legend(loc="upper right")
    plt.ylim([0.5, 1.0])

    # For each diag, also plot bar plots with the number of positive examples for all assessments and for free assessments
    ax2 = plt.twinx()
    ax2.set_ylabel("Number of positive examples")
    ax2.bar(eval_orig_df.index, eval_orig_df["# of Positive Examples free assessments"].values, color="blue", width=0.5, alpha=0.2, fill=False)
    ax2.bar(eval_orig_df.index, eval_orig_df["# of Positive Examples all assessments"].values, color="blue", width=0.5, alpha=0.2)
    ax2.legend(["# of Positive Examples free assessments", "# of Positive Examples all assessments"], loc="lower right")

    # ax2 = plt.twinx()
    # ax2.bar(eval_orig_df.index, eval_orig_df["# of Positive Examples all assessments"], alpha=0.3, color="black")
    # ax2.bar(eval_orig_df.index, eval_orig_df["# of Positive Examples free assessments"], alpha=0.1, color="black")
    # ax2.set_ylabel("# of Positive Examples, free and all assessments", color="black")
    # ax2.tick_params(axis='y', labelcolor="black")



   

    
    output_dir = "output/viz/"
    make_dir_if_not_exists(output_dir)
    plt.savefig(output_dir + "ROC_AUC_all_features.png", bbox_inches="tight", dpi=600)

def plot_manual_vs_ml(eval_subsets_df):
    print("DEBUG", eval_subsets_df.columns)
    plt.figure(figsize=(10, 8))
    plt.title("ROC AUC on test set for subsets of features")
    #plt.plot(eval_subsets_df["AUC all assessments"], label="ML on optimal # of features (all assessments)", marker="o", linestyle="", color="blue")
    #plt.plot(eval_subsets_df["AUC free assessments"], label="ML on optimal # of features (free assessments)", marker="o", markerfacecolor='none', linestyle="", color="blue")
    plt.plot(eval_subsets_df["Best subscale score"], label="Best subscale (and # of features to reach it with ML)", marker="o", linestyle="", color="red")
    plt.plot(eval_subsets_df["ML score at # of items of best subscale (all assessments)"], label="ML on subscale # of features (all assessments)", marker="o", linestyle="", color="blue")
    #plt.plot(eval_subsets_df["ML score at # of items of best subscale (free assessments)"], label="ML on subscale # of features (free assessments)", marker="o", markerfacecolor='none', linestyle="", color="green")
    plt.xticks(rotation=45, ha="right", size=8)
    plt.legend(loc="upper right")

    # Print number of items next to AUC scores to the right of the markers
    for i, row in eval_subsets_df.iterrows():
        #plt.text(i, row["AUC all assessments"]+0.01, str(row["Optimal # of features all assessments"]), ha="left", va="center", size=8)
        plt.text(i, row["ML score at # of items of best subscale (all assessments)"]+0.01, str(row["# of items in best subscale"]), ha="center", va="bottom", size=8)
        plt.text(i, row["Best subscale score"]-0.01, str(row["# of items to reach best subscale (all assessments)"]), ha="center", va="top", size=8)
    
    # Append best subscale name to the diag name on x axis
    eval_subsets_df["Best subscale"] = eval_subsets_df["Best subscale"].str.split(",").str[1] # Remove prefix before , from subscale names
    plt.xticks(range(len(eval_subsets_df.index)), eval_subsets_df.index + " (" + eval_subsets_df["Best subscale"] + ")", rotation=45, ha="right", size=8)

    plt.tight_layout()
    
    plt.savefig("output/viz/ROC_AUC_subsets.png", bbox_inches="tight", dpi=600)

def plot_opt_num_features(opt_vs_all_df):
    opt_vs_all_df = opt_vs_all_df.sort_values(by="AUC optimal features all assessments", ascending=False)
    
    # Plot AUC on all features all assessments, AUC on optimal # of features all assessments, AUC on optimal # of features free assessments, AUC on all features free assessments
    plt.figure(figsize=(10, 8))
    plt.title("ROC AUC on test set for subsets of features vs all features")
    plt.plot(opt_vs_all_df["AUC all features all assessments"], label="AUC on all features (all assessments)", marker="o", linestyle="", color="blue")
    plt.plot(opt_vs_all_df["AUC all features free assessments"], label="AUC on all features (free assessments)", marker="o", linestyle="", color="blue", markerfacecolor='none')
    plt.plot(opt_vs_all_df["AUC optimal features all assessments"], label="AUC on optimal # of features (all assessments)", marker="o", linestyle="", color="red")
    plt.plot(opt_vs_all_df["AUC optimal features free assessments"], label="AUC on optimal # of features (free assessments)", marker="o", linestyle="", color="red", markerfacecolor='none')
    plt.xticks(rotation=45, ha="right", size=8)
    plt.legend(loc="upper right")

    # Print number of items next to AUC scores to the right of the markers
    opt_vs_all_df = opt_vs_all_df.reset_index()
    for i, row in opt_vs_all_df.iterrows():
        plt.text(i+0.1, row["AUC optimal features all assessments"]-0.001, str(row["Optimal # of features all assessments"]), ha="left", va="center", fontsize=6)
        plt.text(i+0.1, row["AUC optimal features free assessments"]-0.0005, str(row["Optimal # of features free assessments"]), ha="left", va="center", fontsize=6)

    plt.tight_layout()

    plt.savefig("output/viz/ROC_AUC_optimal_vs_all_features.png", bbox_inches="tight", dpi=600)

def plot_thresholds(thresholds_df):
    # Drop first row (PPV always 0.5)
    thresholds_df = thresholds_df.drop(thresholds_df.index[0])

    # Plot sensitivity, specificity, PPV, and NPV for each threshold
    plt.figure(figsize=(10, 8))
    plt.title("Sensitivity, specificity, PPV, and NPV for each threshold for ASD diagnosis, optimal # of features, all assessments")
    plt.plot(thresholds_df.index, thresholds_df["Sensitivity"], label="Sensitivity", marker="o", linestyle="-", color="blue")
    plt.plot(thresholds_df.index, thresholds_df["Specificity"], label="Specificity", marker="o", linestyle="-", color="red")
    plt.plot(thresholds_df.index, thresholds_df["PPV"], label="PPV", marker="o", linestyle="-", color="green")
    plt.plot(thresholds_df.index, thresholds_df["NPV"], label="NPV", marker="o", linestyle="-", color="orange")
    plt.xlim(thresholds_df.index[0], thresholds_df.index[-1])
    plt.xticks(thresholds_df.index, rotation=45, ha="right", size=8)
    plt.legend(loc="upper right")

    plt.tight_layout()

    plt.savefig("output/viz/sensitivity_specificity_PPV_NPV.png", bbox_inches="tight", dpi=600)

def plot_what_improves_LD(check_what_improves_LD_df):

    # Make all columns except those that start with ROC AUC into ints (if not NA) (bug in pd)
    for col in check_what_improves_LD_df.columns:
        if not col.startswith("ROC AUC"):
            check_what_improves_LD_df[col] = check_what_improves_LD_df[col].astype('Int64')

    sns.set_style("whitegrid")
    sns.set_context("paper")

    check_what_improves_LD_df = check_what_improves_LD_df.sort_values("ROC AUC Mean CV_nothing", ascending=False)

    # Plot ROC AUC Mean CV, x=diag (index), y=ROC AUC Mean CV, color=_nothing or _nih or _conners or _newdiag
    plt.figure(figsize=(10, 8))
    plt.title("ROC AUC Mean CV for each LD diagnosis, all features, all assessments")
    plt.plot(check_what_improves_LD_df.index, check_what_improves_LD_df["ROC AUC Mean CV_nothing"], label="ROC AUC Mean CV_nothing", marker="D", color="blue", markersize=10, linestyle="")
    plt.plot(check_what_improves_LD_df.index, check_what_improves_LD_df["ROC AUC Mean CV_nih"], label="ROC AUC Mean CV_nih", marker="o", color="red", markersize=10, linestyle="")
    plt.plot(check_what_improves_LD_df.index, check_what_improves_LD_df["ROC AUC Mean CV_conners"], label="ROC AUC Mean CV_conners", marker="o", color="green", markersize=10, linestyle="")
    plt.plot(check_what_improves_LD_df.index, check_what_improves_LD_df["ROC AUC Mean CV_newdiag"], label="ROC AUC Mean CV_newdiag", marker="o", color="orange", markersize=10, linestyle="")
    plt.xlim(check_what_improves_LD_df.index[0], check_what_improves_LD_df.index[-1])
    plt.xticks(check_what_improves_LD_df.index, rotation=45, ha="right", size=8)
    plt.legend(loc="upper right")

    # Add text labels for each point with # of positive examples and Total examples
    check_what_improves_LD_df = check_what_improves_LD_df.reset_index()
    for i, row in check_what_improves_LD_df.iterrows():
        plt.text(i+0.1, row["ROC AUC Mean CV_nothing"]-0.0005, str(row["# of positive examples_nothing"]) + "/" + str(row["Total examples_nothing"]), ha="left", va="center", fontsize=8)
        plt.text(i+0.1, row["ROC AUC Mean CV_nih"]-0.0005, str(row["# of positive examples_nih"]) + "/" + str(row["Total examples_nih"]), ha="left", va="center", fontsize=8)
        plt.text(i+0.1, row["ROC AUC Mean CV_conners"]-0.0005, str(row["# of positive examples_conners"]) + "/" + str(row["Total examples_conners"]), ha="left", va="center", fontsize=8)
        plt.text(i+0.1, row["ROC AUC Mean CV_newdiag"]-0.0005, str(row["# of positive examples_newdiag"]) + "/" + str(row["Total examples_newdiag"]), ha="left", va="center", fontsize=8)

    plt.savefig("output/viz/what_improves_LD.png", dpi=600)

def main():

    # Read performance tables
    eval_orig_df = pd.read_csv("output/eval_orig.csv", index_col=0)
    eval_subsets_df = pd.read_csv("output/eval_subsets.csv", index_col=0)
    compare_orig_subsets_df = pd.read_csv("output/compare_orig_vs_subsets.csv", index_col=0)
    what_improves_LD_df = pd.read_csv("output/what_improves_LD.csv", index_col=0)

    # Read thresholds data from diagnosis_predictor_data (for all assessments, that's why using dir_eval_subsets_all[0])
    dir_eval_subsets_all = read_data_eval_subsets()
    filename = "Diag.Autism Spectrum Disorder.csv"
    thresholds_df = make_df_to_plot_thresholds(dir_eval_subsets_all[0], filename)
    
    # Rephrase diags to shorter names
    eval_orig_df = eval_orig_df.rename(index=diagnosis_dict)
    eval_subsets_df = eval_subsets_df.rename(index=diagnosis_dict)
    what_improves_LD_df = what_improves_LD_df.rename(index=diagnosis_dict)

    plot_eval_orig(eval_orig_df)
    plot_manual_vs_ml(eval_subsets_df)
    plot_opt_num_features(compare_orig_subsets_df)
    plot_thresholds(thresholds_df)
    plot_what_improves_LD(what_improves_LD_df)

if __name__ == "__main__":
    main()



